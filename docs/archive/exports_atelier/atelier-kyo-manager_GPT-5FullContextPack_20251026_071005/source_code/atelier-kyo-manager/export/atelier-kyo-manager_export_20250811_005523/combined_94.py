
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\six.py ===
# Copyright (c) 2010-2024 Benjamin Peterson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Utilities for writing code that runs on Python 2 and 3"""

from __future__ import absolute_import

import functools
import itertools
import operator
import sys
import types

__author__ = "Benjamin Peterson <benjamin@python.org>"
__version__ = "1.17.0"


# Useful for very coarse version differentiation.
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
PY34 = sys.version_info[0:2] >= (3, 4)

if PY3:
    string_types = str,
    integer_types = int,
    class_types = type,
    text_type = str
    binary_type = bytes

    MAXSIZE = sys.maxsize
else:
    string_types = basestring,
    integer_types = (int, long)
    class_types = (type, types.ClassType)
    text_type = unicode
    binary_type = str

    if sys.platform.startswith("java"):
        # Jython always uses 32 bits.
        MAXSIZE = int((1 << 31) - 1)
    else:
        # It's possible to have sizeof(long) != sizeof(Py_ssize_t).
        class X(object):

            def __len__(self):
                return 1 << 31
        try:
            len(X())
        except OverflowError:
            # 32-bit
            MAXSIZE = int((1 << 31) - 1)
        else:
            # 64-bit
            MAXSIZE = int((1 << 63) - 1)
        del X

if PY34:
    from importlib.util import spec_from_loader
else:
    spec_from_loader = None


def _add_doc(func, doc):
    """Add documentation to a function."""
    func.__doc__ = doc


def _import_module(name):
    """Import module, returning the module after the last dot."""
    __import__(name)
    return sys.modules[name]


class _LazyDescr(object):

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, tp):
        result = self._resolve()
        setattr(obj, self.name, result)  # Invokes __set__.
        try:
            # This is a bit ugly, but it avoids running this again by
            # removing this descriptor.
            delattr(obj.__class__, self.name)
        except AttributeError:
            pass
        return result


class MovedModule(_LazyDescr):

    def __init__(self, name, old, new=None):
        super(MovedModule, self).__init__(name)
        if PY3:
            if new is None:
                new = name
            self.mod = new
        else:
            self.mod = old

    def _resolve(self):
        return _import_module(self.mod)

    def __getattr__(self, attr):
        _module = self._resolve()
        value = getattr(_module, attr)
        setattr(self, attr, value)
        return value


class _LazyModule(types.ModuleType):

    def __init__(self, name):
        super(_LazyModule, self).__init__(name)
        self.__doc__ = self.__class__.__doc__

    def __dir__(self):
        attrs = ["__doc__", "__name__"]
        attrs += [attr.name for attr in self._moved_attributes]
        return attrs

    # Subclasses should override this
    _moved_attributes = []


class MovedAttribute(_LazyDescr):

    def __init__(self, name, old_mod, new_mod, old_attr=None, new_attr=None):
        super(MovedAttribute, self).__init__(name)
        if PY3:
            if new_mod is None:
                new_mod = name
            self.mod = new_mod
            if new_attr is None:
                if old_attr is None:
                    new_attr = name
                else:
                    new_attr = old_attr
            self.attr = new_attr
        else:
            self.mod = old_mod
            if old_attr is None:
                old_attr = name
            self.attr = old_attr

    def _resolve(self):
        module = _import_module(self.mod)
        return getattr(module, self.attr)


class _SixMetaPathImporter(object):

    """
    A meta path importer to import six.moves and its submodules.

    This class implements a PEP302 finder and loader. It should be compatible
    with Python 2.5 and all existing versions of Python3
    """

    def __init__(self, six_module_name):
        self.name = six_module_name
        self.known_modules = {}

    def _add_module(self, mod, *fullnames):
        for fullname in fullnames:
            self.known_modules[self.name + "." + fullname] = mod

    def _get_module(self, fullname):
        return self.known_modules[self.name + "." + fullname]

    def find_module(self, fullname, path=None):
        if fullname in self.known_modules:
            return self
        return None

    def find_spec(self, fullname, path, target=None):
        if fullname in self.known_modules:
            return spec_from_loader(fullname, self)
        return None

    def __get_module(self, fullname):
        try:
            return self.known_modules[fullname]
        except KeyError:
            raise ImportError("This loader does not know module " + fullname)

    def load_module(self, fullname):
        try:
            # in case of a reload
            return sys.modules[fullname]
        except KeyError:
            pass
        mod = self.__get_module(fullname)
        if isinstance(mod, MovedModule):
            mod = mod._resolve()
        else:
            mod.__loader__ = self
        sys.modules[fullname] = mod
        return mod

    def is_package(self, fullname):
        """
        Return true, if the named module is a package.

        We need this method to get correct spec objects with
        Python 3.4 (see PEP451)
        """
        return hasattr(self.__get_module(fullname), "__path__")

    def get_code(self, fullname):
        """Return None

        Required, if is_package is implemented"""
        self.__get_module(fullname)  # eventually raises ImportError
        return None
    get_source = get_code  # same as get_code

    def create_module(self, spec):
        return self.load_module(spec.name)

    def exec_module(self, module):
        pass

_importer = _SixMetaPathImporter(__name__)


class _MovedItems(_LazyModule):

    """Lazy loading of moved objects"""
    __path__ = []  # mark as package


_moved_attributes = [
    MovedAttribute("cStringIO", "cStringIO", "io", "StringIO"),
    MovedAttribute("filter", "itertools", "builtins", "ifilter", "filter"),
    MovedAttribute("filterfalse", "itertools", "itertools", "ifilterfalse", "filterfalse"),
    MovedAttribute("input", "__builtin__", "builtins", "raw_input", "input"),
    MovedAttribute("intern", "__builtin__", "sys"),
    MovedAttribute("map", "itertools", "builtins", "imap", "map"),
    MovedAttribute("getcwd", "os", "os", "getcwdu", "getcwd"),
    MovedAttribute("getcwdb", "os", "os", "getcwd", "getcwdb"),
    MovedAttribute("getoutput", "commands", "subprocess"),
    MovedAttribute("range", "__builtin__", "builtins", "xrange", "range"),
    MovedAttribute("reload_module", "__builtin__", "importlib" if PY34 else "imp", "reload"),
    MovedAttribute("reduce", "__builtin__", "functools"),
    MovedAttribute("shlex_quote", "pipes", "shlex", "quote"),
    MovedAttribute("StringIO", "StringIO", "io"),
    MovedAttribute("UserDict", "UserDict", "collections", "IterableUserDict", "UserDict"),
    MovedAttribute("UserList", "UserList", "collections"),
    MovedAttribute("UserString", "UserString", "collections"),
    MovedAttribute("xrange", "__builtin__", "builtins", "xrange", "range"),
    MovedAttribute("zip", "itertools", "builtins", "izip", "zip"),
    MovedAttribute("zip_longest", "itertools", "itertools", "izip_longest", "zip_longest"),
    MovedModule("builtins", "__builtin__"),
    MovedModule("configparser", "ConfigParser"),
    MovedModule("collections_abc", "collections", "collections.abc" if sys.version_info >= (3, 3) else "collections"),
    MovedModule("copyreg", "copy_reg"),
    MovedModule("dbm_gnu", "gdbm", "dbm.gnu"),
    MovedModule("dbm_ndbm", "dbm", "dbm.ndbm"),
    MovedModule("_dummy_thread", "dummy_thread", "_dummy_thread" if sys.version_info < (3, 9) else "_thread"),
    MovedModule("http_cookiejar", "cookielib", "http.cookiejar"),
    MovedModule("http_cookies", "Cookie", "http.cookies"),
    MovedModule("html_entities", "htmlentitydefs", "html.entities"),
    MovedModule("html_parser", "HTMLParser", "html.parser"),
    MovedModule("http_client", "httplib", "http.client"),
    MovedModule("email_mime_base", "email.MIMEBase", "email.mime.base"),
    MovedModule("email_mime_image", "email.MIMEImage", "email.mime.image"),
    MovedModule("email_mime_multipart", "email.MIMEMultipart", "email.mime.multipart"),
    MovedModule("email_mime_nonmultipart", "email.MIMENonMultipart", "email.mime.nonmultipart"),
    MovedModule("email_mime_text", "email.MIMEText", "email.mime.text"),
    MovedModule("BaseHTTPServer", "BaseHTTPServer", "http.server"),
    MovedModule("CGIHTTPServer", "CGIHTTPServer", "http.server"),
    MovedModule("SimpleHTTPServer", "SimpleHTTPServer", "http.server"),
    MovedModule("cPickle", "cPickle", "pickle"),
    MovedModule("queue", "Queue"),
    MovedModule("reprlib", "repr"),
    MovedModule("socketserver", "SocketServer"),
    MovedModule("_thread", "thread", "_thread"),
    MovedModule("tkinter", "Tkinter"),
    MovedModule("tkinter_dialog", "Dialog", "tkinter.dialog"),
    MovedModule("tkinter_filedialog", "FileDialog", "tkinter.filedialog"),
    MovedModule("tkinter_scrolledtext", "ScrolledText", "tkinter.scrolledtext"),
    MovedModule("tkinter_simpledialog", "SimpleDialog", "tkinter.simpledialog"),
    MovedModule("tkinter_tix", "Tix", "tkinter.tix"),
    MovedModule("tkinter_ttk", "ttk", "tkinter.ttk"),
    MovedModule("tkinter_constants", "Tkconstants", "tkinter.constants"),
    MovedModule("tkinter_dnd", "Tkdnd", "tkinter.dnd"),
    MovedModule("tkinter_colorchooser", "tkColorChooser",
                "tkinter.colorchooser"),
    MovedModule("tkinter_commondialog", "tkCommonDialog",
                "tkinter.commondialog"),
    MovedModule("tkinter_tkfiledialog", "tkFileDialog", "tkinter.filedialog"),
    MovedModule("tkinter_font", "tkFont", "tkinter.font"),
    MovedModule("tkinter_messagebox", "tkMessageBox", "tkinter.messagebox"),
    MovedModule("tkinter_tksimpledialog", "tkSimpleDialog",
                "tkinter.simpledialog"),
    MovedModule("urllib_parse", __name__ + ".moves.urllib_parse", "urllib.parse"),
    MovedModule("urllib_error", __name__ + ".moves.urllib_error", "urllib.error"),
    MovedModule("urllib", __name__ + ".moves.urllib", __name__ + ".moves.urllib"),
    MovedModule("urllib_robotparser", "robotparser", "urllib.robotparser"),
    MovedModule("xmlrpc_client", "xmlrpclib", "xmlrpc.client"),
    MovedModule("xmlrpc_server", "SimpleXMLRPCServer", "xmlrpc.server"),
]
# Add windows specific modules.
if sys.platform == "win32":
    _moved_attributes += [
        MovedModule("winreg", "_winreg"),
    ]

for attr in _moved_attributes:
    setattr(_MovedItems, attr.name, attr)
    if isinstance(attr, MovedModule):
        _importer._add_module(attr, "moves." + attr.name)
del attr

_MovedItems._moved_attributes = _moved_attributes

moves = _MovedItems(__name__ + ".moves")
_importer._add_module(moves, "moves")


class Module_six_moves_urllib_parse(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_parse"""


_urllib_parse_moved_attributes = [
    MovedAttribute("ParseResult", "urlparse", "urllib.parse"),
    MovedAttribute("SplitResult", "urlparse", "urllib.parse"),
    MovedAttribute("parse_qs", "urlparse", "urllib.parse"),
    MovedAttribute("parse_qsl", "urlparse", "urllib.parse"),
    MovedAttribute("urldefrag", "urlparse", "urllib.parse"),
    MovedAttribute("urljoin", "urlparse", "urllib.parse"),
    MovedAttribute("urlparse", "urlparse", "urllib.parse"),
    MovedAttribute("urlsplit", "urlparse", "urllib.parse"),
    MovedAttribute("urlunparse", "urlparse", "urllib.parse"),
    MovedAttribute("urlunsplit", "urlparse", "urllib.parse"),
    MovedAttribute("quote", "urllib", "urllib.parse"),
    MovedAttribute("quote_plus", "urllib", "urllib.parse"),
    MovedAttribute("unquote", "urllib", "urllib.parse"),
    MovedAttribute("unquote_plus", "urllib", "urllib.parse"),
    MovedAttribute("unquote_to_bytes", "urllib", "urllib.parse", "unquote", "unquote_to_bytes"),
    MovedAttribute("urlencode", "urllib", "urllib.parse"),
    MovedAttribute("splitquery", "urllib", "urllib.parse"),
    MovedAttribute("splittag", "urllib", "urllib.parse"),
    MovedAttribute("splituser", "urllib", "urllib.parse"),
    MovedAttribute("splitvalue", "urllib", "urllib.parse"),
    MovedAttribute("uses_fragment", "urlparse", "urllib.parse"),
    MovedAttribute("uses_netloc", "urlparse", "urllib.parse"),
    MovedAttribute("uses_params", "urlparse", "urllib.parse"),
    MovedAttribute("uses_query", "urlparse", "urllib.parse"),
    MovedAttribute("uses_relative", "urlparse", "urllib.parse"),
]
for attr in _urllib_parse_moved_attributes:
    setattr(Module_six_moves_urllib_parse, attr.name, attr)
del attr

Module_six_moves_urllib_parse._moved_attributes = _urllib_parse_moved_attributes

_importer._add_module(Module_six_moves_urllib_parse(__name__ + ".moves.urllib_parse"),
                      "moves.urllib_parse", "moves.urllib.parse")


class Module_six_moves_urllib_error(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_error"""


_urllib_error_moved_attributes = [
    MovedAttribute("URLError", "urllib2", "urllib.error"),
    MovedAttribute("HTTPError", "urllib2", "urllib.error"),
    MovedAttribute("ContentTooShortError", "urllib", "urllib.error"),
]
for attr in _urllib_error_moved_attributes:
    setattr(Module_six_moves_urllib_error, attr.name, attr)
del attr

Module_six_moves_urllib_error._moved_attributes = _urllib_error_moved_attributes

_importer._add_module(Module_six_moves_urllib_error(__name__ + ".moves.urllib.error"),
                      "moves.urllib_error", "moves.urllib.error")


class Module_six_moves_urllib_request(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_request"""


_urllib_request_moved_attributes = [
    MovedAttribute("urlopen", "urllib2", "urllib.request"),
    MovedAttribute("install_opener", "urllib2", "urllib.request"),
    MovedAttribute("build_opener", "urllib2", "urllib.request"),
    MovedAttribute("pathname2url", "urllib", "urllib.request"),
    MovedAttribute("url2pathname", "urllib", "urllib.request"),
    MovedAttribute("getproxies", "urllib", "urllib.request"),
    MovedAttribute("Request", "urllib2", "urllib.request"),
    MovedAttribute("OpenerDirector", "urllib2", "urllib.request"),
    MovedAttribute("HTTPDefaultErrorHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPRedirectHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPCookieProcessor", "urllib2", "urllib.request"),
    MovedAttribute("ProxyHandler", "urllib2", "urllib.request"),
    MovedAttribute("BaseHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPPasswordMgr", "urllib2", "urllib.request"),
    MovedAttribute("HTTPPasswordMgrWithDefaultRealm", "urllib2", "urllib.request"),
    MovedAttribute("AbstractBasicAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPBasicAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("ProxyBasicAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("AbstractDigestAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPDigestAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("ProxyDigestAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPSHandler", "urllib2", "urllib.request"),
    MovedAttribute("FileHandler", "urllib2", "urllib.request"),
    MovedAttribute("FTPHandler", "urllib2", "urllib.request"),
    MovedAttribute("CacheFTPHandler", "urllib2", "urllib.request"),
    MovedAttribute("UnknownHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPErrorProcessor", "urllib2", "urllib.request"),
    MovedAttribute("urlretrieve", "urllib", "urllib.request"),
    MovedAttribute("urlcleanup", "urllib", "urllib.request"),
    MovedAttribute("proxy_bypass", "urllib", "urllib.request"),
    MovedAttribute("parse_http_list", "urllib2", "urllib.request"),
    MovedAttribute("parse_keqv_list", "urllib2", "urllib.request"),
]
if sys.version_info[:2] < (3, 14):
    _urllib_request_moved_attributes.extend(
        [
            MovedAttribute("URLopener", "urllib", "urllib.request"),
            MovedAttribute("FancyURLopener", "urllib", "urllib.request"),
        ]
    )
for attr in _urllib_request_moved_attributes:
    setattr(Module_six_moves_urllib_request, attr.name, attr)
del attr

Module_six_moves_urllib_request._moved_attributes = _urllib_request_moved_attributes

_importer._add_module(Module_six_moves_urllib_request(__name__ + ".moves.urllib.request"),
                      "moves.urllib_request", "moves.urllib.request")


class Module_six_moves_urllib_response(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_response"""


_urllib_response_moved_attributes = [
    MovedAttribute("addbase", "urllib", "urllib.response"),
    MovedAttribute("addclosehook", "urllib", "urllib.response"),
    MovedAttribute("addinfo", "urllib", "urllib.response"),
    MovedAttribute("addinfourl", "urllib", "urllib.response"),
]
for attr in _urllib_response_moved_attributes:
    setattr(Module_six_moves_urllib_response, attr.name, attr)
del attr

Module_six_moves_urllib_response._moved_attributes = _urllib_response_moved_attributes

_importer._add_module(Module_six_moves_urllib_response(__name__ + ".moves.urllib.response"),
                      "moves.urllib_response", "moves.urllib.response")


class Module_six_moves_urllib_robotparser(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_robotparser"""


_urllib_robotparser_moved_attributes = [
    MovedAttribute("RobotFileParser", "robotparser", "urllib.robotparser"),
]
for attr in _urllib_robotparser_moved_attributes:
    setattr(Module_six_moves_urllib_robotparser, attr.name, attr)
del attr

Module_six_moves_urllib_robotparser._moved_attributes = _urllib_robotparser_moved_attributes

_importer._add_module(Module_six_moves_urllib_robotparser(__name__ + ".moves.urllib.robotparser"),
                      "moves.urllib_robotparser", "moves.urllib.robotparser")


class Module_six_moves_urllib(types.ModuleType):

    """Create a six.moves.urllib namespace that resembles the Python 3 namespace"""
    __path__ = []  # mark as package
    parse = _importer._get_module("moves.urllib_parse")
    error = _importer._get_module("moves.urllib_error")
    request = _importer._get_module("moves.urllib_request")
    response = _importer._get_module("moves.urllib_response")
    robotparser = _importer._get_module("moves.urllib_robotparser")

    def __dir__(self):
        return ['parse', 'error', 'request', 'response', 'robotparser']

_importer._add_module(Module_six_moves_urllib(__name__ + ".moves.urllib"),
                      "moves.urllib")


def add_move(move):
    """Add an item to six.moves."""
    setattr(_MovedItems, move.name, move)


def remove_move(name):
    """Remove item from six.moves."""
    try:
        delattr(_MovedItems, name)
    except AttributeError:
        try:
            del moves.__dict__[name]
        except KeyError:
            raise AttributeError("no such move, %r" % (name,))


if PY3:
    _meth_func = "__func__"
    _meth_self = "__self__"

    _func_closure = "__closure__"
    _func_code = "__code__"
    _func_defaults = "__defaults__"
    _func_globals = "__globals__"
else:
    _meth_func = "im_func"
    _meth_self = "im_self"

    _func_closure = "func_closure"
    _func_code = "func_code"
    _func_defaults = "func_defaults"
    _func_globals = "func_globals"


try:
    advance_iterator = next
except NameError:
    def advance_iterator(it):
        return it.next()
next = advance_iterator


try:
    callable = callable
except NameError:
    def callable(obj):
        return any("__call__" in klass.__dict__ for klass in type(obj).__mro__)


if PY3:
    def get_unbound_function(unbound):
        return unbound

    create_bound_method = types.MethodType

    def create_unbound_method(func, cls):
        return func

    Iterator = object
else:
    def get_unbound_function(unbound):
        return unbound.im_func

    def create_bound_method(func, obj):
        return types.MethodType(func, obj, obj.__class__)

    def create_unbound_method(func, cls):
        return types.MethodType(func, None, cls)

    class Iterator(object):

        def next(self):
            return type(self).__next__(self)

    callable = callable
_add_doc(get_unbound_function,
         """Get the function out of a possibly unbound function""")


get_method_function = operator.attrgetter(_meth_func)
get_method_self = operator.attrgetter(_meth_self)
get_function_closure = operator.attrgetter(_func_closure)
get_function_code = operator.attrgetter(_func_code)
get_function_defaults = operator.attrgetter(_func_defaults)
get_function_globals = operator.attrgetter(_func_globals)


if PY3:
    def iterkeys(d, **kw):
        return iter(d.keys(**kw))

    def itervalues(d, **kw):
        return iter(d.values(**kw))

    def iteritems(d, **kw):
        return iter(d.items(**kw))

    def iterlists(d, **kw):
        return iter(d.lists(**kw))

    viewkeys = operator.methodcaller("keys")

    viewvalues = operator.methodcaller("values")

    viewitems = operator.methodcaller("items")
else:
    def iterkeys(d, **kw):
        return d.iterkeys(**kw)

    def itervalues(d, **kw):
        return d.itervalues(**kw)

    def iteritems(d, **kw):
        return d.iteritems(**kw)

    def iterlists(d, **kw):
        return d.iterlists(**kw)

    viewkeys = operator.methodcaller("viewkeys")

    viewvalues = operator.methodcaller("viewvalues")

    viewitems = operator.methodcaller("viewitems")

_add_doc(iterkeys, "Return an iterator over the keys of a dictionary.")
_add_doc(itervalues, "Return an iterator over the values of a dictionary.")
_add_doc(iteritems,
         "Return an iterator over the (key, value) pairs of a dictionary.")
_add_doc(iterlists,
         "Return an iterator over the (key, [values]) pairs of a dictionary.")


if PY3:
    def b(s):
        return s.encode("latin-1")

    def u(s):
        return s
    unichr = chr
    import struct
    int2byte = struct.Struct(">B").pack
    del struct
    byte2int = operator.itemgetter(0)
    indexbytes = operator.getitem
    iterbytes = iter
    import io
    StringIO = io.StringIO
    BytesIO = io.BytesIO
    del io
    _assertCountEqual = "assertCountEqual"
    if sys.version_info[1] <= 1:
        _assertRaisesRegex = "assertRaisesRegexp"
        _assertRegex = "assertRegexpMatches"
        _assertNotRegex = "assertNotRegexpMatches"
    else:
        _assertRaisesRegex = "assertRaisesRegex"
        _assertRegex = "assertRegex"
        _assertNotRegex = "assertNotRegex"
else:
    def b(s):
        return s
    # Workaround for standalone backslash

    def u(s):
        return unicode(s.replace(r'\\', r'\\\\'), "unicode_escape")
    unichr = unichr
    int2byte = chr

    def byte2int(bs):
        return ord(bs[0])

    def indexbytes(buf, i):
        return ord(buf[i])
    iterbytes = functools.partial(itertools.imap, ord)
    import StringIO
    StringIO = BytesIO = StringIO.StringIO
    _assertCountEqual = "assertItemsEqual"
    _assertRaisesRegex = "assertRaisesRegexp"
    _assertRegex = "assertRegexpMatches"
    _assertNotRegex = "assertNotRegexpMatches"
_add_doc(b, """Byte literal""")
_add_doc(u, """Text literal""")


def assertCountEqual(self, *args, **kwargs):
    return getattr(self, _assertCountEqual)(*args, **kwargs)


def assertRaisesRegex(self, *args, **kwargs):
    return getattr(self, _assertRaisesRegex)(*args, **kwargs)


def assertRegex(self, *args, **kwargs):
    return getattr(self, _assertRegex)(*args, **kwargs)


def assertNotRegex(self, *args, **kwargs):
    return getattr(self, _assertNotRegex)(*args, **kwargs)


if PY3:
    exec_ = getattr(moves.builtins, "exec")

    def reraise(tp, value, tb=None):
        try:
            if value is None:
                value = tp()
            if value.__traceback__ is not tb:
                raise value.with_traceback(tb)
            raise value
        finally:
            value = None
            tb = None

else:
    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")

    exec_("""def reraise(tp, value, tb=None):
    try:
        raise tp, value, tb
    finally:
        tb = None
""")


if sys.version_info[:2] > (3,):
    exec_("""def raise_from(value, from_value):
    try:
        raise value from from_value
    finally:
        value = None
""")
else:
    def raise_from(value, from_value):
        raise value


print_ = getattr(moves.builtins, "print", None)
if print_ is None:
    def print_(*args, **kwargs):
        """The new-style print function for Python 2.4 and 2.5."""
        fp = kwargs.pop("file", sys.stdout)
        if fp is None:
            return

        def write(data):
            if not isinstance(data, basestring):
                data = str(data)
            # If the file has an encoding, encode unicode with it.
            if (isinstance(fp, file) and
                    isinstance(data, unicode) and
                    fp.encoding is not None):
                errors = getattr(fp, "errors", None)
                if errors is None:
                    errors = "strict"
                data = data.encode(fp.encoding, errors)
            fp.write(data)
        want_unicode = False
        sep = kwargs.pop("sep", None)
        if sep is not None:
            if isinstance(sep, unicode):
                want_unicode = True
            elif not isinstance(sep, str):
                raise TypeError("sep must be None or a string")
        end = kwargs.pop("end", None)
        if end is not None:
            if isinstance(end, unicode):
                want_unicode = True
            elif not isinstance(end, str):
                raise TypeError("end must be None or a string")
        if kwargs:
            raise TypeError("invalid keyword arguments to print()")
        if not want_unicode:
            for arg in args:
                if isinstance(arg, unicode):
                    want_unicode = True
                    break
        if want_unicode:
            newline = unicode("\n")
            space = unicode(" ")
        else:
            newline = "\n"
            space = " "
        if sep is None:
            sep = space
        if end is None:
            end = newline
        for i, arg in enumerate(args):
            if i:
                write(sep)
            write(arg)
        write(end)
if sys.version_info[:2] < (3, 3):
    _print = print_

    def print_(*args, **kwargs):
        fp = kwargs.get("file", sys.stdout)
        flush = kwargs.pop("flush", False)
        _print(*args, **kwargs)
        if flush and fp is not None:
            fp.flush()

_add_doc(reraise, """Reraise an exception.""")

if sys.version_info[0:2] < (3, 4):
    # This does exactly the same what the :func:`py3:functools.update_wrapper`
    # function does on Python versions after 3.2. It sets the ``__wrapped__``
    # attribute on ``wrapper`` object and it doesn't raise an error if any of
    # the attributes mentioned in ``assigned`` and ``updated`` are missing on
    # ``wrapped`` object.
    def _update_wrapper(wrapper, wrapped,
                        assigned=functools.WRAPPER_ASSIGNMENTS,
                        updated=functools.WRAPPER_UPDATES):
        for attr in assigned:
            try:
                value = getattr(wrapped, attr)
            except AttributeError:
                continue
            else:
                setattr(wrapper, attr, value)
        for attr in updated:
            getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
        wrapper.__wrapped__ = wrapped
        return wrapper
    _update_wrapper.__doc__ = functools.update_wrapper.__doc__

    def wraps(wrapped, assigned=functools.WRAPPER_ASSIGNMENTS,
              updated=functools.WRAPPER_UPDATES):
        return functools.partial(_update_wrapper, wrapped=wrapped,
                                 assigned=assigned, updated=updated)
    wraps.__doc__ = functools.wraps.__doc__

else:
    wraps = functools.wraps


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(type):

        def __new__(cls, name, this_bases, d):
            if sys.version_info[:2] >= (3, 7):
                # This version introduced PEP 560 that requires a bit
                # of extra care (we mimic what is done by __build_class__).
                resolved_bases = types.resolve_bases(bases)
                if resolved_bases is not bases:
                    d['__orig_bases__'] = bases
            else:
                resolved_bases = bases
            return meta(name, resolved_bases, d)

        @classmethod
        def __prepare__(cls, name, this_bases):
            return meta.__prepare__(name, bases)
    return type.__new__(metaclass, 'temporary_class', (), {})


def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        if hasattr(cls, '__qualname__'):
            orig_vars['__qualname__'] = cls.__qualname__
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper


def ensure_binary(s, encoding='utf-8', errors='strict'):
    """Coerce **s** to six.binary_type.

    For Python 2:
      - `unicode` -> encoded to `str`
      - `str` -> `str`

    For Python 3:
      - `str` -> encoded to `bytes`
      - `bytes` -> `bytes`
    """
    if isinstance(s, binary_type):
        return s
    if isinstance(s, text_type):
        return s.encode(encoding, errors)
    raise TypeError("not expecting type '%s'" % type(s))


def ensure_str(s, encoding='utf-8', errors='strict'):
    """Coerce *s* to `str`.

    For Python 2:
      - `unicode` -> encoded to `str`
      - `str` -> `str`

    For Python 3:
      - `str` -> `str`
      - `bytes` -> decoded to `str`
    """
    # Optimization: Fast return for the common case.
    if type(s) is str:
        return s
    if PY2 and isinstance(s, text_type):
        return s.encode(encoding, errors)
    elif PY3 and isinstance(s, binary_type):
        return s.decode(encoding, errors)
    elif not isinstance(s, (text_type, binary_type)):
        raise TypeError("not expecting type '%s'" % type(s))
    return s


def ensure_text(s, encoding='utf-8', errors='strict'):
    """Coerce *s* to six.text_type.

    For Python 2:
      - `unicode` -> `unicode`
      - `str` -> `unicode`

    For Python 3:
      - `str` -> `str`
      - `bytes` -> decoded to `str`
    """
    if isinstance(s, binary_type):
        return s.decode(encoding, errors)
    elif isinstance(s, text_type):
        return s
    else:
        raise TypeError("not expecting type '%s'" % type(s))


def python_2_unicode_compatible(klass):
    """
    A class decorator that defines __unicode__ and __str__ methods under Python 2.
    Under Python 3 it does nothing.

    To support Python 2 and 3 with a single code base, define a __str__ method
    returning text and apply this decorator to the class.
    """
    if PY2:
        if '__str__' not in klass.__dict__:
            raise ValueError("@python_2_unicode_compatible cannot be applied "
                             "to %s because it doesn't define __str__()." %
                             klass.__name__)
        klass.__unicode__ = klass.__str__
        klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return klass


# Complete the moves implementation.
# This code is at the end of this module to speed up module loading.
# Turn this module into a package.
__path__ = []  # required for PEP 302 and PEP 451
__package__ = __name__  # see PEP 366 @ReservedAssignment
if globals().get("__spec__") is not None:
    __spec__.submodule_search_locations = []  # PEP 451 @UndefinedVariable
# Remove other six meta path importers, since they cause problems. This can
# happen if six is removed from sys.modules and then reloaded. (Setuptools does
# this for some reason.)
if sys.meta_path:
    for i, importer in enumerate(sys.meta_path):
        # Here's some real nastiness: Another "instance" of the six module might
        # be floating around. Therefore, we can't use isinstance() to check for
        # the six meta path importer, since the other six instance will have
        # inserted an importer with different class.
        if (type(importer).__name__ == "_SixMetaPathImporter" and
                importer.name == __name__):
            del sys.meta_path[i]
            break
    del i, importer
# Finally, add the importer to the meta path import hook.
sys.meta_path.append(_importer)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\shape_base.py ===
__all__ = ['atleast_1d', 'atleast_2d', 'atleast_3d', 'block', 'hstack',
           'stack', 'unstack', 'vstack']

import functools
import itertools
import operator

from . import fromnumeric as _from_nx
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _atleast_1d_dispatcher(*arys):
    return arys


@array_function_dispatch(_atleast_1d_dispatcher)
def atleast_1d(*arys):
    """
    Convert inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more input arrays.

    Returns
    -------
    ret : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_2d, atleast_3d

    Examples
    --------
    >>> import numpy as np
    >>> np.atleast_1d(1.0)
    array([1.])

    >>> x = np.arange(9.0).reshape(3,3)
    >>> np.atleast_1d(x)
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> np.atleast_1d(x) is x
    True

    >>> np.atleast_1d(1, [3, 4])
    (array([1]), array([3, 4]))

    """
    if len(arys) == 1:
        result = asanyarray(arys[0])
        if result.ndim == 0:
            result = result.reshape(1)
        return result
    res = []
    for ary in arys:
        result = asanyarray(ary)
        if result.ndim == 0:
            result = result.reshape(1)
        res.append(result)
    return tuple(res)


def _atleast_2d_dispatcher(*arys):
    return arys


@array_function_dispatch(_atleast_2d_dispatcher)
def atleast_2d(*arys):
    """
    View inputs as arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.

    Returns
    -------
    res, res2, ... : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    See Also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    >>> import numpy as np
    >>> np.atleast_2d(3.0)
    array([[3.]])

    >>> x = np.arange(3.0)
    >>> np.atleast_2d(x)
    array([[0., 1., 2.]])
    >>> np.atleast_2d(x).base is x
    True

    >>> np.atleast_2d(1, [1, 2], [[1, 2]])
    (array([[1]]), array([[1, 2]]), array([[1, 2]]))

    """
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[_nx.newaxis, :]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)


def _atleast_3d_dispatcher(*arys):
    return arys


@array_function_dispatch(_atleast_3d_dispatcher)
def atleast_3d(*arys):
    """
    View inputs as arrays with at least three dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted to
        arrays.  Arrays that already have three or more dimensions are
        preserved.

    Returns
    -------
    res1, res2, ... : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= 3``.  Copies are
        avoided where possible, and views with three or more dimensions are
        returned.  For example, a 1-D array of shape ``(N,)`` becomes a view
        of shape ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a
        view of shape ``(M, N, 1)``.

    See Also
    --------
    atleast_1d, atleast_2d

    Examples
    --------
    >>> import numpy as np
    >>> np.atleast_3d(3.0)
    array([[[3.]]])

    >>> x = np.arange(3.0)
    >>> np.atleast_3d(x).shape
    (1, 3, 1)

    >>> x = np.arange(12.0).reshape(4,3)
    >>> np.atleast_3d(x).shape
    (4, 3, 1)
    >>> np.atleast_3d(x).base is x.base  # x is a reshape, so not base itself
    True

    >>> for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
    ...     print(arr, arr.shape) # doctest: +SKIP
    ...
    [[[1]
      [2]]] (1, 2, 1)
    [[[1]
      [2]]] (1, 2, 1)
    [[[1 2]]] (1, 1, 2)

    """
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1, 1)
        elif ary.ndim == 1:
            result = ary[_nx.newaxis, :, _nx.newaxis]
        elif ary.ndim == 2:
            result = ary[:, :, _nx.newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)


def _arrays_for_stack_dispatcher(arrays):
    if not hasattr(arrays, "__getitem__"):
        raise TypeError('arrays to stack must be passed as a "sequence" type '
                        'such as list or tuple.')

    return tuple(arrays)


def _vhstack_dispatcher(tup, *, dtype=None, casting=None):
    return _arrays_for_stack_dispatcher(tup)


@array_function_dispatch(_vhstack_dispatcher)
def vstack(tup, *, dtype=None, casting="same_kind"):
    """
    Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length. In the case of a single
        array_like input, it will be treated as a sequence of arrays; i.e.,
        each element along the zeroth axis is treated as a separate array.

    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.24

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        .. versionadded:: 1.24

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    hstack : Stack arrays in sequence horizontally (column wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    vsplit : Split an array into multiple sub-arrays vertically (row-wise).
    unstack : Split an array into a tuple of sub-arrays along an axis.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.vstack((a,b))
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[4], [5], [6]])
    >>> np.vstack((a,b))
    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])

    """
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)


@array_function_dispatch(_vhstack_dispatcher)
def hstack(tup, *, dtype=None, casting="same_kind"):
    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length. In the case of a single
        array_like input, it will be treated as a sequence of arrays; i.e.,
        each element along the zeroth axis is treated as a separate array.

    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.24

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        .. versionadded:: 1.24

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    hsplit : Split an array into multiple sub-arrays
             horizontally (column-wise).
    unstack : Split an array into a tuple of sub-arrays along an axis.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((1,2,3))
    >>> b = np.array((4,5,6))
    >>> np.hstack((a,b))
    array([1, 2, 3, 4, 5, 6])
    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[4],[5],[6]])
    >>> np.hstack((a,b))
    array([[1, 4],
           [2, 5],
           [3, 6]])

    """
    arrs = atleast_1d(*tup)
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs and arrs[0].ndim == 1:
        return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    else:
        return _nx.concatenate(arrs, 1, dtype=dtype, casting=casting)


def _stack_dispatcher(arrays, axis=None, out=None, *,
                      dtype=None, casting=None):
    arrays = _arrays_for_stack_dispatcher(arrays)
    if out is not None:
        # optimize for the typical case where only arrays is provided
        arrays = list(arrays)
        arrays.append(out)
    return arrays


@array_function_dispatch(_stack_dispatcher)
def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    """
    Join a sequence of arrays along a new axis.

    The ``axis`` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if ``axis=0`` it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    Parameters
    ----------
    arrays : sequence of ndarrays
        Each array must have the same shape. In the case of a single ndarray
        array_like input, it will be treated as a sequence of arrays; i.e.,
        each element along the zeroth axis is treated as a separate array.

    axis : int, optional
        The axis in the result array along which the input arrays are stacked.

    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what stack would have returned if no
        out argument were specified.

    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.24

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        .. versionadded:: 1.24


    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    block : Assemble an nd-array from nested lists of blocks.
    split : Split array into a list of multiple sub-arrays of equal size.
    unstack : Split an array into a tuple of sub-arrays along an axis.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> arrays = [rng.normal(size=(3,4)) for _ in range(10)]
    >>> np.stack(arrays, axis=0).shape
    (10, 3, 4)

    >>> np.stack(arrays, axis=1).shape
    (3, 10, 4)

    >>> np.stack(arrays, axis=2).shape
    (3, 4, 10)

    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.stack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> np.stack((a, b), axis=-1)
    array([[1, 4],
           [2, 5],
           [3, 6]])

    """
    arrays = [asanyarray(arr) for arr in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')

    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError('all input arrays must have the same shape')

    result_ndim = arrays[0].ndim + 1
    axis = normalize_axis_index(axis, result_ndim)

    sl = (slice(None),) * axis + (_nx.newaxis,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return _nx.concatenate(expanded_arrays, axis=axis, out=out,
                           dtype=dtype, casting=casting)

def _unstack_dispatcher(x, /, *, axis=None):
    return (x,)

@array_function_dispatch(_unstack_dispatcher)
def unstack(x, /, *, axis=0):
    """
    Split an array into a sequence of arrays along the given axis.

    The ``axis`` parameter specifies the dimension along which the array will
    be split. For example, if ``axis=0`` (the default) it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    The result is a tuple of arrays split along ``axis``.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    x : ndarray
        The array to be unstacked.
    axis : int, optional
        Axis along which the array will be split. Default: ``0``.

    Returns
    -------
    unstacked : tuple of ndarrays
        The unstacked arrays.

    See Also
    --------
    stack : Join a sequence of arrays along a new axis.
    concatenate : Join a sequence of arrays along an existing axis.
    block : Assemble an nd-array from nested lists of blocks.
    split : Split array into a list of multiple sub-arrays of equal size.

    Notes
    -----
    ``unstack`` serves as the reverse operation of :py:func:`stack`, i.e.,
    ``stack(unstack(x, axis=axis), axis=axis) == x``.

    This function is equivalent to ``tuple(np.moveaxis(x, axis, 0))``, since
    iterating on an array iterates along the first axis.

    Examples
    --------
    >>> arr = np.arange(24).reshape((2, 3, 4))
    >>> np.unstack(arr)
    (array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]),
     array([[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]))
    >>> np.unstack(arr, axis=1)
    (array([[ 0,  1,  2,  3],
            [12, 13, 14, 15]]),
     array([[ 4,  5,  6,  7],
            [16, 17, 18, 19]]),
     array([[ 8,  9, 10, 11],
            [20, 21, 22, 23]]))
    >>> arr2 = np.stack(np.unstack(arr, axis=1), axis=1)
    >>> arr2.shape
    (2, 3, 4)
    >>> np.all(arr == arr2)
    np.True_

    """
    if x.ndim == 0:
        raise ValueError("Input array must be at least 1-d.")
    return tuple(_nx.moveaxis(x, axis, 0))


# Internal functions to eliminate the overhead of repeated dispatch in one of
# the two possible paths inside np.block.
# Use getattr to protect against __array_function__ being disabled.
_size = getattr(_from_nx.size, '__wrapped__', _from_nx.size)
_ndim = getattr(_from_nx.ndim, '__wrapped__', _from_nx.ndim)
_concatenate = getattr(_from_nx.concatenate,
                       '__wrapped__', _from_nx.concatenate)


def _block_format_index(index):
    """
    Convert a list of indices ``[0, 1, 2]`` into ``"arrays[0][1][2]"``.
    """
    idx_str = ''.join(f'[{i}]' for i in index if i is not None)
    return 'arrays' + idx_str


def _block_check_depths_match(arrays, parent_index=[]):
    """
    Recursive function checking that the depths of nested lists in `arrays`
    all match. Mismatch raises a ValueError as described in the block
    docstring below.

    The entire index (rather than just the depth) needs to be calculated
    for each innermost list, in case an error needs to be raised, so that
    the index of the offending list can be printed as part of the error.

    Parameters
    ----------
    arrays : nested list of arrays
        The arrays to check
    parent_index : list of int
        The full index of `arrays` within the nested lists passed to
        `_block_check_depths_match` at the top of the recursion.

    Returns
    -------
    first_index : list of int
        The full index of an element from the bottom of the nesting in
        `arrays`. If any element at the bottom is an empty list, this will
        refer to it, and the last index along the empty axis will be None.
    max_arr_ndim : int
        The maximum of the ndims of the arrays nested in `arrays`.
    final_size: int
        The number of elements in the final array. This is used the motivate
        the choice of algorithm used using benchmarking wisdom.

    """
    if isinstance(arrays, tuple):
        # not strictly necessary, but saves us from:
        #  - more than one way to do things - no point treating tuples like
        #    lists
        #  - horribly confusing behaviour that results when tuples are
        #    treated like ndarray
        raise TypeError(
            f'{_block_format_index(parent_index)} is a tuple. '
            'Only lists can be used to arrange blocks, and np.block does '
            'not allow implicit conversion from tuple to ndarray.'
        )
    elif isinstance(arrays, list) and len(arrays) > 0:
        idxs_ndims = (_block_check_depths_match(arr, parent_index + [i])
                      for i, arr in enumerate(arrays))

        first_index, max_arr_ndim, final_size = next(idxs_ndims)
        for index, ndim, size in idxs_ndims:
            final_size += size
            if ndim > max_arr_ndim:
                max_arr_ndim = ndim
            if len(index) != len(first_index):
                raise ValueError(
                    "List depths are mismatched. First element was at "
                    f"depth {len(first_index)}, but there is an element at "
                    f"depth {len(index)} ({_block_format_index(index)})"
                )
            # propagate our flag that indicates an empty list at the bottom
            if index[-1] is None:
                first_index = index

        return first_index, max_arr_ndim, final_size
    elif isinstance(arrays, list) and len(arrays) == 0:
        # We've 'bottomed out' on an empty list
        return parent_index + [None], 0, 0
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        size = _size(arrays)
        return parent_index, _ndim(arrays), size


def _atleast_nd(a, ndim):
    # Ensures `a` has at least `ndim` dimensions by prepending
    # ones to `a.shape` as necessary
    return array(a, ndmin=ndim, copy=None, subok=True)


def _accumulate(values):
    return list(itertools.accumulate(values))


def _concatenate_shapes(shapes, axis):
    """Given array shapes, return the resulting shape and slices prefixes.

    These help in nested concatenation.

    Returns
    -------
    shape: tuple of int
        This tuple satisfies::

            shape, _ = _concatenate_shapes([arr.shape for shape in arrs], axis)
            shape == concatenate(arrs, axis).shape

    slice_prefixes: tuple of (slice(start, end), )
        For a list of arrays being concatenated, this returns the slice
        in the larger array at axis that needs to be sliced into.

        For example, the following holds::

            ret = concatenate([a, b, c], axis)
            _, (sl_a, sl_b, sl_c) = concatenate_slices([a, b, c], axis)

            ret[(slice(None),) * axis + sl_a] == a
            ret[(slice(None),) * axis + sl_b] == b
            ret[(slice(None),) * axis + sl_c] == c

        These are called slice prefixes since they are used in the recursive
        blocking algorithm to compute the left-most slices during the
        recursion. Therefore, they must be prepended to rest of the slice
        that was computed deeper in the recursion.

        These are returned as tuples to ensure that they can quickly be added
        to existing slice tuple without creating a new tuple every time.

    """
    # Cache a result that will be reused.
    shape_at_axis = [shape[axis] for shape in shapes]

    # Take a shape, any shape
    first_shape = shapes[0]
    first_shape_pre = first_shape[:axis]
    first_shape_post = first_shape[axis + 1:]

    if any(shape[:axis] != first_shape_pre or
           shape[axis + 1:] != first_shape_post for shape in shapes):
        raise ValueError(
            f'Mismatched array shapes in block along axis {axis}.')

    shape = (first_shape_pre + (sum(shape_at_axis),) + first_shape[axis + 1:])

    offsets_at_axis = _accumulate(shape_at_axis)
    slice_prefixes = [(slice(start, end),)
                      for start, end in zip([0] + offsets_at_axis,
                                            offsets_at_axis)]
    return shape, slice_prefixes


def _block_info_recursion(arrays, max_depth, result_ndim, depth=0):
    """
    Returns the shape of the final array, along with a list
    of slices and a list of arrays that can be used for assignment inside the
    new array

    Parameters
    ----------
    arrays : nested list of arrays
        The arrays to check
    max_depth : list of int
        The number of nested lists
    result_ndim : int
        The number of dimensions in thefinal array.

    Returns
    -------
    shape : tuple of int
        The shape that the final array will take on.
    slices: list of tuple of slices
        The slices into the full array required for assignment. These are
        required to be prepended with ``(Ellipsis, )`` to obtain to correct
        final index.
    arrays: list of ndarray
        The data to assign to each slice of the full array

    """
    if depth < max_depth:
        shapes, slices, arrays = zip(
            *[_block_info_recursion(arr, max_depth, result_ndim, depth + 1)
              for arr in arrays])

        axis = result_ndim - max_depth + depth
        shape, slice_prefixes = _concatenate_shapes(shapes, axis)

        # Prepend the slice prefix and flatten the slices
        slices = [slice_prefix + the_slice
                  for slice_prefix, inner_slices in zip(slice_prefixes, slices)
                  for the_slice in inner_slices]

        # Flatten the array list
        arrays = functools.reduce(operator.add, arrays)

        return shape, slices, arrays
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        # type(arrays) is not list
        # Return the slice and the array inside a list to be consistent with
        # the recursive case.
        arr = _atleast_nd(arrays, result_ndim)
        return arr.shape, [()], [arr]


def _block(arrays, max_depth, result_ndim, depth=0):
    """
    Internal implementation of block based on repeated concatenation.
    `arrays` is the argument passed to
    block. `max_depth` is the depth of nested lists within `arrays` and
    `result_ndim` is the greatest of the dimensions of the arrays in
    `arrays` and the depth of the lists in `arrays` (see block docstring
    for details).
    """
    if depth < max_depth:
        arrs = [_block(arr, max_depth, result_ndim, depth + 1)
                for arr in arrays]
        return _concatenate(arrs, axis=-(max_depth - depth))
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        # type(arrays) is not list
        return _atleast_nd(arrays, result_ndim)


def _block_dispatcher(arrays):
    # Use type(...) is list to match the behavior of np.block(), which special
    # cases list specifically rather than allowing for generic iterables or
    # tuple. Also, we know that list.__array_function__ will never exist.
    if isinstance(arrays, list):
        for subarrays in arrays:
            yield from _block_dispatcher(subarrays)
    else:
        yield arrays


@array_function_dispatch(_block_dispatcher)
def block(arrays):
    """
    Assemble an nd-array from nested lists of blocks.

    Blocks in the innermost lists are concatenated (see `concatenate`) along
    the last dimension (-1), then these are concatenated along the
    second-last dimension (-2), and so on until the outermost list is reached.

    Blocks can be of any dimension, but will not be broadcasted using
    the normal rules. Instead, leading axes of size 1 are inserted,
    to make ``block.ndim`` the same for all blocks. This is primarily useful
    for working with scalars, and means that code like ``np.block([v, 1])``
    is valid, where ``v.ndim == 1``.

    When the nested list is two levels deep, this allows block matrices to be
    constructed from their components.

    Parameters
    ----------
    arrays : nested list of array_like or scalars (but not tuples)
        If passed a single ndarray or scalar (a nested list of depth 0), this
        is returned unmodified (and not copied).

        Elements shapes must match along the appropriate axes (without
        broadcasting), but leading 1s will be prepended to the shape as
        necessary to make the dimensions match.

    Returns
    -------
    block_array : ndarray
        The array assembled from the given blocks.

        The dimensionality of the output is equal to the greatest of:

        * the dimensionality of all the inputs
        * the depth to which the input list is nested

    Raises
    ------
    ValueError
        * If list depths are mismatched - for instance, ``[[a, b], c]`` is
          illegal, and should be spelt ``[[a, b], [c]]``
        * If lists are empty - for instance, ``[[a, b], []]``

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    vstack : Stack arrays in sequence vertically (row wise).
    hstack : Stack arrays in sequence horizontally (column wise).
    dstack : Stack arrays in sequence depth wise (along third axis).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    vsplit : Split an array into multiple sub-arrays vertically (row-wise).
    unstack : Split an array into a tuple of sub-arrays along an axis.

    Notes
    -----
    When called with only scalars, ``np.block`` is equivalent to an ndarray
    call. So ``np.block([[1, 2], [3, 4]])`` is equivalent to
    ``np.array([[1, 2], [3, 4]])``.

    This function does not enforce that the blocks lie on a fixed grid.
    ``np.block([[a, b], [c, d]])`` is not restricted to arrays of the form::

        AAAbb
        AAAbb
        cccDD

    But is also allowed to produce, for some ``a, b, c, d``::

        AAAbb
        AAAbb
        cDDDD

    Since concatenation happens along the last axis first, `block` is *not*
    capable of producing the following directly::

        AAAbb
        cccbb
        cccDD

    Matlab's "square bracket stacking", ``[A, B, ...; p, q, ...]``, is
    equivalent to ``np.block([[A, B, ...], [p, q, ...]])``.

    Examples
    --------
    The most common use of this function is to build a block matrix:

    >>> import numpy as np
    >>> A = np.eye(2) * 2
    >>> B = np.eye(3) * 3
    >>> np.block([
    ...     [A,               np.zeros((2, 3))],
    ...     [np.ones((3, 2)), B               ]
    ... ])
    array([[2., 0., 0., 0., 0.],
           [0., 2., 0., 0., 0.],
           [1., 1., 3., 0., 0.],
           [1., 1., 0., 3., 0.],
           [1., 1., 0., 0., 3.]])

    With a list of depth 1, `block` can be used as `hstack`:

    >>> np.block([1, 2, 3])              # hstack([1, 2, 3])
    array([1, 2, 3])

    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.block([a, b, 10])             # hstack([a, b, 10])
    array([ 1,  2,  3,  4,  5,  6, 10])

    >>> A = np.ones((2, 2), int)
    >>> B = 2 * A
    >>> np.block([A, B])                 # hstack([A, B])
    array([[1, 1, 2, 2],
           [1, 1, 2, 2]])

    With a list of depth 2, `block` can be used in place of `vstack`:

    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.block([[a], [b]])             # vstack([a, b])
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> A = np.ones((2, 2), int)
    >>> B = 2 * A
    >>> np.block([[A], [B]])             # vstack([A, B])
    array([[1, 1],
           [1, 1],
           [2, 2],
           [2, 2]])

    It can also be used in place of `atleast_1d` and `atleast_2d`:

    >>> a = np.array(0)
    >>> b = np.array([1])
    >>> np.block([a])                    # atleast_1d(a)
    array([0])
    >>> np.block([b])                    # atleast_1d(b)
    array([1])

    >>> np.block([[a]])                  # atleast_2d(a)
    array([[0]])
    >>> np.block([[b]])                  # atleast_2d(b)
    array([[1]])


    """
    arrays, list_ndim, result_ndim, final_size = _block_setup(arrays)

    # It was found through benchmarking that making an array of final size
    # around 256x256 was faster by straight concatenation on a
    # i7-7700HQ processor and dual channel ram 2400MHz.
    # It didn't seem to matter heavily on the dtype used.
    #
    # A 2D array using repeated concatenation requires 2 copies of the array.
    #
    # The fastest algorithm will depend on the ratio of CPU power to memory
    # speed.
    # One can monitor the results of the benchmark
    # https://pv.github.io/numpy-bench/#bench_shape_base.Block2D.time_block2d
    # to tune this parameter until a C version of the `_block_info_recursion`
    # algorithm is implemented which would likely be faster than the python
    # version.
    if list_ndim * final_size > (2 * 512 * 512):
        return _block_slicing(arrays, list_ndim, result_ndim)
    else:
        return _block_concatenate(arrays, list_ndim, result_ndim)


# These helper functions are mostly used for testing.
# They allow us to write tests that directly call `_block_slicing`
# or `_block_concatenate` without blocking large arrays to force the wisdom
# to trigger the desired path.
def _block_setup(arrays):
    """
    Returns
    (`arrays`, list_ndim, result_ndim, final_size)
    """
    bottom_index, arr_ndim, final_size = _block_check_depths_match(arrays)
    list_ndim = len(bottom_index)
    if bottom_index and bottom_index[-1] is None:
        raise ValueError(
            f'List at {_block_format_index(bottom_index)} cannot be empty'
        )
    result_ndim = max(arr_ndim, list_ndim)
    return arrays, list_ndim, result_ndim, final_size


def _block_slicing(arrays, list_ndim, result_ndim):
    shape, slices, arrays = _block_info_recursion(
        arrays, list_ndim, result_ndim)
    dtype = _nx.result_type(*[arr.dtype for arr in arrays])

    # Test preferring F only in the case that all input arrays are F
    F_order = all(arr.flags['F_CONTIGUOUS'] for arr in arrays)
    C_order = all(arr.flags['C_CONTIGUOUS'] for arr in arrays)
    order = 'F' if F_order and not C_order else 'C'
    result = _nx.empty(shape=shape, dtype=dtype, order=order)
    # Note: In a c implementation, the function
    # PyArray_CreateMultiSortedStridePerm could be used for more advanced
    # guessing of the desired order.

    for the_slice, arr in zip(slices, arrays):
        result[(Ellipsis,) + the_slice] = arr
    return result


def _block_concatenate(arrays, list_ndim, result_ndim):
    result = _block(arrays, list_ndim, result_ndim)
    if list_ndim == 0:
        # Catch an edge case where _block returns a view because
        # `arrays` is a single numpy array and not a list of numpy arrays.
        # This might copy scalars or lists twice, but this isn't a likely
        # usecase for those interested in performance
        result = result.copy()
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\symbol.py ===
from __future__ import annotations


from .assumptions import StdFactKB, _assume_defined
from .basic import Basic, Atom
from .cache import cacheit
from .containers import Tuple
from .expr import Expr, AtomicExpr
from .function import AppliedUndef, FunctionClass
from .kind import NumberKind, UndefinedKind
from .logic import fuzzy_bool
from .singleton import S
from .sorting import ordered
from .sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.misc import filldedent

import string
import re as _re
import random
from itertools import product
from typing import Any


class Str(Atom):
    """
    Represents string in SymPy.

    Explanation
    ===========

    Previously, ``Symbol`` was used where string is needed in ``args`` of SymPy
    objects, e.g. denoting the name of the instance. However, since ``Symbol``
    represents mathematical scalar, this class should be used instead.

    """
    __slots__ = ('name',)

    def __new__(cls, name, **kwargs):
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))
        obj = Expr.__new__(cls, **kwargs)
        obj.name = name
        return obj

    def __getnewargs__(self):
        return (self.name,)

    def _hashable_content(self):
        return (self.name,)


def _filter_assumptions(kwargs):
    """Split the given dict into assumptions and non-assumptions.
    Keys are taken as assumptions if they correspond to an
    entry in ``_assume_defined``.
    """
    assumptions, nonassumptions = map(dict, sift(kwargs.items(),
        lambda i: i[0] in _assume_defined,
        binary=True))
    Symbol._sanitize(assumptions)
    return assumptions, nonassumptions

def _symbol(s, matching_symbol=None, **assumptions):
    """Return s if s is a Symbol, else if s is a string, return either
    the matching_symbol if the names are the same or else a new symbol
    with the same assumptions as the matching symbol (or the
    assumptions as provided).

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.core.symbol import _symbol
    >>> _symbol('y')
    y
    >>> _.is_real is None
    True
    >>> _symbol('y', real=True).is_real
    True

    >>> x = Symbol('x')
    >>> _symbol(x, real=True)
    x
    >>> _.is_real is None  # ignore attribute if s is a Symbol
    True

    Below, the variable sym has the name 'foo':

    >>> sym = Symbol('foo', real=True)

    Since 'x' is not the same as sym's name, a new symbol is created:

    >>> _symbol('x', sym).name
    'x'

    It will acquire any assumptions give:

    >>> _symbol('x', sym, real=False).is_real
    False

    Since 'foo' is the same as sym's name, sym is returned

    >>> _symbol('foo', sym)
    foo

    Any assumptions given are ignored:

    >>> _symbol('foo', sym, real=False).is_real
    True

    NB: the symbol here may not be the same as a symbol with the same
    name defined elsewhere as a result of different assumptions.

    See Also
    ========

    sympy.core.symbol.Symbol

    """
    if isinstance(s, str):
        if matching_symbol and matching_symbol.name == s:
            return matching_symbol
        return Symbol(s, **assumptions)
    elif isinstance(s, Symbol):
        return s
    else:
        raise ValueError('symbol must be string for symbol name or Symbol')

def uniquely_named_symbol(xname, exprs=(), compare=str, modify=None, **assumptions):
    """
    Return a symbol whose name is derivated from *xname* but is unique
    from any other symbols in *exprs*.

    *xname* and symbol names in *exprs* are passed to *compare* to be
    converted to comparable forms. If ``compare(xname)`` is not unique,
    it is recursively passed to *modify* until unique name is acquired.

    Parameters
    ==========

    xname : str or Symbol
        Base name for the new symbol.

    exprs : Expr or iterable of Expr
        Expressions whose symbols are compared to *xname*.

    compare : function
        Unary function which transforms *xname* and symbol names from
        *exprs* to comparable form.

    modify : function
        Unary function which modifies the string. Default is appending
        the number, or increasing the number if exists.

    Examples
    ========

    By default, a number is appended to *xname* to generate unique name.
    If the number already exists, it is recursively increased.

    >>> from sympy.core.symbol import uniquely_named_symbol, Symbol
    >>> uniquely_named_symbol('x', Symbol('x'))
    x0
    >>> uniquely_named_symbol('x', (Symbol('x'), Symbol('x0')))
    x1
    >>> uniquely_named_symbol('x0', (Symbol('x1'), Symbol('x0')))
    x2

    Name generation can be controlled by passing *modify* parameter.

    >>> from sympy.abc import x
    >>> uniquely_named_symbol('x', x, modify=lambda s: 2*s)
    xx

    """
    def numbered_string_incr(s, start=0):
        if not s:
            return str(start)
        i = len(s) - 1
        while i != -1:
            if not s[i].isdigit():
                break
            i -= 1
        n = str(int(s[i + 1:] or start - 1) + 1)
        return s[:i + 1] + n

    default = None
    if is_sequence(xname):
        xname, default = xname
    x = compare(xname)
    if not exprs:
        return _symbol(x, default, **assumptions)
    if not is_sequence(exprs):
        exprs = [exprs]
    names = set().union(
        [i.name for e in exprs for i in e.atoms(Symbol)] +
        [i.func.name for e in exprs for i in e.atoms(AppliedUndef)])
    if modify is None:
        modify = numbered_string_incr
    while any(x == compare(s) for s in names):
        x = modify(x)
    return _symbol(x, default, **assumptions)
_uniquely_named_symbol = uniquely_named_symbol


# XXX: We need type: ignore below because Expr and Boolean are incompatible as
# superclasses. Really Symbol should not be a subclass of Boolean.


class Symbol(AtomicExpr, Boolean): # type: ignore
    """
    Symbol class is used to create symbolic variables.

    Explanation
    ===========

    Symbolic variables are placeholders for mathematical symbols that can represent numbers, constants, or any other mathematical entities and can be used in mathematical expressions and to perform symbolic computations.

    Assumptions:

    commutative = True
    positive = True
    real = True
    imaginary = True
    complex = True
    complete list of more assumptions- :ref:`predicates`

    You can override the default assumptions in the constructor.

    Examples
    ========

    >>> from sympy import Symbol
    >>> x = Symbol("x", positive=True)
    >>> x.is_positive
    True
    >>> x.is_negative
    False

    passing in greek letters:

    >>> from sympy import Symbol
    >>> alpha = Symbol('alpha')
    >>> alpha #doctest: +SKIP
    α

    Trailing digits are automatically treated like subscripts of what precedes them in the name.
    General format to add subscript to a symbol :
    ``<var_name> = Symbol('<symbol_name>_<subscript>')``

    >>> from sympy import Symbol
    >>> alpha_i = Symbol('alpha_i')
    >>> alpha_i #doctest: +SKIP
    αᵢ

    Parameters
    ==========

    AtomicExpr: variable name
    Boolean: Assumption with a boolean value(True or False)
    """

    is_comparable = False

    __slots__ = ('name', '_assumptions_orig', '_assumptions0')

    name: str

    is_Symbol = True
    is_symbol = True

    @property
    def kind(self):
        if self.is_commutative:
            return NumberKind
        return UndefinedKind

    @property
    def _diff_wrt(self):
        """Allow derivatives wrt Symbols.

        Examples
        ========

        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> x._diff_wrt
        True
        """
        return True

    @staticmethod
    def _sanitize(assumptions, obj=None):
        """Remove None, convert values to bool, check commutativity *in place*.
        """

        # be strict about commutativity: cannot be None
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        if is_commutative is None:
            whose = '%s ' % obj.__name__ if obj else ''
            raise ValueError(
                '%scommutativity must be True or False.' % whose)

        # sanitize other assumptions so 1 -> True and 0 -> False
        for key in list(assumptions.keys()):
            v = assumptions[key]
            if v is None:
                assumptions.pop(key)
                continue
            assumptions[key] = bool(v)

    def _merge(self, assumptions):
        base = self.assumptions0
        for k in set(assumptions) & set(base):
            if assumptions[k] != base[k]:
                raise ValueError(filldedent('''
                    non-matching assumptions for %s: existing value
                    is %s and new value is %s''' % (
                    k, base[k], assumptions[k])))
        base.update(assumptions)
        return base

    def __new__(cls, name, **assumptions):
        """Symbols are identified by name and assumptions::

        >>> from sympy import Symbol
        >>> Symbol("x") == Symbol("x")
        True
        >>> Symbol("x", real=True) == Symbol("x", real=False)
        False

        """
        cls._sanitize(assumptions, cls)
        return Symbol.__xnew_cached_(cls, name, **assumptions)


    @staticmethod
    @cacheit
    def _canonical_assumptions(**assumptions):
        # This is retained purely so that srepr can include commutative=True if
        # that was explicitly specified but not if it was not. Ideally srepr
        # should not distinguish these cases because the symbols otherwise
        # compare equal and are considered equivalent.
        #
        # See https://github.com/sympy/sympy/issues/8873
        #
        assumptions_orig = assumptions.copy()

        # The only assumption that is assumed by default is commutative=True:
        assumptions.setdefault('commutative', True)

        assumptions_kb = StdFactKB(assumptions)
        assumptions0 = dict(assumptions_kb)

        return assumptions_kb, assumptions_orig, assumptions0

    @staticmethod
    def __xnew__(cls, name, **assumptions):  # never cached (e.g. dummy)
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))


        obj = Expr.__new__(cls)
        obj.name = name

        assumptions_kb, assumptions_orig, assumptions0 = Symbol._canonical_assumptions(**assumptions)

        obj._assumptions = assumptions_kb
        obj._assumptions_orig = assumptions_orig
        obj._assumptions0 = tuple(sorted(assumptions0.items()))

        # The three assumptions dicts are all a little different:
        #
        #   >>> from sympy import Symbol
        #   >>> x = Symbol('x', finite=True)
        #   >>> x.is_positive  # query an assumption
        #   >>> x._assumptions
        #   {'finite': True, 'infinite': False, 'commutative': True, 'positive': None}
        #   >>> x._assumptions0
        #   {'finite': True, 'infinite': False, 'commutative': True}
        #   >>> x._assumptions_orig
        #   {'finite': True}
        #
        # Two symbols with the same name are equal if their _assumptions0 are
        # the same. Arguably it should be _assumptions_orig that is being
        # compared because that is more transparent to the user (it is
        # what was passed to the constructor modulo changes made by _sanitize).

        return obj

    @staticmethod
    @cacheit
    def __xnew_cached_(cls, name, **assumptions):  # symbols are always cached
        return Symbol.__xnew__(cls, name, **assumptions)

    def __getnewargs_ex__(self):
        return ((self.name,), self._assumptions_orig)

    # NOTE: __setstate__ is not needed for pickles created by __getnewargs_ex__
    # but was used before Symbol was changed to use __getnewargs_ex__ in v1.9.
    # Pickles created in previous SymPy versions will still need __setstate__
    # so that they can be unpickled in SymPy > v1.9.

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def _hashable_content(self):
        return (self.name,) + self._assumptions0

    def _eval_subs(self, old, new):
        if old.is_Pow:
            from sympy.core.power import Pow
            return Pow(self, S.One, evaluate=False)._eval_subs(old, new)

    def _eval_refine(self, assumptions):
        return self

    @property
    def assumptions0(self):
        return dict(self._assumptions0)

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (self.name,)), S.One.sort_key(), S.One

    def as_dummy(self):
        # only put commutativity in explicitly if it is False
        return Dummy(self.name) if self.is_commutative is not False \
            else Dummy(self.name, commutative=self.is_commutative)

    def as_real_imag(self, deep=True, **hints):
        if hints.get('ignore') == self:
            return None
        else:
            from sympy.functions.elementary.complexes import im, re
            return (re(self), im(self))

    def is_constant(self, *wrt, **flags):
        if not wrt:
            return False
        return self not in wrt

    @property
    def free_symbols(self):
        return {self}

    binary_symbols = free_symbols  # in this case, not always

    def as_set(self):
        return S.UniversalSet


class Dummy(Symbol):
    """Dummy symbols are each unique, even if they have the same name:

    Examples
    ========

    >>> from sympy import Dummy
    >>> Dummy("x") == Dummy("x")
    False

    If a name is not supplied then a string value of an internal count will be
    used. This is useful when a temporary variable is needed and the name
    of the variable used in the expression is not important.

    >>> Dummy() #doctest: +SKIP
    _Dummy_10

    """

    # In the rare event that a Dummy object needs to be recreated, both the
    # `name` and `dummy_index` should be passed.  This is used by `srepr` for
    # example:
    # >>> d1 = Dummy()
    # >>> d2 = eval(srepr(d1))
    # >>> d2 == d1
    # True
    #
    # If a new session is started between `srepr` and `eval`, there is a very
    # small chance that `d2` will be equal to a previously-created Dummy.

    _count = 0
    _prng = random.Random()
    _base_dummy_index = _prng.randint(10**6, 9*10**6)

    __slots__ = ('dummy_index',)

    is_Dummy = True

    def __new__(cls, name=None, dummy_index=None, **assumptions):
        if dummy_index is not None:
            assert name is not None, "If you specify a dummy_index, you must also provide a name"

        if name is None:
            name = "Dummy_" + str(Dummy._count)

        if dummy_index is None:
            dummy_index = Dummy._base_dummy_index + Dummy._count
            Dummy._count += 1

        cls._sanitize(assumptions, cls)
        obj = Symbol.__xnew__(cls, name, **assumptions)

        obj.dummy_index = dummy_index

        return obj

    def __getnewargs_ex__(self):
        return ((self.name, self.dummy_index), self._assumptions_orig)

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (
            2, (self.name, self.dummy_index)), S.One.sort_key(), S.One

    def _hashable_content(self):
        return Symbol._hashable_content(self) + (self.dummy_index,)


class Wild(Symbol):
    """
    A Wild symbol matches anything, or anything
    without whatever is explicitly excluded.

    Parameters
    ==========

    name : str
        Name of the Wild instance.

    exclude : iterable, optional
        Instances in ``exclude`` will not be matched.

    properties : iterable of functions, optional
        Functions, each taking an expressions as input
        and returns a ``bool``. All functions in ``properties``
        need to return ``True`` in order for the Wild instance
        to match the expression.

    Examples
    ========

    >>> from sympy import Wild, WildFunction, cos, pi
    >>> from sympy.abc import x, y, z
    >>> a = Wild('a')
    >>> x.match(a)
    {a_: x}
    >>> pi.match(a)
    {a_: pi}
    >>> (3*x**2).match(a*x)
    {a_: 3*x}
    >>> cos(x).match(a)
    {a_: cos(x)}
    >>> b = Wild('b', exclude=[x])
    >>> (3*x**2).match(b*x)
    >>> b.match(a)
    {a_: b_}
    >>> A = WildFunction('A')
    >>> A.match(a)
    {a_: A_}

    Tips
    ====

    When using Wild, be sure to use the exclude
    keyword to make the pattern more precise.
    Without the exclude pattern, you may get matches
    that are technically correct, but not what you
    wanted. For example, using the above without
    exclude:

    >>> from sympy import symbols
    >>> a, b = symbols('a b', cls=Wild)
    >>> (2 + 3*y).match(a*x + b*y)
    {a_: 2/x, b_: 3}

    This is technically correct, because
    (2/x)*x + 3*y == 2 + 3*y, but you probably
    wanted it to not match at all. The issue is that
    you really did not want a and b to include x and y,
    and the exclude parameter lets you specify exactly
    this.  With the exclude parameter, the pattern will
    not match.

    >>> a = Wild('a', exclude=[x, y])
    >>> b = Wild('b', exclude=[x, y])
    >>> (2 + 3*y).match(a*x + b*y)

    Exclude also helps remove ambiguity from matches.

    >>> E = 2*x**3*y*z
    >>> a, b = symbols('a b', cls=Wild)
    >>> E.match(a*b)
    {a_: 2*y*z, b_: x**3}
    >>> a = Wild('a', exclude=[x, y])
    >>> E.match(a*b)
    {a_: z, b_: 2*x**3*y}
    >>> a = Wild('a', exclude=[x, y, z])
    >>> E.match(a*b)
    {a_: 2, b_: x**3*y*z}

    Wild also accepts a ``properties`` parameter:

    >>> a = Wild('a', properties=[lambda k: k.is_Integer])
    >>> E.match(a*b)
    {a_: 2, b_: x**3*y*z}

    """
    is_Wild = True

    __slots__ = ('exclude', 'properties')

    def __new__(cls, name, exclude=(), properties=(), **assumptions):
        exclude = tuple([sympify(x) for x in exclude])
        properties = tuple(properties)
        cls._sanitize(assumptions, cls)
        return Wild.__xnew__(cls, name, exclude, properties, **assumptions)

    def __getnewargs__(self):
        return (self.name, self.exclude, self.properties)

    @staticmethod
    @cacheit
    def __xnew__(cls, name, exclude, properties, **assumptions):
        obj = Symbol.__xnew__(cls, name, **assumptions)
        obj.exclude = exclude
        obj.properties = properties
        return obj

    def _hashable_content(self):
        return super()._hashable_content() + (self.exclude, self.properties)

    # TODO add check against another Wild
    def matches(self, expr, repl_dict=None, old=False):
        if any(expr.has(x) for x in self.exclude):
            return None
        if not all(f(expr) for f in self.properties):
            return None
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()
        repl_dict[self] = expr
        return repl_dict


_range = _re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')


def symbols(names, *, cls=Symbol, **args) -> Any:
    r"""
    Transform strings into instances of :class:`Symbol` class.

    :func:`symbols` function returns a sequence of symbols with names taken
    from ``names`` argument, which can be a comma or whitespace delimited
    string, or a sequence of strings::

        >>> from sympy import symbols, Function

        >>> x, y, z = symbols('x,y,z')
        >>> a, b, c = symbols('a b c')

    The type of output is dependent on the properties of input arguments::

        >>> symbols('x')
        x
        >>> symbols('x,')
        (x,)
        >>> symbols('x,y')
        (x, y)
        >>> symbols(('a', 'b', 'c'))
        (a, b, c)
        >>> symbols(['a', 'b', 'c'])
        [a, b, c]
        >>> symbols({'a', 'b', 'c'})
        {a, b, c}

    If an iterable container is needed for a single symbol, set the ``seq``
    argument to ``True`` or terminate the symbol name with a comma::

        >>> symbols('x', seq=True)
        (x,)

    To reduce typing, range syntax is supported to create indexed symbols.
    Ranges are indicated by a colon and the type of range is determined by
    the character to the right of the colon. If the character is a digit
    then all contiguous digits to the left are taken as the nonnegative
    starting value (or 0 if there is no digit left of the colon) and all
    contiguous digits to the right are taken as 1 greater than the ending
    value::

        >>> symbols('x:10')
        (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

        >>> symbols('x5:10')
        (x5, x6, x7, x8, x9)
        >>> symbols('x5(:2)')
        (x50, x51)

        >>> symbols('x5:10,y:5')
        (x5, x6, x7, x8, x9, y0, y1, y2, y3, y4)

        >>> symbols(('x5:10', 'y:5'))
        ((x5, x6, x7, x8, x9), (y0, y1, y2, y3, y4))

    If the character to the right of the colon is a letter, then the single
    letter to the left (or 'a' if there is none) is taken as the start
    and all characters in the lexicographic range *through* the letter to
    the right are used as the range::

        >>> symbols('x:z')
        (x, y, z)
        >>> symbols('x:c')  # null range
        ()
        >>> symbols('x(:c)')
        (xa, xb, xc)

        >>> symbols(':c')
        (a, b, c)

        >>> symbols('a:d, x:z')
        (a, b, c, d, x, y, z)

        >>> symbols(('a:d', 'x:z'))
        ((a, b, c, d), (x, y, z))

    Multiple ranges are supported; contiguous numerical ranges should be
    separated by parentheses to disambiguate the ending number of one
    range from the starting number of the next::

        >>> symbols('x:2(1:3)')
        (x01, x02, x11, x12)
        >>> symbols(':3:2')  # parsing is from left to right
        (00, 01, 10, 11, 20, 21)

    Only one pair of parentheses surrounding ranges are removed, so to
    include parentheses around ranges, double them. And to include spaces,
    commas, or colons, escape them with a backslash::

        >>> symbols('x((a:b))')
        (x(a), x(b))
        >>> symbols(r'x(:1\,:2)')  # or r'x((:1)\,(:2))'
        (x(0,0), x(0,1))

    All newly created symbols have assumptions set according to ``args``::

        >>> a = symbols('a', integer=True)
        >>> a.is_integer
        True

        >>> x, y, z = symbols('x,y,z', real=True)
        >>> x.is_real and y.is_real and z.is_real
        True

    Despite its name, :func:`symbols` can create symbol-like objects like
    instances of Function or Wild classes. To achieve this, set ``cls``
    keyword argument to the desired type::

        >>> symbols('f,g,h', cls=Function)
        (f, g, h)

        >>> type(_[0])
        <class 'sympy.core.function.UndefinedFunction'>

    """
    result = []

    if isinstance(names, str):
        marker = 0
        splitters = r'\,', r'\:', r'\ '
        literals: list[tuple[str, str]] = []
        for splitter in splitters:
            if splitter in names:
                while chr(marker) in names:
                    marker += 1
                lit_char = chr(marker)
                marker += 1
                names = names.replace(splitter, lit_char)
                literals.append((lit_char, splitter[1:]))
        def literal(s):
            if literals:
                for c, l in literals:
                    s = s.replace(c, l)
            return s

        names = names.strip()
        as_seq = names.endswith(',')
        if as_seq:
            names = names[:-1].rstrip()
        if not names:
            raise ValueError('no symbols given')

        # split on commas
        names = [n.strip() for n in names.split(',')]
        if not all(n for n in names):
            raise ValueError('missing symbol between commas')
        # split on spaces
        for i in range(len(names) - 1, -1, -1):
            names[i: i + 1] = names[i].split()

        seq = args.pop('seq', as_seq)

        for name in names:
            if not name:
                raise ValueError('missing symbol')

            if ':' not in name:
                symbol = cls(literal(name), **args)
                result.append(symbol)
                continue

            split: list[str] = _range.split(name)
            split_list: list[list[str]] = []
            # remove 1 layer of bounding parentheses around ranges
            for i in range(len(split) - 1):
                if i and ':' in split[i] and split[i] != ':' and \
                        split[i - 1].endswith('(') and \
                        split[i + 1].startswith(')'):
                    split[i - 1] = split[i - 1][:-1]
                    split[i + 1] = split[i + 1][1:]
            for s in split:
                if ':' in s:
                    if s.endswith(':'):
                        raise ValueError('missing end range')
                    a, b = s.split(':')
                    if b[-1] in string.digits:
                        a_i = 0 if not a else int(a)
                        b_i = int(b)
                        split_list.append([str(c) for c in range(a_i, b_i)])
                    else:
                        a = a or 'a'
                        split_list.append([string.ascii_letters[c] for c in range(
                            string.ascii_letters.index(a),
                            string.ascii_letters.index(b) + 1)])  # inclusive
                    if not split_list[-1]:
                        break
                else:
                    split_list.append([s])
            else:
                seq = True
                if len(split_list) == 1:
                    names = split_list[0]
                else:
                    names = [''.join(s) for s in product(*split_list)]
                if literals:
                    result.extend([cls(literal(s), **args) for s in names])
                else:
                    result.extend([cls(s, **args) for s in names])

        if not seq and len(result) <= 1:
            if not result:
                return ()
            return result[0]

        return tuple(result)
    else:
        for name in names:
            result.append(symbols(name, cls=cls, **args))

        return type(names)(result)


def var(names, **args):
    """
    Create symbols and inject them into the global namespace.

    Explanation
    ===========

    This calls :func:`symbols` with the same arguments and puts the results
    into the *global* namespace. It's recommended not to use :func:`var` in
    library code, where :func:`symbols` has to be used::

    Examples
    ========

    >>> from sympy import var

    >>> var('x')
    x
    >>> x # noqa: F821
    x

    >>> var('a,ab,abc')
    (a, ab, abc)
    >>> abc # noqa: F821
    abc

    >>> var('x,y', real=True)
    (x, y)
    >>> x.is_real and y.is_real # noqa: F821
    True

    See :func:`symbols` documentation for more details on what kinds of
    arguments can be passed to :func:`var`.

    """
    def traverse(symbols, frame):
        """Recursively inject symbols to the global namespace. """
        for symbol in symbols:
            if isinstance(symbol, Basic):
                frame.f_globals[symbol.name] = symbol
            elif isinstance(symbol, FunctionClass):
                frame.f_globals[symbol.__name__] = symbol
            else:
                traverse(symbol, frame)

    from inspect import currentframe
    frame = currentframe().f_back

    try:
        syms = symbols(names, **args)

        if syms is not None:
            if isinstance(syms, Basic):
                frame.f_globals[syms.name] = syms
            elif isinstance(syms, FunctionClass):
                frame.f_globals[syms.__name__] = syms
            else:
                traverse(syms, frame)
    finally:
        del frame  # break cyclic dependencies as stated in inspect docs

    return syms

def disambiguate(*iter):
    """
    Return a Tuple containing the passed expressions with symbols
    that appear the same when printed replaced with numerically
    subscripted symbols, and all Dummy symbols replaced with Symbols.

    Parameters
    ==========

    iter: list of symbols or expressions.

    Examples
    ========

    >>> from sympy.core.symbol import disambiguate
    >>> from sympy import Dummy, Symbol, Tuple
    >>> from sympy.abc import y

    >>> tup = Symbol('_x'), Dummy('x'), Dummy('x')
    >>> disambiguate(*tup)
    (x_2, x, x_1)

    >>> eqs = Tuple(Symbol('x')/y, Dummy('x')/y)
    >>> disambiguate(*eqs)
    (x_1/y, x/y)

    >>> ix = Symbol('x', integer=True)
    >>> vx = Symbol('x')
    >>> disambiguate(vx + ix)
    (x + x_1,)

    To make your own mapping of symbols to use, pass only the free symbols
    of the expressions and create a dictionary:

    >>> free = eqs.free_symbols
    >>> mapping = dict(zip(free, disambiguate(*free)))
    >>> eqs.xreplace(mapping)
    (x_1/y, x/y)

    """
    new_iter = Tuple(*iter)
    key = lambda x:tuple(sorted(x.assumptions0.items()))
    syms = ordered(new_iter.free_symbols, keys=key)
    mapping = {}
    for s in syms:
        mapping.setdefault(str(s).lstrip('_'), []).append(s)
    reps = {}
    for k in mapping:
        # the first or only symbol doesn't get subscripted but make
        # sure that it's a Symbol, not a Dummy
        mapk0 = Symbol("%s" % (k), **mapping[k][0].assumptions0)
        if mapping[k][0] != mapk0:
            reps[mapping[k][0]] = mapk0
        # the others get subscripts (and are made into Symbols)
        skip = 0
        for i in range(1, len(mapping[k])):
            while True:
                name = "%s_%i" % (k, i + skip)
                if name not in mapping:
                    break
                skip += 1
            ki = mapping[k][i]
            reps[ki] = Symbol(name, **ki.assumptions0)
    return new_iter.xreplace(reps)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\reshape\reshape.py ===
from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    cast,
)
import warnings

import numpy as np

import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    find_common_type,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_integer,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna

import pandas.core.algorithms as algos
from pandas.core.algorithms import (
    factorize,
    unique,
)
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    RangeIndex,
)
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
    compress_group_index,
    decons_obs_group_ids,
    get_compressed_ids,
    get_group_index,
    get_group_index_sorter,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Level,
        npt,
    )

    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.frozen import FrozenList


class _Unstacker:
    """
    Helper class to unstack data / pivot with multi-level index

    Parameters
    ----------
    index : MultiIndex
    level : int or str, default last level
        Level to "unstack". Accepts a name for the level.
    fill_value : scalar, optional
        Default value to fill in missing values if subgroups do not have the
        same set of labels. By default, missing values will be replaced with
        the default fill value for that data type, NaN for float, NaT for
        datetimelike, etc. For integer types, by default data will converted to
        float and missing values will be set to NaN.
    constructor : object
        Pandas ``DataFrame`` or subclass used to create unstacked
        response.  If None, DataFrame will be used.

    Examples
    --------
    >>> index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
    ...                                    ('two', 'a'), ('two', 'b')])
    >>> s = pd.Series(np.arange(1, 5, dtype=np.int64), index=index)
    >>> s
    one  a    1
         b    2
    two  a    3
         b    4
    dtype: int64

    >>> s.unstack(level=-1)
         a  b
    one  1  2
    two  3  4

    >>> s.unstack(level=0)
       one  two
    a    1    3
    b    2    4

    Returns
    -------
    unstacked : DataFrame
    """

    def __init__(
        self, index: MultiIndex, level: Level, constructor, sort: bool = True
    ) -> None:
        self.constructor = constructor
        self.sort = sort

        self.index = index.remove_unused_levels()

        self.level = self.index._get_level_number(level)

        # when index includes `nan`, need to lift levels/strides by 1
        self.lift = 1 if -1 in self.index.codes[self.level] else 0

        # Note: the "pop" below alters these in-place.
        self.new_index_levels = list(self.index.levels)
        self.new_index_names = list(self.index.names)

        self.removed_name = self.new_index_names.pop(self.level)
        self.removed_level = self.new_index_levels.pop(self.level)
        self.removed_level_full = index.levels[self.level]
        if not self.sort:
            unique_codes = unique(self.index.codes[self.level])
            self.removed_level = self.removed_level.take(unique_codes)
            self.removed_level_full = self.removed_level_full.take(unique_codes)

        # Bug fix GH 20601
        # If the data frame is too big, the number of unique index combination
        # will cause int32 overflow on windows environments.
        # We want to check and raise an warning before this happens
        num_rows = np.max([index_level.size for index_level in self.new_index_levels])
        num_columns = self.removed_level.size

        # GH20601: This forces an overflow if the number of cells is too high.
        num_cells = num_rows * num_columns

        # GH 26314: Previous ValueError raised was too restrictive for many users.
        if num_cells > np.iinfo(np.int32).max:
            warnings.warn(
                f"The following operation may generate {num_cells} cells "
                f"in the resulting pandas object.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

        self._make_selectors()

    @cache_readonly
    def _indexer_and_to_sort(
        self,
    ) -> tuple[
        npt.NDArray[np.intp],
        list[np.ndarray],  # each has _some_ signed integer dtype
    ]:
        v = self.level

        codes = list(self.index.codes)
        levs = list(self.index.levels)
        to_sort = codes[:v] + codes[v + 1 :] + [codes[v]]
        sizes = tuple(len(x) for x in levs[:v] + levs[v + 1 :] + [levs[v]])

        comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
        ngroups = len(obs_ids)

        indexer = get_group_index_sorter(comp_index, ngroups)
        return indexer, to_sort

    @cache_readonly
    def sorted_labels(self) -> list[np.ndarray]:
        indexer, to_sort = self._indexer_and_to_sort
        if self.sort:
            return [line.take(indexer) for line in to_sort]
        return to_sort

    def _make_sorted_values(self, values: np.ndarray) -> np.ndarray:
        if self.sort:
            indexer, _ = self._indexer_and_to_sort

            sorted_values = algos.take_nd(values, indexer, axis=0)
            return sorted_values
        return values

    def _make_selectors(self):
        new_levels = self.new_index_levels

        # make the mask
        remaining_labels = self.sorted_labels[:-1]
        level_sizes = tuple(len(x) for x in new_levels)

        comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
        ngroups = len(obs_ids)

        comp_index = ensure_platform_int(comp_index)
        stride = self.index.levshape[self.level] + self.lift
        self.full_shape = ngroups, stride

        selector = self.sorted_labels[-1] + stride * comp_index + self.lift
        mask = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)

        if mask.sum() < len(self.index):
            raise ValueError("Index contains duplicate entries, cannot reshape")

        self.group_index = comp_index
        self.mask = mask
        if self.sort:
            self.compressor = comp_index.searchsorted(np.arange(ngroups))
        else:
            self.compressor = np.sort(np.unique(comp_index, return_index=True)[1])

    @cache_readonly
    def mask_all(self) -> bool:
        return bool(self.mask.all())

    @cache_readonly
    def arange_result(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
        # We cache this for reuse in ExtensionBlock._unstack
        dummy_arr = np.arange(len(self.index), dtype=np.intp)
        new_values, mask = self.get_new_values(dummy_arr, fill_value=-1)
        return new_values, mask.any(0)
        # TODO: in all tests we have mask.any(0).all(); can we rely on that?

    def get_result(self, values, value_columns, fill_value) -> DataFrame:
        if values.ndim == 1:
            values = values[:, np.newaxis]

        if value_columns is None and values.shape[1] != 1:  # pragma: no cover
            raise ValueError("must pass column labels for multi-column data")

        values, _ = self.get_new_values(values, fill_value)
        columns = self.get_new_columns(value_columns)
        index = self.new_index

        return self.constructor(
            values, index=index, columns=columns, dtype=values.dtype
        )

    def get_new_values(self, values, fill_value=None):
        if values.ndim == 1:
            values = values[:, np.newaxis]

        sorted_values = self._make_sorted_values(values)

        # place the values
        length, width = self.full_shape
        stride = values.shape[1]
        result_width = width * stride
        result_shape = (length, result_width)
        mask = self.mask
        mask_all = self.mask_all

        # we can simply reshape if we don't have a mask
        if mask_all and len(values):
            # TODO: Under what circumstances can we rely on sorted_values
            #  matching values?  When that holds, we can slice instead
            #  of take (in particular for EAs)
            new_values = (
                sorted_values.reshape(length, width, stride)
                .swapaxes(1, 2)
                .reshape(result_shape)
            )
            new_mask = np.ones(result_shape, dtype=bool)
            return new_values, new_mask

        dtype = values.dtype

        # if our mask is all True, then we can use our existing dtype
        if mask_all:
            dtype = values.dtype
            new_values = np.empty(result_shape, dtype=dtype)
        else:
            if isinstance(dtype, ExtensionDtype):
                # GH#41875
                # We are assuming that fill_value can be held by this dtype,
                #  unlike the non-EA case that promotes.
                cls = dtype.construct_array_type()
                new_values = cls._empty(result_shape, dtype=dtype)
                new_values[:] = fill_value
            else:
                dtype, fill_value = maybe_promote(dtype, fill_value)
                new_values = np.empty(result_shape, dtype=dtype)
                new_values.fill(fill_value)

        name = dtype.name
        new_mask = np.zeros(result_shape, dtype=bool)

        # we need to convert to a basic dtype
        # and possibly coerce an input to our output dtype
        # e.g. ints -> floats
        if needs_i8_conversion(values.dtype):
            sorted_values = sorted_values.view("i8")
            new_values = new_values.view("i8")
        else:
            sorted_values = sorted_values.astype(name, copy=False)

        # fill in our values & mask
        libreshape.unstack(
            sorted_values,
            mask.view("u1"),
            stride,
            length,
            width,
            new_values,
            new_mask.view("u1"),
        )

        # reconstruct dtype if needed
        if needs_i8_conversion(values.dtype):
            # view as datetime64 so we can wrap in DatetimeArray and use
            #  DTA's view method
            new_values = new_values.view("M8[ns]")
            new_values = ensure_wrapped_if_datetimelike(new_values)
            new_values = new_values.view(values.dtype)

        return new_values, new_mask

    def get_new_columns(self, value_columns: Index | None):
        if value_columns is None:
            if self.lift == 0:
                return self.removed_level._rename(name=self.removed_name)

            lev = self.removed_level.insert(0, item=self.removed_level._na_value)
            return lev.rename(self.removed_name)

        stride = len(self.removed_level) + self.lift
        width = len(value_columns)
        propagator = np.repeat(np.arange(width), stride)

        new_levels: FrozenList | list[Index]

        if isinstance(value_columns, MultiIndex):
            # error: Cannot determine type of "__add__"  [has-type]
            new_levels = value_columns.levels + (  # type: ignore[has-type]
                self.removed_level_full,
            )
            new_names = value_columns.names + (self.removed_name,)

            new_codes = [lab.take(propagator) for lab in value_columns.codes]
        else:
            new_levels = [
                value_columns,
                self.removed_level_full,
            ]
            new_names = [value_columns.name, self.removed_name]
            new_codes = [propagator]

        repeater = self._repeater

        # The entire level is then just a repetition of the single chunk:
        new_codes.append(np.tile(repeater, width))
        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    @cache_readonly
    def _repeater(self) -> np.ndarray:
        # The two indices differ only if the unstacked level had unused items:
        if len(self.removed_level_full) != len(self.removed_level):
            # In this case, we remap the new codes to the original level:
            repeater = self.removed_level_full.get_indexer(self.removed_level)
            if self.lift:
                repeater = np.insert(repeater, 0, -1)
        else:
            # Otherwise, we just use each level item exactly once:
            stride = len(self.removed_level) + self.lift
            repeater = np.arange(stride) - self.lift

        return repeater

    @cache_readonly
    def new_index(self) -> MultiIndex:
        # Does not depend on values or value_columns
        result_codes = [lab.take(self.compressor) for lab in self.sorted_labels[:-1]]

        # construct the new index
        if len(self.new_index_levels) == 1:
            level, level_codes = self.new_index_levels[0], result_codes[0]
            if (level_codes == -1).any():
                level = level.insert(len(level), level._na_value)
            return level.take(level_codes).rename(self.new_index_names[0])

        return MultiIndex(
            levels=self.new_index_levels,
            codes=result_codes,
            names=self.new_index_names,
            verify_integrity=False,
        )


def _unstack_multiple(
    data: Series | DataFrame, clocs, fill_value=None, sort: bool = True
):
    if len(clocs) == 0:
        return data

    # NOTE: This doesn't deal with hierarchical columns yet

    index = data.index
    index = cast(MultiIndex, index)  # caller is responsible for checking

    # GH 19966 Make sure if MultiIndexed index has tuple name, they will be
    # recognised as a whole
    if clocs in index.names:
        clocs = [clocs]
    clocs = [index._get_level_number(i) for i in clocs]

    rlocs = [i for i in range(index.nlevels) if i not in clocs]

    clevels = [index.levels[i] for i in clocs]
    ccodes = [index.codes[i] for i in clocs]
    cnames = [index.names[i] for i in clocs]
    rlevels = [index.levels[i] for i in rlocs]
    rcodes = [index.codes[i] for i in rlocs]
    rnames = [index.names[i] for i in rlocs]

    shape = tuple(len(x) for x in clevels)
    group_index = get_group_index(ccodes, shape, sort=False, xnull=False)

    comp_ids, obs_ids = compress_group_index(group_index, sort=False)
    recons_codes = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes, xnull=False)

    if not rlocs:
        # Everything is in clocs, so the dummy df has a regular index
        dummy_index = Index(obs_ids, name="__placeholder__")
    else:
        dummy_index = MultiIndex(
            levels=rlevels + [obs_ids],
            codes=rcodes + [comp_ids],
            names=rnames + ["__placeholder__"],
            verify_integrity=False,
        )

    if isinstance(data, Series):
        dummy = data.copy()
        dummy.index = dummy_index

        unstacked = dummy.unstack("__placeholder__", fill_value=fill_value, sort=sort)
        new_levels = clevels
        new_names = cnames
        new_codes = recons_codes
    else:
        if isinstance(data.columns, MultiIndex):
            result = data
            while clocs:
                val = clocs.pop(0)
                result = result.unstack(val, fill_value=fill_value, sort=sort)
                clocs = [v if v < val else v - 1 for v in clocs]

            return result

        # GH#42579 deep=False to avoid consolidating
        dummy_df = data.copy(deep=False)
        dummy_df.index = dummy_index

        unstacked = dummy_df.unstack(
            "__placeholder__", fill_value=fill_value, sort=sort
        )
        if isinstance(unstacked, Series):
            unstcols = unstacked.index
        else:
            unstcols = unstacked.columns
        assert isinstance(unstcols, MultiIndex)  # for mypy
        new_levels = [unstcols.levels[0]] + clevels
        new_names = [data.columns.name] + cnames

        new_codes = [unstcols.codes[0]]
        new_codes.extend(rec.take(unstcols.codes[-1]) for rec in recons_codes)

    new_columns = MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )

    if isinstance(unstacked, Series):
        unstacked.index = new_columns
    else:
        unstacked.columns = new_columns

    return unstacked


def unstack(obj: Series | DataFrame, level, fill_value=None, sort: bool = True):
    if isinstance(level, (tuple, list)):
        if len(level) != 1:
            # _unstack_multiple only handles MultiIndexes,
            # and isn't needed for a single level
            return _unstack_multiple(obj, level, fill_value=fill_value, sort=sort)
        else:
            level = level[0]

    if not is_integer(level) and not level == "__placeholder__":
        # check if level is valid in case of regular index
        obj.index._get_level_number(level)

    if isinstance(obj, DataFrame):
        if isinstance(obj.index, MultiIndex):
            return _unstack_frame(obj, level, fill_value=fill_value, sort=sort)
        else:
            return obj.T.stack(future_stack=True)
    elif not isinstance(obj.index, MultiIndex):
        # GH 36113
        # Give nicer error messages when unstack a Series whose
        # Index is not a MultiIndex.
        raise ValueError(
            f"index must be a MultiIndex to unstack, {type(obj.index)} was passed"
        )
    else:
        if is_1d_only_ea_dtype(obj.dtype):
            return _unstack_extension_series(obj, level, fill_value, sort=sort)
        unstacker = _Unstacker(
            obj.index, level=level, constructor=obj._constructor_expanddim, sort=sort
        )
        return unstacker.get_result(
            obj._values, value_columns=None, fill_value=fill_value
        )


def _unstack_frame(
    obj: DataFrame, level, fill_value=None, sort: bool = True
) -> DataFrame:
    assert isinstance(obj.index, MultiIndex)  # checked by caller
    unstacker = _Unstacker(
        obj.index, level=level, constructor=obj._constructor, sort=sort
    )

    if not obj._can_fast_transpose:
        mgr = obj._mgr.unstack(unstacker, fill_value=fill_value)
        return obj._constructor_from_mgr(mgr, axes=mgr.axes)
    else:
        return unstacker.get_result(
            obj._values, value_columns=obj.columns, fill_value=fill_value
        )


def _unstack_extension_series(
    series: Series, level, fill_value, sort: bool
) -> DataFrame:
    """
    Unstack an ExtensionArray-backed Series.

    The ExtensionDtype is preserved.

    Parameters
    ----------
    series : Series
        A Series with an ExtensionArray for values
    level : Any
        The level name or number.
    fill_value : Any
        The user-level (not physical storage) fill value to use for
        missing values introduced by the reshape. Passed to
        ``series.values.take``.
    sort : bool
        Whether to sort the resulting MuliIndex levels

    Returns
    -------
    DataFrame
        Each column of the DataFrame will have the same dtype as
        the input Series.
    """
    # Defer to the logic in ExtensionBlock._unstack
    df = series.to_frame()
    result = df.unstack(level=level, fill_value=fill_value, sort=sort)

    # equiv: result.droplevel(level=0, axis=1)
    #  but this avoids an extra copy
    result.columns = result.columns._drop_level_numbers([0])
    return result


def stack(frame: DataFrame, level=-1, dropna: bool = True, sort: bool = True):
    """
    Convert DataFrame to Series with multi-level Index. Columns become the
    second level of the resulting hierarchical index

    Returns
    -------
    stacked : Series or DataFrame
    """

    def stack_factorize(index):
        if index.is_unique:
            return index, np.arange(len(index))
        codes, categories = factorize_from_iterable(index)
        return categories, codes

    N, K = frame.shape

    # Will also convert negative level numbers and check if out of bounds.
    level_num = frame.columns._get_level_number(level)

    if isinstance(frame.columns, MultiIndex):
        return _stack_multi_columns(
            frame, level_num=level_num, dropna=dropna, sort=sort
        )
    elif isinstance(frame.index, MultiIndex):
        new_levels = list(frame.index.levels)
        new_codes = [lab.repeat(K) for lab in frame.index.codes]

        clev, clab = stack_factorize(frame.columns)
        new_levels.append(clev)
        new_codes.append(np.tile(clab, N).ravel())

        new_names = list(frame.index.names)
        new_names.append(frame.columns.name)
        new_index = MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )
    else:
        levels, (ilab, clab) = zip(*map(stack_factorize, (frame.index, frame.columns)))
        codes = ilab.repeat(K), np.tile(clab, N).ravel()
        new_index = MultiIndex(
            levels=levels,
            codes=codes,
            names=[frame.index.name, frame.columns.name],
            verify_integrity=False,
        )

    new_values: ArrayLike
    if not frame.empty and frame._is_homogeneous_type:
        # For homogeneous EAs, frame._values will coerce to object. So
        # we concatenate instead.
        dtypes = list(frame.dtypes._values)
        dtype = dtypes[0]

        if isinstance(dtype, ExtensionDtype):
            arr = dtype.construct_array_type()
            new_values = arr._concat_same_type(
                [col._values for _, col in frame.items()]
            )
            new_values = _reorder_for_extension_array_stack(new_values, N, K)
        else:
            # homogeneous, non-EA
            new_values = frame._values.ravel()

    else:
        # non-homogeneous
        new_values = frame._values.ravel()

    if dropna:
        mask = notna(new_values)
        new_values = new_values[mask]
        new_index = new_index[mask]

    return frame._constructor_sliced(new_values, index=new_index)


def stack_multiple(frame: DataFrame, level, dropna: bool = True, sort: bool = True):
    # If all passed levels match up to column names, no
    # ambiguity about what to do
    if all(lev in frame.columns.names for lev in level):
        result = frame
        for lev in level:
            result = stack(result, lev, dropna=dropna, sort=sort)

    # Otherwise, level numbers may change as each successive level is stacked
    elif all(isinstance(lev, int) for lev in level):
        # As each stack is done, the level numbers decrease, so we need
        #  to account for that when level is a sequence of ints
        result = frame
        # _get_level_number() checks level numbers are in range and converts
        # negative numbers to positive
        level = [frame.columns._get_level_number(lev) for lev in level]

        while level:
            lev = level.pop(0)
            result = stack(result, lev, dropna=dropna, sort=sort)
            # Decrement all level numbers greater than current, as these
            # have now shifted down by one
            level = [v if v <= lev else v - 1 for v in level]

    else:
        raise ValueError(
            "level should contain all level names or all level "
            "numbers, not a mixture of the two."
        )

    return result


def _stack_multi_column_index(columns: MultiIndex) -> MultiIndex:
    """Creates a MultiIndex from the first N-1 levels of this MultiIndex."""
    if len(columns.levels) <= 2:
        return columns.levels[0]._rename(name=columns.names[0])

    levs = [
        [lev[c] if c >= 0 else None for c in codes]
        for lev, codes in zip(columns.levels[:-1], columns.codes[:-1])
    ]

    # Remove duplicate tuples in the MultiIndex.
    tuples = zip(*levs)
    unique_tuples = (key for key, _ in itertools.groupby(tuples))
    new_levs = zip(*unique_tuples)

    # The dtype of each level must be explicitly set to avoid inferring the wrong type.
    # See GH-36991.
    return MultiIndex.from_arrays(
        [
            # Not all indices can accept None values.
            Index(new_lev, dtype=lev.dtype) if None not in new_lev else new_lev
            for new_lev, lev in zip(new_levs, columns.levels)
        ],
        names=columns.names[:-1],
    )


def _stack_multi_columns(
    frame: DataFrame, level_num: int = -1, dropna: bool = True, sort: bool = True
) -> DataFrame:
    def _convert_level_number(level_num: int, columns: Index):
        """
        Logic for converting the level number to something we can safely pass
        to swaplevel.

        If `level_num` matches a column name return the name from
        position `level_num`, otherwise return `level_num`.
        """
        if level_num in columns.names:
            return columns.names[level_num]

        return level_num

    this = frame.copy(deep=False)
    mi_cols = this.columns  # cast(MultiIndex, this.columns)
    assert isinstance(mi_cols, MultiIndex)  # caller is responsible

    # this makes life much simpler
    if level_num != mi_cols.nlevels - 1:
        # roll levels to put selected level at end
        roll_columns = mi_cols
        for i in range(level_num, mi_cols.nlevels - 1):
            # Need to check if the ints conflict with level names
            lev1 = _convert_level_number(i, roll_columns)
            lev2 = _convert_level_number(i + 1, roll_columns)
            roll_columns = roll_columns.swaplevel(lev1, lev2)
        this.columns = mi_cols = roll_columns

    if not mi_cols._is_lexsorted() and sort:
        # Workaround the edge case where 0 is one of the column names,
        # which interferes with trying to sort based on the first
        # level
        level_to_sort = _convert_level_number(0, mi_cols)
        this = this.sort_index(level=level_to_sort, axis=1)
        mi_cols = this.columns

    mi_cols = cast(MultiIndex, mi_cols)
    new_columns = _stack_multi_column_index(mi_cols)

    # time to ravel the values
    new_data = {}
    level_vals = mi_cols.levels[-1]
    level_codes = unique(mi_cols.codes[-1])
    if sort:
        level_codes = np.sort(level_codes)
    level_vals_nan = level_vals.insert(len(level_vals), None)

    level_vals_used = np.take(level_vals_nan, level_codes)
    levsize = len(level_codes)
    drop_cols = []
    for key in new_columns:
        try:
            loc = this.columns.get_loc(key)
        except KeyError:
            drop_cols.append(key)
            continue

        # can make more efficient?
        # we almost always return a slice
        # but if unsorted can get a boolean
        # indexer
        if not isinstance(loc, slice):
            slice_len = len(loc)
        else:
            slice_len = loc.stop - loc.start

        if slice_len != levsize:
            chunk = this.loc[:, this.columns[loc]]
            chunk.columns = level_vals_nan.take(chunk.columns.codes[-1])
            value_slice = chunk.reindex(columns=level_vals_used).values
        else:
            subset = this.iloc[:, loc]
            dtype = find_common_type(subset.dtypes.tolist())
            if isinstance(dtype, ExtensionDtype):
                # TODO(EA2D): won't need special case, can go through .values
                #  paths below (might change to ._values)
                value_slice = dtype.construct_array_type()._concat_same_type(
                    [x._values.astype(dtype, copy=False) for _, x in subset.items()]
                )
                N, K = subset.shape
                idx = np.arange(N * K).reshape(K, N).T.ravel()
                value_slice = value_slice.take(idx)
            else:
                value_slice = subset.values

        if value_slice.ndim > 1:
            # i.e. not extension
            value_slice = value_slice.ravel()

        new_data[key] = value_slice

    if len(drop_cols) > 0:
        new_columns = new_columns.difference(drop_cols)

    N = len(this)

    if isinstance(this.index, MultiIndex):
        new_levels = list(this.index.levels)
        new_names = list(this.index.names)
        new_codes = [lab.repeat(levsize) for lab in this.index.codes]
    else:
        old_codes, old_levels = factorize_from_iterable(this.index)
        new_levels = [old_levels]
        new_codes = [old_codes.repeat(levsize)]
        new_names = [this.index.name]  # something better?

    new_levels.append(level_vals)
    new_codes.append(np.tile(level_codes, N))
    new_names.append(frame.columns.names[level_num])

    new_index = MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )

    result = frame._constructor(new_data, index=new_index, columns=new_columns)

    if frame.columns.nlevels > 1:
        desired_columns = frame.columns._drop_level_numbers([level_num]).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]

    # more efficient way to go about this? can do the whole masking biz but
    # will only save a small amount of time...
    if dropna:
        result = result.dropna(axis=0, how="all")

    return result


def _reorder_for_extension_array_stack(
    arr: ExtensionArray, n_rows: int, n_columns: int
) -> ExtensionArray:
    """
    Re-orders the values when stacking multiple extension-arrays.

    The indirect stacking method used for EAs requires a followup
    take to get the order correct.

    Parameters
    ----------
    arr : ExtensionArray
    n_rows, n_columns : int
        The number of rows and columns in the original DataFrame.

    Returns
    -------
    taken : ExtensionArray
        The original `arr` with elements re-ordered appropriately

    Examples
    --------
    >>> arr = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    >>> _reorder_for_extension_array_stack(arr, 2, 3)
    array(['a', 'c', 'e', 'b', 'd', 'f'], dtype='<U1')

    >>> _reorder_for_extension_array_stack(arr, 3, 2)
    array(['a', 'd', 'b', 'e', 'c', 'f'], dtype='<U1')
    """
    # final take to get the order correct.
    # idx is an indexer like
    # [c0r0, c1r0, c2r0, ...,
    #  c0r1, c1r1, c2r1, ...]
    idx = np.arange(n_rows * n_columns).reshape(n_columns, n_rows).T.ravel()
    return arr.take(idx)


def stack_v3(frame: DataFrame, level: list[int]) -> Series | DataFrame:
    if frame.columns.nunique() != len(frame.columns):
        raise ValueError("Columns with duplicate values are not supported in stack")

    # If we need to drop `level` from columns, it needs to be in descending order
    drop_levnums = sorted(level, reverse=True)
    stack_cols = frame.columns._drop_level_numbers(
        [k for k in range(frame.columns.nlevels) if k not in level][::-1]
    )
    if len(level) > 1:
        # Arrange columns in the order we want to take them, e.g. level=[2, 0, 1]
        sorter = np.argsort(level)
        ordered_stack_cols = stack_cols._reorder_ilevels(sorter)
    else:
        ordered_stack_cols = stack_cols

    stack_cols_unique = stack_cols.unique()
    ordered_stack_cols_unique = ordered_stack_cols.unique()

    # Grab data for each unique index to be stacked
    buf = []
    for idx in stack_cols_unique:
        if len(frame.columns) == 1:
            data = frame.copy()
        else:
            # Take the data from frame corresponding to this idx value
            if len(level) == 1:
                idx = (idx,)
            gen = iter(idx)
            column_indexer = tuple(
                next(gen) if k in level else slice(None)
                for k in range(frame.columns.nlevels)
            )
            data = frame.loc[:, column_indexer]

        if len(level) < frame.columns.nlevels:
            data.columns = data.columns._drop_level_numbers(drop_levnums)
        elif stack_cols.nlevels == 1:
            if data.ndim == 1:
                data.name = 0
            else:
                data.columns = RangeIndex(len(data.columns))
        buf.append(data)

    result: Series | DataFrame
    if len(buf) > 0 and not frame.empty:
        result = concat(buf)
        ratio = len(result) // len(frame)
    else:
        # input is empty
        if len(level) < frame.columns.nlevels:
            # concat column order may be different from dropping the levels
            new_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        else:
            new_columns = [0]
        result = DataFrame(columns=new_columns, dtype=frame._values.dtype)
        ratio = 0

    if len(level) < frame.columns.nlevels:
        # concat column order may be different from dropping the levels
        desired_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]

    # Construct the correct MultiIndex by combining the frame's index and
    # stacked columns.
    index_levels: list | FrozenList
    if isinstance(frame.index, MultiIndex):
        index_levels = frame.index.levels
        index_codes = list(np.tile(frame.index.codes, (1, ratio)))
    else:
        codes, uniques = factorize(frame.index, use_na_sentinel=False)
        index_levels = [uniques]
        index_codes = list(np.tile(codes, (1, ratio)))
    if isinstance(stack_cols, MultiIndex):
        column_levels = ordered_stack_cols.levels
        column_codes = ordered_stack_cols.drop_duplicates().codes
    else:
        column_levels = [ordered_stack_cols.unique()]
        column_codes = [factorize(ordered_stack_cols_unique, use_na_sentinel=False)[0]]
    column_codes = [np.repeat(codes, len(frame)) for codes in column_codes]
    result.index = MultiIndex(
        levels=index_levels + column_levels,
        codes=index_codes + column_codes,
        names=frame.index.names + list(ordered_stack_cols.names),
        verify_integrity=False,
    )

    # sort result, but faster than calling sort_index since we know the order we need
    len_df = len(frame)
    n_uniques = len(ordered_stack_cols_unique)
    indexer = np.arange(n_uniques)
    idxs = np.tile(len_df * indexer, len_df) + np.repeat(np.arange(len_df), n_uniques)
    result = result.take(idxs)

    # Reshape/rename if needed and dropna
    if result.ndim == 2 and frame.columns.nlevels == len(level):
        if len(result.columns) == 0:
            result = Series(index=result.index)
        else:
            result = result.iloc[:, 0]
    if result.ndim == 1:
        result.name = None

    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\assertions.py ===
# testing/assertions.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


from __future__ import annotations

from collections import defaultdict
import contextlib
from copy import copy
from itertools import filterfalse
import re
import sys
import warnings

from . import assertsql
from . import config
from . import engines
from . import mock
from .exclusions import db_spec
from .util import fail
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import types as sqltypes
from .. import util
from ..engine import default
from ..engine import url
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import decorator


def expect_warnings(*messages, **kw):
    """Context manager which expects one or more warnings.

    With no arguments, squelches all SAWarning emitted via
    sqlalchemy.util.warn and sqlalchemy.util.warn_limited.   Otherwise
    pass string expressions that will match selected warnings via regex;
    all non-matching warnings are sent through.

    The expect version **asserts** that the warnings were in fact seen.

    Note that the test suite sets SAWarning warnings to raise exceptions.

    """  # noqa
    return _expect_warnings_sqla_only(sa_exc.SAWarning, messages, **kw)


@contextlib.contextmanager
def expect_warnings_on(db, *messages, **kw):
    """Context manager which expects one or more warnings on specific
    dialects.

    The expect version **asserts** that the warnings were in fact seen.

    """
    spec = db_spec(db)

    if isinstance(db, str) and not spec(config._current):
        yield
    else:
        with expect_warnings(*messages, **kw):
            yield


def emits_warning(*messages):
    """Decorator form of expect_warnings().

    Note that emits_warning does **not** assert that the warnings
    were in fact seen.

    """

    @decorator
    def decorate(fn, *args, **kw):
        with expect_warnings(assert_=False, *messages):
            return fn(*args, **kw)

    return decorate


def expect_deprecated(*messages, **kw):
    return _expect_warnings_sqla_only(
        sa_exc.SADeprecationWarning, messages, **kw
    )


def expect_deprecated_20(*messages, **kw):
    return _expect_warnings_sqla_only(
        sa_exc.Base20DeprecationWarning, messages, **kw
    )


def emits_warning_on(db, *messages):
    """Mark a test as emitting a warning on a specific dialect.

    With no arguments, squelches all SAWarning failures.  Or pass one or more
    strings; these will be matched to the root of the warning description by
    warnings.filterwarnings().

    Note that emits_warning_on does **not** assert that the warnings
    were in fact seen.

    """

    @decorator
    def decorate(fn, *args, **kw):
        with expect_warnings_on(db, assert_=False, *messages):
            return fn(*args, **kw)

    return decorate


def uses_deprecated(*messages):
    """Mark a test as immune from fatal deprecation warnings.

    With no arguments, squelches all SADeprecationWarning failures.
    Or pass one or more strings; these will be matched to the root
    of the warning description by warnings.filterwarnings().

    As a special case, you may pass a function name prefixed with //
    and it will be re-written as needed to match the standard warning
    verbiage emitted by the sqlalchemy.util.deprecated decorator.

    Note that uses_deprecated does **not** assert that the warnings
    were in fact seen.

    """

    @decorator
    def decorate(fn, *args, **kw):
        with expect_deprecated(*messages, assert_=False):
            return fn(*args, **kw)

    return decorate


_FILTERS = None
_SEEN = None
_EXC_CLS = None


def _expect_warnings_sqla_only(
    exc_cls,
    messages,
    regex=True,
    search_msg=False,
    assert_=True,
):
    """SQLAlchemy internal use only _expect_warnings().

    Alembic is using _expect_warnings() directly, and should be updated
    to use this new interface.

    """
    return _expect_warnings(
        exc_cls,
        messages,
        regex=regex,
        search_msg=search_msg,
        assert_=assert_,
        raise_on_any_unexpected=True,
    )


@contextlib.contextmanager
def _expect_warnings(
    exc_cls,
    messages,
    regex=True,
    search_msg=False,
    assert_=True,
    raise_on_any_unexpected=False,
    squelch_other_warnings=False,
):
    global _FILTERS, _SEEN, _EXC_CLS

    if regex or search_msg:
        filters = [re.compile(msg, re.I | re.S) for msg in messages]
    else:
        filters = list(messages)

    if _FILTERS is not None:
        # nested call; update _FILTERS and _SEEN, return.  outer
        # block will assert our messages
        assert _SEEN is not None
        assert _EXC_CLS is not None
        _FILTERS.extend(filters)
        _SEEN.update(filters)
        _EXC_CLS += (exc_cls,)
        yield
    else:
        seen = _SEEN = set(filters)
        _FILTERS = filters
        _EXC_CLS = (exc_cls,)

        if raise_on_any_unexpected:

            def real_warn(msg, *arg, **kw):
                raise AssertionError("Got unexpected warning: %r" % msg)

        else:
            real_warn = warnings.warn

        def our_warn(msg, *arg, **kw):
            if isinstance(msg, _EXC_CLS):
                exception = type(msg)
                msg = str(msg)
            elif arg:
                exception = arg[0]
            else:
                exception = None

            if not exception or not issubclass(exception, _EXC_CLS):
                if not squelch_other_warnings:
                    return real_warn(msg, *arg, **kw)
                else:
                    return

            if not filters and not raise_on_any_unexpected:
                return

            for filter_ in filters:
                if (
                    (search_msg and filter_.search(msg))
                    or (regex and filter_.match(msg))
                    or (not regex and filter_ == msg)
                ):
                    seen.discard(filter_)
                    break
            else:
                if not squelch_other_warnings:
                    real_warn(msg, *arg, **kw)

        with mock.patch("warnings.warn", our_warn):
            try:
                yield
            finally:
                _SEEN = _FILTERS = _EXC_CLS = None

                if assert_:
                    assert not seen, "Warnings were not seen: %s" % ", ".join(
                        "%r" % (s.pattern if regex else s) for s in seen
                    )


def global_cleanup_assertions():
    """Check things that have to be finalized at the end of a test suite.

    Hardcoded at the moment, a modular system can be built here
    to support things like PG prepared transactions, tables all
    dropped, etc.

    """
    _assert_no_stray_pool_connections()


def _assert_no_stray_pool_connections():
    engines.testing_reaper.assert_all_closed()


def int_within_variance(expected, received, variance):
    deviance = int(expected * variance)
    assert (
        abs(received - expected) < deviance
    ), "Given int value %s is not within %d%% of expected value %s" % (
        received,
        variance * 100,
        expected,
    )


def eq_regex(a, b, msg=None, flags=0):
    assert re.match(b, a, flags), msg or "%r !~ %r" % (a, b)


def eq_(a, b, msg=None):
    """Assert a == b, with repr messaging on failure."""
    assert a == b, msg or "%r != %r" % (a, b)


def ne_(a, b, msg=None):
    """Assert a != b, with repr messaging on failure."""
    assert a != b, msg or "%r == %r" % (a, b)


def le_(a, b, msg=None):
    """Assert a <= b, with repr messaging on failure."""
    assert a <= b, msg or "%r != %r" % (a, b)


def is_instance_of(a, b, msg=None):
    assert isinstance(a, b), msg or "%r is not an instance of %r" % (a, b)


def is_none(a, msg=None):
    is_(a, None, msg=msg)


def is_not_none(a, msg=None):
    is_not(a, None, msg=msg)


def is_true(a, msg=None):
    is_(bool(a), True, msg=msg)


def is_false(a, msg=None):
    is_(bool(a), False, msg=msg)


def is_(a, b, msg=None):
    """Assert a is b, with repr messaging on failure."""
    assert a is b, msg or "%r is not %r" % (a, b)


def is_not(a, b, msg=None):
    """Assert a is not b, with repr messaging on failure."""
    assert a is not b, msg or "%r is %r" % (a, b)


# deprecated.  See #5429
is_not_ = is_not


def in_(a, b, msg=None):
    """Assert a in b, with repr messaging on failure."""
    assert a in b, msg or "%r not in %r" % (a, b)


def not_in(a, b, msg=None):
    """Assert a in not b, with repr messaging on failure."""
    assert a not in b, msg or "%r is in %r" % (a, b)


# deprecated.  See #5429
not_in_ = not_in


def startswith_(a, fragment, msg=None):
    """Assert a.startswith(fragment), with repr messaging on failure."""
    assert a.startswith(fragment), msg or "%r does not start with %r" % (
        a,
        fragment,
    )


def eq_ignore_whitespace(a, b, msg=None):
    a = re.sub(r"^\s+?|\n", "", a)
    a = re.sub(r" {2,}", " ", a)
    a = re.sub(r"\t", "", a)
    b = re.sub(r"^\s+?|\n", "", b)
    b = re.sub(r" {2,}", " ", b)
    b = re.sub(r"\t", "", b)

    assert a == b, msg or "%r != %r" % (a, b)


def _assert_proper_exception_context(exception):
    """assert that any exception we're catching does not have a __context__
    without a __cause__, and that __suppress_context__ is never set.

    Python 3 will report nested as exceptions as "during the handling of
    error X, error Y occurred". That's not what we want to do.  we want
    these exceptions in a cause chain.

    """

    if (
        exception.__context__ is not exception.__cause__
        and not exception.__suppress_context__
    ):
        assert False, (
            "Exception %r was correctly raised but did not set a cause, "
            "within context %r as its cause."
            % (exception, exception.__context__)
        )


def assert_raises(except_cls, callable_, *args, **kw):
    return _assert_raises(except_cls, callable_, args, kw, check_context=True)


def assert_raises_context_ok(except_cls, callable_, *args, **kw):
    return _assert_raises(except_cls, callable_, args, kw)


def assert_raises_message(except_cls, msg, callable_, *args, **kwargs):
    return _assert_raises(
        except_cls, callable_, args, kwargs, msg=msg, check_context=True
    )


def assert_warns(except_cls, callable_, *args, **kwargs):
    """legacy adapter function for functions that were previously using
    assert_raises with SAWarning or similar.

    has some workarounds to accommodate the fact that the callable completes
    with this approach rather than stopping at the exception raise.


    """
    with _expect_warnings_sqla_only(except_cls, [".*"]):
        return callable_(*args, **kwargs)


def assert_warns_message(except_cls, msg, callable_, *args, **kwargs):
    """legacy adapter function for functions that were previously using
    assert_raises with SAWarning or similar.

    has some workarounds to accommodate the fact that the callable completes
    with this approach rather than stopping at the exception raise.

    Also uses regex.search() to match the given message to the error string
    rather than regex.match().

    """
    with _expect_warnings_sqla_only(
        except_cls,
        [msg],
        search_msg=True,
        regex=False,
    ):
        return callable_(*args, **kwargs)


def assert_raises_message_context_ok(
    except_cls, msg, callable_, *args, **kwargs
):
    return _assert_raises(except_cls, callable_, args, kwargs, msg=msg)


def _assert_raises(
    except_cls, callable_, args, kwargs, msg=None, check_context=False
):
    with _expect_raises(except_cls, msg, check_context) as ec:
        callable_(*args, **kwargs)
    return ec.error


class _ErrorContainer:
    error = None


@contextlib.contextmanager
def _expect_raises(except_cls, msg=None, check_context=False):
    if (
        isinstance(except_cls, type)
        and issubclass(except_cls, Warning)
        or isinstance(except_cls, Warning)
    ):
        raise TypeError(
            "Use expect_warnings for warnings, not "
            "expect_raises / assert_raises"
        )
    ec = _ErrorContainer()
    if check_context:
        are_we_already_in_a_traceback = sys.exc_info()[0]
    try:
        yield ec
        success = False
    except except_cls as err:
        ec.error = err
        success = True
        if msg is not None:
            # I'm often pdbing here, and "err" above isn't
            # in scope, so assign the string explicitly
            error_as_string = str(err)
            assert re.search(msg, error_as_string, re.UNICODE), "%r !~ %s" % (
                msg,
                error_as_string,
            )
        if check_context and not are_we_already_in_a_traceback:
            _assert_proper_exception_context(err)
        print(str(err).encode("utf-8"))

    # it's generally a good idea to not carry traceback objects outside
    # of the except: block, but in this case especially we seem to have
    # hit some bug in either python 3.10.0b2 or greenlet or both which
    # this seems to fix:
    # https://github.com/python-greenlet/greenlet/issues/242
    del ec

    # assert outside the block so it works for AssertionError too !
    assert success, "Callable did not raise an exception"


def expect_raises(except_cls, check_context=True):
    return _expect_raises(except_cls, check_context=check_context)


def expect_raises_message(except_cls, msg, check_context=True):
    return _expect_raises(except_cls, msg=msg, check_context=check_context)


class AssertsCompiledSQL:
    def assert_compile(
        self,
        clause,
        result,
        params=None,
        checkparams=None,
        for_executemany=False,
        check_literal_execute=None,
        check_post_param=None,
        dialect=None,
        checkpositional=None,
        check_prefetch=None,
        use_default_dialect=False,
        allow_dialect_select=False,
        supports_default_values=True,
        supports_default_metavalue=True,
        literal_binds=False,
        render_postcompile=False,
        schema_translate_map=None,
        render_schema_translate=False,
        default_schema_name=None,
        from_linting=False,
        check_param_order=True,
        use_literal_execute_for_simple_int=False,
    ):
        if use_default_dialect:
            dialect = default.DefaultDialect()
            dialect.supports_default_values = supports_default_values
            dialect.supports_default_metavalue = supports_default_metavalue
        elif allow_dialect_select:
            dialect = None
        else:
            if dialect is None:
                dialect = getattr(self, "__dialect__", None)

            if dialect is None:
                dialect = config.db.dialect
            elif dialect == "default" or dialect == "default_qmark":
                if dialect == "default":
                    dialect = default.DefaultDialect()
                else:
                    dialect = default.DefaultDialect("qmark")
                dialect.supports_default_values = supports_default_values
                dialect.supports_default_metavalue = supports_default_metavalue
            elif dialect == "default_enhanced":
                dialect = default.StrCompileDialect()
            elif isinstance(dialect, str):
                dialect = url.URL.create(dialect).get_dialect()()

        if default_schema_name:
            dialect.default_schema_name = default_schema_name

        kw = {}
        compile_kwargs = {}

        if schema_translate_map:
            kw["schema_translate_map"] = schema_translate_map

        if params is not None:
            kw["column_keys"] = list(params)

        if literal_binds:
            compile_kwargs["literal_binds"] = True

        if render_postcompile:
            compile_kwargs["render_postcompile"] = True

        if use_literal_execute_for_simple_int:
            compile_kwargs["use_literal_execute_for_simple_int"] = True

        if for_executemany:
            kw["for_executemany"] = True

        if render_schema_translate:
            kw["render_schema_translate"] = True

        if from_linting or getattr(self, "assert_from_linting", False):
            kw["linting"] = sql.FROM_LINTING

        from sqlalchemy import orm

        if isinstance(clause, orm.Query):
            stmt = clause._statement_20()
            stmt._label_style = LABEL_STYLE_TABLENAME_PLUS_COL
            clause = stmt

        if compile_kwargs:
            kw["compile_kwargs"] = compile_kwargs

        class DontAccess:
            def __getattribute__(self, key):
                raise NotImplementedError(
                    "compiler accessed .statement; use "
                    "compiler.current_executable"
                )

        class CheckCompilerAccess:
            def __init__(self, test_statement):
                self.test_statement = test_statement
                self._annotations = {}
                self.supports_execution = getattr(
                    test_statement, "supports_execution", False
                )

                if self.supports_execution:
                    self._execution_options = test_statement._execution_options

                    if hasattr(test_statement, "_returning"):
                        self._returning = test_statement._returning
                    if hasattr(test_statement, "_inline"):
                        self._inline = test_statement._inline
                    if hasattr(test_statement, "_return_defaults"):
                        self._return_defaults = test_statement._return_defaults

            @property
            def _variant_mapping(self):
                return self.test_statement._variant_mapping

            def _default_dialect(self):
                return self.test_statement._default_dialect()

            def compile(self, dialect, **kw):
                return self.test_statement.compile.__func__(
                    self, dialect=dialect, **kw
                )

            def _compiler(self, dialect, **kw):
                return self.test_statement._compiler.__func__(
                    self, dialect, **kw
                )

            def _compiler_dispatch(self, compiler, **kwargs):
                if hasattr(compiler, "statement"):
                    with mock.patch.object(
                        compiler, "statement", DontAccess()
                    ):
                        return self.test_statement._compiler_dispatch(
                            compiler, **kwargs
                        )
                else:
                    return self.test_statement._compiler_dispatch(
                        compiler, **kwargs
                    )

        # no construct can assume it's the "top level" construct in all cases
        # as anything can be nested.  ensure constructs don't assume they
        # are the "self.statement" element
        c = CheckCompilerAccess(clause).compile(dialect=dialect, **kw)

        if isinstance(clause, sqltypes.TypeEngine):
            cache_key_no_warnings = clause._static_cache_key
            if cache_key_no_warnings:
                hash(cache_key_no_warnings)
        else:
            cache_key_no_warnings = clause._generate_cache_key()
            if cache_key_no_warnings:
                hash(cache_key_no_warnings[0])

        param_str = repr(getattr(c, "params", {}))
        param_str = param_str.encode("utf-8").decode("ascii", "ignore")
        print(("\nSQL String:\n" + str(c) + param_str).encode("utf-8"))

        cc = re.sub(r"[\n\t]", "", str(c))

        eq_(cc, result, "%r != %r on dialect %r" % (cc, result, dialect))

        if checkparams is not None:
            if render_postcompile:
                expanded_state = c.construct_expanded_state(
                    params, escape_names=False
                )
                eq_(expanded_state.parameters, checkparams)
            else:
                eq_(c.construct_params(params), checkparams)
        if checkpositional is not None:
            if render_postcompile:
                expanded_state = c.construct_expanded_state(
                    params, escape_names=False
                )
                eq_(
                    tuple(
                        [
                            expanded_state.parameters[x]
                            for x in expanded_state.positiontup
                        ]
                    ),
                    checkpositional,
                )
            else:
                p = c.construct_params(params, escape_names=False)
                eq_(tuple([p[x] for x in c.positiontup]), checkpositional)
        if check_prefetch is not None:
            eq_(c.prefetch, check_prefetch)
        if check_literal_execute is not None:
            eq_(
                {
                    c.bind_names[b]: b.effective_value
                    for b in c.literal_execute_params
                },
                check_literal_execute,
            )
        if check_post_param is not None:
            eq_(
                {
                    c.bind_names[b]: b.effective_value
                    for b in c.post_compile_params
                },
                check_post_param,
            )
        if check_param_order and getattr(c, "params", None):

            def get_dialect(paramstyle, positional):
                cp = copy(dialect)
                cp.paramstyle = paramstyle
                cp.positional = positional
                return cp

            pyformat_dialect = get_dialect("pyformat", False)
            pyformat_c = clause.compile(dialect=pyformat_dialect, **kw)
            stmt = re.sub(r"[\n\t]", "", str(pyformat_c))

            qmark_dialect = get_dialect("qmark", True)
            qmark_c = clause.compile(dialect=qmark_dialect, **kw)
            values = list(qmark_c.positiontup)
            escaped = qmark_c.escaped_bind_names

            for post_param in (
                qmark_c.post_compile_params | qmark_c.literal_execute_params
            ):
                name = qmark_c.bind_names[post_param]
                if name in values:
                    values = [v for v in values if v != name]
            positions = []
            pos_by_value = defaultdict(list)
            for v in values:
                try:
                    if v in pos_by_value:
                        start = pos_by_value[v][-1]
                    else:
                        start = 0
                    esc = escaped.get(v, v)
                    pos = stmt.index("%%(%s)s" % (esc,), start) + 2
                    positions.append(pos)
                    pos_by_value[v].append(pos)
                except ValueError:
                    msg = "Expected to find bindparam %r in %r" % (v, stmt)
                    assert False, msg

            ordered = all(
                positions[i - 1] < positions[i]
                for i in range(1, len(positions))
            )

            expected = [v for _, v in sorted(zip(positions, values))]

            msg = (
                "Order of parameters %s does not match the order "
                "in the statement %s. Statement %r" % (values, expected, stmt)
            )

            is_true(ordered, msg)


class ComparesTables:
    def assert_tables_equal(
        self,
        table,
        reflected_table,
        strict_types=False,
        strict_constraints=True,
    ):
        assert len(table.c) == len(reflected_table.c)
        for c, reflected_c in zip(table.c, reflected_table.c):
            eq_(c.name, reflected_c.name)
            assert reflected_c is reflected_table.c[c.name]

            if strict_constraints:
                eq_(c.primary_key, reflected_c.primary_key)
                eq_(c.nullable, reflected_c.nullable)

            if strict_types:
                msg = "Type '%s' doesn't correspond to type '%s'"
                assert isinstance(reflected_c.type, type(c.type)), msg % (
                    reflected_c.type,
                    c.type,
                )
            else:
                self.assert_types_base(reflected_c, c)

            if isinstance(c.type, sqltypes.String):
                eq_(c.type.length, reflected_c.type.length)

            if strict_constraints:
                eq_(
                    {f.column.name for f in c.foreign_keys},
                    {f.column.name for f in reflected_c.foreign_keys},
                )
            if c.server_default:
                assert isinstance(
                    reflected_c.server_default, schema.FetchedValue
                )

        if strict_constraints:
            assert len(table.primary_key) == len(reflected_table.primary_key)
            for c in table.primary_key:
                assert reflected_table.primary_key.columns[c.name] is not None

    def assert_types_base(self, c1, c2):
        assert c1.type._compare_type_affinity(
            c2.type
        ), "On column %r, type '%s' doesn't correspond to type '%s'" % (
            c1.name,
            c1.type,
            c2.type,
        )


class AssertsExecutionResults:
    def assert_result(self, result, class_, *objects):
        result = list(result)
        print(repr(result))
        self.assert_list(result, class_, objects)

    def assert_list(self, result, class_, list_):
        self.assert_(
            len(result) == len(list_),
            "result list is not the same size as test list, "
            + "for class "
            + class_.__name__,
        )
        for i in range(0, len(list_)):
            self.assert_row(class_, result[i], list_[i])

    def assert_row(self, class_, rowobj, desc):
        self.assert_(
            rowobj.__class__ is class_, "item class is not " + repr(class_)
        )
        for key, value in desc.items():
            if isinstance(value, tuple):
                if isinstance(value[1], list):
                    self.assert_list(getattr(rowobj, key), value[0], value[1])
                else:
                    self.assert_row(value[0], getattr(rowobj, key), value[1])
            else:
                self.assert_(
                    getattr(rowobj, key) == value,
                    "attribute %s value %s does not match %s"
                    % (key, getattr(rowobj, key), value),
                )

    def assert_unordered_result(self, result, cls, *expected):
        """As assert_result, but the order of objects is not considered.

        The algorithm is very expensive but not a big deal for the small
        numbers of rows that the test suite manipulates.
        """

        class immutabledict(dict):
            def __hash__(self):
                return id(self)

        found = util.IdentitySet(result)
        expected = {immutabledict(e) for e in expected}

        for wrong in filterfalse(lambda o: isinstance(o, cls), found):
            fail(
                'Unexpected type "%s", expected "%s"'
                % (type(wrong).__name__, cls.__name__)
            )

        if len(found) != len(expected):
            fail(
                'Unexpected object count "%s", expected "%s"'
                % (len(found), len(expected))
            )

        NOVALUE = object()

        def _compare_item(obj, spec):
            for key, value in spec.items():
                if isinstance(value, tuple):
                    try:
                        self.assert_unordered_result(
                            getattr(obj, key), value[0], *value[1]
                        )
                    except AssertionError:
                        return False
                else:
                    if getattr(obj, key, NOVALUE) != value:
                        return False
            return True

        for expected_item in expected:
            for found_item in found:
                if _compare_item(found_item, expected_item):
                    found.remove(found_item)
                    break
            else:
                fail(
                    "Expected %s instance with attributes %s not found."
                    % (cls.__name__, repr(expected_item))
                )
        return True

    def sql_execution_asserter(self, db=None):
        if db is None:
            from . import db as db

        return assertsql.assert_engine(db)

    def assert_sql_execution(self, db, callable_, *rules):
        with self.sql_execution_asserter(db) as asserter:
            result = callable_()
        asserter.assert_(*rules)
        return result

    def assert_sql(self, db, callable_, rules):
        newrules = []
        for rule in rules:
            if isinstance(rule, dict):
                newrule = assertsql.AllOf(
                    *[assertsql.CompiledSQL(k, v) for k, v in rule.items()]
                )
            else:
                newrule = assertsql.CompiledSQL(*rule)
            newrules.append(newrule)

        return self.assert_sql_execution(db, callable_, *newrules)

    def assert_sql_count(self, db, callable_, count):
        return self.assert_sql_execution(
            db, callable_, assertsql.CountStatements(count)
        )

    @contextlib.contextmanager
    def assert_execution(self, db, *rules):
        with self.sql_execution_asserter(db) as asserter:
            yield
        asserter.assert_(*rules)

    def assert_statement_count(self, db, count):
        return self.assert_execution(db, assertsql.CountStatements(count))

    @contextlib.contextmanager
    def assert_statement_count_multi_db(self, dbs, counts):
        recs = [
            (self.sql_execution_asserter(db), db, count)
            for (db, count) in zip(dbs, counts)
        ]
        asserters = []
        for ctx, db, count in recs:
            asserters.append(ctx.__enter__())
        try:
            yield
        finally:
            for asserter, (ctx, db, count) in zip(asserters, recs):
                ctx.__exit__(None, None, None)
                asserter.assert_(assertsql.CountStatements(count))


class ComparesIndexes:
    def compare_table_index_with_expected(
        self, table: schema.Table, expected: list, dialect_name: str
    ):
        eq_(len(table.indexes), len(expected))
        idx_dict = {idx.name: idx for idx in table.indexes}
        for exp in expected:
            idx = idx_dict[exp["name"]]
            eq_(idx.unique, exp["unique"])
            cols = [c for c in exp["column_names"] if c is not None]
            eq_(len(idx.columns), len(cols))
            for c in cols:
                is_true(c in idx.columns)
            exprs = exp.get("expressions")
            if exprs:
                eq_(len(idx.expressions), len(exprs))
                for idx_exp, expr, col in zip(
                    idx.expressions, exprs, exp["column_names"]
                ):
                    if col is None:
                        eq_(idx_exp.text, expr)
            if (
                exp.get("dialect_options")
                and f"{dialect_name}_include" in exp["dialect_options"]
            ):
                eq_(
                    idx.dialect_options[dialect_name]["include"],
                    exp["dialect_options"][f"{dialect_name}_include"],
                )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\state.py ===
"""Dirac notation for states."""

from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import oo, equal_valued
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.printing.pretty.stringpict import stringPict
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.physics.quantum.kind import KetKind, BraKind


__all__ = [
    'KetBase',
    'BraBase',
    'StateBase',
    'State',
    'Ket',
    'Bra',
    'TimeDepState',
    'TimeDepBra',
    'TimeDepKet',
    'OrthogonalKet',
    'OrthogonalBra',
    'OrthogonalState',
    'Wavefunction'
]


#-----------------------------------------------------------------------------
# States, bras and kets.
#-----------------------------------------------------------------------------

# ASCII brackets
_lbracket = "<"
_rbracket = ">"
_straight_bracket = "|"


# Unicode brackets
# MATHEMATICAL ANGLE BRACKETS
_lbracket_ucode = "\N{MATHEMATICAL LEFT ANGLE BRACKET}"
_rbracket_ucode = "\N{MATHEMATICAL RIGHT ANGLE BRACKET}"
# LIGHT VERTICAL BAR
_straight_bracket_ucode = "\N{LIGHT VERTICAL BAR}"

# Other options for unicode printing of <, > and | for Dirac notation.

# LEFT-POINTING ANGLE BRACKET
# _lbracket = "\u2329"
# _rbracket = "\u232A"

# LEFT ANGLE BRACKET
# _lbracket = "\u3008"
# _rbracket = "\u3009"

# VERTICAL LINE
# _straight_bracket = "\u007C"


class StateBase(QExpr):
    """Abstract base class for general abstract states in quantum mechanics.

    All other state classes defined will need to inherit from this class. It
    carries the basic structure for all other states such as dual, _eval_adjoint
    and label.

    This is an abstract base class and you should not instantiate it directly,
    instead use State.
    """

    @classmethod
    def _operators_to_state(self, ops, **options):
        """ Returns the eigenstate instance for the passed operators.

        This method should be overridden in subclasses. It will handle being
        passed either an Operator instance or set of Operator instances. It
        should return the corresponding state INSTANCE or simply raise a
        NotImplementedError. See cartesian.py for an example.
        """

        raise NotImplementedError("Cannot map operators to states in this class. Method not implemented!")

    def _state_to_operators(self, op_classes, **options):
        """ Returns the operators which this state instance is an eigenstate
        of.

        This method should be overridden in subclasses. It will be called on
        state instances and be passed the operator classes that we wish to make
        into instances. The state instance will then transform the classes
        appropriately, or raise a NotImplementedError if it cannot return
        operator instances. See cartesian.py for examples,
        """

        raise NotImplementedError(
            "Cannot map this state to operators. Method not implemented!")

    @property
    def operators(self):
        """Return the operator(s) that this state is an eigenstate of"""
        from .operatorset import state_to_operators  # import internally to avoid circular import errors
        return state_to_operators(self)

    def _enumerate_state(self, num_states, **options):
        raise NotImplementedError("Cannot enumerate this state!")

    def _represent_default_basis(self, **options):
        return self._represent(basis=self.operators)

    def _apply_operator(self, op, **options):
        return None

    #-------------------------------------------------------------------------
    # Dagger/dual
    #-------------------------------------------------------------------------

    @property
    def dual(self):
        """Return the dual state of this one."""
        return self.dual_class()._new_rawargs(self.hilbert_space, *self.args)

    @classmethod
    def dual_class(self):
        """Return the class used to construct the dual."""
        raise NotImplementedError(
            'dual_class must be implemented in a subclass'
        )

    def _eval_adjoint(self):
        """Compute the dagger of this state using the dual."""
        return self.dual

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    def _pretty_brackets(self, height, use_unicode=True):
        # Return pretty printed brackets for the state
        # Ideally, this could be done by pform.parens but it does not support the angled < and >

        # Setup for unicode vs ascii
        if use_unicode:
            lbracket, rbracket = getattr(self, 'lbracket_ucode', ""), getattr(self, 'rbracket_ucode', "")
            slash, bslash, vert = '\N{BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT}', \
                                  '\N{BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT}', \
                                  '\N{BOX DRAWINGS LIGHT VERTICAL}'
        else:
            lbracket, rbracket = getattr(self, 'lbracket', ""), getattr(self, 'rbracket', "")
            slash, bslash, vert = '/', '\\', '|'

        # If height is 1, just return brackets
        if height == 1:
            return stringPict(lbracket), stringPict(rbracket)
        # Make height even
        height += (height % 2)

        brackets = []
        for bracket in lbracket, rbracket:
            # Create left bracket
            if bracket in {_lbracket, _lbracket_ucode}:
                bracket_args = [ ' ' * (height//2 - i - 1) +
                                 slash for i in range(height // 2)]
                bracket_args.extend(
                    [' ' * i + bslash for i in range(height // 2)])
            # Create right bracket
            elif bracket in {_rbracket, _rbracket_ucode}:
                bracket_args = [ ' ' * i + bslash for i in range(height // 2)]
                bracket_args.extend([ ' ' * (
                    height//2 - i - 1) + slash for i in range(height // 2)])
            # Create straight bracket
            elif bracket in {_straight_bracket, _straight_bracket_ucode}:
                bracket_args = [vert] * height
            else:
                raise ValueError(bracket)
            brackets.append(
                stringPict('\n'.join(bracket_args), baseline=height//2))
        return brackets

    def _sympystr(self, printer, *args):
        contents = self._print_contents(printer, *args)
        return '%s%s%s' % (getattr(self, 'lbracket', ""), contents, getattr(self, 'rbracket', ""))

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        # Get brackets
        pform = self._print_contents_pretty(printer, *args)
        lbracket, rbracket = self._pretty_brackets(
            pform.height(), printer._use_unicode)
        # Put together state
        pform = prettyForm(*pform.left(lbracket))
        pform = prettyForm(*pform.right(rbracket))
        return pform

    def _latex(self, printer, *args):
        contents = self._print_contents_latex(printer, *args)
        # The extra {} brackets are needed to get matplotlib's latex
        # rendered to render this properly.
        return '{%s%s%s}' % (getattr(self, 'lbracket_latex', ""), contents, getattr(self, 'rbracket_latex', ""))


class KetBase(StateBase):
    """Base class for Kets.

    This class defines the dual property and the brackets for printing. This is
    an abstract base class and you should not instantiate it directly, instead
    use Ket.
    """

    kind = KetKind

    lbracket = _straight_bracket
    rbracket = _rbracket
    lbracket_ucode = _straight_bracket_ucode
    rbracket_ucode = _rbracket_ucode
    lbracket_latex = r'\left|'
    rbracket_latex = r'\right\rangle '

    @classmethod
    def default_args(self):
        return ("psi",)

    @classmethod
    def dual_class(self):
        return BraBase

    #-------------------------------------------------------------------------
    # _eval_* methods
    #-------------------------------------------------------------------------

    def _eval_innerproduct(self, bra, **hints):
        """Evaluate the inner product between this ket and a bra.

        This is called to compute <bra|ket>, where the ket is ``self``.

        This method will dispatch to sub-methods having the format::

            ``def _eval_innerproduct_BraClass(self, **hints):``

        Subclasses should define these methods (one for each BraClass) to
        teach the ket how to take inner products with bras.
        """
        return dispatch_method(self, '_eval_innerproduct', bra, **hints)

    def _apply_from_right_to(self, op, **options):
        """Apply an Operator to this Ket as Operator*Ket

        This method will dispatch to methods having the format::

            ``def _apply_from_right_to_OperatorName(op, **options):``

        Subclasses should define these methods (one for each OperatorName) to
        teach the Ket how to implement OperatorName*Ket

        Parameters
        ==========

        op : Operator
            The Operator that is acting on the Ket as op*Ket
        options : dict
            A dict of key/value pairs that control how the operator is applied
            to the Ket.
        """
        return dispatch_method(self, '_apply_from_right_to', op, **options)


class BraBase(StateBase):
    """Base class for Bras.

    This class defines the dual property and the brackets for printing. This
    is an abstract base class and you should not instantiate it directly,
    instead use Bra.
    """

    kind = BraKind

    lbracket = _lbracket
    rbracket = _straight_bracket
    lbracket_ucode = _lbracket_ucode
    rbracket_ucode = _straight_bracket_ucode
    lbracket_latex = r'\left\langle '
    rbracket_latex = r'\right|'

    @classmethod
    def _operators_to_state(self, ops, **options):
        state = self.dual_class()._operators_to_state(ops, **options)
        return state.dual

    def _state_to_operators(self, op_classes, **options):
        return self.dual._state_to_operators(op_classes, **options)

    def _enumerate_state(self, num_states, **options):
        dual_states = self.dual._enumerate_state(num_states, **options)
        return [x.dual for x in dual_states]

    @classmethod
    def default_args(self):
        return self.dual_class().default_args()

    @classmethod
    def dual_class(self):
        return KetBase

    def _represent(self, **options):
        """A default represent that uses the Ket's version."""
        from sympy.physics.quantum.dagger import Dagger
        return Dagger(self.dual._represent(**options))


class State(StateBase):
    """General abstract quantum state used as a base class for Ket and Bra."""
    pass


class Ket(State, KetBase):
    """A general time-independent Ket in quantum mechanics.

    Inherits from State and KetBase. This class should be used as the base
    class for all physical, time-independent Kets in a system. This class
    and its subclasses will be the main classes that users will use for
    expressing Kets in Dirac notation [1]_.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        ket. This will usually be its symbol or its quantum numbers. For
        time-dependent state, this will include the time.

    Examples
    ========

    Create a simple Ket and looking at its properties::

        >>> from sympy.physics.quantum import Ket
        >>> from sympy import symbols, I
        >>> k = Ket('psi')
        >>> k
        |psi>
        >>> k.hilbert_space
        H
        >>> k.is_commutative
        False
        >>> k.label
        (psi,)

    Ket's know about their associated bra::

        >>> k.dual
        <psi|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.state.Bra'>

    Take a linear combination of two kets::

        >>> k0 = Ket(0)
        >>> k1 = Ket(1)
        >>> 2*I*k0 - 4*k1
        2*I*|0> - 4*|1>

    Compound labels are passed as tuples::

        >>> n, m = symbols('n,m')
        >>> k = Ket(n,m)
        >>> k
        |nm>

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bra-ket_notation
    """

    @classmethod
    def dual_class(self):
        return Bra


class Bra(State, BraBase):
    """A general time-independent Bra in quantum mechanics.

    Inherits from State and BraBase. A Bra is the dual of a Ket [1]_. This
    class and its subclasses will be the main classes that users will use for
    expressing Bras in Dirac notation.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        ket. This will usually be its symbol or its quantum numbers. For
        time-dependent state, this will include the time.

    Examples
    ========

    Create a simple Bra and look at its properties::

        >>> from sympy.physics.quantum import Bra
        >>> from sympy import symbols, I
        >>> b = Bra('psi')
        >>> b
        <psi|
        >>> b.hilbert_space
        H
        >>> b.is_commutative
        False

    Bra's know about their dual Ket's::

        >>> b.dual
        |psi>
        >>> b.dual_class()
        <class 'sympy.physics.quantum.state.Ket'>

    Like Kets, Bras can have compound labels and be manipulated in a similar
    manner::

        >>> n, m = symbols('n,m')
        >>> b = Bra(n,m) - I*Bra(m,n)
        >>> b
        -I*<mn| + <nm|

    Symbols in a Bra can be substituted using ``.subs``::

        >>> b.subs(n,m)
        <mm| - I*<mm|

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bra-ket_notation
    """

    @classmethod
    def dual_class(self):
        return Ket

#-----------------------------------------------------------------------------
# Time dependent states, bras and kets.
#-----------------------------------------------------------------------------


class TimeDepState(StateBase):
    """Base class for a general time-dependent quantum state.

    This class is used as a base class for any time-dependent state. The main
    difference between this class and the time-independent state is that this
    class takes a second argument that is the time in addition to the usual
    label argument.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.
    """

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def default_args(self):
        return ("psi", "t")

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def label(self):
        """The label of the state."""
        return self.args[:-1]

    @property
    def time(self):
        """The time of the state."""
        return self.args[-1]

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    def _print_time(self, printer, *args):
        return printer._print(self.time, *args)

    _print_time_repr = _print_time
    _print_time_latex = _print_time

    def _print_time_pretty(self, printer, *args):
        pform = printer._print(self.time, *args)
        return pform

    def _print_contents(self, printer, *args):
        label = self._print_label(printer, *args)
        time = self._print_time(printer, *args)
        return '%s;%s' % (label, time)

    def _print_label_repr(self, printer, *args):
        label = self._print_sequence(self.label, ',', printer, *args)
        time = self._print_time_repr(printer, *args)
        return '%s,%s' % (label, time)

    def _print_contents_pretty(self, printer, *args):
        label = self._print_label_pretty(printer, *args)
        time = self._print_time_pretty(printer, *args)
        return printer._print_seq((label, time), delimiter=';')

    def _print_contents_latex(self, printer, *args):
        label = self._print_sequence(
            self.label, self._label_separator, printer, *args)
        time = self._print_time_latex(printer, *args)
        return '%s;%s' % (label, time)


class TimeDepKet(TimeDepState, KetBase):
    """General time-dependent Ket in quantum mechanics.

    This inherits from ``TimeDepState`` and ``KetBase`` and is the main class
    that should be used for Kets that vary with time. Its dual is a
    ``TimeDepBra``.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.

    Examples
    ========

    Create a TimeDepKet and look at its attributes::

        >>> from sympy.physics.quantum import TimeDepKet
        >>> k = TimeDepKet('psi', 't')
        >>> k
        |psi;t>
        >>> k.time
        t
        >>> k.label
        (psi,)
        >>> k.hilbert_space
        H

    TimeDepKets know about their dual bra::

        >>> k.dual
        <psi;t|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.state.TimeDepBra'>
    """

    @classmethod
    def dual_class(self):
        return TimeDepBra


class TimeDepBra(TimeDepState, BraBase):
    """General time-dependent Bra in quantum mechanics.

    This inherits from TimeDepState and BraBase and is the main class that
    should be used for Bras that vary with time. Its dual is a TimeDepBra.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.

    Examples
    ========

        >>> from sympy.physics.quantum import TimeDepBra
        >>> b = TimeDepBra('psi', 't')
        >>> b
        <psi;t|
        >>> b.time
        t
        >>> b.label
        (psi,)
        >>> b.hilbert_space
        H
        >>> b.dual
        |psi;t>
    """

    @classmethod
    def dual_class(self):
        return TimeDepKet


class OrthogonalState(State):
    """General abstract quantum state used as a base class for Ket and Bra."""
    pass

class OrthogonalKet(OrthogonalState, KetBase):
    """Orthogonal Ket in quantum mechanics.

    The inner product of two states with different labels will give zero,
    states with the same label will give one.

        >>> from sympy.physics.quantum import OrthogonalBra, OrthogonalKet
        >>> from sympy.abc import m, n
        >>> (OrthogonalBra(n)*OrthogonalKet(n)).doit()
        1
        >>> (OrthogonalBra(n)*OrthogonalKet(n+1)).doit()
        0
        >>> (OrthogonalBra(n)*OrthogonalKet(m)).doit()
        <n|m>
    """

    @classmethod
    def dual_class(self):
        return OrthogonalBra

    def _eval_innerproduct(self, bra, **hints):

        if len(self.args) != len(bra.args):
            raise ValueError('Cannot multiply a ket that has a different number of labels.')

        for arg, bra_arg in zip(self.args, bra.args):
            diff = arg - bra_arg
            diff = diff.expand()

            is_zero = diff.is_zero

            if is_zero is False:
                return S.Zero # i.e. Integer(0)

            if is_zero is None:
                return None

        return S.One # i.e. Integer(1)


class OrthogonalBra(OrthogonalState, BraBase):
    """Orthogonal Bra in quantum mechanics.
    """

    @classmethod
    def dual_class(self):
        return OrthogonalKet


class Wavefunction(Function):
    """Class for representations in continuous bases

    This class takes an expression and coordinates in its constructor. It can
    be used to easily calculate normalizations and probabilities.

    Parameters
    ==========

    expr : Expr
           The expression representing the functional form of the w.f.

    coords : Symbol or tuple
           The coordinates to be integrated over, and their bounds

    Examples
    ========

    Particle in a box, specifying bounds in the more primitive way of using
    Piecewise:

        >>> from sympy import Symbol, Piecewise, pi, N
        >>> from sympy.functions import sqrt, sin
        >>> from sympy.physics.quantum.state import Wavefunction
        >>> x = Symbol('x', real=True)
        >>> n = 1
        >>> L = 1
        >>> g = Piecewise((0, x < 0), (0, x > L), (sqrt(2//L)*sin(n*pi*x/L), True))
        >>> f = Wavefunction(g, x)
        >>> f.norm
        1
        >>> f.is_normalized
        True
        >>> p = f.prob()
        >>> p(0)
        0
        >>> p(L)
        0
        >>> p(0.5)
        2
        >>> p(0.85*L)
        2*sin(0.85*pi)**2
        >>> N(p(0.85*L))
        0.412214747707527

    Additionally, you can specify the bounds of the function and the indices in
    a more compact way:

        >>> from sympy import symbols, pi, diff
        >>> from sympy.functions import sqrt, sin
        >>> from sympy.physics.quantum.state import Wavefunction
        >>> x, L = symbols('x,L', positive=True)
        >>> n = symbols('n', integer=True, positive=True)
        >>> g = sqrt(2/L)*sin(n*pi*x/L)
        >>> f = Wavefunction(g, (x, 0, L))
        >>> f.norm
        1
        >>> f(L+1)
        0
        >>> f(L-1)
        sqrt(2)*sin(pi*n*(L - 1)/L)/sqrt(L)
        >>> f(-1)
        0
        >>> f(0.85)
        sqrt(2)*sin(0.85*pi*n/L)/sqrt(L)
        >>> f(0.85, n=1, L=1)
        sqrt(2)*sin(0.85*pi)
        >>> f.is_commutative
        False

    All arguments are automatically sympified, so you can define the variables
    as strings rather than symbols:

        >>> expr = x**2
        >>> f = Wavefunction(expr, 'x')
        >>> type(f.variables[0])
        <class 'sympy.core.symbol.Symbol'>

    Derivatives of Wavefunctions will return Wavefunctions:

        >>> diff(f, x)
        Wavefunction(2*x, x)

    """

    #Any passed tuples for coordinates and their bounds need to be
    #converted to Tuples before Function's constructor is called, to
    #avoid errors from calling is_Float in the constructor
    def __new__(cls, *args, **options):
        new_args = [None for i in args]
        ct = 0
        for arg in args:
            if isinstance(arg, tuple):
                new_args[ct] = Tuple(*arg)
            else:
                new_args[ct] = arg
            ct += 1

        return super().__new__(cls, *new_args, **options)

    def __call__(self, *args, **options):
        var = self.variables

        if len(args) != len(var):
            raise NotImplementedError(
                "Incorrect number of arguments to function!")

        ct = 0
        #If the passed value is outside the specified bounds, return 0
        for v in var:
            lower, upper = self.limits[v]

            #Do the comparison to limits only if the passed symbol is actually
            #a symbol present in the limits;
            #Had problems with a comparison of x > L
            if isinstance(args[ct], Expr) and \
                not (lower in args[ct].free_symbols
                     or upper in args[ct].free_symbols):
                continue

            if (args[ct] < lower) == True or (args[ct] > upper) == True:
                return S.Zero

            ct += 1

        expr = self.expr

        #Allows user to make a call like f(2, 4, m=1, n=1)
        for symbol in list(expr.free_symbols):
            if str(symbol) in options.keys():
                val = options[str(symbol)]
                expr = expr.subs(symbol, val)

        return expr.subs(zip(var, args))

    def _eval_derivative(self, symbol):
        expr = self.expr
        deriv = expr._eval_derivative(symbol)

        return Wavefunction(deriv, *self.args[1:])

    def _eval_conjugate(self):
        return Wavefunction(conjugate(self.expr), *self.args[1:])

    def _eval_transpose(self):
        return self

    @property
    def is_commutative(self):
        """
        Override Function's is_commutative so that order is preserved in
        represented expressions
        """
        return False

    @classmethod
    def eval(self, *args):
        return None

    @property
    def variables(self):
        """
        Return the coordinates which the wavefunction depends on

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x,y = symbols('x,y')
            >>> f = Wavefunction(x*y, x, y)
            >>> f.variables
            (x, y)
            >>> g = Wavefunction(x*y, x)
            >>> g.variables
            (x,)

        """
        var = [g[0] if isinstance(g, Tuple) else g for g in self._args[1:]]
        return tuple(var)

    @property
    def limits(self):
        """
        Return the limits of the coordinates which the w.f. depends on If no
        limits are specified, defaults to ``(-oo, oo)``.

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x, y = symbols('x, y')
            >>> f = Wavefunction(x**2, (x, 0, 1))
            >>> f.limits
            {x: (0, 1)}
            >>> f = Wavefunction(x**2, x)
            >>> f.limits
            {x: (-oo, oo)}
            >>> f = Wavefunction(x**2 + y**2, x, (y, -1, 2))
            >>> f.limits
            {x: (-oo, oo), y: (-1, 2)}

        """
        limits = [(g[1], g[2]) if isinstance(g, Tuple) else (-oo, oo)
                  for g in self._args[1:]]
        return dict(zip(self.variables, tuple(limits)))

    @property
    def expr(self):
        """
        Return the expression which is the functional form of the Wavefunction

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x, y = symbols('x, y')
            >>> f = Wavefunction(x**2, x)
            >>> f.expr
            x**2

        """
        return self._args[0]

    @property
    def is_normalized(self):
        """
        Returns true if the Wavefunction is properly normalized

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sqrt, sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sqrt(2/L)*sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.is_normalized
            True

        """

        return equal_valued(self.norm, 1)

    @property  # type: ignore
    @cacheit
    def norm(self):
        """
        Return the normalization of the specified functional form.

        This function integrates over the coordinates of the Wavefunction, with
        the bounds specified.

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sqrt, sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sqrt(2/L)*sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.norm
            1
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.norm
            sqrt(2)*sqrt(L)/2

        """

        exp = self.expr*conjugate(self.expr)
        var = self.variables
        limits = self.limits

        for v in var:
            curr_limits = limits[v]
            exp = integrate(exp, (v, curr_limits[0], curr_limits[1]))

        return sqrt(exp)

    def normalize(self):
        """
        Return a normalized version of the Wavefunction

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x = symbols('x', real=True)
            >>> L = symbols('L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.normalize()
            Wavefunction(sqrt(2)*sin(pi*n*x/L)/sqrt(L), (x, 0, L))

        """
        const = self.norm

        if const is oo:
            raise NotImplementedError("The function is not normalizable!")
        else:
            return Wavefunction((const)**(-1)*self.expr, *self.args[1:])

    def prob(self):
        r"""
        Return the absolute magnitude of the w.f., `|\psi(x)|^2`

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', real=True)
            >>> n = symbols('n', integer=True)
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.prob()
            Wavefunction(sin(pi*n*x/L)**2, x)

        """

        return Wavefunction(self.expr*conjugate(self.expr), *self.variables)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\inequalities.py ===
"""Tools for solving inequalities and systems of inequalities. """
import itertools

from sympy.calculus.util import (continuous_domain, periodicity,
    function_range)
from sympy.core import sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Relational, Lt, Ge, Eq
from sympy.core.symbol import Symbol, Dummy
from sympy.sets.sets import Interval, FiniteSet, Union, Intersection
from sympy.core.singleton import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.complexes import Abs
from sympy.logic import And
from sympy.polys import Poly, PolynomialError, parallel_poly_from_expr
from sympy.polys.polyutils import _nsort
from sympy.solvers.solveset import solvify, solveset
from sympy.utilities.iterables import sift, iterable
from sympy.utilities.misc import filldedent


def solve_poly_inequality(poly, rel):
    """Solve a polynomial inequality with rational coefficients.

    Examples
    ========

    >>> from sympy import solve_poly_inequality, Poly
    >>> from sympy.abc import x

    >>> solve_poly_inequality(Poly(x, x, domain='ZZ'), '==')
    [{0}]

    >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '!=')
    [Interval.open(-oo, -1), Interval.open(-1, 1), Interval.open(1, oo)]

    >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '==')
    [{-1}, {1}]

    See Also
    ========
    solve_poly_inequalities
    """
    if not isinstance(poly, Poly):
        raise ValueError(
            'For efficiency reasons, `poly` should be a Poly instance')
    if poly.as_expr().is_number:
        t = Relational(poly.as_expr(), 0, rel)
        if t is S.true:
            return [S.Reals]
        elif t is S.false:
            return [S.EmptySet]
        else:
            raise NotImplementedError(
                "could not determine truth value of %s" % t)

    reals, intervals = poly.real_roots(multiple=False), []

    if rel == '==':
        for root, _ in reals:
            interval = Interval(root, root)
            intervals.append(interval)
    elif rel == '!=':
        left = S.NegativeInfinity

        for right, _ in reals + [(S.Infinity, 1)]:
            interval = Interval(left, right, True, True)
            intervals.append(interval)
            left = right
    else:
        if poly.LC() > 0:
            sign = +1
        else:
            sign = -1

        eq_sign, equal = None, False

        if rel == '>':
            eq_sign = +1
        elif rel == '<':
            eq_sign = -1
        elif rel == '>=':
            eq_sign, equal = +1, True
        elif rel == '<=':
            eq_sign, equal = -1, True
        else:
            raise ValueError("'%s' is not a valid relation" % rel)

        right, right_open = S.Infinity, True

        for left, multiplicity in reversed(reals):
            if multiplicity % 2:
                if sign == eq_sign:
                    intervals.insert(
                        0, Interval(left, right, not equal, right_open))

                sign, right, right_open = -sign, left, not equal
            else:
                if sign == eq_sign and not equal:
                    intervals.insert(
                        0, Interval(left, right, True, right_open))
                    right, right_open = left, True
                elif sign != eq_sign and equal:
                    intervals.insert(0, Interval(left, left))

        if sign == eq_sign:
            intervals.insert(
                0, Interval(S.NegativeInfinity, right, True, right_open))

    return intervals


def solve_poly_inequalities(polys):
    """Solve polynomial inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy import Poly
    >>> from sympy.solvers.inequalities import solve_poly_inequalities
    >>> from sympy.abc import x
    >>> solve_poly_inequalities(((
    ... Poly(x**2 - 3), ">"), (
    ... Poly(-x**2 + 1), ">")))
    Union(Interval.open(-oo, -sqrt(3)), Interval.open(-1, 1), Interval.open(sqrt(3), oo))
    """
    return Union(*[s for p in polys for s in solve_poly_inequality(*p)])


def solve_rational_inequalities(eqs):
    """Solve a system of rational inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import solve_rational_inequalities, Poly

    >>> solve_rational_inequalities([[
    ... ((Poly(-x + 1), Poly(1, x)), '>='),
    ... ((Poly(-x + 1), Poly(1, x)), '<=')]])
    {1}

    >>> solve_rational_inequalities([[
    ... ((Poly(x), Poly(1, x)), '!='),
    ... ((Poly(-x + 1), Poly(1, x)), '>=')]])
    Union(Interval.open(-oo, 0), Interval.Lopen(0, 1))

    See Also
    ========
    solve_poly_inequality
    """
    result = S.EmptySet

    for _eqs in eqs:
        if not _eqs:
            continue

        global_intervals = [Interval(S.NegativeInfinity, S.Infinity)]

        for (numer, denom), rel in _eqs:
            numer_intervals = solve_poly_inequality(numer*denom, rel)
            denom_intervals = solve_poly_inequality(denom, '==')

            intervals = []

            for numer_interval, global_interval in itertools.product(
                    numer_intervals, global_intervals):
                interval = numer_interval.intersect(global_interval)

                if interval is not S.EmptySet:
                    intervals.append(interval)

            global_intervals = intervals

            intervals = []

            for global_interval in global_intervals:
                for denom_interval in denom_intervals:
                    global_interval -= denom_interval

                if global_interval is not S.EmptySet:
                    intervals.append(global_interval)

            global_intervals = intervals

            if not global_intervals:
                break

        for interval in global_intervals:
            result = result.union(interval)

    return result


def reduce_rational_inequalities(exprs, gen, relational=True):
    """Reduce a system of rational inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.solvers.inequalities import reduce_rational_inequalities

    >>> x = Symbol('x', real=True)

    >>> reduce_rational_inequalities([[x**2 <= 0]], x)
    Eq(x, 0)

    >>> reduce_rational_inequalities([[x + 2 > 0]], x)
    -2 < x
    >>> reduce_rational_inequalities([[(x + 2, ">")]], x)
    -2 < x
    >>> reduce_rational_inequalities([[x + 2]], x)
    Eq(x, -2)

    This function find the non-infinite solution set so if the unknown symbol
    is declared as extended real rather than real then the result may include
    finiteness conditions:

    >>> y = Symbol('y', extended_real=True)
    >>> reduce_rational_inequalities([[y + 2 > 0]], y)
    (-2 < y) & (y < oo)
    """
    exact = True
    eqs = []
    solution = S.EmptySet  # add pieces for each group
    for _exprs in exprs:
        if not _exprs:
            continue
        _eqs = []
        _sol = S.Reals
        for expr in _exprs:
            if isinstance(expr, tuple):
                expr, rel = expr
            else:
                if expr.is_Relational:
                    expr, rel = expr.lhs - expr.rhs, expr.rel_op
                else:
                    rel = '=='

            if expr is S.true:
                numer, denom, rel = S.Zero, S.One, '=='
            elif expr is S.false:
                numer, denom, rel = S.One, S.One, '=='
            else:
                numer, denom = expr.together().as_numer_denom()

            try:
                (numer, denom), opt = parallel_poly_from_expr(
                    (numer, denom), gen)
            except PolynomialError:
                raise PolynomialError(filldedent('''
                    only polynomials and rational functions are
                    supported in this context.
                    '''))

            if not opt.domain.is_Exact:
                numer, denom, exact = numer.to_exact(), denom.to_exact(), False

            domain = opt.domain.get_exact()

            if not (domain.is_ZZ or domain.is_QQ):
                expr = numer/denom
                expr = Relational(expr, 0, rel)
                _sol &= solve_univariate_inequality(expr, gen, relational=False)
            else:
                _eqs.append(((numer, denom), rel))

        if _eqs:
            _sol &= solve_rational_inequalities([_eqs])
            exclude = solve_rational_inequalities([[((d, d.one), '==')
                for i in eqs for ((n, d), _) in i if d.has(gen)]])
            _sol -= exclude

        solution |= _sol

    if not exact and solution:
        solution = solution.evalf()

    if relational:
        solution = solution.as_relational(gen)

    return solution


def reduce_abs_inequality(expr, rel, gen):
    """Reduce an inequality with nested absolute values.

    Examples
    ========

    >>> from sympy import reduce_abs_inequality, Abs, Symbol
    >>> x = Symbol('x', real=True)

    >>> reduce_abs_inequality(Abs(x - 5) - 3, '<', x)
    (2 < x) & (x < 8)

    >>> reduce_abs_inequality(Abs(x + 2)*3 - 13, '<', x)
    (-19/3 < x) & (x < 7/3)

    See Also
    ========

    reduce_abs_inequalities
    """
    if gen.is_extended_real is False:
        raise TypeError(filldedent('''
            Cannot solve inequalities with absolute values containing
            non-real variables.
            '''))

    def _bottom_up_scan(expr):
        exprs = []

        if expr.is_Add or expr.is_Mul:
            op = expr.func

            for arg in expr.args:
                _exprs = _bottom_up_scan(arg)

                if not exprs:
                    exprs = _exprs
                else:
                    exprs = [(op(expr, _expr), conds + _conds) for (expr, conds), (_expr, _conds) in
                            itertools.product(exprs, _exprs)]
        elif expr.is_Pow:
            n = expr.exp
            if not n.is_Integer:
                raise ValueError("Only Integer Powers are allowed on Abs.")

            exprs.extend((expr**n, conds) for expr, conds in _bottom_up_scan(expr.base))
        elif isinstance(expr, Abs):
            _exprs = _bottom_up_scan(expr.args[0])

            for expr, conds in _exprs:
                exprs.append(( expr, conds + [Ge(expr, 0)]))
                exprs.append((-expr, conds + [Lt(expr, 0)]))
        else:
            exprs = [(expr, [])]

        return exprs

    mapping = {'<': '>', '<=': '>='}
    inequalities = []

    for expr, conds in _bottom_up_scan(expr):
        if rel not in mapping.keys():
            expr = Relational( expr, 0, rel)
        else:
            expr = Relational(-expr, 0, mapping[rel])

        inequalities.append([expr] + conds)

    return reduce_rational_inequalities(inequalities, gen)


def reduce_abs_inequalities(exprs, gen):
    """Reduce a system of inequalities with nested absolute values.

    Examples
    ========

    >>> from sympy import reduce_abs_inequalities, Abs, Symbol
    >>> x = Symbol('x', extended_real=True)

    >>> reduce_abs_inequalities([(Abs(3*x - 5) - 7, '<'),
    ... (Abs(x + 25) - 13, '>')], x)
    (-2/3 < x) & (x < 4) & (((-oo < x) & (x < -38)) | ((-12 < x) & (x < oo)))

    >>> reduce_abs_inequalities([(Abs(x - 4) + Abs(3*x - 5) - 7, '<')], x)
    (1/2 < x) & (x < 4)

    See Also
    ========

    reduce_abs_inequality
    """
    return And(*[ reduce_abs_inequality(expr, rel, gen)
        for expr, rel in exprs ])


def solve_univariate_inequality(expr, gen, relational=True, domain=S.Reals, continuous=False):
    """Solves a real univariate inequality.

    Parameters
    ==========

    expr : Relational
        The target inequality
    gen : Symbol
        The variable for which the inequality is solved
    relational : bool
        A Relational type output is expected or not
    domain : Set
        The domain over which the equation is solved
    continuous: bool
        True if expr is known to be continuous over the given domain
        (and so continuous_domain() does not need to be called on it)

    Raises
    ======

    NotImplementedError
        The solution of the inequality cannot be determined due to limitation
        in :func:`sympy.solvers.solveset.solvify`.

    Notes
    =====

    Currently, we cannot solve all the inequalities due to limitations in
    :func:`sympy.solvers.solveset.solvify`. Also, the solution returned for trigonometric inequalities
    are restricted in its periodic interval.

    See Also
    ========

    sympy.solvers.solveset.solvify: solver returning solveset solutions with solve's output API

    Examples
    ========

    >>> from sympy import solve_univariate_inequality, Symbol, sin, Interval, S
    >>> x = Symbol('x')

    >>> solve_univariate_inequality(x**2 >= 4, x)
    ((2 <= x) & (x < oo)) | ((-oo < x) & (x <= -2))

    >>> solve_univariate_inequality(x**2 >= 4, x, relational=False)
    Union(Interval(-oo, -2), Interval(2, oo))

    >>> domain = Interval(0, S.Infinity)
    >>> solve_univariate_inequality(x**2 >= 4, x, False, domain)
    Interval(2, oo)

    >>> solve_univariate_inequality(sin(x) > 0, x, relational=False)
    Interval.open(0, pi)

    """
    from sympy.solvers.solvers import denoms

    if domain.is_subset(S.Reals) is False:
        raise NotImplementedError(filldedent('''
        Inequalities in the complex domain are
        not supported. Try the real domain by
        setting domain=S.Reals'''))
    elif domain is not S.Reals:
        rv = solve_univariate_inequality(
        expr, gen, relational=False, continuous=continuous).intersection(domain)
        if relational:
            rv = rv.as_relational(gen)
        return rv
    else:
        pass  # continue with attempt to solve in Real domain

    # This keeps the function independent of the assumptions about `gen`.
    # `solveset` makes sure this function is called only when the domain is
    # real.
    _gen = gen
    _domain = domain
    if gen.is_extended_real is False:
        rv = S.EmptySet
        return rv if not relational else rv.as_relational(_gen)
    elif gen.is_extended_real is None:
        gen = Dummy('gen', extended_real=True)
        try:
            expr = expr.xreplace({_gen: gen})
        except TypeError:
            raise TypeError(filldedent('''
                When gen is real, the relational has a complex part
                which leads to an invalid comparison like I < 0.
                '''))

    rv = None

    if expr is S.true:
        rv = domain

    elif expr is S.false:
        rv = S.EmptySet

    else:
        e = expr.lhs - expr.rhs
        period = periodicity(e, gen)
        if period == S.Zero:
            e = expand_mul(e)
            const = expr.func(e, 0)
            if const is S.true:
                rv = domain
            elif const is S.false:
                rv = S.EmptySet
        elif period is not None:
            frange = function_range(e, gen, domain)

            rel = expr.rel_op
            if rel in ('<', '<='):
                if expr.func(frange.sup, 0):
                    rv = domain
                elif not expr.func(frange.inf, 0):
                    rv = S.EmptySet

            elif rel in ('>', '>='):
                if expr.func(frange.inf, 0):
                    rv = domain
                elif not expr.func(frange.sup, 0):
                    rv = S.EmptySet

            inf, sup = domain.inf, domain.sup
            if sup - inf is S.Infinity:
                domain = Interval(0, period, False, True).intersect(_domain)
                _domain = domain

        if rv is None:
            n, d = e.as_numer_denom()
            try:
                if gen not in n.free_symbols and len(e.free_symbols) > 1:
                    raise ValueError
                # this might raise ValueError on its own
                # or it might give None...
                solns = solvify(e, gen, domain)
                if solns is None:
                    # in which case we raise ValueError
                    raise ValueError
            except (ValueError, NotImplementedError):
                # replace gen with generic x since it's
                # univariate anyway
                raise NotImplementedError(filldedent('''
                    The inequality, %s, cannot be solved using
                    solve_univariate_inequality.
                    ''' % expr.subs(gen, Symbol('x'))))

            expanded_e = expand_mul(e)
            def valid(x):
                # this is used to see if gen=x satisfies the
                # relational by substituting it into the
                # expanded form and testing against 0, e.g.
                # if expr = x*(x + 1) < 2 then e = x*(x + 1) - 2
                # and expanded_e = x**2 + x - 2; the test is
                # whether a given value of x satisfies
                # x**2 + x - 2 < 0
                #
                # expanded_e, expr and gen used from enclosing scope
                v = expanded_e.subs(gen, expand_mul(x))
                try:
                    r = expr.func(v, 0)
                except TypeError:
                    r = S.false
                if r in (S.true, S.false):
                    return r
                if v.is_extended_real is False:
                    return S.false
                else:
                    v = v.n(2)
                    if v.is_comparable:
                        return expr.func(v, 0)
                    # not comparable or couldn't be evaluated
                    raise NotImplementedError(
                        'relationship did not evaluate: %s' % r)

            singularities = []
            for d in denoms(expr, gen):
                singularities.extend(solvify(d, gen, domain))
            if not continuous:
                domain = continuous_domain(expanded_e, gen, domain)

            include_x = '=' in expr.rel_op and expr.rel_op != '!='

            try:
                discontinuities = set(domain.boundary -
                    FiniteSet(domain.inf, domain.sup))
                # remove points that are not between inf and sup of domain
                critical_points = FiniteSet(*(solns + singularities + list(
                    discontinuities))).intersection(
                    Interval(domain.inf, domain.sup,
                    domain.inf not in domain, domain.sup not in domain))
                if all(r.is_number for r in critical_points):
                    reals = _nsort(critical_points, separated=True)[0]
                else:
                    sifted = sift(critical_points, lambda x: x.is_extended_real)
                    if sifted[None]:
                        # there were some roots that weren't known
                        # to be real
                        raise NotImplementedError
                    try:
                        reals = sifted[True]
                        if len(reals) > 1:
                            reals = sorted(reals)
                    except TypeError:
                        raise NotImplementedError
            except NotImplementedError:
                raise NotImplementedError('sorting of these roots is not supported')

            # If expr contains imaginary coefficients, only take real
            # values of x for which the imaginary part is 0
            make_real = S.Reals
            if (coeffI := expanded_e.coeff(S.ImaginaryUnit)) != S.Zero:
                check = True
                im_sol = FiniteSet()
                try:
                    a = solveset(coeffI, gen, domain)
                    if not isinstance(a, Interval):
                        for z in a:
                            if z not in singularities and valid(z) and z.is_extended_real:
                                im_sol += FiniteSet(z)
                    else:
                        start, end = a.inf, a.sup
                        for z in _nsort(critical_points + FiniteSet(end)):
                            valid_start = valid(start)
                            if start != end:
                                valid_z = valid(z)
                                pt = _pt(start, z)
                                if pt not in singularities and pt.is_extended_real and valid(pt):
                                    if valid_start and valid_z:
                                        im_sol += Interval(start, z)
                                    elif valid_start:
                                        im_sol += Interval.Ropen(start, z)
                                    elif valid_z:
                                        im_sol += Interval.Lopen(start, z)
                                    else:
                                        im_sol += Interval.open(start, z)
                            start = z
                        for s in singularities:
                            im_sol -= FiniteSet(s)
                except (TypeError):
                    im_sol = S.Reals
                    check = False

                if im_sol is S.EmptySet:
                    raise ValueError(filldedent('''
                        %s contains imaginary parts which cannot be
                        made 0 for any value of %s satisfying the
                        inequality, leading to relations like I < 0.
                        '''  % (expr.subs(gen, _gen), _gen)))

                make_real = make_real.intersect(im_sol)

            sol_sets = [S.EmptySet]

            start = domain.inf
            if start in domain and valid(start) and start.is_finite:
                sol_sets.append(FiniteSet(start))

            for x in reals:
                end = x

                if valid(_pt(start, end)):
                    sol_sets.append(Interval(start, end, True, True))

                if x in singularities:
                    singularities.remove(x)
                else:
                    if x in discontinuities:
                        discontinuities.remove(x)
                        _valid = valid(x)
                    else:  # it's a solution
                        _valid = include_x
                    if _valid:
                        sol_sets.append(FiniteSet(x))

                start = end

            end = domain.sup
            if end in domain and valid(end) and end.is_finite:
                sol_sets.append(FiniteSet(end))

            if valid(_pt(start, end)):
                sol_sets.append(Interval.open(start, end))

            if coeffI != S.Zero and check:
                rv = (make_real).intersect(_domain)
            else:
                rv = Intersection(
                    (Union(*sol_sets)), make_real, _domain).subs(gen, _gen)

    return rv if not relational else rv.as_relational(_gen)


def _pt(start, end):
    """Return a point between start and end"""
    if not start.is_infinite and not end.is_infinite:
        pt = (start + end)/2
    elif start.is_infinite and end.is_infinite:
        pt = S.Zero
    else:
        if (start.is_infinite and start.is_extended_positive is None or
                end.is_infinite and end.is_extended_positive is None):
            raise ValueError('cannot proceed with unsigned infinite values')
        if (end.is_infinite and end.is_extended_negative or
                start.is_infinite and start.is_extended_positive):
            start, end = end, start
        # if possible, use a multiple of self which has
        # better behavior when checking assumptions than
        # an expression obtained by adding or subtracting 1
        if end.is_infinite:
            if start.is_extended_positive:
                pt = start*2
            elif start.is_extended_negative:
                pt = start*S.Half
            else:
                pt = start + 1
        elif start.is_infinite:
            if end.is_extended_positive:
                pt = end*S.Half
            elif end.is_extended_negative:
                pt = end*2
            else:
                pt = end - 1
    return pt


def _solve_inequality(ie, s, linear=False):
    """Return the inequality with s isolated on the left, if possible.
    If the relationship is non-linear, a solution involving And or Or
    may be returned. False or True are returned if the relationship
    is never True or always True, respectively.

    If `linear` is True (default is False) an `s`-dependent expression
    will be isolated on the left, if possible
    but it will not be solved for `s` unless the expression is linear
    in `s`. Furthermore, only "safe" operations which do not change the
    sense of the relationship are applied: no division by an unsigned
    value is attempted unless the relationship involves Eq or Ne and
    no division by a value not known to be nonzero is ever attempted.

    Examples
    ========

    >>> from sympy import Eq, Symbol
    >>> from sympy.solvers.inequalities import _solve_inequality as f
    >>> from sympy.abc import x, y

    For linear expressions, the symbol can be isolated:

    >>> f(x - 2 < 0, x)
    x < 2
    >>> f(-x - 6 < x, x)
    x > -3

    Sometimes nonlinear relationships will be False

    >>> f(x**2 + 4 < 0, x)
    False

    Or they may involve more than one region of values:

    >>> f(x**2 - 4 < 0, x)
    (-2 < x) & (x < 2)

    To restrict the solution to a relational, set linear=True
    and only the x-dependent portion will be isolated on the left:

    >>> f(x**2 - 4 < 0, x, linear=True)
    x**2 < 4

    Division of only nonzero quantities is allowed, so x cannot
    be isolated by dividing by y:

    >>> y.is_nonzero is None  # it is unknown whether it is 0 or not
    True
    >>> f(x*y < 1, x)
    x*y < 1

    And while an equality (or inequality) still holds after dividing by a
    non-zero quantity

    >>> nz = Symbol('nz', nonzero=True)
    >>> f(Eq(x*nz, 1), x)
    Eq(x, 1/nz)

    the sign must be known for other inequalities involving > or <:

    >>> f(x*nz <= 1, x)
    nz*x <= 1
    >>> p = Symbol('p', positive=True)
    >>> f(x*p <= 1, x)
    x <= 1/p

    When there are denominators in the original expression that
    are removed by expansion, conditions for them will be returned
    as part of the result:

    >>> f(x < x*(2/x - 1), x)
    (x < 1) & Ne(x, 0)
    """
    from sympy.solvers.solvers import denoms
    if s not in ie.free_symbols:
        return ie
    if ie.rhs == s:
        ie = ie.reversed
    if ie.lhs == s and s not in ie.rhs.free_symbols:
        return ie

    def classify(ie, s, i):
        # return True or False if ie evaluates when substituting s with
        # i else None (if unevaluated) or NaN (when there is an error
        # in evaluating)
        try:
            v = ie.subs(s, i)
            if v is S.NaN:
                return v
            elif v not in (True, False):
                return
            return v
        except TypeError:
            return S.NaN

    rv = None
    oo = S.Infinity
    expr = ie.lhs - ie.rhs
    try:
        p = Poly(expr, s)
        if p.degree() == 0:
            rv = ie.func(p.as_expr(), 0)
        elif not linear and p.degree() > 1:
            # handle in except clause
            raise NotImplementedError
    except (PolynomialError, NotImplementedError):
        if not linear:
            try:
                rv = reduce_rational_inequalities([[ie]], s)
            except PolynomialError:
                rv = solve_univariate_inequality(ie, s)
            # remove restrictions wrt +/-oo that may have been
            # applied when using sets to simplify the relationship
            okoo = classify(ie, s, oo)
            if okoo is S.true and classify(rv, s, oo) is S.false:
                rv = rv.subs(s < oo, True)
            oknoo = classify(ie, s, -oo)
            if (oknoo is S.true and
                    classify(rv, s, -oo) is S.false):
                rv = rv.subs(-oo < s, True)
                rv = rv.subs(s > -oo, True)
            if rv is S.true:
                rv = (s <= oo) if okoo is S.true else (s < oo)
                if oknoo is not S.true:
                    rv = And(-oo < s, rv)
        else:
            p = Poly(expr)

    conds = []
    if rv is None:
        e = p.as_expr()  # this is in expanded form
        # Do a safe inversion of e, moving non-s terms
        # to the rhs and dividing by a nonzero factor if
        # the relational is Eq/Ne; for other relationals
        # the sign must also be positive or negative
        rhs = 0
        b, ax = e.as_independent(s, as_Add=True)
        e -= b
        rhs -= b
        ef = factor_terms(e)
        a, e = ef.as_independent(s, as_Add=False)
        if (a.is_zero != False or  # don't divide by potential 0
                a.is_negative ==
                a.is_positive is None and  # if sign is not known then
                ie.rel_op not in ('!=', '==')): # reject if not Eq/Ne
            e = ef
            a = S.One
        rhs /= a
        if a.is_positive:
            rv = ie.func(e, rhs)
        else:
            rv = ie.reversed.func(e, rhs)

        # return conditions under which the value is
        # valid, too.
        beginning_denoms = denoms(ie.lhs) | denoms(ie.rhs)
        current_denoms = denoms(rv)
        for d in beginning_denoms - current_denoms:
            c = _solve_inequality(Eq(d, 0), s, linear=linear)
            if isinstance(c, Eq) and c.lhs == s:
                if classify(rv, s, c.rhs) is S.true:
                    # rv is permitting this value but it shouldn't
                    conds.append(~c)
        for i in (-oo, oo):
            if (classify(rv, s, i) is S.true and
                    classify(ie, s, i) is not S.true):
                conds.append(s < i if i is oo else i < s)

    conds.append(rv)
    return And(*conds)


def _reduce_inequalities(inequalities, symbols):
    # helper for reduce_inequalities

    poly_part, abs_part = {}, {}
    other = []

    for inequality in inequalities:

        expr, rel = inequality.lhs, inequality.rel_op  # rhs is 0

        # check for gens using atoms which is more strict than free_symbols to
        # guard against EX domain which won't be handled by
        # reduce_rational_inequalities
        gens = expr.atoms(Symbol)

        if len(gens) == 1:
            gen = gens.pop()
        else:
            common = expr.free_symbols & symbols
            if len(common) == 1:
                gen = common.pop()
                other.append(_solve_inequality(Relational(expr, 0, rel), gen))
                continue
            else:
                raise NotImplementedError(filldedent('''
                    inequality has more than one symbol of interest.
                    '''))

        if expr.is_polynomial(gen):
            poly_part.setdefault(gen, []).append((expr, rel))
        else:
            components = expr.find(lambda u:
                u.has(gen) and (
                u.is_Function or u.is_Pow and not u.exp.is_Integer))
            if components and all(isinstance(i, Abs) for i in components):
                abs_part.setdefault(gen, []).append((expr, rel))
            else:
                other.append(_solve_inequality(Relational(expr, 0, rel), gen))

    poly_reduced = [reduce_rational_inequalities([exprs], gen) for gen, exprs in poly_part.items()]
    abs_reduced = [reduce_abs_inequalities(exprs, gen) for gen, exprs in abs_part.items()]

    return And(*(poly_reduced + abs_reduced + other))


def reduce_inequalities(inequalities, symbols=[]):
    """Reduce a system of inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import reduce_inequalities

    >>> reduce_inequalities(0 <= x + 3, [])
    (-3 <= x) & (x < oo)

    >>> reduce_inequalities(0 <= x + y*2 - 1, [x])
    (x < oo) & (x >= 1 - 2*y)
    """
    if not iterable(inequalities):
        inequalities = [inequalities]
    inequalities = [sympify(i) for i in inequalities]

    gens = set().union(*[i.free_symbols for i in inequalities])

    if not iterable(symbols):
        symbols = [symbols]
    symbols = (set(symbols) or gens) & gens
    if any(i.is_extended_real is False for i in symbols):
        raise TypeError(filldedent('''
            inequalities cannot contain symbols that are not real.
            '''))

    # make vanilla symbol real
    recast = {i: Dummy(i.name, extended_real=True)
        for i in gens if i.is_extended_real is None}
    inequalities = [i.xreplace(recast) for i in inequalities]
    symbols = {i.xreplace(recast) for i in symbols}

    # prefilter
    keep = []
    for i in inequalities:
        if isinstance(i, Relational):
            i = i.func(i.lhs.as_expr() - i.rhs.as_expr(), 0)
        elif i not in (True, False):
            i = Eq(i, 0)
        if i == True:
            continue
        elif i == False:
            return S.false
        if i.lhs.is_number:
            raise NotImplementedError(
                "could not determine truth value of %s" % i)
        keep.append(i)
    inequalities = keep
    del keep

    # solve system
    rv = _reduce_inequalities(inequalities, symbols)

    # restore original symbols and return
    return rv.xreplace({v: k for k, v in recast.items()})

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_t5.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

import numpy as np
from fusion_attention import AttentionMask, FusionAttention
from fusion_base import Fusion
from fusion_simplified_layernorm import FusionSimplifiedLayerNormalization, FusionSkipSimplifiedLayerNormalization
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class FusionT5Attention(FusionAttention):
    """
    Fuse T5 Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        super().__init__(
            model,
            hidden_size,
            num_heads,
            attention_mask,
            use_multi_head_attention=False,
            search_op_types=["Softmax"],
        )
        self.static_kv = 1

    def make_attention_node(
        self,
        mask_index: str | None,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        attn_bias: str | None,
        scale: float,
    ) -> NodeProto | None:
        """Create an Attention node.
        Args:
            mask_index (str): mask input
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name
        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])

        if q_weight is None or k_weight is None or v_weight is None:
            matmul = q_matmul if q_weight is None else k_matmul if k_weight is None else v_matmul
            print(
                f"{matmul.input[1]} is not an initializer. "
                "Please set do_constant_folding=True in torch.onnx.export to unblock attention fusion"
            )
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        # assert q and k have same shape as expected
        assert qw.shape == kw.shape

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]

        assert qw_in_size == kw_in_size == vw_in_size

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                f"Input hidden size ({hidden_size}) is not same as weight matrix dimension of q,k,v ({qw_in_size}). "
                "Please provide a correct input hidden size or pass in 0"
            )

        qw_out_size = np.prod(qw.shape[1:])
        qkv_weight = np.stack((qw, kw, vw), axis=1)
        qkv_weight_dim = 3 * qw_out_size

        attention_node_name = self.model.create_node_name("Attention")

        weight = helper.make_tensor(
            name=attention_node_name + "_qkv_weight",
            data_type=TensorProto.FLOAT,
            dims=[qw_in_size, qkv_weight_dim],
            vals=qkv_weight.tobytes(),
            raw=True,
        )

        self.model.add_initializer(weight, self.this_graph_name)

        attention_inputs = [
            input,
            attention_node_name + "_qkv_weight",
            "",
        ]
        if mask_index:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")

        if attn_bias:
            attention_inputs.append("")  # no past
            attention_inputs.append(attn_bias)

        while attention_inputs and attention_inputs[-1] == "":
            attention_inputs.pop()

        attention_node = helper.make_node(
            "Attention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        if scale is not None:
            attention_node.attribute.extend([helper.make_attribute("scale", scale)])

        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        return attention_node

    def create_mha_node(
        self,
        query: str,
        key: str,
        value: str,
        mask_index: str | None,
        attn_bias: str | None,
        past_key: str | None,
        past_value: str | None,
        output: str,
        present_key: str | None,
        present_value: str | None,
        num_heads: int,
        hidden_size: int,
    ) -> NodeProto | None:
        assert num_heads > 0 and hidden_size > 0 and query and key and value

        if (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        attention_node_name = self.model.create_node_name("MultiHeadAttention")
        attention_inputs = [
            query,
            key,
            value,
            "",  # bias
        ]

        if mask_index:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")

        if attn_bias:
            attention_inputs.append(attn_bias)
        else:
            attention_inputs.append("")

        if past_key:
            assert past_value
            attention_inputs.append(past_key)
            attention_inputs.append(past_value)

        while attention_inputs and attention_inputs[-1] == "":
            attention_inputs.pop()

        attention_outputs = [output]
        if present_key:
            assert present_value
            attention_outputs.append(present_key)
            attention_outputs.append(present_value)

        print(f"{attention_inputs=}, {attention_outputs=}, {attention_node_name=}")
        attention_node = helper.make_node(
            "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=attention_outputs,
            name=attention_node_name,
        )

        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        attention_node.attribute.extend([helper.make_attribute("scale", 1.0)])
        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        self.increase_counter("MultiHeadAttention")
        return attention_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if self.fuse_t5_encoder(node, input_name_to_nodes, output_name_to_node):
            return

        self.fuse_t5_decoder(node, input_name_to_nodes, output_name_to_node)

    def fuse_t5_encoder(self, softmax_node, input_name_to_nodes, output_name_to_node):
        assert softmax_node.op_type == "Softmax"
        qkv_nodes = self.model.match_child_path(
            softmax_node,
            ["MatMul", "Transpose", "Reshape"],
            edges=[(0, 0), (0, 0), (0, 0)],
            input_name_to_nodes=input_name_to_nodes,
        )
        if qkv_nodes is None:
            return False
        matmul_qkv, _, reshape_qkv = qkv_nodes

        qkv_shape_nodes = self.model.match_parent_path(
            reshape_qkv,
            ["Concat", "Unsqueeze", "Gather", "Shape"],
            [1, 0, 0, 0],
            output_name_to_node,
        )
        if qkv_shape_nodes is None:
            return False
        input_shape_node = qkv_shape_nodes[-1]

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "MatMul"],
            [1, 0, 0],
            output_name_to_node,
        )
        if v_nodes is None:
            return False
        _, reshape_v, matmul_v = v_nodes
        # todo: check reshape_v parent nodes

        qk_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "Add", "MatMul"],
            [0, 0, 0],
            output_name_to_node,
        )
        if qk_nodes is None:
            return False
        _, add_qk, matmul_qk = qk_nodes

        mask_nodes = self.model.match_parent_path(
            add_qk,
            ["Add", "Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
            [1, 1, 0, 1, 0, 0],
            output_name_to_node,
        )

        is_pattern_for_one_graph_input = mask_nodes is None
        if mask_nodes is not None:
            mul_node = mask_nodes[1]
        else:
            # Pattern for SD3 and Flux.
            mask_nodes = self.model.match_parent_path(
                add_qk,
                ["Add", "Slice", "Mul", "Sub", "Unsqueeze", "Unsqueeze"],
                [1, 1, 0, 0, 1, 0],
                output_name_to_node,
            )

            # If the model is not optimized by ORT, there might be an additional Cast node.
            if mask_nodes is None:
                mask_nodes = self.model.match_parent_path(
                    add_qk,
                    ["Add", "Slice", "Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
                    [1, 1, 0, 0, 1, 0, 0],
                    output_name_to_node,
                )
                if mask_nodes is None:
                    return False
            mul_node = mask_nodes[2]

        _, mul_val = self.model.get_constant_input(mul_node)
        if mul_val is None:
            return False

        if mul_val != -10000:
            self.mask_filter_value = float(mul_val)

        # If the mask is derived from shape of input_ids, it means there is no padding mask.
        mask_nodes_2 = self.model.match_parent_path(
            mask_nodes[-1],
            ["ConstantOfShape", "Concat", "Unsqueeze", "Gather", "Shape"],
            [0, 0, 0, 0, 0],
            output_name_to_node,
        )
        mask_nodes_3 = self.model.match_parent_path(
            mask_nodes[-1],
            ["ConstantOfShape", "Concat", "Unsqueeze", "Gather", "Shape"],
            [0, 0, 1, 0, 0],
            output_name_to_node,
        )
        if (
            mask_nodes_2 is not None
            and any(input.name == mask_nodes_2[-1].input[0] for input in self.model.graph().input)
            and mask_nodes_3 is not None
            and mask_nodes_2[-1].input[0] == mask_nodes_3[-1].input[0]
            and len(mask_nodes_2[1].input) == 2
        ):
            mask_index = ""
        else:
            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])

        res_pos_bias = None
        rpb_nodes = self.model.match_parent_path(
            add_qk,
            ["Add", "RelativePositionBias"],
            [1, 0],
        )
        if rpb_nodes is None and is_pattern_for_one_graph_input:
            # Pattern for SD3 and Flux.
            rpb_nodes = self.model.match_parent_path(
                add_qk,
                ["Add", "Slice", "RelativePositionBias"],
                [1, 0, 0],
            )
        if rpb_nodes is None:
            return False

        res_pos_bias = rpb_nodes[-1].output[0]

        k_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "MatMul"],
            [1, 0, 0],
        )
        if k_nodes is None:
            return False
        _, _, matmul_k = k_nodes
        # todo: check reshape_k parent nodes

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "MatMul"],
            [0, 0, 0],
        )
        if q_nodes is None:
            return False

        _, reshape_q, matmul_q = q_nodes
        # todo: check reshape_q parent nodes

        if matmul_q.input[0] != input_shape_node.input[0]:
            return False

        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)

        new_node = self.make_attention_node(
            mask_index,
            matmul_q,
            matmul_k,
            matmul_v,
            num_heads=q_num_heads,
            hidden_size=q_hidden_size,
            input=input_shape_node.input[0],
            output=reshape_qkv.output[0],
            attn_bias=res_pos_bias,
            scale=1.0,
        )
        if new_node is None:
            return False

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.append(reshape_qkv)
        self.prune_graph = True
        return True

    def fuse_t5_decoder(self, softmax_node, input_name_to_nodes, output_name_to_node):
        assert softmax_node.op_type == "Softmax"

        qkv_nodes = self.model.match_child_path(
            softmax_node,
            ["MatMul", "Transpose", "Reshape"],
            edges=[(0, 0), (0, 0), (0, 0)],
            input_name_to_nodes=input_name_to_nodes,
        )
        if qkv_nodes is None:
            return
        matmul_qkv, _transpose_qkv, reshape_qkv = qkv_nodes

        qkv_shape_nodes = self.model.match_parent_path(
            reshape_qkv,
            ["Concat", "Unsqueeze", "Gather", "Shape"],
            [1, 0, 0, 0],
        )
        if qkv_shape_nodes is None:
            return
        input_shape_node = qkv_shape_nodes[-1]

        value = None
        past_value = None
        present_value = None
        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Concat", "Transpose", "Reshape", "MatMul"],
            [1, 1, 0, 0],
        )
        if v_nodes is None:
            v_nodes = self.model.match_parent_path(
                matmul_qkv,
                ["Transpose", "Reshape", "MatMul"],
                [1, 0, 0],
            )
            if v_nodes is not None:
                transpose_v, reshape_v, matmul_v = v_nodes
                value = reshape_v.input[0]
                present_value = transpose_v.output[0]
                if "present_value" not in present_value:
                    return
                if matmul_v.input[0] != input_shape_node.input[0]:
                    self.static_kv = 1
                else:
                    self.static_kv = 0
            else:
                past_value = matmul_qkv.input[1]
                if past_value in output_name_to_node:
                    return
                if "past_value_cross" not in past_value:
                    return
                self.static_kv = 1
        else:
            concat_v, _, reshape_v, _ = v_nodes
            past_value = concat_v.input[0]
            if past_value in output_name_to_node:
                return
            if "past_value_self" not in past_value:
                return
            present_value = concat_v.output[0]
            if "present_value_self" not in present_value:
                return
            value = reshape_v.input[0]
            self.static_kv = 0

        qk_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "Add", "MatMul"],
            [0, 0, 0],
        )
        if qk_nodes is None:
            return
        _, add_qk, matmul_qk = qk_nodes

        mask_index = None
        res_pos_bias = None
        if self.static_kv == 1:
            mask_nodes = self.model.match_parent_path(
                add_qk,
                ["Add", "Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
                [1, 1, 0, 1, 0, 0],
            )
            if mask_nodes is not None:
                mul_node = mask_nodes[1]
            else:
                mask_nodes = self.model.match_parent_path(
                    add_qk,
                    ["Add", "Slice", "Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
                    [1, 1, 0, 0, 1, 0, 0],
                )
                if mask_nodes is None:
                    return
                mul_node = mask_nodes[2]

            _, mul_val = self.model.get_constant_input(mul_node)
            if mul_val != -10000:
                self.mask_filter_value = mul_val

            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])
        else:
            matched_path_index, _, _ = self.model.match_parent_paths(
                add_qk,
                [
                    (["Add", "Slice"], [1, 0]),
                    (["Add", "RelativePositionBias"], [1, 0]),
                ],
                output_name_to_node,
            )
            if matched_path_index < 0:
                logger.debug("Skip MultiHeadAttention fusion since attention bias pattern not matched")
                return

            res_pos_bias = add_qk.input[1]

        key = None
        past_key = None
        present_key = None
        if self.static_kv == 1:
            k_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Transpose", "Reshape", "MatMul"],
                [1, 0, 0],
            )
            if k_nodes is not None:
                transpose_k, reshape_k, _ = k_nodes
                key = reshape_k.input[0]
                present_key_transpose_nodes = input_name_to_nodes[reshape_k.output[0]]
                for present_key_transpose_node in present_key_transpose_nodes:
                    present_key_candidate = self.model.find_graph_output(present_key_transpose_node.output[0])
                    if present_key_candidate is not None:
                        present_key = present_key_candidate.name
                        break
                if present_key is None:
                    return
                if "present_key_cross" not in present_key:
                    return
            else:
                k_nodes = self.model.match_parent_path(
                    matmul_qk,
                    ["Transpose"],
                    [1],
                )
                if k_nodes is None:
                    return
                transpose_k = k_nodes[0]

                past_key = transpose_k.input[0]
                if past_key in output_name_to_node:
                    return
                if "past_key_cross" not in past_key:
                    return
        else:
            idx, k_nodes, _ = self.model.match_parent_paths(
                matmul_qk,
                [
                    (["Transpose", "Concat", "Reshape", "MatMul"], [1, 0, 1, 0]),
                    (["Transpose", "Concat", "Transpose", "Reshape", "MatMul"], [1, 0, 1, 0, 0]),
                ],
                output_name_to_node,
            )
            past_key_transpose_node = None
            present_key_transpose_nodes = None
            if k_nodes is not None:
                concat_k, reshape_k = k_nodes[1], k_nodes[-2]
                key = reshape_k.input[0]

                if idx == 0:
                    past_key_transpose_node = output_name_to_node[concat_k.input[0]]
                    past_key = past_key_transpose_node.input[0]
                else:
                    past_key = concat_k.input[0]
                if past_key in output_name_to_node:
                    return
                if "past_key_self" not in past_key:
                    return

                if idx == 0:
                    present_key_transpose_nodes = input_name_to_nodes[concat_k.output[0]]
                    for present_key_transpose_node in present_key_transpose_nodes:
                        present_key_candidate = self.model.find_graph_output(present_key_transpose_node.output[0])
                        if present_key_candidate is not None:
                            present_key = present_key_candidate.name
                            break
                else:
                    present_key = concat_k.output[0]
                if present_key is None:
                    return
                if "present_key_self" not in present_key:
                    return
            else:
                k_nodes = self.model.match_parent_path(
                    matmul_qk,
                    ["Transpose", "Reshape", "MatMul"],
                    [1, 0, 0],
                )
                if k_nodes is None:
                    return
                _, reshape_k, _ = k_nodes
                key = reshape_k.input[0]
                present_key_transpose_nodes = input_name_to_nodes[reshape_k.output[0]]
                for present_key_transpose_node in present_key_transpose_nodes:
                    present_key_candidate = self.model.find_graph_output(present_key_transpose_node.output[0])
                    if present_key_candidate is not None:
                        present_key = present_key_candidate.name
                        break
                if present_key is None:
                    return
                if "present_key_self" not in present_key:
                    return

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "MatMul"],
            [0, 0, 0],
        )
        if q_nodes is None:
            return

        transpose_q, reshape_q, matmul_q = q_nodes

        if matmul_q.input[0] != input_shape_node.input[0]:
            return

        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)

        if self.static_kv == 1 and past_key is not None:
            key = past_key
            value = past_value
            past_key = None
            past_value = None

        if not (key and value and q_num_heads > 0 and q_hidden_size > 0):
            return

        new_node = self.create_mha_node(
            query=matmul_q.output[0],
            key=key,
            value=value,
            mask_index=mask_index,
            attn_bias=res_pos_bias,
            past_key=past_key,
            past_value=past_value,
            output=reshape_qkv.output[0],
            present_key=present_key,
            present_value=present_value,
            num_heads=q_num_heads,
            hidden_size=q_hidden_size,
        )

        if new_node:
            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            # Since present_* is graph output, we need update the graph to avoid circular.
            if present_key or present_value:
                for graph_output in [present_key, present_value]:
                    if not (graph_output and self.model.find_graph_output(graph_output)):
                        print(f"{graph_output=} does not exist in graph output")
                        return
                    assert graph_output in output_name_to_node
                    output_name_to_node[graph_output].output[0] = graph_output + "_copy"
                    self.model.replace_input_of_all_nodes(graph_output, graph_output + "_copy")

            self.nodes_to_remove.append(reshape_qkv)
            self.prune_graph = False


class FusionRelativePositionBiasBlock(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "RelativePositionBias", ["Softmax"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        compute_bias_nodes = self.model.match_parent_path(
            node,
            ["Add", "Add", "Slice", "Unsqueeze", "Transpose", "Gather", "Where"],
            [0, 1, 0, 0, 0, 0, 1],
            output_name_to_node,
        )

        if compute_bias_nodes is None:
            compute_bias_nodes = self.model.match_parent_path(
                node,
                ["Add", "Add", "Slice", "Unsqueeze", "Transpose", "Gather", "Add", "Where"],
                [0, 1, 0, 0, 0, 0, 1, 1],
                output_name_to_node,
            )
            if compute_bias_nodes is None:
                return

        gather = compute_bias_nodes[5]
        where = compute_bias_nodes[-1]
        slice = compute_bias_nodes[2]
        unsqueeze = compute_bias_nodes[3]

        # Current fusion will not remove the node until the graph is processed.
        # This avoids to fuse it again when it is shared by multiple layers.
        if unsqueeze in self.nodes_to_remove:
            return

        compute_buckets_nodes = self.model.match_parent_path(
            where,
            ["Min", "ConstantOfShape", "Shape", "Add", "Cast", "Mul", "Div", "Log", "Div"],
            [2, 1, 0, 0, 0, 0, 0, 0, 0],
            output_name_to_node,
        )
        if compute_buckets_nodes is None:
            return

        # This value is to used to compute max_distance later.
        log_max = self.model.get_constant_value(compute_buckets_nodes[-3].input[1])

        div = compute_buckets_nodes[-1]

        range_nodes = self.model.match_parent_path(
            div,
            ["Cast", "Neg", "Min", "ConstantOfShape", "Shape", "Sub", "Unsqueeze", "Range"],
            [0, 0, 0, 1, 0, 0, 0, 0],
            output_name_to_node,
        )

        is_bidirectional = False
        if range_nodes is None:
            range_nodes = self.model.match_parent_path(
                div, ["Cast", "Abs", "Sub", "Unsqueeze", "Range"], [0, 0, 0, 0, 0], output_name_to_node
            )
            is_bidirectional = True
            if range_nodes is None:
                return
        range_node = range_nodes[-1]

        # Double check that the constant relative to max_distance and relative_attention_num_buckets.
        # Most t5 models use max_distance=128, so we hardcode it unitl we see a model with different value.

        # The log_max is the value of the following formula:
        #   math.log(max_distance / (relative_attention_num_buckets // (4 if is_bidirectional else 2)))
        # See https://github.com/huggingface/transformers/blob/608e163b527eaee41e650ffb9eb4c422d2679902/src/transformers/models/t5/modeling_t5.py#L397.
        # Here is the value based on max_distance=128 and relative_attention_num_buckets=32:
        max_distance = int(np.round(np.exp(log_max) * (32 // (4 if is_bidirectional else 2))))
        if max_distance != 128:
            logger.warning(
                f"max_distance is {max_distance}, which is different from the default value 128. "
                "Please double check the model configuration."
            )

        node_name = self.model.create_node_name(
            "RelativePositionBias", name_prefix="RelPosBias_" + ("encoder" if is_bidirectional else "decoder")
        )

        table_weight_i = self.model.get_initializer(gather.input[0])
        if table_weight_i is None:
            return
        table_weight = NumpyHelper.to_array(table_weight_i)
        table_weight_t = np.transpose(table_weight)
        bias_table = helper.make_tensor(
            name=node_name + "_bias_table_weight",
            data_type=TensorProto.FLOAT,
            dims=[np.shape(table_weight)[0], np.shape(table_weight)[1]],
            vals=table_weight_t.tobytes(),
            raw=True,
        )
        self.model.add_initializer(bias_table, self.this_graph_name)

        # Relative position is like the following in encoder:
        #                seq_len
        #                   |
        #                Range(0, *)
        #                /      \
        #   Unsqueeze(axes=0)    Unsqueeze(axes=1)
        #                \    /
        #                  Sub
        #                   |
        #                  Abs
        #
        # Relative position is like the following in decoder:
        #       past_seq_len   seq_len
        #                 \    /
        #                  Add
        #                /      \
        #        Range(0, *)    Range(0, *)
        #                \    /
        #                  Sub
        # Note that the graph will slice the attention bias to get last seq_len rows.
        #
        # In new version of transformers, the pattern of decoder is changed like the following
        #
        #      total_seq_len    Range(start=past_seq_len, end=total_seq_len)
        #              |              |
        #          Range(0, *)   Unsqueeze(axes=1)
        #              |              |
        #    Unsqueeze(axes=0)    Cast(to=int64)
        #                   \     /
        #                     Sub
        # Currently, there is still Slice to get last seq_len rows so end result is same.
        # But need to be careful that the shape of bias tensor is changed before Slice.
        #
        # RelativePositionBias operator requires query_length == key_length so we shall pass in total_seq_len.
        # Here we get the end value of the Range node as length to pass to the RelativePositionBias node.

        # TODO: Optimization opportunity: change RelativePositionBias op to support query_length != key_length.
        #       only compute seq_len rows, then we can remove the Slice after the RelativePositionBias node.
        inputs = [bias_table.name, range_node.input[1], range_node.input[1]]

        # Use a new tensor name since the shape might be different as mentioned above.
        bias_output = node_name + "_rel_pos_bias"
        slice.input[0] = bias_output

        rpb_node = helper.make_node(
            "RelativePositionBias",
            inputs=inputs,
            outputs=[bias_output],
            name=node_name,
        )
        rpb_node.domain = "com.microsoft"
        rpb_node.attribute.extend([helper.make_attribute("max_distance", max_distance)])
        rpb_node.attribute.extend([helper.make_attribute("is_bidirectional", is_bidirectional)])
        self.node_name_to_graph_name[rpb_node.name] = self.this_graph_name
        self.nodes_to_add.append(rpb_node)
        self.prune_graph = True


class T5OnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads: int = 0, hidden_size: int = 0):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)

        # When the model has only one input (input_ids), there is no padding mask.
        if len(self.model.graph.input) == 1:
            from fusion_options import AttentionMaskFormat

            self.attention_mask.mask_format = AttentionMaskFormat.NoMask

        self.attention_fusion = FusionT5Attention(self, self.hidden_size, self.num_heads, self.attention_mask)
        self.layer_norm_fusion = FusionSimplifiedLayerNormalization(self)
        self.skip_layer_norm_fusion = FusionSkipSimplifiedLayerNormalization(self)
        self.rpb_fusion = FusionRelativePositionBiasBlock(self)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def fuse_layer_norm(self):
        self.layer_norm_fusion.apply()

    def fuse_skip_layer_norm(self, shape_infer=True):
        self.skip_layer_norm_fusion.apply()

    def adjust_rel_pos_bis_length_input(self):
        # For T5 encoder, it uses complex logic to compute the query and key length when there is only one graph input (input_ids)
        # We can directly get the length from shape (the 2nd dimension) of input_ids.
        for node in self.nodes():
            if node.op_type == "RelativePositionBias":
                nodes = self.match_parent_path(
                    node,
                    [
                        "Gather",
                        "Shape",
                        "Transpose",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "SimplifiedLayerNormalization",
                        "Gather",
                    ],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                )
                # TODO: more validation on node attributes
                if nodes is not None:
                    graph_input_names = [input.name for input in self.model.graph.input]
                    if nodes[-1].input[1] in graph_input_names:
                        node_name = self.create_node_name("Shape", name_prefix="Added_Shape_")
                        shape_node = helper.make_node(
                            "Shape",
                            inputs=[nodes[-1].input[1]],
                            outputs=[node_name + "_Output"],
                            name=node_name,
                        )

                        indices_1 = helper.make_tensor(
                            name="Constant_Index_1",
                            data_type=TensorProto.INT64,
                            dims=[1],  # Shape of the tensor
                            vals=[1],  # Tensor values
                        )
                        self.add_initializer(indices_1)

                        gather = helper.make_node(
                            "Gather",
                            inputs=[node_name + "_Output", "Constant_Index_1"],
                            outputs=[node_name + "_Output_Gather_1"],
                            name=self.create_node_name("Gather", name_prefix="Added_Gather_"),
                            axis=0,
                        )

                        self.add_node(shape_node)
                        self.add_node(gather)
                        node.input[1] = node_name + "_Output_Gather_1"
                        node.input[2] = node_name + "_Output_Gather_1"

                break

    # Remove get_extended_attention_mask() since it generates all zeros.
    def remove_extended_mask_decoder_init(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Add":
                extended_mask_nodes = self.match_parent_path(
                    node,
                    [
                        "Mul",
                        "Sub",
                        "Mul",
                        "Unsqueeze",
                        "Cast",
                        "LessOrEqual",
                        "Tile",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                )
                if extended_mask_nodes is None:
                    continue

                rpb_nodes = self.match_parent_path(node, ["RelativePositionBias"], [0])
                if rpb_nodes is None:
                    continue

                rpb_node = rpb_nodes[0]
                rpb_node.output[0] = node.output[0]

                nodes_to_remove.extend(extended_mask_nodes)
                nodes_to_remove.append(node)
                self.remove_nodes(nodes_to_remove)

    def remove_extended_mask_decoder(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Add":
                extended_mask_nodes = self.match_parent_path(
                    node,
                    [
                        "Mul",
                        "Sub",
                        "Mul",
                        "Unsqueeze",
                        "Concat",
                        "Cast",
                        "LessOrEqual",
                        "Tile",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                )
                if extended_mask_nodes is None:
                    continue

                rpb_nodes = self.match_parent_path(node, ["Slice", "RelativePositionBias"], [0, 0])
                if rpb_nodes is None:
                    continue

                rpb_node = rpb_nodes[0]
                rpb_node.output[0] = node.output[0]

                nodes_to_remove.extend(extended_mask_nodes)
                nodes_to_remove.append(node)
                self.remove_nodes(nodes_to_remove)

    def preprocess(self):
        self.adjust_reshape_and_expand()
        self.rpb_fusion.apply()

    def postprocess(self):
        # remove get_extended_attention_mask() since it generates all zeros.
        self.remove_extended_mask_decoder_init()
        self.remove_extended_mask_decoder()
        self.adjust_rel_pos_bis_length_input()

        self.prune_graph()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\categories\baseclasses.py ===
from sympy.core import S, Basic, Dict, Symbol, Tuple, sympify
from sympy.core.symbol import Str
from sympy.sets import Set, FiniteSet, EmptySet
from sympy.utilities.iterables import iterable


class Class(Set):
    r"""
    The base class for any kind of class in the set-theoretic sense.

    Explanation
    ===========

    In axiomatic set theories, everything is a class.  A class which
    can be a member of another class is a set.  A class which is not a
    member of another class is a proper class.  The class `\{1, 2\}`
    is a set; the class of all sets is a proper class.

    This class is essentially a synonym for :class:`sympy.core.Set`.
    The goal of this class is to assure easier migration to the
    eventual proper implementation of set theory.
    """
    is_proper = False


class Object(Symbol):
    """
    The base class for any kind of object in an abstract category.

    Explanation
    ===========

    While technically any instance of :class:`~.Basic` will do, this
    class is the recommended way to create abstract objects in
    abstract categories.
    """


class Morphism(Basic):
    """
    The base class for any morphism in an abstract category.

    Explanation
    ===========

    In abstract categories, a morphism is an arrow between two
    category objects.  The object where the arrow starts is called the
    domain, while the object where the arrow ends is called the
    codomain.

    Two morphisms between the same pair of objects are considered to
    be the same morphisms.  To distinguish between morphisms between
    the same objects use :class:`NamedMorphism`.

    It is prohibited to instantiate this class.  Use one of the
    derived classes instead.

    See Also
    ========

    IdentityMorphism, NamedMorphism, CompositeMorphism
    """
    def __new__(cls, domain, codomain):
        raise(NotImplementedError(
            "Cannot instantiate Morphism.  Use derived classes instead."))

    @property
    def domain(self):
        """
        Returns the domain of the morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> f.domain
        Object("A")

        """
        return self.args[0]

    @property
    def codomain(self):
        """
        Returns the codomain of the morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> f.codomain
        Object("B")

        """
        return self.args[1]

    def compose(self, other):
        r"""
        Composes self with the supplied morphism.

        The order of elements in the composition is the usual order,
        i.e., to construct `g\circ f` use ``g.compose(f)``.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> g * f
        CompositeMorphism((NamedMorphism(Object("A"), Object("B"), "f"),
        NamedMorphism(Object("B"), Object("C"), "g")))
        >>> (g * f).domain
        Object("A")
        >>> (g * f).codomain
        Object("C")

        """
        return CompositeMorphism(other, self)

    def __mul__(self, other):
        r"""
        Composes self with the supplied morphism.

        The semantics of this operation is given by the following
        equation: ``g * f == g.compose(f)`` for composable morphisms
        ``g`` and ``f``.

        See Also
        ========

        compose
        """
        return self.compose(other)


class IdentityMorphism(Morphism):
    """
    Represents an identity morphism.

    Explanation
    ===========

    An identity morphism is a morphism with equal domain and codomain,
    which acts as an identity with respect to composition.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, IdentityMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> f = NamedMorphism(A, B, "f")
    >>> id_A = IdentityMorphism(A)
    >>> id_B = IdentityMorphism(B)
    >>> f * id_A == f
    True
    >>> id_B * f == f
    True

    See Also
    ========

    Morphism
    """
    def __new__(cls, domain):
        return Basic.__new__(cls, domain)

    @property
    def codomain(self):
        return self.domain


class NamedMorphism(Morphism):
    """
    Represents a morphism which has a name.

    Explanation
    ===========

    Names are used to distinguish between morphisms which have the
    same domain and codomain: two named morphisms are equal if they
    have the same domains, codomains, and names.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> f = NamedMorphism(A, B, "f")
    >>> f
    NamedMorphism(Object("A"), Object("B"), "f")
    >>> f.name
    'f'

    See Also
    ========

    Morphism
    """
    def __new__(cls, domain, codomain, name):
        if not name:
            raise ValueError("Empty morphism names not allowed.")

        if not isinstance(name, Str):
            name = Str(name)

        return Basic.__new__(cls, domain, codomain, name)

    @property
    def name(self):
        """
        Returns the name of the morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> f.name
        'f'

        """
        return self.args[2].name


class CompositeMorphism(Morphism):
    r"""
    Represents a morphism which is a composition of other morphisms.

    Explanation
    ===========

    Two composite morphisms are equal if the morphisms they were
    obtained from (components) are the same and were listed in the
    same order.

    The arguments to the constructor for this class should be listed
    in diagram order: to obtain the composition `g\circ f` from the
    instances of :class:`Morphism` ``g`` and ``f`` use
    ``CompositeMorphism(f, g)``.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, CompositeMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> g * f
    CompositeMorphism((NamedMorphism(Object("A"), Object("B"), "f"),
    NamedMorphism(Object("B"), Object("C"), "g")))
    >>> CompositeMorphism(f, g) == g * f
    True

    """
    @staticmethod
    def _add_morphism(t, morphism):
        """
        Intelligently adds ``morphism`` to tuple ``t``.

        Explanation
        ===========

        If ``morphism`` is a composite morphism, its components are
        added to the tuple.  If ``morphism`` is an identity, nothing
        is added to the tuple.

        No composability checks are performed.
        """
        if isinstance(morphism, CompositeMorphism):
            # ``morphism`` is a composite morphism; we have to
            # denest its components.
            return t + morphism.components
        elif isinstance(morphism, IdentityMorphism):
            # ``morphism`` is an identity.  Nothing happens.
            return t
        else:
            return t + Tuple(morphism)

    def __new__(cls, *components):
        if components and not isinstance(components[0], Morphism):
            # Maybe the user has explicitly supplied a list of
            # morphisms.
            return CompositeMorphism.__new__(cls, *components[0])

        normalised_components = Tuple()

        for current, following in zip(components, components[1:]):
            if not isinstance(current, Morphism) or \
                    not isinstance(following, Morphism):
                raise TypeError("All components must be morphisms.")

            if current.codomain != following.domain:
                raise ValueError("Uncomposable morphisms.")

            normalised_components = CompositeMorphism._add_morphism(
                normalised_components, current)

        # We haven't added the last morphism to the list of normalised
        # components.  Add it now.
        normalised_components = CompositeMorphism._add_morphism(
            normalised_components, components[-1])

        if not normalised_components:
            # If ``normalised_components`` is empty, only identities
            # were supplied.  Since they all were composable, they are
            # all the same identities.
            return components[0]
        elif len(normalised_components) == 1:
            # No sense to construct a whole CompositeMorphism.
            return normalised_components[0]

        return Basic.__new__(cls, normalised_components)

    @property
    def components(self):
        """
        Returns the components of this composite morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).components
        (NamedMorphism(Object("A"), Object("B"), "f"),
        NamedMorphism(Object("B"), Object("C"), "g"))

        """
        return self.args[0]

    @property
    def domain(self):
        """
        Returns the domain of this composite morphism.

        The domain of the composite morphism is the domain of its
        first component.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).domain
        Object("A")

        """
        return self.components[0].domain

    @property
    def codomain(self):
        """
        Returns the codomain of this composite morphism.

        The codomain of the composite morphism is the codomain of its
        last component.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).codomain
        Object("C")

        """
        return self.components[-1].codomain

    def flatten(self, new_name):
        """
        Forgets the composite structure of this morphism.

        Explanation
        ===========

        If ``new_name`` is not empty, returns a :class:`NamedMorphism`
        with the supplied name, otherwise returns a :class:`Morphism`.
        In both cases the domain of the new morphism is the domain of
        this composite morphism and the codomain of the new morphism
        is the codomain of this composite morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).flatten("h")
        NamedMorphism(Object("A"), Object("C"), "h")

        """
        return NamedMorphism(self.domain, self.codomain, new_name)


class Category(Basic):
    r"""
    An (abstract) category.

    Explanation
    ===========

    A category [JoyOfCats] is a quadruple `\mbox{K} = (O, \hom, id,
    \circ)` consisting of

    * a (set-theoretical) class `O`, whose members are called
      `K`-objects,

    * for each pair `(A, B)` of `K`-objects, a set `\hom(A, B)` whose
      members are called `K`-morphisms from `A` to `B`,

    * for a each `K`-object `A`, a morphism `id:A\rightarrow A`,
      called the `K`-identity of `A`,

    * a composition law `\circ` associating with every `K`-morphisms
      `f:A\rightarrow B` and `g:B\rightarrow C` a `K`-morphism `g\circ
      f:A\rightarrow C`, called the composite of `f` and `g`.

    Composition is associative, `K`-identities are identities with
    respect to composition, and the sets `\hom(A, B)` are pairwise
    disjoint.

    This class knows nothing about its objects and morphisms.
    Concrete cases of (abstract) categories should be implemented as
    classes derived from this one.

    Certain instances of :class:`Diagram` can be asserted to be
    commutative in a :class:`Category` by supplying the argument
    ``commutative_diagrams`` in the constructor.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram, Category
    >>> from sympy import FiniteSet
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g])
    >>> K = Category("K", commutative_diagrams=[d])
    >>> K.commutative_diagrams == FiniteSet(d)
    True

    See Also
    ========

    Diagram
    """
    def __new__(cls, name, objects=EmptySet, commutative_diagrams=EmptySet):
        if not name:
            raise ValueError("A Category cannot have an empty name.")

        if not isinstance(name, Str):
            name = Str(name)

        if not isinstance(objects, Class):
            objects = Class(objects)

        new_category = Basic.__new__(cls, name, objects,
                                     FiniteSet(*commutative_diagrams))
        return new_category

    @property
    def name(self):
        """
        Returns the name of this category.

        Examples
        ========

        >>> from sympy.categories import Category
        >>> K = Category("K")
        >>> K.name
        'K'

        """
        return self.args[0].name

    @property
    def objects(self):
        """
        Returns the class of objects of this category.

        Examples
        ========

        >>> from sympy.categories import Object, Category
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> K = Category("K", FiniteSet(A, B))
        >>> K.objects
        Class({Object("A"), Object("B")})

        """
        return self.args[1]

    @property
    def commutative_diagrams(self):
        """
        Returns the :class:`~.FiniteSet` of diagrams which are known to
        be commutative in this category.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram, Category
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> K = Category("K", commutative_diagrams=[d])
        >>> K.commutative_diagrams == FiniteSet(d)
        True

        """
        return self.args[2]

    def hom(self, A, B):
        raise NotImplementedError(
            "hom-sets are not implemented in Category.")

    def all_morphisms(self):
        raise NotImplementedError(
            "Obtaining the class of morphisms is not implemented in Category.")


class Diagram(Basic):
    r"""
    Represents a diagram in a certain category.

    Explanation
    ===========

    Informally, a diagram is a collection of objects of a category and
    certain morphisms between them.  A diagram is still a monoid with
    respect to morphism composition; i.e., identity morphisms, as well
    as all composites of morphisms included in the diagram belong to
    the diagram.  For a more formal approach to this notion see
    [Pare1970].

    The components of composite morphisms are also added to the
    diagram.  No properties are assigned to such morphisms by default.

    A commutative diagram is often accompanied by a statement of the
    following kind: "if such morphisms with such properties exist,
    then such morphisms which such properties exist and the diagram is
    commutative".  To represent this, an instance of :class:`Diagram`
    includes a collection of morphisms which are the premises and
    another collection of conclusions.  ``premises`` and
    ``conclusions`` associate morphisms belonging to the corresponding
    categories with the :class:`~.FiniteSet`'s of their properties.

    The set of properties of a composite morphism is the intersection
    of the sets of properties of its components.  The domain and
    codomain of a conclusion morphism should be among the domains and
    codomains of the morphisms listed as the premises of a diagram.

    No checks are carried out of whether the supplied object and
    morphisms do belong to one and the same category.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy import pprint, default_sort_key
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g])
    >>> premises_keys = sorted(d.premises.keys(), key=default_sort_key)
    >>> pprint(premises_keys, use_unicode=False)
    [g*f:A-->C, id:A-->A, id:B-->B, id:C-->C, f:A-->B, g:B-->C]
    >>> pprint(d.premises, use_unicode=False)
    {g*f:A-->C: EmptySet, id:A-->A: EmptySet, id:B-->B: EmptySet,
     id:C-->C: EmptySet, f:A-->B: EmptySet, g:B-->C: EmptySet}
    >>> d = Diagram([f, g], {g * f: "unique"})
    >>> pprint(d.conclusions,use_unicode=False)
    {g*f:A-->C: {unique}}

    References
    ==========

    [Pare1970] B. Pareigis: Categories and functors.  Academic Press, 1970.

    """
    @staticmethod
    def _set_dict_union(dictionary, key, value):
        """
        If ``key`` is in ``dictionary``, set the new value of ``key``
        to be the union between the old value and ``value``.
        Otherwise, set the value of ``key`` to ``value.

        Returns ``True`` if the key already was in the dictionary and
        ``False`` otherwise.
        """
        if key in dictionary:
            dictionary[key] = dictionary[key] | value
            return True
        else:
            dictionary[key] = value
            return False

    @staticmethod
    def _add_morphism_closure(morphisms, morphism, props, add_identities=True,
                              recurse_composites=True):
        """
        Adds a morphism and its attributes to the supplied dictionary
        ``morphisms``.  If ``add_identities`` is True, also adds the
        identity morphisms for the domain and the codomain of
        ``morphism``.
        """
        if not Diagram._set_dict_union(morphisms, morphism, props):
            # We have just added a new morphism.

            if isinstance(morphism, IdentityMorphism):
                if props:
                    # Properties for identity morphisms don't really
                    # make sense, because very much is known about
                    # identity morphisms already, so much that they
                    # are trivial.  Having properties for identity
                    # morphisms would only be confusing.
                    raise ValueError(
                        "Instances of IdentityMorphism cannot have properties.")
                return

            if add_identities:
                empty = EmptySet

                id_dom = IdentityMorphism(morphism.domain)
                id_cod = IdentityMorphism(morphism.codomain)

                Diagram._set_dict_union(morphisms, id_dom, empty)
                Diagram._set_dict_union(morphisms, id_cod, empty)

            for existing_morphism, existing_props in list(morphisms.items()):
                new_props = existing_props & props
                if morphism.domain == existing_morphism.codomain:
                    left = morphism * existing_morphism
                    Diagram._set_dict_union(morphisms, left, new_props)
                if morphism.codomain == existing_morphism.domain:
                    right = existing_morphism * morphism
                    Diagram._set_dict_union(morphisms, right, new_props)

            if isinstance(morphism, CompositeMorphism) and recurse_composites:
                # This is a composite morphism, add its components as
                # well.
                empty = EmptySet
                for component in morphism.components:
                    Diagram._add_morphism_closure(morphisms, component, empty,
                                                  add_identities)

    def __new__(cls, *args):
        """
        Construct a new instance of Diagram.

        Explanation
        ===========

        If no arguments are supplied, an empty diagram is created.

        If at least an argument is supplied, ``args[0]`` is
        interpreted as the premises of the diagram.  If ``args[0]`` is
        a list, it is interpreted as a list of :class:`Morphism`'s, in
        which each :class:`Morphism` has an empty set of properties.
        If ``args[0]`` is a Python dictionary or a :class:`Dict`, it
        is interpreted as a dictionary associating to some
        :class:`Morphism`'s some properties.

        If at least two arguments are supplied ``args[1]`` is
        interpreted as the conclusions of the diagram.  The type of
        ``args[1]`` is interpreted in exactly the same way as the type
        of ``args[0]``.  If only one argument is supplied, the diagram
        has no conclusions.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import IdentityMorphism, Diagram
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> IdentityMorphism(A) in d.premises.keys()
        True
        >>> g * f in d.premises.keys()
        True
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> d.conclusions[g * f]
        {unique}

        """
        premises = {}
        conclusions = {}

        # Here we will keep track of the objects which appear in the
        # premises.
        objects = EmptySet

        if len(args) >= 1:
            # We've got some premises in the arguments.
            premises_arg = args[0]

            if isinstance(premises_arg, list):
                # The user has supplied a list of morphisms, none of
                # which have any attributes.
                empty = EmptySet

                for morphism in premises_arg:
                    objects |= FiniteSet(morphism.domain, morphism.codomain)
                    Diagram._add_morphism_closure(premises, morphism, empty)
            elif isinstance(premises_arg, (dict, Dict)):
                # The user has supplied a dictionary of morphisms and
                # their properties.
                for morphism, props in premises_arg.items():
                    objects |= FiniteSet(morphism.domain, morphism.codomain)
                    Diagram._add_morphism_closure(
                        premises, morphism, FiniteSet(*props) if iterable(props) else FiniteSet(props))

        if len(args) >= 2:
            # We also have some conclusions.
            conclusions_arg = args[1]

            if isinstance(conclusions_arg, list):
                # The user has supplied a list of morphisms, none of
                # which have any attributes.
                empty = EmptySet

                for morphism in conclusions_arg:
                    # Check that no new objects appear in conclusions.
                    if ((sympify(objects.contains(morphism.domain)) is S.true) and
                        (sympify(objects.contains(morphism.codomain)) is S.true)):
                        # No need to add identities and recurse
                        # composites this time.
                        Diagram._add_morphism_closure(
                            conclusions, morphism, empty, add_identities=False,
                            recurse_composites=False)
            elif isinstance(conclusions_arg, (dict, Dict)):
                # The user has supplied a dictionary of morphisms and
                # their properties.
                for morphism, props in conclusions_arg.items():
                    # Check that no new objects appear in conclusions.
                    if (morphism.domain in objects) and \
                       (morphism.codomain in objects):
                        # No need to add identities and recurse
                        # composites this time.
                        Diagram._add_morphism_closure(
                            conclusions, morphism, FiniteSet(*props) if iterable(props) else FiniteSet(props),
                            add_identities=False, recurse_composites=False)

        return Basic.__new__(cls, Dict(premises), Dict(conclusions), objects)

    @property
    def premises(self):
        """
        Returns the premises of this diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import IdentityMorphism, Diagram
        >>> from sympy import pretty
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> id_A = IdentityMorphism(A)
        >>> id_B = IdentityMorphism(B)
        >>> d = Diagram([f])
        >>> print(pretty(d.premises, use_unicode=False))
        {id:A-->A: EmptySet, id:B-->B: EmptySet, f:A-->B: EmptySet}

        """
        return self.args[0]

    @property
    def conclusions(self):
        """
        Returns the conclusions of this diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import IdentityMorphism, Diagram
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> IdentityMorphism(A) in d.premises.keys()
        True
        >>> g * f in d.premises.keys()
        True
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> d.conclusions[g * f] == FiniteSet("unique")
        True

        """
        return self.args[1]

    @property
    def objects(self):
        """
        Returns the :class:`~.FiniteSet` of objects that appear in this
        diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> d.objects
        {Object("A"), Object("B"), Object("C")}

        """
        return self.args[2]

    def hom(self, A, B):
        """
        Returns a 2-tuple of sets of morphisms between objects ``A`` and
        ``B``: one set of morphisms listed as premises, and the other set
        of morphisms listed as conclusions.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> from sympy import pretty
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> print(pretty(d.hom(A, C), use_unicode=False))
        ({g*f:A-->C}, {g*f:A-->C})

        See Also
        ========
        Object, Morphism
        """
        premises = EmptySet
        conclusions = EmptySet

        for morphism in self.premises.keys():
            if (morphism.domain == A) and (morphism.codomain == B):
                premises |= FiniteSet(morphism)
        for morphism in self.conclusions.keys():
            if (morphism.domain == A) and (morphism.codomain == B):
                conclusions |= FiniteSet(morphism)

        return (premises, conclusions)

    def is_subdiagram(self, diagram):
        """
        Checks whether ``diagram`` is a subdiagram of ``self``.
        Diagram `D'` is a subdiagram of `D` if all premises
        (conclusions) of `D'` are contained in the premises
        (conclusions) of `D`.  The morphisms contained
        both in `D'` and `D` should have the same properties for `D'`
        to be a subdiagram of `D`.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> d1 = Diagram([f])
        >>> d.is_subdiagram(d1)
        True
        >>> d1.is_subdiagram(d)
        False
        """
        premises = all((m in self.premises) and
                       (diagram.premises[m] == self.premises[m])
                       for m in diagram.premises)
        if not premises:
            return False

        conclusions = all((m in self.conclusions) and
                          (diagram.conclusions[m] == self.conclusions[m])
                          for m in diagram.conclusions)

        # Premises is surely ``True`` here.
        return conclusions

    def subdiagram_from_objects(self, objects):
        """
        If ``objects`` is a subset of the objects of ``self``, returns
        a diagram which has as premises all those premises of ``self``
        which have a domains and codomains in ``objects``, likewise
        for conclusions.  Properties are preserved.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {f: "unique", g*f: "veryunique"})
        >>> d1 = d.subdiagram_from_objects(FiniteSet(A, B))
        >>> d1 == Diagram([f], {f: "unique"})
        True
        """
        if not objects.is_subset(self.objects):
            raise ValueError(
                "Supplied objects should all belong to the diagram.")

        new_premises = {}
        for morphism, props in self.premises.items():
            if ((sympify(objects.contains(morphism.domain)) is S.true) and
                (sympify(objects.contains(morphism.codomain)) is S.true)):
                new_premises[morphism] = props

        new_conclusions = {}
        for morphism, props in self.conclusions.items():
            if ((sympify(objects.contains(morphism.domain)) is S.true) and
                (sympify(objects.contains(morphism.codomain)) is S.true)):
                new_conclusions[morphism] = props

        return Diagram(new_premises, new_conclusions)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\blockmatrix.py ===
from sympy.assumptions.ask import (Q, ask)
from sympy.core import Basic, Add, Mul, S
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import re, im
from sympy.strategies import typed, exhaust, condition, do_one, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities.iterables import is_sequence, sift
from sympy.utilities.misc import filldedent

from sympy.matrices import Matrix, ShapeError
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.matrices.expressions.determinant import det, Determinant
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.matrices.expressions.trace import trace
from sympy.matrices.expressions.transpose import Transpose, transpose


class BlockMatrix(MatrixExpr):
    """A BlockMatrix is a Matrix comprised of other matrices.

    The submatrices are stored in a SymPy Matrix object but accessed as part of
    a Matrix Expression

    >>> from sympy import (MatrixSymbol, BlockMatrix, symbols,
    ...     Identity, ZeroMatrix, block_collapse)
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])

    Some matrices might be comprised of rows of blocks with
    the matrices in each row having the same height and the
    rows all having the same total number of columns but
    not having the same number of columns for each matrix
    in each row. In this case, the matrix is not a block
    matrix and should be instantiated by Matrix.

    >>> from sympy import ones, Matrix
    >>> dat = [
    ... [ones(3,2), ones(3,3)*2],
    ... [ones(2,3)*3, ones(2,2)*4]]
    ...
    >>> BlockMatrix(dat)
    Traceback (most recent call last):
    ...
    ValueError:
    Although this matrix is comprised of blocks, the blocks do not fill
    the matrix in a size-symmetric fashion. To create a full matrix from
    these arguments, pass them directly to Matrix.
    >>> Matrix(dat)
    Matrix([
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4],
    [3, 3, 3, 4, 4]])

    See Also
    ========
    sympy.matrices.matrixbase.MatrixBase.irregular
    """
    def __new__(cls, *args, **kwargs):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        isMat = lambda i: getattr(i, 'is_Matrix', False)
        if len(args) != 1 or \
                not is_sequence(args[0]) or \
                len({isMat(r) for r in args[0]}) != 1:
            raise ValueError(filldedent('''
                expecting a sequence of 1 or more rows
                containing Matrices.'''))
        rows = args[0] if args else []
        if not isMat(rows):
            if rows and isMat(rows[0]):
                rows = [rows]  # rows is not list of lists or []
            # regularity check
            # same number of matrices in each row
            blocky = ok = len({len(r) for r in rows}) == 1
            if ok:
                # same number of rows for each matrix in a row
                for r in rows:
                    ok = len({i.rows for i in r}) == 1
                    if not ok:
                        break
                blocky = ok
                if ok:
                    # same number of cols for each matrix in each col
                    for c in range(len(rows[0])):
                        ok = len({rows[i][c].cols
                            for i in range(len(rows))}) == 1
                        if not ok:
                            break
            if not ok:
                # same total cols in each row
                ok = len({
                    sum(i.cols for i in r) for r in rows}) == 1
                if blocky and ok:
                    raise ValueError(filldedent('''
                        Although this matrix is comprised of blocks,
                        the blocks do not fill the matrix in a
                        size-symmetric fashion. To create a full matrix
                        from these arguments, pass them directly to
                        Matrix.'''))
                raise ValueError(filldedent('''
                    When there are not the same number of rows in each
                    row's matrices or there are not the same number of
                    total columns in each row, the matrix is not a
                    block matrix. If this matrix is known to consist of
                    blocks fully filling a 2-D space then see
                    Matrix.irregular.'''))
        mat = ImmutableDenseMatrix(rows, evaluate=False)
        obj = Basic.__new__(cls, mat)
        return obj

    @property
    def shape(self):
        numrows = numcols = 0
        M = self.blocks
        for i in range(M.shape[0]):
            numrows += M[i, 0].shape[0]
        for i in range(M.shape[1]):
            numcols += M[0, i].shape[1]
        return (numrows, numcols)

    @property
    def blockshape(self):
        return self.blocks.shape

    @property
    def blocks(self):
        return self.args[0]

    @property
    def rowblocksizes(self):
        return [self.blocks[i, 0].rows for i in range(self.blockshape[0])]

    @property
    def colblocksizes(self):
        return [self.blocks[0, i].cols for i in range(self.blockshape[1])]

    def structurally_equal(self, other):
        return (isinstance(other, BlockMatrix)
            and self.shape == other.shape
            and self.blockshape == other.blockshape
            and self.rowblocksizes == other.rowblocksizes
            and self.colblocksizes == other.colblocksizes)

    def _blockmul(self, other):
        if (isinstance(other, BlockMatrix) and
                self.colblocksizes == other.rowblocksizes):
            return BlockMatrix(self.blocks*other.blocks)

        return self * other

    def _blockadd(self, other):
        if (isinstance(other, BlockMatrix)
                and self.structurally_equal(other)):
            return BlockMatrix(self.blocks + other.blocks)

        return self + other

    def _eval_transpose(self):
        # Flip all the individual matrices
        matrices = [transpose(matrix) for matrix in self.blocks]
        # Make a copy
        M = Matrix(self.blockshape[0], self.blockshape[1], matrices)
        # Transpose the block structure
        M = M.transpose()
        return BlockMatrix(M)

    def _eval_adjoint(self):
        return BlockMatrix(
            Matrix(self.blockshape[0], self.blockshape[1], self.blocks).adjoint()
        )

    def _eval_trace(self):
        if self.rowblocksizes == self.colblocksizes:
            blocks = [self.blocks[i, i] for i in range(self.blockshape[0])]
            return Add(*[trace(block) for block in blocks])

    def _eval_determinant(self):
        if self.blockshape == (1, 1):
            return det(self.blocks[0, 0])
        if self.blockshape == (2, 2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            if ask(Q.invertible(A)):
                return det(A)*det(D - C*A.I*B)
            elif ask(Q.invertible(D)):
                return det(D)*det(A - B*D.I*C)
        return Determinant(self)

    def _eval_as_real_imag(self):
        real_matrices = [re(matrix) for matrix in self.blocks]
        real_matrices = Matrix(self.blockshape[0], self.blockshape[1], real_matrices)

        im_matrices = [im(matrix) for matrix in self.blocks]
        im_matrices = Matrix(self.blockshape[0], self.blockshape[1], im_matrices)

        return (BlockMatrix(real_matrices), BlockMatrix(im_matrices))

    def _eval_derivative(self, x):
        return BlockMatrix(self.blocks.diff(x))

    def transpose(self):
        """Return transpose of matrix.

        Examples
        ========

        >>> from sympy import MatrixSymbol, BlockMatrix, ZeroMatrix
        >>> from sympy.abc import m, n
        >>> X = MatrixSymbol('X', n, n)
        >>> Y = MatrixSymbol('Y', m, m)
        >>> Z = MatrixSymbol('Z', n, m)
        >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
        >>> B.transpose()
        Matrix([
        [X.T,  0],
        [Z.T, Y.T]])
        >>> _.transpose()
        Matrix([
        [X, Z],
        [0, Y]])
        """
        return self._eval_transpose()

    def schur(self, mat = 'A', generalized = False):
        """Return the Schur Complement of the 2x2 BlockMatrix

        Parameters
        ==========

        mat : String, optional
            The matrix with respect to which the
            Schur Complement is calculated. 'A' is
            used by default

        generalized : bool, optional
            If True, returns the generalized Schur
            Component which uses Moore-Penrose Inverse

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])

        The default Schur Complement is evaluated with "A"

        >>> X.schur()
        -C*A**(-1)*B + D
        >>> X.schur('D')
        A - B*D**(-1)*C

        Schur complement with non-invertible matrices is not
        defined. Instead, the generalized Schur complement can
        be calculated which uses the Moore-Penrose Inverse. To
        achieve this, `generalized` must be set to `True`

        >>> X.schur('B', generalized=True)
        C - D*(B.T*B)**(-1)*B.T*A
        >>> X.schur('C', generalized=True)
        -A*(C.T*C)**(-1)*C.T*D + B

        Returns
        =======

        M : Matrix
            The Schur Complement Matrix

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If given matrix is non-invertible

        References
        ==========

        .. [1] Wikipedia Article on Schur Component : https://en.wikipedia.org/wiki/Schur_complement

        See Also
        ========

        sympy.matrices.matrixbase.MatrixBase.pinv
        """

        if self.blockshape == (2, 2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            d={'A' : A, 'B' : B, 'C' : C, 'D' : D}
            try:
                inv = (d[mat].T*d[mat]).inv()*d[mat].T if generalized else d[mat].inv()
                if mat == 'A':
                    return D - C * inv * B
                elif mat == 'B':
                    return C - D * inv * A
                elif mat == 'C':
                    return B - A * inv * D
                elif mat == 'D':
                    return A - B * inv * C
                #For matrices where no sub-matrix is square
                return self
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('The given matrix is not invertible. Please set generalized=True \
            to compute the generalized Schur Complement which uses Moore-Penrose Inverse')
        else:
            raise ShapeError('Schur Complement can only be calculated for 2x2 block matrices')

    def LDUdecomposition(self):
        """Returns the Block LDU decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (L, D, U) : Matrices
            L : Lower Diagonal Matrix
            D : Diagonal Matrix
            U : Upper Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> L, D, U = X.LDUdecomposition()
        >>> block_collapse(L*D*U)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "A" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        """
        if self.blockshape == (2,2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            try:
                AI = A.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block LDU decomposition cannot be calculated when\
                    "A" is singular')
            Ip = Identity(B.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            L = BlockMatrix([[Ip, Z], [C*AI, Iq]])
            D = BlockDiagMatrix(A, self.schur())
            U = BlockMatrix([[Ip, AI*B],[Z.T, Iq]])
            return L, D, U
        else:
            raise ShapeError("Block LDU decomposition is supported only for 2x2 block matrices")

    def UDLdecomposition(self):
        """Returns the Block UDL decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (U, D, L) : Matrices
            U : Upper Diagonal Matrix
            D : Diagonal Matrix
            L : Lower Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> U, D, L = X.UDLdecomposition()
        >>> block_collapse(U*D*L)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "D" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        """
        if self.blockshape == (2,2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            try:
                DI = D.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block UDL decomposition cannot be calculated when\
                    "D" is singular')
            Ip = Identity(A.shape[0])
            Iq = Identity(B.shape[1])
            Z = ZeroMatrix(*B.shape)
            U = BlockMatrix([[Ip, B*DI], [Z.T, Iq]])
            D = BlockDiagMatrix(self.schur('D'), D)
            L = BlockMatrix([[Ip, Z],[DI*C, Iq]])
            return U, D, L
        else:
            raise ShapeError("Block UDL decomposition is supported only for 2x2 block matrices")

    def LUdecomposition(self):
        """Returns the Block LU decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (L, U) : Matrices
            L : Lower Diagonal Matrix
            U : Upper Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> L, U = X.LUdecomposition()
        >>> block_collapse(L*U)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "A" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LDUdecomposition
        """
        if self.blockshape == (2,2):
            [[A, B],
             [C, D]] = self.blocks.tolist()
            try:
                A = A**S.Half
                AI = A.I
            except NonInvertibleMatrixError:
                raise NonInvertibleMatrixError('Block LU decomposition cannot be calculated when\
                    "A" is singular')
            Z = ZeroMatrix(*B.shape)
            Q = self.schur()**S.Half
            L = BlockMatrix([[A, Z], [C*AI, Q]])
            U = BlockMatrix([[A, AI*B],[Z.T, Q]])
            return L, U
        else:
            raise ShapeError("Block LU decomposition is supported only for 2x2 block matrices")

    def _entry(self, i, j, **kwargs):
        # Find row entry
        orig_i, orig_j = i, j
        for row_block, numrows in enumerate(self.rowblocksizes):
            cmp = i < numrows
            if cmp == True:
                break
            elif cmp == False:
                i -= numrows
            elif row_block < self.blockshape[0] - 1:
                # Can't tell which block and it's not the last one, return unevaluated
                return MatrixElement(self, orig_i, orig_j)
        for col_block, numcols in enumerate(self.colblocksizes):
            cmp = j < numcols
            if cmp == True:
                break
            elif cmp == False:
                j -= numcols
            elif col_block < self.blockshape[1] - 1:
                return MatrixElement(self, orig_i, orig_j)
        return self.blocks[row_block, col_block][i, j]

    @property
    def is_Identity(self):
        if self.blockshape[0] != self.blockshape[1]:
            return False
        for i in range(self.blockshape[0]):
            for j in range(self.blockshape[1]):
                if i==j and not self.blocks[i, j].is_Identity:
                    return False
                if i!=j and not self.blocks[i, j].is_ZeroMatrix:
                    return False
        return True

    @property
    def is_structurally_symmetric(self):
        return self.rowblocksizes == self.colblocksizes

    def equals(self, other):
        if self == other:
            return True
        if (isinstance(other, BlockMatrix) and self.blocks == other.blocks):
            return True
        return super().equals(other)


class BlockDiagMatrix(BlockMatrix):
    """A sparse matrix with block matrices along its diagonals

    Examples
    ========

    >>> from sympy import MatrixSymbol, BlockDiagMatrix, symbols
    >>> n, m, l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> BlockDiagMatrix(X, Y)
    Matrix([
    [X, 0],
    [0, Y]])

    Notes
    =====

    If you want to get the individual diagonal blocks, use
    :meth:`get_diag_blocks`.

    See Also
    ========

    sympy.matrices.dense.diag
    """
    def __new__(cls, *mats):
        return Basic.__new__(BlockDiagMatrix, *[_sympify(m) for m in mats])

    @property
    def diag(self):
        return self.args

    @property
    def blocks(self):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        mats = self.args
        data = [[mats[i] if i == j else ZeroMatrix(mats[i].rows, mats[j].cols)
                        for j in range(len(mats))]
                        for i in range(len(mats))]
        return ImmutableDenseMatrix(data, evaluate=False)

    @property
    def shape(self):
        return (sum(block.rows for block in self.args),
                sum(block.cols for block in self.args))

    @property
    def blockshape(self):
        n = len(self.args)
        return (n, n)

    @property
    def rowblocksizes(self):
        return [block.rows for block in self.args]

    @property
    def colblocksizes(self):
        return [block.cols for block in self.args]

    def _all_square_blocks(self):
        """Returns true if all blocks are square"""
        return all(mat.is_square for mat in self.args)

    def _eval_determinant(self):
        if self._all_square_blocks():
            return Mul(*[det(mat) for mat in self.args])
        # At least one block is non-square.  Since the entire matrix must be square we know there must
        # be at least two blocks in this matrix, in which case the entire matrix is necessarily rank-deficient
        return S.Zero

    def _eval_inverse(self, expand='ignored'):
        if self._all_square_blocks():
            return BlockDiagMatrix(*[mat.inverse() for mat in self.args])
        # See comment in _eval_determinant()
        raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')

    def _eval_transpose(self):
        return BlockDiagMatrix(*[mat.transpose() for mat in self.args])

    def _blockmul(self, other):
        if (isinstance(other, BlockDiagMatrix) and
                self.colblocksizes == other.rowblocksizes):
            return BlockDiagMatrix(*[a*b for a, b in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockmul(self, other)

    def _blockadd(self, other):
        if (isinstance(other, BlockDiagMatrix) and
                self.blockshape == other.blockshape and
                self.rowblocksizes == other.rowblocksizes and
                self.colblocksizes == other.colblocksizes):
            return BlockDiagMatrix(*[a + b for a, b in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockadd(self, other)

    def get_diag_blocks(self):
        """Return the list of diagonal blocks of the matrix.

        Examples
        ========

        >>> from sympy import BlockDiagMatrix, Matrix

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[5, 6], [7, 8]])
        >>> M = BlockDiagMatrix(A, B)

        How to get diagonal blocks from the block diagonal matrix:

        >>> diag_blocks = M.get_diag_blocks()
        >>> diag_blocks[0]
        Matrix([
        [1, 2],
        [3, 4]])
        >>> diag_blocks[1]
        Matrix([
        [5, 6],
        [7, 8]])
        """
        return self.args


def block_collapse(expr):
    """Evaluates a block matrix expression

    >>> from sympy import MatrixSymbol, BlockMatrix, symbols, Identity, ZeroMatrix, block_collapse
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m, n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])
    """
    from sympy.strategies.util import expr_fns

    hasbm = lambda expr: isinstance(expr, MatrixExpr) and expr.has(BlockMatrix)

    conditioned_rl = condition(
        hasbm,
        typed(
            {MatAdd: do_one(bc_matadd, bc_block_plus_ident),
             MatMul: do_one(bc_matmul, bc_dist),
             MatPow: bc_matmul,
             Transpose: bc_transpose,
             Inverse: bc_inverse,
             BlockMatrix: do_one(bc_unpack, deblock)}
        )
    )

    rule = exhaust(
        bottom_up(
            exhaust(conditioned_rl),
            fns=expr_fns
        )
    )

    result = rule(expr)
    doit = getattr(result, 'doit', None)
    if doit is not None:
        return doit()
    else:
        return result

def bc_unpack(expr):
    if expr.blockshape == (1, 1):
        return expr.blocks[0, 0]
    return expr

def bc_matadd(expr):
    args = sift(expr.args, lambda M: isinstance(M, BlockMatrix))
    blocks = args[True]
    if not blocks:
        return expr

    nonblocks = args[False]
    block = blocks[0]
    for b in blocks[1:]:
        block = block._blockadd(b)
    if nonblocks:
        return MatAdd(*nonblocks) + block
    else:
        return block

def bc_block_plus_ident(expr):
    idents = [arg for arg in expr.args if arg.is_Identity]
    if not idents:
        return expr

    blocks = [arg for arg in expr.args if isinstance(arg, BlockMatrix)]
    if (blocks and all(b.structurally_equal(blocks[0]) for b in blocks)
               and blocks[0].is_structurally_symmetric):
        block_id = BlockDiagMatrix(*[Identity(k)
                                        for k in blocks[0].rowblocksizes])
        rest = [arg for arg in expr.args if not arg.is_Identity and not isinstance(arg, BlockMatrix)]
        return MatAdd(block_id * len(idents), *blocks, *rest).doit()

    return expr

def bc_dist(expr):
    """ Turn  a*[X, Y] into [a*X, a*Y] """
    factor, mat = expr.as_coeff_mmul()
    if factor == 1:
        return expr

    unpacked = unpack(mat)

    if isinstance(unpacked, BlockDiagMatrix):
        B = unpacked.diag
        new_B = [factor * mat for mat in B]
        return BlockDiagMatrix(*new_B)
    elif isinstance(unpacked, BlockMatrix):
        B = unpacked.blocks
        new_B = [
            [factor * B[i, j] for j in range(B.cols)] for i in range(B.rows)]
        return BlockMatrix(new_B)
    return expr


def bc_matmul(expr):
    if isinstance(expr, MatPow):
        if expr.args[1].is_Integer and expr.args[1] > 0:
            factor, matrices = 1, [expr.args[0]]*expr.args[1]
        else:
            return expr
    else:
        factor, matrices = expr.as_coeff_matrices()

    i = 0
    while (i+1 < len(matrices)):
        A, B = matrices[i:i+2]
        if isinstance(A, BlockMatrix) and isinstance(B, BlockMatrix):
            matrices[i] = A._blockmul(B)
            matrices.pop(i+1)
        elif isinstance(A, BlockMatrix):
            matrices[i] = A._blockmul(BlockMatrix([[B]]))
            matrices.pop(i+1)
        elif isinstance(B, BlockMatrix):
            matrices[i] = BlockMatrix([[A]])._blockmul(B)
            matrices.pop(i+1)
        else:
            i+=1
    return MatMul(factor, *matrices).doit()

def bc_transpose(expr):
    collapse = block_collapse(expr.arg)
    return collapse._eval_transpose()


def bc_inverse(expr):
    if isinstance(expr.arg, BlockDiagMatrix):
        return expr.inverse()

    expr2 = blockinverse_1x1(expr)
    if expr != expr2:
        return expr2
    return blockinverse_2x2(Inverse(reblock_2x2(expr.arg)))

def blockinverse_1x1(expr):
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (1, 1):
        mat = Matrix([[expr.arg.blocks[0].inverse()]])
        return BlockMatrix(mat)
    return expr


def blockinverse_2x2(expr):
    if isinstance(expr.arg, BlockMatrix) and expr.arg.blockshape == (2, 2):
        # See: Inverses of 2x2 Block Matrices, Tzon-Tzer Lu and Sheng-Hua Shiou
        [[A, B],
         [C, D]] = expr.arg.blocks.tolist()

        formula = _choose_2x2_inversion_formula(A, B, C, D)
        if formula != None:
            MI = expr.arg.schur(formula).I
        if formula == 'A':
            AI = A.I
            return BlockMatrix([[AI + AI * B * MI * C * AI, -AI * B * MI], [-MI * C * AI, MI]])
        if formula == 'B':
            BI = B.I
            return BlockMatrix([[-MI * D * BI, MI], [BI + BI * A * MI * D * BI, -BI * A * MI]])
        if formula == 'C':
            CI = C.I
            return BlockMatrix([[-CI * D * MI, CI + CI * D * MI * A * CI], [MI, -MI * A * CI]])
        if formula == 'D':
            DI = D.I
            return BlockMatrix([[MI, -MI * B * DI], [-DI * C * MI, DI + DI * C * MI * B * DI]])

    return expr


def _choose_2x2_inversion_formula(A, B, C, D):
    """
    Assuming [[A, B], [C, D]] would form a valid square block matrix, find
    which of the classical 2x2 block matrix inversion formulas would be
    best suited.

    Returns 'A', 'B', 'C', 'D' to represent the algorithm involving inversion
    of the given argument or None if the matrix cannot be inverted using
    any of those formulas.
    """
    # Try to find a known invertible matrix.  Note that the Schur complement
    # is currently not being considered for this
    A_inv = ask(Q.invertible(A))
    if A_inv == True:
        return 'A'
    B_inv = ask(Q.invertible(B))
    if B_inv == True:
        return 'B'
    C_inv = ask(Q.invertible(C))
    if C_inv == True:
        return 'C'
    D_inv = ask(Q.invertible(D))
    if D_inv == True:
        return 'D'
    # Otherwise try to find a matrix that isn't known to be non-invertible
    if A_inv != False:
        return 'A'
    if B_inv != False:
        return 'B'
    if C_inv != False:
        return 'C'
    if D_inv != False:
        return 'D'
    return None


def deblock(B):
    """ Flatten a BlockMatrix of BlockMatrices """
    if not isinstance(B, BlockMatrix) or not B.blocks.has(BlockMatrix):
        return B
    wrap = lambda x: x if isinstance(x, BlockMatrix) else BlockMatrix([[x]])
    bb = B.blocks.applyfunc(wrap)  # everything is a block

    try:
        MM = Matrix(0, sum(bb[0, i].blocks.shape[1] for i in range(bb.shape[1])), [])
        for row in range(0, bb.shape[0]):
            M = Matrix(bb[row, 0].blocks)
            for col in range(1, bb.shape[1]):
                M = M.row_join(bb[row, col].blocks)
            MM = MM.col_join(M)

        return BlockMatrix(MM)
    except ShapeError:
        return B


def reblock_2x2(expr):
    """
    Reblock a BlockMatrix so that it has 2x2 blocks of block matrices.  If
    possible in such a way that the matrix continues to be invertible using the
    classical 2x2 block inversion formulas.
    """
    if not isinstance(expr, BlockMatrix) or not all(d > 2 for d in expr.blockshape):
        return expr

    BM = BlockMatrix  # for brevity's sake
    rowblocks, colblocks = expr.blockshape
    blocks = expr.blocks
    for i in range(1, rowblocks):
        for j in range(1, colblocks):
            # try to split rows at i and cols at j
            A = bc_unpack(BM(blocks[:i, :j]))
            B = bc_unpack(BM(blocks[:i, j:]))
            C = bc_unpack(BM(blocks[i:, :j]))
            D = bc_unpack(BM(blocks[i:, j:]))

            formula = _choose_2x2_inversion_formula(A, B, C, D)
            if formula is not None:
                return BlockMatrix([[A, B], [C, D]])

    # else: nothing worked, just split upper left corner
    return BM([[blocks[0, 0], BM(blocks[0, 1:])],
               [BM(blocks[1:, 0]), BM(blocks[1:, 1:])]])


def bounds(sizes):
    """ Convert sequence of numbers into pairs of low-high pairs

    >>> from sympy.matrices.expressions.blockmatrix import bounds
    >>> bounds((1, 10, 50))
    [(0, 1), (1, 11), (11, 61)]
    """
    low = 0
    rv = []
    for size in sizes:
        rv.append((low, low + size))
        low += size
    return rv

def blockcut(expr, rowsizes, colsizes):
    """ Cut a matrix expression into Blocks

    >>> from sympy import ImmutableMatrix, blockcut
    >>> M = ImmutableMatrix(4, 4, range(16))
    >>> B = blockcut(M, (1, 3), (1, 3))
    >>> type(B).__name__
    'BlockMatrix'
    >>> ImmutableMatrix(B.blocks[0, 1])
    Matrix([[1, 2, 3]])
    """

    rowbounds = bounds(rowsizes)
    colbounds = bounds(colsizes)
    return BlockMatrix([[MatrixSlice(expr, rowbound, colbound)
                         for colbound in colbounds]
                         for rowbound in rowbounds])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\calculus\inverselaplace.py ===
# contributed to mpmath by Kristopher L. Kuhlman, February 2017
# contributed to mpmath by Guillermo Navas-Palencia, February 2022

class InverseLaplaceTransform(object):
    r"""
    Inverse Laplace transform methods are implemented using this
    class, in order to simplify the code and provide a common
    infrastructure.

    Implement a custom inverse Laplace transform algorithm by
    subclassing :class:`InverseLaplaceTransform` and implementing the
    appropriate methods. The subclass can then be used by
    :func:`~mpmath.invertlaplace` by passing it as the *method*
    argument.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    def calc_laplace_parameter(self, t, **kwargs):
        r"""
        Determine the vector of Laplace parameter values needed for an
        algorithm, this will depend on the choice of algorithm (de
        Hoog is default), the algorithm-specific parameters passed (or
        default ones), and desired time.
        """
        raise NotImplementedError

    def calc_time_domain_solution(self, fp):
        r"""
        Compute the time domain solution, after computing the
        Laplace-space function evaluations at the abscissa required
        for the algorithm. Abscissa computed for one algorithm are
        typically not useful for another algorithm.
        """
        raise NotImplementedError


class FixedTalbot(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        r"""The "fixed" Talbot method deforms the Bromwich contour towards
        `-\infty` in the shape of a parabola. Traditionally the Talbot
        algorithm has adjustable parameters, but the "fixed" version
        does not. The `r` parameter could be passed in as a parameter,
        if you want to override the default given by (Abate & Valko,
        2004).

        The Laplace parameter is sampled along a parabola opening
        along the negative imaginary axis, with the base of the
        parabola along the real axis at
        `p=\frac{r}{t_\mathrm{max}}`. As the number of terms used in
        the approximation (degree) grows, the abscissa required for
        function evaluation tend towards `-\infty`, requiring high
        precision to prevent overflow.  If any poles, branch cuts or
        other singularities exist such that the deformed Bromwich
        contour lies to the left of the singularity, the method will
        fail.

        **Optional arguments**

        :class:`~mpmath.calculus.inverselaplace.FixedTalbot.calc_laplace_parameter`
        recognizes the following keywords

        *tmax*
            maximum time associated with vector of times
            (typically just the time requested)
        *degree*
            integer order of approximation (M = number of terms)
        *r*
            abscissa for `p_0` (otherwise computed using rule
            of thumb `2M/5`)

        The working precision will be increased according to a rule of
        thumb. If 'degree' is not specified, the working precision and
        degree are chosen to hopefully achieve the dps of the calling
        context. If 'degree' is specified, the working precision is
        chosen to achieve maximum resulting precision for the
        specified degree.

        .. math ::

            p_0=\frac{r}{t}

        .. math ::

            p_i=\frac{i r \pi}{Mt_\mathrm{max}}\left[\cot\left(
            \frac{i\pi}{M}\right) + j \right] \qquad 1\le i <M

        where `j=\sqrt{-1}`, `r=2M/5`, and `t_\mathrm{max}` is the
        maximum specified time.

        """

        # required
        # ------------------------------
        # time of desired approximation
        self.t = self.ctx.convert(t)

        # optional
        # ------------------------------
        # maximum time desired (used for scaling) default is requested
        # time.
        self.tmax = self.ctx.convert(kwargs.get('tmax', self.t))

        # empirical relationships used here based on a linear fit of
        # requested and delivered dps for exponentially decaying time
        # functions for requested dps up to 512.

        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = self.degree
        else:
            self.dps_goal = int(1.72*self.ctx.dps)
            self.degree = max(12, int(1.38*self.dps_goal))

        M = self.degree

        # this is adjusting the dps of the calling context hopefully
        # the caller doesn't monkey around with it between calling
        # this routine and calc_time_domain_solution()
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal

        # Abate & Valko rule of thumb for r parameter
        self.r = kwargs.get('r', self.ctx.fraction(2, 5)*M)

        self.theta = self.ctx.linspace(0.0, self.ctx.pi, M+1)

        self.cot_theta = self.ctx.matrix(M, 1)
        self.cot_theta[0] = 0  # not used

        # all but time-dependent part of p
        self.delta = self.ctx.matrix(M, 1)
        self.delta[0] = self.r

        for i in range(1, M):
            self.cot_theta[i] = self.ctx.cot(self.theta[i])
            self.delta[i] = self.r*self.theta[i]*(self.cot_theta[i] + 1j)

        self.p = self.ctx.matrix(M, 1)
        self.p = self.delta/self.tmax

        # NB: p is complex (mpc)

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        r"""The fixed Talbot time-domain solution is computed from the
        Laplace-space function evaluations using

        .. math ::

            f(t,M)=\frac{2}{5t}\sum_{k=0}^{M-1}\Re \left[
            \gamma_k \bar{f}(p_k)\right]

        where

        .. math ::

            \gamma_0 = \frac{1}{2}e^{r}\bar{f}(p_0)

        .. math ::

            \gamma_k = e^{tp_k}\left\lbrace 1 + \frac{jk\pi}{M}\left[1 +
            \cot \left( \frac{k \pi}{M} \right)^2 \right] - j\cot\left(
            \frac{k \pi}{M}\right)\right \rbrace \qquad 1\le k<M.

        Again, `j=\sqrt{-1}`.

        Before calling this function, call
        :class:`~mpmath.calculus.inverselaplace.FixedTalbot.calc_laplace_parameter`
        to set the parameters and compute the required coefficients.

        **References**

        1. Abate, J., P. Valko (2004). Multi-precision Laplace
           transform inversion. *International Journal for Numerical
           Methods in Engineering* 60:979-993,
           http://dx.doi.org/10.1002/nme.995
        2. Talbot, A. (1979). The accurate numerical inversion of
           Laplace transforms. *IMA Journal of Applied Mathematics*
           23(1):97, http://dx.doi.org/10.1093/imamat/23.1.97
        """

        # required
        # ------------------------------
        self.t = self.ctx.convert(t)

        # assume fp was computed from p matrix returned from
        # calc_laplace_parameter(), so is already a list or matrix of
        # mpmath 'mpc' types

        # these were computed in previous call to
        # calc_laplace_parameter()
        theta = self.theta
        delta = self.delta
        M = self.degree
        p = self.p
        r = self.r

        ans = self.ctx.matrix(M, 1)
        ans[0] = self.ctx.exp(delta[0])*fp[0]/2

        for i in range(1, M):
            ans[i] = self.ctx.exp(delta[i])*fp[i]*(
                1 + 1j*theta[i]*(1 + self.cot_theta[i]**2) -
                1j*self.cot_theta[i])

        result = self.ctx.fraction(2, 5)*self.ctx.fsum(ans)/self.t

        # setting dps back to value when calc_laplace_parameter was
        # called, unless flag is set.
        if not manual_prec:
            self.ctx.dps = self.dps_orig

        return result.real


# ****************************************

class Stehfest(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        r"""
        The Gaver-Stehfest method is a discrete approximation of the
        Widder-Post inversion algorithm, rather than a direct
        approximation of the Bromwich contour integral.

        The method abscissa along the real axis, and therefore has
        issues inverting oscillatory functions (which have poles in
        pairs away from the real axis).

        The working precision will be increased according to a rule of
        thumb. If 'degree' is not specified, the working precision and
        degree are chosen to hopefully achieve the dps of the calling
        context. If 'degree' is specified, the working precision is
        chosen to achieve maximum resulting precision for the
        specified degree.

        .. math ::

            p_k = \frac{k \log 2}{t} \qquad 1 \le k \le M
        """

        # required
        # ------------------------------
        # time of desired approximation
        self.t = self.ctx.convert(t)

        # optional
        # ------------------------------

        # empirical relationships used here based on a linear fit of
        # requested and delivered dps for exponentially decaying time
        # functions for requested dps up to 512.

        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = int(1.38*self.degree)
        else:
            self.dps_goal = int(2.93*self.ctx.dps)
            self.degree = max(16, self.dps_goal)

        # _coeff routine requires even degree
        if self.degree % 2 > 0:
            self.degree += 1

        M = self.degree

        # this is adjusting the dps of the calling context
        # hopefully the caller doesn't monkey around with it
        # between calling this routine and calc_time_domain_solution()
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal

        self.V = self._coeff()
        self.p = self.ctx.matrix(self.ctx.arange(1, M+1))*self.ctx.ln2/self.t

        # NB: p is real (mpf)

    def _coeff(self):
        r"""Salzer summation weights (aka, "Stehfest coefficients")
        only depend on the approximation order (M) and the precision"""

        M = self.degree
        M2 = int(M/2)  # checked earlier that M is even

        V = self.ctx.matrix(M, 1)

        # Salzer summation weights
        # get very large in magnitude and oscillate in sign,
        # if the precision is not high enough, there will be
        # catastrophic cancellation
        for k in range(1, M+1):
            z = self.ctx.matrix(min(k, M2)+1, 1)
            for j in range(int((k+1)/2), min(k, M2)+1):
                z[j] = (self.ctx.power(j, M2)*self.ctx.fac(2*j)/
                        (self.ctx.fac(M2-j)*self.ctx.fac(j)*
                         self.ctx.fac(j-1)*self.ctx.fac(k-j)*
                         self.ctx.fac(2*j-k)))
            V[k-1] = self.ctx.power(-1, k+M2)*self.ctx.fsum(z)

        return V

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        r"""Compute time-domain Stehfest algorithm solution.

        .. math ::

            f(t,M) = \frac{\log 2}{t} \sum_{k=1}^{M} V_k \bar{f}\left(
            p_k \right)

        where

        .. math ::

            V_k = (-1)^{k + N/2} \sum^{\min(k,N/2)}_{i=\lfloor(k+1)/2 \rfloor}
            \frac{i^{\frac{N}{2}}(2i)!}{\left(\frac{N}{2}-i \right)! \, i! \,
            \left(i-1 \right)! \, \left(k-i\right)! \, \left(2i-k \right)!}

        As the degree increases, the abscissa (`p_k`) only increase
        linearly towards `\infty`, but the Stehfest coefficients
        (`V_k`) alternate in sign and increase rapidly in sign,
        requiring high precision to prevent overflow or loss of
        significance when evaluating the sum.

        **References**

        1. Widder, D. (1941). *The Laplace Transform*. Princeton.
        2. Stehfest, H. (1970). Algorithm 368: numerical inversion of
           Laplace transforms. *Communications of the ACM* 13(1):47-49,
           http://dx.doi.org/10.1145/361953.361969

        """

        # required
        self.t = self.ctx.convert(t)

        # assume fp was computed from p matrix returned from
        # calc_laplace_parameter(), so is already
        # a list or matrix of mpmath 'mpf' types

        result = self.ctx.fdot(self.V, fp)*self.ctx.ln2/self.t

        # setting dps back to value when calc_laplace_parameter was called
        if not manual_prec:
            self.ctx.dps = self.dps_orig

        # ignore any small imaginary part
        return result.real


# ****************************************

class deHoog(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        r"""the de Hoog, Knight & Stokes algorithm is an
        accelerated form of the Fourier series numerical
        inverse Laplace transform algorithms.

        .. math ::

            p_k = \gamma + \frac{jk}{T} \qquad 0 \le k < 2M+1

        where

        .. math ::

            \gamma = \alpha - \frac{\log \mathrm{tol}}{2T},

        `j=\sqrt{-1}`, `T = 2t_\mathrm{max}` is a scaled time,
        `\alpha=10^{-\mathrm{dps\_goal}}` is the real part of the
        rightmost pole or singularity, which is chosen based on the
        desired accuracy (assuming the rightmost singularity is 0),
        and `\mathrm{tol}=10\alpha` is the desired tolerance, which is
        chosen in relation to `\alpha`.`

        When increasing the degree, the abscissa increase towards
        `j\infty`, but more slowly than the fixed Talbot
        algorithm. The de Hoog et al. algorithm typically does better
        with oscillatory functions of time, and less well-behaved
        functions. The method tends to be slower than the Talbot and
        Stehfest algorithsm, especially so at very high precision
        (e.g., `>500` digits precision).

        """

        # required
        # ------------------------------
        self.t = self.ctx.convert(t)

        # optional
        # ------------------------------
        self.tmax = kwargs.get('tmax', self.t)

        # empirical relationships used here based on a linear fit of
        # requested and delivered dps for exponentially decaying time
        # functions for requested dps up to 512.

        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = int(1.38*self.degree)
        else:
            self.dps_goal = int(self.ctx.dps*1.36)
            self.degree = max(10, self.dps_goal)

        # 2*M+1 terms in approximation
        M = self.degree

        # adjust alpha component of abscissa of convergence for higher
        # precision
        tmp = self.ctx.power(10.0, -self.dps_goal)
        self.alpha = self.ctx.convert(kwargs.get('alpha', tmp))

        # desired tolerance (here simply related to alpha)
        self.tol = self.ctx.convert(kwargs.get('tol', self.alpha*10.0))
        self.np = 2*self.degree+1  # number of terms in approximation

        # this is adjusting the dps of the calling context
        # hopefully the caller doesn't monkey around with it
        # between calling this routine and calc_time_domain_solution()
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal

        # scaling factor (likely tun-able, but 2 is typical)
        self.scale = kwargs.get('scale', 2)
        self.T = self.ctx.convert(kwargs.get('T', self.scale*self.tmax))

        self.p = self.ctx.matrix(2*M+1, 1)
        self.gamma = self.alpha - self.ctx.log(self.tol)/(self.scale*self.T)
        self.p = (self.gamma + self.ctx.pi*
                  self.ctx.matrix(self.ctx.arange(self.np))/self.T*1j)

        # NB: p is complex (mpc)

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        r"""Calculate time-domain solution for
        de Hoog, Knight & Stokes algorithm.

        The un-accelerated Fourier series approach is:

        .. math ::

            f(t,2M+1) = \frac{e^{\gamma t}}{T} \sum_{k=0}^{2M}{}^{'}
            \Re\left[\bar{f}\left( p_k \right)
            e^{i\pi t/T} \right],

        where the prime on the summation indicates the first term is halved.

        This simplistic approach requires so many function evaluations
        that it is not practical. Non-linear acceleration is
        accomplished via Pade-approximation and an analytic expression
        for the remainder of the continued fraction. See the original
        paper (reference 2 below) a detailed description of the
        numerical approach.

        **References**

        1. Davies, B. (2005). *Integral Transforms and their
           Applications*, Third Edition. Springer.
        2. de Hoog, F., J. Knight, A. Stokes (1982). An improved
           method for numerical inversion of Laplace transforms. *SIAM
           Journal of Scientific and Statistical Computing* 3:357-366,
           http://dx.doi.org/10.1137/0903022

        """

        M = self.degree
        np = self.np
        T = self.T

        self.t = self.ctx.convert(t)

        # would it be useful to try re-using
        # space between e&q and A&B?
        e = self.ctx.zeros(np, M+1)
        q = self.ctx.matrix(2*M, M)
        d = self.ctx.matrix(np, 1)
        A = self.ctx.zeros(np+1, 1)
        B = self.ctx.ones(np+1, 1)

        # initialize Q-D table
        e[:, 0] = 0.0 + 0j
        q[0, 0] = fp[1]/(fp[0]/2)
        for i in range(1, 2*M):
            q[i, 0] = fp[i+1]/fp[i]

        # rhombus rule for filling triangular Q-D table (e & q)
        for r in range(1, M+1):
            # start with e, column 1, 0:2*M-2
            mr = 2*(M-r) + 1
            e[0:mr, r] = q[1:mr+1, r-1] - q[0:mr, r-1] + e[1:mr+1, r-1]
            if not r == M:
                rq = r+1
                mr = 2*(M-rq)+1 + 2
                for i in range(mr):
                    q[i, rq-1] = q[i+1, rq-2]*e[i+1, rq-1]/e[i, rq-1]

        # build up continued fraction coefficients (d)
        d[0] = fp[0]/2
        for r in range(1, M+1):
            d[2*r-1] = -q[0, r-1]  # even terms
            d[2*r]   = -e[0, r]    # odd terms

        # seed A and B for recurrence
        A[0] = 0.0 + 0.0j
        A[1] = d[0]
        B[0:2] = 1.0 + 0.0j

        # base of the power series
        z = self.ctx.expjpi(self.t/T)  # i*pi is already in fcn

        # coefficients of Pade approximation (A & B)
        # using recurrence for all but last term
        for i in range(1, 2*M):
            A[i+1] = A[i] + d[i]*A[i-1]*z
            B[i+1] = B[i] + d[i]*B[i-1]*z

        # "improved remainder" to continued fraction
        brem = (1 + (d[2*M-1] - d[2*M])*z)/2
        # powm1(x,y) computes x^y - 1 more accurately near zero
        rem = brem*self.ctx.powm1(1 + d[2*M]*z/brem,
                                  self.ctx.fraction(1, 2))

        # last term of recurrence using new remainder
        A[np] = A[2*M] + rem*A[2*M-1]
        B[np] = B[2*M] + rem*B[2*M-1]

        # diagonal Pade approximation
        # F=A/B represents accelerated trapezoid rule
        result = self.ctx.exp(self.gamma*self.t)/T*(A[np]/B[np]).real

        # setting dps back to value when calc_laplace_parameter was called
        if not manual_prec:
            self.ctx.dps = self.dps_orig

        return result


# ****************************************

class Cohen(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        r"""The Cohen algorithm accelerates the convergence of the nearly
        alternating series resulting from the application of the trapezoidal
        rule to the Bromwich contour inversion integral.

        .. math ::

            p_k = \frac{\gamma}{2 t} + \frac{\pi i k}{t} \qquad 0 \le k < M

        where

        .. math ::

            \gamma = \frac{2}{3} (d + \log(10) + \log(2 t)),

        `d = \mathrm{dps\_goal}`, which is chosen based on the desired
        accuracy using the method developed in [1] to improve numerical
        stability. The Cohen algorithm shows robustness similar to the de Hoog
        et al. algorithm, but it is faster than the fixed Talbot algorithm.

        **Optional arguments**

        *degree*
            integer order of the approximation (M = number of terms)
        *alpha*
            abscissa for `p_0` (controls the discretization error)

        The working precision will be increased according to a rule of
        thumb. If 'degree' is not specified, the working precision and
        degree are chosen to hopefully achieve the dps of the calling
        context. If 'degree' is specified, the working precision is
        chosen to achieve maximum resulting precision for the
        specified degree.

        **References**

        1. P. Glasserman, J. Ruiz-Mata (2006). Computing the credit loss
        distribution in the Gaussian copula model: a comparison of methods.
        *Journal of Credit Risk* 2(4):33-66, 10.21314/JCR.2006.057

        """
        self.t = self.ctx.convert(t)

        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = int(1.5 * self.degree)
        else:
            self.dps_goal = int(self.ctx.dps * 1.74)
            self.degree = max(22, int(1.31 * self.dps_goal))

        M = self.degree + 1

        # this is adjusting the dps of the calling context hopefully
        # the caller doesn't monkey around with it between calling
        # this routine and calc_time_domain_solution()
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal

        ttwo = 2 * self.t
        tmp = self.ctx.dps * self.ctx.log(10) + self.ctx.log(ttwo)
        tmp = self.ctx.fraction(2, 3) * tmp
        self.alpha = self.ctx.convert(kwargs.get('alpha', tmp))

        # all but time-dependent part of p
        a_t = self.alpha / ttwo
        p_t = self.ctx.pi * 1j / self.t

        self.p = self.ctx.matrix(M, 1)
        self.p[0] = a_t

        for i in range(1, M):
            self.p[i] = a_t + i * p_t

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        r"""Calculate time-domain solution for Cohen algorithm.

        The accelerated nearly alternating series is:

        .. math ::

            f(t, M) = \frac{e^{\gamma / 2}}{t} \left[\frac{1}{2}
            \Re\left(\bar{f}\left(\frac{\gamma}{2t}\right) \right) -
            \sum_{k=0}^{M-1}\frac{c_{M,k}}{d_M}\Re\left(\bar{f}
            \left(\frac{\gamma + 2(k+1) \pi i}{2t}\right)\right)\right],

        where coefficients `\frac{c_{M, k}}{d_M}` are described in [1].

        1. H. Cohen, F. Rodriguez Villegas, D. Zagier (2000). Convergence
        acceleration of alternating series. *Experiment. Math* 9(1):3-12

        """
        self.t = self.ctx.convert(t)

        n = self.degree
        M = n + 1

        A = self.ctx.matrix(M, 1)
        for i in range(M):
            A[i] = fp[i].real

        d = (3 + self.ctx.sqrt(8)) ** n
        d = (d + 1 / d) / 2
        b = -self.ctx.one
        c = -d
        s = 0

        for k in range(n):
            c = b - c
            s = s + c * A[k + 1]
            b = 2 * (k + n) * (k - n) * b / ((2 * k + 1) * (k + self.ctx.one))

        result = self.ctx.exp(self.alpha / 2) / self.t * (A[0] / 2 - s / d)

        # setting dps back to value when calc_laplace_parameter was
        # called, unless flag is set.
        if not manual_prec:
            self.ctx.dps = self.dps_orig

        return result


# ****************************************

class LaplaceTransformInversionMethods(object):
    def __init__(ctx, *args, **kwargs):
        ctx._fixed_talbot = FixedTalbot(ctx)
        ctx._stehfest = Stehfest(ctx)
        ctx._de_hoog = deHoog(ctx)
        ctx._cohen = Cohen(ctx)

    def invertlaplace(ctx, f, t, **kwargs):
        r"""Computes the numerical inverse Laplace transform for a
        Laplace-space function at a given time.  The function being
        evaluated is assumed to be a real-valued function of time.

        The user must supply a Laplace-space function `\bar{f}(p)`,
        and a desired time at which to estimate the time-domain
        solution `f(t)`.

        A few basic examples of Laplace-space functions with known
        inverses (see references [1,2]) :

        .. math ::

            \mathcal{L}\left\lbrace f(t) \right\rbrace=\bar{f}(p)

        .. math ::

            \mathcal{L}^{-1}\left\lbrace \bar{f}(p) \right\rbrace = f(t)

        .. math ::

            \bar{f}(p) = \frac{1}{(p+1)^2}

        .. math ::

            f(t) = t e^{-t}

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> tt = [0.001, 0.01, 0.1, 1, 10]
        >>> fp = lambda p: 1/(p+1)**2
        >>> ft = lambda t: t*exp(-t)
        >>> ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='talbot')
        (0.000999000499833375, 8.57923043561212e-20)
        >>> ft(tt[1]),ft(tt[1])-invertlaplace(fp,tt[1],method='talbot')
        (0.00990049833749168, 3.27007646698047e-19)
        >>> ft(tt[2]),ft(tt[2])-invertlaplace(fp,tt[2],method='talbot')
        (0.090483741803596, -1.75215800052168e-18)
        >>> ft(tt[3]),ft(tt[3])-invertlaplace(fp,tt[3],method='talbot')
        (0.367879441171442, 1.2428864009344e-17)
        >>> ft(tt[4]),ft(tt[4])-invertlaplace(fp,tt[4],method='talbot')
        (0.000453999297624849, 4.04513489306658e-20)

        The methods also work for higher precision:

        >>> mp.dps = 100; mp.pretty = True
        >>> nstr(ft(tt[0]),15),nstr(ft(tt[0])-invertlaplace(fp,tt[0],method='talbot'),15)
        ('0.000999000499833375', '-4.96868310693356e-105')
        >>> nstr(ft(tt[1]),15),nstr(ft(tt[1])-invertlaplace(fp,tt[1],method='talbot'),15)
        ('0.00990049833749168', '1.23032291513122e-104')

        .. math ::

            \bar{f}(p) = \frac{1}{p^2+1}

        .. math ::

            f(t) = \mathrm{J}_0(t)

        >>> mp.dps = 15; mp.pretty = True
        >>> fp = lambda p: 1/sqrt(p*p + 1)
        >>> ft = lambda t: besselj(0,t)
        >>> ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='dehoog')
        (0.999999750000016, -6.09717765032273e-18)
        >>> ft(tt[1]),ft(tt[1])-invertlaplace(fp,tt[1],method='dehoog')
        (0.99997500015625, -5.61756281076169e-17)

        .. math ::

            \bar{f}(p) = \frac{\log p}{p}

        .. math ::

            f(t) = -\gamma -\log t

        >>> mp.dps = 15; mp.pretty = True
        >>> fp = lambda p: log(p)/p
        >>> ft = lambda t: -euler-log(t)
        >>> ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='stehfest')
        (6.3305396140806, -1.92126634837863e-16)
        >>> ft(tt[1]),ft(tt[1])-invertlaplace(fp,tt[1],method='stehfest')
        (4.02795452108656, -4.81486093200704e-16)

        **Options**

        :func:`~mpmath.invertlaplace` recognizes the following optional
        keywords valid for all methods:

        *method*
            Chooses numerical inverse Laplace transform algorithm
            (described below).
        *degree*
            Number of terms used in the approximation

        **Algorithms**

        Mpmath implements four numerical inverse Laplace transform
        algorithms, attributed to: Talbot, Stehfest, and de Hoog,
        Knight and Stokes. These can be selected by using
        *method='talbot'*, *method='stehfest'*, *method='dehoog'* or
        *method='cohen'* or by passing the classes *method=FixedTalbot*,
        *method=Stehfest*, *method=deHoog*, or *method=Cohen*. The functions
        :func:`~mpmath.invlaptalbot`, :func:`~mpmath.invlapstehfest`,
        :func:`~mpmath.invlapdehoog`, and :func:`~mpmath.invlapcohen`
        are also available as shortcuts.

        All four algorithms implement a heuristic balance between the
        requested precision and the precision used internally for the
        calculations. This has been tuned for a typical exponentially
        decaying function and precision up to few hundred decimal
        digits.

        The Laplace transform converts the variable time (i.e., along
        a line) into a parameter given by the right half of the
        complex `p`-plane.  Singularities, poles, and branch cuts in
        the complex `p`-plane contain all the information regarding
        the time behavior of the corresponding function. Any numerical
        method must therefore sample `p`-plane "close enough" to the
        singularities to accurately characterize them, while not
        getting too close to have catastrophic cancellation, overflow,
        or underflow issues. Most significantly, if one or more of the
        singularities in the `p`-plane is not on the left side of the
        Bromwich contour, its effects will be left out of the computed
        solution, and the answer will be completely wrong.

        *Talbot*

        The fixed Talbot method is high accuracy and fast, but the
        method can catastrophically fail for certain classes of time-domain
        behavior, including a Heaviside step function for positive
        time (e.g., `H(t-2)`), or some oscillatory behaviors. The
        Talbot method usually has adjustable parameters, but the
        "fixed" variety implemented here does not. This method
        deforms the Bromwich integral contour in the shape of a
        parabola towards `-\infty`, which leads to problems
        when the solution has a decaying exponential in it (e.g., a
        Heaviside step function is equivalent to multiplying by a
        decaying exponential in Laplace space).

        *Stehfest*

        The Stehfest algorithm only uses abscissa along the real axis
        of the complex `p`-plane to estimate the time-domain
        function. Oscillatory time-domain functions have poles away
        from the real axis, so this method does not work well with
        oscillatory functions, especially high-frequency ones. This
        method also depends on summation of terms in a series that
        grows very large, and will have catastrophic cancellation
        during summation if the working precision is too low.

        *de Hoog et al.*

        The de Hoog, Knight, and Stokes method is essentially a
        Fourier-series quadrature-type approximation to the Bromwich
        contour integral, with non-linear series acceleration and an
        analytical expression for the remainder term. This method is
        typically one of the most robust. This method also involves the
        greatest amount of overhead, so it is typically the slowest of the
        four methods at high precision.

        *Cohen*

        The Cohen method is a trapezoidal rule approximation to the Bromwich
        contour integral, with linear acceleration for alternating
        series. This method is as robust as the de Hoog et al method and the
        fastest of the four methods at high precision, and is therefore the
        default method.

        **Singularities**

        All numerical inverse Laplace transform methods have problems
        at large time when the Laplace-space function has poles,
        singularities, or branch cuts to the right of the origin in
        the complex plane. For simple poles in `\bar{f}(p)` at the
        `p`-plane origin, the time function is constant in time (e.g.,
        `\mathcal{L}\left\lbrace 1 \right\rbrace=1/p` has a pole at
        `p=0`). A pole in `\bar{f}(p)` to the left of the origin is a
        decreasing function of time (e.g., `\mathcal{L}\left\lbrace
        e^{-t/2} \right\rbrace=1/(p+1/2)` has a pole at `p=-1/2`), and
        a pole to the right of the origin leads to an increasing
        function in time (e.g., `\mathcal{L}\left\lbrace t e^{t/4}
        \right\rbrace = 1/(p-1/4)^2` has a pole at `p=1/4`).  When
        singularities occur off the real `p` axis, the time-domain
        function is oscillatory. For example `\mathcal{L}\left\lbrace
        \mathrm{J}_0(t) \right\rbrace=1/\sqrt{p^2+1}` has a branch cut
        starting at `p=j=\sqrt{-1}` and is a decaying oscillatory
        function, This range of behaviors is illustrated in Duffy [3]
        Figure 4.10.4, p. 228.

        In general as `p \rightarrow \infty` `t \rightarrow 0` and
        vice-versa. All numerical inverse Laplace transform methods
        require their abscissa to shift closer to the origin for
        larger times. If the abscissa shift left of the rightmost
        singularity in the Laplace domain, the answer will be
        completely wrong (the effect of singularities to the right of
        the Bromwich contour are not included in the results).

        For example, the following exponentially growing function has
        a pole at `p=3`:

        .. math ::

            \bar{f}(p)=\frac{1}{p^2-9}

        .. math ::

            f(t)=\frac{1}{3}\sinh 3t

        >>> mp.dps = 15; mp.pretty = True
        >>> fp = lambda p: 1/(p*p-9)
        >>> ft = lambda t: sinh(3*t)/3
        >>> tt = [0.01,0.1,1.0,10.0]
        >>> ft(tt[0]),invertlaplace(fp,tt[0],method='talbot')
        (0.0100015000675014, 0.0100015000675014)
        >>> ft(tt[1]),invertlaplace(fp,tt[1],method='talbot')
        (0.101506764482381, 0.101506764482381)
        >>> ft(tt[2]),invertlaplace(fp,tt[2],method='talbot')
        (3.33929164246997, 3.33929164246997)
        >>> ft(tt[3]),invertlaplace(fp,tt[3],method='talbot')
        (1781079096920.74, -1.61331069624091e-14)

        **References**

        1. [DLMF]_ section 1.14 (http://dlmf.nist.gov/1.14T4)
        2. Cohen, A.M. (2007). Numerical Methods for Laplace Transform
           Inversion, Springer.
        3. Duffy, D.G. (1998). Advanced Engineering Mathematics, CRC Press.

        **Numerical Inverse Laplace Transform Reviews**

        1. Bellman, R., R.E. Kalaba, J.A. Lockett (1966). *Numerical
           inversion of the Laplace transform: Applications to Biology,
           Economics, Engineering, and Physics*. Elsevier.
        2. Davies, B., B. Martin (1979). Numerical inversion of the
           Laplace transform: a survey and comparison of methods. *Journal
           of Computational Physics* 33:1-32,
           http://dx.doi.org/10.1016/0021-9991(79)90025-1
        3. Duffy, D.G. (1993). On the numerical inversion of Laplace
           transforms: Comparison of three new methods on characteristic
           problems from applications. *ACM Transactions on Mathematical
           Software* 19(3):333-359, http://dx.doi.org/10.1145/155743.155788
        4. Kuhlman, K.L., (2013). Review of Inverse Laplace Transform
           Algorithms for Laplace-Space Numerical Approaches, *Numerical
           Algorithms*, 63(2):339-355.
           http://dx.doi.org/10.1007/s11075-012-9625-3

        """

        rule = kwargs.get('method', 'cohen')
        if type(rule) is str:
            lrule = rule.lower()
            if lrule == 'talbot':
                rule = ctx._fixed_talbot
            elif lrule == 'stehfest':
                rule = ctx._stehfest
            elif lrule == 'dehoog':
                rule = ctx._de_hoog
            elif rule == 'cohen':
                rule = ctx._cohen
            else:
                raise ValueError("unknown invlap algorithm: %s" % rule)
        else:
            rule = rule(ctx)

        # determine the vector of Laplace-space parameter
        # needed for the requested method and desired time
        rule.calc_laplace_parameter(t, **kwargs)

        # compute the Laplace-space function evalutations
        # at the required abscissa.
        fp = [f(p) for p in rule.p]

        # compute the time-domain solution from the
        # Laplace-space function evaluations
        return rule.calc_time_domain_solution(fp, t)

    # shortcuts for the above function for specific methods
    def invlaptalbot(ctx, *args, **kwargs):
        kwargs['method'] = 'talbot'
        return ctx.invertlaplace(*args, **kwargs)

    def invlapstehfest(ctx, *args, **kwargs):
        kwargs['method'] = 'stehfest'
        return ctx.invertlaplace(*args, **kwargs)

    def invlapdehoog(ctx, *args, **kwargs):
        kwargs['method'] = 'dehoog'
        return ctx.invertlaplace(*args, **kwargs)

    def invlapcohen(ctx, *args, **kwargs):
        kwargs['method'] = 'cohen'
        return ctx.invertlaplace(*args, **kwargs)


# ****************************************

if __name__ == '__main__':
    import doctest
    doctest.testmod()