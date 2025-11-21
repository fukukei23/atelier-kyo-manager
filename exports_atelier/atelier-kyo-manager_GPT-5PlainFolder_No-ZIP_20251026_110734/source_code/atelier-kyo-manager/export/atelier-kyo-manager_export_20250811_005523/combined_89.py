
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\utils.py ===
"""
requests.utils
~~~~~~~~~~~~~~

This module provides utility functions that are used within Requests
that are also useful for external consumption.
"""

import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict

from pip._vendor.urllib3.util import make_headers, parse_url

from . import certs
from .__version__ import __version__

# to_native_string is unused here, but imported here for backwards compatibility
from ._internal_utils import (  # noqa: F401
    _HEADER_VALIDATORS_BYTE,
    _HEADER_VALIDATORS_STR,
    HEADER_VALIDATORS,
    to_native_string,
)
from .compat import (
    Mapping,
    basestring,
    bytes,
    getproxies,
    getproxies_environment,
    integer_types,
)
from .compat import parse_http_list as _parse_list_header
from .compat import (
    proxy_bypass,
    proxy_bypass_environment,
    quote,
    str,
    unquote,
    urlparse,
    urlunparse,
)
from .cookies import cookiejar_from_dict
from .exceptions import (
    FileModeWarning,
    InvalidHeader,
    InvalidURL,
    UnrewindableBodyError,
)
from .structures import CaseInsensitiveDict

NETRC_FILES = (".netrc", "_netrc")

DEFAULT_CA_BUNDLE_PATH = certs.where()

DEFAULT_PORTS = {"http": 80, "https": 443}

# Ensure that ', ' is used to preserve previous delimiter behavior.
DEFAULT_ACCEPT_ENCODING = ", ".join(
    re.split(r",\s*", make_headers(accept_encoding=True)["accept-encoding"])
)


if sys.platform == "win32":
    # provide a proxy_bypass version on Windows without DNS lookups

    def proxy_bypass_registry(host):
        try:
            import winreg
        except ImportError:
            return False

        try:
            internetSettings = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            )
            # ProxyEnable could be REG_SZ or REG_DWORD, normalizing it
            proxyEnable = int(winreg.QueryValueEx(internetSettings, "ProxyEnable")[0])
            # ProxyOverride is almost always a string
            proxyOverride = winreg.QueryValueEx(internetSettings, "ProxyOverride")[0]
        except (OSError, ValueError):
            return False
        if not proxyEnable or not proxyOverride:
            return False

        # make a check value list from the registry entry: replace the
        # '<local>' string by the localhost entry and the corresponding
        # canonical entry.
        proxyOverride = proxyOverride.split(";")
        # filter out empty strings to avoid re.match return true in the following code.
        proxyOverride = filter(None, proxyOverride)
        # now check if we match one of the registry values.
        for test in proxyOverride:
            if test == "<local>":
                if "." not in host:
                    return True
            test = test.replace(".", r"\.")  # mask dots
            test = test.replace("*", r".*")  # change glob sequence
            test = test.replace("?", r".")  # change glob char
            if re.match(test, host, re.I):
                return True
        return False

    def proxy_bypass(host):  # noqa
        """Return True, if the host should be bypassed.

        Checks proxy settings gathered from the environment, if specified,
        or the registry.
        """
        if getproxies_environment():
            return proxy_bypass_environment(host)
        else:
            return proxy_bypass_registry(host)


def dict_to_sequence(d):
    """Returns an internal sequence dictionary update."""

    if hasattr(d, "items"):
        d = d.items()

    return d


def super_len(o):
    total_length = None
    current_position = 0

    if isinstance(o, str):
        o = o.encode("utf-8")

    if hasattr(o, "__len__"):
        total_length = len(o)

    elif hasattr(o, "len"):
        total_length = o.len

    elif hasattr(o, "fileno"):
        try:
            fileno = o.fileno()
        except (io.UnsupportedOperation, AttributeError):
            # AttributeError is a surprising exception, seeing as how we've just checked
            # that `hasattr(o, 'fileno')`.  It happens for objects obtained via
            # `Tarfile.extractfile()`, per issue 5229.
            pass
        else:
            total_length = os.fstat(fileno).st_size

            # Having used fstat to determine the file length, we need to
            # confirm that this file was opened up in binary mode.
            if "b" not in o.mode:
                warnings.warn(
                    (
                        "Requests has determined the content-length for this "
                        "request using the binary size of the file: however, the "
                        "file has been opened in text mode (i.e. without the 'b' "
                        "flag in the mode). This may lead to an incorrect "
                        "content-length. In Requests 3.0, support will be removed "
                        "for files in text mode."
                    ),
                    FileModeWarning,
                )

    if hasattr(o, "tell"):
        try:
            current_position = o.tell()
        except OSError:
            # This can happen in some weird situations, such as when the file
            # is actually a special file descriptor like stdin. In this
            # instance, we don't know what the length is, so set it to zero and
            # let requests chunk it instead.
            if total_length is not None:
                current_position = total_length
        else:
            if hasattr(o, "seek") and total_length is None:
                # StringIO and BytesIO have seek but no usable fileno
                try:
                    # seek to end of file
                    o.seek(0, 2)
                    total_length = o.tell()

                    # seek back to current position to support
                    # partially read file-like objects
                    o.seek(current_position or 0)
                except OSError:
                    total_length = 0

    if total_length is None:
        total_length = 0

    return max(0, total_length - current_position)


def get_netrc_auth(url, raise_errors=False):
    """Returns the Requests tuple auth for a given url from netrc."""

    netrc_file = os.environ.get("NETRC")
    if netrc_file is not None:
        netrc_locations = (netrc_file,)
    else:
        netrc_locations = (f"~/{f}" for f in NETRC_FILES)

    try:
        from netrc import NetrcParseError, netrc

        netrc_path = None

        for f in netrc_locations:
            try:
                loc = os.path.expanduser(f)
            except KeyError:
                # os.path.expanduser can fail when $HOME is undefined and
                # getpwuid fails. See https://bugs.python.org/issue20164 &
                # https://github.com/psf/requests/issues/1846
                return

            if os.path.exists(loc):
                netrc_path = loc
                break

        # Abort early if there isn't one.
        if netrc_path is None:
            return

        ri = urlparse(url)

        # Strip port numbers from netloc. This weird `if...encode`` dance is
        # used for Python 3.2, which doesn't support unicode literals.
        splitstr = b":"
        if isinstance(url, str):
            splitstr = splitstr.decode("ascii")
        host = ri.netloc.split(splitstr)[0]

        try:
            _netrc = netrc(netrc_path).authenticators(host)
            if _netrc:
                # Return with login / password
                login_i = 0 if _netrc[0] else 1
                return (_netrc[login_i], _netrc[2])
        except (NetrcParseError, OSError):
            # If there was a parsing error or a permissions issue reading the file,
            # we'll just skip netrc auth unless explicitly asked to raise errors.
            if raise_errors:
                raise

    # App Engine hackiness.
    except (ImportError, AttributeError):
        pass


def guess_filename(obj):
    """Tries to guess the filename of the given object."""
    name = getattr(obj, "name", None)
    if name and isinstance(name, basestring) and name[0] != "<" and name[-1] != ">":
        return os.path.basename(name)


def extract_zipped_paths(path):
    """Replace nonexistent paths that look like they refer to a member of a zip
    archive with the location of an extracted copy of the target, or else
    just return the provided path unchanged.
    """
    if os.path.exists(path):
        # this is already a valid path, no need to do anything further
        return path

    # find the first valid part of the provided path and treat that as a zip archive
    # assume the rest of the path is the name of a member in the archive
    archive, member = os.path.split(path)
    while archive and not os.path.exists(archive):
        archive, prefix = os.path.split(archive)
        if not prefix:
            # If we don't check for an empty prefix after the split (in other words, archive remains unchanged after the split),
            # we _can_ end up in an infinite loop on a rare corner case affecting a small number of users
            break
        member = "/".join([prefix, member])

    if not zipfile.is_zipfile(archive):
        return path

    zip_file = zipfile.ZipFile(archive)
    if member not in zip_file.namelist():
        return path

    # we have a valid zip archive and a valid member of that archive
    tmp = tempfile.gettempdir()
    extracted_path = os.path.join(tmp, member.split("/")[-1])
    if not os.path.exists(extracted_path):
        # use read + write to avoid the creating nested folders, we only want the file, avoids mkdir racing condition
        with atomic_open(extracted_path) as file_handler:
            file_handler.write(zip_file.read(member))
    return extracted_path


@contextlib.contextmanager
def atomic_open(filename):
    """Write a file to the disk in an atomic fashion"""
    tmp_descriptor, tmp_name = tempfile.mkstemp(dir=os.path.dirname(filename))
    try:
        with os.fdopen(tmp_descriptor, "wb") as tmp_handler:
            yield tmp_handler
        os.replace(tmp_name, filename)
    except BaseException:
        os.remove(tmp_name)
        raise


def from_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. Unless it can not be represented as such, return an
    OrderedDict, e.g.,

    ::

        >>> from_key_val_list([('key', 'val')])
        OrderedDict([('key', 'val')])
        >>> from_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples
        >>> from_key_val_list({'key': 'val'})
        OrderedDict([('key', 'val')])

    :rtype: OrderedDict
    """
    if value is None:
        return None

    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    return OrderedDict(value)


def to_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. If it can be, return a list of tuples, e.g.,

    ::

        >>> to_key_val_list([('key', 'val')])
        [('key', 'val')]
        >>> to_key_val_list({'key': 'val'})
        [('key', 'val')]
        >>> to_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples

    :rtype: list
    """
    if value is None:
        return None

    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    if isinstance(value, Mapping):
        value = value.items()

    return list(value)


# From mitsuhiko/werkzeug (used with permission).
def parse_list_header(value):
    """Parse lists as described by RFC 2068 Section 2.

    In particular, parse comma-separated lists where the elements of
    the list may include quoted-strings.  A quoted-string could
    contain a comma.  A non-quoted string could have quotes in the
    middle.  Quotes are removed automatically after parsing.

    It basically works like :func:`parse_set_header` just that items
    may appear multiple times and case sensitivity is preserved.

    The return value is a standard :class:`list`:

    >>> parse_list_header('token, "quoted value"')
    ['token', 'quoted value']

    To create a header from the :class:`list` again, use the
    :func:`dump_header` function.

    :param value: a string with a list header.
    :return: :class:`list`
    :rtype: list
    """
    result = []
    for item in _parse_list_header(value):
        if item[:1] == item[-1:] == '"':
            item = unquote_header_value(item[1:-1])
        result.append(item)
    return result


# From mitsuhiko/werkzeug (used with permission).
def parse_dict_header(value):
    """Parse lists of key, value pairs as described by RFC 2068 Section 2 and
    convert them into a python dict:

    >>> d = parse_dict_header('foo="is a fish", bar="as well"')
    >>> type(d) is dict
    True
    >>> sorted(d.items())
    [('bar', 'as well'), ('foo', 'is a fish')]

    If there is no value for a key it will be `None`:

    >>> parse_dict_header('key_without_value')
    {'key_without_value': None}

    To create a header from the :class:`dict` again, use the
    :func:`dump_header` function.

    :param value: a string with a dict header.
    :return: :class:`dict`
    :rtype: dict
    """
    result = {}
    for item in _parse_list_header(value):
        if "=" not in item:
            result[item] = None
            continue
        name, value = item.split("=", 1)
        if value[:1] == value[-1:] == '"':
            value = unquote_header_value(value[1:-1])
        result[name] = value
    return result


# From mitsuhiko/werkzeug (used with permission).
def unquote_header_value(value, is_filename=False):
    r"""Unquotes a header value.  (Reversal of :func:`quote_header_value`).
    This does not use the real unquoting but what browsers are actually
    using for quoting.

    :param value: the header value to unquote.
    :rtype: str
    """
    if value and value[0] == value[-1] == '"':
        # this is not the real unquoting, but fixing this so that the
        # RFC is met will result in bugs with internet explorer and
        # probably some other browsers as well.  IE for example is
        # uploading files with "C:\foo\bar.txt" as filename
        value = value[1:-1]

        # if this is a filename and the starting characters look like
        # a UNC path, then just return the value without quotes.  Using the
        # replace sequence below on a UNC path has the effect of turning
        # the leading double slash into a single slash and then
        # _fix_ie_filename() doesn't work correctly.  See #458.
        if not is_filename or value[:2] != "\\\\":
            return value.replace("\\\\", "\\").replace('\\"', '"')
    return value


def dict_from_cookiejar(cj):
    """Returns a key/value dictionary from a CookieJar.

    :param cj: CookieJar object to extract cookies from.
    :rtype: dict
    """

    cookie_dict = {cookie.name: cookie.value for cookie in cj}
    return cookie_dict


def add_dict_to_cookiejar(cj, cookie_dict):
    """Returns a CookieJar from a key/value dictionary.

    :param cj: CookieJar to insert cookies into.
    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :rtype: CookieJar
    """

    return cookiejar_from_dict(cookie_dict, cj)


def get_encodings_from_content(content):
    """Returns encodings from given content string.

    :param content: bytestring to extract encodings from.
    """
    warnings.warn(
        (
            "In requests 3.0, get_encodings_from_content will be removed. For "
            "more information, please see the discussion on issue #2266. (This"
            " warning should only appear once.)"
        ),
        DeprecationWarning,
    )

    charset_re = re.compile(r'<meta.*?charset=["\']*(.+?)["\'>]', flags=re.I)
    pragma_re = re.compile(r'<meta.*?content=["\']*;?charset=(.+?)["\'>]', flags=re.I)
    xml_re = re.compile(r'^<\?xml.*?encoding=["\']*(.+?)["\'>]')

    return (
        charset_re.findall(content)
        + pragma_re.findall(content)
        + xml_re.findall(content)
    )


def _parse_content_type_header(header):
    """Returns content type and parameters from given header

    :param header: string
    :return: tuple containing content type and dictionary of
         parameters
    """

    tokens = header.split(";")
    content_type, params = tokens[0].strip(), tokens[1:]
    params_dict = {}
    items_to_strip = "\"' "

    for param in params:
        param = param.strip()
        if param:
            key, value = param, True
            index_of_equals = param.find("=")
            if index_of_equals != -1:
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1 :].strip(items_to_strip)
            params_dict[key.lower()] = value
    return content_type, params_dict


def get_encoding_from_headers(headers):
    """Returns encodings from given HTTP Header Dict.

    :param headers: dictionary to extract encoding from.
    :rtype: str
    """

    content_type = headers.get("content-type")

    if not content_type:
        return None

    content_type, params = _parse_content_type_header(content_type)

    if "charset" in params:
        return params["charset"].strip("'\"")

    if "text" in content_type:
        return "ISO-8859-1"

    if "application/json" in content_type:
        # Assume UTF-8 based on RFC 4627: https://www.ietf.org/rfc/rfc4627.txt since the charset was unset
        return "utf-8"


def stream_decode_response_unicode(iterator, r):
    """Stream decodes an iterator."""

    if r.encoding is None:
        yield from iterator
        return

    decoder = codecs.getincrementaldecoder(r.encoding)(errors="replace")
    for chunk in iterator:
        rv = decoder.decode(chunk)
        if rv:
            yield rv
    rv = decoder.decode(b"", final=True)
    if rv:
        yield rv


def iter_slices(string, slice_length):
    """Iterate over slices of a string."""
    pos = 0
    if slice_length is None or slice_length <= 0:
        slice_length = len(string)
    while pos < len(string):
        yield string[pos : pos + slice_length]
        pos += slice_length


def get_unicode_from_response(r):
    """Returns the requested content back in unicode.

    :param r: Response object to get unicode content from.

    Tried:

    1. charset from content-type
    2. fall back and replace all unicode characters

    :rtype: str
    """
    warnings.warn(
        (
            "In requests 3.0, get_unicode_from_response will be removed. For "
            "more information, please see the discussion on issue #2266. (This"
            " warning should only appear once.)"
        ),
        DeprecationWarning,
    )

    tried_encodings = []

    # Try charset from content-type
    encoding = get_encoding_from_headers(r.headers)

    if encoding:
        try:
            return str(r.content, encoding)
        except UnicodeError:
            tried_encodings.append(encoding)

    # Fall back:
    try:
        return str(r.content, encoding, errors="replace")
    except TypeError:
        return r.content


# The unreserved URI characters (RFC 3986)
UNRESERVED_SET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + "0123456789-._~"
)


def unquote_unreserved(uri):
    """Un-escape any percent-escape sequences in a URI that are unreserved
    characters. This leaves all reserved, illegal and non-ASCII bytes encoded.

    :rtype: str
    """
    parts = uri.split("%")
    for i in range(1, len(parts)):
        h = parts[i][0:2]
        if len(h) == 2 and h.isalnum():
            try:
                c = chr(int(h, 16))
            except ValueError:
                raise InvalidURL(f"Invalid percent-escape sequence: '{h}'")

            if c in UNRESERVED_SET:
                parts[i] = c + parts[i][2:]
            else:
                parts[i] = f"%{parts[i]}"
        else:
            parts[i] = f"%{parts[i]}"
    return "".join(parts)


def requote_uri(uri):
    """Re-quote the given URI.

    This function passes the given URI through an unquote/quote cycle to
    ensure that it is fully and consistently quoted.

    :rtype: str
    """
    safe_with_percent = "!#$%&'()*+,/:;=?@[]~"
    safe_without_percent = "!#$&'()*+,/:;=?@[]~"
    try:
        # Unquote only the unreserved characters
        # Then quote only illegal characters (do not quote reserved,
        # unreserved, or '%')
        return quote(unquote_unreserved(uri), safe=safe_with_percent)
    except InvalidURL:
        # We couldn't unquote the given URI, so let's try quoting it, but
        # there may be unquoted '%'s in the URI. We need to make sure they're
        # properly quoted so they do not cause issues elsewhere.
        return quote(uri, safe=safe_without_percent)


def address_in_network(ip, net):
    """This function allows you to check if an IP belongs to a network subnet

    Example: returns True if ip = 192.168.1.1 and net = 192.168.1.0/24
             returns False if ip = 192.168.1.1 and net = 192.168.100.0/24

    :rtype: bool
    """
    ipaddr = struct.unpack("=L", socket.inet_aton(ip))[0]
    netaddr, bits = net.split("/")
    netmask = struct.unpack("=L", socket.inet_aton(dotted_netmask(int(bits))))[0]
    network = struct.unpack("=L", socket.inet_aton(netaddr))[0] & netmask
    return (ipaddr & netmask) == (network & netmask)


def dotted_netmask(mask):
    """Converts mask from /xx format to xxx.xxx.xxx.xxx

    Example: if mask is 24 function returns 255.255.255.0

    :rtype: str
    """
    bits = 0xFFFFFFFF ^ (1 << 32 - mask) - 1
    return socket.inet_ntoa(struct.pack(">I", bits))


def is_ipv4_address(string_ip):
    """
    :rtype: bool
    """
    try:
        socket.inet_aton(string_ip)
    except OSError:
        return False
    return True


def is_valid_cidr(string_network):
    """
    Very simple check of the cidr format in no_proxy variable.

    :rtype: bool
    """
    if string_network.count("/") == 1:
        try:
            mask = int(string_network.split("/")[1])
        except ValueError:
            return False

        if mask < 1 or mask > 32:
            return False

        try:
            socket.inet_aton(string_network.split("/")[0])
        except OSError:
            return False
    else:
        return False
    return True


@contextlib.contextmanager
def set_environ(env_name, value):
    """Set the environment variable 'env_name' to 'value'

    Save previous value, yield, and then restore the previous value stored in
    the environment variable 'env_name'.

    If 'value' is None, do nothing"""
    value_changed = value is not None
    if value_changed:
        old_value = os.environ.get(env_name)
        os.environ[env_name] = value
    try:
        yield
    finally:
        if value_changed:
            if old_value is None:
                del os.environ[env_name]
            else:
                os.environ[env_name] = old_value


def should_bypass_proxies(url, no_proxy):
    """
    Returns whether we should bypass proxies or not.

    :rtype: bool
    """

    # Prioritize lowercase environment variables over uppercase
    # to keep a consistent behaviour with other http projects (curl, wget).
    def get_proxy(key):
        return os.environ.get(key) or os.environ.get(key.upper())

    # First check whether no_proxy is defined. If it is, check that the URL
    # we're getting isn't in the no_proxy list.
    no_proxy_arg = no_proxy
    if no_proxy is None:
        no_proxy = get_proxy("no_proxy")
    parsed = urlparse(url)

    if parsed.hostname is None:
        # URLs don't always have hostnames, e.g. file:/// urls.
        return True

    if no_proxy:
        # We need to check whether we match here. We need to see if we match
        # the end of the hostname, both with and without the port.
        no_proxy = (host for host in no_proxy.replace(" ", "").split(",") if host)

        if is_ipv4_address(parsed.hostname):
            for proxy_ip in no_proxy:
                if is_valid_cidr(proxy_ip):
                    if address_in_network(parsed.hostname, proxy_ip):
                        return True
                elif parsed.hostname == proxy_ip:
                    # If no_proxy ip was defined in plain IP notation instead of cidr notation &
                    # matches the IP of the index
                    return True
        else:
            host_with_port = parsed.hostname
            if parsed.port:
                host_with_port += f":{parsed.port}"

            for host in no_proxy:
                if parsed.hostname.endswith(host) or host_with_port.endswith(host):
                    # The URL does match something in no_proxy, so we don't want
                    # to apply the proxies on this URL.
                    return True

    with set_environ("no_proxy", no_proxy_arg):
        # parsed.hostname can be `None` in cases such as a file URI.
        try:
            bypass = proxy_bypass(parsed.hostname)
        except (TypeError, socket.gaierror):
            bypass = False

    if bypass:
        return True

    return False


def get_environ_proxies(url, no_proxy=None):
    """
    Return a dict of environment proxies.

    :rtype: dict
    """
    if should_bypass_proxies(url, no_proxy=no_proxy):
        return {}
    else:
        return getproxies()


def select_proxy(url, proxies):
    """Select a proxy for the url, if applicable.

    :param url: The url being for the request
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    """
    proxies = proxies or {}
    urlparts = urlparse(url)
    if urlparts.hostname is None:
        return proxies.get(urlparts.scheme, proxies.get("all"))

    proxy_keys = [
        urlparts.scheme + "://" + urlparts.hostname,
        urlparts.scheme,
        "all://" + urlparts.hostname,
        "all",
    ]
    proxy = None
    for proxy_key in proxy_keys:
        if proxy_key in proxies:
            proxy = proxies[proxy_key]
            break

    return proxy


def resolve_proxies(request, proxies, trust_env=True):
    """This method takes proxy information from a request and configuration
    input to resolve a mapping of target proxies. This will consider settings
    such as NO_PROXY to strip proxy configurations.

    :param request: Request or PreparedRequest
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    :param trust_env: Boolean declaring whether to trust environment configs

    :rtype: dict
    """
    proxies = proxies if proxies is not None else {}
    url = request.url
    scheme = urlparse(url).scheme
    no_proxy = proxies.get("no_proxy")
    new_proxies = proxies.copy()

    if trust_env and not should_bypass_proxies(url, no_proxy=no_proxy):
        environ_proxies = get_environ_proxies(url, no_proxy=no_proxy)

        proxy = environ_proxies.get(scheme, environ_proxies.get("all"))

        if proxy:
            new_proxies.setdefault(scheme, proxy)
    return new_proxies


def default_user_agent(name="python-requests"):
    """
    Return a string representing the default user agent.

    :rtype: str
    """
    return f"{name}/{__version__}"


def default_headers():
    """
    :rtype: requests.structures.CaseInsensitiveDict
    """
    return CaseInsensitiveDict(
        {
            "User-Agent": default_user_agent(),
            "Accept-Encoding": DEFAULT_ACCEPT_ENCODING,
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )


def parse_header_links(value):
    """Return a list of parsed link headers proxies.

    i.e. Link: <http:/.../front.jpeg>; rel=front; type="image/jpeg",<http://.../back.jpeg>; rel=back;type="image/jpeg"

    :rtype: list
    """

    links = []

    replace_chars = " '\""

    value = value.strip(replace_chars)
    if not value:
        return links

    for val in re.split(", *<", value):
        try:
            url, params = val.split(";", 1)
        except ValueError:
            url, params = val, ""

        link = {"url": url.strip("<> '\"")}

        for param in params.split(";"):
            try:
                key, value = param.split("=")
            except ValueError:
                break

            link[key.strip(replace_chars)] = value.strip(replace_chars)

        links.append(link)

    return links


# Null bytes; no need to recreate these on each call to guess_json_utf
_null = "\x00".encode("ascii")  # encoding to ASCII for Python 3
_null2 = _null * 2
_null3 = _null * 3


def guess_json_utf(data):
    """
    :rtype: str
    """
    # JSON always starts with two ASCII characters, so detection is as
    # easy as counting the nulls and from their location and count
    # determine the encoding. Also detect a BOM, if present.
    sample = data[:4]
    if sample in (codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE):
        return "utf-32"  # BOM included
    if sample[:3] == codecs.BOM_UTF8:
        return "utf-8-sig"  # BOM included, MS style (discouraged)
    if sample[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return "utf-16"  # BOM included
    nullcount = sample.count(_null)
    if nullcount == 0:
        return "utf-8"
    if nullcount == 2:
        if sample[::2] == _null2:  # 1st and 3rd are null
            return "utf-16-be"
        if sample[1::2] == _null2:  # 2nd and 4th are null
            return "utf-16-le"
        # Did not detect 2 valid UTF-16 ascii-range characters
    if nullcount == 3:
        if sample[:3] == _null3:
            return "utf-32-be"
        if sample[1:] == _null3:
            return "utf-32-le"
        # Did not detect a valid UTF-32 ascii-range character
    return None


def prepend_scheme_if_needed(url, new_scheme):
    """Given a URL that may or may not have a scheme, prepend the given scheme.
    Does not replace a present scheme with the one provided as an argument.

    :rtype: str
    """
    parsed = parse_url(url)
    scheme, auth, host, port, path, query, fragment = parsed

    # A defect in urlparse determines that there isn't a netloc present in some
    # urls. We previously assumed parsing was overly cautious, and swapped the
    # netloc and path. Due to a lack of tests on the original defect, this is
    # maintained with parse_url for backwards compatibility.
    netloc = parsed.netloc
    if not netloc:
        netloc, path = path, netloc

    if auth:
        # parse_url doesn't provide the netloc with auth
        # so we'll add it ourselves.
        netloc = "@".join([auth, netloc])
    if scheme is None:
        scheme = new_scheme
    if path is None:
        path = ""

    return urlunparse((scheme, netloc, path, "", query, fragment))


def get_auth_from_url(url):
    """Given a url with authentication components, extract them into a tuple of
    username,password.

    :rtype: (str,str)
    """
    parsed = urlparse(url)

    try:
        auth = (unquote(parsed.username), unquote(parsed.password))
    except (AttributeError, TypeError):
        auth = ("", "")

    return auth


def check_header_validity(header):
    """Verifies that header parts don't contain leading whitespace
    reserved characters, or return characters.

    :param header: tuple, in the format (name, value).
    """
    name, value = header
    _validate_header_part(header, name, 0)
    _validate_header_part(header, value, 1)


def _validate_header_part(header, header_part, header_validator_index):
    if isinstance(header_part, str):
        validator = _HEADER_VALIDATORS_STR[header_validator_index]
    elif isinstance(header_part, bytes):
        validator = _HEADER_VALIDATORS_BYTE[header_validator_index]
    else:
        raise InvalidHeader(
            f"Header part ({header_part!r}) from {header} "
            f"must be of type str or bytes, not {type(header_part)}"
        )

    if not validator.match(header_part):
        header_kind = "name" if header_validator_index == 0 else "value"
        raise InvalidHeader(
            f"Invalid leading whitespace, reserved character(s), or return "
            f"character(s) in header {header_kind}: {header_part!r}"
        )


def urldefragauth(url):
    """
    Given a url remove the fragment and the authentication part.

    :rtype: str
    """
    scheme, netloc, path, params, query, fragment = urlparse(url)

    # see func:`prepend_scheme_if_needed`
    if not netloc:
        netloc, path = path, netloc

    netloc = netloc.rsplit("@", 1)[-1]

    return urlunparse((scheme, netloc, path, params, query, ""))


def rewind_body(prepared_request):
    """Move file pointer back to its recorded starting position
    so it can be read again on redirect.
    """
    body_seek = getattr(prepared_request.body, "seek", None)
    if body_seek is not None and isinstance(
        prepared_request._body_position, integer_types
    ):
        try:
            body_seek(prepared_request._body_position)
        except OSError:
            raise UnrewindableBodyError(
                "An error occurred when rewinding request body for redirect."
            )
    else:
        raise UnrewindableBodyError("Unable to rewind request body for redirect.")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\lie_group.py ===
r"""
This module contains the implementation of the internal helper functions for the lie_group hint for
dsolve. These helper functions apply different heuristics on the given equation
and return the solution. These functions are used by :py:meth:`sympy.solvers.ode.single.LieGroup`

References
=========

- `abaco1_simple`, `function_sum` and `chi`  are referenced from E.S Cheb-Terrab, L.G.S Duarte
and L.A,C.P da Mota, Computer Algebra Solving of First Order ODEs Using
Symmetry Methods, pp. 7 - pp. 8

- `abaco1_product`, `abaco2_similar`, `abaco2_unique_unknown`, `linear`  and `abaco2_unique_general`
are referenced from E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
ODE Patterns, pp. 7 - pp. 12

- `bivariate` from Lie Groups and Differential Equations pp. 327 - pp. 329

"""
from itertools import islice

from sympy.core import Add, S, Mul, Pow
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, AppliedUndef, expand
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.functions import exp, log
from sympy.integrals.integrals import integrate
from sympy.polys import Poly
from sympy.polys.polytools import cancel, div
from sympy.simplify import (collect, powsimp,  # type: ignore
    separatevars, simplify)
from sympy.solvers import solve
from sympy.solvers.pde import pdsolve

from sympy.utilities import numbered_symbols
from sympy.solvers.deutils import _preprocess, ode_order
from .ode import checkinfsol


lie_heuristics = (
    "abaco1_simple",
    "abaco1_product",
    "abaco2_similar",
    "abaco2_unique_unknown",
    "abaco2_unique_general",
    "linear",
    "function_sum",
    "bivariate",
    "chi"
    )


def _ode_lie_group_try_heuristic(eq, heuristic, func, match, inf):

    xi = Function("xi")
    eta = Function("eta")
    f = func.func
    x = func.args[0]
    y = match['y']
    h = match['h']
    tempsol = []
    if not inf:
        try:
            inf = infinitesimals(eq, hint=heuristic, func=func, order=1, match=match)
        except ValueError:
            return None
    for infsim in inf:
        xiinf = (infsim[xi(x, func)]).subs(func, y)
        etainf = (infsim[eta(x, func)]).subs(func, y)
        # This condition creates recursion while using pdsolve.
        # Since the first step while solving a PDE of form
        # a*(f(x, y).diff(x)) + b*(f(x, y).diff(y)) + c = 0
        # is to solve the ODE dy/dx = b/a
        if simplify(etainf/xiinf) == h:
            continue
        rpde = f(x, y).diff(x)*xiinf + f(x, y).diff(y)*etainf
        r = pdsolve(rpde, func=f(x, y)).rhs
        s = pdsolve(rpde - 1, func=f(x, y)).rhs
        newcoord = [_lie_group_remove(coord) for coord in [r, s]]
        r = Dummy("r")
        s = Dummy("s")
        C1 = Symbol("C1")
        rcoord = newcoord[0]
        scoord = newcoord[-1]
        try:
            sol = solve([r - rcoord, s - scoord], x, y, dict=True)
            if sol == []:
                continue
        except NotImplementedError:
            continue
        else:
            sol = sol[0]
            xsub = sol[x]
            ysub = sol[y]
            num = simplify(scoord.diff(x) + scoord.diff(y)*h)
            denom = simplify(rcoord.diff(x) + rcoord.diff(y)*h)
            if num and denom:
                diffeq = simplify((num/denom).subs([(x, xsub), (y, ysub)]))
                sep = separatevars(diffeq, symbols=[r, s], dict=True)
                if sep:
                    # Trying to separate, r and s coordinates
                    deq = integrate((1/sep[s]), s) + C1 - integrate(sep['coeff']*sep[r], r)
                    # Substituting and reverting back to original coordinates
                    deq = deq.subs([(r, rcoord), (s, scoord)])
                    try:
                        sdeq = solve(deq, y)
                    except NotImplementedError:
                        tempsol.append(deq)
                    else:
                        return [Eq(f(x), sol) for sol in sdeq]


            elif denom: # (ds/dr) is zero which means s is constant
                return [Eq(f(x), solve(scoord - C1, y)[0])]

            elif num: # (dr/ds) is zero which means r is constant
                return [Eq(f(x), solve(rcoord - C1, y)[0])]

    # If nothing works, return solution as it is, without solving for y
    if tempsol:
        return [Eq(sol.subs(y, f(x)), 0) for sol in tempsol]
    return None


def _ode_lie_group( s, func, order, match):

    heuristics = lie_heuristics
    inf = {}
    f = func.func
    x = func.args[0]
    df = func.diff(x)
    xi = Function("xi")
    eta = Function("eta")
    xis = match['xi']
    etas = match['eta']
    y = match.pop('y', None)
    if y:
        h = -simplify(match[match['d']]/match[match['e']])
    else:
        y = Dummy("y")
        h = s.subs(func, y)

    if xis is not None and etas is not None:
        inf = [{xi(x, f(x)): S(xis), eta(x, f(x)): S(etas)}]

        if checkinfsol(Eq(df, s), inf, func=f(x), order=1)[0][0]:
            heuristics = ["user_defined"] + list(heuristics)

    match = {'h': h, 'y': y}

    # This is done so that if any heuristic raises a ValueError
    # another heuristic can be used.
    sol = None
    for heuristic in heuristics:
        sol = _ode_lie_group_try_heuristic(Eq(df, s), heuristic, func, match, inf)
        if sol:
            return sol
    return sol


def infinitesimals(eq, func=None, order=None, hint='default', match=None):
    r"""
    The infinitesimal functions of an ordinary differential equation, `\xi(x,y)`
    and `\eta(x,y)`, are the infinitesimals of the Lie group of point transformations
    for which the differential equation is invariant. So, the ODE `y'=f(x,y)`
    would admit a Lie group `x^*=X(x,y;\varepsilon)=x+\varepsilon\xi(x,y)`,
    `y^*=Y(x,y;\varepsilon)=y+\varepsilon\eta(x,y)` such that `(y^*)'=f(x^*, y^*)`.
    A change of coordinates, to `r(x,y)` and `s(x,y)`, can be performed so this Lie group
    becomes the translation group, `r^*=r` and `s^*=s+\varepsilon`.
    They are tangents to the coordinate curves of the new system.

    Consider the transformation `(x, y) \to (X, Y)` such that the
    differential equation remains invariant. `\xi` and `\eta` are the tangents to
    the transformed coordinates `X` and `Y`, at `\varepsilon=0`.

    .. math:: \left(\frac{\partial X(x,y;\varepsilon)}{\partial\varepsilon
                }\right)|_{\varepsilon=0} = \xi,
              \left(\frac{\partial Y(x,y;\varepsilon)}{\partial\varepsilon
                }\right)|_{\varepsilon=0} = \eta,

    The infinitesimals can be found by solving the following PDE:

        >>> from sympy import Function, Eq, pprint
        >>> from sympy.abc import x, y
        >>> xi, eta, h = map(Function, ['xi', 'eta', 'h'])
        >>> h = h(x, y)  # dy/dx = h
        >>> eta = eta(x, y)
        >>> xi = xi(x, y)
        >>> genform = Eq(eta.diff(x) + (eta.diff(y) - xi.diff(x))*h
        ... - (xi.diff(y))*h**2 - xi*(h.diff(x)) - eta*(h.diff(y)), 0)
        >>> pprint(genform)
        /d               d           \                     d              2       d                       d             d
        |--(eta(x, y)) - --(xi(x, y))|*h(x, y) - eta(x, y)*--(h(x, y)) - h (x, y)*--(xi(x, y)) - xi(x, y)*--(h(x, y)) + --(eta(x, y)) = 0
        \dy              dx          /                     dy                     dy                      dx            dx

    Solving the above mentioned PDE is not trivial, and can be solved only by
    making intelligent assumptions for `\xi` and `\eta` (heuristics). Once an
    infinitesimal is found, the attempt to find more heuristics stops. This is done to
    optimise the speed of solving the differential equation. If a list of all the
    infinitesimals is needed, ``hint`` should be flagged as ``all``, which gives
    the complete list of infinitesimals. If the infinitesimals for a particular
    heuristic needs to be found, it can be passed as a flag to ``hint``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.solvers.ode.lie_group import infinitesimals
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = f(x).diff(x) - x**2*f(x)
    >>> infinitesimals(eq)
    [{eta(x, f(x)): exp(x**3/3), xi(x, f(x)): 0}]

    References
    ==========

    - Solving differential equations by Symmetry Groups,
      John Starrett, pp. 1 - pp. 14

    """

    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    if not func:
        eq, func = _preprocess(eq)
    variables = func.args
    if len(variables) != 1:
        raise ValueError("ODE's have only one independent variable")
    else:
        x = variables[0]
        if not order:
            order = ode_order(eq, func)
        if order != 1:
            raise NotImplementedError("Infinitesimals for only "
                "first order ODE's have been implemented")
        else:
            df = func.diff(x)
            # Matching differential equation of the form a*df + b
            a = Wild('a', exclude = [df])
            b = Wild('b', exclude = [df])
            if match:  # Used by lie_group hint
                h = match['h']
                y = match['y']
            else:
                match = collect(expand(eq), df).match(a*df + b)
                if match:
                    h = -simplify(match[b]/match[a])
                else:
                    try:
                        sol = solve(eq, df)
                    except NotImplementedError:
                        raise NotImplementedError("Infinitesimals for the "
                            "first order ODE could not be found")
                    else:
                        h = sol[0]  # Find infinitesimals for one solution
                y = Dummy("y")
                h = h.subs(func, y)

            u = Dummy("u")
            hx = h.diff(x)
            hy = h.diff(y)
            hinv = ((1/h).subs([(x, u), (y, x)])).subs(u, y)  # Inverse ODE
            match = {'h': h, 'func': func, 'hx': hx, 'hy': hy, 'y': y, 'hinv': hinv}
            if hint == 'all':
                xieta = []
                for heuristic in lie_heuristics:
                    function = globals()['lie_heuristic_' + heuristic]
                    inflist = function(match, comp=True)
                    if inflist:
                        xieta.extend([inf for inf in inflist if inf not in xieta])
                if xieta:
                    return xieta
                else:
                    raise NotImplementedError("Infinitesimals could not be found for "
                        "the given ODE")

            elif hint == 'default':
                for heuristic in lie_heuristics:
                    function = globals()['lie_heuristic_' + heuristic]
                    xieta = function(match, comp=False)
                    if xieta:
                        return xieta

                raise NotImplementedError("Infinitesimals could not be found for"
                    " the given ODE")

            elif hint not in lie_heuristics:
                raise ValueError("Heuristic not recognized: " + hint)

            else:
                function = globals()['lie_heuristic_' + hint]
                xieta = function(match, comp=True)
                if xieta:
                    return xieta
                else:
                    raise ValueError("Infinitesimals could not be found using the"
                         " given heuristic")


def lie_heuristic_abaco1_simple(match, comp=False):
    r"""
    The first heuristic uses the following four sets of
    assumptions on `\xi` and `\eta`

    .. math:: \xi = 0, \eta = f(x)

    .. math:: \xi = 0, \eta = f(y)

    .. math:: \xi = f(x), \eta = 0

    .. math:: \xi = f(y), \eta = 0

    The success of this heuristic is determined by algebraic factorisation.
    For the first assumption `\xi = 0` and `\eta` to be a function of `x`, the PDE

    .. math:: \frac{\partial \eta}{\partial x} + (\frac{\partial \eta}{\partial y}
                - \frac{\partial \xi}{\partial x})*h
                - \frac{\partial \xi}{\partial y}*h^{2}
                - \xi*\frac{\partial h}{\partial x} - \eta*\frac{\partial h}{\partial y} = 0

    reduces to `f'(x) - f\frac{\partial h}{\partial y} = 0`
    If `\frac{\partial h}{\partial y}` is a function of `x`, then this can usually
    be integrated easily. A similar idea is applied to the other 3 assumptions as well.


    References
    ==========

    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra
      Solving of First Order ODEs Using Symmetry Methods, pp. 8


    """

    xieta = []
    y = match['y']
    h = match['h']
    func = match['func']
    x = func.args[0]
    hx = match['hx']
    hy = match['hy']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    hysym = hy.free_symbols
    if y not in hysym:
        try:
            fx = exp(integrate(hy, x))
        except NotImplementedError:
            pass
        else:
            inf = {xi: S.Zero, eta: fx}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    factor = hy/h
    facsym = factor.free_symbols
    if x not in facsym:
        try:
            fy = exp(integrate(factor, y))
        except NotImplementedError:
            pass
        else:
            inf = {xi: S.Zero, eta: fy.subs(y, func)}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    factor = -hx/h
    facsym = factor.free_symbols
    if y not in facsym:
        try:
            fx = exp(integrate(factor, x))
        except NotImplementedError:
            pass
        else:
            inf = {xi: fx, eta: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    factor = -hx/(h**2)
    facsym = factor.free_symbols
    if x not in facsym:
        try:
            fy = exp(integrate(factor, y))
        except NotImplementedError:
            pass
        else:
            inf = {xi: fy.subs(y, func), eta: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    if xieta:
        return xieta

def lie_heuristic_abaco1_product(match, comp=False):
    r"""
    The second heuristic uses the following two assumptions on `\xi` and `\eta`

    .. math:: \eta = 0, \xi = f(x)*g(y)

    .. math:: \eta = f(x)*g(y), \xi = 0

    The first assumption of this heuristic holds good if
    `\frac{1}{h^{2}}\frac{\partial^2}{\partial x \partial y}\log(h)` is
    separable in `x` and `y`, then the separated factors containing `x`
    is `f(x)`, and `g(y)` is obtained by

    .. math:: e^{\int f\frac{\partial}{\partial x}\left(\frac{1}{f*h}\right)\,dy}

    provided `f\frac{\partial}{\partial x}\left(\frac{1}{f*h}\right)` is a function
    of `y` only.

    The second assumption holds good if `\frac{dy}{dx} = h(x, y)` is rewritten as
    `\frac{dy}{dx} = \frac{1}{h(y, x)}` and the same properties of the first assumption
    satisfies. After obtaining `f(x)` and `g(y)`, the coordinates are again
    interchanged, to get `\eta` as `f(x)*g(y)`


    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 7 - pp. 8

    """

    xieta = []
    y = match['y']
    h = match['h']
    hinv = match['hinv']
    func = match['func']
    x = func.args[0]
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)


    inf = separatevars(((log(h).diff(y)).diff(x))/h**2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx*((1/(fx*h)).diff(x)))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            inf = {eta: S.Zero, xi: (fx*gy).subs(y, func)}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    u1 = Dummy("u1")
    inf = separatevars(((log(hinv).diff(y)).diff(x))/hinv**2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx*((1/(fx*hinv)).diff(x)))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            etaval = fx*gy
            etaval = (etaval.subs([(x, u1), (y, x)])).subs(u1, y)
            inf = {eta: etaval.subs(y, func), xi: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    if xieta:
        return xieta

def lie_heuristic_bivariate(match, comp=False):
    r"""
    The third heuristic assumes the infinitesimals `\xi` and `\eta`
    to be bi-variate polynomials in `x` and `y`. The assumption made here
    for the logic below is that `h` is a rational function in `x` and `y`
    though that may not be necessary for the infinitesimals to be
    bivariate polynomials. The coefficients of the infinitesimals
    are found out by substituting them in the PDE and grouping similar terms
    that are polynomials and since they form a linear system, solve and check
    for non trivial solutions. The degree of the assumed bivariates
    are increased till a certain maximum value.

    References
    ==========
    - Lie Groups and Differential Equations
      pp. 327 - pp. 329

    """

    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    if h.is_rational_function():
        # The maximum degree that the infinitesimals can take is
        # calculated by this technique.
        etax, etay, etad, xix, xiy, xid = symbols("etax etay etad xix xiy xid")
        ipde = etax + (etay - xix)*h - xiy*h**2 - xid*hx - etad*hy
        num, denom = cancel(ipde).as_numer_denom()
        deg = Poly(num, x, y).total_degree()
        deta = Function('deta')(x, y)
        dxi = Function('dxi')(x, y)
        ipde = (deta.diff(x) + (deta.diff(y) - dxi.diff(x))*h - (dxi.diff(y))*h**2
            - dxi*hx - deta*hy)
        xieq = Symbol("xi0")
        etaeq = Symbol("eta0")

        for i in range(deg + 1):
            if i:
                xieq += Add(*[
                    Symbol("xi_" + str(power) + "_" + str(i - power))*x**power*y**(i - power)
                    for power in range(i + 1)])
                etaeq += Add(*[
                    Symbol("eta_" + str(power) + "_" + str(i - power))*x**power*y**(i - power)
                    for power in range(i + 1)])
            pden, denom = (ipde.subs({dxi: xieq, deta: etaeq}).doit()).as_numer_denom()
            pden = expand(pden)

            # If the individual terms are monomials, the coefficients
            # are grouped
            if pden.is_polynomial(x, y) and pden.is_Add:
                polyy = Poly(pden, x, y).as_dict()
            if polyy:
                symset = xieq.free_symbols.union(etaeq.free_symbols) - {x, y}
                soldict = solve(polyy.values(), *symset)
                if isinstance(soldict, list):
                    soldict = soldict[0]
                if any(soldict.values()):
                    xired = xieq.subs(soldict)
                    etared = etaeq.subs(soldict)
                    # Scaling is done by substituting one for the parameters
                    # This can be any number except zero.
                    dict_ = dict.fromkeys(symset, 1)
                    inf = {eta: etared.subs(dict_).subs(y, func),
                        xi: xired.subs(dict_).subs(y, func)}
                    return [inf]

def lie_heuristic_chi(match, comp=False):
    r"""
    The aim of the fourth heuristic is to find the function `\chi(x, y)`
    that satisfies the PDE `\frac{d\chi}{dx} + h\frac{d\chi}{dx}
    - \frac{\partial h}{\partial y}\chi = 0`.

    This assumes `\chi` to be a bivariate polynomial in `x` and `y`. By intuition,
    `h` should be a rational function in `x` and `y`. The method used here is
    to substitute a general binomial for `\chi` up to a certain maximum degree
    is reached. The coefficients of the polynomials, are calculated by by collecting
    terms of the same order in `x` and `y`.

    After finding `\chi`, the next step is to use `\eta = \xi*h + \chi`, to
    determine `\xi` and `\eta`. This can be done by dividing `\chi` by `h`
    which would give `-\xi` as the quotient and `\eta` as the remainder.


    References
    ==========
    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra
      Solving of First Order ODEs Using Symmetry Methods, pp. 8

    """

    h = match['h']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    if h.is_rational_function():
        schi, schix, schiy = symbols("schi, schix, schiy")
        cpde = schix + h*schiy - hy*schi
        num, denom = cancel(cpde).as_numer_denom()
        deg = Poly(num, x, y).total_degree()

        chi = Function('chi')(x, y)
        chix = chi.diff(x)
        chiy = chi.diff(y)
        cpde = chix + h*chiy - hy*chi
        chieq = Symbol("chi")
        for i in range(1, deg + 1):
            chieq += Add(*[
                Symbol("chi_" + str(power) + "_" + str(i - power))*x**power*y**(i - power)
                for power in range(i + 1)])
            cnum, cden = cancel(cpde.subs({chi : chieq}).doit()).as_numer_denom()
            cnum = expand(cnum)
            if cnum.is_polynomial(x, y) and cnum.is_Add:
                cpoly = Poly(cnum, x, y).as_dict()
                if cpoly:
                    solsyms = chieq.free_symbols - {x, y}
                    soldict = solve(cpoly.values(), *solsyms)
                    if isinstance(soldict, list):
                        soldict = soldict[0]
                    if any(soldict.values()):
                        chieq = chieq.subs(soldict)
                        dict_ = dict.fromkeys(solsyms, 1)
                        chieq = chieq.subs(dict_)
                        # After finding chi, the main aim is to find out
                        # eta, xi by the equation eta = xi*h + chi
                        # One method to set xi, would be rearranging it to
                        # (eta/h) - xi = (chi/h). This would mean dividing
                        # chi by h would give -xi as the quotient and eta
                        # as the remainder. Thanks to Sean Vig for suggesting
                        # this method.
                        xic, etac = div(chieq, h)
                        inf = {eta: etac.subs(y, func), xi: -xic.subs(y, func)}
                        return [inf]

def lie_heuristic_function_sum(match, comp=False):
    r"""
    This heuristic uses the following two assumptions on `\xi` and `\eta`

    .. math:: \eta = 0, \xi = f(x) + g(y)

    .. math:: \eta = f(x) + g(y), \xi = 0

    The first assumption of this heuristic holds good if

    .. math:: \frac{\partial}{\partial y}[(h\frac{\partial^{2}}{
                \partial x^{2}}(h^{-1}))^{-1}]

    is separable in `x` and `y`,

    1. The separated factors containing `y` is `\frac{\partial g}{\partial y}`.
       From this `g(y)` can be determined.
    2. The separated factors containing `x` is `f''(x)`.
    3. `h\frac{\partial^{2}}{\partial x^{2}}(h^{-1})` equals
       `\frac{f''(x)}{f(x) + g(y)}`. From this `f(x)` can be determined.

    The second assumption holds good if `\frac{dy}{dx} = h(x, y)` is rewritten as
    `\frac{dy}{dx} = \frac{1}{h(y, x)}` and the same properties of the first
    assumption satisfies. After obtaining `f(x)` and `g(y)`, the coordinates
    are again interchanged, to get `\eta` as `f(x) + g(y)`.

    For both assumptions, the constant factors are separated among `g(y)`
    and `f''(x)`, such that `f''(x)` obtained from 3] is the same as that
    obtained from 2]. If not possible, then this heuristic fails.


    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 7 - pp. 8

    """

    xieta = []
    h = match['h']
    func = match['func']
    hinv = match['hinv']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    for odefac in [h, hinv]:
        factor = odefac*((1/odefac).diff(x, 2))
        sep = separatevars((1/factor).diff(y), dict=True, symbols=[x, y])
        if sep and sep['coeff'] and sep[x].has(x) and sep[y].has(y):
            k = Dummy("k")
            try:
                gy = k*integrate(sep[y], y)
            except NotImplementedError:
                pass
            else:
                fdd = 1/(k*sep[x]*sep['coeff'])
                fx = simplify(fdd/factor - gy)
                check = simplify(fx.diff(x, 2) - fdd)
                if fx:
                    if not check:
                        fx = fx.subs(k, 1)
                        gy = (gy/k)
                    else:
                        sol = solve(check, k)
                        if sol:
                            sol = sol[0]
                            fx = fx.subs(k, sol)
                            gy = (gy/k)*sol
                        else:
                            continue
                    if odefac == hinv:  # Inverse ODE
                        fx = fx.subs(x, y)
                        gy = gy.subs(y, x)
                    etaval = factor_terms(fx + gy)
                    if etaval.is_Mul:
                        etaval = Mul(*[arg for arg in etaval.args if arg.has(x, y)])
                    if odefac == hinv:  # Inverse ODE
                        inf = {eta: etaval.subs(y, func), xi : S.Zero}
                    else:
                        inf = {xi: etaval.subs(y, func), eta : S.Zero}
                    if not comp:
                        return [inf]
                    else:
                        xieta.append(inf)

        if xieta:
            return xieta

def lie_heuristic_abaco2_similar(match, comp=False):
    r"""
    This heuristic uses the following two assumptions on `\xi` and `\eta`

    .. math:: \eta = g(x), \xi = f(x)

    .. math:: \eta = f(y), \xi = g(y)

    For the first assumption,

    1. First `\frac{\frac{\partial h}{\partial y}}{\frac{\partial^{2} h}{
       \partial yy}}` is calculated. Let us say this value is A

    2. If this is constant, then `h` is matched to the form `A(x) + B(x)e^{
       \frac{y}{C}}` then, `\frac{e^{\int \frac{A(x)}{C} \,dx}}{B(x)}` gives `f(x)`
       and `A(x)*f(x)` gives `g(x)`

    3. Otherwise `\frac{\frac{\partial A}{\partial X}}{\frac{\partial A}{
       \partial Y}} = \gamma` is calculated. If

       a] `\gamma` is a function of `x` alone

       b] `\frac{\gamma\frac{\partial h}{\partial y} - \gamma'(x) - \frac{
       \partial h}{\partial x}}{h + \gamma} = G` is a function of `x` alone.
       then, `e^{\int G \,dx}` gives `f(x)` and `-\gamma*f(x)` gives `g(x)`

    The second assumption holds good if `\frac{dy}{dx} = h(x, y)` is rewritten as
    `\frac{dy}{dx} = \frac{1}{h(y, x)}` and the same properties of the first assumption
    satisfies. After obtaining `f(x)` and `g(x)`, the coordinates are again
    interchanged, to get `\xi` as `f(x^*)` and `\eta` as `g(y^*)`

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """

    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    hinv = match['hinv']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    factor = cancel(h.diff(y)/h.diff(y, 2))
    factorx = factor.diff(x)
    factory = factor.diff(y)
    if not factor.has(x) and not factor.has(y):
        A = Wild('A', exclude=[y])
        B = Wild('B', exclude=[y])
        C = Wild('C', exclude=[x, y])
        match = h.match(A + B*exp(y/C))
        try:
            tau = exp(-integrate(match[A]/match[C]), x)/match[B]
        except NotImplementedError:
            pass
        else:
            gx = match[A]*tau
            return [{xi: tau, eta: gx}]

    else:
        gamma = cancel(factorx/factory)
        if not gamma.has(y):
            tauint = cancel((gamma*hy - gamma.diff(x) - hx)/(h + gamma))
            if not tauint.has(y):
                try:
                    tau = exp(integrate(tauint, x))
                except NotImplementedError:
                    pass
                else:
                    gx = -tau*gamma
                    return [{xi: tau, eta: gx}]

    factor = cancel(hinv.diff(y)/hinv.diff(y, 2))
    factorx = factor.diff(x)
    factory = factor.diff(y)
    if not factor.has(x) and not factor.has(y):
        A = Wild('A', exclude=[y])
        B = Wild('B', exclude=[y])
        C = Wild('C', exclude=[x, y])
        match = h.match(A + B*exp(y/C))
        try:
            tau = exp(-integrate(match[A]/match[C]), x)/match[B]
        except NotImplementedError:
            pass
        else:
            gx = match[A]*tau
            return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]

    else:
        gamma = cancel(factorx/factory)
        if not gamma.has(y):
            tauint = cancel((gamma*hinv.diff(y) - gamma.diff(x) - hinv.diff(x))/(
                hinv + gamma))
            if not tauint.has(y):
                try:
                    tau = exp(integrate(tauint, x))
                except NotImplementedError:
                    pass
                else:
                    gx = -tau*gamma
                    return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]


def lie_heuristic_abaco2_unique_unknown(match, comp=False):
    r"""
    This heuristic assumes the presence of unknown functions or known functions
    with non-integer powers.

    1. A list of all functions and non-integer powers containing x and y
    2. Loop over each element `f` in the list, find `\frac{\frac{\partial f}{\partial x}}{
       \frac{\partial f}{\partial x}} = R`

       If it is separable in `x` and `y`, let `X` be the factors containing `x`. Then

       a] Check if `\xi = X` and `\eta = -\frac{X}{R}` satisfy the PDE. If yes, then return
          `\xi` and `\eta`
       b] Check if `\xi = \frac{-R}{X}` and `\eta = -\frac{1}{X}` satisfy the PDE.
           If yes, then return `\xi` and `\eta`

       If not, then check if

       a] :math:`\xi = -R,\eta = 1`

       b] :math:`\xi = 1, \eta = -\frac{1}{R}`

       are solutions.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """

    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    funclist = []
    for atom in h.atoms(Pow):
        base, exp = atom.as_base_exp()
        if base.has(x) and base.has(y):
            if not exp.is_Integer:
                funclist.append(atom)

    for function in h.atoms(AppliedUndef):
        syms = function.free_symbols
        if x in syms and y in syms:
            funclist.append(function)

    for f in funclist:
        frac = cancel(f.diff(y)/f.diff(x))
        sep = separatevars(frac, dict=True, symbols=[x, y])
        if sep and sep['coeff']:
            xitry1 = sep[x]
            etatry1 = -1/(sep[y]*sep['coeff'])
            pde1 = etatry1.diff(y)*h - xitry1.diff(x)*h - xitry1*hx - etatry1*hy
            if not simplify(pde1):
                return [{xi: xitry1, eta: etatry1.subs(y, func)}]
            xitry2 = 1/etatry1
            etatry2 = 1/xitry1
            pde2 = etatry2.diff(x) - (xitry2.diff(y))*h**2 - xitry2*hx - etatry2*hy
            if not simplify(expand(pde2)):
                return [{xi: xitry2.subs(y, func), eta: etatry2}]

        else:
            etatry = -1/frac
            pde = etatry.diff(x) + etatry.diff(y)*h - hx - etatry*hy
            if not simplify(pde):
                return [{xi: S.One, eta: etatry.subs(y, func)}]
            xitry = -frac
            pde = -xitry.diff(x)*h -xitry.diff(y)*h**2 - xitry*hx -hy
            if not simplify(expand(pde)):
                return [{xi: xitry.subs(y, func), eta: S.One}]


def lie_heuristic_abaco2_unique_general(match, comp=False):
    r"""
    This heuristic finds if infinitesimals of the form `\eta = f(x)`, `\xi = g(y)`
    without making any assumptions on `h`.

    The complete sequence of steps is given in the paper mentioned below.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    A = hx.diff(y)
    B = hy.diff(y) + hy**2
    C = hx.diff(x) - hx**2

    if not (A and B and C):
        return

    Ax = A.diff(x)
    Ay = A.diff(y)
    Axy = Ax.diff(y)
    Axx = Ax.diff(x)
    Ayy = Ay.diff(y)
    D = simplify(2*Axy + hx*Ay - Ax*hy + (hx*hy + 2*A)*A)*A - 3*Ax*Ay
    if not D:
        E1 = simplify(3*Ax**2 + ((hx**2 + 2*C)*A - 2*Axx)*A)
        if E1:
            E2 = simplify((2*Ayy + (2*B - hy**2)*A)*A - 3*Ay**2)
            if not E2:
                E3 = simplify(
                    E1*((28*Ax + 4*hx*A)*A**3 - E1*(hy*A + Ay)) - E1.diff(x)*8*A**4)
                if not E3:
                    etaval = cancel((4*A**3*(Ax - hx*A) + E1*(hy*A - Ay))/(S(2)*A*E1))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            xival = -4*A**3*etaval/E1
                            if y not in xival:
                                return [{xi: xival, eta: etaval.subs(y, func)}]

    else:
        E1 = simplify((2*Ayy + (2*B - hy**2)*A)*A - 3*Ay**2)
        if E1:
            E2 = simplify(
                4*A**3*D - D**2 + E1*((2*Axx - (hx**2 + 2*C)*A)*A - 3*Ax**2))
            if not E2:
                E3 = simplify(
                   -(A*D)*E1.diff(y) + ((E1.diff(x) - hy*D)*A + 3*Ay*D +
                    (A*hx - 3*Ax)*E1)*E1)
                if not E3:
                    etaval = cancel(((A*hx - Ax)*E1 - (Ay + A*hy)*D)/(S(2)*A*D))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            xival = -E1*etaval/D
                            if y not in xival:
                                return [{xi: xival, eta: etaval.subs(y, func)}]


def lie_heuristic_linear(match, comp=False):
    r"""
    This heuristic assumes

    1. `\xi = ax + by + c` and
    2. `\eta = fx + gy + h`

    After substituting the following assumptions in the determining PDE, it
    reduces to

    .. math:: f + (g - a)h - bh^{2} - (ax + by + c)\frac{\partial h}{\partial x}
                 - (fx + gy + c)\frac{\partial h}{\partial y}

    Solving the reduced PDE obtained, using the method of characteristics, becomes
    impractical. The method followed is grouping similar terms and solving the system
    of linear equations obtained. The difference between the bivariate heuristic is that
    `h` need not be a rational function in this case.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """
    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    coeffdict = {}
    symbols = numbered_symbols("c", cls=Dummy)
    symlist = [next(symbols) for _ in islice(symbols, 6)]
    C0, C1, C2, C3, C4, C5 = symlist
    pde = C3 + (C4 - C0)*h - (C0*x + C1*y + C2)*hx - (C3*x + C4*y + C5)*hy - C1*h**2
    pde, denom = pde.as_numer_denom()
    pde = powsimp(expand(pde))
    if pde.is_Add:
        terms = pde.args
        for term in terms:
            if term.is_Mul:
                rem = Mul(*[m for m in term.args if not m.has(x, y)])
                xypart = term/rem
                if xypart not in coeffdict:
                    coeffdict[xypart] = rem
                else:
                    coeffdict[xypart] += rem
            else:
                if term not in coeffdict:
                    coeffdict[term] = S.One
                else:
                    coeffdict[term] += S.One

    sollist = coeffdict.values()
    soldict = solve(sollist, symlist)
    if soldict:
        if isinstance(soldict, list):
            soldict = soldict[0]
        subval = soldict.values()
        if any(t for t in subval):
            onedict = dict(zip(symlist, [1]*6))
            xival = C0*x + C1*func + C2
            etaval = C3*x + C4*func + C5
            xival = xival.subs(soldict)
            etaval = etaval.subs(soldict)
            xival = xival.subs(onedict)
            etaval = etaval.subs(onedict)
            return [{xi: xival, eta: etaval}]


def _lie_group_remove(coords):
    r"""
    This function is strictly meant for internal use by the Lie group ODE solving
    method. It replaces arbitrary functions returned by pdsolve as follows:

    1] If coords is an arbitrary function, then its argument is returned.
    2] An arbitrary function in an Add object is replaced by zero.
    3] An arbitrary function in a Mul object is replaced by one.
    4] If there is no arbitrary function coords is returned unchanged.

    Examples
    ========

    >>> from sympy.solvers.ode.lie_group import _lie_group_remove
    >>> from sympy import Function
    >>> from sympy.abc import x, y
    >>> F = Function("F")
    >>> eq = x**2*y
    >>> _lie_group_remove(eq)
    x**2*y
    >>> eq = F(x**2*y)
    >>> _lie_group_remove(eq)
    x**2*y
    >>> eq = x*y**2 + F(x**3)
    >>> _lie_group_remove(eq)
    x*y**2
    >>> eq = (F(x**3) + y)*x**4
    >>> _lie_group_remove(eq)
    x**4*y

    """
    if isinstance(coords, AppliedUndef):
        return coords.args[0]
    elif coords.is_Add:
        subfunc = coords.atoms(AppliedUndef)
        if subfunc:
            for func in subfunc:
                coords = coords.subs(func, 0)
        return coords
    elif coords.is_Pow:
        base, expr = coords.as_base_exp()
        base = _lie_group_remove(base)
        expr = _lie_group_remove(expr)
        return base**expr
    elif coords.is_Mul:
        mulargs = []
        coordargs = coords.args
        for arg in coordargs:
            if not isinstance(coords, AppliedUndef):
                mulargs.append(_lie_group_remove(arg))
        return Mul(*mulargs)
    return coords

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\window\ewm.py ===
from __future__ import annotations

import datetime
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.tslibs import Timedelta
import pandas._libs.window.aggregations as window_aggregations
from pandas.util._decorators import doc

from pandas.core.dtypes.common import (
    is_datetime64_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna

from pandas.core import common
from pandas.core.arrays.datetimelike import dtype_to_unit
from pandas.core.indexers.objects import (
    BaseIndexer,
    ExponentialMovingWindowIndexer,
    GroupbyIndexer,
)
from pandas.core.util.numba_ import (
    get_jit_arguments,
    maybe_use_numba,
)
from pandas.core.window.common import zsqrt
from pandas.core.window.doc import (
    _shared_docs,
    create_section_header,
    kwargs_numeric_only,
    numba_notes,
    template_header,
    template_returns,
    template_see_also,
    window_agg_numba_parameters,
)
from pandas.core.window.numba_ import (
    generate_numba_ewm_func,
    generate_numba_ewm_table_func,
)
from pandas.core.window.online import (
    EWMMeanState,
    generate_online_numba_ewma_func,
)
from pandas.core.window.rolling import (
    BaseWindow,
    BaseWindowGroupby,
)

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        TimedeltaConvertibleTypes,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame


def get_center_of_mass(
    comass: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
) -> float:
    valid_count = common.count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")

    # Convert to center of mass; domain checks ensure 0 < alpha <= 1
    if comass is not None:
        if comass < 0:
            raise ValueError("comass must satisfy: comass >= 0")
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        comass = (span - 1) / 2
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        comass = (1 - alpha) / alpha
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")

    return float(comass)


def _calculate_deltas(
    times: np.ndarray | NDFrame,
    halflife: float | TimedeltaConvertibleTypes | None,
) -> npt.NDArray[np.float64]:
    """
    Return the diff of the times divided by the half-life. These values are used in
    the calculation of the ewm mean.

    Parameters
    ----------
    times : np.ndarray, Series
        Times corresponding to the observations. Must be monotonically increasing
        and ``datetime64[ns]`` dtype.
    halflife : float, str, timedelta, optional
        Half-life specifying the decay

    Returns
    -------
    np.ndarray
        Diff of the times divided by the half-life
    """
    unit = dtype_to_unit(times.dtype)
    if isinstance(times, ABCSeries):
        times = times._values
    _times = np.asarray(times.view(np.int64), dtype=np.float64)
    _halflife = float(Timedelta(halflife).as_unit(unit)._value)
    return np.diff(_times) / _halflife


class ExponentialMovingWindow(BaseWindow):
    r"""
    Provide exponentially weighted (EW) calculations.

    Exactly one of ``com``, ``span``, ``halflife``, or ``alpha`` must be
    provided if ``times`` is not provided. If ``times`` is provided,
    ``halflife`` and one of ``com``, ``span`` or ``alpha`` may be provided.

    Parameters
    ----------
    com : float, optional
        Specify decay in terms of center of mass

        :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.

    span : float, optional
        Specify decay in terms of span

        :math:`\alpha = 2 / (span + 1)`, for :math:`span \geq 1`.

    halflife : float, str, timedelta, optional
        Specify decay in terms of half-life

        :math:`\alpha = 1 - \exp\left(-\ln(2) / halflife\right)`, for
        :math:`halflife > 0`.

        If ``times`` is specified, a timedelta convertible unit over which an
        observation decays to half its value. Only applicable to ``mean()``,
        and halflife value will not apply to the other functions.

    alpha : float, optional
        Specify smoothing factor :math:`\alpha` directly

        :math:`0 < \alpha \leq 1`.

    min_periods : int, default 0
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

    adjust : bool, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average).

        - When ``adjust=True`` (default), the EW function is calculated using weights
          :math:`w_i = (1 - \alpha)^i`. For example, the EW moving average of the series
          [:math:`x_0, x_1, ..., x_t`] would be:

        .. math::
            y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ... + (1 -
            \alpha)^t x_0}{1 + (1 - \alpha) + (1 - \alpha)^2 + ... + (1 - \alpha)^t}

        - When ``adjust=False``, the exponentially weighted function is calculated
          recursively:

        .. math::
            \begin{split}
                y_0 &= x_0\\
                y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,
            \end{split}
    ignore_na : bool, default False
        Ignore missing values when calculating weights.

        - When ``ignore_na=False`` (default), weights are based on absolute positions.
          For example, the weights of :math:`x_0` and :math:`x_2` used in calculating
          the final weighted average of [:math:`x_0`, None, :math:`x_2`] are
          :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
          :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.

        - When ``ignore_na=True``, weights are based
          on relative positions. For example, the weights of :math:`x_0` and :math:`x_2`
          used in calculating the final weighted average of
          [:math:`x_0`, None, :math:`x_2`] are :math:`1-\alpha` and :math:`1` if
          ``adjust=True``, and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.

    axis : {0, 1}, default 0
        If ``0`` or ``'index'``, calculate across the rows.

        If ``1`` or ``'columns'``, calculate across the columns.

        For `Series` this parameter is unused and defaults to 0.

    times : np.ndarray, Series, default None

        Only applicable to ``mean()``.

        Times corresponding to the observations. Must be monotonically increasing and
        ``datetime64[ns]`` dtype.

        If 1-D array like, a sequence with the same shape as the observations.

    method : str {'single', 'table'}, default 'single'
        .. versionadded:: 1.4.0

        Execute the rolling operation per single column or row (``'single'``)
        or over the entire object (``'table'``).

        This argument is only implemented when specifying ``engine='numba'``
        in the method call.

        Only applicable to ``mean()``

    Returns
    -------
    pandas.api.typing.ExponentialMovingWindow

    See Also
    --------
    rolling : Provides rolling window calculations.
    expanding : Provides expanding transformations.

    Notes
    -----
    See :ref:`Windowing Operations <window.exponentially_weighted>`
    for further usage details and examples.

    Examples
    --------
    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    >>> df.ewm(com=0.5).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213
    >>> df.ewm(alpha=2 / 3).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213

    **adjust**

    >>> df.ewm(com=0.5, adjust=True).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213
    >>> df.ewm(com=0.5, adjust=False).mean()
              B
    0  0.000000
    1  0.666667
    2  1.555556
    3  1.555556
    4  3.650794

    **ignore_na**

    >>> df.ewm(com=0.5, ignore_na=True).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.225000
    >>> df.ewm(com=0.5, ignore_na=False).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213

    **times**

    Exponentially weighted mean with weights calculated with a timedelta ``halflife``
    relative to ``times``.

    >>> times = ['2020-01-01', '2020-01-03', '2020-01-10', '2020-01-15', '2020-01-17']
    >>> df.ewm(halflife='4 days', times=pd.DatetimeIndex(times)).mean()
              B
    0  0.000000
    1  0.585786
    2  1.523889
    3  1.523889
    4  3.233686
    """

    _attributes = [
        "com",
        "span",
        "halflife",
        "alpha",
        "min_periods",
        "adjust",
        "ignore_na",
        "axis",
        "times",
        "method",
    ]

    def __init__(
        self,
        obj: NDFrame,
        com: float | None = None,
        span: float | None = None,
        halflife: float | TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: Axis = 0,
        times: np.ndarray | NDFrame | None = None,
        method: str = "single",
        *,
        selection=None,
    ) -> None:
        super().__init__(
            obj=obj,
            min_periods=1 if min_periods is None else max(int(min_periods), 1),
            on=None,
            center=False,
            closed=None,
            method=method,
            axis=axis,
            selection=selection,
        )
        self.com = com
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.times = times
        if self.times is not None:
            if not self.adjust:
                raise NotImplementedError("times is not supported with adjust=False.")
            times_dtype = getattr(self.times, "dtype", None)
            if not (
                is_datetime64_dtype(times_dtype)
                or isinstance(times_dtype, DatetimeTZDtype)
            ):
                raise ValueError("times must be datetime64 dtype.")
            if len(self.times) != len(obj):
                raise ValueError("times must be the same length as the object.")
            if not isinstance(self.halflife, (str, datetime.timedelta, np.timedelta64)):
                raise ValueError("halflife must be a timedelta convertible object")
            if isna(self.times).any():
                raise ValueError("Cannot convert NaT values to integer")
            self._deltas = _calculate_deltas(self.times, self.halflife)
            # Halflife is no longer applicable when calculating COM
            # But allow COM to still be calculated if the user passes other decay args
            if common.count_not_none(self.com, self.span, self.alpha) > 0:
                self._com = get_center_of_mass(self.com, self.span, None, self.alpha)
            else:
                self._com = 1.0
        else:
            if self.halflife is not None and isinstance(
                self.halflife, (str, datetime.timedelta, np.timedelta64)
            ):
                raise ValueError(
                    "halflife can only be a timedelta convertible argument if "
                    "times is not None."
                )
            # Without times, points are equally spaced
            self._deltas = np.ones(
                max(self.obj.shape[self.axis] - 1, 0), dtype=np.float64
            )
            self._com = get_center_of_mass(
                # error: Argument 3 to "get_center_of_mass" has incompatible type
                # "Union[float, Any, None, timedelta64, signedinteger[_64Bit]]";
                # expected "Optional[float]"
                self.com,
                self.span,
                self.halflife,  # type: ignore[arg-type]
                self.alpha,
            )

    def _check_window_bounds(
        self, start: np.ndarray, end: np.ndarray, num_vals: int
    ) -> None:
        # emw algorithms are iterative with each point
        # ExponentialMovingWindowIndexer "bounds" are the entire window
        pass

    def _get_window_indexer(self) -> BaseIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
        return ExponentialMovingWindowIndexer()

    def online(
        self, engine: str = "numba", engine_kwargs=None
    ) -> OnlineExponentialMovingWindow:
        """
        Return an ``OnlineExponentialMovingWindow`` object to calculate
        exponentially moving window aggregations in an online method.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        engine: str, default ``'numba'``
            Execution engine to calculate online aggregations.
            Applies to all supported aggregation methods.

        engine_kwargs : dict, default None
            Applies to all supported aggregation methods.

            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
              applied to the function

        Returns
        -------
        OnlineExponentialMovingWindow
        """
        return OnlineExponentialMovingWindow(
            obj=self.obj,
            com=self.com,
            span=self.span,
            halflife=self.halflife,
            alpha=self.alpha,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            axis=self.axis,
            times=self.times,
            engine=engine,
            engine_kwargs=engine_kwargs,
            selection=self._selection,
        )

    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        pandas.DataFrame.rolling.aggregate
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.ewm(alpha=0.5).mean()
                  A         B         C
        0  1.000000  4.000000  7.000000
        1  1.666667  4.666667  7.666667
        2  2.428571  5.428571  8.428571
        """
        ),
        klass="Series/Dataframe",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs):
        return super().aggregate(func, *args, **kwargs)

    agg = aggregate

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).mean()
        0    1.000000
        1    1.555556
        2    2.147541
        3    2.775068
        dtype: float64
        """
        ),
        window_method="ewm",
        aggregation_description="(exponential weighted moment) mean",
        agg_method="mean",
    )
    def mean(
        self,
        numeric_only: bool = False,
        engine=None,
        engine_kwargs=None,
    ):
        if maybe_use_numba(engine):
            if self.method == "single":
                func = generate_numba_ewm_func
            else:
                func = generate_numba_ewm_table_func
            ewm_func = func(
                **get_jit_arguments(engine_kwargs),
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=tuple(self._deltas),
                normalize=True,
            )
            return self._apply(ewm_func, name="mean")
        elif engine in ("cython", None):
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")

            deltas = None if self.times is None else self._deltas
            window_func = partial(
                window_aggregations.ewm,
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=deltas,
                normalize=True,
            )
            return self._apply(window_func, name="mean", numeric_only=numeric_only)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).sum()
        0    1.000
        1    2.800
        2    5.240
        3    8.192
        dtype: float64
        """
        ),
        window_method="ewm",
        aggregation_description="(exponential weighted moment) sum",
        agg_method="sum",
    )
    def sum(
        self,
        numeric_only: bool = False,
        engine=None,
        engine_kwargs=None,
    ):
        if not self.adjust:
            raise NotImplementedError("sum is not implemented with adjust=False")
        if maybe_use_numba(engine):
            if self.method == "single":
                func = generate_numba_ewm_func
            else:
                func = generate_numba_ewm_table_func
            ewm_func = func(
                **get_jit_arguments(engine_kwargs),
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=tuple(self._deltas),
                normalize=False,
            )
            return self._apply(ewm_func, name="sum")
        elif engine in ("cython", None):
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")

            deltas = None if self.times is None else self._deltas
            window_func = partial(
                window_aggregations.ewm,
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=deltas,
                normalize=False,
            )
            return self._apply(window_func, name="sum", numeric_only=numeric_only)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """\
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).std()
        0         NaN
        1    0.707107
        2    0.995893
        3    1.277320
        dtype: float64
        """
        ),
        window_method="ewm",
        aggregation_description="(exponential weighted moment) standard deviation",
        agg_method="std",
    )
    def std(self, bias: bool = False, numeric_only: bool = False):
        if (
            numeric_only
            and self._selected_obj.ndim == 1
            and not is_numeric_dtype(self._selected_obj.dtype)
        ):
            # Raise directly so error message says std instead of var
            raise NotImplementedError(
                f"{type(self).__name__}.std does not implement numeric_only"
            )
        return zsqrt(self.var(bias=bias, numeric_only=numeric_only))

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """\
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).var()
        0         NaN
        1    0.500000
        2    0.991803
        3    1.631547
        dtype: float64
        """
        ),
        window_method="ewm",
        aggregation_description="(exponential weighted moment) variance",
        agg_method="var",
    )
    def var(self, bias: bool = False, numeric_only: bool = False):
        window_func = window_aggregations.ewmcov
        wfunc = partial(
            window_func,
            com=self._com,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            bias=bias,
        )

        def var_func(values, begin, end, min_periods):
            return wfunc(values, begin, end, min_periods, values)

        return self._apply(var_func, name="var", numeric_only=numeric_only)

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """\
        other : Series or DataFrame , optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([10, 11, 13, 16])
        >>> ser1.ewm(alpha=.2).cov(ser2)
        0         NaN
        1    0.500000
        2    1.524590
        3    3.408836
        dtype: float64
        """
        ),
        window_method="ewm",
        aggregation_description="(exponential weighted moment) sample covariance",
        agg_method="cov",
    )
    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        bias: bool = False,
        numeric_only: bool = False,
    ):
        from pandas import Series

        self._validate_numeric_only("cov", numeric_only)

        def cov_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )
            result = window_aggregations.ewmcov(
                x_array,
                start,
                end,
                # error: Argument 4 to "ewmcov" has incompatible type
                # "Optional[int]"; expected "int"
                self.min_periods,  # type: ignore[arg-type]
                y_array,
                self._com,
                self.adjust,
                self.ignore_na,
                bias,
            )
            return Series(result, index=x.index, name=x.name, copy=False)

        return self._apply_pairwise(
            self._selected_obj, other, pairwise, cov_func, numeric_only
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """\
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        """
        ),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([10, 11, 13, 16])
        >>> ser1.ewm(alpha=.2).corr(ser2)
        0         NaN
        1    1.000000
        2    0.982821
        3    0.977802
        dtype: float64
        """
        ),
        window_method="ewm",
        aggregation_description="(exponential weighted moment) sample correlation",
        agg_method="corr",
    )
    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        numeric_only: bool = False,
    ):
        from pandas import Series

        self._validate_numeric_only("corr", numeric_only)

        def cov_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )

            def _cov(X, Y):
                return window_aggregations.ewmcov(
                    X,
                    start,
                    end,
                    min_periods,
                    Y,
                    self._com,
                    self.adjust,
                    self.ignore_na,
                    True,
                )

            with np.errstate(all="ignore"):
                cov = _cov(x_array, y_array)
                x_var = _cov(x_array, x_array)
                y_var = _cov(y_array, y_array)
                result = cov / zsqrt(x_var * y_var)
            return Series(result, index=x.index, name=x.name, copy=False)

        return self._apply_pairwise(
            self._selected_obj, other, pairwise, cov_func, numeric_only
        )


class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    """
    Provide an exponential moving window groupby implementation.
    """

    _attributes = ExponentialMovingWindow._attributes + BaseWindowGroupby._attributes

    def __init__(self, obj, *args, _grouper=None, **kwargs) -> None:
        super().__init__(obj, *args, _grouper=_grouper, **kwargs)

        if not obj.empty and self.times is not None:
            # sort the times and recalculate the deltas according to the groups
            groupby_order = np.concatenate(list(self._grouper.indices.values()))
            self._deltas = _calculate_deltas(
                self.times.take(groupby_order),
                self.halflife,
            )

    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
        window_indexer = GroupbyIndexer(
            groupby_indices=self._grouper.indices,
            window_indexer=ExponentialMovingWindowIndexer,
        )
        return window_indexer


class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    def __init__(
        self,
        obj: NDFrame,
        com: float | None = None,
        span: float | None = None,
        halflife: float | TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: Axis = 0,
        times: np.ndarray | NDFrame | None = None,
        engine: str = "numba",
        engine_kwargs: dict[str, bool] | None = None,
        *,
        selection=None,
    ) -> None:
        if times is not None:
            raise NotImplementedError(
                "times is not implemented with online operations."
            )
        super().__init__(
            obj=obj,
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            axis=axis,
            times=times,
            selection=selection,
        )
        self._mean = EWMMeanState(
            self._com, self.adjust, self.ignore_na, self.axis, obj.shape
        )
        if maybe_use_numba(engine):
            self.engine = engine
            self.engine_kwargs = engine_kwargs
        else:
            raise ValueError("'numba' is the only supported engine")

    def reset(self) -> None:
        """
        Reset the state captured by `update` calls.
        """
        self._mean.reset()

    def aggregate(self, func, *args, **kwargs):
        raise NotImplementedError("aggregate is not implemented.")

    def std(self, bias: bool = False, *args, **kwargs):
        raise NotImplementedError("std is not implemented.")

    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        numeric_only: bool = False,
    ):
        raise NotImplementedError("corr is not implemented.")

    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        bias: bool = False,
        numeric_only: bool = False,
    ):
        raise NotImplementedError("cov is not implemented.")

    def var(self, bias: bool = False, numeric_only: bool = False):
        raise NotImplementedError("var is not implemented.")

    def mean(self, *args, update=None, update_times=None, **kwargs):
        """
        Calculate an online exponentially weighted mean.

        Parameters
        ----------
        update: DataFrame or Series, default None
            New values to continue calculating the
            exponentially weighted mean from the last values and weights.
            Values should be float64 dtype.

            ``update`` needs to be ``None`` the first time the
            exponentially weighted mean is calculated.

        update_times: Series or 1-D np.ndarray, default None
            New times to continue calculating the
            exponentially weighted mean from the last values and weights.
            If ``None``, values are assumed to be evenly spaced
            in time.
            This feature is currently unsupported.

        Returns
        -------
        DataFrame or Series

        Examples
        --------
        >>> df = pd.DataFrame({"a": range(5), "b": range(5, 10)})
        >>> online_ewm = df.head(2).ewm(0.5).online()
        >>> online_ewm.mean()
              a     b
        0  0.00  5.00
        1  0.75  5.75
        >>> online_ewm.mean(update=df.tail(3))
                  a         b
        2  1.615385  6.615385
        3  2.550000  7.550000
        4  3.520661  8.520661
        >>> online_ewm.reset()
        >>> online_ewm.mean()
              a     b
        0  0.00  5.00
        1  0.75  5.75
        """
        result_kwargs = {}
        is_frame = self._selected_obj.ndim == 2
        if update_times is not None:
            raise NotImplementedError("update_times is not implemented.")
        update_deltas = np.ones(
            max(self._selected_obj.shape[self.axis - 1] - 1, 0), dtype=np.float64
        )
        if update is not None:
            if self._mean.last_ewm is None:
                raise ValueError(
                    "Must call mean with update=None first before passing update"
                )
            result_from = 1
            result_kwargs["index"] = update.index
            if is_frame:
                last_value = self._mean.last_ewm[np.newaxis, :]
                result_kwargs["columns"] = update.columns
            else:
                last_value = self._mean.last_ewm
                result_kwargs["name"] = update.name
            np_array = np.concatenate((last_value, update.to_numpy()))
        else:
            result_from = 0
            result_kwargs["index"] = self._selected_obj.index
            if is_frame:
                result_kwargs["columns"] = self._selected_obj.columns
            else:
                result_kwargs["name"] = self._selected_obj.name
            np_array = self._selected_obj.astype(np.float64, copy=False).to_numpy()
        ewma_func = generate_online_numba_ewma_func(
            **get_jit_arguments(self.engine_kwargs)
        )
        result = self._mean.run_ewm(
            np_array if is_frame else np_array[:, np.newaxis],
            update_deltas,
            self.min_periods,
            ewma_func,
        )
        if not is_frame:
            result = result.squeeze()
        result = result[result_from:]
        result = self._selected_obj._constructor(result, **result_kwargs)
        return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\mutable.py ===
# ext/mutable.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

r"""Provide support for tracking of in-place changes to scalar values,
which are propagated into ORM change events on owning parent objects.

.. _mutable_scalars:

Establishing Mutability on Scalar Column Values
===============================================

A typical example of a "mutable" structure is a Python dictionary.
Following the example introduced in :ref:`types_toplevel`, we
begin with a custom type that marshals Python dictionaries into
JSON strings before being persisted::

    from sqlalchemy.types import TypeDecorator, VARCHAR
    import json


    class JSONEncodedDict(TypeDecorator):
        "Represents an immutable structure as a json-encoded string."

        impl = VARCHAR

        def process_bind_param(self, value, dialect):
            if value is not None:
                value = json.dumps(value)
            return value

        def process_result_value(self, value, dialect):
            if value is not None:
                value = json.loads(value)
            return value

The usage of ``json`` is only for the purposes of example. The
:mod:`sqlalchemy.ext.mutable` extension can be used
with any type whose target Python type may be mutable, including
:class:`.PickleType`, :class:`_postgresql.ARRAY`, etc.

When using the :mod:`sqlalchemy.ext.mutable` extension, the value itself
tracks all parents which reference it.  Below, we illustrate a simple
version of the :class:`.MutableDict` dictionary object, which applies
the :class:`.Mutable` mixin to a plain Python dictionary::

    from sqlalchemy.ext.mutable import Mutable


    class MutableDict(Mutable, dict):
        @classmethod
        def coerce(cls, key, value):
            "Convert plain dictionaries to MutableDict."

            if not isinstance(value, MutableDict):
                if isinstance(value, dict):
                    return MutableDict(value)

                # this call will raise ValueError
                return Mutable.coerce(key, value)
            else:
                return value

        def __setitem__(self, key, value):
            "Detect dictionary set events and emit change events."

            dict.__setitem__(self, key, value)
            self.changed()

        def __delitem__(self, key):
            "Detect dictionary del events and emit change events."

            dict.__delitem__(self, key)
            self.changed()

The above dictionary class takes the approach of subclassing the Python
built-in ``dict`` to produce a dict
subclass which routes all mutation events through ``__setitem__``.  There are
variants on this approach, such as subclassing ``UserDict.UserDict`` or
``collections.MutableMapping``; the part that's important to this example is
that the :meth:`.Mutable.changed` method is called whenever an in-place
change to the datastructure takes place.

We also redefine the :meth:`.Mutable.coerce` method which will be used to
convert any values that are not instances of ``MutableDict``, such
as the plain dictionaries returned by the ``json`` module, into the
appropriate type.  Defining this method is optional; we could just as well
created our ``JSONEncodedDict`` such that it always returns an instance
of ``MutableDict``, and additionally ensured that all calling code
uses ``MutableDict`` explicitly.  When :meth:`.Mutable.coerce` is not
overridden, any values applied to a parent object which are not instances
of the mutable type will raise a ``ValueError``.

Our new ``MutableDict`` type offers a class method
:meth:`~.Mutable.as_mutable` which we can use within column metadata
to associate with types. This method grabs the given type object or
class and associates a listener that will detect all future mappings
of this type, applying event listening instrumentation to the mapped
attribute. Such as, with classical table metadata::

    from sqlalchemy import Table, Column, Integer

    my_data = Table(
        "my_data",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("data", MutableDict.as_mutable(JSONEncodedDict)),
    )

Above, :meth:`~.Mutable.as_mutable` returns an instance of ``JSONEncodedDict``
(if the type object was not an instance already), which will intercept any
attributes which are mapped against this type.  Below we establish a simple
mapping against the ``my_data`` table::

    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy.orm import Mapped
    from sqlalchemy.orm import mapped_column


    class Base(DeclarativeBase):
        pass


    class MyDataClass(Base):
        __tablename__ = "my_data"
        id: Mapped[int] = mapped_column(primary_key=True)
        data: Mapped[dict[str, str]] = mapped_column(
            MutableDict.as_mutable(JSONEncodedDict)
        )

The ``MyDataClass.data`` member will now be notified of in place changes
to its value.

Any in-place changes to the ``MyDataClass.data`` member
will flag the attribute as "dirty" on the parent object::

    >>> from sqlalchemy.orm import Session

    >>> sess = Session(some_engine)
    >>> m1 = MyDataClass(data={"value1": "foo"})
    >>> sess.add(m1)
    >>> sess.commit()

    >>> m1.data["value1"] = "bar"
    >>> assert m1 in sess.dirty
    True

The ``MutableDict`` can be associated with all future instances
of ``JSONEncodedDict`` in one step, using
:meth:`~.Mutable.associate_with`.  This is similar to
:meth:`~.Mutable.as_mutable` except it will intercept all occurrences
of ``MutableDict`` in all mappings unconditionally, without
the need to declare it individually::

    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy.orm import Mapped
    from sqlalchemy.orm import mapped_column

    MutableDict.associate_with(JSONEncodedDict)


    class Base(DeclarativeBase):
        pass


    class MyDataClass(Base):
        __tablename__ = "my_data"
        id: Mapped[int] = mapped_column(primary_key=True)
        data: Mapped[dict[str, str]] = mapped_column(JSONEncodedDict)

Supporting Pickling
--------------------

The key to the :mod:`sqlalchemy.ext.mutable` extension relies upon the
placement of a ``weakref.WeakKeyDictionary`` upon the value object, which
stores a mapping of parent mapped objects keyed to the attribute name under
which they are associated with this value. ``WeakKeyDictionary`` objects are
not picklable, due to the fact that they contain weakrefs and function
callbacks. In our case, this is a good thing, since if this dictionary were
picklable, it could lead to an excessively large pickle size for our value
objects that are pickled by themselves outside of the context of the parent.
The developer responsibility here is only to provide a ``__getstate__`` method
that excludes the :meth:`~MutableBase._parents` collection from the pickle
stream::

    class MyMutableType(Mutable):
        def __getstate__(self):
            d = self.__dict__.copy()
            d.pop("_parents", None)
            return d

With our dictionary example, we need to return the contents of the dict itself
(and also restore them on __setstate__)::

    class MutableDict(Mutable, dict):
        # ....

        def __getstate__(self):
            return dict(self)

        def __setstate__(self, state):
            self.update(state)

In the case that our mutable value object is pickled as it is attached to one
or more parent objects that are also part of the pickle, the :class:`.Mutable`
mixin will re-establish the :attr:`.Mutable._parents` collection on each value
object as the owning parents themselves are unpickled.

Receiving Events
----------------

The :meth:`.AttributeEvents.modified` event handler may be used to receive
an event when a mutable scalar emits a change event.  This event handler
is called when the :func:`.attributes.flag_modified` function is called
from within the mutable extension::

    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy.orm import Mapped
    from sqlalchemy.orm import mapped_column
    from sqlalchemy import event


    class Base(DeclarativeBase):
        pass


    class MyDataClass(Base):
        __tablename__ = "my_data"
        id: Mapped[int] = mapped_column(primary_key=True)
        data: Mapped[dict[str, str]] = mapped_column(
            MutableDict.as_mutable(JSONEncodedDict)
        )


    @event.listens_for(MyDataClass.data, "modified")
    def modified_json(instance, initiator):
        print("json value modified:", instance.data)

.. _mutable_composites:

Establishing Mutability on Composites
=====================================

Composites are a special ORM feature which allow a single scalar attribute to
be assigned an object value which represents information "composed" from one
or more columns from the underlying mapped table. The usual example is that of
a geometric "point", and is introduced in :ref:`mapper_composite`.

As is the case with :class:`.Mutable`, the user-defined composite class
subclasses :class:`.MutableComposite` as a mixin, and detects and delivers
change events to its parents via the :meth:`.MutableComposite.changed` method.
In the case of a composite class, the detection is usually via the usage of the
special Python method ``__setattr__()``. In the example below, we expand upon the ``Point``
class introduced in :ref:`mapper_composite` to include
:class:`.MutableComposite` in its bases and to route attribute set events via
``__setattr__`` to the :meth:`.MutableComposite.changed` method::

    import dataclasses
    from sqlalchemy.ext.mutable import MutableComposite


    @dataclasses.dataclass
    class Point(MutableComposite):
        x: int
        y: int

        def __setattr__(self, key, value):
            "Intercept set events"

            # set the attribute
            object.__setattr__(self, key, value)

            # alert all parents to the change
            self.changed()

The :class:`.MutableComposite` class makes use of class mapping events to
automatically establish listeners for any usage of :func:`_orm.composite` that
specifies our ``Point`` type. Below, when ``Point`` is mapped to the ``Vertex``
class, listeners are established which will route change events from ``Point``
objects to each of the ``Vertex.start`` and ``Vertex.end`` attributes::

    from sqlalchemy.orm import DeclarativeBase, Mapped
    from sqlalchemy.orm import composite, mapped_column


    class Base(DeclarativeBase):
        pass


    class Vertex(Base):
        __tablename__ = "vertices"

        id: Mapped[int] = mapped_column(primary_key=True)

        start: Mapped[Point] = composite(
            mapped_column("x1"), mapped_column("y1")
        )
        end: Mapped[Point] = composite(
            mapped_column("x2"), mapped_column("y2")
        )

        def __repr__(self):
            return f"Vertex(start={self.start}, end={self.end})"

Any in-place changes to the ``Vertex.start`` or ``Vertex.end`` members
will flag the attribute as "dirty" on the parent object:

.. sourcecode:: python+sql

    >>> from sqlalchemy.orm import Session
    >>> sess = Session(engine)
    >>> v1 = Vertex(start=Point(3, 4), end=Point(12, 15))
    >>> sess.add(v1)
    {sql}>>> sess.flush()
    BEGIN (implicit)
    INSERT INTO vertices (x1, y1, x2, y2) VALUES (?, ?, ?, ?)
    [...] (3, 4, 12, 15)

    {stop}>>> v1.end.x = 8
    >>> assert v1 in sess.dirty
    True
    {sql}>>> sess.commit()
    UPDATE vertices SET x2=? WHERE vertices.id = ?
    [...] (8, 1)
    COMMIT

Coercing Mutable Composites
---------------------------

The :meth:`.MutableBase.coerce` method is also supported on composite types.
In the case of :class:`.MutableComposite`, the :meth:`.MutableBase.coerce`
method is only called for attribute set operations, not load operations.
Overriding the :meth:`.MutableBase.coerce` method is essentially equivalent
to using a :func:`.validates` validation routine for all attributes which
make use of the custom composite type::

    @dataclasses.dataclass
    class Point(MutableComposite):
        # other Point methods
        # ...

        def coerce(cls, key, value):
            if isinstance(value, tuple):
                value = Point(*value)
            elif not isinstance(value, Point):
                raise ValueError("tuple or Point expected")
            return value

Supporting Pickling
--------------------

As is the case with :class:`.Mutable`, the :class:`.MutableComposite` helper
class uses a ``weakref.WeakKeyDictionary`` available via the
:meth:`MutableBase._parents` attribute which isn't picklable. If we need to
pickle instances of ``Point`` or its owning class ``Vertex``, we at least need
to define a ``__getstate__`` that doesn't include the ``_parents`` dictionary.
Below we define both a ``__getstate__`` and a ``__setstate__`` that package up
the minimal form of our ``Point`` class::

    @dataclasses.dataclass
    class Point(MutableComposite):
        # ...

        def __getstate__(self):
            return self.x, self.y

        def __setstate__(self, state):
            self.x, self.y = state

As with :class:`.Mutable`, the :class:`.MutableComposite` augments the
pickling process of the parent's object-relational state so that the
:meth:`MutableBase._parents` collection is restored to all ``Point`` objects.

"""  # noqa: E501

from __future__ import annotations

from collections import defaultdict
from typing import AbstractSet
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from weakref import WeakKeyDictionary

from .. import event
from .. import inspect
from .. import types
from .. import util
from ..orm import Mapper
from ..orm._typing import _ExternalEntityType
from ..orm._typing import _O
from ..orm._typing import _T
from ..orm.attributes import AttributeEventToken
from ..orm.attributes import flag_modified
from ..orm.attributes import InstrumentedAttribute
from ..orm.attributes import QueryableAttribute
from ..orm.context import QueryContext
from ..orm.decl_api import DeclarativeAttributeIntercept
from ..orm.state import InstanceState
from ..orm.unitofwork import UOWTransaction
from ..sql._typing import _TypeEngineArgument
from ..sql.base import SchemaEventTarget
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import memoized_property
from ..util.typing import SupportsIndex
from ..util.typing import TypeGuard

_KT = TypeVar("_KT")  # Key type.
_VT = TypeVar("_VT")  # Value type.


class MutableBase:
    """Common base class to :class:`.Mutable`
    and :class:`.MutableComposite`.

    """

    @memoized_property
    def _parents(self) -> WeakKeyDictionary[Any, Any]:
        """Dictionary of parent object's :class:`.InstanceState`->attribute
        name on the parent.

        This attribute is a so-called "memoized" property.  It initializes
        itself with a new ``weakref.WeakKeyDictionary`` the first time
        it is accessed, returning the same object upon subsequent access.

        .. versionchanged:: 1.4 the :class:`.InstanceState` is now used
           as the key in the weak dictionary rather than the instance
           itself.

        """

        return weakref.WeakKeyDictionary()

    @classmethod
    def coerce(cls, key: str, value: Any) -> Optional[Any]:
        """Given a value, coerce it into the target type.

        Can be overridden by custom subclasses to coerce incoming
        data into a particular type.

        By default, raises ``ValueError``.

        This method is called in different scenarios depending on if
        the parent class is of type :class:`.Mutable` or of type
        :class:`.MutableComposite`.  In the case of the former, it is called
        for both attribute-set operations as well as during ORM loading
        operations.  For the latter, it is only called during attribute-set
        operations; the mechanics of the :func:`.composite` construct
        handle coercion during load operations.


        :param key: string name of the ORM-mapped attribute being set.
        :param value: the incoming value.
        :return: the method should return the coerced value, or raise
         ``ValueError`` if the coercion cannot be completed.

        """
        if value is None:
            return None
        msg = "Attribute '%s' does not accept objects of type %s"
        raise ValueError(msg % (key, type(value)))

    @classmethod
    def _get_listen_keys(cls, attribute: QueryableAttribute[Any]) -> Set[str]:
        """Given a descriptor attribute, return a ``set()`` of the attribute
        keys which indicate a change in the state of this attribute.

        This is normally just ``set([attribute.key])``, but can be overridden
        to provide for additional keys.  E.g. a :class:`.MutableComposite`
        augments this set with the attribute keys associated with the columns
        that comprise the composite value.

        This collection is consulted in the case of intercepting the
        :meth:`.InstanceEvents.refresh` and
        :meth:`.InstanceEvents.refresh_flush` events, which pass along a list
        of attribute names that have been refreshed; the list is compared
        against this set to determine if action needs to be taken.

        """
        return {attribute.key}

    @classmethod
    def _listen_on_attribute(
        cls,
        attribute: QueryableAttribute[Any],
        coerce: bool,
        parent_cls: _ExternalEntityType[Any],
    ) -> None:
        """Establish this type as a mutation listener for the given
        mapped descriptor.

        """
        key = attribute.key
        if parent_cls is not attribute.class_:
            return

        # rely on "propagate" here
        parent_cls = attribute.class_

        listen_keys = cls._get_listen_keys(attribute)

        def load(state: InstanceState[_O], *args: Any) -> None:
            """Listen for objects loaded or refreshed.

            Wrap the target data member's value with
            ``Mutable``.

            """
            val = state.dict.get(key, None)
            if val is not None:
                if coerce:
                    val = cls.coerce(key, val)
                    state.dict[key] = val
                val._parents[state] = key

        def load_attrs(
            state: InstanceState[_O],
            ctx: Union[object, QueryContext, UOWTransaction],
            attrs: Iterable[Any],
        ) -> None:
            if not attrs or listen_keys.intersection(attrs):
                load(state)

        def set_(
            target: InstanceState[_O],
            value: MutableBase | None,
            oldvalue: MutableBase | None,
            initiator: AttributeEventToken,
        ) -> MutableBase | None:
            """Listen for set/replace events on the target
            data member.

            Establish a weak reference to the parent object
            on the incoming value, remove it for the one
            outgoing.

            """
            if value is oldvalue:
                return value

            if not isinstance(value, cls):
                value = cls.coerce(key, value)
            if value is not None:
                value._parents[target] = key
            if isinstance(oldvalue, cls):
                oldvalue._parents.pop(inspect(target), None)
            return value

        def pickle(
            state: InstanceState[_O], state_dict: Dict[str, Any]
        ) -> None:
            val = state.dict.get(key, None)
            if val is not None:
                if "ext.mutable.values" not in state_dict:
                    state_dict["ext.mutable.values"] = defaultdict(list)
                state_dict["ext.mutable.values"][key].append(val)

        def unpickle(
            state: InstanceState[_O], state_dict: Dict[str, Any]
        ) -> None:
            if "ext.mutable.values" in state_dict:
                collection = state_dict["ext.mutable.values"]
                if isinstance(collection, list):
                    # legacy format
                    for val in collection:
                        val._parents[state] = key
                else:
                    for val in state_dict["ext.mutable.values"][key]:
                        val._parents[state] = key

        event.listen(
            parent_cls,
            "_sa_event_merge_wo_load",
            load,
            raw=True,
            propagate=True,
        )

        event.listen(parent_cls, "load", load, raw=True, propagate=True)
        event.listen(
            parent_cls, "refresh", load_attrs, raw=True, propagate=True
        )
        event.listen(
            parent_cls, "refresh_flush", load_attrs, raw=True, propagate=True
        )
        event.listen(
            attribute, "set", set_, raw=True, retval=True, propagate=True
        )
        event.listen(parent_cls, "pickle", pickle, raw=True, propagate=True)
        event.listen(
            parent_cls, "unpickle", unpickle, raw=True, propagate=True
        )


class Mutable(MutableBase):
    """Mixin that defines transparent propagation of change
    events to a parent object.

    See the example in :ref:`mutable_scalars` for usage information.

    """

    def changed(self) -> None:
        """Subclasses should call this method whenever change events occur."""

        for parent, key in self._parents.items():
            flag_modified(parent.obj(), key)

    @classmethod
    def associate_with_attribute(
        cls, attribute: InstrumentedAttribute[_O]
    ) -> None:
        """Establish this type as a mutation listener for the given
        mapped descriptor.

        """
        cls._listen_on_attribute(attribute, True, attribute.class_)

    @classmethod
    def associate_with(cls, sqltype: type) -> None:
        """Associate this wrapper with all future mapped columns
        of the given type.

        This is a convenience method that calls
        ``associate_with_attribute`` automatically.

        .. warning::

           The listeners established by this method are *global*
           to all mappers, and are *not* garbage collected.   Only use
           :meth:`.associate_with` for types that are permanent to an
           application, not with ad-hoc types else this will cause unbounded
           growth in memory usage.

        """

        def listen_for_type(mapper: Mapper[_O], class_: type) -> None:
            if mapper.non_primary:
                return
            for prop in mapper.column_attrs:
                if isinstance(prop.columns[0].type, sqltype):
                    cls.associate_with_attribute(getattr(class_, prop.key))

        event.listen(Mapper, "mapper_configured", listen_for_type)

    @classmethod
    def as_mutable(cls, sqltype: _TypeEngineArgument[_T]) -> TypeEngine[_T]:
        """Associate a SQL type with this mutable Python type.

        This establishes listeners that will detect ORM mappings against
        the given type, adding mutation event trackers to those mappings.

        The type is returned, unconditionally as an instance, so that
        :meth:`.as_mutable` can be used inline::

            Table(
                "mytable",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("data", MyMutableType.as_mutable(PickleType)),
            )

        Note that the returned type is always an instance, even if a class
        is given, and that only columns which are declared specifically with
        that type instance receive additional instrumentation.

        To associate a particular mutable type with all occurrences of a
        particular type, use the :meth:`.Mutable.associate_with` classmethod
        of the particular :class:`.Mutable` subclass to establish a global
        association.

        .. warning::

           The listeners established by this method are *global*
           to all mappers, and are *not* garbage collected.   Only use
           :meth:`.as_mutable` for types that are permanent to an application,
           not with ad-hoc types else this will cause unbounded growth
           in memory usage.

        """
        sqltype = types.to_instance(sqltype)

        # a SchemaType will be copied when the Column is copied,
        # and we'll lose our ability to link that type back to the original.
        # so track our original type w/ columns
        if isinstance(sqltype, SchemaEventTarget):

            @event.listens_for(sqltype, "before_parent_attach")
            def _add_column_memo(
                sqltyp: TypeEngine[Any],
                parent: Column[_T],
            ) -> None:
                parent.info["_ext_mutable_orig_type"] = sqltyp

            schema_event_check = True
        else:
            schema_event_check = False

        def listen_for_type(
            mapper: Mapper[_T],
            class_: Union[DeclarativeAttributeIntercept, type],
        ) -> None:
            if mapper.non_primary:
                return
            _APPLIED_KEY = "_ext_mutable_listener_applied"

            for prop in mapper.column_attrs:
                if (
                    # all Mutable types refer to a Column that's mapped,
                    # since this is the only kind of Core target the ORM can
                    # "mutate"
                    isinstance(prop.expression, Column)
                    and (
                        (
                            schema_event_check
                            and prop.expression.info.get(
                                "_ext_mutable_orig_type"
                            )
                            is sqltype
                        )
                        or prop.expression.type is sqltype
                    )
                ):
                    if not prop.expression.info.get(_APPLIED_KEY, False):
                        prop.expression.info[_APPLIED_KEY] = True
                        cls.associate_with_attribute(getattr(class_, prop.key))

        event.listen(Mapper, "mapper_configured", listen_for_type)

        return sqltype


class MutableComposite(MutableBase):
    """Mixin that defines transparent propagation of change
    events on a SQLAlchemy "composite" object to its
    owning parent or parents.

    See the example in :ref:`mutable_composites` for usage information.

    """

    @classmethod
    def _get_listen_keys(cls, attribute: QueryableAttribute[_O]) -> Set[str]:
        return {attribute.key}.union(attribute.property._attribute_keys)

    def changed(self) -> None:
        """Subclasses should call this method whenever change events occur."""

        for parent, key in self._parents.items():
            prop = parent.mapper.get_property(key)
            for value, attr_name in zip(
                prop._composite_values_from_instance(self),
                prop._attribute_keys,
            ):
                setattr(parent.obj(), attr_name, value)


def _setup_composite_listener() -> None:
    def _listen_for_type(mapper: Mapper[_T], class_: type) -> None:
        for prop in mapper.iterate_properties:
            if (
                hasattr(prop, "composite_class")
                and isinstance(prop.composite_class, type)
                and issubclass(prop.composite_class, MutableComposite)
            ):
                prop.composite_class._listen_on_attribute(
                    getattr(class_, prop.key), False, class_
                )

    if not event.contains(Mapper, "mapper_configured", _listen_for_type):
        event.listen(Mapper, "mapper_configured", _listen_for_type)


_setup_composite_listener()


class MutableDict(Mutable, Dict[_KT, _VT]):
    """A dictionary type that implements :class:`.Mutable`.

    The :class:`.MutableDict` object implements a dictionary that will
    emit change events to the underlying mapping when the contents of
    the dictionary are altered, including when values are added or removed.

    Note that :class:`.MutableDict` does **not** apply mutable tracking to  the
    *values themselves* inside the dictionary. Therefore it is not a sufficient
    solution for the use case of tracking deep changes to a *recursive*
    dictionary structure, such as a JSON structure.  To support this use case,
    build a subclass of  :class:`.MutableDict` that provides appropriate
    coercion to the values placed in the dictionary so that they too are
    "mutable", and emit events up to their parent structure.

    .. seealso::

        :class:`.MutableList`

        :class:`.MutableSet`

    """

    def __setitem__(self, key: _KT, value: _VT) -> None:
        """Detect dictionary set events and emit change events."""
        super().__setitem__(key, value)
        self.changed()

    if TYPE_CHECKING:
        # from https://github.com/python/mypy/issues/14858

        @overload
        def setdefault(
            self: MutableDict[_KT, Optional[_T]], key: _KT, value: None = None
        ) -> Optional[_T]: ...

        @overload
        def setdefault(self, key: _KT, value: _VT) -> _VT: ...

        def setdefault(self, key: _KT, value: object = None) -> object: ...

    else:

        def setdefault(self, *arg):  # noqa: F811
            result = super().setdefault(*arg)
            self.changed()
            return result

    def __delitem__(self, key: _KT) -> None:
        """Detect dictionary del events and emit change events."""
        super().__delitem__(key)
        self.changed()

    def update(self, *a: Any, **kw: _VT) -> None:
        super().update(*a, **kw)
        self.changed()

    if TYPE_CHECKING:

        @overload
        def pop(self, __key: _KT) -> _VT: ...

        @overload
        def pop(self, __key: _KT, __default: _VT | _T) -> _VT | _T: ...

        def pop(
            self, __key: _KT, __default: _VT | _T | None = None
        ) -> _VT | _T: ...

    else:

        def pop(self, *arg):  # noqa: F811
            result = super().pop(*arg)
            self.changed()
            return result

    def popitem(self) -> Tuple[_KT, _VT]:
        result = super().popitem()
        self.changed()
        return result

    def clear(self) -> None:
        super().clear()
        self.changed()

    @classmethod
    def coerce(cls, key: str, value: Any) -> MutableDict[_KT, _VT] | None:
        """Convert plain dictionary to instance of this class."""
        if not isinstance(value, cls):
            if isinstance(value, dict):
                return cls(value)
            return Mutable.coerce(key, value)
        else:
            return value

    def __getstate__(self) -> Dict[_KT, _VT]:
        return dict(self)

    def __setstate__(
        self, state: Union[Dict[str, int], Dict[str, str]]
    ) -> None:
        self.update(state)


class MutableList(Mutable, List[_T]):
    """A list type that implements :class:`.Mutable`.

    The :class:`.MutableList` object implements a list that will
    emit change events to the underlying mapping when the contents of
    the list are altered, including when values are added or removed.

    Note that :class:`.MutableList` does **not** apply mutable tracking to  the
    *values themselves* inside the list. Therefore it is not a sufficient
    solution for the use case of tracking deep changes to a *recursive*
    mutable structure, such as a JSON structure.  To support this use case,
    build a subclass of  :class:`.MutableList` that provides appropriate
    coercion to the values placed in the dictionary so that they too are
    "mutable", and emit events up to their parent structure.

    .. seealso::

        :class:`.MutableDict`

        :class:`.MutableSet`

    """

    def __reduce_ex__(
        self, proto: SupportsIndex
    ) -> Tuple[type, Tuple[List[int]]]:
        return (self.__class__, (list(self),))

    # needed for backwards compatibility with
    # older pickles
    def __setstate__(self, state: Iterable[_T]) -> None:
        self[:] = state

    def is_scalar(self, value: _T | Iterable[_T]) -> TypeGuard[_T]:
        return not util.is_non_string_iterable(value)

    def is_iterable(self, value: _T | Iterable[_T]) -> TypeGuard[Iterable[_T]]:
        return util.is_non_string_iterable(value)

    def __setitem__(
        self, index: SupportsIndex | slice, value: _T | Iterable[_T]
    ) -> None:
        """Detect list set events and emit change events."""
        if isinstance(index, SupportsIndex) and self.is_scalar(value):
            super().__setitem__(index, value)
        elif isinstance(index, slice) and self.is_iterable(value):
            super().__setitem__(index, value)
        self.changed()

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        """Detect list del events and emit change events."""
        super().__delitem__(index)
        self.changed()

    def pop(self, *arg: SupportsIndex) -> _T:
        result = super().pop(*arg)
        self.changed()
        return result

    def append(self, x: _T) -> None:
        super().append(x)
        self.changed()

    def extend(self, x: Iterable[_T]) -> None:
        super().extend(x)
        self.changed()

    def __iadd__(self, x: Iterable[_T]) -> MutableList[_T]:  # type: ignore[override,misc] # noqa: E501
        self.extend(x)
        return self

    def insert(self, i: SupportsIndex, x: _T) -> None:
        super().insert(i, x)
        self.changed()

    def remove(self, i: _T) -> None:
        super().remove(i)
        self.changed()

    def clear(self) -> None:
        super().clear()
        self.changed()

    def sort(self, **kw: Any) -> None:
        super().sort(**kw)
        self.changed()

    def reverse(self) -> None:
        super().reverse()
        self.changed()

    @classmethod
    def coerce(
        cls, key: str, value: MutableList[_T] | _T
    ) -> Optional[MutableList[_T]]:
        """Convert plain list to instance of this class."""
        if not isinstance(value, cls):
            if isinstance(value, list):
                return cls(value)
            return Mutable.coerce(key, value)
        else:
            return value


class MutableSet(Mutable, Set[_T]):
    """A set type that implements :class:`.Mutable`.

    The :class:`.MutableSet` object implements a set that will
    emit change events to the underlying mapping when the contents of
    the set are altered, including when values are added or removed.

    Note that :class:`.MutableSet` does **not** apply mutable tracking to  the
    *values themselves* inside the set. Therefore it is not a sufficient
    solution for the use case of tracking deep changes to a *recursive*
    mutable structure.  To support this use case,
    build a subclass of  :class:`.MutableSet` that provides appropriate
    coercion to the values placed in the dictionary so that they too are
    "mutable", and emit events up to their parent structure.

    .. seealso::

        :class:`.MutableDict`

        :class:`.MutableList`


    """

    def update(self, *arg: Iterable[_T]) -> None:
        super().update(*arg)
        self.changed()

    def intersection_update(self, *arg: Iterable[Any]) -> None:
        super().intersection_update(*arg)
        self.changed()

    def difference_update(self, *arg: Iterable[Any]) -> None:
        super().difference_update(*arg)
        self.changed()

    def symmetric_difference_update(self, *arg: Iterable[_T]) -> None:
        super().symmetric_difference_update(*arg)
        self.changed()

    def __ior__(self, other: AbstractSet[_T]) -> MutableSet[_T]:  # type: ignore[override,misc] # noqa: E501
        self.update(other)
        return self

    def __iand__(self, other: AbstractSet[object]) -> MutableSet[_T]:
        self.intersection_update(other)
        return self

    def __ixor__(self, other: AbstractSet[_T]) -> MutableSet[_T]:  # type: ignore[override,misc] # noqa: E501
        self.symmetric_difference_update(other)
        return self

    def __isub__(self, other: AbstractSet[object]) -> MutableSet[_T]:  # type: ignore[misc] # noqa: E501
        self.difference_update(other)
        return self

    def add(self, elem: _T) -> None:
        super().add(elem)
        self.changed()

    def remove(self, elem: _T) -> None:
        super().remove(elem)
        self.changed()

    def discard(self, elem: _T) -> None:
        super().discard(elem)
        self.changed()

    def pop(self, *arg: Any) -> _T:
        result = super().pop(*arg)
        self.changed()
        return result

    def clear(self) -> None:
        super().clear()
        self.changed()

    @classmethod
    def coerce(cls, index: str, value: Any) -> Optional[MutableSet[_T]]:
        """Convert plain set to instance of this class."""
        if not isinstance(value, cls):
            if isinstance(value, set):
                return cls(value)
            return Mutable.coerce(index, value)
        else:
            return value

    def __getstate__(self) -> Set[_T]:
        return set(self)

    def __setstate__(self, state: Iterable[_T]) -> None:
        self.update(state)

    def __reduce_ex__(
        self, proto: SupportsIndex
    ) -> Tuple[type, Tuple[List[int]]]:
        return (self.__class__, (list(self),))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\dense.py ===
from __future__ import annotations
import random

from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence

from .exceptions import ShapeError
from .decompositions import _cholesky, _LDLdecomposition
from .matrixbase import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .solvers import _lower_triangular_solve, _upper_triangular_solve


__doctest_requires__ = {('symarray',): ['numpy']}


def _iszero(x):
    """Returns True if x is zero."""
    return x.is_zero


class DenseMatrix(RepMatrix):
    """Matrix implementation based on DomainMatrix as the internal representation"""

    #
    # DenseMatrix is a superclass for both MutableDenseMatrix and
    # ImmutableDenseMatrix. Methods shared by both classes but not for the
    # Sparse classes should be implemented here.
    #

    is_MatrixExpr: bool = False

    _op_priority = 10.01
    _class_priority = 4

    @property
    def _mat(self):
        sympy_deprecation_warning(
            """
            The private _mat attribute of Matrix is deprecated. Use the
            .flat() method instead.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-private-matrix-attributes"
        )

        return self.flat()

    def _eval_inverse(self, **kwargs):
        return self.inv(method=kwargs.get('method', 'GE'),
                        iszerofunc=kwargs.get('iszerofunc', _iszero),
                        try_block_diag=kwargs.get('try_block_diag', False))

    def as_immutable(self):
        """Returns an Immutable version of this Matrix
        """
        from .immutable import ImmutableDenseMatrix as cls
        return cls._fromrep(self._rep.copy())

    def as_mutable(self):
        """Returns a mutable version of this matrix

        Examples
        ========

        >>> from sympy import ImmutableMatrix
        >>> X = ImmutableMatrix([[1, 2], [3, 4]])
        >>> Y = X.as_mutable()
        >>> Y[1, 1] = 5 # Can set values in Y
        >>> Y
        Matrix([
        [1, 2],
        [3, 5]])
        """
        return Matrix(self)

    def cholesky(self, hermitian=True):
        return _cholesky(self, hermitian=hermitian)

    def LDLdecomposition(self, hermitian=True):
        return _LDLdecomposition(self, hermitian=hermitian)

    def lower_triangular_solve(self, rhs):
        return _lower_triangular_solve(self, rhs)

    def upper_triangular_solve(self, rhs):
        return _upper_triangular_solve(self, rhs)

    cholesky.__doc__               = _cholesky.__doc__
    LDLdecomposition.__doc__       = _LDLdecomposition.__doc__
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__


def _force_mutable(x):
    """Return a matrix as a Matrix, otherwise return x."""
    if getattr(x, 'is_Matrix', False):
        return x.as_mutable()
    elif isinstance(x, Basic):
        return x
    elif hasattr(x, '__array__'):
        a = x.__array__()
        if len(a.shape) == 0:
            return sympify(a)
        return Matrix(x)
    return x


class MutableDenseMatrix(DenseMatrix, MutableRepMatrix):

    def simplify(self, **kwargs):
        """Applies simplify to the elements of a matrix in place.

        This is a shortcut for M.applyfunc(lambda x: simplify(x, ratio, measure))

        See Also
        ========

        sympy.simplify.simplify.simplify
        """
        from sympy.simplify.simplify import simplify as _simplify
        for (i, j), element in self.todok().items():
            self[i, j] = _simplify(element, **kwargs)


MutableMatrix = Matrix = MutableDenseMatrix

###########
# Numpy Utility Functions:
# list2numpy, matrix2numpy, symmarray
###########


def list2numpy(l, dtype=object):  # pragma: no cover
    """Converts Python list of SymPy expressions to a NumPy array.

    See Also
    ========

    matrix2numpy
    """
    from numpy import empty
    a = empty(len(l), dtype)
    for i, s in enumerate(l):
        a[i] = s
    return a


def matrix2numpy(m, dtype=object):  # pragma: no cover
    """Converts SymPy's matrix to a NumPy array.

    See Also
    ========

    list2numpy
    """
    from numpy import empty
    a = empty(m.shape, dtype)
    for i in range(m.rows):
        for j in range(m.cols):
            a[i, j] = m[i, j]
    return a


###########
# Rotation matrices:
# rot_givens, rot_axis[123], rot_ccw_axis[123]
###########


def rot_givens(i, j, theta, dim=3):
    r"""Returns a a Givens rotation matrix, a a rotation in the
    plane spanned by two coordinates axes.

    Explanation
    ===========

    The Givens rotation corresponds to a generalization of rotation
    matrices to any number of dimensions, given by:

    .. math::
        G(i, j, \theta) =
            \begin{bmatrix}
                1   & \cdots &    0   & \cdots &    0   & \cdots &    0   \\
                \vdots & \ddots & \vdots &        & \vdots &        & \vdots \\
                0   & \cdots &    c   & \cdots &   -s   & \cdots &    0   \\
                \vdots &        & \vdots & \ddots & \vdots &        & \vdots \\
                0   & \cdots &    s   & \cdots &    c   & \cdots &    0   \\
                \vdots &        & \vdots &        & \vdots & \ddots & \vdots \\
                0   & \cdots &    0   & \cdots &    0   & \cdots &    1
            \end{bmatrix}

    Where $c = \cos(\theta)$ and $s = \sin(\theta)$ appear at the intersections
    ``i``\th and ``j``\th rows and columns.

    For fixed ``i > j``\, the non-zero elements of a Givens matrix are
    given by:

    - $g_{kk} = 1$ for $k \ne i,\,j$
    - $g_{kk} = c$ for $k = i,\,j$
    - $g_{ji} = -g_{ij} = -s$

    Parameters
    ==========

    i : int between ``0`` and ``dim - 1``
        Represents first axis
    j : int between ``0`` and ``dim - 1``
        Represents second axis
    dim : int bigger than 1
        Number of dimensions. Defaults to 3.

    Examples
    ========

    >>> from sympy import pi, rot_givens

    A counterclockwise rotation of pi/3 (60 degrees) around
    the third axis (z-axis):

    >>> rot_givens(1, 0, pi/3)
    Matrix([
    [      1/2, -sqrt(3)/2, 0],
    [sqrt(3)/2,        1/2, 0],
    [        0,          0, 1]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_givens(1, 0, pi/2)
    Matrix([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]])

    This can be generalized to any number
    of dimensions:

    >>> rot_givens(1, 0, pi/2, dim=4)
    Matrix([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 1, 0],
    [0,  0, 0, 1]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Givens_rotation

    See Also
    ========

    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (clockwise around the x axis)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    if not isinstance(dim, int) or dim < 2:
        raise ValueError('dim must be an integer biggen than one, '
                         'got {}.'.format(dim))

    if i == j:
        raise ValueError('i and j must be different, '
                         'got ({}, {})'.format(i, j))

    for ij in [i, j]:
        if not isinstance(ij, int) or ij < 0 or ij > dim - 1:
            raise ValueError('i and j must be integers between 0 and '
                             '{}, got i={} and j={}.'.format(dim-1, i, j))

    theta = sympify(theta)
    c = cos(theta)
    s = sin(theta)
    M = eye(dim)
    M[i, i] = c
    M[j, j] = c
    M[i, j] = s
    M[j, i] = -s
    return M


def rot_axis3(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 3-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    clockwise rotation around the `z`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                 \cos(\theta) & \sin(\theta) & 0 \\
                -\sin(\theta) & \cos(\theta) & 0 \\
                            0 &            0 & 1
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_axis3

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_axis3(theta)
    Matrix([
    [       1/2, sqrt(3)/2, 0],
    [-sqrt(3)/2,       1/2, 0],
    [         0,         0, 1]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_axis3(pi/2)
    Matrix([
    [ 0, 1, 0],
    [-1, 0, 0],
    [ 0, 0, 1]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (clockwise around the x axis)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    """
    return rot_givens(0, 1, theta, dim=3)


def rot_axis2(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 2-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    clockwise rotation around the `y`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                \cos(\theta) & 0 & -\sin(\theta) \\
                           0 & 1 &             0 \\
                \sin(\theta) & 0 &  \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_axis2

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_axis2(theta)
    Matrix([
    [      1/2, 0, -sqrt(3)/2],
    [        0, 1,          0],
    [sqrt(3)/2, 0,        1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_axis2(pi/2)
    Matrix([
    [0, 0, -1],
    [0, 1,  0],
    [1, 0,  0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    return rot_givens(2, 0, theta, dim=3)


def rot_axis1(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 1-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    clockwise rotation around the `x`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                1 &             0 &            0 \\
                0 &  \cos(\theta) & \sin(\theta) \\
                0 & -\sin(\theta) & \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_axis1

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_axis1(theta)
    Matrix([
    [1,          0,         0],
    [0,        1/2, sqrt(3)/2],
    [0, -sqrt(3)/2,       1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_axis1(pi/2)
    Matrix([
    [1,  0, 0],
    [0,  0, 1],
    [0, -1, 0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    """
    return rot_givens(1, 2, theta, dim=3)


def rot_ccw_axis3(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 3-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `z`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                \cos(\theta) & -\sin(\theta) & 0 \\
                \sin(\theta) &  \cos(\theta) & 0 \\
                           0 &             0 & 1
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis3

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis3(theta)
    Matrix([
    [      1/2, -sqrt(3)/2, 0],
    [sqrt(3)/2,        1/2, 0],
    [        0,          0, 1]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis3(pi/2)
    Matrix([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    """
    return rot_givens(1, 0, theta, dim=3)


def rot_ccw_axis2(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 2-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `y`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                 \cos(\theta) & 0 & \sin(\theta) \\
                            0 & 1 &            0 \\
                -\sin(\theta) & 0 & \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis2

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis2(theta)
    Matrix([
    [       1/2, 0, sqrt(3)/2],
    [         0, 1,         0],
    [-sqrt(3)/2, 0,       1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis2(pi/2)
    Matrix([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    return rot_givens(0, 2, theta, dim=3)


def rot_ccw_axis1(theta):
    r"""Returns a rotation matrix for a rotation of theta (in radians)
    about the 1-axis.

    Explanation
    ===========

    For a right-handed coordinate system, this corresponds to a
    counterclockwise rotation around the `x`-axis, given by:

    .. math::

        R  = \begin{bmatrix}
                1 &            0 &             0 \\
                0 & \cos(\theta) & -\sin(\theta) \\
                0 & \sin(\theta) &  \cos(\theta)
            \end{bmatrix}

    Examples
    ========

    >>> from sympy import pi, rot_ccw_axis1

    A rotation of pi/3 (60 degrees):

    >>> theta = pi/3
    >>> rot_ccw_axis1(theta)
    Matrix([
    [1,         0,          0],
    [0,       1/2, -sqrt(3)/2],
    [0, sqrt(3)/2,        1/2]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_ccw_axis1(pi/2)
    Matrix([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0]])

    See Also
    ========

    rot_givens: Returns a Givens rotation matrix (generalized rotation for
        any number of dimensions)
    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (clockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    return rot_givens(2, 1, theta, dim=3)


@doctest_depends_on(modules=('numpy',))
def symarray(prefix, shape, **kwargs):  # pragma: no cover
    r"""Create a numpy ndarray of symbols (as an object array).

    The created symbols are named ``prefix_i1_i2_``...  You should thus provide a
    non-empty prefix if you want your symbols to be unique for different output
    arrays, as SymPy symbols with identical names are the same object.

    Parameters
    ----------

    prefix : string
      A prefix prepended to the name of every symbol.

    shape : int or tuple
      Shape of the created array.  If an int, the array is one-dimensional; for
      more than one dimension the shape must be a tuple.

    \*\*kwargs : dict
      keyword arguments passed on to Symbol

    Examples
    ========
    These doctests require numpy.

    >>> from sympy import symarray
    >>> symarray('', 3)
    [_0 _1 _2]

    If you want multiple symarrays to contain distinct symbols, you *must*
    provide unique prefixes:

    >>> a = symarray('', 3)
    >>> b = symarray('', 3)
    >>> a[0] == b[0]
    True
    >>> a = symarray('a', 3)
    >>> b = symarray('b', 3)
    >>> a[0] == b[0]
    False

    Creating symarrays with a prefix:

    >>> symarray('a', 3)
    [a_0 a_1 a_2]

    For more than one dimension, the shape must be given as a tuple:

    >>> symarray('a', (2, 3))
    [[a_0_0 a_0_1 a_0_2]
     [a_1_0 a_1_1 a_1_2]]
    >>> symarray('a', (2, 3, 2))
    [[[a_0_0_0 a_0_0_1]
      [a_0_1_0 a_0_1_1]
      [a_0_2_0 a_0_2_1]]
    <BLANKLINE>
     [[a_1_0_0 a_1_0_1]
      [a_1_1_0 a_1_1_1]
      [a_1_2_0 a_1_2_1]]]

    For setting assumptions of the underlying Symbols:

    >>> [s.is_real for s in symarray('a', 2, real=True)]
    [True, True]
    """
    from numpy import empty, ndindex
    arr = empty(shape, dtype=object)
    for index in ndindex(shape):
        arr[index] = Symbol('%s_%s' % (prefix, '_'.join(map(str, index))),
                            **kwargs)
    return arr


###############
# Functions
###############

def casoratian(seqs, n, zero=True):
    """Given linear difference operator L of order 'k' and homogeneous
       equation Ly = 0 we want to compute kernel of L, which is a set
       of 'k' sequences: a(n), b(n), ... z(n).

       Solutions of L are linearly independent iff their Casoratian,
       denoted as C(a, b, ..., z), do not vanish for n = 0.

       Casoratian is defined by k x k determinant::

                  +  a(n)     b(n)     . . . z(n)     +
                  |  a(n+1)   b(n+1)   . . . z(n+1)   |
                  |    .         .     .        .     |
                  |    .         .       .      .     |
                  |    .         .         .    .     |
                  +  a(n+k-1) b(n+k-1) . . . z(n+k-1) +

       It proves very useful in rsolve_hyper() where it is applied
       to a generating set of a recurrence to factor out linearly
       dependent solutions and return a basis:

       >>> from sympy import Symbol, casoratian, factorial
       >>> n = Symbol('n', integer=True)

       Exponential and factorial are linearly independent:

       >>> casoratian([2**n, factorial(n)], n) != 0
       True

    """

    seqs = list(map(sympify, seqs))

    if not zero:
        f = lambda i, j: seqs[j].subs(n, n + i)
    else:
        f = lambda i, j: seqs[j].subs(n, i)

    k = len(seqs)

    return Matrix(k, k, f).det()


def eye(*args, **kwargs):
    """Create square identity matrix n x n

    See Also
    ========

    diag
    zeros
    ones
    """

    return Matrix.eye(*args, **kwargs)


def diag(*values, strict=True, unpack=False, **kwargs):
    """Returns a matrix with the provided values placed on the
    diagonal. If non-square matrices are included, they will
    produce a block-diagonal matrix.

    Examples
    ========

    This version of diag is a thin wrapper to Matrix.diag that differs
    in that it treats all lists like matrices -- even when a single list
    is given. If this is not desired, either put a `*` before the list or
    set `unpack=True`.

    >>> from sympy import diag

    >>> diag([1, 2, 3], unpack=True)  # = diag(1,2,3) or diag(*[1,2,3])
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])

    >>> diag([1, 2, 3])  # a column vector
    Matrix([
    [1],
    [2],
    [3]])

    See Also
    ========
    .matrixbase.MatrixBase.eye
    .matrixbase.MatrixBase.diagonal
    .matrixbase.MatrixBase.diag
    .expressions.blockmatrix.BlockMatrix
    """
    return Matrix.diag(*values, strict=strict, unpack=unpack, **kwargs)


def GramSchmidt(vlist, orthonormal=False):
    """Apply the Gram-Schmidt process to a set of vectors.

    Parameters
    ==========

    vlist : List of Matrix
        Vectors to be orthogonalized for.

    orthonormal : Bool, optional
        If true, return an orthonormal basis.

    Returns
    =======

    vlist : List of Matrix
        Orthogonalized vectors

    Notes
    =====

    This routine is mostly duplicate from ``Matrix.orthogonalize``,
    except for some difference that this always raises error when
    linearly dependent vectors are found, and the keyword ``normalize``
    has been named as ``orthonormal`` in this function.

    See Also
    ========

    .matrixbase.MatrixBase.orthogonalize

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    """
    return MutableDenseMatrix.orthogonalize(
        *vlist, normalize=orthonormal, rankcheck=True
    )


def hessian(f, varlist, constraints=()):
    """Compute Hessian matrix for a function f wrt parameters in varlist
    which may be given as a sequence or a row/column vector. A list of
    constraints may optionally be given.

    Examples
    ========

    >>> from sympy import Function, hessian, pprint
    >>> from sympy.abc import x, y
    >>> f = Function('f')(x, y)
    >>> g1 = Function('g')(x, y)
    >>> g2 = x**2 + 3*y
    >>> pprint(hessian(f, (x, y), [g1, g2]))
    [                   d               d            ]
    [     0        0    --(g(x, y))     --(g(x, y))  ]
    [                   dx              dy           ]
    [                                                ]
    [     0        0        2*x              3       ]
    [                                                ]
    [                     2               2          ]
    [d                   d               d           ]
    [--(g(x, y))  2*x   ---(f(x, y))   -----(f(x, y))]
    [dx                   2            dy dx         ]
    [                   dx                           ]
    [                                                ]
    [                     2               2          ]
    [d                   d               d           ]
    [--(g(x, y))   3   -----(f(x, y))   ---(f(x, y)) ]
    [dy                dy dx              2          ]
    [                                   dy           ]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hessian_matrix

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.jacobian
    wronskian
    """
    # f is the expression representing a function f, return regular matrix
    if isinstance(varlist, MatrixBase):
        if 1 not in varlist.shape:
            raise ShapeError("`varlist` must be a column or row vector.")
        if varlist.cols == 1:
            varlist = varlist.T
        varlist = varlist.tolist()[0]
    if is_sequence(varlist):
        n = len(varlist)
        if not n:
            raise ShapeError("`len(varlist)` must not be zero.")
    else:
        raise ValueError("Improper variable list in hessian function")
    if not getattr(f, 'diff'):
        # check differentiability
        raise ValueError("Function `f` (%s) is not differentiable" % f)
    m = len(constraints)
    N = m + n
    out = zeros(N)
    for k, g in enumerate(constraints):
        if not getattr(g, 'diff'):
            # check differentiability
            raise ValueError("Function `f` (%s) is not differentiable" % f)
        for i in range(n):
            out[k, i + m] = g.diff(varlist[i])
    for i in range(n):
        for j in range(i, n):
            out[i + m, j + m] = f.diff(varlist[i]).diff(varlist[j])
    for i in range(N):
        for j in range(i + 1, N):
            out[j, i] = out[i, j]
    return out


def jordan_cell(eigenval, n):
    """
    Create a Jordan block:

    Examples
    ========

    >>> from sympy import jordan_cell
    >>> from sympy.abc import x
    >>> jordan_cell(x, 4)
    Matrix([
    [x, 1, 0, 0],
    [0, x, 1, 0],
    [0, 0, x, 1],
    [0, 0, 0, x]])
    """

    return Matrix.jordan_block(size=n, eigenvalue=eigenval)


def matrix_multiply_elementwise(A, B):
    """Return the Hadamard product (elementwise product) of A and B

    >>> from sympy import Matrix, matrix_multiply_elementwise
    >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
    >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
    >>> matrix_multiply_elementwise(A, B)
    Matrix([
    [  0, 10, 200],
    [300, 40,   5]])

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.__mul__
    """
    return A.multiply_elementwise(B)


def ones(*args, **kwargs):
    """Returns a matrix of ones with ``rows`` rows and ``cols`` columns;
    if ``cols`` is omitted a square matrix will be returned.

    See Also
    ========

    zeros
    eye
    diag
    """

    if 'c' in kwargs:
        kwargs['cols'] = kwargs.pop('c')

    return Matrix.ones(*args, **kwargs)


def randMatrix(r, c=None, min=0, max=99, seed=None, symmetric=False,
               percent=100, prng=None):
    """Create random matrix with dimensions ``r`` x ``c``. If ``c`` is omitted
    the matrix will be square. If ``symmetric`` is True the matrix must be
    square. If ``percent`` is less than 100 then only approximately the given
    percentage of elements will be non-zero.

    The pseudo-random number generator used to generate matrix is chosen in the
    following way.

    * If ``prng`` is supplied, it will be used as random number generator.
      It should be an instance of ``random.Random``, or at least have
      ``randint`` and ``shuffle`` methods with same signatures.
    * if ``prng`` is not supplied but ``seed`` is supplied, then new
      ``random.Random`` with given ``seed`` will be created;
    * otherwise, a new ``random.Random`` with default seed will be used.

    Examples
    ========

    >>> from sympy import randMatrix
    >>> randMatrix(3) # doctest:+SKIP
    [25, 45, 27]
    [44, 54,  9]
    [23, 96, 46]
    >>> randMatrix(3, 2) # doctest:+SKIP
    [87, 29]
    [23, 37]
    [90, 26]
    >>> randMatrix(3, 3, 0, 2) # doctest:+SKIP
    [0, 2, 0]
    [2, 0, 1]
    [0, 0, 1]
    >>> randMatrix(3, symmetric=True) # doctest:+SKIP
    [85, 26, 29]
    [26, 71, 43]
    [29, 43, 57]
    >>> A = randMatrix(3, seed=1)
    >>> B = randMatrix(3, seed=2)
    >>> A == B
    False
    >>> A == randMatrix(3, seed=1)
    True
    >>> randMatrix(3, symmetric=True, percent=50) # doctest:+SKIP
    [77, 70,  0],
    [70,  0,  0],
    [ 0,  0, 88]
    """
    # Note that ``Random()`` is equivalent to ``Random(None)``
    prng = prng or random.Random(seed)

    if c is None:
        c = r

    if symmetric and r != c:
        raise ValueError('For symmetric matrices, r must equal c, but %i != %i' % (r, c))

    ij = range(r * c)
    if percent != 100:
        ij = prng.sample(ij, int(len(ij)*percent // 100))

    m = zeros(r, c)

    if not symmetric:
        for ijk in ij:
            i, j = divmod(ijk, c)
            m[i, j] = prng.randint(min, max)
    else:
        for ijk in ij:
            i, j = divmod(ijk, c)
            if i <= j:
                m[i, j] = m[j, i] = prng.randint(min, max)

    return m


def wronskian(functions, var, method='bareiss'):
    """
    Compute Wronskian for [] of functions

    ::

                         | f1       f2        ...   fn      |
                         | f1'      f2'       ...   fn'     |
                         |  .        .        .      .      |
        W(f1, ..., fn) = |  .        .         .     .      |
                         |  .        .          .    .      |
                         |  (n)      (n)            (n)     |
                         | D   (f1) D   (f2)  ...  D   (fn) |

    see: https://en.wikipedia.org/wiki/Wronskian

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.jacobian
    hessian
    """

    functions = [sympify(f) for f in functions]
    n = len(functions)
    if n == 0:
        return S.One
    W = Matrix(n, n, lambda i, j: functions[i].diff(var, j))
    return W.det(method)


def zeros(*args, **kwargs):
    """Returns a matrix of zeros with ``rows`` rows and ``cols`` columns;
    if ``cols`` is omitted a square matrix will be returned.

    See Also
    ========

    ones
    eye
    diag
    """

    if 'c' in kwargs:
        kwargs['cols'] = kwargs.pop('c')

    return Matrix.zeros(*args, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\records.py ===
"""
This module contains a set of functions for record arrays.
"""
import os
import warnings
from collections import Counter
from contextlib import nullcontext

from numpy._utils import set_module

from . import numeric as sb
from . import numerictypes as nt
from .arrayprint import _get_legacy_print_mode

# All of the functions allow formats to be a dtype
__all__ = [
    'record', 'recarray', 'format_parser', 'fromarrays', 'fromrecords',
    'fromstring', 'fromfile', 'array', 'find_duplicate',
]


ndarray = sb.ndarray

_byteorderconv = {'b': '>',
                  'l': '<',
                  'n': '=',
                  'B': '>',
                  'L': '<',
                  'N': '=',
                  'S': 's',
                  's': 's',
                  '>': '>',
                  '<': '<',
                  '=': '=',
                  '|': '|',
                  'I': '|',
                  'i': '|'}

# formats regular expression
# allows multidimensional spec with a tuple syntax in front
# of the letter code '(2,3)f4' and ' (  2 ,  3  )  f4  '
# are equally allowed

numfmt = nt.sctypeDict


@set_module('numpy.rec')
def find_duplicate(list):
    """Find duplication in a list, return a list of duplicated elements"""
    return [
        item
        for item, counts in Counter(list).items()
        if counts > 1
    ]


@set_module('numpy.rec')
class format_parser:
    """
    Class to convert formats, names, titles description to a dtype.

    After constructing the format_parser object, the dtype attribute is
    the converted data-type:
    ``dtype = format_parser(formats, names, titles).dtype``

    Attributes
    ----------
    dtype : dtype
        The converted data-type.

    Parameters
    ----------
    formats : str or list of str
        The format description, either specified as a string with
        comma-separated format descriptions in the form ``'f8, i4, S5'``, or
        a list of format description strings  in the form
        ``['f8', 'i4', 'S5']``.
    names : str or list/tuple of str
        The field names, either specified as a comma-separated string in the
        form ``'col1, col2, col3'``, or as a list or tuple of strings in the
        form ``['col1', 'col2', 'col3']``.
        An empty list can be used, in that case default field names
        ('f0', 'f1', ...) are used.
    titles : sequence
        Sequence of title strings. An empty list can be used to leave titles
        out.
    aligned : bool, optional
        If True, align the fields by padding as the C-compiler would.
        Default is False.
    byteorder : str, optional
        If specified, all the fields will be changed to the
        provided byte-order.  Otherwise, the default byte-order is
        used. For all available string specifiers, see `dtype.newbyteorder`.

    See Also
    --------
    numpy.dtype, numpy.typename

    Examples
    --------
    >>> import numpy as np
    >>> np.rec.format_parser(['<f8', '<i4'], ['col1', 'col2'],
    ...                      ['T1', 'T2']).dtype
    dtype([(('T1', 'col1'), '<f8'), (('T2', 'col2'), '<i4')])

    `names` and/or `titles` can be empty lists. If `titles` is an empty list,
    titles will simply not appear. If `names` is empty, default field names
    will be used.

    >>> np.rec.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],
    ...                      []).dtype
    dtype([('col1', '<f8'), ('col2', '<i4'), ('col3', '<S5')])
    >>> np.rec.format_parser(['<f8', '<i4', '<a5'], [], []).dtype
    dtype([('f0', '<f8'), ('f1', '<i4'), ('f2', 'S5')])

    """

    def __init__(self, formats, names, titles, aligned=False, byteorder=None):
        self._parseFormats(formats, aligned)
        self._setfieldnames(names, titles)
        self._createdtype(byteorder)

    def _parseFormats(self, formats, aligned=False):
        """ Parse the field formats """

        if formats is None:
            raise ValueError("Need formats argument")
        if isinstance(formats, list):
            dtype = sb.dtype(
                [
                    (f'f{i}', format_)
                    for i, format_ in enumerate(formats)
                ],
                aligned,
            )
        else:
            dtype = sb.dtype(formats, aligned)
        fields = dtype.fields
        if fields is None:
            dtype = sb.dtype([('f1', dtype)], aligned)
            fields = dtype.fields
        keys = dtype.names
        self._f_formats = [fields[key][0] for key in keys]
        self._offsets = [fields[key][1] for key in keys]
        self._nfields = len(keys)

    def _setfieldnames(self, names, titles):
        """convert input field names into a list and assign to the _names
        attribute """

        if names:
            if type(names) in [list, tuple]:
                pass
            elif isinstance(names, str):
                names = names.split(',')
            else:
                raise NameError(f"illegal input names {repr(names)}")

            self._names = [n.strip() for n in names[:self._nfields]]
        else:
            self._names = []

        # if the names are not specified, they will be assigned as
        #  "f0, f1, f2,..."
        # if not enough names are specified, they will be assigned as "f[n],
        # f[n+1],..." etc. where n is the number of specified names..."
        self._names += ['f%d' % i for i in range(len(self._names),
                                                 self._nfields)]
        # check for redundant names
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError(f"Duplicate field names: {_dup}")

        if titles:
            self._titles = [n.strip() for n in titles[:self._nfields]]
        else:
            self._titles = []
            titles = []

        if self._nfields > len(titles):
            self._titles += [None] * (self._nfields - len(titles))

    def _createdtype(self, byteorder):
        dtype = sb.dtype({
            'names': self._names,
            'formats': self._f_formats,
            'offsets': self._offsets,
            'titles': self._titles,
        })
        if byteorder is not None:
            byteorder = _byteorderconv[byteorder[0]]
            dtype = dtype.newbyteorder(byteorder)

        self.dtype = dtype


class record(nt.void):
    """A data-type scalar that allows field access as attribute lookup.
    """

    # manually set name and module so that this class's type shows up
    # as numpy.record when printed
    __name__ = 'record'
    __module__ = 'numpy'

    def __repr__(self):
        if _get_legacy_print_mode() <= 113:
            return self.__str__()
        return super().__repr__()

    def __str__(self):
        if _get_legacy_print_mode() <= 113:
            return str(self.item())
        return super().__str__()

    def __getattribute__(self, attr):
        if attr in ('setfield', 'getfield', 'dtype'):
            return nt.void.__getattribute__(self, attr)
        try:
            return nt.void.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            obj = self.getfield(*res[:2])
            # if it has fields return a record,
            # otherwise return the object
            try:
                dt = obj.dtype
            except AttributeError:
                # happens if field is Object type
                return obj
            if dt.names is not None:
                return obj.view((self.__class__, obj.dtype))
            return obj
        else:
            raise AttributeError(f"'record' object has no attribute '{attr}'")

    def __setattr__(self, attr, val):
        if attr in ('setfield', 'getfield', 'dtype'):
            raise AttributeError(f"Cannot set '{attr}' attribute")
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            return self.setfield(val, *res[:2])
        elif getattr(self, attr, None):
            return nt.void.__setattr__(self, attr, val)
        else:
            raise AttributeError(f"'record' object has no attribute '{attr}'")

    def __getitem__(self, indx):
        obj = nt.void.__getitem__(self, indx)

        # copy behavior of record.__getattribute__,
        if isinstance(obj, nt.void) and obj.dtype.names is not None:
            return obj.view((self.__class__, obj.dtype))
        else:
            # return a single element
            return obj

    def pprint(self):
        """Pretty-print all fields."""
        # pretty-print all fields
        names = self.dtype.names
        maxlen = max(len(name) for name in names)
        fmt = '%% %ds: %%s' % maxlen
        rows = [fmt % (name, getattr(self, name)) for name in names]
        return "\n".join(rows)

# The recarray is almost identical to a standard array (which supports
#   named fields already)  The biggest difference is that it can use
#   attribute-lookup to find the fields and it is constructed using
#   a record.

# If byteorder is given it forces a particular byteorder on all
#  the fields (and any subfields)


@set_module("numpy.rec")
class recarray(ndarray):
    """Construct an ndarray that allows field access using attributes.

    Arrays may have a data-types containing fields, analogous
    to columns in a spread sheet.  An example is ``[(x, int), (y, float)]``,
    where each entry in the array is a pair of ``(int, float)``.  Normally,
    these attributes are accessed using dictionary lookups such as ``arr['x']``
    and ``arr['y']``.  Record arrays allow the fields to be accessed as members
    of the array, using ``arr.x`` and ``arr.y``.

    Parameters
    ----------
    shape : tuple
        Shape of output array.
    dtype : data-type, optional
        The desired data-type.  By default, the data-type is determined
        from `formats`, `names`, `titles`, `aligned` and `byteorder`.
    formats : list of data-types, optional
        A list containing the data-types for the different columns, e.g.
        ``['i4', 'f8', 'i4']``.  `formats` does *not* support the new
        convention of using types directly, i.e. ``(int, float, int)``.
        Note that `formats` must be a list, not a tuple.
        Given that `formats` is somewhat limited, we recommend specifying
        `dtype` instead.
    names : tuple of str, optional
        The name of each column, e.g. ``('x', 'y', 'z')``.
    buf : buffer, optional
        By default, a new array is created of the given shape and data-type.
        If `buf` is specified and is an object exposing the buffer interface,
        the array will use the memory from the existing buffer.  In this case,
        the `offset` and `strides` keywords are available.

    Other Parameters
    ----------------
    titles : tuple of str, optional
        Aliases for column names.  For example, if `names` were
        ``('x', 'y', 'z')`` and `titles` is
        ``('x_coordinate', 'y_coordinate', 'z_coordinate')``, then
        ``arr['x']`` is equivalent to both ``arr.x`` and ``arr.x_coordinate``.
    byteorder : {'<', '>', '='}, optional
        Byte-order for all fields.
    aligned : bool, optional
        Align the fields in memory as the C-compiler would.
    strides : tuple of ints, optional
        Buffer (`buf`) is interpreted according to these strides (strides
        define how many bytes each array element, row, column, etc.
        occupy in memory).
    offset : int, optional
        Start reading buffer (`buf`) from this offset onwards.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    rec : recarray
        Empty array of the given shape and type.

    See Also
    --------
    numpy.rec.fromrecords : Construct a record array from data.
    numpy.record : fundamental data-type for `recarray`.
    numpy.rec.format_parser : determine data-type from formats, names, titles.

    Notes
    -----
    This constructor can be compared to ``empty``: it creates a new record
    array but does not fill it with data.  To create a record array from data,
    use one of the following methods:

    1. Create a standard ndarray and convert it to a record array,
       using ``arr.view(np.recarray)``
    2. Use the `buf` keyword.
    3. Use `np.rec.fromrecords`.

    Examples
    --------
    Create an array with two fields, ``x`` and ``y``:

    >>> import numpy as np
    >>> x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', '<f8'), ('y', '<i8')])
    >>> x
    array([(1., 2), (3., 4)], dtype=[('x', '<f8'), ('y', '<i8')])

    >>> x['x']
    array([1., 3.])

    View the array as a record array:

    >>> x = x.view(np.recarray)

    >>> x.x
    array([1., 3.])

    >>> x.y
    array([2, 4])

    Create a new, empty record array:

    >>> np.recarray((2,),
    ... dtype=[('x', int), ('y', float), ('z', int)]) #doctest: +SKIP
    rec.array([(-1073741821, 1.2249118382103472e-301, 24547520),
           (3471280, 1.2134086255804012e-316, 0)],
          dtype=[('x', '<i4'), ('y', '<f8'), ('z', '<i4')])

    """

    def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False, order='C'):

        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(
                formats, names, titles, aligned, byteorder
            ).dtype

        if buf is None:
            self = ndarray.__new__(
                subtype, shape, (record, descr), order=order
            )
        else:
            self = ndarray.__new__(
                subtype, shape, (record, descr), buffer=buf,
                offset=offset, strides=strides, order=order
            )
        return self

    def __array_finalize__(self, obj):
        if self.dtype.type is not record and self.dtype.names is not None:
            # if self.dtype is not np.record, invoke __setattr__ which will
            # convert it to a record if it is a void dtype.
            self.dtype = self.dtype

    def __getattribute__(self, attr):
        # See if ndarray has this attr, and return it if so. (note that this
        # means a field with the same name as an ndarray attr cannot be
        # accessed by attribute).
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass

        # look for a field with this name
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(f"recarray has no attribute {attr}") from e
        obj = self.getfield(*res)

        # At this point obj will always be a recarray, since (see
        # PyArray_GetField) the type of obj is inherited. Next, if obj.dtype is
        # non-structured, convert it to an ndarray. Then if obj is structured
        # with void type convert it to the same dtype.type (eg to preserve
        # numpy.record type if present), since nested structured fields do not
        # inherit type. Don't do this for non-void structures though.
        if obj.dtype.names is not None:
            if issubclass(obj.dtype.type, nt.void):
                return obj.view(dtype=(self.dtype.type, obj.dtype))
            return obj
        else:
            return obj.view(ndarray)

    # Save the dictionary.
    # If the attr is a field name and not in the saved dictionary
    # Undo any "setting" of the attribute and do a setfield
    # Thus, you can't create attributes on-the-fly that are field names.
    def __setattr__(self, attr, val):

        # Automatically convert (void) structured types to records
        # (but not non-void structures, subarrays, or non-structured voids)
        if (
            attr == 'dtype' and
            issubclass(val.type, nt.void) and
            val.names is not None
        ):
            val = sb.dtype((record, val))

        newattr = attr not in self.__dict__
        try:
            ret = object.__setattr__(self, attr, val)
        except Exception:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                raise
        else:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                return ret
            if newattr:
                # We just added this one or this setattr worked on an
                # internal attribute.
                try:
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(
                f"record array has no attribute {attr}"
            ) from e
        return self.setfield(val, *res)

    def __getitem__(self, indx):
        obj = super().__getitem__(indx)

        # copy behavior of getattr, except that here
        # we might also be returning a single element
        if isinstance(obj, ndarray):
            if obj.dtype.names is not None:
                obj = obj.view(type(self))
                if issubclass(obj.dtype.type, nt.void):
                    return obj.view(dtype=(self.dtype.type, obj.dtype))
                return obj
            else:
                return obj.view(type=ndarray)
        else:
            # return a single element
            return obj

    def __repr__(self):

        repr_dtype = self.dtype
        if (
            self.dtype.type is record or
            not issubclass(self.dtype.type, nt.void)
        ):
            # If this is a full record array (has numpy.record dtype),
            # or if it has a scalar (non-void) dtype with no records,
            # represent it using the rec.array function. Since rec.array
            # converts dtype to a numpy.record for us, convert back
            # to non-record before printing
            if repr_dtype.type is record:
                repr_dtype = sb.dtype((nt.void, repr_dtype))
            prefix = "rec.array("
            fmt = 'rec.array(%s,%sdtype=%s)'
        else:
            # otherwise represent it using np.array plus a view
            # This should only happen if the user is playing
            # strange games with dtypes.
            prefix = "array("
            fmt = 'array(%s,%sdtype=%s).view(numpy.recarray)'

        # get data/shape string. logic taken from numeric.array_repr
        if self.size > 0 or self.shape == (0,):
            lst = sb.array2string(
                self, separator=', ', prefix=prefix, suffix=',')
        else:
            # show zero-length shape unless it is (0,)
            lst = f"[], shape={repr(self.shape)}"

        lf = '\n' + ' ' * len(prefix)
        if _get_legacy_print_mode() <= 113:
            lf = ' ' + lf  # trailing space
        return fmt % (lst, lf, repr_dtype)

    def field(self, attr, val=None):
        if isinstance(attr, int):
            names = ndarray.__getattribute__(self, 'dtype').names
            attr = names[attr]

        fielddict = ndarray.__getattribute__(self, 'dtype').fields

        res = fielddict[attr][:2]

        if val is None:
            obj = self.getfield(*res)
            if obj.dtype.names is not None:
                return obj
            return obj.view(ndarray)
        else:
            return self.setfield(val, *res)


def _deprecate_shape_0_as_None(shape):
    if shape == 0:
        warnings.warn(
            "Passing `shape=0` to have the shape be inferred is deprecated, "
            "and in future will be equivalent to `shape=(0,)`. To infer "
            "the shape and suppress this warning, pass `shape=None` instead.",
            FutureWarning, stacklevel=3)
        return None
    else:
        return shape


@set_module("numpy.rec")
def fromarrays(arrayList, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    """Create a record array from a (flat) list of arrays

    Parameters
    ----------
    arrayList : list or tuple
        List of array-like objects (such as lists, tuples,
        and ndarrays).
    dtype : data-type, optional
        valid dtype for all arrays
    shape : int or tuple of ints, optional
        Shape of the resulting array. If not provided, inferred from
        ``arrayList[0]``.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.rec.format_parser` to construct a dtype. See that function for
        detailed documentation.

    Returns
    -------
    np.recarray
        Record array consisting of given arrayList columns.

    Examples
    --------
    >>> x1=np.array([1,2,3,4])
    >>> x2=np.array(['a','dd','xyz','12'])
    >>> x3=np.array([1.1,2,3,4])
    >>> r = np.rec.fromarrays([x1,x2,x3],names='a,b,c')
    >>> print(r[1])
    (2, 'dd', 2.0) # may vary
    >>> x1[1]=34
    >>> r.a
    array([1, 2, 3, 4])

    >>> x1 = np.array([1, 2, 3, 4])
    >>> x2 = np.array(['a', 'dd', 'xyz', '12'])
    >>> x3 = np.array([1.1, 2, 3,4])
    >>> r = np.rec.fromarrays(
    ...     [x1, x2, x3],
    ...     dtype=np.dtype([('a', np.int32), ('b', 'S3'), ('c', np.float32)]))
    >>> r
    rec.array([(1, b'a', 1.1), (2, b'dd', 2. ), (3, b'xyz', 3. ),
               (4, b'12', 4. )],
              dtype=[('a', '<i4'), ('b', 'S3'), ('c', '<f4')])
    """

    arrayList = [sb.asarray(x) for x in arrayList]

    # NumPy 1.19.0, 2020-01-01
    shape = _deprecate_shape_0_as_None(shape)

    if shape is None:
        shape = arrayList[0].shape
    elif isinstance(shape, int):
        shape = (shape,)

    if formats is None and dtype is None:
        # go through each object in the list to see if it is an ndarray
        # and determine the formats.
        formats = [obj.dtype for obj in arrayList]

    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    _names = descr.names

    # Determine shape from data-type.
    if len(descr) != len(arrayList):
        raise ValueError("mismatch between the number of fields "
                "and the number of arrays")

    d0 = descr[0].shape
    nn = len(d0)
    if nn > 0:
        shape = shape[:-nn]

    _array = recarray(shape, descr)

    # populate the record array (makes a copy)
    for k, obj in enumerate(arrayList):
        nn = descr[k].ndim
        testshape = obj.shape[:obj.ndim - nn]
        name = _names[k]
        if testshape != shape:
            raise ValueError(f'array-shape mismatch in array {k} ("{name}")')

        _array[name] = obj

    return _array


@set_module("numpy.rec")
def fromrecords(recList, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None):
    """Create a recarray from a list of records in text form.

    Parameters
    ----------
    recList : sequence
        data in the same field may be heterogeneous - they will be promoted
        to the highest data type.
    dtype : data-type, optional
        valid dtype for all arrays
    shape : int or tuple of ints, optional
        shape of each array.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation.

        If both `formats` and `dtype` are None, then this will auto-detect
        formats. Use list of tuples rather than list of lists for faster
        processing.

    Returns
    -------
    np.recarray
        record array consisting of given recList rows.

    Examples
    --------
    >>> r=np.rec.fromrecords([(456,'dbe',1.2),(2,'de',1.3)],
    ... names='col1,col2,col3')
    >>> print(r[0])
    (456, 'dbe', 1.2)
    >>> r.col1
    array([456,   2])
    >>> r.col2
    array(['dbe', 'de'], dtype='<U3')
    >>> import pickle
    >>> pickle.loads(pickle.dumps(r))
    rec.array([(456, 'dbe', 1.2), (  2, 'de', 1.3)],
              dtype=[('col1', '<i8'), ('col2', '<U3'), ('col3', '<f8')])
    """

    if formats is None and dtype is None:  # slower
        obj = sb.array(recList, dtype=object)
        arrlist = [
            sb.array(obj[..., i].tolist()) for i in range(obj.shape[-1])
        ]
        return fromarrays(arrlist, formats=formats, shape=shape, names=names,
                          titles=titles, aligned=aligned, byteorder=byteorder)

    if dtype is not None:
        descr = sb.dtype((record, dtype))
    else:
        descr = format_parser(
            formats, names, titles, aligned, byteorder
        ).dtype

    try:
        retval = sb.array(recList, dtype=descr)
    except (TypeError, ValueError):
        # NumPy 1.19.0, 2020-01-01
        shape = _deprecate_shape_0_as_None(shape)
        if shape is None:
            shape = len(recList)
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) > 1:
            raise ValueError("Can only deal with 1-d array.")
        _array = recarray(shape, descr)
        for k in range(_array.size):
            _array[k] = tuple(recList[k])
        # list of lists instead of list of tuples ?
        # 2018-02-07, 1.14.1
        warnings.warn(
            "fromrecords expected a list of tuples, may have received a list "
            "of lists instead. In the future that will raise an error",
            FutureWarning, stacklevel=2)
        return _array
    else:
        if shape is not None and retval.shape != shape:
            retval.shape = shape

    res = retval.view(recarray)

    return res


@set_module("numpy.rec")
def fromstring(datastring, dtype=None, shape=None, offset=0, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    r"""Create a record array from binary data

    Note that despite the name of this function it does not accept `str`
    instances.

    Parameters
    ----------
    datastring : bytes-like
        Buffer of binary data
    dtype : data-type, optional
        Valid dtype for all arrays
    shape : int or tuple of ints, optional
        Shape of each array.
    offset : int, optional
        Position in the buffer to start reading from.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation.


    Returns
    -------
    np.recarray
        Record array view into the data in datastring. This will be readonly
        if `datastring` is readonly.

    See Also
    --------
    numpy.frombuffer

    Examples
    --------
    >>> a = b'\x01\x02\x03abc'
    >>> np.rec.fromstring(a, dtype='u1,u1,u1,S3')
    rec.array([(1, 2, 3, b'abc')],
            dtype=[('f0', 'u1'), ('f1', 'u1'), ('f2', 'u1'), ('f3', 'S3')])

    >>> grades_dtype = [('Name', (np.str_, 10)), ('Marks', np.float64),
    ...                 ('GradeLevel', np.int32)]
    >>> grades_array = np.array([('Sam', 33.3, 3), ('Mike', 44.4, 5),
    ...                         ('Aadi', 66.6, 6)], dtype=grades_dtype)
    >>> np.rec.fromstring(grades_array.tobytes(), dtype=grades_dtype)
    rec.array([('Sam', 33.3, 3), ('Mike', 44.4, 5), ('Aadi', 66.6, 6)],
            dtype=[('Name', '<U10'), ('Marks', '<f8'), ('GradeLevel', '<i4')])

    >>> s = '\x01\x02\x03abc'
    >>> np.rec.fromstring(s, dtype='u1,u1,u1,S3')
    Traceback (most recent call last):
       ...
    TypeError: a bytes-like object is required, not 'str'
    """

    if dtype is None and formats is None:
        raise TypeError("fromstring() needs a 'dtype' or 'formats' argument")

    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype

    itemsize = descr.itemsize

    # NumPy 1.19.0, 2020-01-01
    shape = _deprecate_shape_0_as_None(shape)

    if shape in (None, -1):
        shape = (len(datastring) - offset) // itemsize

    _array = recarray(shape, descr, buf=datastring, offset=offset)
    return _array

def get_remaining_size(fd):
    pos = fd.tell()
    try:
        fd.seek(0, 2)
        return fd.tell() - pos
    finally:
        fd.seek(pos, 0)


@set_module("numpy.rec")
def fromfile(fd, dtype=None, shape=None, offset=0, formats=None,
             names=None, titles=None, aligned=False, byteorder=None):
    """Create an array from binary file data

    Parameters
    ----------
    fd : str or file type
        If file is a string or a path-like object then that file is opened,
        else it is assumed to be a file object. The file object must
        support random access (i.e. it must have tell and seek methods).
    dtype : data-type, optional
        valid dtype for all arrays
    shape : int or tuple of ints, optional
        shape of each array.
    offset : int, optional
        Position in the file to start reading from.
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation

    Returns
    -------
    np.recarray
        record array consisting of data enclosed in file.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> a = np.empty(10,dtype='f8,i4,a5')
    >>> a[5] = (0.5,10,'abcde')
    >>>
    >>> fd=TemporaryFile()
    >>> a = a.view(a.dtype.newbyteorder('<'))
    >>> a.tofile(fd)
    >>>
    >>> _ = fd.seek(0)
    >>> r=np.rec.fromfile(fd, formats='f8,i4,a5', shape=10,
    ... byteorder='<')
    >>> print(r[5])
    (0.5, 10, b'abcde')
    >>> r.shape
    (10,)
    """

    if dtype is None and formats is None:
        raise TypeError("fromfile() needs a 'dtype' or 'formats' argument")

    # NumPy 1.19.0, 2020-01-01
    shape = _deprecate_shape_0_as_None(shape)

    if shape is None:
        shape = (-1,)
    elif isinstance(shape, int):
        shape = (shape,)

    if hasattr(fd, 'readinto'):
        # GH issue 2504. fd supports io.RawIOBase or io.BufferedIOBase
        # interface. Example of fd: gzip, BytesIO, BufferedReader
        # file already opened
        ctx = nullcontext(fd)
    else:
        # open file
        ctx = open(os.fspath(fd), 'rb')

    with ctx as fd:
        if offset > 0:
            fd.seek(offset, 1)
        size = get_remaining_size(fd)

        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(
                formats, names, titles, aligned, byteorder
            ).dtype

        itemsize = descr.itemsize

        shapeprod = sb.array(shape).prod(dtype=nt.intp)
        shapesize = shapeprod * itemsize
        if shapesize < 0:
            shape = list(shape)
            shape[shape.index(-1)] = size // -shapesize
            shape = tuple(shape)
            shapeprod = sb.array(shape).prod(dtype=nt.intp)

        nbytes = shapeprod * itemsize

        if nbytes > size:
            raise ValueError(
                    "Not enough bytes left in file for specified "
                    "shape and type."
                )

        # create the array
        _array = recarray(shape, descr)
        nbytesread = fd.readinto(_array.data)
        if nbytesread != nbytes:
            raise OSError("Didn't read as many bytes as expected")

    return _array


@set_module("numpy.rec")
def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
          names=None, titles=None, aligned=False, byteorder=None, copy=True):
    """
    Construct a record array from a wide-variety of objects.

    A general-purpose record array constructor that dispatches to the
    appropriate `recarray` creation function based on the inputs (see Notes).

    Parameters
    ----------
    obj : any
        Input object. See Notes for details on how various input types are
        treated.
    dtype : data-type, optional
        Valid dtype for array.
    shape : int or tuple of ints, optional
        Shape of each array.
    offset : int, optional
        Position in the file or buffer to start reading from.
    strides : tuple of ints, optional
        Buffer (`buf`) is interpreted according to these strides (strides
        define how many bytes each array element, row, column, etc.
        occupy in memory).
    formats, names, titles, aligned, byteorder :
        If `dtype` is ``None``, these arguments are passed to
        `numpy.format_parser` to construct a dtype. See that function for
        detailed documentation.
    copy : bool, optional
        Whether to copy the input object (True), or to use a reference instead.
        This option only applies when the input is an ndarray or recarray.
        Defaults to True.

    Returns
    -------
    np.recarray
        Record array created from the specified object.

    Notes
    -----
    If `obj` is ``None``, then call the `~numpy.recarray` constructor. If
    `obj` is a string, then call the `fromstring` constructor. If `obj` is a
    list or a tuple, then if the first object is an `~numpy.ndarray`, call
    `fromarrays`, otherwise call `fromrecords`. If `obj` is a
    `~numpy.recarray`, then make a copy of the data in the recarray
    (if ``copy=True``) and use the new formats, names, and titles. If `obj`
    is a file, then call `fromfile`. Finally, if obj is an `ndarray`, then
    return ``obj.view(recarray)``, making a copy of the data if ``copy=True``.

    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> np.rec.array(a)
    rec.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]],
              dtype=int64)

    >>> b = [(1, 1), (2, 4), (3, 9)]
    >>> c = np.rec.array(b, formats = ['i2', 'f2'], names = ('x', 'y'))
    >>> c
    rec.array([(1, 1.), (2, 4.), (3, 9.)],
              dtype=[('x', '<i2'), ('y', '<f2')])

    >>> c.x
    array([1, 2, 3], dtype=int16)

    >>> c.y
    array([1.,  4.,  9.], dtype=float16)

    >>> r = np.rec.array(['abc','def'], names=['col1','col2'])
    >>> print(r.col1)
    abc

    >>> r.col1
    array('abc', dtype='<U3')

    >>> r.col2
    array('def', dtype='<U3')
    """

    if ((isinstance(obj, (type(None), str)) or hasattr(obj, 'readinto')) and
           formats is None and dtype is None):
        raise ValueError("Must define formats (or dtype) if object is "
                         "None, string, or an open file")

    kwds = {}
    if dtype is not None:
        dtype = sb.dtype(dtype)
    elif formats is not None:
        dtype = format_parser(formats, names, titles,
                              aligned, byteorder).dtype
    else:
        kwds = {'formats': formats,
                'names': names,
                'titles': titles,
                'aligned': aligned,
                'byteorder': byteorder
                }

    if obj is None:
        if shape is None:
            raise ValueError("Must define a shape if obj is None")
        return recarray(shape, dtype, buf=obj, offset=offset, strides=strides)

    elif isinstance(obj, bytes):
        return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)

    elif isinstance(obj, (list, tuple)):
        if isinstance(obj[0], (tuple, list)):
            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
        else:
            return fromarrays(obj, dtype=dtype, shape=shape, **kwds)

    elif isinstance(obj, recarray):
        if dtype is not None and (obj.dtype != dtype):
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new

    elif hasattr(obj, 'readinto'):
        return fromfile(obj, dtype=dtype, shape=shape, offset=offset)

    elif isinstance(obj, ndarray):
        if dtype is not None and (obj.dtype != dtype):
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new.view(recarray)

    else:
        interface = getattr(obj, "__array_interface__", None)
        if interface is None or not isinstance(interface, dict):
            raise ValueError("Unknown input type")
        obj = sb.array(obj)
        if dtype is not None and (obj.dtype != dtype):
            obj = obj.view(dtype)
        return obj.view(recarray)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\utils.py ===
"""
requests.utils
~~~~~~~~~~~~~~

This module provides utility functions that are used within Requests
that are also useful for external consumption.
"""

import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict

from urllib3.util import make_headers, parse_url

from . import certs
from .__version__ import __version__

# to_native_string is unused here, but imported here for backwards compatibility
from ._internal_utils import (  # noqa: F401
    _HEADER_VALIDATORS_BYTE,
    _HEADER_VALIDATORS_STR,
    HEADER_VALIDATORS,
    to_native_string,
)
from .compat import (
    Mapping,
    basestring,
    bytes,
    getproxies,
    getproxies_environment,
    integer_types,
    is_urllib3_1,
)
from .compat import parse_http_list as _parse_list_header
from .compat import (
    proxy_bypass,
    proxy_bypass_environment,
    quote,
    str,
    unquote,
    urlparse,
    urlunparse,
)
from .cookies import cookiejar_from_dict
from .exceptions import (
    FileModeWarning,
    InvalidHeader,
    InvalidURL,
    UnrewindableBodyError,
)
from .structures import CaseInsensitiveDict

NETRC_FILES = (".netrc", "_netrc")

DEFAULT_CA_BUNDLE_PATH = certs.where()

DEFAULT_PORTS = {"http": 80, "https": 443}

# Ensure that ', ' is used to preserve previous delimiter behavior.
DEFAULT_ACCEPT_ENCODING = ", ".join(
    re.split(r",\s*", make_headers(accept_encoding=True)["accept-encoding"])
)


if sys.platform == "win32":
    # provide a proxy_bypass version on Windows without DNS lookups

    def proxy_bypass_registry(host):
        try:
            import winreg
        except ImportError:
            return False

        try:
            internetSettings = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            )
            # ProxyEnable could be REG_SZ or REG_DWORD, normalizing it
            proxyEnable = int(winreg.QueryValueEx(internetSettings, "ProxyEnable")[0])
            # ProxyOverride is almost always a string
            proxyOverride = winreg.QueryValueEx(internetSettings, "ProxyOverride")[0]
        except (OSError, ValueError):
            return False
        if not proxyEnable or not proxyOverride:
            return False

        # make a check value list from the registry entry: replace the
        # '<local>' string by the localhost entry and the corresponding
        # canonical entry.
        proxyOverride = proxyOverride.split(";")
        # filter out empty strings to avoid re.match return true in the following code.
        proxyOverride = filter(None, proxyOverride)
        # now check if we match one of the registry values.
        for test in proxyOverride:
            if test == "<local>":
                if "." not in host:
                    return True
            test = test.replace(".", r"\.")  # mask dots
            test = test.replace("*", r".*")  # change glob sequence
            test = test.replace("?", r".")  # change glob char
            if re.match(test, host, re.I):
                return True
        return False

    def proxy_bypass(host):  # noqa
        """Return True, if the host should be bypassed.

        Checks proxy settings gathered from the environment, if specified,
        or the registry.
        """
        if getproxies_environment():
            return proxy_bypass_environment(host)
        else:
            return proxy_bypass_registry(host)


def dict_to_sequence(d):
    """Returns an internal sequence dictionary update."""

    if hasattr(d, "items"):
        d = d.items()

    return d


def super_len(o):
    total_length = None
    current_position = 0

    if not is_urllib3_1 and isinstance(o, str):
        # urllib3 2.x+ treats all strings as utf-8 instead
        # of latin-1 (iso-8859-1) like http.client.
        o = o.encode("utf-8")

    if hasattr(o, "__len__"):
        total_length = len(o)

    elif hasattr(o, "len"):
        total_length = o.len

    elif hasattr(o, "fileno"):
        try:
            fileno = o.fileno()
        except (io.UnsupportedOperation, AttributeError):
            # AttributeError is a surprising exception, seeing as how we've just checked
            # that `hasattr(o, 'fileno')`.  It happens for objects obtained via
            # `Tarfile.extractfile()`, per issue 5229.
            pass
        else:
            total_length = os.fstat(fileno).st_size

            # Having used fstat to determine the file length, we need to
            # confirm that this file was opened up in binary mode.
            if "b" not in o.mode:
                warnings.warn(
                    (
                        "Requests has determined the content-length for this "
                        "request using the binary size of the file: however, the "
                        "file has been opened in text mode (i.e. without the 'b' "
                        "flag in the mode). This may lead to an incorrect "
                        "content-length. In Requests 3.0, support will be removed "
                        "for files in text mode."
                    ),
                    FileModeWarning,
                )

    if hasattr(o, "tell"):
        try:
            current_position = o.tell()
        except OSError:
            # This can happen in some weird situations, such as when the file
            # is actually a special file descriptor like stdin. In this
            # instance, we don't know what the length is, so set it to zero and
            # let requests chunk it instead.
            if total_length is not None:
                current_position = total_length
        else:
            if hasattr(o, "seek") and total_length is None:
                # StringIO and BytesIO have seek but no usable fileno
                try:
                    # seek to end of file
                    o.seek(0, 2)
                    total_length = o.tell()

                    # seek back to current position to support
                    # partially read file-like objects
                    o.seek(current_position or 0)
                except OSError:
                    total_length = 0

    if total_length is None:
        total_length = 0

    return max(0, total_length - current_position)


def get_netrc_auth(url, raise_errors=False):
    """Returns the Requests tuple auth for a given url from netrc."""

    netrc_file = os.environ.get("NETRC")
    if netrc_file is not None:
        netrc_locations = (netrc_file,)
    else:
        netrc_locations = (f"~/{f}" for f in NETRC_FILES)

    try:
        from netrc import NetrcParseError, netrc

        netrc_path = None

        for f in netrc_locations:
            loc = os.path.expanduser(f)
            if os.path.exists(loc):
                netrc_path = loc
                break

        # Abort early if there isn't one.
        if netrc_path is None:
            return

        ri = urlparse(url)
        host = ri.hostname

        try:
            _netrc = netrc(netrc_path).authenticators(host)
            if _netrc:
                # Return with login / password
                login_i = 0 if _netrc[0] else 1
                return (_netrc[login_i], _netrc[2])
        except (NetrcParseError, OSError):
            # If there was a parsing error or a permissions issue reading the file,
            # we'll just skip netrc auth unless explicitly asked to raise errors.
            if raise_errors:
                raise

    # App Engine hackiness.
    except (ImportError, AttributeError):
        pass


def guess_filename(obj):
    """Tries to guess the filename of the given object."""
    name = getattr(obj, "name", None)
    if name and isinstance(name, basestring) and name[0] != "<" and name[-1] != ">":
        return os.path.basename(name)


def extract_zipped_paths(path):
    """Replace nonexistent paths that look like they refer to a member of a zip
    archive with the location of an extracted copy of the target, or else
    just return the provided path unchanged.
    """
    if os.path.exists(path):
        # this is already a valid path, no need to do anything further
        return path

    # find the first valid part of the provided path and treat that as a zip archive
    # assume the rest of the path is the name of a member in the archive
    archive, member = os.path.split(path)
    while archive and not os.path.exists(archive):
        archive, prefix = os.path.split(archive)
        if not prefix:
            # If we don't check for an empty prefix after the split (in other words, archive remains unchanged after the split),
            # we _can_ end up in an infinite loop on a rare corner case affecting a small number of users
            break
        member = "/".join([prefix, member])

    if not zipfile.is_zipfile(archive):
        return path

    zip_file = zipfile.ZipFile(archive)
    if member not in zip_file.namelist():
        return path

    # we have a valid zip archive and a valid member of that archive
    tmp = tempfile.gettempdir()
    extracted_path = os.path.join(tmp, member.split("/")[-1])
    if not os.path.exists(extracted_path):
        # use read + write to avoid the creating nested folders, we only want the file, avoids mkdir racing condition
        with atomic_open(extracted_path) as file_handler:
            file_handler.write(zip_file.read(member))
    return extracted_path


@contextlib.contextmanager
def atomic_open(filename):
    """Write a file to the disk in an atomic fashion"""
    tmp_descriptor, tmp_name = tempfile.mkstemp(dir=os.path.dirname(filename))
    try:
        with os.fdopen(tmp_descriptor, "wb") as tmp_handler:
            yield tmp_handler
        os.replace(tmp_name, filename)
    except BaseException:
        os.remove(tmp_name)
        raise


def from_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. Unless it can not be represented as such, return an
    OrderedDict, e.g.,

    ::

        >>> from_key_val_list([('key', 'val')])
        OrderedDict([('key', 'val')])
        >>> from_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples
        >>> from_key_val_list({'key': 'val'})
        OrderedDict([('key', 'val')])

    :rtype: OrderedDict
    """
    if value is None:
        return None

    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    return OrderedDict(value)


def to_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. If it can be, return a list of tuples, e.g.,

    ::

        >>> to_key_val_list([('key', 'val')])
        [('key', 'val')]
        >>> to_key_val_list({'key': 'val'})
        [('key', 'val')]
        >>> to_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples

    :rtype: list
    """
    if value is None:
        return None

    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    if isinstance(value, Mapping):
        value = value.items()

    return list(value)


# From mitsuhiko/werkzeug (used with permission).
def parse_list_header(value):
    """Parse lists as described by RFC 2068 Section 2.

    In particular, parse comma-separated lists where the elements of
    the list may include quoted-strings.  A quoted-string could
    contain a comma.  A non-quoted string could have quotes in the
    middle.  Quotes are removed automatically after parsing.

    It basically works like :func:`parse_set_header` just that items
    may appear multiple times and case sensitivity is preserved.

    The return value is a standard :class:`list`:

    >>> parse_list_header('token, "quoted value"')
    ['token', 'quoted value']

    To create a header from the :class:`list` again, use the
    :func:`dump_header` function.

    :param value: a string with a list header.
    :return: :class:`list`
    :rtype: list
    """
    result = []
    for item in _parse_list_header(value):
        if item[:1] == item[-1:] == '"':
            item = unquote_header_value(item[1:-1])
        result.append(item)
    return result


# From mitsuhiko/werkzeug (used with permission).
def parse_dict_header(value):
    """Parse lists of key, value pairs as described by RFC 2068 Section 2 and
    convert them into a python dict:

    >>> d = parse_dict_header('foo="is a fish", bar="as well"')
    >>> type(d) is dict
    True
    >>> sorted(d.items())
    [('bar', 'as well'), ('foo', 'is a fish')]

    If there is no value for a key it will be `None`:

    >>> parse_dict_header('key_without_value')
    {'key_without_value': None}

    To create a header from the :class:`dict` again, use the
    :func:`dump_header` function.

    :param value: a string with a dict header.
    :return: :class:`dict`
    :rtype: dict
    """
    result = {}
    for item in _parse_list_header(value):
        if "=" not in item:
            result[item] = None
            continue
        name, value = item.split("=", 1)
        if value[:1] == value[-1:] == '"':
            value = unquote_header_value(value[1:-1])
        result[name] = value
    return result


# From mitsuhiko/werkzeug (used with permission).
def unquote_header_value(value, is_filename=False):
    r"""Unquotes a header value.  (Reversal of :func:`quote_header_value`).
    This does not use the real unquoting but what browsers are actually
    using for quoting.

    :param value: the header value to unquote.
    :rtype: str
    """
    if value and value[0] == value[-1] == '"':
        # this is not the real unquoting, but fixing this so that the
        # RFC is met will result in bugs with internet explorer and
        # probably some other browsers as well.  IE for example is
        # uploading files with "C:\foo\bar.txt" as filename
        value = value[1:-1]

        # if this is a filename and the starting characters look like
        # a UNC path, then just return the value without quotes.  Using the
        # replace sequence below on a UNC path has the effect of turning
        # the leading double slash into a single slash and then
        # _fix_ie_filename() doesn't work correctly.  See #458.
        if not is_filename or value[:2] != "\\\\":
            return value.replace("\\\\", "\\").replace('\\"', '"')
    return value


def dict_from_cookiejar(cj):
    """Returns a key/value dictionary from a CookieJar.

    :param cj: CookieJar object to extract cookies from.
    :rtype: dict
    """

    cookie_dict = {cookie.name: cookie.value for cookie in cj}
    return cookie_dict


def add_dict_to_cookiejar(cj, cookie_dict):
    """Returns a CookieJar from a key/value dictionary.

    :param cj: CookieJar to insert cookies into.
    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :rtype: CookieJar
    """

    return cookiejar_from_dict(cookie_dict, cj)


def get_encodings_from_content(content):
    """Returns encodings from given content string.

    :param content: bytestring to extract encodings from.
    """
    warnings.warn(
        (
            "In requests 3.0, get_encodings_from_content will be removed. For "
            "more information, please see the discussion on issue #2266. (This"
            " warning should only appear once.)"
        ),
        DeprecationWarning,
    )

    charset_re = re.compile(r'<meta.*?charset=["\']*(.+?)["\'>]', flags=re.I)
    pragma_re = re.compile(r'<meta.*?content=["\']*;?charset=(.+?)["\'>]', flags=re.I)
    xml_re = re.compile(r'^<\?xml.*?encoding=["\']*(.+?)["\'>]')

    return (
        charset_re.findall(content)
        + pragma_re.findall(content)
        + xml_re.findall(content)
    )


def _parse_content_type_header(header):
    """Returns content type and parameters from given header

    :param header: string
    :return: tuple containing content type and dictionary of
         parameters
    """

    tokens = header.split(";")
    content_type, params = tokens[0].strip(), tokens[1:]
    params_dict = {}
    items_to_strip = "\"' "

    for param in params:
        param = param.strip()
        if param:
            key, value = param, True
            index_of_equals = param.find("=")
            if index_of_equals != -1:
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1 :].strip(items_to_strip)
            params_dict[key.lower()] = value
    return content_type, params_dict


def get_encoding_from_headers(headers):
    """Returns encodings from given HTTP Header Dict.

    :param headers: dictionary to extract encoding from.
    :rtype: str
    """

    content_type = headers.get("content-type")

    if not content_type:
        return None

    content_type, params = _parse_content_type_header(content_type)

    if "charset" in params:
        return params["charset"].strip("'\"")

    if "text" in content_type:
        return "ISO-8859-1"

    if "application/json" in content_type:
        # Assume UTF-8 based on RFC 4627: https://www.ietf.org/rfc/rfc4627.txt since the charset was unset
        return "utf-8"


def stream_decode_response_unicode(iterator, r):
    """Stream decodes an iterator."""

    if r.encoding is None:
        yield from iterator
        return

    decoder = codecs.getincrementaldecoder(r.encoding)(errors="replace")
    for chunk in iterator:
        rv = decoder.decode(chunk)
        if rv:
            yield rv
    rv = decoder.decode(b"", final=True)
    if rv:
        yield rv


def iter_slices(string, slice_length):
    """Iterate over slices of a string."""
    pos = 0
    if slice_length is None or slice_length <= 0:
        slice_length = len(string)
    while pos < len(string):
        yield string[pos : pos + slice_length]
        pos += slice_length


def get_unicode_from_response(r):
    """Returns the requested content back in unicode.

    :param r: Response object to get unicode content from.

    Tried:

    1. charset from content-type
    2. fall back and replace all unicode characters

    :rtype: str
    """
    warnings.warn(
        (
            "In requests 3.0, get_unicode_from_response will be removed. For "
            "more information, please see the discussion on issue #2266. (This"
            " warning should only appear once.)"
        ),
        DeprecationWarning,
    )

    tried_encodings = []

    # Try charset from content-type
    encoding = get_encoding_from_headers(r.headers)

    if encoding:
        try:
            return str(r.content, encoding)
        except UnicodeError:
            tried_encodings.append(encoding)

    # Fall back:
    try:
        return str(r.content, encoding, errors="replace")
    except TypeError:
        return r.content


# The unreserved URI characters (RFC 3986)
UNRESERVED_SET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + "0123456789-._~"
)


def unquote_unreserved(uri):
    """Un-escape any percent-escape sequences in a URI that are unreserved
    characters. This leaves all reserved, illegal and non-ASCII bytes encoded.

    :rtype: str
    """
    parts = uri.split("%")
    for i in range(1, len(parts)):
        h = parts[i][0:2]
        if len(h) == 2 and h.isalnum():
            try:
                c = chr(int(h, 16))
            except ValueError:
                raise InvalidURL(f"Invalid percent-escape sequence: '{h}'")

            if c in UNRESERVED_SET:
                parts[i] = c + parts[i][2:]
            else:
                parts[i] = f"%{parts[i]}"
        else:
            parts[i] = f"%{parts[i]}"
    return "".join(parts)


def requote_uri(uri):
    """Re-quote the given URI.

    This function passes the given URI through an unquote/quote cycle to
    ensure that it is fully and consistently quoted.

    :rtype: str
    """
    safe_with_percent = "!#$%&'()*+,/:;=?@[]~"
    safe_without_percent = "!#$&'()*+,/:;=?@[]~"
    try:
        # Unquote only the unreserved characters
        # Then quote only illegal characters (do not quote reserved,
        # unreserved, or '%')
        return quote(unquote_unreserved(uri), safe=safe_with_percent)
    except InvalidURL:
        # We couldn't unquote the given URI, so let's try quoting it, but
        # there may be unquoted '%'s in the URI. We need to make sure they're
        # properly quoted so they do not cause issues elsewhere.
        return quote(uri, safe=safe_without_percent)


def address_in_network(ip, net):
    """This function allows you to check if an IP belongs to a network subnet

    Example: returns True if ip = 192.168.1.1 and net = 192.168.1.0/24
             returns False if ip = 192.168.1.1 and net = 192.168.100.0/24

    :rtype: bool
    """
    ipaddr = struct.unpack("=L", socket.inet_aton(ip))[0]
    netaddr, bits = net.split("/")
    netmask = struct.unpack("=L", socket.inet_aton(dotted_netmask(int(bits))))[0]
    network = struct.unpack("=L", socket.inet_aton(netaddr))[0] & netmask
    return (ipaddr & netmask) == (network & netmask)


def dotted_netmask(mask):
    """Converts mask from /xx format to xxx.xxx.xxx.xxx

    Example: if mask is 24 function returns 255.255.255.0

    :rtype: str
    """
    bits = 0xFFFFFFFF ^ (1 << 32 - mask) - 1
    return socket.inet_ntoa(struct.pack(">I", bits))


def is_ipv4_address(string_ip):
    """
    :rtype: bool
    """
    try:
        socket.inet_aton(string_ip)
    except OSError:
        return False
    return True


def is_valid_cidr(string_network):
    """
    Very simple check of the cidr format in no_proxy variable.

    :rtype: bool
    """
    if string_network.count("/") == 1:
        try:
            mask = int(string_network.split("/")[1])
        except ValueError:
            return False

        if mask < 1 or mask > 32:
            return False

        try:
            socket.inet_aton(string_network.split("/")[0])
        except OSError:
            return False
    else:
        return False
    return True


@contextlib.contextmanager
def set_environ(env_name, value):
    """Set the environment variable 'env_name' to 'value'

    Save previous value, yield, and then restore the previous value stored in
    the environment variable 'env_name'.

    If 'value' is None, do nothing"""
    value_changed = value is not None
    if value_changed:
        old_value = os.environ.get(env_name)
        os.environ[env_name] = value
    try:
        yield
    finally:
        if value_changed:
            if old_value is None:
                del os.environ[env_name]
            else:
                os.environ[env_name] = old_value


def should_bypass_proxies(url, no_proxy):
    """
    Returns whether we should bypass proxies or not.

    :rtype: bool
    """

    # Prioritize lowercase environment variables over uppercase
    # to keep a consistent behaviour with other http projects (curl, wget).
    def get_proxy(key):
        return os.environ.get(key) or os.environ.get(key.upper())

    # First check whether no_proxy is defined. If it is, check that the URL
    # we're getting isn't in the no_proxy list.
    no_proxy_arg = no_proxy
    if no_proxy is None:
        no_proxy = get_proxy("no_proxy")
    parsed = urlparse(url)

    if parsed.hostname is None:
        # URLs don't always have hostnames, e.g. file:/// urls.
        return True

    if no_proxy:
        # We need to check whether we match here. We need to see if we match
        # the end of the hostname, both with and without the port.
        no_proxy = (host for host in no_proxy.replace(" ", "").split(",") if host)

        if is_ipv4_address(parsed.hostname):
            for proxy_ip in no_proxy:
                if is_valid_cidr(proxy_ip):
                    if address_in_network(parsed.hostname, proxy_ip):
                        return True
                elif parsed.hostname == proxy_ip:
                    # If no_proxy ip was defined in plain IP notation instead of cidr notation &
                    # matches the IP of the index
                    return True
        else:
            host_with_port = parsed.hostname
            if parsed.port:
                host_with_port += f":{parsed.port}"

            for host in no_proxy:
                if parsed.hostname.endswith(host) or host_with_port.endswith(host):
                    # The URL does match something in no_proxy, so we don't want
                    # to apply the proxies on this URL.
                    return True

    with set_environ("no_proxy", no_proxy_arg):
        # parsed.hostname can be `None` in cases such as a file URI.
        try:
            bypass = proxy_bypass(parsed.hostname)
        except (TypeError, socket.gaierror):
            bypass = False

    if bypass:
        return True

    return False


def get_environ_proxies(url, no_proxy=None):
    """
    Return a dict of environment proxies.

    :rtype: dict
    """
    if should_bypass_proxies(url, no_proxy=no_proxy):
        return {}
    else:
        return getproxies()


def select_proxy(url, proxies):
    """Select a proxy for the url, if applicable.

    :param url: The url being for the request
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    """
    proxies = proxies or {}
    urlparts = urlparse(url)
    if urlparts.hostname is None:
        return proxies.get(urlparts.scheme, proxies.get("all"))

    proxy_keys = [
        urlparts.scheme + "://" + urlparts.hostname,
        urlparts.scheme,
        "all://" + urlparts.hostname,
        "all",
    ]
    proxy = None
    for proxy_key in proxy_keys:
        if proxy_key in proxies:
            proxy = proxies[proxy_key]
            break

    return proxy


def resolve_proxies(request, proxies, trust_env=True):
    """This method takes proxy information from a request and configuration
    input to resolve a mapping of target proxies. This will consider settings
    such as NO_PROXY to strip proxy configurations.

    :param request: Request or PreparedRequest
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    :param trust_env: Boolean declaring whether to trust environment configs

    :rtype: dict
    """
    proxies = proxies if proxies is not None else {}
    url = request.url
    scheme = urlparse(url).scheme
    no_proxy = proxies.get("no_proxy")
    new_proxies = proxies.copy()

    if trust_env and not should_bypass_proxies(url, no_proxy=no_proxy):
        environ_proxies = get_environ_proxies(url, no_proxy=no_proxy)

        proxy = environ_proxies.get(scheme, environ_proxies.get("all"))

        if proxy:
            new_proxies.setdefault(scheme, proxy)
    return new_proxies


def default_user_agent(name="python-requests"):
    """
    Return a string representing the default user agent.

    :rtype: str
    """
    return f"{name}/{__version__}"


def default_headers():
    """
    :rtype: requests.structures.CaseInsensitiveDict
    """
    return CaseInsensitiveDict(
        {
            "User-Agent": default_user_agent(),
            "Accept-Encoding": DEFAULT_ACCEPT_ENCODING,
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )


def parse_header_links(value):
    """Return a list of parsed link headers proxies.

    i.e. Link: <http:/.../front.jpeg>; rel=front; type="image/jpeg",<http://.../back.jpeg>; rel=back;type="image/jpeg"

    :rtype: list
    """

    links = []

    replace_chars = " '\""

    value = value.strip(replace_chars)
    if not value:
        return links

    for val in re.split(", *<", value):
        try:
            url, params = val.split(";", 1)
        except ValueError:
            url, params = val, ""

        link = {"url": url.strip("<> '\"")}

        for param in params.split(";"):
            try:
                key, value = param.split("=")
            except ValueError:
                break

            link[key.strip(replace_chars)] = value.strip(replace_chars)

        links.append(link)

    return links


# Null bytes; no need to recreate these on each call to guess_json_utf
_null = "\x00".encode("ascii")  # encoding to ASCII for Python 3
_null2 = _null * 2
_null3 = _null * 3


def guess_json_utf(data):
    """
    :rtype: str
    """
    # JSON always starts with two ASCII characters, so detection is as
    # easy as counting the nulls and from their location and count
    # determine the encoding. Also detect a BOM, if present.
    sample = data[:4]
    if sample in (codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE):
        return "utf-32"  # BOM included
    if sample[:3] == codecs.BOM_UTF8:
        return "utf-8-sig"  # BOM included, MS style (discouraged)
    if sample[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return "utf-16"  # BOM included
    nullcount = sample.count(_null)
    if nullcount == 0:
        return "utf-8"
    if nullcount == 2:
        if sample[::2] == _null2:  # 1st and 3rd are null
            return "utf-16-be"
        if sample[1::2] == _null2:  # 2nd and 4th are null
            return "utf-16-le"
        # Did not detect 2 valid UTF-16 ascii-range characters
    if nullcount == 3:
        if sample[:3] == _null3:
            return "utf-32-be"
        if sample[1:] == _null3:
            return "utf-32-le"
        # Did not detect a valid UTF-32 ascii-range character
    return None


def prepend_scheme_if_needed(url, new_scheme):
    """Given a URL that may or may not have a scheme, prepend the given scheme.
    Does not replace a present scheme with the one provided as an argument.

    :rtype: str
    """
    parsed = parse_url(url)
    scheme, auth, host, port, path, query, fragment = parsed

    # A defect in urlparse determines that there isn't a netloc present in some
    # urls. We previously assumed parsing was overly cautious, and swapped the
    # netloc and path. Due to a lack of tests on the original defect, this is
    # maintained with parse_url for backwards compatibility.
    netloc = parsed.netloc
    if not netloc:
        netloc, path = path, netloc

    if auth:
        # parse_url doesn't provide the netloc with auth
        # so we'll add it ourselves.
        netloc = "@".join([auth, netloc])
    if scheme is None:
        scheme = new_scheme
    if path is None:
        path = ""

    return urlunparse((scheme, netloc, path, "", query, fragment))


def get_auth_from_url(url):
    """Given a url with authentication components, extract them into a tuple of
    username,password.

    :rtype: (str,str)
    """
    parsed = urlparse(url)

    try:
        auth = (unquote(parsed.username), unquote(parsed.password))
    except (AttributeError, TypeError):
        auth = ("", "")

    return auth


def check_header_validity(header):
    """Verifies that header parts don't contain leading whitespace
    reserved characters, or return characters.

    :param header: tuple, in the format (name, value).
    """
    name, value = header
    _validate_header_part(header, name, 0)
    _validate_header_part(header, value, 1)


def _validate_header_part(header, header_part, header_validator_index):
    if isinstance(header_part, str):
        validator = _HEADER_VALIDATORS_STR[header_validator_index]
    elif isinstance(header_part, bytes):
        validator = _HEADER_VALIDATORS_BYTE[header_validator_index]
    else:
        raise InvalidHeader(
            f"Header part ({header_part!r}) from {header} "
            f"must be of type str or bytes, not {type(header_part)}"
        )

    if not validator.match(header_part):
        header_kind = "name" if header_validator_index == 0 else "value"
        raise InvalidHeader(
            f"Invalid leading whitespace, reserved character(s), or return "
            f"character(s) in header {header_kind}: {header_part!r}"
        )


def urldefragauth(url):
    """
    Given a url remove the fragment and the authentication part.

    :rtype: str
    """
    scheme, netloc, path, params, query, fragment = urlparse(url)

    # see func:`prepend_scheme_if_needed`
    if not netloc:
        netloc, path = path, netloc

    netloc = netloc.rsplit("@", 1)[-1]

    return urlunparse((scheme, netloc, path, params, query, ""))


def rewind_body(prepared_request):
    """Move file pointer back to its recorded starting position
    so it can be read again on redirect.
    """
    body_seek = getattr(prepared_request.body, "seek", None)
    if body_seek is not None and isinstance(
        prepared_request._body_position, integer_types
    ):
        try:
            body_seek(prepared_request._body_position)
        except OSError:
            raise UnrewindableBodyError(
                "An error occurred when rewinding request body for redirect."
            )
    else:
        raise UnrewindableBodyError("Unable to rewind request body for redirect.")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\mathematica.py ===
from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable

import sympy
from sympy import Mul, Add, Pow, Rational, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
    acosh, atanh, acoth, asech, acsch, expand, im, flatten, polylog, cancel, expand_trig, sign, simplify, \
    UnevaluatedExpr, S, atan, atan2, Mod, Max, Min, rf, Ei, Si, Ci, airyai, airyaiprime, airybi, primepi, prime, \
    isprime, cot, sec, csc, csch, sech, coth, Function, I, pi, Tuple, GreaterThan, StrictGreaterThan, StrictLessThan, \
    LessThan, Equality, Or, And, Lambda, Integer, Dummy, symbols
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning


def mathematica(s, additional_translations=None):
    sympy_deprecation_warning(
        """The ``mathematica`` function for the Mathematica parser is now
deprecated. Use ``parse_mathematica`` instead.
The parameter ``additional_translation`` can be replaced by SymPy's
.replace( ) or .subs( ) methods on the output expression instead.""",
        deprecated_since_version="1.11",
        active_deprecations_target="mathematica-parser-new",
    )
    parser = MathematicaParser(additional_translations)
    return sympify(parser._parse_old(s))


def parse_mathematica(s):
    """
    Translate a string containing a Wolfram Mathematica expression to a SymPy
    expression.

    If the translator is unable to find a suitable SymPy expression, the
    ``FullForm`` of the Mathematica expression will be output, using SymPy
    ``Function`` objects as nodes of the syntax tree.

    Examples
    ========

    >>> from sympy.parsing.mathematica import parse_mathematica
    >>> parse_mathematica("Sin[x]^2 Tan[y]")
    sin(x)**2*tan(y)
    >>> e = parse_mathematica("F[7,5,3]")
    >>> e
    F(7, 5, 3)
    >>> from sympy import Function, Max, Min
    >>> e.replace(Function("F"), lambda *x: Max(*x)*Min(*x))
    21

    Both standard input form and Mathematica full form are supported:

    >>> parse_mathematica("x*(a + b)")
    x*(a + b)
    >>> parse_mathematica("Times[x, Plus[a, b]]")
    x*(a + b)

    To get a matrix from Wolfram's code:

    >>> m = parse_mathematica("{{a, b}, {c, d}}")
    >>> m
    ((a, b), (c, d))
    >>> from sympy import Matrix
    >>> Matrix(m)
    Matrix([
    [a, b],
    [c, d]])

    If the translation into equivalent SymPy expressions fails, an SymPy
    expression equivalent to Wolfram Mathematica's "FullForm" will be created:

    >>> parse_mathematica("x_.")
    Optional(Pattern(x, Blank()))
    >>> parse_mathematica("Plus @@ {x, y, z}")
    Apply(Plus, (x, y, z))
    >>> parse_mathematica("f[x_, 3] := x^3 /; x > 0")
    SetDelayed(f(Pattern(x, Blank()), 3), Condition(x**3, x > 0))
    """
    parser = MathematicaParser()
    return parser.parse(s)


def _parse_Function(*args):
    if len(args) == 1:
        arg = args[0]
        Slot = Function("Slot")
        slots = arg.atoms(Slot)
        numbers = [a.args[0] for a in slots]
        number_of_arguments = max(numbers)
        if isinstance(number_of_arguments, Integer):
            variables = symbols(f"dummy0:{number_of_arguments}", cls=Dummy)
            return Lambda(variables, arg.xreplace({Slot(i+1): v for i, v in enumerate(variables)}))
        return Lambda((), arg)
    elif len(args) == 2:
        variables = args[0]
        body = args[1]
        return Lambda(variables, body)
    else:
        raise SyntaxError("Function node expects 1 or 2 arguments")


def _deco(cls):
    cls._initialize_class()
    return cls


@_deco
class MathematicaParser:
    """
    An instance of this class converts a string of a Wolfram Mathematica
    expression to a SymPy expression.

    The main parser acts internally in three stages:

    1. tokenizer: tokenizes the Mathematica expression and adds the missing *
        operators. Handled by ``_from_mathematica_to_tokens(...)``
    2. full form list: sort the list of strings output by the tokenizer into a
        syntax tree of nested lists and strings, equivalent to Mathematica's
        ``FullForm`` expression output. This is handled by the function
        ``_from_tokens_to_fullformlist(...)``.
    3. SymPy expression: the syntax tree expressed as full form list is visited
        and the nodes with equivalent classes in SymPy are replaced. Unknown
        syntax tree nodes are cast to SymPy ``Function`` objects. This is
        handled by ``_from_fullformlist_to_sympy(...)``.

    """

    # left: Mathematica, right: SymPy
    CORRESPONDENCES = {
        'Sqrt[x]': 'sqrt(x)',
        'Rational[x,y]': 'Rational(x,y)',
        'Exp[x]': 'exp(x)',
        'Log[x]': 'log(x)',
        'Log[x,y]': 'log(y,x)',
        'Log2[x]': 'log(x,2)',
        'Log10[x]': 'log(x,10)',
        'Mod[x,y]': 'Mod(x,y)',
        'Max[*x]': 'Max(*x)',
        'Min[*x]': 'Min(*x)',
        'Pochhammer[x,y]':'rf(x,y)',
        'ArcTan[x,y]':'atan2(y,x)',
        'ExpIntegralEi[x]': 'Ei(x)',
        'SinIntegral[x]': 'Si(x)',
        'CosIntegral[x]': 'Ci(x)',
        'AiryAi[x]': 'airyai(x)',
        'AiryAiPrime[x]': 'airyaiprime(x)',
        'AiryBi[x]' :'airybi(x)',
        'AiryBiPrime[x]' :'airybiprime(x)',
        'LogIntegral[x]':' li(x)',
        'PrimePi[x]': 'primepi(x)',
        'Prime[x]': 'prime(x)',
        'PrimeQ[x]': 'isprime(x)'
    }

    # trigonometric, e.t.c.
    for arc, tri, h in product(('', 'Arc'), (
            'Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc'), ('', 'h')):
        fm = arc + tri + h + '[x]'
        if arc:  # arc func
            fs = 'a' + tri.lower() + h + '(x)'
        else:    # non-arc func
            fs = tri.lower() + h + '(x)'
        CORRESPONDENCES.update({fm: fs})

    REPLACEMENTS = {
        ' ': '',
        '^': '**',
        '{': '[',
        '}': ']',
    }

    RULES = {
        # a single whitespace to '*'
        'whitespace': (
            re.compile(r'''
                (?:(?<=[a-zA-Z\d])|(?<=\d\.))     # a letter or a number
                \s+                               # any number of whitespaces
                (?:(?=[a-zA-Z\d])|(?=\.\d))       # a letter or a number
                ''', re.VERBOSE),
            '*'),

        # add omitted '*' character
        'add*_1': (
            re.compile(r'''
                (?:(?<=[])\d])|(?<=\d\.))       # ], ) or a number
                                                # ''
                (?=[(a-zA-Z])                   # ( or a single letter
                ''', re.VERBOSE),
            '*'),

        # add omitted '*' character (variable letter preceding)
        'add*_2': (
            re.compile(r'''
                (?<=[a-zA-Z])       # a letter
                \(                  # ( as a character
                (?=.)               # any characters
                ''', re.VERBOSE),
            '*('),

        # convert 'Pi' to 'pi'
        'Pi': (
            re.compile(r'''
                (?:
                \A|(?<=[^a-zA-Z])
                )
                Pi                  # 'Pi' is 3.14159... in Mathematica
                (?=[^a-zA-Z])
                ''', re.VERBOSE),
            'pi'),
    }

    # Mathematica function name pattern
    FM_PATTERN = re.compile(r'''
                (?:
                \A|(?<=[^a-zA-Z])   # at the top or a non-letter
                )
                [A-Z][a-zA-Z\d]*    # Function
                (?=\[)              # [ as a character
                ''', re.VERBOSE)

    # list or matrix pattern (for future usage)
    ARG_MTRX_PATTERN = re.compile(r'''
                \{.*\}
                ''', re.VERBOSE)

    # regex string for function argument pattern
    ARGS_PATTERN_TEMPLATE = r'''
                (?:
                \A|(?<=[^a-zA-Z])
                )
                {arguments}         # model argument like x, y,...
                (?=[^a-zA-Z])
                '''

    # will contain transformed CORRESPONDENCES dictionary
    TRANSLATIONS: dict[tuple[str, int], dict[str, Any]] = {}

    # cache for a raw users' translation dictionary
    cache_original: dict[tuple[str, int], dict[str, Any]] = {}

    # cache for a compiled users' translation dictionary
    cache_compiled: dict[tuple[str, int], dict[str, Any]] = {}

    @classmethod
    def _initialize_class(cls):
        # get a transformed CORRESPONDENCES dictionary
        d = cls._compile_dictionary(cls.CORRESPONDENCES)
        cls.TRANSLATIONS.update(d)

    def __init__(self, additional_translations=None):
        self.translations = {}

        # update with TRANSLATIONS (class constant)
        self.translations.update(self.TRANSLATIONS)

        if additional_translations is None:
            additional_translations = {}

        # check the latest added translations
        if self.__class__.cache_original != additional_translations:
            if not isinstance(additional_translations, dict):
                raise ValueError('The argument must be dict type')

            # get a transformed additional_translations dictionary
            d = self._compile_dictionary(additional_translations)

            # update cache
            self.__class__.cache_original = additional_translations
            self.__class__.cache_compiled = d

        # merge user's own translations
        self.translations.update(self.__class__.cache_compiled)

    @classmethod
    def _compile_dictionary(cls, dic):
        # for return
        d = {}

        for fm, fs in dic.items():
            # check function form
            cls._check_input(fm)
            cls._check_input(fs)

            # uncover '*' hiding behind a whitespace
            fm = cls._apply_rules(fm, 'whitespace')
            fs = cls._apply_rules(fs, 'whitespace')

            # remove whitespace(s)
            fm = cls._replace(fm, ' ')
            fs = cls._replace(fs, ' ')

            # search Mathematica function name
            m = cls.FM_PATTERN.search(fm)

            # if no-hit
            if m is None:
                err = "'{f}' function form is invalid.".format(f=fm)
                raise ValueError(err)

            # get Mathematica function name like 'Log'
            fm_name = m.group()

            # get arguments of Mathematica function
            args, end = cls._get_args(m)

            # function side check. (e.g.) '2*Func[x]' is invalid.
            if m.start() != 0 or end != len(fm):
                err = "'{f}' function form is invalid.".format(f=fm)
                raise ValueError(err)

            # check the last argument's 1st character
            if args[-1][0] == '*':
                key_arg = '*'
            else:
                key_arg = len(args)

            key = (fm_name, key_arg)

            # convert '*x' to '\\*x' for regex
            re_args = [x if x[0] != '*' else '\\' + x for x in args]

            # for regex. Example: (?:(x|y|z))
            xyz = '(?:(' + '|'.join(re_args) + '))'

            # string for regex compile
            patStr = cls.ARGS_PATTERN_TEMPLATE.format(arguments=xyz)

            pat = re.compile(patStr, re.VERBOSE)

            # update dictionary
            d[key] = {}
            d[key]['fs'] = fs  # SymPy function template
            d[key]['args'] = args  # args are ['x', 'y'] for example
            d[key]['pat'] = pat

        return d

    def _convert_function(self, s):
        '''Parse Mathematica function to SymPy one'''

        # compiled regex object
        pat = self.FM_PATTERN

        scanned = ''                # converted string
        cur = 0                     # position cursor
        while True:
            m = pat.search(s)

            if m is None:
                # append the rest of string
                scanned += s
                break

            # get Mathematica function name
            fm = m.group()

            # get arguments, and the end position of fm function
            args, end = self._get_args(m)

            # the start position of fm function
            bgn = m.start()

            # convert Mathematica function to SymPy one
            s = self._convert_one_function(s, fm, args, bgn, end)

            # update cursor
            cur = bgn

            # append converted part
            scanned += s[:cur]

            # shrink s
            s = s[cur:]

        return scanned

    def _convert_one_function(self, s, fm, args, bgn, end):
        # no variable-length argument
        if (fm, len(args)) in self.translations:
            key = (fm, len(args))

            # x, y,... model arguments
            x_args = self.translations[key]['args']

            # make CORRESPONDENCES between model arguments and actual ones
            d = dict(zip(x_args, args))

        # with variable-length argument
        elif (fm, '*') in self.translations:
            key = (fm, '*')

            # x, y,..*args (model arguments)
            x_args = self.translations[key]['args']

            # make CORRESPONDENCES between model arguments and actual ones
            d = {}
            for i, x in enumerate(x_args):
                if x[0] == '*':
                    d[x] = ','.join(args[i:])
                    break
                d[x] = args[i]

        # out of self.translations
        else:
            err = "'{f}' is out of the whitelist.".format(f=fm)
            raise ValueError(err)

        # template string of converted function
        template = self.translations[key]['fs']

        # regex pattern for x_args
        pat = self.translations[key]['pat']

        scanned = ''
        cur = 0
        while True:
            m = pat.search(template)

            if m is None:
                scanned += template
                break

            # get model argument
            x = m.group()

            # get a start position of the model argument
            xbgn = m.start()

            # add the corresponding actual argument
            scanned += template[:xbgn] + d[x]

            # update cursor to the end of the model argument
            cur = m.end()

            # shrink template
            template = template[cur:]

        # update to swapped string
        s = s[:bgn] + scanned + s[end:]

        return s

    @classmethod
    def _get_args(cls, m):
        '''Get arguments of a Mathematica function'''

        s = m.string                # whole string
        anc = m.end() + 1           # pointing the first letter of arguments
        square, curly = [], []      # stack for brackets
        args = []

        # current cursor
        cur = anc
        for i, c in enumerate(s[anc:], anc):
            # extract one argument
            if c == ',' and (not square) and (not curly):
                args.append(s[cur:i])       # add an argument
                cur = i + 1                 # move cursor

            # handle list or matrix (for future usage)
            if c == '{':
                curly.append(c)
            elif c == '}':
                curly.pop()

            # seek corresponding ']' with skipping irrevant ones
            if c == '[':
                square.append(c)
            elif c == ']':
                if square:
                    square.pop()
                else:   # empty stack
                    args.append(s[cur:i])
                    break

        # the next position to ']' bracket (the function end)
        func_end = i + 1

        return args, func_end

    @classmethod
    def _replace(cls, s, bef):
        aft = cls.REPLACEMENTS[bef]
        s = s.replace(bef, aft)
        return s

    @classmethod
    def _apply_rules(cls, s, bef):
        pat, aft = cls.RULES[bef]
        return pat.sub(aft, s)

    @classmethod
    def _check_input(cls, s):
        for bracket in (('[', ']'), ('{', '}'), ('(', ')')):
            if s.count(bracket[0]) != s.count(bracket[1]):
                err = "'{f}' function form is invalid.".format(f=s)
                raise ValueError(err)

        if '{' in s:
            err = "Currently list is not supported."
            raise ValueError(err)

    def _parse_old(self, s):
        # input check
        self._check_input(s)

        # uncover '*' hiding behind a whitespace
        s = self._apply_rules(s, 'whitespace')

        # remove whitespace(s)
        s = self._replace(s, ' ')

        # add omitted '*' character
        s = self._apply_rules(s, 'add*_1')
        s = self._apply_rules(s, 'add*_2')

        # translate function
        s = self._convert_function(s)

        # '^' to '**'
        s = self._replace(s, '^')

        # 'Pi' to 'pi'
        s = self._apply_rules(s, 'Pi')

        # '{', '}' to '[', ']', respectively
#        s = cls._replace(s, '{')   # currently list is not taken into account
#        s = cls._replace(s, '}')

        return s

    def parse(self, s):
        s2 = self._from_mathematica_to_tokens(s)
        s3 = self._from_tokens_to_fullformlist(s2)
        s4 = self._from_fullformlist_to_sympy(s3)
        return s4

    INFIX = "Infix"
    PREFIX = "Prefix"
    POSTFIX = "Postfix"
    FLAT = "Flat"
    RIGHT = "Right"
    LEFT = "Left"

    _mathematica_op_precedence: list[tuple[str, str | None, dict[str, str | Callable]]] = [
        (POSTFIX, None, {";": lambda x: x + ["Null"] if isinstance(x, list) and x and x[0] == "CompoundExpression" else ["CompoundExpression", x, "Null"]}),
        (INFIX, FLAT, {";": "CompoundExpression"}),
        (INFIX, RIGHT, {"=": "Set", ":=": "SetDelayed", "+=": "AddTo", "-=": "SubtractFrom", "*=": "TimesBy", "/=": "DivideBy"}),
        (INFIX, LEFT, {"//": lambda x, y: [x, y]}),
        (POSTFIX, None, {"&": "Function"}),
        (INFIX, LEFT, {"/.": "ReplaceAll"}),
        (INFIX, RIGHT, {"->": "Rule", ":>": "RuleDelayed"}),
        (INFIX, LEFT, {"/;": "Condition"}),
        (INFIX, FLAT, {"|": "Alternatives"}),
        (POSTFIX, None, {"..": "Repeated", "...": "RepeatedNull"}),
        (INFIX, FLAT, {"||": "Or"}),
        (INFIX, FLAT, {"&&": "And"}),
        (PREFIX, None, {"!": "Not"}),
        (INFIX, FLAT, {"===": "SameQ", "=!=": "UnsameQ"}),
        (INFIX, FLAT, {"==": "Equal", "!=": "Unequal", "<=": "LessEqual", "<": "Less", ">=": "GreaterEqual", ">": "Greater"}),
        (INFIX, None, {";;": "Span"}),
        (INFIX, FLAT, {"+": "Plus", "-": "Plus"}),
        (INFIX, FLAT, {"*": "Times", "/": "Times"}),
        (INFIX, FLAT, {".": "Dot"}),
        (PREFIX, None, {"-": lambda x: MathematicaParser._get_neg(x),
                        "+": lambda x: x}),
        (INFIX, RIGHT, {"^": "Power"}),
        (INFIX, RIGHT, {"@@": "Apply", "/@": "Map", "//@": "MapAll", "@@@": lambda x, y: ["Apply", x, y, ["List", "1"]]}),
        (POSTFIX, None, {"'": "Derivative", "!": "Factorial", "!!": "Factorial2", "--": "Decrement"}),
        (INFIX, None, {"[": lambda x, y: [x, *y], "[[": lambda x, y: ["Part", x, *y]}),
        (PREFIX, None, {"{": lambda x: ["List", *x], "(": lambda x: x[0]}),
        (INFIX, None, {"?": "PatternTest"}),
        (POSTFIX, None, {
            "_": lambda x: ["Pattern", x, ["Blank"]],
            "_.": lambda x: ["Optional", ["Pattern", x, ["Blank"]]],
            "__": lambda x: ["Pattern", x, ["BlankSequence"]],
            "___": lambda x: ["Pattern", x, ["BlankNullSequence"]],
        }),
        (INFIX, None, {"_": lambda x, y: ["Pattern", x, ["Blank", y]]}),
        (PREFIX, None, {"#": "Slot", "##": "SlotSequence"}),
    ]

    _missing_arguments_default = {
        "#": lambda: ["Slot", "1"],
        "##": lambda: ["SlotSequence", "1"],
    }

    _literal = r"[A-Za-z][A-Za-z0-9]*"
    _number = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"

    _enclosure_open = ["(", "[", "[[", "{"]
    _enclosure_close = [")", "]", "]]", "}"]

    @classmethod
    def _get_neg(cls, x):
        return f"-{x}" if isinstance(x, str) and re.match(MathematicaParser._number, x) else ["Times", "-1", x]

    @classmethod
    def _get_inv(cls, x):
        return ["Power", x, "-1"]

    _regex_tokenizer = None

    def _get_tokenizer(self):
        if self._regex_tokenizer is not None:
            # Check if the regular expression has already been compiled:
            return self._regex_tokenizer
        tokens = [self._literal, self._number]
        tokens_escape = self._enclosure_open[:] + self._enclosure_close[:]
        for typ, strat, symdict in self._mathematica_op_precedence:
            for k in symdict:
                tokens_escape.append(k)
        tokens_escape.sort(key=lambda x: -len(x))
        tokens.extend(map(re.escape, tokens_escape))
        tokens.append(",")
        tokens.append("\n")
        tokenizer = re.compile("(" + "|".join(tokens) + ")")
        self._regex_tokenizer = tokenizer
        return self._regex_tokenizer

    def _from_mathematica_to_tokens(self, code: str):
        tokenizer = self._get_tokenizer()

        # Find strings:
        code_splits: list[str | list] = []
        while True:
            string_start = code.find("\"")
            if string_start == -1:
                if len(code) > 0:
                    code_splits.append(code)
                break
            match_end = re.search(r'(?<!\\)"', code[string_start+1:])
            if match_end is None:
                raise SyntaxError('mismatch in string "  " expression')
            string_end = string_start + match_end.start() + 1
            if string_start > 0:
                code_splits.append(code[:string_start])
            code_splits.append(["_Str", code[string_start+1:string_end].replace('\\"', '"')])
            code = code[string_end+1:]

        # Remove comments:
        for i, code_split in enumerate(code_splits):
            if isinstance(code_split, list):
                continue
            while True:
                pos_comment_start = code_split.find("(*")
                if pos_comment_start == -1:
                    break
                pos_comment_end = code_split.find("*)")
                if pos_comment_end == -1 or pos_comment_end < pos_comment_start:
                    raise SyntaxError("mismatch in comment (*  *) code")
                code_split = code_split[:pos_comment_start] + code_split[pos_comment_end+2:]
            code_splits[i] = code_split

        # Tokenize the input strings with a regular expression:
        token_lists = [tokenizer.findall(i) if isinstance(i, str) and i.isascii() else [i] for i in code_splits]
        tokens = [j for i in token_lists for j in i]

        # Remove newlines at the beginning
        while tokens and tokens[0] == "\n":
            tokens.pop(0)
        # Remove newlines at the end
        while tokens and tokens[-1] == "\n":
            tokens.pop(-1)

        return tokens

    def _is_op(self, token: str | list) -> bool:
        if isinstance(token, list):
            return False
        if re.match(self._literal, token):
            return False
        if re.match("-?" + self._number, token):
            return False
        return True

    def _is_valid_star1(self, token: str | list) -> bool:
        if token in (")", "}"):
            return True
        return not self._is_op(token)

    def _is_valid_star2(self, token: str | list) -> bool:
        if token in ("(", "{"):
            return True
        return not self._is_op(token)

    def _from_tokens_to_fullformlist(self, tokens: list):
        stack: list[list] = [[]]
        open_seq = []
        pointer: int = 0
        while pointer < len(tokens):
            token = tokens[pointer]
            if token in self._enclosure_open:
                stack[-1].append(token)
                open_seq.append(token)
                stack.append([])
            elif token == ",":
                if len(stack[-1]) == 0 and stack[-2][-1] == open_seq[-1]:
                    raise SyntaxError("%s cannot be followed by comma ," % open_seq[-1])
                stack[-1] = self._parse_after_braces(stack[-1])
                stack.append([])
            elif token in self._enclosure_close:
                ind = self._enclosure_close.index(token)
                if self._enclosure_open[ind] != open_seq[-1]:
                    unmatched_enclosure = SyntaxError("unmatched enclosure")
                    if token == "]]" and open_seq[-1] == "[":
                        if open_seq[-2] == "[":
                            # These two lines would be logically correct, but are
                            # unnecessary:
                            # token = "]"
                            # tokens[pointer] = "]"
                            tokens.insert(pointer+1, "]")
                        elif open_seq[-2] == "[[":
                            if tokens[pointer+1] == "]":
                                tokens[pointer+1] = "]]"
                            elif tokens[pointer+1] == "]]":
                                tokens[pointer+1] = "]]"
                                tokens.insert(pointer+2, "]")
                            else:
                                raise unmatched_enclosure
                    else:
                        raise unmatched_enclosure
                if len(stack[-1]) == 0 and stack[-2][-1] == "(":
                    raise SyntaxError("( ) not valid syntax")
                last_stack = self._parse_after_braces(stack[-1], True)
                stack[-1] = last_stack
                new_stack_element = []
                while stack[-1][-1] != open_seq[-1]:
                    new_stack_element.append(stack.pop())
                new_stack_element.reverse()
                if open_seq[-1] == "(" and len(new_stack_element) != 1:
                    raise SyntaxError("( must be followed by one expression, %i detected" % len(new_stack_element))
                stack[-1].append(new_stack_element)
                open_seq.pop(-1)
            else:
                stack[-1].append(token)
            pointer += 1
        if len(stack) != 1:
            raise RuntimeError("Stack should have only one element")
        return self._parse_after_braces(stack[0])

    def _util_remove_newlines(self, lines: list, tokens: list, inside_enclosure: bool):
        pointer = 0
        size = len(tokens)
        while pointer < size:
            token = tokens[pointer]
            if token == "\n":
                if inside_enclosure:
                    # Ignore newlines inside enclosures
                    tokens.pop(pointer)
                    size -= 1
                    continue
                if pointer == 0:
                    tokens.pop(0)
                    size -= 1
                    continue
                if pointer > 1:
                    try:
                        prev_expr = self._parse_after_braces(tokens[:pointer], inside_enclosure)
                    except SyntaxError:
                        tokens.pop(pointer)
                        size -= 1
                        continue
                else:
                    prev_expr = tokens[0]
                if len(prev_expr) > 0 and prev_expr[0] == "CompoundExpression":
                    lines.extend(prev_expr[1:])
                else:
                    lines.append(prev_expr)
                for i in range(pointer):
                    tokens.pop(0)
                size -= pointer
                pointer = 0
                continue
            pointer += 1

    def _util_add_missing_asterisks(self, tokens: list):
        size: int = len(tokens)
        pointer: int = 0
        while pointer < size:
            if (pointer > 0 and
                    self._is_valid_star1(tokens[pointer - 1]) and
                    self._is_valid_star2(tokens[pointer])):
                # This is a trick to add missing * operators in the expression,
                # `"*" in op_dict` makes sure the precedence level is the same as "*",
                # while `not self._is_op( ... )` makes sure this and the previous
                # expression are not operators.
                if tokens[pointer] == "(":
                    # ( has already been processed by now, replace:
                    tokens[pointer] = "*"
                    tokens[pointer + 1] = tokens[pointer + 1][0]
                else:
                    tokens.insert(pointer, "*")
                    pointer += 1
                    size += 1
            pointer += 1

    def _parse_after_braces(self, tokens: list, inside_enclosure: bool = False):
        op_dict: dict
        changed: bool = False
        lines: list = []

        self._util_remove_newlines(lines, tokens, inside_enclosure)

        for op_type, grouping_strat, op_dict in reversed(self._mathematica_op_precedence):
            if "*" in op_dict:
                self._util_add_missing_asterisks(tokens)
            size: int = len(tokens)
            pointer: int = 0
            while pointer < size:
                token = tokens[pointer]
                if isinstance(token, str) and token in op_dict:
                    op_name: str | Callable = op_dict[token]
                    node: list
                    first_index: int
                    if isinstance(op_name, str):
                        node = [op_name]
                        first_index = 1
                    else:
                        node = []
                        first_index = 0
                    if token in ("+", "-") and op_type == self.PREFIX and pointer > 0 and not self._is_op(tokens[pointer - 1]):
                        # Make sure that PREFIX + - don't match expressions like a + b or a - b,
                        # the INFIX + - are supposed to match that expression:
                        pointer += 1
                        continue
                    if op_type == self.INFIX:
                        if pointer == 0 or pointer == size - 1 or self._is_op(tokens[pointer - 1]) or self._is_op(tokens[pointer + 1]):
                            pointer += 1
                            continue
                    changed = True
                    tokens[pointer] = node
                    if op_type == self.INFIX:
                        arg1 = tokens.pop(pointer-1)
                        arg2 = tokens.pop(pointer)
                        if token == "/":
                            arg2 = self._get_inv(arg2)
                        elif token == "-":
                            arg2 = self._get_neg(arg2)
                        pointer -= 1
                        size -= 2
                        node.append(arg1)
                        node_p = node
                        if grouping_strat == self.FLAT:
                            while pointer + 2 < size and self._check_op_compatible(tokens[pointer+1], token):
                                node_p.append(arg2)
                                other_op = tokens.pop(pointer+1)
                                arg2 = tokens.pop(pointer+1)
                                if other_op == "/":
                                    arg2 = self._get_inv(arg2)
                                elif other_op == "-":
                                    arg2 = self._get_neg(arg2)
                                size -= 2
                            node_p.append(arg2)
                        elif grouping_strat == self.RIGHT:
                            while pointer + 2 < size and tokens[pointer+1] == token:
                                node_p.append([op_name, arg2])
                                node_p = node_p[-1]
                                tokens.pop(pointer+1)
                                arg2 = tokens.pop(pointer+1)
                                size -= 2
                            node_p.append(arg2)
                        elif grouping_strat == self.LEFT:
                            while pointer + 1 < size and tokens[pointer+1] == token:
                                if isinstance(op_name, str):
                                    node_p[first_index] = [op_name, node_p[first_index], arg2]
                                else:
                                    node_p[first_index] = op_name(node_p[first_index], arg2)
                                tokens.pop(pointer+1)
                                arg2 = tokens.pop(pointer+1)
                                size -= 2
                            node_p.append(arg2)
                        else:
                            node.append(arg2)
                    elif op_type == self.PREFIX:
                        if grouping_strat is not None:
                            raise TypeError("'Prefix' op_type should not have a grouping strat")
                        if pointer == size - 1 or self._is_op(tokens[pointer + 1]):
                            tokens[pointer] = self._missing_arguments_default[token]()
                        else:
                            node.append(tokens.pop(pointer+1))
                            size -= 1
                    elif op_type == self.POSTFIX:
                        if grouping_strat is not None:
                            raise TypeError("'Prefix' op_type should not have a grouping strat")
                        if pointer == 0 or self._is_op(tokens[pointer - 1]):
                            tokens[pointer] = self._missing_arguments_default[token]()
                        else:
                            node.append(tokens.pop(pointer-1))
                            pointer -= 1
                            size -= 1
                    if isinstance(op_name, Callable):  # type: ignore
                        op_call: Callable = typing.cast(Callable, op_name)
                        new_node = op_call(*node)
                        node.clear()
                        if isinstance(new_node, list):
                            node.extend(new_node)
                        else:
                            tokens[pointer] = new_node
                pointer += 1
        if len(tokens) > 1 or (len(lines) == 0 and len(tokens) == 0):
            if changed:
                # Trick to deal with cases in which an operator with lower
                # precedence should be transformed before an operator of higher
                # precedence. Such as in the case of `#&[x]` (that is
                # equivalent to `Lambda(d_, d_)(x)` in SymPy). In this case the
                # operator `&` has lower precedence than `[`, but needs to be
                # evaluated first because otherwise `# (&[x])` is not a valid
                # expression:
                return self._parse_after_braces(tokens, inside_enclosure)
            raise SyntaxError("unable to create a single AST for the expression")
        if len(lines) > 0:
            if tokens[0] and tokens[0][0] == "CompoundExpression":
                tokens = tokens[0][1:]
            compound_expression = ["CompoundExpression", *lines, *tokens]
            return compound_expression
        return tokens[0]

    def _check_op_compatible(self, op1: str, op2: str):
        if op1 == op2:
            return True
        muldiv = {"*", "/"}
        addsub = {"+", "-"}
        if op1 in muldiv and op2 in muldiv:
            return True
        if op1 in addsub and op2 in addsub:
            return True
        return False

    def _from_fullform_to_fullformlist(self, wmexpr: str):
        """
        Parses FullForm[Downvalues[]] generated by Mathematica
        """
        out: list = []
        stack = [out]
        generator = re.finditer(r'[\[\],]', wmexpr)
        last_pos = 0
        for match in generator:
            if match is None:
                break
            position = match.start()
            last_expr = wmexpr[last_pos:position].replace(',', '').replace(']', '').replace('[', '').strip()

            if match.group() == ',':
                if last_expr != '':
                    stack[-1].append(last_expr)
            elif match.group() == ']':
                if last_expr != '':
                    stack[-1].append(last_expr)
                stack.pop()
            elif match.group() == '[':
                stack[-1].append([last_expr])
                stack.append(stack[-1][-1])
            last_pos = match.end()
        return out[0]

    def _from_fullformlist_to_fullformsympy(self, pylist: list):
        from sympy import Function, Symbol

        def converter(expr):
            if isinstance(expr, list):
                if len(expr) > 0:
                    head = expr[0]
                    args = [converter(arg) for arg in expr[1:]]
                    return Function(head)(*args)
                else:
                    raise ValueError("Empty list of expressions")
            elif isinstance(expr, str):
                return Symbol(expr)
            else:
                return _sympify(expr)

        return converter(pylist)

    _node_conversions = {
        "Times": Mul,
        "Plus": Add,
        "Power": Pow,
        "Rational": Rational,
        "Log": lambda *a: log(*reversed(a)),
        "Log2": lambda x: log(x, 2),
        "Log10": lambda x: log(x, 10),
        "Exp": exp,
        "Sqrt": sqrt,

        "Sin": sin,
        "Cos": cos,
        "Tan": tan,
        "Cot": cot,
        "Sec": sec,
        "Csc": csc,

        "ArcSin": asin,
        "ArcCos": acos,
        "ArcTan": lambda *a: atan2(*reversed(a)) if len(a) == 2 else atan(*a),
        "ArcCot": acot,
        "ArcSec": asec,
        "ArcCsc": acsc,

        "Sinh": sinh,
        "Cosh": cosh,
        "Tanh": tanh,
        "Coth": coth,
        "Sech": sech,
        "Csch": csch,

        "ArcSinh": asinh,
        "ArcCosh": acosh,
        "ArcTanh": atanh,
        "ArcCoth": acoth,
        "ArcSech": asech,
        "ArcCsch": acsch,

        "Expand": expand,
        "Im": im,
        "Re": sympy.re,
        "Flatten": flatten,
        "Polylog": polylog,
        "Cancel": cancel,
        # Gamma=gamma,
        "TrigExpand": expand_trig,
        "Sign": sign,
        "Simplify": simplify,
        "Defer": UnevaluatedExpr,
        "Identity": S,
        # Sum=Sum_doit,
        # Module=With,
        # Block=With,
        "Null": lambda *a: S.Zero,
        "Mod": Mod,
        "Max": Max,
        "Min": Min,
        "Pochhammer": rf,
        "ExpIntegralEi": Ei,
        "SinIntegral": Si,
        "CosIntegral": Ci,
        "AiryAi": airyai,
        "AiryAiPrime": airyaiprime,
        "AiryBi": airybi,
        "AiryBiPrime": airybiprime,
        "LogIntegral": li,
        "PrimePi": primepi,
        "Prime": prime,
        "PrimeQ": isprime,

        "List": Tuple,
        "Greater": StrictGreaterThan,
        "GreaterEqual": GreaterThan,
        "Less": StrictLessThan,
        "LessEqual": LessThan,
        "Equal": Equality,
        "Or": Or,
        "And": And,

        "Function": _parse_Function,
    }

    _atom_conversions = {
        "I": I,
        "Pi": pi,
    }

    def _from_fullformlist_to_sympy(self, full_form_list):

        def recurse(expr):
            if isinstance(expr, list):
                if isinstance(expr[0], list):
                    head = recurse(expr[0])
                else:
                    head = self._node_conversions.get(expr[0], Function(expr[0]))
                return head(*[recurse(arg) for arg in expr[1:]])
            else:
                return self._atom_conversions.get(expr, sympify(expr))

        return recurse(full_form_list)

    def _from_fullformsympy_to_sympy(self, mform):

        expr = mform
        for mma_form, sympy_node in self._node_conversions.items():
            expr = expr.replace(Function(mma_form), sympy_node)
        return expr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\cffi\vengine_cpy.py ===
#
# DEPRECATED: implementation for ffi.verify()
#
import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp


class VCPythonEngine(object):
    _class_key = 'x'
    _gen_python_module = True

    def __init__(self, verifier):
        self.verifier = verifier
        self.ffi = verifier.ffi
        self._struct_pending_verification = {}
        self._types_of_builtin_functions = {}

    def patch_extension_kwds(self, kwds):
        pass

    def find_module(self, module_name, path, so_suffixes):
        try:
            f, filename, descr = imp.find_module(module_name, path)
        except ImportError:
            return None
        if f is not None:
            f.close()
        # Note that after a setuptools installation, there are both .py
        # and .so files with the same basename.  The code here relies on
        # imp.find_module() locating the .so in priority.
        if descr[0] not in so_suffixes:
            return None
        return filename

    def collect_types(self):
        self._typesdict = {}
        self._generate("collecttype")

    def _prnt(self, what=''):
        self._f.write(what + '\n')

    def _gettypenum(self, type):
        # a KeyError here is a bug.  please report it! :-)
        return self._typesdict[type]

    def _do_collect_type(self, tp):
        if ((not isinstance(tp, model.PrimitiveType)
             or tp.name == 'long double')
                and tp not in self._typesdict):
            num = len(self._typesdict)
            self._typesdict[tp] = num

    def write_source_to_f(self):
        self.collect_types()
        #
        # The new module will have a _cffi_setup() function that receives
        # objects from the ffi world, and that calls some setup code in
        # the module.  This setup code is split in several independent
        # functions, e.g. one per constant.  The functions are "chained"
        # by ending in a tail call to each other.
        #
        # This is further split in two chained lists, depending on if we
        # can do it at import-time or if we must wait for _cffi_setup() to
        # provide us with the <ctype> objects.  This is needed because we
        # need the values of the enum constants in order to build the
        # <ctype 'enum'> that we may have to pass to _cffi_setup().
        #
        # The following two 'chained_list_constants' items contains
        # the head of these two chained lists, as a string that gives the
        # call to do, if any.
        self._chained_list_constants = ['((void)lib,0)', '((void)lib,0)']
        #
        prnt = self._prnt
        # first paste some standard set of lines that are mostly '#define'
        prnt(cffimod_header)
        prnt()
        # then paste the C source given by the user, verbatim.
        prnt(self.verifier.preamble)
        prnt()
        #
        # call generate_cpy_xxx_decl(), for every xxx found from
        # ffi._parser._declarations.  This generates all the functions.
        self._generate("decl")
        #
        # implement the function _cffi_setup_custom() as calling the
        # head of the chained list.
        self._generate_setup_custom()
        prnt()
        #
        # produce the method table, including the entries for the
        # generated Python->C function wrappers, which are done
        # by generate_cpy_function_method().
        prnt('static PyMethodDef _cffi_methods[] = {')
        self._generate("method")
        prnt('  {"_cffi_setup", _cffi_setup, METH_VARARGS, NULL},')
        prnt('  {NULL, NULL, 0, NULL}    /* Sentinel */')
        prnt('};')
        prnt()
        #
        # standard init.
        modname = self.verifier.get_module_name()
        constants = self._chained_list_constants[False]
        prnt('#if PY_MAJOR_VERSION >= 3')
        prnt()
        prnt('static struct PyModuleDef _cffi_module_def = {')
        prnt('  PyModuleDef_HEAD_INIT,')
        prnt('  "%s",' % modname)
        prnt('  NULL,')
        prnt('  -1,')
        prnt('  _cffi_methods,')
        prnt('  NULL, NULL, NULL, NULL')
        prnt('};')
        prnt()
        prnt('PyMODINIT_FUNC')
        prnt('PyInit_%s(void)' % modname)
        prnt('{')
        prnt('  PyObject *lib;')
        prnt('  lib = PyModule_Create(&_cffi_module_def);')
        prnt('  if (lib == NULL)')
        prnt('    return NULL;')
        prnt('  if (%s < 0 || _cffi_init() < 0) {' % (constants,))
        prnt('    Py_DECREF(lib);')
        prnt('    return NULL;')
        prnt('  }')
        prnt('  return lib;')
        prnt('}')
        prnt()
        prnt('#else')
        prnt()
        prnt('PyMODINIT_FUNC')
        prnt('init%s(void)' % modname)
        prnt('{')
        prnt('  PyObject *lib;')
        prnt('  lib = Py_InitModule("%s", _cffi_methods);' % modname)
        prnt('  if (lib == NULL)')
        prnt('    return;')
        prnt('  if (%s < 0 || _cffi_init() < 0)' % (constants,))
        prnt('    return;')
        prnt('  return;')
        prnt('}')
        prnt()
        prnt('#endif')

    def load_library(self, flags=None):
        # XXX review all usages of 'self' here!
        # import it as a new extension module
        imp.acquire_lock()
        try:
            if hasattr(sys, "getdlopenflags"):
                previous_flags = sys.getdlopenflags()
            try:
                if hasattr(sys, "setdlopenflags") and flags is not None:
                    sys.setdlopenflags(flags)
                module = imp.load_dynamic(self.verifier.get_module_name(),
                                          self.verifier.modulefilename)
            except ImportError as e:
                error = "importing %r: %s" % (self.verifier.modulefilename, e)
                raise VerificationError(error)
            finally:
                if hasattr(sys, "setdlopenflags"):
                    sys.setdlopenflags(previous_flags)
        finally:
            imp.release_lock()
        #
        # call loading_cpy_struct() to get the struct layout inferred by
        # the C compiler
        self._load(module, 'loading')
        #
        # the C code will need the <ctype> objects.  Collect them in
        # order in a list.
        revmapping = dict([(value, key)
                           for (key, value) in self._typesdict.items()])
        lst = [revmapping[i] for i in range(len(revmapping))]
        lst = list(map(self.ffi._get_cached_btype, lst))
        #
        # build the FFILibrary class and instance and call _cffi_setup().
        # this will set up some fields like '_cffi_types', and only then
        # it will invoke the chained list of functions that will really
        # build (notably) the constant objects, as <cdata> if they are
        # pointers, and store them as attributes on the 'library' object.
        class FFILibrary(object):
            _cffi_python_module = module
            _cffi_ffi = self.ffi
            _cffi_dir = []
            def __dir__(self):
                return FFILibrary._cffi_dir + list(self.__dict__)
        library = FFILibrary()
        if module._cffi_setup(lst, VerificationError, library):
            import warnings
            warnings.warn("reimporting %r might overwrite older definitions"
                          % (self.verifier.get_module_name()))
        #
        # finally, call the loaded_cpy_xxx() functions.  This will perform
        # the final adjustments, like copying the Python->C wrapper
        # functions from the module to the 'library' object, and setting
        # up the FFILibrary class with properties for the global C variables.
        self._load(module, 'loaded', library=library)
        module._cffi_original_ffi = self.ffi
        module._cffi_types_of_builtin_funcs = self._types_of_builtin_functions
        return library

    def _get_declarations(self):
        lst = [(key, tp) for (key, (tp, qual)) in
                                self.ffi._parser._declarations.items()]
        lst.sort()
        return lst

    def _generate(self, step_name):
        for name, tp in self._get_declarations():
            kind, realname = name.split(' ', 1)
            try:
                method = getattr(self, '_generate_cpy_%s_%s' % (kind,
                                                                step_name))
            except AttributeError:
                raise VerificationError(
                    "not implemented in verify(): %r" % name)
            try:
                method(tp, realname)
            except Exception as e:
                model.attach_exception_info(e, name)
                raise

    def _load(self, module, step_name, **kwds):
        for name, tp in self._get_declarations():
            kind, realname = name.split(' ', 1)
            method = getattr(self, '_%s_cpy_%s' % (step_name, kind))
            try:
                method(tp, realname, module, **kwds)
            except Exception as e:
                model.attach_exception_info(e, name)
                raise

    def _generate_nothing(self, tp, name):
        pass

    def _loaded_noop(self, tp, name, module, **kwds):
        pass

    # ----------

    def _convert_funcarg_to_c(self, tp, fromvar, tovar, errcode):
        extraarg = ''
        if isinstance(tp, model.PrimitiveType):
            if tp.is_integer_type() and tp.name != '_Bool':
                converter = '_cffi_to_c_int'
                extraarg = ', %s' % tp.name
            elif tp.is_complex_type():
                raise VerificationError(
                    "not implemented in verify(): complex types")
            else:
                converter = '(%s)_cffi_to_c_%s' % (tp.get_c_name(''),
                                                   tp.name.replace(' ', '_'))
            errvalue = '-1'
        #
        elif isinstance(tp, model.PointerType):
            self._convert_funcarg_to_c_ptr_or_array(tp, fromvar,
                                                    tovar, errcode)
            return
        #
        elif isinstance(tp, (model.StructOrUnion, model.EnumType)):
            # a struct (not a struct pointer) as a function argument
            self._prnt('  if (_cffi_to_c((char *)&%s, _cffi_type(%d), %s) < 0)'
                      % (tovar, self._gettypenum(tp), fromvar))
            self._prnt('    %s;' % errcode)
            return
        #
        elif isinstance(tp, model.FunctionPtrType):
            converter = '(%s)_cffi_to_c_pointer' % tp.get_c_name('')
            extraarg = ', _cffi_type(%d)' % self._gettypenum(tp)
            errvalue = 'NULL'
        #
        else:
            raise NotImplementedError(tp)
        #
        self._prnt('  %s = %s(%s%s);' % (tovar, converter, fromvar, extraarg))
        self._prnt('  if (%s == (%s)%s && PyErr_Occurred())' % (
            tovar, tp.get_c_name(''), errvalue))
        self._prnt('    %s;' % errcode)

    def _extra_local_variables(self, tp, localvars, freelines):
        if isinstance(tp, model.PointerType):
            localvars.add('Py_ssize_t datasize')
            localvars.add('struct _cffi_freeme_s *large_args_free = NULL')
            freelines.add('if (large_args_free != NULL)'
                          ' _cffi_free_array_arguments(large_args_free);')

    def _convert_funcarg_to_c_ptr_or_array(self, tp, fromvar, tovar, errcode):
        self._prnt('  datasize = _cffi_prepare_pointer_call_argument(')
        self._prnt('      _cffi_type(%d), %s, (char **)&%s);' % (
            self._gettypenum(tp), fromvar, tovar))
        self._prnt('  if (datasize != 0) {')
        self._prnt('    %s = ((size_t)datasize) <= 640 ? '
                   'alloca((size_t)datasize) : NULL;' % (tovar,))
        self._prnt('    if (_cffi_convert_array_argument(_cffi_type(%d), %s, '
                   '(char **)&%s,' % (self._gettypenum(tp), fromvar, tovar))
        self._prnt('            datasize, &large_args_free) < 0)')
        self._prnt('      %s;' % errcode)
        self._prnt('  }')

    def _convert_expr_from_c(self, tp, var, context):
        if isinstance(tp, model.PrimitiveType):
            if tp.is_integer_type() and tp.name != '_Bool':
                return '_cffi_from_c_int(%s, %s)' % (var, tp.name)
            elif tp.name != 'long double':
                return '_cffi_from_c_%s(%s)' % (tp.name.replace(' ', '_'), var)
            else:
                return '_cffi_from_c_deref((char *)&%s, _cffi_type(%d))' % (
                    var, self._gettypenum(tp))
        elif isinstance(tp, (model.PointerType, model.FunctionPtrType)):
            return '_cffi_from_c_pointer((char *)%s, _cffi_type(%d))' % (
                var, self._gettypenum(tp))
        elif isinstance(tp, model.ArrayType):
            return '_cffi_from_c_pointer((char *)%s, _cffi_type(%d))' % (
                var, self._gettypenum(model.PointerType(tp.item)))
        elif isinstance(tp, model.StructOrUnion):
            if tp.fldnames is None:
                raise TypeError("'%s' is used as %s, but is opaque" % (
                    tp._get_c_name(), context))
            return '_cffi_from_c_struct((char *)&%s, _cffi_type(%d))' % (
                var, self._gettypenum(tp))
        elif isinstance(tp, model.EnumType):
            return '_cffi_from_c_deref((char *)&%s, _cffi_type(%d))' % (
                var, self._gettypenum(tp))
        else:
            raise NotImplementedError(tp)

    # ----------
    # typedefs: generates no code so far

    _generate_cpy_typedef_collecttype = _generate_nothing
    _generate_cpy_typedef_decl   = _generate_nothing
    _generate_cpy_typedef_method = _generate_nothing
    _loading_cpy_typedef         = _loaded_noop
    _loaded_cpy_typedef          = _loaded_noop

    # ----------
    # function declarations

    def _generate_cpy_function_collecttype(self, tp, name):
        assert isinstance(tp, model.FunctionPtrType)
        if tp.ellipsis:
            self._do_collect_type(tp)
        else:
            # don't call _do_collect_type(tp) in this common case,
            # otherwise test_autofilled_struct_as_argument fails
            for type in tp.args:
                self._do_collect_type(type)
            self._do_collect_type(tp.result)

    def _generate_cpy_function_decl(self, tp, name):
        assert isinstance(tp, model.FunctionPtrType)
        if tp.ellipsis:
            # cannot support vararg functions better than this: check for its
            # exact type (including the fixed arguments), and build it as a
            # constant function pointer (no CPython wrapper)
            self._generate_cpy_const(False, name, tp)
            return
        prnt = self._prnt
        numargs = len(tp.args)
        if numargs == 0:
            argname = 'noarg'
        elif numargs == 1:
            argname = 'arg0'
        else:
            argname = 'args'
        prnt('static PyObject *')
        prnt('_cffi_f_%s(PyObject *self, PyObject *%s)' % (name, argname))
        prnt('{')
        #
        context = 'argument of %s' % name
        for i, type in enumerate(tp.args):
            prnt('  %s;' % type.get_c_name(' x%d' % i, context))
        #
        localvars = set()
        freelines = set()
        for type in tp.args:
            self._extra_local_variables(type, localvars, freelines)
        for decl in sorted(localvars):
            prnt('  %s;' % (decl,))
        #
        if not isinstance(tp.result, model.VoidType):
            result_code = 'result = '
            context = 'result of %s' % name
            prnt('  %s;' % tp.result.get_c_name(' result', context))
            prnt('  PyObject *pyresult;')
        else:
            result_code = ''
        #
        if len(tp.args) > 1:
            rng = range(len(tp.args))
            for i in rng:
                prnt('  PyObject *arg%d;' % i)
            prnt()
            prnt('  if (!PyArg_ParseTuple(args, "%s:%s", %s))' % (
                'O' * numargs, name, ', '.join(['&arg%d' % i for i in rng])))
            prnt('    return NULL;')
        prnt()
        #
        for i, type in enumerate(tp.args):
            self._convert_funcarg_to_c(type, 'arg%d' % i, 'x%d' % i,
                                       'return NULL')
            prnt()
        #
        prnt('  Py_BEGIN_ALLOW_THREADS')
        prnt('  _cffi_restore_errno();')
        prnt('  { %s%s(%s); }' % (
            result_code, name,
            ', '.join(['x%d' % i for i in range(len(tp.args))])))
        prnt('  _cffi_save_errno();')
        prnt('  Py_END_ALLOW_THREADS')
        prnt()
        #
        prnt('  (void)self; /* unused */')
        if numargs == 0:
            prnt('  (void)noarg; /* unused */')
        if result_code:
            prnt('  pyresult = %s;' %
                 self._convert_expr_from_c(tp.result, 'result', 'result type'))
            for freeline in freelines:
                prnt('  ' + freeline)
            prnt('  return pyresult;')
        else:
            for freeline in freelines:
                prnt('  ' + freeline)
            prnt('  Py_INCREF(Py_None);')
            prnt('  return Py_None;')
        prnt('}')
        prnt()

    def _generate_cpy_function_method(self, tp, name):
        if tp.ellipsis:
            return
        numargs = len(tp.args)
        if numargs == 0:
            meth = 'METH_NOARGS'
        elif numargs == 1:
            meth = 'METH_O'
        else:
            meth = 'METH_VARARGS'
        self._prnt('  {"%s", _cffi_f_%s, %s, NULL},' % (name, name, meth))

    _loading_cpy_function = _loaded_noop

    def _loaded_cpy_function(self, tp, name, module, library):
        if tp.ellipsis:
            return
        func = getattr(module, name)
        setattr(library, name, func)
        self._types_of_builtin_functions[func] = tp

    # ----------
    # named structs

    _generate_cpy_struct_collecttype = _generate_nothing
    def _generate_cpy_struct_decl(self, tp, name):
        assert name == tp.name
        self._generate_struct_or_union_decl(tp, 'struct', name)
    def _generate_cpy_struct_method(self, tp, name):
        self._generate_struct_or_union_method(tp, 'struct', name)
    def _loading_cpy_struct(self, tp, name, module):
        self._loading_struct_or_union(tp, 'struct', name, module)
    def _loaded_cpy_struct(self, tp, name, module, **kwds):
        self._loaded_struct_or_union(tp)

    _generate_cpy_union_collecttype = _generate_nothing
    def _generate_cpy_union_decl(self, tp, name):
        assert name == tp.name
        self._generate_struct_or_union_decl(tp, 'union', name)
    def _generate_cpy_union_method(self, tp, name):
        self._generate_struct_or_union_method(tp, 'union', name)
    def _loading_cpy_union(self, tp, name, module):
        self._loading_struct_or_union(tp, 'union', name, module)
    def _loaded_cpy_union(self, tp, name, module, **kwds):
        self._loaded_struct_or_union(tp)

    def _generate_struct_or_union_decl(self, tp, prefix, name):
        if tp.fldnames is None:
            return     # nothing to do with opaque structs
        checkfuncname = '_cffi_check_%s_%s' % (prefix, name)
        layoutfuncname = '_cffi_layout_%s_%s' % (prefix, name)
        cname = ('%s %s' % (prefix, name)).strip()
        #
        prnt = self._prnt
        prnt('static void %s(%s *p)' % (checkfuncname, cname))
        prnt('{')
        prnt('  /* only to generate compile-time warnings or errors */')
        prnt('  (void)p;')
        for fname, ftype, fbitsize, fqual in tp.enumfields():
            if (isinstance(ftype, model.PrimitiveType)
                and ftype.is_integer_type()) or fbitsize >= 0:
                # accept all integers, but complain on float or double
                prnt('  (void)((p->%s) << 1);' % fname)
            else:
                # only accept exactly the type declared.
                try:
                    prnt('  { %s = &p->%s; (void)tmp; }' % (
                        ftype.get_c_name('*tmp', 'field %r'%fname, quals=fqual),
                        fname))
                except VerificationError as e:
                    prnt('  /* %s */' % str(e))   # cannot verify it, ignore
        prnt('}')
        prnt('static PyObject *')
        prnt('%s(PyObject *self, PyObject *noarg)' % (layoutfuncname,))
        prnt('{')
        prnt('  struct _cffi_aligncheck { char x; %s y; };' % cname)
        prnt('  static Py_ssize_t nums[] = {')
        prnt('    sizeof(%s),' % cname)
        prnt('    offsetof(struct _cffi_aligncheck, y),')
        for fname, ftype, fbitsize, fqual in tp.enumfields():
            if fbitsize >= 0:
                continue      # xxx ignore fbitsize for now
            prnt('    offsetof(%s, %s),' % (cname, fname))
            if isinstance(ftype, model.ArrayType) and ftype.length is None:
                prnt('    0,  /* %s */' % ftype._get_c_name())
            else:
                prnt('    sizeof(((%s *)0)->%s),' % (cname, fname))
        prnt('    -1')
        prnt('  };')
        prnt('  (void)self; /* unused */')
        prnt('  (void)noarg; /* unused */')
        prnt('  return _cffi_get_struct_layout(nums);')
        prnt('  /* the next line is not executed, but compiled */')
        prnt('  %s(0);' % (checkfuncname,))
        prnt('}')
        prnt()

    def _generate_struct_or_union_method(self, tp, prefix, name):
        if tp.fldnames is None:
            return     # nothing to do with opaque structs
        layoutfuncname = '_cffi_layout_%s_%s' % (prefix, name)
        self._prnt('  {"%s", %s, METH_NOARGS, NULL},' % (layoutfuncname,
                                                         layoutfuncname))

    def _loading_struct_or_union(self, tp, prefix, name, module):
        if tp.fldnames is None:
            return     # nothing to do with opaque structs
        layoutfuncname = '_cffi_layout_%s_%s' % (prefix, name)
        #
        function = getattr(module, layoutfuncname)
        layout = function()
        if isinstance(tp, model.StructOrUnion) and tp.partial:
            # use the function()'s sizes and offsets to guide the
            # layout of the struct
            totalsize = layout[0]
            totalalignment = layout[1]
            fieldofs = layout[2::2]
            fieldsize = layout[3::2]
            tp.force_flatten()
            assert len(fieldofs) == len(fieldsize) == len(tp.fldnames)
            tp.fixedlayout = fieldofs, fieldsize, totalsize, totalalignment
        else:
            cname = ('%s %s' % (prefix, name)).strip()
            self._struct_pending_verification[tp] = layout, cname

    def _loaded_struct_or_union(self, tp):
        if tp.fldnames is None:
            return     # nothing to do with opaque structs
        self.ffi._get_cached_btype(tp)   # force 'fixedlayout' to be considered

        if tp in self._struct_pending_verification:
            # check that the layout sizes and offsets match the real ones
            def check(realvalue, expectedvalue, msg):
                if realvalue != expectedvalue:
                    raise VerificationError(
                        "%s (we have %d, but C compiler says %d)"
                        % (msg, expectedvalue, realvalue))
            ffi = self.ffi
            BStruct = ffi._get_cached_btype(tp)
            layout, cname = self._struct_pending_verification.pop(tp)
            check(layout[0], ffi.sizeof(BStruct), "wrong total size")
            check(layout[1], ffi.alignof(BStruct), "wrong total alignment")
            i = 2
            for fname, ftype, fbitsize, fqual in tp.enumfields():
                if fbitsize >= 0:
                    continue        # xxx ignore fbitsize for now
                check(layout[i], ffi.offsetof(BStruct, fname),
                      "wrong offset for field %r" % (fname,))
                if layout[i+1] != 0:
                    BField = ffi._get_cached_btype(ftype)
                    check(layout[i+1], ffi.sizeof(BField),
                          "wrong size for field %r" % (fname,))
                i += 2
            assert i == len(layout)

    # ----------
    # 'anonymous' declarations.  These are produced for anonymous structs
    # or unions; the 'name' is obtained by a typedef.

    _generate_cpy_anonymous_collecttype = _generate_nothing

    def _generate_cpy_anonymous_decl(self, tp, name):
        if isinstance(tp, model.EnumType):
            self._generate_cpy_enum_decl(tp, name, '')
        else:
            self._generate_struct_or_union_decl(tp, '', name)

    def _generate_cpy_anonymous_method(self, tp, name):
        if not isinstance(tp, model.EnumType):
            self._generate_struct_or_union_method(tp, '', name)

    def _loading_cpy_anonymous(self, tp, name, module):
        if isinstance(tp, model.EnumType):
            self._loading_cpy_enum(tp, name, module)
        else:
            self._loading_struct_or_union(tp, '', name, module)

    def _loaded_cpy_anonymous(self, tp, name, module, **kwds):
        if isinstance(tp, model.EnumType):
            self._loaded_cpy_enum(tp, name, module, **kwds)
        else:
            self._loaded_struct_or_union(tp)

    # ----------
    # constants, likely declared with '#define'

    def _generate_cpy_const(self, is_int, name, tp=None, category='const',
                            vartp=None, delayed=True, size_too=False,
                            check_value=None):
        prnt = self._prnt
        funcname = '_cffi_%s_%s' % (category, name)
        prnt('static int %s(PyObject *lib)' % funcname)
        prnt('{')
        prnt('  PyObject *o;')
        prnt('  int res;')
        if not is_int:
            prnt('  %s;' % (vartp or tp).get_c_name(' i', name))
        else:
            assert category == 'const'
        #
        if check_value is not None:
            self._check_int_constant_value(name, check_value)
        #
        if not is_int:
            if category == 'var':
                realexpr = '&' + name
            else:
                realexpr = name
            prnt('  i = (%s);' % (realexpr,))
            prnt('  o = %s;' % (self._convert_expr_from_c(tp, 'i',
                                                          'variable type'),))
            assert delayed
        else:
            prnt('  o = _cffi_from_c_int_const(%s);' % name)
        prnt('  if (o == NULL)')
        prnt('    return -1;')
        if size_too:
            prnt('  {')
            prnt('    PyObject *o1 = o;')
            prnt('    o = Py_BuildValue("On", o1, (Py_ssize_t)sizeof(%s));'
                 % (name,))
            prnt('    Py_DECREF(o1);')
            prnt('    if (o == NULL)')
            prnt('      return -1;')
            prnt('  }')
        prnt('  res = PyObject_SetAttrString(lib, "%s", o);' % name)
        prnt('  Py_DECREF(o);')
        prnt('  if (res < 0)')
        prnt('    return -1;')
        prnt('  return %s;' % self._chained_list_constants[delayed])
        self._chained_list_constants[delayed] = funcname + '(lib)'
        prnt('}')
        prnt()

    def _generate_cpy_constant_collecttype(self, tp, name):
        is_int = isinstance(tp, model.PrimitiveType) and tp.is_integer_type()
        if not is_int:
            self._do_collect_type(tp)

    def _generate_cpy_constant_decl(self, tp, name):
        is_int = isinstance(tp, model.PrimitiveType) and tp.is_integer_type()
        self._generate_cpy_const(is_int, name, tp)

    _generate_cpy_constant_method = _generate_nothing
    _loading_cpy_constant = _loaded_noop
    _loaded_cpy_constant  = _loaded_noop

    # ----------
    # enums

    def _check_int_constant_value(self, name, value, err_prefix=''):
        prnt = self._prnt
        if value <= 0:
            prnt('  if ((%s) > 0 || (long)(%s) != %dL) {' % (
                name, name, value))
        else:
            prnt('  if ((%s) <= 0 || (unsigned long)(%s) != %dUL) {' % (
                name, name, value))
        prnt('    char buf[64];')
        prnt('    if ((%s) <= 0)' % name)
        prnt('        snprintf(buf, 63, "%%ld", (long)(%s));' % name)
        prnt('    else')
        prnt('        snprintf(buf, 63, "%%lu", (unsigned long)(%s));' %
             name)
        prnt('    PyErr_Format(_cffi_VerificationError,')
        prnt('                 "%s%s has the real value %s, not %s",')
        prnt('                 "%s", "%s", buf, "%d");' % (
            err_prefix, name, value))
        prnt('    return -1;')
        prnt('  }')

    def _enum_funcname(self, prefix, name):
        # "$enum_$1" => "___D_enum____D_1"
        name = name.replace('$', '___D_')
        return '_cffi_e_%s_%s' % (prefix, name)

    def _generate_cpy_enum_decl(self, tp, name, prefix='enum'):
        if tp.partial:
            for enumerator in tp.enumerators:
                self._generate_cpy_const(True, enumerator, delayed=False)
            return
        #
        funcname = self._enum_funcname(prefix, name)
        prnt = self._prnt
        prnt('static int %s(PyObject *lib)' % funcname)
        prnt('{')
        for enumerator, enumvalue in zip(tp.enumerators, tp.enumvalues):
            self._check_int_constant_value(enumerator, enumvalue,
                                           "enum %s: " % name)
        prnt('  return %s;' % self._chained_list_constants[True])
        self._chained_list_constants[True] = funcname + '(lib)'
        prnt('}')
        prnt()

    _generate_cpy_enum_collecttype = _generate_nothing
    _generate_cpy_enum_method = _generate_nothing

    def _loading_cpy_enum(self, tp, name, module):
        if tp.partial:
            enumvalues = [getattr(module, enumerator)
                          for enumerator in tp.enumerators]
            tp.enumvalues = tuple(enumvalues)
            tp.partial_resolved = True

    def _loaded_cpy_enum(self, tp, name, module, library):
        for enumerator, enumvalue in zip(tp.enumerators, tp.enumvalues):
            setattr(library, enumerator, enumvalue)

    # ----------
    # macros: for now only for integers

    def _generate_cpy_macro_decl(self, tp, name):
        if tp == '...':
            check_value = None
        else:
            check_value = tp     # an integer
        self._generate_cpy_const(True, name, check_value=check_value)

    _generate_cpy_macro_collecttype = _generate_nothing
    _generate_cpy_macro_method = _generate_nothing
    _loading_cpy_macro = _loaded_noop
    _loaded_cpy_macro  = _loaded_noop

    # ----------
    # global variables

    def _generate_cpy_variable_collecttype(self, tp, name):
        if isinstance(tp, model.ArrayType):
            tp_ptr = model.PointerType(tp.item)
        else:
            tp_ptr = model.PointerType(tp)
        self._do_collect_type(tp_ptr)

    def _generate_cpy_variable_decl(self, tp, name):
        if isinstance(tp, model.ArrayType):
            tp_ptr = model.PointerType(tp.item)
            self._generate_cpy_const(False, name, tp, vartp=tp_ptr,
                                     size_too = tp.length_is_unknown())
        else:
            tp_ptr = model.PointerType(tp)
            self._generate_cpy_const(False, name, tp_ptr, category='var')

    _generate_cpy_variable_method = _generate_nothing
    _loading_cpy_variable = _loaded_noop

    def _loaded_cpy_variable(self, tp, name, module, library):
        value = getattr(library, name)
        if isinstance(tp, model.ArrayType):   # int a[5] is "constant" in the
                                              # sense that "a=..." is forbidden
            if tp.length_is_unknown():
                assert isinstance(value, tuple)
                (value, size) = value
                BItemType = self.ffi._get_cached_btype(tp.item)
                length, rest = divmod(size, self.ffi.sizeof(BItemType))
                if rest != 0:
                    raise VerificationError(
                        "bad size: %r does not seem to be an array of %s" %
                        (name, tp.item))
                tp = tp.resolve_length(length)
            # 'value' is a <cdata 'type *'> which we have to replace with
            # a <cdata 'type[N]'> if the N is actually known
            if tp.length is not None:
                BArray = self.ffi._get_cached_btype(tp)
                value = self.ffi.cast(BArray, value)
                setattr(library, name, value)
            return
        # remove ptr=<cdata 'int *'> from the library instance, and replace
        # it by a property on the class, which reads/writes into ptr[0].
        ptr = value
        delattr(library, name)
        def getter(library):
            return ptr[0]
        def setter(library, value):
            ptr[0] = value
        setattr(type(library), name, property(getter, setter))
        type(library)._cffi_dir.append(name)

    # ----------

    def _generate_setup_custom(self):
        prnt = self._prnt
        prnt('static int _cffi_setup_custom(PyObject *lib)')
        prnt('{')
        prnt('  return %s;' % self._chained_list_constants[True])
        prnt('}')

cffimod_header = r'''
#include <Python.h>
#include <stddef.h>

/* this block of #ifs should be kept exactly identical between
   c/_cffi_backend.c, cffi/vengine_cpy.py, cffi/vengine_gen.py
   and cffi/_cffi_include.h */
#if defined(_MSC_VER)
# include <malloc.h>   /* for alloca() */
# if _MSC_VER < 1600   /* MSVC < 2010 */
   typedef __int8 int8_t;
   typedef __int16 int16_t;
   typedef __int32 int32_t;
   typedef __int64 int64_t;
   typedef unsigned __int8 uint8_t;
   typedef unsigned __int16 uint16_t;
   typedef unsigned __int32 uint32_t;
   typedef unsigned __int64 uint64_t;
   typedef __int8 int_least8_t;
   typedef __int16 int_least16_t;
   typedef __int32 int_least32_t;
   typedef __int64 int_least64_t;
   typedef unsigned __int8 uint_least8_t;
   typedef unsigned __int16 uint_least16_t;
   typedef unsigned __int32 uint_least32_t;
   typedef unsigned __int64 uint_least64_t;
   typedef __int8 int_fast8_t;
   typedef __int16 int_fast16_t;
   typedef __int32 int_fast32_t;
   typedef __int64 int_fast64_t;
   typedef unsigned __int8 uint_fast8_t;
   typedef unsigned __int16 uint_fast16_t;
   typedef unsigned __int32 uint_fast32_t;
   typedef unsigned __int64 uint_fast64_t;
   typedef __int64 intmax_t;
   typedef unsigned __int64 uintmax_t;
# else
#  include <stdint.h>
# endif
# if _MSC_VER < 1800   /* MSVC < 2013 */
#  ifndef __cplusplus
    typedef unsigned char _Bool;
#  endif
# endif
# define _cffi_float_complex_t   _Fcomplex    /* include <complex.h> for it */
# define _cffi_double_complex_t  _Dcomplex    /* include <complex.h> for it */
#else
# include <stdint.h>
# if (defined (__SVR4) && defined (__sun)) || defined(_AIX) || defined(__hpux)
#  include <alloca.h>
# endif
# define _cffi_float_complex_t   float _Complex
# define _cffi_double_complex_t  double _Complex
#endif

#if PY_MAJOR_VERSION < 3
# undef PyCapsule_CheckExact
# undef PyCapsule_GetPointer
# define PyCapsule_CheckExact(capsule) (PyCObject_Check(capsule))
# define PyCapsule_GetPointer(capsule, name) \
    (PyCObject_AsVoidPtr(capsule))
#endif

#if PY_MAJOR_VERSION >= 3
# define PyInt_FromLong PyLong_FromLong
#endif

#define _cffi_from_c_double PyFloat_FromDouble
#define _cffi_from_c_float PyFloat_FromDouble
#define _cffi_from_c_long PyInt_FromLong
#define _cffi_from_c_ulong PyLong_FromUnsignedLong
#define _cffi_from_c_longlong PyLong_FromLongLong
#define _cffi_from_c_ulonglong PyLong_FromUnsignedLongLong
#define _cffi_from_c__Bool PyBool_FromLong

#define _cffi_to_c_double PyFloat_AsDouble
#define _cffi_to_c_float PyFloat_AsDouble

#define _cffi_from_c_int_const(x)                                        \
    (((x) > 0) ?                                                         \
        ((unsigned long long)(x) <= (unsigned long long)LONG_MAX) ?      \
            PyInt_FromLong((long)(x)) :                                  \
            PyLong_FromUnsignedLongLong((unsigned long long)(x)) :       \
        ((long long)(x) >= (long long)LONG_MIN) ?                        \
            PyInt_FromLong((long)(x)) :                                  \
            PyLong_FromLongLong((long long)(x)))

#define _cffi_from_c_int(x, type)                                        \
    (((type)-1) > 0 ? /* unsigned */                                     \
        (sizeof(type) < sizeof(long) ?                                   \
            PyInt_FromLong((long)x) :                                    \
         sizeof(type) == sizeof(long) ?                                  \
            PyLong_FromUnsignedLong((unsigned long)x) :                  \
            PyLong_FromUnsignedLongLong((unsigned long long)x)) :        \
        (sizeof(type) <= sizeof(long) ?                                  \
            PyInt_FromLong((long)x) :                                    \
            PyLong_FromLongLong((long long)x)))

#define _cffi_to_c_int(o, type)                                          \
    ((type)(                                                             \
     sizeof(type) == 1 ? (((type)-1) > 0 ? (type)_cffi_to_c_u8(o)        \
                                         : (type)_cffi_to_c_i8(o)) :     \
     sizeof(type) == 2 ? (((type)-1) > 0 ? (type)_cffi_to_c_u16(o)       \
                                         : (type)_cffi_to_c_i16(o)) :    \
     sizeof(type) == 4 ? (((type)-1) > 0 ? (type)_cffi_to_c_u32(o)       \
                                         : (type)_cffi_to_c_i32(o)) :    \
     sizeof(type) == 8 ? (((type)-1) > 0 ? (type)_cffi_to_c_u64(o)       \
                                         : (type)_cffi_to_c_i64(o)) :    \
     (Py_FatalError("unsupported size for type " #type), (type)0)))

#define _cffi_to_c_i8                                                    \
                 ((int(*)(PyObject *))_cffi_exports[1])
#define _cffi_to_c_u8                                                    \
                 ((int(*)(PyObject *))_cffi_exports[2])
#define _cffi_to_c_i16                                                   \
                 ((int(*)(PyObject *))_cffi_exports[3])
#define _cffi_to_c_u16                                                   \
                 ((int(*)(PyObject *))_cffi_exports[4])
#define _cffi_to_c_i32                                                   \
                 ((int(*)(PyObject *))_cffi_exports[5])
#define _cffi_to_c_u32                                                   \
                 ((unsigned int(*)(PyObject *))_cffi_exports[6])
#define _cffi_to_c_i64                                                   \
                 ((long long(*)(PyObject *))_cffi_exports[7])
#define _cffi_to_c_u64                                                   \
                 ((unsigned long long(*)(PyObject *))_cffi_exports[8])
#define _cffi_to_c_char                                                  \
                 ((int(*)(PyObject *))_cffi_exports[9])
#define _cffi_from_c_pointer                                             \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[10])
#define _cffi_to_c_pointer                                               \
    ((char *(*)(PyObject *, CTypeDescrObject *))_cffi_exports[11])
#define _cffi_get_struct_layout                                          \
    ((PyObject *(*)(Py_ssize_t[]))_cffi_exports[12])
#define _cffi_restore_errno                                              \
    ((void(*)(void))_cffi_exports[13])
#define _cffi_save_errno                                                 \
    ((void(*)(void))_cffi_exports[14])
#define _cffi_from_c_char                                                \
    ((PyObject *(*)(char))_cffi_exports[15])
#define _cffi_from_c_deref                                               \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[16])
#define _cffi_to_c                                                       \
    ((int(*)(char *, CTypeDescrObject *, PyObject *))_cffi_exports[17])
#define _cffi_from_c_struct                                              \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[18])
#define _cffi_to_c_wchar_t                                               \
    ((wchar_t(*)(PyObject *))_cffi_exports[19])
#define _cffi_from_c_wchar_t                                             \
    ((PyObject *(*)(wchar_t))_cffi_exports[20])
#define _cffi_to_c_long_double                                           \
    ((long double(*)(PyObject *))_cffi_exports[21])
#define _cffi_to_c__Bool                                                 \
    ((_Bool(*)(PyObject *))_cffi_exports[22])
#define _cffi_prepare_pointer_call_argument                              \
    ((Py_ssize_t(*)(CTypeDescrObject *, PyObject *, char **))_cffi_exports[23])
#define _cffi_convert_array_from_object                                  \
    ((int(*)(char *, CTypeDescrObject *, PyObject *))_cffi_exports[24])
#define _CFFI_NUM_EXPORTS 25

typedef struct _ctypedescr CTypeDescrObject;

static void *_cffi_exports[_CFFI_NUM_EXPORTS];
static PyObject *_cffi_types, *_cffi_VerificationError;

static int _cffi_setup_custom(PyObject *lib);   /* forward */

static PyObject *_cffi_setup(PyObject *self, PyObject *args)
{
    PyObject *library;
    int was_alive = (_cffi_types != NULL);
    (void)self; /* unused */
    if (!PyArg_ParseTuple(args, "OOO", &_cffi_types, &_cffi_VerificationError,
                                       &library))
        return NULL;
    Py_INCREF(_cffi_types);
    Py_INCREF(_cffi_VerificationError);
    if (_cffi_setup_custom(library) < 0)
        return NULL;
    return PyBool_FromLong(was_alive);
}

union _cffi_union_alignment_u {
    unsigned char m_char;
    unsigned short m_short;
    unsigned int m_int;
    unsigned long m_long;
    unsigned long long m_longlong;
    float m_float;
    double m_double;
    long double m_longdouble;
};

struct _cffi_freeme_s {
    struct _cffi_freeme_s *next;
    union _cffi_union_alignment_u alignment;
};

#ifdef __GNUC__
  __attribute__((unused))
#endif
static int _cffi_convert_array_argument(CTypeDescrObject *ctptr, PyObject *arg,
                                        char **output_data, Py_ssize_t datasize,
                                        struct _cffi_freeme_s **freeme)
{
    char *p;
    if (datasize < 0)
        return -1;

    p = *output_data;
    if (p == NULL) {
        struct _cffi_freeme_s *fp = (struct _cffi_freeme_s *)PyObject_Malloc(
            offsetof(struct _cffi_freeme_s, alignment) + (size_t)datasize);
        if (fp == NULL)
            return -1;
        fp->next = *freeme;
        *freeme = fp;
        p = *output_data = (char *)&fp->alignment;
    }
    memset((void *)p, 0, (size_t)datasize);
    return _cffi_convert_array_from_object(p, ctptr, arg);
}

#ifdef __GNUC__
  __attribute__((unused))
#endif
static void _cffi_free_array_arguments(struct _cffi_freeme_s *freeme)
{
    do {
        void *p = (void *)freeme;
        freeme = freeme->next;
        PyObject_Free(p);
    } while (freeme != NULL);
}

static int _cffi_init(void)
{
    PyObject *module, *c_api_object = NULL;

    module = PyImport_ImportModule("_cffi_backend");
    if (module == NULL)
        goto failure;

    c_api_object = PyObject_GetAttrString(module, "_C_API");
    if (c_api_object == NULL)
        goto failure;
    if (!PyCapsule_CheckExact(c_api_object)) {
        PyErr_SetNone(PyExc_ImportError);
        goto failure;
    }
    memcpy(_cffi_exports, PyCapsule_GetPointer(c_api_object, "cffi"),
           _CFFI_NUM_EXPORTS * sizeof(void *));

    Py_DECREF(module);
    Py_DECREF(c_api_object);
    return 0;

  failure:
    Py_XDECREF(module);
    Py_XDECREF(c_api_object);
    return -1;
}

#define _cffi_type(num) ((CTypeDescrObject *)PyList_GET_ITEM(_cffi_types, num))

/**********/
'''

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\json_format.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Contains routines for printing protocol messages in JSON format.

Simple usage example:

  # Create a proto object and serialize it to a json format string.
  message = my_proto_pb2.MyMessage(foo='bar')
  json_string = json_format.MessageToJson(message)

  # Parse a json format string to proto object.
  message = json_format.Parse(json_string, my_proto_pb2.MyMessage())
"""

__author__ = 'jieluo@google.com (Jie Luo)'


import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re

from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database
from google.protobuf.internal import type_checkers


_INT_TYPES = frozenset([
    descriptor.FieldDescriptor.CPPTYPE_INT32,
    descriptor.FieldDescriptor.CPPTYPE_UINT32,
    descriptor.FieldDescriptor.CPPTYPE_INT64,
    descriptor.FieldDescriptor.CPPTYPE_UINT64,
])
_INT64_TYPES = frozenset([
    descriptor.FieldDescriptor.CPPTYPE_INT64,
    descriptor.FieldDescriptor.CPPTYPE_UINT64,
])
_FLOAT_TYPES = frozenset([
    descriptor.FieldDescriptor.CPPTYPE_FLOAT,
    descriptor.FieldDescriptor.CPPTYPE_DOUBLE,
])
_INFINITY = 'Infinity'
_NEG_INFINITY = '-Infinity'
_NAN = 'NaN'

_UNPAIRED_SURROGATE_PATTERN = re.compile(
    '[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]'
)

_VALID_EXTENSION_NAME = re.compile(r'\[[a-zA-Z0-9\._]*\]$')


class Error(Exception):
  """Top-level module error for json_format."""


class SerializeToJsonError(Error):
  """Thrown if serialization to JSON fails."""


class ParseError(Error):
  """Thrown in case of parsing error."""


class EnumStringValueParseError(ParseError):
  """Thrown if unknown string enum value is encountered.
  This exception is suppressed if ignore_unknown_fields is set.
  """


def MessageToJson(
    message,
    preserving_proto_field_name=False,
    indent=2,
    sort_keys=False,
    use_integers_for_enums=False,
    descriptor_pool=None,
    float_precision=None,
    ensure_ascii=True,
    always_print_fields_with_no_presence=False,
):
  """Converts protobuf message to JSON format.

  Args:
    message: The protocol buffers message instance to serialize.
    always_print_fields_with_no_presence: If True, fields without
      presence (implicit presence scalars, repeated fields, and map fields) will
      always be serialized. Any field that supports presence is not affected by
      this option (including singular message fields and oneof fields).
    preserving_proto_field_name: If True, use the original proto field names as
      defined in the .proto file. If False, convert the field names to
      lowerCamelCase.
    indent: The JSON object will be pretty-printed with this indent level. An
      indent level of 0 or negative will only insert newlines. If the indent
      level is None, no newlines will be inserted.
    sort_keys: If True, then the output will be sorted by field names.
    use_integers_for_enums: If true, print integers instead of enum names.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
      default.
    float_precision: If set, use this to specify float field valid digits.
    ensure_ascii: If True, strings with non-ASCII characters are escaped. If
      False, Unicode strings are returned unchanged.

  Returns:
    A string containing the JSON formatted protocol buffer message.
  """
  printer = _Printer(
      preserving_proto_field_name,
      use_integers_for_enums,
      descriptor_pool,
      float_precision,
      always_print_fields_with_no_presence
  )
  return printer.ToJsonString(message, indent, sort_keys, ensure_ascii)


def MessageToDict(
    message,
    always_print_fields_with_no_presence=False,
    preserving_proto_field_name=False,
    use_integers_for_enums=False,
    descriptor_pool=None,
    float_precision=None,
):
  """Converts protobuf message to a dictionary.

  When the dictionary is encoded to JSON, it conforms to proto3 JSON spec.

  Args:
    message: The protocol buffers message instance to serialize.
    always_print_fields_with_no_presence: If True, fields without
      presence (implicit presence scalars, repeated fields, and map fields) will
      always be serialized. Any field that supports presence is not affected by
      this option (including singular message fields and oneof fields).
    preserving_proto_field_name: If True, use the original proto field names as
      defined in the .proto file. If False, convert the field names to
      lowerCamelCase.
    use_integers_for_enums: If true, print integers instead of enum names.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
      default.
    float_precision: If set, use this to specify float field valid digits.

  Returns:
    A dict representation of the protocol buffer message.
  """
  printer = _Printer(
      preserving_proto_field_name,
      use_integers_for_enums,
      descriptor_pool,
      float_precision,
      always_print_fields_with_no_presence,
  )
  # pylint: disable=protected-access
  return printer._MessageToJsonObject(message)


def _IsMapEntry(field):
  return (
      field.type == descriptor.FieldDescriptor.TYPE_MESSAGE
      and field.message_type.has_options
      and field.message_type.GetOptions().map_entry
  )


class _Printer(object):
  """JSON format printer for protocol message."""

  def __init__(
      self,
      preserving_proto_field_name=False,
      use_integers_for_enums=False,
      descriptor_pool=None,
      float_precision=None,
      always_print_fields_with_no_presence=False,
  ):
    self.always_print_fields_with_no_presence = (
        always_print_fields_with_no_presence
    )
    self.preserving_proto_field_name = preserving_proto_field_name
    self.use_integers_for_enums = use_integers_for_enums
    self.descriptor_pool = descriptor_pool
    if float_precision:
      self.float_format = '.{}g'.format(float_precision)
    else:
      self.float_format = None

  def ToJsonString(self, message, indent, sort_keys, ensure_ascii):
    js = self._MessageToJsonObject(message)
    return json.dumps(
        js, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii
    )

  def _MessageToJsonObject(self, message):
    """Converts message to an object according to Proto3 JSON Specification."""
    message_descriptor = message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
      return self._WrapperMessageToJsonObject(message)
    if full_name in _WKTJSONMETHODS:
      return methodcaller(_WKTJSONMETHODS[full_name][0], message)(self)
    js = {}
    return self._RegularMessageToJsonObject(message, js)

  def _RegularMessageToJsonObject(self, message, js):
    """Converts normal message according to Proto3 JSON Specification."""
    fields = message.ListFields()

    try:
      for field, value in fields:
        if self.preserving_proto_field_name:
          name = field.name
        else:
          name = field.json_name
        if _IsMapEntry(field):
          # Convert a map field.
          v_field = field.message_type.fields_by_name['value']
          js_map = {}
          for key in value:
            if isinstance(key, bool):
              if key:
                recorded_key = 'true'
              else:
                recorded_key = 'false'
            else:
              recorded_key = str(key)
            js_map[recorded_key] = self._FieldToJsonObject(v_field, value[key])
          js[name] = js_map
        elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          # Convert a repeated field.
          js[name] = [self._FieldToJsonObject(field, k) for k in value]
        elif field.is_extension:
          name = '[%s]' % field.full_name
          js[name] = self._FieldToJsonObject(field, value)
        else:
          js[name] = self._FieldToJsonObject(field, value)

      # Serialize default value if including_default_value_fields is True.
      if (
          self.always_print_fields_with_no_presence
      ):
        message_descriptor = message.DESCRIPTOR
        for field in message_descriptor.fields:

          # always_print_fields_with_no_presence doesn't apply to
          # any field which supports presence.
          if (
              self.always_print_fields_with_no_presence
              and field.has_presence
          ):
            continue

          if self.preserving_proto_field_name:
            name = field.name
          else:
            name = field.json_name
          if name in js:
            # Skip the field which has been serialized already.
            continue
          if _IsMapEntry(field):
            js[name] = {}
          elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
            js[name] = []
          else:
            js[name] = self._FieldToJsonObject(field, field.default_value)

    except ValueError as e:
      raise SerializeToJsonError(
          'Failed to serialize {0} field: {1}.'.format(field.name, e)
      ) from e

    return js

  def _FieldToJsonObject(self, field, value):
    """Converts field value according to Proto3 JSON Specification."""
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
      return self._MessageToJsonObject(value)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
      if self.use_integers_for_enums:
        return value
      if field.enum_type.full_name == 'google.protobuf.NullValue':
        return None
      enum_value = field.enum_type.values_by_number.get(value, None)
      if enum_value is not None:
        return enum_value.name
      else:
        if field.enum_type.is_closed:
          raise SerializeToJsonError(
              'Enum field contains an integer value '
              'which can not mapped to an enum value.'
          )
        else:
          return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
      if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
        # Use base64 Data encoding for bytes
        return base64.b64encode(value).decode('utf-8')
      else:
        return str(value)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
      return bool(value)
    elif field.cpp_type in _INT64_TYPES:
      return str(value)
    elif field.cpp_type in _FLOAT_TYPES:
      if math.isinf(value):
        if value < 0.0:
          return _NEG_INFINITY
        else:
          return _INFINITY
      if math.isnan(value):
        return _NAN
      if self.float_format:
        return float(format(value, self.float_format))
      elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_FLOAT:
        return type_checkers.ToShortestFloat(value)

    return value

  def _AnyMessageToJsonObject(self, message):
    """Converts Any message according to Proto3 JSON Specification."""
    if not message.ListFields():
      return {}
    # Must print @type first, use OrderedDict instead of {}
    js = OrderedDict()
    type_url = message.type_url
    js['@type'] = type_url
    sub_message = _CreateMessageFromTypeUrl(type_url, self.descriptor_pool)
    sub_message.ParseFromString(message.value)
    message_descriptor = sub_message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
      js['value'] = self._WrapperMessageToJsonObject(sub_message)
      return js
    if full_name in _WKTJSONMETHODS:
      js['value'] = methodcaller(_WKTJSONMETHODS[full_name][0], sub_message)(
          self
      )
      return js
    return self._RegularMessageToJsonObject(sub_message, js)

  def _GenericMessageToJsonObject(self, message):
    """Converts message according to Proto3 JSON Specification."""
    # Duration, Timestamp and FieldMask have ToJsonString method to do the
    # convert. Users can also call the method directly.
    return message.ToJsonString()

  def _ValueMessageToJsonObject(self, message):
    """Converts Value message according to Proto3 JSON Specification."""
    which = message.WhichOneof('kind')
    # If the Value message is not set treat as null_value when serialize
    # to JSON. The parse back result will be different from original message.
    if which is None or which == 'null_value':
      return None
    if which == 'list_value':
      return self._ListValueMessageToJsonObject(message.list_value)
    if which == 'number_value':
      value = message.number_value
      if math.isinf(value):
        raise ValueError(
            'Fail to serialize Infinity for Value.number_value, '
            'which would parse as string_value'
        )
      if math.isnan(value):
        raise ValueError(
            'Fail to serialize NaN for Value.number_value, '
            'which would parse as string_value'
        )
    else:
      value = getattr(message, which)
    oneof_descriptor = message.DESCRIPTOR.fields_by_name[which]
    return self._FieldToJsonObject(oneof_descriptor, value)

  def _ListValueMessageToJsonObject(self, message):
    """Converts ListValue message according to Proto3 JSON Specification."""
    return [self._ValueMessageToJsonObject(value) for value in message.values]

  def _StructMessageToJsonObject(self, message):
    """Converts Struct message according to Proto3 JSON Specification."""
    fields = message.fields
    ret = {}
    for key in fields:
      ret[key] = self._ValueMessageToJsonObject(fields[key])
    return ret

  def _WrapperMessageToJsonObject(self, message):
    return self._FieldToJsonObject(
        message.DESCRIPTOR.fields_by_name['value'], message.value
    )


def _IsWrapperMessage(message_descriptor):
  return message_descriptor.file.name == 'google/protobuf/wrappers.proto'


def _DuplicateChecker(js):
  result = {}
  for name, value in js:
    if name in result:
      raise ParseError('Failed to load JSON: duplicate key {0}.'.format(name))
    result[name] = value
  return result


def _CreateMessageFromTypeUrl(type_url, descriptor_pool):
  """Creates a message from a type URL."""
  db = symbol_database.Default()
  pool = db.pool if descriptor_pool is None else descriptor_pool
  type_name = type_url.split('/')[-1]
  try:
    message_descriptor = pool.FindMessageTypeByName(type_name)
  except KeyError as e:
    raise TypeError(
        'Can not find message descriptor by type_url: {0}'.format(type_url)
    ) from e
  message_class = message_factory.GetMessageClass(message_descriptor)
  return message_class()


def Parse(
    text,
    message,
    ignore_unknown_fields=False,
    descriptor_pool=None,
    max_recursion_depth=100,
):
  """Parses a JSON representation of a protocol message into a message.

  Args:
    text: Message JSON representation.
    message: A protocol buffer message to merge into.
    ignore_unknown_fields: If True, do not raise errors for unknown fields.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
      default.
    max_recursion_depth: max recursion depth of JSON message to be deserialized.
      JSON messages over this depth will fail to be deserialized. Default value
      is 100.

  Returns:
    The same message passed as argument.

  Raises::
    ParseError: On JSON parsing problems.
  """
  if not isinstance(text, str):
    text = text.decode('utf-8')

  try:
    js = json.loads(text, object_pairs_hook=_DuplicateChecker)
  except Exception as e:
    raise ParseError('Failed to load JSON: {0}.'.format(str(e))) from e

  try:
    return ParseDict(
        js, message, ignore_unknown_fields, descriptor_pool, max_recursion_depth
    )
  except ParseError as e:
    raise e
  except Exception as e:
    raise ParseError(
        'Failed to parse JSON: {0}: {1}.'.format(type(e).__name__, str(e))
    ) from e


def ParseDict(
    js_dict,
    message,
    ignore_unknown_fields=False,
    descriptor_pool=None,
    max_recursion_depth=100,
):
  """Parses a JSON dictionary representation into a message.

  Args:
    js_dict: Dict representation of a JSON message.
    message: A protocol buffer message to merge into.
    ignore_unknown_fields: If True, do not raise errors for unknown fields.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
      default.
    max_recursion_depth: max recursion depth of JSON message to be deserialized.
      JSON messages over this depth will fail to be deserialized. Default value
      is 100.

  Returns:
    The same message passed as argument.
  """
  parser = _Parser(ignore_unknown_fields, descriptor_pool, max_recursion_depth)
  parser.ConvertMessage(js_dict, message, '')
  return message


_INT_OR_FLOAT = (int, float)
_LIST_LIKE = (list, tuple)


class _Parser(object):
  """JSON format parser for protocol message."""

  def __init__(
      self, ignore_unknown_fields, descriptor_pool, max_recursion_depth
  ):
    self.ignore_unknown_fields = ignore_unknown_fields
    self.descriptor_pool = descriptor_pool
    self.max_recursion_depth = max_recursion_depth
    self.recursion_depth = 0

  def ConvertMessage(self, value, message, path):
    """Convert a JSON object into a message.

    Args:
      value: A JSON object.
      message: A WKT or regular protocol message to record the data.
      path: parent path to log parse error info.

    Raises:
      ParseError: In case of convert problems.
    """
    self.recursion_depth += 1
    if self.recursion_depth > self.max_recursion_depth:
      raise ParseError(
          'Message too deep. Max recursion depth is {0}'.format(
              self.max_recursion_depth
          )
      )
    message_descriptor = message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if not path:
      path = message_descriptor.name
    if _IsWrapperMessage(message_descriptor):
      self._ConvertWrapperMessage(value, message, path)
    elif full_name in _WKTJSONMETHODS:
      methodcaller(_WKTJSONMETHODS[full_name][1], value, message, path)(self)
    else:
      self._ConvertFieldValuePair(value, message, path)
    self.recursion_depth -= 1

  def _ConvertFieldValuePair(self, js, message, path):
    """Convert field value pairs into regular message.

    Args:
      js: A JSON object to convert the field value pairs.
      message: A regular protocol message to record the data.
      path: parent path to log parse error info.

    Raises:
      ParseError: In case of problems converting.
    """
    names = []
    message_descriptor = message.DESCRIPTOR
    fields_by_json_name = dict(
        (f.json_name, f) for f in message_descriptor.fields
    )
    for name in js:
      try:
        field = fields_by_json_name.get(name, None)
        if not field:
          field = message_descriptor.fields_by_name.get(name, None)
        if not field and _VALID_EXTENSION_NAME.match(name):
          if not message_descriptor.is_extendable:
            raise ParseError(
                'Message type {0} does not have extensions at {1}'.format(
                    message_descriptor.full_name, path
                )
            )
          identifier = name[1:-1]  # strip [] brackets
          # pylint: disable=protected-access
          field = message.Extensions._FindExtensionByName(identifier)
          # pylint: enable=protected-access
          if not field:
            # Try looking for extension by the message type name, dropping the
            # field name following the final . separator in full_name.
            identifier = '.'.join(identifier.split('.')[:-1])
            # pylint: disable=protected-access
            field = message.Extensions._FindExtensionByName(identifier)
            # pylint: enable=protected-access
        if not field:
          if self.ignore_unknown_fields:
            continue
          raise ParseError(
              (
                  'Message type "{0}" has no field named "{1}" at "{2}".\n'
                  ' Available Fields(except extensions): "{3}"'
              ).format(
                  message_descriptor.full_name,
                  name,
                  path,
                  [f.json_name for f in message_descriptor.fields],
              )
          )
        if name in names:
          raise ParseError(
              'Message type "{0}" should not have multiple '
              '"{1}" fields at "{2}".'.format(
                  message.DESCRIPTOR.full_name, name, path
              )
          )
        names.append(name)
        value = js[name]
        # Check no other oneof field is parsed.
        if field.containing_oneof is not None and value is not None:
          oneof_name = field.containing_oneof.name
          if oneof_name in names:
            raise ParseError(
                'Message type "{0}" should not have multiple '
                '"{1}" oneof fields at "{2}".'.format(
                    message.DESCRIPTOR.full_name, oneof_name, path
                )
            )
          names.append(oneof_name)

        if value is None:
          if (
              field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE
              and field.message_type.full_name == 'google.protobuf.Value'
          ):
            sub_message = getattr(message, field.name)
            sub_message.null_value = 0
          elif (
              field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM
              and field.enum_type.full_name == 'google.protobuf.NullValue'
          ):
            setattr(message, field.name, 0)
          else:
            message.ClearField(field.name)
          continue

        # Parse field value.
        if _IsMapEntry(field):
          message.ClearField(field.name)
          self._ConvertMapFieldValue(
              value, message, field, '{0}.{1}'.format(path, name)
          )
        elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          message.ClearField(field.name)
          if not isinstance(value, _LIST_LIKE):
            raise ParseError(
                'repeated field {0} must be in [] which is {1} at {2}'.format(
                    name, value, path
                )
            )
          if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
            # Repeated message field.
            for index, item in enumerate(value):
              sub_message = getattr(message, field.name).add()
              # None is a null_value in Value.
              if (
                  item is None
                  and sub_message.DESCRIPTOR.full_name
                  != 'google.protobuf.Value'
              ):
                raise ParseError(
                    'null is not allowed to be used as an element'
                    ' in a repeated field at {0}.{1}[{2}]'.format(
                        path, name, index
                    )
                )
              self.ConvertMessage(
                  item, sub_message, '{0}.{1}[{2}]'.format(path, name, index)
              )
          else:
            # Repeated scalar field.
            for index, item in enumerate(value):
              if item is None:
                raise ParseError(
                    'null is not allowed to be used as an element'
                    ' in a repeated field at {0}.{1}[{2}]'.format(
                        path, name, index
                    )
                )
              self._ConvertAndAppendScalar(
                message, field, item, '{0}.{1}[{2}]'.format(path, name, index))
        elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
          if field.is_extension:
            sub_message = message.Extensions[field]
          else:
            sub_message = getattr(message, field.name)
          sub_message.SetInParent()
          self.ConvertMessage(value, sub_message, '{0}.{1}'.format(path, name))
        else:
          if field.is_extension:
            self._ConvertAndSetScalarExtension(message, field, value, '{0}.{1}'.format(path, name))
          else:
            self._ConvertAndSetScalar(message, field, value, '{0}.{1}'.format(path, name))
      except ParseError as e:
        if field and field.containing_oneof is None:
          raise ParseError(
              'Failed to parse {0} field: {1}.'.format(name, e)
          ) from e
        else:
          raise ParseError(str(e)) from e
      except ValueError as e:
        raise ParseError(
            'Failed to parse {0} field: {1}.'.format(name, e)
        ) from e
      except TypeError as e:
        raise ParseError(
            'Failed to parse {0} field: {1}.'.format(name, e)
        ) from e

  def _ConvertAnyMessage(self, value, message, path):
    """Convert a JSON representation into Any message."""
    if isinstance(value, dict) and not value:
      return
    try:
      type_url = value['@type']
    except KeyError as e:
      raise ParseError(
          '@type is missing when parsing any message at {0}'.format(path)
      ) from e

    try:
      sub_message = _CreateMessageFromTypeUrl(type_url, self.descriptor_pool)
    except TypeError as e:
      raise ParseError('{0} at {1}'.format(e, path)) from e
    message_descriptor = sub_message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
      self._ConvertWrapperMessage(
          value['value'], sub_message, '{0}.value'.format(path)
      )
    elif full_name in _WKTJSONMETHODS:
      methodcaller(
          _WKTJSONMETHODS[full_name][1],
          value['value'],
          sub_message,
          '{0}.value'.format(path),
      )(self)
    else:
      del value['@type']
      try:
        self._ConvertFieldValuePair(value, sub_message, path)
      finally:
        value['@type'] = type_url
    # Sets Any message
    message.value = sub_message.SerializeToString()
    message.type_url = type_url

  def _ConvertGenericMessage(self, value, message, path):
    """Convert a JSON representation into message with FromJsonString."""
    # Duration, Timestamp, FieldMask have a FromJsonString method to do the
    # conversion. Users can also call the method directly.
    try:
      message.FromJsonString(value)
    except ValueError as e:
      raise ParseError('{0} at {1}'.format(e, path)) from e

  def _ConvertValueMessage(self, value, message, path):
    """Convert a JSON representation into Value message."""
    if isinstance(value, dict):
      self._ConvertStructMessage(value, message.struct_value, path)
    elif isinstance(value, _LIST_LIKE):
      self._ConvertListOrTupleValueMessage(value, message.list_value, path)
    elif value is None:
      message.null_value = 0
    elif isinstance(value, bool):
      message.bool_value = value
    elif isinstance(value, str):
      message.string_value = value
    elif isinstance(value, _INT_OR_FLOAT):
      message.number_value = value
    else:
      raise ParseError(
          'Value {0} has unexpected type {1} at {2}'.format(
              value, type(value), path
          )
      )

  def _ConvertListOrTupleValueMessage(self, value, message, path):
    """Convert a JSON representation into ListValue message."""
    if not isinstance(value, _LIST_LIKE):
      raise ParseError(
          'ListValue must be in [] which is {0} at {1}'.format(value, path)
      )
    message.ClearField('values')
    for index, item in enumerate(value):
      self._ConvertValueMessage(
          item, message.values.add(), '{0}[{1}]'.format(path, index)
      )

  def _ConvertStructMessage(self, value, message, path):
    """Convert a JSON representation into Struct message."""
    if not isinstance(value, dict):
      raise ParseError(
          'Struct must be in a dict which is {0} at {1}'.format(value, path)
      )
    # Clear will mark the struct as modified so it will be created even if
    # there are no values.
    message.Clear()
    for key in value:
      self._ConvertValueMessage(
          value[key], message.fields[key], '{0}.{1}'.format(path, key)
      )
    return

  def _ConvertWrapperMessage(self, value, message, path):
    """Convert a JSON representation into Wrapper message."""
    field = message.DESCRIPTOR.fields_by_name['value']
    self._ConvertAndSetScalar(message, field, value, path='{0}.value'.format(path))

  def _ConvertMapFieldValue(self, value, message, field, path):
    """Convert map field value for a message map field.

    Args:
      value: A JSON object to convert the map field value.
      message: A protocol message to record the converted data.
      field: The descriptor of the map field to be converted.
      path: parent path to log parse error info.

    Raises:
      ParseError: In case of convert problems.
    """
    if not isinstance(value, dict):
      raise ParseError(
          'Map field {0} must be in a dict which is {1} at {2}'.format(
              field.name, value, path
          )
      )
    key_field = field.message_type.fields_by_name['key']
    value_field = field.message_type.fields_by_name['value']
    for key in value:
      key_value = _ConvertScalarFieldValue(
          key, key_field, '{0}.key'.format(path), True
      )
      if value_field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
        self.ConvertMessage(
            value[key],
            getattr(message, field.name)[key_value],
            '{0}[{1}]'.format(path, key_value),
        )
      else:
        self._ConvertAndSetScalarToMapKey(
            message,
            field,
            key_value,
            value[key],
            path='{0}[{1}]'.format(path, key_value))

  def _ConvertAndSetScalarExtension(self, message, extension_field, js_value, path):
    """Convert scalar from js_value and assign it to message.Extensions[extension_field]."""
    try:
      message.Extensions[extension_field] = _ConvertScalarFieldValue(
          js_value, extension_field, path)
    except EnumStringValueParseError:
      if not self.ignore_unknown_fields:
        raise

  def _ConvertAndSetScalar(self, message, field, js_value, path):
    """Convert scalar from js_value and assign it to message.field."""
    try:
      setattr(
          message,
          field.name,
          _ConvertScalarFieldValue(js_value, field, path))
    except EnumStringValueParseError:
      if not self.ignore_unknown_fields:
        raise

  def _ConvertAndAppendScalar(self, message, repeated_field, js_value, path):
    """Convert scalar from js_value and append it to message.repeated_field."""
    try:
      getattr(message, repeated_field.name).append(
          _ConvertScalarFieldValue(js_value, repeated_field, path))
    except EnumStringValueParseError:
      if not self.ignore_unknown_fields:
        raise

  def _ConvertAndSetScalarToMapKey(self, message, map_field, converted_key, js_value, path):
    """Convert scalar from 'js_value' and add it to message.map_field[converted_key]."""
    try:
      getattr(message, map_field.name)[converted_key] = _ConvertScalarFieldValue(
          js_value, map_field.message_type.fields_by_name['value'], path,
      )
    except EnumStringValueParseError:
      if not self.ignore_unknown_fields:
        raise


def _ConvertScalarFieldValue(value, field, path, require_str=False):
  """Convert a single scalar field value.

  Args:
    value: A scalar value to convert the scalar field value.
    field: The descriptor of the field to convert.
    path: parent path to log parse error info.
    require_str: If True, the field value must be a str.

  Returns:
    The converted scalar field value

  Raises:
    ParseError: In case of convert problems.
    EnumStringValueParseError: In case of unknown enum string value.
  """
  try:
    if field.cpp_type in _INT_TYPES:
      return _ConvertInteger(value)
    elif field.cpp_type in _FLOAT_TYPES:
      return _ConvertFloat(value, field)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
      return _ConvertBool(value, require_str)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
      if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
        if isinstance(value, str):
          encoded = value.encode('utf-8')
        else:
          encoded = value
        # Add extra padding '='
        padded_value = encoded + b'=' * (4 - len(encoded) % 4)
        return base64.urlsafe_b64decode(padded_value)
      else:
        # Checking for unpaired surrogates appears to be unreliable,
        # depending on the specific Python version, so we check manually.
        if _UNPAIRED_SURROGATE_PATTERN.search(value):
          raise ParseError('Unpaired surrogate')
        return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
      # Convert an enum value.
      enum_value = field.enum_type.values_by_name.get(value, None)
      if enum_value is None:
        try:
          number = int(value)
          enum_value = field.enum_type.values_by_number.get(number, None)
        except ValueError as e:
          # Since parsing to integer failed and lookup in values_by_name didn't
          # find this name, we have an enum string value which is unknown.
          raise EnumStringValueParseError(
              'Invalid enum value {0} for enum type {1}'.format(
                  value, field.enum_type.full_name
              )
          ) from e
        if enum_value is None:
          if field.enum_type.is_closed:
            raise ParseError(
                'Invalid enum value {0} for enum type {1}'.format(
                    value, field.enum_type.full_name
                )
            )
          else:
            return number
      return enum_value.number
  except EnumStringValueParseError as e:
    raise EnumStringValueParseError('{0} at {1}'.format(e, path)) from e
  except ParseError as e:
    raise ParseError('{0} at {1}'.format(e, path)) from e


def _ConvertInteger(value):
  """Convert an integer.

  Args:
    value: A scalar value to convert.

  Returns:
    The integer value.

  Raises:
    ParseError: If an integer couldn't be consumed.
  """
  if isinstance(value, float) and not value.is_integer():
    raise ParseError("Couldn't parse integer: {0}".format(value))

  if isinstance(value, str) and value.find(' ') != -1:
    raise ParseError('Couldn\'t parse integer: "{0}"'.format(value))

  if isinstance(value, bool):
    raise ParseError(
        'Bool value {0} is not acceptable for integer field'.format(value)
    )

  try:
    return int(value)
  except ValueError as e:
    # Attempt to parse as an integer-valued float.
    try:
      f = float(value)
    except ValueError:
      # Raise the original exception for the int parse.
      raise e  # pylint: disable=raise-missing-from
    if not f.is_integer():
      raise ParseError(
          'Couldn\'t parse non-integer string: "{0}"'.format(value)
      ) from e
    return int(f)


def _ConvertFloat(value, field):
  """Convert an floating point number."""
  if isinstance(value, float):
    if math.isnan(value):
      raise ParseError('Couldn\'t parse NaN, use quoted "NaN" instead')
    if math.isinf(value):
      if value > 0:
        raise ParseError(
            "Couldn't parse Infinity or value too large, "
            'use quoted "Infinity" instead'
        )
      else:
        raise ParseError(
            "Couldn't parse -Infinity or value too small, "
            'use quoted "-Infinity" instead'
        )
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_FLOAT:
      # pylint: disable=protected-access
      if value > type_checkers._FLOAT_MAX:
        raise ParseError('Float value too large')
      # pylint: disable=protected-access
      if value < type_checkers._FLOAT_MIN:
        raise ParseError('Float value too small')
  if value == 'nan':
    raise ParseError('Couldn\'t parse float "nan", use "NaN" instead')
  try:
    # Assume Python compatible syntax.
    return float(value)
  except ValueError as e:
    # Check alternative spellings.
    if value == _NEG_INFINITY:
      return float('-inf')
    elif value == _INFINITY:
      return float('inf')
    elif value == _NAN:
      return float('nan')
    else:
      raise ParseError("Couldn't parse float: {0}".format(value)) from e


def _ConvertBool(value, require_str):
  """Convert a boolean value.

  Args:
    value: A scalar value to convert.
    require_str: If True, value must be a str.

  Returns:
    The bool parsed.

  Raises:
    ParseError: If a boolean value couldn't be consumed.
  """
  if require_str:
    if value == 'true':
      return True
    elif value == 'false':
      return False
    else:
      raise ParseError('Expected "true" or "false", not {0}'.format(value))

  if not isinstance(value, bool):
    raise ParseError('Expected true or false without quotes')
  return value


_WKTJSONMETHODS = {
    'google.protobuf.Any': ['_AnyMessageToJsonObject', '_ConvertAnyMessage'],
    'google.protobuf.Duration': [
        '_GenericMessageToJsonObject',
        '_ConvertGenericMessage',
    ],
    'google.protobuf.FieldMask': [
        '_GenericMessageToJsonObject',
        '_ConvertGenericMessage',
    ],
    'google.protobuf.ListValue': [
        '_ListValueMessageToJsonObject',
        '_ConvertListOrTupleValueMessage',
    ],
    'google.protobuf.Struct': [
        '_StructMessageToJsonObject',
        '_ConvertStructMessage',
    ],
    'google.protobuf.Timestamp': [
        '_GenericMessageToJsonObject',
        '_ConvertGenericMessage',
    ],
    'google.protobuf.Value': [
        '_ValueMessageToJsonObject',
        '_ConvertValueMessage',
    ],
}