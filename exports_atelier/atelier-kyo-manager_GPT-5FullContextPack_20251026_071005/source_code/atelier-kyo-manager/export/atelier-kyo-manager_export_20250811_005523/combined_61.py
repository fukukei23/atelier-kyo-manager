
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\filters.py ===
"""Built-in template filters used with the ``|`` operator."""

import math
import random
import re
import typing
import typing as t
from collections import abc
from inspect import getattr_static
from itertools import chain
from itertools import groupby

from markupsafe import escape
from markupsafe import Markup
from markupsafe import soft_str

from .async_utils import async_variant
from .async_utils import auto_aiter
from .async_utils import auto_await
from .async_utils import auto_to_list
from .exceptions import FilterArgumentError
from .runtime import Undefined
from .utils import htmlsafe_json_dumps
from .utils import pass_context
from .utils import pass_environment
from .utils import pass_eval_context
from .utils import pformat
from .utils import url_quote
from .utils import urlize

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment
    from .nodes import EvalContext
    from .runtime import Context
    from .sandbox import SandboxedEnvironment  # noqa: F401

    class HasHTML(te.Protocol):
        def __html__(self) -> str:
            pass


F = t.TypeVar("F", bound=t.Callable[..., t.Any])
K = t.TypeVar("K")
V = t.TypeVar("V")


def ignore_case(value: V) -> V:
    """For use as a postprocessor for :func:`make_attrgetter`. Converts strings
    to lowercase and returns other types as-is."""
    if isinstance(value, str):
        return t.cast(V, value.lower())

    return value


def make_attrgetter(
    environment: "Environment",
    attribute: t.Optional[t.Union[str, int]],
    postprocess: t.Optional[t.Callable[[t.Any], t.Any]] = None,
    default: t.Optional[t.Any] = None,
) -> t.Callable[[t.Any], t.Any]:
    """Returns a callable that looks up the given attribute from a
    passed object with the rules of the environment.  Dots are allowed
    to access attributes of attributes.  Integer parts in paths are
    looked up as integers.
    """
    parts = _prepare_attribute_parts(attribute)

    def attrgetter(item: t.Any) -> t.Any:
        for part in parts:
            item = environment.getitem(item, part)

            if default is not None and isinstance(item, Undefined):
                item = default

        if postprocess is not None:
            item = postprocess(item)

        return item

    return attrgetter


def make_multi_attrgetter(
    environment: "Environment",
    attribute: t.Optional[t.Union[str, int]],
    postprocess: t.Optional[t.Callable[[t.Any], t.Any]] = None,
) -> t.Callable[[t.Any], t.List[t.Any]]:
    """Returns a callable that looks up the given comma separated
    attributes from a passed object with the rules of the environment.
    Dots are allowed to access attributes of each attribute.  Integer
    parts in paths are looked up as integers.

    The value returned by the returned callable is a list of extracted
    attribute values.

    Examples of attribute: "attr1,attr2", "attr1.inner1.0,attr2.inner2.0", etc.
    """
    if isinstance(attribute, str):
        split: t.Sequence[t.Union[str, int, None]] = attribute.split(",")
    else:
        split = [attribute]

    parts = [_prepare_attribute_parts(item) for item in split]

    def attrgetter(item: t.Any) -> t.List[t.Any]:
        items = [None] * len(parts)

        for i, attribute_part in enumerate(parts):
            item_i = item

            for part in attribute_part:
                item_i = environment.getitem(item_i, part)

            if postprocess is not None:
                item_i = postprocess(item_i)

            items[i] = item_i

        return items

    return attrgetter


def _prepare_attribute_parts(
    attr: t.Optional[t.Union[str, int]],
) -> t.List[t.Union[str, int]]:
    if attr is None:
        return []

    if isinstance(attr, str):
        return [int(x) if x.isdigit() else x for x in attr.split(".")]

    return [attr]


def do_forceescape(value: "t.Union[str, HasHTML]") -> Markup:
    """Enforce HTML escaping.  This will probably double escape variables."""
    if hasattr(value, "__html__"):
        value = t.cast("HasHTML", value).__html__()

    return escape(str(value))


def do_urlencode(
    value: t.Union[str, t.Mapping[str, t.Any], t.Iterable[t.Tuple[str, t.Any]]],
) -> str:
    """Quote data for use in a URL path or query using UTF-8.

    Basic wrapper around :func:`urllib.parse.quote` when given a
    string, or :func:`urllib.parse.urlencode` for a dict or iterable.

    :param value: Data to quote. A string will be quoted directly. A
        dict or iterable of ``(key, value)`` pairs will be joined as a
        query string.

    When given a string, "/" is not quoted. HTTP servers treat "/" and
    "%2F" equivalently in paths. If you need quoted slashes, use the
    ``|replace("/", "%2F")`` filter.

    .. versionadded:: 2.7
    """
    if isinstance(value, str) or not isinstance(value, abc.Iterable):
        return url_quote(value)

    if isinstance(value, dict):
        items: t.Iterable[t.Tuple[str, t.Any]] = value.items()
    else:
        items = value  # type: ignore

    return "&".join(
        f"{url_quote(k, for_qs=True)}={url_quote(v, for_qs=True)}" for k, v in items
    )


@pass_eval_context
def do_replace(
    eval_ctx: "EvalContext", s: str, old: str, new: str, count: t.Optional[int] = None
) -> str:
    """Return a copy of the value with all occurrences of a substring
    replaced with a new one. The first argument is the substring
    that should be replaced, the second is the replacement string.
    If the optional third argument ``count`` is given, only the first
    ``count`` occurrences are replaced:

    .. sourcecode:: jinja

        {{ "Hello World"|replace("Hello", "Goodbye") }}
            -> Goodbye World

        {{ "aaaaargh"|replace("a", "d'oh, ", 2) }}
            -> d'oh, d'oh, aaargh
    """
    if count is None:
        count = -1

    if not eval_ctx.autoescape:
        return str(s).replace(str(old), str(new), count)

    if (
        hasattr(old, "__html__")
        or hasattr(new, "__html__")
        and not hasattr(s, "__html__")
    ):
        s = escape(s)
    else:
        s = soft_str(s)

    return s.replace(soft_str(old), soft_str(new), count)


def do_upper(s: str) -> str:
    """Convert a value to uppercase."""
    return soft_str(s).upper()


def do_lower(s: str) -> str:
    """Convert a value to lowercase."""
    return soft_str(s).lower()


def do_items(value: t.Union[t.Mapping[K, V], Undefined]) -> t.Iterator[t.Tuple[K, V]]:
    """Return an iterator over the ``(key, value)`` items of a mapping.

    ``x|items`` is the same as ``x.items()``, except if ``x`` is
    undefined an empty iterator is returned.

    This filter is useful if you expect the template to be rendered with
    an implementation of Jinja in another programming language that does
    not have a ``.items()`` method on its mapping type.

    .. code-block:: html+jinja

        <dl>
        {% for key, value in my_dict|items %}
            <dt>{{ key }}
            <dd>{{ value }}
        {% endfor %}
        </dl>

    .. versionadded:: 3.1
    """
    if isinstance(value, Undefined):
        return

    if not isinstance(value, abc.Mapping):
        raise TypeError("Can only get item pairs from a mapping.")

    yield from value.items()


# Check for characters that would move the parser state from key to value.
# https://html.spec.whatwg.org/#attribute-name-state
_attr_key_re = re.compile(r"[\s/>=]", flags=re.ASCII)


@pass_eval_context
def do_xmlattr(
    eval_ctx: "EvalContext", d: t.Mapping[str, t.Any], autospace: bool = True
) -> str:
    """Create an SGML/XML attribute string based on the items in a dict.

    **Values** that are neither ``none`` nor ``undefined`` are automatically
    escaped, safely allowing untrusted user input.

    User input should not be used as **keys** to this filter. If any key
    contains a space, ``/`` solidus, ``>`` greater-than sign, or ``=`` equals
    sign, this fails with a ``ValueError``. Regardless of this, user input
    should never be used as keys to this filter, or must be separately validated
    first.

    .. sourcecode:: html+jinja

        <ul{{ {'class': 'my_list', 'missing': none,
                'id': 'list-%d'|format(variable)}|xmlattr }}>
        ...
        </ul>

    Results in something like this:

    .. sourcecode:: html

        <ul class="my_list" id="list-42">
        ...
        </ul>

    As you can see it automatically prepends a space in front of the item
    if the filter returned something unless the second parameter is false.

    .. versionchanged:: 3.1.4
        Keys with ``/`` solidus, ``>`` greater-than sign, or ``=`` equals sign
        are not allowed.

    .. versionchanged:: 3.1.3
        Keys with spaces are not allowed.
    """
    items = []

    for key, value in d.items():
        if value is None or isinstance(value, Undefined):
            continue

        if _attr_key_re.search(key) is not None:
            raise ValueError(f"Invalid character in attribute name: {key!r}")

        items.append(f'{escape(key)}="{escape(value)}"')

    rv = " ".join(items)

    if autospace and rv:
        rv = " " + rv

    if eval_ctx.autoescape:
        rv = Markup(rv)

    return rv


def do_capitalize(s: str) -> str:
    """Capitalize a value. The first character will be uppercase, all others
    lowercase.
    """
    return soft_str(s).capitalize()


_word_beginning_split_re = re.compile(r"([-\s({\[<]+)")


def do_title(s: str) -> str:
    """Return a titlecased version of the value. I.e. words will start with
    uppercase letters, all remaining characters are lowercase.
    """
    return "".join(
        [
            item[0].upper() + item[1:].lower()
            for item in _word_beginning_split_re.split(soft_str(s))
            if item
        ]
    )


def do_dictsort(
    value: t.Mapping[K, V],
    case_sensitive: bool = False,
    by: 'te.Literal["key", "value"]' = "key",
    reverse: bool = False,
) -> t.List[t.Tuple[K, V]]:
    """Sort a dict and yield (key, value) pairs. Python dicts may not
    be in the order you want to display them in, so sort them first.

    .. sourcecode:: jinja

        {% for key, value in mydict|dictsort %}
            sort the dict by key, case insensitive

        {% for key, value in mydict|dictsort(reverse=true) %}
            sort the dict by key, case insensitive, reverse order

        {% for key, value in mydict|dictsort(true) %}
            sort the dict by key, case sensitive

        {% for key, value in mydict|dictsort(false, 'value') %}
            sort the dict by value, case insensitive
    """
    if by == "key":
        pos = 0
    elif by == "value":
        pos = 1
    else:
        raise FilterArgumentError('You can only sort by either "key" or "value"')

    def sort_func(item: t.Tuple[t.Any, t.Any]) -> t.Any:
        value = item[pos]

        if not case_sensitive:
            value = ignore_case(value)

        return value

    return sorted(value.items(), key=sort_func, reverse=reverse)


@pass_environment
def do_sort(
    environment: "Environment",
    value: "t.Iterable[V]",
    reverse: bool = False,
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.List[V]":
    """Sort an iterable using Python's :func:`sorted`.

    .. sourcecode:: jinja

        {% for city in cities|sort %}
            ...
        {% endfor %}

    :param reverse: Sort descending instead of ascending.
    :param case_sensitive: When sorting strings, sort upper and lower
        case separately.
    :param attribute: When sorting objects or dicts, an attribute or
        key to sort by. Can use dot notation like ``"address.city"``.
        Can be a list of attributes like ``"age,name"``.

    The sort is stable, it does not change the relative order of
    elements that compare equal. This makes it is possible to chain
    sorts on different attributes and ordering.

    .. sourcecode:: jinja

        {% for user in users|sort(attribute="name")
            |sort(reverse=true, attribute="age") %}
            ...
        {% endfor %}

    As a shortcut to chaining when the direction is the same for all
    attributes, pass a comma separate list of attributes.

    .. sourcecode:: jinja

        {% for user in users|sort(attribute="age,name") %}
            ...
        {% endfor %}

    .. versionchanged:: 2.11.0
        The ``attribute`` parameter can be a comma separated list of
        attributes, e.g. ``"age,name"``.

    .. versionchanged:: 2.6
       The ``attribute`` parameter was added.
    """
    key_func = make_multi_attrgetter(
        environment, attribute, postprocess=ignore_case if not case_sensitive else None
    )
    return sorted(value, key=key_func, reverse=reverse)


@pass_environment
def sync_do_unique(
    environment: "Environment",
    value: "t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.Iterator[V]":
    """Returns a list of unique items from the given iterable.

    .. sourcecode:: jinja

        {{ ['foo', 'bar', 'foobar', 'FooBar']|unique|list }}
            -> ['foo', 'bar', 'foobar']

    The unique items are yielded in the same order as their first occurrence in
    the iterable passed to the filter.

    :param case_sensitive: Treat upper and lower case strings as distinct.
    :param attribute: Filter objects with unique values for this attribute.
    """
    getter = make_attrgetter(
        environment, attribute, postprocess=ignore_case if not case_sensitive else None
    )
    seen = set()

    for item in value:
        key = getter(item)

        if key not in seen:
            seen.add(key)
            yield item


@async_variant(sync_do_unique)  # type: ignore
async def do_unique(
    environment: "Environment",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.Iterator[V]":
    return sync_do_unique(
        environment, await auto_to_list(value), case_sensitive, attribute
    )


def _min_or_max(
    environment: "Environment",
    value: "t.Iterable[V]",
    func: "t.Callable[..., V]",
    case_sensitive: bool,
    attribute: t.Optional[t.Union[str, int]],
) -> "t.Union[V, Undefined]":
    it = iter(value)

    try:
        first = next(it)
    except StopIteration:
        return environment.undefined("No aggregated item, sequence was empty.")

    key_func = make_attrgetter(
        environment, attribute, postprocess=ignore_case if not case_sensitive else None
    )
    return func(chain([first], it), key=key_func)


@pass_environment
def do_min(
    environment: "Environment",
    value: "t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.Union[V, Undefined]":
    """Return the smallest item from the sequence.

    .. sourcecode:: jinja

        {{ [1, 2, 3]|min }}
            -> 1

    :param case_sensitive: Treat upper and lower case strings as distinct.
    :param attribute: Get the object with the min value of this attribute.
    """
    return _min_or_max(environment, value, min, case_sensitive, attribute)


@pass_environment
def do_max(
    environment: "Environment",
    value: "t.Iterable[V]",
    case_sensitive: bool = False,
    attribute: t.Optional[t.Union[str, int]] = None,
) -> "t.Union[V, Undefined]":
    """Return the largest item from the sequence.

    .. sourcecode:: jinja

        {{ [1, 2, 3]|max }}
            -> 3

    :param case_sensitive: Treat upper and lower case strings as distinct.
    :param attribute: Get the object with the max value of this attribute.
    """
    return _min_or_max(environment, value, max, case_sensitive, attribute)


def do_default(
    value: V,
    default_value: V = "",  # type: ignore
    boolean: bool = False,
) -> V:
    """If the value is undefined it will return the passed default value,
    otherwise the value of the variable:

    .. sourcecode:: jinja

        {{ my_variable|default('my_variable is not defined') }}

    This will output the value of ``my_variable`` if the variable was
    defined, otherwise ``'my_variable is not defined'``. If you want
    to use default with variables that evaluate to false you have to
    set the second parameter to `true`:

    .. sourcecode:: jinja

        {{ ''|default('the string was empty', true) }}

    .. versionchanged:: 2.11
       It's now possible to configure the :class:`~jinja2.Environment` with
       :class:`~jinja2.ChainableUndefined` to make the `default` filter work
       on nested elements and attributes that may contain undefined values
       in the chain without getting an :exc:`~jinja2.UndefinedError`.
    """
    if isinstance(value, Undefined) or (boolean and not value):
        return default_value

    return value


@pass_eval_context
def sync_do_join(
    eval_ctx: "EvalContext",
    value: t.Iterable[t.Any],
    d: str = "",
    attribute: t.Optional[t.Union[str, int]] = None,
) -> str:
    """Return a string which is the concatenation of the strings in the
    sequence. The separator between elements is an empty string per
    default, you can define it with the optional parameter:

    .. sourcecode:: jinja

        {{ [1, 2, 3]|join('|') }}
            -> 1|2|3

        {{ [1, 2, 3]|join }}
            -> 123

    It is also possible to join certain attributes of an object:

    .. sourcecode:: jinja

        {{ users|join(', ', attribute='username') }}

    .. versionadded:: 2.6
       The `attribute` parameter was added.
    """
    if attribute is not None:
        value = map(make_attrgetter(eval_ctx.environment, attribute), value)

    # no automatic escaping?  joining is a lot easier then
    if not eval_ctx.autoescape:
        return str(d).join(map(str, value))

    # if the delimiter doesn't have an html representation we check
    # if any of the items has.  If yes we do a coercion to Markup
    if not hasattr(d, "__html__"):
        value = list(value)
        do_escape = False

        for idx, item in enumerate(value):
            if hasattr(item, "__html__"):
                do_escape = True
            else:
                value[idx] = str(item)

        if do_escape:
            d = escape(d)
        else:
            d = str(d)

        return d.join(value)

    # no html involved, to normal joining
    return soft_str(d).join(map(soft_str, value))


@async_variant(sync_do_join)  # type: ignore
async def do_join(
    eval_ctx: "EvalContext",
    value: t.Union[t.AsyncIterable[t.Any], t.Iterable[t.Any]],
    d: str = "",
    attribute: t.Optional[t.Union[str, int]] = None,
) -> str:
    return sync_do_join(eval_ctx, await auto_to_list(value), d, attribute)


def do_center(value: str, width: int = 80) -> str:
    """Centers the value in a field of a given width."""
    return soft_str(value).center(width)


@pass_environment
def sync_do_first(
    environment: "Environment", seq: "t.Iterable[V]"
) -> "t.Union[V, Undefined]":
    """Return the first item of a sequence."""
    try:
        return next(iter(seq))
    except StopIteration:
        return environment.undefined("No first item, sequence was empty.")


@async_variant(sync_do_first)  # type: ignore
async def do_first(
    environment: "Environment", seq: "t.Union[t.AsyncIterable[V], t.Iterable[V]]"
) -> "t.Union[V, Undefined]":
    try:
        return await auto_aiter(seq).__anext__()
    except StopAsyncIteration:
        return environment.undefined("No first item, sequence was empty.")


@pass_environment
def do_last(
    environment: "Environment", seq: "t.Reversible[V]"
) -> "t.Union[V, Undefined]":
    """Return the last item of a sequence.

    Note: Does not work with generators. You may want to explicitly
    convert it to a list:

    .. sourcecode:: jinja

        {{ data | selectattr('name', '==', 'Jinja') | list | last }}
    """
    try:
        return next(iter(reversed(seq)))
    except StopIteration:
        return environment.undefined("No last item, sequence was empty.")


# No async do_last, it may not be safe in async mode.


@pass_context
def do_random(context: "Context", seq: "t.Sequence[V]") -> "t.Union[V, Undefined]":
    """Return a random item from the sequence."""
    try:
        return random.choice(seq)
    except IndexError:
        return context.environment.undefined("No random item, sequence was empty.")


def do_filesizeformat(value: t.Union[str, float, int], binary: bool = False) -> str:
    """Format the value like a 'human-readable' file size (i.e. 13 kB,
    4.1 MB, 102 Bytes, etc).  Per default decimal prefixes are used (Mega,
    Giga, etc.), if the second parameter is set to `True` the binary
    prefixes are used (Mebi, Gibi).
    """
    bytes = float(value)
    base = 1024 if binary else 1000
    prefixes = [
        ("KiB" if binary else "kB"),
        ("MiB" if binary else "MB"),
        ("GiB" if binary else "GB"),
        ("TiB" if binary else "TB"),
        ("PiB" if binary else "PB"),
        ("EiB" if binary else "EB"),
        ("ZiB" if binary else "ZB"),
        ("YiB" if binary else "YB"),
    ]

    if bytes == 1:
        return "1 Byte"
    elif bytes < base:
        return f"{int(bytes)} Bytes"
    else:
        for i, prefix in enumerate(prefixes):
            unit = base ** (i + 2)

            if bytes < unit:
                return f"{base * bytes / unit:.1f} {prefix}"

        return f"{base * bytes / unit:.1f} {prefix}"


def do_pprint(value: t.Any) -> str:
    """Pretty print a variable. Useful for debugging."""
    return pformat(value)


_uri_scheme_re = re.compile(r"^([\w.+-]{2,}:(/){0,2})$")


@pass_eval_context
def do_urlize(
    eval_ctx: "EvalContext",
    value: str,
    trim_url_limit: t.Optional[int] = None,
    nofollow: bool = False,
    target: t.Optional[str] = None,
    rel: t.Optional[str] = None,
    extra_schemes: t.Optional[t.Iterable[str]] = None,
) -> str:
    """Convert URLs in text into clickable links.

    This may not recognize links in some situations. Usually, a more
    comprehensive formatter, such as a Markdown library, is a better
    choice.

    Works on ``http://``, ``https://``, ``www.``, ``mailto:``, and email
    addresses. Links with trailing punctuation (periods, commas, closing
    parentheses) and leading punctuation (opening parentheses) are
    recognized excluding the punctuation. Email addresses that include
    header fields are not recognized (for example,
    ``mailto:address@example.com?cc=copy@example.com``).

    :param value: Original text containing URLs to link.
    :param trim_url_limit: Shorten displayed URL values to this length.
    :param nofollow: Add the ``rel=nofollow`` attribute to links.
    :param target: Add the ``target`` attribute to links.
    :param rel: Add the ``rel`` attribute to links.
    :param extra_schemes: Recognize URLs that start with these schemes
        in addition to the default behavior. Defaults to
        ``env.policies["urlize.extra_schemes"]``, which defaults to no
        extra schemes.

    .. versionchanged:: 3.0
        The ``extra_schemes`` parameter was added.

    .. versionchanged:: 3.0
        Generate ``https://`` links for URLs without a scheme.

    .. versionchanged:: 3.0
        The parsing rules were updated. Recognize email addresses with
        or without the ``mailto:`` scheme. Validate IP addresses. Ignore
        parentheses and brackets in more cases.

    .. versionchanged:: 2.8
       The ``target`` parameter was added.
    """
    policies = eval_ctx.environment.policies
    rel_parts = set((rel or "").split())

    if nofollow:
        rel_parts.add("nofollow")

    rel_parts.update((policies["urlize.rel"] or "").split())
    rel = " ".join(sorted(rel_parts)) or None

    if target is None:
        target = policies["urlize.target"]

    if extra_schemes is None:
        extra_schemes = policies["urlize.extra_schemes"] or ()

    for scheme in extra_schemes:
        if _uri_scheme_re.fullmatch(scheme) is None:
            raise FilterArgumentError(f"{scheme!r} is not a valid URI scheme prefix.")

    rv = urlize(
        value,
        trim_url_limit=trim_url_limit,
        rel=rel,
        target=target,
        extra_schemes=extra_schemes,
    )

    if eval_ctx.autoescape:
        rv = Markup(rv)

    return rv


def do_indent(
    s: str, width: t.Union[int, str] = 4, first: bool = False, blank: bool = False
) -> str:
    """Return a copy of the string with each line indented by 4 spaces. The
    first line and blank lines are not indented by default.

    :param width: Number of spaces, or a string, to indent by.
    :param first: Don't skip indenting the first line.
    :param blank: Don't skip indenting empty lines.

    .. versionchanged:: 3.0
        ``width`` can be a string.

    .. versionchanged:: 2.10
        Blank lines are not indented by default.

        Rename the ``indentfirst`` argument to ``first``.
    """
    if isinstance(width, str):
        indention = width
    else:
        indention = " " * width

    newline = "\n"

    if isinstance(s, Markup):
        indention = Markup(indention)
        newline = Markup(newline)

    s += newline  # this quirk is necessary for splitlines method

    if blank:
        rv = (newline + indention).join(s.splitlines())
    else:
        lines = s.splitlines()
        rv = lines.pop(0)

        if lines:
            rv += newline + newline.join(
                indention + line if line else line for line in lines
            )

    if first:
        rv = indention + rv

    return rv


@pass_environment
def do_truncate(
    env: "Environment",
    s: str,
    length: int = 255,
    killwords: bool = False,
    end: str = "...",
    leeway: t.Optional[int] = None,
) -> str:
    """Return a truncated copy of the string. The length is specified
    with the first parameter which defaults to ``255``. If the second
    parameter is ``true`` the filter will cut the text at length. Otherwise
    it will discard the last word. If the text was in fact
    truncated it will append an ellipsis sign (``"..."``). If you want a
    different ellipsis sign than ``"..."`` you can specify it using the
    third parameter. Strings that only exceed the length by the tolerance
    margin given in the fourth parameter will not be truncated.

    .. sourcecode:: jinja

        {{ "foo bar baz qux"|truncate(9) }}
            -> "foo..."
        {{ "foo bar baz qux"|truncate(9, True) }}
            -> "foo ba..."
        {{ "foo bar baz qux"|truncate(11) }}
            -> "foo bar baz qux"
        {{ "foo bar baz qux"|truncate(11, False, '...', 0) }}
            -> "foo bar..."

    The default leeway on newer Jinja versions is 5 and was 0 before but
    can be reconfigured globally.
    """
    if leeway is None:
        leeway = env.policies["truncate.leeway"]

    assert length >= len(end), f"expected length >= {len(end)}, got {length}"
    assert leeway >= 0, f"expected leeway >= 0, got {leeway}"

    if len(s) <= length + leeway:
        return s

    if killwords:
        return s[: length - len(end)] + end

    result = s[: length - len(end)].rsplit(" ", 1)[0]
    return result + end


@pass_environment
def do_wordwrap(
    environment: "Environment",
    s: str,
    width: int = 79,
    break_long_words: bool = True,
    wrapstring: t.Optional[str] = None,
    break_on_hyphens: bool = True,
) -> str:
    """Wrap a string to the given width. Existing newlines are treated
    as paragraphs to be wrapped separately.

    :param s: Original text to wrap.
    :param width: Maximum length of wrapped lines.
    :param break_long_words: If a word is longer than ``width``, break
        it across lines.
    :param break_on_hyphens: If a word contains hyphens, it may be split
        across lines.
    :param wrapstring: String to join each wrapped line. Defaults to
        :attr:`Environment.newline_sequence`.

    .. versionchanged:: 2.11
        Existing newlines are treated as paragraphs wrapped separately.

    .. versionchanged:: 2.11
        Added the ``break_on_hyphens`` parameter.

    .. versionchanged:: 2.7
        Added the ``wrapstring`` parameter.
    """
    import textwrap

    if wrapstring is None:
        wrapstring = environment.newline_sequence

    # textwrap.wrap doesn't consider existing newlines when wrapping.
    # If the string has a newline before width, wrap will still insert
    # a newline at width, resulting in a short line. Instead, split and
    # wrap each paragraph individually.
    return wrapstring.join(
        [
            wrapstring.join(
                textwrap.wrap(
                    line,
                    width=width,
                    expand_tabs=False,
                    replace_whitespace=False,
                    break_long_words=break_long_words,
                    break_on_hyphens=break_on_hyphens,
                )
            )
            for line in s.splitlines()
        ]
    )


_word_re = re.compile(r"\w+")


def do_wordcount(s: str) -> int:
    """Count the words in that string."""
    return len(_word_re.findall(soft_str(s)))


def do_int(value: t.Any, default: int = 0, base: int = 10) -> int:
    """Convert the value into an integer. If the
    conversion doesn't work it will return ``0``. You can
    override this default using the first parameter. You
    can also override the default base (10) in the second
    parameter, which handles input with prefixes such as
    0b, 0o and 0x for bases 2, 8 and 16 respectively.
    The base is ignored for decimal numbers and non-string values.
    """
    try:
        if isinstance(value, str):
            return int(value, base)

        return int(value)
    except (TypeError, ValueError):
        # this quirk is necessary so that "42.23"|int gives 42.
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default


def do_float(value: t.Any, default: float = 0.0) -> float:
    """Convert the value into a floating point number. If the
    conversion doesn't work it will return ``0.0``. You can
    override this default using the first parameter.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def do_format(value: str, *args: t.Any, **kwargs: t.Any) -> str:
    """Apply the given values to a `printf-style`_ format string, like
    ``string % values``.

    .. sourcecode:: jinja

        {{ "%s, %s!"|format(greeting, name) }}
        Hello, World!

    In most cases it should be more convenient and efficient to use the
    ``%`` operator or :meth:`str.format`.

    .. code-block:: text

        {{ "%s, %s!" % (greeting, name) }}
        {{ "{}, {}!".format(greeting, name) }}

    .. _printf-style: https://docs.python.org/library/stdtypes.html
        #printf-style-string-formatting
    """
    if args and kwargs:
        raise FilterArgumentError(
            "can't handle positional and keyword arguments at the same time"
        )

    return soft_str(value) % (kwargs or args)


def do_trim(value: str, chars: t.Optional[str] = None) -> str:
    """Strip leading and trailing characters, by default whitespace."""
    return soft_str(value).strip(chars)


def do_striptags(value: "t.Union[str, HasHTML]") -> str:
    """Strip SGML/XML tags and replace adjacent whitespace by one space."""
    if hasattr(value, "__html__"):
        value = t.cast("HasHTML", value).__html__()

    return Markup(str(value)).striptags()


def sync_do_slice(
    value: "t.Collection[V]", slices: int, fill_with: "t.Optional[V]" = None
) -> "t.Iterator[t.List[V]]":
    """Slice an iterator and return a list of lists containing
    those items. Useful if you want to create a div containing
    three ul tags that represent columns:

    .. sourcecode:: html+jinja

        <div class="columnwrapper">
          {%- for column in items|slice(3) %}
            <ul class="column-{{ loop.index }}">
            {%- for item in column %}
              <li>{{ item }}</li>
            {%- endfor %}
            </ul>
          {%- endfor %}
        </div>

    If you pass it a second argument it's used to fill missing
    values on the last iteration.
    """
    seq = list(value)
    length = len(seq)
    items_per_slice = length // slices
    slices_with_extra = length % slices
    offset = 0

    for slice_number in range(slices):
        start = offset + slice_number * items_per_slice

        if slice_number < slices_with_extra:
            offset += 1

        end = offset + (slice_number + 1) * items_per_slice
        tmp = seq[start:end]

        if fill_with is not None and slice_number >= slices_with_extra:
            tmp.append(fill_with)

        yield tmp


@async_variant(sync_do_slice)  # type: ignore
async def do_slice(
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    slices: int,
    fill_with: t.Optional[t.Any] = None,
) -> "t.Iterator[t.List[V]]":
    return sync_do_slice(await auto_to_list(value), slices, fill_with)


def do_batch(
    value: "t.Iterable[V]", linecount: int, fill_with: "t.Optional[V]" = None
) -> "t.Iterator[t.List[V]]":
    """
    A filter that batches items. It works pretty much like `slice`
    just the other way round. It returns a list of lists with the
    given number of items. If you provide a second parameter this
    is used to fill up missing items. See this example:

    .. sourcecode:: html+jinja

        <table>
        {%- for row in items|batch(3, '&nbsp;') %}
          <tr>
          {%- for column in row %}
            <td>{{ column }}</td>
          {%- endfor %}
          </tr>
        {%- endfor %}
        </table>
    """
    tmp: t.List[V] = []

    for item in value:
        if len(tmp) == linecount:
            yield tmp
            tmp = []

        tmp.append(item)

    if tmp:
        if fill_with is not None and len(tmp) < linecount:
            tmp += [fill_with] * (linecount - len(tmp))

        yield tmp


def do_round(
    value: float,
    precision: int = 0,
    method: 'te.Literal["common", "ceil", "floor"]' = "common",
) -> float:
    """Round the number to a given precision. The first
    parameter specifies the precision (default is ``0``), the
    second the rounding method:

    - ``'common'`` rounds either up or down
    - ``'ceil'`` always rounds up
    - ``'floor'`` always rounds down

    If you don't specify a method ``'common'`` is used.

    .. sourcecode:: jinja

        {{ 42.55|round }}
            -> 43.0
        {{ 42.55|round(1, 'floor') }}
            -> 42.5

    Note that even if rounded to 0 precision, a float is returned.  If
    you need a real integer, pipe it through `int`:

    .. sourcecode:: jinja

        {{ 42.55|round|int }}
            -> 43
    """
    if method not in {"common", "ceil", "floor"}:
        raise FilterArgumentError("method must be common, ceil or floor")

    if method == "common":
        return round(value, precision)

    func = getattr(math, method)
    return t.cast(float, func(value * (10**precision)) / (10**precision))


class _GroupTuple(t.NamedTuple):
    grouper: t.Any
    list: t.List[t.Any]

    # Use the regular tuple repr to hide this subclass if users print
    # out the value during debugging.
    def __repr__(self) -> str:
        return tuple.__repr__(self)

    def __str__(self) -> str:
        return tuple.__str__(self)


@pass_environment
def sync_do_groupby(
    environment: "Environment",
    value: "t.Iterable[V]",
    attribute: t.Union[str, int],
    default: t.Optional[t.Any] = None,
    case_sensitive: bool = False,
) -> "t.List[_GroupTuple]":
    """Group a sequence of objects by an attribute using Python's
    :func:`itertools.groupby`. The attribute can use dot notation for
    nested access, like ``"address.city"``. Unlike Python's ``groupby``,
    the values are sorted first so only one group is returned for each
    unique value.

    For example, a list of ``User`` objects with a ``city`` attribute
    can be rendered in groups. In this example, ``grouper`` refers to
    the ``city`` value of the group.

    .. sourcecode:: html+jinja

        <ul>{% for city, items in users|groupby("city") %}
          <li>{{ city }}
            <ul>{% for user in items %}
              <li>{{ user.name }}
            {% endfor %}</ul>
          </li>
        {% endfor %}</ul>

    ``groupby`` yields namedtuples of ``(grouper, list)``, which
    can be used instead of the tuple unpacking above. ``grouper`` is the
    value of the attribute, and ``list`` is the items with that value.

    .. sourcecode:: html+jinja

        <ul>{% for group in users|groupby("city") %}
          <li>{{ group.grouper }}: {{ group.list|join(", ") }}
        {% endfor %}</ul>

    You can specify a ``default`` value to use if an object in the list
    does not have the given attribute.

    .. sourcecode:: jinja

        <ul>{% for city, items in users|groupby("city", default="NY") %}
          <li>{{ city }}: {{ items|map(attribute="name")|join(", ") }}</li>
        {% endfor %}</ul>

    Like the :func:`~jinja-filters.sort` filter, sorting and grouping is
    case-insensitive by default. The ``key`` for each group will have
    the case of the first item in that group of values. For example, if
    a list of users has cities ``["CA", "NY", "ca"]``, the "CA" group
    will have two values. This can be disabled by passing
    ``case_sensitive=True``.

    .. versionchanged:: 3.1
        Added the ``case_sensitive`` parameter. Sorting and grouping is
        case-insensitive by default, matching other filters that do
        comparisons.

    .. versionchanged:: 3.0
        Added the ``default`` parameter.

    .. versionchanged:: 2.6
        The attribute supports dot notation for nested access.
    """
    expr = make_attrgetter(
        environment,
        attribute,
        postprocess=ignore_case if not case_sensitive else None,
        default=default,
    )
    out = [
        _GroupTuple(key, list(values))
        for key, values in groupby(sorted(value, key=expr), expr)
    ]

    if not case_sensitive:
        # Return the real key from the first value instead of the lowercase key.
        output_expr = make_attrgetter(environment, attribute, default=default)
        out = [_GroupTuple(output_expr(values[0]), values) for _, values in out]

    return out


@async_variant(sync_do_groupby)  # type: ignore
async def do_groupby(
    environment: "Environment",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    attribute: t.Union[str, int],
    default: t.Optional[t.Any] = None,
    case_sensitive: bool = False,
) -> "t.List[_GroupTuple]":
    expr = make_attrgetter(
        environment,
        attribute,
        postprocess=ignore_case if not case_sensitive else None,
        default=default,
    )
    out = [
        _GroupTuple(key, await auto_to_list(values))
        for key, values in groupby(sorted(await auto_to_list(value), key=expr), expr)
    ]

    if not case_sensitive:
        # Return the real key from the first value instead of the lowercase key.
        output_expr = make_attrgetter(environment, attribute, default=default)
        out = [_GroupTuple(output_expr(values[0]), values) for _, values in out]

    return out


@pass_environment
def sync_do_sum(
    environment: "Environment",
    iterable: "t.Iterable[V]",
    attribute: t.Optional[t.Union[str, int]] = None,
    start: V = 0,  # type: ignore
) -> V:
    """Returns the sum of a sequence of numbers plus the value of parameter
    'start' (which defaults to 0).  When the sequence is empty it returns
    start.

    It is also possible to sum up only certain attributes:

    .. sourcecode:: jinja

        Total: {{ items|sum(attribute='price') }}

    .. versionchanged:: 2.6
       The ``attribute`` parameter was added to allow summing up over
       attributes.  Also the ``start`` parameter was moved on to the right.
    """
    if attribute is not None:
        iterable = map(make_attrgetter(environment, attribute), iterable)

    return sum(iterable, start)  # type: ignore[no-any-return, call-overload]


@async_variant(sync_do_sum)  # type: ignore
async def do_sum(
    environment: "Environment",
    iterable: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    attribute: t.Optional[t.Union[str, int]] = None,
    start: V = 0,  # type: ignore
) -> V:
    rv = start

    if attribute is not None:
        func = make_attrgetter(environment, attribute)
    else:

        def func(x: V) -> V:
            return x

    async for item in auto_aiter(iterable):
        rv += func(item)

    return rv


def sync_do_list(value: "t.Iterable[V]") -> "t.List[V]":
    """Convert the value into a list.  If it was a string the returned list
    will be a list of characters.
    """
    return list(value)


@async_variant(sync_do_list)  # type: ignore
async def do_list(value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]") -> "t.List[V]":
    return await auto_to_list(value)


def do_mark_safe(value: str) -> Markup:
    """Mark the value as safe which means that in an environment with automatic
    escaping enabled this variable will not be escaped.
    """
    return Markup(value)


def do_mark_unsafe(value: str) -> str:
    """Mark a value as unsafe.  This is the reverse operation for :func:`safe`."""
    return str(value)


@typing.overload
def do_reverse(value: str) -> str: ...


@typing.overload
def do_reverse(value: "t.Iterable[V]") -> "t.Iterable[V]": ...


def do_reverse(value: t.Union[str, t.Iterable[V]]) -> t.Union[str, t.Iterable[V]]:
    """Reverse the object or return an iterator that iterates over it the other
    way round.
    """
    if isinstance(value, str):
        return value[::-1]

    try:
        return reversed(value)  # type: ignore
    except TypeError:
        try:
            rv = list(value)
            rv.reverse()
            return rv
        except TypeError as e:
            raise FilterArgumentError("argument must be iterable") from e


@pass_environment
def do_attr(
    environment: "Environment", obj: t.Any, name: str
) -> t.Union[Undefined, t.Any]:
    """Get an attribute of an object. ``foo|attr("bar")`` works like
    ``foo.bar``, but returns undefined instead of falling back to ``foo["bar"]``
    if the attribute doesn't exist.

    See :ref:`Notes on subscriptions <notes-on-subscriptions>` for more details.
    """
    # Environment.getattr will fall back to obj[name] if obj.name doesn't exist.
    # But we want to call env.getattr to get behavior such as sandboxing.
    # Determine if the attr exists first, so we know the fallback won't trigger.
    try:
        # This avoids executing properties/descriptors, but misses __getattr__
        # and __getattribute__ dynamic attrs.
        getattr_static(obj, name)
    except AttributeError:
        # This finds dynamic attrs, and we know it's not a descriptor at this point.
        if not hasattr(obj, name):
            return environment.undefined(obj=obj, name=name)

    return environment.getattr(obj, name)


@typing.overload
def sync_do_map(
    context: "Context",
    value: t.Iterable[t.Any],
    name: str,
    *args: t.Any,
    **kwargs: t.Any,
) -> t.Iterable[t.Any]: ...


@typing.overload
def sync_do_map(
    context: "Context",
    value: t.Iterable[t.Any],
    *,
    attribute: str = ...,
    default: t.Optional[t.Any] = None,
) -> t.Iterable[t.Any]: ...


@pass_context
def sync_do_map(
    context: "Context", value: t.Iterable[t.Any], *args: t.Any, **kwargs: t.Any
) -> t.Iterable[t.Any]:
    """Applies a filter on a sequence of objects or looks up an attribute.
    This is useful when dealing with lists of objects but you are really
    only interested in a certain value of it.

    The basic usage is mapping on an attribute.  Imagine you have a list
    of users but you are only interested in a list of usernames:

    .. sourcecode:: jinja

        Users on this page: {{ users|map(attribute='username')|join(', ') }}

    You can specify a ``default`` value to use if an object in the list
    does not have the given attribute.

    .. sourcecode:: jinja

        {{ users|map(attribute="username", default="Anonymous")|join(", ") }}

    Alternatively you can let it invoke a filter by passing the name of the
    filter and the arguments afterwards.  A good example would be applying a
    text conversion filter on a sequence:

    .. sourcecode:: jinja

        Users on this page: {{ titles|map('lower')|join(', ') }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (u.username for u in users)
        (getattr(u, "username", "Anonymous") for u in users)
        (do_lower(x) for x in titles)

    .. versionchanged:: 2.11.0
        Added the ``default`` parameter.

    .. versionadded:: 2.7
    """
    if value:
        func = prepare_map(context, args, kwargs)

        for item in value:
            yield func(item)


@typing.overload
def do_map(
    context: "Context",
    value: t.Union[t.AsyncIterable[t.Any], t.Iterable[t.Any]],
    name: str,
    *args: t.Any,
    **kwargs: t.Any,
) -> t.Iterable[t.Any]: ...


@typing.overload
def do_map(
    context: "Context",
    value: t.Union[t.AsyncIterable[t.Any], t.Iterable[t.Any]],
    *,
    attribute: str = ...,
    default: t.Optional[t.Any] = None,
) -> t.Iterable[t.Any]: ...


@async_variant(sync_do_map)  # type: ignore
async def do_map(
    context: "Context",
    value: t.Union[t.AsyncIterable[t.Any], t.Iterable[t.Any]],
    *args: t.Any,
    **kwargs: t.Any,
) -> t.AsyncIterable[t.Any]:
    if value:
        func = prepare_map(context, args, kwargs)

        async for item in auto_aiter(value):
            yield await auto_await(func(item))


@pass_context
def sync_do_select(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    """Filters a sequence of objects by applying a test to each object,
    and only selecting the objects with the test succeeding.

    If no test is specified, each object will be evaluated as a boolean.

    Example usage:

    .. sourcecode:: jinja

        {{ numbers|select("odd") }}
        {{ numbers|select("odd") }}
        {{ numbers|select("divisibleby", 3) }}
        {{ numbers|select("lessthan", 42) }}
        {{ strings|select("equalto", "mystring") }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (n for n in numbers if test_odd(n))
        (n for n in numbers if test_divisibleby(n, 3))

    .. versionadded:: 2.7
    """
    return select_or_reject(context, value, args, kwargs, lambda x: x, False)


@async_variant(sync_do_select)  # type: ignore
async def do_select(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: x, False)


@pass_context
def sync_do_reject(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    """Filters a sequence of objects by applying a test to each object,
    and rejecting the objects with the test succeeding.

    If no test is specified, each object will be evaluated as a boolean.

    Example usage:

    .. sourcecode:: jinja

        {{ numbers|reject("odd") }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (n for n in numbers if not test_odd(n))

    .. versionadded:: 2.7
    """
    return select_or_reject(context, value, args, kwargs, lambda x: not x, False)


@async_variant(sync_do_reject)  # type: ignore
async def do_reject(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: not x, False)


@pass_context
def sync_do_selectattr(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    """Filters a sequence of objects by applying a test to the specified
    attribute of each object, and only selecting the objects with the
    test succeeding.

    If no test is specified, the attribute's value will be evaluated as
    a boolean.

    Example usage:

    .. sourcecode:: jinja

        {{ users|selectattr("is_active") }}
        {{ users|selectattr("email", "none") }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (user for user in users if user.is_active)
        (user for user in users if test_none(user.email))

    .. versionadded:: 2.7
    """
    return select_or_reject(context, value, args, kwargs, lambda x: x, True)


@async_variant(sync_do_selectattr)  # type: ignore
async def do_selectattr(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: x, True)


@pass_context
def sync_do_rejectattr(
    context: "Context", value: "t.Iterable[V]", *args: t.Any, **kwargs: t.Any
) -> "t.Iterator[V]":
    """Filters a sequence of objects by applying a test to the specified
    attribute of each object, and rejecting the objects with the test
    succeeding.

    If no test is specified, the attribute's value will be evaluated as
    a boolean.

    .. sourcecode:: jinja

        {{ users|rejectattr("is_active") }}
        {{ users|rejectattr("email", "none") }}

    Similar to a generator comprehension such as:

    .. code-block:: python

        (user for user in users if not user.is_active)
        (user for user in users if not test_none(user.email))

    .. versionadded:: 2.7
    """
    return select_or_reject(context, value, args, kwargs, lambda x: not x, True)


@async_variant(sync_do_rejectattr)  # type: ignore
async def do_rejectattr(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    *args: t.Any,
    **kwargs: t.Any,
) -> "t.AsyncIterator[V]":
    return async_select_or_reject(context, value, args, kwargs, lambda x: not x, True)


@pass_eval_context
def do_tojson(
    eval_ctx: "EvalContext", value: t.Any, indent: t.Optional[int] = None
) -> Markup:
    """Serialize an object to a string of JSON, and mark it safe to
    render in HTML. This filter is only for use in HTML documents.

    The returned string is safe to render in HTML documents and
    ``<script>`` tags. The exception is in HTML attributes that are
    double quoted; either use single quotes or the ``|forceescape``
    filter.

    :param value: The object to serialize to JSON.
    :param indent: The ``indent`` parameter passed to ``dumps``, for
        pretty-printing the value.

    .. versionadded:: 2.9
    """
    policies = eval_ctx.environment.policies
    dumps = policies["json.dumps_function"]
    kwargs = policies["json.dumps_kwargs"]

    if indent is not None:
        kwargs = kwargs.copy()
        kwargs["indent"] = indent

    return htmlsafe_json_dumps(value, dumps=dumps, **kwargs)


def prepare_map(
    context: "Context", args: t.Tuple[t.Any, ...], kwargs: t.Dict[str, t.Any]
) -> t.Callable[[t.Any], t.Any]:
    if not args and "attribute" in kwargs:
        attribute = kwargs.pop("attribute")
        default = kwargs.pop("default", None)

        if kwargs:
            raise FilterArgumentError(
                f"Unexpected keyword argument {next(iter(kwargs))!r}"
            )

        func = make_attrgetter(context.environment, attribute, default=default)
    else:
        try:
            name = args[0]
            args = args[1:]
        except LookupError:
            raise FilterArgumentError("map requires a filter argument") from None

        def func(item: t.Any) -> t.Any:
            return context.environment.call_filter(
                name, item, args, kwargs, context=context
            )

    return func


def prepare_select_or_reject(
    context: "Context",
    args: t.Tuple[t.Any, ...],
    kwargs: t.Dict[str, t.Any],
    modfunc: t.Callable[[t.Any], t.Any],
    lookup_attr: bool,
) -> t.Callable[[t.Any], t.Any]:
    if lookup_attr:
        try:
            attr = args[0]
        except LookupError:
            raise FilterArgumentError("Missing parameter for attribute name") from None

        transfunc = make_attrgetter(context.environment, attr)
        off = 1
    else:
        off = 0

        def transfunc(x: V) -> V:
            return x

    try:
        name = args[off]
        args = args[1 + off :]

        def func(item: t.Any) -> t.Any:
            return context.environment.call_test(name, item, args, kwargs, context)

    except LookupError:
        func = bool  # type: ignore

    return lambda item: modfunc(func(transfunc(item)))


def select_or_reject(
    context: "Context",
    value: "t.Iterable[V]",
    args: t.Tuple[t.Any, ...],
    kwargs: t.Dict[str, t.Any],
    modfunc: t.Callable[[t.Any], t.Any],
    lookup_attr: bool,
) -> "t.Iterator[V]":
    if value:
        func = prepare_select_or_reject(context, args, kwargs, modfunc, lookup_attr)

        for item in value:
            if func(item):
                yield item


async def async_select_or_reject(
    context: "Context",
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
    args: t.Tuple[t.Any, ...],
    kwargs: t.Dict[str, t.Any],
    modfunc: t.Callable[[t.Any], t.Any],
    lookup_attr: bool,
) -> "t.AsyncIterator[V]":
    if value:
        func = prepare_select_or_reject(context, args, kwargs, modfunc, lookup_attr)

        async for item in auto_aiter(value):
            if func(item):
                yield item


FILTERS = {
    "abs": abs,
    "attr": do_attr,
    "batch": do_batch,
    "capitalize": do_capitalize,
    "center": do_center,
    "count": len,
    "d": do_default,
    "default": do_default,
    "dictsort": do_dictsort,
    "e": escape,
    "escape": escape,
    "filesizeformat": do_filesizeformat,
    "first": do_first,
    "float": do_float,
    "forceescape": do_forceescape,
    "format": do_format,
    "groupby": do_groupby,
    "indent": do_indent,
    "int": do_int,
    "join": do_join,
    "last": do_last,
    "length": len,
    "list": do_list,
    "lower": do_lower,
    "items": do_items,
    "map": do_map,
    "min": do_min,
    "max": do_max,
    "pprint": do_pprint,
    "random": do_random,
    "reject": do_reject,
    "rejectattr": do_rejectattr,
    "replace": do_replace,
    "reverse": do_reverse,
    "round": do_round,
    "safe": do_mark_safe,
    "select": do_select,
    "selectattr": do_selectattr,
    "slice": do_slice,
    "sort": do_sort,
    "string": soft_str,
    "striptags": do_striptags,
    "sum": do_sum,
    "title": do_title,
    "trim": do_trim,
    "truncate": do_truncate,
    "unique": do_unique,
    "upper": do_upper,
    "urlencode": do_urlencode,
    "urlize": do_urlize,
    "wordcount": do_wordcount,
    "wordwrap": do_wordwrap,
    "xmlattr": do_xmlattr,
    "tojson": do_tojson,
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\_elements_constructors.py ===
# sql/_elements_constructors.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

import typing
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple as typing_Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import coercions
from . import roles
from .base import _NoArg
from .coercions import _document_text_coercion
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import Case
from .elements import Cast
from .elements import CollationClause
from .elements import CollectionAggregate
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Extract
from .elements import False_
from .elements import FunctionFilter
from .elements import Label
from .elements import Null
from .elements import Over
from .elements import TextClause
from .elements import True_
from .elements import TryCast
from .elements import Tuple
from .elements import TypeCoerce
from .elements import UnaryExpression
from .elements import WithinGroup
from .functions import FunctionElement
from ..util.typing import Literal

if typing.TYPE_CHECKING:
    from ._typing import _ByArgument
    from ._typing import _ColumnExpressionArgument
    from ._typing import _ColumnExpressionOrLiteralArgument
    from ._typing import _ColumnExpressionOrStrLabelArgument
    from ._typing import _TypeEngineArgument
    from .elements import BinaryExpression
    from .selectable import FromClause
    from .type_api import TypeEngine

_T = TypeVar("_T")


def all_(expr: _ColumnExpressionArgument[_T]) -> CollectionAggregate[bool]:
    """Produce an ALL expression.

    For dialects such as that of PostgreSQL, this operator applies
    to usage of the :class:`_types.ARRAY` datatype, for that of
    MySQL, it may apply to a subquery.  e.g.::

        # renders on PostgreSQL:
        # '5 = ALL (somearray)'
        expr = 5 == all_(mytable.c.somearray)

        # renders on MySQL:
        # '5 = ALL (SELECT value FROM table)'
        expr = 5 == all_(select(table.c.value))

    Comparison to NULL may work using ``None``::

        None == all_(mytable.c.somearray)

    The any_() / all_() operators also feature a special "operand flipping"
    behavior such that if any_() / all_() are used on the left side of a
    comparison using a standalone operator such as ``==``, ``!=``, etc.
    (not including operator methods such as
    :meth:`_sql.ColumnOperators.is_`) the rendered expression is flipped::

        # would render '5 = ALL (column)`
        all_(mytable.c.column) == 5

    Or with ``None``, which note will not perform
    the usual step of rendering "IS" as is normally the case for NULL::

        # would render 'NULL = ALL(somearray)'
        all_(mytable.c.somearray) == None

    .. versionchanged:: 1.4.26  repaired the use of any_() / all_()
       comparing to NULL on the right side to be flipped to the left.

    The column-level :meth:`_sql.ColumnElement.all_` method (not to be
    confused with :class:`_types.ARRAY` level
    :meth:`_types.ARRAY.Comparator.all`) is shorthand for
    ``all_(col)``::

        5 == mytable.c.somearray.all_()

    .. seealso::

        :meth:`_sql.ColumnOperators.all_`

        :func:`_expression.any_`

    """
    return CollectionAggregate._create_all(expr)


def and_(  # type: ignore[empty-body]
    initial_clause: Union[Literal[True], _ColumnExpressionArgument[bool]],
    *clauses: _ColumnExpressionArgument[bool],
) -> ColumnElement[bool]:
    r"""Produce a conjunction of expressions joined by ``AND``.

    E.g.::

        from sqlalchemy import and_

        stmt = select(users_table).where(
            and_(users_table.c.name == "wendy", users_table.c.enrolled == True)
        )

    The :func:`.and_` conjunction is also available using the
    Python ``&`` operator (though note that compound expressions
    need to be parenthesized in order to function with Python
    operator precedence behavior)::

        stmt = select(users_table).where(
            (users_table.c.name == "wendy") & (users_table.c.enrolled == True)
        )

    The :func:`.and_` operation is also implicit in some cases;
    the :meth:`_expression.Select.where`
    method for example can be invoked multiple
    times against a statement, which will have the effect of each
    clause being combined using :func:`.and_`::

        stmt = (
            select(users_table)
            .where(users_table.c.name == "wendy")
            .where(users_table.c.enrolled == True)
        )

    The :func:`.and_` construct must be given at least one positional
    argument in order to be valid; a :func:`.and_` construct with no
    arguments is ambiguous.   To produce an "empty" or dynamically
    generated :func:`.and_`  expression, from a given list of expressions,
    a "default" element of :func:`_sql.true` (or just ``True``) should be
    specified::

        from sqlalchemy import true

        criteria = and_(true(), *expressions)

    The above expression will compile to SQL as the expression ``true``
    or ``1 = 1``, depending on backend, if no other expressions are
    present.  If expressions are present, then the :func:`_sql.true` value is
    ignored as it does not affect the outcome of an AND expression that
    has other elements.

    .. deprecated:: 1.4  The :func:`.and_` element now requires that at
       least one argument is passed; creating the :func:`.and_` construct
       with no arguments is deprecated, and will emit a deprecation warning
       while continuing to produce a blank SQL string.

    .. seealso::

        :func:`.or_`

    """
    ...


if not TYPE_CHECKING:
    # handle deprecated case which allows zero-arguments
    def and_(*clauses):  # noqa: F811
        r"""Produce a conjunction of expressions joined by ``AND``.

        E.g.::

            from sqlalchemy import and_

            stmt = select(users_table).where(
                and_(users_table.c.name == "wendy", users_table.c.enrolled == True)
            )

        The :func:`.and_` conjunction is also available using the
        Python ``&`` operator (though note that compound expressions
        need to be parenthesized in order to function with Python
        operator precedence behavior)::

            stmt = select(users_table).where(
                (users_table.c.name == "wendy") & (users_table.c.enrolled == True)
            )

        The :func:`.and_` operation is also implicit in some cases;
        the :meth:`_expression.Select.where`
        method for example can be invoked multiple
        times against a statement, which will have the effect of each
        clause being combined using :func:`.and_`::

            stmt = (
                select(users_table)
                .where(users_table.c.name == "wendy")
                .where(users_table.c.enrolled == True)
            )

        The :func:`.and_` construct must be given at least one positional
        argument in order to be valid; a :func:`.and_` construct with no
        arguments is ambiguous.   To produce an "empty" or dynamically
        generated :func:`.and_`  expression, from a given list of expressions,
        a "default" element of :func:`_sql.true` (or just ``True``) should be
        specified::

            from sqlalchemy import true

            criteria = and_(true(), *expressions)

        The above expression will compile to SQL as the expression ``true``
        or ``1 = 1``, depending on backend, if no other expressions are
        present.  If expressions are present, then the :func:`_sql.true` value
        is ignored as it does not affect the outcome of an AND expression that
        has other elements.

        .. deprecated:: 1.4  The :func:`.and_` element now requires that at
          least one argument is passed; creating the :func:`.and_` construct
          with no arguments is deprecated, and will emit a deprecation warning
          while continuing to produce a blank SQL string.

        .. seealso::

            :func:`.or_`

        """  # noqa: E501
        return BooleanClauseList.and_(*clauses)


def any_(expr: _ColumnExpressionArgument[_T]) -> CollectionAggregate[bool]:
    """Produce an ANY expression.

    For dialects such as that of PostgreSQL, this operator applies
    to usage of the :class:`_types.ARRAY` datatype, for that of
    MySQL, it may apply to a subquery.  e.g.::

        # renders on PostgreSQL:
        # '5 = ANY (somearray)'
        expr = 5 == any_(mytable.c.somearray)

        # renders on MySQL:
        # '5 = ANY (SELECT value FROM table)'
        expr = 5 == any_(select(table.c.value))

    Comparison to NULL may work using ``None`` or :func:`_sql.null`::

        None == any_(mytable.c.somearray)

    The any_() / all_() operators also feature a special "operand flipping"
    behavior such that if any_() / all_() are used on the left side of a
    comparison using a standalone operator such as ``==``, ``!=``, etc.
    (not including operator methods such as
    :meth:`_sql.ColumnOperators.is_`) the rendered expression is flipped::

        # would render '5 = ANY (column)`
        any_(mytable.c.column) == 5

    Or with ``None``, which note will not perform
    the usual step of rendering "IS" as is normally the case for NULL::

        # would render 'NULL = ANY(somearray)'
        any_(mytable.c.somearray) == None

    .. versionchanged:: 1.4.26  repaired the use of any_() / all_()
       comparing to NULL on the right side to be flipped to the left.

    The column-level :meth:`_sql.ColumnElement.any_` method (not to be
    confused with :class:`_types.ARRAY` level
    :meth:`_types.ARRAY.Comparator.any`) is shorthand for
    ``any_(col)``::

        5 = mytable.c.somearray.any_()

    .. seealso::

        :meth:`_sql.ColumnOperators.any_`

        :func:`_expression.all_`

    """
    return CollectionAggregate._create_any(expr)


def asc(
    column: _ColumnExpressionOrStrLabelArgument[_T],
) -> UnaryExpression[_T]:
    """Produce an ascending ``ORDER BY`` clause element.

    e.g.::

        from sqlalchemy import asc

        stmt = select(users_table).order_by(asc(users_table.c.name))

    will produce SQL as:

    .. sourcecode:: sql

        SELECT id, name FROM user ORDER BY name ASC

    The :func:`.asc` function is a standalone version of the
    :meth:`_expression.ColumnElement.asc`
    method available on all SQL expressions,
    e.g.::


        stmt = select(users_table).order_by(users_table.c.name.asc())

    :param column: A :class:`_expression.ColumnElement` (e.g.
     scalar SQL expression)
     with which to apply the :func:`.asc` operation.

    .. seealso::

        :func:`.desc`

        :func:`.nulls_first`

        :func:`.nulls_last`

        :meth:`_expression.Select.order_by`

    """
    return UnaryExpression._create_asc(column)


def collate(
    expression: _ColumnExpressionArgument[str], collation: str
) -> BinaryExpression[str]:
    """Return the clause ``expression COLLATE collation``.

    e.g.::

        collate(mycolumn, "utf8_bin")

    produces:

    .. sourcecode:: sql

        mycolumn COLLATE utf8_bin

    The collation expression is also quoted if it is a case sensitive
    identifier, e.g. contains uppercase characters.

    .. versionchanged:: 1.2 quoting is automatically applied to COLLATE
       expressions if they are case sensitive.

    """
    return CollationClause._create_collation_expression(expression, collation)


def between(
    expr: _ColumnExpressionOrLiteralArgument[_T],
    lower_bound: Any,
    upper_bound: Any,
    symmetric: bool = False,
) -> BinaryExpression[bool]:
    """Produce a ``BETWEEN`` predicate clause.

    E.g.::

        from sqlalchemy import between

        stmt = select(users_table).where(between(users_table.c.id, 5, 7))

    Would produce SQL resembling:

    .. sourcecode:: sql

        SELECT id, name FROM user WHERE id BETWEEN :id_1 AND :id_2

    The :func:`.between` function is a standalone version of the
    :meth:`_expression.ColumnElement.between` method available on all
    SQL expressions, as in::

        stmt = select(users_table).where(users_table.c.id.between(5, 7))

    All arguments passed to :func:`.between`, including the left side
    column expression, are coerced from Python scalar values if a
    the value is not a :class:`_expression.ColumnElement` subclass.
    For example,
    three fixed values can be compared as in::

        print(between(5, 3, 7))

    Which would produce::

        :param_1 BETWEEN :param_2 AND :param_3

    :param expr: a column expression, typically a
     :class:`_expression.ColumnElement`
     instance or alternatively a Python scalar expression to be coerced
     into a column expression, serving as the left side of the ``BETWEEN``
     expression.

    :param lower_bound: a column or Python scalar expression serving as the
     lower bound of the right side of the ``BETWEEN`` expression.

    :param upper_bound: a column or Python scalar expression serving as the
     upper bound of the right side of the ``BETWEEN`` expression.

    :param symmetric: if True, will render " BETWEEN SYMMETRIC ". Note
     that not all databases support this syntax.

    .. seealso::

        :meth:`_expression.ColumnElement.between`

    """
    col_expr = coercions.expect(roles.ExpressionElementRole, expr)
    return col_expr.between(lower_bound, upper_bound, symmetric=symmetric)


def outparam(
    key: str, type_: Optional[TypeEngine[_T]] = None
) -> BindParameter[_T]:
    """Create an 'OUT' parameter for usage in functions (stored procedures),
    for databases which support them.

    The ``outparam`` can be used like a regular function parameter.
    The "output" value will be available from the
    :class:`~sqlalchemy.engine.CursorResult` object via its ``out_parameters``
    attribute, which returns a dictionary containing the values.

    """
    return BindParameter(key, None, type_=type_, unique=False, isoutparam=True)


@overload
def not_(clause: BinaryExpression[_T]) -> BinaryExpression[_T]: ...


@overload
def not_(clause: _ColumnExpressionArgument[_T]) -> ColumnElement[_T]: ...


def not_(clause: _ColumnExpressionArgument[_T]) -> ColumnElement[_T]:
    """Return a negation of the given clause, i.e. ``NOT(clause)``.

    The ``~`` operator is also overloaded on all
    :class:`_expression.ColumnElement` subclasses to produce the
    same result.

    """

    return coercions.expect(roles.ExpressionElementRole, clause).__invert__()


def bindparam(
    key: Optional[str],
    value: Any = _NoArg.NO_ARG,
    type_: Optional[_TypeEngineArgument[_T]] = None,
    unique: bool = False,
    required: Union[bool, Literal[_NoArg.NO_ARG]] = _NoArg.NO_ARG,
    quote: Optional[bool] = None,
    callable_: Optional[Callable[[], Any]] = None,
    expanding: bool = False,
    isoutparam: bool = False,
    literal_execute: bool = False,
) -> BindParameter[_T]:
    r"""Produce a "bound expression".

    The return value is an instance of :class:`.BindParameter`; this
    is a :class:`_expression.ColumnElement`
    subclass which represents a so-called
    "placeholder" value in a SQL expression, the value of which is
    supplied at the point at which the statement in executed against a
    database connection.

    In SQLAlchemy, the :func:`.bindparam` construct has
    the ability to carry along the actual value that will be ultimately
    used at expression time.  In this way, it serves not just as
    a "placeholder" for eventual population, but also as a means of
    representing so-called "unsafe" values which should not be rendered
    directly in a SQL statement, but rather should be passed along
    to the :term:`DBAPI` as values which need to be correctly escaped
    and potentially handled for type-safety.

    When using :func:`.bindparam` explicitly, the use case is typically
    one of traditional deferment of parameters; the :func:`.bindparam`
    construct accepts a name which can then be referred to at execution
    time::

        from sqlalchemy import bindparam

        stmt = select(users_table).where(
            users_table.c.name == bindparam("username")
        )

    The above statement, when rendered, will produce SQL similar to:

    .. sourcecode:: sql

        SELECT id, name FROM user WHERE name = :username

    In order to populate the value of ``:username`` above, the value
    would typically be applied at execution time to a method
    like :meth:`_engine.Connection.execute`::

        result = connection.execute(stmt, {"username": "wendy"})

    Explicit use of :func:`.bindparam` is also common when producing
    UPDATE or DELETE statements that are to be invoked multiple times,
    where the WHERE criterion of the statement is to change on each
    invocation, such as::

        stmt = (
            users_table.update()
            .where(user_table.c.name == bindparam("username"))
            .values(fullname=bindparam("fullname"))
        )

        connection.execute(
            stmt,
            [
                {"username": "wendy", "fullname": "Wendy Smith"},
                {"username": "jack", "fullname": "Jack Jones"},
            ],
        )

    SQLAlchemy's Core expression system makes wide use of
    :func:`.bindparam` in an implicit sense.   It is typical that Python
    literal values passed to virtually all SQL expression functions are
    coerced into fixed :func:`.bindparam` constructs.  For example, given
    a comparison operation such as::

        expr = users_table.c.name == "Wendy"

    The above expression will produce a :class:`.BinaryExpression`
    construct, where the left side is the :class:`_schema.Column` object
    representing the ``name`` column, and the right side is a
    :class:`.BindParameter` representing the literal value::

        print(repr(expr.right))
        BindParameter("%(4327771088 name)s", "Wendy", type_=String())

    The expression above will render SQL such as:

    .. sourcecode:: sql

        user.name = :name_1

    Where the ``:name_1`` parameter name is an anonymous name.  The
    actual string ``Wendy`` is not in the rendered string, but is carried
    along where it is later used within statement execution.  If we
    invoke a statement like the following::

        stmt = select(users_table).where(users_table.c.name == "Wendy")
        result = connection.execute(stmt)

    We would see SQL logging output as:

    .. sourcecode:: sql

        SELECT "user".id, "user".name
        FROM "user"
        WHERE "user".name = %(name_1)s
        {'name_1': 'Wendy'}

    Above, we see that ``Wendy`` is passed as a parameter to the database,
    while the placeholder ``:name_1`` is rendered in the appropriate form
    for the target database, in this case the PostgreSQL database.

    Similarly, :func:`.bindparam` is invoked automatically when working
    with :term:`CRUD` statements as far as the "VALUES" portion is
    concerned.   The :func:`_expression.insert` construct produces an
    ``INSERT`` expression which will, at statement execution time, generate
    bound placeholders based on the arguments passed, as in::

        stmt = users_table.insert()
        result = connection.execute(stmt, {"name": "Wendy"})

    The above will produce SQL output as:

    .. sourcecode:: sql

        INSERT INTO "user" (name) VALUES (%(name)s)
        {'name': 'Wendy'}

    The :class:`_expression.Insert` construct, at
    compilation/execution time, rendered a single :func:`.bindparam`
    mirroring the column name ``name`` as a result of the single ``name``
    parameter we passed to the :meth:`_engine.Connection.execute` method.

    :param key:
      the key (e.g. the name) for this bind param.
      Will be used in the generated
      SQL statement for dialects that use named parameters.  This
      value may be modified when part of a compilation operation,
      if other :class:`BindParameter` objects exist with the same
      key, or if its length is too long and truncation is
      required.

      If omitted, an "anonymous" name is generated for the bound parameter;
      when given a value to bind, the end result is equivalent to calling upon
      the :func:`.literal` function with a value to bind, particularly
      if the :paramref:`.bindparam.unique` parameter is also provided.

    :param value:
      Initial value for this bind param.  Will be used at statement
      execution time as the value for this parameter passed to the
      DBAPI, if no other value is indicated to the statement execution
      method for this particular parameter name.  Defaults to ``None``.

    :param callable\_:
      A callable function that takes the place of "value".  The function
      will be called at statement execution time to determine the
      ultimate value.   Used for scenarios where the actual bind
      value cannot be determined at the point at which the clause
      construct is created, but embedded bind values are still desirable.

    :param type\_:
      A :class:`.TypeEngine` class or instance representing an optional
      datatype for this :func:`.bindparam`.  If not passed, a type
      may be determined automatically for the bind, based on the given
      value; for example, trivial Python types such as ``str``,
      ``int``, ``bool``
      may result in the :class:`.String`, :class:`.Integer` or
      :class:`.Boolean` types being automatically selected.

      The type of a :func:`.bindparam` is significant especially in that
      the type will apply pre-processing to the value before it is
      passed to the database.  For example, a :func:`.bindparam` which
      refers to a datetime value, and is specified as holding the
      :class:`.DateTime` type, may apply conversion needed to the
      value (such as stringification on SQLite) before passing the value
      to the database.

    :param unique:
      if True, the key name of this :class:`.BindParameter` will be
      modified if another :class:`.BindParameter` of the same name
      already has been located within the containing
      expression.  This flag is used generally by the internals
      when producing so-called "anonymous" bound expressions, it
      isn't generally applicable to explicitly-named :func:`.bindparam`
      constructs.

    :param required:
      If ``True``, a value is required at execution time.  If not passed,
      it defaults to ``True`` if neither :paramref:`.bindparam.value`
      or :paramref:`.bindparam.callable` were passed.  If either of these
      parameters are present, then :paramref:`.bindparam.required`
      defaults to ``False``.

    :param quote:
      True if this parameter name requires quoting and is not
      currently known as a SQLAlchemy reserved word; this currently
      only applies to the Oracle Database backends, where bound names must
      sometimes be quoted.

    :param isoutparam:
      if True, the parameter should be treated like a stored procedure
      "OUT" parameter.  This applies to backends such as Oracle Database which
      support OUT parameters.

    :param expanding:
      if True, this parameter will be treated as an "expanding" parameter
      at execution time; the parameter value is expected to be a sequence,
      rather than a scalar value, and the string SQL statement will
      be transformed on a per-execution basis to accommodate the sequence
      with a variable number of parameter slots passed to the DBAPI.
      This is to allow statement caching to be used in conjunction with
      an IN clause.

      .. seealso::

        :meth:`.ColumnOperators.in_`

        :ref:`baked_in` - with baked queries

      .. note:: The "expanding" feature does not support "executemany"-
         style parameter sets.

      .. versionadded:: 1.2

      .. versionchanged:: 1.3 the "expanding" bound parameter feature now
         supports empty lists.

    :param literal_execute:
      if True, the bound parameter will be rendered in the compile phase
      with a special "POSTCOMPILE" token, and the SQLAlchemy compiler will
      render the final value of the parameter into the SQL statement at
      statement execution time, omitting the value from the parameter
      dictionary / list passed to DBAPI ``cursor.execute()``.  This
      produces a similar effect as that of using the ``literal_binds``,
      compilation flag,  however takes place as the statement is sent to
      the DBAPI ``cursor.execute()`` method, rather than when the statement
      is compiled.   The primary use of this
      capability is for rendering LIMIT / OFFSET clauses for database
      drivers that can't accommodate for bound parameters in these
      contexts, while allowing SQL constructs to be cacheable at the
      compilation level.

      .. versionadded:: 1.4 Added "post compile" bound parameters

        .. seealso::

            :ref:`change_4808`.

    .. seealso::

        :ref:`tutorial_sending_parameters` - in the
        :ref:`unified_tutorial`


    """
    return BindParameter(
        key,
        value,
        type_,
        unique,
        required,
        quote,
        callable_,
        expanding,
        isoutparam,
        literal_execute,
    )


def case(
    *whens: Union[
        typing_Tuple[_ColumnExpressionArgument[bool], Any], Mapping[Any, Any]
    ],
    value: Optional[Any] = None,
    else_: Optional[Any] = None,
) -> Case[Any]:
    r"""Produce a ``CASE`` expression.

    The ``CASE`` construct in SQL is a conditional object that
    acts somewhat analogously to an "if/then" construct in other
    languages.  It returns an instance of :class:`.Case`.

    :func:`.case` in its usual form is passed a series of "when"
    constructs, that is, a list of conditions and results as tuples::

        from sqlalchemy import case

        stmt = select(users_table).where(
            case(
                (users_table.c.name == "wendy", "W"),
                (users_table.c.name == "jack", "J"),
                else_="E",
            )
        )

    The above statement will produce SQL resembling:

    .. sourcecode:: sql

        SELECT id, name FROM user
        WHERE CASE
            WHEN (name = :name_1) THEN :param_1
            WHEN (name = :name_2) THEN :param_2
            ELSE :param_3
        END

    When simple equality expressions of several values against a single
    parent column are needed, :func:`.case` also has a "shorthand" format
    used via the
    :paramref:`.case.value` parameter, which is passed a column
    expression to be compared.  In this form, the :paramref:`.case.whens`
    parameter is passed as a dictionary containing expressions to be
    compared against keyed to result expressions.  The statement below is
    equivalent to the preceding statement::

        stmt = select(users_table).where(
            case({"wendy": "W", "jack": "J"}, value=users_table.c.name, else_="E")
        )

    The values which are accepted as result values in
    :paramref:`.case.whens` as well as with :paramref:`.case.else_` are
    coerced from Python literals into :func:`.bindparam` constructs.
    SQL expressions, e.g. :class:`_expression.ColumnElement` constructs,
    are accepted
    as well.  To coerce a literal string expression into a constant
    expression rendered inline, use the :func:`_expression.literal_column`
    construct,
    as in::

        from sqlalchemy import case, literal_column

        case(
            (orderline.c.qty > 100, literal_column("'greaterthan100'")),
            (orderline.c.qty > 10, literal_column("'greaterthan10'")),
            else_=literal_column("'lessthan10'"),
        )

    The above will render the given constants without using bound
    parameters for the result values (but still for the comparison
    values), as in:

    .. sourcecode:: sql

        CASE
            WHEN (orderline.qty > :qty_1) THEN 'greaterthan100'
            WHEN (orderline.qty > :qty_2) THEN 'greaterthan10'
            ELSE 'lessthan10'
        END

    :param \*whens: The criteria to be compared against,
     :paramref:`.case.whens` accepts two different forms, based on
     whether or not :paramref:`.case.value` is used.

     .. versionchanged:: 1.4 the :func:`_sql.case`
        function now accepts the series of WHEN conditions positionally

     In the first form, it accepts multiple 2-tuples passed as positional
     arguments; each 2-tuple consists of ``(<sql expression>, <value>)``,
     where the SQL expression is a boolean expression and "value" is a
     resulting value, e.g.::

        case(
            (users_table.c.name == "wendy", "W"),
            (users_table.c.name == "jack", "J"),
        )

     In the second form, it accepts a Python dictionary of comparison
     values mapped to a resulting value; this form requires
     :paramref:`.case.value` to be present, and values will be compared
     using the ``==`` operator, e.g.::

        case({"wendy": "W", "jack": "J"}, value=users_table.c.name)

    :param value: An optional SQL expression which will be used as a
      fixed "comparison point" for candidate values within a dictionary
      passed to :paramref:`.case.whens`.

    :param else\_: An optional SQL expression which will be the evaluated
      result of the ``CASE`` construct if all expressions within
      :paramref:`.case.whens` evaluate to false.  When omitted, most
      databases will produce a result of NULL if none of the "when"
      expressions evaluate to true.


    """  # noqa: E501
    return Case(*whens, value=value, else_=else_)


def cast(
    expression: _ColumnExpressionOrLiteralArgument[Any],
    type_: _TypeEngineArgument[_T],
) -> Cast[_T]:
    r"""Produce a ``CAST`` expression.

    :func:`.cast` returns an instance of :class:`.Cast`.

    E.g.::

        from sqlalchemy import cast, Numeric

        stmt = select(cast(product_table.c.unit_price, Numeric(10, 4)))

    The above statement will produce SQL resembling:

    .. sourcecode:: sql

        SELECT CAST(unit_price AS NUMERIC(10, 4)) FROM product

    The :func:`.cast` function performs two distinct functions when
    used.  The first is that it renders the ``CAST`` expression within
    the resulting SQL string.  The second is that it associates the given
    type (e.g. :class:`.TypeEngine` class or instance) with the column
    expression on the Python side, which means the expression will take
    on the expression operator behavior associated with that type,
    as well as the bound-value handling and result-row-handling behavior
    of the type.

    An alternative to :func:`.cast` is the :func:`.type_coerce` function.
    This function performs the second task of associating an expression
    with a specific type, but does not render the ``CAST`` expression
    in SQL.

    :param expression: A SQL expression, such as a
     :class:`_expression.ColumnElement`
     expression or a Python string which will be coerced into a bound
     literal value.

    :param type\_: A :class:`.TypeEngine` class or instance indicating
     the type to which the ``CAST`` should apply.

    .. seealso::

        :ref:`tutorial_casts`

        :func:`.try_cast` - an alternative to CAST that results in
        NULLs when the cast fails, instead of raising an error.
        Only supported by some dialects.

        :func:`.type_coerce` - an alternative to CAST that coerces the type
        on the Python side only, which is often sufficient to generate the
        correct SQL and data coercion.


    """
    return Cast(expression, type_)


def try_cast(
    expression: _ColumnExpressionOrLiteralArgument[Any],
    type_: _TypeEngineArgument[_T],
) -> TryCast[_T]:
    """Produce a ``TRY_CAST`` expression for backends which support it;
    this is a ``CAST`` which returns NULL for un-castable conversions.

    In SQLAlchemy, this construct is supported **only** by the SQL Server
    dialect, and will raise a :class:`.CompileError` if used on other
    included backends.  However, third party backends may also support
    this construct.

    .. tip:: As :func:`_sql.try_cast` originates from the SQL Server dialect,
       it's importable both from ``sqlalchemy.`` as well as from
       ``sqlalchemy.dialects.mssql``.

    :func:`_sql.try_cast` returns an instance of :class:`.TryCast` and
    generally behaves similarly to the :class:`.Cast` construct;
    at the SQL level, the difference between ``CAST`` and ``TRY_CAST``
    is that ``TRY_CAST`` returns NULL for an un-castable expression,
    such as attempting to cast a string ``"hi"`` to an integer value.

    E.g.::

        from sqlalchemy import select, try_cast, Numeric

        stmt = select(try_cast(product_table.c.unit_price, Numeric(10, 4)))

    The above would render on Microsoft SQL Server as:

    .. sourcecode:: sql

        SELECT TRY_CAST (product_table.unit_price AS NUMERIC(10, 4))
        FROM product_table

    .. versionadded:: 2.0.14  :func:`.try_cast` has been
       generalized from the SQL Server dialect into a general use
       construct that may be supported by additional dialects.

    """
    return TryCast(expression, type_)


def column(
    text: str,
    type_: Optional[_TypeEngineArgument[_T]] = None,
    is_literal: bool = False,
    _selectable: Optional[FromClause] = None,
) -> ColumnClause[_T]:
    """Produce a :class:`.ColumnClause` object.

    The :class:`.ColumnClause` is a lightweight analogue to the
    :class:`_schema.Column` class.  The :func:`_expression.column`
    function can
    be invoked with just a name alone, as in::

        from sqlalchemy import column

        id, name = column("id"), column("name")
        stmt = select(id, name).select_from("user")

    The above statement would produce SQL like:

    .. sourcecode:: sql

        SELECT id, name FROM user

    Once constructed, :func:`_expression.column`
    may be used like any other SQL
    expression element such as within :func:`_expression.select`
    constructs::

        from sqlalchemy.sql import column

        id, name = column("id"), column("name")
        stmt = select(id, name).select_from("user")

    The text handled by :func:`_expression.column`
    is assumed to be handled
    like the name of a database column; if the string contains mixed case,
    special characters, or matches a known reserved word on the target
    backend, the column expression will render using the quoting
    behavior determined by the backend.  To produce a textual SQL
    expression that is rendered exactly without any quoting,
    use :func:`_expression.literal_column` instead,
    or pass ``True`` as the
    value of :paramref:`_expression.column.is_literal`.   Additionally,
    full SQL
    statements are best handled using the :func:`_expression.text`
    construct.

    :func:`_expression.column` can be used in a table-like
    fashion by combining it with the :func:`.table` function
    (which is the lightweight analogue to :class:`_schema.Table`
    ) to produce
    a working table construct with minimal boilerplate::

        from sqlalchemy import table, column, select

        user = table(
            "user",
            column("id"),
            column("name"),
            column("description"),
        )

        stmt = select(user.c.description).where(user.c.name == "wendy")

    A :func:`_expression.column` / :func:`.table`
    construct like that illustrated
    above can be created in an
    ad-hoc fashion and is not associated with any
    :class:`_schema.MetaData`, DDL, or events, unlike its
    :class:`_schema.Table` counterpart.

    :param text: the text of the element.

    :param type: :class:`_types.TypeEngine` object which can associate
      this :class:`.ColumnClause` with a type.

    :param is_literal: if True, the :class:`.ColumnClause` is assumed to
      be an exact expression that will be delivered to the output with no
      quoting rules applied regardless of case sensitive settings. the
      :func:`_expression.literal_column()` function essentially invokes
      :func:`_expression.column` while passing ``is_literal=True``.

    .. seealso::

        :class:`_schema.Column`

        :func:`_expression.literal_column`

        :func:`.table`

        :func:`_expression.text`

        :ref:`tutorial_select_arbitrary_text`

    """
    return ColumnClause(text, type_, is_literal, _selectable)


def desc(
    column: _ColumnExpressionOrStrLabelArgument[_T],
) -> UnaryExpression[_T]:
    """Produce a descending ``ORDER BY`` clause element.

    e.g.::

        from sqlalchemy import desc

        stmt = select(users_table).order_by(desc(users_table.c.name))

    will produce SQL as:

    .. sourcecode:: sql

        SELECT id, name FROM user ORDER BY name DESC

    The :func:`.desc` function is a standalone version of the
    :meth:`_expression.ColumnElement.desc`
    method available on all SQL expressions,
    e.g.::


        stmt = select(users_table).order_by(users_table.c.name.desc())

    :param column: A :class:`_expression.ColumnElement` (e.g.
     scalar SQL expression)
     with which to apply the :func:`.desc` operation.

    .. seealso::

        :func:`.asc`

        :func:`.nulls_first`

        :func:`.nulls_last`

        :meth:`_expression.Select.order_by`

    """
    return UnaryExpression._create_desc(column)


def distinct(expr: _ColumnExpressionArgument[_T]) -> UnaryExpression[_T]:
    """Produce an column-expression-level unary ``DISTINCT`` clause.

    This applies the ``DISTINCT`` keyword to an **individual column
    expression** (e.g. not the whole statement), and renders **specifically
    in that column position**; this is used for containment within
    an aggregate function, as in::

        from sqlalchemy import distinct, func

        stmt = select(users_table.c.id, func.count(distinct(users_table.c.name)))

    The above would produce an statement resembling:

    .. sourcecode:: sql

        SELECT user.id, count(DISTINCT user.name) FROM user

    .. tip:: The :func:`_sql.distinct` function does **not** apply DISTINCT
       to the full SELECT statement, instead applying a DISTINCT modifier
       to **individual column expressions**.  For general ``SELECT DISTINCT``
       support, use the
       :meth:`_sql.Select.distinct` method on :class:`_sql.Select`.

    The :func:`.distinct` function is also available as a column-level
    method, e.g. :meth:`_expression.ColumnElement.distinct`, as in::

        stmt = select(func.count(users_table.c.name.distinct()))

    The :func:`.distinct` operator is different from the
    :meth:`_expression.Select.distinct` method of
    :class:`_expression.Select`,
    which produces a ``SELECT`` statement
    with ``DISTINCT`` applied to the result set as a whole,
    e.g. a ``SELECT DISTINCT`` expression.  See that method for further
    information.

    .. seealso::

        :meth:`_expression.ColumnElement.distinct`

        :meth:`_expression.Select.distinct`

        :data:`.func`

    """  # noqa: E501
    return UnaryExpression._create_distinct(expr)


def bitwise_not(expr: _ColumnExpressionArgument[_T]) -> UnaryExpression[_T]:
    """Produce a unary bitwise NOT clause, typically via the ``~`` operator.

    Not to be confused with boolean negation :func:`_sql.not_`.

    .. versionadded:: 2.0.2

    .. seealso::

        :ref:`operators_bitwise`


    """

    return UnaryExpression._create_bitwise_not(expr)


def extract(field: str, expr: _ColumnExpressionArgument[Any]) -> Extract:
    """Return a :class:`.Extract` construct.

    This is typically available as :func:`.extract`
    as well as ``func.extract`` from the
    :data:`.func` namespace.

    :param field: The field to extract.

     .. warning:: This field is used as a literal SQL string.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.

    :param expr: A column or Python scalar expression serving as the
      right side of the ``EXTRACT`` expression.

    E.g.::

        from sqlalchemy import extract
        from sqlalchemy import table, column

        logged_table = table(
            "user",
            column("id"),
            column("date_created"),
        )

        stmt = select(logged_table.c.id).where(
            extract("YEAR", logged_table.c.date_created) == 2021
        )

    In the above example, the statement is used to select ids from the
    database where the ``YEAR`` component matches a specific value.

    Similarly, one can also select an extracted component::

        stmt = select(extract("YEAR", logged_table.c.date_created)).where(
            logged_table.c.id == 1
        )

    The implementation of ``EXTRACT`` may vary across database backends.
    Users are reminded to consult their database documentation.
    """
    return Extract(field, expr)


def false() -> False_:
    """Return a :class:`.False_` construct.

    E.g.:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy import false
        >>> print(select(t.c.x).where(false()))
        {printsql}SELECT x FROM t WHERE false

    A backend which does not support true/false constants will render as
    an expression against 1 or 0:

    .. sourcecode:: pycon+sql

        >>> print(select(t.c.x).where(false()))
        {printsql}SELECT x FROM t WHERE 0 = 1

    The :func:`.true` and :func:`.false` constants also feature
    "short circuit" operation within an :func:`.and_` or :func:`.or_`
    conjunction:

    .. sourcecode:: pycon+sql

        >>> print(select(t.c.x).where(or_(t.c.x > 5, true())))
        {printsql}SELECT x FROM t WHERE true{stop}

        >>> print(select(t.c.x).where(and_(t.c.x > 5, false())))
        {printsql}SELECT x FROM t WHERE false{stop}

    .. seealso::

        :func:`.true`

    """

    return False_._instance()


def funcfilter(
    func: FunctionElement[_T], *criterion: _ColumnExpressionArgument[bool]
) -> FunctionFilter[_T]:
    """Produce a :class:`.FunctionFilter` object against a function.

    Used against aggregate and window functions,
    for database backends that support the "FILTER" clause.

    E.g.::

        from sqlalchemy import funcfilter

        funcfilter(func.count(1), MyClass.name == "some name")

    Would produce "COUNT(1) FILTER (WHERE myclass.name = 'some name')".

    This function is also available from the :data:`~.expression.func`
    construct itself via the :meth:`.FunctionElement.filter` method.

    .. seealso::

        :ref:`tutorial_functions_within_group` - in the
        :ref:`unified_tutorial`

        :meth:`.FunctionElement.filter`

    """
    return FunctionFilter(func, *criterion)


def label(
    name: str,
    element: _ColumnExpressionArgument[_T],
    type_: Optional[_TypeEngineArgument[_T]] = None,
) -> Label[_T]:
    """Return a :class:`Label` object for the
    given :class:`_expression.ColumnElement`.

    A label changes the name of an element in the columns clause of a
    ``SELECT`` statement, typically via the ``AS`` SQL keyword.

    This functionality is more conveniently available via the
    :meth:`_expression.ColumnElement.label` method on
    :class:`_expression.ColumnElement`.

    :param name: label name

    :param obj: a :class:`_expression.ColumnElement`.

    """
    return Label(name, element, type_)


def null() -> Null:
    """Return a constant :class:`.Null` construct."""

    return Null._instance()


def nulls_first(column: _ColumnExpressionArgument[_T]) -> UnaryExpression[_T]:
    """Produce the ``NULLS FIRST`` modifier for an ``ORDER BY`` expression.

    :func:`.nulls_first` is intended to modify the expression produced
    by :func:`.asc` or :func:`.desc`, and indicates how NULL values
    should be handled when they are encountered during ordering::


        from sqlalchemy import desc, nulls_first

        stmt = select(users_table).order_by(nulls_first(desc(users_table.c.name)))

    The SQL expression from the above would resemble:

    .. sourcecode:: sql

        SELECT id, name FROM user ORDER BY name DESC NULLS FIRST

    Like :func:`.asc` and :func:`.desc`, :func:`.nulls_first` is typically
    invoked from the column expression itself using
    :meth:`_expression.ColumnElement.nulls_first`,
    rather than as its standalone
    function version, as in::

        stmt = select(users_table).order_by(
            users_table.c.name.desc().nulls_first()
        )

    .. versionchanged:: 1.4 :func:`.nulls_first` is renamed from
        :func:`.nullsfirst` in previous releases.
        The previous name remains available for backwards compatibility.

    .. seealso::

        :func:`.asc`

        :func:`.desc`

        :func:`.nulls_last`

        :meth:`_expression.Select.order_by`

    """  # noqa: E501
    return UnaryExpression._create_nulls_first(column)


def nulls_last(column: _ColumnExpressionArgument[_T]) -> UnaryExpression[_T]:
    """Produce the ``NULLS LAST`` modifier for an ``ORDER BY`` expression.

    :func:`.nulls_last` is intended to modify the expression produced
    by :func:`.asc` or :func:`.desc`, and indicates how NULL values
    should be handled when they are encountered during ordering::


        from sqlalchemy import desc, nulls_last

        stmt = select(users_table).order_by(nulls_last(desc(users_table.c.name)))

    The SQL expression from the above would resemble:

    .. sourcecode:: sql

        SELECT id, name FROM user ORDER BY name DESC NULLS LAST

    Like :func:`.asc` and :func:`.desc`, :func:`.nulls_last` is typically
    invoked from the column expression itself using
    :meth:`_expression.ColumnElement.nulls_last`,
    rather than as its standalone
    function version, as in::

        stmt = select(users_table).order_by(users_table.c.name.desc().nulls_last())

    .. versionchanged:: 1.4 :func:`.nulls_last` is renamed from
        :func:`.nullslast` in previous releases.
        The previous name remains available for backwards compatibility.

    .. seealso::

        :func:`.asc`

        :func:`.desc`

        :func:`.nulls_first`

        :meth:`_expression.Select.order_by`

    """  # noqa: E501
    return UnaryExpression._create_nulls_last(column)


def or_(  # type: ignore[empty-body]
    initial_clause: Union[Literal[False], _ColumnExpressionArgument[bool]],
    *clauses: _ColumnExpressionArgument[bool],
) -> ColumnElement[bool]:
    """Produce a conjunction of expressions joined by ``OR``.

    E.g.::

        from sqlalchemy import or_

        stmt = select(users_table).where(
            or_(users_table.c.name == "wendy", users_table.c.name == "jack")
        )

    The :func:`.or_` conjunction is also available using the
    Python ``|`` operator (though note that compound expressions
    need to be parenthesized in order to function with Python
    operator precedence behavior)::

        stmt = select(users_table).where(
            (users_table.c.name == "wendy") | (users_table.c.name == "jack")
        )

    The :func:`.or_` construct must be given at least one positional
    argument in order to be valid; a :func:`.or_` construct with no
    arguments is ambiguous.   To produce an "empty" or dynamically
    generated :func:`.or_`  expression, from a given list of expressions,
    a "default" element of :func:`_sql.false` (or just ``False``) should be
    specified::

        from sqlalchemy import false

        or_criteria = or_(false(), *expressions)

    The above expression will compile to SQL as the expression ``false``
    or ``0 = 1``, depending on backend, if no other expressions are
    present.  If expressions are present, then the :func:`_sql.false` value is
    ignored as it does not affect the outcome of an OR expression which
    has other elements.

    .. deprecated:: 1.4  The :func:`.or_` element now requires that at
       least one argument is passed; creating the :func:`.or_` construct
       with no arguments is deprecated, and will emit a deprecation warning
       while continuing to produce a blank SQL string.

    .. seealso::

        :func:`.and_`

    """
    ...


if not TYPE_CHECKING:
    # handle deprecated case which allows zero-arguments
    def or_(*clauses):  # noqa: F811
        """Produce a conjunction of expressions joined by ``OR``.

        E.g.::

            from sqlalchemy import or_

            stmt = select(users_table).where(
                or_(users_table.c.name == "wendy", users_table.c.name == "jack")
            )

        The :func:`.or_` conjunction is also available using the
        Python ``|`` operator (though note that compound expressions
        need to be parenthesized in order to function with Python
        operator precedence behavior)::

            stmt = select(users_table).where(
                (users_table.c.name == "wendy") | (users_table.c.name == "jack")
            )

        The :func:`.or_` construct must be given at least one positional
        argument in order to be valid; a :func:`.or_` construct with no
        arguments is ambiguous.   To produce an "empty" or dynamically
        generated :func:`.or_`  expression, from a given list of expressions,
        a "default" element of :func:`_sql.false` (or just ``False``) should be
        specified::

            from sqlalchemy import false

            or_criteria = or_(false(), *expressions)

        The above expression will compile to SQL as the expression ``false``
        or ``0 = 1``, depending on backend, if no other expressions are
        present.  If expressions are present, then the :func:`_sql.false` value
        is ignored as it does not affect the outcome of an OR expression which
        has other elements.

        .. deprecated:: 1.4  The :func:`.or_` element now requires that at
           least one argument is passed; creating the :func:`.or_` construct
           with no arguments is deprecated, and will emit a deprecation warning
           while continuing to produce a blank SQL string.

        .. seealso::

            :func:`.and_`

        """  # noqa: E501
        return BooleanClauseList.or_(*clauses)


def over(
    element: FunctionElement[_T],
    partition_by: Optional[_ByArgument] = None,
    order_by: Optional[_ByArgument] = None,
    range_: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
    rows: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
    groups: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
) -> Over[_T]:
    r"""Produce an :class:`.Over` object against a function.

    Used against aggregate or so-called "window" functions,
    for database backends that support window functions.

    :func:`_expression.over` is usually called using
    the :meth:`.FunctionElement.over` method, e.g.::

        func.row_number().over(order_by=mytable.c.some_column)

    Would produce:

    .. sourcecode:: sql

        ROW_NUMBER() OVER(ORDER BY some_column)

    Ranges are also possible using the :paramref:`.expression.over.range_`,
    :paramref:`.expression.over.rows`, and :paramref:`.expression.over.groups`
    parameters.  These
    mutually-exclusive parameters each accept a 2-tuple, which contains
    a combination of integers and None::

        func.row_number().over(order_by=my_table.c.some_column, range_=(None, 0))

    The above would produce:

    .. sourcecode:: sql

        ROW_NUMBER() OVER(ORDER BY some_column
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)

    A value of ``None`` indicates "unbounded", a
    value of zero indicates "current row", and negative / positive
    integers indicate "preceding" and "following":

    * RANGE BETWEEN 5 PRECEDING AND 10 FOLLOWING::

        func.row_number().over(order_by="x", range_=(-5, 10))

    * ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW::

        func.row_number().over(order_by="x", rows=(None, 0))

    * RANGE BETWEEN 2 PRECEDING AND UNBOUNDED FOLLOWING::

        func.row_number().over(order_by="x", range_=(-2, None))

    * RANGE BETWEEN 1 FOLLOWING AND 3 FOLLOWING::

        func.row_number().over(order_by="x", range_=(1, 3))

    * GROUPS BETWEEN 1 FOLLOWING AND 3 FOLLOWING::

        func.row_number().over(order_by="x", groups=(1, 3))

    :param element: a :class:`.FunctionElement`, :class:`.WithinGroup`,
     or other compatible construct.
    :param partition_by: a column element or string, or a list
     of such, that will be used as the PARTITION BY clause
     of the OVER construct.
    :param order_by: a column element or string, or a list
     of such, that will be used as the ORDER BY clause
     of the OVER construct.
    :param range\_: optional range clause for the window.  This is a
     tuple value which can contain integer values or ``None``,
     and will render a RANGE BETWEEN PRECEDING / FOLLOWING clause.
    :param rows: optional rows clause for the window.  This is a tuple
     value which can contain integer values or None, and will render
     a ROWS BETWEEN PRECEDING / FOLLOWING clause.
    :param groups: optional groups clause for the window.  This is a
     tuple value which can contain integer values or ``None``,
     and will render a GROUPS BETWEEN PRECEDING / FOLLOWING clause.

     .. versionadded:: 2.0.40

    This function is also available from the :data:`~.expression.func`
    construct itself via the :meth:`.FunctionElement.over` method.

    .. seealso::

        :ref:`tutorial_window_functions` - in the :ref:`unified_tutorial`

        :data:`.expression.func`

        :func:`_expression.within_group`

    """  # noqa: E501
    return Over(element, partition_by, order_by, range_, rows, groups)


@_document_text_coercion("text", ":func:`.text`", ":paramref:`.text.text`")
def text(text: str) -> TextClause:
    r"""Construct a new :class:`_expression.TextClause` clause,
    representing
    a textual SQL string directly.

    E.g.::

        from sqlalchemy import text

        t = text("SELECT * FROM users")
        result = connection.execute(t)

    The advantages :func:`_expression.text`
    provides over a plain string are
    backend-neutral support for bind parameters, per-statement
    execution options, as well as
    bind parameter and result-column typing behavior, allowing
    SQLAlchemy type constructs to play a role when executing
    a statement that is specified literally.  The construct can also
    be provided with a ``.c`` collection of column elements, allowing
    it to be embedded in other SQL expression constructs as a subquery.

    Bind parameters are specified by name, using the format ``:name``.
    E.g.::

        t = text("SELECT * FROM users WHERE id=:user_id")
        result = connection.execute(t, {"user_id": 12})

    For SQL statements where a colon is required verbatim, as within
    an inline string, use a backslash to escape::

        t = text(r"SELECT * FROM users WHERE name='\:username'")

    The :class:`_expression.TextClause`
    construct includes methods which can
    provide information about the bound parameters as well as the column
    values which would be returned from the textual statement, assuming
    it's an executable SELECT type of statement.  The
    :meth:`_expression.TextClause.bindparams`
    method is used to provide bound
    parameter detail, and :meth:`_expression.TextClause.columns`
    method allows
    specification of return columns including names and types::

        t = (
            text("SELECT * FROM users WHERE id=:user_id")
            .bindparams(user_id=7)
            .columns(id=Integer, name=String)
        )

        for id, name in connection.execute(t):
            print(id, name)

    The :func:`_expression.text` construct is used in cases when
    a literal string SQL fragment is specified as part of a larger query,
    such as for the WHERE clause of a SELECT statement::

        s = select(users.c.id, users.c.name).where(text("id=:user_id"))
        result = connection.execute(s, {"user_id": 12})

    :func:`_expression.text` is also used for the construction
    of a full, standalone statement using plain text.
    As such, SQLAlchemy refers
    to it as an :class:`.Executable` object and may be used
    like any other statement passed to an ``.execute()`` method.

    :param text:
      the text of the SQL statement to be created.  Use ``:<param>``
      to specify bind parameters; they will be compiled to their
      engine-specific format.

    .. seealso::

        :ref:`tutorial_select_arbitrary_text`

    """
    return TextClause(text)


def true() -> True_:
    """Return a constant :class:`.True_` construct.

    E.g.:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy import true
        >>> print(select(t.c.x).where(true()))
        {printsql}SELECT x FROM t WHERE true

    A backend which does not support true/false constants will render as
    an expression against 1 or 0:

    .. sourcecode:: pycon+sql

        >>> print(select(t.c.x).where(true()))
        {printsql}SELECT x FROM t WHERE 1 = 1

    The :func:`.true` and :func:`.false` constants also feature
    "short circuit" operation within an :func:`.and_` or :func:`.or_`
    conjunction:

    .. sourcecode:: pycon+sql

        >>> print(select(t.c.x).where(or_(t.c.x > 5, true())))
        {printsql}SELECT x FROM t WHERE true{stop}

        >>> print(select(t.c.x).where(and_(t.c.x > 5, false())))
        {printsql}SELECT x FROM t WHERE false{stop}

    .. seealso::

        :func:`.false`

    """

    return True_._instance()


def tuple_(
    *clauses: _ColumnExpressionArgument[Any],
    types: Optional[Sequence[_TypeEngineArgument[Any]]] = None,
) -> Tuple:
    """Return a :class:`.Tuple`.

    Main usage is to produce a composite IN construct using
    :meth:`.ColumnOperators.in_` ::

        from sqlalchemy import tuple_

        tuple_(table.c.col1, table.c.col2).in_([(1, 2), (5, 12), (10, 19)])

    .. versionchanged:: 1.3.6 Added support for SQLite IN tuples.

    .. warning::

        The composite IN construct is not supported by all backends, and is
        currently known to work on PostgreSQL, MySQL, and SQLite.
        Unsupported backends will raise a subclass of
        :class:`~sqlalchemy.exc.DBAPIError` when such an expression is
        invoked.

    """
    return Tuple(*clauses, types=types)


def type_coerce(
    expression: _ColumnExpressionOrLiteralArgument[Any],
    type_: _TypeEngineArgument[_T],
) -> TypeCoerce[_T]:
    r"""Associate a SQL expression with a particular type, without rendering
    ``CAST``.

    E.g.::

        from sqlalchemy import type_coerce

        stmt = select(type_coerce(log_table.date_string, StringDateTime()))

    The above construct will produce a :class:`.TypeCoerce` object, which
    does not modify the rendering in any way on the SQL side, with the
    possible exception of a generated label if used in a columns clause
    context:

    .. sourcecode:: sql

        SELECT date_string AS date_string FROM log

    When result rows are fetched, the ``StringDateTime`` type processor
    will be applied to result rows on behalf of the ``date_string`` column.

    .. note:: the :func:`.type_coerce` construct does not render any
       SQL syntax of its own, including that it does not imply
       parenthesization.   Please use :meth:`.TypeCoerce.self_group`
       if explicit parenthesization is required.

    In order to provide a named label for the expression, use
    :meth:`_expression.ColumnElement.label`::

        stmt = select(
            type_coerce(log_table.date_string, StringDateTime()).label("date")
        )

    A type that features bound-value handling will also have that behavior
    take effect when literal values or :func:`.bindparam` constructs are
    passed to :func:`.type_coerce` as targets.
    For example, if a type implements the
    :meth:`.TypeEngine.bind_expression`
    method or :meth:`.TypeEngine.bind_processor` method or equivalent,
    these functions will take effect at statement compilation/execution
    time when a literal value is passed, as in::

        # bound-value handling of MyStringType will be applied to the
        # literal value "some string"
        stmt = select(type_coerce("some string", MyStringType))

    When using :func:`.type_coerce` with composed expressions, note that
    **parenthesis are not applied**.   If :func:`.type_coerce` is being
    used in an operator context where the parenthesis normally present from
    CAST are necessary, use the :meth:`.TypeCoerce.self_group` method:

    .. sourcecode:: pycon+sql

        >>> some_integer = column("someint", Integer)
        >>> some_string = column("somestr", String)
        >>> expr = type_coerce(some_integer + 5, String) + some_string
        >>> print(expr)
        {printsql}someint + :someint_1 || somestr{stop}
        >>> expr = type_coerce(some_integer + 5, String).self_group() + some_string
        >>> print(expr)
        {printsql}(someint + :someint_1) || somestr{stop}

    :param expression: A SQL expression, such as a
     :class:`_expression.ColumnElement`
     expression or a Python string which will be coerced into a bound
     literal value.

    :param type\_: A :class:`.TypeEngine` class or instance indicating
     the type to which the expression is coerced.

    .. seealso::

        :ref:`tutorial_casts`

        :func:`.cast`

    """  # noqa
    return TypeCoerce(expression, type_)


def within_group(
    element: FunctionElement[_T], *order_by: _ColumnExpressionArgument[Any]
) -> WithinGroup[_T]:
    r"""Produce a :class:`.WithinGroup` object against a function.

    Used against so-called "ordered set aggregate" and "hypothetical
    set aggregate" functions, including :class:`.percentile_cont`,
    :class:`.rank`, :class:`.dense_rank`, etc.

    :func:`_expression.within_group` is usually called using
    the :meth:`.FunctionElement.within_group` method, e.g.::

        from sqlalchemy import within_group

        stmt = select(
            department.c.id,
            func.percentile_cont(0.5).within_group(department.c.salary.desc()),
        )

    The above statement would produce SQL similar to
    ``SELECT department.id, percentile_cont(0.5)
    WITHIN GROUP (ORDER BY department.salary DESC)``.

    :param element: a :class:`.FunctionElement` construct, typically
     generated by :data:`~.expression.func`.
    :param \*order_by: one or more column elements that will be used
     as the ORDER BY clause of the WITHIN GROUP construct.

    .. seealso::

        :ref:`tutorial_functions_within_group` - in the
        :ref:`unified_tutorial`

        :data:`.expression.func`

        :func:`_expression.over`

    """
    return WithinGroup(element, *order_by)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\formal.py ===
"""Formal Power Series"""

from collections import defaultdict

from sympy.core.numbers import (nan, oo, zoo)
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.sets.sets import Interval
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol
from sympy.core.sympify import sympify
from sympy.discrete.convolutions import convolution
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.combinatorial.numbers import bell
from sympy.functions.elementary.integers import floor, frac, ceiling
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import sequence
from sympy.series.series_class import SeriesBase
from sympy.utilities.iterables import iterable



def rational_algorithm(f, x, k, order=4, full=False):
    """
    Rational algorithm for computing
    formula of coefficients of Formal Power Series
    of a function.

    Explanation
    ===========

    Applicable when f(x) or some derivative of f(x)
    is a rational function in x.

    :func:`rational_algorithm` uses :func:`~.apart` function for partial fraction
    decomposition. :func:`~.apart` by default uses 'undetermined coefficients
    method'. By setting ``full=True``, 'Bronstein's algorithm' can be used
    instead.

    Looks for derivative of a function up to 4'th order (by default).
    This can be overridden using order option.

    Parameters
    ==========

    x : Symbol
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    full : bool

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int
    full : bool

    Examples
    ========

    >>> from sympy import log, atan
    >>> from sympy.series.formal import rational_algorithm as ra
    >>> from sympy.abc import x, k

    >>> ra(1 / (1 - x), x, k)
    (1, 0, 0)
    >>> ra(log(1 + x), x, k)
    (-1/((-1)**k*k), 0, 1)

    >>> ra(atan(x), x, k, full=True)
    ((-I/(2*(-I)**k) + I/(2*I**k))/k, 0, 1)

    Notes
    =====

    By setting ``full=True``, range of admissible functions to be solved using
    ``rational_algorithm`` can be increased. This option should be used
    carefully as it can significantly slow down the computation as ``doit`` is
    performed on the :class:`~.RootSum` object returned by the :func:`~.apart`
    function. Use ``full=False`` whenever possible.

    See Also
    ========

    sympy.polys.partfrac.apart

    References
    ==========

    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf
    .. [2] Power Series in Computer Algebra - Wolfram Koepf

    """
    from sympy.polys import RootSum, apart
    from sympy.integrals import integrate

    diff = f
    ds = []  # list of diff

    for i in range(order + 1):
        if i:
            diff = diff.diff(x)

        if diff.is_rational_function(x):
            coeff, sep = S.Zero, S.Zero

            terms = apart(diff, x, full=full)
            if terms.has(RootSum):
                terms = terms.doit()

            for t in Add.make_args(terms):
                num, den = t.as_numer_denom()
                if not den.has(x):
                    sep += t
                else:
                    if isinstance(den, Mul):
                        # m*(n*x - a)**j -> (n*x - a)**j
                        ind = den.as_independent(x)
                        den = ind[1]
                        num /= ind[0]

                    # (n*x - a)**j -> (x - b)
                    den, j = den.as_base_exp()
                    a, xterm = den.as_coeff_add(x)

                    # term -> m/x**n
                    if not a:
                        sep += t
                        continue

                    xc = xterm[0].coeff(x)
                    a /= -xc
                    num /= xc**j

                    ak = ((-1)**j * num *
                          binomial(j + k - 1, k).rewrite(factorial) /
                          a**(j + k))
                    coeff += ak

            # Hacky, better way?
            if coeff.is_zero:
                return None
            if (coeff.has(x) or coeff.has(zoo) or coeff.has(oo) or
                    coeff.has(nan)):
                return None

            for j in range(i):
                coeff = (coeff / (k + j + 1))
                sep = integrate(sep, x)
                sep += (ds.pop() - sep).limit(x, 0)  # constant of integration
            return (coeff.subs(k, k - i), sep, i)

        else:
            ds.append(diff)

    return None


def rational_independent(terms, x):
    """
    Returns a list of all the rationally independent terms.

    Examples
    ========

    >>> from sympy import sin, cos
    >>> from sympy.series.formal import rational_independent
    >>> from sympy.abc import x

    >>> rational_independent([cos(x), sin(x)], x)
    [cos(x), sin(x)]
    >>> rational_independent([x**2, sin(x), x*sin(x), x**3], x)
    [x**3 + x**2, x*sin(x) + sin(x)]
    """
    if not terms:
        return []

    ind = terms[0:1]

    for t in terms[1:]:
        n = t.as_independent(x)[1]
        for i, term in enumerate(ind):
            d = term.as_independent(x)[1]
            q = (n / d).cancel()
            if q.is_rational_function(x):
                ind[i] += t
                break
        else:
            ind.append(t)
    return ind


def simpleDE(f, x, g, order=4):
    r"""
    Generates simple DE.

    Explanation
    ===========

    DE is of the form

    .. math::
        f^k(x) + \sum\limits_{j=0}^{k-1} A_j f^j(x) = 0

    where :math:`A_j` should be rational function in x.

    Generates DE's upto order 4 (default). DE's can also have free parameters.

    By increasing order, higher order DE's can be found.

    Yields a tuple of (DE, order).
    """
    from sympy.solvers.solveset import linsolve

    a = symbols('a:%d' % (order))

    def _makeDE(k):
        eq = f.diff(x, k) + Add(*[a[i]*f.diff(x, i) for i in range(0, k)])
        DE = g(x).diff(x, k) + Add(*[a[i]*g(x).diff(x, i) for i in range(0, k)])
        return eq, DE

    found = False
    for k in range(1, order + 1):
        eq, DE = _makeDE(k)
        eq = eq.expand()
        terms = eq.as_ordered_terms()
        ind = rational_independent(terms, x)
        if found or len(ind) == k:
            sol = dict(zip(a, (i for s in linsolve(ind, a[:k]) for i in s)))
            if sol:
                found = True
                DE = DE.subs(sol)
            DE = DE.as_numer_denom()[0]
            DE = DE.factor().as_coeff_mul(Derivative)[1][0]
            yield DE.collect(Derivative(g(x))), k


def exp_re(DE, r, k):
    """Converts a DE with constant coefficients (explike) into a RE.

    Explanation
    ===========

    Performs the substitution:

    .. math::
        f^j(x) \\to r(k + j)

    Normalises the terms so that lowest order of a term is always r(k).

    Examples
    ========

    >>> from sympy import Function, Derivative
    >>> from sympy.series.formal import exp_re
    >>> from sympy.abc import x, k
    >>> f, r = Function('f'), Function('r')

    >>> exp_re(-f(x) + Derivative(f(x)), r, k)
    -r(k) + r(k + 1)
    >>> exp_re(Derivative(f(x), x) + Derivative(f(x), (x, 2)), r, k)
    r(k) + r(k + 1)

    See Also
    ========

    sympy.series.formal.hyper_re
    """
    RE = S.Zero

    g = DE.atoms(Function).pop()

    mini = None
    for t in Add.make_args(DE):
        coeff, d = t.as_independent(g)
        if isinstance(d, Derivative):
            j = d.derivative_count
        else:
            j = 0
        if mini is None or j < mini:
            mini = j
        RE += coeff * r(k + j)
    if mini:
        RE = RE.subs(k, k - mini)
    return RE


def hyper_re(DE, r, k):
    """
    Converts a DE into a RE.

    Explanation
    ===========

    Performs the substitution:

    .. math::
        x^l f^j(x) \\to (k + 1 - l)_j . a_{k + j - l}

    Normalises the terms so that lowest order of a term is always r(k).

    Examples
    ========

    >>> from sympy import Function, Derivative
    >>> from sympy.series.formal import hyper_re
    >>> from sympy.abc import x, k
    >>> f, r = Function('f'), Function('r')

    >>> hyper_re(-f(x) + Derivative(f(x)), r, k)
    (k + 1)*r(k + 1) - r(k)
    >>> hyper_re(-x*f(x) + Derivative(f(x), (x, 2)), r, k)
    (k + 2)*(k + 3)*r(k + 3) - r(k)

    See Also
    ========

    sympy.series.formal.exp_re
    """
    RE = S.Zero

    g = DE.atoms(Function).pop()
    x = g.atoms(Symbol).pop()

    mini = None
    for t in Add.make_args(DE.expand()):
        coeff, d = t.as_independent(g)
        c, v = coeff.as_independent(x)
        l = v.as_coeff_exponent(x)[1]
        if isinstance(d, Derivative):
            j = d.derivative_count
        else:
            j = 0
        RE += c * rf(k + 1 - l, j) * r(k + j - l)
        if mini is None or j - l < mini:
            mini = j - l

    RE = RE.subs(k, k - mini)

    m = Wild('m')
    return RE.collect(r(k + m))


def _transformation_a(f, x, P, Q, k, m, shift):
    f *= x**(-shift)
    P = P.subs(k, k + shift)
    Q = Q.subs(k, k + shift)
    return f, P, Q, m


def _transformation_c(f, x, P, Q, k, m, scale):
    f = f.subs(x, x**scale)
    P = P.subs(k, k / scale)
    Q = Q.subs(k, k / scale)
    m *= scale
    return f, P, Q, m


def _transformation_e(f, x, P, Q, k, m):
    f = f.diff(x)
    P = P.subs(k, k + 1) * (k + m + 1)
    Q = Q.subs(k, k + 1) * (k + 1)
    return f, P, Q, m


def _apply_shift(sol, shift):
    return [(res, cond + shift) for res, cond in sol]


def _apply_scale(sol, scale):
    return [(res, cond / scale) for res, cond in sol]


def _apply_integrate(sol, x, k):
    return [(res / ((cond + 1)*(cond.as_coeff_Add()[1].coeff(k))), cond + 1)
            for res, cond in sol]


def _compute_formula(f, x, P, Q, k, m, k_max):
    """Computes the formula for f."""
    from sympy.polys import roots

    sol = []
    for i in range(k_max + 1, k_max + m + 1):
        if (i < 0) == True:
            continue
        r = f.diff(x, i).limit(x, 0) / factorial(i)
        if r.is_zero:
            continue

        kterm = m*k + i
        res = r

        p = P.subs(k, kterm)
        q = Q.subs(k, kterm)
        c1 = p.subs(k, 1/k).leadterm(k)[0]
        c2 = q.subs(k, 1/k).leadterm(k)[0]
        res *= (-c1 / c2)**k

        res *= Mul(*[rf(-r, k)**mul for r, mul in roots(p, k).items()])
        res /= Mul(*[rf(-r, k)**mul for r, mul in roots(q, k).items()])

        sol.append((res, kterm))

    return sol


def _rsolve_hypergeometric(f, x, P, Q, k, m):
    """
    Recursive wrapper to rsolve_hypergeometric.

    Explanation
    ===========

    Returns a Tuple of (formula, series independent terms,
    maximum power of x in independent terms) if successful
    otherwise ``None``.

    See :func:`rsolve_hypergeometric` for details.
    """
    from sympy.polys import lcm, roots
    from sympy.integrals import integrate

    # transformation - c
    proots, qroots = roots(P, k), roots(Q, k)
    all_roots = dict(proots)
    all_roots.update(qroots)
    scale = lcm([r.as_numer_denom()[1] for r, t in all_roots.items()
                 if r.is_rational])
    f, P, Q, m = _transformation_c(f, x, P, Q, k, m, scale)

    # transformation - a
    qroots = roots(Q, k)
    if qroots:
        k_min = Min(*qroots.keys())
    else:
        k_min = S.Zero
    shift = k_min + m
    f, P, Q, m = _transformation_a(f, x, P, Q, k, m, shift)

    l = (x*f).limit(x, 0)
    if not isinstance(l, Limit) and l != 0:  # Ideally should only be l != 0
        return None

    qroots = roots(Q, k)
    if qroots:
        k_max = Max(*qroots.keys())
    else:
        k_max = S.Zero

    ind, mp = S.Zero, -oo
    for i in range(k_max + m + 1):
        r = f.diff(x, i).limit(x, 0) / factorial(i)
        if r.is_finite is False:
            old_f = f
            f, P, Q, m = _transformation_a(f, x, P, Q, k, m, i)
            f, P, Q, m = _transformation_e(f, x, P, Q, k, m)
            sol, ind, mp = _rsolve_hypergeometric(f, x, P, Q, k, m)
            sol = _apply_integrate(sol, x, k)
            sol = _apply_shift(sol, i)
            ind = integrate(ind, x)
            ind += (old_f - ind).limit(x, 0)  # constant of integration
            mp += 1
            return sol, ind, mp
        elif r:
            ind += r*x**(i + shift)
            pow_x = Rational((i + shift), scale)
            if pow_x > mp:
                mp = pow_x  # maximum power of x
    ind = ind.subs(x, x**(1/scale))

    sol = _compute_formula(f, x, P, Q, k, m, k_max)
    sol = _apply_shift(sol, shift)
    sol = _apply_scale(sol, scale)

    return sol, ind, mp


def rsolve_hypergeometric(f, x, P, Q, k, m):
    """
    Solves RE of hypergeometric type.

    Explanation
    ===========

    Attempts to solve RE of the form

    Q(k)*a(k + m) - P(k)*a(k)

    Transformations that preserve Hypergeometric type:

        a. x**n*f(x): b(k + m) = R(k - n)*b(k)
        b. f(A*x): b(k + m) = A**m*R(k)*b(k)
        c. f(x**n): b(k + n*m) = R(k/n)*b(k)
        d. f(x**(1/m)): b(k + 1) = R(k*m)*b(k)
        e. f'(x): b(k + m) = ((k + m + 1)/(k + 1))*R(k + 1)*b(k)

    Some of these transformations have been used to solve the RE.

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int

    Examples
    ========

    >>> from sympy import exp, ln, S
    >>> from sympy.series.formal import rsolve_hypergeometric as rh
    >>> from sympy.abc import x, k

    >>> rh(exp(x), x, -S.One, (k + 1), k, 1)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> rh(ln(1 + x), x, k**2, k*(k + 1), k, 1)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)

    References
    ==========

    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf
    .. [2] Power Series in Computer Algebra - Wolfram Koepf
    """
    result = _rsolve_hypergeometric(f, x, P, Q, k, m)

    if result is None:
        return None

    sol_list, ind, mp = result

    sol_dict = defaultdict(lambda: S.Zero)
    for res, cond in sol_list:
        j, mk = cond.as_coeff_Add()
        c = mk.coeff(k)

        if j.is_integer is False:
            res *= x**frac(j)
            j = floor(j)

        res = res.subs(k, (k - j) / c)
        cond = Eq(k % c, j % c)
        sol_dict[cond] += res  # Group together formula for same conditions

    sol = [(res, cond) for cond, res in sol_dict.items()]
    sol.append((S.Zero, True))
    sol = Piecewise(*sol)

    if mp is -oo:
        s = S.Zero
    elif mp.is_integer is False:
        s = ceiling(mp)
    else:
        s = mp + 1

    #  save all the terms of
    #  form 1/x**k in ind
    if s < 0:
        ind += sum(sequence(sol * x**k, (k, s, -1)))
        s = S.Zero

    return (sol, ind, s)


def _solve_hyper_RE(f, x, RE, g, k):
    """See docstring of :func:`rsolve_hypergeometric` for details."""
    terms = Add.make_args(RE)

    if len(terms) == 2:
        gs = list(RE.atoms(Function))
        P, Q = map(RE.coeff, gs)
        m = gs[1].args[0] - gs[0].args[0]
        if m < 0:
            P, Q = Q, P
            m = abs(m)
        return rsolve_hypergeometric(f, x, P, Q, k, m)


def _solve_explike_DE(f, x, DE, g, k):
    """Solves DE with constant coefficients."""
    from sympy.solvers import rsolve

    for t in Add.make_args(DE):
        coeff, d = t.as_independent(g)
        if coeff.free_symbols:
            return

    RE = exp_re(DE, g, k)

    init = {}
    for i in range(len(Add.make_args(RE))):
        if i:
            f = f.diff(x)
        init[g(k).subs(k, i)] = f.limit(x, 0)

    sol = rsolve(RE, g(k), init)

    if sol:
        return (sol / factorial(k), S.Zero, S.Zero)


def _solve_simple(f, x, DE, g, k):
    """Converts DE into RE and solves using :func:`rsolve`."""
    from sympy.solvers import rsolve

    RE = hyper_re(DE, g, k)

    init = {}
    for i in range(len(Add.make_args(RE))):
        if i:
            f = f.diff(x)
        init[g(k).subs(k, i)] = f.limit(x, 0) / factorial(i)

    sol = rsolve(RE, g(k), init)

    if sol:
        return (sol, S.Zero, S.Zero)


def _transform_explike_DE(DE, g, x, order, syms):
    """Converts DE with free parameters into DE with constant coefficients."""
    from sympy.solvers.solveset import linsolve

    eq = []
    highest_coeff = DE.coeff(Derivative(g(x), x, order))
    for i in range(order):
        coeff = DE.coeff(Derivative(g(x), x, i))
        coeff = (coeff / highest_coeff).expand().collect(x)
        eq.extend(Add.make_args(coeff))
    temp = []
    for e in eq:
        if e.has(x):
            break
        elif e.has(Symbol):
            temp.append(e)
    else:
        eq = temp
    if eq:
        sol = dict(zip(syms, (i for s in linsolve(eq, list(syms)) for i in s)))
        if sol:
            DE = DE.subs(sol)
            DE = DE.factor().as_coeff_mul(Derivative)[1][0]
            DE = DE.collect(Derivative(g(x)))
    return DE


def _transform_DE_RE(DE, g, k, order, syms):
    """Converts DE with free parameters into RE of hypergeometric type."""
    from sympy.solvers.solveset import linsolve

    RE = hyper_re(DE, g, k)

    eq = [RE.coeff(g(k + i)) for i in range(1, order)]
    sol = dict(zip(syms, (i for s in linsolve(eq, list(syms)) for i in s)))
    if sol:
        m = Wild('m')
        RE = RE.subs(sol)
        RE = RE.factor().as_numer_denom()[0].collect(g(k + m))
        RE = RE.as_coeff_mul(g)[1][0]
        for i in range(order):  # smallest order should be g(k)
            if RE.coeff(g(k + i)) and i:
                RE = RE.subs(k, k - i)
                break
    return RE


def solve_de(f, x, DE, order, g, k):
    """
    Solves the DE.

    Explanation
    ===========

    Tries to solve DE by either converting into a RE containing two terms or
    converting into a DE having constant coefficients.

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int

    Examples
    ========

    >>> from sympy import Derivative as D, Function
    >>> from sympy import exp, ln
    >>> from sympy.series.formal import solve_de
    >>> from sympy.abc import x, k
    >>> f = Function('f')

    >>> solve_de(exp(x), x, D(f(x), x) - f(x), 1, f, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> solve_de(ln(1 + x), x, (x + 1)*D(f(x), x, 2) + D(f(x)), 2, f, k)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)
    """
    sol = None
    syms = DE.free_symbols.difference({g, x})

    if syms:
        RE = _transform_DE_RE(DE, g, k, order, syms)
    else:
        RE = hyper_re(DE, g, k)
    if not RE.free_symbols.difference({k}):
        sol = _solve_hyper_RE(f, x, RE, g, k)

    if sol:
        return sol

    if syms:
        DE = _transform_explike_DE(DE, g, x, order, syms)
    if not DE.free_symbols.difference({x}):
        sol = _solve_explike_DE(f, x, DE, g, k)

    if sol:
        return sol


def hyper_algorithm(f, x, k, order=4):
    """
    Hypergeometric algorithm for computing Formal Power Series.

    Explanation
    ===========

    Steps:
        * Generates DE
        * Convert the DE into RE
        * Solves the RE

    Examples
    ========

    >>> from sympy import exp, ln
    >>> from sympy.series.formal import hyper_algorithm

    >>> from sympy.abc import x, k

    >>> hyper_algorithm(exp(x), x, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> hyper_algorithm(ln(1 + x), x, k)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)

    See Also
    ========

    sympy.series.formal.simpleDE
    sympy.series.formal.solve_de
    """
    g = Function('g')

    des = []  # list of DE's
    sol = None
    for DE, i in simpleDE(f, x, g, order):
        if DE is not None:
            sol = solve_de(f, x, DE, i, g, k)
        if sol:
            return sol
        if not DE.free_symbols.difference({x}):
            des.append(DE)

    # If nothing works
    # Try plain rsolve
    for DE in des:
        sol = _solve_simple(f, x, DE, g, k)
        if sol:
            return sol


def _compute_fps(f, x, x0, dir, hyper, order, rational, full):
    """Recursive wrapper to compute fps.

    See :func:`compute_fps` for details.
    """
    if x0 in [S.Infinity, S.NegativeInfinity]:
        dir = S.One if x0 is S.Infinity else -S.One
        temp = f.subs(x, 1/x)
        result = _compute_fps(temp, x, 0, dir, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, 1/x), result[2].subs(x, 1/x))
    elif x0 or dir == -S.One:
        if dir == -S.One:
            rep = -x + x0
            rep2 = -x
            rep2b = x0
        else:
            rep = x + x0
            rep2 = x
            rep2b = -x0
        temp = f.subs(x, rep)
        result = _compute_fps(temp, x, 0, S.One, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, rep2 + rep2b),
                result[2].subs(x, rep2 + rep2b))

    if f.is_polynomial(x):
        k = Dummy('k')
        ak = sequence(Coeff(f, x, k), (k, 1, oo))
        xk = sequence(x**k, (k, 0, oo))
        ind = f.coeff(x, 0)
        return ak, xk, ind

    #  Break instances of Add
    #  this allows application of different
    #  algorithms on different terms increasing the
    #  range of admissible functions.
    if isinstance(f, Add):
        result = False
        ak = sequence(S.Zero, (0, oo))
        ind, xk = S.Zero, None
        for t in Add.make_args(f):
            res = _compute_fps(t, x, 0, S.One, hyper, order, rational, full)
            if res:
                if not result:
                    result = True
                    xk = res[1]
                if res[0].start > ak.start:
                    seq = ak
                    s, f = ak.start, res[0].start
                else:
                    seq = res[0]
                    s, f = res[0].start, ak.start
                save = Add(*[z[0]*z[1] for z in zip(seq[0:(f - s)], xk[s:f])])
                ak += res[0]
                ind += res[2] + save
            else:
                ind += t
        if result:
            return ak, xk, ind
        return None

    # The symbolic term - symb, if present, is being separated from the function
    # Otherwise symb is being set to S.One
    syms = f.free_symbols.difference({x})
    (f, symb) = expand(f).as_independent(*syms)

    result = None

    # from here on it's x0=0 and dir=1 handling
    k = Dummy('k')
    if rational:
        result = rational_algorithm(f, x, k, order, full)

    if result is None and hyper:
        result = hyper_algorithm(f, x, k, order)

    if result is None:
        return None

    from sympy.simplify.powsimp import powsimp
    if symb.is_zero:
        symb = S.One
    else:
        symb = powsimp(symb)
    ak = sequence(result[0], (k, result[2], oo))
    xk_formula = powsimp(x**k * symb)
    xk = sequence(xk_formula, (k, 0, oo))
    ind = powsimp(result[1] * symb)

    return ak, xk, ind


def compute_fps(f, x, x0=0, dir=1, hyper=True, order=4, rational=True,
                full=False):
    """
    Computes the formula for Formal Power Series of a function.

    Explanation
    ===========

    Tries to compute the formula by applying the following techniques
    (in order):

    * rational_algorithm
    * Hypergeometric algorithm

    Parameters
    ==========

    x : Symbol
    x0 : number, optional
        Point to perform series expansion about. Default is 0.
    dir : {1, -1, '+', '-'}, optional
        If dir is 1 or '+' the series is calculated from the right and
        for -1 or '-' the series is calculated from the left. For smooth
        functions this flag will not alter the results. Default is 1.
    hyper : {True, False}, optional
        Set hyper to False to skip the hypergeometric algorithm.
        By default it is set to False.
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    rational : {True, False}, optional
        Set rational to False to skip rational algorithm. By default it is set
        to True.
    full : {True, False}, optional
        Set full to True to increase the range of rational algorithm.
        See :func:`rational_algorithm` for details. By default it is set to
        False.

    Returns
    =======

    ak : sequence
        Sequence of coefficients.
    xk : sequence
        Sequence of powers of x.
    ind : Expr
        Independent terms.
    mul : Pow
        Common terms.

    See Also
    ========

    sympy.series.formal.rational_algorithm
    sympy.series.formal.hyper_algorithm
    """
    f = sympify(f)
    x = sympify(x)

    if not f.has(x):
        return None

    x0 = sympify(x0)

    if dir == '+':
        dir = S.One
    elif dir == '-':
        dir = -S.One
    elif dir not in [S.One, -S.One]:
        raise ValueError("Dir must be '+' or '-'")
    else:
        dir = sympify(dir)

    return _compute_fps(f, x, x0, dir, hyper, order, rational, full)


class Coeff(Function):
    """
    Coeff(p, x, n) represents the nth coefficient of the polynomial p in x
    """
    @classmethod
    def eval(cls, p, x, n):
        if p.is_polynomial(x) and n.is_integer:
            return p.coeff(x, n)


class FormalPowerSeries(SeriesBase):
    """
    Represents Formal Power Series of a function.

    Explanation
    ===========

    No computation is performed. This class should only to be used to represent
    a series. No checks are performed.

    For computing a series use :func:`fps`.

    See Also
    ========

    sympy.series.formal.fps
    """
    def __new__(cls, *args):
        args = map(sympify, args)
        return Expr.__new__(cls, *args)

    def __init__(self, *args):
        ak = args[4][0]
        k = ak.variables[0]
        self.ak_seq = sequence(ak.formula, (k, 1, oo))
        self.fact_seq = sequence(factorial(k), (k, 1, oo))
        self.bell_coeff_seq = self.ak_seq * self.fact_seq
        self.sign_seq = sequence((-1, 1), (k, 1, oo))

    @property
    def function(self):
        return self.args[0]

    @property
    def x(self):
        return self.args[1]

    @property
    def x0(self):
        return self.args[2]

    @property
    def dir(self):
        return self.args[3]

    @property
    def ak(self):
        return self.args[4][0]

    @property
    def xk(self):
        return self.args[4][1]

    @property
    def ind(self):
        return self.args[4][2]

    @property
    def interval(self):
        return Interval(0, oo)

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return oo

    @property
    def infinite(self):
        """Returns an infinite representation of the series"""
        from sympy.concrete import Sum
        ak, xk = self.ak, self.xk
        k = ak.variables[0]
        inf_sum = Sum(ak.formula * xk.formula, (k, ak.start, ak.stop))

        return self.ind + inf_sum

    def _get_pow_x(self, term):
        """Returns the power of x in a term."""
        xterm, pow_x = term.as_independent(self.x)[1].as_base_exp()
        if not xterm.has(self.x):
            return S.Zero
        return pow_x

    def polynomial(self, n=6):
        """
        Truncated series as polynomial.

        Explanation
        ===========

        Returns series expansion of ``f`` upto order ``O(x**n)``
        as a polynomial(without ``O`` term).
        """
        terms = []
        sym = self.free_symbols
        for i, t in enumerate(self):
            xp = self._get_pow_x(t)
            if xp.has(*sym):
                xp = xp.as_coeff_add(*sym)[0]
            if xp >= n:
                break
            elif xp.is_integer is True and i == n + 1:
                break
            elif t is not S.Zero:
                terms.append(t)

        return Add(*terms)

    def truncate(self, n=6):
        """
        Truncated series.

        Explanation
        ===========

        Returns truncated series expansion of f upto
        order ``O(x**n)``.

        If n is ``None``, returns an infinite iterator.
        """
        if n is None:
            return iter(self)

        x, x0 = self.x, self.x0
        pt_xk = self.xk.coeff(n)
        if x0 is S.NegativeInfinity:
            x0 = S.Infinity

        return self.polynomial(n) + Order(pt_xk, (x, x0))

    def zero_coeff(self):
        return self._eval_term(0)

    def _eval_term(self, pt):
        try:
            pt_xk = self.xk.coeff(pt)
            pt_ak = self.ak.coeff(pt).simplify()  # Simplify the coefficients
        except IndexError:
            term = S.Zero
        else:
            term = (pt_ak * pt_xk)

        if self.ind:
            ind = S.Zero
            sym = self.free_symbols
            for t in Add.make_args(self.ind):
                pow_x = self._get_pow_x(t)
                if pow_x.has(*sym):
                    pow_x = pow_x.as_coeff_add(*sym)[0]
                if pt == 0 and pow_x < 1:
                    ind += t
                elif pow_x >= pt and pow_x < pt + 1:
                    ind += t
            term += ind

        return term.collect(self.x)

    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self

    def _eval_as_leading_term(self, x, logx, cdir):
        for t in self:
            if t is not S.Zero:
                return t

    def _eval_derivative(self, x):
        f = self.function.diff(x)
        ind = self.ind.diff(x)

        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t * (pow_xk + pow_x)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k + 1), (k, ak.start - 1, ak.stop))
        else:
            ak = sequence((ak.formula * pow_xk).subs(k, k + 1),
                          (k, ak.start - 1, ak.stop))

        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def integrate(self, x=None, **kwargs):
        """
        Integrate Formal Power Series.

        Examples
        ========

        >>> from sympy import fps, sin, integrate
        >>> from sympy.abc import x
        >>> f = fps(sin(x))
        >>> f.integrate(x).truncate()
        -1 + x**2/2 - x**4/24 + O(x**6)
        >>> integrate(f, (x, 0, 1))
        1 - cos(1)
        """
        from sympy.integrals import integrate

        if x is None:
            x = self.x
        elif iterable(x):
            return integrate(self.function, x)

        f = integrate(self.function, x)
        ind = integrate(self.ind, x)
        ind += (f - ind).limit(x, 0)  # constant of integration

        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t / (pow_xk + pow_x + 1)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k - 1), (k, ak.start + 1, ak.stop))
        else:
            ak = sequence((ak.formula / (pow_xk + 1)).subs(k, k - 1),
                          (k, ak.start + 1, ak.stop))

        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def product(self, other, x=None, n=6):
        """
        Multiplies two Formal Power Series, using discrete convolution and
        return the truncated terms upto specified order.

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))

        >>> f1.product(f2, x).truncate(4)
        x + x**2 + x**3/3 + O(x**4)

        See Also
        ========

        sympy.discrete.convolutions
        sympy.series.formal.FormalPowerSeriesProduct

        """

        if n is None:
            return iter(self)

        other = sympify(other)

        if not isinstance(other, FormalPowerSeries):
            raise ValueError("Both series should be an instance of FormalPowerSeries"
                             " class.")

        if self.dir != other.dir:
            raise ValueError("Both series should be calculated from the"
                             " same direction.")
        elif self.x0 != other.x0:
            raise ValueError("Both series should be calculated about the"
                             " same point.")

        elif self.x != other.x:
            raise ValueError("Both series should have the same symbol.")

        return FormalPowerSeriesProduct(self, other)

    def coeff_bell(self, n):
        r"""
        self.coeff_bell(n) returns a sequence of Bell polynomials of the second kind.
        Note that ``n`` should be a integer.

        The second kind of Bell polynomials (are sometimes called "partial" Bell
        polynomials or incomplete Bell polynomials) are defined as

        .. math::
            B_{n,k}(x_1, x_2,\dotsc x_{n-k+1}) =
                \sum_{j_1+j_2+j_2+\dotsb=k \atop j_1+2j_2+3j_2+\dotsb=n}
                \frac{n!}{j_1!j_2!\dotsb j_{n-k+1}!}
                \left(\frac{x_1}{1!} \right)^{j_1}
                \left(\frac{x_2}{2!} \right)^{j_2} \dotsb
                \left(\frac{x_{n-k+1}}{(n-k+1)!} \right) ^{j_{n-k+1}}.

        * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
          `B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})`.

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell

        """

        inner_coeffs = [bell(n, j, tuple(self.bell_coeff_seq[:n-j+1])) for j in range(1, n+1)]

        k = Dummy('k')
        return sequence(tuple(inner_coeffs), (k, 1, oo))

    def compose(self, other, x=None, n=6):
        r"""
        Returns the truncated terms of the formal power series of the composed function,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series `fp`
        will be as follows.

        .. math::
            \sum\limits_{k=0}^{n} b_k B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(sin(x))

        >>> f1.compose(f2, x).truncate()
        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)

        >>> f1.compose(f2, x).truncate(8)
        1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesCompose

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """

        if n is None:
            return iter(self)

        other = sympify(other)

        if not isinstance(other, FormalPowerSeries):
            raise ValueError("Both series should be an instance of FormalPowerSeries"
                             " class.")

        if self.dir != other.dir:
            raise ValueError("Both series should be calculated from the"
                             " same direction.")
        elif self.x0 != other.x0:
            raise ValueError("Both series should be calculated about the"
                             " same point.")

        elif self.x != other.x:
            raise ValueError("Both series should have the same symbol.")

        if other._eval_term(0).as_coeff_mul(other.x)[0] is not S.Zero:
            raise ValueError("The formal power series of the inner function should not have any "
                "constant coefficient term.")

        return FormalPowerSeriesCompose(self, other)

    def inverse(self, x=None, n=6):
        r"""
        Returns the truncated terms of the inverse of the formal power series,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series ``fp``
        will be as follows.

        .. math::
            \sum\limits_{k=0}^{n} (-1)^{k} x_0^{-k-1} B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, exp, cos
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(cos(x))

        >>> f1.inverse(x).truncate()
        1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6)

        >>> f2.inverse(x).truncate(8)
        1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesInverse

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """

        if n is None:
            return iter(self)

        if self._eval_term(0).is_zero:
            raise ValueError("Constant coefficient should exist for an inverse of a formal"
                " power series to exist.")

        return FormalPowerSeriesInverse(self)

    def __add__(self, other):
        other = sympify(other)

        if isinstance(other, FormalPowerSeries):
            if self.dir != other.dir:
                raise ValueError("Both series should be calculated from the"
                                 " same direction.")
            elif self.x0 != other.x0:
                raise ValueError("Both series should be calculated about the"
                                 " same point.")

            x, y = self.x, other.x
            f = self.function + other.function.subs(y, x)

            if self.x not in f.free_symbols:
                return f

            ak = self.ak + other.ak
            if self.ak.start > other.ak.start:
                seq = other.ak
                s, e = other.ak.start, self.ak.start
            else:
                seq = self.ak
                s, e = self.ak.start, other.ak.start
            save = Add(*[z[0]*z[1] for z in zip(seq[0:(e - s)], self.xk[s:e])])
            ind = self.ind + other.ind + save

            return self.func(f, x, self.x0, self.dir, (ak, self.xk, ind))

        elif not other.has(self.x):
            f = self.function + other
            ind = self.ind + other

            return self.func(f, self.x, self.x0, self.dir,
                             (self.ak, self.xk, ind))

        return Add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.func(-self.function, self.x, self.x0, self.dir,
                         (-self.ak, self.xk, -self.ind))

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        other = sympify(other)

        if other.has(self.x):
            return Mul(self, other)

        f = self.function * other
        ak = self.ak.coeff_mul(other)
        ind = self.ind * other

        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def __rmul__(self, other):
        return self.__mul__(other)


class FiniteFormalPowerSeries(FormalPowerSeries):
    """Base Class for Product, Compose and Inverse classes"""

    def __init__(self, *args):
        pass

    @property
    def ffps(self):
        return self.args[0]

    @property
    def gfps(self):
        return self.args[1]

    @property
    def f(self):
        return self.ffps.function

    @property
    def g(self):
        return self.gfps.function

    @property
    def infinite(self):
        raise NotImplementedError("No infinite version for an object of"
                     " FiniteFormalPowerSeries class.")

    def _eval_terms(self, n):
        raise NotImplementedError("(%s)._eval_terms()" % self)

    def _eval_term(self, pt):
        raise NotImplementedError("By the current logic, one can get terms"
                                   "upto a certain order, instead of getting term by term.")

    def polynomial(self, n):
        return self._eval_terms(n)

    def truncate(self, n=6):
        ffps = self.ffps
        pt_xk = ffps.xk.coeff(n)
        x, x0 = ffps.x, ffps.x0

        return self.polynomial(n) + Order(pt_xk, (x, x0))

    def _eval_derivative(self, x):
        raise NotImplementedError

    def integrate(self, x):
        raise NotImplementedError


class FormalPowerSeriesProduct(FiniteFormalPowerSeries):
    """Represents the product of two formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesProduct` object. The first argument contains the two
    functions involved in the product. Also, the coefficient sequence contains
    both the coefficient sequence of the formal power series of the involved functions.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    def __init__(self, *args):
        ffps, gfps = self.ffps, self.gfps

        k = ffps.ak.variables[0]
        self.coeff1 = sequence(ffps.ak.formula, (k, 0, oo))

        k = gfps.ak.variables[0]
        self.coeff2 = sequence(gfps.ak.formula, (k, 0, oo))

    @property
    def function(self):
        """Function of the product of two formal power series."""
        return self.f * self.g

    def _eval_terms(self, n):
        """
        Returns the first ``n`` terms of the product formal power series.
        Term by term logic is implemented here.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))
        >>> fprod = f1.product(f2, x)

        >>> fprod._eval_terms(4)
        x**3/3 + x**2 + x

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.product

        """
        coeff1, coeff2 = self.coeff1, self.coeff2

        aks = convolution(coeff1[:n], coeff2[:n])

        terms = []
        for i in range(0, n):
            terms.append(aks[i] * self.ffps.xk.coeff(i))

        return Add(*terms)


class FormalPowerSeriesCompose(FiniteFormalPowerSeries):
    """
    Represents the composed formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesCompose` object. The first argument contains the outer
    function and the inner function involved in the omposition. Also, the
    coefficient sequence contains the generic sequence which is to be multiplied
    by a custom ``bell_seq`` finite sequence. The finite terms will then be added up to
    get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    @property
    def function(self):
        """Function for the composed formal power series."""
        f, g, x = self.f, self.g, self.ffps.x
        return f.subs(x, g)

    def _eval_terms(self, n):
        """
        Returns the first `n` terms of the composed formal power series.
        Term by term logic is implemented here.

        Explanation
        ===========

        The coefficient sequence of the :obj:`FormalPowerSeriesCompose` object is the generic sequence.
        It is multiplied by ``bell_seq`` to get a sequence, whose terms are added up to get
        the final terms for the polynomial.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(sin(x))
        >>> fcomp = f1.compose(f2, x)

        >>> fcomp._eval_terms(6)
        -x**5/15 - x**4/8 + x**2/2 + x + 1

        >>> fcomp._eval_terms(8)
        x**7/90 - x**6/240 - x**5/15 - x**4/8 + x**2/2 + x + 1

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.compose
        sympy.series.formal.FormalPowerSeries.coeff_bell

        """

        ffps, gfps = self.ffps, self.gfps
        terms = [ffps.zero_coeff()]

        for i in range(1, n):
            bell_seq = gfps.coeff_bell(i)
            seq = (ffps.bell_coeff_seq * bell_seq)
            terms.append(Add(*(seq[:i])) / ffps.fact_seq[i-1] * ffps.xk.coeff(i))

        return Add(*terms)


class FormalPowerSeriesInverse(FiniteFormalPowerSeries):
    """
    Represents the Inverse of a formal power series.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There is a single difference between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesInverse` object. The coefficient sequence contains the
    generic sequence which is to be multiplied by a custom ``bell_seq`` finite sequence.
    The finite terms will then be added up to get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """
    def __init__(self, *args):
        ffps = self.ffps
        k = ffps.xk.variables[0]

        inv = ffps.zero_coeff()
        inv_seq = sequence(inv ** (-(k + 1)), (k, 1, oo))
        self.aux_seq = ffps.sign_seq * ffps.fact_seq * inv_seq

    @property
    def function(self):
        """Function for the inverse of a formal power series."""
        f = self.f
        return 1 / f

    @property
    def g(self):
        raise ValueError("Only one function is considered while performing"
                        "inverse of a formal power series.")

    @property
    def gfps(self):
        raise ValueError("Only one function is considered while performing"
                        "inverse of a formal power series.")

    def _eval_terms(self, n):
        """
        Returns the first ``n`` terms of the composed formal power series.
        Term by term logic is implemented here.

        Explanation
        ===========

        The coefficient sequence of the `FormalPowerSeriesInverse` object is the generic sequence.
        It is multiplied by ``bell_seq`` to get a sequence, whose terms are added up to get
        the final terms for the polynomial.

        Examples
        ========

        >>> from sympy import fps, exp, cos
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(cos(x))
        >>> finv1, finv2 = f1.inverse(), f2.inverse()

        >>> finv1._eval_terms(6)
        -x**5/120 + x**4/24 - x**3/6 + x**2/2 - x + 1

        >>> finv2._eval_terms(8)
        61*x**6/720 + 5*x**4/24 + x**2/2 + 1

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.inverse
        sympy.series.formal.FormalPowerSeries.coeff_bell

        """
        ffps = self.ffps
        terms = [ffps.zero_coeff()]

        for i in range(1, n):
            bell_seq = ffps.coeff_bell(i)
            seq = (self.aux_seq * bell_seq)
            terms.append(Add(*(seq[:i])) / ffps.fact_seq[i-1] * ffps.xk.coeff(i))

        return Add(*terms)


def fps(f, x=None, x0=0, dir=1, hyper=True, order=4, rational=True, full=False):
    """
    Generates Formal Power Series of ``f``.

    Explanation
    ===========

    Returns the formal series expansion of ``f`` around ``x = x0``
    with respect to ``x`` in the form of a ``FormalPowerSeries`` object.

    Formal Power Series is represented using an explicit formula
    computed using different algorithms.

    See :func:`compute_fps` for the more details regarding the computation
    of formula.

    Parameters
    ==========

    x : Symbol, optional
        If x is None and ``f`` is univariate, the univariate symbols will be
        supplied, otherwise an error will be raised.
    x0 : number, optional
        Point to perform series expansion about. Default is 0.
    dir : {1, -1, '+', '-'}, optional
        If dir is 1 or '+' the series is calculated from the right and
        for -1 or '-' the series is calculated from the left. For smooth
        functions this flag will not alter the results. Default is 1.
    hyper : {True, False}, optional
        Set hyper to False to skip the hypergeometric algorithm.
        By default it is set to False.
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    rational : {True, False}, optional
        Set rational to False to skip rational algorithm. By default it is set
        to True.
    full : {True, False}, optional
        Set full to True to increase the range of rational algorithm.
        See :func:`rational_algorithm` for details. By default it is set to
        False.

    Examples
    ========

    >>> from sympy import fps, ln, atan, sin
    >>> from sympy.abc import x, n

    Rational Functions

    >>> fps(ln(1 + x)).truncate()
    x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)

    >>> fps(atan(x), full=True).truncate()
    x - x**3/3 + x**5/5 + O(x**6)

    Symbolic Functions

    >>> fps(x**n*sin(x**2), x).truncate(8)
    -x**(n + 6)/6 + x**(n + 2) + O(x**(n + 8))

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.compute_fps
    """
    f = sympify(f)

    if x is None:
        free = f.free_symbols
        if len(free) == 1:
            x = free.pop()
        elif not free:
            return f
        else:
            raise NotImplementedError("multivariate formal power series")

    result = compute_fps(f, x, x0, dir, hyper, order, rational, full)

    if result is None:
        return f

    return FormalPowerSeries(f, x, x0, dir, result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\risch.py ===
"""
The Risch Algorithm for transcendental function integration.

The core algorithms for the Risch algorithm are here.  The subproblem
algorithms are in the rde.py and prde.py files for the Risch
Differential Equation solver and the parametric problems solvers,
respectively.  All important information concerning the differential extension
for an integrand is stored in a DifferentialExtension object, which in the code
is usually called DE.  Throughout the code and Inside the DifferentialExtension
object, the conventions/attribute names are that the base domain is QQ and each
differential extension is x, t0, t1, ..., tn-1 = DE.t. DE.x is the variable of
integration (Dx == 1), DE.D is a list of the derivatives of
x, t1, t2, ..., tn-1 = t, DE.T is the list [x, t1, t2, ..., tn-1], DE.t is the
outer-most variable of the differential extension at the given level (the level
can be adjusted using DE.increment_level() and DE.decrement_level()),
k is the field C(x, t0, ..., tn-2), where C is the constant field.  The
numerator of a fraction is denoted by a and the denominator by
d.  If the fraction is named f, fa == numer(f) and fd == denom(f).
Fractions are returned as tuples (fa, fd).  DE.d and DE.t are used to
represent the topmost derivation and extension variable, respectively.
The docstring of a function signifies whether an argument is in k[t], in
which case it will just return a Poly in t, or in k(t), in which case it
will return the fraction (fa, fd). Other variable names probably come
from the names used in Bronstein's book.
"""
from types import GeneratorType
from functools import reduce

from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.intfunc import ilcm
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh,
    tanh)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, cos,
    tan, acot, cot, asin, acos)
from .integrals import integrate, Integral
from .heurisch import _symbols
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import (real_roots, cancel, Poly, gcd,
    reduced)
from sympy.polys.rootoftools import RootSum
from sympy.utilities.iterables import numbered_symbols


def integer_powers(exprs):
    """
    Rewrites a list of expressions as integer multiples of each other.

    Explanation
    ===========

    For example, if you have [x, x/2, x**2 + 1, 2*x/3], then you can rewrite
    this as [(x/6) * 6, (x/6) * 3, (x**2 + 1) * 1, (x/6) * 4]. This is useful
    in the Risch integration algorithm, where we must write exp(x) + exp(x/2)
    as (exp(x/2))**2 + exp(x/2), but not as exp(x) + sqrt(exp(x)) (this is
    because only the transcendental case is implemented and we therefore cannot
    integrate algebraic extensions). The integer multiples returned by this
    function for each term are the smallest possible (their content equals 1).

    Returns a list of tuples where the first element is the base term and the
    second element is a list of `(item, factor)` terms, where `factor` is the
    integer multiplicative factor that must multiply the base term to obtain
    the original item.

    The easiest way to understand this is to look at an example:

    >>> from sympy.abc import x
    >>> from sympy.integrals.risch import integer_powers
    >>> integer_powers([x, x/2, x**2 + 1, 2*x/3])
    [(x/6, [(x, 6), (x/2, 3), (2*x/3, 4)]), (x**2 + 1, [(x**2 + 1, 1)])]

    We can see how this relates to the example at the beginning of the
    docstring.  It chose x/6 as the first base term.  Then, x can be written as
    (x/2) * 2, so we get (0, 2), and so on. Now only element (x**2 + 1)
    remains, and there are no other terms that can be written as a rational
    multiple of that, so we get that it can be written as (x**2 + 1) * 1.

    """
    # Here is the strategy:

    # First, go through each term and determine if it can be rewritten as a
    # rational multiple of any of the terms gathered so far.
    # cancel(a/b).is_Rational is sufficient for this.  If it is a multiple, we
    # add its multiple to the dictionary.

    terms = {}
    for term in exprs:
        for trm, trm_list in terms.items():
            a = cancel(term/trm)
            if a.is_Rational:
                trm_list.append((term, a))
                break
        else:
            terms[term] = [(term, S.One)]

    # After we have done this, we have all the like terms together, so we just
    # need to find a common denominator so that we can get the base term and
    # integer multiples such that each term can be written as an integer
    # multiple of the base term, and the content of the integers is 1.

    newterms = {}
    for term, term_list in terms.items():
        common_denom = reduce(ilcm, [i.as_numer_denom()[1] for _, i in
            term_list])
        newterm = term/common_denom
        newmults = [(i, j*common_denom) for i, j in term_list]
        newterms[newterm] = newmults

    return sorted(iter(newterms.items()), key=lambda item: item[0].sort_key())


class DifferentialExtension:
    """
    A container for all the information relating to a differential extension.

    Explanation
    ===========

    The attributes of this object are (see also the docstring of __init__):

    - f: The original (Expr) integrand.
    - x: The variable of integration.
    - T: List of variables in the extension.
    - D: List of derivations in the extension; corresponds to the elements of T.
    - fa: Poly of the numerator of the integrand.
    - fd: Poly of the denominator of the integrand.
    - Tfuncs: Lambda() representations of each element of T (except for x).
      For back-substitution after integration.
    - backsubs: A (possibly empty) list of further substitutions to be made on
      the final integral to make it look more like the integrand.
    - exts:
    - extargs:
    - cases: List of string representations of the cases of T.
    - t: The top level extension variable, as defined by the current level
      (see level below).
    - d: The top level extension derivation, as defined by the current
      derivation (see level below).
    - case: The string representation of the case of self.d.
    (Note that self.T and self.D will always contain the complete extension,
    regardless of the level.  Therefore, you should ALWAYS use DE.t and DE.d
    instead of DE.T[-1] and DE.D[-1].  If you want to have a list of the
    derivations or variables only up to the current level, use
    DE.D[:len(DE.D) + DE.level + 1] and DE.T[:len(DE.T) + DE.level + 1].  Note
    that, in particular, the derivation() function does this.)

    The following are also attributes, but will probably not be useful other
    than in internal use:
    - newf: Expr form of fa/fd.
    - level: The number (between -1 and -len(self.T)) such that
      self.T[self.level] == self.t and self.D[self.level] == self.d.
      Use the methods self.increment_level() and self.decrement_level() to change
      the current level.
    """
    # __slots__ is defined mainly so we can iterate over all the attributes
    # of the class easily (the memory use doesn't matter too much, since we
    # only create one DifferentialExtension per integration).  Also, it's nice
    # to have a safeguard when debugging.
    __slots__ = ('f', 'x', 'T', 'D', 'fa', 'fd', 'Tfuncs', 'backsubs',
        'exts', 'extargs', 'cases', 'case', 't', 'd', 'newf', 'level',
        'ts', 'dummy')

    def __init__(self, f=None, x=None, handle_first='log', dummy=False, extension=None, rewrite_complex=None):
        """
        Tries to build a transcendental extension tower from ``f`` with respect to ``x``.

        Explanation
        ===========

        If it is successful, creates a DifferentialExtension object with, among
        others, the attributes fa, fd, D, T, Tfuncs, and backsubs such that
        fa and fd are Polys in T[-1] with rational coefficients in T[:-1],
        fa/fd == f, and D[i] is a Poly in T[i] with rational coefficients in
        T[:i] representing the derivative of T[i] for each i from 1 to len(T).
        Tfuncs is a list of Lambda objects for back replacing the functions
        after integrating.  Lambda() is only used (instead of lambda) to make
        them easier to test and debug. Note that Tfuncs corresponds to the
        elements of T, except for T[0] == x, but they should be back-substituted
        in reverse order.  backsubs is a (possibly empty) back-substitution list
        that should be applied on the completed integral to make it look more
        like the original integrand.

        If it is unsuccessful, it raises NotImplementedError.

        You can also create an object by manually setting the attributes as a
        dictionary to the extension keyword argument.  You must include at least
        D.  Warning, any attribute that is not given will be set to None. The
        attributes T, t, d, cases, case, x, and level are set automatically and
        do not need to be given.  The functions in the Risch Algorithm will NOT
        check to see if an attribute is None before using it.  This also does not
        check to see if the extension is valid (non-algebraic) or even if it is
        self-consistent.  Therefore, this should only be used for
        testing/debugging purposes.
        """
        # XXX: If you need to debug this function, set the break point here

        if extension:
            if 'D' not in extension:
                raise ValueError("At least the key D must be included with "
                    "the extension flag to DifferentialExtension.")
            for attr in extension:
                setattr(self, attr, extension[attr])

            self._auto_attrs()

            return
        elif f is None or x is None:
            raise ValueError("Either both f and x or a manual extension must "
            "be given.")

        if handle_first not in ('log', 'exp'):
            raise ValueError("handle_first must be 'log' or 'exp', not %s." %
                str(handle_first))

        # f will be the original function, self.f might change if we reset
        # (e.g., we pull out a constant from an exponential)
        self.f = f
        self.x = x
        # setting the default value 'dummy'
        self.dummy = dummy
        self.reset()
        exp_new_extension, log_new_extension = True, True

        # case of 'automatic' choosing
        if rewrite_complex is None:
            rewrite_complex = I in self.f.atoms()

        if rewrite_complex:
            rewritables = {
                (sin, cos, cot, tan, sinh, cosh, coth, tanh): exp,
                (asin, acos, acot, atan): log,
            }
            # rewrite the trigonometric components
            for candidates, rule in rewritables.items():
                self.newf = self.newf.rewrite(candidates, rule)
            self.newf = cancel(self.newf)
        else:
            if any(i.has(x) for i in self.f.atoms(sin, cos, tan, atan, asin, acos)):
                raise NotImplementedError("Trigonometric extensions are not "
                "supported (yet!)")

        exps = set()
        pows = set()
        numpows = set()
        sympows = set()
        logs = set()
        symlogs = set()

        while True:
            if self.newf.is_rational_function(*self.T):
                break

            if not exp_new_extension and not log_new_extension:
                # We couldn't find a new extension on the last pass, so I guess
                # we can't do it.
                raise NotImplementedError("Couldn't find an elementary "
                    "transcendental extension for %s.  Try using a " % str(f) +
                    "manual extension with the extension flag.")

            exps, pows, numpows, sympows, log_new_extension = \
                    self._rewrite_exps_pows(exps, pows, numpows, sympows, log_new_extension)

            logs, symlogs = self._rewrite_logs(logs, symlogs)

            if handle_first == 'exp' or not log_new_extension:
                exp_new_extension = self._exp_part(exps)
                if exp_new_extension is None:
                    # reset and restart
                    self.f = self.newf
                    self.reset()
                    exp_new_extension = True
                    continue

            if handle_first == 'log' or not exp_new_extension:
                log_new_extension = self._log_part(logs)

        self.fa, self.fd = frac_in(self.newf, self.t)
        self._auto_attrs()

        return

    def __getattr__(self, attr):
        # Avoid AttributeErrors when debugging
        if attr not in self.__slots__:
            raise AttributeError("%s has no attribute %s" % (repr(self), repr(attr)))
        return None

    def _rewrite_exps_pows(self, exps, pows, numpows,
            sympows, log_new_extension):
        """
        Rewrite exps/pows for better processing.
        """
        from .prde import is_deriv_k

        # Pre-preparsing.
        #################
        # Get all exp arguments, so we can avoid ahead of time doing
        # something like t1 = exp(x), t2 = exp(x/2) == sqrt(t1).

        # Things like sqrt(exp(x)) do not automatically simplify to
        # exp(x/2), so they will be viewed as algebraic.  The easiest way
        # to handle this is to convert all instances of exp(a)**Rational
        # to exp(Rational*a) before doing anything else.  Note that the
        # _exp_part code can generate terms of this form, so we do need to
        # do this at each pass (or else modify it to not do that).

        ratpows = [i for i in self.newf.atoms(Pow)
                   if (isinstance(i.base, exp) and i.exp.is_Rational)]

        ratpows_repl = [
            (i, i.base.base**(i.exp*i.base.exp)) for i in ratpows]
        self.backsubs += [(j, i) for i, j in ratpows_repl]
        self.newf = self.newf.xreplace(dict(ratpows_repl))

        # To make the process deterministic, the args are sorted
        # so that functions with smaller op-counts are processed first.
        # Ties are broken with the default_sort_key.

        # XXX Although the method is deterministic no additional work
        # has been done to guarantee that the simplest solution is
        # returned and that it would be affected be using different
        # variables. Though it is possible that this is the case
        # one should know that it has not been done intentionally, so
        # further improvements may be possible.

        # TODO: This probably doesn't need to be completely recomputed at
        # each pass.
        exps = update_sets(exps, self.newf.atoms(exp),
            lambda i: i.exp.is_rational_function(*self.T) and
            i.exp.has(*self.T))
        pows = update_sets(pows, self.newf.atoms(Pow),
            lambda i: i.exp.is_rational_function(*self.T) and
            i.exp.has(*self.T))
        numpows = update_sets(numpows, set(pows),
            lambda i: not i.base.has(*self.T))
        sympows = update_sets(sympows, set(pows) - set(numpows),
            lambda i: i.base.is_rational_function(*self.T) and
            not i.exp.is_Integer)

        # The easiest way to deal with non-base E powers is to convert them
        # into base E, integrate, and then convert back.
        for i in ordered(pows):
            old = i
            new = exp(i.exp*log(i.base))
            # If exp is ever changed to automatically reduce exp(x*log(2))
            # to 2**x, then this will break.  The solution is to not change
            # exp to do that :)
            if i in sympows:
                if i.exp.is_Rational:
                    raise NotImplementedError("Algebraic extensions are "
                        "not supported (%s)." % str(i))
                # We can add a**b only if log(a) in the extension, because
                # a**b == exp(b*log(a)).
                basea, based = frac_in(i.base, self.t)
                A = is_deriv_k(basea, based, self)
                if A is None:
                    # Nonelementary monomial (so far)

                    # TODO: Would there ever be any benefit from just
                    # adding log(base) as a new monomial?
                    # ANSWER: Yes, otherwise we can't integrate x**x (or
                    # rather prove that it has no elementary integral)
                    # without first manually rewriting it as exp(x*log(x))
                    self.newf = self.newf.xreplace({old: new})
                    self.backsubs += [(new, old)]
                    log_new_extension = self._log_part([log(i.base)])
                    exps = update_sets(exps, self.newf.atoms(exp), lambda i:
                        i.exp.is_rational_function(*self.T) and i.exp.has(*self.T))
                    continue
                ans, u, const = A
                newterm = exp(i.exp*(log(const) + u))
                # Under the current implementation, exp kills terms
                # only if they are of the form a*log(x), where a is a
                # Number.  This case should have already been killed by the
                # above tests.  Again, if this changes to kill more than
                # that, this will break, which maybe is a sign that you
                # shouldn't be changing that.  Actually, if anything, this
                # auto-simplification should be removed.  See
                # https://groups.google.com/group/sympy/browse_thread/thread/a61d48235f16867f

                self.newf = self.newf.xreplace({i: newterm})

            elif i not in numpows:
                continue
            else:
                # i in numpows
                newterm = new
            # TODO: Just put it in self.Tfuncs
            self.backsubs.append((new, old))
            self.newf = self.newf.xreplace({old: newterm})
            exps.append(newterm)

        return exps, pows, numpows, sympows, log_new_extension

    def _rewrite_logs(self, logs, symlogs):
        """
        Rewrite logs for better processing.
        """
        atoms = self.newf.atoms(log)
        logs = update_sets(logs, atoms,
            lambda i: i.args[0].is_rational_function(*self.T) and
            i.args[0].has(*self.T))
        symlogs = update_sets(symlogs, atoms,
            lambda i: i.has(*self.T) and i.args[0].is_Pow and
            i.args[0].base.is_rational_function(*self.T) and
            not i.args[0].exp.is_Integer)

        # We can handle things like log(x**y) by converting it to y*log(x)
        # This will fix not only symbolic exponents of the argument, but any
        # non-Integer exponent, like log(sqrt(x)).  The exponent can also
        # depend on x, like log(x**x).
        for i in ordered(symlogs):
            # Unlike in the exponential case above, we do not ever
            # potentially add new monomials (above we had to add log(a)).
            # Therefore, there is no need to run any is_deriv functions
            # here.  Just convert log(a**b) to b*log(a) and let
            # log_new_extension() handle it from there.
            lbase = log(i.args[0].base)
            logs.append(lbase)
            new = i.args[0].exp*lbase
            self.newf = self.newf.xreplace({i: new})
            self.backsubs.append((new, i))

        # remove any duplicates
        logs = sorted(set(logs), key=default_sort_key)

        return logs, symlogs

    def _auto_attrs(self):
        """
        Set attributes that are generated automatically.
        """
        if not self.T:
            # i.e., when using the extension flag and T isn't given
            self.T = [i.gen for i in self.D]
        if not self.x:
            self.x = self.T[0]
        self.cases = [get_case(d, t) for d, t in zip(self.D, self.T)]
        self.level = -1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]

    def _exp_part(self, exps):
        """
        Try to build an exponential extension.

        Returns
        =======

        Returns True if there was a new extension, False if there was no new
        extension but it was able to rewrite the given exponentials in terms
        of the existing extension, and None if the entire extension building
        process should be restarted.  If the process fails because there is no
        way around an algebraic extension (e.g., exp(log(x)/2)), it will raise
        NotImplementedError.
        """
        from .prde import is_log_deriv_k_t_radical
        new_extension = False
        restart = False
        expargs = [i.exp for i in exps]
        ip = integer_powers(expargs)
        for arg, others in ip:
            # Minimize potential problems with algebraic substitution
            others.sort(key=lambda i: i[1])

            arga, argd = frac_in(arg, self.t)
            A = is_log_deriv_k_t_radical(arga, argd, self)

            if A is not None:
                ans, u, n, const = A
                # if n is 1 or -1, it's algebraic, but we can handle it
                if n == -1:
                    # This probably will never happen, because
                    # Rational.as_numer_denom() returns the negative term in
                    # the numerator.  But in case that changes, reduce it to
                    # n == 1.
                    n = 1
                    u **= -1
                    const *= -1
                    ans = [(i, -j) for i, j in ans]

                if n == 1:
                    # Example: exp(x + x**2) over QQ(x, exp(x), exp(x**2))
                    self.newf = self.newf.xreplace({exp(arg): exp(const)*Mul(*[
                        u**power for u, power in ans])})
                    self.newf = self.newf.xreplace({exp(p*exparg):
                        exp(const*p) * Mul(*[u**power for u, power in ans])
                        for exparg, p in others})
                    # TODO: Add something to backsubs to put exp(const*p)
                    # back together.

                    continue

                else:
                    # Bad news: we have an algebraic radical.  But maybe we
                    # could still avoid it by choosing a different extension.
                    # For example, integer_powers() won't handle exp(x/2 + 1)
                    # over QQ(x, exp(x)), but if we pull out the exp(1), it
                    # will.  Or maybe we have exp(x + x**2/2), over
                    # QQ(x, exp(x), exp(x**2)), which is exp(x)*sqrt(exp(x**2)),
                    # but if we use QQ(x, exp(x), exp(x**2/2)), then they will
                    # all work.
                    #
                    # So here is what we do: If there is a non-zero const, pull
                    # it out and retry.  Also, if len(ans) > 1, then rewrite
                    # exp(arg) as the product of exponentials from ans, and
                    # retry that.  If const == 0 and len(ans) == 1, then we
                    # assume that it would have been handled by either
                    # integer_powers() or n == 1 above if it could be handled,
                    # so we give up at that point.  For example, you can never
                    # handle exp(log(x)/2) because it equals sqrt(x).

                    if const or len(ans) > 1:
                        rad = Mul(*[term**(power/n) for term, power in ans])
                        self.newf = self.newf.xreplace({exp(p*exparg):
                            exp(const*p)*rad for exparg, p in others})
                        self.newf = self.newf.xreplace(dict(list(zip(reversed(self.T),
                            reversed([f(self.x) for f in self.Tfuncs])))))
                        restart = True
                        break
                    else:
                        # TODO: give algebraic dependence in error string
                        raise NotImplementedError("Cannot integrate over "
                            "algebraic extensions.")

            else:
                arga, argd = frac_in(arg, self.t)
                darga = (argd*derivation(Poly(arga, self.t), self) -
                    arga*derivation(Poly(argd, self.t), self))
                dargd = argd**2
                darga, dargd = darga.cancel(dargd, include=True)
                darg = darga.as_expr()/dargd.as_expr()
                self.t = next(self.ts)
                self.T.append(self.t)
                self.extargs.append(arg)
                self.exts.append('exp')
                self.D.append(darg.as_poly(self.t, expand=False)*Poly(self.t,
                    self.t, expand=False))
                if self.dummy:
                    i = Dummy("i")
                else:
                    i = Symbol('i')
                self.Tfuncs += [Lambda(i, exp(arg.subs(self.x, i)))]
                self.newf = self.newf.xreplace(
                        {exp(exparg): self.t**p for exparg, p in others})
                new_extension = True

        if restart:
            return None
        return new_extension

    def _log_part(self, logs):
        """
        Try to build a logarithmic extension.

        Returns
        =======

        Returns True if there was a new extension and False if there was no new
        extension but it was able to rewrite the given logarithms in terms
        of the existing extension.  Unlike with exponential extensions, there
        is no way that a logarithm is not transcendental over and cannot be
        rewritten in terms of an already existing extension in a non-algebraic
        way, so this function does not ever return None or raise
        NotImplementedError.
        """
        from .prde import is_deriv_k
        new_extension = False
        logargs = [i.args[0] for i in logs]
        for arg in ordered(logargs):
            # The log case is easier, because whenever a logarithm is algebraic
            # over the base field, it is of the form a1*t1 + ... an*tn + c,
            # which is a polynomial, so we can just replace it with that.
            # In other words, we don't have to worry about radicals.
            arga, argd = frac_in(arg, self.t)
            A = is_deriv_k(arga, argd, self)
            if A is not None:
                ans, u, const = A
                newterm = log(const) + u
                self.newf = self.newf.xreplace({log(arg): newterm})
                continue

            else:
                arga, argd = frac_in(arg, self.t)
                darga = (argd*derivation(Poly(arga, self.t), self) -
                    arga*derivation(Poly(argd, self.t), self))
                dargd = argd**2
                darg = darga.as_expr()/dargd.as_expr()
                self.t = next(self.ts)
                self.T.append(self.t)
                self.extargs.append(arg)
                self.exts.append('log')
                self.D.append(cancel(darg.as_expr()/arg).as_poly(self.t,
                    expand=False))
                if self.dummy:
                    i = Dummy("i")
                else:
                    i = Symbol('i')
                self.Tfuncs += [Lambda(i, log(arg.subs(self.x, i)))]
                self.newf = self.newf.xreplace({log(arg): self.t})
                new_extension = True

        return new_extension

    @property
    def _important_attrs(self):
        """
        Returns some of the more important attributes of self.

        Explanation
        ===========

        Used for testing and debugging purposes.

        The attributes are (fa, fd, D, T, Tfuncs, backsubs,
        exts, extargs).
        """
        return (self.fa, self.fd, self.D, self.T, self.Tfuncs,
            self.backsubs, self.exts, self.extargs)

    # NOTE: this printing doesn't follow the Python's standard
    # eval(repr(DE)) == DE, where DE is the DifferentialExtension object,
    # also this printing is supposed to contain all the important
    # attributes of a DifferentialExtension object
    def __repr__(self):
        # no need to have GeneratorType object printed in it
        r = [(attr, getattr(self, attr)) for attr in self.__slots__
                if not isinstance(getattr(self, attr), GeneratorType)]
        return self.__class__.__name__ + '(dict(%r))' % (r)

    # fancy printing of DifferentialExtension object
    def __str__(self):
        return (self.__class__.__name__ + '({fa=%s, fd=%s, D=%s})' %
                (self.fa, self.fd, self.D))

    # should only be used for debugging purposes, internally
    # f1 = f2 = log(x) at different places in code execution
    # may return D1 != D2 as True, since 'level' or other attribute
    # may differ
    def __eq__(self, other):
        for attr in self.__class__.__slots__:
            d1, d2 = getattr(self, attr), getattr(other, attr)
            if not (isinstance(d1, GeneratorType) or d1 == d2):
                return False
        return True

    def reset(self):
        """
        Reset self to an initial state.  Used by __init__.
        """
        self.t = self.x
        self.T = [self.x]
        self.D = [Poly(1, self.x)]
        self.level = -1
        self.exts = [None]
        self.extargs = [None]
        if self.dummy:
            self.ts = numbered_symbols('t', cls=Dummy)
        else:
            # For testing
            self.ts = numbered_symbols('t')
        # For various things that we change to make things work that we need to
        # change back when we are done.
        self.backsubs = []
        self.Tfuncs = []
        self.newf = self.f

    def indices(self, extension):
        """
        Parameters
        ==========

        extension : str
            Represents a valid extension type.

        Returns
        =======

        list: A list of indices of 'exts' where extension of
            type 'extension' is present.

        Examples
        ========

        >>> from sympy.integrals.risch import DifferentialExtension
        >>> from sympy import log, exp
        >>> from sympy.abc import x
        >>> DE = DifferentialExtension(log(x) + exp(x), x, handle_first='exp')
        >>> DE.indices('log')
        [2]
        >>> DE.indices('exp')
        [1]

        """
        return [i for i, ext in enumerate(self.exts) if ext == extension]

    def increment_level(self):
        """
        Increment the level of self.

        Explanation
        ===========

        This makes the working differential extension larger.  self.level is
        given relative to the end of the list (-1, -2, etc.), so we do not need
        do worry about it when building the extension.
        """
        if self.level >= -1:
            raise ValueError("The level of the differential extension cannot "
                "be incremented any further.")

        self.level += 1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]
        return None

    def decrement_level(self):
        """
        Decrease the level of self.

        Explanation
        ===========

        This makes the working differential extension smaller.  self.level is
        given relative to the end of the list (-1, -2, etc.), so we do not need
        do worry about it when building the extension.
        """
        if self.level <= -len(self.T):
            raise ValueError("The level of the differential extension cannot "
                "be decremented any further.")

        self.level -= 1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]
        return None


def update_sets(seq, atoms, func):
    s = set(seq)
    s = atoms.intersection(s)
    new = atoms - s
    s.update(list(filter(func, new)))
    return list(s)


class DecrementLevel:
    """
    A context manager for decrementing the level of a DifferentialExtension.
    """
    __slots__ = ('DE',)

    def __init__(self, DE):
        self.DE = DE
        return

    def __enter__(self):
        self.DE.decrement_level()

    def __exit__(self, exc_type, exc_value, traceback):
        self.DE.increment_level()


class NonElementaryIntegralException(Exception):
    """
    Exception used by subroutines within the Risch algorithm to indicate to one
    another that the function being integrated does not have an elementary
    integral in the given differential field.
    """
    # TODO: Rewrite algorithms below to use this (?)

    # TODO: Pass through information about why the integral was nonelementary,
    # and store that in the resulting NonElementaryIntegral somehow.
    pass


def gcdex_diophantine(a, b, c):
    """
    Extended Euclidean Algorithm, Diophantine version.

    Explanation
    ===========

    Given ``a``, ``b`` in K[x] and ``c`` in (a, b), the ideal generated by ``a`` and
    ``b``, return (s, t) such that s*a + t*b == c and either s == 0 or s.degree()
    < b.degree().
    """
    # Extended Euclidean Algorithm (Diophantine Version) pg. 13
    # TODO: This should go in densetools.py.
    # XXX: Better name?

    s, g = a.half_gcdex(b)
    s *= c.exquo(g)  # Inexact division means c is not in (a, b)
    if s and s.degree() >= b.degree():
        _, s = s.div(b)
    t = (c - s*a).exquo(b)
    return (s, t)


def frac_in(f, t, *, cancel=False, **kwargs):
    """
    Returns the tuple (fa, fd), where fa and fd are Polys in t.

    Explanation
    ===========

    This is a common idiom in the Risch Algorithm functions, so we abstract
    it out here. ``f`` should be a basic expression, a Poly, or a tuple (fa, fd),
    where fa and fd are either basic expressions or Polys, and f == fa/fd.
    **kwargs are applied to Poly.
    """
    if isinstance(f, tuple):
        fa, fd = f
        f = fa.as_expr()/fd.as_expr()
    fa, fd = f.as_expr().as_numer_denom()
    fa, fd = fa.as_poly(t, **kwargs), fd.as_poly(t, **kwargs)
    if cancel:
        fa, fd = fa.cancel(fd, include=True)
    if fa is None or fd is None:
        raise ValueError("Could not turn %s into a fraction in %s." % (f, t))
    return (fa, fd)


def as_poly_1t(p, t, z):
    """
    (Hackish) way to convert an element ``p`` of K[t, 1/t] to K[t, z].

    In other words, ``z == 1/t`` will be a dummy variable that Poly can handle
    better.

    See issue 5131.

    Examples
    ========

    >>> from sympy import random_poly
    >>> from sympy.integrals.risch import as_poly_1t
    >>> from sympy.abc import x, z

    >>> p1 = random_poly(x, 10, -10, 10)
    >>> p2 = random_poly(x, 10, -10, 10)
    >>> p = p1 + p2.subs(x, 1/x)
    >>> as_poly_1t(p, x, z).as_expr().subs(z, 1/x) == p
    True
    """
    # TODO: Use this on the final result.  That way, we can avoid answers like
    # (...)*exp(-x).
    pa, pd = frac_in(p, t, cancel=True)
    if not pd.is_monomial:
        # XXX: Is there a better Poly exception that we could raise here?
        # Either way, if you see this (from the Risch Algorithm) it indicates
        # a bug.
        raise PolynomialError("%s is not an element of K[%s, 1/%s]." % (p, t, t))

    t_part, remainder = pa.div(pd)

    ans = t_part.as_poly(t, z, expand=False)

    if remainder:
        one = remainder.one
        tp = t*one
        r = pd.degree() - remainder.degree()
        z_part = remainder.transform(one, tp) * tp**r
        z_part = z_part.replace(t, z).to_field().quo_ground(pd.LC())
        ans += z_part.as_poly(t, z, expand=False)

    return ans


def derivation(p, DE, coefficientD=False, basic=False):
    """
    Computes Dp.

    Explanation
    ===========

    Given the derivation D with D = d/dx and p is a polynomial in t over
    K(x), return Dp.

    If coefficientD is True, it computes the derivation kD
    (kappaD), which is defined as kD(sum(ai*Xi**i, (i, 0, n))) ==
    sum(Dai*Xi**i, (i, 1, n)) (Definition 3.2.2, page 80).  X in this case is
    T[-1], so coefficientD computes the derivative just with respect to T[:-1],
    with T[-1] treated as a constant.

    If ``basic=True``, the returns a Basic expression.  Elements of D can still be
    instances of Poly.
    """
    if basic:
        r = 0
    else:
        r = Poly(0, DE.t)

    t = DE.t
    if coefficientD:
        if DE.level <= -len(DE.T):
            # 'base' case, the answer is 0.
            return r
        DE.decrement_level()

    D = DE.D[:len(DE.D) + DE.level + 1]
    T = DE.T[:len(DE.T) + DE.level + 1]

    for d, v in zip(D, T):
        pv = p.as_poly(v)
        if pv is None or basic:
            pv = p.as_expr()

        if basic:
            r += d.as_expr()*pv.diff(v)
        else:
            r += (d.as_expr()*pv.diff(v).as_expr()).as_poly(t)

    if basic:
        r = cancel(r)
    if coefficientD:
        DE.increment_level()

    return r


def get_case(d, t):
    """
    Returns the type of the derivation d.

    Returns one of {'exp', 'tan', 'base', 'primitive', 'other_linear',
    'other_nonlinear'}.
    """
    if not d.expr.has(t):
        if d.is_one:
            return 'base'
        return 'primitive'
    if d.rem(Poly(t, t)).is_zero:
        return 'exp'
    if d.rem(Poly(1 + t**2, t)).is_zero:
        return 'tan'
    if d.degree(t) > 1:
        return 'other_nonlinear'
    return 'other_linear'


def splitfactor(p, DE, coefficientD=False, z=None):
    """
    Splitting factorization.

    Explanation
    ===========

    Given a derivation D on k[t] and ``p`` in k[t], return (p_n, p_s) in
    k[t] x k[t] such that p = p_n*p_s, p_s is special, and each square
    factor of p_n is normal.

    Page. 100
    """
    kinv = [1/x for x in DE.T[:DE.level]]
    if z:
        kinv.append(z)

    One = Poly(1, DE.t, domain=p.get_domain())
    Dp = derivation(p, DE, coefficientD=coefficientD)
    # XXX: Is this right?
    if p.is_zero:
        return (p, One)

    if not p.expr.has(DE.t):
        s = p.as_poly(*kinv).gcd(Dp.as_poly(*kinv)).as_poly(DE.t)
        n = p.exquo(s)
        return (n, s)

    if not Dp.is_zero:
        h = p.gcd(Dp).to_field()
        g = p.gcd(p.diff(DE.t)).to_field()
        s = h.exquo(g)

        if s.degree(DE.t) == 0:
            return (p, One)

        q_split = splitfactor(p.exquo(s), DE, coefficientD=coefficientD)

        return (q_split[0], q_split[1]*s)
    else:
        return (p, One)


def splitfactor_sqf(p, DE, coefficientD=False, z=None, basic=False):
    """
    Splitting Square-free Factorization.

    Explanation
    ===========

    Given a derivation D on k[t] and ``p`` in k[t], returns (N1, ..., Nm)
    and (S1, ..., Sm) in k[t]^m such that p =
    (N1*N2**2*...*Nm**m)*(S1*S2**2*...*Sm**m) is a splitting
    factorization of ``p`` and the Ni and Si are square-free and coprime.
    """
    # TODO: This algorithm appears to be faster in every case
    # TODO: Verify this and splitfactor() for multiple extensions
    kkinv = [1/x for x in DE.T[:DE.level]] + DE.T[:DE.level]
    if z:
        kkinv = [z]

    S = []
    N = []
    p_sqf = p.sqf_list_include()
    if p.is_zero:
        return (((p, 1),), ())

    for pi, i in p_sqf:
        Si = pi.as_poly(*kkinv).gcd(derivation(pi, DE,
            coefficientD=coefficientD,basic=basic).as_poly(*kkinv)).as_poly(DE.t)
        pi = Poly(pi, DE.t)
        Si = Poly(Si, DE.t)
        Ni = pi.exquo(Si)
        if not Si.is_one:
            S.append((Si, i))
        if not Ni.is_one:
            N.append((Ni, i))

    return (tuple(N), tuple(S))


def canonical_representation(a, d, DE):
    """
    Canonical Representation.

    Explanation
    ===========

    Given a derivation D on k[t] and f = a/d in k(t), return (f_p, f_s,
    f_n) in k[t] x k(t) x k(t) such that f = f_p + f_s + f_n is the
    canonical representation of f (f_p is a polynomial, f_s is reduced
    (has a special denominator), and f_n is simple (has a normal
    denominator).
    """
    # Make d monic
    l = Poly(1/d.LC(), DE.t)
    a, d = a.mul(l), d.mul(l)

    q, r = a.div(d)
    dn, ds = splitfactor(d, DE)

    b, c = gcdex_diophantine(dn.as_poly(DE.t), ds.as_poly(DE.t), r.as_poly(DE.t))
    b, c = b.as_poly(DE.t), c.as_poly(DE.t)

    return (q, (b, ds), (c, dn))


def hermite_reduce(a, d, DE):
    """
    Hermite Reduction - Mack's Linear Version.

    Given a derivation D on k(t) and f = a/d in k(t), returns g, h, r in
    k(t) such that f = Dg + h + r, h is simple, and r is reduced.

    """
    # Make d monic
    l = Poly(1/d.LC(), DE.t)
    a, d = a.mul(l), d.mul(l)

    fp, fs, fn = canonical_representation(a, d, DE)
    a, d = fn
    l = Poly(1/d.LC(), DE.t)
    a, d = a.mul(l), d.mul(l)

    ga = Poly(0, DE.t)
    gd = Poly(1, DE.t)

    dd = derivation(d, DE)
    dm = gcd(d.to_field(), dd.to_field()).as_poly(DE.t)
    ds, _ = d.div(dm)

    while dm.degree(DE.t) > 0:

        ddm = derivation(dm, DE)
        dm2 = gcd(dm.to_field(), ddm.to_field())
        dms, _ = dm.div(dm2)
        ds_ddm = ds.mul(ddm)
        ds_ddm_dm, _ = ds_ddm.div(dm)

        b, c = gcdex_diophantine(-ds_ddm_dm.as_poly(DE.t),
            dms.as_poly(DE.t), a.as_poly(DE.t))
        b, c = b.as_poly(DE.t), c.as_poly(DE.t)

        db = derivation(b, DE).as_poly(DE.t)
        ds_dms, _ = ds.div(dms)
        a = c.as_poly(DE.t) - db.mul(ds_dms).as_poly(DE.t)

        ga = ga*dm + b*gd
        gd = gd*dm
        ga, gd = ga.cancel(gd, include=True)
        dm = dm2

    q, r = a.div(ds)
    ga, gd = ga.cancel(gd, include=True)

    r, d = r.cancel(ds, include=True)
    rra = q*fs[1] + fp*fs[1] + fs[0]
    rrd = fs[1]
    rra, rrd = rra.cancel(rrd, include=True)

    return ((ga, gd), (r, d), (rra, rrd))


def polynomial_reduce(p, DE):
    """
    Polynomial Reduction.

    Explanation
    ===========

    Given a derivation D on k(t) and p in k[t] where t is a nonlinear
    monomial over k, return q, r in k[t] such that p = Dq  + r, and
    deg(r) < deg_t(Dt).
    """
    q = Poly(0, DE.t)
    while p.degree(DE.t) >= DE.d.degree(DE.t):
        m = p.degree(DE.t) - DE.d.degree(DE.t) + 1
        q0 = Poly(DE.t**m, DE.t).mul(Poly(p.as_poly(DE.t).LC()/
            (m*DE.d.LC()), DE.t))
        q += q0
        p = p - derivation(q0, DE)

    return (q, p)


def laurent_series(a, d, F, n, DE):
    """
    Contribution of ``F`` to the full partial fraction decomposition of A/D.

    Explanation
    ===========

    Given a field K of characteristic 0 and ``A``,``D``,``F`` in K[x] with D monic,
    nonzero, coprime with A, and ``F`` the factor of multiplicity n in the square-
    free factorization of D, return the principal parts of the Laurent series of
    A/D at all the zeros of ``F``.
    """
    if F.degree()==0:
        return 0
    Z = _symbols('z', n)
    z = Symbol('z')
    Z.insert(0, z)
    delta_a = Poly(0, DE.t)
    delta_d = Poly(1, DE.t)

    E = d.quo(F**n)
    ha, hd = (a, E*Poly(z**n, DE.t))
    dF = derivation(F,DE)
    B, _ = gcdex_diophantine(E, F, Poly(1,DE.t))
    C, _ = gcdex_diophantine(dF, F, Poly(1,DE.t))

    # initialization
    F_store = F
    V, DE_D_list, H_list= [], [], []

    for j in range(0, n):
    # jth derivative of z would be substituted with dfnth/(j+1) where dfnth =(d^n)f/(dx)^n
        F_store = derivation(F_store, DE)
        v = (F_store.as_expr())/(j + 1)
        V.append(v)
        DE_D_list.append(Poly(Z[j + 1],Z[j]))

    DE_new = DifferentialExtension(extension = {'D': DE_D_list}) #a differential indeterminate
    for j in range(0, n):
        zEha = Poly(z**(n + j), DE.t)*E**(j + 1)*ha
        zEhd = hd
        Pa, Pd = cancel((zEha, zEhd))[1], cancel((zEha, zEhd))[2]
        Q = Pa.quo(Pd)
        for i in range(0, j + 1):
            Q = Q.subs(Z[i], V[i])
        Dha = (hd*derivation(ha, DE, basic=True).as_poly(DE.t)
             + ha*derivation(hd, DE, basic=True).as_poly(DE.t)
             + hd*derivation(ha, DE_new, basic=True).as_poly(DE.t)
             + ha*derivation(hd, DE_new, basic=True).as_poly(DE.t))
        Dhd = Poly(j + 1, DE.t)*hd**2
        ha, hd = Dha, Dhd

        Ff, _ = F.div(gcd(F, Q))
        F_stara, F_stard = frac_in(Ff, DE.t)
        if F_stara.degree(DE.t) - F_stard.degree(DE.t) > 0:
            QBC = Poly(Q, DE.t)*B**(1 + j)*C**(n + j)
            H = QBC
            H_list.append(H)
            H = (QBC*F_stard).rem(F_stara)
            alphas = real_roots(F_stara)
            for alpha in list(alphas):
                delta_a = delta_a*Poly((DE.t - alpha)**(n - j), DE.t) + Poly(H.eval(alpha), DE.t)
                delta_d = delta_d*Poly((DE.t - alpha)**(n - j), DE.t)
    return (delta_a, delta_d, H_list)


def recognize_derivative(a, d, DE, z=None):
    """
    Compute the squarefree factorization of the denominator of f
    and for each Di the polynomial H in K[x] (see Theorem 2.7.1), using the
    LaurentSeries algorithm. Write Di = GiEi where Gj = gcd(Hn, Di) and
    gcd(Ei,Hn) = 1. Since the residues of f at the roots of Gj are all 0, and
    the residue of f at a root alpha of Ei is Hi(a) != 0, f is the derivative of a
    rational function if and only if Ei = 1 for each i, which is equivalent to
    Di | H[-1] for each i.
    """
    flag =True
    a, d = a.cancel(d, include=True)
    _, r = a.div(d)
    Np, Sp = splitfactor_sqf(d, DE, coefficientD=True, z=z)

    j = 1
    for s, _ in Sp:
        delta_a, delta_d, H = laurent_series(r, d, s, j, DE)
        g = gcd(d, H[-1]).as_poly()
        if g is not d:
            flag = False
            break
        j = j + 1
    return flag


def recognize_log_derivative(a, d, DE, z=None):
    """
    There exists a v in K(x)* such that f = dv/v
    where f a rational function if and only if f can be written as f = A/D
    where D is squarefree,deg(A) < deg(D), gcd(A, D) = 1,
    and all the roots of the Rothstein-Trager resultant are integers. In that case,
    any of the Rothstein-Trager, Lazard-Rioboo-Trager or Czichowski algorithm
    produces u in K(x) such that du/dx = uf.
    """

    z = z or Dummy('z')
    a, d = a.cancel(d, include=True)
    _, a = a.div(d)

    pz = Poly(z, DE.t)
    Dd = derivation(d, DE)
    q = a - pz*Dd
    r, _ = d.resultant(q, includePRS=True)
    r = Poly(r, z)
    Np, Sp = splitfactor_sqf(r, DE, coefficientD=True, z=z)

    for s, _ in Sp:
        # TODO also consider the complex roots which should
        # turn the flag false
        a = real_roots(s.as_poly(z))

        if not all(j.is_Integer for j in a):
            return False
    return True

def residue_reduce(a, d, DE, z=None, invert=True):
    """
    Lazard-Rioboo-Rothstein-Trager resultant reduction.

    Explanation
    ===========

    Given a derivation ``D`` on k(t) and f in k(t) simple, return g
    elementary over k(t) and a Boolean b in {True, False} such that f -
    Dg in k[t] if b == True or f + h and f + h - Dg do not have an
    elementary integral over k(t) for any h in k<t> (reduced) if b ==
    False.

    Returns (G, b), where G is a tuple of tuples of the form (s_i, S_i),
    such that g = Add(*[RootSum(s_i, lambda z: z*log(S_i(z, t))) for
    S_i, s_i in G]). f - Dg is the remaining integral, which is elementary
    only if b == True, and hence the integral of f is elementary only if
    b == True.

    f - Dg is not calculated in this function because that would require
    explicitly calculating the RootSum.  Use residue_reduce_derivation().
    """
    # TODO: Use log_to_atan() from rationaltools.py
    # If r = residue_reduce(...), then the logarithmic part is given by:
    # sum([RootSum(a[0].as_poly(z), lambda i: i*log(a[1].as_expr()).subs(z,
    # i)).subs(t, log(x)) for a in r[0]])

    z = z or Dummy('z')
    a, d = a.cancel(d, include=True)
    a, d = a.to_field().mul_ground(1/d.LC()), d.to_field().mul_ground(1/d.LC())
    kkinv = [1/x for x in DE.T[:DE.level]] + DE.T[:DE.level]

    if a.is_zero:
        return ([], True)
    _, a = a.div(d)

    pz = Poly(z, DE.t)

    Dd = derivation(d, DE)
    q = a - pz*Dd

    if Dd.degree(DE.t) <= d.degree(DE.t):
        r, R = d.resultant(q, includePRS=True)
    else:
        r, R = q.resultant(d, includePRS=True)

    R_map, H = {}, []
    for i in R:
        R_map[i.degree()] = i

    r = Poly(r, z)
    Np, Sp = splitfactor_sqf(r, DE, coefficientD=True, z=z)

    for s, i in Sp:
        if i == d.degree(DE.t):
            s = Poly(s, z).monic()
            H.append((s, d))
        else:
            h = R_map.get(i)
            if h is None:
                continue
            h_lc = Poly(h.as_poly(DE.t).LC(), DE.t, field=True)

            h_lc_sqf = h_lc.sqf_list_include(all=True)

            for a, j in h_lc_sqf:
                h = Poly(h, DE.t, field=True).exquo(Poly(gcd(a, s**j, *kkinv),
                    DE.t))

            s = Poly(s, z).monic()

            if invert:
                h_lc = Poly(h.as_poly(DE.t).LC(), DE.t, field=True, expand=False)
                inv, coeffs = h_lc.as_poly(z, field=True).invert(s), [S.One]

                for coeff in h.coeffs()[1:]:
                    L = reduced(inv*coeff.as_poly(inv.gens), [s])[1]
                    coeffs.append(L.as_expr())

                h = Poly(dict(list(zip(h.monoms(), coeffs))), DE.t)

            H.append((s, h))

    b = not any(cancel(i.as_expr()).has(DE.t, z) for i, _ in Np)

    return (H, b)


def residue_reduce_to_basic(H, DE, z):
    """
    Converts the tuple returned by residue_reduce() into a Basic expression.
    """
    # TODO: check what Lambda does with RootOf
    i = Dummy('i')
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    return sum(RootSum(a[0].as_poly(z), Lambda(i, i*log(a[1].as_expr()).subs(
        {z: i}).subs(s))) for a in H)


def residue_reduce_derivation(H, DE, z):
    """
    Computes the derivation of an expression returned by residue_reduce().

    In general, this is a rational function in t, so this returns an
    as_expr() result.
    """
    # TODO: verify that this is correct for multiple extensions
    i = Dummy('i')
    return S(sum(RootSum(a[0].as_poly(z), Lambda(i, i*derivation(a[1],
        DE).as_expr().subs(z, i)/a[1].as_expr().subs(z, i))) for a in H))


def integrate_primitive_polynomial(p, DE):
    """
    Integration of primitive polynomials.

    Explanation
    ===========

    Given a primitive monomial t over k, and ``p`` in k[t], return q in k[t],
    r in k, and a bool b in {True, False} such that r = p - Dq is in k if b is
    True, or r = p - Dq does not have an elementary integral over k(t) if b is
    False.
    """
    Zero = Poly(0, DE.t)
    q = Poly(0, DE.t)

    if not p.expr.has(DE.t):
        return (Zero, p, True)

    from .prde import limited_integrate
    while True:
        if not p.expr.has(DE.t):
            return (q, p, True)

        Dta, Dtb = frac_in(DE.d, DE.T[DE.level - 1])

        with DecrementLevel(DE):  # We had better be integrating the lowest extension (x)
                                  # with ratint().
            a = p.LC()
            aa, ad = frac_in(a, DE.t)

            try:
                rv = limited_integrate(aa, ad, [(Dta, Dtb)], DE)
                if rv is None:
                    raise NonElementaryIntegralException
                (ba, bd), c = rv
            except NonElementaryIntegralException:
                return (q, p, False)

        m = p.degree(DE.t)
        q0 = c[0].as_poly(DE.t)*Poly(DE.t**(m + 1)/(m + 1), DE.t) + \
            (ba.as_expr()/bd.as_expr()).as_poly(DE.t)*Poly(DE.t**m, DE.t)

        p = p - derivation(q0, DE)
        q = q + q0


def integrate_primitive(a, d, DE, z=None):
    """
    Integration of primitive functions.

    Explanation
    ===========

    Given a primitive monomial t over k and f in k(t), return g elementary over
    k(t), i in k(t), and b in {True, False} such that i = f - Dg is in k if b
    is True or i = f - Dg does not have an elementary integral over k(t) if b
    is False.

    This function returns a Basic expression for the first argument.  If b is
    True, the second argument is Basic expression in k to recursively integrate.
    If b is False, the second argument is an unevaluated Integral, which has
    been proven to be nonelementary.
    """
    # XXX: a and d must be canceled, or this might return incorrect results
    z = z or Dummy("z")
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        i = cancel(a.as_expr()/d.as_expr() - (g1[1]*derivation(g1[0], DE) -
            g1[0]*derivation(g1[1], DE)).as_expr()/(g1[1]**2).as_expr() -
            residue_reduce_derivation(g2, DE, z))
        i = NonElementaryIntegral(cancel(i).subs(s), DE.x)
        return ((g1[0].as_expr()/g1[1].as_expr()).subs(s) +
            residue_reduce_to_basic(g2, DE, z), i, b)

    # h - Dg2 + r
    p = cancel(h[0].as_expr()/h[1].as_expr() - residue_reduce_derivation(g2,
        DE, z) + r[0].as_expr()/r[1].as_expr())
    p = p.as_poly(DE.t)

    q, i, b = integrate_primitive_polynomial(p, DE)

    ret = ((g1[0].as_expr()/g1[1].as_expr() + q.as_expr()).subs(s) +
        residue_reduce_to_basic(g2, DE, z))
    if not b:
        # TODO: This does not do the right thing when b is False
        i = NonElementaryIntegral(cancel(i.as_expr()).subs(s), DE.x)
    else:
        i = cancel(i.as_expr())

    return (ret, i, b)


def integrate_hyperexponential_polynomial(p, DE, z):
    """
    Integration of hyperexponential polynomials.

    Explanation
    ===========

    Given a hyperexponential monomial t over k and ``p`` in k[t, 1/t], return q in
    k[t, 1/t] and a bool b in {True, False} such that p - Dq in k if b is True,
    or p - Dq does not have an elementary integral over k(t) if b is False.
    """
    t1 = DE.t
    dtt = DE.d.exquo(Poly(DE.t, DE.t))
    qa = Poly(0, DE.t)
    qd = Poly(1, DE.t)
    b = True

    if p.is_zero:
        return(qa, qd, b)

    from sympy.integrals.rde import rischDE

    with DecrementLevel(DE):
        for i in range(-p.degree(z), p.degree(t1) + 1):
            if not i:
                continue
            elif i < 0:
                # If you get AttributeError: 'NoneType' object has no attribute 'nth'
                # then this should really not have expand=False
                # But it shouldn't happen because p is already a Poly in t and z
                a = p.as_poly(z, expand=False).nth(-i)
            else:
                # If you get AttributeError: 'NoneType' object has no attribute 'nth'
                # then this should really not have expand=False
                a = p.as_poly(t1, expand=False).nth(i)

            aa, ad = frac_in(a, DE.t, field=True)
            aa, ad = aa.cancel(ad, include=True)
            iDt = Poly(i, t1)*dtt
            iDta, iDtd = frac_in(iDt, DE.t, field=True)
            try:
                va, vd = rischDE(iDta, iDtd, Poly(aa, DE.t), Poly(ad, DE.t), DE)
                va, vd = frac_in((va, vd), t1, cancel=True)
            except NonElementaryIntegralException:
                b = False
            else:
                qa = qa*vd + va*Poly(t1**i)*qd
                qd *= vd

    return (qa, qd, b)


def integrate_hyperexponential(a, d, DE, z=None, conds='piecewise'):
    """
    Integration of hyperexponential functions.

    Explanation
    ===========

    Given a hyperexponential monomial t over k and f in k(t), return g
    elementary over k(t), i in k(t), and a bool b in {True, False} such that
    i = f - Dg is in k if b is True or i = f - Dg does not have an elementary
    integral over k(t) if b is False.

    This function returns a Basic expression for the first argument.  If b is
    True, the second argument is Basic expression in k to recursively integrate.
    If b is False, the second argument is an unevaluated Integral, which has
    been proven to be nonelementary.
    """
    # XXX: a and d must be canceled, or this might return incorrect results
    z = z or Dummy("z")
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        i = cancel(a.as_expr()/d.as_expr() - (g1[1]*derivation(g1[0], DE) -
            g1[0]*derivation(g1[1], DE)).as_expr()/(g1[1]**2).as_expr() -
            residue_reduce_derivation(g2, DE, z))
        i = NonElementaryIntegral(cancel(i.subs(s)), DE.x)
        return ((g1[0].as_expr()/g1[1].as_expr()).subs(s) +
            residue_reduce_to_basic(g2, DE, z), i, b)

    # p should be a polynomial in t and 1/t, because Sirr == k[t, 1/t]
    # h - Dg2 + r
    p = cancel(h[0].as_expr()/h[1].as_expr() - residue_reduce_derivation(g2,
        DE, z) + r[0].as_expr()/r[1].as_expr())
    pp = as_poly_1t(p, DE.t, z)

    qa, qd, b = integrate_hyperexponential_polynomial(pp, DE, z)

    i = pp.nth(0, 0)

    ret = ((g1[0].as_expr()/g1[1].as_expr()).subs(s) \
        + residue_reduce_to_basic(g2, DE, z))

    qas = qa.as_expr().subs(s)
    qds = qd.as_expr().subs(s)
    if conds == 'piecewise' and DE.x not in qds.free_symbols:
        # We have to be careful if the exponent is S.Zero!

        # XXX: Does qd = 0 always necessarily correspond to the exponential
        # equaling 1?
        ret += Piecewise(
                (qas/qds, Ne(qds, 0)),
                (integrate((p - i).subs(DE.t, 1).subs(s), DE.x), True)
            )
    else:
        ret += qas/qds

    if not b:
        i = p - (qd*derivation(qa, DE) - qa*derivation(qd, DE)).as_expr()/\
            (qd**2).as_expr()
        i = NonElementaryIntegral(cancel(i).subs(s), DE.x)
    return (ret, i, b)


def integrate_hypertangent_polynomial(p, DE):
    """
    Integration of hypertangent polynomials.

    Explanation
    ===========

    Given a differential field k such that sqrt(-1) is not in k, a
    hypertangent monomial t over k, and p in k[t], return q in k[t] and
    c in k such that p - Dq - c*D(t**2 + 1)/(t**1 + 1) is in k and p -
    Dq does not have an elementary integral over k(t) if Dc != 0.
    """
    # XXX: Make sure that sqrt(-1) is not in k.
    q, r = polynomial_reduce(p, DE)
    a = DE.d.exquo(Poly(DE.t**2 + 1, DE.t))
    c = Poly(r.nth(1)/(2*a.as_expr()), DE.t)
    return (q, c)


def integrate_nonlinear_no_specials(a, d, DE, z=None):
    """
    Integration of nonlinear monomials with no specials.

    Explanation
    ===========

    Given a nonlinear monomial t over k such that Sirr ({p in k[t] | p is
    special, monic, and irreducible}) is empty, and f in k(t), returns g
    elementary over k(t) and a Boolean b in {True, False} such that f - Dg is
    in k if b == True, or f - Dg does not have an elementary integral over k(t)
    if b == False.

    This function is applicable to all nonlinear extensions, but in the case
    where it returns b == False, it will only have proven that the integral of
    f - Dg is nonelementary if Sirr is empty.

    This function returns a Basic expression.
    """
    # TODO: Integral from k?
    # TODO: split out nonelementary integral
    # XXX: a and d must be canceled, or this might not return correct results
    z = z or Dummy("z")
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        return ((g1[0].as_expr()/g1[1].as_expr()).subs(s) +
            residue_reduce_to_basic(g2, DE, z), b)

    # Because f has no specials, this should be a polynomial in t, or else
    # there is a bug.
    p = cancel(h[0].as_expr()/h[1].as_expr() - residue_reduce_derivation(g2,
        DE, z).as_expr() + r[0].as_expr()/r[1].as_expr()).as_poly(DE.t)
    q1, q2 = polynomial_reduce(p, DE)

    if q2.expr.has(DE.t):
        b = False
    else:
        b = True

    ret = (cancel(g1[0].as_expr()/g1[1].as_expr() + q1.as_expr()).subs(s) +
        residue_reduce_to_basic(g2, DE, z))
    return (ret, b)


class NonElementaryIntegral(Integral):
    """
    Represents a nonelementary Integral.

    Explanation
    ===========

    If the result of integrate() is an instance of this class, it is
    guaranteed to be nonelementary.  Note that integrate() by default will try
    to find any closed-form solution, even in terms of special functions which
    may themselves not be elementary.  To make integrate() only give
    elementary solutions, or, in the cases where it can prove the integral to
    be nonelementary, instances of this class, use integrate(risch=True).
    In this case, integrate() may raise NotImplementedError if it cannot make
    such a determination.

    integrate() uses the deterministic Risch algorithm to integrate elementary
    functions or prove that they have no elementary integral.  In some cases,
    this algorithm can split an integral into an elementary and nonelementary
    part, so that the result of integrate will be the sum of an elementary
    expression and a NonElementaryIntegral.

    Examples
    ========

    >>> from sympy import integrate, exp, log, Integral
    >>> from sympy.abc import x

    >>> a = integrate(exp(-x**2), x, risch=True)
    >>> print(a)
    Integral(exp(-x**2), x)
    >>> type(a)
    <class 'sympy.integrals.risch.NonElementaryIntegral'>

    >>> expr = (2*log(x)**2 - log(x) - x**2)/(log(x)**3 - x**2*log(x))
    >>> b = integrate(expr, x, risch=True)
    >>> print(b)
    -log(-x + log(x))/2 + log(x + log(x))/2 + Integral(1/log(x), x)
    >>> type(b.atoms(Integral).pop())
    <class 'sympy.integrals.risch.NonElementaryIntegral'>

    """
    # TODO: This is useful in and of itself, because isinstance(result,
    # NonElementaryIntegral) will tell if the integral has been proven to be
    # elementary. But should we do more?  Perhaps a no-op .doit() if
    # elementary=True?  Or maybe some information on why the integral is
    # nonelementary.
    pass


def risch_integrate(f, x, extension=None, handle_first='log',
                    separate_integral=False, rewrite_complex=None,
                    conds='piecewise'):
    r"""
    The Risch Integration Algorithm.

    Explanation
    ===========

    Only transcendental functions are supported.  Currently, only exponentials
    and logarithms are supported, but support for trigonometric functions is
    forthcoming.

    If this function returns an unevaluated Integral in the result, it means
    that it has proven that integral to be nonelementary.  Any errors will
    result in raising NotImplementedError.  The unevaluated Integral will be
    an instance of NonElementaryIntegral, a subclass of Integral.

    handle_first may be either 'exp' or 'log'.  This changes the order in
    which the extension is built, and may result in a different (but
    equivalent) solution (for an example of this, see issue 5109).  It is also
    possible that the integral may be computed with one but not the other,
    because not all cases have been implemented yet.  It defaults to 'log' so
    that the outer extension is exponential when possible, because more of the
    exponential case has been implemented.

    If ``separate_integral`` is ``True``, the result is returned as a tuple (ans, i),
    where the integral is ans + i, ans is elementary, and i is either a
    NonElementaryIntegral or 0.  This useful if you want to try further
    integrating the NonElementaryIntegral part using other algorithms to
    possibly get a solution in terms of special functions.  It is False by
    default.

    Examples
    ========

    >>> from sympy.integrals.risch import risch_integrate
    >>> from sympy import exp, log, pprint
    >>> from sympy.abc import x

    First, we try integrating exp(-x**2). Except for a constant factor of
    2/sqrt(pi), this is the famous error function.

    >>> pprint(risch_integrate(exp(-x**2), x))
      /
     |
     |    2
     |  -x
     | e    dx
     |
    /

    The unevaluated Integral in the result means that risch_integrate() has
    proven that exp(-x**2) does not have an elementary anti-derivative.

    In many cases, risch_integrate() can split out the elementary
    anti-derivative part from the nonelementary anti-derivative part.
    For example,

    >>> pprint(risch_integrate((2*log(x)**2 - log(x) - x**2)/(log(x)**3 -
    ... x**2*log(x)), x))
                                             /
                                            |
      log(-x + log(x))   log(x + log(x))    |   1
    - ---------------- + --------------- +  | ------ dx
             2                  2           | log(x)
                                            |
                                           /

    This means that it has proven that the integral of 1/log(x) is
    nonelementary.  This function is also known as the logarithmic integral,
    and is often denoted as Li(x).

    risch_integrate() currently only accepts purely transcendental functions
    with exponentials and logarithms, though note that this can include
    nested exponentials and logarithms, as well as exponentials with bases
    other than E.

    >>> pprint(risch_integrate(exp(x)*exp(exp(x)), x))
     / x\
     \e /
    e
    >>> pprint(risch_integrate(exp(exp(x)), x))
      /
     |
     |  / x\
     |  \e /
     | e     dx
     |
    /

    >>> pprint(risch_integrate(x*x**x*log(x) + x**x + x*x**x, x))
       x
    x*x
    >>> pprint(risch_integrate(x**x, x))
      /
     |
     |  x
     | x  dx
     |
    /

    >>> pprint(risch_integrate(-1/(x*log(x)*log(log(x))**2), x))
         1
    -----------
    log(log(x))

    """
    f = S(f)

    DE = extension or DifferentialExtension(f, x, handle_first=handle_first,
            dummy=True, rewrite_complex=rewrite_complex)
    fa, fd = DE.fa, DE.fd

    result = S.Zero
    for case in reversed(DE.cases):
        if not fa.expr.has(DE.t) and not fd.expr.has(DE.t) and not case == 'base':
            DE.decrement_level()
            fa, fd = frac_in((fa, fd), DE.t)
            continue

        fa, fd = fa.cancel(fd, include=True)
        if case == 'exp':
            ans, i, b = integrate_hyperexponential(fa, fd, DE, conds=conds)
        elif case == 'primitive':
            ans, i, b = integrate_primitive(fa, fd, DE)
        elif case == 'base':
            # XXX: We can't call ratint() directly here because it doesn't
            # handle polynomials correctly.
            ans = integrate(fa.as_expr()/fd.as_expr(), DE.x, risch=False)
            b = False
            i = S.Zero
        else:
            raise NotImplementedError("Only exponential and logarithmic "
            "extensions are currently supported.")

        result += ans
        if b:
            DE.decrement_level()
            fa, fd = frac_in(i, DE.t)
        else:
            result = result.subs(DE.backsubs)
            if not i.is_zero:
                i = NonElementaryIntegral(i.function.subs(DE.backsubs),i.limits)
            if not separate_integral:
                result += i
                return result
            else:

                if isinstance(i, NonElementaryIntegral):
                    return (result, i)
                else:
                    return (result, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\tz\tz.py ===
# -*- coding: utf-8 -*-
"""
This module offers timezone implementations subclassing the abstract
:py:class:`datetime.tzinfo` type. There are classes to handle tzfile format
files (usually are in :file:`/etc/localtime`, :file:`/usr/share/zoneinfo`,
etc), TZ environment string (in all known formats), given ranges (with help
from relative deltas), local machine timezone, fixed offset timezone, and UTC
timezone.
"""
import datetime
import struct
import time
import sys
import os
import bisect
import weakref
from collections import OrderedDict

import six
from six import string_types
from six.moves import _thread
from ._common import tzname_in_python2, _tzinfo
from ._common import tzrangebase, enfold
from ._common import _validate_fromutc_inputs

from ._factories import _TzSingleton, _TzOffsetFactory
from ._factories import _TzStrFactory
try:
    from .win import tzwin, tzwinlocal
except ImportError:
    tzwin = tzwinlocal = None

# For warning about rounding tzinfo
from warnings import warn

ZERO = datetime.timedelta(0)
EPOCH = datetime.datetime(1970, 1, 1, 0, 0)
EPOCHORDINAL = EPOCH.toordinal()


@six.add_metaclass(_TzSingleton)
class tzutc(datetime.tzinfo):
    """
    This is a tzinfo object that represents the UTC time zone.

    **Examples:**

    .. doctest::

        >>> from datetime import *
        >>> from dateutil.tz import *

        >>> datetime.now()
        datetime.datetime(2003, 9, 27, 9, 40, 1, 521290)

        >>> datetime.now(tzutc())
        datetime.datetime(2003, 9, 27, 12, 40, 12, 156379, tzinfo=tzutc())

        >>> datetime.now(tzutc()).tzname()
        'UTC'

    .. versionchanged:: 2.7.0
        ``tzutc()`` is now a singleton, so the result of ``tzutc()`` will
        always return the same object.

        .. doctest::

            >>> from dateutil.tz import tzutc, UTC
            >>> tzutc() is tzutc()
            True
            >>> tzutc() is UTC
            True
    """
    def utcoffset(self, dt):
        return ZERO

    def dst(self, dt):
        return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        return "UTC"

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        return False

    @_validate_fromutc_inputs
    def fromutc(self, dt):
        """
        Fast track version of fromutc() returns the original ``dt`` object for
        any valid :py:class:`datetime.datetime` object.
        """
        return dt

    def __eq__(self, other):
        if not isinstance(other, (tzutc, tzoffset)):
            return NotImplemented

        return (isinstance(other, tzutc) or
                (isinstance(other, tzoffset) and other._offset == ZERO))

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s()" % self.__class__.__name__

    __reduce__ = object.__reduce__


#: Convenience constant providing a :class:`tzutc()` instance
#:
#: .. versionadded:: 2.7.0
UTC = tzutc()


@six.add_metaclass(_TzOffsetFactory)
class tzoffset(datetime.tzinfo):
    """
    A simple class for representing a fixed offset from UTC.

    :param name:
        The timezone name, to be returned when ``tzname()`` is called.
    :param offset:
        The time zone offset in seconds, or (since version 2.6.0, represented
        as a :py:class:`datetime.timedelta` object).
    """
    def __init__(self, name, offset):
        self._name = name

        try:
            # Allow a timedelta
            offset = offset.total_seconds()
        except (TypeError, AttributeError):
            pass

        self._offset = datetime.timedelta(seconds=_get_supported_offset(offset))

    def utcoffset(self, dt):
        return self._offset

    def dst(self, dt):
        return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        return self._name

    @_validate_fromutc_inputs
    def fromutc(self, dt):
        return dt + self._offset

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.
        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        return False

    def __eq__(self, other):
        if not isinstance(other, tzoffset):
            return NotImplemented

        return self._offset == other._offset

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               repr(self._name),
                               int(self._offset.total_seconds()))

    __reduce__ = object.__reduce__


class tzlocal(_tzinfo):
    """
    A :class:`tzinfo` subclass built around the ``time`` timezone functions.
    """
    def __init__(self):
        super(tzlocal, self).__init__()

        self._std_offset = datetime.timedelta(seconds=-time.timezone)
        if time.daylight:
            self._dst_offset = datetime.timedelta(seconds=-time.altzone)
        else:
            self._dst_offset = self._std_offset

        self._dst_saved = self._dst_offset - self._std_offset
        self._hasdst = bool(self._dst_saved)
        self._tznames = tuple(time.tzname)

    def utcoffset(self, dt):
        if dt is None and self._hasdst:
            return None

        if self._isdst(dt):
            return self._dst_offset
        else:
            return self._std_offset

    def dst(self, dt):
        if dt is None and self._hasdst:
            return None

        if self._isdst(dt):
            return self._dst_offset - self._std_offset
        else:
            return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        return self._tznames[self._isdst(dt)]

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        naive_dst = self._naive_is_dst(dt)
        return (not naive_dst and
                (naive_dst != self._naive_is_dst(dt - self._dst_saved)))

    def _naive_is_dst(self, dt):
        timestamp = _datetime_to_timestamp(dt)
        return time.localtime(timestamp + time.timezone).tm_isdst

    def _isdst(self, dt, fold_naive=True):
        # We can't use mktime here. It is unstable when deciding if
        # the hour near to a change is DST or not.
        #
        # timestamp = time.mktime((dt.year, dt.month, dt.day, dt.hour,
        #                         dt.minute, dt.second, dt.weekday(), 0, -1))
        # return time.localtime(timestamp).tm_isdst
        #
        # The code above yields the following result:
        #
        # >>> import tz, datetime
        # >>> t = tz.tzlocal()
        # >>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        # 'BRDT'
        # >>> datetime.datetime(2003,2,16,0,tzinfo=t).tzname()
        # 'BRST'
        # >>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        # 'BRST'
        # >>> datetime.datetime(2003,2,15,22,tzinfo=t).tzname()
        # 'BRDT'
        # >>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        # 'BRDT'
        #
        # Here is a more stable implementation:
        #
        if not self._hasdst:
            return False

        # Check for ambiguous times:
        dstval = self._naive_is_dst(dt)
        fold = getattr(dt, 'fold', None)

        if self.is_ambiguous(dt):
            if fold is not None:
                return not self._fold(dt)
            else:
                return True

        return dstval

    def __eq__(self, other):
        if isinstance(other, tzlocal):
            return (self._std_offset == other._std_offset and
                    self._dst_offset == other._dst_offset)
        elif isinstance(other, tzutc):
            return (not self._hasdst and
                    self._tznames[0] in {'UTC', 'GMT'} and
                    self._std_offset == ZERO)
        elif isinstance(other, tzoffset):
            return (not self._hasdst and
                    self._tznames[0] == other._name and
                    self._std_offset == other._offset)
        else:
            return NotImplemented

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s()" % self.__class__.__name__

    __reduce__ = object.__reduce__


class _ttinfo(object):
    __slots__ = ["offset", "delta", "isdst", "abbr",
                 "isstd", "isgmt", "dstoffset"]

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    def __repr__(self):
        l = []
        for attr in self.__slots__:
            value = getattr(self, attr)
            if value is not None:
                l.append("%s=%s" % (attr, repr(value)))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(l))

    def __eq__(self, other):
        if not isinstance(other, _ttinfo):
            return NotImplemented

        return (self.offset == other.offset and
                self.delta == other.delta and
                self.isdst == other.isdst and
                self.abbr == other.abbr and
                self.isstd == other.isstd and
                self.isgmt == other.isgmt and
                self.dstoffset == other.dstoffset)

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __getstate__(self):
        state = {}
        for name in self.__slots__:
            state[name] = getattr(self, name, None)
        return state

    def __setstate__(self, state):
        for name in self.__slots__:
            if name in state:
                setattr(self, name, state[name])


class _tzfile(object):
    """
    Lightweight class for holding the relevant transition and time zone
    information read from binary tzfiles.
    """
    attrs = ['trans_list', 'trans_list_utc', 'trans_idx', 'ttinfo_list',
             'ttinfo_std', 'ttinfo_dst', 'ttinfo_before', 'ttinfo_first']

    def __init__(self, **kwargs):
        for attr in self.attrs:
            setattr(self, attr, kwargs.get(attr, None))


class tzfile(_tzinfo):
    """
    This is a ``tzinfo`` subclass that allows one to use the ``tzfile(5)``
    format timezone files to extract current and historical zone information.

    :param fileobj:
        This can be an opened file stream or a file name that the time zone
        information can be read from.

    :param filename:
        This is an optional parameter specifying the source of the time zone
        information in the event that ``fileobj`` is a file object. If omitted
        and ``fileobj`` is a file stream, this parameter will be set either to
        ``fileobj``'s ``name`` attribute or to ``repr(fileobj)``.

    See `Sources for Time Zone and Daylight Saving Time Data
    <https://data.iana.org/time-zones/tz-link.html>`_ for more information.
    Time zone files can be compiled from the `IANA Time Zone database files
    <https://www.iana.org/time-zones>`_ with the `zic time zone compiler
    <https://www.freebsd.org/cgi/man.cgi?query=zic&sektion=8>`_

    .. note::

        Only construct a ``tzfile`` directly if you have a specific timezone
        file on disk that you want to read into a Python ``tzinfo`` object.
        If you want to get a ``tzfile`` representing a specific IANA zone,
        (e.g. ``'America/New_York'``), you should call
        :func:`dateutil.tz.gettz` with the zone identifier.


    **Examples:**

    Using the US Eastern time zone as an example, we can see that a ``tzfile``
    provides time zone information for the standard Daylight Saving offsets:

    .. testsetup:: tzfile

        from dateutil.tz import gettz
        from datetime import datetime

    .. doctest:: tzfile

        >>> NYC = gettz('America/New_York')
        >>> NYC
        tzfile('/usr/share/zoneinfo/America/New_York')

        >>> print(datetime(2016, 1, 3, tzinfo=NYC))     # EST
        2016-01-03 00:00:00-05:00

        >>> print(datetime(2016, 7, 7, tzinfo=NYC))     # EDT
        2016-07-07 00:00:00-04:00


    The ``tzfile`` structure contains a fully history of the time zone,
    so historical dates will also have the right offsets. For example, before
    the adoption of the UTC standards, New York used local solar  mean time:

    .. doctest:: tzfile

       >>> print(datetime(1901, 4, 12, tzinfo=NYC))    # LMT
       1901-04-12 00:00:00-04:56

    And during World War II, New York was on "Eastern War Time", which was a
    state of permanent daylight saving time:

    .. doctest:: tzfile

        >>> print(datetime(1944, 2, 7, tzinfo=NYC))    # EWT
        1944-02-07 00:00:00-04:00

    """

    def __init__(self, fileobj, filename=None):
        super(tzfile, self).__init__()

        file_opened_here = False
        if isinstance(fileobj, string_types):
            self._filename = fileobj
            fileobj = open(fileobj, 'rb')
            file_opened_here = True
        elif filename is not None:
            self._filename = filename
        elif hasattr(fileobj, "name"):
            self._filename = fileobj.name
        else:
            self._filename = repr(fileobj)

        if fileobj is not None:
            if not file_opened_here:
                fileobj = _nullcontext(fileobj)

            with fileobj as file_stream:
                tzobj = self._read_tzfile(file_stream)

            self._set_tzdata(tzobj)

    def _set_tzdata(self, tzobj):
        """ Set the time zone data of this object from a _tzfile object """
        # Copy the relevant attributes over as private attributes
        for attr in _tzfile.attrs:
            setattr(self, '_' + attr, getattr(tzobj, attr))

    def _read_tzfile(self, fileobj):
        out = _tzfile()

        # From tzfile(5):
        #
        # The time zone information files used by tzset(3)
        # begin with the magic characters "TZif" to identify
        # them as time zone information files, followed by
        # sixteen bytes reserved for future use, followed by
        # six four-byte values of type long, written in a
        # ``standard'' byte order (the high-order  byte
        # of the value is written first).
        if fileobj.read(4).decode() != "TZif":
            raise ValueError("magic not found")

        fileobj.read(16)

        (
            # The number of UTC/local indicators stored in the file.
            ttisgmtcnt,

            # The number of standard/wall indicators stored in the file.
            ttisstdcnt,

            # The number of leap seconds for which data is
            # stored in the file.
            leapcnt,

            # The number of "transition times" for which data
            # is stored in the file.
            timecnt,

            # The number of "local time types" for which data
            # is stored in the file (must not be zero).
            typecnt,

            # The  number  of  characters  of "time zone
            # abbreviation strings" stored in the file.
            charcnt,

        ) = struct.unpack(">6l", fileobj.read(24))

        # The above header is followed by tzh_timecnt four-byte
        # values  of  type long,  sorted  in ascending order.
        # These values are written in ``standard'' byte order.
        # Each is used as a transition time (as  returned  by
        # time(2)) at which the rules for computing local time
        # change.

        if timecnt:
            out.trans_list_utc = list(struct.unpack(">%dl" % timecnt,
                                                    fileobj.read(timecnt*4)))
        else:
            out.trans_list_utc = []

        # Next come tzh_timecnt one-byte values of type unsigned
        # char; each one tells which of the different types of
        # ``local time'' types described in the file is associated
        # with the same-indexed transition time. These values
        # serve as indices into an array of ttinfo structures that
        # appears next in the file.

        if timecnt:
            out.trans_idx = struct.unpack(">%dB" % timecnt,
                                          fileobj.read(timecnt))
        else:
            out.trans_idx = []

        # Each ttinfo structure is written as a four-byte value
        # for tt_gmtoff  of  type long,  in  a  standard  byte
        # order, followed  by a one-byte value for tt_isdst
        # and a one-byte  value  for  tt_abbrind.   In  each
        # structure, tt_gmtoff  gives  the  number  of
        # seconds to be added to UTC, tt_isdst tells whether
        # tm_isdst should be set by  localtime(3),  and
        # tt_abbrind serves  as an index into the array of
        # time zone abbreviation characters that follow the
        # ttinfo structure(s) in the file.

        ttinfo = []

        for i in range(typecnt):
            ttinfo.append(struct.unpack(">lbb", fileobj.read(6)))

        abbr = fileobj.read(charcnt).decode()

        # Then there are tzh_leapcnt pairs of four-byte
        # values, written in  standard byte  order;  the
        # first  value  of  each pair gives the time (as
        # returned by time(2)) at which a leap second
        # occurs;  the  second  gives the  total  number of
        # leap seconds to be applied after the given time.
        # The pairs of values are sorted in ascending order
        # by time.

        # Not used, for now (but seek for correct file position)
        if leapcnt:
            fileobj.seek(leapcnt * 8, os.SEEK_CUR)

        # Then there are tzh_ttisstdcnt standard/wall
        # indicators, each stored as a one-byte value;
        # they tell whether the transition times associated
        # with local time types were specified as standard
        # time or wall clock time, and are used when
        # a time zone file is used in handling POSIX-style
        # time zone environment variables.

        if ttisstdcnt:
            isstd = struct.unpack(">%db" % ttisstdcnt,
                                  fileobj.read(ttisstdcnt))

        # Finally, there are tzh_ttisgmtcnt UTC/local
        # indicators, each stored as a one-byte value;
        # they tell whether the transition times associated
        # with local time types were specified as UTC or
        # local time, and are used when a time zone file
        # is used in handling POSIX-style time zone envi-
        # ronment variables.

        if ttisgmtcnt:
            isgmt = struct.unpack(">%db" % ttisgmtcnt,
                                  fileobj.read(ttisgmtcnt))

        # Build ttinfo list
        out.ttinfo_list = []
        for i in range(typecnt):
            gmtoff, isdst, abbrind = ttinfo[i]
            gmtoff = _get_supported_offset(gmtoff)
            tti = _ttinfo()
            tti.offset = gmtoff
            tti.dstoffset = datetime.timedelta(0)
            tti.delta = datetime.timedelta(seconds=gmtoff)
            tti.isdst = isdst
            tti.abbr = abbr[abbrind:abbr.find('\x00', abbrind)]
            tti.isstd = (ttisstdcnt > i and isstd[i] != 0)
            tti.isgmt = (ttisgmtcnt > i and isgmt[i] != 0)
            out.ttinfo_list.append(tti)

        # Replace ttinfo indexes for ttinfo objects.
        out.trans_idx = [out.ttinfo_list[idx] for idx in out.trans_idx]

        # Set standard, dst, and before ttinfos. before will be
        # used when a given time is before any transitions,
        # and will be set to the first non-dst ttinfo, or to
        # the first dst, if all of them are dst.
        out.ttinfo_std = None
        out.ttinfo_dst = None
        out.ttinfo_before = None
        if out.ttinfo_list:
            if not out.trans_list_utc:
                out.ttinfo_std = out.ttinfo_first = out.ttinfo_list[0]
            else:
                for i in range(timecnt-1, -1, -1):
                    tti = out.trans_idx[i]
                    if not out.ttinfo_std and not tti.isdst:
                        out.ttinfo_std = tti
                    elif not out.ttinfo_dst and tti.isdst:
                        out.ttinfo_dst = tti

                    if out.ttinfo_std and out.ttinfo_dst:
                        break
                else:
                    if out.ttinfo_dst and not out.ttinfo_std:
                        out.ttinfo_std = out.ttinfo_dst

                for tti in out.ttinfo_list:
                    if not tti.isdst:
                        out.ttinfo_before = tti
                        break
                else:
                    out.ttinfo_before = out.ttinfo_list[0]

        # Now fix transition times to become relative to wall time.
        #
        # I'm not sure about this. In my tests, the tz source file
        # is setup to wall time, and in the binary file isstd and
        # isgmt are off, so it should be in wall time. OTOH, it's
        # always in gmt time. Let me know if you have comments
        # about this.
        lastdst = None
        lastoffset = None
        lastdstoffset = None
        lastbaseoffset = None
        out.trans_list = []

        for i, tti in enumerate(out.trans_idx):
            offset = tti.offset
            dstoffset = 0

            if lastdst is not None:
                if tti.isdst:
                    if not lastdst:
                        dstoffset = offset - lastoffset

                    if not dstoffset and lastdstoffset:
                        dstoffset = lastdstoffset

                    tti.dstoffset = datetime.timedelta(seconds=dstoffset)
                    lastdstoffset = dstoffset

            # If a time zone changes its base offset during a DST transition,
            # then you need to adjust by the previous base offset to get the
            # transition time in local time. Otherwise you use the current
            # base offset. Ideally, I would have some mathematical proof of
            # why this is true, but I haven't really thought about it enough.
            baseoffset = offset - dstoffset
            adjustment = baseoffset
            if (lastbaseoffset is not None and baseoffset != lastbaseoffset
                    and tti.isdst != lastdst):
                # The base DST has changed
                adjustment = lastbaseoffset

            lastdst = tti.isdst
            lastoffset = offset
            lastbaseoffset = baseoffset

            out.trans_list.append(out.trans_list_utc[i] + adjustment)

        out.trans_idx = tuple(out.trans_idx)
        out.trans_list = tuple(out.trans_list)
        out.trans_list_utc = tuple(out.trans_list_utc)

        return out

    def _find_last_transition(self, dt, in_utc=False):
        # If there's no list, there are no transitions to find
        if not self._trans_list:
            return None

        timestamp = _datetime_to_timestamp(dt)

        # Find where the timestamp fits in the transition list - if the
        # timestamp is a transition time, it's part of the "after" period.
        trans_list = self._trans_list_utc if in_utc else self._trans_list
        idx = bisect.bisect_right(trans_list, timestamp)

        # We want to know when the previous transition was, so subtract off 1
        return idx - 1

    def _get_ttinfo(self, idx):
        # For no list or after the last transition, default to _ttinfo_std
        if idx is None or (idx + 1) >= len(self._trans_list):
            return self._ttinfo_std

        # If there is a list and the time is before it, return _ttinfo_before
        if idx < 0:
            return self._ttinfo_before

        return self._trans_idx[idx]

    def _find_ttinfo(self, dt):
        idx = self._resolve_ambiguous_time(dt)

        return self._get_ttinfo(idx)

    def fromutc(self, dt):
        """
        The ``tzfile`` implementation of :py:func:`datetime.tzinfo.fromutc`.

        :param dt:
            A :py:class:`datetime.datetime` object.

        :raises TypeError:
            Raised if ``dt`` is not a :py:class:`datetime.datetime` object.

        :raises ValueError:
            Raised if this is called with a ``dt`` which does not have this
            ``tzinfo`` attached.

        :return:
            Returns a :py:class:`datetime.datetime` object representing the
            wall time in ``self``'s time zone.
        """
        # These isinstance checks are in datetime.tzinfo, so we'll preserve
        # them, even if we don't care about duck typing.
        if not isinstance(dt, datetime.datetime):
            raise TypeError("fromutc() requires a datetime argument")

        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        # First treat UTC as wall time and get the transition we're in.
        idx = self._find_last_transition(dt, in_utc=True)
        tti = self._get_ttinfo(idx)

        dt_out = dt + datetime.timedelta(seconds=tti.offset)

        fold = self.is_ambiguous(dt_out, idx=idx)

        return enfold(dt_out, fold=int(fold))

    def is_ambiguous(self, dt, idx=None):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        if idx is None:
            idx = self._find_last_transition(dt)

        # Calculate the difference in offsets from current to previous
        timestamp = _datetime_to_timestamp(dt)
        tti = self._get_ttinfo(idx)

        if idx is None or idx <= 0:
            return False

        od = self._get_ttinfo(idx - 1).offset - tti.offset
        tt = self._trans_list[idx]          # Transition time

        return timestamp < tt + od

    def _resolve_ambiguous_time(self, dt):
        idx = self._find_last_transition(dt)

        # If we have no transitions, return the index
        _fold = self._fold(dt)
        if idx is None or idx == 0:
            return idx

        # If it's ambiguous and we're in a fold, shift to a different index.
        idx_offset = int(not _fold and self.is_ambiguous(dt, idx))

        return idx - idx_offset

    def utcoffset(self, dt):
        if dt is None:
            return None

        if not self._ttinfo_std:
            return ZERO

        return self._find_ttinfo(dt).delta

    def dst(self, dt):
        if dt is None:
            return None

        if not self._ttinfo_dst:
            return ZERO

        tti = self._find_ttinfo(dt)

        if not tti.isdst:
            return ZERO

        # The documentation says that utcoffset()-dst() must
        # be constant for every dt.
        return tti.dstoffset

    @tzname_in_python2
    def tzname(self, dt):
        if not self._ttinfo_std or dt is None:
            return None
        return self._find_ttinfo(dt).abbr

    def __eq__(self, other):
        if not isinstance(other, tzfile):
            return NotImplemented
        return (self._trans_list == other._trans_list and
                self._trans_idx == other._trans_idx and
                self._ttinfo_list == other._ttinfo_list)

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self._filename))

    def __reduce__(self):
        return self.__reduce_ex__(None)

    def __reduce_ex__(self, protocol):
        return (self.__class__, (None, self._filename), self.__dict__)


class tzrange(tzrangebase):
    """
    The ``tzrange`` object is a time zone specified by a set of offsets and
    abbreviations, equivalent to the way the ``TZ`` variable can be specified
    in POSIX-like systems, but using Python delta objects to specify DST
    start, end and offsets.

    :param stdabbr:
        The abbreviation for standard time (e.g. ``'EST'``).

    :param stdoffset:
        An integer or :class:`datetime.timedelta` object or equivalent
        specifying the base offset from UTC.

        If unspecified, +00:00 is used.

    :param dstabbr:
        The abbreviation for DST / "Summer" time (e.g. ``'EDT'``).

        If specified, with no other DST information, DST is assumed to occur
        and the default behavior or ``dstoffset``, ``start`` and ``end`` is
        used. If unspecified and no other DST information is specified, it
        is assumed that this zone has no DST.

        If this is unspecified and other DST information is *is* specified,
        DST occurs in the zone but the time zone abbreviation is left
        unchanged.

    :param dstoffset:
        A an integer or :class:`datetime.timedelta` object or equivalent
        specifying the UTC offset during DST. If unspecified and any other DST
        information is specified, it is assumed to be the STD offset +1 hour.

    :param start:
        A :class:`relativedelta.relativedelta` object or equivalent specifying
        the time and time of year that daylight savings time starts. To
        specify, for example, that DST starts at 2AM on the 2nd Sunday in
        March, pass:

            ``relativedelta(hours=2, month=3, day=1, weekday=SU(+2))``

        If unspecified and any other DST information is specified, the default
        value is 2 AM on the first Sunday in April.

    :param end:
        A :class:`relativedelta.relativedelta` object or equivalent
        representing the time and time of year that daylight savings time
        ends, with the same specification method as in ``start``. One note is
        that this should point to the first time in the *standard* zone, so if
        a transition occurs at 2AM in the DST zone and the clocks are set back
        1 hour to 1AM, set the ``hours`` parameter to +1.


    **Examples:**

    .. testsetup:: tzrange

        from dateutil.tz import tzrange, tzstr

    .. doctest:: tzrange

        >>> tzstr('EST5EDT') == tzrange("EST", -18000, "EDT")
        True

        >>> from dateutil.relativedelta import *
        >>> range1 = tzrange("EST", -18000, "EDT")
        >>> range2 = tzrange("EST", -18000, "EDT", -14400,
        ...                  relativedelta(hours=+2, month=4, day=1,
        ...                                weekday=SU(+1)),
        ...                  relativedelta(hours=+1, month=10, day=31,
        ...                                weekday=SU(-1)))
        >>> tzstr('EST5EDT') == range1 == range2
        True

    """
    def __init__(self, stdabbr, stdoffset=None,
                 dstabbr=None, dstoffset=None,
                 start=None, end=None):

        global relativedelta
        from dateutil import relativedelta

        self._std_abbr = stdabbr
        self._dst_abbr = dstabbr

        try:
            stdoffset = stdoffset.total_seconds()
        except (TypeError, AttributeError):
            pass

        try:
            dstoffset = dstoffset.total_seconds()
        except (TypeError, AttributeError):
            pass

        if stdoffset is not None:
            self._std_offset = datetime.timedelta(seconds=stdoffset)
        else:
            self._std_offset = ZERO

        if dstoffset is not None:
            self._dst_offset = datetime.timedelta(seconds=dstoffset)
        elif dstabbr and stdoffset is not None:
            self._dst_offset = self._std_offset + datetime.timedelta(hours=+1)
        else:
            self._dst_offset = ZERO

        if dstabbr and start is None:
            self._start_delta = relativedelta.relativedelta(
                hours=+2, month=4, day=1, weekday=relativedelta.SU(+1))
        else:
            self._start_delta = start

        if dstabbr and end is None:
            self._end_delta = relativedelta.relativedelta(
                hours=+1, month=10, day=31, weekday=relativedelta.SU(-1))
        else:
            self._end_delta = end

        self._dst_base_offset_ = self._dst_offset - self._std_offset
        self.hasdst = bool(self._start_delta)

    def transitions(self, year):
        """
        For a given year, get the DST on and off transition times, expressed
        always on the standard time side. For zones with no transitions, this
        function returns ``None``.

        :param year:
            The year whose transitions you would like to query.

        :return:
            Returns a :class:`tuple` of :class:`datetime.datetime` objects,
            ``(dston, dstoff)`` for zones with an annual DST transition, or
            ``None`` for fixed offset zones.
        """
        if not self.hasdst:
            return None

        base_year = datetime.datetime(year, 1, 1)

        start = base_year + self._start_delta
        end = base_year + self._end_delta

        return (start, end)

    def __eq__(self, other):
        if not isinstance(other, tzrange):
            return NotImplemented

        return (self._std_abbr == other._std_abbr and
                self._dst_abbr == other._dst_abbr and
                self._std_offset == other._std_offset and
                self._dst_offset == other._dst_offset and
                self._start_delta == other._start_delta and
                self._end_delta == other._end_delta)

    @property
    def _dst_base_offset(self):
        return self._dst_base_offset_


@six.add_metaclass(_TzStrFactory)
class tzstr(tzrange):
    """
    ``tzstr`` objects are time zone objects specified by a time-zone string as
    it would be passed to a ``TZ`` variable on POSIX-style systems (see
    the `GNU C Library: TZ Variable`_ for more details).

    There is one notable exception, which is that POSIX-style time zones use an
    inverted offset format, so normally ``GMT+3`` would be parsed as an offset
    3 hours *behind* GMT. The ``tzstr`` time zone object will parse this as an
    offset 3 hours *ahead* of GMT. If you would like to maintain the POSIX
    behavior, pass a ``True`` value to ``posix_offset``.

    The :class:`tzrange` object provides the same functionality, but is
    specified using :class:`relativedelta.relativedelta` objects. rather than
    strings.

    :param s:
        A time zone string in ``TZ`` variable format. This can be a
        :class:`bytes` (2.x: :class:`str`), :class:`str` (2.x:
        :class:`unicode`) or a stream emitting unicode characters
        (e.g. :class:`StringIO`).

    :param posix_offset:
        Optional. If set to ``True``, interpret strings such as ``GMT+3`` or
        ``UTC+3`` as being 3 hours *behind* UTC rather than ahead, per the
        POSIX standard.

    .. caution::

        Prior to version 2.7.0, this function also supported time zones
        in the format:

            * ``EST5EDT,4,0,6,7200,10,0,26,7200,3600``
            * ``EST5EDT,4,1,0,7200,10,-1,0,7200,3600``

        This format is non-standard and has been deprecated; this function
        will raise a :class:`DeprecatedTZFormatWarning` until
        support is removed in a future version.

    .. _`GNU C Library: TZ Variable`:
        https://www.gnu.org/software/libc/manual/html_node/TZ-Variable.html
    """
    def __init__(self, s, posix_offset=False):
        global parser
        from dateutil.parser import _parser as parser

        self._s = s

        res = parser._parsetz(s)
        if res is None or res.any_unused_tokens:
            raise ValueError("unknown string format")

        # Here we break the compatibility with the TZ variable handling.
        # GMT-3 actually *means* the timezone -3.
        if res.stdabbr in ("GMT", "UTC") and not posix_offset:
            res.stdoffset *= -1

        # We must initialize it first, since _delta() needs
        # _std_offset and _dst_offset set. Use False in start/end
        # to avoid building it two times.
        tzrange.__init__(self, res.stdabbr, res.stdoffset,
                         res.dstabbr, res.dstoffset,
                         start=False, end=False)

        if not res.dstabbr:
            self._start_delta = None
            self._end_delta = None
        else:
            self._start_delta = self._delta(res.start)
            if self._start_delta:
                self._end_delta = self._delta(res.end, isend=1)

        self.hasdst = bool(self._start_delta)

    def _delta(self, x, isend=0):
        from dateutil import relativedelta
        kwargs = {}
        if x.month is not None:
            kwargs["month"] = x.month
            if x.weekday is not None:
                kwargs["weekday"] = relativedelta.weekday(x.weekday, x.week)
                if x.week > 0:
                    kwargs["day"] = 1
                else:
                    kwargs["day"] = 31
            elif x.day:
                kwargs["day"] = x.day
        elif x.yday is not None:
            kwargs["yearday"] = x.yday
        elif x.jyday is not None:
            kwargs["nlyearday"] = x.jyday
        if not kwargs:
            # Default is to start on first sunday of april, and end
            # on last sunday of october.
            if not isend:
                kwargs["month"] = 4
                kwargs["day"] = 1
                kwargs["weekday"] = relativedelta.SU(+1)
            else:
                kwargs["month"] = 10
                kwargs["day"] = 31
                kwargs["weekday"] = relativedelta.SU(-1)
        if x.time is not None:
            kwargs["seconds"] = x.time
        else:
            # Default is 2AM.
            kwargs["seconds"] = 7200
        if isend:
            # Convert to standard time, to follow the documented way
            # of working with the extra hour. See the documentation
            # of the tzinfo class.
            delta = self._dst_offset - self._std_offset
            kwargs["seconds"] -= delta.seconds + delta.days * 86400
        return relativedelta.relativedelta(**kwargs)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self._s))


class _tzicalvtzcomp(object):
    def __init__(self, tzoffsetfrom, tzoffsetto, isdst,
                 tzname=None, rrule=None):
        self.tzoffsetfrom = datetime.timedelta(seconds=tzoffsetfrom)
        self.tzoffsetto = datetime.timedelta(seconds=tzoffsetto)
        self.tzoffsetdiff = self.tzoffsetto - self.tzoffsetfrom
        self.isdst = isdst
        self.tzname = tzname
        self.rrule = rrule


class _tzicalvtz(_tzinfo):
    def __init__(self, tzid, comps=[]):
        super(_tzicalvtz, self).__init__()

        self._tzid = tzid
        self._comps = comps
        self._cachedate = []
        self._cachecomp = []
        self._cache_lock = _thread.allocate_lock()

    def _find_comp(self, dt):
        if len(self._comps) == 1:
            return self._comps[0]

        dt = dt.replace(tzinfo=None)

        try:
            with self._cache_lock:
                return self._cachecomp[self._cachedate.index(
                    (dt, self._fold(dt)))]
        except ValueError:
            pass

        lastcompdt = None
        lastcomp = None

        for comp in self._comps:
            compdt = self._find_compdt(comp, dt)

            if compdt and (not lastcompdt or lastcompdt < compdt):
                lastcompdt = compdt
                lastcomp = comp

        if not lastcomp:
            # RFC says nothing about what to do when a given
            # time is before the first onset date. We'll look for the
            # first standard component, or the first component, if
            # none is found.
            for comp in self._comps:
                if not comp.isdst:
                    lastcomp = comp
                    break
            else:
                lastcomp = comp[0]

        with self._cache_lock:
            self._cachedate.insert(0, (dt, self._fold(dt)))
            self._cachecomp.insert(0, lastcomp)

            if len(self._cachedate) > 10:
                self._cachedate.pop()
                self._cachecomp.pop()

        return lastcomp

    def _find_compdt(self, comp, dt):
        if comp.tzoffsetdiff < ZERO and self._fold(dt):
            dt -= comp.tzoffsetdiff

        compdt = comp.rrule.before(dt, inc=True)

        return compdt

    def utcoffset(self, dt):
        if dt is None:
            return None

        return self._find_comp(dt).tzoffsetto

    def dst(self, dt):
        comp = self._find_comp(dt)
        if comp.isdst:
            return comp.tzoffsetdiff
        else:
            return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        return self._find_comp(dt).tzname

    def __repr__(self):
        return "<tzicalvtz %s>" % repr(self._tzid)

    __reduce__ = object.__reduce__


class tzical(object):
    """
    This object is designed to parse an iCalendar-style ``VTIMEZONE`` structure
    as set out in `RFC 5545`_ Section 4.6.5 into one or more `tzinfo` objects.

    :param `fileobj`:
        A file or stream in iCalendar format, which should be UTF-8 encoded
        with CRLF endings.

    .. _`RFC 5545`: https://tools.ietf.org/html/rfc5545
    """
    def __init__(self, fileobj):
        global rrule
        from dateutil import rrule

        if isinstance(fileobj, string_types):
            self._s = fileobj
            # ical should be encoded in UTF-8 with CRLF
            fileobj = open(fileobj, 'r')
        else:
            self._s = getattr(fileobj, 'name', repr(fileobj))
            fileobj = _nullcontext(fileobj)

        self._vtz = {}

        with fileobj as fobj:
            self._parse_rfc(fobj.read())

    def keys(self):
        """
        Retrieves the available time zones as a list.
        """
        return list(self._vtz.keys())

    def get(self, tzid=None):
        """
        Retrieve a :py:class:`datetime.tzinfo` object by its ``tzid``.

        :param tzid:
            If there is exactly one time zone available, omitting ``tzid``
            or passing :py:const:`None` value returns it. Otherwise a valid
            key (which can be retrieved from :func:`keys`) is required.

        :raises ValueError:
            Raised if ``tzid`` is not specified but there are either more
            or fewer than 1 zone defined.

        :returns:
            Returns either a :py:class:`datetime.tzinfo` object representing
            the relevant time zone or :py:const:`None` if the ``tzid`` was
            not found.
        """
        if tzid is None:
            if len(self._vtz) == 0:
                raise ValueError("no timezones defined")
            elif len(self._vtz) > 1:
                raise ValueError("more than one timezone available")
            tzid = next(iter(self._vtz))

        return self._vtz.get(tzid)

    def _parse_offset(self, s):
        s = s.strip()
        if not s:
            raise ValueError("empty offset")
        if s[0] in ('+', '-'):
            signal = (-1, +1)[s[0] == '+']
            s = s[1:]
        else:
            signal = +1
        if len(s) == 4:
            return (int(s[:2]) * 3600 + int(s[2:]) * 60) * signal
        elif len(s) == 6:
            return (int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:])) * signal
        else:
            raise ValueError("invalid offset: " + s)

    def _parse_rfc(self, s):
        lines = s.splitlines()
        if not lines:
            raise ValueError("empty string")

        # Unfold
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line:
                del lines[i]
            elif i > 0 and line[0] == " ":
                lines[i-1] += line[1:]
                del lines[i]
            else:
                i += 1

        tzid = None
        comps = []
        invtz = False
        comptype = None
        for line in lines:
            if not line:
                continue
            name, value = line.split(':', 1)
            parms = name.split(';')
            if not parms:
                raise ValueError("empty property name")
            name = parms[0].upper()
            parms = parms[1:]
            if invtz:
                if name == "BEGIN":
                    if value in ("STANDARD", "DAYLIGHT"):
                        # Process component
                        pass
                    else:
                        raise ValueError("unknown component: "+value)
                    comptype = value
                    founddtstart = False
                    tzoffsetfrom = None
                    tzoffsetto = None
                    rrulelines = []
                    tzname = None
                elif name == "END":
                    if value == "VTIMEZONE":
                        if comptype:
                            raise ValueError("component not closed: "+comptype)
                        if not tzid:
                            raise ValueError("mandatory TZID not found")
                        if not comps:
                            raise ValueError(
                                "at least one component is needed")
                        # Process vtimezone
                        self._vtz[tzid] = _tzicalvtz(tzid, comps)
                        invtz = False
                    elif value == comptype:
                        if not founddtstart:
                            raise ValueError("mandatory DTSTART not found")
                        if tzoffsetfrom is None:
                            raise ValueError(
                                "mandatory TZOFFSETFROM not found")
                        if tzoffsetto is None:
                            raise ValueError(
                                "mandatory TZOFFSETFROM not found")
                        # Process component
                        rr = None
                        if rrulelines:
                            rr = rrule.rrulestr("\n".join(rrulelines),
                                                compatible=True,
                                                ignoretz=True,
                                                cache=True)
                        comp = _tzicalvtzcomp(tzoffsetfrom, tzoffsetto,
                                              (comptype == "DAYLIGHT"),
                                              tzname, rr)
                        comps.append(comp)
                        comptype = None
                    else:
                        raise ValueError("invalid component end: "+value)
                elif comptype:
                    if name == "DTSTART":
                        # DTSTART in VTIMEZONE takes a subset of valid RRULE
                        # values under RFC 5545.
                        for parm in parms:
                            if parm != 'VALUE=DATE-TIME':
                                msg = ('Unsupported DTSTART param in ' +
                                       'VTIMEZONE: ' + parm)
                                raise ValueError(msg)
                        rrulelines.append(line)
                        founddtstart = True
                    elif name in ("RRULE", "RDATE", "EXRULE", "EXDATE"):
                        rrulelines.append(line)
                    elif name == "TZOFFSETFROM":
                        if parms:
                            raise ValueError(
                                "unsupported %s parm: %s " % (name, parms[0]))
                        tzoffsetfrom = self._parse_offset(value)
                    elif name == "TZOFFSETTO":
                        if parms:
                            raise ValueError(
                                "unsupported TZOFFSETTO parm: "+parms[0])
                        tzoffsetto = self._parse_offset(value)
                    elif name == "TZNAME":
                        if parms:
                            raise ValueError(
                                "unsupported TZNAME parm: "+parms[0])
                        tzname = value
                    elif name == "COMMENT":
                        pass
                    else:
                        raise ValueError("unsupported property: "+name)
                else:
                    if name == "TZID":
                        if parms:
                            raise ValueError(
                                "unsupported TZID parm: "+parms[0])
                        tzid = value
                    elif name in ("TZURL", "LAST-MODIFIED", "COMMENT"):
                        pass
                    else:
                        raise ValueError("unsupported property: "+name)
            elif name == "BEGIN" and value == "VTIMEZONE":
                tzid = None
                comps = []
                invtz = True

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self._s))


if sys.platform != "win32":
    TZFILES = ["/etc/localtime", "localtime"]
    TZPATHS = ["/usr/share/zoneinfo",
               "/usr/lib/zoneinfo",
               "/usr/share/lib/zoneinfo",
               "/etc/zoneinfo"]
else:
    TZFILES = []
    TZPATHS = []


def __get_gettz():
    tzlocal_classes = (tzlocal,)
    if tzwinlocal is not None:
        tzlocal_classes += (tzwinlocal,)

    class GettzFunc(object):
        """
        Retrieve a time zone object from a string representation

        This function is intended to retrieve the :py:class:`tzinfo` subclass
        that best represents the time zone that would be used if a POSIX
        `TZ variable`_ were set to the same value.

        If no argument or an empty string is passed to ``gettz``, local time
        is returned:

        .. code-block:: python3

            >>> gettz()
            tzfile('/etc/localtime')

        This function is also the preferred way to map IANA tz database keys
        to :class:`tzfile` objects:

        .. code-block:: python3

            >>> gettz('Pacific/Kiritimati')
            tzfile('/usr/share/zoneinfo/Pacific/Kiritimati')

        On Windows, the standard is extended to include the Windows-specific
        zone names provided by the operating system:

        .. code-block:: python3

            >>> gettz('Egypt Standard Time')
            tzwin('Egypt Standard Time')

        Passing a GNU ``TZ`` style string time zone specification returns a
        :class:`tzstr` object:

        .. code-block:: python3

            >>> gettz('AEST-10AEDT-11,M10.1.0/2,M4.1.0/3')
            tzstr('AEST-10AEDT-11,M10.1.0/2,M4.1.0/3')

        :param name:
            A time zone name (IANA, or, on Windows, Windows keys), location of
            a ``tzfile(5)`` zoneinfo file or ``TZ`` variable style time zone
            specifier. An empty string, no argument or ``None`` is interpreted
            as local time.

        :return:
            Returns an instance of one of ``dateutil``'s :py:class:`tzinfo`
            subclasses.

        .. versionchanged:: 2.7.0

            After version 2.7.0, any two calls to ``gettz`` using the same
            input strings will return the same object:

            .. code-block:: python3

                >>> tz.gettz('America/Chicago') is tz.gettz('America/Chicago')
                True

            In addition to improving performance, this ensures that
            `"same zone" semantics`_ are used for datetimes in the same zone.


        .. _`TZ variable`:
            https://www.gnu.org/software/libc/manual/html_node/TZ-Variable.html

        .. _`"same zone" semantics`:
            https://blog.ganssle.io/articles/2018/02/aware-datetime-arithmetic.html
        """
        def __init__(self):

            self.__instances = weakref.WeakValueDictionary()
            self.__strong_cache_size = 8
            self.__strong_cache = OrderedDict()
            self._cache_lock = _thread.allocate_lock()

        def __call__(self, name=None):
            with self._cache_lock:
                rv = self.__instances.get(name, None)

                if rv is None:
                    rv = self.nocache(name=name)
                    if not (name is None
                            or isinstance(rv, tzlocal_classes)
                            or rv is None):
                        # tzlocal is slightly more complicated than the other
                        # time zone providers because it depends on environment
                        # at construction time, so don't cache that.
                        #
                        # We also cannot store weak references to None, so we
                        # will also not store that.
                        self.__instances[name] = rv
                    else:
                        # No need for strong caching, return immediately
                        return rv

                self.__strong_cache[name] = self.__strong_cache.pop(name, rv)

                if len(self.__strong_cache) > self.__strong_cache_size:
                    self.__strong_cache.popitem(last=False)

            return rv

        def set_cache_size(self, size):
            with self._cache_lock:
                self.__strong_cache_size = size
                while len(self.__strong_cache) > size:
                    self.__strong_cache.popitem(last=False)

        def cache_clear(self):
            with self._cache_lock:
                self.__instances = weakref.WeakValueDictionary()
                self.__strong_cache.clear()

        @staticmethod
        def nocache(name=None):
            """A non-cached version of gettz"""
            tz = None
            if not name:
                try:
                    name = os.environ["TZ"]
                except KeyError:
                    pass
            if name is None or name in ("", ":"):
                for filepath in TZFILES:
                    if not os.path.isabs(filepath):
                        filename = filepath
                        for path in TZPATHS:
                            filepath = os.path.join(path, filename)
                            if os.path.isfile(filepath):
                                break
                        else:
                            continue
                    if os.path.isfile(filepath):
                        try:
                            tz = tzfile(filepath)
                            break
                        except (IOError, OSError, ValueError):
                            pass
                else:
                    tz = tzlocal()
            else:
                try:
                    if name.startswith(":"):
                        name = name[1:]
                except TypeError as e:
                    if isinstance(name, bytes):
                        new_msg = "gettz argument should be str, not bytes"
                        six.raise_from(TypeError(new_msg), e)
                    else:
                        raise
                if os.path.isabs(name):
                    if os.path.isfile(name):
                        tz = tzfile(name)
                    else:
                        tz = None
                else:
                    for path in TZPATHS:
                        filepath = os.path.join(path, name)
                        if not os.path.isfile(filepath):
                            filepath = filepath.replace(' ', '_')
                            if not os.path.isfile(filepath):
                                continue
                        try:
                            tz = tzfile(filepath)
                            break
                        except (IOError, OSError, ValueError):
                            pass
                    else:
                        tz = None
                        if tzwin is not None:
                            try:
                                tz = tzwin(name)
                            except (WindowsError, UnicodeEncodeError):
                                # UnicodeEncodeError is for Python 2.7 compat
                                tz = None

                        if not tz:
                            from dateutil.zoneinfo import get_zonefile_instance
                            tz = get_zonefile_instance().get(name)

                        if not tz:
                            for c in name:
                                # name is not a tzstr unless it has at least
                                # one offset. For short values of "name", an
                                # explicit for loop seems to be the fastest way
                                # To determine if a string contains a digit
                                if c in "0123456789":
                                    try:
                                        tz = tzstr(name)
                                    except ValueError:
                                        pass
                                    break
                            else:
                                if name in ("GMT", "UTC"):
                                    tz = UTC
                                elif name in time.tzname:
                                    tz = tzlocal()
            return tz

    return GettzFunc()


gettz = __get_gettz()
del __get_gettz


def datetime_exists(dt, tz=None):
    """
    Given a datetime and a time zone, determine whether or not a given datetime
    would fall in a gap.

    :param dt:
        A :class:`datetime.datetime` (whose time zone will be ignored if ``tz``
        is provided.)

    :param tz:
        A :class:`datetime.tzinfo` with support for the ``fold`` attribute. If
        ``None`` or not provided, the datetime's own time zone will be used.

    :return:
        Returns a boolean value whether or not the "wall time" exists in
        ``tz``.

    .. versionadded:: 2.7.0
    """
    if tz is None:
        if dt.tzinfo is None:
            raise ValueError('Datetime is naive and no time zone provided.')
        tz = dt.tzinfo

    dt = dt.replace(tzinfo=None)

    # This is essentially a test of whether or not the datetime can survive
    # a round trip to UTC.
    dt_rt = dt.replace(tzinfo=tz).astimezone(UTC).astimezone(tz)
    dt_rt = dt_rt.replace(tzinfo=None)

    return dt == dt_rt


def datetime_ambiguous(dt, tz=None):
    """
    Given a datetime and a time zone, determine whether or not a given datetime
    is ambiguous (i.e if there are two times differentiated only by their DST
    status).

    :param dt:
        A :class:`datetime.datetime` (whose time zone will be ignored if ``tz``
        is provided.)

    :param tz:
        A :class:`datetime.tzinfo` with support for the ``fold`` attribute. If
        ``None`` or not provided, the datetime's own time zone will be used.

    :return:
        Returns a boolean value whether or not the "wall time" is ambiguous in
        ``tz``.

    .. versionadded:: 2.6.0
    """
    if tz is None:
        if dt.tzinfo is None:
            raise ValueError('Datetime is naive and no time zone provided.')

        tz = dt.tzinfo

    # If a time zone defines its own "is_ambiguous" function, we'll use that.
    is_ambiguous_fn = getattr(tz, 'is_ambiguous', None)
    if is_ambiguous_fn is not None:
        try:
            return tz.is_ambiguous(dt)
        except Exception:
            pass

    # If it doesn't come out and tell us it's ambiguous, we'll just check if
    # the fold attribute has any effect on this particular date and time.
    dt = dt.replace(tzinfo=tz)
    wall_0 = enfold(dt, fold=0)
    wall_1 = enfold(dt, fold=1)

    same_offset = wall_0.utcoffset() == wall_1.utcoffset()
    same_dst = wall_0.dst() == wall_1.dst()

    return not (same_offset and same_dst)


def resolve_imaginary(dt):
    """
    Given a datetime that may be imaginary, return an existing datetime.

    This function assumes that an imaginary datetime represents what the
    wall time would be in a zone had the offset transition not occurred, so
    it will always fall forward by the transition's change in offset.

    .. doctest::

        >>> from dateutil import tz
        >>> from datetime import datetime
        >>> NYC = tz.gettz('America/New_York')
        >>> print(tz.resolve_imaginary(datetime(2017, 3, 12, 2, 30, tzinfo=NYC)))
        2017-03-12 03:30:00-04:00

        >>> KIR = tz.gettz('Pacific/Kiritimati')
        >>> print(tz.resolve_imaginary(datetime(1995, 1, 1, 12, 30, tzinfo=KIR)))
        1995-01-02 12:30:00+14:00

    As a note, :func:`datetime.astimezone` is guaranteed to produce a valid,
    existing datetime, so a round-trip to and from UTC is sufficient to get
    an extant datetime, however, this generally "falls back" to an earlier time
    rather than falling forward to the STD side (though no guarantees are made
    about this behavior).

    :param dt:
        A :class:`datetime.datetime` which may or may not exist.

    :return:
        Returns an existing :class:`datetime.datetime`. If ``dt`` was not
        imaginary, the datetime returned is guaranteed to be the same object
        passed to the function.

    .. versionadded:: 2.7.0
    """
    if dt.tzinfo is not None and not datetime_exists(dt):

        curr_offset = (dt + datetime.timedelta(hours=24)).utcoffset()
        old_offset = (dt - datetime.timedelta(hours=24)).utcoffset()

        dt += curr_offset - old_offset

    return dt


def _datetime_to_timestamp(dt):
    """
    Convert a :class:`datetime.datetime` object to an epoch timestamp in
    seconds since January 1, 1970, ignoring the time zone.
    """
    return (dt.replace(tzinfo=None) - EPOCH).total_seconds()


if sys.version_info >= (3, 6):
    def _get_supported_offset(second_offset):
        return second_offset
else:
    def _get_supported_offset(second_offset):
        # For python pre-3.6, round to full-minutes if that's not the case.
        # Python's datetime doesn't accept sub-minute timezones. Check
        # http://python.org/sf/1447945 or https://bugs.python.org/issue5288
        # for some information.
        old_offset = second_offset
        calculated_offset = 60 * ((second_offset + 30) // 60)
        return calculated_offset


try:
    # Python 3.7 feature
    from contextlib import nullcontext as _nullcontext
except ImportError:
    class _nullcontext(object):
        """
        Class for wrapping contexts so that they are passed through in a
        with statement.
        """
        def __init__(self, context):
            self.context = context

        def __enter__(self):
            return self.context

        def __exit__(*args, **kwargs):
            pass

# vim:ts=4:sw=4:et

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\power.py ===
from __future__ import annotations
from typing import Callable, TYPE_CHECKING
from itertools import product

from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (expand_complex, expand_multinomial,
    expand_mul, _mexpand, PoleError)
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and, fuzzy_or
from .parameters import global_parameters
from .relational import is_gt, is_lt
from .kind import NumberKind, UndefinedKind
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int
from sympy.multipledispatch import Dispatcher


class Pow(Expr):
    """
    Defines the expression x**y as "x raised to a power y"

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Singleton definitions involving (0, 1, -1, oo, -oo, I, -I):

    +--------------+---------+-----------------------------------------------+
    | expr         | value   | reason                                        |
    +==============+=========+===============================================+
    | z**0         | 1       | Although arguments over 0**0 exist, see [2].  |
    +--------------+---------+-----------------------------------------------+
    | z**1         | z       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**(-1)  | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-1)**-1     | -1      |                                               |
    +--------------+---------+-----------------------------------------------+
    | S.Zero**-1   | zoo     | This is not strictly true, as 0**-1 may be    |
    |              |         | undefined, but is convenient in some contexts |
    |              |         | where the base is assumed to be positive.     |
    +--------------+---------+-----------------------------------------------+
    | 1**-1        | 1       |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-1       | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | 0**oo        | 0       | Because for all complex numbers z near        |
    |              |         | 0, z**oo -> 0.                                |
    +--------------+---------+-----------------------------------------------+
    | 0**-oo       | zoo     | This is not strictly true, as 0**oo may be    |
    |              |         | oscillating between positive and negative     |
    |              |         | values or rotating in the complex plane.      |
    |              |         | It is convenient, however, when the base      |
    |              |         | is positive.                                  |
    +--------------+---------+-----------------------------------------------+
    | 1**oo        | nan     | Because there are various cases where         |
    | 1**-oo       |         | lim(x(t),t)=1, lim(y(t),t)=oo (or -oo),       |
    |              |         | but lim( x(t)**y(t), t) != 1.  See [3].       |
    +--------------+---------+-----------------------------------------------+
    | b**zoo       | nan     | Because b**z has no limit as z -> zoo         |
    +--------------+---------+-----------------------------------------------+
    | (-1)**oo     | nan     | Because of oscillations in the limit.         |
    | (-1)**(-oo)  |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**oo       | oo      |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-oo      | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**oo    | nan     |                                               |
    | (-oo)**-oo   |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**I        | nan     | oo**e could probably be best thought of as    |
    | (-oo)**I     |         | the limit of x**e for real x as x tends to    |
    |              |         | oo. If e is I, then the limit does not exist  |
    |              |         | and nan is used to indicate that.             |
    +--------------+---------+-----------------------------------------------+
    | oo**(1+I)    | zoo     | If the real part of e is positive, then the   |
    | (-oo)**(1+I) |         | limit of abs(x**e) is oo. So the limit value  |
    |              |         | is zoo.                                       |
    +--------------+---------+-----------------------------------------------+
    | oo**(-1+I)   | 0       | If the real part of e is negative, then the   |
    | -oo**(-1+I)  |         | limit is 0.                                   |
    +--------------+---------+-----------------------------------------------+

    Because symbolic computations are more flexible than floating point
    calculations and we prefer to never return an incorrect answer,
    we choose not to conform to all IEEE 754 conventions.  This helps
    us avoid extra test-case code in the calculation of limits.

    See Also
    ========

    sympy.core.numbers.Infinity
    sympy.core.numbers.NegativeInfinity
    sympy.core.numbers.NaN

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation
    .. [2] https://en.wikipedia.org/wiki/Zero_to_the_power_of_zero
    .. [3] https://en.wikipedia.org/wiki/Indeterminate_forms

    """
    is_Pow = True

    __slots__ = ('is_commutative',)

    if TYPE_CHECKING:

        @property
        def args(self) -> tuple[Expr, Expr]:
            ...

    @property
    def base(self) -> Expr:
        return self.args[0]

    @property
    def exp(self) -> Expr:
        return self.args[1]

    @property
    def kind(self):
        if self.exp.kind is NumberKind:
            return self.base.kind
        else:
            return UndefinedKind

    @cacheit
    def __new__(cls, b: Expr | complex, e: Expr | complex, evaluate=None) -> Expr: # type: ignore
        if evaluate is None:
            evaluate = global_parameters.evaluate

        base = _sympify(b)
        exp = _sympify(e)

        # XXX: This can be removed when non-Expr args are disallowed rather
        # than deprecated.
        from .relational import Relational
        if isinstance(base, Relational) or isinstance(exp, Relational):
            raise TypeError('Relational cannot be used in Pow')

        # XXX: This should raise TypeError once deprecation period is over:
        for arg in [base, exp]:
            if not isinstance(arg, Expr):
                sympy_deprecation_warning(
                    f"""
    Using non-Expr arguments in Pow is deprecated (in this case, one of the
    arguments is of type {type(arg).__name__!r}).

    If you really did intend to construct a power with this base, use the **
    operator instead.""",
                    deprecated_since_version="1.7",
                    active_deprecations_target="non-expr-args-deprecated",
                    stacklevel=4,
                )

        if evaluate:
            if exp is S.ComplexInfinity:
                return S.NaN
            if exp is S.Infinity:
                if is_gt(base, S.One):
                    return S.Infinity
                if is_gt(base, S.NegativeOne) and is_lt(base, S.One):
                    return S.Zero
                if is_lt(base, S.NegativeOne):
                    if base.is_finite:
                        return S.ComplexInfinity
                    if base.is_finite is False:
                        return S.NaN
            if exp is S.Zero:
                return S.One
            elif exp is S.One:
                return base
            elif exp == -1 and not base:
                return S.ComplexInfinity
            elif exp.__class__.__name__ == "AccumulationBounds":
                if base == S.Exp1:
                    from sympy.calculus.accumulationbounds import AccumBounds
                    return AccumBounds(Pow(base, exp.min), Pow(base, exp.max))
            # autosimplification if base is a number and exp odd/even
            # if base is Number then the base will end up positive; we
            # do not do this with arbitrary expressions since symbolic
            # cancellation might occur as in (x - 1)/(1 - x) -> -1. If
            # we returned Piecewise((-1, Ne(x, 1))) for such cases then
            # we could do this...but we don't
            elif (exp.is_Symbol and exp.is_integer or exp.is_Integer
                    ) and (base.is_number and base.is_Mul or base.is_Number
                    ) and base.could_extract_minus_sign():
                if exp.is_even:
                    base = -base
                elif exp.is_odd:
                    return -Pow(-base, exp)
            if S.NaN in (base, exp):  # XXX S.NaN**x -> S.NaN under assumption that x != 0
                return S.NaN
            elif base is S.One:
                if abs(exp).is_infinite:
                    return S.NaN
                return S.One
            else:
                # recognize base as E
                from sympy.functions.elementary.exponential import exp_polar
                if not exp.is_Atom and base is not S.Exp1 and not isinstance(base, exp_polar):
                    from .exprtools import factor_terms
                    from sympy.functions.elementary.exponential import log
                    from sympy.simplify.radsimp import fraction
                    c, ex = factor_terms(exp, sign=False).as_coeff_Mul()
                    num, den = fraction(ex)
                    if isinstance(den, log) and den.args[0] == base:
                        return S.Exp1**(c*num)
                    elif den.is_Add:
                        from sympy.functions.elementary.complexes import sign, im
                        s = sign(im(base))
                        if s.is_Number and s and den == \
                                log(-factor_terms(base, sign=False)) + s*S.ImaginaryUnit*S.Pi:
                            return S.Exp1**(c*num)

                obj = base._eval_power(exp)
                if obj is not None:
                    return obj
        obj = Expr.__new__(cls, base, exp)
        obj = cls._exec_constructor_postprocessors(obj)
        if not isinstance(obj, Pow):
            return obj
        obj.is_commutative = (base.is_commutative and exp.is_commutative)
        return obj

    def inverse(self, argindex=1):
        if self.base == S.Exp1:
            from sympy.functions.elementary.exponential import log
            return log
        return None

    @classmethod
    def class_key(cls):
        return 3, 2, cls.__name__

    def _eval_refine(self, assumptions):
        from sympy.assumptions.ask import ask, Q
        b, e = self.as_base_exp()
        if ask(Q.integer(e), assumptions) and b.could_extract_minus_sign():
            if ask(Q.even(e), assumptions):
                return Pow(-b, e)
            elif ask(Q.odd(e), assumptions):
                return -Pow(-b, e)

    def _eval_power(self, expt):
        b, e = self.as_base_exp()
        if b is S.NaN:
            return (b**e)**expt  # let __new__ handle it

        s = None
        if expt.is_integer:
            s = 1
        elif b.is_polar:  # e.g. exp_polar, besselj, var('p', polar=True)...
            s = 1
        elif e.is_extended_real is not None:
            from sympy.functions.elementary.complexes import arg, im, re, sign
            from sympy.functions.elementary.exponential import exp, log
            from sympy.functions.elementary.integers import floor
            # helper functions ===========================
            def _half(e):
                """Return True if the exponent has a literal 2 as the
                denominator, else None."""
                if getattr(e, 'q', None) == 2:
                    return True
                n, d = e.as_numer_denom()
                if n.is_integer and d == 2:
                    return True
            def _n2(e):
                """Return ``e`` evaluated to a Number with 2 significant
                digits, else None."""
                try:
                    rv = e.evalf(2, strict=True)
                    if rv.is_Number:
                        return rv
                except PrecisionExhausted:
                    pass
            # ===================================================
            if e.is_extended_real:
                # we need _half(expt) with constant floor or
                # floor(S.Half - e*arg(b)/2/pi) == 0


                # handle -1 as special case
                if e == -1:
                    # floor arg. is 1/2 + arg(b)/2/pi
                    if _half(expt):
                        if b.is_negative is True:
                            return S.NegativeOne**expt*Pow(-b, e*expt)
                        elif b.is_negative is False:  # XXX ok if im(b) != 0?
                            return Pow(b, -expt)
                elif e.is_even:
                    if b.is_extended_real:
                        b = abs(b)
                    if b.is_imaginary:
                        b = abs(im(b))*S.ImaginaryUnit

                if (abs(e) < 1) == True or e == 1:
                    s = 1  # floor = 0
                elif b.is_extended_nonnegative:
                    s = 1  # floor = 0
                elif re(b).is_extended_nonnegative and (abs(e) < 2) == True:
                    s = 1  # floor = 0
                elif _half(expt):
                    s = exp(2*S.Pi*S.ImaginaryUnit*expt*floor(
                        S.Half - e*arg(b)/(2*S.Pi)))
                    if s.is_extended_real and _n2(sign(s) - s) == 0:
                        s = sign(s)
                    else:
                        s = None
            else:
                # e.is_extended_real is False requires:
                #     _half(expt) with constant floor or
                #     floor(S.Half - im(e*log(b))/2/pi) == 0
                try:
                    s = exp(2*S.ImaginaryUnit*S.Pi*expt*
                        floor(S.Half - im(e*log(b))/2/S.Pi))
                    # be careful to test that s is -1 or 1 b/c sign(I) == I:
                    # so check that s is real
                    if s.is_extended_real and _n2(sign(s) - s) == 0:
                        s = sign(s)
                    else:
                        s = None
                except PrecisionExhausted:
                    s = None

        if s is not None:
            return s*Pow(b, e*expt)

    def _eval_Mod(self, q):
        r"""A dispatched function to compute `b^e \bmod q`, dispatched
        by ``Mod``.

        Notes
        =====

        Algorithms:

        1. For unevaluated integer power, use built-in ``pow`` function
        with 3 arguments, if powers are not too large wrt base.

        2. For very large powers, use totient reduction if $e \ge \log(m)$.
        Bound on m, is for safe factorization memory wise i.e. $m^{1/4}$.
        For pollard-rho to be faster than built-in pow $\log(e) > m^{1/4}$
        check is added.

        3. For any unevaluated power found in `b` or `e`, the step 2
        will be recursed down to the base and the exponent
        such that the $b \bmod q$ becomes the new base and
        $\phi(q) + e \bmod \phi(q)$ becomes the new exponent, and then
        the computation for the reduced expression can be done.
        """

        base, exp = self.base, self.exp

        if exp.is_integer and exp.is_positive:
            if q.is_integer and base % q == 0:
                return S.Zero

            from sympy.functions.combinatorial.numbers import totient

            if base.is_Integer and exp.is_Integer and q.is_Integer:
                b, e, m = int(base), int(exp), int(q)
                mb = m.bit_length()
                if mb <= 80 and e >= mb and e.bit_length()**4 >= m:
                    phi = int(totient(m))
                    return Integer(pow(b, phi + e%phi, m))
                return Integer(pow(b, e, m))

            from .mod import Mod

            if isinstance(base, Pow) and base.is_integer and base.is_number:
                base = Mod(base, q)
                return Mod(Pow(base, exp, evaluate=False), q)

            if isinstance(exp, Pow) and exp.is_integer and exp.is_number:
                bit_length = int(q).bit_length()
                # XXX Mod-Pow actually attempts to do a hanging evaluation
                # if this dispatched function returns None.
                # May need some fixes in the dispatcher itself.
                if bit_length <= 80:
                    phi = totient(q)
                    exp = phi + Mod(exp, phi)
                    return Mod(Pow(base, exp, evaluate=False), q)

    def _eval_is_even(self):
        if self.exp.is_integer and self.exp.is_positive:
            return self.base.is_even

    def _eval_is_negative(self):
        ext_neg = Pow._eval_is_extended_negative(self)
        if ext_neg is True:
            return self.is_finite
        return ext_neg

    def _eval_is_extended_positive(self):
        if self.base == self.exp:
            if self.base.is_extended_nonnegative:
                return True
        elif self.base.is_positive:
            if self.exp.is_real:
                return True
        elif self.base.is_extended_negative:
            if self.exp.is_even:
                return True
            if self.exp.is_odd:
                return False
        elif self.base.is_zero:
            if self.exp.is_extended_real:
                return self.exp.is_zero
        elif self.base.is_extended_nonpositive:
            if self.exp.is_odd:
                return False
        elif self.base.is_imaginary:
            if self.exp.is_integer:
                m = self.exp % 4
                if m.is_zero:
                    return True
                if m.is_integer and m.is_zero is False:
                    return False
            if self.exp.is_imaginary:
                from sympy.functions.elementary.exponential import log
                return log(self.base).is_imaginary

    def _eval_is_extended_negative(self):
        if self.exp is S.Half:
            if self.base.is_complex or self.base.is_extended_real:
                return False
        if self.base.is_extended_negative:
            if self.exp.is_odd and self.base.is_finite:
                return True
            if self.exp.is_even:
                return False
        elif self.base.is_extended_positive:
            if self.exp.is_extended_real:
                return False
        elif self.base.is_zero:
            if self.exp.is_extended_real:
                return False
        elif self.base.is_extended_nonnegative:
            if self.exp.is_extended_nonnegative:
                return False
        elif self.base.is_extended_nonpositive:
            if self.exp.is_even:
                return False
        elif self.base.is_extended_real:
            if self.exp.is_even:
                return False

    def _eval_is_zero(self):
        if self.base.is_zero:
            if self.exp.is_extended_positive:
                return True
            elif self.exp.is_extended_nonpositive:
                return False
        elif self.base == S.Exp1:
            return self.exp is S.NegativeInfinity
        elif self.base.is_zero is False:
            if self.base.is_finite and self.exp.is_finite:
                return False
            elif self.exp.is_negative:
                return self.base.is_infinite
            elif self.exp.is_nonnegative:
                return False
            elif self.exp.is_infinite and self.exp.is_extended_real:
                if (1 - abs(self.base)).is_extended_positive:
                    return self.exp.is_extended_positive
                elif (1 - abs(self.base)).is_extended_negative:
                    return self.exp.is_extended_negative
        elif self.base.is_finite and self.exp.is_negative:
            # when self.base.is_zero is None
            return False

    def _eval_is_integer(self):
        b, e = self.args
        if b.is_rational:
            if b.is_integer is False and e.is_positive:
                return False  # rat**nonneg
        if b.is_integer and e.is_integer:
            if b is S.NegativeOne:
                return True
            if e.is_nonnegative or e.is_positive:
                return True
        if b.is_integer and e.is_negative and (e.is_finite or e.is_integer):
            if fuzzy_not((b - 1).is_zero) and fuzzy_not((b + 1).is_zero):
                return False
        if b.is_Number and e.is_Number:
            check = self.func(*self.args)
            return check.is_Integer
        if e.is_negative and b.is_positive and (b - 1).is_positive:
            return False
        if e.is_negative and b.is_negative and (b + 1).is_negative:
            return False

    def _eval_is_extended_real(self):
        if self.base is S.Exp1:
            if self.exp.is_extended_real:
                return True
            elif self.exp.is_imaginary:
                return (2*S.ImaginaryUnit*self.exp/S.Pi).is_even

        from sympy.functions.elementary.exponential import log, exp
        real_b = self.base.is_extended_real
        if real_b is None:
            if self.base.func == exp and self.base.exp.is_imaginary:
                return self.exp.is_imaginary
            if self.base.func == Pow and self.base.base is S.Exp1 and self.base.exp.is_imaginary:
                return self.exp.is_imaginary
            return
        real_e = self.exp.is_extended_real
        if real_e is None:
            return
        if real_b and real_e:
            if self.base.is_extended_positive:
                return True
            elif self.base.is_extended_nonnegative and self.exp.is_extended_nonnegative:
                return True
            elif self.exp.is_integer and self.base.is_extended_nonzero:
                return True
            elif self.exp.is_integer and self.exp.is_nonnegative:
                return True
            elif self.base.is_extended_negative:
                if self.exp.is_Rational:
                    return False
        if real_e and self.exp.is_extended_negative and self.base.is_zero is False:
            return Pow(self.base, -self.exp).is_extended_real
        im_b = self.base.is_imaginary
        im_e = self.exp.is_imaginary
        if im_b:
            if self.exp.is_integer:
                if self.exp.is_even:
                    return True
                elif self.exp.is_odd:
                    return False
            elif im_e and log(self.base).is_imaginary:
                return True
            elif self.exp.is_Add:
                c, a = self.exp.as_coeff_Add()
                if c and c.is_Integer:
                    return Mul(
                        self.base**c, self.base**a, evaluate=False).is_extended_real
            elif self.base in (-S.ImaginaryUnit, S.ImaginaryUnit):
                if (self.exp/2).is_integer is False:
                    return False
        if real_b and im_e:
            if self.base is S.NegativeOne:
                return True
            c = self.exp.coeff(S.ImaginaryUnit)
            if c:
                if self.base.is_rational and c.is_rational:
                    if self.base.is_nonzero and (self.base - 1).is_nonzero and c.is_nonzero:
                        return False
                ok = (c*log(self.base)/S.Pi).is_integer
                if ok is not None:
                    return ok

        if real_b is False and real_e: # we already know it's not imag
            if isinstance(self.exp, Rational) and self.exp.p == 1:
                return False
            from sympy.functions.elementary.complexes import arg
            i = arg(self.base)*self.exp/S.Pi
            if i.is_complex: # finite
                return i.is_integer

    def _eval_is_complex(self):

        if self.base == S.Exp1:
            return fuzzy_or([self.exp.is_complex, self.exp.is_extended_negative])

        if all(a.is_complex for a in self.args) and self._eval_is_finite():
            return True

    def _eval_is_imaginary(self):
        if self.base.is_commutative is False:
            return False

        if self.base.is_imaginary:
            if self.exp.is_integer:
                odd = self.exp.is_odd
                if odd is not None:
                    return odd
                return

        if self.base == S.Exp1:
            f = 2 * self.exp / (S.Pi*S.ImaginaryUnit)
            # exp(pi*integer) = 1 or -1, so not imaginary
            if f.is_even:
                return False
            # exp(pi*integer + pi/2) = I or -I, so it is imaginary
            if f.is_odd:
                return True
            return None

        if self.exp.is_imaginary:
            from sympy.functions.elementary.exponential import log
            imlog = log(self.base).is_imaginary
            if imlog is not None:
                return False  # I**i -> real; (2*I)**i -> complex ==> not imaginary

        if self.base.is_extended_real and self.exp.is_extended_real:
            if self.base.is_positive:
                return False
            else:
                rat = self.exp.is_rational
                if not rat:
                    return rat
                if self.exp.is_integer:
                    return False
                else:
                    half = (2*self.exp).is_integer
                    if half:
                        return self.base.is_negative
                    return half

        if self.base.is_extended_real is False:  # we already know it's not imag
            from sympy.functions.elementary.complexes import arg
            i = arg(self.base)*self.exp/S.Pi
            isodd = (2*i).is_odd
            if isodd is not None:
                return isodd

    def _eval_is_odd(self):
        if self.exp.is_integer:
            if self.exp.is_positive:
                return self.base.is_odd
            elif self.exp.is_nonnegative and self.base.is_odd:
                return True
            elif self.base is S.NegativeOne:
                return True

    def _eval_is_finite(self):
        if self.exp.is_negative:
            if self.base.is_zero:
                return False
            if self.base.is_infinite or self.base.is_nonzero:
                return True
        c1 = self.base.is_finite
        if c1 is None:
            return
        c2 = self.exp.is_finite
        if c2 is None:
            return
        if c1 and c2:
            if self.exp.is_nonnegative or fuzzy_not(self.base.is_zero):
                return True

    def _eval_is_prime(self):
        '''
        An integer raised to the n(>=2)-th power cannot be a prime.
        '''
        if self.base.is_integer and self.exp.is_integer and (self.exp - 1).is_positive:
            return False

    def _eval_is_composite(self):
        """
        A power is composite if both base and exponent are greater than 1
        """
        if (self.base.is_integer and self.exp.is_integer and
            ((self.base - 1).is_positive and (self.exp - 1).is_positive or
            (self.base + 1).is_negative and self.exp.is_positive and self.exp.is_even)):
            return True

    def _eval_is_polar(self):
        return self.base.is_polar

    def _eval_subs(self, old, new):
        from sympy.calculus.accumulationbounds import AccumBounds

        if isinstance(self.exp, AccumBounds):
            b = self.base.subs(old, new)
            e = self.exp.subs(old, new)
            if isinstance(e, AccumBounds):
                return e.__rpow__(b)
            return self.func(b, e)

        from sympy.functions.elementary.exponential import exp, log

        def _check(ct1, ct2, old):
            """Return (bool, pow, remainder_pow) where, if bool is True, then the
            exponent of Pow `old` will combine with `pow` so the substitution
            is valid, otherwise bool will be False.

            For noncommutative objects, `pow` will be an integer, and a factor
            `Pow(old.base, remainder_pow)` needs to be included. If there is
            no such factor, None is returned. For commutative objects,
            remainder_pow is always None.

            cti are the coefficient and terms of an exponent of self or old
            In this _eval_subs routine a change like (b**(2*x)).subs(b**x, y)
            will give y**2 since (b**x)**2 == b**(2*x); if that equality does
            not hold then the substitution should not occur so `bool` will be
            False.

            """
            coeff1, terms1 = ct1
            coeff2, terms2 = ct2
            if terms1 == terms2:
                if old.is_commutative:
                    # Allow fractional powers for commutative objects
                    pow = coeff1/coeff2
                    try:
                        as_int(pow, strict=False)
                        combines = True
                    except ValueError:
                        b, e = old.as_base_exp()
                        # These conditions ensure that (b**e)**f == b**(e*f) for any f
                        combines = b.is_positive and e.is_real or b.is_nonnegative and e.is_nonnegative

                    return combines, pow, None
                else:
                    # With noncommutative symbols, substitute only integer powers
                    if not isinstance(terms1, tuple):
                        terms1 = (terms1,)
                    if not all(term.is_integer for term in terms1):
                        return False, None, None

                    try:
                        # Round pow toward zero
                        pow, remainder = divmod(as_int(coeff1), as_int(coeff2))
                        if pow < 0 and remainder != 0:
                            pow += 1
                            remainder -= as_int(coeff2)

                        if remainder == 0:
                            remainder_pow = None
                        else:
                            remainder_pow = Mul(remainder, *terms1)

                        return True, pow, remainder_pow
                    except ValueError:
                        # Can't substitute
                        pass

            return False, None, None

        if old == self.base or (old == exp and self.base == S.Exp1):
            if new.is_Function and isinstance(new, Callable):
                return new(self.exp._subs(old, new))
            else:
                return new**self.exp._subs(old, new)

        # issue 10829: (4**x - 3*y + 2).subs(2**x, y) -> y**2 - 3*y + 2
        if isinstance(old, self.func) and self.exp == old.exp:
            l = log(self.base, old.base)
            if l.is_Number:
                return Pow(new, l)

        if isinstance(old, self.func) and self.base == old.base:
            if self.exp.is_Add is False:
                ct1 = self.exp.as_independent(Symbol, as_Add=False)
                ct2 = old.exp.as_independent(Symbol, as_Add=False)
                ok, pow, remainder_pow = _check(ct1, ct2, old)
                if ok:
                    # issue 5180: (x**(6*y)).subs(x**(3*y),z)->z**2
                    result = self.func(new, pow)
                    if remainder_pow is not None:
                        result = Mul(result, Pow(old.base, remainder_pow))
                    return result
            else:  # b**(6*x + a).subs(b**(3*x), y) -> y**2 * b**a
                # exp(exp(x) + exp(x**2)).subs(exp(exp(x)), w) -> w * exp(exp(x**2))
                oarg = old.exp
                new_l = []
                o_al = []
                ct2 = oarg.as_coeff_mul()
                for a in self.exp.args:
                    newa = a._subs(old, new)
                    ct1 = newa.as_coeff_mul()
                    ok, pow, remainder_pow = _check(ct1, ct2, old)
                    if ok:
                        new_l.append(new**pow)
                        if remainder_pow is not None:
                            o_al.append(remainder_pow)
                        continue
                    elif not old.is_commutative and not newa.is_integer:
                        # If any term in the exponent is non-integer,
                        # we do not do any substitutions in the noncommutative case
                        return
                    o_al.append(newa)
                if new_l:
                    expo = Add(*o_al)
                    new_l.append(Pow(self.base, expo, evaluate=False) if expo != 1 else self.base)
                    return Mul(*new_l)

        if (isinstance(old, exp) or (old.is_Pow and old.base is S.Exp1)) and self.exp.is_extended_real and self.base.is_positive:
            ct1 = old.exp.as_independent(Symbol, as_Add=False)
            ct2 = (self.exp*log(self.base)).as_independent(
                Symbol, as_Add=False)
            ok, pow, remainder_pow = _check(ct1, ct2, old)
            if ok:
                result = self.func(new, pow)  # (2**x).subs(exp(x*log(2)), z) -> z
                if remainder_pow is not None:
                    result = Mul(result, Pow(old.base, remainder_pow))
                return result

    def as_base_exp(self):
        """Return base and exp of self.

        Explanation
        ===========

        If base a Rational less than 1, then return 1/Rational, -exp.
        If this extra processing is not needed, the base and exp
        properties will give the raw arguments.

        Examples
        ========

        >>> from sympy import Pow, S
        >>> p = Pow(S.Half, 2, evaluate=False)
        >>> p.as_base_exp()
        (2, -2)
        >>> p.args
        (1/2, 2)
        >>> p.base, p.exp
        (1/2, 2)

        """
        b, e = self.args
        if b.is_Rational and b.p == 1 and b.q != 1:
            return Integer(b.q), -e
        return b, e

    def _eval_adjoint(self):
        from sympy.functions.elementary.complexes import adjoint
        i, p = self.exp.is_integer, self.base.is_positive
        if i:
            return adjoint(self.base)**self.exp
        if p:
            return self.base**adjoint(self.exp)
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return adjoint(expanded)

    def _eval_conjugate(self):
        from sympy.functions.elementary.complexes import conjugate as c
        i, p = self.exp.is_integer, self.base.is_positive
        if i:
            return c(self.base)**self.exp
        if p:
            return self.base**c(self.exp)
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return c(expanded)
        if self.is_extended_real:
            return self

    def _eval_transpose(self):
        from sympy.functions.elementary.complexes import transpose
        if self.base == S.Exp1:
            return self.func(S.Exp1, self.exp.transpose())
        i, p = self.exp.is_integer, (self.base.is_complex or self.base.is_infinite)
        if p:
            return self.base**self.exp
        if i:
            return transpose(self.base)**self.exp
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return transpose(expanded)

    def _eval_expand_power_exp(self, **hints):
        """a**(n + m) -> a**n*a**m"""
        b = self.base
        e = self.exp
        if b == S.Exp1:
            from sympy.concrete.summations import Sum
            if isinstance(e, Sum) and e.is_commutative:
                from sympy.concrete.products import Product
                return Product(self.func(b, e.function), *e.limits)
        if e.is_Add and (hints.get('force', False) or
                b.is_zero is False or e._all_nonneg_or_nonppos()):
            if e.is_commutative:
                return Mul(*[self.func(b, x) for x in e.args])
            if b.is_commutative:
                c, nc = sift(e.args, lambda x: x.is_commutative, binary=True)
                if c:
                    return Mul(*[self.func(b, x) for x in c]
                        )*b**Add._from_args(nc)
        return self

    def _eval_expand_power_base(self, **hints):
        """(a*b)**n -> a**n * b**n"""
        force = hints.get('force', False)

        b = self.base
        e = self.exp
        if not b.is_Mul:
            return self

        cargs, nc = b.args_cnc(split_1=False)

        # expand each term - this is top-level-only
        # expansion but we have to watch out for things
        # that don't have an _eval_expand method
        if nc:
            nc = [i._eval_expand_power_base(**hints)
                if hasattr(i, '_eval_expand_power_base') else i
                for i in nc]

            if e.is_Integer:
                if e.is_positive:
                    rv = Mul(*nc*e)
                else:
                    rv = Mul(*[i**-1 for i in nc[::-1]]*-e)
                if cargs:
                    rv *= Mul(*cargs)**e
                return rv

            if not cargs:
                return self.func(Mul(*nc), e, evaluate=False)

            nc = [Mul(*nc)]

        # sift the commutative bases
        other, maybe_real = sift(cargs, lambda x: x.is_extended_real is False,
            binary=True)
        def pred(x):
            if x is S.ImaginaryUnit:
                return S.ImaginaryUnit
            polar = x.is_polar
            if polar:
                return True
            if polar is None:
                return fuzzy_bool(x.is_extended_nonnegative)
        sifted = sift(maybe_real, pred)
        nonneg = sifted[True]
        other += sifted[None]
        neg = sifted[False]
        imag = sifted[S.ImaginaryUnit]
        if imag:
            I = S.ImaginaryUnit
            i = len(imag) % 4
            if i == 0:
                pass
            elif i == 1:
                other.append(I)
            elif i == 2:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
            else:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
                other.append(I)
            del imag

        # bring out the bases that can be separated from the base

        if force or e.is_integer:
            # treat all commutatives the same and put nc in other
            cargs = nonneg + neg + other
            other = nc
        else:
            # this is just like what is happening automatically, except
            # that now we are doing it for an arbitrary exponent for which
            # no automatic expansion is done

            assert not e.is_Integer

            # handle negatives by making them all positive and putting
            # the residual -1 in other
            if len(neg) > 1:
                o = S.One
                if not other and neg[0].is_Number:
                    o *= neg.pop(0)
                if len(neg) % 2:
                    o = -o
                for n in neg:
                    nonneg.append(-n)
                if o is not S.One:
                    other.append(o)
            elif neg and other:
                if neg[0].is_Number and neg[0] is not S.NegativeOne:
                    other.append(S.NegativeOne)
                    nonneg.append(-neg[0])
                else:
                    other.extend(neg)
            else:
                other.extend(neg)
            del neg

            cargs = nonneg
            other += nc

        rv = S.One
        if cargs:
            if e.is_Rational:
                npow, cargs = sift(cargs, lambda x: x.is_Pow and
                    x.exp.is_Rational and x.base.is_number,
                    binary=True)
                rv = Mul(*[self.func(b.func(*b.args), e) for b in npow])
            rv *= Mul(*[self.func(b, e, evaluate=False) for b in cargs])
        if other:
            rv *= self.func(Mul(*other), e, evaluate=False)
        return rv

    def _eval_expand_multinomial(self, **hints):
        """(a + b + ..)**n -> a**n + n*a**(n-1)*b + .., n is nonzero integer"""

        base, exp = self.args
        result = self

        if exp.is_Rational and exp.p > 0 and base.is_Add:
            if not exp.is_Integer:
                n = Integer(exp.p // exp.q)

                if not n:
                    return result
                else:
                    radical, result = self.func(base, exp - n), []

                    expanded_base_n = self.func(base, n)
                    if expanded_base_n.is_Pow:
                        expanded_base_n = \
                            expanded_base_n._eval_expand_multinomial()
                    for term in Add.make_args(expanded_base_n):
                        result.append(term*radical)

                    return Add(*result)

            n = int(exp)

            if base.is_commutative:
                order_terms, other_terms = [], []

                for b in base.args:
                    if b.is_Order:
                        order_terms.append(b)
                    else:
                        other_terms.append(b)

                if order_terms:
                    # (f(x) + O(x^n))^m -> f(x)^m + m*f(x)^{m-1} *O(x^n)
                    f = Add(*other_terms)
                    o = Add(*order_terms)

                    if n == 2:
                        return expand_multinomial(f**n, deep=False) + n*f*o
                    else:
                        g = expand_multinomial(f**(n - 1), deep=False)
                        return expand_mul(f*g, deep=False) + n*g*o

                if base.is_number:
                    # Efficiently expand expressions of the form (a + b*I)**n
                    # where 'a' and 'b' are real numbers and 'n' is integer.
                    a, b = base.as_real_imag()

                    if a.is_Rational and b.is_Rational:
                        if not a.is_Integer:
                            if not b.is_Integer:
                                k = self.func(a.q * b.q, n)
                                a, b = a.p*b.q, a.q*b.p
                            else:
                                k = self.func(a.q, n)
                                a, b = a.p, a.q*b
                        elif not b.is_Integer:
                            k = self.func(b.q, n)
                            a, b = a*b.q, b.p
                        else:
                            k = 1

                        a, b, c, d = int(a), int(b), 1, 0

                        while n:
                            if n & 1:
                                c, d = a*c - b*d, b*c + a*d
                                n -= 1
                            a, b = a*a - b*b, 2*a*b
                            n //= 2

                        I = S.ImaginaryUnit

                        if k == 1:
                            return c + I*d
                        else:
                            return Integer(c)/k + I*d/k

                p = other_terms
                # (x + y)**3 -> x**3 + 3*x**2*y + 3*x*y**2 + y**3
                # in this particular example:
                # p = [x,y]; n = 3
                # so now it's easy to get the correct result -- we get the
                # coefficients first:
                from sympy.ntheory.multinomial import multinomial_coefficients
                from sympy.polys.polyutils import basic_from_dict
                expansion_dict = multinomial_coefficients(len(p), n)
                # in our example: {(3, 0): 1, (1, 2): 3, (0, 3): 1, (2, 1): 3}
                # and now construct the expression.
                return basic_from_dict(expansion_dict, *p)
            else:
                if n == 2:
                    return Add(*[f*g for f in base.args for g in base.args])
                else:
                    multi = (base**(n - 1))._eval_expand_multinomial()
                    if multi.is_Add:
                        return Add(*[f*g for f in base.args
                            for g in multi.args])
                    else:
                        # XXX can this ever happen if base was an Add?
                        return Add(*[f*multi for f in base.args])
        elif (exp.is_Rational and exp.p < 0 and base.is_Add and
                abs(exp.p) > exp.q):
            return 1 / self.func(base, -exp)._eval_expand_multinomial()
        elif exp.is_Add and base.is_Number and (hints.get('force', False) or
                base.is_zero is False or exp._all_nonneg_or_nonppos()):
            #  a + b      a  b
            #  n      --> n  n, where n, a, b are Numbers
            # XXX should be in expand_power_exp?
            coeff, tail = [], []
            for term in exp.args:
                if term.is_Number:
                    coeff.append(self.func(base, term))
                else:
                    tail.append(term)
            return Mul(*(coeff + [self.func(base, Add._from_args(tail))]))
        else:
            return result

    def as_real_imag(self, deep=True, **hints):
        if self.exp.is_Integer:
            from sympy.polys.polytools import poly

            exp = self.exp
            re_e, im_e = self.base.as_real_imag(deep=deep)
            if not im_e:
                return self, S.Zero
            a, b = symbols('a b', cls=Dummy)
            if exp >= 0:
                if re_e.is_Number and im_e.is_Number:
                    # We can be more efficient in this case
                    expr = expand_multinomial(self.base**exp)
                    if expr != self:
                        return expr.as_real_imag()

                expr = poly(
                    (a + b)**exp)  # a = re, b = im; expr = (a + b*I)**exp
            else:
                mag = re_e**2 + im_e**2
                re_e, im_e = re_e/mag, -im_e/mag
                if re_e.is_Number and im_e.is_Number:
                    # We can be more efficient in this case
                    expr = expand_multinomial((re_e + im_e*S.ImaginaryUnit)**-exp)
                    if expr != self:
                        return expr.as_real_imag()

                expr = poly((a + b)**-exp)

            # Terms with even b powers will be real
            r = [i for i in expr.terms() if not i[0][1] % 2]
            re_part = Add(*[cc*a**aa*b**bb for (aa, bb), cc in r])
            # Terms with odd b powers will be imaginary
            r = [i for i in expr.terms() if i[0][1] % 4 == 1]
            im_part1 = Add(*[cc*a**aa*b**bb for (aa, bb), cc in r])
            r = [i for i in expr.terms() if i[0][1] % 4 == 3]
            im_part3 = Add(*[cc*a**aa*b**bb for (aa, bb), cc in r])

            return (re_part.subs({a: re_e, b: S.ImaginaryUnit*im_e}),
            im_part1.subs({a: re_e, b: im_e}) + im_part3.subs({a: re_e, b: -im_e}))

        from sympy.functions.elementary.trigonometric import atan2, cos, sin

        if self.exp.is_Rational:
            re_e, im_e = self.base.as_real_imag(deep=deep)

            if im_e.is_zero and self.exp is S.Half:
                if re_e.is_extended_nonnegative:
                    return self, S.Zero
                if re_e.is_extended_nonpositive:
                    return S.Zero, (-self.base)**self.exp

            # XXX: This is not totally correct since for x**(p/q) with
            #      x being imaginary there are actually q roots, but
            #      only a single one is returned from here.
            r = self.func(self.func(re_e, 2) + self.func(im_e, 2), S.Half)

            t = atan2(im_e, re_e)

            rp, tp = self.func(r, self.exp), t*self.exp

            return rp*cos(tp), rp*sin(tp)
        elif self.base is S.Exp1:
            from sympy.functions.elementary.exponential import exp
            re_e, im_e = self.exp.as_real_imag()
            if deep:
                re_e = re_e.expand(deep, **hints)
                im_e = im_e.expand(deep, **hints)
            c, s = cos(im_e), sin(im_e)
            return exp(re_e)*c, exp(re_e)*s
        else:
            from sympy.functions.elementary.complexes import im, re
            if deep:
                hints['complex'] = False

                expanded = self.expand(deep, **hints)
                if hints.get('ignore') == expanded:
                    return None
                else:
                    return (re(expanded), im(expanded))
            else:
                return re(self), im(self)

    def _eval_derivative(self, s):
        from sympy.functions.elementary.exponential import log
        dbase = self.base.diff(s)
        dexp = self.exp.diff(s)
        return self * (dexp * log(self.base) + dbase * self.exp/self.base)

    def _eval_evalf(self, prec):
        base, exp = self.as_base_exp()
        if base == S.Exp1:
            # Use mpmath function associated to class "exp":
            from sympy.functions.elementary.exponential import exp as exp_function
            return exp_function(self.exp, evaluate=False)._eval_evalf(prec)
        base = base._evalf(prec)
        if not exp.is_Integer:
            exp = exp._evalf(prec)
        if exp.is_negative and base.is_number and base.is_extended_real is False:
            base = base.conjugate() / (base * base.conjugate())._evalf(prec)
            exp = -exp
            return self.func(base, exp).expand()
        return self.func(base, exp)

    def _eval_is_polynomial(self, syms):
        if self.exp.has(*syms):
            return False

        if self.base.has(*syms):
            return bool(self.base._eval_is_polynomial(syms) and
                self.exp.is_Integer and (self.exp >= 0))
        else:
            return True

    def _eval_is_rational(self):
        # The evaluation of self.func below can be very expensive in the case
        # of integer**integer if the exponent is large.  We should try to exit
        # before that if possible:
        if (self.exp.is_integer and self.base.is_rational
                and fuzzy_not(fuzzy_and([self.exp.is_negative, self.base.is_zero]))):
            return True
        p = self.func(*self.as_base_exp())  # in case it's unevaluated
        if not p.is_Pow:
            return p.is_rational
        b, e = p.as_base_exp()
        if e.is_Rational and b.is_Rational:
            # we didn't check that e is not an Integer
            # because Rational**Integer autosimplifies
            return False
        if e.is_integer:
            if b.is_rational:
                if fuzzy_not(b.is_zero) or e.is_nonnegative:
                    return True
                if b == e:  # always rational, even for 0**0
                    return True
            elif b.is_irrational:
                return e.is_zero
        if b is S.Exp1:
            if e.is_rational and e.is_nonzero:
                return False

    def _eval_is_algebraic(self):
        def _is_one(expr):
            try:
                return (expr - 1).is_zero
            except ValueError:
                # when the operation is not allowed
                return False

        if self.base.is_zero or _is_one(self.base):
            return True
        elif self.base is S.Exp1:
            s = self.func(*self.args)
            if s.func == self.func:
                if self.exp.is_nonzero:
                    if self.exp.is_algebraic:
                        return False
                    elif (self.exp/S.Pi).is_rational:
                        return False
                    elif (self.exp/(S.ImaginaryUnit*S.Pi)).is_rational:
                        return True
            else:
                return s.is_algebraic
        elif self.exp.is_rational:
            if self.base.is_algebraic is False:
                return self.exp.is_zero
            if self.base.is_zero is False:
                if self.exp.is_nonzero:
                    return self.base.is_algebraic
                elif self.base.is_algebraic:
                    return True
            if self.exp.is_positive:
                return self.base.is_algebraic
        elif self.base.is_algebraic and self.exp.is_algebraic:
            if ((fuzzy_not(self.base.is_zero)
                and fuzzy_not(_is_one(self.base)))
                or self.base.is_integer is False
                or self.base.is_irrational):
                return self.exp.is_rational

    def _eval_is_rational_function(self, syms):
        if self.exp.has(*syms):
            return False

        if self.base.has(*syms):
            return self.base._eval_is_rational_function(syms) and \
                self.exp.is_Integer
        else:
            return True

    def _eval_is_meromorphic(self, x, a):
        # f**g is meromorphic if g is an integer and f is meromorphic.
        # E**(log(f)*g) is meromorphic if log(f)*g is meromorphic
        # and finite.
        base_merom = self.base._eval_is_meromorphic(x, a)
        exp_integer = self.exp.is_Integer
        if exp_integer:
            return base_merom

        exp_merom = self.exp._eval_is_meromorphic(x, a)
        if base_merom is False:
            # f**g = E**(log(f)*g) may be meromorphic if the
            # singularities of log(f) and g cancel each other,
            # for example, if g = 1/log(f). Hence,
            return False if exp_merom else None
        elif base_merom is None:
            return None

        b = self.base.subs(x, a)
        # b is extended complex as base is meromorphic.
        # log(base) is finite and meromorphic when b != 0, zoo.
        b_zero = b.is_zero
        if b_zero:
            log_defined = False
        else:
            log_defined = fuzzy_and((b.is_finite, fuzzy_not(b_zero)))

        if log_defined is False: # zero or pole of base
            return exp_integer  # False or None
        elif log_defined is None:
            return None

        if not exp_merom:
            return exp_merom  # False or None

        return self.exp.subs(x, a).is_finite

    def _eval_is_algebraic_expr(self, syms):
        if self.exp.has(*syms):
            return False

        if self.base.has(*syms):
            return self.base._eval_is_algebraic_expr(syms) and \
                self.exp.is_Rational
        else:
            return True

    def _eval_rewrite_as_exp(self, base, expo, **kwargs):
        from sympy.functions.elementary.exponential import exp, log

        if base.is_zero or base.has(exp) or expo.has(exp):
            return base**expo

        evaluate = expo.has(Symbol)

        if base.has(Symbol):
            # delay evaluation if expo is non symbolic
            # (as exp(x*log(5)) automatically reduces to x**5)
            if global_parameters.exp_is_pow:
                return Pow(S.Exp1, log(base)*expo, evaluate=evaluate)
            else:
                return exp(log(base)*expo, evaluate=evaluate)

        else:
            from sympy.functions.elementary.complexes import arg, Abs
            return exp((log(Abs(base)) + S.ImaginaryUnit*arg(base))*expo)

    def as_numer_denom(self):
        if not self.is_commutative:
            return self, S.One
        base, exp = self.as_base_exp()
        n, d = base.as_numer_denom()
        # this should be the same as ExpBase.as_numer_denom wrt
        # exponent handling
        neg_exp = exp.is_negative
        if exp.is_Mul and not neg_exp and not exp.is_positive:
            neg_exp = exp.could_extract_minus_sign()
        int_exp = exp.is_integer
        # the denominator cannot be separated from the numerator if
        # its sign is unknown unless the exponent is an integer, e.g.
        # sqrt(a/b) != sqrt(a)/sqrt(b) when a=1 and b=-1. But if the
        # denominator is negative the numerator and denominator can
        # be negated and the denominator (now positive) separated.
        if not (d.is_extended_real or int_exp):
            n = base
            d = S.One
        dnonpos = d.is_nonpositive
        if dnonpos:
            n, d = -n, -d
        elif dnonpos is None and not int_exp:
            n = base
            d = S.One
        if neg_exp:
            n, d = d, n
            exp = -exp
        if exp.is_infinite:
            if n is S.One and d is not S.One:
                return n, self.func(d, exp)
            if n is not S.One and d is S.One:
                return self.func(n, exp), d
        return self.func(n, exp), self.func(d, exp)

    def matches(self, expr, repl_dict=None, old=False):
        expr = _sympify(expr)
        if repl_dict is None:
            repl_dict = {}

        # special case, pattern = 1 and expr.exp can match to 0
        if expr is S.One:
            d = self.exp.matches(S.Zero, repl_dict)
            if d is not None:
                return d

        # make sure the expression to be matched is an Expr
        if not isinstance(expr, Expr):
            return None

        b, e = expr.as_base_exp()

        # special case number
        sb, se = self.as_base_exp()
        if sb.is_Symbol and se.is_Integer and expr:
            if e.is_rational:
                return sb.matches(b**(e/se), repl_dict)
            return sb.matches(expr**(1/se), repl_dict)

        d = repl_dict.copy()
        d = self.base.matches(b, d)
        if d is None:
            return None

        d = self.exp.xreplace(d).matches(e, d)
        if d is None:
            return Expr.matches(self, expr, repl_dict)
        return d

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE! This function is an important part of the gruntz algorithm
        #       for computing limits. It has to return a generalized power
        #       series with coefficients in C(log, log(x)). In more detail:
        # It has to return an expression
        #     c_0*x**e_0 + c_1*x**e_1 + ... (finitely many terms)
        # where e_i are numbers (not necessarily integers) and c_i are
        # expressions involving only numbers, the log function, and log(x).
        # The series expansion of b**e is computed as follows:
        # 1) We express b as f*(1 + g) where f is the leading term of b.
        #    g has order O(x**d) where d is strictly positive.
        # 2) Then b**e = (f**e)*((1 + g)**e).
        #    (1 + g)**e is computed using binomial series.
        from sympy.functions.elementary.exponential import exp, log
        from sympy.series.limits import limit
        from sympy.series.order import Order
        from sympy.core.sympify import sympify
        if self.base is S.Exp1:
            e_series = self.exp.nseries(x, n=n, logx=logx)
            if e_series.is_Order:
                return 1 + e_series
            e0 = limit(e_series.removeO(), x, 0)
            if e0 is S.NegativeInfinity:
                return Order(x**n, x)
            if e0 is S.Infinity:
                return self
            t = e_series - e0
            exp_series = term = exp(e0)
            # series of exp(e0 + t) in t
            for i in range(1, n):
                term *= t/i
                term = term.nseries(x, n=n, logx=logx)
                exp_series += term
            exp_series += Order(t**n, x)
            from sympy.simplify.powsimp import powsimp
            return powsimp(exp_series, deep=True, combine='exp')
        from sympy.simplify.powsimp import powdenest
        from .numbers import _illegal
        self = powdenest(self, force=True).trigsimp()
        b, e = self.as_base_exp()

        if e.has(*_illegal):
            raise PoleError()

        if e.has(x):
            return exp(e*log(b))._eval_nseries(x, n=n, logx=logx, cdir=cdir)

        if logx is not None and b.has(log):
            from .symbol import Wild
            c, ex = symbols('c, ex', cls=Wild, exclude=[x])
            b = b.replace(log(c*x**ex), log(c) + ex*logx)
            self = b**e

        b = b.removeO()
        try:
            from sympy.functions.special.gamma_functions import polygamma
            if b.has(polygamma, S.EulerGamma) and logx is not None:
                raise ValueError()
            _, m = b.leadterm(x)
        except (ValueError, NotImplementedError, PoleError):
            b = b._eval_nseries(x, n=max(2, n), logx=logx, cdir=cdir).removeO()
            if b.has(S.NaN, S.ComplexInfinity):
                raise NotImplementedError()
            _, m = b.leadterm(x)

        if e.has(log):
            from sympy.simplify.simplify import logcombine
            e = logcombine(e).cancel()

        if not (m.is_zero or e.is_number and e.is_real):
            if self == self._eval_as_leading_term(x, logx=logx, cdir=cdir):
                res = exp(e*log(b))._eval_nseries(x, n=n, logx=logx, cdir=cdir)
                if res == exp(e*log(b)):
                    return self
                return res

        f = b.as_leading_term(x, logx=logx)
        g = (_mexpand(b) - f).cancel()
        g = g/f
        if not m.is_number:
            raise NotImplementedError()
        maxpow = n - m*e
        if maxpow.has(Symbol):
            maxpow = sympify(n)

        if maxpow.is_negative:
            return Order(x**(m*e), x)

        if g.is_zero:
            r = f**e
            if r != self:
                r += Order(x**n, x)
            return r

        def coeff_exp(term, x):
            coeff, exp = S.One, S.Zero
            for factor in Mul.make_args(term):
                if factor.has(x):
                    base, exp = factor.as_base_exp()
                    if base != x:
                        try:
                            return term.leadterm(x)
                        except ValueError:
                            return term, S.Zero
                else:
                    coeff *= factor
            return coeff, exp

        def mul(d1, d2):
            res = {}
            for e1, e2 in product(d1, d2):
                ex = e1 + e2
                if ex < maxpow:
                    res[ex] = res.get(ex, S.Zero) + d1[e1]*d2[e2]
            return res

        try:
            c, d = g.leadterm(x, logx=logx)
        except (ValueError, NotImplementedError):
            if limit(g/x**maxpow, x, 0) == 0:
                # g has higher order zero
                return f**e + e*f**e*g  # first term of binomial series
            else:
                raise NotImplementedError()
        if c.is_Float and d == S.Zero:
            # Convert floats like 0.5 to exact SymPy numbers like S.Half, to
            # prevent rounding errors which can induce wrong values of d leading
            # to a NotImplementedError being returned from the block below.
            g = g.replace(lambda x: x.is_Float, lambda x: Rational(x))
            _, d = g.leadterm(x, logx=logx)
        if not d.is_positive:
            g = g.simplify()
            if g.is_zero:
                return f**e
            _, d = g.leadterm(x, logx=logx)
            if not d.is_positive:
                g = ((b - f)/f).expand()
                _, d = g.leadterm(x, logx=logx)
                if not d.is_positive:
                    raise NotImplementedError()

        from sympy.functions.elementary.integers import ceiling
        gpoly = g._eval_nseries(x, n=ceiling(maxpow), logx=logx, cdir=cdir).removeO()
        gterms = {}

        for term in Add.make_args(gpoly):
            co1, e1 = coeff_exp(term, x)
            gterms[e1] = gterms.get(e1, S.Zero) + co1

        k = S.One
        terms = {S.Zero: S.One}
        tk = gterms

        from sympy.functions.combinatorial.factorials import factorial, ff

        while (k*d - maxpow).is_negative:
            coeff = ff(e, k)/factorial(k)
            for ex in tk:
                terms[ex] = terms.get(ex, S.Zero) + coeff*tk[ex]
            tk = mul(tk, gterms)
            k += S.One

        from sympy.functions.elementary.complexes import im

        if not e.is_integer and m.is_zero and f.is_negative:
            ndir = (b - f).dir(x, cdir)
            if im(ndir).is_negative:
                inco, inex = coeff_exp(f**e*(-1)**(-2*e), x)
            elif im(ndir).is_zero:
                inco, inex = coeff_exp(exp(e*log(b)).as_leading_term(x, logx=logx, cdir=cdir), x)
            else:
                inco, inex = coeff_exp(f**e, x)
        else:
            inco, inex = coeff_exp(f**e, x)
        res = S.Zero

        for e1 in terms:
            ex = e1 + inex
            res += terms[e1]*inco*x**(ex)

        if not (e.is_integer and e.is_positive and (e*d - n).is_nonpositive and
                res == _mexpand(self)):
            try:
                res += Order(x**n, x)
            except NotImplementedError:
                return exp(e*log(b))._eval_nseries(x, n=n, logx=logx, cdir=cdir)
        return res

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.functions.elementary.exponential import exp, log
        e = self.exp
        b = self.base
        if self.base is S.Exp1:
            arg = e.as_leading_term(x, logx=logx)
            arg0 = arg.subs(x, 0)
            if arg0 is S.NaN:
                arg0 = arg.limit(x, 0)
            if arg0.is_infinite is False:
                return S.Exp1**arg0
            raise PoleError("Cannot expand %s around 0" % (self))
        elif e.has(x):
            lt = exp(e * log(b))
            return lt.as_leading_term(x, logx=logx, cdir=cdir)
        else:
            from sympy.functions.elementary.complexes import im
            try:
                f = b.as_leading_term(x, logx=logx, cdir=cdir)
            except PoleError:
                return self
            if not e.is_integer and f.is_negative and not f.has(x):
                ndir = (b - f).dir(x, cdir)
                if im(ndir).is_negative:
                    # Normally, f**e would evaluate to exp(e*log(f)) but on branch cuts
                    # an other value is expected through the following computation
                    # exp(e*(log(f) - 2*pi*I)) == f**e*exp(-2*e*pi*I) == f**e*(-1)**(-2*e).
                    return self.func(f, e) * (-1)**(-2*e)
                elif im(ndir).is_zero:
                    log_leadterm = log(b)._eval_as_leading_term(x, logx=logx, cdir=cdir)
                    if log_leadterm.is_infinite is False:
                        return exp(e*log_leadterm)
            return self.func(f, e)

    @cacheit
    def _taylor_term(self, n, x, *previous_terms): # of (1 + x)**e
        from sympy.functions.combinatorial.factorials import binomial
        return binomial(self.exp, n) * self.func(x, n)

    def taylor_term(self, n, x, *previous_terms):
        if self.base is not S.Exp1:
            return super().taylor_term(n, x, *previous_terms)
        if n < 0:
            return S.Zero
        if n == 0:
            return S.One
        from .sympify import sympify
        x = sympify(x)
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return p * x / n
        from sympy.functions.combinatorial.factorials import factorial
        return x**n/factorial(n)

    def _eval_rewrite_as_sin(self, base, exp, **hints):
        if self.base is S.Exp1:
            from sympy.functions.elementary.trigonometric import sin
            return sin(S.ImaginaryUnit*self.exp + S.Pi/2) - S.ImaginaryUnit*sin(S.ImaginaryUnit*self.exp)

    def _eval_rewrite_as_cos(self, base, exp, **hints):
        if self.base is S.Exp1:
            from sympy.functions.elementary.trigonometric import cos
            return cos(S.ImaginaryUnit*self.exp) + S.ImaginaryUnit*cos(S.ImaginaryUnit*self.exp + S.Pi/2)

    def _eval_rewrite_as_tanh(self, base, exp, **hints):
        if self.base is S.Exp1:
            from sympy.functions.elementary.hyperbolic import tanh
            return (1 + tanh(self.exp/2))/(1 - tanh(self.exp/2))

    def _eval_rewrite_as_sqrt(self, base, exp, **kwargs):
        from sympy.functions.elementary.trigonometric import sin, cos
        if base is not S.Exp1:
            return None
        if exp.is_Mul:
            coeff = exp.coeff(S.Pi * S.ImaginaryUnit)
            if coeff and coeff.is_number:
                cosine, sine = cos(S.Pi*coeff), sin(S.Pi*coeff)
                if not isinstance(cosine, cos) and not isinstance (sine, sin):
                    return cosine + S.ImaginaryUnit*sine

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import sqrt
        >>> sqrt(4 + 4*sqrt(2)).as_content_primitive()
        (2, sqrt(1 + sqrt(2)))
        >>> sqrt(3 + 3*sqrt(2)).as_content_primitive()
        (1, sqrt(3)*sqrt(1 + sqrt(2)))

        >>> from sympy import expand_power_base, powsimp, Mul
        >>> from sympy.abc import x, y

        >>> ((2*x + 2)**2).as_content_primitive()
        (4, (x + 1)**2)
        >>> (4**((1 + y)/2)).as_content_primitive()
        (2, 4**(y/2))
        >>> (3**((1 + y)/2)).as_content_primitive()
        (1, 3**((y + 1)/2))
        >>> (3**((5 + y)/2)).as_content_primitive()
        (9, 3**((y + 1)/2))
        >>> eq = 3**(2 + 2*x)
        >>> powsimp(eq) == eq
        True
        >>> eq.as_content_primitive()
        (9, 3**(2*x))
        >>> powsimp(Mul(*_))
        3**(2*x + 2)

        >>> eq = (2 + 2*x)**y
        >>> s = expand_power_base(eq); s.is_Mul, s
        (False, (2*x + 2)**y)
        >>> eq.as_content_primitive()
        (1, (2*(x + 1))**y)
        >>> s = expand_power_base(_[1]); s.is_Mul, s
        (True, 2**y*(x + 1)**y)

        See docstring of Expr.as_content_primitive for more examples.
        """

        b, e = self.as_base_exp()
        b = _keep_coeff(*b.as_content_primitive(radical=radical, clear=clear))
        ce, pe = e.as_content_primitive(radical=radical, clear=clear)
        if b.is_Rational:
            #e
            #= ce*pe
            #= ce*(h + t)
            #= ce*h + ce*t
            #=> self
            #= b**(ce*h)*b**(ce*t)
            #= b**(cehp/cehq)*b**(ce*t)
            #= b**(iceh + r/cehq)*b**(ce*t)
            #= b**(iceh)*b**(r/cehq)*b**(ce*t)
            #= b**(iceh)*b**(ce*t + r/cehq)
            h, t = pe.as_coeff_Add()
            if h.is_Rational and b != S.Zero:
                ceh = ce*h
                c = self.func(b, ceh)
                r = S.Zero
                if not c.is_Rational:
                    iceh, r = divmod(ceh.p, ceh.q)
                    c = self.func(b, iceh)
                return c, self.func(b, _keep_coeff(ce, t + r/ce/ceh.q))
        e = _keep_coeff(ce, pe)
        # b**e = (h*t)**e = h**e*t**e = c*m*t**e
        if e.is_Rational and b.is_Mul:
            h, t = b.as_content_primitive(radical=radical, clear=clear)  # h is positive
            c, m = self.func(h, e).as_coeff_Mul()  # so c is positive
            m, me = m.as_base_exp()
            if m is S.One or me == e:  # probably always true
                # return the following, not return c, m*Pow(t, e)
                # which would change Pow into Mul; we let SymPy
                # decide what to do by using the unevaluated Mul, e.g
                # should it stay as sqrt(2 + 2*sqrt(5)) or become
                # sqrt(2)*sqrt(1 + sqrt(5))
                return c, self.func(_keep_coeff(m, t), e)
        return S.One, self.func(b, e)

    def is_constant(self, *wrt, **flags):
        expr = self
        if flags.get('simplify', True):
            expr = expr.simplify()
        b, e = expr.as_base_exp()
        bz = b.equals(0)
        if bz:  # recalculate with assumptions in case it's unevaluated
            new = b**e
            if new != expr:
                return new.is_constant()
        econ = e.is_constant(*wrt)
        bcon = b.is_constant(*wrt)
        if bcon:
            if econ:
                return True
            bz = b.equals(0)
            if bz is False:
                return False
        elif bcon is None:
            return None

        return e.equals(0)

    def _eval_difference_delta(self, n, step):
        b, e = self.args
        if e.has(n) and not b.has(n):
            new_e = e.subs(n, n + step)
            return (b**(new_e - e) - 1) * self

power = Dispatcher('power')
power.add((object, object), Pow)

from .add import Add
from .numbers import Integer, Rational
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols