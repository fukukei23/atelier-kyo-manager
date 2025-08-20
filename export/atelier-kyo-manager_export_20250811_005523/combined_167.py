
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\error_prop.py ===
"""Tools for arithmetic error propagation."""

from itertools import repeat, combinations

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.simplify.simplify import simplify
from sympy.stats.symbolic_probability import RandomSymbol, Variance, Covariance
from sympy.stats.rv import is_random

_arg0_or_var = lambda var: var.args[0] if len(var.args) > 0 else var


def variance_prop(expr, consts=(), include_covar=False):
    r"""Symbolically propagates variance (`\sigma^2`) for expressions.
    This is computed as as seen in [1]_.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to compute the variance for.
    consts : sequence of Symbols, optional
        Represents symbols that are known constants in the expr,
        and thus have zero variance. All symbols not in consts are
        assumed to be variant.
    include_covar : bool, optional
        Flag for whether or not to include covariances, default=False.

    Returns
    =======

    var_expr : Expr
        An expression for the total variance of the expr.
        The variance for the original symbols (e.g. x) are represented
        via instance of the Variance symbol (e.g. Variance(x)).

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy.stats.error_prop import variance_prop
    >>> x, y = symbols('x y')

    >>> variance_prop(x + y)
    Variance(x) + Variance(y)

    >>> variance_prop(x * y)
    x**2*Variance(y) + y**2*Variance(x)

    >>> variance_prop(exp(2*x))
    4*exp(4*x)*Variance(x)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Propagation_of_uncertainty

    """
    args = expr.args
    if len(args) == 0:
        if expr in consts:
            return S.Zero
        elif is_random(expr):
            return Variance(expr).doit()
        elif isinstance(expr, Symbol):
            return Variance(RandomSymbol(expr)).doit()
        else:
            return S.Zero
    nargs = len(args)
    var_args = list(map(variance_prop, args, repeat(consts, nargs),
                        repeat(include_covar, nargs)))
    if isinstance(expr, Add):
        var_expr = Add(*var_args)
        if include_covar:
            terms = [2 * Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand() \
                     for x, y in combinations(var_args, 2)]
            var_expr += Add(*terms)
    elif isinstance(expr, Mul):
        terms = [v/a**2 for a, v in zip(args, var_args)]
        var_expr = simplify(expr**2 * Add(*terms))
        if include_covar:
            terms = [2*Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand()/(a*b) \
                     for (a, b), (x, y) in zip(combinations(args, 2),
                                               combinations(var_args, 2))]
            var_expr += Add(*terms)
    elif isinstance(expr, Pow):
        b = args[1]
        v = var_args[0] * (expr * b / args[0])**2
        var_expr = simplify(v)
    elif isinstance(expr, exp):
        var_expr = simplify(var_args[0] * expr**2)
    else:
        # unknown how to proceed, return variance of whole expr.
        var_expr = Variance(expr)
    return var_expr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\csp.py ===
from __future__ import annotations

import collections.abc as cabc
import typing as t

from .structures import CallbackDict


def csp_property(key: str) -> t.Any:
    """Return a new property object for a content security policy header.
    Useful if you want to add support for a csp extension in a
    subclass.
    """
    return property(
        lambda x: x._get_value(key),
        lambda x, v: x._set_value(key, v),
        lambda x: x._del_value(key),
        f"accessor for {key!r}",
    )


class ContentSecurityPolicy(CallbackDict[str, str]):
    """Subclass of a dict that stores values for a Content Security Policy
    header. It has accessors for all the level 3 policies.

    Because the csp directives in the HTTP header use dashes the
    python descriptors use underscores for that.

    To get a header of the :class:`ContentSecuirtyPolicy` object again
    you can convert the object into a string or call the
    :meth:`to_header` method.  If you plan to subclass it and add your
    own items have a look at the sourcecode for that class.

    .. versionadded:: 1.0.0
       Support for Content Security Policy headers was added.

    """

    base_uri: str | None = csp_property("base-uri")
    child_src: str | None = csp_property("child-src")
    connect_src: str | None = csp_property("connect-src")
    default_src: str | None = csp_property("default-src")
    font_src: str | None = csp_property("font-src")
    form_action: str | None = csp_property("form-action")
    frame_ancestors: str | None = csp_property("frame-ancestors")
    frame_src: str | None = csp_property("frame-src")
    img_src: str | None = csp_property("img-src")
    manifest_src: str | None = csp_property("manifest-src")
    media_src: str | None = csp_property("media-src")
    navigate_to: str | None = csp_property("navigate-to")
    object_src: str | None = csp_property("object-src")
    prefetch_src: str | None = csp_property("prefetch-src")
    plugin_types: str | None = csp_property("plugin-types")
    report_to: str | None = csp_property("report-to")
    report_uri: str | None = csp_property("report-uri")
    sandbox: str | None = csp_property("sandbox")
    script_src: str | None = csp_property("script-src")
    script_src_attr: str | None = csp_property("script-src-attr")
    script_src_elem: str | None = csp_property("script-src-elem")
    style_src: str | None = csp_property("style-src")
    style_src_attr: str | None = csp_property("style-src-attr")
    style_src_elem: str | None = csp_property("style-src-elem")
    worker_src: str | None = csp_property("worker-src")

    def __init__(
        self,
        values: cabc.Mapping[str, str] | cabc.Iterable[tuple[str, str]] | None = (),
        on_update: cabc.Callable[[ContentSecurityPolicy], None] | None = None,
    ) -> None:
        super().__init__(values, on_update)
        self.provided = values is not None

    def _get_value(self, key: str) -> str | None:
        """Used internally by the accessor properties."""
        return self.get(key)

    def _set_value(self, key: str, value: str | None) -> None:
        """Used internally by the accessor properties."""
        if value is None:
            self.pop(key, None)
        else:
            self[key] = value

    def _del_value(self, key: str) -> None:
        """Used internally by the accessor properties."""
        if key in self:
            del self[key]

    def to_header(self) -> str:
        """Convert the stored values into a cache control header."""
        from ..http import dump_csp_header

        return dump_csp_header(self)

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        kv_str = " ".join(f"{k}={v!r}" for k, v in sorted(self.items()))
        return f"<{type(self).__name__} {kv_str}>"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\async_utils.py ===
import inspect
import typing as t
from functools import WRAPPER_ASSIGNMENTS
from functools import wraps

from .utils import _PassArg
from .utils import pass_eval_context

if t.TYPE_CHECKING:
    import typing_extensions as te

V = t.TypeVar("V")


def async_variant(normal_func):  # type: ignore
    def decorator(async_func):  # type: ignore
        pass_arg = _PassArg.from_obj(normal_func)
        need_eval_context = pass_arg is None

        if pass_arg is _PassArg.environment:

            def is_async(args: t.Any) -> bool:
                return t.cast(bool, args[0].is_async)

        else:

            def is_async(args: t.Any) -> bool:
                return t.cast(bool, args[0].environment.is_async)

        # Take the doc and annotations from the sync function, but the
        # name from the async function. Pallets-Sphinx-Themes
        # build_function_directive expects __wrapped__ to point to the
        # sync function.
        async_func_attrs = ("__module__", "__name__", "__qualname__")
        normal_func_attrs = tuple(set(WRAPPER_ASSIGNMENTS).difference(async_func_attrs))

        @wraps(normal_func, assigned=normal_func_attrs)
        @wraps(async_func, assigned=async_func_attrs, updated=())
        def wrapper(*args, **kwargs):  # type: ignore
            b = is_async(args)

            if need_eval_context:
                args = args[1:]

            if b:
                return async_func(*args, **kwargs)

            return normal_func(*args, **kwargs)

        if need_eval_context:
            wrapper = pass_eval_context(wrapper)

        wrapper.jinja_async_variant = True  # type: ignore[attr-defined]
        return wrapper

    return decorator


_common_primitives = {int, float, bool, str, list, dict, tuple, type(None)}


async def auto_await(value: t.Union[t.Awaitable["V"], "V"]) -> "V":
    # Avoid a costly call to isawaitable
    if type(value) in _common_primitives:
        return t.cast("V", value)

    if inspect.isawaitable(value):
        return await t.cast("t.Awaitable[V]", value)

    return value


class _IteratorToAsyncIterator(t.Generic[V]):
    def __init__(self, iterator: "t.Iterator[V]"):
        self._iterator = iterator

    def __aiter__(self) -> "te.Self":
        return self

    async def __anext__(self) -> V:
        try:
            return next(self._iterator)
        except StopIteration as e:
            raise StopAsyncIteration(e.value) from e


def auto_aiter(
    iterable: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
) -> "t.AsyncIterator[V]":
    if hasattr(iterable, "__aiter__"):
        return iterable.__aiter__()
    else:
        return _IteratorToAsyncIterator(iter(iterable))


async def auto_to_list(
    value: "t.Union[t.AsyncIterable[V], t.Iterable[V]]",
) -> t.List["V"]:
    return [x async for x in auto_aiter(value)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\use_rules.py ===
"""
Build 'use others module data' mechanism for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
__version__ = "$Revision: 1.3 $"[10:-1]

f2py_version = 'See `f2py -v`'


from .auxfuncs import applyrules, dictappend, gentitle, hasnote, outmess

usemodule_rules = {
    'body': """
#begintitle#
static char doc_#apiname#[] = \"\\\nVariable wrapper signature:\\n\\
\t #name# = get_#name#()\\n\\
Arguments:\\n\\
#docstr#\";
extern F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#);
static PyObject *#apiname#(PyObject *capi_self, PyObject *capi_args) {
/*#decl#*/
\tif (!PyArg_ParseTuple(capi_args, \"\")) goto capi_fail;
printf(\"c: %d\\n\",F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#));
\treturn Py_BuildValue(\"\");
capi_fail:
\treturn NULL;
}
""",
    'method': '\t{\"get_#name#\",#apiname#,METH_VARARGS|METH_KEYWORDS,doc_#apiname#},',
    'need': ['F_MODFUNC']
}

################


def buildusevars(m, r):
    ret = {}
    outmess(
        f"\t\tBuilding use variable hooks for module \"{m['name']}\" (feature only for F90/F95)...\n")
    varsmap = {}
    revmap = {}
    if 'map' in r:
        for k in r['map'].keys():
            if r['map'][k] in revmap:
                outmess('\t\t\tVariable "%s<=%s" is already mapped by "%s". Skipping.\n' % (
                    r['map'][k], k, revmap[r['map'][k]]))
            else:
                revmap[r['map'][k]] = k
    if r.get('only'):
        for v in r['map'].keys():
            if r['map'][v] in m['vars']:

                if revmap[r['map'][v]] == v:
                    varsmap[v] = r['map'][v]
                else:
                    outmess(f"\t\t\tIgnoring map \"{v}=>{r['map'][v]}\". See above.\n")
            else:
                outmess(
                    f"\t\t\tNo definition for variable \"{v}=>{r['map'][v]}\". Skipping.\n")
    else:
        for v in m['vars'].keys():
            varsmap[v] = revmap.get(v, v)
    for v in varsmap.keys():
        ret = dictappend(ret, buildusevar(v, varsmap[v], m['vars'], m['name']))
    return ret


def buildusevar(name, realname, vars, usemodulename):
    outmess('\t\t\tConstructing wrapper function for variable "%s=>%s"...\n' % (
        name, realname))
    ret = {}
    vrd = {'name': name,
           'realname': realname,
           'REALNAME': realname.upper(),
           'usemodulename': usemodulename,
           'USEMODULENAME': usemodulename.upper(),
           'texname': name.replace('_', '\\_'),
           'begintitle': gentitle(f'{name}=>{realname}'),
           'endtitle': gentitle(f'end of {name}=>{realname}'),
           'apiname': f'#modulename#_use_{realname}_from_{usemodulename}'
           }
    nummap = {0: 'Ro', 1: 'Ri', 2: 'Rii', 3: 'Riii', 4: 'Riv',
              5: 'Rv', 6: 'Rvi', 7: 'Rvii', 8: 'Rviii', 9: 'Rix'}
    vrd['texnamename'] = name
    for i in nummap.keys():
        vrd['texnamename'] = vrd['texnamename'].replace(repr(i), nummap[i])
    if hasnote(vars[realname]):
        vrd['note'] = vars[realname]['note']
    rd = dictappend({}, vrd)

    print(name, realname, vars[realname])
    ret = applyrules(usemodule_rules, rd)
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_nhwc_conv.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionNhwcConv(Fusion):
    """Convert Conv to NhwcConv"""

    def __init__(self, model: OnnxModel, update_weight=False):
        super().__init__(model, "NhwcConv", ["Conv"], "NhwcConv")
        self.update_weight = update_weight
        self.fusion_utils = FusionUtils(model)

    def create_transpose_node(self, input_name: str, perm: list[int], output_name=None):
        """Append a Transpose node after an input"""
        node_name = self.model.create_node_name("Transpose")

        if output_name is None:
            output_name = node_name + "_out" + "-" + input_name

        transpose_node = helper.make_node("Transpose", inputs=[input_name], outputs=[output_name], name=node_name)
        transpose_node.attribute.extend([helper.make_attribute("perm", perm)])

        return transpose_node

    def fuse(self, conv, input_name_to_nodes, output_name_to_node):
        # Add Transpose node to convert input from NCHW to NHWC
        input_transpose_node = self.create_transpose_node(conv.input[0], [0, 2, 3, 1])

        nhwc_conv_input = input_transpose_node.output[0]

        # Create a tensor for transposed weights (already in NHWC format).
        node_name = self.model.create_node_name("NhwcConv")

        # Make sure the weights is 4D
        weight_tensor = self.model.get_initializer(conv.input[1])
        if weight_tensor is None:
            return
        weight = numpy_helper.to_array(weight_tensor)
        if len(weight.shape) != 4:
            return

        dtype = self.model.get_dtype(nhwc_conv_input)
        if not (dtype is not None and weight_tensor.data_type == dtype):
            cast_node = self.fusion_utils.add_cast_node(
                input_name=nhwc_conv_input,
                to_type=weight_tensor.data_type,
                output_name_to_node=output_name_to_node,
            )
            nhwc_conv_input = cast_node.output[0]

        if self.update_weight:
            # Transpose weights from NCHW to NHWC
            weight = weight.transpose(0, 2, 3, 1)

            weight_name = node_name + "_weight_NHWC"
            self.add_initializer(
                name=weight_name,
                data_type=weight_tensor.data_type,
                dims=list(weight.shape),
                vals=weight,
            )
            weight_transpose_node = None
        else:
            weight_transpose_node = self.create_transpose_node(conv.input[1], [0, 2, 3, 1])
            weight_name = weight_transpose_node.output[0]

        nhwc_output_name = node_name + "_out" + "-" + conv.output[0]
        nhwc_conv = helper.make_node(
            "NhwcConv",
            inputs=[nhwc_conv_input, weight_name] + conv.input[2:],
            outputs=[nhwc_output_name],
            name=node_name + "-" + conv.name,
        )
        nhwc_conv.attribute.extend(conv.attribute)
        nhwc_conv.domain = "com.microsoft"

        output_transpose_node = self.create_transpose_node(nhwc_conv.output[0], [0, 3, 1, 2], conv.output[0])

        self.nodes_to_remove.append(conv)

        nodes_to_add = [input_transpose_node, nhwc_conv, output_transpose_node]
        if weight_transpose_node:
            nodes_to_add.append(weight_transpose_node)
        for node in nodes_to_add:
            self.node_name_to_graph_name[node.name] = self.this_graph_name
        self.nodes_to_add.extend(nodes_to_add)

        self.increase_counter("NhwcConv")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\structures.py ===
"""
requests.structures
~~~~~~~~~~~~~~~~~~~

Data structures that power Requests.
"""

from collections import OrderedDict

from .compat import Mapping, MutableMapping


class CaseInsensitiveDict(MutableMapping):
    """A case-insensitive ``dict``-like object.

    Implements all methods and operations of
    ``MutableMapping`` as well as dict's ``copy``. Also
    provides ``lower_items``.

    All keys are expected to be strings. The structure remembers the
    case of the last key to be set, and ``iter(instance)``,
    ``keys()``, ``items()``, ``iterkeys()``, and ``iteritems()``
    will contain case-sensitive keys. However, querying and contains
    testing is case insensitive::

        cid = CaseInsensitiveDict()
        cid['Accept'] = 'application/json'
        cid['aCCEPT'] == 'application/json'  # True
        list(cid) == ['Accept']  # True

    For example, ``headers['content-encoding']`` will return the
    value of a ``'Content-Encoding'`` response header, regardless
    of how the header name was originally stored.

    If the constructor, ``.update``, or equality comparison
    operations are given keys that have equal ``.lower()``s, the
    behavior is undefined.
    """

    def __init__(self, data=None, **kwargs):
        self._store = OrderedDict()
        if data is None:
            data = {}
        self.update(data, **kwargs)

    def __setitem__(self, key, value):
        # Use the lowercased key for lookups, but store the actual
        # key alongside the value.
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._store[key.lower()][1]

    def __delitem__(self, key):
        del self._store[key.lower()]

    def __iter__(self):
        return (casedkey for casedkey, mappedvalue in self._store.values())

    def __len__(self):
        return len(self._store)

    def lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return ((lowerkey, keyval[1]) for (lowerkey, keyval) in self._store.items())

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = CaseInsensitiveDict(other)
        else:
            return NotImplemented
        # Compare insensitively
        return dict(self.lower_items()) == dict(other.lower_items())

    # Copy is required
    def copy(self):
        return CaseInsensitiveDict(self._store.values())

    def __repr__(self):
        return str(dict(self.items()))


class LookupDict(dict):
    """Dictionary lookup object."""

    def __init__(self, name=None):
        self.name = name
        super().__init__()

    def __repr__(self):
        return f"<lookup '{self.name}'>"

    def __getitem__(self, key):
        # We allow fall-through here, so values default to None

        return self.__dict__.get(key, None)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\structures.py ===
"""
requests.structures
~~~~~~~~~~~~~~~~~~~

Data structures that power Requests.
"""

from collections import OrderedDict

from .compat import Mapping, MutableMapping


class CaseInsensitiveDict(MutableMapping):
    """A case-insensitive ``dict``-like object.

    Implements all methods and operations of
    ``MutableMapping`` as well as dict's ``copy``. Also
    provides ``lower_items``.

    All keys are expected to be strings. The structure remembers the
    case of the last key to be set, and ``iter(instance)``,
    ``keys()``, ``items()``, ``iterkeys()``, and ``iteritems()``
    will contain case-sensitive keys. However, querying and contains
    testing is case insensitive::

        cid = CaseInsensitiveDict()
        cid['Accept'] = 'application/json'
        cid['aCCEPT'] == 'application/json'  # True
        list(cid) == ['Accept']  # True

    For example, ``headers['content-encoding']`` will return the
    value of a ``'Content-Encoding'`` response header, regardless
    of how the header name was originally stored.

    If the constructor, ``.update``, or equality comparison
    operations are given keys that have equal ``.lower()``s, the
    behavior is undefined.
    """

    def __init__(self, data=None, **kwargs):
        self._store = OrderedDict()
        if data is None:
            data = {}
        self.update(data, **kwargs)

    def __setitem__(self, key, value):
        # Use the lowercased key for lookups, but store the actual
        # key alongside the value.
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key):
        return self._store[key.lower()][1]

    def __delitem__(self, key):
        del self._store[key.lower()]

    def __iter__(self):
        return (casedkey for casedkey, mappedvalue in self._store.values())

    def __len__(self):
        return len(self._store)

    def lower_items(self):
        """Like iteritems(), but with all lowercase keys."""
        return ((lowerkey, keyval[1]) for (lowerkey, keyval) in self._store.items())

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = CaseInsensitiveDict(other)
        else:
            return NotImplemented
        # Compare insensitively
        return dict(self.lower_items()) == dict(other.lower_items())

    # Copy is required
    def copy(self):
        return CaseInsensitiveDict(self._store.values())

    def __repr__(self):
        return str(dict(self.items()))


class LookupDict(dict):
    """Dictionary lookup object."""

    def __init__(self, name=None):
        self.name = name
        super().__init__()

    def __repr__(self):
        return f"<lookup '{self.name}'>"

    def __getitem__(self, key):
        # We allow fall-through here, so values default to None

        return self.__dict__.get(key, None)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\desired_capabilities.py ===
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
"""The Desired Capabilities implementation."""


class DesiredCapabilities:
    """Set of default supported desired capabilities.

    Use this as a starting point for creating a desired capabilities object for
    requesting remote webdrivers for connecting to selenium server or selenium grid.

    Usage Example::

        from selenium import webdriver

        selenium_grid_url = "http://198.0.0.1:4444/wd/hub"

        # Create a desired capabilities object as a starting point.
        capabilities = DesiredCapabilities.FIREFOX.copy()
        capabilities["platform"] = "WINDOWS"
        capabilities["version"] = "10"

        # Instantiate an instance of Remote WebDriver with the desired capabilities.
        driver = webdriver.Remote(desired_capabilities=capabilities, command_executor=selenium_grid_url)

    Note: Always use '.copy()' on the DesiredCapabilities object to avoid the side
    effects of altering the Global class instance.
    """

    FIREFOX = {
        "browserName": "firefox",
        "acceptInsecureCerts": True,
        "moz:debuggerAddress": True,
    }

    INTERNETEXPLORER = {
        "browserName": "internet explorer",
        "platformName": "windows",
    }

    EDGE = {
        "browserName": "MicrosoftEdge",
    }

    CHROME = {
        "browserName": "chrome",
    }

    SAFARI = {
        "browserName": "safari",
        "platformName": "mac",
    }

    HTMLUNIT = {
        "browserName": "htmlunit",
        "version": "",
        "platform": "ANY",
    }

    HTMLUNITWITHJS = {
        "browserName": "htmlunit",
        "version": "firefox",
        "platform": "ANY",
        "javascriptEnabled": True,
    }

    IPHONE = {
        "browserName": "iPhone",
        "version": "",
        "platform": "mac",
    }

    IPAD = {
        "browserName": "iPad",
        "version": "",
        "platform": "mac",
    }

    WEBKITGTK = {
        "browserName": "MiniBrowser",
    }

    WPEWEBKIT = {
        "browserName": "MiniBrowser",
    }

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\driver_finder.py ===
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
import logging
from pathlib import Path

from selenium.common.exceptions import NoSuchDriverException
from selenium.webdriver.common.options import BaseOptions
from selenium.webdriver.common.selenium_manager import SeleniumManager
from selenium.webdriver.common.service import Service

logger = logging.getLogger(__name__)


class DriverFinder:
    """A Driver finding class responsible for obtaining the correct driver and
    associated browser.

    :param service: instance of the driver service class.
    :param options: instance of the browser options class.
    """

    def __init__(self, service: Service, options: BaseOptions) -> None:
        self._service = service
        self._options = options
        self._paths = {"driver_path": "", "browser_path": ""}

    """Utility to find if a given file is present and executable.

    This implementation is still in beta, and may change.
    """

    def get_browser_path(self) -> str:
        return self._binary_paths()["browser_path"]

    def get_driver_path(self) -> str:
        return self._binary_paths()["driver_path"]

    def _binary_paths(self) -> dict:
        if self._paths["driver_path"]:
            return self._paths

        browser = self._options.capabilities["browserName"]
        try:
            path = self._service.path
            if path:
                logger.debug(
                    "Skipping Selenium Manager; path to %s driver specified in Service class: %s", browser, path
                )
                if not Path(path).is_file():
                    raise ValueError(f"The path is not a valid file: {path}")
                self._paths["driver_path"] = path
            else:
                output = SeleniumManager().binary_paths(self._to_args())
                if Path(output["driver_path"]).is_file():
                    self._paths["driver_path"] = output["driver_path"]
                else:
                    raise ValueError(f"The driver path is not a valid file: {output['driver_path']}")
                if Path(output["browser_path"]).is_file():
                    self._paths["browser_path"] = output["browser_path"]
                else:
                    raise ValueError(f"The browser path is not a valid file: {output['browser_path']}")
        except Exception as err:
            msg = f"Unable to obtain driver for {browser}"
            raise NoSuchDriverException(msg) from err
        return self._paths

    def _to_args(self) -> list:
        args = ["--browser", self._options.capabilities["browserName"]]

        if self._options.browser_version:
            args.append("--browser-version")
            args.append(str(self._options.browser_version))

        binary_location = getattr(self._options, "binary_location", None)
        if binary_location:
            args.append("--browser-path")
            args.append(str(binary_location))

        proxy = self._options.proxy
        if proxy and (proxy.http_proxy or proxy.ssl_proxy):
            args.append("--proxy")
            value = proxy.ssl_proxy if proxy.ssl_proxy else proxy.http_proxy
            args.append(value)

        return args

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\io.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: IO
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime


class StreamHandle(str):
    '''
    This is either obtained from another method or specified as ``blob:<uuid>`` where
    ``<uuid>`` is an UUID of a Blob.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> StreamHandle:
        return cls(json)

    def __repr__(self):
        return 'StreamHandle({})'.format(super().__repr__())


def close(
        handle: StreamHandle
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Close the stream, discard any temporary backing storage.

    :param handle: Handle of the stream to close.
    '''
    params: T_JSON_DICT = dict()
    params['handle'] = handle.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IO.close',
        'params': params,
    }
    json = yield cmd_dict


def read(
        handle: StreamHandle,
        offset: typing.Optional[int] = None,
        size: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[bool], str, bool]]:
    '''
    Read a chunk of the stream

    :param handle: Handle of the stream to read.
    :param offset: *(Optional)* Seek to the specified offset before reading (if not specified, proceed with offset following the last read). Some types of streams may only support sequential reads.
    :param size: *(Optional)* Maximum number of bytes to read (left upon the agent discretion if not specified).
    :returns: A tuple with the following items:

        0. **base64Encoded** - *(Optional)* Set if the data is base64-encoded
        1. **data** - Data that were read.
        2. **eof** - Set if the end-of-file condition occurred while reading.
    '''
    params: T_JSON_DICT = dict()
    params['handle'] = handle.to_json()
    if offset is not None:
        params['offset'] = offset
    if size is not None:
        params['size'] = size
    cmd_dict: T_JSON_DICT = {
        'method': 'IO.read',
        'params': params,
    }
    json = yield cmd_dict
    return (
        bool(json['base64Encoded']) if 'base64Encoded' in json else None,
        str(json['data']),
        bool(json['eof'])
    )


def resolve_blob(
        object_id: runtime.RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Return UUID of Blob object specified by a remote object id.

    :param object_id: Object id of a Blob object wrapper.
    :returns: UUID of the specified Blob.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IO.resolveBlob',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['uuid'])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\io.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: IO
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime


class StreamHandle(str):
    '''
    This is either obtained from another method or specified as ``blob:<uuid>`` where
    ``<uuid>`` is an UUID of a Blob.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> StreamHandle:
        return cls(json)

    def __repr__(self):
        return 'StreamHandle({})'.format(super().__repr__())


def close(
        handle: StreamHandle
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Close the stream, discard any temporary backing storage.

    :param handle: Handle of the stream to close.
    '''
    params: T_JSON_DICT = dict()
    params['handle'] = handle.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IO.close',
        'params': params,
    }
    json = yield cmd_dict


def read(
        handle: StreamHandle,
        offset: typing.Optional[int] = None,
        size: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[bool], str, bool]]:
    '''
    Read a chunk of the stream

    :param handle: Handle of the stream to read.
    :param offset: *(Optional)* Seek to the specified offset before reading (if not specified, proceed with offset following the last read). Some types of streams may only support sequential reads.
    :param size: *(Optional)* Maximum number of bytes to read (left upon the agent discretion if not specified).
    :returns: A tuple with the following items:

        0. **base64Encoded** - *(Optional)* Set if the data is base64-encoded
        1. **data** - Data that were read.
        2. **eof** - Set if the end-of-file condition occurred while reading.
    '''
    params: T_JSON_DICT = dict()
    params['handle'] = handle.to_json()
    if offset is not None:
        params['offset'] = offset
    if size is not None:
        params['size'] = size
    cmd_dict: T_JSON_DICT = {
        'method': 'IO.read',
        'params': params,
    }
    json = yield cmd_dict
    return (
        bool(json['base64Encoded']) if 'base64Encoded' in json else None,
        str(json['data']),
        bool(json['eof'])
    )


def resolve_blob(
        object_id: runtime.RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Return UUID of Blob object specified by a remote object id.

    :param object_id: Object id of a Blob object wrapper.
    :returns: UUID of the specified Blob.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IO.resolveBlob',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['uuid'])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\io.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: IO
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime


class StreamHandle(str):
    '''
    This is either obtained from another method or specified as ``blob:<uuid>`` where
    ``<uuid>`` is an UUID of a Blob.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> StreamHandle:
        return cls(json)

    def __repr__(self):
        return 'StreamHandle({})'.format(super().__repr__())


def close(
        handle: StreamHandle
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Close the stream, discard any temporary backing storage.

    :param handle: Handle of the stream to close.
    '''
    params: T_JSON_DICT = dict()
    params['handle'] = handle.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IO.close',
        'params': params,
    }
    json = yield cmd_dict


def read(
        handle: StreamHandle,
        offset: typing.Optional[int] = None,
        size: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[bool], str, bool]]:
    '''
    Read a chunk of the stream

    :param handle: Handle of the stream to read.
    :param offset: *(Optional)* Seek to the specified offset before reading (if not specified, proceed with offset following the last read). Some types of streams may only support sequential reads.
    :param size: *(Optional)* Maximum number of bytes to read (left upon the agent discretion if not specified).
    :returns: A tuple with the following items:

        0. **base64Encoded** - *(Optional)* Set if the data is base64-encoded
        1. **data** - Data that were read.
        2. **eof** - Set if the end-of-file condition occurred while reading.
    '''
    params: T_JSON_DICT = dict()
    params['handle'] = handle.to_json()
    if offset is not None:
        params['offset'] = offset
    if size is not None:
        params['size'] = size
    cmd_dict: T_JSON_DICT = {
        'method': 'IO.read',
        'params': params,
    }
    json = yield cmd_dict
    return (
        bool(json['base64Encoded']) if 'base64Encoded' in json else None,
        str(json['data']),
        bool(json['eof'])
    )


def resolve_blob(
        object_id: runtime.RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Return UUID of Blob object specified by a remote object id.

    :param object_id: Object id of a Blob object wrapper.
    :returns: UUID of the specified Blob.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IO.resolveBlob',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['uuid'])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\groundtypes.py ===
"""Ground types for various mathematical domains in SymPy. """

import builtins
from sympy.external.gmpy import GROUND_TYPES, factorial, sqrt, is_square, sqrtrem

PythonInteger = builtins.int
PythonReal = builtins.float
PythonComplex = builtins.complex

from .pythonrational import PythonRational

from sympy.core.intfunc import (
    igcdex as python_gcdex,
    igcd2 as python_gcd,
    ilcm as python_lcm,
)

from sympy.core.numbers import (Float as SymPyReal, Integer as SymPyInteger, Rational as SymPyRational)


class _GMPYInteger:
    def __init__(self, obj):
        pass

class _GMPYRational:
    def __init__(self, obj):
        pass


if GROUND_TYPES == 'gmpy':

    from gmpy2 import (
        mpz as GMPYInteger,
        mpq as GMPYRational,
        numer as gmpy_numer,
        denom as gmpy_denom,
        gcdext as gmpy_gcdex,
        gcd as gmpy_gcd,
        lcm as gmpy_lcm,
        qdiv as gmpy_qdiv,
    )
    gcdex = gmpy_gcdex
    gcd = gmpy_gcd
    lcm = gmpy_lcm

elif GROUND_TYPES == 'flint':

    from flint import fmpz as _fmpz

    GMPYInteger = _GMPYInteger
    GMPYRational = _GMPYRational
    gmpy_numer = None
    gmpy_denom = None
    gmpy_gcdex = None
    gmpy_gcd = None
    gmpy_lcm = None
    gmpy_qdiv = None

    def gcd(a, b):
        return a.gcd(b)

    def gcdex(a, b):
        x, y, g = python_gcdex(a, b)
        return _fmpz(x), _fmpz(y), _fmpz(g)

    def lcm(a, b):
        return a.lcm(b)

else:
    GMPYInteger = _GMPYInteger
    GMPYRational = _GMPYRational
    gmpy_numer = None
    gmpy_denom = None
    gmpy_gcdex = None
    gmpy_gcd = None
    gmpy_lcm = None
    gmpy_qdiv = None
    gcdex = python_gcdex
    gcd = python_gcd
    lcm = python_lcm


__all__ = [
    'PythonInteger', 'PythonReal', 'PythonComplex',

    'PythonRational',

    'python_gcdex', 'python_gcd', 'python_lcm',

    'SymPyReal', 'SymPyInteger', 'SymPyRational',

    'GMPYInteger', 'GMPYRational', 'gmpy_numer',
    'gmpy_denom', 'gmpy_gcdex', 'gmpy_gcd', 'gmpy_lcm',
    'gmpy_qdiv',

    'factorial', 'sqrt', 'is_square', 'sqrtrem',

    'GMPYInteger', 'GMPYRational',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\series_class.py ===
"""
Contains the base class for series
Made using sequences in mind
"""

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.cache import cacheit


class SeriesBase(Expr):
    """Base Class for series"""

    @property
    def interval(self):
        """The interval on which the series is defined"""
        raise NotImplementedError("(%s).interval" % self)

    @property
    def start(self):
        """The starting point of the series. This point is included"""
        raise NotImplementedError("(%s).start" % self)

    @property
    def stop(self):
        """The ending point of the series. This point is included"""
        raise NotImplementedError("(%s).stop" % self)

    @property
    def length(self):
        """Length of the series expansion"""
        raise NotImplementedError("(%s).length" % self)

    @property
    def variables(self):
        """Returns a tuple of variables that are bounded"""
        return ()

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).
        """
        return ({j for i in self.args for j in i.free_symbols}
                .difference(self.variables))

    @cacheit
    def term(self, pt):
        """Term at point pt of a series"""
        if pt < self.start or pt > self.stop:
            raise IndexError("Index %s out of bounds %s" % (pt, self.interval))
        return self._eval_term(pt)

    def _eval_term(self, pt):
        raise NotImplementedError("The _eval_term method should be added to"
                                  "%s to return series term so it is available"
                                  "when 'term' calls it."
                                  % self.func)

    def _ith_point(self, i):
        """
        Returns the i'th point of a series
        If start point is negative infinity, point is returned from the end.
        Assumes the first point to be indexed zero.

        Examples
        ========

        TODO
        """
        if self.start is S.NegativeInfinity:
            initial = self.stop
            step = -1
        else:
            initial = self.start
            step = 1

        return initial + i*step

    def __iter__(self):
        i = 0
        while i < self.length:
            pt = self._ith_point(i)
            yield self.term(pt)
            i += 1

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.term(index)
        elif isinstance(index, slice):
            start, stop = index.start, index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            return [self.term(self._ith_point(i)) for i in
                    range(start, stop, index.step or 1)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\sampling\sample_pymc.py ===
from functools import singledispatch
from sympy.external import import_module
from sympy.stats.crv_types import BetaDistribution, CauchyDistribution, ChiSquaredDistribution, ExponentialDistribution, \
    GammaDistribution, LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, \
    GaussianInverseDistribution
from sympy.stats.drv_types import PoissonDistribution, GeometricDistribution, NegativeBinomialDistribution
from sympy.stats.frv_types import BinomialDistribution, BernoulliDistribution


try:
    import pymc
except ImportError:
    pymc = import_module('pymc3')

@singledispatch
def do_sample_pymc(dist):
    return None


# CRV:

@do_sample_pymc.register(BetaDistribution)
def _(dist: BetaDistribution):
    return pymc.Beta('X', alpha=float(dist.alpha), beta=float(dist.beta))


@do_sample_pymc.register(CauchyDistribution)
def _(dist: CauchyDistribution):
    return pymc.Cauchy('X', alpha=float(dist.x0), beta=float(dist.gamma))


@do_sample_pymc.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution):
    return pymc.ChiSquared('X', nu=float(dist.k))


@do_sample_pymc.register(ExponentialDistribution)
def _(dist: ExponentialDistribution):
    return pymc.Exponential('X', lam=float(dist.rate))


@do_sample_pymc.register(GammaDistribution)
def _(dist: GammaDistribution):
    return pymc.Gamma('X', alpha=float(dist.k), beta=1 / float(dist.theta))


@do_sample_pymc.register(LogNormalDistribution)
def _(dist: LogNormalDistribution):
    return pymc.Lognormal('X', mu=float(dist.mean), sigma=float(dist.std))


@do_sample_pymc.register(NormalDistribution)
def _(dist: NormalDistribution):
    return pymc.Normal('X', float(dist.mean), float(dist.std))


@do_sample_pymc.register(GaussianInverseDistribution)
def _(dist: GaussianInverseDistribution):
    return pymc.Wald('X', mu=float(dist.mean), lam=float(dist.shape))


@do_sample_pymc.register(ParetoDistribution)
def _(dist: ParetoDistribution):
    return pymc.Pareto('X', alpha=float(dist.alpha), m=float(dist.xm))


@do_sample_pymc.register(UniformDistribution)
def _(dist: UniformDistribution):
    return pymc.Uniform('X', lower=float(dist.left), upper=float(dist.right))


# DRV:

@do_sample_pymc.register(GeometricDistribution)
def _(dist: GeometricDistribution):
    return pymc.Geometric('X', p=float(dist.p))


@do_sample_pymc.register(NegativeBinomialDistribution)
def _(dist: NegativeBinomialDistribution):
    return pymc.NegativeBinomial('X', mu=float((dist.p * dist.r) / (1 - dist.p)),
                                  alpha=float(dist.r))


@do_sample_pymc.register(PoissonDistribution)
def _(dist: PoissonDistribution):
    return pymc.Poisson('X', mu=float(dist.lamda))


# FRV:

@do_sample_pymc.register(BernoulliDistribution)
def _(dist: BernoulliDistribution):
    return pymc.Bernoulli('X', p=float(dist.p))


@do_sample_pymc.register(BinomialDistribution)
def _(dist: BinomialDistribution):
    return pymc.Binomial('X', n=int(dist.n), p=float(dist.p))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\_warnings.py ===
"""Define some custom warnings."""


class GuessedAtParserWarning(UserWarning):
    """The warning issued when BeautifulSoup has to guess what parser to
    use -- probably because no parser was specified in the constructor.
    """

    MESSAGE: str = """No parser was explicitly specified, so I'm using the best available %(markup_type)s parser for this system ("%(parser)s"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.

The code that caused this warning is on line %(line_number)s of the file %(filename)s. To get rid of this warning, pass the additional argument 'features="%(parser)s"' to the BeautifulSoup constructor.
"""


class UnusualUsageWarning(UserWarning):
    """A superclass for warnings issued when Beautiful Soup sees
    something that is typically the result of a mistake in the calling
    code, but might be intentional on the part of the user. If it is
    in fact intentional, you can filter the individual warning class
    to get rid of the warning. If you don't like Beautiful Soup
    second-guessing what you are doing, you can filter the
    UnusualUsageWarningclass itself and get rid of these entirely.
    """


class MarkupResemblesLocatorWarning(UnusualUsageWarning):
    """The warning issued when BeautifulSoup is given 'markup' that
    actually looks like a resource locator -- a URL or a path to a file
    on disk.
    """

    #: :meta private:
    GENERIC_MESSAGE: str = """

However, if you want to parse some data that happens to look like a %(what)s, then nothing has gone wrong: you are using Beautiful Soup correctly, and this warning is spurious and can be filtered. To make this warning go away, run this code before calling the BeautifulSoup constructor:

    from bs4 import MarkupResemblesLocatorWarning
    import warnings

    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    """

    URL_MESSAGE: str = (
        """The input passed in on this line looks more like a URL than HTML or XML.

If you meant to use Beautiful Soup to parse the web page found at a certain URL, then something has gone wrong. You should use an Python package like 'requests' to fetch the content behind the URL. Once you have the content as a string, you can feed that string into Beautiful Soup."""
        + GENERIC_MESSAGE
    )

    FILENAME_MESSAGE: str = (
        """The input passed in on this line looks more like a filename than HTML or XML.

If you meant to use Beautiful Soup to parse the contents of a file on disk, then something has gone wrong. You should open the file first, using code like this:

    filehandle = open(your filename)

You can then feed the open filehandle into Beautiful Soup instead of using the filename."""
        + GENERIC_MESSAGE
    )


class AttributeResemblesVariableWarning(UnusualUsageWarning, SyntaxWarning):
    """The warning issued when Beautiful Soup suspects a provided
    attribute name may actually be the misspelled name of a Beautiful
    Soup variable. Generally speaking, this is only used in cases like
    "_class" where it's very unlikely the user would be referencing an
    XML attribute with that name.
    """

    MESSAGE: str = """%(original)r is an unusual attribute name and is a common misspelling for %(autocorrect)r.

If you meant %(autocorrect)r, change your code to use it, and this warning will go away.

If you really did mean to check the %(original)r attribute, this warning is spurious and can be filtered. To make it go away, run this code before creating your BeautifulSoup object:

    from bs4 import AttributeResemblesVariableWarning
    import warnings

    warnings.filterwarnings("ignore", category=AttributeResemblesVariableWarning)
"""


class XMLParsedAsHTMLWarning(UnusualUsageWarning):
    """The warning issued when an HTML parser is used to parse
    XML that is not (as far as we can tell) XHTML.
    """

    MESSAGE: str = """It looks like you're using an HTML parser to parse an XML document.

Assuming this really is an XML document, what you're doing might work, but you should know that using an XML parser will be more reliable. To parse this document as XML, make sure you have the Python package 'lxml' installed, and pass the keyword argument `features="xml"` into the BeautifulSoup constructor.

If you want or need to use an HTML parser on this document, you can make this warning go away by filtering it. To do that, run this code before calling the BeautifulSoup constructor:

    from bs4 import XMLParsedAsHTMLWarning
    import warnings

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\linalg\__init__.py ===
"""
``numpy.linalg``
================

The NumPy linear algebra functions rely on BLAS and LAPACK to provide efficient
low level implementations of standard linear algebra algorithms. Those
libraries may be provided by NumPy itself using C versions of a subset of their
reference implementations but, when possible, highly optimized libraries that
take advantage of specialized processor functionality are preferred. Examples
of such libraries are OpenBLAS, MKL (TM), and ATLAS. Because those libraries
are multithreaded and processor dependent, environmental variables and external
packages such as threadpoolctl may be needed to control the number of threads
or specify the processor architecture.

- OpenBLAS: https://www.openblas.net/
- threadpoolctl: https://github.com/joblib/threadpoolctl

Please note that the most-used linear algebra functions in NumPy are present in
the main ``numpy`` namespace rather than in ``numpy.linalg``.  There are:
``dot``, ``vdot``, ``inner``, ``outer``, ``matmul``, ``tensordot``, ``einsum``,
``einsum_path`` and ``kron``.

Functions present in numpy.linalg are listed below.


Matrix and vector products
--------------------------

   cross
   multi_dot
   matrix_power
   tensordot
   matmul

Decompositions
--------------

   cholesky
   outer
   qr
   svd
   svdvals

Matrix eigenvalues
------------------

   eig
   eigh
   eigvals
   eigvalsh

Norms and other numbers
-----------------------

   norm
   matrix_norm
   vector_norm
   cond
   det
   matrix_rank
   slogdet
   trace (Array API compatible)

Solving equations and inverting matrices
----------------------------------------

   solve
   tensorsolve
   lstsq
   inv
   pinv
   tensorinv

Other matrix operations
-----------------------

   diagonal (Array API compatible)
   matrix_transpose (Array API compatible)

Exceptions
----------

   LinAlgError

"""
# To get sub-modules
from . import (
    _linalg,
    linalg,  # deprecated in NumPy 2.0
)
from ._linalg import *

__all__ = _linalg.__all__.copy()  # noqa: PLE0605

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\bart\export.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import logging
import os
import sys

from utils import (
    chain_enc_dec_with_beamsearch,
    export_summarization_edinit,
    export_summarization_enc_dec_past,
    onnx_inference,
)

# GLOBAL ENVS
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("generate")


def print_args(args):
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")


def user_command():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--max_length", type=int, default=20, help="default to 20")
    parent_parser.add_argument("--min_length", type=int, default=0, help="default to 0")
    parent_parser.add_argument("-o", "--output", type=str, default="onnx_models", help="default name is onnx_models.")
    parent_parser.add_argument("-i", "--input_text", type=str, default=None, help="input text")
    parent_parser.add_argument("-s", "--spm_path", type=str, default=None, help="tokenizer model from sentencepice")
    parent_parser.add_argument("-v", "--vocab_path", type=str, help="vocab dictionary")
    parent_parser.add_argument("-b", "--num_beams", type=int, default=5, help="default to 5")
    parent_parser.add_argument("--repetition_penalty", type=float, default=1.0, help="default to 1.0")
    parent_parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="default to 3")
    parent_parser.add_argument("--early_stopping", type=bool, default=False, help="default to False")
    parent_parser.add_argument("--opset_version", type=int, default=14, help="minimum is 14")

    parent_parser.add_argument("--no_encoder", action="store_true")
    parent_parser.add_argument("--no_decoder", action="store_true")
    parent_parser.add_argument("--no_chain", action="store_true")
    parent_parser.add_argument("--no_inference", action="store_true")

    required_args = parent_parser.add_argument_group("required input arguments")
    required_args.add_argument(
        "-m",
        "--model_dir",
        type=str,
        required=True,
        help="The directory contains input huggingface model. \
                               An official model like facebook/bart-base is also acceptable.",
    )

    print_args(parent_parser.parse_args())
    return parent_parser.parse_args()


if __name__ == "__main__":
    args = user_command()
    if args.opset_version < 14:
        raise ValueError(f"The minimum supported opset version is 14! The given one was {args.opset_version}.")

    isExist = os.path.exists(args.output)  # noqa: N816
    if not isExist:
        os.makedirs(args.output)

    # beam search op only supports CPU for now
    args.device = "cpu"
    logger.info("ENV: CPU ...")

    if not args.input_text:
        args.input_text = (
            "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
            "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
            "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
        )

    if not args.no_encoder:
        logger.info("========== EXPORTING ENCODER ==========")
        export_summarization_edinit.export_encoder(args)
    if not args.no_decoder:
        logger.info("========== EXPORTING DECODER ==========")
        export_summarization_enc_dec_past.export_decoder(args)
    if not args.no_chain:
        logger.info("========== CONVERTING MODELS ==========")
        chain_enc_dec_with_beamsearch.convert_model(args)
    if not args.no_inference:
        logger.info("========== INFERENCING WITH ONNX MODEL ==========")
        onnx_inference.run_inference(args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\trendline.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    String,
    Alias
)
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.nested import (
    NestedBool,
    NestedInteger,
    NestedFloat,
    NestedSet
)

from .data_source import NumFmt
from .shapes import GraphicalProperties
from .text import RichText, Text
from .layout import Layout


class TrendlineLabel(Serialisable):

    tagname = "trendlineLbl"

    layout = Typed(expected_type=Layout, allow_none=True)
    tx = Typed(expected_type=Text, allow_none=True)
    numFmt = Typed(expected_type=NumFmt, allow_none=True)
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias("spPr")
    txPr = Typed(expected_type=RichText, allow_none=True)
    textProperties = Alias("txPr")
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('layout', 'tx', 'numFmt', 'spPr', 'txPr')

    def __init__(self,
                 layout=None,
                 tx=None,
                 numFmt=None,
                 spPr=None,
                 txPr=None,
                 extLst=None,
                ):
        self.layout = layout
        self.tx = tx
        self.numFmt = numFmt
        self.spPr = spPr
        self.txPr = txPr


class Trendline(Serialisable):

    tagname = "trendline"

    name = String(allow_none=True)
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias('spPr')
    trendlineType = NestedSet(values=(['exp', 'linear', 'log', 'movingAvg', 'poly', 'power']))
    order = NestedInteger(allow_none=True)
    period = NestedInteger(allow_none=True)
    forward = NestedFloat(allow_none=True)
    backward = NestedFloat(allow_none=True)
    intercept = NestedFloat(allow_none=True)
    dispRSqr = NestedBool(allow_none=True)
    dispEq = NestedBool(allow_none=True)
    trendlineLbl = Typed(expected_type=TrendlineLabel, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('spPr', 'trendlineType', 'order', 'period', 'forward',
                    'backward', 'intercept', 'dispRSqr', 'dispEq', 'trendlineLbl')

    def __init__(self,
                 name=None,
                 spPr=None,
                 trendlineType='linear',
                 order=None,
                 period=None,
                 forward=None,
                 backward=None,
                 intercept=None,
                 dispRSqr=None,
                 dispEq=None,
                 trendlineLbl=None,
                 extLst=None,
                ):
        self.name = name
        self.spPr = spPr
        self.trendlineType = trendlineType
        self.order = order
        self.period = period
        self.forward = forward
        self.backward = backward
        self.intercept = intercept
        self.dispRSqr = dispRSqr
        self.dispEq = dispEq
        self.trendlineLbl = trendlineLbl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\workbook\web.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Sequence,
    String,
    Float,
    Integer,
    Bool,
    NoneSet,
)


class WebPublishObject(Serialisable):

    tagname = "webPublishingObject"

    id = Integer()
    divId = String()
    sourceObject = String(allow_none=True)
    destinationFile = String()
    title = String(allow_none=True)
    autoRepublish = Bool(allow_none=True)

    def __init__(self,
                 id=None,
                 divId=None,
                 sourceObject=None,
                 destinationFile=None,
                 title=None,
                 autoRepublish=None,
                ):
        self.id = id
        self.divId = divId
        self.sourceObject = sourceObject
        self.destinationFile = destinationFile
        self.title = title
        self.autoRepublish = autoRepublish


class WebPublishObjectList(Serialisable):

    tagname ="webPublishingObjects"

    count = Integer(allow_none=True)
    webPublishObject = Sequence(expected_type=WebPublishObject)

    __elements__ = ('webPublishObject',)

    def __init__(self,
                 count=None,
                 webPublishObject=(),
                ):
        self.webPublishObject = webPublishObject


    @property
    def count(self):
        return len(self.webPublishObject)


class WebPublishing(Serialisable):

    tagname = "webPublishing"

    css = Bool(allow_none=True)
    thicket = Bool(allow_none=True)
    longFileNames = Bool(allow_none=True)
    vml = Bool(allow_none=True)
    allowPng = Bool(allow_none=True)
    targetScreenSize = NoneSet(values=(['544x376', '640x480', '720x512', '800x600',
                                    '1024x768', '1152x882', '1152x900', '1280x1024', '1600x1200',
                                    '1800x1440', '1920x1200']))
    dpi = Integer(allow_none=True)
    codePage = Integer(allow_none=True)
    characterSet = String(allow_none=True)

    def __init__(self,
                 css=None,
                 thicket=None,
                 longFileNames=None,
                 vml=None,
                 allowPng=None,
                 targetScreenSize='800x600',
                 dpi=None,
                 codePage=None,
                 characterSet=None,
                ):
        self.css = css
        self.thicket = thicket
        self.longFileNames = longFileNames
        self.vml = vml
        self.allowPng = allowPng
        self.targetScreenSize = targetScreenSize
        self.dpi = dpi
        self.codePage = codePage
        self.characterSet = characterSet

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\util\numba_.py ===
"""Common utilities for Numba operations"""
from __future__ import annotations

import types
from typing import (
    TYPE_CHECKING,
    Callable,
)

import numpy as np

from pandas.compat._optional import import_optional_dependency
from pandas.errors import NumbaUtilError

GLOBAL_USE_NUMBA: bool = False


def maybe_use_numba(engine: str | None) -> bool:
    """Signal whether to use numba routines."""
    return engine == "numba" or (engine is None and GLOBAL_USE_NUMBA)


def set_use_numba(enable: bool = False) -> None:
    global GLOBAL_USE_NUMBA
    if enable:
        import_optional_dependency("numba")
    GLOBAL_USE_NUMBA = enable


def get_jit_arguments(
    engine_kwargs: dict[str, bool] | None = None, kwargs: dict | None = None
) -> dict[str, bool]:
    """
    Return arguments to pass to numba.JIT, falling back on pandas default JIT settings.

    Parameters
    ----------
    engine_kwargs : dict, default None
        user passed keyword arguments for numba.JIT
    kwargs : dict, default None
        user passed keyword arguments to pass into the JITed function

    Returns
    -------
    dict[str, bool]
        nopython, nogil, parallel

    Raises
    ------
    NumbaUtilError
    """
    if engine_kwargs is None:
        engine_kwargs = {}

    nopython = engine_kwargs.get("nopython", True)
    if kwargs and nopython:
        raise NumbaUtilError(
            "numba does not support kwargs with nopython=True: "
            "https://github.com/numba/numba/issues/2916"
        )
    nogil = engine_kwargs.get("nogil", False)
    parallel = engine_kwargs.get("parallel", False)
    return {"nopython": nopython, "nogil": nogil, "parallel": parallel}


def jit_user_function(func: Callable) -> Callable:
    """
    If user function is not jitted already, mark the user's function
    as jitable.

    Parameters
    ----------
    func : function
        user defined function

    Returns
    -------
    function
        Numba JITed function, or function marked as JITable by numba
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    if numba.extending.is_jitted(func):
        # Don't jit a user passed jitted function
        numba_func = func
    elif getattr(np, func.__name__, False) is func or isinstance(
        func, types.BuiltinFunctionType
    ):
        # Not necessary to jit builtins or np functions
        # This will mess up register_jitable
        numba_func = func
    else:
        numba_func = numba.extending.register_jitable(func)

    return numba_func

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\__init__.py ===
"""
Plotting public API.

Authors of third-party plotting backends should implement a module with a
public ``plot(data, kind, **kwargs)``. The parameter `data` will contain
the data structure and can be a `Series` or a `DataFrame`. For example,
for ``df.plot()`` the parameter `data` will contain the DataFrame `df`.
In some cases, the data structure is transformed before being sent to
the backend (see PlotAccessor.__call__ in pandas/plotting/_core.py for
the exact transformations).

The parameter `kind` will be one of:

- line
- bar
- barh
- box
- hist
- kde
- area
- pie
- scatter
- hexbin

See the pandas API reference for documentation on each kind of plot.

Any other keyword argument is currently assumed to be backend specific,
but some parameters may be unified and added to the signature in the
future (e.g. `title` which should be useful for any backend).

Currently, all the Matplotlib functions in pandas are accessed through
the selected backend. For example, `pandas.plotting.boxplot` (equivalent
to `DataFrame.boxplot`) is also accessed in the selected backend. This
is expected to change, and the exact API is under discussion. But with
the current version, backends are expected to implement the next functions:

- plot (describe above, used for `Series.plot` and `DataFrame.plot`)
- hist_series and hist_frame (for `Series.hist` and `DataFrame.hist`)
- boxplot (`pandas.plotting.boxplot(df)` equivalent to `DataFrame.boxplot`)
- boxplot_frame and boxplot_frame_groupby
- register and deregister (register converters for the tick formats)
- Plots not called as `Series` and `DataFrame` methods:
  - table
  - andrews_curves
  - autocorrelation_plot
  - bootstrap_plot
  - lag_plot
  - parallel_coordinates
  - radviz
  - scatter_matrix

Use the code in pandas/plotting/_matplotib.py and
https://github.com/pyviz/hvplot as a reference on how to write a backend.

For the discussion about the API see
https://github.com/pandas-dev/pandas/issues/26747.
"""
from pandas.plotting._core import (
    PlotAccessor,
    boxplot,
    boxplot_frame,
    boxplot_frame_groupby,
    hist_frame,
    hist_series,
)
from pandas.plotting._misc import (
    andrews_curves,
    autocorrelation_plot,
    bootstrap_plot,
    deregister as deregister_matplotlib_converters,
    lag_plot,
    parallel_coordinates,
    plot_params,
    radviz,
    register as register_matplotlib_converters,
    scatter_matrix,
    table,
)

__all__ = [
    "PlotAccessor",
    "boxplot",
    "boxplot_frame",
    "boxplot_frame_groupby",
    "hist_frame",
    "hist_series",
    "scatter_matrix",
    "radviz",
    "andrews_curves",
    "bootstrap_plot",
    "parallel_coordinates",
    "lag_plot",
    "autocorrelation_plot",
    "table",
    "plot_params",
    "register_matplotlib_converters",
    "deregister_matplotlib_converters",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\network\utils.py ===
from typing import Dict, Generator

from pip._vendor.requests.models import Response

from pip._internal.exceptions import NetworkConnectionError

# The following comments and HTTP headers were originally added by
# Donald Stufft in git commit 22c562429a61bb77172039e480873fb239dd8c03.
#
# We use Accept-Encoding: identity here because requests defaults to
# accepting compressed responses. This breaks in a variety of ways
# depending on how the server is configured.
# - Some servers will notice that the file isn't a compressible file
#   and will leave the file alone and with an empty Content-Encoding
# - Some servers will notice that the file is already compressed and
#   will leave the file alone, adding a Content-Encoding: gzip header
# - Some servers won't notice anything at all and will take a file
#   that's already been compressed and compress it again, and set
#   the Content-Encoding: gzip header
# By setting this to request only the identity encoding we're hoping
# to eliminate the third case.  Hopefully there does not exist a server
# which when given a file will notice it is already compressed and that
# you're not asking for a compressed file and will then decompress it
# before sending because if that's the case I don't think it'll ever be
# possible to make this work.
HEADERS: Dict[str, str] = {"Accept-Encoding": "identity"}

DOWNLOAD_CHUNK_SIZE = 256 * 1024


def raise_for_status(resp: Response) -> None:
    http_error_msg = ""
    if isinstance(resp.reason, bytes):
        # We attempt to decode utf-8 first because some servers
        # choose to localize their reason strings. If the string
        # isn't utf-8, we fall back to iso-8859-1 for all other
        # encodings.
        try:
            reason = resp.reason.decode("utf-8")
        except UnicodeDecodeError:
            reason = resp.reason.decode("iso-8859-1")
    else:
        reason = resp.reason

    if 400 <= resp.status_code < 500:
        http_error_msg = (
            f"{resp.status_code} Client Error: {reason} for url: {resp.url}"
        )

    elif 500 <= resp.status_code < 600:
        http_error_msg = (
            f"{resp.status_code} Server Error: {reason} for url: {resp.url}"
        )

    if http_error_msg:
        raise NetworkConnectionError(http_error_msg, response=resp)


def response_chunks(
    response: Response, chunk_size: int = DOWNLOAD_CHUNK_SIZE
) -> Generator[bytes, None, None]:
    """Given a requests Response, provide the data chunks."""
    try:
        # Special case for urllib3.
        for chunk in response.raw.stream(
            chunk_size,
            # We use decode_content=False here because we don't
            # want urllib3 to mess with the raw bytes we get
            # from the server. If we decompress inside of
            # urllib3 then we cannot verify the checksum
            # because the checksum will be of the compressed
            # file. This breakage will only occur if the
            # server adds a Content-Encoding header, which
            # depends on how the server was configured:
            # - Some servers will notice that the file isn't a
            #   compressible file and will leave the file alone
            #   and with an empty Content-Encoding
            # - Some servers will notice that the file is
            #   already compressed and will leave the file
            #   alone and will add a Content-Encoding: gzip
            #   header
            # - Some servers won't notice anything at all and
            #   will take a file that's already been compressed
            #   and compress it again and set the
            #   Content-Encoding: gzip header
            #
            # By setting this not to decode automatically we
            # hope to eliminate problems with the second case.
            decode_content=False,
        ):
            yield chunk
    except AttributeError:
        # Standard file-like object.
        while True:
            chunk = response.raw.read(chunk_size)
            if not chunk:
                break
            yield chunk

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\filepost.py ===
from __future__ import absolute_import

import binascii
import codecs
import os
from io import BytesIO

from .fields import RequestField
from .packages import six
from .packages.six import b

writer = codecs.lookup("utf-8")[3]


def choose_boundary():
    """
    Our embarrassingly-simple replacement for mimetools.choose_boundary.
    """
    boundary = binascii.hexlify(os.urandom(16))
    if not six.PY2:
        boundary = boundary.decode("ascii")
    return boundary


def iter_field_objects(fields):
    """
    Iterate over fields.

    Supports list of (k, v) tuples and dicts, and lists of
    :class:`~urllib3.fields.RequestField`.

    """
    if isinstance(fields, dict):
        i = six.iteritems(fields)
    else:
        i = iter(fields)

    for field in i:
        if isinstance(field, RequestField):
            yield field
        else:
            yield RequestField.from_tuples(*field)


def iter_fields(fields):
    """
    .. deprecated:: 1.6

    Iterate over fields.

    The addition of :class:`~urllib3.fields.RequestField` makes this function
    obsolete. Instead, use :func:`iter_field_objects`, which returns
    :class:`~urllib3.fields.RequestField` objects.

    Supports list of (k, v) tuples and dicts.
    """
    if isinstance(fields, dict):
        return ((k, v) for k, v in six.iteritems(fields))

    return ((k, v) for k, v in fields)


def encode_multipart_formdata(fields, boundary=None):
    """
    Encode a dictionary of ``fields`` using the multipart/form-data MIME format.

    :param fields:
        Dictionary of fields or list of (key, :class:`~urllib3.fields.RequestField`).

    :param boundary:
        If not specified, then a random boundary will be generated using
        :func:`urllib3.filepost.choose_boundary`.
    """
    body = BytesIO()
    if boundary is None:
        boundary = choose_boundary()

    for field in iter_field_objects(fields):
        body.write(b("--%s\r\n" % (boundary)))

        writer(body).write(field.render_headers())
        data = field.data

        if isinstance(data, int):
            data = str(data)  # Backwards compatibility

        if isinstance(data, six.text_type):
            writer(body).write(data)
        else:
            body.write(data)

        body.write(b"\r\n")

    body.write(b("--%s--\r\n" % (boundary)))

    content_type = str("multipart/form-data; boundary=%s" % boundary)

    return body.getvalue(), content_type

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\holonomic\numerical.py ===
"""Numerical Methods for Holonomic Functions"""

from sympy.core.sympify import sympify
from sympy.holonomic.holonomic import DMFsubs

from mpmath import mp


def _evalf(func, points, derivatives=False, method='RK4'):
    """
    Numerical methods for numerical integration along a given set of
    points in the complex plane.
    """

    ann = func.annihilator
    a = ann.order
    R = ann.parent.base
    K = R.get_field()

    if method == 'Euler':
        meth = _euler
    else:
        meth = _rk4

    dmf = [K.new(j.to_list()) for j in ann.listofpoly]
    red = [-dmf[i] / dmf[a] for i in range(a)]

    y0 = func.y0
    if len(y0) < a:
        raise TypeError("Not Enough Initial Conditions")
    x0 = func.x0
    sol = [meth(red, x0, points[0], y0, a)]

    for i, j in enumerate(points[1:]):
        sol.append(meth(red, points[i], j, sol[-1], a))

    if not derivatives:
        return [sympify(i[0]) for i in sol]
    else:
        return sympify(sol)


def _euler(red, x0, x1, y0, a):
    """
    Euler's method for numerical integration.
    From x0 to x1 with initial values given at x0 as vector y0.
    """

    A = sympify(x0)._to_mpmath(mp.prec)
    B = sympify(x1)._to_mpmath(mp.prec)
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]
    h = B - A
    f_0 = y_0[1:]
    f_0_n = 0

    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)

    return [y_0[i] + h * f_0[i] for i in range(a)]


def _rk4(red, x0, x1, y0, a):
    """
    Runge-Kutta 4th order numerical method.
    """

    A = sympify(x0)._to_mpmath(mp.prec)
    B = sympify(x1)._to_mpmath(mp.prec)
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]
    h = B - A

    f_0_n = 0
    f_1_n = 0
    f_2_n = 0
    f_3_n = 0

    f_0 = y_0[1:]
    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)

    f_1 = [y_0[i] + f_0[i]*h/2 for i in range(1, a)]
    for i in range(a):
        f_1_n += sympify(DMFsubs(red[i], A + h/2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_0[i]*h/2)
    f_1.append(f_1_n)

    f_2 = [y_0[i] + f_1[i]*h/2 for i in range(1, a)]
    for i in range(a):
        f_2_n += sympify(DMFsubs(red[i], A + h/2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_1[i]*h/2)
    f_2.append(f_2_n)

    f_3 = [y_0[i] + f_2[i]*h for i in range(1, a)]
    for i in range(a):
        f_3_n += sympify(DMFsubs(red[i], A + h, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_2[i]*h)
    f_3.append(f_3_n)

    return [y_0[i] + h*(f_0[i]+2*f_1[i]+2*f_2[i]+f_3[i])/6 for i in range(a)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\pythonintegerring.py ===
"""Implementation of :class:`PythonIntegerRing` class. """


from sympy.core.numbers import int_valued
from sympy.polys.domains.groundtypes import (
    PythonInteger, SymPyInteger, sqrt as python_sqrt,
    factorial as python_factorial, python_gcdex, python_gcd, python_lcm,
)
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class PythonIntegerRing(IntegerRing):
    """Integer ring based on Python's ``int`` type.

    This will be used as :ref:`ZZ` if ``gmpy`` and ``gmpy2`` are not
    installed. Elements are instances of the standard Python ``int`` type.
    """

    dtype = PythonInteger
    zero = dtype(0)
    one = dtype(1)
    alias = 'ZZ_python'

    def __init__(self):
        """Allow instantiation of this domain. """

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(a)

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return PythonInteger(a.p)
        elif int_valued(a):
            return PythonInteger(int(a))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to Python's ``int``. """
        return K0.to_int(a)

    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to Python's ``int``. """
        return a

    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to Python's ``int``. """
        if a.denominator == 1:
            return a.numerator

    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to Python's ``int``. """
        if a.denominator == 1:
            return a.numerator

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to Python's ``int``. """
        return PythonInteger(K0.to_int(a))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpz`` to Python's ``int``. """
        return PythonInteger(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpq`` to Python's ``int``. """
        if a.denom() == 1:
            return PythonInteger(a.numer())

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to Python's ``int``. """
        p, q = K0.to_rational(a)

        if q == 1:
            return PythonInteger(p)

    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        return python_gcdex(a, b)

    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        return python_gcd(a, b)

    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        return python_lcm(a, b)

    def sqrt(self, a):
        """Compute square root of ``a``. """
        return python_sqrt(a)

    def factorial(self, a):
        """Compute factorial of ``a``. """
        return python_factorial(a)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_generated_io_epoll.py ===
# ***********************************************************
# ******* WARNING: AUTOGENERATED! ALL EDITS WILL BE LOST ******
# *************************************************************
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ._ki import enable_ki_protection
from ._run import GLOBAL_RUN_CONTEXT

if TYPE_CHECKING:
    from .._file_io import _HasFileNo

assert not TYPE_CHECKING or sys.platform == "linux"


__all__ = ["notify_closing", "wait_readable", "wait_writable"]


@enable_ki_protection
async def wait_readable(fd: int | _HasFileNo) -> None:
    """Block until the kernel reports that the given object is readable.

    On Unix systems, ``fd`` must either be an integer file descriptor,
    or else an object with a ``.fileno()`` method which returns an
    integer file descriptor. Any kind of file descriptor can be passed,
    though the exact semantics will depend on your kernel. For example,
    this probably won't do anything useful for on-disk files.

    On Windows systems, ``fd`` must either be an integer ``SOCKET``
    handle, or else an object with a ``.fileno()`` method which returns
    an integer ``SOCKET`` handle. File descriptors aren't supported,
    and neither are handles that refer to anything besides a
    ``SOCKET``.

    :raises trio.BusyResourceError:
        if another task is already waiting for the given socket to
        become readable.
    :raises trio.ClosedResourceError:
        if another task calls :func:`notify_closing` while this
        function is still working.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_readable(fd)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
async def wait_writable(fd: int | _HasFileNo) -> None:
    """Block until the kernel reports that the given object is writable.

    See `wait_readable` for the definition of ``fd``.

    :raises trio.BusyResourceError:
        if another task is already waiting for the given socket to
        become writable.
    :raises trio.ClosedResourceError:
        if another task calls :func:`notify_closing` while this
        function is still working.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_writable(fd)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def notify_closing(fd: int | _HasFileNo) -> None:
    """Notify waiters of the given object that it will be closed.

    Call this before closing a file descriptor (on Unix) or socket (on
    Windows). This will cause any `wait_readable` or `wait_writable`
    calls on the given object to immediately wake up and raise
    `~trio.ClosedResourceError`.

    This doesn't actually close the object – you still have to do that
    yourself afterwards. Also, you want to be careful to make sure no
    new tasks start waiting on the object in between when you call this
    and when it's actually closed. So to close something properly, you
    usually want to do these steps in order:

    1. Explicitly mark the object as closed, so that any new attempts
       to use it will abort before they start.
    2. Call `notify_closing` to wake up any already-existing users.
    3. Actually close the object.

    It's also possible to do them in a different order if that's more
    convenient, *but only if* you make sure not to have any checkpoints in
    between the steps. This way they all happen in a single atomic
    step, so other tasks won't be able to tell what order they happened
    in anyway.
    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.io_manager.notify_closing(fd)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_highlevel_generic.py ===
from __future__ import annotations

from typing import NoReturn

import attrs
import pytest

from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream


@attrs.define(slots=False)
class RecordSendStream(SendStream):
    record: list[str | tuple[str, object]] = attrs.Factory(list)

    async def send_all(self, data: object) -> None:
        self.record.append(("send_all", data))

    async def wait_send_all_might_not_block(self) -> None:
        self.record.append("wait_send_all_might_not_block")

    async def aclose(self) -> None:
        self.record.append("aclose")


@attrs.define(slots=False)
class RecordReceiveStream(ReceiveStream):
    record: list[str | tuple[str, int | None]] = attrs.Factory(list)

    async def receive_some(self, max_bytes: int | None = None) -> bytes:
        self.record.append(("receive_some", max_bytes))
        return b""

    async def aclose(self) -> None:
        self.record.append("aclose")


async def test_StapledStream() -> None:
    send_stream = RecordSendStream()
    receive_stream = RecordReceiveStream()
    stapled = StapledStream(send_stream, receive_stream)

    assert stapled.send_stream is send_stream
    assert stapled.receive_stream is receive_stream

    await stapled.send_all(b"foo")
    await stapled.wait_send_all_might_not_block()
    assert send_stream.record == [
        ("send_all", b"foo"),
        "wait_send_all_might_not_block",
    ]
    send_stream.record.clear()

    await stapled.send_eof()
    assert send_stream.record == ["aclose"]
    send_stream.record.clear()

    async def fake_send_eof() -> None:
        send_stream.record.append("send_eof")

    send_stream.send_eof = fake_send_eof  # type: ignore[attr-defined]
    await stapled.send_eof()
    assert send_stream.record == ["send_eof"]

    send_stream.record.clear()
    assert receive_stream.record == []

    await stapled.receive_some(1234)
    assert receive_stream.record == [("receive_some", 1234)]
    assert send_stream.record == []
    receive_stream.record.clear()

    await stapled.aclose()
    assert receive_stream.record == ["aclose"]
    assert send_stream.record == ["aclose"]


async def test_StapledStream_with_erroring_close() -> None:
    # Make sure that if one of the aclose methods errors out, then the other
    # one still gets called.
    class BrokenSendStream(RecordSendStream):
        async def aclose(self) -> NoReturn:
            await super().aclose()
            raise ValueError("send error")

    class BrokenReceiveStream(RecordReceiveStream):
        async def aclose(self) -> NoReturn:
            await super().aclose()
            raise ValueError("recv error")

    stapled = StapledStream(BrokenSendStream(), BrokenReceiveStream())

    with pytest.raises(ValueError, match=r"^(send|recv) error$") as excinfo:
        await stapled.aclose()
    assert isinstance(excinfo.value.__context__, ValueError)

    assert stapled.send_stream.record == ["aclose"]
    assert stapled.receive_stream.record == ["aclose"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\fields\form.py ===
from wtforms.utils import unset_value

from .. import widgets
from .core import Field

__all__ = ("FormField",)


class FormField(Field):
    """
    Encapsulate a form as a field in another form.

    :param form_class:
        A subclass of Form that will be encapsulated.
    :param separator:
        A string which will be suffixed to this field's name to create the
        prefix to enclosed fields. The default is fine for most uses.
    """

    widget = widgets.TableWidget()

    def __init__(
        self, form_class, label=None, validators=None, separator="-", **kwargs
    ):
        super().__init__(label, validators, **kwargs)
        self.form_class = form_class
        self.separator = separator
        self._obj = None
        if self.filters:
            raise TypeError(
                "FormField cannot take filters, as the encapsulated"
                " data is not mutable."
            )
        if validators:
            raise TypeError(
                "FormField does not accept any validators. Instead,"
                " define them on the enclosed form."
            )

    def process(self, formdata, data=unset_value, extra_filters=None):
        if extra_filters:
            raise TypeError(
                "FormField cannot take filters, as the encapsulated"
                "data is not mutable."
            )

        if data is unset_value:
            try:
                data = self.default()
            except TypeError:
                data = self.default
            self._obj = data

        self.object_data = data

        prefix = self.name + self.separator
        if isinstance(data, dict):
            self.form = self.form_class(formdata=formdata, prefix=prefix, **data)
        else:
            self.form = self.form_class(formdata=formdata, obj=data, prefix=prefix)

    def validate(self, form, extra_validators=()):
        if extra_validators:
            raise TypeError(
                "FormField does not accept in-line validators, as it"
                " gets errors from the enclosed form."
            )
        return self.form.validate()

    def populate_obj(self, obj, name):
        candidate = getattr(obj, name, None)
        if candidate is None:
            if self._obj is None:
                raise TypeError(
                    "populate_obj: cannot find a value to populate from"
                    " the provided obj or input data/defaults"
                )
            candidate = self._obj

        self.form.populate_obj(candidate)
        setattr(obj, name, candidate)

    def __iter__(self):
        return iter(self.form)

    def __getitem__(self, name):
        return self.form[name]

    def __getattr__(self, name):
        return getattr(self.form, name)

    @property
    def data(self):
        return self.form.data

    @property
    def errors(self):
        return self.form.errors

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_latex.py ===
from sympy import MatAdd, MatMul, Array
from sympy.algebras.quaternion import Quaternion
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.combinatorics.permutations import Cycle, Permutation, AppliedPermutation
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple, Dict
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import (Derivative, Function, Lambda, Subs, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (AlgebraicNumber, Float, I, Integer, Rational, oo, pi)
from sympy.core.parameters import evaluate
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial, binomial, factorial, factorial2, subfactorial)
from sympy.functions.combinatorial.numbers import (bernoulli, bell, catalan, euler, genocchi,
                                                   lucas, fibonacci, tribonacci, divisor_sigma, udivisor_sigma,
                                                   mobius, primenu, primeomega,
                                                   totient, reduced_totient)
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, polar_lift, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, coth)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (Max, Min, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acsc, asin, cos, cot, sin, tan)
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f, elliptic_k, elliptic_pi)
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, expint)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.functions.special.mathieu_functions import (mathieuc, mathieucprime, mathieus, mathieusprime)
from sympy.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.spherical_harmonics import (Ynm, Znm)
from sympy.functions.special.tensor_functions import (KroneckerDelta, LeviCivita)
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, stieltjes, zeta)
from sympy.integrals.integrals import Integral
from sympy.integrals.transforms import (CosineTransform, FourierTransform, InverseCosineTransform, InverseFourierTransform, InverseLaplaceTransform, InverseMellinTransform, InverseSineTransform, LaplaceTransform, MellinTransform, SineTransform)
from sympy.logic import Implies
from sympy.logic.boolalg import (And, Or, Xor, Equivalent, false, Not, true)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.kronecker import KroneckerProduct
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.permutation import PermutationMatrix
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.dotproduct import DotProduct
from sympy.physics.control.lti import TransferFunction, Series, Parallel, Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback
from sympy.physics.quantum import Commutator, Operator
from sympy.physics.quantum.trace import Tr
from sympy.physics.units import meter, gibibyte, gram, microgram, second, milli, micro
from sympy.polys.domains.integerring import ZZ
from sympy.polys.fields import field
from sympy.polys.polytools import Poly
from sympy.polys.rings import ring
from sympy.polys.rootoftools import (RootSum, rootof)
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ComplexRegion, ImageSet, Range)
from sympy.sets.ordinals import Ordinal, OrdinalOmega, OmegaPower
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import (FiniteSet, Interval, Union, Intersection, Complement, SymmetricDifference, ProductSet)
from sympy.sets.setexpr import SetExpr
from sympy.stats.crv_types import Normal
from sympy.stats.symbolic_probability import (Covariance, Expectation,
                                              Probability, Variance)
from sympy.tensor.array import (ImmutableDenseNDimArray,
                                ImmutableSparseNDimArray,
                                MutableSparseNDimArray,
                                MutableDenseNDimArray,
                                tensorproduct)
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)
from sympy.tensor.toperators import PartialDerivative
from sympy.vector import CoordSys3D, Cross, Curl, Dot, Divergence, Gradient, Laplacian


from sympy.testing.pytest import (XFAIL, raises, _both_exp_pow,
                                  warns_deprecated_sympy)
from sympy.printing.latex import (latex, translate, greek_letters_set,
                                  tex_greek_dictionary, multiline_latex,
                                  latex_escape, LatexPrinter)

import sympy as sym

from sympy.abc import mu, tau


class lowergamma(sym.lowergamma):
    pass   # testing notation inheritance by a subclass with same name


x, y, z, t, w, a, b, c, s, p = symbols('x y z t w a b c s p')
k, m, n = symbols('k m n', integer=True)


def test_printmethod():
    class R(Abs):
        def _latex(self, printer):
            return "foo(%s)" % printer._print(self.args[0])
    assert latex(R(x)) == r"foo(x)"

    class R(Abs):
        def _latex(self, printer):
            return "foo"
    assert latex(R(x)) == r"foo"


def test_latex_basic():
    assert latex(1 + x) == r"x + 1"
    assert latex(x**2) == r"x^{2}"
    assert latex(x**(1 + x)) == r"x^{x + 1}"
    assert latex(x**3 + x + 1 + x**2) == r"x^{3} + x^{2} + x + 1"

    assert latex(2*x*y) == r"2 x y"
    assert latex(2*x*y, mul_symbol='dot') == r"2 \cdot x \cdot y"
    assert latex(3*x**2*y, mul_symbol='\\,') == r"3\,x^{2}\,y"
    assert latex(1.5*3**x, mul_symbol='\\,') == r"1.5 \cdot 3^{x}"

    assert latex(x**S.Half**5) == r"\sqrt[32]{x}"
    assert latex(Mul(S.Half, x**2, -5, evaluate=False)) == r"\frac{1}{2} x^{2} \left(-5\right)"
    assert latex(Mul(S.Half, x**2, 5, evaluate=False)) == r"\frac{1}{2} x^{2} \cdot 5"
    assert latex(Mul(-5, -5, evaluate=False)) == r"\left(-5\right) \left(-5\right)"
    assert latex(Mul(5, -5, evaluate=False)) == r"5 \left(-5\right)"
    assert latex(Mul(S.Half, -5, S.Half, evaluate=False)) == r"\frac{1}{2} \left(-5\right) \frac{1}{2}"
    assert latex(Mul(5, I, 5, evaluate=False)) == r"5 i 5"
    assert latex(Mul(5, I, -5, evaluate=False)) == r"5 i \left(-5\right)"
    assert latex(Mul(Pow(x, 2), S.Half*x + 1)) == r"x^{2} \left(\frac{x}{2} + 1\right)"
    assert latex(Mul(Pow(x, 3), Rational(2, 3)*x + 1)) == r"x^{3} \left(\frac{2 x}{3} + 1\right)"
    assert latex(Mul(Pow(x, 11), 2*x + 1)) == r"x^{11} \left(2 x + 1\right)"

    assert latex(Mul(0, 1, evaluate=False)) == r'0 \cdot 1'
    assert latex(Mul(1, 0, evaluate=False)) == r'1 \cdot 0'
    assert latex(Mul(1, 1, evaluate=False)) == r'1 \cdot 1'
    assert latex(Mul(-1, 1, evaluate=False)) == r'\left(-1\right) 1'
    assert latex(Mul(1, 1, 1, evaluate=False)) == r'1 \cdot 1 \cdot 1'
    assert latex(Mul(1, 2, evaluate=False)) == r'1 \cdot 2'
    assert latex(Mul(1, S.Half, evaluate=False)) == r'1 \cdot \frac{1}{2}'
    assert latex(Mul(1, 1, S.Half, evaluate=False)) == \
        r'1 \cdot 1 \cdot \frac{1}{2}'
    assert latex(Mul(1, 1, 2, 3, x, evaluate=False)) == \
        r'1 \cdot 1 \cdot 2 \cdot 3 x'
    assert latex(Mul(1, -1, evaluate=False)) == r'1 \left(-1\right)'
    assert latex(Mul(4, 3, 2, 1, 0, y, x, evaluate=False)) == \
        r'4 \cdot 3 \cdot 2 \cdot 1 \cdot 0 y x'
    assert latex(Mul(4, 3, 2, 1+z, 0, y, x, evaluate=False)) == \
        r'4 \cdot 3 \cdot 2 \left(z + 1\right) 0 y x'
    assert latex(Mul(Rational(2, 3), Rational(5, 7), evaluate=False)) == \
        r'\frac{2}{3} \cdot \frac{5}{7}'

    assert latex(1/x) == r"\frac{1}{x}"
    assert latex(1/x, fold_short_frac=True) == r"1 / x"
    assert latex(-S(3)/2) == r"- \frac{3}{2}"
    assert latex(-S(3)/2, fold_short_frac=True) == r"- 3 / 2"
    assert latex(1/x**2) == r"\frac{1}{x^{2}}"
    assert latex(1/(x + y)/2) == r"\frac{1}{2 \left(x + y\right)}"
    assert latex(x/2) == r"\frac{x}{2}"
    assert latex(x/2, fold_short_frac=True) == r"x / 2"
    assert latex((x + y)/(2*x)) == r"\frac{x + y}{2 x}"
    assert latex((x + y)/(2*x), fold_short_frac=True) == \
        r"\left(x + y\right) / 2 x"
    assert latex((x + y)/(2*x), long_frac_ratio=0) == \
        r"\frac{1}{2 x} \left(x + y\right)"
    assert latex((x + y)/x) == r"\frac{x + y}{x}"
    assert latex((x + y)/x, long_frac_ratio=3) == r"\frac{x + y}{x}"
    assert latex((2*sqrt(2)*x)/3) == r"\frac{2 \sqrt{2} x}{3}"
    assert latex((2*sqrt(2)*x)/3, long_frac_ratio=2) == \
        r"\frac{2 x}{3} \sqrt{2}"
    assert latex(binomial(x, y)) == r"{\binom{x}{y}}"

    x_star = Symbol('x^*')
    f = Function('f')
    assert latex(x_star**2) == r"\left(x^{*}\right)^{2}"
    assert latex(x_star**2, parenthesize_super=False) == r"{x^{*}}^{2}"
    assert latex(Derivative(f(x_star), x_star,2)) == r"\frac{d^{2}}{d \left(x^{*}\right)^{2}} f{\left(x^{*} \right)}"
    assert latex(Derivative(f(x_star), x_star,2), parenthesize_super=False) == r"\frac{d^{2}}{d {x^{*}}^{2}} f{\left(x^{*} \right)}"

    assert latex(2*Integral(x, x)/3) == r"\frac{2 \int x\, dx}{3}"
    assert latex(2*Integral(x, x)/3, fold_short_frac=True) == \
        r"\left(2 \int x\, dx\right) / 3"

    assert latex(sqrt(x)) == r"\sqrt{x}"
    assert latex(x**Rational(1, 3)) == r"\sqrt[3]{x}"
    assert latex(x**Rational(1, 3), root_notation=False) == r"x^{\frac{1}{3}}"
    assert latex(sqrt(x)**3) == r"x^{\frac{3}{2}}"
    assert latex(sqrt(x), itex=True) == r"\sqrt{x}"
    assert latex(x**Rational(1, 3), itex=True) == r"\root{3}{x}"
    assert latex(sqrt(x)**3, itex=True) == r"x^{\frac{3}{2}}"
    assert latex(x**Rational(3, 4)) == r"x^{\frac{3}{4}}"
    assert latex(x**Rational(3, 4), fold_frac_powers=True) == r"x^{3/4}"
    assert latex((x + 1)**Rational(3, 4)) == \
        r"\left(x + 1\right)^{\frac{3}{4}}"
    assert latex((x + 1)**Rational(3, 4), fold_frac_powers=True) == \
        r"\left(x + 1\right)^{3/4}"
    assert latex(AlgebraicNumber(sqrt(2))) == r"\sqrt{2}"
    assert latex(AlgebraicNumber(sqrt(2), [3, -7])) == r"-7 + 3 \sqrt{2}"
    assert latex(AlgebraicNumber(sqrt(2), alias='alpha')) == r"\alpha"
    assert latex(AlgebraicNumber(sqrt(2), [3, -7], alias='alpha')) == \
        r"3 \alpha - 7"
    assert latex(AlgebraicNumber(2**(S(1)/3), [1, 3, -7], alias='beta')) == \
        r"\beta^{2} + 3 \beta - 7"

    k = ZZ.cyclotomic_field(5)
    assert latex(k.ext.field_element([1, 2, 3, 4])) == \
        r"\zeta^{3} + 2 \zeta^{2} + 3 \zeta + 4"
    assert latex(k.ext.field_element([1, 2, 3, 4]), order='old') == \
        r"4 + 3 \zeta + 2 \zeta^{2} + \zeta^{3}"
    assert latex(k.primes_above(19)[0]) == \
        r"\left(19, \zeta^{2} + 5 \zeta + 1\right)"
    assert latex(k.primes_above(19)[0], order='old') == \
           r"\left(19, 1 + 5 \zeta + \zeta^{2}\right)"
    assert latex(k.primes_above(7)[0]) == r"\left(7\right)"

    assert latex(1.5e20*x) == r"1.5 \cdot 10^{20} x"
    assert latex(1.5e20*x, mul_symbol='dot') == r"1.5 \cdot 10^{20} \cdot x"
    assert latex(1.5e20*x, mul_symbol='times') == \
        r"1.5 \times 10^{20} \times x"

    assert latex(1/sin(x)) == r"\frac{1}{\sin{\left(x \right)}}"
    assert latex(sin(x)**-1) == r"\frac{1}{\sin{\left(x \right)}}"
    assert latex(sin(x)**Rational(3, 2)) == \
        r"\sin^{\frac{3}{2}}{\left(x \right)}"
    assert latex(sin(x)**Rational(3, 2), fold_frac_powers=True) == \
        r"\sin^{3/2}{\left(x \right)}"

    assert latex(~x) == r"\neg x"
    assert latex(x & y) == r"x \wedge y"
    assert latex(x & y & z) == r"x \wedge y \wedge z"
    assert latex(x | y) == r"x \vee y"
    assert latex(x | y | z) == r"x \vee y \vee z"
    assert latex((x & y) | z) == r"z \vee \left(x \wedge y\right)"
    assert latex(Implies(x, y)) == r"x \Rightarrow y"
    assert latex(~(x >> ~y)) == r"x \not\Rightarrow \neg y"
    assert latex(Implies(Or(x,y), z)) == r"\left(x \vee y\right) \Rightarrow z"
    assert latex(Implies(z, Or(x,y))) == r"z \Rightarrow \left(x \vee y\right)"
    assert latex(~(x & y)) == r"\neg \left(x \wedge y\right)"

    assert latex(~x, symbol_names={x: "x_i"}) == r"\neg x_i"
    assert latex(x & y, symbol_names={x: "x_i", y: "y_i"}) == \
        r"x_i \wedge y_i"
    assert latex(x & y & z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == \
        r"x_i \wedge y_i \wedge z_i"
    assert latex(x | y, symbol_names={x: "x_i", y: "y_i"}) == r"x_i \vee y_i"
    assert latex(x | y | z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == \
        r"x_i \vee y_i \vee z_i"
    assert latex((x & y) | z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == \
        r"z_i \vee \left(x_i \wedge y_i\right)"
    assert latex(Implies(x, y), symbol_names={x: "x_i", y: "y_i"}) == \
        r"x_i \Rightarrow y_i"
    assert latex(Pow(Rational(1, 3), -1, evaluate=False)) == r"\frac{1}{\frac{1}{3}}"
    assert latex(Pow(Rational(1, 3), -2, evaluate=False)) == r"\frac{1}{(\frac{1}{3})^{2}}"
    assert latex(Pow(Integer(1)/100, -1, evaluate=False)) == r"\frac{1}{\frac{1}{100}}"

    p = Symbol('p', positive=True)
    assert latex(exp(-p)*log(p)) == r"e^{- p} \log{\left(p \right)}"

    assert latex(Pow(Rational(2, 3), -1, evaluate=False)) == r'\frac{1}{\frac{2}{3}}'
    assert latex(Pow(Rational(4, 3), -1, evaluate=False)) == r'\frac{1}{\frac{4}{3}}'
    assert latex(Pow(Rational(-3, 4), -1, evaluate=False)) == r'\frac{1}{- \frac{3}{4}}'
    assert latex(Pow(Rational(-4, 4), -1, evaluate=False)) == r'\frac{1}{-1}'
    assert latex(Pow(Rational(1, 3), -1, evaluate=False)) == r'\frac{1}{\frac{1}{3}}'
    assert latex(Pow(Rational(-1, 3), -1, evaluate=False)) == r'\frac{1}{- \frac{1}{3}}'


def test_latex_builtins():
    assert latex(True) == r"\text{True}"
    assert latex(False) == r"\text{False}"
    assert latex(None) == r"\text{None}"
    assert latex(true) == r"\text{True}"
    assert latex(false) == r'\text{False}'


def test_latex_SingularityFunction():
    assert latex(SingularityFunction(x, 4, 5)) == \
        r"{\left\langle x - 4 \right\rangle}^{5}"
    assert latex(SingularityFunction(x, -3, 4)) == \
        r"{\left\langle x + 3 \right\rangle}^{4}"
    assert latex(SingularityFunction(x, 0, 4)) == \
        r"{\left\langle x \right\rangle}^{4}"
    assert latex(SingularityFunction(x, a, n)) == \
        r"{\left\langle - a + x \right\rangle}^{n}"
    assert latex(SingularityFunction(x, 4, -2)) == \
        r"{\left\langle x - 4 \right\rangle}^{-2}"
    assert latex(SingularityFunction(x, 4, -1)) == \
        r"{\left\langle x - 4 \right\rangle}^{-1}"

    assert latex(SingularityFunction(x, 4, 5)**3) == \
        r"{\left({\langle x - 4 \rangle}^{5}\right)}^{3}"
    assert latex(SingularityFunction(x, -3, 4)**3) == \
        r"{\left({\langle x + 3 \rangle}^{4}\right)}^{3}"
    assert latex(SingularityFunction(x, 0, 4)**3) == \
        r"{\left({\langle x \rangle}^{4}\right)}^{3}"
    assert latex(SingularityFunction(x, a, n)**3) == \
        r"{\left({\langle - a + x \rangle}^{n}\right)}^{3}"
    assert latex(SingularityFunction(x, 4, -2)**3) == \
        r"{\left({\langle x - 4 \rangle}^{-2}\right)}^{3}"
    assert latex((SingularityFunction(x, 4, -1)**3)**3) == \
        r"{\left({\langle x - 4 \rangle}^{-1}\right)}^{9}"


def test_latex_cycle():
    assert latex(Cycle(1, 2, 4)) == r"\left( 1\; 2\; 4\right)"
    assert latex(Cycle(1, 2)(4, 5, 6)) == \
        r"\left( 1\; 2\right)\left( 4\; 5\; 6\right)"
    assert latex(Cycle()) == r"\left( \right)"


def test_latex_permutation():
    assert latex(Permutation(1, 2, 4)) == r"\left( 1\; 2\; 4\right)"
    assert latex(Permutation(1, 2)(4, 5, 6)) == \
        r"\left( 1\; 2\right)\left( 4\; 5\; 6\right)"
    assert latex(Permutation()) == r"\left( \right)"
    assert latex(Permutation(2, 4)*Permutation(5)) == \
        r"\left( 2\; 4\right)\left( 5\right)"
    assert latex(Permutation(5)) == r"\left( 5\right)"

    assert latex(Permutation(0, 1), perm_cyclic=False) == \
        r"\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
    assert latex(Permutation(0, 1)(2, 3), perm_cyclic=False) == \
        r"\begin{pmatrix} 0 & 1 & 2 & 3 \\ 1 & 0 & 3 & 2 \end{pmatrix}"
    assert latex(Permutation(), perm_cyclic=False) == \
        r"\left( \right)"

    with warns_deprecated_sympy():
        old_print_cyclic = Permutation.print_cyclic
        Permutation.print_cyclic = False
        assert latex(Permutation(0, 1)(2, 3)) == \
            r"\begin{pmatrix} 0 & 1 & 2 & 3 \\ 1 & 0 & 3 & 2 \end{pmatrix}"
        Permutation.print_cyclic = old_print_cyclic

def test_latex_Float():
    assert latex(Float(1.0e100)) == r"1.0 \cdot 10^{100}"
    assert latex(Float(1.0e-100)) == r"1.0 \cdot 10^{-100}"
    assert latex(Float(1.0e-100), mul_symbol="times") == \
        r"1.0 \times 10^{-100}"
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=2) == \
        r"1.0 \cdot 10^{4}"
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=4) == \
        r"1.0 \cdot 10^{4}"
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=5) == \
        r"10000.0"
    assert latex(Float('0.099999'), full_prec=True,  min=-2, max=5) == \
        r"9.99990000000000 \cdot 10^{-2}"


def test_latex_vector_expressions():
    A = CoordSys3D('A')

    assert latex(Cross(A.i, A.j*A.x*3+A.k)) == \
        r"\mathbf{\hat{i}_{A}} \times \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}} + \mathbf{\hat{k}_{A}}\right)"
    assert latex(Cross(A.i, A.j)) == \
        r"\mathbf{\hat{i}_{A}} \times \mathbf{\hat{j}_{A}}"
    assert latex(x*Cross(A.i, A.j)) == \
        r"x \left(\mathbf{\hat{i}_{A}} \times \mathbf{\hat{j}_{A}}\right)"
    assert latex(Cross(x*A.i, A.j)) == \
        r'- \mathbf{\hat{j}_{A}} \times \left(\left(x\right)\mathbf{\hat{i}_{A}}\right)'

    assert latex(Curl(3*A.x*A.j)) == \
        r"\nabla\times \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(Curl(3*A.x*A.j+A.i)) == \
        r"\nabla\times \left(\mathbf{\hat{i}_{A}} + \left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(Curl(3*x*A.x*A.j)) == \
        r"\nabla\times \left(\left(3 \mathbf{{x}_{A}} x\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(x*Curl(3*A.x*A.j)) == \
        r"x \left(\nabla\times \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)\right)"

    assert latex(Divergence(3*A.x*A.j+A.i)) == \
        r"\nabla\cdot \left(\mathbf{\hat{i}_{A}} + \left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(Divergence(3*A.x*A.j)) == \
        r"\nabla\cdot \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(x*Divergence(3*A.x*A.j)) == \
        r"x \left(\nabla\cdot \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)\right)"

    assert latex(Dot(A.i, A.j*A.x*3+A.k)) == \
        r"\mathbf{\hat{i}_{A}} \cdot \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}} + \mathbf{\hat{k}_{A}}\right)"
    assert latex(Dot(A.i, A.j)) == \
        r"\mathbf{\hat{i}_{A}} \cdot \mathbf{\hat{j}_{A}}"
    assert latex(Dot(x*A.i, A.j)) == \
        r"\mathbf{\hat{j}_{A}} \cdot \left(\left(x\right)\mathbf{\hat{i}_{A}}\right)"
    assert latex(x*Dot(A.i, A.j)) == \
        r"x \left(\mathbf{\hat{i}_{A}} \cdot \mathbf{\hat{j}_{A}}\right)"

    assert latex(Gradient(A.x)) == r"\nabla \mathbf{{x}_{A}}"
    assert latex(Gradient(A.x + 3*A.y)) == \
        r"\nabla \left(\mathbf{{x}_{A}} + 3 \mathbf{{y}_{A}}\right)"
    assert latex(x*Gradient(A.x)) == r"x \left(\nabla \mathbf{{x}_{A}}\right)"
    assert latex(Gradient(x*A.x)) == r"\nabla \left(\mathbf{{x}_{A}} x\right)"

    assert latex(Laplacian(A.x)) == r"\Delta \mathbf{{x}_{A}}"
    assert latex(Laplacian(A.x + 3*A.y)) == \
        r"\Delta \left(\mathbf{{x}_{A}} + 3 \mathbf{{y}_{A}}\right)"
    assert latex(x*Laplacian(A.x)) == r"x \left(\Delta \mathbf{{x}_{A}}\right)"
    assert latex(Laplacian(x*A.x)) == r"\Delta \left(\mathbf{{x}_{A}} x\right)"

def test_latex_symbols():
    Gamma, lmbda, rho = symbols('Gamma, lambda, rho')
    tau, Tau, TAU, taU = symbols('tau, Tau, TAU, taU')
    assert latex(tau) == r"\tau"
    assert latex(Tau) == r"\mathrm{T}"
    assert latex(TAU) == r"\tau"
    assert latex(taU) == r"\tau"
    # Check that all capitalized greek letters are handled explicitly
    capitalized_letters = {l.capitalize() for l in greek_letters_set}
    assert len(capitalized_letters - set(tex_greek_dictionary.keys())) == 0
    assert latex(Gamma + lmbda) == r"\Gamma + \lambda"
    assert latex(Gamma * lmbda) == r"\Gamma \lambda"
    assert latex(Symbol('q1')) == r"q_{1}"
    assert latex(Symbol('q21')) == r"q_{21}"
    assert latex(Symbol('epsilon0')) == r"\epsilon_{0}"
    assert latex(Symbol('omega1')) == r"\omega_{1}"
    assert latex(Symbol('91')) == r"91"
    assert latex(Symbol('alpha_new')) == r"\alpha_{new}"
    assert latex(Symbol('C^orig')) == r"C^{orig}"
    assert latex(Symbol('x^alpha')) == r"x^{\alpha}"
    assert latex(Symbol('beta^alpha')) == r"\beta^{\alpha}"
    assert latex(Symbol('e^Alpha')) == r"e^{\mathrm{A}}"
    assert latex(Symbol('omega_alpha^beta')) == r"\omega^{\beta}_{\alpha}"
    assert latex(Symbol('omega') ** Symbol('beta')) == r"\omega^{\beta}"


@XFAIL
def test_latex_symbols_failing():
    rho, mass, volume = symbols('rho, mass, volume')
    assert latex(
        volume * rho == mass) == r"\rho \mathrm{volume} = \mathrm{mass}"
    assert latex(volume / mass * rho == 1) == \
        r"\rho \mathrm{volume} {\mathrm{mass}}^{(-1)} = 1"
    assert latex(mass**3 * volume**3) == \
        r"{\mathrm{mass}}^{3} \cdot {\mathrm{volume}}^{3}"


@_both_exp_pow
def test_latex_functions():
    assert latex(exp(x)) == r"e^{x}"
    assert latex(exp(1) + exp(2)) == r"e + e^{2}"

    f = Function('f')
    assert latex(f(x)) == r'f{\left(x \right)}'
    assert latex(f) == r'f'

    g = Function('g')
    assert latex(g(x, y)) == r'g{\left(x,y \right)}'
    assert latex(g) == r'g'

    h = Function('h')
    assert latex(h(x, y, z)) == r'h{\left(x,y,z \right)}'
    assert latex(h) == r'h'

    Li = Function('Li')
    assert latex(Li) == r'\operatorname{Li}'
    assert latex(Li(x)) == r'\operatorname{Li}{\left(x \right)}'

    mybeta = Function('beta')
    # not to be confused with the beta function
    assert latex(mybeta(x, y, z)) == r"\beta{\left(x,y,z \right)}"
    assert latex(beta(x, y)) == r'\operatorname{B}\left(x, y\right)'
    assert latex(beta(x, evaluate=False)) == r'\operatorname{B}\left(x, x\right)'
    assert latex(beta(x, y)**2) == r'\operatorname{B}^{2}\left(x, y\right)'
    assert latex(mybeta(x)) == r"\beta{\left(x \right)}"
    assert latex(mybeta) == r"\beta"

    g = Function('gamma')
    # not to be confused with the gamma function
    assert latex(g(x, y, z)) == r"\gamma{\left(x,y,z \right)}"
    assert latex(g(x)) == r"\gamma{\left(x \right)}"
    assert latex(g) == r"\gamma"

    a_1 = Function('a_1')
    assert latex(a_1) == r"a_{1}"
    assert latex(a_1(x)) == r"a_{1}{\left(x \right)}"
    assert latex(Function('a_1')) == r"a_{1}"

    # Issue #16925
    # multi letter function names
    # > simple
    assert latex(Function('ab')) == r"\operatorname{ab}"
    assert latex(Function('ab1')) == r"\operatorname{ab}_{1}"
    assert latex(Function('ab12')) == r"\operatorname{ab}_{12}"
    assert latex(Function('ab_1')) == r"\operatorname{ab}_{1}"
    assert latex(Function('ab_12')) == r"\operatorname{ab}_{12}"
    assert latex(Function('ab_c')) == r"\operatorname{ab}_{c}"
    assert latex(Function('ab_cd')) == r"\operatorname{ab}_{cd}"
    # > with argument
    assert latex(Function('ab')(Symbol('x'))) == r"\operatorname{ab}{\left(x \right)}"
    assert latex(Function('ab1')(Symbol('x'))) == r"\operatorname{ab}_{1}{\left(x \right)}"
    assert latex(Function('ab12')(Symbol('x'))) == r"\operatorname{ab}_{12}{\left(x \right)}"
    assert latex(Function('ab_1')(Symbol('x'))) == r"\operatorname{ab}_{1}{\left(x \right)}"
    assert latex(Function('ab_c')(Symbol('x'))) == r"\operatorname{ab}_{c}{\left(x \right)}"
    assert latex(Function('ab_cd')(Symbol('x'))) == r"\operatorname{ab}_{cd}{\left(x \right)}"

    # > with power
    #   does not work on functions without brackets

    # > with argument and power combined
    assert latex(Function('ab')()**2) == r"\operatorname{ab}^{2}{\left( \right)}"
    assert latex(Function('ab1')()**2) == r"\operatorname{ab}_{1}^{2}{\left( \right)}"
    assert latex(Function('ab12')()**2) == r"\operatorname{ab}_{12}^{2}{\left( \right)}"
    assert latex(Function('ab_1')()**2) == r"\operatorname{ab}_{1}^{2}{\left( \right)}"
    assert latex(Function('ab_12')()**2) == r"\operatorname{ab}_{12}^{2}{\left( \right)}"
    assert latex(Function('ab')(Symbol('x'))**2) == r"\operatorname{ab}^{2}{\left(x \right)}"
    assert latex(Function('ab1')(Symbol('x'))**2) == r"\operatorname{ab}_{1}^{2}{\left(x \right)}"
    assert latex(Function('ab12')(Symbol('x'))**2) == r"\operatorname{ab}_{12}^{2}{\left(x \right)}"
    assert latex(Function('ab_1')(Symbol('x'))**2) == r"\operatorname{ab}_{1}^{2}{\left(x \right)}"
    assert latex(Function('ab_12')(Symbol('x'))**2) == \
        r"\operatorname{ab}_{12}^{2}{\left(x \right)}"

    # single letter function names
    # > simple
    assert latex(Function('a')) == r"a"
    assert latex(Function('a1')) == r"a_{1}"
    assert latex(Function('a12')) == r"a_{12}"
    assert latex(Function('a_1')) == r"a_{1}"
    assert latex(Function('a_12')) == r"a_{12}"

    # > with argument
    assert latex(Function('a')()) == r"a{\left( \right)}"
    assert latex(Function('a1')()) == r"a_{1}{\left( \right)}"
    assert latex(Function('a12')()) == r"a_{12}{\left( \right)}"
    assert latex(Function('a_1')()) == r"a_{1}{\left( \right)}"
    assert latex(Function('a_12')()) == r"a_{12}{\left( \right)}"

    # > with power
    #   does not work on functions without brackets

    # > with argument and power combined
    assert latex(Function('a')()**2) == r"a^{2}{\left( \right)}"
    assert latex(Function('a1')()**2) == r"a_{1}^{2}{\left( \right)}"
    assert latex(Function('a12')()**2) == r"a_{12}^{2}{\left( \right)}"
    assert latex(Function('a_1')()**2) == r"a_{1}^{2}{\left( \right)}"
    assert latex(Function('a_12')()**2) == r"a_{12}^{2}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**2) == r"a^{2}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**2) == r"a_{1}^{2}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**2) == r"a_{12}^{2}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**2) == r"a_{1}^{2}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**2) == r"a_{12}^{2}{\left(x \right)}"

    assert latex(Function('a')()**32) == r"a^{32}{\left( \right)}"
    assert latex(Function('a1')()**32) == r"a_{1}^{32}{\left( \right)}"
    assert latex(Function('a12')()**32) == r"a_{12}^{32}{\left( \right)}"
    assert latex(Function('a_1')()**32) == r"a_{1}^{32}{\left( \right)}"
    assert latex(Function('a_12')()**32) == r"a_{12}^{32}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**32) == r"a^{32}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**32) == r"a_{1}^{32}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**32) == r"a_{12}^{32}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**32) == r"a_{1}^{32}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**32) == r"a_{12}^{32}{\left(x \right)}"

    assert latex(Function('a')()**a) == r"a^{a}{\left( \right)}"
    assert latex(Function('a1')()**a) == r"a_{1}^{a}{\left( \right)}"
    assert latex(Function('a12')()**a) == r"a_{12}^{a}{\left( \right)}"
    assert latex(Function('a_1')()**a) == r"a_{1}^{a}{\left( \right)}"
    assert latex(Function('a_12')()**a) == r"a_{12}^{a}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**a) == r"a^{a}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**a) == r"a_{1}^{a}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**a) == r"a_{12}^{a}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**a) == r"a_{1}^{a}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**a) == r"a_{12}^{a}{\left(x \right)}"

    ab = Symbol('ab')
    assert latex(Function('a')()**ab) == r"a^{ab}{\left( \right)}"
    assert latex(Function('a1')()**ab) == r"a_{1}^{ab}{\left( \right)}"
    assert latex(Function('a12')()**ab) == r"a_{12}^{ab}{\left( \right)}"
    assert latex(Function('a_1')()**ab) == r"a_{1}^{ab}{\left( \right)}"
    assert latex(Function('a_12')()**ab) == r"a_{12}^{ab}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**ab) == r"a^{ab}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**ab) == r"a_{1}^{ab}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**ab) == r"a_{12}^{ab}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**ab) == r"a_{1}^{ab}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**ab) == r"a_{12}^{ab}{\left(x \right)}"

    assert latex(Function('a^12')(x)) == R"a^{12}{\left(x \right)}"
    assert latex(Function('a^12')(x) ** ab) == R"\left(a^{12}\right)^{ab}{\left(x \right)}"
    assert latex(Function('a__12')(x)) == R"a^{12}{\left(x \right)}"
    assert latex(Function('a__12')(x) ** ab) == R"\left(a^{12}\right)^{ab}{\left(x \right)}"
    assert latex(Function('a_1__1_2')(x)) == R"a^{1}_{1 2}{\left(x \right)}"

    # issue 5868
    omega1 = Function('omega1')
    assert latex(omega1) == r"\omega_{1}"
    assert latex(omega1(x)) == r"\omega_{1}{\left(x \right)}"

    assert latex(sin(x)) == r"\sin{\left(x \right)}"
    assert latex(sin(x), fold_func_brackets=True) == r"\sin {x}"
    assert latex(sin(2*x**2), fold_func_brackets=True) == \
        r"\sin {2 x^{2}}"
    assert latex(sin(x**2), fold_func_brackets=True) == \
        r"\sin {x^{2}}"

    assert latex(asin(x)**2) == r"\operatorname{asin}^{2}{\left(x \right)}"
    assert latex(asin(x)**2, inv_trig_style="full") == \
        r"\arcsin^{2}{\left(x \right)}"
    assert latex(asin(x)**2, inv_trig_style="power") == \
        r"\sin^{-1}{\left(x \right)}^{2}"
    assert latex(asin(x**2), inv_trig_style="power",
                 fold_func_brackets=True) == \
        r"\sin^{-1} {x^{2}}"
    assert latex(acsc(x), inv_trig_style="full") == \
        r"\operatorname{arccsc}{\left(x \right)}"
    assert latex(asinh(x), inv_trig_style="full") == \
        r"\operatorname{arsinh}{\left(x \right)}"

    assert latex(factorial(k)) == r"k!"
    assert latex(factorial(-k)) == r"\left(- k\right)!"
    assert latex(factorial(k)**2) == r"k!^{2}"

    assert latex(subfactorial(k)) == r"!k"
    assert latex(subfactorial(-k)) == r"!\left(- k\right)"
    assert latex(subfactorial(k)**2) == r"\left(!k\right)^{2}"

    assert latex(factorial2(k)) == r"k!!"
    assert latex(factorial2(-k)) == r"\left(- k\right)!!"
    assert latex(factorial2(k)**2) == r"k!!^{2}"

    assert latex(binomial(2, k)) == r"{\binom{2}{k}}"
    assert latex(binomial(2, k)**2) == r"{\binom{2}{k}}^{2}"

    assert latex(FallingFactorial(3, k)) == r"{\left(3\right)}_{k}"
    assert latex(RisingFactorial(3, k)) == r"{3}^{\left(k\right)}"

    assert latex(floor(x)) == r"\left\lfloor{x}\right\rfloor"
    assert latex(ceiling(x)) == r"\left\lceil{x}\right\rceil"
    assert latex(frac(x)) == r"\operatorname{frac}{\left(x\right)}"
    assert latex(floor(x)**2) == r"\left\lfloor{x}\right\rfloor^{2}"
    assert latex(ceiling(x)**2) == r"\left\lceil{x}\right\rceil^{2}"
    assert latex(frac(x)**2) == r"\operatorname{frac}{\left(x\right)}^{2}"

    assert latex(Min(x, 2, x**3)) == r"\min\left(2, x, x^{3}\right)"
    assert latex(Min(x, y)**2) == r"\min\left(x, y\right)^{2}"
    assert latex(Max(x, 2, x**3)) == r"\max\left(2, x, x^{3}\right)"
    assert latex(Max(x, y)**2) == r"\max\left(x, y\right)^{2}"
    assert latex(Abs(x)) == r"\left|{x}\right|"
    assert latex(Abs(x)**2) == r"\left|{x}\right|^{2}"
    assert latex(re(x)) == r"\operatorname{re}{\left(x\right)}"
    assert latex(re(x + y)) == \
        r"\operatorname{re}{\left(x\right)} + \operatorname{re}{\left(y\right)}"
    assert latex(im(x)) == r"\operatorname{im}{\left(x\right)}"
    assert latex(conjugate(x)) == r"\overline{x}"
    assert latex(conjugate(x)**2) == r"\overline{x}^{2}"
    assert latex(conjugate(x**2)) == r"\overline{x}^{2}"
    assert latex(gamma(x)) == r"\Gamma\left(x\right)"
    w = Wild('w')
    assert latex(gamma(w)) == r"\Gamma\left(w\right)"
    assert latex(Order(x)) == r"O\left(x\right)"
    assert latex(Order(x, x)) == r"O\left(x\right)"
    assert latex(Order(x, (x, 0))) == r"O\left(x\right)"
    assert latex(Order(x, (x, oo))) == r"O\left(x; x\rightarrow \infty\right)"
    assert latex(Order(x - y, (x, y))) == \
        r"O\left(x - y; x\rightarrow y\right)"
    assert latex(Order(x, x, y)) == \
        r"O\left(x; \left( x, \  y\right)\rightarrow \left( 0, \  0\right)\right)"
    assert latex(Order(x, x, y)) == \
        r"O\left(x; \left( x, \  y\right)\rightarrow \left( 0, \  0\right)\right)"
    assert latex(Order(x, (x, oo), (y, oo))) == \
        r"O\left(x; \left( x, \  y\right)\rightarrow \left( \infty, \  \infty\right)\right)"
    assert latex(lowergamma(x, y)) == r'\gamma\left(x, y\right)'
    assert latex(lowergamma(x, y)**2) == r'\gamma^{2}\left(x, y\right)'
    assert latex(uppergamma(x, y)) == r'\Gamma\left(x, y\right)'
    assert latex(uppergamma(x, y)**2) == r'\Gamma^{2}\left(x, y\right)'

    assert latex(cot(x)) == r'\cot{\left(x \right)}'
    assert latex(coth(x)) == r'\coth{\left(x \right)}'
    assert latex(re(x)) == r'\operatorname{re}{\left(x\right)}'
    assert latex(im(x)) == r'\operatorname{im}{\left(x\right)}'
    assert latex(root(x, y)) == r'x^{\frac{1}{y}}'
    assert latex(arg(x)) == r'\arg{\left(x \right)}'

    assert latex(zeta(x)) == r"\zeta\left(x\right)"
    assert latex(zeta(x)**2) == r"\zeta^{2}\left(x\right)"
    assert latex(zeta(x, y)) == r"\zeta\left(x, y\right)"
    assert latex(zeta(x, y)**2) == r"\zeta^{2}\left(x, y\right)"
    assert latex(dirichlet_eta(x)) == r"\eta\left(x\right)"
    assert latex(dirichlet_eta(x)**2) == r"\eta^{2}\left(x\right)"
    assert latex(polylog(x, y)) == r"\operatorname{Li}_{x}\left(y\right)"
    assert latex(
        polylog(x, y)**2) == r"\operatorname{Li}_{x}^{2}\left(y\right)"
    assert latex(lerchphi(x, y, n)) == r"\Phi\left(x, y, n\right)"
    assert latex(lerchphi(x, y, n)**2) == r"\Phi^{2}\left(x, y, n\right)"
    assert latex(stieltjes(x)) == r"\gamma_{x}"
    assert latex(stieltjes(x)**2) == r"\gamma_{x}^{2}"
    assert latex(stieltjes(x, y)) == r"\gamma_{x}\left(y\right)"
    assert latex(stieltjes(x, y)**2) == r"\gamma_{x}\left(y\right)^{2}"

    assert latex(elliptic_k(z)) == r"K\left(z\right)"
    assert latex(elliptic_k(z)**2) == r"K^{2}\left(z\right)"
    assert latex(elliptic_f(x, y)) == r"F\left(x\middle| y\right)"
    assert latex(elliptic_f(x, y)**2) == r"F^{2}\left(x\middle| y\right)"
    assert latex(elliptic_e(x, y)) == r"E\left(x\middle| y\right)"
    assert latex(elliptic_e(x, y)**2) == r"E^{2}\left(x\middle| y\right)"
    assert latex(elliptic_e(z)) == r"E\left(z\right)"
    assert latex(elliptic_e(z)**2) == r"E^{2}\left(z\right)"
    assert latex(elliptic_pi(x, y, z)) == r"\Pi\left(x; y\middle| z\right)"
    assert latex(elliptic_pi(x, y, z)**2) == \
        r"\Pi^{2}\left(x; y\middle| z\right)"
    assert latex(elliptic_pi(x, y)) == r"\Pi\left(x\middle| y\right)"
    assert latex(elliptic_pi(x, y)**2) == r"\Pi^{2}\left(x\middle| y\right)"

    assert latex(Ei(x)) == r'\operatorname{Ei}{\left(x \right)}'
    assert latex(Ei(x)**2) == r'\operatorname{Ei}^{2}{\left(x \right)}'
    assert latex(expint(x, y)) == r'\operatorname{E}_{x}\left(y\right)'
    assert latex(expint(x, y)**2) == r'\operatorname{E}_{x}^{2}\left(y\right)'
    assert latex(Shi(x)**2) == r'\operatorname{Shi}^{2}{\left(x \right)}'
    assert latex(Si(x)**2) == r'\operatorname{Si}^{2}{\left(x \right)}'
    assert latex(Ci(x)**2) == r'\operatorname{Ci}^{2}{\left(x \right)}'
    assert latex(Chi(x)**2) == r'\operatorname{Chi}^{2}\left(x\right)'
    assert latex(Chi(x)) == r'\operatorname{Chi}\left(x\right)'
    assert latex(jacobi(n, a, b, x)) == \
        r'P_{n}^{\left(a,b\right)}\left(x\right)'
    assert latex(jacobi(n, a, b, x)**2) == \
        r'\left(P_{n}^{\left(a,b\right)}\left(x\right)\right)^{2}'
    assert latex(gegenbauer(n, a, x)) == \
        r'C_{n}^{\left(a\right)}\left(x\right)'
    assert latex(gegenbauer(n, a, x)**2) == \
        r'\left(C_{n}^{\left(a\right)}\left(x\right)\right)^{2}'
    assert latex(chebyshevt(n, x)) == r'T_{n}\left(x\right)'
    assert latex(chebyshevt(n, x)**2) == \
        r'\left(T_{n}\left(x\right)\right)^{2}'
    assert latex(chebyshevu(n, x)) == r'U_{n}\left(x\right)'
    assert latex(chebyshevu(n, x)**2) == \
        r'\left(U_{n}\left(x\right)\right)^{2}'
    assert latex(legendre(n, x)) == r'P_{n}\left(x\right)'
    assert latex(legendre(n, x)**2) == r'\left(P_{n}\left(x\right)\right)^{2}'
    assert latex(assoc_legendre(n, a, x)) == \
        r'P_{n}^{\left(a\right)}\left(x\right)'
    assert latex(assoc_legendre(n, a, x)**2) == \
        r'\left(P_{n}^{\left(a\right)}\left(x\right)\right)^{2}'
    assert latex(laguerre(n, x)) == r'L_{n}\left(x\right)'
    assert latex(laguerre(n, x)**2) == r'\left(L_{n}\left(x\right)\right)^{2}'
    assert latex(assoc_laguerre(n, a, x)) == \
        r'L_{n}^{\left(a\right)}\left(x\right)'
    assert latex(assoc_laguerre(n, a, x)**2) == \
        r'\left(L_{n}^{\left(a\right)}\left(x\right)\right)^{2}'
    assert latex(hermite(n, x)) == r'H_{n}\left(x\right)'
    assert latex(hermite(n, x)**2) == r'\left(H_{n}\left(x\right)\right)^{2}'

    theta = Symbol("theta", real=True)
    phi = Symbol("phi", real=True)
    assert latex(Ynm(n, m, theta, phi)) == r'Y_{n}^{m}\left(\theta,\phi\right)'
    assert latex(Ynm(n, m, theta, phi)**3) == \
        r'\left(Y_{n}^{m}\left(\theta,\phi\right)\right)^{3}'
    assert latex(Znm(n, m, theta, phi)) == r'Z_{n}^{m}\left(\theta,\phi\right)'
    assert latex(Znm(n, m, theta, phi)**3) == \
        r'\left(Z_{n}^{m}\left(\theta,\phi\right)\right)^{3}'

    # Test latex printing of function names with "_"
    assert latex(polar_lift(0)) == \
        r"\operatorname{polar\_lift}{\left(0 \right)}"
    assert latex(polar_lift(0)**3) == \
        r"\operatorname{polar\_lift}^{3}{\left(0 \right)}"

    assert latex(totient(n)) == r'\phi\left(n\right)'
    assert latex(totient(n) ** 2) == r'\left(\phi\left(n\right)\right)^{2}'

    assert latex(reduced_totient(n)) == r'\lambda\left(n\right)'
    assert latex(reduced_totient(n) ** 2) == \
        r'\left(\lambda\left(n\right)\right)^{2}'

    assert latex(divisor_sigma(x)) == r"\sigma\left(x\right)"
    assert latex(divisor_sigma(x)**2) == r"\sigma^{2}\left(x\right)"
    assert latex(divisor_sigma(x, y)) == r"\sigma_y\left(x\right)"
    assert latex(divisor_sigma(x, y)**2) == r"\sigma^{2}_y\left(x\right)"

    assert latex(udivisor_sigma(x)) == r"\sigma^*\left(x\right)"
    assert latex(udivisor_sigma(x)**2) == r"\sigma^*^{2}\left(x\right)"
    assert latex(udivisor_sigma(x, y)) == r"\sigma^*_y\left(x\right)"
    assert latex(udivisor_sigma(x, y)**2) == r"\sigma^*^{2}_y\left(x\right)"

    assert latex(primenu(n)) == r'\nu\left(n\right)'
    assert latex(primenu(n) ** 2) == r'\left(\nu\left(n\right)\right)^{2}'

    assert latex(primeomega(n)) == r'\Omega\left(n\right)'
    assert latex(primeomega(n) ** 2) == \
        r'\left(\Omega\left(n\right)\right)^{2}'

    assert latex(LambertW(n)) == r'W\left(n\right)'
    assert latex(LambertW(n, -1)) == r'W_{-1}\left(n\right)'
    assert latex(LambertW(n, k)) == r'W_{k}\left(n\right)'
    assert latex(LambertW(n) * LambertW(n)) == r"W^{2}\left(n\right)"
    assert latex(Pow(LambertW(n), 2)) == r"W^{2}\left(n\right)"
    assert latex(LambertW(n)**k) == r"W^{k}\left(n\right)"
    assert latex(LambertW(n, k)**p) == r"W^{p}_{k}\left(n\right)"

    assert latex(Mod(x, 7)) == r'x \bmod 7'
    assert latex(Mod(x + 1, 7)) == r'\left(x + 1\right) \bmod 7'
    assert latex(Mod(7, x + 1)) == r'7 \bmod \left(x + 1\right)'
    assert latex(Mod(2 * x, 7)) == r'2 x \bmod 7'
    assert latex(Mod(7, 2 * x)) == r'7 \bmod 2 x'
    assert latex(Mod(x, 7) + 1) == r'\left(x \bmod 7\right) + 1'
    assert latex(2 * Mod(x, 7)) == r'2 \left(x \bmod 7\right)'
    assert latex(Mod(7, 2 * x)**n) == r'\left(7 \bmod 2 x\right)^{n}'

    # some unknown function name should get rendered with \operatorname
    fjlkd = Function('fjlkd')
    assert latex(fjlkd(x)) == r'\operatorname{fjlkd}{\left(x \right)}'
    # even when it is referred to without an argument
    assert latex(fjlkd) == r'\operatorname{fjlkd}'


# test that notation passes to subclasses of the same name only
def test_function_subclass_different_name():
    class mygamma(gamma):
        pass
    assert latex(mygamma) == r"\operatorname{mygamma}"
    assert latex(mygamma(x)) == r"\operatorname{mygamma}{\left(x \right)}"


def test_hyper_printing():
    from sympy.abc import x, z

    assert latex(meijerg(Tuple(pi, pi, x), Tuple(1),
                         (0, 1), Tuple(1, 2, 3/pi), z)) == \
        r'{G_{4, 5}^{2, 3}\left(\begin{matrix} \pi, \pi, x & 1 \\0, 1 & 1, 2, '\
        r'\frac{3}{\pi} \end{matrix} \middle| {z} \right)}'
    assert latex(meijerg(Tuple(), Tuple(1), (0,), Tuple(), z)) == \
        r'{G_{1, 1}^{1, 0}\left(\begin{matrix}  & 1 \\0 &  \end{matrix} \middle| {z} \right)}'
    assert latex(hyper((x, 2), (3,), z)) == \
        r'{{}_{2}F_{1}\left(\begin{matrix} 2, x ' \
        r'\\ 3 \end{matrix}\middle| {z} \right)}'
    assert latex(hyper(Tuple(), Tuple(1), z)) == \
        r'{{}_{0}F_{1}\left(\begin{matrix}  ' \
        r'\\ 1 \end{matrix}\middle| {z} \right)}'


def test_latex_bessel():
    from sympy.functions.special.bessel import (besselj, bessely, besseli,
                                                besselk, hankel1, hankel2,
                                                jn, yn, hn1, hn2)
    from sympy.abc import z
    assert latex(besselj(n, z**2)**k) == r'J^{k}_{n}\left(z^{2}\right)'
    assert latex(bessely(n, z)) == r'Y_{n}\left(z\right)'
    assert latex(besseli(n, z)) == r'I_{n}\left(z\right)'
    assert latex(besselk(n, z)) == r'K_{n}\left(z\right)'
    assert latex(hankel1(n, z**2)**2) == \
        r'\left(H^{(1)}_{n}\left(z^{2}\right)\right)^{2}'
    assert latex(hankel2(n, z)) == r'H^{(2)}_{n}\left(z\right)'
    assert latex(jn(n, z)) == r'j_{n}\left(z\right)'
    assert latex(yn(n, z)) == r'y_{n}\left(z\right)'
    assert latex(hn1(n, z)) == r'h^{(1)}_{n}\left(z\right)'
    assert latex(hn2(n, z)) == r'h^{(2)}_{n}\left(z\right)'


def test_latex_fresnel():
    from sympy.functions.special.error_functions import (fresnels, fresnelc)
    from sympy.abc import z
    assert latex(fresnels(z)) == r'S\left(z\right)'
    assert latex(fresnelc(z)) == r'C\left(z\right)'
    assert latex(fresnels(z)**2) == r'S^{2}\left(z\right)'
    assert latex(fresnelc(z)**2) == r'C^{2}\left(z\right)'


def test_latex_brackets():
    assert latex((-1)**x) == r"\left(-1\right)^{x}"


def test_latex_indexed():
    Psi_symbol = Symbol('Psi_0', complex=True, real=False)
    Psi_indexed = IndexedBase(Symbol('Psi', complex=True, real=False))
    symbol_latex = latex(Psi_symbol * conjugate(Psi_symbol))
    indexed_latex = latex(Psi_indexed[0] * conjugate(Psi_indexed[0]))
    # \\overline{{\\Psi}_{0}} {\\Psi}_{0}  vs.  \\Psi_{0} \\overline{\\Psi_{0}}
    assert symbol_latex == r'\Psi_{0} \overline{\Psi_{0}}'
    assert indexed_latex == r'\overline{{\Psi}_{0}} {\Psi}_{0}'

    # Symbol('gamma') gives r'\gamma'
    interval = '\\mathrel{..}\\nobreak '
    assert latex(Indexed('x1', Symbol('i'))) == r'{x_{1}}_{i}'
    assert latex(Indexed('x2', Idx('i'))) == r'{x_{2}}_{i}'
    assert latex(Indexed('x3', Idx('i', Symbol('N')))) == r'{x_{3}}_{{i}_{0'+interval+'N - 1}}'
    assert latex(Indexed('x3', Idx('i', Symbol('N')+1))) == r'{x_{3}}_{{i}_{0'+interval+'N}}'
    assert latex(Indexed('x4', Idx('i', (Symbol('a'),Symbol('b'))))) == r'{x_{4}}_{{i}_{a'+interval+'b}}'
    assert latex(IndexedBase('gamma')) == r'\gamma'
    assert latex(IndexedBase('a b')) == r'a b'
    assert latex(IndexedBase('a_b')) == r'a_{b}'


def test_latex_derivatives():
    # regular "d" for ordinary derivatives
    assert latex(diff(x**3, x, evaluate=False)) == \
        r"\frac{d}{d x} x^{3}"
    assert latex(diff(sin(x) + x**2, x, evaluate=False)) == \
        r"\frac{d}{d x} \left(x^{2} + \sin{\left(x \right)}\right)"
    assert latex(diff(diff(sin(x) + x**2, x, evaluate=False), evaluate=False))\
        == \
        r"\frac{d^{2}}{d x^{2}} \left(x^{2} + \sin{\left(x \right)}\right)"
    assert latex(diff(diff(diff(sin(x) + x**2, x, evaluate=False), evaluate=False), evaluate=False)) == \
        r"\frac{d^{3}}{d x^{3}} \left(x^{2} + \sin{\left(x \right)}\right)"

    # \partial for partial derivatives
    assert latex(diff(sin(x * y), x, evaluate=False)) == \
        r"\frac{\partial}{\partial x} \sin{\left(x y \right)}"
    assert latex(diff(sin(x * y) + x**2, x, evaluate=False)) == \
        r"\frac{\partial}{\partial x} \left(x^{2} + \sin{\left(x y \right)}\right)"
    assert latex(diff(diff(sin(x*y) + x**2, x, evaluate=False), x, evaluate=False)) == \
        r"\frac{\partial^{2}}{\partial x^{2}} \left(x^{2} + \sin{\left(x y \right)}\right)"
    assert latex(diff(diff(diff(sin(x*y) + x**2, x, evaluate=False), x, evaluate=False), x, evaluate=False)) == \
        r"\frac{\partial^{3}}{\partial x^{3}} \left(x^{2} + \sin{\left(x y \right)}\right)"

    # mixed partial derivatives
    f = Function("f")
    assert latex(diff(diff(f(x, y), x, evaluate=False), y, evaluate=False)) == \
        r"\frac{\partial^{2}}{\partial y\partial x} " + latex(f(x, y))

    assert latex(diff(diff(diff(f(x, y), x, evaluate=False), x, evaluate=False), y, evaluate=False)) == \
        r"\frac{\partial^{3}}{\partial y\partial x^{2}} " + latex(f(x, y))

    # for negative nested Derivative
    assert latex(diff(-diff(y**2,x,evaluate=False),x,evaluate=False)) == r'\frac{d}{d x} \left(- \frac{d}{d x} y^{2}\right)'
    assert latex(diff(diff(-diff(diff(y,x,evaluate=False),x,evaluate=False),x,evaluate=False),x,evaluate=False)) == \
        r'\frac{d^{2}}{d x^{2}} \left(- \frac{d^{2}}{d x^{2}} y\right)'

    # use ordinary d when one of the variables has been integrated out
    assert latex(diff(Integral(exp(-x*y), (x, 0, oo)), y, evaluate=False)) == \
        r"\frac{d}{d y} \int\limits_{0}^{\infty} e^{- x y}\, dx"

    # Derivative wrapped in power:
    assert latex(diff(x, x, evaluate=False)**2) == \
        r"\left(\frac{d}{d x} x\right)^{2}"

    assert latex(diff(f(x), x)**2) == \
        r"\left(\frac{d}{d x} f{\left(x \right)}\right)^{2}"

    assert latex(diff(f(x), (x, n))) == \
        r"\frac{d^{n}}{d x^{n}} f{\left(x \right)}"

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    assert latex(diff(f(x1, x2), x1)) == r'\frac{\partial}{\partial x_{1}} f{\left(x_{1},x_{2} \right)}'

    n1 = Symbol('n1')
    assert latex(diff(f(x), (x, n1))) == r'\frac{d^{n_{1}}}{d x^{n_{1}}} f{\left(x \right)}'

    n2 = Symbol('n2')
    assert latex(diff(f(x), (x, Max(n1, n2)))) == \
        r'\frac{d^{\max\left(n_{1}, n_{2}\right)}}{d x^{\max\left(n_{1}, n_{2}\right)}} f{\left(x \right)}'

    # set diff operator
    assert latex(diff(f(x), x), diff_operator="rd") == r'\frac{\mathrm{d}}{\mathrm{d} x} f{\left(x \right)}'


def test_latex_subs():
    assert latex(Subs(x*y, (x, y), (1, 2))) == r'\left. x y \right|_{\substack{ x=1\\ y=2 }}'


def test_latex_integrals():
    assert latex(Integral(log(x), x)) == r"\int \log{\left(x \right)}\, dx"
    assert latex(Integral(x**2, (x, 0, 1))) == \
        r"\int\limits_{0}^{1} x^{2}\, dx"
    assert latex(Integral(x**2, (x, 10, 20))) == \
        r"\int\limits_{10}^{20} x^{2}\, dx"
    assert latex(Integral(y*x**2, (x, 0, 1), y)) == \
        r"\int\int\limits_{0}^{1} x^{2} y\, dx\, dy"
    assert latex(Integral(y*x**2, (x, 0, 1), y), mode='equation*') == \
        r"\begin{equation*}\int\int\limits_{0}^{1} x^{2} y\, dx\, dy\end{equation*}"
    assert latex(Integral(y*x**2, (x, 0, 1), y), mode='equation*', itex=True) \
        == r"$$\int\int_{0}^{1} x^{2} y\, dx\, dy$$"
    assert latex(Integral(x, (x, 0))) == r"\int\limits^{0} x\, dx"
    assert latex(Integral(x*y, x, y)) == r"\iint x y\, dx\, dy"
    assert latex(Integral(x*y*z, x, y, z)) == r"\iiint x y z\, dx\, dy\, dz"
    assert latex(Integral(x*y*z*t, x, y, z, t)) == \
        r"\iiiint t x y z\, dx\, dy\, dz\, dt"
    assert latex(Integral(x, x, x, x, x, x, x)) == \
        r"\int\int\int\int\int\int x\, dx\, dx\, dx\, dx\, dx\, dx"
    assert latex(Integral(x, x, y, (z, 0, 1))) == \
        r"\int\limits_{0}^{1}\int\int x\, dx\, dy\, dz"

    # for negative nested Integral
    assert latex(Integral(-Integral(y**2,x),x)) == \
        r'\int \left(- \int y^{2}\, dx\right)\, dx'
    assert latex(Integral(-Integral(-Integral(y,x),x),x)) == \
        r'\int \left(- \int \left(- \int y\, dx\right)\, dx\right)\, dx'

    # fix issue #10806
    assert latex(Integral(z, z)**2) == r"\left(\int z\, dz\right)^{2}"
    assert latex(Integral(x + z, z)) == r"\int \left(x + z\right)\, dz"
    assert latex(Integral(x+z/2, z)) == \
        r"\int \left(x + \frac{z}{2}\right)\, dz"
    assert latex(Integral(x**y, z)) == r"\int x^{y}\, dz"

    # set diff operator
    assert latex(Integral(x, x), diff_operator="rd") == r'\int x\, \mathrm{d}x'
    assert latex(Integral(x, (x, 0, 1)), diff_operator="rd") == r'\int\limits_{0}^{1} x\, \mathrm{d}x'


def test_latex_sets():
    for s in (frozenset, set):
        assert latex(s([x*y, x**2])) == r"\left\{x^{2}, x y\right\}"
        assert latex(s(range(1, 6))) == r"\left\{1, 2, 3, 4, 5\right\}"
        assert latex(s(range(1, 13))) == \
            r"\left\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\right\}"

    s = FiniteSet
    assert latex(s(*[x*y, x**2])) == r"\left\{x^{2}, x y\right\}"
    assert latex(s(*range(1, 6))) == r"\left\{1, 2, 3, 4, 5\right\}"
    assert latex(s(*range(1, 13))) == \
        r"\left\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\right\}"


def test_latex_SetExpr():
    iv = Interval(1, 3)
    se = SetExpr(iv)
    assert latex(se) == r"SetExpr\left(\left[1, 3\right]\right)"


def test_latex_Range():
    assert latex(Range(1, 51)) == r'\left\{1, 2, \ldots, 50\right\}'
    assert latex(Range(1, 4)) == r'\left\{1, 2, 3\right\}'
    assert latex(Range(0, 3, 1)) == r'\left\{0, 1, 2\right\}'
    assert latex(Range(0, 30, 1)) == r'\left\{0, 1, \ldots, 29\right\}'
    assert latex(Range(30, 1, -1)) == r'\left\{30, 29, \ldots, 2\right\}'
    assert latex(Range(0, oo, 2)) == r'\left\{0, 2, \ldots\right\}'
    assert latex(Range(oo, -2, -2)) == r'\left\{\ldots, 2, 0\right\}'
    assert latex(Range(-2, -oo, -1)) == r'\left\{-2, -3, \ldots\right\}'
    assert latex(Range(-oo, oo)) == r'\left\{\ldots, -1, 0, 1, \ldots\right\}'
    assert latex(Range(oo, -oo, -1)) == r'\left\{\ldots, 1, 0, -1, \ldots\right\}'

    a, b, c = symbols('a:c')
    assert latex(Range(a, b, c)) == r'\text{Range}\left(a, b, c\right)'
    assert latex(Range(a, 10, 1)) == r'\text{Range}\left(a, 10\right)'
    assert latex(Range(0, b, 1)) == r'\text{Range}\left(b\right)'
    assert latex(Range(0, 10, c)) == r'\text{Range}\left(0, 10, c\right)'

    i = Symbol('i', integer=True)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)

    assert latex(Range(i, i + 3)) == r'\left\{i, i + 1, i + 2\right\}'
    assert latex(Range(-oo, n, 2)) == r'\left\{\ldots, n - 4, n - 2\right\}'
    assert latex(Range(p, oo)) == r'\left\{p, p + 1, \ldots\right\}'
    # The following will work if __iter__ is improved
    # assert latex(Range(-3, p + 7)) == r'\left\{-3, -2,  \ldots, p + 6\right\}'
    # Must have integer assumptions
    assert latex(Range(a, a + 3)) == r'\text{Range}\left(a, a + 3\right)'


def test_latex_sequences():
    s1 = SeqFormula(a**2, (0, oo))
    s2 = SeqPer((1, 2))

    latex_str = r'\left[0, 1, 4, 9, \ldots\right]'
    assert latex(s1) == latex_str

    latex_str = r'\left[1, 2, 1, 2, \ldots\right]'
    assert latex(s2) == latex_str

    s3 = SeqFormula(a**2, (0, 2))
    s4 = SeqPer((1, 2), (0, 2))

    latex_str = r'\left[0, 1, 4\right]'
    assert latex(s3) == latex_str

    latex_str = r'\left[1, 2, 1\right]'
    assert latex(s4) == latex_str

    s5 = SeqFormula(a**2, (-oo, 0))
    s6 = SeqPer((1, 2), (-oo, 0))

    latex_str = r'\left[\ldots, 9, 4, 1, 0\right]'
    assert latex(s5) == latex_str

    latex_str = r'\left[\ldots, 2, 1, 2, 1\right]'
    assert latex(s6) == latex_str

    latex_str = r'\left[1, 3, 5, 11, \ldots\right]'
    assert latex(SeqAdd(s1, s2)) == latex_str

    latex_str = r'\left[1, 3, 5\right]'
    assert latex(SeqAdd(s3, s4)) == latex_str

    latex_str = r'\left[\ldots, 11, 5, 3, 1\right]'
    assert latex(SeqAdd(s5, s6)) == latex_str

    latex_str = r'\left[0, 2, 4, 18, \ldots\right]'
    assert latex(SeqMul(s1, s2)) == latex_str

    latex_str = r'\left[0, 2, 4\right]'
    assert latex(SeqMul(s3, s4)) == latex_str

    latex_str = r'\left[\ldots, 18, 4, 2, 0\right]'
    assert latex(SeqMul(s5, s6)) == latex_str

    # Sequences with symbolic limits, issue 12629
    s7 = SeqFormula(a**2, (a, 0, x))
    latex_str = r'\left\{a^{2}\right\}_{a=0}^{x}'
    assert latex(s7) == latex_str

    b = Symbol('b')
    s8 = SeqFormula(b*a**2, (a, 0, 2))
    latex_str = r'\left[0, b, 4 b\right]'
    assert latex(s8) == latex_str


def test_latex_FourierSeries():
    latex_str = \
        r'2 \sin{\left(x \right)} - \sin{\left(2 x \right)} + \frac{2 \sin{\left(3 x \right)}}{3} + \ldots'
    assert latex(fourier_series(x, (x, -pi, pi))) == latex_str


def test_latex_FormalPowerSeries():
    latex_str = r'\sum_{k=1}^{\infty} - \frac{\left(-1\right)^{- k} x^{k}}{k}'
    assert latex(fps(log(1 + x))) == latex_str


def test_latex_intervals():
    a = Symbol('a', real=True)
    assert latex(Interval(0, 0)) == r"\left\{0\right\}"
    assert latex(Interval(0, a)) == r"\left[0, a\right]"
    assert latex(Interval(0, a, False, False)) == r"\left[0, a\right]"
    assert latex(Interval(0, a, True, False)) == r"\left(0, a\right]"
    assert latex(Interval(0, a, False, True)) == r"\left[0, a\right)"
    assert latex(Interval(0, a, True, True)) == r"\left(0, a\right)"


def test_latex_AccumuBounds():
    a = Symbol('a', real=True)
    assert latex(AccumBounds(0, 1)) == r"\left\langle 0, 1\right\rangle"
    assert latex(AccumBounds(0, a)) == r"\left\langle 0, a\right\rangle"
    assert latex(AccumBounds(a + 1, a + 2)) == \
        r"\left\langle a + 1, a + 2\right\rangle"


def test_latex_emptyset():
    assert latex(S.EmptySet) == r"\emptyset"


def test_latex_universalset():
    assert latex(S.UniversalSet) == r"\mathbb{U}"


def test_latex_commutator():
    A = Operator('A')
    B = Operator('B')
    comm = Commutator(B, A)
    assert latex(comm.doit()) == r"- (A B - B A)"


def test_latex_union():
    assert latex(Union(Interval(0, 1), Interval(2, 3))) == \
        r"\left[0, 1\right] \cup \left[2, 3\right]"
    assert latex(Union(Interval(1, 1), Interval(2, 2), Interval(3, 4))) == \
        r"\left\{1, 2\right\} \cup \left[3, 4\right]"


def test_latex_intersection():
    assert latex(Intersection(Interval(0, 1), Interval(x, y))) == \
        r"\left[0, 1\right] \cap \left[x, y\right]"


def test_latex_symmetric_difference():
    assert latex(SymmetricDifference(Interval(2, 5), Interval(4, 7),
                                     evaluate=False)) == \
        r'\left[2, 5\right] \triangle \left[4, 7\right]'


def test_latex_Complement():
    assert latex(Complement(S.Reals, S.Naturals)) == \
        r"\mathbb{R} \setminus \mathbb{N}"


def test_latex_productset():
    line = Interval(0, 1)
    bigline = Interval(0, 10)
    fset = FiniteSet(1, 2, 3)
    assert latex(line**2) == r"%s^{2}" % latex(line)
    assert latex(line**10) == r"%s^{10}" % latex(line)
    assert latex((line * bigline * fset).flatten()) == r"%s \times %s \times %s" % (
        latex(line), latex(bigline), latex(fset))


def test_latex_powerset():
    fset = FiniteSet(1, 2, 3)
    assert latex(PowerSet(fset)) == r'\mathcal{P}\left(\left\{1, 2, 3\right\}\right)'


def test_latex_ordinals():
    w = OrdinalOmega()
    assert latex(w) == r"\omega"
    wp = OmegaPower(2, 3)
    assert latex(wp) == r'3 \omega^{2}'
    assert latex(Ordinal(wp, OmegaPower(1, 1))) == r'3 \omega^{2} + \omega'
    assert latex(Ordinal(OmegaPower(2, 1), OmegaPower(1, 2))) == r'\omega^{2} + 2 \omega'


def test_set_operators_parenthesis():
    a, b, c, d = symbols('a:d')
    A = FiniteSet(a)
    B = FiniteSet(b)
    C = FiniteSet(c)
    D = FiniteSet(d)

    U1 = Union(A, B, evaluate=False)
    U2 = Union(C, D, evaluate=False)
    I1 = Intersection(A, B, evaluate=False)
    I2 = Intersection(C, D, evaluate=False)
    C1 = Complement(A, B, evaluate=False)
    C2 = Complement(C, D, evaluate=False)
    D1 = SymmetricDifference(A, B, evaluate=False)
    D2 = SymmetricDifference(C, D, evaluate=False)
    # XXX ProductSet does not support evaluate keyword
    P1 = ProductSet(A, B)
    P2 = ProductSet(C, D)

    assert latex(Intersection(A, U2, evaluate=False)) == \
        r'\left\{a\right\} \cap ' \
        r'\left(\left\{c\right\} \cup \left\{d\right\}\right)'
    assert latex(Intersection(U1, U2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\cap \left(\left\{c\right\} \cup \left\{d\right\}\right)'
    assert latex(Intersection(C1, C2, evaluate=False)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \cap \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    assert latex(Intersection(D1, D2, evaluate=False)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \cap \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'
    assert latex(Intersection(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\cap \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'

    assert latex(Union(A, I2, evaluate=False)) == \
        r'\left\{a\right\} \cup ' \
        r'\left(\left\{c\right\} \cap \left\{d\right\}\right)'
    assert latex(Union(I1, I2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\cup \left(\left\{c\right\} \cap \left\{d\right\}\right)'
    assert latex(Union(C1, C2, evaluate=False)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \cup \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    assert latex(Union(D1, D2, evaluate=False)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \cup \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'
    assert latex(Union(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\cup \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'

    assert latex(Complement(A, C2, evaluate=False)) == \
        r'\left\{a\right\} \setminus \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    assert latex(Complement(U1, U2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\setminus \left(\left\{c\right\} \cup ' \
        r'\left\{d\right\}\right)'
    assert latex(Complement(I1, I2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\setminus \left(\left\{c\right\} \cap ' \
        r'\left\{d\right\}\right)'
    assert latex(Complement(D1, D2, evaluate=False)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \setminus ' \
        r'\left(\left\{c\right\} \triangle \left\{d\right\}\right)'
    assert latex(Complement(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) '\
        r'\setminus \left(\left\{c\right\} \times '\
        r'\left\{d\right\}\right)'

    assert latex(SymmetricDifference(A, D2, evaluate=False)) == \
        r'\left\{a\right\} \triangle \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'
    assert latex(SymmetricDifference(U1, U2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\triangle \left(\left\{c\right\} \cup ' \
        r'\left\{d\right\}\right)'
    assert latex(SymmetricDifference(I1, I2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\triangle \left(\left\{c\right\} \cap ' \
        r'\left\{d\right\}\right)'
    assert latex(SymmetricDifference(C1, C2, evaluate=False)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \triangle ' \
        r'\left(\left\{c\right\} \setminus \left\{d\right\}\right)'
    assert latex(SymmetricDifference(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\triangle \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'

    # XXX This can be incorrect since cartesian product is not associative
    assert latex(ProductSet(A, P2).flatten()) == \
        r'\left\{a\right\} \times \left\{c\right\} \times ' \
        r'\left\{d\right\}'
    assert latex(ProductSet(U1, U2)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\times \left(\left\{c\right\} \cup ' \
        r'\left\{d\right\}\right)'
    assert latex(ProductSet(I1, I2)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\times \left(\left\{c\right\} \cap ' \
        r'\left\{d\right\}\right)'
    assert latex(ProductSet(C1, C2)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \times \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    assert latex(ProductSet(D1, D2)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \times \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'


def test_latex_Complexes():
    assert latex(S.Complexes) == r"\mathbb{C}"


def test_latex_Naturals():
    assert latex(S.Naturals) == r"\mathbb{N}"


def test_latex_Naturals0():
    assert latex(S.Naturals0) == r"\mathbb{N}_0"


def test_latex_Integers():
    assert latex(S.Integers) == r"\mathbb{Z}"


def test_latex_ImageSet():
    x = Symbol('x')
    assert latex(ImageSet(Lambda(x, x**2), S.Naturals)) == \
        r"\left\{x^{2}\; \middle|\; x \in \mathbb{N}\right\}"

    y = Symbol('y')
    imgset = ImageSet(Lambda((x, y), x + y), {1, 2, 3}, {3, 4})
    assert latex(imgset) == \
        r"\left\{x + y\; \middle|\; x \in \left\{1, 2, 3\right\}, y \in \left\{3, 4\right\}\right\}"

    imgset = ImageSet(Lambda(((x, y),), x + y), ProductSet({1, 2, 3}, {3, 4}))
    assert latex(imgset) == \
        r"\left\{x + y\; \middle|\; \left( x, \  y\right) \in \left\{1, 2, 3\right\} \times \left\{3, 4\right\}\right\}"


def test_latex_ConditionSet():
    x = Symbol('x')
    assert latex(ConditionSet(x, Eq(x**2, 1), S.Reals)) == \
        r"\left\{x\; \middle|\; x \in \mathbb{R} \wedge x^{2} = 1 \right\}"
    assert latex(ConditionSet(x, Eq(x**2, 1), S.UniversalSet)) == \
        r"\left\{x\; \middle|\; x^{2} = 1 \right\}"


def test_latex_ComplexRegion():
    assert latex(ComplexRegion(Interval(3, 5)*Interval(4, 6))) == \
        r"\left\{x + y i\; \middle|\; x, y \in \left[3, 5\right] \times \left[4, 6\right] \right\}"
    assert latex(ComplexRegion(Interval(0, 1)*Interval(0, 2*pi), polar=True)) == \
        r"\left\{r \left(i \sin{\left(\theta \right)} + \cos{\left(\theta "\
        r"\right)}\right)\; \middle|\; r, \theta \in \left[0, 1\right] \times \left[0, 2 \pi\right) \right\}"


def test_latex_Contains():
    x = Symbol('x')
    assert latex(Contains(x, S.Naturals)) == r"x \in \mathbb{N}"


def test_latex_sum():
    assert latex(Sum(x*y**2, (x, -2, 2), (y, -5, 5))) == \
        r"\sum_{\substack{-2 \leq x \leq 2\\-5 \leq y \leq 5}} x y^{2}"
    assert latex(Sum(x**2, (x, -2, 2))) == \
        r"\sum_{x=-2}^{2} x^{2}"
    assert latex(Sum(x**2 + y, (x, -2, 2))) == \
        r"\sum_{x=-2}^{2} \left(x^{2} + y\right)"
    assert latex(Sum(x**2 + y, (x, -2, 2))**2) == \
        r"\left(\sum_{x=-2}^{2} \left(x^{2} + y\right)\right)^{2}"


def test_latex_product():
    assert latex(Product(x*y**2, (x, -2, 2), (y, -5, 5))) == \
        r"\prod_{\substack{-2 \leq x \leq 2\\-5 \leq y \leq 5}} x y^{2}"
    assert latex(Product(x**2, (x, -2, 2))) == \
        r"\prod_{x=-2}^{2} x^{2}"
    assert latex(Product(x**2 + y, (x, -2, 2))) == \
        r"\prod_{x=-2}^{2} \left(x^{2} + y\right)"

    assert latex(Product(x, (x, -2, 2))**2) == \
        r"\left(\prod_{x=-2}^{2} x\right)^{2}"


def test_latex_limits():
    assert latex(Limit(x, x, oo)) == r"\lim_{x \to \infty} x"

    # issue 8175
    f = Function('f')
    assert latex(Limit(f(x), x, 0)) == r"\lim_{x \to 0^+} f{\left(x \right)}"
    assert latex(Limit(f(x), x, 0, "-")) == \
        r"\lim_{x \to 0^-} f{\left(x \right)}"

    # issue #10806
    assert latex(Limit(f(x), x, 0)**2) == \
        r"\left(\lim_{x \to 0^+} f{\left(x \right)}\right)^{2}"
    # bi-directional limit
    assert latex(Limit(f(x), x, 0, dir='+-')) == \
        r"\lim_{x \to 0} f{\left(x \right)}"


def test_latex_log():
    assert latex(log(x)) == r"\log{\left(x \right)}"
    assert latex(log(x), ln_notation=True) == r"\ln{\left(x \right)}"
    assert latex(log(x) + log(y)) == \
        r"\log{\left(x \right)} + \log{\left(y \right)}"
    assert latex(log(x) + log(y), ln_notation=True) == \
        r"\ln{\left(x \right)} + \ln{\left(y \right)}"
    assert latex(pow(log(x), x)) == r"\log{\left(x \right)}^{x}"
    assert latex(pow(log(x), x), ln_notation=True) == \
        r"\ln{\left(x \right)}^{x}"


def test_issue_3568():
    beta = Symbol(r'\beta')
    y = beta + x
    assert latex(y) in [r'\beta + x', r'x + \beta']

    beta = Symbol(r'beta')
    y = beta + x
    assert latex(y) in [r'\beta + x', r'x + \beta']


def test_latex():
    assert latex((2*tau)**Rational(7, 2)) == r"8 \sqrt{2} \tau^{\frac{7}{2}}"
    assert latex((2*mu)**Rational(7, 2), mode='equation*') == \
        r"\begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}"
    assert latex((2*mu)**Rational(7, 2), mode='equation', itex=True) == \
        r"$$8 \sqrt{2} \mu^{\frac{7}{2}}$$"
    assert latex([2/x, y]) == r"\left[ \frac{2}{x}, \  y\right]"


def test_latex_dict():
    d = {Rational(1): 1, x**2: 2, x: 3, x**3: 4}
    assert latex(d) == \
        r'\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}'
    D = Dict(d)
    assert latex(D) == \
        r'\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}'


def test_latex_list():
    ll = [Symbol('omega1'), Symbol('a'), Symbol('alpha')]
    assert latex(ll) == r'\left[ \omega_{1}, \  a, \  \alpha\right]'


def test_latex_NumberSymbols():
    assert latex(S.Catalan) == "G"
    assert latex(S.EulerGamma) == r"\gamma"
    assert latex(S.Exp1) == "e"
    assert latex(S.GoldenRatio) == r"\phi"
    assert latex(S.Pi) == r"\pi"
    assert latex(S.TribonacciConstant) == r"\text{TribonacciConstant}"


def test_latex_rational():
    # tests issue 3973
    assert latex(-Rational(1, 2)) == r"- \frac{1}{2}"
    assert latex(Rational(-1, 2)) == r"- \frac{1}{2}"
    assert latex(Rational(1, -2)) == r"- \frac{1}{2}"
    assert latex(-Rational(-1, 2)) == r"\frac{1}{2}"
    assert latex(-Rational(1, 2)*x) == r"- \frac{x}{2}"
    assert latex(-Rational(1, 2)*x + Rational(-2, 3)*y) == \
        r"- \frac{x}{2} - \frac{2 y}{3}"


def test_latex_inverse():
    # tests issue 4129
    assert latex(1/x) == r"\frac{1}{x}"
    assert latex(1/(x + y)) == r"\frac{1}{x + y}"


def test_latex_DiracDelta():
    assert latex(DiracDelta(x)) == r"\delta\left(x\right)"
    assert latex(DiracDelta(x)**2) == r"\left(\delta\left(x\right)\right)^{2}"
    assert latex(DiracDelta(x, 0)) == r"\delta\left(x\right)"
    assert latex(DiracDelta(x, 5)) == \
        r"\delta^{\left( 5 \right)}\left( x \right)"
    assert latex(DiracDelta(x, 5)**2) == \
        r"\left(\delta^{\left( 5 \right)}\left( x \right)\right)^{2}"


def test_latex_Heaviside():
    assert latex(Heaviside(x)) == r"\theta\left(x\right)"
    assert latex(Heaviside(x)**2) == r"\left(\theta\left(x\right)\right)^{2}"


def test_latex_KroneckerDelta():
    assert latex(KroneckerDelta(x, y)) == r"\delta_{x y}"
    assert latex(KroneckerDelta(x, y + 1)) == r"\delta_{x, y + 1}"
    # issue 6578
    assert latex(KroneckerDelta(x + 1, y)) == r"\delta_{y, x + 1}"
    assert latex(Pow(KroneckerDelta(x, y), 2, evaluate=False)) == \
        r"\left(\delta_{x y}\right)^{2}"


def test_latex_LeviCivita():
    assert latex(LeviCivita(x, y, z)) == r"\varepsilon_{x y z}"
    assert latex(LeviCivita(x, y, z)**2) == \
        r"\left(\varepsilon_{x y z}\right)^{2}"
    assert latex(LeviCivita(x, y, z + 1)) == r"\varepsilon_{x, y, z + 1}"
    assert latex(LeviCivita(x, y + 1, z)) == r"\varepsilon_{x, y + 1, z}"
    assert latex(LeviCivita(x + 1, y, z)) == r"\varepsilon_{x + 1, y, z}"


def test_mode():
    expr = x + y
    assert latex(expr) == r'x + y'
    assert latex(expr, mode='plain') == r'x + y'
    assert latex(expr, mode='inline') == r'$x + y$'
    assert latex(
        expr, mode='equation*') == r'\begin{equation*}x + y\end{equation*}'
    assert latex(
        expr, mode='equation') == r'\begin{equation}x + y\end{equation}'
    raises(ValueError, lambda: latex(expr, mode='foo'))


def test_latex_mathieu():
    assert latex(mathieuc(x, y, z)) == r"C\left(x, y, z\right)"
    assert latex(mathieus(x, y, z)) == r"S\left(x, y, z\right)"
    assert latex(mathieuc(x, y, z)**2) == r"C\left(x, y, z\right)^{2}"
    assert latex(mathieus(x, y, z)**2) == r"S\left(x, y, z\right)^{2}"
    assert latex(mathieucprime(x, y, z)) == r"C^{\prime}\left(x, y, z\right)"
    assert latex(mathieusprime(x, y, z)) == r"S^{\prime}\left(x, y, z\right)"
    assert latex(mathieucprime(x, y, z)**2) == r"C^{\prime}\left(x, y, z\right)^{2}"
    assert latex(mathieusprime(x, y, z)**2) == r"S^{\prime}\left(x, y, z\right)^{2}"

def test_latex_Piecewise():
    p = Piecewise((x, x < 1), (x**2, True))
    assert latex(p) == r"\begin{cases} x & \text{for}\: x < 1 \\x^{2} &" \
                       r" \text{otherwise} \end{cases}"
    assert latex(p, itex=True) == \
        r"\begin{cases} x & \text{for}\: x \lt 1 \\x^{2} &" \
        r" \text{otherwise} \end{cases}"
    p = Piecewise((x, x < 0), (0, x >= 0))
    assert latex(p) == r'\begin{cases} x & \text{for}\: x < 0 \\0 &' \
                       r' \text{otherwise} \end{cases}'
    A, B = symbols("A B", commutative=False)
    p = Piecewise((A**2, Eq(A, B)), (A*B, True))
    s = r"\begin{cases} A^{2} & \text{for}\: A = B \\A B & \text{otherwise} \end{cases}"
    assert latex(p) == s
    assert latex(A*p) == r"A \left(%s\right)" % s
    assert latex(p*A) == r"\left(%s\right) A" % s
    assert latex(Piecewise((x, x < 1), (x**2, x < 2))) == \
        r'\begin{cases} x & ' \
        r'\text{for}\: x < 1 \\x^{2} & \text{for}\: x < 2 \end{cases}'


def test_latex_Matrix():
    M = Matrix([[1 + x, y], [y, x - 1]])
    assert latex(M) == \
        r'\left[\begin{matrix}x + 1 & y\\y & x - 1\end{matrix}\right]'
    assert latex(M, mode='inline') == \
        r'$\left[\begin{smallmatrix}x + 1 & y\\' \
        r'y & x - 1\end{smallmatrix}\right]$'
    assert latex(M, mat_str='array') == \
        r'\left[\begin{array}{cc}x + 1 & y\\y & x - 1\end{array}\right]'
    assert latex(M, mat_str='bmatrix') == \
        r'\left[\begin{bmatrix}x + 1 & y\\y & x - 1\end{bmatrix}\right]'
    assert latex(M, mat_delim=None, mat_str='bmatrix') == \
        r'\begin{bmatrix}x + 1 & y\\y & x - 1\end{bmatrix}'

    M2 = Matrix(1, 11, range(11))
    assert latex(M2) == \
        r'\left[\begin{array}{ccccccccccc}' \
        r'0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\end{array}\right]'


def test_latex_matrix_with_functions():
    t = symbols('t')
    theta1 = symbols('theta1', cls=Function)

    M = Matrix([[sin(theta1(t)), cos(theta1(t))],
                [cos(theta1(t).diff(t)), sin(theta1(t).diff(t))]])

    expected = (r'\left[\begin{matrix}\sin{\left('
                r'\theta_{1}{\left(t \right)} \right)} & '
                r'\cos{\left(\theta_{1}{\left(t \right)} \right)'
                r'}\\\cos{\left(\frac{d}{d t} \theta_{1}{\left(t '
                r'\right)} \right)} & \sin{\left(\frac{d}{d t} '
                r'\theta_{1}{\left(t \right)} \right'
                r')}\end{matrix}\right]')

    assert latex(M) == expected


def test_latex_NDimArray():
    x, y, z, w = symbols("x y z w")

    for ArrayType in (ImmutableDenseNDimArray, ImmutableSparseNDimArray,
                      MutableDenseNDimArray, MutableSparseNDimArray):
        # Basic: scalar array
        M = ArrayType(x)

        assert latex(M) == r"x"

        M = ArrayType([[1 / x, y], [z, w]])
        M1 = ArrayType([1 / x, y, z])

        M2 = tensorproduct(M1, M)
        M3 = tensorproduct(M, M)

        assert latex(M) == \
            r'\left[\begin{matrix}\frac{1}{x} & y\\z & w\end{matrix}\right]'
        assert latex(M1) == \
            r"\left[\begin{matrix}\frac{1}{x} & y & z\end{matrix}\right]"
        assert latex(M2) == \
            r"\left[\begin{matrix}" \
            r"\left[\begin{matrix}\frac{1}{x^{2}} & \frac{y}{x}\\\frac{z}{x} & \frac{w}{x}\end{matrix}\right] & " \
            r"\left[\begin{matrix}\frac{y}{x} & y^{2}\\y z & w y\end{matrix}\right] & " \
            r"\left[\begin{matrix}\frac{z}{x} & y z\\z^{2} & w z\end{matrix}\right]" \
            r"\end{matrix}\right]"
        assert latex(M3) == \
            r"""\left[\begin{matrix}"""\
            r"""\left[\begin{matrix}\frac{1}{x^{2}} & \frac{y}{x}\\\frac{z}{x} & \frac{w}{x}\end{matrix}\right] & """\
            r"""\left[\begin{matrix}\frac{y}{x} & y^{2}\\y z & w y\end{matrix}\right]\\"""\
            r"""\left[\begin{matrix}\frac{z}{x} & y z\\z^{2} & w z\end{matrix}\right] & """\
            r"""\left[\begin{matrix}\frac{w}{x} & w y\\w z & w^{2}\end{matrix}\right]"""\
            r"""\end{matrix}\right]"""

        Mrow = ArrayType([[x, y, 1/z]])
        Mcolumn = ArrayType([[x], [y], [1/z]])
        Mcol2 = ArrayType([Mcolumn.tolist()])

        assert latex(Mrow) == \
            r"\left[\left[\begin{matrix}x & y & \frac{1}{z}\end{matrix}\right]\right]"
        assert latex(Mcolumn) == \
            r"\left[\begin{matrix}x\\y\\\frac{1}{z}\end{matrix}\right]"
        assert latex(Mcol2) == \
            r'\left[\begin{matrix}\left[\begin{matrix}x\\y\\\frac{1}{z}\end{matrix}\right]\end{matrix}\right]'


def test_latex_mul_symbol():
    assert latex(4*4**x, mul_symbol='times') == r"4 \times 4^{x}"
    assert latex(4*4**x, mul_symbol='dot') == r"4 \cdot 4^{x}"
    assert latex(4*4**x, mul_symbol='ldot') == r"4 \,.\, 4^{x}"

    assert latex(4*x, mul_symbol='times') == r"4 \times x"
    assert latex(4*x, mul_symbol='dot') == r"4 \cdot x"
    assert latex(4*x, mul_symbol='ldot') == r"4 \,.\, x"


def test_latex_issue_4381():
    y = 4*4**log(2)
    assert latex(y) == r'4 \cdot 4^{\log{\left(2 \right)}}'
    assert latex(1/y) == r'\frac{1}{4 \cdot 4^{\log{\left(2 \right)}}}'


def test_latex_issue_4576():
    assert latex(Symbol("beta_13_2")) == r"\beta_{13 2}"
    assert latex(Symbol("beta_132_20")) == r"\beta_{132 20}"
    assert latex(Symbol("beta_13")) == r"\beta_{13}"
    assert latex(Symbol("x_a_b")) == r"x_{a b}"
    assert latex(Symbol("x_1_2_3")) == r"x_{1 2 3}"
    assert latex(Symbol("x_a_b1")) == r"x_{a b1}"
    assert latex(Symbol("x_a_1")) == r"x_{a 1}"
    assert latex(Symbol("x_1_a")) == r"x_{1 a}"
    assert latex(Symbol("x_1^aa")) == r"x^{aa}_{1}"
    assert latex(Symbol("x_1__aa")) == r"x^{aa}_{1}"
    assert latex(Symbol("x_11^a")) == r"x^{a}_{11}"
    assert latex(Symbol("x_11__a")) == r"x^{a}_{11}"
    assert latex(Symbol("x_a_a_a_a")) == r"x_{a a a a}"
    assert latex(Symbol("x_a_a^a^a")) == r"x^{a a}_{a a}"
    assert latex(Symbol("x_a_a__a__a")) == r"x^{a a}_{a a}"
    assert latex(Symbol("alpha_11")) == r"\alpha_{11}"
    assert latex(Symbol("alpha_11_11")) == r"\alpha_{11 11}"
    assert latex(Symbol("alpha_alpha")) == r"\alpha_{\alpha}"
    assert latex(Symbol("alpha^aleph")) == r"\alpha^{\aleph}"
    assert latex(Symbol("alpha__aleph")) == r"\alpha^{\aleph}"


def test_latex_pow_fraction():
    x = Symbol('x')
    # Testing exp
    assert r'e^{-x}' in latex(exp(-x)/2).replace(' ', '')  # Remove Whitespace

    # Testing e^{-x} in case future changes alter behavior of muls or fracs
    # In particular current output is \frac{1}{2}e^{- x} but perhaps this will
    # change to \frac{e^{-x}}{2}

    # Testing general, non-exp, power
    assert r'3^{-x}' in latex(3**-x/2).replace(' ', '')


def test_noncommutative():
    A, B, C = symbols('A,B,C', commutative=False)

    assert latex(A*B*C**-1) == r"A B C^{-1}"
    assert latex(C**-1*A*B) == r"C^{-1} A B"
    assert latex(A*C**-1*B) == r"A C^{-1} B"


def test_latex_order():
    expr = x**3 + x**2*y + y**4 + 3*x*y**3

    assert latex(expr, order='lex') == r"x^{3} + x^{2} y + 3 x y^{3} + y^{4}"
    assert latex(
        expr, order='rev-lex') == r"y^{4} + 3 x y^{3} + x^{2} y + x^{3}"
    assert latex(expr, order='none') == r"x^{3} + y^{4} + y x^{2} + 3 x y^{3}"


def test_latex_Lambda():
    assert latex(Lambda(x, x + 1)) == r"\left( x \mapsto x + 1 \right)"
    assert latex(Lambda((x, y), x + 1)) == r"\left( \left( x, \  y\right) \mapsto x + 1 \right)"
    assert latex(Lambda(x, x)) == r"\left( x \mapsto x \right)"

def test_latex_PolyElement():
    Ruv, u, v = ring("u,v", ZZ)
    Rxyz, x, y, z = ring("x,y,z", Ruv)

    assert latex(x - x) == r"0"
    assert latex(x - 1) == r"x - 1"
    assert latex(x + 1) == r"x + 1"

    assert latex((u**2 + 3*u*v + 1)*x**2*y + u + 1) == \
        r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + u + 1"
    assert latex((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x) == \
        r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + \left(u + 1\right) x"
    assert latex((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x + 1) == \
        r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + \left(u + 1\right) x + 1"
    assert latex((-u**2 + 3*u*v - 1)*x**2*y - (u + 1)*x - 1) == \
        r"-\left({u}^{2} - 3 u v + 1\right) {x}^{2} y - \left(u + 1\right) x - 1"

    assert latex(-(v**2 + v + 1)*x + 3*u*v + 1) == \
        r"-\left({v}^{2} + v + 1\right) x + 3 u v + 1"
    assert latex(-(v**2 + v + 1)*x - 3*u*v + 1) == \
        r"-\left({v}^{2} + v + 1\right) x - 3 u v + 1"


def test_latex_FracElement():
    Fuv, u, v = field("u,v", ZZ)
    Fxyzt, x, y, z, t = field("x,y,z,t", Fuv)

    assert latex(x - x) == r"0"
    assert latex(x - 1) == r"x - 1"
    assert latex(x + 1) == r"x + 1"

    assert latex(x/3) == r"\frac{x}{3}"
    assert latex(x/z) == r"\frac{x}{z}"
    assert latex(x*y/z) == r"\frac{x y}{z}"
    assert latex(x/(z*t)) == r"\frac{x}{z t}"
    assert latex(x*y/(z*t)) == r"\frac{x y}{z t}"

    assert latex((x - 1)/y) == r"\frac{x - 1}{y}"
    assert latex((x + 1)/y) == r"\frac{x + 1}{y}"
    assert latex((-x - 1)/y) == r"\frac{-x - 1}{y}"
    assert latex((x + 1)/(y*z)) == r"\frac{x + 1}{y z}"
    assert latex(-y/(x + 1)) == r"\frac{-y}{x + 1}"
    assert latex(y*z/(x + 1)) == r"\frac{y z}{x + 1}"

    assert latex(((u + 1)*x*y + 1)/((v - 1)*z - 1)) == \
        r"\frac{\left(u + 1\right) x y + 1}{\left(v - 1\right) z - 1}"
    assert latex(((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)) == \
        r"\frac{\left(u + 1\right) x y + 1}{\left(v - 1\right) z - u v t - 1}"


def test_latex_Poly():
    assert latex(Poly(x**2 + 2 * x, x)) == \
        r"\operatorname{Poly}{\left( x^{2} + 2 x, x, domain=\mathbb{Z} \right)}"
    assert latex(Poly(x/y, x)) == \
        r"\operatorname{Poly}{\left( \frac{1}{y} x, x, domain=\mathbb{Z}\left(y\right) \right)}"
    assert latex(Poly(2.0*x + y)) == \
        r"\operatorname{Poly}{\left( 2.0 x + 1.0 y, x, y, domain=\mathbb{R} \right)}"


def test_latex_Poly_order():
    assert latex(Poly([a, 1, b, 2, c, 3], x)) == \
        r'\operatorname{Poly}{\left( a x^{5} + x^{4} + b x^{3} + 2 x^{2} + c'\
        r' x + 3, x, domain=\mathbb{Z}\left[a, b, c\right] \right)}'
    assert latex(Poly([a, 1, b+c, 2, 3], x)) == \
        r'\operatorname{Poly}{\left( a x^{4} + x^{3} + \left(b + c\right) '\
        r'x^{2} + 2 x + 3, x, domain=\mathbb{Z}\left[a, b, c\right] \right)}'
    assert latex(Poly(a*x**3 + x**2*y - x*y - c*y**3 - b*x*y**2 + y - a*x + b,
                      (x, y))) == \
        r'\operatorname{Poly}{\left( a x^{3} + x^{2}y -  b xy^{2} - xy -  '\
        r'a x -  c y^{3} + y + b, x, y, domain=\mathbb{Z}\left[a, b, c\right] \right)}'


def test_latex_ComplexRootOf():
    assert latex(rootof(x**5 + x + 3, 0)) == \
        r"\operatorname{CRootOf} {\left(x^{5} + x + 3, 0\right)}"


def test_latex_RootSum():
    assert latex(RootSum(x**5 + x + 3, sin)) == \
        r"\operatorname{RootSum} {\left(x^{5} + x + 3, \left( x \mapsto \sin{\left(x \right)} \right)\right)}"


def test_settings():
    raises(TypeError, lambda: latex(x*y, method="garbage"))


def test_latex_numbers():
    assert latex(catalan(n)) == r"C_{n}"
    assert latex(catalan(n)**2) == r"C_{n}^{2}"
    assert latex(bernoulli(n)) == r"B_{n}"
    assert latex(bernoulli(n, x)) == r"B_{n}\left(x\right)"
    assert latex(bernoulli(n)**2) == r"B_{n}^{2}"
    assert latex(bernoulli(n, x)**2) == r"B_{n}^{2}\left(x\right)"
    assert latex(genocchi(n)) == r"G_{n}"
    assert latex(genocchi(n, x)) == r"G_{n}\left(x\right)"
    assert latex(genocchi(n)**2) == r"G_{n}^{2}"
    assert latex(genocchi(n, x)**2) == r"G_{n}^{2}\left(x\right)"
    assert latex(bell(n)) == r"B_{n}"
    assert latex(bell(n, x)) == r"B_{n}\left(x\right)"
    assert latex(bell(n, m, (x, y))) == r"B_{n, m}\left(x, y\right)"
    assert latex(bell(n)**2) == r"B_{n}^{2}"
    assert latex(bell(n, x)**2) == r"B_{n}^{2}\left(x\right)"
    assert latex(bell(n, m, (x, y))**2) == r"B_{n, m}^{2}\left(x, y\right)"
    assert latex(fibonacci(n)) == r"F_{n}"
    assert latex(fibonacci(n, x)) == r"F_{n}\left(x\right)"
    assert latex(fibonacci(n)**2) == r"F_{n}^{2}"
    assert latex(fibonacci(n, x)**2) == r"F_{n}^{2}\left(x\right)"
    assert latex(lucas(n)) == r"L_{n}"
    assert latex(lucas(n)**2) == r"L_{n}^{2}"
    assert latex(tribonacci(n)) == r"T_{n}"
    assert latex(tribonacci(n, x)) == r"T_{n}\left(x\right)"
    assert latex(tribonacci(n)**2) == r"T_{n}^{2}"
    assert latex(tribonacci(n, x)**2) == r"T_{n}^{2}\left(x\right)"
    assert latex(mobius(n)) == r"\mu\left(n\right)"
    assert latex(mobius(n)**2) == r"\mu^{2}\left(n\right)"


def test_latex_euler():
    assert latex(euler(n)) == r"E_{n}"
    assert latex(euler(n, x)) == r"E_{n}\left(x\right)"
    assert latex(euler(n, x)**2) == r"E_{n}^{2}\left(x\right)"


def test_lamda():
    assert latex(Symbol('lamda')) == r"\lambda"
    assert latex(Symbol('Lamda')) == r"\Lambda"


def test_custom_symbol_names():
    x = Symbol('x')
    y = Symbol('y')
    assert latex(x) == r"x"
    assert latex(x, symbol_names={x: "x_i"}) == r"x_i"
    assert latex(x + y, symbol_names={x: "x_i"}) == r"x_i + y"
    assert latex(x**2, symbol_names={x: "x_i"}) == r"x_i^{2}"
    assert latex(x + y, symbol_names={x: "x_i", y: "y_j"}) == r"x_i + y_j"


def test_matAdd():
    C = MatrixSymbol('C', 5, 5)
    B = MatrixSymbol('B', 5, 5)

    n = symbols("n")
    h = MatrixSymbol("h", 1, 1)

    assert latex(C - 2*B) in [r'- 2 B + C', r'C -2 B']
    assert latex(C + 2*B) in [r'2 B + C', r'C + 2 B']
    assert latex(B - 2*C) in [r'B - 2 C', r'- 2 C + B']
    assert latex(B + 2*C) in [r'B + 2 C', r'2 C + B']

    assert latex(n * h - (-h + h.T) * (h + h.T)) == 'n h - \\left(- h + h^{T}\\right) \\left(h + h^{T}\\right)'
    assert latex(MatAdd(MatAdd(h, h), MatAdd(h, h))) == '\\left(h + h\\right) + \\left(h + h\\right)'
    assert latex(MatMul(MatMul(h, h), MatMul(h, h))) == '\\left(h h\\right) \\left(h h\\right)'


def test_matMul():
    A = MatrixSymbol('A', 5, 5)
    B = MatrixSymbol('B', 5, 5)
    x = Symbol('x')
    assert latex(2*A) == r'2 A'
    assert latex(2*x*A) == r'2 x A'
    assert latex(-2*A) == r'- 2 A'
    assert latex(1.5*A) == r'1.5 A'
    assert latex(sqrt(2)*A) == r'\sqrt{2} A'
    assert latex(-sqrt(2)*A) == r'- \sqrt{2} A'
    assert latex(2*sqrt(2)*x*A) == r'2 \sqrt{2} x A'
    assert latex(-2*A*(A + 2*B)) in [r'- 2 A \left(A + 2 B\right)',
                                        r'- 2 A \left(2 B + A\right)']


def test_latex_MatrixSlice():
    n = Symbol('n', integer=True)
    x, y, z, w, t, = symbols('x y z w t')
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 10, 10)
    Z = MatrixSymbol('Z', 10, 10)

    assert latex(MatrixSlice(X, (None, None, None), (None, None, None))) == r'X\left[:, :\right]'
    assert latex(X[x:x + 1, y:y + 1]) == r'X\left[x:x + 1, y:y + 1\right]'
    assert latex(X[x:x + 1:2, y:y + 1:2]) == r'X\left[x:x + 1:2, y:y + 1:2\right]'
    assert latex(X[:x, y:]) == r'X\left[:x, y:\right]'
    assert latex(X[:x, y:]) == r'X\left[:x, y:\right]'
    assert latex(X[x:, :y]) == r'X\left[x:, :y\right]'
    assert latex(X[x:y, z:w]) == r'X\left[x:y, z:w\right]'
    assert latex(X[x:y:t, w:t:x]) == r'X\left[x:y:t, w:t:x\right]'
    assert latex(X[x::y, t::w]) == r'X\left[x::y, t::w\right]'
    assert latex(X[:x:y, :t:w]) == r'X\left[:x:y, :t:w\right]'
    assert latex(X[::x, ::y]) == r'X\left[::x, ::y\right]'
    assert latex(MatrixSlice(X, (0, None, None), (0, None, None))) == r'X\left[:, :\right]'
    assert latex(MatrixSlice(X, (None, n, None), (None, n, None))) == r'X\left[:, :\right]'
    assert latex(MatrixSlice(X, (0, n, None), (0, n, None))) == r'X\left[:, :\right]'
    assert latex(MatrixSlice(X, (0, n, 2), (0, n, 2))) == r'X\left[::2, ::2\right]'
    assert latex(X[1:2:3, 4:5:6]) == r'X\left[1:2:3, 4:5:6\right]'
    assert latex(X[1:3:5, 4:6:8]) == r'X\left[1:3:5, 4:6:8\right]'
    assert latex(X[1:10:2]) == r'X\left[1:10:2, :\right]'
    assert latex(Y[:5, 1:9:2]) == r'Y\left[:5, 1:9:2\right]'
    assert latex(Y[:5, 1:10:2]) == r'Y\left[:5, 1::2\right]'
    assert latex(Y[5, :5:2]) == r'Y\left[5:6, :5:2\right]'
    assert latex(X[0:1, 0:1]) == r'X\left[:1, :1\right]'
    assert latex(X[0:1:2, 0:1:2]) == r'X\left[:1:2, :1:2\right]'
    assert latex((Y + Z)[2:, 2:]) == r'\left(Y + Z\right)\left[2:, 2:\right]'


def test_latex_RandomDomain():
    from sympy.stats import Normal, Die, Exponential, pspace, where
    from sympy.stats.rv import RandomDomain

    X = Normal('x1', 0, 1)
    assert latex(where(X > 0)) == r"\text{Domain: }0 < x_{1} \wedge x_{1} < \infty"

    D = Die('d1', 6)
    assert latex(where(D > 4)) == r"\text{Domain: }d_{1} = 5 \vee d_{1} = 6"

    A = Exponential('a', 1)
    B = Exponential('b', 1)
    assert latex(
        pspace(Tuple(A, B)).domain) == \
        r"\text{Domain: }0 \leq a \wedge 0 \leq b \wedge a < \infty \wedge b < \infty"

    assert latex(RandomDomain(FiniteSet(x), FiniteSet(1, 2))) == \
        r'\text{Domain: }\left\{x\right\} \in \left\{1, 2\right\}'

def test_PrettyPoly():
    from sympy.polys.domains import QQ
    F = QQ.frac_field(x, y)
    R = QQ[x, y]

    assert latex(F.convert(x/(x + y))) == latex(x/(x + y))
    assert latex(R.convert(x + y)) == latex(x + y)


def test_integral_transforms():
    x = Symbol("x")
    k = Symbol("k")
    f = Function("f")
    a = Symbol("a")
    b = Symbol("b")

    assert latex(MellinTransform(f(x), x, k)) == \
        r"\mathcal{M}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseMellinTransform(f(k), k, x, a, b)) == \
        r"\mathcal{M}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    assert latex(LaplaceTransform(f(x), x, k)) == \
        r"\mathcal{L}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseLaplaceTransform(f(k), k, x, (a, b))) == \
        r"\mathcal{L}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    assert latex(FourierTransform(f(x), x, k)) == \
        r"\mathcal{F}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseFourierTransform(f(k), k, x)) == \
        r"\mathcal{F}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    assert latex(CosineTransform(f(x), x, k)) == \
        r"\mathcal{COS}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseCosineTransform(f(k), k, x)) == \
        r"\mathcal{COS}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    assert latex(SineTransform(f(x), x, k)) == \
        r"\mathcal{SIN}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseSineTransform(f(k), k, x)) == \
        r"\mathcal{SIN}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"


def test_PolynomialRingBase():
    from sympy.polys.domains import QQ
    assert latex(QQ.old_poly_ring(x, y)) == r"\mathbb{Q}\left[x, y\right]"
    assert latex(QQ.old_poly_ring(x, y, order="ilex")) == \
        r"S_<^{-1}\mathbb{Q}\left[x, y\right]"


def test_categories():
    from sympy.categories import (Object, IdentityMorphism,
                                  NamedMorphism, Category, Diagram,
                                  DiagramGrid)

    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")

    f1 = NamedMorphism(A1, A2, "f1")
    f2 = NamedMorphism(A2, A3, "f2")
    id_A1 = IdentityMorphism(A1)

    K1 = Category("K1")

    assert latex(A1) == r"A_{1}"
    assert latex(f1) == r"f_{1}:A_{1}\rightarrow A_{2}"
    assert latex(id_A1) == r"id:A_{1}\rightarrow A_{1}"
    assert latex(f2*f1) == r"f_{2}\circ f_{1}:A_{1}\rightarrow A_{3}"

    assert latex(K1) == r"\mathbf{K_{1}}"

    d = Diagram()
    assert latex(d) == r"\emptyset"

    d = Diagram({f1: "unique", f2: S.EmptySet})
    assert latex(d) == r"\left\{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \emptyset, \  id:A_{1}\rightarrow " \
        r"A_{1} : \emptyset, \  id:A_{2}\rightarrow A_{2} : " \
        r"\emptyset, \  id:A_{3}\rightarrow A_{3} : \emptyset, " \
        r"\  f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}, " \
        r"\  f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}"

    d = Diagram({f1: "unique", f2: S.EmptySet}, {f2 * f1: "unique"})
    assert latex(d) == r"\left\{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \emptyset, \  id:A_{1}\rightarrow " \
        r"A_{1} : \emptyset, \  id:A_{2}\rightarrow A_{2} : " \
        r"\emptyset, \  id:A_{3}\rightarrow A_{3} : \emptyset, " \
        r"\  f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}," \
        r" \  f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}" \
        r"\Longrightarrow \left\{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \left\{unique\right\}\right\}"

    # A linear diagram.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    d = Diagram([f, g])
    grid = DiagramGrid(d)

    assert latex(grid) == r"\begin{array}{cc}" + "\n" \
        r"A & B \\" + "\n" \
        r" & C " + "\n" \
        r"\end{array}" + "\n"


def test_Modules():
    from sympy.polys.domains import QQ
    from sympy.polys.agca import homomorphism

    R = QQ.old_poly_ring(x, y)
    F = R.free_module(2)
    M = F.submodule([x, y], [1, x**2])

    assert latex(F) == r"{\mathbb{Q}\left[x, y\right]}^{2}"
    assert latex(M) == \
        r"\left\langle {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right\rangle"

    I = R.ideal(x**2, y)
    assert latex(I) == r"\left\langle {x^{2}},{y} \right\rangle"

    Q = F / M
    assert latex(Q) == \
        r"\frac{{\mathbb{Q}\left[x, y\right]}^{2}}{\left\langle {\left[ {x},"\
        r"{y} \right]},{\left[ {1},{x^{2}} \right]} \right\rangle}"
    assert latex(Q.submodule([1, x**3/2], [2, y])) == \
        r"\left\langle {{\left[ {1},{\frac{x^{3}}{2}} \right]} + {\left"\
        r"\langle {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} "\
        r"\right\rangle}},{{\left[ {2},{y} \right]} + {\left\langle {\left[ "\
        r"{x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right\rangle}} \right\rangle"

    h = homomorphism(QQ.old_poly_ring(x).free_module(2),
                     QQ.old_poly_ring(x).free_module(2), [0, 0])

    assert latex(h) == \
        r"{\left[\begin{matrix}0 & 0\\0 & 0\end{matrix}\right]} : "\
        r"{{\mathbb{Q}\left[x\right]}^{2}} \to {{\mathbb{Q}\left[x\right]}^{2}}"


def test_QuotientRing():
    from sympy.polys.domains import QQ
    R = QQ.old_poly_ring(x)/[x**2 + 1]

    assert latex(R) == \
        r"\frac{\mathbb{Q}\left[x\right]}{\left\langle {x^{2} + 1} \right\rangle}"
    assert latex(R.one) == r"{1} + {\left\langle {x^{2} + 1} \right\rangle}"


def test_Tr():
    #TODO: Handle indices
    A, B = symbols('A B', commutative=False)
    t = Tr(A*B)
    assert latex(t) == r'\operatorname{tr}\left(A B\right)'


def test_Determinant():
    from sympy.matrices import Determinant, Inverse, BlockMatrix, OneMatrix, ZeroMatrix
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Determinant(m)) == '\\left|{\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}}\\right|'
    assert latex(Determinant(Inverse(m))) == \
        '\\left|{\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{-1}}\\right|'
    X = MatrixSymbol('X', 2, 2)
    assert latex(Determinant(X)) == '\\left|{X}\\right|'
    assert latex(Determinant(X + m)) == \
        '\\left|{\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X}\\right|'
    assert latex(Determinant(BlockMatrix(((OneMatrix(2, 2), X),
                                          (m, ZeroMatrix(2, 2)))))) == \
        '\\left|{\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}}\\right|'


def test_Adjoint():
    from sympy.matrices import Adjoint, Inverse, Transpose
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(Adjoint(X)) == r'X^{\dagger}'
    assert latex(Adjoint(X + Y)) == r'\left(X + Y\right)^{\dagger}'
    assert latex(Adjoint(X) + Adjoint(Y)) == r'X^{\dagger} + Y^{\dagger}'
    assert latex(Adjoint(X*Y)) == r'\left(X Y\right)^{\dagger}'
    assert latex(Adjoint(Y)*Adjoint(X)) == r'Y^{\dagger} X^{\dagger}'
    assert latex(Adjoint(X**2)) == r'\left(X^{2}\right)^{\dagger}'
    assert latex(Adjoint(X)**2) == r'\left(X^{\dagger}\right)^{2}'
    assert latex(Adjoint(Inverse(X))) == r'\left(X^{-1}\right)^{\dagger}'
    assert latex(Inverse(Adjoint(X))) == r'\left(X^{\dagger}\right)^{-1}'
    assert latex(Adjoint(Transpose(X))) == r'\left(X^{T}\right)^{\dagger}'
    assert latex(Transpose(Adjoint(X))) == r'\left(X^{\dagger}\right)^{T}'
    assert latex(Transpose(Adjoint(X) + Y)) == r'\left(X^{\dagger} + Y\right)^{T}'
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Adjoint(m)) == '\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{\\dagger}'
    assert latex(Adjoint(m+X)) == \
        '\\left(\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X\\right)^{\\dagger}'
    from sympy.matrices import BlockMatrix, OneMatrix, ZeroMatrix
    assert latex(Adjoint(BlockMatrix(((OneMatrix(2, 2), X),
                                      (m, ZeroMatrix(2, 2)))))) == \
        '\\left[\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}\\right]^{\\dagger}'
    # Issue 20959
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(Adjoint(Mx)) == r'\left(M^{x}\right)^{\dagger}'

    # adjoint style
    assert latex(Adjoint(X), adjoint_style="star") == r'X^{\ast}'
    assert latex(Adjoint(X + Y), adjoint_style="hermitian") == r'\left(X + Y\right)^{\mathsf{H}}'
    assert latex(Adjoint(X) + Adjoint(Y), adjoint_style="dagger") == r'X^{\dagger} + Y^{\dagger}'
    assert latex(Adjoint(Y)*Adjoint(X)) == r'Y^{\dagger} X^{\dagger}'
    assert latex(Adjoint(X**2), adjoint_style="star") == r'\left(X^{2}\right)^{\ast}'
    assert latex(Adjoint(X)**2, adjoint_style="hermitian") == r'\left(X^{\mathsf{H}}\right)^{2}'

def test_Transpose():
    from sympy.matrices import Transpose, MatPow, HadamardPower
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(Transpose(X)) == r'X^{T}'
    assert latex(Transpose(X + Y)) == r'\left(X + Y\right)^{T}'

    assert latex(Transpose(HadamardPower(X, 2))) == r'\left(X^{\circ {2}}\right)^{T}'
    assert latex(HadamardPower(Transpose(X), 2)) == r'\left(X^{T}\right)^{\circ {2}}'
    assert latex(Transpose(MatPow(X, 2))) == r'\left(X^{2}\right)^{T}'
    assert latex(MatPow(Transpose(X), 2)) == r'\left(X^{T}\right)^{2}'
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Transpose(m)) == '\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{T}'
    assert latex(Transpose(m+X)) == \
        '\\left(\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X\\right)^{T}'
    from sympy.matrices import BlockMatrix, OneMatrix, ZeroMatrix
    assert latex(Transpose(BlockMatrix(((OneMatrix(2, 2), X),
                                        (m, ZeroMatrix(2, 2)))))) == \
        '\\left[\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}\\right]^{T}'
    # Issue 20959
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(Transpose(Mx)) == r'\left(M^{x}\right)^{T}'


def test_Hadamard():
    from sympy.matrices import HadamardProduct, HadamardPower
    from sympy.matrices.expressions import MatAdd, MatMul, MatPow
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(HadamardProduct(X, Y*Y)) == r'X \circ Y^{2}'
    assert latex(HadamardProduct(X, Y)*Y) == r'\left(X \circ Y\right) Y'

    assert latex(HadamardPower(X, 2)) == r'X^{\circ {2}}'
    assert latex(HadamardPower(X, -1)) == r'X^{\circ \left({-1}\right)}'
    assert latex(HadamardPower(MatAdd(X, Y), 2)) == \
        r'\left(X + Y\right)^{\circ {2}}'
    assert latex(HadamardPower(MatMul(X, Y), 2)) == \
        r'\left(X Y\right)^{\circ {2}}'

    assert latex(HadamardPower(MatPow(X, -1), -1)) == \
        r'\left(X^{-1}\right)^{\circ \left({-1}\right)}'
    assert latex(MatPow(HadamardPower(X, -1), -1)) == \
        r'\left(X^{\circ \left({-1}\right)}\right)^{-1}'

    assert latex(HadamardPower(X, n+1)) == \
        r'X^{\circ \left({n + 1}\right)}'


def test_MatPow():
    from sympy.matrices.expressions import MatPow
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(MatPow(X, 2)) == 'X^{2}'
    assert latex(MatPow(X*X, 2)) == '\\left(X^{2}\\right)^{2}'
    assert latex(MatPow(X*Y, 2)) == '\\left(X Y\\right)^{2}'
    assert latex(MatPow(X + Y, 2)) == '\\left(X + Y\\right)^{2}'
    assert latex(MatPow(X + X, 2)) == '\\left(2 X\\right)^{2}'
    # Issue 20959
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(MatPow(Mx, 2)) == r'\left(M^{x}\right)^{2}'


def test_ElementwiseApplyFunction():
    X = MatrixSymbol('X', 2, 2)
    expr = (X.T*X).applyfunc(sin)
    assert latex(expr) == r"{\left( d \mapsto \sin{\left(d \right)} \right)}_{\circ}\left({X^{T} X}\right)"
    expr = X.applyfunc(Lambda(x, 1/x))
    assert latex(expr) == r'{\left( x \mapsto \frac{1}{x} \right)}_{\circ}\left({X}\right)'


def test_ZeroMatrix():
    from sympy.matrices.expressions.special import ZeroMatrix
    assert latex(ZeroMatrix(1, 1), mat_symbol_style='plain') == r"0"
    assert latex(ZeroMatrix(1, 1), mat_symbol_style='bold') == r"\mathbf{0}"


def test_OneMatrix():
    from sympy.matrices.expressions.special import OneMatrix
    assert latex(OneMatrix(3, 4), mat_symbol_style='plain') == r"1"
    assert latex(OneMatrix(3, 4), mat_symbol_style='bold') == r"\mathbf{1}"


def test_Identity():
    from sympy.matrices.expressions.special import Identity
    assert latex(Identity(1), mat_symbol_style='plain') == r"\mathbb{I}"
    assert latex(Identity(1), mat_symbol_style='bold') == r"\mathbf{I}"


def test_latex_DFT_IDFT():
    from sympy.matrices.expressions.fourier import DFT, IDFT
    assert latex(DFT(13)) == r"\text{DFT}_{13}"
    assert latex(IDFT(x)) == r"\text{IDFT}_{x}"


def test_boolean_args_order():
    syms = symbols('a:f')

    expr = And(*syms)
    assert latex(expr) == r'a \wedge b \wedge c \wedge d \wedge e \wedge f'

    expr = Or(*syms)
    assert latex(expr) == r'a \vee b \vee c \vee d \vee e \vee f'

    expr = Equivalent(*syms)
    assert latex(expr) == \
        r'a \Leftrightarrow b \Leftrightarrow c \Leftrightarrow d \Leftrightarrow e \Leftrightarrow f'

    expr = Xor(*syms)
    assert latex(expr) == \
        r'a \veebar b \veebar c \veebar d \veebar e \veebar f'


def test_imaginary():
    i = sqrt(-1)
    assert latex(i) == r'i'


def test_builtins_without_args():
    assert latex(sin) == r'\sin'
    assert latex(cos) == r'\cos'
    assert latex(tan) == r'\tan'
    assert latex(log) == r'\log'
    assert latex(Ei) == r'\operatorname{Ei}'
    assert latex(zeta) == r'\zeta'


def test_latex_greek_functions():
    # bug because capital greeks that have roman equivalents should not use
    # \Alpha, \Beta, \Eta, etc.
    s = Function('Alpha')
    assert latex(s) == r'\mathrm{A}'
    assert latex(s(x)) == r'\mathrm{A}{\left(x \right)}'
    s = Function('Beta')
    assert latex(s) == r'\mathrm{B}'
    s = Function('Eta')
    assert latex(s) == r'\mathrm{H}'
    assert latex(s(x)) == r'\mathrm{H}{\left(x \right)}'

    # bug because sympy.core.numbers.Pi is special
    p = Function('Pi')
    # assert latex(p(x)) == r'\Pi{\left(x \right)}'
    assert latex(p) == r'\Pi'

    # bug because not all greeks are included
    c = Function('chi')
    assert latex(c(x)) == r'\chi{\left(x \right)}'
    assert latex(c) == r'\chi'


def test_translate():
    s = 'Alpha'
    assert translate(s) == r'\mathrm{A}'
    s = 'Beta'
    assert translate(s) == r'\mathrm{B}'
    s = 'Eta'
    assert translate(s) == r'\mathrm{H}'
    s = 'omicron'
    assert translate(s) == r'o'
    s = 'Pi'
    assert translate(s) == r'\Pi'
    s = 'pi'
    assert translate(s) == r'\pi'
    s = 'LamdaHatDOT'
    assert translate(s) == r'\dot{\hat{\Lambda}}'


def test_other_symbols():
    from sympy.printing.latex import other_symbols
    for s in other_symbols:
        assert latex(symbols(s)) == r"" "\\" + s


def test_modifiers():
    # Test each modifier individually in the simplest case
    # (with funny capitalizations)
    assert latex(symbols("xMathring")) == r"\mathring{x}"
    assert latex(symbols("xCheck")) == r"\check{x}"
    assert latex(symbols("xBreve")) == r"\breve{x}"
    assert latex(symbols("xAcute")) == r"\acute{x}"
    assert latex(symbols("xGrave")) == r"\grave{x}"
    assert latex(symbols("xTilde")) == r"\tilde{x}"
    assert latex(symbols("xPrime")) == r"{x}'"
    assert latex(symbols("xddDDot")) == r"\ddddot{x}"
    assert latex(symbols("xDdDot")) == r"\dddot{x}"
    assert latex(symbols("xDDot")) == r"\ddot{x}"
    assert latex(symbols("xBold")) == r"\boldsymbol{x}"
    assert latex(symbols("xnOrM")) == r"\left\|{x}\right\|"
    assert latex(symbols("xAVG")) == r"\left\langle{x}\right\rangle"
    assert latex(symbols("xHat")) == r"\hat{x}"
    assert latex(symbols("xDot")) == r"\dot{x}"
    assert latex(symbols("xBar")) == r"\bar{x}"
    assert latex(symbols("xVec")) == r"\vec{x}"
    assert latex(symbols("xAbs")) == r"\left|{x}\right|"
    assert latex(symbols("xMag")) == r"\left|{x}\right|"
    assert latex(symbols("xPrM")) == r"{x}'"
    assert latex(symbols("xBM")) == r"\boldsymbol{x}"
    # Test strings that are *only* the names of modifiers
    assert latex(symbols("Mathring")) == r"Mathring"
    assert latex(symbols("Check")) == r"Check"
    assert latex(symbols("Breve")) == r"Breve"
    assert latex(symbols("Acute")) == r"Acute"
    assert latex(symbols("Grave")) == r"Grave"
    assert latex(symbols("Tilde")) == r"Tilde"
    assert latex(symbols("Prime")) == r"Prime"
    assert latex(symbols("DDot")) == r"\dot{D}"
    assert latex(symbols("Bold")) == r"Bold"
    assert latex(symbols("NORm")) == r"NORm"
    assert latex(symbols("AVG")) == r"AVG"
    assert latex(symbols("Hat")) == r"Hat"
    assert latex(symbols("Dot")) == r"Dot"
    assert latex(symbols("Bar")) == r"Bar"
    assert latex(symbols("Vec")) == r"Vec"
    assert latex(symbols("Abs")) == r"Abs"
    assert latex(symbols("Mag")) == r"Mag"
    assert latex(symbols("PrM")) == r"PrM"
    assert latex(symbols("BM")) == r"BM"
    assert latex(symbols("hbar")) == r"\hbar"
    # Check a few combinations
    assert latex(symbols("xvecdot")) == r"\dot{\vec{x}}"
    assert latex(symbols("xDotVec")) == r"\vec{\dot{x}}"
    assert latex(symbols("xHATNorm")) == r"\left\|{\hat{x}}\right\|"
    # Check a couple big, ugly combinations
    assert latex(symbols('xMathringBm_yCheckPRM__zbreveAbs')) == \
        r"\boldsymbol{\mathring{x}}^{\left|{\breve{z}}\right|}_{{\check{y}}'}"
    assert latex(symbols('alphadothat_nVECDOT__tTildePrime')) == \
        r"\hat{\dot{\alpha}}^{{\tilde{t}}'}_{\dot{\vec{n}}}"


def test_greek_symbols():
    assert latex(Symbol('alpha'))   == r'\alpha'
    assert latex(Symbol('beta'))    == r'\beta'
    assert latex(Symbol('gamma'))   == r'\gamma'
    assert latex(Symbol('delta'))   == r'\delta'
    assert latex(Symbol('epsilon')) == r'\epsilon'
    assert latex(Symbol('zeta'))    == r'\zeta'
    assert latex(Symbol('eta'))     == r'\eta'
    assert latex(Symbol('theta'))   == r'\theta'
    assert latex(Symbol('iota'))    == r'\iota'
    assert latex(Symbol('kappa'))   == r'\kappa'
    assert latex(Symbol('lambda'))  == r'\lambda'
    assert latex(Symbol('mu'))      == r'\mu'
    assert latex(Symbol('nu'))      == r'\nu'
    assert latex(Symbol('xi'))      == r'\xi'
    assert latex(Symbol('omicron')) == r'o'
    assert latex(Symbol('pi'))      == r'\pi'
    assert latex(Symbol('rho'))     == r'\rho'
    assert latex(Symbol('sigma'))   == r'\sigma'
    assert latex(Symbol('tau'))     == r'\tau'
    assert latex(Symbol('upsilon')) == r'\upsilon'
    assert latex(Symbol('phi'))     == r'\phi'
    assert latex(Symbol('chi'))     == r'\chi'
    assert latex(Symbol('psi'))     == r'\psi'
    assert latex(Symbol('omega'))   == r'\omega'

    assert latex(Symbol('Alpha'))   == r'\mathrm{A}'
    assert latex(Symbol('Beta'))    == r'\mathrm{B}'
    assert latex(Symbol('Gamma'))   == r'\Gamma'
    assert latex(Symbol('Delta'))   == r'\Delta'
    assert latex(Symbol('Epsilon')) == r'\mathrm{E}'
    assert latex(Symbol('Zeta'))    == r'\mathrm{Z}'
    assert latex(Symbol('Eta'))     == r'\mathrm{H}'
    assert latex(Symbol('Theta'))   == r'\Theta'
    assert latex(Symbol('Iota'))    == r'\mathrm{I}'
    assert latex(Symbol('Kappa'))   == r'\mathrm{K}'
    assert latex(Symbol('Lambda'))  == r'\Lambda'
    assert latex(Symbol('Mu'))      == r'\mathrm{M}'
    assert latex(Symbol('Nu'))      == r'\mathrm{N}'
    assert latex(Symbol('Xi'))      == r'\Xi'
    assert latex(Symbol('Omicron')) == r'\mathrm{O}'
    assert latex(Symbol('Pi'))      == r'\Pi'
    assert latex(Symbol('Rho'))     == r'\mathrm{P}'
    assert latex(Symbol('Sigma'))   == r'\Sigma'
    assert latex(Symbol('Tau'))     == r'\mathrm{T}'
    assert latex(Symbol('Upsilon')) == r'\Upsilon'
    assert latex(Symbol('Phi'))     == r'\Phi'
    assert latex(Symbol('Chi'))     == r'\mathrm{X}'
    assert latex(Symbol('Psi'))     == r'\Psi'
    assert latex(Symbol('Omega'))   == r'\Omega'

    assert latex(Symbol('varepsilon')) == r'\varepsilon'
    assert latex(Symbol('varkappa')) == r'\varkappa'
    assert latex(Symbol('varphi')) == r'\varphi'
    assert latex(Symbol('varpi')) == r'\varpi'
    assert latex(Symbol('varrho')) == r'\varrho'
    assert latex(Symbol('varsigma')) == r'\varsigma'
    assert latex(Symbol('vartheta')) == r'\vartheta'


def test_fancyset_symbols():
    assert latex(S.Rationals) == r'\mathbb{Q}'
    assert latex(S.Naturals) == r'\mathbb{N}'
    assert latex(S.Naturals0) == r'\mathbb{N}_0'
    assert latex(S.Integers) == r'\mathbb{Z}'
    assert latex(S.Reals) == r'\mathbb{R}'
    assert latex(S.Complexes) == r'\mathbb{C}'


@XFAIL
def test_builtin_without_args_mismatched_names():
    assert latex(CosineTransform) == r'\mathcal{COS}'


def test_builtin_no_args():
    assert latex(Chi) == r'\operatorname{Chi}'
    assert latex(beta) == r'\operatorname{B}'
    assert latex(gamma) == r'\Gamma'
    assert latex(KroneckerDelta) == r'\delta'
    assert latex(DiracDelta) == r'\delta'
    assert latex(lowergamma) == r'\gamma'


def test_issue_6853():
    p = Function('Pi')
    assert latex(p(x)) == r"\Pi{\left(x \right)}"


def test_Mul():
    e = Mul(-2, x + 1, evaluate=False)
    assert latex(e) == r'- 2 \left(x + 1\right)'
    e = Mul(2, x + 1, evaluate=False)
    assert latex(e) == r'2 \left(x + 1\right)'
    e = Mul(S.Half, x + 1, evaluate=False)
    assert latex(e) == r'\frac{x + 1}{2}'
    e = Mul(y, x + 1, evaluate=False)
    assert latex(e) == r'y \left(x + 1\right)'
    e = Mul(-y, x + 1, evaluate=False)
    assert latex(e) == r'- y \left(x + 1\right)'
    e = Mul(-2, x + 1)
    assert latex(e) == r'- 2 x - 2'
    e = Mul(2, x + 1)
    assert latex(e) == r'2 x + 2'


def test_Pow():
    e = Pow(2, 2, evaluate=False)
    assert latex(e) == r'2^{2}'
    assert latex(x**(Rational(-1, 3))) == r'\frac{1}{\sqrt[3]{x}}'
    x2 = Symbol(r'x^2')
    assert latex(x2**2) == r'\left(x^{2}\right)^{2}'
    # Issue 11011
    assert latex(S('1.453e4500')**x) == r'{1.453 \cdot 10^{4500}}^{x}'


def test_issue_7180():
    assert latex(Equivalent(x, y)) == r"x \Leftrightarrow y"
    assert latex(Not(Equivalent(x, y))) == r"x \not\Leftrightarrow y"


def test_issue_8409():
    assert latex(S.Half**n) == r"\left(\frac{1}{2}\right)^{n}"


def test_issue_8470():
    from sympy.parsing.sympy_parser import parse_expr
    e = parse_expr("-B*A", evaluate=False)
    assert latex(e) == r"A \left(- B\right)"


def test_issue_15439():
    x = MatrixSymbol('x', 2, 2)
    y = MatrixSymbol('y', 2, 2)
    assert latex((x * y).subs(y, -y)) == r"x \left(- y\right)"
    assert latex((x * y).subs(y, -2*y)) == r"x \left(- 2 y\right)"
    assert latex((x * y).subs(x, -x)) == r"\left(- x\right) y"


def test_issue_2934():
    assert latex(Symbol(r'\frac{a_1}{b_1}')) == r'\frac{a_1}{b_1}'


def test_issue_10489():
    latexSymbolWithBrace = r'C_{x_{0}}'
    s = Symbol(latexSymbolWithBrace)
    assert latex(s) == latexSymbolWithBrace
    assert latex(cos(s)) == r'\cos{\left(C_{x_{0}} \right)}'


def test_issue_12886():
    m__1, l__1 = symbols('m__1, l__1')
    assert latex(m__1**2 + l__1**2) == \
        r'\left(l^{1}\right)^{2} + \left(m^{1}\right)^{2}'


def test_issue_13559():
    from sympy.parsing.sympy_parser import parse_expr
    expr = parse_expr('5/1', evaluate=False)
    assert latex(expr) == r"\frac{5}{1}"


def test_issue_13651():
    expr = c + Mul(-1, a + b, evaluate=False)
    assert latex(expr) == r"c - \left(a + b\right)"


def test_latex_UnevaluatedExpr():
    x = symbols("x")
    he = UnevaluatedExpr(1/x)
    assert latex(he) == latex(1/x) == r"\frac{1}{x}"
    assert latex(he**2) == r"\left(\frac{1}{x}\right)^{2}"
    assert latex(he + 1) == r"1 + \frac{1}{x}"
    assert latex(x*he) == r"x \frac{1}{x}"


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert latex(A[0, 0]) == r"{A}_{0,0}"
    assert latex(3 * A[0, 0]) == r"3 {A}_{0,0}"

    F = C[0, 0].subs(C, A - B)
    assert latex(F) == r"{\left(A - B\right)}_{0,0}"

    i, j, k = symbols("i j k")
    M = MatrixSymbol("M", k, k)
    N = MatrixSymbol("N", k, k)
    assert latex((M*N)[i, j]) == \
        r'\sum_{i_{1}=0}^{k - 1} {M}_{i,i_{1}} {N}_{i_{1},j}'

    X_a = MatrixSymbol('X_a', 3, 3)
    assert latex(X_a[0, 0]) == r"{X_{a}}_{0,0}"


def test_MatrixSymbol_printing():
    # test cases for issue #14237
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)

    assert latex(-A) == r"- A"
    assert latex(A - A*B - B) == r"A - A B - B"
    assert latex(-A*B - A*B*C - B) == r"- A B - A B C - B"


def test_DotProduct_printing():
    X = MatrixSymbol('X', 3, 1)
    Y = MatrixSymbol('Y', 3, 1)
    a = Symbol('a')
    assert latex(DotProduct(X, Y)) == r"X \cdot Y"
    assert latex(DotProduct(a * X, Y)) == r"a X \cdot Y"
    assert latex(a * DotProduct(X, Y)) == r"a \left(X \cdot Y\right)"


def test_KroneckerProduct_printing():
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 2, 2)
    assert latex(KroneckerProduct(A, B)) == r'A \otimes B'


def test_Series_printing():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert latex(Series(tf1, tf2)) == \
        r'\left(\frac{x y^{2} - z}{- t^{3} + y^{3}}\right) \left(\frac{x - y}{x + y}\right)'
    assert latex(Series(tf1, tf2, tf3)) == \
        r'\left(\frac{x y^{2} - z}{- t^{3} + y^{3}}\right) \left(\frac{x - y}{x + y}\right) \left(\frac{t x^{2} - t^{w} x + w}{t - y}\right)'
    assert latex(Series(-tf2, tf1)) == \
        r'\left(\frac{- x + y}{x + y}\right) \left(\frac{x y^{2} - z}{- t^{3} + y^{3}}\right)'

    M_1 = Matrix([[5/s], [5/(2*s)]])
    T_1 = TransferFunctionMatrix.from_Matrix(M_1, s)
    M_2 = Matrix([[5, 6*s**3]])
    T_2 = TransferFunctionMatrix.from_Matrix(M_2, s)
    # Brackets
    assert latex(T_1*(T_2 + T_2)) == \
        r'\left[\begin{matrix}\frac{5}{s}\\\frac{5}{2 s}\end{matrix}\right]_\tau\cdot\left(\left[\begin{matrix}\frac{5}{1} &' \
        r' \frac{6 s^{3}}{1}\end{matrix}\right]_\tau + \left[\begin{matrix}\frac{5}{1} & \frac{6 s^{3}}{1}\end{matrix}\right]_\tau\right)' \
            == latex(MIMOSeries(MIMOParallel(T_2, T_2), T_1))
    # No Brackets
    M_3 = Matrix([[5, 6], [6, 5/s]])
    T_3 = TransferFunctionMatrix.from_Matrix(M_3, s)
    assert latex(T_1*T_2 + T_3) == r'\left[\begin{matrix}\frac{5}{s}\\\frac{5}{2 s}\end{matrix}\right]_\tau\cdot\left[\begin{matrix}' \
        r'\frac{5}{1} & \frac{6 s^{3}}{1}\end{matrix}\right]_\tau + \left[\begin{matrix}\frac{5}{1} & \frac{6}{1}\\\frac{6}{1} & ' \
            r'\frac{5}{s}\end{matrix}\right]_\tau' == latex(MIMOParallel(MIMOSeries(T_2, T_1), T_3))


def test_TransferFunction_printing():
    tf1 = TransferFunction(x - 1, x + 1, x)
    assert latex(tf1) == r"\frac{x - 1}{x + 1}"
    tf2 = TransferFunction(x + 1, 2 - y, x)
    assert latex(tf2) == r"\frac{x + 1}{2 - y}"
    tf3 = TransferFunction(y, y**2 + 2*y + 3, y)
    assert latex(tf3) == r"\frac{y}{y^{2} + 2 y + 3}"


def test_Parallel_printing():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    assert latex(Parallel(tf1, tf2)) == \
        r'\frac{x y^{2} - z}{- t^{3} + y^{3}} + \frac{x - y}{x + y}'
    assert latex(Parallel(-tf2, tf1)) == \
        r'\frac{- x + y}{x + y} + \frac{x y^{2} - z}{- t^{3} + y^{3}}'

    M_1 = Matrix([[5, 6], [6, 5/s]])
    T_1 = TransferFunctionMatrix.from_Matrix(M_1, s)
    M_2 = Matrix([[5/s, 6], [6, 5/(s - 1)]])
    T_2 = TransferFunctionMatrix.from_Matrix(M_2, s)
    M_3 = Matrix([[6, 5/(s*(s - 1))], [5, 6]])
    T_3 = TransferFunctionMatrix.from_Matrix(M_3, s)
    assert latex(T_1 + T_2 + T_3) == r'\left[\begin{matrix}\frac{5}{1} & \frac{6}{1}\\\frac{6}{1} & \frac{5}{s}\end{matrix}\right]' \
        r'_\tau + \left[\begin{matrix}\frac{5}{s} & \frac{6}{1}\\\frac{6}{1} & \frac{5}{s - 1}\end{matrix}\right]_\tau + \left[\begin{matrix}' \
            r'\frac{6}{1} & \frac{5}{s \left(s - 1\right)}\\\frac{5}{1} & \frac{6}{1}\end{matrix}\right]_\tau' \
                == latex(MIMOParallel(T_1, T_2, T_3)) == latex(MIMOParallel(T_1, MIMOParallel(T_2, T_3))) == latex(MIMOParallel(MIMOParallel(T_1, T_2), T_3))


def test_TransferFunctionMatrix_printing():
    tf1 = TransferFunction(p, p + x, p)
    tf2 = TransferFunction(-s + p, p + s, p)
    tf3 = TransferFunction(p, y**2 + 2*y + 3, p)
    assert latex(TransferFunctionMatrix([[tf1], [tf2]])) == \
        r'\left[\begin{matrix}\frac{p}{p + x}\\\frac{p - s}{p + s}\end{matrix}\right]_\tau'
    assert latex(TransferFunctionMatrix([[tf1, tf2], [tf3, -tf1]])) == \
        r'\left[\begin{matrix}\frac{p}{p + x} & \frac{p - s}{p + s}\\\frac{p}{y^{2} + 2 y + 3} & \frac{\left(-1\right) p}{p + x}\end{matrix}\right]_\tau'


def test_Feedback_printing():
    tf1 = TransferFunction(p, p + x, p)
    tf2 = TransferFunction(-s + p, p + s, p)
    # Negative Feedback (Default)
    assert latex(Feedback(tf1, tf2)) == \
        r'\frac{\frac{p}{p + x}}{\frac{1}{1} + \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
    assert latex(Feedback(tf1*tf2, TransferFunction(1, 1, p))) == \
        r'\frac{\left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}{\frac{1}{1} + \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
    # Positive Feedback
    assert latex(Feedback(tf1, tf2, 1)) == \
        r'\frac{\frac{p}{p + x}}{\frac{1}{1} - \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
    assert latex(Feedback(tf1*tf2, sign=1)) == \
        r'\frac{\left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}{\frac{1}{1} - \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'


def test_MIMOFeedback_printing():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s**2 - 1, s)
    tf3 = TransferFunction(s, s - 1, s)
    tf4 = TransferFunction(s**2, s**2 - 1, s)

    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf4, tf3], [tf2, tf1]])

    # Negative Feedback (Default)
    assert latex(MIMOFeedback(tfm_1, tfm_2)) == \
        r'\left(I_{\tau} + \left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\cdot\left[' \
        r'\begin{matrix}\frac{s^{2}}{s^{2} - 1} & \frac{s}{s - 1}\\\frac{s}{s^{2} - 1} & \frac{1}{s}\end{matrix}\right]_\tau\right)^{-1} \cdot \left[\begin{matrix}' \
        r'\frac{1}{s} & \frac{s}{s^{2} - 1}\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau'

    # Positive Feedback
    assert latex(MIMOFeedback(tfm_1*tfm_2, tfm_1, 1)) == \
        r'\left(I_{\tau} - \left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\cdot\left' \
        r'[\begin{matrix}\frac{s^{2}}{s^{2} - 1} & \frac{s}{s - 1}\\\frac{s}{s^{2} - 1} & \frac{1}{s}\end{matrix}\right]_\tau\cdot\left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}' \
        r'\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\right)^{-1} \cdot \left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}' \
        r'\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\cdot\left[\begin{matrix}\frac{s^{2}}{s^{2} - 1} & \frac{s}{s - 1}\\\frac{s}{s^{2} - 1}' \
        r' & \frac{1}{s}\end{matrix}\right]_\tau'


def test_Quaternion_latex_printing():
    q = Quaternion(x, y, z, t)
    assert latex(q) == r"x + y i + z j + t k"
    q = Quaternion(x, y, z, x*t)
    assert latex(q) == r"x + y i + z j + t x k"
    q = Quaternion(x, y, z, x + t)
    assert latex(q) == r"x + y i + z j + \left(t + x\right) k"


def test_TensorProduct_printing():
    from sympy.tensor.functions import TensorProduct
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    assert latex(TensorProduct(A, B)) == r"A \otimes B"


def test_WedgeProduct_printing():
    from sympy.diffgeom.rn import R2
    from sympy.diffgeom import WedgeProduct
    wp = WedgeProduct(R2.dx, R2.dy)
    assert latex(wp) == r"\operatorname{d}x \wedge \operatorname{d}y"


def test_issue_9216():
    expr_1 = Pow(1, -1, evaluate=False)
    assert latex(expr_1) == r"1^{-1}"

    expr_2 = Pow(1, Pow(1, -1, evaluate=False), evaluate=False)
    assert latex(expr_2) == r"1^{1^{-1}}"

    expr_3 = Pow(3, -2, evaluate=False)
    assert latex(expr_3) == r"\frac{1}{9}"

    expr_4 = Pow(1, -2, evaluate=False)
    assert latex(expr_4) == r"1^{-2}"


def test_latex_printer_tensor():
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, tensor_heads
    L = TensorIndexType("L")
    i, j, k, l = tensor_indices("i j k l", L)
    i0 = tensor_indices("i_0", L)
    A, B, C, D = tensor_heads("A B C D", [L])
    H = TensorHead("H", [L, L])
    K = TensorHead("K", [L, L, L, L])

    assert latex(i) == r"{}^{i}"
    assert latex(-i) == r"{}_{i}"

    expr = A(i)
    assert latex(expr) == r"A{}^{i}"

    expr = A(i0)
    assert latex(expr) == r"A{}^{i_{0}}"

    expr = A(-i)
    assert latex(expr) == r"A{}_{i}"

    expr = -3*A(i)
    assert latex(expr) == r"-3A{}^{i}"

    expr = K(i, j, -k, -i0)
    assert latex(expr) == r"K{}^{ij}{}_{ki_{0}}"

    expr = K(i, -j, -k, i0)
    assert latex(expr) == r"K{}^{i}{}_{jk}{}^{i_{0}}"

    expr = K(i, -j, k, -i0)
    assert latex(expr) == r"K{}^{i}{}_{j}{}^{k}{}_{i_{0}}"

    expr = H(i, -j)
    assert latex(expr) == r"H{}^{i}{}_{j}"

    expr = H(i, j)
    assert latex(expr) == r"H{}^{ij}"

    expr = H(-i, -j)
    assert latex(expr) == r"H{}_{ij}"

    expr = (1+x)*A(i)
    assert latex(expr) == r"\left(x + 1\right)A{}^{i}"

    expr = H(i, -i)
    assert latex(expr) == r"H{}^{L_{0}}{}_{L_{0}}"

    expr = H(i, -j)*A(j)*B(k)
    assert latex(expr) == r"H{}^{i}{}_{L_{0}}A{}^{L_{0}}B{}^{k}"

    expr = A(i) + 3*B(i)
    assert latex(expr) == r"3B{}^{i} + A{}^{i}"

    # Test ``TensorElement``:
    from sympy.tensor.tensor import TensorElement

    expr = TensorElement(K(i, j, k, l), {i: 3, k: 2})
    assert latex(expr) == r'K{}^{i=3,j,k=2,l}'

    expr = TensorElement(K(i, j, k, l), {i: 3})
    assert latex(expr) == r'K{}^{i=3,jkl}'

    expr = TensorElement(K(i, -j, k, l), {i: 3, k: 2})
    assert latex(expr) == r'K{}^{i=3}{}_{j}{}^{k=2,l}'

    expr = TensorElement(K(i, -j, k, -l), {i: 3, k: 2})
    assert latex(expr) == r'K{}^{i=3}{}_{j}{}^{k=2}{}_{l}'

    expr = TensorElement(K(i, j, -k, -l), {i: 3, -k: 2})
    assert latex(expr) == r'K{}^{i=3,j}{}_{k=2,l}'

    expr = TensorElement(K(i, j, -k, -l), {i: 3})
    assert latex(expr) == r'K{}^{i=3,j}{}_{kl}'

    expr = PartialDerivative(A(i), A(i))
    assert latex(expr) == r"\frac{\partial}{\partial {A{}^{L_{0}}}}{A{}^{L_{0}}}"

    expr = PartialDerivative(A(-i), A(-j))
    assert latex(expr) == r"\frac{\partial}{\partial {A{}_{j}}}{A{}_{i}}"

    expr = PartialDerivative(K(i, j, -k, -l), A(m), A(-n))
    assert latex(expr) == r"\frac{\partial^{2}}{\partial {A{}^{m}} \partial {A{}_{n}}}{K{}^{ij}{}_{kl}}"

    expr = PartialDerivative(B(-i) + A(-i), A(-j), A(-n))
    assert latex(expr) == r"\frac{\partial^{2}}{\partial {A{}_{j}} \partial {A{}_{n}}}{\left(A{}_{i} + B{}_{i}\right)}"

    expr = PartialDerivative(3*A(-i), A(-j), A(-n))
    assert latex(expr) == r"\frac{\partial^{2}}{\partial {A{}_{j}} \partial {A{}_{n}}}{\left(3A{}_{i}\right)}"


def test_multiline_latex():
    a, b, c, d, e, f = symbols('a b c d e f')
    expr = -a + 2*b -3*c +4*d -5*e
    expected = r"\begin{eqnarray}" + "\n"\
        r"f & = &- a \nonumber\\" + "\n"\
        r"& & + 2 b \nonumber\\" + "\n"\
        r"& & - 3 c \nonumber\\" + "\n"\
        r"& & + 4 d \nonumber\\" + "\n"\
        r"& & - 5 e " + "\n"\
        r"\end{eqnarray}"
    assert multiline_latex(f, expr, environment="eqnarray") == expected

    expected2 = r'\begin{eqnarray}' + '\n'\
        r'f & = &- a + 2 b \nonumber\\' + '\n'\
        r'& & - 3 c + 4 d \nonumber\\' + '\n'\
        r'& & - 5 e ' + '\n'\
        r'\end{eqnarray}'

    assert multiline_latex(f, expr, 2, environment="eqnarray") == expected2

    expected3 = r'\begin{eqnarray}' + '\n'\
        r'f & = &- a + 2 b - 3 c \nonumber\\'+ '\n'\
        r'& & + 4 d - 5 e ' + '\n'\
        r'\end{eqnarray}'

    assert multiline_latex(f, expr, 3, environment="eqnarray") == expected3

    expected3dots = r'\begin{eqnarray}' + '\n'\
        r'f & = &- a + 2 b - 3 c \dots\nonumber\\'+ '\n'\
        r'& & + 4 d - 5 e ' + '\n'\
        r'\end{eqnarray}'

    assert multiline_latex(f, expr, 3, environment="eqnarray", use_dots=True) == expected3dots

    expected3align = r'\begin{align*}' + '\n'\
        r'f = &- a + 2 b - 3 c \\'+ '\n'\
        r'& + 4 d - 5 e ' + '\n'\
        r'\end{align*}'

    assert multiline_latex(f, expr, 3) == expected3align
    assert multiline_latex(f, expr, 3, environment='align*') == expected3align

    expected2ieee = r'\begin{IEEEeqnarray}{rCl}' + '\n'\
        r'f & = &- a + 2 b \nonumber\\' + '\n'\
        r'& & - 3 c + 4 d \nonumber\\' + '\n'\
        r'& & - 5 e ' + '\n'\
        r'\end{IEEEeqnarray}'

    assert multiline_latex(f, expr, 2, environment="IEEEeqnarray") == expected2ieee

    raises(ValueError, lambda: multiline_latex(f, expr, environment="foo"))

def test_issue_15353():
    a, x = symbols('a x')
    # Obtained from nonlinsolve([(sin(a*x)),cos(a*x)],[x,a])
    sol = ConditionSet(
        Tuple(x, a), Eq(sin(a*x), 0) & Eq(cos(a*x), 0), S.Complexes**2)
    assert latex(sol) == \
        r'\left\{\left( x, \  a\right)\; \middle|\; \left( x, \  a\right) \in ' \
        r'\mathbb{C}^{2} \wedge \sin{\left(a x \right)} = 0 \wedge ' \
        r'\cos{\left(a x \right)} = 0 \right\}'


def test_latex_symbolic_probability():
    mu = symbols("mu")
    sigma = symbols("sigma", positive=True)
    X = Normal("X", mu, sigma)
    assert latex(Expectation(X)) == r'\operatorname{E}\left[X\right]'
    assert latex(Variance(X)) == r'\operatorname{Var}\left(X\right)'
    assert latex(Probability(X > 0)) == r'\operatorname{P}\left(X > 0\right)'
    Y = Normal("Y", mu, sigma)
    assert latex(Covariance(X, Y)) == r'\operatorname{Cov}\left(X, Y\right)'


def test_trace():
    # Issue 15303
    from sympy.matrices.expressions.trace import trace
    A = MatrixSymbol("A", 2, 2)
    assert latex(trace(A)) == r"\operatorname{tr}\left(A \right)"
    assert latex(trace(A**2)) == r"\operatorname{tr}\left(A^{2} \right)"


def test_print_basic():
    # Issue 15303
    from sympy.core.basic import Basic
    from sympy.core.expr import Expr

    # dummy class for testing printing where the function is not
    # implemented in latex.py
    class UnimplementedExpr(Expr):
        def __new__(cls, e):
            return Basic.__new__(cls, e)

    # dummy function for testing
    def unimplemented_expr(expr):
        return UnimplementedExpr(expr).doit()

    # override class name to use superscript / subscript
    def unimplemented_expr_sup_sub(expr):
        result = UnimplementedExpr(expr)
        result.__class__.__name__ = 'UnimplementedExpr_x^1'
        return result

    assert latex(unimplemented_expr(x)) == r'\operatorname{UnimplementedExpr}\left(x\right)'
    assert latex(unimplemented_expr(x**2)) == \
        r'\operatorname{UnimplementedExpr}\left(x^{2}\right)'
    assert latex(unimplemented_expr_sup_sub(x)) == \
        r'\operatorname{UnimplementedExpr^{1}_{x}}\left(x\right)'


def test_MatrixSymbol_bold():
    # Issue #15871
    from sympy.matrices.expressions.trace import trace
    A = MatrixSymbol("A", 2, 2)
    assert latex(trace(A), mat_symbol_style='bold') == \
        r"\operatorname{tr}\left(\mathbf{A} \right)"
    assert latex(trace(A), mat_symbol_style='plain') == \
        r"\operatorname{tr}\left(A \right)"

    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)

    assert latex(-A, mat_symbol_style='bold') == r"- \mathbf{A}"
    assert latex(A - A*B - B, mat_symbol_style='bold') == \
        r"\mathbf{A} - \mathbf{A} \mathbf{B} - \mathbf{B}"
    assert latex(-A*B - A*B*C - B, mat_symbol_style='bold') == \
        r"- \mathbf{A} \mathbf{B} - \mathbf{A} \mathbf{B} \mathbf{C} - \mathbf{B}"

    A_k = MatrixSymbol("A_k", 3, 3)
    assert latex(A_k, mat_symbol_style='bold') == r"\mathbf{A}_{k}"

    A = MatrixSymbol(r"\nabla_k", 3, 3)
    assert latex(A, mat_symbol_style='bold') == r"\mathbf{\nabla}_{k}"

def test_AppliedPermutation():
    p = Permutation(0, 1, 2)
    x = Symbol('x')
    assert latex(AppliedPermutation(p, x)) == \
        r'\sigma_{\left( 0\; 1\; 2\right)}(x)'


def test_PermutationMatrix():
    p = Permutation(0, 1, 2)
    assert latex(PermutationMatrix(p)) == r'P_{\left( 0\; 1\; 2\right)}'
    p = Permutation(0, 3)(1, 2)
    assert latex(PermutationMatrix(p)) == \
        r'P_{\left( 0\; 3\right)\left( 1\; 2\right)}'


def test_issue_21758():
    from sympy.functions.elementary.piecewise import piecewise_fold
    from sympy.series.fourier import FourierSeries
    x = Symbol('x')
    k, n = symbols('k n')
    fo = FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)), SeqFormula(
        Piecewise((-2*pi*cos(n*pi)/n + 2*sin(n*pi)/n**2, (n > -oo) & (n < oo) & Ne(n, 0)),
                  (0, True))*sin(n*x)/pi, (n, 1, oo))))
    assert latex(piecewise_fold(fo)) == '\\begin{cases} 2 \\sin{\\left(x \\right)}' \
            ' - \\sin{\\left(2 x \\right)} + \\frac{2 \\sin{\\left(3 x \\right)}}{3} +' \
            ' \\ldots & \\text{for}\\: n > -\\infty \\wedge n < \\infty \\wedge ' \
                'n \\neq 0 \\\\0 & \\text{otherwise} \\end{cases}'
    assert latex(FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)),
                                                 SeqFormula(0, (n, 1, oo))))) == '0'


def test_imaginary_unit():
    assert latex(1 + I) == r'1 + i'
    assert latex(1 + I, imaginary_unit='i') == r'1 + i'
    assert latex(1 + I, imaginary_unit='j') == r'1 + j'
    assert latex(1 + I, imaginary_unit='foo') == r'1 + foo'
    assert latex(I, imaginary_unit="ti") == r'\text{i}'
    assert latex(I, imaginary_unit="tj") == r'\text{j}'


def test_text_re_im():
    assert latex(im(x), gothic_re_im=True) == r'\Im{\left(x\right)}'
    assert latex(im(x), gothic_re_im=False) == r'\operatorname{im}{\left(x\right)}'
    assert latex(re(x), gothic_re_im=True) == r'\Re{\left(x\right)}'
    assert latex(re(x), gothic_re_im=False) == r'\operatorname{re}{\left(x\right)}'


def test_latex_diffgeom():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential
    from sympy.diffgeom.rn import R2
    x,y = symbols('x y', real=True)
    m = Manifold('M', 2)
    assert latex(m) == r'\text{M}'
    p = Patch('P', m)
    assert latex(p) == r'\text{P}_{\text{M}}'
    rect = CoordSystem('rect', p, [x, y])
    assert latex(rect) == r'\text{rect}^{\text{P}}_{\text{M}}'
    b = BaseScalarField(rect, 0)
    assert latex(b) == r'\mathbf{x}'

    g = Function('g')
    s_field = g(R2.x, R2.y)
    assert latex(Differential(s_field)) == \
        r'\operatorname{d}\left(g{\left(\mathbf{x},\mathbf{y} \right)}\right)'


def test_unit_printing():
    assert latex(5*meter) == r'5 \text{m}'
    assert latex(3*gibibyte) == r'3 \text{gibibyte}'
    assert latex(4*microgram/second) == r'\frac{4 \mu\text{g}}{\text{s}}'
    assert latex(4*micro*gram/second) == r'\frac{4 \mu \text{g}}{\text{s}}'
    assert latex(5*milli*meter) == r'5 \text{m} \text{m}'
    assert latex(milli) == r'\text{m}'


def test_issue_17092():
    x_star = Symbol('x^*')
    assert latex(Derivative(x_star, x_star,2)) == r'\frac{d^{2}}{d \left(x^{*}\right)^{2}} x^{*}'


def test_latex_decimal_separator():

    x, y, z, t = symbols('x y z t')
    k, m, n = symbols('k m n', integer=True)
    f, g, h = symbols('f g h', cls=Function)

    # comma decimal_separator
    assert(latex([1, 2.3, 4.5], decimal_separator='comma') == r'\left[ 1; \  2{,}3; \  4{,}5\right]')
    assert(latex(FiniteSet(1, 2.3, 4.5), decimal_separator='comma') == r'\left\{1; 2{,}3; 4{,}5\right\}')
    assert(latex((1, 2.3, 4.6), decimal_separator = 'comma') == r'\left( 1; \  2{,}3; \  4{,}6\right)')
    assert(latex((1,), decimal_separator='comma') == r'\left( 1;\right)')

    # period decimal_separator
    assert(latex([1, 2.3, 4.5], decimal_separator='period') == r'\left[ 1, \  2.3, \  4.5\right]' )
    assert(latex(FiniteSet(1, 2.3, 4.5), decimal_separator='period') == r'\left\{1, 2.3, 4.5\right\}')
    assert(latex((1, 2.3, 4.6), decimal_separator = 'period') == r'\left( 1, \  2.3, \  4.6\right)')
    assert(latex((1,), decimal_separator='period') == r'\left( 1,\right)')

    # default decimal_separator
    assert(latex([1, 2.3, 4.5]) == r'\left[ 1, \  2.3, \  4.5\right]')
    assert(latex(FiniteSet(1, 2.3, 4.5)) == r'\left\{1, 2.3, 4.5\right\}')
    assert(latex((1, 2.3, 4.6)) == r'\left( 1, \  2.3, \  4.6\right)')
    assert(latex((1,)) == r'\left( 1,\right)')

    assert(latex(Mul(3.4,5.3), decimal_separator = 'comma') == r'18{,}02')
    assert(latex(3.4*5.3, decimal_separator = 'comma') == r'18{,}02')
    x = symbols('x')
    y = symbols('y')
    z = symbols('z')
    assert(latex(x*5.3 + 2**y**3.4 + 4.5 + z, decimal_separator = 'comma') == r'2^{y^{3{,}4}} + 5{,}3 x + z + 4{,}5')

    assert(latex(0.987, decimal_separator='comma') == r'0{,}987')
    assert(latex(S(0.987), decimal_separator='comma') == r'0{,}987')
    assert(latex(.3, decimal_separator='comma') == r'0{,}3')
    assert(latex(S(.3), decimal_separator='comma') == r'0{,}3')


    assert(latex(5.8*10**(-7), decimal_separator='comma') == r'5{,}8 \cdot 10^{-7}')
    assert(latex(S(5.7)*10**(-7), decimal_separator='comma') == r'5{,}7 \cdot 10^{-7}')
    assert(latex(S(5.7*10**(-7)), decimal_separator='comma') == r'5{,}7 \cdot 10^{-7}')

    x = symbols('x')
    assert(latex(1.2*x+3.4, decimal_separator='comma') == r'1{,}2 x + 3{,}4')
    assert(latex(FiniteSet(1, 2.3, 4.5), decimal_separator='period') == r'\left\{1, 2.3, 4.5\right\}')

    # Error Handling tests
    raises(ValueError, lambda: latex([1,2.3,4.5], decimal_separator='non_existing_decimal_separator_in_list'))
    raises(ValueError, lambda: latex(FiniteSet(1,2.3,4.5), decimal_separator='non_existing_decimal_separator_in_set'))
    raises(ValueError, lambda: latex((1,2.3,4.5), decimal_separator='non_existing_decimal_separator_in_tuple'))

def test_Str():
    from sympy.core.symbol import Str
    assert str(Str('x')) == r'x'

def test_latex_escape():
    assert latex_escape(r"~^\&%$#_{}") == "".join([
        r'\textasciitilde',
        r'\textasciicircum',
        r'\textbackslash',
        r'\&',
        r'\%',
        r'\$',
        r'\#',
        r'\_',
        r'\{',
        r'\}',
    ])

def test_emptyPrinter():
    class MyObject:
        def __repr__(self):
            return "<MyObject with {...}>"

    # unknown objects are monospaced
    assert latex(MyObject()) == r"\mathtt{\text{<MyObject with \{...\}>}}"

    # even if they are nested within other objects
    assert latex((MyObject(),)) == r"\left( \mathtt{\text{<MyObject with \{...\}>}},\right)"

def test_global_settings():
    import inspect

    # settings should be visible in the signature of `latex`
    assert inspect.signature(latex).parameters['imaginary_unit'].default == r'i'
    assert latex(I) == r'i'
    try:
        # but changing the defaults...
        LatexPrinter.set_global_settings(imaginary_unit='j')
        # ... should change the signature
        assert inspect.signature(latex).parameters['imaginary_unit'].default == r'j'
        assert latex(I) == r'j'
    finally:
        # there's no public API to undo this, but we need to make sure we do
        # so as not to impact other tests
        del LatexPrinter._global_settings['imaginary_unit']

    # check we really did undo it
    assert inspect.signature(latex).parameters['imaginary_unit'].default == r'i'
    assert latex(I) == r'i'

def test_pickleable():
    # this tests that the _PrintFunction instance is pickleable
    import pickle
    assert pickle.loads(pickle.dumps(latex)) is latex

def test_printing_latex_array_expressions():
    assert latex(ArraySymbol("A", (2, 3, 4))) == "A"
    assert latex(ArrayElement("A", (2, 1/(1-x), 0))) == "{{A}_{2, \\frac{1}{1 - x}, 0}}"
    M = MatrixSymbol("M", 3, 3)
    N = MatrixSymbol("N", 3, 3)
    assert latex(ArrayElement(M*N, [x, 0])) == "{{\\left(M N\\right)}_{x, 0}}"

def test_Array():
    arr = Array(range(10))
    assert latex(arr) == r'\left[\begin{matrix}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9\end{matrix}\right]'

    arr = Array(range(11))
    # fill the empty argument with a bunch of 'c' to avoid latex errors
    assert latex(arr) == r'\left[\begin{array}{ccccccccccc}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\end{array}\right]'

def test_latex_with_unevaluated():
    with evaluate(False):
        assert latex(a * a) == r"a a"


def test_latex_disable_split_super_sub():
    assert latex(Symbol('u^a_b')) == 'u^{a}_{b}'
    assert latex(Symbol('u^a_b'), disable_split_super_sub=False) == 'u^{a}_{b}'
    assert latex(Symbol('u^a_b'), disable_split_super_sub=True) == 'u\\^a\\_b'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chartsheet\relation.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors import (
    Integer,
    Alias
)
from openpyxl.descriptors.excel import Relation
from openpyxl.descriptors.serialisable import Serialisable


class SheetBackgroundPicture(Serialisable):
    tagname = "picture"
    id = Relation()

    def __init__(self, id):
        self.id = id


class DrawingHF(Serialisable):
    id = Relation()
    lho = Integer(allow_none=True)
    leftHeaderOddPages = Alias('lho')
    lhe = Integer(allow_none=True)
    leftHeaderEvenPages = Alias('lhe')
    lhf = Integer(allow_none=True)
    leftHeaderFirstPage = Alias('lhf')
    cho = Integer(allow_none=True)
    centerHeaderOddPages = Alias('cho')
    che = Integer(allow_none=True)
    centerHeaderEvenPages = Alias('che')
    chf = Integer(allow_none=True)
    centerHeaderFirstPage = Alias('chf')
    rho = Integer(allow_none=True)
    rightHeaderOddPages = Alias('rho')
    rhe = Integer(allow_none=True)
    rightHeaderEvenPages = Alias('rhe')
    rhf = Integer(allow_none=True)
    rightHeaderFirstPage = Alias('rhf')
    lfo = Integer(allow_none=True)
    leftFooterOddPages = Alias('lfo')
    lfe = Integer(allow_none=True)
    leftFooterEvenPages = Alias('lfe')
    lff = Integer(allow_none=True)
    leftFooterFirstPage = Alias('lff')
    cfo = Integer(allow_none=True)
    centerFooterOddPages = Alias('cfo')
    cfe = Integer(allow_none=True)
    centerFooterEvenPages = Alias('cfe')
    cff = Integer(allow_none=True)
    centerFooterFirstPage = Alias('cff')
    rfo = Integer(allow_none=True)
    rightFooterOddPages = Alias('rfo')
    rfe = Integer(allow_none=True)
    rightFooterEvenPages = Alias('rfe')
    rff = Integer(allow_none=True)
    rightFooterFirstPage = Alias('rff')

    def __init__(self,
                 id=None,
                 lho=None,
                 lhe=None,
                 lhf=None,
                 cho=None,
                 che=None,
                 chf=None,
                 rho=None,
                 rhe=None,
                 rhf=None,
                 lfo=None,
                 lfe=None,
                 lff=None,
                 cfo=None,
                 cfe=None,
                 cff=None,
                 rfo=None,
                 rfe=None,
                 rff=None,
                 ):
        self.id = id
        self.lho = lho
        self.lhe = lhe
        self.lhf = lhf
        self.cho = cho
        self.che = che
        self.chf = chf
        self.rho = rho
        self.rhe = rhe
        self.rhf = rhf
        self.lfo = lfo
        self.lfe = lfe
        self.lff = lff
        self.cfo = cfo
        self.cfe = cfe
        self.cff = cff
        self.rfo = rfo
        self.rfe = rfe
        self.rff = rff

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\properties.py ===
# Copyright (c) 2010-2024 openpyxl

"""Worksheet Properties"""

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import String, Bool, Typed
from openpyxl.styles.colors import ColorDescriptor


class Outline(Serialisable):

    tagname = "outlinePr"

    applyStyles = Bool(allow_none=True)
    summaryBelow = Bool(allow_none=True)
    summaryRight = Bool(allow_none=True)
    showOutlineSymbols = Bool(allow_none=True)


    def __init__(self,
                 applyStyles=None,
                 summaryBelow=None,
                 summaryRight=None,
                 showOutlineSymbols=None
                 ):
        self.applyStyles = applyStyles
        self.summaryBelow = summaryBelow
        self.summaryRight = summaryRight
        self.showOutlineSymbols = showOutlineSymbols


class PageSetupProperties(Serialisable):

    tagname = "pageSetUpPr"

    autoPageBreaks = Bool(allow_none=True)
    fitToPage = Bool(allow_none=True)

    def __init__(self, autoPageBreaks=None, fitToPage=None):
        self.autoPageBreaks = autoPageBreaks
        self.fitToPage = fitToPage


class WorksheetProperties(Serialisable):

    tagname = "sheetPr"

    codeName = String(allow_none=True)
    enableFormatConditionsCalculation = Bool(allow_none=True)
    filterMode = Bool(allow_none=True)
    published = Bool(allow_none=True)
    syncHorizontal = Bool(allow_none=True)
    syncRef = String(allow_none=True)
    syncVertical = Bool(allow_none=True)
    transitionEvaluation = Bool(allow_none=True)
    transitionEntry = Bool(allow_none=True)
    tabColor = ColorDescriptor(allow_none=True)
    outlinePr = Typed(expected_type=Outline, allow_none=True)
    pageSetUpPr = Typed(expected_type=PageSetupProperties, allow_none=True)

    __elements__ = ('tabColor', 'outlinePr', 'pageSetUpPr')


    def __init__(self,
                 codeName=None,
                 enableFormatConditionsCalculation=None,
                 filterMode=None,
                 published=None,
                 syncHorizontal=None,
                 syncRef=None,
                 syncVertical=None,
                 transitionEvaluation=None,
                 transitionEntry=None,
                 tabColor=None,
                 outlinePr=None,
                 pageSetUpPr=None
                 ):
        """ Attributes """
        self.codeName = codeName
        self.enableFormatConditionsCalculation = enableFormatConditionsCalculation
        self.filterMode = filterMode
        self.published = published
        self.syncHorizontal = syncHorizontal
        self.syncRef = syncRef
        self.syncVertical = syncVertical
        self.transitionEvaluation = transitionEvaluation
        self.transitionEntry = transitionEntry
        """ Elements """
        self.tabColor = tabColor
        if outlinePr is None:
            self.outlinePr = Outline(summaryBelow=True, summaryRight=True)
        else:
            self.outlinePr = outlinePr

        if pageSetUpPr is None:
            pageSetUpPr = PageSetupProperties()
        self.pageSetUpPr = pageSetUpPr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\kind.py ===
# sympy.matrices.kind

from sympy.core.kind import Kind, _NumberKind, NumberKind
from sympy.core.mul import Mul


class MatrixKind(Kind):
    """
    Kind for all matrices in SymPy.

    Basic class for this kind is ``MatrixBase`` and ``MatrixExpr``,
    but any expression representing the matrix can have this.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is
        :class:`sympy.core.kind.NumberKind`,
        which means that the matrix contains only numbers.

    Examples
    ========

    Any instance of matrix class has kind ``MatrixKind``:

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2, 2)
    >>> A.kind
    MatrixKind(NumberKind)

    An expression representing a matrix may not be an instance of
    the Matrix class, but it will have kind ``MatrixKind``:

    >>> from sympy import MatrixExpr, Integral
    >>> from sympy.abc import x
    >>> intM = Integral(A, x)
    >>> isinstance(intM, MatrixExpr)
    False
    >>> intM.kind
    MatrixKind(NumberKind)

    Use ``isinstance()`` to check for ``MatrixKind`` without specifying the
    element kind. Use ``is`` to check the kind including the element kind:

    >>> from sympy import Matrix
    >>> from sympy.core import NumberKind
    >>> from sympy.matrices import MatrixKind
    >>> M = Matrix([1, 2])
    >>> isinstance(M.kind, MatrixKind)
    True
    >>> M.kind is MatrixKind(NumberKind)
    True

    See Also
    ========

    sympy.core.kind.NumberKind
    sympy.core.kind.UndefinedKind
    sympy.core.containers.TupleKind
    sympy.sets.sets.SetKind

    """
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return "MatrixKind(%s)" % self.element_kind


@Mul._kind_dispatcher.register(_NumberKind, MatrixKind)
def num_mat_mul(k1, k2):
    """
    Return MatrixKind. The element kind is selected by recursive dispatching.
    Do not need to dispatch in reversed order because KindDispatcher
    searches for this automatically.
    """
    # Deal with Mul._kind_dispatcher's commutativity
    # XXX: this function is called with either k1 or k2 as MatrixKind because
    # the Mul kind dispatcher is commutative. Maybe it shouldn't be. Need to
    # swap the args here because NumberKind does not have an element_kind
    # attribute.
    if not isinstance(k2, MatrixKind):
        k1, k2 = k2, k1
    elemk = Mul._kind_dispatcher(k1, k2.element_kind)
    return MatrixKind(elemk)


@Mul._kind_dispatcher.register(MatrixKind, MatrixKind)
def mat_mat_mul(k1, k2):
    """
    Return MatrixKind. The element kind is selected by recursive dispatching.
    """
    elemk = Mul._kind_dispatcher(k1.element_kind, k2.element_kind)
    return MatrixKind(elemk)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\__init__.py ===
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on

@doctest_depends_on(modules=('antlr4',))
def parse_autolev(autolev_code, include_numeric=False):
    """Parses Autolev code (version 4.1) to SymPy code.

    Parameters
    =========
    autolev_code : Can be an str or any object with a readlines() method (such as a file handle or StringIO).
    include_numeric : boolean, optional
        If True NumPy, PyDy, or other numeric code is included for numeric evaluation lines in the Autolev code.

    Returns
    =======
    sympy_code : str
        Equivalent SymPy and/or numpy/pydy code as the input code.


    Example (Double Pendulum)
    =========================
    >>> my_al_text = ("MOTIONVARIABLES' Q{2}', U{2}'",
    ... "CONSTANTS L,M,G",
    ... "NEWTONIAN N",
    ... "FRAMES A,B",
    ... "SIMPROT(N, A, 3, Q1)",
    ... "SIMPROT(N, B, 3, Q2)",
    ... "W_A_N>=U1*N3>",
    ... "W_B_N>=U2*N3>",
    ... "POINT O",
    ... "PARTICLES P,R",
    ... "P_O_P> = L*A1>",
    ... "P_P_R> = L*B1>",
    ... "V_O_N> = 0>",
    ... "V2PTS(N, A, O, P)",
    ... "V2PTS(N, B, P, R)",
    ... "MASS P=M, R=M",
    ... "Q1' = U1",
    ... "Q2' = U2",
    ... "GRAVITY(G*N1>)",
    ... "ZERO = FR() + FRSTAR()",
    ... "KANE()",
    ... "INPUT M=1,G=9.81,L=1",
    ... "INPUT Q1=.1,Q2=.2,U1=0,U2=0",
    ... "INPUT TFINAL=10, INTEGSTP=.01",
    ... "CODE DYNAMICS() some_filename.c")
    >>> my_al_text = '\\n'.join(my_al_text)
    >>> from sympy.parsing.autolev import parse_autolev
    >>> print(parse_autolev(my_al_text, include_numeric=True))
    import sympy.physics.mechanics as _me
    import sympy as _sm
    import math as m
    import numpy as _np
    <BLANKLINE>
    q1, q2, u1, u2 = _me.dynamicsymbols('q1 q2 u1 u2')
    q1_d, q2_d, u1_d, u2_d = _me.dynamicsymbols('q1_ q2_ u1_ u2_', 1)
    l, m, g = _sm.symbols('l m g', real=True)
    frame_n = _me.ReferenceFrame('n')
    frame_a = _me.ReferenceFrame('a')
    frame_b = _me.ReferenceFrame('b')
    frame_a.orient(frame_n, 'Axis', [q1, frame_n.z])
    frame_b.orient(frame_n, 'Axis', [q2, frame_n.z])
    frame_a.set_ang_vel(frame_n, u1*frame_n.z)
    frame_b.set_ang_vel(frame_n, u2*frame_n.z)
    point_o = _me.Point('o')
    particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))
    particle_r = _me.Particle('r', _me.Point('r_pt'), _sm.Symbol('m'))
    particle_p.point.set_pos(point_o, l*frame_a.x)
    particle_r.point.set_pos(particle_p.point, l*frame_b.x)
    point_o.set_vel(frame_n, 0)
    particle_p.point.v2pt_theory(point_o,frame_n,frame_a)
    particle_r.point.v2pt_theory(particle_p.point,frame_n,frame_b)
    particle_p.mass = m
    particle_r.mass = m
    force_p = particle_p.mass*(g*frame_n.x)
    force_r = particle_r.mass*(g*frame_n.x)
    kd_eqs = [q1_d - u1, q2_d - u2]
    forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x)), (particle_r.point,particle_r.mass*(g*frame_n.x))]
    kane = _me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u1, u2], kd_eqs = kd_eqs)
    fr, frstar = kane.kanes_equations([particle_p, particle_r], forceList)
    zero = fr+frstar
    from pydy.system import System
    sys = System(kane, constants = {l:1, m:1, g:9.81},
    specifieds={},
    initial_conditions={q1:.1, q2:.2, u1:0, u2:0},
    times = _np.linspace(0.0, 10, 10/.01))
    <BLANKLINE>
    y=sys.integrate()
    <BLANKLINE>
    """

    _autolev = import_module(
        'sympy.parsing.autolev._parse_autolev_antlr',
        import_kwargs={'fromlist': ['X']})

    if _autolev is not None:
        return _autolev.parse_autolev(autolev_code, include_numeric)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\setexpr.py ===
from sympy.core import Expr
from sympy.core.decorators import call_highest_priority, _sympifyit
from .fancysets import ImageSet
from .sets import set_add, set_sub, set_mul, set_div, set_pow, set_function


class SetExpr(Expr):
    """An expression that can take on values of a set.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet
    >>> from sympy.sets.setexpr import SetExpr

    >>> a = SetExpr(Interval(0, 5))
    >>> b = SetExpr(FiniteSet(1, 10))
    >>> (a + b).set
    Union(Interval(1, 6), Interval(10, 15))
    >>> (2*a + b).set
    Interval(1, 20)
    """
    _op_priority = 11.0

    def __new__(cls, setarg):
        return Expr.__new__(cls, setarg)

    set = property(lambda self: self.args[0])

    def _latex(self, printer):
        return r"SetExpr\left({}\right)".format(printer._print(self.set))

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return _setexpr_apply_operation(set_add, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return _setexpr_apply_operation(set_add, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return _setexpr_apply_operation(set_mul, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return _setexpr_apply_operation(set_mul, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return _setexpr_apply_operation(set_sub, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return _setexpr_apply_operation(set_sub, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return _setexpr_apply_operation(set_pow, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return _setexpr_apply_operation(set_pow, other, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return _setexpr_apply_operation(set_div, self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return _setexpr_apply_operation(set_div, other, self)

    def _eval_func(self, func):
        # TODO: this could be implemented straight into `imageset`:
        res = set_function(func, self.set)
        if res is None:
            return SetExpr(ImageSet(func, self.set))
        return SetExpr(res)


def _setexpr_apply_operation(op, x, y):
    if isinstance(x, SetExpr):
        x = x.set
    if isinstance(y, SetExpr):
        y = y.set
    out = op(x, y)
    return SetExpr(out)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\drivers\chrome.py ===
from packaging import version

from webdriver_manager.core.driver import Driver
from webdriver_manager.core.logger import log
from webdriver_manager.core.os_manager import ChromeType
import json


class ChromeDriver(Driver):

    def __init__(
            self,
            name,
            driver_version,
            url,
            latest_release_url,
            http_client,
            os_system_manager,
            chrome_type=ChromeType.GOOGLE
    ):
        super(ChromeDriver, self).__init__(
            name,
            driver_version,
            url,
            latest_release_url,
            http_client,
            os_system_manager
        )
        self._browser_type = chrome_type

    def get_driver_download_url(self, os_type):
        driver_version_to_download = self.get_driver_version_to_download()
        # For Mac ARM CPUs after version 106.0.5249.61 the format of OS type changed
        # to more unified "mac_arm64". For newer versions, it'll be "mac_arm64"
        # by default, for lower versions we replace "mac_arm64" to old format - "mac64_m1".
        if version.parse(driver_version_to_download) < version.parse("106.0.5249.61"):
            os_type = os_type.replace("mac_arm64", "mac64_m1")

        if version.parse(driver_version_to_download) >= version.parse("115"):
            if os_type == "mac64":
                os_type = "mac-x64"
            if os_type in ["mac_64", "mac64_m1", "mac_arm64"]:
                os_type = "mac-arm64"

            modern_version_url = self.get_url_for_version_and_platform(driver_version_to_download, os_type)
            log(f"Modern chrome version {modern_version_url}")
            return modern_version_url

        return f"{self._url}/{driver_version_to_download}/{self.get_name()}_{os_type}.zip"

    def get_browser_type(self):
        return self._browser_type

    def get_latest_release_version(self):
        determined_browser_version = self.get_browser_version_from_os()
        log(f"Get LATEST {self._name} version for {self._browser_type}")
        if determined_browser_version is not None and version.parse(determined_browser_version) >= version.parse("115"):
            url = "https://googlechromelabs.github.io/chrome-for-testing/latest-patch-versions-per-build.json"
            response = self._http_client.get(url)
            response_dict = json.loads(response.text)
            determined_browser_version = response_dict.get("builds").get(determined_browser_version).get("version")
            return determined_browser_version
        elif determined_browser_version is not None:
            # Remove the build version (the last segment) from determined_browser_version for version < 113
            determined_browser_version = ".".join(determined_browser_version.split(".")[:3])
            latest_release_url = f"{self._latest_release_url}_{determined_browser_version}"
        else:
            latest_release_url = self._latest_release_url

        resp = self._http_client.get(url=latest_release_url)
        return resp.text.rstrip()

    def get_url_for_version_and_platform(self, browser_version, platform):
        url = "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"
        response = self._http_client.get(url)
        data = response.json()
        versions = data["versions"]

        if version.parse(browser_version) >= version.parse("115"):
            short_version = ".".join(browser_version.split(".")[:3])
            compatible_versions = [v for v in versions if short_version in v["version"]]
            if compatible_versions:
                latest_version = compatible_versions[-1]
                log(f"WebDriver version {latest_version['version']} selected")
                downloads = latest_version["downloads"]["chromedriver"]
                for d in downloads:
                    if d["platform"] == platform:
                        return d["url"]
        else:
            for v in versions:
                if v["version"] == browser_version:
                    downloads = v["downloads"]["chromedriver"]
                    for d in downloads:
                        if d["platform"] == platform:
                            return d["url"]

        raise Exception(f"No such driver version {browser_version} for {platform}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_wester.py ===
""" Tests from Michael Wester's 1999 paper "Review of CAS mathematical
capabilities".

http://www.math.unm.edu/~wester/cas/book/Wester.pdf
See also http://math.unm.edu/~wester/cas_review.html for detailed output of
each tested system.
"""

from sympy.assumptions.ask import Q, ask
from sympy.assumptions.refine import refine
from sympy.concrete.products import product
from sympy.core import EulerGamma
from sympy.core.evalf import N
from sympy.core.function import (Derivative, Function, Lambda, Subs,
    diff, expand, expand_func)
from sympy.core.mul import Mul
from sympy.core.intfunc import igcd
from sympy.core.numbers import (AlgebraicNumber, E, I, Rational,
    nan, oo, pi, zoo)
from sympy.core.relational import Eq, Lt
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.functions.combinatorial.factorials import (rf, binomial,
    factorial, factorial2)
from sympy.functions.combinatorial.numbers import bernoulli, fibonacci, totient, partition
from sympy.functions.elementary.complexes import (conjugate, im, re,
    sign)
from sympy.functions.elementary.exponential import LambertW, exp, log
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh,
    tanh)
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, asin,
    atan, cos, cot, csc, sec, sin, tan)
from sympy.functions.special.bessel import besselj
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.elliptic_integrals import (elliptic_e,
    elliptic_f)
from sympy.functions.special.gamma_functions import gamma, polygamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.polynomials import (assoc_legendre,
    chebyshevt)
from sympy.functions.special.zeta_functions import polylog
from sympy.geometry.util import idiff
from sympy.logic.boolalg import And
from sympy.matrices.dense import hessian, wronskian
from sympy.matrices.expressions.matmul import MatMul
from sympy.ntheory.continued_fraction import (
    continued_fraction_convergents as cf_c,
    continued_fraction_iterator as cf_i, continued_fraction_periodic as
    cf_p, continued_fraction_reduce as cf_r)
from sympy.ntheory.factor_ import factorint
from sympy.ntheory.generate import primerange
from sympy.polys.domains.integerring import ZZ
from sympy.polys.orthopolys import legendre_poly
from sympy.polys.partfrac import apart
from sympy.polys.polytools import Poly, factor, gcd, resultant
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.residues import residue
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import FiniteSet, Intersection, Interval, Union
from sympy.simplify.combsimp import combsimp
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.radsimp import radsimp
from sympy.simplify.simplify import logcombine, simplify
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.trigsimp import trigsimp
from sympy.solvers.solvers import solve

import mpmath
from sympy.functions.combinatorial.numbers import stirling
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.error_functions import Ci, Si, erf
from sympy.functions.special.zeta_functions import zeta
from sympy.testing.pytest import (XFAIL, slow, SKIP, tooslow, raises)
from sympy.utilities.iterables import partitions
from mpmath import mpi, mpc
from sympy.matrices import Matrix, GramSchmidt, eye
from sympy.matrices.expressions.blockmatrix import BlockMatrix, block_collapse
from sympy.matrices.expressions import MatrixSymbol, ZeroMatrix
from sympy.physics.quantum import Commutator
from sympy.polys.rings import PolyRing
from sympy.polys.fields import FracField
from sympy.polys.solvers import solve_lin_sys
from sympy.concrete import Sum
from sympy.concrete.products import Product
from sympy.integrals import integrate
from sympy.integrals.transforms import laplace_transform,\
    inverse_laplace_transform, LaplaceTransform, fourier_transform,\
    mellin_transform, laplace_correspondence, laplace_initial_conds
from sympy.solvers.recurr import rsolve
from sympy.solvers.solveset import solveset, solveset_real, linsolve
from sympy.solvers.ode import dsolve
from sympy.core.relational import Equality
from itertools import islice, takewhile
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.calculus.util import minimum


EmptySet = S.EmptySet
R = Rational
x, y, z = symbols('x y z')
i, j, k, l, m, n = symbols('i j k l m n', integer=True)
f = Function('f')
g = Function('g')

# A. Boolean Logic and Quantifier Elimination
#   Not implemented.

# B. Set Theory


def test_B1():
    assert (FiniteSet(i, j, j, k, k, k) | FiniteSet(l, k, j) |
            FiniteSet(j, m, j)) == FiniteSet(i, j, k, l, m)


def test_B2():
    assert (FiniteSet(i, j, j, k, k, k) & FiniteSet(l, k, j) &
            FiniteSet(j, m, j)) == Intersection({j, m}, {i, j, k}, {j, k, l})
    # Previous output below. Not sure why that should be the expected output.
    # There should probably be a way to rewrite Intersections that way but I
    # don't see why an Intersection should evaluate like that:
    #
    # == Union({j}, Intersection({m}, Union({j, k}, Intersection({i}, {l}))))


def test_B3():
    assert (FiniteSet(i, j, k, l, m) - FiniteSet(j) ==
            FiniteSet(i, k, l, m))


def test_B4():
    assert (FiniteSet(*(FiniteSet(i, j)*FiniteSet(k, l))) ==
            FiniteSet((i, k), (i, l), (j, k), (j, l)))


# C. Numbers


def test_C1():
    assert (factorial(50) ==
        30414093201713378043612608166064768844377641568960512000000000000)


def test_C2():
    assert (factorint(factorial(50)) == {2: 47, 3: 22, 5: 12, 7: 8,
        11: 4, 13: 3, 17: 2, 19: 2, 23: 2, 29: 1, 31: 1, 37: 1,
        41: 1, 43: 1, 47: 1})


def test_C3():
    assert (factorial2(10), factorial2(9)) == (3840, 945)


# Base conversions; not really implemented by SymPy
# Whatever. Take credit!
def test_C4():
    assert 0xABC == 2748


def test_C5():
    assert 123 == int('234', 7)


def test_C6():
    assert int('677', 8) == int('1BF', 16) == 447


def test_C7():
    assert log(32768, 8) == 5


def test_C8():
    # Modular multiplicative inverse. Would be nice if divmod could do this.
    assert ZZ.invert(5, 7) == 3
    assert ZZ.invert(5, 6) == 5


def test_C9():
    assert igcd(igcd(1776, 1554), 5698) == 74


def test_C10():
    x = 0
    for n in range(2, 11):
        x += R(1, n)
    assert x == R(4861, 2520)


def test_C11():
    assert R(1, 7) == S('0.[142857]')


def test_C12():
    assert R(7, 11) * R(22, 7) == 2


def test_C13():
    test = R(10, 7) * (1 + R(29, 1000)) ** R(1, 3)
    good = 3 ** R(1, 3)
    assert test == good


def test_C14():
    assert sqrtdenest(sqrt(2*sqrt(3) + 4)) == 1 + sqrt(3)


def test_C15():
    test = sqrtdenest(sqrt(14 + 3*sqrt(3 + 2*sqrt(5 - 12*sqrt(3 - 2*sqrt(2))))))
    good = sqrt(2) + 3
    assert test == good


def test_C16():
    test = sqrtdenest(sqrt(10 + 2*sqrt(6) + 2*sqrt(10) + 2*sqrt(15)))
    good = sqrt(2) + sqrt(3) + sqrt(5)
    assert test == good


def test_C17():
    test = radsimp((sqrt(3) + sqrt(2)) / (sqrt(3) - sqrt(2)))
    good = 5 + 2*sqrt(6)
    assert test == good


def test_C18():
    assert simplify((sqrt(-2 + sqrt(-5)) * sqrt(-2 - sqrt(-5))).expand(complex=True)) == 3


@XFAIL
def test_C19():
    assert radsimp(simplify((90 + 34*sqrt(7)) ** R(1, 3))) == 3 + sqrt(7)


def test_C20():
    inside = (135 + 78*sqrt(3))
    test = AlgebraicNumber((inside**R(2, 3) + 3) * sqrt(3) / inside**R(1, 3))
    assert simplify(test) == AlgebraicNumber(12)


def test_C21():
    assert simplify(AlgebraicNumber((41 + 29*sqrt(2)) ** R(1, 5))) == \
        AlgebraicNumber(1 + sqrt(2))


@XFAIL
def test_C22():
    test = simplify(((6 - 4*sqrt(2))*log(3 - 2*sqrt(2)) + (3 - 2*sqrt(2))*log(17
        - 12*sqrt(2)) + 32 - 24*sqrt(2)) / (48*sqrt(2) - 72))
    good = sqrt(2)/3 - log(sqrt(2) - 1)/3
    assert test == good


def test_C23():
    assert 2 * oo - 3 is oo


@XFAIL
def test_C24():
    raise NotImplementedError("2**aleph_null == aleph_1")

# D. Numerical Analysis


def test_D1():
    assert 0.0 / sqrt(2) == 0


def test_D2():
    assert str(exp(-1000000).evalf()) == '3.29683147808856e-434295'


def test_D3():
    assert exp(pi*sqrt(163)).evalf(50).num.ae(262537412640768744)


def test_D4():
    assert floor(R(-5, 3)) == -2
    assert ceiling(R(-5, 3)) == -1


@XFAIL
def test_D5():
    raise NotImplementedError("cubic_spline([1, 2, 4, 5], [1, 4, 2, 3], x)(3) == 27/8")


@XFAIL
def test_D6():
    raise NotImplementedError("translate sum(a[i]*x**i, (i,1,n)) to FORTRAN")


@XFAIL
def test_D7():
    raise NotImplementedError("translate sum(a[i]*x**i, (i,1,n)) to C")


@XFAIL
def test_D8():
    # One way is to cheat by converting the sum to a string,
    # and replacing the '[' and ']' with ''.
    # E.g., horner(S(str(_).replace('[','').replace(']','')))
    raise NotImplementedError("apply Horner's rule to sum(a[i]*x**i, (i,1,5))")


@XFAIL
def test_D9():
    raise NotImplementedError("translate D8 to FORTRAN")


@XFAIL
def test_D10():
    raise NotImplementedError("translate D8 to C")


@XFAIL
def test_D11():
    #Is there a way to use count_ops?
    raise NotImplementedError("flops(sum(product(f[i][k], (i,1,k)), (k,1,n)))")


@XFAIL
def test_D12():
    assert (mpi(-4, 2) * x + mpi(1, 3)) ** 2 == mpi(-8, 16)*x**2 + mpi(-24, 12)*x + mpi(1, 9)


@XFAIL
def test_D13():
    raise NotImplementedError("discretize a PDE: diff(f(x,t),t) == diff(diff(f(x,t),x),x)")

# E. Statistics
#   See scipy; all of this is numerical.

# F. Combinatorial Theory.


def test_F1():
    assert rf(x, 3) == x*(1 + x)*(2 + x)


def test_F2():
    assert expand_func(binomial(n, 3)) == n*(n - 1)*(n - 2)/6


@XFAIL
def test_F3():
    assert combsimp(2**n * factorial(n) * factorial2(2*n - 1)) == factorial(2*n)


@XFAIL
def test_F4():
    assert combsimp(2**n * factorial(n) * product(2*k - 1, (k, 1, n))) == factorial(2*n)


@XFAIL
def test_F5():
    assert gamma(n + R(1, 2)) / sqrt(pi) / factorial(n) == factorial(2*n)/2**(2*n)/factorial(n)**2


def test_F6():
    partTest = [p.copy() for p in partitions(4)]
    partDesired = [{4: 1}, {1: 1, 3: 1}, {2: 2}, {1: 2, 2:1}, {1: 4}]
    assert partTest == partDesired


def test_F7():
    assert partition(4) == 5


def test_F8():
    assert stirling(5, 2, signed=True) == -50  # if signed, then kind=1


def test_F9():
    assert totient(1776) == 576

# G. Number Theory


def test_G1():
    assert list(primerange(999983, 1000004)) == [999983, 1000003]


@XFAIL
def test_G2():
    raise NotImplementedError("find the primitive root of 191 == 19")


@XFAIL
def test_G3():
    raise NotImplementedError("(a+b)**p mod p == a**p + b**p mod p; p prime")

# ... G14 Modular equations are not implemented.

def test_G15():
    assert Rational(sqrt(3).evalf()).limit_denominator(15) == R(26, 15)
    assert list(takewhile(lambda x: x.q <= 15, cf_c(cf_i(sqrt(3)))))[-1] == \
        R(26, 15)


def test_G16():
    assert list(islice(cf_i(pi),10)) == [3, 7, 15, 1, 292, 1, 1, 1, 2, 1]


def test_G17():
    assert cf_p(0, 1, 23) == [4, [1, 3, 1, 8]]


def test_G18():
    assert cf_p(1, 2, 5) == [[1]]
    assert cf_r([[1]]).expand() == S.Half + sqrt(5)/2


@XFAIL
def test_G19():
    s = symbols('s', integer=True, positive=True)
    it = cf_i((exp(1/s) - 1)/(exp(1/s) + 1))
    assert list(islice(it, 5)) == [0, 2*s, 6*s, 10*s, 14*s]


def test_G20():
    s = symbols('s', integer=True, positive=True)
    # Wester erroneously has this as -s + sqrt(s**2 + 1)
    assert cf_r([[2*s]]) == s + sqrt(s**2 + 1)


@XFAIL
def test_G20b():
    s = symbols('s', integer=True, positive=True)
    assert cf_p(s, 1, s**2 + 1) == [[2*s]]


# H. Algebra


def test_H1():
    assert simplify(2*2**n) == simplify(2**(n + 1))
    assert powdenest(2*2**n) == simplify(2**(n + 1))


def test_H2():
    assert powsimp(4 * 2**n) == 2**(n + 2)


def test_H3():
    assert (-1)**(n*(n + 1)) == 1


def test_H4():
    expr = factor(6*x - 10)
    assert type(expr) is Mul
    assert expr.args[0] == 2
    assert expr.args[1] == 3*x - 5

p1 = 64*x**34 - 21*x**47 - 126*x**8 - 46*x**5 - 16*x**60 - 81
p2 = 72*x**60 - 25*x**25 - 19*x**23 - 22*x**39 - 83*x**52 + 54*x**10 + 81
q = 34*x**19 - 25*x**16 + 70*x**7 + 20*x**3 - 91*x - 86


def test_H5():
    assert gcd(p1, p2, x) == 1


def test_H6():
    assert gcd(expand(p1 * q), expand(p2 * q)) == q


def test_H7():
    p1 = 24*x*y**19*z**8 - 47*x**17*y**5*z**8 + 6*x**15*y**9*z**2 - 3*x**22 + 5
    p2 = 34*x**5*y**8*z**13 + 20*x**7*y**7*z**7 + 12*x**9*y**16*z**4 + 80*y**14*z
    assert gcd(p1, p2, x, y, z) == 1


def test_H8():
    p1 = 24*x*y**19*z**8 - 47*x**17*y**5*z**8 + 6*x**15*y**9*z**2 - 3*x**22 + 5
    p2 = 34*x**5*y**8*z**13 + 20*x**7*y**7*z**7 + 12*x**9*y**16*z**4 + 80*y**14*z
    q = 11*x**12*y**7*z**13 - 23*x**2*y**8*z**10 + 47*x**17*y**5*z**8
    assert gcd(p1 * q, p2 * q, x, y, z) == q


def test_H9():
    x = Symbol('x', zero=False)
    p1 = 2*x**(n + 4) - x**(n + 2)
    p2 = 4*x**(n + 1) + 3*x**n
    assert gcd(p1, p2) == x**n


def test_H10():
    p1 = 3*x**4 + 3*x**3 + x**2 - x - 2
    p2 = x**3 - 3*x**2 + x + 5
    assert resultant(p1, p2, x) == 0


def test_H11():
    assert resultant(p1 * q, p2 * q, x) == 0


def test_H12():
    num = x**2 - 4
    den = x**2 + 4*x + 4
    assert simplify(num/den) == (x - 2)/(x + 2)


@XFAIL
def test_H13():
    assert simplify((exp(x) - 1) / (exp(x/2) + 1)) == exp(x/2) - 1


def test_H14():
    p = (x + 1) ** 20
    ep = expand(p)
    assert ep == (1 + 20*x + 190*x**2 + 1140*x**3 + 4845*x**4 + 15504*x**5
        + 38760*x**6 + 77520*x**7 + 125970*x**8 + 167960*x**9 + 184756*x**10
        + 167960*x**11 + 125970*x**12 + 77520*x**13 + 38760*x**14 + 15504*x**15
        + 4845*x**16 + 1140*x**17 + 190*x**18 + 20*x**19 + x**20)
    dep = diff(ep, x)
    assert dep == (20 + 380*x + 3420*x**2 + 19380*x**3 + 77520*x**4
        + 232560*x**5 + 542640*x**6 + 1007760*x**7 + 1511640*x**8 + 1847560*x**9
        + 1847560*x**10 + 1511640*x**11 + 1007760*x**12 + 542640*x**13
        + 232560*x**14 + 77520*x**15 + 19380*x**16 + 3420*x**17 + 380*x**18
        + 20*x**19)
    assert factor(dep) == 20*(1 + x)**19


def test_H15():
    assert simplify(Mul(*[x - r for r in solveset(x**3 + x**2 - 7)])) == x**3 + x**2 - 7


def test_H16():
    assert factor(x**100 - 1) == ((x - 1)*(x + 1)*(x**2 + 1)*(x**4 - x**3
        + x**2 - x + 1)*(x**4 + x**3 + x**2 + x + 1)*(x**8 - x**6 + x**4
        - x**2 + 1)*(x**20 - x**15 + x**10 - x**5 + 1)*(x**20 + x**15 + x**10
        + x**5 + 1)*(x**40 - x**30 + x**20 - x**10 + 1))


def test_H17():
    assert simplify(factor(expand(p1 * p2)) - p1*p2) == 0


@XFAIL
def test_H18():
    # Factor over complex rationals.
    test = factor(4*x**4 + 8*x**3 + 77*x**2 + 18*x + 153)
    good = (2*x + 3*I)*(2*x - 3*I)*(x + 1 - 4*I)*(x + 1 + 4*I)
    assert test == good


def test_H19():
    a = symbols('a')
    # The idea is to let a**2 == 2, then solve 1/(a-1). Answer is a+1")
    assert Poly(a - 1).invert(Poly(a**2 - 2)) == a + 1


@XFAIL
def test_H20():
    raise NotImplementedError("let a**2==2; (x**3 + (a-2)*x**2 - "
        + "(2*a+3)*x - 3*a) / (x**2-2) = (x**2 - 2*x - 3) / (x-a)")


@XFAIL
def test_H21():
    raise NotImplementedError("evaluate (b+c)**4 assuming b**3==2, c**2==3. \
                              Answer is 2*b + 8*c + 18*b**2 + 12*b*c + 9")


def test_H22():
    assert factor(x**4 - 3*x**2 + 1, modulus=5) == (x - 2)**2 * (x + 2)**2


def test_H23():
    f = x**11 + x + 1
    g = (x**2 + x + 1) * (x**9 - x**8 + x**6 - x**5 + x**3 - x**2 + 1)
    assert factor(f, modulus=65537) == g


def test_H24():
    phi = AlgebraicNumber(S.GoldenRatio.expand(func=True), alias='phi')
    assert factor(x**4 - 3*x**2 + 1, extension=phi) == \
        (x - phi)*(x + 1 - phi)*(x - 1 + phi)*(x + phi)


def test_H25():
    e = (x - 2*y**2 + 3*z**3) ** 20
    assert factor(expand(e)) == e


def test_H26():
    g = expand((sin(x) - 2*cos(y)**2 + 3*tan(z)**3)**20)
    assert factor(g, expand=False) == (-sin(x) + 2*cos(y)**2 - 3*tan(z)**3)**20


def test_H27():
    f = 24*x*y**19*z**8 - 47*x**17*y**5*z**8 + 6*x**15*y**9*z**2 - 3*x**22 + 5
    g = 34*x**5*y**8*z**13 + 20*x**7*y**7*z**7 + 12*x**9*y**16*z**4 + 80*y**14*z
    h = -2*z*y**7 \
        *(6*x**9*y**9*z**3 + 10*x**7*z**6 + 17*y*x**5*z**12 + 40*y**7) \
        *(3*x**22 + 47*x**17*y**5*z**8 - 6*x**15*y**9*z**2 - 24*x*y**19*z**8 - 5)
    assert factor(expand(f*g)) == h


@XFAIL
def test_H28():
    raise NotImplementedError("expand ((1 - c**2)**5 * (1 - s**2)**5 * "
        + "(c**2 + s**2)**10) with c**2 + s**2 = 1. Answer is c**10*s**10.")


@XFAIL
def test_H29():
    assert factor(4*x**2 - 21*x*y + 20*y**2, modulus=3) == (x + y)*(x - y)


def test_H30():
    test = factor(x**3 + y**3, extension=sqrt(-3))
    answer = (x + y)*(x + y*(-R(1, 2) - sqrt(3)/2*I))*(x + y*(-R(1, 2) + sqrt(3)/2*I))
    assert answer == test


def test_H31():
    f = (x**2 + 2*x + 3)/(x**3 + 4*x**2 + 5*x + 2)
    g = 2 / (x + 1)**2 - 2 / (x + 1) + 3 / (x + 2)
    assert apart(f) == g


@XFAIL
def test_H32():  # issue 6558
    raise NotImplementedError("[A*B*C - (A*B*C)**(-1)]*A*C*B (product \
                              of a non-commuting product and its inverse)")


def test_H33():
    A, B, C = symbols('A, B, C', commutative=False)
    assert (Commutator(A, Commutator(B, C))
        + Commutator(B, Commutator(C, A))
        + Commutator(C, Commutator(A, B))).doit().expand() == 0


# I. Trigonometry

def test_I1():
    assert tan(pi*R(7, 10)) == -sqrt(1 + 2/sqrt(5))


@XFAIL
def test_I2():
    assert sqrt((1 + cos(6))/2) == -cos(3)


def test_I3():
    assert cos(n*pi) + sin((4*n - 1)*pi/2) == (-1)**n - 1


def test_I4():
    assert refine(cos(pi*cos(n*pi)) + sin(pi/2*cos(n*pi)), Q.integer(n)) == (-1)**n - 1


@XFAIL
def test_I5():
    assert sin((n**5/5 + n**4/2 + n**3/3 - n/30) * pi) == 0


@XFAIL
def test_I6():
    raise NotImplementedError("assuming -3*pi<x<-5*pi/2, abs(cos(x)) == -cos(x), abs(sin(x)) == -sin(x)")


@XFAIL
def test_I7():
    assert cos(3*x)/cos(x) == cos(x)**2 - 3*sin(x)**2


@XFAIL
def test_I8():
    assert cos(3*x)/cos(x) == 2*cos(2*x) - 1


@XFAIL
def test_I9():
    # Supposed to do this with rewrite rules.
    assert cos(3*x)/cos(x) == cos(x)**2 - 3*sin(x)**2


def test_I10():
    assert trigsimp((tan(x)**2 + 1 - cos(x)**-2) / (sin(x)**2 + cos(x)**2 - 1)) is nan


@SKIP("hangs")
@XFAIL
def test_I11():
    assert limit((tan(x)**2 + 1 - cos(x)**-2) / (sin(x)**2 + cos(x)**2 - 1), x, 0) != 0


@XFAIL
def test_I12():
    # This should fail or return nan or something.
    res = diff((tan(x)**2 + 1 - cos(x)**-2) / (sin(x)**2 + cos(x)**2 - 1), x)
    assert res is nan # trigsimp(res) gives nan

# J. Special functions.


def test_J1():
    assert bernoulli(16) == R(-3617, 510)


def test_J2():
    assert diff(elliptic_e(x, y**2), y) == (elliptic_e(x, y**2) - elliptic_f(x, y**2))/y


@XFAIL
def test_J3():
    raise NotImplementedError("Jacobi elliptic functions: diff(dn(u,k), u) == -k**2*sn(u,k)*cn(u,k)")


def test_J4():
    assert gamma(R(-1, 2)) == -2*sqrt(pi)


def test_J5():
    assert polygamma(0, R(1, 3)) == -log(3) - sqrt(3)*pi/6 - EulerGamma - log(sqrt(3))


def test_J6():
    assert mpmath.besselj(2, 1 + 1j).ae(mpc('0.04157988694396212', '0.24739764151330632'))


def test_J7():
    assert simplify(besselj(R(-5,2), pi/2)) == 12/(pi**2)


def test_J8():
    p = besselj(R(3,2), z)
    q = (sin(z)/z - cos(z))/sqrt(pi*z/2)
    assert simplify(expand_func(p) -q) == 0


def test_J9():
    assert besselj(0, z).diff(z) == - besselj(1, z)


def test_J10():
    mu, nu = symbols('mu, nu', integer=True)
    assert assoc_legendre(nu, mu, 0) == 2**mu*sqrt(pi)/gamma((nu - mu)/2 + 1)/gamma((-nu - mu + 1)/2)


def test_J11():
    assert simplify(assoc_legendre(3, 1, x)) == simplify(-R(3, 2)*sqrt(1 - x**2)*(5*x**2 - 1))


@slow
def test_J12():
    assert simplify(chebyshevt(1008, x) - 2*x*chebyshevt(1007, x) + chebyshevt(1006, x)) == 0


def test_J13():
    a = symbols('a', integer=True, negative=False)
    assert chebyshevt(a, -1) == (-1)**a


def test_J14():
    p = hyper([S.Half, S.Half], [R(3, 2)], z**2)
    assert hyperexpand(p) == asin(z)/z


@XFAIL
def test_J15():
    raise NotImplementedError("F((n+2)/2,-(n-2)/2,R(3,2),sin(z)**2) == sin(n*z)/(n*sin(z)*cos(z)); F(.) is hypergeometric function")


@XFAIL
def test_J16():
    raise NotImplementedError("diff(zeta(x), x) @ x=0 == -log(2*pi)/2")


def test_J17():
    assert integrate(f((x + 2)/5)*DiracDelta((x - 2)/3) - g(x)*diff(DiracDelta(x - 1), x), (x, 0, 3)) == 3*f(R(4, 5)) + Subs(Derivative(g(x), x), x, 1)


@XFAIL
def test_J18():
    raise NotImplementedError("define an antisymmetric function")


# K. The Complex Domain

def test_K1():
    z1, z2 = symbols('z1, z2', complex=True)
    assert re(z1 + I*z2) == -im(z2) + re(z1)
    assert im(z1 + I*z2) == im(z1) + re(z2)


def test_K2():
    assert abs(3 - sqrt(7) + I*sqrt(6*sqrt(7) - 15)) == 1


@XFAIL
def test_K3():
    a, b = symbols('a, b', real=True)
    assert simplify(abs(1/(a + I/a + I*b))) == 1/sqrt(a**2 + (I/a + b)**2)


def test_K4():
    assert log(3 + 4*I).expand(complex=True) == log(5) + I*atan(R(4, 3))


def test_K5():
    x, y = symbols('x, y', real=True)
    assert tan(x + I*y).expand(complex=True) == (sin(2*x)/(cos(2*x) +
        cosh(2*y)) + I*sinh(2*y)/(cos(2*x) + cosh(2*y)))


def test_K6():
    assert sqrt(x*y*abs(z)**2)/(sqrt(x)*abs(z)) == sqrt(x*y)/sqrt(x)
    assert sqrt(x*y*abs(z)**2)/(sqrt(x)*abs(z)) != sqrt(y)


def test_K7():
    y = symbols('y', real=True, negative=False)
    expr = sqrt(x*y*abs(z)**2)/(sqrt(x)*abs(z))
    sexpr = simplify(expr)
    assert sexpr == sqrt(y)


def test_K8():
    z = symbols('z', complex=True)
    assert simplify(sqrt(1/z) - 1/sqrt(z)) != 0  # Passes
    z = symbols('z', complex=True, negative=False)
    assert simplify(sqrt(1/z) - 1/sqrt(z)) == 0  # Fails


def test_K9():
    z = symbols('z', positive=True)
    assert simplify(sqrt(1/z) - 1/sqrt(z)) == 0


def test_K10():
    z = symbols('z', negative=True)
    assert simplify(sqrt(1/z) + 1/sqrt(z)) == 0

# This goes up to K25

# L. Determining Zero Equivalence


def test_L1():
    assert sqrt(997) - (997**3)**R(1, 6) == 0


def test_L2():
    assert sqrt(999983) - (999983**3)**R(1, 6) == 0


def test_L3():
    assert simplify((2**R(1, 3) + 4**R(1, 3))**3 - 6*(2**R(1, 3) + 4**R(1, 3)) - 6) == 0


def test_L4():
    assert trigsimp(cos(x)**3 + cos(x)*sin(x)**2 - cos(x)) == 0


@XFAIL
def test_L5():
    assert log(tan(R(1, 2)*x + pi/4)) - asinh(tan(x)) == 0


def test_L6():
    assert (log(tan(x/2 + pi/4)) - asinh(tan(x))).diff(x).subs({x: 0}) == 0


@XFAIL
def test_L7():
    assert simplify(log((2*sqrt(x) + 1)/(sqrt(4*x + 4*sqrt(x) + 1)))) == 0


@XFAIL
def test_L8():
    assert simplify((4*x + 4*sqrt(x) + 1)**(sqrt(x)/(2*sqrt(x) + 1)) \
        *(2*sqrt(x) + 1)**(1/(2*sqrt(x) + 1)) - 2*sqrt(x) - 1) == 0


@XFAIL
def test_L9():
    z = symbols('z', complex=True)
    assert simplify(2**(1 - z)*gamma(z)*zeta(z)*cos(z*pi/2) - pi**2*zeta(1 - z)) == 0

# M. Equations


@XFAIL
def test_M1():
    assert Equality(x, 2)/2 + Equality(1, 1) == Equality(x/2 + 1, 2)


def test_M2():
    # The roots of this equation should all be real. Note that this
    # doesn't test that they are correct.
    sol = solveset(3*x**3 - 18*x**2 + 33*x - 19, x)
    assert all(s.expand(complex=True).is_real for s in sol)


@XFAIL
def test_M5():
    assert solveset(x**6 - 9*x**4 - 4*x**3 + 27*x**2 - 36*x - 23, x) == FiniteSet(2**(1/3) + sqrt(3), 2**(1/3) - sqrt(3), +sqrt(3) - 1/2**(2/3) + I*sqrt(3)/2**(2/3), +sqrt(3) - 1/2**(2/3) - I*sqrt(3)/2**(2/3), -sqrt(3) - 1/2**(2/3) + I*sqrt(3)/2**(2/3), -sqrt(3) - 1/2**(2/3) - I*sqrt(3)/2**(2/3))


def test_M6():
    assert set(solveset(x**7 - 1, x)) == \
        {cos(n*pi*R(2, 7)) + I*sin(n*pi*R(2, 7)) for n in range(0, 7)}
    # The paper asks for exp terms, but sin's and cos's may be acceptable;
    # if the results are simplified, exp terms appear for all but
    # -sin(pi/14) - I*cos(pi/14) and -sin(pi/14) + I*cos(pi/14) which
    # will simplify if you apply the transformation foo.rewrite(exp).expand()


def test_M7():
    # TODO: Replace solve with solveset, as of now test fails for solveset
    assert set(solve(x**8 - 8*x**7 + 34*x**6 - 92*x**5 + 175*x**4 - 236*x**3 +
        226*x**2 - 140*x + 46, x)) == {
        1 - sqrt(2)*I*sqrt(-sqrt(-3 + 4*sqrt(3)) + 3)/2,
        1 - sqrt(2)*sqrt(-3 + I*sqrt(3 + 4*sqrt(3)))/2,
        1 - sqrt(2)*I*sqrt(sqrt(-3 + 4*sqrt(3)) + 3)/2,
        1 - sqrt(2)*sqrt(-3 - I*sqrt(3 + 4*sqrt(3)))/2,
        1 + sqrt(2)*I*sqrt(sqrt(-3 + 4*sqrt(3)) + 3)/2,
        1 + sqrt(2)*sqrt(-3 - I*sqrt(3 + 4*sqrt(3)))/2,
        1 + sqrt(2)*sqrt(-3 + I*sqrt(3 + 4*sqrt(3)))/2,
        1 + sqrt(2)*I*sqrt(-sqrt(-3 + 4*sqrt(3)) + 3)/2,
        }


@XFAIL  # There are an infinite number of solutions.
def test_M8():
    x = Symbol('x')
    z = symbols('z', complex=True)
    assert solveset(exp(2*x) + 2*exp(x) + 1 - z, x, S.Reals) == \
        FiniteSet(log(1 + z - 2*sqrt(z))/2, log(1 + z + 2*sqrt(z))/2)
    # This one could be simplified better (the 1/2 could be pulled into the log
    # as a sqrt, and the function inside the log can be factored as a square,
    # giving [log(sqrt(z) - 1), log(sqrt(z) + 1)]). Also, there should be an
    # infinite number of solutions.
    # x = {log(sqrt(z) - 1), log(sqrt(z) + 1) + i pi} [+ n 2 pi i, + n 2 pi i]
    # where n is an arbitrary integer.  See url of detailed output above.


@XFAIL
def test_M9():
    # x = symbols('x')
    # solutions are 1/2*(1 +/- sqrt(9 + 8*I*pi*n)) for integer n
    raise NotImplementedError("solveset(exp(2-x**2)-exp(-x),x) has complex solutions.")


def test_M10():
    # TODO: Replace solve with solveset when it gives Lambert solution
    assert solve(exp(x) - x, x) == [-LambertW(-1)]


@XFAIL
def test_M11():
    assert solveset(x**x - x, x) == FiniteSet(-1, 1)


def test_M12():
    # TODO: x = [-1, 2*(+/-asinh(1)*I + n*pi}, 3*(pi/6 + n*pi/3)]
    # TODO: Replace solve with solveset, as of now test fails for solveset
    assert solve((x + 1)*(sin(x)**2 + 1)**2*cos(3*x)**3, x) == [
        -1, pi/6, pi/2,
           - I*log(1 + sqrt(2)),      I*log(1 + sqrt(2)),
        pi - I*log(1 + sqrt(2)), pi + I*log(1 + sqrt(2)),
    ]


@XFAIL
def test_M13():
    n = Dummy('n')
    assert solveset_real(sin(x) - cos(x), x) == ImageSet(Lambda(n, n*pi - pi*R(7, 4)), S.Integers)


@XFAIL
def test_M14():
    n = Dummy('n')
    assert solveset_real(tan(x) - 1, x) == ImageSet(Lambda(n, n*pi + pi/4), S.Integers)


def test_M15():
    n = Dummy('n')
    got = solveset(sin(x) - S.Half)
    assert any(got.dummy_eq(i) for i in (
        Union(ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi*R(5, 6)), S.Integers)),
        Union(ImageSet(Lambda(n, 2*n*pi + pi*R(5, 6)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers))))


@XFAIL
def test_M16():
    n = Dummy('n')
    assert solveset(sin(x) - tan(x), x) == ImageSet(Lambda(n, n*pi), S.Integers)


@XFAIL
def test_M17():
    assert solveset_real(asin(x) - atan(x), x) == FiniteSet(0)


@XFAIL
def test_M18():
    assert solveset_real(acos(x) - atan(x), x) == FiniteSet(sqrt((sqrt(5) - 1)/2))


def test_M19():
    # TODO: Replace solve with solveset, as of now test fails for solveset
    assert solve((x - 2)/x**R(1, 3), x) == [2]


def test_M20():
    assert solveset(sqrt(x**2 + 1) - x + 2, x) == EmptySet


def test_M21():
    assert solveset(x + sqrt(x) - 2) == FiniteSet(1)


def test_M22():
    assert solveset(2*sqrt(x) + 3*x**R(1, 4) - 2) == FiniteSet(R(1, 16))


def test_M23():
    x = symbols('x', complex=True)
    # TODO: Replace solve with solveset, as of now test fails for solveset
    assert solve(x - 1/sqrt(1 + x**2)) == [
        -I*sqrt(S.Half + sqrt(5)/2), sqrt(Rational(-1, 2) + sqrt(5)/2)]


def test_M24():
    # TODO: Replace solve with solveset, as of now test fails for solveset
    solution = solve(1 - binomial(m, 2)*2**k, k)
    answer = log(2/(m*(m - 1)), 2)
    assert solution[0].expand() == answer.expand()


def test_M25():
    a, b, c, d = symbols(':d', positive=True)
    x = symbols('x')
    # TODO: Replace solve with solveset, as of now test fails for solveset
    assert solve(a*b**x - c*d**x, x)[0].expand() == (log(c/a)/log(b/d)).expand()


def test_M26():
    # TODO: Replace solve with solveset, as of now test fails for solveset
    assert solve(sqrt(log(x)) - log(sqrt(x))) == [1, exp(4)]


def test_M27():
    x = symbols('x', real=True)
    b = symbols('b', real=True)
    # TODO: Replace solve with solveset which gives both [+/- current answer]
    # note that there is a typo in this test in the wester.pdf; there is no
    # real solution for the equation as it appears in wester.pdf
    assert solve(log(acos(asin(x**R(2, 3) - b)) - 1) + 2, x
        ) == [(b + sin(cos(exp(-2) + 1)))**R(3, 2)]


@XFAIL
def test_M28():
    assert solveset_real(5*x + exp((x - 5)/2) - 8*x**3, x, assume=Q.real(x)) == [-0.784966, -0.016291, 0.802557]


def test_M29():
    x = symbols('x')
    assert solveset(abs(x - 1) - 2, domain=S.Reals) == FiniteSet(-1, 3)


def test_M30():
    # TODO: Replace solve with solveset, as of now
    # solveset doesn't supports assumptions
    # assert solve(abs(2*x + 5) - abs(x - 2),x, assume=Q.real(x)) == [-1, -7]
    assert solveset_real(abs(2*x + 5) - abs(x - 2), x) == FiniteSet(-1, -7)


def test_M31():
    # TODO: Replace solve with solveset, as of now
    # solveset doesn't supports assumptions
    # assert solve(1 - abs(x) - max(-x - 2, x - 2),x, assume=Q.real(x)) == [-3/2, 3/2]
    assert solveset_real(1 - abs(x) - Max(-x - 2, x - 2), x) == FiniteSet(R(-3, 2), R(3, 2))


@XFAIL
def test_M32():
    # TODO: Replace solve with solveset, as of now
    # solveset doesn't supports assumptions
    assert solveset_real(Max(2 - x**2, x)- Max(-x, (x**3)/9), x) == FiniteSet(-1, 3)


@XFAIL
def test_M33():
    # TODO: Replace solve with solveset, as of now
    # solveset doesn't supports assumptions

    # Second answer can be written in another form. The second answer is the root of x**3 + 9*x**2 - 18 = 0 in the interval (-2, -1).
    assert solveset_real(Max(2 - x**2, x) - x**3/9, x) == FiniteSet(-3, -1.554894, 3)


@XFAIL
def test_M34():
    z = symbols('z', complex=True)
    assert solveset((1 + I) * z + (2 - I) * conjugate(z) + 3*I, z) == FiniteSet(2 + 3*I)


def test_M35():
    x, y = symbols('x y', real=True)
    assert linsolve((3*x - 2*y - I*y + 3*I).as_real_imag(), y, x) == FiniteSet((3, 2))


def test_M36():
    # TODO: Replace solve with solveset, as of now
    # solveset doesn't supports solving for function
    # assert solve(f**2 + f - 2, x) == [Eq(f(x), 1), Eq(f(x), -2)]
    assert solveset(f(x)**2 + f(x) - 2, f(x)) == FiniteSet(-2, 1)


def test_M37():
    assert linsolve([x + y + z - 6, 2*x + y + 2*z - 10, x + 3*y + z - 10 ], x, y, z) == \
        FiniteSet((-z + 4, 2, z))


def test_M38():
    a, b, c = symbols('a, b, c')
    domain = FracField([a, b, c], ZZ).to_domain()
    ring = PolyRing('k1:50', domain)
    (k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
    k11, k12, k13, k14, k15, k16, k17, k18, k19, k20,
    k21, k22, k23, k24, k25, k26, k27, k28, k29, k30,
    k31, k32, k33, k34, k35, k36, k37, k38, k39, k40,
    k41, k42, k43, k44, k45, k46, k47, k48, k49) = ring.gens

    system = [
        -b*k8/a + c*k8/a, -b*k11/a + c*k11/a, -b*k10/a + c*k10/a + k2, -k3 - b*k9/a + c*k9/a,
        -b*k14/a + c*k14/a, -b*k15/a + c*k15/a, -b*k18/a + c*k18/a - k2, -b*k17/a + c*k17/a,
        -b*k16/a + c*k16/a + k4, -b*k13/a + c*k13/a - b*k21/a + c*k21/a + b*k5/a - c*k5/a,
        b*k44/a - c*k44/a, -b*k45/a + c*k45/a, -b*k20/a + c*k20/a, -b*k44/a + c*k44/a,
        b*k46/a - c*k46/a, b**2*k47/a**2 - 2*b*c*k47/a**2 + c**2*k47/a**2, k3, -k4,
        -b*k12/a + c*k12/a - a*k6/b + c*k6/b, -b*k19/a + c*k19/a + a*k7/c - b*k7/c,
        b*k45/a - c*k45/a, -b*k46/a + c*k46/a, -k48 + c*k48/a + c*k48/b - c**2*k48/(a*b),
        -k49 + b*k49/a + b*k49/c - b**2*k49/(a*c), a*k1/b - c*k1/b, a*k4/b - c*k4/b,
        a*k3/b - c*k3/b + k9, -k10 + a*k2/b - c*k2/b, a*k7/b - c*k7/b, -k9, k11,
        b*k12/a - c*k12/a + a*k6/b - c*k6/b, a*k15/b - c*k15/b, k10 + a*k18/b - c*k18/b,
        -k11 + a*k17/b - c*k17/b, a*k16/b - c*k16/b, -a*k13/b + c*k13/b + a*k21/b - c*k21/b + a*k5/b - c*k5/b,
        -a*k44/b + c*k44/b, a*k45/b - c*k45/b, a*k14/c - b*k14/c + a*k20/b - c*k20/b,
        a*k44/b - c*k44/b, -a*k46/b + c*k46/b, -k47 + c*k47/a + c*k47/b - c**2*k47/(a*b),
        a*k19/b - c*k19/b, -a*k45/b + c*k45/b, a*k46/b - c*k46/b, a**2*k48/b**2 - 2*a*c*k48/b**2 + c**2*k48/b**2,
        -k49 + a*k49/b + a*k49/c - a**2*k49/(b*c), k16, -k17, -a*k1/c + b*k1/c,
        -k16 - a*k4/c + b*k4/c, -a*k3/c + b*k3/c, k18 - a*k2/c + b*k2/c, b*k19/a - c*k19/a - a*k7/c + b*k7/c,
        -a*k6/c + b*k6/c, -a*k8/c + b*k8/c, -a*k11/c + b*k11/c + k17, -a*k10/c + b*k10/c - k18,
        -a*k9/c + b*k9/c, -a*k14/c + b*k14/c - a*k20/b + c*k20/b, -a*k13/c + b*k13/c + a*k21/c - b*k21/c - a*k5/c + b*k5/c,
        a*k44/c - b*k44/c, -a*k45/c + b*k45/c, -a*k44/c + b*k44/c, a*k46/c - b*k46/c,
        -k47 + b*k47/a + b*k47/c - b**2*k47/(a*c), -a*k12/c + b*k12/c, a*k45/c - b*k45/c,
        -a*k46/c + b*k46/c, -k48 + a*k48/b + a*k48/c - a**2*k48/(b*c),
        a**2*k49/c**2 - 2*a*b*k49/c**2 + b**2*k49/c**2, k8, k11, -k15, k10 - k18,
        -k17, k9, -k16, -k29, k14 - k32, -k21 + k23 - k31, -k24 - k30, -k35, k44,
        -k45, k36, k13 - k23 + k39, -k20 + k38, k25 + k37, b*k26/a - c*k26/a - k34 + k42,
        -2*k44, k45, k46, b*k47/a - c*k47/a, k41, k44, -k46, -b*k47/a + c*k47/a,
        k12 + k24, -k19 - k25, -a*k27/b + c*k27/b - k33, k45, -k46, -a*k48/b + c*k48/b,
        a*k28/c - b*k28/c + k40, -k45, k46, a*k48/b - c*k48/b, a*k49/c - b*k49/c,
        -a*k49/c + b*k49/c, -k1, -k4, -k3, k15, k18 - k2, k17, k16, k22, k25 - k7,
        k24 + k30, k21 + k23 - k31, k28, -k44, k45, -k30 - k6, k20 + k32, k27 + b*k33/a - c*k33/a,
        k44, -k46, -b*k47/a + c*k47/a, -k36, k31 - k39 - k5, -k32 - k38, k19 - k37,
        k26 - a*k34/b + c*k34/b - k42, k44, -2*k45, k46, a*k48/b - c*k48/b,
        a*k35/c - b*k35/c - k41, -k44, k46, b*k47/a - c*k47/a, -a*k49/c + b*k49/c,
        -k40, k45, -k46, -a*k48/b + c*k48/b, a*k49/c - b*k49/c, k1, k4, k3, -k8,
        -k11, -k10 + k2, -k9, k37 + k7, -k14 - k38, -k22, -k25 - k37, -k24 + k6,
        -k13 - k23 + k39, -k28 + b*k40/a - c*k40/a, k44, -k45, -k27, -k44, k46,
        b*k47/a - c*k47/a, k29, k32 + k38, k31 - k39 + k5, -k12 + k30, k35 - a*k41/b + c*k41/b,
        -k44, k45, -k26 + k34 + a*k42/c - b*k42/c, k44, k45, -2*k46, -b*k47/a + c*k47/a,
        -a*k48/b + c*k48/b, a*k49/c - b*k49/c, k33, -k45, k46, a*k48/b - c*k48/b,
        -a*k49/c + b*k49/c
        ]
    solution = {
        k49: 0, k48: 0, k47: 0, k46: 0, k45: 0, k44: 0, k41: 0, k40: 0,
        k38: 0, k37: 0, k36: 0, k35: 0, k33: 0, k32: 0, k30: 0, k29: 0,
        k28: 0, k27: 0, k25: 0, k24: 0, k22: 0, k21: 0, k20: 0, k19: 0,
        k18: 0, k17: 0, k16: 0, k15: 0, k14: 0, k13: 0, k12: 0, k11: 0,
        k10: 0, k9:  0, k8:  0, k7:  0, k6:  0, k5:  0, k4:  0, k3:  0,
        k2:  0, k1:  0,
        k34: b/c*k42, k31: k39, k26: a/c*k42, k23: k39
    }
    assert solve_lin_sys(system, ring) == solution


def test_M39():
    x, y, z = symbols('x y z', complex=True)
    # TODO: Replace solve with solveset, as of now
    # solveset doesn't supports non-linear multivariate
    assert solve([x**2*y + 3*y*z - 4, -3*x**2*z + 2*y**2 + 1, 2*y*z**2 - z**2 - 1 ]) ==\
            [{y: 1, z: 1, x: -1}, {y: 1, z: 1, x: 1},\
             {y: sqrt(2)*I, z: R(1,3) - sqrt(2)*I/3, x: -sqrt(-1 - sqrt(2)*I)},\
             {y: sqrt(2)*I, z: R(1,3) - sqrt(2)*I/3, x: sqrt(-1 - sqrt(2)*I)},\
             {y: -sqrt(2)*I, z: R(1,3) + sqrt(2)*I/3, x: -sqrt(-1 + sqrt(2)*I)},\
             {y: -sqrt(2)*I, z: R(1,3) + sqrt(2)*I/3, x: sqrt(-1 + sqrt(2)*I)}]

# N. Inequalities


def test_N1():
    assert ask(E**pi > pi**E)


@XFAIL
def test_N2():
    x = symbols('x', real=True)
    assert ask(x**4 - x + 1 > 0) is True
    assert ask(x**4 - x + 1 > 1) is False


@XFAIL
def test_N3():
    x = symbols('x', real=True)
    assert ask(And(Lt(-1, x), Lt(x, 1)), abs(x) < 1 )

@XFAIL
def test_N4():
    x, y = symbols('x y', real=True)
    assert ask(2*x**2 > 2*y**2, (x > y) & (y > 0)) is True


@XFAIL
def test_N5():
    x, y, k = symbols('x y k', real=True)
    assert ask(k*x**2 > k*y**2, (x > y) & (y > 0) & (k > 0)) is True


@slow
@XFAIL
def test_N6():
    x, y, k, n = symbols('x y k n', real=True)
    assert ask(k*x**n > k*y**n, (x > y) & (y > 0) & (k > 0) & (n > 0)) is True


@XFAIL
def test_N7():
    x, y = symbols('x y', real=True)
    assert ask(y > 0, (x > 1) & (y >= x - 1)) is True


@XFAIL
@slow
def test_N8():
    x, y, z = symbols('x y z', real=True)
    assert ask(Eq(x, y) & Eq(y, z),
               (x >= y) & (y >= z) & (z >= x))


def test_N9():
    x = Symbol('x')
    assert solveset(abs(x - 1) > 2, domain=S.Reals) == Union(Interval(-oo, -1, False, True),
                                             Interval(3, oo, True))


def test_N10():
    x = Symbol('x')
    p = (x - 1)*(x - 2)*(x - 3)*(x - 4)*(x - 5)
    assert solveset(expand(p) < 0, domain=S.Reals) == Union(Interval(-oo, 1, True, True),
                                            Interval(2, 3, True, True),
                                            Interval(4, 5, True, True))


def test_N11():
    x = Symbol('x')
    assert solveset(6/(x - 3) <= 3, domain=S.Reals) == Union(Interval(-oo, 3, True, True), Interval(5, oo))


def test_N12():
    x = Symbol('x')
    assert solveset(sqrt(x) < 2, domain=S.Reals) == Interval(0, 4, False, True)


def test_N13():
    x = Symbol('x')
    assert solveset(sin(x) < 2, domain=S.Reals) == S.Reals


@XFAIL
def test_N14():
    x = Symbol('x')
    # Gives 'Union(Interval(Integer(0), Mul(Rational(1, 2), pi), false, true),
    #        Interval(Mul(Rational(1, 2), pi), Mul(Integer(2), pi), true, false))'
    # which is not the correct answer, but the provided also seems wrong.
    assert solveset(sin(x) < 1, x, domain=S.Reals) == Union(Interval(-oo, pi/2, True, True),
                                         Interval(pi/2, oo, True, True))


def test_N15():
    r, t = symbols('r t')
    # raises NotImplementedError: only univariate inequalities are supported
    solveset(abs(2*r*(cos(t) - 1) + 1) <= 1, r, S.Reals)


def test_N16():
    r, t = symbols('r t')
    solveset((r**2)*((cos(t) - 4)**2)*sin(t)**2 < 9, r, S.Reals)


@XFAIL
def test_N17():
    # currently only univariate inequalities are supported
    assert solveset((x + y > 0, x - y < 0), (x, y)) == (abs(x) < y)


def test_O1():
    M = Matrix((1 + I, -2, 3*I))
    assert sqrt(expand(M.dot(M.H))) == sqrt(15)


def test_O2():
    assert Matrix((2, 2, -3)).cross(Matrix((1, 3, 1))) == Matrix([[11],
                                                                  [-5],
                                                                  [4]])

# The vector module has no way of representing vectors symbolically (without
# respect to a basis)
@XFAIL
def test_O3():
    # assert (va ^ vb) | (vc ^ vd) == -(va | vc)*(vb | vd) + (va | vd)*(vb | vc)
    raise NotImplementedError("""The vector module has no way of representing
        vectors symbolically (without respect to a basis)""")

def test_O4():
    from sympy.vector import CoordSys3D, Del
    N = CoordSys3D("N")
    delop = Del()
    i, j, k = N.base_vectors()
    x, y, z = N.base_scalars()
    F = i*(x*y*z) + j*((x*y*z)**2) + k*((y**2)*(z**3))
    assert delop.cross(F).doit() == (-2*x**2*y**2*z + 2*y*z**3)*i + x*y*j + (2*x*y**2*z**2 - x*z)*k

@XFAIL
def test_O5():
    #assert grad|(f^g)-g|(grad^f)+f|(grad^g)  == 0
    raise NotImplementedError("""The vector module has no way of representing
        vectors symbolically (without respect to a basis)""")

#testO8-O9 MISSING!!


def test_O10():
    L = [Matrix([2, 3, 5]), Matrix([3, 6, 2]), Matrix([8, 3, 6])]
    assert GramSchmidt(L) == [Matrix([
                              [2],
                              [3],
                              [5]]),
                              Matrix([
                              [R(23, 19)],
                              [R(63, 19)],
                              [R(-47, 19)]]),
                              Matrix([
                              [R(1692, 353)],
                              [R(-1551, 706)],
                              [R(-423, 706)]])]


def test_P1():
    assert Matrix(3, 3, lambda i, j: j - i).diagonal(-1) == Matrix(
        1, 2, [-1, -1])


def test_P2():
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    M.row_del(1)
    M.col_del(2)
    assert M == Matrix([[1, 2],
                        [7, 8]])


def test_P3():
    A = Matrix([
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
        [41, 42, 43, 44]])

    A11 = A[0:3, 1:4]
    A12 = A[(0, 1, 3), (2, 0, 3)]
    A21 = A
    A221 = -A[0:2, 2:4]
    A222 = -A[(3, 0), (2, 1)]
    A22 = BlockMatrix([[A221, A222]]).T
    rows = [[-A11, A12], [A21, A22]]
    raises(ValueError, lambda: BlockMatrix(rows))
    B = Matrix(rows)
    assert B == Matrix([
        [-12, -13, -14, 13, 11, 14],
        [-22, -23, -24, 23, 21, 24],
        [-32, -33, -34, 43, 41, 44],
        [11, 12, 13, 14, -13, -23],
        [21, 22, 23, 24, -14, -24],
        [31, 32, 33, 34, -43, -13],
        [41, 42, 43, 44, -42, -12]])


@XFAIL
def test_P4():
    raise NotImplementedError("Block matrix diagonalization not supported")


def test_P5():
    M = Matrix([[7, 11],
                [3, 8]])
    assert  M % 2 == Matrix([[1, 1],
                             [1, 0]])


def test_P6():
    M = Matrix([[cos(x), sin(x)],
                [-sin(x), cos(x)]])
    assert M.diff(x, 2) == Matrix([[-cos(x), -sin(x)],
                                   [sin(x), -cos(x)]])


def test_P7():
    M = Matrix([[x, y]])*(
        z*Matrix([[1, 3, 5],
                  [2, 4, 6]]) + Matrix([[7, -9, 11],
                                        [-8, 10, -12]]))
    assert M == Matrix([[x*(z + 7) + y*(2*z - 8), x*(3*z - 9) + y*(4*z + 10),
                         x*(5*z + 11) + y*(6*z - 12)]])


def test_P8():
    M = Matrix([[1, -2*I],
                [-3*I, 4]])
    assert M.norm(ord=S.Infinity) == 7


def test_P9():
    a, b, c = symbols('a b c', nonzero=True)
    M = Matrix([[a/(b*c), 1/c, 1/b],
                [1/c, b/(a*c), 1/a],
                [1/b, 1/a, c/(a*b)]])
    assert factor(M.norm('fro')) == (a**2 + b**2 + c**2)/(abs(a)*abs(b)*abs(c))


@XFAIL
def test_P10():
    M = Matrix([[1, 2 + 3*I],
                [f(4 - 5*I), 6]])
    # conjugate(f(4 - 5*i)) is not simplified to f(4+5*I)
    assert M.H == Matrix([[1, f(4 + 5*I)],
                          [2 + 3*I, 6]])


@XFAIL
def test_P11():
    # raises NotImplementedError("Matrix([[x,y],[1,x*y]]).inv()
    #   not simplifying to extract common factor")
    assert Matrix([[x, y],
                   [1, x*y]]).inv() == (1/(x**2 - 1))*Matrix([[x, -1],
                                                              [-1/y, x/y]])


def test_P11_workaround():
    # This test was changed to inverse method ADJ because it depended on the
    # specific form of inverse returned from the 'GE' method which has changed.
    M = Matrix([[x, y], [1, x*y]]).inv('ADJ')
    c = gcd(tuple(M))
    assert MatMul(c, M/c, evaluate=False) == MatMul(c, Matrix([
        [x*y, -y],
        [ -1,  x]]), evaluate=False)


def test_P12():
    A11 = MatrixSymbol('A11', n, n)
    A12 = MatrixSymbol('A12', n, n)
    A22 = MatrixSymbol('A22', n, n)
    B = BlockMatrix([[A11, A12],
                     [ZeroMatrix(n, n), A22]])
    assert block_collapse(B.I) == BlockMatrix([[A11.I, (-1)*A11.I*A12*A22.I],
                                               [ZeroMatrix(n, n), A22.I]])


def test_P13():
    M = Matrix([[1,     x - 2,                         x - 3],
                [x - 1, x**2 - 3*x + 6,       x**2 - 3*x - 2],
                [x - 2, x**2 - 8,       2*(x**2) - 12*x + 14]])
    L, U, _ = M.LUdecomposition()
    assert simplify(L) == Matrix([[1,     0,     0],
                                  [x - 1, 1,     0],
                                  [x - 2, x - 3, 1]])
    assert simplify(U) == Matrix([[1, x - 2, x - 3],
                                  [0,     4, x - 5],
                                  [0,     0, x - 7]])


def test_P14():
    M = Matrix([[1, 2, 3, 1, 3],
                [3, 2, 1, 1, 7],
                [0, 2, 4, 1, 1],
                [1, 1, 1, 1, 4]])
    R, _ = M.rref()
    assert R == Matrix([[1, 0, -1, 0,  2],
                        [0, 1,  2, 0, -1],
                        [0, 0,  0, 1,  3],
                        [0, 0,  0, 0,  0]])


def test_P15():
    M = Matrix([[-1, 3,  7, -5],
                [4, -2,  1,  3],
                [2,  4, 15, -7]])
    assert M.rank() == 2


def test_P16():
    M = Matrix([[2*sqrt(2), 8],
                [6*sqrt(6), 24*sqrt(3)]])
    assert M.rank() == 1


def test_P17():
    t = symbols('t', real=True)
    M=Matrix([
        [sin(2*t), cos(2*t)],
        [2*(1 - (cos(t)**2))*cos(t), (1 - 2*(sin(t)**2))*sin(t)]])
    assert M.rank() == 1


def test_P18():
    M = Matrix([[1,  0, -2, 0],
                [-2, 1,  0, 3],
                [-1, 2, -6, 6]])
    assert M.nullspace() == [Matrix([[2],
                                     [4],
                                     [1],
                                     [0]]),
                             Matrix([[0],
                                     [-3],
                                     [0],
                                     [1]])]


def test_P19():
    w = symbols('w')
    M = Matrix([[1,    1,    1,    1],
                [w,    x,    y,    z],
                [w**2, x**2, y**2, z**2],
                [w**3, x**3, y**3, z**3]])
    assert M.det() == (w**3*x**2*y   - w**3*x**2*z - w**3*x*y**2 + w**3*x*z**2
                       + w**3*y**2*z - w**3*y*z**2 - w**2*x**3*y + w**2*x**3*z
                       + w**2*x*y**3 - w**2*x*z**3 - w**2*y**3*z + w**2*y*z**3
                       + w*x**3*y**2 - w*x**3*z**2 - w*x**2*y**3 + w*x**2*z**3
                       + w*y**3*z**2 - w*y**2*z**3 - x**3*y**2*z + x**3*y*z**2
                       + x**2*y**3*z - x**2*y*z**3 - x*y**3*z**2 + x*y**2*z**3
                       )


@XFAIL
def test_P20():
    raise NotImplementedError("Matrix minimal polynomial not supported")


def test_P21():
    M = Matrix([[5, -3, -7],
                [-2, 1,  2],
                [2, -3, -4]])
    assert M.charpoly(x).as_expr() == x**3 - 2*x**2 - 5*x + 6


def test_P22():
    d = 100
    M = (2 - x)*eye(d)
    assert M.eigenvals() == {-x + 2: d}


def test_P23():
    M = Matrix([
        [2, 1, 0, 0, 0],
        [1, 2, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 2, 1],
        [0, 0, 0, 1, 2]])
    assert M.eigenvals() == {
        S('1'): 1,
        S('2'): 1,
        S('3'): 1,
        S('sqrt(3) + 2'): 1,
        S('-sqrt(3) + 2'): 1}


def test_P24():
    M = Matrix([[611,  196, -192,  407,   -8,  -52,  -49,   29],
                [196,  899,  113, -192,  -71,  -43,   -8,  -44],
                [-192,  113,  899,  196,   61,   49,    8,   52],
                [ 407, -192,  196,  611,    8,   44,   59,  -23],
                [  -8,  -71,   61,    8,  411, -599,  208,  208],
                [ -52,  -43,   49,   44, -599,  411,  208,  208],
                [ -49,   -8,    8,   59,  208,  208,   99, -911],
                [  29,  -44,   52,  -23,  208,  208, -911,   99]])
    assert M.eigenvals() == {
        S('0'): 1,
        S('10*sqrt(10405)'): 1,
        S('100*sqrt(26) + 510'): 1,
        S('1000'): 2,
        S('-100*sqrt(26) + 510'): 1,
        S('-10*sqrt(10405)'): 1,
        S('1020'): 1}


def test_P25():
    MF = N(Matrix([[ 611,  196, -192,  407,   -8,  -52,  -49,   29],
                   [ 196,  899,  113, -192,  -71,  -43,   -8,  -44],
                   [-192,  113,  899,  196,   61,   49,    8,   52],
                   [ 407, -192,  196,  611,    8,   44,   59,  -23],
                   [  -8,  -71,   61,    8,  411, -599,  208,  208],
                   [ -52,  -43,   49,   44, -599,  411,  208,  208],
                   [ -49,   -8,    8,   59,  208,  208,   99, -911],
                   [  29,  -44,   52,  -23,  208,  208, -911,   99]]))

    ev_1 = sorted(MF.eigenvals(multiple=True))
    ev_2 = sorted(
        [-1020.0490184299969, 0.0, 0.09804864072151699, 1000.0, 1000.0,
        1019.9019513592784, 1020.0, 1020.0490184299969])

    for x, y in zip(ev_1, ev_2):
        assert abs(x - y) < 1e-12


def test_P26():
    a0, a1, a2, a3, a4 = symbols('a0 a1 a2 a3 a4')
    M = Matrix([[-a4, -a3, -a2, -a1, -a0,  0,  0,  0,  0],
                [  1,   0,   0,   0,   0,  0,  0,  0,  0],
                [  0,   1,   0,   0,   0,  0,  0,  0,  0],
                [  0,   0,   1,   0,   0,  0,  0,  0,  0],
                [  0,   0,   0,   1,   0,  0,  0,  0,  0],
                [  0,   0,   0,   0,   0, -1, -1,  0,  0],
                [  0,   0,   0,   0,   0,  1,  0,  0,  0],
                [  0,   0,   0,   0,   0,  0,  1, -1, -1],
                [  0,   0,   0,   0,   0,  0,  0,  1,  0]])
    assert M.eigenvals(error_when_incomplete=False) == {
        S('-1/2 - sqrt(3)*I/2'): 2,
        S('-1/2 + sqrt(3)*I/2'): 2}


def test_P27():
    a = symbols('a')
    M = Matrix([[a,  0, 0, 0, 0],
                [0,  0, 0, 0, 1],
                [0,  0, a, 0, 0],
                [0,  0, 0, a, 0],
                [0, -2, 0, 0, 2]])

    assert M.eigenvects() == [
        (a, 3, [
            Matrix([1, 0, 0, 0, 0]),
            Matrix([0, 0, 1, 0, 0]),
            Matrix([0, 0, 0, 1, 0])
        ]),
        (1 - I, 1, [
            Matrix([0, (1 + I)/2, 0, 0, 1])
        ]),
        (1 + I, 1, [
            Matrix([0, (1 - I)/2, 0, 0, 1])
        ]),
    ]


@XFAIL
def test_P28():
    raise NotImplementedError("Generalized eigenvectors not supported \
https://github.com/sympy/sympy/issues/5293")


@XFAIL
def test_P29():
    raise NotImplementedError("Generalized eigenvectors not supported \
https://github.com/sympy/sympy/issues/5293")


def test_P30():
    M = Matrix([[1,  0,  0,  1, -1],
                [0,  1, -2,  3, -3],
                [0,  0, -1,  2, -2],
                [1, -1,  1,  0,  1],
                [1, -1,  1, -1,  2]])
    _, J = M.jordan_form()
    assert J == Matrix([[-1, 0, 0, 0, 0],
                        [0,  1, 1, 0, 0],
                        [0,  0, 1, 0, 0],
                        [0,  0, 0, 1, 1],
                        [0,  0, 0, 0, 1]])


@XFAIL
def test_P31():
    raise NotImplementedError("Smith normal form not implemented")


def test_P32():
    M = Matrix([[1, -2],
                [2, 1]])
    assert exp(M).rewrite(cos).simplify() == Matrix([[E*cos(2), -E*sin(2)],
                                                     [E*sin(2),  E*cos(2)]])


def test_P33():
    w, t = symbols('w t')
    M = Matrix([[0,    1,      0,   0],
                [0,    0,      0, 2*w],
                [0,    0,      0,   1],
                [0, -2*w, 3*w**2,   0]])
    assert exp(M*t).rewrite(cos).expand() == Matrix([
        [1, -3*t + 4*sin(t*w)/w,  6*t*w - 6*sin(t*w), -2*cos(t*w)/w + 2/w],
        [0,      4*cos(t*w) - 3, -6*w*cos(t*w) + 6*w,          2*sin(t*w)],
        [0,  2*cos(t*w)/w - 2/w,     -3*cos(t*w) + 4,          sin(t*w)/w],
        [0,         -2*sin(t*w),        3*w*sin(t*w),            cos(t*w)]])


@XFAIL
def test_P34():
    a, b, c = symbols('a b c', real=True)
    M = Matrix([[a, 1, 0, 0, 0, 0],
                [0, a, 0, 0, 0, 0],
                [0, 0, b, 0, 0, 0],
                [0, 0, 0, c, 1, 0],
                [0, 0, 0, 0, c, 1],
                [0, 0, 0, 0, 0, c]])
    # raises exception, sin(M) not supported. exp(M*I) also not supported
    # https://github.com/sympy/sympy/issues/6218
    assert sin(M) == Matrix([[sin(a), cos(a), 0, 0, 0, 0],
                             [0, sin(a), 0, 0, 0, 0],
                             [0, 0, sin(b), 0, 0, 0],
                             [0, 0, 0, sin(c), cos(c), -sin(c)/2],
                             [0, 0, 0, 0, sin(c), cos(c)],
                             [0, 0, 0, 0, 0, sin(c)]])


@XFAIL
def test_P35():
    M = pi/2*Matrix([[2, 1, 1],
                     [2, 3, 2],
                     [1, 1, 2]])
    # raises exception, sin(M) not supported. exp(M*I) also not supported
    # https://github.com/sympy/sympy/issues/6218
    assert sin(M) == eye(3)


@XFAIL
def test_P36():
    M = Matrix([[10, 7],
                [7, 17]])
    assert sqrt(M) == Matrix([[3, 1],
                              [1, 4]])


def test_P37():
    M = Matrix([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])
    assert M**S.Half == Matrix([[1, R(1, 2), 0],
                                        [0, 1,       0],
                                        [0, 0,       1]])


@XFAIL
def test_P38():
    M=Matrix([[0, 1, 0],
              [0, 0, 0],
              [0, 0, 0]])

    with raises(AssertionError):
        # raises ValueError: Matrix det == 0; not invertible
        M**S.Half
        # if it doesn't raise then this assertion will be
        # raised and the test will be flagged as not XFAILing
        assert None

@XFAIL
def test_P39():
    """
    M=Matrix([
        [1, 1],
        [2, 2],
        [3, 3]])
    M.SVD()
    """
    raise NotImplementedError("Singular value decomposition not implemented")


def test_P40():
    r, t = symbols('r t', real=True)
    M = Matrix([r*cos(t), r*sin(t)])
    assert M.jacobian(Matrix([r, t])) == Matrix([[cos(t), -r*sin(t)],
                                                 [sin(t),  r*cos(t)]])


def test_P41():
    r, t = symbols('r t', real=True)
    assert hessian(r**2*sin(t),(r,t)) == Matrix([[  2*sin(t),   2*r*cos(t)],
                                                 [2*r*cos(t), -r**2*sin(t)]])


def test_P42():
    assert wronskian([cos(x), sin(x)], x).simplify() == 1


def test_P43():
    def __my_jacobian(M, Y):
        return Matrix([M.diff(v).T for v in Y]).T
    r, t = symbols('r t', real=True)
    M = Matrix([r*cos(t), r*sin(t)])
    assert __my_jacobian(M,[r,t]) == Matrix([[cos(t), -r*sin(t)],
                                             [sin(t),  r*cos(t)]])


def test_P44():
    def __my_hessian(f, Y):
        V = Matrix([diff(f, v) for v in Y])
        return  Matrix([V.T.diff(v) for v in Y])
    r, t = symbols('r t', real=True)
    assert __my_hessian(r**2*sin(t), (r, t)) == Matrix([
                                            [  2*sin(t),   2*r*cos(t)],
                                            [2*r*cos(t), -r**2*sin(t)]])


def test_P45():
    def __my_wronskian(Y, v):
        M = Matrix([Matrix(Y).T.diff(x, n) for n in range(0, len(Y))])
        return  M.det()
    assert __my_wronskian([cos(x), sin(x)], x).simplify() == 1

# Q1-Q6  Tensor tests missing


@XFAIL
def test_R1():
    i, j, n = symbols('i j n', integer=True, positive=True)
    xn = MatrixSymbol('xn', n, 1)
    Sm = Sum((xn[i, 0] - Sum(xn[j, 0], (j, 0, n - 1))/n)**2, (i, 0, n - 1))
    # sum does not calculate
    # Unknown result
    Sm.doit()
    raise NotImplementedError('Unknown result')

@XFAIL
def test_R2():
    m, b = symbols('m b')
    i, n = symbols('i n', integer=True, positive=True)
    xn = MatrixSymbol('xn', n, 1)
    yn = MatrixSymbol('yn', n, 1)
    f = Sum((yn[i, 0] - m*xn[i, 0] - b)**2, (i, 0, n - 1))
    f1 = diff(f, m)
    f2 = diff(f, b)
    # raises TypeError: solveset() takes at most 2 arguments (3 given)
    solveset((f1, f2), (m, b), domain=S.Reals)


@XFAIL
def test_R3():
    n, k = symbols('n k', integer=True, positive=True)
    sk = ((-1)**k) * (binomial(2*n, k))**2
    Sm = Sum(sk, (k, 1, oo))
    T = Sm.doit()
    T2 = T.combsimp()
    # returns -((-1)**n*factorial(2*n)
    #           - (factorial(n))**2)*exp_polar(-I*pi)/(factorial(n))**2
    assert T2 == (-1)**n*binomial(2*n, n)


@XFAIL
def test_R4():
# Macsyma indefinite sum test case:
#(c15) /* Check whether the full Gosper algorithm is implemented
#   => 1/2^(n + 1) binomial(n, k - 1) */
#closedform(indefsum(binomial(n, k)/2^n - binomial(n + 1, k)/2^(n + 1), k));
#Time= 2690 msecs
#                      (- n + k - 1) binomial(n + 1, k)
#(d15)               - --------------------------------
#                                     n
#                                   2 2  (n + 1)
#
#(c16) factcomb(makefact(%));
#Time= 220 msecs
#                                 n!
#(d16)                     ----------------
#                                n
#                          2 k! 2  (n - k)!
# Might be possible after fixing https://github.com/sympy/sympy/pull/1879
    raise NotImplementedError("Indefinite sum not supported")


@XFAIL
def test_R5():
    a, b, c, n, k = symbols('a b c n k', integer=True, positive=True)
    sk = ((-1)**k)*(binomial(a + b, a + k)
                    *binomial(b + c, b + k)*binomial(c + a, c + k))
    Sm = Sum(sk, (k, 1, oo))
    T = Sm.doit()  # hypergeometric series not calculated
    assert T == factorial(a+b+c)/(factorial(a)*factorial(b)*factorial(c))


def test_R6():
    n, k = symbols('n k', integer=True, positive=True)
    gn = MatrixSymbol('gn', n + 2, 1)
    Sm = Sum(gn[k, 0] - gn[k - 1, 0], (k, 1, n + 1))
    assert Sm.doit() == -gn[0, 0] + gn[n + 1, 0]


def test_R7():
    n, k = symbols('n k', integer=True, positive=True)
    T = Sum(k**3,(k,1,n)).doit()
    assert T.factor() == n**2*(n + 1)**2/4

@XFAIL
def test_R8():
    n, k = symbols('n k', integer=True, positive=True)
    Sm = Sum(k**2*binomial(n, k), (k, 1, n))
    T = Sm.doit() #returns Piecewise function
    assert T.combsimp() == n*(n + 1)*2**(n - 2)


def test_R9():
    n, k = symbols('n k', integer=True, positive=True)
    Sm = Sum(binomial(n, k - 1)/k, (k, 1, n + 1))
    assert Sm.doit().simplify() == (2**(n + 1) - 1)/(n + 1)


@XFAIL
def test_R10():
    n, m, r, k = symbols('n m r k', integer=True, positive=True)
    Sm = Sum(binomial(n, k)*binomial(m, r - k), (k, 0, r))
    T = Sm.doit()
    T2 = T.combsimp().rewrite(factorial)
    assert T2 == factorial(m + n)/(factorial(r)*factorial(m + n - r))
    assert T2 == binomial(m + n, r).rewrite(factorial)
    # rewrite(binomial) is not working.
    # https://github.com/sympy/sympy/issues/7135
    T3 = T2.rewrite(binomial)
    assert T3 == binomial(m + n, r)


@XFAIL
def test_R11():
    n, k = symbols('n k', integer=True, positive=True)
    sk = binomial(n, k)*fibonacci(k)
    Sm = Sum(sk, (k, 0, n))
    T = Sm.doit()
    # Fibonacci simplification not implemented
    # https://github.com/sympy/sympy/issues/7134
    assert T == fibonacci(2*n)


@XFAIL
def test_R12():
    n, k = symbols('n k', integer=True, positive=True)
    Sm = Sum(fibonacci(k)**2, (k, 0, n))
    T = Sm.doit()
    assert T == fibonacci(n)*fibonacci(n + 1)


@XFAIL
def test_R13():
    n, k = symbols('n k', integer=True, positive=True)
    Sm = Sum(sin(k*x), (k, 1, n))
    T = Sm.doit()  # Sum is not calculated
    assert T.simplify() == cot(x/2)/2 - cos(x*(2*n + 1)/2)/(2*sin(x/2))


@XFAIL
def test_R14():
    n, k = symbols('n k', integer=True, positive=True)
    Sm = Sum(sin((2*k - 1)*x), (k, 1, n))
    T = Sm.doit()  # Sum is not calculated
    assert T.simplify() == sin(n*x)**2/sin(x)


@XFAIL
def test_R15():
    n, k = symbols('n k', integer=True, positive=True)
    Sm = Sum(binomial(n - k, k), (k, 0, floor(n/2)))
    T = Sm.doit()  # Sum is not calculated
    assert T.simplify() == fibonacci(n + 1)


def test_R16():
    k = symbols('k', integer=True, positive=True)
    Sm = Sum(1/k**2 + 1/k**3, (k, 1, oo))
    assert Sm.doit() == zeta(3) + pi**2/6


def test_R17():
    k = symbols('k', integer=True, positive=True)
    assert abs(float(Sum(1/k**2 + 1/k**3, (k, 1, oo)))
               - 2.8469909700078206) < 1e-15


def test_R18():
    k = symbols('k', integer=True, positive=True)
    Sm = Sum(1/(2**k*k**2), (k, 1, oo))
    T = Sm.doit()
    assert T.simplify() == -log(2)**2/2 + pi**2/12


@slow
@XFAIL
def test_R19():
    k = symbols('k', integer=True, positive=True)
    Sm = Sum(1/((3*k + 1)*(3*k + 2)*(3*k + 3)), (k, 0, oo))
    T = Sm.doit()
    # assert fails, T not  simplified
    assert T.simplify() == -log(3)/4 + sqrt(3)*pi/12


@XFAIL
def test_R20():
    n, k = symbols('n k', integer=True, positive=True)
    Sm = Sum(binomial(n, 4*k), (k, 0, oo))
    T = Sm.doit()
    # assert fails, T not  simplified
    assert T.simplify() == 2**(n/2)*cos(pi*n/4)/2 + 2**(n - 1)/2


@XFAIL
def test_R21():
    k = symbols('k', integer=True, positive=True)
    Sm = Sum(1/(sqrt(k*(k + 1)) * (sqrt(k) + sqrt(k + 1))), (k, 1, oo))
    T = Sm.doit()  # Sum not calculated
    assert T.simplify() == 1


# test_R22 answer not available in Wester samples
# Sum(Sum(binomial(n, k)*binomial(n - k, n - 2*k)*x**n*y**(n - 2*k),
#                 (k, 0, floor(n/2))), (n, 0, oo)) with abs(x*y)<1?


@XFAIL
def test_R23():
    n, k = symbols('n k', integer=True, positive=True)
    Sm = Sum(Sum((factorial(n)/(factorial(k)**2*factorial(n - 2*k)))*
                 (x/y)**k*(x*y)**(n - k), (n, 2*k, oo)), (k, 0, oo))
    # Missing how to express constraint abs(x*y)<1?
    T = Sm.doit()  # Sum not calculated
    assert T == -1/sqrt(x**2*y**2 - 4*x**2 - 2*x*y + 1)


def test_R24():
    m, k = symbols('m k', integer=True, positive=True)
    Sm = Sum(Product(k/(2*k - 1), (k, 1, m)), (m, 2, oo))
    assert Sm.doit() == pi/2


def test_S1():
    k = symbols('k', integer=True, positive=True)
    Pr = Product(gamma(k/3), (k, 1, 8))
    assert Pr.doit().simplify() == 640*sqrt(3)*pi**3/6561


def test_S2():
    n, k = symbols('n k', integer=True, positive=True)
    assert Product(k, (k, 1, n)).doit() == factorial(n)


def test_S3():
    n, k = symbols('n k', integer=True, positive=True)
    assert Product(x**k, (k, 1, n)).doit().simplify() == x**(n*(n + 1)/2)


def test_S4():
    n, k = symbols('n k', integer=True, positive=True)
    assert Product(1 + 1/k, (k, 1, n -1)).doit().simplify() == n


def test_S5():
    n, k = symbols('n k', integer=True, positive=True)
    assert (Product((2*k - 1)/(2*k), (k, 1, n)).doit().gammasimp() ==
            gamma(n + S.Half)/(sqrt(pi)*gamma(n + 1)))


@XFAIL
def test_S6():
    n, k = symbols('n k', integer=True, positive=True)
    # Product does not evaluate
    assert (Product(x**2 -2*x*cos(k*pi/n) + 1, (k, 1, n - 1)).doit().simplify()
            == (x**(2*n) - 1)/(x**2 - 1))


@XFAIL
def test_S7():
    k = symbols('k', integer=True, positive=True)
    Pr = Product((k**3 - 1)/(k**3 + 1), (k, 2, oo))
    T = Pr.doit()     # Product does not evaluate
    assert T.simplify() == R(2, 3)


@XFAIL
def test_S8():
    k = symbols('k', integer=True, positive=True)
    Pr = Product(1 - 1/(2*k)**2, (k, 1, oo))
    T = Pr.doit()
    # Product does not evaluate
    assert T.simplify() == 2/pi


@XFAIL
def test_S9():
    k = symbols('k', integer=True, positive=True)
    Pr = Product(1 + (-1)**(k + 1)/(2*k - 1), (k, 1, oo))
    T = Pr.doit()
    # Product produces 0
    # https://github.com/sympy/sympy/issues/7133
    assert T.simplify() == sqrt(2)


@XFAIL
def test_S10():
    k = symbols('k', integer=True, positive=True)
    Pr = Product((k*(k + 1) + 1 + I)/(k*(k + 1) + 1 - I), (k, 0, oo))
    T = Pr.doit()
    # Product does not evaluate
    assert T.simplify() == -1


def test_T1():
    assert limit((1 + 1/n)**n, n, oo) == E
    assert limit((1 - cos(x))/x**2, x, 0) == S.Half


def test_T2():
    assert limit((3**x + 5**x)**(1/x), x, oo) == 5


def test_T3():
    assert limit(log(x)/(log(x) + sin(x)), x, oo) == 1


def test_T4():
    assert limit((exp(x*exp(-x)/(exp(-x) + exp(-2*x**2/(x + 1))))
                 - exp(x))/x, x, oo) == -exp(2)


def test_T5():
    assert  limit(x*log(x)*log(x*exp(x) - x**2)**2/log(log(x**2
                  + 2*exp(exp(3*x**3*log(x))))), x, oo) == R(1, 3)


def test_T6():
    assert limit(1/n * factorial(n)**(1/n), n, oo) == exp(-1)


def test_T7():
    limit(1/n * gamma(n + 1)**(1/n), n, oo)


def test_T8():
    a, z = symbols('a z', positive=True)
    assert limit(gamma(z + a)/gamma(z)*exp(-a*log(z)), z, oo) == 1


@XFAIL
def test_T9():
    z, k = symbols('z k', positive=True)
    # raises NotImplementedError:
    #           Don't know how to calculate the mrv of '(1, k)'
    assert limit(hyper((1, k), (1,), z/k), k, oo) == exp(z)


@XFAIL
def test_T10():
    # No longer raises PoleError, but should return euler-mascheroni constant
    assert limit(zeta(x) - 1/(x - 1), x, 1) == integrate(-1/x + 1/floor(x), (x, 1, oo))

@XFAIL
def test_T11():
    n, k = symbols('n k', integer=True, positive=True)
    # evaluates to 0
    assert limit(n**x/(x*product((1 + x/k), (k, 1, n))), n, oo) == gamma(x)


def test_T12():
    x, t = symbols('x t', real=True)
    # Does not evaluate the limit but returns an expression with erf
    assert limit(x * integrate(exp(-t**2), (t, 0, x))/(1 - exp(-x**2)),
                 x, 0) == 1


def test_T13():
    x = symbols('x', real=True)
    assert [limit(x/abs(x), x, 0, dir='-'),
            limit(x/abs(x), x, 0, dir='+')] == [-1, 1]


def test_T14():
    x = symbols('x', real=True)
    assert limit(atan(-log(x)), x, 0, dir='+') == pi/2


def test_U1():
    x = symbols('x', real=True)
    assert diff(abs(x), x) == sign(x)


def test_U2():
    f = Lambda(x, Piecewise((-x, x < 0), (x, x >= 0)))
    assert diff(f(x), x) == Piecewise((-1, x < 0), (1, x >= 0))


def test_U3():
    f = Lambda(x, Piecewise((x**2 - 1, x == 1), (x**3, x != 1)))
    f1 = Lambda(x, diff(f(x), x))
    assert f1(x) == 3*x**2
    assert f1(1) == 3


@XFAIL
def test_U4():
    n = symbols('n', integer=True, positive=True)
    x = symbols('x', real=True)
    d = diff(x**n, x, n)
    assert d.rewrite(factorial) == factorial(n)


def test_U5():
    # issue 6681
    t = symbols('t')
    ans = (
        Derivative(f(g(t)), g(t))*Derivative(g(t), (t, 2)) +
        Derivative(f(g(t)), (g(t), 2))*Derivative(g(t), t)**2)
    assert f(g(t)).diff(t, 2) == ans
    assert ans.doit() == ans


def test_U6():
    h = Function('h')
    T = integrate(f(y), (y, h(x), g(x)))
    assert T.diff(x) == (
        f(g(x))*Derivative(g(x), x) - f(h(x))*Derivative(h(x), x))


@XFAIL
def test_U7():
    p, t = symbols('p t', real=True)
    # Exact differential => d(V(P, T)) => dV/dP DP + dV/dT DT
    # raises ValueError:  Since there is more than one variable in the
    # expression, the variable(s) of differentiation must be supplied to
    # differentiate f(p,t)
    diff(f(p, t))


def test_U8():
    x, y = symbols('x y', real=True)
    eq = cos(x*y) + x
    #  If SymPy had implicit_diff() function this hack could be avoided
    # TODO: Replace solve with solveset, current test fails for solveset
    assert idiff(y - eq, y, x) == (-y*sin(x*y) + 1)/(x*sin(x*y) + 1)


def test_U9():
    # Wester sample case for Maple:
    # O29 := diff(f(x, y), x) + diff(f(x, y), y);
    #                      /d         \   /d         \
    #                      |-- f(x, y)| + |-- f(x, y)|
    #                      \dx        /   \dy        /
    #
    # O30 := factor(subs(f(x, y) = g(x^2 + y^2), %));
    #                                2    2
    #                        2 D(g)(x  + y ) (x + y)
    x, y = symbols('x y', real=True)
    su = diff(f(x, y), x) + diff(f(x, y), y)
    s2 = su.subs(f(x, y), g(x**2 + y**2))
    s3 = s2.doit().factor()
    # Subs not performed, s3 = 2*(x + y)*Subs(Derivative(
    #   g(_xi_1), _xi_1), _xi_1, x**2 + y**2)
    # Derivative(g(x*2 + y**2), x**2 + y**2) is not valid in SymPy,
    # and probably will remain that way. You can take derivatives with respect
    # to other expressions only if they are atomic, like a symbol or a
    # function.
    # D operator should be added to SymPy
    # See https://github.com/sympy/sympy/issues/4719.
    assert s3 == (x + y)*Subs(Derivative(g(x), x), x, x**2 + y**2)*2


def test_U10():
    # see issue 2519:
    assert residue((z**3 + 5)/((z**4 - 1)*(z + 1)), z, -1) == R(-9, 4)

@XFAIL
def test_U11():
    # assert (2*dx + dz) ^ (3*dx + dy + dz) ^ (dx + dy + 4*dz) == 8*dx ^ dy ^dz
    raise NotImplementedError


@XFAIL
def test_U12():
    # Wester sample case:
    # (c41) /* d(3 x^5 dy /\ dz + 5 x y^2 dz /\ dx + 8 z dx /\ dy)
    #    => (15 x^4 + 10 x y + 8) dx /\ dy /\ dz */
    # factor(ext_diff(3*x^5 * dy ~ dz + 5*x*y^2 * dz ~ dx + 8*z * dx ~ dy));
    #                      4
    # (d41)              (10 x y + 15 x  + 8) dx dy dz
    raise NotImplementedError(
        "External diff of differential form not supported")


def test_U13():
    assert minimum(x**4 - x + 1, x) == -3*2**R(1,3)/8 + 1


@XFAIL
def test_U14():
    #f = 1/(x**2 + y**2 + 1)
    #assert [minimize(f), maximize(f)] == [0,1]
    raise NotImplementedError("minimize(), maximize() not supported")


@XFAIL
def test_U15():
    raise NotImplementedError("minimize() not supported and also solve does \
not support multivariate inequalities")


@XFAIL
def test_U16():
    raise NotImplementedError("minimize() not supported in SymPy and also \
solve does not support multivariate inequalities")


@XFAIL
def test_U17():
    raise NotImplementedError("Linear programming, symbolic simplex not \
supported in SymPy")


def test_V1():
    x = symbols('x', real=True)
    assert integrate(abs(x), x) == Piecewise((-x**2/2, x <= 0), (x**2/2, True))


def test_V2():
    assert integrate(Piecewise((-x, x < 0), (x, x >= 0)), x
        ) == Piecewise((-x**2/2, x < 0), (x**2/2, True))


def test_V3():
    assert integrate(1/(x**3 + 2),x).diff().simplify() == 1/(x**3 + 2)


def test_V4():
    assert integrate(2**x/sqrt(1 + 4**x), x) == asinh(2**x)/log(2)


@XFAIL
def test_V5():
    # Returns (-45*x**2 + 80*x - 41)/(5*sqrt(2*x - 1)*(4*x**2 - 4*x + 1))
    assert (integrate((3*x - 5)**2/(2*x - 1)**R(7, 2), x).simplify() ==
            (-41 + 80*x - 45*x**2)/(5*(2*x - 1)**R(5, 2)))


@XFAIL
def test_V6():
    # returns RootSum(40*_z**2 - 1, Lambda(_i, _i*log(-4*_i + exp(-m*x))))/m
    assert (integrate(1/(2*exp(m*x) - 5*exp(-m*x)), x) == sqrt(10)*(
            log(2*exp(m*x) - sqrt(10)) - log(2*exp(m*x) + sqrt(10)))/(20*m))


def test_V7():
    r1 = integrate(sinh(x)**4/cosh(x)**2)
    assert r1.simplify() == x*R(-3, 2) + sinh(x)**3/(2*cosh(x)) + 3*tanh(x)/2


@XFAIL
def test_V8_V9():
#Macsyma test case:
#(c27) /* This example involves several symbolic parameters
#   => 1/sqrt(b^2 - a^2) log([sqrt(b^2 - a^2) tan(x/2) + a + b]/
#                            [sqrt(b^2 - a^2) tan(x/2) - a - b])   (a^2 < b^2)
#      [Gradshteyn and Ryzhik 2.553(3)] */
#assume(b^2 > a^2)$
#(c28) integrate(1/(a + b*cos(x)), x);
#(c29) trigsimp(ratsimp(diff(%, x)));
#                        1
#(d29)             ------------
#                  b cos(x) + a
    raise NotImplementedError(
        "Integrate with assumption not supported")


def test_V10():
    assert integrate(1/(3 + 3*cos(x) + 4*sin(x)), x) == log(4*tan(x/2) + 3)/4


def test_V11():
    r1 = integrate(1/(4 + 3*cos(x) + 4*sin(x)), x)
    r2 = factor(r1)
    assert (logcombine(r2, force=True) ==
            log(((tan(x/2) + 1)/(tan(x/2) + 7))**R(1, 3)))


def test_V12():
    r1 = integrate(1/(5 + 3*cos(x) + 4*sin(x)), x)
    assert r1 == -1/(tan(x/2) + 2)


@XFAIL
def test_V13():
    r1 = integrate(1/(6 + 3*cos(x) + 4*sin(x)), x)
    # expression not simplified, returns: -sqrt(11)*I*log(tan(x/2) + 4/3
    #   - sqrt(11)*I/3)/11 + sqrt(11)*I*log(tan(x/2) + 4/3 + sqrt(11)*I/3)/11
    assert r1.simplify() == 2*sqrt(11)*atan(sqrt(11)*(3*tan(x/2) + 4)/11)/11


@slow
@XFAIL
def test_V14():
    r1 = integrate(log(abs(x**2 - y**2)), x)
    # Piecewise result does not simplify to the desired result.
    assert (r1.simplify() == x*log(abs(x**2  - y**2))
                            + y*log(x + y) - y*log(x - y) - 2*x)


def test_V15():
    r1 = integrate(x*acot(x/y), x)
    assert simplify(r1 - (x*y + (x**2 + y**2)*acot(x/y))/2) == 0


@XFAIL
def test_V16():
    # Integral not calculated
    assert integrate(cos(5*x)*Ci(2*x), x) == Ci(2*x)*sin(5*x)/5 - (Si(3*x) + Si(7*x))/10

@XFAIL
def test_V17():
    r1 = integrate((diff(f(x), x)*g(x)
                   - f(x)*diff(g(x), x))/(f(x)**2 - g(x)**2), x)
    # integral not calculated
    assert simplify(r1 - (f(x) - g(x))/(f(x) + g(x))/2) == 0


@XFAIL
def test_W1():
    # The function has a pole at y.
    # The integral has a Cauchy principal value of zero but SymPy returns -I*pi
    # https://github.com/sympy/sympy/issues/7159
    assert integrate(1/(x - y), (x, y - 1, y + 1)) == 0


@XFAIL
def test_W2():
    # The function has a pole at y.
    # The integral is divergent but SymPy returns -2
    # https://github.com/sympy/sympy/issues/7160
    # Test case in Macsyma:
    # (c6) errcatch(integrate(1/(x - a)^2, x, a - 1, a + 1));
    # Integral is divergent
    assert integrate(1/(x - y)**2, (x, y - 1, y + 1)) is zoo


@XFAIL
@slow
def test_W3():
    # integral is not  calculated
    # https://github.com/sympy/sympy/issues/7161
    assert integrate(sqrt(x + 1/x - 2), (x, 0, 1)) == R(4, 3)


@XFAIL
@slow
def test_W4():
    # integral is not  calculated
    assert integrate(sqrt(x + 1/x - 2), (x, 1, 2)) == -2*sqrt(2)/3 + R(4, 3)


@XFAIL
@slow
def test_W5():
    # integral is not  calculated
    assert integrate(sqrt(x + 1/x - 2), (x, 0, 2)) == -2*sqrt(2)/3 + R(8, 3)


@XFAIL
@slow
def test_W6():
    # integral is not  calculated
    assert integrate(sqrt(2 - 2*cos(2*x))/2, (x, pi*R(-3, 4), -pi/4)) == sqrt(2)


def test_W7():
    a = symbols('a', positive=True)
    r1 = integrate(cos(x)/(x**2 + a**2), (x, -oo, oo))
    assert r1.simplify() == pi*exp(-a)/a


@XFAIL
def test_W8():
    # Test case in Mathematica:
    # In[19]:= Integrate[t^(a - 1)/(1 + t), {t, 0, Infinity},
    #                    Assumptions -> 0 < a < 1]
    # Out[19]= Pi Csc[a Pi]
    raise NotImplementedError(
        "Integrate with assumption 0 < a < 1 not supported")


@XFAIL
@slow
def test_W9():
    # Integrand with a residue at infinity => -2 pi [sin(pi/5) + sin(2pi/5)]
    # (principal value)   [Levinson and Redheffer, p. 234] *)
    r1 = integrate(5*x**3/(1 + x + x**2 + x**3 + x**4), (x, -oo, oo))
    r2 = r1.doit()
    assert r2 == -2*pi*(sqrt(-sqrt(5)/8 + 5/8) + sqrt(sqrt(5)/8 + 5/8))


@XFAIL
def test_W10():
    # integrate(1/[1 + x + x^2 + ... + x^(2 n)], x = -infinity..infinity) =
    #        2 pi/(2 n + 1) [1 + cos(pi/[2 n + 1])] csc(2 pi/[2 n + 1])
    # [Levinson and Redheffer, p. 255] => 2 pi/5 [1 + cos(pi/5)] csc(2 pi/5) */
    r1 = integrate(x/(1 + x + x**2 + x**4), (x, -oo, oo))
    r2 = r1.doit()
    assert r2 == 2*pi*(sqrt(5)/4 + 5/4)*csc(pi*R(2, 5))/5


@XFAIL
def test_W11():
    # integral not calculated
    assert (integrate(sqrt(1 - x**2)/(1 + x**2), (x, -1, 1)) ==
            pi*(-1 + sqrt(2)))


def test_W12():
    p = symbols('p', positive=True)
    q = symbols('q', real=True)
    r1 = integrate(x*exp(-p*x**2 + 2*q*x), (x, -oo, oo))
    assert r1.simplify() == sqrt(pi)*q*exp(q**2/p)/p**R(3, 2)


@XFAIL
def test_W13():
    # Integral not calculated. Expected result is 2*(Euler_mascheroni_constant)
    r1 = integrate(1/log(x) + 1/(1 - x) - log(log(1/x)), (x, 0, 1))
    assert r1 == 2*EulerGamma


def test_W14():
    assert integrate(sin(x)/x*exp(2*I*x), (x, -oo, oo)) == 0


@XFAIL
def test_W15():
    # integral not calculated
    assert integrate(log(gamma(x))*cos(6*pi*x), (x, 0, 1)) == R(1, 12)


def test_W16():
    assert integrate((1 + x)**3*legendre_poly(1, x)*legendre_poly(2, x),
                     (x, -1, 1)) == R(36, 35)


def test_W17():
    a, b = symbols('a b', positive=True)
    assert integrate(exp(-a*x)*besselj(0, b*x),
                 (x, 0, oo)) == 1/(b*sqrt(a**2/b**2 + 1))


def test_W18():
    assert integrate((besselj(1, x)/x)**2, (x, 0, oo)) == 4/(3*pi)


@XFAIL
def test_W19():
    # Integral not calculated
    # Expected result is (cos 7 - 1)/7   [Gradshteyn and Ryzhik 6.782(3)]
    assert integrate(Ci(x)*besselj(0, 2*sqrt(7*x)), (x, 0, oo)) == (cos(7) - 1)/7


@XFAIL
def test_W20():
    # integral not calculated
    assert (integrate(x**2*polylog(3, 1/(x + 1)), (x, 0, 1)) ==
            -pi**2/36 - R(17, 108) + zeta(3)/4 +
            (-pi**2/2 - 4*log(2) + log(2)**2 + 35/3)*log(2)/9)


def test_W21():
    assert abs(N(integrate(x**2*polylog(3, 1/(x + 1)), (x, 0, 1)))
        - 0.210882859565594) < 1e-15


def test_W22():
    t, u = symbols('t u', real=True)
    s = Lambda(x, Piecewise((1, And(x >= 1, x <= 2)), (0, True)))
    assert integrate(s(t)*cos(t), (t, 0, u)) == Piecewise(
        (0, u < 0),
        (-sin(Min(1, u)) + sin(Min(2, u)), True))


@slow
def test_W23():
    a, b = symbols('a b', positive=True)
    r1 = integrate(integrate(x/(x**2 + y**2), (x, a, b)), (y, -oo, oo))
    assert r1.collect(pi).cancel() == -pi*a + pi*b


def test_W23b():
    # like W23 but limits are reversed
    a, b = symbols('a b', positive=True)
    r2 = integrate(integrate(x/(x**2 + y**2), (y, -oo, oo)), (x, a, b))
    assert r2.collect(pi) == pi*(-a + b)


@XFAIL
@tooslow
def test_W24():
    # Not that slow, but does not fully evaluate so simplify is slow.
    # Maybe also require doit()
    x, y = symbols('x y', real=True)
    r1 = integrate(integrate(sqrt(x**2 + y**2), (x, 0, 1)), (y, 0, 1))
    assert (r1 - (sqrt(2) + asinh(1))/3).simplify() == 0


@XFAIL
@tooslow
def test_W25():
    a, x, y = symbols('a x y', real=True)
    i1 = integrate(
        sin(a)*sin(y)/sqrt(1 - sin(a)**2*sin(x)**2*sin(y)**2),
        (x, 0, pi/2))
    i2 = integrate(i1, (y, 0, pi/2))
    assert (i2 - pi*a/2).simplify() == 0


def test_W26():
    x, y = symbols('x y', real=True)
    assert integrate(integrate(abs(y - x**2), (y, 0, 2)),
                     (x, -1, 1)) == R(46, 15)


def test_W27():
    a, b, c = symbols('a b c')
    assert integrate(integrate(integrate(1, (z, 0, c*(1 - x/a - y/b))),
                               (y, 0, b*(1 - x/a))),
                     (x, 0, a)) == a*b*c/6


def test_X1():
    v, c = symbols('v c', real=True)
    assert (series(1/sqrt(1 - (v/c)**2), v, x0=0, n=8) ==
            5*v**6/(16*c**6) + 3*v**4/(8*c**4) + v**2/(2*c**2) + 1 + O(v**8))


def test_X2():
    v, c = symbols('v c', real=True)
    s1 = series(1/sqrt(1 - (v/c)**2), v, x0=0, n=8)
    assert (1/s1**2).series(v, x0=0, n=8) == -v**2/c**2 + 1 + O(v**8)


def test_X3():
    s1 = (sin(x).series()/cos(x).series()).series()
    s2 = tan(x).series()
    assert s2 == x + x**3/3 + 2*x**5/15 + O(x**6)
    assert s1 == s2


def test_X4():
    s1 = log(sin(x)/x).series()
    assert s1 == -x**2/6 - x**4/180 + O(x**6)
    assert log(series(sin(x)/x)).series() == s1


@XFAIL
def test_X5():
    # test case in Mathematica syntax:
    # In[21]:= (* => [a f'(a d) + g(b d) + integrate(h(c y), y = 0..d)]
    #       + [a^2 f''(a d) + b g'(b d) + h(c d)] (x - d) *)
    # In[22]:= D[f[a*x], x] + g[b*x] + Integrate[h[c*y], {y, 0, x}]
    # Out[22]= g[b x] + Integrate[h[c y], {y, 0, x}] + a f'[a x]
    # In[23]:= Series[%, {x, d, 1}]
    # Out[23]= (g[b d] + Integrate[h[c y], {y, 0, d}] + a f'[a d]) +
    #                                    2                               2
    #             (h[c d] + b g'[b d] + a  f''[a d]) (-d + x) + O[-d + x]
    h = Function('h')
    a, b, c, d = symbols('a b c d', real=True)
    # series() raises NotImplementedError:
    # The _eval_nseries method should be added to <class
    # 'sympy.core.function.Subs'> to give terms up to O(x**n) at x=0
    series(diff(f(a*x), x) + g(b*x) + integrate(h(c*y), (y, 0, x)),
           x, x0=d, n=2)
    # assert missing, until exception is removed


def test_X6():
    # Taylor series of nonscalar objects (noncommutative multiplication)
    # expected result => (B A - A B) t^2/2 + O(t^3)   [Stanly Steinberg]
    a, b = symbols('a b', commutative=False, scalar=False)
    assert (series(exp((a + b)*x) - exp(a*x) * exp(b*x), x, x0=0, n=3) ==
              x**2*(-a*b/2 + b*a/2) + O(x**3))


def test_X7():
    # => sum( Bernoulli[k]/k! x^(k - 2), k = 1..infinity )
    #    = 1/x^2 - 1/(2 x) + 1/12 - x^2/720 + x^4/30240 + O(x^6)
    #    [Levinson and Redheffer, p. 173]
    assert (series(1/(x*(exp(x) - 1)), x, 0, 7) == x**(-2) - 1/(2*x) +
            R(1, 12) - x**2/720 + x**4/30240 - x**6/1209600 + O(x**7))


def test_X8():
    # Puiseux series (terms with fractional degree):
    # => 1/sqrt(x - 3/2 pi) + (x - 3/2 pi)^(3/2) / 12 + O([x - 3/2 pi]^(7/2))

    # see issue 7167:
    x = symbols('x', real=True)
    assert (series(sqrt(sec(x)), x, x0=pi*3/2, n=4) ==
            1/sqrt(x - pi*R(3, 2)) + (x - pi*R(3, 2))**R(3, 2)/12 +
            (x - pi*R(3, 2))**R(7, 2)/160 + O((x - pi*R(3, 2))**4, (x, pi*R(3, 2))))


def test_X9():
    assert (series(x**x, x, x0=0, n=4) == 1 + x*log(x) + x**2*log(x)**2/2 +
            x**3*log(x)**3/6 + O(x**4*log(x)**4))


def test_X10():
    z, w = symbols('z w')
    assert (series(log(sinh(z)) + log(cosh(z + w)), z, x0=0, n=2) ==
            log(cosh(w)) + log(z) + z*sinh(w)/cosh(w) + O(z**2))


def test_X11():
    z, w = symbols('z w')
    assert (series(log(sinh(z) * cosh(z + w)), z, x0=0, n=2) ==
            log(cosh(w)) + log(z) + z*sinh(w)/cosh(w) + O(z**2))


@XFAIL
def test_X12():
    # Look at the generalized Taylor series around x = 1
    # Result => (x - 1)^a/e^b [1 - (a + 2 b) (x - 1) / 2 + O((x - 1)^2)]
    a, b, x = symbols('a b x', real=True)
    # series returns O(log(x-1)**2)
    # https://github.com/sympy/sympy/issues/7168
    assert (series(log(x)**a*exp(-b*x), x, x0=1, n=2) ==
            (x - 1)**a/exp(b)*(1 - (a + 2*b)*(x - 1)/2 + O((x - 1)**2)))


def test_X13():
    assert series(sqrt(2*x**2 + 1), x, x0=oo, n=1) == sqrt(2)*x + O(1/x, (x, oo))


@XFAIL
def test_X14():
    # Wallis' product => 1/sqrt(pi n) + ...   [Knopp, p. 385]
    assert series(1/2**(2*n)*binomial(2*n, n),
                  n, x==oo, n=1) == 1/(sqrt(pi)*sqrt(n)) + O(1/x, (x, oo))


@SKIP("https://github.com/sympy/sympy/issues/7164")
def test_X15():
    # => 0!/x - 1!/x^2 + 2!/x^3 - 3!/x^4 + O(1/x^5)   [Knopp, p. 544]
    x, t = symbols('x t', real=True)
    # raises RuntimeError: maximum recursion depth exceeded
    # https://github.com/sympy/sympy/issues/7164
    # 2019-02-17: Raises
    # PoleError:
    # Asymptotic expansion of Ei around [-oo] is not implemented.
    e1 = integrate(exp(-t)/t, (t, x, oo))
    assert (series(e1, x, x0=oo, n=5) ==
            6/x**4 + 2/x**3 - 1/x**2 + 1/x + O(x**(-5), (x, oo)))


def test_X16():
    # Multivariate Taylor series expansion => 1 - (x^2 + 2 x y + y^2)/2 + O(x^4)
    assert (series(cos(x + y), x + y, x0=0, n=4) == 1 - (x + y)**2/2 +
            O(x**4 + x**3*y + x**2*y**2 + x*y**3 + y**4, x, y))


@XFAIL
def test_X17():
    # Power series (compute the general formula)
    # (c41) powerseries(log(sin(x)/x), x, 0);
    # /aquarius/data2/opt/local/macsyma_422/library1/trgred.so being loaded.
    #              inf
    #              ====     i1  2 i1          2 i1
    #              \        (- 1)   2      bern(2 i1) x
    # (d41)         >        ------------------------------
    #              /             2 i1 (2 i1)!
    #              ====
    #              i1 = 1
    # fps does not calculate
    assert fps(log(sin(x)/x)) == \
        Sum((-1)**k*2**(2*k - 1)*bernoulli(2*k)*x**(2*k)/(k*factorial(2*k)), (k, 1, oo))


@XFAIL
def test_X18():
    # Power series (compute the general formula). Maple FPS:
    # > FormalPowerSeries(exp(-x)*sin(x), x = 0);
    #                        infinity
    #                         -----    (1/2 k)                k
    #                          \      2        sin(3/4 k Pi) x
    #                           )     -------------------------
    #                          /                 k!
    #                         -----
    #
    # Now, SymPy returns
    #      oo
    #    _____
    #    \    `
    #     \        /          k             k\
    #      \     k |I*(-1 - I)    I*(-1 + I) |
    #       \   x *|----------- - -----------|
    #       /      \     2             2     /
    #      /    ------------------------------
    #     /                   k!
    #    /____,
    #    k = 0
    k = Dummy('k')
    assert fps(exp(-x)*sin(x)) == \
        Sum(2**(S.Half*k)*sin(R(3, 4)*k*pi)*x**k/factorial(k), (k, 0, oo))


@XFAIL
def test_X19():
    # (c45) /* Derive an explicit Taylor series solution of y as a function of
    # x from the following implicit relation:
    #    y = x - 1 + (x - 1)^2/2 + 2/3 (x - 1)^3 + (x - 1)^4 +
    #        17/10 (x - 1)^5 + ...
    #    */
    # x = sin(y) + cos(y);
    # Time= 0 msecs
    # (d45)                   x = sin(y) + cos(y)
    #
    # (c46) taylor_revert(%, y, 7);
    raise NotImplementedError("Solve using series not supported. \
Inverse Taylor series expansion also not supported")


@XFAIL
def test_X20():
    # Pade (rational function) approximation => (2 - x)/(2 + x)
    # > numapprox[pade](exp(-x), x = 0, [1, 1]);
    # bytes used=9019816, alloc=3669344, time=13.12
    #                                    1 - 1/2 x
    #                                    ---------
    #                                    1 + 1/2 x
    # mpmath support numeric Pade approximant but there is
    # no symbolic implementation in SymPy
    # https://en.wikipedia.org/wiki/Pad%C3%A9_approximant
    raise NotImplementedError("Symbolic Pade approximant not supported")


def test_X21():
    """
    Test whether `fourier_series` of x periodical on the [-p, p] interval equals
    `- (2 p / pi) sum( (-1)^n / n sin(n pi x / p), n = 1..infinity )`.
    """
    p = symbols('p', positive=True)
    n = symbols('n', positive=True, integer=True)
    s = fourier_series(x, (x, -p, p))

    # All cosine coefficients are equal to 0
    assert s.an.formula == 0

    # Check for sine coefficients
    assert s.bn.formula.subs(s.bn.variables[0], 0) == 0
    assert s.bn.formula.subs(s.bn.variables[0], n) == \
        -2*p/pi * (-1)**n / n * sin(n*pi*x/p)


@XFAIL
def test_X22():
    # (c52) /* => p / 2
    #    - (2 p / pi^2) sum( [1 - (-1)^n] cos(n pi x / p) / n^2,
    #                        n = 1..infinity ) */
    # fourier_series(abs(x), x, p);
    #                       p
    # (e52)                      a  = -
    #                       0      2
    #
    #                       %nn
    #                   (2 (- 1)    - 2) p
    # (e53)                a    = ------------------
    #                 %nn         2    2
    #                       %pi  %nn
    #
    # (e54)                     b    = 0
    #                      %nn
    #
    # Time= 5290 msecs
    #            inf           %nn            %pi %nn x
    #            ====       (2 (- 1)    - 2) cos(---------)
    #            \                    p
    #          p  >       -------------------------------
    #            /               2
    #            ====                %nn
    #            %nn = 1                     p
    # (d54)          ----------------------------------------- + -
    #                       2                 2
    #                    %pi
    raise NotImplementedError("Fourier series not supported")


def test_Y1():
    t = symbols('t', positive=True)
    w = symbols('w', real=True)
    s = symbols('s')
    F, _, _ = laplace_transform(cos((w - 1)*t), t, s)
    assert F == s/(s**2 + (w - 1)**2)


def test_Y2():
    t = symbols('t', positive=True)
    w = symbols('w', real=True)
    s = symbols('s')
    f = inverse_laplace_transform(s/(s**2 + (w - 1)**2), s, t, simplify=True)
    assert f == cos(t*(w - 1))


def test_Y3():
    t = symbols('t', positive=True)
    w = symbols('w', real=True)
    s = symbols('s')
    F, _, _ = laplace_transform(sinh(w*t)*cosh(w*t), t, s, simplify=True)
    assert F == w/(s**2 - 4*w**2)


def test_Y4():
    t = symbols('t', positive=True)
    s = symbols('s')
    F, _, _ = laplace_transform(erf(3/sqrt(t)), t, s, simplify=True)
    assert F == 1/s - exp(-6*sqrt(s))/s


def test_Y5_Y6():
# Solve y'' + y = 4 [H(t - 1) - H(t - 2)], y(0) = 1, y'(0) = 0 where H is the
# Heaviside (unit step) function (the RHS describes a pulse of magnitude 4 and
# duration 1).  See David A. Sanchez, Richard C. Allen, Jr. and Walter T.
# Kyner, _Differential Equations: An Introduction_, Addison-Wesley Publishing
# Company, 1983, p. 211.  First, take the Laplace transform of the ODE
# => s^2 Y(s) - s + Y(s) = 4/s [e^(-s) - e^(-2 s)]
# where Y(s) is the Laplace transform of y(t)
    t = symbols('t', real=True)
    s = symbols('s')
    y = Function('y')
    Y = Function('Y')
    F = laplace_correspondence(laplace_transform(diff(y(t), t, 2) + y(t)
                                - 4*(Heaviside(t - 1) - Heaviside(t - 2)),
                                t, s, noconds=True), {y: Y})
    D = (
        -F + s**2*Y(s) - s*y(0) + Y(s) - Subs(Derivative(y(t), t), t, 0) -
        4*exp(-s)/s + 4*exp(-2*s)/s)
    assert D == 0
# Now, solve for Y(s) and then take the inverse Laplace transform
#   => Y(s) = s/(s^2 + 1) + 4 [1/s - s/(s^2 + 1)] [e^(-s) - e^(-2 s)]
#   => y(t) = cos t + 4 {[1 - cos(t - 1)] H(t - 1) - [1 - cos(t - 2)] H(t - 2)}
    Yf = solve(F, Y(s))[0]
    Yf = laplace_initial_conds(Yf, t, {y: [1, 0]})
    assert Yf == (s**2*exp(2*s) + 4*exp(s) - 4)*exp(-2*s)/(s*(s**2 + 1))
    yf = inverse_laplace_transform(Yf, s, t)
    yf = yf.collect(Heaviside(t-1)).collect(Heaviside(t-2))
    assert yf == (
        (4 - 4*cos(t - 1))*Heaviside(t - 1) +
        (4*cos(t - 2) - 4)*Heaviside(t - 2) +
        cos(t)*Heaviside(t))


@XFAIL
def test_Y7():
    # What is the Laplace transform of an infinite square wave?
    # => 1/s + 2 sum( (-1)^n e^(- s n a)/s, n = 1..infinity )
    #    [Sanchez, Allen and Kyner, p. 213]
    t = symbols('t', positive=True)
    a = symbols('a', real=True)
    s = symbols('s')
    F, _, _ = laplace_transform(1 + 2*Sum((-1)**n*Heaviside(t - n*a),
                                          (n, 1, oo)), t, s)
    # returns 2*LaplaceTransform(Sum((-1)**n*Heaviside(-a*n + t),
    #                                (n, 1, oo)), t, s) + 1/s
    # https://github.com/sympy/sympy/issues/7177
    assert F == 2*Sum((-1)**n*exp(-a*n*s)/s, (n, 1, oo)) + 1/s


@XFAIL
def test_Y8():
    assert fourier_transform(1, x, z) == DiracDelta(z)


def test_Y9():
    assert (fourier_transform(exp(-9*x**2), x, z) ==
            sqrt(pi)*exp(-pi**2*z**2/9)/3)


def test_Y10():
    assert (fourier_transform(abs(x)*exp(-3*abs(x)), x, z).cancel() ==
            (-8*pi**2*z**2 + 18)/(16*pi**4*z**4 + 72*pi**2*z**2 + 81))


@SKIP("https://github.com/sympy/sympy/issues/7181")
@slow
def test_Y11():
    # => pi cot(pi s)   (0 < Re s < 1)   [Gradshteyn and Ryzhik 17.43(5)]
    x, s = symbols('x s')
    # raises RuntimeError: maximum recursion depth exceeded
    # https://github.com/sympy/sympy/issues/7181
    # Update 2019-02-17 raises:
    # TypeError: cannot unpack non-iterable MellinTransform object
    F, _, _ =  mellin_transform(1/(1 - x), x, s)
    assert F == pi*cot(pi*s)


@XFAIL
def test_Y12():
    # => 2^(s - 4) gamma(s/2)/gamma(4 - s/2)   (0 < Re s < 1)
    # [Gradshteyn and Ryzhik 17.43(16)]
    x, s = symbols('x s')
    # returns Wrong value -2**(s - 4)*gamma(s/2 - 3)/gamma(-s/2 + 1)
    # https://github.com/sympy/sympy/issues/7182
    F, _, _ = mellin_transform(besselj(3, x)/x**3, x, s)
    assert F == -2**(s - 4)*gamma(s/2)/gamma(-s/2 + 4)


@XFAIL
def test_Y13():
# Z[H(t - m T)] => z/[z^m (z - 1)]   (H is the Heaviside (unit step) function)                                                 z
    raise NotImplementedError("z-transform not supported")


@XFAIL
def test_Y14():
# Z[H(t - m T)] => z/[z^m (z - 1)]   (H is the Heaviside (unit step) function)
    raise NotImplementedError("z-transform not supported")


def test_Z1():
    r = Function('r')
    assert (rsolve(r(n + 2) - 2*r(n + 1) + r(n) - 2, r(n),
                   {r(0): 1, r(1): m}).simplify() == n**2 + n*(m - 2) + 1)


def test_Z2():
    r = Function('r')
    assert (rsolve(r(n) - (5*r(n - 1) - 6*r(n - 2)), r(n), {r(0): 0, r(1): 1})
            == -2**n + 3**n)


def test_Z3():
    # => r(n) = Fibonacci[n + 1]   [Cohen, p. 83]
    r = Function('r')
    # recurrence solution is correct, Wester expects it to be simplified to
    # fibonacci(n+1), but that is quite hard
    expected = ((S(1)/2 - sqrt(5)/2)**n*(S(1)/2 - sqrt(5)/10)
              + (S(1)/2 + sqrt(5)/2)**n*(sqrt(5)/10 + S(1)/2))
    sol = rsolve(r(n) - (r(n - 1) + r(n - 2)), r(n), {r(1): 1, r(2): 2})
    assert sol == expected


@XFAIL
def test_Z4():
# => [c^(n+1) [c^(n+1) - 2 c - 2] + (n+1) c^2 + 2 c - n] / [(c-1)^3 (c+1)]
#    [Joan Z. Yu and Robert Israel in sci.math.symbolic]
    r = Function('r')
    c = symbols('c')
    # raises ValueError: Polynomial or rational function expected,
    #     got '(c**2 - c**n)/(c - c**n)
    s = rsolve(r(n) - ((1 + c - c**(n-1) - c**(n+1))/(1 - c**n)*r(n - 1)
                   - c*(1 - c**(n-2))/(1 - c**(n-1))*r(n - 2) + 1),
           r(n), {r(1): 1, r(2): (2 + 2*c + c**2)/(1 + c)})
    assert (s - (c*(n + 1)*(c*(n + 1) - 2*c - 2) +
             (n + 1)*c**2 + 2*c - n)/((c-1)**3*(c+1)) == 0)


@XFAIL
def test_Z5():
    # Second order ODE with initial conditions---solve directly
    # transform: f(t) = sin(2 t)/8 - t cos(2 t)/4
    C1, C2 = symbols('C1 C2')
    # initial conditions not supported, this is a manual workaround
    # https://github.com/sympy/sympy/issues/4720
    eq = Derivative(f(x), x, 2) + 4*f(x) - sin(2*x)
    sol = dsolve(eq, f(x))
    f0 = Lambda(x, sol.rhs)
    assert f0(x) == C2*sin(2*x) + (C1 - x/4)*cos(2*x)
    f1 = Lambda(x, diff(f0(x), x))
    # TODO: Replace solve with solveset, when it works for solveset
    const_dict = solve((f0(0), f1(0)))
    result = f0(x).subs(C1, const_dict[C1]).subs(C2, const_dict[C2])
    assert result == -x*cos(2*x)/4 + sin(2*x)/8
    # Result is OK, but ODE solving with initial conditions should be
    # supported without all this manual work
    raise NotImplementedError('ODE solving with initial conditions \
not supported')


@XFAIL
def test_Z6():
    # Second order ODE with initial conditions---solve  using Laplace
    # transform: f(t) = sin(2 t)/8 - t cos(2 t)/4
    t = symbols('t', positive=True)
    s = symbols('s')
    eq = Derivative(f(t), t, 2) + 4*f(t) - sin(2*t)
    F, _, _ = laplace_transform(eq, t, s)
    # Laplace transform for diff() not calculated
    # https://github.com/sympy/sympy/issues/7176
    assert (F == s**2*LaplaceTransform(f(t), t, s) +
            4*LaplaceTransform(f(t), t, s) - 2/(s**2 + 4))
    # rest of test case not implemented

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\unknown_fields.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Contains Unknown Fields APIs.

Simple usage example:
  unknown_field_set = UnknownFieldSet(message)
  for unknown_field in unknown_field_set:
    wire_type = unknown_field.wire_type
    field_number = unknown_field.field_number
    data = unknown_field.data
"""


from google.protobuf.internal import api_implementation

if api_implementation._c_module is not None:  # pylint: disable=protected-access
  UnknownFieldSet = api_implementation._c_module.UnknownFieldSet  # pylint: disable=protected-access
else:
  from google.protobuf.internal import decoder  # pylint: disable=g-import-not-at-top
  from google.protobuf.internal import wire_format  # pylint: disable=g-import-not-at-top

  class UnknownField:
    """A parsed unknown field."""

    # Disallows assignment to other attributes.
    __slots__ = ['_field_number', '_wire_type', '_data']

    def __init__(self, field_number, wire_type, data):
      self._field_number = field_number
      self._wire_type = wire_type
      self._data = data
      return

    @property
    def field_number(self):
      return self._field_number

    @property
    def wire_type(self):
      return self._wire_type

    @property
    def data(self):
      return self._data

  class UnknownFieldSet:
    """UnknownField container."""

    # Disallows assignment to other attributes.
    __slots__ = ['_values']

    def __init__(self, msg):

      def InternalAdd(field_number, wire_type, data):
        unknown_field = UnknownField(field_number, wire_type, data)
        self._values.append(unknown_field)

      self._values = []
      msg_des = msg.DESCRIPTOR
      # pylint: disable=protected-access
      unknown_fields = msg._unknown_fields
      if (msg_des.has_options and
          msg_des.GetOptions().message_set_wire_format):
        local_decoder = decoder.UnknownMessageSetItemDecoder()
        for _, buffer in unknown_fields:
          (field_number, data) = local_decoder(memoryview(buffer))
          InternalAdd(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED, data)
      else:
        for tag_bytes, buffer in unknown_fields:
          field_number, wire_type = decoder.DecodeTag(tag_bytes)
          if field_number == 0:
            raise RuntimeError('Field number 0 is illegal.')
          (data, _) = decoder._DecodeUnknownField(
              memoryview(buffer), 0, len(buffer), field_number, wire_type
          )
          InternalAdd(field_number, wire_type, data)

    def __getitem__(self, index):
      size = len(self._values)
      if index < 0:
        index += size
      if index < 0 or index >= size:
        raise IndexError('index %d out of range'.index)

      return self._values[index]

    def __len__(self):
      return len(self._values)

    def __iter__(self):
      return iter(self._values)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_globals.py ===
"""
Module defining global singleton classes.

This module raises a RuntimeError if an attempt to reload it is made. In that
way the identities of the classes defined here are fixed and will remain so
even if numpy itself is reloaded. In particular, a function like the following
will still work correctly after numpy is reloaded::

    def foo(arg=np._NoValue):
        if arg is np._NoValue:
            ...

That was not the case when the singleton classes were defined in the numpy
``__init__.py`` file. See gh-7844 for a discussion of the reload problem that
motivated this module.

"""
import enum

from ._utils import set_module as _set_module

__all__ = ['_NoValue', '_CopyMode']


# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if '_is_loaded' in globals():
    raise RuntimeError('Reloading numpy._globals is not allowed')
_is_loaded = True


class _NoValueType:
    """Special keyword value.

    The instance of this class may be used as the default value assigned to a
    keyword if no other obvious default (e.g., `None`) is suitable,

    Common reasons for using this keyword are:

    - A new keyword is added to a function, and that function forwards its
      inputs to another function or method which can be defined outside of
      NumPy. For example, ``np.std(x)`` calls ``x.std``, so when a ``keepdims``
      keyword was added that could only be forwarded if the user explicitly
      specified ``keepdims``; downstream array libraries may not have added
      the same keyword, so adding ``x.std(..., keepdims=keepdims)``
      unconditionally could have broken previously working code.
    - A keyword is being deprecated, and a deprecation warning must only be
      emitted when the keyword is used.

    """
    __instance = None

    def __new__(cls):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __repr__(self):
        return "<no value>"


_NoValue = _NoValueType()


@_set_module("numpy")
class _CopyMode(enum.Enum):
    """
    An enumeration for the copy modes supported
    by numpy.copy() and numpy.array(). The following three modes are supported,

    - ALWAYS: This means that a deep copy of the input
              array will always be taken.
    - IF_NEEDED: This means that a deep copy of the input
                 array will be taken only if necessary.
    - NEVER: This means that the deep copy will never be taken.
             If a copy cannot be avoided then a `ValueError` will be
             raised.

    Note that the buffer-protocol could in theory do copies.  NumPy currently
    assumes an object exporting the buffer protocol will never do this.
    """

    ALWAYS = True
    NEVER = False
    IF_NEEDED = 2

    def __bool__(self):
        # For backwards compatibility
        if self == _CopyMode.ALWAYS:
            return True

        if self == _CopyMode.NEVER:
            return False

        raise ValueError(f"{self} is neither True nor False.")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\DeprecatedSessionState.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

# deprecated: no longer using kernel def hashes
class DeprecatedSessionState(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DeprecatedSessionState()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDeprecatedSessionState(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def DeprecatedSessionStateBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # DeprecatedSessionState
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DeprecatedSessionState
    def Kernels(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.DeprecatedKernelCreateInfos import DeprecatedKernelCreateInfos
            obj = DeprecatedKernelCreateInfos()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # DeprecatedSessionState
    def SubGraphSessionStates(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.DeprecatedSubGraphSessionState import DeprecatedSubGraphSessionState
            obj = DeprecatedSubGraphSessionState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # DeprecatedSessionState
    def SubGraphSessionStatesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # DeprecatedSessionState
    def SubGraphSessionStatesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def DeprecatedSessionStateStart(builder):
    builder.StartObject(2)

def Start(builder):
    DeprecatedSessionStateStart(builder)

def DeprecatedSessionStateAddKernels(builder, kernels):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(kernels), 0)

def AddKernels(builder, kernels):
    DeprecatedSessionStateAddKernels(builder, kernels)

def DeprecatedSessionStateAddSubGraphSessionStates(builder, subGraphSessionStates):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(subGraphSessionStates), 0)

def AddSubGraphSessionStates(builder, subGraphSessionStates):
    DeprecatedSessionStateAddSubGraphSessionStates(builder, subGraphSessionStates)

def DeprecatedSessionStateStartSubGraphSessionStatesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartSubGraphSessionStatesVector(builder, numElems: int) -> int:
    return DeprecatedSessionStateStartSubGraphSessionStatesVector(builder, numElems)

def DeprecatedSessionStateEnd(builder):
    return builder.EndObject()

def End(builder):
    return DeprecatedSessionStateEnd(builder)