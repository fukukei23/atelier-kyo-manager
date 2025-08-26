
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\cxx.py ===
"""
C++ code printer
"""

from itertools import chain
from sympy.codegen.ast import Type, none
from .codeprinter import requires
from .c import C89CodePrinter, C99CodePrinter

# These are defined in the other file so we can avoid importing sympy.codegen
# from the top-level 'import sympy'. Export them here as well.
from sympy.printing.codeprinter import cxxcode # noqa:F401

# from https://en.cppreference.com/w/cpp/keyword
reserved = {
    'C++98': [
        'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break',
        'case', 'catch,', 'char', 'class', 'compl', 'const', 'const_cast',
        'continue', 'default', 'delete', 'do', 'double', 'dynamic_cast',
        'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float',
        'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable',
        'namespace', 'new', 'not', 'not_eq', 'operator', 'or', 'or_eq',
        'private', 'protected', 'public', 'register', 'reinterpret_cast',
        'return', 'short', 'signed', 'sizeof', 'static', 'static_cast',
        'struct', 'switch', 'template', 'this', 'throw', 'true', 'try',
        'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using',
        'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq'
    ]
}

reserved['C++11'] = reserved['C++98'][:] + [
    'alignas', 'alignof', 'char16_t', 'char32_t', 'constexpr', 'decltype',
    'noexcept', 'nullptr', 'static_assert', 'thread_local'
]
reserved['C++17'] = reserved['C++11'][:]
reserved['C++17'].remove('register')
# TM TS: atomic_cancel, atomic_commit, atomic_noexcept, synchronized
# concepts TS: concept, requires
# module TS: import, module


_math_functions = {
    'C++98': {
        'Mod': 'fmod',
        'ceiling': 'ceil',
    },
    'C++11': {
        'gamma': 'tgamma',
    },
    'C++17': {
        'beta': 'beta',
        'Ei': 'expint',
        'zeta': 'riemann_zeta',
    }
}

# from https://en.cppreference.com/w/cpp/header/cmath
for k in ('Abs', 'exp', 'log', 'log10', 'sqrt', 'sin', 'cos', 'tan',  # 'Pow'
          'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'floor'):
    _math_functions['C++98'][k] = k.lower()


for k in ('asinh', 'acosh', 'atanh', 'erf', 'erfc'):
    _math_functions['C++11'][k] = k.lower()


def _attach_print_method(cls, sympy_name, func_name):
    meth_name = '_print_%s' % sympy_name
    if hasattr(cls, meth_name):
        raise ValueError("Edit method (or subclass) instead of overwriting.")
    def _print_method(self, expr):
        return '{}{}({})'.format(self._ns, func_name, ', '.join(map(self._print, expr.args)))
    _print_method.__doc__ = "Prints code for %s" % k
    setattr(cls, meth_name, _print_method)


def _attach_print_methods(cls, cont):
    for sympy_name, cxx_name in cont[cls.standard].items():
        _attach_print_method(cls, sympy_name, cxx_name)


class _CXXCodePrinterBase:
    printmethod = "_cxxcode"
    language = 'C++'
    _ns = 'std::'  # namespace

    def __init__(self, settings=None):
        super().__init__(settings or {})

    @requires(headers={'algorithm'})
    def _print_Max(self, expr):
        from sympy.functions.elementary.miscellaneous import Max
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "%smax(%s, %s)" % (self._ns, self._print(expr.args[0]),
                                  self._print(Max(*expr.args[1:])))

    @requires(headers={'algorithm'})
    def _print_Min(self, expr):
        from sympy.functions.elementary.miscellaneous import Min
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "%smin(%s, %s)" % (self._ns, self._print(expr.args[0]),
                                  self._print(Min(*expr.args[1:])))

    def _print_using(self, expr):
        if expr.alias == none:
            return 'using %s' % expr.type
        else:
            raise ValueError("C++98 does not support type aliases")

    def _print_Raise(self, rs):
        arg, = rs.args
        return 'throw %s' % self._print(arg)

    @requires(headers={'stdexcept'})
    def _print_RuntimeError_(self, re):
        message, = re.args
        return "%sruntime_error(%s)" % (self._ns, self._print(message))


class CXX98CodePrinter(_CXXCodePrinterBase, C89CodePrinter):
    standard = 'C++98'
    reserved_words = set(reserved['C++98'])


# _attach_print_methods(CXX98CodePrinter, _math_functions)


class CXX11CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    standard = 'C++11'
    reserved_words = set(reserved['C++11'])
    type_mappings = dict(chain(
        CXX98CodePrinter.type_mappings.items(),
        {
            Type('int8'): ('int8_t', {'cstdint'}),
            Type('int16'): ('int16_t', {'cstdint'}),
            Type('int32'): ('int32_t', {'cstdint'}),
            Type('int64'): ('int64_t', {'cstdint'}),
            Type('uint8'): ('uint8_t', {'cstdint'}),
            Type('uint16'): ('uint16_t', {'cstdint'}),
            Type('uint32'): ('uint32_t', {'cstdint'}),
            Type('uint64'): ('uint64_t', {'cstdint'}),
            Type('complex64'): ('std::complex<float>', {'complex'}),
            Type('complex128'): ('std::complex<double>', {'complex'}),
            Type('bool'): ('bool', None),
        }.items()
    ))

    def _print_using(self, expr):
        if expr.alias == none:
            return super()._print_using(expr)
        else:
            return 'using %(alias)s = %(type)s' % expr.kwargs(apply=self._print)

# _attach_print_methods(CXX11CodePrinter, _math_functions)


class CXX17CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    standard = 'C++17'
    reserved_words = set(reserved['C++17'])

    _kf = dict(C99CodePrinter._kf, **_math_functions['C++17'])

    def _print_beta(self, expr):
        return self._print_math_func(expr)

    def _print_Ei(self, expr):
        return self._print_math_func(expr)

    def _print_zeta(self, expr):
        return self._print_math_func(expr)


# _attach_print_methods(CXX17CodePrinter, _math_functions)

cxx_code_printers = {
    'c++98': CXX98CodePrinter,
    'c++11': CXX11CodePrinter,
    'c++17': CXX17CodePrinter
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\win32.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.

# from winbase.h
STDOUT = -11
STDERR = -12

ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

try:
    import ctypes
    from ctypes import LibraryLoader
    windll = LibraryLoader(ctypes.WinDLL)
    from ctypes import wintypes
except (AttributeError, ImportError):
    windll = None
    SetConsoleTextAttribute = lambda *_: None
    winapi_test = lambda *_: None
else:
    from ctypes import byref, Structure, c_char, POINTER

    COORD = wintypes._COORD

    class CONSOLE_SCREEN_BUFFER_INFO(Structure):
        """struct in wincon.h."""
        _fields_ = [
            ("dwSize", COORD),
            ("dwCursorPosition", COORD),
            ("wAttributes", wintypes.WORD),
            ("srWindow", wintypes.SMALL_RECT),
            ("dwMaximumWindowSize", COORD),
        ]
        def __str__(self):
            return '(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)' % (
                self.dwSize.Y, self.dwSize.X
                , self.dwCursorPosition.Y, self.dwCursorPosition.X
                , self.wAttributes
                , self.srWindow.Top, self.srWindow.Left, self.srWindow.Bottom, self.srWindow.Right
                , self.dwMaximumWindowSize.Y, self.dwMaximumWindowSize.X
            )

    _GetStdHandle = windll.kernel32.GetStdHandle
    _GetStdHandle.argtypes = [
        wintypes.DWORD,
    ]
    _GetStdHandle.restype = wintypes.HANDLE

    _GetConsoleScreenBufferInfo = windll.kernel32.GetConsoleScreenBufferInfo
    _GetConsoleScreenBufferInfo.argtypes = [
        wintypes.HANDLE,
        POINTER(CONSOLE_SCREEN_BUFFER_INFO),
    ]
    _GetConsoleScreenBufferInfo.restype = wintypes.BOOL

    _SetConsoleTextAttribute = windll.kernel32.SetConsoleTextAttribute
    _SetConsoleTextAttribute.argtypes = [
        wintypes.HANDLE,
        wintypes.WORD,
    ]
    _SetConsoleTextAttribute.restype = wintypes.BOOL

    _SetConsoleCursorPosition = windll.kernel32.SetConsoleCursorPosition
    _SetConsoleCursorPosition.argtypes = [
        wintypes.HANDLE,
        COORD,
    ]
    _SetConsoleCursorPosition.restype = wintypes.BOOL

    _FillConsoleOutputCharacterA = windll.kernel32.FillConsoleOutputCharacterA
    _FillConsoleOutputCharacterA.argtypes = [
        wintypes.HANDLE,
        c_char,
        wintypes.DWORD,
        COORD,
        POINTER(wintypes.DWORD),
    ]
    _FillConsoleOutputCharacterA.restype = wintypes.BOOL

    _FillConsoleOutputAttribute = windll.kernel32.FillConsoleOutputAttribute
    _FillConsoleOutputAttribute.argtypes = [
        wintypes.HANDLE,
        wintypes.WORD,
        wintypes.DWORD,
        COORD,
        POINTER(wintypes.DWORD),
    ]
    _FillConsoleOutputAttribute.restype = wintypes.BOOL

    _SetConsoleTitleW = windll.kernel32.SetConsoleTitleW
    _SetConsoleTitleW.argtypes = [
        wintypes.LPCWSTR
    ]
    _SetConsoleTitleW.restype = wintypes.BOOL

    _GetConsoleMode = windll.kernel32.GetConsoleMode
    _GetConsoleMode.argtypes = [
        wintypes.HANDLE,
        POINTER(wintypes.DWORD)
    ]
    _GetConsoleMode.restype = wintypes.BOOL

    _SetConsoleMode = windll.kernel32.SetConsoleMode
    _SetConsoleMode.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD
    ]
    _SetConsoleMode.restype = wintypes.BOOL

    def _winapi_test(handle):
        csbi = CONSOLE_SCREEN_BUFFER_INFO()
        success = _GetConsoleScreenBufferInfo(
            handle, byref(csbi))
        return bool(success)

    def winapi_test():
        return any(_winapi_test(h) for h in
                   (_GetStdHandle(STDOUT), _GetStdHandle(STDERR)))

    def GetConsoleScreenBufferInfo(stream_id=STDOUT):
        handle = _GetStdHandle(stream_id)
        csbi = CONSOLE_SCREEN_BUFFER_INFO()
        success = _GetConsoleScreenBufferInfo(
            handle, byref(csbi))
        return csbi

    def SetConsoleTextAttribute(stream_id, attrs):
        handle = _GetStdHandle(stream_id)
        return _SetConsoleTextAttribute(handle, attrs)

    def SetConsoleCursorPosition(stream_id, position, adjust=True):
        position = COORD(*position)
        # If the position is out of range, do nothing.
        if position.Y <= 0 or position.X <= 0:
            return
        # Adjust for Windows' SetConsoleCursorPosition:
        #    1. being 0-based, while ANSI is 1-based.
        #    2. expecting (x,y), while ANSI uses (y,x).
        adjusted_position = COORD(position.Y - 1, position.X - 1)
        if adjust:
            # Adjust for viewport's scroll position
            sr = GetConsoleScreenBufferInfo(STDOUT).srWindow
            adjusted_position.Y += sr.Top
            adjusted_position.X += sr.Left
        # Resume normal processing
        handle = _GetStdHandle(stream_id)
        return _SetConsoleCursorPosition(handle, adjusted_position)

    def FillConsoleOutputCharacter(stream_id, char, length, start):
        handle = _GetStdHandle(stream_id)
        char = c_char(char.encode())
        length = wintypes.DWORD(length)
        num_written = wintypes.DWORD(0)
        # Note that this is hard-coded for ANSI (vs wide) bytes.
        success = _FillConsoleOutputCharacterA(
            handle, char, length, start, byref(num_written))
        return num_written.value

    def FillConsoleOutputAttribute(stream_id, attr, length, start):
        ''' FillConsoleOutputAttribute( hConsole, csbi.wAttributes, dwConSize, coordScreen, &cCharsWritten )'''
        handle = _GetStdHandle(stream_id)
        attribute = wintypes.WORD(attr)
        length = wintypes.DWORD(length)
        num_written = wintypes.DWORD(0)
        # Note that this is hard-coded for ANSI (vs wide) bytes.
        return _FillConsoleOutputAttribute(
            handle, attribute, length, start, byref(num_written))

    def SetConsoleTitle(title):
        return _SetConsoleTitleW(title)

    def GetConsoleMode(handle):
        mode = wintypes.DWORD()
        success = _GetConsoleMode(handle, byref(mode))
        if not success:
            raise ctypes.WinError()
        return mode.value

    def SetConsoleMode(handle, mode):
        success = _SetConsoleMode(handle, mode)
        if not success:
            raise ctypes.WinError()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_group_norm.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

import numpy as np
from fusion_base import Fusion
from onnx import TensorProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionGroupNorm(Fusion):
    def __init__(self, model: OnnxModel, channels_last=True):
        super().__init__(model, "GroupNorm", "Add")
        self.channels_last = channels_last

    def fuse(self, add_node, input_name_to_nodes: dict, output_name_to_node: dict):
        """
         Fuse Group Normalization subgraph into one node GroupNorm.
         The following is the pattern with swish activation:
               +----------------Shape-------------------------------+
               |                                                    |
               |    (0, 32, -1)                                     v     (512x1x1) (512x1x1) (optional)
           [Root] --> Reshape -------> InstanceNormalization --> Reshape ---> Mul --> Add --> Mul--> [output]
        Bx512xHxW                 (scale=ones(32), B=zeros(32))                        |       ^     Bx512xHxW
                                                                                       |       |
                                                                                       +--->Sigmoid (optional)
        The Mul and Sigmoid before output is for Swish activation. They are optional.
        """
        nodes = self.model.match_parent_path(
            add_node, ["Mul", "Reshape", "InstanceNormalization", "Reshape"], [0, 0, 0, 0], output_name_to_node
        )
        if nodes is None:
            return

        weight_mul, reshape_4d, instance_norm, reshape_3d = nodes
        root = reshape_3d.input[0]

        parents = self.model.match_parent_path(reshape_4d, ["Shape"], [1], output_name_to_node)
        if parents is None:
            return
        if parents[0].input[0] != root:
            return
        shape_node = parents[0]

        # Check whether it has swish activation.
        swish_mul = self.model.find_first_child_by_type(add_node, "Mul")
        swish_sigmoid = None
        if swish_mul is not None:
            sigmoid_path = self.model.match_parent_path(swish_mul, ["Sigmoid"], [None], output_name_to_node)
            if sigmoid_path is not None:
                swish_sigmoid = sigmoid_path[0]

        weight_input = weight_mul.input[1 - self.model.input_index(reshape_4d.output[0], weight_mul)]
        if not self.model.is_constant_with_specified_dimension(weight_input, 3, "group norm weight"):
            return

        bias_input = add_node.input[1 - self.model.input_index(weight_mul.output[0], add_node)]
        if not self.model.is_constant_with_specified_dimension(bias_input, 3, "layernorm bias"):
            return

        weight = self.model.get_constant_value(weight_input)
        if weight is None:
            return

        if not (len(weight.shape) == 3 and weight.shape[1] == 1 and weight.shape[2] == 1):
            return

        bias = self.model.get_constant_value(bias_input)
        if bias is None:
            return
        if not (len(bias.shape) == 3 and bias.shape[1] == 1 and bias.shape[2] == 1):
            return

        weight_elements = int(np.prod(weight.shape))
        bias_elements = int(np.prod(bias.shape))
        if weight_elements != bias_elements:
            return

        instance_norm_scale = self.model.get_constant_value(instance_norm.input[1])
        if instance_norm_scale is None or len(instance_norm_scale.shape) != 1:
            return
        num_groups = int(instance_norm_scale.shape[0])

        instance_norm_bias = self.model.get_constant_value(instance_norm.input[2])
        if instance_norm_bias is None or instance_norm_scale.shape != instance_norm_scale.shape:
            return

        if not np.allclose(np.ones_like(instance_norm_scale), instance_norm_scale):
            return
        if not np.allclose(np.zeros_like(instance_norm_bias), instance_norm_bias):
            return

        group_norm_name = self.model.create_node_name("GroupNorm", name_prefix="GroupNorm")

        self.add_initializer(
            name=group_norm_name + "_gamma",
            data_type=TensorProto.FLOAT,
            dims=[weight_elements],
            vals=weight,
        )

        self.add_initializer(
            name=group_norm_name + "_beta",
            data_type=TensorProto.FLOAT,
            dims=[bias_elements],
            vals=bias,
        )

        last_node = add_node
        subgraph_nodes = [add_node, weight_mul, reshape_4d, instance_norm, reshape_3d, shape_node]
        has_swish_activation = swish_mul and swish_sigmoid
        if swish_mul and swish_sigmoid:
            subgraph_nodes.extend([swish_mul, swish_sigmoid])
            last_node = swish_mul

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            last_node.output,
            input_name_to_nodes,
            output_name_to_node,
        ):
            self.nodes_to_remove.extend([last_node])
        else:
            self.nodes_to_remove.extend(subgraph_nodes)

        # instance_norm_scale might from Constant node. Use prune graph to clear it.
        self.prune_graph = True

        input_name = root
        output_name = last_node.output[0]

        group_norm_input_name = input_name + "_NHWC" if self.channels_last else input_name
        group_norm_output_name = output_name + "_NHWC" if self.channels_last else output_name

        # NCHW to NHWC
        if self.channels_last:
            transpose_input = helper.make_node(
                "Transpose",
                [input_name],
                [group_norm_input_name],
                name=self.model.create_node_name("Transpose", name_prefix="Transpose_NCHW_to_NHWC"),
                perm=[0, 2, 3, 1],
            )
            self.nodes_to_add.append(transpose_input)
            self.node_name_to_graph_name[transpose_input.name] = self.this_graph_name

        new_node = helper.make_node(
            "GroupNorm",
            inputs=[group_norm_input_name, group_norm_name + "_gamma", group_norm_name + "_beta"],
            outputs=[group_norm_output_name],
            name=group_norm_name,
        )

        new_node.attribute.extend(instance_norm.attribute)

        new_node.attribute.extend([helper.make_attribute("groups", num_groups)])
        new_node.attribute.extend([helper.make_attribute("activation", 1 if has_swish_activation else 0)])

        if not self.channels_last:
            new_node.attribute.extend([helper.make_attribute("channels_last", 0)])

        new_node.domain = "com.microsoft"
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        # NHWC to NCHW
        if self.channels_last:
            transpose_output = helper.make_node(
                "Transpose",
                [group_norm_output_name],
                [output_name],
                name=self.model.create_node_name("Transpose", name_prefix="Transpose_NHWC_to_NCHW"),
                perm=[0, 3, 1, 2],
            )
            self.nodes_to_add.append(transpose_output)
            self.node_name_to_graph_name[transpose_output.name] = self.this_graph_name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\check.py ===
"""Validation of dependencies of packages"""

import logging
from contextlib import suppress
from email.parser import Parser
from functools import reduce
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import Tag, parse_tag
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import Version

from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.metadata import get_default_environment
from pip._internal.metadata.base import BaseDistribution
from pip._internal.req.req_install import InstallRequirement

logger = logging.getLogger(__name__)


class PackageDetails(NamedTuple):
    version: Version
    dependencies: List[Requirement]


# Shorthands
PackageSet = Dict[NormalizedName, PackageDetails]
Missing = Tuple[NormalizedName, Requirement]
Conflicting = Tuple[NormalizedName, Version, Requirement]

MissingDict = Dict[NormalizedName, List[Missing]]
ConflictingDict = Dict[NormalizedName, List[Conflicting]]
CheckResult = Tuple[MissingDict, ConflictingDict]
ConflictDetails = Tuple[PackageSet, CheckResult]


def create_package_set_from_installed() -> Tuple[PackageSet, bool]:
    """Converts a list of distributions into a PackageSet."""
    package_set = {}
    problems = False
    env = get_default_environment()
    for dist in env.iter_installed_distributions(local_only=False, skip=()):
        name = dist.canonical_name
        try:
            dependencies = list(dist.iter_dependencies())
            package_set[name] = PackageDetails(dist.version, dependencies)
        except (OSError, ValueError) as e:
            # Don't crash on unreadable or broken metadata.
            logger.warning("Error parsing dependencies of %s: %s", name, e)
            problems = True
    return package_set, problems


def check_package_set(
    package_set: PackageSet, should_ignore: Optional[Callable[[str], bool]] = None
) -> CheckResult:
    """Check if a package set is consistent

    If should_ignore is passed, it should be a callable that takes a
    package name and returns a boolean.
    """

    missing = {}
    conflicting = {}

    for package_name, package_detail in package_set.items():
        # Info about dependencies of package_name
        missing_deps: Set[Missing] = set()
        conflicting_deps: Set[Conflicting] = set()

        if should_ignore and should_ignore(package_name):
            continue

        for req in package_detail.dependencies:
            name = canonicalize_name(req.name)

            # Check if it's missing
            if name not in package_set:
                missed = True
                if req.marker is not None:
                    missed = req.marker.evaluate({"extra": ""})
                if missed:
                    missing_deps.add((name, req))
                continue

            # Check if there's a conflict
            version = package_set[name].version
            if not req.specifier.contains(version, prereleases=True):
                conflicting_deps.add((name, version, req))

        if missing_deps:
            missing[package_name] = sorted(missing_deps, key=str)
        if conflicting_deps:
            conflicting[package_name] = sorted(conflicting_deps, key=str)

    return missing, conflicting


def check_install_conflicts(to_install: List[InstallRequirement]) -> ConflictDetails:
    """For checking if the dependency graph would be consistent after \
    installing given requirements
    """
    # Start from the current state
    package_set, _ = create_package_set_from_installed()
    # Install packages
    would_be_installed = _simulate_installation_of(to_install, package_set)

    # Only warn about directly-dependent packages; create a whitelist of them
    whitelist = _create_whitelist(would_be_installed, package_set)

    return (
        package_set,
        check_package_set(
            package_set, should_ignore=lambda name: name not in whitelist
        ),
    )


def check_unsupported(
    packages: Iterable[BaseDistribution],
    supported_tags: Iterable[Tag],
) -> Generator[BaseDistribution, None, None]:
    for p in packages:
        with suppress(FileNotFoundError):
            wheel_file = p.read_text("WHEEL")
            wheel_tags: FrozenSet[Tag] = reduce(
                frozenset.union,
                map(parse_tag, Parser().parsestr(wheel_file).get_all("Tag", [])),
                frozenset(),
            )
            if wheel_tags.isdisjoint(supported_tags):
                yield p


def _simulate_installation_of(
    to_install: List[InstallRequirement], package_set: PackageSet
) -> Set[NormalizedName]:
    """Computes the version of packages after installing to_install."""
    # Keep track of packages that were installed
    installed = set()

    # Modify it as installing requirement_set would (assuming no errors)
    for inst_req in to_install:
        abstract_dist = make_distribution_for_install_requirement(inst_req)
        dist = abstract_dist.get_metadata_distribution()
        name = dist.canonical_name
        package_set[name] = PackageDetails(dist.version, list(dist.iter_dependencies()))

        installed.add(name)

    return installed


def _create_whitelist(
    would_be_installed: Set[NormalizedName], package_set: PackageSet
) -> Set[NormalizedName]:
    packages_affected = set(would_be_installed)

    for package_name in package_set:
        if package_name in packages_affected:
            continue

        for req in package_set[package_name].dependencies:
            if canonicalize_name(req.name) in packages_affected:
                packages_affected.add(package_name)
                break

    return packages_affected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\algorithms.py ===
from sympy.core.containers import Tuple
from sympy.core.numbers import oo
from sympy.core.relational import (Gt, Lt)
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import And
from sympy.codegen.ast import (
    Assignment, AddAugmentedAssignment, break_, CodeBlock, Declaration, FunctionDefinition,
    Print, Return, Scope, While, Variable, Pointer, real
)
from sympy.codegen.cfunctions import isnan

""" This module collects functions for constructing ASTs representing algorithms. """

def newtons_method(expr, wrt, atol=1e-12, delta=None, *, rtol=4e-16, debug=False,
                   itermax=None, counter=None, delta_fn=lambda e, x: -e/e.diff(x),
                   cse=False, handle_nan=None,
                   bounds=None):
    """ Generates an AST for Newton-Raphson method (a root-finding algorithm).

    Explanation
    ===========

    Returns an abstract syntax tree (AST) based on ``sympy.codegen.ast`` for Netwon's
    method of root-finding.

    Parameters
    ==========

    expr : expression
    wrt : Symbol
        With respect to, i.e. what is the variable.
    atol : number or expression
        Absolute tolerance (stopping criterion)
    rtol : number or expression
        Relative tolerance (stopping criterion)
    delta : Symbol
        Will be a ``Dummy`` if ``None``.
    debug : bool
        Whether to print convergence information during iterations
    itermax : number or expr
        Maximum number of iterations.
    counter : Symbol
        Will be a ``Dummy`` if ``None``.
    delta_fn: Callable[[Expr, Symbol], Expr]
        computes the step, default is newtons method. For e.g. Halley's method
        use delta_fn=lambda e, x: -2*e*e.diff(x)/(2*e.diff(x)**2 - e*e.diff(x, 2))
    cse: bool
        Perform common sub-expression elimination on delta expression
    handle_nan: Token
        How to handle occurrence of not-a-number (NaN).
    bounds: Optional[tuple[Expr, Expr]]
        Perform optimization within bounds

    Examples
    ========

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.ast import Assignment
    >>> from sympy.codegen.algorithms import newtons_method
    >>> x, dx, atol = symbols('x dx atol')
    >>> expr = cos(x) - x**3
    >>> algo = newtons_method(expr, x, atol=atol, delta=dx)
    >>> algo.has(Assignment(dx, -expr/expr.diff(x)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Newton%27s_method

    """

    if delta is None:
        delta = Dummy()
        Wrapper = Scope
        name_d = 'delta'
    else:
        Wrapper = lambda x: x
        name_d = delta.name

    delta_expr = delta_fn(expr, wrt)
    if cse:
        from sympy.simplify.cse_main import cse
        cses, (red,) = cse([delta_expr.factor()])
        whl_bdy = [Assignment(dum, sub_e) for dum, sub_e in cses]
        whl_bdy += [Assignment(delta, red)]
    else:
        whl_bdy = [Assignment(delta, delta_expr)]
    if handle_nan is not None:
        whl_bdy += [While(isnan(delta), CodeBlock(handle_nan, break_))]
    whl_bdy += [AddAugmentedAssignment(wrt, delta)]
    if bounds is not None:
        whl_bdy += [Assignment(wrt, Min(Max(wrt, bounds[0]), bounds[1]))]
    if debug:
        prnt = Print([wrt, delta], r"{}=%12.5g {}=%12.5g\n".format(wrt.name, name_d))
        whl_bdy += [prnt]
    req = Gt(Abs(delta), atol + rtol*Abs(wrt))
    declars = [Declaration(Variable(delta, type=real, value=oo))]
    if itermax is not None:
        counter = counter or Dummy(integer=True)
        v_counter = Variable.deduced(counter, 0)
        declars.append(Declaration(v_counter))
        whl_bdy.append(AddAugmentedAssignment(counter, 1))
        req = And(req, Lt(counter, itermax))
    whl = While(req, CodeBlock(*whl_bdy))
    blck = declars
    if debug:
        blck.append(Print([wrt], r"{}=%12.5g\n".format(wrt.name)))
    blck += [whl]
    return Wrapper(CodeBlock(*blck))


def _symbol_of(arg):
    if isinstance(arg, Declaration):
        arg = arg.variable.symbol
    elif isinstance(arg, Variable):
        arg = arg.symbol
    return arg


def newtons_method_function(expr, wrt, params=None, func_name="newton", attrs=Tuple(), *, delta=None, **kwargs):
    """ Generates an AST for a function implementing the Newton-Raphson method.

    Parameters
    ==========

    expr : expression
    wrt : Symbol
        With respect to, i.e. what is the variable
    params : iterable of symbols
        Symbols appearing in expr that are taken as constants during the iterations
        (these will be accepted as parameters to the generated function).
    func_name : str
        Name of the generated function.
    attrs : Tuple
        Attribute instances passed as ``attrs`` to ``FunctionDefinition``.
    \\*\\*kwargs :
        Keyword arguments passed to :func:`sympy.codegen.algorithms.newtons_method`.

    Examples
    ========

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.algorithms import newtons_method_function
    >>> from sympy.codegen.pyutils import render_as_module
    >>> x = symbols('x')
    >>> expr = cos(x) - x**3
    >>> func = newtons_method_function(expr, x)
    >>> py_mod = render_as_module(func)  # source code as string
    >>> namespace = {}
    >>> exec(py_mod, namespace, namespace)
    >>> res = eval('newton(0.5)', namespace)
    >>> abs(res - 0.865474033102) < 1e-12
    True

    See Also
    ========

    sympy.codegen.algorithms.newtons_method

    """
    if params is None:
        params = (wrt,)
    pointer_subs = {p.symbol: Symbol('(*%s)' % p.symbol.name)
                    for p in params if isinstance(p, Pointer)}
    if delta is None:
        delta = Symbol('d_' + wrt.name)
        if expr.has(delta):
            delta = None  # will use Dummy
    algo = newtons_method(expr, wrt, delta=delta, **kwargs).xreplace(pointer_subs)
    if isinstance(algo, Scope):
        algo = algo.body
    not_in_params = expr.free_symbols.difference({_symbol_of(p) for p in params})
    if not_in_params:
        raise ValueError("Missing symbols in params: %s" % ', '.join(map(str, not_in_params)))
    declars = tuple(Variable(p, real) for p in params)
    body = CodeBlock(algo, Return(wrt))
    return FunctionDefinition(real, func_name, declars, body, attrs=attrs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\precedence.py ===
"""A module providing information about the necessity of brackets"""


# Default precedence values for some basic types
PRECEDENCE = {
    "Lambda": 1,
    "Xor": 10,
    "Or": 20,
    "And": 30,
    "Relational": 35,
    "Add": 40,
    "Mul": 50,
    "Pow": 60,
    "Func": 70,
    "Not": 100,
    "Atom": 1000,
    "BitwiseOr": 36,
    "BitwiseXor": 37,
    "BitwiseAnd": 38
}

# A dictionary assigning precedence values to certain classes. These values are
# treated like they were inherited, so not every single class has to be named
# here.
# Do not use this with printers other than StrPrinter
PRECEDENCE_VALUES = {
    "Equivalent": PRECEDENCE["Xor"],
    "Xor": PRECEDENCE["Xor"],
    "Implies": PRECEDENCE["Xor"],
    "Or": PRECEDENCE["Or"],
    "And": PRECEDENCE["And"],
    "Add": PRECEDENCE["Add"],
    "Pow": PRECEDENCE["Pow"],
    "Relational": PRECEDENCE["Relational"],
    "Sub": PRECEDENCE["Add"],
    "Not": PRECEDENCE["Not"],
    "Function" : PRECEDENCE["Func"],
    "NegativeInfinity": PRECEDENCE["Add"],
    "MatAdd": PRECEDENCE["Add"],
    "MatPow": PRECEDENCE["Pow"],
    "MatrixSolve": PRECEDENCE["Mul"],
    "Mod": PRECEDENCE["Mul"],
    "TensAdd": PRECEDENCE["Add"],
    # As soon as `TensMul` is a subclass of `Mul`, remove this:
    "TensMul": PRECEDENCE["Mul"],
    "HadamardProduct": PRECEDENCE["Mul"],
    "HadamardPower": PRECEDENCE["Pow"],
    "KroneckerProduct": PRECEDENCE["Mul"],
    "Equality": PRECEDENCE["Mul"],
    "Unequality": PRECEDENCE["Mul"],
}

# Sometimes it's not enough to assign a fixed precedence value to a
# class. Then a function can be inserted in this dictionary that takes
# an instance of this class as argument and returns the appropriate
# precedence value.

# Precedence functions


def precedence_Mul(item):
    from sympy.core.function import Function
    if any(hasattr(arg, 'precedence') and isinstance(arg, Function) and
           arg.precedence < PRECEDENCE["Mul"] for arg in item.args):
        return PRECEDENCE["Mul"]

    if item.could_extract_minus_sign():
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]


def precedence_Rational(item):
    if item.p < 0:
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]


def precedence_Integer(item):
    if item.p < 0:
        return PRECEDENCE["Add"]
    return PRECEDENCE["Atom"]


def precedence_Float(item):
    if item < 0:
        return PRECEDENCE["Add"]
    return PRECEDENCE["Atom"]


def precedence_PolyElement(item):
    if item.is_generator:
        return PRECEDENCE["Atom"]
    elif item.is_ground:
        return precedence(item.coeff(1))
    elif item.is_term:
        return PRECEDENCE["Mul"]
    else:
        return PRECEDENCE["Add"]


def precedence_FracElement(item):
    if item.denom == 1:
        return precedence_PolyElement(item.numer)
    else:
        return PRECEDENCE["Mul"]


def precedence_UnevaluatedExpr(item):
    return precedence(item.args[0]) - 0.5


PRECEDENCE_FUNCTIONS = {
    "Integer": precedence_Integer,
    "Mul": precedence_Mul,
    "Rational": precedence_Rational,
    "Float": precedence_Float,
    "PolyElement": precedence_PolyElement,
    "FracElement": precedence_FracElement,
    "UnevaluatedExpr": precedence_UnevaluatedExpr,
}


def precedence(item):
    """Returns the precedence of a given object.

    This is the precedence for StrPrinter.
    """
    if hasattr(item, "precedence"):
        return item.precedence
    if not isinstance(item, type):
        for i in type(item).mro():
            n = i.__name__
            if n in PRECEDENCE_FUNCTIONS:
                return PRECEDENCE_FUNCTIONS[n](item)
            elif n in PRECEDENCE_VALUES:
                return PRECEDENCE_VALUES[n]
    return PRECEDENCE["Atom"]


PRECEDENCE_TRADITIONAL = PRECEDENCE.copy()
PRECEDENCE_TRADITIONAL['Integral'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Sum'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Product'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Limit'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Derivative'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['TensorProduct'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Transpose'] = PRECEDENCE["Pow"]
PRECEDENCE_TRADITIONAL['Adjoint'] = PRECEDENCE["Pow"]
PRECEDENCE_TRADITIONAL['Dot'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Cross'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Gradient'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Divergence'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Curl'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Laplacian'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Union'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['Intersection'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['Complement'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['SymmetricDifference'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['ProductSet'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['DotProduct'] = PRECEDENCE_TRADITIONAL['Dot']


def precedence_traditional(item):
    """Returns the precedence of a given object according to the
    traditional rules of mathematics.

    This is the precedence for the LaTeX and pretty printer.
    """
    # Integral, Sum, Product, Limit have the precedence of Mul in LaTeX,
    # the precedence of Atom for other printers:
    from sympy.core.expr import UnevaluatedExpr

    if isinstance(item, UnevaluatedExpr):
        return precedence_traditional(item.args[0])

    n = item.__class__.__name__
    if n in PRECEDENCE_TRADITIONAL:
        return PRECEDENCE_TRADITIONAL[n]

    return precedence(item)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_highlevel_ssl_helpers.py ===
from __future__ import annotations

import ssl
from typing import TYPE_CHECKING, NoReturn, TypeVar

import trio

from ._highlevel_open_tcp_stream import DEFAULT_DELAY

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ._highlevel_socket import SocketStream


# It might have been nice to take a ssl_protocols= argument here to set up
# NPN/ALPN, but to do this we have to mutate the context object, which is OK
# if it's one we created, but not OK if it's one that was passed in... and
# the one major protocol using NPN/ALPN is HTTP/2, which mandates that you use
# a specially configured SSLContext anyway! I also thought maybe we could copy
# the given SSLContext and then mutate the copy, but it's no good as SSLContext
# objects can't be copied: https://bugs.python.org/issue33023.
# So... let's punt on that for now. Hopefully we'll be getting a new Python
# TLS API soon and can revisit this then.
async def open_ssl_over_tcp_stream(
    host: str | bytes,
    port: int,
    *,
    https_compatible: bool = False,
    ssl_context: ssl.SSLContext | None = None,
    happy_eyeballs_delay: float | None = DEFAULT_DELAY,
) -> trio.SSLStream[SocketStream]:
    """Make a TLS-encrypted Connection to the given host and port over TCP.

    This is a convenience wrapper that calls :func:`open_tcp_stream` and
    wraps the result in an :class:`~trio.SSLStream`.

    This function does not perform the TLS handshake; you can do it
    manually by calling :meth:`~trio.SSLStream.do_handshake`, or else
    it will be performed automatically the first time you send or receive
    data.

    Args:
      host (bytes or str): The host to connect to. We require the server
          to have a TLS certificate valid for this hostname.
      port (int): The port to connect to.
      https_compatible (bool): Set this to True if you're connecting to a web
          server. See :class:`~trio.SSLStream` for details. Default:
          False.
      ssl_context (:class:`~ssl.SSLContext` or None): The SSL context to
          use. If None (the default), :func:`ssl.create_default_context`
          will be called to create a context.
      happy_eyeballs_delay (float): See :func:`open_tcp_stream`.

    Returns:
      trio.SSLStream: the encrypted connection to the server.

    """
    tcp_stream = await trio.open_tcp_stream(
        host,
        port,
        happy_eyeballs_delay=happy_eyeballs_delay,
    )
    if ssl_context is None:
        ssl_context = ssl.create_default_context()

        if hasattr(ssl, "OP_IGNORE_UNEXPECTED_EOF"):
            ssl_context.options &= ~ssl.OP_IGNORE_UNEXPECTED_EOF

    return trio.SSLStream(
        tcp_stream,
        ssl_context,
        server_hostname=host,
        https_compatible=https_compatible,
    )


async def open_ssl_over_tcp_listeners(
    port: int,
    ssl_context: ssl.SSLContext,
    *,
    host: str | bytes | None = None,
    https_compatible: bool = False,
    backlog: int | None = None,
) -> list[trio.SSLListener[SocketStream]]:
    """Start listening for SSL/TLS-encrypted TCP connections to the given port.

    Args:
      port (int): The port to listen on. See :func:`open_tcp_listeners`.
      ssl_context (~ssl.SSLContext): The SSL context to use for all incoming
          connections.
      host (str, bytes, or None): The address to bind to; use ``None`` to bind
          to the wildcard address. See :func:`open_tcp_listeners`.
      https_compatible (bool): See :class:`~trio.SSLStream` for details.
      backlog (int or None): See :func:`open_tcp_listeners` for details.

    """
    tcp_listeners = await trio.open_tcp_listeners(port, host=host, backlog=backlog)
    ssl_listeners = [
        trio.SSLListener(tcp_listener, ssl_context, https_compatible=https_compatible)
        for tcp_listener in tcp_listeners
    ]
    return ssl_listeners


async def serve_ssl_over_tcp(
    handler: Callable[[trio.SSLStream[SocketStream]], Awaitable[object]],
    port: int,
    ssl_context: ssl.SSLContext,
    *,
    host: str | bytes | None = None,
    https_compatible: bool = False,
    backlog: int | None = None,
    handler_nursery: trio.Nursery | None = None,
    task_status: trio.TaskStatus[
        list[trio.SSLListener[SocketStream]]
    ] = trio.TASK_STATUS_IGNORED,
) -> NoReturn:
    """Listen for incoming TCP connections, and for each one start a task
    running ``handler(stream)``.

    This is a thin convenience wrapper around
    :func:`open_ssl_over_tcp_listeners` and :func:`serve_listeners` – see them
    for full details.

    .. warning::

       If ``handler`` raises an exception, then this function doesn't do
       anything special to catch it – so by default the exception will
       propagate out and crash your server. If you don't want this, then catch
       exceptions inside your ``handler``, or use a ``handler_nursery`` object
       that responds to exceptions in some other way.

    When used with ``nursery.start`` you get back the newly opened listeners.
    See the documentation for :func:`serve_tcp` for an example where this is
    useful.

    Args:
      handler: The handler to start for each incoming connection. Passed to
          :func:`serve_listeners`.

      port (int): The port to listen on. Use 0 to let the kernel pick
          an open port. Ultimately passed to :func:`open_tcp_listeners`.

      ssl_context (~ssl.SSLContext): The SSL context to use for all incoming
          connections. Passed to :func:`open_ssl_over_tcp_listeners`.

      host (str, bytes, or None): The address to bind to; use ``None`` to bind
          to the wildcard address. Ultimately passed to
          :func:`open_tcp_listeners`.

      https_compatible (bool): Set this to True if you want to use
          "HTTPS-style" TLS. See :class:`~trio.SSLStream` for details.

      backlog (int or None): See :class:`~trio.SSLStream` for details.

      handler_nursery: The nursery to start handlers in, or None to use an
          internal nursery. Passed to :func:`serve_listeners`.

      task_status: This function can be used with ``nursery.start``.

    Returns:
      This function only returns when cancelled.

    """
    listeners = await trio.open_ssl_over_tcp_listeners(
        port,
        ssl_context,
        host=host,
        https_compatible=https_compatible,
        backlog=backlog,
    )
    await trio.serve_listeners(
        handler,
        listeners,
        handler_nursery=handler_nursery,
        task_status=task_status,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\symbol_database.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""A database of Python protocol buffer generated symbols.

SymbolDatabase is the MessageFactory for messages generated at compile time,
and makes it easy to create new instances of a registered type, given only the
type's protocol buffer symbol name.

Example usage::

  db = symbol_database.SymbolDatabase()

  # Register symbols of interest, from one or multiple files.
  db.RegisterFileDescriptor(my_proto_pb2.DESCRIPTOR)
  db.RegisterMessage(my_proto_pb2.MyMessage)
  db.RegisterEnumDescriptor(my_proto_pb2.MyEnum.DESCRIPTOR)

  # The database can be used as a MessageFactory, to generate types based on
  # their name:
  types = db.GetMessages(['my_proto.proto'])
  my_message_instance = types['MyMessage']()

  # The database's underlying descriptor pool can be queried, so it's not
  # necessary to know a type's filename to be able to generate it:
  filename = db.pool.FindFileContainingSymbol('MyMessage')
  my_message_instance = db.GetMessages([filename])['MyMessage']()

  # This functionality is also provided directly via a convenience method:
  my_message_instance = db.GetSymbol('MyMessage')()
"""

import warnings

from google.protobuf.internal import api_implementation
from google.protobuf import descriptor_pool
from google.protobuf import message_factory


class SymbolDatabase():
  """A database of Python generated symbols."""

  # local cache of registered classes.
  _classes = {}

  def __init__(self, pool=None):
    """Initializes a new SymbolDatabase."""
    self.pool = pool or descriptor_pool.DescriptorPool()

  def RegisterMessage(self, message):
    """Registers the given message type in the local database.

    Calls to GetSymbol() and GetMessages() will return messages registered here.

    Args:
      message: A :class:`google.protobuf.message.Message` subclass (or
        instance); its descriptor will be registered.

    Returns:
      The provided message.
    """

    desc = message.DESCRIPTOR
    self._classes[desc] = message
    self.RegisterMessageDescriptor(desc)
    return message

  def RegisterMessageDescriptor(self, message_descriptor):
    """Registers the given message descriptor in the local database.

    Args:
      message_descriptor (Descriptor): the message descriptor to add.
    """
    if api_implementation.Type() == 'python':
      # pylint: disable=protected-access
      self.pool._AddDescriptor(message_descriptor)

  def RegisterEnumDescriptor(self, enum_descriptor):
    """Registers the given enum descriptor in the local database.

    Args:
      enum_descriptor (EnumDescriptor): The enum descriptor to register.

    Returns:
      EnumDescriptor: The provided descriptor.
    """
    if api_implementation.Type() == 'python':
      # pylint: disable=protected-access
      self.pool._AddEnumDescriptor(enum_descriptor)
    return enum_descriptor

  def RegisterServiceDescriptor(self, service_descriptor):
    """Registers the given service descriptor in the local database.

    Args:
      service_descriptor (ServiceDescriptor): the service descriptor to
        register.
    """
    if api_implementation.Type() == 'python':
      # pylint: disable=protected-access
      self.pool._AddServiceDescriptor(service_descriptor)

  def RegisterFileDescriptor(self, file_descriptor):
    """Registers the given file descriptor in the local database.

    Args:
      file_descriptor (FileDescriptor): The file descriptor to register.
    """
    if api_implementation.Type() == 'python':
      # pylint: disable=protected-access
      self.pool._InternalAddFileDescriptor(file_descriptor)

  def GetSymbol(self, symbol):
    """Tries to find a symbol in the local database.

    Currently, this method only returns message.Message instances, however, if
    may be extended in future to support other symbol types.

    Args:
      symbol (str): a protocol buffer symbol.

    Returns:
      A Python class corresponding to the symbol.

    Raises:
      KeyError: if the symbol could not be found.
    """

    return self._classes[self.pool.FindMessageTypeByName(symbol)]

  def GetMessages(self, files):
    # TODO: Fix the differences with MessageFactory.
    """Gets all registered messages from a specified file.

    Only messages already created and registered will be returned; (this is the
    case for imported _pb2 modules)
    But unlike MessageFactory, this version also returns already defined nested
    messages, but does not register any message extensions.

    Args:
      files (list[str]): The file names to extract messages from.

    Returns:
      A dictionary mapping proto names to the message classes.

    Raises:
      KeyError: if a file could not be found.
    """

    def _GetAllMessages(desc):
      """Walk a message Descriptor and recursively yields all message names."""
      yield desc
      for msg_desc in desc.nested_types:
        for nested_desc in _GetAllMessages(msg_desc):
          yield nested_desc

    result = {}
    for file_name in files:
      file_desc = self.pool.FindFileByName(file_name)
      for msg_desc in file_desc.message_types_by_name.values():
        for desc in _GetAllMessages(msg_desc):
          try:
            result[desc.full_name] = self._classes[desc]
          except KeyError:
            # This descriptor has no registered class, skip it.
            pass
    return result


_DEFAULT = SymbolDatabase(pool=descriptor_pool.Default())


def Default():
  """Returns the default SymbolDatabase."""
  return _DEFAULT

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\__init__.py ===
#   __
#  /__)  _  _     _   _ _/   _
# / (   (- (/ (/ (- _)  /  _)
#          /

"""
Requests HTTP Library
~~~~~~~~~~~~~~~~~~~~~

Requests is an HTTP library, written in Python, for human beings.
Basic GET usage:

   >>> import requests
   >>> r = requests.get('https://www.python.org')
   >>> r.status_code
   200
   >>> b'Python is a programming language' in r.content
   True

... or POST:

   >>> payload = dict(key1='value1', key2='value2')
   >>> r = requests.post('https://httpbin.org/post', data=payload)
   >>> print(r.text)
   {
     ...
     "form": {
       "key1": "value1",
       "key2": "value2"
     },
     ...
   }

The other HTTP methods are supported - see `requests.api`. Full documentation
is at <https://requests.readthedocs.io>.

:copyright: (c) 2017 by Kenneth Reitz.
:license: Apache 2.0, see LICENSE for more details.
"""

import warnings

from pip._vendor import urllib3

from .exceptions import RequestsDependencyWarning

charset_normalizer_version = None
chardet_version = None


def check_compatibility(urllib3_version, chardet_version, charset_normalizer_version):
    urllib3_version = urllib3_version.split(".")
    assert urllib3_version != ["dev"]  # Verify urllib3 isn't installed from git.

    # Sometimes, urllib3 only reports its version as 16.1.
    if len(urllib3_version) == 2:
        urllib3_version.append("0")

    # Check urllib3 for compatibility.
    major, minor, patch = urllib3_version  # noqa: F811
    major, minor, patch = int(major), int(minor), int(patch)
    # urllib3 >= 1.21.1
    assert major >= 1
    if major == 1:
        assert minor >= 21

    # Check charset_normalizer for compatibility.
    if chardet_version:
        major, minor, patch = chardet_version.split(".")[:3]
        major, minor, patch = int(major), int(minor), int(patch)
        # chardet_version >= 3.0.2, < 6.0.0
        assert (3, 0, 2) <= (major, minor, patch) < (6, 0, 0)
    elif charset_normalizer_version:
        major, minor, patch = charset_normalizer_version.split(".")[:3]
        major, minor, patch = int(major), int(minor), int(patch)
        # charset_normalizer >= 2.0.0 < 4.0.0
        assert (2, 0, 0) <= (major, minor, patch) < (4, 0, 0)
    else:
        # pip does not need or use character detection
        pass


def _check_cryptography(cryptography_version):
    # cryptography < 1.3.4
    try:
        cryptography_version = list(map(int, cryptography_version.split(".")))
    except ValueError:
        return

    if cryptography_version < [1, 3, 4]:
        warning = "Old version of cryptography ({}) may cause slowdown.".format(
            cryptography_version
        )
        warnings.warn(warning, RequestsDependencyWarning)


# Check imported dependencies for compatibility.
try:
    check_compatibility(
        urllib3.__version__, chardet_version, charset_normalizer_version
    )
except (AssertionError, ValueError):
    warnings.warn(
        "urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
        "version!".format(
            urllib3.__version__, chardet_version, charset_normalizer_version
        ),
        RequestsDependencyWarning,
    )

# Attempt to enable urllib3's fallback for SNI support
# if the standard library doesn't support SNI or the
# 'ssl' library isn't available.
try:
    # Note: This logic prevents upgrading cryptography on Windows, if imported
    #       as part of pip.
    from pip._internal.utils.compat import WINDOWS
    if not WINDOWS:
        raise ImportError("pip internals: don't import cryptography on Windows")
    try:
        import ssl
    except ImportError:
        ssl = None

    if not getattr(ssl, "HAS_SNI", False):
        from pip._vendor.urllib3.contrib import pyopenssl

        pyopenssl.inject_into_urllib3()

        # Check cryptography version
        from cryptography import __version__ as cryptography_version

        _check_cryptography(cryptography_version)
except ImportError:
    pass

# urllib3's DependencyWarnings should be silenced.
from pip._vendor.urllib3.exceptions import DependencyWarning

warnings.simplefilter("ignore", DependencyWarning)

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

from . import packages, utils
from .__version__ import (
    __author__,
    __author_email__,
    __build__,
    __cake__,
    __copyright__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
)
from .api import delete, get, head, options, patch, post, put, request
from .exceptions import (
    ConnectionError,
    ConnectTimeout,
    FileModeWarning,
    HTTPError,
    JSONDecodeError,
    ReadTimeout,
    RequestException,
    Timeout,
    TooManyRedirects,
    URLRequired,
)
from .models import PreparedRequest, Request, Response
from .sessions import Session, session
from .status_codes import codes

logging.getLogger(__name__).addHandler(NullHandler())

# FileModeWarnings go off per the default.
warnings.simplefilter("default", FileModeWarning, append=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\_typing.py ===
# orm/_typing.py
# Copyright (C) 2022-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

import operator
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from ..engine.interfaces import _CoreKnownExecutionOptions
from ..sql import roles
from ..sql._orm_types import DMLStrategyArgument as DMLStrategyArgument
from ..sql._orm_types import (
    SynchronizeSessionArgument as SynchronizeSessionArgument,
)
from ..sql._typing import _HasClauseElement
from ..sql.elements import ColumnElement
from ..util.typing import Protocol
from ..util.typing import TypeGuard

if TYPE_CHECKING:
    from .attributes import AttributeImpl
    from .attributes import CollectionAttributeImpl
    from .attributes import HasCollectionAdapter
    from .attributes import QueryableAttribute
    from .base import PassiveFlag
    from .decl_api import registry as _registry_type
    from .interfaces import InspectionAttr
    from .interfaces import MapperProperty
    from .interfaces import ORMOption
    from .interfaces import UserDefinedOption
    from .mapper import Mapper
    from .relationships import RelationshipProperty
    from .state import InstanceState
    from .util import AliasedClass
    from .util import AliasedInsp
    from ..sql._typing import _CE
    from ..sql.base import ExecutableOption

_T = TypeVar("_T", bound=Any)


_T_co = TypeVar("_T_co", bound=Any, covariant=True)

_O = TypeVar("_O", bound=object)
"""The 'ORM mapped object' type.

"""


if TYPE_CHECKING:
    _RegistryType = _registry_type

_InternalEntityType = Union["Mapper[_T]", "AliasedInsp[_T]"]

_ExternalEntityType = Union[Type[_T], "AliasedClass[_T]"]

_EntityType = Union[
    Type[_T], "AliasedClass[_T]", "Mapper[_T]", "AliasedInsp[_T]"
]


_ClassDict = Mapping[str, Any]
_InstanceDict = Dict[str, Any]

_IdentityKeyType = Tuple[Type[_T], Tuple[Any, ...], Optional[Any]]

_ORMColumnExprArgument = Union[
    ColumnElement[_T],
    _HasClauseElement[_T],
    roles.ExpressionElementRole[_T],
]


_ORMCOLEXPR = TypeVar("_ORMCOLEXPR", bound=ColumnElement[Any])


class _OrmKnownExecutionOptions(_CoreKnownExecutionOptions, total=False):
    populate_existing: bool
    autoflush: bool
    synchronize_session: SynchronizeSessionArgument
    dml_strategy: DMLStrategyArgument
    is_delete_using: bool
    is_update_from: bool
    render_nulls: bool


OrmExecuteOptionsParameter = Union[
    _OrmKnownExecutionOptions, Mapping[str, Any]
]


class _ORMAdapterProto(Protocol):
    """protocol for the :class:`.AliasedInsp._orm_adapt_element` method
    which is a synonym for :class:`.AliasedInsp._adapt_element`.


    """

    def __call__(self, obj: _CE, key: Optional[str] = None) -> _CE: ...


class _LoaderCallable(Protocol):
    def __call__(
        self, state: InstanceState[Any], passive: PassiveFlag
    ) -> Any: ...


def is_orm_option(
    opt: ExecutableOption,
) -> TypeGuard[ORMOption]:
    return not opt._is_core


def is_user_defined_option(
    opt: ExecutableOption,
) -> TypeGuard[UserDefinedOption]:
    return not opt._is_core and opt._is_user_defined  # type: ignore


def is_composite_class(obj: Any) -> bool:
    # inlining is_dataclass(obj)
    return hasattr(obj, "__composite_values__") or hasattr(
        obj, "__dataclass_fields__"
    )


if TYPE_CHECKING:

    def insp_is_mapper_property(
        obj: Any,
    ) -> TypeGuard[MapperProperty[Any]]: ...

    def insp_is_mapper(obj: Any) -> TypeGuard[Mapper[Any]]: ...

    def insp_is_aliased_class(obj: Any) -> TypeGuard[AliasedInsp[Any]]: ...

    def insp_is_attribute(
        obj: InspectionAttr,
    ) -> TypeGuard[QueryableAttribute[Any]]: ...

    def attr_is_internal_proxy(
        obj: InspectionAttr,
    ) -> TypeGuard[QueryableAttribute[Any]]: ...

    def prop_is_relationship(
        prop: MapperProperty[Any],
    ) -> TypeGuard[RelationshipProperty[Any]]: ...

    def is_collection_impl(
        impl: AttributeImpl,
    ) -> TypeGuard[CollectionAttributeImpl]: ...

    def is_has_collection_adapter(
        impl: AttributeImpl,
    ) -> TypeGuard[HasCollectionAdapter]: ...

else:
    insp_is_mapper_property = operator.attrgetter("is_property")
    insp_is_mapper = operator.attrgetter("is_mapper")
    insp_is_aliased_class = operator.attrgetter("is_aliased_class")
    insp_is_attribute = operator.attrgetter("is_attribute")
    attr_is_internal_proxy = operator.attrgetter("_is_internal_proxy")
    is_collection_impl = operator.attrgetter("collection")
    prop_is_relationship = operator.attrgetter("_is_relationship")
    is_has_collection_adapter = operator.attrgetter(
        "_is_has_collection_adapter"
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\tools\test_gen_exports.py ===
import ast
import sys
from pathlib import Path

import pytest

from trio._tests.pytest_plugin import skip_if_optional_else_raise

# imports in gen_exports that are not in `install_requires` in setup.py
try:
    import astor  # noqa: F401
    import isort  # noqa: F401
except ImportError as error:
    skip_if_optional_else_raise(error)


from trio._tools.gen_exports import (
    File,
    create_passthrough_args,
    get_public_methods,
    process,
    run_linters,
    run_ruff,
)

from ..._core._tests.tutil import slow

SOURCE = '''from _run import _public
from collections import Counter

class Test:
    @_public
    def public_func(self):
        """With doc string"""

    @ignore_this
    @_public
    @another_decorator
    async def public_async_func(self) -> Counter:
        pass  # no doc string

    def not_public(self):
        pass

    async def not_public_async(self):
        pass
'''

IMPORT_1 = """\
from collections import Counter
"""

IMPORT_2 = """\
from collections import Counter
import os
"""

IMPORT_3 = """\
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections import Counter
"""


def test_get_public_methods() -> None:
    methods = list(get_public_methods(ast.parse(SOURCE)))
    assert {m.name for m in methods} == {"public_func", "public_async_func"}


def test_create_pass_through_args() -> None:
    testcases = [
        ("def f()", "()"),
        ("def f(one)", "(one)"),
        ("def f(one, two)", "(one, two)"),
        ("def f(one, *args)", "(one, *args)"),
        (
            "def f(one, *args, kw1, kw2=None, **kwargs)",
            "(one, *args, kw1=kw1, kw2=kw2, **kwargs)",
        ),
    ]

    for funcdef, expected in testcases:
        func_node = ast.parse(funcdef + ":\n  pass").body[0]
        assert isinstance(func_node, ast.FunctionDef)
        assert create_passthrough_args(func_node) == expected


skip_lints = pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="gen_exports is internal, black/isort only runs on CPython",
)


@slow
@skip_lints
@pytest.mark.parametrize("imports", [IMPORT_1, IMPORT_2, IMPORT_3])
def test_process(
    tmp_path: Path,
    imports: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    try:
        import black  # noqa: F401
    # there's no dedicated CI run that has astor+isort, but lacks black.
    except ImportError as error:  # pragma: no cover
        skip_if_optional_else_raise(error)

    modpath = tmp_path / "_module.py"
    genpath = tmp_path / "_generated_module.py"
    modpath.write_text(SOURCE, encoding="utf-8")
    file = File(modpath, "runner", platform="linux", imports=imports)
    assert not genpath.exists()
    with pytest.raises(SystemExit) as excinfo:
        process([file], do_test=True)
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Generated sources are outdated. Please regenerate." in captured.out
    with pytest.raises(SystemExit) as excinfo:
        process([file], do_test=False)
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Regenerated sources successfully." in captured.out
    assert genpath.exists()
    process([file], do_test=True)
    # But if we change the lookup path it notices
    with pytest.raises(SystemExit) as excinfo:
        process(
            [File(modpath, "runner.io_manager", platform="linux", imports=imports)],
            do_test=True,
        )
    assert excinfo.value.code == 1
    # Also if the platform is changed.
    with pytest.raises(SystemExit) as excinfo:
        process([File(modpath, "runner", imports=imports)], do_test=True)
    assert excinfo.value.code == 1


@skip_lints
def test_run_ruff(tmp_path: Path) -> None:
    """Test that processing properly fails if ruff does."""
    try:
        import ruff  # noqa: F401
    except ImportError as error:  # pragma: no cover
        skip_if_optional_else_raise(error)

    file = File(tmp_path / "module.py", "module")

    success, _ = run_ruff(file, "class not valid code ><")
    assert not success

    test_function = '''def combine_and(data: list[str]) -> str:
    """Join values of text, and have 'and' with the last one properly."""
    if len(data) >= 2:
        data[-1] = 'and ' + data[-1]
    if len(data) > 2:
        return ', '.join(data)
    return ' '.join(data)'''

    success, response = run_ruff(file, test_function)
    assert success
    assert response == test_function


@skip_lints
def test_lint_failure(tmp_path: Path) -> None:
    """Test that processing properly fails if black or ruff does."""
    try:
        import black  # noqa: F401
        import ruff  # noqa: F401
    except ImportError as error:  # pragma: no cover
        skip_if_optional_else_raise(error)

    file = File(tmp_path / "module.py", "module")

    with pytest.raises(SystemExit):
        run_linters(file, "class not valid code ><")

    with pytest.raises(SystemExit):
        run_linters(file, "import waffle\n;import trio")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\debughelpers.py ===
from __future__ import annotations

import typing as t

from jinja2.loaders import BaseLoader
from werkzeug.routing import RequestRedirect

from .blueprints import Blueprint
from .globals import request_ctx
from .sansio.app import App

if t.TYPE_CHECKING:
    from .sansio.scaffold import Scaffold
    from .wrappers import Request


class UnexpectedUnicodeError(AssertionError, UnicodeError):
    """Raised in places where we want some better error reporting for
    unexpected unicode or binary data.
    """


class DebugFilesKeyError(KeyError, AssertionError):
    """Raised from request.files during debugging.  The idea is that it can
    provide a better error message than just a generic KeyError/BadRequest.
    """

    def __init__(self, request: Request, key: str) -> None:
        form_matches = request.form.getlist(key)
        buf = [
            f"You tried to access the file {key!r} in the request.files"
            " dictionary but it does not exist. The mimetype for the"
            f" request is {request.mimetype!r} instead of"
            " 'multipart/form-data' which means that no file contents"
            " were transmitted. To fix this error you should provide"
            ' enctype="multipart/form-data" in your form.'
        ]
        if form_matches:
            names = ", ".join(repr(x) for x in form_matches)
            buf.append(
                "\n\nThe browser instead transmitted some file names. "
                f"This was submitted: {names}"
            )
        self.msg = "".join(buf)

    def __str__(self) -> str:
        return self.msg


class FormDataRoutingRedirect(AssertionError):
    """This exception is raised in debug mode if a routing redirect
    would cause the browser to drop the method or body. This happens
    when method is not GET, HEAD or OPTIONS and the status code is not
    307 or 308.
    """

    def __init__(self, request: Request) -> None:
        exc = request.routing_exception
        assert isinstance(exc, RequestRedirect)
        buf = [
            f"A request was sent to '{request.url}', but routing issued"
            f" a redirect to the canonical URL '{exc.new_url}'."
        ]

        if f"{request.base_url}/" == exc.new_url.partition("?")[0]:
            buf.append(
                " The URL was defined with a trailing slash. Flask"
                " will redirect to the URL with a trailing slash if it"
                " was accessed without one."
            )

        buf.append(
            " Send requests to the canonical URL, or use 307 or 308 for"
            " routing redirects. Otherwise, browsers will drop form"
            " data.\n\n"
            "This exception is only raised in debug mode."
        )
        super().__init__("".join(buf))


def attach_enctype_error_multidict(request: Request) -> None:
    """Patch ``request.files.__getitem__`` to raise a descriptive error
    about ``enctype=multipart/form-data``.

    :param request: The request to patch.
    :meta private:
    """
    oldcls = request.files.__class__

    class newcls(oldcls):  # type: ignore[valid-type, misc]
        def __getitem__(self, key: str) -> t.Any:
            try:
                return super().__getitem__(key)
            except KeyError as e:
                if key not in request.form:
                    raise

                raise DebugFilesKeyError(request, key).with_traceback(
                    e.__traceback__
                ) from None

    newcls.__name__ = oldcls.__name__
    newcls.__module__ = oldcls.__module__
    request.files.__class__ = newcls


def _dump_loader_info(loader: BaseLoader) -> t.Iterator[str]:
    yield f"class: {type(loader).__module__}.{type(loader).__name__}"
    for key, value in sorted(loader.__dict__.items()):
        if key.startswith("_"):
            continue
        if isinstance(value, (tuple, list)):
            if not all(isinstance(x, str) for x in value):
                continue
            yield f"{key}:"
            for item in value:
                yield f"  - {item}"
            continue
        elif not isinstance(value, (str, int, float, bool)):
            continue
        yield f"{key}: {value!r}"


def explain_template_loading_attempts(
    app: App,
    template: str,
    attempts: list[
        tuple[
            BaseLoader,
            Scaffold,
            tuple[str, str | None, t.Callable[[], bool] | None] | None,
        ]
    ],
) -> None:
    """This should help developers understand what failed"""
    info = [f"Locating template {template!r}:"]
    total_found = 0
    blueprint = None
    if request_ctx and request_ctx.request.blueprint is not None:
        blueprint = request_ctx.request.blueprint

    for idx, (loader, srcobj, triple) in enumerate(attempts):
        if isinstance(srcobj, App):
            src_info = f"application {srcobj.import_name!r}"
        elif isinstance(srcobj, Blueprint):
            src_info = f"blueprint {srcobj.name!r} ({srcobj.import_name})"
        else:
            src_info = repr(srcobj)

        info.append(f"{idx + 1:5}: trying loader of {src_info}")

        for line in _dump_loader_info(loader):
            info.append(f"       {line}")

        if triple is None:
            detail = "no match"
        else:
            detail = f"found ({triple[1] or '<string>'!r})"
            total_found += 1
        info.append(f"       -> {detail}")

    seems_fishy = False
    if total_found == 0:
        info.append("Error: the template could not be found.")
        seems_fishy = True
    elif total_found > 1:
        info.append("Warning: multiple loaders returned a match for the template.")
        seems_fishy = True

    if blueprint is not None and seems_fishy:
        info.append(
            "  The template was looked up from an endpoint that belongs"
            f" to the blueprint {blueprint!r}."
        )
        info.append("  Maybe you did not place a template in the right folder?")
        info.append("  See https://flask.palletsprojects.com/blueprints/#templates")

    app.logger.info("\n".join(info))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\sas\sasreader.py ===
"""
Read SAS sas7bdat or xport files.
"""
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    overload,
)

from pandas.util._decorators import doc

from pandas.core.shared_docs import _shared_docs

from pandas.io.common import stringify_path

if TYPE_CHECKING:
    from collections.abc import Hashable
    from types import TracebackType

    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
        Self,
    )

    from pandas import DataFrame


class ReaderBase(ABC):
    """
    Protocol for XportReader and SAS7BDATReader classes.
    """

    @abstractmethod
    def read(self, nrows: int | None = None) -> DataFrame:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: int = ...,
    iterator: bool = ...,
    compression: CompressionOptions = ...,
) -> ReaderBase:
    ...


@overload
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = ...,
    index: Hashable | None = ...,
    encoding: str | None = ...,
    chunksize: None = ...,
    iterator: bool = ...,
    compression: CompressionOptions = ...,
) -> DataFrame | ReaderBase:
    ...


@doc(decompression_options=_shared_docs["decompression_options"] % "filepath_or_buffer")
def read_sas(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    format: str | None = None,
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
) -> DataFrame | ReaderBase:
    """
    Read SAS files stored as either XPORT or SAS7BDAT format files.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.sas7bdat``.
    format : str {{'xport', 'sas7bdat'}} or None
        If None, file format is inferred from file extension. If 'xport' or
        'sas7bdat', uses the corresponding format.
    index : identifier of index column, defaults to None
        Identifier of column that should be used as index of the DataFrame.
    encoding : str, default is None
        Encoding for text data.  If None, text data are stored as raw bytes.
    chunksize : int
        Read file `chunksize` lines at a time, returns iterator.
    iterator : bool, defaults to False
        If True, returns an iterator for reading the file incrementally.
    {decompression_options}

    Returns
    -------
    DataFrame if iterator=False and chunksize=None, else SAS7BDATReader
    or XportReader

    Examples
    --------
    >>> df = pd.read_sas("sas_data.sas7bdat")  # doctest: +SKIP
    """
    if format is None:
        buffer_error_msg = (
            "If this is a buffer object rather "
            "than a string name, you must specify a format string"
        )
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        if not isinstance(filepath_or_buffer, str):
            raise ValueError(buffer_error_msg)
        fname = filepath_or_buffer.lower()
        if ".xpt" in fname:
            format = "xport"
        elif ".sas7bdat" in fname:
            format = "sas7bdat"
        else:
            raise ValueError(
                f"unable to infer format of SAS file from filename: {repr(fname)}"
            )

    reader: ReaderBase
    if format.lower() == "xport":
        from pandas.io.sas.sas_xport import XportReader

        reader = XportReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    elif format.lower() == "sas7bdat":
        from pandas.io.sas.sas7bdat import SAS7BDATReader

        reader = SAS7BDATReader(
            filepath_or_buffer,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
        )
    else:
        raise ValueError("unknown SAS format")

    if iterator or chunksize:
        return reader

    with reader:
        return reader.read()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\__init__.py ===
r"""
Array expressions are expressions representing N-dimensional arrays, without
evaluating them. These expressions represent in a certain way abstract syntax
trees of operations on N-dimensional arrays.

Every N-dimensional array operator has a corresponding array expression object.

Table of correspondences:

=============================== =============================
         Array operator           Array expression operator
=============================== =============================
        tensorproduct                 ArrayTensorProduct
        tensorcontraction             ArrayContraction
        tensordiagonal                ArrayDiagonal
        permutedims                   PermuteDims
=============================== =============================

Examples
========

``ArraySymbol`` objects are the N-dimensional equivalent of ``MatrixSymbol``
objects in the matrix module:

>>> from sympy.tensor.array.expressions import ArraySymbol
>>> from sympy.abc import i, j, k
>>> A = ArraySymbol("A", (3, 2, 4))
>>> A.shape
(3, 2, 4)
>>> A[i, j, k]
A[i, j, k]
>>> A.as_explicit()
[[[A[0, 0, 0], A[0, 0, 1], A[0, 0, 2], A[0, 0, 3]],
  [A[0, 1, 0], A[0, 1, 1], A[0, 1, 2], A[0, 1, 3]]],
 [[A[1, 0, 0], A[1, 0, 1], A[1, 0, 2], A[1, 0, 3]],
  [A[1, 1, 0], A[1, 1, 1], A[1, 1, 2], A[1, 1, 3]]],
 [[A[2, 0, 0], A[2, 0, 1], A[2, 0, 2], A[2, 0, 3]],
  [A[2, 1, 0], A[2, 1, 1], A[2, 1, 2], A[2, 1, 3]]]]

Component-explicit arrays can be added inside array expressions:

>>> from sympy import Array
>>> from sympy import tensorproduct
>>> from sympy.tensor.array.expressions import ArrayTensorProduct
>>> a = Array([1, 2, 3])
>>> b = Array([i, j, k])
>>> expr = ArrayTensorProduct(a, b, b)
>>> expr
ArrayTensorProduct([1, 2, 3], [i, j, k], [i, j, k])
>>> expr.as_explicit() == tensorproduct(a, b, b)
True

Constructing array expressions from index-explicit forms
--------------------------------------------------------

Array expressions are index-implicit. This means they do not use any indices to
represent array operations. The function ``convert_indexed_to_array( ... )``
may be used to convert index-explicit expressions to array expressions.
It takes as input two parameters: the index-explicit expression and the order
of the indices:

>>> from sympy.tensor.array.expressions import convert_indexed_to_array
>>> from sympy import Sum
>>> A = ArraySymbol("A", (3, 3))
>>> B = ArraySymbol("B", (3, 3))
>>> convert_indexed_to_array(A[i, j], [i, j])
A
>>> convert_indexed_to_array(A[i, j], [j, i])
PermuteDims(A, (0 1))
>>> convert_indexed_to_array(A[i, j] + B[j, i], [i, j])
ArrayAdd(A, PermuteDims(B, (0 1)))
>>> convert_indexed_to_array(Sum(A[i, j]*B[j, k], (j, 0, 2)), [i, k])
ArrayContraction(ArrayTensorProduct(A, B), (1, 2))

The diagonal of a matrix in the array expression form:

>>> convert_indexed_to_array(A[i, i], [i])
ArrayDiagonal(A, (0, 1))

The trace of a matrix in the array expression form:

>>> convert_indexed_to_array(Sum(A[i, i], (i, 0, 2)), [i])
ArrayContraction(A, (0, 1))

Compatibility with matrices
---------------------------

Array expressions can be mixed with objects from the matrix module:

>>> from sympy import MatrixSymbol
>>> from sympy.tensor.array.expressions import ArrayContraction
>>> M = MatrixSymbol("M", 3, 3)
>>> N = MatrixSymbol("N", 3, 3)

Express the matrix product in the array expression form:

>>> from sympy.tensor.array.expressions import convert_matrix_to_array
>>> expr = convert_matrix_to_array(M*N)
>>> expr
ArrayContraction(ArrayTensorProduct(M, N), (1, 2))

The expression can be converted back to matrix form:

>>> from sympy.tensor.array.expressions import convert_array_to_matrix
>>> convert_array_to_matrix(expr)
M*N

Add a second contraction on the remaining axes in order to get the trace of `M \cdot N`:

>>> expr_tr = ArrayContraction(expr, (0, 1))
>>> expr_tr
ArrayContraction(ArrayContraction(ArrayTensorProduct(M, N), (1, 2)), (0, 1))

Flatten the expression by calling ``.doit()`` and remove the nested array contraction operations:

>>> expr_tr.doit()
ArrayContraction(ArrayTensorProduct(M, N), (0, 3), (1, 2))

Get the explicit form of the array expression:

>>> expr.as_explicit()
[[M[0, 0]*N[0, 0] + M[0, 1]*N[1, 0] + M[0, 2]*N[2, 0], M[0, 0]*N[0, 1] + M[0, 1]*N[1, 1] + M[0, 2]*N[2, 1], M[0, 0]*N[0, 2] + M[0, 1]*N[1, 2] + M[0, 2]*N[2, 2]],
 [M[1, 0]*N[0, 0] + M[1, 1]*N[1, 0] + M[1, 2]*N[2, 0], M[1, 0]*N[0, 1] + M[1, 1]*N[1, 1] + M[1, 2]*N[2, 1], M[1, 0]*N[0, 2] + M[1, 1]*N[1, 2] + M[1, 2]*N[2, 2]],
 [M[2, 0]*N[0, 0] + M[2, 1]*N[1, 0] + M[2, 2]*N[2, 0], M[2, 0]*N[0, 1] + M[2, 1]*N[1, 1] + M[2, 2]*N[2, 1], M[2, 0]*N[0, 2] + M[2, 1]*N[1, 2] + M[2, 2]*N[2, 2]]]

Express the trace of a matrix:

>>> from sympy import Trace
>>> convert_matrix_to_array(Trace(M))
ArrayContraction(M, (0, 1))
>>> convert_matrix_to_array(Trace(M*N))
ArrayContraction(ArrayTensorProduct(M, N), (0, 3), (1, 2))

Express the transposition of a matrix (will be expressed as a permutation of the axes:

>>> convert_matrix_to_array(M.T)
PermuteDims(M, (0 1))

Compute the derivative array expressions:

>>> from sympy.tensor.array.expressions import array_derive
>>> d = array_derive(M, M)
>>> d
PermuteDims(ArrayTensorProduct(I, I), (3)(1 2))

Verify that the derivative corresponds to the form computed with explicit matrices:

>>> d.as_explicit()
[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]]
>>> Me = M.as_explicit()
>>> Me.diff(Me)
[[[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]]

"""

__all__ = [
    "ArraySymbol", "ArrayElement", "ZeroArray", "OneArray",
    "ArrayTensorProduct",
    "ArrayContraction",
    "ArrayDiagonal",
    "PermuteDims",
    "ArrayAdd",
    "ArrayElementwiseApplyFunc",
    "Reshape",
    "convert_array_to_matrix",
    "convert_matrix_to_array",
    "convert_array_to_indexed",
    "convert_indexed_to_array",
    "array_derive",
]

from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, PermuteDims, ArrayDiagonal, \
    ArrayContraction, Reshape, ArraySymbol, ArrayElement, ZeroArray, OneArray, ArrayElementwiseApplyFunc
from sympy.tensor.array.expressions.arrayexpr_derivatives import array_derive
from sympy.tensor.array.expressions.from_array_to_indexed import convert_array_to_indexed
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\pie_chart.py ===
#Autogenerated schema
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Bool,
    MinMax,
    Integer,
    NoneSet,
    Float,
    Alias,
    Sequence,
)
from openpyxl.descriptors.excel import ExtensionList, Percentage
from openpyxl.descriptors.nested import (
    NestedBool,
    NestedMinMax,
    NestedInteger,
    NestedFloat,
    NestedNoneSet,
    NestedSet,
)
from openpyxl.descriptors.sequence import ValueSequence

from ._chart import ChartBase
from .axis import ChartLines
from .descriptors import NestedGapAmount
from .series import Series
from .label import DataLabelList


class _PieChartBase(ChartBase):

    varyColors = NestedBool(allow_none=True)
    ser = Sequence(expected_type=Series, allow_none=True)
    dLbls = Typed(expected_type=DataLabelList, allow_none=True)
    dataLabels = Alias("dLbls")

    _series_type = "pie"

    __elements__ = ('varyColors', 'ser', 'dLbls')

    def __init__(self,
                 varyColors=True,
                 ser=(),
                 dLbls=None,
                ):
        self.varyColors = varyColors
        self.ser = ser
        self.dLbls = dLbls
        super().__init__()



class PieChart(_PieChartBase):

    tagname = "pieChart"

    varyColors = _PieChartBase.varyColors
    ser = _PieChartBase.ser
    dLbls = _PieChartBase.dLbls

    firstSliceAng = NestedMinMax(min=0, max=360)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = _PieChartBase.__elements__ + ('firstSliceAng', )

    def __init__(self,
                 firstSliceAng=0,
                 extLst=None,
                 **kw
                ):
        self.firstSliceAng = firstSliceAng
        super().__init__(**kw)


class PieChart3D(_PieChartBase):

    tagname = "pie3DChart"

    varyColors = _PieChartBase.varyColors
    ser = _PieChartBase.ser
    dLbls = _PieChartBase.dLbls

    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = _PieChartBase.__elements__


class DoughnutChart(_PieChartBase):

    tagname = "doughnutChart"

    varyColors = _PieChartBase.varyColors
    ser = _PieChartBase.ser
    dLbls = _PieChartBase.dLbls

    firstSliceAng = NestedMinMax(min=0, max=360)
    holeSize = NestedMinMax(min=1, max=90, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = _PieChartBase.__elements__ + ('firstSliceAng', 'holeSize')

    def __init__(self,
                 firstSliceAng=0,
                 holeSize=10,
                 extLst=None,
                 **kw
                ):
        self.firstSliceAng = firstSliceAng
        self.holeSize = holeSize
        super().__init__(**kw)


class CustomSplit(Serialisable):

    tagname = "custSplit"

    secondPiePt = ValueSequence(expected_type=int)

    __elements__ = ('secondPiePt',)

    def __init__(self,
                 secondPiePt=(),
                ):
        self.secondPiePt = secondPiePt


class ProjectedPieChart(_PieChartBase):

    """
    From the spec 21.2.2.126

    This element contains the pie of pie or bar of pie series on this
    chart. Only the first series shall be displayed. The splitType element
    shall determine whether the splitPos and custSplit elements apply.
    """

    tagname = "ofPieChart"

    varyColors = _PieChartBase.varyColors
    ser = _PieChartBase.ser
    dLbls = _PieChartBase.dLbls

    ofPieType = NestedSet(values=(['pie', 'bar']))
    type = Alias('ofPieType')
    gapWidth = NestedGapAmount()
    splitType = NestedNoneSet(values=(['auto', 'cust', 'percent', 'pos', 'val']))
    splitPos = NestedFloat(allow_none=True)
    custSplit = Typed(expected_type=CustomSplit, allow_none=True)
    secondPieSize = NestedMinMax(min=5, max=200, allow_none=True)
    serLines = Typed(expected_type=ChartLines, allow_none=True)
    join_lines = Alias('serLines')
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = _PieChartBase.__elements__ + ('ofPieType', 'gapWidth',
                                                 'splitType', 'splitPos', 'custSplit', 'secondPieSize', 'serLines')

    def __init__(self,
                 ofPieType="pie",
                 gapWidth=None,
                 splitType="auto",
                 splitPos=None,
                 custSplit=None,
                 secondPieSize=75,
                 serLines=None,
                 extLst=None,
                 **kw
                ):
        self.ofPieType = ofPieType
        self.gapWidth = gapWidth
        self.splitType = splitType
        self.splitPos = splitPos
        self.custSplit = custSplit
        self.secondPieSize = secondPieSize
        if serLines is None:
            self.serLines = ChartLines()
        super().__init__(**kw)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\graphic.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.xml.constants import CHART_NS, DRAWING_NS
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Bool,
    String,
    Alias,
)
from openpyxl.descriptors.excel import ExtensionList as OfficeArtExtensionList

from .effect import (
    EffectList,
    EffectContainer,
)
from .fill import (
    Blip,
    GradientFillProperties,
    BlipFillProperties,
)
from .picture import PictureFrame
from .properties import (
    NonVisualDrawingProps,
    NonVisualGroupShape,
    GroupShapeProperties,
)
from .relation import ChartRelation
from .xdr import XDRTransform2D


class GraphicFrameLocking(Serialisable):

    noGrp = Bool(allow_none=True)
    noDrilldown = Bool(allow_none=True)
    noSelect = Bool(allow_none=True)
    noChangeAspect = Bool(allow_none=True)
    noMove = Bool(allow_none=True)
    noResize = Bool(allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    def __init__(self,
                 noGrp=None,
                 noDrilldown=None,
                 noSelect=None,
                 noChangeAspect=None,
                 noMove=None,
                 noResize=None,
                 extLst=None,
                ):
        self.noGrp = noGrp
        self.noDrilldown = noDrilldown
        self.noSelect = noSelect
        self.noChangeAspect = noChangeAspect
        self.noMove = noMove
        self.noResize = noResize
        self.extLst = extLst


class NonVisualGraphicFrameProperties(Serialisable):

    tagname = "cNvGraphicFramePr"

    graphicFrameLocks = Typed(expected_type=GraphicFrameLocking, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    def __init__(self,
                 graphicFrameLocks=None,
                 extLst=None,
                ):
        self.graphicFrameLocks = graphicFrameLocks
        self.extLst = extLst


class NonVisualGraphicFrame(Serialisable):

    tagname = "nvGraphicFramePr"

    cNvPr = Typed(expected_type=NonVisualDrawingProps)
    cNvGraphicFramePr = Typed(expected_type=NonVisualGraphicFrameProperties)

    __elements__ = ('cNvPr', 'cNvGraphicFramePr')

    def __init__(self,
                 cNvPr=None,
                 cNvGraphicFramePr=None,
                ):
        if cNvPr is None:
            cNvPr = NonVisualDrawingProps(id=0, name="Chart 0")
        self.cNvPr = cNvPr
        if cNvGraphicFramePr is None:
            cNvGraphicFramePr = NonVisualGraphicFrameProperties()
        self.cNvGraphicFramePr = cNvGraphicFramePr


class GraphicData(Serialisable):

    tagname = "graphicData"
    namespace = DRAWING_NS

    uri = String()
    chart = Typed(expected_type=ChartRelation, allow_none=True)


    def __init__(self,
                 uri=CHART_NS,
                 chart=None,
                ):
        self.uri = uri
        self.chart = chart


class GraphicObject(Serialisable):

    tagname = "graphic"
    namespace = DRAWING_NS

    graphicData = Typed(expected_type=GraphicData)

    def __init__(self,
                 graphicData=None,
                ):
        if graphicData is None:
            graphicData = GraphicData()
        self.graphicData = graphicData


class GraphicFrame(Serialisable):

    tagname = "graphicFrame"

    nvGraphicFramePr = Typed(expected_type=NonVisualGraphicFrame)
    xfrm = Typed(expected_type=XDRTransform2D)
    graphic = Typed(expected_type=GraphicObject)
    macro = String(allow_none=True)
    fPublished = Bool(allow_none=True)

    __elements__ = ('nvGraphicFramePr', 'xfrm', 'graphic', 'macro', 'fPublished')

    def __init__(self,
                 nvGraphicFramePr=None,
                 xfrm=None,
                 graphic=None,
                 macro=None,
                 fPublished=None,
                 ):
        if nvGraphicFramePr is None:
            nvGraphicFramePr = NonVisualGraphicFrame()
        self.nvGraphicFramePr = nvGraphicFramePr
        if xfrm is None:
            xfrm = XDRTransform2D()
        self.xfrm = xfrm
        if graphic is None:
            graphic = GraphicObject()
        self.graphic = graphic
        self.macro = macro
        self.fPublished = fPublished


class GroupShape(Serialisable):

    nvGrpSpPr = Typed(expected_type=NonVisualGroupShape)
    nonVisualProperties = Alias("nvGrpSpPr")
    grpSpPr = Typed(expected_type=GroupShapeProperties)
    visualProperties = Alias("grpSpPr")
    pic = Typed(expected_type=PictureFrame, allow_none=True)

    __elements__ = ["nvGrpSpPr", "grpSpPr", "pic"]

    def __init__(self,
                 nvGrpSpPr=None,
                 grpSpPr=None,
                 pic=None,
                ):
        self.nvGrpSpPr = nvGrpSpPr
        self.grpSpPr = grpSpPr
        self.pic = pic

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\__init__.py ===
"""Rich text and beautiful formatting in the terminal."""

import os
from typing import IO, TYPE_CHECKING, Any, Callable, Optional, Union

from ._extension import load_ipython_extension  # noqa: F401

__all__ = ["get_console", "reconfigure", "print", "inspect", "print_json"]

if TYPE_CHECKING:
    from .console import Console

# Global console used by alternative print
_console: Optional["Console"] = None

try:
    _IMPORT_CWD = os.path.abspath(os.getcwd())
except FileNotFoundError:
    # Can happen if the cwd has been deleted
    _IMPORT_CWD = ""


def get_console() -> "Console":
    """Get a global :class:`~rich.console.Console` instance. This function is used when Rich requires a Console,
    and hasn't been explicitly given one.

    Returns:
        Console: A console instance.
    """
    global _console
    if _console is None:
        from .console import Console

        _console = Console()

    return _console


def reconfigure(*args: Any, **kwargs: Any) -> None:
    """Reconfigures the global console by replacing it with another.

    Args:
        *args (Any): Positional arguments for the replacement :class:`~rich.console.Console`.
        **kwargs (Any): Keyword arguments for the replacement :class:`~rich.console.Console`.
    """
    from pip._vendor.rich.console import Console

    new_console = Console(*args, **kwargs)
    _console = get_console()
    _console.__dict__ = new_console.__dict__


def print(
    *objects: Any,
    sep: str = " ",
    end: str = "\n",
    file: Optional[IO[str]] = None,
    flush: bool = False,
) -> None:
    r"""Print object(s) supplied via positional arguments.
    This function has an identical signature to the built-in print.
    For more advanced features, see the :class:`~rich.console.Console` class.

    Args:
        sep (str, optional): Separator between printed objects. Defaults to " ".
        end (str, optional): Character to write at end of output. Defaults to "\\n".
        file (IO[str], optional): File to write to, or None for stdout. Defaults to None.
        flush (bool, optional): Has no effect as Rich always flushes output. Defaults to False.

    """
    from .console import Console

    write_console = get_console() if file is None else Console(file=file)
    return write_console.print(*objects, sep=sep, end=end)


def print_json(
    json: Optional[str] = None,
    *,
    data: Any = None,
    indent: Union[None, int, str] = 2,
    highlight: bool = True,
    skip_keys: bool = False,
    ensure_ascii: bool = False,
    check_circular: bool = True,
    allow_nan: bool = True,
    default: Optional[Callable[[Any], Any]] = None,
    sort_keys: bool = False,
) -> None:
    """Pretty prints JSON. Output will be valid JSON.

    Args:
        json (str): A string containing JSON.
        data (Any): If json is not supplied, then encode this data.
        indent (int, optional): Number of spaces to indent. Defaults to 2.
        highlight (bool, optional): Enable highlighting of output: Defaults to True.
        skip_keys (bool, optional): Skip keys not of a basic type. Defaults to False.
        ensure_ascii (bool, optional): Escape all non-ascii characters. Defaults to False.
        check_circular (bool, optional): Check for circular references. Defaults to True.
        allow_nan (bool, optional): Allow NaN and Infinity values. Defaults to True.
        default (Callable, optional): A callable that converts values that can not be encoded
            in to something that can be JSON encoded. Defaults to None.
        sort_keys (bool, optional): Sort dictionary keys. Defaults to False.
    """

    get_console().print_json(
        json,
        data=data,
        indent=indent,
        highlight=highlight,
        skip_keys=skip_keys,
        ensure_ascii=ensure_ascii,
        check_circular=check_circular,
        allow_nan=allow_nan,
        default=default,
        sort_keys=sort_keys,
    )


def inspect(
    obj: Any,
    *,
    console: Optional["Console"] = None,
    title: Optional[str] = None,
    help: bool = False,
    methods: bool = False,
    docs: bool = True,
    private: bool = False,
    dunder: bool = False,
    sort: bool = True,
    all: bool = False,
    value: bool = True,
) -> None:
    """Inspect any Python object.

    * inspect(<OBJECT>) to see summarized info.
    * inspect(<OBJECT>, methods=True) to see methods.
    * inspect(<OBJECT>, help=True) to see full (non-abbreviated) help.
    * inspect(<OBJECT>, private=True) to see private attributes (single underscore).
    * inspect(<OBJECT>, dunder=True) to see attributes beginning with double underscore.
    * inspect(<OBJECT>, all=True) to see all attributes.

    Args:
        obj (Any): An object to inspect.
        title (str, optional): Title to display over inspect result, or None use type. Defaults to None.
        help (bool, optional): Show full help text rather than just first paragraph. Defaults to False.
        methods (bool, optional): Enable inspection of callables. Defaults to False.
        docs (bool, optional): Also render doc strings. Defaults to True.
        private (bool, optional): Show private attributes (beginning with underscore). Defaults to False.
        dunder (bool, optional): Show attributes starting with double underscore. Defaults to False.
        sort (bool, optional): Sort attributes alphabetically. Defaults to True.
        all (bool, optional): Show all attributes. Defaults to False.
        value (bool, optional): Pretty print value. Defaults to True.
    """
    _console = console or get_console()
    from pip._vendor.rich._inspect import Inspect

    # Special case for inspect(inspect)
    is_inspect = obj is inspect

    _inspect = Inspect(
        obj,
        title=title,
        help=is_inspect or help,
        methods=is_inspect or methods,
        docs=is_inspect or docs,
        private=private,
        dunder=dunder,
        sort=sort,
        all=all,
        value=value,
    )
    _console.print(_inspect)


if __name__ == "__main__":  # pragma: no cover
    print("Hello, **World**")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\server.py ===
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

import collections
import os
import re
import shutil
import socket
import subprocess
import time
import urllib

from selenium.webdriver.common.selenium_manager import SeleniumManager


class Server:
    """Manage a Selenium Grid (Remote) Server in standalone mode.

    This class contains functionality for downloading the server and starting/stopping it.

    For more information on Selenium Grid, see:
        - https://www.selenium.dev/documentation/grid/getting_started/

    Parameters:
    -----------
    host : str
        Hostname or IP address to bind to (determined automatically if not specified)
    port : int or str
        Port to listen on (4444 if not specified)
    path : str
        Path/filename of existing server .jar file (Selenium Manager is used if not specified)
    version : str
        Version of server to download (latest version if not specified)
    log_level : str
        Logging level to control logging output ("INFO" if not specified)
        Available levels: "SEVERE", "WARNING", "INFO", "CONFIG", "FINE", "FINER", "FINEST"
    env: collections.abc.Mapping
        Mapping that defines the environment variables for the server process
    """

    def __init__(self, host=None, port=4444, path=None, version=None, log_level="INFO", env=None):
        if path and version:
            raise TypeError("Not allowed to specify a version when using an existing server path")

        self.host = host
        self.port = self._validate_port(port)
        self.path = self._validate_path(path)
        self.version = self._validate_version(version)
        self.log_level = self._validate_log_level(log_level)
        self.env = self._validate_env(env)

        self.process = None
        self.status_url = self._get_status_url()

    def _get_status_url(self):
        host = self.host if self.host is not None else "localhost"
        return f"http://{host}:{self.port}/status"

    def _validate_path(self, path):
        if path and not os.path.exists(path):
            raise OSError(f"Can't find server .jar located at {path}")
        return path

    def _validate_port(self, port):
        try:
            port = int(port)
        except ValueError:
            raise TypeError(f"{__class__.__name__}.__init__() got an invalid port: '{port}'")
        if not (0 <= port <= 65535):
            raise ValueError("port must be 0-65535")
        return port

    def _validate_version(self, version):
        if version:
            if not re.match(r"^\d+\.\d+\.\d+$", str(version)):
                raise TypeError(f"{__class__.__name__}.__init__() got an invalid version: '{version}'")
        return version

    def _validate_log_level(self, log_level):
        levels = ("SEVERE", "WARNING", "INFO", "CONFIG", "FINE", "FINER", "FINEST")
        if log_level not in levels:
            raise TypeError(f"log_level must be one of: {', '.join(levels)}")
        return log_level

    def _validate_env(self, env):
        if env is not None and not isinstance(env, collections.abc.Mapping):
            raise TypeError("env must be a mapping of environment variables")
        return env

    def _wait_for_server(self, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(self.status_url)
                return True
            except urllib.error.URLError:
                time.sleep(0.2)
        return False

    def download_if_needed(self, version=None):
        """Download the server if it doesn't already exist.

        Latest version is downloaded unless specified.
        """
        args = ["--grid"]
        if version is not None:
            args.append(version)
        return SeleniumManager().binary_paths(args)["driver_path"]

    def start(self):
        """Start the server.

        Selenium Manager will detect the server location and download it if necessary,
        unless an existing server path was specified.
        """
        path = self.download_if_needed(self.version) if self.path is None else self.path

        java_path = shutil.which("java")
        if java_path is None:
            raise OSError("Can't find java on system PATH. JRE is required to run the Selenium server")

        command = [
            java_path,
            "-jar",
            path,
            "standalone",
            "--port",
            str(self.port),
            "--log-level",
            self.log_level,
            "--selenium-manager",
            "true",
            "--enable-managed-downloads",
            "true",
        ]
        if self.host is not None:
            command.extend(["--host", self.host])

        host = self.host if self.host is not None else "localhost"

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((host, self.port))
            raise ConnectionError(f"Selenium server is already running, or something else is using port {self.port}")
        except ConnectionRefusedError:
            print("Starting Selenium server...")
            self.process = subprocess.Popen(command, env=self.env)
            print(f"Selenium server running as process: {self.process.pid}")
            if not self._wait_for_server():
                raise TimeoutError(f"Timed out waiting for Selenium server at {self.status_url}")
            print("Selenium server is ready")
        return self.process

    def stop(self):
        """Stop the server."""
        if self.process is None:
            raise RuntimeError("Selenium server isn't running")
        else:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait()
            self.process = None
            print("Selenium server has been terminated")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\numpy_nodes.py ===
from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from .ast import Token, none


def _logaddexp(x1, x2, *, evaluate=True):
    return log(Add(exp(x1, evaluate=evaluate), exp(x2, evaluate=evaluate), evaluate=evaluate))


_two = S.One*2
_ln2 = log(_two)


def _lb(x, *, evaluate=True):
    return log(x, evaluate=evaluate)/_ln2


def _exp2(x, *, evaluate=True):
    return Pow(_two, x, evaluate=evaluate)


def _logaddexp2(x1, x2, *, evaluate=True):
    return _lb(Add(_exp2(x1, evaluate=evaluate),
                   _exp2(x2, evaluate=evaluate), evaluate=evaluate))


class logaddexp(Function):
    """ Logarithm of the sum of exponentiations of the inputs.

    Helper class for use with e.g. numpy.logaddexp

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    """
    nargs = 2

    def __new__(cls, *args):
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            wrt, other = self.args
        elif argindex == 2:
            other, wrt = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One/(S.One + exp(other-wrt))

    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        return _logaddexp(x1, x2)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_simplify(self, *args, **kwargs):
        a, b = (x.simplify(**kwargs) for x in self.args)
        candidate = _logaddexp(a, b)
        if candidate != _logaddexp(a, b, evaluate=False):
            return candidate
        else:
            return logaddexp(a, b)


class logaddexp2(Function):
    """ Logarithm of the sum of exponentiations of the inputs in base-2.

    Helper class for use with e.g. numpy.logaddexp2

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html
    """
    nargs = 2

    def __new__(cls, *args):
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            wrt, other = self.args
        elif argindex == 2:
            other, wrt = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One/(S.One + _exp2(other-wrt))

    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        return _logaddexp2(x1, x2)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_simplify(self, *args, **kwargs):
        a, b = (x.simplify(**kwargs).factor() for x in self.args)
        candidate = _logaddexp2(a, b)
        if candidate != _logaddexp2(a, b, evaluate=False):
            return candidate
        else:
            return logaddexp2(a, b)


class amin(Token):
    """ Minimum value along an axis.

    Helper class for use with e.g. numpy.amin


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.amin.html
    """
    __slots__ = _fields = ('array', 'axis')
    defaults = {'axis': none}
    _construct_axis = staticmethod(sympify)


class amax(Token):
    """ Maximum value along an axis.

    Helper class for use with e.g. numpy.amax


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.amax.html
    """
    __slots__ = _fields = ('array', 'axis')
    defaults = {'axis': none}
    _construct_axis = staticmethod(sympify)


class maximum(Function):
    """ Element-wise maximum of array elements.

    Helper class for use with e.g. numpy.maximum


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    """

    def _eval_rewrite_as_Max(self, *args):
        return Max(*self.args)


class minimum(Function):
    """ Element-wise minimum of array elements.

    Helper class for use with e.g. numpy.minimum


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
    """

    def _eval_rewrite_as_Min(self, *args):
        return Min(*self.args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\loads.py ===
from abc import ABC
from collections import namedtuple
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.vector import Vector, ReferenceFrame, Point

__all__ = ['LoadBase', 'Force', 'Torque']


class LoadBase(ABC, namedtuple('LoadBase', ['location', 'vector'])):
    """Abstract base class for the various loading types."""

    def __add__(self, other):
        raise TypeError(f"unsupported operand type(s) for +: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    def __mul__(self, other):
        raise TypeError(f"unsupported operand type(s) for *: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    __radd__ = __add__
    __rmul__ = __mul__


class Force(LoadBase):
    """Force acting upon a point.

    Explanation
    ===========

    A force is a vector that is bound to a line of action. This class stores
    both a point, which lies on the line of action, and the vector. A tuple can
    also be used, with the location as the first entry and the vector as second
    entry.

    Examples
    ========

    A force of magnitude 2 along N.x acting on a point Po can be created as
    follows:

    >>> from sympy.physics.mechanics import Point, ReferenceFrame, Force
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Force(Po, 2 * N.x)
    (Po, 2*N.x)

    If a body is supplied, then the center of mass of that body is used.

    >>> from sympy.physics.mechanics import Particle
    >>> P = Particle('P', point=Po)
    >>> Force(P, 2 * N.x)
    (Po, 2*N.x)

    """

    def __new__(cls, point, force):
        if isinstance(point, BodyBase):
            point = point.masscenter
        if not isinstance(point, Point):
            raise TypeError('Force location should be a Point.')
        if not isinstance(force, Vector):
            raise TypeError('Force vector should be a Vector.')
        return super().__new__(cls, point, force)

    def __repr__(self):
        return (f'{self.__class__.__name__}(point={self.point}, '
                f'force={self.force})')

    @property
    def point(self):
        return self.location

    @property
    def force(self):
        return self.vector


class Torque(LoadBase):
    """Torque acting upon a frame.

    Explanation
    ===========

    A torque is a free vector that is acting on a reference frame, which is
    associated with a rigid body. This class stores both the frame and the
    vector. A tuple can also be used, with the location as the first item and
    the vector as second item.

    Examples
    ========

    A torque of magnitude 2 about N.x acting on a frame N can be created as
    follows:

    >>> from sympy.physics.mechanics import ReferenceFrame, Torque
    >>> N = ReferenceFrame('N')
    >>> Torque(N, 2 * N.x)
    (N, 2*N.x)

    If a body is supplied, then the frame fixed to that body is used.

    >>> from sympy.physics.mechanics import RigidBody
    >>> rb = RigidBody('rb', frame=N)
    >>> Torque(rb, 2 * N.x)
    (N, 2*N.x)

    """

    def __new__(cls, frame, torque):
        if isinstance(frame, BodyBase):
            frame = frame.frame
        if not isinstance(frame, ReferenceFrame):
            raise TypeError('Torque location should be a ReferenceFrame.')
        if not isinstance(torque, Vector):
            raise TypeError('Torque vector should be a Vector.')
        return super().__new__(cls, frame, torque)

    def __repr__(self):
        return (f'{self.__class__.__name__}(frame={self.frame}, '
                f'torque={self.torque})')

    @property
    def frame(self):
        return self.location

    @property
    def torque(self):
        return self.vector


def gravity(acceleration, *bodies):
    """
    Returns a list of gravity forces given the acceleration
    due to gravity and any number of particles or rigidbodies.

    Example
    =======

    >>> from sympy.physics.mechanics import ReferenceFrame, Particle, RigidBody
    >>> from sympy.physics.mechanics.loads import gravity
    >>> from sympy import symbols
    >>> N = ReferenceFrame('N')
    >>> g = symbols('g')
    >>> P = Particle('P')
    >>> B = RigidBody('B')
    >>> gravity(g*N.y, P, B)
    [(P_masscenter, P_mass*g*N.y),
     (B_masscenter, B_mass*g*N.y)]

    """

    gravity_force = []
    for body in bodies:
        if not isinstance(body, BodyBase):
            raise TypeError(f'{type(body)} is not a body type')
        gravity_force.append(Force(body.masscenter, body.mass * acceleration))
    return gravity_force


def _parse_load(load):
    """Helper function to parse loads and convert tuples to load objects."""
    if isinstance(load, LoadBase):
        return load
    elif isinstance(load, tuple):
        if len(load) != 2:
            raise ValueError(f'Load {load} should have a length of 2.')
        if isinstance(load[0], Point):
            return Force(load[0], load[1])
        elif isinstance(load[0], ReferenceFrame):
            return Torque(load[0], load[1])
        else:
            raise ValueError(f'Load not recognized. The load location {load[0]}'
                             f' should either be a Point or a ReferenceFrame.')
    raise TypeError(f'Load type {type(load)} not recognized as a load. It '
                    f'should be a Force, Torque or tuple.')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\ops\missing.py ===
"""
Missing data handling for arithmetic operations.

In particular, pandas conventions regarding division by zero differ
from numpy in the following ways:
    1) np.array([-1, 0, 1], dtype=dtype1) // np.array([0, 0, 0], dtype=dtype2)
       gives [nan, nan, nan] for most dtype combinations, and [0, 0, 0] for
       the remaining pairs
       (the remaining being dtype1==dtype2==intN and dtype==dtype2==uintN).

       pandas convention is to return [-inf, nan, inf] for all dtype
       combinations.

       Note: the numpy behavior described here is py3-specific.

    2) np.array([-1, 0, 1], dtype=dtype1) % np.array([0, 0, 0], dtype=dtype2)
       gives precisely the same results as the // operation.

       pandas convention is to return [nan, nan, nan] for all dtype
       combinations.

    3) divmod behavior consistent with 1) and 2).
"""
from __future__ import annotations

import operator

import numpy as np

from pandas.core import roperator


def _fill_zeros(result: np.ndarray, x, y):
    """
    If this is a reversed op, then flip x,y

    If we have an integer value (or array in y)
    and we have 0's, fill them with np.nan,
    return the result.

    Mask the nan's from x.
    """
    if result.dtype.kind == "f":
        return result

    is_variable_type = hasattr(y, "dtype")
    is_scalar_type = not isinstance(y, np.ndarray)

    if not is_variable_type and not is_scalar_type:
        # e.g. test_series_ops_name_retention with mod we get here with list/tuple
        return result

    if is_scalar_type:
        y = np.array(y)

    if y.dtype.kind in "iu":
        ymask = y == 0
        if ymask.any():
            # GH#7325, mask and nans must be broadcastable
            mask = ymask & ~np.isnan(result)

            # GH#9308 doing ravel on result and mask can improve putmask perf,
            #  but can also make unwanted copies.
            result = result.astype("float64", copy=False)

            np.putmask(result, mask, np.nan)

    return result


def mask_zero_div_zero(x, y, result: np.ndarray) -> np.ndarray:
    """
    Set results of  0 // 0 to np.nan, regardless of the dtypes
    of the numerator or the denominator.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    result : ndarray

    Returns
    -------
    ndarray
        The filled result.

    Examples
    --------
    >>> x = np.array([1, 0, -1], dtype=np.int64)
    >>> x
    array([ 1,  0, -1])
    >>> y = 0       # int 0; numpy behavior is different with float
    >>> result = x // y
    >>> result      # raw numpy result does not fill division by zero
    array([0, 0, 0])
    >>> mask_zero_div_zero(x, y, result)
    array([ inf,  nan, -inf])
    """

    if not hasattr(y, "dtype"):
        # e.g. scalar, tuple
        y = np.array(y)
    if not hasattr(x, "dtype"):
        # e.g scalar, tuple
        x = np.array(x)

    zmask = y == 0

    if zmask.any():
        # Flip sign if necessary for -0.0
        zneg_mask = zmask & np.signbit(y)
        zpos_mask = zmask & ~zneg_mask

        x_lt0 = x < 0
        x_gt0 = x > 0
        nan_mask = zmask & (x == 0)
        neginf_mask = (zpos_mask & x_lt0) | (zneg_mask & x_gt0)
        posinf_mask = (zpos_mask & x_gt0) | (zneg_mask & x_lt0)

        if nan_mask.any() or neginf_mask.any() or posinf_mask.any():
            # Fill negative/0 with -inf, positive/0 with +inf, 0/0 with NaN
            result = result.astype("float64", copy=False)

            result[nan_mask] = np.nan
            result[posinf_mask] = np.inf
            result[neginf_mask] = -np.inf

    return result


def dispatch_fill_zeros(op, left, right, result):
    """
    Call _fill_zeros with the appropriate fill value depending on the operation,
    with special logic for divmod and rdivmod.

    Parameters
    ----------
    op : function (operator.add, operator.div, ...)
    left : object (np.ndarray for non-reversed ops)
        We have excluded ExtensionArrays here
    right : object (np.ndarray for reversed ops)
        We have excluded ExtensionArrays here
    result : ndarray

    Returns
    -------
    result : np.ndarray

    Notes
    -----
    For divmod and rdivmod, the `result` parameter and returned `result`
    is a 2-tuple of ndarray objects.
    """
    if op is divmod:
        result = (
            mask_zero_div_zero(left, right, result[0]),
            _fill_zeros(result[1], left, right),
        )
    elif op is roperator.rdivmod:
        result = (
            mask_zero_div_zero(right, left, result[0]),
            _fill_zeros(result[1], right, left),
        )
    elif op is operator.floordiv:
        # Note: no need to do this for truediv; in py3 numpy behaves the way
        #  we want.
        result = mask_zero_div_zero(left, right, result)
    elif op is roperator.rfloordiv:
        # Note: no need to do this for rtruediv; in py3 numpy behaves the way
        #  we want.
        result = mask_zero_div_zero(right, left, result)
    elif op is operator.mod:
        result = _fill_zeros(result, left, right)
    elif op is roperator.rmod:
        result = _fill_zeros(result, right, left)
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\search.py ===
import logging
import shutil
import sys
import textwrap
import xmlrpc.client
from collections import OrderedDict
from optparse import Values
from typing import Dict, List, Optional, TypedDict

from pip._vendor.packaging.version import parse as parse_version

from pip._internal.cli.base_command import Command
from pip._internal.cli.req_command import SessionCommandMixin
from pip._internal.cli.status_codes import NO_MATCHES_FOUND, SUCCESS
from pip._internal.exceptions import CommandError
from pip._internal.metadata import get_default_environment
from pip._internal.metadata.base import BaseDistribution
from pip._internal.models.index import PyPI
from pip._internal.network.xmlrpc import PipXmlrpcTransport
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import write_output


class TransformedHit(TypedDict):
    name: str
    summary: str
    versions: List[str]


logger = logging.getLogger(__name__)


class SearchCommand(Command, SessionCommandMixin):
    """Search for PyPI packages whose name or summary contains <query>."""

    usage = """
      %prog [options] <query>"""
    ignore_require_venv = True

    def add_options(self) -> None:
        self.cmd_opts.add_option(
            "-i",
            "--index",
            dest="index",
            metavar="URL",
            default=PyPI.pypi_url,
            help="Base URL of Python Package Index (default %default)",
        )

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options: Values, args: List[str]) -> int:
        if not args:
            raise CommandError("Missing required argument (search query).")
        query = args
        pypi_hits = self.search(query, options)
        hits = transform_hits(pypi_hits)

        terminal_width = None
        if sys.stdout.isatty():
            terminal_width = shutil.get_terminal_size()[0]

        print_results(hits, terminal_width=terminal_width)
        if pypi_hits:
            return SUCCESS
        return NO_MATCHES_FOUND

    def search(self, query: List[str], options: Values) -> List[Dict[str, str]]:
        index_url = options.index

        session = self.get_default_session(options)

        transport = PipXmlrpcTransport(index_url, session)
        pypi = xmlrpc.client.ServerProxy(index_url, transport)
        try:
            hits = pypi.search({"name": query, "summary": query}, "or")
        except xmlrpc.client.Fault as fault:
            message = (
                f"XMLRPC request failed [code: {fault.faultCode}]\n{fault.faultString}"
            )
            raise CommandError(message)
        assert isinstance(hits, list)
        return hits


def transform_hits(hits: List[Dict[str, str]]) -> List["TransformedHit"]:
    """
    The list from pypi is really a list of versions. We want a list of
    packages with the list of versions stored inline. This converts the
    list from pypi into one we can use.
    """
    packages: Dict[str, TransformedHit] = OrderedDict()
    for hit in hits:
        name = hit["name"]
        summary = hit["summary"]
        version = hit["version"]

        if name not in packages.keys():
            packages[name] = {
                "name": name,
                "summary": summary,
                "versions": [version],
            }
        else:
            packages[name]["versions"].append(version)

            # if this is the highest version, replace summary and score
            if version == highest_version(packages[name]["versions"]):
                packages[name]["summary"] = summary

    return list(packages.values())


def print_dist_installation_info(latest: str, dist: Optional[BaseDistribution]) -> None:
    if dist is not None:
        with indent_log():
            if dist.version == latest:
                write_output("INSTALLED: %s (latest)", dist.version)
            else:
                write_output("INSTALLED: %s", dist.version)
                if parse_version(latest).pre:
                    write_output(
                        "LATEST:    %s (pre-release; install"
                        " with `pip install --pre`)",
                        latest,
                    )
                else:
                    write_output("LATEST:    %s", latest)


def get_installed_distribution(name: str) -> Optional[BaseDistribution]:
    env = get_default_environment()
    return env.get_distribution(name)


def print_results(
    hits: List["TransformedHit"],
    name_column_width: Optional[int] = None,
    terminal_width: Optional[int] = None,
) -> None:
    if not hits:
        return
    if name_column_width is None:
        name_column_width = (
            max(
                [
                    len(hit["name"]) + len(highest_version(hit.get("versions", ["-"])))
                    for hit in hits
                ]
            )
            + 4
        )

    for hit in hits:
        name = hit["name"]
        summary = hit["summary"] or ""
        latest = highest_version(hit.get("versions", ["-"]))
        if terminal_width is not None:
            target_width = terminal_width - name_column_width - 5
            if target_width > 10:
                # wrap and indent summary to fit terminal
                summary_lines = textwrap.wrap(summary, target_width)
                summary = ("\n" + " " * (name_column_width + 3)).join(summary_lines)

        name_latest = f"{name} ({latest})"
        line = f"{name_latest:{name_column_width}} - {summary}"
        try:
            write_output(line)
            dist = get_installed_distribution(name)
            print_dist_installation_info(latest, dist)
        except UnicodeEncodeError:
            pass


def highest_version(versions: List[str]) -> str:
    return max(versions, key=parse_version)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\matrices.py ===
"""Known matrices related to physics"""

from sympy.core.numbers import I
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.utilities.decorator import deprecated


def msigma(i):
    r"""Returns a Pauli matrix `\sigma_i` with `i=1,2,3`.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pauli_matrices

    Examples
    ========

    >>> from sympy.physics.matrices import msigma
    >>> msigma(1)
    Matrix([
    [0, 1],
    [1, 0]])
    """
    if i == 1:
        mat = (
            (0, 1),
            (1, 0)
        )
    elif i == 2:
        mat = (
            (0, -I),
            (I, 0)
        )
    elif i == 3:
        mat = (
            (1, 0),
            (0, -1)
        )
    else:
        raise IndexError("Invalid Pauli index")
    return Matrix(mat)


def pat_matrix(m, dx, dy, dz):
    """Returns the Parallel Axis Theorem matrix to translate the inertia
    matrix a distance of `(dx, dy, dz)` for a body of mass m.

    Examples
    ========

    To translate a body having a mass of 2 units a distance of 1 unit along
    the `x`-axis we get:

    >>> from sympy.physics.matrices import pat_matrix
    >>> pat_matrix(2, 1, 0, 0)
    Matrix([
    [0, 0, 0],
    [0, 2, 0],
    [0, 0, 2]])

    """
    dxdy = -dx*dy
    dydz = -dy*dz
    dzdx = -dz*dx
    dxdx = dx**2
    dydy = dy**2
    dzdz = dz**2
    mat = ((dydy + dzdz, dxdy, dzdx),
           (dxdy, dxdx + dzdz, dydz),
           (dzdx, dydz, dydy + dxdx))
    return m*Matrix(mat)


def mgamma(mu, lower=False):
    r"""Returns a Dirac gamma matrix `\gamma^\mu` in the standard
    (Dirac) representation.

    Explanation
    ===========

    If you want `\gamma_\mu`, use ``gamma(mu, True)``.

    We use a convention:

    `\gamma^5 = i \cdot \gamma^0 \cdot \gamma^1 \cdot \gamma^2 \cdot \gamma^3`

    `\gamma_5 = i \cdot \gamma_0 \cdot \gamma_1 \cdot \gamma_2 \cdot \gamma_3 = - \gamma^5`

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_matrices

    Examples
    ========

    >>> from sympy.physics.matrices import mgamma
    >>> mgamma(1)
    Matrix([
    [ 0,  0, 0, 1],
    [ 0,  0, 1, 0],
    [ 0, -1, 0, 0],
    [-1,  0, 0, 0]])
    """
    if mu not in (0, 1, 2, 3, 5):
        raise IndexError("Invalid Dirac index")
    if mu == 0:
        mat = (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 0, -1)
        )
    elif mu == 1:
        mat = (
            (0, 0, 0, 1),
            (0, 0, 1, 0),
            (0, -1, 0, 0),
            (-1, 0, 0, 0)
        )
    elif mu == 2:
        mat = (
            (0, 0, 0, -I),
            (0, 0, I, 0),
            (0, I, 0, 0),
            (-I, 0, 0, 0)
        )
    elif mu == 3:
        mat = (
            (0, 0, 1, 0),
            (0, 0, 0, -1),
            (-1, 0, 0, 0),
            (0, 1, 0, 0)
        )
    elif mu == 5:
        mat = (
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 0, 0, 0),
            (0, 1, 0, 0)
        )
    m = Matrix(mat)
    if lower:
        if mu in (1, 2, 3, 5):
            m = -m
    return m

#Minkowski tensor using the convention (+,-,-,-) used in the Quantum Field
#Theory
minkowski_tensor = Matrix( (
    (1, 0, 0, 0),
    (0, -1, 0, 0),
    (0, 0, -1, 0),
    (0, 0, 0, -1)
))


@deprecated(
    """
    The sympy.physics.matrices.mdft method is deprecated. Use
    sympy.DFT(n).as_explicit() instead.
    """,
    deprecated_since_version="1.9",
    active_deprecations_target="deprecated-physics-mdft",
)
def mdft(n):
    r"""
    .. deprecated:: 1.9

       Use DFT from sympy.matrices.expressions.fourier instead.

       To get identical behavior to ``mdft(n)``, use ``DFT(n).as_explicit()``.
    """
    from sympy.matrices.expressions.fourier import DFT
    return DFT(n).as_mutable()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\rl.py ===
""" Generic Rules for SymPy

This file assumes knowledge of Basic and little else.
"""
from sympy.utilities.iterables import sift
from .util import new


# Functions that create rules
def rm_id(isid, new=new):
    """ Create a rule to remove identities.

    isid - fn :: x -> Bool  --- whether or not this element is an identity.

    Examples
    ========

    >>> from sympy.strategies import rm_id
    >>> from sympy import Basic, S
    >>> remove_zeros = rm_id(lambda x: x==0)
    >>> remove_zeros(Basic(S(1), S(0), S(2)))
    Basic(1, 2)
    >>> remove_zeros(Basic(S(0), S(0))) # If only identities then we keep one
    Basic(0)

    See Also:
        unpack
    """
    def ident_remove(expr):
        """ Remove identities """
        ids = list(map(isid, expr.args))
        if sum(ids) == 0:           # No identities. Common case
            return expr
        elif sum(ids) != len(ids):  # there is at least one non-identity
            return new(expr.__class__,
                       *[arg for arg, x in zip(expr.args, ids) if not x])
        else:
            return new(expr.__class__, expr.args[0])

    return ident_remove


def glom(key, count, combine):
    """ Create a rule to conglomerate identical args.

    Examples
    ========

    >>> from sympy.strategies import glom
    >>> from sympy import Add
    >>> from sympy.abc import x

    >>> key     = lambda x: x.as_coeff_Mul()[1]
    >>> count   = lambda x: x.as_coeff_Mul()[0]
    >>> combine = lambda cnt, arg: cnt * arg
    >>> rl = glom(key, count, combine)

    >>> rl(Add(x, -x, 3*x, 2, 3, evaluate=False))
    3*x + 5

    Wait, how are key, count and combine supposed to work?

    >>> key(2*x)
    x
    >>> count(2*x)
    2
    >>> combine(2, x)
    2*x
    """
    def conglomerate(expr):
        """ Conglomerate together identical args x + x -> 2x """
        groups = sift(expr.args, key)
        counts = {k: sum(map(count, args)) for k, args in groups.items()}
        newargs = [combine(cnt, mat) for mat, cnt in counts.items()]
        if set(newargs) != set(expr.args):
            return new(type(expr), *newargs)
        else:
            return expr

    return conglomerate


def sort(key, new=new):
    """ Create a rule to sort by a key function.

    Examples
    ========

    >>> from sympy.strategies import sort
    >>> from sympy import Basic, S
    >>> sort_rl = sort(str)
    >>> sort_rl(Basic(S(3), S(1), S(2)))
    Basic(1, 2, 3)
    """

    def sort_rl(expr):
        return new(expr.__class__, *sorted(expr.args, key=key))
    return sort_rl


def distribute(A, B):
    """ Turns an A containing Bs into a B of As

    where A, B are container types

    >>> from sympy.strategies import distribute
    >>> from sympy import Add, Mul, symbols
    >>> x, y = symbols('x,y')
    >>> dist = distribute(Mul, Add)
    >>> expr = Mul(2, x+y, evaluate=False)
    >>> expr
    2*(x + y)
    >>> dist(expr)
    2*x + 2*y
    """

    def distribute_rl(expr):
        for i, arg in enumerate(expr.args):
            if isinstance(arg, B):
                first, b, tail = expr.args[:i], expr.args[i], expr.args[i + 1:]
                return B(*[A(*(first + (arg,) + tail)) for arg in b.args])
        return expr
    return distribute_rl


def subs(a, b):
    """ Replace expressions exactly """
    def subs_rl(expr):
        if expr == a:
            return b
        else:
            return expr
    return subs_rl


# Functions that are rules
def unpack(expr):
    """ Rule to unpack singleton args

    >>> from sympy.strategies import unpack
    >>> from sympy import Basic, S
    >>> unpack(Basic(S(2)))
    2
    """
    if len(expr.args) == 1:
        return expr.args[0]
    else:
        return expr


def flatten(expr, new=new):
    """ Flatten T(a, b, T(c, d), T2(e)) to T(a, b, c, d, T2(e)) """
    cls = expr.__class__
    args = []
    for arg in expr.args:
        if arg.__class__ == cls:
            args.extend(arg.args)
        else:
            args.append(arg)
    return new(expr.__class__, *args)


def rebuild(expr):
    """ Rebuild a SymPy tree.

    Explanation
    ===========

    This function recursively calls constructors in the expression tree.
    This forces canonicalization and removes ugliness introduced by the use of
    Basic.__new__
    """
    if expr.is_Atom:
        return expr
    else:
        return expr.func(*list(map(rebuild, expr.args)))

# === atelier-kyo-manager/tools\research_tool.py ===
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import csv
import time
import random
import os
import re
from urllib.parse import urljoin

# ====================
# 設定部分（必要に応じて変更）
# ====================
BASE_URL = "https://www.buyma.com/r/_%s/"  # ブランド名を%sに挿入
BRANDS = ["GUCCI", "PRADA"]                # 対象ブランドリスト
MAX_PAGES = 2                              # 最大取得ページ数
DELAY_RANGE = (10, 30)                     # ランダム待機時間（秒）
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X)..."
]
PROXIES = [
    "http://proxy1.example.com:8080",
    "http://proxy2.example.com:8080"
]

# ====================
# Chrome設定
# ====================
def setup_driver():
    chrome_options = Options()

    # Headlessモード
    chrome_options.add_argument("--headless=new")

    # プロキシ設定
    chrome_options.add_argument(f"--proxy-server={random.choice(PROXIES)}")

    # ユーザーエージェントローテーション
    chrome_options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")

    # 基本設定
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    return webdriver.Chrome(options=chrome_options)

# ====================
# 商品詳細取得関数
# ====================
def get_product_details(driver, product_url):
    """商品詳細ページから追加情報を取得"""
    try:
        driver.get(product_url)
        time.sleep(random.uniform(*DELAY_RANGE))

        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.product_detail"))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 追加情報取得例
        description = soup.select_one('div.product_description').text.strip()[:200] + "..." if soup.select_one('div.product_description') else ""
        seller_info = soup.select_one('div.seller_profile').text.strip()[:100] + "..." if soup.select_one('div.seller_profile') else ""

        return {
            'description': description,
            'seller_info': seller_info
        }

    except Exception as e:
        print(f"詳細ページエラー: {str(e)}")
        return {}

# ====================
# メインスクレイピング関数
# ====================
def scrape_brand(driver, brand):
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools', f'buyma_{brand.lower()}_products.csv')
    products = []

    for page in range(1, MAX_PAGES + 1):
        try:
            # ページアクセス
            url = BASE_URL % brand + f"?pageno={page}"
            driver.get(url)
            time.sleep(random.uniform(*DELAY_RANGE))

            # 商品コンテナ待機
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.product_body"))
            )

            # ページネーション確認
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            next_page = soup.select_one('a.pagination_next:not(.disabled)')
            if not next_page and page != 1:
                break

            # 商品情報取得
            for item in soup.select('div.product_body'):
                # 基本情報
                name_elem = item.select_one('div.product_name')
                name = name_elem.text.strip()[:50] + "..." if name_elem else "商品名不明"
                name = re.sub(r'★|即発送|SALE\!|即納|新品', '', name).strip()

                # 価格
                price_elem = item.select_one('span.Price_Txt') or \
                            item.select_one('div.product_price span') or \
                            item.select_one('.price')
                price = re.sub(r'[^\d,]', '', price_elem.text.strip()).replace(',', '') if price_elem else "0"

                # 画像URL
                image_elem = item.select_one('img[loading="lazy"]')
                image_url = ""
                if image_elem:
                    image_url = image_elem.get('data-src') or image_elem.get('src')
                    if image_url.startswith('//'):
                        image_url = 'https:' + image_url
                    elif image_url.startswith('/'):
                        image_url = urljoin(url, image_url)

                # 詳細情報（確率制限）
                details = {}
                if random.random() < 0.3:
                    detail_link = item.select_one('a.product_name')['href']
                    details = get_product_details(driver, urljoin(url, detail_link)) if detail_link else {}

                products.append({
                    'brand': brand,
                    'name': name,
                    'price': price,
                    'image_url': image_url,
                    **details
                })

        except Exception as e:
            print(f"ページ{page}処理中にエラー: {str(e)}")
            continue

    # CSV保存
    if products:
        with open(csv_path, 'w', newline='', encoding='cp932', errors='replace') as f:
            writer = csv.DictWriter(f, fieldnames=['brand', 'name', 'price', 'image_url', 'description', 'seller_info'])
            writer.writeheader()
            writer.writerows(products)
        print(f"{brand} - {len(products)}件の商品を保存しました")
    else:
        print(f"{brand} - 商品データが0件のため保存をスキップしました")

# ====================
# メイン実行
# ====================
def main():
    driver = setup_driver()
    try:
        for brand in BRANDS:
            start_time = time.time()
            scrape_brand(driver, brand)
            elapsed_time = time.time() - start_time
            sleep_time = max(random.randint(*DELAY_RANGE) - elapsed_time, 5)
            time.sleep(sleep_time)
    finally:
        driver.quit()

if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dotenv\parser.py ===
import codecs
import re
from typing import (IO, Iterator, Match, NamedTuple, Optional,  # noqa:F401
                    Pattern, Sequence, Tuple)


def make_regex(string: str, extra_flags: int = 0) -> Pattern[str]:
    return re.compile(string, re.UNICODE | extra_flags)


_newline = make_regex(r"(\r\n|\n|\r)")
_multiline_whitespace = make_regex(r"\s*", extra_flags=re.MULTILINE)
_whitespace = make_regex(r"[^\S\r\n]*")
_export = make_regex(r"(?:export[^\S\r\n]+)?")
_single_quoted_key = make_regex(r"'([^']+)'")
_unquoted_key = make_regex(r"([^=\#\s]+)")
_equal_sign = make_regex(r"(=[^\S\r\n]*)")
_single_quoted_value = make_regex(r"'((?:\\'|[^'])*)'")
_double_quoted_value = make_regex(r'"((?:\\"|[^"])*)"')
_unquoted_value = make_regex(r"([^\r\n]*)")
_comment = make_regex(r"(?:[^\S\r\n]*#[^\r\n]*)?")
_end_of_line = make_regex(r"[^\S\r\n]*(?:\r\n|\n|\r|$)")
_rest_of_line = make_regex(r"[^\r\n]*(?:\r|\n|\r\n)?")
_double_quote_escapes = make_regex(r"\\[\\'\"abfnrtv]")
_single_quote_escapes = make_regex(r"\\[\\']")


class Original(NamedTuple):
    string: str
    line: int


class Binding(NamedTuple):
    key: Optional[str]
    value: Optional[str]
    original: Original
    error: bool


class Position:
    def __init__(self, chars: int, line: int) -> None:
        self.chars = chars
        self.line = line

    @classmethod
    def start(cls) -> "Position":
        return cls(chars=0, line=1)

    def set(self, other: "Position") -> None:
        self.chars = other.chars
        self.line = other.line

    def advance(self, string: str) -> None:
        self.chars += len(string)
        self.line += len(re.findall(_newline, string))


class Error(Exception):
    pass


class Reader:
    def __init__(self, stream: IO[str]) -> None:
        self.string = stream.read()
        self.position = Position.start()
        self.mark = Position.start()

    def has_next(self) -> bool:
        return self.position.chars < len(self.string)

    def set_mark(self) -> None:
        self.mark.set(self.position)

    def get_marked(self) -> Original:
        return Original(
            string=self.string[self.mark.chars:self.position.chars],
            line=self.mark.line,
        )

    def peek(self, count: int) -> str:
        return self.string[self.position.chars:self.position.chars + count]

    def read(self, count: int) -> str:
        result = self.string[self.position.chars:self.position.chars + count]
        if len(result) < count:
            raise Error("read: End of string")
        self.position.advance(result)
        return result

    def read_regex(self, regex: Pattern[str]) -> Sequence[str]:
        match = regex.match(self.string, self.position.chars)
        if match is None:
            raise Error("read_regex: Pattern not found")
        self.position.advance(self.string[match.start():match.end()])
        return match.groups()


def decode_escapes(regex: Pattern[str], string: str) -> str:
    def decode_match(match: Match[str]) -> str:
        return codecs.decode(match.group(0), 'unicode-escape')  # type: ignore

    return regex.sub(decode_match, string)


def parse_key(reader: Reader) -> Optional[str]:
    char = reader.peek(1)
    if char == "#":
        return None
    elif char == "'":
        (key,) = reader.read_regex(_single_quoted_key)
    else:
        (key,) = reader.read_regex(_unquoted_key)
    return key


def parse_unquoted_value(reader: Reader) -> str:
    (part,) = reader.read_regex(_unquoted_value)
    return re.sub(r"\s+#.*", "", part).rstrip()


def parse_value(reader: Reader) -> str:
    char = reader.peek(1)
    if char == u"'":
        (value,) = reader.read_regex(_single_quoted_value)
        return decode_escapes(_single_quote_escapes, value)
    elif char == u'"':
        (value,) = reader.read_regex(_double_quoted_value)
        return decode_escapes(_double_quote_escapes, value)
    elif char in (u"", u"\n", u"\r"):
        return u""
    else:
        return parse_unquoted_value(reader)


def parse_binding(reader: Reader) -> Binding:
    reader.set_mark()
    try:
        reader.read_regex(_multiline_whitespace)
        if not reader.has_next():
            return Binding(
                key=None,
                value=None,
                original=reader.get_marked(),
                error=False,
            )
        reader.read_regex(_export)
        key = parse_key(reader)
        reader.read_regex(_whitespace)
        if reader.peek(1) == "=":
            reader.read_regex(_equal_sign)
            value: Optional[str] = parse_value(reader)
        else:
            value = None
        reader.read_regex(_comment)
        reader.read_regex(_end_of_line)
        return Binding(
            key=key,
            value=value,
            original=reader.get_marked(),
            error=False,
        )
    except Error:
        reader.read_regex(_rest_of_line)
        return Binding(
            key=None,
            value=None,
            original=reader.get_marked(),
            error=True,
        )


def parse_stream(stream: IO[str]) -> Iterator[Binding]:
    reader = Reader(stream)
    while reader.has_next():
        yield parse_binding(reader)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\backend\backend.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Implements ONNX's backend API.
"""

import os
import unittest

import packaging.version
from onnx import ModelProto, helper, version  # noqa: F401
from onnx.backend.base import Backend
from onnx.checker import check_model

from onnxruntime import InferenceSession, SessionOptions, get_available_providers, get_device
from onnxruntime.backend.backend_rep import OnnxRuntimeBackendRep


class OnnxRuntimeBackend(Backend):
    """
    Implements
    `ONNX's backend API <https://github.com/onnx/onnx/blob/main/docs/ImplementingAnOnnxBackend.md>`_
    with *ONNX Runtime*.
    The backend is mostly used when you need to switch between
    multiple runtimes with the same API.
    `Importing models from ONNX to Caffe2 <https://github.com/onnx/tutorials/blob/master/tutorials/OnnxCaffe2Import.ipynb>`_
    shows how to use *caffe2* as a backend for a converted model.
    Note: This is not the official Python API.
    """

    allowReleasedOpsetsOnly = bool(os.getenv("ALLOW_RELEASED_ONNX_OPSET_ONLY", "1") == "1")  # noqa: N815

    @classmethod
    def is_compatible(cls, model, device=None, **kwargs):
        """
        Return whether the model is compatible with the backend.

        :param model: unused
        :param device: None to use the default device or a string (ex: `'CPU'`)
        :return: boolean
        """
        if device is None:
            device = get_device()
        return cls.supports_device(device)

    @classmethod
    def is_opset_supported(cls, model):
        """
        Return whether the opset for the model is supported by the backend.
        When By default only released onnx opsets are allowed by the backend
        To test new opsets env variable ALLOW_RELEASED_ONNX_OPSET_ONLY should be set to 0

        :param model: Model whose opsets needed to be verified.
        :return: boolean and error message if opset is not supported.
        """
        if cls.allowReleasedOpsetsOnly:
            for opset in model.opset_import:
                domain = opset.domain if opset.domain else "ai.onnx"
                try:
                    key = (domain, opset.version)
                    if key not in helper.OP_SET_ID_VERSION_MAP:
                        error_message = (
                            "Skipping this test as only released onnx opsets are supported."
                            "To run this test set env variable ALLOW_RELEASED_ONNX_OPSET_ONLY to 0."
                            f" Got Domain '{domain}' version '{opset.version}'."
                        )
                        return False, error_message
                except AttributeError:
                    # for some CI pipelines accessing helper.OP_SET_ID_VERSION_MAP
                    # is generating attribute error. TODO investigate the pipelines to
                    # fix this error. Falling back to a simple version check when this error is encountered
                    if (domain == "ai.onnx" and opset.version > 12) or (domain == "ai.ommx.ml" and opset.version > 2):
                        error_message = (
                            "Skipping this test as only released onnx opsets are supported."
                            "To run this test set env variable ALLOW_RELEASED_ONNX_OPSET_ONLY to 0."
                            f" Got Domain '{domain}' version '{opset.version}'."
                        )
                        return False, error_message
        return True, ""

    @classmethod
    def supports_device(cls, device):
        """
        Check whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        if device == "CUDA":
            device = "GPU"
        return "-" + device in get_device() or device + "-" in get_device() or device == get_device()

    @classmethod
    def prepare(cls, model, device=None, **kwargs):
        """
        Load the model and creates a :class:`onnxruntime.InferenceSession`
        ready to be used as a backend.

        :param model: ModelProto (returned by `onnx.load`),
            string for a filename or bytes for a serialized model
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: see :class:`onnxruntime.SessionOptions`
        :return: :class:`onnxruntime.InferenceSession`
        """
        if isinstance(model, OnnxRuntimeBackendRep):
            return model
        elif isinstance(model, InferenceSession):
            return OnnxRuntimeBackendRep(model)
        elif isinstance(model, (str, bytes)):
            options = SessionOptions()
            for k, v in kwargs.items():
                if hasattr(options, k):
                    setattr(options, k, v)

            excluded_providers = os.getenv("ORT_ONNX_BACKEND_EXCLUDE_PROVIDERS", default="").split(",")
            providers = [x for x in get_available_providers() if (x not in excluded_providers)]

            inf = InferenceSession(model, sess_options=options, providers=providers)
            # backend API is primarily used for ONNX test/validation. As such, we should disable session.run() fallback
            # which may hide test failures.
            inf.disable_fallback()
            if device is not None and not cls.supports_device(device):
                raise RuntimeError(f"Incompatible device expected '{device}', got '{get_device()}'")
            return cls.prepare(inf, device, **kwargs)
        else:
            # type: ModelProto
            # check_model serializes the model anyways, so serialize the model once here
            # and reuse it below in the cls.prepare call to avoid an additional serialization
            # only works with onnx >= 1.10.0 hence the version check
            onnx_version = packaging.version.parse(version.version) or packaging.version.Version("0")
            onnx_supports_serialized_model_check = onnx_version.release >= (1, 10, 0)
            bin_or_model = model.SerializeToString() if onnx_supports_serialized_model_check else model
            check_model(bin_or_model)
            opset_supported, error_message = cls.is_opset_supported(model)
            if not opset_supported:
                raise unittest.SkipTest(error_message)
            # Now bin might be serialized, if it's not we need to serialize it otherwise we'll have
            # an infinite recursive call
            bin = bin_or_model
            if not isinstance(bin, (str, bytes)):
                bin = bin.SerializeToString()
            return cls.prepare(bin, device, **kwargs)

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        """
        Compute the prediction.

        :param model: :class:`onnxruntime.InferenceSession` returned
            by function *prepare*
        :param inputs: inputs
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: see :class:`onnxruntime.RunOptions`
        :return: predictions
        """
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        """
        This method is not implemented as it is much more efficient
        to run a whole model than every node independently.
        """
        raise NotImplementedError("It is much more efficient to run a whole model than every node independently.")


is_compatible = OnnxRuntimeBackend.is_compatible
prepare = OnnxRuntimeBackend.prepare
run = OnnxRuntimeBackend.run_model
supports_device = OnnxRuntimeBackend.supports_device

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\autocompletion.py ===
"""Logic that powers autocompletion installed by ``pip completion``."""

import optparse
import os
import sys
from itertools import chain
from typing import Any, Iterable, List, Optional

from pip._internal.cli.main_parser import create_main_parser
from pip._internal.commands import commands_dict, create_command
from pip._internal.metadata import get_default_environment


def autocomplete() -> None:
    """Entry Point for completion of main and subcommand options."""
    # Don't complete if user hasn't sourced bash_completion file.
    if "PIP_AUTO_COMPLETE" not in os.environ:
        return
    # Don't complete if autocompletion environment variables
    # are not present
    if not os.environ.get("COMP_WORDS") or not os.environ.get("COMP_CWORD"):
        return
    cwords = os.environ["COMP_WORDS"].split()[1:]
    cword = int(os.environ["COMP_CWORD"])
    try:
        current = cwords[cword - 1]
    except IndexError:
        current = ""

    parser = create_main_parser()
    subcommands = list(commands_dict)
    options = []

    # subcommand
    subcommand_name: Optional[str] = None
    for word in cwords:
        if word in subcommands:
            subcommand_name = word
            break
    # subcommand options
    if subcommand_name is not None:
        # special case: 'help' subcommand has no options
        if subcommand_name == "help":
            sys.exit(1)
        # special case: list locally installed dists for show and uninstall
        should_list_installed = not current.startswith("-") and subcommand_name in [
            "show",
            "uninstall",
        ]
        if should_list_installed:
            env = get_default_environment()
            lc = current.lower()
            installed = [
                dist.canonical_name
                for dist in env.iter_installed_distributions(local_only=True)
                if dist.canonical_name.startswith(lc)
                and dist.canonical_name not in cwords[1:]
            ]
            # if there are no dists installed, fall back to option completion
            if installed:
                for dist in installed:
                    print(dist)
                sys.exit(1)

        should_list_installables = (
            not current.startswith("-") and subcommand_name == "install"
        )
        if should_list_installables:
            for path in auto_complete_paths(current, "path"):
                print(path)
            sys.exit(1)

        subcommand = create_command(subcommand_name)

        for opt in subcommand.parser.option_list_all:
            if opt.help != optparse.SUPPRESS_HELP:
                options += [
                    (opt_str, opt.nargs) for opt_str in opt._long_opts + opt._short_opts
                ]

        # filter out previously specified options from available options
        prev_opts = [x.split("=")[0] for x in cwords[1 : cword - 1]]
        options = [(x, v) for (x, v) in options if x not in prev_opts]
        # filter options by current input
        options = [(k, v) for k, v in options if k.startswith(current)]
        # get completion type given cwords and available subcommand options
        completion_type = get_path_completion_type(
            cwords,
            cword,
            subcommand.parser.option_list_all,
        )
        # get completion files and directories if ``completion_type`` is
        # ``<file>``, ``<dir>`` or ``<path>``
        if completion_type:
            paths = auto_complete_paths(current, completion_type)
            options = [(path, 0) for path in paths]
        for option in options:
            opt_label = option[0]
            # append '=' to options which require args
            if option[1] and option[0][:2] == "--":
                opt_label += "="
            print(opt_label)
    else:
        # show main parser options only when necessary

        opts = [i.option_list for i in parser.option_groups]
        opts.append(parser.option_list)
        flattened_opts = chain.from_iterable(opts)
        if current.startswith("-"):
            for opt in flattened_opts:
                if opt.help != optparse.SUPPRESS_HELP:
                    subcommands += opt._long_opts + opt._short_opts
        else:
            # get completion type given cwords and all available options
            completion_type = get_path_completion_type(cwords, cword, flattened_opts)
            if completion_type:
                subcommands = list(auto_complete_paths(current, completion_type))

        print(" ".join([x for x in subcommands if x.startswith(current)]))
    sys.exit(1)


def get_path_completion_type(
    cwords: List[str], cword: int, opts: Iterable[Any]
) -> Optional[str]:
    """Get the type of path completion (``file``, ``dir``, ``path`` or None)

    :param cwords: same as the environmental variable ``COMP_WORDS``
    :param cword: same as the environmental variable ``COMP_CWORD``
    :param opts: The available options to check
    :return: path completion type (``file``, ``dir``, ``path`` or None)
    """
    if cword < 2 or not cwords[cword - 2].startswith("-"):
        return None
    for opt in opts:
        if opt.help == optparse.SUPPRESS_HELP:
            continue
        for o in str(opt).split("/"):
            if cwords[cword - 2].split("=")[0] == o:
                if not opt.metavar or any(
                    x in ("path", "file", "dir") for x in opt.metavar.split("/")
                ):
                    return opt.metavar
    return None


def auto_complete_paths(current: str, completion_type: str) -> Iterable[str]:
    """If ``completion_type`` is ``file`` or ``path``, list all regular files
    and directories starting with ``current``; otherwise only list directories
    starting with ``current``.

    :param current: The word to be completed
    :param completion_type: path completion type(``file``, ``path`` or ``dir``)
    :return: A generator of regular files and/or directories
    """
    directory, filename = os.path.split(current)
    current_path = os.path.abspath(directory)
    # Don't complete paths if they can't be accessed
    if not os.access(current_path, os.R_OK):
        return
    filename = os.path.normcase(filename)
    # list all files that start with ``filename``
    file_list = (
        x for x in os.listdir(current_path) if os.path.normcase(x).startswith(filename)
    )
    for f in file_list:
        opt = os.path.join(current_path, f)
        comp_file = os.path.normcase(os.path.join(directory, f))
        # complete regular files when there is not ``<dir>`` after option
        # complete directories when there is ``<file>``, ``<path>`` or
        # ``<dir>``after option
        if completion_type != "dir" and os.path.isfile(opt):
            yield comp_file
        elif os.path.isdir(opt):
            yield os.path.join(comp_file, "")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\provision.py ===
# dialects/postgresql/provision.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

import time

from ... import exc
from ... import inspect
from ... import text
from ...testing import warn_test_suite
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_post_tables
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import prepare_for_drop_tables
from ...testing.provision import set_default_schema_on_connection
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert


@create_db.for_db("postgresql")
def _pg_create_db(cfg, eng, ident):
    template_db = cfg.options.postgresql_templatedb

    with eng.execution_options(isolation_level="AUTOCOMMIT").begin() as conn:
        if not template_db:
            template_db = conn.exec_driver_sql(
                "select current_database()"
            ).scalar()

        attempt = 0
        while True:
            try:
                conn.exec_driver_sql(
                    "CREATE DATABASE %s TEMPLATE %s" % (ident, template_db)
                )
            except exc.OperationalError as err:
                attempt += 1
                if attempt >= 3:
                    raise
                if "accessed by other users" in str(err):
                    log.info(
                        "Waiting to create %s, URI %r, "
                        "template DB %s is in use sleeping for .5",
                        ident,
                        eng.url,
                        template_db,
                    )
                    time.sleep(0.5)
            except:
                raise
            else:
                break


@drop_db.for_db("postgresql")
def _pg_drop_db(cfg, eng, ident):
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        with conn.begin():
            conn.execute(
                text(
                    "select pg_terminate_backend(pid) from pg_stat_activity "
                    "where usename=current_user and pid != pg_backend_pid() "
                    "and datname=:dname"
                ),
                dict(dname=ident),
            )
            conn.exec_driver_sql("DROP DATABASE %s" % ident)


@temp_table_keyword_args.for_db("postgresql")
def _postgresql_temp_table_keyword_args(cfg, eng):
    return {"prefixes": ["TEMPORARY"]}


@set_default_schema_on_connection.for_db("postgresql")
def _postgresql_set_default_schema_on_connection(
    cfg, dbapi_connection, schema_name
):
    existing_autocommit = dbapi_connection.autocommit
    dbapi_connection.autocommit = True
    cursor = dbapi_connection.cursor()
    cursor.execute("SET SESSION search_path='%s'" % schema_name)
    cursor.close()
    dbapi_connection.autocommit = existing_autocommit


@drop_all_schema_objects_pre_tables.for_db("postgresql")
def drop_all_schema_objects_pre_tables(cfg, eng):
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for xid in conn.exec_driver_sql(
            "select gid from pg_prepared_xacts"
        ).scalars():
            conn.exec_driver_sql("ROLLBACK PREPARED '%s'" % xid)


@drop_all_schema_objects_post_tables.for_db("postgresql")
def drop_all_schema_objects_post_tables(cfg, eng):
    from sqlalchemy.dialects import postgresql

    inspector = inspect(eng)
    with eng.begin() as conn:
        for enum in inspector.get_enums("*"):
            conn.execute(
                postgresql.DropEnumType(
                    postgresql.ENUM(name=enum["name"], schema=enum["schema"])
                )
            )


@prepare_for_drop_tables.for_db("postgresql")
def prepare_for_drop_tables(config, connection):
    """Ensure there are no locks on the current username/database."""

    result = connection.exec_driver_sql(
        "select pid, state, wait_event_type, query "
        # "select pg_terminate_backend(pid), state, wait_event_type "
        "from pg_stat_activity where "
        "usename=current_user "
        "and datname=current_database() and state='idle in transaction' "
        "and pid != pg_backend_pid()"
    )
    rows = result.all()  # noqa
    if rows:
        warn_test_suite(
            "PostgreSQL may not be able to DROP tables due to "
            "idle in transaction: %s"
            % ("; ".join(row._mapping["query"] for row in rows))
        )


@upsert.for_db("postgresql")
def _upsert(
    cfg, table, returning, *, set_lambda=None, sort_by_parameter_order=False
):
    from sqlalchemy.dialects.postgresql import insert

    stmt = insert(table)

    table_pk = inspect(table).selectable

    if set_lambda:
        stmt = stmt.on_conflict_do_update(
            index_elements=table_pk.primary_key, set_=set_lambda(stmt.excluded)
        )
    else:
        stmt = stmt.on_conflict_do_nothing()

    stmt = stmt.returning(
        *returning, sort_by_parameter_order=sort_by_parameter_order
    )
    return stmt


_extensions = [
    ("citext", (13,)),
    ("hstore", (13,)),
]


@post_configure_engine.for_db("postgresql")
def _create_citext_extension(url, engine, follower_ident):
    with engine.connect() as conn:
        for extension, min_version in _extensions:
            if conn.dialect.server_version_info >= min_version:
                conn.execute(
                    text(f"CREATE EXTENSION IF NOT EXISTS {extension}")
                )
                conn.commit()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tree.py ===
def pprint_nodes(subtrees):
    """
    Prettyprints systems of nodes.

    Examples
    ========

    >>> from sympy.printing.tree import pprint_nodes
    >>> print(pprint_nodes(["a", "b1\\nb2", "c"]))
    +-a
    +-b1
    | b2
    +-c

    """
    def indent(s, type=1):
        x = s.split("\n")
        r = "+-%s\n" % x[0]
        for a in x[1:]:
            if a == "":
                continue
            if type == 1:
                r += "| %s\n" % a
            else:
                r += "  %s\n" % a
        return r
    if not subtrees:
        return ""
    f = ""
    for a in subtrees[:-1]:
        f += indent(a)
    f += indent(subtrees[-1], 2)
    return f


def print_node(node, assumptions=True):
    """
    Returns information about the "node".

    This includes class name, string representation and assumptions.

    Parameters
    ==========

    assumptions : bool, optional
        See the ``assumptions`` keyword in ``tree``
    """
    s = "%s: %s\n" % (node.__class__.__name__, str(node))

    if assumptions:
        d = node._assumptions
    else:
        d = None

    if d:
        for a in sorted(d):
            v = d[a]
            if v is None:
                continue
            s += "%s: %s\n" % (a, v)

    return s


def tree(node, assumptions=True):
    """
    Returns a tree representation of "node" as a string.

    It uses print_node() together with pprint_nodes() on node.args recursively.

    Parameters
    ==========

    assumptions : bool, optional
        The flag to decide whether to print out all the assumption data
        (such as ``is_integer`, ``is_real``) associated with the
        expression or not.

        Enabling the flag makes the result verbose, and the printed
        result may not be deterministic because of the randomness used
        in backtracing the assumptions.

    See Also
    ========

    print_tree

    """
    subtrees = []
    for arg in node.args:
        subtrees.append(tree(arg, assumptions=assumptions))
    s = print_node(node, assumptions=assumptions) + pprint_nodes(subtrees)
    return s


def print_tree(node, assumptions=True):
    """
    Prints a tree representation of "node".

    Parameters
    ==========

    assumptions : bool, optional
        The flag to decide whether to print out all the assumption data
        (such as ``is_integer`, ``is_real``) associated with the
        expression or not.

        Enabling the flag makes the result verbose, and the printed
        result may not be deterministic because of the randomness used
        in backtracing the assumptions.

    Examples
    ========

    >>> from sympy.printing import print_tree
    >>> from sympy import Symbol
    >>> x = Symbol('x', odd=True)
    >>> y = Symbol('y', even=True)

    Printing with full assumptions information:

    >>> print_tree(y**x)
    Pow: y**x
    +-Symbol: y
    | algebraic: True
    | commutative: True
    | complex: True
    | even: True
    | extended_real: True
    | finite: True
    | hermitian: True
    | imaginary: False
    | infinite: False
    | integer: True
    | irrational: False
    | noninteger: False
    | odd: False
    | rational: True
    | real: True
    | transcendental: False
    +-Symbol: x
      algebraic: True
      commutative: True
      complex: True
      even: False
      extended_nonzero: True
      extended_real: True
      finite: True
      hermitian: True
      imaginary: False
      infinite: False
      integer: True
      irrational: False
      noninteger: False
      nonzero: True
      odd: True
      rational: True
      real: True
      transcendental: False
      zero: False

    Hiding the assumptions:

    >>> print_tree(y**x, assumptions=False)
    Pow: y**x
    +-Symbol: y
    +-Symbol: x

    See Also
    ========

    tree

    """
    print(tree(node, assumptions=assumptions))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\properties.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.xml.constants import DRAWING_NS
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Bool,
    Integer,
    Set,
    String,
    Alias,
    NoneSet,
)
from openpyxl.descriptors.excel import ExtensionList as OfficeArtExtensionList

from .geometry import GroupTransform2D, Scene3D
from .text import Hyperlink


class GroupShapeProperties(Serialisable):

    tagname = "grpSpPr"

    bwMode = NoneSet(values=(['clr', 'auto', 'gray', 'ltGray', 'invGray',
                          'grayWhite', 'blackGray', 'blackWhite', 'black', 'white', 'hidden']))
    xfrm = Typed(expected_type=GroupTransform2D, allow_none=True)
    scene3d = Typed(expected_type=Scene3D, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    def __init__(self,
                 bwMode=None,
                 xfrm=None,
                 scene3d=None,
                 extLst=None,
                ):
        self.bwMode = bwMode
        self.xfrm = xfrm
        self.scene3d = scene3d
        self.extLst = extLst


class GroupLocking(Serialisable):

    tagname = "grpSpLocks"
    namespace = DRAWING_NS

    noGrp = Bool(allow_none=True)
    noUngrp = Bool(allow_none=True)
    noSelect = Bool(allow_none=True)
    noRot = Bool(allow_none=True)
    noChangeAspect = Bool(allow_none=True)
    noMove = Bool(allow_none=True)
    noResize = Bool(allow_none=True)
    noChangeArrowheads = Bool(allow_none=True)
    noEditPoints = Bool(allow_none=True)
    noAdjustHandles = Bool(allow_none=True)
    noChangeArrowheads = Bool(allow_none=True)
    noChangeShapeType = Bool(allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    __elements__ = ()

    def __init__(self,
                 noGrp=None,
                 noUngrp=None,
                 noSelect=None,
                 noRot=None,
                 noChangeAspect=None,
                 noChangeArrowheads=None,
                 noMove=None,
                 noResize=None,
                 noEditPoints=None,
                 noAdjustHandles=None,
                 noChangeShapeType=None,
                 extLst=None,
                ):
        self.noGrp = noGrp
        self.noUngrp = noUngrp
        self.noSelect = noSelect
        self.noRot = noRot
        self.noChangeAspect = noChangeAspect
        self.noChangeArrowheads = noChangeArrowheads
        self.noMove = noMove
        self.noResize = noResize
        self.noEditPoints = noEditPoints
        self.noAdjustHandles = noAdjustHandles
        self.noChangeShapeType = noChangeShapeType


class NonVisualGroupDrawingShapeProps(Serialisable):

    tagname = "cNvGrpSpPr"

    grpSpLocks = Typed(expected_type=GroupLocking, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    __elements__ = ("grpSpLocks",)

    def __init__(self,
                 grpSpLocks=None,
                 extLst=None,
                ):
        self.grpSpLocks = grpSpLocks


class NonVisualDrawingShapeProps(Serialisable):

    tagname = "cNvSpPr"

    spLocks = Typed(expected_type=GroupLocking, allow_none=True)
    txBax = Bool(allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    __elements__ = ("spLocks", "txBax")

    def __init__(self,
                 spLocks=None,
                 txBox=None,
                 extLst=None,
                ):
        self.spLocks = spLocks
        self.txBox = txBox


class NonVisualDrawingProps(Serialisable):

    tagname = "cNvPr"

    id = Integer()
    name = String()
    descr = String(allow_none=True)
    hidden = Bool(allow_none=True)
    title = String(allow_none=True)
    hlinkClick = Typed(expected_type=Hyperlink, allow_none=True)
    hlinkHover = Typed(expected_type=Hyperlink, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    __elements__ = ["hlinkClick", "hlinkHover"]

    def __init__(self,
                 id=None,
                 name=None,
                 descr=None,
                 hidden=None,
                 title=None,
                 hlinkClick=None,
                 hlinkHover=None,
                 extLst=None,
                ):
        self.id = id
        self.name = name
        self.descr = descr
        self.hidden = hidden
        self.title = title
        self.hlinkClick = hlinkClick
        self.hlinkHover = hlinkHover
        self.extLst = extLst

class NonVisualGroupShape(Serialisable):

    tagname = "nvGrpSpPr"

    cNvPr = Typed(expected_type=NonVisualDrawingProps)
    cNvGrpSpPr = Typed(expected_type=NonVisualGroupDrawingShapeProps)

    __elements__ = ("cNvPr", "cNvGrpSpPr")

    def __init__(self,
                 cNvPr=None,
                 cNvGrpSpPr=None,
                ):
        self.cNvPr = cNvPr
        self.cNvGrpSpPr = cNvGrpSpPr


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\page.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Float,
    Bool,
    Integer,
    NoneSet,
    )
from openpyxl.descriptors.excel import UniversalMeasure, Relation


class PrintPageSetup(Serialisable):
    """ Worksheet print page setup """

    tagname = "pageSetup"

    orientation = NoneSet(values=("default", "portrait", "landscape"))
    paperSize = Integer(allow_none=True)
    scale = Integer(allow_none=True)
    fitToHeight = Integer(allow_none=True)
    fitToWidth = Integer(allow_none=True)
    firstPageNumber = Integer(allow_none=True)
    useFirstPageNumber = Bool(allow_none=True)
    paperHeight = UniversalMeasure(allow_none=True)
    paperWidth = UniversalMeasure(allow_none=True)
    pageOrder = NoneSet(values=("downThenOver", "overThenDown"))
    usePrinterDefaults = Bool(allow_none=True)
    blackAndWhite = Bool(allow_none=True)
    draft = Bool(allow_none=True)
    cellComments = NoneSet(values=("asDisplayed", "atEnd"))
    errors = NoneSet(values=("displayed", "blank", "dash", "NA"))
    horizontalDpi = Integer(allow_none=True)
    verticalDpi = Integer(allow_none=True)
    copies = Integer(allow_none=True)
    id = Relation()


    def __init__(self,
                 worksheet=None,
                 orientation=None,
                 paperSize=None,
                 scale=None,
                 fitToHeight=None,
                 fitToWidth=None,
                 firstPageNumber=None,
                 useFirstPageNumber=None,
                 paperHeight=None,
                 paperWidth=None,
                 pageOrder=None,
                 usePrinterDefaults=None,
                 blackAndWhite=None,
                 draft=None,
                 cellComments=None,
                 errors=None,
                 horizontalDpi=None,
                 verticalDpi=None,
                 copies=None,
                 id=None):
        self._parent = worksheet
        self.orientation = orientation
        self.paperSize = paperSize
        self.scale = scale
        self.fitToHeight = fitToHeight
        self.fitToWidth = fitToWidth
        self.firstPageNumber = firstPageNumber
        self.useFirstPageNumber = useFirstPageNumber
        self.paperHeight = paperHeight
        self.paperWidth = paperWidth
        self.pageOrder = pageOrder
        self.usePrinterDefaults = usePrinterDefaults
        self.blackAndWhite = blackAndWhite
        self.draft = draft
        self.cellComments = cellComments
        self.errors = errors
        self.horizontalDpi = horizontalDpi
        self.verticalDpi = verticalDpi
        self.copies = copies
        self.id = id


    def __bool__(self):
        return bool(dict(self))




    @property
    def sheet_properties(self):
        """
        Proxy property
        """
        return self._parent.sheet_properties.pageSetUpPr


    @property
    def fitToPage(self):
        return self.sheet_properties.fitToPage


    @fitToPage.setter
    def fitToPage(self, value):
        self.sheet_properties.fitToPage = value


    @property
    def autoPageBreaks(self):
        return self.sheet_properties.autoPageBreaks


    @autoPageBreaks.setter
    def autoPageBreaks(self, value):
        self.sheet_properties.autoPageBreaks = value


    @classmethod
    def from_tree(cls, node):
        self = super().from_tree(node)
        self.id = None # strip link to binary settings
        return self


class PrintOptions(Serialisable):
    """ Worksheet print options """

    tagname = "printOptions"
    horizontalCentered = Bool(allow_none=True)
    verticalCentered = Bool(allow_none=True)
    headings = Bool(allow_none=True)
    gridLines = Bool(allow_none=True)
    gridLinesSet = Bool(allow_none=True)

    def __init__(self, horizontalCentered=None,
                 verticalCentered=None,
                 headings=None,
                 gridLines=None,
                 gridLinesSet=None,
                 ):
        self.horizontalCentered = horizontalCentered
        self.verticalCentered = verticalCentered
        self.headings = headings
        self.gridLines = gridLines
        self.gridLinesSet = gridLinesSet


    def __bool__(self):
        return bool(dict(self))


class PageMargins(Serialisable):
    """
    Information about page margins for view/print layouts.
    Standard values (in inches)
    left, right = 0.75
    top, bottom = 1
    header, footer = 0.5
    """
    tagname = "pageMargins"

    left = Float()
    right = Float()
    top = Float()
    bottom = Float()
    header = Float()
    footer = Float()

    def __init__(self, left=0.75, right=0.75, top=1, bottom=1, header=0.5,
                 footer=0.5):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.header = header
        self.footer = footer

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\arrow\extension_types.py ===
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow

from pandas.compat import pa_version_under14p1

from pandas.core.dtypes.dtypes import (
    IntervalDtype,
    PeriodDtype,
)

from pandas.core.arrays.interval import VALID_CLOSED

if TYPE_CHECKING:
    from pandas._typing import IntervalClosedType


class ArrowPeriodType(pyarrow.ExtensionType):
    def __init__(self, freq) -> None:
        # attributes need to be set first before calling
        # super init (as that calls serialize)
        self._freq = freq
        pyarrow.ExtensionType.__init__(self, pyarrow.int64(), "pandas.period")

    @property
    def freq(self):
        return self._freq

    def __arrow_ext_serialize__(self) -> bytes:
        metadata = {"freq": self.freq}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowPeriodType:
        metadata = json.loads(serialized.decode())
        return ArrowPeriodType(metadata["freq"])

    def __eq__(self, other):
        if isinstance(other, pyarrow.BaseExtensionType):
            return type(self) == type(other) and self.freq == other.freq
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((str(self), self.freq))

    def to_pandas_dtype(self) -> PeriodDtype:
        return PeriodDtype(freq=self.freq)


# register the type with a dummy instance
_period_type = ArrowPeriodType("D")
pyarrow.register_extension_type(_period_type)


class ArrowIntervalType(pyarrow.ExtensionType):
    def __init__(self, subtype, closed: IntervalClosedType) -> None:
        # attributes need to be set first before calling
        # super init (as that calls serialize)
        assert closed in VALID_CLOSED
        self._closed: IntervalClosedType = closed
        if not isinstance(subtype, pyarrow.DataType):
            subtype = pyarrow.type_for_alias(str(subtype))
        self._subtype = subtype

        storage_type = pyarrow.struct([("left", subtype), ("right", subtype)])
        pyarrow.ExtensionType.__init__(self, storage_type, "pandas.interval")

    @property
    def subtype(self):
        return self._subtype

    @property
    def closed(self) -> IntervalClosedType:
        return self._closed

    def __arrow_ext_serialize__(self) -> bytes:
        metadata = {"subtype": str(self.subtype), "closed": self.closed}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowIntervalType:
        metadata = json.loads(serialized.decode())
        subtype = pyarrow.type_for_alias(metadata["subtype"])
        closed = metadata["closed"]
        return ArrowIntervalType(subtype, closed)

    def __eq__(self, other):
        if isinstance(other, pyarrow.BaseExtensionType):
            return (
                type(self) == type(other)
                and self.subtype == other.subtype
                and self.closed == other.closed
            )
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((str(self), str(self.subtype), self.closed))

    def to_pandas_dtype(self) -> IntervalDtype:
        return IntervalDtype(self.subtype.to_pandas_dtype(), self.closed)


# register the type with a dummy instance
_interval_type = ArrowIntervalType(pyarrow.int64(), "left")
pyarrow.register_extension_type(_interval_type)


_ERROR_MSG = """\
Disallowed deserialization of 'arrow.py_extension_type':
storage_type = {storage_type}
serialized = {serialized}
pickle disassembly:\n{pickle_disassembly}

Reading of untrusted Parquet or Feather files with a PyExtensionType column
allows arbitrary code execution.
If you trust this file, you can enable reading the extension type by one of:

- upgrading to pyarrow >= 14.0.1, and call `pa.PyExtensionType.set_auto_load(True)`
- install pyarrow-hotfix (`pip install pyarrow-hotfix`) and disable it by running
  `import pyarrow_hotfix; pyarrow_hotfix.uninstall()`

We strongly recommend updating your Parquet/Feather files to use extension types
derived from `pyarrow.ExtensionType` instead, and register this type explicitly.
"""


def patch_pyarrow():
    # starting from pyarrow 14.0.1, it has its own mechanism
    if not pa_version_under14p1:
        return

    # if https://github.com/pitrou/pyarrow-hotfix was installed and enabled
    if getattr(pyarrow, "_hotfix_installed", False):
        return

    class ForbiddenExtensionType(pyarrow.ExtensionType):
        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            import io
            import pickletools

            out = io.StringIO()
            pickletools.dis(serialized, out)
            raise RuntimeError(
                _ERROR_MSG.format(
                    storage_type=storage_type,
                    serialized=serialized,
                    pickle_disassembly=out.getvalue(),
                )
            )

    pyarrow.unregister_extension_type("arrow.py_extension_type")
    pyarrow.register_extension_type(
        ForbiddenExtensionType(pyarrow.null(), "arrow.py_extension_type")
    )

    pyarrow._hotfix_installed = True


patch_pyarrow()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\cells.py ===
from __future__ import annotations

from functools import lru_cache
from typing import Callable

from ._cell_widths import CELL_WIDTHS

# Ranges of unicode ordinals that produce a 1-cell wide character
# This is non-exhaustive, but covers most common Western characters
_SINGLE_CELL_UNICODE_RANGES: list[tuple[int, int]] = [
    (0x20, 0x7E),  # Latin (excluding non-printable)
    (0xA0, 0xAC),
    (0xAE, 0x002FF),
    (0x00370, 0x00482),  # Greek / Cyrillic
    (0x02500, 0x025FC),  # Box drawing, box elements, geometric shapes
    (0x02800, 0x028FF),  # Braille
]

# A set of characters that are a single cell wide
_SINGLE_CELLS = frozenset(
    [
        character
        for _start, _end in _SINGLE_CELL_UNICODE_RANGES
        for character in map(chr, range(_start, _end + 1))
    ]
)

# When called with a string this will return True if all
# characters are single-cell, otherwise False
_is_single_cell_widths: Callable[[str], bool] = _SINGLE_CELLS.issuperset


@lru_cache(4096)
def cached_cell_len(text: str) -> int:
    """Get the number of cells required to display text.

    This method always caches, which may use up a lot of memory. It is recommended to use
    `cell_len` over this method.

    Args:
        text (str): Text to display.

    Returns:
        int: Get the number of cells required to display text.
    """
    if _is_single_cell_widths(text):
        return len(text)
    return sum(map(get_character_cell_size, text))


def cell_len(text: str, _cell_len: Callable[[str], int] = cached_cell_len) -> int:
    """Get the number of cells required to display text.

    Args:
        text (str): Text to display.

    Returns:
        int: Get the number of cells required to display text.
    """
    if len(text) < 512:
        return _cell_len(text)
    if _is_single_cell_widths(text):
        return len(text)
    return sum(map(get_character_cell_size, text))


@lru_cache(maxsize=4096)
def get_character_cell_size(character: str) -> int:
    """Get the cell size of a character.

    Args:
        character (str): A single character.

    Returns:
        int: Number of cells (0, 1 or 2) occupied by that character.
    """
    codepoint = ord(character)
    _table = CELL_WIDTHS
    lower_bound = 0
    upper_bound = len(_table) - 1
    index = (lower_bound + upper_bound) // 2
    while True:
        start, end, width = _table[index]
        if codepoint < start:
            upper_bound = index - 1
        elif codepoint > end:
            lower_bound = index + 1
        else:
            return 0 if width == -1 else width
        if upper_bound < lower_bound:
            break
        index = (lower_bound + upper_bound) // 2
    return 1


def set_cell_size(text: str, total: int) -> str:
    """Set the length of a string to fit within given number of cells."""

    if _is_single_cell_widths(text):
        size = len(text)
        if size < total:
            return text + " " * (total - size)
        return text[:total]

    if total <= 0:
        return ""
    cell_size = cell_len(text)
    if cell_size == total:
        return text
    if cell_size < total:
        return text + " " * (total - cell_size)

    start = 0
    end = len(text)

    # Binary search until we find the right size
    while True:
        pos = (start + end) // 2
        before = text[: pos + 1]
        before_len = cell_len(before)
        if before_len == total + 1 and cell_len(before[-1]) == 2:
            return before[:-1] + " "
        if before_len == total:
            return before
        if before_len > total:
            end = pos
        else:
            start = pos


def chop_cells(
    text: str,
    width: int,
) -> list[str]:
    """Split text into lines such that each line fits within the available (cell) width.

    Args:
        text: The text to fold such that it fits in the given width.
        width: The width available (number of cells).

    Returns:
        A list of strings such that each string in the list has cell width
        less than or equal to the available width.
    """
    _get_character_cell_size = get_character_cell_size
    lines: list[list[str]] = [[]]

    append_new_line = lines.append
    append_to_last_line = lines[-1].append

    total_width = 0

    for character in text:
        cell_width = _get_character_cell_size(character)
        char_doesnt_fit = total_width + cell_width > width

        if char_doesnt_fit:
            append_new_line([character])
            append_to_last_line = lines[-1].append
            total_width = cell_width
        else:
            append_to_last_line(character)
            total_width += cell_width

    return ["".join(line) for line in lines]


if __name__ == "__main__":  # pragma: no cover
    print(get_character_cell_size("😽"))
    for line in chop_cells("""这是对亚洲语言支持的测试。面对模棱两可的想法，拒绝猜测的诱惑。""", 8):
        print(line)
    for n in range(80, 1, -1):
        print(set_cell_size("""这是对亚洲语言支持的测试。面对模棱两可的想法，拒绝猜测的诱惑。""", n) + "|")
        print("x" * n)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\inspection.py ===
# inspection.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""The inspection module provides the :func:`_sa.inspect` function,
which delivers runtime information about a wide variety
of SQLAlchemy objects, both within the Core as well as the
ORM.

The :func:`_sa.inspect` function is the entry point to SQLAlchemy's
public API for viewing the configuration and construction
of in-memory objects.   Depending on the type of object
passed to :func:`_sa.inspect`, the return value will either be
a related object which provides a known interface, or in many
cases it will return the object itself.

The rationale for :func:`_sa.inspect` is twofold.  One is that
it replaces the need to be aware of a large variety of "information
getting" functions in SQLAlchemy, such as
:meth:`_reflection.Inspector.from_engine` (deprecated in 1.4),
:func:`.orm.attributes.instance_state`, :func:`_orm.class_mapper`,
and others.    The other is that the return value of :func:`_sa.inspect`
is guaranteed to obey a documented API, thus allowing third party
tools which build on top of SQLAlchemy configurations to be constructed
in a forwards-compatible way.

"""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Optional
from typing import overload
from typing import Type
from typing import TypeVar
from typing import Union

from . import exc
from .util.typing import Literal
from .util.typing import Protocol

_T = TypeVar("_T", bound=Any)
_TCov = TypeVar("_TCov", bound=Any, covariant=True)
_F = TypeVar("_F", bound=Callable[..., Any])

_IN = TypeVar("_IN", bound=Any)

_registrars: Dict[type, Union[Literal[True], Callable[[Any], Any]]] = {}


class Inspectable(Generic[_T]):
    """define a class as inspectable.

    This allows typing to set up a linkage between an object that
    can be inspected and the type of inspection it returns.

    Unfortunately we cannot at the moment get all classes that are
    returned by inspection to suit this interface as we get into
    MRO issues.

    """

    __slots__ = ()


class _InspectableTypeProtocol(Protocol[_TCov]):
    """a protocol defining a method that's used when a type (ie the class
    itself) is passed to inspect().

    """

    def _sa_inspect_type(self) -> _TCov: ...


class _InspectableProtocol(Protocol[_TCov]):
    """a protocol defining a method that's used when an instance is
    passed to inspect().

    """

    def _sa_inspect_instance(self) -> _TCov: ...


@overload
def inspect(
    subject: Type[_InspectableTypeProtocol[_IN]], raiseerr: bool = True
) -> _IN: ...


@overload
def inspect(
    subject: _InspectableProtocol[_IN], raiseerr: bool = True
) -> _IN: ...


@overload
def inspect(subject: Inspectable[_IN], raiseerr: bool = True) -> _IN: ...


@overload
def inspect(subject: Any, raiseerr: Literal[False] = ...) -> Optional[Any]: ...


@overload
def inspect(subject: Any, raiseerr: bool = True) -> Any: ...


def inspect(subject: Any, raiseerr: bool = True) -> Any:
    """Produce an inspection object for the given target.

    The returned value in some cases may be the
    same object as the one given, such as if a
    :class:`_orm.Mapper` object is passed.   In other
    cases, it will be an instance of the registered
    inspection type for the given object, such as
    if an :class:`_engine.Engine` is passed, an
    :class:`_reflection.Inspector` object is returned.

    :param subject: the subject to be inspected.
    :param raiseerr: When ``True``, if the given subject
     does not
     correspond to a known SQLAlchemy inspected type,
     :class:`sqlalchemy.exc.NoInspectionAvailable`
     is raised.  If ``False``, ``None`` is returned.

    """
    type_ = type(subject)
    for cls in type_.__mro__:
        if cls in _registrars:
            reg = _registrars.get(cls, None)
            if reg is None:
                continue
            elif reg is True:
                return subject
            ret = reg(subject)
            if ret is not None:
                return ret
    else:
        reg = ret = None

    if raiseerr and (reg is None or ret is None):
        raise exc.NoInspectionAvailable(
            "No inspection system is "
            "available for object of type %s" % type_
        )
    return ret


def _inspects(
    *types: Type[Any],
) -> Callable[[_F], _F]:
    def decorate(fn_or_cls: _F) -> _F:
        for type_ in types:
            if type_ in _registrars:
                raise AssertionError("Type %s is already registered" % type_)
            _registrars[type_] = fn_or_cls
        return fn_or_cls

    return decorate


_TT = TypeVar("_TT", bound="Type[Any]")


def _self_inspects(cls: _TT) -> _TT:
    if cls in _registrars:
        raise AssertionError("Type %s is already registered" % cls)
    _registrars[cls] = True
    return cls

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\connectors\aioodbc.py ===
# connectors/aioodbc.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING

from .asyncio import AsyncAdapt_dbapi_connection
from .asyncio import AsyncAdapt_dbapi_cursor
from .asyncio import AsyncAdapt_dbapi_ss_cursor
from .asyncio import AsyncAdaptFallback_dbapi_connection
from .pyodbc import PyODBCConnector
from .. import pool
from .. import util
from ..util.concurrency import await_fallback
from ..util.concurrency import await_only

if TYPE_CHECKING:
    from ..engine.interfaces import ConnectArgsType
    from ..engine.url import URL


class AsyncAdapt_aioodbc_cursor(AsyncAdapt_dbapi_cursor):
    __slots__ = ()

    def setinputsizes(self, *inputsizes):
        # see https://github.com/aio-libs/aioodbc/issues/451
        return self._cursor._impl.setinputsizes(*inputsizes)

        # how it's supposed to work
        # return self.await_(self._cursor.setinputsizes(*inputsizes))


class AsyncAdapt_aioodbc_ss_cursor(
    AsyncAdapt_aioodbc_cursor, AsyncAdapt_dbapi_ss_cursor
):
    __slots__ = ()


class AsyncAdapt_aioodbc_connection(AsyncAdapt_dbapi_connection):
    _cursor_cls = AsyncAdapt_aioodbc_cursor
    _ss_cursor_cls = AsyncAdapt_aioodbc_ss_cursor
    __slots__ = ()

    @property
    def autocommit(self):
        return self._connection.autocommit

    @autocommit.setter
    def autocommit(self, value):
        # https://github.com/aio-libs/aioodbc/issues/448
        # self._connection.autocommit = value

        self._connection._conn.autocommit = value

    def cursor(self, server_side=False):
        # aioodbc sets connection=None when closed and just fails with
        # AttributeError here.  Here we use the same ProgrammingError +
        # message that pyodbc uses, so it triggers is_disconnect() as well.
        if self._connection.closed:
            raise self.dbapi.ProgrammingError(
                "Attempt to use a closed connection."
            )
        return super().cursor(server_side=server_side)

    def rollback(self):
        # aioodbc sets connection=None when closed and just fails with
        # AttributeError here.  should be a no-op
        if not self._connection.closed:
            super().rollback()

    def commit(self):
        # aioodbc sets connection=None when closed and just fails with
        # AttributeError here.  should be a no-op
        if not self._connection.closed:
            super().commit()

    def close(self):
        # aioodbc sets connection=None when closed and just fails with
        # AttributeError here.  should be a no-op
        if not self._connection.closed:
            super().close()


class AsyncAdaptFallback_aioodbc_connection(
    AsyncAdaptFallback_dbapi_connection, AsyncAdapt_aioodbc_connection
):
    __slots__ = ()


class AsyncAdapt_aioodbc_dbapi:
    def __init__(self, aioodbc, pyodbc):
        self.aioodbc = aioodbc
        self.pyodbc = pyodbc
        self.paramstyle = pyodbc.paramstyle
        self._init_dbapi_attributes()
        self.Cursor = AsyncAdapt_dbapi_cursor
        self.version = pyodbc.version

    def _init_dbapi_attributes(self):
        for name in (
            "Warning",
            "Error",
            "InterfaceError",
            "DataError",
            "DatabaseError",
            "OperationalError",
            "InterfaceError",
            "IntegrityError",
            "ProgrammingError",
            "InternalError",
            "NotSupportedError",
            "NUMBER",
            "STRING",
            "DATETIME",
            "BINARY",
            "Binary",
            "BinaryNull",
            "SQL_VARCHAR",
            "SQL_WVARCHAR",
        ):
            setattr(self, name, getattr(self.pyodbc, name))

    def connect(self, *arg, **kw):
        async_fallback = kw.pop("async_fallback", False)
        creator_fn = kw.pop("async_creator_fn", self.aioodbc.connect)

        if util.asbool(async_fallback):
            return AsyncAdaptFallback_aioodbc_connection(
                self,
                await_fallback(creator_fn(*arg, **kw)),
            )
        else:
            return AsyncAdapt_aioodbc_connection(
                self,
                await_only(creator_fn(*arg, **kw)),
            )


class aiodbcConnector(PyODBCConnector):
    is_async = True
    supports_statement_cache = True

    supports_server_side_cursors = True

    @classmethod
    def import_dbapi(cls):
        return AsyncAdapt_aioodbc_dbapi(
            __import__("aioodbc"), __import__("pyodbc")
        )

    def create_connect_args(self, url: URL) -> ConnectArgsType:
        arg, kw = super().create_connect_args(url)
        if arg and arg[0]:
            kw["dsn"] = arg[0]

        return (), kw

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get("async_fallback", False)

        if util.asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def get_driver_connection(self, connection):
        return connection._connection

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\subspaces.py ===
from .utilities import _iszero


def _columnspace(M, simplify=False):
    """Returns a list of vectors (Matrix objects) that span columnspace of ``M``

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.columnspace()
    [Matrix([
    [ 1],
    [-2],
    [ 3]]), Matrix([
    [0],
    [0],
    [6]])]

    See Also
    ========

    nullspace
    rowspace
    """

    reduced, pivots = M.echelon_form(simplify=simplify, with_pivots=True)

    return [M.col(i) for i in pivots]


def _nullspace(M, simplify=False, iszerofunc=_iszero):
    """Returns list of vectors (Matrix objects) that span nullspace of ``M``

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.nullspace()
    [Matrix([
    [-3],
    [ 1],
    [ 0]])]

    See Also
    ========

    columnspace
    rowspace
    """

    reduced, pivots = M.rref(iszerofunc=iszerofunc, simplify=simplify)

    free_vars = [i for i in range(M.cols) if i not in pivots]
    basis     = []

    for free_var in free_vars:
        # for each free variable, we will set it to 1 and all others
        # to 0.  Then, we will use back substitution to solve the system
        vec           = [M.zero] * M.cols
        vec[free_var] = M.one

        for piv_row, piv_col in enumerate(pivots):
            vec[piv_col] -= reduced[piv_row, free_var]

        basis.append(vec)

    return [M._new(M.cols, 1, b) for b in basis]


def _rowspace(M, simplify=False):
    """Returns a list of vectors that span the row space of ``M``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.rowspace()
    [Matrix([[1, 3, 0]]), Matrix([[0, 0, 6]])]
    """

    reduced, pivots = M.echelon_form(simplify=simplify, with_pivots=True)

    return [reduced.row(i) for i in range(len(pivots))]


def _orthogonalize(cls, *vecs, normalize=False, rankcheck=False):
    """Apply the Gram-Schmidt orthogonalization procedure
    to vectors supplied in ``vecs``.

    Parameters
    ==========

    vecs
        vectors to be made orthogonal

    normalize : bool
        If ``True``, return an orthonormal basis.

    rankcheck : bool
        If ``True``, the computation does not stop when encountering
        linearly dependent vectors.

        If ``False``, it will raise ``ValueError`` when any zero
        or linearly dependent vectors are found.

    Returns
    =======

    list
        List of orthogonal (or orthonormal) basis vectors.

    Examples
    ========

    >>> from sympy import I, Matrix
    >>> v = [Matrix([1, I]), Matrix([1, -I])]
    >>> Matrix.orthogonalize(*v)
    [Matrix([
    [1],
    [I]]), Matrix([
    [ 1],
    [-I]])]

    See Also
    ========

    MatrixBase.QRdecomposition

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    """
    from .decompositions import _QRdecomposition_optional

    if not vecs:
        return []

    all_row_vecs = (vecs[0].rows == 1)

    vecs = [x.vec() for x in vecs]
    M = cls.hstack(*vecs)
    Q, R = _QRdecomposition_optional(M, normalize=normalize)

    if rankcheck and Q.cols < len(vecs):
        raise ValueError("GramSchmidt: vector set not linearly independent")

    ret = []
    for i in range(Q.cols):
        if all_row_vecs:
            col = cls(Q[:, i].T)
        else:
            col = cls(Q[:, i])
        ret.append(col)
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_reshape.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

import numpy as np
from fusion_base import Fusion
from onnx import TensorProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionReshape(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "Reshape", "Reshape")
        self.prune_graph: bool = False

    def replace_reshape_node(self, shape, reshape_node, concat_node):
        shape_value = np.asarray(shape, dtype=np.int64)
        constant_shape_name = self.model.create_node_name("Constant", "constant_shape")
        new_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[constant_shape_name],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.INT64,
                dims=shape_value.shape,
                vals=bytes(shape_value),
                raw=True,
            ),
        )
        reshape_node.input[1] = constant_shape_name
        reshape_node.name = self.model.create_node_name("Reshape", "Reshape_Fuse")
        self.nodes_to_remove.extend([concat_node])
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

    def fuse(self, reshape_node, input_name_to_nodes, output_name_to_node):
        if reshape_node.input[1] not in output_name_to_node:
            return

        concat_node = output_name_to_node[reshape_node.input[1]]
        if concat_node.op_type != "Concat" or len(concat_node.input) < 3 or len(concat_node.input) > 4:
            return

        path0 = self.model.match_parent_path(
            concat_node,
            ["Unsqueeze", "Gather", "Shape"],
            [0, 0, 0],
            output_name_to_node,
        )
        if path0 is None:
            return

        (unsqueeze_0, gather_0, shape_0) = path0

        path1 = self.model.match_parent_path(
            concat_node,
            ["Unsqueeze", "Gather", "Shape"],
            [1, 0, 0],
            output_name_to_node,
        )
        if path1 is None:
            return
        (unsqueeze_1, gather_1, shape_1) = path1

        shape = []
        gather_value = self.model.get_constant_value(gather_0.input[1])
        if gather_value == 0:
            shape.append(0)

        gather_value = self.model.get_constant_value(gather_1.input[1])
        if gather_value == 1:
            shape.append(0)

        if len(shape) != 2:
            return

        path2 = []
        path3 = []
        shape_nodes = [shape_0, shape_1]
        if len(concat_node.input) == 3 and self.model.get_constant_value(concat_node.input[2]) is None:
            path2 = self.model.match_parent_path(
                concat_node,
                ["Unsqueeze", "Mul", "Gather", "Shape"],
                [2, 0, 0, 0],
                output_name_to_node,
            )
            if path2 is None:
                path2 = self.model.match_parent_path(
                    concat_node,
                    ["Unsqueeze", "Mul", "Squeeze", "Slice", "Shape"],
                    [2, 0, 0, 0, 0],
                    output_name_to_node,
                )  # GPT2 exported by PyTorch 1.4 with opset_version=11
                if path2 is None:
                    return

            path3 = self.model.match_parent_path(
                concat_node,
                ["Unsqueeze", "Mul", "Gather", "Shape"],
                [2, 0, 1, 0],
                output_name_to_node,
            )
            if path3 is None:
                path3 = self.model.match_parent_path(
                    concat_node,
                    ["Unsqueeze", "Mul", "Squeeze", "Slice", "Shape"],
                    [2, 0, 1, 0, 0],
                    output_name_to_node,
                )  # GPT2 exported by PyTorch 1.4 with opset_version=11
                if path3 is None:
                    return

            shape_nodes.extend([path2[-1], path3[-1]])
            shape.append(-1)
        elif len(concat_node.input) > 2:
            concat_value = self.model.get_constant_value(concat_node.input[2])
            if concat_value is None:
                return
            if isinstance(concat_value, np.ndarray):
                shape.extend(concat_value.tolist())
            else:
                shape.append(concat_value)

        if len(concat_node.input) == 4 and self.model.get_constant_value(concat_node.input[3]) is None:
            if -1 in shape:
                return

            path2 = self.model.match_parent_path(
                concat_node,
                ["Unsqueeze", "Div", "Gather", "Shape"],
                [3, 0, 0, 0],
                output_name_to_node,
            )
            if path2 is None:
                path2 = self.model.match_parent_path(
                    concat_node,
                    ["Unsqueeze", "Div", "Squeeze", "Slice", "Shape"],
                    [3, 0, 0, 0, 0],
                    output_name_to_node,
                )  # GPT2 exported by PyTorch 1.4 with opset_version=11
                if path2 is None:
                    return
            shape_nodes.extend([path2[-1]])
            shape.append(-1)
        elif len(concat_node.input) > 3:
            concat_value = self.model.get_constant_value(concat_node.input[3])
            if concat_value is None:
                return

            if isinstance(concat_value, np.ndarray):
                shape.extend(concat_value.tolist())
            else:
                shape.append(concat_value)

        root_input = reshape_node.input[0]
        same_shape_input = True
        for shape_node in shape_nodes:
            if shape_node.input[0] != root_input:
                same_shape_input = False

        if not same_shape_input:
            return

        self.replace_reshape_node(shape, reshape_node, concat_node)

        # TODO(tlwu): Subgraph blocks pruning un-used nodes. Add code to remove un-used nodes safely.
        self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\floating.py ===
from __future__ import annotations

from typing import ClassVar

import numpy as np

from pandas.core.dtypes.base import register_extension_dtype
from pandas.core.dtypes.common import is_float_dtype

from pandas.core.arrays.numeric import (
    NumericArray,
    NumericDtype,
)


class FloatingDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size of floating dtype.

    These specific implementations are subclasses of the non-public
    FloatingDtype. For example we have Float32Dtype to represent float32.

    The attributes name & type are set when these subclasses are created.
    """

    _default_np_dtype = np.dtype(np.float64)
    _checker = is_float_dtype

    @classmethod
    def construct_array_type(cls) -> type[FloatingArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return FloatingArray

    @classmethod
    def _get_dtype_mapping(cls) -> dict[np.dtype, FloatingDtype]:
        return NUMPY_FLOAT_TO_DTYPE

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        """
        # This is really only here for compatibility with IntegerDtype
        # Here for compat with IntegerDtype
        return values.astype(dtype, copy=copy)


class FloatingArray(NumericArray):
    """
    Array of floating (optional missing) values.

    .. warning::

       FloatingArray is currently experimental, and its API or internal
       implementation may change without warning. Especially the behaviour
       regarding NaN (distinct from NA missing values) is subject to change.

    We represent a FloatingArray with 2 numpy arrays:

    - data: contains a numpy float array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an FloatingArray from generic array-like input, use
    :func:`pandas.array` with one of the float dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d float-dtype array.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values.
    copy : bool, default False
        Whether to copy the `values` and `mask`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    FloatingArray

    Examples
    --------
    Create an FloatingArray with :func:`pandas.array`:

    >>> pd.array([0.1, None, 0.3], dtype=pd.Float32Dtype())
    <FloatingArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([0.1, None, 0.3], dtype="Float32")
    <FloatingArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32
    """

    _dtype_cls = FloatingDtype

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = np.nan
    # Fill values used for any/all
    # Incompatible types in assignment (expression has type "float", base class
    # "BaseMaskedArray" defined the type as "<typing special form>")
    _truthy_value = 1.0  # type: ignore[assignment]
    _falsey_value = 0.0  # type: ignore[assignment]


_dtype_docstring = """
An ExtensionDtype for {dtype} data.

This dtype uses ``pd.NA`` as missing value indicator.

Attributes
----------
None

Methods
-------
None

Examples
--------
For Float32Dtype:

>>> ser = pd.Series([2.25, pd.NA], dtype=pd.Float32Dtype())
>>> ser.dtype
Float32Dtype()

For Float64Dtype:

>>> ser = pd.Series([2.25, pd.NA], dtype=pd.Float64Dtype())
>>> ser.dtype
Float64Dtype()
"""

# create the Dtype


@register_extension_dtype
class Float32Dtype(FloatingDtype):
    type = np.float32
    name: ClassVar[str] = "Float32"
    __doc__ = _dtype_docstring.format(dtype="float32")


@register_extension_dtype
class Float64Dtype(FloatingDtype):
    type = np.float64
    name: ClassVar[str] = "Float64"
    __doc__ = _dtype_docstring.format(dtype="float64")


NUMPY_FLOAT_TO_DTYPE: dict[np.dtype, FloatingDtype] = {
    np.dtype(np.float32): Float32Dtype(),
    np.dtype(np.float64): Float64Dtype(),
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\util\_test_decorators.py ===
"""
This module provides decorator functions which can be applied to test objects
in order to skip those objects when certain conditions occur. A sample use case
is to detect if the platform is missing ``matplotlib``. If so, any test objects
which require ``matplotlib`` and decorated with ``@td.skip_if_no("matplotlib")``
will be skipped by ``pytest`` during the execution of the test suite.

To illustrate, after importing this module:

import pandas.util._test_decorators as td

The decorators can be applied to classes:

@td.skip_if_no("package")
class Foo:
    ...

Or individual functions:

@td.skip_if_no("package")
def test_foo():
    ...

For more information, refer to the ``pytest`` documentation on ``skipif``.
"""
from __future__ import annotations

import locale
from typing import (
    TYPE_CHECKING,
    Callable,
)

import pytest

from pandas._config import get_option

if TYPE_CHECKING:
    from pandas._typing import F

from pandas._config.config import _get_option

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat._optional import import_optional_dependency


def skip_if_installed(package: str) -> pytest.MarkDecorator:
    """
    Skip a test if a package is installed.

    Parameters
    ----------
    package : str
        The name of the package.

    Returns
    -------
    pytest.MarkDecorator
        a pytest.mark.skipif to use as either a test decorator or a
        parametrization mark.
    """
    return pytest.mark.skipif(
        bool(import_optional_dependency(package, errors="ignore")),
        reason=f"Skipping because {package} is installed.",
    )


def skip_if_no(package: str, min_version: str | None = None) -> pytest.MarkDecorator:
    """
    Generic function to help skip tests when required packages are not
    present on the testing system.

    This function returns a pytest mark with a skip condition that will be
    evaluated during test collection. An attempt will be made to import the
    specified ``package`` and optionally ensure it meets the ``min_version``

    The mark can be used as either a decorator for a test class or to be
    applied to parameters in pytest.mark.parametrize calls or parametrized
    fixtures. Use pytest.importorskip if an imported moduled is later needed
    or for test functions.

    If the import and version check are unsuccessful, then the test function
    (or test case when used in conjunction with parametrization) will be
    skipped.

    Parameters
    ----------
    package: str
        The name of the required package.
    min_version: str or None, default None
        Optional minimum version of the package.

    Returns
    -------
    pytest.MarkDecorator
        a pytest.mark.skipif to use as either a test decorator or a
        parametrization mark.
    """
    msg = f"Could not import '{package}'"
    if min_version:
        msg += f" satisfying a min_version of {min_version}"
    return pytest.mark.skipif(
        not bool(
            import_optional_dependency(
                package, errors="ignore", min_version=min_version
            )
        ),
        reason=msg,
    )


skip_if_32bit = pytest.mark.skipif(not IS64, reason="skipping for 32 bit")
skip_if_windows = pytest.mark.skipif(is_platform_windows(), reason="Running on Windows")
skip_if_not_us_locale = pytest.mark.skipif(
    locale.getlocale()[0] != "en_US",
    reason=f"Set local {locale.getlocale()[0]} is not en_US",
)


def parametrize_fixture_doc(*args) -> Callable[[F], F]:
    """
    Intended for use as a decorator for parametrized fixture,
    this function will wrap the decorated function with a pytest
    ``parametrize_fixture_doc`` mark. That mark will format
    initial fixture docstring by replacing placeholders {0}, {1} etc
    with parameters passed as arguments.

    Parameters
    ----------
    args: iterable
        Positional arguments for docstring.

    Returns
    -------
    function
        The decorated function wrapped within a pytest
        ``parametrize_fixture_doc`` mark
    """

    def documented_fixture(fixture):
        fixture.__doc__ = fixture.__doc__.format(*args)
        return fixture

    return documented_fixture


def mark_array_manager_not_yet_implemented(request) -> None:
    mark = pytest.mark.xfail(reason="Not yet implemented for ArrayManager")
    request.applymarker(mark)


skip_array_manager_not_yet_implemented = pytest.mark.xfail(
    _get_option("mode.data_manager", silent=True) == "array",
    reason="Not yet implemented for ArrayManager",
)

skip_array_manager_invalid_test = pytest.mark.skipif(
    _get_option("mode.data_manager", silent=True) == "array",
    reason="Test that relies on BlockManager internals or specific behaviour",
)

skip_copy_on_write_not_yet_implemented = pytest.mark.xfail(
    get_option("mode.copy_on_write") is True,
    reason="Not yet implemented/adapted for Copy-on-Write mode",
)

skip_copy_on_write_invalid_test = pytest.mark.skipif(
    get_option("mode.copy_on_write") is True,
    reason="Test not valid for Copy-on-Write mode",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\index_command.py ===
"""
Contains command classes which may interact with an index / the network.

Unlike its sister module, req_command, this module still uses lazy imports
so commands which don't always hit the network (e.g. list w/o --outdated or
--uptodate) don't need waste time importing PipSession and friends.
"""

import logging
import os
import sys
from functools import lru_cache
from optparse import Values
from typing import TYPE_CHECKING, List, Optional

from pip._vendor import certifi

from pip._internal.cli.base_command import Command
from pip._internal.cli.command_context import CommandContextMixIn

if TYPE_CHECKING:
    from ssl import SSLContext

    from pip._internal.network.session import PipSession

logger = logging.getLogger(__name__)


@lru_cache
def _create_truststore_ssl_context() -> Optional["SSLContext"]:
    if sys.version_info < (3, 10):
        logger.debug("Disabling truststore because Python version isn't 3.10+")
        return None

    try:
        import ssl
    except ImportError:
        logger.warning("Disabling truststore since ssl support is missing")
        return None

    try:
        from pip._vendor import truststore
    except ImportError:
        logger.warning("Disabling truststore because platform isn't supported")
        return None

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(certifi.where())
    return ctx


class SessionCommandMixin(CommandContextMixIn):
    """
    A class mixin for command classes needing _build_session().
    """

    def __init__(self) -> None:
        super().__init__()
        self._session: Optional[PipSession] = None

    @classmethod
    def _get_index_urls(cls, options: Values) -> Optional[List[str]]:
        """Return a list of index urls from user-provided options."""
        index_urls = []
        if not getattr(options, "no_index", False):
            url = getattr(options, "index_url", None)
            if url:
                index_urls.append(url)
        urls = getattr(options, "extra_index_urls", None)
        if urls:
            index_urls.extend(urls)
        # Return None rather than an empty list
        return index_urls or None

    def get_default_session(self, options: Values) -> "PipSession":
        """Get a default-managed session."""
        if self._session is None:
            self._session = self.enter_context(self._build_session(options))
            # there's no type annotation on requests.Session, so it's
            # automatically ContextManager[Any] and self._session becomes Any,
            # then https://github.com/python/mypy/issues/7696 kicks in
            assert self._session is not None
        return self._session

    def _build_session(
        self,
        options: Values,
        retries: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> "PipSession":
        from pip._internal.network.session import PipSession

        cache_dir = options.cache_dir
        assert not cache_dir or os.path.isabs(cache_dir)

        if "legacy-certs" not in options.deprecated_features_enabled:
            ssl_context = _create_truststore_ssl_context()
        else:
            ssl_context = None

        session = PipSession(
            cache=os.path.join(cache_dir, "http-v2") if cache_dir else None,
            retries=retries if retries is not None else options.retries,
            trusted_hosts=options.trusted_hosts,
            index_urls=self._get_index_urls(options),
            ssl_context=ssl_context,
        )

        # Handle custom ca-bundles from the user
        if options.cert:
            session.verify = options.cert

        # Handle SSL client certificate
        if options.client_cert:
            session.cert = options.client_cert

        # Handle timeouts
        if options.timeout or timeout:
            session.timeout = timeout if timeout is not None else options.timeout

        # Handle configured proxies
        if options.proxy:
            session.proxies = {
                "http": options.proxy,
                "https": options.proxy,
            }
            session.trust_env = False
            session.pip_proxy = options.proxy

        # Determine if we can prompt the user for authentication or not
        session.auth.prompting = not options.no_input
        session.auth.keyring_provider = options.keyring_provider

        return session


def _pip_self_version_check(session: "PipSession", options: Values) -> None:
    from pip._internal.self_outdated_check import pip_self_version_check as check

    check(session, options)


class IndexGroupCommand(Command, SessionCommandMixin):
    """
    Abstract base class for commands with the index_group options.

    This also corresponds to the commands that permit the pip version check.
    """

    def handle_pip_version_check(self, options: Values) -> None:
        """
        Do the pip version check if not disabled.

        This overrides the default behavior of not doing the check.
        """
        # Make sure the index_group options are present.
        assert hasattr(options, "no_index")

        if options.disable_pip_version_check or options.no_index:
            return

        try:
            # Otherwise, check if we're using the latest version of pip available.
            session = self._build_session(
                options,
                retries=0,
                timeout=min(5, options.timeout),
            )
            with session:
                _pip_self_version_check(session, options)
        except Exception:
            logger.warning("There was an error checking the latest version of pip.")
            logger.debug("See below for error", exc_info=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\type_d.py ===
from .cartan_type import Standard_Cartan
from sympy.core.backend import eye

class TypeD(Standard_Cartan):

    def __new__(cls, n):
        if n < 3:
            raise ValueError("n cannot be less than 3")
        return Standard_Cartan.__new__(cls, "D", n)


    def dimension(self):
        """Dmension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("D4")
        >>> c.dimension()
        4
        """

        return self.n

    def basic_root(self, i, j):
        """
        This is a method just to generate roots
        with a 1 iin the ith position and a -1
        in the jth position.

        """

        n = self.n
        root = [0]*n
        root[i] = 1
        root[j] = -1
        return root

    def simple_root(self, i):
        """
        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        In D_n, the first n-1 simple roots are the same as
        the roots in A_(n-1) (a 1 in the ith position, a -1
        in the (i+1)th position, and zeroes elsewhere).
        The nth simple root is the root in which there 1s in
        the nth and (n-1)th positions, and zeroes elsewhere.

        This method returns the ith simple root for the D series.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("D4")
        >>> c.simple_root(2)
        [0, 1, -1, 0]

        """

        n = self.n
        if i < n:
            return self.basic_root(i-1, i)
        else:
            root = [0]*n
            root[n-2] = 1
            root[n-1] = 1
            return root


    def positive_roots(self):
        """
        This method generates all the positive roots of
        A_n.  This is half of all of the roots of D_n
        by multiplying all the positive roots by -1 we
        get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
        """

        n = self.n
        posroots = {}
        k = 0
        for i in range(0, n-1):
            for j in range(i+1, n):
                k += 1
                posroots[k] = self.basic_root(i, j)
                k += 1
                root = self.basic_root(i, j)
                root[j] = 1
                posroots[k] = root
        return posroots

    def roots(self):
        """
        Returns the total number of roots for D_n"
        """

        n = self.n
        return 2*n*(n-1)

    def cartan_matrix(self):
        """
        Returns the Cartan matrix for D_n.
        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('D4')
        >>> c.cartan_matrix()
            Matrix([
            [ 2, -1,  0,  0],
            [-1,  2, -1, -1],
            [ 0, -1,  2,  0],
            [ 0, -1,  0,  2]])

        """

        n = self.n
        m = 2*eye(n)
        for i in range(1, n - 2):
            m[i,i+1] = -1
            m[i,i-1] = -1
        m[n-2, n-3] = -1
        m[n-3, n-1] = -1
        m[n-1, n-3] = -1
        m[0, 1] = -1
        return m

    def basis(self):
        """
        Returns the number of independent generators of D_n
        """
        n = self.n
        return n*(n-1)/2

    def lie_algebra(self):
        """
        Returns the Lie algebra associated with D_n"
        """

        n = self.n
        return "so(" + str(2*n) + ")"

    def dynkin_diagram(self):
        n = self.n
        diag = " "*4*(n-3) + str(n-1) + "\n"
        diag += " "*4*(n-3) + "0\n"
        diag += " "*4*(n-3) +"|\n"
        diag += " "*4*(n-3) + "|\n"
        diag += "---".join("0" for i in range(1,n)) + "\n"
        diag += "   ".join(str(i) for i in range(1, n-1)) + "   "+str(n)
        return diag

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\shor.py ===
"""Shor's algorithm and helper functions.

Todo:

* Get the CMod gate working again using the new Gate API.
* Fix everything.
* Update docstrings and reformat.
"""

import math
import random

from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.intfunc import igcd
from sympy.ntheory import continued_fraction_periodic as continued_fraction
from sympy.utilities.iterables import variations

from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import Qubit, measure_partial_oneshot
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qft import QFT
from sympy.physics.quantum.qexpr import QuantumError


class OrderFindingException(QuantumError):
    pass


class CMod(Gate):
    """A controlled mod gate.

    This is black box controlled Mod function for use by shor's algorithm.
    TODO: implement a decompose property that returns how to do this in terms
    of elementary gates
    """

    @classmethod
    def _eval_args(cls, args):
        # t = args[0]
        # a = args[1]
        # N = args[2]
        raise NotImplementedError('The CMod gate has not been completed.')

    @property
    def t(self):
        """Size of 1/2 input register.  First 1/2 holds output."""
        return self.label[0]

    @property
    def a(self):
        """Base of the controlled mod function."""
        return self.label[1]

    @property
    def N(self):
        """N is the type of modular arithmetic we are doing."""
        return self.label[2]

    def _apply_operator_Qubit(self, qubits, **options):
        """
            This directly calculates the controlled mod of the second half of
            the register and puts it in the second
            This will look pretty when we get Tensor Symbolically working
        """
        n = 1
        k = 0
        # Determine the value stored in high memory.
        for i in range(self.t):
            k += n*qubits[self.t + i]
            n *= 2

        # The value to go in low memory will be out.
        out = int(self.a**k % self.N)

        # Create array for new qbit-ket which will have high memory unaffected
        outarray = list(qubits.args[0][:self.t])

        # Place out in low memory
        for i in reversed(range(self.t)):
            outarray.append((out >> i) & 1)

        return Qubit(*outarray)


def shor(N):
    """This function implements Shor's factoring algorithm on the Integer N

    The algorithm starts by picking a random number (a) and seeing if it is
    coprime with N. If it is not, then the gcd of the two numbers is a factor
    and we are done. Otherwise, it begins the period_finding subroutine which
    finds the period of a in modulo N arithmetic. This period, if even, can
    be used to calculate factors by taking a**(r/2)-1 and a**(r/2)+1.
    These values are returned.
    """
    a = random.randrange(N - 2) + 2
    if igcd(N, a) != 1:
        return igcd(N, a)
    r = period_find(a, N)
    if r % 2 == 1:
        shor(N)
    answer = (igcd(a**(r/2) - 1, N), igcd(a**(r/2) + 1, N))
    return answer


def getr(x, y, N):
    fraction = continued_fraction(x, y)
    # Now convert into r
    total = ratioize(fraction, N)
    return total


def ratioize(list, N):
    if list[0] > N:
        return S.Zero
    if len(list) == 1:
        return list[0]
    return list[0] + ratioize(list[1:], N)


def period_find(a, N):
    """Finds the period of a in modulo N arithmetic

    This is quantum part of Shor's algorithm. It takes two registers,
    puts first in superposition of states with Hadamards so: ``|k>|0>``
    with k being all possible choices. It then does a controlled mod and
    a QFT to determine the order of a.
    """
    epsilon = .5
    # picks out t's such that maintains accuracy within epsilon
    t = int(2*math.ceil(log(N, 2)))
    # make the first half of register be 0's |000...000>
    start = [0 for x in range(t)]
    # Put second half into superposition of states so we have |1>x|0> + |2>x|0> + ... |k>x>|0> + ... + |2**n-1>x|0>
    factor = 1/sqrt(2**t)
    qubits = 0
    for arr in variations(range(2), t, repetition=True):
        qbitArray = list(arr) + start
        qubits = qubits + Qubit(*qbitArray)
    circuit = (factor*qubits).expand()
    # Controlled second half of register so that we have:
    # |1>x|a**1 %N> + |2>x|a**2 %N> + ... + |k>x|a**k %N >+ ... + |2**n-1=k>x|a**k % n>
    circuit = CMod(t, a, N)*circuit
    # will measure first half of register giving one of the a**k%N's

    circuit = qapply(circuit)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i)
    # Now apply Inverse Quantum Fourier Transform on the second half of the register

    circuit = qapply(QFT(t, t*2).decompose()*circuit, floatingPoint=True)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i + t)
    if isinstance(circuit, Qubit):
        register = circuit
    elif isinstance(circuit, Mul):
        register = circuit.args[-1]
    else:
        register = circuit.args[-1].args[-1]

    n = 1
    answer = 0
    for i in range(len(register)/2):
        answer += n*register[i + t]
        n = n << 1
    if answer == 0:
        raise OrderFindingException(
            "Order finder returned 0. Happens with chance %f" % epsilon)
    #turn answer into r using continued fractions
    g = getr(answer, 2**t, N)
    return g

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\fields\simple.py ===
from .. import widgets
from .core import Field

__all__ = (
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
)


class BooleanField(Field):
    """
    Represents an ``<input type="checkbox">``. Set the ``checked``-status by using the
    ``default``-option. Any value for ``default``, e.g. ``default="checked"`` puts
    ``checked`` into the html-element and sets the ``data`` to ``True``

    :param false_values:
        If provided, a sequence of strings each of which is an exact match
        string of what is considered a "false" value. Defaults to the tuple
        ``(False, "false", "")``
    """

    widget = widgets.CheckboxInput()
    false_values = (False, "false", "")

    def __init__(self, label=None, validators=None, false_values=None, **kwargs):
        super().__init__(label, validators, **kwargs)
        if false_values is not None:
            self.false_values = false_values

    def process_data(self, value):
        self.data = bool(value)

    def process_formdata(self, valuelist):
        if not valuelist or valuelist[0] in self.false_values:
            self.data = False
        else:
            self.data = True

    def _value(self):
        if self.raw_data:
            return str(self.raw_data[0])
        return "y"


class StringField(Field):
    """
    This field is the base for most of the more complicated fields, and
    represents an ``<input type="text">``.
    """

    widget = widgets.TextInput()

    def process_formdata(self, valuelist):
        if valuelist:
            self.data = valuelist[0]

    def _value(self):
        return str(self.data) if self.data is not None else ""


class TextAreaField(StringField):
    """
    This field represents an HTML ``<textarea>`` and can be used to take
    multi-line input.
    """

    widget = widgets.TextArea()


class PasswordField(StringField):
    """
    A StringField, except renders an ``<input type="password">``.

    Also, whatever value is accepted by this field is not rendered back
    to the browser like normal fields.
    """

    widget = widgets.PasswordInput()


class FileField(Field):
    """Renders a file upload field.

    By default, the value will be the filename sent in the form data.
    WTForms **does not** deal with frameworks' file handling capabilities.
    A WTForms extension for a framework may replace the filename value
    with an object representing the uploaded data.
    """

    widget = widgets.FileInput()

    def _value(self):
        # browser ignores value of file input for security
        return False


class MultipleFileField(FileField):
    """A :class:`FileField` that allows choosing multiple files."""

    widget = widgets.FileInput(multiple=True)

    def process_formdata(self, valuelist):
        self.data = valuelist


class HiddenField(StringField):
    """
    HiddenField is a convenience for a StringField with a HiddenInput widget.

    It will render as an ``<input type="hidden">`` but otherwise coerce to a string.
    """

    widget = widgets.HiddenInput()


class SubmitField(BooleanField):
    """
    Represents an ``<input type="submit">``.  This allows checking if a given
    submit button has been pressed.
    """

    widget = widgets.SubmitInput()


class SearchField(StringField):
    """
    Represents an ``<input type="search">``.
    """

    widget = widgets.SearchInput()


class TelField(StringField):
    """
    Represents an ``<input type="tel">``.
    """

    widget = widgets.TelInput()


class URLField(StringField):
    """
    Represents an ``<input type="url">``.
    """

    widget = widgets.URLInput()


class EmailField(StringField):
    """
    Represents an ``<input type="email">``.
    """

    widget = widgets.EmailInput()


class ColorField(StringField):
    """
    Represents an ``<input type="color">``.
    """

    widget = widgets.ColorInput()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\descriptor_database.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Provides a container for DescriptorProtos."""

__author__ = 'matthewtoia@google.com (Matt Toia)'

import warnings


class Error(Exception):
  pass


class DescriptorDatabaseConflictingDefinitionError(Error):
  """Raised when a proto is added with the same name & different descriptor."""


class DescriptorDatabase(object):
  """A container accepting FileDescriptorProtos and maps DescriptorProtos."""

  def __init__(self):
    self._file_desc_protos_by_file = {}
    self._file_desc_protos_by_symbol = {}

  def Add(self, file_desc_proto):
    """Adds the FileDescriptorProto and its types to this database.

    Args:
      file_desc_proto: The FileDescriptorProto to add.
    Raises:
      DescriptorDatabaseConflictingDefinitionError: if an attempt is made to
        add a proto with the same name but different definition than an
        existing proto in the database.
    """
    proto_name = file_desc_proto.name
    if proto_name not in self._file_desc_protos_by_file:
      self._file_desc_protos_by_file[proto_name] = file_desc_proto
    elif self._file_desc_protos_by_file[proto_name] != file_desc_proto:
      raise DescriptorDatabaseConflictingDefinitionError(
          '%s already added, but with different descriptor.' % proto_name)
    else:
      return

    # Add all the top-level descriptors to the index.
    package = file_desc_proto.package
    for message in file_desc_proto.message_type:
      for name in _ExtractSymbols(message, package):
        self._AddSymbol(name, file_desc_proto)
    for enum in file_desc_proto.enum_type:
      self._AddSymbol(
          ('.'.join((package, enum.name)) if package else enum.name),
          file_desc_proto,
      )
      for enum_value in enum.value:
        self._file_desc_protos_by_symbol[
            '.'.join((package, enum_value.name)) if package else enum_value.name
        ] = file_desc_proto
    for extension in file_desc_proto.extension:
      self._AddSymbol(
          ('.'.join((package, extension.name)) if package else extension.name),
          file_desc_proto,
      )
    for service in file_desc_proto.service:
      self._AddSymbol(
          ('.'.join((package, service.name)) if package else service.name),
          file_desc_proto,
      )

  def FindFileByName(self, name):
    """Finds the file descriptor proto by file name.

    Typically the file name is a relative path ending to a .proto file. The
    proto with the given name will have to have been added to this database
    using the Add method or else an error will be raised.

    Args:
      name: The file name to find.

    Returns:
      The file descriptor proto matching the name.

    Raises:
      KeyError if no file by the given name was added.
    """

    return self._file_desc_protos_by_file[name]

  def FindFileContainingSymbol(self, symbol):
    """Finds the file descriptor proto containing the specified symbol.

    The symbol should be a fully qualified name including the file descriptor's
    package and any containing messages. Some examples:

    'some.package.name.Message'
    'some.package.name.Message.NestedEnum'
    'some.package.name.Message.some_field'

    The file descriptor proto containing the specified symbol must be added to
    this database using the Add method or else an error will be raised.

    Args:
      symbol: The fully qualified symbol name.

    Returns:
      The file descriptor proto containing the symbol.

    Raises:
      KeyError if no file contains the specified symbol.
    """
    if symbol.count('.') == 1 and symbol[0] == '.':
      symbol = symbol.lstrip('.')
      warnings.warn(
          'Please remove the leading "." when '
          'FindFileContainingSymbol, this will turn to error '
          'in 2026 Jan.',
          RuntimeWarning,
      )
    try:
      return self._file_desc_protos_by_symbol[symbol]
    except KeyError:
      # Fields, enum values, and nested extensions are not in
      # _file_desc_protos_by_symbol. Try to find the top level
      # descriptor. Non-existent nested symbol under a valid top level
      # descriptor can also be found. The behavior is the same with
      # protobuf C++.
      top_level, _, _ = symbol.rpartition('.')
      try:
        return self._file_desc_protos_by_symbol[top_level]
      except KeyError:
        # Raise the original symbol as a KeyError for better diagnostics.
        raise KeyError(symbol)

  def FindFileContainingExtension(self, extendee_name, extension_number):
    # TODO: implement this API.
    return None

  def FindAllExtensionNumbers(self, extendee_name):
    # TODO: implement this API.
    return []

  def _AddSymbol(self, name, file_desc_proto):
    if name in self._file_desc_protos_by_symbol:
      warn_msg = ('Conflict register for file "' + file_desc_proto.name +
                  '": ' + name +
                  ' is already defined in file "' +
                  self._file_desc_protos_by_symbol[name].name + '"')
      warnings.warn(warn_msg, RuntimeWarning)
    self._file_desc_protos_by_symbol[name] = file_desc_proto


def _ExtractSymbols(desc_proto, package):
  """Pulls out all the symbols from a descriptor proto.

  Args:
    desc_proto: The proto to extract symbols from.
    package: The package containing the descriptor type.

  Yields:
    The fully qualified name found in the descriptor.
  """
  message_name = package + '.' + desc_proto.name if package else desc_proto.name
  yield message_name
  for nested_type in desc_proto.nested_type:
    for symbol in _ExtractSymbols(nested_type, message_name):
      yield symbol
  for enum_type in desc_proto.enum_type:
    yield '.'.join((message_name, enum_type.name))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\gemm.py ===
import logging

import numpy as np  # noqa: F401
import onnx

from ..quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizedValue,
    QuantizedValueType,
    attribute_to_kwarg,
    find_by_name,  # noqa: F401
    get_mul_node,  # noqa: F401
    ms_domain,
)
from .base_operator import QuantOperatorBase  # noqa: F401
from .matmul import QOpMatMul
from .qdq_base_operator import QDQOperatorBase


def is_B_transposed(gemm_node):  # noqa: N802
    transB_attribute = [attr for attr in gemm_node.attribute if attr.name == "transB"]  # noqa: N806
    if len(transB_attribute):
        return onnx.helper.get_attribute_value(transB_attribute[0]) > 0

    return False


def get_beta(gemm_node):
    beta_attribute = [attr for attr in gemm_node.attribute if attr.name == "beta"]
    if len(beta_attribute):
        return onnx.helper.get_attribute_value(beta_attribute[0])

    return 1.0


def set_default_beta(gemm_node):
    beta_attribute = [attr for attr in gemm_node.attribute if attr.name == "beta"]
    if len(beta_attribute):
        beta_attribute[0].f = 1.0

    return 1.0


class QLinearGemm(QOpMatMul):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Gemm"

        (
            data_found,
            output_scale_name,
            output_zp_name,
            _,
            _,
        ) = self.quantizer._get_quantization_params(node.output[0])

        if self.quantizer.is_input_a_initializer(node.input[1]) and self.quantizer.is_per_channel():
            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_activation(node, [0])
            quant_weight_tuple = self.quantizer.quantize_weight_per_channel(
                node.input[1],
                self.quantizer.weight_qType,
                0 if is_B_transposed(node) else 1,
            )
            quantized_input_names.append(quant_weight_tuple[0])
            zero_point_names.append(quant_weight_tuple[1])
            scale_names.append(quant_weight_tuple[2])
        else:
            #  Get Quantized from both activation(input[0]) and weight(input[1])
            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_activation(node, [0])

            (
                quantized_input_names_weight,
                zero_point_names_weight,
                scale_names_weight,
                nodes_weight,
            ) = self.quantizer.quantize_weight(node, [1], reduce_range=self.quantizer.reduce_range)
            quantized_input_names.extend(quantized_input_names_weight)
            zero_point_names.extend(zero_point_names_weight)
            scale_names.extend(scale_names_weight)
            nodes.extend(nodes_weight)

        if not data_found or quantized_input_names is None:
            return super().quantize()

        quantized_bias_name = ""
        if len(node.input) == 3:
            if not self.quantizer.is_input_a_initializer(node.input[2]):
                return super().quantize()

            # Note: if the quantized type is float 8, the bias is converted into float 16.
            # cublasLtMatMul only supports (b)float16 or float32 bias.
            quantized_bias_name = self.quantizer.quantize_bias_static(
                node.input[2], node.input[0], node.input[1], get_beta(self.node)
            )

        qgemm_output = node.output[0] + TENSOR_NAME_QUANT_SUFFIX
        qgemm_name = node.name + "_quant" if node.name else ""

        kwargs = {}
        for attribute in node.attribute:
            if attribute.name != "beta":
                kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        # generate input
        qgemm_inputs = []
        for i in range(2):
            qgemm_inputs.extend([quantized_input_names[i], scale_names[i], zero_point_names[i]])

        qgemm_inputs.extend([quantized_bias_name, output_scale_name, output_zp_name])

        qgemm_node = onnx.helper.make_node("QGemm", qgemm_inputs, [qgemm_output], qgemm_name, **kwargs)
        nodes.append(qgemm_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(
            node.output[0],
            qgemm_output,
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
            node_type=node.op_type,
            node_qtype=self.quantizer.weight_qType,
        )
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        self.quantizer.new_nodes += nodes


class QDQGemm(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Gemm"

        self.quantizer.quantize_activation_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_activation_tensor(node.output[0])

        is_weight_per_channel, weight_axis = self.quantizer.is_tensor_per_channel(
            node.input[1], default_axis=0 if is_B_transposed(node) else 1
        )
        if is_weight_per_channel:
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], weight_axis)
        else:
            self.quantizer.quantize_weight_tensor(node.input[1])

        if len(node.input) == 3:
            if self.quantizer.is_input_a_initializer(node.input[2]):
                self.quantizer.quantize_bias_tensor(
                    node.name, node.input[2], node.input[0], node.input[1], get_beta(self.node)
                )
                set_default_beta(self.node)
            else:
                logging.warning(
                    f"Bias of Gemm node '{self.node.name}' is not constant. Please exclude this node for better performance."
                )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\pad.py ===
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import onnx

from ..quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizedValue,
    QuantizedValueType,
    attribute_to_kwarg,
    quantize_nparray,
)
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase


class QPad(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Pad"

        # Only after version 11, it has the optional constant_value
        # If input[0] is not quantized, do not quanitize this node
        if (self.quantizer.opset_version < 11) or (node.input[0] not in self.quantizer.quantized_value_map):
            super().quantize()
            return
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]

        kwargs = {}
        for attribute in node.attribute:
            kv = attribute_to_kwarg(attribute)
            kwargs.update(kv)

        if "mode" not in kwargs or kwargs["mode"] == b"constant":
            if len(node.input) > 2 and node.input[2] != "":  # There is 3rd input 'constant_value'
                zp_tensor = self.quantizer.model.get_initializer(quantized_input_value.zp_name)
                scale_tensor = self.quantizer.model.get_initializer(quantized_input_value.scale_name)
                if zp_tensor is None or scale_tensor is None:
                    super().quantize()
                    return

                padding_constant_initializer = self.quantizer.model.get_initializer(node.input[2])
                if padding_constant_initializer is not None:
                    zp_array = onnx.numpy_helper.to_array(zp_tensor)
                    zp_value = zp_array.item() if zp_array.ndim == 0 else zp_array[0]
                    scale_array = onnx.numpy_helper.to_array(scale_tensor)
                    scale_value = scale_array.item() if scale_array.ndim == 0 else scale_array[0]
                    padding_constant_array = onnx.numpy_helper.to_array(padding_constant_initializer)
                    quantized_padding_constant_array = quantize_nparray(
                        self.quantizer.activation_qType,
                        padding_constant_array,
                        scale_value,
                        zp_value,
                    )
                    quantized_padding_constant_name = node.input[2] + TENSOR_NAME_QUANT_SUFFIX
                    quantized_padding_constant_initializer = onnx.numpy_helper.from_array(
                        quantized_padding_constant_array,
                        quantized_padding_constant_name,
                    )
                    # Suppose this padding constant initializer only used by the node
                    self.quantizer.model.remove_initializer(padding_constant_initializer)
                    self.quantizer.model.add_initializer(quantized_padding_constant_initializer)
                    node.input[2] = quantized_padding_constant_name
                else:
                    # TODO: check quantize_inputs after sub graph is supported
                    pad_value_qnodes = self.quantizer._get_quantize_input_nodes(
                        node,
                        2,
                        self.quantizer.activation_qType,
                        quantized_input_value.scale_name,
                        quantized_input_value.zp_name,
                        initial_type=scale_tensor.data_type,
                    )
                    self.quantizer.new_nodes.extend(pad_value_qnodes)
                    node.input[2] = pad_value_qnodes[0].output[0]
            else:
                # In quantized format, the `zero` before quantization is mapped
                # to quantized_input_value.zp_name. Thus, padding 0 to
                # original tensor should become padding zero point to quantized
                # tensor.
                if len(node.input) == 2:
                    # Feed quantization's zero point to padding node.
                    node.input.append(quantized_input_value.zp_name)
                else:
                    # Assign quantization's zero point to padding node.
                    assert node.input[2] == ""
                    node.input[2] = quantized_input_value.zp_name

        # Create an entry for output quantized value
        quantized_output_value = QuantizedValue(
            node.output[0],
            node.output[0] + TENSOR_NAME_QUANT_SUFFIX,
            quantized_input_value.scale_name,
            quantized_input_value.zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        node.input[0] = quantized_input_value.q_name
        node.output[0] = quantized_output_value.q_name
        self.quantizer.new_nodes += [node]


class QDQPad(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def _get_pad_const_val(self, attrs_dict: dict[str, Any]) -> np.ndarray | None:
        """
        Returns the Pad's constant padding value. Returns `None` if the padding value is
        not constant (i.e., comes from a dynamic input).
        """
        const_val = None
        onnx_tensor_type = self.quantizer.model.get_tensor_type(self.node.input[0])
        if onnx_tensor_type is None:
            return None

        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_tensor_type.elem_type)
        if self.quantizer.opset_version < 11:
            const_val = np.array(attrs_dict.get("value", 0), dtype=np_dtype)
        elif len(self.node.input) >= 3 and self.node.input[2]:
            const_val = self.quantizer.model.get_constant_value(self.node.input[2])
        else:
            const_val = np.array(0, dtype=np_dtype)

        return const_val

    def _should_quantize_output_same_as_input(self) -> bool:
        """
        Returns true if Pad's output should use the same quantization parameters as input[0]
        """
        attrs_dict = {}
        for attribute in self.node.attribute:
            kv = attribute_to_kwarg(attribute)
            attrs_dict.update(kv)

        pad_mode = attrs_dict.get("mode", b"constant")
        if pad_mode in (b"reflect", b"edge", b"wrap"):
            # These modes pad the output with a value that already exists in the input.
            # So, we can quantize the output the same as the input.
            return True

        # For 'constant' mode, if padding with 0, we can also quantize the output the same as the input
        # because our quantization floating-point range always includes 0.
        if pad_mode == b"constant":
            pad_val = self._get_pad_const_val(attrs_dict)
            if pad_val is not None and pad_val.dtype in (np.float32, np.float16):
                return float(pad_val.item()) == 0

        return False

    def quantize(self):
        assert self.node.op_type == "Pad"

        for input_name in self.node.input:
            if input_name:
                self.quantizer.quantize_activation_tensor(input_name)

        if not self.disable_qdq_for_node_output:
            if self._should_quantize_output_same_as_input():
                self.quantizer.quantize_output_same_as_input(self.node.output[0], self.node.input[0], self.node.name)
            else:
                self.quantizer.quantize_activation_tensor(self.node.output[0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\colors.py ===
# Copyright (c) 2010-2024 openpyxl

import re
from openpyxl.compat import safe_string
from openpyxl.descriptors import (
    String,
    Bool,
    MinMax,
    Integer,
    Typed,
)
from openpyxl.descriptors.sequence import NestedSequence
from openpyxl.descriptors.serialisable import Serialisable

# Default Color Index as per 18.8.27 of ECMA Part 4
COLOR_INDEX = (
    '00000000', '00FFFFFF', '00FF0000', '0000FF00', '000000FF', #0-4
    '00FFFF00', '00FF00FF', '0000FFFF', '00000000', '00FFFFFF', #5-9
    '00FF0000', '0000FF00', '000000FF', '00FFFF00', '00FF00FF', #10-14
    '0000FFFF', '00800000', '00008000', '00000080', '00808000', #15-19
    '00800080', '00008080', '00C0C0C0', '00808080', '009999FF', #20-24
    '00993366', '00FFFFCC', '00CCFFFF', '00660066', '00FF8080', #25-29
    '000066CC', '00CCCCFF', '00000080', '00FF00FF', '00FFFF00', #30-34
    '0000FFFF', '00800080', '00800000', '00008080', '000000FF', #35-39
    '0000CCFF', '00CCFFFF', '00CCFFCC', '00FFFF99', '0099CCFF', #40-44
    '00FF99CC', '00CC99FF', '00FFCC99', '003366FF', '0033CCCC', #45-49
    '0099CC00', '00FFCC00', '00FF9900', '00FF6600', '00666699', #50-54
    '00969696', '00003366', '00339966', '00003300', '00333300', #55-59
    '00993300', '00993366', '00333399', '00333333',  #60-63
)
# indices 64 and 65 are reserved for the system foreground and background colours respectively

# Will remove these definitions in a future release
BLACK = COLOR_INDEX[0]
WHITE = COLOR_INDEX[1]
#RED = COLOR_INDEX[2]
#DARKRED = COLOR_INDEX[8]
BLUE = COLOR_INDEX[4]
#DARKBLUE = COLOR_INDEX[12]
#GREEN = COLOR_INDEX[3]
#DARKGREEN = COLOR_INDEX[9]
#YELLOW = COLOR_INDEX[5]
#DARKYELLOW = COLOR_INDEX[19]


aRGB_REGEX = re.compile("^([A-Fa-f0-9]{8}|[A-Fa-f0-9]{6})$")


class RGB(Typed):
    """
    Descriptor for aRGB values
    If not supplied alpha is 00
    """

    expected_type = str

    def __set__(self, instance, value):
        if not self.allow_none:
            m = aRGB_REGEX.match(value)
            if m is None:
                raise ValueError("Colors must be aRGB hex values")
            if len(value) == 6:
                value = "00" + value
        super().__set__(instance, value)


class Color(Serialisable):
    """Named colors for use in styles."""

    tagname = "color"

    rgb = RGB()
    indexed = Integer()
    auto = Bool()
    theme = Integer()
    tint = MinMax(min=-1, max=1, expected_type=float)
    type = String()


    def __init__(self, rgb=BLACK, indexed=None, auto=None, theme=None, tint=0.0, index=None, type='rgb'):
        if index is not None:
            indexed = index
        if indexed is not None:
            self.type = 'indexed'
            self.indexed = indexed
        elif theme is not None:
            self.type = 'theme'
            self.theme = theme
        elif auto is not None:
            self.type = 'auto'
            self.auto = auto
        else:
            self.rgb = rgb
            self.type = 'rgb'
        self.tint = tint

    @property
    def value(self):
        return getattr(self, self.type)

    @value.setter
    def value(self, value):
        setattr(self, self.type, value)

    def __iter__(self):
        attrs = [(self.type, self.value)]
        if self.tint != 0:
            attrs.append(('tint', self.tint))
        for k, v in attrs:
            yield k, safe_string(v)

    @property
    def index(self):
        # legacy
        return self.value


    def __add__(self, other):
        """
        Adding colours is undefined behaviour best do nothing
        """
        if not isinstance(other, Color):
            return super().__add__(other)
        return self


class ColorDescriptor(Typed):

    expected_type = Color

    def __set__(self, instance, value):
        if isinstance(value, str):
            value = Color(rgb=value)
        super().__set__(instance, value)


class RgbColor(Serialisable):

    tagname = "rgbColor"

    rgb = RGB()

    def __init__(self,
                 rgb=None,
                ):
        self.rgb = rgb


class ColorList(Serialisable):

    tagname = "colors"

    indexedColors = NestedSequence(expected_type=RgbColor)
    mruColors = NestedSequence(expected_type=Color)

    __elements__ = ('indexedColors', 'mruColors')

    def __init__(self,
                 indexedColors=(),
                 mruColors=(),
                ):
        self.indexedColors = indexedColors
        self.mruColors = mruColors


    def __bool__(self):
        return bool(self.indexedColors) or bool(self.mruColors)


    @property
    def index(self):
        return [val.rgb for val in self.indexedColors]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\extension.py ===
"""
Shared methods for Index subclasses backed by ExtensionArray.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    TypeVar,
)

from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.generic import ABCDataFrame

from pandas.core.indexes.base import Index

if TYPE_CHECKING:
    import numpy as np

    from pandas._typing import (
        ArrayLike,
        npt,
    )

    from pandas.core.arrays import IntervalArray
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

_ExtensionIndexT = TypeVar("_ExtensionIndexT", bound="ExtensionIndex")


def _inherit_from_data(
    name: str, delegate: type, cache: bool = False, wrap: bool = False
):
    """
    Make an alias for a method of the underlying ExtensionArray.

    Parameters
    ----------
    name : str
        Name of an attribute the class should inherit from its EA parent.
    delegate : class
    cache : bool, default False
        Whether to convert wrapped properties into cache_readonly
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.

    Returns
    -------
    attribute, method, property, or cache_readonly
    """
    attr = getattr(delegate, name)

    if isinstance(attr, property) or type(attr).__name__ == "getset_descriptor":
        # getset_descriptor i.e. property defined in cython class
        if cache:

            def cached(self):
                return getattr(self._data, name)

            cached.__name__ = name
            cached.__doc__ = attr.__doc__
            method = cache_readonly(cached)

        else:

            def fget(self):
                result = getattr(self._data, name)
                if wrap:
                    if isinstance(result, type(self._data)):
                        return type(self)._simple_new(result, name=self.name)
                    elif isinstance(result, ABCDataFrame):
                        return result.set_index(self)
                    return Index(result, name=self.name, dtype=result.dtype)
                return result

            def fset(self, value) -> None:
                setattr(self._data, name, value)

            fget.__name__ = name
            fget.__doc__ = attr.__doc__

            method = property(fget, fset)

    elif not callable(attr):
        # just a normal attribute, no wrapping
        method = attr

    else:
        # error: Incompatible redefinition (redefinition with type "Callable[[Any,
        # VarArg(Any), KwArg(Any)], Any]", original type "property")
        def method(self, *args, **kwargs):  # type: ignore[misc]
            if "inplace" in kwargs:
                raise ValueError(f"cannot use inplace with {type(self).__name__}")
            result = attr(self._data, *args, **kwargs)
            if wrap:
                if isinstance(result, type(self._data)):
                    return type(self)._simple_new(result, name=self.name)
                elif isinstance(result, ABCDataFrame):
                    return result.set_index(self)
                return Index(result, name=self.name, dtype=result.dtype)
            return result

        # error: "property" has no attribute "__name__"
        method.__name__ = name  # type: ignore[attr-defined]
        method.__doc__ = attr.__doc__
    return method


def inherit_names(
    names: list[str], delegate: type, cache: bool = False, wrap: bool = False
) -> Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]:
    """
    Class decorator to pin attributes from an ExtensionArray to a Index subclass.

    Parameters
    ----------
    names : List[str]
    delegate : class
    cache : bool, default False
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.
    """

    def wrapper(cls: type[_ExtensionIndexT]) -> type[_ExtensionIndexT]:
        for name in names:
            meth = _inherit_from_data(name, delegate, cache=cache, wrap=wrap)
            setattr(cls, name, meth)

        return cls

    return wrapper


class ExtensionIndex(Index):
    """
    Index subclass for indexes backed by ExtensionArray.
    """

    # The base class already passes through to _data:
    #  size, __len__, dtype

    _data: IntervalArray | NDArrayBackedExtensionArray

    # ---------------------------------------------------------------------

    def _validate_fill_value(self, value):
        """
        Convert value to be insertable to underlying array.
        """
        return self._data._validate_setitem_value(value)

    @cache_readonly
    def _isnan(self) -> npt.NDArray[np.bool_]:
        # error: Incompatible return value type (got "ExtensionArray", expected
        # "ndarray")
        return self._data.isna()  # type: ignore[return-value]


class NDArrayBackedExtensionIndex(ExtensionIndex):
    """
    Index subclass for indexes backed by NDArrayBackedExtensionArray.
    """

    _data: NDArrayBackedExtensionArray

    def _get_engine_target(self) -> np.ndarray:
        return self._data._ndarray

    def _from_join_target(self, result: np.ndarray) -> ArrayLike:
        assert result.dtype == self._data._ndarray.dtype
        return self._data._from_backing_data(result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_config\localization.py ===
"""
Helpers for configuring locale settings.

Name `localization` is chosen to avoid overlap with builtin `locale` module.
"""
from __future__ import annotations

from contextlib import contextmanager
import locale
import platform
import re
import subprocess
from typing import TYPE_CHECKING

from pandas._config.config import options

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def set_locale(
    new_locale: str | tuple[str, str], lc_var: int = locale.LC_ALL
) -> Generator[str | tuple[str, str], None, None]:
    """
    Context manager for temporarily setting a locale.

    Parameters
    ----------
    new_locale : str or tuple
        A string of the form <language_country>.<encoding>. For example to set
        the current locale to US English with a UTF8 encoding, you would pass
        "en_US.UTF-8".
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Notes
    -----
    This is useful when you want to run a particular block of code under a
    particular locale, without globally setting the locale. This probably isn't
    thread-safe.
    """
    # getlocale is not always compliant with setlocale, use setlocale. GH#46595
    current_locale = locale.setlocale(lc_var)

    try:
        locale.setlocale(lc_var, new_locale)
        normalized_code, normalized_encoding = locale.getlocale()
        if normalized_code is not None and normalized_encoding is not None:
            yield f"{normalized_code}.{normalized_encoding}"
        else:
            yield new_locale
    finally:
        locale.setlocale(lc_var, current_locale)


def can_set_locale(lc: str, lc_var: int = locale.LC_ALL) -> bool:
    """
    Check to see if we can set a locale, and subsequently get the locale,
    without raising an Exception.

    Parameters
    ----------
    lc : str
        The locale to attempt to set.
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Returns
    -------
    bool
        Whether the passed locale can be set
    """
    try:
        with set_locale(lc, lc_var=lc_var):
            pass
    except (ValueError, locale.Error):
        # horrible name for a Exception subclass
        return False
    else:
        return True


def _valid_locales(locales: list[str] | str, normalize: bool) -> list[str]:
    """
    Return a list of normalized locales that do not throw an ``Exception``
    when set.

    Parameters
    ----------
    locales : str
        A string where each locale is separated by a newline.
    normalize : bool
        Whether to call ``locale.normalize`` on each locale.

    Returns
    -------
    valid_locales : list
        A list of valid locales.
    """
    return [
        loc
        for loc in (
            locale.normalize(loc.strip()) if normalize else loc.strip()
            for loc in locales
        )
        if can_set_locale(loc)
    ]


def get_locales(
    prefix: str | None = None,
    normalize: bool = True,
) -> list[str]:
    """
    Get all the locales that are available on the system.

    Parameters
    ----------
    prefix : str
        If not ``None`` then return only those locales with the prefix
        provided. For example to get all English language locales (those that
        start with ``"en"``), pass ``prefix="en"``.
    normalize : bool
        Call ``locale.normalize`` on the resulting list of available locales.
        If ``True``, only locales that can be set without throwing an
        ``Exception`` are returned.

    Returns
    -------
    locales : list of strings
        A list of locale strings that can be set with ``locale.setlocale()``.
        For example::

            locale.setlocale(locale.LC_ALL, locale_string)

    On error will return an empty list (no locale available, e.g. Windows)

    """
    if platform.system() in ("Linux", "Darwin"):
        raw_locales = subprocess.check_output(["locale", "-a"])
    else:
        # Other platforms e.g. windows platforms don't define "locale -a"
        #  Note: is_platform_windows causes circular import here
        return []

    try:
        # raw_locales is "\n" separated list of locales
        # it may contain non-decodable parts, so split
        # extract what we can and then rejoin.
        split_raw_locales = raw_locales.split(b"\n")
        out_locales = []
        for x in split_raw_locales:
            try:
                out_locales.append(str(x, encoding=options.display.encoding))
            except UnicodeError:
                # 'locale -a' is used to populated 'raw_locales' and on
                # Redhat 7 Linux (and maybe others) prints locale names
                # using windows-1252 encoding.  Bug only triggered by
                # a few special characters and when there is an
                # extensive list of installed locales.
                out_locales.append(str(x, encoding="windows-1252"))

    except TypeError:
        pass

    if prefix is None:
        return _valid_locales(out_locales, normalize)

    pattern = re.compile(f"{prefix}.*")
    found = pattern.findall("\n".join(out_locales))
    return _valid_locales(found, normalize)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\locations\_distutils.py ===
"""Locations where we look for configs, install stuff, etc"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False

# If pip's going to use distutils, it should not be using the copy that setuptools
# might have injected into the environment. This is done by removing the injected
# shim, if it's injected.
#
# See https://github.com/pypa/pip/issues/8761 for the original discussion and
# rationale for why this is done within pip.
try:
    __import__("_distutils_hack").remove_shim()
except (ImportError, AttributeError):
    pass

import logging
import os
import sys
from distutils.cmd import Command as DistutilsCommand
from distutils.command.install import SCHEME_KEYS
from distutils.command.install import install as distutils_install_command
from distutils.sysconfig import get_python_lib
from typing import Dict, List, Optional, Union

from pip._internal.models.scheme import Scheme
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.virtualenv import running_under_virtualenv

from .base import get_major_minor_version

logger = logging.getLogger(__name__)


def distutils_scheme(
    dist_name: str,
    user: bool = False,
    home: Optional[str] = None,
    root: Optional[str] = None,
    isolated: bool = False,
    prefix: Optional[str] = None,
    *,
    ignore_config_files: bool = False,
) -> Dict[str, str]:
    """
    Return a distutils install scheme
    """
    from distutils.dist import Distribution

    dist_args: Dict[str, Union[str, List[str]]] = {"name": dist_name}
    if isolated:
        dist_args["script_args"] = ["--no-user-cfg"]

    d = Distribution(dist_args)
    if not ignore_config_files:
        try:
            d.parse_config_files()
        except UnicodeDecodeError:
            paths = d.find_config_files()
            logger.warning(
                "Ignore distutils configs in %s due to encoding errors.",
                ", ".join(os.path.basename(p) for p in paths),
            )
    obj: Optional[DistutilsCommand] = None
    obj = d.get_command_obj("install", create=True)
    assert obj is not None
    i: distutils_install_command = obj
    # NOTE: setting user or home has the side-effect of creating the home dir
    # or user base for installations during finalize_options()
    # ideally, we'd prefer a scheme class that has no side-effects.
    assert not (user and prefix), f"user={user} prefix={prefix}"
    assert not (home and prefix), f"home={home} prefix={prefix}"
    i.user = user or i.user
    if user or home:
        i.prefix = ""
    i.prefix = prefix or i.prefix
    i.home = home or i.home
    i.root = root or i.root
    i.finalize_options()

    scheme: Dict[str, str] = {}
    for key in SCHEME_KEYS:
        scheme[key] = getattr(i, "install_" + key)

    # install_lib specified in setup.cfg should install *everything*
    # into there (i.e. it takes precedence over both purelib and
    # platlib).  Note, i.install_lib is *always* set after
    # finalize_options(); we only want to override here if the user
    # has explicitly requested it hence going back to the config
    if "install_lib" in d.get_option_dict("install"):
        scheme.update({"purelib": i.install_lib, "platlib": i.install_lib})

    if running_under_virtualenv():
        if home:
            prefix = home
        elif user:
            prefix = i.install_userbase
        else:
            prefix = i.prefix
        scheme["headers"] = os.path.join(
            prefix,
            "include",
            "site",
            f"python{get_major_minor_version()}",
            dist_name,
        )

        if root is not None:
            path_no_drive = os.path.splitdrive(os.path.abspath(scheme["headers"]))[1]
            scheme["headers"] = os.path.join(root, path_no_drive[1:])

    return scheme


def get_scheme(
    dist_name: str,
    user: bool = False,
    home: Optional[str] = None,
    root: Optional[str] = None,
    isolated: bool = False,
    prefix: Optional[str] = None,
) -> Scheme:
    """
    Get the "scheme" corresponding to the input parameters. The distutils
    documentation provides the context for the available schemes:
    https://docs.python.org/3/install/index.html#alternate-installation

    :param dist_name: the name of the package to retrieve the scheme for, used
        in the headers scheme path
    :param user: indicates to use the "user" scheme
    :param home: indicates to use the "home" scheme and provides the base
        directory for the same
    :param root: root under which other directories are re-based
    :param isolated: equivalent to --no-user-cfg, i.e. do not consider
        ~/.pydistutils.cfg (posix) or ~/pydistutils.cfg (non-posix) for
        scheme paths
    :param prefix: indicates to use the "prefix" scheme and provides the
        base directory for the same
    """
    scheme = distutils_scheme(dist_name, user, home, root, isolated, prefix)
    return Scheme(
        platlib=scheme["platlib"],
        purelib=scheme["purelib"],
        headers=scheme["headers"],
        scripts=scheme["scripts"],
        data=scheme["data"],
    )


def get_bin_prefix() -> str:
    # XXX: In old virtualenv versions, sys.prefix can contain '..' components,
    # so we need to call normpath to eliminate them.
    prefix = os.path.normpath(sys.prefix)
    if WINDOWS:
        bin_py = os.path.join(prefix, "Scripts")
        # buildout uses 'bin' on Windows too?
        if not os.path.exists(bin_py):
            bin_py = os.path.join(prefix, "bin")
        return bin_py
    # Forcing to use /usr/local/bin for standard macOS framework installs
    # Also log to ~/Library/Logs/ for use with the Console.app log viewer
    if sys.platform[:6] == "darwin" and prefix[:16] == "/System/Library/":
        return "/usr/local/bin"
    return os.path.join(prefix, "bin")


def get_purelib() -> str:
    return get_python_lib(plat_specific=False)


def get_platlib() -> str:
    return get_python_lib(plat_specific=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\keysyms\winconstants.py ===
# This file contains constants that are normally found in win32all
# But included here to avoid the dependency


VK_LBUTTON = 1
VK_RBUTTON = 2
VK_CANCEL = 3
VK_MBUTTON = 4
VK_XBUTTON1 = 5
VK_XBUTTON2 = 6
VK_BACK = 8
VK_TAB = 9
VK_CLEAR = 12
VK_RETURN = 13
VK_SHIFT = 16
VK_CONTROL = 17
VK_MENU = 18
VK_PAUSE = 19
VK_CAPITAL = 20
VK_KANA = 0x15
VK_HANGEUL = 0x15
VK_HANGUL = 0x15
VK_JUNJA = 0x17
VK_FINAL = 0x18
VK_HANJA = 0x19
VK_KANJI = 0x19
VK_ESCAPE = 0x1B
VK_CONVERT = 0x1C
VK_NONCONVERT = 0x1D
VK_ACCEPT = 0x1E
VK_MODECHANGE = 0x1F
VK_SPACE = 32
VK_PRIOR = 33
VK_NEXT = 34
VK_END = 35
VK_HOME = 36
VK_LEFT = 37
VK_UP = 38
VK_RIGHT = 39
VK_DOWN = 40
VK_SELECT = 41
VK_PRINT = 42
VK_EXECUTE = 43
VK_SNAPSHOT = 44
VK_INSERT = 45
VK_DELETE = 46
VK_HELP = 47
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_APPS = 0x5D
VK_SLEEP = 0x5F
VK_NUMPAD0 = 0x60
VK_NUMPAD1 = 0x61
VK_NUMPAD2 = 0x62
VK_NUMPAD3 = 0x63
VK_NUMPAD4 = 0x64
VK_NUMPAD5 = 0x65
VK_NUMPAD6 = 0x66
VK_NUMPAD7 = 0x67
VK_NUMPAD8 = 0x68
VK_NUMPAD9 = 0x69
VK_MULTIPLY = 0x6A
VK_ADD = 0x6B
VK_SEPARATOR = 0x6C
VK_SUBTRACT = 0x6D
VK_DECIMAL = 0x6E
VK_DIVIDE = 0x6F
VK_F1 = 0x70
VK_F2 = 0x71
VK_F3 = 0x72
VK_F4 = 0x73
VK_F5 = 0x74
VK_F6 = 0x75
VK_F7 = 0x76
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79
VK_F11 = 0x7A
VK_F12 = 0x7B
VK_F13 = 0x7C
VK_F14 = 0x7D
VK_F15 = 0x7E
VK_F16 = 0x7F
VK_F17 = 0x80
VK_F18 = 0x81
VK_F19 = 0x82
VK_F20 = 0x83
VK_F21 = 0x84
VK_F22 = 0x85
VK_F23 = 0x86
VK_F24 = 0x87
VK_NUMLOCK = 0x90
VK_SCROLL = 0x91
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5
VK_BROWSER_BACK = 0xA6
VK_BROWSER_FORWARD = 0xA7
VK_BROWSER_REFRESH = 0xA8
VK_BROWSER_STOP = 0xA9
VK_BROWSER_SEARCH = 0xAA
VK_BROWSER_FAVORITES = 0xAB
VK_BROWSER_HOME = 0xAC
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_LAUNCH_MAIL = 0xB4
VK_LAUNCH_MEDIA_SELECT = 0xB5
VK_LAUNCH_APP1 = 0xB6
VK_LAUNCH_APP2 = 0xB7
VK_OEM_1 = 0xBA
VK_OEM_PLUS = 0xBB
VK_OEM_COMMA = 0xBC
VK_OEM_MINUS = 0xBD
VK_OEM_PERIOD = 0xBE
VK_OEM_2 = 0xBF
VK_OEM_3 = 0xC0
VK_OEM_4 = 0xDB
VK_OEM_5 = 0xDC
VK_OEM_6 = 0xDD
VK_OEM_7 = 0xDE
VK_OEM_8 = 0xDF
VK_OEM_102 = 0xE2
VK_PROCESSKEY = 0xE5
VK_PACKET = 0xE7
VK_ATTN = 0xF6
VK_CRSEL = 0xF7
VK_EXSEL = 0xF8
VK_EREOF = 0xF9
VK_PLAY = 0xFA
VK_ZOOM = 0xFB
VK_NONAME = 0xFC
VK_PA1 = 0xFD
VK_OEM_CLEAR = 0xFE

CF_TEXT = 1
CF_BITMAP = 2
CF_METAFILEPICT = 3
CF_SYLK = 4
CF_DIF = 5
CF_TIFF = 6
CF_OEMTEXT = 7
CF_DIB = 8
CF_PALETTE = 9
CF_PENDATA = 10
CF_RIFF = 11
CF_WAVE = 12
CF_UNICODETEXT = 13
CF_ENHMETAFILE = 14
CF_HDROP = 15
CF_LOCALE = 16
CF_MAX = 17
CF_OWNERDISPLAY = 128
CF_DSPTEXT = 129
CF_DSPBITMAP = 130
CF_DSPMETAFILEPICT = 131
CF_DSPENHMETAFILE = 142
CF_PRIVATEFIRST = 512
CF_PRIVATELAST = 767
CF_GDIOBJFIRST = 768
CF_GDIOBJLAST = 1023


GPTR = 64
GHND = 66

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pytz\lazy.py ===
from threading import RLock
try:
    from collections.abc import Mapping as DictMixin
except ImportError:  # Python < 3.3
    try:
        from UserDict import DictMixin  # Python 2
    except ImportError:  # Python 3.0-3.3
        from collections import Mapping as DictMixin


# With lazy loading, we might end up with multiple threads triggering
# it at the same time. We need a lock.
_fill_lock = RLock()


class LazyDict(DictMixin):
    """Dictionary populated on first use."""
    data = None

    def __getitem__(self, key):
        if self.data is None:
            _fill_lock.acquire()
            try:
                if self.data is None:
                    self._fill()
            finally:
                _fill_lock.release()
        return self.data[key.upper()]

    def __contains__(self, key):
        if self.data is None:
            _fill_lock.acquire()
            try:
                if self.data is None:
                    self._fill()
            finally:
                _fill_lock.release()
        return key in self.data

    def __iter__(self):
        if self.data is None:
            _fill_lock.acquire()
            try:
                if self.data is None:
                    self._fill()
            finally:
                _fill_lock.release()
        return iter(self.data)

    def __len__(self):
        if self.data is None:
            _fill_lock.acquire()
            try:
                if self.data is None:
                    self._fill()
            finally:
                _fill_lock.release()
        return len(self.data)

    def keys(self):
        if self.data is None:
            _fill_lock.acquire()
            try:
                if self.data is None:
                    self._fill()
            finally:
                _fill_lock.release()
        return self.data.keys()


class LazyList(list):
    """List populated on first use."""

    _props = [
        '__str__', '__repr__', '__unicode__',
        '__hash__', '__sizeof__', '__cmp__',
        '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__',
        'append', 'count', 'index', 'extend', 'insert', 'pop', 'remove',
        'reverse', 'sort', '__add__', '__radd__', '__iadd__', '__mul__',
        '__rmul__', '__imul__', '__contains__', '__len__', '__nonzero__',
        '__getitem__', '__setitem__', '__delitem__', '__iter__',
        '__reversed__', '__getslice__', '__setslice__', '__delslice__']

    def __new__(cls, fill_iter=None):

        if fill_iter is None:
            return list()

        # We need a new class as we will be dynamically messing with its
        # methods.
        class LazyList(list):
            pass

        fill_iter = [fill_iter]

        def lazy(name):
            def _lazy(self, *args, **kw):
                _fill_lock.acquire()
                try:
                    if len(fill_iter) > 0:
                        list.extend(self, fill_iter.pop())
                        for method_name in cls._props:
                            delattr(LazyList, method_name)
                finally:
                    _fill_lock.release()
                return getattr(list, name)(self, *args, **kw)
            return _lazy

        for name in cls._props:
            setattr(LazyList, name, lazy(name))

        new_list = LazyList()
        return new_list

# Not all versions of Python declare the same magic methods.
# Filter out properties that don't exist in this version of Python
# from the list.
LazyList._props = [prop for prop in LazyList._props if hasattr(list, prop)]


class LazySet(set):
    """Set populated on first use."""

    _props = (
        '__str__', '__repr__', '__unicode__',
        '__hash__', '__sizeof__', '__cmp__',
        '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__',
        '__contains__', '__len__', '__nonzero__',
        '__getitem__', '__setitem__', '__delitem__', '__iter__',
        '__sub__', '__and__', '__xor__', '__or__',
        '__rsub__', '__rand__', '__rxor__', '__ror__',
        '__isub__', '__iand__', '__ixor__', '__ior__',
        'add', 'clear', 'copy', 'difference', 'difference_update',
        'discard', 'intersection', 'intersection_update', 'isdisjoint',
        'issubset', 'issuperset', 'pop', 'remove',
        'symmetric_difference', 'symmetric_difference_update',
        'union', 'update')

    def __new__(cls, fill_iter=None):

        if fill_iter is None:
            return set()

        class LazySet(set):
            pass

        fill_iter = [fill_iter]

        def lazy(name):
            def _lazy(self, *args, **kw):
                _fill_lock.acquire()
                try:
                    if len(fill_iter) > 0:
                        for i in fill_iter.pop():
                            set.add(self, i)
                        for method_name in cls._props:
                            delattr(LazySet, method_name)
                finally:
                    _fill_lock.release()
                return getattr(set, name)(self, *args, **kw)
            return _lazy

        for name in cls._props:
            setattr(LazySet, name, lazy(name))

        new_set = LazySet()
        return new_set

# Not all versions of Python declare the same magic methods.
# Filter out properties that don't exist in this version of Python
# from the list.
LazySet._props = [prop for prop in LazySet._props if hasattr(set, prop)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_args.py ===
"""Test whether all elements of cls.args are instances of Basic. """

# NOTE: keep tests sorted by (module, class name) key. If a class can't
# be instantiated, add it here anyway with @SKIP("abstract class) (see
# e.g. Function).

import os
import re
from pathlib import Path

from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.function import (Function, Lambda)
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin

from sympy.testing.pytest import SKIP, warns_deprecated_sympy

a, b, c, x, y, z, s = symbols('a,b,c,x,y,z,s')


whitelist = [
     "sympy.assumptions.predicates",    # tested by test_predicates()
     "sympy.assumptions.relation.equality",    # tested by test_predicates()
]

def test_all_classes_are_tested():
    this = os.path.split(__file__)[0]
    path = os.path.join(this, os.pardir, os.pardir)
    sympy_path = os.path.abspath(path)
    prefix = os.path.split(sympy_path)[0] + os.sep

    re_cls = re.compile(r"^class ([A-Za-z][A-Za-z0-9_]*)\s*\(", re.MULTILINE)

    modules = {}

    for root, dirs, files in os.walk(sympy_path):
        module = root.replace(prefix, "").replace(os.sep, ".")

        for file in files:
            if file.startswith(("_", "test_", "bench_")):
                continue
            if not file.endswith(".py"):
                continue

            text = Path(os.path.join(root, file)).read_text(encoding='utf-8')

            submodule = module + '.' + file[:-3]

            if any(submodule.startswith(wpath) for wpath in whitelist):
                continue

            names = re_cls.findall(text)

            if not names:
                continue

            try:
                mod = __import__(submodule, fromlist=names)
            except ImportError:
                continue

            def is_Basic(name):
                cls = getattr(mod, name)
                if hasattr(cls, '_sympy_deprecated_func'):
                    cls = cls._sympy_deprecated_func
                if not isinstance(cls, type):
                    # check instance of singleton class with same name
                    cls = type(cls)
                return issubclass(cls, Basic)

            names = list(filter(is_Basic, names))

            if names:
                modules[submodule] = names

    ns = globals()
    failed = []

    for module, names in modules.items():
        mod = module.replace('.', '__')

        for name in names:
            test = 'test_' + mod + '__' + name

            if test not in ns:
                failed.append(module + '.' + name)

    assert not failed, "Missing classes: %s.  Please add tests for these to sympy/core/tests/test_args.py." % ", ".join(failed)


def _test_args(obj):
    all_basic = all(isinstance(arg, Basic) for arg in obj.args)
    # Ideally obj.func(*obj.args) would always recreate the object, but for
    # now, we only require it for objects with non-empty .args
    recreatable = not obj.args or obj.func(*obj.args) == obj
    return all_basic and recreatable


def test_sympy__algebras__quaternion__Quaternion():
    from sympy.algebras.quaternion import Quaternion
    assert _test_args(Quaternion(x, 1, 2, 3))


def test_sympy__assumptions__assume__AppliedPredicate():
    from sympy.assumptions.assume import AppliedPredicate, Predicate
    assert _test_args(AppliedPredicate(Predicate("test"), 2))
    assert _test_args(Q.is_true(True))

@SKIP("abstract class")
def test_sympy__assumptions__assume__Predicate():
    pass

def test_predicates():
    predicates = [
        getattr(Q, attr)
        for attr in Q.__class__.__dict__
        if not attr.startswith('__')]
    for p in predicates:
        assert _test_args(p)

def test_sympy__assumptions__assume__UndefinedPredicate():
    from sympy.assumptions.assume import Predicate
    assert _test_args(Predicate("test"))

@SKIP('abstract class')
def test_sympy__assumptions__relation__binrel__BinaryRelation():
    pass

def test_sympy__assumptions__relation__binrel__AppliedBinaryRelation():
    assert _test_args(Q.eq(1, 2))

def test_sympy__assumptions__wrapper__AssumptionsWrapper():
    from sympy.assumptions.wrapper import AssumptionsWrapper
    assert _test_args(AssumptionsWrapper(x, Q.positive(x)))

@SKIP("abstract Class")
def test_sympy__codegen__ast__CodegenAST():
    from sympy.codegen.ast import CodegenAST
    assert _test_args(CodegenAST())

@SKIP("abstract Class")
def test_sympy__codegen__ast__AssignmentBase():
    from sympy.codegen.ast import AssignmentBase
    assert _test_args(AssignmentBase(x, 1))

@SKIP("abstract Class")
def test_sympy__codegen__ast__AugmentedAssignment():
    from sympy.codegen.ast import AugmentedAssignment
    assert _test_args(AugmentedAssignment(x, 1))

def test_sympy__codegen__ast__AddAugmentedAssignment():
    from sympy.codegen.ast import AddAugmentedAssignment
    assert _test_args(AddAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__SubAugmentedAssignment():
    from sympy.codegen.ast import SubAugmentedAssignment
    assert _test_args(SubAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__MulAugmentedAssignment():
    from sympy.codegen.ast import MulAugmentedAssignment
    assert _test_args(MulAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__DivAugmentedAssignment():
    from sympy.codegen.ast import DivAugmentedAssignment
    assert _test_args(DivAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__ModAugmentedAssignment():
    from sympy.codegen.ast import ModAugmentedAssignment
    assert _test_args(ModAugmentedAssignment(x, 1))

def test_sympy__codegen__ast__CodeBlock():
    from sympy.codegen.ast import CodeBlock, Assignment
    assert _test_args(CodeBlock(Assignment(x, 1), Assignment(y, 2)))

def test_sympy__codegen__ast__For():
    from sympy.codegen.ast import For, CodeBlock, AddAugmentedAssignment
    from sympy.sets import Range
    assert _test_args(For(x, Range(10), CodeBlock(AddAugmentedAssignment(y, 1))))


def test_sympy__codegen__ast__Token():
    from sympy.codegen.ast import Token
    assert _test_args(Token())


def test_sympy__codegen__ast__ContinueToken():
    from sympy.codegen.ast import ContinueToken
    assert _test_args(ContinueToken())

def test_sympy__codegen__ast__BreakToken():
    from sympy.codegen.ast import BreakToken
    assert _test_args(BreakToken())

def test_sympy__codegen__ast__NoneToken():
    from sympy.codegen.ast import NoneToken
    assert _test_args(NoneToken())

def test_sympy__codegen__ast__String():
    from sympy.codegen.ast import String
    assert _test_args(String('foobar'))

def test_sympy__codegen__ast__QuotedString():
    from sympy.codegen.ast import QuotedString
    assert _test_args(QuotedString('foobar'))

def test_sympy__codegen__ast__Comment():
    from sympy.codegen.ast import Comment
    assert _test_args(Comment('this is a comment'))

def test_sympy__codegen__ast__Node():
    from sympy.codegen.ast import Node
    assert _test_args(Node())
    assert _test_args(Node(attrs={1, 2, 3}))


def test_sympy__codegen__ast__Type():
    from sympy.codegen.ast import Type
    assert _test_args(Type('float128'))


def test_sympy__codegen__ast__IntBaseType():
    from sympy.codegen.ast import IntBaseType
    assert _test_args(IntBaseType('bigint'))


def test_sympy__codegen__ast___SizedIntType():
    from sympy.codegen.ast import _SizedIntType
    assert _test_args(_SizedIntType('int128', 128))


def test_sympy__codegen__ast__SignedIntType():
    from sympy.codegen.ast import SignedIntType
    assert _test_args(SignedIntType('int128_with_sign', 128))


def test_sympy__codegen__ast__UnsignedIntType():
    from sympy.codegen.ast import UnsignedIntType
    assert _test_args(UnsignedIntType('unt128', 128))


def test_sympy__codegen__ast__FloatBaseType():
    from sympy.codegen.ast import FloatBaseType
    assert _test_args(FloatBaseType('positive_real'))


def test_sympy__codegen__ast__FloatType():
    from sympy.codegen.ast import FloatType
    assert _test_args(FloatType('float242', 242, nmant=142, nexp=99))


def test_sympy__codegen__ast__ComplexBaseType():
    from sympy.codegen.ast import ComplexBaseType
    assert _test_args(ComplexBaseType('positive_cmplx'))

def test_sympy__codegen__ast__ComplexType():
    from sympy.codegen.ast import ComplexType
    assert _test_args(ComplexType('complex42', 42, nmant=15, nexp=5))


def test_sympy__codegen__ast__Attribute():
    from sympy.codegen.ast import Attribute
    assert _test_args(Attribute('noexcept'))


def test_sympy__codegen__ast__Variable():
    from sympy.codegen.ast import Variable, Type, value_const
    assert _test_args(Variable(x))
    assert _test_args(Variable(y, Type('float32'), {value_const}))
    assert _test_args(Variable(z, type=Type('float64')))


def test_sympy__codegen__ast__Pointer():
    from sympy.codegen.ast import Pointer, Type, pointer_const
    assert _test_args(Pointer(x))
    assert _test_args(Pointer(y, type=Type('float32')))
    assert _test_args(Pointer(z, Type('float64'), {pointer_const}))


def test_sympy__codegen__ast__Declaration():
    from sympy.codegen.ast import Declaration, Variable, Type
    vx = Variable(x, type=Type('float'))
    assert _test_args(Declaration(vx))


def test_sympy__codegen__ast__While():
    from sympy.codegen.ast import While, AddAugmentedAssignment
    assert _test_args(While(abs(x) < 1, [AddAugmentedAssignment(x, -1)]))


def test_sympy__codegen__ast__Scope():
    from sympy.codegen.ast import Scope, AddAugmentedAssignment
    assert _test_args(Scope([AddAugmentedAssignment(x, -1)]))


def test_sympy__codegen__ast__Stream():
    from sympy.codegen.ast import Stream
    assert _test_args(Stream('stdin'))

def test_sympy__codegen__ast__Print():
    from sympy.codegen.ast import Print
    assert _test_args(Print([x, y]))
    assert _test_args(Print([x, y], "%d %d"))


def test_sympy__codegen__ast__FunctionPrototype():
    from sympy.codegen.ast import FunctionPrototype, real, Declaration, Variable
    inp_x = Declaration(Variable(x, type=real))
    assert _test_args(FunctionPrototype(real, 'pwer', [inp_x]))


def test_sympy__codegen__ast__FunctionDefinition():
    from sympy.codegen.ast import FunctionDefinition, real, Declaration, Variable, Assignment
    inp_x = Declaration(Variable(x, type=real))
    assert _test_args(FunctionDefinition(real, 'pwer', [inp_x], [Assignment(x, x**2)]))


def test_sympy__codegen__ast__Raise():
    from sympy.codegen.ast import Raise
    assert _test_args(Raise(x))


def test_sympy__codegen__ast__Return():
    from sympy.codegen.ast import Return
    assert _test_args(Return(x))


def test_sympy__codegen__ast__RuntimeError_():
    from sympy.codegen.ast import RuntimeError_
    assert _test_args(RuntimeError_('"message"'))


def test_sympy__codegen__ast__FunctionCall():
    from sympy.codegen.ast import FunctionCall
    assert _test_args(FunctionCall('pwer', [x]))


def test_sympy__codegen__ast__Element():
    from sympy.codegen.ast import Element
    assert _test_args(Element('x', range(3)))


def test_sympy__codegen__cnodes__CommaOperator():
    from sympy.codegen.cnodes import CommaOperator
    assert _test_args(CommaOperator(1, 2))


def test_sympy__codegen__cnodes__goto():
    from sympy.codegen.cnodes import goto
    assert _test_args(goto('early_exit'))


def test_sympy__codegen__cnodes__Label():
    from sympy.codegen.cnodes import Label
    assert _test_args(Label('early_exit'))


def test_sympy__codegen__cnodes__PreDecrement():
    from sympy.codegen.cnodes import PreDecrement
    assert _test_args(PreDecrement(x))


def test_sympy__codegen__cnodes__PostDecrement():
    from sympy.codegen.cnodes import PostDecrement
    assert _test_args(PostDecrement(x))


def test_sympy__codegen__cnodes__PreIncrement():
    from sympy.codegen.cnodes import PreIncrement
    assert _test_args(PreIncrement(x))


def test_sympy__codegen__cnodes__PostIncrement():
    from sympy.codegen.cnodes import PostIncrement
    assert _test_args(PostIncrement(x))


def test_sympy__codegen__cnodes__struct():
    from sympy.codegen.ast import real, Variable
    from sympy.codegen.cnodes import struct
    assert _test_args(struct(declarations=[
        Variable(x, type=real),
        Variable(y, type=real)
    ]))


def test_sympy__codegen__cnodes__union():
    from sympy.codegen.ast import float32, int32, Variable
    from sympy.codegen.cnodes import union
    assert _test_args(union(declarations=[
        Variable(x, type=float32),
        Variable(y, type=int32)
    ]))


def test_sympy__codegen__cxxnodes__using():
    from sympy.codegen.cxxnodes import using
    assert _test_args(using('std::vector'))
    assert _test_args(using('std::vector', 'vec'))


def test_sympy__codegen__fnodes__Program():
    from sympy.codegen.fnodes import Program
    assert _test_args(Program('foobar', []))

def test_sympy__codegen__fnodes__Module():
    from sympy.codegen.fnodes import Module
    assert _test_args(Module('foobar', [], []))


def test_sympy__codegen__fnodes__Subroutine():
    from sympy.codegen.fnodes import Subroutine
    x = symbols('x', real=True)
    assert _test_args(Subroutine('foo', [x], []))


def test_sympy__codegen__fnodes__GoTo():
    from sympy.codegen.fnodes import GoTo
    assert _test_args(GoTo([10]))
    assert _test_args(GoTo([10, 20], x > 1))


def test_sympy__codegen__fnodes__FortranReturn():
    from sympy.codegen.fnodes import FortranReturn
    assert _test_args(FortranReturn(10))


def test_sympy__codegen__fnodes__Extent():
    from sympy.codegen.fnodes import Extent
    assert _test_args(Extent())
    assert _test_args(Extent(None))
    assert _test_args(Extent(':'))
    assert _test_args(Extent(-3, 4))
    assert _test_args(Extent(x, y))


def test_sympy__codegen__fnodes__use_rename():
    from sympy.codegen.fnodes import use_rename
    assert _test_args(use_rename('loc', 'glob'))


def test_sympy__codegen__fnodes__use():
    from sympy.codegen.fnodes import use
    assert _test_args(use('modfoo', only='bar'))


def test_sympy__codegen__fnodes__SubroutineCall():
    from sympy.codegen.fnodes import SubroutineCall
    assert _test_args(SubroutineCall('foo', ['bar', 'baz']))


def test_sympy__codegen__fnodes__Do():
    from sympy.codegen.fnodes import Do
    assert _test_args(Do([], 'i', 1, 42))


def test_sympy__codegen__fnodes__ImpliedDoLoop():
    from sympy.codegen.fnodes import ImpliedDoLoop
    assert _test_args(ImpliedDoLoop('i', 'i', 1, 42))


def test_sympy__codegen__fnodes__ArrayConstructor():
    from sympy.codegen.fnodes import ArrayConstructor
    assert _test_args(ArrayConstructor([1, 2, 3]))
    from sympy.codegen.fnodes import ImpliedDoLoop
    idl = ImpliedDoLoop('i', 'i', 1, 42)
    assert _test_args(ArrayConstructor([1, idl, 3]))


def test_sympy__codegen__fnodes__sum_():
    from sympy.codegen.fnodes import sum_
    assert _test_args(sum_('arr'))


def test_sympy__codegen__fnodes__product_():
    from sympy.codegen.fnodes import product_
    assert _test_args(product_('arr'))


def test_sympy__codegen__numpy_nodes__logaddexp():
    from sympy.codegen.numpy_nodes import logaddexp
    assert _test_args(logaddexp(x, y))


def test_sympy__codegen__numpy_nodes__logaddexp2():
    from sympy.codegen.numpy_nodes import logaddexp2
    assert _test_args(logaddexp2(x, y))


def test_sympy__codegen__numpy_nodes__amin():
    from sympy.codegen.numpy_nodes import amin
    assert _test_args(amin(x))


def test_sympy__codegen__numpy_nodes__amax():
    from sympy.codegen.numpy_nodes import amax
    assert _test_args(amax(x))


def test_sympy__codegen__numpy_nodes__minimum():
    from sympy.codegen.numpy_nodes import minimum
    assert _test_args(minimum(x, y, z))


def test_sympy__codegen__numpy_nodes__maximum():
    from sympy.codegen.numpy_nodes import maximum
    assert _test_args(maximum(x, y, z))


def test_sympy__codegen__pynodes__List():
    from sympy.codegen.pynodes import List
    assert _test_args(List(1, 2, 3))


def test_sympy__codegen__pynodes__NumExprEvaluate():
    from sympy.codegen.pynodes import NumExprEvaluate
    assert _test_args(NumExprEvaluate(x))


def test_sympy__codegen__scipy_nodes__cosm1():
    from sympy.codegen.scipy_nodes import cosm1
    assert _test_args(cosm1(x))


def test_sympy__codegen__scipy_nodes__powm1():
    from sympy.codegen.scipy_nodes import powm1
    assert _test_args(powm1(x, y))


def test_sympy__codegen__abstract_nodes__List():
    from sympy.codegen.abstract_nodes import List
    assert _test_args(List(1, 2, 3))

def test_sympy__combinatorics__graycode__GrayCode():
    from sympy.combinatorics.graycode import GrayCode
    # an integer is given and returned from GrayCode as the arg
    assert _test_args(GrayCode(3, start='100'))
    assert _test_args(GrayCode(3, rank=1))


def test_sympy__combinatorics__permutations__Permutation():
    from sympy.combinatorics.permutations import Permutation
    assert _test_args(Permutation([0, 1, 2, 3]))

def test_sympy__combinatorics__permutations__AppliedPermutation():
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.permutations import AppliedPermutation
    p = Permutation([0, 1, 2, 3])
    assert _test_args(AppliedPermutation(p, x))

def test_sympy__combinatorics__perm_groups__PermutationGroup():
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.perm_groups import PermutationGroup
    assert _test_args(PermutationGroup([Permutation([0, 1])]))


def test_sympy__combinatorics__polyhedron__Polyhedron():
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.polyhedron import Polyhedron
    from sympy.abc import w, x, y, z
    pgroup = [Permutation([[0, 1, 2], [3]]),
              Permutation([[0, 1, 3], [2]]),
              Permutation([[0, 2, 3], [1]]),
              Permutation([[1, 2, 3], [0]]),
              Permutation([[0, 1], [2, 3]]),
              Permutation([[0, 2], [1, 3]]),
              Permutation([[0, 3], [1, 2]]),
              Permutation([[0, 1, 2, 3]])]
    corners = [w, x, y, z]
    faces = [(w, x, y), (w, y, z), (w, z, x), (x, y, z)]
    assert _test_args(Polyhedron(corners, faces, pgroup))


def test_sympy__combinatorics__prufer__Prufer():
    from sympy.combinatorics.prufer import Prufer
    assert _test_args(Prufer([[0, 1], [0, 2], [0, 3]], 4))


def test_sympy__combinatorics__partitions__Partition():
    from sympy.combinatorics.partitions import Partition
    assert _test_args(Partition([1]))


def test_sympy__combinatorics__partitions__IntegerPartition():
    from sympy.combinatorics.partitions import IntegerPartition
    assert _test_args(IntegerPartition([1]))


def test_sympy__concrete__products__Product():
    from sympy.concrete.products import Product
    assert _test_args(Product(x, (x, 0, 10)))
    assert _test_args(Product(x, (x, 0, y), (y, 0, 10)))


@SKIP("abstract Class")
def test_sympy__concrete__expr_with_limits__ExprWithLimits():
    from sympy.concrete.expr_with_limits import ExprWithLimits
    assert _test_args(ExprWithLimits(x, (x, 0, 10)))
    assert _test_args(ExprWithLimits(x*y, (x, 0, 10.),(y,1.,3)))


@SKIP("abstract Class")
def test_sympy__concrete__expr_with_limits__AddWithLimits():
    from sympy.concrete.expr_with_limits import AddWithLimits
    assert _test_args(AddWithLimits(x, (x, 0, 10)))
    assert _test_args(AddWithLimits(x*y, (x, 0, 10),(y,1,3)))


@SKIP("abstract Class")
def test_sympy__concrete__expr_with_intlimits__ExprWithIntLimits():
    from sympy.concrete.expr_with_intlimits import ExprWithIntLimits
    assert _test_args(ExprWithIntLimits(x, (x, 0, 10)))
    assert _test_args(ExprWithIntLimits(x*y, (x, 0, 10),(y,1,3)))


def test_sympy__concrete__summations__Sum():
    from sympy.concrete.summations import Sum
    assert _test_args(Sum(x, (x, 0, 10)))
    assert _test_args(Sum(x, (x, 0, y), (y, 0, 10)))


def test_sympy__core__add__Add():
    from sympy.core.add import Add
    assert _test_args(Add(x, y, z, 2))


def test_sympy__core__basic__Atom():
    from sympy.core.basic import Atom
    assert _test_args(Atom())


def test_sympy__core__basic__Basic():
    from sympy.core.basic import Basic
    assert _test_args(Basic())


def test_sympy__core__containers__Dict():
    from sympy.core.containers import Dict
    assert _test_args(Dict({x: y, y: z}))


def test_sympy__core__containers__Tuple():
    from sympy.core.containers import Tuple
    assert _test_args(Tuple(x, y, z, 2))


def test_sympy__core__expr__AtomicExpr():
    from sympy.core.expr import AtomicExpr
    assert _test_args(AtomicExpr())


def test_sympy__core__expr__Expr():
    from sympy.core.expr import Expr
    assert _test_args(Expr())


def test_sympy__core__expr__UnevaluatedExpr():
    from sympy.core.expr import UnevaluatedExpr
    from sympy.abc import x
    assert _test_args(UnevaluatedExpr(x))


def test_sympy__core__function__Application():
    from sympy.core.function import Application
    assert _test_args(Application(1, 2, 3))


def test_sympy__core__function__AppliedUndef():
    from sympy.core.function import AppliedUndef
    assert _test_args(AppliedUndef(1, 2, 3))


def test_sympy__core__function__DefinedFunction():
    from sympy.core.function import DefinedFunction
    assert _test_args(DefinedFunction(1, 2, 3))


def test_sympy__core__function__Derivative():
    from sympy.core.function import Derivative
    assert _test_args(Derivative(2, x, y, 3))


@SKIP("abstract class")
def test_sympy__core__function__Function():
    pass


def test_sympy__core__function__Lambda():
    assert _test_args(Lambda((x, y), x + y + z))


def test_sympy__core__function__Subs():
    from sympy.core.function import Subs
    assert _test_args(Subs(x + y, x, 2))


def test_sympy__core__function__WildFunction():
    from sympy.core.function import WildFunction
    assert _test_args(WildFunction('f'))


def test_sympy__core__mod__Mod():
    from sympy.core.mod import Mod
    assert _test_args(Mod(x, 2))


def test_sympy__core__mul__Mul():
    from sympy.core.mul import Mul
    assert _test_args(Mul(2, x, y, z))


def test_sympy__core__numbers__Catalan():
    from sympy.core.numbers import Catalan
    assert _test_args(Catalan())


def test_sympy__core__numbers__ComplexInfinity():
    from sympy.core.numbers import ComplexInfinity
    assert _test_args(ComplexInfinity())


def test_sympy__core__numbers__EulerGamma():
    from sympy.core.numbers import EulerGamma
    assert _test_args(EulerGamma())


def test_sympy__core__numbers__Exp1():
    from sympy.core.numbers import Exp1
    assert _test_args(Exp1())


def test_sympy__core__numbers__Float():
    from sympy.core.numbers import Float
    assert _test_args(Float(1.23))


def test_sympy__core__numbers__GoldenRatio():
    from sympy.core.numbers import GoldenRatio
    assert _test_args(GoldenRatio())


def test_sympy__core__numbers__TribonacciConstant():
    from sympy.core.numbers import TribonacciConstant
    assert _test_args(TribonacciConstant())


def test_sympy__core__numbers__Half():
    from sympy.core.numbers import Half
    assert _test_args(Half())


def test_sympy__core__numbers__ImaginaryUnit():
    from sympy.core.numbers import ImaginaryUnit
    assert _test_args(ImaginaryUnit())


def test_sympy__core__numbers__Infinity():
    from sympy.core.numbers import Infinity
    assert _test_args(Infinity())


def test_sympy__core__numbers__Integer():
    from sympy.core.numbers import Integer
    assert _test_args(Integer(7))


@SKIP("abstract class")
def test_sympy__core__numbers__IntegerConstant():
    pass


def test_sympy__core__numbers__NaN():
    from sympy.core.numbers import NaN
    assert _test_args(NaN())


def test_sympy__core__numbers__NegativeInfinity():
    from sympy.core.numbers import NegativeInfinity
    assert _test_args(NegativeInfinity())


def test_sympy__core__numbers__NegativeOne():
    from sympy.core.numbers import NegativeOne
    assert _test_args(NegativeOne())


def test_sympy__core__numbers__Number():
    from sympy.core.numbers import Number
    assert _test_args(Number(1, 7))


def test_sympy__core__numbers__NumberSymbol():
    from sympy.core.numbers import NumberSymbol
    assert _test_args(NumberSymbol())


def test_sympy__core__numbers__One():
    from sympy.core.numbers import One
    assert _test_args(One())


def test_sympy__core__numbers__Pi():
    from sympy.core.numbers import Pi
    assert _test_args(Pi())


def test_sympy__core__numbers__Rational():
    from sympy.core.numbers import Rational
    assert _test_args(Rational(1, 7))


@SKIP("abstract class")
def test_sympy__core__numbers__RationalConstant():
    pass


def test_sympy__core__numbers__Zero():
    from sympy.core.numbers import Zero
    assert _test_args(Zero())


@SKIP("abstract class")
def test_sympy__core__operations__AssocOp():
    pass


@SKIP("abstract class")
def test_sympy__core__operations__LatticeOp():
    pass


def test_sympy__core__power__Pow():
    from sympy.core.power import Pow
    assert _test_args(Pow(x, 2))


def test_sympy__core__relational__Equality():
    from sympy.core.relational import Equality
    assert _test_args(Equality(x, 2))


def test_sympy__core__relational__GreaterThan():
    from sympy.core.relational import GreaterThan
    assert _test_args(GreaterThan(x, 2))


def test_sympy__core__relational__LessThan():
    from sympy.core.relational import LessThan
    assert _test_args(LessThan(x, 2))


@SKIP("abstract class")
def test_sympy__core__relational__Relational():
    pass


def test_sympy__core__relational__StrictGreaterThan():
    from sympy.core.relational import StrictGreaterThan
    assert _test_args(StrictGreaterThan(x, 2))


def test_sympy__core__relational__StrictLessThan():
    from sympy.core.relational import StrictLessThan
    assert _test_args(StrictLessThan(x, 2))


def test_sympy__core__relational__Unequality():
    from sympy.core.relational import Unequality
    assert _test_args(Unequality(x, 2))


def test_sympy__sandbox__indexed_integrals__IndexedIntegral():
    from sympy.tensor import IndexedBase, Idx
    from sympy.sandbox.indexed_integrals import IndexedIntegral
    A = IndexedBase('A')
    i, j = symbols('i j', integer=True)
    a1, a2 = symbols('a1:3', cls=Idx)
    assert _test_args(IndexedIntegral(A[a1], A[a2]))
    assert _test_args(IndexedIntegral(A[i], A[j]))


def test_sympy__calculus__accumulationbounds__AccumulationBounds():
    from sympy.calculus.accumulationbounds import AccumulationBounds
    assert _test_args(AccumulationBounds(0, 1))


def test_sympy__sets__ordinals__OmegaPower():
    from sympy.sets.ordinals import OmegaPower
    assert _test_args(OmegaPower(1, 1))

def test_sympy__sets__ordinals__Ordinal():
    from sympy.sets.ordinals import Ordinal, OmegaPower
    assert _test_args(Ordinal(OmegaPower(2, 1)))

def test_sympy__sets__ordinals__OrdinalOmega():
    from sympy.sets.ordinals import OrdinalOmega
    assert _test_args(OrdinalOmega())

def test_sympy__sets__ordinals__OrdinalZero():
    from sympy.sets.ordinals import OrdinalZero
    assert _test_args(OrdinalZero())


def test_sympy__sets__powerset__PowerSet():
    from sympy.sets.powerset import PowerSet
    from sympy.core.singleton import S
    assert _test_args(PowerSet(S.EmptySet))


def test_sympy__sets__sets__EmptySet():
    from sympy.sets.sets import EmptySet
    assert _test_args(EmptySet())


def test_sympy__sets__sets__UniversalSet():
    from sympy.sets.sets import UniversalSet
    assert _test_args(UniversalSet())


def test_sympy__sets__sets__FiniteSet():
    from sympy.sets.sets import FiniteSet
    assert _test_args(FiniteSet(x, y, z))


def test_sympy__sets__sets__Interval():
    from sympy.sets.sets import Interval
    assert _test_args(Interval(0, 1))


def test_sympy__sets__sets__ProductSet():
    from sympy.sets.sets import ProductSet, Interval
    assert _test_args(ProductSet(Interval(0, 1), Interval(0, 1)))


@SKIP("does it make sense to test this?")
def test_sympy__sets__sets__Set():
    from sympy.sets.sets import Set
    assert _test_args(Set())


def test_sympy__sets__sets__Intersection():
    from sympy.sets.sets import Intersection, Interval
    from sympy.core.symbol import Symbol
    x = Symbol('x')
    y = Symbol('y')
    S = Intersection(Interval(0, x), Interval(y, 1))
    assert isinstance(S, Intersection)
    assert _test_args(S)


def test_sympy__sets__sets__Union():
    from sympy.sets.sets import Union, Interval
    assert _test_args(Union(Interval(0, 1), Interval(2, 3)))


def test_sympy__sets__sets__Complement():
    from sympy.sets.sets import Complement, Interval
    assert _test_args(Complement(Interval(0, 2), Interval(0, 1)))


def test_sympy__sets__sets__SymmetricDifference():
    from sympy.sets.sets import FiniteSet, SymmetricDifference
    assert _test_args(SymmetricDifference(FiniteSet(1, 2, 3), \
           FiniteSet(2, 3, 4)))

def test_sympy__sets__sets__DisjointUnion():
    from sympy.sets.sets import FiniteSet, DisjointUnion
    assert _test_args(DisjointUnion(FiniteSet(1, 2, 3), \
           FiniteSet(2, 3, 4)))


def test_sympy__physics__quantum__trace__Tr():
    from sympy.physics.quantum.trace import Tr
    a, b = symbols('a b', commutative=False)
    assert _test_args(Tr(a + b))


def test_sympy__sets__setexpr__SetExpr():
    from sympy.sets.setexpr import SetExpr
    from sympy.sets.sets import Interval
    assert _test_args(SetExpr(Interval(0, 1)))


def test_sympy__sets__fancysets__Rationals():
    from sympy.sets.fancysets import Rationals
    assert _test_args(Rationals())


def test_sympy__sets__fancysets__Naturals():
    from sympy.sets.fancysets import Naturals
    assert _test_args(Naturals())


def test_sympy__sets__fancysets__Naturals0():
    from sympy.sets.fancysets import Naturals0
    assert _test_args(Naturals0())


def test_sympy__sets__fancysets__Integers():
    from sympy.sets.fancysets import Integers
    assert _test_args(Integers())


def test_sympy__sets__fancysets__Reals():
    from sympy.sets.fancysets import Reals
    assert _test_args(Reals())


def test_sympy__sets__fancysets__Complexes():
    from sympy.sets.fancysets import Complexes
    assert _test_args(Complexes())


def test_sympy__sets__fancysets__ComplexRegion():
    from sympy.sets.fancysets import ComplexRegion
    from sympy.core.singleton import S
    from sympy.sets import Interval
    a = Interval(0, 1)
    b = Interval(2, 3)
    theta = Interval(0, 2*S.Pi)
    assert _test_args(ComplexRegion(a*b))
    assert _test_args(ComplexRegion(a*theta, polar=True))


def test_sympy__sets__fancysets__CartesianComplexRegion():
    from sympy.sets.fancysets import CartesianComplexRegion
    from sympy.sets import Interval
    a = Interval(0, 1)
    b = Interval(2, 3)
    assert _test_args(CartesianComplexRegion(a*b))


def test_sympy__sets__fancysets__PolarComplexRegion():
    from sympy.sets.fancysets import PolarComplexRegion
    from sympy.core.singleton import S
    from sympy.sets import Interval
    a = Interval(0, 1)
    theta = Interval(0, 2*S.Pi)
    assert _test_args(PolarComplexRegion(a*theta))


def test_sympy__sets__fancysets__ImageSet():
    from sympy.sets.fancysets import ImageSet
    from sympy.core.singleton import S
    from sympy.core.symbol import Symbol
    x = Symbol('x')
    assert _test_args(ImageSet(Lambda(x, x**2), S.Naturals))


def test_sympy__sets__fancysets__Range():
    from sympy.sets.fancysets import Range
    assert _test_args(Range(1, 5, 1))


def test_sympy__sets__conditionset__ConditionSet():
    from sympy.sets.conditionset import ConditionSet
    from sympy.core.singleton import S
    from sympy.core.symbol import Symbol
    x = Symbol('x')
    assert _test_args(ConditionSet(x, Eq(x**2, 1), S.Reals))


def test_sympy__sets__contains__Contains():
    from sympy.sets.fancysets import Range
    from sympy.sets.contains import Contains
    assert _test_args(Contains(x, Range(0, 10, 2)))


# STATS


from sympy.stats.crv_types import NormalDistribution
nd = NormalDistribution(0, 1)
from sympy.stats.frv_types import DieDistribution
die = DieDistribution(6)


def test_sympy__stats__crv__ContinuousDomain():
    from sympy.sets.sets import Interval
    from sympy.stats.crv import ContinuousDomain
    assert _test_args(ContinuousDomain({x}, Interval(-oo, oo)))


def test_sympy__stats__crv__SingleContinuousDomain():
    from sympy.sets.sets import Interval
    from sympy.stats.crv import SingleContinuousDomain
    assert _test_args(SingleContinuousDomain(x, Interval(-oo, oo)))


def test_sympy__stats__crv__ProductContinuousDomain():
    from sympy.sets.sets import Interval
    from sympy.stats.crv import SingleContinuousDomain, ProductContinuousDomain
    D = SingleContinuousDomain(x, Interval(-oo, oo))
    E = SingleContinuousDomain(y, Interval(0, oo))
    assert _test_args(ProductContinuousDomain(D, E))


def test_sympy__stats__crv__ConditionalContinuousDomain():
    from sympy.sets.sets import Interval
    from sympy.stats.crv import (SingleContinuousDomain,
            ConditionalContinuousDomain)
    D = SingleContinuousDomain(x, Interval(-oo, oo))
    assert _test_args(ConditionalContinuousDomain(D, x > 0))


def test_sympy__stats__crv__ContinuousPSpace():
    from sympy.sets.sets import Interval
    from sympy.stats.crv import ContinuousPSpace, SingleContinuousDomain
    D = SingleContinuousDomain(x, Interval(-oo, oo))
    assert _test_args(ContinuousPSpace(D, nd))


def test_sympy__stats__crv__SingleContinuousPSpace():
    from sympy.stats.crv import SingleContinuousPSpace
    assert _test_args(SingleContinuousPSpace(x, nd))

@SKIP("abstract class")
def test_sympy__stats__rv__Distribution():
    pass

@SKIP("abstract class")
def test_sympy__stats__crv__SingleContinuousDistribution():
    pass


def test_sympy__stats__drv__SingleDiscreteDomain():
    from sympy.stats.drv import SingleDiscreteDomain
    assert _test_args(SingleDiscreteDomain(x, S.Naturals))


def test_sympy__stats__drv__ProductDiscreteDomain():
    from sympy.stats.drv import SingleDiscreteDomain, ProductDiscreteDomain
    X = SingleDiscreteDomain(x, S.Naturals)
    Y = SingleDiscreteDomain(y, S.Integers)
    assert _test_args(ProductDiscreteDomain(X, Y))


def test_sympy__stats__drv__SingleDiscretePSpace():
    from sympy.stats.drv import SingleDiscretePSpace
    from sympy.stats.drv_types import PoissonDistribution
    assert _test_args(SingleDiscretePSpace(x, PoissonDistribution(1)))

def test_sympy__stats__drv__DiscretePSpace():
    from sympy.stats.drv import DiscretePSpace, SingleDiscreteDomain
    density = Lambda(x, 2**(-x))
    domain = SingleDiscreteDomain(x, S.Naturals)
    assert _test_args(DiscretePSpace(domain, density))

def test_sympy__stats__drv__ConditionalDiscreteDomain():
    from sympy.stats.drv import ConditionalDiscreteDomain, SingleDiscreteDomain
    X = SingleDiscreteDomain(x, S.Naturals0)
    assert _test_args(ConditionalDiscreteDomain(X, x > 2))

def test_sympy__stats__joint_rv__JointPSpace():
    from sympy.stats.joint_rv import JointPSpace, JointDistribution
    assert _test_args(JointPSpace('X', JointDistribution(1)))

def test_sympy__stats__joint_rv__JointRandomSymbol():
    from sympy.stats.joint_rv import JointRandomSymbol
    assert _test_args(JointRandomSymbol(x))

def test_sympy__stats__joint_rv_types__JointDistributionHandmade():
    from sympy.tensor.indexed import Indexed
    from sympy.stats.joint_rv_types import JointDistributionHandmade
    x1, x2 = (Indexed('x', i) for i in (1, 2))
    assert _test_args(JointDistributionHandmade(x1 + x2, S.Reals**2))


def test_sympy__stats__joint_rv__MarginalDistribution():
    from sympy.stats.rv import RandomSymbol
    from sympy.stats.joint_rv import MarginalDistribution
    r = RandomSymbol(S('r'))
    assert _test_args(MarginalDistribution(r, (r,)))


def test_sympy__stats__compound_rv__CompoundDistribution():
    from sympy.stats.compound_rv import CompoundDistribution
    from sympy.stats.drv_types import PoissonDistribution, Poisson
    r = Poisson('r', 10)
    assert _test_args(CompoundDistribution(PoissonDistribution(r)))


def test_sympy__stats__compound_rv__CompoundPSpace():
    from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
    from sympy.stats.drv_types import PoissonDistribution, Poisson
    r = Poisson('r', 5)
    C = CompoundDistribution(PoissonDistribution(r))
    assert _test_args(CompoundPSpace('C', C))


@SKIP("abstract class")
def test_sympy__stats__drv__SingleDiscreteDistribution():
    pass

@SKIP("abstract class")
def test_sympy__stats__drv__DiscreteDistribution():
    pass

@SKIP("abstract class")
def test_sympy__stats__drv__DiscreteDomain():
    pass


def test_sympy__stats__rv__RandomDomain():
    from sympy.stats.rv import RandomDomain
    from sympy.sets.sets import FiniteSet
    assert _test_args(RandomDomain(FiniteSet(x), FiniteSet(1, 2, 3)))


def test_sympy__stats__rv__SingleDomain():
    from sympy.stats.rv import SingleDomain
    from sympy.sets.sets import FiniteSet
    assert _test_args(SingleDomain(x, FiniteSet(1, 2, 3)))


def test_sympy__stats__rv__ConditionalDomain():
    from sympy.stats.rv import ConditionalDomain, RandomDomain
    from sympy.sets.sets import FiniteSet
    D = RandomDomain(FiniteSet(x), FiniteSet(1, 2))
    assert _test_args(ConditionalDomain(D, x > 1))

def test_sympy__stats__rv__MatrixDomain():
    from sympy.stats.rv import MatrixDomain
    from sympy.matrices import MatrixSet
    from sympy.core.singleton import S
    assert _test_args(MatrixDomain(x, MatrixSet(2, 2, S.Reals)))

def test_sympy__stats__rv__PSpace():
    from sympy.stats.rv import PSpace, RandomDomain
    from sympy.sets.sets import FiniteSet
    D = RandomDomain(FiniteSet(x), FiniteSet(1, 2, 3, 4, 5, 6))
    assert _test_args(PSpace(D, die))


@SKIP("abstract Class")
def test_sympy__stats__rv__SinglePSpace():
    pass


def test_sympy__stats__rv__RandomSymbol():
    from sympy.stats.rv import RandomSymbol
    from sympy.stats.crv import SingleContinuousPSpace
    A = SingleContinuousPSpace(x, nd)
    assert _test_args(RandomSymbol(x, A))


@SKIP("abstract Class")
def test_sympy__stats__rv__ProductPSpace():
    pass


def test_sympy__stats__rv__IndependentProductPSpace():
    from sympy.stats.rv import IndependentProductPSpace
    from sympy.stats.crv import SingleContinuousPSpace
    A = SingleContinuousPSpace(x, nd)
    B = SingleContinuousPSpace(y, nd)
    assert _test_args(IndependentProductPSpace(A, B))


def test_sympy__stats__rv__ProductDomain():
    from sympy.sets.sets import Interval
    from sympy.stats.rv import ProductDomain, SingleDomain
    D = SingleDomain(x, Interval(-oo, oo))
    E = SingleDomain(y, Interval(0, oo))
    assert _test_args(ProductDomain(D, E))


def test_sympy__stats__symbolic_probability__Probability():
    from sympy.stats.symbolic_probability import Probability
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(Probability(X > 0))


def test_sympy__stats__symbolic_probability__Expectation():
    from sympy.stats.symbolic_probability import Expectation
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(Expectation(X > 0))


def test_sympy__stats__symbolic_probability__Covariance():
    from sympy.stats.symbolic_probability import Covariance
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 3)
    assert _test_args(Covariance(X, Y))


def test_sympy__stats__symbolic_probability__Variance():
    from sympy.stats.symbolic_probability import Variance
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(Variance(X))


def test_sympy__stats__symbolic_probability__Moment():
    from sympy.stats.symbolic_probability import Moment
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(Moment(X, 3, 2, X > 3))


def test_sympy__stats__symbolic_probability__CentralMoment():
    from sympy.stats.symbolic_probability import CentralMoment
    from sympy.stats import Normal
    X = Normal('X', 0, 1)
    assert _test_args(CentralMoment(X, 2, X > 1))


def test_sympy__stats__frv_types__DiscreteUniformDistribution():
    from sympy.stats.frv_types import DiscreteUniformDistribution
    from sympy.core.containers import Tuple
    assert _test_args(DiscreteUniformDistribution(Tuple(*list(range(6)))))


def test_sympy__stats__frv_types__DieDistribution():
    assert _test_args(die)


def test_sympy__stats__frv_types__BernoulliDistribution():
    from sympy.stats.frv_types import BernoulliDistribution
    assert _test_args(BernoulliDistribution(S.Half, 0, 1))


def test_sympy__stats__frv_types__BinomialDistribution():
    from sympy.stats.frv_types import BinomialDistribution
    assert _test_args(BinomialDistribution(5, S.Half, 1, 0))

def test_sympy__stats__frv_types__BetaBinomialDistribution():
    from sympy.stats.frv_types import BetaBinomialDistribution
    assert _test_args(BetaBinomialDistribution(5, 1, 1))


def test_sympy__stats__frv_types__HypergeometricDistribution():
    from sympy.stats.frv_types import HypergeometricDistribution
    assert _test_args(HypergeometricDistribution(10, 5, 3))


def test_sympy__stats__frv_types__RademacherDistribution():
    from sympy.stats.frv_types import RademacherDistribution
    assert _test_args(RademacherDistribution())

def test_sympy__stats__frv_types__IdealSolitonDistribution():
    from sympy.stats.frv_types import IdealSolitonDistribution
    assert _test_args(IdealSolitonDistribution(10))

def test_sympy__stats__frv_types__RobustSolitonDistribution():
    from sympy.stats.frv_types import RobustSolitonDistribution
    assert _test_args(RobustSolitonDistribution(1000, 0.5, 0.1))

def test_sympy__stats__frv__FiniteDomain():
    from sympy.stats.frv import FiniteDomain
    assert _test_args(FiniteDomain({(x, 1), (x, 2)}))  # x can be 1 or 2


def test_sympy__stats__frv__SingleFiniteDomain():
    from sympy.stats.frv import SingleFiniteDomain
    assert _test_args(SingleFiniteDomain(x, {1, 2}))  # x can be 1 or 2


def test_sympy__stats__frv__ProductFiniteDomain():
    from sympy.stats.frv import SingleFiniteDomain, ProductFiniteDomain
    xd = SingleFiniteDomain(x, {1, 2})
    yd = SingleFiniteDomain(y, {1, 2})
    assert _test_args(ProductFiniteDomain(xd, yd))


def test_sympy__stats__frv__ConditionalFiniteDomain():
    from sympy.stats.frv import SingleFiniteDomain, ConditionalFiniteDomain
    xd = SingleFiniteDomain(x, {1, 2})
    assert _test_args(ConditionalFiniteDomain(xd, x > 1))


def test_sympy__stats__frv__FinitePSpace():
    from sympy.stats.frv import FinitePSpace, SingleFiniteDomain
    xd = SingleFiniteDomain(x, {1, 2, 3, 4, 5, 6})
    assert _test_args(FinitePSpace(xd, {(x, 1): S.Half, (x, 2): S.Half}))

    xd = SingleFiniteDomain(x, {1, 2})
    assert _test_args(FinitePSpace(xd, {(x, 1): S.Half, (x, 2): S.Half}))


def test_sympy__stats__frv__SingleFinitePSpace():
    from sympy.stats.frv import SingleFinitePSpace
    from sympy.core.symbol import Symbol

    assert _test_args(SingleFinitePSpace(Symbol('x'), die))


def test_sympy__stats__frv__ProductFinitePSpace():
    from sympy.stats.frv import SingleFinitePSpace, ProductFinitePSpace
    from sympy.core.symbol import Symbol
    xp = SingleFinitePSpace(Symbol('x'), die)
    yp = SingleFinitePSpace(Symbol('y'), die)
    assert _test_args(ProductFinitePSpace(xp, yp))

@SKIP("abstract class")
def test_sympy__stats__frv__SingleFiniteDistribution():
    pass

@SKIP("abstract class")
def test_sympy__stats__crv__ContinuousDistribution():
    pass


def test_sympy__stats__frv_types__FiniteDistributionHandmade():
    from sympy.stats.frv_types import FiniteDistributionHandmade
    from sympy.core.containers import Dict
    assert _test_args(FiniteDistributionHandmade(Dict({1: 1})))


def test_sympy__stats__crv_types__ContinuousDistributionHandmade():
    from sympy.stats.crv_types import ContinuousDistributionHandmade
    from sympy.core.function import Lambda
    from sympy.sets.sets import Interval
    from sympy.abc import x
    assert _test_args(ContinuousDistributionHandmade(Lambda(x, 2*x),
                                                     Interval(0, 1)))


def test_sympy__stats__drv_types__DiscreteDistributionHandmade():
    from sympy.stats.drv_types import DiscreteDistributionHandmade
    from sympy.core.function import Lambda
    from sympy.sets.sets import FiniteSet
    from sympy.abc import x
    assert _test_args(DiscreteDistributionHandmade(Lambda(x, Rational(1, 10)),
                                                    FiniteSet(*range(10))))


def test_sympy__stats__rv__Density():
    from sympy.stats.rv import Density
    from sympy.stats.crv_types import Normal
    assert _test_args(Density(Normal('x', 0, 1)))


def test_sympy__stats__crv_types__ArcsinDistribution():
    from sympy.stats.crv_types import ArcsinDistribution
    assert _test_args(ArcsinDistribution(0, 1))


def test_sympy__stats__crv_types__BeniniDistribution():
    from sympy.stats.crv_types import BeniniDistribution
    assert _test_args(BeniniDistribution(1, 1, 1))


def test_sympy__stats__crv_types__BetaDistribution():
    from sympy.stats.crv_types import BetaDistribution
    assert _test_args(BetaDistribution(1, 1))

def test_sympy__stats__crv_types__BetaNoncentralDistribution():
    from sympy.stats.crv_types import BetaNoncentralDistribution
    assert _test_args(BetaNoncentralDistribution(1, 1, 1))


def test_sympy__stats__crv_types__BetaPrimeDistribution():
    from sympy.stats.crv_types import BetaPrimeDistribution
    assert _test_args(BetaPrimeDistribution(1, 1))

def test_sympy__stats__crv_types__BoundedParetoDistribution():
    from sympy.stats.crv_types import BoundedParetoDistribution
    assert _test_args(BoundedParetoDistribution(1, 1, 2))

def test_sympy__stats__crv_types__CauchyDistribution():
    from sympy.stats.crv_types import CauchyDistribution
    assert _test_args(CauchyDistribution(0, 1))


def test_sympy__stats__crv_types__ChiDistribution():
    from sympy.stats.crv_types import ChiDistribution
    assert _test_args(ChiDistribution(1))


def test_sympy__stats__crv_types__ChiNoncentralDistribution():
    from sympy.stats.crv_types import ChiNoncentralDistribution
    assert _test_args(ChiNoncentralDistribution(1,1))


def test_sympy__stats__crv_types__ChiSquaredDistribution():
    from sympy.stats.crv_types import ChiSquaredDistribution
    assert _test_args(ChiSquaredDistribution(1))


def test_sympy__stats__crv_types__DagumDistribution():
    from sympy.stats.crv_types import DagumDistribution
    assert _test_args(DagumDistribution(1, 1, 1))


def test_sympy__stats__crv_types__DavisDistribution():
    from sympy.stats.crv_types import DavisDistribution
    assert _test_args(DavisDistribution(1, 1, 1))


def test_sympy__stats__crv_types__ExGaussianDistribution():
    from sympy.stats.crv_types import ExGaussianDistribution
    assert _test_args(ExGaussianDistribution(1, 1, 1))


def test_sympy__stats__crv_types__ExponentialDistribution():
    from sympy.stats.crv_types import ExponentialDistribution
    assert _test_args(ExponentialDistribution(1))


def test_sympy__stats__crv_types__ExponentialPowerDistribution():
    from sympy.stats.crv_types import ExponentialPowerDistribution
    assert _test_args(ExponentialPowerDistribution(0, 1, 1))


def test_sympy__stats__crv_types__FDistributionDistribution():
    from sympy.stats.crv_types import FDistributionDistribution
    assert _test_args(FDistributionDistribution(1, 1))


def test_sympy__stats__crv_types__FisherZDistribution():
    from sympy.stats.crv_types import FisherZDistribution
    assert _test_args(FisherZDistribution(1, 1))


def test_sympy__stats__crv_types__FrechetDistribution():
    from sympy.stats.crv_types import FrechetDistribution
    assert _test_args(FrechetDistribution(1, 1, 1))


def test_sympy__stats__crv_types__GammaInverseDistribution():
    from sympy.stats.crv_types import GammaInverseDistribution
    assert _test_args(GammaInverseDistribution(1, 1))


def test_sympy__stats__crv_types__GammaDistribution():
    from sympy.stats.crv_types import GammaDistribution
    assert _test_args(GammaDistribution(1, 1))

def test_sympy__stats__crv_types__GumbelDistribution():
    from sympy.stats.crv_types import GumbelDistribution
    assert _test_args(GumbelDistribution(1, 1, False))

def test_sympy__stats__crv_types__GompertzDistribution():
    from sympy.stats.crv_types import GompertzDistribution
    assert _test_args(GompertzDistribution(1, 1))

def test_sympy__stats__crv_types__KumaraswamyDistribution():
    from sympy.stats.crv_types import KumaraswamyDistribution
    assert _test_args(KumaraswamyDistribution(1, 1))

def test_sympy__stats__crv_types__LaplaceDistribution():
    from sympy.stats.crv_types import LaplaceDistribution
    assert _test_args(LaplaceDistribution(0, 1))

def test_sympy__stats__crv_types__LevyDistribution():
    from sympy.stats.crv_types import LevyDistribution
    assert _test_args(LevyDistribution(0, 1))

def test_sympy__stats__crv_types__LogCauchyDistribution():
    from sympy.stats.crv_types import LogCauchyDistribution
    assert _test_args(LogCauchyDistribution(0, 1))

def test_sympy__stats__crv_types__LogisticDistribution():
    from sympy.stats.crv_types import LogisticDistribution
    assert _test_args(LogisticDistribution(0, 1))

def test_sympy__stats__crv_types__LogLogisticDistribution():
    from sympy.stats.crv_types import LogLogisticDistribution
    assert _test_args(LogLogisticDistribution(1, 1))

def test_sympy__stats__crv_types__LogitNormalDistribution():
    from sympy.stats.crv_types import LogitNormalDistribution
    assert _test_args(LogitNormalDistribution(0, 1))

def test_sympy__stats__crv_types__LogNormalDistribution():
    from sympy.stats.crv_types import LogNormalDistribution
    assert _test_args(LogNormalDistribution(0, 1))

def test_sympy__stats__crv_types__LomaxDistribution():
    from sympy.stats.crv_types import LomaxDistribution
    assert _test_args(LomaxDistribution(1, 2))

def test_sympy__stats__crv_types__MaxwellDistribution():
    from sympy.stats.crv_types import MaxwellDistribution
    assert _test_args(MaxwellDistribution(1))

def test_sympy__stats__crv_types__MoyalDistribution():
    from sympy.stats.crv_types import MoyalDistribution
    assert _test_args(MoyalDistribution(1,2))

def test_sympy__stats__crv_types__NakagamiDistribution():
    from sympy.stats.crv_types import NakagamiDistribution
    assert _test_args(NakagamiDistribution(1, 1))


def test_sympy__stats__crv_types__NormalDistribution():
    from sympy.stats.crv_types import NormalDistribution
    assert _test_args(NormalDistribution(0, 1))

def test_sympy__stats__crv_types__GaussianInverseDistribution():
    from sympy.stats.crv_types import GaussianInverseDistribution
    assert _test_args(GaussianInverseDistribution(1, 1))


def test_sympy__stats__crv_types__ParetoDistribution():
    from sympy.stats.crv_types import ParetoDistribution
    assert _test_args(ParetoDistribution(1, 1))

def test_sympy__stats__crv_types__PowerFunctionDistribution():
    from sympy.stats.crv_types import PowerFunctionDistribution
    assert _test_args(PowerFunctionDistribution(2,0,1))

def test_sympy__stats__crv_types__QuadraticUDistribution():
    from sympy.stats.crv_types import QuadraticUDistribution
    assert _test_args(QuadraticUDistribution(1, 2))

def test_sympy__stats__crv_types__RaisedCosineDistribution():
    from sympy.stats.crv_types import RaisedCosineDistribution
    assert _test_args(RaisedCosineDistribution(1, 1))

def test_sympy__stats__crv_types__RayleighDistribution():
    from sympy.stats.crv_types import RayleighDistribution
    assert _test_args(RayleighDistribution(1))

def test_sympy__stats__crv_types__ReciprocalDistribution():
    from sympy.stats.crv_types import ReciprocalDistribution
    assert _test_args(ReciprocalDistribution(5, 30))

def test_sympy__stats__crv_types__ShiftedGompertzDistribution():
    from sympy.stats.crv_types import ShiftedGompertzDistribution
    assert _test_args(ShiftedGompertzDistribution(1, 1))

def test_sympy__stats__crv_types__StudentTDistribution():
    from sympy.stats.crv_types import StudentTDistribution
    assert _test_args(StudentTDistribution(1))

def test_sympy__stats__crv_types__TrapezoidalDistribution():
    from sympy.stats.crv_types import TrapezoidalDistribution
    assert _test_args(TrapezoidalDistribution(1, 2, 3, 4))

def test_sympy__stats__crv_types__TriangularDistribution():
    from sympy.stats.crv_types import TriangularDistribution
    assert _test_args(TriangularDistribution(-1, 0, 1))


def test_sympy__stats__crv_types__UniformDistribution():
    from sympy.stats.crv_types import UniformDistribution
    assert _test_args(UniformDistribution(0, 1))


def test_sympy__stats__crv_types__UniformSumDistribution():
    from sympy.stats.crv_types import UniformSumDistribution
    assert _test_args(UniformSumDistribution(1))


def test_sympy__stats__crv_types__VonMisesDistribution():
    from sympy.stats.crv_types import VonMisesDistribution
    assert _test_args(VonMisesDistribution(1, 1))


def test_sympy__stats__crv_types__WeibullDistribution():
    from sympy.stats.crv_types import WeibullDistribution
    assert _test_args(WeibullDistribution(1, 1))


def test_sympy__stats__crv_types__WignerSemicircleDistribution():
    from sympy.stats.crv_types import WignerSemicircleDistribution
    assert _test_args(WignerSemicircleDistribution(1))


def test_sympy__stats__drv_types__GeometricDistribution():
    from sympy.stats.drv_types import GeometricDistribution
    assert _test_args(GeometricDistribution(.5))

def test_sympy__stats__drv_types__HermiteDistribution():
    from sympy.stats.drv_types import HermiteDistribution
    assert _test_args(HermiteDistribution(1, 2))

def test_sympy__stats__drv_types__LogarithmicDistribution():
    from sympy.stats.drv_types import LogarithmicDistribution
    assert _test_args(LogarithmicDistribution(.5))


def test_sympy__stats__drv_types__NegativeBinomialDistribution():
    from sympy.stats.drv_types import NegativeBinomialDistribution
    assert _test_args(NegativeBinomialDistribution(.5, .5))

def test_sympy__stats__drv_types__FlorySchulzDistribution():
    from sympy.stats.drv_types import FlorySchulzDistribution
    assert _test_args(FlorySchulzDistribution(.5))

def test_sympy__stats__drv_types__PoissonDistribution():
    from sympy.stats.drv_types import PoissonDistribution
    assert _test_args(PoissonDistribution(1))


def test_sympy__stats__drv_types__SkellamDistribution():
    from sympy.stats.drv_types import SkellamDistribution
    assert _test_args(SkellamDistribution(1, 1))


def test_sympy__stats__drv_types__YuleSimonDistribution():
    from sympy.stats.drv_types import YuleSimonDistribution
    assert _test_args(YuleSimonDistribution(.5))


def test_sympy__stats__drv_types__ZetaDistribution():
    from sympy.stats.drv_types import ZetaDistribution
    assert _test_args(ZetaDistribution(1.5))


def test_sympy__stats__joint_rv__JointDistribution():
    from sympy.stats.joint_rv import JointDistribution
    assert _test_args(JointDistribution(1, 2, 3, 4))


def test_sympy__stats__joint_rv_types__MultivariateNormalDistribution():
    from sympy.stats.joint_rv_types import MultivariateNormalDistribution
    assert _test_args(
        MultivariateNormalDistribution([0, 1], [[1, 0],[0, 1]]))

def test_sympy__stats__joint_rv_types__MultivariateLaplaceDistribution():
    from sympy.stats.joint_rv_types import MultivariateLaplaceDistribution
    assert _test_args(MultivariateLaplaceDistribution([0, 1], [[1, 0],[0, 1]]))


def test_sympy__stats__joint_rv_types__MultivariateTDistribution():
    from sympy.stats.joint_rv_types import MultivariateTDistribution
    assert _test_args(MultivariateTDistribution([0, 1], [[1, 0],[0, 1]], 1))


def test_sympy__stats__joint_rv_types__NormalGammaDistribution():
    from sympy.stats.joint_rv_types import NormalGammaDistribution
    assert _test_args(NormalGammaDistribution(1, 2, 3, 4))

def test_sympy__stats__joint_rv_types__GeneralizedMultivariateLogGammaDistribution():
    from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaDistribution
    v, l, mu = (4, [1, 2, 3, 4], [1, 2, 3, 4])
    assert _test_args(GeneralizedMultivariateLogGammaDistribution(S.Half, v, l, mu))

def test_sympy__stats__joint_rv_types__MultivariateBetaDistribution():
    from sympy.stats.joint_rv_types import MultivariateBetaDistribution
    assert _test_args(MultivariateBetaDistribution([1, 2, 3]))

def test_sympy__stats__joint_rv_types__MultivariateEwensDistribution():
    from sympy.stats.joint_rv_types import MultivariateEwensDistribution
    assert _test_args(MultivariateEwensDistribution(5, 1))

def test_sympy__stats__joint_rv_types__MultinomialDistribution():
    from sympy.stats.joint_rv_types import MultinomialDistribution
    assert _test_args(MultinomialDistribution(5, [0.5, 0.1, 0.3]))

def test_sympy__stats__joint_rv_types__NegativeMultinomialDistribution():
    from sympy.stats.joint_rv_types import NegativeMultinomialDistribution
    assert _test_args(NegativeMultinomialDistribution(5, [0.5, 0.1, 0.3]))

def test_sympy__stats__rv__RandomIndexedSymbol():
    from sympy.stats.rv import RandomIndexedSymbol, pspace
    from sympy.stats.stochastic_process_types import DiscreteMarkovChain
    X = DiscreteMarkovChain("X")
    assert _test_args(RandomIndexedSymbol(X[0].symbol, pspace(X[0])))

def test_sympy__stats__rv__RandomMatrixSymbol():
    from sympy.stats.rv import RandomMatrixSymbol
    from sympy.stats.random_matrix import RandomMatrixPSpace
    pspace = RandomMatrixPSpace('P')
    assert _test_args(RandomMatrixSymbol('M', 3, 3, pspace))

def test_sympy__stats__stochastic_process__StochasticPSpace():
    from sympy.stats.stochastic_process import StochasticPSpace
    from sympy.stats.stochastic_process_types import StochasticProcess
    from sympy.stats.frv_types import BernoulliDistribution
    assert _test_args(StochasticPSpace("Y", StochasticProcess("Y", [1, 2, 3]), BernoulliDistribution(S.Half, 1, 0)))

def test_sympy__stats__stochastic_process_types__StochasticProcess():
    from sympy.stats.stochastic_process_types import StochasticProcess
    assert _test_args(StochasticProcess("Y", [1, 2, 3]))

def test_sympy__stats__stochastic_process_types__MarkovProcess():
    from sympy.stats.stochastic_process_types import MarkovProcess
    assert _test_args(MarkovProcess("Y", [1, 2, 3]))

def test_sympy__stats__stochastic_process_types__DiscreteTimeStochasticProcess():
    from sympy.stats.stochastic_process_types import DiscreteTimeStochasticProcess
    assert _test_args(DiscreteTimeStochasticProcess("Y", [1, 2, 3]))

def test_sympy__stats__stochastic_process_types__ContinuousTimeStochasticProcess():
    from sympy.stats.stochastic_process_types import ContinuousTimeStochasticProcess
    assert _test_args(ContinuousTimeStochasticProcess("Y", [1, 2, 3]))

def test_sympy__stats__stochastic_process_types__TransitionMatrixOf():
    from sympy.stats.stochastic_process_types import TransitionMatrixOf, DiscreteMarkovChain
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    DMC = DiscreteMarkovChain("Y")
    assert _test_args(TransitionMatrixOf(DMC, MatrixSymbol('T', 3, 3)))

def test_sympy__stats__stochastic_process_types__GeneratorMatrixOf():
    from sympy.stats.stochastic_process_types import GeneratorMatrixOf, ContinuousMarkovChain
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    DMC = ContinuousMarkovChain("Y")
    assert _test_args(GeneratorMatrixOf(DMC, MatrixSymbol('T', 3, 3)))

def test_sympy__stats__stochastic_process_types__StochasticStateSpaceOf():
    from sympy.stats.stochastic_process_types import StochasticStateSpaceOf, DiscreteMarkovChain
    DMC = DiscreteMarkovChain("Y")
    assert _test_args(StochasticStateSpaceOf(DMC, [0, 1, 2]))

def test_sympy__stats__stochastic_process_types__DiscreteMarkovChain():
    from sympy.stats.stochastic_process_types import DiscreteMarkovChain
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    assert _test_args(DiscreteMarkovChain("Y", [0, 1, 2], MatrixSymbol('T', 3, 3)))

def test_sympy__stats__stochastic_process_types__ContinuousMarkovChain():
    from sympy.stats.stochastic_process_types import ContinuousMarkovChain
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    assert _test_args(ContinuousMarkovChain("Y", [0, 1, 2], MatrixSymbol('T', 3, 3)))

def test_sympy__stats__stochastic_process_types__BernoulliProcess():
    from sympy.stats.stochastic_process_types import BernoulliProcess
    assert _test_args(BernoulliProcess("B", 0.5, 1, 0))

def test_sympy__stats__stochastic_process_types__CountingProcess():
    from sympy.stats.stochastic_process_types import CountingProcess
    assert _test_args(CountingProcess("C"))

def test_sympy__stats__stochastic_process_types__PoissonProcess():
    from sympy.stats.stochastic_process_types import PoissonProcess
    assert _test_args(PoissonProcess("X", 2))

def test_sympy__stats__stochastic_process_types__WienerProcess():
    from sympy.stats.stochastic_process_types import WienerProcess
    assert _test_args(WienerProcess("X"))

def test_sympy__stats__stochastic_process_types__GammaProcess():
    from sympy.stats.stochastic_process_types import GammaProcess
    assert _test_args(GammaProcess("X", 1, 2))

def test_sympy__stats__random_matrix__RandomMatrixPSpace():
    from sympy.stats.random_matrix import RandomMatrixPSpace
    from sympy.stats.random_matrix_models import RandomMatrixEnsembleModel
    model = RandomMatrixEnsembleModel('R', 3)
    assert _test_args(RandomMatrixPSpace('P', model=model))

def test_sympy__stats__random_matrix_models__RandomMatrixEnsembleModel():
    from sympy.stats.random_matrix_models import RandomMatrixEnsembleModel
    assert _test_args(RandomMatrixEnsembleModel('R', 3))

def test_sympy__stats__random_matrix_models__GaussianEnsembleModel():
    from sympy.stats.random_matrix_models import GaussianEnsembleModel
    assert _test_args(GaussianEnsembleModel('G', 3))

def test_sympy__stats__random_matrix_models__GaussianUnitaryEnsembleModel():
    from sympy.stats.random_matrix_models import GaussianUnitaryEnsembleModel
    assert _test_args(GaussianUnitaryEnsembleModel('U', 3))

def test_sympy__stats__random_matrix_models__GaussianOrthogonalEnsembleModel():
    from sympy.stats.random_matrix_models import GaussianOrthogonalEnsembleModel
    assert _test_args(GaussianOrthogonalEnsembleModel('U', 3))

def test_sympy__stats__random_matrix_models__GaussianSymplecticEnsembleModel():
    from sympy.stats.random_matrix_models import GaussianSymplecticEnsembleModel
    assert _test_args(GaussianSymplecticEnsembleModel('U', 3))

def test_sympy__stats__random_matrix_models__CircularEnsembleModel():
    from sympy.stats.random_matrix_models import CircularEnsembleModel
    assert _test_args(CircularEnsembleModel('C', 3))

def test_sympy__stats__random_matrix_models__CircularUnitaryEnsembleModel():
    from sympy.stats.random_matrix_models import CircularUnitaryEnsembleModel
    assert _test_args(CircularUnitaryEnsembleModel('U', 3))

def test_sympy__stats__random_matrix_models__CircularOrthogonalEnsembleModel():
    from sympy.stats.random_matrix_models import CircularOrthogonalEnsembleModel
    assert _test_args(CircularOrthogonalEnsembleModel('O', 3))

def test_sympy__stats__random_matrix_models__CircularSymplecticEnsembleModel():
    from sympy.stats.random_matrix_models import CircularSymplecticEnsembleModel
    assert _test_args(CircularSymplecticEnsembleModel('S', 3))

def test_sympy__stats__symbolic_multivariate_probability__ExpectationMatrix():
    from sympy.stats import ExpectationMatrix
    from sympy.stats.rv import RandomMatrixSymbol
    assert _test_args(ExpectationMatrix(RandomMatrixSymbol('R', 2, 1)))

def test_sympy__stats__symbolic_multivariate_probability__VarianceMatrix():
    from sympy.stats import VarianceMatrix
    from sympy.stats.rv import RandomMatrixSymbol
    assert _test_args(VarianceMatrix(RandomMatrixSymbol('R', 3, 1)))

def test_sympy__stats__symbolic_multivariate_probability__CrossCovarianceMatrix():
    from sympy.stats import CrossCovarianceMatrix
    from sympy.stats.rv import RandomMatrixSymbol
    assert _test_args(CrossCovarianceMatrix(RandomMatrixSymbol('R', 3, 1),
                        RandomMatrixSymbol('X', 3, 1)))

def test_sympy__stats__matrix_distributions__MatrixPSpace():
    from sympy.stats.matrix_distributions import MatrixDistribution, MatrixPSpace
    from sympy.matrices.dense import Matrix
    M = MatrixDistribution(1, Matrix([[1, 0], [0, 1]]))
    assert _test_args(MatrixPSpace('M', M, 2, 2))

def test_sympy__stats__matrix_distributions__MatrixDistribution():
    from sympy.stats.matrix_distributions import MatrixDistribution
    from sympy.matrices.dense import Matrix
    assert _test_args(MatrixDistribution(1, Matrix([[1, 0], [0, 1]])))

def test_sympy__stats__matrix_distributions__MatrixGammaDistribution():
    from sympy.stats.matrix_distributions import MatrixGammaDistribution
    from sympy.matrices.dense import Matrix
    assert _test_args(MatrixGammaDistribution(3, 4, Matrix([[1, 0], [0, 1]])))

def test_sympy__stats__matrix_distributions__WishartDistribution():
    from sympy.stats.matrix_distributions import WishartDistribution
    from sympy.matrices.dense import Matrix
    assert _test_args(WishartDistribution(3, Matrix([[1, 0], [0, 1]])))

def test_sympy__stats__matrix_distributions__MatrixNormalDistribution():
    from sympy.stats.matrix_distributions import MatrixNormalDistribution
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    L = MatrixSymbol('L', 1, 2)
    S1 = MatrixSymbol('S1', 1, 1)
    S2 = MatrixSymbol('S2', 2, 2)
    assert _test_args(MatrixNormalDistribution(L, S1, S2))

def test_sympy__stats__matrix_distributions__MatrixStudentTDistribution():
    from sympy.stats.matrix_distributions import MatrixStudentTDistribution
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    v = symbols('v', positive=True)
    Omega = MatrixSymbol('Omega', 3, 3)
    Sigma = MatrixSymbol('Sigma', 1, 1)
    Location = MatrixSymbol('Location', 1, 3)
    assert _test_args(MatrixStudentTDistribution(v, Location, Omega, Sigma))

def test_sympy__utilities__matchpy_connector__WildDot():
    from sympy.utilities.matchpy_connector import WildDot
    assert _test_args(WildDot("w_"))


def test_sympy__utilities__matchpy_connector__WildPlus():
    from sympy.utilities.matchpy_connector import WildPlus
    assert _test_args(WildPlus("w__"))


def test_sympy__utilities__matchpy_connector__WildStar():
    from sympy.utilities.matchpy_connector import WildStar
    assert _test_args(WildStar("w___"))


def test_sympy__core__symbol__Str():
    from sympy.core.symbol import Str
    assert _test_args(Str('t'))

def test_sympy__core__symbol__Dummy():
    from sympy.core.symbol import Dummy
    assert _test_args(Dummy('t'))


def test_sympy__core__symbol__Symbol():
    from sympy.core.symbol import Symbol
    assert _test_args(Symbol('t'))


def test_sympy__core__symbol__Wild():
    from sympy.core.symbol import Wild
    assert _test_args(Wild('x', exclude=[x]))


@SKIP("abstract class")
def test_sympy__functions__combinatorial__factorials__CombinatorialFunction():
    pass


def test_sympy__functions__combinatorial__factorials__FallingFactorial():
    from sympy.functions.combinatorial.factorials import FallingFactorial
    assert _test_args(FallingFactorial(2, x))


def test_sympy__functions__combinatorial__factorials__MultiFactorial():
    from sympy.functions.combinatorial.factorials import MultiFactorial
    assert _test_args(MultiFactorial(x))


def test_sympy__functions__combinatorial__factorials__RisingFactorial():
    from sympy.functions.combinatorial.factorials import RisingFactorial
    assert _test_args(RisingFactorial(2, x))


def test_sympy__functions__combinatorial__factorials__binomial():
    from sympy.functions.combinatorial.factorials import binomial
    assert _test_args(binomial(2, x))


def test_sympy__functions__combinatorial__factorials__subfactorial():
    from sympy.functions.combinatorial.factorials import subfactorial
    assert _test_args(subfactorial(x))


def test_sympy__functions__combinatorial__factorials__factorial():
    from sympy.functions.combinatorial.factorials import factorial
    assert _test_args(factorial(x))


def test_sympy__functions__combinatorial__factorials__factorial2():
    from sympy.functions.combinatorial.factorials import factorial2
    assert _test_args(factorial2(x))


def test_sympy__functions__combinatorial__numbers__bell():
    from sympy.functions.combinatorial.numbers import bell
    assert _test_args(bell(x, y))


def test_sympy__functions__combinatorial__numbers__bernoulli():
    from sympy.functions.combinatorial.numbers import bernoulli
    assert _test_args(bernoulli(x))


def test_sympy__functions__combinatorial__numbers__catalan():
    from sympy.functions.combinatorial.numbers import catalan
    assert _test_args(catalan(x))


def test_sympy__functions__combinatorial__numbers__genocchi():
    from sympy.functions.combinatorial.numbers import genocchi
    assert _test_args(genocchi(x))


def test_sympy__functions__combinatorial__numbers__euler():
    from sympy.functions.combinatorial.numbers import euler
    assert _test_args(euler(x))


def test_sympy__functions__combinatorial__numbers__andre():
    from sympy.functions.combinatorial.numbers import andre
    assert _test_args(andre(x))


def test_sympy__functions__combinatorial__numbers__carmichael():
    from sympy.functions.combinatorial.numbers import carmichael
    assert _test_args(carmichael(x))


def test_sympy__functions__combinatorial__numbers__divisor_sigma():
    from sympy.functions.combinatorial.numbers import divisor_sigma
    k = symbols('k', integer=True)
    n = symbols('n', integer=True)
    t = divisor_sigma(n, k)
    assert _test_args(t)


def test_sympy__functions__combinatorial__numbers__fibonacci():
    from sympy.functions.combinatorial.numbers import fibonacci
    assert _test_args(fibonacci(x))


def test_sympy__functions__combinatorial__numbers__jacobi_symbol():
    from sympy.functions.combinatorial.numbers import jacobi_symbol
    assert _test_args(jacobi_symbol(2, 3))


def test_sympy__functions__combinatorial__numbers__kronecker_symbol():
    from sympy.functions.combinatorial.numbers import kronecker_symbol
    assert _test_args(kronecker_symbol(2, 3))


def test_sympy__functions__combinatorial__numbers__legendre_symbol():
    from sympy.functions.combinatorial.numbers import legendre_symbol
    assert _test_args(legendre_symbol(2, 3))


def test_sympy__functions__combinatorial__numbers__mobius():
    from sympy.functions.combinatorial.numbers import mobius
    assert _test_args(mobius(2))


def test_sympy__functions__combinatorial__numbers__motzkin():
    from sympy.functions.combinatorial.numbers import motzkin
    assert _test_args(motzkin(5))


def test_sympy__functions__combinatorial__numbers__partition():
    from sympy.core.symbol import Symbol
    from sympy.functions.combinatorial.numbers import partition
    assert _test_args(partition(Symbol('a', integer=True)))


def test_sympy__functions__combinatorial__numbers__primenu():
    from sympy.functions.combinatorial.numbers import primenu
    n = symbols('n', integer=True)
    t = primenu(n)
    assert _test_args(t)


def test_sympy__functions__combinatorial__numbers__primeomega():
    from sympy.functions.combinatorial.numbers import primeomega
    n = symbols('n', integer=True)
    t = primeomega(n)
    assert _test_args(t)


def test_sympy__functions__combinatorial__numbers__primepi():
    from sympy.functions.combinatorial.numbers import primepi
    n = symbols('n')
    t = primepi(n)
    assert _test_args(t)


def test_sympy__functions__combinatorial__numbers__reduced_totient():
    from sympy.functions.combinatorial.numbers import reduced_totient
    k = symbols('k', integer=True)
    t = reduced_totient(k)
    assert _test_args(t)


def test_sympy__functions__combinatorial__numbers__totient():
    from sympy.functions.combinatorial.numbers import totient
    k = symbols('k', integer=True)
    t = totient(k)
    assert _test_args(t)


def test_sympy__functions__combinatorial__numbers__tribonacci():
    from sympy.functions.combinatorial.numbers import tribonacci
    assert _test_args(tribonacci(x))


def test_sympy__functions__combinatorial__numbers__udivisor_sigma():
    from sympy.functions.combinatorial.numbers import udivisor_sigma
    k = symbols('k', integer=True)
    n = symbols('n', integer=True)
    t = udivisor_sigma(n, k)
    assert _test_args(t)


def test_sympy__functions__combinatorial__numbers__harmonic():
    from sympy.functions.combinatorial.numbers import harmonic
    assert _test_args(harmonic(x, 2))


def test_sympy__functions__combinatorial__numbers__lucas():
    from sympy.functions.combinatorial.numbers import lucas
    assert _test_args(lucas(x))


def test_sympy__functions__elementary__complexes__Abs():
    from sympy.functions.elementary.complexes import Abs
    assert _test_args(Abs(x))


def test_sympy__functions__elementary__complexes__adjoint():
    from sympy.functions.elementary.complexes import adjoint
    assert _test_args(adjoint(x))


def test_sympy__functions__elementary__complexes__arg():
    from sympy.functions.elementary.complexes import arg
    assert _test_args(arg(x))


def test_sympy__functions__elementary__complexes__conjugate():
    from sympy.functions.elementary.complexes import conjugate
    assert _test_args(conjugate(x))


def test_sympy__functions__elementary__complexes__im():
    from sympy.functions.elementary.complexes import im
    assert _test_args(im(x))


def test_sympy__functions__elementary__complexes__re():
    from sympy.functions.elementary.complexes import re
    assert _test_args(re(x))


def test_sympy__functions__elementary__complexes__sign():
    from sympy.functions.elementary.complexes import sign
    assert _test_args(sign(x))


def test_sympy__functions__elementary__complexes__polar_lift():
    from sympy.functions.elementary.complexes import polar_lift
    assert _test_args(polar_lift(x))


def test_sympy__functions__elementary__complexes__periodic_argument():
    from sympy.functions.elementary.complexes import periodic_argument
    assert _test_args(periodic_argument(x, y))


def test_sympy__functions__elementary__complexes__principal_branch():
    from sympy.functions.elementary.complexes import principal_branch
    assert _test_args(principal_branch(x, y))


def test_sympy__functions__elementary__complexes__transpose():
    from sympy.functions.elementary.complexes import transpose
    assert _test_args(transpose(x))


def test_sympy__functions__elementary__exponential__LambertW():
    from sympy.functions.elementary.exponential import LambertW
    assert _test_args(LambertW(2))


@SKIP("abstract class")
def test_sympy__functions__elementary__exponential__ExpBase():
    pass


def test_sympy__functions__elementary__exponential__exp():
    from sympy.functions.elementary.exponential import exp
    assert _test_args(exp(2))


def test_sympy__functions__elementary__exponential__exp_polar():
    from sympy.functions.elementary.exponential import exp_polar
    assert _test_args(exp_polar(2))


def test_sympy__functions__elementary__exponential__log():
    from sympy.functions.elementary.exponential import log
    assert _test_args(log(2))


@SKIP("abstract class")
def test_sympy__functions__elementary__hyperbolic__HyperbolicFunction():
    pass


@SKIP("abstract class")
def test_sympy__functions__elementary__hyperbolic__ReciprocalHyperbolicFunction():
    pass


@SKIP("abstract class")
def test_sympy__functions__elementary__hyperbolic__InverseHyperbolicFunction():
    pass


def test_sympy__functions__elementary__hyperbolic__acosh():
    from sympy.functions.elementary.hyperbolic import acosh
    assert _test_args(acosh(2))


def test_sympy__functions__elementary__hyperbolic__acoth():
    from sympy.functions.elementary.hyperbolic import acoth
    assert _test_args(acoth(2))


def test_sympy__functions__elementary__hyperbolic__asinh():
    from sympy.functions.elementary.hyperbolic import asinh
    assert _test_args(asinh(2))


def test_sympy__functions__elementary__hyperbolic__atanh():
    from sympy.functions.elementary.hyperbolic import atanh
    assert _test_args(atanh(2))


def test_sympy__functions__elementary__hyperbolic__asech():
    from sympy.functions.elementary.hyperbolic import asech
    assert _test_args(asech(x))

def test_sympy__functions__elementary__hyperbolic__acsch():
    from sympy.functions.elementary.hyperbolic import acsch
    assert _test_args(acsch(x))

def test_sympy__functions__elementary__hyperbolic__cosh():
    from sympy.functions.elementary.hyperbolic import cosh
    assert _test_args(cosh(2))


def test_sympy__functions__elementary__hyperbolic__coth():
    from sympy.functions.elementary.hyperbolic import coth
    assert _test_args(coth(2))


def test_sympy__functions__elementary__hyperbolic__csch():
    from sympy.functions.elementary.hyperbolic import csch
    assert _test_args(csch(2))


def test_sympy__functions__elementary__hyperbolic__sech():
    from sympy.functions.elementary.hyperbolic import sech
    assert _test_args(sech(2))


def test_sympy__functions__elementary__hyperbolic__sinh():
    from sympy.functions.elementary.hyperbolic import sinh
    assert _test_args(sinh(2))


def test_sympy__functions__elementary__hyperbolic__tanh():
    from sympy.functions.elementary.hyperbolic import tanh
    assert _test_args(tanh(2))


@SKIP("abstract class")
def test_sympy__functions__elementary__integers__RoundFunction():
    pass


def test_sympy__functions__elementary__integers__ceiling():
    from sympy.functions.elementary.integers import ceiling
    assert _test_args(ceiling(x))


def test_sympy__functions__elementary__integers__floor():
    from sympy.functions.elementary.integers import floor
    assert _test_args(floor(x))


def test_sympy__functions__elementary__integers__frac():
    from sympy.functions.elementary.integers import frac
    assert _test_args(frac(x))


def test_sympy__functions__elementary__miscellaneous__IdentityFunction():
    from sympy.functions.elementary.miscellaneous import IdentityFunction
    assert _test_args(IdentityFunction())


def test_sympy__functions__elementary__miscellaneous__Max():
    from sympy.functions.elementary.miscellaneous import Max
    assert _test_args(Max(x, 2))


def test_sympy__functions__elementary__miscellaneous__Min():
    from sympy.functions.elementary.miscellaneous import Min
    assert _test_args(Min(x, 2))


@SKIP("abstract class")
def test_sympy__functions__elementary__miscellaneous__MinMaxBase():
    pass


def test_sympy__functions__elementary__miscellaneous__Rem():
    from sympy.functions.elementary.miscellaneous import Rem
    assert _test_args(Rem(x, 2))


def test_sympy__functions__elementary__piecewise__ExprCondPair():
    from sympy.functions.elementary.piecewise import ExprCondPair
    assert _test_args(ExprCondPair(1, True))


def test_sympy__functions__elementary__piecewise__Piecewise():
    from sympy.functions.elementary.piecewise import Piecewise
    assert _test_args(Piecewise((1, x >= 0), (0, True)))


@SKIP("abstract class")
def test_sympy__functions__elementary__trigonometric__TrigonometricFunction():
    pass

@SKIP("abstract class")
def test_sympy__functions__elementary__trigonometric__ReciprocalTrigonometricFunction():
    pass

@SKIP("abstract class")
def test_sympy__functions__elementary__trigonometric__InverseTrigonometricFunction():
    pass

def test_sympy__functions__elementary__trigonometric__acos():
    from sympy.functions.elementary.trigonometric import acos
    assert _test_args(acos(2))


def test_sympy__functions__elementary__trigonometric__acot():
    from sympy.functions.elementary.trigonometric import acot
    assert _test_args(acot(2))


def test_sympy__functions__elementary__trigonometric__asin():
    from sympy.functions.elementary.trigonometric import asin
    assert _test_args(asin(2))


def test_sympy__functions__elementary__trigonometric__asec():
    from sympy.functions.elementary.trigonometric import asec
    assert _test_args(asec(x))


def test_sympy__functions__elementary__trigonometric__acsc():
    from sympy.functions.elementary.trigonometric import acsc
    assert _test_args(acsc(x))


def test_sympy__functions__elementary__trigonometric__atan():
    from sympy.functions.elementary.trigonometric import atan
    assert _test_args(atan(2))


def test_sympy__functions__elementary__trigonometric__atan2():
    from sympy.functions.elementary.trigonometric import atan2
    assert _test_args(atan2(2, 3))


def test_sympy__functions__elementary__trigonometric__cos():
    from sympy.functions.elementary.trigonometric import cos
    assert _test_args(cos(2))


def test_sympy__functions__elementary__trigonometric__csc():
    from sympy.functions.elementary.trigonometric import csc
    assert _test_args(csc(2))


def test_sympy__functions__elementary__trigonometric__cot():
    from sympy.functions.elementary.trigonometric import cot
    assert _test_args(cot(2))


def test_sympy__functions__elementary__trigonometric__sin():
    assert _test_args(sin(2))


def test_sympy__functions__elementary__trigonometric__sinc():
    from sympy.functions.elementary.trigonometric import sinc
    assert _test_args(sinc(2))


def test_sympy__functions__elementary__trigonometric__sec():
    from sympy.functions.elementary.trigonometric import sec
    assert _test_args(sec(2))


def test_sympy__functions__elementary__trigonometric__tan():
    from sympy.functions.elementary.trigonometric import tan
    assert _test_args(tan(2))


@SKIP("abstract class")
def test_sympy__functions__special__bessel__BesselBase():
    pass


@SKIP("abstract class")
def test_sympy__functions__special__bessel__SphericalBesselBase():
    pass


@SKIP("abstract class")
def test_sympy__functions__special__bessel__SphericalHankelBase():
    pass


def test_sympy__functions__special__bessel__besseli():
    from sympy.functions.special.bessel import besseli
    assert _test_args(besseli(x, 1))


def test_sympy__functions__special__bessel__besselj():
    from sympy.functions.special.bessel import besselj
    assert _test_args(besselj(x, 1))


def test_sympy__functions__special__bessel__besselk():
    from sympy.functions.special.bessel import besselk
    assert _test_args(besselk(x, 1))


def test_sympy__functions__special__bessel__bessely():
    from sympy.functions.special.bessel import bessely
    assert _test_args(bessely(x, 1))


def test_sympy__functions__special__bessel__hankel1():
    from sympy.functions.special.bessel import hankel1
    assert _test_args(hankel1(x, 1))


def test_sympy__functions__special__bessel__hankel2():
    from sympy.functions.special.bessel import hankel2
    assert _test_args(hankel2(x, 1))


def test_sympy__functions__special__bessel__jn():
    from sympy.functions.special.bessel import jn
    assert _test_args(jn(0, x))


def test_sympy__functions__special__bessel__yn():
    from sympy.functions.special.bessel import yn
    assert _test_args(yn(0, x))


def test_sympy__functions__special__bessel__hn1():
    from sympy.functions.special.bessel import hn1
    assert _test_args(hn1(0, x))


def test_sympy__functions__special__bessel__hn2():
    from sympy.functions.special.bessel import hn2
    assert _test_args(hn2(0, x))


def test_sympy__functions__special__bessel__AiryBase():
    pass


def test_sympy__functions__special__bessel__airyai():
    from sympy.functions.special.bessel import airyai
    assert _test_args(airyai(2))


def test_sympy__functions__special__bessel__airybi():
    from sympy.functions.special.bessel import airybi
    assert _test_args(airybi(2))


def test_sympy__functions__special__bessel__airyaiprime():
    from sympy.functions.special.bessel import airyaiprime
    assert _test_args(airyaiprime(2))


def test_sympy__functions__special__bessel__airybiprime():
    from sympy.functions.special.bessel import airybiprime
    assert _test_args(airybiprime(2))


def test_sympy__functions__special__bessel__marcumq():
    from sympy.functions.special.bessel import marcumq
    assert _test_args(marcumq(x, y, z))


def test_sympy__functions__special__elliptic_integrals__elliptic_k():
    from sympy.functions.special.elliptic_integrals import elliptic_k as K
    assert _test_args(K(x))


def test_sympy__functions__special__elliptic_integrals__elliptic_f():
    from sympy.functions.special.elliptic_integrals import elliptic_f as F
    assert _test_args(F(x, y))


def test_sympy__functions__special__elliptic_integrals__elliptic_e():
    from sympy.functions.special.elliptic_integrals import elliptic_e as E
    assert _test_args(E(x))
    assert _test_args(E(x, y))


def test_sympy__functions__special__elliptic_integrals__elliptic_pi():
    from sympy.functions.special.elliptic_integrals import elliptic_pi as P
    assert _test_args(P(x, y))
    assert _test_args(P(x, y, z))


def test_sympy__functions__special__delta_functions__DiracDelta():
    from sympy.functions.special.delta_functions import DiracDelta
    assert _test_args(DiracDelta(x, 1))


def test_sympy__functions__special__singularity_functions__SingularityFunction():
    from sympy.functions.special.singularity_functions import SingularityFunction
    assert _test_args(SingularityFunction(x, y, z))


def test_sympy__functions__special__delta_functions__Heaviside():
    from sympy.functions.special.delta_functions import Heaviside
    assert _test_args(Heaviside(x))


def test_sympy__functions__special__error_functions__erf():
    from sympy.functions.special.error_functions import erf
    assert _test_args(erf(2))

def test_sympy__functions__special__error_functions__erfc():
    from sympy.functions.special.error_functions import erfc
    assert _test_args(erfc(2))

def test_sympy__functions__special__error_functions__erfi():
    from sympy.functions.special.error_functions import erfi
    assert _test_args(erfi(2))

def test_sympy__functions__special__error_functions__erf2():
    from sympy.functions.special.error_functions import erf2
    assert _test_args(erf2(2, 3))

def test_sympy__functions__special__error_functions__erfinv():
    from sympy.functions.special.error_functions import erfinv
    assert _test_args(erfinv(2))

def test_sympy__functions__special__error_functions__erfcinv():
    from sympy.functions.special.error_functions import erfcinv
    assert _test_args(erfcinv(2))

def test_sympy__functions__special__error_functions__erf2inv():
    from sympy.functions.special.error_functions import erf2inv
    assert _test_args(erf2inv(2, 3))

@SKIP("abstract class")
def test_sympy__functions__special__error_functions__FresnelIntegral():
    pass


def test_sympy__functions__special__error_functions__fresnels():
    from sympy.functions.special.error_functions import fresnels
    assert _test_args(fresnels(2))


def test_sympy__functions__special__error_functions__fresnelc():
    from sympy.functions.special.error_functions import fresnelc
    assert _test_args(fresnelc(2))


def test_sympy__functions__special__error_functions__erfs():
    from sympy.functions.special.error_functions import _erfs
    assert _test_args(_erfs(2))


def test_sympy__functions__special__error_functions__Ei():
    from sympy.functions.special.error_functions import Ei
    assert _test_args(Ei(2))


def test_sympy__functions__special__error_functions__li():
    from sympy.functions.special.error_functions import li
    assert _test_args(li(2))


def test_sympy__functions__special__error_functions__Li():
    from sympy.functions.special.error_functions import Li
    assert _test_args(Li(5))


@SKIP("abstract class")
def test_sympy__functions__special__error_functions__TrigonometricIntegral():
    pass


def test_sympy__functions__special__error_functions__Si():
    from sympy.functions.special.error_functions import Si
    assert _test_args(Si(2))


def test_sympy__functions__special__error_functions__Ci():
    from sympy.functions.special.error_functions import Ci
    assert _test_args(Ci(2))


def test_sympy__functions__special__error_functions__Shi():
    from sympy.functions.special.error_functions import Shi
    assert _test_args(Shi(2))


def test_sympy__functions__special__error_functions__Chi():
    from sympy.functions.special.error_functions import Chi
    assert _test_args(Chi(2))


def test_sympy__functions__special__error_functions__expint():
    from sympy.functions.special.error_functions import expint
    assert _test_args(expint(y, x))


def test_sympy__functions__special__gamma_functions__gamma():
    from sympy.functions.special.gamma_functions import gamma
    assert _test_args(gamma(x))


def test_sympy__functions__special__gamma_functions__loggamma():
    from sympy.functions.special.gamma_functions import loggamma
    assert _test_args(loggamma(x))


def test_sympy__functions__special__gamma_functions__lowergamma():
    from sympy.functions.special.gamma_functions import lowergamma
    assert _test_args(lowergamma(x, 2))


def test_sympy__functions__special__gamma_functions__polygamma():
    from sympy.functions.special.gamma_functions import polygamma
    assert _test_args(polygamma(x, 2))

def test_sympy__functions__special__gamma_functions__digamma():
    from sympy.functions.special.gamma_functions import digamma
    assert _test_args(digamma(x))

def test_sympy__functions__special__gamma_functions__trigamma():
    from sympy.functions.special.gamma_functions import trigamma
    assert _test_args(trigamma(x))

def test_sympy__functions__special__gamma_functions__uppergamma():
    from sympy.functions.special.gamma_functions import uppergamma
    assert _test_args(uppergamma(x, 2))

def test_sympy__functions__special__gamma_functions__multigamma():
    from sympy.functions.special.gamma_functions import multigamma
    assert _test_args(multigamma(x, 1))


def test_sympy__functions__special__beta_functions__beta():
    from sympy.functions.special.beta_functions import beta
    assert _test_args(beta(x))
    assert _test_args(beta(x, x))

def test_sympy__functions__special__beta_functions__betainc():
    from sympy.functions.special.beta_functions import betainc
    assert _test_args(betainc(a, b, x, y))

def test_sympy__functions__special__beta_functions__betainc_regularized():
    from sympy.functions.special.beta_functions import betainc_regularized
    assert _test_args(betainc_regularized(a, b, x, y))


def test_sympy__functions__special__mathieu_functions__MathieuBase():
    pass


def test_sympy__functions__special__mathieu_functions__mathieus():
    from sympy.functions.special.mathieu_functions import mathieus
    assert _test_args(mathieus(1, 1, 1))


def test_sympy__functions__special__mathieu_functions__mathieuc():
    from sympy.functions.special.mathieu_functions import mathieuc
    assert _test_args(mathieuc(1, 1, 1))


def test_sympy__functions__special__mathieu_functions__mathieusprime():
    from sympy.functions.special.mathieu_functions import mathieusprime
    assert _test_args(mathieusprime(1, 1, 1))


def test_sympy__functions__special__mathieu_functions__mathieucprime():
    from sympy.functions.special.mathieu_functions import mathieucprime
    assert _test_args(mathieucprime(1, 1, 1))


@SKIP("abstract class")
def test_sympy__functions__special__hyper__TupleParametersBase():
    pass


@SKIP("abstract class")
def test_sympy__functions__special__hyper__TupleArg():
    pass


def test_sympy__functions__special__hyper__hyper():
    from sympy.functions.special.hyper import hyper
    assert _test_args(hyper([1, 2, 3], [4, 5], x))


def test_sympy__functions__special__hyper__meijerg():
    from sympy.functions.special.hyper import meijerg
    assert _test_args(meijerg([1, 2, 3], [4, 5], [6], [], x))


@SKIP("abstract class")
def test_sympy__functions__special__hyper__HyperRep():
    pass


def test_sympy__functions__special__hyper__HyperRep_power1():
    from sympy.functions.special.hyper import HyperRep_power1
    assert _test_args(HyperRep_power1(x, y))


def test_sympy__functions__special__hyper__HyperRep_power2():
    from sympy.functions.special.hyper import HyperRep_power2
    assert _test_args(HyperRep_power2(x, y))


def test_sympy__functions__special__hyper__HyperRep_log1():
    from sympy.functions.special.hyper import HyperRep_log1
    assert _test_args(HyperRep_log1(x))


def test_sympy__functions__special__hyper__HyperRep_atanh():
    from sympy.functions.special.hyper import HyperRep_atanh
    assert _test_args(HyperRep_atanh(x))


def test_sympy__functions__special__hyper__HyperRep_asin1():
    from sympy.functions.special.hyper import HyperRep_asin1
    assert _test_args(HyperRep_asin1(x))


def test_sympy__functions__special__hyper__HyperRep_asin2():
    from sympy.functions.special.hyper import HyperRep_asin2
    assert _test_args(HyperRep_asin2(x))


def test_sympy__functions__special__hyper__HyperRep_sqrts1():
    from sympy.functions.special.hyper import HyperRep_sqrts1
    assert _test_args(HyperRep_sqrts1(x, y))


def test_sympy__functions__special__hyper__HyperRep_sqrts2():
    from sympy.functions.special.hyper import HyperRep_sqrts2
    assert _test_args(HyperRep_sqrts2(x, y))


def test_sympy__functions__special__hyper__HyperRep_log2():
    from sympy.functions.special.hyper import HyperRep_log2
    assert _test_args(HyperRep_log2(x))


def test_sympy__functions__special__hyper__HyperRep_cosasin():
    from sympy.functions.special.hyper import HyperRep_cosasin
    assert _test_args(HyperRep_cosasin(x, y))


def test_sympy__functions__special__hyper__HyperRep_sinasin():
    from sympy.functions.special.hyper import HyperRep_sinasin
    assert _test_args(HyperRep_sinasin(x, y))

def test_sympy__functions__special__hyper__appellf1():
    from sympy.functions.special.hyper import appellf1
    a, b1, b2, c, x, y = symbols('a b1 b2 c x y')
    assert _test_args(appellf1(a, b1, b2, c, x, y))

@SKIP("abstract class")
def test_sympy__functions__special__polynomials__OrthogonalPolynomial():
    pass


def test_sympy__functions__special__polynomials__jacobi():
    from sympy.functions.special.polynomials import jacobi
    assert _test_args(jacobi(x, y, 2, 2))


def test_sympy__functions__special__polynomials__gegenbauer():
    from sympy.functions.special.polynomials import gegenbauer
    assert _test_args(gegenbauer(x, 2, 2))


def test_sympy__functions__special__polynomials__chebyshevt():
    from sympy.functions.special.polynomials import chebyshevt
    assert _test_args(chebyshevt(x, 2))


def test_sympy__functions__special__polynomials__chebyshevt_root():
    from sympy.functions.special.polynomials import chebyshevt_root
    assert _test_args(chebyshevt_root(3, 2))


def test_sympy__functions__special__polynomials__chebyshevu():
    from sympy.functions.special.polynomials import chebyshevu
    assert _test_args(chebyshevu(x, 2))


def test_sympy__functions__special__polynomials__chebyshevu_root():
    from sympy.functions.special.polynomials import chebyshevu_root
    assert _test_args(chebyshevu_root(3, 2))


def test_sympy__functions__special__polynomials__hermite():
    from sympy.functions.special.polynomials import hermite
    assert _test_args(hermite(x, 2))


def test_sympy__functions__special__polynomials__hermite_prob():
    from sympy.functions.special.polynomials import hermite_prob
    assert _test_args(hermite_prob(x, 2))


def test_sympy__functions__special__polynomials__legendre():
    from sympy.functions.special.polynomials import legendre
    assert _test_args(legendre(x, 2))


def test_sympy__functions__special__polynomials__assoc_legendre():
    from sympy.functions.special.polynomials import assoc_legendre
    assert _test_args(assoc_legendre(x, 0, y))


def test_sympy__functions__special__polynomials__laguerre():
    from sympy.functions.special.polynomials import laguerre
    assert _test_args(laguerre(x, 2))


def test_sympy__functions__special__polynomials__assoc_laguerre():
    from sympy.functions.special.polynomials import assoc_laguerre
    assert _test_args(assoc_laguerre(x, 0, y))


def test_sympy__functions__special__spherical_harmonics__Ynm():
    from sympy.functions.special.spherical_harmonics import Ynm
    assert _test_args(Ynm(1, 1, x, y))


def test_sympy__functions__special__spherical_harmonics__Znm():
    from sympy.functions.special.spherical_harmonics import Znm
    assert _test_args(Znm(x, y, 1, 1))


def test_sympy__functions__special__tensor_functions__LeviCivita():
    from sympy.functions.special.tensor_functions import LeviCivita
    assert _test_args(LeviCivita(x, y, 2))


def test_sympy__functions__special__tensor_functions__KroneckerDelta():
    from sympy.functions.special.tensor_functions import KroneckerDelta
    assert _test_args(KroneckerDelta(x, y))


def test_sympy__functions__special__zeta_functions__dirichlet_eta():
    from sympy.functions.special.zeta_functions import dirichlet_eta
    assert _test_args(dirichlet_eta(x))


def test_sympy__functions__special__zeta_functions__riemann_xi():
    from sympy.functions.special.zeta_functions import riemann_xi
    assert _test_args(riemann_xi(x))


def test_sympy__functions__special__zeta_functions__zeta():
    from sympy.functions.special.zeta_functions import zeta
    assert _test_args(zeta(101))


def test_sympy__functions__special__zeta_functions__lerchphi():
    from sympy.functions.special.zeta_functions import lerchphi
    assert _test_args(lerchphi(x, y, z))


def test_sympy__functions__special__zeta_functions__polylog():
    from sympy.functions.special.zeta_functions import polylog
    assert _test_args(polylog(x, y))


def test_sympy__functions__special__zeta_functions__stieltjes():
    from sympy.functions.special.zeta_functions import stieltjes
    assert _test_args(stieltjes(x, y))


def test_sympy__integrals__integrals__Integral():
    from sympy.integrals.integrals import Integral
    assert _test_args(Integral(2, (x, 0, 1)))


def test_sympy__integrals__risch__NonElementaryIntegral():
    from sympy.integrals.risch import NonElementaryIntegral
    assert _test_args(NonElementaryIntegral(exp(-x**2), x))


@SKIP("abstract class")
def test_sympy__integrals__transforms__IntegralTransform():
    pass


def test_sympy__integrals__transforms__MellinTransform():
    from sympy.integrals.transforms import MellinTransform
    assert _test_args(MellinTransform(2, x, y))


def test_sympy__integrals__transforms__InverseMellinTransform():
    from sympy.integrals.transforms import InverseMellinTransform
    assert _test_args(InverseMellinTransform(2, x, y, 0, 1))


def test_sympy__integrals__laplace__LaplaceTransform():
    from sympy.integrals.laplace import LaplaceTransform
    assert _test_args(LaplaceTransform(2, x, y))


def test_sympy__integrals__laplace__InverseLaplaceTransform():
    from sympy.integrals.laplace import InverseLaplaceTransform
    assert _test_args(InverseLaplaceTransform(2, x, y, 0))


@SKIP("abstract class")
def test_sympy__integrals__transforms__FourierTypeTransform():
    pass


def test_sympy__integrals__transforms__InverseFourierTransform():
    from sympy.integrals.transforms import InverseFourierTransform
    assert _test_args(InverseFourierTransform(2, x, y))


def test_sympy__integrals__transforms__FourierTransform():
    from sympy.integrals.transforms import FourierTransform
    assert _test_args(FourierTransform(2, x, y))


@SKIP("abstract class")
def test_sympy__integrals__transforms__SineCosineTypeTransform():
    pass


def test_sympy__integrals__transforms__InverseSineTransform():
    from sympy.integrals.transforms import InverseSineTransform
    assert _test_args(InverseSineTransform(2, x, y))


def test_sympy__integrals__transforms__SineTransform():
    from sympy.integrals.transforms import SineTransform
    assert _test_args(SineTransform(2, x, y))


def test_sympy__integrals__transforms__InverseCosineTransform():
    from sympy.integrals.transforms import InverseCosineTransform
    assert _test_args(InverseCosineTransform(2, x, y))


def test_sympy__integrals__transforms__CosineTransform():
    from sympy.integrals.transforms import CosineTransform
    assert _test_args(CosineTransform(2, x, y))


@SKIP("abstract class")
def test_sympy__integrals__transforms__HankelTypeTransform():
    pass


def test_sympy__integrals__transforms__InverseHankelTransform():
    from sympy.integrals.transforms import InverseHankelTransform
    assert _test_args(InverseHankelTransform(2, x, y, 0))


def test_sympy__integrals__transforms__HankelTransform():
    from sympy.integrals.transforms import HankelTransform
    assert _test_args(HankelTransform(2, x, y, 0))


def test_sympy__liealgebras__cartan_type__Standard_Cartan():
    from sympy.liealgebras.cartan_type import Standard_Cartan
    assert _test_args(Standard_Cartan("A", 2))

def test_sympy__liealgebras__weyl_group__WeylGroup():
    from sympy.liealgebras.weyl_group import WeylGroup
    assert _test_args(WeylGroup("B4"))

def test_sympy__liealgebras__root_system__RootSystem():
    from sympy.liealgebras.root_system import RootSystem
    assert _test_args(RootSystem("A2"))

def test_sympy__liealgebras__type_a__TypeA():
    from sympy.liealgebras.type_a import TypeA
    assert _test_args(TypeA(2))

def test_sympy__liealgebras__type_b__TypeB():
    from sympy.liealgebras.type_b import TypeB
    assert _test_args(TypeB(4))

def test_sympy__liealgebras__type_c__TypeC():
    from sympy.liealgebras.type_c import TypeC
    assert _test_args(TypeC(4))

def test_sympy__liealgebras__type_d__TypeD():
    from sympy.liealgebras.type_d import TypeD
    assert _test_args(TypeD(4))

def test_sympy__liealgebras__type_e__TypeE():
    from sympy.liealgebras.type_e import TypeE
    assert _test_args(TypeE(6))

def test_sympy__liealgebras__type_f__TypeF():
    from sympy.liealgebras.type_f import TypeF
    assert _test_args(TypeF(4))

def test_sympy__liealgebras__type_g__TypeG():
    from sympy.liealgebras.type_g import TypeG
    assert _test_args(TypeG(2))


def test_sympy__logic__boolalg__And():
    from sympy.logic.boolalg import And
    assert _test_args(And(x, y, 1))


@SKIP("abstract class")
def test_sympy__logic__boolalg__Boolean():
    pass


def test_sympy__logic__boolalg__BooleanFunction():
    from sympy.logic.boolalg import BooleanFunction
    assert _test_args(BooleanFunction(1, 2, 3))

@SKIP("abstract class")
def test_sympy__logic__boolalg__BooleanAtom():
    pass

def test_sympy__logic__boolalg__BooleanTrue():
    from sympy.logic.boolalg import true
    assert _test_args(true)

def test_sympy__logic__boolalg__BooleanFalse():
    from sympy.logic.boolalg import false
    assert _test_args(false)

def test_sympy__logic__boolalg__Equivalent():
    from sympy.logic.boolalg import Equivalent
    assert _test_args(Equivalent(x, 2))


def test_sympy__logic__boolalg__ITE():
    from sympy.logic.boolalg import ITE
    assert _test_args(ITE(x, y, 1))


def test_sympy__logic__boolalg__Implies():
    from sympy.logic.boolalg import Implies
    assert _test_args(Implies(x, y))


def test_sympy__logic__boolalg__Nand():
    from sympy.logic.boolalg import Nand
    assert _test_args(Nand(x, y, 1))


def test_sympy__logic__boolalg__Nor():
    from sympy.logic.boolalg import Nor
    assert _test_args(Nor(x, y))


def test_sympy__logic__boolalg__Not():
    from sympy.logic.boolalg import Not
    assert _test_args(Not(x))


def test_sympy__logic__boolalg__Or():
    from sympy.logic.boolalg import Or
    assert _test_args(Or(x, y))


def test_sympy__logic__boolalg__Xor():
    from sympy.logic.boolalg import Xor
    assert _test_args(Xor(x, y, 2))

def test_sympy__logic__boolalg__Xnor():
    from sympy.logic.boolalg import Xnor
    assert _test_args(Xnor(x, y, 2))

def test_sympy__logic__boolalg__Exclusive():
    from sympy.logic.boolalg import Exclusive
    assert _test_args(Exclusive(x, y, z))


def test_sympy__matrices__matrixbase__DeferredVector():
    from sympy.matrices.matrixbase import DeferredVector
    assert _test_args(DeferredVector("X"))


@SKIP("abstract class")
def test_sympy__matrices__expressions__matexpr__MatrixBase():
    pass


@SKIP("abstract class")
def test_sympy__matrices__immutable__ImmutableRepMatrix():
    pass


def test_sympy__matrices__immutable__ImmutableDenseMatrix():
    from sympy.matrices.immutable import ImmutableDenseMatrix
    m = ImmutableDenseMatrix([[1, 2], [3, 4]])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableDenseMatrix(1, 1, [1])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableDenseMatrix(2, 2, lambda i, j: 1)
    assert m[0, 0] is S.One
    m = ImmutableDenseMatrix(2, 2, lambda i, j: 1/(1 + i) + 1/(1 + j))
    assert m[1, 1] is S.One  # true div. will give 1.0 if i,j not sympified
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))


def test_sympy__matrices__immutable__ImmutableSparseMatrix():
    from sympy.matrices.immutable import ImmutableSparseMatrix
    m = ImmutableSparseMatrix([[1, 2], [3, 4]])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableSparseMatrix(1, 1, {(0, 0): 1})
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableSparseMatrix(1, 1, [1])
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))
    m = ImmutableSparseMatrix(2, 2, lambda i, j: 1)
    assert m[0, 0] is S.One
    m = ImmutableSparseMatrix(2, 2, lambda i, j: 1/(1 + i) + 1/(1 + j))
    assert m[1, 1] is S.One  # true div. will give 1.0 if i,j not sympified
    assert _test_args(m)
    assert _test_args(Basic(*list(m)))


def test_sympy__matrices__expressions__slice__MatrixSlice():
    from sympy.matrices.expressions.slice import MatrixSlice
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', 4, 4)
    assert _test_args(MatrixSlice(X, (0, 2), (0, 2)))


def test_sympy__matrices__expressions__applyfunc__ElementwiseApplyFunction():
    from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol("X", x, x)
    func = Lambda(x, x**2)
    assert _test_args(ElementwiseApplyFunction(func, X))


def test_sympy__matrices__expressions__blockmatrix__BlockDiagMatrix():
    from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, x)
    Y = MatrixSymbol('Y', y, y)
    assert _test_args(BlockDiagMatrix(X, Y))


def test_sympy__matrices__expressions__blockmatrix__BlockMatrix():
    from sympy.matrices.expressions.blockmatrix import BlockMatrix
    from sympy.matrices.expressions import MatrixSymbol, ZeroMatrix
    X = MatrixSymbol('X', x, x)
    Y = MatrixSymbol('Y', y, y)
    Z = MatrixSymbol('Z', x, y)
    O = ZeroMatrix(y, x)
    assert _test_args(BlockMatrix([[X, Z], [O, Y]]))


def test_sympy__matrices__expressions__inverse__Inverse():
    from sympy.matrices.expressions.inverse import Inverse
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Inverse(MatrixSymbol('A', 3, 3)))


def test_sympy__matrices__expressions__matadd__MatAdd():
    from sympy.matrices.expressions.matadd import MatAdd
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, y)
    Y = MatrixSymbol('Y', x, y)
    assert _test_args(MatAdd(X, Y))


@SKIP("abstract class")
def test_sympy__matrices__expressions__matexpr__MatrixExpr():
    pass

def test_sympy__matrices__expressions__matexpr__MatrixElement():
    from sympy.matrices.expressions.matexpr import MatrixSymbol, MatrixElement
    from sympy.core.singleton import S
    assert _test_args(MatrixElement(MatrixSymbol('A', 3, 5), S(2), S(3)))

def test_sympy__matrices__expressions__matexpr__MatrixSymbol():
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    assert _test_args(MatrixSymbol('A', 3, 5))


def test_sympy__matrices__expressions__special__OneMatrix():
    from sympy.matrices.expressions.special import OneMatrix
    assert _test_args(OneMatrix(3, 5))


def test_sympy__matrices__expressions__special__ZeroMatrix():
    from sympy.matrices.expressions.special import ZeroMatrix
    assert _test_args(ZeroMatrix(3, 5))


def test_sympy__matrices__expressions__special__GenericZeroMatrix():
    from sympy.matrices.expressions.special import GenericZeroMatrix
    assert _test_args(GenericZeroMatrix())


def test_sympy__matrices__expressions__special__Identity():
    from sympy.matrices.expressions.special import Identity
    assert _test_args(Identity(3))


def test_sympy__matrices__expressions__special__GenericIdentity():
    from sympy.matrices.expressions.special import GenericIdentity
    assert _test_args(GenericIdentity())


def test_sympy__matrices__expressions__sets__MatrixSet():
    from sympy.matrices.expressions.sets import MatrixSet
    from sympy.core.singleton import S
    assert _test_args(MatrixSet(2, 2, S.Reals))

def test_sympy__matrices__expressions__matmul__MatMul():
    from sympy.matrices.expressions.matmul import MatMul
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, y)
    Y = MatrixSymbol('Y', y, x)
    assert _test_args(MatMul(X, Y))


def test_sympy__matrices__expressions__dotproduct__DotProduct():
    from sympy.matrices.expressions.dotproduct import DotProduct
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, 1)
    Y = MatrixSymbol('Y', x, 1)
    assert _test_args(DotProduct(X, Y))

def test_sympy__matrices__expressions__diagonal__DiagonalMatrix():
    from sympy.matrices.expressions.diagonal import DiagonalMatrix
    from sympy.matrices.expressions import MatrixSymbol
    x = MatrixSymbol('x', 10, 1)
    assert _test_args(DiagonalMatrix(x))

def test_sympy__matrices__expressions__diagonal__DiagonalOf():
    from sympy.matrices.expressions.diagonal import DiagonalOf
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('x', 10, 10)
    assert _test_args(DiagonalOf(X))

def test_sympy__matrices__expressions__diagonal__DiagMatrix():
    from sympy.matrices.expressions.diagonal import DiagMatrix
    from sympy.matrices.expressions import MatrixSymbol
    x = MatrixSymbol('x', 10, 1)
    assert _test_args(DiagMatrix(x))

def test_sympy__matrices__expressions__hadamard__HadamardProduct():
    from sympy.matrices.expressions.hadamard import HadamardProduct
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, y)
    Y = MatrixSymbol('Y', x, y)
    assert _test_args(HadamardProduct(X, Y))

def test_sympy__matrices__expressions__hadamard__HadamardPower():
    from sympy.matrices.expressions.hadamard import HadamardPower
    from sympy.matrices.expressions import MatrixSymbol
    from sympy.core.symbol import Symbol
    X = MatrixSymbol('X', x, y)
    n = Symbol("n")
    assert _test_args(HadamardPower(X, n))

def test_sympy__matrices__expressions__kronecker__KroneckerProduct():
    from sympy.matrices.expressions.kronecker import KroneckerProduct
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, y)
    Y = MatrixSymbol('Y', x, y)
    assert _test_args(KroneckerProduct(X, Y))


def test_sympy__matrices__expressions__matpow__MatPow():
    from sympy.matrices.expressions.matpow import MatPow
    from sympy.matrices.expressions import MatrixSymbol
    X = MatrixSymbol('X', x, x)
    assert _test_args(MatPow(X, 2))


def test_sympy__matrices__expressions__transpose__Transpose():
    from sympy.matrices.expressions.transpose import Transpose
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Transpose(MatrixSymbol('A', 3, 5)))


def test_sympy__matrices__expressions__adjoint__Adjoint():
    from sympy.matrices.expressions.adjoint import Adjoint
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Adjoint(MatrixSymbol('A', 3, 5)))


def test_sympy__matrices__expressions__trace__Trace():
    from sympy.matrices.expressions.trace import Trace
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Trace(MatrixSymbol('A', 3, 3)))

def test_sympy__matrices__expressions__determinant__Determinant():
    from sympy.matrices.expressions.determinant import Determinant
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Determinant(MatrixSymbol('A', 3, 3)))

def test_sympy__matrices__expressions__determinant__Permanent():
    from sympy.matrices.expressions.determinant import Permanent
    from sympy.matrices.expressions import MatrixSymbol
    assert _test_args(Permanent(MatrixSymbol('A', 3, 4)))

def test_sympy__matrices__expressions__funcmatrix__FunctionMatrix():
    from sympy.matrices.expressions.funcmatrix import FunctionMatrix
    from sympy.core.symbol import symbols
    i, j = symbols('i,j')
    assert _test_args(FunctionMatrix(3, 3, Lambda((i, j), i - j) ))

def test_sympy__matrices__expressions__fourier__DFT():
    from sympy.matrices.expressions.fourier import DFT
    from sympy.core.singleton import S
    assert _test_args(DFT(S(2)))

def test_sympy__matrices__expressions__fourier__IDFT():
    from sympy.matrices.expressions.fourier import IDFT
    from sympy.core.singleton import S
    assert _test_args(IDFT(S(2)))

from sympy.matrices.expressions import MatrixSymbol
X = MatrixSymbol('X', 10, 10)

def test_sympy__matrices__expressions__factorizations__LofLU():
    from sympy.matrices.expressions.factorizations import LofLU
    assert _test_args(LofLU(X))

def test_sympy__matrices__expressions__factorizations__UofLU():
    from sympy.matrices.expressions.factorizations import UofLU
    assert _test_args(UofLU(X))

def test_sympy__matrices__expressions__factorizations__QofQR():
    from sympy.matrices.expressions.factorizations import QofQR
    assert _test_args(QofQR(X))

def test_sympy__matrices__expressions__factorizations__RofQR():
    from sympy.matrices.expressions.factorizations import RofQR
    assert _test_args(RofQR(X))

def test_sympy__matrices__expressions__factorizations__LofCholesky():
    from sympy.matrices.expressions.factorizations import LofCholesky
    assert _test_args(LofCholesky(X))

def test_sympy__matrices__expressions__factorizations__UofCholesky():
    from sympy.matrices.expressions.factorizations import UofCholesky
    assert _test_args(UofCholesky(X))

def test_sympy__matrices__expressions__factorizations__EigenVectors():
    from sympy.matrices.expressions.factorizations import EigenVectors
    assert _test_args(EigenVectors(X))

def test_sympy__matrices__expressions__factorizations__EigenValues():
    from sympy.matrices.expressions.factorizations import EigenValues
    assert _test_args(EigenValues(X))

def test_sympy__matrices__expressions__factorizations__UofSVD():
    from sympy.matrices.expressions.factorizations import UofSVD
    assert _test_args(UofSVD(X))

def test_sympy__matrices__expressions__factorizations__VofSVD():
    from sympy.matrices.expressions.factorizations import VofSVD
    assert _test_args(VofSVD(X))

def test_sympy__matrices__expressions__factorizations__SofSVD():
    from sympy.matrices.expressions.factorizations import SofSVD
    assert _test_args(SofSVD(X))

@SKIP("abstract class")
def test_sympy__matrices__expressions__factorizations__Factorization():
    pass

def test_sympy__matrices__expressions__permutation__PermutationMatrix():
    from sympy.combinatorics import Permutation
    from sympy.matrices.expressions.permutation import PermutationMatrix
    assert _test_args(PermutationMatrix(Permutation([2, 0, 1])))

def test_sympy__matrices__expressions__permutation__MatrixPermute():
    from sympy.combinatorics import Permutation
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    from sympy.matrices.expressions.permutation import MatrixPermute
    A = MatrixSymbol('A', 3, 3)
    assert _test_args(MatrixPermute(A, Permutation([2, 0, 1])))

def test_sympy__matrices__expressions__companion__CompanionMatrix():
    from sympy.core.symbol import Symbol
    from sympy.matrices.expressions.companion import CompanionMatrix
    from sympy.polys.polytools import Poly

    x = Symbol('x')
    p = Poly([1, 2, 3], x)
    assert _test_args(CompanionMatrix(p))

def test_sympy__physics__vector__frame__CoordinateSym():
    from sympy.physics.vector import CoordinateSym
    from sympy.physics.vector import ReferenceFrame
    assert _test_args(CoordinateSym('R_x', ReferenceFrame('R'), 0))


@SKIP("abstract class")
def test_sympy__physics__biomechanics__curve__CharacteristicCurveFunction():
    pass


def test_sympy__physics__biomechanics__curve__TendonForceLengthDeGroote2016():
    from sympy.physics.biomechanics import TendonForceLengthDeGroote2016
    l_T_tilde, c0, c1, c2, c3 = symbols('l_T_tilde, c0, c1, c2, c3')
    assert _test_args(TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3))


def test_sympy__physics__biomechanics__curve__TendonForceLengthInverseDeGroote2016():
    from sympy.physics.biomechanics import TendonForceLengthInverseDeGroote2016
    fl_T, c0, c1, c2, c3 = symbols('fl_T, c0, c1, c2, c3')
    assert _test_args(TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3))


def test_sympy__physics__biomechanics__curve__FiberForceLengthPassiveDeGroote2016():
    from sympy.physics.biomechanics import FiberForceLengthPassiveDeGroote2016
    l_M_tilde, c0, c1 = symbols('l_M_tilde, c0, c1')
    assert _test_args(FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1))


def test_sympy__physics__biomechanics__curve__FiberForceLengthPassiveInverseDeGroote2016():
    from sympy.physics.biomechanics import FiberForceLengthPassiveInverseDeGroote2016
    fl_M_pas, c0, c1 = symbols('fl_M_pas, c0, c1')
    assert _test_args(FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1))


def test_sympy__physics__biomechanics__curve__FiberForceLengthActiveDeGroote2016():
    from sympy.physics.biomechanics import FiberForceLengthActiveDeGroote2016
    l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = symbols('l_M_tilde, c0:12')
    assert _test_args(FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11))


def test_sympy__physics__biomechanics__curve__FiberForceVelocityDeGroote2016():
    from sympy.physics.biomechanics import FiberForceVelocityDeGroote2016
    v_M_tilde, c0, c1, c2, c3 = symbols('v_M_tilde, c0, c1, c2, c3')
    assert _test_args(FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3))


def test_sympy__physics__biomechanics__curve__FiberForceVelocityInverseDeGroote2016():
    from sympy.physics.biomechanics import FiberForceVelocityInverseDeGroote2016
    fv_M, c0, c1, c2, c3 = symbols('fv_M, c0, c1, c2, c3')
    assert _test_args(FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3))


def test_sympy__physics__paulialgebra__Pauli():
    from sympy.physics.paulialgebra import Pauli
    assert _test_args(Pauli(1))


def test_sympy__physics__quantum__anticommutator__AntiCommutator():
    from sympy.physics.quantum.anticommutator import AntiCommutator
    assert _test_args(AntiCommutator(x, y))


def test_sympy__physics__quantum__cartesian__PositionBra3D():
    from sympy.physics.quantum.cartesian import PositionBra3D
    assert _test_args(PositionBra3D(x, y, z))


def test_sympy__physics__quantum__cartesian__PositionKet3D():
    from sympy.physics.quantum.cartesian import PositionKet3D
    assert _test_args(PositionKet3D(x, y, z))


def test_sympy__physics__quantum__cartesian__PositionState3D():
    from sympy.physics.quantum.cartesian import PositionState3D
    assert _test_args(PositionState3D(x, y, z))


def test_sympy__physics__quantum__cartesian__PxBra():
    from sympy.physics.quantum.cartesian import PxBra
    assert _test_args(PxBra(x, y, z))


def test_sympy__physics__quantum__cartesian__PxKet():
    from sympy.physics.quantum.cartesian import PxKet
    assert _test_args(PxKet(x, y, z))


def test_sympy__physics__quantum__cartesian__PxOp():
    from sympy.physics.quantum.cartesian import PxOp
    assert _test_args(PxOp(x, y, z))


def test_sympy__physics__quantum__cartesian__XBra():
    from sympy.physics.quantum.cartesian import XBra
    assert _test_args(XBra(x))


def test_sympy__physics__quantum__cartesian__XKet():
    from sympy.physics.quantum.cartesian import XKet
    assert _test_args(XKet(x))


def test_sympy__physics__quantum__cartesian__XOp():
    from sympy.physics.quantum.cartesian import XOp
    assert _test_args(XOp(x))


def test_sympy__physics__quantum__cartesian__YOp():
    from sympy.physics.quantum.cartesian import YOp
    assert _test_args(YOp(x))


def test_sympy__physics__quantum__cartesian__ZOp():
    from sympy.physics.quantum.cartesian import ZOp
    assert _test_args(ZOp(x))


def test_sympy__physics__quantum__cg__CG():
    from sympy.physics.quantum.cg import CG
    from sympy.core.singleton import S
    assert _test_args(CG(Rational(3, 2), Rational(3, 2), S.Half, Rational(-1, 2), 1, 1))


def test_sympy__physics__quantum__cg__Wigner3j():
    from sympy.physics.quantum.cg import Wigner3j
    assert _test_args(Wigner3j(6, 0, 4, 0, 2, 0))


def test_sympy__physics__quantum__cg__Wigner6j():
    from sympy.physics.quantum.cg import Wigner6j
    assert _test_args(Wigner6j(1, 2, 3, 2, 1, 2))


def test_sympy__physics__quantum__cg__Wigner9j():
    from sympy.physics.quantum.cg import Wigner9j
    assert _test_args(Wigner9j(2, 1, 1, Rational(3, 2), S.Half, 1, S.Half, S.Half, 0))

def test_sympy__physics__quantum__circuitplot__Mz():
    from sympy.physics.quantum.circuitplot import Mz
    assert _test_args(Mz(0))

def test_sympy__physics__quantum__circuitplot__Mx():
    from sympy.physics.quantum.circuitplot import Mx
    assert _test_args(Mx(0))

def test_sympy__physics__quantum__commutator__Commutator():
    from sympy.physics.quantum.commutator import Commutator
    A, B = symbols('A,B', commutative=False)
    assert _test_args(Commutator(A, B))


def test_sympy__physics__quantum__constants__HBar():
    from sympy.physics.quantum.constants import HBar
    assert _test_args(HBar())


def test_sympy__physics__quantum__dagger__Dagger():
    from sympy.physics.quantum.dagger import Dagger
    from sympy.physics.quantum.state import Ket
    assert _test_args(Dagger(Dagger(Ket('psi'))))


def test_sympy__physics__quantum__gate__CGate():
    from sympy.physics.quantum.gate import CGate, Gate
    assert _test_args(CGate((0, 1), Gate(2)))


def test_sympy__physics__quantum__gate__CGateS():
    from sympy.physics.quantum.gate import CGateS, Gate
    assert _test_args(CGateS((0, 1), Gate(2)))


def test_sympy__physics__quantum__gate__CNotGate():
    from sympy.physics.quantum.gate import CNotGate
    assert _test_args(CNotGate(0, 1))


def test_sympy__physics__quantum__gate__Gate():
    from sympy.physics.quantum.gate import Gate
    assert _test_args(Gate(0))


def test_sympy__physics__quantum__gate__HadamardGate():
    from sympy.physics.quantum.gate import HadamardGate
    assert _test_args(HadamardGate(0))


def test_sympy__physics__quantum__gate__IdentityGate():
    from sympy.physics.quantum.gate import IdentityGate
    assert _test_args(IdentityGate(0))


def test_sympy__physics__quantum__gate__OneQubitGate():
    from sympy.physics.quantum.gate import OneQubitGate
    assert _test_args(OneQubitGate(0))


def test_sympy__physics__quantum__gate__PhaseGate():
    from sympy.physics.quantum.gate import PhaseGate
    assert _test_args(PhaseGate(0))


def test_sympy__physics__quantum__gate__SwapGate():
    from sympy.physics.quantum.gate import SwapGate
    assert _test_args(SwapGate(0, 1))


def test_sympy__physics__quantum__gate__TGate():
    from sympy.physics.quantum.gate import TGate
    assert _test_args(TGate(0))


def test_sympy__physics__quantum__gate__TwoQubitGate():
    from sympy.physics.quantum.gate import TwoQubitGate
    assert _test_args(TwoQubitGate(0))


def test_sympy__physics__quantum__gate__UGate():
    from sympy.physics.quantum.gate import UGate
    from sympy.matrices.immutable import ImmutableDenseMatrix
    from sympy.core.containers import Tuple
    from sympy.core.numbers import Integer
    assert _test_args(
        UGate(Tuple(Integer(1)), ImmutableDenseMatrix([[1, 0], [0, 2]])))


def test_sympy__physics__quantum__gate__XGate():
    from sympy.physics.quantum.gate import XGate
    assert _test_args(XGate(0))


def test_sympy__physics__quantum__gate__YGate():
    from sympy.physics.quantum.gate import YGate
    assert _test_args(YGate(0))


def test_sympy__physics__quantum__gate__ZGate():
    from sympy.physics.quantum.gate import ZGate
    assert _test_args(ZGate(0))


def test_sympy__physics__quantum__grover__OracleGateFunction():
    from sympy.physics.quantum.grover import OracleGateFunction
    @OracleGateFunction
    def f(qubit):
        return
    assert _test_args(f)

def test_sympy__physics__quantum__grover__OracleGate():
    from sympy.physics.quantum.grover import OracleGate
    def f(qubit):
        return
    assert _test_args(OracleGate(1,f))


def test_sympy__physics__quantum__grover__WGate():
    from sympy.physics.quantum.grover import WGate
    assert _test_args(WGate(1))


def test_sympy__physics__quantum__hilbert__ComplexSpace():
    from sympy.physics.quantum.hilbert import ComplexSpace
    assert _test_args(ComplexSpace(x))


def test_sympy__physics__quantum__hilbert__DirectSumHilbertSpace():
    from sympy.physics.quantum.hilbert import DirectSumHilbertSpace, ComplexSpace, FockSpace
    c = ComplexSpace(2)
    f = FockSpace()
    assert _test_args(DirectSumHilbertSpace(c, f))


def test_sympy__physics__quantum__hilbert__FockSpace():
    from sympy.physics.quantum.hilbert import FockSpace
    assert _test_args(FockSpace())


def test_sympy__physics__quantum__hilbert__HilbertSpace():
    from sympy.physics.quantum.hilbert import HilbertSpace
    assert _test_args(HilbertSpace())


def test_sympy__physics__quantum__hilbert__L2():
    from sympy.physics.quantum.hilbert import L2
    from sympy.core.numbers import oo
    from sympy.sets.sets import Interval
    assert _test_args(L2(Interval(0, oo)))


def test_sympy__physics__quantum__hilbert__TensorPowerHilbertSpace():
    from sympy.physics.quantum.hilbert import TensorPowerHilbertSpace, FockSpace
    f = FockSpace()
    assert _test_args(TensorPowerHilbertSpace(f, 2))


def test_sympy__physics__quantum__hilbert__TensorProductHilbertSpace():
    from sympy.physics.quantum.hilbert import TensorProductHilbertSpace, FockSpace, ComplexSpace
    c = ComplexSpace(2)
    f = FockSpace()
    assert _test_args(TensorProductHilbertSpace(f, c))


def test_sympy__physics__quantum__innerproduct__InnerProduct():
    from sympy.physics.quantum import Bra, Ket, InnerProduct
    b = Bra('b')
    k = Ket('k')
    assert _test_args(InnerProduct(b, k))


def test_sympy__physics__quantum__operator__DifferentialOperator():
    from sympy.physics.quantum.operator import DifferentialOperator
    from sympy.core.function import (Derivative, Function)
    f = Function('f')
    assert _test_args(DifferentialOperator(1/x*Derivative(f(x), x), f(x)))


def test_sympy__physics__quantum__operator__HermitianOperator():
    from sympy.physics.quantum.operator import HermitianOperator
    assert _test_args(HermitianOperator('H'))


def test_sympy__physics__quantum__operator__IdentityOperator():
    with warns_deprecated_sympy():
        from sympy.physics.quantum.operator import IdentityOperator
        assert _test_args(IdentityOperator(5))


def test_sympy__physics__quantum__operator__Operator():
    from sympy.physics.quantum.operator import Operator
    assert _test_args(Operator('A'))


def test_sympy__physics__quantum__operator__OuterProduct():
    from sympy.physics.quantum.operator import OuterProduct
    from sympy.physics.quantum import Ket, Bra
    b = Bra('b')
    k = Ket('k')
    assert _test_args(OuterProduct(k, b))


def test_sympy__physics__quantum__operator__UnitaryOperator():
    from sympy.physics.quantum.operator import UnitaryOperator
    assert _test_args(UnitaryOperator('U'))


def test_sympy__physics__quantum__piab__PIABBra():
    from sympy.physics.quantum.piab import PIABBra
    assert _test_args(PIABBra('B'))


def test_sympy__physics__quantum__boson__BosonOp():
    from sympy.physics.quantum.boson import BosonOp
    assert _test_args(BosonOp('a'))
    assert _test_args(BosonOp('a', False))


def test_sympy__physics__quantum__boson__BosonFockKet():
    from sympy.physics.quantum.boson import BosonFockKet
    assert _test_args(BosonFockKet(1))


def test_sympy__physics__quantum__boson__BosonFockBra():
    from sympy.physics.quantum.boson import BosonFockBra
    assert _test_args(BosonFockBra(1))


def test_sympy__physics__quantum__boson__BosonCoherentKet():
    from sympy.physics.quantum.boson import BosonCoherentKet
    assert _test_args(BosonCoherentKet(1))


def test_sympy__physics__quantum__boson__BosonCoherentBra():
    from sympy.physics.quantum.boson import BosonCoherentBra
    assert _test_args(BosonCoherentBra(1))


def test_sympy__physics__quantum__fermion__FermionOp():
    from sympy.physics.quantum.fermion import FermionOp
    assert _test_args(FermionOp('c'))
    assert _test_args(FermionOp('c', False))


def test_sympy__physics__quantum__fermion__FermionFockKet():
    from sympy.physics.quantum.fermion import FermionFockKet
    assert _test_args(FermionFockKet(1))


def test_sympy__physics__quantum__fermion__FermionFockBra():
    from sympy.physics.quantum.fermion import FermionFockBra
    assert _test_args(FermionFockBra(1))


def test_sympy__physics__quantum__pauli__SigmaOpBase():
    from sympy.physics.quantum.pauli import SigmaOpBase
    assert _test_args(SigmaOpBase())


def test_sympy__physics__quantum__pauli__SigmaX():
    from sympy.physics.quantum.pauli import SigmaX
    assert _test_args(SigmaX())


def test_sympy__physics__quantum__pauli__SigmaY():
    from sympy.physics.quantum.pauli import SigmaY
    assert _test_args(SigmaY())


def test_sympy__physics__quantum__pauli__SigmaZ():
    from sympy.physics.quantum.pauli import SigmaZ
    assert _test_args(SigmaZ())


def test_sympy__physics__quantum__pauli__SigmaMinus():
    from sympy.physics.quantum.pauli import SigmaMinus
    assert _test_args(SigmaMinus())


def test_sympy__physics__quantum__pauli__SigmaPlus():
    from sympy.physics.quantum.pauli import SigmaPlus
    assert _test_args(SigmaPlus())


def test_sympy__physics__quantum__pauli__SigmaZKet():
    from sympy.physics.quantum.pauli import SigmaZKet
    assert _test_args(SigmaZKet(0))


def test_sympy__physics__quantum__pauli__SigmaZBra():
    from sympy.physics.quantum.pauli import SigmaZBra
    assert _test_args(SigmaZBra(0))


def test_sympy__physics__quantum__piab__PIABHamiltonian():
    from sympy.physics.quantum.piab import PIABHamiltonian
    assert _test_args(PIABHamiltonian('P'))


def test_sympy__physics__quantum__piab__PIABKet():
    from sympy.physics.quantum.piab import PIABKet
    assert _test_args(PIABKet('K'))


def test_sympy__physics__quantum__qexpr__QExpr():
    from sympy.physics.quantum.qexpr import QExpr
    assert _test_args(QExpr(0))


def test_sympy__physics__quantum__qft__Fourier():
    from sympy.physics.quantum.qft import Fourier
    assert _test_args(Fourier(0, 1))


def test_sympy__physics__quantum__qft__IQFT():
    from sympy.physics.quantum.qft import IQFT
    assert _test_args(IQFT(0, 1))


def test_sympy__physics__quantum__qft__QFT():
    from sympy.physics.quantum.qft import QFT
    assert _test_args(QFT(0, 1))


def test_sympy__physics__quantum__qft__RkGate():
    from sympy.physics.quantum.qft import RkGate
    assert _test_args(RkGate(0, 1))


def test_sympy__physics__quantum__qubit__IntQubit():
    from sympy.physics.quantum.qubit import IntQubit
    assert _test_args(IntQubit(0))


def test_sympy__physics__quantum__qubit__IntQubitBra():
    from sympy.physics.quantum.qubit import IntQubitBra
    assert _test_args(IntQubitBra(0))


def test_sympy__physics__quantum__qubit__IntQubitState():
    from sympy.physics.quantum.qubit import IntQubitState, QubitState
    assert _test_args(IntQubitState(QubitState(0, 1)))


def test_sympy__physics__quantum__qubit__Qubit():
    from sympy.physics.quantum.qubit import Qubit
    assert _test_args(Qubit(0, 0, 0))


def test_sympy__physics__quantum__qubit__QubitBra():
    from sympy.physics.quantum.qubit import QubitBra
    assert _test_args(QubitBra('1', 0))


def test_sympy__physics__quantum__qubit__QubitState():
    from sympy.physics.quantum.qubit import QubitState
    assert _test_args(QubitState(0, 1))


def test_sympy__physics__quantum__density__Density():
    from sympy.physics.quantum.density import Density
    from sympy.physics.quantum.state import Ket
    assert _test_args(Density([Ket(0), 0.5], [Ket(1), 0.5]))


@SKIP("TODO: sympy.physics.quantum.shor: Cmod Not Implemented")
def test_sympy__physics__quantum__shor__CMod():
    from sympy.physics.quantum.shor import CMod
    assert _test_args(CMod())


def test_sympy__physics__quantum__spin__CoupledSpinState():
    from sympy.physics.quantum.spin import CoupledSpinState
    assert _test_args(CoupledSpinState(1, 0, (1, 1)))
    assert _test_args(CoupledSpinState(1, 0, (1, S.Half, S.Half)))
    assert _test_args(CoupledSpinState(
        1, 0, (1, S.Half, S.Half), ((2, 3, S.Half), (1, 2, 1)) ))
    j, m, j1, j2, j3, j12, x = symbols('j m j1:4 j12 x')
    assert CoupledSpinState(
        j, m, (j1, j2, j3)).subs(j2, x) == CoupledSpinState(j, m, (j1, x, j3))
    assert CoupledSpinState(j, m, (j1, j2, j3), ((1, 3, j12), (1, 2, j)) ).subs(j12, x) == \
        CoupledSpinState(j, m, (j1, j2, j3), ((1, 3, x), (1, 2, j)) )


def test_sympy__physics__quantum__spin__J2Op():
    from sympy.physics.quantum.spin import J2Op
    assert _test_args(J2Op('J'))


def test_sympy__physics__quantum__spin__JminusOp():
    from sympy.physics.quantum.spin import JminusOp
    assert _test_args(JminusOp('J'))


def test_sympy__physics__quantum__spin__JplusOp():
    from sympy.physics.quantum.spin import JplusOp
    assert _test_args(JplusOp('J'))


def test_sympy__physics__quantum__spin__JxBra():
    from sympy.physics.quantum.spin import JxBra
    assert _test_args(JxBra(1, 0))


def test_sympy__physics__quantum__spin__JxBraCoupled():
    from sympy.physics.quantum.spin import JxBraCoupled
    assert _test_args(JxBraCoupled(1, 0, (1, 1)))


def test_sympy__physics__quantum__spin__JxKet():
    from sympy.physics.quantum.spin import JxKet
    assert _test_args(JxKet(1, 0))


def test_sympy__physics__quantum__spin__JxKetCoupled():
    from sympy.physics.quantum.spin import JxKetCoupled
    assert _test_args(JxKetCoupled(1, 0, (1, 1)))


def test_sympy__physics__quantum__spin__JxOp():
    from sympy.physics.quantum.spin import JxOp
    assert _test_args(JxOp('J'))


def test_sympy__physics__quantum__spin__JyBra():
    from sympy.physics.quantum.spin import JyBra
    assert _test_args(JyBra(1, 0))


def test_sympy__physics__quantum__spin__JyBraCoupled():
    from sympy.physics.quantum.spin import JyBraCoupled
    assert _test_args(JyBraCoupled(1, 0, (1, 1)))


def test_sympy__physics__quantum__spin__JyKet():
    from sympy.physics.quantum.spin import JyKet
    assert _test_args(JyKet(1, 0))


def test_sympy__physics__quantum__spin__JyKetCoupled():
    from sympy.physics.quantum.spin import JyKetCoupled
    assert _test_args(JyKetCoupled(1, 0, (1, 1)))


def test_sympy__physics__quantum__spin__JyOp():
    from sympy.physics.quantum.spin import JyOp
    assert _test_args(JyOp('J'))


def test_sympy__physics__quantum__spin__JzBra():
    from sympy.physics.quantum.spin import JzBra
    assert _test_args(JzBra(1, 0))


def test_sympy__physics__quantum__spin__JzBraCoupled():
    from sympy.physics.quantum.spin import JzBraCoupled
    assert _test_args(JzBraCoupled(1, 0, (1, 1)))


def test_sympy__physics__quantum__spin__JzKet():
    from sympy.physics.quantum.spin import JzKet
    assert _test_args(JzKet(1, 0))


def test_sympy__physics__quantum__spin__JzKetCoupled():
    from sympy.physics.quantum.spin import JzKetCoupled
    assert _test_args(JzKetCoupled(1, 0, (1, 1)))


def test_sympy__physics__quantum__spin__JzOp():
    from sympy.physics.quantum.spin import JzOp
    assert _test_args(JzOp('J'))


def test_sympy__physics__quantum__spin__Rotation():
    from sympy.physics.quantum.spin import Rotation
    assert _test_args(Rotation(pi, 0, pi/2))


def test_sympy__physics__quantum__spin__SpinState():
    from sympy.physics.quantum.spin import SpinState
    assert _test_args(SpinState(1, 0))


def test_sympy__physics__quantum__spin__WignerD():
    from sympy.physics.quantum.spin import WignerD
    assert _test_args(WignerD(0, 1, 2, 3, 4, 5))


def test_sympy__physics__quantum__state__Bra():
    from sympy.physics.quantum.state import Bra
    assert _test_args(Bra(0))


def test_sympy__physics__quantum__state__BraBase():
    from sympy.physics.quantum.state import BraBase
    assert _test_args(BraBase(0))


def test_sympy__physics__quantum__state__Ket():
    from sympy.physics.quantum.state import Ket
    assert _test_args(Ket(0))


def test_sympy__physics__quantum__state__KetBase():
    from sympy.physics.quantum.state import KetBase
    assert _test_args(KetBase(0))


def test_sympy__physics__quantum__state__State():
    from sympy.physics.quantum.state import State
    assert _test_args(State(0))


def test_sympy__physics__quantum__state__StateBase():
    from sympy.physics.quantum.state import StateBase
    assert _test_args(StateBase(0))


def test_sympy__physics__quantum__state__OrthogonalBra():
    from sympy.physics.quantum.state import OrthogonalBra
    assert _test_args(OrthogonalBra(0))


def test_sympy__physics__quantum__state__OrthogonalKet():
    from sympy.physics.quantum.state import OrthogonalKet
    assert _test_args(OrthogonalKet(0))


def test_sympy__physics__quantum__state__OrthogonalState():
    from sympy.physics.quantum.state import OrthogonalState
    assert _test_args(OrthogonalState(0))


def test_sympy__physics__quantum__state__TimeDepBra():
    from sympy.physics.quantum.state import TimeDepBra
    assert _test_args(TimeDepBra('psi', 't'))


def test_sympy__physics__quantum__state__TimeDepKet():
    from sympy.physics.quantum.state import TimeDepKet
    assert _test_args(TimeDepKet('psi', 't'))


def test_sympy__physics__quantum__state__TimeDepState():
    from sympy.physics.quantum.state import TimeDepState
    assert _test_args(TimeDepState('psi', 't'))


def test_sympy__physics__quantum__state__Wavefunction():
    from sympy.physics.quantum.state import Wavefunction
    from sympy.functions import sin
    from sympy.functions.elementary.piecewise import Piecewise
    n = 1
    L = 1
    g = Piecewise((0, x < 0), (0, x > L), (sqrt(2//L)*sin(n*pi*x/L), True))
    assert _test_args(Wavefunction(g, x))


def test_sympy__physics__quantum__tensorproduct__TensorProduct():
    from sympy.physics.quantum.tensorproduct import TensorProduct
    x, y = symbols("x y", commutative=False)
    assert _test_args(TensorProduct(x, y))


def test_sympy__physics__quantum__identitysearch__GateIdentity():
    from sympy.physics.quantum.gate import X
    from sympy.physics.quantum.identitysearch import GateIdentity
    assert _test_args(GateIdentity(X(0), X(0)))


def test_sympy__physics__quantum__sho1d__SHOOp():
    from sympy.physics.quantum.sho1d import SHOOp
    assert _test_args(SHOOp('a'))


def test_sympy__physics__quantum__sho1d__RaisingOp():
    from sympy.physics.quantum.sho1d import RaisingOp
    assert _test_args(RaisingOp('a'))


def test_sympy__physics__quantum__sho1d__LoweringOp():
    from sympy.physics.quantum.sho1d import LoweringOp
    assert _test_args(LoweringOp('a'))


def test_sympy__physics__quantum__sho1d__NumberOp():
    from sympy.physics.quantum.sho1d import NumberOp
    assert _test_args(NumberOp('N'))


def test_sympy__physics__quantum__sho1d__Hamiltonian():
    from sympy.physics.quantum.sho1d import Hamiltonian
    assert _test_args(Hamiltonian('H'))


def test_sympy__physics__quantum__sho1d__SHOState():
    from sympy.physics.quantum.sho1d import SHOState
    assert _test_args(SHOState(0))


def test_sympy__physics__quantum__sho1d__SHOKet():
    from sympy.physics.quantum.sho1d import SHOKet
    assert _test_args(SHOKet(0))


def test_sympy__physics__quantum__sho1d__SHOBra():
    from sympy.physics.quantum.sho1d import SHOBra
    assert _test_args(SHOBra(0))


def test_sympy__physics__secondquant__AnnihilateBoson():
    from sympy.physics.secondquant import AnnihilateBoson
    assert _test_args(AnnihilateBoson(0))


def test_sympy__physics__secondquant__AnnihilateFermion():
    from sympy.physics.secondquant import AnnihilateFermion
    assert _test_args(AnnihilateFermion(0))


@SKIP("abstract class")
def test_sympy__physics__secondquant__Annihilator():
    pass


def test_sympy__physics__secondquant__AntiSymmetricTensor():
    from sympy.physics.secondquant import AntiSymmetricTensor
    i, j = symbols('i j', below_fermi=True)
    a, b = symbols('a b', above_fermi=True)
    assert _test_args(AntiSymmetricTensor('v', (a, i), (b, j)))


def test_sympy__physics__secondquant__BosonState():
    from sympy.physics.secondquant import BosonState
    assert _test_args(BosonState((0, 1)))


@SKIP("abstract class")
def test_sympy__physics__secondquant__BosonicOperator():
    pass


def test_sympy__physics__secondquant__Commutator():
    from sympy.physics.secondquant import Commutator
    x, y = symbols('x y', commutative=False)
    assert _test_args(Commutator(x, y))


def test_sympy__physics__secondquant__CreateBoson():
    from sympy.physics.secondquant import CreateBoson
    assert _test_args(CreateBoson(0))


def test_sympy__physics__secondquant__CreateFermion():
    from sympy.physics.secondquant import CreateFermion
    assert _test_args(CreateFermion(0))


@SKIP("abstract class")
def test_sympy__physics__secondquant__Creator():
    pass


def test_sympy__physics__secondquant__Dagger():
    from sympy.physics.secondquant import Dagger
    assert _test_args(Dagger(x))


def test_sympy__physics__secondquant__FermionState():
    from sympy.physics.secondquant import FermionState
    assert _test_args(FermionState((0, 1)))


def test_sympy__physics__secondquant__FermionicOperator():
    from sympy.physics.secondquant import FermionicOperator
    assert _test_args(FermionicOperator(0))


def test_sympy__physics__secondquant__FockState():
    from sympy.physics.secondquant import FockState
    assert _test_args(FockState((0, 1)))


def test_sympy__physics__secondquant__FockStateBosonBra():
    from sympy.physics.secondquant import FockStateBosonBra
    assert _test_args(FockStateBosonBra((0, 1)))


def test_sympy__physics__secondquant__FockStateBosonKet():
    from sympy.physics.secondquant import FockStateBosonKet
    assert _test_args(FockStateBosonKet((0, 1)))


def test_sympy__physics__secondquant__FockStateBra():
    from sympy.physics.secondquant import FockStateBra
    assert _test_args(FockStateBra((0, 1)))


def test_sympy__physics__secondquant__FockStateFermionBra():
    from sympy.physics.secondquant import FockStateFermionBra
    assert _test_args(FockStateFermionBra((0, 1)))


def test_sympy__physics__secondquant__FockStateFermionKet():
    from sympy.physics.secondquant import FockStateFermionKet
    assert _test_args(FockStateFermionKet((0, 1)))


def test_sympy__physics__secondquant__FockStateKet():
    from sympy.physics.secondquant import FockStateKet
    assert _test_args(FockStateKet((0, 1)))


def test_sympy__physics__secondquant__InnerProduct():
    from sympy.physics.secondquant import InnerProduct
    from sympy.physics.secondquant import FockStateKet, FockStateBra
    assert _test_args(InnerProduct(FockStateBra((0, 1)), FockStateKet((0, 1))))


def test_sympy__physics__secondquant__NO():
    from sympy.physics.secondquant import NO, F, Fd
    assert _test_args(NO(Fd(x)*F(y)))


def test_sympy__physics__secondquant__PermutationOperator():
    from sympy.physics.secondquant import PermutationOperator
    assert _test_args(PermutationOperator(0, 1))


def test_sympy__physics__secondquant__SqOperator():
    from sympy.physics.secondquant import SqOperator
    assert _test_args(SqOperator(0))


def test_sympy__physics__secondquant__TensorSymbol():
    from sympy.physics.secondquant import TensorSymbol
    assert _test_args(TensorSymbol(x))


def test_sympy__physics__control__lti__LinearTimeInvariant():
    # Direct instances of LinearTimeInvariant class are not allowed.
    # func(*args) tests for its derived classes (TransferFunction,
    # Series, Parallel and TransferFunctionMatrix) should pass.
    pass


def test_sympy__physics__control__lti__SISOLinearTimeInvariant():
    # Direct instances of SISOLinearTimeInvariant class are not allowed.
    pass


def test_sympy__physics__control__lti__MIMOLinearTimeInvariant():
    # Direct instances of MIMOLinearTimeInvariant class are not allowed.
    pass


def test_sympy__physics__control__lti__TransferFunction():
    from sympy.physics.control.lti import TransferFunction
    assert _test_args(TransferFunction(2, 3, x))


def _test_args_PIDController(obj):
    from sympy.physics.control.lti import PIDController
    if isinstance(obj, PIDController):
        kp, ki, kd, tf = obj.kp, obj.ki, obj.kd, obj.tf
        recreated_pid = PIDController(kp, ki, kd, tf, s)
        return recreated_pid == obj
    return False


def test_sympy__physics__control__lti__PIDController():
    from sympy.physics.control.lti import PIDController
    kp, ki, kd, tf = 1, 0.1, 0.01, 0
    assert _test_args_PIDController(PIDController(kp, ki, kd, tf, s))


def test_sympy__physics__control__lti__Series():
    from sympy.physics.control import Series, TransferFunction
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    assert _test_args(Series(tf1, tf2))


def test_sympy__physics__control__lti__MIMOSeries():
    from sympy.physics.control import MIMOSeries, TransferFunction, TransferFunctionMatrix
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    tfm_1 = TransferFunctionMatrix([[tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_3 = TransferFunctionMatrix([[tf1], [tf2]])
    assert _test_args(MIMOSeries(tfm_3, tfm_2, tfm_1))


def test_sympy__physics__control__lti__Parallel():
    from sympy.physics.control import Parallel, TransferFunction
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    assert _test_args(Parallel(tf1, tf2))


def test_sympy__physics__control__lti__MIMOParallel():
    from sympy.physics.control import MIMOParallel, TransferFunction, TransferFunctionMatrix
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    assert _test_args(MIMOParallel(tfm_1, tfm_2))


def test_sympy__physics__control__lti__Feedback():
    from sympy.physics.control import TransferFunction, Feedback
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    assert _test_args(Feedback(tf1, tf2))
    assert _test_args(Feedback(tf1, tf2, 1))


def test_sympy__physics__control__lti__MIMOFeedback():
    from sympy.physics.control import TransferFunction, MIMOFeedback, TransferFunctionMatrix
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    tfm_1 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    tfm_2 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    assert _test_args(MIMOFeedback(tfm_1, tfm_2))
    assert _test_args(MIMOFeedback(tfm_1, tfm_2, 1))


def test_sympy__physics__control__lti__TransferFunctionMatrix():
    from sympy.physics.control import TransferFunction, TransferFunctionMatrix
    tf1 = TransferFunction(x**2 - y**3, y - z, x)
    tf2 = TransferFunction(y - x, z + y, x)
    assert _test_args(TransferFunctionMatrix([[tf1, tf2]]))


def test_sympy__physics__control__lti__StateSpace():
    from sympy.matrices.dense import Matrix
    from sympy.physics.control import StateSpace
    A = Matrix([[-5, -1], [3, -1]])
    B = Matrix([2, 5])
    C = Matrix([[1, 2]])
    D = Matrix([0])
    assert _test_args(StateSpace(A, B, C, D))


def test_sympy__physics__units__dimensions__Dimension():
    from sympy.physics.units.dimensions import Dimension
    assert _test_args(Dimension("length", "L"))


def test_sympy__physics__units__dimensions__DimensionSystem():
    from sympy.physics.units.dimensions import DimensionSystem
    from sympy.physics.units.definitions.dimension_definitions import length, time, velocity
    assert _test_args(DimensionSystem((length, time), (velocity,)))


def test_sympy__physics__units__quantities__Quantity():
    from sympy.physics.units.quantities import Quantity
    assert _test_args(Quantity("dam"))


def test_sympy__physics__units__quantities__PhysicalConstant():
    from sympy.physics.units.quantities import PhysicalConstant
    assert _test_args(PhysicalConstant("foo"))


def test_sympy__physics__units__prefixes__Prefix():
    from sympy.physics.units.prefixes import Prefix
    assert _test_args(Prefix('kilo', 'k', 3))


def test_sympy__core__numbers__AlgebraicNumber():
    from sympy.core.numbers import AlgebraicNumber
    assert _test_args(AlgebraicNumber(sqrt(2), [1, 2, 3]))


def test_sympy__polys__polytools__GroebnerBasis():
    from sympy.polys.polytools import GroebnerBasis
    assert _test_args(GroebnerBasis([x, y, z], x, y, z))


def test_sympy__polys__polytools__Poly():
    from sympy.polys.polytools import Poly
    assert _test_args(Poly(2, x, y))


def test_sympy__polys__polytools__PurePoly():
    from sympy.polys.polytools import PurePoly
    assert _test_args(PurePoly(2, x, y))


@SKIP('abstract class')
def test_sympy__polys__rootoftools__RootOf():
    pass


def test_sympy__polys__rootoftools__ComplexRootOf():
    from sympy.polys.rootoftools import ComplexRootOf
    assert _test_args(ComplexRootOf(x**3 + x + 1, 0))


def test_sympy__polys__rootoftools__RootSum():
    from sympy.polys.rootoftools import RootSum
    assert _test_args(RootSum(x**3 + x + 1, sin))


def test_sympy__series__limits__Limit():
    from sympy.series.limits import Limit
    assert _test_args(Limit(x, x, 0, dir='-'))


def test_sympy__series__order__Order():
    from sympy.series.order import Order
    assert _test_args(Order(1, x, y))


@SKIP('Abstract Class')
def test_sympy__series__sequences__SeqBase():
    pass


def test_sympy__series__sequences__EmptySequence():
    # Need to import the instance from series not the class from
    # series.sequence
    from sympy.series import EmptySequence
    assert _test_args(EmptySequence)


@SKIP('Abstract Class')
def test_sympy__series__sequences__SeqExpr():
    pass


def test_sympy__series__sequences__SeqPer():
    from sympy.series.sequences import SeqPer
    assert _test_args(SeqPer((1, 2, 3), (0, 10)))


def test_sympy__series__sequences__SeqFormula():
    from sympy.series.sequences import SeqFormula
    assert _test_args(SeqFormula(x**2, (0, 10)))


def test_sympy__series__sequences__RecursiveSeq():
    from sympy.series.sequences import RecursiveSeq
    y = Function("y")
    n = symbols("n")
    assert _test_args(RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, (0, 1)))
    assert _test_args(RecursiveSeq(y(n - 1) + y(n - 2), y(n), n))


def test_sympy__series__sequences__SeqExprOp():
    from sympy.series.sequences import SeqExprOp, sequence
    s1 = sequence((1, 2, 3))
    s2 = sequence(x**2)
    assert _test_args(SeqExprOp(s1, s2))


def test_sympy__series__sequences__SeqAdd():
    from sympy.series.sequences import SeqAdd, sequence
    s1 = sequence((1, 2, 3))
    s2 = sequence(x**2)
    assert _test_args(SeqAdd(s1, s2))


def test_sympy__series__sequences__SeqMul():
    from sympy.series.sequences import SeqMul, sequence
    s1 = sequence((1, 2, 3))
    s2 = sequence(x**2)
    assert _test_args(SeqMul(s1, s2))


@SKIP('Abstract Class')
def test_sympy__series__series_class__SeriesBase():
    pass


def test_sympy__series__fourier__FourierSeries():
    from sympy.series.fourier import fourier_series
    assert _test_args(fourier_series(x, (x, -pi, pi)))

def test_sympy__series__fourier__FiniteFourierSeries():
    from sympy.series.fourier import fourier_series
    assert _test_args(fourier_series(sin(pi*x), (x, -1, 1)))


def test_sympy__series__formal__FormalPowerSeries():
    from sympy.series.formal import fps
    assert _test_args(fps(log(1 + x), x))


def test_sympy__series__formal__Coeff():
    from sympy.series.formal import fps
    assert _test_args(fps(x**2 + x + 1, x))


@SKIP('Abstract Class')
def test_sympy__series__formal__FiniteFormalPowerSeries():
    pass


def test_sympy__series__formal__FormalPowerSeriesProduct():
    from sympy.series.formal import fps
    f1, f2 = fps(sin(x)), fps(exp(x))
    assert _test_args(f1.product(f2, x))


def test_sympy__series__formal__FormalPowerSeriesCompose():
    from sympy.series.formal import fps
    f1, f2 = fps(exp(x)), fps(sin(x))
    assert _test_args(f1.compose(f2, x))


def test_sympy__series__formal__FormalPowerSeriesInverse():
    from sympy.series.formal import fps
    f1 = fps(exp(x))
    assert _test_args(f1.inverse(x))


def test_sympy__simplify__hyperexpand__Hyper_Function():
    from sympy.simplify.hyperexpand import Hyper_Function
    assert _test_args(Hyper_Function([2], [1]))


def test_sympy__simplify__hyperexpand__G_Function():
    from sympy.simplify.hyperexpand import G_Function
    assert _test_args(G_Function([2], [1], [], []))


@SKIP("abstract class")
def test_sympy__tensor__array__ndim_array__ImmutableNDimArray():
    pass


def test_sympy__tensor__array__dense_ndim_array__ImmutableDenseNDimArray():
    from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
    densarr = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))
    assert _test_args(densarr)


def test_sympy__tensor__array__sparse_ndim_array__ImmutableSparseNDimArray():
    from sympy.tensor.array.sparse_ndim_array import ImmutableSparseNDimArray
    sparr = ImmutableSparseNDimArray(range(10, 34), (2, 3, 4))
    assert _test_args(sparr)


def test_sympy__tensor__array__array_comprehension__ArrayComprehension():
    from sympy.tensor.array.array_comprehension import ArrayComprehension
    arrcom = ArrayComprehension(x, (x, 1, 5))
    assert _test_args(arrcom)

def test_sympy__tensor__array__array_comprehension__ArrayComprehensionMap():
    from sympy.tensor.array.array_comprehension import ArrayComprehensionMap
    arrcomma = ArrayComprehensionMap(lambda: 0, (x, 1, 5))
    assert _test_args(arrcomma)


def test_sympy__tensor__array__array_derivatives__ArrayDerivative():
    from sympy.tensor.array.array_derivatives import ArrayDerivative
    A = MatrixSymbol("A", 2, 2)
    arrder = ArrayDerivative(A, A, evaluate=False)
    assert _test_args(arrder)

def test_sympy__tensor__array__expressions__array_expressions__ArraySymbol():
    from sympy.tensor.array.expressions.array_expressions import ArraySymbol
    m, n, k = symbols("m n k")
    array = ArraySymbol("A", (m, n, k, 2))
    assert _test_args(array)

def test_sympy__tensor__array__expressions__array_expressions__ArrayElement():
    from sympy.tensor.array.expressions.array_expressions import ArrayElement
    m, n, k = symbols("m n k")
    ae = ArrayElement("A", (m, n, k, 2))
    assert _test_args(ae)

def test_sympy__tensor__array__expressions__array_expressions__ZeroArray():
    from sympy.tensor.array.expressions.array_expressions import ZeroArray
    m, n, k = symbols("m n k")
    za = ZeroArray(m, n, k, 2)
    assert _test_args(za)

def test_sympy__tensor__array__expressions__array_expressions__OneArray():
    from sympy.tensor.array.expressions.array_expressions import OneArray
    m, n, k = symbols("m n k")
    za = OneArray(m, n, k, 2)
    assert _test_args(za)

def test_sympy__tensor__functions__TensorProduct():
    from sympy.tensor.functions import TensorProduct
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    tp = TensorProduct(A, B)
    assert _test_args(tp)


def test_sympy__tensor__indexed__Idx():
    from sympy.tensor.indexed import Idx
    assert _test_args(Idx('test'))
    assert _test_args(Idx('test', (0, 10)))
    assert _test_args(Idx('test', 2))
    assert _test_args(Idx('test', x))


def test_sympy__tensor__indexed__Indexed():
    from sympy.tensor.indexed import Indexed, Idx
    assert _test_args(Indexed('A', Idx('i'), Idx('j')))


def test_sympy__tensor__indexed__IndexedBase():
    from sympy.tensor.indexed import IndexedBase
    assert _test_args(IndexedBase('A', shape=(x, y)))
    assert _test_args(IndexedBase('A', 1))
    assert _test_args(IndexedBase('A')[0, 1])


def test_sympy__tensor__tensor__TensorIndexType():
    from sympy.tensor.tensor import TensorIndexType
    assert _test_args(TensorIndexType('Lorentz'))


@SKIP("deprecated class")
def test_sympy__tensor__tensor__TensorType():
    pass


def test_sympy__tensor__tensor__TensorSymmetry():
    from sympy.tensor.tensor import TensorSymmetry, get_symmetric_group_sgs
    assert _test_args(TensorSymmetry(get_symmetric_group_sgs(2)))


def test_sympy__tensor__tensor__TensorHead():
    from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, TensorHead
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    sym = TensorSymmetry(get_symmetric_group_sgs(1))
    assert _test_args(TensorHead('p', [Lorentz], sym, 0))


def test_sympy__tensor__tensor__TensorIndex():
    from sympy.tensor.tensor import TensorIndexType, TensorIndex
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    assert _test_args(TensorIndex('i', Lorentz))

@SKIP("abstract class")
def test_sympy__tensor__tensor__TensExpr():
    pass

def test_sympy__tensor__tensor__TensAdd():
    from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, tensor_indices, TensAdd, tensor_heads
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b = tensor_indices('a,b', Lorentz)
    sym = TensorSymmetry(get_symmetric_group_sgs(1))
    p, q = tensor_heads('p,q', [Lorentz], sym)
    t1 = p(a)
    t2 = q(a)
    assert _test_args(TensAdd(t1, t2))


def test_sympy__tensor__tensor__Tensor():
    from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, tensor_indices, TensorHead
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b = tensor_indices('a,b', Lorentz)
    sym = TensorSymmetry(get_symmetric_group_sgs(1))
    p = TensorHead('p', [Lorentz], sym)
    assert _test_args(p(a))


def test_sympy__tensor__tensor__TensMul():
    from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, tensor_indices, tensor_heads
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b = tensor_indices('a,b', Lorentz)
    sym = TensorSymmetry(get_symmetric_group_sgs(1))
    p, q = tensor_heads('p, q', [Lorentz], sym)
    assert _test_args(3*p(a)*q(b))


def test_sympy__tensor__tensor__TensorElement():
    from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorElement
    L = TensorIndexType("L")
    A = TensorHead("A", [L, L])
    telem = TensorElement(A(x, y), {x: 1})
    assert _test_args(telem)

def test_sympy__tensor__tensor__WildTensor():
    from sympy.tensor.tensor import TensorIndexType, WildTensorHead, TensorIndex
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a = TensorIndex('a', Lorentz)
    p = WildTensorHead('p')
    assert _test_args(p(a))

def test_sympy__tensor__tensor__WildTensorHead():
    from sympy.tensor.tensor import WildTensorHead
    assert _test_args(WildTensorHead('p'))

def test_sympy__tensor__tensor__WildTensorIndex():
    from sympy.tensor.tensor import TensorIndexType, WildTensorIndex
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    assert _test_args(WildTensorIndex('i', Lorentz))

def test_sympy__tensor__toperators__PartialDerivative():
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead
    from sympy.tensor.toperators import PartialDerivative
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b = tensor_indices('a,b', Lorentz)
    A = TensorHead("A", [Lorentz])
    assert _test_args(PartialDerivative(A(a), A(b)))


def test_as_coeff_add():
    assert (7, (3*x, 4*x**2)) == (7 + 3*x + 4*x**2).as_coeff_add()


def test_sympy__geometry__curve__Curve():
    from sympy.geometry.curve import Curve
    assert _test_args(Curve((x, 1), (x, 0, 1)))


def test_sympy__geometry__point__Point():
    from sympy.geometry.point import Point
    assert _test_args(Point(0, 1))


def test_sympy__geometry__point__Point2D():
    from sympy.geometry.point import Point2D
    assert _test_args(Point2D(0, 1))


def test_sympy__geometry__point__Point3D():
    from sympy.geometry.point import Point3D
    assert _test_args(Point3D(0, 1, 2))


def test_sympy__geometry__ellipse__Ellipse():
    from sympy.geometry.ellipse import Ellipse
    assert _test_args(Ellipse((0, 1), 2, 3))


def test_sympy__geometry__ellipse__Circle():
    from sympy.geometry.ellipse import Circle
    assert _test_args(Circle((0, 1), 2))


def test_sympy__geometry__parabola__Parabola():
    from sympy.geometry.parabola import Parabola
    from sympy.geometry.line import Line
    assert _test_args(Parabola((0, 0), Line((2, 3), (4, 3))))


@SKIP("abstract class")
def test_sympy__geometry__line__LinearEntity():
    pass


def test_sympy__geometry__line__Line():
    from sympy.geometry.line import Line
    assert _test_args(Line((0, 1), (2, 3)))


def test_sympy__geometry__line__Ray():
    from sympy.geometry.line import Ray
    assert _test_args(Ray((0, 1), (2, 3)))


def test_sympy__geometry__line__Segment():
    from sympy.geometry.line import Segment
    assert _test_args(Segment((0, 1), (2, 3)))

@SKIP("abstract class")
def test_sympy__geometry__line__LinearEntity2D():
    pass


def test_sympy__geometry__line__Line2D():
    from sympy.geometry.line import Line2D
    assert _test_args(Line2D((0, 1), (2, 3)))


def test_sympy__geometry__line__Ray2D():
    from sympy.geometry.line import Ray2D
    assert _test_args(Ray2D((0, 1), (2, 3)))


def test_sympy__geometry__line__Segment2D():
    from sympy.geometry.line import Segment2D
    assert _test_args(Segment2D((0, 1), (2, 3)))


@SKIP("abstract class")
def test_sympy__geometry__line__LinearEntity3D():
    pass


def test_sympy__geometry__line__Line3D():
    from sympy.geometry.line import Line3D
    assert _test_args(Line3D((0, 1, 1), (2, 3, 4)))


def test_sympy__geometry__line__Segment3D():
    from sympy.geometry.line import Segment3D
    assert _test_args(Segment3D((0, 1, 1), (2, 3, 4)))


def test_sympy__geometry__line__Ray3D():
    from sympy.geometry.line import Ray3D
    assert _test_args(Ray3D((0, 1, 1), (2, 3, 4)))


def test_sympy__geometry__plane__Plane():
    from sympy.geometry.plane import Plane
    assert _test_args(Plane((1, 1, 1), (-3, 4, -2), (1, 2, 3)))


def test_sympy__geometry__polygon__Polygon():
    from sympy.geometry.polygon import Polygon
    assert _test_args(Polygon((0, 1), (2, 3), (4, 5), (6, 7)))


def test_sympy__geometry__polygon__RegularPolygon():
    from sympy.geometry.polygon import RegularPolygon
    assert _test_args(RegularPolygon((0, 1), 2, 3, 4))


def test_sympy__geometry__polygon__Triangle():
    from sympy.geometry.polygon import Triangle
    assert _test_args(Triangle((0, 1), (2, 3), (4, 5)))


def test_sympy__geometry__entity__GeometryEntity():
    from sympy.geometry.entity import GeometryEntity
    from sympy.geometry.point import Point
    assert _test_args(GeometryEntity(Point(1, 0), 1, [1, 2]))

@SKIP("abstract class")
def test_sympy__geometry__entity__GeometrySet():
    pass

def test_sympy__diffgeom__diffgeom__Manifold():
    from sympy.diffgeom import Manifold
    assert _test_args(Manifold('name', 3))


def test_sympy__diffgeom__diffgeom__Patch():
    from sympy.diffgeom import Manifold, Patch
    assert _test_args(Patch('name', Manifold('name', 3)))


def test_sympy__diffgeom__diffgeom__CoordSystem():
    from sympy.diffgeom import Manifold, Patch, CoordSystem
    assert _test_args(CoordSystem('name', Patch('name', Manifold('name', 3))))
    assert _test_args(CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c]))


def test_sympy__diffgeom__diffgeom__CoordinateSymbol():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, CoordinateSymbol
    assert _test_args(CoordinateSymbol(CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c]), 0))


def test_sympy__diffgeom__diffgeom__Point():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, Point
    assert _test_args(Point(
        CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c]), [x, y]))


def test_sympy__diffgeom__diffgeom__BaseScalarField():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    assert _test_args(BaseScalarField(cs, 0))


def test_sympy__diffgeom__diffgeom__BaseVectorField():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseVectorField
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    assert _test_args(BaseVectorField(cs, 0))


def test_sympy__diffgeom__diffgeom__Differential():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    assert _test_args(Differential(BaseScalarField(cs, 0)))


def test_sympy__diffgeom__diffgeom__Commutator():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseVectorField, Commutator
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    cs1 = CoordSystem('name1', Patch('name', Manifold('name', 3)), [a, b, c])
    v = BaseVectorField(cs, 0)
    v1 = BaseVectorField(cs1, 0)
    assert _test_args(Commutator(v, v1))


def test_sympy__diffgeom__diffgeom__TensorProduct():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential, TensorProduct
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    d = Differential(BaseScalarField(cs, 0))
    assert _test_args(TensorProduct(d, d))


def test_sympy__diffgeom__diffgeom__WedgeProduct():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential, WedgeProduct
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    d = Differential(BaseScalarField(cs, 0))
    d1 = Differential(BaseScalarField(cs, 1))
    assert _test_args(WedgeProduct(d, d1))


def test_sympy__diffgeom__diffgeom__LieDerivative():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential, BaseVectorField, LieDerivative
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    d = Differential(BaseScalarField(cs, 0))
    v = BaseVectorField(cs, 0)
    assert _test_args(LieDerivative(v, d))


def test_sympy__diffgeom__diffgeom__BaseCovarDerivativeOp():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseCovarDerivativeOp
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    assert _test_args(BaseCovarDerivativeOp(cs, 0, [[[0, ]*3, ]*3, ]*3))


def test_sympy__diffgeom__diffgeom__CovarDerivativeOp():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseVectorField, CovarDerivativeOp
    cs = CoordSystem('name', Patch('name', Manifold('name', 3)), [a, b, c])
    v = BaseVectorField(cs, 0)
    _test_args(CovarDerivativeOp(v, [[[0, ]*3, ]*3, ]*3))


def test_sympy__categories__baseclasses__Class():
    from sympy.categories.baseclasses import Class
    assert _test_args(Class())


def test_sympy__categories__baseclasses__Object():
    from sympy.categories import Object
    assert _test_args(Object("A"))


@SKIP("abstract class")
def test_sympy__categories__baseclasses__Morphism():
    pass


def test_sympy__categories__baseclasses__IdentityMorphism():
    from sympy.categories import Object, IdentityMorphism
    assert _test_args(IdentityMorphism(Object("A")))


def test_sympy__categories__baseclasses__NamedMorphism():
    from sympy.categories import Object, NamedMorphism
    assert _test_args(NamedMorphism(Object("A"), Object("B"), "f"))


def test_sympy__categories__baseclasses__CompositeMorphism():
    from sympy.categories import Object, NamedMorphism, CompositeMorphism
    A = Object("A")
    B = Object("B")
    C = Object("C")
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    assert _test_args(CompositeMorphism(f, g))


def test_sympy__categories__baseclasses__Diagram():
    from sympy.categories import Object, NamedMorphism, Diagram
    A = Object("A")
    B = Object("B")
    f = NamedMorphism(A, B, "f")
    d = Diagram([f])
    assert _test_args(d)


def test_sympy__categories__baseclasses__Category():
    from sympy.categories import Object, NamedMorphism, Diagram, Category
    A = Object("A")
    B = Object("B")
    C = Object("C")
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    d1 = Diagram([f, g])
    d2 = Diagram([f])
    K = Category("K", commutative_diagrams=[d1, d2])
    assert _test_args(K)


def test_sympy__physics__optics__waves__TWave():
    from sympy.physics.optics import TWave
    A, f, phi = symbols('A, f, phi')
    assert _test_args(TWave(A, f, phi))


def test_sympy__physics__optics__gaussopt__BeamParameter():
    from sympy.physics.optics import BeamParameter
    assert _test_args(BeamParameter(530e-9, 1, w=1e-3, n=1))


def test_sympy__physics__optics__medium__Medium():
    from sympy.physics.optics import Medium
    assert _test_args(Medium('m'))


def test_sympy__physics__optics__medium__MediumN():
    from sympy.physics.optics.medium import Medium
    assert _test_args(Medium('m', n=2))


def test_sympy__physics__optics__medium__MediumPP():
    from sympy.physics.optics.medium import Medium
    assert _test_args(Medium('m', permittivity=2, permeability=2))


def test_sympy__tensor__array__expressions__array_expressions__ArrayContraction():
    from sympy.tensor.array.expressions.array_expressions import ArrayContraction
    from sympy.tensor.indexed import IndexedBase
    A = symbols("A", cls=IndexedBase)
    assert _test_args(ArrayContraction(A, (0, 1)))


def test_sympy__tensor__array__expressions__array_expressions__ArrayDiagonal():
    from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
    from sympy.tensor.indexed import IndexedBase
    A = symbols("A", cls=IndexedBase)
    assert _test_args(ArrayDiagonal(A, (0, 1)))


def test_sympy__tensor__array__expressions__array_expressions__ArrayTensorProduct():
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
    from sympy.tensor.indexed import IndexedBase
    A, B = symbols("A B", cls=IndexedBase)
    assert _test_args(ArrayTensorProduct(A, B))


def test_sympy__tensor__array__expressions__array_expressions__ArrayAdd():
    from sympy.tensor.array.expressions.array_expressions import ArrayAdd
    from sympy.tensor.indexed import IndexedBase
    A, B = symbols("A B", cls=IndexedBase)
    assert _test_args(ArrayAdd(A, B))


def test_sympy__tensor__array__expressions__array_expressions__PermuteDims():
    from sympy.tensor.array.expressions.array_expressions import PermuteDims
    A = MatrixSymbol("A", 4, 4)
    assert _test_args(PermuteDims(A, (1, 0)))


def test_sympy__tensor__array__expressions__array_expressions__ArrayElementwiseApplyFunc():
    from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElementwiseApplyFunc
    A = ArraySymbol("A", (4,))
    assert _test_args(ArrayElementwiseApplyFunc(exp, A))


def test_sympy__tensor__array__expressions__array_expressions__Reshape():
    from sympy.tensor.array.expressions.array_expressions import ArraySymbol, Reshape
    A = ArraySymbol("A", (4,))
    assert _test_args(Reshape(A, (2, 2)))


def test_sympy__codegen__ast__Assignment():
    from sympy.codegen.ast import Assignment
    assert _test_args(Assignment(x, y))


def test_sympy__codegen__cfunctions__expm1():
    from sympy.codegen.cfunctions import expm1
    assert _test_args(expm1(x))


def test_sympy__codegen__cfunctions__log1p():
    from sympy.codegen.cfunctions import log1p
    assert _test_args(log1p(x))


def test_sympy__codegen__cfunctions__exp2():
    from sympy.codegen.cfunctions import exp2
    assert _test_args(exp2(x))


def test_sympy__codegen__cfunctions__log2():
    from sympy.codegen.cfunctions import log2
    assert _test_args(log2(x))


def test_sympy__codegen__cfunctions__fma():
    from sympy.codegen.cfunctions import fma
    assert _test_args(fma(x, y, z))


def test_sympy__codegen__cfunctions__log10():
    from sympy.codegen.cfunctions import log10
    assert _test_args(log10(x))


def test_sympy__codegen__cfunctions__Sqrt():
    from sympy.codegen.cfunctions import Sqrt
    assert _test_args(Sqrt(x))

def test_sympy__codegen__cfunctions__Cbrt():
    from sympy.codegen.cfunctions import Cbrt
    assert _test_args(Cbrt(x))

def test_sympy__codegen__cfunctions__hypot():
    from sympy.codegen.cfunctions import hypot
    assert _test_args(hypot(x, y))


def test_sympy__codegen__cfunctions__isnan():
    from sympy.codegen.cfunctions import isnan
    assert _test_args(isnan(x))


def test_sympy__codegen__cfunctions__isinf():
    from sympy.codegen.cfunctions import isinf
    assert _test_args(isinf(x))


def test_sympy__codegen__fnodes__FFunction():
    from sympy.codegen.fnodes import FFunction
    assert _test_args(FFunction('f'))


def test_sympy__codegen__fnodes__F95Function():
    from sympy.codegen.fnodes import F95Function
    assert _test_args(F95Function('f'))


def test_sympy__codegen__fnodes__isign():
    from sympy.codegen.fnodes import isign
    assert _test_args(isign(1, x))


def test_sympy__codegen__fnodes__dsign():
    from sympy.codegen.fnodes import dsign
    assert _test_args(dsign(1, x))


def test_sympy__codegen__fnodes__cmplx():
    from sympy.codegen.fnodes import cmplx
    assert _test_args(cmplx(x, y))


def test_sympy__codegen__fnodes__kind():
    from sympy.codegen.fnodes import kind
    assert _test_args(kind(x))


def test_sympy__codegen__fnodes__merge():
    from sympy.codegen.fnodes import merge
    assert _test_args(merge(1, 2, Eq(x, 0)))


def test_sympy__codegen__fnodes___literal():
    from sympy.codegen.fnodes import _literal
    assert _test_args(_literal(1))


def test_sympy__codegen__fnodes__literal_sp():
    from sympy.codegen.fnodes import literal_sp
    assert _test_args(literal_sp(1))


def test_sympy__codegen__fnodes__literal_dp():
    from sympy.codegen.fnodes import literal_dp
    assert _test_args(literal_dp(1))


def test_sympy__codegen__matrix_nodes__MatrixSolve():
    from sympy.matrices import MatrixSymbol
    from sympy.codegen.matrix_nodes import MatrixSolve
    A = MatrixSymbol('A', 3, 3)
    v = MatrixSymbol('x', 3, 1)
    assert _test_args(MatrixSolve(A, v))


def test_sympy__printing__rust__TypeCast():
    from sympy.printing.rust import TypeCast
    from sympy.codegen.ast import real
    assert _test_args(TypeCast(x, real))


def test_sympy__printing__rust__float_floor():
    from sympy.printing.rust import float_floor
    assert _test_args(float_floor(x))


def test_sympy__printing__rust__float_ceiling():
    from sympy.printing.rust import float_ceiling
    assert _test_args(float_ceiling(x))


def test_sympy__vector__coordsysrect__CoordSys3D():
    from sympy.vector.coordsysrect import CoordSys3D
    assert _test_args(CoordSys3D('C'))


def test_sympy__vector__point__Point():
    from sympy.vector.point import Point
    assert _test_args(Point('P'))


def test_sympy__vector__basisdependent__BasisDependent():
    #from sympy.vector.basisdependent import BasisDependent
    #These classes have been created to maintain an OOP hierarchy
    #for Vectors and Dyadics. Are NOT meant to be initialized
    pass


def test_sympy__vector__basisdependent__BasisDependentMul():
    #from sympy.vector.basisdependent import BasisDependentMul
    #These classes have been created to maintain an OOP hierarchy
    #for Vectors and Dyadics. Are NOT meant to be initialized
    pass


def test_sympy__vector__basisdependent__BasisDependentAdd():
    #from sympy.vector.basisdependent import BasisDependentAdd
    #These classes have been created to maintain an OOP hierarchy
    #for Vectors and Dyadics. Are NOT meant to be initialized
    pass


def test_sympy__vector__basisdependent__BasisDependentZero():
    #from sympy.vector.basisdependent import BasisDependentZero
    #These classes have been created to maintain an OOP hierarchy
    #for Vectors and Dyadics. Are NOT meant to be initialized
    pass


def test_sympy__vector__vector__BaseVector():
    from sympy.vector.vector import BaseVector
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(BaseVector(0, C, ' ', ' '))


def test_sympy__vector__vector__VectorAdd():
    from sympy.vector.vector import VectorAdd, VectorMul
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    from sympy.abc import a, b, c, x, y, z
    v1 = a*C.i + b*C.j + c*C.k
    v2 = x*C.i + y*C.j + z*C.k
    assert _test_args(VectorAdd(v1, v2))
    assert _test_args(VectorMul(x, v1))


def test_sympy__vector__vector__VectorMul():
    from sympy.vector.vector import VectorMul
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    from sympy.abc import a
    assert _test_args(VectorMul(a, C.i))


def test_sympy__vector__vector__VectorZero():
    from sympy.vector.vector import VectorZero
    assert _test_args(VectorZero())


def test_sympy__vector__vector__Vector():
    #from sympy.vector.vector import Vector
    #Vector is never to be initialized using args
    pass


def test_sympy__vector__vector__Cross():
    from sympy.vector.vector import Cross
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    _test_args(Cross(C.i, C.j))


def test_sympy__vector__vector__Dot():
    from sympy.vector.vector import Dot
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    _test_args(Dot(C.i, C.j))


def test_sympy__vector__dyadic__Dyadic():
    #from sympy.vector.dyadic import Dyadic
    #Dyadic is never to be initialized using args
    pass


def test_sympy__vector__dyadic__BaseDyadic():
    from sympy.vector.dyadic import BaseDyadic
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(BaseDyadic(C.i, C.j))


def test_sympy__vector__dyadic__DyadicMul():
    from sympy.vector.dyadic import BaseDyadic, DyadicMul
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(DyadicMul(3, BaseDyadic(C.i, C.j)))


def test_sympy__vector__dyadic__DyadicAdd():
    from sympy.vector.dyadic import BaseDyadic, DyadicAdd
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(2 * DyadicAdd(BaseDyadic(C.i, C.i),
                                    BaseDyadic(C.i, C.j)))


def test_sympy__vector__dyadic__DyadicZero():
    from sympy.vector.dyadic import DyadicZero
    assert _test_args(DyadicZero())


def test_sympy__vector__deloperator__Del():
    from sympy.vector.deloperator import Del
    assert _test_args(Del())


def test_sympy__vector__implicitregion__ImplicitRegion():
    from sympy.vector.implicitregion import ImplicitRegion
    from sympy.abc import x, y
    assert _test_args(ImplicitRegion((x, y), y**3 - 4*x))


def test_sympy__vector__integrals__ParametricIntegral():
    from sympy.vector.integrals import ParametricIntegral
    from sympy.vector.parametricregion import ParametricRegion
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(ParametricIntegral(C.y*C.i - 10*C.j,\
                    ParametricRegion((x, y), (x, 1, 3), (y, -2, 2))))

def test_sympy__vector__operators__Curl():
    from sympy.vector.operators import Curl
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(Curl(C.i))


def test_sympy__vector__operators__Laplacian():
    from sympy.vector.operators import Laplacian
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(Laplacian(C.i))


def test_sympy__vector__operators__Divergence():
    from sympy.vector.operators import Divergence
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(Divergence(C.i))


def test_sympy__vector__operators__Gradient():
    from sympy.vector.operators import Gradient
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(Gradient(C.x))


def test_sympy__vector__orienters__Orienter():
    #from sympy.vector.orienters import Orienter
    #Not to be initialized
    pass


def test_sympy__vector__orienters__ThreeAngleOrienter():
    #from sympy.vector.orienters import ThreeAngleOrienter
    #Not to be initialized
    pass


def test_sympy__vector__orienters__AxisOrienter():
    from sympy.vector.orienters import AxisOrienter
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(AxisOrienter(x, C.i))


def test_sympy__vector__orienters__BodyOrienter():
    from sympy.vector.orienters import BodyOrienter
    assert _test_args(BodyOrienter(x, y, z, '123'))


def test_sympy__vector__orienters__SpaceOrienter():
    from sympy.vector.orienters import SpaceOrienter
    assert _test_args(SpaceOrienter(x, y, z, '123'))


def test_sympy__vector__orienters__QuaternionOrienter():
    from sympy.vector.orienters import QuaternionOrienter
    a, b, c, d = symbols('a b c d')
    assert _test_args(QuaternionOrienter(a, b, c, d))


def test_sympy__vector__parametricregion__ParametricRegion():
    from sympy.abc import t
    from sympy.vector.parametricregion import ParametricRegion
    assert _test_args(ParametricRegion((t, t**3), (t, 0, 2)))


def test_sympy__vector__scalar__BaseScalar():
    from sympy.vector.scalar import BaseScalar
    from sympy.vector.coordsysrect import CoordSys3D
    C = CoordSys3D('C')
    assert _test_args(BaseScalar(0, C, ' ', ' '))


def test_sympy__physics__wigner__Wigner3j():
    from sympy.physics.wigner import Wigner3j
    assert _test_args(Wigner3j(0, 0, 0, 0, 0, 0))


def test_sympy__combinatorics__schur_number__SchurNumber():
    from sympy.combinatorics.schur_number import SchurNumber
    assert _test_args(SchurNumber(x))


def test_sympy__combinatorics__perm_groups__SymmetricPermutationGroup():
    from sympy.combinatorics.perm_groups import SymmetricPermutationGroup
    assert _test_args(SymmetricPermutationGroup(5))


def test_sympy__combinatorics__perm_groups__Coset():
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.perm_groups import PermutationGroup, Coset
    a = Permutation(1, 2)
    b = Permutation(0, 1)
    G = PermutationGroup([a, b])
    assert _test_args(Coset(a, G))