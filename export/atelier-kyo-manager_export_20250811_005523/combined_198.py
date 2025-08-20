
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\multiarray.py ===
from numpy._core import multiarray

# these must import without warning or error from numpy.core.multiarray to
# support old pickle files
for item in ["_reconstruct", "scalar"]:
    globals()[item] = getattr(multiarray, item)

# Pybind11 (in versions <= 2.11.1) imports _ARRAY_API from the multiarray
# submodule as a part of NumPy initialization, therefore it must be importable
# without a warning.
_ARRAY_API = multiarray._ARRAY_API

def __getattr__(attr_name):
    from numpy._core import multiarray

    from ._utils import _raise_warning
    ret = getattr(multiarray, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.multiarray' has no attribute {attr_name}")
    _raise_warning(attr_name, "multiarray")
    return ret


del multiarray

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_gelu_approximation.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from fusion_base import Fusion
from onnx import helper
from onnx_model import OnnxModel


class FusionGeluApproximation(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "FastGelu", ["Gelu", "BiasGelu"], "GeluApproximation")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        new_node = helper.make_node(
            "FastGelu",
            inputs=node.input,
            outputs=node.output,
            name=self.model.create_node_name("FastGelu", node.op_type + "_Approximation"),
        )
        new_node.domain = "com.microsoft"
        self.nodes_to_remove.append(node)
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\compat\strings.py ===
# Copyright (c) 2010-2024 openpyxl

from datetime import datetime
from math import isnan, isinf
import sys

VER = sys.version_info

from .numbers import NUMERIC_TYPES


def safe_string(value):
    """Safely and consistently format numeric values"""
    if isinstance(value, NUMERIC_TYPES):
        if isnan(value) or isinf(value):
            value = ""
        else:
            value = "%.16g" % value
    elif value is None:
        value = "none"
    elif isinstance(value, datetime):
        value = value.isoformat()
    elif not isinstance(value, str):
        value = str(value)
    return value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_config\dates.py ===
"""
config for datetime formatting
"""
from __future__ import annotations

from pandas._config import config as cf

pc_date_dayfirst_doc = """
: boolean
    When True, prints and parses dates with the day first, eg 20/01/2005
"""

pc_date_yearfirst_doc = """
: boolean
    When True, prints and parses dates with the year first, eg 2005/01/20
"""

with cf.config_prefix("display"):
    # Needed upstream of `_libs` because these are used in tslibs.parsing
    cf.register_option(
        "date_dayfirst", False, pc_date_dayfirst_doc, validator=cf.is_bool
    )
    cf.register_option(
        "date_yearfirst", False, pc_date_yearfirst_doc, validator=cf.is_bool
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\models\candidate.py ===
from dataclasses import dataclass

from pip._vendor.packaging.version import Version
from pip._vendor.packaging.version import parse as parse_version

from pip._internal.models.link import Link


@dataclass(frozen=True)
class InstallationCandidate:
    """Represents a potential "candidate" for installation."""

    __slots__ = ["name", "version", "link"]

    name: str
    version: Version
    link: Link

    def __init__(self, name: str, version: str, link: Link) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", parse_version(version))
        object.__setattr__(self, "link", link)

    def __str__(self) -> str:
        return f"{self.name!r} candidate (version {self.version} at {self.link})"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\models\scheme.py ===
"""
For types associated with installation schemes.

For a general overview of available schemes and their context, see
https://docs.python.org/3/install/index.html#alternate-installation.
"""

from dataclasses import dataclass

SCHEME_KEYS = ["platlib", "purelib", "headers", "scripts", "data"]


@dataclass(frozen=True)
class Scheme:
    """A Scheme holds paths which are used as the base directories for
    artifacts associated with a Python package.
    """

    __slots__ = SCHEME_KEYS

    platlib: str
    purelib: str
    headers: str
    scripts: str
    data: str

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\packages.py ===
import sys

from .compat import chardet

# This code exists for backwards compatibility reasons.
# I don't like it either. Just look the other way. :)

for package in ("urllib3", "idna"):
    vendored_package = "pip._vendor." + package
    locals()[package] = __import__(vendored_package)
    # This traversal is apparently necessary such that the identities are
    # preserved (requests.packages.urllib3.* is urllib3.*)
    for mod in list(sys.modules):
        if mod == vendored_package or mod.startswith(vendored_package + '.'):
            unprefixed_mod = mod[len("pip._vendor."):]
            sys.modules['pip._vendor.requests.packages.' + unprefixed_mod] = sys.modules[mod]

if chardet is not None:
    target = chardet.__name__
    for mod in list(sys.modules):
        if mod == target or mod.startswith(f"{target}."):
            imported_mod = sys.modules[mod]
            sys.modules[f"requests.packages.{mod}"] = imported_mod
            mod = mod.replace(target, "chardet")
            sys.modules[f"requests.packages.{mod}"] = imported_mod

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\types.py ===
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
"""Selenium type definitions."""

from typing import IO, Any, Iterable, Type, Union

AnyKey = Union[str, int, float]
WaitExcTypes = Iterable[Type[Exception]]

# Service Types
SubprocessStdAlias = Union[int, str, IO[Any]]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\event\__init__.py ===
# event/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from .api import CANCEL as CANCEL
from .api import contains as contains
from .api import listen as listen
from .api import listens_for as listens_for
from .api import NO_RETVAL as NO_RETVAL
from .api import remove as remove
from .attr import _InstanceLevelDispatch as _InstanceLevelDispatch
from .attr import RefCollection as RefCollection
from .base import _Dispatch as _Dispatch
from .base import _DispatchCommon as _DispatchCommon
from .base import dispatcher as dispatcher
from .base import Events as Events
from .legacy import _legacy_signature as _legacy_signature
from .registry import _EventKey as _EventKey
from .registry import _ListenerFnType as _ListenerFnType
from .registry import EventTarget as EventTarget

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\asyncio\__init__.py ===
# ext/asyncio/__init__.py
# Copyright (C) 2020-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from .engine import async_engine_from_config as async_engine_from_config
from .engine import AsyncConnection as AsyncConnection
from .engine import AsyncEngine as AsyncEngine
from .engine import AsyncTransaction as AsyncTransaction
from .engine import create_async_engine as create_async_engine
from .engine import create_async_pool_from_url as create_async_pool_from_url
from .result import AsyncMappingResult as AsyncMappingResult
from .result import AsyncResult as AsyncResult
from .result import AsyncScalarResult as AsyncScalarResult
from .result import AsyncTupleResult as AsyncTupleResult
from .scoping import async_scoped_session as async_scoped_session
from .session import async_object_session as async_object_session
from .session import async_session as async_session
from .session import async_sessionmaker as async_sessionmaker
from .session import AsyncAttrs as AsyncAttrs
from .session import AsyncSession as AsyncSession
from .session import AsyncSessionTransaction as AsyncSessionTransaction
from .session import close_all_sessions as close_all_sessions

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\__init__.py ===
"""Calculus-related methods."""

from .euler import euler_equations
from .singularities import (singularities, is_increasing,
                            is_strictly_increasing, is_decreasing,
                            is_strictly_decreasing, is_monotonic)
from .finite_diff import finite_diff_weights, apply_finite_diff, differentiate_finite
from .util import (periodicity, not_empty_in, is_convex,
                   stationary_points, minimum, maximum)
from .accumulationbounds import AccumBounds

__all__ = [
'euler_equations',

'singularities', 'is_increasing',
'is_strictly_increasing', 'is_decreasing',
'is_strictly_decreasing', 'is_monotonic',

'finite_diff_weights', 'apply_finite_diff', 'differentiate_finite',

'periodicity', 'not_empty_in', 'is_convex', 'stationary_points',
'minimum', 'maximum',

'AccumBounds'
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\cartan_matrix.py ===
from .cartan_type import CartanType

def CartanMatrix(ct):
    """Access the Cartan matrix of a specific Lie algebra

    Examples
    ========

    >>> from sympy.liealgebras.cartan_matrix import CartanMatrix
    >>> CartanMatrix("A2")
    Matrix([
    [ 2, -1],
    [-1,  2]])

    >>> CartanMatrix(['C', 3])
    Matrix([
    [ 2, -1,  0],
    [-1,  2, -1],
    [ 0, -2,  2]])

    This method works by returning the Cartan matrix
    which corresponds to Cartan type t.
    """

    return CartanType(ct).cartan_matrix()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_system_class.py ===
import pytest

from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.dense import eye, zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.physics.mechanics import (
    Force, KanesMethod, LagrangesMethod, Particle, PinJoint, Point,
    PrismaticJoint, ReferenceFrame, RigidBody, Torque, TorqueActuator, System,
    dynamicsymbols)
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve

t = dynamicsymbols._t  # type: ignore
q = dynamicsymbols('q:6')  # type: ignore
qd = dynamicsymbols('q:6', 1)  # type: ignore
u = dynamicsymbols('u:6')  # type: ignore
ua = dynamicsymbols('ua:3')  # type: ignore


class TestSystemBase:
    @pytest.fixture()
    def _empty_system_setup(self):
        self.system = System(ReferenceFrame('frame'), Point('fixed_point'))

    def _empty_system_check(self, exclude=()):
        matrices = ('q_ind', 'q_dep', 'q', 'u_ind', 'u_dep', 'u', 'u_aux',
                    'kdes', 'holonomic_constraints', 'nonholonomic_constraints')
        tuples = ('loads', 'bodies', 'joints', 'actuators')
        for attr in matrices:
            if attr not in exclude:
                assert getattr(self.system, attr)[:] == []
        for attr in tuples:
            if attr not in exclude:
                assert getattr(self.system, attr) == ()
        if 'eom_method' not in exclude:
            assert self.system.eom_method is None

    def _create_filled_system(self, with_speeds=True):
        self.system = System(ReferenceFrame('frame'), Point('fixed_point'))
        u = dynamicsymbols('u:6') if with_speeds else qd
        self.bodies = symbols('rb1:5', cls=RigidBody)
        self.joints = (
            PinJoint('J1', self.bodies[0], self.bodies[1], q[0], u[0]),
            PrismaticJoint('J2', self.bodies[1], self.bodies[2], q[1], u[1]),
            PinJoint('J3', self.bodies[2], self.bodies[3], q[2], u[2])
        )
        self.system.add_joints(*self.joints)
        self.system.add_coordinates(q[3], independent=[False])
        self.system.add_speeds(u[3], independent=False)
        if with_speeds:
            self.system.add_kdes(u[3] - qd[3])
            self.system.add_auxiliary_speeds(ua[0], ua[1])
        self.system.add_holonomic_constraints(q[2] - q[0] + q[1])
        self.system.add_nonholonomic_constraints(u[3] - qd[1] + u[2])
        self.system.u_ind = u[:2]
        self.system.u_dep = u[2:4]
        self.q_ind, self.q_dep = self.system.q_ind[:], self.system.q_dep[:]
        self.u_ind, self.u_dep = self.system.u_ind[:], self.system.u_dep[:]
        self.kdes = self.system.kdes[:]
        self.hc = self.system.holonomic_constraints[:]
        self.vc = self.system.velocity_constraints[:]
        self.nhc = self.system.nonholonomic_constraints[:]

    @pytest.fixture()
    def _filled_system_setup(self):
        self._create_filled_system(with_speeds=True)

    @pytest.fixture()
    def _filled_system_setup_no_speeds(self):
        self._create_filled_system(with_speeds=False)

    def _filled_system_check(self, exclude=()):
        assert 'q_ind' in exclude or self.system.q_ind[:] == q[:3]
        assert 'q_dep' in exclude or self.system.q_dep[:] == [q[3]]
        assert 'q' in exclude or self.system.q[:] == q[:4]
        assert 'u_ind' in exclude or self.system.u_ind[:] == u[:2]
        assert 'u_dep' in exclude or self.system.u_dep[:] == u[2:4]
        assert 'u' in exclude or self.system.u[:] == u[:4]
        assert 'u_aux' in exclude or self.system.u_aux[:] == ua[:2]
        assert 'kdes' in exclude or self.system.kdes[:] == [
            ui - qdi for ui, qdi in zip(u[:4], qd[:4])]
        assert ('holonomic_constraints' in exclude or
                self.system.holonomic_constraints[:] == [q[2] - q[0] + q[1]])
        assert ('nonholonomic_constraints' in exclude or
                self.system.nonholonomic_constraints[:] == [u[3] - qd[1] + u[2]]
                )
        assert ('velocity_constraints' in exclude or
                self.system.velocity_constraints[:] == [
                    qd[2] - qd[0] + qd[1], u[3] - qd[1] + u[2]])
        assert ('bodies' in exclude or
                self.system.bodies == tuple(self.bodies))
        assert ('joints' in exclude or
                self.system.joints == tuple(self.joints))

    @pytest.fixture()
    def _moving_point_mass(self, _empty_system_setup):
        self.system.q_ind = q[0]
        self.system.u_ind = u[0]
        self.system.kdes = u[0] - q[0].diff(t)
        p = Particle('p', mass=symbols('m'))
        self.system.add_bodies(p)
        p.masscenter.set_pos(self.system.fixed_point, q[0] * self.system.x)


class TestSystem(TestSystemBase):
    def test_empty_system(self, _empty_system_setup):
        self._empty_system_check()
        self.system.validate_system()

    def test_filled_system(self, _filled_system_setup):
        self._filled_system_check()
        self.system.validate_system()

    @pytest.mark.parametrize('frame', [None, ReferenceFrame('frame')])
    @pytest.mark.parametrize('fixed_point', [None, Point('fixed_point')])
    def test_init(self, frame, fixed_point):
        if fixed_point is None and frame is None:
            self.system = System()
        else:
            self.system = System(frame, fixed_point)
        if fixed_point is None:
            assert self.system.fixed_point.name == 'inertial_point'
        else:
            assert self.system.fixed_point == fixed_point
        if frame is None:
            assert self.system.frame.name == 'inertial_frame'
        else:
            assert self.system.frame == frame
        self._empty_system_check()
        assert isinstance(self.system.q_ind, ImmutableMatrix)
        assert isinstance(self.system.q_dep, ImmutableMatrix)
        assert isinstance(self.system.q, ImmutableMatrix)
        assert isinstance(self.system.u_ind, ImmutableMatrix)
        assert isinstance(self.system.u_dep, ImmutableMatrix)
        assert isinstance(self.system.u, ImmutableMatrix)
        assert isinstance(self.system.kdes, ImmutableMatrix)
        assert isinstance(self.system.holonomic_constraints, ImmutableMatrix)
        assert isinstance(self.system.nonholonomic_constraints, ImmutableMatrix)

    def test_from_newtonian_rigid_body(self):
        rb = RigidBody('body')
        self.system = System.from_newtonian(rb)
        assert self.system.fixed_point == rb.masscenter
        assert self.system.frame == rb.frame
        self._empty_system_check(exclude=('bodies',))
        self.system.bodies = (rb,)

    def test_from_newtonian_particle(self):
        pt = Particle('particle')
        with pytest.raises(TypeError):
            System.from_newtonian(pt)

    @pytest.mark.parametrize('args, kwargs, exp_q_ind, exp_q_dep, exp_q', [
        (q[:3], {}, q[:3], [], q[:3]),
        (q[:3], {'independent': True}, q[:3], [], q[:3]),
        (q[:3], {'independent': False}, [], q[:3], q[:3]),
        (q[:3], {'independent': [True, False, True]}, [q[0], q[2]], [q[1]],
         [q[0], q[2], q[1]]),
    ])
    def test_coordinates(self, _empty_system_setup, args, kwargs,
                         exp_q_ind, exp_q_dep, exp_q):
        # Test add_coordinates
        self.system.add_coordinates(*args, **kwargs)
        assert self.system.q_ind[:] == exp_q_ind
        assert self.system.q_dep[:] == exp_q_dep
        assert self.system.q[:] == exp_q
        self._empty_system_check(exclude=('q_ind', 'q_dep', 'q'))
        # Test setter for q_ind and q_dep
        self.system.q_ind = exp_q_ind
        self.system.q_dep = exp_q_dep
        assert self.system.q_ind[:] == exp_q_ind
        assert self.system.q_dep[:] == exp_q_dep
        assert self.system.q[:] == exp_q
        self._empty_system_check(exclude=('q_ind', 'q_dep', 'q'))

    @pytest.mark.parametrize('func', ['add_coordinates', 'add_speeds'])
    @pytest.mark.parametrize('args, kwargs', [
        ((q[0], q[5]), {}),
        ((u[0], u[5]), {}),
        ((q[0],), {'independent': False}),
        ((u[0],), {'independent': False}),
        ((u[0], q[5]), {}),
        ((symbols('a'), q[5]), {}),
    ])
    def test_coordinates_speeds_invalid(self, _filled_system_setup, func, args,
                                        kwargs):
        with pytest.raises(ValueError):
            getattr(self.system, func)(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('args, kwargs, exp_u_ind, exp_u_dep, exp_u', [
        (u[:3], {}, u[:3], [], u[:3]),
        (u[:3], {'independent': True}, u[:3], [], u[:3]),
        (u[:3], {'independent': False}, [], u[:3], u[:3]),
        (u[:3], {'independent': [True, False, True]}, [u[0], u[2]], [u[1]],
         [u[0], u[2], u[1]]),
    ])
    def test_speeds(self, _empty_system_setup, args, kwargs, exp_u_ind,
                    exp_u_dep, exp_u):
        # Test add_speeds
        self.system.add_speeds(*args, **kwargs)
        assert self.system.u_ind[:] == exp_u_ind
        assert self.system.u_dep[:] == exp_u_dep
        assert self.system.u[:] == exp_u
        self._empty_system_check(exclude=('u_ind', 'u_dep', 'u'))
        # Test setter for u_ind and u_dep
        self.system.u_ind = exp_u_ind
        self.system.u_dep = exp_u_dep
        assert self.system.u_ind[:] == exp_u_ind
        assert self.system.u_dep[:] == exp_u_dep
        assert self.system.u[:] == exp_u
        self._empty_system_check(exclude=('u_ind', 'u_dep', 'u'))

    @pytest.mark.parametrize('args, kwargs, exp_u_aux', [
        (ua[:3], {}, ua[:3]),
    ])
    def test_auxiliary_speeds(self, _empty_system_setup, args, kwargs,
                              exp_u_aux):
        # Test add_speeds
        self.system.add_auxiliary_speeds(*args, **kwargs)
        assert self.system.u_aux[:] == exp_u_aux
        self._empty_system_check(exclude=('u_aux',))
        # Test setter for u_ind and u_dep
        self.system.u_aux = exp_u_aux
        assert self.system.u_aux[:] == exp_u_aux
        self._empty_system_check(exclude=('u_aux',))

    @pytest.mark.parametrize('args, kwargs', [
        ((ua[2], q[0]), {}),
        ((ua[2], u[1]), {}),
        ((ua[0], ua[2]), {}),
        ((symbols('a'), ua[2]), {}),
    ])
    def test_auxiliary_invalid(self, _filled_system_setup, args, kwargs):
        with pytest.raises(ValueError):
            self.system.add_auxiliary_speeds(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('prop, add_func, args, kwargs', [
        ('q_ind', 'add_coordinates', (q[0],), {}),
        ('q_dep', 'add_coordinates', (q[3],), {'independent': False}),
        ('u_ind', 'add_speeds', (u[0],), {}),
        ('u_dep', 'add_speeds', (u[3],), {'independent': False}),
        ('u_aux', 'add_auxiliary_speeds', (ua[2],), {}),
        ('kdes', 'add_kdes', (qd[0] - u[0],), {}),
        ('holonomic_constraints', 'add_holonomic_constraints',
         (q[0] - q[1],), {}),
        ('nonholonomic_constraints', 'add_nonholonomic_constraints',
         (u[0] - u[1],), {}),
        ('bodies', 'add_bodies', (RigidBody('body'),), {}),
        ('loads', 'add_loads', (Force(Point('P'), ReferenceFrame('N').x),), {}),
        ('actuators', 'add_actuators', (TorqueActuator(
            symbols('T'), ReferenceFrame('N').x, ReferenceFrame('A')),), {}),
    ])
    def test_add_after_reset(self, _filled_system_setup, prop, add_func, args,
                             kwargs):
        setattr(self.system, prop, ())
        exclude = (prop, 'q', 'u')
        if prop in ('holonomic_constraints', 'nonholonomic_constraints'):
            exclude += ('velocity_constraints',)
        self._filled_system_check(exclude=exclude)
        assert list(getattr(self.system, prop)[:]) == []
        getattr(self.system, add_func)(*args, **kwargs)
        assert list(getattr(self.system, prop)[:]) == list(args)

    @pytest.mark.parametrize('prop, add_func, value, error', [
        ('q_ind', 'add_coordinates', symbols('a'), ValueError),
        ('q_dep', 'add_coordinates', symbols('a'), ValueError),
        ('u_ind', 'add_speeds', symbols('a'), ValueError),
        ('u_dep', 'add_speeds', symbols('a'), ValueError),
        ('u_aux', 'add_auxiliary_speeds', symbols('a'), ValueError),
        ('kdes', 'add_kdes', 7, TypeError),
        ('holonomic_constraints', 'add_holonomic_constraints', 7, TypeError),
        ('nonholonomic_constraints', 'add_nonholonomic_constraints', 7,
         TypeError),
        ('bodies', 'add_bodies', symbols('a'), TypeError),
        ('loads', 'add_loads', symbols('a'), TypeError),
        ('actuators', 'add_actuators', symbols('a'), TypeError),
    ])
    def test_type_error(self, _filled_system_setup, prop, add_func, value,
                        error):
        with pytest.raises(error):
            getattr(self.system, add_func)(value)
        with pytest.raises(error):
            setattr(self.system, prop, value)
        self._filled_system_check()

    @pytest.mark.parametrize('args, kwargs, exp_kdes', [
        ((), {}, [ui - qdi for ui, qdi in zip(u[:4], qd[:4])]),
        ((u[4] - qd[4], u[5] - qd[5]), {},
         [ui - qdi for ui, qdi in zip(u[:6], qd[:6])]),
    ])
    def test_kdes(self, _filled_system_setup, args, kwargs, exp_kdes):
        # Test add_speeds
        self.system.add_kdes(*args, **kwargs)
        self._filled_system_check(exclude=('kdes',))
        assert self.system.kdes[:] == exp_kdes
        # Test setter for kdes
        self.system.kdes = exp_kdes
        self._filled_system_check(exclude=('kdes',))
        assert self.system.kdes[:] == exp_kdes

    @pytest.mark.parametrize('args, kwargs', [
        ((u[0] - qd[0], u[4] - qd[4]), {}),
        ((-(u[0] - qd[0]), u[4] - qd[4]), {}),
        (([u[0] - u[0], u[4] - qd[4]]), {}),
    ])
    def test_kdes_invalid(self, _filled_system_setup, args, kwargs):
        with pytest.raises(ValueError):
            self.system.add_kdes(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('args, kwargs, exp_con', [
        ((), {}, [q[2] - q[0] + q[1]]),
        ((q[4] - q[5], q[5] + q[3]), {},
         [q[2] - q[0] + q[1], q[4] - q[5], q[5] + q[3]]),
    ])
    def test_holonomic_constraints(self, _filled_system_setup, args, kwargs,
                                   exp_con):
        exclude = ('holonomic_constraints', 'velocity_constraints')
        exp_vel_con = [c.diff(t) for c in exp_con] + self.nhc
        # Test add_holonomic_constraints
        self.system.add_holonomic_constraints(*args, **kwargs)
        self._filled_system_check(exclude=exclude)
        assert self.system.holonomic_constraints[:] == exp_con
        assert self.system.velocity_constraints[:] == exp_vel_con
        # Test setter for holonomic_constraints
        self.system.holonomic_constraints = exp_con
        self._filled_system_check(exclude=exclude)
        assert self.system.holonomic_constraints[:] == exp_con
        assert self.system.velocity_constraints[:] == exp_vel_con

    @pytest.mark.parametrize('args, kwargs', [
        ((q[2] - q[0] + q[1], q[4] - q[3]), {}),
        ((-(q[2] - q[0] + q[1]), q[4] - q[3]), {}),
        ((q[0] - q[0], q[4] - q[3]), {}),
    ])
    def test_holonomic_constraints_invalid(self, _filled_system_setup, args,
                                           kwargs):
        with pytest.raises(ValueError):
            self.system.add_holonomic_constraints(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('args, kwargs, exp_con', [
        ((), {}, [u[3] - qd[1] + u[2]]),
        ((u[4] - u[5], u[5] + u[3]), {},
         [u[3] - qd[1] + u[2], u[4] - u[5], u[5] + u[3]]),
    ])
    def test_nonholonomic_constraints(self, _filled_system_setup, args, kwargs,
                                      exp_con):
        exclude = ('nonholonomic_constraints', 'velocity_constraints')
        exp_vel_con = self.vc[:len(self.hc)] + exp_con
        # Test add_nonholonomic_constraints
        self.system.add_nonholonomic_constraints(*args, **kwargs)
        self._filled_system_check(exclude=exclude)
        assert self.system.nonholonomic_constraints[:] == exp_con
        assert self.system.velocity_constraints[:] == exp_vel_con
        # Test setter for nonholonomic_constraints
        self.system.nonholonomic_constraints = exp_con
        self._filled_system_check(exclude=exclude)
        assert self.system.nonholonomic_constraints[:] == exp_con
        assert self.system.velocity_constraints[:] == exp_vel_con

    @pytest.mark.parametrize('args, kwargs', [
        ((u[3] - qd[1] + u[2], u[4] - u[3]), {}),
        ((-(u[3] - qd[1] + u[2]), u[4] - u[3]), {}),
        ((u[0] - u[0], u[4] - u[3]), {}),
        (([u[0] - u[0], u[4] - u[3]]), {}),
    ])
    def test_nonholonomic_constraints_invalid(self, _filled_system_setup, args,
                                              kwargs):
        with pytest.raises(ValueError):
            self.system.add_nonholonomic_constraints(*args, **kwargs)
        self._filled_system_check()

    @pytest.mark.parametrize('constraints, expected', [
        ([], []),
        (qd[2] - qd[0] + qd[1], [qd[2] - qd[0] + qd[1]]),
        ([qd[2] + qd[1], u[2] - u[1]], [qd[2] + qd[1], u[2] - u[1]]),
    ])
    def test_velocity_constraints_overwrite(self, _filled_system_setup,
                                            constraints, expected):
        self.system.velocity_constraints = constraints
        self._filled_system_check(exclude=('velocity_constraints',))
        assert self.system.velocity_constraints[:] == expected

    def test_velocity_constraints_back_to_auto(self, _filled_system_setup):
        self.system.velocity_constraints = qd[3] - qd[2]
        self._filled_system_check(exclude=('velocity_constraints',))
        assert self.system.velocity_constraints[:] == [qd[3] - qd[2]]
        self.system.velocity_constraints = None
        self._filled_system_check()

    def test_bodies(self, _filled_system_setup):
        rb1, rb2 = RigidBody('rb1'), RigidBody('rb2')
        p1, p2 = Particle('p1'), Particle('p2')
        self.system.add_bodies(rb1, p1)
        assert self.system.bodies == (*self.bodies, rb1, p1)
        self.system.add_bodies(p2)
        assert self.system.bodies == (*self.bodies, rb1, p1, p2)
        self.system.bodies = []
        assert self.system.bodies == ()
        self.system.bodies = p2
        assert self.system.bodies == (p2,)
        symb = symbols('symb')
        pytest.raises(TypeError, lambda: self.system.add_bodies(symb))
        pytest.raises(ValueError, lambda: self.system.add_bodies(p2))
        with pytest.raises(TypeError):
            self.system.bodies = (rb1, rb2, p1, p2, symb)
        assert self.system.bodies == (p2,)

    def test_add_loads(self):
        system = System()
        N, A = ReferenceFrame('N'), ReferenceFrame('A')
        rb1 = RigidBody('rb1', frame=N)
        mc1 = Point('mc1')
        p1 = Particle('p1', mc1)
        system.add_loads(Torque(rb1, N.x), (mc1, A.x), Force(p1, A.x))
        assert system.loads == ((N, N.x), (mc1, A.x), (mc1, A.x))
        system.loads = [(A, A.x)]
        assert system.loads == ((A, A.x),)
        pytest.raises(ValueError, lambda: system.add_loads((N, N.x, N.y)))
        with pytest.raises(TypeError):
            system.loads = (N, N.x)
        assert system.loads == ((A, A.x),)

    def test_add_actuators(self):
        system = System()
        N, A = ReferenceFrame('N'), ReferenceFrame('A')
        act1 = TorqueActuator(symbols('T1'), N.x, N)
        act2 = TorqueActuator(symbols('T2'), N.y, N, A)
        system.add_actuators(act1)
        assert system.actuators == (act1,)
        assert system.loads == ()
        system.actuators = (act2,)
        assert system.actuators == (act2,)

    def test_add_joints(self):
        q1, q2, q3, q4, u1, u2, u3 = dynamicsymbols('q1:5 u1:4')
        rb1, rb2, rb3, rb4, rb5 = symbols('rb1:6', cls=RigidBody)
        J1 = PinJoint('J1', rb1, rb2, q1, u1)
        J2 = PrismaticJoint('J2', rb2, rb3, q2, u2)
        J3 = PinJoint('J3', rb3, rb4, q3, u3)
        J_lag = PinJoint('J_lag', rb4, rb5, q4, q4.diff(t))
        system = System()
        system.add_joints(J1)
        assert system.joints == (J1,)
        assert system.bodies == (rb1, rb2)
        assert system.q_ind == ImmutableMatrix([q1])
        assert system.u_ind == ImmutableMatrix([u1])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t)])
        system.add_bodies(rb4)
        system.add_coordinates(q3)
        system.add_kdes(u3 - q3.diff(t))
        system.add_joints(J3)
        assert system.joints == (J1, J3)
        assert system.bodies == (rb1, rb2, rb4, rb3)
        assert system.q_ind == ImmutableMatrix([q1, q3])
        assert system.u_ind == ImmutableMatrix([u1, u3])
        assert system.kdes == ImmutableMatrix(
            [u1 - q1.diff(t), u3 - q3.diff(t)])
        system.add_kdes(-(u2 - q2.diff(t)))
        system.add_joints(J2)
        assert system.joints == (J1, J3, J2)
        assert system.bodies == (rb1, rb2, rb4, rb3)
        assert system.q_ind == ImmutableMatrix([q1, q3, q2])
        assert system.u_ind == ImmutableMatrix([u1, u3, u2])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t), u3 - q3.diff(t),
                                               -(u2 - q2.diff(t))])
        system.add_joints(J_lag)
        assert system.joints == (J1, J3, J2, J_lag)
        assert system.bodies == (rb1, rb2, rb4, rb3, rb5)
        assert system.q_ind == ImmutableMatrix([q1, q3, q2, q4])
        assert system.u_ind == ImmutableMatrix([u1, u3, u2, q4.diff(t)])
        assert system.kdes == ImmutableMatrix([u1 - q1.diff(t), u3 - q3.diff(t),
                                               -(u2 - q2.diff(t))])
        assert system.q_dep[:] == []
        assert system.u_dep[:] == []
        pytest.raises(ValueError, lambda: system.add_joints(J2))
        pytest.raises(TypeError, lambda: system.add_joints(rb1))

    def test_joints_setter(self, _filled_system_setup):
        self.system.joints = self.joints[1:]
        assert self.system.joints == self.joints[1:]
        self._filled_system_check(exclude=('joints',))
        self.system.q_ind = ()
        self.system.u_ind = ()
        self.system.joints = self.joints
        self._filled_system_check()

    @pytest.mark.parametrize('name, joint_index', [
        ('J1', 0),
        ('J2', 1),
        ('not_existing', None),
    ])
    def test_get_joint(self, _filled_system_setup, name, joint_index):
        joint = self.system.get_joint(name)
        if joint_index is None:
            assert joint is None
        else:
            assert joint == self.joints[joint_index]

    @pytest.mark.parametrize('name, body_index', [
        ('rb1', 0),
        ('rb3', 2),
        ('not_existing', None),
    ])
    def test_get_body(self, _filled_system_setup, name, body_index):
        body = self.system.get_body(name)
        if body_index is None:
            assert body is None
        else:
            assert body == self.bodies[body_index]

    @pytest.mark.parametrize('eom_method', [KanesMethod, LagrangesMethod])
    def test_form_eoms_calls_subclass(self, _moving_point_mass, eom_method):
        class MyMethod(eom_method):
            pass

        self.system.form_eoms(eom_method=MyMethod)
        assert isinstance(self.system.eom_method, MyMethod)

    @pytest.mark.parametrize('kwargs, expected', [
        ({}, ImmutableMatrix([[-1, 0], [0, symbols('m')]])),
        ({'explicit_kinematics': True}, ImmutableMatrix([[1, 0],
                                                         [0, symbols('m')]])),
    ])
    def test_system_kane_form_eoms_kwargs(self, _moving_point_mass, kwargs,
                                          expected):
        self.system.form_eoms(**kwargs)
        assert self.system.mass_matrix_full == expected

    @pytest.mark.parametrize('kwargs, mm, gm', [
        ({}, ImmutableMatrix([[1, 0], [0, symbols('m')]]),
         ImmutableMatrix([q[0].diff(t), 0])),
    ])
    def test_system_lagrange_form_eoms_kwargs(self, _moving_point_mass, kwargs,
                                              mm, gm):
        self.system.form_eoms(eom_method=LagrangesMethod, **kwargs)
        assert self.system.mass_matrix_full == mm
        assert self.system.forcing_full == gm

    @pytest.mark.parametrize('eom_method, kwargs, error', [
        (KanesMethod, {'non_existing_kwarg': 1}, TypeError),
        (LagrangesMethod, {'non_existing_kwarg': 1}, TypeError),
        (KanesMethod, {'bodies': []}, ValueError),
        (KanesMethod, {'kd_eqs': []}, ValueError),
        (LagrangesMethod, {'bodies': []}, ValueError),
        (LagrangesMethod, {'Lagrangian': 1}, ValueError),
    ])
    def test_form_eoms_kwargs_errors(self, _empty_system_setup, eom_method,
                                     kwargs, error):
        self.system.q_ind = q[0]
        p = Particle('p', mass=symbols('m'))
        self.system.add_bodies(p)
        p.masscenter.set_pos(self.system.fixed_point, q[0] * self.system.x)
        with pytest.raises(error):
            self.system.form_eoms(eom_method=eom_method, **kwargs)


class TestValidateSystem(TestSystemBase):
    @pytest.mark.parametrize('valid_method, invalid_method, with_speeds', [
        (KanesMethod, LagrangesMethod, True),
        (LagrangesMethod, KanesMethod, False)
    ])
    def test_only_valid(self, valid_method, invalid_method, with_speeds):
        self._create_filled_system(with_speeds=with_speeds)
        self.system.validate_system(valid_method)
        # Test Lagrange should fail due to the usage of generalized speeds
        with pytest.raises(ValueError):
            self.system.validate_system(invalid_method)

    @pytest.mark.parametrize('method, with_speeds', [
        (KanesMethod, True), (LagrangesMethod, False)])
    def test_missing_joint_coordinate(self, method, with_speeds):
        self._create_filled_system(with_speeds=with_speeds)
        self.system.q_ind = self.q_ind[1:]
        self.system.u_ind = self.u_ind[:-1]
        self.system.kdes = self.kdes[:-1]
        pytest.raises(ValueError, lambda: self.system.validate_system(method))

    def test_missing_joint_speed(self, _filled_system_setup):
        self.system.q_ind = self.q_ind[:-1]
        self.system.u_ind = self.u_ind[1:]
        self.system.kdes = self.kdes[:-1]
        pytest.raises(ValueError, lambda: self.system.validate_system())

    def test_missing_joint_kdes(self, _filled_system_setup):
        self.system.kdes = self.kdes[1:]
        pytest.raises(ValueError, lambda: self.system.validate_system())

    def test_negative_joint_kdes(self, _filled_system_setup):
        self.system.kdes = [-self.kdes[0]] + self.kdes[1:]
        self.system.validate_system()

    @pytest.mark.parametrize('method, with_speeds', [
        (KanesMethod, True), (LagrangesMethod, False)])
    def test_missing_holonomic_constraint(self, method, with_speeds):
        self._create_filled_system(with_speeds=with_speeds)
        self.system.holonomic_constraints = []
        self.system.nonholonomic_constraints = self.nhc + [
            self.u_ind[1] - self.u_dep[0] + self.u_ind[0]]
        pytest.raises(ValueError, lambda: self.system.validate_system(method))
        self.system.q_dep = []
        self.system.q_ind = self.q_ind + self.q_dep
        self.system.validate_system(method)

    def test_missing_nonholonomic_constraint(self, _filled_system_setup):
        self.system.nonholonomic_constraints = []
        pytest.raises(ValueError, lambda: self.system.validate_system())
        self.system.u_dep = self.u_dep[1]
        self.system.u_ind = self.u_ind + [self.u_dep[0]]
        self.system.validate_system()

    def test_number_of_coordinates_speeds(self, _filled_system_setup):
        # Test more speeds than coordinates
        self.system.u_ind = self.u_ind + [u[5]]
        self.system.kdes = self.kdes + [u[5] - qd[5]]
        self.system.validate_system()
        # Test more coordinates than speeds
        self.system.q_ind = self.q_ind
        self.system.u_ind = self.u_ind[:-1]
        self.system.kdes = self.kdes[:-1]
        pytest.raises(ValueError, lambda: self.system.validate_system())

    def test_number_of_kdes(self, _filled_system_setup):
        # Test wrong number of kdes
        self.system.kdes = self.kdes[:-1]
        pytest.raises(ValueError, lambda: self.system.validate_system())
        self.system.kdes = self.kdes + [u[2] + u[1] - qd[2]]
        pytest.raises(ValueError, lambda: self.system.validate_system())

    def test_duplicates(self, _filled_system_setup):
        # This is basically a redundant feature, which should never fail
        self.system.validate_system(check_duplicates=True)

    def test_speeds_in_lagrange(self, _filled_system_setup_no_speeds):
        self.system.u_ind = u[:len(self.u_ind)]
        with pytest.raises(ValueError):
            self.system.validate_system(LagrangesMethod)
        self.system.u_ind = []
        self.system.validate_system(LagrangesMethod)
        self.system.u_aux = ua
        with pytest.raises(ValueError):
            self.system.validate_system(LagrangesMethod)
        self.system.u_aux = []
        self.system.validate_system(LagrangesMethod)
        self.system.add_joints(
            PinJoint('Ju', RigidBody('rbu1'), RigidBody('rbu2')))
        self.system.u_ind = []
        with pytest.raises(ValueError):
            self.system.validate_system(LagrangesMethod)


class TestSystemExamples:
    def test_cart_pendulum_kanes(self):
        # This example is the same as in the top documentation of System
        # Added a spring to the cart
        g, l, mc, mp, k = symbols('g l mc mp k')
        F, qp, qc, up, uc = dynamicsymbols('F qp qc up uc')
        rail = RigidBody('rail')
        cart = RigidBody('cart', mass=mc)
        bob = Particle('bob', mass=mp)
        bob_frame = ReferenceFrame('bob_frame')
        system = System.from_newtonian(rail)
        assert system.bodies == (rail,)
        assert system.frame == rail.frame
        assert system.fixed_point == rail.masscenter
        slider = PrismaticJoint('slider', rail, cart, qc, uc, joint_axis=rail.x)
        pin = PinJoint('pin', cart, bob, qp, up, joint_axis=cart.z,
                       child_interframe=bob_frame, child_point=l * bob_frame.y)
        system.add_joints(slider, pin)
        assert system.joints == (slider, pin)
        assert system.get_joint('slider') == slider
        assert system.get_body('bob') == bob
        system.apply_uniform_gravity(-g * system.y)
        system.add_loads((cart.masscenter, F * rail.x))
        system.add_actuators(TorqueActuator(k * qp, cart.z, bob_frame, cart))
        system.validate_system()
        system.form_eoms()
        assert isinstance(system.eom_method, KanesMethod)
        assert (simplify(system.mass_matrix - ImmutableMatrix(
            [[mp + mc, mp * l * cos(qp)], [mp * l * cos(qp), mp * l ** 2]]))
                == zeros(2, 2))
        assert (simplify(system.forcing - ImmutableMatrix([
            [mp * l * up ** 2 * sin(qp) + F],
            [-mp * g * l * sin(qp) + k * qp]])) == zeros(2, 1))

        system.add_holonomic_constraints(
            sympify(bob.masscenter.pos_from(rail.masscenter).dot(system.x)))
        assert system.eom_method is None
        system.q_ind, system.q_dep = qp, qc
        system.u_ind, system.u_dep = up, uc
        system.validate_system()

        # Computed solution based on manually solving the constraints
        subs = {qc: -l * sin(qp),
                uc: -l * cos(qp) * up,
                uc.diff(t): l * (up ** 2 * sin(qp) - up.diff(t) * cos(qp))}
        upd_expected = (
            (-g * mp * sin(qp) + k * qp / l + l * mc * sin(2 * qp) * up ** 2 / 2
             - l * mp * sin(2 * qp) * up ** 2 / 2 - F * cos(qp)) /
            (l * (mc * cos(qp) ** 2 + mp * sin(qp) ** 2)))
        upd_sol = tuple(solve(system.form_eoms().xreplace(subs),
                              up.diff(t)).values())[0]
        assert simplify(upd_sol - upd_expected) == 0
        assert isinstance(system.eom_method, KanesMethod)

        # Test other output
        Mk = -ImmutableMatrix([[0, 1], [1, 0]])
        gk = -ImmutableMatrix([uc, up])
        Md = ImmutableMatrix([[-l ** 2 * mp * cos(qp) ** 2 + l ** 2 * mp,
                               l * mp * cos(qp) - l * (mc + mp) * cos(qp)],
                              [l * cos(qp), 1]])
        gd = ImmutableMatrix(
            [[-g * l * mp * sin(qp) + k * qp - l ** 2 * mp * up ** 2 * sin(qp) *
              cos(qp) - l * F * cos(qp)], [l * up ** 2 * sin(qp)]])
        Mm = (Mk.row_join(zeros(2, 2))).col_join(zeros(2, 2).row_join(Md))
        gm = gk.col_join(gd)
        assert simplify(system.mass_matrix - Md) == zeros(2, 2)
        assert simplify(system.forcing - gd) == zeros(2, 1)
        assert simplify(system.mass_matrix_full - Mm) == zeros(4, 4)
        assert simplify(system.forcing_full - gm) == zeros(4, 1)

    def test_cart_pendulum_lagrange(self):
        # Lagrange version of test_cart_pendulus_kanes
        # Added a spring to the cart
        g, l, mc, mp, k = symbols('g l mc mp k')
        F, qp, qc = dynamicsymbols('F qp qc')
        qpd, qcd = dynamicsymbols('qp qc', 1)
        rail = RigidBody('rail')
        cart = RigidBody('cart', mass=mc)
        bob = Particle('bob', mass=mp)
        bob_frame = ReferenceFrame('bob_frame')
        system = System.from_newtonian(rail)
        assert system.bodies == (rail,)
        assert system.frame == rail.frame
        assert system.fixed_point == rail.masscenter
        slider = PrismaticJoint('slider', rail, cart, qc, qcd,
                                joint_axis=rail.x)
        pin = PinJoint('pin', cart, bob, qp, qpd, joint_axis=cart.z,
                       child_interframe=bob_frame, child_point=l * bob_frame.y)
        system.add_joints(slider, pin)
        assert system.joints == (slider, pin)
        assert system.get_joint('slider') == slider
        assert system.get_body('bob') == bob
        for body in system.bodies:
            body.potential_energy = body.mass * g * body.masscenter.pos_from(
                system.fixed_point).dot(system.y)
        system.add_loads((cart.masscenter, F * rail.x))
        system.add_actuators(TorqueActuator(k * qp, cart.z, bob_frame, cart))
        system.validate_system(LagrangesMethod)
        system.form_eoms(LagrangesMethod)
        assert (simplify(system.mass_matrix - ImmutableMatrix(
            [[mp + mc, mp * l * cos(qp)], [mp * l * cos(qp), mp * l ** 2]]))
                == zeros(2, 2))
        assert (simplify(system.forcing - ImmutableMatrix([
            [mp * l * qpd ** 2 * sin(qp) + F], [-mp * g * l * sin(qp) + k * qp]]
        )) == zeros(2, 1))

        system.add_holonomic_constraints(
            sympify(bob.masscenter.pos_from(rail.masscenter).dot(system.x)))
        assert system.eom_method is None
        system.q_ind, system.q_dep = qp, qc

        # Computed solution based on manually solving the constraints
        subs = {qc: -l * sin(qp),
                qcd: -l * cos(qp) * qpd,
                qcd.diff(t): l * (qpd ** 2 * sin(qp) - qpd.diff(t) * cos(qp))}
        qpdd_expected = (
            (-g * mp * sin(qp) + k * qp / l + l * mc * sin(2 * qp) * qpd ** 2 /
             2 - l * mp * sin(2 * qp) * qpd ** 2 / 2 - F * cos(qp)) /
            (l * (mc * cos(qp) ** 2 + mp * sin(qp) ** 2)))
        eoms = system.form_eoms(LagrangesMethod)
        lam1 = system.eom_method.lam_vec[0]
        lam1_sol = system.eom_method.solve_multipliers()[lam1]
        qpdd_sol = solve(eoms[0].xreplace({lam1: lam1_sol}).xreplace(subs),
                         qpd.diff(t))[0]
        assert simplify(qpdd_sol - qpdd_expected) == 0
        assert isinstance(system.eom_method, LagrangesMethod)

        # Test other output
        Md = ImmutableMatrix([[l ** 2 * mp, l * mp * cos(qp), -l * cos(qp)],
                              [l * mp * cos(qp), mc + mp, -1]])
        gd = ImmutableMatrix(
            [[-g * l * mp * sin(qp) + k * qp],
             [l * mp * sin(qp) * qpd ** 2 + F]])
        Mm = (eye(2).row_join(zeros(2, 3))).col_join(zeros(3, 2).row_join(
            Md.col_join(ImmutableMatrix([l * cos(qp), 1, 0]).T)))
        gm = ImmutableMatrix([qpd, qcd] + gd[:] + [l * sin(qp) * qpd ** 2])
        assert simplify(system.mass_matrix - Md) == zeros(2, 3)
        assert simplify(system.forcing - gd) == zeros(2, 1)
        assert simplify(system.mass_matrix_full - Mm) == zeros(5, 5)
        assert simplify(system.forcing_full - gm) == zeros(5, 1)

    def test_box_on_ground(self):
        # Particle sliding on ground with friction. The applied force is assumed
        # to be positive and to be higher than the friction force.
        g, m, mu = symbols('g m mu')
        q, u, ua = dynamicsymbols('q u ua')
        N, F = dynamicsymbols('N F', positive=True)
        P = Particle("P", mass=m)
        system = System()
        system.add_bodies(P)
        P.masscenter.set_pos(system.fixed_point, q * system.x)
        P.masscenter.set_vel(system.frame, u * system.x + ua * system.y)
        system.q_ind, system.u_ind, system.u_aux = [q], [u], [ua]
        system.kdes = [q.diff(t) - u]
        system.apply_uniform_gravity(-g * system.y)
        system.add_loads(
            Force(P, N * system.y),
            Force(P, F * system.x - mu * N * system.x))
        system.validate_system()
        system.form_eoms()

        # Test other output
        Mk = ImmutableMatrix([1])
        gk = ImmutableMatrix([u])
        Md = ImmutableMatrix([m])
        gd = ImmutableMatrix([F - mu * N])
        Mm = (Mk.row_join(zeros(1, 1))).col_join(zeros(1, 1).row_join(Md))
        gm = gk.col_join(gd)
        aux_eqs = ImmutableMatrix([N - m * g])
        assert simplify(system.mass_matrix - Md) == zeros(1, 1)
        assert simplify(system.forcing - gd) == zeros(1, 1)
        assert simplify(system.mass_matrix_full - Mm) == zeros(2, 2)
        assert simplify(system.forcing_full - gm) == zeros(2, 1)
        assert simplify(system.eom_method.auxiliary_eqs - aux_eqs
                        ) == zeros(1, 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\benchmarks\bench_groebnertools.py ===
"""Benchmark of the Groebner bases algorithms. """


from sympy.polys.rings import ring
from sympy.polys.domains import QQ
from sympy.polys.groebnertools import groebner

R, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = ring("x1:13", QQ)

V = R.gens
E = [(x1, x2), (x2, x3), (x1, x4), (x1, x6), (x1, x12), (x2, x5), (x2, x7), (x3, x8),
     (x3, x10), (x4, x11), (x4, x9), (x5, x6), (x6, x7), (x7, x8), (x8, x9), (x9, x10),
     (x10, x11), (x11, x12), (x5, x12), (x5, x9), (x6, x10), (x7, x11), (x8, x12)]

F3 = [ x**3 - 1 for x in V ]
Fg = [ x**2 + x*y + y**2 for x, y in E ]

F_1 = F3 + Fg
F_2 = F3 + Fg + [x3**2 + x3*x4 + x4**2]

def time_vertex_color_12_vertices_23_edges():
    assert groebner(F_1, R) != [1]

def time_vertex_color_12_vertices_24_edges():
    assert groebner(F_2, R) == [1]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_ring_series.py ===
from sympy.polys.domains import ZZ, QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.puiseux import puiseux_ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
    rs_trunc, rs_mul, rs_square, rs_pow, _has_constant_term, rs_hadamard_exp,
    rs_series_from_list, rs_exp, rs_log, rs_newton, rs_series_inversion,
    rs_compose_add, rs_asin, _atan, rs_atan, _atanh, rs_atanh, rs_asinh, rs_tan,
    rs_cot, rs_sin, rs_cos, rs_cos_sin, rs_sinh, rs_cosh, rs_cosh_sinh, rs_tanh,
    _tan1, rs_fun, rs_nth_root, rs_LambertW, rs_series_reversion, rs_is_puiseux,
    rs_series)
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, sinh, cosh, atan, atanh,
    asinh, tanh, log, sqrt)
from sympy.core.numbers import Rational, pi
from sympy.core import expand, S

def is_close(a, b):
    tol = 10**(-10)
    assert abs(a - b) < tol


def test_ring_series1():
    R, x = ring('x', QQ)
    p = x**4 + 2*x**3 + 3*x + 4
    assert _invert_monoms(p) == 4*x**4 + 3*x**3 + 2*x + 1
    assert rs_hadamard_exp(p) == x**4/24 + x**3/3 + 3*x + 4
    R, x = ring('x', QQ)
    p = x**4 + 2*x**3 + 3*x + 4
    assert rs_integrate(p, x) == x**5/5 + x**4/2 + 3*x**2/2 + 4*x
    R, x, y = ring('x, y', QQ)
    p = x**2*y**2 + x + 1
    assert rs_integrate(p, x) == x**3*y**2/3 + x**2/2 + x
    assert rs_integrate(p, y) == x**2*y**3/3 + x*y + y


def test_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = (y + t*x)**4
    p1 = rs_trunc(p, x, 3)
    assert p1 == y**4 + 4*y**3*t*x + 6*y**2*t**2*x**2


def test_mul_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = 1 + t*x + t*y
    for i in range(2):
        p = rs_mul(p, p, t, 3)

    assert p == 6*x**2*t**2 + 12*x*y*t**2 + 6*y**2*t**2 + 4*x*t + 4*y*t + 1
    p = 1 + t*x + t*y + t**2*x*y
    p1 = rs_mul(p, p, t, 2)
    assert p1 == 1 + 2*t*x + 2*t*y
    R1, z = ring('z', QQ)
    raises(ValueError, lambda: rs_mul(p, z, x, 2))

    p1 = 2 + 2*x + 3*x**2
    p2 = 3 + x**2
    assert rs_mul(p1, p2, x, 4) == 2*x**3 + 11*x**2 + 6*x + 6


def test_square_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = (1 + t*x + t*y)*2
    p1 = rs_mul(p, p, x, 3)
    p2 = rs_square(p, x, 3)
    assert p1 == p2
    p = 1 + x + x**2 + x**3
    assert rs_square(p, x, 4) == 4*x**3 + 3*x**2 + 2*x + 1


def test_pow_trunc():
    R, x, y, z = ring('x, y, z', QQ)
    p0 = y + x*z
    p = p0**16
    for xx in (x, y, z):
        p1 = rs_trunc(p, xx, 8)
        p2 = rs_pow(p0, 16, xx, 8)
        assert p1 == p2

    p = 1 + x
    p1 = rs_pow(p, 3, x, 2)
    assert p1 == 1 + 3*x
    assert rs_pow(p, 0, x, 2) == 1
    assert rs_pow(p, -2, x, 2) == 1 - 2*x
    p = x + y
    assert rs_pow(p, 3, y, 3) == x**3 + 3*x**2*y + 3*x*y**2
    assert rs_pow(1 + x, Rational(2, 3), x, 4) == 4*x**3/81 - x**2/9 + x*Rational(2, 3) + 1


def test_has_constant_term():
    R, x, y, z = ring('x, y, z', QQ)
    p = y + x*z
    assert _has_constant_term(p, x)
    p = x + x**4
    assert not _has_constant_term(p, x)
    p = 1 + x + x**4
    assert _has_constant_term(p, x)
    p = x + y + x*z


def test_inversion():
    R, x = ring('x', QQ)
    p = 2 + x + 2*x**2
    n = 5
    p1 = rs_series_inversion(p, x, n)
    assert rs_trunc(p*p1, x, n) == 1
    R, x, y = ring('x, y', QQ)
    p = 2 + x + 2*x**2 + y*x + x**2*y
    p1 = rs_series_inversion(p, x, n)
    assert rs_trunc(p*p1, x, n) == 1

    R, x, y = ring('x, y', QQ)
    p = 1 + x + y
    raises(NotImplementedError, lambda: rs_series_inversion(p, x, 4))
    p = R.zero
    raises(ZeroDivisionError, lambda: rs_series_inversion(p, x, 3))

    R, x = ring('x', ZZ)
    p = 2 + x
    raises(ValueError, lambda: rs_series_inversion(p, x, 3))


def test_series_reversion():
    R, x, y = ring('x, y', QQ)

    p = rs_tan(x, x, 10)
    assert rs_series_reversion(p, x, 8, y) == rs_atan(y, y, 8)

    p = rs_sin(x, x, 10)
    assert rs_series_reversion(p, x, 8, y) == 5*y**7/112 + 3*y**5/40 + \
        y**3/6 + y


def test_series_from_list():
    R, x = ring('x', QQ)
    p = 1 + 2*x + x**2 + 3*x**3
    c = [1, 2, 0, 4, 4]
    r = rs_series_from_list(p, c, x, 5)
    pc = R.from_list(list(reversed(c)))
    r1 = rs_trunc(pc.compose(x, p), x, 5)
    assert r == r1
    R, x, y = ring('x, y', QQ)
    c = [1, 3, 5, 7]
    p1 = rs_series_from_list(x + y, c, x, 3, concur=0)
    p2 = rs_trunc((1 + 3*(x+y) + 5*(x+y)**2 + 7*(x+y)**3), x, 3)
    assert p1 == p2

    R, x = ring('x', QQ)
    h = 25
    p = rs_exp(x, x, h) - 1
    p1 = rs_series_from_list(p, c, x, h)
    p2 = 0
    for i, cx in enumerate(c):
        p2 += cx*rs_pow(p, i, x, h)
    assert p1 == p2


def test_log():
    R, x = ring('x', QQ)
    p = 1 + x
    assert rs_log(p, x, 4) == x - x**2/2 + x**3/3
    p = 1 + x +2*x**2/3
    p1 = rs_log(p, x, 9)
    assert p1 == -17*x**8/648 + 13*x**7/189 - 11*x**6/162 - x**5/45 + \
      7*x**4/36 - x**3/3 + x**2/6 + x
    p2 = rs_series_inversion(p, x, 9)
    p3 = rs_log(p2, x, 9)
    assert p3 == -p1

    R, x, y = ring('x, y', QQ)
    p = 1 + x + 2*y*x**2
    p1 = rs_log(p, x, 6)
    assert p1 == (4*x**5*y**2 - 2*x**5*y - 2*x**4*y**2 + x**5/5 + 2*x**4*y -
                  x**4/4 - 2*x**3*y + x**3/3 + 2*x**2*y - x**2/2 + x)

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_log(x + a, x, 5) == -EX(1/(4*a**4))*x**4 + EX(1/(3*a**3))*x**3 \
        - EX(1/(2*a**2))*x**2 + EX(1/a)*x + EX(log(a))
    assert rs_log(x + x**2*y + a, x, 4) == -EX(a**(-2))*x**3*y + \
        EX(1/(3*a**3))*x**3 + EX(1/a)*x**2*y - EX(1/(2*a**2))*x**2 + \
        EX(1/a)*x + EX(log(a))

    p = x + x**2 + 3
    assert rs_log(p, x, 10).compose(x, 5) == EX(log(3) + Rational(19281291595, 9920232))


def test_exp():
    R, x = ring('x', QQ)
    p = x + x**4
    for h in [10, 30]:
        q = rs_series_inversion(1 + p, x, h) - 1
        p1 = rs_exp(q, x, h)
        q1 = rs_log(p1, x, h)
        assert q1 == q
    p1 = rs_exp(p, x, 30)
    assert p1.coeff(x**29) == QQ(74274246775059676726972369, 353670479749588078181744640000)
    prec = 21
    p = rs_log(1 + x, x, prec)
    p1 = rs_exp(p, x, prec)
    assert p1 == x + 1

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[exp(a), a])
    assert rs_exp(x + a, x, 5) == exp(a)*x**4/24 + exp(a)*x**3/6 + \
        exp(a)*x**2/2 + exp(a)*x + exp(a)
    assert rs_exp(x + x**2*y + a, x, 5) == exp(a)*x**4*y**2/2 + \
            exp(a)*x**4*y/2 + exp(a)*x**4/24 + exp(a)*x**3*y + \
            exp(a)*x**3/6 + exp(a)*x**2*y + exp(a)*x**2/2 + exp(a)*x + exp(a)

    R, x, y = ring('x, y', EX)
    assert rs_exp(x + a, x, 5) ==  EX(exp(a)/24)*x**4 + EX(exp(a)/6)*x**3 + \
        EX(exp(a)/2)*x**2 + EX(exp(a))*x + EX(exp(a))
    assert rs_exp(x + x**2*y + a, x, 5) == EX(exp(a)/2)*x**4*y**2 + \
        EX(exp(a)/2)*x**4*y + EX(exp(a)/24)*x**4 + EX(exp(a))*x**3*y + \
        EX(exp(a)/6)*x**3 + EX(exp(a))*x**2*y + EX(exp(a)/2)*x**2 + \
        EX(exp(a))*x + EX(exp(a))


def test_newton():
    R, x = ring('x', QQ)
    p = x**2 - 2
    r = rs_newton(p, x, 4)
    assert r == 8*x**4 + 4*x**2 + 2


def test_compose_add():
    R, x = ring('x', QQ)
    p1 = x**3 - 1
    p2 = x**2 - 2
    assert rs_compose_add(p1, p2) == x**6 - 6*x**4 - 2*x**3 + 12*x**2 - 12*x - 7


def test_fun():
    R, x, y = ring('x, y', QQ)
    p = x*y + x**2*y**3 + x**5*y
    assert rs_fun(p, rs_tan, x, 10) == rs_tan(p, x, 10)
    assert rs_fun(p, _tan1, x, 10) == _tan1(p, x, 10)


def test_nth_root():
    R, x, y = puiseux_ring('x, y', QQ)
    assert rs_nth_root(1 + x**2*y, 4, x, 10) == -77*x**8*y**4/2048 + \
        7*x**6*y**3/128 - 3*x**4*y**2/32 + x**2*y/4 + 1
    assert rs_nth_root(1 + x*y + x**2*y**3, 3, x, 5) == -x**4*y**6/9 + \
        5*x**4*y**5/27 - 10*x**4*y**4/243 - 2*x**3*y**4/9 + 5*x**3*y**3/81 + \
        x**2*y**3/3 - x**2*y**2/9 + x*y/3 + 1
    assert rs_nth_root(8*x, 3, x, 3) == 2*x**QQ(1, 3)
    assert rs_nth_root(8*x + x**2 + x**3, 3, x, 3) == x**QQ(4,3)/12 + 2*x**QQ(1,3)
    r = rs_nth_root(8*x + x**2*y + x**3, 3, x, 4)
    assert r == -x**QQ(7,3)*y**2/288 + x**QQ(7,3)/12 + x**QQ(4,3)*y/12 + 2*x**QQ(1,3)

    # Constant term in series
    a = symbols('a')
    R, x, y = puiseux_ring('x, y', EX)
    assert rs_nth_root(x + EX(a), 3, x, 4) == EX(5/(81*a**QQ(8, 3)))*x**3 - \
        EX(1/(9*a**QQ(5, 3)))*x**2 + EX(1/(3*a**QQ(2, 3)))*x + EX(a**QQ(1, 3))
    assert rs_nth_root(x**QQ(2, 3) + x**2*y + 5, 2, x, 3) == -EX(sqrt(5)/100)*\
        x**QQ(8, 3)*y - EX(sqrt(5)/16000)*x**QQ(8, 3) + EX(sqrt(5)/10)*x**2*y + \
        EX(sqrt(5)/2000)*x**2 - EX(sqrt(5)/200)*x**QQ(4, 3) + \
        EX(sqrt(5)/10)*x**QQ(2, 3) + EX(sqrt(5))


def test_atan():
    R, x, y = ring('x, y', QQ)
    assert rs_atan(x, x, 9) == -x**7/7 + x**5/5 - x**3/3 + x
    assert rs_atan(x*y + x**2*y**3, x, 9) == 2*x**8*y**11 - x**8*y**9 + \
        2*x**7*y**9 - x**7*y**7/7 - x**6*y**9/3 + x**6*y**7 - x**5*y**7 + \
        x**5*y**5/5 - x**4*y**5 - x**3*y**3/3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_atan(x + a, x, 5) == -EX((a**3 - a)/(a**8 + 4*a**6 + 6*a**4 + \
        4*a**2 + 1))*x**4 + EX((3*a**2 - 1)/(3*a**6 + 9*a**4 + \
        9*a**2 + 3))*x**3 - EX(a/(a**4 + 2*a**2 + 1))*x**2 + \
        EX(1/(a**2 + 1))*x + EX(atan(a))
    assert rs_atan(x + x**2*y + a, x, 4) == -EX(2*a/(a**4 + 2*a**2 + 1)) \
        *x**3*y + EX((3*a**2 - 1)/(3*a**6 + 9*a**4 + 9*a**2 + 3))*x**3 + \
        EX(1/(a**2 + 1))*x**2*y - EX(a/(a**4 + 2*a**2 + 1))*x**2 + EX(1/(a**2 \
        + 1))*x + EX(atan(a))

    # Test for _atan faster for small and univariate series
    R, x = ring('x', QQ)
    p = x**2 + 2*x
    assert _atan(p, x, 5) == rs_atan(p, x, 5)

    R, x = ring('x', EX)
    p = x**2 + 2*x
    assert _atan(p, x, 9) == rs_atan(p, x, 9)


def test_asin():
    R, x, y = ring('x, y', QQ)
    assert rs_asin(x + x*y, x, 5) == x**3*y**3/6 + x**3*y**2/2 + x**3*y/2 + \
        x**3/6 + x*y + x
    assert rs_asin(x*y + x**2*y**3, x, 6) == x**5*y**7/2 + 3*x**5*y**5/40 + \
        x**4*y**5/2 + x**3*y**3/6 + x**2*y**3 + x*y


def test_tan():
    R, x, y = ring('x, y', QQ)
    assert rs_tan(x, x, 9) == x + x**3/3 + QQ(2,15)*x**5 + QQ(17,315)*x**7
    assert rs_tan(x*y + x**2*y**3, x, 9) == 4*x**8*y**11/3 + 17*x**8*y**9/45 + \
        4*x**7*y**9/3 + 17*x**7*y**7/315 + x**6*y**9/3 + 2*x**6*y**7/3 + \
        x**5*y**7 + 2*x**5*y**5/15 + x**4*y**5 + x**3*y**3/3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[tan(a), a])
    assert rs_tan(x + a, x, 5) == (tan(a)**5 + 5*tan(a)**3/3 +
        2*tan(a)/3)*x**4 + (tan(a)**4 + 4*tan(a)**2/3 + Rational(1, 3))*x**3 + \
        (tan(a)**3 + tan(a))*x**2 + (tan(a)**2 + 1)*x + tan(a)
    assert rs_tan(x + x**2*y + a, x, 4) == (2*tan(a)**3 + 2*tan(a))*x**3*y + \
        (tan(a)**4 + Rational(4, 3)*tan(a)**2 + Rational(1, 3))*x**3 + (tan(a)**2 + 1)*x**2*y + \
        (tan(a)**3 + tan(a))*x**2 + (tan(a)**2 + 1)*x + tan(a)

    R, x, y = ring('x, y', EX)
    assert rs_tan(x + a, x, 5) == EX(tan(a)**5 + 5*tan(a)**3/3 +
        2*tan(a)/3)*x**4 + EX(tan(a)**4 + 4*tan(a)**2/3 + EX(1)/3)*x**3 + \
        EX(tan(a)**3 + tan(a))*x**2 + EX(tan(a)**2 + 1)*x + EX(tan(a))
    assert rs_tan(x + x**2*y + a, x, 4) == EX(2*tan(a)**3 +
        2*tan(a))*x**3*y + EX(tan(a)**4 + 4*tan(a)**2/3 + EX(1)/3)*x**3 + \
        EX(tan(a)**2 + 1)*x**2*y + EX(tan(a)**3 + tan(a))*x**2 + \
        EX(tan(a)**2 + 1)*x + EX(tan(a))

    p = x + x**2 + 5
    assert rs_atan(p, x, 10).compose(x, 10) == EX(atan(5) + S(67701870330562640) / \
        668083460499)


def test_cot():
    R, x, y = puiseux_ring('x, y', QQ)
    assert rs_cot(x**6 + x**7, x, 8) == x**(-6) - x**(-5) + x**(-4) - \
        x**(-3) + x**(-2) - x**(-1) + 1 - x + x**2 - x**3 + x**4 - x**5 + \
        2*x**6/3 - 4*x**7/3
    assert rs_cot(x + x**2*y, x, 5) == -x**4*y**5 - x**4*y/15 + x**3*y**4 - \
        x**3/45 - x**2*y**3 - x**2*y/3 + x*y**2 - x/3 - y + x**(-1)


def test_sin():
    R, x, y = ring('x, y', QQ)
    assert rs_sin(x, x, 9) == x - x**3/6 + x**5/120 - x**7/5040
    assert rs_sin(x*y + x**2*y**3, x, 9) == x**8*y**11/12 - \
        x**8*y**9/720 + x**7*y**9/12 - x**7*y**7/5040 - x**6*y**9/6 + \
        x**6*y**7/24 - x**5*y**7/2 + x**5*y**5/120 - x**4*y**5/2 - \
        x**3*y**3/6 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[sin(a), cos(a), a])
    assert rs_sin(x + a, x, 5) == sin(a)*x**4/24 - cos(a)*x**3/6 - \
        sin(a)*x**2/2 + cos(a)*x + sin(a)
    assert rs_sin(x + x**2*y + a, x, 5) == -sin(a)*x**4*y**2/2 - \
        cos(a)*x**4*y/2 + sin(a)*x**4/24 - sin(a)*x**3*y - cos(a)*x**3/6 + \
        cos(a)*x**2*y - sin(a)*x**2/2 + cos(a)*x + sin(a)

    R, x, y = ring('x, y', EX)
    assert rs_sin(x + a, x, 5) == EX(sin(a)/24)*x**4 - EX(cos(a)/6)*x**3 - \
        EX(sin(a)/2)*x**2 + EX(cos(a))*x + EX(sin(a))
    assert rs_sin(x + x**2*y + a, x, 5) == -EX(sin(a)/2)*x**4*y**2 - \
        EX(cos(a)/2)*x**4*y + EX(sin(a)/24)*x**4 - EX(sin(a))*x**3*y - \
        EX(cos(a)/6)*x**3 + EX(cos(a))*x**2*y - EX(sin(a)/2)*x**2 + \
        EX(cos(a))*x + EX(sin(a))


def test_cos():
    R, x, y = ring('x, y', QQ)
    assert rs_cos(x, x, 9) == 1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320
    assert rs_cos(x*y + x**2*y**3, x, 9) == x**8*y**12/24 - \
        x**8*y**10/48 + x**8*y**8/40320 + x**7*y**10/6 - \
        x**7*y**8/120 + x**6*y**8/4 - x**6*y**6/720 + x**5*y**6/6 - \
        x**4*y**6/2 + x**4*y**4/24 - x**3*y**4 - x**2*y**2/2 + 1

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[sin(a), cos(a), a])
    assert rs_cos(x + a, x, 5) == cos(a)*x**4/24 + sin(a)*x**3/6 - \
        cos(a)*x**2/2 - sin(a)*x + cos(a)
    assert rs_cos(x + x**2*y + a, x, 5) == -cos(a)*x**4*y**2/2 + \
        sin(a)*x**4*y/2 + cos(a)*x**4/24 - cos(a)*x**3*y + sin(a)*x**3/6 - \
        sin(a)*x**2*y - cos(a)*x**2/2 - sin(a)*x + cos(a)

    R, x, y = ring('x, y', EX)
    assert rs_cos(x + a, x, 5) == EX(cos(a)/24)*x**4 + EX(sin(a)/6)*x**3 - \
        EX(cos(a)/2)*x**2 - EX(sin(a))*x + EX(cos(a))
    assert rs_cos(x + x**2*y + a, x, 5) == -EX(cos(a)/2)*x**4*y**2 + \
        EX(sin(a)/2)*x**4*y + EX(cos(a)/24)*x**4 - EX(cos(a))*x**3*y + \
        EX(sin(a)/6)*x**3 - EX(sin(a))*x**2*y - EX(cos(a)/2)*x**2 - \
        EX(sin(a))*x + EX(cos(a))


def test_cos_sin():
    R, x, y = ring('x, y', QQ)
    c, s = rs_cos_sin(x, x, 9)
    assert c == rs_cos(x, x, 9)
    assert s == rs_sin(x, x, 9)
    c, s = rs_cos_sin(x + x*y, x, 5)
    assert c == rs_cos(x + x*y, x, 5)
    assert s == rs_sin(x + x*y, x, 5)

    # constant term in series
    c, s = rs_cos_sin(1 + x + x**2, x, 5)
    assert c == rs_cos(1 + x + x**2, x, 5)
    assert s == rs_sin(1 + x + x**2, x, 5)

    a = symbols('a')
    R, x, y = ring('x, y', QQ[sin(a), cos(a), a])
    c, s = rs_cos_sin(x + a, x, 5)
    assert c == rs_cos(x + a, x, 5)
    assert s == rs_sin(x + a, x, 5)

    R, x, y = ring('x, y', EX)
    c, s = rs_cos_sin(x + a, x, 5)
    assert c == rs_cos(x + a, x, 5)
    assert s == rs_sin(x + a, x, 5)


def test_atanh():
    R, x, y = ring('x, y', QQ)
    assert rs_atanh(x, x, 9) == x + x**3/3 + x**5/5 + x**7/7
    assert rs_atanh(x*y + x**2*y**3, x, 9) == 2*x**8*y**11 + x**8*y**9 + \
        2*x**7*y**9 + x**7*y**7/7 + x**6*y**9/3 + x**6*y**7 + x**5*y**7 + \
        x**5*y**5/5 + x**4*y**5 + x**3*y**3/3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_atanh(x + a, x, 5) == EX((a**3 + a)/(a**8 - 4*a**6 + 6*a**4 - \
        4*a**2 + 1))*x**4 - EX((3*a**2 + 1)/(3*a**6 - 9*a**4 + \
        9*a**2 - 3))*x**3 + EX(a/(a**4 - 2*a**2 + 1))*x**2 - EX(1/(a**2 - \
        1))*x + EX(atanh(a))
    assert rs_atanh(x + x**2*y + a, x, 4) == EX(2*a/(a**4 - 2*a**2 + \
        1))*x**3*y - EX((3*a**2 + 1)/(3*a**6 - 9*a**4 + 9*a**2 - 3))*x**3 - \
        EX(1/(a**2 - 1))*x**2*y + EX(a/(a**4 - 2*a**2 + 1))*x**2 - \
        EX(1/(a**2 - 1))*x + EX(atanh(a))

    p = x + x**2 + 5
    assert rs_atanh(p, x, 10).compose(x, 10) == EX(Rational(-733442653682135, 5079158784) \
        + atanh(5))

    # Test for _atanh faster for small and univariate series
    R,x  = ring('x', QQ)
    p = x**2 + 2*x
    assert _atanh(p, x, 5) == rs_atanh(p, x, 5)

    R,x = ring('x', EX)
    p = x**2 + 2*x
    assert _atanh(p, x, 9) == rs_atanh(p, x, 9)


def test_asinh():
    R, x, y = ring('x, y', QQ)
    assert rs_asinh(x, x, 9) == -5/112*x**7 + 3/40*x**5 - 1/6*x**3 + x
    assert rs_asinh(x*y + x**2*y**3, x, 9) == 3/4*x**8*y**11 - 5/16*x**8*y**9 + \
        3/4*x**7*y**9 - 5/112*x**7*y**7 - 1/6*x**6*y**9 + 3/8*x**6*y**7 - 1/2*x \
        **5*y**7 + 3/40*x**5*y**5 - 1/2*x**4*y**5 - 1/6*x**3*y**3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_asinh(x + a, x, 3) == -EX(a/(2*a**2*sqrt(a**2 + 1) + 2*sqrt(a**2 + 1))) \
        *x**2 + EX(1/sqrt(a**2 + 1))*x + EX(asinh(a))
    assert rs_asinh(x + x**2*y + a, x, 3) == EX(1/sqrt(a**2 + 1))*x**2*y - EX(a/(2*a**2 \
        *sqrt(a**2 + 1) + 2*sqrt(a**2 + 1)))*x**2 + EX(1/sqrt(a**2 + 1))*x + EX(asinh(a))

    p = x + x ** 2 + 5
    assert rs_asinh(p, x, 10).compose(x, 10) == EX(asinh(5) + 4643789843094995*sqrt(26)/\
        205564141692)


def test_sinh():
    R, x, y = ring('x, y', QQ)
    assert rs_sinh(x, x, 9) == x + x**3/6 + x**5/120 + x**7/5040
    assert rs_sinh(x*y + x**2*y**3, x, 9) == x**8*y**11/12 + \
        x**8*y**9/720 + x**7*y**9/12 + x**7*y**7/5040 + x**6*y**9/6 + \
        x**6*y**7/24 + x**5*y**7/2 + x**5*y**5/120 + x**4*y**5/2 + \
        x**3*y**3/6 + x**2*y**3 + x*y

    # constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[sinh(a), cosh(a), a])
    assert rs_sinh(x + a, x, 5) == 1/24*x**4*(sinh(a)) + 1/6*x**3*(cosh(a)) + 1/\
        2*x**2*(sinh(a)) + x*(cosh(a)) + (sinh(a))
    assert rs_sinh(x + x**2*y + a, x, 5) == 1/2*(sinh(a))*x**4*y**2 + 1/2*(cosh(a))\
        *x**4*y + 1/24*(sinh(a))*x**4 + (sinh(a))*x**3*y + 1/6*(cosh(a))*x**3 + \
        (cosh(a))*x**2*y + 1/2*(sinh(a))*x**2 + (cosh(a))*x + (sinh(a))

    R, x, y = ring('x, y', EX)
    assert rs_sinh(x + a, x, 5) == EX(sinh(a)/24)*x**4 + EX(cosh(a)/6)*x**3 + \
        EX(sinh(a)/2)*x**2 + EX(cosh(a))*x + EX(sinh(a))
    assert rs_sinh(x + x**2*y + a, x, 5) == EX(sinh(a)/2)*x**4*y**2 + EX(cosh(a)/\
        2)*x**4*y + EX(sinh(a)/24)*x**4 + EX(sinh(a))*x**3*y + EX(cosh(a)/6)*x**3 \
        + EX(cosh(a))*x**2*y + EX(sinh(a)/2)*x**2 + EX(cosh(a))*x + EX(sinh(a))


def test_cosh():
    R, x, y = ring('x, y', QQ)
    assert rs_cosh(x, x, 9) == 1 + x**2/2 + x**4/24 + x**6/720 + x**8/40320
    assert rs_cosh(x*y + x**2*y**3, x, 9) == x**8*y**12/24 + \
        x**8*y**10/48 + x**8*y**8/40320 + x**7*y**10/6 + \
        x**7*y**8/120 + x**6*y**8/4 + x**6*y**6/720 + x**5*y**6/6 + \
        x**4*y**6/2 + x**4*y**4/24 + x**3*y**4 + x**2*y**2/2 + 1

    # constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[sinh(a), cosh(a), a])
    assert rs_cosh(x + a, x, 5) == 1/24*(cosh(a))*x**4 + 1/6*(sinh(a))*x**3 + \
        1/2*(cosh(a))*x**2 + (sinh(a))*x + (cosh(a))
    assert rs_cosh(x + x**2*y + a, x, 5) == 1/2*(cosh(a))*x**4*y**2 + 1/2*(sinh(a))\
        *x**4*y + 1/24*(cosh(a))*x**4 + (cosh(a))*x**3*y + 1/6*(sinh(a))*x**3 + \
        (sinh(a))*x**2*y + 1/2*(cosh(a))*x**2 + (sinh(a))*x + (cosh(a))
    R, x, y = ring('x, y', EX)
    assert rs_cosh(x + a, x, 5) == EX(cosh(a)/24)*x**4 + EX(sinh(a)/6)*x**3 + \
        EX(cosh(a)/2)*x**2 + EX(sinh(a))*x + EX(cosh(a))
    assert rs_cosh(x + x**2*y + a, x, 5) == EX(cosh(a)/2)*x**4*y**2 + EX(sinh(a)/\
        2)*x**4*y + EX(cosh(a)/24)*x**4 + EX(cosh(a))*x**3*y + EX(sinh(a)/6)*x**3 \
        + EX(sinh(a))*x**2*y + EX(cosh(a)/2)*x**2 + EX(sinh(a))*x + EX(cosh(a))


def test_cosh_sinh():
    R, x, y = ring('x, y', QQ)
    ch, sh = rs_cosh_sinh(x, x, 9)
    assert ch == rs_cosh(x, x, 9)
    assert sh == rs_sinh(x, x, 9)
    ch, sh = rs_cosh_sinh(x + x*y, x, 5)
    assert ch == rs_cosh(x + x*y, x, 5)
    assert sh == rs_sinh(x + x*y, x, 5)

    # constant term in series
    c, s = rs_cosh_sinh(1 + x + x**2, x, 5)
    assert c == rs_cosh(1 + x + x**2, x, 5)
    assert s == rs_sinh(1 + x + x**2, x, 5)

    a = symbols('a')
    R, x, y = ring('x, y', QQ[sinh(a), cosh(a), a])
    ch, sh = rs_cosh_sinh(x + a, x, 5)
    assert ch == rs_cosh(x + a, x, 5)
    assert sh == rs_sinh(x + a, x, 5)
    R, x, y = ring('x, y', EX)
    ch, sh = rs_cosh_sinh(x + a, x, 5)
    assert ch == rs_cosh(x + a, x, 5)
    assert sh == rs_sinh(x + a, x, 5)


def test_tanh():
    R, x, y = ring('x, y', QQ)
    assert rs_tanh(x, x, 9) == x - QQ(1,3)*x**3 + QQ(2,15)*x**5 - QQ(17,315)*x**7
    assert rs_tanh(x*y + x**2*y**3, x, 9) == 4*x**8*y**11/3 - \
        17*x**8*y**9/45 + 4*x**7*y**9/3 - 17*x**7*y**7/315 - x**6*y**9/3 + \
        2*x**6*y**7/3 - x**5*y**7 + 2*x**5*y**5/15 - x**4*y**5 - \
        x**3*y**3/3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_tanh(x + a, x, 5) == EX(tanh(a)**5 - 5*tanh(a)**3/3 +
        2*tanh(a)/3)*x**4 + EX(-tanh(a)**4 + 4*tanh(a)**2/3 - QQ(1, 3))*x**3 + \
        EX(tanh(a)**3 - tanh(a))*x**2 + EX(-tanh(a)**2 + 1)*x + EX(tanh(a))

    p = rs_tanh(x + x**2*y + a, x, 4)
    assert (p.compose(x, 10)).compose(y, 5) == EX(-1000*tanh(a)**4 + \
        10100*tanh(a)**3 + 2470*tanh(a)**2/3 - 10099*tanh(a) + QQ(530, 3))


def test_RR():
    rs_funcs = [rs_sin, rs_cos, rs_tan, rs_cot, rs_atan, rs_tanh]
    sympy_funcs = [sin, cos, tan, cot, atan, tanh]
    R, x, y = ring('x, y', RR)
    a = symbols('a')
    for rs_func, sympy_func in zip(rs_funcs, sympy_funcs):
        p = rs_func(2 + x, x, 5).compose(x, 5)
        q = sympy_func(2 + a).series(a, 0, 5).removeO()
        is_close(p.as_expr(), q.subs(a, 5).n())

    p = rs_nth_root(2 + x, 5, x, 5).compose(x, 5)
    q = ((2 + a)**QQ(1, 5)).series(a, 0, 5).removeO()
    is_close(p.as_expr(), q.subs(a, 5).n())


def test_is_regular():
    R, x, y = puiseux_ring('x, y', QQ)
    p = 1 + 2*x + x**2 + 3*x**3
    assert not rs_is_puiseux(p, x)

    p = x + x**QQ(1,5)*y
    assert rs_is_puiseux(p, x)
    assert not rs_is_puiseux(p, y)

    p = x + x**2*y**QQ(1,5)*y
    assert not rs_is_puiseux(p, x)


def test_puiseux():
    R, x, y = puiseux_ring('x, y', QQ)
    p = x**QQ(2,5) + x**QQ(2,3) + x

    r = rs_series_inversion(p, x, 1)
    r1 = -x**QQ(14,15) + x**QQ(4,5) - 3*x**QQ(11,15) + x**QQ(2,3) + \
        2*x**QQ(7,15) - x**QQ(2,5) - x**QQ(1,5) + x**QQ(2,15) - x**QQ(-2,15) \
        + x**QQ(-2,5)
    assert r == r1

    r = rs_nth_root(1 + p, 3, x, 1)
    assert r == -x**QQ(4,5)/9 + x**QQ(2,3)/3 + x**QQ(2,5)/3 + 1

    r = rs_log(1 + p, x, 1)
    assert r == -x**QQ(4,5)/2 + x**QQ(2,3) + x**QQ(2,5)

    r = rs_LambertW(p, x, 1)
    assert r == -x**QQ(4,5) + x**QQ(2,3) + x**QQ(2,5)

    p1 = x + x**QQ(1,5)*y
    r = rs_exp(p1, x, 1)
    assert r == x**QQ(4,5)*y**4/24 + x**QQ(3,5)*y**3/6 + x**QQ(2,5)*y**2/2 + \
        x**QQ(1,5)*y + 1

    r = rs_atan(p, x, 2)
    assert r ==  -x**QQ(9,5) - x**QQ(26,15) - x**QQ(22,15) - x**QQ(6,5)/3 + \
        x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_atan(p1, x, 2)
    assert r ==  x**QQ(9,5)*y**9/9 + x**QQ(9,5)*y**4 - x**QQ(7,5)*y**7/7 - \
        x**QQ(7,5)*y**2 + x*y**5/5 + x - x**QQ(3,5)*y**3/3 + x**QQ(1,5)*y

    r = rs_tan(p, x, 2)
    assert r == x**QQ(2,5) + x**QQ(2,3) + x + QQ(1,3)*x**QQ(6,5) + x**QQ(22,15)\
        + x**QQ(26,15) + x**QQ(9,5)

    r = rs_sin(p, x, 2)
    assert r == x**QQ(2,5) + x**QQ(2,3) + x - QQ(1,6)*x**QQ(6,5) - QQ(1,2)*x**\
        QQ(22,15) - QQ(1,2)*x**QQ(26,15) - QQ(1,2)*x**QQ(9,5)

    r = rs_cos(p, x, 2)
    assert r == 1 - QQ(1,2)*x**QQ(4,5) - x**QQ(16,15) - QQ(1,2)*x**QQ(4,3) - \
        x**QQ(7,5) + QQ(1,24)*x**QQ(8,5) - x**QQ(5,3) + QQ(1,6)*x**QQ(28,15)

    r = rs_asin(p, x, 2)
    assert r == x**QQ(9,5)/2 + x**QQ(26,15)/2 + x**QQ(22,15)/2 + \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_cot(p, x, 1)
    assert r == -x**QQ(14,15) + x**QQ(4,5) - 3*x**QQ(11,15) + \
        2*x**QQ(2,3)/3 + 2*x**QQ(7,15) - 4*x**QQ(2,5)/3 - x**QQ(1,5) + \
        x**QQ(2,15) - x**QQ(-2,15) + x**QQ(-2,5)

    r = rs_cos_sin(p, x, 2)
    assert r[0] == x**QQ(28,15)/6 - x**QQ(5,3) + x**QQ(8,5)/24 - x**QQ(7,5) - \
        x**QQ(4,3)/2 - x**QQ(16,15) - x**QQ(4,5)/2 + 1
    assert r[1] == -x**QQ(9,5)/2 - x**QQ(26,15)/2 - x**QQ(22,15)/2 - \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_atanh(p, x, 2)
    assert r == x**QQ(9,5) + x**QQ(26,15) + x**QQ(22,15) + x**QQ(6,5)/3 + x + \
        x**QQ(2,3) + x**QQ(2,5)

    r = rs_asinh(p, x, 2)
    assert r == x**QQ(2,5) + x**QQ(2,3) + x - QQ(1,6)*x**QQ(6,5) - QQ(1,2)*x**\
        QQ(22,15) - QQ(1,2)*x**QQ(26,15) - QQ(1,2)*x**QQ(9,5)

    r = rs_cosh(p, x, 2)
    assert r == x**QQ(28,15)/6 + x**QQ(5,3) + x**QQ(8,5)/24 + x**QQ(7,5) + \
        x**QQ(4,3)/2 + x**QQ(16,15) + x**QQ(4,5)/2 + 1

    r = rs_sinh(p, x, 2)
    assert r == x**QQ(9,5)/2 + x**QQ(26,15)/2 + x**QQ(22,15)/2 + \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_cosh_sinh(p, x, 2)
    assert r[0] == x**QQ(28,15)/6 + x**QQ(5,3) + x**QQ(8,5)/24 + x**QQ(7,5) + \
        x**QQ(4,3)/2 + x**QQ(16,15) + x**QQ(4,5)/2 + 1
    assert r[1] == x**QQ(9,5)/2 + x**QQ(26,15)/2 + x**QQ(22,15)/2 + \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_tanh(p, x, 2)
    assert r == -x**QQ(9,5) - x**QQ(26,15) - x**QQ(22,15) - x**QQ(6,5)/3 + \
        x + x**QQ(2,3) + x**QQ(2,5)


def test_puiseux_algebraic(): # https://github.com/sympy/sympy/issues/24395

    K = QQ.algebraic_field(sqrt(2))
    sqrt2 = K.from_sympy(sqrt(2))
    x, y = symbols('x, y')
    R, xr, yr = puiseux_ring([x, y], K)
    p = (1+sqrt2)*xr**QQ(1,2) + (1-sqrt2)*yr**QQ(2,3)

    assert p.to_dict() == {(QQ(1,2),QQ(0)):1+sqrt2, (QQ(0),QQ(2,3)):1-sqrt2}
    assert p.as_expr() == (1 + sqrt(2))*x**(S(1)/2) + (1 - sqrt(2))*y**(S(2)/3)


def test1():
    R, x = puiseux_ring('x', QQ)
    r = rs_sin(x, x, 15)*x**(-5)
    assert r == x**8/6227020800 - x**6/39916800 + x**4/362880 - x**2/5040 + \
        QQ(1,120) - x**-2/6 + x**-4

    p = rs_sin(x, x, 10)
    r = rs_nth_root(p, 2, x, 10)
    assert  r == -67*x**QQ(17,2)/29030400 - x**QQ(13,2)/24192 + \
        x**QQ(9,2)/1440 - x**QQ(5,2)/12 + x**QQ(1,2)

    p = rs_sin(x, x, 10)
    r = rs_nth_root(p, 7, x, 10)
    r = rs_pow(r, 5, x, 10)
    assert r == -97*x**QQ(61,7)/124467840 - x**QQ(47,7)/16464 + \
        11*x**QQ(33,7)/3528 - 5*x**QQ(19,7)/42 + x**QQ(5,7)

    r = rs_exp(x**QQ(1,2), x, 10)
    assert r == x**QQ(19,2)/121645100408832000 + x**9/6402373705728000 + \
        x**QQ(17,2)/355687428096000 + x**8/20922789888000 + \
        x**QQ(15,2)/1307674368000 + x**7/87178291200 + \
        x**QQ(13,2)/6227020800 + x**6/479001600 + x**QQ(11,2)/39916800 + \
        x**5/3628800 + x**QQ(9,2)/362880 + x**4/40320 + x**QQ(7,2)/5040 + \
        x**3/720 + x**QQ(5,2)/120 + x**2/24 + x**QQ(3,2)/6 + x/2 + \
        x**QQ(1,2) + 1


def test_puiseux2():
    R, y = ring('y', QQ)
    S, x = puiseux_ring('x', R.to_domain())

    p = x + x**QQ(1,5)*y
    r = rs_atan(p, x, 3)
    assert r == (y**13/13 + y**8 + 2*y**3)*x**QQ(13,5) - (y**11/11 + y**6 +
        y)*x**QQ(11,5) + (y**9/9 + y**4)*x**QQ(9,5) - (y**7/7 +
        y**2)*x**QQ(7,5) + (y**5/5 + 1)*x - y**3*x**QQ(3,5)/3 + y*x**QQ(1,5)


@slow
def test_rs_series():
    x, a, b, c = symbols('x, a, b, c')

    assert rs_series(a, a, 5).as_expr() == a
    assert rs_series(sin(a), a, 5).as_expr() == (sin(a).series(a, 0,
        5)).removeO()
    assert rs_series(sin(a) + cos(a), a, 5).as_expr() == ((sin(a) +
        cos(a)).series(a, 0, 5)).removeO()
    assert rs_series(sin(a)*cos(a), a, 5).as_expr() == ((sin(a)*
        cos(a)).series(a, 0, 5)).removeO()

    p = (sin(a) - a)*(cos(a**2) + a**4/2)
    assert expand(rs_series(p, a, 10).as_expr()) == expand(p.series(a, 0,
        10).removeO())

    p = sin(a**2/2 + a/3) + cos(a/5)*sin(a/2)**3
    assert expand(rs_series(p, a, 5).as_expr()) == expand(p.series(a, 0,
        5).removeO())

    p = sin(x**2 + a)*(cos(x**3 - 1) - a - a**2)
    assert expand(rs_series(p, a, 5).as_expr()) == expand(p.series(a, 0,
        5).removeO())

    p = sin(a**2 - a/3 + 2)**5*exp(a**3 - a/2)
    assert expand(rs_series(p, a, 10).as_expr()) == expand(p.series(a, 0,
        10).removeO())

    p = sin(a + b + c)
    assert expand(rs_series(p, a, 5).as_expr()) == expand(p.series(a, 0,
        5).removeO())

    p = tan(sin(a**2 + 4) + b + c)
    assert expand(rs_series(p, a, 6).as_expr()) == expand(p.series(a, 0,
        6).removeO())

    p = a**QQ(2,5) + a**QQ(2,3) + a

    r = rs_series(tan(p), a, 2)
    assert r.as_expr() == a**QQ(9,5) + a**QQ(26,15) + a**QQ(22,15) + a**QQ(6,5)/3 + \
        a + a**QQ(2,3) + a**QQ(2,5)

    r = rs_series(exp(p), a, 1)
    assert r.as_expr() == a**QQ(4,5)/2 + a**QQ(2,3) + a**QQ(2,5) + 1

    r = rs_series(sin(p), a, 2)
    assert r.as_expr() == -a**QQ(9,5)/2 - a**QQ(26,15)/2 - a**QQ(22,15)/2 - \
        a**QQ(6,5)/6 + a + a**QQ(2,3) + a**QQ(2,5)

    r = rs_series(cos(p), a, 2)
    assert r.as_expr() == a**QQ(28,15)/6 - a**QQ(5,3) + a**QQ(8,5)/24 - a**QQ(7,5) - \
        a**QQ(4,3)/2 - a**QQ(16,15) - a**QQ(4,5)/2 + 1

    assert rs_series(sin(a)/7, a, 5).as_expr() == (sin(a)/7).series(a, 0,
            5).removeO()


def test_rs_series_ConstantInExpr():
    x, a = symbols('x a')
    assert rs_series(log(1 + x), x, 5).as_expr() == -x**4/4 + x**3/3 - \
            x**2/2 + x
    assert rs_series(log(1 + 4*x), x, 5).as_expr() == -64*x**4 + 64*x**3/3 - \
            8*x**2 + 4*x
    assert rs_series(log(1 + x + x**2), x, 10).as_expr() == -2*x**9/9 + \
            x**8/8 + x**7/7 - x**6/3 + x**5/5 + x**4/4 - 2*x**3/3 + x**2/2 + x
    assert rs_series(log(1 + x*a**2), x, 7).as_expr() == -x**6*a**12/6 + \
            x**5*a**10/5 - x**4*a**8/4 + x**3*a**6/3 - x**2*a**4/2 + x*a**2

    assert rs_series(atan(1 + x), x, 9).as_expr() == -x**7/112 + x**6/48 - x**5/40 \
            + x**3/12 - x**2/4 + x/2 + pi/4
    assert rs_series(atan(1 + x + x**2),x, 9).as_expr() == -15*x**7/112 - x**6/48 + \
            9*x**5/40 - 5*x**3/12 + x**2/4 + x/2 + pi/4
    assert rs_series(atan(1 + x * a), x, 9).as_expr() == -a**7*x**7/112 + a**6*x**6/48 \
            - a**5*x**5/40 + a**3*x**3/12 - a**2*x**2/4 + a*x/2 + pi/4

    assert rs_series(tanh(1 + x), x, 5).as_expr() == -5*x**4*tanh(1)**3/3 + x**4* \
            tanh(1)**5 + 2*x**4*tanh(1)/3 - x**3*tanh(1)**4 - x**3/3 + 4*x**3*tanh(1) \
           **2/3 - x**2*tanh(1) + x**2*tanh(1)**3 - x*tanh(1)**2 + x + tanh(1)
    assert rs_series(tanh(1 + x * a), x, 3).as_expr() == -a**2*x**2*tanh(1) + a**2*x** \
            2*tanh(1)**3 - a*x*tanh(1)**2 + a*x + tanh(1)

    assert rs_series(sinh(1 + x), x, 5).as_expr() == x**4*sinh(1)/24 + x**3*cosh(1)/6 + \
            x**2*sinh(1)/2 + x*cosh(1) + sinh(1)
    assert rs_series(sinh(1 + x * a), x, 5).as_expr() == a**4*x**4*sinh(1)/24 + \
            a**3*x**3*cosh(1)/6 + a**2*x**2*sinh(1)/2 + a*x*cosh(1) + sinh(1)

    assert rs_series(cosh(1 + x), x, 5).as_expr() == x**4*cosh(1)/24 + x**3*sinh(1)/6 + \
            x**2*cosh(1)/2 + x*sinh(1) + cosh(1)
    assert rs_series(cosh(1 + x * a), x, 5).as_expr() == a**4*x**4*cosh(1)/24 + \
            a**3*x**3*sinh(1)/6 + a**2*x**2*cosh(1)/2 + a*x*sinh(1) + cosh(1)


def test_issue():
    # https://github.com/sympy/sympy/issues/10191
    # https://github.com/sympy/sympy/issues/19543

    a, b = symbols('a b')
    assert rs_series(sin(a**QQ(3,7))*exp(a + b**QQ(6,7)), a,2).as_expr() == \
        a**QQ(10,7)*exp(b**QQ(6,7)) - a**QQ(9,7)*exp(b**QQ(6,7))/6 + a**QQ(3,7)*exp(b**QQ(6,7))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\branch\traverse.py ===
""" Branching Strategies to Traverse a Tree """
from itertools import product
from sympy.strategies.util import basic_fns
from .core import chain, identity, do_one


def top_down(brule, fns=basic_fns):
    """ Apply a rule down a tree running it on the top nodes first """
    return chain(do_one(brule, identity),
                 lambda expr: sall(top_down(brule, fns), fns)(expr))


def sall(brule, fns=basic_fns):
    """ Strategic all - apply rule to args """
    op, new, children, leaf = map(fns.get, ('op', 'new', 'children', 'leaf'))

    def all_rl(expr):
        if leaf(expr):
            yield expr
        else:
            myop = op(expr)
            argss = product(*map(brule, children(expr)))
            for args in argss:
                yield new(myop, *args)
    return all_rl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\period\test_asfreq.py ===
import pytest

from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime

from pandas import (
    Period,
    Timestamp,
    offsets,
)
import pandas._testing as tm

bday_msg = "Period with BDay freq is deprecated"


class TestFreqConversion:
    """Test frequency conversion of date objects"""

    @pytest.mark.filterwarnings("ignore:Period with BDay:FutureWarning")
    @pytest.mark.parametrize("freq", ["Y", "Q", "M", "W", "B", "D"])
    def test_asfreq_near_zero(self, freq):
        # GH#19643, GH#19650
        per = Period("0001-01-01", freq=freq)
        tup1 = (per.year, per.hour, per.day)

        prev = per - 1
        assert prev.ordinal == per.ordinal - 1
        tup2 = (prev.year, prev.month, prev.day)
        assert tup2 < tup1

    def test_asfreq_near_zero_weekly(self):
        # GH#19834
        per1 = Period("0001-01-01", "D") + 6
        per2 = Period("0001-01-01", "D") - 6
        week1 = per1.asfreq("W")
        week2 = per2.asfreq("W")
        assert week1 != week2
        assert week1.asfreq("D", "E") >= per1
        assert week2.asfreq("D", "S") <= per2

    def test_to_timestamp_out_of_bounds(self):
        # GH#19643, used to incorrectly give Timestamp in 1754
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            per = Period("0001-01-01", freq="B")
        msg = "Out of bounds nanosecond timestamp"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=bday_msg):
                per.to_timestamp()

    def test_asfreq_corner(self):
        val = Period(freq="Y", year=2007)
        result1 = val.asfreq("5min")
        result2 = val.asfreq("min")
        expected = Period("2007-12-31 23:59", freq="min")
        assert result1.ordinal == expected.ordinal
        assert result1.freqstr == "5min"
        assert result2.ordinal == expected.ordinal
        assert result2.freqstr == "min"

    def test_conv_annual(self):
        # frequency conversion tests: from Annual Frequency

        ival_A = Period(freq="Y", year=2007)

        ival_AJAN = Period(freq="Y-JAN", year=2007)
        ival_AJUN = Period(freq="Y-JUN", year=2007)
        ival_ANOV = Period(freq="Y-NOV", year=2007)

        ival_A_to_Q_start = Period(freq="Q", year=2007, quarter=1)
        ival_A_to_Q_end = Period(freq="Q", year=2007, quarter=4)
        ival_A_to_M_start = Period(freq="M", year=2007, month=1)
        ival_A_to_M_end = Period(freq="M", year=2007, month=12)
        ival_A_to_W_start = Period(freq="W", year=2007, month=1, day=1)
        ival_A_to_W_end = Period(freq="W", year=2007, month=12, day=31)
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_A_to_B_start = Period(freq="B", year=2007, month=1, day=1)
            ival_A_to_B_end = Period(freq="B", year=2007, month=12, day=31)
        ival_A_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_A_to_D_end = Period(freq="D", year=2007, month=12, day=31)
        ival_A_to_H_start = Period(freq="h", year=2007, month=1, day=1, hour=0)
        ival_A_to_H_end = Period(freq="h", year=2007, month=12, day=31, hour=23)
        ival_A_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_A_to_T_end = Period(
            freq="Min", year=2007, month=12, day=31, hour=23, minute=59
        )
        ival_A_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_A_to_S_end = Period(
            freq="s", year=2007, month=12, day=31, hour=23, minute=59, second=59
        )

        ival_AJAN_to_D_end = Period(freq="D", year=2007, month=1, day=31)
        ival_AJAN_to_D_start = Period(freq="D", year=2006, month=2, day=1)
        ival_AJUN_to_D_end = Period(freq="D", year=2007, month=6, day=30)
        ival_AJUN_to_D_start = Period(freq="D", year=2006, month=7, day=1)
        ival_ANOV_to_D_end = Period(freq="D", year=2007, month=11, day=30)
        ival_ANOV_to_D_start = Period(freq="D", year=2006, month=12, day=1)

        assert ival_A.asfreq("Q", "s") == ival_A_to_Q_start
        assert ival_A.asfreq("Q", "e") == ival_A_to_Q_end
        assert ival_A.asfreq("M", "s") == ival_A_to_M_start
        assert ival_A.asfreq("M", "E") == ival_A_to_M_end
        assert ival_A.asfreq("W", "s") == ival_A_to_W_start
        assert ival_A.asfreq("W", "E") == ival_A_to_W_end
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_A.asfreq("B", "s") == ival_A_to_B_start
            assert ival_A.asfreq("B", "E") == ival_A_to_B_end
        assert ival_A.asfreq("D", "s") == ival_A_to_D_start
        assert ival_A.asfreq("D", "E") == ival_A_to_D_end
        msg = "'H' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert ival_A.asfreq("H", "s") == ival_A_to_H_start
            assert ival_A.asfreq("H", "E") == ival_A_to_H_end
        assert ival_A.asfreq("min", "s") == ival_A_to_T_start
        assert ival_A.asfreq("min", "E") == ival_A_to_T_end
        msg = "'T' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert ival_A.asfreq("T", "s") == ival_A_to_T_start
            assert ival_A.asfreq("T", "E") == ival_A_to_T_end
        msg = "'S' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert ival_A.asfreq("S", "S") == ival_A_to_S_start
            assert ival_A.asfreq("S", "E") == ival_A_to_S_end

        assert ival_AJAN.asfreq("D", "s") == ival_AJAN_to_D_start
        assert ival_AJAN.asfreq("D", "E") == ival_AJAN_to_D_end

        assert ival_AJUN.asfreq("D", "s") == ival_AJUN_to_D_start
        assert ival_AJUN.asfreq("D", "E") == ival_AJUN_to_D_end

        assert ival_ANOV.asfreq("D", "s") == ival_ANOV_to_D_start
        assert ival_ANOV.asfreq("D", "E") == ival_ANOV_to_D_end

        assert ival_A.asfreq("Y") == ival_A

    def test_conv_quarterly(self):
        # frequency conversion tests: from Quarterly Frequency

        ival_Q = Period(freq="Q", year=2007, quarter=1)
        ival_Q_end_of_year = Period(freq="Q", year=2007, quarter=4)

        ival_QEJAN = Period(freq="Q-JAN", year=2007, quarter=1)
        ival_QEJUN = Period(freq="Q-JUN", year=2007, quarter=1)

        ival_Q_to_A = Period(freq="Y", year=2007)
        ival_Q_to_M_start = Period(freq="M", year=2007, month=1)
        ival_Q_to_M_end = Period(freq="M", year=2007, month=3)
        ival_Q_to_W_start = Period(freq="W", year=2007, month=1, day=1)
        ival_Q_to_W_end = Period(freq="W", year=2007, month=3, day=31)
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_Q_to_B_start = Period(freq="B", year=2007, month=1, day=1)
            ival_Q_to_B_end = Period(freq="B", year=2007, month=3, day=30)
        ival_Q_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_Q_to_D_end = Period(freq="D", year=2007, month=3, day=31)
        ival_Q_to_H_start = Period(freq="h", year=2007, month=1, day=1, hour=0)
        ival_Q_to_H_end = Period(freq="h", year=2007, month=3, day=31, hour=23)
        ival_Q_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_Q_to_T_end = Period(
            freq="Min", year=2007, month=3, day=31, hour=23, minute=59
        )
        ival_Q_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_Q_to_S_end = Period(
            freq="s", year=2007, month=3, day=31, hour=23, minute=59, second=59
        )

        ival_QEJAN_to_D_start = Period(freq="D", year=2006, month=2, day=1)
        ival_QEJAN_to_D_end = Period(freq="D", year=2006, month=4, day=30)

        ival_QEJUN_to_D_start = Period(freq="D", year=2006, month=7, day=1)
        ival_QEJUN_to_D_end = Period(freq="D", year=2006, month=9, day=30)

        assert ival_Q.asfreq("Y") == ival_Q_to_A
        assert ival_Q_end_of_year.asfreq("Y") == ival_Q_to_A

        assert ival_Q.asfreq("M", "s") == ival_Q_to_M_start
        assert ival_Q.asfreq("M", "E") == ival_Q_to_M_end
        assert ival_Q.asfreq("W", "s") == ival_Q_to_W_start
        assert ival_Q.asfreq("W", "E") == ival_Q_to_W_end
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_Q.asfreq("B", "s") == ival_Q_to_B_start
            assert ival_Q.asfreq("B", "E") == ival_Q_to_B_end
        assert ival_Q.asfreq("D", "s") == ival_Q_to_D_start
        assert ival_Q.asfreq("D", "E") == ival_Q_to_D_end
        assert ival_Q.asfreq("h", "s") == ival_Q_to_H_start
        assert ival_Q.asfreq("h", "E") == ival_Q_to_H_end
        assert ival_Q.asfreq("Min", "s") == ival_Q_to_T_start
        assert ival_Q.asfreq("Min", "E") == ival_Q_to_T_end
        assert ival_Q.asfreq("s", "s") == ival_Q_to_S_start
        assert ival_Q.asfreq("s", "E") == ival_Q_to_S_end

        assert ival_QEJAN.asfreq("D", "s") == ival_QEJAN_to_D_start
        assert ival_QEJAN.asfreq("D", "E") == ival_QEJAN_to_D_end
        assert ival_QEJUN.asfreq("D", "s") == ival_QEJUN_to_D_start
        assert ival_QEJUN.asfreq("D", "E") == ival_QEJUN_to_D_end

        assert ival_Q.asfreq("Q") == ival_Q

    def test_conv_monthly(self):
        # frequency conversion tests: from Monthly Frequency

        ival_M = Period(freq="M", year=2007, month=1)
        ival_M_end_of_year = Period(freq="M", year=2007, month=12)
        ival_M_end_of_quarter = Period(freq="M", year=2007, month=3)
        ival_M_to_A = Period(freq="Y", year=2007)
        ival_M_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_M_to_W_start = Period(freq="W", year=2007, month=1, day=1)
        ival_M_to_W_end = Period(freq="W", year=2007, month=1, day=31)
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_M_to_B_start = Period(freq="B", year=2007, month=1, day=1)
            ival_M_to_B_end = Period(freq="B", year=2007, month=1, day=31)
        ival_M_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_M_to_D_end = Period(freq="D", year=2007, month=1, day=31)
        ival_M_to_H_start = Period(freq="h", year=2007, month=1, day=1, hour=0)
        ival_M_to_H_end = Period(freq="h", year=2007, month=1, day=31, hour=23)
        ival_M_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_M_to_T_end = Period(
            freq="Min", year=2007, month=1, day=31, hour=23, minute=59
        )
        ival_M_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_M_to_S_end = Period(
            freq="s", year=2007, month=1, day=31, hour=23, minute=59, second=59
        )

        assert ival_M.asfreq("Y") == ival_M_to_A
        assert ival_M_end_of_year.asfreq("Y") == ival_M_to_A
        assert ival_M.asfreq("Q") == ival_M_to_Q
        assert ival_M_end_of_quarter.asfreq("Q") == ival_M_to_Q

        assert ival_M.asfreq("W", "s") == ival_M_to_W_start
        assert ival_M.asfreq("W", "E") == ival_M_to_W_end
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_M.asfreq("B", "s") == ival_M_to_B_start
            assert ival_M.asfreq("B", "E") == ival_M_to_B_end
        assert ival_M.asfreq("D", "s") == ival_M_to_D_start
        assert ival_M.asfreq("D", "E") == ival_M_to_D_end
        assert ival_M.asfreq("h", "s") == ival_M_to_H_start
        assert ival_M.asfreq("h", "E") == ival_M_to_H_end
        assert ival_M.asfreq("Min", "s") == ival_M_to_T_start
        assert ival_M.asfreq("Min", "E") == ival_M_to_T_end
        assert ival_M.asfreq("s", "s") == ival_M_to_S_start
        assert ival_M.asfreq("s", "E") == ival_M_to_S_end

        assert ival_M.asfreq("M") == ival_M

    def test_conv_weekly(self):
        # frequency conversion tests: from Weekly Frequency
        ival_W = Period(freq="W", year=2007, month=1, day=1)

        ival_WSUN = Period(freq="W", year=2007, month=1, day=7)
        ival_WSAT = Period(freq="W-SAT", year=2007, month=1, day=6)
        ival_WFRI = Period(freq="W-FRI", year=2007, month=1, day=5)
        ival_WTHU = Period(freq="W-THU", year=2007, month=1, day=4)
        ival_WWED = Period(freq="W-WED", year=2007, month=1, day=3)
        ival_WTUE = Period(freq="W-TUE", year=2007, month=1, day=2)
        ival_WMON = Period(freq="W-MON", year=2007, month=1, day=1)

        ival_WSUN_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_WSUN_to_D_end = Period(freq="D", year=2007, month=1, day=7)
        ival_WSAT_to_D_start = Period(freq="D", year=2006, month=12, day=31)
        ival_WSAT_to_D_end = Period(freq="D", year=2007, month=1, day=6)
        ival_WFRI_to_D_start = Period(freq="D", year=2006, month=12, day=30)
        ival_WFRI_to_D_end = Period(freq="D", year=2007, month=1, day=5)
        ival_WTHU_to_D_start = Period(freq="D", year=2006, month=12, day=29)
        ival_WTHU_to_D_end = Period(freq="D", year=2007, month=1, day=4)
        ival_WWED_to_D_start = Period(freq="D", year=2006, month=12, day=28)
        ival_WWED_to_D_end = Period(freq="D", year=2007, month=1, day=3)
        ival_WTUE_to_D_start = Period(freq="D", year=2006, month=12, day=27)
        ival_WTUE_to_D_end = Period(freq="D", year=2007, month=1, day=2)
        ival_WMON_to_D_start = Period(freq="D", year=2006, month=12, day=26)
        ival_WMON_to_D_end = Period(freq="D", year=2007, month=1, day=1)

        ival_W_end_of_year = Period(freq="W", year=2007, month=12, day=31)
        ival_W_end_of_quarter = Period(freq="W", year=2007, month=3, day=31)
        ival_W_end_of_month = Period(freq="W", year=2007, month=1, day=31)
        ival_W_to_A = Period(freq="Y", year=2007)
        ival_W_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_W_to_M = Period(freq="M", year=2007, month=1)

        if Period(freq="D", year=2007, month=12, day=31).weekday == 6:
            ival_W_to_A_end_of_year = Period(freq="Y", year=2007)
        else:
            ival_W_to_A_end_of_year = Period(freq="Y", year=2008)

        if Period(freq="D", year=2007, month=3, day=31).weekday == 6:
            ival_W_to_Q_end_of_quarter = Period(freq="Q", year=2007, quarter=1)
        else:
            ival_W_to_Q_end_of_quarter = Period(freq="Q", year=2007, quarter=2)

        if Period(freq="D", year=2007, month=1, day=31).weekday == 6:
            ival_W_to_M_end_of_month = Period(freq="M", year=2007, month=1)
        else:
            ival_W_to_M_end_of_month = Period(freq="M", year=2007, month=2)

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_W_to_B_start = Period(freq="B", year=2007, month=1, day=1)
            ival_W_to_B_end = Period(freq="B", year=2007, month=1, day=5)
        ival_W_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_W_to_D_end = Period(freq="D", year=2007, month=1, day=7)
        ival_W_to_H_start = Period(freq="h", year=2007, month=1, day=1, hour=0)
        ival_W_to_H_end = Period(freq="h", year=2007, month=1, day=7, hour=23)
        ival_W_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_W_to_T_end = Period(
            freq="Min", year=2007, month=1, day=7, hour=23, minute=59
        )
        ival_W_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_W_to_S_end = Period(
            freq="s", year=2007, month=1, day=7, hour=23, minute=59, second=59
        )

        assert ival_W.asfreq("Y") == ival_W_to_A
        assert ival_W_end_of_year.asfreq("Y") == ival_W_to_A_end_of_year

        assert ival_W.asfreq("Q") == ival_W_to_Q
        assert ival_W_end_of_quarter.asfreq("Q") == ival_W_to_Q_end_of_quarter

        assert ival_W.asfreq("M") == ival_W_to_M
        assert ival_W_end_of_month.asfreq("M") == ival_W_to_M_end_of_month

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_W.asfreq("B", "s") == ival_W_to_B_start
            assert ival_W.asfreq("B", "E") == ival_W_to_B_end

        assert ival_W.asfreq("D", "s") == ival_W_to_D_start
        assert ival_W.asfreq("D", "E") == ival_W_to_D_end

        assert ival_WSUN.asfreq("D", "s") == ival_WSUN_to_D_start
        assert ival_WSUN.asfreq("D", "E") == ival_WSUN_to_D_end
        assert ival_WSAT.asfreq("D", "s") == ival_WSAT_to_D_start
        assert ival_WSAT.asfreq("D", "E") == ival_WSAT_to_D_end
        assert ival_WFRI.asfreq("D", "s") == ival_WFRI_to_D_start
        assert ival_WFRI.asfreq("D", "E") == ival_WFRI_to_D_end
        assert ival_WTHU.asfreq("D", "s") == ival_WTHU_to_D_start
        assert ival_WTHU.asfreq("D", "E") == ival_WTHU_to_D_end
        assert ival_WWED.asfreq("D", "s") == ival_WWED_to_D_start
        assert ival_WWED.asfreq("D", "E") == ival_WWED_to_D_end
        assert ival_WTUE.asfreq("D", "s") == ival_WTUE_to_D_start
        assert ival_WTUE.asfreq("D", "E") == ival_WTUE_to_D_end
        assert ival_WMON.asfreq("D", "s") == ival_WMON_to_D_start
        assert ival_WMON.asfreq("D", "E") == ival_WMON_to_D_end

        assert ival_W.asfreq("h", "s") == ival_W_to_H_start
        assert ival_W.asfreq("h", "E") == ival_W_to_H_end
        assert ival_W.asfreq("Min", "s") == ival_W_to_T_start
        assert ival_W.asfreq("Min", "E") == ival_W_to_T_end
        assert ival_W.asfreq("s", "s") == ival_W_to_S_start
        assert ival_W.asfreq("s", "E") == ival_W_to_S_end

        assert ival_W.asfreq("W") == ival_W

        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            ival_W.asfreq("WK")

    def test_conv_weekly_legacy(self):
        # frequency conversion tests: from Weekly Frequency
        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK", year=2007, month=1, day=1)

        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-SAT", year=2007, month=1, day=6)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-FRI", year=2007, month=1, day=5)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-THU", year=2007, month=1, day=4)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-WED", year=2007, month=1, day=3)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-TUE", year=2007, month=1, day=2)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-MON", year=2007, month=1, day=1)

    def test_conv_business(self):
        # frequency conversion tests: from Business Frequency"

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_B = Period(freq="B", year=2007, month=1, day=1)
            ival_B_end_of_year = Period(freq="B", year=2007, month=12, day=31)
            ival_B_end_of_quarter = Period(freq="B", year=2007, month=3, day=30)
            ival_B_end_of_month = Period(freq="B", year=2007, month=1, day=31)
            ival_B_end_of_week = Period(freq="B", year=2007, month=1, day=5)

        ival_B_to_A = Period(freq="Y", year=2007)
        ival_B_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_B_to_M = Period(freq="M", year=2007, month=1)
        ival_B_to_W = Period(freq="W", year=2007, month=1, day=7)
        ival_B_to_D = Period(freq="D", year=2007, month=1, day=1)
        ival_B_to_H_start = Period(freq="h", year=2007, month=1, day=1, hour=0)
        ival_B_to_H_end = Period(freq="h", year=2007, month=1, day=1, hour=23)
        ival_B_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_B_to_T_end = Period(
            freq="Min", year=2007, month=1, day=1, hour=23, minute=59
        )
        ival_B_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_B_to_S_end = Period(
            freq="s", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )

        assert ival_B.asfreq("Y") == ival_B_to_A
        assert ival_B_end_of_year.asfreq("Y") == ival_B_to_A
        assert ival_B.asfreq("Q") == ival_B_to_Q
        assert ival_B_end_of_quarter.asfreq("Q") == ival_B_to_Q
        assert ival_B.asfreq("M") == ival_B_to_M
        assert ival_B_end_of_month.asfreq("M") == ival_B_to_M
        assert ival_B.asfreq("W") == ival_B_to_W
        assert ival_B_end_of_week.asfreq("W") == ival_B_to_W

        assert ival_B.asfreq("D") == ival_B_to_D

        assert ival_B.asfreq("h", "s") == ival_B_to_H_start
        assert ival_B.asfreq("h", "E") == ival_B_to_H_end
        assert ival_B.asfreq("Min", "s") == ival_B_to_T_start
        assert ival_B.asfreq("Min", "E") == ival_B_to_T_end
        assert ival_B.asfreq("s", "s") == ival_B_to_S_start
        assert ival_B.asfreq("s", "E") == ival_B_to_S_end

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_B.asfreq("B") == ival_B

    def test_conv_daily(self):
        # frequency conversion tests: from Business Frequency"

        ival_D = Period(freq="D", year=2007, month=1, day=1)
        ival_D_end_of_year = Period(freq="D", year=2007, month=12, day=31)
        ival_D_end_of_quarter = Period(freq="D", year=2007, month=3, day=31)
        ival_D_end_of_month = Period(freq="D", year=2007, month=1, day=31)
        ival_D_end_of_week = Period(freq="D", year=2007, month=1, day=7)

        ival_D_friday = Period(freq="D", year=2007, month=1, day=5)
        ival_D_saturday = Period(freq="D", year=2007, month=1, day=6)
        ival_D_sunday = Period(freq="D", year=2007, month=1, day=7)

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_B_friday = Period(freq="B", year=2007, month=1, day=5)
            ival_B_monday = Period(freq="B", year=2007, month=1, day=8)

        ival_D_to_A = Period(freq="Y", year=2007)

        ival_Deoq_to_AJAN = Period(freq="Y-JAN", year=2008)
        ival_Deoq_to_AJUN = Period(freq="Y-JUN", year=2007)
        ival_Deoq_to_ADEC = Period(freq="Y-DEC", year=2007)

        ival_D_to_QEJAN = Period(freq="Q-JAN", year=2007, quarter=4)
        ival_D_to_QEJUN = Period(freq="Q-JUN", year=2007, quarter=3)
        ival_D_to_QEDEC = Period(freq="Q-DEC", year=2007, quarter=1)

        ival_D_to_M = Period(freq="M", year=2007, month=1)
        ival_D_to_W = Period(freq="W", year=2007, month=1, day=7)

        ival_D_to_H_start = Period(freq="h", year=2007, month=1, day=1, hour=0)
        ival_D_to_H_end = Period(freq="h", year=2007, month=1, day=1, hour=23)
        ival_D_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_D_to_T_end = Period(
            freq="Min", year=2007, month=1, day=1, hour=23, minute=59
        )
        ival_D_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_D_to_S_end = Period(
            freq="s", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )

        assert ival_D.asfreq("Y") == ival_D_to_A

        assert ival_D_end_of_quarter.asfreq("Y-JAN") == ival_Deoq_to_AJAN
        assert ival_D_end_of_quarter.asfreq("Y-JUN") == ival_Deoq_to_AJUN
        assert ival_D_end_of_quarter.asfreq("Y-DEC") == ival_Deoq_to_ADEC

        assert ival_D_end_of_year.asfreq("Y") == ival_D_to_A
        assert ival_D_end_of_quarter.asfreq("Q") == ival_D_to_QEDEC
        assert ival_D.asfreq("Q-JAN") == ival_D_to_QEJAN
        assert ival_D.asfreq("Q-JUN") == ival_D_to_QEJUN
        assert ival_D.asfreq("Q-DEC") == ival_D_to_QEDEC
        assert ival_D.asfreq("M") == ival_D_to_M
        assert ival_D_end_of_month.asfreq("M") == ival_D_to_M
        assert ival_D.asfreq("W") == ival_D_to_W
        assert ival_D_end_of_week.asfreq("W") == ival_D_to_W

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_D_friday.asfreq("B") == ival_B_friday
            assert ival_D_saturday.asfreq("B", "s") == ival_B_friday
            assert ival_D_saturday.asfreq("B", "E") == ival_B_monday
            assert ival_D_sunday.asfreq("B", "s") == ival_B_friday
            assert ival_D_sunday.asfreq("B", "E") == ival_B_monday

        assert ival_D.asfreq("h", "s") == ival_D_to_H_start
        assert ival_D.asfreq("h", "E") == ival_D_to_H_end
        assert ival_D.asfreq("Min", "s") == ival_D_to_T_start
        assert ival_D.asfreq("Min", "E") == ival_D_to_T_end
        assert ival_D.asfreq("s", "s") == ival_D_to_S_start
        assert ival_D.asfreq("s", "E") == ival_D_to_S_end

        assert ival_D.asfreq("D") == ival_D

    def test_conv_hourly(self):
        # frequency conversion tests: from Hourly Frequency"

        ival_H = Period(freq="h", year=2007, month=1, day=1, hour=0)
        ival_H_end_of_year = Period(freq="h", year=2007, month=12, day=31, hour=23)
        ival_H_end_of_quarter = Period(freq="h", year=2007, month=3, day=31, hour=23)
        ival_H_end_of_month = Period(freq="h", year=2007, month=1, day=31, hour=23)
        ival_H_end_of_week = Period(freq="h", year=2007, month=1, day=7, hour=23)
        ival_H_end_of_day = Period(freq="h", year=2007, month=1, day=1, hour=23)
        ival_H_end_of_bus = Period(freq="h", year=2007, month=1, day=1, hour=23)

        ival_H_to_A = Period(freq="Y", year=2007)
        ival_H_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_H_to_M = Period(freq="M", year=2007, month=1)
        ival_H_to_W = Period(freq="W", year=2007, month=1, day=7)
        ival_H_to_D = Period(freq="D", year=2007, month=1, day=1)
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_H_to_B = Period(freq="B", year=2007, month=1, day=1)

        ival_H_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_H_to_T_end = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=59
        )
        ival_H_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_H_to_S_end = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=59, second=59
        )

        assert ival_H.asfreq("Y") == ival_H_to_A
        assert ival_H_end_of_year.asfreq("Y") == ival_H_to_A
        assert ival_H.asfreq("Q") == ival_H_to_Q
        assert ival_H_end_of_quarter.asfreq("Q") == ival_H_to_Q
        assert ival_H.asfreq("M") == ival_H_to_M
        assert ival_H_end_of_month.asfreq("M") == ival_H_to_M
        assert ival_H.asfreq("W") == ival_H_to_W
        assert ival_H_end_of_week.asfreq("W") == ival_H_to_W
        assert ival_H.asfreq("D") == ival_H_to_D
        assert ival_H_end_of_day.asfreq("D") == ival_H_to_D
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_H.asfreq("B") == ival_H_to_B
            assert ival_H_end_of_bus.asfreq("B") == ival_H_to_B

        assert ival_H.asfreq("Min", "s") == ival_H_to_T_start
        assert ival_H.asfreq("Min", "E") == ival_H_to_T_end
        assert ival_H.asfreq("s", "s") == ival_H_to_S_start
        assert ival_H.asfreq("s", "E") == ival_H_to_S_end

        assert ival_H.asfreq("h") == ival_H

    def test_conv_minutely(self):
        # frequency conversion tests: from Minutely Frequency"

        ival_T = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)
        ival_T_end_of_year = Period(
            freq="Min", year=2007, month=12, day=31, hour=23, minute=59
        )
        ival_T_end_of_quarter = Period(
            freq="Min", year=2007, month=3, day=31, hour=23, minute=59
        )
        ival_T_end_of_month = Period(
            freq="Min", year=2007, month=1, day=31, hour=23, minute=59
        )
        ival_T_end_of_week = Period(
            freq="Min", year=2007, month=1, day=7, hour=23, minute=59
        )
        ival_T_end_of_day = Period(
            freq="Min", year=2007, month=1, day=1, hour=23, minute=59
        )
        ival_T_end_of_bus = Period(
            freq="Min", year=2007, month=1, day=1, hour=23, minute=59
        )
        ival_T_end_of_hour = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=59
        )

        ival_T_to_A = Period(freq="Y", year=2007)
        ival_T_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_T_to_M = Period(freq="M", year=2007, month=1)
        ival_T_to_W = Period(freq="W", year=2007, month=1, day=7)
        ival_T_to_D = Period(freq="D", year=2007, month=1, day=1)
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_T_to_B = Period(freq="B", year=2007, month=1, day=1)
        ival_T_to_H = Period(freq="h", year=2007, month=1, day=1, hour=0)

        ival_T_to_S_start = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_T_to_S_end = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=59
        )

        assert ival_T.asfreq("Y") == ival_T_to_A
        assert ival_T_end_of_year.asfreq("Y") == ival_T_to_A
        assert ival_T.asfreq("Q") == ival_T_to_Q
        assert ival_T_end_of_quarter.asfreq("Q") == ival_T_to_Q
        assert ival_T.asfreq("M") == ival_T_to_M
        assert ival_T_end_of_month.asfreq("M") == ival_T_to_M
        assert ival_T.asfreq("W") == ival_T_to_W
        assert ival_T_end_of_week.asfreq("W") == ival_T_to_W
        assert ival_T.asfreq("D") == ival_T_to_D
        assert ival_T_end_of_day.asfreq("D") == ival_T_to_D
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_T.asfreq("B") == ival_T_to_B
            assert ival_T_end_of_bus.asfreq("B") == ival_T_to_B
        assert ival_T.asfreq("h") == ival_T_to_H
        assert ival_T_end_of_hour.asfreq("h") == ival_T_to_H

        assert ival_T.asfreq("s", "s") == ival_T_to_S_start
        assert ival_T.asfreq("s", "E") == ival_T_to_S_end

        assert ival_T.asfreq("Min") == ival_T

    def test_conv_secondly(self):
        # frequency conversion tests: from Secondly Frequency"

        ival_S = Period(freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=0)
        ival_S_end_of_year = Period(
            freq="s", year=2007, month=12, day=31, hour=23, minute=59, second=59
        )
        ival_S_end_of_quarter = Period(
            freq="s", year=2007, month=3, day=31, hour=23, minute=59, second=59
        )
        ival_S_end_of_month = Period(
            freq="s", year=2007, month=1, day=31, hour=23, minute=59, second=59
        )
        ival_S_end_of_week = Period(
            freq="s", year=2007, month=1, day=7, hour=23, minute=59, second=59
        )
        ival_S_end_of_day = Period(
            freq="s", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )
        ival_S_end_of_bus = Period(
            freq="s", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )
        ival_S_end_of_hour = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=59, second=59
        )
        ival_S_end_of_minute = Period(
            freq="s", year=2007, month=1, day=1, hour=0, minute=0, second=59
        )

        ival_S_to_A = Period(freq="Y", year=2007)
        ival_S_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_S_to_M = Period(freq="M", year=2007, month=1)
        ival_S_to_W = Period(freq="W", year=2007, month=1, day=7)
        ival_S_to_D = Period(freq="D", year=2007, month=1, day=1)
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            ival_S_to_B = Period(freq="B", year=2007, month=1, day=1)
        ival_S_to_H = Period(freq="h", year=2007, month=1, day=1, hour=0)
        ival_S_to_T = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)

        assert ival_S.asfreq("Y") == ival_S_to_A
        assert ival_S_end_of_year.asfreq("Y") == ival_S_to_A
        assert ival_S.asfreq("Q") == ival_S_to_Q
        assert ival_S_end_of_quarter.asfreq("Q") == ival_S_to_Q
        assert ival_S.asfreq("M") == ival_S_to_M
        assert ival_S_end_of_month.asfreq("M") == ival_S_to_M
        assert ival_S.asfreq("W") == ival_S_to_W
        assert ival_S_end_of_week.asfreq("W") == ival_S_to_W
        assert ival_S.asfreq("D") == ival_S_to_D
        assert ival_S_end_of_day.asfreq("D") == ival_S_to_D
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert ival_S.asfreq("B") == ival_S_to_B
            assert ival_S_end_of_bus.asfreq("B") == ival_S_to_B
        assert ival_S.asfreq("h") == ival_S_to_H
        assert ival_S_end_of_hour.asfreq("h") == ival_S_to_H
        assert ival_S.asfreq("Min") == ival_S_to_T
        assert ival_S_end_of_minute.asfreq("Min") == ival_S_to_T

        assert ival_S.asfreq("s") == ival_S

    def test_conv_microsecond(self):
        # GH#31475 Avoid floating point errors dropping the start_time to
        #  before the beginning of the Period
        per = Period("2020-01-30 15:57:27.576166", freq="us")
        assert per.ordinal == 1580399847576166

        start = per.start_time
        expected = Timestamp("2020-01-30 15:57:27.576166")
        assert start == expected
        assert start._value == per.ordinal * 1000

        per2 = Period("2300-01-01", "us")
        msg = "2300-01-01"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            per2.start_time
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            per2.end_time

    def test_asfreq_mult(self):
        # normal freq to mult freq
        p = Period(freq="Y", year=2007)
        # ordinal will not change
        for freq in ["3Y", offsets.YearEnd(3)]:
            result = p.asfreq(freq)
            expected = Period("2007", freq="3Y")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq
        # ordinal will not change
        for freq in ["3Y", offsets.YearEnd(3)]:
            result = p.asfreq(freq, how="S")
            expected = Period("2007", freq="3Y")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

        # mult freq to normal freq
        p = Period(freq="3Y", year=2007)
        # ordinal will change because how=E is the default
        for freq in ["Y", offsets.YearEnd()]:
            result = p.asfreq(freq)
            expected = Period("2009", freq="Y")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq
        # ordinal will not change
        for freq in ["Y", offsets.YearEnd()]:
            result = p.asfreq(freq, how="s")
            expected = Period("2007", freq="Y")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

        p = Period(freq="Y", year=2007)
        for freq in ["2M", offsets.MonthEnd(2)]:
            result = p.asfreq(freq)
            expected = Period("2007-12", freq="2M")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq
        for freq in ["2M", offsets.MonthEnd(2)]:
            result = p.asfreq(freq, how="s")
            expected = Period("2007-01", freq="2M")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

        p = Period(freq="3Y", year=2007)
        for freq in ["2M", offsets.MonthEnd(2)]:
            result = p.asfreq(freq)
            expected = Period("2009-12", freq="2M")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq
        for freq in ["2M", offsets.MonthEnd(2)]:
            result = p.asfreq(freq, how="s")
            expected = Period("2007-01", freq="2M")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

    def test_asfreq_combined(self):
        # normal freq to combined freq
        p = Period("2007", freq="h")

        # ordinal will not change
        expected = Period("2007", freq="25h")
        for freq, how in zip(["1D1h", "1h1D"], ["E", "S"]):
            result = p.asfreq(freq, how=how)
            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

        # combined freq to normal freq
        p1 = Period(freq="1D1h", year=2007)
        p2 = Period(freq="1h1D", year=2007)

        # ordinal will change because how=E is the default
        result1 = p1.asfreq("h")
        result2 = p2.asfreq("h")
        expected = Period("2007-01-02", freq="h")
        assert result1 == expected
        assert result1.ordinal == expected.ordinal
        assert result1.freq == expected.freq
        assert result2 == expected
        assert result2.ordinal == expected.ordinal
        assert result2.freq == expected.freq

        # ordinal will not change
        result1 = p1.asfreq("h", how="S")
        result2 = p2.asfreq("h", how="S")
        expected = Period("2007-01-01", freq="h")
        assert result1 == expected
        assert result1.ordinal == expected.ordinal
        assert result1.freq == expected.freq
        assert result2 == expected
        assert result2.ordinal == expected.ordinal
        assert result2.freq == expected.freq

    def test_asfreq_MS(self):
        initial = Period("2013")

        assert initial.asfreq(freq="M", how="S") == Period("2013-01", "M")

        msg = "MS is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            initial.asfreq(freq="MS", how="S")

        with pytest.raises(ValueError, match=msg):
            Period("2013-01", "MS")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_defchararray.py ===
import pytest

import numpy as np
from numpy._core.multiarray import _vec_string
from numpy.testing import (
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)

kw_unicode_true = {'unicode': True}  # make 2to3 work properly
kw_unicode_false = {'unicode': False}

class TestBasic:
    def test_from_object_array(self):
        A = np.array([['abc', 2],
                      ['long   ', '0123456789']], dtype='O')
        B = np.char.array(A)
        assert_equal(B.dtype.itemsize, 10)
        assert_array_equal(B, [[b'abc', b'2'],
                               [b'long', b'0123456789']])

    def test_from_object_array_unicode(self):
        A = np.array([['abc', 'Sigma \u03a3'],
                      ['long   ', '0123456789']], dtype='O')
        assert_raises(ValueError, np.char.array, (A,))
        B = np.char.array(A, **kw_unicode_true)
        assert_equal(B.dtype.itemsize, 10 * np.array('a', 'U').dtype.itemsize)
        assert_array_equal(B, [['abc', 'Sigma \u03a3'],
                               ['long', '0123456789']])

    def test_from_string_array(self):
        A = np.array([[b'abc', b'foo'],
                      [b'long   ', b'0123456789']])
        assert_equal(A.dtype.type, np.bytes_)
        B = np.char.array(A)
        assert_array_equal(B, A)
        assert_equal(B.dtype, A.dtype)
        assert_equal(B.shape, A.shape)
        B[0, 0] = 'changed'
        assert_(B[0, 0] != A[0, 0])
        C = np.char.asarray(A)
        assert_array_equal(C, A)
        assert_equal(C.dtype, A.dtype)
        C[0, 0] = 'changed again'
        assert_(C[0, 0] != B[0, 0])
        assert_(C[0, 0] == A[0, 0])

    def test_from_unicode_array(self):
        A = np.array([['abc', 'Sigma \u03a3'],
                      ['long   ', '0123456789']])
        assert_equal(A.dtype.type, np.str_)
        B = np.char.array(A)
        assert_array_equal(B, A)
        assert_equal(B.dtype, A.dtype)
        assert_equal(B.shape, A.shape)
        B = np.char.array(A, **kw_unicode_true)
        assert_array_equal(B, A)
        assert_equal(B.dtype, A.dtype)
        assert_equal(B.shape, A.shape)

        def fail():
            np.char.array(A, **kw_unicode_false)

        assert_raises(UnicodeEncodeError, fail)

    def test_unicode_upconvert(self):
        A = np.char.array(['abc'])
        B = np.char.array(['\u03a3'])
        assert_(issubclass((A + B).dtype.type, np.str_))

    def test_from_string(self):
        A = np.char.array(b'abc')
        assert_equal(len(A), 1)
        assert_equal(len(A[0]), 3)
        assert_(issubclass(A.dtype.type, np.bytes_))

    def test_from_unicode(self):
        A = np.char.array('\u03a3')
        assert_equal(len(A), 1)
        assert_equal(len(A[0]), 1)
        assert_equal(A.itemsize, 4)
        assert_(issubclass(A.dtype.type, np.str_))

class TestVecString:
    def test_non_existent_method(self):

        def fail():
            _vec_string('a', np.bytes_, 'bogus')

        assert_raises(AttributeError, fail)

    def test_non_string_array(self):

        def fail():
            _vec_string(1, np.bytes_, 'strip')

        assert_raises(TypeError, fail)

    def test_invalid_args_tuple(self):

        def fail():
            _vec_string(['a'], np.bytes_, 'strip', 1)

        assert_raises(TypeError, fail)

    def test_invalid_type_descr(self):

        def fail():
            _vec_string(['a'], 'BOGUS', 'strip')

        assert_raises(TypeError, fail)

    def test_invalid_function_args(self):

        def fail():
            _vec_string(['a'], np.bytes_, 'strip', (1,))

        assert_raises(TypeError, fail)

    def test_invalid_result_type(self):

        def fail():
            _vec_string(['a'], np.int_, 'strip')

        assert_raises(TypeError, fail)

    def test_broadcast_error(self):

        def fail():
            _vec_string([['abc', 'def']], np.int_, 'find', (['a', 'd', 'j'],))

        assert_raises(ValueError, fail)


class TestWhitespace:
    def setup_method(self):
        self.A = np.array([['abc ', '123  '],
                           ['789 ', 'xyz ']]).view(np.char.chararray)
        self.B = np.array([['abc', '123'],
                           ['789', 'xyz']]).view(np.char.chararray)

    def test1(self):
        assert_(np.all(self.A == self.B))
        assert_(np.all(self.A >= self.B))
        assert_(np.all(self.A <= self.B))
        assert_(not np.any(self.A > self.B))
        assert_(not np.any(self.A < self.B))
        assert_(not np.any(self.A != self.B))

class TestChar:
    def setup_method(self):
        self.A = np.array('abc1', dtype='c').view(np.char.chararray)

    def test_it(self):
        assert_equal(self.A.shape, (4,))
        assert_equal(self.A.upper()[:2].tobytes(), b'AB')

class TestComparisons:
    def setup_method(self):
        self.A = np.array([['abc', 'abcc', '123'],
                           ['789', 'abc', 'xyz']]).view(np.char.chararray)
        self.B = np.array([['efg', 'efg', '123  '],
                           ['051', 'efgg', 'tuv']]).view(np.char.chararray)

    def test_not_equal(self):
        assert_array_equal((self.A != self.B),
                           [[True, True, False], [True, True, True]])

    def test_equal(self):
        assert_array_equal((self.A == self.B),
                           [[False, False, True], [False, False, False]])

    def test_greater_equal(self):
        assert_array_equal((self.A >= self.B),
                           [[False, False, True], [True, False, True]])

    def test_less_equal(self):
        assert_array_equal((self.A <= self.B),
                           [[True, True, True], [False, True, False]])

    def test_greater(self):
        assert_array_equal((self.A > self.B),
                           [[False, False, False], [True, False, True]])

    def test_less(self):
        assert_array_equal((self.A < self.B),
                           [[True, True, False], [False, True, False]])

    def test_type(self):
        out1 = np.char.equal(self.A, self.B)
        out2 = np.char.equal('a', 'a')
        assert_(isinstance(out1, np.ndarray))
        assert_(isinstance(out2, np.ndarray))

class TestComparisonsMixed1(TestComparisons):
    """Ticket #1276"""

    def setup_method(self):
        TestComparisons.setup_method(self)
        self.B = np.array(
            [['efg', 'efg', '123  '],
             ['051', 'efgg', 'tuv']], np.str_).view(np.char.chararray)

class TestComparisonsMixed2(TestComparisons):
    """Ticket #1276"""

    def setup_method(self):
        TestComparisons.setup_method(self)
        self.A = np.array(
            [['abc', 'abcc', '123'],
             ['789', 'abc', 'xyz']], np.str_).view(np.char.chararray)

class TestInformation:
    def setup_method(self):
        self.A = np.array([[' abc ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']]) \
                            .view(np.char.chararray)
        self.B = np.array([[' \u03a3 ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']]) \
                            .view(np.char.chararray)
        # Array with longer strings, > MEMCHR_CUT_OFF in code.
        self.C = (np.array(['ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                            '01234567890123456789012345'])
                  .view(np.char.chararray))

    def test_len(self):
        assert_(issubclass(np.char.str_len(self.A).dtype.type, np.integer))
        assert_array_equal(np.char.str_len(self.A), [[5, 0], [5, 9], [12, 5]])
        assert_array_equal(np.char.str_len(self.B), [[3, 0], [5, 9], [12, 5]])

    def test_count(self):
        assert_(issubclass(self.A.count('').dtype.type, np.integer))
        assert_array_equal(self.A.count('a'), [[1, 0], [0, 1], [0, 0]])
        assert_array_equal(self.A.count('123'), [[0, 0], [1, 0], [1, 0]])
        # Python doesn't seem to like counting NULL characters
        # assert_array_equal(self.A.count('\0'), [[0, 0], [0, 0], [1, 0]])
        assert_array_equal(self.A.count('a', 0, 2), [[1, 0], [0, 0], [0, 0]])
        assert_array_equal(self.B.count('a'), [[0, 0], [0, 1], [0, 0]])
        assert_array_equal(self.B.count('123'), [[0, 0], [1, 0], [1, 0]])
        # assert_array_equal(self.B.count('\0'), [[0, 0], [0, 0], [1, 0]])

    def test_endswith(self):
        assert_(issubclass(self.A.endswith('').dtype.type, np.bool))
        assert_array_equal(self.A.endswith(' '), [[1, 0], [0, 0], [1, 0]])
        assert_array_equal(self.A.endswith('3', 0, 3), [[0, 0], [1, 0], [1, 0]])

        def fail():
            self.A.endswith('3', 'fdjk')

        assert_raises(TypeError, fail)

    @pytest.mark.parametrize(
        "dtype, encode",
        [("U", str),
         ("S", lambda x: x.encode('ascii')),
         ])
    def test_find(self, dtype, encode):
        A = self.A.astype(dtype)
        assert_(issubclass(A.find(encode('a')).dtype.type, np.integer))
        assert_array_equal(A.find(encode('a')),
                           [[1, -1], [-1, 6], [-1, -1]])
        assert_array_equal(A.find(encode('3')),
                           [[-1, -1], [2, -1], [2, -1]])
        assert_array_equal(A.find(encode('a'), 0, 2),
                           [[1, -1], [-1, -1], [-1, -1]])
        assert_array_equal(A.find([encode('1'), encode('P')]),
                           [[-1, -1], [0, -1], [0, 1]])
        C = self.C.astype(dtype)
        assert_array_equal(C.find(encode('M')), [12, -1])

    def test_index(self):

        def fail():
            self.A.index('a')

        assert_raises(ValueError, fail)
        assert_(np.char.index('abcba', 'b') == 1)
        assert_(issubclass(np.char.index('abcba', 'b').dtype.type, np.integer))

    def test_isalnum(self):
        assert_(issubclass(self.A.isalnum().dtype.type, np.bool))
        assert_array_equal(self.A.isalnum(), [[False, False], [True, True], [False, True]])

    def test_isalpha(self):
        assert_(issubclass(self.A.isalpha().dtype.type, np.bool))
        assert_array_equal(self.A.isalpha(), [[False, False], [False, True], [False, True]])

    def test_isdigit(self):
        assert_(issubclass(self.A.isdigit().dtype.type, np.bool))
        assert_array_equal(self.A.isdigit(), [[False, False], [True, False], [False, False]])

    def test_islower(self):
        assert_(issubclass(self.A.islower().dtype.type, np.bool))
        assert_array_equal(self.A.islower(), [[True, False], [False, False], [False, False]])

    def test_isspace(self):
        assert_(issubclass(self.A.isspace().dtype.type, np.bool))
        assert_array_equal(self.A.isspace(), [[False, False], [False, False], [False, False]])

    def test_istitle(self):
        assert_(issubclass(self.A.istitle().dtype.type, np.bool))
        assert_array_equal(self.A.istitle(), [[False, False], [False, False], [False, False]])

    def test_isupper(self):
        assert_(issubclass(self.A.isupper().dtype.type, np.bool))
        assert_array_equal(self.A.isupper(), [[False, False], [False, False], [False, True]])

    def test_rfind(self):
        assert_(issubclass(self.A.rfind('a').dtype.type, np.integer))
        assert_array_equal(self.A.rfind('a'), [[1, -1], [-1, 6], [-1, -1]])
        assert_array_equal(self.A.rfind('3'), [[-1, -1], [2, -1], [6, -1]])
        assert_array_equal(self.A.rfind('a', 0, 2), [[1, -1], [-1, -1], [-1, -1]])
        assert_array_equal(self.A.rfind(['1', 'P']), [[-1, -1], [0, -1], [0, 2]])

    def test_rindex(self):

        def fail():
            self.A.rindex('a')

        assert_raises(ValueError, fail)
        assert_(np.char.rindex('abcba', 'b') == 3)
        assert_(issubclass(np.char.rindex('abcba', 'b').dtype.type, np.integer))

    def test_startswith(self):
        assert_(issubclass(self.A.startswith('').dtype.type, np.bool))
        assert_array_equal(self.A.startswith(' '), [[1, 0], [0, 0], [0, 0]])
        assert_array_equal(self.A.startswith('1', 0, 3), [[0, 0], [1, 0], [1, 0]])

        def fail():
            self.A.startswith('3', 'fdjk')

        assert_raises(TypeError, fail)


class TestMethods:
    def setup_method(self):
        self.A = np.array([[' abc ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']],
                          dtype='S').view(np.char.chararray)
        self.B = np.array([[' \u03a3 ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']]).view(
                                                            np.char.chararray)

    def test_capitalize(self):
        tgt = [[b' abc ', b''],
               [b'12345', b'Mixedcase'],
               [b'123 \t 345 \0 ', b'Upper']]
        assert_(issubclass(self.A.capitalize().dtype.type, np.bytes_))
        assert_array_equal(self.A.capitalize(), tgt)

        tgt = [[' \u03c3 ', ''],
               ['12345', 'Mixedcase'],
               ['123 \t 345 \0 ', 'Upper']]
        assert_(issubclass(self.B.capitalize().dtype.type, np.str_))
        assert_array_equal(self.B.capitalize(), tgt)

    def test_center(self):
        assert_(issubclass(self.A.center(10).dtype.type, np.bytes_))
        C = self.A.center([10, 20])
        assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])

        C = self.A.center(20, b'#')
        assert_(np.all(C.startswith(b'#')))
        assert_(np.all(C.endswith(b'#')))

        C = np.char.center(b'FOO', [[10, 20], [15, 8]])
        tgt = [[b'   FOO    ', b'        FOO         '],
               [b'      FOO      ', b'  FOO   ']]
        assert_(issubclass(C.dtype.type, np.bytes_))
        assert_array_equal(C, tgt)

    def test_decode(self):
        A = np.char.array([b'\\u03a3'])
        assert_(A.decode('unicode-escape')[0] == '\u03a3')

    def test_encode(self):
        B = self.B.encode('unicode_escape')
        assert_(B[0][0] == ' \\u03a3 '.encode('latin1'))

    def test_expandtabs(self):
        T = self.A.expandtabs()
        assert_(T[2, 0] == b'123      345 \0')

    def test_join(self):
        # NOTE: list(b'123') == [49, 50, 51]
        #       so that b','.join(b'123') results to an error on Py3
        A0 = self.A.decode('ascii')

        A = np.char.join([',', '#'], A0)
        assert_(issubclass(A.dtype.type, np.str_))
        tgt = np.array([[' ,a,b,c, ', ''],
                        ['1,2,3,4,5', 'M#i#x#e#d#C#a#s#e'],
                        ['1,2,3, ,\t, ,3,4,5, ,\x00, ', 'U#P#P#E#R']])
        assert_array_equal(np.char.join([',', '#'], A0), tgt)

    def test_ljust(self):
        assert_(issubclass(self.A.ljust(10).dtype.type, np.bytes_))

        C = self.A.ljust([10, 20])
        assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])

        C = self.A.ljust(20, b'#')
        assert_array_equal(C.startswith(b'#'), [
                [False, True], [False, False], [False, False]])
        assert_(np.all(C.endswith(b'#')))

        C = np.char.ljust(b'FOO', [[10, 20], [15, 8]])
        tgt = [[b'FOO       ', b'FOO                 '],
               [b'FOO            ', b'FOO     ']]
        assert_(issubclass(C.dtype.type, np.bytes_))
        assert_array_equal(C, tgt)

    def test_lower(self):
        tgt = [[b' abc ', b''],
               [b'12345', b'mixedcase'],
               [b'123 \t 345 \0 ', b'upper']]
        assert_(issubclass(self.A.lower().dtype.type, np.bytes_))
        assert_array_equal(self.A.lower(), tgt)

        tgt = [[' \u03c3 ', ''],
               ['12345', 'mixedcase'],
               ['123 \t 345 \0 ', 'upper']]
        assert_(issubclass(self.B.lower().dtype.type, np.str_))
        assert_array_equal(self.B.lower(), tgt)

    def test_lstrip(self):
        tgt = [[b'abc ', b''],
               [b'12345', b'MixedCase'],
               [b'123 \t 345 \0 ', b'UPPER']]
        assert_(issubclass(self.A.lstrip().dtype.type, np.bytes_))
        assert_array_equal(self.A.lstrip(), tgt)

        tgt = [[b' abc', b''],
               [b'2345', b'ixedCase'],
               [b'23 \t 345 \x00', b'UPPER']]
        assert_array_equal(self.A.lstrip([b'1', b'M']), tgt)

        tgt = [['\u03a3 ', ''],
               ['12345', 'MixedCase'],
               ['123 \t 345 \0 ', 'UPPER']]
        assert_(issubclass(self.B.lstrip().dtype.type, np.str_))
        assert_array_equal(self.B.lstrip(), tgt)

    def test_partition(self):
        P = self.A.partition([b'3', b'M'])
        tgt = [[(b' abc ', b'', b''), (b'', b'', b'')],
               [(b'12', b'3', b'45'), (b'', b'M', b'ixedCase')],
               [(b'12', b'3', b' \t 345 \0 '), (b'UPPER', b'', b'')]]
        assert_(issubclass(P.dtype.type, np.bytes_))
        assert_array_equal(P, tgt)

    def test_replace(self):
        R = self.A.replace([b'3', b'a'],
                           [b'##########', b'@'])
        tgt = [[b' abc ', b''],
               [b'12##########45', b'MixedC@se'],
               [b'12########## \t ##########45 \x00 ', b'UPPER']]
        assert_(issubclass(R.dtype.type, np.bytes_))
        assert_array_equal(R, tgt)
        # Test special cases that should just return the input array,
        # since replacements are not possible or do nothing.
        S1 = self.A.replace(b'A very long byte string, longer than A', b'')
        assert_array_equal(S1, self.A)
        S2 = self.A.replace(b'', b'')
        assert_array_equal(S2, self.A)
        S3 = self.A.replace(b'3', b'3')
        assert_array_equal(S3, self.A)
        S4 = self.A.replace(b'3', b'', count=0)
        assert_array_equal(S4, self.A)

    def test_replace_count_and_size(self):
        a = np.array(['0123456789' * i for i in range(4)]
                     ).view(np.char.chararray)
        r1 = a.replace('5', 'ABCDE')
        assert r1.dtype.itemsize == (3 * 10 + 3 * 4) * 4
        assert_array_equal(r1, np.array(['01234ABCDE6789' * i
                                         for i in range(4)]))
        r2 = a.replace('5', 'ABCDE', count=1)
        assert r2.dtype.itemsize == (3 * 10 + 4) * 4
        r3 = a.replace('5', 'ABCDE', count=0)
        assert r3.dtype.itemsize == a.dtype.itemsize
        assert_array_equal(r3, a)
        # Negative values mean to replace all.
        r4 = a.replace('5', 'ABCDE', count=-1)
        assert r4.dtype.itemsize == (3 * 10 + 3 * 4) * 4
        assert_array_equal(r4, r1)
        # We can do count on an element-by-element basis.
        r5 = a.replace('5', 'ABCDE', count=[-1, -1, -1, 1])
        assert r5.dtype.itemsize == (3 * 10 + 4) * 4
        assert_array_equal(r5, np.array(
            ['01234ABCDE6789' * i for i in range(3)]
            + ['01234ABCDE6789' + '0123456789' * 2]))

    def test_replace_broadcasting(self):
        a = np.array('0,0,0').view(np.char.chararray)
        r1 = a.replace('0', '1', count=np.arange(3))
        assert r1.dtype == a.dtype
        assert_array_equal(r1, np.array(['0,0,0', '1,0,0', '1,1,0']))
        r2 = a.replace('0', [['1'], ['2']], count=np.arange(1, 4))
        assert_array_equal(r2, np.array([['1,0,0', '1,1,0', '1,1,1'],
                                         ['2,0,0', '2,2,0', '2,2,2']]))
        r3 = a.replace(['0', '0,0', '0,0,0'], 'X')
        assert_array_equal(r3, np.array(['X,X,X', 'X,0', 'X']))

    def test_rjust(self):
        assert_(issubclass(self.A.rjust(10).dtype.type, np.bytes_))

        C = self.A.rjust([10, 20])
        assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])

        C = self.A.rjust(20, b'#')
        assert_(np.all(C.startswith(b'#')))
        assert_array_equal(C.endswith(b'#'),
                           [[False, True], [False, False], [False, False]])

        C = np.char.rjust(b'FOO', [[10, 20], [15, 8]])
        tgt = [[b'       FOO', b'                 FOO'],
               [b'            FOO', b'     FOO']]
        assert_(issubclass(C.dtype.type, np.bytes_))
        assert_array_equal(C, tgt)

    def test_rpartition(self):
        P = self.A.rpartition([b'3', b'M'])
        tgt = [[(b'', b'', b' abc '), (b'', b'', b'')],
               [(b'12', b'3', b'45'), (b'', b'M', b'ixedCase')],
               [(b'123 \t ', b'3', b'45 \0 '), (b'', b'', b'UPPER')]]
        assert_(issubclass(P.dtype.type, np.bytes_))
        assert_array_equal(P, tgt)

    def test_rsplit(self):
        A = self.A.rsplit(b'3')
        tgt = [[[b' abc '], [b'']],
               [[b'12', b'45'], [b'MixedCase']],
               [[b'12', b' \t ', b'45 \x00 '], [b'UPPER']]]
        assert_(issubclass(A.dtype.type, np.object_))
        assert_equal(A.tolist(), tgt)

    def test_rstrip(self):
        assert_(issubclass(self.A.rstrip().dtype.type, np.bytes_))

        tgt = [[b' abc', b''],
               [b'12345', b'MixedCase'],
               [b'123 \t 345', b'UPPER']]
        assert_array_equal(self.A.rstrip(), tgt)

        tgt = [[b' abc ', b''],
               [b'1234', b'MixedCase'],
               [b'123 \t 345 \x00', b'UPP']
               ]
        assert_array_equal(self.A.rstrip([b'5', b'ER']), tgt)

        tgt = [[' \u03a3', ''],
               ['12345', 'MixedCase'],
               ['123 \t 345', 'UPPER']]
        assert_(issubclass(self.B.rstrip().dtype.type, np.str_))
        assert_array_equal(self.B.rstrip(), tgt)

    def test_strip(self):
        tgt = [[b'abc', b''],
               [b'12345', b'MixedCase'],
               [b'123 \t 345', b'UPPER']]
        assert_(issubclass(self.A.strip().dtype.type, np.bytes_))
        assert_array_equal(self.A.strip(), tgt)

        tgt = [[b' abc ', b''],
               [b'234', b'ixedCas'],
               [b'23 \t 345 \x00', b'UPP']]
        assert_array_equal(self.A.strip([b'15', b'EReM']), tgt)

        tgt = [['\u03a3', ''],
               ['12345', 'MixedCase'],
               ['123 \t 345', 'UPPER']]
        assert_(issubclass(self.B.strip().dtype.type, np.str_))
        assert_array_equal(self.B.strip(), tgt)

    def test_split(self):
        A = self.A.split(b'3')
        tgt = [
               [[b' abc '], [b'']],
               [[b'12', b'45'], [b'MixedCase']],
               [[b'12', b' \t ', b'45 \x00 '], [b'UPPER']]]
        assert_(issubclass(A.dtype.type, np.object_))
        assert_equal(A.tolist(), tgt)

    def test_splitlines(self):
        A = np.char.array(['abc\nfds\nwer']).splitlines()
        assert_(issubclass(A.dtype.type, np.object_))
        assert_(A.shape == (1,))
        assert_(len(A[0]) == 3)

    def test_swapcase(self):
        tgt = [[b' ABC ', b''],
               [b'12345', b'mIXEDcASE'],
               [b'123 \t 345 \0 ', b'upper']]
        assert_(issubclass(self.A.swapcase().dtype.type, np.bytes_))
        assert_array_equal(self.A.swapcase(), tgt)

        tgt = [[' \u03c3 ', ''],
               ['12345', 'mIXEDcASE'],
               ['123 \t 345 \0 ', 'upper']]
        assert_(issubclass(self.B.swapcase().dtype.type, np.str_))
        assert_array_equal(self.B.swapcase(), tgt)

    def test_title(self):
        tgt = [[b' Abc ', b''],
               [b'12345', b'Mixedcase'],
               [b'123 \t 345 \0 ', b'Upper']]
        assert_(issubclass(self.A.title().dtype.type, np.bytes_))
        assert_array_equal(self.A.title(), tgt)

        tgt = [[' \u03a3 ', ''],
               ['12345', 'Mixedcase'],
               ['123 \t 345 \0 ', 'Upper']]
        assert_(issubclass(self.B.title().dtype.type, np.str_))
        assert_array_equal(self.B.title(), tgt)

    def test_upper(self):
        tgt = [[b' ABC ', b''],
               [b'12345', b'MIXEDCASE'],
               [b'123 \t 345 \0 ', b'UPPER']]
        assert_(issubclass(self.A.upper().dtype.type, np.bytes_))
        assert_array_equal(self.A.upper(), tgt)

        tgt = [[' \u03a3 ', ''],
               ['12345', 'MIXEDCASE'],
               ['123 \t 345 \0 ', 'UPPER']]
        assert_(issubclass(self.B.upper().dtype.type, np.str_))
        assert_array_equal(self.B.upper(), tgt)

    def test_isnumeric(self):

        def fail():
            self.A.isnumeric()

        assert_raises(TypeError, fail)
        assert_(issubclass(self.B.isnumeric().dtype.type, np.bool))
        assert_array_equal(self.B.isnumeric(), [
                [False, False], [True, False], [False, False]])

    def test_isdecimal(self):

        def fail():
            self.A.isdecimal()

        assert_raises(TypeError, fail)
        assert_(issubclass(self.B.isdecimal().dtype.type, np.bool))
        assert_array_equal(self.B.isdecimal(), [
                [False, False], [True, False], [False, False]])


class TestOperations:
    def setup_method(self):
        self.A = np.array([['abc', '123'],
                           ['789', 'xyz']]).view(np.char.chararray)
        self.B = np.array([['efg', '456'],
                           ['051', 'tuv']]).view(np.char.chararray)

    def test_add(self):
        AB = np.array([['abcefg', '123456'],
                       ['789051', 'xyztuv']]).view(np.char.chararray)
        assert_array_equal(AB, (self.A + self.B))
        assert_(len((self.A + self.B)[0][0]) == 6)

    def test_radd(self):
        QA = np.array([['qabc', 'q123'],
                       ['q789', 'qxyz']]).view(np.char.chararray)
        assert_array_equal(QA, ('q' + self.A))

    def test_mul(self):
        A = self.A
        for r in (2, 3, 5, 7, 197):
            Ar = np.array([[A[0, 0] * r, A[0, 1] * r],
                           [A[1, 0] * r, A[1, 1] * r]]).view(np.char.chararray)

            assert_array_equal(Ar, (self.A * r))

        for ob in [object(), 'qrs']:
            with assert_raises_regex(ValueError,
                                     'Can only multiply by integers'):
                A * ob

    def test_rmul(self):
        A = self.A
        for r in (2, 3, 5, 7, 197):
            Ar = np.array([[A[0, 0] * r, A[0, 1] * r],
                           [A[1, 0] * r, A[1, 1] * r]]).view(np.char.chararray)
            assert_array_equal(Ar, (r * self.A))

        for ob in [object(), 'qrs']:
            with assert_raises_regex(ValueError,
                                     'Can only multiply by integers'):
                ob * A

    def test_mod(self):
        """Ticket #856"""
        F = np.array([['%d', '%f'], ['%s', '%r']]).view(np.char.chararray)
        C = np.array([[3, 7], [19, 1]], dtype=np.int64)
        FC = np.array([['3', '7.000000'],
                       ['19', 'np.int64(1)']]).view(np.char.chararray)
        assert_array_equal(FC, F % C)

        A = np.array([['%.3f', '%d'], ['%s', '%r']]).view(np.char.chararray)
        A1 = np.array([['1.000', '1'],
                       ['1', repr(np.array(1)[()])]]).view(np.char.chararray)
        assert_array_equal(A1, (A % 1))

        A2 = np.array([['1.000', '2'],
                       ['3', repr(np.array(4)[()])]]).view(np.char.chararray)
        assert_array_equal(A2, (A % [[1, 2], [3, 4]]))

    def test_rmod(self):
        assert_(f"{self.A}" == str(self.A))
        assert_(f"{self.A!r}" == repr(self.A))

        for ob in [42, object()]:
            with assert_raises_regex(
                    TypeError, "unsupported operand type.* and 'chararray'"):
                ob % self.A

    def test_slice(self):
        """Regression test for https://github.com/numpy/numpy/issues/5982"""

        arr = np.array([['abc ', 'def '], ['geh ', 'ijk ']],
                       dtype='S4').view(np.char.chararray)
        sl1 = arr[:]
        assert_array_equal(sl1, arr)
        assert_(sl1.base is arr)
        assert_(sl1.base.base is arr.base)

        sl2 = arr[:, :]
        assert_array_equal(sl2, arr)
        assert_(sl2.base is arr)
        assert_(sl2.base.base is arr.base)

        assert_(arr[0, 0] == b'abc')

    @pytest.mark.parametrize('data', [['plate', '   ', 'shrimp'],
                                      [b'retro', b'  ', b'encabulator']])
    def test_getitem_length_zero_item(self, data):
        # Regression test for gh-26375.
        a = np.char.array(data)
        # a.dtype.type() will be an empty string or bytes instance.
        # The equality test will fail if a[1] has the wrong type
        # or does not have length 0.
        assert_equal(a[1], a.dtype.type())


class TestMethodsEmptyArray:
    def setup_method(self):
        self.U = np.array([], dtype='U')
        self.S = np.array([], dtype='S')

    def test_encode(self):
        res = np.char.encode(self.U)
        assert_array_equal(res, [])
        assert_(res.dtype.char == 'S')

    def test_decode(self):
        res = np.char.decode(self.S)
        assert_array_equal(res, [])
        assert_(res.dtype.char == 'U')

    def test_decode_with_reshape(self):
        res = np.char.decode(self.S.reshape((1, 0, 1)))
        assert_(res.shape == (1, 0, 1))


class TestMethodsScalarValues:
    def test_mod(self):
        A = np.array([[' abc ', ''],
                      ['12345', 'MixedCase'],
                      ['123 \t 345 \0 ', 'UPPER']], dtype='S')
        tgt = [[b'123 abc ', b'123'],
               [b'12312345', b'123MixedCase'],
               [b'123123 \t 345 \0 ', b'123UPPER']]
        assert_array_equal(np.char.mod(b"123%s", A), tgt)

    def test_decode(self):
        bytestring = b'\x81\xc1\x81\xc1\x81\xc1'
        assert_equal(np.char.decode(bytestring, encoding='cp037'),
                     'aAaAaA')

    def test_encode(self):
        unicode = 'aAaAaA'
        assert_equal(np.char.encode(unicode, encoding='cp037'),
                     b'\x81\xc1\x81\xc1\x81\xc1')

    def test_expandtabs(self):
        s = "\tone level of indentation\n\t\ttwo levels of indentation"
        assert_equal(
            np.char.expandtabs(s, tabsize=2),
            "  one level of indentation\n    two levels of indentation"
        )

    def test_join(self):
        seps = np.array(['-', '_'])
        assert_array_equal(np.char.join(seps, 'hello'),
                           ['h-e-l-l-o', 'h_e_l_l_o'])

    def test_partition(self):
        assert_equal(np.char.partition('This string', ' '),
                     ['This', ' ', 'string'])

    def test_rpartition(self):
        assert_equal(np.char.rpartition('This string here', ' '),
                     ['This string', ' ', 'here'])

    def test_replace(self):
        assert_equal(np.char.replace('Python is good', 'good', 'great'),
                     'Python is great')


def test_empty_indexing():
    """Regression test for ticket 1948."""
    # Check that indexing a chararray with an empty list/array returns an
    # empty chararray instead of a chararray with a single empty string in it.
    s = np.char.chararray((4,))
    assert_(s[[]].size == 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_subclass.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)


@pytest.fixture()
def gpd_style_subclass_df():
    class SubclassedDataFrame(DataFrame):
        @property
        def _constructor(self):
            return SubclassedDataFrame

    return SubclassedDataFrame({"a": [1, 2, 3]})


class TestDataFrameSubclassing:
    def test_no_warning_on_mgr(self):
        # GH#57032
        df = tm.SubclassedDataFrame(
            {"X": [1, 2, 3], "Y": [1, 2, 3]}, index=["a", "b", "c"]
        )
        with tm.assert_produces_warning(None):
            # df.isna() goes through _constructor_from_mgr, which we want to
            #  *not* pass a Manager do __init__
            df.isna()
            df["X"].isna()

    def test_frame_subclassing_and_slicing(self):
        # Subclass frame and ensure it returns the right class on slicing it
        # In reference to PR 9632

        class CustomSeries(Series):
            @property
            def _constructor(self):
                return CustomSeries

            def custom_series_function(self):
                return "OK"

        class CustomDataFrame(DataFrame):
            """
            Subclasses pandas DF, fills DF with simulation results, adds some
            custom plotting functions.
            """

            def __init__(self, *args, **kw) -> None:
                super().__init__(*args, **kw)

            @property
            def _constructor(self):
                return CustomDataFrame

            _constructor_sliced = CustomSeries

            def custom_frame_function(self):
                return "OK"

        data = {"col1": range(10), "col2": range(10)}
        cdf = CustomDataFrame(data)

        # Did we get back our own DF class?
        assert isinstance(cdf, CustomDataFrame)

        # Do we get back our own Series class after selecting a column?
        cdf_series = cdf.col1
        assert isinstance(cdf_series, CustomSeries)
        assert cdf_series.custom_series_function() == "OK"

        # Do we get back our own DF class after slicing row-wise?
        cdf_rows = cdf[1:5]
        assert isinstance(cdf_rows, CustomDataFrame)
        assert cdf_rows.custom_frame_function() == "OK"

        # Make sure sliced part of multi-index frame is custom class
        mcol = MultiIndex.from_tuples([("A", "A"), ("A", "B")])
        cdf_multi = CustomDataFrame([[0, 1], [2, 3]], columns=mcol)
        assert isinstance(cdf_multi["A"], CustomDataFrame)

        mcol = MultiIndex.from_tuples([("A", ""), ("B", "")])
        cdf_multi2 = CustomDataFrame([[0, 1], [2, 3]], columns=mcol)
        assert isinstance(cdf_multi2["A"], CustomSeries)

    def test_dataframe_metadata(self):
        df = tm.SubclassedDataFrame(
            {"X": [1, 2, 3], "Y": [1, 2, 3]}, index=["a", "b", "c"]
        )
        df.testattr = "XXX"

        assert df.testattr == "XXX"
        assert df[["X"]].testattr == "XXX"
        assert df.loc[["a", "b"], :].testattr == "XXX"
        assert df.iloc[[0, 1], :].testattr == "XXX"

        # see gh-9776
        assert df.iloc[0:1, :].testattr == "XXX"

        # see gh-10553
        unpickled = tm.round_trip_pickle(df)
        tm.assert_frame_equal(df, unpickled)
        assert df._metadata == unpickled._metadata
        assert df.testattr == unpickled.testattr

    def test_indexing_sliced(self):
        # GH 11559
        df = tm.SubclassedDataFrame(
            {"X": [1, 2, 3], "Y": [4, 5, 6], "Z": [7, 8, 9]}, index=["a", "b", "c"]
        )
        res = df.loc[:, "X"]
        exp = tm.SubclassedSeries([1, 2, 3], index=list("abc"), name="X")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.iloc[:, 1]
        exp = tm.SubclassedSeries([4, 5, 6], index=list("abc"), name="Y")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.loc[:, "Z"]
        exp = tm.SubclassedSeries([7, 8, 9], index=list("abc"), name="Z")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.loc["a", :]
        exp = tm.SubclassedSeries([1, 4, 7], index=list("XYZ"), name="a")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.iloc[1, :]
        exp = tm.SubclassedSeries([2, 5, 8], index=list("XYZ"), name="b")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

        res = df.loc["c", :]
        exp = tm.SubclassedSeries([3, 6, 9], index=list("XYZ"), name="c")
        tm.assert_series_equal(res, exp)
        assert isinstance(res, tm.SubclassedSeries)

    def test_subclass_attr_err_propagation(self):
        # GH 11808
        class A(DataFrame):
            @property
            def nonexistence(self):
                return self.i_dont_exist

        with pytest.raises(AttributeError, match=".*i_dont_exist.*"):
            A().nonexistence

    def test_subclass_align(self):
        # GH 12983
        df1 = tm.SubclassedDataFrame(
            {"a": [1, 3, 5], "b": [1, 3, 5]}, index=list("ACE")
        )
        df2 = tm.SubclassedDataFrame(
            {"c": [1, 2, 4], "d": [1, 2, 4]}, index=list("ABD")
        )

        res1, res2 = df1.align(df2, axis=0)
        exp1 = tm.SubclassedDataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, 3, np.nan, 5]},
            index=list("ABCDE"),
        )
        exp2 = tm.SubclassedDataFrame(
            {"c": [1, 2, np.nan, 4, np.nan], "d": [1, 2, np.nan, 4, np.nan]},
            index=list("ABCDE"),
        )
        assert isinstance(res1, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res1, exp1)
        assert isinstance(res2, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res2, exp2)

        res1, res2 = df1.a.align(df2.c)
        assert isinstance(res1, tm.SubclassedSeries)
        tm.assert_series_equal(res1, exp1.a)
        assert isinstance(res2, tm.SubclassedSeries)
        tm.assert_series_equal(res2, exp2.c)

    def test_subclass_align_combinations(self):
        # GH 12983
        df = tm.SubclassedDataFrame({"a": [1, 3, 5], "b": [1, 3, 5]}, index=list("ACE"))
        s = tm.SubclassedSeries([1, 2, 4], index=list("ABD"), name="x")

        # frame + series
        res1, res2 = df.align(s, axis=0)
        exp1 = tm.SubclassedDataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, 3, np.nan, 5]},
            index=list("ABCDE"),
        )
        # name is lost when
        exp2 = tm.SubclassedSeries(
            [1, 2, np.nan, 4, np.nan], index=list("ABCDE"), name="x"
        )

        assert isinstance(res1, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res1, exp1)
        assert isinstance(res2, tm.SubclassedSeries)
        tm.assert_series_equal(res2, exp2)

        # series + frame
        res1, res2 = s.align(df)
        assert isinstance(res1, tm.SubclassedSeries)
        tm.assert_series_equal(res1, exp2)
        assert isinstance(res2, tm.SubclassedDataFrame)
        tm.assert_frame_equal(res2, exp1)

    def test_subclass_iterrows(self):
        # GH 13977
        df = tm.SubclassedDataFrame({"a": [1]})
        for i, row in df.iterrows():
            assert isinstance(row, tm.SubclassedSeries)
            tm.assert_series_equal(row, df.loc[i])

    def test_subclass_stack(self):
        # GH 15564
        df = tm.SubclassedDataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            index=["a", "b", "c"],
            columns=["X", "Y", "Z"],
        )

        res = df.stack(future_stack=True)
        exp = tm.SubclassedSeries(
            [1, 2, 3, 4, 5, 6, 7, 8, 9], index=[list("aaabbbccc"), list("XYZXYZXYZ")]
        )

        tm.assert_series_equal(res, exp)

    def test_subclass_stack_multi(self):
        # GH 15564
        df = tm.SubclassedDataFrame(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        exp = tm.SubclassedDataFrame(
            [
                [10, 12],
                [11, 13],
                [20, 22],
                [21, 23],
                [30, 32],
                [31, 33],
                [40, 42],
                [41, 43],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("yzyzyzyz"))),
                names=["aaa", "ccc", "yyy"],
            ),
            columns=Index(["W", "X"], name="www"),
        )

        res = df.stack(future_stack=True)
        tm.assert_frame_equal(res, exp)

        res = df.stack("yyy", future_stack=True)
        tm.assert_frame_equal(res, exp)

        exp = tm.SubclassedDataFrame(
            [
                [10, 11],
                [12, 13],
                [20, 21],
                [22, 23],
                [30, 31],
                [32, 33],
                [40, 41],
                [42, 43],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("WXWXWXWX"))),
                names=["aaa", "ccc", "www"],
            ),
            columns=Index(["y", "z"], name="yyy"),
        )

        res = df.stack("www", future_stack=True)
        tm.assert_frame_equal(res, exp)

    def test_subclass_stack_multi_mixed(self):
        # GH 15564
        df = tm.SubclassedDataFrame(
            [
                [10, 11, 12.0, 13.0],
                [20, 21, 22.0, 23.0],
                [30, 31, 32.0, 33.0],
                [40, 41, 42.0, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        exp = tm.SubclassedDataFrame(
            [
                [10, 12.0],
                [11, 13.0],
                [20, 22.0],
                [21, 23.0],
                [30, 32.0],
                [31, 33.0],
                [40, 42.0],
                [41, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("yzyzyzyz"))),
                names=["aaa", "ccc", "yyy"],
            ),
            columns=Index(["W", "X"], name="www"),
        )

        res = df.stack(future_stack=True)
        tm.assert_frame_equal(res, exp)

        res = df.stack("yyy", future_stack=True)
        tm.assert_frame_equal(res, exp)

        exp = tm.SubclassedDataFrame(
            [
                [10.0, 11.0],
                [12.0, 13.0],
                [20.0, 21.0],
                [22.0, 23.0],
                [30.0, 31.0],
                [32.0, 33.0],
                [40.0, 41.0],
                [42.0, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AAAABBBB"), list("ccddccdd"), list("WXWXWXWX"))),
                names=["aaa", "ccc", "www"],
            ),
            columns=Index(["y", "z"], name="yyy"),
        )

        res = df.stack("www", future_stack=True)
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack(self):
        # GH 15564
        df = tm.SubclassedDataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            index=["a", "b", "c"],
            columns=["X", "Y", "Z"],
        )

        res = df.unstack()
        exp = tm.SubclassedSeries(
            [1, 4, 7, 2, 5, 8, 3, 6, 9], index=[list("XXXYYYZZZ"), list("abcabcabc")]
        )

        tm.assert_series_equal(res, exp)

    def test_subclass_unstack_multi(self):
        # GH 15564
        df = tm.SubclassedDataFrame(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        exp = tm.SubclassedDataFrame(
            [[10, 20, 11, 21, 12, 22, 13, 23], [30, 40, 31, 41, 32, 42, 33, 43]],
            index=Index(["A", "B"], name="aaa"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("cdcdcdcd"))),
                names=["www", "yyy", "ccc"],
            ),
        )

        res = df.unstack()
        tm.assert_frame_equal(res, exp)

        res = df.unstack("ccc")
        tm.assert_frame_equal(res, exp)

        exp = tm.SubclassedDataFrame(
            [[10, 30, 11, 31, 12, 32, 13, 33], [20, 40, 21, 41, 22, 42, 23, 43]],
            index=Index(["c", "d"], name="ccc"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("ABABABAB"))),
                names=["www", "yyy", "aaa"],
            ),
        )

        res = df.unstack("aaa")
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack_multi_mixed(self):
        # GH 15564
        df = tm.SubclassedDataFrame(
            [
                [10, 11, 12.0, 13.0],
                [20, 21, 22.0, 23.0],
                [30, 31, 32.0, 33.0],
                [40, 41, 42.0, 43.0],
            ],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )

        exp = tm.SubclassedDataFrame(
            [
                [10, 20, 11, 21, 12.0, 22.0, 13.0, 23.0],
                [30, 40, 31, 41, 32.0, 42.0, 33.0, 43.0],
            ],
            index=Index(["A", "B"], name="aaa"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("cdcdcdcd"))),
                names=["www", "yyy", "ccc"],
            ),
        )

        res = df.unstack()
        tm.assert_frame_equal(res, exp)

        res = df.unstack("ccc")
        tm.assert_frame_equal(res, exp)

        exp = tm.SubclassedDataFrame(
            [
                [10, 30, 11, 31, 12.0, 32.0, 13.0, 33.0],
                [20, 40, 21, 41, 22.0, 42.0, 23.0, 43.0],
            ],
            index=Index(["c", "d"], name="ccc"),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWWWXXXX"), list("yyzzyyzz"), list("ABABABAB"))),
                names=["www", "yyy", "aaa"],
            ),
        )

        res = df.unstack("aaa")
        tm.assert_frame_equal(res, exp)

    def test_subclass_pivot(self):
        # GH 15564
        df = tm.SubclassedDataFrame(
            {
                "index": ["A", "B", "C", "C", "B", "A"],
                "columns": ["One", "One", "One", "Two", "Two", "Two"],
                "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            }
        )

        pivoted = df.pivot(index="index", columns="columns", values="values")

        expected = tm.SubclassedDataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )

        expected.index.name, expected.columns.name = "index", "columns"

        tm.assert_frame_equal(pivoted, expected)

    def test_subclassed_melt(self):
        # GH 15564
        cheese = tm.SubclassedDataFrame(
            {
                "first": ["John", "Mary"],
                "last": ["Doe", "Bo"],
                "height": [5.5, 6.0],
                "weight": [130, 150],
            }
        )

        melted = pd.melt(cheese, id_vars=["first", "last"])

        expected = tm.SubclassedDataFrame(
            [
                ["John", "Doe", "height", 5.5],
                ["Mary", "Bo", "height", 6.0],
                ["John", "Doe", "weight", 130],
                ["Mary", "Bo", "weight", 150],
            ],
            columns=["first", "last", "variable", "value"],
        )

        tm.assert_frame_equal(melted, expected)

    def test_subclassed_wide_to_long(self):
        # GH 9762

        x = np.random.default_rng(2).standard_normal(3)
        df = tm.SubclassedDataFrame(
            {
                "A1970": {0: "a", 1: "b", 2: "c"},
                "A1980": {0: "d", 1: "e", 2: "f"},
                "B1970": {0: 2.5, 1: 1.2, 2: 0.7},
                "B1980": {0: 3.2, 1: 1.3, 2: 0.1},
                "X": dict(zip(range(3), x)),
            }
        )

        df["id"] = df.index
        exp_data = {
            "X": x.tolist() + x.tolist(),
            "A": ["a", "b", "c", "d", "e", "f"],
            "B": [2.5, 1.2, 0.7, 3.2, 1.3, 0.1],
            "year": [1970, 1970, 1970, 1980, 1980, 1980],
            "id": [0, 1, 2, 0, 1, 2],
        }
        expected = tm.SubclassedDataFrame(exp_data)
        expected = expected.set_index(["id", "year"])[["X", "A", "B"]]
        long_frame = pd.wide_to_long(df, ["A", "B"], i="id", j="year")

        tm.assert_frame_equal(long_frame, expected)

    def test_subclassed_apply(self):
        # GH 19822

        def check_row_subclass(row):
            assert isinstance(row, tm.SubclassedSeries)

        def stretch(row):
            if row["variable"] == "height":
                row["value"] += 0.5
            return row

        df = tm.SubclassedDataFrame(
            [
                ["John", "Doe", "height", 5.5],
                ["Mary", "Bo", "height", 6.0],
                ["John", "Doe", "weight", 130],
                ["Mary", "Bo", "weight", 150],
            ],
            columns=["first", "last", "variable", "value"],
        )

        df.apply(lambda x: check_row_subclass(x))
        df.apply(lambda x: check_row_subclass(x), axis=1)

        expected = tm.SubclassedDataFrame(
            [
                ["John", "Doe", "height", 6.0],
                ["Mary", "Bo", "height", 6.5],
                ["John", "Doe", "weight", 130],
                ["Mary", "Bo", "weight", 150],
            ],
            columns=["first", "last", "variable", "value"],
        )

        result = df.apply(lambda x: stretch(x), axis=1)
        assert isinstance(result, tm.SubclassedDataFrame)
        tm.assert_frame_equal(result, expected)

        expected = tm.SubclassedDataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

        result = df.apply(lambda x: tm.SubclassedSeries([1, 2, 3]), axis=1)
        assert isinstance(result, tm.SubclassedDataFrame)
        tm.assert_frame_equal(result, expected)

        result = df.apply(lambda x: [1, 2, 3], axis=1, result_type="expand")
        assert isinstance(result, tm.SubclassedDataFrame)
        tm.assert_frame_equal(result, expected)

        expected = tm.SubclassedSeries([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

        result = df.apply(lambda x: [1, 2, 3], axis=1)
        assert not isinstance(result, tm.SubclassedDataFrame)
        tm.assert_series_equal(result, expected)

    def test_subclassed_reductions(self, all_reductions):
        # GH 25596

        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = getattr(df, all_reductions)()
        assert isinstance(result, tm.SubclassedSeries)

    def test_subclassed_count(self):
        df = tm.SubclassedDataFrame(
            {
                "Person": ["John", "Myla", "Lewis", "John", "Myla"],
                "Age": [24.0, np.nan, 21.0, 33, 26],
                "Single": [False, True, True, True, False],
            }
        )
        result = df.count()
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame({"A": [1, 0, 3], "B": [0, 5, 6], "C": [7, 8, 0]})
        result = df.count()
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]],
            index=MultiIndex.from_tuples(
                list(zip(list("AABB"), list("cdcd"))), names=["aaa", "ccc"]
            ),
            columns=MultiIndex.from_tuples(
                list(zip(list("WWXX"), list("yzyz"))), names=["www", "yyy"]
            ),
        )
        result = df.count()
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame()
        result = df.count()
        assert isinstance(result, tm.SubclassedSeries)

    def test_isin(self):
        df = tm.SubclassedDataFrame(
            {"num_legs": [2, 4], "num_wings": [2, 0]}, index=["falcon", "dog"]
        )
        result = df.isin([0, 2])
        assert isinstance(result, tm.SubclassedDataFrame)

    def test_duplicated(self):
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.duplicated()
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame()
        result = df.duplicated()
        assert isinstance(result, tm.SubclassedSeries)

    @pytest.mark.parametrize("idx_method", ["idxmax", "idxmin"])
    def test_idx(self, idx_method):
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = getattr(df, idx_method)()
        assert isinstance(result, tm.SubclassedSeries)

    def test_dot(self):
        df = tm.SubclassedDataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        s = tm.SubclassedSeries([1, 1, 2, 1])
        result = df.dot(s)
        assert isinstance(result, tm.SubclassedSeries)

        df = tm.SubclassedDataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        s = tm.SubclassedDataFrame([1, 1, 2, 1])
        result = df.dot(s)
        assert isinstance(result, tm.SubclassedDataFrame)

    def test_memory_usage(self):
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.memory_usage()
        assert isinstance(result, tm.SubclassedSeries)

        result = df.memory_usage(index=False)
        assert isinstance(result, tm.SubclassedSeries)

    def test_corrwith(self):
        pytest.importorskip("scipy")
        index = ["a", "b", "c", "d", "e"]
        columns = ["one", "two", "three", "four"]
        df1 = tm.SubclassedDataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            index=index,
            columns=columns,
        )
        df2 = tm.SubclassedDataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=index[:4],
            columns=columns,
        )
        correls = df1.corrwith(df2, axis=1, drop=True, method="kendall")

        assert isinstance(correls, (tm.SubclassedSeries))

    def test_asof(self):
        N = 3
        rng = pd.date_range("1/1/1990", periods=N, freq="53s")
        df = tm.SubclassedDataFrame(
            {
                "A": [np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan],
                "C": [np.nan, np.nan, np.nan],
            },
            index=rng,
        )

        result = df.asof(rng[-2:])
        assert isinstance(result, tm.SubclassedDataFrame)

        result = df.asof(rng[-2])
        assert isinstance(result, tm.SubclassedSeries)

        result = df.asof("1989-12-31")
        assert isinstance(result, tm.SubclassedSeries)

    def test_idxmin_preserves_subclass(self):
        # GH 28330

        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.idxmin()
        assert isinstance(result, tm.SubclassedSeries)

    def test_idxmax_preserves_subclass(self):
        # GH 28330

        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.idxmax()
        assert isinstance(result, tm.SubclassedSeries)

    def test_convert_dtypes_preserves_subclass(self, gpd_style_subclass_df):
        # GH 43668
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = df.convert_dtypes()
        assert isinstance(result, tm.SubclassedDataFrame)

        result = gpd_style_subclass_df.convert_dtypes()
        assert isinstance(result, type(gpd_style_subclass_df))

    def test_astype_preserves_subclass(self):
        # GH#40810
        df = tm.SubclassedDataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

        result = df.astype({"A": np.int64, "B": np.int32, "C": np.float64})
        assert isinstance(result, tm.SubclassedDataFrame)

    def test_equals_subclass(self):
        # https://github.com/pandas-dev/pandas/pull/34402
        # allow subclass in both directions
        df1 = DataFrame({"a": [1, 2, 3]})
        df2 = tm.SubclassedDataFrame({"a": [1, 2, 3]})
        assert df1.equals(df2)
        assert df2.equals(df1)

    def test_replace_list_method(self):
        # https://github.com/pandas-dev/pandas/pull/46018
        df = tm.SubclassedDataFrame({"A": [0, 1, 2]})
        msg = "The 'method' keyword in SubclassedDataFrame.replace is deprecated"
        with tm.assert_produces_warning(
            FutureWarning, match=msg, raise_on_extra_warnings=False
        ):
            result = df.replace([1, 2], method="ffill")
        expected = tm.SubclassedDataFrame({"A": [0, 0, 0]})
        assert isinstance(result, tm.SubclassedDataFrame)
        tm.assert_frame_equal(result, expected)


class MySubclassWithMetadata(DataFrame):
    _metadata = ["my_metadata"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        my_metadata = kwargs.pop("my_metadata", None)
        if args and isinstance(args[0], MySubclassWithMetadata):
            my_metadata = args[0].my_metadata  # type: ignore[has-type]
        self.my_metadata = my_metadata

    @property
    def _constructor(self):
        return MySubclassWithMetadata


def test_constructor_with_metadata():
    # https://github.com/pandas-dev/pandas/pull/54922
    # https://github.com/pandas-dev/pandas/issues/55120
    df = MySubclassWithMetadata(
        np.random.default_rng(2).random((5, 3)), columns=["A", "B", "C"]
    )
    subset = df[["A", "B"]]
    assert isinstance(subset, MySubclassWithMetadata)


class SimpleDataFrameSubClass(DataFrame):
    """A subclass of DataFrame that does not define a constructor."""


class SimpleSeriesSubClass(Series):
    """A subclass of Series that does not define a constructor."""


class TestSubclassWithoutConstructor:
    def test_copy_df(self):
        expected = DataFrame({"a": [1, 2, 3]})
        result = SimpleDataFrameSubClass(expected).copy()

        assert (
            type(result) is DataFrame
        )  # assert_frame_equal only checks isinstance(lhs, type(rhs))
        tm.assert_frame_equal(result, expected)

    def test_copy_series(self):
        expected = Series([1, 2, 3])
        result = SimpleSeriesSubClass(expected).copy()

        tm.assert_series_equal(result, expected)

    def test_series_to_frame(self):
        orig = Series([1, 2, 3])
        expected = orig.to_frame()
        result = SimpleSeriesSubClass(orig).to_frame()

        assert (
            type(result) is DataFrame
        )  # assert_frame_equal only checks isinstance(lhs, type(rhs))
        tm.assert_frame_equal(result, expected)

    def test_groupby(self):
        df = SimpleDataFrameSubClass(DataFrame({"a": [1, 2, 3]}))

        for _, v in df.groupby("a"):
            assert type(v) is DataFrame

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_coset_table.py ===
from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.coset_table import (CosetTable,
                    coset_enumeration_r, coset_enumeration_c)
from sympy.combinatorics.coset_table import modified_coset_enumeration_r
from sympy.combinatorics.free_groups import free_group

from sympy.testing.pytest import slow

"""
References
==========

[1] Holt, D., Eick, B., O'Brien, E.
"Handbook of Computational Group Theory"

[2] John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490.
"Implementation and Analysis of the Todd-Coxeter Algorithm"

"""

def test_scan_1():
    # Example 5.1 from [1]
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    c = CosetTable(f, [x])

    c.scan_and_fill(0, x)
    assert c.table == [[0, 0, None, None]]
    assert c.p == [0]
    assert c.n == 1
    assert c.omega == [0]

    c.scan_and_fill(0, x**3)
    assert c.table == [[0, 0, None, None]]
    assert c.p == [0]
    assert c.n == 1
    assert c.omega == [0]

    c.scan_and_fill(0, y**3)
    assert c.table == [[0, 0, 1, 2], [None, None, 2, 0], [None, None, 0, 1]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    c.scan_and_fill(0, x**-1*y**-1*x*y)
    assert c.table == [[0, 0, 1, 2], [None, None, 2, 0], [2, 2, 0, 1]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    c.scan_and_fill(1, x**3)
    assert c.table == [[0, 0, 1, 2], [3, 4, 2, 0], [2, 2, 0, 1], \
            [4, 1, None, None], [1, 3, None, None]]
    assert c.p == [0, 1, 2, 3, 4]
    assert c.n == 5
    assert c.omega == [0, 1, 2, 3, 4]

    c.scan_and_fill(1, y**3)
    assert c.table == [[0, 0, 1, 2], [3, 4, 2, 0], [2, 2, 0, 1], \
            [4, 1, None, None], [1, 3, None, None]]
    assert c.p == [0, 1, 2, 3, 4]
    assert c.n == 5
    assert c.omega == [0, 1, 2, 3, 4]

    c.scan_and_fill(1, x**-1*y**-1*x*y)
    assert c.table == [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1], \
            [None, 1, None, None], [1, 3, None, None]]
    assert c.p == [0, 1, 2, 1, 1]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    # Example 5.2 from [1]
    f = FpGroup(F, [x**2, y**3, (x*y)**3])
    c = CosetTable(f, [x*y])

    c.scan_and_fill(0, x*y)
    assert c.table == [[1, None, None, 1], [None, 0, 0, None]]
    assert c.p == [0, 1]
    assert c.n == 2
    assert c.omega == [0, 1]

    c.scan_and_fill(0, x**2)
    assert c.table == [[1, 1, None, 1], [0, 0, 0, None]]
    assert c.p == [0, 1]
    assert c.n == 2
    assert c.omega == [0, 1]

    c.scan_and_fill(0, y**3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    c.scan_and_fill(0, (x*y)**3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    c.scan_and_fill(1, x**2)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    c.scan_and_fill(1, y**3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]

    c.scan_and_fill(1, (x*y)**3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [3, 4, 1, 0], [None, 2, 4, None], [2, None, None, 3]]
    assert c.p == [0, 1, 2, 3, 4]
    assert c.n == 5
    assert c.omega == [0, 1, 2, 3, 4]

    c.scan_and_fill(2, x**2)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [3, 3, 1, 0], [2, 2, 3, 3], [2, None, None, 3]]
    assert c.p == [0, 1, 2, 3, 3]
    assert c.n == 4
    assert c.omega == [0, 1, 2, 3]


@slow
def test_coset_enumeration():
    # this test function contains the combined tests for the two strategies
    # i.e. HLT and Felsch strategies.

    # Example 5.1 from [1]
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    C_r = coset_enumeration_r(f, [x])
    C_r.compress(); C_r.standardize()
    C_c = coset_enumeration_c(f, [x])
    C_c.compress(); C_c.standardize()
    table1 = [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]
    assert C_r.table == table1
    assert C_c.table == table1

    # E1 from [2] Pg. 474
    F, r, s, t = free_group("r, s, t")
    E1 = FpGroup(F, [t**-1*r*t*r**-2, r**-1*s*r*s**-2, s**-1*t*s*t**-2])
    C_r = coset_enumeration_r(E1, [])
    C_r.compress()
    C_c = coset_enumeration_c(E1, [])
    C_c.compress()
    table2 = [[0, 0, 0, 0, 0, 0]]
    assert C_r.table == table2
    # test for issue #11449
    assert C_c.table == table2

    # Cox group from [2] Pg. 474
    F, a, b = free_group("a, b")
    Cox = FpGroup(F, [a**6, b**6, (a*b)**2, (a**2*b**2)**2, (a**3*b**3)**5])
    C_r = coset_enumeration_r(Cox, [a])
    C_r.compress(); C_r.standardize()
    C_c = coset_enumeration_c(Cox, [a])
    C_c.compress(); C_c.standardize()
    table3 = [[0, 0, 1, 2],
             [2, 3, 4, 0],
             [5, 1, 0, 6],
             [1, 7, 8, 9],
             [9, 10, 11, 1],
             [12, 2, 9, 13],
             [14, 9, 2, 11],
             [3, 12, 15, 16],
             [16, 17, 18, 3],
             [6, 4, 3, 5],
             [4, 19, 20, 21],
             [21, 22, 6, 4],
             [7, 5, 23, 24],
             [25, 23, 5, 18],
             [19, 6, 22, 26],
             [24, 27, 28, 7],
             [29, 8, 7, 30],
             [8, 31, 32, 33],
             [33, 34, 13, 8],
             [10, 14, 35, 35],
             [35, 36, 37, 10],
             [30, 11, 10, 29],
             [11, 38, 39, 14],
             [13, 39, 38, 12],
             [40, 15, 12, 41],
             [42, 13, 34, 43],
             [44, 35, 14, 45],
             [15, 46, 47, 34],
             [34, 48, 49, 15],
             [50, 16, 21, 51],
             [52, 21, 16, 49],
             [17, 50, 53, 54],
             [54, 55, 56, 17],
             [41, 18, 17, 40],
             [18, 28, 27, 25],
             [26, 20, 19, 19],
             [20, 57, 58, 59],
             [59, 60, 51, 20],
             [22, 52, 61, 23],
             [23, 62, 63, 22],
             [64, 24, 33, 65],
             [48, 33, 24, 61],
             [62, 25, 54, 66],
             [67, 54, 25, 68],
             [57, 26, 59, 69],
             [70, 59, 26, 63],
             [27, 64, 71, 72],
             [72, 73, 68, 27],
             [28, 41, 74, 75],
             [75, 76, 30, 28],
             [31, 29, 77, 78],
             [79, 77, 29, 37],
             [38, 30, 76, 80],
             [78, 81, 82, 31],
             [43, 32, 31, 42],
             [32, 83, 84, 85],
             [85, 86, 65, 32],
             [36, 44, 87, 88],
             [88, 89, 90, 36],
             [45, 37, 36, 44],
             [37, 82, 81, 79],
             [80, 74, 41, 38],
             [39, 42, 91, 92],
             [92, 93, 45, 39],
             [46, 40, 94, 95],
             [96, 94, 40, 56],
             [97, 91, 42, 82],
             [83, 43, 98, 99],
             [100, 98, 43, 47],
             [101, 87, 44, 90],
             [82, 45, 93, 97],
             [95, 102, 103, 46],
             [104, 47, 46, 105],
             [47, 106, 107, 100],
             [61, 108, 109, 48],
             [105, 49, 48, 104],
             [49, 110, 111, 52],
             [51, 111, 110, 50],
             [112, 53, 50, 113],
             [114, 51, 60, 115],
             [116, 61, 52, 117],
             [53, 118, 119, 60],
             [60, 70, 66, 53],
             [55, 67, 120, 121],
             [121, 122, 123, 55],
             [113, 56, 55, 112],
             [56, 103, 102, 96],
             [69, 124, 125, 57],
             [115, 58, 57, 114],
             [58, 126, 127, 128],
             [128, 128, 69, 58],
             [66, 129, 130, 62],
             [117, 63, 62, 116],
             [63, 125, 124, 70],
             [65, 109, 108, 64],
             [131, 71, 64, 132],
             [133, 65, 86, 134],
             [135, 66, 70, 136],
             [68, 130, 129, 67],
             [137, 120, 67, 138],
             [132, 68, 73, 131],
             [139, 69, 128, 140],
             [71, 141, 142, 86],
             [86, 143, 144, 71],
             [145, 72, 75, 146],
             [147, 75, 72, 144],
             [73, 145, 148, 120],
             [120, 149, 150, 73],
             [74, 151, 152, 94],
             [94, 153, 146, 74],
             [76, 147, 154, 77],
             [77, 155, 156, 76],
             [157, 78, 85, 158],
             [143, 85, 78, 154],
             [155, 79, 88, 159],
             [160, 88, 79, 161],
             [151, 80, 92, 162],
             [163, 92, 80, 156],
             [81, 157, 164, 165],
             [165, 166, 161, 81],
             [99, 107, 106, 83],
             [134, 84, 83, 133],
             [84, 167, 168, 169],
             [169, 170, 158, 84],
             [87, 171, 172, 93],
             [93, 163, 159, 87],
             [89, 160, 173, 174],
             [174, 175, 176, 89],
             [90, 90, 89, 101],
             [91, 177, 178, 98],
             [98, 179, 162, 91],
             [180, 95, 100, 181],
             [179, 100, 95, 152],
             [153, 96, 121, 148],
             [182, 121, 96, 183],
             [177, 97, 165, 184],
             [185, 165, 97, 172],
             [186, 99, 169, 187],
             [188, 169, 99, 178],
             [171, 101, 174, 189],
             [190, 174, 101, 176],
             [102, 180, 191, 192],
             [192, 193, 183, 102],
             [103, 113, 194, 195],
             [195, 196, 105, 103],
             [106, 104, 197, 198],
             [199, 197, 104, 109],
             [110, 105, 196, 200],
             [198, 201, 133, 106],
             [107, 186, 202, 203],
             [203, 204, 181, 107],
             [108, 116, 205, 206],
             [206, 207, 132, 108],
             [109, 133, 201, 199],
             [200, 194, 113, 110],
             [111, 114, 208, 209],
             [209, 210, 117, 111],
             [118, 112, 211, 212],
             [213, 211, 112, 123],
             [214, 208, 114, 125],
             [126, 115, 215, 216],
             [217, 215, 115, 119],
             [218, 205, 116, 130],
             [125, 117, 210, 214],
             [212, 219, 220, 118],
             [136, 119, 118, 135],
             [119, 221, 222, 217],
             [122, 182, 223, 224],
             [224, 225, 226, 122],
             [138, 123, 122, 137],
             [123, 220, 219, 213],
             [124, 139, 227, 228],
             [228, 229, 136, 124],
             [216, 222, 221, 126],
             [140, 127, 126, 139],
             [127, 230, 231, 232],
             [232, 233, 140, 127],
             [129, 135, 234, 235],
             [235, 236, 138, 129],
             [130, 132, 207, 218],
             [141, 131, 237, 238],
             [239, 237, 131, 150],
             [167, 134, 240, 241],
             [242, 240, 134, 142],
             [243, 234, 135, 220],
             [221, 136, 229, 244],
             [149, 137, 245, 246],
             [247, 245, 137, 226],
             [220, 138, 236, 243],
             [244, 227, 139, 221],
             [230, 140, 233, 248],
             [238, 249, 250, 141],
             [251, 142, 141, 252],
             [142, 253, 254, 242],
             [154, 255, 256, 143],
             [252, 144, 143, 251],
             [144, 257, 258, 147],
             [146, 258, 257, 145],
             [259, 148, 145, 260],
             [261, 146, 153, 262],
             [263, 154, 147, 264],
             [148, 265, 266, 153],
             [246, 267, 268, 149],
             [260, 150, 149, 259],
             [150, 250, 249, 239],
             [162, 269, 270, 151],
             [262, 152, 151, 261],
             [152, 271, 272, 179],
             [159, 273, 274, 155],
             [264, 156, 155, 263],
             [156, 270, 269, 163],
             [158, 256, 255, 157],
             [275, 164, 157, 276],
             [277, 158, 170, 278],
             [279, 159, 163, 280],
             [161, 274, 273, 160],
             [281, 173, 160, 282],
             [276, 161, 166, 275],
             [283, 162, 179, 284],
             [164, 285, 286, 170],
             [170, 188, 184, 164],
             [166, 185, 189, 173],
             [173, 287, 288, 166],
             [241, 254, 253, 167],
             [278, 168, 167, 277],
             [168, 289, 290, 291],
             [291, 292, 187, 168],
             [189, 293, 294, 171],
             [280, 172, 171, 279],
             [172, 295, 296, 185],
             [175, 190, 297, 297],
             [297, 298, 299, 175],
             [282, 176, 175, 281],
             [176, 294, 293, 190],
             [184, 296, 295, 177],
             [284, 178, 177, 283],
             [178, 300, 301, 188],
             [181, 272, 271, 180],
             [302, 191, 180, 303],
             [304, 181, 204, 305],
             [183, 266, 265, 182],
             [306, 223, 182, 307],
             [303, 183, 193, 302],
             [308, 184, 188, 309],
             [310, 189, 185, 311],
             [187, 301, 300, 186],
             [305, 202, 186, 304],
             [312, 187, 292, 313],
             [314, 297, 190, 315],
             [191, 316, 317, 204],
             [204, 318, 319, 191],
             [320, 192, 195, 321],
             [322, 195, 192, 319],
             [193, 320, 323, 223],
             [223, 324, 325, 193],
             [194, 326, 327, 211],
             [211, 328, 321, 194],
             [196, 322, 329, 197],
             [197, 330, 331, 196],
             [332, 198, 203, 333],
             [318, 203, 198, 329],
             [330, 199, 206, 334],
             [335, 206, 199, 336],
             [326, 200, 209, 337],
             [338, 209, 200, 331],
             [201, 332, 339, 240],
             [240, 340, 336, 201],
             [202, 341, 342, 292],
             [292, 343, 333, 202],
             [205, 344, 345, 210],
             [210, 338, 334, 205],
             [207, 335, 346, 237],
             [237, 347, 348, 207],
             [208, 349, 350, 215],
             [215, 351, 337, 208],
             [352, 212, 217, 353],
             [351, 217, 212, 327],
             [328, 213, 224, 323],
             [354, 224, 213, 355],
             [349, 214, 228, 356],
             [357, 228, 214, 345],
             [358, 216, 232, 359],
             [360, 232, 216, 350],
             [344, 218, 235, 361],
             [362, 235, 218, 348],
             [219, 352, 363, 364],
             [364, 365, 355, 219],
             [222, 358, 366, 367],
             [367, 368, 353, 222],
             [225, 354, 369, 370],
             [370, 371, 372, 225],
             [307, 226, 225, 306],
             [226, 268, 267, 247],
             [227, 373, 374, 233],
             [233, 360, 356, 227],
             [229, 357, 361, 234],
             [234, 375, 376, 229],
             [248, 231, 230, 230],
             [231, 377, 378, 379],
             [379, 380, 359, 231],
             [236, 362, 381, 245],
             [245, 382, 383, 236],
             [384, 238, 242, 385],
             [340, 242, 238, 346],
             [347, 239, 246, 381],
             [386, 246, 239, 387],
             [388, 241, 291, 389],
             [343, 291, 241, 339],
             [375, 243, 364, 390],
             [391, 364, 243, 383],
             [373, 244, 367, 392],
             [393, 367, 244, 376],
             [382, 247, 370, 394],
             [395, 370, 247, 396],
             [377, 248, 379, 397],
             [398, 379, 248, 374],
             [249, 384, 399, 400],
             [400, 401, 387, 249],
             [250, 260, 402, 403],
             [403, 404, 252, 250],
             [253, 251, 405, 406],
             [407, 405, 251, 256],
             [257, 252, 404, 408],
             [406, 409, 277, 253],
             [254, 388, 410, 411],
             [411, 412, 385, 254],
             [255, 263, 413, 414],
             [414, 415, 276, 255],
             [256, 277, 409, 407],
             [408, 402, 260, 257],
             [258, 261, 416, 417],
             [417, 418, 264, 258],
             [265, 259, 419, 420],
             [421, 419, 259, 268],
             [422, 416, 261, 270],
             [271, 262, 423, 424],
             [425, 423, 262, 266],
             [426, 413, 263, 274],
             [270, 264, 418, 422],
             [420, 427, 307, 265],
             [266, 303, 428, 425],
             [267, 386, 429, 430],
             [430, 431, 396, 267],
             [268, 307, 427, 421],
             [269, 283, 432, 433],
             [433, 434, 280, 269],
             [424, 428, 303, 271],
             [272, 304, 435, 436],
             [436, 437, 284, 272],
             [273, 279, 438, 439],
             [439, 440, 282, 273],
             [274, 276, 415, 426],
             [285, 275, 441, 442],
             [443, 441, 275, 288],
             [289, 278, 444, 445],
             [446, 444, 278, 286],
             [447, 438, 279, 294],
             [295, 280, 434, 448],
             [287, 281, 449, 450],
             [451, 449, 281, 299],
             [294, 282, 440, 447],
             [448, 432, 283, 295],
             [300, 284, 437, 452],
             [442, 453, 454, 285],
             [309, 286, 285, 308],
             [286, 455, 456, 446],
             [450, 457, 458, 287],
             [311, 288, 287, 310],
             [288, 454, 453, 443],
             [445, 456, 455, 289],
             [313, 290, 289, 312],
             [290, 459, 460, 461],
             [461, 462, 389, 290],
             [293, 310, 463, 464],
             [464, 465, 315, 293],
             [296, 308, 466, 467],
             [467, 468, 311, 296],
             [298, 314, 469, 470],
             [470, 471, 472, 298],
             [315, 299, 298, 314],
             [299, 458, 457, 451],
             [452, 435, 304, 300],
             [301, 312, 473, 474],
             [474, 475, 309, 301],
             [316, 302, 476, 477],
             [478, 476, 302, 325],
             [341, 305, 479, 480],
             [481, 479, 305, 317],
             [324, 306, 482, 483],
             [484, 482, 306, 372],
             [485, 466, 308, 454],
             [455, 309, 475, 486],
             [487, 463, 310, 458],
             [454, 311, 468, 485],
             [486, 473, 312, 455],
             [459, 313, 488, 489],
             [490, 488, 313, 342],
             [491, 469, 314, 472],
             [458, 315, 465, 487],
             [477, 492, 485, 316],
             [463, 317, 316, 468],
             [317, 487, 493, 481],
             [329, 447, 464, 318],
             [468, 319, 318, 463],
             [319, 467, 448, 322],
             [321, 448, 467, 320],
             [475, 323, 320, 466],
             [432, 321, 328, 437],
             [438, 329, 322, 434],
             [323, 474, 452, 328],
             [483, 494, 486, 324],
             [466, 325, 324, 475],
             [325, 485, 492, 478],
             [337, 422, 433, 326],
             [437, 327, 326, 432],
             [327, 436, 424, 351],
             [334, 426, 439, 330],
             [434, 331, 330, 438],
             [331, 433, 422, 338],
             [333, 464, 447, 332],
             [449, 339, 332, 440],
             [465, 333, 343, 469],
             [413, 334, 338, 418],
             [336, 439, 426, 335],
             [441, 346, 335, 415],
             [440, 336, 340, 449],
             [416, 337, 351, 423],
             [339, 451, 470, 343],
             [346, 443, 450, 340],
             [480, 493, 487, 341],
             [469, 342, 341, 465],
             [342, 491, 495, 490],
             [361, 407, 414, 344],
             [418, 345, 344, 413],
             [345, 417, 408, 357],
             [381, 446, 442, 347],
             [415, 348, 347, 441],
             [348, 414, 407, 362],
             [356, 408, 417, 349],
             [423, 350, 349, 416],
             [350, 425, 420, 360],
             [353, 424, 436, 352],
             [479, 363, 352, 435],
             [428, 353, 368, 476],
             [355, 452, 474, 354],
             [488, 369, 354, 473],
             [435, 355, 365, 479],
             [402, 356, 360, 419],
             [405, 361, 357, 404],
             [359, 420, 425, 358],
             [476, 366, 358, 428],
             [427, 359, 380, 482],
             [444, 381, 362, 409],
             [363, 481, 477, 368],
             [368, 393, 390, 363],
             [365, 391, 394, 369],
             [369, 490, 480, 365],
             [366, 478, 483, 380],
             [380, 398, 392, 366],
             [371, 395, 496, 497],
             [497, 498, 489, 371],
             [473, 372, 371, 488],
             [372, 486, 494, 484],
             [392, 400, 403, 373],
             [419, 374, 373, 402],
             [374, 421, 430, 398],
             [390, 411, 406, 375],
             [404, 376, 375, 405],
             [376, 403, 400, 393],
             [397, 430, 421, 377],
             [482, 378, 377, 427],
             [378, 484, 497, 499],
             [499, 499, 397, 378],
             [394, 461, 445, 382],
             [409, 383, 382, 444],
             [383, 406, 411, 391],
             [385, 450, 443, 384],
             [492, 399, 384, 453],
             [457, 385, 412, 493],
             [387, 442, 446, 386],
             [494, 429, 386, 456],
             [453, 387, 401, 492],
             [389, 470, 451, 388],
             [493, 410, 388, 457],
             [471, 389, 462, 495],
             [412, 390, 393, 399],
             [462, 394, 391, 410],
             [401, 392, 398, 429],
             [396, 445, 461, 395],
             [498, 496, 395, 460],
             [456, 396, 431, 494],
             [431, 397, 499, 496],
             [399, 477, 481, 412],
             [429, 483, 478, 401],
             [410, 480, 490, 462],
             [496, 497, 484, 431],
             [489, 495, 491, 459],
             [495, 460, 459, 471],
             [460, 489, 498, 498],
             [472, 472, 471, 491]]

    assert C_r.table == table3
    assert C_c.table == table3

    # Group denoted by B2,4 from [2] Pg. 474
    F, a, b = free_group("a, b")
    B_2_4 = FpGroup(F, [a**4, b**4, (a*b)**4, (a**-1*b)**4, (a**2*b)**4, \
            (a*b**2)**4, (a**2*b**2)**4, (a**-1*b*a*b)**4, (a*b**-1*a*b)**4])
    C_r = coset_enumeration_r(B_2_4, [a])
    C_c = coset_enumeration_c(B_2_4, [a])
    index_r = 0
    for i in range(len(C_r.p)):
        if C_r.p[i] == i:
            index_r += 1
    assert index_r == 1024

    index_c = 0
    for i in range(len(C_c.p)):
        if C_c.p[i] == i:
            index_c += 1
    assert index_c == 1024

    # trivial Macdonald group G(2,2) from [2] Pg. 480
    M = FpGroup(F, [b**-1*a**-1*b*a*b**-1*a*b*a**-2, a**-1*b**-1*a*b*a**-1*b*a*b**-2])
    C_r = coset_enumeration_r(M, [a])
    C_r.compress(); C_r.standardize()
    C_c = coset_enumeration_c(M, [a])
    C_c.compress(); C_c.standardize()
    table4 = [[0, 0, 0, 0]]
    assert C_r.table == table4
    assert C_c.table == table4


def test_look_ahead():
    # Section 3.2 [Test Example] Example (d) from [2]
    F, a, b, c = free_group("a, b, c")
    f = FpGroup(F, [a**11, b**5, c**4, (a*c)**3, b**2*c**-1*b**-1*c, a**4*b**-1*a**-1*b])
    H = [c, b, c**2]
    table0 = [[1, 2, 0, 0, 0, 0],
              [3, 0, 4, 5, 6, 7],
              [0, 8, 9, 10, 11, 12],
              [5, 1, 10, 13, 14, 15],
              [16, 5, 16, 1, 17, 18],
              [4, 3, 1, 8, 19, 20],
              [12, 21, 22, 23, 24, 1],
              [25, 26, 27, 28, 1, 24],
              [2, 10, 5, 16, 22, 28],
              [10, 13, 13, 2, 29, 30]]
    CosetTable.max_stack_size = 10
    C_c = coset_enumeration_c(f, H)
    C_c.compress(); C_c.standardize()
    assert C_c.table[: 10] == table0

def test_modified_methods():
    '''
    Tests for modified coset table methods.
    Example 5.7 from [1] Holt, D., Eick, B., O'Brien
    "Handbook of Computational Group Theory".

    '''
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x**3, y**5, (x*y)**2])
    H = [x*y, x**-1*y**-1*x*y*x]
    C = CosetTable(f, H)
    C.modified_define(0, x)
    identity = C._grp.identity
    a_0 = C._grp.generators[0]
    a_1 = C._grp.generators[1]

    assert C.P == [[identity, None, None, None],
                    [None, identity, None, None]]
    assert C.table == [[1, None, None, None],
                        [None, 0, None, None]]

    C.modified_define(1, x)
    assert C.table == [[1, None, None, None],
                        [2, 0, None, None],
                        [None, 1, None, None]]
    assert C.P == [[identity, None, None, None],
                    [identity, identity, None, None],
                    [None, identity, None, None]]

    C.modified_scan(0, x**3, C._grp.identity, fill=False)
    assert C.P == [[identity, identity, None, None],
                     [identity, identity, None, None],
                     [identity, identity, None, None]]
    assert C.table == [[1, 2, None, None],
                        [2, 0, None, None],
                        [0, 1, None, None]]

    C.modified_scan(0, x*y, C._grp.generators[0], fill=False)
    assert C.P == [[identity, identity, None, a_0**-1],
                    [identity, identity, a_0, None],
                    [identity, identity, None, None]]
    assert C.table == [[1, 2, None, 1],
                        [2, 0, 0, None],
                        [0, 1, None, None]]

    C.modified_define(2, y**-1)
    assert C.table == [[1, 2, None, 1],
                        [2, 0, 0, None],
                        [0, 1, None, 3],
                        [None, None, 2, None]]
    assert C.P == [[identity, identity, None, a_0**-1],
                    [identity, identity, a_0, None],
                    [identity, identity, None, identity],
                    [None, None, identity, None]]

    C.modified_scan(0, x**-1*y**-1*x*y*x, C._grp.generators[1])
    assert C.table == [[1, 2, None, 1],
                        [2, 0, 0, None],
                        [0, 1, None, 3],
                        [3, 3, 2, None]]
    assert C.P == [[identity, identity, None, a_0**-1],
                    [identity, identity, a_0, None],
                    [identity, identity, None, identity],
                    [a_1, a_1**-1, identity, None]]

    C.modified_scan(2, (x*y)**2, C._grp.identity)
    assert C.table == [[1, 2, 3, 1],
                        [2, 0, 0, None],
                        [0, 1, None, 3],
                        [3, 3, 2, 0]]
    assert C.P == [[identity, identity, a_1**-1, a_0**-1],
                    [identity, identity, a_0, None],
                    [identity, identity, None, identity],
                    [a_1, a_1**-1, identity, a_1]]

    C.modified_define(2, y)
    assert C.table == [[1, 2, 3, 1],
                        [2, 0, 0, None],
                        [0, 1, 4, 3],
                        [3, 3, 2, 0],
                        [None, None, None, 2]]
    assert C.P == [[identity, identity, a_1**-1, a_0**-1],
                    [identity, identity, a_0, None],
                    [identity, identity, identity, identity],
                    [a_1, a_1**-1, identity, a_1],
                    [None, None, None, identity]]

    C.modified_scan(0, y**5, C._grp.identity)
    assert C.table == [[1, 2, 3, 1], [2, 0, 0, 4], [0, 1, 4, 3], [3, 3, 2, 0], [None, None, 1, 2]]
    assert C.P == [[identity, identity, a_1**-1, a_0**-1],
                    [identity, identity, a_0, a_0*a_1**-1],
                    [identity, identity, identity, identity],
                    [a_1, a_1**-1, identity, a_1],
                    [None, None, a_1*a_0**-1, identity]]

    C.modified_scan(1, (x*y)**2, C._grp.identity)
    assert C.table == [[1, 2, 3, 1],
                        [2, 0, 0, 4],
                        [0, 1, 4, 3],
                        [3, 3, 2, 0],
                        [4, 4, 1, 2]]
    assert C.P == [[identity, identity, a_1**-1, a_0**-1],
                    [identity, identity, a_0, a_0*a_1**-1],
                    [identity, identity, identity, identity],
                    [a_1, a_1**-1, identity, a_1],
                    [a_0*a_1**-1, a_1*a_0**-1, a_1*a_0**-1, identity]]

    # Modified coset enumeration test
    f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    C = coset_enumeration_r(f, [x])
    C_m = modified_coset_enumeration_r(f, [x])
    assert C_m.table == C.table

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_rootisolation.py ===
"""Tests for real and complex root isolation and refinement algorithms. """

from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
    dup_cauchy_upper_bound, dup_cauchy_lower_bound,
    dup_mignotte_sep_bound_squared,
)
from sympy.testing.pytest import raises

def test_dup_sturm():
    R, x = ring("x", QQ)

    assert R.dup_sturm(5) == [1]
    assert R.dup_sturm(x) == [x, 1]

    f = x**3 - 2*x**2 + 3*x - 5
    assert R.dup_sturm(f) == [f, 3*x**2 - 4*x + 3, -QQ(10,9)*x + QQ(13,3), -QQ(3303,100)]


def test_dup_cauchy_upper_bound():
    raises(PolynomialError, lambda: dup_cauchy_upper_bound([], QQ))
    raises(PolynomialError, lambda: dup_cauchy_upper_bound([QQ(1)], QQ))
    raises(DomainError, lambda: dup_cauchy_upper_bound([ZZ_I(1), ZZ_I(1)], ZZ_I))

    assert dup_cauchy_upper_bound([QQ(1), QQ(0), QQ(0)], QQ) == QQ.zero
    assert dup_cauchy_upper_bound([QQ(1), QQ(0), QQ(-2)], QQ) == QQ(3)


def test_dup_cauchy_lower_bound():
    raises(PolynomialError, lambda: dup_cauchy_lower_bound([], QQ))
    raises(PolynomialError, lambda: dup_cauchy_lower_bound([QQ(1)], QQ))
    raises(PolynomialError, lambda: dup_cauchy_lower_bound([QQ(1), QQ(0), QQ(0)], QQ))
    raises(DomainError, lambda: dup_cauchy_lower_bound([ZZ_I(1), ZZ_I(1)], ZZ_I))

    assert dup_cauchy_lower_bound([QQ(1), QQ(0), QQ(-2)], QQ) == QQ(2, 3)


def test_dup_mignotte_sep_bound_squared():
    raises(PolynomialError, lambda: dup_mignotte_sep_bound_squared([], QQ))
    raises(PolynomialError, lambda: dup_mignotte_sep_bound_squared([QQ(1)], QQ))

    assert dup_mignotte_sep_bound_squared([QQ(1), QQ(0), QQ(-2)], QQ) == QQ(3, 5)


def test_dup_refine_real_root():
    R, x = ring("x", ZZ)
    f = x**2 - 2

    assert R.dup_refine_real_root(f, QQ(1), QQ(1), steps=1) == (QQ(1), QQ(1))
    assert R.dup_refine_real_root(f, QQ(1), QQ(1), steps=9) == (QQ(1), QQ(1))

    raises(ValueError, lambda: R.dup_refine_real_root(f, QQ(-2), QQ(2)))

    s, t = QQ(1, 1), QQ(2, 1)

    assert R.dup_refine_real_root(f, s, t, steps=0) == (QQ(1, 1), QQ(2, 1))
    assert R.dup_refine_real_root(f, s, t, steps=1) == (QQ(1, 1), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=2) == (QQ(4, 3), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=3) == (QQ(7, 5), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=4) == (QQ(7, 5), QQ(10, 7))

    s, t = QQ(1, 1), QQ(3, 2)

    assert R.dup_refine_real_root(f, s, t, steps=0) == (QQ(1, 1), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=1) == (QQ(4, 3), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=2) == (QQ(7, 5), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=3) == (QQ(7, 5), QQ(10, 7))
    assert R.dup_refine_real_root(f, s, t, steps=4) == (QQ(7, 5), QQ(17, 12))

    s, t = QQ(1, 1), QQ(5, 3)

    assert R.dup_refine_real_root(f, s, t, steps=0) == (QQ(1, 1), QQ(5, 3))
    assert R.dup_refine_real_root(f, s, t, steps=1) == (QQ(1, 1), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=2) == (QQ(7, 5), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=3) == (QQ(7, 5), QQ(13, 9))
    assert R.dup_refine_real_root(f, s, t, steps=4) == (QQ(7, 5), QQ(27, 19))

    s, t = QQ(-1, 1), QQ(-2, 1)

    assert R.dup_refine_real_root(f, s, t, steps=0) == (-QQ(2, 1), -QQ(1, 1))
    assert R.dup_refine_real_root(f, s, t, steps=1) == (-QQ(3, 2), -QQ(1, 1))
    assert R.dup_refine_real_root(f, s, t, steps=2) == (-QQ(3, 2), -QQ(4, 3))
    assert R.dup_refine_real_root(f, s, t, steps=3) == (-QQ(3, 2), -QQ(7, 5))
    assert R.dup_refine_real_root(f, s, t, steps=4) == (-QQ(10, 7), -QQ(7, 5))

    raises(RefinementFailed, lambda: R.dup_refine_real_root(f, QQ(0), QQ(1)))

    s, t, u, v, w = QQ(1), QQ(2), QQ(24, 17), QQ(17, 12), QQ(7, 5)

    assert R.dup_refine_real_root(f, s, t, eps=QQ(1, 100)) == (u, v)
    assert R.dup_refine_real_root(f, s, t, steps=6) == (u, v)

    assert R.dup_refine_real_root(f, s, t, eps=QQ(1, 100), steps=5) == (w, v)
    assert R.dup_refine_real_root(f, s, t, eps=QQ(1, 100), steps=6) == (u, v)
    assert R.dup_refine_real_root(f, s, t, eps=QQ(1, 100), steps=7) == (u, v)

    s, t, u, v = QQ(-2), QQ(-1), QQ(-3, 2), QQ(-4, 3)

    assert R.dup_refine_real_root(f, s, t, disjoint=QQ(-5)) == (s, t)
    assert R.dup_refine_real_root(f, s, t, disjoint=-v) == (s, t)
    assert R.dup_refine_real_root(f, s, t, disjoint=v) == (u, v)

    s, t, u, v = QQ(1), QQ(2), QQ(4, 3), QQ(3, 2)

    assert R.dup_refine_real_root(f, s, t, disjoint=QQ(5)) == (s, t)
    assert R.dup_refine_real_root(f, s, t, disjoint=-u) == (s, t)
    assert R.dup_refine_real_root(f, s, t, disjoint=u) == (u, v)


def test_dup_isolate_real_roots_sqf():
    R, x = ring("x", ZZ)

    assert R.dup_isolate_real_roots_sqf(0) == []
    assert R.dup_isolate_real_roots_sqf(5) == []

    assert R.dup_isolate_real_roots_sqf(x**2 + x) == [(-1, -1), (0, 0)]
    assert R.dup_isolate_real_roots_sqf(x**2 - x) == [( 0,  0), (1, 1)]

    assert R.dup_isolate_real_roots_sqf(x**4 + x + 1) == []

    I = [(-2, -1), (1, 2)]

    assert R.dup_isolate_real_roots_sqf(x**2 - 2) == I
    assert R.dup_isolate_real_roots_sqf(-x**2 + 2) == I

    assert R.dup_isolate_real_roots_sqf(x - 1) == \
        [(1, 1)]
    assert R.dup_isolate_real_roots_sqf(x**2 - 3*x + 2) == \
        [(1, 1), (2, 2)]
    assert R.dup_isolate_real_roots_sqf(x**3 - 6*x**2 + 11*x - 6) == \
        [(1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(x**4 - 10*x**3 + 35*x**2 - 50*x + 24) == \
        [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert R.dup_isolate_real_roots_sqf(x**5 - 15*x**4 + 85*x**3 - 225*x**2 + 274*x - 120) == \
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    assert R.dup_isolate_real_roots_sqf(x - 10) == \
        [(10, 10)]
    assert R.dup_isolate_real_roots_sqf(x**2 - 30*x + 200) == \
        [(10, 10), (20, 20)]
    assert R.dup_isolate_real_roots_sqf(x**3 - 60*x**2 + 1100*x - 6000) == \
        [(10, 10), (20, 20), (30, 30)]
    assert R.dup_isolate_real_roots_sqf(x**4 - 100*x**3 + 3500*x**2 - 50000*x + 240000) == \
        [(10, 10), (20, 20), (30, 30), (40, 40)]
    assert R.dup_isolate_real_roots_sqf(x**5 - 150*x**4 + 8500*x**3 - 225000*x**2 + 2740000*x - 12000000) == \
        [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]

    assert R.dup_isolate_real_roots_sqf(x + 1) == \
        [(-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x**2 + 3*x + 2) == \
        [(-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x**3 + 6*x**2 + 11*x + 6) == \
        [(-3, -3), (-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x**4 + 10*x**3 + 35*x**2 + 50*x + 24) == \
        [(-4, -4), (-3, -3), (-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x**5 + 15*x**4 + 85*x**3 + 225*x**2 + 274*x + 120) == \
        [(-5, -5), (-4, -4), (-3, -3), (-2, -2), (-1, -1)]

    assert R.dup_isolate_real_roots_sqf(x + 10) == \
        [(-10, -10)]
    assert R.dup_isolate_real_roots_sqf(x**2 + 30*x + 200) == \
        [(-20, -20), (-10, -10)]
    assert R.dup_isolate_real_roots_sqf(x**3 + 60*x**2 + 1100*x + 6000) == \
        [(-30, -30), (-20, -20), (-10, -10)]
    assert R.dup_isolate_real_roots_sqf(x**4 + 100*x**3 + 3500*x**2 + 50000*x + 240000) == \
        [(-40, -40), (-30, -30), (-20, -20), (-10, -10)]
    assert R.dup_isolate_real_roots_sqf(x**5 + 150*x**4 + 8500*x**3 + 225000*x**2 + 2740000*x + 12000000) == \
        [(-50, -50), (-40, -40), (-30, -30), (-20, -20), (-10, -10)]

    assert R.dup_isolate_real_roots_sqf(x**2 - 5) == [(-3, -2), (2, 3)]
    assert R.dup_isolate_real_roots_sqf(x**3 - 5) == [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**4 - 5) == [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**5 - 5) == [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**6 - 5) == [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**7 - 5) == [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**8 - 5) == [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**9 - 5) == [(1, 2)]

    assert R.dup_isolate_real_roots_sqf(x**2 - 1) == \
        [(-1, -1), (1, 1)]
    assert R.dup_isolate_real_roots_sqf(x**3 + 2*x**2 - x - 2) == \
        [(-2, -2), (-1, -1), (1, 1)]
    assert R.dup_isolate_real_roots_sqf(x**4 - 5*x**2 + 4) == \
        [(-2, -2), (-1, -1), (1, 1), (2, 2)]
    assert R.dup_isolate_real_roots_sqf(x**5 + 3*x**4 - 5*x**3 - 15*x**2 + 4*x + 12) == \
        [(-3, -3), (-2, -2), (-1, -1), (1, 1), (2, 2)]
    assert R.dup_isolate_real_roots_sqf(x**6 - 14*x**4 + 49*x**2 - 36) == \
        [(-3, -3), (-2, -2), (-1, -1), (1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(2*x**7 + x**6 - 28*x**5 - 14*x**4 + 98*x**3 + 49*x**2 - 72*x - 36) == \
        [(-3, -3), (-2, -2), (-1, -1), (-1, 0), (1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(4*x**8 - 57*x**6 + 210*x**4 - 193*x**2 + 36) == \
        [(-3, -3), (-2, -2), (-1, -1), (-1, 0), (0, 1), (1, 1), (2, 2), (3, 3)]

    f = 9*x**2 - 2

    assert R.dup_isolate_real_roots_sqf(f) == \
        [(-1, 0), (0, 1)]

    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 10)) == \
        [(QQ(-1, 2), QQ(-3, 7)), (QQ(3, 7), QQ(1, 2))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100)) == \
        [(QQ(-9, 19), QQ(-8, 17)), (QQ(8, 17), QQ(9, 19))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 1000)) == \
        [(QQ(-33, 70), QQ(-8, 17)), (QQ(8, 17), QQ(33, 70))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 10000)) == \
        [(QQ(-33, 70), QQ(-107, 227)), (QQ(107, 227), QQ(33, 70))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100000)) == \
        [(QQ(-305, 647), QQ(-272, 577)), (QQ(272, 577), QQ(305, 647))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 1000000)) == \
        [(QQ(-1121, 2378), QQ(-272, 577)), (QQ(272, 577), QQ(1121, 2378))]

    f = 200100012*x**5 - 700390052*x**4 + 700490079*x**3 - 200240054*x**2 + 40017*x - 2

    assert R.dup_isolate_real_roots_sqf(f) == \
        [(QQ(0), QQ(1, 10002)), (QQ(1, 10002), QQ(1, 10002)),
         (QQ(1, 2), QQ(1, 2)), (QQ(1), QQ(1)), (QQ(2), QQ(2))]

    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100000)) == \
        [(QQ(1, 10003), QQ(1, 10003)), (QQ(1, 10002), QQ(1, 10002)),
         (QQ(1, 2), QQ(1, 2)), (QQ(1), QQ(1)), (QQ(2), QQ(2))]

    a, b, c, d = 10000090000001, 2000100003, 10000300007, 10000005000008

    f = 20001600074001600021*x**4 \
      + 1700135866278935491773999857*x**3 \
      - 2000179008931031182161141026995283662899200197*x**2 \
      - 800027600594323913802305066986600025*x \
      + 100000950000540000725000008

    assert R.dup_isolate_real_roots_sqf(f) == \
        [(-a, -a), (-1, 0), (0, 1), (d, d)]

    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100000000000)) == \
        [(-QQ(a), -QQ(a)), (-QQ(1, b), -QQ(1, b)), (QQ(1, c), QQ(1, c)), (QQ(d), QQ(d))]

    (u, v), B, C, (s, t) = R.dup_isolate_real_roots_sqf(f, fast=True)

    assert u < -a < v and B == (-QQ(1), QQ(0)) and C == (QQ(0), QQ(1)) and s < d < t

    assert R.dup_isolate_real_roots_sqf(f, fast=True, eps=QQ(1, 100000000000000000000000000000)) == \
        [(-QQ(a), -QQ(a)), (-QQ(1, b), -QQ(1, b)), (QQ(1, c), QQ(1, c)), (QQ(d), QQ(d))]

    f = -10*x**4 + 8*x**3 + 80*x**2 - 32*x - 160

    assert R.dup_isolate_real_roots_sqf(f) == \
        [(-2, -2), (-2, -1), (2, 2), (2, 3)]

    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100)) == \
        [(-QQ(2), -QQ(2)), (-QQ(23, 14), -QQ(18, 11)), (QQ(2), QQ(2)), (QQ(39, 16), QQ(22, 9))]

    f = x - 1

    assert R.dup_isolate_real_roots_sqf(f, inf=2) == []
    assert R.dup_isolate_real_roots_sqf(f, sup=0) == []

    assert R.dup_isolate_real_roots_sqf(f) == [(1, 1)]
    assert R.dup_isolate_real_roots_sqf(f, inf=1) == [(1, 1)]
    assert R.dup_isolate_real_roots_sqf(f, sup=1) == [(1, 1)]
    assert R.dup_isolate_real_roots_sqf(f, inf=1, sup=1) == [(1, 1)]

    f = x**2 - 2

    assert R.dup_isolate_real_roots_sqf(f, inf=QQ(7, 4)) == []
    assert R.dup_isolate_real_roots_sqf(f, inf=QQ(7, 5)) == [(QQ(7, 5), QQ(3, 2))]
    assert R.dup_isolate_real_roots_sqf(f, sup=QQ(7, 5)) == [(-2, -1)]
    assert R.dup_isolate_real_roots_sqf(f, sup=QQ(7, 4)) == [(-2, -1), (1, QQ(3, 2))]
    assert R.dup_isolate_real_roots_sqf(f, sup=-QQ(7, 4)) == []
    assert R.dup_isolate_real_roots_sqf(f, sup=-QQ(7, 5)) == [(-QQ(3, 2), -QQ(7, 5))]
    assert R.dup_isolate_real_roots_sqf(f, inf=-QQ(7, 5)) == [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(f, inf=-QQ(7, 4)) == [(-QQ(3, 2), -1), (1, 2)]

    I = [(-2, -1), (1, 2)]

    assert R.dup_isolate_real_roots_sqf(f, inf=-2) == I
    assert R.dup_isolate_real_roots_sqf(f, sup=+2) == I

    assert R.dup_isolate_real_roots_sqf(f, inf=-2, sup=2) == I

    R, x = ring("x", QQ)
    f = QQ(8, 5)*x**2 - QQ(87374, 3855)*x - QQ(17, 771)

    assert R.dup_isolate_real_roots_sqf(f) == [(-1, 0), (14, 15)]

    R, x = ring("x", EX)
    raises(DomainError, lambda: R.dup_isolate_real_roots_sqf(x + 3))

def test_dup_isolate_real_roots():
    R, x = ring("x", ZZ)

    assert R.dup_isolate_real_roots(0) == []
    assert R.dup_isolate_real_roots(3) == []

    assert R.dup_isolate_real_roots(5*x) == [((0, 0), 1)]
    assert R.dup_isolate_real_roots(7*x**4) == [((0, 0), 4)]

    assert R.dup_isolate_real_roots(x**2 + x) == [((-1, -1), 1), ((0, 0), 1)]
    assert R.dup_isolate_real_roots(x**2 - x) == [((0, 0), 1), ((1, 1), 1)]

    assert R.dup_isolate_real_roots(x**4 + x + 1) == []

    I = [((-2, -1), 1), ((1, 2), 1)]

    assert R.dup_isolate_real_roots(x**2 - 2) == I
    assert R.dup_isolate_real_roots(-x**2 + 2) == I

    f = 16*x**14 - 96*x**13 + 24*x**12 + 936*x**11 - 1599*x**10 - 2880*x**9 + 9196*x**8 \
      + 552*x**7 - 21831*x**6 + 13968*x**5 + 21690*x**4 - 26784*x**3 - 2916*x**2 + 15552*x - 5832
    g = R.dup_sqf_part(f)

    assert R.dup_isolate_real_roots(f) == \
        [((-QQ(2), -QQ(3, 2)), 2), ((-QQ(3, 2), -QQ(1, 1)), 3), ((QQ(1), QQ(3, 2)), 3),
         ((QQ(3, 2), QQ(3, 2)), 4), ((QQ(5, 3), QQ(2)), 2)]

    assert R.dup_isolate_real_roots_sqf(g) == \
        [(-QQ(2), -QQ(3, 2)), (-QQ(3, 2), -QQ(1, 1)), (QQ(1), QQ(3, 2)),
         (QQ(3, 2), QQ(3, 2)), (QQ(3, 2), QQ(2))]
    assert R.dup_isolate_real_roots(g) == \
        [((-QQ(2), -QQ(3, 2)), 1), ((-QQ(3, 2), -QQ(1, 1)), 1), ((QQ(1), QQ(3, 2)), 1),
         ((QQ(3, 2), QQ(3, 2)), 1), ((QQ(3, 2), QQ(2)), 1)]

    f = x - 1

    assert R.dup_isolate_real_roots(f, inf=2) == []
    assert R.dup_isolate_real_roots(f, sup=0) == []

    assert R.dup_isolate_real_roots(f) == [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f, inf=1) == [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f, sup=1) == [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f, inf=1, sup=1) == [((1, 1), 1)]

    f = x**4 - 4*x**2 + 4

    assert R.dup_isolate_real_roots(f, inf=QQ(7, 4)) == []
    assert R.dup_isolate_real_roots(f, inf=QQ(7, 5)) == [((QQ(7, 5), QQ(3, 2)), 2)]
    assert R.dup_isolate_real_roots(f, sup=QQ(7, 5)) == [((-2, -1), 2)]
    assert R.dup_isolate_real_roots(f, sup=QQ(7, 4)) == [((-2, -1), 2), ((1, QQ(3, 2)), 2)]
    assert R.dup_isolate_real_roots(f, sup=-QQ(7, 4)) == []
    assert R.dup_isolate_real_roots(f, sup=-QQ(7, 5)) == [((-QQ(3, 2), -QQ(7, 5)), 2)]
    assert R.dup_isolate_real_roots(f, inf=-QQ(7, 5)) == [((1, 2), 2)]
    assert R.dup_isolate_real_roots(f, inf=-QQ(7, 4)) == [((-QQ(3, 2), -1), 2), ((1, 2), 2)]

    I = [((-2, -1), 2), ((1, 2), 2)]

    assert R.dup_isolate_real_roots(f, inf=-2) == I
    assert R.dup_isolate_real_roots(f, sup=+2) == I

    assert R.dup_isolate_real_roots(f, inf=-2, sup=2) == I

    f = x**11 - 3*x**10 - x**9 + 11*x**8 - 8*x**7 - 8*x**6 + 12*x**5 - 4*x**4

    assert R.dup_isolate_real_roots(f, basis=False) == \
        [((-2, -1), 2), ((0, 0), 4), ((1, 1), 3), ((1, 2), 2)]
    assert R.dup_isolate_real_roots(f, basis=True) == \
        [((-2, -1), 2, [1, 0, -2]), ((0, 0), 4, [1, 0]), ((1, 1), 3, [1, -1]), ((1, 2), 2, [1, 0, -2])]

    f = (x**45 - 45*x**44 + 990*x**43 - 1)
    g = (x**46 - 15180*x**43 + 9366819*x**40 - 53524680*x**39 + 260932815*x**38 - 1101716330*x**37 + 4076350421*x**36 - 13340783196*x**35 + 38910617655*x**34 - 101766230790*x**33 + 239877544005*x**32 - 511738760544*x**31 + 991493848554*x**30 - 1749695026860*x**29 + 2818953098830*x**28 - 4154246671960*x**27 + 5608233007146*x**26 - 6943526580276*x**25 + 7890371113950*x**24 - 8233430727600*x**23 + 7890371113950*x**22 - 6943526580276*x**21 + 5608233007146*x**20 - 4154246671960*x**19 + 2818953098830*x**18 - 1749695026860*x**17 + 991493848554*x**16 - 511738760544*x**15 + 239877544005*x**14 - 101766230790*x**13 + 38910617655*x**12 - 13340783196*x**11 + 4076350421*x**10 - 1101716330*x**9 + 260932815*x**8 - 53524680*x**7 + 9366819*x**6 - 1370754*x**5 + 163185*x**4 - 15180*x**3 + 1035*x**2 - 47*x + 1)

    assert R.dup_isolate_real_roots(f*g) == \
        [((0, QQ(1, 2)), 1), ((QQ(2, 3), QQ(3, 4)), 1), ((QQ(3, 4), 1), 1), ((6, 7), 1), ((24, 25), 1)]

    R, x = ring("x", EX)
    raises(DomainError, lambda: R.dup_isolate_real_roots(x + 3))


def test_dup_isolate_real_roots_list():
    R, x = ring("x", ZZ)

    assert R.dup_isolate_real_roots_list([x**2 + x, x]) == \
        [((-1, -1), {0: 1}), ((0, 0), {0: 1, 1: 1})]
    assert R.dup_isolate_real_roots_list([x**2 - x, x]) == \
        [((0, 0), {0: 1, 1: 1}), ((1, 1), {0: 1})]

    assert R.dup_isolate_real_roots_list([x + 1, x + 2, x - 1, x + 1, x - 1, x - 1]) == \
        [((-QQ(2), -QQ(2)), {1: 1}), ((-QQ(1), -QQ(1)), {0: 1, 3: 1}), ((QQ(1), QQ(1)), {2: 1, 4: 1, 5: 1})]

    assert R.dup_isolate_real_roots_list([x + 1, x + 2, x - 1, x + 1, x - 1, x + 2]) == \
        [((-QQ(2), -QQ(2)), {1: 1, 5: 1}), ((-QQ(1), -QQ(1)), {0: 1, 3: 1}), ((QQ(1), QQ(1)), {2: 1, 4: 1})]

    f, g = x**4 - 4*x**2 + 4, x - 1

    assert R.dup_isolate_real_roots_list([f, g], inf=QQ(7, 4)) == []
    assert R.dup_isolate_real_roots_list([f, g], inf=QQ(7, 5)) == \
        [((QQ(7, 5), QQ(3, 2)), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], sup=QQ(7, 5)) == \
        [((-2, -1), {0: 2}), ((1, 1), {1: 1})]
    assert R.dup_isolate_real_roots_list([f, g], sup=QQ(7, 4)) == \
        [((-2, -1), {0: 2}), ((1, 1), {1: 1}), ((1, QQ(3, 2)), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], sup=-QQ(7, 4)) == []
    assert R.dup_isolate_real_roots_list([f, g], sup=-QQ(7, 5)) == \
        [((-QQ(3, 2), -QQ(7, 5)), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], inf=-QQ(7, 5)) == \
        [((1, 1), {1: 1}), ((1, 2), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], inf=-QQ(7, 4)) == \
        [((-QQ(3, 2), -1), {0: 2}), ((1, 1), {1: 1}), ((1, 2), {0: 2})]

    f, g = 2*x**2 - 1, x**2 - 2

    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((-QQ(2), -QQ(1)), {1: 1}), ((-QQ(1), QQ(0)), {0: 1}),
         ((QQ(0), QQ(1)), {0: 1}), ((QQ(1), QQ(2)), {1: 1})]
    assert R.dup_isolate_real_roots_list([f, g], strict=True) == \
        [((-QQ(3, 2), -QQ(4, 3)), {1: 1}), ((-QQ(1), -QQ(2, 3)), {0: 1}),
         ((QQ(2, 3), QQ(1)), {0: 1}), ((QQ(4, 3), QQ(3, 2)), {1: 1})]

    f, g = x**2 - 2, x**3 - x**2 - 2*x + 2

    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((-QQ(2), -QQ(1)), {1: 1, 0: 1}), ((QQ(1), QQ(1)), {1: 1}), ((QQ(1), QQ(2)), {1: 1, 0: 1})]

    f, g = x**3 - 2*x, x**5 - x**4 - 2*x**3 + 2*x**2

    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((-QQ(2), -QQ(1)), {1: 1, 0: 1}), ((QQ(0), QQ(0)), {0: 1, 1: 2}),
         ((QQ(1), QQ(1)), {1: 1}), ((QQ(1), QQ(2)), {1: 1, 0: 1})]

    f, g = x**9 - 3*x**8 - x**7 + 11*x**6 - 8*x**5 - 8*x**4 + 12*x**3 - 4*x**2, x**5 - 2*x**4 + 3*x**3 - 4*x**2 + 2*x

    assert R.dup_isolate_real_roots_list([f, g], basis=False) == \
        [((-2, -1), {0: 2}), ((0, 0), {0: 2, 1: 1}), ((1, 1), {0: 3, 1: 2}), ((1, 2), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], basis=True) == \
        [((-2, -1), {0: 2}, [1, 0, -2]), ((0, 0), {0: 2, 1: 1}, [1, 0]),
         ((1, 1), {0: 3, 1: 2}, [1, -1]), ((1, 2), {0: 2}, [1, 0, -2])]

    R, x = ring("x", EX)
    raises(DomainError, lambda: R.dup_isolate_real_roots_list([x + 3]))


def test_dup_isolate_real_roots_list_QQ():
    R, x = ring("x", ZZ)

    f = x**5 - 200
    g = x**5 - 201

    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((QQ(75, 26), QQ(101, 35)), {0: 1}), ((QQ(309, 107), QQ(26, 9)), {1: 1})]

    R, x = ring("x", QQ)

    f = -QQ(1, 200)*x**5 + 1
    g = -QQ(1, 201)*x**5 + 1

    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((QQ(75, 26), QQ(101, 35)), {0: 1}), ((QQ(309, 107), QQ(26, 9)), {1: 1})]


def test_dup_count_real_roots():
    R, x = ring("x", ZZ)

    assert R.dup_count_real_roots(0) == 0
    assert R.dup_count_real_roots(7) == 0

    f = x - 1
    assert R.dup_count_real_roots(f) == 1
    assert R.dup_count_real_roots(f, inf=1) == 1
    assert R.dup_count_real_roots(f, sup=0) == 0
    assert R.dup_count_real_roots(f, sup=1) == 1
    assert R.dup_count_real_roots(f, inf=0, sup=1) == 1
    assert R.dup_count_real_roots(f, inf=0, sup=2) == 1
    assert R.dup_count_real_roots(f, inf=1, sup=2) == 1

    f = x**2 - 2
    assert R.dup_count_real_roots(f) == 2
    assert R.dup_count_real_roots(f, sup=0) == 1
    assert R.dup_count_real_roots(f, inf=-1, sup=1) == 0


# parameters for test_dup_count_complex_roots_n(): n = 1..8
a, b = (-QQ(1), -QQ(1)), (QQ(1), QQ(1))
c, d = ( QQ(0),  QQ(0)), (QQ(1), QQ(1))

def test_dup_count_complex_roots_1():
    R, x = ring("x", ZZ)

    # z-1
    f = x - 1
    assert R.dup_count_complex_roots(f, a, b) == 1
    assert R.dup_count_complex_roots(f, c, d) == 1

    # z+1
    f = x + 1
    assert R.dup_count_complex_roots(f, a, b) == 1
    assert R.dup_count_complex_roots(f, c, d) == 0


def test_dup_count_complex_roots_2():
    R, x = ring("x", ZZ)

    # (z-1)*(z)
    f = x**2 - x
    assert R.dup_count_complex_roots(f, a, b) == 2
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-1)*(-z)
    f = -x**2 + x
    assert R.dup_count_complex_roots(f, a, b) == 2
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z+1)*(z)
    f = x**2 + x
    assert R.dup_count_complex_roots(f, a, b) == 2
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z+1)*(-z)
    f = -x**2 - x
    assert R.dup_count_complex_roots(f, a, b) == 2
    assert R.dup_count_complex_roots(f, c, d) == 1


def test_dup_count_complex_roots_3():
    R, x = ring("x", ZZ)

    # (z-1)*(z+1)
    f = x**2 - 1
    assert R.dup_count_complex_roots(f, a, b) == 2
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-1)*(z+1)*(z)
    f = x**3 - x
    assert R.dup_count_complex_roots(f, a, b) == 3
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-1)*(z+1)*(-z)
    f = -x**3 + x
    assert R.dup_count_complex_roots(f, a, b) == 3
    assert R.dup_count_complex_roots(f, c, d) == 2


def test_dup_count_complex_roots_4():
    R, x = ring("x", ZZ)

    # (z-I)*(z+I)
    f = x**2 + 1
    assert R.dup_count_complex_roots(f, a, b) == 2
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I)*(z+I)*(z)
    f = x**3 + x
    assert R.dup_count_complex_roots(f, a, b) == 3
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I)*(z+I)*(-z)
    f = -x**3 - x
    assert R.dup_count_complex_roots(f, a, b) == 3
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I)*(z+I)*(z-1)
    f = x**3 - x**2 + x - 1
    assert R.dup_count_complex_roots(f, a, b) == 3
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I)*(z+I)*(z-1)*(z)
    f = x**4 - x**3 + x**2 - x
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 3

    # (z-I)*(z+I)*(z-1)*(-z)
    f = -x**4 + x**3 - x**2 + x
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 3

    # (z-I)*(z+I)*(z-1)*(z+1)
    f = x**4 - 1
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I)*(z+I)*(z-1)*(z+1)*(z)
    f = x**5 - x
    assert R.dup_count_complex_roots(f, a, b) == 5
    assert R.dup_count_complex_roots(f, c, d) == 3

    # (z-I)*(z+I)*(z-1)*(z+1)*(-z)
    f = -x**5 + x
    assert R.dup_count_complex_roots(f, a, b) == 5
    assert R.dup_count_complex_roots(f, c, d) == 3


def test_dup_count_complex_roots_5():
    R, x = ring("x", ZZ)

    # (z-I+1)*(z+I+1)
    f = x**2 + 2*x + 2
    assert R.dup_count_complex_roots(f, a, b) == 2
    assert R.dup_count_complex_roots(f, c, d) == 0

    # (z-I+1)*(z+I+1)*(z-1)
    f = x**3 + x**2 - 2
    assert R.dup_count_complex_roots(f, a, b) == 3
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I+1)*(z+I+1)*(z-1)*z
    f = x**4 + x**3 - 2*x
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I+1)*(z+I+1)*(z+1)
    f = x**3 + 3*x**2 + 4*x + 2
    assert R.dup_count_complex_roots(f, a, b) == 3
    assert R.dup_count_complex_roots(f, c, d) == 0

    # (z-I+1)*(z+I+1)*(z+1)*z
    f = x**4 + 3*x**3 + 4*x**2 + 2*x
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I+1)*(z+I+1)*(z-1)*(z+1)
    f = x**4 + 2*x**3 + x**2 - 2*x - 2
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I+1)*(z+I+1)*(z-1)*(z+1)*z
    f = x**5 + 2*x**4 + x**3 - 2*x**2 - 2*x
    assert R.dup_count_complex_roots(f, a, b) == 5
    assert R.dup_count_complex_roots(f, c, d) == 2


def test_dup_count_complex_roots_6():
    R, x = ring("x", ZZ)

    # (z-I-1)*(z+I-1)
    f = x**2 - 2*x + 2
    assert R.dup_count_complex_roots(f, a, b) == 2
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I-1)*(z+I-1)*(z-1)
    f = x**3 - 3*x**2 + 4*x - 2
    assert R.dup_count_complex_roots(f, a, b) == 3
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I-1)*(z+I-1)*(z-1)*z
    f = x**4 - 3*x**3 + 4*x**2 - 2*x
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 3

    # (z-I-1)*(z+I-1)*(z+1)
    f = x**3 - x**2 + 2
    assert R.dup_count_complex_roots(f, a, b) == 3
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I-1)*(z+I-1)*(z+1)*z
    f = x**4 - x**3 + 2*x
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I-1)*(z+I-1)*(z-1)*(z+1)
    f = x**4 - 2*x**3 + x**2 + 2*x - 2
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I-1)*(z+I-1)*(z-1)*(z+1)*z
    f = x**5 - 2*x**4 + x**3 + 2*x**2 - 2*x
    assert R.dup_count_complex_roots(f, a, b) == 5
    assert R.dup_count_complex_roots(f, c, d) == 3


def test_dup_count_complex_roots_7():
    R, x = ring("x", ZZ)

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)
    f = x**4 + 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-2)
    f = x**5 - 2*x**4 + 4*x - 8
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z**2-2)
    f = x**6 - 2*x**4 + 4*x**2 - 8
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)
    f = x**5 - x**4 + 4*x - 4
    assert R.dup_count_complex_roots(f, a, b) == 5
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*z
    f = x**6 - x**5 + 4*x**2 - 4*x
    assert R.dup_count_complex_roots(f, a, b) == 6
    assert R.dup_count_complex_roots(f, c, d) == 3

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z+1)
    f = x**5 + x**4 + 4*x + 4
    assert R.dup_count_complex_roots(f, a, b) == 5
    assert R.dup_count_complex_roots(f, c, d) == 1

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z+1)*z
    f = x**6 + x**5 + 4*x**2 + 4*x
    assert R.dup_count_complex_roots(f, a, b) == 6
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*(z+1)
    f = x**6 - x**4 + 4*x**2 - 4
    assert R.dup_count_complex_roots(f, a, b) == 6
    assert R.dup_count_complex_roots(f, c, d) == 2

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*(z+1)*z
    f = x**7 - x**5 + 4*x**3 - 4*x
    assert R.dup_count_complex_roots(f, a, b) == 7
    assert R.dup_count_complex_roots(f, c, d) == 3

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*(z+1)*(z-I)*(z+I)
    f = x**8 + 3*x**4 - 4
    assert R.dup_count_complex_roots(f, a, b) == 8
    assert R.dup_count_complex_roots(f, c, d) == 3


def test_dup_count_complex_roots_8():
    R, x = ring("x", ZZ)

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*(z+1)*(z-I)*(z+I)*z
    f = x**9 + 3*x**5 - 4*x
    assert R.dup_count_complex_roots(f, a, b) == 9
    assert R.dup_count_complex_roots(f, c, d) == 4

    # (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*(z+1)*(z-I)*(z+I)*(z**2-2)*z
    f = x**11 - 2*x**9 + 3*x**7 - 6*x**5 - 4*x**3 + 8*x
    assert R.dup_count_complex_roots(f, a, b) == 9
    assert R.dup_count_complex_roots(f, c, d) == 4


def test_dup_count_complex_roots_implicit():
    R, x = ring("x", ZZ)

    # z*(z-1)*(z+1)*(z-I)*(z+I)
    f = x**5 - x

    assert R.dup_count_complex_roots(f) == 5

    assert R.dup_count_complex_roots(f, sup=(0, 0)) == 3
    assert R.dup_count_complex_roots(f, inf=(0, 0)) == 3


def test_dup_count_complex_roots_exclude():
    R, x = ring("x", ZZ)

    # z*(z-1)*(z+1)*(z-I)*(z+I)
    f = x**5 - x

    a, b = (-QQ(1), QQ(0)), (QQ(1), QQ(1))

    assert R.dup_count_complex_roots(f, a, b) == 4

    assert R.dup_count_complex_roots(f, a, b, exclude=['S']) == 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['N']) == 3

    assert R.dup_count_complex_roots(f, a, b, exclude=['S', 'N']) == 2

    assert R.dup_count_complex_roots(f, a, b, exclude=['E']) == 4
    assert R.dup_count_complex_roots(f, a, b, exclude=['W']) == 4

    assert R.dup_count_complex_roots(f, a, b, exclude=['E', 'W']) == 4

    assert R.dup_count_complex_roots(f, a, b, exclude=['N', 'S', 'E', 'W']) == 2

    assert R.dup_count_complex_roots(f, a, b, exclude=['SW']) == 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['SE']) == 3

    assert R.dup_count_complex_roots(f, a, b, exclude=['SW', 'SE']) == 2
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW', 'SE', 'S']) == 1
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW', 'SE', 'S', 'N']) == 0

    a, b = (QQ(0), QQ(0)), (QQ(1), QQ(1))

    assert R.dup_count_complex_roots(f, a, b, exclude=True) == 1


def test_dup_isolate_complex_roots_sqf():
    R, x = ring("x", ZZ)
    f = x**2 - 2*x + 3

    assert R.dup_isolate_complex_roots_sqf(f) == \
        [((0, -6), (6, 0)), ((0, 0), (6, 6))]
    assert [ r.as_tuple() for r in R.dup_isolate_complex_roots_sqf(f, blackbox=True) ] == \
        [((0, -6), (6, 0)), ((0, 0), (6, 6))]

    assert R.dup_isolate_complex_roots_sqf(f, eps=QQ(1, 10)) == \
        [((QQ(15, 16), -QQ(3, 2)), (QQ(33, 32), -QQ(45, 32))),
         ((QQ(15, 16), QQ(45, 32)), (QQ(33, 32), QQ(3, 2)))]
    assert R.dup_isolate_complex_roots_sqf(f, eps=QQ(1, 100)) == \
        [((QQ(255, 256), -QQ(363, 256)), (QQ(513, 512), -QQ(723, 512))),
         ((QQ(255, 256), QQ(723, 512)), (QQ(513, 512), QQ(363, 256)))]

    f = 7*x**4 - 19*x**3 + 20*x**2 + 17*x + 20

    assert R.dup_isolate_complex_roots_sqf(f) == \
        [((-QQ(40, 7), -QQ(40, 7)), (0, 0)), ((-QQ(40, 7), 0), (0, QQ(40, 7))),
         ((0, -QQ(40, 7)), (QQ(40, 7), 0)), ((0, 0), (QQ(40, 7), QQ(40, 7)))]


def test_dup_isolate_all_roots_sqf():
    R, x = ring("x", ZZ)
    f = 4*x**4 - x**3 + 2*x**2 + 5*x

    assert R.dup_isolate_all_roots_sqf(f) == \
        ([(-1, 0), (0, 0)],
         [((0, -QQ(5, 2)), (QQ(5, 2), 0)), ((0, 0), (QQ(5, 2), QQ(5, 2)))])

    assert R.dup_isolate_all_roots_sqf(f, eps=QQ(1, 10)) == \
        ([(QQ(-7, 8), QQ(-6, 7)), (0, 0)],
         [((QQ(35, 64), -QQ(35, 32)), (QQ(5, 8), -QQ(65, 64))), ((QQ(35, 64), QQ(65, 64)), (QQ(5, 8), QQ(35, 32)))])


def test_dup_isolate_all_roots():
    R, x = ring("x", ZZ)
    f = 4*x**4 - x**3 + 2*x**2 + 5*x

    assert R.dup_isolate_all_roots(f) == \
        ([((-1, 0), 1), ((0, 0), 1)],
         [(((0, -QQ(5, 2)), (QQ(5, 2), 0)), 1),
          (((0, 0), (QQ(5, 2), QQ(5, 2))), 1)])

    assert R.dup_isolate_all_roots(f, eps=QQ(1, 10)) == \
        ([((QQ(-7, 8), QQ(-6, 7)), 1), ((0, 0), 1)],
         [(((QQ(35, 64), -QQ(35, 32)), (QQ(5, 8), -QQ(65, 64))), 1),
          (((QQ(35, 64), QQ(65, 64)), (QQ(5, 8), QQ(35, 32))), 1)])

    f = x**5 + x**4 - 2*x**3 - 2*x**2 + x + 1
    raises(NotImplementedError, lambda: R.dup_isolate_all_roots(f))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_smoke.py ===
import pickle
from functools import partial

import pytest

import numpy as np
from numpy.random import MT19937, PCG64, PCG64DXSM, SFC64, Generator, Philox
from numpy.testing import assert_, assert_array_equal, assert_equal


@pytest.fixture(scope='module',
                params=(np.bool, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64))
def dtype(request):
    return request.param


def params_0(f):
    val = f()
    assert_(np.isscalar(val))
    val = f(10)
    assert_(val.shape == (10,))
    val = f((10, 10))
    assert_(val.shape == (10, 10))
    val = f((10, 10, 10))
    assert_(val.shape == (10, 10, 10))
    val = f(size=(5, 5))
    assert_(val.shape == (5, 5))


def params_1(f, bounded=False):
    a = 5.0
    b = np.arange(2.0, 12.0)
    c = np.arange(2.0, 102.0).reshape((10, 10))
    d = np.arange(2.0, 1002.0).reshape((10, 10, 10))
    e = np.array([2.0, 3.0])
    g = np.arange(2.0, 12.0).reshape((1, 10, 1))
    if bounded:
        a = 0.5
        b = b / (1.5 * b.max())
        c = c / (1.5 * c.max())
        d = d / (1.5 * d.max())
        e = e / (1.5 * e.max())
        g = g / (1.5 * g.max())

    # Scalar
    f(a)
    # Scalar - size
    f(a, size=(10, 10))
    # 1d
    f(b)
    # 2d
    f(c)
    # 3d
    f(d)
    # 1d size
    f(b, size=10)
    # 2d - size - broadcast
    f(e, size=(10, 2))
    # 3d - size
    f(g, size=(10, 10, 10))


def comp_state(state1, state2):
    identical = True
    if isinstance(state1, dict):
        for key in state1:
            identical &= comp_state(state1[key], state2[key])
    elif type(state1) != type(state2):
        identical &= type(state1) == type(state2)
    elif (isinstance(state1, (list, tuple, np.ndarray)) and isinstance(
            state2, (list, tuple, np.ndarray))):
        for s1, s2 in zip(state1, state2):
            identical &= comp_state(s1, s2)
    else:
        identical &= state1 == state2
    return identical


def warmup(rg, n=None):
    if n is None:
        n = 11 + np.random.randint(0, 20)
    rg.standard_normal(n)
    rg.standard_normal(n)
    rg.standard_normal(n, dtype=np.float32)
    rg.standard_normal(n, dtype=np.float32)
    rg.integers(0, 2 ** 24, n, dtype=np.uint64)
    rg.integers(0, 2 ** 48, n, dtype=np.uint64)
    rg.standard_gamma(11.0, n)
    rg.standard_gamma(11.0, n, dtype=np.float32)
    rg.random(n, dtype=np.float64)
    rg.random(n, dtype=np.float32)


class RNG:
    @classmethod
    def setup_class(cls):
        # Overridden in test classes. Place holder to silence IDE noise
        cls.bit_generator = PCG64
        cls.advance = None
        cls.seed = [12345]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 64
        cls._extra_setup()

    @classmethod
    def _extra_setup(cls):
        cls.vec_1d = np.arange(2.0, 102.0)
        cls.vec_2d = np.arange(2.0, 102.0)[None, :]
        cls.mat = np.arange(2.0, 102.0, 0.01).reshape((100, 100))
        cls.seed_error = TypeError

    def _reset_state(self):
        self.rg.bit_generator.state = self.initial_state

    def test_init(self):
        rg = Generator(self.bit_generator())
        state = rg.bit_generator.state
        rg.standard_normal(1)
        rg.standard_normal(1)
        rg.bit_generator.state = state
        new_state = rg.bit_generator.state
        assert_(comp_state(state, new_state))

    def test_advance(self):
        state = self.rg.bit_generator.state
        if hasattr(self.rg.bit_generator, 'advance'):
            self.rg.bit_generator.advance(self.advance)
            assert_(not comp_state(state, self.rg.bit_generator.state))
        else:
            bitgen_name = self.rg.bit_generator.__class__.__name__
            pytest.skip(f'Advance is not supported by {bitgen_name}')

    def test_jump(self):
        state = self.rg.bit_generator.state
        if hasattr(self.rg.bit_generator, 'jumped'):
            bit_gen2 = self.rg.bit_generator.jumped()
            jumped_state = bit_gen2.state
            assert_(not comp_state(state, jumped_state))
            self.rg.random(2 * 3 * 5 * 7 * 11 * 13 * 17)
            self.rg.bit_generator.state = state
            bit_gen3 = self.rg.bit_generator.jumped()
            rejumped_state = bit_gen3.state
            assert_(comp_state(jumped_state, rejumped_state))
        else:
            bitgen_name = self.rg.bit_generator.__class__.__name__
            if bitgen_name not in ('SFC64',):
                raise AttributeError(f'no "jumped" in {bitgen_name}')
            pytest.skip(f'Jump is not supported by {bitgen_name}')

    def test_uniform(self):
        r = self.rg.uniform(-1.0, 0.0, size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())

    def test_uniform_array(self):
        r = self.rg.uniform(np.array([-1.0] * 10), 0.0, size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())
        r = self.rg.uniform(np.array([-1.0] * 10),
                            np.array([0.0] * 10), size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())
        r = self.rg.uniform(-1.0, np.array([0.0] * 10), size=10)
        assert_(len(r) == 10)
        assert_((r > -1).all())
        assert_((r <= 0).all())

    def test_random(self):
        assert_(len(self.rg.random(10)) == 10)
        params_0(self.rg.random)

    def test_standard_normal_zig(self):
        assert_(len(self.rg.standard_normal(10)) == 10)

    def test_standard_normal(self):
        assert_(len(self.rg.standard_normal(10)) == 10)
        params_0(self.rg.standard_normal)

    def test_standard_gamma(self):
        assert_(len(self.rg.standard_gamma(10, 10)) == 10)
        assert_(len(self.rg.standard_gamma(np.array([10] * 10), 10)) == 10)
        params_1(self.rg.standard_gamma)

    def test_standard_exponential(self):
        assert_(len(self.rg.standard_exponential(10)) == 10)
        params_0(self.rg.standard_exponential)

    def test_standard_exponential_float(self):
        randoms = self.rg.standard_exponential(10, dtype='float32')
        assert_(len(randoms) == 10)
        assert randoms.dtype == np.float32
        params_0(partial(self.rg.standard_exponential, dtype='float32'))

    def test_standard_exponential_float_log(self):
        randoms = self.rg.standard_exponential(10, dtype='float32',
                                               method='inv')
        assert_(len(randoms) == 10)
        assert randoms.dtype == np.float32
        params_0(partial(self.rg.standard_exponential, dtype='float32',
                         method='inv'))

    def test_standard_cauchy(self):
        assert_(len(self.rg.standard_cauchy(10)) == 10)
        params_0(self.rg.standard_cauchy)

    def test_standard_t(self):
        assert_(len(self.rg.standard_t(10, 10)) == 10)
        params_1(self.rg.standard_t)

    def test_binomial(self):
        assert_(self.rg.binomial(10, .5) >= 0)
        assert_(self.rg.binomial(1000, .5) >= 0)

    def test_reset_state(self):
        state = self.rg.bit_generator.state
        int_1 = self.rg.integers(2**31)
        self.rg.bit_generator.state = state
        int_2 = self.rg.integers(2**31)
        assert_(int_1 == int_2)

    def test_entropy_init(self):
        rg = Generator(self.bit_generator())
        rg2 = Generator(self.bit_generator())
        assert_(not comp_state(rg.bit_generator.state,
                               rg2.bit_generator.state))

    def test_seed(self):
        rg = Generator(self.bit_generator(*self.seed))
        rg2 = Generator(self.bit_generator(*self.seed))
        rg.random()
        rg2.random()
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))

    def test_reset_state_gauss(self):
        rg = Generator(self.bit_generator(*self.seed))
        rg.standard_normal()
        state = rg.bit_generator.state
        n1 = rg.standard_normal(size=10)
        rg2 = Generator(self.bit_generator())
        rg2.bit_generator.state = state
        n2 = rg2.standard_normal(size=10)
        assert_array_equal(n1, n2)

    def test_reset_state_uint32(self):
        rg = Generator(self.bit_generator(*self.seed))
        rg.integers(0, 2 ** 24, 120, dtype=np.uint32)
        state = rg.bit_generator.state
        n1 = rg.integers(0, 2 ** 24, 10, dtype=np.uint32)
        rg2 = Generator(self.bit_generator())
        rg2.bit_generator.state = state
        n2 = rg2.integers(0, 2 ** 24, 10, dtype=np.uint32)
        assert_array_equal(n1, n2)

    def test_reset_state_float(self):
        rg = Generator(self.bit_generator(*self.seed))
        rg.random(dtype='float32')
        state = rg.bit_generator.state
        n1 = rg.random(size=10, dtype='float32')
        rg2 = Generator(self.bit_generator())
        rg2.bit_generator.state = state
        n2 = rg2.random(size=10, dtype='float32')
        assert_((n1 == n2).all())

    def test_shuffle(self):
        original = np.arange(200, 0, -1)
        permuted = self.rg.permutation(original)
        assert_((original != permuted).any())

    def test_permutation(self):
        original = np.arange(200, 0, -1)
        permuted = self.rg.permutation(original)
        assert_((original != permuted).any())

    def test_beta(self):
        vals = self.rg.beta(2.0, 2.0, 10)
        assert_(len(vals) == 10)
        vals = self.rg.beta(np.array([2.0] * 10), 2.0)
        assert_(len(vals) == 10)
        vals = self.rg.beta(2.0, np.array([2.0] * 10))
        assert_(len(vals) == 10)
        vals = self.rg.beta(np.array([2.0] * 10), np.array([2.0] * 10))
        assert_(len(vals) == 10)
        vals = self.rg.beta(np.array([2.0] * 10), np.array([[2.0]] * 10))
        assert_(vals.shape == (10, 10))

    def test_bytes(self):
        vals = self.rg.bytes(10)
        assert_(len(vals) == 10)

    def test_chisquare(self):
        vals = self.rg.chisquare(2.0, 10)
        assert_(len(vals) == 10)
        params_1(self.rg.chisquare)

    def test_exponential(self):
        vals = self.rg.exponential(2.0, 10)
        assert_(len(vals) == 10)
        params_1(self.rg.exponential)

    def test_f(self):
        vals = self.rg.f(3, 1000, 10)
        assert_(len(vals) == 10)

    def test_gamma(self):
        vals = self.rg.gamma(3, 2, 10)
        assert_(len(vals) == 10)

    def test_geometric(self):
        vals = self.rg.geometric(0.5, 10)
        assert_(len(vals) == 10)
        params_1(self.rg.exponential, bounded=True)

    def test_gumbel(self):
        vals = self.rg.gumbel(2.0, 2.0, 10)
        assert_(len(vals) == 10)

    def test_laplace(self):
        vals = self.rg.laplace(2.0, 2.0, 10)
        assert_(len(vals) == 10)

    def test_logitic(self):
        vals = self.rg.logistic(2.0, 2.0, 10)
        assert_(len(vals) == 10)

    def test_logseries(self):
        vals = self.rg.logseries(0.5, 10)
        assert_(len(vals) == 10)

    def test_negative_binomial(self):
        vals = self.rg.negative_binomial(10, 0.2, 10)
        assert_(len(vals) == 10)

    def test_noncentral_chisquare(self):
        vals = self.rg.noncentral_chisquare(10, 2, 10)
        assert_(len(vals) == 10)

    def test_noncentral_f(self):
        vals = self.rg.noncentral_f(3, 1000, 2, 10)
        assert_(len(vals) == 10)
        vals = self.rg.noncentral_f(np.array([3] * 10), 1000, 2)
        assert_(len(vals) == 10)
        vals = self.rg.noncentral_f(3, np.array([1000] * 10), 2)
        assert_(len(vals) == 10)
        vals = self.rg.noncentral_f(3, 1000, np.array([2] * 10))
        assert_(len(vals) == 10)

    def test_normal(self):
        vals = self.rg.normal(10, 0.2, 10)
        assert_(len(vals) == 10)

    def test_pareto(self):
        vals = self.rg.pareto(3.0, 10)
        assert_(len(vals) == 10)

    def test_poisson(self):
        vals = self.rg.poisson(10, 10)
        assert_(len(vals) == 10)
        vals = self.rg.poisson(np.array([10] * 10))
        assert_(len(vals) == 10)
        params_1(self.rg.poisson)

    def test_power(self):
        vals = self.rg.power(0.2, 10)
        assert_(len(vals) == 10)

    def test_integers(self):
        vals = self.rg.integers(10, 20, 10)
        assert_(len(vals) == 10)

    def test_rayleigh(self):
        vals = self.rg.rayleigh(0.2, 10)
        assert_(len(vals) == 10)
        params_1(self.rg.rayleigh, bounded=True)

    def test_vonmises(self):
        vals = self.rg.vonmises(10, 0.2, 10)
        assert_(len(vals) == 10)

    def test_wald(self):
        vals = self.rg.wald(1.0, 1.0, 10)
        assert_(len(vals) == 10)

    def test_weibull(self):
        vals = self.rg.weibull(1.0, 10)
        assert_(len(vals) == 10)

    def test_zipf(self):
        vals = self.rg.zipf(10, 10)
        assert_(len(vals) == 10)
        vals = self.rg.zipf(self.vec_1d)
        assert_(len(vals) == 100)
        vals = self.rg.zipf(self.vec_2d)
        assert_(vals.shape == (1, 100))
        vals = self.rg.zipf(self.mat)
        assert_(vals.shape == (100, 100))

    def test_hypergeometric(self):
        vals = self.rg.hypergeometric(25, 25, 20)
        assert_(np.isscalar(vals))
        vals = self.rg.hypergeometric(np.array([25] * 10), 25, 20)
        assert_(vals.shape == (10,))

    def test_triangular(self):
        vals = self.rg.triangular(-5, 0, 5)
        assert_(np.isscalar(vals))
        vals = self.rg.triangular(-5, np.array([0] * 10), 5)
        assert_(vals.shape == (10,))

    def test_multivariate_normal(self):
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]  # diagonal covariance
        x = self.rg.multivariate_normal(mean, cov, 5000)
        assert_(x.shape == (5000, 2))
        x_zig = self.rg.multivariate_normal(mean, cov, 5000)
        assert_(x.shape == (5000, 2))
        x_inv = self.rg.multivariate_normal(mean, cov, 5000)
        assert_(x.shape == (5000, 2))
        assert_((x_zig != x_inv).any())

    def test_multinomial(self):
        vals = self.rg.multinomial(100, [1.0 / 3, 2.0 / 3])
        assert_(vals.shape == (2,))
        vals = self.rg.multinomial(100, [1.0 / 3, 2.0 / 3], size=10)
        assert_(vals.shape == (10, 2))

    def test_dirichlet(self):
        s = self.rg.dirichlet((10, 5, 3), 20)
        assert_(s.shape == (20, 3))

    def test_pickle(self):
        pick = pickle.dumps(self.rg)
        unpick = pickle.loads(pick)
        assert_(type(self.rg) == type(unpick))
        assert_(comp_state(self.rg.bit_generator.state,
                           unpick.bit_generator.state))

        pick = pickle.dumps(self.rg)
        unpick = pickle.loads(pick)
        assert_(type(self.rg) == type(unpick))
        assert_(comp_state(self.rg.bit_generator.state,
                           unpick.bit_generator.state))

    def test_seed_array(self):
        if self.seed_vector_bits is None:
            bitgen_name = self.bit_generator.__name__
            pytest.skip(f'Vector seeding is not supported by {bitgen_name}')

        if self.seed_vector_bits == 32:
            dtype = np.uint32
        else:
            dtype = np.uint64
        seed = np.array([1], dtype=dtype)
        bg = self.bit_generator(seed)
        state1 = bg.state
        bg = self.bit_generator(1)
        state2 = bg.state
        assert_(comp_state(state1, state2))

        seed = np.arange(4, dtype=dtype)
        bg = self.bit_generator(seed)
        state1 = bg.state
        bg = self.bit_generator(seed[0])
        state2 = bg.state
        assert_(not comp_state(state1, state2))

        seed = np.arange(1500, dtype=dtype)
        bg = self.bit_generator(seed)
        state1 = bg.state
        bg = self.bit_generator(seed[0])
        state2 = bg.state
        assert_(not comp_state(state1, state2))

        seed = 2 ** np.mod(np.arange(1500, dtype=dtype),
                           self.seed_vector_bits - 1) + 1
        bg = self.bit_generator(seed)
        state1 = bg.state
        bg = self.bit_generator(seed[0])
        state2 = bg.state
        assert_(not comp_state(state1, state2))

    def test_uniform_float(self):
        rg = Generator(self.bit_generator(12345))
        warmup(rg)
        state = rg.bit_generator.state
        r1 = rg.random(11, dtype=np.float32)
        rg2 = Generator(self.bit_generator())
        warmup(rg2)
        rg2.bit_generator.state = state
        r2 = rg2.random(11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))

    def test_gamma_floats(self):
        rg = Generator(self.bit_generator())
        warmup(rg)
        state = rg.bit_generator.state
        r1 = rg.standard_gamma(4.0, 11, dtype=np.float32)
        rg2 = Generator(self.bit_generator())
        warmup(rg2)
        rg2.bit_generator.state = state
        r2 = rg2.standard_gamma(4.0, 11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))

    def test_normal_floats(self):
        rg = Generator(self.bit_generator())
        warmup(rg)
        state = rg.bit_generator.state
        r1 = rg.standard_normal(11, dtype=np.float32)
        rg2 = Generator(self.bit_generator())
        warmup(rg2)
        rg2.bit_generator.state = state
        r2 = rg2.standard_normal(11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))

    def test_normal_zig_floats(self):
        rg = Generator(self.bit_generator())
        warmup(rg)
        state = rg.bit_generator.state
        r1 = rg.standard_normal(11, dtype=np.float32)
        rg2 = Generator(self.bit_generator())
        warmup(rg2)
        rg2.bit_generator.state = state
        r2 = rg2.standard_normal(11, dtype=np.float32)
        assert_array_equal(r1, r2)
        assert_equal(r1.dtype, np.float32)
        assert_(comp_state(rg.bit_generator.state, rg2.bit_generator.state))

    def test_output_fill(self):
        rg = self.rg
        state = rg.bit_generator.state
        size = (31, 7, 97)
        existing = np.empty(size)
        rg.bit_generator.state = state
        rg.standard_normal(out=existing)
        rg.bit_generator.state = state
        direct = rg.standard_normal(size=size)
        assert_equal(direct, existing)

        sized = np.empty(size)
        rg.bit_generator.state = state
        rg.standard_normal(out=sized, size=sized.shape)

        existing = np.empty(size, dtype=np.float32)
        rg.bit_generator.state = state
        rg.standard_normal(out=existing, dtype=np.float32)
        rg.bit_generator.state = state
        direct = rg.standard_normal(size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_uniform(self):
        rg = self.rg
        state = rg.bit_generator.state
        size = (31, 7, 97)
        existing = np.empty(size)
        rg.bit_generator.state = state
        rg.random(out=existing)
        rg.bit_generator.state = state
        direct = rg.random(size=size)
        assert_equal(direct, existing)

        existing = np.empty(size, dtype=np.float32)
        rg.bit_generator.state = state
        rg.random(out=existing, dtype=np.float32)
        rg.bit_generator.state = state
        direct = rg.random(size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_exponential(self):
        rg = self.rg
        state = rg.bit_generator.state
        size = (31, 7, 97)
        existing = np.empty(size)
        rg.bit_generator.state = state
        rg.standard_exponential(out=existing)
        rg.bit_generator.state = state
        direct = rg.standard_exponential(size=size)
        assert_equal(direct, existing)

        existing = np.empty(size, dtype=np.float32)
        rg.bit_generator.state = state
        rg.standard_exponential(out=existing, dtype=np.float32)
        rg.bit_generator.state = state
        direct = rg.standard_exponential(size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_gamma(self):
        rg = self.rg
        state = rg.bit_generator.state
        size = (31, 7, 97)
        existing = np.zeros(size)
        rg.bit_generator.state = state
        rg.standard_gamma(1.0, out=existing)
        rg.bit_generator.state = state
        direct = rg.standard_gamma(1.0, size=size)
        assert_equal(direct, existing)

        existing = np.zeros(size, dtype=np.float32)
        rg.bit_generator.state = state
        rg.standard_gamma(1.0, out=existing, dtype=np.float32)
        rg.bit_generator.state = state
        direct = rg.standard_gamma(1.0, size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_filling_gamma_broadcast(self):
        rg = self.rg
        state = rg.bit_generator.state
        size = (31, 7, 97)
        mu = np.arange(97.0) + 1.0
        existing = np.zeros(size)
        rg.bit_generator.state = state
        rg.standard_gamma(mu, out=existing)
        rg.bit_generator.state = state
        direct = rg.standard_gamma(mu, size=size)
        assert_equal(direct, existing)

        existing = np.zeros(size, dtype=np.float32)
        rg.bit_generator.state = state
        rg.standard_gamma(mu, out=existing, dtype=np.float32)
        rg.bit_generator.state = state
        direct = rg.standard_gamma(mu, size=size, dtype=np.float32)
        assert_equal(direct, existing)

    def test_output_fill_error(self):
        rg = self.rg
        size = (31, 7, 97)
        existing = np.empty(size)
        with pytest.raises(TypeError):
            rg.standard_normal(out=existing, dtype=np.float32)
        with pytest.raises(ValueError):
            rg.standard_normal(out=existing[::3])
        existing = np.empty(size, dtype=np.float32)
        with pytest.raises(TypeError):
            rg.standard_normal(out=existing, dtype=np.float64)

        existing = np.zeros(size, dtype=np.float32)
        with pytest.raises(TypeError):
            rg.standard_gamma(1.0, out=existing, dtype=np.float64)
        with pytest.raises(ValueError):
            rg.standard_gamma(1.0, out=existing[::3], dtype=np.float32)
        existing = np.zeros(size, dtype=np.float64)
        with pytest.raises(TypeError):
            rg.standard_gamma(1.0, out=existing, dtype=np.float32)
        with pytest.raises(ValueError):
            rg.standard_gamma(1.0, out=existing[::3])

    def test_integers_broadcast(self, dtype):
        if dtype == np.bool:
            upper = 2
            lower = 0
        else:
            info = np.iinfo(dtype)
            upper = int(info.max) + 1
            lower = info.min
        self._reset_state()
        a = self.rg.integers(lower, [upper] * 10, dtype=dtype)
        self._reset_state()
        b = self.rg.integers([lower] * 10, upper, dtype=dtype)
        assert_equal(a, b)
        self._reset_state()
        c = self.rg.integers(lower, upper, size=10, dtype=dtype)
        assert_equal(a, c)
        self._reset_state()
        d = self.rg.integers(np.array(
            [lower] * 10), np.array([upper], dtype=object), size=10,
            dtype=dtype)
        assert_equal(a, d)
        self._reset_state()
        e = self.rg.integers(
            np.array([lower] * 10), np.array([upper] * 10), size=10,
            dtype=dtype)
        assert_equal(a, e)

        self._reset_state()
        a = self.rg.integers(0, upper, size=10, dtype=dtype)
        self._reset_state()
        b = self.rg.integers([upper] * 10, dtype=dtype)
        assert_equal(a, b)

    def test_integers_numpy(self, dtype):
        high = np.array([1])
        low = np.array([0])

        out = self.rg.integers(low, high, dtype=dtype)
        assert out.shape == (1,)

        out = self.rg.integers(low[0], high, dtype=dtype)
        assert out.shape == (1,)

        out = self.rg.integers(low, high[0], dtype=dtype)
        assert out.shape == (1,)

    def test_integers_broadcast_errors(self, dtype):
        if dtype == np.bool:
            upper = 2
            lower = 0
        else:
            info = np.iinfo(dtype)
            upper = int(info.max) + 1
            lower = info.min
        with pytest.raises(ValueError):
            self.rg.integers(lower, [upper + 1] * 10, dtype=dtype)
        with pytest.raises(ValueError):
            self.rg.integers(lower - 1, [upper] * 10, dtype=dtype)
        with pytest.raises(ValueError):
            self.rg.integers([lower - 1], [upper] * 10, dtype=dtype)
        with pytest.raises(ValueError):
            self.rg.integers([0], [0], dtype=dtype)


class TestMT19937(RNG):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = MT19937
        cls.advance = None
        cls.seed = [2 ** 21 + 2 ** 16 + 2 ** 5 + 1]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 32
        cls._extra_setup()
        cls.seed_error = ValueError

    def test_numpy_state(self):
        nprg = np.random.RandomState()
        nprg.standard_normal(99)
        state = nprg.get_state()
        self.rg.bit_generator.state = state
        state2 = self.rg.bit_generator.state
        assert_((state[1] == state2['state']['key']).all())
        assert_(state[2] == state2['state']['pos'])


class TestPhilox(RNG):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = Philox
        cls.advance = 2**63 + 2**31 + 2**15 + 1
        cls.seed = [12345]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 64
        cls._extra_setup()


class TestSFC64(RNG):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = SFC64
        cls.advance = None
        cls.seed = [12345]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 192
        cls._extra_setup()


class TestPCG64(RNG):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64
        cls.advance = 2**63 + 2**31 + 2**15 + 1
        cls.seed = [12345]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 64
        cls._extra_setup()


class TestPCG64DXSM(RNG):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64DXSM
        cls.advance = 2**63 + 2**31 + 2**15 + 1
        cls.seed = [12345]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 64
        cls._extra_setup()


class TestDefaultRNG(RNG):
    @classmethod
    def setup_class(cls):
        # This will duplicate some tests that directly instantiate a fresh
        # Generator(), but that's okay.
        cls.bit_generator = PCG64
        cls.advance = 2**63 + 2**31 + 2**15 + 1
        cls.seed = [12345]
        cls.rg = np.random.default_rng(*cls.seed)
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 64
        cls._extra_setup()

    def test_default_is_pcg64(self):
        # In order to change the default BitGenerator, we'll go through
        # a deprecation cycle to move to a different function.
        assert_(isinstance(self.rg.bit_generator, PCG64))

    def test_seed(self):
        np.random.default_rng()
        np.random.default_rng(None)
        np.random.default_rng(12345)
        np.random.default_rng(0)
        np.random.default_rng(43660444402423911716352051725018508569)
        np.random.default_rng([43660444402423911716352051725018508569,
                               279705150948142787361475340226491943209])
        with pytest.raises(ValueError):
            np.random.default_rng(-1)
        with pytest.raises(ValueError):
            np.random.default_rng([12345, -1])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_casting_unittests.py ===
"""
The tests exercise the casting machinery in a more low-level manner.
The reason is mostly to test a new implementation of the casting machinery.

Unlike most tests in NumPy, these are closer to unit-tests rather
than integration tests.
"""

import ctypes
import enum
import random
import textwrap

import pytest
from numpy._core._multiarray_umath import _get_castingimpl as get_castingimpl

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal

# Simple skips object, parametric and long double (unsupported by struct)
simple_dtypes = "?bhilqBHILQefdFD"
if np.dtype("l").itemsize != np.dtype("q").itemsize:
    # Remove l and L, the table was generated with 64bit linux in mind.
    simple_dtypes = simple_dtypes.replace("l", "").replace("L", "")
simple_dtypes = [type(np.dtype(c)) for c in simple_dtypes]


def simple_dtype_instances():
    for dtype_class in simple_dtypes:
        dt = dtype_class()
        yield pytest.param(dt, id=str(dt))
        if dt.byteorder != "|":
            dt = dt.newbyteorder()
            yield pytest.param(dt, id=str(dt))


def get_expected_stringlength(dtype):
    """Returns the string length when casting the basic dtypes to strings.
    """
    if dtype == np.bool:
        return 5
    if dtype.kind in "iu":
        if dtype.itemsize == 1:
            length = 3
        elif dtype.itemsize == 2:
            length = 5
        elif dtype.itemsize == 4:
            length = 10
        elif dtype.itemsize == 8:
            length = 20
        else:
            raise AssertionError(f"did not find expected length for {dtype}")

        if dtype.kind == "i":
            length += 1  # adds one character for the sign

        return length

    # Note: Can't do dtype comparison for longdouble on windows
    if dtype.char == "g":
        return 48
    elif dtype.char == "G":
        return 48 * 2
    elif dtype.kind == "f":
        return 32  # also for half apparently.
    elif dtype.kind == "c":
        return 32 * 2

    raise AssertionError(f"did not find expected length for {dtype}")


class Casting(enum.IntEnum):
    no = 0
    equiv = 1
    safe = 2
    same_kind = 3
    unsafe = 4


def _get_cancast_table():
    table = textwrap.dedent("""
        X ? b h i l q B H I L Q e f d g F D G S U V O M m
        ? # = = = = = = = = = = = = = = = = = = = = = . =
        b . # = = = = . . . . . = = = = = = = = = = = . =
        h . ~ # = = = . . . . . ~ = = = = = = = = = = . =
        i . ~ ~ # = = . . . . . ~ ~ = = ~ = = = = = = . =
        l . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        q . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        B . ~ = = = = # = = = = = = = = = = = = = = = . =
        H . ~ ~ = = = ~ # = = = ~ = = = = = = = = = = . =
        I . ~ ~ ~ = = ~ ~ # = = ~ ~ = = ~ = = = = = = . =
        L . ~ ~ ~ ~ ~ ~ ~ ~ # # ~ ~ = = ~ = = = = = = . ~
        Q . ~ ~ ~ ~ ~ ~ ~ ~ # # ~ ~ = = ~ = = = = = = . ~
        e . . . . . . . . . . . # = = = = = = = = = = . .
        f . . . . . . . . . . . ~ # = = = = = = = = = . .
        d . . . . . . . . . . . ~ ~ # = ~ = = = = = = . .
        g . . . . . . . . . . . ~ ~ ~ # ~ ~ = = = = = . .
        F . . . . . . . . . . . . . . . # = = = = = = . .
        D . . . . . . . . . . . . . . . ~ # = = = = = . .
        G . . . . . . . . . . . . . . . ~ ~ # = = = = . .
        S . . . . . . . . . . . . . . . . . . # = = = . .
        U . . . . . . . . . . . . . . . . . . . # = = . .
        V . . . . . . . . . . . . . . . . . . . . # = . .
        O . . . . . . . . . . . . . . . . . . . . = # . .
        M . . . . . . . . . . . . . . . . . . . . = = # .
        m . . . . . . . . . . . . . . . . . . . . = = . #
        """).strip().split("\n")
    dtypes = [type(np.dtype(c)) for c in table[0][2::2]]

    convert_cast = {".": Casting.unsafe, "~": Casting.same_kind,
                    "=": Casting.safe, "#": Casting.equiv,
                    " ": -1}

    cancast = {}
    for from_dt, row in zip(dtypes, table[1:]):
        cancast[from_dt] = {}
        for to_dt, c in zip(dtypes, row[2::2]):
            cancast[from_dt][to_dt] = convert_cast[c]

    return cancast


CAST_TABLE = _get_cancast_table()


class TestChanges:
    """
    These test cases exercise some behaviour changes
    """
    @pytest.mark.parametrize("string", ["S", "U"])
    @pytest.mark.parametrize("floating", ["e", "f", "d", "g"])
    def test_float_to_string(self, floating, string):
        assert np.can_cast(floating, string)
        # 100 is long enough to hold any formatted floating
        assert np.can_cast(floating, f"{string}100")

    def test_to_void(self):
        # But in general, we do consider these safe:
        assert np.can_cast("d", "V")
        assert np.can_cast("S20", "V")

        # Do not consider it a safe cast if the void is too smaller:
        assert not np.can_cast("d", "V1")
        assert not np.can_cast("S20", "V1")
        assert not np.can_cast("U1", "V1")
        # Structured to unstructured is just like any other:
        assert np.can_cast("d,i", "V", casting="same_kind")
        # Unstructured void to unstructured is actually no cast at all:
        assert np.can_cast("V3", "V", casting="no")
        assert np.can_cast("V0", "V", casting="no")


class TestCasting:
    size = 1500  # Best larger than NPY_LOWLEVEL_BUFFER_BLOCKSIZE * itemsize

    def get_data(self, dtype1, dtype2):
        if dtype2 is None or dtype1.itemsize >= dtype2.itemsize:
            length = self.size // dtype1.itemsize
        else:
            length = self.size // dtype2.itemsize

        # Assume that the base array is well enough aligned for all inputs.
        arr1 = np.empty(length, dtype=dtype1)
        assert arr1.flags.c_contiguous
        assert arr1.flags.aligned

        values = [random.randrange(-128, 128) for _ in range(length)]

        for i, value in enumerate(values):
            # Use item assignment to ensure this is not using casting:
            if value < 0 and dtype1.kind == "u":
                # Manually rollover unsigned integers (-1 -> int.max)
                value = value + np.iinfo(dtype1).max + 1
            arr1[i] = value

        if dtype2 is None:
            if dtype1.char == "?":
                values = [bool(v) for v in values]
            return arr1, values

        if dtype2.char == "?":
            values = [bool(v) for v in values]

        arr2 = np.empty(length, dtype=dtype2)
        assert arr2.flags.c_contiguous
        assert arr2.flags.aligned

        for i, value in enumerate(values):
            # Use item assignment to ensure this is not using casting:
            if value < 0 and dtype2.kind == "u":
                # Manually rollover unsigned integers (-1 -> int.max)
                value = value + np.iinfo(dtype2).max + 1
            arr2[i] = value

        return arr1, arr2, values

    def get_data_variation(self, arr1, arr2, aligned=True, contig=True):
        """
        Returns a copy of arr1 that may be non-contiguous or unaligned, and a
        matching array for arr2 (although not a copy).
        """
        if contig:
            stride1 = arr1.dtype.itemsize
            stride2 = arr2.dtype.itemsize
        elif aligned:
            stride1 = 2 * arr1.dtype.itemsize
            stride2 = 2 * arr2.dtype.itemsize
        else:
            stride1 = arr1.dtype.itemsize + 1
            stride2 = arr2.dtype.itemsize + 1

        max_size1 = len(arr1) * 3 * arr1.dtype.itemsize + 1
        max_size2 = len(arr2) * 3 * arr2.dtype.itemsize + 1
        from_bytes = np.zeros(max_size1, dtype=np.uint8)
        to_bytes = np.zeros(max_size2, dtype=np.uint8)

        # Sanity check that the above is large enough:
        assert stride1 * len(arr1) <= from_bytes.nbytes
        assert stride2 * len(arr2) <= to_bytes.nbytes

        if aligned:
            new1 = as_strided(from_bytes[:-1].view(arr1.dtype),
                              arr1.shape, (stride1,))
            new2 = as_strided(to_bytes[:-1].view(arr2.dtype),
                              arr2.shape, (stride2,))
        else:
            new1 = as_strided(from_bytes[1:].view(arr1.dtype),
                              arr1.shape, (stride1,))
            new2 = as_strided(to_bytes[1:].view(arr2.dtype),
                              arr2.shape, (stride2,))

        new1[...] = arr1

        if not contig:
            # Ensure we did not overwrite bytes that should not be written:
            offset = arr1.dtype.itemsize if aligned else 0
            buf = from_bytes[offset::stride1].tobytes()
            assert buf.count(b"\0") == len(buf)

        if contig:
            assert new1.flags.c_contiguous
            assert new2.flags.c_contiguous
        else:
            assert not new1.flags.c_contiguous
            assert not new2.flags.c_contiguous

        if aligned:
            assert new1.flags.aligned
            assert new2.flags.aligned
        else:
            assert not new1.flags.aligned or new1.dtype.alignment == 1
            assert not new2.flags.aligned or new2.dtype.alignment == 1

        return new1, new2

    @pytest.mark.parametrize("from_Dt", simple_dtypes)
    def test_simple_cancast(self, from_Dt):
        for to_Dt in simple_dtypes:
            cast = get_castingimpl(from_Dt, to_Dt)

            for from_dt in [from_Dt(), from_Dt().newbyteorder()]:
                default = cast._resolve_descriptors((from_dt, None))[1][1]
                assert default == to_Dt()
                del default

                for to_dt in [to_Dt(), to_Dt().newbyteorder()]:
                    casting, (from_res, to_res), view_off = (
                            cast._resolve_descriptors((from_dt, to_dt)))
                    assert type(from_res) == from_Dt
                    assert type(to_res) == to_Dt
                    if view_off is not None:
                        # If a view is acceptable, this is "no" casting
                        # and byte order must be matching.
                        assert casting == Casting.no
                        # The above table lists this as "equivalent"
                        assert Casting.equiv == CAST_TABLE[from_Dt][to_Dt]
                        # Note that to_res may not be the same as from_dt
                        assert from_res.isnative == to_res.isnative
                    else:
                        if from_Dt == to_Dt:
                            # Note that to_res may not be the same as from_dt
                            assert from_res.isnative != to_res.isnative
                        assert casting == CAST_TABLE[from_Dt][to_Dt]

                    if from_Dt is to_Dt:
                        assert from_dt is from_res
                        assert to_dt is to_res

    @pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
    @pytest.mark.parametrize("from_dt", simple_dtype_instances())
    def test_simple_direct_casts(self, from_dt):
        """
        This test checks numeric direct casts for dtypes supported also by the
        struct module (plus complex).  It tries to be test a wide range of
        inputs, but skips over possibly undefined behaviour (e.g. int rollover).
        Longdouble and CLongdouble are tested, but only using double precision.

        If this test creates issues, it should possibly just be simplified
        or even removed (checking whether unaligned/non-contiguous casts give
        the same results is useful, though).
        """
        for to_dt in simple_dtype_instances():
            to_dt = to_dt.values[0]
            cast = get_castingimpl(type(from_dt), type(to_dt))

            casting, (from_res, to_res), view_off = cast._resolve_descriptors(
                (from_dt, to_dt))

            if from_res is not from_dt or to_res is not to_dt:
                # Do not test this case, it is handled in multiple steps,
                # each of which should is tested individually.
                return

            safe = casting <= Casting.safe
            del from_res, to_res, casting

            arr1, arr2, values = self.get_data(from_dt, to_dt)

            cast._simple_strided_call((arr1, arr2))

            # Check via python list
            assert arr2.tolist() == values

            # Check that the same results are achieved for strided loops
            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, True, False)
            cast._simple_strided_call((arr1_o, arr2_o))

            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()

            # Check if alignment makes a difference, but only if supported
            # and only if the alignment can be wrong
            if ((from_dt.alignment == 1 and to_dt.alignment == 1) or
                    not cast._supports_unaligned):
                return

            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, False, True)
            cast._simple_strided_call((arr1_o, arr2_o))

            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()

            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, False, False)
            cast._simple_strided_call((arr1_o, arr2_o))

            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()

            del arr1_o, arr2_o, cast

    @pytest.mark.parametrize("from_Dt", simple_dtypes)
    def test_numeric_to_times(self, from_Dt):
        # We currently only implement contiguous loops, so only need to
        # test those.
        from_dt = from_Dt()

        time_dtypes = [np.dtype("M8"), np.dtype("M8[ms]"), np.dtype("M8[4D]"),
                       np.dtype("m8"), np.dtype("m8[ms]"), np.dtype("m8[4D]")]
        for time_dt in time_dtypes:
            cast = get_castingimpl(type(from_dt), type(time_dt))

            casting, (from_res, to_res), view_off = cast._resolve_descriptors(
                (from_dt, time_dt))

            assert from_res is from_dt
            assert to_res is time_dt
            del from_res, to_res

            assert casting & CAST_TABLE[from_Dt][type(time_dt)]
            assert view_off is None

            int64_dt = np.dtype(np.int64)
            arr1, arr2, values = self.get_data(from_dt, int64_dt)
            arr2 = arr2.view(time_dt)
            arr2[...] = np.datetime64("NaT")

            if time_dt == np.dtype("M8"):
                # This is a bit of a strange path, and could probably be removed
                arr1[-1] = 0  # ensure at least one value is not NaT

                # The cast currently succeeds, but the values are invalid:
                cast._simple_strided_call((arr1, arr2))
                with pytest.raises(ValueError):
                    str(arr2[-1])  # e.g. conversion to string fails
                return

            cast._simple_strided_call((arr1, arr2))

            assert [int(v) for v in arr2.tolist()] == values

            # Check that the same results are achieved for strided loops
            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, True, False)
            cast._simple_strided_call((arr1_o, arr2_o))

            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()

    @pytest.mark.parametrize(
            ["from_dt", "to_dt", "expected_casting", "expected_view_off",
             "nom", "denom"],
            [("M8[ns]", None, Casting.no, 0, 1, 1),
             (str(np.dtype("M8[ns]").newbyteorder()), None,
                  Casting.equiv, None, 1, 1),
             ("M8", "M8[ms]", Casting.safe, 0, 1, 1),
             # should be invalid cast:
             ("M8[ms]", "M8", Casting.unsafe, None, 1, 1),
             ("M8[5ms]", "M8[5ms]", Casting.no, 0, 1, 1),
             ("M8[ns]", "M8[ms]", Casting.same_kind, None, 1, 10**6),
             ("M8[ms]", "M8[ns]", Casting.safe, None, 10**6, 1),
             ("M8[ms]", "M8[7ms]", Casting.same_kind, None, 1, 7),
             ("M8[4D]", "M8[1M]", Casting.same_kind, None, None,
                  # give full values based on NumPy 1.19.x
                  [-2**63, 0, -1, 1314, -1315, 564442610]),
             ("m8[ns]", None, Casting.no, 0, 1, 1),
             (str(np.dtype("m8[ns]").newbyteorder()), None,
                  Casting.equiv, None, 1, 1),
             ("m8", "m8[ms]", Casting.safe, 0, 1, 1),
             # should be invalid cast:
             ("m8[ms]", "m8", Casting.unsafe, None, 1, 1),
             ("m8[5ms]", "m8[5ms]", Casting.no, 0, 1, 1),
             ("m8[ns]", "m8[ms]", Casting.same_kind, None, 1, 10**6),
             ("m8[ms]", "m8[ns]", Casting.safe, None, 10**6, 1),
             ("m8[ms]", "m8[7ms]", Casting.same_kind, None, 1, 7),
             ("m8[4D]", "m8[1M]", Casting.unsafe, None, None,
                  # give full values based on NumPy 1.19.x
                  [-2**63, 0, 0, 1314, -1315, 564442610])])
    def test_time_to_time(self, from_dt, to_dt,
                          expected_casting, expected_view_off,
                          nom, denom):
        from_dt = np.dtype(from_dt)
        if to_dt is not None:
            to_dt = np.dtype(to_dt)

        # Test a few values for casting (results generated with NumPy 1.19)
        values = np.array([-2**63, 1, 2**63 - 1, 10000, -10000, 2**32])
        values = values.astype(np.dtype("int64").newbyteorder(from_dt.byteorder))
        assert values.dtype.byteorder == from_dt.byteorder
        assert np.isnat(values.view(from_dt)[0])

        DType = type(from_dt)
        cast = get_castingimpl(DType, DType)
        casting, (from_res, to_res), view_off = cast._resolve_descriptors(
                (from_dt, to_dt))
        assert from_res is from_dt
        assert to_res is to_dt or to_dt is None
        assert casting == expected_casting
        assert view_off == expected_view_off

        if nom is not None:
            expected_out = (values * nom // denom).view(to_res)
            expected_out[0] = "NaT"
        else:
            expected_out = np.empty_like(values)
            expected_out[...] = denom
            expected_out = expected_out.view(to_dt)

        orig_arr = values.view(from_dt)
        orig_out = np.empty_like(expected_out)

        if casting == Casting.unsafe and (to_dt == "m8" or to_dt == "M8"):  # noqa: PLR1714
            # Casting from non-generic to generic units is an error and should
            # probably be reported as an invalid cast earlier.
            with pytest.raises(ValueError):
                cast._simple_strided_call((orig_arr, orig_out))
            return

        for aligned in [True, True]:
            for contig in [True, True]:
                arr, out = self.get_data_variation(
                        orig_arr, orig_out, aligned, contig)
                out[...] = 0
                cast._simple_strided_call((arr, out))
                assert_array_equal(out.view("int64"), expected_out.view("int64"))

    def string_with_modified_length(self, dtype, change_length):
        fact = 1 if dtype.char == "S" else 4
        length = dtype.itemsize // fact + change_length
        return np.dtype(f"{dtype.byteorder}{dtype.char}{length}")

    @pytest.mark.parametrize("other_DT", simple_dtypes)
    @pytest.mark.parametrize("string_char", ["S", "U"])
    def test_string_cancast(self, other_DT, string_char):
        fact = 1 if string_char == "S" else 4

        string_DT = type(np.dtype(string_char))
        cast = get_castingimpl(other_DT, string_DT)

        other_dt = other_DT()
        expected_length = get_expected_stringlength(other_dt)
        string_dt = np.dtype(f"{string_char}{expected_length}")

        safety, (res_other_dt, res_dt), view_off = cast._resolve_descriptors(
                (other_dt, None))
        assert res_dt.itemsize == expected_length * fact
        assert safety == Casting.safe  # we consider to string casts "safe"
        assert view_off is None
        assert isinstance(res_dt, string_DT)

        # These casts currently implement changing the string length, so
        # check the cast-safety for too long/fixed string lengths:
        for change_length in [-1, 0, 1]:
            if change_length >= 0:
                expected_safety = Casting.safe
            else:
                expected_safety = Casting.same_kind

            to_dt = self.string_with_modified_length(string_dt, change_length)
            safety, (_, res_dt), view_off = cast._resolve_descriptors(
                    (other_dt, to_dt))
            assert res_dt is to_dt
            assert safety == expected_safety
            assert view_off is None

        # The opposite direction is always considered unsafe:
        cast = get_castingimpl(string_DT, other_DT)

        safety, _, view_off = cast._resolve_descriptors((string_dt, other_dt))
        assert safety == Casting.unsafe
        assert view_off is None

        cast = get_castingimpl(string_DT, other_DT)
        safety, (_, res_dt), view_off = cast._resolve_descriptors(
            (string_dt, None))
        assert safety == Casting.unsafe
        assert view_off is None
        assert other_dt is res_dt  # returns the singleton for simple dtypes

    @pytest.mark.parametrize("string_char", ["S", "U"])
    @pytest.mark.parametrize("other_dt", simple_dtype_instances())
    def test_simple_string_casts_roundtrip(self, other_dt, string_char):
        """
        Tests casts from and to string by checking the roundtripping property.

        The test also covers some string to string casts (but not all).

        If this test creates issues, it should possibly just be simplified
        or even removed (checking whether unaligned/non-contiguous casts give
        the same results is useful, though).
        """
        string_DT = type(np.dtype(string_char))

        cast = get_castingimpl(type(other_dt), string_DT)
        cast_back = get_castingimpl(string_DT, type(other_dt))
        _, (res_other_dt, string_dt), _ = cast._resolve_descriptors(
                (other_dt, None))

        if res_other_dt is not other_dt:
            # do not support non-native byteorder, skip test in that case
            assert other_dt.byteorder != res_other_dt.byteorder
            return

        orig_arr, values = self.get_data(other_dt, None)
        str_arr = np.zeros(len(orig_arr), dtype=string_dt)
        string_dt_short = self.string_with_modified_length(string_dt, -1)
        str_arr_short = np.zeros(len(orig_arr), dtype=string_dt_short)
        string_dt_long = self.string_with_modified_length(string_dt, 1)
        str_arr_long = np.zeros(len(orig_arr), dtype=string_dt_long)

        assert not cast._supports_unaligned  # if support is added, should test
        assert not cast_back._supports_unaligned

        for contig in [True, False]:
            other_arr, str_arr = self.get_data_variation(
                orig_arr, str_arr, True, contig)
            _, str_arr_short = self.get_data_variation(
                orig_arr, str_arr_short.copy(), True, contig)
            _, str_arr_long = self.get_data_variation(
                orig_arr, str_arr_long, True, contig)

            cast._simple_strided_call((other_arr, str_arr))

            cast._simple_strided_call((other_arr, str_arr_short))
            assert_array_equal(str_arr.astype(string_dt_short), str_arr_short)

            cast._simple_strided_call((other_arr, str_arr_long))
            assert_array_equal(str_arr, str_arr_long)

            if other_dt.kind == "b":
                # Booleans do not roundtrip
                continue

            other_arr[...] = 0
            cast_back._simple_strided_call((str_arr, other_arr))
            assert_array_equal(orig_arr, other_arr)

            other_arr[...] = 0
            cast_back._simple_strided_call((str_arr_long, other_arr))
            assert_array_equal(orig_arr, other_arr)

    @pytest.mark.parametrize("other_dt", ["S8", "<U8", ">U8"])
    @pytest.mark.parametrize("string_char", ["S", "U"])
    def test_string_to_string_cancast(self, other_dt, string_char):
        other_dt = np.dtype(other_dt)

        fact = 1 if string_char == "S" else 4
        div = 1 if other_dt.char == "S" else 4

        string_DT = type(np.dtype(string_char))
        cast = get_castingimpl(type(other_dt), string_DT)

        expected_length = other_dt.itemsize // div
        string_dt = np.dtype(f"{string_char}{expected_length}")

        safety, (res_other_dt, res_dt), view_off = cast._resolve_descriptors(
                (other_dt, None))
        assert res_dt.itemsize == expected_length * fact
        assert isinstance(res_dt, string_DT)

        expected_view_off = None
        if other_dt.char == string_char:
            if other_dt.isnative:
                expected_safety = Casting.no
                expected_view_off = 0
            else:
                expected_safety = Casting.equiv
        elif string_char == "U":
            expected_safety = Casting.safe
        else:
            expected_safety = Casting.unsafe

        assert view_off == expected_view_off
        assert expected_safety == safety

        for change_length in [-1, 0, 1]:
            to_dt = self.string_with_modified_length(string_dt, change_length)
            safety, (_, res_dt), view_off = cast._resolve_descriptors(
                    (other_dt, to_dt))

            assert res_dt is to_dt
            if change_length <= 0:
                assert view_off == expected_view_off
            else:
                assert view_off is None
            if expected_safety == Casting.unsafe:
                assert safety == expected_safety
            elif change_length < 0:
                assert safety == Casting.same_kind
            elif change_length == 0:
                assert safety == expected_safety
            elif change_length > 0:
                assert safety == Casting.safe

    @pytest.mark.parametrize("order1", [">", "<"])
    @pytest.mark.parametrize("order2", [">", "<"])
    def test_unicode_byteswapped_cast(self, order1, order2):
        # Very specific tests (not using the castingimpl directly)
        # that tests unicode bytedwaps including for unaligned array data.
        dtype1 = np.dtype(f"{order1}U30")
        dtype2 = np.dtype(f"{order2}U30")
        data1 = np.empty(30 * 4 + 1, dtype=np.uint8)[1:].view(dtype1)
        data2 = np.empty(30 * 4 + 1, dtype=np.uint8)[1:].view(dtype2)
        if dtype1.alignment != 1:
            # alignment should always be >1, but skip the check if not
            assert not data1.flags.aligned
            assert not data2.flags.aligned

        element = "this is a ünicode string‽"
        data1[()] = element
        # Test both `data1` and `data1.copy()`  (which should be aligned)
        for data in [data1, data1.copy()]:
            data2[...] = data1
            assert data2[()] == element
            assert data2.copy()[()] == element

    def test_void_to_string_special_case(self):
        # Cover a small special case in void to string casting that could
        # probably just as well be turned into an error (compare
        # `test_object_to_parametric_internal_error` below).
        assert np.array([], dtype="V5").astype("S").dtype.itemsize == 5
        assert np.array([], dtype="V5").astype("U").dtype.itemsize == 4 * 5

    def test_object_to_parametric_internal_error(self):
        # We reject casting from object to a parametric type, without
        # figuring out the correct instance first.
        object_dtype = type(np.dtype(object))
        other_dtype = type(np.dtype(str))
        cast = get_castingimpl(object_dtype, other_dtype)
        with pytest.raises(TypeError,
                    match="casting from object to the parametric DType"):
            cast._resolve_descriptors((np.dtype("O"), None))

    @pytest.mark.parametrize("dtype", simple_dtype_instances())
    def test_object_and_simple_resolution(self, dtype):
        # Simple test to exercise the cast when no instance is specified
        object_dtype = type(np.dtype(object))
        cast = get_castingimpl(object_dtype, type(dtype))

        safety, (_, res_dt), view_off = cast._resolve_descriptors(
                (np.dtype("O"), dtype))
        assert safety == Casting.unsafe
        assert view_off is None
        assert res_dt is dtype

        safety, (_, res_dt), view_off = cast._resolve_descriptors(
                (np.dtype("O"), None))
        assert safety == Casting.unsafe
        assert view_off is None
        assert res_dt == dtype.newbyteorder("=")

    @pytest.mark.parametrize("dtype", simple_dtype_instances())
    def test_simple_to_object_resolution(self, dtype):
        # Simple test to exercise the cast when no instance is specified
        object_dtype = type(np.dtype(object))
        cast = get_castingimpl(type(dtype), object_dtype)

        safety, (_, res_dt), view_off = cast._resolve_descriptors(
                (dtype, None))
        assert safety == Casting.safe
        assert view_off is None
        assert res_dt is np.dtype("O")

    @pytest.mark.parametrize("casting", ["no", "unsafe"])
    def test_void_and_structured_with_subarray(self, casting):
        # test case corresponding to gh-19325
        dtype = np.dtype([("foo", "<f4", (3, 2))])
        expected = casting == "unsafe"
        assert np.can_cast("V4", dtype, casting=casting) == expected
        assert np.can_cast(dtype, "V4", casting=casting) == expected

    @pytest.mark.parametrize(["to_dt", "expected_off"],
            [  # Same as `from_dt` but with both fields shifted:
             (np.dtype({"names": ["a", "b"], "formats": ["i4", "f4"],
                        "offsets": [0, 4]}), 2),
             # Additional change of the names
             (np.dtype({"names": ["b", "a"], "formats": ["i4", "f4"],
                        "offsets": [0, 4]}), 2),
             # Incompatible field offset change
             (np.dtype({"names": ["b", "a"], "formats": ["i4", "f4"],
                        "offsets": [0, 6]}), None)])
    def test_structured_field_offsets(self, to_dt, expected_off):
        # This checks the cast-safety and view offset for swapped and "shifted"
        # fields which are viewable
        from_dt = np.dtype({"names": ["a", "b"],
                            "formats": ["i4", "f4"],
                            "offsets": [2, 6]})
        cast = get_castingimpl(type(from_dt), type(to_dt))
        safety, _, view_off = cast._resolve_descriptors((from_dt, to_dt))
        if from_dt.names == to_dt.names:
            assert safety == Casting.equiv
        else:
            assert safety == Casting.safe
        # Shifting the original data pointer by -2 will align both by
        # effectively adding 2 bytes of spacing before `from_dt`.
        assert view_off == expected_off

    @pytest.mark.parametrize(("from_dt", "to_dt", "expected_off"), [
            # Subarray cases:
            ("i", "(1,1)i", 0),
            ("(1,1)i", "i", 0),
            ("(2,1)i", "(2,1)i", 0),
            # field cases (field to field is tested explicitly also):
            # Not considered viewable, because a negative offset would allow
            # may structured dtype to indirectly access invalid memory.
            ("i", {"names": ["a"], "formats": ["i"], "offsets": [2]}, None),
            ({"names": ["a"], "formats": ["i"], "offsets": [2]}, "i", 2),
            # Currently considered not viewable, due to multiple fields
            # even though they overlap (maybe we should not allow that?)
            ("i", {"names": ["a", "b"], "formats": ["i", "i"], "offsets": [2, 2]},
             None),
            # different number of fields can't work, should probably just fail
            # so it never reports "viewable":
            ("i,i", "i,i,i", None),
            # Unstructured void cases:
            ("i4", "V3", 0),  # void smaller or equal
            ("i4", "V4", 0),  # void smaller or equal
            ("i4", "V10", None),  # void is larger (no view)
            ("O", "V4", None),  # currently reject objects for view here.
            ("O", "V8", None),  # currently reject objects for view here.
            ("V4", "V3", 0),
            ("V4", "V4", 0),
            ("V3", "V4", None),
            # Note that currently void-to-other cast goes via byte-strings
            # and is not a "view" based cast like the opposite direction:
            ("V4", "i4", None),
            # completely invalid/impossible cast:
            ("i,i", "i,i,i", None),
        ])
    def test_structured_view_offsets_parametric(
            self, from_dt, to_dt, expected_off):
        # TODO: While this test is fairly thorough, right now, it does not
        # really test some paths that may have nonzero offsets (they don't
        # really exists).
        from_dt = np.dtype(from_dt)
        to_dt = np.dtype(to_dt)
        cast = get_castingimpl(type(from_dt), type(to_dt))
        _, _, view_off = cast._resolve_descriptors((from_dt, to_dt))
        assert view_off == expected_off

    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_object_casts_NULL_None_equivalence(self, dtype):
        # None to <other> casts may succeed or fail, but a NULL'ed array must
        # behave the same as one filled with None's.
        arr_normal = np.array([None] * 5)
        arr_NULLs = np.empty_like(arr_normal)
        ctypes.memset(arr_NULLs.ctypes.data, 0, arr_NULLs.nbytes)
        # If the check fails (maybe it should) the test would lose its purpose:
        assert arr_NULLs.tobytes() == b"\x00" * arr_NULLs.nbytes

        try:
            expected = arr_normal.astype(dtype)
        except TypeError:
            with pytest.raises(TypeError):
                arr_NULLs.astype(dtype)
        else:
            assert_array_equal(expected, arr_NULLs.astype(dtype))

    @pytest.mark.parametrize("dtype",
            np.typecodes["AllInteger"] + np.typecodes["AllFloat"])
    def test_nonstandard_bool_to_other(self, dtype):
        # simple test for casting bool_ to numeric types, which should not
        # expose the detail that NumPy bools can sometimes take values other
        # than 0 and 1.  See also gh-19514.
        nonstandard_bools = np.array([0, 3, -7], dtype=np.int8).view(bool)
        res = nonstandard_bools.astype(dtype)
        expected = [0, 1, 1]
        assert_array_equal(res, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_replace.py ===
import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


class TestSeriesReplace:
    def test_replace_explicit_none(self):
        # GH#36984 if the user explicitly passes value=None, give it to them
        ser = pd.Series([0, 0, ""], dtype=object)
        result = ser.replace("", None)
        expected = pd.Series([0, 0, None], dtype=object)
        tm.assert_series_equal(result, expected)

        # Cast column 2 to object to avoid implicit cast when setting entry to ""
        df = pd.DataFrame(np.zeros((3, 3))).astype({2: object})
        df.iloc[2, 2] = ""
        result = df.replace("", None)
        expected = pd.DataFrame(
            {
                0: np.zeros(3),
                1: np.zeros(3),
                2: np.array([0.0, 0.0, None], dtype=object),
            }
        )
        assert expected.iloc[2, 2] is None
        tm.assert_frame_equal(result, expected)

        # GH#19998 same thing with object dtype
        ser = pd.Series([10, 20, 30, "a", "a", "b", "a"])
        result = ser.replace("a", None)
        expected = pd.Series([10, 20, 30, None, None, "b", None])
        assert expected.iloc[-1] is None
        tm.assert_series_equal(result, expected)

    def test_replace_noop_doesnt_downcast(self):
        # GH#44498
        ser = pd.Series([None, None, pd.Timestamp("2021-12-16 17:31")], dtype=object)
        res = ser.replace({np.nan: None})  # should be a no-op
        tm.assert_series_equal(res, ser)
        assert res.dtype == object

        # same thing but different calling convention
        res = ser.replace(np.nan, None)
        tm.assert_series_equal(res, ser)
        assert res.dtype == object

    def test_replace(self):
        N = 50
        ser = pd.Series(np.random.default_rng(2).standard_normal(N))
        ser[0:4] = np.nan
        ser[6:10] = 0

        # replace list with a single value
        return_value = ser.replace([np.nan], -1, inplace=True)
        assert return_value is None

        exp = ser.fillna(-1)
        tm.assert_series_equal(ser, exp)

        rs = ser.replace(0.0, np.nan)
        ser[ser == 0.0] = np.nan
        tm.assert_series_equal(rs, ser)

        ser = pd.Series(
            np.fabs(np.random.default_rng(2).standard_normal(N)),
            pd.date_range("2020-01-01", periods=N),
            dtype=object,
        )
        ser[:5] = np.nan
        ser[6:10] = "foo"
        ser[20:30] = "bar"

        # replace list with a single value
        msg = "Downcasting behavior in `replace`"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = ser.replace([np.nan, "foo", "bar"], -1)

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values with 2 lists
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)

        # replace inplace
        with tm.assert_produces_warning(FutureWarning, match=msg):
            return_value = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        assert return_value is None

        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

    def test_replace_nan_with_inf(self):
        ser = pd.Series([np.nan, 0, np.inf])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        ser = pd.Series([np.nan, 0, "foo", "bar", np.inf, None, pd.NaT])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
        filled = ser.copy()
        filled[4] = 0
        tm.assert_series_equal(ser.replace(np.inf, 0), filled)

    def test_replace_listlike_value_listlike_target(self, datetime_series):
        ser = pd.Series(datetime_series.index)
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        # malformed
        msg = r"Replacement lists must match in length\. Expecting 3 got 2"
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3], [np.nan, 0])

        # ser is dt64 so can't hold 1 or 2, so this replace is a no-op
        result = ser.replace([1, 2], [np.nan, 0])
        tm.assert_series_equal(result, ser)

        ser = pd.Series([0, 1, 2, 3, 4])
        result = ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
        tm.assert_series_equal(result, pd.Series([4, 3, 2, 1, 0]))

    def test_replace_gh5319(self):
        # API change from 0.12?
        # GH 5319
        ser = pd.Series([0, np.nan, 2, 3, 4])
        expected = ser.ffill()
        msg = (
            "Series.replace without 'value' and with non-dict-like "
            "'to_replace' is deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.replace([np.nan])
        tm.assert_series_equal(result, expected)

        ser = pd.Series([0, np.nan, 2, 3, 4])
        expected = ser.ffill()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.replace(np.nan)
        tm.assert_series_equal(result, expected)

    def test_replace_datetime64(self):
        # GH 5797
        ser = pd.Series(pd.date_range("20130101", periods=5))
        expected = ser.copy()
        expected.loc[2] = pd.Timestamp("20120101")
        result = ser.replace({pd.Timestamp("20130103"): pd.Timestamp("20120101")})
        tm.assert_series_equal(result, expected)
        result = ser.replace(pd.Timestamp("20130103"), pd.Timestamp("20120101"))
        tm.assert_series_equal(result, expected)

    def test_replace_nat_with_tz(self):
        # GH 11792: Test with replacing NaT in a list with tz data
        ts = pd.Timestamp("2015/01/01", tz="UTC")
        s = pd.Series([pd.NaT, pd.Timestamp("2015/01/01", tz="UTC")])
        result = s.replace([np.nan, pd.NaT], pd.Timestamp.min)
        expected = pd.Series([pd.Timestamp.min, ts], dtype=object)
        tm.assert_series_equal(expected, result)

    def test_replace_timedelta_td64(self):
        tdi = pd.timedelta_range(0, periods=5)
        ser = pd.Series(tdi)

        # Using a single dict argument means we go through replace_list
        result = ser.replace({ser[1]: ser[3]})

        expected = pd.Series([ser[0], ser[3], ser[2], ser[3], ser[4]])
        tm.assert_series_equal(result, expected)

    def test_replace_with_single_list(self):
        ser = pd.Series([0, 1, 2, 3, 4])
        msg2 = (
            "Series.replace without 'value' and with non-dict-like "
            "'to_replace' is deprecated"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            result = ser.replace([1, 2, 3])
        tm.assert_series_equal(result, pd.Series([0, 0, 0, 0, 4]))

        s = ser.copy()
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            return_value = s.replace([1, 2, 3], inplace=True)
        assert return_value is None
        tm.assert_series_equal(s, pd.Series([0, 0, 0, 0, 4]))

        # make sure things don't get corrupted when fillna call fails
        s = ser.copy()
        msg = (
            r"Invalid fill method\. Expecting pad \(ffill\) or backfill "
            r"\(bfill\)\. Got crash_cymbal"
        )
        msg3 = "The 'method' keyword in Series.replace is deprecated"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg3):
                return_value = s.replace([1, 2, 3], inplace=True, method="crash_cymbal")
            assert return_value is None
        tm.assert_series_equal(s, ser)

    def test_replace_mixed_types(self):
        ser = pd.Series(np.arange(5), dtype="int64")

        def check_replace(to_rep, val, expected):
            sc = ser.copy()
            result = ser.replace(to_rep, val)
            return_value = sc.replace(to_rep, val, inplace=True)
            assert return_value is None
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(expected, sc)

        # 3.0 can still be held in our int64 series, so we do not upcast GH#44940
        tr, v = [3], [3.0]
        check_replace(tr, v, ser)
        # Note this matches what we get with the scalars 3 and 3.0
        check_replace(tr[0], v[0], ser)

        # MUST upcast to float
        e = pd.Series([0, 1, 2, 3.5, 4])
        tr, v = [3], [3.5]
        check_replace(tr, v, e)

        # casts to object
        e = pd.Series([0, 1, 2, 3.5, "a"])
        tr, v = [3, 4], [3.5, "a"]
        check_replace(tr, v, e)

        # again casts to object
        e = pd.Series([0, 1, 2, 3.5, pd.Timestamp("20130101")])
        tr, v = [3, 4], [3.5, pd.Timestamp("20130101")]
        check_replace(tr, v, e)

        # casts to object
        e = pd.Series([0, 1, 2, 3.5, True], dtype="object")
        tr, v = [3, 4], [3.5, True]
        check_replace(tr, v, e)

        # test an object with dates + floats + integers + strings
        dr = pd.Series(pd.date_range("1/1/2001", "1/10/2001", freq="D"))
        result = dr.astype(object).replace([dr[0], dr[1], dr[2]], [1.0, 2, "a"])
        expected = pd.Series([1.0, 2, "a"] + dr[3:].tolist(), dtype=object)
        tm.assert_series_equal(result, expected)

    def test_replace_bool_with_string_no_op(self):
        s = pd.Series([True, False, True])
        result = s.replace("fun", "in-the-sun")
        tm.assert_series_equal(s, result)

    def test_replace_bool_with_string(self):
        # nonexistent elements
        s = pd.Series([True, False, True])
        result = s.replace(True, "2u")
        expected = pd.Series(["2u", False, "2u"])
        tm.assert_series_equal(expected, result)

    def test_replace_bool_with_bool(self):
        s = pd.Series([True, False, True])
        result = s.replace(True, False)
        expected = pd.Series([False] * len(s))
        tm.assert_series_equal(expected, result)

    def test_replace_with_dict_with_bool_keys(self):
        s = pd.Series([True, False, True])
        result = s.replace({"asdf": "asdb", True: "yes"})
        expected = pd.Series(["yes", False, "yes"])
        tm.assert_series_equal(result, expected)

    def test_replace_Int_with_na(self, any_int_ea_dtype):
        # GH 38267
        result = pd.Series([0, None], dtype=any_int_ea_dtype).replace(0, pd.NA)
        expected = pd.Series([pd.NA, pd.NA], dtype=any_int_ea_dtype)
        tm.assert_series_equal(result, expected)
        result = pd.Series([0, 1], dtype=any_int_ea_dtype).replace(0, pd.NA)
        result.replace(1, pd.NA, inplace=True)
        tm.assert_series_equal(result, expected)

    def test_replace2(self):
        N = 50
        ser = pd.Series(
            np.fabs(np.random.default_rng(2).standard_normal(N)),
            pd.date_range("2020-01-01", periods=N),
            dtype=object,
        )
        ser[:5] = np.nan
        ser[6:10] = "foo"
        ser[20:30] = "bar"

        # replace list with a single value
        msg = "Downcasting behavior in `replace`"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = ser.replace([np.nan, "foo", "bar"], -1)

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values with 2 lists
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)

        # replace inplace
        with tm.assert_produces_warning(FutureWarning, match=msg):
            return_value = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        assert return_value is None
        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

    @pytest.mark.parametrize("inplace", [True, False])
    def test_replace_cascade(self, inplace):
        # Test that replaced values are not replaced again
        # GH #50778
        ser = pd.Series([1, 2, 3])
        expected = pd.Series([2, 3, 4])

        res = ser.replace([1, 2, 3], [2, 3, 4], inplace=inplace)
        if inplace:
            tm.assert_series_equal(ser, expected)
        else:
            tm.assert_series_equal(res, expected)

    def test_replace_with_dictlike_and_string_dtype(self, nullable_string_dtype):
        # GH 32621, GH#44940
        ser = pd.Series(["one", "two", np.nan], dtype=nullable_string_dtype)
        expected = pd.Series(["1", "2", np.nan], dtype=nullable_string_dtype)
        result = ser.replace({"one": "1", "two": "2"})
        tm.assert_series_equal(expected, result)

    def test_replace_with_empty_dictlike(self):
        # GH 15289
        s = pd.Series(list("abcd"))
        tm.assert_series_equal(s, s.replace({}))

        empty_series = pd.Series([])
        tm.assert_series_equal(s, s.replace(empty_series))

    def test_replace_string_with_number(self):
        # GH 15743
        s = pd.Series([1, 2, 3])
        result = s.replace("2", np.nan)
        expected = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_replacer_equals_replacement(self):
        # GH 20656
        # make sure all replacers are matching against original values
        s = pd.Series(["a", "b"])
        expected = pd.Series(["b", "a"])
        result = s.replace({"a": "b", "b": "a"})
        tm.assert_series_equal(expected, result)

    def test_replace_unicode_with_number(self):
        # GH 15743
        s = pd.Series([1, 2, 3])
        result = s.replace("2", np.nan)
        expected = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_mixed_types_with_string(self):
        # Testing mixed
        s = pd.Series([1, 2, 3, "4", 4, 5])
        msg = "Downcasting behavior in `replace`"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.replace([2, "4"], np.nan)
        expected = pd.Series([1, np.nan, 3, np.nan, 4, 5])
        tm.assert_series_equal(expected, result)

    @pytest.mark.parametrize(
        "categorical, numeric",
        [
            (pd.Categorical(["A"], categories=["A", "B"]), [1]),
            (pd.Categorical(["A", "B"], categories=["A", "B"]), [1, 2]),
        ],
    )
    def test_replace_categorical(self, categorical, numeric, using_infer_string):
        # GH 24971, GH#23305
        ser = pd.Series(categorical)
        msg = "Downcasting behavior in `replace`"
        msg = "with CategoricalDtype is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.replace({"A": 1, "B": 2})
        expected = pd.Series(numeric).astype("category")
        if 2 not in expected.cat.categories:
            # i.e. categories should be [1, 2] even if there are no "B"s present
            # GH#44940
            expected = expected.cat.add_categories(2)
        tm.assert_series_equal(expected, result)

    @pytest.mark.parametrize(
        "data, data_exp", [(["a", "b", "c"], ["b", "b", "c"]), (["a"], ["b"])]
    )
    def test_replace_categorical_inplace(self, data, data_exp):
        # GH 53358
        result = pd.Series(data, dtype="category")
        msg = "with CategoricalDtype is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result.replace(to_replace="a", value="b", inplace=True)
        expected = pd.Series(data_exp, dtype="category")
        tm.assert_series_equal(result, expected)

    def test_replace_categorical_single(self):
        # GH 26988
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        s = pd.Series(dti)
        c = s.astype("category")

        expected = c.copy()
        expected = expected.cat.add_categories("foo")
        expected[2] = "foo"
        expected = expected.cat.remove_unused_categories()
        assert c[2] != "foo"

        msg = "with CategoricalDtype is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = c.replace(c[2], "foo")
        tm.assert_series_equal(expected, result)
        assert c[2] != "foo"  # ensure non-inplace call does not alter original

        msg = "with CategoricalDtype is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            return_value = c.replace(c[2], "foo", inplace=True)
        assert return_value is None
        tm.assert_series_equal(expected, c)

        first_value = c[0]
        msg = "with CategoricalDtype is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            return_value = c.replace(c[1], c[0], inplace=True)
        assert return_value is None
        assert c[0] == c[1] == first_value  # test replacing with existing value

    def test_replace_with_no_overflowerror(self):
        # GH 25616
        # casts to object without Exception from OverflowError
        s = pd.Series([0, 1, 2, 3, 4])
        result = s.replace([3], ["100000000000000000000"])
        expected = pd.Series([0, 1, 2, "100000000000000000000", 4])
        tm.assert_series_equal(result, expected)

        s = pd.Series([0, "100000000000000000000", "100000000000000000001"])
        result = s.replace(["100000000000000000000"], [1])
        expected = pd.Series([0, 1, "100000000000000000001"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, to_replace, exp",
        [
            ([1, 2, 3], {1: 2, 2: 3, 3: 4}, [2, 3, 4]),
            (["1", "2", "3"], {"1": "2", "2": "3", "3": "4"}, ["2", "3", "4"]),
        ],
    )
    def test_replace_commutative(self, ser, to_replace, exp):
        # GH 16051
        # DataFrame.replace() overwrites when values are non-numeric

        series = pd.Series(ser)

        expected = pd.Series(exp)
        result = series.replace(to_replace)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, exp", [([1, 2, 3], [1, True, 3]), (["x", 2, 3], ["x", True, 3])]
    )
    def test_replace_no_cast(self, ser, exp):
        # GH 9113
        # BUG: replace int64 dtype with bool coerces to int64

        series = pd.Series(ser)
        result = series.replace(2, True)
        expected = pd.Series(exp)

        tm.assert_series_equal(result, expected)

    def test_replace_invalid_to_replace(self):
        # GH 18634
        # API: replace() should raise an exception if invalid argument is given
        series = pd.Series(["a", "b", "c "])
        msg = (
            r"Expecting 'to_replace' to be either a scalar, array-like, "
            r"dict or None, got invalid type.*"
        )
        msg2 = (
            "Series.replace without 'value' and with non-dict-like "
            "'to_replace' is deprecated"
        )
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                series.replace(lambda x: x.strip())

    @pytest.mark.parametrize("frame", [False, True])
    def test_replace_nonbool_regex(self, frame):
        obj = pd.Series(["a", "b", "c "])
        if frame:
            obj = obj.to_frame()

        msg = "'to_replace' must be 'None' if 'regex' is not a bool"
        with pytest.raises(ValueError, match=msg):
            obj.replace(to_replace=["a"], regex="foo")

    @pytest.mark.parametrize("frame", [False, True])
    def test_replace_empty_copy(self, frame):
        obj = pd.Series([], dtype=np.float64)
        if frame:
            obj = obj.to_frame()

        res = obj.replace(4, 5, inplace=True)
        assert res is None

        res = obj.replace(4, 5, inplace=False)
        tm.assert_equal(res, obj)
        assert res is not obj

    def test_replace_only_one_dictlike_arg(self, fixed_now_ts):
        # GH#33340

        ser = pd.Series([1, 2, "A", fixed_now_ts, True])
        to_replace = {0: 1, 2: "A"}
        value = "foo"
        msg = "Series.replace cannot use dict-like to_replace and non-None value"
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)

        to_replace = 1
        value = {0: "foo", 2: "bar"}
        msg = "Series.replace cannot use dict-value and non-None to_replace"
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)

    def test_replace_extension_other(self, frame_or_series):
        # https://github.com/pandas-dev/pandas/issues/34530
        obj = frame_or_series(pd.array([1, 2, 3], dtype="Int64"))
        result = obj.replace("", "")  # no exception
        # should not have changed dtype
        tm.assert_equal(obj, result)

    def _check_replace_with_method(self, ser: pd.Series):
        df = ser.to_frame()

        msg1 = "The 'method' keyword in Series.replace is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg1):
            res = ser.replace(ser[1], method="pad")
        expected = pd.Series([ser[0], ser[0]] + list(ser[2:]), dtype=ser.dtype)
        tm.assert_series_equal(res, expected)

        msg2 = "The 'method' keyword in DataFrame.replace is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            res_df = df.replace(ser[1], method="pad")
        tm.assert_frame_equal(res_df, expected.to_frame())

        ser2 = ser.copy()
        with tm.assert_produces_warning(FutureWarning, match=msg1):
            res2 = ser2.replace(ser[1], method="pad", inplace=True)
        assert res2 is None
        tm.assert_series_equal(ser2, expected)

        with tm.assert_produces_warning(FutureWarning, match=msg2):
            res_df2 = df.replace(ser[1], method="pad", inplace=True)
        assert res_df2 is None
        tm.assert_frame_equal(df, expected.to_frame())

    def test_replace_ea_dtype_with_method(self, any_numeric_ea_dtype):
        arr = pd.array([1, 2, pd.NA, 4], dtype=any_numeric_ea_dtype)
        ser = pd.Series(arr)

        self._check_replace_with_method(ser)

    @pytest.mark.parametrize("as_categorical", [True, False])
    def test_replace_interval_with_method(self, as_categorical):
        # in particular interval that can't hold NA

        idx = pd.IntervalIndex.from_breaks(range(4))
        ser = pd.Series(idx)
        if as_categorical:
            ser = ser.astype("category")

        self._check_replace_with_method(ser)

    @pytest.mark.parametrize("as_period", [True, False])
    @pytest.mark.parametrize("as_categorical", [True, False])
    def test_replace_datetimelike_with_method(self, as_period, as_categorical):
        idx = pd.date_range("2016-01-01", periods=5, tz="US/Pacific")
        if as_period:
            idx = idx.tz_localize(None).to_period("D")

        ser = pd.Series(idx)
        ser.iloc[-2] = pd.NaT
        if as_categorical:
            ser = ser.astype("category")

        self._check_replace_with_method(ser)

    def test_replace_with_compiled_regex(self):
        # https://github.com/pandas-dev/pandas/issues/35680
        s = pd.Series(["a", "b", "c"])
        regex = re.compile("^a$")
        result = s.replace({regex: "z"}, regex=True)
        expected = pd.Series(["z", "b", "c"])
        tm.assert_series_equal(result, expected)

    def test_pandas_replace_na(self):
        # GH#43344
        ser = pd.Series(["AA", "BB", "CC", "DD", "EE", "", pd.NA], dtype="string")
        regex_mapping = {
            "AA": "CC",
            "BB": "CC",
            "EE": "CC",
            "CC": "CC-REPL",
        }
        result = ser.replace(regex_mapping, regex=True)
        exp = pd.Series(["CC", "CC", "CC-REPL", "DD", "CC", "", pd.NA], dtype="string")
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "dtype, input_data, to_replace, expected_data",
        [
            ("bool", [True, False], {True: False}, [False, False]),
            ("int64", [1, 2], {1: 10, 2: 20}, [10, 20]),
            ("Int64", [1, 2], {1: 10, 2: 20}, [10, 20]),
            ("float64", [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]),
            ("Float64", [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]),
            ("string", ["one", "two"], {"one": "1", "two": "2"}, ["1", "2"]),
            (
                pd.IntervalDtype("int64"),
                IntervalArray([pd.Interval(1, 2), pd.Interval(2, 3)]),
                {pd.Interval(1, 2): pd.Interval(10, 20)},
                IntervalArray([pd.Interval(10, 20), pd.Interval(2, 3)]),
            ),
            (
                pd.IntervalDtype("float64"),
                IntervalArray([pd.Interval(1.0, 2.7), pd.Interval(2.8, 3.1)]),
                {pd.Interval(1.0, 2.7): pd.Interval(10.6, 20.8)},
                IntervalArray([pd.Interval(10.6, 20.8), pd.Interval(2.8, 3.1)]),
            ),
            (
                pd.PeriodDtype("M"),
                [pd.Period("2020-05", freq="M")],
                {pd.Period("2020-05", freq="M"): pd.Period("2020-06", freq="M")},
                [pd.Period("2020-06", freq="M")],
            ),
        ],
    )
    def test_replace_dtype(self, dtype, input_data, to_replace, expected_data):
        # GH#33484
        ser = pd.Series(input_data, dtype=dtype)
        result = ser.replace(to_replace)
        expected = pd.Series(expected_data, dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_replace_string_dtype(self):
        # GH#40732, GH#44940
        ser = pd.Series(["one", "two", np.nan], dtype="string")
        res = ser.replace({"one": "1", "two": "2"})
        expected = pd.Series(["1", "2", np.nan], dtype="string")
        tm.assert_series_equal(res, expected)

        # GH#31644
        ser2 = pd.Series(["A", np.nan], dtype="string")
        res2 = ser2.replace("A", "B")
        expected2 = pd.Series(["B", np.nan], dtype="string")
        tm.assert_series_equal(res2, expected2)

        ser3 = pd.Series(["A", "B"], dtype="string")
        res3 = ser3.replace("A", pd.NA)
        expected3 = pd.Series([pd.NA, "B"], dtype="string")
        tm.assert_series_equal(res3, expected3)

    def test_replace_string_dtype_list_to_replace(self):
        # GH#41215, GH#44940
        ser = pd.Series(["abc", "def"], dtype="string")
        res = ser.replace(["abc", "any other string"], "xyz")
        expected = pd.Series(["xyz", "def"], dtype="string")
        tm.assert_series_equal(res, expected)

    def test_replace_string_dtype_regex(self):
        # GH#31644
        ser = pd.Series(["A", "B"], dtype="string")
        res = ser.replace(r".", "C", regex=True)
        expected = pd.Series(["C", "C"], dtype="string")
        tm.assert_series_equal(res, expected)

    def test_replace_nullable_numeric(self):
        # GH#40732, GH#44940

        floats = pd.Series([1.0, 2.0, 3.999, 4.4], dtype=pd.Float64Dtype())
        assert floats.replace({1.0: 9}).dtype == floats.dtype
        assert floats.replace(1.0, 9).dtype == floats.dtype
        assert floats.replace({1.0: 9.0}).dtype == floats.dtype
        assert floats.replace(1.0, 9.0).dtype == floats.dtype

        res = floats.replace(to_replace=[1.0, 2.0], value=[9.0, 10.0])
        assert res.dtype == floats.dtype

        ints = pd.Series([1, 2, 3, 4], dtype=pd.Int64Dtype())
        assert ints.replace({1: 9}).dtype == ints.dtype
        assert ints.replace(1, 9).dtype == ints.dtype
        assert ints.replace({1: 9.0}).dtype == ints.dtype
        assert ints.replace(1, 9.0).dtype == ints.dtype

        # nullable (for now) raises instead of casting
        with pytest.raises(TypeError, match="Invalid value"):
            ints.replace({1: 9.5})
        with pytest.raises(TypeError, match="Invalid value"):
            ints.replace(1, 9.5)

    @pytest.mark.parametrize("regex", [False, True])
    def test_replace_regex_dtype_series(self, regex):
        # GH-48644
        series = pd.Series(["0"], dtype=object)
        expected = pd.Series([1])
        msg = "Downcasting behavior in `replace`"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = series.replace(to_replace="0", value=1, regex=regex)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("regex", [False, True])
    def test_replace_regex_dtype_series_string(self, regex):
        series = pd.Series(["0"], dtype="str")
        expected = pd.Series([1], dtype="int64")
        msg = "Downcasting behavior in `replace`"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = series.replace(to_replace="0", value=1, regex=regex)
        tm.assert_series_equal(result, expected)

    def test_replace_different_int_types(self, any_int_numpy_dtype):
        # GH#45311
        labs = pd.Series([1, 1, 1, 0, 0, 2, 2, 2], dtype=any_int_numpy_dtype)

        maps = pd.Series([0, 2, 1], dtype=any_int_numpy_dtype)
        map_dict = dict(zip(maps.values, maps.index))

        result = labs.replace(map_dict)
        expected = labs.replace({0: 0, 2: 1, 1: 2})
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [2, np.nan, 2.0])
    def test_replace_value_none_dtype_numeric(self, val):
        # GH#48231
        ser = pd.Series([1, val])
        result = ser.replace(val, None)
        expected = pd.Series([1, None], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_replace_change_dtype_series(self):
        # GH#25797
        df = pd.DataFrame({"Test": ["0.5", True, "0.6"]}, dtype=object)
        df["Test"] = df["Test"].replace([True], [np.nan])
        expected = pd.DataFrame({"Test": ["0.5", np.nan, "0.6"]}, dtype=object)
        tm.assert_frame_equal(df, expected)

        df = pd.DataFrame({"Test": ["0.5", None, "0.6"]}, dtype=object)
        df["Test"] = df["Test"].replace([None], [np.nan])
        tm.assert_frame_equal(df, expected)

        df = pd.DataFrame({"Test": ["0.5", None, "0.6"]}, dtype=object)
        df["Test"] = df["Test"].fillna(np.nan)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("dtype", ["object", "Int64"])
    def test_replace_na_in_obj_column(self, dtype):
        # GH#47480
        ser = pd.Series([0, 1, pd.NA], dtype=dtype)
        expected = pd.Series([0, 2, pd.NA], dtype=dtype)
        result = ser.replace(to_replace=1, value=2)
        tm.assert_series_equal(result, expected)

        ser.replace(to_replace=1, value=2, inplace=True)
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("val", [0, 0.5])
    def test_replace_numeric_column_with_na(self, val):
        # GH#50758
        ser = pd.Series([val, 1])
        expected = pd.Series([val, pd.NA])
        result = ser.replace(to_replace=1, value=pd.NA)
        tm.assert_series_equal(result, expected)

        ser.replace(to_replace=1, value=pd.NA, inplace=True)
        tm.assert_series_equal(ser, expected)

    def test_replace_ea_float_with_bool(self):
        # GH#55398
        ser = pd.Series([0.0], dtype="Float64")
        expected = ser.copy()
        result = ser.replace(False, 1.0)
        tm.assert_series_equal(result, expected)

        ser = pd.Series([False], dtype="boolean")
        expected = ser.copy()
        result = ser.replace(0.0, True)
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_indexing.py ===
from datetime import datetime
import re

import numpy as np
import pytest

from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError

import pandas as pd
from pandas import (
    DatetimeIndex,
    NaT,
    Period,
    PeriodIndex,
    Series,
    Timedelta,
    date_range,
    notna,
    period_range,
)
import pandas._testing as tm

dti4 = date_range("2016-01-01", periods=4)
dti = dti4[:-1]
rng = pd.Index(range(3))


@pytest.fixture(
    params=[
        dti,
        dti.tz_localize("UTC"),
        dti.to_period("W"),
        dti - dti[0],
        rng,
        pd.Index([1, 2, 3]),
        pd.Index([2.0, 3.0, 4.0]),
        pd.Index([4, 5, 6], dtype="u8"),
        pd.IntervalIndex.from_breaks(dti4),
    ]
)
def non_comparable_idx(request):
    # All have length 3
    return request.param


class TestGetItem:
    def test_getitem_slice_keeps_name(self):
        idx = period_range("20010101", periods=10, freq="D", name="bob")
        assert idx.name == idx[1:].name

    def test_getitem(self):
        idx1 = period_range("2011-01-01", "2011-01-31", freq="D", name="idx")

        for idx in [idx1]:
            result = idx[0]
            assert result == Period("2011-01-01", freq="D")

            result = idx[-1]
            assert result == Period("2011-01-31", freq="D")

            result = idx[0:5]
            expected = period_range("2011-01-01", "2011-01-05", freq="D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            result = idx[0:10:2]
            expected = PeriodIndex(
                ["2011-01-01", "2011-01-03", "2011-01-05", "2011-01-07", "2011-01-09"],
                freq="D",
                name="idx",
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            result = idx[-20:-5:3]
            expected = PeriodIndex(
                ["2011-01-12", "2011-01-15", "2011-01-18", "2011-01-21", "2011-01-24"],
                freq="D",
                name="idx",
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            result = idx[4::-1]
            expected = PeriodIndex(
                ["2011-01-05", "2011-01-04", "2011-01-03", "2011-01-02", "2011-01-01"],
                freq="D",
                name="idx",
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

    def test_getitem_index(self):
        idx = period_range("2007-01", periods=10, freq="M", name="x")

        result = idx[[1, 3, 5]]
        exp = PeriodIndex(["2007-02", "2007-04", "2007-06"], freq="M", name="x")
        tm.assert_index_equal(result, exp)

        result = idx[[True, True, False, False, False, True, True, False, False, False]]
        exp = PeriodIndex(
            ["2007-01", "2007-02", "2007-06", "2007-07"], freq="M", name="x"
        )
        tm.assert_index_equal(result, exp)

    def test_getitem_partial(self):
        rng = period_range("2007-01", periods=50, freq="M")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)

        with pytest.raises(KeyError, match=r"^'2006'$"):
            ts["2006"]

        result = ts["2008"]
        assert (result.index.year == 2008).all()

        result = ts["2008":"2009"]
        assert len(result) == 24

        result = ts["2008-1":"2009-12"]
        assert len(result) == 24

        result = ts["2008Q1":"2009Q4"]
        assert len(result) == 24

        result = ts[:"2009"]
        assert len(result) == 36

        result = ts["2009":]
        assert len(result) == 50 - 24

        exp = result
        result = ts[24:]
        tm.assert_series_equal(exp, result)

        ts = pd.concat([ts[10:], ts[10:]])
        msg = "left slice bound for non-unique label: '2008'"
        with pytest.raises(KeyError, match=msg):
            ts[slice("2008", "2009")]

    def test_getitem_datetime(self):
        rng = period_range(start="2012-01-01", periods=10, freq="W-MON")
        ts = Series(range(len(rng)), index=rng)

        dt1 = datetime(2011, 10, 2)
        dt4 = datetime(2012, 4, 20)

        rs = ts[dt1:dt4]
        tm.assert_series_equal(rs, ts)

    def test_getitem_nat(self):
        idx = PeriodIndex(["2011-01", "NaT", "2011-02"], freq="M")
        assert idx[0] == Period("2011-01", freq="M")
        assert idx[1] is NaT

        s = Series([0, 1, 2], index=idx)
        assert s[NaT] == 1

        s = Series(idx, index=idx)
        assert s[Period("2011-01", freq="M")] == Period("2011-01", freq="M")
        assert s[NaT] is NaT

    def test_getitem_list_periods(self):
        # GH 7710
        rng = period_range(start="2012-01-01", periods=10, freq="D")
        ts = Series(range(len(rng)), index=rng)
        exp = ts.iloc[[1]]
        tm.assert_series_equal(ts[[Period("2012-01-02", freq="D")]], exp)

    @pytest.mark.arm_slow
    def test_getitem_seconds(self):
        # GH#6716
        didx = date_range(start="2013/01/01 09:00:00", freq="s", periods=4000)
        pidx = period_range(start="2013/01/01 09:00:00", freq="s", periods=4000)

        for idx in [didx, pidx]:
            # getitem against index should raise ValueError
            values = [
                "2014",
                "2013/02",
                "2013/01/02",
                "2013/02/01 9h",
                "2013/02/01 09:00",
            ]
            for val in values:
                # GH7116
                # these show deprecations as we are trying
                # to slice with non-integer indexers
                with pytest.raises(IndexError, match="only integers, slices"):
                    idx[val]

            ser = Series(np.random.default_rng(2).random(len(idx)), index=idx)
            tm.assert_series_equal(ser["2013/01/01 10:00"], ser[3600:3660])
            tm.assert_series_equal(ser["2013/01/01 9h"], ser[:3600])
            for d in ["2013/01/01", "2013/01", "2013"]:
                tm.assert_series_equal(ser[d], ser)

    @pytest.mark.parametrize(
        "idx_range",
        [
            date_range,
            period_range,
        ],
    )
    def test_getitem_day(self, idx_range):
        # GH#6716
        # Confirm DatetimeIndex and PeriodIndex works identically
        # getitem against index should raise ValueError
        idx = idx_range(start="2013/01/01", freq="D", periods=400)
        values = [
            "2014",
            "2013/02",
            "2013/01/02",
            "2013/02/01 9h",
            "2013/02/01 09:00",
        ]
        for val in values:
            # GH7116
            # these show deprecations as we are trying
            # to slice with non-integer indexers
            with pytest.raises(IndexError, match="only integers, slices"):
                idx[val]

        ser = Series(np.random.default_rng(2).random(len(idx)), index=idx)
        tm.assert_series_equal(ser["2013/01"], ser[0:31])
        tm.assert_series_equal(ser["2013/02"], ser[31:59])
        tm.assert_series_equal(ser["2014"], ser[365:])

        invalid = ["2013/02/01 9h", "2013/02/01 09:00"]
        for val in invalid:
            with pytest.raises(KeyError, match=val):
                ser[val]


class TestGetLoc:
    def test_get_loc_msg(self):
        idx = period_range("2000-1-1", freq="Y", periods=10)
        bad_period = Period("2012", "Y")
        with pytest.raises(KeyError, match=r"^Period\('2012', 'Y-DEC'\)$"):
            idx.get_loc(bad_period)

        try:
            idx.get_loc(bad_period)
        except KeyError as inst:
            assert inst.args[0] == bad_period

    def test_get_loc_nat(self):
        didx = DatetimeIndex(["2011-01-01", "NaT", "2011-01-03"])
        pidx = PeriodIndex(["2011-01-01", "NaT", "2011-01-03"], freq="M")

        # check DatetimeIndex compat
        for idx in [didx, pidx]:
            assert idx.get_loc(NaT) == 1
            assert idx.get_loc(None) == 1
            assert idx.get_loc(float("nan")) == 1
            assert idx.get_loc(np.nan) == 1

    def test_get_loc(self):
        # GH 17717
        p0 = Period("2017-09-01")
        p1 = Period("2017-09-02")
        p2 = Period("2017-09-03")

        # get the location of p1/p2 from
        # monotonic increasing PeriodIndex with non-duplicate
        idx0 = PeriodIndex([p0, p1, p2])
        expected_idx1_p1 = 1
        expected_idx1_p2 = 2

        assert idx0.get_loc(p1) == expected_idx1_p1
        assert idx0.get_loc(str(p1)) == expected_idx1_p1
        assert idx0.get_loc(p2) == expected_idx1_p2
        assert idx0.get_loc(str(p2)) == expected_idx1_p2

        msg = "Cannot interpret 'foo' as period"
        with pytest.raises(KeyError, match=msg):
            idx0.get_loc("foo")
        with pytest.raises(KeyError, match=r"^1\.1$"):
            idx0.get_loc(1.1)

        with pytest.raises(InvalidIndexError, match=re.escape(str(idx0))):
            idx0.get_loc(idx0)

        # get the location of p1/p2 from
        # monotonic increasing PeriodIndex with duplicate
        idx1 = PeriodIndex([p1, p1, p2])
        expected_idx1_p1 = slice(0, 2)
        expected_idx1_p2 = 2

        assert idx1.get_loc(p1) == expected_idx1_p1
        assert idx1.get_loc(str(p1)) == expected_idx1_p1
        assert idx1.get_loc(p2) == expected_idx1_p2
        assert idx1.get_loc(str(p2)) == expected_idx1_p2

        msg = "Cannot interpret 'foo' as period"
        with pytest.raises(KeyError, match=msg):
            idx1.get_loc("foo")

        with pytest.raises(KeyError, match=r"^1\.1$"):
            idx1.get_loc(1.1)

        with pytest.raises(InvalidIndexError, match=re.escape(str(idx1))):
            idx1.get_loc(idx1)

        # get the location of p1/p2 from
        # non-monotonic increasing/decreasing PeriodIndex with duplicate
        idx2 = PeriodIndex([p2, p1, p2])
        expected_idx2_p1 = 1
        expected_idx2_p2 = np.array([True, False, True])

        assert idx2.get_loc(p1) == expected_idx2_p1
        assert idx2.get_loc(str(p1)) == expected_idx2_p1
        tm.assert_numpy_array_equal(idx2.get_loc(p2), expected_idx2_p2)
        tm.assert_numpy_array_equal(idx2.get_loc(str(p2)), expected_idx2_p2)

    def test_get_loc_integer(self):
        dti = date_range("2016-01-01", periods=3)
        pi = dti.to_period("D")
        with pytest.raises(KeyError, match="16801"):
            pi.get_loc(16801)

        pi2 = dti.to_period("Y")  # duplicates, ordinals are all 46
        with pytest.raises(KeyError, match="46"):
            pi2.get_loc(46)

    def test_get_loc_invalid_string_raises_keyerror(self):
        # GH#34240
        pi = period_range("2000", periods=3, name="A")
        with pytest.raises(KeyError, match="A"):
            pi.get_loc("A")

        ser = Series([1, 2, 3], index=pi)
        with pytest.raises(KeyError, match="A"):
            ser.loc["A"]

        with pytest.raises(KeyError, match="A"):
            ser["A"]

        assert "A" not in ser
        assert "A" not in pi

    def test_get_loc_mismatched_freq(self):
        # see also test_get_indexer_mismatched_dtype testing we get analogous
        # behavior for get_loc
        dti = date_range("2016-01-01", periods=3)
        pi = dti.to_period("D")
        pi2 = dti.to_period("W")
        pi3 = pi.view(pi2.dtype)  # i.e. matching i8 representations

        with pytest.raises(KeyError, match="W-SUN"):
            pi.get_loc(pi2[0])

        with pytest.raises(KeyError, match="W-SUN"):
            # even though we have matching i8 values
            pi.get_loc(pi3[0])


class TestGetIndexer:
    def test_get_indexer(self):
        # GH 17717
        p1 = Period("2017-09-01")
        p2 = Period("2017-09-04")
        p3 = Period("2017-09-07")

        tp0 = Period("2017-08-31")
        tp1 = Period("2017-09-02")
        tp2 = Period("2017-09-05")
        tp3 = Period("2017-09-09")

        idx = PeriodIndex([p1, p2, p3])

        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        target = PeriodIndex([tp0, tp1, tp2, tp3])
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1, 2], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2, -1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 0, 1, 2], dtype=np.intp)
        )

        res = idx.get_indexer(target, "nearest", tolerance=Timedelta("1 day"))
        tm.assert_numpy_array_equal(res, np.array([0, 0, 1, -1], dtype=np.intp))

    def test_get_indexer_mismatched_dtype(self):
        # Check that we return all -1s and do not raise or cast incorrectly

        dti = date_range("2016-01-01", periods=3)
        pi = dti.to_period("D")
        pi2 = dti.to_period("W")

        expected = np.array([-1, -1, -1], dtype=np.intp)

        result = pi.get_indexer(dti)
        tm.assert_numpy_array_equal(result, expected)

        # This should work in both directions
        result = dti.get_indexer(pi)
        tm.assert_numpy_array_equal(result, expected)

        result = pi.get_indexer(pi2)
        tm.assert_numpy_array_equal(result, expected)

        # We expect the same from get_indexer_non_unique
        result = pi.get_indexer_non_unique(dti)[0]
        tm.assert_numpy_array_equal(result, expected)

        result = dti.get_indexer_non_unique(pi)[0]
        tm.assert_numpy_array_equal(result, expected)

        result = pi.get_indexer_non_unique(pi2)[0]
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_mismatched_dtype_different_length(self, non_comparable_idx):
        # without method we aren't checking inequalities, so get all-missing
        #  but do not raise
        dti = date_range("2016-01-01", periods=3)
        pi = dti.to_period("D")

        other = non_comparable_idx

        res = pi[:-1].get_indexer(other)
        expected = -np.ones(other.shape, dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize("method", ["pad", "backfill", "nearest"])
    def test_get_indexer_mismatched_dtype_with_method(self, non_comparable_idx, method):
        dti = date_range("2016-01-01", periods=3)
        pi = dti.to_period("D")

        other = non_comparable_idx

        msg = re.escape(f"Cannot compare dtypes {pi.dtype} and {other.dtype}")
        with pytest.raises(TypeError, match=msg):
            pi.get_indexer(other, method=method)

        for dtype in ["object", "category"]:
            other2 = other.astype(dtype)
            if dtype == "object" and isinstance(other, PeriodIndex):
                continue
            # Two different error message patterns depending on dtypes
            msg = "|".join(
                [
                    re.escape(msg)
                    for msg in (
                        f"Cannot compare dtypes {pi.dtype} and {other.dtype}",
                        " not supported between instances of ",
                    )
                ]
            )
            with pytest.raises(TypeError, match=msg):
                pi.get_indexer(other2, method=method)

    def test_get_indexer_non_unique(self):
        # GH 17717
        p1 = Period("2017-09-02")
        p2 = Period("2017-09-03")
        p3 = Period("2017-09-04")
        p4 = Period("2017-09-05")

        idx1 = PeriodIndex([p1, p2, p1])
        idx2 = PeriodIndex([p2, p1, p3, p4])

        result = idx1.get_indexer_non_unique(idx2)
        expected_indexer = np.array([1, 0, 2, -1, -1], dtype=np.intp)
        expected_missing = np.array([2, 3], dtype=np.intp)

        tm.assert_numpy_array_equal(result[0], expected_indexer)
        tm.assert_numpy_array_equal(result[1], expected_missing)

    # TODO: This method came from test_period; de-dup with version above
    def test_get_indexer2(self):
        idx = period_range("2000-01-01", periods=3).asfreq("h", how="start")
        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        target = PeriodIndex(
            ["1999-12-31T23", "2000-01-01T12", "2000-01-02T01"], freq="h"
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest", tolerance="1 hour"),
            np.array([0, -1, 1], dtype=np.intp),
        )

        msg = "Input has different freq=None from PeriodArray\\(freq=h\\)"
        with pytest.raises(ValueError, match=msg):
            idx.get_indexer(target, "nearest", tolerance="1 minute")

        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest", tolerance="1 day"),
            np.array([0, 1, 1], dtype=np.intp),
        )
        tol_raw = [
            Timedelta("1 hour"),
            Timedelta("1 hour"),
            np.timedelta64(1, "D"),
        ]
        tm.assert_numpy_array_equal(
            idx.get_indexer(
                target, "nearest", tolerance=[np.timedelta64(x) for x in tol_raw]
            ),
            np.array([0, -1, 1], dtype=np.intp),
        )
        tol_bad = [
            Timedelta("2 hour").to_timedelta64(),
            Timedelta("1 hour").to_timedelta64(),
            np.timedelta64(1, "M"),
        ]
        with pytest.raises(
            libperiod.IncompatibleFrequency, match="Input has different freq=None from"
        ):
            idx.get_indexer(target, "nearest", tolerance=tol_bad)


class TestWhere:
    def test_where(self, listlike_box):
        i = period_range("20130101", periods=5, freq="D")
        cond = [True] * len(i)
        expected = i
        result = i.where(listlike_box(cond))
        tm.assert_index_equal(result, expected)

        cond = [False] + [True] * (len(i) - 1)
        expected = PeriodIndex([NaT] + i[1:].tolist(), freq="D")
        result = i.where(listlike_box(cond))
        tm.assert_index_equal(result, expected)

    def test_where_other(self):
        i = period_range("20130101", periods=5, freq="D")
        for arr in [np.nan, NaT]:
            result = i.where(notna(i), other=arr)
            expected = i
            tm.assert_index_equal(result, expected)

        i2 = i.copy()
        i2 = PeriodIndex([NaT, NaT] + i[2:].tolist(), freq="D")
        result = i.where(notna(i2), i2)
        tm.assert_index_equal(result, i2)

        i2 = i.copy()
        i2 = PeriodIndex([NaT, NaT] + i[2:].tolist(), freq="D")
        result = i.where(notna(i2), i2.values)
        tm.assert_index_equal(result, i2)

    def test_where_invalid_dtypes(self):
        pi = period_range("20130101", periods=5, freq="D")

        tail = pi[2:].tolist()
        i2 = PeriodIndex([NaT, NaT] + tail, freq="D")
        mask = notna(i2)

        result = pi.where(mask, i2.asi8)
        expected = pd.Index([NaT._value, NaT._value] + tail, dtype=object)
        assert isinstance(expected[0], int)
        tm.assert_index_equal(result, expected)

        tdi = i2.asi8.view("timedelta64[ns]")
        expected = pd.Index([tdi[0], tdi[1]] + tail, dtype=object)
        assert isinstance(expected[0], np.timedelta64)
        result = pi.where(mask, tdi)
        tm.assert_index_equal(result, expected)

        dti = i2.to_timestamp("s")
        expected = pd.Index([dti[0], dti[1]] + tail, dtype=object)
        assert expected[0] is NaT
        result = pi.where(mask, dti)
        tm.assert_index_equal(result, expected)

        td = Timedelta(days=4)
        expected = pd.Index([td, td] + tail, dtype=object)
        assert expected[0] == td
        result = pi.where(mask, td)
        tm.assert_index_equal(result, expected)

    def test_where_mismatched_nat(self):
        pi = period_range("20130101", periods=5, freq="D")
        cond = np.array([True, False, True, True, False])

        tdnat = np.timedelta64("NaT", "ns")
        expected = pd.Index([pi[0], tdnat, pi[2], pi[3], tdnat], dtype=object)
        assert expected[1] is tdnat
        result = pi.where(cond, tdnat)
        tm.assert_index_equal(result, expected)


class TestTake:
    def test_take(self):
        # GH#10295
        idx1 = period_range("2011-01-01", "2011-01-31", freq="D", name="idx")

        for idx in [idx1]:
            result = idx.take([0])
            assert result == Period("2011-01-01", freq="D")

            result = idx.take([5])
            assert result == Period("2011-01-06", freq="D")

            result = idx.take([0, 1, 2])
            expected = period_range("2011-01-01", "2011-01-03", freq="D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == "D"
            assert result.freq == expected.freq

            result = idx.take([0, 2, 4])
            expected = PeriodIndex(
                ["2011-01-01", "2011-01-03", "2011-01-05"], freq="D", name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            result = idx.take([7, 4, 1])
            expected = PeriodIndex(
                ["2011-01-08", "2011-01-05", "2011-01-02"], freq="D", name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            result = idx.take([3, 2, 5])
            expected = PeriodIndex(
                ["2011-01-04", "2011-01-03", "2011-01-06"], freq="D", name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

            result = idx.take([-3, 2, 5])
            expected = PeriodIndex(
                ["2011-01-29", "2011-01-03", "2011-01-06"], freq="D", name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq
            assert result.freq == "D"

    def test_take_misc(self):
        index = period_range(start="1/1/10", end="12/31/12", freq="D", name="idx")
        expected = PeriodIndex(
            [
                datetime(2010, 1, 6),
                datetime(2010, 1, 7),
                datetime(2010, 1, 9),
                datetime(2010, 1, 13),
            ],
            freq="D",
            name="idx",
        )

        taken1 = index.take([5, 6, 8, 12])
        taken2 = index[[5, 6, 8, 12]]

        for taken in [taken1, taken2]:
            tm.assert_index_equal(taken, expected)
            assert isinstance(taken, PeriodIndex)
            assert taken.freq == index.freq
            assert taken.name == expected.name

    def test_take_fill_value(self):
        # GH#12631
        idx = PeriodIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01"], name="xxx", freq="D"
        )
        result = idx.take(np.array([1, 0, -1]))
        expected = PeriodIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx", freq="D"
        )
        tm.assert_index_equal(result, expected)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = PeriodIndex(
            ["2011-02-01", "2011-01-01", "NaT"], name="xxx", freq="D"
        )
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = PeriodIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx", freq="D"
        )
        tm.assert_index_equal(result, expected)

        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for( axis 0 with)? size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))


class TestGetValue:
    @pytest.mark.parametrize("freq", ["h", "D"])
    def test_get_value_datetime_hourly(self, freq):
        # get_loc and get_value should treat datetime objects symmetrically
        # TODO: this test used to test get_value, which is removed in 2.0.
        #  should this test be moved somewhere, or is what's left redundant?
        dti = date_range("2016-01-01", periods=3, freq="MS")
        pi = dti.to_period(freq)
        ser = Series(range(7, 10), index=pi)

        ts = dti[0]

        assert pi.get_loc(ts) == 0
        assert ser[ts] == 7
        assert ser.loc[ts] == 7

        ts2 = ts + Timedelta(hours=3)
        if freq == "h":
            with pytest.raises(KeyError, match="2016-01-01 03:00"):
                pi.get_loc(ts2)
            with pytest.raises(KeyError, match="2016-01-01 03:00"):
                ser[ts2]
            with pytest.raises(KeyError, match="2016-01-01 03:00"):
                ser.loc[ts2]
        else:
            assert pi.get_loc(ts2) == 0
            assert ser[ts2] == 7
            assert ser.loc[ts2] == 7


class TestContains:
    def test_contains(self):
        # GH 17717
        p0 = Period("2017-09-01")
        p1 = Period("2017-09-02")
        p2 = Period("2017-09-03")
        p3 = Period("2017-09-04")

        ps0 = [p0, p1, p2]
        idx0 = PeriodIndex(ps0)

        for p in ps0:
            assert p in idx0
            assert str(p) in idx0

        # GH#31172
        # Higher-resolution period-like are _not_ considered as contained
        key = "2017-09-01 00:00:01"
        assert key not in idx0
        with pytest.raises(KeyError, match=key):
            idx0.get_loc(key)

        assert "2017-09" in idx0

        assert p3 not in idx0

    def test_contains_freq_mismatch(self):
        rng = period_range("2007-01", freq="M", periods=10)

        assert Period("2007-01", freq="M") in rng
        assert Period("2007-01", freq="D") not in rng
        assert Period("2007-01", freq="2M") not in rng

    def test_contains_nat(self):
        # see gh-13582
        idx = period_range("2007-01", freq="M", periods=10)
        assert NaT not in idx
        assert None not in idx
        assert float("nan") not in idx
        assert np.nan not in idx

        idx = PeriodIndex(["2011-01", "NaT", "2011-02"], freq="M")
        assert NaT in idx
        assert None in idx
        assert float("nan") in idx
        assert np.nan in idx


class TestAsOfLocs:
    def test_asof_locs_mismatched_type(self):
        dti = date_range("2016-01-01", periods=3)
        pi = dti.to_period("D")
        pi2 = dti.to_period("h")

        mask = np.array([0, 1, 0], dtype=bool)

        msg = "must be DatetimeIndex or PeriodIndex"
        with pytest.raises(TypeError, match=msg):
            pi.asof_locs(pd.Index(pi.asi8, dtype=np.int64), mask)

        with pytest.raises(TypeError, match=msg):
            pi.asof_locs(pd.Index(pi.asi8, dtype=np.float64), mask)

        with pytest.raises(TypeError, match=msg):
            # TimedeltaIndex
            pi.asof_locs(dti - dti, mask)

        msg = "Input has different freq=h"
        with pytest.raises(libperiod.IncompatibleFrequency, match=msg):
            pi.asof_locs(pi2, mask)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\tests\test_exponential.py ===
from sympy.assumptions.refine import refine
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import expand_log
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, re, sign, transpose)
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.polytools import gcd
from sympy.series.order import O
from sympy.simplify.simplify import simplify
from sympy.core.parameters import global_parameters
from sympy.functions.elementary.exponential import match_real_imag
from sympy.abc import x, y, z
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises, XFAIL, _both_exp_pow


@_both_exp_pow
def test_exp_values():
    if global_parameters.exp_is_pow:
        assert type(exp(x)) is Pow
    else:
        assert type(exp(x)) is exp

    k = Symbol('k', integer=True)

    assert exp(nan) is nan

    assert exp(oo) is oo
    assert exp(-oo) == 0

    assert exp(0) == 1
    assert exp(1) == E
    assert exp(-1 + x).as_base_exp() == (S.Exp1, x - 1)
    assert exp(1 + x).as_base_exp() == (S.Exp1, x + 1)

    assert exp(pi*I/2) == I
    assert exp(pi*I) == -1
    assert exp(pi*I*Rational(3, 2)) == -I
    assert exp(2*pi*I) == 1

    assert refine(exp(pi*I*2*k)) == 1
    assert refine(exp(pi*I*2*(k + S.Half))) == -1
    assert refine(exp(pi*I*2*(k + Rational(1, 4)))) == I
    assert refine(exp(pi*I*2*(k + Rational(3, 4)))) == -I

    assert exp(log(x)) == x
    assert exp(2*log(x)) == x**2
    assert exp(pi*log(x)) == x**pi

    assert exp(17*log(x) + E*log(y)) == x**17 * y**E

    assert exp(x*log(x)) != x**x
    assert exp(sin(x)*log(x)) != x

    assert exp(3*log(x) + oo*x) == exp(oo*x) * x**3
    assert exp(4*log(x)*log(y) + 3*log(x)) == x**3 * exp(4*log(x)*log(y))

    assert exp(-oo, evaluate=False).is_finite is True
    assert exp(oo, evaluate=False).is_finite is False


@_both_exp_pow
def test_exp_period():
    assert exp(I*pi*Rational(9, 4)) == exp(I*pi/4)
    assert exp(I*pi*Rational(46, 18)) == exp(I*pi*Rational(5, 9))
    assert exp(I*pi*Rational(25, 7)) == exp(I*pi*Rational(-3, 7))
    assert exp(I*pi*Rational(-19, 3)) == exp(-I*pi/3)
    assert exp(I*pi*Rational(37, 8)) - exp(I*pi*Rational(-11, 8)) == 0
    assert exp(I*pi*Rational(-5, 3)) / exp(I*pi*Rational(11, 5)) * exp(I*pi*Rational(148, 15)) == 1

    assert exp(2 - I*pi*Rational(17, 5)) == exp(2 + I*pi*Rational(3, 5))
    assert exp(log(3) + I*pi*Rational(29, 9)) == 3 * exp(I*pi*Rational(-7, 9))

    n = Symbol('n', integer=True)
    e = Symbol('e', even=True)
    assert exp(e*I*pi) == 1
    assert exp((e + 1)*I*pi) == -1
    assert exp((1 + 4*n)*I*pi/2) == I
    assert exp((-1 + 4*n)*I*pi/2) == -I


@_both_exp_pow
def test_exp_log():
    x = Symbol("x", real=True)
    assert log(exp(x)) == x
    assert exp(log(x)) == x

    if not global_parameters.exp_is_pow:
        assert log(x).inverse() == exp
        assert exp(x).inverse() == log

    y = Symbol("y", polar=True)
    assert log(exp_polar(z)) == z
    assert exp(log(y)) == y


@_both_exp_pow
def test_exp_expand():
    e = exp(log(Rational(2))*(1 + x) - log(Rational(2))*x)
    assert e.expand() == 2
    assert exp(x + y) != exp(x)*exp(y)
    assert exp(x + y).expand() == exp(x)*exp(y)


@_both_exp_pow
def test_exp__as_base_exp():
    assert exp(x).as_base_exp() == (E, x)
    assert exp(2*x).as_base_exp() == (E, 2*x)
    assert exp(x*y).as_base_exp() == (E, x*y)
    assert exp(-x).as_base_exp() == (E, -x)

    # Pow( *expr.as_base_exp() ) == expr    invariant should hold
    assert E**x == exp(x)
    assert E**(2*x) == exp(2*x)
    assert E**(x*y) == exp(x*y)

    assert exp(x).base is S.Exp1
    assert exp(x).exp == x


@_both_exp_pow
def test_exp_infinity():
    assert exp(I*y) != nan
    assert refine(exp(I*oo)) is nan
    assert refine(exp(-I*oo)) is nan
    assert exp(y*I*oo) != nan
    assert exp(zoo) is nan
    x = Symbol('x', extended_real=True, finite=False)
    assert exp(x).is_complex is None


@_both_exp_pow
def test_exp_subs():
    x = Symbol('x')
    e = (exp(3*log(x), evaluate=False))  # evaluates to x**3
    assert e.subs(x**3, y**3) == e
    assert e.subs(x**2, 5) == e
    assert (x**3).subs(x**2, y) != y**Rational(3, 2)
    assert exp(exp(x) + exp(x**2)).subs(exp(exp(x)), y) == y * exp(exp(x**2))
    assert exp(x).subs(E, y) == y**x
    x = symbols('x', real=True)
    assert exp(5*x).subs(exp(7*x), y) == y**Rational(5, 7)
    assert exp(2*x + 7).subs(exp(3*x), y) == y**Rational(2, 3) * exp(7)
    x = symbols('x', positive=True)
    assert exp(3*log(x)).subs(x**2, y) == y**Rational(3, 2)
    # differentiate between E and exp
    assert exp(exp(x + E)).subs(exp, 3) == 3**(3**(x + E))
    assert exp(exp(x + E)).subs(exp, sin) == sin(sin(x + E))
    assert exp(exp(x + E)).subs(E, 3) == 3**(3**(x + 3))
    assert exp(3).subs(E, sin) == sin(3)


def test_exp_adjoint():
    x = Symbol('x', commutative=False)
    assert adjoint(exp(x)) == exp(adjoint(x))


def test_exp_conjugate():
    assert conjugate(exp(x)) == exp(conjugate(x))


@_both_exp_pow
def test_exp_transpose():
    assert transpose(exp(x)) == exp(transpose(x))


@_both_exp_pow
def test_exp_rewrite():
    assert exp(x).rewrite(sin) == sinh(x) + cosh(x)
    assert exp(x*I).rewrite(cos) == cos(x) + I*sin(x)
    assert exp(1).rewrite(cos) == sinh(1) + cosh(1)
    assert exp(1).rewrite(sin) == sinh(1) + cosh(1)
    assert exp(1).rewrite(sin) == sinh(1) + cosh(1)
    assert exp(x).rewrite(tanh) == (1 + tanh(x/2))/(1 - tanh(x/2))
    assert exp(pi*I/4).rewrite(sqrt) == sqrt(2)/2 + sqrt(2)*I/2
    assert exp(pi*I/3).rewrite(sqrt) == S.Half + sqrt(3)*I/2
    if not global_parameters.exp_is_pow:
        assert exp(x*log(y)).rewrite(Pow) == y**x
        assert exp(log(x)*log(y)).rewrite(Pow) in [x**log(y), y**log(x)]
        assert exp(log(log(x))*y).rewrite(Pow) == log(x)**y

    n = Symbol('n', integer=True)

    assert Sum((exp(pi*I/2)/2)**n, (n, 0, oo)).rewrite(sqrt).doit() == Rational(4, 5) + I*2/5
    assert Sum((exp(pi*I/4)/2)**n, (n, 0, oo)).rewrite(sqrt).doit() == 1/(1 - sqrt(2)*(1 + I)/4)
    assert (Sum((exp(pi*I/3)/2)**n, (n, 0, oo)).rewrite(sqrt).doit().cancel()
            == 4*I/(sqrt(3) + 3*I))


@_both_exp_pow
def test_exp_leading_term():
    assert exp(x).as_leading_term(x) == 1
    assert exp(2 + x).as_leading_term(x) == exp(2)
    assert exp((2*x + 3) / (x+1)).as_leading_term(x) == exp(3)

    # The following tests are commented, since now SymPy returns the
    # original function when the leading term in the series expansion does
    # not exist.
    # raises(NotImplementedError, lambda: exp(1/x).as_leading_term(x))
    # raises(NotImplementedError, lambda: exp((x + 1) / x**2).as_leading_term(x))
    # raises(NotImplementedError, lambda: exp(x + 1/x).as_leading_term(x))


@_both_exp_pow
def test_exp_taylor_term():
    x = symbols('x')
    assert exp(x).taylor_term(1, x) == x
    assert exp(x).taylor_term(3, x) == x**3/6
    assert exp(x).taylor_term(4, x) == x**4/24
    assert exp(x).taylor_term(-1, x) is S.Zero


def test_exp_MatrixSymbol():
    A = MatrixSymbol("A", 2, 2)
    assert exp(A).has(exp)


def test_exp_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: exp(x).fdiff(2))


def test_log_values():
    assert log(nan) is nan

    assert log(oo) is oo
    assert log(-oo) is oo

    assert log(zoo) is zoo
    assert log(-zoo) is zoo

    assert log(0) is zoo

    assert log(1) == 0
    assert log(-1) == I*pi

    assert log(E) == 1
    assert log(-E).expand() == 1 + I*pi

    assert unchanged(log, pi)
    assert log(-pi).expand() == log(pi) + I*pi

    assert unchanged(log, 17)
    assert log(-17) == log(17) + I*pi

    assert log(I) == I*pi/2
    assert log(-I) == -I*pi/2

    assert log(17*I) == I*pi/2 + log(17)
    assert log(-17*I).expand() == -I*pi/2 + log(17)

    assert log(oo*I) is oo
    assert log(-oo*I) is oo
    assert log(0, 2) is zoo
    assert log(0, 5) is zoo

    assert exp(-log(3))**(-1) == 3

    assert log(S.Half) == -log(2)
    assert log(2*3).func is log
    assert log(2*3**2).func is log


def test_match_real_imag():
    x, y = symbols('x,y', real=True)
    i = Symbol('i', imaginary=True)
    assert match_real_imag(S.One) == (1, 0)
    assert match_real_imag(I) == (0, 1)
    assert match_real_imag(3 - 5*I) == (3, -5)
    assert match_real_imag(-sqrt(3) + S.Half*I) == (-sqrt(3), S.Half)
    assert match_real_imag(x + y*I) == (x, y)
    assert match_real_imag(x*I + y*I) == (0, x + y)
    assert match_real_imag((x + y)*I) == (0, x + y)
    assert match_real_imag(Rational(-2, 3)*i*I) == (None, None)
    assert match_real_imag(1 - 2*i) == (None, None)
    assert match_real_imag(sqrt(2)*(3 - 5*I)) == (None, None)


def test_log_exact():
    # check for pi/2, pi/3, pi/4, pi/6, pi/8, pi/12; pi/5, pi/10:
    for n in range(-23, 24):
        if gcd(n, 24) != 1:
            assert log(exp(n*I*pi/24).rewrite(sqrt)) == n*I*pi/24
        for n in range(-9, 10):
            assert log(exp(n*I*pi/10).rewrite(sqrt)) == n*I*pi/10

    assert log(S.Half - I*sqrt(3)/2) == -I*pi/3
    assert log(Rational(-1, 2) + I*sqrt(3)/2) == I*pi*Rational(2, 3)
    assert log(-sqrt(2)/2 - I*sqrt(2)/2) == -I*pi*Rational(3, 4)
    assert log(-sqrt(3)/2 - I*S.Half) == -I*pi*Rational(5, 6)

    assert log(Rational(-1, 4) + sqrt(5)/4 - I*sqrt(sqrt(5)/8 + Rational(5, 8))) == -I*pi*Rational(2, 5)
    assert log(sqrt(Rational(5, 8) - sqrt(5)/8) + I*(Rational(1, 4) + sqrt(5)/4)) == I*pi*Rational(3, 10)
    assert log(-sqrt(sqrt(2)/4 + S.Half) + I*sqrt(S.Half - sqrt(2)/4)) == I*pi*Rational(7, 8)
    assert log(-sqrt(6)/4 - sqrt(2)/4 + I*(-sqrt(6)/4 + sqrt(2)/4)) == -I*pi*Rational(11, 12)

    assert log(-1 + I*sqrt(3)) == log(2) + I*pi*Rational(2, 3)
    assert log(5 + 5*I) == log(5*sqrt(2)) + I*pi/4
    assert log(sqrt(-12)) == log(2*sqrt(3)) + I*pi/2
    assert log(-sqrt(6) + sqrt(2) - I*sqrt(6) - I*sqrt(2)) == log(4) - I*pi*Rational(7, 12)
    assert log(-sqrt(6-3*sqrt(2)) - I*sqrt(6+3*sqrt(2))) == log(2*sqrt(3)) - I*pi*Rational(5, 8)
    assert log(1 + I*sqrt(2-sqrt(2))/sqrt(2+sqrt(2))) == log(2/sqrt(sqrt(2) + 2)) + I*pi/8
    assert log(cos(pi*Rational(7, 12)) + I*sin(pi*Rational(7, 12))) == I*pi*Rational(7, 12)
    assert log(cos(pi*Rational(6, 5)) + I*sin(pi*Rational(6, 5))) == I*pi*Rational(-4, 5)

    assert log(5*(1 + I)/sqrt(2)) == log(5) + I*pi/4
    assert log(sqrt(2)*(-sqrt(3) + 1 - sqrt(3)*I - I)) == log(4) - I*pi*Rational(7, 12)
    assert log(-sqrt(2)*(1 - I*sqrt(3))) == log(2*sqrt(2)) + I*pi*Rational(2, 3)
    assert log(sqrt(3)*I*(-sqrt(6 - 3*sqrt(2)) - I*sqrt(3*sqrt(2) + 6))) == log(6) - I*pi/8

    zero = (1 + sqrt(2))**2 - 3 - 2*sqrt(2)
    assert log(zero - I*sqrt(3)) == log(sqrt(3)) - I*pi/2
    assert unchanged(log, zero + I*zero) or log(zero + zero*I) is zoo

    # bail quickly if no obvious simplification is possible:
    assert unchanged(log, (sqrt(2)-1/sqrt(sqrt(3)+I))**1000)
    # beware of non-real coefficients
    assert unchanged(log, sqrt(2-sqrt(5))*(1 + I))


def test_log_base():
    assert log(1, 2) == 0
    assert log(2, 2) == 1
    assert log(3, 2) == log(3)/log(2)
    assert log(6, 2) == 1 + log(3)/log(2)
    assert log(6, 3) == 1 + log(2)/log(3)
    assert log(2**3, 2) == 3
    assert log(3**3, 3) == 3
    assert log(5, 1) is zoo
    assert log(1, 1) is nan
    assert log(Rational(2, 3), 10) == log(Rational(2, 3))/log(10)
    assert log(Rational(2, 3), Rational(1, 3)) == -log(2)/log(3) + 1
    assert log(Rational(2, 3), Rational(2, 5)) == \
        log(Rational(2, 3))/log(Rational(2, 5))
    # issue 17148
    assert log(Rational(8, 3), 2) == -log(3)/log(2) + 3


def test_log_symbolic():
    assert log(x, exp(1)) == log(x)
    assert log(exp(x)) != x

    assert log(x, exp(1)) == log(x)
    assert log(x*y) != log(x) + log(y)
    assert log(x/y).expand() != log(x) - log(y)
    assert log(x/y).expand(force=True) == log(x) - log(y)
    assert log(x**y).expand() != y*log(x)
    assert log(x**y).expand(force=True) == y*log(x)

    assert log(x, 2) == log(x)/log(2)
    assert log(E, 2) == 1/log(2)

    p, q = symbols('p,q', positive=True)
    r = Symbol('r', real=True)

    assert log(p**2) != 2*log(p)
    assert log(p**2).expand() == 2*log(p)
    assert log(x**2).expand() != 2*log(x)
    assert log(p**q) != q*log(p)
    assert log(exp(p)) == p
    assert log(p*q) != log(p) + log(q)
    assert log(p*q).expand() == log(p) + log(q)

    assert log(-sqrt(3)) == log(sqrt(3)) + I*pi
    assert log(-exp(p)) != p + I*pi
    assert log(-exp(x)).expand() != x + I*pi
    assert log(-exp(r)).expand() == r + I*pi

    assert log(x**y) != y*log(x)

    assert (log(x**-5)**-1).expand() != -1/log(x)/5
    assert (log(p**-5)**-1).expand() == -1/log(p)/5
    assert log(-x).func is log and log(-x).args[0] == -x
    assert log(-p).func is log and log(-p).args[0] == -p


def test_log_exp():
    assert log(exp(4*I*pi)) == 0     # exp evaluates
    assert log(exp(-5*I*pi)) == I*pi # exp evaluates
    assert log(exp(I*pi*Rational(19, 4))) == I*pi*Rational(3, 4)
    assert log(exp(I*pi*Rational(25, 7))) == I*pi*Rational(-3, 7)
    assert log(exp(-5*I)) == -5*I + 2*I*pi


@_both_exp_pow
def test_exp_assumptions():
    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)
    for e in exp, exp_polar:
        assert e(x).is_real is None
        assert e(x).is_imaginary is None
        assert e(i).is_real is None
        assert e(i).is_imaginary is None
        assert e(r).is_real is True
        assert e(r).is_imaginary is False
        assert e(re(x)).is_extended_real is True
        assert e(re(x)).is_imaginary is False

    assert Pow(E, I*pi, evaluate=False).is_imaginary == False
    assert Pow(E, 2*I*pi, evaluate=False).is_imaginary == False
    assert Pow(E, I*pi/2, evaluate=False).is_imaginary == True
    assert Pow(E, I*pi/3, evaluate=False).is_imaginary is None

    assert exp(0, evaluate=False).is_algebraic

    a = Symbol('a', algebraic=True)
    an = Symbol('an', algebraic=True, nonzero=True)
    r = Symbol('r', rational=True)
    rn = Symbol('rn', rational=True, nonzero=True)
    assert exp(a).is_algebraic is None
    assert exp(an).is_algebraic is False
    assert exp(pi*r).is_algebraic is None
    assert exp(pi*rn).is_algebraic is False

    assert exp(0, evaluate=False).is_algebraic is True
    assert exp(I*pi/3, evaluate=False).is_algebraic is True
    assert exp(I*pi*r, evaluate=False).is_algebraic is True


@_both_exp_pow
def test_exp_AccumBounds():
    assert exp(AccumBounds(1, 2)) == AccumBounds(E, E**2)


def test_log_assumptions():
    p = symbols('p', positive=True)
    n = symbols('n', negative=True)
    z = symbols('z', zero=True)
    x = symbols('x', infinite=True, extended_positive=True)

    assert log(z).is_positive is False
    assert log(x).is_extended_positive is True
    assert log(2) > 0
    assert log(1, evaluate=False).is_zero
    assert log(1 + z).is_zero
    assert log(p).is_zero is None
    assert log(n).is_zero is False
    assert log(0.5).is_negative is True
    assert log(exp(p) + 1).is_positive

    assert log(1, evaluate=False).is_algebraic
    assert log(42, evaluate=False).is_algebraic is False

    assert log(1 + z).is_rational


def test_log_hashing():
    assert x != log(log(x))
    assert hash(x) != hash(log(log(x)))
    assert log(x) != log(log(log(x)))

    e = 1/log(log(x) + log(log(x)))
    assert e.base.func is log
    e = 1/log(log(x) + log(log(log(x))))
    assert e.base.func is log

    e = log(log(x))
    assert e.func is log
    assert x.func is not log
    assert hash(log(log(x))) != hash(x)
    assert e != x


def test_log_sign():
    assert sign(log(2)) == 1


def test_log_expand_complex():
    assert log(1 + I).expand(complex=True) == log(2)/2 + I*pi/4
    assert log(1 - sqrt(2)).expand(complex=True) == log(sqrt(2) - 1) + I*pi


def test_log_apply_evalf():
    value = (log(3)/log(2) - 1).evalf()
    assert value.epsilon_eq(Float("0.58496250072115618145373"))


def test_log_leading_term():
    p = Symbol('p')

    # Test for STEP 3
    assert log(1 + x + x**2).as_leading_term(x, cdir=1) == x
    # Test for STEP 4
    assert log(2*x).as_leading_term(x, cdir=1) == log(x) + log(2)
    assert log(2*x).as_leading_term(x, cdir=-1) == log(x) + log(2)
    assert log(-2*x).as_leading_term(x, cdir=1, logx=p) == p + log(2) + I*pi
    assert log(-2*x).as_leading_term(x, cdir=-1, logx=p) == p + log(2) - I*pi
    # Test for STEP 5
    assert log(-2*x + (3 - I)*x**2).as_leading_term(x, cdir=1) == log(x) + log(2) - I*pi
    assert log(-2*x + (3 - I)*x**2).as_leading_term(x, cdir=-1) == log(x) + log(2) - I*pi
    assert log(2*x + (3 - I)*x**2).as_leading_term(x, cdir=1) == log(x) + log(2)
    assert log(2*x + (3 - I)*x**2).as_leading_term(x, cdir=-1) == log(x) + log(2) - 2*I*pi
    assert log(-1 + x - I*x**2 + I*x**3).as_leading_term(x, cdir=1) == -I*pi
    assert log(-1 + x - I*x**2 + I*x**3).as_leading_term(x, cdir=-1) == -I*pi
    assert log(-1/(1 - x)).as_leading_term(x, cdir=1) == I*pi
    assert log(-1/(1 - x)).as_leading_term(x, cdir=-1) == I*pi


def test_log_nseries():
    p = Symbol('p')
    assert log(1/x)._eval_nseries(x, 4, logx=-p, cdir=1) == p
    assert log(1/x)._eval_nseries(x, 4, logx=-p, cdir=-1) == p + 2*I*pi
    assert log(x - 1)._eval_nseries(x, 4, None, I) == I*pi - x - x**2/2 - x**3/3 + O(x**4)
    assert log(x - 1)._eval_nseries(x, 4, None, -I) == -I*pi - x - x**2/2 - x**3/3 + O(x**4)
    assert log(I*x + I*x**3 - 1)._eval_nseries(x, 3, None, 1) == I*pi - I*x + x**2/2 + O(x**3)
    assert log(I*x + I*x**3 - 1)._eval_nseries(x, 3, None, -1) == -I*pi - I*x + x**2/2 + O(x**3)
    assert log(I*x**2 + I*x**3 - 1)._eval_nseries(x, 3, None, 1) == I*pi - I*x**2 + O(x**3)
    assert log(I*x**2 + I*x**3 - 1)._eval_nseries(x, 3, None, -1) == I*pi - I*x**2 + O(x**3)
    assert log(2*x + (3 - I)*x**2)._eval_nseries(x, 3, None, 1) == log(2) + log(x) + \
    x*(S(3)/2 - I/2) + x**2*(-1 + 3*I/4) + O(x**3)
    assert log(2*x + (3 - I)*x**2)._eval_nseries(x, 3, None, -1) == -2*I*pi + log(2) + \
    log(x) - x*(-S(3)/2 + I/2) + x**2*(-1 + 3*I/4) + O(x**3)
    assert log(-2*x + (3 - I)*x**2)._eval_nseries(x, 3, None, 1) == -I*pi + log(2) + log(x) + \
    x*(-S(3)/2 + I/2) + x**2*(-1 + 3*I/4) + O(x**3)
    assert log(-2*x + (3 - I)*x**2)._eval_nseries(x, 3, None, -1) == -I*pi + log(2) + log(x) - \
    x*(S(3)/2 - I/2) + x**2*(-1 + 3*I/4) + O(x**3)
    assert log(sqrt(-I*x**2 - 3)*sqrt(-I*x**2 - 1) - 2)._eval_nseries(x, 3, None, 1) == -I*pi + \
    log(sqrt(3) + 2) + 2*sqrt(3)*I*x**2/(3*sqrt(3) + 6) + O(x**3)
    assert log(-1/(1 - x))._eval_nseries(x, 3, None, 1) == I*pi + x + x**2/2 + O(x**3)
    assert log(-1/(1 - x))._eval_nseries(x, 3, None, -1) == I*pi + x + x**2/2 + O(x**3)


def test_log_series():
    # Note Series at infinities other than oo/-oo were introduced as a part of
    # pull request 23798. Refer https://github.com/sympy/sympy/pull/23798 for
    # more information.
    expr1 = log(1 + x)
    expr2 = log(x + sqrt(x**2 + 1))

    assert expr1.series(x, x0=I*oo, n=4) == 1/(3*x**3) - 1/(2*x**2) + 1/x + \
    I*pi/2 - log(I/x) + O(x**(-4), (x, oo*I))
    assert expr1.series(x, x0=-I*oo, n=4) == 1/(3*x**3) - 1/(2*x**2) + 1/x - \
    I*pi/2 - log(-I/x) + O(x**(-4), (x, -oo*I))
    assert expr2.series(x, x0=I*oo, n=4) == 1/(4*x**2) + I*pi/2 + log(2) - \
    log(I/x) + O(x**(-4), (x, oo*I))
    assert expr2.series(x, x0=-I*oo, n=4) == -1/(4*x**2) - I*pi/2 - log(2) + \
    log(-I/x) + O(x**(-4), (x, -oo*I))


def test_log_expand():
    w = Symbol("w", positive=True)
    e = log(w**(log(5)/log(3)))
    assert e.expand() == log(5)/log(3) * log(w)
    x, y, z = symbols('x,y,z', positive=True)
    assert log(x*(y + z)).expand(mul=False) == log(x) + log(y + z)
    assert log(log(x**2)*log(y*z)).expand() in [log(2*log(x)*log(y) +
        2*log(x)*log(z)), log(log(x)*log(z) + log(y)*log(x)) + log(2),
        log((log(y) + log(z))*log(x)) + log(2)]
    assert log(x**log(x**2)).expand(deep=False) == log(x)*log(x**2)
    assert log(x**log(x**2)).expand() == 2*log(x)**2
    x, y = symbols('x,y')
    assert log(x*y).expand(force=True) == log(x) + log(y)
    assert log(x**y).expand(force=True) == y*log(x)
    assert log(exp(x)).expand(force=True) == x

    # there's generally no need to expand out logs since this requires
    # factoring and if simplification is sought, it's cheaper to put
    # logs together than it is to take them apart.
    assert log(2*3**2).expand() != 2*log(3) + log(2)


@XFAIL
def test_log_expand_fail():
    x, y, z = symbols('x,y,z', positive=True)
    assert (log(x*(y + z))*(x + y)).expand(mul=True, log=True) == y*log(
        x) + y*log(y + z) + z*log(x) + z*log(y + z)


def test_log_simplify():
    x = Symbol("x", positive=True)
    assert log(x**2).expand() == 2*log(x)
    assert expand_log(log(x**(2 + log(2)))) == (2 + log(2))*log(x)

    z = Symbol('z')
    assert log(sqrt(z)).expand() == log(z)/2
    assert expand_log(log(z**(log(2) - 1))) == (log(2) - 1)*log(z)
    assert log(z**(-1)).expand() != -log(z)
    assert log(z**(x/(x+1))).expand() == x*log(z)/(x + 1)


def test_log_AccumBounds():
    assert log(AccumBounds(1, E)) == AccumBounds(0, 1)
    assert log(AccumBounds(0, E)) == AccumBounds(-oo, 1)
    assert log(AccumBounds(-1, E)) == S.NaN
    assert log(AccumBounds(0, oo)) == AccumBounds(-oo, oo)
    assert log(AccumBounds(-oo, 0)) == S.NaN
    assert log(AccumBounds(-oo, oo)) == S.NaN


@_both_exp_pow
def test_lambertw():
    k = Symbol('k')

    assert LambertW(x, 0) == LambertW(x)
    assert LambertW(x, 0, evaluate=False) != LambertW(x)
    assert LambertW(0) == 0
    assert LambertW(E) == 1
    assert LambertW(-1/E) == -1
    assert LambertW(-log(2)/2) == -log(2)
    assert LambertW(oo) is oo
    assert LambertW(0, 1) is -oo
    assert LambertW(0, 42) is -oo
    assert LambertW(-pi/2, -1) == -I*pi/2
    assert LambertW(-1/E, -1) == -1
    assert LambertW(-2*exp(-2), -1) == -2
    assert LambertW(2*log(2)) == log(2)
    assert LambertW(-pi/2) == I*pi/2
    assert LambertW(exp(1 + E)) == E

    assert LambertW(x**2).diff(x) == 2*LambertW(x**2)/x/(1 + LambertW(x**2))
    assert LambertW(x, k).diff(x) == LambertW(x, k)/x/(1 + LambertW(x, k))

    assert LambertW(sqrt(2)).evalf(30).epsilon_eq(
        Float("0.701338383413663009202120278965", 30), 1e-29)
    assert re(LambertW(2, -1)).evalf().epsilon_eq(Float("-0.834310366631110"))

    assert LambertW(-1).is_real is False  # issue 5215
    assert LambertW(2, evaluate=False).is_real
    p = Symbol('p', positive=True)
    assert LambertW(p, evaluate=False).is_real
    assert LambertW(p - 1, evaluate=False).is_real is None
    assert LambertW(-p - 2/S.Exp1, evaluate=False).is_real is False
    assert LambertW(S.Half, -1, evaluate=False).is_real is False
    assert LambertW(Rational(-1, 10), -1, evaluate=False).is_real
    assert LambertW(-10, -1, evaluate=False).is_real is False
    assert LambertW(-2, 2, evaluate=False).is_real is False

    assert LambertW(0, evaluate=False).is_algebraic
    na = Symbol('na', nonzero=True, algebraic=True)
    assert LambertW(na).is_algebraic is False
    assert LambertW(p).is_zero is False
    n = Symbol('n', negative=True)
    assert LambertW(n).is_zero is False


def test_issue_5673():
    e = LambertW(-1)
    assert e.is_comparable is False
    assert e.is_positive is not True
    e2 = 1 - 1/(1 - exp(-1000))
    assert e2.is_positive is not True
    e3 = -2 + exp(exp(LambertW(log(2)))*LambertW(log(2)))
    assert e3.is_nonzero is not True


def test_log_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: log(x).fdiff(2))


def test_log_taylor_term():
    x = symbols('x')
    assert log(x).taylor_term(0, x) == x
    assert log(x).taylor_term(1, x) == -x**2/2
    assert log(x).taylor_term(4, x) == x**5/5
    assert log(x).taylor_term(-1, x) is S.Zero


def test_exp_expand_NC():
    A, B, C = symbols('A,B,C', commutative=False)

    assert exp(A + B).expand() == exp(A + B)
    assert exp(A + B + C).expand() == exp(A + B + C)
    assert exp(x + y).expand() == exp(x)*exp(y)
    assert exp(x + y + z).expand() == exp(x)*exp(y)*exp(z)


@_both_exp_pow
def test_as_numer_denom():
    n = symbols('n', negative=True)
    assert exp(x).as_numer_denom() == (exp(x), 1)
    assert exp(-x).as_numer_denom() == (1, exp(x))
    assert exp(-2*x).as_numer_denom() == (1, exp(2*x))
    assert exp(-2).as_numer_denom() == (1, exp(2))
    assert exp(n).as_numer_denom() == (1, exp(-n))
    assert exp(-n).as_numer_denom() == (exp(-n), 1)
    assert exp(-I*x).as_numer_denom() == (1, exp(I*x))
    assert exp(-I*n).as_numer_denom() == (1, exp(I*n))
    assert exp(-n).as_numer_denom() == (exp(-n), 1)
    # Check noncommutativity
    a = symbols('a', commutative=False)
    assert exp(-a).as_numer_denom() == (exp(-a), 1)


@_both_exp_pow
def test_polar():
    x, y = symbols('x y', polar=True)

    assert abs(exp_polar(I*4)) == 1
    assert abs(exp_polar(0)) == 1
    assert abs(exp_polar(2 + 3*I)) == exp(2)
    assert exp_polar(I*10).n() == exp_polar(I*10)

    assert log(exp_polar(z)) == z
    assert log(x*y).expand() == log(x) + log(y)
    assert log(x**z).expand() == z*log(x)

    assert exp_polar(3).exp == 3

    # Compare exp(1.0*pi*I).
    assert (exp_polar(1.0*pi*I).n(n=5)).as_real_imag()[1] >= 0

    assert exp_polar(0).is_rational is True  # issue 8008


def test_exp_summation():
    w = symbols("w")
    m, n, i, j = symbols("m n i j")
    expr = exp(Sum(w*i, (i, 0, n), (j, 0, m)))
    assert expr.expand() == Product(exp(w*i), (i, 0, n), (j, 0, m))


def test_log_product():
    from sympy.abc import n, m

    i, j = symbols('i,j', positive=True, integer=True)
    x, y = symbols('x,y', positive=True)
    z = symbols('z', real=True)
    w = symbols('w')

    expr = log(Product(x**i, (i, 1, n)))
    assert simplify(expr) == expr
    assert expr.expand() == Sum(i*log(x), (i, 1, n))
    expr = log(Product(x**i*y**j, (i, 1, n), (j, 1, m)))
    assert simplify(expr) == expr
    assert expr.expand() == Sum(i*log(x) + j*log(y), (i, 1, n), (j, 1, m))

    expr = log(Product(-2, (n, 0, 4)))
    assert simplify(expr) == expr
    assert expr.expand() == expr
    assert expr.expand(force=True) == Sum(log(-2), (n, 0, 4))

    expr = log(Product(exp(z*i), (i, 0, n)))
    assert expr.expand() == Sum(z*i, (i, 0, n))

    expr = log(Product(exp(w*i), (i, 0, n)))
    assert expr.expand() == expr
    assert expr.expand(force=True) == Sum(w*i, (i, 0, n))

    expr = log(Product(i**2*abs(j), (i, 1, n), (j, 1, m)))
    assert expr.expand() == Sum(2*log(i) + log(j), (i, 1, n), (j, 1, m))


@XFAIL
def test_log_product_simplify_to_sum():
    from sympy.abc import n, m
    i, j = symbols('i,j', positive=True, integer=True)
    x, y = symbols('x,y', positive=True)
    assert simplify(log(Product(x**i, (i, 1, n)))) == Sum(i*log(x), (i, 1, n))
    assert simplify(log(Product(x**i*y**j, (i, 1, n), (j, 1, m)))) == \
            Sum(i*log(x) + j*log(y), (i, 1, n), (j, 1, m))


def test_issue_8866():
    assert simplify(log(x, 10, evaluate=False)) == simplify(log(x, 10))
    assert expand_log(log(x, 10, evaluate=False)) == expand_log(log(x, 10))

    y = Symbol('y', positive=True)
    l1 = log(exp(y), exp(10))
    b1 = log(exp(y), exp(5))
    l2 = log(exp(y), exp(10), evaluate=False)
    b2 = log(exp(y), exp(5), evaluate=False)
    assert simplify(log(l1, b1)) == simplify(log(l2, b2))
    assert expand_log(log(l1, b1)) == expand_log(log(l2, b2))


def test_log_expand_factor():
    assert (log(18)/log(3) - 2).expand(factor=True) == log(2)/log(3)
    assert (log(12)/log(2)).expand(factor=True) == log(3)/log(2) + 2
    assert (log(15)/log(3)).expand(factor=True) == 1 + log(5)/log(3)
    assert (log(2)/(-log(12) + log(24))).expand(factor=True) == 1

    assert expand_log(log(12), factor=True) == log(3) + 2*log(2)
    assert expand_log(log(21)/log(7), factor=False) == log(3)/log(7) + 1
    assert expand_log(log(45)/log(5) + log(20), factor=False) == \
        1 + 2*log(3)/log(5) + log(20)
    assert expand_log(log(45)/log(5) + log(26), factor=True) == \
        log(2) + log(13) + (log(5) + 2*log(3))/log(5)


def test_issue_9116():
    n = Symbol('n', positive=True, integer=True)
    assert log(n).is_nonnegative is True


def test_issue_18473():
    assert exp(x*log(cos(1/x))).as_leading_term(x) == S.NaN
    assert exp(x*log(tan(1/x))).as_leading_term(x) == S.NaN
    assert log(cos(1/x)).as_leading_term(x) == S.NaN
    assert log(tan(1/x)).as_leading_term(x) == S.NaN
    assert log(cos(1/x) + 2).as_leading_term(x) == AccumBounds(0, log(3))
    assert exp(x*log(cos(1/x) + 2)).as_leading_term(x) == 1
    assert log(cos(1/x) - 2).as_leading_term(x) == S.NaN
    assert exp(x*log(cos(1/x) - 2)).as_leading_term(x) == S.NaN
    assert log(cos(1/x) + 1).as_leading_term(x) == AccumBounds(-oo, log(2))
    assert exp(x*log(cos(1/x) + 1)).as_leading_term(x) == AccumBounds(0, 1)
    assert log(sin(1/x)**2).as_leading_term(x) == AccumBounds(-oo, 0)
    assert exp(x*log(sin(1/x)**2)).as_leading_term(x) == AccumBounds(0, 1)
    assert log(tan(1/x)**2).as_leading_term(x) == AccumBounds(-oo, oo)
    assert exp(2*x*(log(tan(1/x)**2))).as_leading_term(x) == AccumBounds(0, oo)