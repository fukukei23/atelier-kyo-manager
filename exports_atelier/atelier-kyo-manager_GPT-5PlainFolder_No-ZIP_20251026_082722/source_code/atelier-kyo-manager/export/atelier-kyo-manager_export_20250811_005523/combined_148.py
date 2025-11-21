
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_skip_group_norm.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionSkipGroupNorm(Fusion):
    """
    Fuse Add + GroupNorm into one node: SkipGroupNorm.
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipGroupNorm", "GroupNorm")
        # Update shape inference is needed since other fusions might add new edge which does not have shape info yet.
        self.shape_infer_helper = self.model.infer_runtime_shape(update=True)

        if self.shape_infer_helper is None:
            logger.warning("SkipGroupNorm fusion will be skipped since symbolic shape inference disabled or failed.")

    def create_transpose_node(self, input_name: str, perm: list[int], output_name=None):
        """Append a Transpose node after an input"""
        node_name = self.model.create_node_name("Transpose")
        if output_name is None:
            output_name = node_name + "_out" + "-" + input_name
        transpose_node = helper.make_node("Transpose", inputs=[input_name], outputs=[output_name], name=node_name)
        transpose_node.attribute.extend([helper.make_attribute("perm", perm)])
        return transpose_node

    def get_skip_index(self, add, is_channel_last: bool):
        """Add has two inputs. This classifies which input is skip based on shape info (skip allows broadcast)."""
        skip = -1
        broadcast = False

        assert self.shape_infer_helper is not None
        shape_a = self.shape_infer_helper.get_edge_shape(add.input[0])
        shape_b = self.shape_infer_helper.get_edge_shape(add.input[1])
        assert shape_a is not None and shape_b is not None

        if len(shape_a) == 4 and len(shape_b) == 4:
            if shape_a == shape_b:
                skip = 1
            else:
                c = 3 if is_channel_last else 1
                h = 1 if is_channel_last else 2
                w = 2 if is_channel_last else 3
                if shape_a[0] == shape_b[0] and shape_a[c] == shape_b[c]:
                    if shape_b[h] == 1 and shape_b[w] == 1:
                        skip = 1
                        broadcast = True
                    elif shape_a[h] == 1 and shape_a[w] == 1:
                        skip = 0
                        broadcast = True

        if skip < 0:
            logger.debug(
                "skip SkipGroupNorm fusion since shape of Add inputs (%s, %s) are not expected",
                add.input[0],
                add.input[1],
            )
        return skip, broadcast

    def has_multiple_consumers(self, output_name, input_name_to_nodes):
        """Whether an output has multiple consumers (like graph output or more than one children nodes)"""
        return self.model.find_graph_output(output_name) is not None or (
            output_name in input_name_to_nodes and len(input_name_to_nodes[output_name]) > 1
        )

    def remove_if_safe(self, node, input_name_to_nodes):
        """Remove a node if it is safe (only one children, and not graph output)"""
        if not self.has_multiple_consumers(node.output[0], input_name_to_nodes):
            self.nodes_to_remove.extend([node])

    def is_bias_1d(self, bias_name: str):
        """Whether bias is an initializer of one dimension"""
        initializer = self.model.get_initializer(bias_name)
        if initializer is None:
            return False

        bias_weight = NumpyHelper.to_array(initializer)
        if bias_weight is None:
            logger.debug("Bias weight not found")
            return False

        if len(bias_weight.shape) != 1:
            logger.debug("Bias weight is not 1D")
            return False
        return True

    def match_bias_path(self, node, input_name_to_nodes, output_name_to_node):
        """
        Match the bias graph pattern from an Transpose node after Reshape node like in below example.
        It checks whether the bias is 1D initializer. If so, remove Add and redirect MatMul output to Reshape.
        """
        # Before Fusion:
        #                        MatMul  (bias)
        #                            \  /     (shape)
        #                             Add    /
        #                               \   /
        #       (a)                   Reshape
        #        \                       |
        # Transpose([0, 3, 1, 2])   Transpose([0, 3, 1, 2])  --- the start node, this func only handles the above nodes.
        #                        \  /
        #                         Add
        #                         / \
        #                      (c)  Transpose([0,2,3,1])
        #                              |
        #                           GroupNorm
        #                              |
        #                             (d)
        #
        # After Fusion (the nodes below Reshape is handled in the fuse function):
        #                    MatMul (shape)
        #                       \   /
        #                (a)   Reshape
        #                  \    /
        #                SkipGroupNorm
        #                  /    \
        #                (d)   Transpose([0, 3, 1, 2])
        #                         \
        #                         (c)

        add_input_index = []
        bias_nodes = self.model.match_parent_path(
            node, ["Reshape", "Add", "MatMul"], [0, 0, None], output_name_to_node, add_input_index
        )
        if bias_nodes is None:
            return None

        (reshape, add_bias, matmul) = bias_nodes
        bias = bias_nodes[1].input[1 - add_input_index[0]]
        if not self.is_bias_1d(bias):
            return None

        reshape.input[0] = matmul.output[0]
        self.remove_if_safe(add_bias, input_name_to_nodes)

        return bias

    def match_transpose_from_nhwc(self, output_name, input_name_to_nodes, output_name_to_node):
        """Match whether an output is from a Transpose(perm=[0,3,1,2]) node."""
        parent = output_name_to_node.get(output_name, None)
        if parent is not None and parent.op_type == "Transpose":
            permutation = OnnxModel.get_node_attribute(parent, "perm")
            if permutation == [0, 3, 1, 2]:
                self.remove_if_safe(parent, input_name_to_nodes)
                return parent
        return None

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # This fusion requires shape information, so skip it if shape is not available.
        if self.shape_infer_helper is None:
            return

        # Before Fusion:
        #     (a)  (b)
        #       \  /
        #       Add
        #       /\
        #   (c)   Transpose([0,2,3,1])
        #            \
        #          GroupNorm
        #             |
        #            (d)
        #
        # After Fusion:
        #           (a)              (b)
        #             \              /
        #   Transpose([0,2,3,1])   Transpose([0,2,3,1])
        #                \        /
        #              SkipGroupNorm
        #                  /    \
        #                 /    Transpose([0, 3, 1, 2])
        #                /        \
        #               (d)       (c)
        nodes = self.model.match_parent_path(node, ["Transpose", "Add"], [0, 0], output_name_to_node)
        if nodes is None:
            return

        (transpose, add) = nodes
        if transpose in self.nodes_to_remove or add in self.nodes_to_remove:
            return

        if self.has_multiple_consumers(transpose.output[0], input_name_to_nodes):
            return

        permutation = OnnxModel.get_node_attribute(transpose, "perm")
        if permutation != [0, 2, 3, 1]:
            return

        inputs = []
        bias = None
        for i in range(2):
            matched_transpose = self.match_transpose_from_nhwc(add.input[i], input_name_to_nodes, output_name_to_node)
            if matched_transpose:
                # When there is an Transpose node before Add (see examples in match_bias_path), we do not need to
                # insert another Transpose node. The existing Transpose node will be removed in prune_graph if it
                # has only one consumer.
                inputs.append(matched_transpose.input[0])
                # See whether it match bias pattern.
                if bias is None:
                    bias = self.match_bias_path(matched_transpose, input_name_to_nodes, output_name_to_node)
            else:
                # Otherwise, insert a Transpose node before Add.
                new_transpose = self.create_transpose_node(add.input[i], [0, 2, 3, 1])
                self.model.add_node(new_transpose, self.this_graph_name)
                inputs.append(new_transpose.output[0])

        skip, broadcast = self.get_skip_index(add, is_channel_last=False)
        if skip < 0:
            return

        inputs = [inputs[1 - skip], node.input[1], node.input[2], inputs[skip]]
        if bias:
            inputs = [*inputs, bias]

        outputs = node.output

        new_node_name = self.model.create_node_name(self.fused_op_type, name_prefix="SkipGroupNorm")
        if self.has_multiple_consumers(add.output[0], input_name_to_nodes):
            add_out_name = new_node_name + "_add_out"
            outputs.append(add_out_name)

            # Insert a Transpose node after add output.
            add_out_transpose = self.create_transpose_node(add_out_name, [0, 3, 1, 2], add.output[0])
            self.model.add_node(add_out_transpose, self.this_graph_name)

        skip_group_norm = helper.make_node(
            self.fused_op_type,
            inputs=inputs,
            outputs=outputs,
            name=new_node_name,
        )
        skip_group_norm.domain = "com.microsoft"

        self.increase_counter(
            f"SkipGroupNorm(add_out={int(len(outputs) > 1)} bias={int(bias is not None)} broadcast={int(broadcast)})"
        )

        # Pass attributes from GroupNorm node to SkipGroupNorm
        for att in node.attribute:
            skip_group_norm.attribute.extend([att])

        self.nodes_to_remove.extend([add, transpose, node])
        self.nodes_to_add.append(skip_group_norm)
        self.node_name_to_graph_name[skip_group_norm.name] = self.this_graph_name
        self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mssql\information_schema.py ===
# dialects/mssql/information_schema.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from ... import cast
from ... import Column
from ... import MetaData
from ... import Table
from ...ext.compiler import compiles
from ...sql import expression
from ...types import Boolean
from ...types import Integer
from ...types import Numeric
from ...types import NVARCHAR
from ...types import String
from ...types import TypeDecorator
from ...types import Unicode


ischema = MetaData()


class CoerceUnicode(TypeDecorator):
    impl = Unicode
    cache_ok = True

    def bind_expression(self, bindvalue):
        return _cast_on_2005(bindvalue)


class _cast_on_2005(expression.ColumnElement):
    def __init__(self, bindvalue):
        self.bindvalue = bindvalue


@compiles(_cast_on_2005)
def _compile(element, compiler, **kw):
    from . import base

    if (
        compiler.dialect.server_version_info is None
        or compiler.dialect.server_version_info < base.MS_2005_VERSION
    ):
        return compiler.process(element.bindvalue, **kw)
    else:
        return compiler.process(cast(element.bindvalue, Unicode), **kw)


schemata = Table(
    "SCHEMATA",
    ischema,
    Column("CATALOG_NAME", CoerceUnicode, key="catalog_name"),
    Column("SCHEMA_NAME", CoerceUnicode, key="schema_name"),
    Column("SCHEMA_OWNER", CoerceUnicode, key="schema_owner"),
    schema="INFORMATION_SCHEMA",
)

tables = Table(
    "TABLES",
    ischema,
    Column("TABLE_CATALOG", CoerceUnicode, key="table_catalog"),
    Column("TABLE_SCHEMA", CoerceUnicode, key="table_schema"),
    Column("TABLE_NAME", CoerceUnicode, key="table_name"),
    Column("TABLE_TYPE", CoerceUnicode, key="table_type"),
    schema="INFORMATION_SCHEMA",
)

columns = Table(
    "COLUMNS",
    ischema,
    Column("TABLE_SCHEMA", CoerceUnicode, key="table_schema"),
    Column("TABLE_NAME", CoerceUnicode, key="table_name"),
    Column("COLUMN_NAME", CoerceUnicode, key="column_name"),
    Column("IS_NULLABLE", Integer, key="is_nullable"),
    Column("DATA_TYPE", String, key="data_type"),
    Column("ORDINAL_POSITION", Integer, key="ordinal_position"),
    Column(
        "CHARACTER_MAXIMUM_LENGTH", Integer, key="character_maximum_length"
    ),
    Column("NUMERIC_PRECISION", Integer, key="numeric_precision"),
    Column("NUMERIC_SCALE", Integer, key="numeric_scale"),
    Column("COLUMN_DEFAULT", Integer, key="column_default"),
    Column("COLLATION_NAME", String, key="collation_name"),
    schema="INFORMATION_SCHEMA",
)

mssql_temp_table_columns = Table(
    "COLUMNS",
    ischema,
    Column("TABLE_SCHEMA", CoerceUnicode, key="table_schema"),
    Column("TABLE_NAME", CoerceUnicode, key="table_name"),
    Column("COLUMN_NAME", CoerceUnicode, key="column_name"),
    Column("IS_NULLABLE", Integer, key="is_nullable"),
    Column("DATA_TYPE", String, key="data_type"),
    Column("ORDINAL_POSITION", Integer, key="ordinal_position"),
    Column(
        "CHARACTER_MAXIMUM_LENGTH", Integer, key="character_maximum_length"
    ),
    Column("NUMERIC_PRECISION", Integer, key="numeric_precision"),
    Column("NUMERIC_SCALE", Integer, key="numeric_scale"),
    Column("COLUMN_DEFAULT", Integer, key="column_default"),
    Column("COLLATION_NAME", String, key="collation_name"),
    schema="tempdb.INFORMATION_SCHEMA",
)

constraints = Table(
    "TABLE_CONSTRAINTS",
    ischema,
    Column("TABLE_SCHEMA", CoerceUnicode, key="table_schema"),
    Column("TABLE_NAME", CoerceUnicode, key="table_name"),
    Column("CONSTRAINT_NAME", CoerceUnicode, key="constraint_name"),
    Column("CONSTRAINT_TYPE", CoerceUnicode, key="constraint_type"),
    schema="INFORMATION_SCHEMA",
)

column_constraints = Table(
    "CONSTRAINT_COLUMN_USAGE",
    ischema,
    Column("TABLE_SCHEMA", CoerceUnicode, key="table_schema"),
    Column("TABLE_NAME", CoerceUnicode, key="table_name"),
    Column("COLUMN_NAME", CoerceUnicode, key="column_name"),
    Column("CONSTRAINT_NAME", CoerceUnicode, key="constraint_name"),
    schema="INFORMATION_SCHEMA",
)

key_constraints = Table(
    "KEY_COLUMN_USAGE",
    ischema,
    Column("TABLE_SCHEMA", CoerceUnicode, key="table_schema"),
    Column("TABLE_NAME", CoerceUnicode, key="table_name"),
    Column("COLUMN_NAME", CoerceUnicode, key="column_name"),
    Column("CONSTRAINT_NAME", CoerceUnicode, key="constraint_name"),
    Column("CONSTRAINT_SCHEMA", CoerceUnicode, key="constraint_schema"),
    Column("ORDINAL_POSITION", Integer, key="ordinal_position"),
    schema="INFORMATION_SCHEMA",
)

ref_constraints = Table(
    "REFERENTIAL_CONSTRAINTS",
    ischema,
    Column("CONSTRAINT_CATALOG", CoerceUnicode, key="constraint_catalog"),
    Column("CONSTRAINT_SCHEMA", CoerceUnicode, key="constraint_schema"),
    Column("CONSTRAINT_NAME", CoerceUnicode, key="constraint_name"),
    # TODO: is CATLOG misspelled ?
    Column(
        "UNIQUE_CONSTRAINT_CATLOG",
        CoerceUnicode,
        key="unique_constraint_catalog",
    ),
    Column(
        "UNIQUE_CONSTRAINT_SCHEMA",
        CoerceUnicode,
        key="unique_constraint_schema",
    ),
    Column(
        "UNIQUE_CONSTRAINT_NAME", CoerceUnicode, key="unique_constraint_name"
    ),
    Column("MATCH_OPTION", String, key="match_option"),
    Column("UPDATE_RULE", String, key="update_rule"),
    Column("DELETE_RULE", String, key="delete_rule"),
    schema="INFORMATION_SCHEMA",
)

views = Table(
    "VIEWS",
    ischema,
    Column("TABLE_CATALOG", CoerceUnicode, key="table_catalog"),
    Column("TABLE_SCHEMA", CoerceUnicode, key="table_schema"),
    Column("TABLE_NAME", CoerceUnicode, key="table_name"),
    Column("VIEW_DEFINITION", CoerceUnicode, key="view_definition"),
    Column("CHECK_OPTION", String, key="check_option"),
    Column("IS_UPDATABLE", String, key="is_updatable"),
    schema="INFORMATION_SCHEMA",
)

computed_columns = Table(
    "computed_columns",
    ischema,
    Column("object_id", Integer),
    Column("name", CoerceUnicode),
    Column("is_computed", Boolean),
    Column("is_persisted", Boolean),
    Column("definition", CoerceUnicode),
    schema="sys",
)

sequences = Table(
    "SEQUENCES",
    ischema,
    Column("SEQUENCE_CATALOG", CoerceUnicode, key="sequence_catalog"),
    Column("SEQUENCE_SCHEMA", CoerceUnicode, key="sequence_schema"),
    Column("SEQUENCE_NAME", CoerceUnicode, key="sequence_name"),
    schema="INFORMATION_SCHEMA",
)


class NumericSqlVariant(TypeDecorator):
    r"""This type casts sql_variant columns in the identity_columns view
    to numeric. This is required because:

    * pyodbc does not support sql_variant
    * pymssql under python 2 return the byte representation of the number,
      int 1 is returned as "\x01\x00\x00\x00". On python 3 it returns the
      correct value as string.
    """

    impl = Unicode
    cache_ok = True

    def column_expression(self, colexpr):
        return cast(colexpr, Numeric(38, 0))


identity_columns = Table(
    "identity_columns",
    ischema,
    Column("object_id", Integer),
    Column("name", CoerceUnicode),
    Column("is_identity", Boolean),
    Column("seed_value", NumericSqlVariant),
    Column("increment_value", NumericSqlVariant),
    Column("last_value", NumericSqlVariant),
    Column("is_not_for_replication", Boolean),
    schema="sys",
)


class NVarcharSqlVariant(TypeDecorator):
    """This type casts sql_variant columns in the extended_properties view
    to nvarchar. This is required because pyodbc does not support sql_variant
    """

    impl = Unicode
    cache_ok = True

    def column_expression(self, colexpr):
        return cast(colexpr, NVARCHAR)


extended_properties = Table(
    "extended_properties",
    ischema,
    Column("class", Integer),  # TINYINT
    Column("class_desc", CoerceUnicode),
    Column("major_id", Integer),
    Column("minor_id", Integer),
    Column("name", CoerceUnicode),
    Column("value", NVarcharSqlVariant),
    schema="sys",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\ctx_fp.py ===
from .ctx_base import StandardBaseContext

import math
import cmath
from . import math2

from . import function_docs

from .libmp import mpf_bernoulli, to_float, int_types
from . import libmp

class FPContext(StandardBaseContext):
    """
    Context for fast low-precision arithmetic (53-bit precision, giving at most
    about 15-digit accuracy), using Python's builtin float and complex.
    """

    def __init__(ctx):
        StandardBaseContext.__init__(ctx)

        # Override SpecialFunctions implementation
        ctx.loggamma = math2.loggamma
        ctx._bernoulli_cache = {}
        ctx.pretty = False

        ctx._init_aliases()

    _mpq = lambda cls, x: float(x[0])/x[1]

    NoConvergence = libmp.NoConvergence

    def _get_prec(ctx): return 53
    def _set_prec(ctx, p): return
    def _get_dps(ctx): return 15
    def _set_dps(ctx, p): return

    _fixed_precision = True

    prec = property(_get_prec, _set_prec)
    dps = property(_get_dps, _set_dps)

    zero = 0.0
    one = 1.0
    eps = math2.EPS
    inf = math2.INF
    ninf = math2.NINF
    nan = math2.NAN
    j = 1j

    # Called by SpecialFunctions.__init__()
    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        if wrap:
            def f_wrapped(ctx, *args, **kwargs):
                convert = ctx.convert
                args = [convert(a) for a in args]
                return f(ctx, *args, **kwargs)
        else:
            f_wrapped = f
        f_wrapped.__doc__ = function_docs.__dict__.get(name, f.__doc__)
        setattr(cls, name, f_wrapped)

    def bernoulli(ctx, n):
        cache = ctx._bernoulli_cache
        if n in cache:
            return cache[n]
        cache[n] = to_float(mpf_bernoulli(n, 53, 'n'), strict=True)
        return cache[n]

    pi = math2.pi
    e = math2.e
    euler = math2.euler
    sqrt2 = 1.4142135623730950488
    sqrt5 = 2.2360679774997896964
    phi = 1.6180339887498948482
    ln2 = 0.69314718055994530942
    ln10 = 2.302585092994045684
    euler = 0.57721566490153286061
    catalan = 0.91596559417721901505
    khinchin = 2.6854520010653064453
    apery = 1.2020569031595942854
    glaisher = 1.2824271291006226369

    absmin = absmax = abs

    def is_special(ctx, x):
        return x - x != 0.0

    def isnan(ctx, x):
        return x != x

    def isinf(ctx, x):
        return abs(x) == math2.INF

    def isnormal(ctx, x):
        if x:
            return x - x == 0.0
        return False

    def isnpint(ctx, x):
        if type(x) is complex:
            if x.imag:
                return False
            x = x.real
        return x <= 0.0 and round(x) == x

    mpf = float
    mpc = complex

    def convert(ctx, x):
        try:
            return float(x)
        except:
            return complex(x)

    power = staticmethod(math2.pow)
    sqrt = staticmethod(math2.sqrt)
    exp = staticmethod(math2.exp)
    ln = log = staticmethod(math2.log)
    cos = staticmethod(math2.cos)
    sin = staticmethod(math2.sin)
    tan = staticmethod(math2.tan)
    cos_sin = staticmethod(math2.cos_sin)
    acos = staticmethod(math2.acos)
    asin = staticmethod(math2.asin)
    atan = staticmethod(math2.atan)
    cosh = staticmethod(math2.cosh)
    sinh = staticmethod(math2.sinh)
    tanh = staticmethod(math2.tanh)
    gamma = staticmethod(math2.gamma)
    rgamma = staticmethod(math2.rgamma)
    fac = factorial = staticmethod(math2.factorial)
    floor = staticmethod(math2.floor)
    ceil = staticmethod(math2.ceil)
    cospi = staticmethod(math2.cospi)
    sinpi = staticmethod(math2.sinpi)
    cbrt = staticmethod(math2.cbrt)
    _nthroot = staticmethod(math2.nthroot)
    _ei = staticmethod(math2.ei)
    _e1 = staticmethod(math2.e1)
    _zeta = _zeta_int = staticmethod(math2.zeta)

    # XXX: math2
    def arg(ctx, z):
        z = complex(z)
        return math.atan2(z.imag, z.real)

    def expj(ctx, x):
        return ctx.exp(ctx.j*x)

    def expjpi(ctx, x):
        return ctx.exp(ctx.j*ctx.pi*x)

    ldexp = math.ldexp
    frexp = math.frexp

    def mag(ctx, z):
        if z:
            return ctx.frexp(abs(z))[1]
        return ctx.ninf

    def isint(ctx, z):
        if hasattr(z, "imag"):   # float/int don't have .real/.imag in py2.5
            if z.imag:
                return False
            z = z.real
        try:
            return z == int(z)
        except:
            return False

    def nint_distance(ctx, z):
        if hasattr(z, "imag"):   # float/int don't have .real/.imag in py2.5
            n = round(z.real)
        else:
            n = round(z)
        if n == z:
            return n, ctx.ninf
        return n, ctx.mag(abs(z-n))

    def _convert_param(ctx, z):
        if type(z) is tuple:
            p, q = z
            return ctx.mpf(p) / q, 'R'
        if hasattr(z, "imag"):    # float/int don't have .real/.imag in py2.5
            intz = int(z.real)
        else:
            intz = int(z)
        if z == intz:
            return intz, 'Z'
        return z, 'R'

    def _is_real_type(ctx, z):
        return isinstance(z, float) or isinstance(z, int_types)

    def _is_complex_type(ctx, z):
        return isinstance(z, complex)

    def hypsum(ctx, p, q, types, coeffs, z, maxterms=6000, **kwargs):
        coeffs = list(coeffs)
        num = range(p)
        den = range(p,p+q)
        tol = ctx.eps
        s = t = 1.0
        k = 0
        while 1:
            for i in num: t *= (coeffs[i]+k)
            for i in den: t /= (coeffs[i]+k)
            k += 1; t /= k; t *= z; s += t
            if abs(t) < tol:
                return s
            if k > maxterms:
                raise ctx.NoConvergence

    def atan2(ctx, x, y):
        return math.atan2(x, y)

    def psi(ctx, m, z):
        m = int(m)
        if m == 0:
            return ctx.digamma(z)
        return (-1)**(m+1) * ctx.fac(m) * ctx.zeta(m+1, z)

    digamma = staticmethod(math2.digamma)

    def harmonic(ctx, x):
        x = ctx.convert(x)
        if x == 0 or x == 1:
            return x
        return ctx.digamma(x+1) + ctx.euler

    nstr = str

    def to_fixed(ctx, x, prec):
        return int(math.ldexp(x, prec))

    def rand(ctx):
        import random
        return random.random()

    _erf = staticmethod(math2.erf)
    _erfc = staticmethod(math2.erfc)

    def sum_accurately(ctx, terms, check_step=1):
        s = ctx.zero
        k = 0
        for term in terms():
            s += term
            if (not k % check_step) and term:
                if abs(term) <= 1e-18*abs(s):
                    break
            k += 1
        return s

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\excel\_odfreader.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._typing import (
    FilePath,
    ReadBuffer,
    Scalar,
    StorageOptions,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

import pandas as pd
from pandas.core.shared_docs import _shared_docs

from pandas.io.excel._base import BaseExcelReader

if TYPE_CHECKING:
    from odf.opendocument import OpenDocument

    from pandas._libs.tslibs.nattype import NaTType


@doc(storage_options=_shared_docs["storage_options"])
class ODFReader(BaseExcelReader["OpenDocument"]):
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[bytes],
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        """
        Read tables out of OpenDocument formatted files.

        Parameters
        ----------
        filepath_or_buffer : str, path to be parsed or
            an open readable stream.
        {storage_options}
        engine_kwargs : dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
        import_optional_dependency("odf")
        super().__init__(
            filepath_or_buffer,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    @property
    def _workbook_class(self) -> type[OpenDocument]:
        from odf.opendocument import OpenDocument

        return OpenDocument

    def load_workbook(
        self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs
    ) -> OpenDocument:
        from odf.opendocument import load

        return load(filepath_or_buffer, **engine_kwargs)

    @property
    def empty_value(self) -> str:
        """Property for compat with other readers."""
        return ""

    @property
    def sheet_names(self) -> list[str]:
        """Return a list of sheet names present in the document"""
        from odf.table import Table

        tables = self.book.getElementsByType(Table)
        return [t.getAttribute("name") for t in tables]

    def get_sheet_by_index(self, index: int):
        from odf.table import Table

        self.raise_if_bad_sheet_by_index(index)
        tables = self.book.getElementsByType(Table)
        return tables[index]

    def get_sheet_by_name(self, name: str):
        from odf.table import Table

        self.raise_if_bad_sheet_by_name(name)
        tables = self.book.getElementsByType(Table)

        for table in tables:
            if table.getAttribute("name") == name:
                return table

        self.close()
        raise ValueError(f"sheet {name} not found")

    def get_sheet_data(
        self, sheet, file_rows_needed: int | None = None
    ) -> list[list[Scalar | NaTType]]:
        """
        Parse an ODF Table into a list of lists
        """
        from odf.table import (
            CoveredTableCell,
            TableCell,
            TableRow,
        )

        covered_cell_name = CoveredTableCell().qname
        table_cell_name = TableCell().qname
        cell_names = {covered_cell_name, table_cell_name}

        sheet_rows = sheet.getElementsByType(TableRow)
        empty_rows = 0
        max_row_len = 0

        table: list[list[Scalar | NaTType]] = []

        for sheet_row in sheet_rows:
            sheet_cells = [
                x
                for x in sheet_row.childNodes
                if hasattr(x, "qname") and x.qname in cell_names
            ]
            empty_cells = 0
            table_row: list[Scalar | NaTType] = []

            for sheet_cell in sheet_cells:
                if sheet_cell.qname == table_cell_name:
                    value = self._get_cell_value(sheet_cell)
                else:
                    value = self.empty_value

                column_repeat = self._get_column_repeat(sheet_cell)

                # Queue up empty values, writing only if content succeeds them
                if value == self.empty_value:
                    empty_cells += column_repeat
                else:
                    table_row.extend([self.empty_value] * empty_cells)
                    empty_cells = 0
                    table_row.extend([value] * column_repeat)

            if max_row_len < len(table_row):
                max_row_len = len(table_row)

            row_repeat = self._get_row_repeat(sheet_row)
            if len(table_row) == 0:
                empty_rows += row_repeat
            else:
                # add blank rows to our table
                table.extend([[self.empty_value]] * empty_rows)
                empty_rows = 0
                table.extend(table_row for _ in range(row_repeat))
            if file_rows_needed is not None and len(table) >= file_rows_needed:
                break

        # Make our table square
        for row in table:
            if len(row) < max_row_len:
                row.extend([self.empty_value] * (max_row_len - len(row)))

        return table

    def _get_row_repeat(self, row) -> int:
        """
        Return number of times this row was repeated
        Repeating an empty row appeared to be a common way
        of representing sparse rows in the table.
        """
        from odf.namespaces import TABLENS

        return int(row.attributes.get((TABLENS, "number-rows-repeated"), 1))

    def _get_column_repeat(self, cell) -> int:
        from odf.namespaces import TABLENS

        return int(cell.attributes.get((TABLENS, "number-columns-repeated"), 1))

    def _get_cell_value(self, cell) -> Scalar | NaTType:
        from odf.namespaces import OFFICENS

        if str(cell) == "#N/A":
            return np.nan

        cell_type = cell.attributes.get((OFFICENS, "value-type"))
        if cell_type == "boolean":
            if str(cell) == "TRUE":
                return True
            return False
        if cell_type is None:
            return self.empty_value
        elif cell_type == "float":
            # GH5394
            cell_value = float(cell.attributes.get((OFFICENS, "value")))
            val = int(cell_value)
            if val == cell_value:
                return val
            return cell_value
        elif cell_type == "percentage":
            cell_value = cell.attributes.get((OFFICENS, "value"))
            return float(cell_value)
        elif cell_type == "string":
            return self._get_cell_string_value(cell)
        elif cell_type == "currency":
            cell_value = cell.attributes.get((OFFICENS, "value"))
            return float(cell_value)
        elif cell_type == "date":
            cell_value = cell.attributes.get((OFFICENS, "date-value"))
            return pd.Timestamp(cell_value)
        elif cell_type == "time":
            stamp = pd.Timestamp(str(cell))
            # cast needed here because Scalar doesn't include datetime.time
            return cast(Scalar, stamp.time())
        else:
            self.close()
            raise ValueError(f"Unrecognized type {cell_type}")

    def _get_cell_string_value(self, cell) -> str:
        """
        Find and decode OpenDocument text:s tags that represent
        a run length encoded sequence of space characters.
        """
        from odf.element import Element
        from odf.namespaces import TEXTNS
        from odf.office import Annotation
        from odf.text import S

        office_annotation = Annotation().qname
        text_s = S().qname

        value = []

        for fragment in cell.childNodes:
            if isinstance(fragment, Element):
                if fragment.qname == text_s:
                    spaces = int(fragment.attributes.get((TEXTNS, "c"), 1))
                    value.append(" " * spaces)
                elif fragment.qname == office_annotation:
                    continue
                else:
                    # recursive impl needed in case of nested fragments
                    # with multiple spaces
                    # https://github.com/pandas-dev/pandas/pull/36175#discussion_r484639704
                    value.append(self._get_cell_string_value(fragment))
            else:
                value.append(str(fragment).strip("\n"))
        return "".join(value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\console\ansi.py ===
# -*- coding: ISO-8859-1 -*-


import os
import re
import sys

terminal_escape = re.compile("(\001?\033\\[[0-9;]*m\002?)")
escape_parts = re.compile("\001?\033\\[([0-9;]*)m\002?")


class AnsiState(object):
    def __init__(
        self,
        bold=False,
        inverse=False,
        color="white",
        background="black",
        backgroundbold=False,
    ):
        self.bold = bold
        self.inverse = inverse
        self.color = color
        self.background = background
        self.backgroundbold = backgroundbold

    trtable = {
        "black": 0,
        "red": 4,
        "green": 2,
        "yellow": 6,
        "blue": 1,
        "magenta": 5,
        "cyan": 3,
        "white": 7,
    }
    revtable = dict(zip(trtable.values(), trtable.keys()))

    def get_winattr(self):
        attr = 0
        if self.bold:
            attr |= 0x0008
        if self.backgroundbold:
            attr |= 0x0080
        if self.inverse:
            attr |= 0x4000
        attr |= self.trtable[self.color]
        attr |= self.trtable[self.background] << 4
        return attr

    def set_winattr(self, attr):
        self.bold = bool(attr & 0x0008)
        self.backgroundbold = bool(attr & 0x0080)
        self.inverse = bool(attr & 0x4000)
        self.color = self.revtable[attr & 0x0007]
        self.background = self.revtable[(attr & 0x0070) >> 4]

    winattr = property(get_winattr, set_winattr)

    def __repr__(self):
        return (
            "AnsiState(bold=%s,inverse=%s,color=%9s,"
            "background=%9s,backgroundbold=%s)# 0x%x"
            % (
                self.bold,
                self.inverse,
                '"%s"' % self.color,
                '"%s"' % self.background,
                self.backgroundbold,
                self.winattr,
            )
        )

    def copy(self):
        x = AnsiState()
        x.bold = self.bold
        x.inverse = self.inverse
        x.color = self.color
        x.background = self.background
        x.backgroundbold = self.backgroundbold
        return x


defaultstate = AnsiState(False, False, "white")

trtable = {
    0: "black",
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "magenta",
    6: "cyan",
    7: "white",
}


class AnsiWriter(object):
    def __init__(self, default=defaultstate):
        if isinstance(defaultstate, AnsiState):
            self.defaultstate = default
        else:
            self.defaultstate = AnsiState()
            self.defaultstate.winattr = defaultstate

    def write_color(self, text, attr=None):
        """write text at current cursor position and interpret color escapes.

        return the number of characters written.
        """
        if isinstance(attr, AnsiState):
            defaultstate = attr
        elif attr is None:  # use attribute form initial console
            attr = self.defaultstate.copy()
        else:
            defaultstate = AnsiState()
            defaultstate.winattr = attr
            attr = defaultstate
        chunks = terminal_escape.split(text)
        n = 0  # count the characters we actually write, omitting the escapes
        res = []
        for chunk in chunks:
            m = escape_parts.match(chunk)
            if m:
                parts = m.group(1).split(";")
                if len(parts) == 1 and parts[0] == "0":
                    attr = self.defaultstate.copy()
                    continue
                for part in parts:
                    if part == "0":  # No text attribute
                        attr = self.defaultstate.copy()
                        attr.bold = False
                    elif part == "7":  # switch on reverse
                        attr.inverse = True
                    # switch on bold (i.e. intensify foreground color)
                    elif part == "1":
                        attr.bold = True
                    elif len(part) == 2:
                        if part == "22":  # Normal foreground color
                            attr.bold = False
                        elif "30" <= part <= "37":  # set foreground color
                            attr.color = trtable[int(part) - 30]
                        elif part == "39":  # Default foreground color
                            attr.color = self.defaultstate.color
                        elif "40" <= part <= "47":  # set background color
                            attr.background = trtable[int(part) - 40]
                        elif part == "49":  # Default background color
                            attr.background = self.defaultstate.background
                continue
            n += len(chunk)

            res.append((attr.copy(), chunk))

        return n, res

    def parse_color(self, text, attr=None):
        n, res = self.write_color(text, attr)
        return n, [attr.winattr for attr, text in res]


def write_color(text, attr=None):
    a = AnsiWriter(defaultstate)
    return a.write_color(text, attr)


def write_color_old(text, attr=None):
    """write text at current cursor position and interpret color escapes.

    return the number of characters written.
    """
    res = []
    chunks = terminal_escape.split(text)
    n = 0  # count the characters we actually write, omitting the escapes
    if attr is None:  # use attribute from initial console
        attr = 15
    for chunk in chunks:
        m = escape_parts.match(chunk)
        if m:
            for part in m.group(1).split(";"):
                if part == "0":  # No text attribute
                    attr = 0
                elif part == "7":  # switch on reverse
                    attr |= 0x4000
                # switch on bold (i.e. intensify foreground color)
                if part == "1":
                    attr |= 0x08
                elif len(part) == 2 and "30" <= part <= "37":  # set foreground color
                    part = int(part) - 30
                    # we have to mirror bits
                    attr = (
                        (attr & ~0x07)
                        | ((part & 0x1) << 2)
                        | (part & 0x2)
                        | ((part & 0x4) >> 2)
                    )
                elif len(part) == 2 and "40" <= part <= "47":  # set background color
                    part = int(part) - 40
                    # we have to mirror bits
                    attr = (
                        (attr & ~0x70)
                        | ((part & 0x1) << 6)
                        | ((part & 0x2) << 4)
                        | ((part & 0x4) << 2)
                    )
                # ignore blink, underline and anything we don't understand
            continue
        n += len(chunk)
        if chunk:
            res.append(("0x%x" % attr, chunk))
    return res


# trtable={0:"black",1:"red",2:"green",3:"yellow",4:"blue",5:"magenta",6:"cyan",7:"white"}

if __name__ == "__main__x":
    import pprint

    pprint = pprint.pprint

    s = "\033[0;31mred\033[0;32mgreen\033[0;33myellow\033[0;34mblue\033[0;35mmagenta\033[0;36mcyan\033[0;37mwhite\033[0m"
    pprint(write_color(s))
    pprint(write_color_old(s))
    s = "\033[1;31mred\033[1;32mgreen\033[1;33myellow\033[1;34mblue\033[1;35mmagenta\033[1;36mcyan\033[1;37mwhite\033[0m"
    pprint(write_color(s))
    pprint(write_color_old(s))

    s = "\033[0;7;31mred\033[0;7;32mgreen\033[0;7;33myellow\033[0;7;34mblue\033[0;7;35mmagenta\033[0;7;36mcyan\033[0;7;37mwhite\033[0m"
    pprint(write_color(s))
    pprint(write_color_old(s))
    s = "\033[1;7;31mred\033[1;7;32mgreen\033[1;7;33myellow\033[1;7;34mblue\033[1;7;35mmagenta\033[1;7;36mcyan\033[1;7;37mwhite\033[0m"
    pprint(write_color(s))
    pprint(write_color_old(s))


if __name__ == "__main__":
    import pprint

    import console

    pprint = pprint.pprint

    c = console.Console()
    c.write_color("dhsjdhs")
    c.write_color("\033[0;32mIn [\033[1;32m1\033[0;32m]:")
    print
    pprint(write_color("\033[0;32mIn [\033[1;32m1\033[0;32m]:"))

if __name__ == "__main__x":
    import pprint

    pprint = pprint.pprint
    s = "\033[0;31mred\033[0;32mgreen\033[0;33myellow\033[0;34mblue\033[0;35mmagenta\033[0;36mcyan\033[0;37mwhite\033[0m"
    pprint(write_color(s))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mysql\mysqlconnector.py ===
# dialects/mysql/mysqlconnector.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


r"""
.. dialect:: mysql+mysqlconnector
    :name: MySQL Connector/Python
    :dbapi: myconnpy
    :connectstring: mysql+mysqlconnector://<user>:<password>@<host>[:<port>]/<dbname>
    :url: https://pypi.org/project/mysql-connector-python/

Driver Status
-------------

MySQL Connector/Python is supported as of SQLAlchemy 2.0.39 to the
degree which the driver is functional.   There are still ongoing issues
with features such as server side cursors which remain disabled until
upstream issues are repaired.

.. warning:: The MySQL Connector/Python driver published by Oracle is subject
   to frequent, major regressions of essential functionality such as being able
   to correctly persist simple binary strings which indicate it is not well
   tested.  The SQLAlchemy project is not able to maintain this dialect fully as
   regressions in the driver prevent it from being included in continuous
   integration.

.. versionchanged:: 2.0.39

    The MySQL Connector/Python dialect has been updated to support the
    latest version of this DBAPI.   Previously, MySQL Connector/Python
    was not fully supported.  However, support remains limited due to ongoing
    regressions introduced in this driver.

Connecting to MariaDB with MySQL Connector/Python
--------------------------------------------------

MySQL Connector/Python may attempt to pass an incompatible collation to the
database when connecting to MariaDB.  Experimentation has shown that using
``?charset=utf8mb4&collation=utfmb4_general_ci`` or similar MariaDB-compatible
charset/collation will allow connectivity.


"""  # noqa

import re

from .base import BIT
from .base import MariaDBIdentifierPreparer
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from .base import MySQLIdentifierPreparer
from .mariadb import MariaDBDialect
from ... import util


class MySQLExecutionContext_mysqlconnector(MySQLExecutionContext):
    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor(buffered=False)

    def create_default_cursor(self):
        return self._dbapi_connection.cursor(buffered=True)


class MySQLCompiler_mysqlconnector(MySQLCompiler):
    def visit_mod_binary(self, binary, operator, **kw):
        return (
            self.process(binary.left, **kw)
            + " % "
            + self.process(binary.right, **kw)
        )


class IdentifierPreparerCommon_mysqlconnector:
    @property
    def _double_percents(self):
        return False

    @_double_percents.setter
    def _double_percents(self, value):
        pass

    def _escape_identifier(self, value):
        value = value.replace(self.escape_quote, self.escape_to_quote)
        return value


class MySQLIdentifierPreparer_mysqlconnector(
    IdentifierPreparerCommon_mysqlconnector, MySQLIdentifierPreparer
):
    pass


class MariaDBIdentifierPreparer_mysqlconnector(
    IdentifierPreparerCommon_mysqlconnector, MariaDBIdentifierPreparer
):
    pass


class _myconnpyBIT(BIT):
    def result_processor(self, dialect, coltype):
        """MySQL-connector already converts mysql bits, so."""

        return None


class MySQLDialect_mysqlconnector(MySQLDialect):
    driver = "mysqlconnector"
    supports_statement_cache = True

    supports_sane_rowcount = True
    supports_sane_multi_rowcount = True

    supports_native_decimal = True

    supports_native_bit = True

    # not until https://bugs.mysql.com/bug.php?id=117548
    supports_server_side_cursors = False

    default_paramstyle = "format"
    statement_compiler = MySQLCompiler_mysqlconnector

    execution_ctx_cls = MySQLExecutionContext_mysqlconnector

    preparer = MySQLIdentifierPreparer_mysqlconnector

    colspecs = util.update_copy(MySQLDialect.colspecs, {BIT: _myconnpyBIT})

    @classmethod
    def import_dbapi(cls):
        from mysql import connector

        return connector

    def do_ping(self, dbapi_connection):
        dbapi_connection.ping(False)
        return True

    def create_connect_args(self, url):
        opts = url.translate_connect_args(username="user")

        opts.update(url.query)

        util.coerce_kw_type(opts, "allow_local_infile", bool)
        util.coerce_kw_type(opts, "autocommit", bool)
        util.coerce_kw_type(opts, "buffered", bool)
        util.coerce_kw_type(opts, "client_flag", int)
        util.coerce_kw_type(opts, "compress", bool)
        util.coerce_kw_type(opts, "connection_timeout", int)
        util.coerce_kw_type(opts, "connect_timeout", int)
        util.coerce_kw_type(opts, "consume_results", bool)
        util.coerce_kw_type(opts, "force_ipv6", bool)
        util.coerce_kw_type(opts, "get_warnings", bool)
        util.coerce_kw_type(opts, "pool_reset_session", bool)
        util.coerce_kw_type(opts, "pool_size", int)
        util.coerce_kw_type(opts, "raise_on_warnings", bool)
        util.coerce_kw_type(opts, "raw", bool)
        util.coerce_kw_type(opts, "ssl_verify_cert", bool)
        util.coerce_kw_type(opts, "use_pure", bool)
        util.coerce_kw_type(opts, "use_unicode", bool)

        # note that "buffered" is set to False by default in MySQL/connector
        # python.  If you set it to True, then there is no way to get a server
        # side cursor because the logic is written to disallow that.

        # leaving this at True until
        # https://bugs.mysql.com/bug.php?id=117548 can be fixed
        opts["buffered"] = True

        # FOUND_ROWS must be set in ClientFlag to enable
        # supports_sane_rowcount.
        if self.dbapi is not None:
            try:
                from mysql.connector.constants import ClientFlag

                client_flags = opts.get(
                    "client_flags", ClientFlag.get_default()
                )
                client_flags |= ClientFlag.FOUND_ROWS
                opts["client_flags"] = client_flags
            except Exception:
                pass

        return [[], opts]

    @util.memoized_property
    def _mysqlconnector_version_info(self):
        if self.dbapi and hasattr(self.dbapi, "__version__"):
            m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", self.dbapi.__version__)
            if m:
                return tuple(int(x) for x in m.group(1, 2, 3) if x is not None)

    def _detect_charset(self, connection):
        return connection.connection.charset

    def _extract_error_code(self, exception):
        return exception.errno

    def is_disconnect(self, e, connection, cursor):
        errnos = (2006, 2013, 2014, 2045, 2055, 2048)
        exceptions = (
            self.dbapi.OperationalError,
            self.dbapi.InterfaceError,
            self.dbapi.ProgrammingError,
        )
        if isinstance(e, exceptions):
            return (
                e.errno in errnos
                or "MySQL Connection not available." in str(e)
                or "Connection to MySQL is not available" in str(e)
            )
        else:
            return False

    def _compat_fetchall(self, rp, charset=None):
        return rp.fetchall()

    def _compat_fetchone(self, rp, charset=None):
        return rp.fetchone()

    def get_isolation_level_values(self, dbapi_connection):
        return (
            "SERIALIZABLE",
            "READ UNCOMMITTED",
            "READ COMMITTED",
            "REPEATABLE READ",
            "AUTOCOMMIT",
        )

    def set_isolation_level(self, connection, level):
        if level == "AUTOCOMMIT":
            connection.autocommit = True
        else:
            connection.autocommit = False
            super().set_isolation_level(connection, level)


class MariaDBDialect_mysqlconnector(
    MariaDBDialect, MySQLDialect_mysqlconnector
):
    supports_statement_cache = True
    _allows_uuid_binds = False
    preparer = MariaDBIdentifierPreparer_mysqlconnector


dialect = MySQLDialect_mysqlconnector
mariadb_dialect = MariaDBDialect_mysqlconnector

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\_antlr\autolevlexer.py ===
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,0,49,393,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,
        2,6,7,6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,
        13,7,13,2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,
        19,2,20,7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,
        26,7,26,2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,
        32,2,33,7,33,2,34,7,34,2,35,7,35,2,36,7,36,2,37,7,37,2,38,7,38,2,
        39,7,39,2,40,7,40,2,41,7,41,2,42,7,42,2,43,7,43,2,44,7,44,2,45,7,
        45,2,46,7,46,2,47,7,47,2,48,7,48,2,49,7,49,2,50,7,50,1,0,1,0,1,1,
        1,1,1,2,1,2,1,3,1,3,1,3,1,4,1,4,1,4,1,5,1,5,1,5,1,6,1,6,1,6,1,7,
        1,7,1,7,1,8,1,8,1,8,1,9,1,9,1,10,1,10,1,11,1,11,1,12,1,12,1,13,1,
        13,1,14,1,14,1,15,1,15,1,16,1,16,1,17,1,17,1,18,1,18,1,19,1,19,1,
        20,1,20,1,21,1,21,1,21,1,22,1,22,1,22,1,22,1,23,1,23,1,24,1,24,1,
        25,1,25,1,26,1,26,1,26,1,26,1,26,1,27,1,27,1,27,1,27,1,27,1,27,1,
        27,1,27,1,28,1,28,1,28,1,28,1,28,1,28,3,28,184,8,28,1,29,1,29,1,
        29,1,29,1,29,1,29,1,29,1,30,1,30,1,30,1,30,1,30,1,31,1,31,1,31,1,
        31,1,31,1,31,1,31,1,31,1,31,1,31,1,31,1,32,1,32,1,32,1,32,1,32,1,
        32,1,32,1,33,1,33,1,33,1,33,1,33,1,33,1,33,1,33,1,33,1,33,1,34,1,
        34,1,34,1,34,1,34,1,34,3,34,232,8,34,1,35,1,35,1,35,1,35,1,35,1,
        35,3,35,240,8,35,1,36,1,36,1,36,1,36,1,36,1,36,1,36,1,36,1,36,3,
        36,251,8,36,1,37,1,37,1,37,1,37,1,37,1,37,3,37,259,8,37,1,38,1,38,
        1,38,1,38,1,38,1,38,1,38,1,38,1,38,3,38,270,8,38,1,39,1,39,1,39,
        1,39,1,39,1,39,1,39,1,39,1,39,1,39,3,39,282,8,39,1,40,1,40,1,40,
        1,40,1,40,1,40,1,40,1,40,1,40,1,40,1,41,1,41,1,41,1,41,1,41,1,41,
        1,41,1,41,1,41,3,41,303,8,41,1,42,1,42,1,42,1,42,1,42,1,42,1,42,
        1,42,1,42,1,42,1,42,1,42,1,42,1,42,1,42,3,42,320,8,42,1,43,5,43,
        323,8,43,10,43,12,43,326,9,43,1,44,1,44,1,45,4,45,331,8,45,11,45,
        12,45,332,1,46,4,46,336,8,46,11,46,12,46,337,1,46,1,46,5,46,342,
        8,46,10,46,12,46,345,9,46,1,46,1,46,4,46,349,8,46,11,46,12,46,350,
        3,46,353,8,46,1,47,1,47,1,47,1,47,1,47,1,47,1,47,1,47,1,47,3,47,
        364,8,47,1,48,1,48,5,48,368,8,48,10,48,12,48,371,9,48,1,48,3,48,
        374,8,48,1,48,1,48,1,48,1,48,1,49,1,49,5,49,382,8,49,10,49,12,49,
        385,9,49,1,50,4,50,388,8,50,11,50,12,50,389,1,50,1,50,1,369,0,51,
        1,1,3,2,5,3,7,4,9,5,11,6,13,7,15,8,17,9,19,10,21,11,23,12,25,13,
        27,14,29,15,31,16,33,17,35,18,37,19,39,20,41,21,43,22,45,23,47,24,
        49,25,51,26,53,27,55,28,57,29,59,30,61,31,63,32,65,33,67,34,69,35,
        71,36,73,37,75,38,77,39,79,40,81,41,83,42,85,43,87,0,89,0,91,44,
        93,45,95,46,97,47,99,48,101,49,1,0,24,2,0,77,77,109,109,2,0,65,65,
        97,97,2,0,83,83,115,115,2,0,73,73,105,105,2,0,78,78,110,110,2,0,
        69,69,101,101,2,0,82,82,114,114,2,0,84,84,116,116,2,0,80,80,112,
        112,2,0,85,85,117,117,2,0,79,79,111,111,2,0,86,86,118,118,2,0,89,
        89,121,121,2,0,67,67,99,99,2,0,68,68,100,100,2,0,87,87,119,119,2,
        0,70,70,102,102,2,0,66,66,98,98,2,0,76,76,108,108,2,0,71,71,103,
        103,1,0,48,57,2,0,65,90,97,122,4,0,48,57,65,90,95,95,97,122,4,0,
        9,10,13,13,32,32,38,38,410,0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,0,0,0,0,
        7,1,0,0,0,0,9,1,0,0,0,0,11,1,0,0,0,0,13,1,0,0,0,0,15,1,0,0,0,0,17,
        1,0,0,0,0,19,1,0,0,0,0,21,1,0,0,0,0,23,1,0,0,0,0,25,1,0,0,0,0,27,
        1,0,0,0,0,29,1,0,0,0,0,31,1,0,0,0,0,33,1,0,0,0,0,35,1,0,0,0,0,37,
        1,0,0,0,0,39,1,0,0,0,0,41,1,0,0,0,0,43,1,0,0,0,0,45,1,0,0,0,0,47,
        1,0,0,0,0,49,1,0,0,0,0,51,1,0,0,0,0,53,1,0,0,0,0,55,1,0,0,0,0,57,
        1,0,0,0,0,59,1,0,0,0,0,61,1,0,0,0,0,63,1,0,0,0,0,65,1,0,0,0,0,67,
        1,0,0,0,0,69,1,0,0,0,0,71,1,0,0,0,0,73,1,0,0,0,0,75,1,0,0,0,0,77,
        1,0,0,0,0,79,1,0,0,0,0,81,1,0,0,0,0,83,1,0,0,0,0,85,1,0,0,0,0,91,
        1,0,0,0,0,93,1,0,0,0,0,95,1,0,0,0,0,97,1,0,0,0,0,99,1,0,0,0,0,101,
        1,0,0,0,1,103,1,0,0,0,3,105,1,0,0,0,5,107,1,0,0,0,7,109,1,0,0,0,
        9,112,1,0,0,0,11,115,1,0,0,0,13,118,1,0,0,0,15,121,1,0,0,0,17,124,
        1,0,0,0,19,127,1,0,0,0,21,129,1,0,0,0,23,131,1,0,0,0,25,133,1,0,
        0,0,27,135,1,0,0,0,29,137,1,0,0,0,31,139,1,0,0,0,33,141,1,0,0,0,
        35,143,1,0,0,0,37,145,1,0,0,0,39,147,1,0,0,0,41,149,1,0,0,0,43,151,
        1,0,0,0,45,154,1,0,0,0,47,158,1,0,0,0,49,160,1,0,0,0,51,162,1,0,
        0,0,53,164,1,0,0,0,55,169,1,0,0,0,57,177,1,0,0,0,59,185,1,0,0,0,
        61,192,1,0,0,0,63,197,1,0,0,0,65,208,1,0,0,0,67,215,1,0,0,0,69,225,
        1,0,0,0,71,233,1,0,0,0,73,241,1,0,0,0,75,252,1,0,0,0,77,260,1,0,
        0,0,79,271,1,0,0,0,81,283,1,0,0,0,83,293,1,0,0,0,85,304,1,0,0,0,
        87,324,1,0,0,0,89,327,1,0,0,0,91,330,1,0,0,0,93,352,1,0,0,0,95,363,
        1,0,0,0,97,365,1,0,0,0,99,379,1,0,0,0,101,387,1,0,0,0,103,104,5,
        91,0,0,104,2,1,0,0,0,105,106,5,93,0,0,106,4,1,0,0,0,107,108,5,61,
        0,0,108,6,1,0,0,0,109,110,5,43,0,0,110,111,5,61,0,0,111,8,1,0,0,
        0,112,113,5,45,0,0,113,114,5,61,0,0,114,10,1,0,0,0,115,116,5,58,
        0,0,116,117,5,61,0,0,117,12,1,0,0,0,118,119,5,42,0,0,119,120,5,61,
        0,0,120,14,1,0,0,0,121,122,5,47,0,0,122,123,5,61,0,0,123,16,1,0,
        0,0,124,125,5,94,0,0,125,126,5,61,0,0,126,18,1,0,0,0,127,128,5,44,
        0,0,128,20,1,0,0,0,129,130,5,39,0,0,130,22,1,0,0,0,131,132,5,40,
        0,0,132,24,1,0,0,0,133,134,5,41,0,0,134,26,1,0,0,0,135,136,5,123,
        0,0,136,28,1,0,0,0,137,138,5,125,0,0,138,30,1,0,0,0,139,140,5,58,
        0,0,140,32,1,0,0,0,141,142,5,43,0,0,142,34,1,0,0,0,143,144,5,45,
        0,0,144,36,1,0,0,0,145,146,5,59,0,0,146,38,1,0,0,0,147,148,5,46,
        0,0,148,40,1,0,0,0,149,150,5,62,0,0,150,42,1,0,0,0,151,152,5,48,
        0,0,152,153,5,62,0,0,153,44,1,0,0,0,154,155,5,49,0,0,155,156,5,62,
        0,0,156,157,5,62,0,0,157,46,1,0,0,0,158,159,5,94,0,0,159,48,1,0,
        0,0,160,161,5,42,0,0,161,50,1,0,0,0,162,163,5,47,0,0,163,52,1,0,
        0,0,164,165,7,0,0,0,165,166,7,1,0,0,166,167,7,2,0,0,167,168,7,2,
        0,0,168,54,1,0,0,0,169,170,7,3,0,0,170,171,7,4,0,0,171,172,7,5,0,
        0,172,173,7,6,0,0,173,174,7,7,0,0,174,175,7,3,0,0,175,176,7,1,0,
        0,176,56,1,0,0,0,177,178,7,3,0,0,178,179,7,4,0,0,179,180,7,8,0,0,
        180,181,7,9,0,0,181,183,7,7,0,0,182,184,7,2,0,0,183,182,1,0,0,0,
        183,184,1,0,0,0,184,58,1,0,0,0,185,186,7,10,0,0,186,187,7,9,0,0,
        187,188,7,7,0,0,188,189,7,8,0,0,189,190,7,9,0,0,190,191,7,7,0,0,
        191,60,1,0,0,0,192,193,7,2,0,0,193,194,7,1,0,0,194,195,7,11,0,0,
        195,196,7,5,0,0,196,62,1,0,0,0,197,198,7,9,0,0,198,199,7,4,0,0,199,
        200,7,3,0,0,200,201,7,7,0,0,201,202,7,2,0,0,202,203,7,12,0,0,203,
        204,7,2,0,0,204,205,7,7,0,0,205,206,7,5,0,0,206,207,7,0,0,0,207,
        64,1,0,0,0,208,209,7,5,0,0,209,210,7,4,0,0,210,211,7,13,0,0,211,
        212,7,10,0,0,212,213,7,14,0,0,213,214,7,5,0,0,214,66,1,0,0,0,215,
        216,7,4,0,0,216,217,7,5,0,0,217,218,7,15,0,0,218,219,7,7,0,0,219,
        220,7,10,0,0,220,221,7,4,0,0,221,222,7,3,0,0,222,223,7,1,0,0,223,
        224,7,4,0,0,224,68,1,0,0,0,225,226,7,16,0,0,226,227,7,6,0,0,227,
        228,7,1,0,0,228,229,7,0,0,0,229,231,7,5,0,0,230,232,7,2,0,0,231,
        230,1,0,0,0,231,232,1,0,0,0,232,70,1,0,0,0,233,234,7,17,0,0,234,
        235,7,10,0,0,235,236,7,14,0,0,236,237,7,3,0,0,237,239,7,5,0,0,238,
        240,7,2,0,0,239,238,1,0,0,0,239,240,1,0,0,0,240,72,1,0,0,0,241,242,
        7,8,0,0,242,243,7,1,0,0,243,244,7,6,0,0,244,245,7,7,0,0,245,246,
        7,3,0,0,246,247,7,13,0,0,247,248,7,18,0,0,248,250,7,5,0,0,249,251,
        7,2,0,0,250,249,1,0,0,0,250,251,1,0,0,0,251,74,1,0,0,0,252,253,7,
        8,0,0,253,254,7,10,0,0,254,255,7,3,0,0,255,256,7,4,0,0,256,258,7,
        7,0,0,257,259,7,2,0,0,258,257,1,0,0,0,258,259,1,0,0,0,259,76,1,0,
        0,0,260,261,7,13,0,0,261,262,7,10,0,0,262,263,7,4,0,0,263,264,7,
        2,0,0,264,265,7,7,0,0,265,266,7,1,0,0,266,267,7,4,0,0,267,269,7,
        7,0,0,268,270,7,2,0,0,269,268,1,0,0,0,269,270,1,0,0,0,270,78,1,0,
        0,0,271,272,7,2,0,0,272,273,7,8,0,0,273,274,7,5,0,0,274,275,7,13,
        0,0,275,276,7,3,0,0,276,277,7,16,0,0,277,278,7,3,0,0,278,279,7,5,
        0,0,279,281,7,14,0,0,280,282,7,2,0,0,281,280,1,0,0,0,281,282,1,0,
        0,0,282,80,1,0,0,0,283,284,7,3,0,0,284,285,7,0,0,0,285,286,7,1,0,
        0,286,287,7,19,0,0,287,288,7,3,0,0,288,289,7,4,0,0,289,290,7,1,0,
        0,290,291,7,6,0,0,291,292,7,12,0,0,292,82,1,0,0,0,293,294,7,11,0,
        0,294,295,7,1,0,0,295,296,7,6,0,0,296,297,7,3,0,0,297,298,7,1,0,
        0,298,299,7,17,0,0,299,300,7,18,0,0,300,302,7,5,0,0,301,303,7,2,
        0,0,302,301,1,0,0,0,302,303,1,0,0,0,303,84,1,0,0,0,304,305,7,0,0,
        0,305,306,7,10,0,0,306,307,7,7,0,0,307,308,7,3,0,0,308,309,7,10,
        0,0,309,310,7,4,0,0,310,311,7,11,0,0,311,312,7,1,0,0,312,313,7,6,
        0,0,313,314,7,3,0,0,314,315,7,1,0,0,315,316,7,17,0,0,316,317,7,18,
        0,0,317,319,7,5,0,0,318,320,7,2,0,0,319,318,1,0,0,0,319,320,1,0,
        0,0,320,86,1,0,0,0,321,323,5,39,0,0,322,321,1,0,0,0,323,326,1,0,
        0,0,324,322,1,0,0,0,324,325,1,0,0,0,325,88,1,0,0,0,326,324,1,0,0,
        0,327,328,7,20,0,0,328,90,1,0,0,0,329,331,7,20,0,0,330,329,1,0,0,
        0,331,332,1,0,0,0,332,330,1,0,0,0,332,333,1,0,0,0,333,92,1,0,0,0,
        334,336,3,89,44,0,335,334,1,0,0,0,336,337,1,0,0,0,337,335,1,0,0,
        0,337,338,1,0,0,0,338,339,1,0,0,0,339,343,5,46,0,0,340,342,3,89,
        44,0,341,340,1,0,0,0,342,345,1,0,0,0,343,341,1,0,0,0,343,344,1,0,
        0,0,344,353,1,0,0,0,345,343,1,0,0,0,346,348,5,46,0,0,347,349,3,89,
        44,0,348,347,1,0,0,0,349,350,1,0,0,0,350,348,1,0,0,0,350,351,1,0,
        0,0,351,353,1,0,0,0,352,335,1,0,0,0,352,346,1,0,0,0,353,94,1,0,0,
        0,354,355,3,93,46,0,355,356,5,69,0,0,356,357,3,91,45,0,357,364,1,
        0,0,0,358,359,3,93,46,0,359,360,5,69,0,0,360,361,5,45,0,0,361,362,
        3,91,45,0,362,364,1,0,0,0,363,354,1,0,0,0,363,358,1,0,0,0,364,96,
        1,0,0,0,365,369,5,37,0,0,366,368,9,0,0,0,367,366,1,0,0,0,368,371,
        1,0,0,0,369,370,1,0,0,0,369,367,1,0,0,0,370,373,1,0,0,0,371,369,
        1,0,0,0,372,374,5,13,0,0,373,372,1,0,0,0,373,374,1,0,0,0,374,375,
        1,0,0,0,375,376,5,10,0,0,376,377,1,0,0,0,377,378,6,48,0,0,378,98,
        1,0,0,0,379,383,7,21,0,0,380,382,7,22,0,0,381,380,1,0,0,0,382,385,
        1,0,0,0,383,381,1,0,0,0,383,384,1,0,0,0,384,100,1,0,0,0,385,383,
        1,0,0,0,386,388,7,23,0,0,387,386,1,0,0,0,388,389,1,0,0,0,389,387,
        1,0,0,0,389,390,1,0,0,0,390,391,1,0,0,0,391,392,6,50,0,0,392,102,
        1,0,0,0,21,0,183,231,239,250,258,269,281,302,319,324,332,337,343,
        350,352,363,369,373,383,389,1,6,0,0
    ]

class AutolevLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    T__11 = 12
    T__12 = 13
    T__13 = 14
    T__14 = 15
    T__15 = 16
    T__16 = 17
    T__17 = 18
    T__18 = 19
    T__19 = 20
    T__20 = 21
    T__21 = 22
    T__22 = 23
    T__23 = 24
    T__24 = 25
    T__25 = 26
    Mass = 27
    Inertia = 28
    Input = 29
    Output = 30
    Save = 31
    UnitSystem = 32
    Encode = 33
    Newtonian = 34
    Frames = 35
    Bodies = 36
    Particles = 37
    Points = 38
    Constants = 39
    Specifieds = 40
    Imaginary = 41
    Variables = 42
    MotionVariables = 43
    INT = 44
    FLOAT = 45
    EXP = 46
    LINE_COMMENT = 47
    ID = 48
    WS = 49

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'['", "']'", "'='", "'+='", "'-='", "':='", "'*='", "'/='",
            "'^='", "','", "'''", "'('", "')'", "'{'", "'}'", "':'", "'+'",
            "'-'", "';'", "'.'", "'>'", "'0>'", "'1>>'", "'^'", "'*'", "'/'" ]

    symbolicNames = [ "<INVALID>",
            "Mass", "Inertia", "Input", "Output", "Save", "UnitSystem",
            "Encode", "Newtonian", "Frames", "Bodies", "Particles", "Points",
            "Constants", "Specifieds", "Imaginary", "Variables", "MotionVariables",
            "INT", "FLOAT", "EXP", "LINE_COMMENT", "ID", "WS" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "T__6",
                  "T__7", "T__8", "T__9", "T__10", "T__11", "T__12", "T__13",
                  "T__14", "T__15", "T__16", "T__17", "T__18", "T__19",
                  "T__20", "T__21", "T__22", "T__23", "T__24", "T__25",
                  "Mass", "Inertia", "Input", "Output", "Save", "UnitSystem",
                  "Encode", "Newtonian", "Frames", "Bodies", "Particles",
                  "Points", "Constants", "Specifieds", "Imaginary", "Variables",
                  "MotionVariables", "DIFF", "DIGIT", "INT", "FLOAT", "EXP",
                  "LINE_COMMENT", "ID", "WS" ]

    grammarFileName = "Autolev.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.11.1")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None



# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\medium.py ===
"""
**Contains**

* Medium
"""
from sympy.physics.units import second, meter, kilogram, ampere

__all__ = ['Medium']

from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units import speed_of_light, u0, e0


c = speed_of_light.convert_to(meter/second)
_e0mksa = e0.convert_to(ampere**2*second**4/(kilogram*meter**3))
_u0mksa = u0.convert_to(meter*kilogram/(ampere**2*second**2))


class Medium(Basic):

    """
    This class represents an optical medium. The prime reason to implement this is
    to facilitate refraction, Fermat's principle, etc.

    Explanation
    ===========

    An optical medium is a material through which electromagnetic waves propagate.
    The permittivity and permeability of the medium define how electromagnetic
    waves propagate in it.


    Parameters
    ==========

    name: string
        The display name of the Medium.

    permittivity: Sympifyable
        Electric permittivity of the space.

    permeability: Sympifyable
        Magnetic permeability of the space.

    n: Sympifyable
        Index of refraction of the medium.


    Examples
    ========

    >>> from sympy.abc import epsilon, mu
    >>> from sympy.physics.optics import Medium
    >>> m1 = Medium('m1')
    >>> m2 = Medium('m2', epsilon, mu)
    >>> m1.intrinsic_impedance
    149896229*pi*kilogram*meter**2/(1250000*ampere**2*second**3)
    >>> m2.refractive_index
    299792458*meter*sqrt(epsilon*mu)/second


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Optical_medium

    """

    def __new__(cls, name, permittivity=None, permeability=None, n=None):
        if not isinstance(name, Str):
            name = Str(name)

        permittivity = _sympify(permittivity) if permittivity is not None else permittivity
        permeability = _sympify(permeability) if permeability is not None else permeability
        n = _sympify(n) if n is not None else n

        if n is not None:
            if permittivity is not None and permeability is None:
                permeability = n**2/(c**2*permittivity)
                return MediumPP(name, permittivity, permeability)
            elif permeability is not None and permittivity is None:
                permittivity = n**2/(c**2*permeability)
                return MediumPP(name, permittivity, permeability)
            elif permittivity is not None and permittivity is not None:
                raise ValueError("Specifying all of permittivity, permeability, and n is not allowed")
            else:
                return MediumN(name, n)
        elif permittivity is not None and permeability is not None:
            return MediumPP(name, permittivity, permeability)
        elif permittivity is None and permeability is None:
            return MediumPP(name, _e0mksa, _u0mksa)
        else:
            raise ValueError("Arguments are underspecified. Either specify n or any two of permittivity, "
                             "permeability, and n")

    @property
    def name(self):
        return self.args[0]

    @property
    def speed(self):
        """
        Returns speed of the electromagnetic wave travelling in the medium.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.speed
        299792458*meter/second
        >>> m2 = Medium('m2', n=1)
        >>> m.speed == m2.speed
        True

        """
        return c / self.n

    @property
    def refractive_index(self):
        """
        Returns refractive index of the medium.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.refractive_index
        1

        """
        return (c/self.speed)


class MediumN(Medium):

    """
    Represents an optical medium for which only the refractive index is known.
    Useful for simple ray optics.

    This class should never be instantiated directly.
    Instead it should be instantiated indirectly by instantiating Medium with
    only n specified.

    Examples
    ========
    >>> from sympy.physics.optics import Medium
    >>> m = Medium('m', n=2)
    >>> m
    MediumN(Str('m'), 2)
    """

    def __new__(cls, name, n):
        obj = super(Medium, cls).__new__(cls, name, n)
        return obj

    @property
    def n(self):
        return self.args[1]


class MediumPP(Medium):
    """
    Represents an optical medium for which the permittivity and permeability are known.

    This class should never be instantiated directly. Instead it should be
    instantiated indirectly by instantiating Medium with any two of
    permittivity, permeability, and n specified, or by not specifying any
    of permittivity, permeability, or n, in which case default values for
    permittivity and permeability will be used.

    Examples
    ========
    >>> from sympy.physics.optics import Medium
    >>> from sympy.abc import epsilon, mu
    >>> m1 = Medium('m1', permittivity=epsilon, permeability=mu)
    >>> m1
    MediumPP(Str('m1'), epsilon, mu)
    >>> m2 = Medium('m2')
    >>> m2
    MediumPP(Str('m2'), 625000*ampere**2*second**4/(22468879468420441*pi*kilogram*meter**3), pi*kilogram*meter/(2500000*ampere**2*second**2))
    """


    def __new__(cls, name, permittivity, permeability):
        obj = super(Medium, cls).__new__(cls, name, permittivity, permeability)
        return obj

    @property
    def intrinsic_impedance(self):
        """
        Returns intrinsic impedance of the medium.

        Explanation
        ===========

        The intrinsic impedance of a medium is the ratio of the
        transverse components of the electric and magnetic fields
        of the electromagnetic wave travelling in the medium.
        In a region with no electrical conductivity it simplifies
        to the square root of ratio of magnetic permeability to
        electric permittivity.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.intrinsic_impedance
        149896229*pi*kilogram*meter**2/(1250000*ampere**2*second**3)

        """
        return sqrt(self.permeability / self.permittivity)

    @property
    def permittivity(self):
        """
        Returns electric permittivity of the medium.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.permittivity
        625000*ampere**2*second**4/(22468879468420441*pi*kilogram*meter**3)

        """
        return self.args[1]

    @property
    def permeability(self):
        """
        Returns magnetic permeability of the medium.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.permeability
        pi*kilogram*meter/(2500000*ampere**2*second**2)

        """
        return self.args[2]

    @property
    def n(self):
        return c*sqrt(self.permittivity*self.permeability)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\self_outdated_check.py ===
import datetime
import functools
import hashlib
import json
import logging
import optparse
import os.path
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from pip._vendor.packaging.version import Version
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.rich.console import Group
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text

from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import get_default_environment
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.entrypoints import (
    get_best_invocation_for_this_pip,
    get_best_invocation_for_this_python,
)
from pip._internal.utils.filesystem import adjacent_tmp_file, check_path_owner, replace
from pip._internal.utils.misc import (
    ExternallyManagedEnvironment,
    check_externally_managed,
    ensure_dir,
)

_WEEK = datetime.timedelta(days=7)

logger = logging.getLogger(__name__)


def _get_statefile_name(key: str) -> str:
    key_bytes = key.encode()
    name = hashlib.sha224(key_bytes).hexdigest()
    return name


def _convert_date(isodate: str) -> datetime.datetime:
    """Convert an ISO format string to a date.

    Handles the format 2020-01-22T14:24:01Z (trailing Z)
    which is not supported by older versions of fromisoformat.
    """
    return datetime.datetime.fromisoformat(isodate.replace("Z", "+00:00"))


class SelfCheckState:
    def __init__(self, cache_dir: str) -> None:
        self._state: Dict[str, Any] = {}
        self._statefile_path = None

        # Try to load the existing state
        if cache_dir:
            self._statefile_path = os.path.join(
                cache_dir, "selfcheck", _get_statefile_name(self.key)
            )
            try:
                with open(self._statefile_path, encoding="utf-8") as statefile:
                    self._state = json.load(statefile)
            except (OSError, ValueError, KeyError):
                # Explicitly suppressing exceptions, since we don't want to
                # error out if the cache file is invalid.
                pass

    @property
    def key(self) -> str:
        return sys.prefix

    def get(self, current_time: datetime.datetime) -> Optional[str]:
        """Check if we have a not-outdated version loaded already."""
        if not self._state:
            return None

        if "last_check" not in self._state:
            return None

        if "pypi_version" not in self._state:
            return None

        # Determine if we need to refresh the state
        last_check = _convert_date(self._state["last_check"])
        time_since_last_check = current_time - last_check
        if time_since_last_check > _WEEK:
            return None

        return self._state["pypi_version"]

    def set(self, pypi_version: str, current_time: datetime.datetime) -> None:
        # If we do not have a path to cache in, don't bother saving.
        if not self._statefile_path:
            return

        # Check to make sure that we own the directory
        if not check_path_owner(os.path.dirname(self._statefile_path)):
            return

        # Now that we've ensured the directory is owned by this user, we'll go
        # ahead and make sure that all our directories are created.
        ensure_dir(os.path.dirname(self._statefile_path))

        state = {
            # Include the key so it's easy to tell which pip wrote the
            # file.
            "key": self.key,
            "last_check": current_time.isoformat(),
            "pypi_version": pypi_version,
        }

        text = json.dumps(state, sort_keys=True, separators=(",", ":"))

        with adjacent_tmp_file(self._statefile_path) as f:
            f.write(text.encode())

        try:
            # Since we have a prefix-specific state file, we can just
            # overwrite whatever is there, no need to check.
            replace(f.name, self._statefile_path)
        except OSError:
            # Best effort.
            pass


@dataclass
class UpgradePrompt:
    old: str
    new: str

    def __rich__(self) -> Group:
        if WINDOWS:
            pip_cmd = f"{get_best_invocation_for_this_python()} -m pip"
        else:
            pip_cmd = get_best_invocation_for_this_pip()

        notice = "[bold][[reset][blue]notice[reset][bold]][reset]"
        return Group(
            Text(),
            Text.from_markup(
                f"{notice} A new release of pip is available: "
                f"[red]{self.old}[reset] -> [green]{self.new}[reset]"
            ),
            Text.from_markup(
                f"{notice} To update, run: "
                f"[green]{escape(pip_cmd)} install --upgrade pip"
            ),
        )


def was_installed_by_pip(pkg: str) -> bool:
    """Checks whether pkg was installed by pip

    This is used not to display the upgrade message when pip is in fact
    installed by system package manager, such as dnf on Fedora.
    """
    dist = get_default_environment().get_distribution(pkg)
    return dist is not None and "pip" == dist.installer


def _get_current_remote_pip_version(
    session: PipSession, options: optparse.Values
) -> Optional[str]:
    # Lets use PackageFinder to see what the latest pip version is
    link_collector = LinkCollector.create(
        session,
        options=options,
        suppress_no_index=True,
    )

    # Pass allow_yanked=False so we don't suggest upgrading to a
    # yanked version.
    selection_prefs = SelectionPreferences(
        allow_yanked=False,
        allow_all_prereleases=False,  # Explicitly set to False
    )

    finder = PackageFinder.create(
        link_collector=link_collector,
        selection_prefs=selection_prefs,
    )
    best_candidate = finder.find_best_candidate("pip").best_candidate
    if best_candidate is None:
        return None

    return str(best_candidate.version)


def _self_version_check_logic(
    *,
    state: SelfCheckState,
    current_time: datetime.datetime,
    local_version: Version,
    get_remote_version: Callable[[], Optional[str]],
) -> Optional[UpgradePrompt]:
    remote_version_str = state.get(current_time)
    if remote_version_str is None:
        remote_version_str = get_remote_version()
        if remote_version_str is None:
            logger.debug("No remote pip version found")
            return None
        state.set(remote_version_str, current_time)

    remote_version = parse_version(remote_version_str)
    logger.debug("Remote version of pip: %s", remote_version)
    logger.debug("Local version of pip:  %s", local_version)

    pip_installed_by_pip = was_installed_by_pip("pip")
    logger.debug("Was pip installed by pip? %s", pip_installed_by_pip)
    if not pip_installed_by_pip:
        return None  # Only suggest upgrade if pip is installed by pip.

    local_version_is_older = (
        local_version < remote_version
        and local_version.base_version != remote_version.base_version
    )
    if local_version_is_older:
        return UpgradePrompt(old=str(local_version), new=remote_version_str)

    return None


def pip_self_version_check(session: PipSession, options: optparse.Values) -> None:
    """Check for an update for pip.

    Limit the frequency of checks to once per week. State is stored either in
    the active virtualenv or in the user's USER_CACHE_DIR keyed off the prefix
    of the pip script path.
    """
    installed_dist = get_default_environment().get_distribution("pip")
    if not installed_dist:
        return
    try:
        check_externally_managed()
    except ExternallyManagedEnvironment:
        return

    upgrade_prompt = _self_version_check_logic(
        state=SelfCheckState(cache_dir=options.cache_dir),
        current_time=datetime.datetime.now(datetime.timezone.utc),
        local_version=installed_dist.version,
        get_remote_version=functools.partial(
            _get_current_remote_pip_version, session, options
        ),
    )
    if upgrade_prompt is not None:
        logger.warning("%s", upgrade_prompt, extra={"rich": True})

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\deprecation.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: March 2, 2020
# URL: https://humanfriendly.readthedocs.io

"""
Support for deprecation warnings when importing names from old locations.

When software evolves, things tend to move around. This is usually detrimental
to backwards compatibility (in Python this primarily manifests itself as
:exc:`~exceptions.ImportError` exceptions).

While backwards compatibility is very important, it should not get in the way
of progress. It would be great to have the agility to move things around
without breaking backwards compatibility.

This is where the :mod:`humanfriendly.deprecation` module comes in: It enables
the definition of backwards compatible aliases that emit a deprecation warning
when they are accessed.

The way it works is that it wraps the original module in an :class:`DeprecationProxy`
object that defines a :func:`~DeprecationProxy.__getattr__()` special method to
override attribute access of the module.
"""

# Standard library modules.
import collections
import functools
import importlib
import inspect
import sys
import types
import warnings

# Modules included in our package.
from humanfriendly.text import format

# Registry of known aliases (used by humanfriendly.sphinx).
REGISTRY = collections.defaultdict(dict)

# Public identifiers that require documentation.
__all__ = ("DeprecationProxy", "define_aliases", "deprecated_args", "get_aliases", "is_method")


def define_aliases(module_name, **aliases):
    """
    Update a module with backwards compatible aliases.

    :param module_name: The ``__name__`` of the module (a string).
    :param aliases: Each keyword argument defines an alias. The values
                    are expected to be "dotted paths" (strings).

    The behavior of this function depends on whether the Sphinx documentation
    generator is active, because the use of :class:`DeprecationProxy` to shadow the
    real module in :data:`sys.modules` has the unintended side effect of
    breaking autodoc support for ``:data:`` members (module variables).

    To avoid breaking Sphinx the proxy object is omitted and instead the
    aliased names are injected into the original module namespace, to make sure
    that imports can be satisfied when the documentation is being rendered.

    If you run into cyclic dependencies caused by :func:`define_aliases()` when
    running Sphinx, you can try moving the call to :func:`define_aliases()` to
    the bottom of the Python module you're working on.
    """
    module = sys.modules[module_name]
    proxy = DeprecationProxy(module, aliases)
    # Populate the registry of aliases.
    for name, target in aliases.items():
        REGISTRY[module.__name__][name] = target
    # Avoid confusing Sphinx.
    if "sphinx" in sys.modules:
        for name, target in aliases.items():
            setattr(module, name, proxy.resolve(target))
    else:
        # Install a proxy object to raise DeprecationWarning.
        sys.modules[module_name] = proxy


def get_aliases(module_name):
    """
    Get the aliases defined by a module.

    :param module_name: The ``__name__`` of the module (a string).
    :returns: A dictionary with string keys and values:

              1. Each key gives the name of an alias
                 created for backwards compatibility.

              2. Each value gives the dotted path of
                 the proper location of the identifier.

              An empty dictionary is returned for modules that
              don't define any backwards compatible aliases.
    """
    return REGISTRY.get(module_name, {})


def deprecated_args(*names):
    """
    Deprecate positional arguments without dropping backwards compatibility.

    :param names:

      The positional arguments to :func:`deprecated_args()` give the names of
      the positional arguments that the to-be-decorated function should warn
      about being deprecated and translate to keyword arguments.

    :returns: A decorator function specialized to `names`.

    The :func:`deprecated_args()` decorator function was created to make it
    easy to switch from positional arguments to keyword arguments [#]_ while
    preserving backwards compatibility [#]_ and informing call sites
    about the change.

    .. [#] Increased flexibility is the main reason why I find myself switching
           from positional arguments to (optional) keyword arguments as my code
           evolves to support more use cases.

    .. [#] In my experience positional argument order implicitly becomes part
           of API compatibility whether intended or not. While this makes sense
           for functions that over time adopt more and more optional arguments,
           at a certain point it becomes an inconvenience to code maintenance.

    Here's an example of how to use the decorator::

      @deprecated_args('text')
      def report_choice(**options):
          print(options['text'])

    When the decorated function is called with positional arguments
    a deprecation warning is given::

      >>> report_choice('this will give a deprecation warning')
      DeprecationWarning: report_choice has deprecated positional arguments, please switch to keyword arguments
      this will give a deprecation warning

    But when the function is called with keyword arguments no deprecation
    warning is emitted::

      >>> report_choice(text='this will not give a deprecation warning')
      this will not give a deprecation warning
    """
    def decorator(function):
        def translate(args, kw):
            # Raise TypeError when too many positional arguments are passed to the decorated function.
            if len(args) > len(names):
                raise TypeError(
                    format(
                        "{name} expected at most {limit} arguments, got {count}",
                        name=function.__name__,
                        limit=len(names),
                        count=len(args),
                    )
                )
            # Emit a deprecation warning when positional arguments are used.
            if args:
                warnings.warn(
                    format(
                        "{name} has deprecated positional arguments, please switch to keyword arguments",
                        name=function.__name__,
                    ),
                    category=DeprecationWarning,
                    stacklevel=3,
                )
            # Translate positional arguments to keyword arguments.
            for name, value in zip(names, args):
                kw[name] = value
        if is_method(function):
            @functools.wraps(function)
            def wrapper(*args, **kw):
                """Wrapper for instance methods."""
                args = list(args)
                self = args.pop(0)
                translate(args, kw)
                return function(self, **kw)
        else:
            @functools.wraps(function)
            def wrapper(*args, **kw):
                """Wrapper for module level functions."""
                translate(args, kw)
                return function(**kw)
        return wrapper
    return decorator


def is_method(function):
    """Check if the expected usage of the given function is as an instance method."""
    try:
        # Python 3.3 and newer.
        signature = inspect.signature(function)
        return "self" in signature.parameters
    except AttributeError:
        # Python 3.2 and older.
        metadata = inspect.getargspec(function)
        return "self" in metadata.args


class DeprecationProxy(types.ModuleType):

    """Emit deprecation warnings for imports that should be updated."""

    def __init__(self, module, aliases):
        """
        Initialize an :class:`DeprecationProxy` object.

        :param module: The original module object.
        :param aliases: A dictionary of aliases.
        """
        # Initialize our superclass.
        super(DeprecationProxy, self).__init__(name=module.__name__)
        # Store initializer arguments.
        self.module = module
        self.aliases = aliases

    def __getattr__(self, name):
        """
        Override module attribute lookup.

        :param name: The name to look up (a string).
        :returns: The attribute value.
        """
        # Check if the given name is an alias.
        target = self.aliases.get(name)
        if target is not None:
            # Emit the deprecation warning.
            warnings.warn(
                format("%s.%s was moved to %s, please update your imports", self.module.__name__, name, target),
                category=DeprecationWarning,
                stacklevel=2,
            )
            # Resolve the dotted path.
            return self.resolve(target)
        # Look up the name in the original module namespace.
        value = getattr(self.module, name, None)
        if value is not None:
            return value
        # Fall back to the default behavior.
        raise AttributeError(format("module '%s' has no attribute '%s'", self.module.__name__, name))

    def resolve(self, target):
        """
        Look up the target of an alias.

        :param target: The fully qualified dotted path (a string).
        :returns: The value of the given target.
        """
        module_name, _, member = target.rpartition(".")
        module = importlib.import_module(module_name)
        return getattr(module, member)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\markup.py ===
import re
from ast import literal_eval
from operator import attrgetter
from typing import Callable, Iterable, List, Match, NamedTuple, Optional, Tuple, Union

from ._emoji_replace import _emoji_replace
from .emoji import EmojiVariant
from .errors import MarkupError
from .style import Style
from .text import Span, Text

RE_TAGS = re.compile(
    r"""((\\*)\[([a-z#/@][^[]*?)])""",
    re.VERBOSE,
)

RE_HANDLER = re.compile(r"^([\w.]*?)(\(.*?\))?$")


class Tag(NamedTuple):
    """A tag in console markup."""

    name: str
    """The tag name. e.g. 'bold'."""
    parameters: Optional[str]
    """Any additional parameters after the name."""

    def __str__(self) -> str:
        return (
            self.name if self.parameters is None else f"{self.name} {self.parameters}"
        )

    @property
    def markup(self) -> str:
        """Get the string representation of this tag."""
        return (
            f"[{self.name}]"
            if self.parameters is None
            else f"[{self.name}={self.parameters}]"
        )


_ReStringMatch = Match[str]  # regex match object
_ReSubCallable = Callable[[_ReStringMatch], str]  # Callable invoked by re.sub
_EscapeSubMethod = Callable[[_ReSubCallable, str], str]  # Sub method of a compiled re


def escape(
    markup: str,
    _escape: _EscapeSubMethod = re.compile(r"(\\*)(\[[a-z#/@][^[]*?])").sub,
) -> str:
    """Escapes text so that it won't be interpreted as markup.

    Args:
        markup (str): Content to be inserted in to markup.

    Returns:
        str: Markup with square brackets escaped.
    """

    def escape_backslashes(match: Match[str]) -> str:
        """Called by re.sub replace matches."""
        backslashes, text = match.groups()
        return f"{backslashes}{backslashes}\\{text}"

    markup = _escape(escape_backslashes, markup)
    if markup.endswith("\\") and not markup.endswith("\\\\"):
        return markup + "\\"

    return markup


def _parse(markup: str) -> Iterable[Tuple[int, Optional[str], Optional[Tag]]]:
    """Parse markup in to an iterable of tuples of (position, text, tag).

    Args:
        markup (str): A string containing console markup

    """
    position = 0
    _divmod = divmod
    _Tag = Tag
    for match in RE_TAGS.finditer(markup):
        full_text, escapes, tag_text = match.groups()
        start, end = match.span()
        if start > position:
            yield start, markup[position:start], None
        if escapes:
            backslashes, escaped = _divmod(len(escapes), 2)
            if backslashes:
                # Literal backslashes
                yield start, "\\" * backslashes, None
                start += backslashes * 2
            if escaped:
                # Escape of tag
                yield start, full_text[len(escapes) :], None
                position = end
                continue
        text, equals, parameters = tag_text.partition("=")
        yield start, None, _Tag(text, parameters if equals else None)
        position = end
    if position < len(markup):
        yield position, markup[position:], None


def render(
    markup: str,
    style: Union[str, Style] = "",
    emoji: bool = True,
    emoji_variant: Optional[EmojiVariant] = None,
) -> Text:
    """Render console markup in to a Text instance.

    Args:
        markup (str): A string containing console markup.
        style: (Union[str, Style]): The style to use.
        emoji (bool, optional): Also render emoji code. Defaults to True.
        emoji_variant (str, optional): Optional emoji variant, either "text" or "emoji". Defaults to None.


    Raises:
        MarkupError: If there is a syntax error in the markup.

    Returns:
        Text: A test instance.
    """
    emoji_replace = _emoji_replace
    if "[" not in markup:
        return Text(
            emoji_replace(markup, default_variant=emoji_variant) if emoji else markup,
            style=style,
        )
    text = Text(style=style)
    append = text.append
    normalize = Style.normalize

    style_stack: List[Tuple[int, Tag]] = []
    pop = style_stack.pop

    spans: List[Span] = []
    append_span = spans.append

    _Span = Span
    _Tag = Tag

    def pop_style(style_name: str) -> Tuple[int, Tag]:
        """Pop tag matching given style name."""
        for index, (_, tag) in enumerate(reversed(style_stack), 1):
            if tag.name == style_name:
                return pop(-index)
        raise KeyError(style_name)

    for position, plain_text, tag in _parse(markup):
        if plain_text is not None:
            # Handle open brace escapes, where the brace is not part of a tag.
            plain_text = plain_text.replace("\\[", "[")
            append(emoji_replace(plain_text) if emoji else plain_text)
        elif tag is not None:
            if tag.name.startswith("/"):  # Closing tag
                style_name = tag.name[1:].strip()

                if style_name:  # explicit close
                    style_name = normalize(style_name)
                    try:
                        start, open_tag = pop_style(style_name)
                    except KeyError:
                        raise MarkupError(
                            f"closing tag '{tag.markup}' at position {position} doesn't match any open tag"
                        ) from None
                else:  # implicit close
                    try:
                        start, open_tag = pop()
                    except IndexError:
                        raise MarkupError(
                            f"closing tag '[/]' at position {position} has nothing to close"
                        ) from None

                if open_tag.name.startswith("@"):
                    if open_tag.parameters:
                        handler_name = ""
                        parameters = open_tag.parameters.strip()
                        handler_match = RE_HANDLER.match(parameters)
                        if handler_match is not None:
                            handler_name, match_parameters = handler_match.groups()
                            parameters = (
                                "()" if match_parameters is None else match_parameters
                            )

                        try:
                            meta_params = literal_eval(parameters)
                        except SyntaxError as error:
                            raise MarkupError(
                                f"error parsing {parameters!r} in {open_tag.parameters!r}; {error.msg}"
                            )
                        except Exception as error:
                            raise MarkupError(
                                f"error parsing {open_tag.parameters!r}; {error}"
                            ) from None

                        if handler_name:
                            meta_params = (
                                handler_name,
                                meta_params
                                if isinstance(meta_params, tuple)
                                else (meta_params,),
                            )

                    else:
                        meta_params = ()

                    append_span(
                        _Span(
                            start, len(text), Style(meta={open_tag.name: meta_params})
                        )
                    )
                else:
                    append_span(_Span(start, len(text), str(open_tag)))

            else:  # Opening tag
                normalized_tag = _Tag(normalize(tag.name), tag.parameters)
                style_stack.append((len(text), normalized_tag))

    text_length = len(text)
    while style_stack:
        start, tag = style_stack.pop()
        style = str(tag)
        if style:
            append_span(_Span(start, text_length, style))

    text.spans = sorted(spans[::-1], key=attrgetter("start"))
    return text


if __name__ == "__main__":  # pragma: no cover
    MARKUP = [
        "[red]Hello World[/red]",
        "[magenta]Hello [b]World[/b]",
        "[bold]Bold[italic] bold and italic [/bold]italic[/italic]",
        "Click [link=https://www.willmcgugan.com]here[/link] to visit my Blog",
        ":warning-emoji: [bold red blink] DANGER![/]",
    ]

    from pip._vendor.rich import print
    from pip._vendor.rich.table import Table

    grid = Table("Markup", "Result", padding=(0, 1))

    for markup in MARKUP:
        grid.add_row(Text(markup), markup)

    print(grid)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\plot_axes.py ===
import pyglet.gl as pgl
from pyglet import font

from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
        get_direction_vectors, strided_range, vec_mag, vec_sub
from sympy.utilities.iterables import is_sequence


class PlotAxes(PlotObject):

    def __init__(self, *args,
            style='', none=None, frame=None, box=None, ordinate=None,
            stride=0.25,
            visible='', overlay='', colored='', label_axes='', label_ticks='',
            tick_length=0.1,
            font_face='Arial', font_size=28,
            **kwargs):
        # initialize style parameter
        style = style.lower()

        # allow alias kwargs to override style kwarg
        if none is not None:
            style = 'none'
        if frame is not None:
            style = 'frame'
        if box is not None:
            style = 'box'
        if ordinate is not None:
            style = 'ordinate'

        if style in ['', 'ordinate']:
            self._render_object = PlotAxesOrdinate(self)
        elif style in ['frame', 'box']:
            self._render_object = PlotAxesFrame(self)
        elif style in ['none']:
            self._render_object = None
        else:
            raise ValueError(("Unrecognized axes style %s.") % (style))

        # initialize stride parameter
        try:
            stride = eval(stride)
        except TypeError:
            pass
        if is_sequence(stride):
            if len(stride) != 3:
                raise ValueError("length should be equal to 3")
            self._stride = stride
        else:
            self._stride = [stride, stride, stride]
        self._tick_length = float(tick_length)

        # setup bounding box and ticks
        self._origin = [0, 0, 0]
        self.reset_bounding_box()

        def flexible_boolean(input, default):
            if input in [True, False]:
                return input
            if input in ('f', 'F', 'false', 'False'):
                return False
            if input in ('t', 'T', 'true', 'True'):
                return True
            return default

        # initialize remaining parameters
        self.visible = flexible_boolean(kwargs, True)
        self._overlay = flexible_boolean(overlay, True)
        self._colored = flexible_boolean(colored, False)
        self._label_axes = flexible_boolean(label_axes, False)
        self._label_ticks = flexible_boolean(label_ticks, True)

        # setup label font
        self.font_face = font_face
        self.font_size = font_size

        # this is also used to reinit the
        # font on window close/reopen
        self.reset_resources()

    def reset_resources(self):
        self.label_font = None

    def reset_bounding_box(self):
        self._bounding_box = [[None, None], [None, None], [None, None]]
        self._axis_ticks = [[], [], []]

    def draw(self):
        if self._render_object:
            pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT | pgl.GL_DEPTH_BUFFER_BIT)
            if self._overlay:
                pgl.glDisable(pgl.GL_DEPTH_TEST)
            self._render_object.draw()
            pgl.glPopAttrib()

    def adjust_bounds(self, child_bounds):
        b = self._bounding_box
        c = child_bounds
        for i in range(3):
            if abs(c[i][0]) is S.Infinity or abs(c[i][1]) is S.Infinity:
                continue
            b[i][0] = c[i][0] if b[i][0] is None else min([b[i][0], c[i][0]])
            b[i][1] = c[i][1] if b[i][1] is None else max([b[i][1], c[i][1]])
            self._bounding_box = b
            self._recalculate_axis_ticks(i)

    def _recalculate_axis_ticks(self, axis):
        b = self._bounding_box
        if b[axis][0] is None or b[axis][1] is None:
            self._axis_ticks[axis] = []
        else:
            self._axis_ticks[axis] = strided_range(b[axis][0], b[axis][1],
                                                   self._stride[axis])

    def toggle_visible(self):
        self.visible = not self.visible

    def toggle_colors(self):
        self._colored = not self._colored


class PlotAxesBase(PlotObject):

    def __init__(self, parent_axes):
        self._p = parent_axes

    def draw(self):
        color = [([0.2, 0.1, 0.3], [0.2, 0.1, 0.3], [0.2, 0.1, 0.3]),
                 ([0.9, 0.3, 0.5], [0.5, 1.0, 0.5], [0.3, 0.3, 0.9])][self._p._colored]
        self.draw_background(color)
        self.draw_axis(2, color[2])
        self.draw_axis(1, color[1])
        self.draw_axis(0, color[0])

    def draw_background(self, color):
        pass  # optional

    def draw_axis(self, axis, color):
        raise NotImplementedError()

    def draw_text(self, text, position, color, scale=1.0):
        if len(color) == 3:
            color = (color[0], color[1], color[2], 1.0)

        if self._p.label_font is None:
            self._p.label_font = font.load(self._p.font_face,
                                           self._p.font_size,
                                           bold=True, italic=False)

        label = font.Text(self._p.label_font, text,
                          color=color,
                          valign=font.Text.BASELINE,
                          halign=font.Text.CENTER)

        pgl.glPushMatrix()
        pgl.glTranslatef(*position)
        billboard_matrix()
        scale_factor = 0.005 * scale
        pgl.glScalef(scale_factor, scale_factor, scale_factor)
        pgl.glColor4f(0, 0, 0, 0)
        label.draw()
        pgl.glPopMatrix()

    def draw_line(self, v, color):
        o = self._p._origin
        pgl.glBegin(pgl.GL_LINES)
        pgl.glColor3f(*color)
        pgl.glVertex3f(v[0][0] + o[0], v[0][1] + o[1], v[0][2] + o[2])
        pgl.glVertex3f(v[1][0] + o[0], v[1][1] + o[1], v[1][2] + o[2])
        pgl.glEnd()


class PlotAxesOrdinate(PlotAxesBase):

    def __init__(self, parent_axes):
        super().__init__(parent_axes)

    def draw_axis(self, axis, color):
        ticks = self._p._axis_ticks[axis]
        radius = self._p._tick_length / 2.0
        if len(ticks) < 2:
            return

        # calculate the vector for this axis
        axis_lines = [[0, 0, 0], [0, 0, 0]]
        axis_lines[0][axis], axis_lines[1][axis] = ticks[0], ticks[-1]
        axis_vector = vec_sub(axis_lines[1], axis_lines[0])

        # calculate angle to the z direction vector
        pos_z = get_direction_vectors()[2]
        d = abs(dot_product(axis_vector, pos_z))
        d = d / vec_mag(axis_vector)

        # don't draw labels if we're looking down the axis
        labels_visible = abs(d - 1.0) > 0.02

        # draw the ticks and labels
        for tick in ticks:
            self.draw_tick_line(axis, color, radius, tick, labels_visible)

        # draw the axis line and labels
        self.draw_axis_line(axis, color, ticks[0], ticks[-1], labels_visible)

    def draw_axis_line(self, axis, color, a_min, a_max, labels_visible):
        axis_line = [[0, 0, 0], [0, 0, 0]]
        axis_line[0][axis], axis_line[1][axis] = a_min, a_max
        self.draw_line(axis_line, color)
        if labels_visible:
            self.draw_axis_line_labels(axis, color, axis_line)

    def draw_axis_line_labels(self, axis, color, axis_line):
        if not self._p._label_axes:
            return
        axis_labels = [axis_line[0][::], axis_line[1][::]]
        axis_labels[0][axis] -= 0.3
        axis_labels[1][axis] += 0.3
        a_str = ['X', 'Y', 'Z'][axis]
        self.draw_text("-" + a_str, axis_labels[0], color)
        self.draw_text("+" + a_str, axis_labels[1], color)

    def draw_tick_line(self, axis, color, radius, tick, labels_visible):
        tick_axis = {0: 1, 1: 0, 2: 1}[axis]
        tick_line = [[0, 0, 0], [0, 0, 0]]
        tick_line[0][axis] = tick_line[1][axis] = tick
        tick_line[0][tick_axis], tick_line[1][tick_axis] = -radius, radius
        self.draw_line(tick_line, color)
        if labels_visible:
            self.draw_tick_line_label(axis, color, radius, tick)

    def draw_tick_line_label(self, axis, color, radius, tick):
        if not self._p._label_axes:
            return
        tick_label_vector = [0, 0, 0]
        tick_label_vector[axis] = tick
        tick_label_vector[{0: 1, 1: 0, 2: 1}[axis]] = [-1, 1, 1][
            axis] * radius * 3.5
        self.draw_text(str(tick), tick_label_vector, color, scale=0.5)


class PlotAxesFrame(PlotAxesBase):

    def __init__(self, parent_axes):
        super().__init__(parent_axes)

    def draw_background(self, color):
        pass

    def draw_axis(self, axis, color):
        raise NotImplementedError()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\lambdarepr.py ===
from .pycode import (
    PythonCodePrinter,
    MpmathPrinter,
)
from .numpy import NumPyPrinter  # NumPyPrinter is imported for backward compatibility
from sympy.core.sorting import default_sort_key


__all__ = [
    'PythonCodePrinter',
    'MpmathPrinter',  # MpmathPrinter is published for backward compatibility
    'NumPyPrinter',
    'LambdaPrinter',
    'NumPyPrinter',
    'IntervalPrinter',
    'lambdarepr',
]


class LambdaPrinter(PythonCodePrinter):
    """
    This printer converts expressions into strings that can be used by
    lambdify.
    """
    printmethod = "_lambdacode"


    def _print_And(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' and ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Or(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' or ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Not(self, expr):
        result = ['(', 'not (', self._print(expr.args[0]), '))']
        return ''.join(result)

    def _print_BooleanTrue(self, expr):
        return "True"

    def _print_BooleanFalse(self, expr):
        return "False"

    def _print_ITE(self, expr):
        result = [
            '((', self._print(expr.args[1]),
            ') if (', self._print(expr.args[0]),
            ') else (', self._print(expr.args[2]), '))'
        ]
        return ''.join(result)

    def _print_NumberSymbol(self, expr):
        return str(expr)

    def _print_Pow(self, expr, **kwargs):
        # XXX Temporary workaround. Should Python math printer be
        # isolated from PythonCodePrinter?
        return super(PythonCodePrinter, self)._print_Pow(expr, **kwargs)


# numexpr works by altering the string passed to numexpr.evaluate
# rather than by populating a namespace.  Thus a special printer...
class NumExprPrinter(LambdaPrinter):
    # key, value pairs correspond to SymPy name and numexpr name
    # functions not appearing in this dict will raise a TypeError
    printmethod = "_numexprcode"

    _numexpr_functions = {
        'sin' : 'sin',
        'cos' : 'cos',
        'tan' : 'tan',
        'asin': 'arcsin',
        'acos': 'arccos',
        'atan': 'arctan',
        'atan2' : 'arctan2',
        'sinh' : 'sinh',
        'cosh' : 'cosh',
        'tanh' : 'tanh',
        'asinh': 'arcsinh',
        'acosh': 'arccosh',
        'atanh': 'arctanh',
        'ln' : 'log',
        'log': 'log',
        'exp': 'exp',
        'sqrt' : 'sqrt',
        'Abs' : 'abs',
        'conjugate' : 'conj',
        'im' : 'imag',
        're' : 'real',
        'where' : 'where',
        'complex' : 'complex',
        'contains' : 'contains',
    }

    module = 'numexpr'

    def _print_ImaginaryUnit(self, expr):
        return '1j'

    def _print_seq(self, seq, delimiter=', '):
        # simplified _print_seq taken from pretty.py
        s = [self._print(item) for item in seq]
        if s:
            return delimiter.join(s)
        else:
            return ""

    def _print_Function(self, e):
        func_name = e.func.__name__

        nstr = self._numexpr_functions.get(func_name, None)
        if nstr is None:
            # check for implemented_function
            if hasattr(e, '_imp_'):
                return "(%s)" % self._print(e._imp_(*e.args))
            else:
                raise TypeError("numexpr does not support function '%s'" %
                                func_name)
        return "%s(%s)" % (nstr, self._print_seq(e.args))

    def _print_Piecewise(self, expr):
        "Piecewise function printer"
        exprs = [self._print(arg.expr) for arg in expr.args]
        conds = [self._print(arg.cond) for arg in expr.args]
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in expr.args.
        # If this is not the case, it may be triggered prematurely.
        ans = []
        parenthesis_count = 0
        is_last_cond_True = False
        for cond, expr in zip(conds, exprs):
            if cond == 'True':
                ans.append(expr)
                is_last_cond_True = True
                break
            else:
                ans.append('where(%s, %s, ' % (cond, expr))
                parenthesis_count += 1
        if not is_last_cond_True:
            # See https://github.com/pydata/numexpr/issues/298
            #
            # simplest way to put a nan but raises
            # 'RuntimeWarning: invalid value encountered in log'
            #
            # There are other ways to do this such as
            #
            #   >>> import numexpr as ne
            #   >>> nan = float('nan')
            #   >>> ne.evaluate('where(x < 0, -1, nan)', {'x': [-1, 2, 3], 'nan':nan})
            #   array([-1., nan, nan])
            #
            # That needs to be handled in the lambdified function though rather
            # than here in the printer.
            ans.append('log(-1)')
        return ''.join(ans) + ')' * parenthesis_count

    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def blacklisted(self, expr):
        raise TypeError("numexpr cannot be used with %s" %
                        expr.__class__.__name__)

    # blacklist all Matrix printing
    _print_SparseRepMatrix = \
    _print_MutableSparseMatrix = \
    _print_ImmutableSparseMatrix = \
    _print_Matrix = \
    _print_DenseMatrix = \
    _print_MutableDenseMatrix = \
    _print_ImmutableMatrix = \
    _print_ImmutableDenseMatrix = \
    blacklisted
    # blacklist some Python expressions
    _print_list = \
    _print_tuple = \
    _print_Tuple = \
    _print_dict = \
    _print_Dict = \
    blacklisted

    def _print_NumExprEvaluate(self, expr):
        evaluate = self._module_format(self.module +".evaluate")
        return "%s('%s', truediv=True)" % (evaluate, self._print(expr.expr))

    def doprint(self, expr):
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        if not isinstance(expr, CodegenAST):
            expr = NumExprEvaluate(expr)
        return super().doprint(expr)

    def _print_Return(self, expr):
        from sympy.codegen.pynodes import NumExprEvaluate
        r, = expr.args
        if not isinstance(r, NumExprEvaluate):
            expr = expr.func(NumExprEvaluate(r))
        return super()._print_Return(expr)

    def _print_Assignment(self, expr):
        from sympy.codegen.pynodes import NumExprEvaluate
        lhs, rhs, *args = expr.args
        if not isinstance(rhs, NumExprEvaluate):
            expr = expr.func(lhs, NumExprEvaluate(rhs), *args)
        return super()._print_Assignment(expr)

    def _print_CodeBlock(self, expr):
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        args = [ arg if isinstance(arg, CodegenAST) else NumExprEvaluate(arg) for arg in expr.args ]
        return super()._print_CodeBlock(self, expr.func(*args))


class IntervalPrinter(MpmathPrinter, LambdaPrinter):
    """Use ``lambda`` printer but print numbers as ``mpi`` intervals. """

    def _print_Integer(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Integer(expr)

    def _print_Rational(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Half(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Pow(self, expr):
        return super(MpmathPrinter, self)._print_Pow(expr, rational=True)


for k in NumExprPrinter._numexpr_functions:
    setattr(NumExprPrinter, '_print_%s' % k, NumExprPrinter._print_Function)

def lambdarepr(expr, **settings):
    """
    Returns a string usable for lambdifying.
    """
    return LambdaPrinter(settings).doprint(expr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_highlevel_open_tcp_listeners.py ===
from __future__ import annotations

import errno
import sys
from typing import TYPE_CHECKING

import trio
from trio import TaskStatus

from . import socket as tsocket

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


# Default backlog size:
#
# Having the backlog too low can cause practical problems (a perfectly healthy
# service that starts failing to accept connections if they arrive in a
# burst).
#
# Having it too high doesn't really cause any problems. Like any buffer, you
# want backlog queue to be zero usually, and it won't save you if you're
# getting connection attempts faster than you can call accept() on an ongoing
# basis. But unlike other buffers, this one doesn't really provide any
# backpressure. If a connection gets stuck waiting in the backlog queue, then
# from the peer's point of view the connection succeeded but then their
# send/recv will stall until we get to it, possibly for a long time. OTOH if
# there isn't room in the backlog queue, then their connect stalls, possibly
# for a long time, which is pretty much the same thing.
#
# A large backlog can also use a bit more kernel memory, but this seems fairly
# negligible these days.
#
# So this suggests we should make the backlog as large as possible. This also
# matches what Golang does. However, they do it in a weird way, where they
# have a bunch of code to sniff out the configured upper limit for backlog on
# different operating systems. But on every system, passing in a too-large
# backlog just causes it to be silently truncated to the configured maximum,
# so this is unnecessary -- we can just pass in "infinity" and get the maximum
# that way. (Verified on Windows, Linux, macOS using
# https://github.com/python-trio/trio/wiki/notes-to-self#measure-listen-backlogpy
def _compute_backlog(backlog: int | None) -> int:
    # Many systems (Linux, BSDs, ...) store the backlog in a uint16 and are
    # missing overflow protection, so we apply our own overflow protection.
    # https://github.com/golang/go/issues/5030
    if not isinstance(backlog, int) and backlog is not None:
        raise TypeError(f"backlog must be an int or None, not {backlog!r}")
    if backlog is None:
        return 0xFFFF
    return min(backlog, 0xFFFF)


async def open_tcp_listeners(
    port: int,
    *,
    host: str | bytes | None = None,
    backlog: int | None = None,
) -> list[trio.SocketListener]:
    """Create :class:`SocketListener` objects to listen for TCP connections.

    Args:

      port (int): The port to listen on.

          If you use 0 as your port, then the kernel will automatically pick
          an arbitrary open port. But be careful: if you use this feature when
          binding to multiple IP addresses, then each IP address will get its
          own random port, and the returned listeners will probably be
          listening on different ports. In particular, this will happen if you
          use ``host=None`` – which is the default – because in this case
          :func:`open_tcp_listeners` will bind to both the IPv4 wildcard
          address (``0.0.0.0``) and also the IPv6 wildcard address (``::``).

      host (str, bytes, or None): The local interface to bind to. This is
          passed to :func:`~socket.getaddrinfo` with the ``AI_PASSIVE`` flag
          set.

          If you want to bind to the wildcard address on both IPv4 and IPv6,
          in order to accept connections on all available interfaces, then
          pass ``None``. This is the default.

          If you have a specific interface you want to bind to, pass its IP
          address or hostname here. If a hostname resolves to multiple IP
          addresses, this function will open one listener on each of them.

          If you want to use only IPv4, or only IPv6, but want to accept on
          all interfaces, pass the family-specific wildcard address:
          ``"0.0.0.0"`` for IPv4-only and ``"::"`` for IPv6-only.

      backlog (int or None): The listen backlog to use. If you leave this as
          ``None`` then Trio will pick a good default. (Currently: whatever
          your system has configured as the maximum backlog.)

    Returns:
      list of :class:`SocketListener`

    Raises:
      :class:`TypeError` if invalid arguments.

    """
    # getaddrinfo sometimes allows port=None, sometimes not (depending on
    # whether host=None). And on some systems it treats "" as 0, others it
    # doesn't:
    #   http://klickverbot.at/blog/2012/01/getaddrinfo-edge-case-behavior-on-windows-linux-and-osx/
    if not isinstance(port, int):
        raise TypeError(f"port must be an int not {port!r}")

    computed_backlog = _compute_backlog(backlog)

    addresses = await tsocket.getaddrinfo(
        host,
        port,
        type=tsocket.SOCK_STREAM,
        flags=tsocket.AI_PASSIVE,
    )

    listeners = []
    unsupported_address_families = []
    try:
        for family, type_, proto, _, sockaddr in addresses:
            try:
                sock = tsocket.socket(family, type_, proto)
            except OSError as ex:
                if ex.errno == errno.EAFNOSUPPORT:
                    # If a system only supports IPv4, or only IPv6, it
                    # is still likely that getaddrinfo will return
                    # both an IPv4 and an IPv6 address. As long as at
                    # least one of the returned addresses can be
                    # turned into a socket, we won't complain about a
                    # failure to create the other.
                    unsupported_address_families.append(ex)
                    continue
                else:
                    raise
            try:
                # See https://github.com/python-trio/trio/issues/39
                if sys.platform != "win32":
                    sock.setsockopt(tsocket.SOL_SOCKET, tsocket.SO_REUSEADDR, 1)

                if family == tsocket.AF_INET6:
                    sock.setsockopt(tsocket.IPPROTO_IPV6, tsocket.IPV6_V6ONLY, 1)

                await sock.bind(sockaddr)
                sock.listen(computed_backlog)

                listeners.append(trio.SocketListener(sock))
            except:
                sock.close()
                raise
    except:
        for listener in listeners:
            listener.socket.close()
        raise

    if unsupported_address_families and not listeners:
        msg = (
            "This system doesn't support any of the kinds of "
            "socket that that address could use"
        )
        raise OSError(errno.EAFNOSUPPORT, msg) from ExceptionGroup(
            msg,
            unsupported_address_families,
        )

    return listeners


async def serve_tcp(
    handler: Callable[[trio.SocketStream], Awaitable[object]],
    port: int,
    *,
    host: str | bytes | None = None,
    backlog: int | None = None,
    handler_nursery: trio.Nursery | None = None,
    task_status: TaskStatus[list[trio.SocketListener]] = trio.TASK_STATUS_IGNORED,
) -> None:
    """Listen for incoming TCP connections, and for each one start a task
    running ``handler(stream)``.

    This is a thin convenience wrapper around :func:`open_tcp_listeners` and
    :func:`serve_listeners` – see them for full details.

    .. warning::

       If ``handler`` raises an exception, then this function doesn't do
       anything special to catch it – so by default the exception will
       propagate out and crash your server. If you don't want this, then catch
       exceptions inside your ``handler``, or use a ``handler_nursery`` object
       that responds to exceptions in some other way.

    When used with ``nursery.start`` you get back the newly opened listeners.
    So, for example, if you want to start a server in your test suite and then
    connect to it to check that it's working properly, you can use something
    like::

        from trio import SocketListener, SocketStream
        from trio.testing import open_stream_to_socket_listener

        async with trio.open_nursery() as nursery:
            listeners: list[SocketListener] = await nursery.start(serve_tcp, handler, 0)
            client_stream: SocketStream = await open_stream_to_socket_listener(listeners[0])

            # Then send and receive data on 'client_stream', for example:
            await client_stream.send_all(b"GET / HTTP/1.0\\r\\n\\r\\n")

    This avoids several common pitfalls:

    1. It lets the kernel pick a random open port, so your test suite doesn't
       depend on any particular port being open.

    2. It waits for the server to be accepting connections on that port before
       ``start`` returns, so there's no race condition where the incoming
       connection arrives before the server is ready.

    3. It uses the Listener object to find out which port was picked, so it
       can connect to the right place.

    Args:
      handler: The handler to start for each incoming connection. Passed to
          :func:`serve_listeners`.

      port: The port to listen on. Use 0 to let the kernel pick an open port.
          Passed to :func:`open_tcp_listeners`.

      host (str, bytes, or None): The host interface to listen on; use
          ``None`` to bind to the wildcard address. Passed to
          :func:`open_tcp_listeners`.

      backlog: The listen backlog, or None to have a good default picked.
          Passed to :func:`open_tcp_listeners`.

      handler_nursery: The nursery to start handlers in, or None to use an
          internal nursery. Passed to :func:`serve_listeners`.

      task_status: This function can be used with ``nursery.start``.

    Returns:
      This function only returns when cancelled.

    """
    listeners = await trio.open_tcp_listeners(port, host=host, backlog=backlog)
    await trio.serve_listeners(
        handler,
        listeners,
        handler_nursery=handler_nursery,
        task_status=task_status,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\h11\_readers.py ===
# Code to read HTTP data
#
# Strategy: each reader is a callable which takes a ReceiveBuffer object, and
# either:
# 1) consumes some of it and returns an Event
# 2) raises a LocalProtocolError (for consistency -- e.g. we call validate()
#    and it might raise a LocalProtocolError, so simpler just to always use
#    this)
# 3) returns None, meaning "I need more data"
#
# If they have a .read_eof attribute, then this will be called if an EOF is
# received -- but this is optional. Either way, the actual ConnectionClosed
# event will be generated afterwards.
#
# READERS is a dict describing how to pick a reader. It maps states to either:
# - a reader
# - or, for body readers, a dict of per-framing reader factories

import re
from typing import Any, Callable, Dict, Iterable, NoReturn, Optional, Tuple, Type, Union

from ._abnf import chunk_header, header_field, request_line, status_line
from ._events import Data, EndOfMessage, InformationalResponse, Request, Response
from ._receivebuffer import ReceiveBuffer
from ._state import (
    CLIENT,
    CLOSED,
    DONE,
    IDLE,
    MUST_CLOSE,
    SEND_BODY,
    SEND_RESPONSE,
    SERVER,
)
from ._util import LocalProtocolError, RemoteProtocolError, Sentinel, validate

__all__ = ["READERS"]

header_field_re = re.compile(header_field.encode("ascii"))
obs_fold_re = re.compile(rb"[ \t]+")


def _obsolete_line_fold(lines: Iterable[bytes]) -> Iterable[bytes]:
    it = iter(lines)
    last: Optional[bytes] = None
    for line in it:
        match = obs_fold_re.match(line)
        if match:
            if last is None:
                raise LocalProtocolError("continuation line at start of headers")
            if not isinstance(last, bytearray):
                # Cast to a mutable type, avoiding copy on append to ensure O(n) time
                last = bytearray(last)
            last += b" "
            last += line[match.end() :]
        else:
            if last is not None:
                yield last
            last = line
    if last is not None:
        yield last


def _decode_header_lines(
    lines: Iterable[bytes],
) -> Iterable[Tuple[bytes, bytes]]:
    for line in _obsolete_line_fold(lines):
        matches = validate(header_field_re, line, "illegal header line: {!r}", line)
        yield (matches["field_name"], matches["field_value"])


request_line_re = re.compile(request_line.encode("ascii"))


def maybe_read_from_IDLE_client(buf: ReceiveBuffer) -> Optional[Request]:
    lines = buf.maybe_extract_lines()
    if lines is None:
        if buf.is_next_line_obviously_invalid_request_line():
            raise LocalProtocolError("illegal request line")
        return None
    if not lines:
        raise LocalProtocolError("no request line received")
    matches = validate(
        request_line_re, lines[0], "illegal request line: {!r}", lines[0]
    )
    return Request(
        headers=list(_decode_header_lines(lines[1:])), _parsed=True, **matches
    )


status_line_re = re.compile(status_line.encode("ascii"))


def maybe_read_from_SEND_RESPONSE_server(
    buf: ReceiveBuffer,
) -> Union[InformationalResponse, Response, None]:
    lines = buf.maybe_extract_lines()
    if lines is None:
        if buf.is_next_line_obviously_invalid_request_line():
            raise LocalProtocolError("illegal request line")
        return None
    if not lines:
        raise LocalProtocolError("no response line received")
    matches = validate(status_line_re, lines[0], "illegal status line: {!r}", lines[0])
    http_version = (
        b"1.1" if matches["http_version"] is None else matches["http_version"]
    )
    reason = b"" if matches["reason"] is None else matches["reason"]
    status_code = int(matches["status_code"])
    class_: Union[Type[InformationalResponse], Type[Response]] = (
        InformationalResponse if status_code < 200 else Response
    )
    return class_(
        headers=list(_decode_header_lines(lines[1:])),
        _parsed=True,
        status_code=status_code,
        reason=reason,
        http_version=http_version,
    )


class ContentLengthReader:
    def __init__(self, length: int) -> None:
        self._length = length
        self._remaining = length

    def __call__(self, buf: ReceiveBuffer) -> Union[Data, EndOfMessage, None]:
        if self._remaining == 0:
            return EndOfMessage()
        data = buf.maybe_extract_at_most(self._remaining)
        if data is None:
            return None
        self._remaining -= len(data)
        return Data(data=data)

    def read_eof(self) -> NoReturn:
        raise RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(received {} bytes, expected {})".format(
                self._length - self._remaining, self._length
            )
        )


chunk_header_re = re.compile(chunk_header.encode("ascii"))


class ChunkedReader:
    def __init__(self) -> None:
        self._bytes_in_chunk = 0
        # After reading a chunk, we have to throw away the trailing \r\n.
        # This tracks the bytes that we need to match and throw away.
        self._bytes_to_discard = b""
        self._reading_trailer = False

    def __call__(self, buf: ReceiveBuffer) -> Union[Data, EndOfMessage, None]:
        if self._reading_trailer:
            lines = buf.maybe_extract_lines()
            if lines is None:
                return None
            return EndOfMessage(headers=list(_decode_header_lines(lines)))
        if self._bytes_to_discard:
            data = buf.maybe_extract_at_most(len(self._bytes_to_discard))
            if data is None:
                return None
            if data != self._bytes_to_discard[: len(data)]:
                raise LocalProtocolError(
                    f"malformed chunk footer: {data!r} (expected {self._bytes_to_discard!r})"
                )
            self._bytes_to_discard = self._bytes_to_discard[len(data) :]
            if self._bytes_to_discard:
                return None
            # else, fall through and read some more
        assert self._bytes_to_discard == b""
        if self._bytes_in_chunk == 0:
            # We need to refill our chunk count
            chunk_header = buf.maybe_extract_next_line()
            if chunk_header is None:
                return None
            matches = validate(
                chunk_header_re,
                chunk_header,
                "illegal chunk header: {!r}",
                chunk_header,
            )
            # XX FIXME: we discard chunk extensions. Does anyone care?
            self._bytes_in_chunk = int(matches["chunk_size"], base=16)
            if self._bytes_in_chunk == 0:
                self._reading_trailer = True
                return self(buf)
            chunk_start = True
        else:
            chunk_start = False
        assert self._bytes_in_chunk > 0
        data = buf.maybe_extract_at_most(self._bytes_in_chunk)
        if data is None:
            return None
        self._bytes_in_chunk -= len(data)
        if self._bytes_in_chunk == 0:
            self._bytes_to_discard = b"\r\n"
            chunk_end = True
        else:
            chunk_end = False
        return Data(data=data, chunk_start=chunk_start, chunk_end=chunk_end)

    def read_eof(self) -> NoReturn:
        raise RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        )


class Http10Reader:
    def __call__(self, buf: ReceiveBuffer) -> Optional[Data]:
        data = buf.maybe_extract_at_most(999999999)
        if data is None:
            return None
        return Data(data=data)

    def read_eof(self) -> EndOfMessage:
        return EndOfMessage()


def expect_nothing(buf: ReceiveBuffer) -> None:
    if buf:
        raise LocalProtocolError("Got data when expecting EOF")
    return None


ReadersType = Dict[
    Union[Type[Sentinel], Tuple[Type[Sentinel], Type[Sentinel]]],
    Union[Callable[..., Any], Dict[str, Callable[..., Any]]],
]

READERS: ReadersType = {
    (CLIENT, IDLE): maybe_read_from_IDLE_client,
    (SERVER, IDLE): maybe_read_from_SEND_RESPONSE_server,
    (SERVER, SEND_RESPONSE): maybe_read_from_SEND_RESPONSE_server,
    (CLIENT, DONE): expect_nothing,
    (CLIENT, MUST_CLOSE): expect_nothing,
    (CLIENT, CLOSED): expect_nothing,
    (SERVER, DONE): expect_nothing,
    (SERVER, MUST_CLOSE): expect_nothing,
    (SERVER, CLOSED): expect_nothing,
    SEND_BODY: {
        "chunked": ChunkedReader,
        "content-length": ContentLengthReader,
        "http/1.0": Http10Reader,
    },
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\testing\_private\extbuild.py ===
"""
Build a c-extension module on-the-fly in tests.
See build_and_import_extensions for usage hints

"""

import os
import pathlib
import subprocess
import sys
import sysconfig
import textwrap

__all__ = ['build_and_import_extension', 'compile_extension_module']


def build_and_import_extension(
        modname, functions, *, prologue="", build_dir=None,
        include_dirs=None, more_init=""):
    """
    Build and imports a c-extension module `modname` from a list of function
    fragments `functions`.


    Parameters
    ----------
    functions : list of fragments
        Each fragment is a sequence of func_name, calling convention, snippet.
    prologue : string
        Code to precede the rest, usually extra ``#include`` or ``#define``
        macros.
    build_dir : pathlib.Path
        Where to build the module, usually a temporary directory
    include_dirs : list
        Extra directories to find include files when compiling
    more_init : string
        Code to appear in the module PyMODINIT_FUNC

    Returns
    -------
    out: module
        The module will have been loaded and is ready for use

    Examples
    --------
    >>> functions = [("test_bytes", "METH_O", \"\"\"
        if ( !PyBytesCheck(args)) {
            Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    \"\"\")]
    >>> mod = build_and_import_extension("testme", functions)
    >>> assert not mod.test_bytes('abc')
    >>> assert mod.test_bytes(b'abc')
    """
    if include_dirs is None:
        include_dirs = []
    body = prologue + _make_methods(functions, modname)
    init = """
    PyObject *mod = PyModule_Create(&moduledef);
    #ifdef Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(mod, Py_MOD_GIL_NOT_USED);
    #endif
           """
    if not build_dir:
        build_dir = pathlib.Path('.')
    if more_init:
        init += """#define INITERROR return NULL
                """
        init += more_init
    init += "\nreturn mod;"
    source_string = _make_source(modname, init, body)
    mod_so = compile_extension_module(
        modname, build_dir, include_dirs, source_string)
    import importlib.util
    spec = importlib.util.spec_from_file_location(modname, mod_so)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


def compile_extension_module(
        name, builddir, include_dirs,
        source_string, libraries=None, library_dirs=None):
    """
    Build an extension module and return the filename of the resulting
    native code file.

    Parameters
    ----------
    name : string
        name of the module, possibly including dots if it is a module inside a
        package.
    builddir : pathlib.Path
        Where to build the module, usually a temporary directory
    include_dirs : list
        Extra directories to find include files when compiling
    libraries : list
        Libraries to link into the extension module
    library_dirs: list
        Where to find the libraries, ``-L`` passed to the linker
    """
    modname = name.split('.')[-1]
    dirname = builddir / name
    dirname.mkdir(exist_ok=True)
    cfile = _convert_str_to_file(source_string, dirname)
    include_dirs = include_dirs or []
    libraries = libraries or []
    library_dirs = library_dirs or []

    return _c_compile(
        cfile, outputfilename=dirname / modname,
        include_dirs=include_dirs, libraries=libraries,
        library_dirs=library_dirs,
        )


def _convert_str_to_file(source, dirname):
    """Helper function to create a file ``source.c`` in `dirname` that contains
    the string in `source`. Returns the file name
    """
    filename = dirname / 'source.c'
    with filename.open('w') as f:
        f.write(str(source))
    return filename


def _make_methods(functions, modname):
    """ Turns the name, signature, code in functions into complete functions
    and lists them in a methods_table. Then turns the methods_table into a
    ``PyMethodDef`` structure and returns the resulting code fragment ready
    for compilation
    """
    methods_table = []
    codes = []
    for funcname, flags, code in functions:
        cfuncname = f"{modname}_{funcname}"
        if 'METH_KEYWORDS' in flags:
            signature = '(PyObject *self, PyObject *args, PyObject *kwargs)'
        else:
            signature = '(PyObject *self, PyObject *args)'
        methods_table.append(
            "{\"%s\", (PyCFunction)%s, %s}," % (funcname, cfuncname, flags))
        func_code = f"""
        static PyObject* {cfuncname}{signature}
        {{
        {code}
        }}
        """
        codes.append(func_code)

    body = "\n".join(codes) + """
    static PyMethodDef methods[] = {
    %(methods)s
    { NULL }
    };
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "%(modname)s",  /* m_name */
        NULL,           /* m_doc */
        -1,             /* m_size */
        methods,        /* m_methods */
    };
    """ % {'methods': '\n'.join(methods_table), 'modname': modname}
    return body


def _make_source(name, init, body):
    """ Combines the code fragments into source code ready to be compiled
    """
    code = """
    #include <Python.h>

    %(body)s

    PyMODINIT_FUNC
    PyInit_%(name)s(void) {
    %(init)s
    }
    """ % {
        'name': name, 'init': init, 'body': body,
    }
    return code


def _c_compile(cfile, outputfilename, include_dirs, libraries,
               library_dirs):
    link_extra = []
    if sys.platform == 'win32':
        compile_extra = ["/we4013"]
        link_extra.append('/DEBUG')  # generate .pdb file
    elif sys.platform.startswith('linux'):
        compile_extra = [
            "-O0", "-g", "-Werror=implicit-function-declaration", "-fPIC"]
    else:
        compile_extra = []

    return build(
        cfile, outputfilename,
        compile_extra, link_extra,
        include_dirs, libraries, library_dirs)


def build(cfile, outputfilename, compile_extra, link_extra,
          include_dirs, libraries, library_dirs):
    "use meson to build"

    build_dir = cfile.parent / "build"
    os.makedirs(build_dir, exist_ok=True)
    with open(cfile.parent / "meson.build", "wt") as fid:
        link_dirs = ['-L' + d for d in library_dirs]
        fid.write(textwrap.dedent(f"""\
            project('foo', 'c')
            py = import('python').find_installation(pure: false)
            py.extension_module(
                '{outputfilename.parts[-1]}',
                '{cfile.parts[-1]}',
                c_args: {compile_extra},
                link_args: {link_dirs},
                include_directories: {include_dirs},
            )
        """))
    native_file_name = cfile.parent / ".mesonpy-native-file.ini"
    with open(native_file_name, "wt") as fid:
        fid.write(textwrap.dedent(f"""\
            [binaries]
            python = '{sys.executable}'
        """))
    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release",
                               "--vsenv", ".."],
                              cwd=build_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup", "--vsenv",
                               "..", f'--native-file={os.fspath(native_file_name)}'],
                              cwd=build_dir
                              )

    so_name = outputfilename.parts[-1] + get_so_suffix()
    subprocess.check_call(["meson", "compile"], cwd=build_dir)
    os.rename(str(build_dir / so_name), cfile.parent / so_name)
    return cfile.parent / so_name


def get_so_suffix():
    ret = sysconfig.get_config_var('EXT_SUFFIX')
    assert ret
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\decorators.py ===
"""
SymPy core decorators.

The purpose of this module is to expose decorators without any other
dependencies, so that they can be easily imported anywhere in sympy/core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import wraps
from .sympify import SympifyError, sympify


if TYPE_CHECKING:
    from typing import Callable, TypeVar, Union
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')
    T3 = TypeVar('T3')


def _sympifyit(arg, retval=None) -> Callable[[Callable[[T1, T2], T3]], Callable[[T1, T2], T3]]:
    """
    decorator to smartly _sympify function arguments

    Explanation
    ===========

    @_sympifyit('other', NotImplemented)
    def add(self, other):
        ...

    In add, other can be thought of as already being a SymPy object.

    If it is not, the code is likely to catch an exception, then other will
    be explicitly _sympified, and the whole code restarted.

    if _sympify(arg) fails, NotImplemented will be returned

    See also
    ========

    __sympifyit
    """
    def deco(func):
        return __sympifyit(func, arg, retval)

    return deco


def __sympifyit(func, arg, retval=None):
    """Decorator to _sympify `arg` argument for function `func`.

       Do not use directly -- use _sympifyit instead.
    """

    # we support f(a,b) only
    if not func.__code__.co_argcount:
        raise LookupError("func not found")
    # only b is _sympified
    assert func.__code__.co_varnames[1] == arg
    if retval is None:
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            return func(a, sympify(b, strict=True))

    else:
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                # If an external class has _op_priority, it knows how to deal
                # with SymPy objects. Otherwise, it must be converted.
                if not hasattr(b, '_op_priority'):
                    b = sympify(b, strict=True)
                return func(a, b)
            except SympifyError:
                return retval

    return __sympifyit_wrapper


def call_highest_priority(method_name: str
    ) -> Callable[[Callable[[T1, T2], T3]], Callable[[T1, T2], T3]]:
    """A decorator for binary special methods to handle _op_priority.

    Explanation
    ===========

    Binary special methods in Expr and its subclasses use a special attribute
    '_op_priority' to determine whose special method will be called to
    handle the operation. In general, the object having the highest value of
    '_op_priority' will handle the operation. Expr and subclasses that define
    custom binary special methods (__mul__, etc.) should decorate those
    methods with this decorator to add the priority logic.

    The ``method_name`` argument is the name of the method of the other class
    that will be called.  Use this decorator in the following manner::

        # Call other.__rmul__ if other._op_priority > self._op_priority
        @call_highest_priority('__rmul__')
        def __mul__(self, other):
            ...

        # Call other.__mul__ if other._op_priority > self._op_priority
        @call_highest_priority('__mul__')
        def __rmul__(self, other):
        ...
    """
    def priority_decorator(func: Callable[[T1, T2], T3]) -> Callable[[T1, T2], T3]:
        @wraps(func)
        def binary_op_wrapper(self: T1, other: T2) -> T3:
            if hasattr(other, '_op_priority'):
                if other._op_priority > self._op_priority:  # type: ignore
                    f: Union[Callable[[T1], T3], None] = getattr(other, method_name, None)
                    if f is not None:
                        return f(self)
            return func(self, other)
        return binary_op_wrapper
    return priority_decorator


def sympify_method_args(cls: type[T1]) -> type[T1]:
    '''Decorator for a class with methods that sympify arguments.

    Explanation
    ===========

    The sympify_method_args decorator is to be used with the sympify_return
    decorator for automatic sympification of method arguments. This is
    intended for the common idiom of writing a class like :

    Examples
    ========

    >>> from sympy import Basic, SympifyError, S
    >>> from sympy.core.sympify import _sympify

    >>> class MyTuple(Basic):
    ...     def __add__(self, other):
    ...         try:
    ...             other = _sympify(other)
    ...         except SympifyError:
    ...             return NotImplemented
    ...         if not isinstance(other, MyTuple):
    ...             return NotImplemented
    ...         return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    In the above it is important that we return NotImplemented when other is
    not sympifiable and also when the sympified result is not of the expected
    type. This allows the MyTuple class to be used cooperatively with other
    classes that overload __add__ and want to do something else in combination
    with instance of Tuple.

    Using this decorator the above can be written as

    >>> from sympy.core.decorators import sympify_method_args, sympify_return

    >>> @sympify_method_args
    ... class MyTuple(Basic):
    ...     @sympify_return([('other', 'MyTuple')], NotImplemented)
    ...     def __add__(self, other):
    ...          return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    The idea here is that the decorators take care of the boiler-plate code
    for making this happen in each method that potentially needs to accept
    unsympified arguments. Then the body of e.g. the __add__ method can be
    written without needing to worry about calling _sympify or checking the
    type of the resulting object.

    The parameters for sympify_return are a list of tuples of the form
    (parameter_name, expected_type) and the value to return (e.g.
    NotImplemented). The expected_type parameter can be a type e.g. Tuple or a
    string 'Tuple'. Using a string is useful for specifying a Type within its
    class body (as in the above example).

    Notes: Currently sympify_return only works for methods that take a single
    argument (not including self). Specifying an expected_type as a string
    only works for the class in which the method is defined.
    '''
    # Extract the wrapped methods from each of the wrapper objects created by
    # the sympify_return decorator. Doing this here allows us to provide the
    # cls argument which is used for forward string referencing.
    for attrname, obj in cls.__dict__.items():
        if isinstance(obj, _SympifyWrapper):
            setattr(cls, attrname, obj.make_wrapped(cls))
    return cls


def sympify_return(*args):
    '''Function/method decorator to sympify arguments automatically

    See the docstring of sympify_method_args for explanation.
    '''
    # Store a wrapper object for the decorated method
    def wrapper(func: Callable[[T1, T2], T3]) -> Callable[[T1, T2], T3]:
        return _SympifyWrapper(func, args)  # type: ignore
    return wrapper


class _SympifyWrapper:
    '''Internal class used by sympify_return and sympify_method_args'''

    def __init__(self, func, args):
        self.func = func
        self.args = args

    def make_wrapped(self, cls):
        func = self.func
        parameters, retval = self.args

        # XXX: Handle more than one parameter?
        [(parameter, expectedcls)] = parameters

        # Handle forward references to the current class using strings
        if expectedcls == cls.__name__:
            expectedcls = cls

        # Raise RuntimeError since this is a failure at import time and should
        # not be recoverable.
        nargs = func.__code__.co_argcount
        # we support f(a, b) only
        if nargs != 2:
            raise RuntimeError('sympify_return can only be used with 2 argument functions')
        # only b is _sympified
        if func.__code__.co_varnames[1] != parameter:
            raise RuntimeError('parameter name mismatch "%s" in %s' %
                    (parameter, func.__name__))

        @wraps(func)
        def _func(self, other):
            # XXX: The check for _op_priority here should be removed. It is
            # needed to stop mutable matrices from being sympified to
            # immutable matrices which breaks things in quantum...
            if not hasattr(other, '_op_priority'):
                try:
                    other = sympify(other, strict=True)
                except SympifyError:
                    return retval
            if not isinstance(other, expectedcls):
                return retval
            return func(self, other)

        return _func

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\platformdirs\android.py ===
"""Android."""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from .api import PlatformDirsABC


class Android(PlatformDirsABC):
    """
    Follows the guidance `from here <https://android.stackexchange.com/a/216132>`_.

    Makes use of the `appname <platformdirs.api.PlatformDirsABC.appname>`, `version
    <platformdirs.api.PlatformDirsABC.version>`, `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.

    """

    @property
    def user_data_dir(self) -> str:
        """:return: data directory tied to the user, e.g. ``/data/user/<userid>/<packagename>/files/<AppName>``"""
        return self._append_app_name_and_version(cast("str", _android_folder()), "files")

    @property
    def site_data_dir(self) -> str:
        """:return: data directory shared by users, same as `user_data_dir`"""
        return self.user_data_dir

    @property
    def user_config_dir(self) -> str:
        """
        :return: config directory tied to the user, e.g. \
        ``/data/user/<userid>/<packagename>/shared_prefs/<AppName>``
        """
        return self._append_app_name_and_version(cast("str", _android_folder()), "shared_prefs")

    @property
    def site_config_dir(self) -> str:
        """:return: config directory shared by the users, same as `user_config_dir`"""
        return self.user_config_dir

    @property
    def user_cache_dir(self) -> str:
        """:return: cache directory tied to the user, e.g.,``/data/user/<userid>/<packagename>/cache/<AppName>``"""
        return self._append_app_name_and_version(cast("str", _android_folder()), "cache")

    @property
    def site_cache_dir(self) -> str:
        """:return: cache directory shared by users, same as `user_cache_dir`"""
        return self.user_cache_dir

    @property
    def user_state_dir(self) -> str:
        """:return: state directory tied to the user, same as `user_data_dir`"""
        return self.user_data_dir

    @property
    def user_log_dir(self) -> str:
        """
        :return: log directory tied to the user, same as `user_cache_dir` if not opinionated else ``log`` in it,
          e.g. ``/data/user/<userid>/<packagename>/cache/<AppName>/log``
        """
        path = self.user_cache_dir
        if self.opinion:
            path = os.path.join(path, "log")  # noqa: PTH118
        return path

    @property
    def user_documents_dir(self) -> str:
        """:return: documents directory tied to the user e.g. ``/storage/emulated/0/Documents``"""
        return _android_documents_folder()

    @property
    def user_downloads_dir(self) -> str:
        """:return: downloads directory tied to the user e.g. ``/storage/emulated/0/Downloads``"""
        return _android_downloads_folder()

    @property
    def user_pictures_dir(self) -> str:
        """:return: pictures directory tied to the user e.g. ``/storage/emulated/0/Pictures``"""
        return _android_pictures_folder()

    @property
    def user_videos_dir(self) -> str:
        """:return: videos directory tied to the user e.g. ``/storage/emulated/0/DCIM/Camera``"""
        return _android_videos_folder()

    @property
    def user_music_dir(self) -> str:
        """:return: music directory tied to the user e.g. ``/storage/emulated/0/Music``"""
        return _android_music_folder()

    @property
    def user_desktop_dir(self) -> str:
        """:return: desktop directory tied to the user e.g. ``/storage/emulated/0/Desktop``"""
        return "/storage/emulated/0/Desktop"

    @property
    def user_runtime_dir(self) -> str:
        """
        :return: runtime directory tied to the user, same as `user_cache_dir` if not opinionated else ``tmp`` in it,
          e.g. ``/data/user/<userid>/<packagename>/cache/<AppName>/tmp``
        """
        path = self.user_cache_dir
        if self.opinion:
            path = os.path.join(path, "tmp")  # noqa: PTH118
        return path

    @property
    def site_runtime_dir(self) -> str:
        """:return: runtime directory shared by users, same as `user_runtime_dir`"""
        return self.user_runtime_dir


@lru_cache(maxsize=1)
def _android_folder() -> str | None:  # noqa: C901
    """:return: base folder for the Android OS or None if it cannot be found"""
    result: str | None = None
    # type checker isn't happy with our "import android", just don't do this when type checking see
    # https://stackoverflow.com/a/61394121
    if not TYPE_CHECKING:
        try:
            # First try to get a path to android app using python4android (if available)...
            from android import mActivity  # noqa: PLC0415

            context = cast("android.content.Context", mActivity.getApplicationContext())  # noqa: F821
            result = context.getFilesDir().getParentFile().getAbsolutePath()
        except Exception:  # noqa: BLE001
            result = None
    if result is None:
        try:
            # ...and fall back to using plain pyjnius, if python4android isn't available or doesn't deliver any useful
            # result...
            from jnius import autoclass  # noqa: PLC0415

            context = autoclass("android.content.Context")
            result = context.getFilesDir().getParentFile().getAbsolutePath()
        except Exception:  # noqa: BLE001
            result = None
    if result is None:
        # and if that fails, too, find an android folder looking at path on the sys.path
        # warning: only works for apps installed under /data, not adopted storage etc.
        pattern = re.compile(r"/data/(data|user/\d+)/(.+)/files")
        for path in sys.path:
            if pattern.match(path):
                result = path.split("/files")[0]
                break
        else:
            result = None
    if result is None:
        # one last try: find an android folder looking at path on the sys.path taking adopted storage paths into
        # account
        pattern = re.compile(r"/mnt/expand/[a-fA-F0-9-]{36}/(data|user/\d+)/(.+)/files")
        for path in sys.path:
            if pattern.match(path):
                result = path.split("/files")[0]
                break
        else:
            result = None
    return result


@lru_cache(maxsize=1)
def _android_documents_folder() -> str:
    """:return: documents folder for the Android OS"""
    # Get directories with pyjnius
    try:
        from jnius import autoclass  # noqa: PLC0415

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        documents_dir: str = context.getExternalFilesDir(environment.DIRECTORY_DOCUMENTS).getAbsolutePath()
    except Exception:  # noqa: BLE001
        documents_dir = "/storage/emulated/0/Documents"

    return documents_dir


@lru_cache(maxsize=1)
def _android_downloads_folder() -> str:
    """:return: downloads folder for the Android OS"""
    # Get directories with pyjnius
    try:
        from jnius import autoclass  # noqa: PLC0415

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        downloads_dir: str = context.getExternalFilesDir(environment.DIRECTORY_DOWNLOADS).getAbsolutePath()
    except Exception:  # noqa: BLE001
        downloads_dir = "/storage/emulated/0/Downloads"

    return downloads_dir


@lru_cache(maxsize=1)
def _android_pictures_folder() -> str:
    """:return: pictures folder for the Android OS"""
    # Get directories with pyjnius
    try:
        from jnius import autoclass  # noqa: PLC0415

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        pictures_dir: str = context.getExternalFilesDir(environment.DIRECTORY_PICTURES).getAbsolutePath()
    except Exception:  # noqa: BLE001
        pictures_dir = "/storage/emulated/0/Pictures"

    return pictures_dir


@lru_cache(maxsize=1)
def _android_videos_folder() -> str:
    """:return: videos folder for the Android OS"""
    # Get directories with pyjnius
    try:
        from jnius import autoclass  # noqa: PLC0415

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        videos_dir: str = context.getExternalFilesDir(environment.DIRECTORY_DCIM).getAbsolutePath()
    except Exception:  # noqa: BLE001
        videos_dir = "/storage/emulated/0/DCIM/Camera"

    return videos_dir


@lru_cache(maxsize=1)
def _android_music_folder() -> str:
    """:return: music folder for the Android OS"""
    # Get directories with pyjnius
    try:
        from jnius import autoclass  # noqa: PLC0415

        context = autoclass("android.content.Context")
        environment = autoclass("android.os.Environment")
        music_dir: str = context.getExternalFilesDir(environment.DIRECTORY_MUSIC).getAbsolutePath()
    except Exception:  # noqa: BLE001
        music_dir = "/storage/emulated/0/Music"

    return music_dir


__all__ = [
    "Android",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\pretty\tests\test_pretty.py ===
# -*- coding: utf-8 -*-
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core import (EulerGamma, GoldenRatio, Catalan)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import LambertW
from sympy.functions.special.bessel import (airyai, airyaiprime, airybi, airybiprime)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.error_functions import (fresnelc, fresnels)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import dirichlet_eta
from sympy.geometry.line import (Ray, Segment)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Equivalent, ITE, Implies, Nand, Nor, Not, Or, Xor)
from sympy.matrices.dense import (Matrix, diag)
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.trace import Trace
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.realfield import RR
from sympy.polys.orderings import (grlex, ilex)
from sympy.polys.polytools import groebner
from sympy.polys.rootoftools import (RootSum, rootof)
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.series.limits import Limit
from sympy.series.order import O
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Complement, FiniteSet, Intersection, Interval, Union)
from sympy.codegen.ast import (Assignment, AddAugmentedAssignment,
    SubAugmentedAssignment, MulAugmentedAssignment, DivAugmentedAssignment, ModAugmentedAssignment)
from sympy.core.expr import UnevaluatedExpr
from sympy.physics.quantum.trace import Tr

from sympy.functions import (Abs, Chi, Ci, Ei, KroneckerDelta,
    Piecewise, Shi, Si, atan2, beta, binomial, catalan, ceiling, cos,
    euler, exp, expint, factorial, factorial2, floor, gamma, hyper, log,
    meijerg, sin, sqrt, subfactorial, tan, uppergamma, lerchphi, polylog,
    elliptic_k, elliptic_f, elliptic_e, elliptic_pi, DiracDelta, bell,
    bernoulli, fibonacci, tribonacci, lucas, stieltjes, mathieuc, mathieus,
    mathieusprime, mathieucprime)

from sympy.matrices import (Adjoint, Inverse, MatrixSymbol, Transpose,
                            KroneckerProduct, BlockMatrix, OneMatrix, ZeroMatrix)
from sympy.matrices.expressions import hadamard_power

from sympy.physics import mechanics
from sympy.physics.control.lti import (TransferFunction, Feedback, TransferFunctionMatrix,
    Series, Parallel, MIMOSeries, MIMOParallel, MIMOFeedback, StateSpace)
from sympy.physics.units import joule, degree
from sympy.printing.pretty import pprint, pretty as xpretty
from sympy.printing.pretty.pretty_symbology import center_accent, is_combining, center
from sympy.sets.conditionset import ConditionSet

from sympy.sets import ImageSet, ProductSet
from sympy.sets.setexpr import SetExpr
from sympy.stats.crv_types import Normal
from sympy.stats.symbolic_probability import (Covariance, Expectation,
                                              Probability, Variance)
from sympy.tensor.array import (ImmutableDenseNDimArray, ImmutableSparseNDimArray,
                                MutableDenseNDimArray, MutableSparseNDimArray, tensorproduct)
from sympy.tensor.functions import TensorProduct
from sympy.tensor.tensor import (TensorIndexType, tensor_indices, TensorHead,
                                 TensorElement, tensor_heads)

from sympy.testing.pytest import raises, _both_exp_pow, warns_deprecated_sympy

from sympy.vector import CoordSys3D, Gradient, Curl, Divergence, Dot, Cross, Laplacian



import sympy as sym
class lowergamma(sym.lowergamma):
    pass   # testing notation inheritance by a subclass with same name

a, b, c, d, x, y, z, k, n, s, p = symbols('a,b,c,d,x,y,z,k,n,s,p')
f = Function("f")
th = Symbol('theta')
ph = Symbol('phi')

"""
Expressions whose pretty-printing is tested here:
(A '#' to the right of an expression indicates that its various acceptable
orderings are accounted for by the tests.)


BASIC EXPRESSIONS:

oo
(x**2)
1/x
y*x**-2
x**Rational(-5,2)
(-2)**x
Pow(3, 1, evaluate=False)
(x**2 + x + 1)  #
1-x  #
1-2*x  #
x/y
-x/y
(x+2)/y  #
(1+x)*y  #3
-5*x/(x+10)  # correct placement of negative sign
1 - Rational(3,2)*(x+1)
-(-x + 5)*(-x - 2*sqrt(2) + 5) - (-y + 5)*(-y + 5) # issue 5524


ORDERING:

x**2 + x + 1
1 - x
1 - 2*x
2*x**4 + y**2 - x**2 + y**3


RELATIONAL:

Eq(x, y)
Lt(x, y)
Gt(x, y)
Le(x, y)
Ge(x, y)
Ne(x/(y+1), y**2)  #


RATIONAL NUMBERS:

y*x**-2
y**Rational(3,2) * x**Rational(-5,2)
sin(x)**3/tan(x)**2


FUNCTIONS (ABS, CONJ, EXP, FUNCTION BRACES, FACTORIAL, FLOOR, CEILING):

(2*x + exp(x))  #
Abs(x)
Abs(x/(x**2+1)) #
Abs(1 / (y - Abs(x)))
factorial(n)
factorial(2*n)
subfactorial(n)
subfactorial(2*n)
factorial(factorial(factorial(n)))
factorial(n+1) #
conjugate(x)
conjugate(f(x+1)) #
f(x)
f(x, y)
f(x/(y+1), y) #
f(x**x**x**x**x**x)
sin(x)**2
conjugate(a+b*I)
conjugate(exp(a+b*I))
conjugate( f(1 + conjugate(f(x))) ) #
f(x/(y+1), y)  # denom of first arg
floor(1 / (y - floor(x)))
ceiling(1 / (y - ceiling(x)))


SQRT:

sqrt(2)
2**Rational(1,3)
2**Rational(1,1000)
sqrt(x**2 + 1)
(1 + sqrt(5))**Rational(1,3)
2**(1/x)
sqrt(2+pi)
(2+(1+x**2)/(2+x))**Rational(1,4)+(1+x**Rational(1,1000))/sqrt(3+x**2)


DERIVATIVES:

Derivative(log(x), x, evaluate=False)
Derivative(log(x), x, evaluate=False) + x  #
Derivative(log(x) + x**2, x, y, evaluate=False)
Derivative(2*x*y, y, x, evaluate=False) + x**2  #
beta(alpha).diff(alpha)


INTEGRALS:

Integral(log(x), x)
Integral(x**2, x)
Integral((sin(x))**2 / (tan(x))**2)
Integral(x**(2**x), x)
Integral(x**2, (x,1,2))
Integral(x**2, (x,Rational(1,2),10))
Integral(x**2*y**2, x,y)
Integral(x**2, (x, None, 1))
Integral(x**2, (x, 1, None))
Integral(sin(th)/cos(ph), (th,0,pi), (ph, 0, 2*pi))


MATRICES:

Matrix([[x**2+1, 1], [y, x+y]])  #
Matrix([[x/y, y, th], [0, exp(I*k*ph), 1]])


PIECEWISE:

Piecewise((x,x<1),(x**2,True))

ITE:

ITE(x, y, z)

SEQUENCES (TUPLES, LISTS, DICTIONARIES):

()
[]
{}
(1/x,)
[x**2, 1/x, x, y, sin(th)**2/cos(ph)**2]
(x**2, 1/x, x, y, sin(th)**2/cos(ph)**2)
{x: sin(x)}
{1/x: 1/y, x: sin(x)**2}  #
[x**2]
(x**2,)
{x**2: 1}


LIMITS:

Limit(x, x, oo)
Limit(x**2, x, 0)
Limit(1/x, x, 0)
Limit(sin(x)/x, x, 0)


UNITS:

joule => kg*m**2/s


SUBS:

Subs(f(x), x, ph**2)
Subs(f(x).diff(x), x, 0)
Subs(f(x).diff(x)/y, (x, y), (0, Rational(1, 2)))


ORDER:

O(1)
O(1/x)
O(x**2 + y**2)

"""


def pretty(expr, order=None):
    """ASCII pretty-printing"""
    return xpretty(expr, order=order, use_unicode=False, wrap_line=False)


def upretty(expr, order=None):
    """Unicode pretty-printing"""
    return xpretty(expr, order=order, use_unicode=True, wrap_line=False)


def test_pretty_ascii_str():
    assert pretty( 'xxx' ) == 'xxx'
    assert pretty( "xxx" ) == 'xxx'
    assert pretty( 'xxx\'xxx' ) == 'xxx\'xxx'
    assert pretty( 'xxx"xxx' ) == 'xxx\"xxx'
    assert pretty( 'xxx\"xxx' ) == 'xxx\"xxx'
    assert pretty( "xxx'xxx" ) == 'xxx\'xxx'
    assert pretty( "xxx\'xxx" ) == 'xxx\'xxx'
    assert pretty( "xxx\"xxx" ) == 'xxx\"xxx'
    assert pretty( "xxx\"xxx\'xxx" ) == 'xxx"xxx\'xxx'
    assert pretty( "xxx\nxxx" ) == 'xxx\nxxx'


def test_pretty_unicode_str():
    assert pretty( 'xxx' ) == 'xxx'
    assert pretty( 'xxx' ) == 'xxx'
    assert pretty( 'xxx\'xxx' ) == 'xxx\'xxx'
    assert pretty( 'xxx"xxx' ) == 'xxx\"xxx'
    assert pretty( 'xxx\"xxx' ) == 'xxx\"xxx'
    assert pretty( "xxx'xxx" ) == 'xxx\'xxx'
    assert pretty( "xxx\'xxx" ) == 'xxx\'xxx'
    assert pretty( "xxx\"xxx" ) == 'xxx\"xxx'
    assert pretty( "xxx\"xxx\'xxx" ) == 'xxx"xxx\'xxx'
    assert pretty( "xxx\nxxx" ) == 'xxx\nxxx'


def test_upretty_greek():
    assert upretty( oo ) == '∞'
    assert upretty( Symbol('alpha^+_1') ) == 'α⁺₁'
    assert upretty( Symbol('beta') ) == 'β'
    assert upretty(Symbol('lambda')) == 'λ'


def test_upretty_multiindex():
    assert upretty( Symbol('beta12') ) == 'β₁₂'
    assert upretty( Symbol('Y00') ) == 'Y₀₀'
    assert upretty( Symbol('Y_00') ) == 'Y₀₀'
    assert upretty( Symbol('F^+-') ) == 'F⁺⁻'


def test_upretty_sub_super():
    assert upretty( Symbol('beta_1_2') ) == 'β₁ ₂'
    assert upretty( Symbol('beta^1^2') ) == 'β¹ ²'
    assert upretty( Symbol('beta_1^2') ) == 'β²₁'
    assert upretty( Symbol('beta_10_20') ) == 'β₁₀ ₂₀'
    assert upretty( Symbol('beta_ax_gamma^i') ) == 'βⁱₐₓ ᵧ'
    assert upretty( Symbol("F^1^2_3_4") ) == 'F¹ ²₃ ₄'
    assert upretty( Symbol("F_1_2^3^4") ) == 'F³ ⁴₁ ₂'
    assert upretty( Symbol("F_1_2_3_4") ) == 'F₁ ₂ ₃ ₄'
    assert upretty( Symbol("F^1^2^3^4") ) == 'F¹ ² ³ ⁴'


def test_upretty_subs_missing_in_24():
    assert upretty( Symbol('F_beta') ) == 'Fᵦ'
    assert upretty( Symbol('F_gamma') ) == 'Fᵧ'
    assert upretty( Symbol('F_rho') ) == 'Fᵨ'
    assert upretty( Symbol('F_phi') ) == 'Fᵩ'
    assert upretty( Symbol('F_chi') ) == 'Fᵪ'

    assert upretty( Symbol('F_a') ) == 'Fₐ'
    assert upretty( Symbol('F_e') ) == 'Fₑ'
    assert upretty( Symbol('F_i') ) == 'Fᵢ'
    assert upretty( Symbol('F_o') ) == 'Fₒ'
    assert upretty( Symbol('F_u') ) == 'Fᵤ'
    assert upretty( Symbol('F_r') ) == 'Fᵣ'
    assert upretty( Symbol('F_v') ) == 'Fᵥ'
    assert upretty( Symbol('F_x') ) == 'Fₓ'


def test_missing_in_2X_issue_9047():
    assert upretty( Symbol('F_h') ) == 'Fₕ'
    assert upretty( Symbol('F_k') ) == 'Fₖ'
    assert upretty( Symbol('F_l') ) == 'Fₗ'
    assert upretty( Symbol('F_m') ) == 'Fₘ'
    assert upretty( Symbol('F_n') ) == 'Fₙ'
    assert upretty( Symbol('F_p') ) == 'Fₚ'
    assert upretty( Symbol('F_s') ) == 'Fₛ'
    assert upretty( Symbol('F_t') ) == 'Fₜ'


def test_upretty_modifiers():
    # Accents
    assert upretty( Symbol('Fmathring') ) == 'F̊'
    assert upretty( Symbol('Fddddot') ) == 'F⃜'
    assert upretty( Symbol('Fdddot') ) == 'F⃛'
    assert upretty( Symbol('Fddot') ) == 'F̈'
    assert upretty( Symbol('Fdot') ) == 'Ḟ'
    assert upretty( Symbol('Fcheck') ) == 'F̌'
    assert upretty( Symbol('Fbreve') ) == 'F̆'
    assert upretty( Symbol('Facute') ) == 'F́'
    assert upretty( Symbol('Fgrave') ) == 'F̀'
    assert upretty( Symbol('Ftilde') ) == 'F̃'
    assert upretty( Symbol('Fhat') ) == 'F̂'
    assert upretty( Symbol('Fbar') ) == 'F̅'
    assert upretty( Symbol('Fvec') ) == 'F⃗'
    assert upretty( Symbol('Fprime') ) == 'F′'
    assert upretty( Symbol('Fprm') ) == 'F′'
    # No faces are actually implemented, but test to make sure the modifiers are stripped
    assert upretty( Symbol('Fbold') ) == 'Fbold'
    assert upretty( Symbol('Fbm') ) == 'Fbm'
    assert upretty( Symbol('Fcal') ) == 'Fcal'
    assert upretty( Symbol('Fscr') ) == 'Fscr'
    assert upretty( Symbol('Ffrak') ) == 'Ffrak'
    # Brackets
    assert upretty( Symbol('Fnorm') ) == '‖F‖'
    assert upretty( Symbol('Favg') ) == '⟨F⟩'
    assert upretty( Symbol('Fabs') ) == '|F|'
    assert upretty( Symbol('Fmag') ) == '|F|'
    # Combinations
    assert upretty( Symbol('xvecdot') ) == 'x⃗̇'
    assert upretty( Symbol('xDotVec') ) == 'ẋ⃗'
    assert upretty( Symbol('xHATNorm') ) == '‖x̂‖'
    assert upretty( Symbol('xMathring_yCheckPRM__zbreveAbs') ) == 'x̊_y̌′__|z̆|'
    assert upretty( Symbol('alphadothat_nVECDOT__tTildePrime') ) == 'α̇̂_n⃗̇__t̃′'
    assert upretty( Symbol('x_dot') ) == 'x_dot'
    assert upretty( Symbol('x__dot') ) == 'x__dot'


def test_pretty_Cycle():
    from sympy.combinatorics.permutations import Cycle
    assert pretty(Cycle(1, 2)) == '(1 2)'
    assert pretty(Cycle(2)) == '(2)'
    assert pretty(Cycle(1, 3)(4, 5)) == '(1 3)(4 5)'
    assert pretty(Cycle()) == '()'


def test_pretty_Permutation():
    from sympy.combinatorics.permutations import Permutation
    p1 = Permutation(1, 2)(3, 4)
    assert xpretty(p1, perm_cyclic=True, use_unicode=True) == "(1 2)(3 4)"
    assert xpretty(p1, perm_cyclic=True, use_unicode=False) == "(1 2)(3 4)"
    assert xpretty(p1, perm_cyclic=False, use_unicode=True) == \
    '⎛0 1 2 3 4⎞\n'\
    '⎝0 2 1 4 3⎠'
    assert xpretty(p1, perm_cyclic=False, use_unicode=False) == \
    "/0 1 2 3 4\\\n"\
    "\\0 2 1 4 3/"

    with warns_deprecated_sympy():
        old_print_cyclic = Permutation.print_cyclic
        Permutation.print_cyclic = False
        assert xpretty(p1, use_unicode=True) == \
        '⎛0 1 2 3 4⎞\n'\
        '⎝0 2 1 4 3⎠'
        assert xpretty(p1, use_unicode=False) == \
        "/0 1 2 3 4\\\n"\
        "\\0 2 1 4 3/"
        Permutation.print_cyclic = old_print_cyclic


def test_pretty_basic():
    assert pretty( -Rational(1)/2 ) == '-1/2'
    assert pretty( -Rational(13)/22 ) == \
"""\
-13 \n\
----\n\
 22 \
"""
    expr = oo
    ascii_str = \
"""\
oo\
"""
    ucode_str = \
"""\
∞\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (x**2)
    ascii_str = \
"""\
 2\n\
x \
"""
    ucode_str = \
"""\
 2\n\
x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = 1/x
    ascii_str = \
"""\
1\n\
-\n\
x\
"""
    ucode_str = \
"""\
1\n\
─\n\
x\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # not the same as 1/x
    expr = x**-1.0
    ascii_str = \
"""\
 -1.0\n\
x    \
"""
    ucode_str = \
"""\
 -1.0\n\
x    \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # see issue #2860
    expr = Pow(S(2), -1.0, evaluate=False)
    ascii_str = \
"""\
 -1.0\n\
2    \
"""
    ucode_str = \
"""\
 -1.0\n\
2    \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = y*x**-2
    ascii_str = \
"""\
y \n\
--\n\
 2\n\
x \
"""
    ucode_str = \
"""\
y \n\
──\n\
 2\n\
x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    #see issue #14033
    expr = x**Rational(1, 3)
    ascii_str = \
"""\
 1/3\n\
x   \
"""
    ucode_str = \
"""\
 1/3\n\
x   \
"""
    assert xpretty(expr, use_unicode=False, wrap_line=False,\
    root_notation = False) == ascii_str
    assert xpretty(expr, use_unicode=True, wrap_line=False,\
    root_notation = False) == ucode_str

    expr = x**Rational(-5, 2)
    ascii_str = \
"""\
 1  \n\
----\n\
 5/2\n\
x   \
"""
    ucode_str = \
"""\
 1  \n\
────\n\
 5/2\n\
x   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (-2)**x
    ascii_str = \
"""\
    x\n\
(-2) \
"""
    ucode_str = \
"""\
    x\n\
(-2) \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # See issue 4923
    expr = Pow(3, 1, evaluate=False)
    ascii_str = \
"""\
 1\n\
3 \
"""
    ucode_str = \
"""\
 1\n\
3 \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (x**2 + x + 1)
    ascii_str_1 = \
"""\
         2\n\
1 + x + x \
"""
    ascii_str_2 = \
"""\
 2        \n\
x  + x + 1\
"""
    ascii_str_3 = \
"""\
 2        \n\
x  + 1 + x\
"""
    ucode_str_1 = \
"""\
         2\n\
1 + x + x \
"""
    ucode_str_2 = \
"""\
 2        \n\
x  + x + 1\
"""
    ucode_str_3 = \
"""\
 2        \n\
x  + 1 + x\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2, ascii_str_3]
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]

    expr = 1 - x
    ascii_str_1 = \
"""\
1 - x\
"""
    ascii_str_2 = \
"""\
-x + 1\
"""
    ucode_str_1 = \
"""\
1 - x\
"""
    ucode_str_2 = \
"""\
-x + 1\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = 1 - 2*x
    ascii_str_1 = \
"""\
1 - 2*x\
"""
    ascii_str_2 = \
"""\
-2*x + 1\
"""
    ucode_str_1 = \
"""\
1 - 2⋅x\
"""
    ucode_str_2 = \
"""\
-2⋅x + 1\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = x/y
    ascii_str = \
"""\
x\n\
-\n\
y\
"""
    ucode_str = \
"""\
x\n\
─\n\
y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -x/y
    ascii_str = \
"""\
-x \n\
---\n\
 y \
"""
    ucode_str = \
"""\
-x \n\
───\n\
 y \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (x + 2)/y
    ascii_str_1 = \
"""\
2 + x\n\
-----\n\
  y  \
"""
    ascii_str_2 = \
"""\
x + 2\n\
-----\n\
  y  \
"""
    ucode_str_1 = \
"""\
2 + x\n\
─────\n\
  y  \
"""
    ucode_str_2 = \
"""\
x + 2\n\
─────\n\
  y  \
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = (1 + x)*y
    ascii_str_1 = \
"""\
y*(1 + x)\
"""
    ascii_str_2 = \
"""\
(1 + x)*y\
"""
    ascii_str_3 = \
"""\
y*(x + 1)\
"""
    ucode_str_1 = \
"""\
y⋅(1 + x)\
"""
    ucode_str_2 = \
"""\
(1 + x)⋅y\
"""
    ucode_str_3 = \
"""\
y⋅(x + 1)\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2, ascii_str_3]
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]

    # Test for correct placement of the negative sign
    expr = -5*x/(x + 10)
    ascii_str_1 = \
"""\
-5*x  \n\
------\n\
10 + x\
"""
    ascii_str_2 = \
"""\
-5*x  \n\
------\n\
x + 10\
"""
    ucode_str_1 = \
"""\
-5⋅x  \n\
──────\n\
10 + x\
"""
    ucode_str_2 = \
"""\
-5⋅x  \n\
──────\n\
x + 10\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = -S.Half - 3*x
    ascii_str = \
"""\
-3*x - 1/2\
"""
    ucode_str = \
"""\
-3⋅x - 1/2\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = S.Half - 3*x
    ascii_str = \
"""\
1/2 - 3*x\
"""
    ucode_str = \
"""\
1/2 - 3⋅x\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -S.Half - 3*x/2
    ascii_str = \
"""\
  3*x   1\n\
- --- - -\n\
   2    2\
"""
    ucode_str = \
"""\
  3⋅x   1\n\
- ─── - ─\n\
   2    2\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = S.Half - 3*x/2
    ascii_str = \
"""\
1   3*x\n\
- - ---\n\
2    2 \
"""
    ucode_str = \
"""\
1   3⋅x\n\
─ - ───\n\
2    2 \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_negative_fractions():
    expr = -x/y
    ascii_str =\
"""\
-x \n\
---\n\
 y \
"""
    ucode_str =\
"""\
-x \n\
───\n\
 y \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = -x*z/y
    ascii_str =\
"""\
-x*z \n\
-----\n\
  y  \
"""
    ucode_str =\
"""\
-x⋅z \n\
─────\n\
  y  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = x**2/y
    ascii_str =\
"""\
 2\n\
x \n\
--\n\
y \
"""
    ucode_str =\
"""\
 2\n\
x \n\
──\n\
y \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = -x**2/y
    ascii_str =\
"""\
  2 \n\
-x  \n\
----\n\
 y  \
"""
    ucode_str =\
"""\
  2 \n\
-x  \n\
────\n\
 y  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = -x/(y*z)
    ascii_str =\
"""\
-x \n\
---\n\
y*z\
"""
    ucode_str =\
"""\
-x \n\
───\n\
y⋅z\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = -a/y**2
    ascii_str =\
"""\
-a \n\
---\n\
 2 \n\
y  \
"""
    ucode_str =\
"""\
-a \n\
───\n\
 2 \n\
y  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = y**(-a/b)
    ascii_str =\
"""\
 -a \n\
 ---\n\
  b \n\
y   \
"""
    ucode_str =\
"""\
 -a \n\
 ───\n\
  b \n\
y   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = -1/y**2
    ascii_str =\
"""\
-1 \n\
---\n\
 2 \n\
y  \
"""
    ucode_str =\
"""\
-1 \n\
───\n\
 2 \n\
y  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = -10/b**2
    ascii_str =\
"""\
-10 \n\
----\n\
  2 \n\
 b  \
"""
    ucode_str =\
"""\
-10 \n\
────\n\
  2 \n\
 b  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    expr = Rational(-200, 37)
    ascii_str =\
"""\
-200 \n\
-----\n\
 37  \
"""
    ucode_str =\
"""\
-200 \n\
─────\n\
 37  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_Mul():
    expr = Mul(0, 1, evaluate=False)
    assert pretty(expr) == "0*1"
    assert upretty(expr) == "0⋅1"
    expr = Mul(1, 0, evaluate=False)
    assert pretty(expr) == "1*0"
    assert upretty(expr) == "1⋅0"
    expr = Mul(1, 1, evaluate=False)
    assert pretty(expr) == "1*1"
    assert upretty(expr) == "1⋅1"
    expr = Mul(1, 1, 1, evaluate=False)
    assert pretty(expr) == "1*1*1"
    assert upretty(expr) == "1⋅1⋅1"
    expr = Mul(1, 2, evaluate=False)
    assert pretty(expr) == "1*2"
    assert upretty(expr) == "1⋅2"
    expr = Add(0, 1, evaluate=False)
    assert pretty(expr) == "0 + 1"
    assert upretty(expr) == "0 + 1"
    expr = Mul(1, 1, 2, evaluate=False)
    assert pretty(expr) == "1*1*2"
    assert upretty(expr) == "1⋅1⋅2"
    expr = Add(0, 0, 1, evaluate=False)
    assert pretty(expr) == "0 + 0 + 1"
    assert upretty(expr) == "0 + 0 + 1"
    expr = Mul(1, -1, evaluate=False)
    assert pretty(expr) == "1*-1"
    assert upretty(expr) == "1⋅-1"
    expr = Mul(1.0, x, evaluate=False)
    assert pretty(expr) == "1.0*x"
    assert upretty(expr) == "1.0⋅x"
    expr = Mul(1, 1, 2, 3, x, evaluate=False)
    assert pretty(expr) == "1*1*2*3*x"
    assert upretty(expr) == "1⋅1⋅2⋅3⋅x"
    expr = Mul(-1, 1, evaluate=False)
    assert pretty(expr) == "-1*1"
    assert upretty(expr) == "-1⋅1"
    expr = Mul(4, 3, 2, 1, 0, y, x, evaluate=False)
    assert pretty(expr) == "4*3*2*1*0*y*x"
    assert upretty(expr) == "4⋅3⋅2⋅1⋅0⋅y⋅x"
    expr = Mul(4, 3, 2, 1+z, 0, y, x, evaluate=False)
    assert pretty(expr) == "4*3*2*(z + 1)*0*y*x"
    assert upretty(expr) == "4⋅3⋅2⋅(z + 1)⋅0⋅y⋅x"
    expr = Mul(Rational(2, 3), Rational(5, 7), evaluate=False)
    assert pretty(expr) == "2/3*5/7"
    assert upretty(expr) == "2/3⋅5/7"
    expr = Mul(x + y, Rational(1, 2), evaluate=False)
    assert pretty(expr) == "(x + y)*1/2"
    assert upretty(expr) == "(x + y)⋅1/2"
    expr = Mul(Rational(1, 2), x + y, evaluate=False)
    assert pretty(expr) == "x + y\n-----\n  2  "
    assert upretty(expr) == "x + y\n─────\n  2  "
    expr = Mul(S.One, x + y, evaluate=False)
    assert pretty(expr) == "1*(x + y)"
    assert upretty(expr) == "1⋅(x + y)"
    expr = Mul(x - y, S.One, evaluate=False)
    assert pretty(expr) == "(x - y)*1"
    assert upretty(expr) == "(x - y)⋅1"
    expr = Mul(Rational(1, 2), x - y, S.One, x + y, evaluate=False)
    assert pretty(expr) == "1/2*(x - y)*1*(x + y)"
    assert upretty(expr) == "1/2⋅(x - y)⋅1⋅(x + y)"
    expr = Mul(x + y, Rational(3, 4), S.One, y - z, evaluate=False)
    assert pretty(expr) == "(x + y)*3/4*1*(y - z)"
    assert upretty(expr) == "(x + y)⋅3/4⋅1⋅(y - z)"
    expr = Mul(x + y, Rational(1, 1), Rational(3, 4), Rational(5, 6),evaluate=False)
    assert pretty(expr) == "(x + y)*1*3/4*5/6"
    assert upretty(expr) == "(x + y)⋅1⋅3/4⋅5/6"
    expr = Mul(Rational(3, 4), x + y, S.One, y - z, evaluate=False)
    assert pretty(expr) == "3/4*(x + y)*1*(y - z)"
    assert upretty(expr) == "3/4⋅(x + y)⋅1⋅(y - z)"


def test_issue_5524():
    assert pretty(-(-x + 5)*(-x - 2*sqrt(2) + 5) - (-y + 5)*(-y + 5)) == \
"""\
         2           /         ___    \\\n\
- (5 - y)  + (x - 5)*\\-x - 2*\\/ 2  + 5/\
"""

    assert upretty(-(-x + 5)*(-x - 2*sqrt(2) + 5) - (-y + 5)*(-y + 5)) == \
"""\
         2                          \n\
- (5 - y)  + (x - 5)⋅(-x - 2⋅√2 + 5)\
"""


def test_pretty_ordering():
    assert pretty(x**2 + x + 1, order='lex') == \
"""\
 2        \n\
x  + x + 1\
"""
    assert pretty(x**2 + x + 1, order='rev-lex') == \
"""\
         2\n\
1 + x + x \
"""
    assert pretty(1 - x, order='lex') == '-x + 1'
    assert pretty(1 - x, order='rev-lex') == '1 - x'

    assert pretty(1 - 2*x, order='lex') == '-2*x + 1'
    assert pretty(1 - 2*x, order='rev-lex') == '1 - 2*x'

    f = 2*x**4 + y**2 - x**2 + y**3
    assert pretty(f, order=None) == \
"""\
   4    2    3    2\n\
2*x  - x  + y  + y \
"""
    assert pretty(f, order='lex') == \
"""\
   4    2    3    2\n\
2*x  - x  + y  + y \
"""
    assert pretty(f, order='rev-lex') == \
"""\
 2    3    2      4\n\
y  + y  - x  + 2*x \
"""

    expr = x - x**3/6 + x**5/120 + O(x**6)
    ascii_str = \
"""\
     3    5         \n\
    x    x      / 6\\\n\
x - -- + --- + O\\x /\n\
    6    120        \
"""
    ucode_str = \
"""\
     3    5         \n\
    x    x      ⎛ 6⎞\n\
x - ── + ─── + O⎝x ⎠\n\
    6    120        \
"""
    assert pretty(expr, order=None) == ascii_str
    assert upretty(expr, order=None) == ucode_str

    assert pretty(expr, order='lex') == ascii_str
    assert upretty(expr, order='lex') == ucode_str

    assert pretty(expr, order='rev-lex') == ascii_str
    assert upretty(expr, order='rev-lex') == ucode_str


def test_EulerGamma():
    assert pretty(EulerGamma) == str(EulerGamma) == "EulerGamma"
    assert upretty(EulerGamma) == "γ"


def test_GoldenRatio():
    assert pretty(GoldenRatio) == str(GoldenRatio) == "GoldenRatio"
    assert upretty(GoldenRatio) == "φ"


def test_Catalan():
    assert pretty(Catalan) == upretty(Catalan) == "G"


def test_pretty_relational():
    expr = Eq(x, y)
    ascii_str = \
"""\
x = y\
"""
    ucode_str = \
"""\
x = y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Lt(x, y)
    ascii_str = \
"""\
x < y\
"""
    ucode_str = \
"""\
x < y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Gt(x, y)
    ascii_str = \
"""\
x > y\
"""
    ucode_str = \
"""\
x > y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Le(x, y)
    ascii_str = \
"""\
x <= y\
"""
    ucode_str = \
"""\
x ≤ y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Ge(x, y)
    ascii_str = \
"""\
x >= y\
"""
    ucode_str = \
"""\
x ≥ y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Ne(x/(y + 1), y**2)
    ascii_str_1 = \
"""\
  x       2\n\
----- != y \n\
1 + y      \
"""
    ascii_str_2 = \
"""\
  x       2\n\
----- != y \n\
y + 1      \
"""
    ucode_str_1 = \
"""\
  x      2\n\
───── ≠ y \n\
1 + y     \
"""
    ucode_str_2 = \
"""\
  x      2\n\
───── ≠ y \n\
y + 1     \
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]


def test_Assignment():
    expr = Assignment(x, y)
    ascii_str = \
"""\
x := y\
"""
    ucode_str = \
"""\
x := y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_AugmentedAssignment():
    expr = AddAugmentedAssignment(x, y)
    ascii_str = \
"""\
x += y\
"""
    ucode_str = \
"""\
x += y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = SubAugmentedAssignment(x, y)
    ascii_str = \
"""\
x -= y\
"""
    ucode_str = \
"""\
x -= y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = MulAugmentedAssignment(x, y)
    ascii_str = \
"""\
x *= y\
"""
    ucode_str = \
"""\
x *= y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = DivAugmentedAssignment(x, y)
    ascii_str = \
"""\
x /= y\
"""
    ucode_str = \
"""\
x /= y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = ModAugmentedAssignment(x, y)
    ascii_str = \
"""\
x %= y\
"""
    ucode_str = \
"""\
x %= y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_rational():
    expr = y*x**-2
    ascii_str = \
"""\
y \n\
--\n\
 2\n\
x \
"""
    ucode_str = \
"""\
y \n\
──\n\
 2\n\
x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = y**Rational(3, 2) * x**Rational(-5, 2)
    ascii_str = \
"""\
 3/2\n\
y   \n\
----\n\
 5/2\n\
x   \
"""
    ucode_str = \
"""\
 3/2\n\
y   \n\
────\n\
 5/2\n\
x   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = sin(x)**3/tan(x)**2
    ascii_str = \
"""\
   3   \n\
sin (x)\n\
-------\n\
   2   \n\
tan (x)\
"""
    ucode_str = \
"""\
   3   \n\
sin (x)\n\
───────\n\
   2   \n\
tan (x)\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


@_both_exp_pow
def test_pretty_functions():
    """Tests for Abs, conjugate, exp, function braces, and factorial."""
    expr = (2*x + exp(x))
    ascii_str_1 = \
"""\
       x\n\
2*x + e \
"""
    ascii_str_2 = \
"""\
 x      \n\
e  + 2*x\
"""
    ucode_str_1 = \
"""\
       x\n\
2⋅x + ℯ \
"""
    ucode_str_2 = \
"""\
 x     \n\
ℯ + 2⋅x\
"""
    ucode_str_3 = \
"""\
 x      \n\
ℯ  + 2⋅x\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]

    expr = Abs(x)
    ascii_str = \
"""\
|x|\
"""
    ucode_str = \
"""\
│x│\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Abs(x/(x**2 + 1))
    ascii_str_1 = \
"""\
|  x   |\n\
|------|\n\
|     2|\n\
|1 + x |\
"""
    ascii_str_2 = \
"""\
|  x   |\n\
|------|\n\
| 2    |\n\
|x  + 1|\
"""
    ucode_str_1 = \
"""\
│  x   │\n\
│──────│\n\
│     2│\n\
│1 + x │\
"""
    ucode_str_2 = \
"""\
│  x   │\n\
│──────│\n\
│ 2    │\n\
│x  + 1│\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = Abs(1 / (y - Abs(x)))
    ascii_str = \
"""\
    1    \n\
---------\n\
|y - |x||\
"""
    ucode_str = \
"""\
    1    \n\
─────────\n\
│y - │x││\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    n = Symbol('n', integer=True)
    expr = factorial(n)
    ascii_str = \
"""\
n!\
"""
    ucode_str = \
"""\
n!\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = factorial(2*n)
    ascii_str = \
"""\
(2*n)!\
"""
    ucode_str = \
"""\
(2⋅n)!\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = factorial(factorial(factorial(n)))
    ascii_str = \
"""\
((n!)!)!\
"""
    ucode_str = \
"""\
((n!)!)!\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = factorial(n + 1)
    ascii_str_1 = \
"""\
(1 + n)!\
"""
    ascii_str_2 = \
"""\
(n + 1)!\
"""
    ucode_str_1 = \
"""\
(1 + n)!\
"""
    ucode_str_2 = \
"""\
(n + 1)!\
"""

    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = subfactorial(n)
    ascii_str = \
"""\
!n\
"""
    ucode_str = \
"""\
!n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = subfactorial(2*n)
    ascii_str = \
"""\
!(2*n)\
"""
    ucode_str = \
"""\
!(2⋅n)\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    n = Symbol('n', integer=True)
    expr = factorial2(n)
    ascii_str = \
"""\
n!!\
"""
    ucode_str = \
"""\
n!!\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = factorial2(2*n)
    ascii_str = \
"""\
(2*n)!!\
"""
    ucode_str = \
"""\
(2⋅n)!!\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = factorial2(factorial2(factorial2(n)))
    ascii_str = \
"""\
((n!!)!!)!!\
"""
    ucode_str = \
"""\
((n!!)!!)!!\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = factorial2(n + 1)
    ascii_str_1 = \
"""\
(1 + n)!!\
"""
    ascii_str_2 = \
"""\
(n + 1)!!\
"""
    ucode_str_1 = \
"""\
(1 + n)!!\
"""
    ucode_str_2 = \
"""\
(n + 1)!!\
"""

    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = 2*binomial(n, k)
    ascii_str = \
"""\
  /n\\\n\
2*| |\n\
  \\k/\
"""
    ucode_str = \
"""\
  ⎛n⎞\n\
2⋅⎜ ⎟\n\
  ⎝k⎠\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = 2*binomial(2*n, k)
    ascii_str = \
"""\
  /2*n\\\n\
2*|   |\n\
  \\ k /\
"""
    ucode_str = \
"""\
  ⎛2⋅n⎞\n\
2⋅⎜   ⎟\n\
  ⎝ k ⎠\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = 2*binomial(n**2, k)
    ascii_str = \
"""\
  / 2\\\n\
  |n |\n\
2*|  |\n\
  \\k /\
"""
    ucode_str = \
"""\
  ⎛ 2⎞\n\
  ⎜n ⎟\n\
2⋅⎜  ⎟\n\
  ⎝k ⎠\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = catalan(n)
    ascii_str = \
"""\
C \n\
 n\
"""
    ucode_str = \
"""\
C \n\
 n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = catalan(n)
    ascii_str = \
"""\
C \n\
 n\
"""
    ucode_str = \
"""\
C \n\
 n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = bell(n)
    ascii_str = \
"""\
B \n\
 n\
"""
    ucode_str = \
"""\
B \n\
 n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = bernoulli(n)
    ascii_str = \
"""\
B \n\
 n\
"""
    ucode_str = \
"""\
B \n\
 n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = bernoulli(n, x)
    ascii_str = \
"""\
B (x)\n\
 n   \
"""
    ucode_str = \
"""\
B (x)\n\
 n   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = fibonacci(n)
    ascii_str = \
"""\
F \n\
 n\
"""
    ucode_str = \
"""\
F \n\
 n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = lucas(n)
    ascii_str = \
"""\
L \n\
 n\
"""
    ucode_str = \
"""\
L \n\
 n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = tribonacci(n)
    ascii_str = \
"""\
T \n\
 n\
"""
    ucode_str = \
"""\
T \n\
 n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = stieltjes(n)
    ascii_str = \
"""\
stieltjes \n\
         n\
"""
    ucode_str = \
"""\
γ \n\
 n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = stieltjes(n, x)
    ascii_str = \
"""\
stieltjes (x)\n\
         n   \
"""
    ucode_str = \
"""\
γ (x)\n\
 n   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = mathieuc(x, y, z)
    ascii_str = 'C(x, y, z)'
    ucode_str = 'C(x, y, z)'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = mathieus(x, y, z)
    ascii_str = 'S(x, y, z)'
    ucode_str = 'S(x, y, z)'
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = mathieucprime(x, y, z)
    ascii_str = "C'(x, y, z)"
    ucode_str = "C'(x, y, z)"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = mathieusprime(x, y, z)
    ascii_str = "S'(x, y, z)"
    ucode_str = "S'(x, y, z)"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = conjugate(x)
    ascii_str = \
"""\
_\n\
x\
"""
    ucode_str = \
"""\
_\n\
x\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    f = Function('f')
    expr = conjugate(f(x + 1))
    ascii_str_1 = \
"""\
________\n\
f(1 + x)\
"""
    ascii_str_2 = \
"""\
________\n\
f(x + 1)\
"""
    ucode_str_1 = \
"""\
________\n\
f(1 + x)\
"""
    ucode_str_2 = \
"""\
________\n\
f(x + 1)\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = f(x)
    ascii_str = \
"""\
f(x)\
"""
    ucode_str = \
"""\
f(x)\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = f(x, y)
    ascii_str = \
"""\
f(x, y)\
"""
    ucode_str = \
"""\
f(x, y)\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = f(x/(y + 1), y)
    ascii_str_1 = \
"""\
 /  x     \\\n\
f|-----, y|\n\
 \\1 + y   /\
"""
    ascii_str_2 = \
"""\
 /  x     \\\n\
f|-----, y|\n\
 \\y + 1   /\
"""
    ucode_str_1 = \
"""\
 ⎛  x     ⎞\n\
f⎜─────, y⎟\n\
 ⎝1 + y   ⎠\
"""
    ucode_str_2 = \
"""\
 ⎛  x     ⎞\n\
f⎜─────, y⎟\n\
 ⎝y + 1   ⎠\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = f(x**x**x**x**x**x)
    ascii_str = \
"""\
 / / / / / x\\\\\\\\\\
 | | | | \\x /||||
 | | | \\x    /|||
 | | \\x       /||
 | \\x          /|
f\\x             /\
"""
    ucode_str = \
"""\
 ⎛ ⎛ ⎛ ⎛ ⎛ x⎞⎞⎞⎞⎞
 ⎜ ⎜ ⎜ ⎜ ⎝x ⎠⎟⎟⎟⎟
 ⎜ ⎜ ⎜ ⎝x    ⎠⎟⎟⎟
 ⎜ ⎜ ⎝x       ⎠⎟⎟
 ⎜ ⎝x          ⎠⎟
f⎝x             ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = sin(x)**2
    ascii_str = \
"""\
   2   \n\
sin (x)\
"""
    ucode_str = \
"""\
   2   \n\
sin (x)\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = conjugate(a + b*I)
    ascii_str = \
"""\
_     _\n\
a - I*b\
"""
    ucode_str = \
"""\
_     _\n\
a - ⅈ⋅b\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = conjugate(exp(a + b*I))
    ascii_str = \
"""\
 _     _\n\
 a - I*b\n\
e       \
"""
    ucode_str = \
"""\
 _     _\n\
 a - ⅈ⋅b\n\
ℯ       \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = conjugate( f(1 + conjugate(f(x))) )
    ascii_str_1 = \
"""\
___________\n\
 /    ____\\\n\
f\\1 + f(x)/\
"""
    ascii_str_2 = \
"""\
___________\n\
 /____    \\\n\
f\\f(x) + 1/\
"""
    ucode_str_1 = \
"""\
___________\n\
 ⎛    ____⎞\n\
f⎝1 + f(x)⎠\
"""
    ucode_str_2 = \
"""\
___________\n\
 ⎛____    ⎞\n\
f⎝f(x) + 1⎠\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = f(x/(y + 1), y)
    ascii_str_1 = \
"""\
 /  x     \\\n\
f|-----, y|\n\
 \\1 + y   /\
"""
    ascii_str_2 = \
"""\
 /  x     \\\n\
f|-----, y|\n\
 \\y + 1   /\
"""
    ucode_str_1 = \
"""\
 ⎛  x     ⎞\n\
f⎜─────, y⎟\n\
 ⎝1 + y   ⎠\
"""
    ucode_str_2 = \
"""\
 ⎛  x     ⎞\n\
f⎜─────, y⎟\n\
 ⎝y + 1   ⎠\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = floor(1 / (y - floor(x)))
    ascii_str = \
"""\
     /     1      \\\n\
floor|------------|\n\
     \\y - floor(x)/\
"""
    ucode_str = \
"""\
⎢   1   ⎥\n\
⎢───────⎥\n\
⎣y - ⌊x⌋⎦\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = ceiling(1 / (y - ceiling(x)))
    ascii_str = \
"""\
       /      1       \\\n\
ceiling|--------------|\n\
       \\y - ceiling(x)/\
"""
    ucode_str = \
"""\
⎡   1   ⎤\n\
⎢───────⎥\n\
⎢y - ⌈x⌉⎥\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = euler(n)
    ascii_str = \
"""\
E \n\
 n\
"""
    ucode_str = \
"""\
E \n\
 n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = euler(1/(1 + 1/(1 + 1/n)))
    ascii_str = \
"""\
E         \n\
     1    \n\
 ---------\n\
       1  \n\
 1 + -----\n\
         1\n\
     1 + -\n\
         n\
"""

    ucode_str = \
"""\
E         \n\
     1    \n\
 ─────────\n\
       1  \n\
 1 + ─────\n\
         1\n\
     1 + ─\n\
         n\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = euler(n, x)
    ascii_str = \
"""\
E (x)\n\
 n   \
"""
    ucode_str = \
"""\
E (x)\n\
 n   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = euler(n, x/2)
    ascii_str = \
"""\
  /x\\\n\
E |-|\n\
 n\\2/\
"""
    ucode_str = \
"""\
  ⎛x⎞\n\
E ⎜─⎟\n\
 n⎝2⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_sqrt():
    expr = sqrt(2)
    ascii_str = \
"""\
  ___\n\
\\/ 2 \
"""
    ucode_str = \
"√2"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = 2**Rational(1, 3)
    ascii_str = \
"""\
3 ___\n\
\\/ 2 \
"""
    ucode_str = \
"""\
3 ___\n\
╲╱ 2 \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = 2**Rational(1, 1000)
    ascii_str = \
"""\
1000___\n\
  \\/ 2 \
"""
    ucode_str = \
"""\
1000___\n\
  ╲╱ 2 \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = sqrt(x**2 + 1)
    ascii_str = \
"""\
   ________\n\
  /  2     \n\
\\/  x  + 1 \
"""
    ucode_str = \
"""\
   ________\n\
  ╱  2     \n\
╲╱  x  + 1 \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (1 + sqrt(5))**Rational(1, 3)
    ascii_str = \
"""\
   ___________\n\
3 /       ___ \n\
\\/  1 + \\/ 5  \
"""
    ucode_str = \
"""\
3 ________\n\
╲╱ 1 + √5 \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = 2**(1/x)
    ascii_str = \
"""\
x ___\n\
\\/ 2 \
"""
    ucode_str = \
"""\
x ___\n\
╲╱ 2 \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = sqrt(2 + pi)
    ascii_str = \
"""\
  ________\n\
\\/ 2 + pi \
"""
    ucode_str = \
"""\
  _______\n\
╲╱ 2 + π \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (2 + (
        1 + x**2)/(2 + x))**Rational(1, 4) + (1 + x**Rational(1, 1000))/sqrt(3 + x**2)
    ascii_str = \
"""\
     ____________              \n\
    /      2        1000___    \n\
   /      x  + 1      \\/ x  + 1\n\
4 /   2 + ------  + -----------\n\
\\/        x + 2        ________\n\
                      /  2     \n\
                    \\/  x  + 3 \
"""
    ucode_str = \
"""\
     ____________              \n\
    ╱      2        1000___    \n\
   ╱      x  + 1      ╲╱ x  + 1\n\
4 ╱   2 + ──────  + ───────────\n\
╲╱        x + 2        ________\n\
                      ╱  2     \n\
                    ╲╱  x  + 3 \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_sqrt_char_knob():
    # See PR #9234.
    expr = sqrt(2)
    ucode_str1 = \
"""\
  ___\n\
╲╱ 2 \
"""
    ucode_str2 = \
"√2"
    assert xpretty(expr, use_unicode=True,
                   use_unicode_sqrt_char=False) == ucode_str1
    assert xpretty(expr, use_unicode=True,
                   use_unicode_sqrt_char=True) == ucode_str2


def test_pretty_sqrt_longsymbol_no_sqrt_char():
    # Do not use unicode sqrt char for long symbols (see PR #9234).
    expr = sqrt(Symbol('C1'))
    ucode_str = \
"""\
  ____\n\
╲╱ C₁ \
"""
    assert upretty(expr) == ucode_str


def test_pretty_KroneckerDelta():
    x, y = symbols("x, y")
    expr = KroneckerDelta(x, y)
    ascii_str = \
"""\
d   \n\
 x,y\
"""
    ucode_str = \
"""\
δ   \n\
 x,y\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_product():
    n, m, k, l = symbols('n m k l')
    f = symbols('f', cls=Function)
    expr = Product(f((n/3)**2), (n, k**2, l))

    unicode_str = \
"""\
    l           \n\
─┬──────┬─      \n\
 │      │   ⎛ 2⎞\n\
 │      │   ⎜n ⎟\n\
 │      │  f⎜──⎟\n\
 │      │   ⎝9 ⎠\n\
 │      │       \n\
       2        \n\
  n = k         """
    ascii_str = \
"""\
    l           \n\
__________      \n\
 |      |   / 2\\\n\
 |      |   |n |\n\
 |      |  f|--|\n\
 |      |   \\9 /\n\
 |      |       \n\
       2        \n\
  n = k         """

    expr = Product(f((n/3)**2), (n, k**2, l), (l, 1, m))

    unicode_str = \
"""\
    m          l           \n\
─┬──────┬─ ─┬──────┬─      \n\
 │      │   │      │   ⎛ 2⎞\n\
 │      │   │      │   ⎜n ⎟\n\
 │      │   │      │  f⎜──⎟\n\
 │      │   │      │   ⎝9 ⎠\n\
 │      │   │      │       \n\
  l = 1           2        \n\
             n = k         """
    ascii_str = \
"""\
    m          l           \n\
__________ __________      \n\
 |      |   |      |   / 2\\\n\
 |      |   |      |   |n |\n\
 |      |   |      |  f|--|\n\
 |      |   |      |   \\9 /\n\
 |      |   |      |       \n\
  l = 1           2        \n\
             n = k         """

    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str


def test_pretty_Lambda():
    # S.IdentityFunction is a special case
    expr = Lambda(y, y)
    assert pretty(expr) == "x -> x"
    assert upretty(expr) == "x ↦ x"

    expr = Lambda(x, x+1)
    assert pretty(expr) == "x -> x + 1"
    assert upretty(expr) == "x ↦ x + 1"

    expr = Lambda(x, x**2)
    ascii_str = \
"""\
      2\n\
x -> x \
"""
    ucode_str = \
"""\
     2\n\
x ↦ x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Lambda(x, x**2)**2
    ascii_str = \
"""\
         2
/      2\\ \n\
\\x -> x / \
"""
    ucode_str = \
"""\
        2
⎛     2⎞ \n\
⎝x ↦ x ⎠ \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Lambda((x, y), x)
    ascii_str = "(x, y) -> x"
    ucode_str = "(x, y) ↦ x"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Lambda((x, y), x**2)
    ascii_str = \
"""\
           2\n\
(x, y) -> x \
"""
    ucode_str = \
"""\
          2\n\
(x, y) ↦ x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Lambda(((x, y),), x**2)
    ascii_str = \
"""\
              2\n\
((x, y),) -> x \
"""
    ucode_str = \
"""\
             2\n\
((x, y),) ↦ x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_TransferFunction():
    tf1 = TransferFunction(s - 1, s + 1, s)
    assert upretty(tf1) == "s - 1\n─────\ns + 1"
    tf2 = TransferFunction(2*s + 1, 3 - p, s)
    assert upretty(tf2) == "2⋅s + 1\n───────\n 3 - p "
    tf3 = TransferFunction(p, p + 1, p)
    assert upretty(tf3) == "  p  \n─────\np + 1"


def test_pretty_Series():
    tf1 = TransferFunction(x + y, x - 2*y, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(x**2 + y, y - x, y)
    tf4 = TransferFunction(2, 3, y)

    tfm1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm2 = TransferFunctionMatrix([[tf3], [-tf4]])
    tfm3 = TransferFunctionMatrix([[tf1, -tf2, -tf3], [tf3, -tf4, tf2]])
    tfm4 = TransferFunctionMatrix([[tf1, tf2], [tf3, -tf4], [-tf2, -tf1]])
    tfm5 = TransferFunctionMatrix([[-tf2, -tf1], [tf4, -tf3], [tf1, tf2]])

    expected1 = \
"""\
          ⎛ 2    ⎞\n\
⎛ x + y ⎞ ⎜x  + y⎟\n\
⎜───────⎟⋅⎜──────⎟\n\
⎝x - 2⋅y⎠ ⎝-x + y⎠\
"""
    expected2 = \
"""\
⎛-x + y⎞ ⎛-x - y ⎞\n\
⎜──────⎟⋅⎜───────⎟\n\
⎝x + y ⎠ ⎝x - 2⋅y⎠\
"""
    expected3 = \
"""\
⎛ 2    ⎞                            \n\
⎜x  + y⎟ ⎛ x + y ⎞ ⎛-x - y    x - y⎞\n\
⎜──────⎟⋅⎜───────⎟⋅⎜─────── + ─────⎟\n\
⎝-x + y⎠ ⎝x - 2⋅y⎠ ⎝x - 2⋅y   x + y⎠\
"""
    expected4 = \
"""\
                  ⎛         2    ⎞\n\
⎛ x + y    x - y⎞ ⎜x - y   x  + y⎟\n\
⎜─────── + ─────⎟⋅⎜───── + ──────⎟\n\
⎝x - 2⋅y   x + y⎠ ⎝x + y   -x + y⎠\
"""
    expected5 = \
"""\
⎡ x + y   x - y⎤  ⎡ 2    ⎤ \n\
⎢───────  ─────⎥  ⎢x  + y⎥ \n\
⎢x - 2⋅y  x + y⎥  ⎢──────⎥ \n\
⎢              ⎥  ⎢-x + y⎥ \n\
⎢ 2            ⎥ ⋅⎢      ⎥ \n\
⎢x  + y     2  ⎥  ⎢ -2   ⎥ \n\
⎢──────     ─  ⎥  ⎢ ───  ⎥ \n\
⎣-x + y     3  ⎦τ ⎣  3   ⎦τ\
"""
    expected6 = \
"""\
                                               ⎛⎡ x + y    x - y ⎤    ⎡ x - y    x + y ⎤ ⎞\n\
                                               ⎜⎢───────   ───── ⎥    ⎢ ─────   ───────⎥ ⎟\n\
⎡ x + y   x - y⎤  ⎡                    2    ⎤  ⎜⎢x - 2⋅y   x + y ⎥    ⎢ x + y   x - 2⋅y⎥ ⎟\n\
⎢───────  ─────⎥  ⎢ x + y   -x + y  - x  - y⎥  ⎜⎢                ⎥    ⎢                ⎥ ⎟\n\
⎢x - 2⋅y  x + y⎥  ⎢───────  ──────  ────────⎥  ⎜⎢ 2              ⎥    ⎢          2     ⎥ ⎟\n\
⎢              ⎥  ⎢x - 2⋅y  x + y    -x + y ⎥  ⎜⎢x  + y     -2   ⎥    ⎢  -2     x  + y ⎥ ⎟\n\
⎢ 2            ⎥ ⋅⎢                         ⎥ ⋅⎜⎢──────     ───  ⎥  + ⎢  ───    ────── ⎥ ⎟\n\
⎢x  + y     2  ⎥  ⎢ 2                       ⎥  ⎜⎢-x + y      3   ⎥    ⎢   3     -x + y ⎥ ⎟\n\
⎢──────     ─  ⎥  ⎢x  + y    -2      x - y  ⎥  ⎜⎢                ⎥    ⎢                ⎥ ⎟\n\
⎣-x + y     3  ⎦τ ⎢──────    ───     ─────  ⎥  ⎜⎢-x + y   -x - y ⎥    ⎢-x - y   -x + y ⎥ ⎟\n\
                  ⎣-x + y     3      x + y  ⎦τ ⎜⎢──────   ───────⎥    ⎢───────  ────── ⎥ ⎟\n\
                                               ⎝⎣x + y    x - 2⋅y⎦τ   ⎣x - 2⋅y  x + y  ⎦τ⎠\
"""

    assert upretty(Series(tf1, tf3)) == expected1
    assert upretty(Series(-tf2, -tf1)) == expected2
    assert upretty(Series(tf3, tf1, Parallel(-tf1, tf2))) == expected3
    assert upretty(Series(Parallel(tf1, tf2), Parallel(tf2, tf3))) == expected4
    assert upretty(MIMOSeries(tfm2, tfm1)) == expected5
    assert upretty(MIMOSeries(MIMOParallel(tfm4, -tfm5), tfm3, tfm1)) == expected6


def test_pretty_Parallel():
    tf1 = TransferFunction(x + y, x - 2*y, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(x**2 + y, y - x, y)
    tf4 = TransferFunction(y**2 - x, x**3 + x, y)

    tfm1 = TransferFunctionMatrix([[tf1, tf2], [tf3, -tf4], [-tf2, -tf1]])
    tfm2 = TransferFunctionMatrix([[-tf2, -tf1], [tf4, -tf3], [tf1, tf2]])
    tfm3 = TransferFunctionMatrix([[-tf1, tf2], [-tf3, tf4], [tf2, tf1]])
    tfm4 = TransferFunctionMatrix([[-tf1, -tf2], [-tf3, -tf4]])

    expected1 = \
"""\
 x + y    x - y\n\
─────── + ─────\n\
x - 2⋅y   x + y\
"""
    expected2 = \
"""\
-x + y   -x - y \n\
────── + ───────
x + y    x - 2⋅y\
"""
    expected3 = \
"""\
 2                                  \n\
x  + y    x + y    ⎛-x - y ⎞ ⎛x - y⎞
────── + ─────── + ⎜───────⎟⋅⎜─────⎟
-x + y   x - 2⋅y   ⎝x - 2⋅y⎠ ⎝x + y⎠\
"""

    expected4 = \
"""\
                            ⎛ 2    ⎞\n\
⎛ x + y ⎞ ⎛x - y⎞   ⎛x - y⎞ ⎜x  + y⎟\n\
⎜───────⎟⋅⎜─────⎟ + ⎜─────⎟⋅⎜──────⎟\n\
⎝x - 2⋅y⎠ ⎝x + y⎠   ⎝x + y⎠ ⎝-x + y⎠\
"""
    expected5 = \
"""\
⎡ x + y   -x + y ⎤    ⎡ x - y    x + y ⎤    ⎡ x + y    x - y ⎤ \n\
⎢───────  ────── ⎥    ⎢ ─────   ───────⎥    ⎢───────   ───── ⎥ \n\
⎢x - 2⋅y  x + y  ⎥    ⎢ x + y   x - 2⋅y⎥    ⎢x - 2⋅y   x + y ⎥ \n\
⎢                ⎥    ⎢                ⎥    ⎢                ⎥ \n\
⎢ 2            2 ⎥    ⎢     2    2     ⎥    ⎢ 2            2 ⎥ \n\
⎢x  + y   x - y  ⎥    ⎢x - y    x  + y ⎥    ⎢x  + y   x - y  ⎥ \n\
⎢──────   ────── ⎥  + ⎢──────   ────── ⎥  + ⎢──────   ────── ⎥ \n\
⎢-x + y    3     ⎥    ⎢ 3       -x + y ⎥    ⎢-x + y    3     ⎥ \n\
⎢         x  + x ⎥    ⎢x  + x          ⎥    ⎢         x  + x ⎥ \n\
⎢                ⎥    ⎢                ⎥    ⎢                ⎥ \n\
⎢-x + y   -x - y ⎥    ⎢-x - y   -x + y ⎥    ⎢-x + y   -x - y ⎥ \n\
⎢──────   ───────⎥    ⎢───────  ────── ⎥    ⎢──────   ───────⎥ \n\
⎣x + y    x - 2⋅y⎦τ   ⎣x - 2⋅y  x + y  ⎦τ   ⎣x + y    x - 2⋅y⎦τ\
"""
    expected6 = \
"""\
⎡ x - y    x + y ⎤                        ⎡-x + y   -x - y  ⎤ \n\
⎢ ─────   ───────⎥                        ⎢──────   ─────── ⎥ \n\
⎢ x + y   x - 2⋅y⎥  ⎡-x - y    -x + y⎤    ⎢x + y    x - 2⋅y ⎥ \n\
⎢                ⎥  ⎢───────   ──────⎥    ⎢                 ⎥ \n\
⎢     2    2     ⎥  ⎢x - 2⋅y   x + y ⎥    ⎢      2     2    ⎥ \n\
⎢x - y    x  + y ⎥  ⎢                ⎥    ⎢-x + y   - x  - y⎥ \n\
⎢──────   ────── ⎥ ⋅⎢   2           2⎥  + ⎢───────  ────────⎥ \n\
⎢ 3       -x + y ⎥  ⎢- x  - y  x - y ⎥    ⎢ 3        -x + y ⎥ \n\
⎢x  + x          ⎥  ⎢────────  ──────⎥    ⎢x  + x           ⎥ \n\
⎢                ⎥  ⎢ -x + y    3    ⎥    ⎢                 ⎥ \n\
⎢-x - y   -x + y ⎥  ⎣          x  + x⎦τ   ⎢ x + y    x - y  ⎥ \n\
⎢───────  ────── ⎥                        ⎢───────   ─────  ⎥ \n\
⎣x - 2⋅y  x + y  ⎦τ                       ⎣x - 2⋅y   x + y  ⎦τ\
"""
    assert upretty(Parallel(tf1, tf2)) == expected1
    assert upretty(Parallel(-tf2, -tf1)) == expected2
    assert upretty(Parallel(tf3, tf1, Series(-tf1, tf2))) == expected3
    assert upretty(Parallel(Series(tf1, tf2), Series(tf2, tf3))) == expected4
    assert upretty(MIMOParallel(-tfm3, -tfm2, tfm1)) == expected5
    assert upretty(MIMOParallel(MIMOSeries(tfm4, -tfm2), tfm2)) == expected6


def test_pretty_Feedback():
    tf = TransferFunction(1, 1, y)
    tf1 = TransferFunction(x + y, x - 2*y, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(y**2 - 2*y + 1, y + 5, y)
    tf4 = TransferFunction(x - 2*y**3, x + y, x)
    tf5 = TransferFunction(1 - x, x - y, y)
    tf6 = TransferFunction(2, 2, x)
    expected1 = \
"""\
     ⎛1⎞     \n\
     ⎜─⎟     \n\
     ⎝1⎠     \n\
─────────────\n\
1   ⎛ x + y ⎞\n\
─ + ⎜───────⎟\n\
1   ⎝x - 2⋅y⎠\
"""
    expected2 = \
"""\
                ⎛1⎞                 \n\
                ⎜─⎟                 \n\
                ⎝1⎠                 \n\
────────────────────────────────────\n\
                      ⎛ 2          ⎞\n\
1   ⎛x - y⎞ ⎛ x + y ⎞ ⎜y  - 2⋅y + 1⎟\n\
─ + ⎜─────⎟⋅⎜───────⎟⋅⎜────────────⎟\n\
1   ⎝x + y⎠ ⎝x - 2⋅y⎠ ⎝   y + 5    ⎠\
"""
    expected3 = \
"""\
                 ⎛ x + y ⎞                  \n\
                 ⎜───────⎟                  \n\
                 ⎝x - 2⋅y⎠                  \n\
────────────────────────────────────────────\n\
                      ⎛ 2          ⎞        \n\
1   ⎛ x + y ⎞ ⎛x - y⎞ ⎜y  - 2⋅y + 1⎟ ⎛1 - x⎞\n\
─ + ⎜───────⎟⋅⎜─────⎟⋅⎜────────────⎟⋅⎜─────⎟\n\
1   ⎝x - 2⋅y⎠ ⎝x + y⎠ ⎝   y + 5    ⎠ ⎝x - y⎠\
"""
    expected4 = \
"""\
  ⎛ x + y ⎞ ⎛x - y⎞  \n\
  ⎜───────⎟⋅⎜─────⎟  \n\
  ⎝x - 2⋅y⎠ ⎝x + y⎠  \n\
─────────────────────\n\
1   ⎛ x + y ⎞ ⎛x - y⎞\n\
─ + ⎜───────⎟⋅⎜─────⎟\n\
1   ⎝x - 2⋅y⎠ ⎝x + y⎠\
"""
    expected5 = \
"""\
      ⎛ x + y ⎞ ⎛x - y⎞      \n\
      ⎜───────⎟⋅⎜─────⎟      \n\
      ⎝x - 2⋅y⎠ ⎝x + y⎠      \n\
─────────────────────────────\n\
1   ⎛ x + y ⎞ ⎛x - y⎞ ⎛1 - x⎞\n\
─ + ⎜───────⎟⋅⎜─────⎟⋅⎜─────⎟\n\
1   ⎝x - 2⋅y⎠ ⎝x + y⎠ ⎝x - y⎠\
"""
    expected6 = \
"""\
           ⎛ 2          ⎞                   \n\
           ⎜y  - 2⋅y + 1⎟ ⎛1 - x⎞           \n\
           ⎜────────────⎟⋅⎜─────⎟           \n\
           ⎝   y + 5    ⎠ ⎝x - y⎠           \n\
────────────────────────────────────────────\n\
    ⎛ 2          ⎞                          \n\
1   ⎜y  - 2⋅y + 1⎟ ⎛1 - x⎞ ⎛x - y⎞ ⎛ x + y ⎞\n\
─ + ⎜────────────⎟⋅⎜─────⎟⋅⎜─────⎟⋅⎜───────⎟\n\
1   ⎝   y + 5    ⎠ ⎝x - y⎠ ⎝x + y⎠ ⎝x - 2⋅y⎠\
"""
    expected7 = \
"""\
    ⎛       3⎞    \n\
    ⎜x - 2⋅y ⎟    \n\
    ⎜────────⎟    \n\
    ⎝ x + y  ⎠    \n\
──────────────────\n\
    ⎛       3⎞    \n\
1   ⎜x - 2⋅y ⎟ ⎛2⎞\n\
─ + ⎜────────⎟⋅⎜─⎟\n\
1   ⎝ x + y  ⎠ ⎝2⎠\
"""
    expected8 = \
"""\
  ⎛1 - x⎞  \n\
  ⎜─────⎟  \n\
  ⎝x - y⎠  \n\
───────────\n\
1   ⎛1 - x⎞\n\
─ + ⎜─────⎟\n\
1   ⎝x - y⎠\
"""
    expected9 = \
"""\
      ⎛ x + y ⎞ ⎛x - y⎞      \n\
      ⎜───────⎟⋅⎜─────⎟      \n\
      ⎝x - 2⋅y⎠ ⎝x + y⎠      \n\
─────────────────────────────\n\
1   ⎛ x + y ⎞ ⎛x - y⎞ ⎛1 - x⎞\n\
─ - ⎜───────⎟⋅⎜─────⎟⋅⎜─────⎟\n\
1   ⎝x - 2⋅y⎠ ⎝x + y⎠ ⎝x - y⎠\
"""
    expected10 = \
"""\
  ⎛1 - x⎞  \n\
  ⎜─────⎟  \n\
  ⎝x - y⎠  \n\
───────────\n\
1   ⎛1 - x⎞\n\
─ - ⎜─────⎟\n\
1   ⎝x - y⎠\
"""
    assert upretty(Feedback(tf, tf1)) == expected1
    assert upretty(Feedback(tf, tf2*tf1*tf3)) == expected2
    assert upretty(Feedback(tf1, tf2*tf3*tf5)) == expected3
    assert upretty(Feedback(tf1*tf2, tf)) == expected4
    assert upretty(Feedback(tf1*tf2, tf5)) == expected5
    assert upretty(Feedback(tf3*tf5, tf2*tf1)) == expected6
    assert upretty(Feedback(tf4, tf6)) == expected7
    assert upretty(Feedback(tf5, tf)) == expected8

    assert upretty(Feedback(tf1*tf2, tf5, 1)) == expected9
    assert upretty(Feedback(tf5, tf, 1)) == expected10


def test_pretty_MIMOFeedback():
    tf1 = TransferFunction(x + y, x - 2*y, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
    tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
    tfm_3 = TransferFunctionMatrix([[tf1, tf1], [tf2, tf2]])

    expected1 = \
"""\
⎛    ⎡ x + y    x - y ⎤  ⎡ x - y    x + y ⎤ ⎞-1   ⎡ x + y    x - y ⎤ \n\
⎜    ⎢───────   ───── ⎥  ⎢ ─────   ───────⎥ ⎟     ⎢───────   ───── ⎥ \n\
⎜    ⎢x - 2⋅y   x + y ⎥  ⎢ x + y   x - 2⋅y⎥ ⎟     ⎢x - 2⋅y   x + y ⎥ \n\
⎜I - ⎢                ⎥ ⋅⎢                ⎥ ⎟   ⋅ ⎢                ⎥ \n\
⎜    ⎢ x - y    x + y ⎥  ⎢ x + y    x - y ⎥ ⎟     ⎢ x - y    x + y ⎥ \n\
⎜    ⎢ ─────   ───────⎥  ⎢───────   ───── ⎥ ⎟     ⎢ ─────   ───────⎥ \n\
⎝    ⎣ x + y   x - 2⋅y⎦τ ⎣x - 2⋅y   x + y ⎦τ⎠     ⎣ x + y   x - 2⋅y⎦τ\
"""
    expected2 = \
"""\
⎛    ⎡ x + y    x - y ⎤  ⎡ x - y    x + y ⎤  ⎡ x + y    x + y ⎤ ⎞-1   ⎡ x + y    x - y ⎤  ⎡ x - y    x + y ⎤ \n\
⎜    ⎢───────   ───── ⎥  ⎢ ─────   ───────⎥  ⎢───────  ───────⎥ ⎟     ⎢───────   ───── ⎥  ⎢ ─────   ───────⎥ \n\
⎜    ⎢x - 2⋅y   x + y ⎥  ⎢ x + y   x - 2⋅y⎥  ⎢x - 2⋅y  x - 2⋅y⎥ ⎟     ⎢x - 2⋅y   x + y ⎥  ⎢ x + y   x - 2⋅y⎥ \n\
⎜I + ⎢                ⎥ ⋅⎢                ⎥ ⋅⎢                ⎥ ⎟   ⋅ ⎢                ⎥ ⋅⎢                ⎥ \n\
⎜    ⎢ x - y    x + y ⎥  ⎢ x + y    x - y ⎥  ⎢ x - y    x - y ⎥ ⎟     ⎢ x - y    x + y ⎥  ⎢ x + y    x - y ⎥ \n\
⎜    ⎢ ─────   ───────⎥  ⎢───────   ───── ⎥  ⎢ ─────    ───── ⎥ ⎟     ⎢ ─────   ───────⎥  ⎢───────   ───── ⎥ \n\
⎝    ⎣ x + y   x - 2⋅y⎦τ ⎣x - 2⋅y   x + y ⎦τ ⎣ x + y    x + y ⎦τ⎠     ⎣ x + y   x - 2⋅y⎦τ ⎣x - 2⋅y   x + y ⎦τ\
"""

    assert upretty(MIMOFeedback(tfm_1, tfm_2, 1)) == \
        expected1  # Positive MIMOFeedback
    assert upretty(MIMOFeedback(tfm_1*tfm_2, tfm_3)) == \
        expected2  # Negative MIMOFeedback (Default)


def test_pretty_TransferFunctionMatrix():
    tf1 = TransferFunction(x + y, x - 2*y, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(y**2 - 2*y + 1, y + 5, y)
    tf4 = TransferFunction(y, x**2 + x + 1, y)
    tf5 = TransferFunction(1 - x, x - y, y)
    tf6 = TransferFunction(2, 2, y)
    expected1 = \
"""\
⎡ x + y ⎤ \n\
⎢───────⎥ \n\
⎢x - 2⋅y⎥ \n\
⎢       ⎥ \n\
⎢ x - y ⎥ \n\
⎢ ───── ⎥ \n\
⎣ x + y ⎦τ\
"""
    expected2 = \
"""\
⎡    x + y     ⎤ \n\
⎢   ───────    ⎥ \n\
⎢   x - 2⋅y    ⎥ \n\
⎢              ⎥ \n\
⎢    x - y     ⎥ \n\
⎢    ─────     ⎥ \n\
⎢    x + y     ⎥ \n\
⎢              ⎥ \n\
⎢   2          ⎥ \n\
⎢- y  + 2⋅y - 1⎥ \n\
⎢──────────────⎥ \n\
⎣    y + 5     ⎦τ\
"""
    expected3 = \
"""\
⎡   x + y        x - y   ⎤ \n\
⎢  ───────       ─────   ⎥ \n\
⎢  x - 2⋅y       x + y   ⎥ \n\
⎢                        ⎥ \n\
⎢ 2                      ⎥ \n\
⎢y  - 2⋅y + 1      y     ⎥ \n\
⎢────────────  ──────────⎥ \n\
⎢   y + 5       2        ⎥ \n\
⎢              x  + x + 1⎥ \n\
⎢                        ⎥ \n\
⎢   1 - x          2     ⎥ \n\
⎢   ─────          ─     ⎥ \n\
⎣   x - y          2     ⎦τ\
"""
    expected4 = \
"""\
⎡    x - y        x + y       y     ⎤ \n\
⎢    ─────       ───────  ──────────⎥ \n\
⎢    x + y       x - 2⋅y   2        ⎥ \n\
⎢                         x  + x + 1⎥ \n\
⎢                                   ⎥ \n\
⎢   2                               ⎥ \n\
⎢- y  + 2⋅y - 1   x - 1      -2     ⎥ \n\
⎢──────────────   ─────      ───    ⎥ \n\
⎣    y + 5        x - y       2     ⎦τ\
"""
    expected5 = \
"""\
⎡ x + y  x - y   x + y       y     ⎤ \n\
⎢───────⋅─────  ───────  ──────────⎥ \n\
⎢x - 2⋅y x + y  x - 2⋅y   2        ⎥ \n\
⎢                        x  + x + 1⎥ \n\
⎢                                  ⎥ \n\
⎢  1 - x   2     x + y      -2     ⎥ \n\
⎢  ───── + ─    ───────     ───    ⎥ \n\
⎣  x - y   2    x - 2⋅y      2     ⎦τ\
"""

    assert upretty(TransferFunctionMatrix([[tf1], [tf2]])) == expected1
    assert upretty(TransferFunctionMatrix([[tf1], [tf2], [-tf3]])) == expected2
    assert upretty(TransferFunctionMatrix([[tf1, tf2], [tf3, tf4], [tf5, tf6]])) == expected3
    assert upretty(TransferFunctionMatrix([[tf2, tf1, tf4], [-tf3, -tf5, -tf6]])) == expected4
    assert upretty(TransferFunctionMatrix([[Series(tf2, tf1), tf1, tf4], [Parallel(tf6, tf5), tf1, -tf6]])) == \
        expected5


def test_pretty_StateSpace():
    ss1 = StateSpace(Matrix([a]), Matrix([b]), Matrix([c]), Matrix([d]))
    A = Matrix([[0, 1], [1, 0]])
    B = Matrix([1, 0])
    C = Matrix([[0, 1]])
    D = Matrix([0])
    ss2 = StateSpace(A, B, C, D)
    ss3 = StateSpace(Matrix([[-1.5, -2], [1, 0]]),
                    Matrix([[0.5, 0], [0, 1]]),
                    Matrix([[0, 1], [0, 2]]),
                    Matrix([[2, 2], [1, 1]]))

    expected1 = \
"""\
⎡[a]  [b]⎤\n\
⎢        ⎥\n\
⎣[c]  [d]⎦\
"""
    expected2 = \
"""\
⎡⎡0  1⎤  ⎡1⎤⎤\n\
⎢⎢    ⎥  ⎢ ⎥⎥\n\
⎢⎣1  0⎦  ⎣0⎦⎥\n\
⎢           ⎥\n\
⎣[0  1]  [0]⎦\
"""
    expected3 = \
"""\
⎡⎡-1.5  -2⎤  ⎡0.5  0⎤⎤\n\
⎢⎢        ⎥  ⎢      ⎥⎥\n\
⎢⎣ 1    0 ⎦  ⎣ 0   1⎦⎥\n\
⎢                    ⎥\n\
⎢  ⎡0  1⎤     ⎡2  2⎤ ⎥\n\
⎢  ⎢    ⎥     ⎢    ⎥ ⎥\n\
⎣  ⎣0  2⎦     ⎣1  1⎦ ⎦\
"""

    assert upretty(ss1) == expected1
    assert upretty(ss2) == expected2
    assert upretty(ss3) == expected3

def test_pretty_order():
    expr = O(1)
    ascii_str = \
"""\
O(1)\
"""
    ucode_str = \
"""\
O(1)\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = O(1/x)
    ascii_str = \
"""\
 /1\\\n\
O|-|\n\
 \\x/\
"""
    ucode_str = \
"""\
 ⎛1⎞\n\
O⎜─⎟\n\
 ⎝x⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = O(x**2 + y**2)
    ascii_str = \
"""\
 / 2    2                  \\\n\
O\\x  + y ; (x, y) -> (0, 0)/\
"""
    ucode_str = \
"""\
 ⎛ 2    2                 ⎞\n\
O⎝x  + y ; (x, y) → (0, 0)⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = O(1, (x, oo))
    ascii_str = \
"""\
O(1; x -> oo)\
"""
    ucode_str = \
"""\
O(1; x → ∞)\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = O(1/x, (x, oo))
    ascii_str = \
"""\
 /1         \\\n\
O|-; x -> oo|\n\
 \\x         /\
"""
    ucode_str = \
"""\
 ⎛1       ⎞\n\
O⎜─; x → ∞⎟\n\
 ⎝x       ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = O(x**2 + y**2, (x, oo), (y, oo))
    ascii_str = \
"""\
 / 2    2                    \\\n\
O\\x  + y ; (x, y) -> (oo, oo)/\
"""
    ucode_str = \
"""\
 ⎛ 2    2                 ⎞\n\
O⎝x  + y ; (x, y) → (∞, ∞)⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_derivatives():
    # Simple
    expr = Derivative(log(x), x, evaluate=False)
    ascii_str = \
"""\
d         \n\
--(log(x))\n\
dx        \
"""
    ucode_str = \
"""\
d         \n\
──(log(x))\n\
dx        \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Derivative(log(x), x, evaluate=False) + x
    ascii_str_1 = \
"""\
    d         \n\
x + --(log(x))\n\
    dx        \
"""
    ascii_str_2 = \
"""\
d             \n\
--(log(x)) + x\n\
dx            \
"""
    ucode_str_1 = \
"""\
    d         \n\
x + ──(log(x))\n\
    dx        \
"""
    ucode_str_2 = \
"""\
d             \n\
──(log(x)) + x\n\
dx            \
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    # basic partial derivatives
    expr = Derivative(log(x + y) + x, x)
    ascii_str_1 = \
"""\
d                 \n\
--(log(x + y) + x)\n\
dx                \
"""
    ascii_str_2 = \
"""\
d                 \n\
--(x + log(x + y))\n\
dx                \
"""
    ucode_str_1 = \
"""\
∂                 \n\
──(log(x + y) + x)\n\
∂x                \
"""
    ucode_str_2 = \
"""\
∂                 \n\
──(x + log(x + y))\n\
∂x                \
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2], upretty(expr)

    # Multiple symbols
    expr = Derivative(log(x) + x**2, x, y)
    ascii_str_1 = \
"""\
   2              \n\
  d  /          2\\\n\
-----\\log(x) + x /\n\
dy dx             \
"""
    ascii_str_2 = \
"""\
   2              \n\
  d  / 2         \\\n\
-----\\x  + log(x)/\n\
dy dx             \
"""
    ascii_str_3 = \
"""\
  2               \n\
 d   / 2         \\\n\
-----\\x  + log(x)/\n\
dy dx             \
"""
    ucode_str_1 = \
"""\
   2              \n\
  d  ⎛          2⎞\n\
─────⎝log(x) + x ⎠\n\
dy dx             \
"""
    ucode_str_2 = \
"""\
   2              \n\
  d  ⎛ 2         ⎞\n\
─────⎝x  + log(x)⎠\n\
dy dx             \
"""
    ucode_str_3 = \
"""\
  2               \n\
 d   ⎛ 2         ⎞\n\
─────⎝x  + log(x)⎠\n\
dy dx             \
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2, ascii_str_3]
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]

    expr = Derivative(2*x*y, y, x) + x**2
    ascii_str_1 = \
"""\
   2             \n\
  d             2\n\
-----(2*x*y) + x \n\
dx dy            \
"""
    ascii_str_2 = \
"""\
        2        \n\
 2     d         \n\
x  + -----(2*x*y)\n\
     dx dy       \
"""
    ascii_str_3 = \
"""\
       2         \n\
 2    d          \n\
x  + -----(2*x*y)\n\
     dx dy       \
"""
    ucode_str_1 = \
"""\
   2             \n\
  ∂             2\n\
─────(2⋅x⋅y) + x \n\
∂x ∂y            \
"""
    ucode_str_2 = \
"""\
        2        \n\
 2     ∂         \n\
x  + ─────(2⋅x⋅y)\n\
     ∂x ∂y       \
"""
    ucode_str_3 = \
"""\
       2         \n\
 2    ∂          \n\
x  + ─────(2⋅x⋅y)\n\
     ∂x ∂y       \
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2, ascii_str_3]
    assert upretty(expr) in [ucode_str_1, ucode_str_2, ucode_str_3]

    expr = Derivative(2*x*y, x, x)
    ascii_str = \
"""\
 2        \n\
d         \n\
---(2*x*y)\n\
  2       \n\
dx        \
"""
    ucode_str = \
"""\
 2        \n\
∂         \n\
───(2⋅x⋅y)\n\
  2       \n\
∂x        \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Derivative(2*x*y, x, 17)
    ascii_str = \
"""\
 17        \n\
d          \n\
----(2*x*y)\n\
  17       \n\
dx         \
"""
    ucode_str = \
"""\
 17        \n\
∂          \n\
────(2⋅x⋅y)\n\
  17       \n\
∂x         \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Derivative(2*x*y, x, x, y)
    ascii_str = \
"""\
   3         \n\
  d          \n\
------(2*x*y)\n\
     2       \n\
dy dx        \
"""
    ucode_str = \
"""\
   3         \n\
  ∂          \n\
──────(2⋅x⋅y)\n\
     2       \n\
∂y ∂x        \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # Greek letters
    alpha = Symbol('alpha')
    beta = Function('beta')
    expr = beta(alpha).diff(alpha)
    ascii_str = \
"""\
  d                \n\
------(beta(alpha))\n\
dalpha             \
"""
    ucode_str = \
"""\
d       \n\
──(β(α))\n\
dα      \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Derivative(f(x), (x, n))

    ascii_str = \
"""\
 n       \n\
d        \n\
---(f(x))\n\
  n      \n\
dx       \
"""
    ucode_str = \
"""\
 n       \n\
d        \n\
───(f(x))\n\
  n      \n\
dx       \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_integrals():
    expr = Integral(log(x), x)
    ascii_str = \
"""\
  /         \n\
 |          \n\
 | log(x) dx\n\
 |          \n\
/           \
"""
    ucode_str = \
"""\
⌠          \n\
⎮ log(x) dx\n\
⌡          \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Integral(x**2, x)
    ascii_str = \
"""\
  /     \n\
 |      \n\
 |  2   \n\
 | x  dx\n\
 |      \n\
/       \
"""
    ucode_str = \
"""\
⌠      \n\
⎮  2   \n\
⎮ x  dx\n\
⌡      \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Integral((sin(x))**2 / (tan(x))**2)
    ascii_str = \
"""\
  /          \n\
 |           \n\
 |    2      \n\
 | sin (x)   \n\
 | ------- dx\n\
 |    2      \n\
 | tan (x)   \n\
 |           \n\
/            \
"""
    ucode_str = \
"""\
⌠           \n\
⎮    2      \n\
⎮ sin (x)   \n\
⎮ ─────── dx\n\
⎮    2      \n\
⎮ tan (x)   \n\
⌡           \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Integral(x**(2**x), x)
    ascii_str = \
"""\
  /        \n\
 |         \n\
 |  / x\\   \n\
 |  \\2 /   \n\
 | x     dx\n\
 |         \n\
/          \
"""
    ucode_str = \
"""\
⌠         \n\
⎮  ⎛ x⎞   \n\
⎮  ⎝2 ⎠   \n\
⎮ x     dx\n\
⌡         \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Integral(x**2, (x, 1, 2))
    ascii_str = \
"""\
  2      \n\
  /      \n\
 |       \n\
 |   2   \n\
 |  x  dx\n\
 |       \n\
/        \n\
1        \
"""
    ucode_str = \
"""\
2      \n\
⌠      \n\
⎮  2   \n\
⎮ x  dx\n\
⌡      \n\
1      \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Integral(x**2, (x, Rational(1, 2), 10))
    ascii_str = \
"""\
 10      \n\
  /      \n\
 |       \n\
 |   2   \n\
 |  x  dx\n\
 |       \n\
/        \n\
1/2      \
"""
    ucode_str = \
"""\
10       \n\
⌠        \n\
⎮    2   \n\
⎮   x  dx\n\
⌡        \n\
1/2      \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Integral(x**2*y**2, x, y)
    ascii_str = \
"""\
  /  /           \n\
 |  |            \n\
 |  |  2  2      \n\
 |  | x *y  dx dy\n\
 |  |            \n\
/  /             \
"""
    ucode_str = \
"""\
⌠ ⌠            \n\
⎮ ⎮  2  2      \n\
⎮ ⎮ x ⋅y  dx dy\n\
⌡ ⌡            \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Integral(sin(th)/cos(ph), (th, 0, pi), (ph, 0, 2*pi))
    ascii_str = \
"""\
 2*pi pi                           \n\
   /   /                           \n\
  |   |                            \n\
  |   |  sin(theta)                \n\
  |   |  ---------- d(theta) d(phi)\n\
  |   |   cos(phi)                 \n\
  |   |                            \n\
 /   /                             \n\
0    0                             \
"""
    ucode_str = \
"""\
2⋅π π             \n\
 ⌠  ⌠             \n\
 ⎮  ⎮ sin(θ)      \n\
 ⎮  ⎮ ────── dθ dφ\n\
 ⎮  ⎮ cos(φ)      \n\
 ⌡  ⌡             \n\
 0  0             \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_matrix():
    # Empty Matrix
    expr = Matrix()
    ascii_str = "[]"
    unicode_str = "[]"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str
    expr = Matrix(2, 0, lambda i, j: 0)
    ascii_str = "[]"
    unicode_str = "[]"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str
    expr = Matrix(0, 2, lambda i, j: 0)
    ascii_str = "[]"
    unicode_str = "[]"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str
    expr = Matrix([[x**2 + 1, 1], [y, x + y]])
    ascii_str_1 = \
"""\
[     2       ]
[1 + x     1  ]
[             ]
[  y     x + y]\
"""
    ascii_str_2 = \
"""\
[ 2           ]
[x  + 1    1  ]
[             ]
[  y     x + y]\
"""
    ucode_str_1 = \
"""\
⎡     2       ⎤
⎢1 + x     1  ⎥
⎢             ⎥
⎣  y     x + y⎦\
"""
    ucode_str_2 = \
"""\
⎡ 2           ⎤
⎢x  + 1    1  ⎥
⎢             ⎥
⎣  y     x + y⎦\
"""
    assert pretty(expr) in [ascii_str_1, ascii_str_2]
    assert upretty(expr) in [ucode_str_1, ucode_str_2]

    expr = Matrix([[x/y, y, th], [0, exp(I*k*ph), 1]])
    ascii_str = \
"""\
[x                 ]
[-     y      theta]
[y                 ]
[                  ]
[    I*k*phi       ]
[0  e           1  ]\
"""
    ucode_str = \
"""\
⎡x           ⎤
⎢─    y     θ⎥
⎢y           ⎥
⎢            ⎥
⎢    ⅈ⋅k⋅φ   ⎥
⎣0  ℯ       1⎦\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    unicode_str = \
"""\
⎡v̇_msc_00     0         0    ⎤
⎢                            ⎥
⎢   0      v̇_msc_01     0    ⎥
⎢                            ⎥
⎣   0         0      v̇_msc_02⎦\
"""

    expr = diag(*MatrixSymbol('vdot_msc',1,3))
    assert upretty(expr) == unicode_str


def test_pretty_ndim_arrays():
    x, y, z, w = symbols("x y z w")

    for ArrayType in (ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableDenseNDimArray, MutableSparseNDimArray):
        # Basic: scalar array
        M = ArrayType(x)

        assert pretty(M) == "x"
        assert upretty(M) == "x"

        M = ArrayType([[1/x, y], [z, w]])
        M1 = ArrayType([1/x, y, z])

        M2 = tensorproduct(M1, M)
        M3 = tensorproduct(M, M)

        ascii_str = \
"""\
[1   ]\n\
[-  y]\n\
[x   ]\n\
[    ]\n\
[z  w]\
"""
        ucode_str = \
"""\
⎡1   ⎤\n\
⎢─  y⎥\n\
⎢x   ⎥\n\
⎢    ⎥\n\
⎣z  w⎦\
"""
        assert pretty(M) == ascii_str
        assert upretty(M) == ucode_str

        ascii_str = \
"""\
[1      ]\n\
[-  y  z]\n\
[x      ]\
"""
        ucode_str = \
"""\
⎡1      ⎤\n\
⎢─  y  z⎥\n\
⎣x      ⎦\
"""
        assert pretty(M1) == ascii_str
        assert upretty(M1) == ucode_str

        ascii_str = \
"""\
[[1   y]                       ]\n\
[[--  -]              [z      ]]\n\
[[ 2  x]  [ y    2 ]  [-   y*z]]\n\
[[x    ]  [ -   y  ]  [x      ]]\n\
[[     ]  [ x      ]  [       ]]\n\
[[z   w]  [        ]  [ 2     ]]\n\
[[-   -]  [y*z  w*y]  [z   w*z]]\n\
[[x   x]                       ]\
"""
        ucode_str = \
"""\
⎡⎡1   y⎤                       ⎤\n\
⎢⎢──  ─⎥              ⎡z      ⎤⎥\n\
⎢⎢ 2  x⎥  ⎡ y    2 ⎤  ⎢─   y⋅z⎥⎥\n\
⎢⎢x    ⎥  ⎢ ─   y  ⎥  ⎢x      ⎥⎥\n\
⎢⎢     ⎥  ⎢ x      ⎥  ⎢       ⎥⎥\n\
⎢⎢z   w⎥  ⎢        ⎥  ⎢ 2     ⎥⎥\n\
⎢⎢─   ─⎥  ⎣y⋅z  w⋅y⎦  ⎣z   w⋅z⎦⎥\n\
⎣⎣x   x⎦                       ⎦\
"""
        assert pretty(M2) == ascii_str
        assert upretty(M2) == ucode_str

        ascii_str = \
"""\
[ [1   y]             ]\n\
[ [--  -]             ]\n\
[ [ 2  x]   [ y    2 ]]\n\
[ [x    ]   [ -   y  ]]\n\
[ [     ]   [ x      ]]\n\
[ [z   w]   [        ]]\n\
[ [-   -]   [y*z  w*y]]\n\
[ [x   x]             ]\n\
[                     ]\n\
[[z      ]  [ w      ]]\n\
[[-   y*z]  [ -   w*y]]\n\
[[x      ]  [ x      ]]\n\
[[       ]  [        ]]\n\
[[ 2     ]  [      2 ]]\n\
[[z   w*z]  [w*z  w  ]]\
"""
        ucode_str = \
"""\
⎡ ⎡1   y⎤             ⎤\n\
⎢ ⎢──  ─⎥             ⎥\n\
⎢ ⎢ 2  x⎥   ⎡ y    2 ⎤⎥\n\
⎢ ⎢x    ⎥   ⎢ ─   y  ⎥⎥\n\
⎢ ⎢     ⎥   ⎢ x      ⎥⎥\n\
⎢ ⎢z   w⎥   ⎢        ⎥⎥\n\
⎢ ⎢─   ─⎥   ⎣y⋅z  w⋅y⎦⎥\n\
⎢ ⎣x   x⎦             ⎥\n\
⎢                     ⎥\n\
⎢⎡z      ⎤  ⎡ w      ⎤⎥\n\
⎢⎢─   y⋅z⎥  ⎢ ─   w⋅y⎥⎥\n\
⎢⎢x      ⎥  ⎢ x      ⎥⎥\n\
⎢⎢       ⎥  ⎢        ⎥⎥\n\
⎢⎢ 2     ⎥  ⎢      2 ⎥⎥\n\
⎣⎣z   w⋅z⎦  ⎣w⋅z  w  ⎦⎦\
"""
        assert pretty(M3) == ascii_str
        assert upretty(M3) == ucode_str

        Mrow = ArrayType([[x, y, 1 / z]])
        Mcolumn = ArrayType([[x], [y], [1 / z]])
        Mcol2 = ArrayType([Mcolumn.tolist()])

        ascii_str = \
"""\
[[      1]]\n\
[[x  y  -]]\n\
[[      z]]\
"""
        ucode_str = \
"""\
⎡⎡      1⎤⎤\n\
⎢⎢x  y  ─⎥⎥\n\
⎣⎣      z⎦⎦\
"""
        assert pretty(Mrow) == ascii_str
        assert upretty(Mrow) == ucode_str

        ascii_str = \
"""\
[x]\n\
[ ]\n\
[y]\n\
[ ]\n\
[1]\n\
[-]\n\
[z]\
"""
        ucode_str = \
"""\
⎡x⎤\n\
⎢ ⎥\n\
⎢y⎥\n\
⎢ ⎥\n\
⎢1⎥\n\
⎢─⎥\n\
⎣z⎦\
"""
        assert pretty(Mcolumn) == ascii_str
        assert upretty(Mcolumn) == ucode_str

        ascii_str = \
"""\
[[x]]\n\
[[ ]]\n\
[[y]]\n\
[[ ]]\n\
[[1]]\n\
[[-]]\n\
[[z]]\
"""
        ucode_str = \
"""\
⎡⎡x⎤⎤\n\
⎢⎢ ⎥⎥\n\
⎢⎢y⎥⎥\n\
⎢⎢ ⎥⎥\n\
⎢⎢1⎥⎥\n\
⎢⎢─⎥⎥\n\
⎣⎣z⎦⎦\
"""
        assert pretty(Mcol2) == ascii_str
        assert upretty(Mcol2) == ucode_str


def test_tensor_TensorProduct():
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    assert upretty(TensorProduct(A, B)) == "A\u2297B"
    assert upretty(TensorProduct(A, B, A)) == "A\u2297B\u2297A"


def test_diffgeom_print_WedgeProduct():
    from sympy.diffgeom.rn import R2
    from sympy.diffgeom import WedgeProduct
    wp = WedgeProduct(R2.dx, R2.dy)
    assert upretty(wp) == "ⅆ x∧ⅆ y"
    assert pretty(wp) == r"d x/\d y"


def test_Adjoint():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert pretty(Adjoint(X)) == " +\nX "
    assert pretty(Adjoint(X + Y)) == "       +\n(X + Y) "
    assert pretty(Adjoint(X) + Adjoint(Y)) == " +    +\nX  + Y "
    assert pretty(Adjoint(X*Y)) == "     +\n(X*Y) "
    assert pretty(Adjoint(Y)*Adjoint(X)) == " +  +\nY *X "
    assert pretty(Adjoint(X**2)) == "    +\n/ 2\\ \n\\X / "
    assert pretty(Adjoint(X)**2) == "    2\n/ +\\ \n\\X / "
    assert pretty(Adjoint(Inverse(X))) == "     +\n/ -1\\ \n\\X  / "
    assert pretty(Inverse(Adjoint(X))) == "    -1\n/ +\\  \n\\X /  "
    assert pretty(Adjoint(Transpose(X))) == "    +\n/ T\\ \n\\X / "
    assert pretty(Transpose(Adjoint(X))) == "    T\n/ +\\ \n\\X / "
    assert upretty(Adjoint(X)) == " †\nX "
    assert upretty(Adjoint(X + Y)) == "       †\n(X + Y) "
    assert upretty(Adjoint(X) + Adjoint(Y)) == " †    †\nX  + Y "
    assert upretty(Adjoint(X*Y)) == "     †\n(X⋅Y) "
    assert upretty(Adjoint(Y)*Adjoint(X)) == " †  †\nY ⋅X "
    assert upretty(Adjoint(X**2)) == \
        "    †\n⎛ 2⎞ \n⎝X ⎠ "
    assert upretty(Adjoint(X)**2) == \
        "    2\n⎛ †⎞ \n⎝X ⎠ "
    assert upretty(Adjoint(Inverse(X))) == \
        "     †\n⎛ -1⎞ \n⎝X  ⎠ "
    assert upretty(Inverse(Adjoint(X))) == \
        "    -1\n⎛ †⎞  \n⎝X ⎠  "
    assert upretty(Adjoint(Transpose(X))) == \
        "    †\n⎛ T⎞ \n⎝X ⎠ "
    assert upretty(Transpose(Adjoint(X))) == \
        "    T\n⎛ †⎞ \n⎝X ⎠ "
    m = Matrix(((1, 2), (3, 4)))
    assert upretty(Adjoint(m)) == \
        '      †\n'\
        '⎡1  2⎤ \n'\
        '⎢    ⎥ \n'\
        '⎣3  4⎦ '
    assert upretty(Adjoint(m+X)) == \
        '            †\n'\
        '⎛⎡1  2⎤    ⎞ \n'\
        '⎜⎢    ⎥ + X⎟ \n'\
        '⎝⎣3  4⎦    ⎠ '
    assert upretty(Adjoint(BlockMatrix(((OneMatrix(2, 2), X),
                                        (m, ZeroMatrix(2, 2)))))) == \
        '           †\n'\
        '⎡  𝟙     X⎤ \n'\
        '⎢         ⎥ \n'\
        '⎢⎡1  2⎤   ⎥ \n'\
        '⎢⎢    ⎥  𝟘⎥ \n'\
        '⎣⎣3  4⎦   ⎦ '


def test_Transpose():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert pretty(Transpose(X)) == " T\nX "
    assert pretty(Transpose(X + Y)) == "       T\n(X + Y) "
    assert pretty(Transpose(X) + Transpose(Y)) == " T    T\nX  + Y "
    assert pretty(Transpose(X*Y)) == "     T\n(X*Y) "
    assert pretty(Transpose(Y)*Transpose(X)) == " T  T\nY *X "
    assert pretty(Transpose(X**2)) == "    T\n/ 2\\ \n\\X / "
    assert pretty(Transpose(X)**2) == "    2\n/ T\\ \n\\X / "
    assert pretty(Transpose(Inverse(X))) == "     T\n/ -1\\ \n\\X  / "
    assert pretty(Inverse(Transpose(X))) == "    -1\n/ T\\  \n\\X /  "
    assert upretty(Transpose(X)) == " T\nX "
    assert upretty(Transpose(X + Y)) == "       T\n(X + Y) "
    assert upretty(Transpose(X) + Transpose(Y)) == " T    T\nX  + Y "
    assert upretty(Transpose(X*Y)) == "     T\n(X⋅Y) "
    assert upretty(Transpose(Y)*Transpose(X)) == " T  T\nY ⋅X "
    assert upretty(Transpose(X**2)) == \
        "    T\n⎛ 2⎞ \n⎝X ⎠ "
    assert upretty(Transpose(X)**2) == \
        "    2\n⎛ T⎞ \n⎝X ⎠ "
    assert upretty(Transpose(Inverse(X))) == \
        "     T\n⎛ -1⎞ \n⎝X  ⎠ "
    assert upretty(Inverse(Transpose(X))) == \
        "    -1\n⎛ T⎞  \n⎝X ⎠  "
    m = Matrix(((1, 2), (3, 4)))
    assert upretty(Transpose(m)) == \
        '      T\n'\
        '⎡1  2⎤ \n'\
        '⎢    ⎥ \n'\
        '⎣3  4⎦ '
    assert upretty(Transpose(m+X)) == \
        '            T\n'\
        '⎛⎡1  2⎤    ⎞ \n'\
        '⎜⎢    ⎥ + X⎟ \n'\
        '⎝⎣3  4⎦    ⎠ '
    assert upretty(Transpose(BlockMatrix(((OneMatrix(2, 2), X),
                                          (m, ZeroMatrix(2, 2)))))) == \
        '           T\n'\
        '⎡  𝟙     X⎤ \n'\
        '⎢         ⎥ \n'\
        '⎢⎡1  2⎤   ⎥ \n'\
        '⎢⎢    ⎥  𝟘⎥ \n'\
        '⎣⎣3  4⎦   ⎦ '


def test_pretty_Trace_issue_9044():
    X = Matrix([[1, 2], [3, 4]])
    Y = Matrix([[2, 4], [6, 8]])
    ascii_str_1 = \
"""\
  /[1  2]\\
tr|[    ]|
  \\[3  4]/\
"""
    ucode_str_1 = \
"""\
  ⎛⎡1  2⎤⎞
tr⎜⎢    ⎥⎟
  ⎝⎣3  4⎦⎠\
"""
    ascii_str_2 = \
"""\
  /[1  2]\\     /[2  4]\\
tr|[    ]| + tr|[    ]|
  \\[3  4]/     \\[6  8]/\
"""
    ucode_str_2 = \
"""\
  ⎛⎡1  2⎤⎞     ⎛⎡2  4⎤⎞
tr⎜⎢    ⎥⎟ + tr⎜⎢    ⎥⎟
  ⎝⎣3  4⎦⎠     ⎝⎣6  8⎦⎠\
"""
    assert pretty(Trace(X)) == ascii_str_1
    assert upretty(Trace(X)) == ucode_str_1

    assert pretty(Trace(X) + Trace(Y)) == ascii_str_2
    assert upretty(Trace(X) + Trace(Y)) == ucode_str_2


def test_MatrixSlice():
    n = Symbol('n', integer=True)
    x, y, z, w, t, = symbols('x y z w t')
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 10, 10)
    Z = MatrixSymbol('Z', 10, 10)

    expr = MatrixSlice(X, (None, None, None), (None, None, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    expr = X[x:x + 1, y:y + 1]
    assert pretty(expr) == upretty(expr) == 'X[x:x + 1, y:y + 1]'
    expr = X[x:x + 1:2, y:y + 1:2]
    assert pretty(expr) == upretty(expr) == 'X[x:x + 1:2, y:y + 1:2]'
    expr = X[:x, y:]
    assert pretty(expr) == upretty(expr) == 'X[:x, y:]'
    expr = X[:x, y:]
    assert pretty(expr) == upretty(expr) == 'X[:x, y:]'
    expr = X[x:, :y]
    assert pretty(expr) == upretty(expr) == 'X[x:, :y]'
    expr = X[x:y, z:w]
    assert pretty(expr) == upretty(expr) == 'X[x:y, z:w]'
    expr = X[x:y:t, w:t:x]
    assert pretty(expr) == upretty(expr) == 'X[x:y:t, w:t:x]'
    expr = X[x::y, t::w]
    assert pretty(expr) == upretty(expr) == 'X[x::y, t::w]'
    expr = X[:x:y, :t:w]
    assert pretty(expr) == upretty(expr) == 'X[:x:y, :t:w]'
    expr = X[::x, ::y]
    assert pretty(expr) == upretty(expr) == 'X[::x, ::y]'
    expr = MatrixSlice(X, (0, None, None), (0, None, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    expr = MatrixSlice(X, (None, n, None), (None, n, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    expr = MatrixSlice(X, (0, n, None), (0, n, None))
    assert pretty(expr) == upretty(expr) == 'X[:, :]'
    expr = MatrixSlice(X, (0, n, 2), (0, n, 2))
    assert pretty(expr) == upretty(expr) == 'X[::2, ::2]'
    expr = X[1:2:3, 4:5:6]
    assert pretty(expr) == upretty(expr) == 'X[1:2:3, 4:5:6]'
    expr = X[1:3:5, 4:6:8]
    assert pretty(expr) == upretty(expr) == 'X[1:3:5, 4:6:8]'
    expr = X[1:10:2]
    assert pretty(expr) == upretty(expr) == 'X[1:10:2, :]'
    expr = Y[:5, 1:9:2]
    assert pretty(expr) == upretty(expr) == 'Y[:5, 1:9:2]'
    expr = Y[:5, 1:10:2]
    assert pretty(expr) == upretty(expr) == 'Y[:5, 1::2]'
    expr = Y[5, :5:2]
    assert pretty(expr) == upretty(expr) == 'Y[5:6, :5:2]'
    expr = X[0:1, 0:1]
    assert pretty(expr) == upretty(expr) == 'X[:1, :1]'
    expr = X[0:1:2, 0:1:2]
    assert pretty(expr) == upretty(expr) == 'X[:1:2, :1:2]'
    expr = (Y + Z)[2:, 2:]
    assert pretty(expr) == upretty(expr) == '(Y + Z)[2:, 2:]'


def test_MatrixExpressions():
    n = Symbol('n', integer=True)
    X = MatrixSymbol('X', n, n)

    assert pretty(X) == upretty(X) == "X"

    # Apply function elementwise (`ElementwiseApplyFunc`):

    expr = (X.T*X).applyfunc(sin)

    ascii_str = """\
              / T  \\\n\
(d -> sin(d)).\\X *X/\
"""
    ucode_str = """\
             ⎛ T  ⎞\n\
(d ↦ sin(d))˳⎝X ⋅X⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    lamda = Lambda(x, 1/x)
    expr = (n*X).applyfunc(lamda)
    ascii_str = """\
/     1\\      \n\
|x -> -|.(n*X)\n\
\\     x/      \
"""
    ucode_str = """\
⎛    1⎞      \n\
⎜x ↦ ─⎟˳(n⋅X)\n\
⎝    x⎠      \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_dotproduct():
    from sympy.matrices.expressions.dotproduct import DotProduct
    n = symbols("n", integer=True)
    A = MatrixSymbol('A', n, 1)
    B = MatrixSymbol('B', n, 1)
    C = Matrix(1, 3, [1, 2, 3])
    D = Matrix(1, 3, [1, 3, 4])

    assert pretty(DotProduct(A, B)) == "A*B"
    assert pretty(DotProduct(C, D)) == "[1  2  3]*[1  3  4]"
    assert upretty(DotProduct(A, B)) == "A⋅B"
    assert upretty(DotProduct(C, D)) == "[1  2  3]⋅[1  3  4]"


def test_pretty_Determinant():
    from sympy.matrices import Determinant, Inverse, BlockMatrix, OneMatrix, ZeroMatrix
    m = Matrix(((1, 2), (3, 4)))
    assert upretty(Determinant(m)) == '│1  2│\n│    │\n│3  4│'
    assert upretty(Determinant(Inverse(m))) == \
        '│      -1│\n'\
        '│⎡1  2⎤  │\n'\
        '│⎢    ⎥  │\n'\
        '│⎣3  4⎦  │'
    X = MatrixSymbol('X', 2, 2)
    assert upretty(Determinant(X)) == '│X│'
    assert upretty(Determinant(X + m)) == \
        '│⎡1  2⎤    │\n'\
        '│⎢    ⎥ + X│\n'\
        '│⎣3  4⎦    │'
    assert upretty(Determinant(BlockMatrix(((OneMatrix(2, 2), X),
                                            (m, ZeroMatrix(2, 2)))))) == \
        '│  𝟙     X│\n'\
        '│         │\n'\
        '│⎡1  2⎤   │\n'\
        '│⎢    ⎥  𝟘│\n'\
        '│⎣3  4⎦   │'


def test_pretty_piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    ascii_str = \
"""\
/x   for x < 1\n\
|             \n\
< 2           \n\
|x   otherwise\n\
\\             \
"""
    ucode_str = \
"""\
⎧x   for x < 1\n\
⎪             \n\
⎨ 2           \n\
⎪x   otherwise\n\
⎩             \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -Piecewise((x, x < 1), (x**2, True))
    ascii_str = \
"""\
 //x   for x < 1\\\n\
 ||             |\n\
-|< 2           |\n\
 ||x   otherwise|\n\
 \\\\             /\
"""
    ucode_str = \
"""\
 ⎛⎧x   for x < 1⎞\n\
 ⎜⎪             ⎟\n\
-⎜⎨ 2           ⎟\n\
 ⎜⎪x   otherwise⎟\n\
 ⎝⎩             ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = x + Piecewise((x, x > 0), (y, True)) + Piecewise((x/y, x < 2),
    (y**2, x > 2), (1, True)) + 1
    ascii_str = \
"""\
                      //x            \\    \n\
                      ||-   for x < 2|    \n\
                      ||y            |    \n\
    //x  for x > 0\\   ||             |    \n\
x + |<            | + |< 2           | + 1\n\
    \\\\y  otherwise/   ||y   for x > 2|    \n\
                      ||             |    \n\
                      ||1   otherwise|    \n\
                      \\\\             /    \
"""
    ucode_str = \
"""\
                      ⎛⎧x            ⎞    \n\
                      ⎜⎪─   for x < 2⎟    \n\
                      ⎜⎪y            ⎟    \n\
    ⎛⎧x  for x > 0⎞   ⎜⎪             ⎟    \n\
x + ⎜⎨            ⎟ + ⎜⎨ 2           ⎟ + 1\n\
    ⎝⎩y  otherwise⎠   ⎜⎪y   for x > 2⎟    \n\
                      ⎜⎪             ⎟    \n\
                      ⎜⎪1   otherwise⎟    \n\
                      ⎝⎩             ⎠    \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = x - Piecewise((x, x > 0), (y, True)) + Piecewise((x/y, x < 2),
    (y**2, x > 2), (1, True)) + 1
    ascii_str = \
"""\
                      //x            \\    \n\
                      ||-   for x < 2|    \n\
                      ||y            |    \n\
    //x  for x > 0\\   ||             |    \n\
x - |<            | + |< 2           | + 1\n\
    \\\\y  otherwise/   ||y   for x > 2|    \n\
                      ||             |    \n\
                      ||1   otherwise|    \n\
                      \\\\             /    \
"""
    ucode_str = \
"""\
                      ⎛⎧x            ⎞    \n\
                      ⎜⎪─   for x < 2⎟    \n\
                      ⎜⎪y            ⎟    \n\
    ⎛⎧x  for x > 0⎞   ⎜⎪             ⎟    \n\
x - ⎜⎨            ⎟ + ⎜⎨ 2           ⎟ + 1\n\
    ⎝⎩y  otherwise⎠   ⎜⎪y   for x > 2⎟    \n\
                      ⎜⎪             ⎟    \n\
                      ⎜⎪1   otherwise⎟    \n\
                      ⎝⎩             ⎠    \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = x*Piecewise((x, x > 0), (y, True))
    ascii_str = \
"""\
  //x  for x > 0\\\n\
x*|<            |\n\
  \\\\y  otherwise/\
"""
    ucode_str = \
"""\
  ⎛⎧x  for x > 0⎞\n\
x⋅⎜⎨            ⎟\n\
  ⎝⎩y  otherwise⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Piecewise((x, x > 0), (y, True))*Piecewise((x/y, x < 2), (y**2, x >
    2), (1, True))
    ascii_str = \
"""\
                //x            \\\n\
                ||-   for x < 2|\n\
                ||y            |\n\
//x  for x > 0\\ ||             |\n\
|<            |*|< 2           |\n\
\\\\y  otherwise/ ||y   for x > 2|\n\
                ||             |\n\
                ||1   otherwise|\n\
                \\\\             /\
"""
    ucode_str = \
"""\
                ⎛⎧x            ⎞\n\
                ⎜⎪─   for x < 2⎟\n\
                ⎜⎪y            ⎟\n\
⎛⎧x  for x > 0⎞ ⎜⎪             ⎟\n\
⎜⎨            ⎟⋅⎜⎨ 2           ⎟\n\
⎝⎩y  otherwise⎠ ⎜⎪y   for x > 2⎟\n\
                ⎜⎪             ⎟\n\
                ⎜⎪1   otherwise⎟\n\
                ⎝⎩             ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -Piecewise((x, x > 0), (y, True))*Piecewise((x/y, x < 2), (y**2, x
        > 2), (1, True))
    ascii_str = \
"""\
                 //x            \\\n\
                 ||-   for x < 2|\n\
                 ||y            |\n\
 //x  for x > 0\\ ||             |\n\
-|<            |*|< 2           |\n\
 \\\\y  otherwise/ ||y   for x > 2|\n\
                 ||             |\n\
                 ||1   otherwise|\n\
                 \\\\             /\
"""
    ucode_str = \
"""\
                 ⎛⎧x            ⎞\n\
                 ⎜⎪─   for x < 2⎟\n\
                 ⎜⎪y            ⎟\n\
 ⎛⎧x  for x > 0⎞ ⎜⎪             ⎟\n\
-⎜⎨            ⎟⋅⎜⎨ 2           ⎟\n\
 ⎝⎩y  otherwise⎠ ⎜⎪y   for x > 2⎟\n\
                 ⎜⎪             ⎟\n\
                 ⎜⎪1   otherwise⎟\n\
                 ⎝⎩             ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Piecewise((0, Abs(1/y) < 1), (1, Abs(y) < 1), (y*meijerg(((2, 1),
        ()), ((), (1, 0)), 1/y), True))
    ascii_str = \
"""\
/                                 1     \n\
|            0               for --- < 1\n\
|                                |y|    \n\
|                                       \n\
<            1               for |y| < 1\n\
|                                       \n\
|   __0, 2 /1, 2       | 1\\             \n\
|y*/__     |           | -|   otherwise \n\
\\  \\_|2, 2 \\      0, 1 | y/             \
"""
    ucode_str = \
"""\
⎧                                 1     \n\
⎪            0               for ─── < 1\n\
⎪                                │y│    \n\
⎪                                       \n\
⎨            1               for │y│ < 1\n\
⎪                                       \n\
⎪  ╭─╮0, 2 ⎛1, 2       │ 1⎞             \n\
⎪y⋅│╶┐     ⎜           │ ─⎟   otherwise \n\
⎩  ╰─╯2, 2 ⎝      0, 1 │ y⎠             \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    # XXX: We have to use evaluate=False here because Piecewise._eval_power
    # denests the power.
    expr = Pow(Piecewise((x, x > 0), (y, True)), 2, evaluate=False)
    ascii_str = \
"""\
               2\n\
//x  for x > 0\\ \n\
|<            | \n\
\\\\y  otherwise/ \
"""
    ucode_str = \
"""\
               2\n\
⎛⎧x  for x > 0⎞ \n\
⎜⎨            ⎟ \n\
⎝⎩y  otherwise⎠ \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_ITE():
    expr = ITE(x, y, z)
    assert pretty(expr) == (
        '/y    for x  \n'
        '<            \n'
        '\\z  otherwise'
        )
    assert upretty(expr) == """\
⎧y    for x  \n\
⎨            \n\
⎩z  otherwise\
"""


def test_pretty_seq():
    expr = ()
    ascii_str = \
"""\
()\
"""
    ucode_str = \
"""\
()\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = []
    ascii_str = \
"""\
[]\
"""
    ucode_str = \
"""\
[]\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = {}
    expr_2 = {}
    ascii_str = \
"""\
{}\
"""
    ucode_str = \
"""\
{}\
"""
    assert pretty(expr) == ascii_str
    assert pretty(expr_2) == ascii_str
    assert upretty(expr) == ucode_str
    assert upretty(expr_2) == ucode_str

    expr = (1/x,)
    ascii_str = \
"""\
 1  \n\
(-,)\n\
 x  \
"""
    ucode_str = \
"""\
⎛1 ⎞\n\
⎜─,⎟\n\
⎝x ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = [x**2, 1/x, x, y, sin(th)**2/cos(ph)**2]
    ascii_str = \
"""\
                 2        \n\
  2  1        sin (theta) \n\
[x , -, x, y, -----------]\n\
     x            2       \n\
               cos (phi)  \
"""
    ucode_str = \
"""\
⎡                2   ⎤\n\
⎢ 2  1        sin (θ)⎥\n\
⎢x , ─, x, y, ───────⎥\n\
⎢    x           2   ⎥\n\
⎣             cos (φ)⎦\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (x**2, 1/x, x, y, sin(th)**2/cos(ph)**2)
    ascii_str = \
"""\
                 2        \n\
  2  1        sin (theta) \n\
(x , -, x, y, -----------)\n\
     x            2       \n\
               cos (phi)  \
"""
    ucode_str = \
"""\
⎛                2   ⎞\n\
⎜ 2  1        sin (θ)⎟\n\
⎜x , ─, x, y, ───────⎟\n\
⎜    x           2   ⎟\n\
⎝             cos (φ)⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Tuple(x**2, 1/x, x, y, sin(th)**2/cos(ph)**2)
    ascii_str = \
"""\
                 2        \n\
  2  1        sin (theta) \n\
(x , -, x, y, -----------)\n\
     x            2       \n\
               cos (phi)  \
"""
    ucode_str = \
"""\
⎛                2   ⎞\n\
⎜ 2  1        sin (θ)⎟\n\
⎜x , ─, x, y, ───────⎟\n\
⎜    x           2   ⎟\n\
⎝             cos (φ)⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = {x: sin(x)}
    expr_2 = Dict({x: sin(x)})
    ascii_str = \
"""\
{x: sin(x)}\
"""
    ucode_str = \
"""\
{x: sin(x)}\
"""
    assert pretty(expr) == ascii_str
    assert pretty(expr_2) == ascii_str
    assert upretty(expr) == ucode_str
    assert upretty(expr_2) == ucode_str

    expr = {1/x: 1/y, x: sin(x)**2}
    expr_2 = Dict({1/x: 1/y, x: sin(x)**2})
    ascii_str = \
"""\
 1  1        2    \n\
{-: -, x: sin (x)}\n\
 x  y             \
"""
    ucode_str = \
"""\
⎧1  1        2   ⎫\n\
⎨─: ─, x: sin (x)⎬\n\
⎩x  y            ⎭\
"""
    assert pretty(expr) == ascii_str
    assert pretty(expr_2) == ascii_str
    assert upretty(expr) == ucode_str
    assert upretty(expr_2) == ucode_str

    # There used to be a bug with pretty-printing sequences of even height.
    expr = [x**2]
    ascii_str = \
"""\
  2 \n\
[x ]\
"""
    ucode_str = \
"""\
⎡ 2⎤\n\
⎣x ⎦\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (x**2,)
    ascii_str = \
"""\
  2  \n\
(x ,)\
"""
    ucode_str = \
"""\
⎛ 2 ⎞\n\
⎝x ,⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Tuple(x**2)
    ascii_str = \
"""\
  2  \n\
(x ,)\
"""
    ucode_str = \
"""\
⎛ 2 ⎞\n\
⎝x ,⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = {x**2: 1}
    expr_2 = Dict({x**2: 1})
    ascii_str = \
"""\
  2    \n\
{x : 1}\
"""
    ucode_str = \
"""\
⎧ 2   ⎫\n\
⎨x : 1⎬\n\
⎩     ⎭\
"""
    assert pretty(expr) == ascii_str
    assert pretty(expr_2) == ascii_str
    assert upretty(expr) == ucode_str
    assert upretty(expr_2) == ucode_str


def test_any_object_in_sequence():
    # Cf. issue 5306
    b1 = Basic()
    b2 = Basic(Basic())

    expr = [b2, b1]
    assert pretty(expr) == "[Basic(Basic()), Basic()]"
    assert upretty(expr) == "[Basic(Basic()), Basic()]"

    expr = {b2, b1}
    assert pretty(expr) == "{Basic(), Basic(Basic())}"
    assert upretty(expr) == "{Basic(), Basic(Basic())}"

    expr = {b2: b1, b1: b2}
    expr2 = Dict({b2: b1, b1: b2})
    assert pretty(expr) == "{Basic(): Basic(Basic()), Basic(Basic()): Basic()}"
    assert pretty(
        expr2) == "{Basic(): Basic(Basic()), Basic(Basic()): Basic()}"
    assert upretty(
        expr) == "{Basic(): Basic(Basic()), Basic(Basic()): Basic()}"
    assert upretty(
        expr2) == "{Basic(): Basic(Basic()), Basic(Basic()): Basic()}"


def test_print_builtin_set():
    assert pretty(set()) == 'set()'
    assert upretty(set()) == 'set()'

    assert pretty(frozenset()) == 'frozenset()'
    assert upretty(frozenset()) == 'frozenset()'

    s1 = {1/x, x}
    s2 = frozenset(s1)

    assert pretty(s1) == \
"""\
 1    \n\
{-, x}
 x    \
"""
    assert upretty(s1) == \
"""\
⎧1   ⎫
⎨─, x⎬
⎩x   ⎭\
"""

    assert pretty(s2) == \
"""\
           1     \n\
frozenset({-, x})
           x     \
"""
    assert upretty(s2) == \
"""\
         ⎛⎧1   ⎫⎞
frozenset⎜⎨─, x⎬⎟
         ⎝⎩x   ⎭⎠\
"""


def test_pretty_sets():
    s = FiniteSet
    assert pretty(s(*[x*y, x**2])) == \
"""\
  2      \n\
{x , x*y}\
"""
    assert pretty(s(*range(1, 6))) == "{1, 2, 3, 4, 5}"
    assert pretty(s(*range(1, 13))) == "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}"

    assert pretty({x*y, x**2}) == \
"""\
  2      \n\
{x , x*y}\
"""
    assert pretty(set(range(1, 6))) == "{1, 2, 3, 4, 5}"
    assert pretty(set(range(1, 13))) == \
        "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}"

    assert pretty(frozenset([x*y, x**2])) == \
"""\
            2       \n\
frozenset({x , x*y})\
"""
    assert pretty(frozenset(range(1, 6))) == "frozenset({1, 2, 3, 4, 5})"
    assert pretty(frozenset(range(1, 13))) == \
        "frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})"

    assert pretty(Range(0, 3, 1)) == '{0, 1, 2}'

    ascii_str = '{0, 1, ..., 29}'
    ucode_str = '{0, 1, …, 29}'
    assert pretty(Range(0, 30, 1)) == ascii_str
    assert upretty(Range(0, 30, 1)) == ucode_str

    ascii_str = '{30, 29, ..., 2}'
    ucode_str = '{30, 29, …, 2}'
    assert pretty(Range(30, 1, -1)) == ascii_str
    assert upretty(Range(30, 1, -1)) == ucode_str

    ascii_str = '{0, 2, ...}'
    ucode_str = '{0, 2, …}'
    assert pretty(Range(0, oo, 2)) == ascii_str
    assert upretty(Range(0, oo, 2)) == ucode_str

    ascii_str = '{..., 2, 0}'
    ucode_str = '{…, 2, 0}'
    assert pretty(Range(oo, -2, -2)) == ascii_str
    assert upretty(Range(oo, -2, -2)) == ucode_str

    ascii_str = '{-2, -3, ...}'
    ucode_str = '{-2, -3, …}'
    assert pretty(Range(-2, -oo, -1)) == ascii_str
    assert upretty(Range(-2, -oo, -1)) == ucode_str


def test_pretty_SetExpr():
    iv = Interval(1, 3)
    se = SetExpr(iv)
    ascii_str = "SetExpr([1, 3])"
    ucode_str = "SetExpr([1, 3])"
    assert pretty(se) == ascii_str
    assert upretty(se) == ucode_str


def test_pretty_ImageSet():
    imgset = ImageSet(Lambda((x, y), x + y), {1, 2, 3}, {3, 4})
    ascii_str = '{x + y | x in {1, 2, 3}, y in {3, 4}}'
    ucode_str = '{x + y │ x ∊ {1, 2, 3}, y ∊ {3, 4}}'
    assert pretty(imgset) == ascii_str
    assert upretty(imgset) == ucode_str

    imgset = ImageSet(Lambda(((x, y),), x + y), ProductSet({1, 2, 3}, {3, 4}))
    ascii_str = '{x + y | (x, y) in {1, 2, 3} x {3, 4}}'
    ucode_str = '{x + y │ (x, y) ∊ {1, 2, 3} × {3, 4}}'
    assert pretty(imgset) == ascii_str
    assert upretty(imgset) == ucode_str

    imgset = ImageSet(Lambda(x, x**2), S.Naturals)
    ascii_str = '''\
  2                 \n\
{x  | x in Naturals}'''
    ucode_str = '''\
⎧ 2 │      ⎫\n\
⎨x  │ x ∊ ℕ⎬\n\
⎩   │      ⎭'''
    assert pretty(imgset) == ascii_str
    assert upretty(imgset) == ucode_str

    # TODO: The "x in N" parts below should be centered independently of the
    # 1/x**2 fraction
    imgset = ImageSet(Lambda(x, 1/x**2), S.Naturals)
    ascii_str = '''\
 1                  \n\
{-- | x in Naturals}
  2                 \n\
 x                  '''
    ucode_str = '''\
⎧1  │      ⎫\n\
⎪── │ x ∊ ℕ⎪\n\
⎨ 2 │      ⎬\n\
⎪x  │      ⎪\n\
⎩   │      ⎭'''
    assert pretty(imgset) == ascii_str
    assert upretty(imgset) == ucode_str

    imgset = ImageSet(Lambda((x, y), 1/(x + y)**2), S.Naturals, S.Naturals)
    ascii_str = '''\
    1                                    \n\
{-------- | x in Naturals, y in Naturals}
        2                                \n\
 (x + y)                                 '''
    ucode_str = '''\
⎧   1     │             ⎫
⎪──────── │ x ∊ ℕ, y ∊ ℕ⎪
⎨       2 │             ⎬
⎪(x + y)  │             ⎪
⎩         │             ⎭'''
    assert pretty(imgset) == ascii_str
    assert upretty(imgset) == ucode_str

    # issue 23449 centering issue
    assert upretty([Symbol("ihat") / (Symbol("i") + 1)]) == '''\
⎡  î  ⎤
⎢─────⎥
⎣i + 1⎦\
'''
    assert upretty(Matrix([Symbol("ihat"), Symbol("i") + 1])) == '''\
⎡  î  ⎤
⎢     ⎥
⎣i + 1⎦\
'''


def test_pretty_ConditionSet():
    ascii_str = '{x | x in (-oo, oo) and sin(x) = 0}'
    ucode_str = '{x │ x ∊ ℝ ∧ (sin(x) = 0)}'
    assert pretty(ConditionSet(x, Eq(sin(x), 0), S.Reals)) == ascii_str
    assert upretty(ConditionSet(x, Eq(sin(x), 0), S.Reals)) == ucode_str

    assert pretty(ConditionSet(x, Contains(x, S.Reals, evaluate=False), FiniteSet(1))) == '{1}'
    assert upretty(ConditionSet(x, Contains(x, S.Reals, evaluate=False), FiniteSet(1))) == '{1}'

    assert pretty(ConditionSet(x, And(x > 1, x < -1), FiniteSet(1, 2, 3))) == "EmptySet"
    assert upretty(ConditionSet(x, And(x > 1, x < -1), FiniteSet(1, 2, 3))) == "∅"

    assert pretty(ConditionSet(x, Or(x > 1, x < -1), FiniteSet(1, 2))) == '{2}'
    assert upretty(ConditionSet(x, Or(x > 1, x < -1), FiniteSet(1, 2))) == '{2}'

    condset = ConditionSet(x, 1/x**2 > 0)
    ascii_str = '''\
     1      \n\
{x | -- > 0}
      2     \n\
     x      '''
    ucode_str = '''\
⎧  │ ⎛1     ⎞⎫
⎪x │ ⎜── > 0⎟⎪
⎨  │ ⎜ 2    ⎟⎬
⎪  │ ⎝x     ⎠⎪
⎩  │         ⎭'''
    assert pretty(condset) == ascii_str
    assert upretty(condset) == ucode_str

    condset = ConditionSet(x, 1/x**2 > 0, S.Reals)
    ascii_str = '''\
                        1      \n\
{x | x in (-oo, oo) and -- > 0}
                         2     \n\
                        x      '''
    ucode_str = '''\
⎧  │         ⎛1     ⎞⎫
⎪x │ x ∊ ℝ ∧ ⎜── > 0⎟⎪
⎨  │         ⎜ 2    ⎟⎬
⎪  │         ⎝x     ⎠⎪
⎩  │                 ⎭'''
    assert pretty(condset) == ascii_str
    assert upretty(condset) == ucode_str


def test_pretty_ComplexRegion():
    from sympy.sets.fancysets import ComplexRegion
    cregion = ComplexRegion(Interval(3, 5)*Interval(4, 6))
    ascii_str = '{x + y*I | x, y in [3, 5] x [4, 6]}'
    ucode_str = '{x + y⋅ⅈ │ x, y ∊ [3, 5] × [4, 6]}'
    assert pretty(cregion) == ascii_str
    assert upretty(cregion) == ucode_str

    cregion = ComplexRegion(Interval(0, 1)*Interval(0, 2*pi), polar=True)
    ascii_str = '{r*(I*sin(theta) + cos(theta)) | r, theta in [0, 1] x [0, 2*pi)}'
    ucode_str = '{r⋅(ⅈ⋅sin(θ) + cos(θ)) │ r, θ ∊ [0, 1] × [0, 2⋅π)}'
    assert pretty(cregion) == ascii_str
    assert upretty(cregion) == ucode_str

    cregion = ComplexRegion(Interval(3, 1/a**2)*Interval(4, 6))
    ascii_str = '''\
                       1            \n\
{x + y*I | x, y in [3, --] x [4, 6]}
                        2           \n\
                       a            '''
    ucode_str = '''\
⎧        │        ⎡   1 ⎤         ⎫
⎪x + y⋅ⅈ │ x, y ∊ ⎢3, ──⎥ × [4, 6]⎪
⎨        │        ⎢    2⎥         ⎬
⎪        │        ⎣   a ⎦         ⎪
⎩        │                        ⎭'''
    assert pretty(cregion) == ascii_str
    assert upretty(cregion) == ucode_str

    cregion = ComplexRegion(Interval(0, 1/a**2)*Interval(0, 2*pi), polar=True)
    ascii_str = '''\
                                                 1               \n\
{r*(I*sin(theta) + cos(theta)) | r, theta in [0, --] x [0, 2*pi)}
                                                  2              \n\
                                                 a               '''
    ucode_str = '''\
⎧                      │        ⎡   1 ⎤           ⎫
⎪r⋅(ⅈ⋅sin(θ) + cos(θ)) │ r, θ ∊ ⎢0, ──⎥ × [0, 2⋅π)⎪
⎨                      │        ⎢    2⎥           ⎬
⎪                      │        ⎣   a ⎦           ⎪
⎩                      │                          ⎭'''
    assert pretty(cregion) == ascii_str
    assert upretty(cregion) == ucode_str


def test_pretty_Union_issue_10414():
    a, b = Interval(2, 3), Interval(4, 7)
    ucode_str = '[2, 3] ∪ [4, 7]'
    ascii_str = '[2, 3] U [4, 7]'
    assert upretty(Union(a, b)) == ucode_str
    assert pretty(Union(a, b)) == ascii_str


def test_pretty_Intersection_issue_10414():
    x, y, z, w = symbols('x, y, z, w')
    a, b = Interval(x, y), Interval(z, w)
    ucode_str = '[x, y] ∩ [z, w]'
    ascii_str = '[x, y] n [z, w]'
    assert upretty(Intersection(a, b)) == ucode_str
    assert pretty(Intersection(a, b)) == ascii_str


def test_ProductSet_exponent():
    ucode_str = '      1\n[0, 1] '
    assert upretty(Interval(0, 1)**1) == ucode_str
    ucode_str = '      2\n[0, 1] '
    assert upretty(Interval(0, 1)**2) == ucode_str


def test_ProductSet_parenthesis():
    ucode_str = '([4, 7] × {1, 2}) ∪ ([2, 3] × [4, 7])'

    a, b = Interval(2, 3), Interval(4, 7)
    assert upretty(Union(a*b, b*FiniteSet(1, 2))) == ucode_str


def test_ProductSet_prod_char_issue_10413():
    ascii_str = '[2, 3] x [4, 7]'
    ucode_str = '[2, 3] × [4, 7]'

    a, b = Interval(2, 3), Interval(4, 7)
    assert pretty(a*b) == ascii_str
    assert upretty(a*b) == ucode_str


def test_pretty_sequences():
    s1 = SeqFormula(a**2, (0, oo))
    s2 = SeqPer((1, 2))

    ascii_str = '[0, 1, 4, 9, ...]'
    ucode_str = '[0, 1, 4, 9, …]'

    assert pretty(s1) == ascii_str
    assert upretty(s1) == ucode_str

    ascii_str = '[1, 2, 1, 2, ...]'
    ucode_str = '[1, 2, 1, 2, …]'
    assert pretty(s2) == ascii_str
    assert upretty(s2) == ucode_str

    s3 = SeqFormula(a**2, (0, 2))
    s4 = SeqPer((1, 2), (0, 2))

    ascii_str = '[0, 1, 4]'
    ucode_str = '[0, 1, 4]'

    assert pretty(s3) == ascii_str
    assert upretty(s3) == ucode_str

    ascii_str = '[1, 2, 1]'
    ucode_str = '[1, 2, 1]'
    assert pretty(s4) == ascii_str
    assert upretty(s4) == ucode_str

    s5 = SeqFormula(a**2, (-oo, 0))
    s6 = SeqPer((1, 2), (-oo, 0))

    ascii_str = '[..., 9, 4, 1, 0]'
    ucode_str = '[…, 9, 4, 1, 0]'

    assert pretty(s5) == ascii_str
    assert upretty(s5) == ucode_str

    ascii_str = '[..., 2, 1, 2, 1]'
    ucode_str = '[…, 2, 1, 2, 1]'
    assert pretty(s6) == ascii_str
    assert upretty(s6) == ucode_str

    ascii_str = '[1, 3, 5, 11, ...]'
    ucode_str = '[1, 3, 5, 11, …]'

    assert pretty(SeqAdd(s1, s2)) == ascii_str
    assert upretty(SeqAdd(s1, s2)) == ucode_str

    ascii_str = '[1, 3, 5]'
    ucode_str = '[1, 3, 5]'

    assert pretty(SeqAdd(s3, s4)) == ascii_str
    assert upretty(SeqAdd(s3, s4)) == ucode_str

    ascii_str = '[..., 11, 5, 3, 1]'
    ucode_str = '[…, 11, 5, 3, 1]'

    assert pretty(SeqAdd(s5, s6)) == ascii_str
    assert upretty(SeqAdd(s5, s6)) == ucode_str

    ascii_str = '[0, 2, 4, 18, ...]'
    ucode_str = '[0, 2, 4, 18, …]'

    assert pretty(SeqMul(s1, s2)) == ascii_str
    assert upretty(SeqMul(s1, s2)) == ucode_str

    ascii_str = '[0, 2, 4]'
    ucode_str = '[0, 2, 4]'

    assert pretty(SeqMul(s3, s4)) == ascii_str
    assert upretty(SeqMul(s3, s4)) == ucode_str

    ascii_str = '[..., 18, 4, 2, 0]'
    ucode_str = '[…, 18, 4, 2, 0]'

    assert pretty(SeqMul(s5, s6)) == ascii_str
    assert upretty(SeqMul(s5, s6)) == ucode_str

    # Sequences with symbolic limits, issue 12629
    s7 = SeqFormula(a**2, (a, 0, x))
    raises(NotImplementedError, lambda: pretty(s7))
    raises(NotImplementedError, lambda: upretty(s7))

    b = Symbol('b')
    s8 = SeqFormula(b*a**2, (a, 0, 2))
    ascii_str = '[0, b, 4*b]'
    ucode_str = '[0, b, 4⋅b]'
    assert pretty(s8) == ascii_str
    assert upretty(s8) == ucode_str


def test_pretty_FourierSeries():
    f = fourier_series(x, (x, -pi, pi))

    ascii_str = \
"""\
                      2*sin(3*x)      \n\
2*sin(x) - sin(2*x) + ---------- + ...\n\
                          3           \
"""

    ucode_str = \
"""\
                      2⋅sin(3⋅x)    \n\
2⋅sin(x) - sin(2⋅x) + ────────── + …\n\
                          3         \
"""

    assert pretty(f) == ascii_str
    assert upretty(f) == ucode_str


def test_pretty_FormalPowerSeries():
    f = fps(log(1 + x))


    ascii_str = \
"""\
 oo              \n\
____             \n\
\\   `            \n\
 \\         -k  k \n\
  \\   -(-1)  *x  \n\
  /   -----------\n\
 /         k     \n\
/___,            \n\
k = 1            \
"""

    ucode_str = \
"""\
 ∞               \n\
____             \n\
╲                \n\
 ╲         -k  k \n\
  ╲   -(-1)  ⋅x  \n\
  ╱   ───────────\n\
 ╱         k     \n\
╱                \n\
‾‾‾‾             \n\
k = 1            \
"""

    assert pretty(f) == ascii_str
    assert upretty(f) == ucode_str


def test_pretty_limits():
    expr = Limit(x, x, oo)
    ascii_str = \
"""\
 lim x\n\
x->oo \
"""
    ucode_str = \
"""\
lim x\n\
x─→∞ \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Limit(x**2, x, 0)
    ascii_str = \
"""\
      2\n\
 lim x \n\
x->0+  \
"""
    ucode_str = \
"""\
      2\n\
 lim x \n\
x─→0⁺  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Limit(1/x, x, 0)
    ascii_str = \
"""\
     1\n\
 lim -\n\
x->0+x\
"""
    ucode_str = \
"""\
     1\n\
 lim ─\n\
x─→0⁺x\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Limit(sin(x)/x, x, 0)
    ascii_str = \
"""\
     /sin(x)\\\n\
 lim |------|\n\
x->0+\\  x   /\
"""
    ucode_str = \
"""\
     ⎛sin(x)⎞\n\
 lim ⎜──────⎟\n\
x─→0⁺⎝  x   ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Limit(sin(x)/x, x, 0, "-")
    ascii_str = \
"""\
     /sin(x)\\\n\
 lim |------|\n\
x->0-\\  x   /\
"""
    ucode_str = \
"""\
     ⎛sin(x)⎞\n\
 lim ⎜──────⎟\n\
x─→0⁻⎝  x   ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Limit(x + sin(x), x, 0)
    ascii_str = \
"""\
 lim (x + sin(x))\n\
x->0+            \
"""
    ucode_str = \
"""\
 lim (x + sin(x))\n\
x─→0⁺            \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Limit(x, x, 0)**2
    ascii_str = \
"""\
        2\n\
/ lim x\\ \n\
\\x->0+ / \
"""
    ucode_str = \
"""\
        2\n\
⎛ lim x⎞ \n\
⎝x─→0⁺ ⎠ \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Limit(x*Limit(y/2,y,0), x, 0)
    ascii_str = \
"""\
     /       /y\\\\\n\
 lim |x* lim |-||\n\
x->0+\\  y->0+\\2//\
"""
    ucode_str = \
"""\
     ⎛       ⎛y⎞⎞\n\
 lim ⎜x⋅ lim ⎜─⎟⎟\n\
x─→0⁺⎝  y─→0⁺⎝2⎠⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = 2*Limit(x*Limit(y/2,y,0), x, 0)
    ascii_str = \
"""\
       /       /y\\\\\n\
2* lim |x* lim |-||\n\
  x->0+\\  y->0+\\2//\
"""
    ucode_str = \
"""\
       ⎛       ⎛y⎞⎞\n\
2⋅ lim ⎜x⋅ lim ⎜─⎟⎟\n\
  x─→0⁺⎝  y─→0⁺⎝2⎠⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Limit(sin(x), x, 0, dir='+-')
    ascii_str = \
"""\
lim sin(x)\n\
x->0      \
"""
    ucode_str = \
"""\
lim sin(x)\n\
x─→0      \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_ComplexRootOf():
    expr = rootof(x**5 + 11*x - 2, 0)
    ascii_str = \
"""\
       / 5              \\\n\
CRootOf\\x  + 11*x - 2, 0/\
"""
    ucode_str = \
"""\
       ⎛ 5              ⎞\n\
CRootOf⎝x  + 11⋅x - 2, 0⎠\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_RootSum():
    expr = RootSum(x**5 + 11*x - 2, auto=False)
    ascii_str = \
"""\
       / 5           \\\n\
RootSum\\x  + 11*x - 2/\
"""
    ucode_str = \
"""\
       ⎛ 5           ⎞\n\
RootSum⎝x  + 11⋅x - 2⎠\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = RootSum(x**5 + 11*x - 2, Lambda(z, exp(z)))
    ascii_str = \
"""\
       / 5                   z\\\n\
RootSum\\x  + 11*x - 2, z -> e /\
"""
    ucode_str = \
"""\
       ⎛ 5                  z⎞\n\
RootSum⎝x  + 11⋅x - 2, z ↦ ℯ ⎠\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_GroebnerBasis():
    expr = groebner([], x, y)

    ascii_str = \
"""\
GroebnerBasis([], x, y, domain=ZZ, order=lex)\
"""
    ucode_str = \
"""\
GroebnerBasis([], x, y, domain=ℤ, order=lex)\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    F = [x**2 - 3*y - x + 1, y**2 - 2*x + y - 1]
    expr = groebner(F, x, y, order='grlex')

    ascii_str = \
"""\
             /[ 2                 2              ]                              \\\n\
GroebnerBasis\\[x  - x - 3*y + 1, y  - 2*x + y - 1], x, y, domain=ZZ, order=grlex/\
"""
    ucode_str = \
"""\
             ⎛⎡ 2                 2              ⎤                             ⎞\n\
GroebnerBasis⎝⎣x  - x - 3⋅y + 1, y  - 2⋅x + y - 1⎦, x, y, domain=ℤ, order=grlex⎠\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = expr.fglm('lex')

    ascii_str = \
"""\
             /[       2           4      3      2           ]                            \\\n\
GroebnerBasis\\[2*x - y  - y + 1, y  + 2*y  - 3*y  - 16*y + 7], x, y, domain=ZZ, order=lex/\
"""
    ucode_str = \
"""\
             ⎛⎡       2           4      3      2           ⎤                           ⎞\n\
GroebnerBasis⎝⎣2⋅x - y  - y + 1, y  + 2⋅y  - 3⋅y  - 16⋅y + 7⎦, x, y, domain=ℤ, order=lex⎠\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_UniversalSet():
    assert pretty(S.UniversalSet) == "UniversalSet"
    assert upretty(S.UniversalSet) == '𝕌'


def test_pretty_Boolean():
    expr = Not(x, evaluate=False)

    assert pretty(expr) == "Not(x)"
    assert upretty(expr) == "¬x"

    expr = And(x, y)

    assert pretty(expr) == "And(x, y)"
    assert upretty(expr) == "x ∧ y"

    expr = Or(x, y)

    assert pretty(expr) == "Or(x, y)"
    assert upretty(expr) == "x ∨ y"

    syms = symbols('a:f')
    expr = And(*syms)

    assert pretty(expr) == "And(a, b, c, d, e, f)"
    assert upretty(expr) == "a ∧ b ∧ c ∧ d ∧ e ∧ f"

    expr = Or(*syms)

    assert pretty(expr) == "Or(a, b, c, d, e, f)"
    assert upretty(expr) == "a ∨ b ∨ c ∨ d ∨ e ∨ f"

    expr = Xor(x, y, evaluate=False)

    assert pretty(expr) == "Xor(x, y)"
    assert upretty(expr) == "x ⊻ y"

    expr = Nand(x, y, evaluate=False)

    assert pretty(expr) == "Nand(x, y)"
    assert upretty(expr) == "x ⊼ y"

    expr = Nor(x, y, evaluate=False)

    assert pretty(expr) == "Nor(x, y)"
    assert upretty(expr) == "x ⊽ y"

    expr = Implies(x, y, evaluate=False)

    assert pretty(expr) == "Implies(x, y)"
    assert upretty(expr) == "x → y"

    # don't sort args
    expr = Implies(y, x, evaluate=False)

    assert pretty(expr) == "Implies(y, x)"
    assert upretty(expr) == "y → x"

    expr = Equivalent(x, y, evaluate=False)

    assert pretty(expr) == "Equivalent(x, y)"
    assert upretty(expr) == "x ⇔ y"

    expr = Equivalent(y, x, evaluate=False)

    assert pretty(expr) == "Equivalent(x, y)"
    assert upretty(expr) == "x ⇔ y"


def test_pretty_Domain():
    expr = FF(23)

    assert pretty(expr) == "GF(23)"
    assert upretty(expr) == "ℤ₂₃"

    expr = ZZ

    assert pretty(expr) == "ZZ"
    assert upretty(expr) == "ℤ"

    expr = QQ

    assert pretty(expr) == "QQ"
    assert upretty(expr) == "ℚ"

    expr = RR

    assert pretty(expr) == "RR"
    assert upretty(expr) == "ℝ"

    expr = QQ[x]

    assert pretty(expr) == "QQ[x]"
    assert upretty(expr) == "ℚ[x]"

    expr = QQ[x, y]

    assert pretty(expr) == "QQ[x, y]"
    assert upretty(expr) == "ℚ[x, y]"

    expr = ZZ.frac_field(x)

    assert pretty(expr) == "ZZ(x)"
    assert upretty(expr) == "ℤ(x)"

    expr = ZZ.frac_field(x, y)

    assert pretty(expr) == "ZZ(x, y)"
    assert upretty(expr) == "ℤ(x, y)"

    expr = QQ.poly_ring(x, y, order=grlex)

    assert pretty(expr) == "QQ[x, y, order=grlex]"
    assert upretty(expr) == "ℚ[x, y, order=grlex]"

    expr = QQ.poly_ring(x, y, order=ilex)

    assert pretty(expr) == "QQ[x, y, order=ilex]"
    assert upretty(expr) == "ℚ[x, y, order=ilex]"


def test_pretty_prec():
    assert xpretty(S("0.3"), full_prec=True, wrap_line=False) == "0.300000000000000"
    assert xpretty(S("0.3"), full_prec="auto", wrap_line=False) == "0.300000000000000"
    assert xpretty(S("0.3"), full_prec=False, wrap_line=False) == "0.3"
    assert xpretty(S("0.3")*x, full_prec=True, use_unicode=False, wrap_line=False) in [
        "0.300000000000000*x",
        "x*0.300000000000000"
    ]
    assert xpretty(S("0.3")*x, full_prec="auto", use_unicode=False, wrap_line=False) in [
        "0.3*x",
        "x*0.3"
    ]
    assert xpretty(S("0.3")*x, full_prec=False, use_unicode=False, wrap_line=False) in [
        "0.3*x",
        "x*0.3"
    ]


def test_pprint():
    import sys
    from io import StringIO
    fd = StringIO()
    sso = sys.stdout
    sys.stdout = fd
    try:
        pprint(pi, use_unicode=False, wrap_line=False)
    finally:
        sys.stdout = sso
    assert fd.getvalue() == 'pi\n'


def test_pretty_class():
    """Test that the printer dispatcher correctly handles classes."""
    class C:
        pass   # C has no .__class__ and this was causing problems

    class D:
        pass

    assert pretty( C ) == str( C )
    assert pretty( D ) == str( D )


def test_pretty_no_wrap_line():
    huge_expr = 0
    for i in range(20):
        huge_expr += i*sin(i + x)
    assert xpretty(huge_expr            ).find('\n') != -1
    assert xpretty(huge_expr, wrap_line=False).find('\n') == -1


def test_settings():
    raises(TypeError, lambda: pretty(S(4), method="garbage"))


def test_pretty_sum():
    from sympy.abc import x, a, b, k, m, n

    expr = Sum(k**k, (k, 0, n))
    ascii_str = \
"""\
 n      \n\
___     \n\
\\  `    \n\
 \\     k\n\
 /    k \n\
/__,    \n\
k = 0   \
"""
    ucode_str = \
"""\
  n     \n\
 ___    \n\
 ╲      \n\
  ╲    k\n\
  ╱   k \n\
 ╱      \n\
 ‾‾‾    \n\
k = 0   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(k**k, (k, oo, n))
    ascii_str = \
"""\
  n      \n\
 ___     \n\
 \\  `    \n\
  \\     k\n\
  /    k \n\
 /__,    \n\
k = oo   \
"""
    ucode_str = \
"""\
  n     \n\
 ___    \n\
 ╲      \n\
  ╲    k\n\
  ╱   k \n\
 ╱      \n\
 ‾‾‾    \n\
k = ∞   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(k**(Integral(x**n, (x, -oo, oo))), (k, 0, n**n))
    ascii_str = \
"""\
   n              \n\
  n               \n\
______            \n\
\\     `           \n\
 \\        oo      \n\
  \\        /      \n\
   \\      |       \n\
    \\     |   n   \n\
     )    |  x  dx\n\
    /     |       \n\
   /     /        \n\
  /      -oo      \n\
 /      k         \n\
/_____,           \n\
 k = 0            \
"""
    ucode_str = \
"""\
   n            \n\
  n             \n\
______          \n\
╲               \n\
 ╲              \n\
  ╲     ∞       \n\
   ╲    ⌠       \n\
    ╲   ⎮   n   \n\
    ╱   ⎮  x  dx\n\
   ╱    ⌡       \n\
  ╱     -∞      \n\
 ╱     k        \n\
╱               \n\
‾‾‾‾‾‾          \n\
k = 0           \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(k**(
        Integral(x**n, (x, -oo, oo))), (k, 0, Integral(x**x, (x, -oo, oo))))
    ascii_str = \
"""\
 oo                 \n\
  /                 \n\
 |                  \n\
 |   x              \n\
 |  x  dx           \n\
 |                  \n\
/                   \n\
-oo                 \n\
 ______             \n\
 \\     `            \n\
  \\         oo      \n\
   \\         /      \n\
    \\       |       \n\
     \\      |   n   \n\
      )     |  x  dx\n\
     /      |       \n\
    /      /        \n\
   /       -oo      \n\
  /       k         \n\
 /_____,            \n\
  k = 0             \
"""
    ucode_str = \
"""\
∞                 \n\
⌠                 \n\
⎮   x             \n\
⎮  x  dx          \n\
⌡                 \n\
-∞                \n\
 ______           \n\
 ╲                \n\
  ╲               \n\
   ╲      ∞       \n\
    ╲     ⌠       \n\
     ╲    ⎮   n   \n\
     ╱    ⎮  x  dx\n\
    ╱     ⌡       \n\
   ╱      -∞      \n\
  ╱      k        \n\
 ╱                \n\
 ‾‾‾‾‾‾           \n\
 k = 0            \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(k**(Integral(x**n, (x, -oo, oo))), (
        k, x + n + x**2 + n**2 + (x/n) + (1/x), Integral(x**x, (x, -oo, oo))))
    ascii_str = \
"""\
          oo                          \n\
           /                          \n\
          |                           \n\
          |   x                       \n\
          |  x  dx                    \n\
          |                           \n\
         /                            \n\
         -oo                          \n\
          ______                      \n\
          \\     `                     \n\
           \\                  oo      \n\
            \\                  /      \n\
             \\                |       \n\
              \\               |   n   \n\
               )              |  x  dx\n\
              /               |       \n\
             /               /        \n\
            /                -oo      \n\
           /                k         \n\
          /_____,                     \n\
     2        2       1   x           \n\
k = n  + n + x  + x + - + -           \n\
                      x   n           \
"""
    ucode_str = \
"""\
         ∞                           \n\
         ⌠                           \n\
         ⎮   x                       \n\
         ⎮  x  dx                    \n\
         ⌡                           \n\
         -∞                          \n\
          ______                     \n\
          ╲                          \n\
           ╲                         \n\
            ╲                ∞       \n\
             ╲               ⌠       \n\
              ╲              ⎮   n   \n\
              ╱              ⎮  x  dx\n\
             ╱               ⌡       \n\
            ╱                -∞      \n\
           ╱                k        \n\
          ╱                          \n\
          ‾‾‾‾‾‾                     \n\
     2        2       1   x          \n\
k = n  + n + x  + x + ─ + ─          \n\
                      x   n          \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(k**(
        Integral(x**n, (x, -oo, oo))), (k, 0, x + n + x**2 + n**2 + (x/n) + (1/x)))
    ascii_str = \
"""\
 2        2       1   x           \n\
n  + n + x  + x + - + -           \n\
                  x   n           \n\
        ______                    \n\
        \\     `                   \n\
         \\                oo      \n\
          \\                /      \n\
           \\              |       \n\
            \\             |   n   \n\
             )            |  x  dx\n\
            /             |       \n\
           /             /        \n\
          /              -oo      \n\
         /              k         \n\
        /_____,                   \n\
         k = 0                    \
"""
    ucode_str = \
"""\
 2        2       1   x          \n\
n  + n + x  + x + ─ + ─          \n\
                  x   n          \n\
        ______                   \n\
        ╲                        \n\
         ╲                       \n\
          ╲              ∞       \n\
           ╲             ⌠       \n\
            ╲            ⎮   n   \n\
            ╱            ⎮  x  dx\n\
           ╱             ⌡       \n\
          ╱              -∞      \n\
         ╱              k        \n\
        ╱                        \n\
        ‾‾‾‾‾‾                   \n\
         k = 0                   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(x, (x, 0, oo))
    ascii_str = \
"""\
 oo    \n\
 __    \n\
 \\ `   \n\
  )   x\n\
 /_,   \n\
x = 0  \
"""
    ucode_str = \
"""\
  ∞    \n\
 ___   \n\
 ╲     \n\
  ╲    \n\
  ╱   x\n\
 ╱     \n\
 ‾‾‾   \n\
x = 0  \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(x**2, (x, 0, oo))
    ascii_str = \
"""\
 oo     \n\
___     \n\
\\  `    \n\
 \\     2\n\
 /    x \n\
/__,    \n\
x = 0   \
"""
    ucode_str = \
"""\
  ∞     \n\
 ___    \n\
 ╲      \n\
  ╲    2\n\
  ╱   x \n\
 ╱      \n\
 ‾‾‾    \n\
x = 0   \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(x/2, (x, 0, oo))
    ascii_str = \
"""\
 oo    \n\
___    \n\
\\  `   \n\
 \\    x\n\
  )   -\n\
 /    2\n\
/__,   \n\
x = 0  \
"""
    ucode_str = \
"""\
 ∞     \n\
____   \n\
╲      \n\
 ╲     \n\
  ╲   x\n\
  ╱   ─\n\
 ╱    2\n\
╱      \n\
‾‾‾‾   \n\
x = 0  \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(x**3/2, (x, 0, oo))
    ascii_str = \
"""\
 oo     \n\
____    \n\
\\   `   \n\
 \\     3\n\
  \\   x \n\
  /   --\n\
 /    2 \n\
/___,   \n\
x = 0   \
"""
    ucode_str = \
"""\
 ∞      \n\
____    \n\
╲       \n\
 ╲     3\n\
  ╲   x \n\
  ╱   ──\n\
 ╱    2 \n\
╱       \n\
‾‾‾‾    \n\
x = 0   \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum((x**3*y**(x/2))**n, (x, 0, oo))
    ascii_str = \
"""\
 oo           \n\
____          \n\
\\   `         \n\
 \\           n\n\
  \\   /    x\\ \n\
   )  |    -| \n\
  /   | 3  2| \n\
 /    \\x *y / \n\
/___,         \n\
x = 0         \
"""
    ucode_str = \
"""\
  ∞           \n\
_____         \n\
╲             \n\
 ╲            \n\
  ╲          n\n\
   ╲  ⎛    x⎞ \n\
   ╱  ⎜    ─⎟ \n\
  ╱   ⎜ 3  2⎟ \n\
 ╱    ⎝x ⋅y ⎠ \n\
╱             \n\
‾‾‾‾‾         \n\
x = 0         \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(1/x**2, (x, 0, oo))
    ascii_str = \
"""\
 oo     \n\
____    \n\
\\   `   \n\
 \\    1 \n\
  \\   --\n\
  /    2\n\
 /    x \n\
/___,   \n\
x = 0   \
"""
    ucode_str = \
"""\
 ∞      \n\
____    \n\
╲       \n\
 ╲    1 \n\
  ╲   ──\n\
  ╱    2\n\
 ╱    x \n\
╱       \n\
‾‾‾‾    \n\
x = 0   \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(1/y**(a/b), (x, 0, oo))
    ascii_str = \
"""\
 oo       \n\
____      \n\
\\   `     \n\
 \\     -a \n\
  \\    ---\n\
  /     b \n\
 /    y   \n\
/___,     \n\
x = 0     \
"""
    ucode_str = \
"""\
 ∞        \n\
____      \n\
╲         \n\
 ╲     -a \n\
  ╲    ───\n\
  ╱     b \n\
 ╱    y   \n\
╱         \n\
‾‾‾‾      \n\
x = 0     \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Sum(1/y**(a/b), (x, 0, oo), (y, 1, 2))
    ascii_str = \
"""\
  2     oo     \n\
____  ____     \n\
\\   ` \\   `    \n\
 \\     \\     -a\n\
  \\     \\    --\n\
  /     /    b \n\
 /     /    y  \n\
/___, /___,    \n\
y = 1 x = 0    \
"""
    ucode_str = \
"""\
  2     ∞      \n\
____  ____     \n\
╲     ╲        \n\
 ╲     ╲     -a\n\
  ╲     ╲    ──\n\
  ╱     ╱    b \n\
 ╱     ╱    y  \n\
╱     ╱        \n\
‾‾‾‾  ‾‾‾‾     \n\
y = 1 x = 0    \
"""
    expr = Sum(1/(1 + 1/(
        1 + 1/k)) + 1, (k, 111, 1 + 1/n), (k, 1/(1 + m), oo)) + 1/(1 + 1/k)
    ascii_str = \
"""\
              1                          \n\
          1 + -                          \n\
   oo         n                          \n\
 _____    _____                          \n\
 \\    `   \\    `                         \n\
  \\        \\      /        1    \\        \n\
   \\        \\     |1 + ---------|        \n\
    \\        \\    |          1  |     1  \n\
     )        )   |    1 + -----| + -----\n\
    /        /    |            1|       1\n\
   /        /     |        1 + -|   1 + -\n\
  /        /      \\            k/       k\n\
 /____,   /____,                         \n\
      1   k = 111                        \n\
k = -----                                \n\
    m + 1                                \
"""
    ucode_str = \
"""\
              1                          \n\
          1 + ─                          \n\
   ∞          n                          \n\
 ______   ______                         \n\
 ╲        ╲                              \n\
  ╲        ╲                             \n\
   ╲        ╲     ⎛        1    ⎞        \n\
    ╲        ╲    ⎜1 + ─────────⎟        \n\
     ╲        ╲   ⎜          1  ⎟     1  \n\
     ╱        ╱   ⎜    1 + ─────⎟ + ─────\n\
    ╱        ╱    ⎜            1⎟       1\n\
   ╱        ╱     ⎜        1 + ─⎟   1 + ─\n\
  ╱        ╱      ⎝            k⎠       k\n\
 ╱        ╱                              \n\
 ‾‾‾‾‾‾   ‾‾‾‾‾‾                         \n\
      1   k = 111                        \n\
k = ─────                                \n\
    m + 1                                \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_units():
    expr = joule
    ascii_str1 = \
"""\
              2\n\
kilogram*meter \n\
---------------\n\
          2    \n\
    second     \
"""
    unicode_str1 = \
"""\
              2\n\
kilogram⋅meter \n\
───────────────\n\
          2    \n\
    second     \
"""

    ascii_str2 = \
"""\
                    2\n\
3*x*y*kilogram*meter \n\
---------------------\n\
             2       \n\
       second        \
"""
    unicode_str2 = \
"""\
                    2\n\
3⋅x⋅y⋅kilogram⋅meter \n\
─────────────────────\n\
             2       \n\
       second        \
"""

    from sympy.physics.units import kg, m, s
    assert upretty(expr) == "joule"
    assert pretty(expr) == "joule"
    assert upretty(expr.convert_to(kg*m**2/s**2)) == unicode_str1
    assert pretty(expr.convert_to(kg*m**2/s**2)) == ascii_str1
    assert upretty(3*kg*x*m**2*y/s**2) == unicode_str2
    assert pretty(3*kg*x*m**2*y/s**2) == ascii_str2


def test_pretty_Subs():
    f = Function('f')
    expr = Subs(f(x), x, ph**2)
    ascii_str = \
"""\
(f(x))|     2\n\
      |x=phi \
"""
    unicode_str = \
"""\
(f(x))│   2\n\
      │x=φ \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str

    expr = Subs(f(x).diff(x), x, 0)
    ascii_str = \
"""\
/d       \\|   \n\
|--(f(x))||   \n\
\\dx      /|x=0\
"""
    unicode_str = \
"""\
⎛d       ⎞│   \n\
⎜──(f(x))⎟│   \n\
⎝dx      ⎠│x=0\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str

    expr = Subs(f(x).diff(x)/y, (x, y), (0, Rational(1, 2)))
    ascii_str = \
"""\
/d       \\|          \n\
|--(f(x))||          \n\
|dx      ||          \n\
|--------||          \n\
\\   y    /|x=0, y=1/2\
"""
    unicode_str = \
"""\
⎛d       ⎞│          \n\
⎜──(f(x))⎟│          \n\
⎜dx      ⎟│          \n\
⎜────────⎟│          \n\
⎝   y    ⎠│x=0, y=1/2\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == unicode_str


def test_gammas():
    assert upretty(lowergamma(x, y)) == "γ(x, y)"
    assert upretty(uppergamma(x, y)) == "Γ(x, y)"
    assert xpretty(gamma(x), use_unicode=True) == 'Γ(x)'
    assert xpretty(gamma, use_unicode=True) == 'Γ'
    assert xpretty(symbols('gamma', cls=Function)(x), use_unicode=True) == 'γ(x)'
    assert xpretty(symbols('gamma', cls=Function), use_unicode=True) == 'γ'


def test_beta():
    assert xpretty(beta(x,y), use_unicode=True) == 'Β(x, y)'
    assert xpretty(beta(x,y), use_unicode=False) == 'B(x, y)'
    assert xpretty(beta, use_unicode=True) == 'Β'
    assert xpretty(beta, use_unicode=False) == 'B'
    mybeta = Function('beta')
    assert xpretty(mybeta(x), use_unicode=True) == 'β(x)'
    assert xpretty(mybeta(x, y, z), use_unicode=False) == 'beta(x, y, z)'
    assert xpretty(mybeta, use_unicode=True) == 'β'


# test that notation passes to subclasses of the same name only
def test_function_subclass_different_name():
    class mygamma(gamma):
        pass
    assert xpretty(mygamma, use_unicode=True) == r"mygamma"
    assert xpretty(mygamma(x), use_unicode=True) == r"mygamma(x)"


def test_SingularityFunction():
    assert xpretty(SingularityFunction(x, 0, n), use_unicode=True) == (
"""\
   n\n\
<x> \
""")
    assert xpretty(SingularityFunction(x, 1, n), use_unicode=True) == (
"""\
       n\n\
<x - 1> \
""")
    assert xpretty(SingularityFunction(x, -1, n), use_unicode=True) == (
"""\
       n\n\
<x + 1> \
""")
    assert xpretty(SingularityFunction(x, a, n), use_unicode=True) == (
"""\
        n\n\
<-a + x> \
""")
    assert xpretty(SingularityFunction(x, y, n), use_unicode=True) == (
"""\
       n\n\
<x - y> \
""")
    assert xpretty(SingularityFunction(x, 0, n), use_unicode=False) == (
"""\
   n\n\
<x> \
""")
    assert xpretty(SingularityFunction(x, 1, n), use_unicode=False) == (
"""\
       n\n\
<x - 1> \
""")
    assert xpretty(SingularityFunction(x, -1, n), use_unicode=False) == (
"""\
       n\n\
<x + 1> \
""")
    assert xpretty(SingularityFunction(x, a, n), use_unicode=False) == (
"""\
        n\n\
<-a + x> \
""")
    assert xpretty(SingularityFunction(x, y, n), use_unicode=False) == (
"""\
       n\n\
<x - y> \
""")


def test_deltas():
    assert xpretty(DiracDelta(x), use_unicode=True) == 'δ(x)'
    assert xpretty(DiracDelta(x, 1), use_unicode=True) == \
"""\
 (1)    \n\
δ    (x)\
"""
    assert xpretty(x*DiracDelta(x, 1), use_unicode=True) == \
"""\
   (1)    \n\
x⋅δ    (x)\
"""


def test_hyper():
    expr = hyper((), (), z)
    ucode_str = \
"""\
 ┌─  ⎛  │  ⎞\n\
 ├─  ⎜  │ z⎟\n\
0╵ 0 ⎝  │  ⎠\
"""
    ascii_str = \
"""\
  _         \n\
 |_  /  |  \\\n\
 |   |  | z|\n\
0  0 \\  |  /\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = hyper((), (1,), x)
    ucode_str = \
"""\
 ┌─  ⎛  │  ⎞\n\
 ├─  ⎜  │ x⎟\n\
0╵ 1 ⎝1 │  ⎠\
"""
    ascii_str = \
"""\
  _         \n\
 |_  /  |  \\\n\
 |   |  | x|\n\
0  1 \\1 |  /\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = hyper([2], [1], x)
    ucode_str = \
"""\
 ┌─  ⎛2 │  ⎞\n\
 ├─  ⎜  │ x⎟\n\
1╵ 1 ⎝1 │  ⎠\
"""
    ascii_str = \
"""\
  _         \n\
 |_  /2 |  \\\n\
 |   |  | x|\n\
1  1 \\1 |  /\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = hyper((pi/3, -2*k), (3, 4, 5, -3), x)
    ucode_str = \
"""\
     ⎛  π         │  ⎞\n\
 ┌─  ⎜  ─, -2⋅k   │  ⎟\n\
 ├─  ⎜  3         │ x⎟\n\
2╵ 4 ⎜            │  ⎟\n\
     ⎝-3, 3, 4, 5 │  ⎠\
"""
    ascii_str = \
"""\
                      \n\
  _  / pi         |  \\\n\
 |_  | --, -2*k   |  |\n\
 |   | 3          | x|\n\
2  4 |            |  |\n\
     \\-3, 3, 4, 5 |  /\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = hyper((pi, S('2/3'), -2*k), (3, 4, 5, -3), x**2)
    ucode_str = \
"""\
 ┌─  ⎛2/3, π, -2⋅k │  2⎞\n\
 ├─  ⎜             │ x ⎟\n\
3╵ 4 ⎝-3, 3, 4, 5  │   ⎠\
"""
    ascii_str = \
"""\
  _                      \n\
 |_  /2/3, pi, -2*k |  2\\
 |   |              | x |
3  4 \\ -3, 3, 4, 5  |   /"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = hyper([1, 2], [3, 4], 1/(1/(1/(1/x + 1) + 1) + 1))
    ucode_str = \
"""\
     ⎛     │       1      ⎞\n\
     ⎜     │ ─────────────⎟\n\
     ⎜     │         1    ⎟\n\
 ┌─  ⎜1, 2 │ 1 + ─────────⎟\n\
 ├─  ⎜     │           1  ⎟\n\
2╵ 2 ⎜3, 4 │     1 + ─────⎟\n\
     ⎜     │             1⎟\n\
     ⎜     │         1 + ─⎟\n\
     ⎝     │             x⎠\
"""

    ascii_str = \
"""\
                           \n\
     /     |       1      \\\n\
     |     | -------------|\n\
  _  |     |         1    |\n\
 |_  |1, 2 | 1 + ---------|\n\
 |   |     |           1  |\n\
2  2 |3, 4 |     1 + -----|\n\
     |     |             1|\n\
     |     |         1 + -|\n\
     \\     |             x/\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_meijerg():
    expr = meijerg([pi, pi, x], [1], [0, 1], [1, 2, 3], z)
    ucode_str = \
"""\
╭─╮2, 3 ⎛π, π, x     1    │  ⎞\n\
│╶┐     ⎜                 │ z⎟\n\
╰─╯4, 5 ⎝ 0, 1    1, 2, 3 │  ⎠\
"""
    ascii_str = \
"""\
 __2, 3 /pi, pi, x     1    |  \\\n\
/__     |                   | z|\n\
\\_|4, 5 \\  0, 1     1, 2, 3 |  /\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = meijerg([1, pi/7], [2, pi, 5], [], [], z**2)
    ucode_str = \
"""\
        ⎛   π          │   ⎞\n\
╭─╮0, 2 ⎜1, ─  2, 5, π │  2⎟\n\
│╶┐     ⎜   7          │ z ⎟\n\
╰─╯5, 0 ⎜              │   ⎟\n\
        ⎝              │   ⎠\
"""
    ascii_str = \
"""\
        /   pi           |   \\\n\
 __0, 2 |1, --  2, 5, pi |  2|\n\
/__     |   7            | z |\n\
\\_|5, 0 |                |   |\n\
        \\                |   /\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ucode_str = \
"""\
╭─╮ 1, 10 ⎛1, 1, 1, 1, 1, 1, 1, 1, 1, 1  1 │  ⎞\n\
│╶┐       ⎜                                │ z⎟\n\
╰─╯11,  2 ⎝             1                1 │  ⎠\
"""
    ascii_str = \
"""\
 __ 1, 10 /1, 1, 1, 1, 1, 1, 1, 1, 1, 1  1 |  \\\n\
/__       |                                | z|\n\
\\_|11,  2 \\             1                1 |  /\
"""

    expr = meijerg([1]*10, [1], [1], [1], z)
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = meijerg([1, 2, ], [4, 3], [3], [4, 5], 1/(1/(1/(1/x + 1) + 1) + 1))

    ucode_str = \
"""\
        ⎛           │       1      ⎞\n\
        ⎜           │ ─────────────⎟\n\
        ⎜           │         1    ⎟\n\
╭─╮1, 2 ⎜1, 2  3, 4 │ 1 + ─────────⎟\n\
│╶┐     ⎜           │           1  ⎟\n\
╰─╯4, 3 ⎜ 3    4, 5 │     1 + ─────⎟\n\
        ⎜           │             1⎟\n\
        ⎜           │         1 + ─⎟\n\
        ⎝           │             x⎠\
"""

    ascii_str = \
"""\
        /           |       1      \\\n\
        |           | -------------|\n\
        |           |         1    |\n\
 __1, 2 |1, 2  3, 4 | 1 + ---------|\n\
/__     |           |           1  |\n\
\\_|4, 3 | 3    4, 5 |     1 + -----|\n\
        |           |             1|\n\
        |           |         1 + -|\n\
        \\           |             x/\
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = Integral(expr, x)

    ucode_str = \
"""\
⌠                                        \n\
⎮         ⎛           │       1      ⎞   \n\
⎮         ⎜           │ ─────────────⎟   \n\
⎮         ⎜           │         1    ⎟   \n\
⎮ ╭─╮1, 2 ⎜1, 2  3, 4 │ 1 + ─────────⎟   \n\
⎮ │╶┐     ⎜           │           1  ⎟ dx\n\
⎮ ╰─╯4, 3 ⎜ 3    4, 5 │     1 + ─────⎟   \n\
⎮         ⎜           │             1⎟   \n\
⎮         ⎜           │         1 + ─⎟   \n\
⎮         ⎝           │             x⎠   \n\
⌡                                        \
"""

    ascii_str = \
"""\
  /                                       \n\
 |                                        \n\
 |         /           |       1      \\   \n\
 |         |           | -------------|   \n\
 |         |           |         1    |   \n\
 |  __1, 2 |1, 2  3, 4 | 1 + ---------|   \n\
 | /__     |           |           1  | dx\n\
 | \\_|4, 3 | 3    4, 5 |     1 + -----|   \n\
 |         |           |             1|   \n\
 |         |           |         1 + -|   \n\
 |         \\           |             x/   \n\
 |                                        \n\
/                                         \
"""

    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_noncommutative():
    A, B, C = symbols('A,B,C', commutative=False)

    expr = A*B*C**-1
    ascii_str = \
"""\
     -1\n\
A*B*C  \
"""
    ucode_str = \
"""\
     -1\n\
A⋅B⋅C  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = C**-1*A*B
    ascii_str = \
"""\
 -1    \n\
C  *A*B\
"""
    ucode_str = \
"""\
 -1    \n\
C  ⋅A⋅B\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = A*C**-1*B
    ascii_str = \
"""\
   -1  \n\
A*C  *B\
"""
    ucode_str = \
"""\
   -1  \n\
A⋅C  ⋅B\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = A*C**-1*B/x
    ascii_str = \
"""\
   -1  \n\
A*C  *B\n\
-------\n\
   x   \
"""
    ucode_str = \
"""\
   -1  \n\
A⋅C  ⋅B\n\
───────\n\
   x   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_special_functions():
    x, y = symbols("x y")

    # atan2
    expr = atan2(y/sqrt(200), sqrt(x))
    ascii_str = \
"""\
     /  ___         \\\n\
     |\\/ 2 *y    ___|\n\
atan2|-------, \\/ x |\n\
     \\  20          /\
"""
    ucode_str = \
"""\
     ⎛√2⋅y    ⎞\n\
atan2⎜────, √x⎟\n\
     ⎝ 20     ⎠\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_geometry():
    e = Segment((0, 1), (0, 2))
    assert pretty(e) == 'Segment2D(Point2D(0, 1), Point2D(0, 2))'
    e = Ray((1, 1), angle=4.02*pi)
    assert pretty(e) == 'Ray2D(Point2D(1, 1), Point2D(2, tan(pi/50) + 1))'


def test_expint():
    expr = Ei(x)
    string = 'Ei(x)'
    assert pretty(expr) == string
    assert upretty(expr) == string

    expr = expint(1, z)
    ucode_str = "E₁(z)"
    ascii_str = "expint(1, z)"
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    assert pretty(Shi(x)) == 'Shi(x)'
    assert pretty(Si(x)) == 'Si(x)'
    assert pretty(Ci(x)) == 'Ci(x)'
    assert pretty(Chi(x)) == 'Chi(x)'
    assert upretty(Shi(x)) == 'Shi(x)'
    assert upretty(Si(x)) == 'Si(x)'
    assert upretty(Ci(x)) == 'Ci(x)'
    assert upretty(Chi(x)) == 'Chi(x)'


def test_elliptic_functions():
    ascii_str = \
"""\
 /  1  \\\n\
K|-----|\n\
 \\z + 1/\
"""
    ucode_str = \
"""\
 ⎛  1  ⎞\n\
K⎜─────⎟\n\
 ⎝z + 1⎠\
"""
    expr = elliptic_k(1/(z + 1))
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ascii_str = \
"""\
 / |  1  \\\n\
F|1|-----|\n\
 \\ |z + 1/\
"""
    ucode_str = \
"""\
 ⎛ │  1  ⎞\n\
F⎜1│─────⎟\n\
 ⎝ │z + 1⎠\
"""
    expr = elliptic_f(1, 1/(1 + z))
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ascii_str = \
"""\
 /  1  \\\n\
E|-----|\n\
 \\z + 1/\
"""
    ucode_str = \
"""\
 ⎛  1  ⎞\n\
E⎜─────⎟\n\
 ⎝z + 1⎠\
"""
    expr = elliptic_e(1/(z + 1))
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ascii_str = \
"""\
 / |  1  \\\n\
E|1|-----|\n\
 \\ |z + 1/\
"""
    ucode_str = \
"""\
 ⎛ │  1  ⎞\n\
E⎜1│─────⎟\n\
 ⎝ │z + 1⎠\
"""
    expr = elliptic_e(1, 1/(1 + z))
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ascii_str = \
"""\
  / |4\\\n\
Pi|3|-|\n\
  \\ |x/\
"""
    ucode_str = \
"""\
 ⎛ │4⎞\n\
Π⎜3│─⎟\n\
 ⎝ │x⎠\
"""
    expr = elliptic_pi(3, 4/x)
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    ascii_str = \
"""\
  /   4| \\\n\
Pi|3; -|6|\n\
  \\   x| /\
"""
    ucode_str = \
"""\
 ⎛   4│ ⎞\n\
Π⎜3; ─│6⎟\n\
 ⎝   x│ ⎠\
"""
    expr = elliptic_pi(3, 4/x, 6)
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_RandomDomain():
    from sympy.stats import Normal, Die, Exponential, pspace, where
    X = Normal('x1', 0, 1)
    assert upretty(where(X > 0)) == "Domain: 0 < x₁ ∧ x₁ < ∞"

    D = Die('d1', 6)
    assert upretty(where(D > 4)) == 'Domain: d₁ = 5 ∨ d₁ = 6'

    A = Exponential('a', 1)
    B = Exponential('b', 1)
    assert upretty(pspace(Tuple(A, B)).domain) == \
        'Domain: 0 ≤ a ∧ 0 ≤ b ∧ a < ∞ ∧ b < ∞'


def test_PrettyPoly():
    F = QQ.frac_field(x, y)
    R = QQ.poly_ring(x, y)

    expr = F.convert(x/(x + y))
    assert pretty(expr) == "x/(x + y)"
    assert upretty(expr) == "x/(x + y)"

    expr = R.convert(x + y)
    assert pretty(expr) == "x + y"
    assert upretty(expr) == "x + y"


def test_issue_6285():
    assert pretty(Pow(2, -5, evaluate=False)) == '1 \n--\n 5\n2 '
    assert pretty(Pow(x, (1/pi))) == \
    ' 1 \n'\
    ' --\n'\
    ' pi\n'\
    'x  '


def test_issue_6359():
    assert pretty(Integral(x**2, x)**2) == \
"""\
          2
/  /     \\ \n\
| |      | \n\
| |  2   | \n\
| | x  dx| \n\
| |      | \n\
\\/       / \
"""
    assert upretty(Integral(x**2, x)**2) == \
"""\
         2
⎛⌠      ⎞ \n\
⎜⎮  2   ⎟ \n\
⎜⎮ x  dx⎟ \n\
⎝⌡      ⎠ \
"""

    assert pretty(Sum(x**2, (x, 0, 1))**2) == \
"""\
          2\n\
/ 1      \\ \n\
|___     | \n\
|\\  `    | \n\
| \\     2| \n\
| /    x | \n\
|/__,    | \n\
\\x = 0   / \
"""
    assert upretty(Sum(x**2, (x, 0, 1))**2) == \
"""\
          2
⎛  1     ⎞ \n\
⎜ ___    ⎟ \n\
⎜ ╲      ⎟ \n\
⎜  ╲    2⎟ \n\
⎜  ╱   x ⎟ \n\
⎜ ╱      ⎟ \n\
⎜ ‾‾‾    ⎟ \n\
⎝x = 0   ⎠ \
"""

    assert pretty(Product(x**2, (x, 1, 2))**2) == \
"""\
           2
/  2      \\ \n\
|______   | \n\
| |  |   2| \n\
| |  |  x | \n\
| |  |    | \n\
\\x = 1    / \
"""
    assert upretty(Product(x**2, (x, 1, 2))**2) == \
"""\
           2
⎛  2      ⎞ \n\
⎜─┬──┬─   ⎟ \n\
⎜ │  │   2⎟ \n\
⎜ │  │  x ⎟ \n\
⎜ │  │    ⎟ \n\
⎝x = 1    ⎠ \
"""

    f = Function('f')
    assert pretty(Derivative(f(x), x)**2) == \
"""\
          2
/d       \\ \n\
|--(f(x))| \n\
\\dx      / \
"""
    assert upretty(Derivative(f(x), x)**2) == \
"""\
          2
⎛d       ⎞ \n\
⎜──(f(x))⎟ \n\
⎝dx      ⎠ \
"""


def test_issue_6739():
    ascii_str = \
"""\
  1  \n\
-----\n\
  ___\n\
\\/ x \
"""
    ucode_str = \
"""\
1 \n\
──\n\
√x\
"""
    assert pretty(1/sqrt(x)) == ascii_str
    assert upretty(1/sqrt(x)) == ucode_str


def test_complicated_symbol_unchanged():
    for symb_name in ["dexpr2_d1tau", "dexpr2^d1tau"]:
        assert pretty(Symbol(symb_name)) == symb_name


def test_categories():
    from sympy.categories import (Object, IdentityMorphism,
        NamedMorphism, Category, Diagram, DiagramGrid)

    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")

    f1 = NamedMorphism(A1, A2, "f1")
    f2 = NamedMorphism(A2, A3, "f2")
    id_A1 = IdentityMorphism(A1)

    K1 = Category("K1")

    assert pretty(A1) == "A1"
    assert upretty(A1) == "A₁"

    assert pretty(f1) == "f1:A1-->A2"
    assert upretty(f1) == "f₁:A₁——▶A₂"
    assert pretty(id_A1) == "id:A1-->A1"
    assert upretty(id_A1) == "id:A₁——▶A₁"

    assert pretty(f2*f1) == "f2*f1:A1-->A3"
    assert upretty(f2*f1) == "f₂∘f₁:A₁——▶A₃"

    assert pretty(K1) == "K1"
    assert upretty(K1) == "K₁"

    # Test how diagrams are printed.
    d = Diagram()
    assert pretty(d) == "EmptySet"
    assert upretty(d) == "∅"

    d = Diagram({f1: "unique", f2: S.EmptySet})
    assert pretty(d) == "{f2*f1:A1-->A3: EmptySet, id:A1-->A1: " \
        "EmptySet, id:A2-->A2: EmptySet, id:A3-->A3: " \
        "EmptySet, f1:A1-->A2: {unique}, f2:A2-->A3: EmptySet}"

    assert upretty(d) == "{f₂∘f₁:A₁——▶A₃: ∅, id:A₁——▶A₁: ∅, " \
        "id:A₂——▶A₂: ∅, id:A₃——▶A₃: ∅, f₁:A₁——▶A₂: {unique}, f₂:A₂——▶A₃: ∅}"

    d = Diagram({f1: "unique", f2: S.EmptySet}, {f2 * f1: "unique"})
    assert pretty(d) == "{f2*f1:A1-->A3: EmptySet, id:A1-->A1: " \
        "EmptySet, id:A2-->A2: EmptySet, id:A3-->A3: " \
        "EmptySet, f1:A1-->A2: {unique}, f2:A2-->A3: EmptySet}" \
        " ==> {f2*f1:A1-->A3: {unique}}"
    assert upretty(d) == "{f₂∘f₁:A₁——▶A₃: ∅, id:A₁——▶A₁: ∅, id:A₂——▶A₂: " \
        "∅, id:A₃——▶A₃: ∅, f₁:A₁——▶A₂: {unique}, f₂:A₂——▶A₃: ∅}" \
        " ══▶ {f₂∘f₁:A₁——▶A₃: {unique}}"

    grid = DiagramGrid(d)
    assert pretty(grid) == "A1  A2\n      \nA3    "
    assert upretty(grid) == "A₁  A₂\n      \nA₃    "


def test_PrettyModules():
    R = QQ.old_poly_ring(x, y)
    F = R.free_module(2)
    M = F.submodule([x, y], [1, x**2])

    ucode_str = \
"""\
       2\n\
ℚ[x, y] \
"""
    ascii_str = \
"""\
        2\n\
QQ[x, y] \
"""

    assert upretty(F) == ucode_str
    assert pretty(F) == ascii_str

    ucode_str = \
"""\
╱        ⎡    2⎤╲\n\
╲[x, y], ⎣1, x ⎦╱\
"""
    ascii_str = \
"""\
              2  \n\
<[x, y], [1, x ]>\
"""

    assert upretty(M) == ucode_str
    assert pretty(M) == ascii_str

    I = R.ideal(x**2, y)

    ucode_str = \
"""\
╱ 2   ╲\n\
╲x , y╱\
"""

    ascii_str = \
"""\
  2    \n\
<x , y>\
"""

    assert upretty(I) == ucode_str
    assert pretty(I) == ascii_str

    Q = F / M

    ucode_str = \
"""\
           2     \n\
    ℚ[x, y]      \n\
─────────────────\n\
╱        ⎡    2⎤╲\n\
╲[x, y], ⎣1, x ⎦╱\
"""

    ascii_str = \
"""\
            2    \n\
    QQ[x, y]     \n\
-----------------\n\
              2  \n\
<[x, y], [1, x ]>\
"""

    assert upretty(Q) == ucode_str
    assert pretty(Q) == ascii_str

    ucode_str = \
"""\
╱⎡    3⎤                                                ╲\n\
│⎢   x ⎥   ╱        ⎡    2⎤╲           ╱        ⎡    2⎤╲│\n\
│⎢1, ──⎥ + ╲[x, y], ⎣1, x ⎦╱, [2, y] + ╲[x, y], ⎣1, x ⎦╱│\n\
╲⎣   2 ⎦                                                ╱\
"""

    ascii_str = \
"""\
      3                                                  \n\
     x                   2                           2   \n\
<[1, --] + <[x, y], [1, x ]>, [2, y] + <[x, y], [1, x ]>>\n\
     2                                                   \
"""


def test_QuotientRing():
    R = QQ.old_poly_ring(x)/[x**2 + 1]

    ucode_str = \
"""\
  ℚ[x]  \n\
────────\n\
╱ 2    ╲\n\
╲x  + 1╱\
"""

    ascii_str = \
"""\
 QQ[x]  \n\
--------\n\
  2     \n\
<x  + 1>\
"""

    assert upretty(R) == ucode_str
    assert pretty(R) == ascii_str

    ucode_str = \
"""\
    ╱ 2    ╲\n\
1 + ╲x  + 1╱\
"""

    ascii_str = \
"""\
      2     \n\
1 + <x  + 1>\
"""

    assert upretty(R.one) == ucode_str
    assert pretty(R.one) == ascii_str


def test_Homomorphism():
    from sympy.polys.agca import homomorphism

    R = QQ.old_poly_ring(x)

    expr = homomorphism(R.free_module(1), R.free_module(1), [0])

    ucode_str = \
"""\
          1         1\n\
[0] : ℚ[x]  ──> ℚ[x] \
"""

    ascii_str = \
"""\
           1          1\n\
[0] : QQ[x]  --> QQ[x] \
"""

    assert upretty(expr) == ucode_str
    assert pretty(expr) == ascii_str

    expr = homomorphism(R.free_module(2), R.free_module(2), [0, 0])

    ucode_str = \
"""\
⎡0  0⎤       2         2\n\
⎢    ⎥ : ℚ[x]  ──> ℚ[x] \n\
⎣0  0⎦                  \
"""

    ascii_str = \
"""\
[0  0]        2          2\n\
[    ] : QQ[x]  --> QQ[x] \n\
[0  0]                    \
"""

    assert upretty(expr) == ucode_str
    assert pretty(expr) == ascii_str

    expr = homomorphism(R.free_module(1), R.free_module(1) / [[x]], [0])

    ucode_str = \
"""\
                    1\n\
          1     ℚ[x] \n\
[0] : ℚ[x]  ──> ─────\n\
                <[x]>\
"""

    ascii_str = \
"""\
                      1\n\
           1     QQ[x] \n\
[0] : QQ[x]  --> ------\n\
                 <[x]> \
"""

    assert upretty(expr) == ucode_str
    assert pretty(expr) == ascii_str


def test_Tr():
    A, B = symbols('A B', commutative=False)
    t = Tr(A*B)
    assert pretty(t) == r'Tr(A*B)'
    assert upretty(t) == 'Tr(A⋅B)'


def test_pretty_Add():
    eq = Mul(-2, x - 2, evaluate=False) + 5
    assert pretty(eq) == '5 - 2*(x - 2)'


def test_issue_7179():
    assert upretty(Not(Equivalent(x, y))) == 'x ⇎ y'
    assert upretty(Not(Implies(x, y))) == 'x ↛ y'


def test_issue_7180():
    assert upretty(Equivalent(x, y)) == 'x ⇔ y'


def test_pretty_Complement():
    assert pretty(S.Reals - S.Naturals) == '(-oo, oo) \\ Naturals'
    assert upretty(S.Reals - S.Naturals) == 'ℝ \\ ℕ'
    assert pretty(S.Reals - S.Naturals0) == '(-oo, oo) \\ Naturals0'
    assert upretty(S.Reals - S.Naturals0) == 'ℝ \\ ℕ₀'


def test_pretty_SymmetricDifference():
    from sympy.sets.sets import SymmetricDifference
    assert upretty(SymmetricDifference(Interval(2,3), Interval(3,5), \
           evaluate = False)) == '[2, 3] ∆ [3, 5]'
    with raises(NotImplementedError):
        pretty(SymmetricDifference(Interval(2,3), Interval(3,5), evaluate = False))


def test_pretty_Contains():
    assert pretty(Contains(x, S.Integers)) == 'Contains(x, Integers)'
    assert upretty(Contains(x, S.Integers)) == 'x ∈ ℤ'


def test_issue_8292():
    from sympy.core import sympify
    e = sympify('((x+x**4)/(x-1))-(2*(x-1)**4/(x-1)**4)', evaluate=False)
    ucode_str = \
"""\
           4    4    \n\
  2⋅(x - 1)    x  + x\n\
- ────────── + ──────\n\
          4    x - 1 \n\
   (x - 1)           \
"""
    ascii_str = \
"""\
           4    4    \n\
  2*(x - 1)    x  + x\n\
- ---------- + ------\n\
          4    x - 1 \n\
   (x - 1)           \
"""
    assert pretty(e) == ascii_str
    assert upretty(e) == ucode_str


def test_issue_4335():
    y = Function('y')
    expr = -y(x).diff(x)
    ucode_str = \
"""\
 d       \n\
-──(y(x))\n\
 dx      \
"""
    ascii_str = \
"""\
  d       \n\
- --(y(x))\n\
  dx      \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_issue_8344():
    from sympy.core import sympify
    e = sympify('2*x*y**2/1**2 + 1', evaluate=False)
    ucode_str = \
"""\
     2    \n\
2⋅x⋅y     \n\
────── + 1\n\
   2      \n\
  1       \
"""
    assert upretty(e) == ucode_str


def test_issue_6324():
    x = Pow(2, 3, evaluate=False)
    y = Pow(10, -2, evaluate=False)
    e = Mul(x, y, evaluate=False)
    ucode_str = \
"""\
 3 \n\
2  \n\
───\n\
  2\n\
10 \
"""
    assert upretty(e) == ucode_str


def test_issue_7927():
    e = sin(x/2)**cos(x/2)
    ucode_str = \
"""\
           ⎛x⎞\n\
        cos⎜─⎟\n\
           ⎝2⎠\n\
⎛   ⎛x⎞⎞      \n\
⎜sin⎜─⎟⎟      \n\
⎝   ⎝2⎠⎠      \
"""
    assert upretty(e) == ucode_str
    e = sin(x)**(S(11)/13)
    ucode_str = \
"""\
        11\n\
        ──\n\
        13\n\
(sin(x))  \
"""
    assert upretty(e) == ucode_str


def test_issue_6134():
    from sympy.abc import lamda, t
    phi = Function('phi')

    e = lamda*x*Integral(phi(t)*pi*sin(pi*t), (t, 0, 1)) + lamda*x**2*Integral(phi(t)*2*pi*sin(2*pi*t), (t, 0, 1))
    ucode_str = \
"""\
     1                              1                   \n\
   2 ⌠                              ⌠                   \n\
λ⋅x ⋅⎮ 2⋅π⋅φ(t)⋅sin(2⋅π⋅t) dt + λ⋅x⋅⎮ π⋅φ(t)⋅sin(π⋅t) dt\n\
     ⌡                              ⌡                   \n\
     0                              0                   \
"""
    assert upretty(e) == ucode_str


def test_issue_9877():
    ucode_str1 = '(2, 3) ∪ ([1, 2] \\ {x})'
    a, b, c = Interval(2, 3, True, True), Interval(1, 2), FiniteSet(x)
    assert upretty(Union(a, Complement(b, c))) == ucode_str1

    ucode_str2 = '{x} ∩ {y} ∩ ({z} \\ [1, 2])'
    d, e, f, g = FiniteSet(x), FiniteSet(y), FiniteSet(z), Interval(1, 2)
    assert upretty(Intersection(d, e, Complement(f, g))) == ucode_str2


def test_issue_13651():
    expr1 = c + Mul(-1, a + b, evaluate=False)
    assert pretty(expr1) == 'c - (a + b)'
    expr2 = c + Mul(-1, a - b + d, evaluate=False)
    assert pretty(expr2) == 'c - (a - b + d)'


def test_pretty_primenu():
    from sympy.functions.combinatorial.numbers import primenu

    ascii_str1 = "nu(n)"
    ucode_str1 = "ν(n)"

    n = symbols('n', integer=True)
    assert pretty(primenu(n)) == ascii_str1
    assert upretty(primenu(n)) == ucode_str1


def test_pretty_primeomega():
    from sympy.functions.combinatorial.numbers import primeomega

    ascii_str1 = "Omega(n)"
    ucode_str1 = "Ω(n)"

    n = symbols('n', integer=True)
    assert pretty(primeomega(n)) == ascii_str1
    assert upretty(primeomega(n)) == ucode_str1


def test_pretty_Mod():
    from sympy.core import Mod

    ascii_str1 = "x mod 7"
    ucode_str1 = "x mod 7"

    ascii_str2 = "(x + 1) mod 7"
    ucode_str2 = "(x + 1) mod 7"

    ascii_str3 = "2*x mod 7"
    ucode_str3 = "2⋅x mod 7"

    ascii_str4 = "(x mod 7) + 1"
    ucode_str4 = "(x mod 7) + 1"

    ascii_str5 = "2*(x mod 7)"
    ucode_str5 = "2⋅(x mod 7)"

    x = symbols('x', integer=True)
    assert pretty(Mod(x, 7)) == ascii_str1
    assert upretty(Mod(x, 7)) == ucode_str1
    assert pretty(Mod(x + 1, 7)) == ascii_str2
    assert upretty(Mod(x + 1, 7)) == ucode_str2
    assert pretty(Mod(2 * x, 7)) == ascii_str3
    assert upretty(Mod(2 * x, 7)) == ucode_str3
    assert pretty(Mod(x, 7) + 1) == ascii_str4
    assert upretty(Mod(x, 7) + 1) == ucode_str4
    assert pretty(2 * Mod(x, 7)) == ascii_str5
    assert upretty(2 * Mod(x, 7)) == ucode_str5


def test_issue_11801():
    assert pretty(Symbol("")) == ""
    assert upretty(Symbol("")) == ""


def test_pretty_UnevaluatedExpr():
    x = symbols('x')
    he = UnevaluatedExpr(1/x)

    ucode_str = \
"""\
1\n\
─\n\
x\
"""

    assert upretty(he) == ucode_str

    ucode_str = \
"""\
   2\n\
⎛1⎞ \n\
⎜─⎟ \n\
⎝x⎠ \
"""

    assert upretty(he**2) == ucode_str

    ucode_str = \
"""\
    1\n\
1 + ─\n\
    x\
"""

    assert upretty(he + 1) == ucode_str

    ucode_str = \
('''\
  1\n\
x⋅─\n\
  x\
''')
    assert upretty(x*he) == ucode_str


def test_issue_10472():
    M = (Matrix([[0, 0], [0, 0]]), Matrix([0, 0]))

    ucode_str = \
"""\
⎛⎡0  0⎤  ⎡0⎤⎞
⎜⎢    ⎥, ⎢ ⎥⎟
⎝⎣0  0⎦  ⎣0⎦⎠\
"""
    assert upretty(M) == ucode_str


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    ascii_str1 = "A_00"
    ucode_str1 = "A₀₀"
    assert pretty(A[0, 0])  == ascii_str1
    assert upretty(A[0, 0]) == ucode_str1

    ascii_str1 = "3*A_00"
    ucode_str1 = "3⋅A₀₀"
    assert pretty(3*A[0, 0])  == ascii_str1
    assert upretty(3*A[0, 0]) == ucode_str1

    ascii_str1 = "(-B + A)[0, 0]"
    ucode_str1 = "(-B + A)[0, 0]"
    F = C[0, 0].subs(C, A - B)
    assert pretty(F)  == ascii_str1
    assert upretty(F) == ucode_str1


def test_issue_12675():
    x, y, t, j = symbols('x y t j')
    e = CoordSys3D('e')

    ucode_str = \
"""\
⎛   t⎞    \n\
⎜⎛x⎞ ⎟ j_e\n\
⎜⎜─⎟ ⎟    \n\
⎝⎝y⎠ ⎠    \
"""
    assert upretty((x/y)**t*e.j) == ucode_str
    ucode_str = \
"""\
⎛1⎞    \n\
⎜─⎟ j_e\n\
⎝y⎠    \
"""
    assert upretty((1/y)*e.j) == ucode_str


def test_MatrixSymbol_printing():
    # test cases for issue #14237
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)
    assert pretty(-A*B*C) == "-A*B*C"
    assert pretty(A - B) == "-B + A"
    assert pretty(A*B*C - A*B - B*C) == "-A*B -B*C + A*B*C"

    # issue #14814
    x = MatrixSymbol('x', n, n)
    y = MatrixSymbol('y*', n, n)
    assert pretty(x + y) == "x + y*"
    ascii_str = \
"""\
     2     \n\
-2*y*  -a*x\
"""
    assert pretty(-a*x + -2*y*y) == ascii_str


def test_degree_printing():
    expr1 = 90*degree
    assert pretty(expr1) == '90°'
    expr2 = x*degree
    assert pretty(expr2) == 'x°'
    expr3 = cos(x*degree + 90*degree)
    assert pretty(expr3) == 'cos(x° + 90°)'


def test_vector_expr_pretty_printing():
    A = CoordSys3D('A')

    assert upretty(Cross(A.i, A.x*A.i+3*A.y*A.j)) == "(i_A)×((x_A) i_A + (3⋅y_A) j_A)"
    assert upretty(x*Cross(A.i, A.j)) == 'x⋅(i_A)×(j_A)'

    assert upretty(Curl(A.x*A.i + 3*A.y*A.j)) == "∇×((x_A) i_A + (3⋅y_A) j_A)"

    assert upretty(Divergence(A.x*A.i + 3*A.y*A.j)) == "∇⋅((x_A) i_A + (3⋅y_A) j_A)"

    assert upretty(Dot(A.i, A.x*A.i+3*A.y*A.j)) == "(i_A)⋅((x_A) i_A + (3⋅y_A) j_A)"

    assert upretty(Gradient(A.x+3*A.y)) == "∇(x_A + 3⋅y_A)"
    assert upretty(Laplacian(A.x+3*A.y)) == "∆(x_A + 3⋅y_A)"
    # TODO: add support for ASCII pretty.


def test_pretty_print_tensor_expr():
    L = TensorIndexType("L")
    i, j, k = tensor_indices("i j k", L)
    i0 = tensor_indices("i_0", L)
    A, B, C, D = tensor_heads("A B C D", [L])
    H = TensorHead("H", [L, L])

    expr = -i
    ascii_str = \
"""\
-i\
"""
    ucode_str = \
"""\
-i\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = A(i)
    ascii_str = \
"""\
 i\n\
A \n\
  \
"""
    ucode_str = \
"""\
 i\n\
A \n\
  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = A(i0)
    ascii_str = \
"""\
 i_0\n\
A   \n\
    \
"""
    ucode_str = \
"""\
 i₀\n\
A  \n\
   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = A(-i)
    ascii_str = \
"""\
  \n\
A \n\
 i\
"""
    ucode_str = \
"""\
  \n\
A \n\
 i\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = -3*A(-i)
    ascii_str = \
"""\
     \n\
-3*A \n\
    i\
"""
    ucode_str = \
"""\
     \n\
-3⋅A \n\
    i\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = H(i, -j)
    ascii_str = \
"""\
 i \n\
H  \n\
  j\
"""
    ucode_str = \
"""\
 i \n\
H  \n\
  j\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = H(i, -i)
    ascii_str = \
"""\
 L_0   \n\
H      \n\
    L_0\
"""
    ucode_str = \
"""\
 L₀  \n\
H    \n\
   L₀\
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = H(i, -j)*A(j)*B(k)
    ascii_str = \
"""\
 i     L_0  k\n\
H    *A   *B \n\
  L_0        \
"""
    ucode_str = \
"""\
 i    L₀  k\n\
H   ⋅A  ⋅B \n\
  L₀       \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (1+x)*A(i)
    ascii_str = \
"""\
         i\n\
(x + 1)*A \n\
          \
"""
    ucode_str = \
"""\
         i\n\
(x + 1)⋅A \n\
          \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = A(i) + 3*B(i)
    ascii_str = \
"""\
   i    i\n\
3*B  + A \n\
         \
"""
    ucode_str = \
"""\
   i    i\n\
3⋅B  + A \n\
         \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_pretty_print_tensor_partial_deriv():
    from sympy.tensor.toperators import PartialDerivative

    L = TensorIndexType("L")
    i, j, k = tensor_indices("i j k", L)

    A, B, C, D = tensor_heads("A B C D", [L])

    H = TensorHead("H", [L, L])

    expr = PartialDerivative(A(i), A(j))
    ascii_str = \
"""\
 d / i\\\n\
---|A |\n\
  j\\  /\n\
dA     \n\
       \
"""
    ucode_str = \
"""\
 ∂ ⎛ i⎞\n\
───⎜A ⎟\n\
  j⎝  ⎠\n\
∂A     \n\
       \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = A(i)*PartialDerivative(H(k, -i), A(j))
    ascii_str = \
"""\
 L_0  d / k   \\\n\
A   *---|H    |\n\
       j\\  L_0/\n\
     dA        \n\
               \
"""
    ucode_str = \
"""\
 L₀  ∂ ⎛ k  ⎞\n\
A  ⋅───⎜H   ⎟\n\
      j⎝  L₀⎠\n\
    ∂A       \n\
             \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = A(i)*PartialDerivative(B(k)*C(-i) + 3*H(k, -i), A(j))
    ascii_str = \
"""\
 L_0  d /   k       k     \\\n\
A   *---|3*H     + B *C   |\n\
       j\\    L_0       L_0/\n\
     dA                    \n\
                           \
"""
    ucode_str = \
"""\
 L₀  ∂ ⎛   k      k    ⎞\n\
A  ⋅───⎜3⋅H    + B ⋅C  ⎟\n\
      j⎝    L₀       L₀⎠\n\
    ∂A                  \n\
                        \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (A(i) + B(i))*PartialDerivative(C(j), D(j))
    ascii_str = \
"""\
/ i    i\\   d  / L_0\\\n\
|A  + B |*-----|C   |\n\
\\       /   L_0\\    /\n\
          dD         \n\
                     \
"""
    ucode_str = \
"""\
⎛ i    i⎞  ∂  ⎛ L₀⎞\n\
⎜A  + B ⎟⋅────⎜C  ⎟\n\
⎝       ⎠   L₀⎝   ⎠\n\
          ∂D       \n\
                   \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = (A(i) + B(i))*PartialDerivative(C(-i), D(j))
    ascii_str = \
"""\
/ L_0    L_0\\  d /    \\\n\
|A    + B   |*---|C   |\n\
\\           /   j\\ L_0/\n\
              dD       \n\
                       \
"""
    ucode_str = \
"""\
⎛ L₀    L₀⎞  ∂ ⎛   ⎞\n\
⎜A   + B  ⎟⋅───⎜C  ⎟\n\
⎝         ⎠   j⎝ L₀⎠\n\
            ∂D      \n\
                    \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = PartialDerivative(B(-i) + A(-i), A(-j), A(-n))
    ucode_str = """\
   2            \n\
  ∂    ⎛       ⎞\n\
───────⎜A  + B ⎟\n\
       ⎝ i    i⎠\n\
∂A  ∂A          \n\
  n   j         \
"""
    assert upretty(expr) == ucode_str

    expr = PartialDerivative(3*A(-i), A(-j), A(-n))
    ucode_str = """\
   2         \n\
  ∂    ⎛    ⎞\n\
───────⎜3⋅A ⎟\n\
       ⎝   i⎠\n\
∂A  ∂A       \n\
  n   j      \
"""
    assert upretty(expr) == ucode_str

    expr = TensorElement(H(i, j), {i:1})
    ascii_str = \
"""\
 i=1,j\n\
H     \n\
      \
"""
    ucode_str = ascii_str
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = TensorElement(H(i, j), {i: 1, j: 1})
    ascii_str = \
"""\
 i=1,j=1\n\
H       \n\
        \
"""
    ucode_str = ascii_str
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = TensorElement(H(i, j), {j: 1})
    ascii_str = \
"""\
 i,j=1\n\
H     \n\
      \
"""
    ucode_str = ascii_str

    expr = TensorElement(H(-i, j), {-i: 1})
    ascii_str = \
"""\
    j\n\
H    \n\
 i=1 \
"""
    ucode_str = ascii_str
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_issue_15560():
    a = MatrixSymbol('a', 1, 1)
    e = pretty(a*(KroneckerProduct(a, a)))
    result = 'a*(a x a)'
    assert e == result


def test_print_polylog():
    # Part of issue 6013
    uresult = 'Li₂(3)'
    aresult = 'polylog(2, 3)'
    assert pretty(polylog(2, 3)) == aresult
    assert upretty(polylog(2, 3)) == uresult


# Issue #25312
def test_print_expint_polylog_symbolic_order():
    s, z = symbols("s, z")
    uresult = 'Liₛ(z)'
    aresult = 'polylog(s, z)'
    assert pretty(polylog(s, z)) == aresult
    assert upretty(polylog(s, z)) == uresult
    # TODO: TBD polylog(s - 1, z)
    uresult = 'Eₛ(z)'
    aresult = 'expint(s, z)'
    assert pretty(expint(s, z)) == aresult
    assert upretty(expint(s, z)) == uresult



def test_print_polylog_long_order_issue_25309():
    s, z = symbols("s, z")
    ucode_str = \
"""\
       ⎛ 2   ⎞\n\
polylog⎝s , z⎠\
"""
    assert upretty(polylog(s**2, z)) == ucode_str


def test_print_lerchphi():
    # Part of issue 6013
    a = Symbol('a')
    pretty(lerchphi(a, 1, 2))
    uresult = 'Φ(a, 1, 2)'
    aresult = 'lerchphi(a, 1, 2)'
    assert pretty(lerchphi(a, 1, 2)) == aresult
    assert upretty(lerchphi(a, 1, 2)) == uresult


def test_issue_15583():

    N = mechanics.ReferenceFrame('N')
    result = '(n_x, n_y, n_z)'
    e = pretty((N.x, N.y, N.z))
    assert e == result


def test_matrixSymbolBold():
    # Issue 15871
    def boldpretty(expr):
        return xpretty(expr, use_unicode=True, wrap_line=False, mat_symbol_style="bold")

    from sympy.matrices.expressions.trace import trace
    A = MatrixSymbol("A", 2, 2)
    assert boldpretty(trace(A)) == 'tr(𝐀)'

    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)

    assert boldpretty(-A) == '-𝐀'
    assert boldpretty(A - A*B - B) == '-𝐁 -𝐀⋅𝐁 + 𝐀'
    assert boldpretty(-A*B - A*B*C - B) == '-𝐁 -𝐀⋅𝐁 -𝐀⋅𝐁⋅𝐂'

    A = MatrixSymbol("Addot", 3, 3)
    assert boldpretty(A) == '𝐀̈'
    omega = MatrixSymbol("omega", 3, 3)
    assert boldpretty(omega) == 'ω'
    omega = MatrixSymbol("omeganorm", 3, 3)
    assert boldpretty(omega) == '‖ω‖'

    a = Symbol('alpha')
    b = Symbol('b')
    c = MatrixSymbol("c", 3, 1)
    d = MatrixSymbol("d", 3, 1)

    assert boldpretty(a*B*c+b*d) == 'b⋅𝐝 + α⋅𝐁⋅𝐜'

    d = MatrixSymbol("delta", 3, 1)
    B = MatrixSymbol("Beta", 3, 3)

    assert boldpretty(a*B*c+b*d) == 'b⋅δ + α⋅Β⋅𝐜'

    A = MatrixSymbol("A_2", 3, 3)
    assert boldpretty(A) == '𝐀₂'


def test_center_accent():
    assert center_accent('a', '\N{COMBINING TILDE}') == 'ã'
    assert center_accent('aa', '\N{COMBINING TILDE}') == 'aã'
    assert center_accent('aaa', '\N{COMBINING TILDE}') == 'aãa'
    assert center_accent('aaaa', '\N{COMBINING TILDE}') == 'aaãa'
    assert center_accent('aaaaa', '\N{COMBINING TILDE}') == 'aaãaa'
    assert center_accent('abcdefg', '\N{COMBINING FOUR DOTS ABOVE}') == 'abcd⃜efg'


def test_imaginary_unit():
    from sympy.printing.pretty import pretty  # b/c it was redefined above
    assert pretty(1 + I, use_unicode=False) == '1 + I'
    assert pretty(1 + I, use_unicode=True) == '1 + ⅈ'
    assert pretty(1 + I, use_unicode=False, imaginary_unit='j') == '1 + I'
    assert pretty(1 + I, use_unicode=True, imaginary_unit='j') == '1 + ⅉ'

    raises(TypeError, lambda: pretty(I, imaginary_unit=I))
    raises(ValueError, lambda: pretty(I, imaginary_unit="kkk"))


def test_str_special_matrices():
    from sympy.matrices import Identity, ZeroMatrix, OneMatrix
    assert pretty(Identity(4)) == 'I'
    assert upretty(Identity(4)) == '𝕀'
    assert pretty(ZeroMatrix(2, 2)) == '0'
    assert upretty(ZeroMatrix(2, 2)) == '𝟘'
    assert pretty(OneMatrix(2, 2)) == '1'
    assert upretty(OneMatrix(2, 2)) == '𝟙'


def test_pretty_misc_functions():
    assert pretty(LambertW(x)) == 'W(x)'
    assert upretty(LambertW(x)) == 'W(x)'
    assert pretty(LambertW(x, y)) == 'W(x, y)'
    assert upretty(LambertW(x, y)) == 'W(x, y)'
    assert pretty(airyai(x)) == 'Ai(x)'
    assert upretty(airyai(x)) == 'Ai(x)'
    assert pretty(airybi(x)) == 'Bi(x)'
    assert upretty(airybi(x)) == 'Bi(x)'
    assert pretty(airyaiprime(x)) == "Ai'(x)"
    assert upretty(airyaiprime(x)) == "Ai'(x)"
    assert pretty(airybiprime(x)) == "Bi'(x)"
    assert upretty(airybiprime(x)) == "Bi'(x)"
    assert pretty(fresnelc(x)) == 'C(x)'
    assert upretty(fresnelc(x)) == 'C(x)'
    assert pretty(fresnels(x)) == 'S(x)'
    assert upretty(fresnels(x)) == 'S(x)'
    assert pretty(Heaviside(x)) == 'Heaviside(x)'
    assert upretty(Heaviside(x)) == 'θ(x)'
    assert pretty(Heaviside(x, y)) == 'Heaviside(x, y)'
    assert upretty(Heaviside(x, y)) == 'θ(x, y)'
    assert pretty(dirichlet_eta(x)) == 'dirichlet_eta(x)'
    assert upretty(dirichlet_eta(x)) == 'η(x)'


def test_hadamard_power():
    m, n, p = symbols('m, n, p', integer=True)
    A = MatrixSymbol('A', m, n)
    B = MatrixSymbol('B', m, n)

    # Testing printer:
    expr = hadamard_power(A, n)
    ascii_str = \
"""\
 .n\n\
A  \
"""
    ucode_str = \
"""\
 ∘n\n\
A  \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = hadamard_power(A, 1+n)
    ascii_str = \
"""\
 .(n + 1)\n\
A        \
"""
    ucode_str = \
"""\
 ∘(n + 1)\n\
A        \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str

    expr = hadamard_power(A*B.T, 1+n)
    ascii_str = \
"""\
      .(n + 1)\n\
/   T\\        \n\
\\A*B /        \
"""
    ucode_str = \
"""\
      ∘(n + 1)\n\
⎛   T⎞        \n\
⎝A⋅B ⎠        \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str


def test_issue_17258():
    n = Symbol('n', integer=True)
    assert pretty(Sum(n, (n, -oo, 1))) == \
    '   1     \n'\
    '  __     \n'\
    '  \\ `    \n'\
    '   )    n\n'\
    '  /_,    \n'\
    'n = -oo  '

    assert upretty(Sum(n, (n, -oo, 1))) == \
"""\
  1     \n\
 ___    \n\
 ╲      \n\
  ╲     \n\
  ╱    n\n\
 ╱      \n\
 ‾‾‾    \n\
n = -∞  \
"""


def test_is_combining():
    line = "v̇_m"
    assert [is_combining(sym) for sym in line] == \
        [False, True, False, False]


def test_issue_17616():
    assert pretty(pi**(1/exp(1))) == \
   '  / -1\\\n'\
   '  \\e  /\n'\
   'pi     '

    assert upretty(pi**(1/exp(1))) == \
   ' ⎛ -1⎞\n'\
   ' ⎝ℯ  ⎠\n'\
   'π     '

    assert pretty(pi**(1/pi)) == \
    '  1 \n'\
    '  --\n'\
    '  pi\n'\
    'pi  '

    assert upretty(pi**(1/pi)) == \
    ' 1\n'\
    ' ─\n'\
    ' π\n'\
    'π '

    assert pretty(pi**(1/EulerGamma)) == \
    '      1     \n'\
    '  ----------\n'\
    '  EulerGamma\n'\
    'pi          '

    assert upretty(pi**(1/EulerGamma)) == \
    ' 1\n'\
    ' ─\n'\
    ' γ\n'\
    'π '

    z = Symbol("x_17")
    assert upretty(7**(1/z)) == \
    'x₁₇___\n'\
    ' ╲╱ 7 '

    assert pretty(7**(1/z)) == \
    'x_17___\n'\
    '  \\/ 7 '


def test_issue_17857():
    assert pretty(Range(-oo, oo)) == '{..., -1, 0, 1, ...}'
    assert pretty(Range(oo, -oo, -1)) == '{..., 1, 0, -1, ...}'


def test_issue_18272():
    x = Symbol('x')
    n = Symbol('n')

    assert upretty(ConditionSet(x, Eq(-x + exp(x), 0), S.Complexes)) == \
    '⎧  │         ⎛      x    ⎞⎫\n'\
    '⎨x │ x ∊ ℂ ∧ ⎝-x + ℯ  = 0⎠⎬\n'\
    '⎩  │                      ⎭'
    assert upretty(ConditionSet(x, Contains(n/2, Interval(0, oo)), FiniteSet(-n/2, n/2))) == \
    '⎧  │     ⎧-n   n⎫   ⎛n         ⎞⎫\n'\
    '⎨x │ x ∊ ⎨───, ─⎬ ∧ ⎜─ ∈ [0, ∞)⎟⎬\n'\
    '⎩  │     ⎩ 2   2⎭   ⎝2         ⎠⎭'
    assert upretty(ConditionSet(x, Eq(Piecewise((1, x >= 3), (x/2 - 1/2, x >= 2), (1/2, x >= 1),
                (x/2, True)) - 1/2, 0), Interval(0, 3))) == \
    '⎧  │              ⎛⎛⎧   1     for x ≥ 3⎞          ⎞⎫\n'\
    '⎪  │              ⎜⎜⎪                  ⎟          ⎟⎪\n'\
    '⎪  │              ⎜⎜⎪x                 ⎟          ⎟⎪\n'\
    '⎪  │              ⎜⎜⎪─ - 0.5  for x ≥ 2⎟          ⎟⎪\n'\
    '⎪  │              ⎜⎜⎪2                 ⎟          ⎟⎪\n'\
    '⎨x │ x ∊ [0, 3] ∧ ⎜⎜⎨                  ⎟ - 0.5 = 0⎟⎬\n'\
    '⎪  │              ⎜⎜⎪  0.5    for x ≥ 1⎟          ⎟⎪\n'\
    '⎪  │              ⎜⎜⎪                  ⎟          ⎟⎪\n'\
    '⎪  │              ⎜⎜⎪   x              ⎟          ⎟⎪\n'\
    '⎪  │              ⎜⎜⎪   ─     otherwise⎟          ⎟⎪\n'\
    '⎩  │              ⎝⎝⎩   2              ⎠          ⎠⎭'


def test_Str():
    from sympy.core.symbol import Str
    assert pretty(Str('x')) == 'x'


def test_symbolic_probability():
    mu = symbols("mu")
    sigma = symbols("sigma", positive=True)
    X = Normal("X", mu, sigma)
    assert pretty(Expectation(X)) == r'E[X]'
    assert pretty(Variance(X)) == r'Var(X)'
    assert pretty(Probability(X > 0)) == r'P(X > 0)'
    Y = Normal("Y", mu, sigma)
    assert pretty(Covariance(X, Y)) == 'Cov(X, Y)'


def test_issue_21758():
    from sympy.functions.elementary.piecewise import piecewise_fold
    from sympy.series.fourier import FourierSeries
    x = Symbol('x')
    k, n = symbols('k n')
    fo = FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)), SeqFormula(
        Piecewise((-2*pi*cos(n*pi)/n + 2*sin(n*pi)/n**2, (n > -oo) & (n < oo) & Ne(n, 0)),
                  (0, True))*sin(n*x)/pi, (n, 1, oo))))
    assert upretty(piecewise_fold(fo)) == \
        '⎧                      2⋅sin(3⋅x)                                \n'\
        '⎪2⋅sin(x) - sin(2⋅x) + ────────── + …  for n > -∞ ∧ n < ∞ ∧ n ≠ 0\n'\
        '⎨                          3                                     \n'\
        '⎪                                                                \n'\
        '⎩                 0                            otherwise         '
    assert pretty(FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)),
                                                 SeqFormula(0, (n, 1, oo))))) == '0'


def test_diffgeom():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField
    x,y = symbols('x y', real=True)
    m = Manifold('M', 2)
    assert pretty(m) == 'M'
    p = Patch('P', m)
    assert pretty(p) == "P"
    rect = CoordSystem('rect', p, [x, y])
    assert pretty(rect) == "rect"
    b = BaseScalarField(rect, 0)
    assert pretty(b) == "x"


def test_deprecated_prettyForm():
    with warns_deprecated_sympy():
        from sympy.printing.pretty.pretty_symbology import xstr
        assert xstr(1) == '1'

    with warns_deprecated_sympy():
        from sympy.printing.pretty.stringpict import prettyForm
        p = prettyForm('s', unicode='s')

    with warns_deprecated_sympy():
        assert p.unicode == p.s == 's'


def test_center():
    assert center('1', 2) == '1 '
    assert center('1', 3) == ' 1 '
    assert center('1', 3, '-') == '-1-'
    assert center('1', 5, '-') == '--1--'