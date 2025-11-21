
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\__init__.py ===
"""Printing subsystem"""

from .pretty import pager_print, pretty, pretty_print, pprint, pprint_use_unicode, pprint_try_use_unicode

from .latex import latex, print_latex, multiline_latex

from .mathml import mathml, print_mathml

from .python import python, print_python

from .pycode import pycode

from .codeprinter import print_ccode, print_fcode

from .codeprinter import ccode, fcode, cxxcode, rust_code # noqa:F811

from .smtlib import smtlib_code

from .glsl import glsl_code, print_glsl

from .rcode import rcode, print_rcode

from .jscode import jscode, print_jscode

from .julia import julia_code

from .mathematica import mathematica_code

from .octave import octave_code

from .gtk import print_gtk

from .preview import preview

from .repr import srepr

from .tree import print_tree

from .str import StrPrinter, sstr, sstrrepr

from .tableform import TableForm

from .dot import dotprint

from .maple import maple_code, print_maple_code

__all__ = [
    # sympy.printing.pretty
    'pager_print', 'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
    'pprint_try_use_unicode',

    # sympy.printing.latex
    'latex', 'print_latex', 'multiline_latex',

    # sympy.printing.mathml
    'mathml', 'print_mathml',

    # sympy.printing.python
    'python', 'print_python',

    # sympy.printing.pycode
    'pycode',

    # sympy.printing.codeprinter
    'ccode', 'print_ccode', 'cxxcode', 'fcode', 'print_fcode', 'rust_code',

    # sympy.printing.smtlib
    'smtlib_code',

    # sympy.printing.glsl
    'glsl_code', 'print_glsl',

    # sympy.printing.rcode
    'rcode', 'print_rcode',

    # sympy.printing.jscode
    'jscode', 'print_jscode',

    # sympy.printing.julia
    'julia_code',

    # sympy.printing.mathematica
    'mathematica_code',

    # sympy.printing.octave
    'octave_code',

    # sympy.printing.gtk
    'print_gtk',

    # sympy.printing.preview
    'preview',

    # sympy.printing.repr
    'srepr',

    # sympy.printing.tree
    'print_tree',

    # sympy.printing.str
    'StrPrinter', 'sstr', 'sstrrepr',

    # sympy.printing.tableform
    'TableForm',

    # sympy.printing.dot
    'dotprint',

    # sympy.printing.maple
    'maple_code', 'print_maple_code',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_biassplitgelu.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

from fusion_base import Fusion
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionBiasSplitGelu(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "BiasSplitGelu", "Gelu")

    def fuse(self, gelu_node, input_name_to_nodes: dict, output_name_to_node: dict):
        """
        [root] --->Add -------------------->  Slice ---------------> Mul -->
                   |                            ^                    ^
                   |                            |                    |
                   +----------------------------+---Slice --> Gelu---+
                   |                            |     ^
                   |                            |-----|
                   |                            |     |
                   |                           Mul   Mul
                   |                            ^     ^
                   v                            |     |
                  Shape ---> Gather --> Add --> Div --+
        """
        if gelu_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[gelu_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_after_gelu = children[0]

        slice_before_gelu = self.model.match_parent(gelu_node, "Slice", 0, output_name_to_node)
        if slice_before_gelu is None:
            return

        if self.model.find_constant_input(slice_before_gelu, -1, delta=0.001) != 3:
            return

        add_output = slice_before_gelu.input[0]

        start_index_nodes = self.model.match_parent_path(
            slice_before_gelu,
            ["Div", "Add", "Gather", "Shape", "Add"],
            [1, 0, 0, 0, 0],
            output_name_to_node,  # Mul(1) is optional
        )
        if start_index_nodes is None:
            start_index_nodes = self.model.match_parent_path(
                slice_before_gelu,
                ["Mul", "Div", "Add", "Gather", "Shape", "Add"],
                [1, 0, 0, 0, 0, 0],
                output_name_to_node,
            )

        if start_index_nodes is None or start_index_nodes[-2].input[0] != add_output:
            return

        end_index_nodes = self.model.match_parent_path(slice_before_gelu, ["Mul", "Div"], [2, 0], output_name_to_node)

        if (
            end_index_nodes is None or end_index_nodes[1] not in start_index_nodes
        ):  # the Div is parent of both two Mul nodes
            return

        slice_before_mul = self.model.match_parent(mul_after_gelu, "Slice", 0, output_name_to_node)
        if slice_before_mul is None:
            return

        if (
            slice_before_mul.input[2] != slice_before_gelu.input[1]
        ):  # end index of slice_before_mul is start index of slice_before_gelu
            return

        subgraph_nodes = [
            *start_index_nodes,
            end_index_nodes[0],
            mul_after_gelu,
            gelu_node,
            slice_before_mul,
            slice_before_gelu,
        ]
        subgraph_output = mul_after_gelu.output[0]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes, [subgraph_output], input_name_to_nodes, output_name_to_node
        ):
            logger.info("Skip fuse BiasSplitGelu since it is not safe to fuse the subgraph.")
            return

        add_node = start_index_nodes[-1]
        bias_index, _value = self.model.get_constant_input(add_node)
        if not isinstance(bias_index, int):
            return
        self.nodes_to_remove.extend(subgraph_nodes)
        node_name = self.model.create_node_name("BiasSplitGelu", name_prefix="BiasSplitGelu")
        fused_node = helper.make_node(
            "BiasSplitGelu",
            inputs=[add_node.input[1 - bias_index], add_node.input[bias_index]],
            outputs=[subgraph_output],
            name=node_name,
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[node_name] = self.this_graph_name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\test_solveset.py ===
from math import isclose

from sympy.calculus.util import stationary_points
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda, nfloat, diff)
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, Integer, all_close)
from sympy.core.relational import (Eq, Gt, Ne, Ge)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign, conjugate)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
    sinh, cosh, tanh, coth, sech, csch, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
    TrigonometricFunction, acos, acot, acsc, asec, asin, atan, atan2,
    cos, cot, csc, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc,
    erfcinv, erfinv)
from sympy.logic.boolalg import And
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import ImageSet, Range
from sympy.sets.sets import (Complement, FiniteSet,
    Intersection, Interval, Union, imageset, ProductSet)
from sympy.simplify import simplify
from sympy.tensor.indexed import Indexed
from sympy.utilities.iterables import numbered_symbols

from sympy.testing.pytest import (XFAIL, raises, skip, slow, SKIP, _both_exp_pow)
from sympy.core.random import verify_numerically as tn
from sympy.physics.units import cm

from sympy.solvers import solve
from sympy.solvers.solveset import (
    solveset_real, domain_check, solveset_complex, linear_eq_to_matrix,
    linsolve, _is_function_class_equation, invert_real, invert_complex,
    _invert_trig_hyp_real, solveset, solve_decomposition, substitution,
    nonlinsolve, solvify,
    _is_finite_with_finite_vars, _transolve, _is_exponential,
    _solve_exponential, _is_logarithmic, _is_lambert,
    _solve_logarithm, _term_factors, _is_modular, NonlinearError)

from sympy.abc import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, q, r,
    t, w, x, y, z)


def dumeq(i, j):
    if type(i) in (list, tuple):
        return all(dumeq(i, j) for i, j in zip(i, j))
    return i == j or i.dummy_eq(j)


def assert_close_ss(sol1, sol2):
    """Test solutions with floats from solveset are close"""
    sol1 = sympify(sol1)
    sol2 = sympify(sol2)
    assert isinstance(sol1, FiniteSet)
    assert isinstance(sol2, FiniteSet)
    assert len(sol1) == len(sol2)
    assert all(isclose(v1, v2) for v1, v2 in zip(sol1, sol2))


def assert_close_nl(sol1, sol2):
    """Test solutions with floats from nonlinsolve are close"""
    sol1 = sympify(sol1)
    sol2 = sympify(sol2)
    assert isinstance(sol1, FiniteSet)
    assert isinstance(sol2, FiniteSet)
    assert len(sol1) == len(sol2)
    for s1, s2 in zip(sol1, sol2):
        assert len(s1) == len(s2)
        assert all(isclose(v1, v2) for v1, v2 in zip(s1, s2))


@_both_exp_pow
def test_invert_real():
    x = Symbol('x', real=True)

    def ireal(x, s=S.Reals):
        return Intersection(s, x)

    assert invert_real(exp(x), z, x) == (x, ireal(FiniteSet(log(z))))

    y = Symbol('y', positive=True)
    n = Symbol('n', real=True)
    assert invert_real(x + 3, y, x) == (x, FiniteSet(y - 3))
    assert invert_real(x*3, y, x) == (x, FiniteSet(y / 3))

    assert invert_real(exp(x), y, x) == (x, FiniteSet(log(y)))
    assert invert_real(exp(3*x), y, x) == (x, FiniteSet(log(y) / 3))
    assert invert_real(exp(x + 3), y, x) == (x, FiniteSet(log(y) - 3))

    assert invert_real(exp(x) + 3, y, x) == (x, ireal(FiniteSet(log(y - 3))))
    assert invert_real(exp(x)*3, y, x) == (x, FiniteSet(log(y / 3)))

    assert invert_real(log(x), y, x) == (x, FiniteSet(exp(y)))
    assert invert_real(log(3*x), y, x) == (x, FiniteSet(exp(y) / 3))
    assert invert_real(log(x + 3), y, x) == (x, FiniteSet(exp(y) - 3))

    assert invert_real(Abs(x), y, x) == (x, FiniteSet(y, -y))

    assert invert_real(2**x, y, x) == (x, FiniteSet(log(y)/log(2)))
    assert invert_real(2**exp(x), y, x) == (x, ireal(FiniteSet(log(log(y)/log(2)))))

    assert invert_real(x**2, y, x) == (x, FiniteSet(sqrt(y), -sqrt(y)))
    assert invert_real(x**S.Half, y, x) == (x, FiniteSet(y**2))

    raises(ValueError, lambda: invert_real(x, x, x))

    # issue 21236
    assert invert_real(x**pi, y, x) == (x, FiniteSet(y**(1/pi)))
    assert invert_real(x**pi, -E, x) == (x, S.EmptySet)
    assert invert_real(x**Rational(3/2), 1000, x) == (x, FiniteSet(100))
    assert invert_real(x**1.0, 1, x) == (x**1.0, FiniteSet(1))

    raises(ValueError, lambda: invert_real(S.One, y, x))

    assert invert_real(x**31 + x, y, x) == (x**31 + x, FiniteSet(y))

    lhs = x**31 + x
    base_values =  FiniteSet(y - 1, -y - 1)
    assert invert_real(Abs(x**31 + x + 1), y, x) == (lhs, base_values)

    assert dumeq(invert_real(sin(x), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi + asin(y)), S.Integers),
            ImageSet(Lambda(n, pi*2*n + pi - asin(y)), S.Integers)))))

    assert dumeq(invert_real(sin(exp(x)), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, log(2*n*pi + asin(y))), S.Integers),
            ImageSet(Lambda(n, log(pi*2*n + pi - asin(y))), S.Integers)))))

    assert dumeq(invert_real(csc(x), y, x), (x,
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            Union(ImageSet(Lambda(n, 2*n*pi + acsc(y)), S.Integers),
                ImageSet(Lambda(n, 2*n*pi - acsc(y) + pi), S.Integers)))))

    assert dumeq(invert_real(csc(exp(x)), y, x), (x,
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            Union(ImageSet(Lambda(n, log(2*n*pi + acsc(y))), S.Integers),
                ImageSet(Lambda(n, log(2*n*pi - acsc(y) + pi)), S.Integers)))))

    assert dumeq(invert_real(cos(x), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi + acos(y)), S.Integers),
            ImageSet(Lambda(n, 2*n*pi - acos(y)), S.Integers)))))

    assert dumeq(invert_real(cos(exp(x)), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, log(2*n*pi + acos(y))), S.Integers),
            ImageSet(Lambda(n, log(2*n*pi - acos(y))), S.Integers)))))

    assert dumeq(invert_real(sec(x), y, x), (x,
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            Union(ImageSet(Lambda(n, 2*n*pi + asec(y)), S.Integers), \
                ImageSet(Lambda(n, 2*n*pi - asec(y)), S.Integers)))))

    assert dumeq(invert_real(sec(exp(x)), y, x), (x,
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            Union(ImageSet(Lambda(n, log(2*n*pi - asec(y))), S.Integers),
                ImageSet(Lambda(n, log(2*n*pi + asec(y))), S.Integers)))))

    assert dumeq(invert_real(tan(x), y, x), (x,
        ConditionSet(x, (-oo < y) & (y < oo),
            ImageSet(Lambda(n, n*pi + atan(y)), S.Integers))))

    assert dumeq(invert_real(tan(exp(x)), y, x), (x,
        ConditionSet(x, (-oo < y) & (y < oo),
            ImageSet(Lambda(n, log(n*pi + atan(y))), S.Integers))))

    assert dumeq(invert_real(cot(x), y, x), (x,
        ConditionSet(x, (-oo < y) & (y < oo),
            ImageSet(Lambda(n, n*pi + acot(y)), S.Integers))))

    assert dumeq(invert_real(cot(exp(x)), y, x), (x,
        ConditionSet(x, (-oo < y) & (y < oo),
            ImageSet(Lambda(n, log(n*pi + acot(y))), S.Integers))))

    assert dumeq(invert_real(tan(tan(x)), y, x),
        (x, ConditionSet(x, Eq(tan(tan(x)), y), S.Reals)))
        # slight regression compared to previous result:
        # (tan(x), imageset(Lambda(n, n*pi + atan(y)), S.Integers)))

    x = Symbol('x', positive=True)
    assert invert_real(x**pi, y, x) == (x, FiniteSet(y**(1/pi)))

    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    assert invert_real(sinh(x), r, x) == (x, FiniteSet(asinh(r)))
    assert invert_real(sinh(log(x)), p, x) == (x, FiniteSet(exp(asinh(p))))

    assert invert_real(cosh(x), r, x) == (x, Intersection(
        FiniteSet(-acosh(r), acosh(r)), S.Reals))
    assert invert_real(cosh(x), p + 1, x) == (x,
        FiniteSet(-acosh(p + 1), acosh(p + 1)))

    assert invert_real(tanh(x), r, x) == (x, Intersection(FiniteSet(atanh(r)), S.Reals))
    assert invert_real(coth(x), p+1, x) == (x, FiniteSet(acoth(p+1)))
    assert invert_real(sech(x), r, x) == (x, Intersection(
        FiniteSet(-asech(r), asech(r)), S.Reals))
    assert invert_real(csch(x), p, x) == (x, FiniteSet(acsch(p)))

    assert dumeq(invert_real(tanh(sin(x)), r, x), (x,
        ConditionSet(x, (S(-1) <= atanh(r)) & (atanh(r) <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi + asin(atanh(r))), S.Integers),
            ImageSet(Lambda(n, 2*n*pi - asin(atanh(r)) + pi), S.Integers)))))


def test_invert_trig_hyp_real():
    # check some codepaths that are not as easily reached otherwise
    n = Dummy('n')
    assert _invert_trig_hyp_real(cosh(x), Range(-5, 10, 1), x)[1].dummy_eq(Union(
        ImageSet(Lambda(n, -acosh(n)), Range(1, 10, 1)),
        ImageSet(Lambda(n, acosh(n)), Range(1, 10, 1))))
    assert _invert_trig_hyp_real(coth(x), Interval(-3, 2), x) == (x, Union(
        Interval(-oo, -acoth(3)), Interval(acoth(2), oo)))
    assert _invert_trig_hyp_real(tanh(x), Interval(-S.Half, 1), x) == (x,
        Interval(-atanh(S.Half), oo))
    assert _invert_trig_hyp_real(sech(x), imageset(n, S.Half + n/3, S.Naturals0), x) == \
        (x, FiniteSet(-asech(S(1)/2), asech(S(1)/2), -asech(S(5)/6), asech(S(5)/6)))
    assert _invert_trig_hyp_real(csch(x), S.Reals, x) == (x,
        Union(Interval.open(-oo, 0), Interval.open(0, oo)))


def test_invert_complex():
    assert invert_complex(x + 3, y, x) == (x, FiniteSet(y - 3))
    assert invert_complex(x*3, y, x) == (x, FiniteSet(y / 3))
    assert invert_complex((x - 1)**3, 0, x) == (x, FiniteSet(1))

    assert dumeq(invert_complex(exp(x), y, x),
        (x, imageset(Lambda(n, I*(2*pi*n + arg(y)) + log(Abs(y))), S.Integers)))

    assert invert_complex(log(x), y, x) == (x, FiniteSet(exp(y)))

    raises(ValueError, lambda: invert_real(1, y, x))
    raises(ValueError, lambda: invert_complex(x, x, x))
    raises(ValueError, lambda: invert_complex(x, x, 1))

    assert dumeq(invert_complex(sin(x), I, x), (x, Union(
        ImageSet(Lambda(n, 2*n*pi + I*log(1 + sqrt(2))), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi - I*log(1 + sqrt(2))), S.Integers))))
    assert dumeq(invert_complex(cos(x), 1+I, x), (x, Union(
        ImageSet(Lambda(n, 2*n*pi - acos(1 + I)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + acos(1 + I)), S.Integers))))
    assert dumeq(invert_complex(tan(2*x), 1, x), (x,
        ImageSet(Lambda(n, n*pi/2 + pi/8), S.Integers)))
    assert dumeq(invert_complex(cot(x), 2*I, x), (x,
        ImageSet(Lambda(n, n*pi - I*acoth(2)), S.Integers)))

    assert dumeq(invert_complex(sinh(x), 0, x), (x, Union(
        ImageSet(Lambda(n, 2*n*I*pi), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + I*pi), S.Integers))))
    assert dumeq(invert_complex(cosh(x), 0, x), (x, Union(
        ImageSet(Lambda(n, 2*n*I*pi + I*pi/2), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + 3*I*pi/2), S.Integers))))
    assert invert_complex(tanh(x), 1, x) == (x, S.EmptySet)
    assert dumeq(invert_complex(tanh(x), a, x), (x,
        ConditionSet(x, Ne(a, -1) & Ne(a, 1),
        ImageSet(Lambda(n, n*I*pi + atanh(a)), S.Integers))))
    assert invert_complex(coth(x), 1, x) == (x, S.EmptySet)
    assert dumeq(invert_complex(coth(x), a, x), (x,
        ConditionSet(x, Ne(a, -1) & Ne(a, 1),
        ImageSet(Lambda(n, n*I*pi + acoth(a)), S.Integers))))
    assert dumeq(invert_complex(sech(x), 2, x), (x, Union(
        ImageSet(Lambda(n, 2*n*I*pi + I*pi/3), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + 5*I*pi/3), S.Integers))))


def test_domain_check():
    assert domain_check(1/(1 + (1/(x+1))**2), x, -1) is False
    assert domain_check(x**2, x, 0) is True
    assert domain_check(x, x, oo) is False
    assert domain_check(0, x, oo) is False


def test_issue_11536():
    assert solveset(0**x - 100, x, S.Reals) == S.EmptySet
    assert solveset(0**x - 1, x, S.Reals) == FiniteSet(0)


def test_issue_17479():
    f = (x**2 + y**2)**2 + (x**2 + z**2)**2 - 2*(2*x**2 + y**2 + z**2)
    fx = f.diff(x)
    fy = f.diff(y)
    fz = f.diff(z)
    sol = nonlinsolve([fx, fy, fz], [x, y, z])
    assert len(sol) >= 4 and len(sol) <= 20
    # nonlinsolve has been giving a varying number of solutions
    # (originally 18, then 20, now 19) due to various internal changes.
    # Unfortunately not all the solutions are actually valid and some are
    # redundant. Since the original issue was that an exception was raised,
    # this first test only checks that nonlinsolve returns a "plausible"
    # solution set. The next test checks the result for correctness.


@XFAIL
def test_issue_18449():
    x, y, z = symbols("x, y, z")
    f = (x**2 + y**2)**2 + (x**2 + z**2)**2 - 2*(2*x**2 + y**2 + z**2)
    fx = diff(f, x)
    fy = diff(f, y)
    fz = diff(f, z)
    sol = nonlinsolve([fx, fy, fz], [x, y, z])
    for (xs, ys, zs) in sol:
        d = {x: xs, y: ys, z: zs}
        assert tuple(_.subs(d).simplify() for _ in (fx, fy, fz)) == (0, 0, 0)
    # After simplification and removal of duplicate elements, there should
    # only be 4 parametric solutions left:
    # simplifiedsolutions = FiniteSet((sqrt(1 - z**2), z, z),
    #                                 (-sqrt(1 - z**2), z, z),
    #                                 (sqrt(1 - z**2), -z, z),
    #                                 (-sqrt(1 - z**2), -z, z))
    # TODO: Is the above solution set definitely complete?


def test_issue_21047():
    f = (2 - x)**2 + (sqrt(x - 1) - 1)**6
    assert solveset(f, x, S.Reals) == FiniteSet(2)

    f = (sqrt(x)-1)**2 + (sqrt(x)+1)**2 -2*x**2 + sqrt(2)
    assert solveset(f, x, S.Reals) == FiniteSet(
        S.Half - sqrt(2*sqrt(2) + 5)/2, S.Half + sqrt(2*sqrt(2) + 5)/2)


def test_is_function_class_equation():
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) - 1, x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) + sin(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) + sin(x) - a, x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       sin(x)*tan(x) + sin(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       sin(x)*tan(x + a) + sin(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       sin(x)*tan(x*a) + sin(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       a*tan(x) - 1, x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x)**2 + sin(x) - 1, x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) + x, x) is False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x**2), x) is False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x**2) + sin(x), x) is False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x)**sin(x), x) is False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(sin(x)) + sin(x), x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) - 1, x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) + sinh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) + sinh(x) - a, x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       sinh(x)*tanh(x) + sinh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       sinh(x)*tanh(x + a) + sinh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       sinh(x)*tanh(x*a) + sinh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       a*tanh(x) - 1, x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x)**2 + sinh(x) - 1, x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) + x, x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x**2), x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x**2) + sinh(x), x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x)**sinh(x), x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(sinh(x)) + sinh(x), x) is False


def test_garbage_input():
    raises(ValueError, lambda: solveset_real([y], y))
    x = Symbol('x', real=True)
    assert solveset_real(x, 1) == S.EmptySet
    assert solveset_real(x - 1, 1) == FiniteSet(x)
    assert solveset_real(x, pi) == S.EmptySet
    assert solveset_real(x, x**2) == S.EmptySet

    raises(ValueError, lambda: solveset_complex([x], x))
    assert solveset_complex(x, pi) == S.EmptySet

    raises(ValueError, lambda: solveset((x, y), x))
    raises(ValueError, lambda: solveset(x + 1, S.Reals))
    raises(ValueError, lambda: solveset(x + 1, x, 2))


def test_solve_mul():
    assert solveset_real((a*x + b)*(exp(x) - 3), x) == \
        Union({log(3)}, Intersection({-b/a}, S.Reals))
    anz = Symbol('anz', nonzero=True)
    bb = Symbol('bb', real=True)
    assert solveset_real((anz*x + bb)*(exp(x) - 3), x) == \
        FiniteSet(-bb/anz, log(3))
    assert solveset_real((2*x + 8)*(8 + exp(x)), x) == FiniteSet(S(-4))
    assert solveset_real(x/log(x), x) is S.EmptySet


def test_solve_invert():
    assert solveset_real(exp(x) - 3, x) == FiniteSet(log(3))
    assert solveset_real(log(x) - 3, x) == FiniteSet(exp(3))

    assert solveset_real(3**(x + 2), x) == FiniteSet()
    assert solveset_real(3**(2 - x), x) == FiniteSet()

    assert solveset_real(y - b*exp(a/x), x) == Intersection(
        S.Reals, FiniteSet(a/log(y/b)))

    # issue 4504
    assert solveset_real(2**x - 10, x) == FiniteSet(1 + log(5)/log(2))


def test_issue_25768():
    assert dumeq(solveset_real(sin(x) - S.Half, x), Union(
        ImageSet(Lambda(n, pi*2*n + pi/6), S.Integers),
        ImageSet(Lambda(n, pi*2*n + pi*5/6), S.Integers)))
    n1 = solveset_real(sin(x) - 0.5, x).n(5)
    n2 = solveset_real(sin(x) - S.Half, x).n(5)
    # help pass despite fp differences
    eq = [i.replace(
        lambda x:x.is_Float,
        lambda x:Rational(x).limit_denominator(1000)) for i in (n1, n2)]
    assert dumeq(*eq),(n1,n2)


def test_errorinverses():
    assert solveset_real(erf(x) - S.Half, x) == \
        FiniteSet(erfinv(S.Half))
    assert solveset_real(erfinv(x) - 2, x) == \
        FiniteSet(erf(2))
    assert solveset_real(erfc(x) - S.One, x) == \
        FiniteSet(erfcinv(S.One))
    assert solveset_real(erfcinv(x) - 2, x) == FiniteSet(erfc(2))


def test_solve_polynomial():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert solveset_real(3*x - 2, x) == FiniteSet(Rational(2, 3))

    assert solveset_real(x**2 - 1, x) == FiniteSet(-S.One, S.One)
    assert solveset_real(x - y**3, x) == FiniteSet(y ** 3)

    assert solveset_real(x**3 - 15*x - 4, x) == FiniteSet(
        -2 + 3 ** S.Half,
        S(4),
        -2 - 3 ** S.Half)

    assert solveset_real(sqrt(x) - 1, x) == FiniteSet(1)
    assert solveset_real(sqrt(x) - 2, x) == FiniteSet(4)
    assert solveset_real(x**Rational(1, 4) - 2, x) == FiniteSet(16)
    assert solveset_real(x**Rational(1, 3) - 3, x) == FiniteSet(27)
    assert len(solveset_real(x**5 + x**3 + 1, x)) == 1
    assert len(solveset_real(-2*x**3 + 4*x**2 - 2*x + 6, x)) > 0
    assert solveset_real(x**6 + x**4 + I, x) is S.EmptySet


def test_return_root_of():
    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    s = list(solveset_complex(f, x))
    for root in s:
        assert root.func == CRootOf

    # if one uses solve to get the roots of a polynomial that has a CRootOf
    # solution, make sure that the use of nfloat during the solve process
    # doesn't fail. Note: if you want numerical solutions to a polynomial
    # it is *much* faster to use nroots to get them than to solve the
    # equation only to get CRootOf solutions which are then numerically
    # evaluated. So for eq = x**5 + 3*x + 7 do Poly(eq).nroots() rather
    # than [i.n() for i in solve(eq)] to get the numerical roots of eq.
    assert nfloat(list(solveset_complex(x**5 + 3*x**3 + 7, x))[0],
                  exponent=False) == CRootOf(x**5 + 3*x**3 + 7, 0).n()

    sol = list(solveset_complex(x**6 - 2*x + 2, x))
    assert all(isinstance(i, CRootOf) for i in sol) and len(sol) == 6

    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    s = list(solveset_complex(f, x))
    for root in s:
        assert root.func == CRootOf

    s = x**5 + 4*x**3 + 3*x**2 + Rational(7, 4)
    assert solveset_complex(s, x) == \
        FiniteSet(*Poly(s*4, domain='ZZ').all_roots())

    # Refer issue #7876
    eq = x*(x - 1)**2*(x + 1)*(x**6 - x + 1)
    assert solveset_complex(eq, x) == \
        FiniteSet(-1, 0, 1, CRootOf(x**6 - x + 1, 0),
                       CRootOf(x**6 - x + 1, 1),
                       CRootOf(x**6 - x + 1, 2),
                       CRootOf(x**6 - x + 1, 3),
                       CRootOf(x**6 - x + 1, 4),
                       CRootOf(x**6 - x + 1, 5))


def test_solveset_sqrt_1():
    assert solveset_real(sqrt(5*x + 6) - 2 - x, x) == \
        FiniteSet(-S.One, S(2))
    assert solveset_real(sqrt(x - 1) - x + 7, x) == FiniteSet(10)
    assert solveset_real(sqrt(x - 2) - 5, x) == FiniteSet(27)
    assert solveset_real(sqrt(x) - 2 - 5, x) == FiniteSet(49)
    assert solveset_real(sqrt(x**3), x) == FiniteSet(0)
    assert solveset_real(sqrt(x - 1), x) == FiniteSet(1)
    assert solveset_real(sqrt((x-3)/x), x) == FiniteSet(3)
    assert solveset_real(sqrt((x-3)/x)-Rational(1, 2), x) == \
        FiniteSet(4)

def test_solveset_sqrt_2():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    # http://tutorial.math.lamar.edu/Classes/Alg/SolveRadicalEqns.aspx#Solve_Rad_Ex2_a
    assert solveset_real(sqrt(2*x - 1) - sqrt(x - 4) - 2, x) == \
        FiniteSet(S(5), S(13))
    assert solveset_real(sqrt(x + 7) + 2 - sqrt(3 - x), x) == \
        FiniteSet(-6)

    # http://www.purplemath.com/modules/solverad.htm
    assert solveset_real(sqrt(17*x - sqrt(x**2 - 5)) - 7, x) == \
        FiniteSet(3)

    eq = x + 1 - (x**4 + 4*x**3 - x)**Rational(1, 4)
    assert solveset_real(eq, x) == FiniteSet(Rational(-1, 2), Rational(-1, 3))

    eq = sqrt(2*x + 9) - sqrt(x + 1) - sqrt(x + 4)
    assert solveset_real(eq, x) == FiniteSet(0)

    eq = sqrt(x + 4) + sqrt(2*x - 1) - 3*sqrt(x - 1)
    assert solveset_real(eq, x) == FiniteSet(5)

    eq = sqrt(x)*sqrt(x - 7) - 12
    assert solveset_real(eq, x) == FiniteSet(16)

    eq = sqrt(x - 3) + sqrt(x) - 3
    assert solveset_real(eq, x) == FiniteSet(4)

    eq = sqrt(2*x**2 - 7) - (3 - x)
    assert solveset_real(eq, x) == FiniteSet(-S(8), S(2))

    # others
    eq = sqrt(9*x**2 + 4) - (3*x + 2)
    assert solveset_real(eq, x) == FiniteSet(0)

    assert solveset_real(sqrt(x - 3) - sqrt(x) - 3, x) == FiniteSet()

    eq = (2*x - 5)**Rational(1, 3) - 3
    assert solveset_real(eq, x) == FiniteSet(16)

    assert solveset_real(sqrt(x) + sqrt(sqrt(x)) - 4, x) == \
        FiniteSet((Rational(-1, 2) + sqrt(17)/2)**4)

    eq = sqrt(x) - sqrt(x - 1) + sqrt(sqrt(x))
    assert solveset_real(eq, x) == FiniteSet()

    eq = (x - 4)**2 + (sqrt(x) - 2)**4
    assert solveset_real(eq, x) == FiniteSet(-4, 4)

    eq = (sqrt(x) + sqrt(x + 1) + sqrt(1 - x) - 6*sqrt(5)/5)
    ans = solveset_real(eq, x)
    ra = S('''-1484/375 - 4*(-S(1)/2 + sqrt(3)*I/2)*(-12459439/52734375 +
    114*sqrt(12657)/78125)**(S(1)/3) - 172564/(140625*(-S(1)/2 +
    sqrt(3)*I/2)*(-12459439/52734375 + 114*sqrt(12657)/78125)**(S(1)/3))''')
    rb = Rational(4, 5)
    assert all(abs(eq.subs(x, i).n()) < 1e-10 for i in (ra, rb)) and \
        len(ans) == 2 and \
        {i.n(chop=True) for i in ans} == \
        {i.n(chop=True) for i in (ra, rb)}

    assert solveset_real(sqrt(x) + x**Rational(1, 3) +
                                 x**Rational(1, 4), x) == FiniteSet(0)

    assert solveset_real(x/sqrt(x**2 + 1), x) == FiniteSet(0)

    eq = (x - y**3)/((y**2)*sqrt(1 - y**2))
    assert solveset_real(eq, x) == FiniteSet(y**3)

    # issue 4497
    assert solveset_real(1/(5 + x)**Rational(1, 5) - 9, x) == \
        FiniteSet(Rational(-295244, 59049))


@XFAIL
def test_solve_sqrt_fail():
    # this only works if we check real_root(eq.subs(x, Rational(1, 3)))
    # but checksol doesn't work like that
    eq = (x**3 - 3*x**2)**Rational(1, 3) + 1 - x
    assert solveset_real(eq, x) == FiniteSet(Rational(1, 3))


@slow
def test_solve_sqrt_3():
    R = Symbol('R')
    eq = sqrt(2)*R*sqrt(1/(R + 1)) + (R + 1)*(sqrt(2)*sqrt(1/(R + 1)) - 1)
    sol = solveset_complex(eq, R)
    fset = [Rational(5, 3) + 4*sqrt(10)*cos(atan(3*sqrt(111)/251)/3)/3,
            -sqrt(10)*cos(atan(3*sqrt(111)/251)/3)/3 +
            40*re(1/((Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9 +
            sqrt(30)*sin(atan(3*sqrt(111)/251)/3)/3 + Rational(5, 3) +
            I*(-sqrt(30)*cos(atan(3*sqrt(111)/251)/3)/3 -
               sqrt(10)*sin(atan(3*sqrt(111)/251)/3)/3 +
               40*im(1/((Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9)]
    cset = [40*re(1/((Rational(-1, 2) + sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9 -
            sqrt(10)*cos(atan(3*sqrt(111)/251)/3)/3 - sqrt(30)*sin(atan(3*sqrt(111)/251)/3)/3 +
            Rational(5, 3) +
            I*(40*im(1/((Rational(-1, 2) + sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9 -
               sqrt(10)*sin(atan(3*sqrt(111)/251)/3)/3 +
               sqrt(30)*cos(atan(3*sqrt(111)/251)/3)/3)]

    fs = FiniteSet(*fset)
    cs = ConditionSet(R, Eq(eq, 0), FiniteSet(*cset))
    assert sol == (fs - {-1}) | (cs - {-1})

    # the number of real roots will depend on the value of m: for m=1 there are 4
    # and for m=-1 there are none.
    eq = -sqrt((m - q)**2 + (-m/(2*q) + S.Half)**2) + sqrt((-m**2/2 - sqrt(
        4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2 + (m**2/2 - m - sqrt(
            4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2)
    unsolved_object = ConditionSet(q, Eq(sqrt((m - q)**2 + (-m/(2*q) + S.Half)**2) -
        sqrt((-m**2/2 - sqrt(4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2 + (m**2/2 - m -
        sqrt(4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2), 0), S.Reals)
    assert solveset_real(eq, q) == unsolved_object


def test_solve_polynomial_symbolic_param():
    assert solveset_complex((x**2 - 1)**2 - a, x) == \
        FiniteSet(sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
                  sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a)))

    # issue 4507
    assert solveset_complex(y - b/(1 + a*x), x) == \
        FiniteSet((b/y - 1)/a) - FiniteSet(-1/a)

    # issue 4508
    assert solveset_complex(y - b*x/(a + x), x) == \
        FiniteSet(-a*y/(y - b)) - FiniteSet(-a)


def test_solve_rational():
    assert solveset_real(1/x + 1, x) == FiniteSet(-S.One)
    assert solveset_real(1/exp(x) - 1, x) == FiniteSet(0)
    assert solveset_real(x*(1 - 5/x), x) == FiniteSet(5)
    assert solveset_real(2*x/(x + 2) - 1, x) == FiniteSet(2)
    assert solveset_real((x**2/(7 - x)).diff(x), x) == \
        FiniteSet(S.Zero, S(14))


def test_solveset_real_gen_is_pow():
    assert solveset_real(sqrt(1) + 1, x) is S.EmptySet


def test_no_sol():
    assert solveset(1 - oo*x) is S.EmptySet
    assert solveset(oo*x, x) is S.EmptySet
    assert solveset(oo*x - oo, x) is S.EmptySet
    assert solveset_real(4, x) is S.EmptySet
    assert solveset_real(exp(x), x) is S.EmptySet
    assert solveset_real(x**2 + 1, x) is S.EmptySet
    assert solveset_real(-3*a/sqrt(x), x) is S.EmptySet
    assert solveset_real(1/x, x) is S.EmptySet
    assert solveset_real(-(1 + x)/(2 + x)**2 + 1/(2 + x), x
        ) is S.EmptySet


def test_sol_zero_real():
    assert solveset_real(0, x) == S.Reals
    assert solveset(0, x, Interval(1, 2)) == Interval(1, 2)
    assert solveset_real(-x**2 - 2*x + (x + 1)**2 - 1, x) == S.Reals


def test_no_sol_rational_extragenous():
    assert solveset_real((x/(x + 1) + 3)**(-2), x) is S.EmptySet
    assert solveset_real((x - 1)/(1 + 1/(x - 1)), x) is S.EmptySet


def test_solve_polynomial_cv_1a():
    """
    Test for solving on equations that can be converted to
    a polynomial equation using the change of variable y -> x**Rational(p, q)
    """
    assert solveset_real(sqrt(x) - 1, x) == FiniteSet(1)
    assert solveset_real(sqrt(x) - 2, x) == FiniteSet(4)
    assert solveset_real(x**Rational(1, 4) - 2, x) == FiniteSet(16)
    assert solveset_real(x**Rational(1, 3) - 3, x) == FiniteSet(27)
    assert solveset_real(x*(x**(S.One / 3) - 3), x) == \
        FiniteSet(S.Zero, S(27))


def test_solveset_real_rational():
    """Test solveset_real for rational functions"""
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert solveset_real((x - y**3) / ((y**2)*sqrt(1 - y**2)), x) \
        == FiniteSet(y**3)
    # issue 4486
    assert solveset_real(2*x/(x + 2) - 1, x) == FiniteSet(2)


def test_solveset_real_log():
    assert solveset_real(log((x-1)*(x+1)), x) == \
        FiniteSet(sqrt(2), -sqrt(2))


def test_poly_gens():
    assert solveset_real(4**(2*(x**2) + 2*x) - 8, x) == \
        FiniteSet(Rational(-3, 2), S.Half)


def test_solve_abs():
    n = Dummy('n')
    raises(ValueError, lambda: solveset(Abs(x) - 1, x))
    assert solveset(Abs(x) - n, x, S.Reals).dummy_eq(
        ConditionSet(x, Contains(n, Interval(0, oo)), {-n, n}))
    assert solveset_real(Abs(x) - 2, x) == FiniteSet(-2, 2)
    assert solveset_real(Abs(x) + 2, x) is S.EmptySet
    assert solveset_real(Abs(x + 3) - 2*Abs(x - 3), x) == \
        FiniteSet(1, 9)
    assert solveset_real(2*Abs(x) - Abs(x - 1), x) == \
        FiniteSet(-1, Rational(1, 3))

    sol = ConditionSet(
            x,
            And(
                Contains(b, Interval(0, oo)),
                Contains(a + b, Interval(0, oo)),
                Contains(a - b, Interval(0, oo))),
            FiniteSet(-a - b - 3, -a + b - 3, a - b - 3, a + b - 3))
    eq = Abs(Abs(x + 3) - a) - b
    assert invert_real(eq, 0, x)[1] == sol
    reps = {a: 3, b: 1}
    eqab = eq.subs(reps)
    for si in sol.subs(reps):
        assert not eqab.subs(x, si)
    assert dumeq(solveset(Eq(sin(Abs(x)), 1), x, domain=S.Reals), Union(
        Intersection(Interval(0, oo), Union(
        Intersection(ImageSet(Lambda(n, 2*n*pi + 3*pi/2), S.Integers),
            Interval(-oo, 0)),
        Intersection(ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers),
            Interval(0, oo))))))


def test_issue_9824():
    assert dumeq(solveset(sin(x)**2 - 2*sin(x) + 1, x), ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers))
    assert dumeq(solveset(cos(x)**2 - 2*cos(x) + 1, x), ImageSet(Lambda(n, 2*n*pi), S.Integers))


def test_issue_9565():
    assert solveset_real(Abs((x - 1)/(x - 5)) <= Rational(1, 3), x) == Interval(-1, 2)


def test_issue_10069():
    eq = abs(1/(x - 1)) - 1 > 0
    assert solveset_real(eq, x) == Union(
        Interval.open(0, 1), Interval.open(1, 2))


def test_real_imag_splitting():
    a, b = symbols('a b', real=True)
    assert solveset_real(sqrt(a**2 - b**2) - 3, a) == \
        FiniteSet(-sqrt(b**2 + 9), sqrt(b**2 + 9))
    assert solveset_real(sqrt(a**2 + b**2) - 3, a) != \
        S.EmptySet


def test_units():
    assert solveset_real(1/x - 1/(2*cm), x) == FiniteSet(2*cm)


def test_solve_only_exp_1():
    y = Symbol('y', positive=True)
    assert solveset_real(exp(x) - y, x) == FiniteSet(log(y))
    assert solveset_real(exp(x) + exp(-x) - 4, x) == \
        FiniteSet(log(-sqrt(3) + 2), log(sqrt(3) + 2))
    assert solveset_real(exp(x) + exp(-x) - y, x) != S.EmptySet


def test_atan2():
    # The .inverse() method on atan2 works only if x.is_real is True and the
    # second argument is a real constant
    assert solveset_real(atan2(x, 2) - pi/3, x) == FiniteSet(2*sqrt(3))


def test_piecewise_solveset():
    eq = Piecewise((x - 2, Gt(x, 2)), (2 - x, True)) - 3
    assert set(solveset_real(eq, x)) == set(FiniteSet(-1, 5))

    absxm3 = Piecewise(
        (x - 3, 0 <= x - 3),
        (3 - x, 0 > x - 3))
    y = Symbol('y', positive=True)
    assert solveset_real(absxm3 - y, x) == FiniteSet(-y + 3, y + 3)

    f = Piecewise(((x - 2)**2, x >= 0), (0, True))
    assert solveset(f, x, domain=S.Reals) == Union(FiniteSet(2), Interval(-oo, 0, True, True))

    assert solveset(
        Piecewise((x + 1, x > 0), (I, True)) - I, x, S.Reals
        ) == Interval(-oo, 0)

    assert solveset(Piecewise((x - 1, Ne(x, I)), (x, True)), x) == FiniteSet(1)

    # issue 19718
    g = Piecewise((1, x > 10), (0, True))
    assert solveset(g > 0, x, S.Reals) == Interval.open(10, oo)

    from sympy.logic.boolalg import BooleanTrue
    f = BooleanTrue()
    assert solveset(f, x, domain=Interval(-3, 10)) == Interval(-3, 10)

    # issue 20552
    f = Piecewise((0, Eq(x, 0)), (x**2/Abs(x), True))
    g = Piecewise((0, Eq(x, pi)), ((x - pi)/sin(x), True))
    assert solveset(f, x, domain=S.Reals) == FiniteSet(0)
    assert solveset(g) == FiniteSet(pi)


def test_solveset_complex_polynomial():
    assert solveset_complex(a*x**2 + b*x + c, x) == \
        FiniteSet(-b/(2*a) - sqrt(-4*a*c + b**2)/(2*a),
                  -b/(2*a) + sqrt(-4*a*c + b**2)/(2*a))

    assert solveset_complex(x - y**3, y) == FiniteSet(
        (-x**Rational(1, 3))/2 + I*sqrt(3)*x**Rational(1, 3)/2,
        x**Rational(1, 3),
        (-x**Rational(1, 3))/2 - I*sqrt(3)*x**Rational(1, 3)/2)

    assert solveset_complex(x + 1/x - 1, x) == \
        FiniteSet(S.Half + I*sqrt(3)/2, S.Half - I*sqrt(3)/2)


def test_sol_zero_complex():
    assert solveset_complex(0, x) is S.Complexes


def test_solveset_complex_rational():
    assert solveset_complex((x - 1)*(x - I)/(x - 3), x) == \
        FiniteSet(1, I)

    assert solveset_complex((x - y**3)/((y**2)*sqrt(1 - y**2)), x) == \
        FiniteSet(y**3)
    assert solveset_complex(-x**2 - I, x) == \
        FiniteSet(-sqrt(2)/2 + sqrt(2)*I/2, sqrt(2)/2 - sqrt(2)*I/2)


def test_solve_quintics():
    skip("This test is too slow")
    f = x**5 - 110*x**3 - 55*x**2 + 2310*x + 979
    s = solveset_complex(f, x)
    for root in s:
        res = f.subs(x, root.n()).n()
        assert tn(res, 0)

    f = x**5 + 15*x + 12
    s = solveset_complex(f, x)
    for root in s:
        res = f.subs(x, root.n()).n()
        assert tn(res, 0)


def test_solveset_complex_exp():
    assert dumeq(solveset_complex(exp(x) - 1, x),
        imageset(Lambda(n, I*2*n*pi), S.Integers))
    assert dumeq(solveset_complex(exp(x) - I, x),
        imageset(Lambda(n, I*(2*n*pi + pi/2)), S.Integers))
    assert solveset_complex(1/exp(x), x) == S.EmptySet
    assert dumeq(solveset_complex(sinh(x).rewrite(exp), x),
        imageset(Lambda(n, n*pi*I), S.Integers))


def test_solveset_real_exp():
    assert solveset(Eq((-2)**x, 4), x, S.Reals) == FiniteSet(2)
    assert solveset(Eq(-2**x, 4), x, S.Reals) == S.EmptySet
    assert solveset(Eq((-3)**x, 27), x, S.Reals) == S.EmptySet
    assert solveset(Eq((-5)**(x+1), 625), x, S.Reals) == FiniteSet(3)
    assert solveset(Eq(2**(x-3), -16), x, S.Reals) == S.EmptySet
    assert solveset(Eq((-3)**(x - 3), -3**39), x, S.Reals) == FiniteSet(42)
    assert solveset(Eq(2**x, y), x, S.Reals) == Intersection(S.Reals, FiniteSet(log(y)/log(2)))

    assert invert_real((-2)**(2*x) - 16, 0, x) == (x, FiniteSet(2))


def test_solve_complex_log():
    assert solveset_complex(log(x), x) == FiniteSet(1)
    assert solveset_complex(1 - log(a + 4*x**2), x) == \
        FiniteSet(-sqrt(-a + E)/2, sqrt(-a + E)/2)


def test_solve_complex_sqrt():
    assert solveset_complex(sqrt(5*x + 6) - 2 - x, x) == \
        FiniteSet(-S.One, S(2))
    assert solveset_complex(sqrt(5*x + 6) - (2 + 2*I) - x, x) == \
        FiniteSet(-S(2), 3 - 4*I)
    assert solveset_complex(4*x*(1 - a * sqrt(x)), x) == \
        FiniteSet(S.Zero, 1 / a ** 2)


def test_solveset_complex_tan():
    s = solveset_complex(tan(x).rewrite(exp), x)
    assert dumeq(s, imageset(Lambda(n, pi*n), S.Integers) - \
        imageset(Lambda(n, pi*n + pi/2), S.Integers))


@_both_exp_pow
def test_solve_trig():
    assert dumeq(solveset_real(sin(x), x),
        Union(imageset(Lambda(n, 2*pi*n), S.Integers),
              imageset(Lambda(n, 2*pi*n + pi), S.Integers)))

    assert dumeq(solveset_real(sin(x) - 1, x),
        imageset(Lambda(n, 2*pi*n + pi/2), S.Integers))

    assert dumeq(solveset_real(cos(x), x),
        Union(imageset(Lambda(n, 2*pi*n + pi/2), S.Integers),
              imageset(Lambda(n, 2*pi*n + pi*Rational(3, 2)), S.Integers)))

    assert dumeq(solveset_real(sin(x) + cos(x), x),
        Union(imageset(Lambda(n, 2*n*pi + pi*Rational(3, 4)), S.Integers),
              imageset(Lambda(n, 2*n*pi + pi*Rational(7, 4)), S.Integers)))

    assert solveset_real(sin(x)**2 + cos(x)**2, x) == S.EmptySet

    assert dumeq(solveset_complex(cos(x) - S.Half, x),
        Union(imageset(Lambda(n, 2*n*pi + pi*Rational(5, 3)), S.Integers),
              imageset(Lambda(n, 2*n*pi + pi/3), S.Integers)))

    assert dumeq(solveset(sin(y + a) - sin(y), a, domain=S.Reals),
        ConditionSet(a, (S(-1) <= sin(y)) & (sin(y) <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi - y + asin(sin(y))), S.Integers),
            ImageSet(Lambda(n, 2*n*pi - y - asin(sin(y)) + pi), S.Integers))))

    assert dumeq(solveset_real(sin(2*x)*cos(x) + cos(2*x)*sin(x)-1, x),
        ImageSet(Lambda(n, n*pi*Rational(2, 3) + pi/6), S.Integers))

    assert dumeq(solveset_real(2*tan(x)*sin(x) + 1, x), Union(
        ImageSet(Lambda(n, 2*n*pi + atan(sqrt(2)*sqrt(-1 + sqrt(17))/
            (1 - sqrt(17))) + pi), S.Integers),
        ImageSet(Lambda(n, 2*n*pi - atan(sqrt(2)*sqrt(-1 + sqrt(17))/
            (1 - sqrt(17))) + pi), S.Integers)))

    assert dumeq(solveset_real(cos(2*x)*cos(4*x) - 1, x),
                            ImageSet(Lambda(n, n*pi), S.Integers))

    assert dumeq(solveset(sin(x/10) + Rational(3, 4)), Union(
        ImageSet(Lambda(n, 20*n*pi - 10*asin(S(3)/4) + 20*pi), S.Integers),
        ImageSet(Lambda(n, 20*n*pi + 10*asin(S(3)/4) + 10*pi), S.Integers)))

    assert dumeq(solveset(cos(x/15) + cos(x/5)), Union(
        ImageSet(Lambda(n, 30*n*pi + 15*pi/2), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 45*pi/2), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 75*pi/4), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 45*pi/4), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 105*pi/4), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 15*pi/4), S.Integers)))

    assert dumeq(solveset(sec(sqrt(2)*x/3) + 5), Union(
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*pi - asec(-5))/2), S.Integers),
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*pi + asec(-5))/2), S.Integers)))

    assert dumeq(simplify(solveset(tan(pi*x) - cot(pi/2*x))), Union(
        ImageSet(Lambda(n, 4*n + 1), S.Integers),
        ImageSet(Lambda(n, 4*n + 3), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(7, 3)), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(5, 3)), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(11, 3)), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(1, 3)), S.Integers)))

    assert dumeq(solveset(cos(9*x)), Union(
        ImageSet(Lambda(n, 2*n*pi/9 + pi/18), S.Integers),
        ImageSet(Lambda(n, 2*n*pi/9 + pi/6), S.Integers)))

    assert dumeq(solveset(sin(8*x) + cot(12*x), x, S.Reals), Union(
        ImageSet(Lambda(n, n*pi/2 + pi/8), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 3*pi/8), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 5*pi/16), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 3*pi/16), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 7*pi/16), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + pi/16), S.Integers)))

    # This is the only remaining solveset test that actually ends up being solved
    # by _solve_trig2(). All others are handled by the improved _solve_trig1.
    assert dumeq(solveset_real(2*cos(x)*cos(2*x) - 1, x),
          Union(ImageSet(Lambda(n, 2*n*pi + 2*atan(sqrt(-2*2**Rational(1, 3)*(67 +
                  9*sqrt(57))**Rational(2, 3) + 8*2**Rational(2, 3) + 11*(67 +
                  9*sqrt(57))**Rational(1, 3))/(3*(67 + 9*sqrt(57))**Rational(1, 6)))), S.Integers),
                  ImageSet(Lambda(n, 2*n*pi - 2*atan(sqrt(-2*2**Rational(1, 3)*(67 +
                  9*sqrt(57))**Rational(2, 3) + 8*2**Rational(2, 3) + 11*(67 +
                  9*sqrt(57))**Rational(1, 3))/(3*(67 + 9*sqrt(57))**Rational(1, 6))) +
                  2*pi), S.Integers)))

    # issue #16870
    assert dumeq(simplify(solveset(sin(x/180*pi) - S.Half, x, S.Reals)), Union(
        ImageSet(Lambda(n, 360*n + 150), S.Integers),
        ImageSet(Lambda(n, 360*n + 30), S.Integers)))


def test_solve_trig_hyp_by_inversion():
    n = Dummy('n')
    assert solveset_real(sin(2*x + 3) - S(1)/2, x).dummy_eq(Union(
        ImageSet(Lambda(n, n*pi - S(3)/2 + 13*pi/12), S.Integers),
        ImageSet(Lambda(n, n*pi - S(3)/2 + 17*pi/12), S.Integers)))
    assert solveset_complex(sin(2*x + 3) - S(1)/2, x).dummy_eq(Union(
        ImageSet(Lambda(n, n*pi - S(3)/2 + 13*pi/12), S.Integers),
        ImageSet(Lambda(n, n*pi - S(3)/2 + 17*pi/12), S.Integers)))
    assert solveset_real(tan(x) - tan(pi/10), x).dummy_eq(
        ImageSet(Lambda(n, n*pi + pi/10), S.Integers))
    assert solveset_complex(tan(x) - tan(pi/10), x).dummy_eq(
        ImageSet(Lambda(n, n*pi + pi/10), S.Integers))

    assert solveset_real(3*cosh(2*x) - 5, x) == FiniteSet(
        -acosh(S(5)/3)/2, acosh(S(5)/3)/2)
    assert solveset_complex(3*cosh(2*x) - 5, x).dummy_eq(Union(
        ImageSet(Lambda(n, n*I*pi - acosh(S(5)/3)/2), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acosh(S(5)/3)/2), S.Integers)))
    assert solveset_real(sinh(x - 3) - 2, x) == FiniteSet(
        asinh(2) + 3)
    assert solveset_complex(sinh(x - 3) - 2, x).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*I*pi + asinh(2) + 3), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi - asinh(2) + 3 + I*pi), S.Integers)))

    assert solveset_real(cos(sinh(x))-cos(pi/12), x).dummy_eq(Union(
        ImageSet(Lambda(n, asinh(2*n*pi + pi/12)), S.Integers),
        ImageSet(Lambda(n, asinh(2*n*pi + 23*pi/12)), S.Integers)))
    assert solveset(cos(sinh(x))-cos(pi/12), x, Interval(2,3)) == \
        FiniteSet(asinh(23*pi/12), asinh(25*pi/12))
    assert solveset_real(cosh(x**2-1)-2, x) == FiniteSet(
        -sqrt(1 + acosh(2)), sqrt(1 + acosh(2)))

    assert solveset_real(sin(x) - 2, x) == S.EmptySet   # issue #17334
    assert solveset_real(cos(x) + 2, x) == S.EmptySet
    assert solveset_real(sec(x), x) == S.EmptySet
    assert solveset_real(csc(x), x) == S.EmptySet
    assert solveset_real(cosh(x) + 1, x) == S.EmptySet
    assert solveset_real(coth(x), x) == S.EmptySet
    assert solveset_real(sech(x) - 2, x) == S.EmptySet
    assert solveset_real(sech(x), x) == S.EmptySet
    assert solveset_real(tanh(x) + 1, x) == S.EmptySet
    assert solveset_complex(tanh(x), 1) == S.EmptySet
    assert solveset_complex(coth(x), -1) == S.EmptySet
    assert solveset_complex(sech(x), 0) == S.EmptySet
    assert solveset_complex(csch(x), 0) == S.EmptySet

    assert solveset_real(abs(csch(x)) - 3, x) == FiniteSet(-acsch(3), acsch(3))

    assert solveset_real(tanh(x**2 - 1) - exp(-9), x) == FiniteSet(
        -sqrt(atanh(exp(-9)) + 1), sqrt(atanh(exp(-9)) + 1))

    assert solveset_real(coth(log(x)) + 2, x) == FiniteSet(exp(-acoth(2)))
    assert solveset_real(coth(exp(x)) + 2, x) == S.EmptySet

    assert solveset_complex(sinh(x) - I/2, x).dummy_eq(Union(
        ImageSet(Lambda(n, 2*I*pi*n + 5*I*pi/6), S.Integers),
        ImageSet(Lambda(n, 2*I*pi*n + I*pi/6), S.Integers)))
    assert solveset_complex(sinh(x/10) + Rational(3, 4), x).dummy_eq(Union(
        ImageSet(Lambda(n, 20*n*I*pi - 10*asinh(S(3)/4)), S.Integers),
        ImageSet(Lambda(n, 20*n*I*pi + 10*asinh(S(3)/4) + 10*I*pi), S.Integers)))
    assert solveset_complex(sech(sqrt(2)*x/3) + 5, x).dummy_eq(Union(
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*I*pi - asech(-5))/2), S.Integers),
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*I*pi + asech(-5))/2), S.Integers)))
    assert solveset_complex(cosh(9*x), x).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*I*pi/9 + I*pi/18), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi/9 + I*pi/6), S.Integers)))

    eq = (x**5 -4*x + 1).subs(x, coth(z))
    assert solveset(eq, z, S.Complexes).dummy_eq(Union(
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 0))), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 1))), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 2))), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 3))), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 4))), S.Integers)))
    assert solveset(eq, z, S.Reals) == FiniteSet(
        acoth(CRootOf(x**5 - 4*x + 1, 0)), acoth(CRootOf(x**5 - 4*x + 1, 2)))

    eq = ((x-sqrt(3)/2)*(x+2)).expand().subs(x, cos(x))
    assert solveset(eq, x, S.Complexes).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi - acos(-2)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + acos(-2)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + 11*pi/6), S.Integers)))
    assert solveset(eq, x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + 11*pi/6), S.Integers)))

    assert solveset((1+sec(sqrt(3)*x+4)**2)/(1-sec(sqrt(3)*x+4))).dummy_eq(Union(
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 - asec(I))/3), S.Integers),
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 + asec(I))/3), S.Integers),
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 - asec(-I))/3), S.Integers),
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 + asec(-I))/3), S.Integers)))

    assert all_close(solveset(tan(3.14*x)**(S(3)/2)-5.678, x, Interval(0, 3)),
        FiniteSet(0.403301114561067, 0.403301114561067 + 0.318471337579618*pi,
                0.403301114561067 + 0.636942675159236*pi))


def test_old_trig_issues():
    # issues #9606 / #9531:
    assert solveset(sinh(x), x, S.Reals) == FiniteSet(0)
    assert solveset(sinh(x), x, S.Complexes).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*I*pi), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + I*pi), S.Integers)))

    # issues #11218 / #18427
    assert solveset(sin(pi*x), x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, (2*n*pi + pi)/pi), S.Integers),
        ImageSet(Lambda(n, 2*n), S.Integers)))
    assert solveset(sin(pi*x), x).dummy_eq(Union(
        ImageSet(Lambda(n, (2*n*pi + pi)/pi), S.Integers),
        ImageSet(Lambda(n, 2*n), S.Integers)))

    # issue #17543
    assert solveset(I*cot(8*x - 8*E), x).dummy_eq(
        ImageSet(Lambda(n, pi*n/8 - 13*pi/16 + E), S.Integers))

    # issue #20798
    assert all_close(solveset(cos(2*x) - 0.5, x, Interval(0, 2*pi)), FiniteSet(
        0.523598775598299, -0.523598775598299 + pi,
        -0.523598775598299 + 2*pi, 0.523598775598299 + pi))
    sol = Union(ImageSet(Lambda(n, n*pi - 0.523598775598299), S.Integers),
                ImageSet(Lambda(n, n*pi + 0.523598775598299), S.Integers))
    ret = solveset(cos(2*x) - 0.5, x, S.Reals)
    # replace Dummy n by the regular Symbol n to allow all_close comparison.
    ret = ret.subs(ret.atoms(Dummy).pop(), n)
    assert all_close(ret, sol)
    ret = solveset(cos(2*x) - 0.5, x, S.Complexes)
    ret = ret.subs(ret.atoms(Dummy).pop(), n)
    assert all_close(ret, sol)

    # issue #21296 / #17667
    assert solveset(tan(x)-sqrt(2), x, Interval(0, pi/2)) == FiniteSet(atan(sqrt(2)))
    assert solveset(tan(x)-pi, x, Interval(0, pi/2)) == FiniteSet(atan(pi))

    # issue #17667
    # not yet working properly:
    # solveset(cos(x)-y, x, Interval(0, pi))
    assert solveset(cos(x)-y, x, S.Reals).dummy_eq(
        ConditionSet(x,(S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi - acos(y)), S.Integers),
            ImageSet(Lambda(n, 2*n*pi + acos(y)), S.Integers))))

    # issue #17579
    # Valid result, but the intersection could potentially be simplified.
    assert solveset(sin(log(x)), x, Interval(0,1, True, False)).dummy_eq(
        Union(Intersection(ImageSet(Lambda(n, exp(2*n*pi)), S.Integers), Interval.Lopen(0, 1)),
              Intersection(ImageSet(Lambda(n, exp(2*n*pi + pi)), S.Integers), Interval.Lopen(0, 1))))

    # issue #17334
    assert solveset(sin(x) - sin(1), x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi + 1), S.Integers),
        ImageSet(Lambda(n, 2*n*pi - 1 + pi), S.Integers)))
    assert solveset(sin(x) - sqrt(5)/3, x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi + asin(sqrt(5)/3)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi - asin(sqrt(5)/3) + pi), S.Integers)))
    assert solveset(sinh(x)-cosh(2), x, S.Reals) == FiniteSet(asinh(cosh(2)))

    # issue 9825
    assert solveset(Eq(tan(x), y), x, domain=S.Reals).dummy_eq(
        ConditionSet(x, (-oo < y) & (y < oo),
                     ImageSet(Lambda(n, n*pi + atan(y)), S.Integers)))
    r = Symbol('r', real=True)
    assert solveset(Eq(tan(x), r), x, domain=S.Reals).dummy_eq(
        ImageSet(Lambda(n, n*pi + atan(r)), S.Integers))


def test_solve_hyperbolic():
    # actual solver: _solve_trig1
    n = Dummy('n')
    assert solveset(sinh(x) + cosh(x), x) == S.EmptySet
    assert solveset(sinh(x) + cos(x), x) == ConditionSet(x,
        Eq(cos(x) + sinh(x), 0), S.Complexes)
    assert solveset_real(sinh(x) + sech(x), x) == FiniteSet(
        log(sqrt(sqrt(5) - 2)))
    assert solveset_real(cosh(2*x) + 2*sinh(x) - 5, x) == FiniteSet(
        log(-2 + sqrt(5)), log(1 + sqrt(2)))
    assert solveset_real((coth(x) + sinh(2*x))/cosh(x) - 3, x) == FiniteSet(
        log(S.Half + sqrt(5)/2), log(1 + sqrt(2)))
    assert solveset_real(cosh(x)*sinh(x) - 2, x) == FiniteSet(
        log(4 + sqrt(17))/2)
    assert solveset_real(sinh(x) + tanh(x) - 1, x) == FiniteSet(
        log(sqrt(2)/2 + sqrt(-S(1)/2 + sqrt(2))))

    assert dumeq(solveset_complex(sinh(x) + sech(x), x), Union(
        ImageSet(Lambda(n, 2*n*I*pi + log(sqrt(-2 + sqrt(5)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi + pi/2) + log(sqrt(2 + sqrt(5)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi + pi) + log(sqrt(-2 + sqrt(5)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi - pi/2) + log(sqrt(2 + sqrt(5)))), S.Integers)))

    assert dumeq(solveset(cosh(x/15) + cosh(x/5)), Union(
        ImageSet(Lambda(n, 15*I*(2*n*pi + pi/2)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi - pi/2)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi - 3*pi/4)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi + 3*pi/4)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi - pi/4)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi + pi/4)), S.Integers)))

    assert dumeq(solveset(tanh(pi*x) - coth(pi/2*x)), Union(
        ImageSet(Lambda(n, 2*I*(2*n*pi + pi/2)/pi), S.Integers),
        ImageSet(Lambda(n, 2*I*(2*n*pi - pi/2)/pi), S.Integers)))

    # issues #18490 / #19489
    assert solveset(cosh(x) + cosh(3*x) - cosh(5*x), x, S.Reals
        ).dummy_eq(ConditionSet(x,
        Eq(cosh(x) + cosh(3*x) - cosh(5*x), 0), S.Reals))
    assert solveset(sinh(8*x) + coth(12*x)).dummy_eq(
        ConditionSet(x, Eq(sinh(8*x) + coth(12*x), 0), S.Complexes))


def test_solve_trig_hyp_symbolic():
    # actual solver: invert_trig_hyp
    assert dumeq(solveset(sin(a*x), x), ConditionSet(x, Ne(a, 0), Union(
        ImageSet(Lambda(n, (2*n*pi + pi)/a), S.Integers),
        ImageSet(Lambda(n, 2*n*pi/a), S.Integers))))

    assert dumeq(solveset(cosh(x/a), x), ConditionSet(x, Ne(a, 0), Union(
        ImageSet(Lambda(n, a*(2*n*I*pi + I*pi/2)), S.Integers),
        ImageSet(Lambda(n, a*(2*n*I*pi + 3*I*pi/2)), S.Integers))))

    assert dumeq(solveset(sin(2*sqrt(3)/3*a**2/(b*pi)*x)
        + cos(4*sqrt(3)/3*a**2/(b*pi)*x), x),
       ConditionSet(x, Ne(b, 0) & Ne(a**2, 0), Union(
           ImageSet(Lambda(n, sqrt(3)*pi*b*(2*n*pi + pi/2)/(2*a**2)), S.Integers),
           ImageSet(Lambda(n, sqrt(3)*pi*b*(2*n*pi - 5*pi/6)/(2*a**2)), S.Integers),
           ImageSet(Lambda(n, sqrt(3)*pi*b*(2*n*pi - pi/6)/(2*a**2)), S.Integers))))

    assert dumeq(solveset(cosh((a**2 + 1)*x) - 3, x), ConditionSet(
        x, Ne(a**2 + 1, 0), Union(
            ImageSet(Lambda(n, (2*n*I*pi - acosh(3))/(a**2 + 1)), S.Integers),
            ImageSet(Lambda(n, (2*n*I*pi + acosh(3))/(a**2 + 1)), S.Integers))))

    ar = Symbol('ar', real=True)
    assert solveset(cosh((ar**2 + 1)*x) - 2, x, S.Reals) == FiniteSet(
        -acosh(2)/(ar**2 + 1), acosh(2)/(ar**2 + 1))

    # actual solver: _solve_trig1
    assert dumeq(simplify(solveset(cot((1 + I)*x) - cot((3 + 3*I)*x), x)), Union(
        ImageSet(Lambda(n, pi*(1 - I)*(4*n + 1)/4), S.Integers),
        ImageSet(Lambda(n, pi*(1 - I)*(4*n - 1)/4), S.Integers)))


def test_issue_9616():
    assert dumeq(solveset(sinh(x) + tanh(x) - 1, x), Union(
        ImageSet(Lambda(n, 2*n*I*pi + log(sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi - atan(sqrt(2)*sqrt(S.Half + sqrt(2))) + pi)
            + log(sqrt(1 + sqrt(2)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi + pi) + log(-sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi - pi + atan(sqrt(2)*sqrt(S.Half + sqrt(2))))
            + log(sqrt(1 + sqrt(2)))), S.Integers)))
    f1 = (sinh(x)).rewrite(exp)
    f2 = (tanh(x)).rewrite(exp)
    assert dumeq(solveset(f1 + f2 - 1, x), Union(
        Complement(ImageSet(
            Lambda(n, I*(2*n*pi + pi) + log(-sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers)),
        Complement(ImageSet(Lambda(n, I*(2*n*pi - pi + atan(sqrt(2)*sqrt(S.Half + sqrt(2))))
                + log(sqrt(1 + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers)),
        Complement(ImageSet(Lambda(n, I*(2*n*pi - atan(sqrt(2)*sqrt(S.Half + sqrt(2))) + pi)
                + log(sqrt(1 + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers)),
        Complement(
            ImageSet(Lambda(n, 2*n*I*pi + log(sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers))))


def test_solve_invalid_sol():
    assert 0 not in solveset_real(sin(x)/x, x)
    assert 0 not in solveset_complex((exp(x) - 1)/x, x)


@XFAIL
def test_solve_trig_simplified():
    n = Dummy('n')
    assert dumeq(solveset_real(sin(x), x),
        imageset(Lambda(n, n*pi), S.Integers))

    assert dumeq(solveset_real(cos(x), x),
        imageset(Lambda(n, n*pi + pi/2), S.Integers))

    assert dumeq(solveset_real(cos(x) + sin(x), x),
        imageset(Lambda(n, n*pi - pi/4), S.Integers))


@XFAIL
def test_solve_lambert():
    assert solveset_real(x*exp(x) - 1, x) == FiniteSet(LambertW(1))
    assert solveset_real(exp(x) + x, x) == FiniteSet(-LambertW(1))
    assert solveset_real(x + 2**x, x) == \
        FiniteSet(-LambertW(log(2))/log(2))

    # issue 4739
    ans = solveset_real(3*x + 5 + 2**(-5*x + 3), x)
    assert ans == FiniteSet(Rational(-5, 3) +
                            LambertW(-10240*2**Rational(1, 3)*log(2)/3)/(5*log(2)))

    eq = 2*(3*x + 4)**5 - 6*7**(3*x + 9)
    result = solveset_real(eq, x)
    ans = FiniteSet((log(2401) +
                     5*LambertW(-log(7**(7*3**Rational(1, 5)/5))))/(3*log(7))/-1)
    assert result == ans
    assert solveset_real(eq.expand(), x) == result

    assert solveset_real(5*x - 1 + 3*exp(2 - 7*x), x) == \
        FiniteSet(Rational(1, 5) + LambertW(-21*exp(Rational(3, 5))/5)/7)

    assert solveset_real(2*x + 5 + log(3*x - 2), x) == \
        FiniteSet(Rational(2, 3) + LambertW(2*exp(Rational(-19, 3))/3)/2)

    assert solveset_real(3*x + log(4*x), x) == \
        FiniteSet(LambertW(Rational(3, 4))/3)

    assert solveset_real(x**x - 2) == FiniteSet(exp(LambertW(log(2))))

    a = Symbol('a')
    assert solveset_real(-a*x + 2*x*log(x), x) == FiniteSet(exp(a/2))
    a = Symbol('a', real=True)
    assert solveset_real(a/x + exp(x/2), x) == \
        FiniteSet(2*LambertW(-a/2))
    assert solveset_real((a/x + exp(x/2)).diff(x), x) == \
        FiniteSet(4*LambertW(sqrt(2)*sqrt(a)/4))

    # coverage test
    assert solveset_real(tanh(x + 3)*tanh(x - 3) - 1, x) is S.EmptySet

    assert solveset_real((x**2 - 2*x + 1).subs(x, log(x) + 3*x), x) == \
        FiniteSet(LambertW(3*S.Exp1)/3)
    assert solveset_real((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x) == \
        FiniteSet(LambertW(3*exp(-sqrt(2)))/3, LambertW(3*exp(sqrt(2)))/3)
    assert solveset_real((x**2 - 2*x - 2).subs(x, log(x) + 3*x), x) == \
        FiniteSet(LambertW(3*exp(1 + sqrt(3)))/3, LambertW(3*exp(-sqrt(3) + 1))/3)
    assert solveset_real(x*log(x) + 3*x + 1, x) == \
        FiniteSet(exp(-3 + LambertW(-exp(3))))
    eq = (x*exp(x) - 3).subs(x, x*exp(x))
    assert solveset_real(eq, x) == \
        FiniteSet(LambertW(3*exp(-LambertW(3))))

    assert solveset_real(3*log(a**(3*x + 5)) + a**(3*x + 5), x) == \
        FiniteSet(-((log(a**5) + LambertW(Rational(1, 3)))/(3*log(a))))
    p = symbols('p', positive=True)
    assert solveset_real(3*log(p**(3*x + 5)) + p**(3*x + 5), x) == \
        FiniteSet(
        log((-3**Rational(1, 3) - 3**Rational(5, 6)*I)*LambertW(Rational(1, 3))**Rational(1, 3)/(2*p**Rational(5, 3)))/log(p),
        log((-3**Rational(1, 3) + 3**Rational(5, 6)*I)*LambertW(Rational(1, 3))**Rational(1, 3)/(2*p**Rational(5, 3)))/log(p),
        log((3*LambertW(Rational(1, 3))/p**5)**(1/(3*log(p)))),)  # checked numerically
    # check collection
    b = Symbol('b')
    eq = 3*log(a**(3*x + 5)) + b*log(a**(3*x + 5)) + a**(3*x + 5)
    assert solveset_real(eq, x) == FiniteSet(
        -((log(a**5) + LambertW(1/(b + 3)))/(3*log(a))))

    # issue 4271
    assert solveset_real((a/x + exp(x/2)).diff(x, 2), x) == FiniteSet(
        6*LambertW((-1)**Rational(1, 3)*a**Rational(1, 3)/3))

    assert solveset_real(x**3 - 3**x, x) == \
        FiniteSet(-3/log(3)*LambertW(-log(3)/3))
    assert solveset_real(3**cos(x) - cos(x)**3) == FiniteSet(
        acos(-3*LambertW(-log(3)/3)/log(3)))

    assert solveset_real(x**2 - 2**x, x) == \
        solveset_real(-x**2 + 2**x, x)

    assert solveset_real(3*log(x) - x*log(3)) == FiniteSet(
        -3*LambertW(-log(3)/3)/log(3),
        -3*LambertW(-log(3)/3, -1)/log(3))

    assert solveset_real(LambertW(2*x) - y) == FiniteSet(
        y*exp(y)/2)


@XFAIL
def test_other_lambert():
    a = Rational(6, 5)
    assert solveset_real(x**a - a**x, x) == FiniteSet(
        a, -a*LambertW(-log(a)/a)/log(a))


@_both_exp_pow
def test_solveset():
    f = Function('f')
    raises(ValueError, lambda: solveset(x + y))
    assert solveset(x, 1) == S.EmptySet
    assert solveset(f(1)**2 + y + 1, f(1)
        ) == FiniteSet(-sqrt(-y - 1), sqrt(-y - 1))
    assert solveset(f(1)**2 - 1, f(1), S.Reals) == FiniteSet(-1, 1)
    assert solveset(f(1)**2 + 1, f(1)) == FiniteSet(-I, I)
    assert solveset(x - 1, 1) == FiniteSet(x)
    assert solveset(sin(x) - cos(x), sin(x)) == FiniteSet(cos(x))

    assert solveset(0, domain=S.Reals) == S.Reals
    assert solveset(1) == S.EmptySet
    assert solveset(True, domain=S.Reals) == S.Reals  # issue 10197
    assert solveset(False, domain=S.Reals) == S.EmptySet

    assert solveset(exp(x) - 1, domain=S.Reals) == FiniteSet(0)
    assert solveset(exp(x) - 1, x, S.Reals) == FiniteSet(0)
    assert solveset(Eq(exp(x), 1), x, S.Reals) == FiniteSet(0)
    assert solveset(exp(x) - 1, exp(x), S.Reals) == FiniteSet(1)
    A = Indexed('A', x)
    assert solveset(A - 1, A, S.Reals) == FiniteSet(1)

    assert solveset(x - 1 >= 0, x, S.Reals) == Interval(1, oo)
    assert solveset(exp(x) - 1 >= 0, x, S.Reals) == Interval(0, oo)

    assert dumeq(solveset(exp(x) - 1, x), imageset(Lambda(n, 2*I*pi*n), S.Integers))
    assert dumeq(solveset(Eq(exp(x), 1), x), imageset(Lambda(n, 2*I*pi*n),
                                                  S.Integers))
    # issue 13825
    assert solveset(x**2 + f(0) + 1, x) == {-sqrt(-f(0) - 1), sqrt(-f(0) - 1)}

    # issue 19977
    assert solveset(atan(log(x)) > 0, x, domain=Interval.open(0, oo)) == Interval.open(1, oo)


@_both_exp_pow
def test_multi_exp():
    k1, k2, k3 = symbols('k1, k2, k3')
    assert dumeq(solveset(exp(exp(x)) - 5, x),\
         imageset(Lambda(((k1, n),), I*(2*k1*pi + arg(2*n*I*pi + log(5))) + log(Abs(2*n*I*pi + log(5)))),\
             ProductSet(S.Integers, S.Integers)))
    assert dumeq(solveset((d*exp(exp(a*x + b)) + c), x),\
        imageset(Lambda(x, (-b + x)/a), ImageSet(Lambda(((k1, n),), \
            I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d))))), \
                ProductSet(S.Integers, S.Integers))))

    assert dumeq(solveset((d*exp(exp(exp(a*x + b))) + c), x),\
        imageset(Lambda(x, (-b + x)/a), ImageSet(Lambda(((k2, k1, n),), \
            I*(2*k2*pi + arg(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + \
                log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))) + log(Abs(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + \
                    log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d))))))), \
                        ProductSet(S.Integers, S.Integers, S.Integers))))

    assert dumeq(solveset((d*exp(exp(exp(exp(a*x + b)))) + c), x),\
        ImageSet(Lambda(x, (-b + x)/a), ImageSet(Lambda(((k3, k2, k1, n),), \
            I*(2*k3*pi + arg(I*(2*k2*pi + arg(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + \
                log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))) + log(Abs(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + \
                    log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))))) + log(Abs(I*(2*k2*pi + \
                        arg(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))) + \
                            log(Abs(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d))))))))), \
             ProductSet(S.Integers, S.Integers, S.Integers, S.Integers))))


def test__solveset_multi():
    from sympy.solvers.solveset import _solveset_multi
    from sympy.sets import Reals

    # Basic univariate case:
    assert _solveset_multi([x**2-1], [x], [S.Reals]) == FiniteSet((1,), (-1,))

    # Linear systems of two equations
    assert _solveset_multi([x+y, x+1], [x, y], [Reals, Reals]) == FiniteSet((-1, 1))
    assert _solveset_multi([x+y, x+1], [y, x], [Reals, Reals]) == FiniteSet((1, -1))
    assert _solveset_multi([x+y, x-y-1], [x, y], [Reals, Reals]) == FiniteSet((S(1)/2, -S(1)/2))
    assert _solveset_multi([x-1, y-2], [x, y], [Reals, Reals]) == FiniteSet((1, 2))
    # assert dumeq(_solveset_multi([x+y], [x, y], [Reals, Reals]), ImageSet(Lambda(x, (x, -x)), Reals))
    assert dumeq(_solveset_multi([x+y], [x, y], [Reals, Reals]), Union(
            ImageSet(Lambda(((x,),), (x, -x)), ProductSet(Reals)),
            ImageSet(Lambda(((y,),), (-y, y)), ProductSet(Reals))))
    assert _solveset_multi([x+y, x+y+1], [x, y], [Reals, Reals]) == S.EmptySet
    assert _solveset_multi([x+y, x-y, x-1], [x, y], [Reals, Reals]) == S.EmptySet
    assert _solveset_multi([x+y, x-y, x-1], [y, x], [Reals, Reals]) == S.EmptySet

    # Systems of three equations:
    assert _solveset_multi([x+y+z-1, x+y-z-2, x-y-z-3], [x, y, z], [Reals,
        Reals, Reals]) == FiniteSet((2, -S.Half, -S.Half))

    # Nonlinear systems:
    from sympy.abc import theta
    assert _solveset_multi([x**2+y**2-2, x+y], [x, y], [Reals, Reals]) == FiniteSet((-1, 1), (1, -1))
    assert _solveset_multi([x**2-1, y], [x, y], [Reals, Reals]) == FiniteSet((1, 0), (-1, 0))
    #assert _solveset_multi([x**2-y**2], [x, y], [Reals, Reals]) == Union(
    #        ImageSet(Lambda(x, (x, -x)), Reals), ImageSet(Lambda(x, (x, x)), Reals))
    assert dumeq(_solveset_multi([x**2-y**2], [x, y], [Reals, Reals]), Union(
            ImageSet(Lambda(((x,),), (x, -Abs(x))), ProductSet(Reals)),
            ImageSet(Lambda(((x,),), (x, Abs(x))), ProductSet(Reals)),
            ImageSet(Lambda(((y,),), (-Abs(y), y)), ProductSet(Reals)),
            ImageSet(Lambda(((y,),), (Abs(y), y)), ProductSet(Reals))))
    assert _solveset_multi([r*cos(theta)-1, r*sin(theta)], [theta, r],
            [Interval(0, pi), Interval(-1, 1)]) == FiniteSet((0, 1), (pi, -1))
    assert _solveset_multi([r*cos(theta)-1, r*sin(theta)], [r, theta],
            [Interval(0, 1), Interval(0, pi)]) == FiniteSet((1, 0))
    assert _solveset_multi([r*cos(theta)-r, r*sin(theta)], [r, theta],
           [Interval(0, 1), Interval(0, pi)]) == Union(
           ImageSet(Lambda(((r,),), (r, 0)),
           ImageSet(Lambda(r, (r,)), Interval(0, 1))),
           ImageSet(Lambda(((theta,),), (0, theta)),
           ImageSet(Lambda(theta, (theta,)), Interval(0, pi))))


def test_conditionset():
    assert solveset(Eq(sin(x)**2 + cos(x)**2, 1), x, domain=S.Reals
        ) is S.Reals

    assert solveset(Eq(x**2 + x*sin(x), 1), x, domain=S.Reals
        ).dummy_eq(ConditionSet(x, Eq(x**2 + x*sin(x) - 1, 0), S.Reals))

    assert dumeq(solveset(Eq(-I*(exp(I*x) - exp(-I*x))/2, 1), x
        ), imageset(Lambda(n, 2*n*pi + pi/2), S.Integers))

    assert solveset(x + sin(x) > 1, x, domain=S.Reals
        ).dummy_eq(ConditionSet(x, x + sin(x) > 1, S.Reals))

    assert solveset(Eq(sin(Abs(x)), x), x, domain=S.Reals
        ).dummy_eq(ConditionSet(x, Eq(-x + sin(Abs(x)), 0), S.Reals))

    assert solveset(y**x-z, x, S.Reals
        ).dummy_eq(ConditionSet(x, Eq(y**x - z, 0), S.Reals))


@XFAIL
def test_conditionset_equality():
    ''' Checking equality of different representations of ConditionSet'''
    assert solveset(Eq(tan(x), y), x) == ConditionSet(x, Eq(tan(x), y), S.Complexes)


def test_solveset_domain():
    assert solveset(x**2 - x - 6, x, Interval(0, oo)) == FiniteSet(3)
    assert solveset(x**2 - 1, x, Interval(0, oo)) == FiniteSet(1)
    assert solveset(x**4 - 16, x, Interval(0, 10)) == FiniteSet(2)


def test_improve_coverage():
    solution = solveset(exp(x) + sin(x), x, S.Reals)
    unsolved_object = ConditionSet(x, Eq(exp(x) + sin(x), 0), S.Reals)
    assert solution.dummy_eq(unsolved_object)


def test_issue_9522():
    expr1 = Eq(1/(x**2 - 4) + x, 1/(x**2 - 4) + 2)
    expr2 = Eq(1/x + x, 1/x)

    assert solveset(expr1, x, S.Reals) is S.EmptySet
    assert solveset(expr2, x, S.Reals) is S.EmptySet


def test_solvify():
    assert solvify(x**2 + 10, x, S.Reals) == []
    assert solvify(x**3 + 1, x, S.Complexes) == [-1, S.Half - sqrt(3)*I/2,
                                                 S.Half + sqrt(3)*I/2]
    assert solvify(log(x), x, S.Reals) == [1]
    assert solvify(cos(x), x, S.Reals) == [pi/2, pi*Rational(3, 2)]
    assert solvify(sin(x) + 1, x, S.Reals) == [pi*Rational(3, 2)]
    raises(NotImplementedError, lambda: solvify(sin(exp(x)), x, S.Complexes))


def test_solvify_piecewise():
    p1 = Piecewise((0, x < -1), (x**2, x <= 1), (log(x), True))
    p2 = Piecewise((0, x < -10), (x**2 + 5*x - 6, x >= -9))
    p3 = Piecewise((0, Eq(x, 0)), (x**2/Abs(x), True))
    p4 = Piecewise((0, Eq(x, pi)), ((x - pi)/sin(x), True))

    # issue 21079
    assert solvify(p1, x, S.Reals) == [0]
    assert solvify(p2, x, S.Reals) == [-6, 1]
    assert solvify(p3, x, S.Reals) == [0]
    assert solvify(p4, x, S.Reals) == [pi]


def test_abs_invert_solvify():

    x = Symbol('x',positive=True)
    assert solvify(sin(Abs(x)), x, S.Reals) == [0, pi]
    x = Symbol('x')
    assert solvify(sin(Abs(x)), x, S.Reals) is None


def test_linear_eq_to_matrix():
    assert linear_eq_to_matrix(0, x) == (Matrix([[0]]), Matrix([[0]]))
    assert linear_eq_to_matrix(1, x) == (Matrix([[0]]), Matrix([[-1]]))

    # integer coefficients
    eqns1 = [2*x + y - 2*z - 3, x - y - z, x + y + 3*z - 12]
    eqns2 = [Eq(3*x + 2*y - z, 1), Eq(2*x - 2*y + 4*z, -2), -2*x + y - 2*z]

    A, B = linear_eq_to_matrix(eqns1, x, y, z)
    assert A == Matrix([[2, 1, -2], [1, -1, -1], [1, 1, 3]])
    assert B == Matrix([[3], [0], [12]])

    A, B = linear_eq_to_matrix(eqns2, x, y, z)
    assert A == Matrix([[3, 2, -1], [2, -2, 4], [-2, 1, -2]])
    assert B == Matrix([[1], [-2], [0]])

    # Pure symbolic coefficients
    eqns3 = [a*b*x + b*y + c*z - d, e*x + d*x + f*y + g*z - h, i*x + j*y + k*z - l]
    A, B = linear_eq_to_matrix(eqns3, x, y, z)
    assert A == Matrix([[a*b, b, c], [d + e, f, g], [i, j, k]])
    assert B == Matrix([[d], [h], [l]])

    # raise Errors if
    # 1) no symbols are given
    raises(ValueError, lambda: linear_eq_to_matrix(eqns3))
    # 2) there are duplicates
    raises(ValueError, lambda: linear_eq_to_matrix(eqns3, [x, x, y]))
    # 3) a nonlinear term is detected in the original expression
    raises(NonlinearError, lambda: linear_eq_to_matrix(Eq(1/x + x, 1/x), [x]))
    raises(NonlinearError, lambda: linear_eq_to_matrix([x**2], [x]))
    raises(NonlinearError, lambda: linear_eq_to_matrix([x*y], [x, y]))
    # 4) Eq being used to represent equations autoevaluates
    # (use unevaluated Eq instead)
    raises(ValueError, lambda: linear_eq_to_matrix(Eq(x, x), x))
    raises(ValueError, lambda: linear_eq_to_matrix(Eq(x, x + 1), x))


    # if non-symbols are passed, the user is responsible for interpreting
    assert linear_eq_to_matrix([x], [1/x]) == (Matrix([[0]]), Matrix([[-x]]))

    # issue 15195
    assert linear_eq_to_matrix(x + y*(z*(3*x + 2) + 3), x) == (
        Matrix([[3*y*z + 1]]), Matrix([[-y*(2*z + 3)]]))
    assert linear_eq_to_matrix(Matrix(
        [[a*x + b*y - 7], [5*x + 6*y - c]]), x, y) == (
        Matrix([[a, b], [5, 6]]), Matrix([[7], [c]]))

    # issue 15312
    assert linear_eq_to_matrix(Eq(x + 2, 1), x) == (
        Matrix([[1]]), Matrix([[-1]]))

    # issue 25423
    raises(TypeError, lambda: linear_eq_to_matrix([], {x, y}))
    raises(TypeError, lambda: linear_eq_to_matrix([x + y], {x, y}))
    raises(ValueError, lambda: linear_eq_to_matrix({x + y}, (x, y)))


def test_issue_16577():
    assert linear_eq_to_matrix(Eq(a*(2*x + 3*y) + 4*y, 5), x, y) == (
        Matrix([[2*a, 3*a + 4]]), Matrix([[5]]))


def test_issue_10085():
    assert invert_real(exp(x),0,x) == (x, S.EmptySet)


def test_linsolve():
    x1, x2, x3, x4 = symbols('x1, x2, x3, x4')

    # Test for different input forms

    M = Matrix([[1, 2, 1, 1, 7], [1, 2, 2, -1, 12], [2, 4, 0, 6, 4]])
    system1 = A, B = M[:, :-1], M[:, -1]
    Eqns = [x1 + 2*x2 + x3 + x4 - 7, x1 + 2*x2 + 2*x3 - x4 - 12,
            2*x1 + 4*x2 + 6*x4 - 4]

    sol = FiniteSet((-2*x2 - 3*x4 + 2, x2, 2*x4 + 5, x4))
    assert linsolve(Eqns, (x1, x2, x3, x4)) == sol
    assert linsolve(Eqns, *(x1, x2, x3, x4)) == sol
    assert linsolve(system1, (x1, x2, x3, x4)) == sol
    assert linsolve(system1, *(x1, x2, x3, x4)) == sol
    # issue 9667 - symbols can be Dummy symbols
    x1, x2, x3, x4 = symbols('x:4', cls=Dummy)
    assert linsolve(system1, x1, x2, x3, x4) == FiniteSet(
        (-2*x2 - 3*x4 + 2, x2, 2*x4 + 5, x4))

    # raise ValueError for garbage value
    raises(ValueError, lambda: linsolve(Eqns))
    raises(ValueError, lambda: linsolve(x1))
    raises(ValueError, lambda: linsolve(x1, x2))
    raises(ValueError, lambda: linsolve((A,), x1, x2))
    raises(ValueError, lambda: linsolve(A, B, x1, x2))
    raises(ValueError, lambda: linsolve([x1], x1, x1))
    raises(ValueError, lambda: linsolve([x1], (i for i in (x1, x1))))

    #raise ValueError if equations are non-linear in given variables
    raises(NonlinearError, lambda: linsolve([x + y - 1, x ** 2 + y - 3], [x, y]))
    raises(NonlinearError, lambda: linsolve([cos(x) + y, x + y], [x, y]))
    assert linsolve([x + z - 1, x ** 2 + y - 3], [z, y]) == {(-x + 1, -x**2 + 3)}

    # Fully symbolic test
    A = Matrix([[a, b], [c, d]])
    B = Matrix([[e], [g]])
    system2 = (A, B)
    sol = FiniteSet(((-b*g + d*e)/(a*d - b*c), (a*g - c*e)/(a*d - b*c)))
    assert linsolve(system2, [x, y]) == sol

    # No solution
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    B = Matrix([0, 0, 1])
    assert linsolve((A, B), (x, y, z)) is S.EmptySet

    # Issue #10056
    A, B, J1, J2 = symbols('A B J1 J2')
    Augmatrix = Matrix([
        [2*I*J1, 2*I*J2, -2/J1],
        [-2*I*J2, -2*I*J1, 2/J2],
        [0, 2, 2*I/(J1*J2)],
        [2, 0,  0],
        ])

    assert linsolve(Augmatrix, A, B) == FiniteSet((0, I/(J1*J2)))

    # Issue #10121 - Assignment of free variables
    Augmatrix = Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    assert linsolve(Augmatrix, a, b, c, d, e) == FiniteSet((a, 0, c, 0, e))
    #raises(IndexError, lambda: linsolve(Augmatrix, a, b, c))

    x0, x1, x2, _x0 = symbols('tau0 tau1 tau2 _tau0')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
        ) == FiniteSet((x0, 0, x1, _x0, x2))
    x0, x1, x2, _x0 = symbols('tau00 tau01 tau02 tau0')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
        ) == FiniteSet((x0, 0, x1, _x0, x2))
    x0, x1, x2, _x0 = symbols('tau00 tau01 tau02 tau1')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
        ) == FiniteSet((x0, 0, x1, _x0, x2))
    # symbols can be given as generators
    x0, x2, x4 = symbols('x0, x2, x4')
    assert linsolve(Augmatrix, numbered_symbols('x')
        ) == FiniteSet((x0, 0, x2, 0, x4))
    Augmatrix[-1, -1] = x0
    # use Dummy to avoid clash; the names may clash but the symbols
    # will not
    Augmatrix[-1, -1] = symbols('_x0')
    assert len(linsolve(
        Augmatrix, numbered_symbols('x', cls=Dummy)).free_symbols) == 4

    # Issue #12604
    f = Function('f')
    assert linsolve([f(x) - 5], f(x)) == FiniteSet((5,))

    # Issue #14860
    from sympy.physics.units import meter, newton, kilo
    kN = kilo*newton
    Eqns = [8*kN + x + y, 28*kN*meter + 3*x*meter]
    assert linsolve(Eqns, x, y) == {
            (kilo*newton*Rational(-28, 3), kN*Rational(4, 3))}

    # linsolve does not allow expansion (real or implemented)
    # to remove singularities, but it will cancel linear terms
    assert linsolve([Eq(x, x + y)], [x, y]) == {(x, 0)}
    assert linsolve([Eq(x + x*y, 1 + y)], [x]) == {(1,)}
    assert linsolve([Eq(1 + y, x + x*y)], [x]) == {(1,)}
    raises(NonlinearError, lambda:
        linsolve([Eq(x**2, x**2 + y)], [x, y]))

    # corner cases
    #
    # XXX: The case below should give the same as for [0]
    # assert linsolve([], [x]) == {(x,)}
    assert linsolve([], [x]) is S.EmptySet
    assert linsolve([0], [x]) == {(x,)}
    assert linsolve([x], [x, y]) == {(0, y)}
    assert linsolve([x, 0], [x, y]) == {(0, y)}


def test_linsolve_large_sparse():
    #
    # This is mainly a performance test
    #

    def _mk_eqs_sol(n):
        xs = symbols('x:{}'.format(n))
        ys = symbols('y:{}'.format(n))
        syms = xs + ys
        eqs = []
        sol = (-S.Half,) * n + (S.Half,) * n
        for xi, yi in zip(xs, ys):
            eqs.extend([xi + yi, xi - yi + 1])
        return eqs, syms, FiniteSet(sol)

    n = 500
    eqs, syms, sol = _mk_eqs_sol(n)
    assert linsolve(eqs, syms) == sol


def test_linsolve_immutable():
    A = ImmutableDenseMatrix([[1, 1, 2], [0, 1, 2], [0, 0, 1]])
    B = ImmutableDenseMatrix([2, 1, -1])
    assert linsolve([A, B], (x, y, z)) == FiniteSet((1, 3, -1))

    A = ImmutableDenseMatrix([[1, 1, 7], [1, -1, 3]])
    assert linsolve(A) == FiniteSet((5, 2))


def test_solve_decomposition():
    n = Dummy('n')

    f1 = exp(3*x) - 6*exp(2*x) + 11*exp(x) - 6
    f2 = sin(x)**2 - 2*sin(x) + 1
    f3 = sin(x)**2 - sin(x)
    f4 = sin(x + 1)
    f5 = exp(x + 2) - 1
    f6 = 1/log(x)
    f7 = 1/x

    s1 = ImageSet(Lambda(n, 2*n*pi), S.Integers)
    s2 = ImageSet(Lambda(n, 2*n*pi + pi), S.Integers)
    s3 = ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers)
    s4 = ImageSet(Lambda(n, 2*n*pi - 1), S.Integers)
    s5 = ImageSet(Lambda(n, 2*n*pi - 1 + pi), S.Integers)

    assert solve_decomposition(f1, x, S.Reals) == FiniteSet(0, log(2), log(3))
    assert dumeq(solve_decomposition(f2, x, S.Reals), s3)
    assert dumeq(solve_decomposition(f3, x, S.Reals), Union(s1, s2, s3))
    assert dumeq(solve_decomposition(f4, x, S.Reals), Union(s4, s5))
    assert solve_decomposition(f5, x, S.Reals) == FiniteSet(-2)
    assert solve_decomposition(f6, x, S.Reals) == S.EmptySet
    assert solve_decomposition(f7, x, S.Reals) == S.EmptySet
    assert solve_decomposition(x, x, Interval(1, 2)) == S.EmptySet


# nonlinsolve testcases
def test_nonlinsolve_basic():
    assert nonlinsolve([],[]) == S.EmptySet
    assert nonlinsolve([],[x, y]) == S.EmptySet

    system = [x, y - x - 5]
    assert nonlinsolve([x],[x, y]) == FiniteSet((0, y))
    assert nonlinsolve(system, [y]) == S.EmptySet
    soln = (ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers),)
    assert dumeq(nonlinsolve([sin(x) - 1], [x]), FiniteSet(tuple(soln)))
    soln = ((ImageSet(Lambda(n, 2*n*pi + pi), S.Integers), 1),
            (ImageSet(Lambda(n, 2*n*pi), S.Integers), 1))
    assert dumeq(nonlinsolve([sin(x), y - 1], [x, y]), FiniteSet(*soln))
    assert nonlinsolve([x**2 - 1], [x]) == FiniteSet((-1,), (1,))

    soln = FiniteSet((y, y))
    assert nonlinsolve([x - y, 0], x, y) == soln
    assert nonlinsolve([0, x - y], x, y) == soln
    assert nonlinsolve([x - y, x - y], x, y) == soln
    assert nonlinsolve([x, 0], x, y) == FiniteSet((0, y))
    f = Function('f')
    assert nonlinsolve([f(x), 0], f(x), y) == FiniteSet((0, y))
    assert nonlinsolve([f(x), 0], f(x), f(y)) == FiniteSet((0, f(y)))
    A = Indexed('A', x)
    assert nonlinsolve([A, 0], A, y) == FiniteSet((0, y))
    assert nonlinsolve([x**2 -1], [sin(x)]) == FiniteSet((S.EmptySet,))
    assert nonlinsolve([x**2 -1], sin(x)) == FiniteSet((S.EmptySet,))
    assert nonlinsolve([x**2 -1], 1) == FiniteSet((x**2,))
    assert nonlinsolve([x**2 -1], x + y) == FiniteSet((S.EmptySet,))
    assert nonlinsolve([Eq(1, x + y), Eq(1, -x + y - 1), Eq(1, -x + y - 1)], x, y) == FiniteSet(
        (-S.Half, 3*S.Half))


def test_nonlinsolve_abs():
    soln = FiniteSet((y, y), (-y, y))
    assert nonlinsolve([Abs(x) - y], x, y) == soln


def test_raise_exception_nonlinsolve():
    raises(IndexError, lambda: nonlinsolve([x**2 -1], []))
    raises(ValueError, lambda: nonlinsolve([x**2 -1]))


def test_trig_system():
    # TODO: add more simple testcases when solveset returns
    # simplified soln for Trig eq
    assert nonlinsolve([sin(x) - 1, cos(x) -1 ], x) == S.EmptySet
    soln1 = (ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers),)
    soln = FiniteSet(soln1)
    assert dumeq(nonlinsolve([sin(x) - 1, cos(x)], x), soln)


@XFAIL
def test_trig_system_fail():
    # fails because solveset trig solver is not much smart.
    sys = [x + y - pi/2, sin(x) + sin(y) - 1]
    # solveset returns conditionset for sin(x) + sin(y) - 1
    soln_1 = (ImageSet(Lambda(n, n*pi + pi/2), S.Integers),
        ImageSet(Lambda(n, n*pi), S.Integers))
    soln_1 = FiniteSet(soln_1)
    soln_2 = (ImageSet(Lambda(n, n*pi), S.Integers),
        ImageSet(Lambda(n, n*pi+ pi/2), S.Integers))
    soln_2 = FiniteSet(soln_2)
    soln = soln_1 + soln_2
    assert dumeq(nonlinsolve(sys, [x, y]), soln)

    # Add more cases from here
    # http://www.vitutor.com/geometry/trigonometry/equations_systems.html#uno
    sys = [sin(x) + sin(y) - (sqrt(3)+1)/2, sin(x) - sin(y) - (sqrt(3) - 1)/2]
    soln_x = Union(ImageSet(Lambda(n, 2*n*pi + pi/3), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi*Rational(2, 3)), S.Integers))
    soln_y = Union(ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi*Rational(5, 6)), S.Integers))
    assert dumeq(nonlinsolve(sys, [x, y]), FiniteSet((soln_x, soln_y)))


def test_nonlinsolve_positive_dimensional():
    x, y, a, b, c, d = symbols('x, y, a, b, c, d', extended_real=True)
    assert nonlinsolve([x*y, x*y - x], [x, y]) == FiniteSet((0, y))

    system = [a**2 + a*c, a - b]
    assert nonlinsolve(system, [a, b]) == FiniteSet((0, 0), (-c, -c))
    # here (a= 0, b = 0) is independent soln so both is printed.
    # if symbols = [a, b, c] then only {a : -c ,b : -c}

    eq1 =  a + b + c + d
    eq2 = a*b + b*c + c*d + d*a
    eq3 = a*b*c + b*c*d + c*d*a + d*a*b
    eq4 = a*b*c*d - 1
    system = [eq1, eq2, eq3, eq4]
    sol1 = (-1/d, -d, 1/d, FiniteSet(d) - FiniteSet(0))
    sol2 = (1/d, -d, -1/d, FiniteSet(d) - FiniteSet(0))
    soln = FiniteSet(sol1, sol2)
    assert nonlinsolve(system, [a, b, c, d]) == soln

    assert nonlinsolve([x**4 - 3*x**2 + y*x, x*z**2, y*z - 1], [x, y, z]) == \
           {(0, 1/z, z)}


def test_nonlinsolve_polysys():
    x, y, z = symbols('x, y, z', real=True)
    assert nonlinsolve([x**2 + y - 2, x**2 + y], [x, y]) == S.EmptySet

    s = (-y + 2, y)
    assert nonlinsolve([(x + y)**2 - 4, x + y - 2], [x, y]) == FiniteSet(s)

    system = [x**2 - y**2]
    soln_real = FiniteSet((-y, y), (y, y))
    soln_complex = FiniteSet((-Abs(y), y), (Abs(y), y))
    soln =soln_real + soln_complex
    assert nonlinsolve(system, [x, y]) == soln

    system = [x**2 - y**2]
    soln_real= FiniteSet((y, -y), (y, y))
    soln_complex = FiniteSet((y, -Abs(y)), (y, Abs(y)))
    soln = soln_real + soln_complex
    assert nonlinsolve(system, [y, x]) == soln

    system = [x**2 + y - 3, x - y - 4]
    assert nonlinsolve(system, (x, y)) != nonlinsolve(system, (y, x))

    assert nonlinsolve([-x**2 - y**2 + z, -2*x, -2*y, S.One], [x, y, z]) == S.EmptySet
    assert nonlinsolve([x + y + z, S.One, S.One, S.One], [x, y, z]) == S.EmptySet

    system = [-x**2*z**2 + x*y*z + y**4, -2*x*z**2 + y*z, x*z + 4*y**3, -2*x**2*z + x*y]
    assert nonlinsolve(system, [x, y, z]) == FiniteSet((0, 0, z), (x, 0, 0))


def test_nonlinsolve_using_substitution():
    x, y, z, n = symbols('x, y, z, n', real = True)
    system = [(x + y)*n - y**2 + 2]
    s_x = (n*y - y**2 + 2)/n
    soln = (-s_x, y)
    assert nonlinsolve(system, [x, y]) == FiniteSet(soln)

    system = [z**2*x**2 - z**2*y**2/exp(x)]
    soln_real_1 = (y, x, 0)
    soln_real_2 = (-exp(x/2)*Abs(x), x, z)
    soln_real_3 = (exp(x/2)*Abs(x), x, z)
    soln_complex_1 = (-x*exp(x/2), x, z)
    soln_complex_2 = (x*exp(x/2), x, z)
    syms = [y, x, z]
    soln = FiniteSet(soln_real_1, soln_complex_1, soln_complex_2,\
        soln_real_2, soln_real_3)
    assert nonlinsolve(system,syms) == soln


def test_nonlinsolve_complex():
    n = Dummy('n')
    assert dumeq(nonlinsolve([exp(x) - sin(y), 1/y - 3], [x, y]), {
        (ImageSet(Lambda(n, 2*n*I*pi + log(sin(Rational(1, 3)))), S.Integers), Rational(1, 3))})

    system = [exp(x) - sin(y), 1/exp(y) - 3]
    assert dumeq(nonlinsolve(system, [x, y]), {
        (ImageSet(Lambda(n, I*(2*n*pi + pi)
                         + log(sin(log(3)))), S.Integers), -log(3)),
        (ImageSet(Lambda(n, I*(2*n*pi + arg(sin(2*n*I*pi - log(3))))
                         + log(Abs(sin(2*n*I*pi - log(3))))), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi - log(3)), S.Integers))})

    system = [exp(x) - sin(y), y**2 - 4]
    assert dumeq(nonlinsolve(system, [x, y]), {
        (ImageSet(Lambda(n, I*(2*n*pi + pi) + log(sin(2))), S.Integers), -2),
        (ImageSet(Lambda(n, 2*n*I*pi + log(sin(2))), S.Integers), 2)})

    system = [exp(x) - 2, y ** 2 - 2]
    assert dumeq(nonlinsolve(system, [x, y]), {
        (log(2), -sqrt(2)), (log(2), sqrt(2)),
        (ImageSet(Lambda(n, 2*n*I*pi + log(2)), S.Integers), -sqrt(2)),
        (ImageSet(Lambda(n, 2 * n * I * pi + log(2)), S.Integers), sqrt(2))})


def test_nonlinsolve_radical():
    assert nonlinsolve([sqrt(y) - x - z, y - 1], [x, y, z]) == {(1 - z, 1, z)}


def test_nonlinsolve_inexact():
    sol = [(-1.625, -1.375), (1.625, 1.375)]
    res = nonlinsolve([(x + y)**2 - 9, x**2 - y**2 - 0.75], [x, y])
    assert all(abs(res.args[i][j]-sol[i][j]) < 1e-9
               for i in range(2) for j in range(2))

    assert nonlinsolve([(x + y)**2 - 9, (x + y)**2 - 0.75], [x, y]) == S.EmptySet

    assert nonlinsolve([y**2 + (x - 0.5)**2 - 0.0625, 2*x - 1.0, 2*y], [x, y]) == \
           S.EmptySet

    res = nonlinsolve([x**2 + y - 0.5, (x + y)**2, log(z)], [x, y, z])
    sol = [(-0.366025403784439, 0.366025403784439, 1),
           (-0.366025403784439, 0.366025403784439, 1),
           (1.36602540378444, -1.36602540378444, 1)]
    assert all(abs(res.args[i][j]-sol[i][j]) < 1e-9
               for i in range(3) for j in range(3))

    res = nonlinsolve([y - x**2, x**5 - x + 1.0], [x, y])
    sol = [(-1.16730397826142, 1.36259857766493),
           (-0.181232444469876 - 1.08395410131771*I,
            -1.14211129483496 + 0.392895302949911*I),
           (-0.181232444469876 + 1.08395410131771*I,
            -1.14211129483496 - 0.392895302949911*I),
           (0.764884433600585 - 0.352471546031726*I,
            0.460812006002492 - 0.539199997693599*I),
           (0.764884433600585 + 0.352471546031726*I,
            0.460812006002492 + 0.539199997693599*I)]
    assert all(abs(res.args[i][j] - sol[i][j]) < 1e-9
               for i in range(5) for j in range(2))

@XFAIL
def test_solve_nonlinear_trans():
    # After the transcendental equation solver these will work
    x, y = symbols('x, y', real=True)
    soln1 = FiniteSet((2*LambertW(y/2), y))
    soln2 = FiniteSet((-x*sqrt(exp(x)), y), (x*sqrt(exp(x)), y))
    soln3 = FiniteSet((x*exp(x/2), x))
    soln4 = FiniteSet(2*LambertW(y/2), y)
    assert nonlinsolve([x**2 - y**2/exp(x)], [x, y]) == soln1
    assert nonlinsolve([x**2 - y**2/exp(x)], [y, x]) == soln2
    assert nonlinsolve([x**2 - y**2/exp(x)], [y, x]) == soln3
    assert nonlinsolve([x**2 - y**2/exp(x)], [x, y]) == soln4


def test_nonlinsolve_issue_25182():
    a1, b1, c1, ca, cb, cg = symbols('a1, b1, c1, ca, cb, cg')
    eq1 = a1*a1 + b1*b1 - 2.*a1*b1*cg - c1*c1
    eq2 = a1*a1 + c1*c1 - 2.*a1*c1*cb - b1*b1
    eq3 = b1*b1 + c1*c1 - 2.*b1*c1*ca - a1*a1
    assert nonlinsolve([eq1, eq2, eq3], [c1, cb, cg]) == FiniteSet(
        (1.0*b1*ca - 1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2),
        -1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1,
        -1.0*b1*(ca - 1)*(ca + 1)/a1 + 1.0*ca*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1),
        (1.0*b1*ca + 1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2),
        1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1,
        -1.0*b1*(ca - 1)*(ca + 1)/a1 - 1.0*ca*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1))


def test_issue_14642():
    x = Symbol('x')
    n1 = 0.5*x**3+x**2+0.5+I #add I in the Polynomials
    solution = solveset(n1, x)
    assert abs(solution.args[0] - (-2.28267560928153 - 0.312325580497716*I)) <= 1e-9
    assert abs(solution.args[1] - (-0.297354141679308 + 1.01904778618762*I)) <= 1e-9
    assert abs(solution.args[2] - (0.580029750960839 - 0.706722205689907*I)) <= 1e-9

    # Symbolic
    n1 = S.Half*x**3+x**2+S.Half+I
    res = FiniteSet(-((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)
            /2)/2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(
            S(172)/49)/2)/2 + S(43)/2))/3)/3 - S(2)/3 - 4*cos(atan((27 +
            3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*
            31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 + S(43)/2))/3)/(3*((3*
            sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 + S(43)/2)**2 +
            (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)/2)**2)**(S(1)/
            6)) + I*(-((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/
            2)/2)**2)**(S(1)/6)*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(
            atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)
            /2)/2 + S(43)/2))/3)/3 + 4*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)
            /49)/2)/2 + S(43)/2))/3)/(3*((3*sqrt(3)*31985**(S(1)/4)*sin(atan(
            S(172)/49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)**2)**(S(1)/6))), -S(2)/3 - sqrt(3)*((3*
            sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 + S(43)/2)**2 +
            (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)/2)**2)**(S(1)
            /6)*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)
            /2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 + S(43)/2))
            /3)/6 - 4*re(1/((-S(1)/2 - sqrt(3)*I/2)*(S(43)/2 + 27*I + sqrt(-256 +
            (43 + 54*I)**2)/2)**(S(1)/3)))/3 + ((3*sqrt(3)*31985**(S(1)/4)*sin(
            atan(S(172)/49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*
            31985**(S(1)/4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*
            sin(atan(S(172)/49)/2)/2 + S(43)/2))/3)/6 + I*(-4*im(1/((-S(1)/2 -
            sqrt(3)*I/2)*(S(43)/2 + 27*I + sqrt(-256 + (43 + 54*I)**2)/2)**(S(1)/
            3)))/3 + ((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)
            /2)**2)**(S(1)/6)*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(
            S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2))/3)/6 + sqrt(3)*((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/
            49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(
            S(172)/49)/2)/2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*31985**(S(1)/
            4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(
            S(172)/49)/2)/2 + S(43)/2))/3)/6), -S(2)/3 - 4*re(1/((-S(1)/2 +
            sqrt(3)*I/2)*(S(43)/2 + 27*I + sqrt(-256 + (43 + 54*I)**2)/2)**(S(1)
            /3)))/3 + sqrt(3)*((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)
            /2)**2)**(S(1)/6)*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(
            S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2))/3)/6 + ((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)
            /2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(
            S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2))/3)/6 + I*(-sqrt(3)*((3*sqrt(3)*31985**(S(1)/4)*sin(atan(
            S(172)/49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(
            atan(S(172)/49)/2)/2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*31985**(
            S(1)/4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(
            atan(S(172)/49)/2)/2 + S(43)/2))/3)/6 + ((3*sqrt(3)*31985**(S(1)/4)*
            sin(atan(S(172)/49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)**2)**(S(1)/6)*sin(atan((27 + 3*sqrt(3)*31985**(
            S(1)/4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(
            atan(S(172)/49)/2)/2 + S(43)/2))/3)/6 - 4*im(1/((-S(1)/2 + sqrt(3)*I/2)*
            (S(43)/2 + 27*I + sqrt(-256 + (43 + 54*I)**2)/2)**(S(1)/3)))/3))

    assert solveset(n1, x) == res


def test_issue_13961():
    V = (ax, bx, cx, gx, jx, lx, mx, nx, q) = symbols('ax bx cx gx jx lx mx nx q')
    S = (ax*q - lx*q - mx, ax - gx*q - lx, bx*q**2 + cx*q - jx*q - nx, q*(-ax*q + lx*q + mx), q*(-ax + gx*q + lx))

    sol = FiniteSet((lx + mx/q, (-cx*q + jx*q + nx)/q**2, cx, mx/q**2, jx, lx, mx, nx, Complement({q}, {0})),
                    (lx + mx/q, (cx*q - jx*q - nx)/q**2*-1, cx, mx/q**2, jx, lx, mx, nx, Complement({q}, {0})))
    assert nonlinsolve(S, *V) == sol
    # The two solutions are in fact identical, so even better if only one is returned


def test_issue_14541():
    solutions = solveset(sqrt(-x**2 - 2.0), x)
    assert abs(solutions.args[0]+1.4142135623731*I) <= 1e-9
    assert abs(solutions.args[1]-1.4142135623731*I) <= 1e-9


def test_issue_13396():
    expr = -2*y*exp(-x**2 - y**2)*Abs(x)
    sol = FiniteSet(0)

    assert solveset(expr, y, domain=S.Reals) == sol

    # Related type of equation also solved here
    assert solveset(atan(x**2 - y**2)-pi/2, y, S.Reals) is S.EmptySet


def test_issue_12032():
    sol = FiniteSet(-sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                          2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2 +
                    sqrt(Abs(-2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                             2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                             2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                                    2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2,
                    -sqrt(Abs(-2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                              2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                              2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                                     2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2 -
                    sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                         2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2,
                    sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                         2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2 -
                    I*sqrt(Abs(-2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                                       2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) -
                               2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                               2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2,
                    sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                         2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2 +
                    I*sqrt(Abs(-2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                                       2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) -
                               2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                               2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1,3)))))/2)
    assert solveset(x**4 + x - 1, x) == sol


def test_issue_10876():
    assert solveset(1/sqrt(x), x) == S.EmptySet


def test_issue_19050():
    # test_issue_19050 --> TypeError removed
    assert dumeq(nonlinsolve([x + y, sin(y)], [x, y]),
        FiniteSet((ImageSet(Lambda(n, -2*n*pi), S.Integers), ImageSet(Lambda(n, 2*n*pi), S.Integers)),\
             (ImageSet(Lambda(n, -2*n*pi - pi), S.Integers), ImageSet(Lambda(n, 2*n*pi + pi), S.Integers))))
    assert dumeq(nonlinsolve([x + y, sin(y) + cos(y)], [x, y]),
        FiniteSet((ImageSet(Lambda(n, -2*n*pi - 3*pi/4), S.Integers), ImageSet(Lambda(n, 2*n*pi + 3*pi/4), S.Integers)), \
            (ImageSet(Lambda(n, -2*n*pi - 7*pi/4), S.Integers), ImageSet(Lambda(n, 2*n*pi + 7*pi/4), S.Integers))))


def test_issue_16618():
    eqn = [sin(x)*sin(y), cos(x)*cos(y) - 1]
    # nonlinsolve's answer is still suspicious since it contains only three
    # distinct Dummys instead of 4. (Both 'x' ImageSets share the same Dummy.)
    ans = FiniteSet((ImageSet(Lambda(n, 2*n*pi), S.Integers), ImageSet(Lambda(n, 2*n*pi), S.Integers)),
        (ImageSet(Lambda(n, 2*n*pi + pi), S.Integers), ImageSet(Lambda(n, 2*n*pi + pi), S.Integers)))
    sol = nonlinsolve(eqn, [x, y])

    for i0, j0 in zip(ordered(sol), ordered(ans)):
        assert len(i0) == len(j0) == 2
        assert all(a.dummy_eq(b) for a, b in zip(i0, j0))
    assert len(sol) == len(ans)


def test_issue_17566():
    assert nonlinsolve([32*(2**x)/2**(-y) - 4**y, 27*(3**x) - S(1)/3**y], x, y) ==\
        FiniteSet((-log(81)/log(3), 1))


def test_issue_16643():
    n = Dummy('n')
    assert solveset(x**2*sin(x), x).dummy_eq(Union(ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
                                                   ImageSet(Lambda(n, 2*n*pi), S.Integers)))


def test_issue_19587():
    n,m = symbols('n m')
    assert nonlinsolve([32*2**m*2**n - 4**n, 27*3**m - 3**(-n)], m, n) ==\
        FiniteSet((-log(81)/log(3), 1))


def test_issue_5132_1():
    system = [sqrt(x**2 + y**2) - sqrt(10), x + y - 4]
    assert nonlinsolve(system, [x, y]) == FiniteSet((1, 3), (3, 1))

    n = Dummy('n')
    eqs = [exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3]
    s_real_y = -log(3)
    s_real_z = sqrt(-exp(2*x) - sin(log(3)))
    soln_real = FiniteSet((s_real_y, s_real_z), (s_real_y, -s_real_z))
    lam = Lambda(n, 2*n*I*pi + -log(3))
    s_complex_y = ImageSet(lam, S.Integers)
    lam = Lambda(n, sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_1 = ImageSet(lam, S.Integers)
    lam = Lambda(n, -sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_2 = ImageSet(lam, S.Integers)
    soln_complex = FiniteSet(
                                            (s_complex_y, s_complex_z_1),
                                            (s_complex_y, s_complex_z_2)
                                        )
    soln = soln_real + soln_complex
    assert dumeq(nonlinsolve(eqs, [y, z]), soln)


def test_issue_5132_2():
    x, y = symbols('x, y', real=True)
    eqs = [exp(x)**2 - sin(y) + z**2]
    n = Dummy('n')
    soln_real = (log(-z**2 + sin(y))/2, z)
    lam = Lambda( n, I*(2*n*pi + arg(-z**2 + sin(y)))/2 + log(Abs(z**2 - sin(y)))/2)
    img = ImageSet(lam, S.Integers)
    # not sure about the complex soln. But it looks correct.
    soln_complex = (img, z)
    soln = FiniteSet(soln_real, soln_complex)
    assert dumeq(nonlinsolve(eqs, [x, z]), soln)

    system = [r - x**2 - y**2, tan(t) - y/x]
    s_x = sqrt(r/(tan(t)**2 + 1))
    s_y = sqrt(r/(tan(t)**2 + 1))*tan(t)
    soln = FiniteSet((s_x, s_y), (-s_x, -s_y))
    assert nonlinsolve(system, [x, y]) == soln


def test_issue_6752():
    a, b = symbols('a, b', real=True)
    assert nonlinsolve([a**2 + a, a - b], [a, b]) == {(-1, -1), (0, 0)}


@SKIP("slow")
def test_issue_5114_solveset():
    # slow testcase
    from sympy.abc import o, p

    # there is no 'a' in the equation set but this is how the
    # problem was originally posed
    syms = [a, b, c, f, h, k, n]
    eqs = [b + r/d - c/d,
    c*(1/d + 1/e + 1/g) - f/g - r/d,
        f*(1/g + 1/i + 1/j) - c/g - h/i,
        h*(1/i + 1/l + 1/m) - f/i - k/m,
        k*(1/m + 1/o + 1/p) - h/m - n/p,
        n*(1/p + 1/q) - k/p]
    assert len(nonlinsolve(eqs, syms)) == 1


@SKIP("Hangs")
def _test_issue_5335():
    # Not able to check zero dimensional system.
    # is_zero_dimensional Hangs
    lam, a0, conc = symbols('lam a0 conc')
    eqs = [lam + 2*y - a0*(1 - x/2)*x - 0.005*x/2*x,
           a0*(1 - x/2)*x - 1*y - 0.743436700916726*y,
           x + y - conc]
    sym = [x, y, a0]
    # there are 4 solutions but only two are valid
    assert len(nonlinsolve(eqs, sym)) == 2
    # float
    eqs = [lam + 2*y - a0*(1 - x/2)*x - 0.005*x/2*x,
           a0*(1 - x/2)*x - 1*y - 0.743436700916726*y,
           x + y - conc]
    sym = [x, y, a0]
    assert len(nonlinsolve(eqs, sym)) == 2


def test_issue_2777():
    # the equations represent two circles
    x, y = symbols('x y', real=True)
    e1, e2 = sqrt(x**2 + y**2) - 10, sqrt(y**2 + (-x + 10)**2) - 3
    a, b = Rational(191, 20), 3*sqrt(391)/20
    ans = {(a, -b), (a, b)}
    assert nonlinsolve((e1, e2), (x, y)) == ans
    assert nonlinsolve((e1, e2/(x - a)), (x, y)) == S.EmptySet
    # make the 2nd circle's radius be -3
    e2 += 6
    assert nonlinsolve((e1, e2), (x, y)) == S.EmptySet


def test_issue_8828():
    x1 = 0
    y1 = -620
    r1 = 920
    x2 = 126
    y2 = 276
    x3 = 51
    y3 = 205
    r3 = 104
    v = [x, y, z]

    f1 = (x - x1)**2 + (y - y1)**2 - (r1 - z)**2
    f2 = (x2 - x)**2 + (y2 - y)**2 - z**2
    f3 = (x - x3)**2 + (y - y3)**2 - (r3 - z)**2
    F = [f1, f2, f3]

    g1 = sqrt((x - x1)**2 + (y - y1)**2) + z - r1
    g2 = f2
    g3 = sqrt((x - x3)**2 + (y - y3)**2) + z - r3
    G = [g1, g2, g3]

    # both soln same
    A = nonlinsolve(F, v)
    B = nonlinsolve(G, v)
    assert A == B


def test_nonlinsolve_conditionset():
    # when solveset failed to solve all the eq
    # return conditionset
    f = Function('f')
    f1 = f(x) - pi/2
    f2 = f(y) - pi*Rational(3, 2)
    intermediate_system = Eq(2*f(x) - pi, 0) & Eq(2*f(y) - 3*pi, 0)
    syms = Tuple(x, y)
    soln = ConditionSet(
        syms,
        intermediate_system,
        S.Complexes**2)
    assert nonlinsolve([f1, f2], [x, y]) == soln


def test_substitution_basic():
    assert substitution([], [x, y]) == S.EmptySet
    assert substitution([], []) == S.EmptySet
    system = [2*x**2 + 3*y**2 - 30, 3*x**2 - 2*y**2 - 19]
    soln = FiniteSet((-3, -2), (-3, 2), (3, -2), (3, 2))
    assert substitution(system, [x, y]) == soln

    soln = FiniteSet((-1, 1))
    assert substitution([x + y], [x], [{y: 1}], [y], set(), [x, y]) == soln
    assert substitution(
        [x + y], [x], [{y: 1}], [y],
        {x + 1}, [y, x]) == S.EmptySet


def test_substitution_incorrect():
    # the solutions in the following two tests are incorrect. The
    # correct result is EmptySet in both cases.
    assert substitution([h - 1, k - 1, f - 2, f - 4, -2 * k],
                        [h, k, f]) == {(1, 1, f)}
    assert substitution([x + y + z, S.One, S.One, S.One], [x, y, z]) == \
                        {(-y - z, y, z)}

    # the correct result in the test below is {(-I, I, I, -I),
    # (I, -I, -I, I)}
    assert substitution([a - d, b + d, c + d, d**2 + 1], [a, b, c, d]) == \
                        {(d, -d, -d, d)}

    # the result in the test below is incomplete. The complete result
    # is {(0, b), (log(2), 2)}
    assert substitution([a*(a - log(b)), a*(b - 2)], [a, b]) == \
           {(0, b)}

    # The system in the test below is zero-dimensional, so the result
    # should have no free symbols
    assert substitution([-k*y + 6*x - 4*y, -81*k + 49*y**2 - 270,
                         -3*k*z + k + z**3, k**2 - 2*k + 4],
                        [x, y, z, k]).free_symbols == {z}


def test_substitution_redundant():
    # the third and fourth solutions are redundant in the test below
    assert substitution([x**2 - y**2, z - 1], [x, z]) == \
           {(-y, 1), (y, 1), (-sqrt(y**2), 1), (sqrt(y**2), 1)}

    # the system below has three solutions. Two of the solutions
    # returned by substitution are redundant.
    res = substitution([x - y, y**3 - 3*y**2 + 1], [x, y])
    assert len(res) == 5


def test_issue_5132_substitution():
    x, y, z, r, t = symbols('x, y, z, r, t', real=True)
    system = [r - x**2 - y**2, tan(t) - y/x]
    s_x_1 = Complement(FiniteSet(-sqrt(r/(tan(t)**2 + 1))), FiniteSet(0))
    s_x_2 = Complement(FiniteSet(sqrt(r/(tan(t)**2 + 1))), FiniteSet(0))
    s_y = sqrt(r/(tan(t)**2 + 1))*tan(t)
    soln = FiniteSet((s_x_2, s_y)) + FiniteSet((s_x_1, -s_y))
    assert substitution(system, [x, y]) == soln

    n = Dummy('n')
    eqs = [exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3]
    s_real_y = -log(3)
    s_real_z = sqrt(-exp(2*x) - sin(log(3)))
    soln_real = FiniteSet((s_real_y, s_real_z), (s_real_y, -s_real_z))
    lam = Lambda(n, 2*n*I*pi + -log(3))
    s_complex_y = ImageSet(lam, S.Integers)
    lam = Lambda(n, sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_1 = ImageSet(lam, S.Integers)
    lam = Lambda(n, -sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_2 = ImageSet(lam, S.Integers)
    soln_complex = FiniteSet(
        (s_complex_y, s_complex_z_1),
        (s_complex_y, s_complex_z_2))
    soln = soln_real + soln_complex
    assert dumeq(substitution(eqs, [y, z]), soln)


def test_raises_substitution():
    raises(ValueError, lambda: substitution([x**2 -1], []))
    raises(TypeError, lambda: substitution([x**2 -1]))
    raises(ValueError, lambda: substitution([x**2 -1], [sin(x)]))
    raises(TypeError, lambda: substitution([x**2 -1], x))
    raises(TypeError, lambda: substitution([x**2 -1], 1))


def test_issue_21022():
    from sympy.core.sympify import sympify

    eqs = [
    'k-16',
    'p-8',
    'y*y+z*z-x*x',
    'd - x + p',
    'd*d+k*k-y*y',
    'z*z-p*p-k*k',
    'abc-efg',
    ]
    efg = Symbol('efg')
    eqs = [sympify(x) for x in eqs]

    syb = list(ordered(set.union(*[x.free_symbols for x in eqs])))
    res = nonlinsolve(eqs, syb)

    ans = FiniteSet(
    (efg, 32, efg, 16, 8, 40, -16*sqrt(5), -8*sqrt(5)),
    (efg, 32, efg, 16, 8, 40, -16*sqrt(5), 8*sqrt(5)),
    (efg, 32, efg, 16, 8, 40, 16*sqrt(5), -8*sqrt(5)),
    (efg, 32, efg, 16, 8, 40, 16*sqrt(5), 8*sqrt(5)),
    )
    assert len(res) == len(ans) == 4
    assert res == ans
    for result in res.args:
        assert len(result) == 8


def test_issue_17940():
    n = Dummy('n')
    k1 = Dummy('k1')
    sol = ImageSet(Lambda(((k1, n),), I*(2*k1*pi + arg(2*n*I*pi + log(5)))
                          + log(Abs(2*n*I*pi + log(5)))),
                   ProductSet(S.Integers, S.Integers))
    assert solveset(exp(exp(x)) - 5, x).dummy_eq(sol)


def test_issue_17906():
    assert solveset(7**(x**2 - 80) - 49**x, x) == FiniteSet(-8, 10)


@XFAIL
def test_issue_17933():
    eq1 = x*sin(45) - y*cos(q)
    eq2 = x*cos(45) - y*sin(q)
    eq3 = 9*x*sin(45)/10 + y*cos(q)
    eq4 = 9*x*cos(45)/10 + y*sin(z) - z
    assert nonlinsolve([eq1, eq2, eq3, eq4], x, y, z, q) ==\
        FiniteSet((0, 0, 0, q))

def test_issue_17933_bis():
    # nonlinsolve's result depends on the 'default_sort_key' ordering of
    # the unknowns.
    eq1 = x*sin(45) - y*cos(q)
    eq2 = x*cos(45) - y*sin(q)
    eq3 = 9*x*sin(45)/10 + y*cos(q)
    eq4 = 9*x*cos(45)/10 + y*sin(z) - z
    zz = Symbol('zz')
    eqs = [e.subs(q, zz) for e in (eq1, eq2, eq3, eq4)]
    assert nonlinsolve(eqs, x, y, z, zz) == FiniteSet((0, 0, 0, zz))


def test_issue_14565():
    # removed redundancy
    assert dumeq(nonlinsolve([k + m, k + m*exp(-2*pi*k)], [k, m]) ,
        FiniteSet((-n*I, ImageSet(Lambda(n, n*I), S.Integers))))


# end of tests for nonlinsolve


def test_issue_9556():
    b = Symbol('b', positive=True)

    assert solveset(Abs(x) + 1, x, S.Reals) is S.EmptySet
    assert solveset(Abs(x) + b, x, S.Reals) is S.EmptySet
    assert solveset(Eq(b, -1), b, S.Reals) is S.EmptySet


def test_issue_9611():
    assert solveset(Eq(x - x + a, a), x, S.Reals) == S.Reals
    assert solveset(Eq(y - y + a, a), y) == S.Complexes


def test_issue_9557():
    assert solveset(x**2 + a, x, S.Reals) == Intersection(S.Reals,
        FiniteSet(-sqrt(-a), sqrt(-a)))


def test_issue_9778():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert solveset(x**3 + 1, x, S.Reals) == FiniteSet(-1)
    assert solveset(x**Rational(3, 5) + 1, x, S.Reals) == S.EmptySet
    assert solveset(x**3 + y, x, S.Reals) == \
        FiniteSet(-Abs(y)**Rational(1, 3)*sign(y))


def test_issue_10214():
    assert solveset(x**Rational(3, 2) + 4, x, S.Reals) == S.EmptySet
    assert solveset(x**(Rational(-3, 2)) + 4, x, S.Reals) == S.EmptySet

    ans = FiniteSet(-2**Rational(2, 3))
    assert solveset(x**(S(3)) + 4, x, S.Reals) == ans
    assert (x**(S(3)) + 4).subs(x,list(ans)[0]) == 0 # substituting ans and verifying the result.
    assert (x**(S(3)) + 4).subs(x,-(-2)**Rational(2, 3)) == 0


def test_issue_9849():
    assert solveset(Abs(sin(x)) + 1, x, S.Reals) == S.EmptySet


def test_issue_9953():
    assert linsolve([ ], x) == S.EmptySet


def test_issue_9913():
    assert solveset(2*x + 1/(x - 10)**2, x, S.Reals) == \
        FiniteSet(-(3*sqrt(24081)/4 + Rational(4027, 4))**Rational(1, 3)/3 - 100/
                (3*(3*sqrt(24081)/4 + Rational(4027, 4))**Rational(1, 3)) + Rational(20, 3))


def test_issue_10397():
    assert solveset(sqrt(x), x, S.Complexes) == FiniteSet(0)


def test_issue_14987():
    raises(ValueError, lambda: linear_eq_to_matrix(
        [x**2], x))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [x*(-3/x + 1) + 2*y - a], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [(x**2 - 3*x)/(x - 3) - 3], x))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [(x + 1)**3 - x**3 - 3*x**2 + 7], x))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [x*(1/x + 1) + y], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [(x + 1)*y], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [Eq(1/x, 1/x + y)], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [Eq(y/x, y/x + y)], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [Eq(x*(x + 1), x**2 + y)], [x, y]))


def test_simplification():
    eq = x + (a - b)/(-2*a + 2*b)
    assert solveset(eq, x) == FiniteSet(S.Half)
    assert solveset(eq, x, S.Reals) == Intersection({-((a - b)/(-2*a + 2*b))}, S.Reals)
    # So that ap - bn is not zero:
    ap = Symbol('ap', positive=True)
    bn = Symbol('bn', negative=True)
    eq = x + (ap - bn)/(-2*ap + 2*bn)
    assert solveset(eq, x) == FiniteSet(S.Half)
    assert solveset(eq, x, S.Reals) == FiniteSet(S.Half)


def test_integer_domain_relational():
    eq1 = 2*x + 3 > 0
    eq2 = x**2 + 3*x - 2 >= 0
    eq3 = x + 1/x > -2 + 1/x
    eq4 = x + sqrt(x**2 - 5) > 0
    eq = x + 1/x > -2 + 1/x
    eq5 = eq.subs(x,log(x))
    eq6 = log(x)/x <= 0
    eq7 = log(x)/x < 0
    eq8 = x/(x-3) < 3
    eq9 = x/(x**2-3) < 3

    assert solveset(eq1, x, S.Integers) == Range(-1, oo, 1)
    assert solveset(eq2, x, S.Integers) == Union(Range(-oo, -3, 1), Range(1, oo, 1))
    assert solveset(eq3, x, S.Integers) == Union(Range(-1, 0, 1), Range(1, oo, 1))
    assert solveset(eq4, x, S.Integers) == Range(3, oo, 1)
    assert solveset(eq5, x, S.Integers) == Range(2, oo, 1)
    assert solveset(eq6, x, S.Integers) == Range(1, 2, 1)
    assert solveset(eq7, x, S.Integers) == S.EmptySet
    assert solveset(eq8, x, domain=Range(0,5)) == Range(0, 3, 1)
    assert solveset(eq9, x, domain=Range(0,5)) == Union(Range(0, 2, 1), Range(2, 5, 1))

    # test_issue_19794
    assert solveset(x + 2 < 0, x, S.Integers) == Range(-oo, -2, 1)


def test_issue_10555():
    f = Function('f')
    g = Function('g')
    assert solveset(f(x) - pi/2, x, S.Reals).dummy_eq(
        ConditionSet(x, Eq(f(x) - pi/2, 0), S.Reals))
    assert solveset(f(g(x)) - pi/2, g(x), S.Reals).dummy_eq(
        ConditionSet(g(x), Eq(f(g(x)) - pi/2, 0), S.Reals))


def test_issue_8715():
    eq = x + 1/x > -2 + 1/x
    assert solveset(eq, x, S.Reals) == \
        (Interval.open(-2, oo) - FiniteSet(0))
    assert solveset(eq.subs(x,log(x)), x, S.Reals) == \
        Interval.open(exp(-2), oo) - FiniteSet(1)


def test_issue_11174():
    eq = z**2 + exp(2*x) - sin(y)
    soln = Intersection(S.Reals, FiniteSet(log(-z**2 + sin(y))/2))
    assert solveset(eq, x, S.Reals) == soln

    eq = sqrt(r)*Abs(tan(t))/sqrt(tan(t)**2 + 1) + x*tan(t)
    s = -sqrt(r)*Abs(tan(t))/(sqrt(tan(t)**2 + 1)*tan(t))
    soln = Intersection(S.Reals, FiniteSet(s))
    assert solveset(eq, x, S.Reals) == soln


def test_issue_11534():
    # eq1 and eq2 should not have the same solutions because squaring both
    # sides of the radical equation introduces a spurious solution branch.
    # The equations have a symbolic parameter y and it is easy to see that for
    # y != 0 the solution s1 will not be valid for eq1.
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    eq1 = -y + x/sqrt(-x**2 + 1)
    eq2 = -y**2 + x**2/(-x**2 + 1)

    # We get a ConditionSet here because s1 works in eq1 if y is equal to zero
    # although not for any other value of y. That case is redundant though
    # because if y=0 then s1=s2 so the solution for eq1 could just be returned
    # as s2 - {-1, 1}. In fact we have
    #   |y/sqrt(y**2 + 1)| < 1
    # So the complements are not needed either. The ideal output here would be
    #   sol1 = s2
    #   sol2 = s1 | s2.
    s1, s2 = FiniteSet(-y/sqrt(y**2 + 1)), FiniteSet(y/sqrt(y**2 + 1))
    cset = ConditionSet(x, Eq(eq1, 0), s1)
    sol1 = (s2 - {-1, 1}) | (cset - {-1, 1})
    sol2 = (s1 | s2) - {-1, 1}

    assert solveset(eq1, x, S.Reals) == sol1
    assert solveset(eq2, x, S.Reals) == sol2


def test_issue_10477():
    assert solveset((x**2 + 4*x - 3)/x < 2, x, S.Reals) == \
        Union(Interval.open(-oo, -3), Interval.open(0, 1))


def test_issue_10671():
    assert solveset(sin(y), y, Interval(0, pi)) == FiniteSet(0, pi)
    i = Interval(1, 10)
    assert solveset((1/x).diff(x) < 0, x, i) == i


def test_issue_11064():
    eq = x + sqrt(x**2 - 5)
    assert solveset(eq > 0, x, S.Reals) == \
        Interval(sqrt(5), oo)
    assert solveset(eq < 0, x, S.Reals) == \
        Interval(-oo, -sqrt(5))
    assert solveset(eq > sqrt(5), x, S.Reals) == \
        Interval.Lopen(sqrt(5), oo)


def test_issue_12478():
    eq = sqrt(x - 2) + 2
    soln = solveset_real(eq, x)
    assert soln is S.EmptySet
    assert solveset(eq < 0, x, S.Reals) is S.EmptySet
    assert solveset(eq > 0, x, S.Reals) == Interval(2, oo)


def test_issue_12429():
    eq = solveset(log(x)/x <= 0, x, S.Reals)
    sol = Interval.Lopen(0, 1)
    assert eq == sol


def test_issue_19506():
    eq = arg(x + I)
    C = Dummy('C')
    assert solveset(eq).dummy_eq(Intersection(ConditionSet(C, Eq(im(C) + 1, 0), S.Complexes),
                                                 ConditionSet(C, re(C) > 0, S.Complexes)))


def test_solveset_arg():
    assert solveset(arg(x), x, S.Reals)  == Interval.open(0, oo)
    assert solveset(arg(4*x -3), x, S.Reals) == Interval.open(Rational(3, 4), oo)


def test__is_finite_with_finite_vars():
    f = _is_finite_with_finite_vars
    # issue 12482
    assert all(f(1/x) is None for x in (
        Dummy(), Dummy(real=True), Dummy(complex=True)))
    assert f(1/Dummy(real=False)) is True  # b/c it's finite but not 0


def test_issue_13550():
    assert solveset(x**2 - 2*x - 15, symbol = x, domain = Interval(-oo, 0)) == FiniteSet(-3)


def test_issue_13849():
    assert nonlinsolve((t*(sqrt(5) + sqrt(2)) - sqrt(2), t), t) is S.EmptySet


def test_issue_14223():
    assert solveset((Abs(x + Min(x, 2)) - 2).rewrite(Piecewise), x,
        S.Reals) == FiniteSet(-1, 1)
    assert solveset((Abs(x + Min(x, 2)) - 2).rewrite(Piecewise), x,
        Interval(0, 2)) == FiniteSet(1)
    assert solveset(x, x, FiniteSet(1, 2)) is S.EmptySet


def test_issue_10158():
    dom = S.Reals
    assert solveset(x*Max(x, 15) - 10, x, dom) == FiniteSet(Rational(2, 3))
    assert solveset(x*Min(x, 15) - 10, x, dom) == FiniteSet(-sqrt(10), sqrt(10))
    assert solveset(Max(Abs(x - 3) - 1, x + 2) - 3, x, dom) == FiniteSet(-1, 1)
    assert solveset(Abs(x - 1) - Abs(y), x, dom) == FiniteSet(-Abs(y) + 1, Abs(y) + 1)
    assert solveset(Abs(x + 4*Abs(x + 1)), x, dom) == FiniteSet(Rational(-4, 3), Rational(-4, 5))
    assert solveset(2*Abs(x + Abs(x + Max(3, x))) - 2, x, S.Reals) == FiniteSet(-1, -2)
    dom = S.Complexes
    raises(ValueError, lambda: solveset(x*Max(x, 15) - 10, x, dom))
    raises(ValueError, lambda: solveset(x*Min(x, 15) - 10, x, dom))
    raises(ValueError, lambda: solveset(Max(Abs(x - 3) - 1, x + 2) - 3, x, dom))
    raises(ValueError, lambda: solveset(Abs(x - 1) - Abs(y), x, dom))
    raises(ValueError, lambda: solveset(Abs(x + 4*Abs(x + 1)), x, dom))


def test_issue_14300():
    f = 1 - exp(-18000000*x) - y
    a1 = FiniteSet(-log(-y + 1)/18000000)

    assert solveset(f, x, S.Reals) == \
        Intersection(S.Reals, a1)
    assert dumeq(solveset(f, x),
        ImageSet(Lambda(n, -I*(2*n*pi + arg(-y + 1))/18000000 -
            log(Abs(y - 1))/18000000), S.Integers))


def test_issue_14454():
    number = CRootOf(x**4 + x - 1, 2)
    raises(ValueError, lambda: invert_real(number, 0, x))
    assert invert_real(x**2, number, x)  # no error


def test_issue_17882():
    assert solveset(-8*x**2/(9*(x**2 - 1)**(S(4)/3)) + 4/(3*(x**2 - 1)**(S(1)/3)), x, S.Complexes) == \
        FiniteSet(sqrt(3), -sqrt(3))


def test_term_factors():
    assert list(_term_factors(3**x - 2)) == [-2, 3**x]
    expr = 4**(x + 1) + 4**(x + 2) + 4**(x - 1) - 3**(x + 2) - 3**(x + 3)
    assert set(_term_factors(expr)) == {
        3**(x + 2), 4**(x + 2), 3**(x + 3), 4**(x - 1), -1, 4**(x + 1)}


#################### tests for transolve and its helpers ###############

def test_transolve():

    assert _transolve(3**x, x, S.Reals) == S.EmptySet
    assert _transolve(3**x - 9**(x + 5), x, S.Reals) == FiniteSet(-10)


def test_issue_21276():
    eq = (2*x*(y - z) - y*erf(y - z) - y + z*erf(y - z) + z)**2
    assert solveset(eq.expand(), y) == FiniteSet(z, z + erfinv(2*x - 1))


# exponential tests
def test_exponential_real():
    from sympy.abc import y

    e1 = 3**(2*x) - 2**(x + 3)
    e2 = 4**(5 - 9*x) - 8**(2 - x)
    e3 = 2**x + 4**x
    e4 = exp(log(5)*x) - 2**x
    e5 = exp(x/y)*exp(-z/y) - 2
    e6 = 5**(x/2) - 2**(x/3)
    e7 = 4**(x + 1) + 4**(x + 2) + 4**(x - 1) - 3**(x + 2) - 3**(x + 3)
    e8 = -9*exp(-2*x + 5) + 4*exp(3*x + 1)
    e9 = 2**x + 4**x + 8**x - 84
    e10 = 29*2**(x + 1)*615**(x) - 123*2726**(x)

    assert solveset(e1, x, S.Reals) == FiniteSet(
        -3*log(2)/(-2*log(3) + log(2)))
    assert solveset(e2, x, S.Reals) == FiniteSet(Rational(4, 15))
    assert solveset(e3, x, S.Reals) == S.EmptySet
    assert solveset(e4, x, S.Reals) == FiniteSet(0)
    assert solveset(e5, x, S.Reals) == Intersection(
        S.Reals, FiniteSet(y*log(2*exp(z/y))))
    assert solveset(e6, x, S.Reals) == FiniteSet(0)
    assert solveset(e7, x, S.Reals) == FiniteSet(2)
    assert solveset(e8, x, S.Reals) == FiniteSet(-2*log(2)/5 + 2*log(3)/5 + Rational(4, 5))
    assert solveset(e9, x, S.Reals) == FiniteSet(2)
    assert solveset(e10,x, S.Reals) == FiniteSet((-log(29) - log(2) + log(123))/(-log(2726) + log(2) + log(615)))

    assert solveset_real(-9*exp(-2*x + 5) + 2**(x + 1), x) == FiniteSet(
        -((-5 - 2*log(3) + log(2))/(log(2) + 2)))
    assert solveset_real(4**(x/2) - 2**(x/3), x) == FiniteSet(0)
    b = sqrt(6)*sqrt(log(2))/sqrt(log(5))
    assert solveset_real(5**(x/2) - 2**(3/x), x) == FiniteSet(-b, b)

    # coverage test
    C1, C2 = symbols('C1 C2')
    f = Function('f')
    assert solveset_real(C1 + C2/x**2 - exp(-f(x)), f(x)) == Intersection(
        S.Reals, FiniteSet(-log(C1 + C2/x**2)))
    y = symbols('y', positive=True)
    assert solveset_real(x**2 - y**2/exp(x), y) == Intersection(
        S.Reals, FiniteSet(-sqrt(x**2*exp(x)), sqrt(x**2*exp(x))))
    p = Symbol('p', positive=True)
    assert solveset_real((1/p + 1)**(p + 1), p).dummy_eq(
        ConditionSet(x, Eq((1 + 1/x)**(x + 1), 0), S.Reals))
    assert solveset(2**x - 4**x + 12, x, S.Reals) == {2}
    assert solveset(2**x - 2**(2*x) + 12, x, S.Reals) == {2}


@XFAIL
def test_exponential_complex():
    n = Dummy('n')

    assert dumeq(solveset_complex(2**x + 4**x, x),imageset(
        Lambda(n, I*(2*n*pi + pi)/log(2)), S.Integers))
    assert solveset_complex(x**z*y**z - 2, z) == FiniteSet(
        log(2)/(log(x) + log(y)))
    assert dumeq(solveset_complex(4**(x/2) - 2**(x/3), x), imageset(
        Lambda(n, 3*n*I*pi/log(2)), S.Integers))
    assert dumeq(solveset(2**x + 32, x), imageset(
        Lambda(n, (I*(2*n*pi + pi) + 5*log(2))/log(2)), S.Integers))

    eq = (2**exp(y**2/x) + 2)/(x**2 + 15)
    a = sqrt(x)*sqrt(-log(log(2)) + log(log(2) + 2*n*I*pi))
    assert solveset_complex(eq, y) == FiniteSet(-a, a)

    union1 = imageset(Lambda(n, I*(2*n*pi - pi*Rational(2, 3))/log(2)), S.Integers)
    union2 = imageset(Lambda(n, I*(2*n*pi + pi*Rational(2, 3))/log(2)), S.Integers)
    assert dumeq(solveset(2**x + 4**x + 8**x, x), Union(union1, union2))

    eq = 4**(x + 1) + 4**(x + 2) + 4**(x - 1) - 3**(x + 2) - 3**(x + 3)
    res = solveset(eq, x)
    num = 2*n*I*pi - 4*log(2) + 2*log(3)
    den = -2*log(2) + log(3)
    ans = imageset(Lambda(n, num/den), S.Integers)
    assert dumeq(res, ans)


def test_expo_conditionset():

    f1 = (exp(x) + 1)**x - 2
    f2 = (x + 2)**y*x - 3
    f3 = 2**x - exp(x) - 3
    f4 = log(x) - exp(x)
    f5 = 2**x + 3**x - 5**x

    assert solveset(f1, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq((exp(x) + 1)**x - 2, 0), S.Reals))
    assert solveset(f2, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(x*(x + 2)**y - 3, 0), S.Reals))
    assert solveset(f3, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(2**x - exp(x) - 3, 0), S.Reals))
    assert solveset(f4, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(-exp(x) + log(x), 0), S.Reals))
    assert solveset(f5, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(2**x + 3**x - 5**x, 0), S.Reals))


def test_exponential_symbols():
    x, y, z = symbols('x y z', positive=True)
    xr, zr = symbols('xr, zr', real=True)

    assert solveset(z**x - y, x, S.Reals) == Intersection(
        S.Reals, FiniteSet(log(y)/log(z)))

    f1 = 2*x**w - 4*y**w
    f2 = (x/y)**w - 2
    sol1 = Intersection({log(2)/(log(x) - log(y))}, S.Reals)
    sol2 = Intersection({log(2)/log(x/y)}, S.Reals)
    assert solveset(f1, w, S.Reals) == sol1, solveset(f1, w, S.Reals)
    assert solveset(f2, w, S.Reals) == sol2, solveset(f2, w, S.Reals)

    assert solveset(x**x, x, Interval.Lopen(0,oo)).dummy_eq(
        ConditionSet(w, Eq(w**w, 0), Interval.open(0, oo)))
    assert solveset(x**y - 1, y, S.Reals) == FiniteSet(0)
    assert solveset(exp(x/y)*exp(-z/y) - 2, y, S.Reals) == \
    Complement(ConditionSet(y, Eq(im(x)/y, 0) & Eq(im(z)/y, 0), \
    Complement(Intersection(FiniteSet((x - z)/log(2)), S.Reals), FiniteSet(0))), FiniteSet(0))
    assert solveset(exp(xr/y)*exp(-zr/y) - 2, y, S.Reals) == \
        Complement(FiniteSet((xr - zr)/log(2)), FiniteSet(0))

    assert solveset(a**x - b**x, x).dummy_eq(ConditionSet(
        w, Ne(a, 0) & Ne(b, 0), FiniteSet(0)))


def test_ignore_assumptions():
    # make sure assumptions are ignored
    xpos = symbols('x', positive=True)
    x = symbols('x')
    assert solveset_complex(xpos**2 - 4, xpos
        ) == solveset_complex(x**2 - 4, x)


@XFAIL
def test_issue_10864():
    assert solveset(x**(y*z) - x, x, S.Reals) == FiniteSet(1)


@XFAIL
def test_solve_only_exp_2():
    assert solveset_real(sqrt(exp(x)) + sqrt(exp(-x)) - 4, x) == \
        FiniteSet(2*log(-sqrt(3) + 2), 2*log(sqrt(3) + 2))


def test_is_exponential():
    assert _is_exponential(y, x) is False
    assert _is_exponential(3**x - 2, x) is True
    assert _is_exponential(5**x - 7**(2 - x), x) is True
    assert _is_exponential(sin(2**x) - 4*x, x) is False
    assert _is_exponential(x**y - z, y) is True
    assert _is_exponential(x**y - z, x) is False
    assert _is_exponential(2**x + 4**x - 1, x) is True
    assert _is_exponential(x**(y*z) - x, x) is False
    assert _is_exponential(x**(2*x) - 3**x, x) is False
    assert _is_exponential(x**y - y*z, y) is False
    assert _is_exponential(x**y - x*z, y) is True


def test_solve_exponential():
    assert _solve_exponential(3**(2*x) - 2**(x + 3), 0, x, S.Reals) == \
        FiniteSet(-3*log(2)/(-2*log(3) + log(2)))
    assert _solve_exponential(2**y + 4**y, 1, y, S.Reals) == \
        FiniteSet(log(Rational(-1, 2) + sqrt(5)/2)/log(2))
    assert _solve_exponential(2**y + 4**y, 0, y, S.Reals) == \
        S.EmptySet
    assert _solve_exponential(2**x + 3**x - 5**x, 0, x, S.Reals) == \
        ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), S.Reals)

# end of exponential tests


# logarithmic tests
def test_logarithmic():
    assert solveset_real(log(x - 3) + log(x + 3), x) == FiniteSet(
        -sqrt(10), sqrt(10))
    assert solveset_real(log(x + 1) - log(2*x - 1), x) == FiniteSet(2)
    assert solveset_real(log(x + 3) + log(1 + 3/x) - 3, x) == FiniteSet(
        -3 + sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 + exp(3)/2,
        -sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 - 3 + exp(3)/2)

    eq = z - log(x) + log(y/(x*(-1 + y**2/x**2)))
    assert solveset_real(eq, x) == \
        Intersection(S.Reals, FiniteSet(-sqrt(y**2 - y*exp(z)),
            sqrt(y**2 - y*exp(z)))) - \
        Intersection(S.Reals, FiniteSet(-sqrt(y**2), sqrt(y**2)))
    assert solveset_real(
        log(3*x) - log(-x + 1) - log(4*x + 1), x) == FiniteSet(Rational(-1, 2), S.Half)
    assert solveset(log(x**y) - y*log(x), x, S.Reals) == S.Reals

@XFAIL
def test_uselogcombine_2():
    eq = log(exp(2*x) + 1) + log(-tanh(x) + 1) - log(2)
    assert solveset_real(eq, x) is S.EmptySet
    eq = log(8*x) - log(sqrt(x) + 1) - 2
    assert solveset_real(eq, x) is S.EmptySet


def test_is_logarithmic():
    assert _is_logarithmic(y, x) is False
    assert _is_logarithmic(log(x), x) is True
    assert _is_logarithmic(log(x) - 3, x) is True
    assert _is_logarithmic(log(x)*log(y), x) is True
    assert _is_logarithmic(log(x)**2, x) is False
    assert _is_logarithmic(log(x - 3) + log(x + 3), x) is True
    assert _is_logarithmic(log(x**y) - y*log(x), x) is True
    assert _is_logarithmic(sin(log(x)), x) is False
    assert _is_logarithmic(x + y, x) is False
    assert _is_logarithmic(log(3*x) - log(1 - x) + 4, x) is True
    assert _is_logarithmic(log(x) + log(y) + x, x) is False
    assert _is_logarithmic(log(log(x - 3)) + log(x - 3), x) is True
    assert _is_logarithmic(log(log(3) + x) + log(x), x) is True
    assert _is_logarithmic(log(x)*(y + 3) + log(x), y) is False


def test_solve_logarithm():
    y = Symbol('y')
    assert _solve_logarithm(log(x**y) - y*log(x), 0, x, S.Reals) == S.Reals
    y = Symbol('y', positive=True)
    assert _solve_logarithm(log(x)*log(y), 0, x, S.Reals) == FiniteSet(1)

# end of logarithmic tests


# lambert tests
def test_is_lambert():
    a, b, c = symbols('a,b,c')
    assert _is_lambert(x**2, x) is False
    assert _is_lambert(a**x**2+b*x+c, x) is True
    assert _is_lambert(E**2, x) is False
    assert _is_lambert(x*E**2, x) is False
    assert _is_lambert(3*log(x) - x*log(3), x) is True
    assert _is_lambert(log(log(x - 3)) + log(x-3), x) is True
    assert _is_lambert(5*x - 1 + 3*exp(2 - 7*x), x) is True
    assert _is_lambert((a/x + exp(x/2)).diff(x, 2), x) is True
    assert _is_lambert((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x) is True
    assert _is_lambert(x*sinh(x) - 1, x) is True
    assert _is_lambert(x*cos(x) - 5, x) is True
    assert _is_lambert(tanh(x) - 5*x, x) is True
    assert _is_lambert(cosh(x) - sinh(x), x) is False

# end of lambert tests


def test_linear_coeffs():
    from sympy.solvers.solveset import linear_coeffs
    assert linear_coeffs(0, x) == [0, 0]
    assert all(i is S.Zero for i in linear_coeffs(0, x))
    assert linear_coeffs(x + 2*y + 3, x, y) == [1, 2, 3]
    assert linear_coeffs(x + 2*y + 3, y, x) == [2, 1, 3]
    assert linear_coeffs(x + 2*x**2 + 3, x, x**2) == [1, 2, 3]
    raises(ValueError, lambda:
        linear_coeffs(x + 2*x**2 + x**3, x, x**2))
    raises(ValueError, lambda:
        linear_coeffs(1/x*(x - 1) + 1/x, x))
    raises(ValueError, lambda:
        linear_coeffs(x, x, x))
    assert linear_coeffs(a*(x + y), x, y) == [a, a, 0]
    assert linear_coeffs(1.0, x, y) == [0, 0, 1.0]
    # don't include coefficients of 0
    assert linear_coeffs(Eq(x, x + y), x, y, dict=True) == {y: -1}
    assert linear_coeffs(0, x, y, dict=True) == {}


def test_is_modular():
    assert _is_modular(y, x) is False
    assert _is_modular(Mod(x, 3) - 1, x) is True
    assert _is_modular(Mod(x**3 - 3*x**2 - x + 1, 3) - 1, x) is True
    assert _is_modular(Mod(exp(x + y), 3) - 2, x) is True
    assert _is_modular(Mod(exp(x + y), 3) - log(x), x) is True
    assert _is_modular(Mod(x, 3) - 1, y) is False
    assert _is_modular(Mod(x, 3)**2 - 5, x) is False
    assert _is_modular(Mod(x, 3)**2 - y, x) is False
    assert _is_modular(exp(Mod(x, 3)) - 1, x) is False
    assert _is_modular(Mod(3, y) - 1, y) is False


def test_invert_modular():
    n = Dummy('n', integer=True)
    from sympy.solvers.solveset import _invert_modular as invert_modular

    # no solutions
    assert invert_modular(Mod(x, 12), S(1)/2, n, x) == (x, S.EmptySet)
    # non invertible cases
    assert invert_modular(Mod(sin(x), 7), S(5), n, x) == (Mod(sin(x), 7), 5)
    assert invert_modular(Mod(exp(x), 7), S(5), n, x) == (Mod(exp(x), 7), 5)
    assert invert_modular(Mod(log(x), 7), S(5), n, x) == (Mod(log(x), 7), 5)
    # a is symbol
    assert dumeq(invert_modular(Mod(x, 7), S(5), n, x),
            (x, ImageSet(Lambda(n, 7*n + 5), S.Integers)))
    # a.is_Add
    assert dumeq(invert_modular(Mod(x + 8, 7), S(5), n, x),
            (x, ImageSet(Lambda(n, 7*n + 4), S.Integers)))
    assert invert_modular(Mod(x**2 + x, 7), S(5), n, x) == \
            (Mod(x**2 + x, 7), 5)
    # a.is_Mul
    assert dumeq(invert_modular(Mod(3*x, 7), S(5), n, x),
            (x, ImageSet(Lambda(n, 7*n + 4), S.Integers)))
    assert invert_modular(Mod((x + 1)*(x + 2), 7), S(5), n, x) == \
            (Mod((x + 1)*(x + 2), 7), 5)
    # a.is_Pow
    assert invert_modular(Mod(x**4, 7), S(5), n, x) == \
            (x, S.EmptySet)
    assert dumeq(invert_modular(Mod(3**x, 4), S(3), n, x),
            (x, ImageSet(Lambda(n, 2*n + 1), S.Naturals0)))
    assert dumeq(invert_modular(Mod(2**(x**2 + x + 1), 7), S(2), n, x),
            (x**2 + x + 1, ImageSet(Lambda(n, 3*n + 1), S.Naturals0)))
    assert invert_modular(Mod(sin(x)**4, 7), S(5), n, x) == (x, S.EmptySet)


def test_solve_modular():
    n = Dummy('n', integer=True)
    # if rhs has symbol (need to be implemented in future).
    assert solveset(Mod(x, 4) - x, x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(-x + Mod(x, 4), 0),
            S.Integers))
    # when _invert_modular fails to invert
    assert solveset(3 - Mod(sin(x), 7), x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(sin(x), 7) - 3, 0), S.Integers))
    assert solveset(3 - Mod(log(x), 7), x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(log(x), 7) - 3, 0), S.Integers))
    assert solveset(3 - Mod(exp(x), 7), x, S.Integers
        ).dummy_eq(ConditionSet(x, Eq(Mod(exp(x), 7) - 3, 0),
        S.Integers))
    # EmptySet solution definitely
    assert solveset(7 - Mod(x, 5), x, S.Integers) is S.EmptySet
    assert solveset(5 - Mod(x, 5), x, S.Integers) is S.EmptySet
    # Negative m
    assert dumeq(solveset(2 + Mod(x, -3), x, S.Integers),
            ImageSet(Lambda(n, -3*n - 2), S.Integers))
    assert solveset(4 + Mod(x, -3), x, S.Integers) is S.EmptySet
    # linear expression in Mod
    assert dumeq(solveset(3 - Mod(x, 5), x, S.Integers),
        ImageSet(Lambda(n, 5*n + 3), S.Integers))
    assert dumeq(solveset(3 - Mod(5*x - 8, 7), x, S.Integers),
                ImageSet(Lambda(n, 7*n + 5), S.Integers))
    assert dumeq(solveset(3 - Mod(5*x, 7), x, S.Integers),
                ImageSet(Lambda(n, 7*n + 2), S.Integers))
    # higher degree expression in Mod
    assert dumeq(solveset(Mod(x**2, 160) - 9, x, S.Integers),
            Union(ImageSet(Lambda(n, 160*n + 3), S.Integers),
            ImageSet(Lambda(n, 160*n + 13), S.Integers),
            ImageSet(Lambda(n, 160*n + 67), S.Integers),
            ImageSet(Lambda(n, 160*n + 77), S.Integers),
            ImageSet(Lambda(n, 160*n + 83), S.Integers),
            ImageSet(Lambda(n, 160*n + 93), S.Integers),
            ImageSet(Lambda(n, 160*n + 147), S.Integers),
            ImageSet(Lambda(n, 160*n + 157), S.Integers)))
    assert solveset(3 - Mod(x**4, 7), x, S.Integers) is S.EmptySet
    assert dumeq(solveset(Mod(x**4, 17) - 13, x, S.Integers),
            Union(ImageSet(Lambda(n, 17*n + 3), S.Integers),
            ImageSet(Lambda(n, 17*n + 5), S.Integers),
            ImageSet(Lambda(n, 17*n + 12), S.Integers),
            ImageSet(Lambda(n, 17*n + 14), S.Integers)))
    # a.is_Pow tests
    assert dumeq(solveset(Mod(7**x, 41) - 15, x, S.Integers),
            ImageSet(Lambda(n, 40*n + 3), S.Naturals0))
    assert dumeq(solveset(Mod(12**x, 21) - 18, x, S.Integers),
            ImageSet(Lambda(n, 6*n + 2), S.Naturals0))
    assert dumeq(solveset(Mod(3**x, 4) - 3, x, S.Integers),
            ImageSet(Lambda(n, 2*n + 1), S.Naturals0))
    assert dumeq(solveset(Mod(2**x, 7) - 2 , x, S.Integers),
            ImageSet(Lambda(n, 3*n + 1), S.Naturals0))
    assert dumeq(solveset(Mod(3**(3**x), 4) - 3, x, S.Integers),
            Intersection(ImageSet(Lambda(n, Intersection({log(2*n + 1)/log(3)},
            S.Integers)), S.Naturals0), S.Integers))
    # Implemented for m without primitive root
    assert solveset(Mod(x**3, 7) - 2, x, S.Integers) is S.EmptySet
    assert dumeq(solveset(Mod(x**3, 8) - 1, x, S.Integers),
            ImageSet(Lambda(n, 8*n + 1), S.Integers))
    assert dumeq(solveset(Mod(x**4, 9) - 4, x, S.Integers),
            Union(ImageSet(Lambda(n, 9*n + 4), S.Integers),
            ImageSet(Lambda(n, 9*n + 5), S.Integers)))
    # domain intersection
    assert dumeq(solveset(3 - Mod(5*x - 8, 7), x, S.Naturals0),
            Intersection(ImageSet(Lambda(n, 7*n + 5), S.Integers), S.Naturals0))
    # Complex args
    assert solveset(Mod(x, 3) - I, x, S.Integers) == \
            S.EmptySet
    assert solveset(Mod(I*x, 3) - 2, x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(I*x, 3) - 2, 0), S.Integers))
    assert solveset(Mod(I + x, 3) - 2, x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(x + I, 3) - 2, 0), S.Integers))

    # issue 17373 (https://github.com/sympy/sympy/issues/17373)
    assert dumeq(solveset(Mod(x**4, 14) - 11, x, S.Integers),
            Union(ImageSet(Lambda(n, 14*n + 3), S.Integers),
            ImageSet(Lambda(n, 14*n + 11), S.Integers)))
    assert dumeq(solveset(Mod(x**31, 74) - 43, x, S.Integers),
            ImageSet(Lambda(n, 74*n + 31), S.Integers))

    # issue 13178
    n = symbols('n', integer=True)
    a = 742938285
    b = 1898888478
    m = 2**31 - 1
    c = 20170816
    assert dumeq(solveset(c - Mod(a**n*b, m), n, S.Integers),
            ImageSet(Lambda(n, 2147483646*n + 100), S.Naturals0))
    assert dumeq(solveset(c - Mod(a**n*b, m), n, S.Naturals0),
            Intersection(ImageSet(Lambda(n, 2147483646*n + 100), S.Naturals0),
            S.Naturals0))
    assert dumeq(solveset(c - Mod(a**(2*n)*b, m), n, S.Integers),
            Intersection(ImageSet(Lambda(n, 1073741823*n + 50), S.Naturals0),
            S.Integers))
    assert solveset(c - Mod(a**(2*n + 7)*b, m), n, S.Integers) is S.EmptySet
    assert dumeq(solveset(c - Mod(a**(n - 4)*b, m), n, S.Integers),
            Intersection(ImageSet(Lambda(n, 2147483646*n + 104), S.Naturals0),
            S.Integers))

# end of modular tests

def test_issue_17276():
    assert nonlinsolve([Eq(x, 5**(S(1)/5)), Eq(x*y, 25*sqrt(5))], x, y) == \
     FiniteSet((5**(S(1)/5), 25*5**(S(3)/10)))


def test_issue_10426():
    x = Dummy('x')
    a = Symbol('a')
    n = Dummy('n')
    assert (solveset(sin(x + a) - sin(x), a)).dummy_eq(Dummy('x')) == (Union(
        ImageSet(Lambda(n, 2*n*pi), S.Integers),
        Intersection(S.Complexes, ImageSet(Lambda(n, -I*(I*(2*n*pi + arg(-exp(-2*I*x))) + 2*im(x))),
        S.Integers)))).dummy_eq(Dummy('x,n'))


def test_solveset_conjugate():
    """Test solveset for simple conjugate functions"""
    assert solveset(conjugate(x) -3 + I) == FiniteSet(3 + I)


def test_issue_18208():
    variables = symbols('x0:16') + symbols('y0:12')
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,\
    y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = variables

    eqs = [x0 + x1 + x2 + x3 - 51,
           x0 + x1 + x4 + x5 - 46,
           x2 + x3 + x6 + x7 - 39,
           x0 + x3 + x4 + x7 - 50,
           x1 + x2 + x5 + x6 - 35,
           x4 + x5 + x6 + x7 - 34,
           x4 + x5 + x8 + x9 - 46,
           x10 + x11 + x6 + x7 - 23,
           x11 + x4 + x7 + x8 - 25,
           x10 + x5 + x6 + x9 - 44,
           x10 + x11 + x8 + x9 - 35,
           x12 + x13 + x8 + x9 - 35,
           x10 + x11 + x14 + x15 - 29,
           x11 + x12 + x15 + x8 - 35,
           x10 + x13 + x14 + x9 - 29,
           x12 + x13 + x14 + x15 - 29,
           y0 + y1 + y2 + y3 - 55,
           y0 + y1 + y4 + y5 - 53,
           y2 + y3 + y6 + y7 - 56,
           y0 + y3 + y4 + y7 - 57,
           y1 + y2 + y5 + y6 - 52,
           y4 + y5 + y6 + y7 - 54,
           y4 + y5 + y8 + y9 - 48,
           y10 + y11 + y6 + y7 - 60,
           y11 + y4 + y7 + y8 - 51,
           y10 + y5 + y6 + y9 - 57,
           y10 + y11 + y8 + y9 - 54,
           x10 - 2,
           x11 - 5,
           x12 - 1,
           x13 - 6,
           x14 - 1,
           x15 - 21,
           y0 - 12,
           y1 - 20]

    expected = [38 - x3, x3 - 10, 23 - x3, x3, 12 - x7, x7 + 6, 16 - x7, x7,
                8, 20, 2, 5, 1, 6, 1, 21, 12, 20, -y11 + y9 + 2, y11 - y9 + 21,
                -y11 - y7 + y9 + 24, y11 + y7 - y9 - 3, 33 - y7, y7, 27 - y9, y9,
                27 - y11, y11]

    A, b = linear_eq_to_matrix(eqs, variables)

    # solve
    solve_expected = {v:eq for v, eq in zip(variables, expected) if v != eq}

    assert solve(eqs, variables) == solve_expected

    # linsolve
    linsolve_expected = FiniteSet(Tuple(*expected))

    assert linsolve(eqs, variables) == linsolve_expected
    assert linsolve((A, b), variables) == linsolve_expected

    # gauss_jordan_solve
    gj_solve, new_vars = A.gauss_jordan_solve(b)
    gj_solve = list(gj_solve)

    gj_expected = linsolve_expected.subs(zip([x3, x7, y7, y9, y11], new_vars))

    assert FiniteSet(Tuple(*gj_solve)) == gj_expected

    # nonlinsolve
    # The solution set of nonlinsolve is currently equivalent to linsolve and is
    # also correct. However, we would prefer to use the same symbols as parameters
    # for the solution to the underdetermined system in all cases if possible.
    # We want a solution that is not just equivalent but also given in the same form.
    # This test may be changed should nonlinsolve be modified in this way.

    nonlinsolve_expected = FiniteSet((38 - x3, x3 - 10, 23 - x3, x3, 12 - x7, x7 + 6,
                                      16 - x7, x7, 8, 20, 2, 5, 1, 6, 1, 21, 12, 20,
                                      -y5 + y7 - 1, y5 - y7 + 24, 21 - y5, y5, 33 - y7,
                                      y7, 27 - y9, y9, -y5 + y7 - y9 + 24, y5 - y7 + y9 + 3))

    assert nonlinsolve(eqs, variables) == nonlinsolve_expected


def test_substitution_with_infeasible_solution():
    a00, a01, a10, a11, l0, l1, l2, l3, m0, m1, m2, m3, m4, m5, m6, m7, c00, c01, c10, c11, p00, p01, p10, p11 = symbols(
        'a00, a01, a10, a11, l0, l1, l2, l3, m0, m1, m2, m3, m4, m5, m6, m7, c00, c01, c10, c11, p00, p01, p10, p11'
    )
    solvefor = [p00, p01, p10, p11, c00, c01, c10, c11, m0, m1, m3, l0, l1, l2, l3]
    system = [
        -l0 * c00 - l1 * c01 + m0 + c00 + c01,
        -l0 * c10 - l1 * c11 + m1,
        -l2 * c00 - l3 * c01 + c00 + c01,
        -l2 * c10 - l3 * c11 + m3,
        -l0 * p00 - l2 * p10 + p00 + p10,
        -l1 * p00 - l3 * p10 + p00 + p10,
        -l0 * p01 - l2 * p11,
        -l1 * p01 - l3 * p11,
        -a00 + c00 * p00 + c10 * p01,
        -a01 + c01 * p00 + c11 * p01,
        -a10 + c00 * p10 + c10 * p11,
        -a11 + c01 * p10 + c11 * p11,
        -m0 * p00,
        -m1 * p01,
        -m2 * p10,
        -m3 * p11,
        -m4 * c00,
        -m5 * c01,
        -m6 * c10,
        -m7 * c11,
        m2,
        m4,
        m5,
        m6,
        m7
    ]
    sol = FiniteSet(
        (0, Complement(FiniteSet(p01), FiniteSet(0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, l2, l3),
        (p00, Complement(FiniteSet(p01), FiniteSet(0)), 0, p11, 0, 0, 0, 0, 0, 0, 0, 1, 1, -p01/p11, -p01/p11),
        (0, Complement(FiniteSet(p01), FiniteSet(0)), 0, p11, 0, 0, 0, 0, 0, 0, 0, 1, -l3*p11/p01, -p01/p11, l3),
        (0, Complement(FiniteSet(p01), FiniteSet(0)), 0, p11, 0, 0, 0, 0, 0, 0, 0, -l2*p11/p01, -l3*p11/p01, l2, l3),
    )
    assert sol != nonlinsolve(system, solvefor)


def test_issue_20097():
    assert solveset(1/sqrt(x)) is S.EmptySet


def test_issue_15350():
    assert solveset(diff(sqrt(1/x+x))) == FiniteSet(-1, 1)


def test_issue_18359():
    c1 = Piecewise((0, x < 0), (Min(1, x)/2 - Min(2, x)/2 + Min(3, x)/2, True))
    c2 = Piecewise((Piecewise((0, x < 0), (Min(1, x)/2 - Min(2, x)/2 + Min(3, x)/2, True)), x >= 0), (0, True))
    correct_result = Interval(1, 2)
    result1 = solveset(c1 - Rational(1, 2), x, Interval(0, 3))
    result2 = solveset(c2 - Rational(1, 2), x, Interval(0, 3))
    assert result1 == correct_result
    assert result2 == correct_result


def test_issue_17604():
    lhs = -2**(3*x/11)*exp(x/11) + pi**(x/11)
    assert _is_exponential(lhs, x)
    assert _solve_exponential(lhs, 0, x, S.Complexes) == FiniteSet(0)


def test_issue_17580():
    assert solveset(1/(1 - x**3)**2, x, S.Reals) is S.EmptySet


def test_issue_17566_actual():
    sys = [2**x + 2**y - 3, 4**x + 9**y - 5]
    # Not clear this is the correct result, but at least no recursion error
    assert nonlinsolve(sys, x, y) == FiniteSet((log(3 - 2**y)/log(2), y))


def test_issue_17565():
    eq = Ge(2*(x - 2)**2/(3*(x + 1)**(Integer(1)/3)) + 2*(x - 2)*(x + 1)**(Integer(2)/3), 0)
    res = Union(Interval.Lopen(-1, -Rational(1, 4)), Interval(2, oo))
    assert solveset(eq, x, S.Reals) == res


def test_issue_15024():
    function = (x + 5)/sqrt(-x**2 - 10*x)
    assert solveset(function, x, S.Reals) == FiniteSet(Integer(-5))


def test_issue_16877():
    assert dumeq(nonlinsolve([x - 1, sin(y)], x, y),
                 FiniteSet((1, ImageSet(Lambda(n, 2*n*pi), S.Integers)),
                           (1, ImageSet(Lambda(n, 2*n*pi + pi), S.Integers))))
    # Even better if (1, ImageSet(Lambda(n, n*pi), S.Integers)) is obtained


def test_issue_16876():
    assert dumeq(nonlinsolve([sin(x), 2*x - 4*y], x, y),
                 FiniteSet((ImageSet(Lambda(n, 2*n*pi), S.Integers),
                            ImageSet(Lambda(n, n*pi), S.Integers)),
                           (ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
                            ImageSet(Lambda(n, n*pi + pi/2), S.Integers))))
    # Even better if (ImageSet(Lambda(n, n*pi), S.Integers),
    #                 ImageSet(Lambda(n, n*pi/2), S.Integers)) is obtained

def test_issue_21236():
    x, z = symbols("x z")
    y = symbols('y', rational=True)
    assert solveset(x**y - z, x, S.Reals) == ConditionSet(x, Eq(x**y - z, 0), S.Reals)
    e1, e2 = symbols('e1 e2', even=True)
    y = e1/e2  # don't know if num or den will be odd and the other even
    assert solveset(x**y - z, x, S.Reals) == ConditionSet(x, Eq(x**y - z, 0), S.Reals)


def test_issue_21908():
    assert nonlinsolve([(x**2 + 2*x - y**2)*exp(x), -2*y*exp(x)], x, y
                      ) == {(-2, 0), (0, 0)}


def test_issue_19144():
    # test case 1
    expr1 = [x + y - 1, y**2 + 1]
    eq1 = [Eq(i, 0) for i in expr1]
    soln1 = {(1 - I, I), (1 + I, -I)}
    soln_expr1 = nonlinsolve(expr1, [x, y])
    soln_eq1 = nonlinsolve(eq1, [x, y])
    assert soln_eq1 == soln_expr1 == soln1
    # test case 2 - with denoms
    expr2 = [x/y - 1, y**2 + 1]
    eq2 = [Eq(i, 0) for i in expr2]
    soln2 = {(-I, -I), (I, I)}
    soln_expr2 = nonlinsolve(expr2, [x, y])
    soln_eq2 = nonlinsolve(eq2, [x, y])
    assert soln_eq2 == soln_expr2 == soln2
    # denominators that cancel in expression
    assert nonlinsolve([Eq(x + 1/x, 1/x)], [x]) == FiniteSet((S.EmptySet,))


def test_issue_22413():
    res =  nonlinsolve((4*y*(2*x + 2*exp(y) + 1)*exp(2*x),
                         4*x*exp(2*x) + 4*y*exp(2*x + y) + 4*exp(2*x + y) + 1),
                        x, y)
    # First solution is not correct, but the issue was an exception
    sols = FiniteSet((x, S.Zero), (-exp(y) - S.Half, y))
    assert res == sols


def test_issue_23318():
    eqs_eq = [
        Eq(53.5780461486929, x * log(y / (5.0 - y) + 1) / y),
        Eq(x, 0.0015 * z),
        Eq(0.0015, 7845.32 * y / z),
    ]
    eqs_expr = [eq.lhs - eq.rhs for eq in eqs_eq]

    sol = {(266.97755814852, 0.0340301680681629, 177985.03876568)}

    assert_close_nl(nonlinsolve(eqs_eq, [x, y, z]), sol)
    assert_close_nl(nonlinsolve(eqs_expr, [x, y, z]), sol)

    logterm = log(1.91196789933362e-7*z/(5.0 - 1.91196789933362e-7*z) + 1)
    eq = -0.0015*z*logterm + 1.02439504345316e-5*z
    assert_close_ss(solveset(eq, z), {0, 177985.038765679})


def test_issue_19814():
    assert nonlinsolve([ 2**m - 2**(2*n), 4*2**m - 2**(4*n)], m, n
                      ) == FiniteSet((log(2**(2*n))/log(2), S.Complexes))


def test_issue_22058():
    sol = solveset(-sqrt(t)*x**2 + 2*x + sqrt(t), x, S.Reals)
    # doesn't fail (and following numerical check)
    assert sol.xreplace({t: 1}) == {1 - sqrt(2), 1 + sqrt(2)}, sol.xreplace({t: 1})


def test_issue_11184():
    assert solveset(20*sqrt(y**2 + (sqrt(-(y - 10)*(y + 10)) + 10)**2) - 60, y, S.Reals) is S.EmptySet


def test_issue_21890():
    e = S(2)/3
    assert nonlinsolve([4*x**3*y**4 - 2*y, 4*x**4*y**3 - 2*x], x, y) == {
        (2**e/(2*y), y), ((-2**e/4 - 2**e*sqrt(3)*I/4)/y, y),
        ((-2**e/4 + 2**e*sqrt(3)*I/4)/y, y)}
    assert nonlinsolve([(1 - 4*x**2)*exp(-2*x**2 - 2*y**2),
        -4*x*y*exp(-2*x**2)*exp(-2*y**2)], x, y) == {(-S(1)/2, 0), (S(1)/2, 0)}
    rx, ry = symbols('x y', real=True)
    sol = nonlinsolve([4*rx**3*ry**4 - 2*ry, 4*rx**4*ry**3 - 2*rx], rx, ry)
    ans = {(2**(S(2)/3)/(2*ry), ry),
        ((-2**(S(2)/3)/4 - 2**(S(2)/3)*sqrt(3)*I/4)/ry, ry),
        ((-2**(S(2)/3)/4 + 2**(S(2)/3)*sqrt(3)*I/4)/ry, ry)}
    assert sol == ans


def test_issue_22628():
    assert nonlinsolve([h - 1, k - 1, f - 2, f - 4, -2*k], h, k, f) == S.EmptySet
    assert nonlinsolve([x**3 - 1, x + y, x**2 - 4], [x, y]) == S.EmptySet


def test_issue_25781():
    assert solve(sqrt(x/2) - x) == [0, S.Half]


def test_issue_26077():
    _n = Symbol('_n')
    function = x*cot(5*x)
    critical_points = stationary_points(function, x, S.Reals)
    excluded_points = Union(
        ImageSet(Lambda(_n, 2*_n*pi/5), S.Integers),
        ImageSet(Lambda(_n, 2*_n*pi/5 + pi/5), S.Integers)
    )
    solution = ConditionSet(x,
        Eq(x*(-5*cot(5*x)**2 - 5) + cot(5*x), 0),
        Complement(S.Reals, excluded_points)
    )
    assert solution.as_dummy() == critical_points.as_dummy()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_shape.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_base import Fusion
from fusion_utils import FusionUtils
from numpy import ndarray
from onnx import NodeProto, TensorProto
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionShape(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "Shape", "Concat")
        self.utils = FusionUtils(model)
        self.shape_infer = None
        self.shape_infer_done = False

    def get_dimensions_from_tensor_proto(self, tensor_proto: TensorProto) -> int | None:
        if tensor_proto.type.tensor_type.HasField("shape"):
            return len(tensor_proto.type.tensor_type.shape.dim)
        else:
            return None

    def get_dimensions(self, input_name: str) -> int | None:
        shape = self.model.get_shape(input_name)
        if shape is not None:
            return len(shape)

        if not self.shape_infer_done:
            self.shape_infer = self.model.infer_runtime_shape(update=True)
            self.shape_infer_done = True

        if self.shape_infer is not None:
            return self.get_dimensions_from_tensor_proto(self.shape_infer.known_vi_[input_name])

        return None

    def fuse(
        self,
        concat_node: NodeProto,
        input_name_to_nodes: dict[str, list[NodeProto]],
        output_name_to_node: dict[str, NodeProto],
    ):
        #
        # Simplify subgraph like
        #
        #          (2d_input)
        #           /       \
        #       Shape       shape
        #       /             \
        #   Gather(indices=0)  Gather(indices=1)
        #       |                |
        #   Unsqueeze(axes=0)   Unsqueeze(axes=0)
        #          \           /
        #             Concat
        #               |
        #
        # into  (2d_input) --> Shape -->
        #
        opset_version = self.model.get_opset_version()

        inputs = len(concat_node.input)
        root = None
        shape_output = None
        for i in range(inputs):
            path = self.model.match_parent_path(
                concat_node,
                ["Unsqueeze", "Gather", "Shape"],
                [i, 0, 0],
                output_name_to_node,
            )
            if path is None:
                return

            unsqueeze, gather, shape = path
            if i == 0:
                shape_output = shape.output[0]
            if root is None:
                root = shape.input[0]
                if self.get_dimensions(root) != inputs:
                    return
            elif shape.input[0] != root:
                return

            if not FusionUtils.check_node_attribute(unsqueeze, "axis", 0, default_value=0):
                return

            if opset_version < 13:
                if not FusionUtils.check_node_attribute(unsqueeze, "axes", [0]):
                    return
            else:
                if not self.utils.check_node_input_value(unsqueeze, 1, [0]):
                    return

            value = self.model.get_constant_value(gather.input[1])

            if not (isinstance(value, ndarray) and value.size == 1 and value.item() == i):
                return

        if self.model.find_graph_output(concat_node.output[0]) is None:
            self.model.replace_input_of_all_nodes(concat_node.output[0], shape_output)
            self.increase_counter("Reshape")
            self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\_elffile.py ===
"""
ELF file parser.

This provides a class ``ELFFile`` that parses an ELF executable in a similar
interface to ``ZipFile``. Only the read interface is implemented.

Based on: https://gist.github.com/lyssdod/f51579ae8d93c8657a5564aefc2ffbca
ELF header: https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.eheader.html
"""

from __future__ import annotations

import enum
import os
import struct
from typing import IO


class ELFInvalid(ValueError):
    pass


class EIClass(enum.IntEnum):
    C32 = 1
    C64 = 2


class EIData(enum.IntEnum):
    Lsb = 1
    Msb = 2


class EMachine(enum.IntEnum):
    I386 = 3
    S390 = 22
    Arm = 40
    X8664 = 62
    AArc64 = 183


class ELFFile:
    """
    Representation of an ELF executable.
    """

    def __init__(self, f: IO[bytes]) -> None:
        self._f = f

        try:
            ident = self._read("16B")
        except struct.error as e:
            raise ELFInvalid("unable to parse identification") from e
        magic = bytes(ident[:4])
        if magic != b"\x7fELF":
            raise ELFInvalid(f"invalid magic: {magic!r}")

        self.capacity = ident[4]  # Format for program header (bitness).
        self.encoding = ident[5]  # Data structure encoding (endianness).

        try:
            # e_fmt: Format for program header.
            # p_fmt: Format for section header.
            # p_idx: Indexes to find p_type, p_offset, and p_filesz.
            e_fmt, self._p_fmt, self._p_idx = {
                (1, 1): ("<HHIIIIIHHH", "<IIIIIIII", (0, 1, 4)),  # 32-bit LSB.
                (1, 2): (">HHIIIIIHHH", ">IIIIIIII", (0, 1, 4)),  # 32-bit MSB.
                (2, 1): ("<HHIQQQIHHH", "<IIQQQQQQ", (0, 2, 5)),  # 64-bit LSB.
                (2, 2): (">HHIQQQIHHH", ">IIQQQQQQ", (0, 2, 5)),  # 64-bit MSB.
            }[(self.capacity, self.encoding)]
        except KeyError as e:
            raise ELFInvalid(
                f"unrecognized capacity ({self.capacity}) or encoding ({self.encoding})"
            ) from e

        try:
            (
                _,
                self.machine,  # Architecture type.
                _,
                _,
                self._e_phoff,  # Offset of program header.
                _,
                self.flags,  # Processor-specific flags.
                _,
                self._e_phentsize,  # Size of section.
                self._e_phnum,  # Number of sections.
            ) = self._read(e_fmt)
        except struct.error as e:
            raise ELFInvalid("unable to parse machine and section information") from e

    def _read(self, fmt: str) -> tuple[int, ...]:
        return struct.unpack(fmt, self._f.read(struct.calcsize(fmt)))

    @property
    def interpreter(self) -> str | None:
        """
        The path recorded in the ``PT_INTERP`` section header.
        """
        for index in range(self._e_phnum):
            self._f.seek(self._e_phoff + self._e_phentsize * index)
            try:
                data = self._read(self._p_fmt)
            except struct.error:
                continue
            if data[self._p_idx[0]] != 3:  # Not PT_INTERP.
                continue
            self._f.seek(data[self._p_idx[1]])
            return os.fsdecode(self._f.read(data[self._p_idx[2]])).strip("\0")
        return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\_jaraco_text.py ===
"""Functions brought over from jaraco.text.

These functions are not supposed to be used within `pip._internal`. These are
helper functions brought over from `jaraco.text` to enable vendoring newer
copies of `pkg_resources` without having to vendor `jaraco.text` and its entire
dependency cone; something that our vendoring setup is not currently capable of
handling.

License reproduced from original source below:

Copyright Jason R. Coombs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import functools
import itertools


def _nonblank(str):
    return str and not str.startswith("#")


@functools.singledispatch
def yield_lines(iterable):
    r"""
    Yield valid lines of a string or iterable.

    >>> list(yield_lines(''))
    []
    >>> list(yield_lines(['foo', 'bar']))
    ['foo', 'bar']
    >>> list(yield_lines('foo\nbar'))
    ['foo', 'bar']
    >>> list(yield_lines('\nfoo\n#bar\nbaz #comment'))
    ['foo', 'baz #comment']
    >>> list(yield_lines(['foo\nbar', 'baz', 'bing\n\n\n']))
    ['foo', 'bar', 'baz', 'bing']
    """
    return itertools.chain.from_iterable(map(yield_lines, iterable))


@yield_lines.register(str)
def _(text):
    return filter(_nonblank, map(str.strip, text.splitlines()))


def drop_comment(line):
    """
    Drop comments.

    >>> drop_comment('foo # bar')
    'foo'

    A hash without a space may be in a URL.

    >>> drop_comment('http://example.com/foo#bar')
    'http://example.com/foo#bar'
    """
    return line.partition(" #")[0]


def join_continuation(lines):
    r"""
    Join lines continued by a trailing backslash.

    >>> list(join_continuation(['foo \\', 'bar', 'baz']))
    ['foobar', 'baz']
    >>> list(join_continuation(['foo \\', 'bar', 'baz']))
    ['foobar', 'baz']
    >>> list(join_continuation(['foo \\', 'bar \\', 'baz']))
    ['foobarbaz']

    Not sure why, but...
    The character preceding the backslash is also elided.

    >>> list(join_continuation(['goo\\', 'dly']))
    ['godly']

    A terrible idea, but...
    If no line is available to continue, suppress the lines.

    >>> list(join_continuation(['foo', 'bar\\', 'baz\\']))
    ['foo']
    """
    lines = iter(lines)
    for item in lines:
        while item.endswith("\\"):
            try:
                item = item[:-2].strip() + next(lines)
            except StopIteration:
                return
        yield item

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\_elffile.py ===
"""
ELF file parser.

This provides a class ``ELFFile`` that parses an ELF executable in a similar
interface to ``ZipFile``. Only the read interface is implemented.

Based on: https://gist.github.com/lyssdod/f51579ae8d93c8657a5564aefc2ffbca
ELF header: https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.eheader.html
"""

from __future__ import annotations

import enum
import os
import struct
from typing import IO


class ELFInvalid(ValueError):
    pass


class EIClass(enum.IntEnum):
    C32 = 1
    C64 = 2


class EIData(enum.IntEnum):
    Lsb = 1
    Msb = 2


class EMachine(enum.IntEnum):
    I386 = 3
    S390 = 22
    Arm = 40
    X8664 = 62
    AArc64 = 183


class ELFFile:
    """
    Representation of an ELF executable.
    """

    def __init__(self, f: IO[bytes]) -> None:
        self._f = f

        try:
            ident = self._read("16B")
        except struct.error as e:
            raise ELFInvalid("unable to parse identification") from e
        magic = bytes(ident[:4])
        if magic != b"\x7fELF":
            raise ELFInvalid(f"invalid magic: {magic!r}")

        self.capacity = ident[4]  # Format for program header (bitness).
        self.encoding = ident[5]  # Data structure encoding (endianness).

        try:
            # e_fmt: Format for program header.
            # p_fmt: Format for section header.
            # p_idx: Indexes to find p_type, p_offset, and p_filesz.
            e_fmt, self._p_fmt, self._p_idx = {
                (1, 1): ("<HHIIIIIHHH", "<IIIIIIII", (0, 1, 4)),  # 32-bit LSB.
                (1, 2): (">HHIIIIIHHH", ">IIIIIIII", (0, 1, 4)),  # 32-bit MSB.
                (2, 1): ("<HHIQQQIHHH", "<IIQQQQQQ", (0, 2, 5)),  # 64-bit LSB.
                (2, 2): (">HHIQQQIHHH", ">IIQQQQQQ", (0, 2, 5)),  # 64-bit MSB.
            }[(self.capacity, self.encoding)]
        except KeyError as e:
            raise ELFInvalid(
                f"unrecognized capacity ({self.capacity}) or encoding ({self.encoding})"
            ) from e

        try:
            (
                _,
                self.machine,  # Architecture type.
                _,
                _,
                self._e_phoff,  # Offset of program header.
                _,
                self.flags,  # Processor-specific flags.
                _,
                self._e_phentsize,  # Size of section.
                self._e_phnum,  # Number of sections.
            ) = self._read(e_fmt)
        except struct.error as e:
            raise ELFInvalid("unable to parse machine and section information") from e

    def _read(self, fmt: str) -> tuple[int, ...]:
        return struct.unpack(fmt, self._f.read(struct.calcsize(fmt)))

    @property
    def interpreter(self) -> str | None:
        """
        The path recorded in the ``PT_INTERP`` section header.
        """
        for index in range(self._e_phnum):
            self._f.seek(self._e_phoff + self._e_phentsize * index)
            try:
                data = self._read(self._p_fmt)
            except struct.error:
                continue
            if data[self._p_idx[0]] != 3:  # Not PT_INTERP.
                continue
            self._f.seek(data[self._p_idx[1]])
            return os.fsdecode(self._f.read(data[self._p_idx[2]])).strip("\0")
        return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_nditer.py ===
import subprocess
import sys
import textwrap

import numpy._core._multiarray_tests as _multiarray_tests
import pytest

import numpy as np
import numpy._core.umath as ncu
from numpy import all, arange, array, nditer
from numpy.testing import (
    HAS_REFCOUNT,
    IS_WASM,
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
    suppress_warnings,
)
from numpy.testing._private.utils import requires_memory


def iter_multi_index(i):
    ret = []
    while not i.finished:
        ret.append(i.multi_index)
        i.iternext()
    return ret

def iter_indices(i):
    ret = []
    while not i.finished:
        ret.append(i.index)
        i.iternext()
    return ret

def iter_iterindices(i):
    ret = []
    while not i.finished:
        ret.append(i.iterindex)
        i.iternext()
    return ret

@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_iter_refcount():
    # Make sure the iterator doesn't leak

    # Basic
    a = arange(6)
    dt = np.dtype('f4').newbyteorder()
    rc_a = sys.getrefcount(a)
    rc_dt = sys.getrefcount(dt)
    with nditer(a, [],
                [['readwrite', 'updateifcopy']],
                casting='unsafe',
                op_dtypes=[dt]) as it:
        assert_(not it.iterationneedsapi)
        assert_(sys.getrefcount(a) > rc_a)
        assert_(sys.getrefcount(dt) > rc_dt)
    # del 'it'
    it = None
    assert_equal(sys.getrefcount(a), rc_a)
    assert_equal(sys.getrefcount(dt), rc_dt)

    # With a copy
    a = arange(6, dtype='f4')
    dt = np.dtype('f4')
    rc_a = sys.getrefcount(a)
    rc_dt = sys.getrefcount(dt)
    it = nditer(a, [],
                [['readwrite']],
                op_dtypes=[dt])
    rc2_a = sys.getrefcount(a)
    rc2_dt = sys.getrefcount(dt)
    it2 = it.copy()
    assert_(sys.getrefcount(a) > rc2_a)
    if sys.version_info < (3, 13):
        # np.dtype('f4') is immortal after Python 3.13
        assert_(sys.getrefcount(dt) > rc2_dt)
    it = None
    assert_equal(sys.getrefcount(a), rc2_a)
    assert_equal(sys.getrefcount(dt), rc2_dt)
    it2 = None
    assert_equal(sys.getrefcount(a), rc_a)
    assert_equal(sys.getrefcount(dt), rc_dt)

def test_iter_best_order():
    # The iterator should always find the iteration order
    # with increasing memory addresses

    # Test the ordering for 1-D to 5-D shapes
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        a = arange(np.prod(shape))
        # Test each combination of positive and negative strides
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if ((2**bit) & dirs):
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = nditer(aview, [], [['readonly']])
            assert_equal(list(i), a)
            # Fortran-order
            i = nditer(aview.T, [], [['readonly']])
            assert_equal(list(i), a)
            # Other order
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), [], [['readonly']])
                assert_equal(list(i), a)

def test_iter_c_order():
    # Test forcing C order

    # Test the ordering for 1-D to 5-D shapes
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        a = arange(np.prod(shape))
        # Test each combination of positive and negative strides
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if ((2**bit) & dirs):
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = nditer(aview, order='C')
            assert_equal(list(i), aview.ravel(order='C'))
            # Fortran-order
            i = nditer(aview.T, order='C')
            assert_equal(list(i), aview.T.ravel(order='C'))
            # Other order
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), order='C')
                assert_equal(list(i),
                                    aview.swapaxes(0, 1).ravel(order='C'))

def test_iter_f_order():
    # Test forcing F order

    # Test the ordering for 1-D to 5-D shapes
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        a = arange(np.prod(shape))
        # Test each combination of positive and negative strides
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if ((2**bit) & dirs):
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = nditer(aview, order='F')
            assert_equal(list(i), aview.ravel(order='F'))
            # Fortran-order
            i = nditer(aview.T, order='F')
            assert_equal(list(i), aview.T.ravel(order='F'))
            # Other order
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), order='F')
                assert_equal(list(i),
                                    aview.swapaxes(0, 1).ravel(order='F'))

def test_iter_c_or_f_order():
    # Test forcing any contiguous (C or F) order

    # Test the ordering for 1-D to 5-D shapes
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        a = arange(np.prod(shape))
        # Test each combination of positive and negative strides
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if ((2**bit) & dirs):
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = nditer(aview, order='A')
            assert_equal(list(i), aview.ravel(order='A'))
            # Fortran-order
            i = nditer(aview.T, order='A')
            assert_equal(list(i), aview.T.ravel(order='A'))
            # Other order
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1), order='A')
                assert_equal(list(i),
                                    aview.swapaxes(0, 1).ravel(order='A'))

def test_nditer_multi_index_set():
    # Test the multi_index set
    a = np.arange(6).reshape(2, 3)
    it = np.nditer(a, flags=['multi_index'])

    # Removes the iteration on two first elements of a[0]
    it.multi_index = (0, 2,)

    assert_equal(list(it), [2, 3, 4, 5])

@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_nditer_multi_index_set_refcount():
    # Test if the reference count on index variable is decreased

    index = 0
    i = np.nditer(np.array([111, 222, 333, 444]), flags=['multi_index'])

    start_count = sys.getrefcount(index)
    i.multi_index = (index,)
    end_count = sys.getrefcount(index)

    assert_equal(start_count, end_count)

def test_iter_best_order_multi_index_1d():
    # The multi-indices should be correct with any reordering

    a = arange(4)
    # 1D order
    i = nditer(a, ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0,), (1,), (2,), (3,)])
    # 1D reversed order
    i = nditer(a[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(3,), (2,), (1,), (0,)])

def test_iter_best_order_multi_index_2d():
    # The multi-indices should be correct with any reordering

    a = arange(6)
    # 2D C-order
    i = nditer(a.reshape(2, 3), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
    # 2D Fortran-order
    i = nditer(a.reshape(2, 3).copy(order='F'), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)])
    # 2D reversed C-order
    i = nditer(a.reshape(2, 3)[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2)])
    i = nditer(a.reshape(2, 3)[:, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 2), (0, 1), (0, 0), (1, 2), (1, 1), (1, 0)])
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)])
    # 2D reversed Fortran-order
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)])
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1],
                                                   ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(0, 2), (1, 2), (0, 1), (1, 1), (0, 0), (1, 0)])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1],
                                                   ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i), [(1, 2), (0, 2), (1, 1), (0, 1), (1, 0), (0, 0)])

def test_iter_best_order_multi_index_3d():
    # The multi-indices should be correct with any reordering

    a = arange(12)
    # 3D C-order
    i = nditer(a.reshape(2, 3, 2), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i),
                            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1),
                             (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)])
    # 3D Fortran-order
    i = nditer(a.reshape(2, 3, 2).copy(order='F'), ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i),
                            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0),
                             (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1)])
    # 3D reversed C-order
    i = nditer(a.reshape(2, 3, 2)[::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i),
                            [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1),
                             (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1)])
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i),
                            [(0, 2, 0), (0, 2, 1), (0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1),
                             (1, 2, 0), (1, 2, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)])
    i = nditer(a.reshape(2, 3, 2)[:, :, ::-1], ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i),
                            [(0, 0, 1), (0, 0, 0), (0, 1, 1), (0, 1, 0), (0, 2, 1), (0, 2, 0),
                             (1, 0, 1), (1, 0, 0), (1, 1, 1), (1, 1, 0), (1, 2, 1), (1, 2, 0)])
    # 3D reversed Fortran-order
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1],
                                                    ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i),
                            [(1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0),
                             (1, 0, 1), (0, 0, 1), (1, 1, 1), (0, 1, 1), (1, 2, 1), (0, 2, 1)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1],
                                                    ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i),
                            [(0, 2, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0), (1, 0, 0),
                             (0, 2, 1), (1, 2, 1), (0, 1, 1), (1, 1, 1), (0, 0, 1), (1, 0, 1)])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, :, ::-1],
                                                    ['multi_index'], [['readonly']])
    assert_equal(iter_multi_index(i),
                            [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1),
                             (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0)])

def test_iter_best_order_c_index_1d():
    # The C index should be correct with any reordering

    a = arange(4)
    # 1D order
    i = nditer(a, ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3])
    # 1D reversed order
    i = nditer(a[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 2, 1, 0])

def test_iter_best_order_c_index_2d():
    # The C index should be correct with any reordering

    a = arange(6)
    # 2D C-order
    i = nditer(a.reshape(2, 3), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5])
    # 2D Fortran-order
    i = nditer(a.reshape(2, 3).copy(order='F'),
                                    ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 3, 1, 4, 2, 5])
    # 2D reversed C-order
    i = nditer(a.reshape(2, 3)[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 4, 5, 0, 1, 2])
    i = nditer(a.reshape(2, 3)[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [2, 1, 0, 5, 4, 3])
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 4, 3, 2, 1, 0])
    # 2D reversed Fortran-order
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1],
                                    ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 0, 4, 1, 5, 2])
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1],
                                    ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [2, 5, 1, 4, 0, 3])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1],
                                    ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 2, 4, 1, 3, 0])

def test_iter_best_order_c_index_3d():
    # The C index should be correct with any reordering

    a = arange(12)
    # 3D C-order
    i = nditer(a.reshape(2, 3, 2), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # 3D Fortran-order
    i = nditer(a.reshape(2, 3, 2).copy(order='F'),
                                    ['c_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])
    # 3D reversed C-order
    i = nditer(a.reshape(2, 3, 2)[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7])
    i = nditer(a.reshape(2, 3, 2)[:, :, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])
    # 3D reversed Fortran-order
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1],
                                    ['c_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [6, 0, 8, 2, 10, 4, 7, 1, 9, 3, 11, 5])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1],
                                    ['c_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [4, 10, 2, 8, 0, 6, 5, 11, 3, 9, 1, 7])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, :, ::-1],
                                    ['c_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [1, 7, 3, 9, 5, 11, 0, 6, 2, 8, 4, 10])

def test_iter_best_order_f_index_1d():
    # The Fortran index should be correct with any reordering

    a = arange(4)
    # 1D order
    i = nditer(a, ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3])
    # 1D reversed order
    i = nditer(a[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 2, 1, 0])

def test_iter_best_order_f_index_2d():
    # The Fortran index should be correct with any reordering

    a = arange(6)
    # 2D C-order
    i = nditer(a.reshape(2, 3), ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 2, 4, 1, 3, 5])
    # 2D Fortran-order
    i = nditer(a.reshape(2, 3).copy(order='F'),
                                    ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5])
    # 2D reversed C-order
    i = nditer(a.reshape(2, 3)[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 3, 5, 0, 2, 4])
    i = nditer(a.reshape(2, 3)[:, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 2, 0, 5, 3, 1])
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 3, 1, 4, 2, 0])
    # 2D reversed Fortran-order
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1],
                                    ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [1, 0, 3, 2, 5, 4])
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1],
                                    ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [4, 5, 2, 3, 0, 1])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1],
                                    ['f_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 4, 3, 2, 1, 0])

def test_iter_best_order_f_index_3d():
    # The Fortran index should be correct with any reordering

    a = arange(12)
    # 3D C-order
    i = nditer(a.reshape(2, 3, 2), ['f_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])
    # 3D Fortran-order
    i = nditer(a.reshape(2, 3, 2).copy(order='F'),
                                    ['f_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # 3D reversed C-order
    i = nditer(a.reshape(2, 3, 2)[::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [1, 7, 3, 9, 5, 11, 0, 6, 2, 8, 4, 10])
    i = nditer(a.reshape(2, 3, 2)[:, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [4, 10, 2, 8, 0, 6, 5, 11, 3, 9, 1, 7])
    i = nditer(a.reshape(2, 3, 2)[:, :, ::-1], ['f_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [6, 0, 8, 2, 10, 4, 7, 1, 9, 3, 11, 5])
    # 3D reversed Fortran-order
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[::-1],
                                    ['f_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, ::-1],
                                    ['f_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7])
    i = nditer(a.reshape(2, 3, 2).copy(order='F')[:, :, ::-1],
                                    ['f_index'], [['readonly']])
    assert_equal(iter_indices(i),
                            [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])

def test_iter_no_inner_full_coalesce():
    # Check no_inner iterators which coalesce into a single inner loop

    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 3), (2, 3, 2, 2, 3)]:
        size = np.prod(shape)
        a = arange(size)
        # Test each combination of forward and backwards indexing
        for dirs in range(2**len(shape)):
            dirs_index = [slice(None)] * len(shape)
            for bit in range(len(shape)):
                if ((2**bit) & dirs):
                    dirs_index[bit] = slice(None, None, -1)
            dirs_index = tuple(dirs_index)

            aview = a.reshape(shape)[dirs_index]
            # C-order
            i = nditer(aview, ['external_loop'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i[0].shape, (size,))
            # Fortran-order
            i = nditer(aview.T, ['external_loop'], [['readonly']])
            assert_equal(i.ndim, 1)
            assert_equal(i[0].shape, (size,))
            # Other order
            if len(shape) > 2:
                i = nditer(aview.swapaxes(0, 1),
                                    ['external_loop'], [['readonly']])
                assert_equal(i.ndim, 1)
                assert_equal(i[0].shape, (size,))

def test_iter_no_inner_dim_coalescing():
    # Check no_inner iterators whose dimensions may not coalesce completely

    # Skipping the last element in a dimension prevents coalescing
    # with the next-bigger dimension
    a = arange(24).reshape(2, 3, 4)[:, :, :-1]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i[0].shape, (3,))
    a = arange(24).reshape(2, 3, 4)[:, :-1, :]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i[0].shape, (8,))
    a = arange(24).reshape(2, 3, 4)[:-1, :, :]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i[0].shape, (12,))

    # Even with lots of 1-sized dimensions, should still coalesce
    a = arange(24).reshape(1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 1)
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i[0].shape, (24,))

def test_iter_dim_coalescing():
    # Check that the correct number of dimensions are coalesced

    # Tracking a multi-index disables coalescing
    a = arange(24).reshape(2, 3, 4)
    i = nditer(a, ['multi_index'], [['readonly']])
    assert_equal(i.ndim, 3)

    # A tracked index can allow coalescing if it's compatible with the array
    a3d = arange(24).reshape(2, 3, 4)
    i = nditer(a3d, ['c_index'], [['readonly']])
    assert_equal(i.ndim, 1)
    i = nditer(a3d.swapaxes(0, 1), ['c_index'], [['readonly']])
    assert_equal(i.ndim, 3)
    i = nditer(a3d.T, ['c_index'], [['readonly']])
    assert_equal(i.ndim, 3)
    i = nditer(a3d.T, ['f_index'], [['readonly']])
    assert_equal(i.ndim, 1)
    i = nditer(a3d.T.swapaxes(0, 1), ['f_index'], [['readonly']])
    assert_equal(i.ndim, 3)

    # When C or F order is forced, coalescing may still occur
    a3d = arange(24).reshape(2, 3, 4)
    i = nditer(a3d, order='C')
    assert_equal(i.ndim, 1)
    i = nditer(a3d.T, order='C')
    assert_equal(i.ndim, 3)
    i = nditer(a3d, order='F')
    assert_equal(i.ndim, 3)
    i = nditer(a3d.T, order='F')
    assert_equal(i.ndim, 1)
    i = nditer(a3d, order='A')
    assert_equal(i.ndim, 1)
    i = nditer(a3d.T, order='A')
    assert_equal(i.ndim, 1)

def test_iter_broadcasting():
    # Standard NumPy broadcasting rules

    # 1D with scalar
    i = nditer([arange(6), np.int32(2)], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (6,))

    # 2D with scalar
    i = nditer([arange(6).reshape(2, 3), np.int32(2)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2, 3))
    # 2D with 1D
    i = nditer([arange(6).reshape(2, 3), arange(3)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2, 3))
    i = nditer([arange(2).reshape(2, 1), arange(3)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2, 3))
    # 2D with 2D
    i = nditer([arange(2).reshape(2, 1), arange(3).reshape(1, 3)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 6)
    assert_equal(i.shape, (2, 3))

    # 3D with scalar
    i = nditer([np.int32(2), arange(24).reshape(4, 2, 3)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    # 3D with 1D
    i = nditer([arange(3), arange(24).reshape(4, 2, 3)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(3), arange(8).reshape(4, 2, 1)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    # 3D with 2D
    i = nditer([arange(6).reshape(2, 3), arange(24).reshape(4, 2, 3)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(2).reshape(2, 1), arange(24).reshape(4, 2, 3)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(3).reshape(1, 3), arange(8).reshape(4, 2, 1)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    # 3D with 3D
    i = nditer([arange(2).reshape(1, 2, 1), arange(3).reshape(1, 1, 3),
                        arange(4).reshape(4, 1, 1)],
                        ['multi_index'], [['readonly']] * 3)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(6).reshape(1, 2, 3), arange(4).reshape(4, 1, 1)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))
    i = nditer([arange(24).reshape(4, 2, 3), arange(12).reshape(4, 1, 3)],
                        ['multi_index'], [['readonly']] * 2)
    assert_equal(i.itersize, 24)
    assert_equal(i.shape, (4, 2, 3))

def test_iter_itershape():
    # Check that allocated outputs work with a specified shape
    a = np.arange(6, dtype='i2').reshape(2, 3)
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']],
                            op_axes=[[0, 1, None], None],
                            itershape=(-1, -1, 4))
    assert_equal(i.operands[1].shape, (2, 3, 4))
    assert_equal(i.operands[1].strides, (24, 8, 2))

    i = nditer([a.T, None], [], [['readonly'], ['writeonly', 'allocate']],
                            op_axes=[[0, 1, None], None],
                            itershape=(-1, -1, 4))
    assert_equal(i.operands[1].shape, (3, 2, 4))
    assert_equal(i.operands[1].strides, (8, 24, 2))

    i = nditer([a.T, None], [], [['readonly'], ['writeonly', 'allocate']],
                            order='F',
                            op_axes=[[0, 1, None], None],
                            itershape=(-1, -1, 4))
    assert_equal(i.operands[1].shape, (3, 2, 4))
    assert_equal(i.operands[1].strides, (2, 6, 12))

    # If we specify 1 in the itershape, it shouldn't allow broadcasting
    # of that dimension to a bigger value
    assert_raises(ValueError, nditer, [a, None], [],
                            [['readonly'], ['writeonly', 'allocate']],
                            op_axes=[[0, 1, None], None],
                            itershape=(-1, 1, 4))
    # Test bug that for no op_axes but itershape, they are NULLed correctly
    i = np.nditer([np.ones(2), None, None], itershape=(2,))

def test_iter_broadcasting_errors():
    # Check that errors are thrown for bad broadcasting shapes

    # 1D with 1D
    assert_raises(ValueError, nditer, [arange(2), arange(3)],
                    [], [['readonly']] * 2)
    # 2D with 1D
    assert_raises(ValueError, nditer,
                    [arange(6).reshape(2, 3), arange(2)],
                    [], [['readonly']] * 2)
    # 2D with 2D
    assert_raises(ValueError, nditer,
                    [arange(6).reshape(2, 3), arange(9).reshape(3, 3)],
                    [], [['readonly']] * 2)
    assert_raises(ValueError, nditer,
                    [arange(6).reshape(2, 3), arange(4).reshape(2, 2)],
                    [], [['readonly']] * 2)
    # 3D with 3D
    assert_raises(ValueError, nditer,
                    [arange(36).reshape(3, 3, 4), arange(24).reshape(2, 3, 4)],
                    [], [['readonly']] * 2)
    assert_raises(ValueError, nditer,
                    [arange(8).reshape(2, 4, 1), arange(24).reshape(2, 3, 4)],
                    [], [['readonly']] * 2)

    # Verify that the error message mentions the right shapes
    try:
        nditer([arange(2).reshape(1, 2, 1),
                arange(3).reshape(1, 3),
                arange(6).reshape(2, 3)],
               [],
               [['readonly'], ['readonly'], ['writeonly', 'no_broadcast']])
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        # The message should contain the shape of the 3rd operand
        assert_(msg.find('(2,3)') >= 0,
                f'Message "{msg}" doesn\'t contain operand shape (2,3)')
        # The message should contain the broadcast shape
        assert_(msg.find('(1,2,3)') >= 0,
                f'Message "{msg}" doesn\'t contain broadcast shape (1,2,3)')

    try:
        nditer([arange(6).reshape(2, 3), arange(2)],
               [],
               [['readonly'], ['readonly']],
               op_axes=[[0, 1], [0, np.newaxis]],
               itershape=(4, 3))
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        # The message should contain "shape->remappedshape" for each operand
        assert_(msg.find('(2,3)->(2,3)') >= 0,
            f'Message "{msg}" doesn\'t contain operand shape (2,3)->(2,3)')
        assert_(msg.find('(2,)->(2,newaxis)') >= 0,
                ('Message "%s" doesn\'t contain remapped operand shape'
                '(2,)->(2,newaxis)') % msg)
        # The message should contain the itershape parameter
        assert_(msg.find('(4,3)') >= 0,
                f'Message "{msg}" doesn\'t contain itershape parameter (4,3)')

    try:
        nditer([np.zeros((2, 1, 1)), np.zeros((2,))],
               [],
               [['writeonly', 'no_broadcast'], ['readonly']])
        raise AssertionError('Should have raised a broadcast error')
    except ValueError as e:
        msg = str(e)
        # The message should contain the shape of the bad operand
        assert_(msg.find('(2,1,1)') >= 0,
            f'Message "{msg}" doesn\'t contain operand shape (2,1,1)')
        # The message should contain the broadcast shape
        assert_(msg.find('(2,1,2)') >= 0,
                f'Message "{msg}" doesn\'t contain the broadcast shape (2,1,2)')

def test_iter_flags_errors():
    # Check that bad combinations of flags produce errors

    a = arange(6)

    # Not enough operands
    assert_raises(ValueError, nditer, [], [], [])
    # Bad global flag
    assert_raises(ValueError, nditer, [a], ['bad flag'], [['readonly']])
    # Bad op flag
    assert_raises(ValueError, nditer, [a], [], [['readonly', 'bad flag']])
    # Bad order parameter
    assert_raises(ValueError, nditer, [a], [], [['readonly']], order='G')
    # Bad casting parameter
    assert_raises(ValueError, nditer, [a], [], [['readonly']], casting='noon')
    # op_flags must match ops
    assert_raises(ValueError, nditer, [a] * 3, [], [['readonly']] * 2)
    # Cannot track both a C and an F index
    assert_raises(ValueError, nditer, a,
                ['c_index', 'f_index'], [['readonly']])
    # Inner iteration and multi-indices/indices are incompatible
    assert_raises(ValueError, nditer, a,
                ['external_loop', 'multi_index'], [['readonly']])
    assert_raises(ValueError, nditer, a,
                ['external_loop', 'c_index'], [['readonly']])
    assert_raises(ValueError, nditer, a,
                ['external_loop', 'f_index'], [['readonly']])
    # Must specify exactly one of readwrite/readonly/writeonly per operand
    assert_raises(ValueError, nditer, a, [], [[]])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'writeonly']])
    assert_raises(ValueError, nditer, a, [], [['readonly', 'readwrite']])
    assert_raises(ValueError, nditer, a, [], [['writeonly', 'readwrite']])
    assert_raises(ValueError, nditer, a,
                [], [['readonly', 'writeonly', 'readwrite']])
    # Python scalars are always readonly
    assert_raises(TypeError, nditer, 1.5, [], [['writeonly']])
    assert_raises(TypeError, nditer, 1.5, [], [['readwrite']])
    # Array scalars are always readonly
    assert_raises(TypeError, nditer, np.int32(1), [], [['writeonly']])
    assert_raises(TypeError, nditer, np.int32(1), [], [['readwrite']])
    # Check readonly array
    a.flags.writeable = False
    assert_raises(ValueError, nditer, a, [], [['writeonly']])
    assert_raises(ValueError, nditer, a, [], [['readwrite']])
    a.flags.writeable = True
    # Multi-indices available only with the multi_index flag
    i = nditer(arange(6), [], [['readonly']])
    assert_raises(ValueError, lambda i: i.multi_index, i)
    # Index available only with an index flag
    assert_raises(ValueError, lambda i: i.index, i)
    # GotoCoords and GotoIndex incompatible with buffering or no_inner

    def assign_multi_index(i):
        i.multi_index = (0,)

    def assign_index(i):
        i.index = 0

    def assign_iterindex(i):
        i.iterindex = 0

    def assign_iterrange(i):
        i.iterrange = (0, 1)
    i = nditer(arange(6), ['external_loop'])
    assert_raises(ValueError, assign_multi_index, i)
    assert_raises(ValueError, assign_index, i)
    assert_raises(ValueError, assign_iterindex, i)
    assert_raises(ValueError, assign_iterrange, i)
    i = nditer(arange(6), ['buffered'])
    assert_raises(ValueError, assign_multi_index, i)
    assert_raises(ValueError, assign_index, i)
    assert_raises(ValueError, assign_iterrange, i)
    # Can't iterate if size is zero
    assert_raises(ValueError, nditer, np.array([]))

def test_iter_slice():
    a, b, c = np.arange(3), np.arange(3), np.arange(3.)
    i = nditer([a, b, c], [], ['readwrite'])
    with i:
        i[0:2] = (3, 3)
        assert_equal(a, [3, 1, 2])
        assert_equal(b, [3, 1, 2])
        assert_equal(c, [0, 1, 2])
        i[1] = 12
        assert_equal(i[0:2], [3, 12])

def test_iter_assign_mapping():
    a = np.arange(24, dtype='f8').reshape(2, 3, 4).T
    it = np.nditer(a, [], [['readwrite', 'updateifcopy']],
                       casting='same_kind', op_dtypes=[np.dtype('f4')])
    with it:
        it.operands[0][...] = 3
        it.operands[0][...] = 14
    assert_equal(a, 14)
    it = np.nditer(a, [], [['readwrite', 'updateifcopy']],
                       casting='same_kind', op_dtypes=[np.dtype('f4')])
    with it:
        x = it.operands[0][-1:1]
        x[...] = 14
        it.operands[0][...] = -1234
    assert_equal(a, -1234)
    # check for no warnings on dealloc
    x = None
    it = None

def test_iter_nbo_align_contig():
    # Check that byte order, alignment, and contig changes work

    # Byte order change by requesting a specific dtype
    a = np.arange(6, dtype='f4')
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    i = nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv',
                        op_dtypes=[np.dtype('f4')])
    with i:
        # context manager triggers WRITEBACKIFCOPY on i at exit
        assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0], a)
        i.operands[0][:] = 2
    assert_equal(au, [2] * 6)
    del i  # should not raise a warning
    # Byte order change by requesting NBO
    a = np.arange(6, dtype='f4')
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    with nditer(au, [], [['readwrite', 'updateifcopy', 'nbo']],
                        casting='equiv') as i:
        # context manager triggers UPDATEIFCOPY on i at exit
        assert_equal(i.dtypes[0].byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0].dtype.byteorder, a.dtype.byteorder)
        assert_equal(i.operands[0], a)
        i.operands[0][:] = 12345
        i.operands[0][:] = 2
    assert_equal(au, [2] * 6)

    # Unaligned input
    a = np.zeros((6 * 4 + 1,), dtype='i1')[1:]
    a.dtype = 'f4'
    a[:] = np.arange(6, dtype='f4')
    assert_(not a.flags.aligned)
    # Without 'aligned', shouldn't copy
    i = nditer(a, [], [['readonly']])
    assert_(not i.operands[0].flags.aligned)
    assert_equal(i.operands[0], a)
    # With 'aligned', should make a copy
    with nditer(a, [], [['readwrite', 'updateifcopy', 'aligned']]) as i:
        assert_(i.operands[0].flags.aligned)
        # context manager triggers UPDATEIFCOPY on i at exit
        assert_equal(i.operands[0], a)
        i.operands[0][:] = 3
    assert_equal(a, [3] * 6)

    # Discontiguous input
    a = arange(12)
    # If it is contiguous, shouldn't copy
    i = nditer(a[:6], [], [['readonly']])
    assert_(i.operands[0].flags.contiguous)
    assert_equal(i.operands[0], a[:6])
    # If it isn't contiguous, should buffer
    i = nditer(a[::2], ['buffered', 'external_loop'],
                        [['readonly', 'contig']],
                        buffersize=10)
    assert_(i[0].flags.contiguous)
    assert_equal(i[0], a[::2])

def test_iter_array_cast():
    # Check that arrays are cast as requested

    # No cast 'f4' -> 'f4'
    a = np.arange(6, dtype='f4').reshape(2, 3)
    i = nditer(a, [], [['readwrite']], op_dtypes=[np.dtype('f4')])
    with i:
        assert_equal(i.operands[0], a)
        assert_equal(i.operands[0].dtype, np.dtype('f4'))

    # Byte-order cast '<f4' -> '>f4'
    a = np.arange(6, dtype='<f4').reshape(2, 3)
    with nditer(a, [], [['readwrite', 'updateifcopy']],
            casting='equiv',
            op_dtypes=[np.dtype('>f4')]) as i:
        assert_equal(i.operands[0], a)
        assert_equal(i.operands[0].dtype, np.dtype('>f4'))

    # Safe case 'f4' -> 'f8'
    a = np.arange(24, dtype='f4').reshape(2, 3, 4).swapaxes(1, 2)
    i = nditer(a, [], [['readonly', 'copy']],
            casting='safe',
            op_dtypes=[np.dtype('f8')])
    assert_equal(i.operands[0], a)
    assert_equal(i.operands[0].dtype, np.dtype('f8'))
    # The memory layout of the temporary should match a (a is (48,4,16))
    # except negative strides get flipped to positive strides.
    assert_equal(i.operands[0].strides, (96, 8, 32))
    a = a[::-1, :, ::-1]
    i = nditer(a, [], [['readonly', 'copy']],
            casting='safe',
            op_dtypes=[np.dtype('f8')])
    assert_equal(i.operands[0], a)
    assert_equal(i.operands[0].dtype, np.dtype('f8'))
    assert_equal(i.operands[0].strides, (96, 8, 32))

    # Same-kind cast 'f8' -> 'f4' -> 'f8'
    a = np.arange(24, dtype='f8').reshape(2, 3, 4).T
    with nditer(a, [],
            [['readwrite', 'updateifcopy']],
            casting='same_kind',
            op_dtypes=[np.dtype('f4')]) as i:
        assert_equal(i.operands[0], a)
        assert_equal(i.operands[0].dtype, np.dtype('f4'))
        assert_equal(i.operands[0].strides, (4, 16, 48))
        # Check that WRITEBACKIFCOPY is activated at exit
        i.operands[0][2, 1, 1] = -12.5
        assert_(a[2, 1, 1] != -12.5)
    assert_equal(a[2, 1, 1], -12.5)

    a = np.arange(6, dtype='i4')[::-2]
    with nditer(a, [],
            [['writeonly', 'updateifcopy']],
            casting='unsafe',
            op_dtypes=[np.dtype('f4')]) as i:
        assert_equal(i.operands[0].dtype, np.dtype('f4'))
        # Even though the stride was negative in 'a', it
        # becomes positive in the temporary
        assert_equal(i.operands[0].strides, (4,))
        i.operands[0][:] = [1, 2, 3]
    assert_equal(a, [1, 2, 3])

def test_iter_array_cast_errors():
    # Check that invalid casts are caught

    # Need to enable copying for casts to occur
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
                [['readonly']], op_dtypes=[np.dtype('f8')])
    # Also need to allow casting for casts to occur
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
                [['readonly', 'copy']], casting='no',
                op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
                [['readonly', 'copy']], casting='equiv',
                op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, arange(2, dtype='f8'), [],
                [['writeonly', 'updateifcopy']],
                casting='no',
                op_dtypes=[np.dtype('f4')])
    assert_raises(TypeError, nditer, arange(2, dtype='f8'), [],
                [['writeonly', 'updateifcopy']],
                casting='equiv',
                op_dtypes=[np.dtype('f4')])
    # '<f4' -> '>f4' should not work with casting='no'
    assert_raises(TypeError, nditer, arange(2, dtype='<f4'), [],
                [['readonly', 'copy']], casting='no',
                op_dtypes=[np.dtype('>f4')])
    # 'f4' -> 'f8' is a safe cast, but 'f8' -> 'f4' isn't
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
                [['readwrite', 'updateifcopy']],
                casting='safe',
                op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, arange(2, dtype='f8'), [],
                [['readwrite', 'updateifcopy']],
                casting='safe',
                op_dtypes=[np.dtype('f4')])
    # 'f4' -> 'i4' is neither a safe nor a same-kind cast
    assert_raises(TypeError, nditer, arange(2, dtype='f4'), [],
                [['readonly', 'copy']],
                casting='same_kind',
                op_dtypes=[np.dtype('i4')])
    assert_raises(TypeError, nditer, arange(2, dtype='i4'), [],
                [['writeonly', 'updateifcopy']],
                casting='same_kind',
                op_dtypes=[np.dtype('f4')])

def test_iter_scalar_cast():
    # Check that scalars are cast as requested

    # No cast 'f4' -> 'f4'
    i = nditer(np.float32(2.5), [], [['readonly']],
                    op_dtypes=[np.dtype('f4')])
    assert_equal(i.dtypes[0], np.dtype('f4'))
    assert_equal(i.value.dtype, np.dtype('f4'))
    assert_equal(i.value, 2.5)
    # Safe cast 'f4' -> 'f8'
    i = nditer(np.float32(2.5), [],
                    [['readonly', 'copy']],
                    casting='safe',
                    op_dtypes=[np.dtype('f8')])
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.value.dtype, np.dtype('f8'))
    assert_equal(i.value, 2.5)
    # Same-kind cast 'f8' -> 'f4'
    i = nditer(np.float64(2.5), [],
                    [['readonly', 'copy']],
                    casting='same_kind',
                    op_dtypes=[np.dtype('f4')])
    assert_equal(i.dtypes[0], np.dtype('f4'))
    assert_equal(i.value.dtype, np.dtype('f4'))
    assert_equal(i.value, 2.5)
    # Unsafe cast 'f8' -> 'i4'
    i = nditer(np.float64(3.0), [],
                    [['readonly', 'copy']],
                    casting='unsafe',
                    op_dtypes=[np.dtype('i4')])
    assert_equal(i.dtypes[0], np.dtype('i4'))
    assert_equal(i.value.dtype, np.dtype('i4'))
    assert_equal(i.value, 3)
    # Readonly scalars may be cast even without setting COPY or BUFFERED
    i = nditer(3, [], [['readonly']], op_dtypes=[np.dtype('f8')])
    assert_equal(i[0].dtype, np.dtype('f8'))
    assert_equal(i[0], 3.)

def test_iter_scalar_cast_errors():
    # Check that invalid casts are caught

    # Need to allow copying/buffering for write casts of scalars to occur
    assert_raises(TypeError, nditer, np.float32(2), [],
                [['readwrite']], op_dtypes=[np.dtype('f8')])
    assert_raises(TypeError, nditer, 2.5, [],
                [['readwrite']], op_dtypes=[np.dtype('f4')])
    # 'f8' -> 'f4' isn't a safe cast if the value would overflow
    assert_raises(TypeError, nditer, np.float64(1e60), [],
                [['readonly']],
                casting='safe',
                op_dtypes=[np.dtype('f4')])
    # 'f4' -> 'i4' is neither a safe nor a same-kind cast
    assert_raises(TypeError, nditer, np.float32(2), [],
                [['readonly']],
                casting='same_kind',
                op_dtypes=[np.dtype('i4')])

def test_iter_object_arrays_basic():
    # Check that object arrays work

    obj = {'a': 3, 'b': 'd'}
    a = np.array([[1, 2, 3], None, obj, None], dtype='O')
    if HAS_REFCOUNT:
        rc = sys.getrefcount(obj)

    # Need to allow references for object arrays
    assert_raises(TypeError, nditer, a)
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)

    i = nditer(a, ['refs_ok'], ['readonly'])
    vals = [x_[()] for x_ in i]
    assert_equal(np.array(vals, dtype='O'), a)
    vals, i, x = [None] * 3
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)

    i = nditer(a.reshape(2, 2).T, ['refs_ok', 'buffered'],
                        ['readonly'], order='C')
    assert_(i.iterationneedsapi)
    vals = [x_[()] for x_ in i]
    assert_equal(np.array(vals, dtype='O'), a.reshape(2, 2).ravel(order='F'))
    vals, i, x = [None] * 3
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)

    i = nditer(a.reshape(2, 2).T, ['refs_ok', 'buffered'],
                        ['readwrite'], order='C')
    with i:
        for x in i:
            x[...] = None
        vals, i, x = [None] * 3
    if HAS_REFCOUNT:
        assert_(sys.getrefcount(obj) == rc - 1)
    assert_equal(a, np.array([None] * 4, dtype='O'))

def test_iter_object_arrays_conversions():
    # Conversions to/from objects
    a = np.arange(6, dtype='O')
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'],
                    casting='unsafe', op_dtypes='i4')
    with i:
        for x in i:
            x[...] += 1
    assert_equal(a, np.arange(6) + 1)

    a = np.arange(6, dtype='i4')
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'],
                    casting='unsafe', op_dtypes='O')
    with i:
        for x in i:
            x[...] += 1
    assert_equal(a, np.arange(6) + 1)

    # Non-contiguous object array
    a = np.zeros((6,), dtype=[('p', 'i1'), ('a', 'O')])
    a = a['a']
    a[:] = np.arange(6)
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'],
                    casting='unsafe', op_dtypes='i4')
    with i:
        for x in i:
            x[...] += 1
    assert_equal(a, np.arange(6) + 1)

    # Non-contiguous value array
    a = np.zeros((6,), dtype=[('p', 'i1'), ('a', 'i4')])
    a = a['a']
    a[:] = np.arange(6) + 98172488
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'],
                    casting='unsafe', op_dtypes='O')
    with i:
        ob = i[0][()]
        if HAS_REFCOUNT:
            rc = sys.getrefcount(ob)
        for x in i:
            x[...] += 1
        if HAS_REFCOUNT:
            newrc = sys.getrefcount(ob)
            assert_(newrc == rc - 1)
    assert_equal(a, np.arange(6) + 98172489)

def test_iter_common_dtype():
    # Check that the iterator finds a common data type correctly
    # (some checks are somewhat duplicate after adopting NEP 50)

    i = nditer([array([3], dtype='f4'), array([0], dtype='f8')],
                    ['common_dtype'],
                    [['readonly', 'copy']] * 2,
                    casting='safe')
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.dtypes[1], np.dtype('f8'))
    i = nditer([array([3], dtype='i4'), array([0], dtype='f4')],
                    ['common_dtype'],
                    [['readonly', 'copy']] * 2,
                    casting='safe')
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.dtypes[1], np.dtype('f8'))
    i = nditer([array([3], dtype='f4'), array(0, dtype='f8')],
                    ['common_dtype'],
                    [['readonly', 'copy']] * 2,
                    casting='same_kind')
    assert_equal(i.dtypes[0], np.dtype('f8'))
    assert_equal(i.dtypes[1], np.dtype('f8'))
    i = nditer([array([3], dtype='u4'), array(0, dtype='i4')],
                    ['common_dtype'],
                    [['readonly', 'copy']] * 2,
                    casting='safe')
    assert_equal(i.dtypes[0], np.dtype('i8'))
    assert_equal(i.dtypes[1], np.dtype('i8'))
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4')],
                    ['common_dtype'],
                    [['readonly', 'copy']] * 2,
                    casting='safe')
    assert_equal(i.dtypes[0], np.dtype('i8'))
    assert_equal(i.dtypes[1], np.dtype('i8'))
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4'),
                 array([2j], dtype='c8'), array([9], dtype='f8')],
                    ['common_dtype'],
                    [['readonly', 'copy']] * 4,
                    casting='safe')
    assert_equal(i.dtypes[0], np.dtype('c16'))
    assert_equal(i.dtypes[1], np.dtype('c16'))
    assert_equal(i.dtypes[2], np.dtype('c16'))
    assert_equal(i.dtypes[3], np.dtype('c16'))
    assert_equal(i.value, (3, -12, 2j, 9))

    # When allocating outputs, other outputs aren't factored in
    i = nditer([array([3], dtype='i4'), None, array([2j], dtype='c16')], [],
                    [['readonly', 'copy'],
                     ['writeonly', 'allocate'],
                     ['writeonly']],
                    casting='safe')
    assert_equal(i.dtypes[0], np.dtype('i4'))
    assert_equal(i.dtypes[1], np.dtype('i4'))
    assert_equal(i.dtypes[2], np.dtype('c16'))
    # But, if common data types are requested, they are
    i = nditer([array([3], dtype='i4'), None, array([2j], dtype='c16')],
                    ['common_dtype'],
                    [['readonly', 'copy'],
                     ['writeonly', 'allocate'],
                     ['writeonly']],
                    casting='safe')
    assert_equal(i.dtypes[0], np.dtype('c16'))
    assert_equal(i.dtypes[1], np.dtype('c16'))
    assert_equal(i.dtypes[2], np.dtype('c16'))

def test_iter_copy_if_overlap():
    # Ensure the iterator makes copies on read/write overlap, if requested

    # Copy not needed, 1 op
    for flag in ['readonly', 'writeonly', 'readwrite']:
        a = arange(10)
        i = nditer([a], ['copy_if_overlap'], [[flag]])
        with i:
            assert_(i.operands[0] is a)

    # Copy needed, 2 ops, read-write overlap
    x = arange(10)
    a = x[1:]
    b = x[:-1]
    with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['readwrite']]) as i:
        assert_(not np.shares_memory(*i.operands))

    # Copy not needed with elementwise, 2 ops, exactly same arrays
    x = arange(10)
    a = x
    b = x
    i = nditer([a, b], ['copy_if_overlap'], [['readonly', 'overlap_assume_elementwise'],
                                             ['readwrite', 'overlap_assume_elementwise']])
    with i:
        assert_(i.operands[0] is a and i.operands[1] is b)
    with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['readwrite']]) as i:
        assert_(i.operands[0] is a and not np.shares_memory(i.operands[1], b))

    # Copy not needed, 2 ops, no overlap
    x = arange(10)
    a = x[::2]
    b = x[1::2]
    i = nditer([a, b], ['copy_if_overlap'], [['readonly'], ['writeonly']])
    assert_(i.operands[0] is a and i.operands[1] is b)

    # Copy needed, 2 ops, read-write overlap
    x = arange(4, dtype=np.int8)
    a = x[3:]
    b = x.view(np.int32)[:1]
    with nditer([a, b], ['copy_if_overlap'], [['readonly'], ['writeonly']]) as i:
        assert_(not np.shares_memory(*i.operands))

    # Copy needed, 3 ops, read-write overlap
    for flag in ['writeonly', 'readwrite']:
        x = np.ones([10, 10])
        a = x
        b = x.T
        c = x
        with nditer([a, b, c], ['copy_if_overlap'],
                   [['readonly'], ['readonly'], [flag]]) as i:
            a2, b2, c2 = i.operands
            assert_(not np.shares_memory(a2, c2))
            assert_(not np.shares_memory(b2, c2))

    # Copy not needed, 3 ops, read-only overlap
    x = np.ones([10, 10])
    a = x
    b = x.T
    c = x
    i = nditer([a, b, c], ['copy_if_overlap'],
               [['readonly'], ['readonly'], ['readonly']])
    a2, b2, c2 = i.operands
    assert_(a is a2)
    assert_(b is b2)
    assert_(c is c2)

    # Copy not needed, 3 ops, read-only overlap
    x = np.ones([10, 10])
    a = x
    b = np.ones([10, 10])
    c = x.T
    i = nditer([a, b, c], ['copy_if_overlap'],
               [['readonly'], ['writeonly'], ['readonly']])
    a2, b2, c2 = i.operands
    assert_(a is a2)
    assert_(b is b2)
    assert_(c is c2)

    # Copy not needed, 3 ops, write-only overlap
    x = np.arange(7)
    a = x[:3]
    b = x[3:6]
    c = x[4:7]
    i = nditer([a, b, c], ['copy_if_overlap'],
               [['readonly'], ['writeonly'], ['writeonly']])
    a2, b2, c2 = i.operands
    assert_(a is a2)
    assert_(b is b2)
    assert_(c is c2)

def test_iter_op_axes():
    # Check that custom axes work

    # Reverse the axes
    a = arange(6).reshape(2, 3)
    i = nditer([a, a.T], [], [['readonly']] * 2, op_axes=[[0, 1], [1, 0]])
    assert_(all([x == y for (x, y) in i]))
    a = arange(24).reshape(2, 3, 4)
    i = nditer([a.T, a], [], [['readonly']] * 2, op_axes=[[2, 1, 0], None])
    assert_(all([x == y for (x, y) in i]))

    # Broadcast 1D to any dimension
    a = arange(1, 31).reshape(2, 3, 5)
    b = arange(1, 3)
    i = nditer([a, b], [], [['readonly']] * 2, op_axes=[None, [0, -1, -1]])
    assert_equal([x * y for (x, y) in i], (a * b.reshape(2, 1, 1)).ravel())
    b = arange(1, 4)
    i = nditer([a, b], [], [['readonly']] * 2, op_axes=[None, [-1, 0, -1]])
    assert_equal([x * y for (x, y) in i], (a * b.reshape(1, 3, 1)).ravel())
    b = arange(1, 6)
    i = nditer([a, b], [], [['readonly']] * 2,
                            op_axes=[None, [np.newaxis, np.newaxis, 0]])
    assert_equal([x * y for (x, y) in i], (a * b.reshape(1, 1, 5)).ravel())

    # Inner product-style broadcasting
    a = arange(24).reshape(2, 3, 4)
    b = arange(40).reshape(5, 2, 4)
    i = nditer([a, b], ['multi_index'], [['readonly']] * 2,
                            op_axes=[[0, 1, -1, -1], [-1, -1, 0, 1]])
    assert_equal(i.shape, (2, 3, 5, 2))

    # Matrix product-style broadcasting
    a = arange(12).reshape(3, 4)
    b = arange(20).reshape(4, 5)
    i = nditer([a, b], ['multi_index'], [['readonly']] * 2,
                            op_axes=[[0, -1], [-1, 1]])
    assert_equal(i.shape, (3, 5))

def test_iter_op_axes_errors():
    # Check that custom axes throws errors for bad inputs

    # Wrong number of items in op_axes
    a = arange(6).reshape(2, 3)
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2,
                                    op_axes=[[0], [1], [0]])
    # Out of bounds items in op_axes
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2,
                                    op_axes=[[2, 1], [0, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2,
                                    op_axes=[[0, 1], [2, -1]])
    # Duplicate items in op_axes
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2,
                                    op_axes=[[0, 0], [0, 1]])
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2,
                                    op_axes=[[0, 1], [1, 1]])

    # Different sized arrays in op_axes
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2,
                                    op_axes=[[0, 1], [0, 1, 0]])

    # Non-broadcastable dimensions in the result
    assert_raises(ValueError, nditer, [a, a], [], [['readonly']] * 2,
                                    op_axes=[[0, 1], [1, 0]])

def test_iter_copy():
    # Check that copying the iterator works correctly
    a = arange(24).reshape(2, 3, 4)

    # Simple iterator
    i = nditer(a)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    i.iterindex = 3
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    # Buffered iterator
    i = nditer(a, ['buffered', 'ranged'], order='F', buffersize=3)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    i.iterindex = 3
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    i.iterrange = (3, 9)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    i.iterrange = (2, 18)
    next(i)
    next(i)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])

    # Casting iterator
    with nditer(a, ['buffered'], order='F', casting='unsafe',
                op_dtypes='f8', buffersize=5) as i:
        j = i.copy()
    assert_equal([x[()] for x in j], a.ravel(order='F'))

    a = arange(24, dtype='<i4').reshape(2, 3, 4)
    with nditer(a, ['buffered'], order='F', casting='unsafe',
                op_dtypes='>f8', buffersize=5) as i:
        j = i.copy()
    assert_equal([x[()] for x in j], a.ravel(order='F'))


@pytest.mark.parametrize("dtype", np.typecodes["All"])
@pytest.mark.parametrize("loop_dtype", np.typecodes["All"])
@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
def test_iter_copy_casts(dtype, loop_dtype):
    # Ensure the dtype is never flexible:
    if loop_dtype.lower() == "m":
        loop_dtype = loop_dtype + "[ms]"
    elif np.dtype(loop_dtype).itemsize == 0:
        loop_dtype = loop_dtype + "50"

    # Make things a bit more interesting by requiring a byte-swap as well:
    arr = np.ones(1000, dtype=np.dtype(dtype).newbyteorder())
    try:
        expected = arr.astype(loop_dtype)
    except Exception:
        # Some casts are not possible, do not worry about them
        return

    it = np.nditer((arr,), ["buffered", "external_loop", "refs_ok"],
                   op_dtypes=[loop_dtype], casting="unsafe")

    if np.issubdtype(np.dtype(loop_dtype), np.number):
        # Casting to strings may be strange, but for simple dtypes do not rely
        # on the cast being correct:
        assert_array_equal(expected, np.ones(1000, dtype=loop_dtype))

    it_copy = it.copy()
    res = next(it)
    del it
    res_copy = next(it_copy)
    del it_copy

    assert_array_equal(res, expected)
    assert_array_equal(res_copy, expected)


def test_iter_copy_casts_structured():
    # Test a complicated structured dtype for casting, as it requires
    # both multiple steps and a more complex casting setup.
    # Includes a structured -> unstructured (any to object), and many other
    # casts, which cause this to require all steps in the casting machinery
    # one level down as well as the iterator copy (which uses NpyAuxData clone)
    in_dtype = np.dtype([("a", np.dtype("i,")),
                         ("b", np.dtype(">i,<i,>d,S17,>d,3f,O,i1"))])
    out_dtype = np.dtype([("a", np.dtype("O")),
                          ("b", np.dtype(">i,>i,S17,>d,>U3,3d,i1,O"))])
    arr = np.ones(1000, dtype=in_dtype)

    it = np.nditer((arr,), ["buffered", "external_loop", "refs_ok"],
                   op_dtypes=[out_dtype], casting="unsafe")
    it_copy = it.copy()

    res1 = next(it)
    del it
    res2 = next(it_copy)
    del it_copy

    expected = arr["a"].astype(out_dtype["a"])
    assert_array_equal(res1["a"], expected)
    assert_array_equal(res2["a"], expected)

    for field in in_dtype["b"].names:
        # Note that the .base avoids the subarray field
        expected = arr["b"][field].astype(out_dtype["b"][field].base)
        assert_array_equal(res1["b"][field], expected)
        assert_array_equal(res2["b"][field], expected)


def test_iter_copy_casts_structured2():
    # Similar to the above, this is a fairly arcane test to cover internals
    in_dtype = np.dtype([("a", np.dtype("O,O")),
                         ("b", np.dtype("5O,3O,(1,)O,(1,)i,(1,)O"))])
    out_dtype = np.dtype([("a", np.dtype("O")),
                          ("b", np.dtype("O,3i,4O,4O,4i"))])

    arr = np.ones(1, dtype=in_dtype)
    it = np.nditer((arr,), ["buffered", "external_loop", "refs_ok"],
                   op_dtypes=[out_dtype], casting="unsafe")
    it_copy = it.copy()

    res1 = next(it)
    del it
    res2 = next(it_copy)
    del it_copy

    # Array of two structured scalars:
    for res in res1, res2:
        # Cast to tuple by getitem, which may be weird and changeable?:
        assert isinstance(res["a"][0], tuple)
        assert res["a"][0] == (1, 1)

    for res in res1, res2:
        assert_array_equal(res["b"]["f0"][0], np.ones(5, dtype=object))
        assert_array_equal(res["b"]["f1"], np.ones((1, 3), dtype="i"))
        assert res["b"]["f2"].shape == (1, 4)
        assert_array_equal(res["b"]["f2"][0], np.ones(4, dtype=object))
        assert_array_equal(res["b"]["f3"][0], np.ones(4, dtype=object))
        assert_array_equal(res["b"]["f3"][0], np.ones(4, dtype="i"))


def test_iter_allocate_output_simple():
    # Check that the iterator will properly allocate outputs

    # Simple case
    a = arange(6)
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].dtype, np.dtype('f4'))

def test_iter_allocate_output_buffered_readwrite():
    # Allocated output with buffering + delay_bufalloc

    a = arange(6)
    i = nditer([a, None], ['buffered', 'delay_bufalloc'],
                        [['readonly'], ['allocate', 'readwrite']])
    with i:
        i.operands[1][:] = 1
        i.reset()
        for x in i:
            x[1][...] += x[0][...]
        assert_equal(i.operands[1], a + 1)

def test_iter_allocate_output_itorder():
    # The allocated output should match the iteration order

    # C-order input, best iteration order
    a = arange(6, dtype='i4').reshape(2, 3)
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].strides, a.strides)
    assert_equal(i.operands[1].dtype, np.dtype('f4'))
    # F-order input, best iteration order
    a = arange(24, dtype='i4').reshape(2, 3, 4).T
    i = nditer([a, None], [], [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].strides, a.strides)
    assert_equal(i.operands[1].dtype, np.dtype('f4'))
    # Non-contiguous input, C iteration order
    a = arange(24, dtype='i4').reshape(2, 3, 4).swapaxes(0, 1)
    i = nditer([a, None], [],
                        [['readonly'], ['writeonly', 'allocate']],
                        order='C',
                        op_dtypes=[None, np.dtype('f4')])
    assert_equal(i.operands[1].shape, a.shape)
    assert_equal(i.operands[1].strides, (32, 16, 4))
    assert_equal(i.operands[1].dtype, np.dtype('f4'))

def test_iter_allocate_output_opaxes():
    # Specifying op_axes should work

    a = arange(24, dtype='i4').reshape(2, 3, 4)
    i = nditer([None, a], [], [['writeonly', 'allocate'], ['readonly']],
                        op_dtypes=[np.dtype('u4'), None],
                        op_axes=[[1, 2, 0], None])
    assert_equal(i.operands[0].shape, (4, 2, 3))
    assert_equal(i.operands[0].strides, (4, 48, 16))
    assert_equal(i.operands[0].dtype, np.dtype('u4'))

def test_iter_allocate_output_types_promotion():
    # Check type promotion of automatic outputs (this was more interesting
    # before NEP 50...)

    i = nditer([array([3], dtype='f4'), array([0], dtype='f8'), None], [],
                    [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'))
    i = nditer([array([3], dtype='i4'), array([0], dtype='f4'), None], [],
                    [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'))
    i = nditer([array([3], dtype='f4'), array(0, dtype='f8'), None], [],
                    [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('f8'))
    i = nditer([array([3], dtype='u4'), array(0, dtype='i4'), None], [],
                    [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('i8'))
    i = nditer([array([3], dtype='u4'), array(-12, dtype='i4'), None], [],
                    [['readonly']] * 2 + [['writeonly', 'allocate']])
    assert_equal(i.dtypes[2], np.dtype('i8'))

def test_iter_allocate_output_types_byte_order():
    # Verify the rules for byte order changes

    # When there's just one input, the output type exactly matches
    a = array([3], dtype='u4')
    a = a.view(a.dtype.newbyteorder())
    i = nditer([a, None], [],
                    [['readonly'], ['writeonly', 'allocate']])
    assert_equal(i.dtypes[0], i.dtypes[1])
    # With two or more inputs, the output type is in native byte order
    i = nditer([a, a, None], [],
                    [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    assert_(i.dtypes[0] != i.dtypes[2])
    assert_equal(i.dtypes[0].newbyteorder('='), i.dtypes[2])

def test_iter_allocate_output_types_scalar():
    # If the inputs are all scalars, the output should be a scalar

    i = nditer([None, 1, 2.3, np.float32(12), np.complex128(3)], [],
                [['writeonly', 'allocate']] + [['readonly']] * 4)
    assert_equal(i.operands[0].dtype, np.dtype('complex128'))
    assert_equal(i.operands[0].ndim, 0)

def test_iter_allocate_output_subtype():
    # Make sure that the subtype with priority wins
    class MyNDArray(np.ndarray):
        __array_priority__ = 15

    # subclass vs ndarray
    a = np.array([[1, 2], [3, 4]]).view(MyNDArray)
    b = np.arange(4).reshape(2, 2).T
    i = nditer([a, b, None], [],
               [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    assert_equal(type(a), type(i.operands[2]))
    assert_(type(b) is not type(i.operands[2]))
    assert_equal(i.operands[2].shape, (2, 2))

    # If subtypes are disabled, we should get back an ndarray.
    i = nditer([a, b, None], [],
               [['readonly'], ['readonly'],
                ['writeonly', 'allocate', 'no_subtype']])
    assert_equal(type(b), type(i.operands[2]))
    assert_(type(a) is not type(i.operands[2]))
    assert_equal(i.operands[2].shape, (2, 2))

def test_iter_allocate_output_errors():
    # Check that the iterator will throw errors for bad output allocations

    # Need an input if no output data type is specified
    a = arange(6)
    assert_raises(TypeError, nditer, [a, None], [],
                        [['writeonly'], ['writeonly', 'allocate']])
    # Allocated output should be flagged for writing
    assert_raises(ValueError, nditer, [a, None], [],
                        [['readonly'], ['allocate', 'readonly']])
    # Allocated output can't have buffering without delayed bufalloc
    assert_raises(ValueError, nditer, [a, None], ['buffered'],
                                            ['allocate', 'readwrite'])
    # Must specify dtype if there are no inputs (cannot promote existing ones;
    # maybe this should use the 'f4' here, but it does not historically.)
    assert_raises(TypeError, nditer, [None, None], [],
                        [['writeonly', 'allocate'],
                         ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')])
    # If using op_axes, must specify all the axes
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    assert_raises(ValueError, nditer, [a, None], [],
                        [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')],
                        op_axes=[None, [0, np.newaxis, 1]])
    # If using op_axes, the axes must be within bounds
    assert_raises(ValueError, nditer, [a, None], [],
                        [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')],
                        op_axes=[None, [0, 3, 1]])
    # If using op_axes, there can't be duplicates
    assert_raises(ValueError, nditer, [a, None], [],
                        [['readonly'], ['writeonly', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')],
                        op_axes=[None, [0, 2, 1, 0]])
    # Not all axes may be specified if a reduction. If there is a hole
    # in op_axes, this is an error.
    a = arange(24, dtype='i4').reshape(2, 3, 4)
    assert_raises(ValueError, nditer, [a, None], ["reduce_ok"],
                        [['readonly'], ['readwrite', 'allocate']],
                        op_dtypes=[None, np.dtype('f4')],
                        op_axes=[None, [0, np.newaxis, 2]])

def test_all_allocated():
    # When no output and no shape is given, `()` is used as shape.
    i = np.nditer([None], op_dtypes=["int64"])
    assert i.operands[0].shape == ()
    assert i.dtypes == (np.dtype("int64"),)

    i = np.nditer([None], op_dtypes=["int64"], itershape=(2, 3, 4))
    assert i.operands[0].shape == (2, 3, 4)

def test_iter_remove_axis():
    a = arange(24).reshape(2, 3, 4)

    i = nditer(a, ['multi_index'])
    i.remove_axis(1)
    assert_equal(list(i), a[:, 0, :].ravel())

    a = a[::-1, :, :]
    i = nditer(a, ['multi_index'])
    i.remove_axis(0)
    assert_equal(list(i), a[0, :, :].ravel())

def test_iter_remove_multi_index_inner_loop():
    # Check that removing multi-index support works

    a = arange(24).reshape(2, 3, 4)

    i = nditer(a, ['multi_index'])
    assert_equal(i.ndim, 3)
    assert_equal(i.shape, (2, 3, 4))
    assert_equal(i.itviews[0].shape, (2, 3, 4))

    # Removing the multi-index tracking causes all dimensions to coalesce
    before = list(i)
    i.remove_multi_index()
    after = list(i)

    assert_equal(before, after)
    assert_equal(i.ndim, 1)
    assert_raises(ValueError, lambda i: i.shape, i)
    assert_equal(i.itviews[0].shape, (24,))

    # Removing the inner loop means there's just one iteration
    i.reset()
    assert_equal(i.itersize, 24)
    assert_equal(i[0].shape, ())
    i.enable_external_loop()
    assert_equal(i.itersize, 24)
    assert_equal(i[0].shape, (24,))
    assert_equal(i.value, arange(24))

def test_iter_iterindex():
    # Make sure iterindex works

    buffersize = 5
    a = arange(24).reshape(4, 3, 2)
    for flags in ([], ['buffered']):
        i = nditer(a, flags, buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 2
        assert_equal(iter_iterindices(i), list(range(2, 24)))

        i = nditer(a, flags, order='F', buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 5
        assert_equal(iter_iterindices(i), list(range(5, 24)))

        i = nditer(a[::-1], flags, order='F', buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 9
        assert_equal(iter_iterindices(i), list(range(9, 24)))

        i = nditer(a[::-1, ::-1], flags, order='C', buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 13
        assert_equal(iter_iterindices(i), list(range(13, 24)))

        i = nditer(a[::1, ::-1], flags, buffersize=buffersize)
        assert_equal(iter_iterindices(i), list(range(24)))
        i.iterindex = 23
        assert_equal(iter_iterindices(i), list(range(23, 24)))
        i.reset()
        i.iterindex = 2
        assert_equal(iter_iterindices(i), list(range(2, 24)))

def test_iter_iterrange():
    # Make sure getting and resetting the iterrange works

    buffersize = 5
    a = arange(24, dtype='i4').reshape(4, 3, 2)
    a_fort = a.ravel(order='F')

    i = nditer(a, ['ranged'], ['readonly'], order='F',
                buffersize=buffersize)
    assert_equal(i.iterrange, (0, 24))
    assert_equal([x[()] for x in i], a_fort)
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        assert_equal([x[()] for x in i], a_fort[r[0]:r[1]])

    i = nditer(a, ['ranged', 'buffered'], ['readonly'], order='F',
                op_dtypes='f8', buffersize=buffersize)
    assert_equal(i.iterrange, (0, 24))
    assert_equal([x[()] for x in i], a_fort)
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        assert_equal([x[()] for x in i], a_fort[r[0]:r[1]])

    def get_array(i):
        val = np.array([], dtype='f8')
        for x in i:
            val = np.concatenate((val, x))
        return val

    i = nditer(a, ['ranged', 'buffered', 'external_loop'],
                ['readonly'], order='F',
                op_dtypes='f8', buffersize=buffersize)
    assert_equal(i.iterrange, (0, 24))
    assert_equal(get_array(i), a_fort)
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        assert_equal(get_array(i), a_fort[r[0]:r[1]])

def test_iter_buffering():
    # Test buffering with several buffer sizes and types
    arrays = []
    # F-order swapped array
    _tmp = np.arange(24, dtype='c16').reshape(2, 3, 4).T
    _tmp = _tmp.view(_tmp.dtype.newbyteorder()).byteswap()
    arrays.append(_tmp)
    # Contiguous 1-dimensional array
    arrays.append(np.arange(10, dtype='f4'))
    # Unaligned array
    a = np.zeros((4 * 16 + 1,), dtype='i1')[1:]
    a.dtype = 'i4'
    a[:] = np.arange(16, dtype='i4')
    arrays.append(a)
    # 4-D F-order array
    arrays.append(np.arange(120, dtype='i4').reshape(5, 3, 2, 4).T)
    for a in arrays:
        for buffersize in (1, 2, 3, 5, 8, 11, 16, 1024):
            vals = []
            i = nditer(a, ['buffered', 'external_loop'],
                           [['readonly', 'nbo', 'aligned']],
                           order='C',
                           casting='equiv',
                           buffersize=buffersize)
            while not i.finished:
                assert_(i[0].size <= buffersize)
                vals.append(i[0].copy())
                i.iternext()
            assert_equal(np.concatenate(vals), a.ravel(order='C'))

def test_iter_write_buffering():
    # Test that buffering of writes is working

    # F-order swapped array
    a = np.arange(24).reshape(2, 3, 4).T
    a = a.view(a.dtype.newbyteorder()).byteswap()
    i = nditer(a, ['buffered'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='equiv',
                   order='C',
                   buffersize=16)
    x = 0
    with i:
        while not i.finished:
            i[0] = x
            x += 1
            i.iternext()
    assert_equal(a.ravel(order='C'), np.arange(24))

def test_iter_buffering_delayed_alloc():
    # Test that delaying buffer allocation works

    a = np.arange(6)
    b = np.arange(1, dtype='f4')
    i = nditer([a, b], ['buffered', 'delay_bufalloc', 'multi_index', 'reduce_ok'],
                    ['readwrite'],
                    casting='unsafe',
                    op_dtypes='f4')
    assert_(i.has_delayed_bufalloc)
    assert_raises(ValueError, lambda i: i.multi_index, i)
    assert_raises(ValueError, lambda i: i[0], i)
    assert_raises(ValueError, lambda i: i[0:2], i)

    def assign_iter(i):
        i[0] = 0
    assert_raises(ValueError, assign_iter, i)

    i.reset()
    assert_(not i.has_delayed_bufalloc)
    assert_equal(i.multi_index, (0,))
    with i:
        assert_equal(i[0], 0)
        i[1] = 1
        assert_equal(i[0:2], [0, 1])
        assert_equal([[x[0][()], x[1][()]] for x in i], list(zip(range(6), [1] * 6)))

def test_iter_buffered_cast_simple():
    # Test that buffering can handle a simple cast

    a = np.arange(10, dtype='f4')
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('f8')],
                   buffersize=3)
    with i:
        for v in i:
            v[...] *= 2

    assert_equal(a, 2 * np.arange(10, dtype='f4'))

def test_iter_buffered_cast_byteswapped():
    # Test that buffering can handle a cast which requires swap->cast->swap

    a = np.arange(10, dtype='f4')
    a = a.view(a.dtype.newbyteorder()).byteswap()
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('f8').newbyteorder()],
                   buffersize=3)
    with i:
        for v in i:
            v[...] *= 2

    assert_equal(a, 2 * np.arange(10, dtype='f4'))

    with suppress_warnings() as sup:
        sup.filter(np.exceptions.ComplexWarning)

        a = np.arange(10, dtype='f8')
        a = a.view(a.dtype.newbyteorder()).byteswap()
        i = nditer(a, ['buffered', 'external_loop'],
                       [['readwrite', 'nbo', 'aligned']],
                       casting='unsafe',
                       op_dtypes=[np.dtype('c8').newbyteorder()],
                       buffersize=3)
        with i:
            for v in i:
                v[...] *= 2

        assert_equal(a, 2 * np.arange(10, dtype='f8'))

def test_iter_buffered_cast_byteswapped_complex():
    # Test that buffering can handle a cast which requires swap->cast->copy

    a = np.arange(10, dtype='c8')
    a = a.view(a.dtype.newbyteorder()).byteswap()
    a += 2j
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('c16')],
                   buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype='c8') + 4j)

    a = np.arange(10, dtype='c8')
    a += 2j
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('c16').newbyteorder()],
                   buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype='c8') + 4j)

    a = np.arange(10, dtype=np.clongdouble)
    a = a.view(a.dtype.newbyteorder()).byteswap()
    a += 2j
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('c16')],
                   buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype=np.clongdouble) + 4j)

    a = np.arange(10, dtype=np.longdouble)
    a = a.view(a.dtype.newbyteorder()).byteswap()
    i = nditer(a, ['buffered', 'external_loop'],
                   [['readwrite', 'nbo', 'aligned']],
                   casting='same_kind',
                   op_dtypes=[np.dtype('f4')],
                   buffersize=7)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype=np.longdouble))

def test_iter_buffered_cast_structured_type():
    # Tests buffering of structured types

    # simple -> struct type (duplicates the value)
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.arange(3, dtype='f4') + 0.5
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt)
    vals = [np.array(x) for x in i]
    assert_equal(vals[0]['a'], 0.5)
    assert_equal(vals[0]['b'], 0)
    assert_equal(vals[0]['c'], [[(0.5)] * 3] * 2)
    assert_equal(vals[0]['d'], 0.5)
    assert_equal(vals[1]['a'], 1.5)
    assert_equal(vals[1]['b'], 1)
    assert_equal(vals[1]['c'], [[(1.5)] * 3] * 2)
    assert_equal(vals[1]['d'], 1.5)
    assert_equal(vals[0].dtype, np.dtype(sdt))

    # object -> struct type
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.zeros((3,), dtype='O')
    a[0] = (0.5, 0.5, [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], 0.5)
    a[1] = (1.5, 1.5, [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]], 1.5)
    a[2] = (2.5, 2.5, [[2.5, 2.5, 2.5], [2.5, 2.5, 2.5]], 2.5)
    if HAS_REFCOUNT:
        rc = sys.getrefcount(a[0])
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt)
    vals = [x.copy() for x in i]
    assert_equal(vals[0]['a'], 0.5)
    assert_equal(vals[0]['b'], 0)
    assert_equal(vals[0]['c'], [[(0.5)] * 3] * 2)
    assert_equal(vals[0]['d'], 0.5)
    assert_equal(vals[1]['a'], 1.5)
    assert_equal(vals[1]['b'], 1)
    assert_equal(vals[1]['c'], [[(1.5)] * 3] * 2)
    assert_equal(vals[1]['d'], 1.5)
    assert_equal(vals[0].dtype, np.dtype(sdt))
    vals, i, x = [None] * 3
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(a[0]), rc)

    # single-field struct type -> simple
    sdt = [('a', 'f4')]
    a = np.array([(5.5,), (8,)], dtype=sdt)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes='i4')
    assert_equal([x_[()] for x_ in i], [5, 8])

    # make sure multi-field struct type -> simple doesn't work
    sdt = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    a = np.array([(5.5, 7, 'test'), (8, 10, 11)], dtype=sdt)
    assert_raises(TypeError, lambda: (
        nditer(a, ['buffered', 'refs_ok'], ['readonly'],
               casting='unsafe',
               op_dtypes='i4')))

    # struct type -> struct type (field-wise copy)
    sdt1 = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    sdt2 = [('d', 'u2'), ('a', 'O'), ('b', 'f8')]
    a = np.array([(1, 2, 3), (4, 5, 6)], dtype=sdt1)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    assert_equal([np.array(x_) for x_ in i],
                 [np.array((1, 2, 3), dtype=sdt2),
                  np.array((4, 5, 6), dtype=sdt2)])


def test_iter_buffered_cast_structured_type_failure_with_cleanup():
    # make sure struct type -> struct type with different
    # number of fields fails
    sdt1 = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    sdt2 = [('b', 'O'), ('a', 'f8')]
    a = np.array([(1, 2, 3), (4, 5, 6)], dtype=sdt1)

    for intent in ["readwrite", "readonly", "writeonly"]:
        # This test was initially designed to test an error at a different
        # place, but will now raise earlier to to the cast not being possible:
        # `assert np.can_cast(a.dtype, sdt2, casting="unsafe")` fails.
        # Without a faulty DType, there is probably no reliable
        # way to get the initial tested behaviour.
        simple_arr = np.array([1, 2], dtype="i,i")  # requires clean up
        with pytest.raises(TypeError):
            nditer((simple_arr, a), ['buffered', 'refs_ok'], [intent, intent],
                   casting='unsafe', op_dtypes=["f,f", sdt2])


def test_buffered_cast_error_paths():
    with pytest.raises(ValueError):
        # The input is cast into an `S3` buffer
        np.nditer((np.array("a", dtype="S1"),), op_dtypes=["i"],
                  casting="unsafe", flags=["buffered"])

    # The `M8[ns]` is cast into the `S3` output
    it = np.nditer((np.array(1, dtype="i"),), op_dtypes=["S1"],
                   op_flags=["writeonly"], casting="unsafe", flags=["buffered"])
    with pytest.raises(ValueError):
        with it:
            buf = next(it)
            buf[...] = "a"  # cannot be converted to int.

@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
@pytest.mark.skipif(not HAS_REFCOUNT, reason="PyPy seems to not hit this.")
def test_buffered_cast_error_paths_unraisable():
    # The following gives an unraisable error. Pytest sometimes captures that
    # (depending python and/or pytest version). So with Python>=3.8 this can
    # probably be cleaned out in the future to check for
    # pytest.PytestUnraisableExceptionWarning:
    code = textwrap.dedent("""
        import numpy as np

        it = np.nditer((np.array(1, dtype="i"),), op_dtypes=["S1"],
                       op_flags=["writeonly"], casting="unsafe", flags=["buffered"])
        buf = next(it)
        buf[...] = "a"
        del buf, it  # Flushing only happens during deallocate right now.
        """)
    res = subprocess.check_output([sys.executable, "-c", code],
                                  stderr=subprocess.STDOUT, text=True)
    assert "ValueError" in res


def test_iter_buffered_cast_subarray():
    # Tests buffering of subarrays

    # one element -> many (copies it to all)
    sdt1 = [('a', 'f4')]
    sdt2 = [('a', 'f8', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    for x, count in zip(i, list(range(6))):
        assert_(np.all(x['a'] == count))

    # one element -> many -> back (copies it to all)
    sdt1 = [('a', 'O', (1, 1))]
    sdt2 = [('a', 'O', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readwrite'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    with i:
        assert_equal(i[0].dtype, np.dtype(sdt2))
        count = 0
        for x in i:
            assert_(np.all(x['a'] == count))
            x['a'][0] += 2
            count += 1
    assert_equal(a['a'], np.arange(6).reshape(6, 1, 1) + 2)

    # many -> one element -> back (copies just element 0)
    sdt1 = [('a', 'O', (3, 2, 2))]
    sdt2 = [('a', 'O', (1,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readwrite'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    with i:
        assert_equal(i[0].dtype, np.dtype(sdt2))
        count = 0
        for x in i:
            assert_equal(x['a'], count)
            x['a'] += 2
            count += 1
    assert_equal(a['a'], np.arange(6).reshape(6, 1, 1, 1) * np.ones((1, 3, 2, 2)) + 2)

    # many -> one element -> back (copies just element 0)
    sdt1 = [('a', 'f8', (3, 2, 2))]
    sdt2 = [('a', 'O', (1,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], count)
        count += 1

    # many -> one element (copies just element 0)
    sdt1 = [('a', 'O', (3, 2, 2))]
    sdt2 = [('a', 'f4', (1,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], count)
        count += 1

    # many -> matching shape (straightforward copy)
    sdt1 = [('a', 'O', (3, 2, 2))]
    sdt2 = [('a', 'f4', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 3 * 2 * 2).reshape(6, 3, 2, 2)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], a[count]['a'])
        count += 1

    # vector -> smaller vector (truncates)
    sdt1 = [('a', 'f8', (6,))]
    sdt2 = [('a', 'f4', (2,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 6).reshape(6, 6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], a[count]['a'][:2])
        count += 1

    # vector -> bigger vector (pads with zeros)
    sdt1 = [('a', 'f8', (2,))]
    sdt2 = [('a', 'f4', (6,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2).reshape(6, 2)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][:2], a[count]['a'])
        assert_equal(x['a'][2:], [0, 0, 0, 0])
        count += 1

    # vector -> matrix (broadcasts)
    sdt1 = [('a', 'f8', (2,))]
    sdt2 = [('a', 'f4', (2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2).reshape(6, 2)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][0], a[count]['a'])
        assert_equal(x['a'][1], a[count]['a'])
        count += 1

    # vector -> matrix (broadcasts and zero-pads)
    sdt1 = [('a', 'f8', (2, 1))]
    sdt2 = [('a', 'f4', (3, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2).reshape(6, 2, 1)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][:2, 0], a[count]['a'][:, 0])
        assert_equal(x['a'][:2, 1], a[count]['a'][:, 0])
        assert_equal(x['a'][2, :], [0, 0])
        count += 1

    # matrix -> matrix (truncates and zero-pads)
    sdt1 = [('a', 'f8', (2, 3))]
    sdt2 = [('a', 'f4', (3, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2 * 3).reshape(6, 2, 3)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe',
                    op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][:2, 0], a[count]['a'][:, 0])
        assert_equal(x['a'][:2, 1], a[count]['a'][:, 1])
        assert_equal(x['a'][2, :], [0, 0])
        count += 1

def test_iter_buffering_badwriteback():
    # Writing back from a buffer cannot combine elements

    # a needs write buffering, but had a broadcast dimension
    a = np.arange(6).reshape(2, 3, 1)
    b = np.arange(12).reshape(2, 3, 2)
    assert_raises(ValueError, nditer, [a, b],
                  ['buffered', 'external_loop'],
                  [['readwrite'], ['writeonly']],
                  order='C')

    # But if a is readonly, it's fine
    nditer([a, b], ['buffered', 'external_loop'],
           [['readonly'], ['writeonly']],
           order='C')

    # If a has just one element, it's fine too (constant 0 stride, a reduction)
    a = np.arange(1).reshape(1, 1, 1)
    nditer([a, b], ['buffered', 'external_loop', 'reduce_ok'],
           [['readwrite'], ['writeonly']],
           order='C')

    # check that it fails on other dimensions too
    a = np.arange(6).reshape(1, 3, 2)
    assert_raises(ValueError, nditer, [a, b],
                  ['buffered', 'external_loop'],
                  [['readwrite'], ['writeonly']],
                  order='C')
    a = np.arange(4).reshape(2, 1, 2)
    assert_raises(ValueError, nditer, [a, b],
                  ['buffered', 'external_loop'],
                  [['readwrite'], ['writeonly']],
                  order='C')

def test_iter_buffering_string():
    # Safe casting disallows shrinking strings
    a = np.array(['abc', 'a', 'abcd'], dtype=np.bytes_)
    assert_equal(a.dtype, np.dtype('S4'))
    assert_raises(TypeError, nditer, a, ['buffered'], ['readonly'],
                  op_dtypes='S2')
    i = nditer(a, ['buffered'], ['readonly'], op_dtypes='S6')
    assert_equal(i[0], b'abc')
    assert_equal(i[0].dtype, np.dtype('S6'))

    a = np.array(['abc', 'a', 'abcd'], dtype=np.str_)
    assert_equal(a.dtype, np.dtype('U4'))
    assert_raises(TypeError, nditer, a, ['buffered'], ['readonly'],
                    op_dtypes='U2')
    i = nditer(a, ['buffered'], ['readonly'], op_dtypes='U6')
    assert_equal(i[0], 'abc')
    assert_equal(i[0].dtype, np.dtype('U6'))

def test_iter_buffering_growinner():
    # Test that the inner loop grows when no buffering is needed
    a = np.arange(30)
    i = nditer(a, ['buffered', 'growinner', 'external_loop'],
                           buffersize=5)
    # Should end up with just one inner loop here
    assert_equal(i[0].size, a.size)


@pytest.mark.parametrize("read_or_readwrite", ["readonly", "readwrite"])
def test_iter_contig_flag_reduce_error(read_or_readwrite):
    # Test that a non-contiguous operand is rejected without buffering.
    # NOTE: This is true even for a reduction, where we return a 0-stride
    #       below!
    with pytest.raises(TypeError, match="Iterator operand required buffering"):
        it = np.nditer(
            (np.zeros(()),), flags=["external_loop", "reduce_ok"],
            op_flags=[(read_or_readwrite, "contig"),], itershape=(10,))


@pytest.mark.parametrize("arr", [
        lambda: np.zeros(()),
        lambda: np.zeros((20, 1))[::20],
        lambda: np.zeros((1, 20))[:, ::20]
    ])
def test_iter_contig_flag_single_operand_strides(arr):
    """
    Tests the strides with the contig flag for both broadcast and non-broadcast
    operands in 3 cases where the logic is needed:
    1. When everything has a zero stride, the broadcast op needs to repeated
    2. When the reduce axis is the last axis (first to iterate).
    3. When the reduce axis is the first axis (last to iterate).

    NOTE: The semantics of the cast flag are not clearly defined when
          it comes to reduction.  It is unclear that there are any users.
    """
    first_op = np.ones((10, 10))
    broadcast_op = arr()
    red_op = arr()
    # Add a first operand to ensure no axis-reordering and the result shape.
    iterator = np.nditer(
        (first_op, broadcast_op, red_op),
        flags=["external_loop", "reduce_ok", "buffered", "delay_bufalloc"],
        op_flags=[("readonly", "contig")] * 2 + [("readwrite", "contig")])

    with iterator:
        iterator.reset()
        for f, b, r in iterator:
            # The first operand is contigouos, we should have a view
            assert np.shares_memory(f, first_op)
            # Although broadcast, the second op always has a contiguous stride
            assert b.strides[0] == 8
            assert not np.shares_memory(b, broadcast_op)
            # The reduction has a contiguous stride or a 0 stride
            if red_op.ndim == 0 or red_op.shape[-1] == 1:
                assert r.strides[0] == 0
            else:
                # The stride is 8, although it was not originally:
                assert r.strides[0] == 8
            # If the reduce stride is 0, buffering makes no difference, but we
            # do it anyway right now:
            assert not np.shares_memory(r, red_op)


@pytest.mark.xfail(reason="The contig flag was always buggy.")
def test_iter_contig_flag_incorrect():
    # This case does the wrong thing...
    iterator = np.nditer(
        (np.ones((10, 10)).T, np.ones((1, 10))),
        flags=["external_loop", "reduce_ok", "buffered", "delay_bufalloc"],
        op_flags=[("readonly", "contig")] * 2)

    with iterator:
        iterator.reset()
        for a, b in iterator:
            # Remove a and b from locals (pytest may want to format them)
            a, b = a.strides, b.strides
            assert a == 8
            assert b == 8  # should be 8 but is 0 due to axis reorder


@pytest.mark.slow
def test_iter_buffered_reduce_reuse():
    # large enough array for all views, including negative strides.
    a = np.arange(2 * 3**5)[3**5:3**5 + 1]
    flags = ['buffered', 'delay_bufalloc', 'multi_index', 'reduce_ok', 'refs_ok']
    op_flags = [('readonly',), ('readwrite', 'allocate')]
    op_axes_list = [[(0, 1, 2), (0, 1, -1)], [(0, 1, 2), (0, -1, -1)]]
    # wrong dtype to force buffering
    op_dtypes = [float, a.dtype]

    def get_params():
        for xs in range(-3**2, 3**2 + 1):
            for ys in range(xs, 3**2 + 1):
                for op_axes in op_axes_list:
                    # last stride is reduced and because of that not
                    # important for this test, as it is the inner stride.
                    strides = (xs * a.itemsize, ys * a.itemsize, a.itemsize)
                    arr = np.lib.stride_tricks.as_strided(a, (3, 3, 3), strides)

                    for skip in [0, 1]:
                        yield arr, op_axes, skip

    for arr, op_axes, skip in get_params():
        nditer2 = np.nditer([arr.copy(), None],
                            op_axes=op_axes, flags=flags, op_flags=op_flags,
                            op_dtypes=op_dtypes)
        with nditer2:
            nditer2.operands[-1][...] = 0
            nditer2.reset()
            nditer2.iterindex = skip

            for (a2_in, b2_in) in nditer2:
                b2_in += a2_in.astype(np.int_)

            comp_res = nditer2.operands[-1]

        for bufsize in range(3**3):
            nditer1 = np.nditer([arr, None],
                                op_axes=op_axes, flags=flags, op_flags=op_flags,
                                buffersize=bufsize, op_dtypes=op_dtypes)
            with nditer1:
                nditer1.operands[-1][...] = 0
                nditer1.reset()
                nditer1.iterindex = skip

                for (a1_in, b1_in) in nditer1:
                    b1_in += a1_in.astype(np.int_)

                res = nditer1.operands[-1]
            assert_array_equal(res, comp_res)


def test_iter_buffered_reduce_reuse_core():
    # NumPy re-uses buffers for broadcast operands (as of writing when reading).
    # Test this even if the offset is manually set at some point during
    # the iteration.  (not a particularly tricky path)
    arr = np.empty((1, 6, 4, 1)).reshape(1, 6, 4, 1)[:, ::3, ::2, :]
    arr[...] = np.arange(arr.size).reshape(arr.shape)
    # First and last dimension are broadcast dimensions.
    arr = np.broadcast_to(arr, (100, 2, 2, 2))

    flags = ['buffered', 'reduce_ok', 'refs_ok', 'multi_index']
    op_flags = [('readonly',)]

    buffersize = 100  # small enough to not fit the whole array
    it = np.nditer(arr, flags=flags, op_flags=op_flags, buffersize=100)

    # Iterate a bit (this will cause buffering internally)
    expected = [next(it) for i in range(11)]
    # Now, manually advance to inside the core (the +1)
    it.iterindex = 10 * (2 * 2 * 2) + 1
    result = [next(it) for i in range(10)]

    assert expected[1:] == result


def test_iter_no_broadcast():
    # Test that the no_broadcast flag works
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(6).reshape(2, 3, 1)
    c = np.arange(12).reshape(3, 4)

    nditer([a, b, c], [],
           [['readonly', 'no_broadcast'],
            ['readonly'], ['readonly']])
    assert_raises(ValueError, nditer, [a, b, c], [],
                  [['readonly'], ['readonly', 'no_broadcast'], ['readonly']])
    assert_raises(ValueError, nditer, [a, b, c], [],
                  [['readonly'], ['readonly'], ['readonly', 'no_broadcast']])


class TestIterNested:

    def test_basic(self):
        # Test nested iteration basic usage
        a = arange(12).reshape(2, 3, 2)

        i, j = np.nested_iters(a, [[0], [1, 2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

        i, j = np.nested_iters(a, [[0, 1], [2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])

        i, j = np.nested_iters(a, [[0, 2], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])

    def test_reorder(self):
        # Test nested iteration basic usage
        a = arange(12).reshape(2, 3, 2)

        # In 'K' order (default), it gets reordered
        i, j = np.nested_iters(a, [[0], [2, 1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

        i, j = np.nested_iters(a, [[1, 0], [2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])

        i, j = np.nested_iters(a, [[2, 0], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])

        # In 'C' order, it doesn't
        i, j = np.nested_iters(a, [[0], [2, 1]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4, 1, 3, 5], [6, 8, 10, 7, 9, 11]])

        i, j = np.nested_iters(a, [[1, 0], [2]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [6, 7], [2, 3], [8, 9], [4, 5], [10, 11]])

        i, j = np.nested_iters(a, [[2, 0], [1]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [6, 8, 10], [1, 3, 5], [7, 9, 11]])

    def test_flip_axes(self):
        # Test nested iteration with negative axes
        a = arange(12).reshape(2, 3, 2)[::-1, ::-1, ::-1]

        # In 'K' order (default), the axes all get flipped
        i, j = np.nested_iters(a, [[0], [1, 2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

        i, j = np.nested_iters(a, [[0, 1], [2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])

        i, j = np.nested_iters(a, [[0, 2], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])

        # In 'C' order, flipping axes is disabled
        i, j = np.nested_iters(a, [[0], [1, 2]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[11, 10, 9, 8, 7, 6], [5, 4, 3, 2, 1, 0]])

        i, j = np.nested_iters(a, [[0, 1], [2]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[11, 10], [9, 8], [7, 6], [5, 4], [3, 2], [1, 0]])

        i, j = np.nested_iters(a, [[0, 2], [1]], order='C')
        vals = [list(j) for _ in i]
        assert_equal(vals, [[11, 9, 7], [10, 8, 6], [5, 3, 1], [4, 2, 0]])

    def test_broadcast(self):
        # Test nested iteration with broadcasting
        a = arange(2).reshape(2, 1)
        b = arange(3).reshape(1, 3)

        i, j = np.nested_iters([a, b], [[0], [1]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]]])

        i, j = np.nested_iters([a, b], [[1], [0]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[[0, 0], [1, 0]], [[0, 1], [1, 1]], [[0, 2], [1, 2]]])

    def test_dtype_copy(self):
        # Test nested iteration with a copy to change dtype

        # copy
        a = arange(6, dtype='i4').reshape(2, 3)
        i, j = np.nested_iters(a, [[0], [1]],
                            op_flags=['readonly', 'copy'],
                            op_dtypes='f8')
        assert_equal(j[0].dtype, np.dtype('f8'))
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2], [3, 4, 5]])
        vals = None

        # writebackifcopy - using context manager
        a = arange(6, dtype='f4').reshape(2, 3)
        i, j = np.nested_iters(a, [[0], [1]],
                            op_flags=['readwrite', 'updateifcopy'],
                            casting='same_kind',
                            op_dtypes='f8')
        with i, j:
            assert_equal(j[0].dtype, np.dtype('f8'))
            for x in i:
                for y in j:
                    y[...] += 1
            assert_equal(a, [[0, 1, 2], [3, 4, 5]])
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

        # writebackifcopy - using close()
        a = arange(6, dtype='f4').reshape(2, 3)
        i, j = np.nested_iters(a, [[0], [1]],
                            op_flags=['readwrite', 'updateifcopy'],
                            casting='same_kind',
                            op_dtypes='f8')
        assert_equal(j[0].dtype, np.dtype('f8'))
        for x in i:
            for y in j:
                y[...] += 1
        assert_equal(a, [[0, 1, 2], [3, 4, 5]])
        i.close()
        j.close()
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

    def test_dtype_buffered(self):
        # Test nested iteration with buffering to change dtype

        a = arange(6, dtype='f4').reshape(2, 3)
        i, j = np.nested_iters(a, [[0], [1]],
                            flags=['buffered'],
                            op_flags=['readwrite'],
                            casting='same_kind',
                            op_dtypes='f8')
        assert_equal(j[0].dtype, np.dtype('f8'))
        for x in i:
            for y in j:
                y[...] += 1
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

    def test_0d(self):
        a = np.arange(12).reshape(2, 3, 2)
        i, j = np.nested_iters(a, [[], [1, 0, 2]])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

        i, j = np.nested_iters(a, [[1, 0, 2], []])
        vals = [list(j) for _ in i]
        assert_equal(vals, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]])

        i, j, k = np.nested_iters(a, [[2, 0], [], [1]])
        vals = []
        for x in i:
            for y in j:
                vals.append(list(k))
        assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])

    def test_iter_nested_iters_dtype_buffered(self):
        # Test nested iteration with buffering to change dtype

        a = arange(6, dtype='f4').reshape(2, 3)
        i, j = np.nested_iters(a, [[0], [1]],
                            flags=['buffered'],
                            op_flags=['readwrite'],
                            casting='same_kind',
                            op_dtypes='f8')
        with i, j:
            assert_equal(j[0].dtype, np.dtype('f8'))
            for x in i:
                for y in j:
                    y[...] += 1
        assert_equal(a, [[1, 2, 3], [4, 5, 6]])

def test_iter_reduction_error():

    a = np.arange(6)
    assert_raises(ValueError, nditer, [a, None], [],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[[0], [-1]])

    a = np.arange(6).reshape(2, 3)
    assert_raises(ValueError, nditer, [a, None], ['external_loop'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[[0, 1], [-1, -1]])

def test_iter_reduction():
    # Test doing reductions with the iterator

    a = np.arange(6)
    i = nditer([a, None], ['reduce_ok'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[[0], [-1]])
    # Need to initialize the output operand to the addition unit
    with i:
        i.operands[1][...] = 0
        # Do the reduction
        for x, y in i:
            y[...] += x
        # Since no axes were specified, should have allocated a scalar
        assert_equal(i.operands[1].ndim, 0)
        assert_equal(i.operands[1], np.sum(a))

    a = np.arange(6).reshape(2, 3)
    i = nditer([a, None], ['reduce_ok', 'external_loop'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[[0, 1], [-1, -1]])
    # Need to initialize the output operand to the addition unit
    with i:
        i.operands[1][...] = 0
        # Reduction shape/strides for the output
        assert_equal(i[1].shape, (6,))
        assert_equal(i[1].strides, (0,))
        # Do the reduction
        for x, y in i:
            # Use a for loop instead of ``y[...] += x``
            # (equivalent to ``y[...] = y[...].copy() + x``),
            # because y has zero strides we use for the reduction
            for j in range(len(y)):
                y[j] += x[j]
        # Since no axes were specified, should have allocated a scalar
        assert_equal(i.operands[1].ndim, 0)
        assert_equal(i.operands[1], np.sum(a))

    # This is a tricky reduction case for the buffering double loop
    # to handle
    a = np.ones((2, 3, 5))
    it1 = nditer([a, None], ['reduce_ok', 'external_loop'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[None, [0, -1, 1]])
    it2 = nditer([a, None], ['reduce_ok', 'external_loop',
                            'buffered', 'delay_bufalloc'],
                    [['readonly'], ['readwrite', 'allocate']],
                    op_axes=[None, [0, -1, 1]], buffersize=10)
    with it1, it2:
        it1.operands[1].fill(0)
        it2.operands[1].fill(0)
        it2.reset()
        for x in it1:
            x[1][...] += x[0]
        for x in it2:
            x[1][...] += x[0]
        assert_equal(it1.operands[1], it2.operands[1])
        assert_equal(it2.operands[1].sum(), a.size)

def test_iter_buffering_reduction():
    # Test doing buffered reductions with the iterator

    a = np.arange(6)
    b = np.array(0., dtype='f8').byteswap()
    b = b.view(b.dtype.newbyteorder())
    i = nditer([a, b], ['reduce_ok', 'buffered'],
                    [['readonly'], ['readwrite', 'nbo']],
                    op_axes=[[0], [-1]])
    with i:
        assert_equal(i[1].dtype, np.dtype('f8'))
        assert_(i[1].dtype != b.dtype)
        # Do the reduction
        for x, y in i:
            y[...] += x
    # Since no axes were specified, should have allocated a scalar
    assert_equal(b, np.sum(a))

    a = np.arange(6).reshape(2, 3)
    b = np.array([0, 0], dtype='f8').byteswap()
    b = b.view(b.dtype.newbyteorder())
    i = nditer([a, b], ['reduce_ok', 'external_loop', 'buffered'],
                    [['readonly'], ['readwrite', 'nbo']],
                    op_axes=[[0, 1], [0, -1]])
    # Reduction shape/strides for the output
    with i:
        assert_equal(i[1].shape, (3,))
        assert_equal(i[1].strides, (0,))
        # Do the reduction
        for x, y in i:
            # Use a for loop instead of ``y[...] += x``
            # (equivalent to ``y[...] = y[...].copy() + x``),
            # because y has zero strides we use for the reduction
            for j in range(len(y)):
                y[j] += x[j]
    assert_equal(b, np.sum(a, axis=1))

    # Iterator inner double loop was wrong on this one
    p = np.arange(2) + 1
    it = np.nditer([p, None],
            ['delay_bufalloc', 'reduce_ok', 'buffered', 'external_loop'],
            [['readonly'], ['readwrite', 'allocate']],
            op_axes=[[-1, 0], [-1, -1]],
            itershape=(2, 2))
    with it:
        it.operands[1].fill(0)
        it.reset()
        assert_equal(it[0], [1, 2, 1, 2])

    # Iterator inner loop should take argument contiguity into account
    x = np.ones((7, 13, 8), np.int8)[4:6, 1:11:6, 1:5].transpose(1, 2, 0)
    x[...] = np.arange(x.size).reshape(x.shape)
    y_base = np.arange(4 * 4, dtype=np.int8).reshape(4, 4)
    y_base_copy = y_base.copy()
    y = y_base[::2, :, None]

    it = np.nditer([y, x],
                   ['buffered', 'external_loop', 'reduce_ok'],
                   [['readwrite'], ['readonly']])
    with it:
        for a, b in it:
            a.fill(2)

    assert_equal(y_base[1::2], y_base_copy[1::2])
    assert_equal(y_base[::2], 2)

def test_iter_buffering_reduction_reuse_reduce_loops():
    # There was a bug triggering reuse of the reduce loop inappropriately,
    # which caused processing to happen in unnecessarily small chunks
    # and overran the buffer.

    a = np.zeros((2, 7))
    b = np.zeros((1, 7))
    it = np.nditer([a, b], flags=['reduce_ok', 'external_loop', 'buffered'],
                    op_flags=[['readonly'], ['readwrite']],
                    buffersize=5)

    with it:
        bufsizes = [x.shape[0] for x, y in it]
    assert_equal(bufsizes, [5, 2, 5, 2])
    assert_equal(sum(bufsizes), a.size)

def test_iter_writemasked_badinput():
    a = np.zeros((2, 3))
    b = np.zeros((3,))
    m = np.array([[True, True, False], [False, True, False]])
    m2 = np.array([True, True, False])
    m3 = np.array([0, 1, 1], dtype='u1')
    mbad1 = np.array([0, 1, 1], dtype='i1')
    mbad2 = np.array([0, 1, 1], dtype='f4')

    # Need an 'arraymask' if any operand is 'writemasked'
    assert_raises(ValueError, nditer, [a, m], [],
                    [['readwrite', 'writemasked'], ['readonly']])

    # A 'writemasked' operand must not be readonly
    assert_raises(ValueError, nditer, [a, m], [],
                    [['readonly', 'writemasked'], ['readonly', 'arraymask']])

    # 'writemasked' and 'arraymask' may not be used together
    assert_raises(ValueError, nditer, [a, m], [],
                    [['readonly'], ['readwrite', 'arraymask', 'writemasked']])

    # 'arraymask' may only be specified once
    assert_raises(ValueError, nditer, [a, m, m2], [],
                    [['readwrite', 'writemasked'],
                     ['readonly', 'arraymask'],
                     ['readonly', 'arraymask']])

    # An 'arraymask' with nothing 'writemasked' also doesn't make sense
    assert_raises(ValueError, nditer, [a, m], [],
                    [['readwrite'], ['readonly', 'arraymask']])

    # A writemasked reduction requires a similarly smaller mask
    assert_raises(ValueError, nditer, [a, b, m], ['reduce_ok'],
                    [['readonly'],
                     ['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']])
    # But this should work with a smaller/equal mask to the reduction operand
    np.nditer([a, b, m2], ['reduce_ok'],
                    [['readonly'],
                     ['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']])
    # The arraymask itself cannot be a reduction
    assert_raises(ValueError, nditer, [a, b, m2], ['reduce_ok'],
                    [['readonly'],
                     ['readwrite', 'writemasked'],
                     ['readwrite', 'arraymask']])

    # A uint8 mask is ok too
    np.nditer([a, m3], ['buffered'],
                    [['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']],
                    op_dtypes=['f4', None],
                    casting='same_kind')
    # An int8 mask isn't ok
    assert_raises(TypeError, np.nditer, [a, mbad1], ['buffered'],
                    [['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']],
                    op_dtypes=['f4', None],
                    casting='same_kind')
    # A float32 mask isn't ok
    assert_raises(TypeError, np.nditer, [a, mbad2], ['buffered'],
                    [['readwrite', 'writemasked'],
                     ['readonly', 'arraymask']],
                    op_dtypes=['f4', None],
                    casting='same_kind')


def _is_buffered(iterator):
    try:
        iterator.itviews
    except ValueError:
        return True
    return False

@pytest.mark.parametrize("a",
        [np.zeros((3,), dtype='f8'),
         np.zeros((9876, 3 * 5), dtype='f8')[::2, :],
         np.zeros((4, 312, 124, 3), dtype='f8')[::2, :, ::2, :],
         # Also test with the last dimension strided (so it does not fit if
         # there is repeated access)
         np.zeros((9,), dtype='f8')[::3],
         np.zeros((9876, 3 * 10), dtype='f8')[::2, ::5],
         np.zeros((4, 312, 124, 3), dtype='f8')[::2, :, ::2, ::-1]])
def test_iter_writemasked(a):
    # Note, the slicing above is to ensure that nditer cannot combine multiple
    # axes into one.  The repetition is just to make things a bit more
    # interesting.
    shape = a.shape
    reps = shape[-1] // 3
    msk = np.empty(shape, dtype=bool)
    msk[...] = [True, True, False] * reps

    # When buffering is unused, 'writemasked' effectively does nothing.
    # It's up to the user of the iterator to obey the requested semantics.
    it = np.nditer([a, msk], [],
                [['readwrite', 'writemasked'],
                 ['readonly', 'arraymask']])
    with it:
        for x, m in it:
            x[...] = 1
    # Because we violated the semantics, all the values became 1
    assert_equal(a, np.broadcast_to([1, 1, 1] * reps, shape))

    # Even if buffering is enabled, we still may be accessing the array
    # directly.
    it = np.nditer([a, msk], ['buffered'],
                [['readwrite', 'writemasked'],
                 ['readonly', 'arraymask']])
    # @seberg: I honestly don't currently understand why a "buffered" iterator
    # would end up not using a buffer for the small array here at least when
    # "writemasked" is used, that seems confusing...  Check by testing for
    # actual memory overlap!
    is_buffered = True
    with it:
        for x, m in it:
            x[...] = 2.5
            if np.may_share_memory(x, a):
                is_buffered = False

    if not is_buffered:
        # Because we violated the semantics, all the values became 2.5
        assert_equal(a, np.broadcast_to([2.5, 2.5, 2.5] * reps, shape))
    else:
        # For large sizes, the iterator may be buffered:
        assert_equal(a, np.broadcast_to([2.5, 2.5, 1] * reps, shape))
        a[...] = 2.5

    # If buffering will definitely happening, for instance because of
    # a cast, only the items selected by the mask will be copied back from
    # the buffer.
    it = np.nditer([a, msk], ['buffered'],
                [['readwrite', 'writemasked'],
                 ['readonly', 'arraymask']],
                op_dtypes=['i8', None],
                casting='unsafe')
    with it:
        for x, m in it:
            x[...] = 3
    # Even though we violated the semantics, only the selected values
    # were copied back
    assert_equal(a, np.broadcast_to([3, 3, 2.5] * reps, shape))


@pytest.mark.parametrize(["mask", "mask_axes"], [
        # Allocated operand (only broadcasts with -1)
        (None, [-1, 0]),
        # Reduction along the first dimension (with and without op_axes)
        (np.zeros((1, 4), dtype="bool"), [0, 1]),
        (np.zeros((1, 4), dtype="bool"), None),
        # Test 0-D and -1 op_axes
        (np.zeros(4, dtype="bool"), [-1, 0]),
        (np.zeros((), dtype="bool"), [-1, -1]),
        (np.zeros((), dtype="bool"), None)])
def test_iter_writemasked_broadcast_error(mask, mask_axes):
    # This assumes that a readwrite mask makes sense. This is likely not the
    # case and should simply be deprecated.
    arr = np.zeros((3, 4))
    itflags = ["reduce_ok"]
    mask_flags = ["arraymask", "readwrite", "allocate"]
    a_flags = ["writeonly", "writemasked"]
    if mask_axes is None:
        op_axes = None
    else:
        op_axes = [mask_axes, [0, 1]]

    with assert_raises(ValueError):
        np.nditer((mask, arr), flags=itflags, op_flags=[mask_flags, a_flags],
                  op_axes=op_axes)


def test_iter_writemasked_decref():
    # force casting (to make it interesting) by using a structured dtype.
    arr = np.arange(10000).astype(">i,O")
    original = arr.copy()
    mask = np.random.randint(0, 2, size=10000).astype(bool)

    it = np.nditer([arr, mask], ['buffered', "refs_ok"],
                   [['readwrite', 'writemasked'],
                    ['readonly', 'arraymask']],
                   op_dtypes=["<i,O", "?"])
    singleton = object()
    if HAS_REFCOUNT:
        count = sys.getrefcount(singleton)
    for buf, mask_buf in it:
        buf[...] = (3, singleton)

    del buf, mask_buf, it   # delete everything to ensure correct cleanup

    if HAS_REFCOUNT:
        # The buffer would have included additional items, they must be
        # cleared correctly:
        assert sys.getrefcount(singleton) - count == np.count_nonzero(mask)

    assert_array_equal(arr[~mask], original[~mask])
    assert (arr[mask] == np.array((3, singleton), arr.dtype)).all()
    del arr

    if HAS_REFCOUNT:
        assert sys.getrefcount(singleton) == count


def test_iter_non_writable_attribute_deletion():
    it = np.nditer(np.ones(2))
    attr = ["value", "shape", "operands", "itviews", "has_delayed_bufalloc",
            "iterationneedsapi", "has_multi_index", "has_index", "dtypes",
            "ndim", "nop", "itersize", "finished"]

    for s in attr:
        assert_raises(AttributeError, delattr, it, s)


def test_iter_writable_attribute_deletion():
    it = np.nditer(np.ones(2))
    attr = ["multi_index", "index", "iterrange", "iterindex"]
    for s in attr:
        assert_raises(AttributeError, delattr, it, s)


def test_iter_element_deletion():
    it = np.nditer(np.ones(3))
    try:
        del it[1]
        del it[1:2]
    except TypeError:
        pass
    except Exception:
        raise AssertionError

def test_iter_allocated_array_dtypes():
    # If the dtype of an allocated output has a shape, the shape gets
    # tacked onto the end of the result.
    it = np.nditer(([1, 3, 20], None), op_dtypes=[None, ('i4', (2,))])
    for a, b in it:
        b[0] = a - 1
        b[1] = a + 1
    assert_equal(it.operands[1], [[0, 2], [2, 4], [19, 21]])

    # Check the same (less sensitive) thing when `op_axes` with -1 is given.
    it = np.nditer(([[1, 3, 20]], None), op_dtypes=[None, ('i4', (2,))],
                   flags=["reduce_ok"], op_axes=[None, (-1, 0)])
    for a, b in it:
        b[0] = a - 1
        b[1] = a + 1
    assert_equal(it.operands[1], [[0, 2], [2, 4], [19, 21]])

    # Make sure this works for scalars too
    it = np.nditer((10, 2, None), op_dtypes=[None, None, ('i4', (2, 2))])
    for a, b, c in it:
        c[0, 0] = a - b
        c[0, 1] = a + b
        c[1, 0] = a * b
        c[1, 1] = a / b
    assert_equal(it.operands[2], [[8, 12], [20, 5]])


def test_0d_iter():
    # Basic test for iteration of 0-d arrays:
    i = nditer([2, 3], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.ndim, 0)
    assert_equal(next(i), (2, 3))
    assert_equal(i.multi_index, ())
    assert_equal(i.iterindex, 0)
    assert_raises(StopIteration, next, i)
    # test reset:
    i.reset()
    assert_equal(next(i), (2, 3))
    assert_raises(StopIteration, next, i)

    # test forcing to 0-d
    i = nditer(np.arange(5), ['multi_index'], [['readonly']], op_axes=[()])
    assert_equal(i.ndim, 0)
    assert_equal(len(i), 1)

    i = nditer(np.arange(5), ['multi_index'], [['readonly']],
               op_axes=[()], itershape=())
    assert_equal(i.ndim, 0)
    assert_equal(len(i), 1)

    # passing an itershape alone is not enough, the op_axes are also needed
    with assert_raises(ValueError):
        nditer(np.arange(5), ['multi_index'], [['readonly']], itershape=())

    # Test a more complex buffered casting case (same as another test above)
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.array(0.5, dtype='f4')
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'],
                    casting='unsafe', op_dtypes=sdt)
    vals = next(i)
    assert_equal(vals['a'], 0.5)
    assert_equal(vals['b'], 0)
    assert_equal(vals['c'], [[(0.5)] * 3] * 2)
    assert_equal(vals['d'], 0.5)

def test_object_iter_cleanup():
    # see gh-18450
    # object arrays can raise a python exception in ufunc inner loops using
    # nditer, which should cause iteration to stop & cleanup. There were bugs
    # in the nditer cleanup when decref'ing object arrays.
    # This test would trigger valgrind "uninitialized read" before the bugfix.
    assert_raises(TypeError, lambda: np.zeros((17000, 2), dtype='f4') * None)

    # this more explicit code also triggers the invalid access
    arr = np.arange(ncu.BUFSIZE * 10).reshape(10, -1).astype(str)
    oarr = arr.astype(object)
    oarr[:, -1] = None
    assert_raises(TypeError, lambda: np.add(oarr[:, ::-1], arr[:, ::-1]))

    # followup: this tests for a bug introduced in the first pass of gh-18450,
    # caused by an incorrect fallthrough of the TypeError
    class T:
        def __bool__(self):
            raise TypeError("Ambiguous")
    assert_raises(TypeError, np.logical_or.reduce,
                             np.array([T(), T()], dtype='O'))

def test_object_iter_cleanup_reduce():
    # Similar as above, but a complex reduction case that was previously
    # missed (see gh-18810).
    # The following array is special in that it cannot be flattened:
    arr = np.array([[None, 1], [-1, -1], [None, 2], [-1, -1]])[::2]
    with pytest.raises(TypeError):
        np.sum(arr)

@pytest.mark.parametrize("arr", [
        np.ones((8000, 4, 2), dtype=object)[:, ::2, :],
        np.ones((8000, 4, 2), dtype=object, order="F")[:, ::2, :],
        np.ones((8000, 4, 2), dtype=object)[:, ::2, :].copy("F")])
def test_object_iter_cleanup_large_reduce(arr):
    # More complicated calls are possible for large arrays:
    out = np.ones(8000, dtype=np.intp)
    # force casting with `dtype=object`
    res = np.sum(arr, axis=(1, 2), dtype=object, out=out)
    assert_array_equal(res, np.full(8000, 4, dtype=object))

def test_iter_too_large():
    # The total size of the iterator must not exceed the maximum intp due
    # to broadcasting. Dividing by 1024 will keep it small enough to
    # give a legal array.
    size = np.iinfo(np.intp).max // 1024
    arr = np.lib.stride_tricks.as_strided(np.zeros(1), (size,), (0,))
    assert_raises(ValueError, nditer, (arr, arr[:, None]))
    # test the same for multiindex. That may get more interesting when
    # removing 0 dimensional axis is allowed (since an iterator can grow then)
    assert_raises(ValueError, nditer,
                  (arr, arr[:, None]), flags=['multi_index'])


def test_iter_too_large_with_multiindex():
    # When a multi index is being tracked, the error is delayed this
    # checks the delayed error messages and getting below that by
    # removing an axis.
    base_size = 2**10
    num = 1
    while base_size**num < np.iinfo(np.intp).max:
        num += 1

    shape_template = [1, 1] * num
    arrays = []
    for i in range(num):
        shape = shape_template[:]
        shape[i * 2] = 2**10
        arrays.append(np.empty(shape))
    arrays = tuple(arrays)

    # arrays are now too large to be broadcast. The different modes test
    # different nditer functionality with or without GIL.
    for mode in range(6):
        with assert_raises(ValueError):
            _multiarray_tests.test_nditer_too_large(arrays, -1, mode)
    # but if we do nothing with the nditer, it can be constructed:
    _multiarray_tests.test_nditer_too_large(arrays, -1, 7)

    # When an axis is removed, things should work again (half the time):
    for i in range(num):
        for mode in range(6):
            # an axis with size 1024 is removed:
            _multiarray_tests.test_nditer_too_large(arrays, i * 2, mode)
            # an axis with size 1 is removed:
            with assert_raises(ValueError):
                _multiarray_tests.test_nditer_too_large(arrays, i * 2 + 1, mode)

def test_writebacks():
    a = np.arange(6, dtype='f4')
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    assert_(a.dtype.byteorder != au.dtype.byteorder)
    it = nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        it.operands[0][:] = 100
    assert_equal(au, 100)
    # do it again, this time raise an error,
    it = nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv', op_dtypes=[np.dtype('f4')])
    try:
        with it:
            assert_equal(au.flags.writeable, False)
            it.operands[0][:] = 0
            raise ValueError('exit context manager on exception')
    except Exception:
        pass
    assert_equal(au, 0)
    assert_equal(au.flags.writeable, True)
    # cannot reuse i outside context manager
    assert_raises(ValueError, getattr, it, 'operands')

    it = nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv', op_dtypes=[np.dtype('f4')])
    with it:
        x = it.operands[0]
        x[:] = 6
        assert_(x.flags.writebackifcopy)
    assert_equal(au, 6)
    assert_(not x.flags.writebackifcopy)
    x[:] = 123  # x.data still valid
    assert_equal(au, 6)  # but not connected to au

    it = nditer(au, [],
                 [['readwrite', 'updateifcopy']],
                 casting='equiv', op_dtypes=[np.dtype('f4')])
    # reentering works
    with it:
        with it:
            for x in it:
                x[...] = 123

    it = nditer(au, [],
                 [['readwrite', 'updateifcopy']],
                 casting='equiv', op_dtypes=[np.dtype('f4')])
    # make sure exiting the inner context manager closes the iterator
    with it:
        with it:
            for x in it:
                x[...] = 123
        assert_raises(ValueError, getattr, it, 'operands')
    # do not crash if original data array is decrefed
    it = nditer(au, [],
                 [['readwrite', 'updateifcopy']],
                 casting='equiv', op_dtypes=[np.dtype('f4')])
    del au
    with it:
        for x in it:
            x[...] = 123
    # make sure we cannot reenter the closed iterator
    enter = it.__enter__
    assert_raises(RuntimeError, enter)

def test_close_equivalent():
    ''' using a context amanger and using nditer.close are equivalent
    '''
    def add_close(x, y, out=None):
        addop = np.add
        it = np.nditer([x, y, out], [],
                    [['readonly'], ['readonly'], ['writeonly', 'allocate']])
        for (a, b, c) in it:
            addop(a, b, out=c)
        ret = it.operands[2]
        it.close()
        return ret

    def add_context(x, y, out=None):
        addop = np.add
        it = np.nditer([x, y, out], [],
                    [['readonly'], ['readonly'], ['writeonly', 'allocate']])
        with it:
            for (a, b, c) in it:
                addop(a, b, out=c)
            return it.operands[2]
    z = add_close(range(5), range(5))
    assert_equal(z, range(0, 10, 2))
    z = add_context(range(5), range(5))
    assert_equal(z, range(0, 10, 2))

def test_close_raises():
    it = np.nditer(np.arange(3))
    assert_equal(next(it), 0)
    it.close()
    assert_raises(StopIteration, next, it)
    assert_raises(ValueError, getattr, it, 'operands')

def test_close_parameters():
    it = np.nditer(np.arange(3))
    assert_raises(TypeError, it.close, 1)

@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_warn_noclose():
    a = np.arange(6, dtype='f4')
    au = a.byteswap()
    au = au.view(au.dtype.newbyteorder())
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning)
        it = np.nditer(au, [], [['readwrite', 'updateifcopy']],
                        casting='equiv', op_dtypes=[np.dtype('f4')])
        del it
        assert len(sup.log) == 1


@pytest.mark.parametrize(["in_dtype", "buf_dtype"],
        [("i", "O"), ("O", "i"),  # most simple cases
         ("i,O", "O,O"),  # structured partially only copying O
         ("O,i", "i,O"),  # structured casting to and from O
         ])
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_partial_iteration_cleanup(in_dtype, buf_dtype, steps):
    """
    Checks for reference counting leaks during cleanup.  Using explicit
    reference counts lead to occasional false positives (at least in parallel
    test setups).  This test now should still test leaks correctly when
    run e.g. with pytest-valgrind or pytest-leaks
    """
    value = 2**30 + 1  # just a random value that Python won't intern
    arr = np.full(int(ncu.BUFSIZE * 2.5), value).astype(in_dtype)

    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)],
            flags=["buffered", "external_loop", "refs_ok"], casting="unsafe")
    for step in range(steps):
        # The iteration finishes in 3 steps, the first two are partial
        next(it)

    del it  # not necessary, but we test the cleanup

    # Repeat the test with `iternext`
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)],
                   flags=["buffered", "external_loop", "refs_ok"], casting="unsafe")
    for step in range(steps):
        it.iternext()

    del it  # not necessary, but we test the cleanup

@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
@pytest.mark.parametrize(["in_dtype", "buf_dtype"],
         [("O", "i"),  # most simple cases
          ("O,i", "i,O"),  # structured casting to and from O
          ])
def test_partial_iteration_error(in_dtype, buf_dtype):
    value = 123  # relies on python cache (leak-check will still find it)
    arr = np.full(int(ncu.BUFSIZE * 2.5), value).astype(in_dtype)
    if in_dtype == "O":
        arr[int(ncu.BUFSIZE * 1.5)] = None
    else:
        arr[int(ncu.BUFSIZE * 1.5)]["f0"] = None

    count = sys.getrefcount(value)

    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)],
            flags=["buffered", "external_loop", "refs_ok"], casting="unsafe")
    with pytest.raises(TypeError):
        # pytest.raises seems to have issues with the error originating
        # in the for loop, so manually unravel:
        next(it)
        next(it)  # raises TypeError

    # Repeat the test with `iternext` after resetting, the buffers should
    # already be cleared from any references, so resetting is sufficient.
    it.reset()
    with pytest.raises(TypeError):
        it.iternext()
        it.iternext()

    assert count == sys.getrefcount(value)


def test_arbitrary_number_of_ops():
    # 2*16 + 1 is still just a few kiB, so should be fast and easy to deal with
    # but larger than any small custom integer.
    ops = [np.arange(10) for a in range(2**16 + 1)]

    it = np.nditer(ops)
    for i, vals in enumerate(it):
        assert all(v == i for v in vals)


def test_arbitrary_number_of_ops_nested():
    # 2*16 + 1 is still just a few kiB, so should be fast and easy to deal with
    # but larger than any small custom integer.
    ops = [np.arange(10) for a in range(2**16 + 1)]

    it = np.nested_iters(ops, [[0], []])
    for i, vals in enumerate(it):
        assert all(v == i for v in vals)


@pytest.mark.slow
@requires_memory(9 * np.iinfo(np.intc).max)
def test_arbitrary_number_of_ops_error():
    # A different error may happen for more than integer operands, but that
    # is too large to test nicely.
    a = np.ones(1)
    args = [a] * (np.iinfo(np.intc).max + 1)
    with pytest.raises(ValueError, match="Too many operands to nditer"):
        np.nditer(args)

    with pytest.raises(ValueError, match="Too many operands to nditer"):
        np.nested_iters(args, [[0], []])


def test_debug_print(capfd):
    """
    Matches the expected output of a debug print with the actual output.
    Note that the iterator dump should not be considered stable API,
    this test is mainly to ensure the print does not crash.

    Currently uses a subprocess to avoid dealing with the C level `printf`s.
    """
    # the expected output with all addresses and sizes stripped (they vary
    # and/or are platform dependent).
    expected = """
    ------ BEGIN ITERATOR DUMP ------
    | Iterator Address:
    | ItFlags: BUFFER REDUCE
    | NDim: 2
    | NOp: 2
    | IterSize: 50
    | IterStart: 0
    | IterEnd: 50
    | IterIndex: 0
    | Iterator SizeOf:
    | BufferData SizeOf:
    | AxisData SizeOf:
    |
    | Perm: 0 1
    | DTypes:
    | DTypes: dtype('float64') dtype('int32')
    | InitDataPtrs:
    | BaseOffsets: 0 0
    | Ptrs:
    | User/buffer ptrs:
    | Operands:
    | Operand DTypes: dtype('int64') dtype('float64')
    | OpItFlags:
    |   Flags[0]: READ CAST
    |   Flags[1]: READ WRITE CAST REDUCE
    |
    | BufferData:
    |   BufferSize: 50
    |   Size: 5
    |   BufIterEnd: 5
    |   BUFFER CoreSize: 5
    |   REDUCE Pos: 0
    |   REDUCE OuterSize: 10
    |   REDUCE OuterDim: 1
    |   Strides: 8 4
    |   REDUCE Outer Strides: 40 0
    |   REDUCE Outer Ptrs:
    |   ReadTransferFn:
    |   ReadTransferData:
    |   WriteTransferFn:
    |   WriteTransferData:
    |   Buffers:
    |
    | AxisData[0]:
    |   Shape: 5
    |   Index: 0
    |   Strides: 16 8
    | AxisData[1]:
    |   Shape: 10
    |   Index: 0
    |   Strides: 80 0
    ------- END ITERATOR DUMP -------
    """.strip().splitlines()

    arr1 = np.arange(100, dtype=np.int64).reshape(10, 10)[:, ::2]
    arr2 = np.arange(5.)
    it = np.nditer((arr1, arr2), op_dtypes=["d", "i4"], casting="unsafe",
                   flags=["reduce_ok", "buffered"],
                   op_flags=[["readonly"], ["readwrite"]])
    it.debug_print()
    res = capfd.readouterr().out
    res = res.strip().splitlines()

    assert len(res) == len(expected)
    for res_line, expected_line in zip(res, expected):
        # The actual output may have additional pointers listed that are
        # stripped from the example output:
        assert res_line.startswith(expected_line.strip())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\registry.py ===
from .operators.activation import QDQRemovableActivation, QLinearActivation
from .operators.argmax import QArgMax
from .operators.attention import AttentionQuant
from .operators.base_operator import QuantOperatorBase
from .operators.binary_op import QLinearBinaryOp
from .operators.concat import QLinearConcat
from .operators.conv import ConvInteger, QDQConv, QLinearConv
from .operators.direct_q8 import Direct8BitOp, QDQDirect8BitOp
from .operators.embed_layernorm import EmbedLayerNormalizationQuant
from .operators.gather import GatherQuant, QDQGather
from .operators.gavgpool import QGlobalAveragePool
from .operators.gemm import QDQGemm, QLinearGemm
from .operators.lstm import LSTMQuant
from .operators.matmul import MatMulInteger, QDQMatMul, QLinearMatMul
from .operators.maxpool import QDQMaxPool, QMaxPool
from .operators.norm import QDQNormalization
from .operators.pad import QDQPad, QPad
from .operators.pooling import QLinearPool
from .operators.qdq_base_operator import QDQOperatorBase
from .operators.resize import QDQResize, QResize
from .operators.softmax import QLinearSoftmax
from .operators.split import QDQSplit, QSplit
from .operators.where import QDQWhere, QLinearWhere
from .quant_utils import QuantizationMode

CommonOpsRegistry = {
    "Gather": GatherQuant,
    "Transpose": Direct8BitOp,
    "EmbedLayerNormalization": EmbedLayerNormalizationQuant,
}

IntegerOpsRegistry = {
    "Conv": ConvInteger,
    "MatMul": MatMulInteger,
    "Attention": AttentionQuant,
    "LSTM": LSTMQuant,
}
IntegerOpsRegistry.update(CommonOpsRegistry)

QLinearOpsRegistry = {
    "ArgMax": QArgMax,
    "Conv": QLinearConv,
    "Gemm": QLinearGemm,
    "MatMul": QLinearMatMul,
    "Add": QLinearBinaryOp,
    "Mul": QLinearBinaryOp,
    "Relu": QLinearActivation,
    "Clip": QLinearActivation,
    "LeakyRelu": QLinearActivation,
    "Sigmoid": QLinearActivation,
    "MaxPool": QMaxPool,
    "GlobalAveragePool": QGlobalAveragePool,
    "Split": QSplit,
    "Pad": QPad,
    "Reshape": Direct8BitOp,
    "Squeeze": Direct8BitOp,
    "Unsqueeze": Direct8BitOp,
    "Resize": QResize,
    "AveragePool": QLinearPool,
    "Concat": QLinearConcat,
    "Softmax": QLinearSoftmax,
    "Where": QLinearWhere,
}
QLinearOpsRegistry.update(CommonOpsRegistry)

QDQRegistry = {
    "Conv": QDQConv,
    "ConvTranspose": QDQConv,
    "Gemm": QDQGemm,
    "Clip": QDQRemovableActivation,
    "Relu": QDQRemovableActivation,
    "Reshape": QDQDirect8BitOp,
    "Transpose": QDQDirect8BitOp,
    "Squeeze": QDQDirect8BitOp,
    "Unsqueeze": QDQDirect8BitOp,
    "Resize": QDQResize,
    "MaxPool": QDQMaxPool,
    "AveragePool": QDQDirect8BitOp,
    "Slice": QDQDirect8BitOp,
    "Pad": QDQPad,
    "MatMul": QDQMatMul,
    "Split": QDQSplit,
    "Gather": QDQGather,
    "GatherElements": QDQGather,
    "Where": QDQWhere,
    "InstanceNormalization": QDQNormalization,
    "LayerNormalization": QDQNormalization,
    "BatchNormalization": QDQNormalization,
}


def CreateDefaultOpQuantizer(onnx_quantizer, node):  # noqa: N802
    return QuantOperatorBase(onnx_quantizer, node)


def CreateOpQuantizer(onnx_quantizer, node):  # noqa: N802
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps else QLinearOpsRegistry
    if node.op_type in registry:
        op_quantizer = registry[node.op_type](onnx_quantizer, node)
        if op_quantizer.should_quantize():
            return op_quantizer
    return QuantOperatorBase(onnx_quantizer, node)


def CreateQDQQuantizer(onnx_quantizer, node):  # noqa: N802
    if node.op_type in QDQRegistry:
        return QDQRegistry[node.op_type](onnx_quantizer, node)
    return QDQOperatorBase(onnx_quantizer, node)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\llama\quant_kv_dataloader.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import argparse

import numpy as np
import torch
from benchmark_helper import create_onnxruntime_session
from datasets import load_dataset
from llama_inputs import get_position_ids
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer


class QuantKVDataLoader:
    def __init__(self, args: argparse.Namespace, onnx_model_path: str = ""):
        self.batch_size = 1
        self.pad_max = args.pad_max

        tokenizer = LlamaTokenizer.from_pretrained(args.original_model_name, use_auth_token=args.use_auth_token)
        dataset = load_dataset(args.smooth_quant_dataset, split="train")
        dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        self.decoder_model = (
            create_onnxruntime_session(
                onnx_model_path,
                args.execution_provider != "cpu",  # use_gpu
                provider=args.execution_provider,
                verbose=args.verbose,
            )
            if onnx_model_path
            else None
        )

    def collate_batch(self, batch):
        input_ids_batched = []
        attention_mask_batched = []
        position_ids_batched = []
        labels = []

        for text in batch:
            # Set inputs for model
            input_ids = text["input_ids"]
            attention_mask = torch.ones(len(input_ids))
            position_ids = get_position_ids(attention_mask, use_past_kv=False)
            label = len(input_ids) - 1

            # Pad input data because all model inputs must have same shape
            pad_len = self.pad_max - input_ids.shape[0]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            position_ids = pad(position_ids, (0, pad_len), value=0)

            input_ids_batched.append(input_ids)
            attention_mask_batched.append(attention_mask)
            position_ids_batched.append(position_ids)
            labels.append(label)

        input_ids_batched = torch.vstack(input_ids_batched)
        attention_mask_batched = torch.vstack(attention_mask_batched)
        position_ids_batched = torch.vstack(position_ids_batched)
        labels = torch.tensor(labels)

        return (input_ids_batched, attention_mask_batched, position_ids_batched), labels

    def __iter__(self):
        try:
            for (input_ids, attention_mask, position_ids), labels in self.dataloader:
                # Inputs for decoder_model.onnx
                inputs = {
                    "input_ids": input_ids[:, :-1].detach().cpu().numpy().astype(np.int64),
                    "attention_mask": attention_mask[:, :-1].detach().cpu().numpy().astype(np.int64),
                    "position_ids": position_ids[:, :-1].detach().cpu().numpy().astype(np.int64),
                }
                label = labels.detach().cpu().numpy()

                if self.decoder_model is not None:
                    # Run decoder_model.onnx to get inputs for decoder_with_past_model.onnx
                    outputs = self.decoder_model.run(None, inputs)

                    for i in range(int((len(outputs) - 1) / 2)):
                        inputs[f"past_key_values.{i}.key"] = outputs[i * 2 + 1]
                        inputs[f"past_key_values.{i}.value"] = outputs[i * 2 + 2]
                    past_sequence_length = inputs["past_key_values.0.key"].shape[2]

                    inputs["input_ids"] = input_ids[:, -1].unsqueeze(0).detach().cpu().numpy().astype(np.int64)
                    attn_mask_torch = torch.ones((self.batch_size, past_sequence_length + 1), dtype=torch.int64)
                    inputs["attention_mask"] = attn_mask_torch.detach().cpu().numpy().astype(np.int64)
                    inputs["position_ids"] = (
                        get_position_ids(attn_mask_torch, use_past_kv=True).detach().cpu().numpy().astype(np.int64)
                    )

                # Yield (inputs, label) tuple for Intel's Neural Compressor:
                # https://github.com/intel/neural-compressor/blob/d4baed9ea11614e1f0dc8a1f4f55b73ed3ed585c/neural_compressor/quantization.py#L55-L62
                yield (inputs, label)

        except StopIteration:
            return

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\engine_builder_torch.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from diffusion_models import PipelineInfo
from engine_builder import EngineBuilder, EngineType

logger = logging.getLogger(__name__)


class TorchEngineBuilder(EngineBuilder):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        max_batch_size=16,
        device="cuda",
        use_cuda_graph=False,
    ):
        """
        Initializes the ONNX Runtime TensorRT ExecutionProvider Engine Builder.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            device (str):
                device to run.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
        """
        super().__init__(
            EngineType.TORCH,
            pipeline_info,
            max_batch_size=max_batch_size,
            device=device,
            use_cuda_graph=use_cuda_graph,
        )

        self.compile_config = {}
        if use_cuda_graph:
            self.compile_config = {
                "clip": {"mode": "reduce-overhead", "dynamic": False},
                "clip2": {"mode": "reduce-overhead", "dynamic": False},
                "unet": {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False},
                "unetxl": {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False},
                "vae": {"mode": "reduce-overhead", "fullgraph": False, "dynamic": False},
            }

    def build_engines(
        self,
        framework_model_dir: str,
    ):
        import torch

        self.torch_device = torch.device("cuda", torch.cuda.current_device())
        self.load_models(framework_model_dir)

        pipe = self.load_pipeline_with_lora() if self.pipeline_info.lora_weights else None

        built_engines = {}
        for model_name, model_obj in self.models.items():
            model = self.get_or_load_model(pipe, model_name, model_obj, framework_model_dir)
            if self.pipeline_info.is_xl() and not self.custom_fp16_vae:
                model = model.to(device=self.torch_device, dtype=torch.float32)
            else:
                model = model.to(device=self.torch_device, dtype=torch.float16)

            if model_name in self.compile_config:
                compile_config = self.compile_config[model_name]
                if model_name in ["unet", "unetxl"]:
                    model.to(memory_format=torch.channels_last)
                engine = torch.compile(model, **compile_config)
                built_engines[model_name] = engine
            else:  # eager mode
                built_engines[model_name] = model

        self.engines = built_engines

    def run_engine(self, model_name, feed_dict):
        if model_name in ["unet", "unetxl"]:
            if "controlnet_images" in feed_dict:
                return {"latent": self.engines[model_name](**feed_dict)}

            if model_name == "unetxl":
                added_cond_kwargs = {k: feed_dict[k] for k in feed_dict if k in ["text_embeds", "time_ids"]}
                return {
                    "latent": self.engines[model_name](
                        feed_dict["sample"],
                        feed_dict["timestep"],
                        feed_dict["encoder_hidden_states"],
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                }

            return {
                "latent": self.engines[model_name](
                    feed_dict["sample"], feed_dict["timestep"], feed_dict["encoder_hidden_states"], return_dict=False
                )[0]
            }

        if model_name in ["vae_encoder"]:
            return {"latent": self.engines[model_name](feed_dict["images"])}

        raise RuntimeError(f"Shall not reach here: {model_name}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\utils\units.py ===

# Copyright (c) 2010-2024 openpyxl

import math


#constants

DEFAULT_ROW_HEIGHT = 15.  # Default row height measured in point size.
BASE_COL_WIDTH = 8 # in characters
DEFAULT_COLUMN_WIDTH = BASE_COL_WIDTH + 5
#  = baseColumnWidth + {margin padding (2 pixels on each side, totalling 4 pixels)} + {gridline (1pixel)}


DEFAULT_LEFT_MARGIN = 0.7 # in inches, = right margin
DEFAULT_TOP_MARGIN = 0.7874 # in inches = bottom margin
DEFAULT_HEADER = 0.3 # in inches


# Conversion functions
"""
From the ECMA Spec (4th Edition part 1)
Page setup: "Left Page Margin in inches" p. 1647

Docs from
http://startbigthinksmall.wordpress.com/2010/01/04/points-inches-and-emus-measuring-units-in-office-open-xml/

See also http://msdn.microsoft.com/en-us/library/dd560821(v=office.12).aspx

dxa: The main unit in OOXML is a twentieth of a point. Also called twips.
pt: point. In Excel there are 72 points to an inch
hp: half-points are used to specify font sizes. A font-size of 12pt equals 24 half points
pct: Half-points are used to specify font sizes. A font-size of 12pt equals 24 half points

EMU: English Metric Unit, EMUs are used for coordinates in vector-based
drawings and embedded pictures. One inch equates to 914400 EMUs and a
centimeter is 360000. For bitmaps the default resolution is 96 dpi (known as
PixelsPerInch in Excel). Spec p. 1122

For radial geometry Excel uses integer units of 1/60000th of a degree.
"""



def inch_to_dxa(value):
    """1 inch = 72 * 20 dxa"""
    return int(value * 20 * 72)

def dxa_to_inch(value):
    return value / 72 / 20


def dxa_to_cm(value):
    return 2.54 * dxa_to_inch(value)

def cm_to_dxa(value):
    emu = cm_to_EMU(value)
    inch = EMU_to_inch(emu)
    return inch_to_dxa(inch)


def pixels_to_EMU(value):
    """1 pixel = 9525 EMUs"""
    return int(value * 9525)

def EMU_to_pixels(value):
    return round(value / 9525)


def cm_to_EMU(value):
    """1 cm = 360000 EMUs"""
    return int(value * 360000)

def EMU_to_cm(value):
    return round(value / 360000, 4)


def inch_to_EMU(value):
    """1 inch = 914400 EMUs"""
    return int(value * 914400)

def EMU_to_inch(value):
    return round(value / 914400, 4)


def pixels_to_points(value, dpi=96):
    """96 dpi, 72i"""
    return value * 72 / dpi


def points_to_pixels(value, dpi=96):
    return int(math.ceil(value * dpi / 72))


def degrees_to_angle(value):
    """1 degree = 60000 angles"""
    return int(round(value * 60000))


def angle_to_degrees(value):
    return round(value / 60000, 2)


def short_color(color):
    """ format a color to its short size """
    if len(color) > 6:
        return color[2:]
    return color

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\freeze.py ===
import sys
from optparse import Values
from typing import AbstractSet, List

from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.operations.freeze import freeze
from pip._internal.utils.compat import stdlib_pkgs


def _should_suppress_build_backends() -> bool:
    return sys.version_info < (3, 12)


def _dev_pkgs() -> AbstractSet[str]:
    pkgs = {"pip"}

    if _should_suppress_build_backends():
        pkgs |= {"setuptools", "distribute", "wheel"}

    return pkgs


class FreezeCommand(Command):
    """
    Output installed packages in requirements format.

    packages are listed in a case-insensitive sorted order.
    """

    ignore_require_venv = True
    usage = """
      %prog [options]"""

    def add_options(self) -> None:
        self.cmd_opts.add_option(
            "-r",
            "--requirement",
            dest="requirements",
            action="append",
            default=[],
            metavar="file",
            help=(
                "Use the order in the given requirements file and its "
                "comments when generating output. This option can be "
                "used multiple times."
            ),
        )
        self.cmd_opts.add_option(
            "-l",
            "--local",
            dest="local",
            action="store_true",
            default=False,
            help=(
                "If in a virtualenv that has global access, do not output "
                "globally-installed packages."
            ),
        )
        self.cmd_opts.add_option(
            "--user",
            dest="user",
            action="store_true",
            default=False,
            help="Only output packages installed in user-site.",
        )
        self.cmd_opts.add_option(cmdoptions.list_path())
        self.cmd_opts.add_option(
            "--all",
            dest="freeze_all",
            action="store_true",
            help=(
                "Do not skip these packages in the output:"
                " {}".format(", ".join(_dev_pkgs()))
            ),
        )
        self.cmd_opts.add_option(
            "--exclude-editable",
            dest="exclude_editable",
            action="store_true",
            help="Exclude editable package from output.",
        )
        self.cmd_opts.add_option(cmdoptions.list_exclude())

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options: Values, args: List[str]) -> int:
        skip = set(stdlib_pkgs)
        if not options.freeze_all:
            skip.update(_dev_pkgs())

        if options.excludes:
            skip.update(options.excludes)

        cmdoptions.check_list_path_option(options)

        for line in freeze(
            requirement=options.requirements,
            local_only=options.local,
            user_only=options.user,
            paths=options.path,
            isolated=options.isolated_mode,
            skip=skip,
            exclude_editable=options.exclude_editable,
        ):
            sys.stdout.write(line + "\n")
        return SUCCESS

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\concurrency.py ===
# util/concurrency.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

from __future__ import annotations

import asyncio  # noqa
import typing
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import TypeVar

have_greenlet = False
greenlet_error = None
try:
    import greenlet  # type: ignore[import-untyped,unused-ignore]  # noqa: F401,E501
except ImportError as e:
    greenlet_error = str(e)
    pass
else:
    have_greenlet = True
    from ._concurrency_py3k import await_only as await_only
    from ._concurrency_py3k import await_fallback as await_fallback
    from ._concurrency_py3k import in_greenlet as in_greenlet
    from ._concurrency_py3k import greenlet_spawn as greenlet_spawn
    from ._concurrency_py3k import is_exit_exception as is_exit_exception
    from ._concurrency_py3k import AsyncAdaptedLock as AsyncAdaptedLock
    from ._concurrency_py3k import _Runner

_T = TypeVar("_T")


class _AsyncUtil:
    """Asyncio util for test suite/ util only"""

    def __init__(self) -> None:
        if have_greenlet:
            self.runner = _Runner()

    def run(
        self,
        fn: Callable[..., Coroutine[Any, Any, _T]],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run coroutine on the loop"""
        return self.runner.run(fn(*args, **kwargs))

    def run_in_greenlet(
        self, fn: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> _T:
        """Run sync function in greenlet. Support nested calls"""
        if have_greenlet:
            if self.runner.get_loop().is_running():
                return fn(*args, **kwargs)
            else:
                return self.runner.run(greenlet_spawn(fn, *args, **kwargs))
        else:
            return fn(*args, **kwargs)

    def close(self) -> None:
        if have_greenlet:
            self.runner.close()


if not typing.TYPE_CHECKING and not have_greenlet:

    def _not_implemented():
        # this conditional is to prevent pylance from considering
        # greenlet_spawn() etc as "no return" and dimming out code below it
        if have_greenlet:
            return None

        raise ValueError(
            "the greenlet library is required to use this function."
            " %s" % greenlet_error
            if greenlet_error
            else ""
        )

    def is_exit_exception(e):  # noqa: F811
        return not isinstance(e, Exception)

    def await_only(thing):  # type: ignore  # noqa: F811
        _not_implemented()

    def await_fallback(thing):  # type: ignore  # noqa: F811
        return thing

    def in_greenlet():  # type: ignore  # noqa: F811
        _not_implemented()

    def greenlet_spawn(fn, *args, **kw):  # type: ignore  # noqa: F811
        _not_implemented()

    def AsyncAdaptedLock(*args, **kw):  # type: ignore  # noqa: F811
        _not_implemented()

    def _util_async_run(fn, *arg, **kw):  # type: ignore  # noqa: F811
        return fn(*arg, **kw)

    def _util_async_run_coroutine_function(fn, *arg, **kw):  # type: ignore  # noqa: F811,E501
        _not_implemented()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\euler.py ===
"""
This module implements a method to find
Euler-Lagrange Equations for given Lagrangian.
"""
from itertools import combinations_with_replacement
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities.iterables import iterable


def euler_equations(L, funcs=(), vars=()):
    r"""
    Find the Euler-Lagrange equations [1]_ for a given Lagrangian.

    Parameters
    ==========

    L : Expr
        The Lagrangian that should be a function of the functions listed
        in the second argument and their derivatives.

        For example, in the case of two functions $f(x,y)$, $g(x,y)$ and
        two independent variables $x$, $y$ the Lagrangian has the form:

            .. math:: L\left(f(x,y),g(x,y),\frac{\partial f(x,y)}{\partial x},
                      \frac{\partial f(x,y)}{\partial y},
                      \frac{\partial g(x,y)}{\partial x},
                      \frac{\partial g(x,y)}{\partial y},x,y\right)

        In many cases it is not necessary to provide anything, except the
        Lagrangian, it will be auto-detected (and an error raised if this
        cannot be done).

    funcs : Function or an iterable of Functions
        The functions that the Lagrangian depends on. The Euler equations
        are differential equations for each of these functions.

    vars : Symbol or an iterable of Symbols
        The Symbols that are the independent variables of the functions.

    Returns
    =======

    eqns : list of Eq
        The list of differential equations, one for each function.

    Examples
    ========

    >>> from sympy import euler_equations, Symbol, Function
    >>> x = Function('x')
    >>> t = Symbol('t')
    >>> L = (x(t).diff(t))**2/2 - x(t)**2/2
    >>> euler_equations(L, x(t), t)
    [Eq(-x(t) - Derivative(x(t), (t, 2)), 0)]
    >>> u = Function('u')
    >>> x = Symbol('x')
    >>> L = (u(t, x).diff(t))**2/2 - (u(t, x).diff(x))**2/2
    >>> euler_equations(L, u(t, x), [t, x])
    [Eq(-Derivative(u(t, x), (t, 2)) + Derivative(u(t, x), (x, 2)), 0)]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation

    """

    funcs = tuple(funcs) if iterable(funcs) else (funcs,)

    if not funcs:
        funcs = tuple(L.atoms(Function))
    else:
        for f in funcs:
            if not isinstance(f, Function):
                raise TypeError('Function expected, got: %s' % f)

    vars = tuple(vars) if iterable(vars) else (vars,)

    if not vars:
        vars = funcs[0].args
    else:
        vars = tuple(sympify(var) for var in vars)

    if not all(isinstance(v, Symbol) for v in vars):
        raise TypeError('Variables are not symbols, got %s' % vars)

    for f in funcs:
        if not vars == f.args:
            raise ValueError("Variables %s do not match args: %s" % (vars, f))

    order = max([len(d.variables) for d in L.atoms(Derivative)
                        if d.expr in funcs] + [0])

    eqns = []
    for f in funcs:
        eq = diff(L, f)
        for i in range(1, order + 1):
            for p in combinations_with_replacement(vars, i):
                eq = eq + S.NegativeOne**i*diff(L, diff(f, *p), *p)
        new_eq = Eq(eq, 0)
        if isinstance(new_eq, Eq):
            eqns.append(new_eq)

    return eqns

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_matrices.py ===
#
# Code for testing deprecated matrix classes. New test code should not be added
# here. Instead, add it to test_matrixbase.py.
#
# This entire test module and the corresponding sympy/matrices/matrices.py
# module will be removed in a future release.
#
import random
import concurrent.futures
from collections.abc import Hashable

from sympy.core.add import Add
from sympy.core.function import Function, diff, expand
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.integrals.integrals import integrate
from sympy.matrices.expressions.transpose import transpose
from sympy.physics.quantum.operator import HermitianOperator, Operator, Dagger
from sympy.polys.polytools import (Poly, PurePoly)
from sympy.polys.rootoftools import RootOf
from sympy.printing.str import sstr
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.matrices.exceptions import (ShapeError, MatrixError,
    NonSquareMatrixError)
from sympy.matrices.matrixbase import DeferredVector
from sympy.matrices.determinant import _find_reasonable_pivot_naive
from sympy.matrices.utilities import _simplify
from sympy.matrices import (
    GramSchmidt, ImmutableMatrix, ImmutableSparseMatrix, Matrix,
    SparseMatrix, casoratian, diag, eye, hessian,
    matrix_multiply_elementwise, ones, randMatrix, rot_axis1, rot_axis2,
    rot_axis3, wronskian, zeros, MutableDenseMatrix, ImmutableDenseMatrix,
    MatrixSymbol, dotprodsimp, rot_ccw_axis1, rot_ccw_axis2, rot_ccw_axis3)
from sympy.matrices.utilities import _dotprodsimp_state
from sympy.core import Tuple, Wild
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.utilities.iterables import flatten, capture, iterable
from sympy.utilities.exceptions import ignore_warnings
from sympy.testing.pytest import (raises, XFAIL, slow, skip, skip_under_pyodide,
                                  warns_deprecated_sympy)
from sympy.assumptions import Q
from sympy.tensor.array import Array
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.matrices.expressions import MatPow
from sympy.algebras import Quaternion

from sympy import O

from sympy.abc import a, b, c, d, x, y, z, t


# don't re-order this list
classes = (Matrix, SparseMatrix, ImmutableMatrix, ImmutableSparseMatrix)


# Test the deprecated matrixmixins
from sympy.matrices.common import _MinimalMatrix, _CastableMatrix
from sympy.matrices.matrices import MatrixSubspaces, MatrixReductions


with warns_deprecated_sympy():
    class SubspaceOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixSubspaces):
        pass


with warns_deprecated_sympy():
    class ReductionsOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixReductions):
        pass


def eye_Reductions(n):
    return ReductionsOnlyMatrix(n, n, lambda i, j: int(i == j))


def zeros_Reductions(n):
    return ReductionsOnlyMatrix(n, n, lambda i, j: 0)


def test_args():
    for n, cls in enumerate(classes):
        m = cls.zeros(3, 2)
        # all should give back the same type of arguments, e.g. ints for shape
        assert m.shape == (3, 2) and all(type(i) is int for i in m.shape)
        assert m.rows == 3 and type(m.rows) is int
        assert m.cols == 2 and type(m.cols) is int
        if not n % 2:
            assert type(m.flat()) in (list, tuple, Tuple)
        else:
            assert type(m.todok()) is dict


def test_deprecated_mat_smat():
    for cls in Matrix, ImmutableMatrix:
        m = cls.zeros(3, 2)
        with warns_deprecated_sympy():
            mat = m._mat
        assert mat == m.flat()
    for cls in SparseMatrix, ImmutableSparseMatrix:
        m = cls.zeros(3, 2)
        with warns_deprecated_sympy():
            smat = m._smat
        assert smat == m.todok()


def test_division():
    v = Matrix(1, 2, [x, y])
    assert v/z == Matrix(1, 2, [x/z, y/z])


def test_sum():
    m = Matrix([[1, 2, 3], [x, y, x], [2*y, -50, z*x]])
    assert m + m == Matrix([[2, 4, 6], [2*x, 2*y, 2*x], [4*y, -100, 2*z*x]])
    n = Matrix(1, 2, [1, 2])
    raises(ShapeError, lambda: m + n)

def test_abs():
    m = Matrix(1, 2, [-3, x])
    n = Matrix(1, 2, [3, Abs(x)])
    assert abs(m) == n

def test_addition():
    a = Matrix((
        (1, 2),
        (3, 1),
    ))

    b = Matrix((
        (1, 2),
        (3, 0),
    ))

    assert a + b == a.add(b) == Matrix([[2, 4], [6, 1]])


def test_fancy_index_matrix():
    for M in (Matrix, SparseMatrix):
        a = M(3, 3, range(9))
        assert a == a[:, :]
        assert a[1, :] == Matrix(1, 3, [3, 4, 5])
        assert a[:, 1] == Matrix([1, 4, 7])
        assert a[[0, 1], :] == Matrix([[0, 1, 2], [3, 4, 5]])
        assert a[[0, 1], 2] == a[[0, 1], [2]]
        assert a[2, [0, 1]] == a[[2], [0, 1]]
        assert a[:, [0, 1]] == Matrix([[0, 1], [3, 4], [6, 7]])
        assert a[0, 0] == 0
        assert a[0:2, :] == Matrix([[0, 1, 2], [3, 4, 5]])
        assert a[:, 0:2] == Matrix([[0, 1], [3, 4], [6, 7]])
        assert a[::2, 1] == a[[0, 2], 1]
        assert a[1, ::2] == a[1, [0, 2]]
        a = M(3, 3, range(9))
        assert a[[0, 2, 1, 2, 1], :] == Matrix([
            [0, 1, 2],
            [6, 7, 8],
            [3, 4, 5],
            [6, 7, 8],
            [3, 4, 5]])
        assert a[:, [0,2,1,2,1]] == Matrix([
            [0, 2, 1, 2, 1],
            [3, 5, 4, 5, 4],
            [6, 8, 7, 8, 7]])

    a = SparseMatrix.zeros(3)
    a[1, 2] = 2
    a[0, 1] = 3
    a[2, 0] = 4
    assert a.extract([1, 1], [2]) == Matrix([
    [2],
    [2]])
    assert a.extract([1, 0], [2, 2, 2]) == Matrix([
    [2, 2, 2],
    [0, 0, 0]])
    assert a.extract([1, 0, 1, 2], [2, 0, 1, 0]) == Matrix([
        [2, 0, 0, 0],
        [0, 0, 3, 0],
        [2, 0, 0, 0],
        [0, 4, 0, 4]])


def test_multiplication():
    a = Matrix((
        (1, 2),
        (3, 1),
        (0, 6),
    ))

    b = Matrix((
        (1, 2),
        (3, 0),
    ))

    c = a*b
    assert c[0, 0] == 7
    assert c[0, 1] == 2
    assert c[1, 0] == 6
    assert c[1, 1] == 6
    assert c[2, 0] == 18
    assert c[2, 1] == 0

    try:
        eval('c = a @ b')
    except SyntaxError:
        pass
    else:
        assert c[0, 0] == 7
        assert c[0, 1] == 2
        assert c[1, 0] == 6
        assert c[1, 1] == 6
        assert c[2, 0] == 18
        assert c[2, 1] == 0

    h = matrix_multiply_elementwise(a, c)
    assert h == a.multiply_elementwise(c)
    assert h[0, 0] == 7
    assert h[0, 1] == 4
    assert h[1, 0] == 18
    assert h[1, 1] == 6
    assert h[2, 0] == 0
    assert h[2, 1] == 0
    raises(ShapeError, lambda: matrix_multiply_elementwise(a, b))

    c = b * Symbol("x")
    assert isinstance(c, Matrix)
    assert c[0, 0] == x
    assert c[0, 1] == 2*x
    assert c[1, 0] == 3*x
    assert c[1, 1] == 0

    c2 = x * b
    assert c == c2

    c = 5 * b
    assert isinstance(c, Matrix)
    assert c[0, 0] == 5
    assert c[0, 1] == 2*5
    assert c[1, 0] == 3*5
    assert c[1, 1] == 0

    try:
        eval('c = 5 @ b')
    except SyntaxError:
        pass
    else:
        assert isinstance(c, Matrix)
        assert c[0, 0] == 5
        assert c[0, 1] == 2*5
        assert c[1, 0] == 3*5
        assert c[1, 1] == 0


def test_multiplication_inf_zero():

    M = Matrix([[oo, 0], [0, oo]])
    assert M ** 2 == M

    M = Matrix([[oo, oo], [0, 0]])
    assert M ** 2 == Matrix([[nan, nan], [nan, nan]])

    A = Matrix([
        [0, 0, 0, -S(1)/2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-S(1)/2, 0, 0, 0]])

    B = Matrix([
        [pi*x**2, 0, pi*b*x**4/8 + pi*a*x**4/8 + O(x**5), pi*x**4/2 + pi*b**2*x**6/32 + pi*a*b*x**6/48 + pi*a**2*x**6/32 + O(x**7)],
        [0, pi*x**4/4, O(x**6), O(x**8)],
        [pi*b*x**4/8 + pi*a*x**4/8 + O(x**5), O(x**6), pi*b**2*x**6/32 + pi*a*b*x**6/48 + pi*a**2*x**6/32 + O(x**7), pi*b*x**6/12 + pi*a*x**6/12 + O(x**7)],
        [pi*x**4/2 + pi*b**2*x**6/32 + pi*a*b*x**6/48 + pi*a**2*x**6/32 + O(x**7), O(x**8), pi*b*x**6/12 + pi*a*x**6/12 + O(x**7), pi*x**6/3 + 3*pi*b**2*x**8/64 + pi*a*b*x**8/32 + 3*pi*a**2*x**8/64 + O(x**9)]])

    C = Matrix([
    [-pi*x**4/4 - pi*b**2*x**6/64 - pi*a*b*x**6/96 - pi*a**2*x**6/64 + O(x**7),   O(x**8),                       -pi*b*x**6/24 - pi*a*x**6/24 + O(x**7), -pi*x**6/6 - 3*pi*b**2*x**8/128 - pi*a*b*x**8/64 - 3*pi*a**2*x**8/128 + O(x**9)],
    [                                                                        0, pi*x**4/4,                                                      O(x**6),                                                                         O(x**8)],
    [                                      pi*b*x**4/8 + pi*a*x**4/8 + O(x**5),   O(x**6), pi*b**2*x**6/32 + pi*a*b*x**6/48 + pi*a**2*x**6/32 + O(x**7),                                           pi*b*x**6/12 + pi*a*x**6/12 + O(x**7)],
    [                                                               -pi*x**2/2,         0,                       -pi*b*x**4/16 - pi*a*x**4/16 + O(x**5),       -pi*x**4/4 - pi*b**2*x**6/64 - pi*a*b*x**6/96 - pi*a**2*x**6/64 + O(x**7)]])

    C2 = Matrix(4, 4, lambda i, j: Add(*(A[i,k]*B[k,j] for k in range(4))))

    assert A*B == C == C2


def test_power():
    raises(NonSquareMatrixError, lambda: Matrix((1, 2))**2)

    R = Rational
    A = Matrix([[2, 3], [4, 5]])
    assert (A**-3)[:] == [R(-269)/8, R(153)/8, R(51)/2, R(-29)/2]
    assert (A**5)[:] == [6140, 8097, 10796, 14237]
    A = Matrix([[2, 1, 3], [4, 2, 4], [6, 12, 1]])
    assert (A**3)[:] == [290, 262, 251, 448, 440, 368, 702, 954, 433]
    assert A**0 == eye(3)
    assert A**1 == A
    assert (Matrix([[2]]) ** 100)[0, 0] == 2**100
    assert eye(2)**10000000 == eye(2)
    assert Matrix([[1, 2], [3, 4]])**Integer(2) == Matrix([[7, 10], [15, 22]])

    A = Matrix([[33, 24], [48, 57]])
    assert (A**S.Half)[:] == [5, 2, 4, 7]
    A = Matrix([[0, 4], [-1, 5]])
    assert (A**S.Half)**2 == A

    assert Matrix([[1, 0], [1, 1]])**S.Half == Matrix([[1, 0], [S.Half, 1]])
    assert Matrix([[1, 0], [1, 1]])**0.5 == Matrix([[1, 0], [0.5, 1]])
    from sympy.abc import n
    assert Matrix([[1, a], [0, 1]])**n == Matrix([[1, a*n], [0, 1]])
    assert Matrix([[b, a], [0, b]])**n == Matrix([[b**n, a*b**(n-1)*n], [0, b**n]])
    assert Matrix([
        [a**n, a**(n - 1)*n, (a**n*n**2 - a**n*n)/(2*a**2)],
        [   0,         a**n,                  a**(n - 1)*n],
        [   0,            0,                          a**n]])
    assert Matrix([[a, 1, 0], [0, a, 0], [0, 0, b]])**n == Matrix([
        [a**n, a**(n-1)*n, 0],
        [0, a**n, 0],
        [0, 0, b**n]])

    A = Matrix([[1, 0], [1, 7]])
    assert A._matrix_pow_by_jordan_blocks(S(3)) == A._eval_pow_by_recursion(3)
    A = Matrix([[2]])
    assert A**10 == Matrix([[2**10]]) == A._matrix_pow_by_jordan_blocks(S(10)) == \
        A._eval_pow_by_recursion(10)

    # testing a matrix that cannot be jordan blocked issue 11766
    m = Matrix([[3, 0, 0, 0, -3], [0, -3, -3, 0, 3], [0, 3, 0, 3, 0], [0, 0, 3, 0, 3], [3, 0, 0, 3, 0]])
    raises(MatrixError, lambda: m._matrix_pow_by_jordan_blocks(S(10)))

    # test issue 11964
    raises(MatrixError, lambda: Matrix([[1, 1], [3, 3]])._matrix_pow_by_jordan_blocks(S(-10)))
    A = Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # Nilpotent jordan block size 3
    assert A**10.0 == Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    raises(ValueError, lambda: A**2.1)
    raises(ValueError, lambda: A**Rational(3, 2))
    A = Matrix([[8, 1], [3, 2]])
    assert A**10.0 == Matrix([[1760744107, 272388050], [817164150, 126415807]])
    A = Matrix([[0, 0, 1], [0, 0, 1], [0, 0, 1]])  # Nilpotent jordan block size 1
    assert A**10.0 == Matrix([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    A = Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 1]])  # Nilpotent jordan block size 2
    assert A**10.0 == Matrix([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    n = Symbol('n', integer=True)
    assert isinstance(A**n, MatPow)
    n = Symbol('n', integer=True, negative=True)
    raises(ValueError, lambda: A**n)
    n = Symbol('n', integer=True, nonnegative=True)
    assert A**n == Matrix([
        [KroneckerDelta(0, n), KroneckerDelta(1, n), -KroneckerDelta(0, n) - KroneckerDelta(1, n) + 1],
        [                   0, KroneckerDelta(0, n),                         1 - KroneckerDelta(0, n)],
        [                   0,                    0,                                                1]])
    assert A**(n + 2) == Matrix([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    raises(ValueError, lambda: A**Rational(3, 2))
    A = Matrix([[0, 0, 1], [3, 0, 1], [4, 3, 1]])
    assert A**5.0 == Matrix([[168,  72,  89], [291, 144, 161], [572, 267, 329]])
    assert A**5.0 == A**5
    A = Matrix([[0, 1, 0],[-1, 0, 0],[0, 0, 0]])
    n = Symbol("n")
    An = A**n
    assert An.subs(n, 2).doit() == A**2
    raises(ValueError, lambda: An.subs(n, -2).doit())
    assert An * An == A**(2*n)

    # concretizing behavior for non-integer and complex powers
    A = Matrix([[0,0,0],[0,0,0],[0,0,0]])
    n = Symbol('n', integer=True, positive=True)
    assert A**n == A
    n = Symbol('n', integer=True, nonnegative=True)
    assert A**n == diag(0**n, 0**n, 0**n)
    assert (A**n).subs(n, 0) == eye(3)
    assert (A**n).subs(n, 1) == zeros(3)
    A = Matrix ([[2,0,0],[0,2,0],[0,0,2]])
    assert A**2.1 == diag (2**2.1, 2**2.1, 2**2.1)
    assert A**I == diag (2**I, 2**I, 2**I)
    A = Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 1]])
    raises(ValueError, lambda: A**2.1)
    raises(ValueError, lambda: A**I)
    A = Matrix([[S.Half, S.Half], [S.Half, S.Half]])
    assert A**S.Half == A
    A = Matrix([[1, 1],[3, 3]])
    assert A**S.Half == Matrix ([[S.Half, S.Half], [3*S.Half, 3*S.Half]])


def test_issue_17247_expression_blowup_1():
    M = Matrix([[1+x, 1-x], [1-x, 1+x]])
    with dotprodsimp(True):
        assert M.exp().expand() == Matrix([
            [ (exp(2*x) + exp(2))/2, (-exp(2*x) + exp(2))/2],
            [(-exp(2*x) + exp(2))/2,  (exp(2*x) + exp(2))/2]])

def test_issue_17247_expression_blowup_2():
    M = Matrix([[1+x, 1-x], [1-x, 1+x]])
    with dotprodsimp(True):
        P, J = M.jordan_form ()
        assert P*J*P.inv()

def test_issue_17247_expression_blowup_3():
    M = Matrix([[1+x, 1-x], [1-x, 1+x]])
    with dotprodsimp(True):
        assert M**100 == Matrix([
            [633825300114114700748351602688*x**100 + 633825300114114700748351602688, 633825300114114700748351602688 - 633825300114114700748351602688*x**100],
            [633825300114114700748351602688 - 633825300114114700748351602688*x**100, 633825300114114700748351602688*x**100 + 633825300114114700748351602688]])

def test_issue_17247_expression_blowup_4():
# This matrix takes extremely long on current master even with intermediate simplification so an abbreviated version is used. It is left here for test in case of future optimizations.
#     M = Matrix(S('''[
#         [             -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,      1/4 - 5*I/16,      65/128 + 87*I/64,         -9/32 - I/16,      183/256 - 97*I/128,       3/64 + 13*I/64,         -23/32 - 59*I/256,      15/128 - 3*I/32,        19/256 + 551*I/1024],
#         [-149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,  85/256 - 33*I/16,  805/128 + 2415*I/512, -219/128 + 115*I/256, 6301/4096 - 6609*I/1024,  119/128 + 143*I/128, -10879/2048 + 4343*I/4096,  129/256 - 549*I/512, 42533/16384 + 29103*I/8192],
#         [          1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,         1/4 - 5*I/16,        65/128 + 87*I/64,         -9/32 - I/16,        183/256 - 97*I/128,       3/64 + 13*I/64,          -23/32 - 59*I/256],
#         [   -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,     85/256 - 33*I/16,    805/128 + 2415*I/512, -219/128 + 115*I/256,   6301/4096 - 6609*I/1024,  119/128 + 143*I/128,  -10879/2048 + 4343*I/4096],
#         [            1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,            1/4 + I/2,        -129/64 - 9*I/64,         1/4 - 5*I/16,          65/128 + 87*I/64,         -9/32 - I/16,         183/256 - 97*I/128],
#         [         21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,     125/64 + 87*I/64,   -2063/256 + 541*I/128,     85/256 - 33*I/16,      805/128 + 2415*I/512, -219/128 + 115*I/256,    6301/4096 - 6609*I/1024],
#         [               -2,         17/4 - 13*I/2,             1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,                 -3/4,         45/32 - 37*I/16,            1/4 + I/2,          -129/64 - 9*I/64,         1/4 - 5*I/16,           65/128 + 87*I/64],
#         [     1/4 + 13*I/4,    -825/64 - 147*I/32,          21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64,    -149/64 + 49*I/32,   -177/128 - 1369*I/128,     125/64 + 87*I/64,     -2063/256 + 541*I/128,     85/256 - 33*I/16,       805/128 + 2415*I/512],
#         [             -4*I,            27/2 + 6*I,                -2,         17/4 - 13*I/2,             1 + I,         -19/4 + 5*I/4,              1/2 - I,           9/4 + 55*I/16,                 -3/4,           45/32 - 37*I/16,            1/4 + I/2,           -129/64 - 9*I/64],
#         [      1/4 + 5*I/2,       -23/8 - 57*I/16,      1/4 + 13*I/4,    -825/64 - 147*I/32,          21/8 + I,    -537/64 + 143*I/16,       -5/8 - 39*I/16,     2473/256 + 137*I/64,    -149/64 + 49*I/32,     -177/128 - 1369*I/128,     125/64 + 87*I/64,      -2063/256 + 541*I/128],
#         [               -4,               9 - 5*I,              -4*I,            27/2 + 6*I,                -2,         17/4 - 13*I/2,                1 + I,           -19/4 + 5*I/4,              1/2 - I,             9/4 + 55*I/16,                 -3/4,            45/32 - 37*I/16],
#         [             -2*I,        119/8 + 29*I/4,       1/4 + 5*I/2,       -23/8 - 57*I/16,      1/4 + 13*I/4,    -825/64 - 147*I/32,             21/8 + I,      -537/64 + 143*I/16,       -5/8 - 39*I/16,       2473/256 + 137*I/64,    -149/64 + 49*I/32,      -177/128 - 1369*I/128]]'''))
#     assert M**10 == Matrix([
#         [    7*(-221393644768594642173548179825793834595 - 1861633166167425978847110897013541127952*I)/9671406556917033397649408,      15*(31670992489131684885307005100073928751695 + 10329090958303458811115024718207404523808*I)/77371252455336267181195264,   7*(-3710978679372178839237291049477017392703 + 1377706064483132637295566581525806894169*I)/19342813113834066795298816,            (9727707023582419994616144751727760051598 - 59261571067013123836477348473611225724433*I)/9671406556917033397649408,      (31896723509506857062605551443641668183707 + 54643444538699269118869436271152084599580*I)/38685626227668133590597632,       (-2024044860947539028275487595741003997397402 + 130959428791783397562960461903698670485863*I)/309485009821345068724781056,     3*(26190251453797590396533756519358368860907 - 27221191754180839338002754608545400941638*I)/77371252455336267181195264,      (1154643595139959842768960128434994698330461 + 3385496216250226964322872072260446072295634*I)/618970019642690137449562112,     3*(-31849347263064464698310044805285774295286 - 11877437776464148281991240541742691164309*I)/77371252455336267181195264,     (4661330392283532534549306589669150228040221 - 4171259766019818631067810706563064103956871*I)/1237940039285380274899124224,        (9598353794289061833850770474812760144506 + 358027153990999990968244906482319780943983*I)/309485009821345068724781056,     (-9755135335127734571547571921702373498554177 - 4837981372692695195747379349593041939686540*I)/2475880078570760549798248448],
#         [(-379516731607474268954110071392894274962069 - 422272153179747548473724096872271700878296*I)/77371252455336267181195264, (41324748029613152354787280677832014263339501 - 12715121258662668420833935373453570749288074*I)/1237940039285380274899124224, (-339216903907423793947110742819264306542397 + 494174755147303922029979279454787373566517*I)/77371252455336267181195264, (-18121350839962855576667529908850640619878381 - 37413012454129786092962531597292531089199003*I)/1237940039285380274899124224, (2489661087330511608618880408199633556675926 + 1137821536550153872137379935240732287260863*I)/309485009821345068724781056, (-136644109701594123227587016790354220062972119 + 110130123468183660555391413889600443583585272*I)/4951760157141521099596496896, (1488043981274920070468141664150073426459593 - 9691968079933445130866371609614474474327650*I)/1237940039285380274899124224,  27*(4636797403026872518131756991410164760195942 + 3369103221138229204457272860484005850416533*I)/4951760157141521099596496896, (-8534279107365915284081669381642269800472363 + 2241118846262661434336333368511372725482742*I)/1237940039285380274899124224,  (60923350128174260992536531692058086830950875 - 263673488093551053385865699805250505661590126*I)/9903520314283042199192993792, (18520943561240714459282253753348921824172569 + 24846649186468656345966986622110971925703604*I)/4951760157141521099596496896,  (-232781130692604829085973604213529649638644431 + 35981505277760667933017117949103953338570617*I)/9903520314283042199192993792],
#         [      (8742968295129404279528270438201520488950 + 3061473358639249112126847237482570858327*I)/4835703278458516698824704,      (-245657313712011778432792959787098074935273 + 253113767861878869678042729088355086740856*I)/38685626227668133590597632,      (1947031161734702327107371192008011621193 - 19462330079296259148177542369999791122762*I)/9671406556917033397649408,        (552856485625209001527688949522750288619217 + 392928441196156725372494335248099016686580*I)/77371252455336267181195264,      (-44542866621905323121630214897126343414629 + 3265340021421335059323962377647649632959*I)/19342813113834066795298816,          (136272594005759723105646069956434264218730 - 330975364731707309489523680957584684763587*I)/38685626227668133590597632,       (27392593965554149283318732469825168894401 + 75157071243800133880129376047131061115278*I)/38685626227668133590597632,      7*(-357821652913266734749960136017214096276154 - 45509144466378076475315751988405961498243*I)/309485009821345068724781056,       (104485001373574280824835174390219397141149 - 99041000529599568255829489765415726168162*I)/77371252455336267181195264,      (1198066993119982409323525798509037696321291 + 4249784165667887866939369628840569844519936*I)/618970019642690137449562112,       (-114985392587849953209115599084503853611014 - 52510376847189529234864487459476242883449*I)/77371252455336267181195264,      (6094620517051332877965959223269600650951573 - 4683469779240530439185019982269137976201163*I)/1237940039285380274899124224],
#         [ (611292255597977285752123848828590587708323 - 216821743518546668382662964473055912169502*I)/77371252455336267181195264,  (-1144023204575811464652692396337616594307487 + 12295317806312398617498029126807758490062855*I)/309485009821345068724781056, (-374093027769390002505693378578475235158281 - 573533923565898290299607461660384634333639*I)/77371252455336267181195264,   (47405570632186659000138546955372796986832987 - 2837476058950808941605000274055970055096534*I)/1237940039285380274899124224,   (-571573207393621076306216726219753090535121 + 533381457185823100878764749236639320783831*I)/77371252455336267181195264,     (-7096548151856165056213543560958582513797519 - 24035731898756040059329175131592138642195366*I)/618970019642690137449562112,  (2396762128833271142000266170154694033849225 + 1448501087375679588770230529017516492953051*I)/309485009821345068724781056, (-150609293845161968447166237242456473262037053 + 92581148080922977153207018003184520294188436*I)/4951760157141521099596496896, 5*(270278244730804315149356082977618054486347 - 1997830155222496880429743815321662710091562*I)/1237940039285380274899124224,   (62978424789588828258068912690172109324360330 + 44803641177219298311493356929537007630129097*I)/2475880078570760549798248448, 19*(-451431106327656743945775812536216598712236 + 114924966793632084379437683991151177407937*I)/1237940039285380274899124224,   (63417747628891221594106738815256002143915995 - 261508229397507037136324178612212080871150958*I)/9903520314283042199192993792],
#         [     (-2144231934021288786200752920446633703357 + 2305614436009705803670842248131563850246*I)/1208925819614629174706176,       (-90720949337459896266067589013987007078153 - 221951119475096403601562347412753844534569*I)/19342813113834066795298816,      (11590973613116630788176337262688659880376 + 6514520676308992726483494976339330626159*I)/4835703278458516698824704,      3*(-131776217149000326618649542018343107657237 + 79095042939612668486212006406818285287004*I)/38685626227668133590597632,       (10100577916793945997239221374025741184951 - 28631383488085522003281589065994018550748*I)/9671406556917033397649408,         67*(10090295594251078955008130473573667572549 + 10449901522697161049513326446427839676762*I)/77371252455336267181195264,       (-54270981296988368730689531355811033930513 - 3413683117592637309471893510944045467443*I)/19342813113834066795298816,         (440372322928679910536575560069973699181278 - 736603803202303189048085196176918214409081*I)/77371252455336267181195264,        (33220374714789391132887731139763250155295 + 92055083048787219934030779066298919603554*I)/38685626227668133590597632,      5*(-594638554579967244348856981610805281527116 - 82309245323128933521987392165716076704057*I)/309485009821345068724781056,       (128056368815300084550013708313312073721955 - 114619107488668120303579745393765245911404*I)/77371252455336267181195264,       21*(59839959255173222962789517794121843393573 + 241507883613676387255359616163487405826334*I)/618970019642690137449562112],
#         [ (-13454485022325376674626653802541391955147 + 184471402121905621396582628515905949793486*I)/19342813113834066795298816,   (-6158730123400322562149780662133074862437105 - 3416173052604643794120262081623703514107476*I)/154742504910672534362390528,  (770558003844914708453618983120686116100419 - 127758381209767638635199674005029818518766*I)/77371252455336267181195264,   (-4693005771813492267479835161596671660631703 + 12703585094750991389845384539501921531449948*I)/309485009821345068724781056,   (-295028157441149027913545676461260860036601 - 841544569970643160358138082317324743450770*I)/77371252455336267181195264,     (56716442796929448856312202561538574275502893 + 7216818824772560379753073185990186711454778*I)/1237940039285380274899124224,  15*(-87061038932753366532685677510172566368387 + 61306141156647596310941396434445461895538*I)/154742504910672534362390528,    (-3455315109680781412178133042301025723909347 - 24969329563196972466388460746447646686670670*I)/618970019642690137449562112,   (2453418854160886481106557323699250865361849 + 1497886802326243014471854112161398141242514*I)/309485009821345068724781056, (-151343224544252091980004429001205664193082173 + 90471883264187337053549090899816228846836628*I)/4951760157141521099596496896,   (1652018205533026103358164026239417416432989 - 9959733619236515024261775397109724431400162*I)/1237940039285380274899124224,  3*(40676374242956907656984876692623172736522006 + 31023357083037817469535762230872667581366205*I)/4951760157141521099596496896],
#         [     (-1226990509403328460274658603410696548387 - 4131739423109992672186585941938392788458*I)/1208925819614629174706176,         (162392818524418973411975140074368079662703 + 23706194236915374831230612374344230400704*I)/9671406556917033397649408,      (-3935678233089814180000602553655565621193 + 2283744757287145199688061892165659502483*I)/1208925819614629174706176,         (-2400210250844254483454290806930306285131 - 315571356806370996069052930302295432758205*I)/19342813113834066795298816,       (13365917938215281056563183751673390817910 + 15911483133819801118348625831132324863881*I)/4835703278458516698824704,        3*(-215950551370668982657516660700301003897855 + 51684341999223632631602864028309400489378*I)/38685626227668133590597632,        (20886089946811765149439844691320027184765 - 30806277083146786592790625980769214361844*I)/9671406556917033397649408,        (562180634592713285745940856221105667874855 + 1031543963988260765153550559766662245114916*I)/77371252455336267181195264,       (-65820625814810177122941758625652476012867 - 12429918324787060890804395323920477537595*I)/19342813113834066795298816,         (319147848192012911298771180196635859221089 - 402403304933906769233365689834404519960394*I)/38685626227668133590597632,        (23035615120921026080284733394359587955057 + 115351677687031786114651452775242461310624*I)/38685626227668133590597632,      (-3426830634881892756966440108592579264936130 - 1022954961164128745603407283836365128598559*I)/309485009821345068724781056],
#         [ (-192574788060137531023716449082856117537757 - 69222967328876859586831013062387845780692*I)/19342813113834066795298816,     (2736383768828013152914815341491629299773262 - 2773252698016291897599353862072533475408743*I)/77371252455336267181195264,  (-23280005281223837717773057436155921656805 + 214784953368021840006305033048142888879224*I)/19342813113834066795298816,     (-3035247484028969580570400133318947903462326 - 2195168903335435855621328554626336958674325*I)/77371252455336267181195264,     (984552428291526892214541708637840971548653 - 64006622534521425620714598573494988589378*I)/77371252455336267181195264,      (-3070650452470333005276715136041262898509903 + 7286424705750810474140953092161794621989080*I)/154742504910672534362390528,    (-147848877109756404594659513386972921139270 - 416306113044186424749331418059456047650861*I)/38685626227668133590597632,    (55272118474097814260289392337160619494260781 + 7494019668394781211907115583302403519488058*I)/1237940039285380274899124224,     (-581537886583682322424771088996959213068864 + 542191617758465339135308203815256798407429*I)/77371252455336267181195264,    (-6422548983676355789975736799494791970390991 - 23524183982209004826464749309156698827737702*I)/618970019642690137449562112,     7*(180747195387024536886923192475064903482083 + 84352527693562434817771649853047924991804*I)/154742504910672534362390528, (-135485179036717001055310712747643466592387031 + 102346575226653028836678855697782273460527608*I)/4951760157141521099596496896],
#         [        (3384238362616083147067025892852431152105 + 156724444932584900214919898954874618256*I)/604462909807314587353088,        (-59558300950677430189587207338385764871866 + 114427143574375271097298201388331237478857*I)/4835703278458516698824704,      (-1356835789870635633517710130971800616227 - 7023484098542340388800213478357340875410*I)/1208925819614629174706176,          (234884918567993750975181728413524549575881 + 79757294640629983786895695752733890213506*I)/9671406556917033397649408,        (-7632732774935120473359202657160313866419 + 2905452608512927560554702228553291839465*I)/1208925819614629174706176,           (52291747908702842344842889809762246649489 - 520996778817151392090736149644507525892649*I)/19342813113834066795298816,        (17472406829219127839967951180375981717322 + 23464704213841582137898905375041819568669*I)/4835703278458516698824704,        (-911026971811893092350229536132730760943307 + 150799318130900944080399439626714846752360*I)/38685626227668133590597632,         (26234457233977042811089020440646443590687 - 45650293039576452023692126463683727692890*I)/9671406556917033397649408,       3*(288348388717468992528382586652654351121357 + 454526517721403048270274049572136109264668*I)/77371252455336267181195264,        (-91583492367747094223295011999405657956347 - 12704691128268298435362255538069612411331*I)/19342813113834066795298816,          (411208730251327843849027957710164064354221 - 569898526380691606955496789378230959965898*I)/38685626227668133590597632],
#         [    (27127513117071487872628354831658811211795 - 37765296987901990355760582016892124833857*I)/4835703278458516698824704,     (1741779916057680444272938534338833170625435 + 3083041729779495966997526404685535449810378*I)/77371252455336267181195264, 3*(-60642236251815783728374561836962709533401 - 24630301165439580049891518846174101510744*I)/19342813113834066795298816,      3*(445885207364591681637745678755008757483408 - 350948497734812895032502179455610024541643*I)/38685626227668133590597632,    (-47373295621391195484367368282471381775684 + 219122969294089357477027867028071400054973*I)/19342813113834066795298816,       (-2801565819673198722993348253876353741520438 - 2250142129822658548391697042460298703335701*I)/77371252455336267181195264,      (801448252275607253266997552356128790317119 - 50890367688077858227059515894356594900558*I)/77371252455336267181195264,    (-5082187758525931944557763799137987573501207 + 11610432359082071866576699236013484487676124*I)/309485009821345068724781056,     (-328925127096560623794883760398247685166830 - 643447969697471610060622160899409680422019*I)/77371252455336267181195264,    15*(2954944669454003684028194956846659916299765 + 33434406416888505837444969347824812608566*I)/1237940039285380274899124224,      (-415749104352001509942256567958449835766827 + 479330966144175743357171151440020955412219*I)/77371252455336267181195264,  3*(-4639987285852134369449873547637372282914255 - 11994411888966030153196659207284951579243273*I)/1237940039285380274899124224],
#         [       (-478846096206269117345024348666145495601 + 1249092488629201351470551186322814883283*I)/302231454903657293676544,         (-17749319421930878799354766626365926894989 - 18264580106418628161818752318217357231971*I)/1208925819614629174706176,         (2801110795431528876849623279389579072819 + 363258850073786330770713557775566973248*I)/604462909807314587353088,          (-59053496693129013745775512127095650616252 + 78143588734197260279248498898321500167517*I)/4835703278458516698824704,         (-283186724922498212468162690097101115349 - 6443437753863179883794497936345437398276*I)/1208925819614629174706176,            (188799118826748909206887165661384998787543 + 84274736720556630026311383931055307398820*I)/9671406556917033397649408,         (-5482217151670072904078758141270295025989 + 1818284338672191024475557065444481298568*I)/1208925819614629174706176,          (56564463395350195513805521309731217952281 - 360208541416798112109946262159695452898431*I)/19342813113834066795298816,        11*(1259539805728870739006416869463689438068 + 1409136581547898074455004171305324917387*I)/4835703278458516698824704,       5*(-123701190701414554945251071190688818343325 + 30997157322590424677294553832111902279712*I)/38685626227668133590597632,          (16130917381301373033736295883982414239781 - 32752041297570919727145380131926943374516*I)/9671406556917033397649408,          (650301385108223834347093740500375498354925 + 899526407681131828596801223402866051809258*I)/77371252455336267181195264],
#         [      (9011388245256140876590294262420614839483 + 8167917972423946282513000869327525382672*I)/1208925819614629174706176,       (-426393174084720190126376382194036323028924 + 180692224825757525982858693158209545430621*I)/9671406556917033397649408,     (24588556702197802674765733448108154175535 - 45091766022876486566421953254051868331066*I)/4835703278458516698824704,      (1872113939365285277373877183750416985089691 + 3030392393733212574744122057679633775773130*I)/77371252455336267181195264,    (-222173405538046189185754954524429864167549 - 75193157893478637039381059488387511299116*I)/19342813113834066795298816,        (2670821320766222522963689317316937579844558 - 2645837121493554383087981511645435472169191*I)/77371252455336267181195264,     5*(-2100110309556476773796963197283876204940 + 41957457246479840487980315496957337371937*I)/19342813113834066795298816,     (-5733743755499084165382383818991531258980593 - 3328949988392698205198574824396695027195732*I)/154742504910672534362390528,      (707827994365259025461378911159398206329247 - 265730616623227695108042528694302299777294*I)/77371252455336267181195264,    (-1442501604682933002895864804409322823788319 + 11504137805563265043376405214378288793343879*I)/309485009821345068724781056,         (-56130472299445561499538726459719629522285 - 61117552419727805035810982426639329818864*I)/9671406556917033397649408,    (39053692321126079849054272431599539429908717 - 10209127700342570953247177602860848130710666*I)/1237940039285380274899124224]])
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,      1/4 - 5*I/16,      65/128 + 87*I/64],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,  85/256 - 33*I/16,  805/128 + 2415*I/512],
        [          1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64],
        [   -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128],
        [            1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16],
        [         21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M**10 == Matrix(S('''[
            [     7369525394972778926719607798014571861/604462909807314587353088 - 229284202061790301477392339912557559*I/151115727451828646838272,   -19704281515163975949388435612632058035/1208925819614629174706176 + 14319858347987648723768698170712102887*I/302231454903657293676544,      -3623281909451783042932142262164941211/604462909807314587353088 - 6039240602494288615094338643452320495*I/604462909807314587353088,    109260497799140408739847239685705357695/2417851639229258349412352 - 7427566006564572463236368211555511431*I/2417851639229258349412352, -16095803767674394244695716092817006641/2417851639229258349412352 + 10336681897356760057393429626719177583*I/1208925819614629174706176,    -42207883340488041844332828574359769743/2417851639229258349412352 - 182332262671671273188016400290188468499*I/4835703278458516698824704],
            [50566491050825573392726324995779608259/1208925819614629174706176 - 90047007594468146222002432884052362145*I/2417851639229258349412352,  74273703462900000967697427843983822011/1208925819614629174706176 + 265947522682943571171988741842776095421*I/1208925819614629174706176, -116900341394390200556829767923360888429/2417851639229258349412352 - 53153263356679268823910621474478756845*I/2417851639229258349412352, 195407378023867871243426523048612490249/1208925819614629174706176 - 1242417915995360200584837585002906728929*I/9671406556917033397649408,   -863597594389821970177319682495878193/302231454903657293676544 + 476936100741548328800725360758734300481*I/9671406556917033397649408, -3154451590535653853562472176601754835575/19342813113834066795298816 - 232909875490506237386836489998407329215*I/2417851639229258349412352],
            [   -1715444997702484578716037230949868543/302231454903657293676544 + 5009695651321306866158517287924120777*I/302231454903657293676544,     -30551582497996879620371947949342101301/604462909807314587353088 - 7632518367986526187139161303331519629*I/151115727451828646838272,           312680739924495153190604170938220575/18889465931478580854784 - 108664334509328818765959789219208459*I/75557863725914323419136,    -14693696966703036206178521686918865509/604462909807314587353088 + 72345386220900843930147151999899692401*I/1208925819614629174706176,  -8218872496728882299722894680635296519/1208925819614629174706176 - 16776782833358893712645864791807664983*I/1208925819614629174706176,      143237839169380078671242929143670635137/2417851639229258349412352 + 2883817094806115974748882735218469447*I/2417851639229258349412352],
            [   3087979417831061365023111800749855987/151115727451828646838272 + 34441942370802869368851419102423997089*I/604462909807314587353088, -148309181940158040917731426845476175667/604462909807314587353088 - 263987151804109387844966835369350904919*I/9671406556917033397649408,   50259518594816377378747711930008883165/1208925819614629174706176 - 95713974916869240305450001443767979653*I/2417851639229258349412352,  153466447023875527996457943521467271119/2417851639229258349412352 + 517285524891117105834922278517084871349*I/2417851639229258349412352,  -29184653615412989036678939366291205575/604462909807314587353088 - 27551322282526322041080173287022121083*I/1208925819614629174706176,   196404220110085511863671393922447671649/1208925819614629174706176 - 1204712019400186021982272049902206202145*I/9671406556917033397649408],
            [     -2632581805949645784625606590600098779/151115727451828646838272 - 589957435912868015140272627522612771*I/37778931862957161709568,     26727850893953715274702844733506310247/302231454903657293676544 - 10825791956782128799168209600694020481*I/302231454903657293676544,      -1036348763702366164044671908440791295/151115727451828646838272 + 3188624571414467767868303105288107375*I/151115727451828646838272,     -36814959939970644875593411585393242449/604462909807314587353088 - 18457555789119782404850043842902832647*I/302231454903657293676544,      12454491297984637815063964572803058647/604462909807314587353088 - 340489532842249733975074349495329171*I/302231454903657293676544,      -19547211751145597258386735573258916681/604462909807314587353088 + 87299583775782199663414539883938008933*I/1208925819614629174706176],
            [  -40281994229560039213253423262678393183/604462909807314587353088 - 2939986850065527327299273003299736641*I/604462909807314587353088, 331940684638052085845743020267462794181/2417851639229258349412352 - 284574901963624403933361315517248458969*I/1208925819614629174706176,      6453843623051745485064693628073010961/302231454903657293676544 + 36062454107479732681350914931391590957*I/604462909807314587353088,  -147665869053634695632880753646441962067/604462909807314587353088 - 305987938660447291246597544085345123927*I/9671406556917033397649408,  107821369195275772166593879711259469423/2417851639229258349412352 - 11645185518211204108659001435013326687*I/302231454903657293676544,     64121228424717666402009446088588091619/1208925819614629174706176 + 265557133337095047883844369272389762133*I/1208925819614629174706176]]'''))

def test_issue_17247_expression_blowup_5():
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    with dotprodsimp(True):
        assert M.charpoly('x') == PurePoly(x**6 + (-6 - 6*I)*x**5 + 36*I*x**4, x, domain='EX')

def test_issue_17247_expression_blowup_6():
    M = Matrix(8, 8, [x+i for i in range (64)])
    with dotprodsimp(True):
        assert M.det('bareiss') == 0

def test_issue_17247_expression_blowup_7():
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    with dotprodsimp(True):
        assert M.det('berkowitz') == 0

def test_issue_17247_expression_blowup_8():
    M = Matrix(8, 8, [x+i for i in range (64)])
    with dotprodsimp(True):
        assert M.det('lu') == 0

def test_issue_17247_expression_blowup_9():
    M = Matrix(8, 8, [x+i for i in range (64)])
    with dotprodsimp(True):
        assert M.rref() == (Matrix([
            [1, 0, -1, -2, -3, -4, -5, -6],
            [0, 1,  2,  3,  4,  5,  6,  7],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0]]), (0, 1))

def test_issue_17247_expression_blowup_10():
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    with dotprodsimp(True):
        assert M.cofactor(0, 0) == 0

def test_issue_17247_expression_blowup_11():
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    with dotprodsimp(True):
        assert M.cofactor_matrix() == Matrix(6, 6, [0]*36)

def test_issue_17247_expression_blowup_12():
    M = Matrix(6, 6, lambda i, j: 1 + (-1)**(i+j)*I)
    with dotprodsimp(True):
        assert M.eigenvals() == {6: 1, 6*I: 1, 0: 4}

def test_issue_17247_expression_blowup_13():
    M = Matrix([
        [    0, 1 - x, x + 1, 1 - x],
        [1 - x, x + 1,     0, x + 1],
        [    0, 1 - x, x + 1, 1 - x],
        [    0,     0,     1 - x, 0]])

    ev = M.eigenvects()
    assert ev[0] == (0, 2, [Matrix([0, -1, 0, 1])])
    assert ev[1][0] == x - sqrt(2)*(x - 1) + 1
    assert ev[1][1] == 1
    assert ev[1][2][0].expand(deep=False, numer=True) == Matrix([
        [(-x + sqrt(2)*(x - 1) - 1)/(x - 1)],
        [-4*x/(x**2 - 2*x + 1) + (x + 1)*(x - sqrt(2)*(x - 1) + 1)/(x**2 - 2*x + 1)],
        [(-x + sqrt(2)*(x - 1) - 1)/(x - 1)],
        [1]
    ])

    assert ev[2][0] == x + sqrt(2)*(x - 1) + 1
    assert ev[2][1] == 1
    assert ev[2][2][0].expand(deep=False, numer=True) == Matrix([
        [(-x - sqrt(2)*(x - 1) - 1)/(x - 1)],
        [-4*x/(x**2 - 2*x + 1) + (x + 1)*(x + sqrt(2)*(x - 1) + 1)/(x**2 - 2*x + 1)],
        [(-x - sqrt(2)*(x - 1) - 1)/(x - 1)],
        [1]
    ])


def test_issue_17247_expression_blowup_14():
    M = Matrix(8, 8, ([1+x, 1-x]*4 + [1-x, 1+x]*4)*4)
    with dotprodsimp(True):
        assert M.echelon_form() == Matrix([
            [x + 1, 1 - x, x + 1, 1 - x, x + 1, 1 - x, x + 1, 1 - x],
            [    0,   4*x,     0,   4*x,     0,   4*x,     0,   4*x],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0],
            [    0,     0,     0,     0,     0,     0,     0,     0]])

def test_issue_17247_expression_blowup_15():
    M = Matrix(8, 8, ([1+x, 1-x]*4 + [1-x, 1+x]*4)*4)
    with dotprodsimp(True):
        assert M.rowspace() == [Matrix([[x + 1, 1 - x, x + 1, 1 - x, x + 1, 1 - x, x + 1, 1 - x]]), Matrix([[0, 4*x, 0, 4*x, 0, 4*x, 0, 4*x]])]

def test_issue_17247_expression_blowup_16():
    M = Matrix(8, 8, ([1+x, 1-x]*4 + [1-x, 1+x]*4)*4)
    with dotprodsimp(True):
        assert M.columnspace() == [Matrix([[x + 1],[1 - x],[x + 1],[1 - x],[x + 1],[1 - x],[x + 1],[1 - x]]), Matrix([[1 - x],[x + 1],[1 - x],[x + 1],[1 - x],[x + 1],[1 - x],[x + 1]])]

def test_issue_17247_expression_blowup_17():
    M = Matrix(8, 8, [x+i for i in range (64)])
    with dotprodsimp(True):
        assert M.nullspace() == [
            Matrix([[1],[-2],[1],[0],[0],[0],[0],[0]]),
            Matrix([[2],[-3],[0],[1],[0],[0],[0],[0]]),
            Matrix([[3],[-4],[0],[0],[1],[0],[0],[0]]),
            Matrix([[4],[-5],[0],[0],[0],[1],[0],[0]]),
            Matrix([[5],[-6],[0],[0],[0],[0],[1],[0]]),
            Matrix([[6],[-7],[0],[0],[0],[0],[0],[1]])]

def test_issue_17247_expression_blowup_18():
    M = Matrix(6, 6, ([1+x, 1-x]*3 + [1-x, 1+x]*3)*3)
    with dotprodsimp(True):
        assert not M.is_nilpotent()

def test_issue_17247_expression_blowup_19():
    M = Matrix(S('''[
        [             -3/4,                     0,         1/4 + I/2,                     0],
        [                0, -177/128 - 1369*I/128,                 0, -2063/256 + 541*I/128],
        [          1/2 - I,                     0,                 0,                     0],
        [                0,                     0,                 0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert not M.is_diagonalizable()

def test_issue_17247_expression_blowup_20():
    M = Matrix([
    [x + 1,  1 - x,      0,      0],
    [1 - x,  x + 1,      0,  x + 1],
    [    0,  1 - x,  x + 1,      0],
    [    0,      0,      0,  x + 1]])
    with dotprodsimp(True):
        assert M.diagonalize() == (Matrix([
            [1,  1, 0, (x + 1)/(x - 1)],
            [1, -1, 0,               0],
            [1,  1, 1,               0],
            [0,  0, 0,               1]]),
            Matrix([
            [2,   0,     0,     0],
            [0, 2*x,     0,     0],
            [0,   0, x + 1,     0],
            [0,   0,     0, x + 1]]))

def test_issue_17247_expression_blowup_21():
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.inv(method='GE') == Matrix(S('''[
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))

def test_issue_17247_expression_blowup_22():
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.inv(method='LU') == Matrix(S('''[
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))

def test_issue_17247_expression_blowup_23():
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.inv(method='ADJ').expand() == Matrix(S('''[
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))

def test_issue_17247_expression_blowup_24():
    M = SparseMatrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.inv(method='CH') == Matrix(S('''[
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))

def test_issue_17247_expression_blowup_25():
    M = SparseMatrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.inv(method='LDL') == Matrix(S('''[
            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],
            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],
            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],
            [0, 0, 0, -11328/952745 + 87616*I/952745]]'''))

def test_issue_17247_expression_blowup_26():
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,      1/4 - 5*I/16,      65/128 + 87*I/64,         -9/32 - I/16,      183/256 - 97*I/128],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,  85/256 - 33*I/16,  805/128 + 2415*I/512, -219/128 + 115*I/256, 6301/4096 - 6609*I/1024],
        [          1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,         1/4 - 5*I/16,        65/128 + 87*I/64],
        [   -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,     85/256 - 33*I/16,    805/128 + 2415*I/512],
        [            1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,            1/4 + I/2,        -129/64 - 9*I/64],
        [         21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,     125/64 + 87*I/64,   -2063/256 + 541*I/128],
        [               -2,         17/4 - 13*I/2,             1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,                 -3/4,         45/32 - 37*I/16],
        [     1/4 + 13*I/4,    -825/64 - 147*I/32,          21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64,    -149/64 + 49*I/32,   -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.rank() == 4

def test_issue_17247_expression_blowup_27():
    M = Matrix([
        [    0, 1 - x, x + 1, 1 - x],
        [1 - x, x + 1,     0, x + 1],
        [    0, 1 - x, x + 1, 1 - x],
        [    0,     0,     1 - x, 0]])
    with dotprodsimp(True):
        P, J = M.jordan_form()
        assert P.expand() == Matrix(S('''[
            [    0,  4*x/(x**2 - 2*x + 1), -(-17*x**4 + 12*sqrt(2)*x**4 - 4*sqrt(2)*x**3 + 6*x**3 - 6*x - 4*sqrt(2)*x + 12*sqrt(2) + 17)/(-7*x**4 + 5*sqrt(2)*x**4 - 6*sqrt(2)*x**3 + 8*x**3 - 2*x**2 + 8*x + 6*sqrt(2)*x - 5*sqrt(2) - 7), -(12*sqrt(2)*x**4 + 17*x**4 - 6*x**3 - 4*sqrt(2)*x**3 - 4*sqrt(2)*x + 6*x - 17 + 12*sqrt(2))/(7*x**4 + 5*sqrt(2)*x**4 - 6*sqrt(2)*x**3 - 8*x**3 + 2*x**2 - 8*x + 6*sqrt(2)*x - 5*sqrt(2) + 7)],
            [x - 1, x/(x - 1) + 1/(x - 1),                       (-7*x**3 + 5*sqrt(2)*x**3 - x**2 + sqrt(2)*x**2 - sqrt(2)*x - x - 5*sqrt(2) - 7)/(-3*x**3 + 2*sqrt(2)*x**3 - 2*sqrt(2)*x**2 + 3*x**2 + 2*sqrt(2)*x + 3*x - 3 - 2*sqrt(2)),                       (7*x**3 + 5*sqrt(2)*x**3 + x**2 + sqrt(2)*x**2 - sqrt(2)*x + x - 5*sqrt(2) + 7)/(2*sqrt(2)*x**3 + 3*x**3 - 3*x**2 - 2*sqrt(2)*x**2 - 3*x + 2*sqrt(2)*x - 2*sqrt(2) + 3)],
            [    0,                     1,                                                                                            -(-3*x**2 + 2*sqrt(2)*x**2 + 2*x - 3 - 2*sqrt(2))/(-x**2 + sqrt(2)*x**2 - 2*sqrt(2)*x + 1 + sqrt(2)),                                                                                            -(2*sqrt(2)*x**2 + 3*x**2 - 2*x - 2*sqrt(2) + 3)/(x**2 + sqrt(2)*x**2 - 2*sqrt(2)*x - 1 + sqrt(2))],
            [1 - x,                     0,                                                                                                                                                                                               1,                                                                                                                                                                                             1]]''')).expand()
        assert J == Matrix(S('''[
            [0, 1,                       0,                       0],
            [0, 0,                       0,                       0],
            [0, 0, x - sqrt(2)*(x - 1) + 1,                       0],
            [0, 0,                       0, x + sqrt(2)*(x - 1) + 1]]'''))

def test_issue_17247_expression_blowup_28():
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.singular_values() == S('''[
            sqrt(14609315/131072 + sqrt(64789115132571/2147483648 - 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3) + 76627253330829751075/(35184372088832*sqrt(64789115132571/4294967296 + 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)) + 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3))) - 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)))/2 + sqrt(64789115132571/4294967296 + 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)) + 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3))/2),
            sqrt(14609315/131072 - sqrt(64789115132571/2147483648 - 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3) + 76627253330829751075/(35184372088832*sqrt(64789115132571/4294967296 + 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)) + 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3))) - 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)))/2 + sqrt(64789115132571/4294967296 + 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)) + 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3))/2),
            sqrt(14609315/131072 - sqrt(64789115132571/4294967296 + 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)) + 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3))/2 + sqrt(64789115132571/2147483648 - 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3) - 76627253330829751075/(35184372088832*sqrt(64789115132571/4294967296 + 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)) + 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3))) - 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)))/2),
            sqrt(14609315/131072 - sqrt(64789115132571/4294967296 + 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)) + 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3))/2 - sqrt(64789115132571/2147483648 - 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3) - 76627253330829751075/(35184372088832*sqrt(64789115132571/4294967296 + 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)) + 2*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3))) - 3546944054712886603889144627/(110680464442257309696*(25895222463957462655758224991455280215303/633825300114114700748351602688 + sqrt(1213909058710955930446995195883114969038524625997915131236390724543989220134670)*I/22282920707136844948184236032)**(1/3)))/2)]''')


def test_issue_16823():
    # This still needs to be fixed if not using dotprodsimp.
    M = Matrix(S('''[
        [1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I,-9/32-1/16*I,183/256-97/128*I,3/64+13/64*I,-23/32-59/256*I,15/128-3/32*I,19/256+551/1024*I],
        [21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I,-219/128+115/256*I,6301/4096-6609/1024*I,119/128+143/128*I,-10879/2048+4343/4096*I,129/256-549/512*I,42533/16384+29103/8192*I],
        [-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I,-9/32-1/16*I,183/256-97/128*I,3/64+13/64*I,-23/32-59/256*I],
        [1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I,-219/128+115/256*I,6301/4096-6609/1024*I,119/128+143/128*I,-10879/2048+4343/4096*I],
        [-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I,-9/32-1/16*I,183/256-97/128*I],
        [1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I,-219/128+115/256*I,6301/4096-6609/1024*I],
        [-4,9-5*I,-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I],
        [-2*I,119/8+29/4*I,1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I],
        [0,-6,-4,9-5*I,-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I],
        [1,-9/4+3*I,-2*I,119/8+29/4*I,1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I],
        [0,-4*I,0,-6,-4,9-5*I,-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I],
        [0,1/4+1/2*I,1,-9/4+3*I,-2*I,119/8+29/4*I,1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I]]'''))
    with dotprodsimp(True):
        assert M.rank() == 8


def test_issue_18531():
    # solve_linear_system still needs fixing but the rref works.
    M = Matrix([
        [1, 1, 1, 1, 1, 0, 1, 0, 0],
        [1 + sqrt(2), -1 + sqrt(2), 1 - sqrt(2), -sqrt(2) - 1, 1, 1, -1, 1, 1],
        [-5 + 2*sqrt(2), -5 - 2*sqrt(2), -5 - 2*sqrt(2), -5 + 2*sqrt(2), -7, 2, -7, -2, 0],
        [-3*sqrt(2) - 1, 1 - 3*sqrt(2), -1 + 3*sqrt(2), 1 + 3*sqrt(2), -7, -5, 7, -5, 3],
        [7 - 4*sqrt(2), 4*sqrt(2) + 7, 4*sqrt(2) + 7, 7 - 4*sqrt(2), 7, -12, 7, 12, 0],
        [-1 + 3*sqrt(2), 1 + 3*sqrt(2), -3*sqrt(2) - 1, 1 - 3*sqrt(2), 7, -5, -7, -5, 3],
        [-3 + 2*sqrt(2), -3 - 2*sqrt(2), -3 - 2*sqrt(2), -3 + 2*sqrt(2), -1, 2, -1, -2, 0],
        [1 - sqrt(2), -sqrt(2) - 1, 1 + sqrt(2), -1 + sqrt(2), -1, 1, 1, 1, 1]
        ])
    with dotprodsimp(True):
        assert M.rref() == (Matrix([
            [1, 0, 0, 0, 0, 0, 0, 0,  S(1)/2],
            [0, 1, 0, 0, 0, 0, 0, 0, -S(1)/2],
            [0, 0, 1, 0, 0, 0, 0, 0,  S(1)/2],
            [0, 0, 0, 1, 0, 0, 0, 0, -S(1)/2],
            [0, 0, 0, 0, 1, 0, 0, 0,    0],
            [0, 0, 0, 0, 0, 1, 0, 0, -S(1)/2],
            [0, 0, 0, 0, 0, 0, 1, 0,    0],
            [0, 0, 0, 0, 0, 0, 0, 1, -S(1)/2]]), (0, 1, 2, 3, 4, 5, 6, 7))


def test_creation():
    raises(ValueError, lambda: Matrix(5, 5, range(20)))
    raises(ValueError, lambda: Matrix(5, -1, []))
    raises(IndexError, lambda: Matrix((1, 2))[2])
    with raises(IndexError):
        Matrix((1, 2))[3] = 5

    assert Matrix() == Matrix([]) == Matrix(0, 0, [])
    assert Matrix([[]]) == Matrix(1, 0, [])
    assert Matrix([[], []]) == Matrix(2, 0, [])

    # anything used to be allowed in a matrix
    with warns_deprecated_sympy():
        assert Matrix([[[1], (2,)]]).tolist() == [[[1], (2,)]]
    with warns_deprecated_sympy():
        assert Matrix([[[1], (2,)]]).T.tolist() == [[[1]], [(2,)]]
    M = Matrix([[0]])
    with warns_deprecated_sympy():
        M[0, 0] = S.EmptySet

    a = Matrix([[x, 0], [0, 0]])
    m = a
    assert m.cols == m.rows
    assert m.cols == 2
    assert m[:] == [x, 0, 0, 0]

    b = Matrix(2, 2, [x, 0, 0, 0])
    m = b
    assert m.cols == m.rows
    assert m.cols == 2
    assert m[:] == [x, 0, 0, 0]

    assert a == b

    assert Matrix(b) == b

    c23 = Matrix(2, 3, range(1, 7))
    c13 = Matrix(1, 3, range(7, 10))
    c = Matrix([c23, c13])
    assert c.cols == 3
    assert c.rows == 3
    assert c[:] == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert Matrix(eye(2)) == eye(2)
    assert ImmutableMatrix(ImmutableMatrix(eye(2))) == ImmutableMatrix(eye(2))
    assert ImmutableMatrix(c) == c.as_immutable()
    assert Matrix(ImmutableMatrix(c)) == ImmutableMatrix(c).as_mutable()

    assert c is not Matrix(c)

    dat = [[ones(3,2), ones(3,3)*2], [ones(2,3)*3, ones(2,2)*4]]
    M = Matrix(dat)
    assert M == Matrix([
        [1, 1, 2, 2, 2],
        [1, 1, 2, 2, 2],
        [1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4],
        [3, 3, 3, 4, 4]])
    assert M.tolist() != dat
    # keep block form if evaluate=False
    assert Matrix(dat, evaluate=False).tolist() == dat
    A = MatrixSymbol("A", 2, 2)
    dat = [ones(2), A]
    assert Matrix(dat) == Matrix([
    [      1,       1],
    [      1,       1],
    [A[0, 0], A[0, 1]],
    [A[1, 0], A[1, 1]]])
    with warns_deprecated_sympy():
        assert Matrix(dat, evaluate=False).tolist() == [[i] for i in dat]

    # 0-dim tolerance
    assert Matrix([ones(2), ones(0)]) == Matrix([ones(2)])
    raises(ValueError, lambda: Matrix([ones(2), ones(0, 3)]))
    raises(ValueError, lambda: Matrix([ones(2), ones(3, 0)]))

    # mix of Matrix and iterable
    M = Matrix([[1, 2], [3, 4]])
    M2 = Matrix([M, (5, 6)])
    assert M2 == Matrix([[1, 2], [3, 4], [5, 6]])


def test_irregular_block():
    assert Matrix.irregular(3, ones(2,1), ones(3,3)*2, ones(2,2)*3,
        ones(1,1)*4, ones(2,2)*5, ones(1,2)*6, ones(1,2)*7) == Matrix([
        [1, 2, 2, 2, 3, 3],
        [1, 2, 2, 2, 3, 3],
        [4, 2, 2, 2, 5, 5],
        [6, 6, 7, 7, 5, 5]])


def test_tolist():
    lst = [[S.One, S.Half, x*y, S.Zero], [x, y, z, x**2], [y, -S.One, z*x, 3]]
    m = Matrix(lst)
    assert m.tolist() == lst


def test_as_mutable():
    assert zeros(0, 3).as_mutable() == zeros(0, 3)
    assert zeros(0, 3).as_immutable() == ImmutableMatrix(zeros(0, 3))
    assert zeros(3, 0).as_immutable() == ImmutableMatrix(zeros(3, 0))


def test_slicing():
    m0 = eye(4)
    assert m0[:3, :3] == eye(3)
    assert m0[2:4, 0:2] == zeros(2)

    m1 = Matrix(3, 3, lambda i, j: i + j)
    assert m1[0, :] == Matrix(1, 3, (0, 1, 2))
    assert m1[1:3, 1] == Matrix(2, 1, (2, 3))

    m2 = Matrix([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    assert m2[:, -1] == Matrix(4, 1, [3, 7, 11, 15])
    assert m2[-2:, :] == Matrix([[8, 9, 10, 11], [12, 13, 14, 15]])


def test_submatrix_assignment():
    m = zeros(4)
    m[2:4, 2:4] = eye(2)
    assert m == Matrix(((0, 0, 0, 0),
                        (0, 0, 0, 0),
                        (0, 0, 1, 0),
                        (0, 0, 0, 1)))
    m[:2, :2] = eye(2)
    assert m == eye(4)
    m[:, 0] = Matrix(4, 1, (1, 2, 3, 4))
    assert m == Matrix(((1, 0, 0, 0),
                        (2, 1, 0, 0),
                        (3, 0, 1, 0),
                        (4, 0, 0, 1)))
    m[:, :] = zeros(4)
    assert m == zeros(4)
    m[:, :] = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)]
    assert m == Matrix(((1, 2, 3, 4),
                        (5, 6, 7, 8),
                        (9, 10, 11, 12),
                        (13, 14, 15, 16)))
    m[:2, 0] = [0, 0]
    assert m == Matrix(((0, 2, 3, 4),
                        (0, 6, 7, 8),
                        (9, 10, 11, 12),
                        (13, 14, 15, 16)))


def test_extract():
    m = Matrix(4, 3, lambda i, j: i*3 + j)
    assert m.extract([0, 1, 3], [0, 1]) == Matrix(3, 2, [0, 1, 3, 4, 9, 10])
    assert m.extract([0, 3], [0, 0, 2]) == Matrix(2, 3, [0, 0, 2, 9, 9, 11])
    assert m.extract(range(4), range(3)) == m
    raises(IndexError, lambda: m.extract([4], [0]))
    raises(IndexError, lambda: m.extract([0], [3]))


def test_reshape():
    m0 = eye(3)
    assert m0.reshape(1, 9) == Matrix(1, 9, (1, 0, 0, 0, 1, 0, 0, 0, 1))
    m1 = Matrix(3, 4, lambda i, j: i + j)
    assert m1.reshape(
        4, 3) == Matrix(((0, 1, 2), (3, 1, 2), (3, 4, 2), (3, 4, 5)))
    assert m1.reshape(2, 6) == Matrix(((0, 1, 2, 3, 1, 2), (3, 4, 2, 3, 4, 5)))


def test_applyfunc():
    m0 = eye(3)
    assert m0.applyfunc(lambda x: 2*x) == eye(3)*2
    assert m0.applyfunc(lambda x: 0) == zeros(3)


def test_expand():
    m0 = Matrix([[x*(x + y), 2], [((x + y)*y)*x, x*(y + x*(x + y))]])
    # Test if expand() returns a matrix
    m1 = m0.expand()
    assert m1 == Matrix(
        [[x*y + x**2, 2], [x*y**2 + y*x**2, x*y + y*x**2 + x**3]])

    a = Symbol('a', real=True)

    assert Matrix([exp(I*a)]).expand(complex=True) == \
        Matrix([cos(a) + I*sin(a)])

    assert Matrix([[0, 1, 2], [0, 0, -1], [0, 0, 0]]).exp() == Matrix([
        [1, 1, Rational(3, 2)],
        [0, 1, -1],
        [0, 0, 1]]
    )

def test_refine():
    m0 = Matrix([[Abs(x)**2, sqrt(x**2)],
                [sqrt(x**2)*Abs(y)**2, sqrt(y**2)*Abs(x)**2]])
    m1 = m0.refine(Q.real(x) & Q.real(y))
    assert m1 == Matrix([[x**2, Abs(x)], [y**2*Abs(x), x**2*Abs(y)]])

    m1 = m0.refine(Q.positive(x) & Q.positive(y))
    assert m1 == Matrix([[x**2, x], [x*y**2, x**2*y]])

    m1 = m0.refine(Q.negative(x) & Q.negative(y))
    assert m1 == Matrix([[x**2, -x], [-x*y**2, -x**2*y]])

def test_random():
    M = randMatrix(3, 3)
    M = randMatrix(3, 3, seed=3)
    assert M == randMatrix(3, 3, seed=3)

    M = randMatrix(3, 4, 0, 150)
    M = randMatrix(3, seed=4, symmetric=True)
    assert M == randMatrix(3, seed=4, symmetric=True)

    S = M.copy()
    S.simplify()
    assert S == M  # doesn't fail when elements are Numbers, not int

    rng = random.Random(4)
    assert M == randMatrix(3, symmetric=True, prng=rng)

    # Ensure symmetry
    for size in (10, 11): # Test odd and even
        for percent in (100, 70, 30):
            M = randMatrix(size, symmetric=True, percent=percent, prng=rng)
            assert M == M.T

    M = randMatrix(10, min=1, percent=70)
    zero_count = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == 0:
                zero_count += 1
    assert zero_count == 30

def test_inverse():
    A = eye(4)
    assert A.inv() == eye(4)
    assert A.inv(method="LU") == eye(4)
    assert A.inv(method="ADJ") == eye(4)
    assert A.inv(method="CH") == eye(4)
    assert A.inv(method="LDL") == eye(4)
    assert A.inv(method="QR") == eye(4)
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    Ainv = A.inv()
    assert A*Ainv == eye(3)
    assert A.inv(method="LU") == Ainv
    assert A.inv(method="ADJ") == Ainv
    assert A.inv(method="CH") == Ainv
    assert A.inv(method="LDL") == Ainv
    assert A.inv(method="QR") == Ainv

    AA = Matrix([[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0]])
    assert AA.inv(method="BLOCK") * AA == eye(AA.shape[0])
    # test that immutability is not a problem
    cls = ImmutableMatrix
    m = cls([[48, 49, 31],
             [ 9, 71, 94],
             [59, 28, 65]])
    assert all(type(m.inv(s)) is cls for s in 'GE ADJ LU CH LDL QR'.split())
    cls = ImmutableSparseMatrix
    m = cls([[48, 49, 31],
             [ 9, 71, 94],
             [59, 28, 65]])
    assert all(type(m.inv(s)) is cls for s in 'GE ADJ LU CH LDL QR'.split())


def test_jacobian_hessian():
    L = Matrix(1, 2, [x**2*y, 2*y**2 + x*y])
    syms = [x, y]
    assert L.jacobian(syms) == Matrix([[2*x*y, x**2], [y, 4*y + x]])

    L = Matrix(1, 2, [x, x**2*y**3])
    assert L.jacobian(syms) == Matrix([[1, 0], [2*x*y**3, x**2*3*y**2]])

    f = x**2*y
    syms = [x, y]
    assert hessian(f, syms) == Matrix([[2*y, 2*x], [2*x, 0]])

    f = x**2*y**3
    assert hessian(f, syms) == \
        Matrix([[2*y**3, 6*x*y**2], [6*x*y**2, 6*x**2*y]])

    f = z + x*y**2
    g = x**2 + 2*y**3
    ans = Matrix([[0,   2*y],
                  [2*y, 2*x]])
    assert ans == hessian(f, Matrix([x, y]))
    assert ans == hessian(f, Matrix([x, y]).T)
    assert hessian(f, (y, x), [g]) == Matrix([
        [     0, 6*y**2, 2*x],
        [6*y**2,    2*x, 2*y],
        [   2*x,    2*y,   0]])


def test_wronskian():
    assert wronskian([cos(x), sin(x)], x) == cos(x)**2 + sin(x)**2
    assert wronskian([exp(x), exp(2*x)], x) == exp(3*x)
    assert wronskian([exp(x), x], x) == exp(x) - x*exp(x)
    assert wronskian([1, x, x**2], x) == 2
    w1 = -6*exp(x)*sin(x)*x + 6*cos(x)*exp(x)*x**2 - 6*exp(x)*cos(x)*x - \
        exp(x)*cos(x)*x**3 + exp(x)*sin(x)*x**3
    assert wronskian([exp(x), cos(x), x**3], x).expand() == w1
    assert wronskian([exp(x), cos(x), x**3], x, method='berkowitz').expand() \
        == w1
    w2 = -x**3*cos(x)**2 - x**3*sin(x)**2 - 6*x*cos(x)**2 - 6*x*sin(x)**2
    assert wronskian([sin(x), cos(x), x**3], x).expand() == w2
    assert wronskian([sin(x), cos(x), x**3], x, method='berkowitz').expand() \
        == w2
    assert wronskian([], x) == 1


def test_subs():
    assert Matrix([[1, x], [x, 4]]).subs(x, 5) == Matrix([[1, 5], [5, 4]])
    assert Matrix([[x, 2], [x + y, 4]]).subs([[x, -1], [y, -2]]) == \
        Matrix([[-1, 2], [-3, 4]])
    assert Matrix([[x, 2], [x + y, 4]]).subs([(x, -1), (y, -2)]) == \
        Matrix([[-1, 2], [-3, 4]])
    assert Matrix([[x, 2], [x + y, 4]]).subs({x: -1, y: -2}) == \
        Matrix([[-1, 2], [-3, 4]])
    assert Matrix([x*y]).subs({x: y - 1, y: x - 1}, simultaneous=True) == \
        Matrix([(x - 1)*(y - 1)])

    for cls in classes:
        assert Matrix([[2, 0], [0, 2]]) == cls.eye(2).subs(1, 2)

def test_xreplace():
    assert Matrix([[1, x], [x, 4]]).xreplace({x: 5}) == \
        Matrix([[1, 5], [5, 4]])
    assert Matrix([[x, 2], [x + y, 4]]).xreplace({x: -1, y: -2}) == \
        Matrix([[-1, 2], [-3, 4]])
    for cls in classes:
        assert Matrix([[2, 0], [0, 2]]) == cls.eye(2).xreplace({1: 2})

def test_simplify():
    n = Symbol('n')
    f = Function('f')

    M = Matrix([[            1/x + 1/y,                 (x + x*y) / x  ],
                [ (f(x) + y*f(x))/f(x), 2 * (1/n - cos(n * pi)/n) / pi ]])
    M.simplify()
    assert M == Matrix([[ (x + y)/(x * y),                        1 + y ],
                        [           1 + y, 2*((1 - 1*cos(pi*n))/(pi*n)) ]])
    eq = (1 + x)**2
    M = Matrix([[eq]])
    M.simplify()
    assert M == Matrix([[eq]])
    M.simplify(ratio=oo)
    assert M == Matrix([[eq.simplify(ratio=oo)]])


def test_transpose():
    M = Matrix([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    assert M.T == Matrix( [ [1, 1],
                            [2, 2],
                            [3, 3],
                            [4, 4],
                            [5, 5],
                            [6, 6],
                            [7, 7],
                            [8, 8],
                            [9, 9],
                            [0, 0] ])
    assert M.T.T == M
    assert M.T == M.transpose()


def test_conjugate():
    M = Matrix([[0, I, 5],
                [1, 2, 0]])

    assert M.T == Matrix([[0, 1],
                          [I, 2],
                          [5, 0]])

    assert M.C == Matrix([[0, -I, 5],
                          [1,  2, 0]])
    assert M.C == M.conjugate()

    assert M.H == M.T.C
    assert M.H == Matrix([[ 0, 1],
                          [-I, 2],
                          [ 5, 0]])


def test_conj_dirac():
    raises(AttributeError, lambda: eye(3).D)

    M = Matrix([[1, I, I, I],
                [0, 1, I, I],
                [0, 0, 1, I],
                [0, 0, 0, 1]])

    assert M.D == Matrix([[ 1,  0,  0,  0],
                          [-I,  1,  0,  0],
                          [-I, -I, -1,  0],
                          [-I, -I,  I, -1]])


def test_trace():
    M = Matrix([[1, 0, 0],
                [0, 5, 0],
                [0, 0, 8]])
    assert M.trace() == 14


def test_shape():
    M = Matrix([[x, 0, 0],
                [0, y, 0]])
    assert M.shape == (2, 3)


def test_col_row_op():
    M = Matrix([[x, 0, 0],
                [0, y, 0]])
    M.row_op(1, lambda r, j: r + j + 1)
    assert M == Matrix([[x,     0, 0],
                        [1, y + 2, 3]])

    M.col_op(0, lambda c, j: c + y**j)
    assert M == Matrix([[x + 1,     0, 0],
                        [1 + y, y + 2, 3]])

    # neither row nor slice give copies that allow the original matrix to
    # be changed
    assert M.row(0) == Matrix([[x + 1, 0, 0]])
    r1 = M.row(0)
    r1[0] = 42
    assert M[0, 0] == x + 1
    r1 = M[0, :-1]  # also testing negative slice
    r1[0] = 42
    assert M[0, 0] == x + 1
    c1 = M.col(0)
    assert c1 == Matrix([x + 1, 1 + y])
    c1[0] = 0
    assert M[0, 0] == x + 1
    c1 = M[:, 0]
    c1[0] = 42
    assert M[0, 0] == x + 1


def test_row_mult():
    M = Matrix([[1,2,3],
               [4,5,6]])
    M.row_mult(1,3)
    assert M[1,0] == 12
    assert M[0,0] == 1
    assert M[1,2] == 18


def test_row_add():
    M = Matrix([[1,2,3],
               [4,5,6],
               [1,1,1]])
    M.row_add(2,0,5)
    assert M[0,0] == 6
    assert M[1,0] == 4
    assert M[0,2] == 8


def test_zip_row_op():
    for cls in classes[:2]: # XXX: immutable matrices don't support row ops
        M = cls.eye(3)
        M.zip_row_op(1, 0, lambda v, u: v + 2*u)
        assert M == cls([[1, 0, 0],
                         [2, 1, 0],
                         [0, 0, 1]])

        M = cls.eye(3)*2
        M[0, 1] = -1
        M.zip_row_op(1, 0, lambda v, u: v + 2*u); M
        assert M == cls([[2, -1, 0],
                         [4,  0, 0],
                         [0,  0, 2]])

def test_issue_3950():
    m = Matrix([1, 2, 3])
    a = Matrix([1, 2, 3])
    b = Matrix([2, 2, 3])
    assert not (m in [])
    assert not (m in [1])
    assert m != 1
    assert m == a
    assert m != b


def test_issue_3981():
    class Index1:
        def __index__(self):
            return 1

    class Index2:
        def __index__(self):
            return 2
    index1 = Index1()
    index2 = Index2()

    m = Matrix([1, 2, 3])

    assert m[index2] == 3

    m[index2] = 5
    assert m[2] == 5

    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m[index1, index2] == 6
    assert m[1, index2] == 6
    assert m[index1, 2] == 6

    m[index1, index2] = 4
    assert m[1, 2] == 4
    m[1, index2] = 6
    assert m[1, 2] == 6
    m[index1, 2] = 8
    assert m[1, 2] == 8


def test_evalf():
    a = Matrix([sqrt(5), 6])
    assert all(a.evalf()[i] == a[i].evalf() for i in range(2))
    assert all(a.evalf(2)[i] == a[i].evalf(2) for i in range(2))
    assert all(a.n(2)[i] == a[i].n(2) for i in range(2))


def test_is_symbolic():
    a = Matrix([[x, x], [x, x]])
    assert a.is_symbolic() is True
    a = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert a.is_symbolic() is False
    a = Matrix([[1, 2, 3, 4], [5, 6, x, 8]])
    assert a.is_symbolic() is True
    a = Matrix([[1, x, 3]])
    assert a.is_symbolic() is True
    a = Matrix([[1, 2, 3]])
    assert a.is_symbolic() is False
    a = Matrix([[1], [x], [3]])
    assert a.is_symbolic() is True
    a = Matrix([[1], [2], [3]])
    assert a.is_symbolic() is False


def test_is_upper():
    a = Matrix([[1, 2, 3]])
    assert a.is_upper is True
    a = Matrix([[1], [2], [3]])
    assert a.is_upper is False
    a = zeros(4, 2)
    assert a.is_upper is True


def test_is_lower():
    a = Matrix([[1, 2, 3]])
    assert a.is_lower is False
    a = Matrix([[1], [2], [3]])
    assert a.is_lower is True


def test_is_nilpotent():
    a = Matrix(4, 4, [0, 2, 1, 6, 0, 0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 0])
    assert a.is_nilpotent()
    a = Matrix([[1, 0], [0, 1]])
    assert not a.is_nilpotent()
    a = Matrix([])
    assert a.is_nilpotent()


def test_zeros_ones_fill():
    n, m = 3, 5

    a = zeros(n, m)
    a.fill( 5 )

    b = 5 * ones(n, m)

    assert a == b
    assert a.rows == b.rows == 3
    assert a.cols == b.cols == 5
    assert a.shape == b.shape == (3, 5)
    assert zeros(2) == zeros(2, 2)
    assert ones(2) == ones(2, 2)
    assert zeros(2, 3) == Matrix(2, 3, [0]*6)
    assert ones(2, 3) == Matrix(2, 3, [1]*6)

    a.fill(0)
    assert a == zeros(n, m)


def test_empty_zeros():
    a = zeros(0)
    assert a == Matrix()
    a = zeros(0, 2)
    assert a.rows == 0
    assert a.cols == 2
    a = zeros(2, 0)
    assert a.rows == 2
    assert a.cols == 0


def test_issue_3749():
    a = Matrix([[x**2, x*y], [x*sin(y), x*cos(y)]])
    assert a.diff(x) == Matrix([[2*x, y], [sin(y), cos(y)]])
    assert Matrix([
        [x, -x, x**2],
        [exp(x), 1/x - exp(-x), x + 1/x]]).limit(x, oo) == \
        Matrix([[oo, -oo, oo], [oo, 0, oo]])
    assert Matrix([
        [(exp(x) - 1)/x, 2*x + y*x, x**x ],
        [1/x, abs(x), abs(sin(x + 1))]]).limit(x, 0) == \
        Matrix([[1, 0, 1], [oo, 0, sin(1)]])
    assert a.integrate(x) == Matrix([
        [Rational(1, 3)*x**3, y*x**2/2],
        [x**2*sin(y)/2, x**2*cos(y)/2]])


def test_inv_iszerofunc():
    A = eye(4)
    A.col_swap(0, 1)
    for method in "GE", "LU":
        assert A.inv(method=method, iszerofunc=lambda x: x == 0) == \
            A.inv(method="ADJ")


def test_jacobian_metrics():
    rho, phi = symbols("rho,phi")
    X = Matrix([rho*cos(phi), rho*sin(phi)])
    Y = Matrix([rho, phi])
    J = X.jacobian(Y)
    assert J == X.jacobian(Y.T)
    assert J == (X.T).jacobian(Y)
    assert J == (X.T).jacobian(Y.T)
    g = J.T*eye(J.shape[0])*J
    g = g.applyfunc(trigsimp)
    assert g == Matrix([[1, 0], [0, rho**2]])


def test_jacobian2():
    rho, phi = symbols("rho,phi")
    X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
    Y = Matrix([rho, phi])
    J = Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)],
        [   2*rho,             0],
    ])
    assert X.jacobian(Y) == J


def test_issue_4564():
    X = Matrix([exp(x + y + z), exp(x + y + z), exp(x + y + z)])
    Y = Matrix([x, y, z])
    for i in range(1, 3):
        for j in range(1, 3):
            X_slice = X[:i, :]
            Y_slice = Y[:j, :]
            J = X_slice.jacobian(Y_slice)
            assert J.rows == i
            assert J.cols == j
            for k in range(j):
                assert J[:, k] == X_slice


def test_nonvectorJacobian():
    X = Matrix([[exp(x + y + z), exp(x + y + z)],
                [exp(x + y + z), exp(x + y + z)]])
    raises(TypeError, lambda: X.jacobian(Matrix([x, y, z])))
    X = X[0, :]
    Y = Matrix([[x, y], [x, z]])
    raises(TypeError, lambda: X.jacobian(Y))
    raises(TypeError, lambda: X.jacobian(Matrix([ [x, y], [x, z] ])))


def test_vec():
    m = Matrix([[1, 3], [2, 4]])
    m_vec = m.vec()
    assert m_vec.cols == 1
    for i in range(4):
        assert m_vec[i] == i + 1


def test_vech():
    m = Matrix([[1, 2], [2, 3]])
    m_vech = m.vech()
    assert m_vech.cols == 1
    for i in range(3):
        assert m_vech[i] == i + 1
    m_vech = m.vech(diagonal=False)
    assert m_vech[0] == 2

    m = Matrix([[1, x*(x + y)], [y*x + x**2, 1]])
    m_vech = m.vech(diagonal=False)
    assert m_vech[0] == y*x + x**2

    m = Matrix([[1, x*(x + y)], [y*x, 1]])
    m_vech = m.vech(diagonal=False, check_symmetry=False)
    assert m_vech[0] == y*x

    raises(ShapeError, lambda: Matrix([[1, 3]]).vech())
    raises(ValueError, lambda: Matrix([[1, 3], [2, 4]]).vech())
    raises(ShapeError, lambda: Matrix([[1, 3]]).vech())
    raises(ValueError, lambda: Matrix([[1, 3], [2, 4]]).vech())


def test_diag():
    # mostly tested in testcommonmatrix.py
    assert diag([1, 2, 3]) == Matrix([1, 2, 3])
    m = [1, 2, [3]]
    raises(ValueError, lambda: diag(m))
    assert diag(m, strict=False) == Matrix([1, 2, 3])


def test_get_diag_blocks1():
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    assert a.get_diag_blocks() == [a]
    assert b.get_diag_blocks() == [b]
    assert c.get_diag_blocks() == [c]


def test_get_diag_blocks2():
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    assert diag(a, b, b).get_diag_blocks() == [a, b, b]
    assert diag(a, b, c).get_diag_blocks() == [a, b, c]
    assert diag(a, c, b).get_diag_blocks() == [a, c, b]
    assert diag(c, c, b).get_diag_blocks() == [c, c, b]


def test_inv_block():
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    A = diag(a, b, b)
    assert A.inv(try_block_diag=True) == diag(a.inv(), b.inv(), b.inv())
    A = diag(a, b, c)
    assert A.inv(try_block_diag=True) == diag(a.inv(), b.inv(), c.inv())
    A = diag(a, c, b)
    assert A.inv(try_block_diag=True) == diag(a.inv(), c.inv(), b.inv())
    A = diag(a, a, b, a, c, a)
    assert A.inv(try_block_diag=True) == diag(
        a.inv(), a.inv(), b.inv(), a.inv(), c.inv(), a.inv())
    assert A.inv(try_block_diag=True, method="ADJ") == diag(
        a.inv(method="ADJ"), a.inv(method="ADJ"), b.inv(method="ADJ"),
        a.inv(method="ADJ"), c.inv(method="ADJ"), a.inv(method="ADJ"))


def test_creation_args():
    """
    Check that matrix dimensions can be specified using any reasonable type
    (see issue 4614).
    """
    raises(ValueError, lambda: zeros(3, -1))
    raises(TypeError, lambda: zeros(1, 2, 3, 4))
    assert zeros(int(3)) == zeros(3)
    assert zeros(Integer(3)) == zeros(3)
    raises(ValueError, lambda: zeros(3.))
    assert eye(int(3)) == eye(3)
    assert eye(Integer(3)) == eye(3)
    raises(ValueError, lambda: eye(3.))
    assert ones(int(3), Integer(4)) == ones(3, 4)
    raises(TypeError, lambda: Matrix(5))
    raises(TypeError, lambda: Matrix(1, 2))
    raises(ValueError, lambda: Matrix([1, [2]]))


def test_diagonal_symmetrical():
    m = Matrix(2, 2, [0, 1, 1, 0])
    assert not m.is_diagonal()
    assert m.is_symmetric()
    assert m.is_symmetric(simplify=False)

    m = Matrix(2, 2, [1, 0, 0, 1])
    assert m.is_diagonal()

    m = diag(1, 2, 3)
    assert m.is_diagonal()
    assert m.is_symmetric()

    m = Matrix(3, 3, [1, 0, 0, 0, 2, 0, 0, 0, 3])
    assert m == diag(1, 2, 3)

    m = Matrix(2, 3, zeros(2, 3))
    assert not m.is_symmetric()
    assert m.is_diagonal()

    m = Matrix(((5, 0), (0, 6), (0, 0)))
    assert m.is_diagonal()

    m = Matrix(((5, 0, 0), (0, 6, 0)))
    assert m.is_diagonal()

    m = Matrix(3, 3, [1, x**2 + 2*x + 1, y, (x + 1)**2, 2, 0, y, 0, 3])
    assert m.is_symmetric()
    assert not m.is_symmetric(simplify=False)
    assert m.expand().is_symmetric(simplify=False)


def test_diagonalization():
    m = Matrix([[1, 2+I], [2-I, 3]])
    assert m.is_diagonalizable()

    m = Matrix(3, 2, [-3, 1, -3, 20, 3, 10])
    assert not m.is_diagonalizable()
    assert not m.is_symmetric()
    raises(NonSquareMatrixError, lambda: m.diagonalize())

    # diagonalizable
    m = diag(1, 2, 3)
    (P, D) = m.diagonalize()
    assert P == eye(3)
    assert D == m

    m = Matrix(2, 2, [0, 1, 1, 0])
    assert m.is_symmetric()
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    assert P.inv() * m * P == D

    m = Matrix(2, 2, [1, 0, 0, 3])
    assert m.is_symmetric()
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    assert P.inv() * m * P == D
    assert P == eye(2)
    assert D == m

    m = Matrix(2, 2, [1, 1, 0, 0])
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    assert P.inv() * m * P == D

    m = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    assert P.inv() * m * P == D
    for i in P:
        assert i.as_numer_denom()[1] == 1

    m = Matrix(2, 2, [1, 0, 0, 0])
    assert m.is_diagonal()
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    assert P.inv() * m * P == D
    assert P == Matrix([[0, 1], [1, 0]])

    # diagonalizable, complex only
    m = Matrix(2, 2, [0, 1, -1, 0])
    assert not m.is_diagonalizable(True)
    raises(MatrixError, lambda: m.diagonalize(True))
    assert m.is_diagonalizable()
    (P, D) = m.diagonalize()
    assert P.inv() * m * P == D

    # not diagonalizable
    m = Matrix(2, 2, [0, 1, 0, 0])
    assert not m.is_diagonalizable()
    raises(MatrixError, lambda: m.diagonalize())

    m = Matrix(3, 3, [-3, 1, -3, 20, 3, 10, 2, -2, 4])
    assert not m.is_diagonalizable()
    raises(MatrixError, lambda: m.diagonalize())

    # symbolic
    a, b, c, d = symbols('a b c d')
    m = Matrix(2, 2, [a, c, c, b])
    assert m.is_symmetric()
    assert m.is_diagonalizable()


def test_issue_15887():
    # Mutable matrix should not use cache
    a = MutableDenseMatrix([[0, 1], [1, 0]])
    assert a.is_diagonalizable() is True
    a[1, 0] = 0
    assert a.is_diagonalizable() is False

    a = MutableDenseMatrix([[0, 1], [1, 0]])
    a.diagonalize()
    a[1, 0] = 0
    raises(MatrixError, lambda: a.diagonalize())


def test_jordan_form():

    m = Matrix(3, 2, [-3, 1, -3, 20, 3, 10])
    raises(NonSquareMatrixError, lambda: m.jordan_form())

    # diagonalizable
    m = Matrix(3, 3, [7, -12, 6, 10, -19, 10, 12, -24, 13])
    Jmust = Matrix(3, 3, [-1, 0, 0, 0, 1, 0, 0, 0, 1])
    P, J = m.jordan_form()
    assert Jmust == J
    assert Jmust == m.diagonalize()[1]

    # m = Matrix(3, 3, [0, 6, 3, 1, 3, 1, -2, 2, 1])
    # m.jordan_form()  # very long
    # m.jordan_form()  #

    # diagonalizable, complex only

    # Jordan cells
    # complexity: one of eigenvalues is zero
    m = Matrix(3, 3, [0, 1, 0, -4, 4, 0, -2, 1, 2])
    # The blocks are ordered according to the value of their eigenvalues,
    # in order to make the matrix compatible with .diagonalize()
    Jmust = Matrix(3, 3, [2, 1, 0, 0, 2, 0, 0, 0, 2])
    P, J = m.jordan_form()
    assert Jmust == J

    # complexity: all of eigenvalues are equal
    m = Matrix(3, 3, [2, 6, -15, 1, 1, -5, 1, 2, -6])
    # Jmust = Matrix(3, 3, [-1, 0, 0, 0, -1, 1, 0, 0, -1])
    # same here see 1456ff
    Jmust = Matrix(3, 3, [-1, 1, 0, 0, -1, 0, 0, 0, -1])
    P, J = m.jordan_form()
    assert Jmust == J

    # complexity: two of eigenvalues are zero
    m = Matrix(3, 3, [4, -5, 2, 5, -7, 3, 6, -9, 4])
    Jmust = Matrix(3, 3, [0, 1, 0, 0, 0, 0, 0, 0, 1])
    P, J = m.jordan_form()
    assert Jmust == J

    m = Matrix(4, 4, [6, 5, -2, -3, -3, -1, 3, 3, 2, 1, -2, -3, -1, 1, 5, 5])
    Jmust = Matrix(4, 4, [2, 1, 0, 0,
                          0, 2, 0, 0,
              0, 0, 2, 1,
              0, 0, 0, 2]
              )
    P, J = m.jordan_form()
    assert Jmust == J

    m = Matrix(4, 4, [6, 2, -8, -6, -3, 2, 9, 6, 2, -2, -8, -6, -1, 0, 3, 4])
    # Jmust = Matrix(4, 4, [2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, -2])
    # same here see 1456ff
    Jmust = Matrix(4, 4, [-2, 0, 0, 0,
                           0, 2, 1, 0,
                           0, 0, 2, 0,
                           0, 0, 0, 2])
    P, J = m.jordan_form()
    assert Jmust == J

    m = Matrix(4, 4, [5, 4, 2, 1, 0, 1, -1, -1, -1, -1, 3, 0, 1, 1, -1, 2])
    assert not m.is_diagonalizable()
    Jmust = Matrix(4, 4, [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 1, 0, 0, 0, 4])
    P, J = m.jordan_form()
    assert Jmust == J

    # checking for maximum precision to remain unchanged
    m = Matrix([[Float('1.0', precision=110), Float('2.0', precision=110)],
                [Float('3.14159265358979323846264338327', precision=110), Float('4.0', precision=110)]])
    P, J = m.jordan_form()
    for term in J.values():
        if isinstance(term, Float):
            assert term._prec == 110


def test_jordan_form_complex_issue_9274():
    A = Matrix([[ 2,  4,  1,  0],
                [-4,  2,  0,  1],
                [ 0,  0,  2,  4],
                [ 0,  0, -4,  2]])
    p = 2 - 4*I
    q = 2 + 4*I
    Jmust1 = Matrix([[p, 1, 0, 0],
                     [0, p, 0, 0],
                     [0, 0, q, 1],
                     [0, 0, 0, q]])
    Jmust2 = Matrix([[q, 1, 0, 0],
                     [0, q, 0, 0],
                     [0, 0, p, 1],
                     [0, 0, 0, p]])
    P, J = A.jordan_form()
    assert J == Jmust1 or J == Jmust2
    assert simplify(P*J*P.inv()) == A

def test_issue_10220():
    # two non-orthogonal Jordan blocks with eigenvalue 1
    M = Matrix([[1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1]])
    P, J = M.jordan_form()
    assert P == Matrix([[0, 1, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])
    assert J == Matrix([
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

def test_jordan_form_issue_15858():
    A = Matrix([
        [1, 1, 1, 0],
        [-2, -1, 0, -1],
        [0, 0, -1, -1],
        [0, 0, 2, 1]])
    (P, J) = A.jordan_form()
    assert P.expand() == Matrix([
        [    -I,          -I/2,      I,           I/2],
        [-1 + I,             0, -1 - I,             0],
        [     0, -S(1)/2 - I/2,      0, -S(1)/2 + I/2],
        [     0,             1,      0,             1]])
    assert J == Matrix([
        [-I, 1, 0, 0],
        [0, -I, 0, 0],
        [0, 0, I, 1],
        [0, 0, 0, I]])

def test_Matrix_berkowitz_charpoly():
    UA, K_i, K_w = symbols('UA K_i K_w')

    A = Matrix([[-K_i - UA + K_i**2/(K_i + K_w),       K_i*K_w/(K_i + K_w)],
                [           K_i*K_w/(K_i + K_w), -K_w + K_w**2/(K_i + K_w)]])

    charpoly = A.charpoly(x)

    assert charpoly == \
        Poly(x**2 + (K_i*UA + K_w*UA + 2*K_i*K_w)/(K_i + K_w)*x +
        K_i*K_w*UA/(K_i + K_w), x, domain='ZZ(K_i,K_w,UA)')

    assert type(charpoly) is PurePoly

    A = Matrix([[1, 3], [2, 0]])
    assert A.charpoly() == A.charpoly(x) == PurePoly(x**2 - x - 6)

    A = Matrix([[1, 2], [x, 0]])
    p = A.charpoly(x)
    assert p.gen != x
    assert p.as_expr().subs(p.gen, x) == x**2 - 3*x


def test_exp_jordan_block():
    l = Symbol('lamda')

    m = Matrix.jordan_block(1, l)
    assert m._eval_matrix_exp_jblock() == Matrix([[exp(l)]])

    m = Matrix.jordan_block(3, l)
    assert m._eval_matrix_exp_jblock() == \
        Matrix([
            [exp(l), exp(l), exp(l)/2],
            [0, exp(l), exp(l)],
            [0, 0, exp(l)]])


def test_exp():
    m = Matrix([[3, 4], [0, -2]])
    m_exp = Matrix([[exp(3), -4*exp(-2)/5 + 4*exp(3)/5], [0, exp(-2)]])
    assert m.exp() == m_exp
    assert exp(m) == m_exp

    m = Matrix([[1, 0], [0, 1]])
    assert m.exp() == Matrix([[E, 0], [0, E]])
    assert exp(m) == Matrix([[E, 0], [0, E]])

    m = Matrix([[1, -1], [1, 1]])
    assert m.exp() == Matrix([[E*cos(1), -E*sin(1)], [E*sin(1), E*cos(1)]])


def test_log():
    l = Symbol('lamda')

    m = Matrix.jordan_block(1, l)
    assert m._eval_matrix_log_jblock() == Matrix([[log(l)]])

    m = Matrix.jordan_block(4, l)
    assert m._eval_matrix_log_jblock() == \
        Matrix(
            [
                [log(l), 1/l, -1/(2*l**2), 1/(3*l**3)],
                [0, log(l), 1/l, -1/(2*l**2)],
                [0, 0, log(l), 1/l],
                [0, 0, 0, log(l)]
            ]
        )

    m = Matrix(
        [[0, 0, 1],
        [0, 0, 0],
        [-1, 0, 0]]
    )
    raises(MatrixError, lambda: m.log())


def test_has():
    A = Matrix(((x, y), (2, 3)))
    assert A.has(x)
    assert not A.has(z)
    assert A.has(Symbol)

    A = A.subs(x, 2)
    assert not A.has(x)


def test_find_reasonable_pivot_naive_finds_guaranteed_nonzero1():
    # Test if matrices._find_reasonable_pivot_naive()
    # finds a guaranteed non-zero pivot when the
    # some of the candidate pivots are symbolic expressions.
    # Keyword argument: simpfunc=None indicates that no simplifications
    # should be performed during the search.
    x = Symbol('x')
    column = Matrix(3, 1, [x, cos(x)**2 + sin(x)**2, S.Half])
    pivot_offset, pivot_val, pivot_assumed_nonzero, simplified =\
        _find_reasonable_pivot_naive(column)
    assert pivot_val == S.Half

def test_find_reasonable_pivot_naive_finds_guaranteed_nonzero2():
    # Test if matrices._find_reasonable_pivot_naive()
    # finds a guaranteed non-zero pivot when the
    # some of the candidate pivots are symbolic expressions.
    # Keyword argument: simpfunc=_simplify indicates that the search
    # should attempt to simplify candidate pivots.
    x = Symbol('x')
    column = Matrix(3, 1,
                    [x,
                     cos(x)**2+sin(x)**2+x**2,
                     cos(x)**2+sin(x)**2])
    pivot_offset, pivot_val, pivot_assumed_nonzero, simplified =\
        _find_reasonable_pivot_naive(column, simpfunc=_simplify)
    assert pivot_val == 1

def test_find_reasonable_pivot_naive_simplifies():
    # Test if matrices._find_reasonable_pivot_naive()
    # simplifies candidate pivots, and reports
    # their offsets correctly.
    x = Symbol('x')
    column = Matrix(3, 1,
                    [x,
                     cos(x)**2+sin(x)**2+x,
                     cos(x)**2+sin(x)**2])
    pivot_offset, pivot_val, pivot_assumed_nonzero, simplified =\
        _find_reasonable_pivot_naive(column, simpfunc=_simplify)

    assert len(simplified) == 2
    assert simplified[0][0] == 1
    assert simplified[0][1] == 1+x
    assert simplified[1][0] == 2
    assert simplified[1][1] == 1

def test_errors():
    raises(ValueError, lambda: Matrix([[1, 2], [1]]))
    raises(IndexError, lambda: Matrix([[1, 2]])[1.2, 5])
    raises(IndexError, lambda: Matrix([[1, 2]])[1, 5.2])
    raises(ValueError, lambda: randMatrix(3, c=4, symmetric=True))
    raises(ValueError, lambda: Matrix([1, 2]).reshape(4, 6))
    raises(ShapeError,
        lambda: Matrix([[1, 2], [3, 4]]).copyin_matrix([1, 0], Matrix([1, 2])))
    raises(TypeError, lambda: Matrix([[1, 2], [3, 4]]).copyin_list([0,
           1], set()))
    raises(NonSquareMatrixError, lambda: Matrix([[1, 2, 3], [2, 3, 0]]).inv())
    raises(ShapeError,
        lambda: Matrix(1, 2, [1, 2]).row_join(Matrix([[1, 2], [3, 4]])))
    raises(
        ShapeError, lambda: Matrix([1, 2]).col_join(Matrix([[1, 2], [3, 4]])))
    raises(ShapeError, lambda: Matrix([1]).row_insert(1, Matrix([[1,
           2], [3, 4]])))
    raises(ShapeError, lambda: Matrix([1]).col_insert(1, Matrix([[1,
           2], [3, 4]])))
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).trace())
    raises(TypeError, lambda: Matrix([1]).applyfunc(1))
    raises(ValueError, lambda: Matrix([[1, 2], [3, 4]]).minor(4, 5))
    raises(ValueError, lambda: Matrix([[1, 2], [3, 4]]).minor_submatrix(4, 5))
    raises(TypeError, lambda: Matrix([1, 2, 3]).cross(1))
    raises(TypeError, lambda: Matrix([1, 2, 3]).dot(1))
    raises(ShapeError, lambda: Matrix([1, 2, 3]).dot(Matrix([1, 2])))
    raises(ShapeError, lambda: Matrix([1, 2]).dot([]))
    raises(TypeError, lambda: Matrix([1, 2]).dot('a'))
    raises(ShapeError, lambda: Matrix([1, 2]).dot([1, 2, 3]))
    raises(NonSquareMatrixError, lambda: Matrix([1, 2, 3]).exp())
    raises(ShapeError, lambda: Matrix([[1, 2], [3, 4]]).normalized())
    raises(ValueError, lambda: Matrix([1, 2]).inv(method='not a method'))
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).inverse_GE())
    raises(ValueError, lambda: Matrix([[1, 2], [1, 2]]).inverse_GE())
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).inverse_ADJ())
    raises(ValueError, lambda: Matrix([[1, 2], [1, 2]]).inverse_ADJ())
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).inverse_LU())
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).is_nilpotent())
    raises(NonSquareMatrixError, lambda: Matrix([1, 2]).det())
    raises(ValueError,
        lambda: Matrix([[1, 2], [3, 4]]).det(method='Not a real method'))
    raises(ValueError,
        lambda: Matrix([[1, 2, 3, 4], [5, 6, 7, 8],
        [9, 10, 11, 12], [13, 14, 15, 16]]).det(iszerofunc="Not function"))
    raises(ValueError,
        lambda: Matrix([[1, 2, 3, 4], [5, 6, 7, 8],
        [9, 10, 11, 12], [13, 14, 15, 16]]).det(iszerofunc=False))
    raises(ValueError,
        lambda: hessian(Matrix([[1, 2], [3, 4]]), Matrix([[1, 2], [2, 1]])))
    raises(ValueError, lambda: hessian(Matrix([[1, 2], [3, 4]]), []))
    raises(ValueError, lambda: hessian(Symbol('x')**2, 'a'))
    raises(IndexError, lambda: eye(3)[5, 2])
    raises(IndexError, lambda: eye(3)[2, 5])
    M = Matrix(((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)))
    raises(ValueError, lambda: M.det('method=LU_decomposition()'))
    V = Matrix([[10, 10, 10]])
    M = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    raises(ValueError, lambda: M.row_insert(4.7, V))
    M = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    raises(ValueError, lambda: M.col_insert(-4.2, V))

def test_len():
    assert len(Matrix()) == 0
    assert len(Matrix([[1, 2]])) == len(Matrix([[1], [2]])) == 2
    assert len(Matrix(0, 2, lambda i, j: 0)) == \
        len(Matrix(2, 0, lambda i, j: 0)) == 0
    assert len(Matrix([[0, 1, 2], [3, 4, 5]])) == 6
    assert Matrix([1]) == Matrix([[1]])
    assert not Matrix()
    assert Matrix() == Matrix([])


def test_integrate():
    A = Matrix(((1, 4, x), (y, 2, 4), (10, 5, x**2)))
    assert A.integrate(x) == \
        Matrix(((x, 4*x, x**2/2), (x*y, 2*x, 4*x), (10*x, 5*x, x**3/3)))
    assert A.integrate(y) == \
        Matrix(((y, 4*y, x*y), (y**2/2, 2*y, 4*y), (10*y, 5*y, y*x**2)))


def test_limit():
    A = Matrix(((1, 4, sin(x)/x), (y, 2, 4), (10, 5, x**2 + 1)))
    assert A.limit(x, 0) == Matrix(((1, 4, 1), (y, 2, 4), (10, 5, 1)))


def test_diff():
    A = MutableDenseMatrix(((1, 4, x), (y, 2, 4), (10, 5, x**2 + 1)))
    assert isinstance(A.diff(x), type(A))
    assert A.diff(x) == MutableDenseMatrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))
    assert A.diff(y) == MutableDenseMatrix(((0, 0, 0), (1, 0, 0), (0, 0, 0)))

    assert diff(A, x) == MutableDenseMatrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))
    assert diff(A, y) == MutableDenseMatrix(((0, 0, 0), (1, 0, 0), (0, 0, 0)))

    A_imm = A.as_immutable()
    assert isinstance(A_imm.diff(x), type(A_imm))
    assert A_imm.diff(x) == ImmutableDenseMatrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))
    assert A_imm.diff(y) == ImmutableDenseMatrix(((0, 0, 0), (1, 0, 0), (0, 0, 0)))

    assert diff(A_imm, x) == ImmutableDenseMatrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))
    assert diff(A_imm, y) == ImmutableDenseMatrix(((0, 0, 0), (1, 0, 0), (0, 0, 0)))

    assert A.diff(x, evaluate=False) == ArrayDerivative(A, x, evaluate=False)
    assert diff(A, x, evaluate=False) == ArrayDerivative(A, x, evaluate=False)


def test_diff_by_matrix():

    # Derive matrix by matrix:

    A = MutableDenseMatrix([[x, y], [z, t]])
    assert A.diff(A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
    assert diff(A, A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    A_imm = A.as_immutable()
    assert A_imm.diff(A_imm) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
    assert diff(A_imm, A_imm) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    # Derive a constant matrix:
    assert A.diff(a) == MutableDenseMatrix([[0, 0], [0, 0]])

    B = ImmutableDenseMatrix([a, b])
    assert A.diff(B) == Array.zeros(2, 1, 2, 2)
    assert A.diff(A) == Array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])

    # Test diff with tuples:

    dB = B.diff([[a, b]])
    assert dB.shape == (2, 2, 1)
    assert dB == Array([[[1], [0]], [[0], [1]]])

    f = Function("f")
    fxyz = f(x, y, z)
    assert fxyz.diff([[x, y, z]]) == Array([fxyz.diff(x), fxyz.diff(y), fxyz.diff(z)])
    assert fxyz.diff(([x, y, z], 2)) == Array([
        [fxyz.diff(x, 2), fxyz.diff(x, y), fxyz.diff(x, z)],
        [fxyz.diff(x, y), fxyz.diff(y, 2), fxyz.diff(y, z)],
        [fxyz.diff(x, z), fxyz.diff(z, y), fxyz.diff(z, 2)],
    ])

    expr = sin(x)*exp(y)
    assert expr.diff([[x, y]]) == Array([cos(x)*exp(y), sin(x)*exp(y)])
    assert expr.diff(y, ((x, y),)) == Array([cos(x)*exp(y), sin(x)*exp(y)])
    assert expr.diff(x, ((x, y),)) == Array([-sin(x)*exp(y), cos(x)*exp(y)])
    assert expr.diff(((y, x),), [[x, y]]) == Array([[cos(x)*exp(y), -sin(x)*exp(y)], [sin(x)*exp(y), cos(x)*exp(y)]])

    # Test different notations:

    assert fxyz.diff(x).diff(y).diff(x) == fxyz.diff(((x, y, z),), 3)[0, 1, 0]
    assert fxyz.diff(z).diff(y).diff(x) == fxyz.diff(((x, y, z),), 3)[2, 1, 0]
    assert fxyz.diff([[x, y, z]], ((z, y, x),)) == Array([[fxyz.diff(i).diff(j) for i in (x, y, z)] for j in (z, y, x)])

    # Test scalar derived by matrix remains matrix:
    res = x.diff(Matrix([[x, y]]))
    assert isinstance(res, ImmutableDenseMatrix)
    assert res == Matrix([[1, 0]])
    res = (x**3).diff(Matrix([[x, y]]))
    assert isinstance(res, ImmutableDenseMatrix)
    assert res == Matrix([[3*x**2, 0]])


def test_getattr():
    A = Matrix(((1, 4, x), (y, 2, 4), (10, 5, x**2 + 1)))
    raises(AttributeError, lambda: A.nonexistantattribute)
    assert getattr(A, 'diff')(x) == Matrix(((0, 0, 1), (0, 0, 0), (0, 0, 2*x)))


def test_hessenberg():
    A = Matrix([[3, 4, 1], [2, 4, 5], [0, 1, 2]])
    assert A.is_upper_hessenberg
    A = A.T
    assert A.is_lower_hessenberg
    A[0, -1] = 1
    assert A.is_lower_hessenberg is False

    A = Matrix([[3, 4, 1], [2, 4, 5], [3, 1, 2]])
    assert not A.is_upper_hessenberg

    A = zeros(5, 2)
    assert A.is_upper_hessenberg


def test_cholesky():
    raises(NonSquareMatrixError, lambda: Matrix((1, 2)).cholesky())
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).cholesky())
    raises(ValueError, lambda: Matrix(((5 + I, 0), (0, 1))).cholesky())
    raises(ValueError, lambda: Matrix(((1, 5), (5, 1))).cholesky())
    raises(ValueError, lambda: Matrix(((1, 2), (3, 4))).cholesky(hermitian=False))
    assert Matrix(((5 + I, 0), (0, 1))).cholesky(hermitian=False) == Matrix([
        [sqrt(5 + I), 0], [0, 1]])
    A = Matrix(((1, 5), (5, 1)))
    L = A.cholesky(hermitian=False)
    assert L == Matrix([[1, 0], [5, 2*sqrt(6)*I]])
    assert L*L.T == A
    A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L = A.cholesky()
    assert L * L.T == A
    assert L.is_lower
    assert L == Matrix([[5, 0, 0], [3, 3, 0], [-1, 1, 3]])
    A = Matrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    assert A.cholesky().expand() == Matrix(((2, 0, 0), (I, 1, 0), (1 - I, 0, 3)))

    raises(NonSquareMatrixError, lambda: SparseMatrix((1, 2)).cholesky())
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).cholesky())
    raises(ValueError, lambda: SparseMatrix(((5 + I, 0), (0, 1))).cholesky())
    raises(ValueError, lambda: SparseMatrix(((1, 5), (5, 1))).cholesky())
    raises(ValueError, lambda: SparseMatrix(((1, 2), (3, 4))).cholesky(hermitian=False))
    assert SparseMatrix(((5 + I, 0), (0, 1))).cholesky(hermitian=False) == Matrix([
        [sqrt(5 + I), 0], [0, 1]])
    A = SparseMatrix(((1, 5), (5, 1)))
    L = A.cholesky(hermitian=False)
    assert L == Matrix([[1, 0], [5, 2*sqrt(6)*I]])
    assert L*L.T == A
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L = A.cholesky()
    assert L * L.T == A
    assert L.is_lower
    assert L == Matrix([[5, 0, 0], [3, 3, 0], [-1, 1, 3]])
    A = SparseMatrix(((4, -2*I, 2 + 2*I), (2*I, 2, -1 + I), (2 - 2*I, -1 - I, 11)))
    assert A.cholesky() == Matrix(((2, 0, 0), (I, 1, 0), (1 - I, 0, 3)))


def test_matrix_norm():
    # Vector Tests
    # Test columns and symbols
    x = Symbol('x', real=True)
    v = Matrix([cos(x), sin(x)])
    assert trigsimp(v.norm(2)) == 1
    assert v.norm(10) == Pow(cos(x)**10 + sin(x)**10, Rational(1, 10))

    # Test Rows
    A = Matrix([[5, Rational(3, 2)]])
    assert A.norm() == Pow(25 + Rational(9, 4), S.Half)
    assert A.norm(oo) == max(A)
    assert A.norm(-oo) == min(A)

    # Matrix Tests
    # Intuitive test
    A = Matrix([[1, 1], [1, 1]])
    assert A.norm(2) == 2
    assert A.norm(-2) == 0
    assert A.norm('frobenius') == 2
    assert eye(10).norm(2) == eye(10).norm(-2) == 1
    assert A.norm(oo) == 2

    # Test with Symbols and more complex entries
    A = Matrix([[3, y, y], [x, S.Half, -pi]])
    assert (A.norm('fro')
           == sqrt(Rational(37, 4) + 2*abs(y)**2 + pi**2 + x**2))

    # Check non-square
    A = Matrix([[1, 2, -3], [4, 5, Rational(13, 2)]])
    assert A.norm(2) == sqrt(Rational(389, 8) + sqrt(78665)/8)
    assert A.norm(-2) is S.Zero
    assert A.norm('frobenius') == sqrt(389)/2

    # Test properties of matrix norms
    # https://en.wikipedia.org/wiki/Matrix_norm#Definition
    # Two matrices
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 5], [-2, 2]])
    C = Matrix([[0, -I], [I, 0]])
    D = Matrix([[1, 0], [0, -1]])
    L = [A, B, C, D]
    alpha = Symbol('alpha', real=True)

    for order in ['fro', 2, -2]:
        # Zero Check
        assert zeros(3).norm(order) is S.Zero
        # Check Triangle Inequality for all Pairs of Matrices
        for X in L:
            for Y in L:
                dif = (X.norm(order) + Y.norm(order) -
                    (X + Y).norm(order))
                assert (dif >= 0)
        # Scalar multiplication linearity
        for M in [A, B, C, D]:
            dif = simplify((alpha*M).norm(order) -
                    abs(alpha) * M.norm(order))
            assert dif == 0

    # Test Properties of Vector Norms
    # https://en.wikipedia.org/wiki/Vector_norm
    # Two column vectors
    a = Matrix([1, 1 - 1*I, -3])
    b = Matrix([S.Half, 1*I, 1])
    c = Matrix([-1, -1, -1])
    d = Matrix([3, 2, I])
    e = Matrix([Integer(1e2), Rational(1, 1e2), 1])
    L = [a, b, c, d, e]
    alpha = Symbol('alpha', real=True)

    for order in [1, 2, -1, -2, S.Infinity, S.NegativeInfinity, pi]:
        # Zero Check
        if order > 0:
            assert Matrix([0, 0, 0]).norm(order) is S.Zero
        # Triangle inequality on all pairs
        if order >= 1:  # Triangle InEq holds only for these norms
            for X in L:
                for Y in L:
                    dif = (X.norm(order) + Y.norm(order) -
                        (X + Y).norm(order))
                    assert simplify(dif >= 0) is S.true
        # Linear to scalar multiplication
        if order in [1, 2, -1, -2, S.Infinity, S.NegativeInfinity]:
            for X in L:
                dif = simplify((alpha*X).norm(order) -
                    (abs(alpha) * X.norm(order)))
                assert dif == 0

    # ord=1
    M = Matrix(3, 3, [1, 3, 0, -2, -1, 0, 3, 9, 6])
    assert M.norm(1) == 13


def test_condition_number():
    x = Symbol('x', real=True)
    A = eye(3)
    A[0, 0] = 10
    A[2, 2] = Rational(1, 10)
    assert A.condition_number() == 100

    A[1, 1] = x
    assert A.condition_number() == Max(10, Abs(x)) / Min(Rational(1, 10), Abs(x))

    M = Matrix([[cos(x), sin(x)], [-sin(x), cos(x)]])
    Mc = M.condition_number()
    assert all(Float(1.).epsilon_eq(Mc.subs(x, val).evalf()) for val in
        [Rational(1, 5), S.Half, Rational(1, 10), pi/2, pi, pi*Rational(7, 4) ])

    #issue 10782
    assert Matrix([]).condition_number() == 0


def test_equality():
    A = Matrix(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    B = Matrix(((9, 8, 7), (6, 5, 4), (3, 2, 1)))
    assert A == A[:, :]
    assert not A != A[:, :]
    assert not A == B
    assert A != B
    assert A != 10
    assert not A == 10

    # A SparseMatrix can be equal to a Matrix
    C = SparseMatrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    D = Matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert C == D
    assert not C != D


def test_col_join():
    assert eye(3).col_join(Matrix([[7, 7, 7]])) == \
        Matrix([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [7, 7, 7]])


def test_row_insert():
    r4 = Matrix([[4, 4, 4]])
    for i in range(-4, 5):
        l = [1, 0, 0]
        l.insert(i, 4)
        assert flatten(eye(3).row_insert(i, r4).col(0).tolist()) == l


def test_col_insert():
    c4 = Matrix([4, 4, 4])
    for i in range(-4, 5):
        l = [0, 0, 0]
        l.insert(i, 4)
        assert flatten(zeros(3).col_insert(i, c4).row(0).tolist()) == l


def test_normalized():
    assert Matrix([3, 4]).normalized() == \
        Matrix([Rational(3, 5), Rational(4, 5)])

    # Zero vector trivial cases
    assert Matrix([0, 0, 0]).normalized() == Matrix([0, 0, 0])

    # Machine precision error truncation trivial cases
    m = Matrix([0,0,1.e-100])
    assert m.normalized(
    iszerofunc=lambda x: x.evalf(n=10, chop=True).is_zero
    ) == Matrix([0, 0, 0])


def test_print_nonzero():
    assert capture(lambda: eye(3).print_nonzero()) == \
        '[X  ]\n[ X ]\n[  X]\n'
    assert capture(lambda: eye(3).print_nonzero('.')) == \
        '[.  ]\n[ . ]\n[  .]\n'


def test_zeros_eye():
    assert Matrix.eye(3) == eye(3)
    assert Matrix.zeros(3) == zeros(3)
    assert ones(3, 4) == Matrix(3, 4, [1]*12)

    i = Matrix([[1, 0], [0, 1]])
    z = Matrix([[0, 0], [0, 0]])
    for cls in classes:
        m = cls.eye(2)
        assert i == m  # but m == i will fail if m is immutable
        assert i == eye(2, cls=cls)
        assert type(m) == cls
        m = cls.zeros(2)
        assert z == m
        assert z == zeros(2, cls=cls)
        assert type(m) == cls


def test_is_zero():
    assert Matrix().is_zero_matrix
    assert Matrix([[0, 0], [0, 0]]).is_zero_matrix
    assert zeros(3, 4).is_zero_matrix
    assert not eye(3).is_zero_matrix
    assert Matrix([[x, 0], [0, 0]]).is_zero_matrix == None
    assert SparseMatrix([[x, 0], [0, 0]]).is_zero_matrix == None
    assert ImmutableMatrix([[x, 0], [0, 0]]).is_zero_matrix == None
    assert ImmutableSparseMatrix([[x, 0], [0, 0]]).is_zero_matrix == None
    assert Matrix([[x, 1], [0, 0]]).is_zero_matrix == False
    a = Symbol('a', nonzero=True)
    assert Matrix([[a, 0], [0, 0]]).is_zero_matrix == False


def test_rotation_matrices():
    # This tests the rotation matrices by rotating about an axis and back.
    theta = pi/3
    r3_plus = rot_axis3(theta)
    r3_minus = rot_axis3(-theta)
    r2_plus = rot_axis2(theta)
    r2_minus = rot_axis2(-theta)
    r1_plus = rot_axis1(theta)
    r1_minus = rot_axis1(-theta)
    assert r3_minus*r3_plus*eye(3) == eye(3)
    assert r2_minus*r2_plus*eye(3) == eye(3)
    assert r1_minus*r1_plus*eye(3) == eye(3)

    # Check the correctness of the trace of the rotation matrix
    assert r1_plus.trace() == 1 + 2*cos(theta)
    assert r2_plus.trace() == 1 + 2*cos(theta)
    assert r3_plus.trace() == 1 + 2*cos(theta)

    # Check that a rotation with zero angle doesn't change anything.
    assert rot_axis1(0) == eye(3)
    assert rot_axis2(0) == eye(3)
    assert rot_axis3(0) == eye(3)

    # Check left-hand convention
    # see Issue #24529
    q1 = Quaternion.from_axis_angle([1, 0, 0], pi / 2)
    q2 = Quaternion.from_axis_angle([0, 1, 0], pi / 2)
    q3 = Quaternion.from_axis_angle([0, 0, 1], pi / 2)
    assert rot_axis1(- pi / 2) == q1.to_rotation_matrix()
    assert rot_axis2(- pi / 2) == q2.to_rotation_matrix()
    assert rot_axis3(- pi / 2) == q3.to_rotation_matrix()
    # Check right-hand convention
    assert rot_ccw_axis1(+ pi / 2) == q1.to_rotation_matrix()
    assert rot_ccw_axis2(+ pi / 2) == q2.to_rotation_matrix()
    assert rot_ccw_axis3(+ pi / 2) == q3.to_rotation_matrix()


def test_DeferredVector():
    assert str(DeferredVector("vector")[4]) == "vector[4]"
    assert sympify(DeferredVector("d")) == DeferredVector("d")
    raises(IndexError, lambda: DeferredVector("d")[-1])
    assert str(DeferredVector("d")) == "d"
    assert repr(DeferredVector("test")) == "DeferredVector('test')"

def test_DeferredVector_not_iterable():
    assert not iterable(DeferredVector('X'))

def test_DeferredVector_Matrix():
    raises(TypeError, lambda: Matrix(DeferredVector("V")))

def test_GramSchmidt():
    R = Rational
    m1 = Matrix(1, 2, [1, 2])
    m2 = Matrix(1, 2, [2, 3])
    assert GramSchmidt([m1, m2]) == \
        [Matrix(1, 2, [1, 2]), Matrix(1, 2, [R(2)/5, R(-1)/5])]
    assert GramSchmidt([m1.T, m2.T]) == \
        [Matrix(2, 1, [1, 2]), Matrix(2, 1, [R(2)/5, R(-1)/5])]
    # from wikipedia
    assert GramSchmidt([Matrix([3, 1]), Matrix([2, 2])], True) == [
        Matrix([3*sqrt(10)/10, sqrt(10)/10]),
        Matrix([-sqrt(10)/10, 3*sqrt(10)/10])]
    # https://github.com/sympy/sympy/issues/9488
    L = FiniteSet(Matrix([1]))
    assert GramSchmidt(L) == [Matrix([[1]])]


def test_casoratian():
    assert casoratian([1, 2, 3, 4], 1) == 0
    assert casoratian([1, 2, 3, 4], 1, zero=False) == 0


def test_zero_dimension_multiply():
    assert (Matrix()*zeros(0, 3)).shape == (0, 3)
    assert zeros(3, 0)*zeros(0, 3) == zeros(3, 3)
    assert zeros(0, 3)*zeros(3, 0) == Matrix()


def test_slice_issue_2884():
    m = Matrix(2, 2, range(4))
    assert m[1, :] == Matrix([[2, 3]])
    assert m[-1, :] == Matrix([[2, 3]])
    assert m[:, 1] == Matrix([[1, 3]]).T
    assert m[:, -1] == Matrix([[1, 3]]).T
    raises(IndexError, lambda: m[2, :])
    raises(IndexError, lambda: m[2, 2])


def test_slice_issue_3401():
    assert zeros(0, 3)[:, -1].shape == (0, 1)
    assert zeros(3, 0)[0, :] == Matrix(1, 0, [])


def test_copyin():
    s = zeros(3, 3)
    s[3] = 1
    assert s[:, 0] == Matrix([0, 1, 0])
    assert s[3] == 1
    assert s[3: 4] == [1]
    s[1, 1] = 42
    assert s[1, 1] == 42
    assert s[1, 1:] == Matrix([[42, 0]])
    s[1, 1:] = Matrix([[5, 6]])
    assert s[1, :] == Matrix([[1, 5, 6]])
    s[1, 1:] = [[42, 43]]
    assert s[1, :] == Matrix([[1, 42, 43]])
    s[0, 0] = 17
    assert s[:, :1] == Matrix([17, 1, 0])
    s[0, 0] = [1, 1, 1]
    assert s[:, 0] == Matrix([1, 1, 1])
    s[0, 0] = Matrix([1, 1, 1])
    assert s[:, 0] == Matrix([1, 1, 1])
    s[0, 0] = SparseMatrix([1, 1, 1])
    assert s[:, 0] == Matrix([1, 1, 1])


def test_invertible_check():
    # sometimes a singular matrix will have a pivot vector shorter than
    # the number of rows in a matrix...
    assert Matrix([[1, 2], [1, 2]]).rref() == (Matrix([[1, 2], [0, 0]]), (0,))
    raises(ValueError, lambda: Matrix([[1, 2], [1, 2]]).inv())
    m = Matrix([
        [-1, -1,  0],
        [ x,  1,  1],
        [ 1,  x, -1],
    ])
    assert len(m.rref()[1]) != m.rows
    # in addition, unless simplify=True in the call to rref, the identity
    # matrix will be returned even though m is not invertible
    assert m.rref()[0] != eye(3)
    assert m.rref(simplify=signsimp)[0] != eye(3)
    raises(ValueError, lambda: m.inv(method="ADJ"))
    raises(ValueError, lambda: m.inv(method="GE"))
    raises(ValueError, lambda: m.inv(method="LU"))


def test_issue_3959():
    x, y = symbols('x, y')
    e = x*y
    assert e.subs(x, Matrix([3, 5, 3])) == Matrix([3, 5, 3])*y


def test_issue_5964():
    assert str(Matrix([[1, 2], [3, 4]])) == 'Matrix([[1, 2], [3, 4]])'


def test_issue_7604():
    x, y = symbols("x y")
    assert sstr(Matrix([[x, 2*y], [y**2, x + 3]])) == \
        'Matrix([\n[   x,   2*y],\n[y**2, x + 3]])'


def test_is_Identity():
    assert eye(3).is_Identity
    assert eye(3).as_immutable().is_Identity
    assert not zeros(3).is_Identity
    assert not ones(3).is_Identity
    # issue 6242
    assert not Matrix([[1, 0, 0]]).is_Identity
    # issue 8854
    assert SparseMatrix(3,3, {(0,0):1, (1,1):1, (2,2):1}).is_Identity
    assert not SparseMatrix(2,3, range(6)).is_Identity
    assert not SparseMatrix(3,3, {(0,0):1, (1,1):1}).is_Identity
    assert not SparseMatrix(3,3, {(0,0):1, (1,1):1, (2,2):1, (0,1):2, (0,2):3}).is_Identity


def test_dot():
    assert ones(1, 3).dot(ones(3, 1)) == 3
    assert ones(1, 3).dot([1, 1, 1]) == 3
    assert Matrix([1, 2, 3]).dot(Matrix([1, 2, 3])) == 14
    assert Matrix([1, 2, 3*I]).dot(Matrix([I, 2, 3*I])) == -5 + I
    assert Matrix([1, 2, 3*I]).dot(Matrix([I, 2, 3*I]), hermitian=False) == -5 + I
    assert Matrix([1, 2, 3*I]).dot(Matrix([I, 2, 3*I]), hermitian=True) == 13 + I
    assert Matrix([1, 2, 3*I]).dot(Matrix([I, 2, 3*I]), hermitian=True, conjugate_convention="physics") == 13 - I
    assert Matrix([1, 2, 3*I]).dot(Matrix([4, 5*I, 6]), hermitian=True, conjugate_convention="right") == 4 + 8*I
    assert Matrix([1, 2, 3*I]).dot(Matrix([4, 5*I, 6]), hermitian=True, conjugate_convention="left") == 4 - 8*I
    assert Matrix([I, 2*I]).dot(Matrix([I, 2*I]), hermitian=False, conjugate_convention="left") == -5
    assert Matrix([I, 2*I]).dot(Matrix([I, 2*I]), conjugate_convention="left") == 5
    raises(ValueError, lambda: Matrix([1, 2]).dot(Matrix([3, 4]), hermitian=True, conjugate_convention="test"))


def test_dual():
    B_x, B_y, B_z, E_x, E_y, E_z = symbols(
        'B_x B_y B_z E_x E_y E_z', real=True)
    F = Matrix((
        (   0,  E_x,  E_y,  E_z),
        (-E_x,    0,  B_z, -B_y),
        (-E_y, -B_z,    0,  B_x),
        (-E_z,  B_y, -B_x,    0)
    ))
    Fd = Matrix((
        (  0, -B_x, -B_y, -B_z),
        (B_x,    0,  E_z, -E_y),
        (B_y, -E_z,    0,  E_x),
        (B_z,  E_y, -E_x,    0)
    ))
    assert F.dual().equals(Fd)
    assert eye(3).dual().equals(zeros(3))
    assert F.dual().dual().equals(-F)


def test_anti_symmetric():
    assert Matrix([1, 2]).is_anti_symmetric() is False
    m = Matrix(3, 3, [0, x**2 + 2*x + 1, y, -(x + 1)**2, 0, x*y, -y, -x*y, 0])
    assert m.is_anti_symmetric() is True
    assert m.is_anti_symmetric(simplify=False) is None
    assert m.is_anti_symmetric(simplify=lambda x: x) is None

    # tweak to fail
    m[2, 1] = -m[2, 1]
    assert m.is_anti_symmetric() is None
    # untweak
    m[2, 1] = -m[2, 1]

    m = m.expand()
    assert m.is_anti_symmetric(simplify=False) is True
    m[0, 0] = 1
    assert m.is_anti_symmetric() is False


def test_normalize_sort_diogonalization():
    A = Matrix(((1, 2), (2, 1)))
    P, Q = A.diagonalize(normalize=True)
    assert P*P.T == P.T*P == eye(P.cols)
    P, Q = A.diagonalize(normalize=True, sort=True)
    assert P*P.T == P.T*P == eye(P.cols)
    assert P*Q*P.inv() == A


def test_issue_5321():
    raises(ValueError, lambda: Matrix([[1, 2, 3], Matrix(0, 1, [])]))


def test_issue_5320():
    assert Matrix.hstack(eye(2), 2*eye(2)) == Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]
    ])
    assert Matrix.vstack(eye(2), 2*eye(2)) == Matrix([
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2]
    ])
    cls = SparseMatrix
    assert cls.hstack(cls(eye(2)), cls(2*eye(2))) == Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2]
    ])

def test_issue_11944():
    A = Matrix([[1]])
    AIm = sympify(A)
    assert Matrix.hstack(AIm, A) == Matrix([[1, 1]])
    assert Matrix.vstack(AIm, A) == Matrix([[1], [1]])

def test_cross():
    a = [1, 2, 3]
    b = [3, 4, 5]
    col = Matrix([-2, 4, -2])
    row = col.T

    def test(M, ans):
        assert ans == M
        assert type(M) == cls
    for cls in classes:
        A = cls(a)
        B = cls(b)
        test(A.cross(B), col)
        test(A.cross(B.T), col)
        test(A.T.cross(B.T), row)
        test(A.T.cross(B), row)
    raises(ShapeError, lambda:
        Matrix(1, 2, [1, 1]).cross(Matrix(1, 2, [1, 1])))

def test_hat_vee():
    v1 = Matrix([x, y, z])
    v2 = Matrix([a, b, c])
    assert v1.hat() * v2 == v1.cross(v2)
    assert v1.hat().is_anti_symmetric()
    assert v1.hat().vee() == v1

def test_hash():
    for cls in classes[-2:]:
        s = {cls.eye(1), cls.eye(1)}
        assert len(s) == 1 and s.pop() == cls.eye(1)
    # issue 3979
    for cls in classes[:2]:
        assert not isinstance(cls.eye(1), Hashable)


@XFAIL
def test_issue_3979():
    # when this passes, delete this and change the [1:2]
    # to [:2] in the test_hash above for issue 3979
    cls = classes[0]
    raises(AttributeError, lambda: hash(cls.eye(1)))


def test_adjoint():
    dat = [[0, I], [1, 0]]
    ans = Matrix([[0, 1], [-I, 0]])
    for cls in classes:
        assert ans == cls(dat).adjoint()


def test_adjoint_with_operator():
    # Regression test for issue 25130: adjoint() should propagate to operators
    import sympy.physics.quantum
    a = sympy.physics.quantum.operator.Operator('a')
    a_dag = sympy.physics.quantum.Dagger(a)
    dat = [[0, I * a], [0, a_dag]]
    ans = Matrix([[0, 0], [-I * a_dag, a]])
    for cls in classes:
        assert ans == cls(dat).adjoint()


def test_simplify_immutable():
    assert simplify(ImmutableMatrix([[sin(x)**2 + cos(x)**2]])) == \
                    ImmutableMatrix([[1]])

def test_replace():
    F, G = symbols('F, G', cls=Function)
    K = Matrix(2, 2, lambda i, j: G(i+j))
    M = Matrix(2, 2, lambda i, j: F(i+j))
    N = M.replace(F, G)
    assert N == K


def test_atoms():
    m = Matrix([[1, 2], [x, 1 - 1/x]])
    assert m.atoms() == {S.One,S(2),S.NegativeOne, x}
    assert m.atoms(Symbol) == {x}


def test_pinv():
    # Pseudoinverse of an invertible matrix is the inverse.
    A1 = Matrix([[a, b], [c, d]])
    assert simplify(A1.pinv(method="RD")) == simplify(A1.inv())

    # Test the four properties of the pseudoinverse for various matrices.
    As = [Matrix([[13, 104], [2212, 3], [-3, 5]]),
          Matrix([[1, 7, 9], [11, 17, 19]]),
          Matrix([a, b])]

    for A in As:
        A_pinv = A.pinv(method="RD")
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA

    # XXX Pinv with diagonalization makes expression too complicated.
    for A in As:
        A_pinv = simplify(A.pinv(method="ED"))
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA

    # XXX Computing pinv using diagonalization makes an expression that
    # is too complicated to simplify.
    # A1 = Matrix([[a, b], [c, d]])
    # assert simplify(A1.pinv(method="ED")) == simplify(A1.inv())
    # so this is tested numerically at a fixed random point

    from sympy.core.numbers import comp
    q = A1.pinv(method="ED")
    w = A1.inv()
    reps = {a: -73633, b: 11362, c: 55486, d: 62570}
    assert all(
        comp(i.n(), j.n())
        for i, j in zip(q.subs(reps), w.subs(reps))
        )


@slow
def test_pinv_rank_deficient_when_diagonalization_fails():
    # Test the four properties of the pseudoinverse for matrices when
    # diagonalization of A.H*A fails.
    As = [
        Matrix([
            [61, 89, 55, 20, 71, 0],
            [62, 96, 85, 85, 16, 0],
            [69, 56, 17,  4, 54, 0],
            [10, 54, 91, 41, 71, 0],
            [ 7, 30, 10, 48, 90, 0],
            [0, 0, 0, 0, 0, 0]])
    ]
    for A in As:
        A_pinv = A.pinv(method="ED")
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert AAp.H == AAp

        # Here ApA.H and ApA are equivalent expressions but they are very
        # complicated expressions involving RootOfs. Using simplify would be
        # too slow and so would evalf so we substitute approximate values for
        # the RootOfs and then evalf which is less accurate but good enough to
        # confirm that these two matrices are equivalent.
        #
        # assert ApA.H == ApA  # <--- would fail (structural equality)
        # assert simplify(ApA.H - ApA).is_zero_matrix  # <--- too slow
        # (ApA.H - ApA).evalf()  # <--- too slow

        def allclose(M1, M2):
            rootofs = M1.atoms(RootOf)
            rootofs_approx = {r: r.evalf() for r in rootofs}
            diff_approx = (M1 - M2).xreplace(rootofs_approx).evalf()
            return all(abs(e) < 1e-10 for e in diff_approx)

        assert allclose(ApA.H, ApA)


def test_issue_7201():
    assert ones(0, 1) + ones(0, 1) == Matrix(0, 1, [])
    assert ones(1, 0) + ones(1, 0) == Matrix(1, 0, [])

def test_free_symbols():
    for M in ImmutableMatrix, ImmutableSparseMatrix, Matrix, SparseMatrix:
        assert M([[x], [0]]).free_symbols == {x}

def test_from_ndarray():
    """See issue 7465."""
    try:
        from numpy import array
    except ImportError:
        skip('NumPy must be available to test creating matrices from ndarrays')

    assert Matrix(array([1, 2, 3])) == Matrix([1, 2, 3])
    assert Matrix(array([[1, 2, 3]])) == Matrix([[1, 2, 3]])
    assert Matrix(array([[1, 2, 3], [4, 5, 6]])) == \
        Matrix([[1, 2, 3], [4, 5, 6]])
    assert Matrix(array([x, y, z])) == Matrix([x, y, z])
    raises(NotImplementedError,
        lambda: Matrix(array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])))
    assert Matrix([array([1, 2]), array([3, 4])]) == Matrix([[1, 2], [3, 4]])
    assert Matrix([array([1, 2]), [3, 4]]) == Matrix([[1, 2], [3, 4]])
    assert Matrix([array([]), array([])]) == Matrix(2, 0, []) != Matrix(0, 0, [])

def test_17522_numpy():
    from sympy.matrices.common import _matrixify
    try:
        from numpy import array, matrix
    except ImportError:
        skip('NumPy must be available to test indexing matrixified NumPy ndarrays and matrices')

    m = _matrixify(array([[1, 2], [3, 4]]))
    assert m[3] == 4
    assert list(m) == [1, 2, 3, 4]

    with ignore_warnings(PendingDeprecationWarning):
        m = _matrixify(matrix([[1, 2], [3, 4]]))
    assert m[3] == 4
    assert list(m) == [1, 2, 3, 4]

def test_17522_mpmath():
    from sympy.matrices.common import _matrixify
    try:
        from mpmath import matrix
    except ImportError:
        skip('mpmath must be available to test indexing matrixified mpmath matrices')

    m = _matrixify(matrix([[1, 2], [3, 4]]))
    assert m[3] == 4.0
    assert list(m) == [1.0, 2.0, 3.0, 4.0]

def test_17522_scipy():
    from sympy.matrices.common import _matrixify
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        skip('SciPy must be available to test indexing matrixified SciPy sparse matrices')

    m = _matrixify(csr_matrix([[1, 2], [3, 4]]))
    assert m[3] == 4
    assert list(m) == [1, 2, 3, 4]

def test_hermitian():
    a = Matrix([[1, I], [-I, 1]])
    assert a.is_hermitian
    a[0, 0] = 2*I
    assert a.is_hermitian is False
    a[0, 0] = x
    assert a.is_hermitian is None
    a[0, 1] = a[1, 0]*I
    assert a.is_hermitian is False
    b = HermitianOperator("b")
    c = Operator("c")
    assert Matrix([[b]]).is_hermitian is True
    assert Matrix([[b, c], [Dagger(c), b]]).is_hermitian is True
    assert Matrix([[b, c], [c, b]]).is_hermitian is False
    assert Matrix([[b, c], [transpose(c), b]]).is_hermitian is False

def test_doit():
    a = Matrix([[Add(x,x, evaluate=False)]])
    assert a[0] != 2*x
    assert a.doit() == Matrix([[2*x]])

def test_issue_9457_9467_9876():
    # for row_del(index)
    M = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    M.row_del(1)
    assert M == Matrix([[1, 2, 3], [3, 4, 5]])
    N = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    N.row_del(-2)
    assert N == Matrix([[1, 2, 3], [3, 4, 5]])
    O = Matrix([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
    O.row_del(-1)
    assert O == Matrix([[1, 2, 3], [5, 6, 7]])
    P = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    raises(IndexError, lambda: P.row_del(10))
    Q = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    raises(IndexError, lambda: Q.row_del(-10))

    # for col_del(index)
    M = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    M.col_del(1)
    assert M == Matrix([[1, 3], [2, 4], [3, 5]])
    N = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    N.col_del(-2)
    assert N == Matrix([[1, 3], [2, 4], [3, 5]])
    P = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    raises(IndexError, lambda: P.col_del(10))
    Q = Matrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    raises(IndexError, lambda: Q.col_del(-10))

def test_issue_9422():
    x, y = symbols('x y', commutative=False)
    a, b = symbols('a b')
    M = eye(2)
    M1 = Matrix(2, 2, [x, y, y, z])
    assert y*x*M != x*y*M
    assert b*a*M == a*b*M
    assert x*M1 != M1*x
    assert a*M1 == M1*a
    assert y*x*M == Matrix([[y*x, 0], [0, y*x]])


def test_issue_10770():
    M = Matrix([])
    a = ['col_insert', 'row_join'], Matrix([9, 6, 3])
    b = ['row_insert', 'col_join'], a[1].T
    c = ['row_insert', 'col_insert'], Matrix([[1, 2], [3, 4]])
    for ops, m in (a, b, c):
        for op in ops:
            f = getattr(M, op)
            new = f(m) if 'join' in op else f(42, m)
            assert new == m and id(new) != id(m)


def test_issue_10658():
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A.extract([0, 1, 2], [True, True, False]) == \
        Matrix([[1, 2], [4, 5], [7, 8]])
    assert A.extract([0, 1, 2], [True, False, False]) == Matrix([[1], [4], [7]])
    assert A.extract([True, False, False], [0, 1, 2]) == Matrix([[1, 2, 3]])
    assert A.extract([True, False, True], [0, 1, 2]) == \
        Matrix([[1, 2, 3], [7, 8, 9]])
    assert A.extract([0, 1, 2], [False, False, False]) == Matrix(3, 0, [])
    assert A.extract([False, False, False], [0, 1, 2]) == Matrix(0, 3, [])
    assert A.extract([True, False, True], [False, True, False]) == \
        Matrix([[2], [8]])

def test_opportunistic_simplification():
    # this test relates to issue #10718, #9480, #11434

    # issue #9480
    m = Matrix([[-5 + 5*sqrt(2), -5], [-5*sqrt(2)/2 + 5, -5*sqrt(2)/2]])
    assert m.rank() == 1

    # issue #10781
    m = Matrix([[3+3*sqrt(3)*I, -9],[4,-3+3*sqrt(3)*I]])
    assert simplify(m.rref()[0] - Matrix([[1, -9/(3 + 3*sqrt(3)*I)], [0, 0]])) == zeros(2, 2)

    # issue #11434
    ax,ay,bx,by,cx,cy,dx,dy,ex,ey,t0,t1 = symbols('a_x a_y b_x b_y c_x c_y d_x d_y e_x e_y t_0 t_1')
    m = Matrix([[ax,ay,ax*t0,ay*t0,0],[bx,by,bx*t0,by*t0,0],[cx,cy,cx*t0,cy*t0,1],[dx,dy,dx*t0,dy*t0,1],[ex,ey,2*ex*t1-ex*t0,2*ey*t1-ey*t0,0]])
    assert m.rank() == 4

def test_partial_pivoting():
    # example from https://en.wikipedia.org/wiki/Pivot_element
    # partial pivoting with back substitution gives a perfect result
    # naive pivoting give an error ~1e-13, so anything better than
    # 1e-15 is good
    mm=Matrix([[0.003, 59.14, 59.17], [5.291, -6.13, 46.78]])
    assert (mm.rref()[0] - Matrix([[1.0,   0, 10.0],
                                   [  0, 1.0,  1.0]])).norm() < 1e-15

    # issue #11549
    m_mixed = Matrix([[6e-17, 1.0, 4],
                      [ -1.0,   0, 8],
                      [    0,   0, 1]])
    m_float = Matrix([[6e-17,  1.0, 4.],
                      [ -1.0,   0., 8.],
                      [   0.,   0., 1.]])
    m_inv = Matrix([[  0,    -1.0,  8.0],
                    [1.0, 6.0e-17, -4.0],
                    [  0,       0,    1]])
    # this example is numerically unstable and involves a matrix with a norm >= 8,
    # this comparing the difference of the results with 1e-15 is numerically sound.
    assert (m_mixed.inv() - m_inv).norm() < 1e-15
    assert (m_float.inv() - m_inv).norm() < 1e-15

def test_iszero_substitution():
    """ When doing numerical computations, all elements that pass
    the iszerofunc test should be set to numerically zero if they
    aren't already. """

    # Matrix from issue #9060
    m = Matrix([[0.9, -0.1, -0.2, 0],[-0.8, 0.9, -0.4, 0],[-0.1, -0.8, 0.6, 0]])
    m_rref = m.rref(iszerofunc=lambda x: abs(x)<6e-15)[0]
    m_correct = Matrix([[1.0,   0, -0.301369863013699, 0],[  0, 1.0, -0.712328767123288, 0],[  0,   0,                  0, 0]])
    m_diff = m_rref - m_correct
    assert m_diff.norm() < 1e-15
    # if a zero-substitution wasn't made, this entry will be -1.11022302462516e-16
    assert m_rref[2,2] == 0

def test_issue_11238():
    from sympy.geometry.point import Point
    xx = 8*tan(pi*Rational(13, 45))/(tan(pi*Rational(13, 45)) + sqrt(3))
    yy = (-8*sqrt(3)*tan(pi*Rational(13, 45))**2 + 24*tan(pi*Rational(13, 45)))/(-3 + tan(pi*Rational(13, 45))**2)
    p1 = Point(0, 0)
    p2 = Point(1, -sqrt(3))
    p0 = Point(xx,yy)
    m1 = Matrix([p1 - simplify(p0), p2 - simplify(p0)])
    m2 = Matrix([p1 - p0, p2 - p0])
    m3 = Matrix([simplify(p1 - p0), simplify(p2 - p0)])

    # This system has expressions which are zero and
    # cannot be easily proved to be such, so without
    # numerical testing, these assertions will fail.
    Z = lambda x: abs(x.n()) < 1e-20
    assert m1.rank(simplify=True, iszerofunc=Z) == 1
    assert m2.rank(simplify=True, iszerofunc=Z) == 1
    assert m3.rank(simplify=True, iszerofunc=Z) == 1

def test_as_real_imag():
    m1 = Matrix(2,2,[1,2,3,4])
    m2 = m1*S.ImaginaryUnit
    m3 = m1 + m2

    for kls in classes:
        a,b = kls(m3).as_real_imag()
        assert list(a) == list(m1)
        assert list(b) == list(m1)

def test_deprecated():
    # Maintain tests for deprecated functions.  We must capture
    # the deprecation warnings.  When the deprecated functionality is
    # removed, the corresponding tests should be removed.

    m = Matrix(3, 3, [0, 1, 0, -4, 4, 0, -2, 1, 2])
    P, Jcells = m.jordan_cells()
    assert Jcells[1] == Matrix(1, 1, [2])
    assert Jcells[0] == Matrix(2, 2, [2, 1, 0, 2])


def test_issue_14489():
    from sympy.core.mod import Mod
    A = Matrix([-1, 1, 2])
    B = Matrix([10, 20, -15])

    assert Mod(A, 3) == Matrix([2, 1, 2])
    assert Mod(B, 4) == Matrix([2, 0, 1])

def test_issue_14943():
    # Test that __array__ accepts the optional dtype argument
    try:
        from numpy import array
    except ImportError:
        skip('NumPy must be available to test creating matrices from ndarrays')

    M = Matrix([[1,2], [3,4]])
    assert array(M, dtype=float).dtype.name == 'float64'

def test_case_6913():
    m = MatrixSymbol('m', 1, 1)
    a = Symbol("a")
    a = m[0, 0]>0
    assert str(a) == 'm[0, 0] > 0'

def test_issue_11948():
    A = MatrixSymbol('A', 3, 3)
    a = Wild('a')
    assert A.match(a) == {a: A}

def test_gramschmidt_conjugate_dot():
    vecs = [Matrix([1, I]), Matrix([1, -I])]
    assert Matrix.orthogonalize(*vecs) == \
        [Matrix([[1], [I]]), Matrix([[1], [-I]])]

    vecs = [Matrix([1, I, 0]), Matrix([I, 0, -I])]
    assert Matrix.orthogonalize(*vecs) == \
        [Matrix([[1], [I], [0]]), Matrix([[I/2], [S(1)/2], [-I]])]

    mat = Matrix([[1, I], [1, -I]])
    Q, R = mat.QRdecomposition()
    assert Q * Q.H == Matrix.eye(2)

def test_issue_8207():
    a = Matrix(MatrixSymbol('a', 3, 1))
    b = Matrix(MatrixSymbol('b', 3, 1))
    c = a.dot(b)
    d = diff(c, a[0, 0])
    e = diff(d, a[0, 0])
    assert d == b[0, 0]
    assert e == 0

def test_func():
    from sympy.simplify.simplify import nthroot

    A = Matrix([[1, 2],[0, 3]])
    assert A.analytic_func(sin(x*t), x) == Matrix([[sin(t), sin(3*t) - sin(t)], [0, sin(3*t)]])

    A = Matrix([[2, 1],[1, 2]])
    assert (pi * A / 6).analytic_func(cos(x), x) == Matrix([[sqrt(3)/4, -sqrt(3)/4], [-sqrt(3)/4, sqrt(3)/4]])


    raises(ValueError, lambda : zeros(5).analytic_func(log(x), x))
    raises(ValueError, lambda : (A*x).analytic_func(log(x), x))

    A = Matrix([[0, -1, -2, 3], [0, -1, -2, 3], [0, 1, 0, -1], [0, 0, -1, 1]])
    assert A.analytic_func(exp(x), x) == A.exp()
    raises(ValueError, lambda : A.analytic_func(sqrt(x), x))

    A = Matrix([[41, 12],[12, 34]])
    assert simplify(A.analytic_func(sqrt(x), x)**2) == A

    A = Matrix([[3, -12, 4], [-1, 0, -2], [-1, 5, -1]])
    assert simplify(A.analytic_func(nthroot(x, 3), x)**3) == A

    A = Matrix([[2, 0, 0, 0], [1, 2, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3]])
    assert A.analytic_func(exp(x), x) == A.exp()

    A = Matrix([[0, 2, 1, 6], [0, 0, 1, 2], [0, 0, 0, 3], [0, 0, 0, 0]])
    assert A.analytic_func(exp(x*t), x) == expand(simplify((A*t).exp()))


@skip_under_pyodide("Cannot create threads under pyodide.")
def test_issue_19809():

    def f():
        assert _dotprodsimp_state.state == None
        m = Matrix([[1]])
        m = m * m
        return True

    with dotprodsimp(True):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(f)
            assert future.result()


def test_issue_23276():
    M = Matrix([x, y])
    assert integrate(M, (x, 0, 1), (y, 0, 1)) == Matrix([
        [S.Half],
        [S.Half]])


# SubspaceOnlyMatrix tests
def test_columnspace_one():
    m = SubspaceOnlyMatrix([[ 1,  2,  0,  2,  5],
                            [-2, -5,  1, -1, -8],
                            [ 0, -3,  3,  4,  1],
                            [ 3,  6,  0, -7,  2]])

    basis = m.columnspace()
    assert basis[0] == Matrix([1, -2, 0, 3])
    assert basis[1] == Matrix([2, -5, -3, 6])
    assert basis[2] == Matrix([2, -1, 4, -7])

    assert len(basis) == 3
    assert Matrix.hstack(m, *basis).columnspace() == basis


def test_rowspace():
    m = SubspaceOnlyMatrix([[ 1,  2,  0,  2,  5],
                            [-2, -5,  1, -1, -8],
                            [ 0, -3,  3,  4,  1],
                            [ 3,  6,  0, -7,  2]])

    basis = m.rowspace()
    assert basis[0] == Matrix([[1, 2, 0, 2, 5]])
    assert basis[1] == Matrix([[0, -1, 1, 3, 2]])
    assert basis[2] == Matrix([[0, 0, 0, 5, 5]])

    assert len(basis) == 3


def test_nullspace_one():
    m = SubspaceOnlyMatrix([[ 1,  2,  0,  2,  5],
                            [-2, -5,  1, -1, -8],
                            [ 0, -3,  3,  4,  1],
                            [ 3,  6,  0, -7,  2]])

    basis = m.nullspace()
    assert basis[0] == Matrix([-2, 1, 1, 0, 0])
    assert basis[1] == Matrix([-1, -1, 0, -1, 1])
    # make sure the null space is really gets zeroed
    assert all(e.is_zero for e in m*basis[0])
    assert all(e.is_zero for e in m*basis[1])


# ReductionsOnlyMatrix tests
def test_row_op():
    e = eye_Reductions(3)

    raises(ValueError, lambda: e.elementary_row_op("abc"))
    raises(ValueError, lambda: e.elementary_row_op())
    raises(ValueError, lambda: e.elementary_row_op('n->kn', row=5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->kn', row=-5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=1, row2=5))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=5, row2=1))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=-5, row2=1))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=1, row2=-5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=1, row2=5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=5, row2=1, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=-5, row2=1, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=1, row2=-5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=1, row2=1, k=5))

    # test various ways to set arguments
    assert e.elementary_row_op("n->kn", 0, 5) == Matrix([[5, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->kn", 1, 5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->kn", row=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->kn", row1=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_row_op("n<->m", 0, 1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_row_op("n<->m", row1=0, row2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_row_op("n<->m", row=0, row2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->n+km", 0, 5, 1) == Matrix([[1, 5, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->n+km", row=0, k=5, row2=1) == Matrix([[1, 5, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->n+km", row1=0, k=5, row2=1) == Matrix([[1, 5, 0], [0, 1, 0], [0, 0, 1]])

    # make sure the matrix doesn't change size
    a = ReductionsOnlyMatrix(2, 3, [0]*6)
    assert a.elementary_row_op("n->kn", 1, 5) == Matrix(2, 3, [0]*6)
    assert a.elementary_row_op("n<->m", 0, 1) == Matrix(2, 3, [0]*6)
    assert a.elementary_row_op("n->n+km", 0, 5, 1) == Matrix(2, 3, [0]*6)


def test_col_op():
    e = eye_Reductions(3)

    raises(ValueError, lambda: e.elementary_col_op("abc"))
    raises(ValueError, lambda: e.elementary_col_op())
    raises(ValueError, lambda: e.elementary_col_op('n->kn', col=5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->kn', col=-5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=1, col2=5))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=5, col2=1))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=-5, col2=1))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=1, col2=-5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=5, col2=1, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=-5, col2=1, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=-5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=1, k=5))

    # test various ways to set arguments
    assert e.elementary_col_op("n->kn", 0, 5) == Matrix([[5, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->kn", 1, 5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->kn", col=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->kn", col1=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op("n<->m", 0, 1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op("n<->m", col1=0, col2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op("n<->m", col=0, col2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->n+km", 0, 5, 1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->n+km", col=0, k=5, col2=1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->n+km", col1=0, k=5, col2=1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])

    # make sure the matrix doesn't change size
    a = ReductionsOnlyMatrix(2, 3, [0]*6)
    assert a.elementary_col_op("n->kn", 1, 5) == Matrix(2, 3, [0]*6)
    assert a.elementary_col_op("n<->m", 0, 1) == Matrix(2, 3, [0]*6)
    assert a.elementary_col_op("n->n+km", 0, 5, 1) == Matrix(2, 3, [0]*6)


def test_is_echelon():
    zro = zeros_Reductions(3)
    ident = eye_Reductions(3)

    assert zro.is_echelon
    assert ident.is_echelon

    a = ReductionsOnlyMatrix(0, 0, [])
    assert a.is_echelon

    a = ReductionsOnlyMatrix(2, 3, [3, 2, 1, 0, 0, 6])
    assert a.is_echelon

    a = ReductionsOnlyMatrix(2, 3, [0, 0, 6, 3, 2, 1])
    assert not a.is_echelon

    x = Symbol('x')
    a = ReductionsOnlyMatrix(3, 1, [x, 0, 0])
    assert a.is_echelon

    a = ReductionsOnlyMatrix(3, 1, [x, x, 0])
    assert not a.is_echelon

    a = ReductionsOnlyMatrix(3, 3, [0, 0, 0, 1, 2, 3, 0, 0, 0])
    assert not a.is_echelon


def test_echelon_form():
    # echelon form is not unique, but the result
    # must be row-equivalent to the original matrix
    # and it must be in echelon form.

    a = zeros_Reductions(3)
    e = eye_Reductions(3)

    # we can assume the zero matrix and the identity matrix shouldn't change
    assert a.echelon_form() == a
    assert e.echelon_form() == e

    a = ReductionsOnlyMatrix(0, 0, [])
    assert a.echelon_form() == a

    a = ReductionsOnlyMatrix(1, 1, [5])
    assert a.echelon_form() == a

    # now we get to the real tests

    def verify_row_null_space(mat, rows, nulls):
        for v in nulls:
            assert all(t.is_zero for t in a_echelon*v)
        for v in rows:
            if not all(t.is_zero for t in v):
                assert not all(t.is_zero for t in a_echelon*v.transpose())

    a = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    nulls = [Matrix([
                     [ 1],
                     [-2],
                     [ 1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)


    a = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 8])
    nulls = []
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)

    a = ReductionsOnlyMatrix(3, 3, [2, 1, 3, 0, 0, 0, 2, 1, 3])
    nulls = [Matrix([
             [Rational(-1, 2)],
             [   1],
             [   0]]),
             Matrix([
             [Rational(-3, 2)],
             [   0],
             [   1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)

    # this one requires a row swap
    a = ReductionsOnlyMatrix(3, 3, [2, 1, 3, 0, 0, 0, 1, 1, 3])
    nulls = [Matrix([
             [   0],
             [  -3],
             [   1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)

    a = ReductionsOnlyMatrix(3, 3, [0, 3, 3, 0, 2, 2, 0, 1, 1])
    nulls = [Matrix([
             [1],
             [0],
             [0]]),
             Matrix([
             [ 0],
             [-1],
             [ 1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)

    a = ReductionsOnlyMatrix(2, 3, [2, 2, 3, 3, 3, 0])
    nulls = [Matrix([
             [-1],
             [1],
             [0]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)


def test_rref():
    e = ReductionsOnlyMatrix(0, 0, [])
    assert e.rref(pivots=False) == e

    e = ReductionsOnlyMatrix(1, 1, [1])
    a = ReductionsOnlyMatrix(1, 1, [5])
    assert e.rref(pivots=False) == a.rref(pivots=False) == e

    a = ReductionsOnlyMatrix(3, 1, [1, 2, 3])
    assert a.rref(pivots=False) == Matrix([[1], [0], [0]])

    a = ReductionsOnlyMatrix(1, 3, [1, 2, 3])
    assert a.rref(pivots=False) == Matrix([[1, 2, 3]])

    a = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert a.rref(pivots=False) == Matrix([
                                     [1, 0, -1],
                                     [0, 1,  2],
                                     [0, 0,  0]])

    a = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    b = ReductionsOnlyMatrix(3, 3, [1, 2, 3, 0, 0, 0, 0, 0, 0])
    c = ReductionsOnlyMatrix(3, 3, [0, 0, 0, 1, 2, 3, 0, 0, 0])
    d = ReductionsOnlyMatrix(3, 3, [0, 0, 0, 0, 0, 0, 1, 2, 3])
    assert a.rref(pivots=False) == \
            b.rref(pivots=False) == \
            c.rref(pivots=False) == \
            d.rref(pivots=False) == b

    e = eye_Reductions(3)
    z = zeros_Reductions(3)
    assert e.rref(pivots=False) == e
    assert z.rref(pivots=False) == z

    a = ReductionsOnlyMatrix([
            [ 0, 0,  1,  2,  2, -5,  3],
            [-1, 5,  2,  2,  1, -7,  5],
            [ 0, 0, -2, -3, -3,  8, -5],
            [-1, 5,  0, -1, -2,  1,  0]])
    mat, pivot_offsets = a.rref()
    assert mat == Matrix([
                     [1, -5, 0, 0, 1,  1, -1],
                     [0,  0, 1, 0, 0, -1,  1],
                     [0,  0, 0, 1, 1, -2,  1],
                     [0,  0, 0, 0, 0,  0,  0]])
    assert pivot_offsets == (0, 2, 3)

    a = ReductionsOnlyMatrix([[Rational(1, 19),  Rational(1, 5),    2,    3],
                        [   4,    5,    6,    7],
                        [   8,    9,   10,   11],
                        [  12,   13,   14,   15]])
    assert a.rref(pivots=False) == Matrix([
                                         [1, 0, 0, Rational(-76, 157)],
                                         [0, 1, 0,  Rational(-5, 157)],
                                         [0, 0, 1, Rational(238, 157)],
                                         [0, 0, 0,       0]])

    x = Symbol('x')
    a = ReductionsOnlyMatrix(2, 3, [x, 1, 1, sqrt(x), x, 1])
    for i, j in zip(a.rref(pivots=False),
            [1, 0, sqrt(x)*(-x + 1)/(-x**Rational(5, 2) + x),
                0, 1, 1/(sqrt(x) + x + 1)]):
        assert simplify(i - j).is_zero


def test_rref_rhs():
    a, b, c, d = symbols('a b c d')
    A = Matrix([[0, 0], [0, 0], [1, 2], [3, 4]])
    B = Matrix([a, b, c, d])
    assert A.rref_rhs(B) == (Matrix([
    [1, 0],
    [0, 1],
    [0, 0],
    [0, 0]]), Matrix([
    [   -2*c + d],
    [3*c/2 - d/2],
    [          a],
    [          b]]))


def test_issue_17827():
    C = Matrix([
        [3, 4, -1, 1],
        [9, 12, -3, 3],
        [0, 2, 1, 3],
        [2, 3, 0, -2],
        [0, 3, 3, -5],
        [8, 15, 0, 6]
    ])
    # Tests for row/col within valid range
    D = C.elementary_row_op('n<->m', row1=2, row2=5)
    E = C.elementary_row_op('n->n+km', row1=5, row2=3, k=-4)
    F = C.elementary_row_op('n->kn', row=5, k=2)
    assert(D[5, :] == Matrix([[0, 2, 1, 3]]))
    assert(E[5, :] == Matrix([[0, 3, 0, 14]]))
    assert(F[5, :] == Matrix([[16, 30, 0, 12]]))
    # Tests for row/col out of range
    raises(ValueError, lambda: C.elementary_row_op('n<->m', row1=2, row2=6))
    raises(ValueError, lambda: C.elementary_row_op('n->kn', row=7, k=2))
    raises(ValueError, lambda: C.elementary_row_op('n->n+km', row1=-1, row2=5, k=2))

def test_rank():
    m = Matrix([[1, 2], [x, 1 - 1/x]])
    assert m.rank() == 2
    n = Matrix(3, 3, range(1, 10))
    assert n.rank() == 2
    p = zeros(3)
    assert p.rank() == 0

def test_issue_11434():
    ax, ay, bx, by, cx, cy, dx, dy, ex, ey, t0, t1 = \
        symbols('a_x a_y b_x b_y c_x c_y d_x d_y e_x e_y t_0 t_1')
    M = Matrix([[ax, ay, ax*t0, ay*t0, 0],
                [bx, by, bx*t0, by*t0, 0],
                [cx, cy, cx*t0, cy*t0, 1],
                [dx, dy, dx*t0, dy*t0, 1],
                [ex, ey, 2*ex*t1 - ex*t0, 2*ey*t1 - ey*t0, 0]])
    assert M.rank() == 4

def test_rank_regression_from_so():
    # see:
    # https://stackoverflow.com/questions/19072700/why-does-sympy-give-me-the-wrong-answer-when-i-row-reduce-a-symbolic-matrix

    nu, lamb = symbols('nu, lambda')
    A = Matrix([[-3*nu,         1,                  0,  0],
                [ 3*nu, -2*nu - 1,                  2,  0],
                [    0,      2*nu, (-1*nu) - lamb - 2,  3],
                [    0,         0,          nu + lamb, -3]])
    expected_reduced = Matrix([[1, 0, 0, 1/(nu**2*(-lamb - nu))],
                               [0, 1, 0,    3/(nu*(-lamb - nu))],
                               [0, 0, 1,         3/(-lamb - nu)],
                               [0, 0, 0,                      0]])
    expected_pivots = (0, 1, 2)

    reduced, pivots = A.rref()

    assert simplify(expected_reduced - reduced) == zeros(*A.shape)
    assert pivots == expected_pivots

def test_issue_15872():
    A = Matrix([[1, 1, 1, 0], [-2, -1, 0, -1], [0, 0, -1, -1], [0, 0, 2, 1]])
    B = A - Matrix.eye(4) * I
    assert B.rank() == 3
    assert (B**2).rank() == 2
    assert (B**3).rank() == 2