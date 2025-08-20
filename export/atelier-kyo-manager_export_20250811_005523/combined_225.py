
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_reshape.py ===
from datetime import datetime

import numpy as np
import pytest
import pytz

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)
import pandas._testing as tm


def test_insert(idx):
    # key contained in all levels
    new_index = idx.insert(0, ("bar", "two"))
    assert new_index.equal_levels(idx)
    assert new_index[0] == ("bar", "two")

    # key not contained in all levels
    new_index = idx.insert(0, ("abc", "three"))

    exp0 = Index(list(idx.levels[0]) + ["abc"], name="first")
    tm.assert_index_equal(new_index.levels[0], exp0)
    assert new_index.names == ["first", "second"]

    exp1 = Index(list(idx.levels[1]) + ["three"], name="second")
    tm.assert_index_equal(new_index.levels[1], exp1)
    assert new_index[0] == ("abc", "three")

    # key wrong length
    msg = "Item must have length equal to number of levels"
    with pytest.raises(ValueError, match=msg):
        idx.insert(0, ("foo2",))

    left = pd.DataFrame([["a", "b", 0], ["b", "d", 1]], columns=["1st", "2nd", "3rd"])
    left.set_index(["1st", "2nd"], inplace=True)
    ts = left["3rd"].copy(deep=True)

    left.loc[("b", "x"), "3rd"] = 2
    left.loc[("b", "a"), "3rd"] = -1
    left.loc[("b", "b"), "3rd"] = 3
    left.loc[("a", "x"), "3rd"] = 4
    left.loc[("a", "w"), "3rd"] = 5
    left.loc[("a", "a"), "3rd"] = 6

    ts.loc[("b", "x")] = 2
    ts.loc["b", "a"] = -1
    ts.loc[("b", "b")] = 3
    ts.loc["a", "x"] = 4
    ts.loc[("a", "w")] = 5
    ts.loc["a", "a"] = 6

    right = pd.DataFrame(
        [
            ["a", "b", 0],
            ["b", "d", 1],
            ["b", "x", 2],
            ["b", "a", -1],
            ["b", "b", 3],
            ["a", "x", 4],
            ["a", "w", 5],
            ["a", "a", 6],
        ],
        columns=["1st", "2nd", "3rd"],
    )
    right.set_index(["1st", "2nd"], inplace=True)
    # FIXME data types changes to float because
    # of intermediate nan insertion;
    tm.assert_frame_equal(left, right, check_dtype=False)
    tm.assert_series_equal(ts, right["3rd"])


def test_insert2():
    # GH9250
    idx = (
        [("test1", i) for i in range(5)]
        + [("test2", i) for i in range(6)]
        + [("test", 17), ("test", 18)]
    )

    left = pd.Series(np.linspace(0, 10, 11), MultiIndex.from_tuples(idx[:-2]))

    left.loc[("test", 17)] = 11
    left.loc[("test", 18)] = 12

    right = pd.Series(np.linspace(0, 12, 13), MultiIndex.from_tuples(idx))

    tm.assert_series_equal(left, right)


def test_append(idx):
    result = idx[:3].append(idx[3:])
    assert result.equals(idx)

    foos = [idx[:1], idx[1:3], idx[3:]]
    result = foos[0].append(foos[1:])
    assert result.equals(idx)

    # empty
    result = idx.append([])
    assert result.equals(idx)


def test_append_index():
    idx1 = Index([1.1, 1.2, 1.3])
    idx2 = pd.date_range("2011-01-01", freq="D", periods=3, tz="Asia/Tokyo")
    idx3 = Index(["A", "B", "C"])

    midx_lv2 = MultiIndex.from_arrays([idx1, idx2])
    midx_lv3 = MultiIndex.from_arrays([idx1, idx2, idx3])

    result = idx1.append(midx_lv2)

    # see gh-7112
    tz = pytz.timezone("Asia/Tokyo")
    expected_tuples = [
        (1.1, tz.localize(datetime(2011, 1, 1))),
        (1.2, tz.localize(datetime(2011, 1, 2))),
        (1.3, tz.localize(datetime(2011, 1, 3))),
    ]
    expected = Index([1.1, 1.2, 1.3] + expected_tuples)
    tm.assert_index_equal(result, expected)

    result = midx_lv2.append(idx1)
    expected = Index(expected_tuples + [1.1, 1.2, 1.3])
    tm.assert_index_equal(result, expected)

    result = midx_lv2.append(midx_lv2)
    expected = MultiIndex.from_arrays([idx1.append(idx1), idx2.append(idx2)])
    tm.assert_index_equal(result, expected)

    result = midx_lv2.append(midx_lv3)
    tm.assert_index_equal(result, expected)

    result = midx_lv3.append(midx_lv2)
    expected = Index._simple_new(
        np.array(
            [
                (1.1, tz.localize(datetime(2011, 1, 1)), "A"),
                (1.2, tz.localize(datetime(2011, 1, 2)), "B"),
                (1.3, tz.localize(datetime(2011, 1, 3)), "C"),
            ]
            + expected_tuples,
            dtype=object,
        ),
        None,
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("name, exp", [("b", "b"), ("c", None)])
def test_append_names_match(name, exp):
    # GH#48288
    midx = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    midx2 = MultiIndex.from_arrays([[3], [5]], names=["a", name])
    result = midx.append(midx2)
    expected = MultiIndex.from_arrays([[1, 2, 3], [3, 4, 5]], names=["a", exp])
    tm.assert_index_equal(result, expected)


def test_append_names_dont_match():
    # GH#48288
    midx = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    midx2 = MultiIndex.from_arrays([[3], [5]], names=["x", "y"])
    result = midx.append(midx2)
    expected = MultiIndex.from_arrays([[1, 2, 3], [3, 4, 5]], names=None)
    tm.assert_index_equal(result, expected)


def test_append_overlapping_interval_levels():
    # GH 54934
    ivl1 = pd.IntervalIndex.from_breaks([0.0, 1.0, 2.0])
    ivl2 = pd.IntervalIndex.from_breaks([0.5, 1.5, 2.5])
    mi1 = MultiIndex.from_product([ivl1, ivl1])
    mi2 = MultiIndex.from_product([ivl2, ivl2])
    result = mi1.append(mi2)
    expected = MultiIndex.from_tuples(
        [
            (pd.Interval(0.0, 1.0), pd.Interval(0.0, 1.0)),
            (pd.Interval(0.0, 1.0), pd.Interval(1.0, 2.0)),
            (pd.Interval(1.0, 2.0), pd.Interval(0.0, 1.0)),
            (pd.Interval(1.0, 2.0), pd.Interval(1.0, 2.0)),
            (pd.Interval(0.5, 1.5), pd.Interval(0.5, 1.5)),
            (pd.Interval(0.5, 1.5), pd.Interval(1.5, 2.5)),
            (pd.Interval(1.5, 2.5), pd.Interval(0.5, 1.5)),
            (pd.Interval(1.5, 2.5), pd.Interval(1.5, 2.5)),
        ]
    )
    tm.assert_index_equal(result, expected)


def test_repeat():
    reps = 2
    numbers = [1, 2, 3]
    names = np.array(["foo", "bar"])

    m = MultiIndex.from_product([numbers, names], names=names)
    expected = MultiIndex.from_product([numbers, names.repeat(reps)], names=names)
    tm.assert_index_equal(m.repeat(reps), expected)


def test_insert_base(idx):
    result = idx[1:4]

    # test 0th element
    assert idx[0:4].equals(result.insert(0, idx[0]))


def test_delete_base(idx):
    expected = idx[1:]
    result = idx.delete(0)
    assert result.equals(expected)
    assert result.name == expected.name

    expected = idx[:-1]
    result = idx.delete(-1)
    assert result.equals(expected)
    assert result.name == expected.name

    msg = "index 6 is out of bounds for axis 0 with size 6"
    with pytest.raises(IndexError, match=msg):
        idx.delete(len(idx))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_llvmjit.py ===
from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes


if import_module('llvmlite'):
    import sympy.printing.llvmjitcode as g
else:
    disabled = True

import sympy
from sympy.abc import a, b, n


# copied from numpy.isclose documentation
def isclose(a, b):
    rtol = 1e-5
    atol = 1e-8
    return abs(a-b) <= atol + rtol*abs(b)


def test_simple_expr():
    e = a + 1.0
    f = g.llvm_callable([a], e)
    res = float(e.subs({a: 4.0}).evalf())
    jit_res = f(4.0)

    assert isclose(jit_res, res)


def test_two_arg():
    e = 4.0*a + b + 3.0
    f = g.llvm_callable([a, b], e)
    res = float(e.subs({a: 4.0, b: 3.0}).evalf())
    jit_res = f(4.0, 3.0)

    assert isclose(jit_res, res)


def test_func():
    e = 4.0*sympy.exp(-a)
    f = g.llvm_callable([a], e)
    res = float(e.subs({a: 1.5}).evalf())
    jit_res = f(1.5)

    assert isclose(jit_res, res)


def test_two_func():
    e = 4.0*sympy.exp(-a) + sympy.exp(b)
    f = g.llvm_callable([a, b], e)
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())
    jit_res = f(1.5, 2.0)

    assert isclose(jit_res, res)


def test_two_sqrt():
    e = 4.0*sympy.sqrt(a) + sympy.sqrt(b)
    f = g.llvm_callable([a, b], e)
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())
    jit_res = f(1.5, 2.0)

    assert isclose(jit_res, res)


def test_two_pow():
    e = a**1.5 + b**7
    f = g.llvm_callable([a, b], e)
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())
    jit_res = f(1.5, 2.0)

    assert isclose(jit_res, res)


def test_callback():
    e = a + 1.2
    f = g.llvm_callable([a], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(1)
    array_type = ctypes.c_double * 1
    inp = {a: 2.2}
    array = array_type(inp[a])
    jit_res = f(m, array)

    res = float(e.subs(inp).evalf())

    assert isclose(jit_res, res)


def test_callback_cubature():
    e = a + 1.2
    f = g.llvm_callable([a], e, callback_type='cubature')
    m = ctypes.c_int(1)
    array_type = ctypes.c_double * 1
    inp = {a: 2.2}
    array = array_type(inp[a])
    out_array = array_type(0.0)
    jit_ret = f(m, array, None, m, out_array)

    assert jit_ret == 0

    res = float(e.subs(inp).evalf())

    assert isclose(out_array[0], res)


def test_callback_two():
    e = 3*a*b
    f = g.llvm_callable([a, b], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(2)
    array_type = ctypes.c_double * 2
    inp = {a: 0.2, b: 1.7}
    array = array_type(inp[a], inp[b])
    jit_res = f(m, array)

    res = float(e.subs(inp).evalf())

    assert isclose(jit_res, res)


def test_callback_alt_two():
    d = sympy.IndexedBase('d')
    e = 3*d[0]*d[1]
    f = g.llvm_callable([n, d], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(2)
    array_type = ctypes.c_double * 2
    inp = {d[0]: 0.2, d[1]: 1.7}
    array = array_type(inp[d[0]], inp[d[1]])
    jit_res = f(m, array)

    res = float(e.subs(inp).evalf())

    assert isclose(jit_res, res)


def test_multiple_statements():
    # Match return from CSE
    e = [[(b, 4.0*a)], [b + 5]]
    f = g.llvm_callable([a], e)
    b_val = e[0][0][1].subs({a: 1.5})
    res = float(e[1][0].subs({b: b_val}).evalf())
    jit_res = f(1.5)
    assert isclose(jit_res, res)

    f_callback = g.llvm_callable([a], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(1)
    array_type = ctypes.c_double * 1
    array = array_type(1.5)
    jit_callback_res = f_callback(m, array)
    assert isclose(jit_callback_res, res)


def test_cse():
    e = a*a + b*b + sympy.exp(-a*a - b*b)
    e2 = sympy.cse(e)
    f = g.llvm_callable([a, b], e2)
    res = float(e.subs({a: 2.3, b: 0.1}).evalf())
    jit_res = f(2.3, 0.1)

    assert isclose(jit_res, res)


def eval_cse(e, sub_dict):
    tmp_dict = {}
    for tmp_name, tmp_expr in e[0]:
        e2 = tmp_expr.subs(sub_dict)
        e3 = e2.subs(tmp_dict)
        tmp_dict[tmp_name] = e3
    return [e.subs(sub_dict).subs(tmp_dict) for e in e[1]]


def test_cse_multiple():
    e1 = a*a
    e2 = a*a + b*b
    e3 = sympy.cse([e1, e2])

    raises(NotImplementedError,
           lambda: g.llvm_callable([a, b], e3, callback_type='scipy.integrate'))

    f = g.llvm_callable([a, b], e3)
    jit_res = f(0.1, 1.5)
    assert len(jit_res) == 2
    res = eval_cse(e3, {a: 0.1, b: 1.5})
    assert isclose(res[0], jit_res[0])
    assert isclose(res[1], jit_res[1])


def test_callback_cubature_multiple():
    e1 = a*a
    e2 = a*a + b*b
    e3 = sympy.cse([e1, e2, 4*e2])
    f = g.llvm_callable([a, b], e3, callback_type='cubature')

    # Number of input variables
    ndim = 2
    # Number of output expression values
    outdim = 3

    m = ctypes.c_int(ndim)
    fdim = ctypes.c_int(outdim)
    array_type = ctypes.c_double * ndim
    out_array_type = ctypes.c_double * outdim
    inp = {a: 0.2, b: 1.5}
    array = array_type(inp[a], inp[b])
    out_array = out_array_type()
    jit_ret = f(m, array, None, fdim, out_array)

    assert jit_ret == 0

    res = eval_cse(e3, inp)

    assert isclose(out_array[0], res[0])
    assert isclose(out_array[1], res[1])
    assert isclose(out_array[2], res[2])


def test_symbol_not_found():
    e = a*a + b
    raises(LookupError, lambda: g.llvm_callable([a], e))


def test_bad_callback():
    e = a
    raises(ValueError, lambda: g.llvm_callable([a], e, callback_type='bad_callback'))

# === atelier-kyo-manager/run.py ===
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\charset_normalizer\__main__.py ===
from __future__ import annotations

from .cli import cli_detect

if __name__ == "__main__":
    cli_detect()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dotenv\__main__.py ===
"""Entry point for cli, enables execution with `python -m dotenv`"""

from .cli import cli

if __name__ == "__main__":
    cli()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\_identifier.py ===
import re

# generated by scripts/generate_identifier_pattern.py
pattern = re.compile(
    r"[\w·̀-ͯ·҃-֑҇-ׇֽֿׁׂׅׄؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ܑۭܰ-݊ަ-ް߫-߽߳ࠖ-࠙ࠛ-ࠣࠥ-ࠧࠩ-࡙࠭-࡛࣓-ࣣ࣡-ःऺ-़ा-ॏ॑-ॗॢॣঁ-ঃ়া-ৄেৈো-্ৗৢৣ৾ਁ-ਃ਼ਾ-ੂੇੈੋ-੍ੑੰੱੵઁ-ઃ઼ા-ૅે-ૉો-્ૢૣૺ-૿ଁ-ଃ଼ା-ୄେୈୋ-୍ୖୗୢୣஂா-ூெ-ைொ-்ௗఀ-ఄా-ౄె-ైొ-్ౕౖౢౣಁ-ಃ಼ಾ-ೄೆ-ೈೊ-್ೕೖೢೣഀ-ഃ഻഼ാ-ൄെ-ൈൊ-്ൗൢൣංඃ්ා-ුූෘ-ෟෲෳัิ-ฺ็-๎ັິ-ູົຼ່-ໍ༹༘༙༵༷༾༿ཱ-྄྆྇ྍ-ྗྙ-ྼ࿆ါ-ှၖ-ၙၞ-ၠၢ-ၤၧ-ၭၱ-ၴႂ-ႍႏႚ-ႝ፝-፟ᜒ-᜔ᜲ-᜴ᝒᝓᝲᝳ឴-៓៝᠋-᠍ᢅᢆᢩᤠ-ᤫᤰ-᤻ᨗ-ᨛᩕ-ᩞ᩠-᩿᩼᪰-᪽ᬀ-ᬄ᬴-᭄᭫-᭳ᮀ-ᮂᮡ-ᮭ᯦-᯳ᰤ-᰷᳐-᳔᳒-᳨᳭ᳲ-᳴᳷-᳹᷀-᷹᷻-᷿‿⁀⁔⃐-⃥⃜⃡-⃰℘℮⳯-⵿⳱ⷠ-〪ⷿ-゙゚〯꙯ꙴ-꙽ꚞꚟ꛰꛱ꠂ꠆ꠋꠣ-ꠧꢀꢁꢴ-ꣅ꣠-꣱ꣿꤦ-꤭ꥇ-꥓ꦀ-ꦃ꦳-꧀ꧥꨩ-ꨶꩃꩌꩍꩻ-ꩽꪰꪲ-ꪴꪷꪸꪾ꪿꫁ꫫ-ꫯꫵ꫶ꯣ-ꯪ꯬꯭ﬞ︀-️︠-︯︳︴﹍-﹏＿𐇽𐋠𐍶-𐍺𐨁-𐨃𐨅𐨆𐨌-𐨏𐨸-𐨿𐨺𐫦𐫥𐴤-𐽆𐴧-𐽐𑀀-𑀂𑀸-𑁆𑁿-𑂂𑂰-𑂺𑄀-𑄂𑄧-𑄴𑅅𑅆𑅳𑆀-𑆂𑆳-𑇀𑇉-𑇌𑈬-𑈷𑈾𑋟-𑋪𑌀-𑌃𑌻𑌼𑌾-𑍄𑍇𑍈𑍋-𑍍𑍗𑍢𑍣𑍦-𑍬𑍰-𑍴𑐵-𑑆𑑞𑒰-𑓃𑖯-𑖵𑖸-𑗀𑗜𑗝𑘰-𑙀𑚫-𑚷𑜝-𑜫𑠬-𑠺𑨁-𑨊𑨳-𑨹𑨻-𑨾𑩇𑩑-𑩛𑪊-𑪙𑰯-𑰶𑰸-𑰿𑲒-𑲧𑲩-𑲶𑴱-𑴶𑴺𑴼𑴽𑴿-𑵅𑵇𑶊-𑶎𑶐𑶑𑶓-𑶗𑻳-𑻶𖫰-𖫴𖬰-𖬶𖽑-𖽾𖾏-𖾒𛲝𛲞𝅥-𝅩𝅭-𝅲𝅻-𝆂𝆅-𝆋𝆪-𝆭𝉂-𝉄𝨀-𝨶𝨻-𝩬𝩵𝪄𝪛-𝪟𝪡-𝪯𞀀-𞀆𞀈-𞀘𞀛-𞀡𞀣𞀤𞀦-𞣐𞀪-𞣖𞥄-𞥊󠄀-󠇯]+"  # noqa: B950
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\calculus\calculus.py ===
class CalculusMethods(object):
    pass

def defun(f):
    setattr(CalculusMethods, f.__name__, f)
    return f

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\calculus\__init__.py ===
from . import calculus
# XXX: hack to set methods
from . import approximation
from . import differentiation
from . import extrapolation
from . import polynomials

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\backend\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from .backend import is_compatible, prepare, run, supports_device  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\__init__.py ===
from os.path import dirname, basename, isfile, join, splitext
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [splitext(basename(f))[0] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from . import *

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\status_codes.py ===
SUCCESS = 0
ERROR = 1
UNKNOWN_ERROR = 2
VIRTUALENV_NOT_FOUND = 3
PREVIOUS_BUILD_DIR_ERROR = 4
NO_MATCHES_FOUND = 23

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\metadata\importlib\__init__.py ===
from ._dists import Distribution
from ._envs import Environment

__all__ = ["NAME", "Distribution", "Environment"]

NAME = "importlib"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\cyextension\__init__.py ===
# cyextension/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\mypy\__init__.py ===
# ext/mypy/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\plugin\__init__.py ===
# testing/plugin/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\systems\__init__.py ===
from sympy.physics.units.systems.mks import MKS
from sympy.physics.units.systems.mksa import MKSA
from sympy.physics.units.systems.natural import natural
from sympy.physics.units.systems.si import SI

__all__ = ['MKS', 'MKSA', 'natural', 'SI']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\conv_array_to_matrix.py ===
from sympy.tensor.array.expressions import from_array_to_matrix
from sympy.tensor.array.expressions.conv_array_to_indexed import _conv_to_from_decorator

convert_array_to_matrix = _conv_to_from_decorator(from_array_to_matrix.convert_array_to_matrix)
_array2matrix = _conv_to_from_decorator(from_array_to_matrix._array2matrix)
_remove_trivial_dims = _conv_to_from_decorator(from_array_to_matrix._remove_trivial_dims)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\__init__.py ===
# IANA versions like 2020a are not valid PEP 440 identifiers; the recommended
# way to translate the version is to use YYYY.n where `n` is a 0-based index.
__version__ = "2025.2"

# This exposes the original IANA version number.
IANA_VERSION = "2025b"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_numpy_array_equal.py ===
import copy

import numpy as np
import pytest

import pandas as pd
from pandas import Timestamp
import pandas._testing as tm


def test_assert_numpy_array_equal_shape_mismatch():
    msg = """numpy array are different

numpy array shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([3, 4, 5]))


def test_assert_numpy_array_equal_bad_type():
    expected = "Expected type"

    with pytest.raises(AssertionError, match=expected):
        tm.assert_numpy_array_equal(1, 2)


@pytest.mark.parametrize(
    "a,b,klass1,klass2",
    [(np.array([1]), 1, "ndarray", "int"), (1, np.array([1]), "int", "ndarray")],
)
def test_assert_numpy_array_equal_class_mismatch(a, b, klass1, klass2):
    msg = f"""numpy array are different

numpy array classes are different
\\[left\\]:  {klass1}
\\[right\\]: {klass2}"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)


def test_assert_numpy_array_equal_value_mismatch1():
    msg = """numpy array are different

numpy array values are different \\(66\\.66667 %\\)
\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]
\\[right\\]: \\[1\\.0, nan, 3\\.0\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))


def test_assert_numpy_array_equal_value_mismatch2():
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([1, 3]))


def test_assert_numpy_array_equal_value_mismatch3():
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )


def test_assert_numpy_array_equal_value_mismatch4():
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1\\.1, 2\\.000001\\]
\\[right\\]: \\[1\\.1, 2.0\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1.1, 2.000001]), np.array([1.1, 2.0]))


def test_assert_numpy_array_equal_value_mismatch5():
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )


def test_assert_numpy_array_equal_value_mismatch6():
    msg = """numpy array are different

numpy array values are different \\(25\\.0 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array([[1, 2], [3, 4]]), np.array([[1, 3], [3, 4]])
        )


def test_assert_numpy_array_equal_shape_mismatch_override():
    msg = """Index are different

Index shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1, 2]), np.array([3, 4, 5]), obj="Index")


def test_numpy_array_equal_unicode():
    # see gh-20503
    #
    # Test ensures that `tm.assert_numpy_array_equals` raises the right
    # exception when comparing np.arrays containing differing unicode objects.
    msg = """numpy array are different

numpy array values are different \\(33\\.33333 %\\)
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[á, à, å\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(
            np.array(["á", "à", "ä"]), np.array(["á", "à", "å"])
        )


def test_numpy_array_equal_object():
    a = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-01")])
    b = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-02")])

    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]
\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)


@pytest.mark.parametrize("other_type", ["same", "copy"])
@pytest.mark.parametrize("check_same", ["same", "copy"])
def test_numpy_array_equal_copy_flag(other_type, check_same):
    a = np.array([1, 2, 3])
    msg = None

    if other_type == "same":
        other = a.view()
    else:
        other = a.copy()

    if check_same != other_type:
        msg = (
            r"array\(\[1, 2, 3\]\) is not array\(\[1, 2, 3\]\)"
            if check_same == "same"
            else r"array\(\[1, 2, 3\]\) is array\(\[1, 2, 3\]\)"
        )

    if msg is not None:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_numpy_array_equal(a, other, check_same=check_same)
    else:
        tm.assert_numpy_array_equal(a, other, check_same=check_same)


def test_numpy_array_equal_contains_na():
    # https://github.com/pandas-dev/pandas/issues/31881
    a = np.array([True, False])
    b = np.array([True, pd.NA], dtype=object)

    msg = """numpy array are different

numpy array values are different \\(50.0 %\\)
\\[left\\]:  \\[True, False\\]
\\[right\\]: \\[True, <NA>\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)


def test_numpy_array_equal_identical_na(nulls_fixture):
    a = np.array([nulls_fixture], dtype=object)

    tm.assert_numpy_array_equal(a, a)

    # matching but not the identical object
    if hasattr(nulls_fixture, "copy"):
        other = nulls_fixture.copy()
    else:
        other = copy.copy(nulls_fixture)
    b = np.array([other], dtype=object)
    tm.assert_numpy_array_equal(a, b)


def test_numpy_array_equal_different_na():
    a = np.array([np.nan], dtype=object)
    b = np.array([pd.NA], dtype=object)

    msg = """numpy array are different

numpy array values are different \\(100.0 %\\)
\\[left\\]:  \\[nan\\]
\\[right\\]: \\[<NA>\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\tests\test_clebsch_gordan.py ===
from sympy.core.numbers import (I, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import Matrix
from sympy.physics.wigner import (clebsch_gordan, wigner_9j, wigner_6j, gaunt,
        real_gaunt, racah, dot_rot_grad_Ynm, wigner_3j, wigner_d_small, wigner_d)
from sympy.testing.pytest import raises, skip

# for test cases, refer : https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients

def test_clebsch_gordan_docs():
    assert clebsch_gordan(Rational(3, 2), S.Half, 2, Rational(3, 2), S.Half, 2) == 1
    assert clebsch_gordan(Rational(3, 2), S.Half, 1, Rational(3, 2), Rational(-1, 2), 1) == sqrt(3)/2
    assert clebsch_gordan(Rational(3, 2), S.Half, 1, Rational(-1, 2), S.Half, 0) == -sqrt(2)/2


def test_clebsch_gordan():
    # Argument order: (j_1, j_2, j, m_1, m_2, m)

    h = S.One
    k = S.Half
    l = Rational(3, 2)
    i = Rational(-1, 2)
    n = Rational(7, 2)
    p = Rational(5, 2)
    assert clebsch_gordan(k, k, 1, k, k, 1) == 1
    assert clebsch_gordan(k, k, 1, k, k, 0) == 0
    assert clebsch_gordan(k, k, 1, i, i, -1) == 1
    assert clebsch_gordan(k, k, 1, k, i, 0) == sqrt(2)/2
    assert clebsch_gordan(k, k, 0, k, i, 0) == sqrt(2)/2
    assert clebsch_gordan(k, k, 1, i, k, 0) == sqrt(2)/2
    assert clebsch_gordan(k, k, 0, i, k, 0) == -sqrt(2)/2
    assert clebsch_gordan(h, k, l, 1, k, l) == 1
    assert clebsch_gordan(h, k, l, 1, i, k) == 1/sqrt(3)
    assert clebsch_gordan(h, k, k, 1, i, k) == sqrt(2)/sqrt(3)
    assert clebsch_gordan(h, k, k, 0, k, k) == -1/sqrt(3)
    assert clebsch_gordan(h, k, l, 0, k, k) == sqrt(2)/sqrt(3)
    assert clebsch_gordan(h, h, S(2), 1, 1, S(2)) == 1
    assert clebsch_gordan(h, h, S(2), 1, 0, 1) == 1/sqrt(2)
    assert clebsch_gordan(h, h, S(2), 0, 1, 1) == 1/sqrt(2)
    assert clebsch_gordan(h, h, 1, 1, 0, 1) == 1/sqrt(2)
    assert clebsch_gordan(h, h, 1, 0, 1, 1) == -1/sqrt(2)
    assert clebsch_gordan(l, l, S(3), l, l, S(3)) == 1
    assert clebsch_gordan(l, l, S(2), l, k, S(2)) == 1/sqrt(2)
    assert clebsch_gordan(l, l, S(3), l, k, S(2)) == 1/sqrt(2)
    assert clebsch_gordan(S(2), S(2), S(4), S(2), S(2), S(4)) == 1
    assert clebsch_gordan(S(2), S(2), S(3), S(2), 1, S(3)) == 1/sqrt(2)
    assert clebsch_gordan(S(2), S(2), S(3), 1, 1, S(2)) == 0
    assert clebsch_gordan(p, h, n, p, 1, n) == 1
    assert clebsch_gordan(p, h, p, p, 0, p) == sqrt(5)/sqrt(7)
    assert clebsch_gordan(p, h, l, k, 1, l) == 1/sqrt(15)


def test_clebsch_gordan_numpy():
    try:
        import numpy as np
    except ImportError:
        skip("numpy not installed")
    assert clebsch_gordan(*np.zeros(6).astype(np.int64)) == 1
    assert wigner_3j(2, np.float64(6.0), 4.0, 0, 0, 0) == sqrt(715)/143
    assert wigner_3j(0, 0.5, 0.5, 0, 0.5, -0.5) == sqrt(2)/2
    raises(ValueError, lambda: wigner_3j(2.1, 6, 4, 0, 0, 0))


def test_wigner():
    try:
        import numpy as np
    except ImportError:
        skip("numpy not installed")
    def tn(a, b):
        return (a - b).n(64) < S('1e-64')
    assert tn(wigner_9j(1, 1, 1, 1, 1, 1, 1, 1, 0, prec=64), Rational(1, 18))
    assert wigner_9j(3, 3, 2, 3, 3, 2, 3, 3, 2) == 3221*sqrt(
        70)/(246960*sqrt(105)) - 365/(3528*sqrt(70)*sqrt(105))
    assert wigner_6j(5, 5, 5, 5, 5, 5) == Rational(1, 52)
    assert tn(wigner_6j(8, 8, 8, 8, 8, 8, prec=64), Rational(-12219, 965770))
    assert wigner_6j(1, 1, 1, 1.0, np.float64(1.0), 1) == Rational(1, 6)
    assert wigner_6j(3.0, np.float32(3), 3.0, 3, 3, 3) == Rational(-1, 14)
    # regression test for #8747
    half = S.Half
    assert wigner_9j(0, 0, 0, 0, half, half, 0, half, half) == half
    assert (wigner_9j(3, 5, 4,
                      7 * half, 5 * half, 4,
                      9 * half, 9 * half, 0)
            == -sqrt(Rational(361, 205821000)))
    assert (wigner_9j(1, 4, 3,
                      5 * half, 4, 5 * half,
                      5 * half, 2, 7 * half)
            == -sqrt(Rational(3971, 373403520)))
    assert (wigner_9j(4, 9 * half, 5 * half,
                      2, 4, 4,
                      5, 7 * half, 7 * half)
            == -sqrt(Rational(3481, 5042614500)))
    assert (wigner_9j(5, 5, 5.0,
                      np.float64(5.0), 5, 5,
                      5, 5, 5)
            == 0)
    assert (wigner_9j(1.0, 2.0, 3.0,
                      3, 2, 1,
                      2, 1, 3)
            == -4*sqrt(70)/11025)


def test_gaunt():
    def tn(a, b):
        return (a - b).n(64) < S('1e-64')
    assert gaunt(1, 0, 1, 1, 0, -1) == -1/(2*sqrt(pi))
    assert isinstance(gaunt(1, 1, 0, -1, 1, 0).args[0], Rational)
    assert isinstance(gaunt(0, 1, 1, 0, -1, 1).args[0], Rational)

    assert tn(gaunt(
        10, 10, 12, 9, 3, -12, prec=64), (Rational(-98, 62031)) * sqrt(6279)/sqrt(pi))
    def gaunt_ref(l1, l2, l3, m1, m2, m3):
        return (
            sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4 * pi)) *
            wigner_3j(l1, l2, l3, 0, 0, 0) *
            wigner_3j(l1, l2, l3, m1, m2, m3)
        )
    threshold = 1e-10
    l_max = 3
    l3_max = 24
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for l3 in range(l3_max + 1):
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        for m3 in range(-l3, l3 + 1):
                            args = l1, l2, l3, m1, m2, m3
                            g  = gaunt(*args)
                            g0 = gaunt_ref(*args)
                            assert abs(g - g0) < threshold
                            if m1 + m2 + m3 != 0:
                                assert abs(g) < threshold
                            if (l1 + l2 + l3) % 2:
                                assert abs(g) < threshold
    assert gaunt(1, 1, 0, 0, 2, -2) is S.Zero


def test_realgaunt():
    # All non-zero values corresponding to l values from 0 to 2
    for l in range(3):
        for m in range(-l, l+1):
            assert real_gaunt(0, l, l, 0, m, m) == 1/(2*sqrt(pi))
    assert real_gaunt(1, 1, 2, 0, 0, 0) == sqrt(5)/(5*sqrt(pi))
    assert real_gaunt(1, 1, 2, 1, 1, 0) == -sqrt(5)/(10*sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 0, 0) == sqrt(5)/(7*sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 2, 2) == -sqrt(5)/(7*sqrt(pi))
    assert real_gaunt(2, 2, 2, -2, -2, 0) == -sqrt(5)/(7*sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, 0, -1) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, 0, 1, 1) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, 1, 1, 2) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, 1, -2) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, -1, 2) == -sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 1, 1) == sqrt(5)/(14*sqrt(pi))
    assert real_gaunt(2, 2, 2, 1, 1, 2) == sqrt(15)/(14*sqrt(pi))
    assert real_gaunt(2, 2, 2, -1, -1, 2) == -sqrt(15)/(14*sqrt(pi))

    assert real_gaunt(-2, -2, -2, -2, -2, 0) is S.Zero  # m test
    assert real_gaunt(-2, 1, 0, 1, 1, 1) is S.Zero  # l test
    assert real_gaunt(-2, -1, -2, -1, -1, 0) is S.Zero  # m and l test
    assert real_gaunt(-2, -2, -2, -2, -2, -2) is S.Zero  # m and k test
    assert real_gaunt(-2, -1, -2, -1, -1, -1) is S.Zero  # m, l and k test

    x = symbols('x', integer=True)
    v = [0]*6
    for i in range(len(v)):
        v[i] = x  # non literal ints fail
        raises(ValueError, lambda: real_gaunt(*v))
        v[i] = 0


def test_racah():
    assert racah(3,3,3,3,3,3) == Rational(-1,14)
    assert racah(2,2,2,2,2,2) == Rational(-3,70)
    assert racah(7,8,7,1,7,7, prec=4).is_Float
    assert racah(5.5,7.5,9.5,6.5,8,9) == -719*sqrt(598)/1158924
    assert abs(racah(5.5,7.5,9.5,6.5,8,9, prec=4) - (-0.01517)) < S('1e-4')


def test_dot_rota_grad_SH():
    theta, phi = symbols("theta phi")
    assert dot_rot_grad_Ynm(1, 1, 1, 1, 1, 0) !=  \
        sqrt(30)*Ynm(2, 2, 1, 0)/(10*sqrt(pi))
    assert dot_rot_grad_Ynm(1, 1, 1, 1, 1, 0).doit() ==  \
        sqrt(30)*Ynm(2, 2, 1, 0)/(10*sqrt(pi))
    assert dot_rot_grad_Ynm(1, 5, 1, 1, 1, 2) !=  \
        0
    assert dot_rot_grad_Ynm(1, 5, 1, 1, 1, 2).doit() ==  \
        0
    assert dot_rot_grad_Ynm(3, 3, 3, 3, theta, phi).doit() ==  \
        15*sqrt(3003)*Ynm(6, 6, theta, phi)/(143*sqrt(pi))
    assert dot_rot_grad_Ynm(3, 3, 1, 1, theta, phi).doit() ==  \
        sqrt(3)*Ynm(4, 4, theta, phi)/sqrt(pi)
    assert dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit() ==  \
        3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))
    assert dot_rot_grad_Ynm(3, 2, 3, 2, theta, phi).doit().expand() ==  \
        -sqrt(70)*Ynm(4, 4, theta, phi)/(11*sqrt(pi)) + \
        45*sqrt(182)*Ynm(6, 4, theta, phi)/(143*sqrt(pi))


def test_wigner_d():
    half = S(1)/2
    assert wigner_d_small(half, 0) == Matrix([[1, 0], [0, 1]])
    assert wigner_d_small(half, pi/2) == Matrix([[1, 1], [-1, 1]])/sqrt(2)
    assert wigner_d_small(half, pi) == Matrix([[0, 1], [-1, 0]])

    alpha, beta, gamma = symbols("alpha, beta, gamma", real=True)
    D = wigner_d(half, alpha, beta, gamma)
    assert D[0, 0] == exp(I*alpha/2)*exp(I*gamma/2)*cos(beta/2)
    assert D[0, 1] == exp(I*alpha/2)*exp(-I*gamma/2)*sin(beta/2)
    assert D[1, 0] == -exp(-I*alpha/2)*exp(I*gamma/2)*sin(beta/2)
    assert D[1, 1] == exp(-I*alpha/2)*exp(-I*gamma/2)*cos(beta/2)

    # Test Y_{n mi}(g*x)=\sum_{mj}D^n_{mi mj}*Y_{n mj}(x)
    theta, phi = symbols("theta phi", real=True)
    v = Matrix([Ynm(1, mj, theta, phi) for mj in range(1, -2, -1)])
    w = wigner_d(1, -pi/2, pi/2, -pi/2)@v.subs({theta: pi/4, phi: pi})
    w_ = v.subs({theta: pi/2, phi: pi/4})
    assert w.expand(func=True).as_real_imag() == w_.expand(func=True).as_real_imag()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_array_interface.py ===
import sys
import sysconfig

import pytest

import numpy as np
from numpy.testing import IS_EDITABLE, IS_WASM, extbuild


@pytest.fixture
def get_module(tmp_path):
    """ Some codes to generate data and manage temporary buffers use when
    sharing with numpy via the array interface protocol.
    """
    if sys.platform.startswith('cygwin'):
        pytest.skip('link fails on cygwin')
    if IS_WASM:
        pytest.skip("Can't build module inside Wasm")
    if IS_EDITABLE:
        pytest.skip("Can't build module for editable install")

    prologue = '''
        #include <Python.h>
        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <numpy/arrayobject.h>
        #include <stdio.h>
        #include <math.h>

        NPY_NO_EXPORT
        void delete_array_struct(PyObject *cap) {

            /* get the array interface structure */
            PyArrayInterface *inter = (PyArrayInterface*)
                PyCapsule_GetPointer(cap, NULL);

            /* get the buffer by which data was shared */
            double *ptr = (double*)PyCapsule_GetContext(cap);

            /* for the purposes of the regression test set the elements
               to nan */
            for (npy_intp i = 0; i < inter->shape[0]; ++i)
                ptr[i] = nan("");

            /* free the shared buffer */
            free(ptr);

            /* free the array interface structure */
            free(inter->shape);
            free(inter);

            fprintf(stderr, "delete_array_struct\\ncap = %ld inter = %ld"
                " ptr = %ld\\n", (long)cap, (long)inter, (long)ptr);
        }
        '''

    functions = [
        ("new_array_struct", "METH_VARARGS", """

            long long n_elem = 0;
            double value = 0.0;

            if (!PyArg_ParseTuple(args, "Ld", &n_elem, &value)) {
                Py_RETURN_NONE;
            }

            /* allocate and initialize the data to share with numpy */
            long long n_bytes = n_elem*sizeof(double);
            double *data = (double*)malloc(n_bytes);

            if (!data) {
                PyErr_Format(PyExc_MemoryError,
                    "Failed to malloc %lld bytes", n_bytes);

                Py_RETURN_NONE;
            }

            for (long long i = 0; i < n_elem; ++i) {
                data[i] = value;
            }

            /* calculate the shape and stride */
            int nd = 1;

            npy_intp *ss = (npy_intp*)malloc(2*nd*sizeof(npy_intp));
            npy_intp *shape = ss;
            npy_intp *stride = ss + nd;

            shape[0] = n_elem;
            stride[0] = sizeof(double);

            /* construct the array interface */
            PyArrayInterface *inter = (PyArrayInterface*)
                malloc(sizeof(PyArrayInterface));

            memset(inter, 0, sizeof(PyArrayInterface));

            inter->two = 2;
            inter->nd = nd;
            inter->typekind = 'f';
            inter->itemsize = sizeof(double);
            inter->shape = shape;
            inter->strides = stride;
            inter->data = data;
            inter->flags = NPY_ARRAY_WRITEABLE | NPY_ARRAY_NOTSWAPPED |
                           NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;

            /* package into a capsule */
            PyObject *cap = PyCapsule_New(inter, NULL, delete_array_struct);

            /* save the pointer to the data */
            PyCapsule_SetContext(cap, data);

            fprintf(stderr, "new_array_struct\\ncap = %ld inter = %ld"
                " ptr = %ld\\n", (long)cap, (long)inter, (long)data);

            return cap;
        """)
        ]

    more_init = "import_array();"

    try:
        import array_interface_testing
        return array_interface_testing
    except ImportError:
        pass

    # if it does not exist, build and load it
    if sysconfig.get_platform() == "win-arm64":
        pytest.skip("Meson unable to find MSVC linker on win-arm64")
    return extbuild.build_and_import_extension('array_interface_testing',
                                               functions,
                                               prologue=prologue,
                                               include_dirs=[np.get_include()],
                                               build_dir=tmp_path,
                                               more_init=more_init)


@pytest.mark.slow
def test_cstruct(get_module):

    class data_source:
        """
        This class is for testing the timing of the PyCapsule destructor
        invoked when numpy release its reference to the shared data as part of
        the numpy array interface protocol. If the PyCapsule destructor is
        called early the shared data is freed and invalid memory accesses will
        occur.
        """

        def __init__(self, size, value):
            self.size = size
            self.value = value

        @property
        def __array_struct__(self):
            return get_module.new_array_struct(self.size, self.value)

    # write to the same stream as the C code
    stderr = sys.__stderr__

    # used to validate the shared data.
    expected_value = -3.1415
    multiplier = -10000.0

    # create some data to share with numpy via the array interface
    # assign the data an expected value.
    stderr.write(' ---- create an object to share data ---- \n')
    buf = data_source(256, expected_value)
    stderr.write(' ---- OK!\n\n')

    # share the data
    stderr.write(' ---- share data via the array interface protocol ---- \n')
    arr = np.array(buf, copy=False)
    stderr.write(f'arr.__array_interface___ = {str(arr.__array_interface__)}\n')
    stderr.write(f'arr.base = {str(arr.base)}\n')
    stderr.write(' ---- OK!\n\n')

    # release the source of the shared data. this will not release the data
    # that was shared with numpy, that is done in the PyCapsule destructor.
    stderr.write(' ---- destroy the object that shared data ---- \n')
    buf = None
    stderr.write(' ---- OK!\n\n')

    # check that we got the expected data. If the PyCapsule destructor we
    # defined was prematurely called then this test will fail because our
    # destructor sets the elements of the array to NaN before free'ing the
    # buffer. Reading the values here may also cause a SEGV
    assert np.allclose(arr, expected_value)

    # read the data. If the PyCapsule destructor we defined was prematurely
    # called then reading the values here may cause a SEGV and will be reported
    # as invalid reads by valgrind
    stderr.write(' ---- read shared data ---- \n')
    stderr.write(f'arr = {str(arr)}\n')
    stderr.write(' ---- OK!\n\n')

    # write to the shared buffer. If the shared data was prematurely deleted
    # this will may cause a SEGV and valgrind will report invalid writes
    stderr.write(' ---- modify shared data ---- \n')
    arr *= multiplier
    expected_value *= multiplier
    stderr.write(f'arr.__array_interface___ = {str(arr.__array_interface__)}\n')
    stderr.write(f'arr.base = {str(arr.base)}\n')
    stderr.write(' ---- OK!\n\n')

    # read the data. If the shared data was prematurely deleted this
    # will may cause a SEGV and valgrind will report invalid reads
    stderr.write(' ---- read modified shared data ---- \n')
    stderr.write(f'arr = {str(arr)}\n')
    stderr.write(' ---- OK!\n\n')

    # check that we got the expected data. If the PyCapsule destructor we
    # defined was prematurely called then this test will fail because our
    # destructor sets the elements of the array to NaN before free'ing the
    # buffer. Reading the values here may also cause a SEGV
    assert np.allclose(arr, expected_value)

    # free the shared data, the PyCapsule destructor should run here
    stderr.write(' ---- free shared data ---- \n')
    arr = None
    stderr.write(' ---- OK!\n\n')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_np_datetime.py ===
import numpy as np
import pytest

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
    astype_overflowsafe,
    is_unitless,
    py_get_unit_from_dtype,
    py_td64_to_tdstruct,
)

import pandas._testing as tm


def test_is_unitless():
    dtype = np.dtype("M8[ns]")
    assert not is_unitless(dtype)

    dtype = np.dtype("datetime64")
    assert is_unitless(dtype)

    dtype = np.dtype("m8[ns]")
    assert not is_unitless(dtype)

    dtype = np.dtype("timedelta64")
    assert is_unitless(dtype)

    msg = "dtype must be datetime64 or timedelta64"
    with pytest.raises(ValueError, match=msg):
        is_unitless(np.dtype(np.int64))

    msg = "Argument 'dtype' has incorrect type"
    with pytest.raises(TypeError, match=msg):
        is_unitless("foo")


def test_get_unit_from_dtype():
    # datetime64
    assert py_get_unit_from_dtype(np.dtype("M8[Y]")) == NpyDatetimeUnit.NPY_FR_Y.value
    assert py_get_unit_from_dtype(np.dtype("M8[M]")) == NpyDatetimeUnit.NPY_FR_M.value
    assert py_get_unit_from_dtype(np.dtype("M8[W]")) == NpyDatetimeUnit.NPY_FR_W.value
    # B has been deprecated and removed -> no 3
    assert py_get_unit_from_dtype(np.dtype("M8[D]")) == NpyDatetimeUnit.NPY_FR_D.value
    assert py_get_unit_from_dtype(np.dtype("M8[h]")) == NpyDatetimeUnit.NPY_FR_h.value
    assert py_get_unit_from_dtype(np.dtype("M8[m]")) == NpyDatetimeUnit.NPY_FR_m.value
    assert py_get_unit_from_dtype(np.dtype("M8[s]")) == NpyDatetimeUnit.NPY_FR_s.value
    assert py_get_unit_from_dtype(np.dtype("M8[ms]")) == NpyDatetimeUnit.NPY_FR_ms.value
    assert py_get_unit_from_dtype(np.dtype("M8[us]")) == NpyDatetimeUnit.NPY_FR_us.value
    assert py_get_unit_from_dtype(np.dtype("M8[ns]")) == NpyDatetimeUnit.NPY_FR_ns.value
    assert py_get_unit_from_dtype(np.dtype("M8[ps]")) == NpyDatetimeUnit.NPY_FR_ps.value
    assert py_get_unit_from_dtype(np.dtype("M8[fs]")) == NpyDatetimeUnit.NPY_FR_fs.value
    assert py_get_unit_from_dtype(np.dtype("M8[as]")) == NpyDatetimeUnit.NPY_FR_as.value

    # timedelta64
    assert py_get_unit_from_dtype(np.dtype("m8[Y]")) == NpyDatetimeUnit.NPY_FR_Y.value
    assert py_get_unit_from_dtype(np.dtype("m8[M]")) == NpyDatetimeUnit.NPY_FR_M.value
    assert py_get_unit_from_dtype(np.dtype("m8[W]")) == NpyDatetimeUnit.NPY_FR_W.value
    # B has been deprecated and removed -> no 3
    assert py_get_unit_from_dtype(np.dtype("m8[D]")) == NpyDatetimeUnit.NPY_FR_D.value
    assert py_get_unit_from_dtype(np.dtype("m8[h]")) == NpyDatetimeUnit.NPY_FR_h.value
    assert py_get_unit_from_dtype(np.dtype("m8[m]")) == NpyDatetimeUnit.NPY_FR_m.value
    assert py_get_unit_from_dtype(np.dtype("m8[s]")) == NpyDatetimeUnit.NPY_FR_s.value
    assert py_get_unit_from_dtype(np.dtype("m8[ms]")) == NpyDatetimeUnit.NPY_FR_ms.value
    assert py_get_unit_from_dtype(np.dtype("m8[us]")) == NpyDatetimeUnit.NPY_FR_us.value
    assert py_get_unit_from_dtype(np.dtype("m8[ns]")) == NpyDatetimeUnit.NPY_FR_ns.value
    assert py_get_unit_from_dtype(np.dtype("m8[ps]")) == NpyDatetimeUnit.NPY_FR_ps.value
    assert py_get_unit_from_dtype(np.dtype("m8[fs]")) == NpyDatetimeUnit.NPY_FR_fs.value
    assert py_get_unit_from_dtype(np.dtype("m8[as]")) == NpyDatetimeUnit.NPY_FR_as.value


def test_td64_to_tdstruct():
    val = 12454636234  # arbitrary value

    res1 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_ns.value)
    exp1 = {
        "days": 0,
        "hrs": 0,
        "min": 0,
        "sec": 12,
        "ms": 454,
        "us": 636,
        "ns": 234,
        "seconds": 12,
        "microseconds": 454636,
        "nanoseconds": 234,
    }
    assert res1 == exp1

    res2 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_us.value)
    exp2 = {
        "days": 0,
        "hrs": 3,
        "min": 27,
        "sec": 34,
        "ms": 636,
        "us": 234,
        "ns": 0,
        "seconds": 12454,
        "microseconds": 636234,
        "nanoseconds": 0,
    }
    assert res2 == exp2

    res3 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_ms.value)
    exp3 = {
        "days": 144,
        "hrs": 3,
        "min": 37,
        "sec": 16,
        "ms": 234,
        "us": 0,
        "ns": 0,
        "seconds": 13036,
        "microseconds": 234000,
        "nanoseconds": 0,
    }
    assert res3 == exp3

    # Note this out of bounds for nanosecond Timedelta
    res4 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_s.value)
    exp4 = {
        "days": 144150,
        "hrs": 21,
        "min": 10,
        "sec": 34,
        "ms": 0,
        "us": 0,
        "ns": 0,
        "seconds": 76234,
        "microseconds": 0,
        "nanoseconds": 0,
    }
    assert res4 == exp4


class TestAstypeOverflowSafe:
    def test_pass_non_dt64_array(self):
        # check that we raise, not segfault
        arr = np.arange(5)
        dtype = np.dtype("M8[ns]")

        msg = (
            "astype_overflowsafe values.dtype and dtype must be either "
            "both-datetime64 or both-timedelta64"
        )
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=True)

        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=False)

    def test_pass_non_dt64_dtype(self):
        # check that we raise, not segfault
        arr = np.arange(5, dtype="i8").view("M8[D]")
        dtype = np.dtype("m8[ns]")

        msg = (
            "astype_overflowsafe values.dtype and dtype must be either "
            "both-datetime64 or both-timedelta64"
        )
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=True)

        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=False)

    def test_astype_overflowsafe_dt64(self):
        dtype = np.dtype("M8[ns]")

        dt = np.datetime64("2262-04-05", "D")
        arr = dt + np.arange(10, dtype="m8[D]")

        # arr.astype silently overflows, so this
        wrong = arr.astype(dtype)
        roundtrip = wrong.astype(arr.dtype)
        assert not (wrong == roundtrip).all()

        msg = "Out of bounds nanosecond timestamp"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            astype_overflowsafe(arr, dtype)

        # But converting to microseconds is fine, and we match numpy's results.
        dtype2 = np.dtype("M8[us]")
        result = astype_overflowsafe(arr, dtype2)
        expected = arr.astype(dtype2)
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_overflowsafe_td64(self):
        dtype = np.dtype("m8[ns]")

        dt = np.datetime64("2262-04-05", "D")
        arr = dt + np.arange(10, dtype="m8[D]")
        arr = arr.view("m8[D]")

        # arr.astype silently overflows, so this
        wrong = arr.astype(dtype)
        roundtrip = wrong.astype(arr.dtype)
        assert not (wrong == roundtrip).all()

        msg = r"Cannot convert 106752 days to timedelta64\[ns\] without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            astype_overflowsafe(arr, dtype)

        # But converting to microseconds is fine, and we match numpy's results.
        dtype2 = np.dtype("m8[us]")
        result = astype_overflowsafe(arr, dtype2)
        expected = arr.astype(dtype2)
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_overflowsafe_disallow_rounding(self):
        arr = np.array([-1500, 1500], dtype="M8[ns]")
        dtype = np.dtype("M8[us]")

        msg = "Cannot losslessly cast '-1500 ns' to us"
        with pytest.raises(ValueError, match=msg):
            astype_overflowsafe(arr, dtype, round_ok=False)

        result = astype_overflowsafe(arr, dtype, round_ok=True)
        expected = arr.astype(dtype)
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_round.py ===
import pytest

from pandas._libs.tslibs import to_offset
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG

from pandas import (
    DatetimeIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDatetimeIndexRound:
    def test_round_daily(self):
        dti = date_range("20130101 09:10:11", periods=5)
        result = dti.round("D")
        expected = date_range("20130101", periods=5)
        tm.assert_index_equal(result, expected)

        dti = dti.tz_localize("UTC").tz_convert("US/Eastern")
        result = dti.round("D")
        expected = date_range("20130101", periods=5).tz_localize("US/Eastern")
        tm.assert_index_equal(result, expected)

        result = dti.round("s")
        tm.assert_index_equal(result, dti)

    @pytest.mark.parametrize(
        "freq, error_msg",
        [
            ("YE", "<YearEnd: month=12> is a non-fixed frequency"),
            ("ME", "<MonthEnd> is a non-fixed frequency"),
            ("foobar", "Invalid frequency: foobar"),
        ],
    )
    def test_round_invalid(self, freq, error_msg):
        dti = date_range("20130101 09:10:11", periods=5)
        dti = dti.tz_localize("UTC").tz_convert("US/Eastern")
        with pytest.raises(ValueError, match=error_msg):
            dti.round(freq)

    def test_round(self, tz_naive_fixture, unit):
        tz = tz_naive_fixture
        rng = date_range(start="2016-01-01", periods=5, freq="30Min", tz=tz, unit=unit)
        elt = rng[1]

        expected_rng = DatetimeIndex(
            [
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 01:00:00", tz=tz),
                Timestamp("2016-01-01 02:00:00", tz=tz),
                Timestamp("2016-01-01 02:00:00", tz=tz),
            ]
        ).as_unit(unit)
        expected_elt = expected_rng[1]

        result = rng.round(freq="h")
        tm.assert_index_equal(result, expected_rng)
        assert elt.round(freq="h") == expected_elt

        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            rng.round(freq="foo")
        with pytest.raises(ValueError, match=msg):
            elt.round(freq="foo")

        msg = "<MonthEnd> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):
            rng.round(freq="ME")
        with pytest.raises(ValueError, match=msg):
            elt.round(freq="ME")

    def test_round2(self, tz_naive_fixture):
        tz = tz_naive_fixture
        # GH#14440 & GH#15578
        index = DatetimeIndex(["2016-10-17 12:00:00.0015"], tz=tz).as_unit("ns")
        result = index.round("ms")
        expected = DatetimeIndex(["2016-10-17 12:00:00.002000"], tz=tz).as_unit("ns")
        tm.assert_index_equal(result, expected)

        for freq in ["us", "ns"]:
            tm.assert_index_equal(index, index.round(freq))

    def test_round3(self, tz_naive_fixture):
        tz = tz_naive_fixture
        index = DatetimeIndex(["2016-10-17 12:00:00.00149"], tz=tz).as_unit("ns")
        result = index.round("ms")
        expected = DatetimeIndex(["2016-10-17 12:00:00.001000"], tz=tz).as_unit("ns")
        tm.assert_index_equal(result, expected)

    def test_round4(self, tz_naive_fixture):
        index = DatetimeIndex(["2016-10-17 12:00:00.001501031"], dtype="M8[ns]")
        result = index.round("10ns")
        expected = DatetimeIndex(["2016-10-17 12:00:00.001501030"], dtype="M8[ns]")
        tm.assert_index_equal(result, expected)

        ts = "2016-10-17 12:00:00.001501031"
        dti = DatetimeIndex([ts], dtype="M8[ns]")
        with tm.assert_produces_warning(False):
            dti.round("1010ns")

    def test_no_rounding_occurs(self, tz_naive_fixture):
        # GH 21262
        tz = tz_naive_fixture
        rng = date_range(start="2016-01-01", periods=5, freq="2Min", tz=tz)

        expected_rng = DatetimeIndex(
            [
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:02:00", tz=tz),
                Timestamp("2016-01-01 00:04:00", tz=tz),
                Timestamp("2016-01-01 00:06:00", tz=tz),
                Timestamp("2016-01-01 00:08:00", tz=tz),
            ]
        ).as_unit("ns")

        result = rng.round(freq="2min")
        tm.assert_index_equal(result, expected_rng)

    @pytest.mark.parametrize(
        "test_input, rounder, freq, expected",
        [
            (["2117-01-01 00:00:45"], "floor", "15s", ["2117-01-01 00:00:45"]),
            (["2117-01-01 00:00:45"], "ceil", "15s", ["2117-01-01 00:00:45"]),
            (
                ["2117-01-01 00:00:45.000000012"],
                "floor",
                "10ns",
                ["2117-01-01 00:00:45.000000010"],
            ),
            (
                ["1823-01-01 00:00:01.000000012"],
                "ceil",
                "10ns",
                ["1823-01-01 00:00:01.000000020"],
            ),
            (["1823-01-01 00:00:01"], "floor", "1s", ["1823-01-01 00:00:01"]),
            (["1823-01-01 00:00:01"], "ceil", "1s", ["1823-01-01 00:00:01"]),
            (["2018-01-01 00:15:00"], "ceil", "15min", ["2018-01-01 00:15:00"]),
            (["2018-01-01 00:15:00"], "floor", "15min", ["2018-01-01 00:15:00"]),
            (["1823-01-01 03:00:00"], "ceil", "3h", ["1823-01-01 03:00:00"]),
            (["1823-01-01 03:00:00"], "floor", "3h", ["1823-01-01 03:00:00"]),
            (
                ("NaT", "1823-01-01 00:00:01"),
                "floor",
                "1s",
                ("NaT", "1823-01-01 00:00:01"),
            ),
            (
                ("NaT", "1823-01-01 00:00:01"),
                "ceil",
                "1s",
                ("NaT", "1823-01-01 00:00:01"),
            ),
        ],
    )
    def test_ceil_floor_edge(self, test_input, rounder, freq, expected):
        dt = DatetimeIndex(list(test_input))
        func = getattr(dt, rounder)
        result = func(freq)
        expected = DatetimeIndex(list(expected))
        assert expected.equals(result)

    @pytest.mark.parametrize(
        "start, index_freq, periods",
        [("2018-01-01", "12h", 25), ("2018-01-01 0:0:0.124999", "1ns", 1000)],
    )
    @pytest.mark.parametrize(
        "round_freq",
        [
            "2ns",
            "3ns",
            "4ns",
            "5ns",
            "6ns",
            "7ns",
            "250ns",
            "500ns",
            "750ns",
            "1us",
            "19us",
            "250us",
            "500us",
            "750us",
            "1s",
            "2s",
            "3s",
            "12h",
            "1D",
        ],
    )
    def test_round_int64(self, start, index_freq, periods, round_freq):
        dt = date_range(start=start, freq=index_freq, periods=periods)
        unit = to_offset(round_freq).nanos

        # test floor
        result = dt.floor(round_freq)
        diff = dt.asi8 - result.asi8
        mod = result.asi8 % unit
        assert (mod == 0).all(), f"floor not a {round_freq} multiple"
        assert (0 <= diff).all() and (diff < unit).all(), "floor error"

        # test ceil
        result = dt.ceil(round_freq)
        diff = result.asi8 - dt.asi8
        mod = result.asi8 % unit
        assert (mod == 0).all(), f"ceil not a {round_freq} multiple"
        assert (0 <= diff).all() and (diff < unit).all(), "ceil error"

        # test round
        result = dt.round(round_freq)
        diff = abs(result.asi8 - dt.asi8)
        mod = result.asi8 % unit
        assert (mod == 0).all(), f"round not a {round_freq} multiple"
        assert (diff <= unit // 2).all(), "round error"
        if unit % 2 == 0:
            assert (
                result.asi8[diff == unit // 2] % 2 == 0
            ).all(), "round half to even error"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_printing.py ===
# -*- coding: utf-8 -*-
from sympy.core.function import Function
from sympy.integrals.integrals import Integral
from sympy.printing.latex import latex
from sympy.printing.pretty import pretty as xpretty
from sympy.vector import CoordSys3D, Del, Vector, express
from sympy.abc import a, b, c
from sympy.testing.pytest import XFAIL


def pretty(expr):
    """ASCII pretty-printing"""
    return xpretty(expr, use_unicode=False, wrap_line=False)


def upretty(expr):
    """Unicode pretty-printing"""
    return xpretty(expr, use_unicode=True, wrap_line=False)


# Initialize the basic and tedious vector/dyadic expressions
# needed for testing.
# Some of the pretty forms shown denote how the expressions just
# above them should look with pretty printing.
N = CoordSys3D('N')
C = N.orient_new_axis('C', a, N.k)  # type: ignore
v = []
d = []
v.append(Vector.zero)
v.append(N.i)  # type: ignore
v.append(-N.i)  # type: ignore
v.append(N.i + N.j)  # type: ignore
v.append(a*N.i)  # type: ignore
v.append(a*N.i - b*N.j)  # type: ignore
v.append((a**2 + N.x)*N.i + N.k)  # type: ignore
v.append((a**2 + b)*N.i + 3*(C.y - c)*N.k)  # type: ignore
f = Function('f')
v.append(N.j - (Integral(f(b)) - C.x**2)*N.k)  # type: ignore
upretty_v_8 = """\
      ⎛   2   ⌠        ⎞    \n\
j_N + ⎜x_C  - ⎮ f(b) db⎟ k_N\n\
      ⎝       ⌡        ⎠    \
"""
pretty_v_8 = """\
j_N + /         /       \\\n\
      |   2    |        |\n\
      |x_C  -  | f(b) db|\n\
      |        |        |\n\
      \\       /         / \
"""

v.append(N.i + C.k)  # type: ignore
v.append(express(N.i, C))  # type: ignore
v.append((a**2 + b)*N.i + (Integral(f(b)))*N.k)  # type: ignore
upretty_v_11 = """\
⎛ 2    ⎞        ⎛⌠        ⎞    \n\
⎝a  + b⎠ i_N  + ⎜⎮ f(b) db⎟ k_N\n\
                ⎝⌡        ⎠    \
"""
pretty_v_11 = """\
/ 2    \\ + /  /       \\\n\
\\a  + b/ i_N| |        |\n\
           | | f(b) db|\n\
           | |        |\n\
           \\/         / \
"""

for x in v:
    d.append(x | N.k)  # type: ignore
s = 3*N.x**2*C.y  # type: ignore
upretty_s = """\
         2\n\
3⋅y_C⋅x_N \
"""
pretty_s = """\
         2\n\
3*y_C*x_N \
"""

# This is the pretty form for ((a**2 + b)*N.i + 3*(C.y - c)*N.k) | N.k
upretty_d_7 = """\
⎛ 2    ⎞                                     \n\
⎝a  + b⎠ (i_N|k_N)  + (3⋅y_C - 3⋅c) (k_N|k_N)\
"""
pretty_d_7 = """\
/ 2    \\ (i_N|k_N) + (3*y_C - 3*c) (k_N|k_N)\n\
\\a  + b/                                    \
"""


def test_str_printing():
    assert str(v[0]) == '0'
    assert str(v[1]) == 'N.i'
    assert str(v[2]) == '(-1)*N.i'
    assert str(v[3]) == 'N.i + N.j'
    assert str(v[8]) == 'N.j + (C.x**2 - Integral(f(b), b))*N.k'
    assert str(v[9]) == 'C.k + N.i'
    assert str(s) == '3*C.y*N.x**2'
    assert str(d[0]) == '0'
    assert str(d[1]) == '(N.i|N.k)'
    assert str(d[4]) == 'a*(N.i|N.k)'
    assert str(d[5]) == 'a*(N.i|N.k) + (-b)*(N.j|N.k)'
    assert str(d[8]) == ('(N.j|N.k) + (C.x**2 - ' +
                         'Integral(f(b), b))*(N.k|N.k)')


@XFAIL
def test_pretty_printing_ascii():
    assert pretty(v[0]) == '0'
    assert pretty(v[1]) == 'i_N'
    assert pretty(v[5]) == '(a) i_N + (-b) j_N'
    assert pretty(v[8]) == pretty_v_8
    assert pretty(v[2]) == '(-1) i_N'
    assert pretty(v[11]) == pretty_v_11
    assert pretty(s) == pretty_s
    assert pretty(d[0]) == '(0|0)'
    assert pretty(d[5]) == '(a) (i_N|k_N) + (-b) (j_N|k_N)'
    assert pretty(d[7]) == pretty_d_7
    assert pretty(d[10]) == '(cos(a)) (i_C|k_N) + (-sin(a)) (j_C|k_N)'


def test_pretty_print_unicode_v():
    assert upretty(v[0]) == '0'
    assert upretty(v[1]) == 'i_N'
    assert upretty(v[5]) == '(a) i_N + (-b) j_N'
    # Make sure the printing works in other objects
    assert upretty(v[5].args) == '((a) i_N, (-b) j_N)'
    assert upretty(v[8]) == upretty_v_8
    assert upretty(v[2]) == '(-1) i_N'
    assert upretty(v[11]) == upretty_v_11
    assert upretty(s) == upretty_s
    assert upretty(d[0]) == '(0|0)'
    assert upretty(d[5]) == '(a) (i_N|k_N) + (-b) (j_N|k_N)'
    assert upretty(d[7]) == upretty_d_7
    assert upretty(d[10]) == '(cos(a)) (i_C|k_N) + (-sin(a)) (j_C|k_N)'


def test_latex_printing():
    assert latex(v[0]) == '\\mathbf{\\hat{0}}'
    assert latex(v[1]) == '\\mathbf{\\hat{i}_{N}}'
    assert latex(v[2]) == '- \\mathbf{\\hat{i}_{N}}'
    assert latex(v[5]) == ('\\left(a\\right)\\mathbf{\\hat{i}_{N}} + ' +
                           '\\left(- b\\right)\\mathbf{\\hat{j}_{N}}')
    assert latex(v[6]) == ('\\left(\\mathbf{{x}_{N}} + a^{2}\\right)\\mathbf{\\hat{i}_' +
                          '{N}} + \\mathbf{\\hat{k}_{N}}')
    assert latex(v[8]) == ('\\mathbf{\\hat{j}_{N}} + \\left(\\mathbf{{x}_' +
                           '{C}}^{2} - \\int f{\\left(b \\right)}\\,' +
                           ' db\\right)\\mathbf{\\hat{k}_{N}}')
    assert latex(s) == '3 \\mathbf{{y}_{C}} \\mathbf{{x}_{N}}^{2}'
    assert latex(d[0]) == '(\\mathbf{\\hat{0}}|\\mathbf{\\hat{0}})'
    assert latex(d[4]) == ('\\left(a\\right)\\left(\\mathbf{\\hat{i}_{N}}{\\middle|}' +
                           '\\mathbf{\\hat{k}_{N}}\\right)')
    assert latex(d[9]) == ('\\left(\\mathbf{\\hat{k}_{C}}{\\middle|}' +
                           '\\mathbf{\\hat{k}_{N}}\\right) + \\left(' +
                           '\\mathbf{\\hat{i}_{N}}{\\middle|}\\mathbf{' +
                           '\\hat{k}_{N}}\\right)')
    assert latex(d[11]) == ('\\left(a^{2} + b\\right)\\left(\\mathbf{\\hat{i}_{N}}' +
                            '{\\middle|}\\mathbf{\\hat{k}_{N}}\\right) + ' +
                            '\\left(\\int f{\\left(b \\right)}\\, db\\right)\\left(' +
                            '\\mathbf{\\hat{k}_{N}}{\\middle|}\\mathbf{' +
                            '\\hat{k}_{N}}\\right)')

def test_issue_23058():
    from sympy import symbols, sin, cos, pi, UnevaluatedExpr

    delop = Del()
    CC_   = CoordSys3D("C")
    y     = CC_.y
    xhat  = CC_.i

    t = symbols("t")
    ten = symbols("10", positive=True)
    eps, mu = 4*pi*ten**(-11), ten**(-5)

    Bx = 2 * ten**(-4) * cos(ten**5 * t) * sin(ten**(-3) * y)
    vecB = Bx * xhat
    vecE = (1/eps) * Integral(delop.cross(vecB/mu).doit(), t)
    vecE = vecE.doit()

    vecB_str = """\
⎛     ⎛y_C⎞    ⎛  5  ⎞⎞    \n\
⎜2⋅sin⎜───⎟⋅cos⎝10 ⋅t⎠⎟ i_C\n\
⎜     ⎜  3⎟           ⎟    \n\
⎜     ⎝10 ⎠           ⎟    \n\
⎜─────────────────────⎟    \n\
⎜           4         ⎟    \n\
⎝         10          ⎠    \
"""
    vecE_str = """\
⎛   4    ⎛  5  ⎞    ⎛y_C⎞ ⎞    \n\
⎜-10 ⋅sin⎝10 ⋅t⎠⋅cos⎜───⎟ ⎟ k_C\n\
⎜                   ⎜  3⎟ ⎟    \n\
⎜                   ⎝10 ⎠ ⎟    \n\
⎜─────────────────────────⎟    \n\
⎝           2⋅π           ⎠    \
"""

    assert upretty(vecB) == vecB_str
    assert upretty(vecE) == vecE_str

    ten = UnevaluatedExpr(10)
    eps, mu = 4*pi*ten**(-11), ten**(-5)

    Bx = 2 * ten**(-4) * cos(ten**5 * t) * sin(ten**(-3) * y)
    vecB = Bx * xhat

    vecB_str = """\
⎛    -4    ⎛    5⎞    ⎛      -3⎞⎞     \n\
⎝2⋅10  ⋅cos⎝t⋅10 ⎠⋅sin⎝y_C⋅10  ⎠⎠ i_C \
"""
    assert upretty(vecB) == vecB_str

def test_custom_names():
    A = CoordSys3D('A', vector_names=['x', 'y', 'z'],
                   variable_names=['i', 'j', 'k'])
    assert A.i.__str__() == 'A.i'
    assert A.x.__str__() == 'A.x'
    assert A.i._pretty_form == 'i_A'
    assert A.x._pretty_form == 'x_A'
    assert A.i._latex_form == r'\mathbf{{i}_{A}}'
    assert A.x._latex_form == r"\mathbf{\hat{x}_{A}}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\resample\test_timedelta.py ===
from datetime import timedelta

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range


def test_asfreq_bug():
    df = DataFrame(data=[1, 3], index=[timedelta(), timedelta(minutes=3)])
    result = df.resample("1min").asfreq()
    expected = DataFrame(
        data=[1, np.nan, np.nan, 3],
        index=timedelta_range("0 day", periods=4, freq="1min"),
    )
    tm.assert_frame_equal(result, expected)


def test_resample_with_nat():
    # GH 13223
    index = pd.to_timedelta(["0s", pd.NaT, "2s"])
    result = DataFrame({"value": [2, 3, 5]}, index).resample("1s").mean()
    expected = DataFrame(
        {"value": [2.5, np.nan, 5.0]},
        index=timedelta_range("0 day", periods=3, freq="1s"),
    )
    tm.assert_frame_equal(result, expected)


def test_resample_as_freq_with_subperiod():
    # GH 13022
    index = timedelta_range("00:00:00", "00:10:00", freq="5min")
    df = DataFrame(data={"value": [1, 5, 10]}, index=index)
    result = df.resample("2min").asfreq()
    expected_data = {"value": [1, np.nan, np.nan, np.nan, np.nan, 10]}
    expected = DataFrame(
        data=expected_data, index=timedelta_range("00:00:00", "00:10:00", freq="2min")
    )
    tm.assert_frame_equal(result, expected)


def test_resample_with_timedeltas():
    expected = DataFrame({"A": np.arange(1480)})
    expected = expected.groupby(expected.index // 30).sum()
    expected.index = timedelta_range("0 days", freq="30min", periods=50)

    df = DataFrame(
        {"A": np.arange(1480)}, index=pd.to_timedelta(np.arange(1480), unit="min")
    )
    result = df.resample("30min").sum()

    tm.assert_frame_equal(result, expected)

    s = df["A"]
    result = s.resample("30min").sum()
    tm.assert_series_equal(result, expected["A"])


def test_resample_single_period_timedelta():
    s = Series(list(range(5)), index=timedelta_range("1 day", freq="s", periods=5))
    result = s.resample("2s").sum()
    expected = Series([1, 5, 4], index=timedelta_range("1 day", freq="2s", periods=3))
    tm.assert_series_equal(result, expected)


def test_resample_timedelta_idempotency():
    # GH 12072
    index = timedelta_range("0", periods=9, freq="10ms")
    series = Series(range(9), index=index)
    result = series.resample("10ms").mean()
    expected = series.astype(float)
    tm.assert_series_equal(result, expected)


def test_resample_offset_with_timedeltaindex():
    # GH 10530 & 31809
    rng = timedelta_range(start="0s", periods=25, freq="s")
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    with_base = ts.resample("2s", offset="5s").mean()
    without_base = ts.resample("2s").mean()

    exp_without_base = timedelta_range(start="0s", end="25s", freq="2s")
    exp_with_base = timedelta_range(start="5s", end="29s", freq="2s")

    tm.assert_index_equal(without_base.index, exp_without_base)
    tm.assert_index_equal(with_base.index, exp_with_base)


def test_resample_categorical_data_with_timedeltaindex():
    # GH #12169
    df = DataFrame({"Group_obj": "A"}, index=pd.to_timedelta(list(range(20)), unit="s"))
    df["Group"] = df["Group_obj"].astype("category")
    result = df.resample("10s").agg(lambda x: (x.value_counts().index[0]))
    exp_tdi = pd.TimedeltaIndex(np.array([0, 10], dtype="m8[s]"), freq="10s").as_unit(
        "ns"
    )
    expected = DataFrame(
        {"Group_obj": ["A", "A"], "Group": ["A", "A"]},
        index=exp_tdi,
    )
    expected = expected.reindex(["Group_obj", "Group"], axis=1)
    expected["Group"] = expected["Group_obj"].astype("category")
    tm.assert_frame_equal(result, expected)


def test_resample_timedelta_values():
    # GH 13119
    # check that timedelta dtype is preserved when NaT values are
    # introduced by the resampling

    times = timedelta_range("1 day", "6 day", freq="4D")
    df = DataFrame({"time": times}, index=times)

    times2 = timedelta_range("1 day", "6 day", freq="2D")
    exp = Series(times2, index=times2, name="time")
    exp.iloc[1] = pd.NaT

    res = df.resample("2D").first()["time"]
    tm.assert_series_equal(res, exp)
    res = df["time"].resample("2D").first()
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize(
    "start, end, freq, resample_freq",
    [
        ("8h", "21h59min50s", "10s", "3h"),  # GH 30353 example
        ("3h", "22h", "1h", "5h"),
        ("527D", "5006D", "3D", "10D"),
        ("1D", "10D", "1D", "2D"),  # GH 13022 example
        # tests that worked before GH 33498:
        ("8h", "21h59min50s", "10s", "2h"),
        ("0h", "21h59min50s", "10s", "3h"),
        ("10D", "85D", "D", "2D"),
    ],
)
def test_resample_timedelta_edge_case(start, end, freq, resample_freq):
    # GH 33498
    # check that the timedelta bins does not contains an extra bin
    idx = timedelta_range(start=start, end=end, freq=freq)
    s = Series(np.arange(len(idx)), index=idx)
    result = s.resample(resample_freq).min()
    expected_index = timedelta_range(freq=resample_freq, start=start, end=end)
    tm.assert_index_equal(result.index, expected_index)
    assert result.index.freq == expected_index.freq
    assert not np.isnan(result.iloc[-1])


@pytest.mark.parametrize("duplicates", [True, False])
def test_resample_with_timedelta_yields_no_empty_groups(duplicates):
    # GH 10603
    df = DataFrame(
        np.random.default_rng(2).normal(size=(10000, 4)),
        index=timedelta_range(start="0s", periods=10000, freq="3906250ns"),
    )
    if duplicates:
        # case with non-unique columns
        df.columns = ["A", "B", "A", "C"]

    result = df.loc["1s":, :].resample("3s").apply(lambda x: len(x))

    expected = DataFrame(
        [[768] * 4] * 12 + [[528] * 4],
        index=timedelta_range(start="1s", periods=13, freq="3s"),
    )
    expected.columns = df.columns
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_resample_quantile_timedelta(unit):
    # GH: 29485
    dtype = np.dtype(f"m8[{unit}]")
    df = DataFrame(
        {"value": pd.to_timedelta(np.arange(4), unit="s").astype(dtype)},
        index=pd.date_range("20200101", periods=4, tz="UTC"),
    )
    result = df.resample("2D").quantile(0.99)
    expected = DataFrame(
        {
            "value": [
                pd.Timedelta("0 days 00:00:00.990000"),
                pd.Timedelta("0 days 00:00:02.990000"),
            ]
        },
        index=pd.date_range("20200101", periods=2, tz="UTC", freq="2D"),
    ).astype(dtype)
    tm.assert_frame_equal(result, expected)


def test_resample_closed_right():
    # GH#45414
    idx = pd.Index([pd.Timedelta(seconds=120 + i * 30) for i in range(10)])
    ser = Series(range(10), index=idx)
    result = ser.resample("min", closed="right", label="right").sum()
    expected = Series(
        [0, 3, 7, 11, 15, 9],
        index=pd.TimedeltaIndex(
            [pd.Timedelta(seconds=120 + i * 60) for i in range(6)], freq="min"
        ),
    )
    tm.assert_series_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_arrow_duration_resample():
    # GH 56371
    idx = pd.Index(timedelta_range("1 day", periods=5), dtype="duration[ns][pyarrow]")
    expected = Series(np.arange(5, dtype=np.float64), index=idx)
    result = expected.resample("1D").mean()
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_to_offset.py ===
import re

import pytest

from pandas._libs.tslibs import (
    Timedelta,
    offsets,
    to_offset,
)


@pytest.mark.parametrize(
    "freq_input,expected",
    [
        (to_offset("10us"), offsets.Micro(10)),
        (offsets.Hour(), offsets.Hour()),
        ("2h30min", offsets.Minute(150)),
        ("2h 30min", offsets.Minute(150)),
        ("2h30min15s", offsets.Second(150 * 60 + 15)),
        ("2h 60min", offsets.Hour(3)),
        ("2h 20.5min", offsets.Second(8430)),
        ("1.5min", offsets.Second(90)),
        ("0.5s", offsets.Milli(500)),
        ("15ms500us", offsets.Micro(15500)),
        ("10s75ms", offsets.Milli(10075)),
        ("1s0.25ms", offsets.Micro(1000250)),
        ("1s0.25ms", offsets.Micro(1000250)),
        ("2800ns", offsets.Nano(2800)),
        ("2SME", offsets.SemiMonthEnd(2)),
        ("2SME-16", offsets.SemiMonthEnd(2, day_of_month=16)),
        ("2SMS-14", offsets.SemiMonthBegin(2, day_of_month=14)),
        ("2SMS-15", offsets.SemiMonthBegin(2)),
    ],
)
def test_to_offset(freq_input, expected):
    result = to_offset(freq_input)
    assert result == expected


@pytest.mark.parametrize(
    "freqstr,expected", [("-1s", -1), ("-2SME", -2), ("-1SMS", -1), ("-5min10s", -310)]
)
def test_to_offset_negative(freqstr, expected):
    result = to_offset(freqstr)
    assert result.n == expected


@pytest.mark.filterwarnings("ignore:.*'m' is deprecated.*:FutureWarning")
@pytest.mark.parametrize(
    "freqstr",
    [
        "2h20m",
        "us1",
        "-us",
        "3us1",
        "-2-3us",
        "-2D:3h",
        "1.5.0s",
        "2SMS-15-15",
        "2SMS-15D",
        "100foo",
        # Invalid leading +/- signs.
        "+-1d",
        "-+1h",
        "+1",
        "-7",
        "+d",
        "-m",
        # Invalid shortcut anchors.
        "SME-0",
        "SME-28",
        "SME-29",
        "SME-FOO",
        "BSM",
        "SME--1",
        "SMS-1",
        "SMS-28",
        "SMS-30",
        "SMS-BAR",
        "SMS-BYR",
        "BSMS",
        "SMS--2",
    ],
)
def test_to_offset_invalid(freqstr):
    # see gh-13930

    # We escape string because some of our
    # inputs contain regex special characters.
    msg = re.escape(f"Invalid frequency: {freqstr}")
    with pytest.raises(ValueError, match=msg):
        to_offset(freqstr)


def test_to_offset_no_evaluate():
    msg = str(("", ""))
    with pytest.raises(TypeError, match=msg):
        to_offset(("", ""))


def test_to_offset_tuple_unsupported():
    with pytest.raises(TypeError, match="pass as a string instead"):
        to_offset((5, "T"))


@pytest.mark.parametrize(
    "freqstr,expected",
    [
        ("2D 3h", offsets.Hour(51)),
        ("2 D3 h", offsets.Hour(51)),
        ("2 D 3 h", offsets.Hour(51)),
        ("  2 D 3 h  ", offsets.Hour(51)),
        ("   h    ", offsets.Hour()),
        (" 3  h    ", offsets.Hour(3)),
    ],
)
def test_to_offset_whitespace(freqstr, expected):
    result = to_offset(freqstr)
    assert result == expected


@pytest.mark.parametrize(
    "freqstr,expected", [("00h 00min 01s", 1), ("-00h 03min 14s", -194)]
)
def test_to_offset_leading_zero(freqstr, expected):
    result = to_offset(freqstr)
    assert result.n == expected


@pytest.mark.parametrize("freqstr,expected", [("+1d", 1), ("+2h30min", 150)])
def test_to_offset_leading_plus(freqstr, expected):
    result = to_offset(freqstr)
    assert result.n == expected


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"days": 1, "seconds": 1}, offsets.Second(86401)),
        ({"days": -1, "seconds": 1}, offsets.Second(-86399)),
        ({"hours": 1, "minutes": 10}, offsets.Minute(70)),
        ({"hours": 1, "minutes": -10}, offsets.Minute(50)),
        ({"weeks": 1}, offsets.Day(7)),
        ({"hours": 1}, offsets.Hour(1)),
        ({"hours": 1}, to_offset("60min")),
        ({"microseconds": 1}, offsets.Micro(1)),
        ({"microseconds": 0}, offsets.Nano(0)),
    ],
)
def test_to_offset_pd_timedelta(kwargs, expected):
    # see gh-9064
    td = Timedelta(**kwargs)
    result = to_offset(td)
    assert result == expected


@pytest.mark.parametrize(
    "shortcut,expected",
    [
        ("W", offsets.Week(weekday=6)),
        ("W-SUN", offsets.Week(weekday=6)),
        ("QE", offsets.QuarterEnd(startingMonth=12)),
        ("QE-DEC", offsets.QuarterEnd(startingMonth=12)),
        ("QE-MAY", offsets.QuarterEnd(startingMonth=5)),
        ("SME", offsets.SemiMonthEnd(day_of_month=15)),
        ("SME-15", offsets.SemiMonthEnd(day_of_month=15)),
        ("SME-1", offsets.SemiMonthEnd(day_of_month=1)),
        ("SME-27", offsets.SemiMonthEnd(day_of_month=27)),
        ("SMS-2", offsets.SemiMonthBegin(day_of_month=2)),
        ("SMS-27", offsets.SemiMonthBegin(day_of_month=27)),
    ],
)
def test_anchored_shortcuts(shortcut, expected):
    result = to_offset(shortcut)
    assert result == expected


@pytest.mark.parametrize(
    "freq_depr",
    [
        "2ye-mar",
        "2ys",
        "2qe",
        "2qs-feb",
        "2bqs",
        "2sms",
        "2bms",
        "2cbme",
        "2me",
        "2w",
    ],
)
def test_to_offset_lowercase_frequency_deprecated(freq_depr):
    # GH#54939
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
    f"future version, please use '{freq_depr.upper()[1:]}' instead."

    with pytest.raises(FutureWarning, match=depr_msg):
        to_offset(freq_depr)


@pytest.mark.parametrize(
    "freq_depr",
    [
        "2H",
        "2BH",
        "2MIN",
        "2S",
        "2Us",
        "2NS",
    ],
)
def test_to_offset_uppercase_frequency_deprecated(freq_depr):
    # GH#54939
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
    f"future version, please use '{freq_depr.lower()[1:]}' instead."

    with pytest.raises(FutureWarning, match=depr_msg):
        to_offset(freq_depr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\timedeltas\test_reductions.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays import TimedeltaArray


class TestReductions:
    @pytest.mark.parametrize("name", ["std", "min", "max", "median", "mean"])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reductions_empty(self, name, skipna):
        tdi = pd.TimedeltaIndex([])
        arr = tdi.array

        result = getattr(tdi, name)(skipna=skipna)
        assert result is pd.NaT

        result = getattr(arr, name)(skipna=skipna)
        assert result is pd.NaT

    @pytest.mark.parametrize("skipna", [True, False])
    def test_sum_empty(self, skipna):
        tdi = pd.TimedeltaIndex([])
        arr = tdi.array

        result = tdi.sum(skipna=skipna)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

        result = arr.sum(skipna=skipna)
        assert isinstance(result, Timedelta)
        assert result == Timedelta(0)

    def test_min_max(self, unit):
        dtype = f"m8[{unit}]"
        arr = TimedeltaArray._from_sequence(
            ["3h", "3h", "NaT", "2h", "5h", "4h"], dtype=dtype
        )

        result = arr.min()
        expected = Timedelta("2h")
        assert result == expected

        result = arr.max()
        expected = Timedelta("5h")
        assert result == expected

        result = arr.min(skipna=False)
        assert result is pd.NaT

        result = arr.max(skipna=False)
        assert result is pd.NaT

    def test_sum(self):
        tdi = pd.TimedeltaIndex(["3h", "3h", "NaT", "2h", "5h", "4h"])
        arr = tdi.array

        result = arr.sum(skipna=True)
        expected = Timedelta(hours=17)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = tdi.sum(skipna=True)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = arr.sum(skipna=False)
        assert result is pd.NaT

        result = tdi.sum(skipna=False)
        assert result is pd.NaT

        result = arr.sum(min_count=9)
        assert result is pd.NaT

        result = tdi.sum(min_count=9)
        assert result is pd.NaT

        result = arr.sum(min_count=1)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = tdi.sum(min_count=1)
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_npsum(self):
        # GH#25282, GH#25335 np.sum should return a Timedelta, not timedelta64
        tdi = pd.TimedeltaIndex(["3h", "3h", "2h", "5h", "4h"])
        arr = tdi.array

        result = np.sum(tdi)
        expected = Timedelta(hours=17)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = np.sum(arr)
        assert isinstance(result, Timedelta)
        assert result == expected

    def test_sum_2d_skipna_false(self):
        arr = np.arange(8).astype(np.int64).view("m8[s]").astype("m8[ns]").reshape(4, 2)
        arr[-1, -1] = "Nat"

        tda = TimedeltaArray._from_sequence(arr)

        result = tda.sum(skipna=False)
        assert result is pd.NaT

        result = tda.sum(axis=0, skipna=False)
        expected = pd.TimedeltaIndex([Timedelta(seconds=12), pd.NaT])._values
        tm.assert_timedelta_array_equal(result, expected)

        result = tda.sum(axis=1, skipna=False)
        expected = pd.TimedeltaIndex(
            [
                Timedelta(seconds=1),
                Timedelta(seconds=5),
                Timedelta(seconds=9),
                pd.NaT,
            ]
        )._values
        tm.assert_timedelta_array_equal(result, expected)

    # Adding a Timestamp makes this a test for DatetimeArray.std
    @pytest.mark.parametrize(
        "add",
        [
            Timedelta(0),
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-01-01", tz="Asia/Tokyo"),
        ],
    )
    def test_std(self, add):
        tdi = pd.TimedeltaIndex(["0h", "4h", "NaT", "4h", "0h", "2h"]) + add
        arr = tdi.array

        result = arr.std(skipna=True)
        expected = Timedelta(hours=2)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = tdi.std(skipna=True)
        assert isinstance(result, Timedelta)
        assert result == expected

        if getattr(arr, "tz", None) is None:
            result = nanops.nanstd(np.asarray(arr), skipna=True)
            assert isinstance(result, np.timedelta64)
            assert result == expected

        result = arr.std(skipna=False)
        assert result is pd.NaT

        result = tdi.std(skipna=False)
        assert result is pd.NaT

        if getattr(arr, "tz", None) is None:
            result = nanops.nanstd(np.asarray(arr), skipna=False)
            assert isinstance(result, np.timedelta64)
            assert np.isnat(result)

    def test_median(self):
        tdi = pd.TimedeltaIndex(["0h", "3h", "NaT", "5h06m", "0h", "2h"])
        arr = tdi.array

        result = arr.median(skipna=True)
        expected = Timedelta(hours=2)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = tdi.median(skipna=True)
        assert isinstance(result, Timedelta)
        assert result == expected

        result = arr.median(skipna=False)
        assert result is pd.NaT

        result = tdi.median(skipna=False)
        assert result is pd.NaT

    def test_mean(self):
        tdi = pd.TimedeltaIndex(["0h", "3h", "NaT", "5h06m", "0h", "2h"])
        arr = tdi._data

        # manually verified result
        expected = Timedelta(arr.dropna()._ndarray.mean())

        result = arr.mean()
        assert result == expected
        result = arr.mean(skipna=False)
        assert result is pd.NaT

        result = arr.dropna().mean(skipna=False)
        assert result == expected

        result = arr.mean(axis=0)
        assert result == expected

    def test_mean_2d(self):
        tdi = pd.timedelta_range("14 days", periods=6)
        tda = tdi._data.reshape(3, 2)

        result = tda.mean(axis=0)
        expected = tda[1]
        tm.assert_timedelta_array_equal(result, expected)

        result = tda.mean(axis=1)
        expected = tda[:, 0] + Timedelta(hours=12)
        tm.assert_timedelta_array_equal(result, expected)

        result = tda.mean(axis=None)
        expected = tdi.mean()
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_array.py ===
import numpy as np
import pytest

from pandas.compat.numpy import np_version_gt2

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

# -----------------------------------------------------------------------------
# Copy/view behaviour for accessing underlying array of Series/DataFrame


@pytest.mark.parametrize(
    "method",
    [
        lambda ser: ser.values,
        lambda ser: np.asarray(ser),
        lambda ser: np.array(ser, copy=False),
    ],
    ids=["values", "asarray", "array"],
)
def test_series_values(using_copy_on_write, method):
    ser = Series([1, 2, 3], name="name")
    ser_orig = ser.copy()

    arr = method(ser)

    if using_copy_on_write:
        # .values still gives a view but is read-only
        assert np.shares_memory(arr, get_array(ser, "name"))
        assert arr.flags.writeable is False

        # mutating series through arr therefore doesn't work
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0
        tm.assert_series_equal(ser, ser_orig)

        # mutating the series itself still works
        ser.iloc[0] = 0
        assert ser.values[0] == 0
    else:
        assert arr.flags.writeable is True
        arr[0] = 0
        assert ser.iloc[0] == 0


@pytest.mark.parametrize(
    "method",
    [
        lambda df: df.values,
        lambda df: np.asarray(df),
        lambda ser: np.array(ser, copy=False),
    ],
    ids=["values", "asarray", "array"],
)
def test_dataframe_values(using_copy_on_write, using_array_manager, method):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_orig = df.copy()

    arr = method(df)

    if using_copy_on_write:
        # .values still gives a view but is read-only
        assert np.shares_memory(arr, get_array(df, "a"))
        assert arr.flags.writeable is False

        # mutating series through arr therefore doesn't work
        with pytest.raises(ValueError, match="read-only"):
            arr[0, 0] = 0
        tm.assert_frame_equal(df, df_orig)

        # mutating the series itself still works
        df.iloc[0, 0] = 0
        assert df.values[0, 0] == 0
    else:
        assert arr.flags.writeable is True
        arr[0, 0] = 0
        if not using_array_manager:
            assert df.iloc[0, 0] == 0
        else:
            tm.assert_frame_equal(df, df_orig)


def test_series_to_numpy(using_copy_on_write):
    ser = Series([1, 2, 3], name="name")
    ser_orig = ser.copy()

    # default: copy=False, no dtype or NAs
    arr = ser.to_numpy()
    if using_copy_on_write:
        # to_numpy still gives a view but is read-only
        assert np.shares_memory(arr, get_array(ser, "name"))
        assert arr.flags.writeable is False

        # mutating series through arr therefore doesn't work
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0
        tm.assert_series_equal(ser, ser_orig)

        # mutating the series itself still works
        ser.iloc[0] = 0
        assert ser.values[0] == 0
    else:
        assert arr.flags.writeable is True
        arr[0] = 0
        assert ser.iloc[0] == 0

    # specify copy=True gives a writeable array
    ser = Series([1, 2, 3], name="name")
    arr = ser.to_numpy(copy=True)
    assert not np.shares_memory(arr, get_array(ser, "name"))
    assert arr.flags.writeable is True

    # specifying a dtype that already causes a copy also gives a writeable array
    ser = Series([1, 2, 3], name="name")
    arr = ser.to_numpy(dtype="float64")
    assert not np.shares_memory(arr, get_array(ser, "name"))
    assert arr.flags.writeable is True


@pytest.mark.parametrize("order", ["F", "C"])
def test_ravel_read_only(using_copy_on_write, order):
    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match="is deprecated"):
        arr = ser.ravel(order=order)
    if using_copy_on_write:
        assert arr.flags.writeable is False
    assert np.shares_memory(get_array(ser), arr)


def test_series_array_ea_dtypes(using_copy_on_write):
    ser = Series([1, 2, 3], dtype="Int64")
    arr = np.asarray(ser, dtype="int64")
    assert np.shares_memory(arr, get_array(ser))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True

    arr = np.asarray(ser)
    assert np.shares_memory(arr, get_array(ser))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True


def test_dataframe_array_ea_dtypes(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    arr = np.asarray(df, dtype="int64")
    assert np.shares_memory(arr, get_array(df, "a"))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True

    arr = np.asarray(df)
    assert np.shares_memory(arr, get_array(df, "a"))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True


def test_dataframe_array_string_dtype(using_copy_on_write, using_array_manager):
    df = DataFrame({"a": ["a", "b"]}, dtype="string")
    arr = np.asarray(df)
    if not using_array_manager:
        assert np.shares_memory(arr, get_array(df, "a"))
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True


def test_dataframe_multiple_numpy_dtypes():
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    arr = np.asarray(df)
    assert not np.shares_memory(arr, get_array(df, "a"))
    assert arr.flags.writeable is True

    if np_version_gt2:
        # copy=False semantics are only supported in NumPy>=2.

        msg = "Starting with NumPy 2.0, the behavior of the 'copy' keyword has changed"
        with pytest.raises(FutureWarning, match=msg):
            arr = np.array(df, copy=False)

    arr = np.array(df, copy=True)
    assert arr.flags.writeable is True


def test_dataframe_single_block_copy_true():
    # the copy=False/None cases are tested above in test_dataframe_values
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    arr = np.array(df, copy=True)
    assert not np.shares_memory(arr, get_array(df, "a"))
    assert arr.flags.writeable is True


def test_values_is_ea(using_copy_on_write):
    df = DataFrame({"a": date_range("2012-01-01", periods=3)})
    arr = np.asarray(df)
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True


def test_empty_dataframe():
    df = DataFrame()
    arr = np.asarray(df)
    assert arr.flags.writeable is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_highlight.py ===
import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    IndexSlice,
)

pytest.importorskip("jinja2")

from pandas.io.formats.style import Styler


@pytest.fixture(params=[(None, "float64"), (NA, "Int64")])
def df(request):
    # GH 45804
    return DataFrame(
        {"A": [0, np.nan, 10], "B": [1, request.param[0], 2]}, dtype=request.param[1]
    )


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0)


def test_highlight_null(styler):
    result = styler.highlight_null()._compute().ctx
    expected = {
        (1, 0): [("background-color", "red")],
        (1, 1): [("background-color", "red")],
    }
    assert result == expected


def test_highlight_null_subset(styler):
    # GH 31345
    result = (
        styler.highlight_null(color="red", subset=["A"])
        .highlight_null(color="green", subset=["B"])
        ._compute()
        .ctx
    )
    expected = {
        (1, 0): [("background-color", "red")],
        (1, 1): [("background-color", "green")],
    }
    assert result == expected


@pytest.mark.parametrize("f", ["highlight_min", "highlight_max"])
def test_highlight_minmax_basic(df, f):
    expected = {
        (0, 1): [("background-color", "red")],
        # ignores NaN row,
        (2, 0): [("background-color", "red")],
    }
    if f == "highlight_min":
        df = -df
    result = getattr(df.style, f)(axis=1, color="red")._compute().ctx
    assert result == expected


@pytest.mark.parametrize("f", ["highlight_min", "highlight_max"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis": None, "color": "red"},  # test axis
        {"axis": 0, "subset": ["A"], "color": "red"},  # test subset and ignores NaN
        {"axis": None, "props": "background-color: red"},  # test props
    ],
)
def test_highlight_minmax_ext(df, f, kwargs):
    expected = {(2, 0): [("background-color", "red")]}
    if f == "highlight_min":
        df = -df
    result = getattr(df.style, f)(**kwargs)._compute().ctx
    assert result == expected


@pytest.mark.parametrize("f", ["highlight_min", "highlight_max"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_highlight_minmax_nulls(f, axis):
    # GH 42750
    expected = {
        (1, 0): [("background-color", "yellow")],
        (1, 1): [("background-color", "yellow")],
    }
    if axis == 1:
        expected.update({(2, 1): [("background-color", "yellow")]})

    if f == "highlight_max":
        df = DataFrame({"a": [NA, 1, None], "b": [np.nan, 1, -1]})
    else:
        df = DataFrame({"a": [NA, -1, None], "b": [np.nan, -1, 1]})

    result = getattr(df.style, f)(axis=axis)._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left": 0, "right": 1},  # test basic range
        {"left": 0, "right": 1, "props": "background-color: yellow"},  # test props
        {"left": -100, "right": 100, "subset": IndexSlice[[0, 1], :]},  # test subset
        {"left": 0, "subset": IndexSlice[[0, 1], :]},  # test no right
        {"right": 1},  # test no left
        {"left": [0, 0, 11], "axis": 0},  # test left as sequence
        {"left": DataFrame({"A": [0, 0, 11], "B": [1, 1, 11]}), "axis": None},  # axis
        {"left": 0, "right": [0, 1], "axis": 1},  # test sequence right
    ],
)
def test_highlight_between(styler, kwargs):
    expected = {
        (0, 0): [("background-color", "yellow")],
        (0, 1): [("background-color", "yellow")],
    }
    result = styler.highlight_between(**kwargs)._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "arg, map, axis",
    [
        ("left", [1, 2], 0),  # 0 axis has 3 elements not 2
        ("left", [1, 2, 3], 1),  # 1 axis has 2 elements not 3
        ("left", np.array([[1, 2], [1, 2]]), None),  # df is (2,3) not (2,2)
        ("right", [1, 2], 0),  # same tests as above for 'right' not 'left'
        ("right", [1, 2, 3], 1),  # ..
        ("right", np.array([[1, 2], [1, 2]]), None),  # ..
    ],
)
def test_highlight_between_raises(arg, styler, map, axis):
    msg = f"supplied '{arg}' is not correct shape"
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(**{arg: map, "axis": axis})._compute()


def test_highlight_between_raises2(styler):
    msg = "values can be 'both', 'left', 'right', or 'neither'"
    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(inclusive="badstring")._compute()

    with pytest.raises(ValueError, match=msg):
        styler.highlight_between(inclusive=1)._compute()


@pytest.mark.parametrize(
    "inclusive, expected",
    [
        (
            "both",
            {
                (0, 0): [("background-color", "yellow")],
                (0, 1): [("background-color", "yellow")],
            },
        ),
        ("neither", {}),
        ("left", {(0, 0): [("background-color", "yellow")]}),
        ("right", {(0, 1): [("background-color", "yellow")]}),
    ],
)
def test_highlight_between_inclusive(styler, inclusive, expected):
    kwargs = {"left": 0, "right": 1, "subset": IndexSlice[[0, 1], :]}
    result = styler.highlight_between(**kwargs, inclusive=inclusive)._compute()
    assert result.ctx == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q_left": 0.5, "q_right": 1, "axis": 0},  # base case
        {"q_left": 0.5, "q_right": 1, "axis": None},  # test axis
        {"q_left": 0, "q_right": 1, "subset": IndexSlice[2, :]},  # test subset
        {"q_left": 0.5, "axis": 0},  # test no high
        {"q_right": 1, "subset": IndexSlice[2, :], "axis": 1},  # test no low
        {"q_left": 0.5, "axis": 0, "props": "background-color: yellow"},  # tst prop
    ],
)
def test_highlight_quantile(styler, kwargs):
    expected = {
        (2, 0): [("background-color", "yellow")],
        (2, 1): [("background-color", "yellow")],
    }
    result = styler.highlight_quantile(**kwargs)._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "f,kwargs",
    [
        ("highlight_min", {"axis": 1, "subset": IndexSlice[1, :]}),
        ("highlight_max", {"axis": 0, "subset": [0]}),
        ("highlight_quantile", {"axis": None, "q_left": 0.6, "q_right": 0.8}),
        ("highlight_between", {"subset": [0]}),
    ],
)
@pytest.mark.parametrize(
    "df",
    [
        DataFrame([[0, 10], [20, 30]], dtype=int),
        DataFrame([[0, 10], [20, 30]], dtype=float),
        DataFrame([[0, 10], [20, 30]], dtype="datetime64[ns]"),
        DataFrame([[0, 10], [20, 30]], dtype=str),
        DataFrame([[0, 10], [20, 30]], dtype="timedelta64[ns]"),
    ],
)
def test_all_highlight_dtypes(f, kwargs, df):
    if f == "highlight_quantile" and isinstance(df.iloc[0, 0], (str)):
        return None  # quantile incompatible with str
    if f == "highlight_between":
        kwargs["left"] = df.iloc[1, 0]  # set the range low for testing

    expected = {(1, 0): [("background-color", "yellow")]}
    result = getattr(df.style, f)(**kwargs)._compute().ctx
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_symbol.py ===
"""
Tests related to the ``symbol`` attribute of the ABCPolyBase class.
"""

import pytest

import numpy.polynomial as poly
from numpy._core import array
from numpy.testing import assert_, assert_equal, assert_raises


class TestInit:
    """
    Test polynomial creation with symbol kwarg.
    """
    c = [1, 2, 3]

    def test_default_symbol(self):
        p = poly.Polynomial(self.c)
        assert_equal(p.symbol, 'x')

    @pytest.mark.parametrize(('bad_input', 'exception'), (
        ('', ValueError),
        ('3', ValueError),
        (None, TypeError),
        (1, TypeError),
    ))
    def test_symbol_bad_input(self, bad_input, exception):
        with pytest.raises(exception):
            p = poly.Polynomial(self.c, symbol=bad_input)

    @pytest.mark.parametrize('symbol', (
        'x',
        'x_1',
        'A',
        'xyz',
        'β',
    ))
    def test_valid_symbols(self, symbol):
        """
        Values for symbol that should pass input validation.
        """
        p = poly.Polynomial(self.c, symbol=symbol)
        assert_equal(p.symbol, symbol)

    def test_property(self):
        """
        'symbol' attribute is read only.
        """
        p = poly.Polynomial(self.c, symbol='x')
        with pytest.raises(AttributeError):
            p.symbol = 'z'

    def test_change_symbol(self):
        p = poly.Polynomial(self.c, symbol='y')
        # Create new polynomial from p with different symbol
        pt = poly.Polynomial(p.coef, symbol='t')
        assert_equal(pt.symbol, 't')


class TestUnaryOperators:
    p = poly.Polynomial([1, 2, 3], symbol='z')

    def test_neg(self):
        n = -self.p
        assert_equal(n.symbol, 'z')

    def test_scalarmul(self):
        out = self.p * 10
        assert_equal(out.symbol, 'z')

    def test_rscalarmul(self):
        out = 10 * self.p
        assert_equal(out.symbol, 'z')

    def test_pow(self):
        out = self.p ** 3
        assert_equal(out.symbol, 'z')


@pytest.mark.parametrize(
    'rhs',
    (
        poly.Polynomial([4, 5, 6], symbol='z'),
        array([4, 5, 6]),
    ),
)
class TestBinaryOperatorsSameSymbol:
    """
    Ensure symbol is preserved for numeric operations on polynomials with
    the same symbol
    """
    p = poly.Polynomial([1, 2, 3], symbol='z')

    def test_add(self, rhs):
        out = self.p + rhs
        assert_equal(out.symbol, 'z')

    def test_sub(self, rhs):
        out = self.p - rhs
        assert_equal(out.symbol, 'z')

    def test_polymul(self, rhs):
        out = self.p * rhs
        assert_equal(out.symbol, 'z')

    def test_divmod(self, rhs):
        for out in divmod(self.p, rhs):
            assert_equal(out.symbol, 'z')

    def test_radd(self, rhs):
        out = rhs + self.p
        assert_equal(out.symbol, 'z')

    def test_rsub(self, rhs):
        out = rhs - self.p
        assert_equal(out.symbol, 'z')

    def test_rmul(self, rhs):
        out = rhs * self.p
        assert_equal(out.symbol, 'z')

    def test_rdivmod(self, rhs):
        for out in divmod(rhs, self.p):
            assert_equal(out.symbol, 'z')


class TestBinaryOperatorsDifferentSymbol:
    p = poly.Polynomial([1, 2, 3], symbol='x')
    other = poly.Polynomial([4, 5, 6], symbol='y')
    ops = (p.__add__, p.__sub__, p.__mul__, p.__floordiv__, p.__mod__)

    @pytest.mark.parametrize('f', ops)
    def test_binops_fails(self, f):
        assert_raises(ValueError, f, self.other)


class TestEquality:
    p = poly.Polynomial([1, 2, 3], symbol='x')

    def test_eq(self):
        other = poly.Polynomial([1, 2, 3], symbol='x')
        assert_(self.p == other)

    def test_neq(self):
        other = poly.Polynomial([1, 2, 3], symbol='y')
        assert_(not self.p == other)


class TestExtraMethods:
    """
    Test other methods for manipulating/creating polynomial objects.
    """
    p = poly.Polynomial([1, 2, 3, 0], symbol='z')

    def test_copy(self):
        other = self.p.copy()
        assert_equal(other.symbol, 'z')

    def test_trim(self):
        other = self.p.trim()
        assert_equal(other.symbol, 'z')

    def test_truncate(self):
        other = self.p.truncate(2)
        assert_equal(other.symbol, 'z')

    @pytest.mark.parametrize('kwarg', (
        {'domain': [-10, 10]},
        {'window': [-10, 10]},
        {'kind': poly.Chebyshev},
    ))
    def test_convert(self, kwarg):
        other = self.p.convert(**kwarg)
        assert_equal(other.symbol, 'z')

    def test_integ(self):
        other = self.p.integ()
        assert_equal(other.symbol, 'z')

    def test_deriv(self):
        other = self.p.deriv()
        assert_equal(other.symbol, 'z')


def test_composition():
    p = poly.Polynomial([3, 2, 1], symbol="t")
    q = poly.Polynomial([5, 1, 0, -1], symbol="λ_1")
    r = p(q)
    assert r.symbol == "λ_1"


#
# Class methods that result in new polynomial class instances
#


def test_fit():
    x, y = (range(10),) * 2
    p = poly.Polynomial.fit(x, y, deg=1, symbol='z')
    assert_equal(p.symbol, 'z')


def test_froomroots():
    roots = [-2, 2]
    p = poly.Polynomial.fromroots(roots, symbol='z')
    assert_equal(p.symbol, 'z')


def test_identity():
    p = poly.Polynomial.identity(domain=[-1, 1], window=[5, 20], symbol='z')
    assert_equal(p.symbol, 'z')


def test_basis():
    p = poly.Polynomial.basis(3, symbol='z')
    assert_equal(p.symbol, 'z')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_randomstate_regression.py ===
import sys

import pytest

import numpy as np
from numpy import random
from numpy.testing import (
    assert_,
    assert_array_equal,
    assert_raises,
)


class TestRegression:

    def test_VonMises_range(self):
        # Make sure generated random variables are in [-pi, pi].
        # Regression test for ticket #986.
        for mu in np.linspace(-7., 7., 5):
            r = random.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_hypergeometric_range(self):
        # Test for ticket #921
        assert_(np.all(random.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(random.hypergeometric(18, 3, 11, size=10) > 0))

        # Test for ticket #5623
        args = [
            (2**20 - 2, 2**20 - 2, 2**20 - 2),  # Check for 32-bit systems
        ]
        is_64bits = sys.maxsize > 2**32
        if is_64bits and sys.platform != 'win32':
            # Check for 64-bit systems
            args.append((2**40 - 2, 2**40 - 2, 2**40 - 2))
        for arg in args:
            assert_(random.hypergeometric(*arg) > 0)

    def test_logseries_convergence(self):
        # Test for ticket #923
        N = 1000
        random.seed(0)
        rvsn = random.logseries(0.8, size=N)
        # these two frequency counts should be close to theoretical
        # numbers with this large sample
        # theoretical large N result is 0.49706795
        freq = np.sum(rvsn == 1) / N
        msg = f'Frequency was {freq:f}, should be > 0.45'
        assert_(freq > 0.45, msg)
        # theoretical large N result is 0.19882718
        freq = np.sum(rvsn == 2) / N
        msg = f'Frequency was {freq:f}, should be < 0.23'
        assert_(freq < 0.23, msg)

    def test_shuffle_mixed_dimension(self):
        # Test for trac ticket #2074
        for t in [[1, 2, 3, None],
                  [(1, 1), (2, 2), (3, 3), None],
                  [1, (2, 2), (3, 3), None],
                  [(1, 1), 2, 3, None]]:
            random.seed(12345)
            shuffled = list(t)
            random.shuffle(shuffled)
            expected = np.array([t[0], t[3], t[1], t[2]], dtype=object)
            assert_array_equal(np.array(shuffled, dtype=object), expected)

    def test_call_within_randomstate(self):
        # Check that custom RandomState does not call into global state
        m = random.RandomState()
        res = np.array([0, 8, 7, 2, 1, 9, 4, 7, 0, 3])
        for i in range(3):
            random.seed(i)
            m.seed(4321)
            # If m.state is not honored, the result will change
            assert_array_equal(m.choice(10, size=10, p=np.ones(10) / 10.), res)

    def test_multivariate_normal_size_types(self):
        # Test for multivariate_normal issue with 'size' argument.
        # Check that the multivariate_normal size argument can be a
        # numpy integer.
        random.multivariate_normal([0], [[0]], size=1)
        random.multivariate_normal([0], [[0]], size=np.int_(1))
        random.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        # Test that beta with small a and b parameters does not produce
        # NaNs due to roundoff errors causing 0 / 0, gh-5851
        random.seed(1234567890)
        x = random.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in random.beta')

    def test_choice_sum_of_probs_tolerance(self):
        # The sum of probs should be 1.0 with some tolerance.
        # For low precision dtypes the tolerance was too tight.
        # See numpy github issue 6123.
        random.seed(1234)
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = random.choice(a, p=probs)
            assert_(c in a)
            assert_raises(ValueError, random.choice, a, p=probs * 0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # Test that permuting an array of different length strings
        # will not cause a segfault on garbage collection
        # Tests gh-7710
        random.seed(1234)

        a = np.array(['a', 'a' * 1000])

        for _ in range(100):
            random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        # Test that permuting an array of objects will not cause
        # a segfault on garbage collection.
        # See gh-7719
        random.seed(1234)
        a = np.array([np.arange(1), np.arange(4)], dtype=object)

        for _ in range(1000):
            random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_permutation_subclass(self):
        class N(np.ndarray):
            pass

        random.seed(1)
        orig = np.arange(3).view(N)
        perm = random.permutation(orig)
        assert_array_equal(perm, np.array([0, 2, 1]))
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self, dtype=None, copy=None):
                return self.a

        random.seed(1)
        m = M()
        perm = random.permutation(m)
        assert_array_equal(perm, np.array([2, 1, 4, 0, 3]))
        assert_array_equal(m.__array__(), np.arange(5))

    def test_warns_byteorder(self):
        # GH 13159
        other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
        with pytest.deprecated_call(match='non-native byteorder is not'):
            random.randint(0, 200, size=10, dtype=other_byteord_dt)

    def test_named_argument_initialization(self):
        # GH 13669
        rs1 = np.random.RandomState(123456789)
        rs2 = np.random.RandomState(seed=123456789)
        assert rs1.randint(0, 100) == rs2.randint(0, 100)

    def test_choice_retun_dtype(self):
        # GH 9867, now long since the NumPy default changed.
        c = np.random.choice(10, p=[.1] * 10, size=2)
        assert c.dtype == np.dtype(np.long)
        c = np.random.choice(10, p=[.1] * 10, replace=False, size=2)
        assert c.dtype == np.dtype(np.long)
        c = np.random.choice(10, size=2)
        assert c.dtype == np.dtype(np.long)
        c = np.random.choice(10, replace=False, size=2)
        assert c.dtype == np.dtype(np.long)

    @pytest.mark.skipif(np.iinfo('l').max < 2**32,
                        reason='Cannot test with 32-bit C long')
    def test_randint_117(self):
        # GH 14189
        random.seed(0)
        expected = np.array([2357136044, 2546248239, 3071714933, 3626093760,
                             2588848963, 3684848379, 2340255427, 3638918503,
                             1819583497, 2678185683], dtype='int64')
        actual = random.randint(2**32, size=10)
        assert_array_equal(actual, expected)

    def test_p_zero_stream(self):
        # Regression test for gh-14522.  Ensure that future versions
        # generate the same variates as version 1.16.
        np.random.seed(12345)
        assert_array_equal(random.binomial(1, [0, 0.25, 0.5, 0.75, 1]),
                           [0, 0, 0, 1, 1])

    def test_n_zero_stream(self):
        # Regression test for gh-14522.  Ensure that future versions
        # generate the same variates as version 1.16.
        np.random.seed(8675309)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [3, 4, 2, 3, 3, 1, 5, 3, 1, 3]])
        assert_array_equal(random.binomial([[0], [10]], 0.25, size=(2, 10)),
                           expected)


def test_multinomial_empty():
    # gh-20483
    # Ensure that empty p-vals are correctly handled
    assert random.multinomial(10, []).shape == (0,)
    assert random.multinomial(3, [], size=(7, 5, 3)).shape == (7, 5, 3, 0)


def test_multinomial_1d_pval():
    # gh-20483
    with pytest.raises(TypeError, match="pvals must be a 1-d"):
        random.multinomial(10, 0.3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_extint128.py ===
import contextlib
import itertools
import operator

import numpy._core._multiarray_tests as mt
import pytest

import numpy as np
from numpy.testing import assert_equal, assert_raises

INT64_MAX = np.iinfo(np.int64).max
INT64_MIN = np.iinfo(np.int64).min
INT64_MID = 2**32

# int128 is not two's complement, the sign bit is separate
INT128_MAX = 2**128 - 1
INT128_MIN = -INT128_MAX
INT128_MID = 2**64

INT64_VALUES = (
    [INT64_MIN + j for j in range(20)] +
    [INT64_MAX - j for j in range(20)] +
    [INT64_MID + j for j in range(-20, 20)] +
    [2 * INT64_MID + j for j in range(-20, 20)] +
    [INT64_MID // 2 + j for j in range(-20, 20)] +
    list(range(-70, 70))
)

INT128_VALUES = (
    [INT128_MIN + j for j in range(20)] +
    [INT128_MAX - j for j in range(20)] +
    [INT128_MID + j for j in range(-20, 20)] +
    [2 * INT128_MID + j for j in range(-20, 20)] +
    [INT128_MID // 2 + j for j in range(-20, 20)] +
    list(range(-70, 70)) +
    [False]  # negative zero
)

INT64_POS_VALUES = [x for x in INT64_VALUES if x > 0]


@contextlib.contextmanager
def exc_iter(*args):
    """
    Iterate over Cartesian product of *args, and if an exception is raised,
    add information of the current iterate.
    """

    value = [None]

    def iterate():
        for v in itertools.product(*args):
            value[0] = v
            yield v

    try:
        yield iterate()
    except Exception:
        import traceback
        msg = f"At: {repr(value[0])!r}\n{traceback.format_exc()}"
        raise AssertionError(msg)


def test_safe_binop():
    # Test checked arithmetic routines

    ops = [
        (operator.add, 1),
        (operator.sub, 2),
        (operator.mul, 3)
    ]

    with exc_iter(ops, INT64_VALUES, INT64_VALUES) as it:
        for xop, a, b in it:
            pyop, op = xop
            c = pyop(a, b)

            if not (INT64_MIN <= c <= INT64_MAX):
                assert_raises(OverflowError, mt.extint_safe_binop, a, b, op)
            else:
                d = mt.extint_safe_binop(a, b, op)
                if c != d:
                    # assert_equal is slow
                    assert_equal(d, c)


def test_to_128():
    with exc_iter(INT64_VALUES) as it:
        for a, in it:
            b = mt.extint_to_128(a)
            if a != b:
                assert_equal(b, a)


def test_to_64():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            if not (INT64_MIN <= a <= INT64_MAX):
                assert_raises(OverflowError, mt.extint_to_64, a)
            else:
                b = mt.extint_to_64(a)
                if a != b:
                    assert_equal(b, a)


def test_mul_64_64():
    with exc_iter(INT64_VALUES, INT64_VALUES) as it:
        for a, b in it:
            c = a * b
            d = mt.extint_mul_64_64(a, b)
            if c != d:
                assert_equal(d, c)


def test_add_128():
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        for a, b in it:
            c = a + b
            if not (INT128_MIN <= c <= INT128_MAX):
                assert_raises(OverflowError, mt.extint_add_128, a, b)
            else:
                d = mt.extint_add_128(a, b)
                if c != d:
                    assert_equal(d, c)


def test_sub_128():
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        for a, b in it:
            c = a - b
            if not (INT128_MIN <= c <= INT128_MAX):
                assert_raises(OverflowError, mt.extint_sub_128, a, b)
            else:
                d = mt.extint_sub_128(a, b)
                if c != d:
                    assert_equal(d, c)


def test_neg_128():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            b = -a
            c = mt.extint_neg_128(a)
            if b != c:
                assert_equal(c, b)


def test_shl_128():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            if a < 0:
                b = -(((-a) << 1) & (2**128 - 1))
            else:
                b = (a << 1) & (2**128 - 1)
            c = mt.extint_shl_128(a)
            if b != c:
                assert_equal(c, b)


def test_shr_128():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            if a < 0:
                b = -((-a) >> 1)
            else:
                b = a >> 1
            c = mt.extint_shr_128(a)
            if b != c:
                assert_equal(c, b)


def test_gt_128():
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        for a, b in it:
            c = a > b
            d = mt.extint_gt_128(a, b)
            if c != d:
                assert_equal(d, c)


@pytest.mark.slow
def test_divmod_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            if a >= 0:
                c, cr = divmod(a, b)
            else:
                c, cr = divmod(-a, b)
                c = -c
                cr = -cr

            d, dr = mt.extint_divmod_128_64(a, b)

            if c != d or d != dr or b * d + dr != a:
                assert_equal(d, c)
                assert_equal(dr, cr)
                assert_equal(b * d + dr, a)


def test_floordiv_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            c = a // b
            d = mt.extint_floordiv_128_64(a, b)

            if c != d:
                assert_equal(d, c)


def test_ceildiv_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            c = (a + b - 1) // b
            d = mt.extint_ceildiv_128_64(a, b)

            if c != d:
                assert_equal(d, c)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_reductions.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm


@pytest.mark.parametrize("operation, expected", [("min", "a"), ("max", "b")])
def test_reductions_series_strings(operation, expected):
    # GH#31746
    ser = Series(["a", "b"], dtype="string")
    res_operation_serie = getattr(ser, operation)()
    assert res_operation_serie == expected


@pytest.mark.parametrize("as_period", [True, False])
def test_mode_extension_dtype(as_period):
    # GH#41927 preserve dt64tz dtype
    ser = Series([pd.Timestamp(1979, 4, n) for n in range(1, 5)])

    if as_period:
        ser = ser.dt.to_period("D")
    else:
        ser = ser.dt.tz_localize("US/Central")

    res = ser.mode()
    assert res.dtype == ser.dtype
    tm.assert_series_equal(res, ser)


def test_mode_nullable_dtype(any_numeric_ea_dtype):
    # GH#55340
    ser = Series([1, 3, 2, pd.NA, 3, 2, pd.NA], dtype=any_numeric_ea_dtype)
    result = ser.mode(dropna=False)
    expected = Series([2, 3, pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.mode(dropna=True)
    expected = Series([2, 3], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    ser[-1] = pd.NA

    result = ser.mode(dropna=True)
    expected = Series([2, 3], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.mode(dropna=False)
    expected = Series([pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)


def test_mode_infer_string():
    # GH#56183
    pytest.importorskip("pyarrow")
    ser = Series(["a", "b"], dtype=object)
    with pd.option_context("future.infer_string", True):
        result = ser.mode()
    expected = Series(["a", "b"], dtype=object)
    tm.assert_series_equal(result, expected)


def test_reductions_td64_with_nat():
    # GH#8617
    ser = Series([0, pd.NaT], dtype="m8[ns]")
    exp = ser[0]
    assert ser.median() == exp
    assert ser.min() == exp
    assert ser.max() == exp


@pytest.mark.parametrize("skipna", [True, False])
def test_td64_sum_empty(skipna):
    # GH#37151
    ser = Series([], dtype="timedelta64[ns]")

    result = ser.sum(skipna=skipna)
    assert isinstance(result, pd.Timedelta)
    assert result == pd.Timedelta(0)


def test_td64_summation_overflow():
    # GH#9442
    ser = Series(pd.date_range("20130101", periods=100000, freq="h"))
    ser[0] += pd.Timedelta("1s 1ms")

    # mean
    result = (ser - ser.min()).mean()
    expected = pd.Timedelta((pd.TimedeltaIndex(ser - ser.min()).asi8 / len(ser)).sum())

    # the computation is converted to float so
    # might be some loss of precision
    assert np.allclose(result._value / 1000, expected._value / 1000)

    # sum
    msg = "overflow in timedelta operation"
    with pytest.raises(ValueError, match=msg):
        (ser - ser.min()).sum()

    s1 = ser[0:10000]
    with pytest.raises(ValueError, match=msg):
        (s1 - s1.min()).sum()
    s2 = ser[0:1000]
    (s2 - s2.min()).sum()


def test_prod_numpy16_bug():
    ser = Series([1.0, 1.0, 1.0], index=range(3))
    result = ser.prod()

    assert not isinstance(result, Series)


@pytest.mark.parametrize("func", [np.any, np.all])
@pytest.mark.parametrize("kwargs", [{"keepdims": True}, {"out": object()}])
def test_validate_any_all_out_keepdims_raises(kwargs, func):
    ser = Series([1, 2])
    param = next(iter(kwargs))
    name = func.__name__

    msg = (
        f"the '{param}' parameter is not "
        "supported in the pandas "
        rf"implementation of {name}\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        func(ser, **kwargs)


def test_validate_sum_initial():
    ser = Series([1, 2])
    msg = (
        r"the 'initial' parameter is not "
        r"supported in the pandas "
        r"implementation of sum\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, initial=10)


def test_validate_median_initial():
    ser = Series([1, 2])
    msg = (
        r"the 'overwrite_input' parameter is not "
        r"supported in the pandas "
        r"implementation of median\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        # It seems like np.median doesn't dispatch, so we use the
        # method instead of the ufunc.
        ser.median(overwrite_input=True)


def test_validate_stat_keepdims():
    ser = Series([1, 2])
    msg = (
        r"the 'keepdims' parameter is not "
        r"supported in the pandas "
        r"implementation of sum\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, keepdims=True)


def test_mean_with_convertible_string_raises(using_array_manager, using_infer_string):
    # GH#44008
    ser = Series(["1", "2"])
    assert ser.sum() == "12"

    msg = "Could not convert string '12' to numeric|does not support|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        ser.mean()

    df = ser.to_frame()
    if not using_array_manager:
        msg = r"Could not convert \['12'\] to numeric|does not support|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        df.mean()


def test_mean_dont_convert_j_to_complex(using_array_manager):
    # GH#36703
    df = pd.DataFrame([{"db": "J", "numeric": 123}])
    if using_array_manager:
        msg = "Could not convert string 'J' to numeric"
    else:
        msg = r"Could not convert \['J'\] to numeric|does not support|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        df.mean()

    with pytest.raises(TypeError, match=msg):
        df.agg("mean")

    msg = "Could not convert string 'J' to numeric|does not support|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        df["db"].mean()
    msg = "Could not convert string 'J' to numeric|ufunc 'divide'|Cannot perform"
    with pytest.raises(TypeError, match=msg):
        np.mean(df["db"].astype("string").array)


def test_median_with_convertible_string_raises(using_array_manager):
    # GH#34671 this _could_ return a string "2", but definitely not float 2.0
    msg = r"Cannot convert \['1' '2' '3'\] to numeric|does not support|Cannot perform"
    ser = Series(["1", "2", "3"])
    with pytest.raises(TypeError, match=msg):
        ser.median()

    if not using_array_manager:
        msg = (
            r"Cannot convert \[\['1' '2' '3'\]\] to numeric|does not support"
            "|Cannot perform"
        )
    df = ser.to_frame()
    with pytest.raises(TypeError, match=msg):
        df.median()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_business_month.py ===
"""
Tests for the following offsets:
- BMonthBegin
- BMonthEnd
"""
from __future__ import annotations

from datetime import datetime

import pytest

import pandas as pd
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
)


@pytest.mark.parametrize("n", [-2, 1])
@pytest.mark.parametrize(
    "cls",
    [
        BMonthBegin,
        BMonthEnd,
    ],
)
def test_apply_index(cls, n):
    offset = cls(n=n)
    rng = pd.date_range(start="1/1/2000", periods=100000, freq="min")
    ser = pd.Series(rng)

    res = rng + offset
    assert res.freq is None  # not retained
    assert res[0] == rng[0] + offset
    assert res[-1] == rng[-1] + offset
    res2 = ser + offset
    # apply_index is only for indexes, not series, so no res2_v2
    assert res2.iloc[0] == ser.iloc[0] + offset
    assert res2.iloc[-1] == ser.iloc[-1] + offset


class TestBMonthBegin:
    def test_offsets_compare_equal(self):
        # root cause of #456
        offset1 = BMonthBegin()
        offset2 = BMonthBegin()
        assert not offset1 != offset2

    offset_cases = []
    offset_cases.append(
        (
            BMonthBegin(),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 9, 1): datetime(2006, 10, 2),
                datetime(2007, 1, 1): datetime(2007, 2, 1),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2006, 10, 2): datetime(2006, 10, 2),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 9, 15): datetime(2006, 10, 2),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(2),
            {
                datetime(2008, 1, 1): datetime(2008, 3, 3),
                datetime(2008, 1, 15): datetime(2008, 3, 3),
                datetime(2006, 12, 29): datetime(2007, 2, 1),
                datetime(2006, 12, 31): datetime(2007, 2, 1),
                datetime(2007, 1, 1): datetime(2007, 3, 1),
                datetime(2006, 11, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            BMonthBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),
                datetime(2008, 6, 30): datetime(2008, 6, 2),
                datetime(2008, 6, 1): datetime(2008, 5, 1),
                datetime(2008, 3, 10): datetime(2008, 3, 3),
                datetime(2008, 12, 31): datetime(2008, 12, 1),
                datetime(2006, 12, 29): datetime(2006, 12, 1),
                datetime(2006, 12, 30): datetime(2006, 12, 1),
                datetime(2007, 1, 1): datetime(2006, 12, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BMonthBegin(), datetime(2007, 12, 31), False),
        (BMonthBegin(), datetime(2008, 1, 1), True),
        (BMonthBegin(), datetime(2001, 4, 2), True),
        (BMonthBegin(), datetime(2008, 3, 3), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBMonthEnd:
    def test_normalize(self):
        dt = datetime(2007, 1, 1, 3)

        result = dt + BMonthEnd(normalize=True)
        expected = dt.replace(hour=0) + BMonthEnd()
        assert result == expected

    def test_offsets_compare_equal(self):
        # root cause of #456
        offset1 = BMonthEnd()
        offset2 = BMonthEnd()
        assert not offset1 != offset2

    offset_cases = []
    offset_cases.append(
        (
            BMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2007, 1, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2006, 12, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 29),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 3, 31),
                datetime(2006, 12, 29): datetime(2007, 2, 28),
                datetime(2006, 12, 31): datetime(2007, 2, 28),
                datetime(2007, 1, 1): datetime(2007, 2, 28),
                datetime(2006, 11, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BMonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 29),
                datetime(2008, 6, 30): datetime(2008, 5, 30),
                datetime(2008, 12, 31): datetime(2008, 11, 28),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 30): datetime(2006, 12, 29),
                datetime(2007, 1, 1): datetime(2006, 12, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BMonthEnd(), datetime(2007, 12, 31), True),
        (BMonthEnd(), datetime(2008, 1, 1), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_containers.py ===
from collections import defaultdict

from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.numbers import Integer
from sympy.core.kind import NumberKind
from sympy.matrices.kind import MatrixKind
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.sets.sets import FiniteSet
from sympy.core.containers import tuple_wrapper, TupleKind
from sympy.core.expr import unchanged
from sympy.core.function import Function, Lambda
from sympy.core.relational import Eq
from sympy.testing.pytest import raises
from sympy.utilities.iterables import is_sequence, iterable

from sympy.abc import x, y


def test_Tuple():
    t = (1, 2, 3, 4)
    st = Tuple(*t)
    assert set(sympify(t)) == set(st)
    assert len(t) == len(st)
    assert set(sympify(t[:2])) == set(st[:2])
    assert isinstance(st[:], Tuple)
    assert st == Tuple(1, 2, 3, 4)
    assert st.func(*st.args) == st
    p, q, r, s = symbols('p q r s')
    t2 = (p, q, r, s)
    st2 = Tuple(*t2)
    assert st2.atoms() == set(t2)
    assert st == st2.subs({p: 1, q: 2, r: 3, s: 4})
    # issue 5505
    assert all(isinstance(arg, Basic) for arg in st.args)
    assert Tuple(p, 1).subs(p, 0) == Tuple(0, 1)
    assert Tuple(p, Tuple(p, 1)).subs(p, 0) == Tuple(0, Tuple(0, 1))

    assert Tuple(t2) == Tuple(Tuple(*t2))
    assert Tuple.fromiter(t2) == Tuple(*t2)
    assert Tuple.fromiter(x for x in range(4)) == Tuple(0, 1, 2, 3)
    assert st2.fromiter(st2.args) == st2


def test_Tuple_contains():
    t1, t2 = Tuple(1), Tuple(2)
    assert t1 in Tuple(1, 2, 3, t1, Tuple(t2))
    assert t2 not in Tuple(1, 2, 3, t1, Tuple(t2))


def test_Tuple_concatenation():
    assert Tuple(1, 2) + Tuple(3, 4) == Tuple(1, 2, 3, 4)
    assert (1, 2) + Tuple(3, 4) == Tuple(1, 2, 3, 4)
    assert Tuple(1, 2) + (3, 4) == Tuple(1, 2, 3, 4)
    raises(TypeError, lambda: Tuple(1, 2) + 3)
    raises(TypeError, lambda: 1 + Tuple(2, 3))

    #the Tuple case in __radd__ is only reached when a subclass is involved
    class Tuple2(Tuple):
        def __radd__(self, other):
            return Tuple.__radd__(self, other + other)
    assert Tuple(1, 2) + Tuple2(3, 4) == Tuple(1, 2, 1, 2, 3, 4)
    assert Tuple2(1, 2) + Tuple(3, 4) == Tuple(1, 2, 3, 4)


def test_Tuple_equality():
    assert not isinstance(Tuple(1, 2), tuple)
    assert (Tuple(1, 2) == (1, 2)) is True
    assert (Tuple(1, 2) != (1, 2)) is False
    assert (Tuple(1, 2) == (1, 3)) is False
    assert (Tuple(1, 2) != (1, 3)) is True
    assert (Tuple(1, 2) == Tuple(1, 2)) is True
    assert (Tuple(1, 2) != Tuple(1, 2)) is False
    assert (Tuple(1, 2) == Tuple(1, 3)) is False
    assert (Tuple(1, 2) != Tuple(1, 3)) is True


def test_Tuple_Eq():
    assert Eq(Tuple(), Tuple()) is S.true
    assert Eq(Tuple(1), 1) is S.false
    assert Eq(Tuple(1, 2), Tuple(1)) is S.false
    assert Eq(Tuple(1), Tuple(1)) is S.true
    assert Eq(Tuple(1, 2), Tuple(1, 3)) is S.false
    assert Eq(Tuple(1, 2), Tuple(1, 2)) is S.true
    assert unchanged(Eq, Tuple(1, x), Tuple(1, 2))
    assert Eq(Tuple(1, x), Tuple(1, 2)).subs(x, 2) is S.true
    assert unchanged(Eq, Tuple(1, 2), x)
    f = Function('f')
    assert unchanged(Eq, Tuple(1), f(x))
    assert Eq(Tuple(1), f(x)).subs(x, 1).subs(f, Lambda(y, (y,))) is S.true


def test_Tuple_comparision():
    assert (Tuple(1, 3) >= Tuple(-10, 30)) is S.true
    assert (Tuple(1, 3) <= Tuple(-10, 30)) is S.false
    assert (Tuple(1, 3) >= Tuple(1, 3)) is S.true
    assert (Tuple(1, 3) <= Tuple(1, 3)) is S.true


def test_Tuple_tuple_count():
    assert Tuple(0, 1, 2, 3).tuple_count(4) == 0
    assert Tuple(0, 4, 1, 2, 3).tuple_count(4) == 1
    assert Tuple(0, 4, 1, 4, 2, 3).tuple_count(4) == 2
    assert Tuple(0, 4, 1, 4, 2, 4, 3).tuple_count(4) == 3


def test_Tuple_index():
    assert Tuple(4, 0, 1, 2, 3).index(4) == 0
    assert Tuple(0, 4, 1, 2, 3).index(4) == 1
    assert Tuple(0, 1, 4, 2, 3).index(4) == 2
    assert Tuple(0, 1, 2, 4, 3).index(4) == 3
    assert Tuple(0, 1, 2, 3, 4).index(4) == 4

    raises(ValueError, lambda: Tuple(0, 1, 2, 3).index(4))
    raises(ValueError, lambda: Tuple(4, 0, 1, 2, 3).index(4, 1))
    raises(ValueError, lambda: Tuple(0, 1, 2, 3, 4).index(4, 1, 4))


def test_Tuple_mul():
    assert Tuple(1, 2, 3)*2 == Tuple(1, 2, 3, 1, 2, 3)
    assert 2*Tuple(1, 2, 3) == Tuple(1, 2, 3, 1, 2, 3)
    assert Tuple(1, 2, 3)*Integer(2) == Tuple(1, 2, 3, 1, 2, 3)
    assert Integer(2)*Tuple(1, 2, 3) == Tuple(1, 2, 3, 1, 2, 3)

    raises(TypeError, lambda: Tuple(1, 2, 3)*S.Half)
    raises(TypeError, lambda: S.Half*Tuple(1, 2, 3))


def test_tuple_wrapper():

    @tuple_wrapper
    def wrap_tuples_and_return(*t):
        return t

    p = symbols('p')
    assert wrap_tuples_and_return(p, 1) == (p, 1)
    assert wrap_tuples_and_return((p, 1)) == (Tuple(p, 1),)
    assert wrap_tuples_and_return(1, (p, 2), 3) == (1, Tuple(p, 2), 3)


def test_iterable_is_sequence():
    ordered = [[], (), Tuple(), Matrix([[]])]
    unordered = [set()]
    not_sympy_iterable = [{}, '', '']
    assert all(is_sequence(i) for i in ordered)
    assert all(not is_sequence(i) for i in unordered)
    assert all(iterable(i) for i in ordered + unordered)
    assert all(not iterable(i) for i in not_sympy_iterable)
    assert all(iterable(i, exclude=None) for i in not_sympy_iterable)


def test_TupleKind():
    kind = TupleKind(NumberKind, MatrixKind(NumberKind))
    assert Tuple(1, Matrix([1, 2])).kind is kind
    assert Tuple(1, 2).kind is TupleKind(NumberKind, NumberKind)
    assert Tuple(1, 2).kind.element_kind == (NumberKind, NumberKind)


def test_Dict():
    x, y, z = symbols('x y z')
    d = Dict({x: 1, y: 2, z: 3})
    assert d[x] == 1
    assert d[y] == 2
    raises(KeyError, lambda: d[2])
    raises(KeyError, lambda: d['2'])
    assert len(d) == 3
    assert set(d.keys()) == {x, y, z}
    assert set(d.values()) == {S.One, S(2), S(3)}
    assert d.get(5, 'default') == 'default'
    assert d.get('5', 'default') == 'default'
    assert x in d and z in d and 5 not in d and '5' not in d
    assert d.has(x) and d.has(1)  # SymPy Basic .has method

    # Test input types
    # input - a Python dict
    # input - items as args - SymPy style
    assert (Dict({x: 1, y: 2, z: 3}) ==
            Dict((x, 1), (y, 2), (z, 3)))

    raises(TypeError, lambda: Dict(((x, 1), (y, 2), (z, 3))))
    with raises(NotImplementedError):
        d[5] = 6  # assert immutability

    assert set(
        d.items()) == {Tuple(x, S.One), Tuple(y, S(2)), Tuple(z, S(3))}
    assert set(d) == {x, y, z}
    assert str(d) == '{x: 1, y: 2, z: 3}'
    assert d.__repr__() == '{x: 1, y: 2, z: 3}'

    # Test creating a Dict from a Dict.
    d = Dict({x: 1, y: 2, z: 3})
    assert d == Dict(d)

    # Test for supporting defaultdict
    d = defaultdict(int)
    assert d[x] == 0
    assert d[y] == 0
    assert d[z] == 0
    assert Dict(d)
    d = Dict(d)
    assert len(d) == 3
    assert set(d.keys()) == {x, y, z}
    assert set(d.values()) == {S.Zero, S.Zero, S.Zero}


def test_issue_5788():
    args = [(1, 2), (2, 1)]
    for o in [Dict, Tuple, FiniteSet]:
        # __eq__ and arg handling
        if o != Tuple:
            assert o(*args) == o(*reversed(args))
        pair = [o(*args), o(*reversed(args))]
        assert sorted(pair) == sorted(pair)
        assert set(o(*args))  # doesn't fail

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_matpow.py ===
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.powsimp import powsimp
from sympy.testing.pytest import raises
from sympy.core.expr import unchanged
from sympy.core import symbols, S
from sympy.matrices import Identity, MatrixSymbol, ImmutableMatrix, ZeroMatrix, OneMatrix, Matrix
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.expressions import MatPow, MatAdd, MatMul
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixElement

n, m, l, k = symbols('n m l k', integer=True)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
E = MatrixSymbol('E', m, n)


def test_entry_matrix():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    assert MatPow(X, 0)[0, 0] == 1
    assert MatPow(X, 0)[0, 1] == 0
    assert MatPow(X, 1)[0, 0] == 1
    assert MatPow(X, 1)[0, 1] == 2
    assert MatPow(X, 2)[0, 0] == 7


def test_entry_symbol():
    from sympy.concrete import Sum
    assert MatPow(C, 0)[0, 0] == 1
    assert MatPow(C, 0)[0, 1] == 0
    assert MatPow(C, 1)[0, 0] == C[0, 0]
    assert isinstance(MatPow(C, 2)[0, 0], Sum)
    assert isinstance(MatPow(C, n)[0, 0], MatrixElement)


def test_as_explicit_symbol():
    X = MatrixSymbol('X', 2, 2)
    assert MatPow(X, 0).as_explicit() == ImmutableMatrix(Identity(2))
    assert MatPow(X, 1).as_explicit() == X.as_explicit()
    assert MatPow(X, 2).as_explicit() == (X.as_explicit())**2
    assert MatPow(X, n).as_explicit() == ImmutableMatrix([
        [(X ** n)[0, 0], (X ** n)[0, 1]],
        [(X ** n)[1, 0], (X ** n)[1, 1]],
    ])

    a = MatrixSymbol("a", 3, 1)
    b = MatrixSymbol("b", 3, 1)
    c = MatrixSymbol("c", 3, 1)

    expr = (a.T*b)**S.Half
    assert expr.as_explicit() == Matrix([[sqrt(a[0, 0]*b[0, 0] + a[1, 0]*b[1, 0] + a[2, 0]*b[2, 0])]])

    expr = c*(a.T*b)**S.Half
    m = sqrt(a[0, 0]*b[0, 0] + a[1, 0]*b[1, 0] + a[2, 0]*b[2, 0])
    assert expr.as_explicit() == Matrix([[c[0, 0]*m], [c[1, 0]*m], [c[2, 0]*m]])

    expr = (a*b.T)**S.Half
    denom = sqrt(a[0, 0]*b[0, 0] + a[1, 0]*b[1, 0] + a[2, 0]*b[2, 0])
    expected = (a*b.T).as_explicit()/denom
    assert expr.as_explicit() == expected

    expr = X**-1
    det = X[0, 0]*X[1, 1] - X[1, 0]*X[0, 1]
    expected = Matrix([[X[1, 1], -X[0, 1]], [-X[1, 0], X[0, 0]]])/det
    assert expr.as_explicit() == expected

    expr = X**m
    assert expr.as_explicit() == X.as_explicit()**m


def test_as_explicit_matrix():
    A = ImmutableMatrix([[1, 2], [3, 4]])
    assert MatPow(A, 0).as_explicit() == ImmutableMatrix(Identity(2))
    assert MatPow(A, 1).as_explicit() == A
    assert MatPow(A, 2).as_explicit() == A**2
    assert MatPow(A, -1).as_explicit() == A.inv()
    assert MatPow(A, -2).as_explicit() == (A.inv())**2
    # less expensive than testing on a 2x2
    A = ImmutableMatrix([4])
    assert MatPow(A, S.Half).as_explicit() == A**S.Half


def test_doit_symbol():
    assert MatPow(C, 0).doit() == Identity(n)
    assert MatPow(C, 1).doit() == C
    assert MatPow(C, -1).doit() == C.I
    for r in [2, S.Half, S.Pi, n]:
        assert MatPow(C, r).doit() == MatPow(C, r)


def test_doit_matrix():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    assert MatPow(X, 0).doit() == ImmutableMatrix(Identity(2))
    assert MatPow(X, 1).doit() == X
    assert MatPow(X, 2).doit() == X**2
    assert MatPow(X, -1).doit() == X.inv()
    assert MatPow(X, -2).doit() == (X.inv())**2
    # less expensive than testing on a 2x2
    assert MatPow(ImmutableMatrix([4]), S.Half).doit() == ImmutableMatrix([2])
    X = ImmutableMatrix([[0, 2], [0, 4]]) # det() == 0
    raises(ValueError, lambda: MatPow(X,-1).doit())
    raises(ValueError, lambda: MatPow(X,-2).doit())


def test_nonsquare():
    A = MatrixSymbol('A', 2, 3)
    B = ImmutableMatrix([[1, 2, 3], [4, 5, 6]])
    for r in [-1, 0, 1, 2, S.Half, S.Pi, n]:
        raises(NonSquareMatrixError, lambda: MatPow(A, r))
        raises(NonSquareMatrixError, lambda: MatPow(B, r))


def test_doit_equals_pow(): #17179
    X = ImmutableMatrix ([[1,0],[0,1]])
    assert MatPow(X, n).doit() == X**n == X


def test_doit_nested_MatrixExpr():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[2, 3], [4, 5]])
    assert MatPow(MatMul(X, Y), 2).doit() == (X*Y)**2
    assert MatPow(MatAdd(X, Y), 2).doit() == (X + Y)**2


def test_identity_power():
    k = Identity(n)
    assert MatPow(k, 4).doit() == k
    assert MatPow(k, n).doit() == k
    assert MatPow(k, -3).doit() == k
    assert MatPow(k, 0).doit() == k
    l = Identity(3)
    assert MatPow(l, n).doit() == l
    assert MatPow(l, -1).doit() == l
    assert MatPow(l, 0).doit() == l


def test_zero_power():
    z1 = ZeroMatrix(n, n)
    assert MatPow(z1, 3).doit() == z1
    raises(ValueError, lambda:MatPow(z1, -1).doit())
    assert MatPow(z1, 0).doit() == Identity(n)
    assert MatPow(z1, n).doit() == z1
    raises(ValueError, lambda:MatPow(z1, -2).doit())
    z2 = ZeroMatrix(4, 4)
    assert MatPow(z2, n).doit() == z2
    raises(ValueError, lambda:MatPow(z2, -3).doit())
    assert MatPow(z2, 2).doit() == z2
    assert MatPow(z2, 0).doit() == Identity(4)
    raises(ValueError, lambda:MatPow(z2, -1).doit())


def test_OneMatrix_power():
    o = OneMatrix(3, 3)
    assert o ** 0 == Identity(3)
    assert o ** 1 == o
    assert o * o == o ** 2 == 3 * o
    assert o * o * o == o ** 3 == 9 * o

    o = OneMatrix(n, n)
    assert o * o == o ** 2 == n * o
    # powsimp necessary as n ** (n - 2) * n does not produce n ** (n - 1)
    assert powsimp(o ** (n - 1) * o) == o ** n == n ** (n - 1) * o


def test_transpose_power():
    from sympy.matrices.expressions.transpose import Transpose as TP

    assert (C*D).T**5 == ((C*D)**5).T == (D.T * C.T)**5
    assert ((C*D).T**5).T == (C*D)**5

    assert (C.T.I.T)**7 == C**-7
    assert (C.T**l).T**k == C**(l*k)

    assert ((E.T * A.T)**5).T == (A*E)**5
    assert ((A*E).T**5).T**7 == (A*E)**35
    assert TP(TP(C**2 * D**3)**5).doit() == (C**2 * D**3)**5

    assert ((D*C)**-5).T**-5 == ((D*C)**25).T
    assert (((D*C)**l).T**k).T == (D*C)**(l*k)


def test_Inverse():
    assert Inverse(MatPow(C, 0)).doit() == Identity(n)
    assert Inverse(MatPow(C, 1)).doit() == Inverse(C)
    assert Inverse(MatPow(C, 2)).doit() == MatPow(C, -2)
    assert Inverse(MatPow(C, -1)).doit() == C

    assert MatPow(Inverse(C), 0).doit() == Identity(n)
    assert MatPow(Inverse(C), 1).doit() == Inverse(C)
    assert MatPow(Inverse(C), 2).doit() == MatPow(C, -2)
    assert MatPow(Inverse(C), -1).doit() == C


def test_combine_powers():
    assert (C ** 1) ** 1 == C
    assert (C ** 2) ** 3 == MatPow(C, 6)
    assert (C ** -2) ** -3 == MatPow(C, 6)
    assert (C ** -1) ** -1 == C
    assert (((C ** 2) ** 3) ** 4) ** 5 == MatPow(C, 120)
    assert (C ** n) ** n == C ** (n ** 2)


def test_unchanged():
    assert unchanged(MatPow, C, 0)
    assert unchanged(MatPow, C, 1)
    assert unchanged(MatPow, Inverse(C), -1)
    assert unchanged(Inverse, MatPow(C, -1), -1)
    assert unchanged(MatPow, MatPow(C, -1), -1)
    assert unchanged(MatPow, MatPow(C, 1), 1)


def test_no_exponentiation():
    # if this passes, Pow.as_numer_denom should recognize
    # MatAdd as exponent
    raises(NotImplementedError, lambda: 3**(-2*C))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_calculus.py ===
import pytest
from mpmath import *

def test_approximation():
    mp.dps = 15
    f = lambda x: cos(2-2*x)/x
    p, err = chebyfit(f, [2, 4], 8, error=True)
    assert err < 1e-5
    for i in range(10):
        x = 2 + i/5.
        assert abs(polyval(p, x) - f(x)) < err

def test_limits():
    mp.dps = 15
    assert limit(lambda x: (x-sin(x))/x**3, 0).ae(mpf(1)/6)
    assert limit(lambda n: (1+1/n)**n, inf).ae(e)

def test_polyval():
    assert polyval([], 3) == 0
    assert polyval([0], 3) == 0
    assert polyval([5], 3) == 5
    # 4x^3 - 2x + 5
    p = [4, 0, -2, 5]
    assert polyval(p,4) == 253
    assert polyval(p,4,derivative=True) == (253, 190)

def test_polyroots():
    p = polyroots([1,-4])
    assert p[0].ae(4)
    p, q = polyroots([1,2,3])
    assert p.ae(-1 - sqrt(2)*j)
    assert q.ae(-1 + sqrt(2)*j)
    #this is not a real test, it only tests a specific case
    assert polyroots([1]) == []
    pytest.raises(ValueError, lambda: polyroots([0]))

def test_polyroots_legendre():
    n = 64
    coeffs = [11975573020964041433067793888190275875, 0,
        -190100434726484311252477736051902332000, 0,
        1437919688271127330313741595496589239248, 0,
        -6897338342113537600691931230430793911840, 0,
        23556405536185284408974715545252277554280, 0,
        -60969520211303089058522793175947071316960, 0,
        124284021969194758465450309166353645376880, 0,
        -204721258548015217049921875719981284186016, 0,
        277415422258095841688223780704620656114900, 0,
        -313237834141273382807123548182995095192800, 0,
        297432255354328395601259515935229287637200, 0,
        -239057700565161140389797367947941296605600, 0,
        163356095386193445933028201431093219347160, 0,
        -95158890516229191805647495979277603503200, 0,
        47310254620162038075933656063247634556400, 0,
        -20071017111583894941305187420771723751200, 0,
        7255051932731034189479516844750603752850, 0,
        -2228176940331017311443863996901733412640, 0,
        579006552594977616773047095969088431600, 0,
        -126584428502545713788439446082310831200, 0,
        23112325428835593809686977515028663000, 0,
        -3491517141958743235617737161547844000, 0,
        431305058712550634988073414073557200, 0,
        -42927166660756742088912492757452000, 0,
        3378527005707706553294038781836500, 0,
        -205277590220215081719131470288800, 0,
        9330799555464321896324157740400, 0,
        -304114948474392713657972548576, 0,
        6695289961520387531608984680, 0,
        -91048139350447232095702560, 0,
        659769125727878493447120, 0,
        -1905929106580294155360, 0,
        916312070471295267]

    with mp.workdps(3):
        with pytest.raises(mp.NoConvergence):
            polyroots(coeffs, maxsteps=5, cleanup=True, error=False,
                      extraprec=n*10)

        roots = polyroots(coeffs, maxsteps=50, cleanup=True, error=False,
                    extraprec=n*10)
        roots = [str(r) for r in roots]
        assert roots == \
            ['-0.999', '-0.996', '-0.991', '-0.983', '-0.973', '-0.961',
            '-0.946', '-0.93', '-0.911', '-0.889', '-0.866', '-0.841',
            '-0.813', '-0.784', '-0.753', '-0.72', '-0.685', '-0.649',
            '-0.611', '-0.572', '-0.531', '-0.489', '-0.446', '-0.402',
            '-0.357', '-0.311', '-0.265', '-0.217', '-0.17', '-0.121',
            '-0.073', '-0.0243', '0.0243', '0.073', '0.121', '0.17', '0.217',
            '0.265', '0.311', '0.357', '0.402', '0.446', '0.489', '0.531',
            '0.572', '0.611', '0.649', '0.685', '0.72', '0.753', '0.784',
            '0.813', '0.841', '0.866', '0.889', '0.911', '0.93', '0.946',
            '0.961', '0.973', '0.983', '0.991', '0.996', '0.999']

def test_polyroots_legendre_init():
    extra_prec = 100
    coeffs = [11975573020964041433067793888190275875, 0,
        -190100434726484311252477736051902332000, 0,
        1437919688271127330313741595496589239248, 0,
        -6897338342113537600691931230430793911840, 0,
        23556405536185284408974715545252277554280, 0,
        -60969520211303089058522793175947071316960, 0,
        124284021969194758465450309166353645376880, 0,
        -204721258548015217049921875719981284186016, 0,
        277415422258095841688223780704620656114900, 0,
        -313237834141273382807123548182995095192800, 0,
        297432255354328395601259515935229287637200, 0,
        -239057700565161140389797367947941296605600, 0,
        163356095386193445933028201431093219347160, 0,
        -95158890516229191805647495979277603503200, 0,
        47310254620162038075933656063247634556400, 0,
        -20071017111583894941305187420771723751200, 0,
        7255051932731034189479516844750603752850, 0,
        -2228176940331017311443863996901733412640, 0,
        579006552594977616773047095969088431600, 0,
        -126584428502545713788439446082310831200, 0,
        23112325428835593809686977515028663000, 0,
        -3491517141958743235617737161547844000, 0,
        431305058712550634988073414073557200, 0,
        -42927166660756742088912492757452000, 0,
        3378527005707706553294038781836500, 0,
        -205277590220215081719131470288800, 0,
        9330799555464321896324157740400, 0,
        -304114948474392713657972548576, 0,
        6695289961520387531608984680, 0,
        -91048139350447232095702560, 0,
        659769125727878493447120, 0,
        -1905929106580294155360, 0,
        916312070471295267]

    roots_init =  matrix(['-0.999', '-0.996',  '-0.991', '-0.983', '-0.973',
                          '-0.961', '-0.946',  '-0.93',  '-0.911', '-0.889',
                          '-0.866', '-0.841',  '-0.813', '-0.784', '-0.753',
                          '-0.72',  '-0.685',  '-0.649', '-0.611', '-0.572',
                          '-0.531', '-0.489',  '-0.446', '-0.402', '-0.357',
                          '-0.311', '-0.265',  '-0.217', '-0.17',  '-0.121',
                          '-0.073', '-0.0243',  '0.0243', '0.073',  '0.121',
                          '0.17',    '0.217',   '0.265', ' 0.311',  '0.357',
                          '0.402',   '0.446',   '0.489',  '0.531',  '0.572',
                          '0.611',   '0.649',   '0.685',  '0.72',   '0.753',
                          '0.784',   '0.813',   '0.841',  '0.866',  '0.889',
                          '0.911',   '0.93',    '0.946',  '0.961',  '0.973',
                          '0.983',   '0.991',   '0.996',  '0.999',  '1.0'])
    with mp.workdps(2*mp.dps):
        roots_exact = polyroots(coeffs, maxsteps=50, cleanup=True, error=False,
                                extraprec=2*extra_prec)
    with pytest.raises(mp.NoConvergence):
        polyroots(coeffs, maxsteps=5, cleanup=True, error=False,
                  extraprec=extra_prec)
    roots,err = polyroots(coeffs, maxsteps=5, cleanup=True, error=True,
                          extraprec=extra_prec,roots_init=roots_init)
    assert max(matrix(roots_exact)-matrix(roots).apply(abs)) < err
    roots1,err1 = polyroots(coeffs, maxsteps=25, cleanup=True, error=True,
                            extraprec=extra_prec,roots_init=roots_init[:60])
    assert max(matrix(roots_exact)-matrix(roots1).apply(abs)) < err1

def test_pade():
    one = mpf(1)
    mp.dps = 20
    N = 10
    a = [one]
    k = 1
    for i in range(1, N+1):
        k *= i
        a.append(one/k)
    p, q = pade(a, N//2, N//2)
    for x in arange(0, 1, 0.1):
        r = polyval(p[::-1], x)/polyval(q[::-1], x)
        assert(r.ae(exp(x), 1.0e-10))
    mp.dps = 15

def test_fourier():
    mp.dps = 15
    c, s = fourier(lambda x: x+1, [-1, 2], 2)
    #plot([lambda x: x+1, lambda x: fourierval((c, s), [-1, 2], x)], [-1, 2])
    assert c[0].ae(1.5)
    assert c[1].ae(-3*sqrt(3)/(2*pi))
    assert c[2].ae(3*sqrt(3)/(4*pi))
    assert s[0] == 0
    assert s[1].ae(3/(2*pi))
    assert s[2].ae(3/(4*pi))
    assert fourierval((c, s), [-1, 2], 1).ae(1.9134966715663442)

def test_differint():
    mp.dps = 15
    assert differint(lambda t: t, 2, -0.5).ae(8*sqrt(2/pi)/3)

def test_invlap():
    mp.dps = 15
    t = 0.01
    fp = lambda p: 1/(p+1)**2
    ft = lambda t: t*exp(-t)
    ftt = ft(t)
    assert invertlaplace(fp,t,method='talbot').ae(ftt)
    assert invertlaplace(fp,t,method='stehfest').ae(ftt)
    assert invertlaplace(fp,t,method='dehoog').ae(ftt)
    assert invertlaplace(fp,t,method='cohen').ae(ftt)
    t = 1.0
    ftt = ft(t)
    assert invertlaplace(fp,t,method='talbot').ae(ftt)
    assert invertlaplace(fp,t,method='stehfest').ae(ftt)
    assert invertlaplace(fp,t,method='dehoog').ae(ftt)
    assert invertlaplace(fp,t,method='cohen').ae(ftt)

    t = 0.01
    fp = lambda p: log(p)/p
    ft = lambda t: -euler-log(t)
    ftt = ft(t)
    assert invertlaplace(fp,t,method='talbot').ae(ftt)
    assert invertlaplace(fp,t,method='stehfest').ae(ftt)
    assert invertlaplace(fp,t,method='dehoog').ae(ftt)
    assert invertlaplace(fp,t,method='cohen').ae(ftt)
    t = 1.0
    ftt = ft(t)
    assert invertlaplace(fp,t,method='talbot').ae(ftt)
    assert invertlaplace(fp,t,method='stehfest').ae(ftt)
    assert invertlaplace(fp,t,method='dehoog').ae(ftt)
    assert invertlaplace(fp,t,method='cohen').ae(ftt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_missing.py ===
import collections

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
    isna,
)
import pandas._testing as tm


class TestCategoricalMissing:
    def test_isna(self):
        exp = np.array([False, False, True])
        cat = Categorical(["a", "b", np.nan])
        res = cat.isna()

        tm.assert_numpy_array_equal(res, exp)

    def test_na_flags_int_categories(self):
        # #1457

        categories = list(range(10))
        labels = np.random.default_rng(2).integers(0, 10, 20)
        labels[::5] = -1

        cat = Categorical(labels, categories)
        repr(cat)

        tm.assert_numpy_array_equal(isna(cat), labels == -1)

    def test_nan_handling(self):
        # Nans are represented as -1 in codes
        c = Categorical(["a", "b", np.nan, "a"])
        tm.assert_index_equal(c.categories, Index(["a", "b"]))
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, -1, 0], dtype=np.int8))
        c[1] = np.nan
        tm.assert_index_equal(c.categories, Index(["a", "b"]))
        tm.assert_numpy_array_equal(c._codes, np.array([0, -1, -1, 0], dtype=np.int8))

        # Adding nan to categories should make assigned nan point to the
        # category!
        c = Categorical(["a", "b", np.nan, "a"])
        tm.assert_index_equal(c.categories, Index(["a", "b"]))
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, -1, 0], dtype=np.int8))

    def test_set_dtype_nans(self):
        c = Categorical(["a", "b", np.nan])
        result = c._set_dtype(CategoricalDtype(["a", "c"]))
        tm.assert_numpy_array_equal(result.codes, np.array([0, -1, -1], dtype="int8"))

    def test_set_item_nan(self):
        cat = Categorical([1, 2, 3])
        cat[1] = np.nan

        exp = Categorical([1, np.nan, 3], categories=[1, 2, 3])
        tm.assert_categorical_equal(cat, exp)

    @pytest.mark.parametrize(
        "fillna_kwargs, msg",
        [
            (
                {"value": 1, "method": "ffill"},
                "Cannot specify both 'value' and 'method'.",
            ),
            ({}, "Must specify a fill 'value' or 'method'."),
            ({"method": "bad"}, "Invalid fill method. Expecting .* bad"),
            (
                {"value": Series([1, 2, 3, 4, "a"])},
                "Cannot setitem on a Categorical with a new category",
            ),
        ],
    )
    def test_fillna_raises(self, fillna_kwargs, msg):
        # https://github.com/pandas-dev/pandas/issues/19682
        # https://github.com/pandas-dev/pandas/issues/13628
        cat = Categorical([1, 2, 3, None, None])

        if len(fillna_kwargs) == 1 and "value" in fillna_kwargs:
            err = TypeError
        else:
            err = ValueError

        with pytest.raises(err, match=msg):
            cat.fillna(**fillna_kwargs)

    @pytest.mark.parametrize("named", [True, False])
    def test_fillna_iterable_category(self, named):
        # https://github.com/pandas-dev/pandas/issues/21097
        if named:
            Point = collections.namedtuple("Point", "x y")
        else:
            Point = lambda *args: args  # tuple
        cat = Categorical(np.array([Point(0, 0), Point(0, 1), None], dtype=object))
        result = cat.fillna(Point(0, 0))
        expected = Categorical([Point(0, 0), Point(0, 1), Point(0, 0)])

        tm.assert_categorical_equal(result, expected)

        # Case where the Point is not among our categories; we want ValueError,
        #  not NotImplementedError GH#41914
        cat = Categorical(np.array([Point(1, 0), Point(0, 1), None], dtype=object))
        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(TypeError, match=msg):
            cat.fillna(Point(0, 0))

    def test_fillna_array(self):
        # accept Categorical or ndarray value if it holds appropriate values
        cat = Categorical(["A", "B", "C", None, None])

        other = cat.fillna("C")
        result = cat.fillna(other)
        tm.assert_categorical_equal(result, other)
        assert isna(cat[-1])  # didn't modify original inplace

        other = np.array(["A", "B", "C", "B", "A"])
        result = cat.fillna(other)
        expected = Categorical(["A", "B", "C", "B", "A"], dtype=cat.dtype)
        tm.assert_categorical_equal(result, expected)
        assert isna(cat[-1])  # didn't modify original inplace

    @pytest.mark.parametrize(
        "values, expected",
        [
            ([1, 2, 3], np.array([False, False, False])),
            ([1, 2, np.nan], np.array([False, False, True])),
            ([1, 2, np.inf], np.array([False, False, True])),
            ([1, 2, pd.NA], np.array([False, False, True])),
        ],
    )
    def test_use_inf_as_na(self, values, expected):
        # https://github.com/pandas-dev/pandas/issues/33594
        msg = "use_inf_as_na option is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context("mode.use_inf_as_na", True):
                cat = Categorical(values)
                result = cat.isna()
                tm.assert_numpy_array_equal(result, expected)

                result = Series(cat).isna()
                expected = Series(expected)
                tm.assert_series_equal(result, expected)

                result = DataFrame(cat).isna()
                expected = DataFrame(expected)
                tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "values, expected",
        [
            ([1, 2, 3], np.array([False, False, False])),
            ([1, 2, np.nan], np.array([False, False, True])),
            ([1, 2, np.inf], np.array([False, False, True])),
            ([1, 2, pd.NA], np.array([False, False, True])),
        ],
    )
    def test_use_inf_as_na_outside_context(self, values, expected):
        # https://github.com/pandas-dev/pandas/issues/33594
        # Using isna directly for Categorical will fail in general here
        cat = Categorical(values)

        msg = "use_inf_as_na option is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context("mode.use_inf_as_na", True):
                result = isna(cat)
                tm.assert_numpy_array_equal(result, expected)

                result = isna(Series(cat))
                expected = Series(expected)
                tm.assert_series_equal(result, expected)

                result = isna(DataFrame(cat))
                expected = DataFrame(expected)
                tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "a1, a2, categories",
        [
            (["a", "b", "c"], [np.nan, "a", "b"], ["a", "b", "c"]),
            ([1, 2, 3], [np.nan, 1, 2], [1, 2, 3]),
        ],
    )
    def test_compare_categorical_with_missing(self, a1, a2, categories):
        # GH 28384
        cat_type = CategoricalDtype(categories)

        # !=
        result = Series(a1, dtype=cat_type) != Series(a2, dtype=cat_type)
        expected = Series(a1) != Series(a2)
        tm.assert_series_equal(result, expected)

        # ==
        result = Series(a1, dtype=cat_type) == Series(a2, dtype=cat_type)
        expected = Series(a1) == Series(a2)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "na_value, dtype",
        [
            (pd.NaT, "datetime64[ns]"),
            (None, "float64"),
            (np.nan, "float64"),
            (pd.NA, "float64"),
        ],
    )
    def test_categorical_only_missing_values_no_cast(self, na_value, dtype):
        # GH#44900
        result = Categorical([na_value, na_value])
        tm.assert_index_equal(result.categories, Index([], dtype=dtype))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_infer_dtype.py ===
from datetime import (
    date,
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.core.dtypes.cast import (
    infer_dtype_from,
    infer_dtype_from_array,
    infer_dtype_from_scalar,
)
from pandas.core.dtypes.common import is_dtype_equal

from pandas import (
    Categorical,
    Interval,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)


def test_infer_dtype_from_int_scalar(any_int_numpy_dtype):
    # Test that infer_dtype_from_scalar is
    # returning correct dtype for int and float.
    data = np.dtype(any_int_numpy_dtype).type(12)
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == type(data)


def test_infer_dtype_from_float_scalar(float_numpy_dtype):
    float_numpy_dtype = np.dtype(float_numpy_dtype).type
    data = float_numpy_dtype(12)

    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == float_numpy_dtype


@pytest.mark.parametrize(
    "data,exp_dtype", [(12, np.int64), (np.float64(12), np.float64)]
)
def test_infer_dtype_from_python_scalar(data, exp_dtype):
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == exp_dtype


@pytest.mark.parametrize("bool_val", [True, False])
def test_infer_dtype_from_boolean(bool_val):
    dtype, val = infer_dtype_from_scalar(bool_val)
    assert dtype == np.bool_


def test_infer_dtype_from_complex(complex_dtype):
    data = np.dtype(complex_dtype).type(1)
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == np.complex128


def test_infer_dtype_from_datetime():
    dt64 = np.datetime64(1, "ns")
    dtype, val = infer_dtype_from_scalar(dt64)
    assert dtype == "M8[ns]"

    ts = Timestamp(1)
    dtype, val = infer_dtype_from_scalar(ts)
    assert dtype == "M8[ns]"

    dt = datetime(2000, 1, 1, 0, 0)
    dtype, val = infer_dtype_from_scalar(dt)
    assert dtype == "M8[us]"


def test_infer_dtype_from_timedelta():
    td64 = np.timedelta64(1, "ns")
    dtype, val = infer_dtype_from_scalar(td64)
    assert dtype == "m8[ns]"

    pytd = timedelta(1)
    dtype, val = infer_dtype_from_scalar(pytd)
    assert dtype == "m8[us]"

    td = Timedelta(1)
    dtype, val = infer_dtype_from_scalar(td)
    assert dtype == "m8[ns]"


@pytest.mark.parametrize("freq", ["M", "D"])
def test_infer_dtype_from_period(freq):
    p = Period("2011-01-01", freq=freq)
    dtype, val = infer_dtype_from_scalar(p)

    exp_dtype = f"period[{freq}]"

    assert dtype == exp_dtype
    assert val == p


def test_infer_dtype_misc():
    dt = date(2000, 1, 1)
    dtype, val = infer_dtype_from_scalar(dt)
    assert dtype == np.object_

    ts = Timestamp(1, tz="US/Eastern")
    dtype, val = infer_dtype_from_scalar(ts)
    assert dtype == "datetime64[ns, US/Eastern]"


@pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Asia/Tokyo"])
def test_infer_from_scalar_tz(tz):
    dt = Timestamp(1, tz=tz)
    dtype, val = infer_dtype_from_scalar(dt)

    exp_dtype = f"datetime64[ns, {tz}]"

    assert dtype == exp_dtype
    assert val == dt


@pytest.mark.parametrize(
    "left, right, subtype",
    [
        (0, 1, "int64"),
        (0.0, 1.0, "float64"),
        (Timestamp(0), Timestamp(1), "datetime64[ns]"),
        (Timestamp(0, tz="UTC"), Timestamp(1, tz="UTC"), "datetime64[ns, UTC]"),
        (Timedelta(0), Timedelta(1), "timedelta64[ns]"),
    ],
)
def test_infer_from_interval(left, right, subtype, closed):
    # GH 30337
    interval = Interval(left, right, closed)
    result_dtype, result_value = infer_dtype_from_scalar(interval)
    expected_dtype = f"interval[{subtype}, {closed}]"
    assert result_dtype == expected_dtype
    assert result_value == interval


def test_infer_dtype_from_scalar_errors():
    msg = "invalid ndarray passed to infer_dtype_from_scalar"

    with pytest.raises(ValueError, match=msg):
        infer_dtype_from_scalar(np.array([1]))


@pytest.mark.parametrize(
    "value, expected",
    [
        ("foo", np.object_),
        (b"foo", np.object_),
        (1, np.int64),
        (1.5, np.float64),
        (np.datetime64("2016-01-01"), np.dtype("M8[s]")),
        (Timestamp("20160101"), np.dtype("M8[s]")),
        (Timestamp("20160101", tz="UTC"), "datetime64[s, UTC]"),
    ],
)
def test_infer_dtype_from_scalar(value, expected, using_infer_string):
    dtype, _ = infer_dtype_from_scalar(value)
    if using_infer_string and value == "foo":
        expected = "string"
    assert is_dtype_equal(dtype, expected)

    with pytest.raises(TypeError, match="must be list-like"):
        infer_dtype_from_array(value)


@pytest.mark.parametrize(
    "arr, expected",
    [
        ([1], np.dtype(int)),
        (np.array([1], dtype=np.int64), np.int64),
        ([np.nan, 1, ""], np.object_),
        (np.array([[1.0, 2.0]]), np.float64),
        (Categorical(list("aabc")), "category"),
        (Categorical([1, 2, 3]), "category"),
        (date_range("20160101", periods=3), np.dtype("=M8[ns]")),
        (
            date_range("20160101", periods=3, tz="US/Eastern"),
            "datetime64[ns, US/Eastern]",
        ),
        (Series([1.0, 2, 3]), np.float64),
        (Series(list("abc")), np.object_),
        (
            Series(date_range("20160101", periods=3, tz="US/Eastern")),
            "datetime64[ns, US/Eastern]",
        ),
    ],
)
def test_infer_dtype_from_array(arr, expected, using_infer_string):
    dtype, _ = infer_dtype_from_array(arr)
    if (
        using_infer_string
        and isinstance(arr, Series)
        and arr.tolist() == ["a", "b", "c"]
    ):
        expected = "string"
    assert is_dtype_equal(dtype, expected)


@pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
def test_infer_dtype_from_scalar_zerodim_datetimelike(cls):
    # ndarray.item() can incorrectly return int instead of td64/dt64
    val = cls(1234, "ns")
    arr = np.array(val)

    dtype, res = infer_dtype_from_scalar(arr)
    assert dtype.type is cls
    assert isinstance(res, cls)

    dtype, res = infer_dtype_from(arr)
    assert dtype.type is cls

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_map.py ===
from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import BDay


def test_map(float_frame):
    result = float_frame.map(lambda x: x * 2)
    tm.assert_frame_equal(result, float_frame * 2)
    float_frame.map(type)

    # GH 465: function returning tuples
    result = float_frame.map(lambda x: (x, x))["A"].iloc[0]
    assert isinstance(result, tuple)


@pytest.mark.parametrize("val", [1, 1.0])
def test_map_float_object_conversion(val):
    # GH 2909: object conversion to float in constructor?
    df = DataFrame(data=[val, "a"])
    result = df.map(lambda x: x).dtypes[0]
    assert result == object


@pytest.mark.parametrize("na_action", [None, "ignore"])
def test_map_keeps_dtype(na_action):
    # GH52219
    arr = Series(["a", np.nan, "b"])
    sparse_arr = arr.astype(pd.SparseDtype(object))
    df = DataFrame(data={"a": arr, "b": sparse_arr})

    def func(x):
        return str.upper(x) if not pd.isna(x) else x

    result = df.map(func, na_action=na_action)

    expected_sparse = pd.array(["A", np.nan, "B"], dtype=pd.SparseDtype(object))
    expected_arr = expected_sparse.astype(object)
    expected = DataFrame({"a": expected_arr, "b": expected_sparse})

    tm.assert_frame_equal(result, expected)

    result_empty = df.iloc[:0, :].map(func, na_action=na_action)
    expected_empty = expected.iloc[:0, :]
    tm.assert_frame_equal(result_empty, expected_empty)


def test_map_str():
    # GH 2786
    df = DataFrame(np.random.default_rng(2).random((3, 4)))
    df2 = df.copy()
    cols = ["a", "a", "a", "a"]
    df.columns = cols

    expected = df2.map(str)
    expected.columns = cols
    result = df.map(str)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "col, val",
    [["datetime", Timestamp("20130101")], ["timedelta", pd.Timedelta("1 min")]],
)
def test_map_datetimelike(col, val):
    # datetime/timedelta
    df = DataFrame(np.random.default_rng(2).random((3, 4)))
    df[col] = val
    result = df.map(str)
    assert result.loc[0, col] == str(df.loc[0, col])


@pytest.mark.parametrize(
    "expected",
    [
        DataFrame(),
        DataFrame(columns=list("ABC")),
        DataFrame(index=list("ABC")),
        DataFrame({"A": [], "B": [], "C": []}),
    ],
)
@pytest.mark.parametrize("func", [round, lambda x: x])
def test_map_empty(expected, func):
    # GH 8222
    result = expected.map(func)
    tm.assert_frame_equal(result, expected)


def test_map_kwargs():
    # GH 40652
    result = DataFrame([[1, 2], [3, 4]]).map(lambda x, y: x + y, y=2)
    expected = DataFrame([[3, 4], [5, 6]])
    tm.assert_frame_equal(result, expected)


def test_map_na_ignore(float_frame):
    # GH 23803
    strlen_frame = float_frame.map(lambda x: len(str(x)))
    float_frame_with_na = float_frame.copy()
    mask = np.random.default_rng(2).integers(0, 2, size=float_frame.shape, dtype=bool)
    float_frame_with_na[mask] = pd.NA
    strlen_frame_na_ignore = float_frame_with_na.map(
        lambda x: len(str(x)), na_action="ignore"
    )
    # Set float64 type to avoid upcast when setting NA below
    strlen_frame_with_na = strlen_frame.copy().astype("float64")
    strlen_frame_with_na[mask] = pd.NA
    tm.assert_frame_equal(strlen_frame_na_ignore, strlen_frame_with_na)


def test_map_box_timestamps():
    # GH 2689, GH 2627
    ser = Series(date_range("1/1/2000", periods=10))

    def func(x):
        return (x.hour, x.day, x.month)

    # it works!
    DataFrame(ser).map(func)


def test_map_box():
    # ufunc will not be boxed. Same test cases as the test_map_box
    df = DataFrame(
        {
            "a": [Timestamp("2011-01-01"), Timestamp("2011-01-02")],
            "b": [
                Timestamp("2011-01-01", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
            ],
            "c": [pd.Timedelta("1 days"), pd.Timedelta("2 days")],
            "d": [
                pd.Period("2011-01-01", freq="M"),
                pd.Period("2011-01-02", freq="M"),
            ],
        }
    )

    result = df.map(lambda x: type(x).__name__)
    expected = DataFrame(
        {
            "a": ["Timestamp", "Timestamp"],
            "b": ["Timestamp", "Timestamp"],
            "c": ["Timedelta", "Timedelta"],
            "d": ["Period", "Period"],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_frame_map_dont_convert_datetime64():
    df = DataFrame({"x1": [datetime(1996, 1, 1)]})

    df = df.map(lambda x: x + BDay())
    df = df.map(lambda x: x + BDay())

    result = df.x1.dtype
    assert result == "M8[ns]"


def test_map_function_runs_once():
    df = DataFrame({"a": [1, 2, 3]})
    values = []  # Save values function is applied to

    def reducing_function(val):
        values.append(val)

    def non_reducing_function(val):
        values.append(val)
        return val

    for func in [reducing_function, non_reducing_function]:
        del values[:]

        df.map(func)
        assert values == df.a.to_list()


def test_map_type():
    # GH 46719
    df = DataFrame(
        {"col1": [3, "string", float], "col2": [0.25, datetime(2020, 1, 1), np.nan]},
        index=["a", "b", "c"],
    )

    result = df.map(type)
    expected = DataFrame(
        {"col1": [int, str, type], "col2": [float, datetime, float]},
        index=["a", "b", "c"],
    )
    tm.assert_frame_equal(result, expected)


def test_map_invalid_na_action(float_frame):
    # GH 23803
    with pytest.raises(ValueError, match="na_action must be .*Got 'abc'"):
        float_frame.map(lambda x: len(str(x)), na_action="abc")


def test_applymap_deprecated():
    # GH52353
    df = DataFrame({"a": [1, 2, 3]})
    msg = "DataFrame.applymap has been deprecated. Use DataFrame.map instead."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.applymap(lambda x: x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_datetime.py ===
import datetime as dt
from datetime import date
import re

import numpy as np
import pytest

from pandas.compat.numpy import np_long

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm


class TestDatetimeIndex:
    def test_is_(self):
        dti = date_range(start="1/1/2005", end="12/1/2005", freq="ME")
        assert dti.is_(dti)
        assert dti.is_(dti.view())
        assert not dti.is_(dti.copy())

    def test_time_overflow_for_32bit_machines(self):
        # GH8943.  On some machines NumPy defaults to np.int32 (for example,
        # 32-bit Linux machines).  In the function _generate_regular_range
        # found in tseries/index.py, `periods` gets multiplied by `strides`
        # (which has value 1e9) and since the max value for np.int32 is ~2e9,
        # and since those machines won't promote np.int32 to np.int64, we get
        # overflow.
        periods = np_long(1000)

        idx1 = date_range(start="2000", periods=periods, freq="s")
        assert len(idx1) == periods

        idx2 = date_range(end="2000", periods=periods, freq="s")
        assert len(idx2) == periods

    def test_nat(self):
        assert DatetimeIndex([np.nan])[0] is pd.NaT

    def test_week_of_month_frequency(self):
        # GH 5348: "ValueError: Could not evaluate WOM-1SUN" shouldn't raise
        d1 = date(2002, 9, 1)
        d2 = date(2013, 10, 27)
        d3 = date(2012, 9, 30)
        idx1 = DatetimeIndex([d1, d2])
        idx2 = DatetimeIndex([d3])
        result_append = idx1.append(idx2)
        expected = DatetimeIndex([d1, d2, d3])
        tm.assert_index_equal(result_append, expected)
        result_union = idx1.union(idx2)
        expected = DatetimeIndex([d1, d3, d2])
        tm.assert_index_equal(result_union, expected)

    def test_append_nondatetimeindex(self):
        rng = date_range("1/1/2000", periods=10)
        idx = Index(["a", "b", "c", "d"])

        result = rng.append(idx)
        assert isinstance(result[0], Timestamp)

    def test_misc_coverage(self):
        rng = date_range("1/1/2000", periods=5)
        result = rng.groupby(rng.day)
        assert isinstance(next(iter(result.values()))[0], Timestamp)

    # TODO: belongs in frame groupby tests?
    def test_groupby_function_tuple_1677(self):
        df = DataFrame(
            np.random.default_rng(2).random(100),
            index=date_range("1/1/2000", periods=100),
        )
        monthly_group = df.groupby(lambda x: (x.year, x.month))

        result = monthly_group.mean()
        assert isinstance(result.index[0], tuple)

    def assert_index_parameters(self, index):
        assert index.freq == "40960ns"
        assert index.inferred_freq == "40960ns"

    def test_ns_index(self):
        nsamples = 400
        ns = int(1e9 / 24414)
        dtstart = np.datetime64("2012-09-20T00:00:00")

        dt = dtstart + np.arange(nsamples) * np.timedelta64(ns, "ns")
        freq = ns * offsets.Nano()
        index = DatetimeIndex(dt, freq=freq, name="time")
        self.assert_index_parameters(index)

        new_index = date_range(start=index[0], end=index[-1], freq=index.freq)
        self.assert_index_parameters(new_index)

    def test_asarray_tz_naive(self):
        # This shouldn't produce a warning.
        idx = date_range("2000", periods=2)
        # M8[ns] by default
        result = np.asarray(idx)

        expected = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        tm.assert_numpy_array_equal(result, expected)

        # optionally, object
        result = np.asarray(idx, dtype=object)

        expected = np.array([Timestamp("2000-01-01"), Timestamp("2000-01-02")])
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_aware(self):
        tz = "US/Central"
        idx = date_range("2000", periods=2, tz=tz)
        expected = np.array(["2000-01-01T06", "2000-01-02T06"], dtype="M8[ns]")
        result = np.asarray(idx, dtype="datetime64[ns]")

        tm.assert_numpy_array_equal(result, expected)

        # Old behavior with no warning
        result = np.asarray(idx, dtype="M8[ns]")

        tm.assert_numpy_array_equal(result, expected)

        # Future behavior with no warning
        expected = np.array(
            [Timestamp("2000-01-01", tz=tz), Timestamp("2000-01-02", tz=tz)]
        )
        result = np.asarray(idx, dtype=object)

        tm.assert_numpy_array_equal(result, expected)

    def test_CBH_deprecated(self):
        msg = "'CBH' is deprecated and will be removed in a future version."

        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = date_range(
                dt.datetime(2022, 12, 11), dt.datetime(2022, 12, 13), freq="CBH"
            )
        result = DatetimeIndex(
            [
                "2022-12-12 09:00:00",
                "2022-12-12 10:00:00",
                "2022-12-12 11:00:00",
                "2022-12-12 12:00:00",
                "2022-12-12 13:00:00",
                "2022-12-12 14:00:00",
                "2022-12-12 15:00:00",
                "2022-12-12 16:00:00",
            ],
            dtype="datetime64[ns]",
            freq="cbh",
        )

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "freq_depr, expected_values, expected_freq",
        [
            (
                "AS-AUG",
                ["2021-08-01", "2022-08-01", "2023-08-01"],
                "YS-AUG",
            ),
            (
                "1BAS-MAY",
                ["2021-05-03", "2022-05-02", "2023-05-01"],
                "1BYS-MAY",
            ),
        ],
    )
    def test_AS_BAS_deprecated(self, freq_depr, expected_values, expected_freq):
        # GH#55479
        freq_msg = re.split("[0-9]*", freq_depr, maxsplit=1)[1]
        msg = f"'{freq_msg}' is deprecated and will be removed in a future version."

        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = date_range(
                dt.datetime(2020, 12, 1), dt.datetime(2023, 12, 1), freq=freq_depr
            )
        result = DatetimeIndex(
            expected_values,
            dtype="datetime64[ns]",
            freq=expected_freq,
        )

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, expected_values, freq_depr",
        [
            ("2BYE-MAR", ["2016-03-31"], "2BA-MAR"),
            ("2BYE-JUN", ["2016-06-30"], "2BY-JUN"),
            ("2BME", ["2016-02-29", "2016-04-29", "2016-06-30"], "2BM"),
            ("2BQE", ["2016-03-31"], "2BQ"),
            ("1BQE-MAR", ["2016-03-31", "2016-06-30"], "1BQ-MAR"),
        ],
    )
    def test_BM_BQ_BY_deprecated(self, freq, expected_values, freq_depr):
        # GH#52064
        msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
        f"in a future version, please use '{freq[1:]}' instead."

        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = date_range(start="2016-02-21", end="2016-08-21", freq=freq_depr)
        result = DatetimeIndex(
            data=expected_values,
            dtype="datetime64[ns]",
            freq=freq,
        )

        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\tests\diagnose_imports.py ===
#!/usr/bin/env python

"""
Import diagnostics. Run bin/diagnose_imports.py --help for details.
"""

from __future__ import annotations

if __name__ == "__main__":

    import sys
    import inspect
    import builtins

    import optparse

    from os.path import abspath, dirname, join, normpath
    this_file = abspath(__file__)
    sympy_dir = join(dirname(this_file), '..', '..', '..')
    sympy_dir = normpath(sympy_dir)
    sys.path.insert(0, sympy_dir)

    option_parser = optparse.OptionParser(
        usage=
            "Usage: %prog option [options]\n"
            "\n"
            "Import analysis for imports between SymPy modules.")
    option_group = optparse.OptionGroup(
        option_parser,
        'Analysis options',
        'Options that define what to do. Exactly one of these must be given.')
    option_group.add_option(
        '--problems',
        help=
            'Print all import problems, that is: '
            'If an import pulls in a package instead of a module '
            '(e.g. sympy.core instead of sympy.core.add); ' # see ##PACKAGE##
            'if it imports a symbol that is already present; ' # see ##DUPLICATE##
            'if it imports a symbol '
            'from somewhere other than the defining module.', # see ##ORIGIN##
        action='count')
    option_group.add_option(
        '--origins',
        help=
            'For each imported symbol in each module, '
            'print the module that defined it. '
            '(This is useful for import refactoring.)',
        action='count')
    option_parser.add_option_group(option_group)
    option_group = optparse.OptionGroup(
        option_parser,
        'Sort options',
        'These options define the sort order for output lines. '
        'At most one of these options is allowed. '
        'Unsorted output will reflect the order in which imports happened.')
    option_group.add_option(
        '--by-importer',
        help='Sort output lines by name of importing module.',
        action='count')
    option_group.add_option(
        '--by-origin',
        help='Sort output lines by name of imported module.',
        action='count')
    option_parser.add_option_group(option_group)
    (options, args) = option_parser.parse_args()
    if args:
        option_parser.error(
            'Unexpected arguments %s (try %s --help)' % (args, sys.argv[0]))
    if options.problems > 1:
        option_parser.error('--problems must not be given more than once.')
    if options.origins > 1:
        option_parser.error('--origins must not be given more than once.')
    if options.by_importer > 1:
        option_parser.error('--by-importer must not be given more than once.')
    if options.by_origin > 1:
        option_parser.error('--by-origin must not be given more than once.')
    options.problems = options.problems == 1
    options.origins = options.origins == 1
    options.by_importer = options.by_importer == 1
    options.by_origin = options.by_origin == 1
    if not options.problems and not options.origins:
        option_parser.error(
            'At least one of --problems and --origins is required')
    if options.problems and options.origins:
        option_parser.error(
            'At most one of --problems and --origins is allowed')
    if options.by_importer and options.by_origin:
        option_parser.error(
            'At most one of --by-importer and --by-origin is allowed')
    options.by_process = not options.by_importer and not options.by_origin

    builtin_import = builtins.__import__

    class Definition:
        """Information about a symbol's definition."""
        def __init__(self, name, value, definer):
            self.name = name
            self.value = value
            self.definer = definer
        def __hash__(self):
            return hash(self.name)
        def __eq__(self, other):
            return self.name == other.name and self.value == other.value
        def __ne__(self, other):
            return not (self == other)
        def __repr__(self):
            return 'Definition(%s, ..., %s)' % (
                repr(self.name), repr(self.definer))

    # Maps each function/variable to name of module to define it
    symbol_definers: dict[Definition, str] = {}

    def in_module(a, b):
        """Is a the same module as or a submodule of b?"""
        return a == b or a != None and b != None and a.startswith(b + '.')

    def relevant(module):
        """Is module relevant for import checking?

        Only imports between relevant modules will be checked."""
        return in_module(module, 'sympy')

    sorted_messages = []

    def msg(msg, *args):
        if options.by_process:
            print(msg % args)
        else:
            sorted_messages.append(msg % args)

    def tracking_import(module, globals=globals(), locals=[], fromlist=None, level=-1):
        """__import__ wrapper - does not change imports at all, but tracks them.

        Default order is implemented by doing output directly.
        All other orders are implemented by collecting output information into
        a sorted list that will be emitted after all imports are processed.

        Indirect imports can only occur after the requested symbol has been
        imported directly (because the indirect import would not have a module
        to pick the symbol up from).
        So this code detects indirect imports by checking whether the symbol in
        question was already imported.

        Keeps the semantics of __import__ unchanged."""
        caller_frame = inspect.getframeinfo(sys._getframe(1))
        importer_filename = caller_frame.filename
        importer_module = globals['__name__']
        if importer_filename == caller_frame.filename:
            importer_reference = '%s line %s' % (
                importer_filename, str(caller_frame.lineno))
        else:
            importer_reference = importer_filename
        result = builtin_import(module, globals, locals, fromlist, level)
        importee_module = result.__name__
        # We're only interested if importer and importee are in SymPy
        if relevant(importer_module) and relevant(importee_module):
            for symbol in result.__dict__.iterkeys():
                definition = Definition(
                    symbol, result.__dict__[symbol], importer_module)
                if definition not in symbol_definers:
                    symbol_definers[definition] = importee_module
            if hasattr(result, '__path__'):
                ##PACKAGE##
                # The existence of __path__ is documented in the tutorial on modules.
                # Python 3.3 documents this in http://docs.python.org/3.3/reference/import.html
                if options.by_origin:
                    msg('Error: %s (a package) is imported by %s',
                        module, importer_reference)
                else:
                    msg('Error: %s contains package import %s',
                        importer_reference, module)
            if fromlist != None:
                symbol_list = fromlist
                if '*' in symbol_list:
                    if (importer_filename.endswith(("__init__.py", "__init__.pyc", "__init__.pyo"))):
                        # We do not check starred imports inside __init__
                        # That's the normal "please copy over its imports to my namespace"
                        symbol_list = []
                    else:
                        symbol_list = result.__dict__.iterkeys()
                for symbol in symbol_list:
                    if symbol not in result.__dict__:
                        if options.by_origin:
                            msg('Error: %s.%s is not defined (yet), but %s tries to import it',
                                importee_module, symbol, importer_reference)
                        else:
                            msg('Error: %s tries to import %s.%s, which did not define it (yet)',
                                importer_reference, importee_module, symbol)
                    else:
                        definition = Definition(
                            symbol, result.__dict__[symbol], importer_module)
                        symbol_definer = symbol_definers[definition]
                        if symbol_definer == importee_module:
                            ##DUPLICATE##
                            if options.by_origin:
                                msg('Error: %s.%s is imported again into %s',
                                    importee_module, symbol, importer_reference)
                            else:
                                msg('Error: %s imports %s.%s again',
                                      importer_reference, importee_module, symbol)
                        else:
                            ##ORIGIN##
                            if options.by_origin:
                                msg('Error: %s.%s is imported by %s, which should import %s.%s instead',
                                      importee_module, symbol, importer_reference, symbol_definer, symbol)
                            else:
                                msg('Error: %s imports %s.%s but should import %s.%s instead',
                                      importer_reference, importee_module, symbol, symbol_definer, symbol)
        return result

    builtins.__import__ = tracking_import
    __import__('sympy')

    sorted_messages.sort()
    for message in sorted_messages:
        print(message)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\extratest_gamma.py ===
from mpmath import *
from mpmath.libmp import ifac

import sys
if "-dps" in sys.argv:
    maxdps = int(sys.argv[sys.argv.index("-dps")+1])
else:
    maxdps = 1000

raise_ = "-raise" in sys.argv

errcount = 0

def check(name, func, z, y):
    global errcount
    try:
        x = func(z)
    except:
        errcount += 1
        if raise_:
            raise
        print()
        print(name)
        print("EXCEPTION")
        import traceback
        traceback.print_tb(sys.exc_info()[2])
        print()
        return
    xre = x.real
    xim = x.imag
    yre = y.real
    yim = y.imag
    tol = eps*8
    err = 0
    if abs(xre-yre) > abs(yre)*tol:
        err = 1
        print()
        print("Error! %s (re = %s, wanted %s, err=%s)" % (name, nstr(xre,10), nstr(yre,10), nstr(abs(xre-yre))))
        errcount += 1
        if raise_:
            raise SystemExit
    if abs(xim-yim) > abs(yim)*tol:
        err = 1
        print()
        print("Error! %s (im = %s, wanted %s, err=%s)" % (name, nstr(xim,10), nstr(yim,10), nstr(abs(xim-yim))))
        errcount += 1
        if raise_:
            raise SystemExit
    if not err:
        sys.stdout.write("%s ok; " % name)

def testcase(case):
    z, result = case
    print("Testing z =", z)
    mp.dps = 1010
    z = eval(z)
    mp.dps = maxdps + 50
    if result is None:
        gamma_val = gamma(z)
        loggamma_val = loggamma(z)
        factorial_val = factorial(z)
        rgamma_val = rgamma(z)
    else:
        loggamma_val = eval(result)
        gamma_val = exp(loggamma_val)
        factorial_val = z * gamma_val
        rgamma_val = 1/gamma_val
    for dps in [5, 10, 15, 25, 40, 60, 90, 120, 250, 600, 1000, 1800, 3600]:
        if dps > maxdps:
            break
        mp.dps = dps
        print("dps = %s" % dps)
        check("gamma", gamma, z, gamma_val)
        check("rgamma", rgamma, z, rgamma_val)
        check("loggamma", loggamma, z, loggamma_val)
        check("factorial", factorial, z, factorial_val)
        print()
        mp.dps = 15

testcases = []

# Basic values
for n in list(range(1,200)) + list(range(201,2000,17)):
    testcases.append(["%s" % n, None])
for n in range(-200,200):
    testcases.append(["%s+0.5" % n, None])
    testcases.append(["%s+0.37" % n, None])

testcases += [\
["(0.1+1j)", None],
["(-0.1+1j)", None],
["(0.1-1j)", None],
["(-0.1-1j)", None],
["10j", None],
["-10j", None],
["100j", None],
["10000j", None],
["-10000000j", None],
["(10**100)*j", None],
["125+(10**100)*j", None],
["-125+(10**100)*j", None],
["(10**10)*(1+j)", None],
["(10**10)*(-1+j)", None],
["(10**100)*(1+j)", None],
["(10**100)*(-1+j)", None],
["(1.5-1j)", None],
["(6+4j)", None],
["(4+1j)", None],
["(3.5+2j)", None],
["(1.5-1j)", None],
["(-6-4j)", None],
["(-2-3j)", None],
["(-2.5-2j)", None],
["(4+1j)", None],
["(3+3j)", None],
["(2-2j)", None],
["1", "0"],
["2", "0"],
["3", "log(2)"],
["4", "log(6)"],
["5", "log(24)"],
["0.5", "log(pi)/2"],
["1.5", "log(sqrt(pi)/2)"],
["2.5", "log(3*sqrt(pi)/4)"],
["mpf('0.37')", None],
["0.25", "log(sqrt(2*sqrt(2*pi**3)/agm(1,sqrt(2))))"],
["-0.4", None],
["mpf('-1.9')", None],
["mpf('12.8')", None],
["mpf('33.7')", None],
["mpf('95.2')", None],
["mpf('160.3')", None],
["mpf('2057.8')", None],
["25", "log(ifac(24))"],
["80", "log(ifac(79))"],
["500", "log(ifac(500-1))"],
["8000", "log(ifac(8000-1))"],
["8000.5", None],
["mpf('8000.1')", None],
["mpf('1.37e10')", None],
["mpf('1.37e10')*(1+j)", None],
["mpf('1.37e10')*(-1+j)", None],
["mpf('1.37e10')*(-1-j)", None],
["mpf('1.37e10')*(-1+j)", None],
["mpf('1.37e100')", None],
["mpf('1.37e100')*(1+j)", None],
["mpf('1.37e100')*(-1+j)", None],
["mpf('1.37e100')*(-1-j)", None],
["mpf('1.37e100')*(-1+j)", None],
["3+4j",
"mpc('"
"-1.7566267846037841105306041816232757851567066070613445016197619371316057169"
"4723618263960834804618463052988607348289672535780644470689771115236512106002"
"5970873471563240537307638968509556191696167970488390423963867031934333890838"
"8009531786948197210025029725361069435208930363494971027388382086721660805397"
"9163230643216054580167976201709951509519218635460317367338612500626714783631"
"7498317478048447525674016344322545858832610325861086336204591943822302971823"
"5161814175530618223688296232894588415495615809337292518431903058265147109853"
"1710568942184987827643886816200452860853873815413367529829631430146227470517"
"6579967222200868632179482214312673161276976117132204633283806161971389519137"
"1243359764435612951384238091232760634271570950240717650166551484551654327989"
"9360285030081716934130446150245110557038117075172576825490035434069388648124"
"6678152254554001586736120762641422590778766100376515737713938521275749049949"
"1284143906816424244705094759339932733567910991920631339597278805393743140853"
"391550313363278558195609260225928','"
"4.74266443803465792819488940755002274088830335171164611359052405215840070271"
"5906813009373171139767051863542508136875688550817670379002790304870822775498"
"2809996675877564504192565392367259119610438951593128982646945990372179860613"
"4294436498090428077839141927485901735557543641049637962003652638924845391650"
"9546290137755550107224907606529385248390667634297183361902055842228798984200"
"9591180450211798341715874477629099687609819466457990642030707080894518168924"
"6805549314043258530272479246115112769957368212585759640878745385160943755234"
"9398036774908108204370323896757543121853650025529763655312360354244898913463"
"7115955702828838923393113618205074162812089732064414530813087483533203244056"
"0546577484241423134079056537777170351934430586103623577814746004431994179990"
"5318522939077992613855205801498201930221975721246498720895122345420698451980"
"0051215797310305885845964334761831751370672996984756815410977750799748813563"
"8784405288158432214886648743541773208808731479748217023665577802702269468013"
"673719173759245720489020315779001')"],
]

for z in [4, 14, 34, 64]:
    testcases.append(["(2+j)*%s/3" % z, None])
    testcases.append(["(-2+j)*%s/3" % z, None])
    testcases.append(["(1+2*j)*%s/3" % z, None])
    testcases.append(["(2-j)*%s/3" % z, None])
    testcases.append(["(20+j)*%s/3" % z, None])
    testcases.append(["(-20+j)*%s/3" % z, None])
    testcases.append(["(1+20*j)*%s/3" % z, None])
    testcases.append(["(20-j)*%s/3" % z, None])
    testcases.append(["(200+j)*%s/3" % z, None])
    testcases.append(["(-200+j)*%s/3" % z, None])
    testcases.append(["(1+200*j)*%s/3" % z, None])
    testcases.append(["(200-j)*%s/3" % z, None])

# Poles
for n in [0,1,2,3,4,25,-1,-2,-3,-4,-20,-21,-50,-51,-200,-201,-20000,-20001]:
    for t in ['1e-5', '1e-20', '1e-100', '1e-10000']:
        testcases.append(["fadd(%s,'%s',exact=True)" % (n, t), None])
        testcases.append(["fsub(%s,'%s',exact=True)" % (n, t), None])
        testcases.append(["fadd(%s,'%sj',exact=True)" % (n, t), None])
        testcases.append(["fsub(%s,'%sj',exact=True)" % (n, t), None])

if __name__ == "__main__":
    from timeit import default_timer as clock
    tot_time = 0.0
    for case in testcases:
        t1 = clock()
        testcase(case)
        t2 = clock()
        print("Test time:", t2-t1)
        print()
        tot_time += (t2-t1)
    print("Total time:", tot_time)
    print("Errors:", errcount)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_logical_ops.py ===
import operator
import re

import numpy as np
import pytest

from pandas import (
    CategoricalIndex,
    DataFrame,
    Interval,
    Series,
    isnull,
)
import pandas._testing as tm


class TestDataFrameLogicalOperators:
    # &, |, ^

    @pytest.mark.parametrize(
        "left, right, op, expected",
        [
            (
                [True, False, np.nan],
                [True, False, True],
                operator.and_,
                [True, False, False],
            ),
            (
                [True, False, True],
                [True, False, np.nan],
                operator.and_,
                [True, False, False],
            ),
            (
                [True, False, np.nan],
                [True, False, True],
                operator.or_,
                [True, False, False],
            ),
            (
                [True, False, True],
                [True, False, np.nan],
                operator.or_,
                [True, False, True],
            ),
        ],
    )
    def test_logical_operators_nans(self, left, right, op, expected, frame_or_series):
        # GH#13896
        result = op(frame_or_series(left), frame_or_series(right))
        expected = frame_or_series(expected)

        tm.assert_equal(result, expected)

    def test_logical_ops_empty_frame(self):
        # GH#5808
        # empty frames, non-mixed dtype
        df = DataFrame(index=[1])

        result = df & df
        tm.assert_frame_equal(result, df)

        result = df | df
        tm.assert_frame_equal(result, df)

        df2 = DataFrame(index=[1, 2])
        result = df & df2
        tm.assert_frame_equal(result, df2)

        dfa = DataFrame(index=[1], columns=["A"])

        result = dfa & dfa
        expected = DataFrame(False, index=[1], columns=["A"])
        tm.assert_frame_equal(result, expected)

    def test_logical_ops_bool_frame(self):
        # GH#5808
        df1a_bool = DataFrame(True, index=[1], columns=["A"])

        result = df1a_bool & df1a_bool
        tm.assert_frame_equal(result, df1a_bool)

        result = df1a_bool | df1a_bool
        tm.assert_frame_equal(result, df1a_bool)

    def test_logical_ops_int_frame(self):
        # GH#5808
        df1a_int = DataFrame(1, index=[1], columns=["A"])
        df1a_bool = DataFrame(True, index=[1], columns=["A"])

        result = df1a_int | df1a_bool
        tm.assert_frame_equal(result, df1a_bool)

        # Check that this matches Series behavior
        res_ser = df1a_int["A"] | df1a_bool["A"]
        tm.assert_series_equal(res_ser, df1a_bool["A"])

    def test_logical_ops_invalid(self, using_infer_string):
        # GH#5808

        df1 = DataFrame(1.0, index=[1], columns=["A"])
        df2 = DataFrame(True, index=[1], columns=["A"])
        msg = re.escape("unsupported operand type(s) for |: 'float' and 'bool'")
        with pytest.raises(TypeError, match=msg):
            df1 | df2

        df1 = DataFrame("foo", index=[1], columns=["A"])
        df2 = DataFrame(True, index=[1], columns=["A"])
        if using_infer_string and df1["A"].dtype.storage == "pyarrow":
            msg = "operation 'or_' not supported for dtype 'str'"
        else:
            msg = re.escape("unsupported operand type(s) for |: 'str' and 'bool'")
        with pytest.raises(TypeError, match=msg):
            df1 | df2

    def test_logical_operators(self):
        def _check_bin_op(op):
            result = op(df1, df2)
            expected = DataFrame(
                op(df1.values, df2.values), index=df1.index, columns=df1.columns
            )
            assert result.values.dtype == np.bool_
            tm.assert_frame_equal(result, expected)

        def _check_unary_op(op):
            result = op(df1)
            expected = DataFrame(op(df1.values), index=df1.index, columns=df1.columns)
            assert result.values.dtype == np.bool_
            tm.assert_frame_equal(result, expected)

        df1 = {
            "a": {"a": True, "b": False, "c": False, "d": True, "e": True},
            "b": {"a": False, "b": True, "c": False, "d": False, "e": False},
            "c": {"a": False, "b": False, "c": True, "d": False, "e": False},
            "d": {"a": True, "b": False, "c": False, "d": True, "e": True},
            "e": {"a": True, "b": False, "c": False, "d": True, "e": True},
        }

        df2 = {
            "a": {"a": True, "b": False, "c": True, "d": False, "e": False},
            "b": {"a": False, "b": True, "c": False, "d": False, "e": False},
            "c": {"a": True, "b": False, "c": True, "d": False, "e": False},
            "d": {"a": False, "b": False, "c": False, "d": True, "e": False},
            "e": {"a": False, "b": False, "c": False, "d": False, "e": True},
        }

        df1 = DataFrame(df1)
        df2 = DataFrame(df2)

        _check_bin_op(operator.and_)
        _check_bin_op(operator.or_)
        _check_bin_op(operator.xor)

        _check_unary_op(operator.inv)  # TODO: belongs elsewhere

    @pytest.mark.filterwarnings("ignore:Downcasting object dtype arrays:FutureWarning")
    def test_logical_with_nas(self):
        d = DataFrame({"a": [np.nan, False], "b": [True, True]})

        # GH4947
        # bool comparisons should return bool
        result = d["a"] | d["b"]
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)

        # GH4604, automatic casting here
        result = d["a"].fillna(False) | d["b"]
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = d["a"].fillna(False, downcast=False) | d["b"]
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

    def test_logical_ops_categorical_columns(self):
        # GH#38367
        intervals = [Interval(1, 2), Interval(3, 4)]
        data = DataFrame(
            [[1, np.nan], [2, np.nan]],
            columns=CategoricalIndex(
                intervals, categories=intervals + [Interval(5, 6)]
            ),
        )
        mask = DataFrame(
            [[False, False], [False, False]], columns=data.columns, dtype=bool
        )
        result = mask | isnull(data)
        expected = DataFrame(
            [[False, True], [False, True]],
            columns=CategoricalIndex(
                intervals, categories=intervals + [Interval(5, 6)]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_int_dtype_different_index_not_bool(self):
        # GH 52500
        df1 = DataFrame([1, 2, 3], index=[10, 11, 23], columns=["a"])
        df2 = DataFrame([10, 20, 30], index=[11, 10, 23], columns=["a"])
        result = np.bitwise_xor(df1, df2)
        expected = DataFrame([21, 8, 29], index=[10, 11, 23], columns=["a"])
        tm.assert_frame_equal(result, expected)

        result = df1 ^ df2
        tm.assert_frame_equal(result, expected)

    def test_different_dtypes_different_index_raises(self):
        # GH 52538
        df1 = DataFrame([1, 2], index=["a", "b"])
        df2 = DataFrame([3, 4], index=["b", "c"])
        with pytest.raises(TypeError, match="unsupported operand type"):
            df1 & df2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_business_year.py ===
"""
Tests for the following offsets:
- BYearBegin
- BYearEnd
"""
from __future__ import annotations

from datetime import datetime

import pytest

from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries.offsets import (
    BYearBegin,
    BYearEnd,
)


class TestBYearBegin:
    def test_misspecified(self):
        msg = "Month must go from 1 to 12"
        with pytest.raises(ValueError, match=msg):
            BYearBegin(month=13)
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)

    offset_cases = []
    offset_cases.append(
        (
            BYearBegin(),
            {
                datetime(2008, 1, 1): datetime(2009, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2011, 1, 1): datetime(2011, 1, 3),
                datetime(2011, 1, 3): datetime(2012, 1, 2),
                datetime(2005, 12, 30): datetime(2006, 1, 2),
                datetime(2005, 12, 31): datetime(2006, 1, 2),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2009, 1, 1),
                datetime(2008, 12, 31): datetime(2009, 1, 1),
                datetime(2005, 12, 30): datetime(2006, 1, 2),
                datetime(2005, 12, 31): datetime(2006, 1, 2),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 1, 2),
                datetime(2009, 1, 4): datetime(2009, 1, 1),
                datetime(2009, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 6, 30): datetime(2008, 1, 1),
                datetime(2008, 12, 31): datetime(2008, 1, 1),
                datetime(2006, 12, 29): datetime(2006, 1, 2),
                datetime(2006, 12, 30): datetime(2006, 1, 2),
                datetime(2006, 1, 1): datetime(2005, 1, 3),
            },
        )
    )

    offset_cases.append(
        (
            BYearBegin(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 1, 3),
                datetime(2007, 6, 30): datetime(2006, 1, 2),
                datetime(2008, 12, 31): datetime(2007, 1, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestBYearEnd:
    offset_cases = []
    offset_cases.append(
        (
            BYearEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2009, 12, 31),
                datetime(2005, 12, 30): datetime(2006, 12, 29),
                datetime(2005, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 12, 31),
                datetime(2008, 12, 31): datetime(2008, 12, 31),
                datetime(2005, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 29),
                datetime(2008, 6, 30): datetime(2007, 12, 31),
                datetime(2008, 12, 31): datetime(2007, 12, 31),
                datetime(2006, 12, 29): datetime(2005, 12, 30),
                datetime(2006, 12, 30): datetime(2006, 12, 29),
                datetime(2007, 1, 1): datetime(2006, 12, 29),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2005, 12, 30),
                datetime(2008, 6, 30): datetime(2006, 12, 29),
                datetime(2008, 12, 31): datetime(2006, 12, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (BYearEnd(), datetime(2007, 12, 31), True),
        (BYearEnd(), datetime(2008, 1, 1), False),
        (BYearEnd(), datetime(2006, 12, 31), False),
        (BYearEnd(), datetime(2006, 12, 29), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)


class TestBYearEndLagged:
    def test_bad_month_fail(self):
        msg = "Month must go from 1 to 12"
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=13)
        with pytest.raises(ValueError, match=msg):
            BYearEnd(month=0)

    offset_cases = []
    offset_cases.append(
        (
            BYearEnd(month=6),
            {
                datetime(2008, 1, 1): datetime(2008, 6, 30),
                datetime(2007, 6, 30): datetime(2008, 6, 30),
            },
        )
    )

    offset_cases.append(
        (
            BYearEnd(n=-1, month=6),
            {
                datetime(2008, 1, 1): datetime(2007, 6, 29),
                datetime(2007, 6, 30): datetime(2007, 6, 29),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_roll(self):
        offset = BYearEnd(month=6)
        date = datetime(2009, 11, 30)

        assert offset.rollforward(date) == datetime(2010, 6, 30)
        assert offset.rollback(date) == datetime(2009, 6, 30)

    on_offset_cases = [
        (BYearEnd(month=2), datetime(2007, 2, 28), True),
        (BYearEnd(month=6), datetime(2007, 6, 30), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_coercion.py ===
"""
Tests for values coercion in setitem-like operations on DataFrame.

For the most part, these should be multi-column DataFrames, otherwise
we would share the tests with Series.
"""
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDataFrameSetitemCoercion:
    @pytest.mark.parametrize("consolidate", [True, False])
    def test_loc_setitem_multiindex_columns(self, consolidate):
        # GH#18415 Setting values in a single column preserves dtype,
        #  while setting them in multiple columns did unwanted cast.

        # Note that A here has 2 blocks, below we do the same thing
        #  with a consolidated frame.
        A = DataFrame(np.zeros((6, 5), dtype=np.float32))
        A = pd.concat([A, A], axis=1, keys=[1, 2])
        if consolidate:
            A = A._consolidate()

        A.loc[2:3, (1, slice(2, 3))] = np.ones((2, 2), dtype=np.float32)
        assert (A.dtypes == np.float32).all()

        A.loc[0:5, (1, slice(2, 3))] = np.ones((6, 2), dtype=np.float32)

        assert (A.dtypes == np.float32).all()

        A.loc[:, (1, slice(2, 3))] = np.ones((6, 2), dtype=np.float32)
        assert (A.dtypes == np.float32).all()

        # TODO: i think this isn't about MultiIndex and could be done with iloc?


def test_37477():
    # fixed by GH#45121
    orig = DataFrame({"A": [1, 2, 3], "B": [3, 4, 5]})
    expected = DataFrame({"A": [1, 2, 3], "B": [3, 1.2, 5]})

    df = orig.copy()
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        df.at[1, "B"] = 1.2
    tm.assert_frame_equal(df, expected)

    df = orig.copy()
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        df.loc[1, "B"] = 1.2
    tm.assert_frame_equal(df, expected)

    df = orig.copy()
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        df.iat[1, 1] = 1.2
    tm.assert_frame_equal(df, expected)

    df = orig.copy()
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        df.iloc[1, 1] = 1.2
    tm.assert_frame_equal(df, expected)


def test_6942(indexer_al):
    # check that the .at __setitem__ after setting "Live" actually sets the data
    start = Timestamp("2014-04-01")
    t1 = Timestamp("2014-04-23 12:42:38.883082")
    t2 = Timestamp("2014-04-24 01:33:30.040039")

    dti = date_range(start, periods=1)
    orig = DataFrame(index=dti, columns=["timenow", "Live"])

    df = orig.copy()
    indexer_al(df)[start, "timenow"] = t1

    df["Live"] = True

    df.at[start, "timenow"] = t2
    assert df.iloc[0, 0] == t2


def test_26395(indexer_al):
    # .at case fixed by GH#45121 (best guess)
    df = DataFrame(index=["A", "B", "C"])
    df["D"] = 0

    indexer_al(df)["C", "D"] = 2
    expected = DataFrame(
        {"D": [0, 0, 2]},
        index=["A", "B", "C"],
        columns=pd.Index(["D"], dtype=object),
        dtype=np.int64,
    )
    tm.assert_frame_equal(df, expected)

    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        indexer_al(df)["C", "D"] = 44.5
    expected = DataFrame(
        {"D": [0, 0, 44.5]},
        index=["A", "B", "C"],
        columns=pd.Index(["D"], dtype=object),
        dtype=np.float64,
    )
    tm.assert_frame_equal(df, expected)

    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        indexer_al(df)["C", "D"] = "hello"
    expected = DataFrame(
        {"D": [0, 0, "hello"]},
        index=["A", "B", "C"],
        columns=pd.Index(["D"], dtype=object),
        dtype=object,
    )
    tm.assert_frame_equal(df, expected)


@pytest.mark.xfail(reason="unwanted upcast")
def test_15231():
    df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    df.loc[2] = Series({"a": 5, "b": 6})
    assert (df.dtypes == np.int64).all()

    df.loc[3] = Series({"a": 7})

    # df["a"] doesn't have any NaNs, should not have been cast
    exp_dtypes = Series([np.int64, np.float64], dtype=object, index=["a", "b"])
    tm.assert_series_equal(df.dtypes, exp_dtypes)


def test_iloc_setitem_unnecesssary_float_upcasting():
    # GH#12255
    df = DataFrame(
        {
            0: np.array([1, 3], dtype=np.float32),
            1: np.array([2, 4], dtype=np.float32),
            2: ["a", "b"],
        }
    )
    orig = df.copy()

    values = df[0].values.reshape(2, 1)
    df.iloc[:, 0:1] = values

    tm.assert_frame_equal(df, orig)


@pytest.mark.xfail(reason="unwanted casting to dt64")
def test_12499():
    # TODO: OP in GH#12499 used np.datetim64("NaT") instead of pd.NaT,
    #  which has consequences for the expected df["two"] (though i think at
    #  the time it might not have because of a separate bug). See if it makes
    #  a difference which one we use here.
    ts = Timestamp("2016-03-01 03:13:22.98986", tz="UTC")

    data = [{"one": 0, "two": ts}]
    orig = DataFrame(data)
    df = orig.copy()
    df.loc[1] = [np.nan, NaT]

    expected = DataFrame(
        {"one": [0, np.nan], "two": Series([ts, NaT], dtype="datetime64[ns, UTC]")}
    )
    tm.assert_frame_equal(df, expected)

    data = [{"one": 0, "two": ts}]
    df = orig.copy()
    df.loc[1, :] = [np.nan, NaT]
    tm.assert_frame_equal(df, expected)


def test_20476():
    mi = MultiIndex.from_product([["A", "B"], ["a", "b", "c"]])
    df = DataFrame(-1, index=range(3), columns=mi)
    filler = DataFrame([[1, 2, 3.0]] * 3, index=range(3), columns=["a", "b", "c"])
    df["A"] = filler

    expected = DataFrame(
        {
            0: [1, 1, 1],
            1: [2, 2, 2],
            2: [3.0, 3.0, 3.0],
            3: [-1, -1, -1],
            4: [-1, -1, -1],
            5: [-1, -1, -1],
        }
    )
    expected.columns = mi
    exp_dtypes = Series(
        [np.dtype(np.int64)] * 2 + [np.dtype(np.float64)] + [np.dtype(np.int64)] * 3,
        index=mi,
    )
    tm.assert_series_equal(df.dtypes, exp_dtypes)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_categorical.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    Series,
    _testing as tm,
    concat,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)

pytestmark = [pytest.mark.single_cpu]


def test_categorical(setup_path):
    with ensure_clean_store(setup_path) as store:
        # Basic
        _maybe_remove(store, "s")
        s = Series(
            Categorical(
                ["a", "b", "b", "a", "a", "c"],
                categories=["a", "b", "c", "d"],
                ordered=False,
            )
        )
        store.append("s", s, format="table")
        result = store.select("s")
        tm.assert_series_equal(s, result)

        _maybe_remove(store, "s_ordered")
        s = Series(
            Categorical(
                ["a", "b", "b", "a", "a", "c"],
                categories=["a", "b", "c", "d"],
                ordered=True,
            )
        )
        store.append("s_ordered", s, format="table")
        result = store.select("s_ordered")
        tm.assert_series_equal(s, result)

        _maybe_remove(store, "df")
        df = DataFrame({"s": s, "vals": [1, 2, 3, 4, 5, 6]})
        store.append("df", df, format="table")
        result = store.select("df")
        tm.assert_frame_equal(result, df)

        # Dtypes
        _maybe_remove(store, "si")
        s = Series([1, 1, 2, 2, 3, 4, 5]).astype("category")
        store.append("si", s)
        result = store.select("si")
        tm.assert_series_equal(result, s)

        _maybe_remove(store, "si2")
        s = Series([1, 1, np.nan, 2, 3, 4, 5]).astype("category")
        store.append("si2", s)
        result = store.select("si2")
        tm.assert_series_equal(result, s)

        # Multiple
        _maybe_remove(store, "df2")
        df2 = df.copy()
        df2["s2"] = Series(list("abcdefg")).astype("category")
        store.append("df2", df2)
        result = store.select("df2")
        tm.assert_frame_equal(result, df2)

        # Make sure the metadata is OK
        info = store.info()
        assert "/df2   " in info
        # df2._mgr.blocks[0] and df2._mgr.blocks[2] are Categorical
        assert "/df2/meta/values_block_0/meta" in info
        assert "/df2/meta/values_block_2/meta" in info

        # unordered
        _maybe_remove(store, "s2")
        s = Series(
            Categorical(
                ["a", "b", "b", "a", "a", "c"],
                categories=["a", "b", "c", "d"],
                ordered=False,
            )
        )
        store.append("s2", s, format="table")
        result = store.select("s2")
        tm.assert_series_equal(result, s)

        # Query
        _maybe_remove(store, "df3")
        store.append("df3", df, data_columns=["s"])
        expected = df[df.s.isin(["b", "c"])]
        result = store.select("df3", where=['s in ["b","c"]'])
        tm.assert_frame_equal(result, expected)

        expected = df[df.s.isin(["b", "c"])]
        result = store.select("df3", where=['s = ["b","c"]'])
        tm.assert_frame_equal(result, expected)

        expected = df[df.s.isin(["d"])]
        result = store.select("df3", where=['s in ["d"]'])
        tm.assert_frame_equal(result, expected)

        expected = df[df.s.isin(["f"])]
        result = store.select("df3", where=['s in ["f"]'])
        tm.assert_frame_equal(result, expected)

        # Appending with same categories is ok
        store.append("df3", df)

        df = concat([df, df])
        expected = df[df.s.isin(["b", "c"])]
        result = store.select("df3", where=['s in ["b","c"]'])
        tm.assert_frame_equal(result, expected)

        # Appending must have the same categories
        df3 = df.copy()
        df3["s"] = df3["s"].cat.remove_unused_categories()

        msg = "cannot append a categorical with different categories to the existing"
        with pytest.raises(ValueError, match=msg):
            store.append("df3", df3)

        # Remove, and make sure meta data is removed (its a recursive
        # removal so should be).
        result = store.select("df3/meta/s/meta")
        assert result is not None
        store.remove("df3")

        with pytest.raises(
            KeyError, match="'No object named df3/meta/s/meta in the file'"
        ):
            store.select("df3/meta/s/meta")


def test_categorical_conversion(tmp_path, setup_path):
    # GH13322
    # Check that read_hdf with categorical columns doesn't return rows if
    # where criteria isn't met.
    obsids = ["ESP_012345_6789", "ESP_987654_3210"]
    imgids = ["APF00006np", "APF0001imm"]
    data = [4.3, 9.8]

    # Test without categories
    df = DataFrame({"obsids": obsids, "imgids": imgids, "data": data})

    # We are expecting an empty DataFrame matching types of df
    expected = df.iloc[[], :]
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table", data_columns=True)
    result = read_hdf(path, "df", where="obsids=B")
    tm.assert_frame_equal(result, expected)

    # Test with categories
    df.obsids = df.obsids.astype("category")
    df.imgids = df.imgids.astype("category")

    # We are expecting an empty DataFrame matching types of df
    expected = df.iloc[[], :]
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table", data_columns=True)
    result = read_hdf(path, "df", where="obsids=B")
    tm.assert_frame_equal(result, expected)


def test_categorical_nan_only_columns(tmp_path, setup_path):
    # GH18413
    # Check that read_hdf with categorical columns with NaN-only values can
    # be read back.
    df = DataFrame(
        {
            "a": ["a", "b", "c", np.nan],
            "b": [np.nan, np.nan, np.nan, np.nan],
            "c": [1, 2, 3, 4],
            "d": Series([None] * 4, dtype=object),
        }
    )
    df["a"] = df.a.astype("category")
    df["b"] = df.b.astype("category")
    df["d"] = df.b.astype("category")
    expected = df
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table", data_columns=True)
    result = read_hdf(path, "df")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "where, df, expected",
    [
        ('col=="q"', DataFrame({"col": ["a", "b", "s"]}), DataFrame({"col": []})),
        ('col=="a"', DataFrame({"col": ["a", "b", "s"]}), DataFrame({"col": ["a"]})),
    ],
)
def test_convert_value(
    tmp_path, setup_path, where: str, df: DataFrame, expected: DataFrame
):
    # GH39420
    # Check that read_hdf with categorical columns can filter by where condition.
    df.col = df.col.astype("category")
    max_widths = {"col": 1}
    categorical_values = sorted(df.col.unique())
    expected.col = expected.col.astype("category")
    expected.col = expected.col.cat.set_categories(categorical_values)

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table", min_itemsize=max_widths)
    result = read_hdf(path, where=where)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_fnodes.py ===
import os
import tempfile
from sympy.core.symbol import (Symbol, symbols)
from sympy.codegen.ast import (
    Assignment, Print, Declaration, FunctionDefinition, Return, real,
    FunctionCall, Variable, Element, integer
)
from sympy.codegen.fnodes import (
    allocatable, ArrayConstructor, isign, dsign, cmplx, kind, literal_dp,
    Program, Module, use, Subroutine, dimension, assumed_extent, ImpliedDoLoop,
    intent_out, size, Do, SubroutineCall, sum_, array, bind_C
)
from sympy.codegen.futils import render_as_module
from sympy.core.expr import unchanged
from sympy.external import import_module
from sympy.printing.codeprinter import fcode
from sympy.utilities._compilation import has_fortran, compile_run_strings, compile_link_import_strings
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, XFAIL

cython = import_module('cython')
np = import_module('numpy')


def test_size():
    x = Symbol('x', real=True)
    sx = size(x)
    assert fcode(sx, source_format='free') == 'size(x)'


@may_xfail
def test_size_assumed_shape():
    if not has_fortran():
        skip("No fortran compiler found.")
    a = Symbol('a', real=True)
    body = [Return((sum_(a**2)/size(a))**.5)]
    arr = array(a, dim=[':'], intent='in')
    fd = FunctionDefinition(real, 'rms', [arr], body)
    render_as_module([fd], 'mod_rms')

    (stdout, stderr), info = compile_run_strings([
        ('rms.f90', render_as_module([fd], 'mod_rms')),
        ('main.f90', (
            'program myprog\n'
            'use mod_rms, only: rms\n'
            'real*8, dimension(4), parameter :: x = [4, 2, 2, 2]\n'
            'print "(f7.5)", dsqrt(7d0) - rms(x)\n'
            'end program\n'
        ))
    ], clean=True)
    assert '0.00000' in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK


@XFAIL  # https://github.com/sympy/sympy/issues/20265
@may_xfail
def test_ImpliedDoLoop():
    if not has_fortran():
        skip("No fortran compiler found.")

    a, i = symbols('a i', integer=True)
    idl = ImpliedDoLoop(i**3, i, -3, 3, 2)
    ac = ArrayConstructor([-28, idl, 28])
    a = array(a, dim=[':'], attrs=[allocatable])
    prog = Program('idlprog', [
        a.as_Declaration(),
        Assignment(a, ac),
        Print([a])
    ])
    fsrc = fcode(prog, standard=2003, source_format='free')
    (stdout, stderr), info = compile_run_strings([('main.f90', fsrc)], clean=True)
    for numstr in '-28 -27 -1 1 27 28'.split():
        assert numstr in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK


@may_xfail
def test_Program():
    x = Symbol('x', real=True)
    vx = Variable.deduced(x, 42)
    decl = Declaration(vx)
    prnt = Print([x, x+1])
    prog = Program('foo', [decl, prnt])
    if not has_fortran():
        skip("No fortran compiler found.")

    (stdout, stderr), info = compile_run_strings([('main.f90', fcode(prog, standard=90))], clean=True)
    assert '42' in stdout
    assert '43' in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK


@may_xfail
def test_Module():
    x = Symbol('x', real=True)
    v_x = Variable.deduced(x)
    sq = FunctionDefinition(real, 'sqr', [v_x], [Return(x**2)])
    mod_sq = Module('mod_sq', [], [sq])
    sq_call = FunctionCall('sqr', [42.])
    prg_sq = Program('foobar', [
        use('mod_sq', only=['sqr']),
        Print(['"Square of 42 = "', sq_call])
    ])
    if not has_fortran():
        skip("No fortran compiler found.")
    (stdout, stderr), info = compile_run_strings([
        ('mod_sq.f90', fcode(mod_sq, standard=90)),
        ('main.f90', fcode(prg_sq, standard=90))
    ], clean=True)
    assert '42' in stdout
    assert str(42**2) in stdout
    assert stderr == ''


@XFAIL  # https://github.com/sympy/sympy/issues/20265
@may_xfail
def test_Subroutine():
    # Code to generate the subroutine in the example from
    # http://www.fortran90.org/src/best-practices.html#arrays
    r = Symbol('r', real=True)
    i = Symbol('i', integer=True)
    v_r = Variable.deduced(r, attrs=(dimension(assumed_extent), intent_out))
    v_i = Variable.deduced(i)
    v_n = Variable('n', integer)
    do_loop = Do([
        Assignment(Element(r, [i]), literal_dp(1)/i**2)
    ], i, 1, v_n)
    sub = Subroutine("f", [v_r], [
        Declaration(v_n),
        Declaration(v_i),
        Assignment(v_n, size(r)),
        do_loop
    ])
    x = Symbol('x', real=True)
    v_x3 = Variable.deduced(x, attrs=[dimension(3)])
    mod = Module('mymod', definitions=[sub])
    prog = Program('foo', [
        use(mod, only=[sub]),
        Declaration(v_x3),
        SubroutineCall(sub, [v_x3]),
        Print([sum_(v_x3), v_x3])
    ])

    if not has_fortran():
        skip("No fortran compiler found.")

    (stdout, stderr), info = compile_run_strings([
        ('a.f90', fcode(mod, standard=90)),
        ('b.f90', fcode(prog, standard=90))
    ], clean=True)
    ref = [1.0/i**2 for i in range(1, 4)]
    assert str(sum(ref))[:-3] in stdout
    for _ in ref:
        assert str(_)[:-3] in stdout
    assert stderr == ''


def test_isign():
    x = Symbol('x', integer=True)
    assert unchanged(isign, 1, x)
    assert fcode(isign(1, x), standard=95, source_format='free') == 'isign(1, x)'


def test_dsign():
    x = Symbol('x')
    assert unchanged(dsign, 1, x)
    assert fcode(dsign(literal_dp(1), x), standard=95, source_format='free') == 'dsign(1d0, x)'


def test_cmplx():
    x = Symbol('x')
    assert unchanged(cmplx, 1, x)


def test_kind():
    x = Symbol('x')
    assert unchanged(kind, x)


def test_literal_dp():
    assert fcode(literal_dp(0), source_format='free') == '0d0'


@may_xfail
def test_bind_C():
    if not has_fortran():
        skip("No fortran compiler found.")
    if not cython:
        skip("Cython not found.")
    if not np:
        skip("NumPy not found.")

    a = Symbol('a', real=True)
    s = Symbol('s', integer=True)
    body = [Return((sum_(a**2)/s)**.5)]
    arr = array(a, dim=[s], intent='in')
    fd = FunctionDefinition(real, 'rms', [arr, s], body, attrs=[bind_C('rms')])
    f_mod = render_as_module([fd], 'mod_rms')

    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([
            ('rms.f90', f_mod),
            ('_rms.pyx', (
                "#cython: language_level={}\n".format("3") +
                "cdef extern double rms(double*, int*)\n"
                "def py_rms(double[::1] x):\n"
                "    cdef int s = x.size\n"
                "    return rms(&x[0], &s)\n"))
        ], build_dir=folder)
        assert abs(mod.py_rms(np.array([2., 4., 2., 2.])) - 7**0.5) < 1e-14

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\multipledispatch\tests\test_core.py ===
from __future__ import annotations
from typing import Any

from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns
from functools import partial

test_namespace: dict[str, Any] = {}

orig_dispatch = dispatch
dispatch = partial(dispatch, namespace=test_namespace)


def test_singledispatch():
    @dispatch(int)
    def f(x): # noqa:F811
        return x + 1

    @dispatch(int)
    def g(x): # noqa:F811
        return x + 2

    @dispatch(float) # noqa:F811
    def f(x): # noqa:F811
        return x - 1

    assert f(1) == 2
    assert g(1) == 3
    assert f(1.0) == 0

    assert raises(NotImplementedError, lambda: f('hello'))


def test_multipledispatch():
    @dispatch(int, int)
    def f(x, y): # noqa:F811
        return x + y

    @dispatch(float, float) # noqa:F811
    def f(x, y): # noqa:F811
        return x - y

    assert f(1, 2) == 3
    assert f(1.0, 2.0) == -1.0


class A: pass
class B: pass
class C(A): pass
class D(C): pass
class E(C): pass


def test_inheritance():
    @dispatch(A)
    def f(x): # noqa:F811
        return 'a'

    @dispatch(B) # noqa:F811
    def f(x): # noqa:F811
        return 'b'

    assert f(A()) == 'a'
    assert f(B()) == 'b'
    assert f(C()) == 'a'


def test_inheritance_and_multiple_dispatch():
    @dispatch(A, A)
    def f(x, y): # noqa:F811
        return type(x), type(y)

    @dispatch(A, B) # noqa:F811
    def f(x, y): # noqa:F811
        return 0

    assert f(A(), A()) == (A, A)
    assert f(A(), C()) == (A, C)
    assert f(A(), B()) == 0
    assert f(C(), B()) == 0
    assert raises(NotImplementedError, lambda: f(B(), B()))


def test_competing_solutions():
    @dispatch(A)
    def h(x): # noqa:F811
        return 1

    @dispatch(C) # noqa:F811
    def h(x): # noqa:F811
        return 2

    assert h(D()) == 2


def test_competing_multiple():
    @dispatch(A, B)
    def h(x, y): # noqa:F811
        return 1

    @dispatch(C, B) # noqa:F811
    def h(x, y): # noqa:F811
        return 2

    assert h(D(), B()) == 2


def test_competing_ambiguous():
    test_namespace = {}
    dispatch = partial(orig_dispatch, namespace=test_namespace)

    @dispatch(A, C)
    def f(x, y): # noqa:F811
        return 2

    with warns(AmbiguityWarning, test_stacklevel=False):
        @dispatch(C, A) # noqa:F811
        def f(x, y): # noqa:F811
            return 2

    assert f(A(), C()) == f(C(), A()) == 2
    # assert raises(Warning, lambda : f(C(), C()))


def test_caching_correct_behavior():
    @dispatch(A)
    def f(x): # noqa:F811
        return 1

    assert f(C()) == 1

    @dispatch(C)
    def f(x): # noqa:F811
        return 2

    assert f(C()) == 2


def test_union_types():
    @dispatch((A, C))
    def f(x): # noqa:F811
        return 1

    assert f(A()) == 1
    assert f(C()) == 1


def test_namespaces():
    ns1 = {}
    ns2 = {}

    def foo(x):
        return 1
    foo1 = orig_dispatch(int, namespace=ns1)(foo)

    def foo(x):
        return 2
    foo2 = orig_dispatch(int, namespace=ns2)(foo)

    assert foo1(0) == 1
    assert foo2(0) == 2


"""
Fails
def test_dispatch_on_dispatch():
    @dispatch(A)
    @dispatch(C)
    def q(x): # noqa:F811
        return 1

    assert q(A()) == 1
    assert q(C()) == 1
"""


def test_methods():
    class Foo:
        @dispatch(float)
        def f(self, x): # noqa:F811
            return x - 1

        @dispatch(int) # noqa:F811
        def f(self, x): # noqa:F811
            return x + 1

        @dispatch(int)
        def g(self, x): # noqa:F811
            return x + 3


    foo = Foo()
    assert foo.f(1) == 2
    assert foo.f(1.0) == 0.0
    assert foo.g(1) == 4


def test_methods_multiple_dispatch():
    class Foo:
        @dispatch(A, A)
        def f(x, y): # noqa:F811
            return 1

        @dispatch(A, C) # noqa:F811
        def f(x, y): # noqa:F811
            return 2


    foo = Foo()
    assert foo.f(A(), A()) == 1
    assert foo.f(A(), C()) == 2
    assert foo.f(C(), C()) == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\intervalmath\tests\test_intervalmath.py ===
from sympy.plotting.intervalmath import interval
from sympy.testing.pytest import raises


def test_interval():
    assert (interval(1, 1) == interval(1, 1, is_valid=True)) == (True, True)
    assert (interval(1, 1) == interval(1, 1, is_valid=False)) == (True, False)
    assert (interval(1, 1) == interval(1, 1, is_valid=None)) == (True, None)
    assert (interval(1, 1.5) == interval(1, 2)) == (None, True)
    assert (interval(0, 1) == interval(2, 3)) == (False, True)
    assert (interval(0, 1) == interval(1, 2)) == (None, True)
    assert (interval(1, 2) != interval(1, 2)) == (False, True)
    assert (interval(1, 3) != interval(2, 3)) == (None, True)
    assert (interval(1, 3) != interval(-5, -3)) == (True, True)
    assert (
        interval(1, 3, is_valid=False) != interval(-5, -3)) == (True, False)
    assert (interval(1, 3, is_valid=None) != interval(-5, 3)) == (None, None)
    assert (interval(4, 4) != 4) == (False, True)
    assert (interval(1, 1) == 1) == (True, True)
    assert (interval(1, 3, is_valid=False) == interval(1, 3)) == (True, False)
    assert (interval(1, 3, is_valid=None) == interval(1, 3)) == (True, None)
    inter = interval(-5, 5)
    assert (interval(inter) == interval(-5, 5)) == (True, True)
    assert inter.width == 10
    assert 0 in inter
    assert -5 in inter
    assert 5 in inter
    assert interval(0, 3) in inter
    assert interval(-6, 2) not in inter
    assert -5.05 not in inter
    assert 5.3 not in inter
    interb = interval(-float('inf'), float('inf'))
    assert 0 in inter
    assert inter in interb
    assert interval(0, float('inf')) in interb
    assert interval(-float('inf'), 5) in interb
    assert interval(-1e50, 1e50) in interb
    assert (
        -interval(-1, -2, is_valid=False) == interval(1, 2)) == (True, False)
    raises(ValueError, lambda: interval(1, 2, 3))


def test_interval_add():
    assert (interval(1, 2) + interval(2, 3) == interval(3, 5)) == (True, True)
    assert (1 + interval(1, 2) == interval(2, 3)) == (True, True)
    assert (interval(1, 2) + 1 == interval(2, 3)) == (True, True)
    compare = (1 + interval(0, float('inf')) == interval(1, float('inf')))
    assert compare == (True, True)
    a = 1 + interval(2, 5, is_valid=False)
    assert a.is_valid is False
    a = 1 + interval(2, 5, is_valid=None)
    assert a.is_valid is None
    a = interval(2, 5, is_valid=False) + interval(3, 5, is_valid=None)
    assert a.is_valid is False
    a = interval(3, 5) + interval(-1, 1, is_valid=None)
    assert a.is_valid is None
    a = interval(2, 5, is_valid=False) + 1
    assert a.is_valid is False


def test_interval_sub():
    assert (interval(1, 2) - interval(1, 5) == interval(-4, 1)) == (True, True)
    assert (interval(1, 2) - 1 == interval(0, 1)) == (True, True)
    assert (1 - interval(1, 2) == interval(-1, 0)) == (True, True)
    a = 1 - interval(1, 2, is_valid=False)
    assert a.is_valid is False
    a = interval(1, 4, is_valid=None) - 1
    assert a.is_valid is None
    a = interval(1, 3, is_valid=False) - interval(1, 3)
    assert a.is_valid is False
    a = interval(1, 3, is_valid=None) - interval(1, 3)
    assert a.is_valid is None


def test_interval_inequality():
    assert (interval(1, 2) < interval(3, 4)) == (True, True)
    assert (interval(1, 2) < interval(2, 4)) == (None, True)
    assert (interval(1, 2) < interval(-2, 0)) == (False, True)
    assert (interval(1, 2) <= interval(2, 4)) == (True, True)
    assert (interval(1, 2) <= interval(1.5, 6)) == (None, True)
    assert (interval(2, 3) <= interval(1, 2)) == (None, True)
    assert (interval(2, 3) <= interval(1, 1.5)) == (False, True)
    assert (
        interval(1, 2, is_valid=False) <= interval(-2, 0)) == (False, False)
    assert (interval(1, 2, is_valid=None) <= interval(-2, 0)) == (False, None)
    assert (interval(1, 2) <= 1.5) == (None, True)
    assert (interval(1, 2) <= 3) == (True, True)
    assert (interval(1, 2) <= 0) == (False, True)
    assert (interval(5, 8) > interval(2, 3)) == (True, True)
    assert (interval(2, 5) > interval(1, 3)) == (None, True)
    assert (interval(2, 3) > interval(3.1, 5)) == (False, True)

    assert (interval(-1, 1) == 0) == (None, True)
    assert (interval(-1, 1) == 2) == (False, True)
    assert (interval(-1, 1) != 0) == (None, True)
    assert (interval(-1, 1) != 2) == (True, True)

    assert (interval(3, 5) > 2) == (True, True)
    assert (interval(3, 5) < 2) == (False, True)
    assert (interval(1, 5) < 2) == (None, True)
    assert (interval(1, 5) > 2) == (None, True)
    assert (interval(0, 1) > 2) == (False, True)
    assert (interval(1, 2) >= interval(0, 1)) == (True, True)
    assert (interval(1, 2) >= interval(0, 1.5)) == (None, True)
    assert (interval(1, 2) >= interval(3, 4)) == (False, True)
    assert (interval(1, 2) >= 0) == (True, True)
    assert (interval(1, 2) >= 1.2) == (None, True)
    assert (interval(1, 2) >= 3) == (False, True)
    assert (2 > interval(0, 1)) == (True, True)
    a = interval(-1, 1, is_valid=False) < interval(2, 5, is_valid=None)
    assert a == (True, False)
    a = interval(-1, 1, is_valid=None) < interval(2, 5, is_valid=False)
    assert a == (True, False)
    a = interval(-1, 1, is_valid=None) < interval(2, 5, is_valid=None)
    assert a == (True, None)
    a = interval(-1, 1, is_valid=False) > interval(-5, -2, is_valid=None)
    assert a == (True, False)
    a = interval(-1, 1, is_valid=None) > interval(-5, -2, is_valid=False)
    assert a == (True, False)
    a = interval(-1, 1, is_valid=None) > interval(-5, -2, is_valid=None)
    assert a == (True, None)


def test_interval_mul():
    assert (
        interval(1, 5) * interval(2, 10) == interval(2, 50)) == (True, True)
    a = interval(-1, 1) * interval(2, 10) == interval(-10, 10)
    assert a == (True, True)

    a = interval(-1, 1) * interval(-5, 3) == interval(-5, 5)
    assert a == (True, True)

    assert (interval(1, 3) * 2 == interval(2, 6)) == (True, True)
    assert (3 * interval(-1, 2) == interval(-3, 6)) == (True, True)

    a = 3 * interval(1, 2, is_valid=False)
    assert a.is_valid is False

    a = 3 * interval(1, 2, is_valid=None)
    assert a.is_valid is None

    a = interval(1, 5, is_valid=False) * interval(1, 2, is_valid=None)
    assert a.is_valid is False


def test_interval_div():
    div = interval(1, 2, is_valid=False) / 3
    assert div == interval(-float('inf'), float('inf'), is_valid=False)

    div = interval(1, 2, is_valid=None) / 3
    assert div == interval(-float('inf'), float('inf'), is_valid=None)

    div = 3 / interval(1, 2, is_valid=None)
    assert div == interval(-float('inf'), float('inf'), is_valid=None)
    a = interval(1, 2) / 0
    assert a.is_valid is False
    a = interval(0.5, 1) / interval(-1, 0)
    assert a.is_valid is None
    a = interval(0, 1) / interval(0, 1)
    assert a.is_valid is None

    a = interval(-1, 1) / interval(-1, 1)
    assert a.is_valid is None

    a = interval(-1, 2) / interval(0.5, 1) == interval(-2.0, 4.0)
    assert a == (True, True)
    a = interval(0, 1) / interval(0.5, 1) == interval(0.0, 2.0)
    assert a == (True, True)
    a = interval(-1, 0) / interval(0.5, 1) == interval(-2.0, 0.0)
    assert a == (True, True)
    a = interval(-0.5, -0.25) / interval(0.5, 1) == interval(-1.0, -0.25)
    assert a == (True, True)
    a = interval(0.5, 1) / interval(0.5, 1) == interval(0.5, 2.0)
    assert a == (True, True)
    a = interval(0.5, 4) / interval(0.5, 1) == interval(0.5, 8.0)
    assert a == (True, True)
    a = interval(-1, -0.5) / interval(0.5, 1) == interval(-2.0, -0.5)
    assert a == (True, True)
    a = interval(-4, -0.5) / interval(0.5, 1) == interval(-8.0, -0.5)
    assert a == (True, True)
    a = interval(-1, 2) / interval(-2, -0.5) == interval(-4.0, 2.0)
    assert a == (True, True)
    a = interval(0, 1) / interval(-2, -0.5) == interval(-2.0, 0.0)
    assert a == (True, True)
    a = interval(-1, 0) / interval(-2, -0.5) == interval(0.0, 2.0)
    assert a == (True, True)
    a = interval(-0.5, -0.25) / interval(-2, -0.5) == interval(0.125, 1.0)
    assert a == (True, True)
    a = interval(0.5, 1) / interval(-2, -0.5) == interval(-2.0, -0.25)
    assert a == (True, True)
    a = interval(0.5, 4) / interval(-2, -0.5) == interval(-8.0, -0.25)
    assert a == (True, True)
    a = interval(-1, -0.5) / interval(-2, -0.5) == interval(0.25, 2.0)
    assert a == (True, True)
    a = interval(-4, -0.5) / interval(-2, -0.5) == interval(0.25, 8.0)
    assert a == (True, True)
    a = interval(-5, 5, is_valid=False) / 2
    assert a.is_valid is False

def test_hashable():
    '''
    test that interval objects are hashable.
    this is required in order to be able to put them into the cache, which
    appears to be necessary for plotting in py3k. For details, see:

    https://github.com/sympy/sympy/pull/2101
    https://github.com/sympy/sympy/issues/6533
    '''
    hash(interval(1, 1))
    hash(interval(1, 1, is_valid=True))
    hash(interval(-4, -0.5))
    hash(interval(-2, -0.5))
    hash(interval(0.25, 8.0))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_cumulative.py ===
"""
Tests for Series cumulative operations.

See also
--------
tests.frame.test_cumulative
"""

import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

methods = {
    "cumsum": np.cumsum,
    "cumprod": np.cumprod,
    "cummin": np.minimum.accumulate,
    "cummax": np.maximum.accumulate,
}


class TestSeriesCumulativeOps:
    @pytest.mark.parametrize("func", [np.cumsum, np.cumprod])
    def test_datetime_series(self, datetime_series, func):
        tm.assert_numpy_array_equal(
            func(datetime_series).values,
            func(np.array(datetime_series)),
            check_dtype=True,
        )

        # with missing values
        ts = datetime_series.copy()
        ts[::2] = np.nan

        result = func(ts)[1::2]
        expected = func(np.array(ts.dropna()))

        tm.assert_numpy_array_equal(result.values, expected, check_dtype=False)

    @pytest.mark.parametrize("method", ["cummin", "cummax"])
    def test_cummin_cummax(self, datetime_series, method):
        ufunc = methods[method]

        result = getattr(datetime_series, method)().values
        expected = ufunc(np.array(datetime_series))

        tm.assert_numpy_array_equal(result, expected)
        ts = datetime_series.copy()
        ts[::2] = np.nan
        result = getattr(ts, method)()[1::2]
        expected = ufunc(ts.dropna())

        result.index = result.index._with_freq(None)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ts",
        [
            pd.Timedelta(0),
            pd.Timestamp("1999-12-31"),
            pd.Timestamp("1999-12-31").tz_localize("US/Pacific"),
        ],
    )
    @pytest.mark.parametrize(
        "method, skipna, exp_tdi",
        [
            ["cummax", True, ["NaT", "2 days", "NaT", "2 days", "NaT", "3 days"]],
            ["cummin", True, ["NaT", "2 days", "NaT", "1 days", "NaT", "1 days"]],
            [
                "cummax",
                False,
                ["NaT", "NaT", "NaT", "NaT", "NaT", "NaT"],
            ],
            [
                "cummin",
                False,
                ["NaT", "NaT", "NaT", "NaT", "NaT", "NaT"],
            ],
        ],
    )
    def test_cummin_cummax_datetimelike(self, ts, method, skipna, exp_tdi):
        # with ts==pd.Timedelta(0), we are testing td64; with naive Timestamp
        #  we are testing datetime64[ns]; with Timestamp[US/Pacific]
        #  we are testing dt64tz
        tdi = pd.to_timedelta(["NaT", "2 days", "NaT", "1 days", "NaT", "3 days"])
        ser = pd.Series(tdi + ts)

        exp_tdi = pd.to_timedelta(exp_tdi)
        expected = pd.Series(exp_tdi + ts)
        result = getattr(ser, method)(skipna=skipna)
        tm.assert_series_equal(expected, result)

    @pytest.mark.parametrize(
        "func, exp",
        [
            ("cummin", pd.Period("2012-1-1", freq="D")),
            ("cummax", pd.Period("2012-1-2", freq="D")),
        ],
    )
    def test_cummin_cummax_period(self, func, exp):
        # GH#28385
        ser = pd.Series(
            [pd.Period("2012-1-1", freq="D"), pd.NaT, pd.Period("2012-1-2", freq="D")]
        )
        result = getattr(ser, func)(skipna=False)
        expected = pd.Series([pd.Period("2012-1-1", freq="D"), pd.NaT, pd.NaT])
        tm.assert_series_equal(result, expected)

        result = getattr(ser, func)(skipna=True)
        expected = pd.Series([pd.Period("2012-1-1", freq="D"), pd.NaT, exp])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "arg",
        [
            [False, False, False, True, True, False, False],
            [False, False, False, False, False, False, False],
        ],
    )
    @pytest.mark.parametrize(
        "func", [lambda x: x, lambda x: ~x], ids=["identity", "inverse"]
    )
    @pytest.mark.parametrize("method", methods.keys())
    def test_cummethods_bool(self, arg, func, method):
        # GH#6270
        # checking Series method vs the ufunc applied to the values

        ser = func(pd.Series(arg))
        ufunc = methods[method]

        exp_vals = ufunc(ser.values)
        expected = pd.Series(exp_vals)

        result = getattr(ser, method)()

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "method, expected",
        [
            ["cumsum", pd.Series([0, 1, np.nan, 1], dtype=object)],
            ["cumprod", pd.Series([False, 0, np.nan, 0])],
            ["cummin", pd.Series([False, False, np.nan, False])],
            ["cummax", pd.Series([False, True, np.nan, True])],
        ],
    )
    def test_cummethods_bool_in_object_dtype(self, method, expected):
        ser = pd.Series([False, True, np.nan, False])
        result = getattr(ser, method)()
        tm.assert_series_equal(result, expected)

    def test_cumprod_timedelta(self):
        # GH#48111
        ser = pd.Series([pd.Timedelta(days=1), pd.Timedelta(days=3)])
        with pytest.raises(TypeError, match="cumprod not supported for Timedelta"):
            ser.cumprod()

    @pytest.mark.parametrize(
        "data, op, skipna, expected_data",
        [
            ([], "cumsum", True, []),
            ([], "cumsum", False, []),
            (["x", "z", "y"], "cumsum", True, ["x", "xz", "xzy"]),
            (["x", "z", "y"], "cumsum", False, ["x", "xz", "xzy"]),
            (["x", pd.NA, "y"], "cumsum", True, ["x", pd.NA, "xy"]),
            (["x", pd.NA, "y"], "cumsum", False, ["x", pd.NA, pd.NA]),
            ([pd.NA, "x", "y"], "cumsum", True, [pd.NA, "x", "xy"]),
            ([pd.NA, "x", "y"], "cumsum", False, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cumsum", True, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cumsum", False, [pd.NA, pd.NA, pd.NA]),
            ([], "cummin", True, []),
            ([], "cummin", False, []),
            (["y", "z", "x"], "cummin", True, ["y", "y", "x"]),
            (["y", "z", "x"], "cummin", False, ["y", "y", "x"]),
            (["y", pd.NA, "x"], "cummin", True, ["y", pd.NA, "x"]),
            (["y", pd.NA, "x"], "cummin", False, ["y", pd.NA, pd.NA]),
            ([pd.NA, "y", "x"], "cummin", True, [pd.NA, "y", "x"]),
            ([pd.NA, "y", "x"], "cummin", False, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cummin", True, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cummin", False, [pd.NA, pd.NA, pd.NA]),
            ([], "cummax", True, []),
            ([], "cummax", False, []),
            (["x", "z", "y"], "cummax", True, ["x", "z", "z"]),
            (["x", "z", "y"], "cummax", False, ["x", "z", "z"]),
            (["x", pd.NA, "y"], "cummax", True, ["x", pd.NA, "y"]),
            (["x", pd.NA, "y"], "cummax", False, ["x", pd.NA, pd.NA]),
            ([pd.NA, "x", "y"], "cummax", True, [pd.NA, "x", "y"]),
            ([pd.NA, "x", "y"], "cummax", False, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cummax", True, [pd.NA, pd.NA, pd.NA]),
            ([pd.NA, pd.NA, pd.NA], "cummax", False, [pd.NA, pd.NA, pd.NA]),
        ],
    )
    def test_cum_methods_ea_strings(
        self, string_dtype_no_object, data, op, skipna, expected_data
    ):
        # https://github.com/pandas-dev/pandas/pull/60633 - pyarrow
        # https://github.com/pandas-dev/pandas/pull/60938 - Python
        ser = pd.Series(data, dtype=string_dtype_no_object)
        method = getattr(ser, op)
        expected = pd.Series(expected_data, dtype=string_dtype_no_object)
        result = method(skipna=skipna)
        tm.assert_series_equal(result, expected)

    def test_cumprod_pyarrow_strings(self, pyarrow_string_dtype, skipna):
        # https://github.com/pandas-dev/pandas/pull/60633
        ser = pd.Series(list("xyz"), dtype=pyarrow_string_dtype)
        msg = re.escape(f"operation 'cumprod' not supported for dtype '{ser.dtype}'")
        with pytest.raises(TypeError, match=msg):
            ser.cumprod(skipna=skipna)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_compression.py ===
"""
Tests compressed data parsing functionality for all
of the parsers defined in parsers.py
"""

import os
from pathlib import Path
import tarfile
import zipfile

import pytest

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture(params=[True, False])
def buffer(request):
    return request.param


@pytest.fixture
def parser_and_data(all_parsers, csv1):
    parser = all_parsers

    with open(csv1, "rb") as f:
        data = f.read()
    expected = parser.read_csv(csv1)

    return parser, data, expected


@pytest.mark.parametrize("compression", ["zip", "infer", "zip2"])
def test_zip(parser_and_data, compression):
    parser, data, expected = parser_and_data

    with tm.ensure_clean("test_file.zip") as path:
        with zipfile.ZipFile(path, mode="w") as tmp:
            tmp.writestr("test_file", data)

        if compression == "zip2":
            with open(path, "rb") as f:
                result = parser.read_csv(f, compression="zip")
        else:
            result = parser.read_csv(path, compression=compression)

        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("compression", ["zip", "infer"])
def test_zip_error_multiple_files(parser_and_data, compression):
    parser, data, expected = parser_and_data

    with tm.ensure_clean("combined_zip.zip") as path:
        inner_file_names = ["test_file", "second_file"]

        with zipfile.ZipFile(path, mode="w") as tmp:
            for file_name in inner_file_names:
                tmp.writestr(file_name, data)

        with pytest.raises(ValueError, match="Multiple files"):
            parser.read_csv(path, compression=compression)


def test_zip_error_no_files(parser_and_data):
    parser, _, _ = parser_and_data

    with tm.ensure_clean() as path:
        with zipfile.ZipFile(path, mode="w"):
            pass

        with pytest.raises(ValueError, match="Zero files"):
            parser.read_csv(path, compression="zip")


def test_zip_error_invalid_zip(parser_and_data):
    parser, _, _ = parser_and_data

    with tm.ensure_clean() as path:
        with open(path, "rb") as f:
            with pytest.raises(zipfile.BadZipFile, match="File is not a zip file"):
                parser.read_csv(f, compression="zip")


@pytest.mark.parametrize("filename", [None, "test.{ext}"])
def test_compression(
    request,
    parser_and_data,
    compression_only,
    buffer,
    filename,
    compression_to_extension,
):
    parser, data, expected = parser_and_data
    compress_type = compression_only

    ext = compression_to_extension[compress_type]
    filename = filename if filename is None else filename.format(ext=ext)

    if filename and buffer:
        request.applymarker(
            pytest.mark.xfail(
                reason="Cannot deduce compression from buffer of compressed data."
            )
        )

    with tm.ensure_clean(filename=filename) as path:
        tm.write_to_compressed(compress_type, path, data)
        compression = "infer" if filename else compress_type

        if buffer:
            with open(path, "rb") as f:
                result = parser.read_csv(f, compression=compression)
        else:
            result = parser.read_csv(path, compression=compression)

        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ext", [None, "gz", "bz2"])
def test_infer_compression(all_parsers, csv1, buffer, ext):
    # see gh-9770
    parser = all_parsers
    kwargs = {"index_col": 0, "parse_dates": True}

    expected = parser.read_csv(csv1, **kwargs)
    kwargs["compression"] = "infer"

    if buffer:
        with open(csv1, encoding="utf-8") as f:
            result = parser.read_csv(f, **kwargs)
    else:
        ext = "." + ext if ext else ""
        result = parser.read_csv(csv1 + ext, **kwargs)

    tm.assert_frame_equal(result, expected)


def test_compression_utf_encoding(all_parsers, csv_dir_path, utf_value, encoding_fmt):
    # see gh-18071, gh-24130
    parser = all_parsers
    encoding = encoding_fmt.format(utf_value)
    path = os.path.join(csv_dir_path, f"utf{utf_value}_ex_small.zip")

    result = parser.read_csv(path, encoding=encoding, compression="zip", sep="\t")
    expected = DataFrame(
        {
            "Country": ["Venezuela", "Venezuela"],
            "Twitter": ["Hugo Chávez Frías", "Henrique Capriles R."],
        }
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("invalid_compression", ["sfark", "bz3", "zipper"])
def test_invalid_compression(all_parsers, invalid_compression):
    parser = all_parsers
    compress_kwargs = {"compression": invalid_compression}

    msg = f"Unrecognized compression type: {invalid_compression}"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv("test_file.zip", **compress_kwargs)


def test_compression_tar_archive(all_parsers, csv_dir_path):
    parser = all_parsers
    path = os.path.join(csv_dir_path, "tar_csv.tar.gz")
    df = parser.read_csv(path)
    assert list(df.columns) == ["a"]


def test_ignore_compression_extension(all_parsers):
    parser = all_parsers
    df = DataFrame({"a": [0, 1]})
    with tm.ensure_clean("test.csv") as path_csv:
        with tm.ensure_clean("test.csv.zip") as path_zip:
            # make sure to create un-compressed file with zip extension
            df.to_csv(path_csv, index=False)
            Path(path_zip).write_text(
                Path(path_csv).read_text(encoding="utf-8"), encoding="utf-8"
            )

            tm.assert_frame_equal(parser.read_csv(path_zip, compression=None), df)


def test_writes_tar_gz(all_parsers):
    parser = all_parsers
    data = DataFrame(
        {
            "Country": ["Venezuela", "Venezuela"],
            "Twitter": ["Hugo Chávez Frías", "Henrique Capriles R."],
        }
    )
    with tm.ensure_clean("test.tar.gz") as tar_path:
        data.to_csv(tar_path, index=False)

        # test that read_csv infers .tar.gz to gzip:
        tm.assert_frame_equal(parser.read_csv(tar_path), data)

        # test that file is indeed gzipped:
        with tarfile.open(tar_path, "r:gz") as tar:
            result = parser.read_csv(
                tar.extractfile(tar.getnames()[0]), compression="infer"
            )
            tm.assert_frame_equal(result, data)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\tests\test_pytest.py ===
import warnings

from sympy.testing.pytest import (raises, warns, ignore_warnings,
                                    warns_deprecated_sympy, Failed)
from sympy.utilities.exceptions import sympy_deprecation_warning



# Test callables


def test_expected_exception_is_silent_callable():
    def f():
        raise ValueError()
    raises(ValueError, f)


# Under pytest raises will raise Failed rather than AssertionError
def test_lack_of_exception_triggers_AssertionError_callable():
    try:
        raises(Exception, lambda: 1 + 1)
        assert False
    except Failed as e:
        assert "DID NOT RAISE" in str(e)


def test_unexpected_exception_is_passed_through_callable():
    def f():
        raise ValueError("some error message")
    try:
        raises(TypeError, f)
        assert False
    except ValueError as e:
        assert str(e) == "some error message"

# Test with statement

def test_expected_exception_is_silent_with():
    with raises(ValueError):
        raise ValueError()


def test_lack_of_exception_triggers_AssertionError_with():
    try:
        with raises(Exception):
            1 + 1
        assert False
    except Failed as e:
        assert "DID NOT RAISE" in str(e)


def test_unexpected_exception_is_passed_through_with():
    try:
        with raises(TypeError):
            raise ValueError("some error message")
        assert False
    except ValueError as e:
        assert str(e) == "some error message"

# Now we can use raises() instead of try/catch
# to test that a specific exception class is raised


def test_second_argument_should_be_callable_or_string():
    raises(TypeError, lambda: raises("irrelevant", 42))


def test_warns_catches_warning():
    with warnings.catch_warnings(record=True) as w:
        with warns(UserWarning):
            warnings.warn('this is the warning message')
        assert len(w) == 0


def test_warns_raises_without_warning():
    with raises(Failed):
        with warns(UserWarning):
            pass


def test_warns_hides_other_warnings():
    with raises(RuntimeWarning):
        with warns(UserWarning):
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)


def test_warns_continues_after_warning():
    with warnings.catch_warnings(record=True) as w:
        finished = False
        with warns(UserWarning):
            warnings.warn('this is the warning message')
            finished = True
        assert finished
        assert len(w) == 0


def test_warns_many_warnings():
    with warns(UserWarning):
        warnings.warn('this is the warning message', UserWarning)
        warnings.warn('this is the other warning message', UserWarning)


def test_warns_match_matching():
    with warnings.catch_warnings(record=True) as w:
        with warns(UserWarning, match='this is the warning message'):
            warnings.warn('this is the warning message', UserWarning)
        assert len(w) == 0


def test_warns_match_non_matching():
    with warnings.catch_warnings(record=True) as w:
        with raises(Failed):
            with warns(UserWarning, match='this is the warning message'):
                warnings.warn('this is not the expected warning message', UserWarning)
        assert len(w) == 0

def _warn_sympy_deprecation(stacklevel=3):
    sympy_deprecation_warning(
        "feature",
        active_deprecations_target="active-deprecations",
        deprecated_since_version="0.0.0",
        stacklevel=stacklevel,
    )

def test_warns_deprecated_sympy_catches_warning():
    with warnings.catch_warnings(record=True) as w:
        with warns_deprecated_sympy():
            _warn_sympy_deprecation()
        assert len(w) == 0


def test_warns_deprecated_sympy_raises_without_warning():
    with raises(Failed):
        with warns_deprecated_sympy():
            pass

def test_warns_deprecated_sympy_wrong_stacklevel():
    with raises(Failed):
        with warns_deprecated_sympy():
            _warn_sympy_deprecation(stacklevel=1)

def test_warns_deprecated_sympy_doesnt_hide_other_warnings():
    # Unlike pytest's deprecated_call, we should not hide other warnings.
    with raises(RuntimeWarning):
        with warns_deprecated_sympy():
            _warn_sympy_deprecation()
            warnings.warn('this is the other message', RuntimeWarning)


def test_warns_deprecated_sympy_continues_after_warning():
    with warnings.catch_warnings(record=True) as w:
        finished = False
        with warns_deprecated_sympy():
            _warn_sympy_deprecation()
            finished = True
        assert finished
        assert len(w) == 0

def test_ignore_ignores_warning():
    with warnings.catch_warnings(record=True) as w:
        with ignore_warnings(UserWarning):
            warnings.warn('this is the warning message')
        assert len(w) == 0


def test_ignore_does_not_raise_without_warning():
    with warnings.catch_warnings(record=True) as w:
        with ignore_warnings(UserWarning):
            pass
        assert len(w) == 0


def test_ignore_allows_other_warnings():
    with warnings.catch_warnings(record=True) as w:
        # This is needed when pytest is run as -Werror
        # the setting is reverted at the end of the catch_Warnings block.
        warnings.simplefilter("always")
        with ignore_warnings(UserWarning):
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)
        assert len(w) == 1
        assert isinstance(w[0].message, RuntimeWarning)
        assert str(w[0].message) == 'this is the other message'


def test_ignore_continues_after_warning():
    with warnings.catch_warnings(record=True) as w:
        finished = False
        with ignore_warnings(UserWarning):
            warnings.warn('this is the warning message')
            finished = True
        assert finished
        assert len(w) == 0


def test_ignore_many_warnings():
    with warnings.catch_warnings(record=True) as w:
        # This is needed when pytest is run as -Werror
        # the setting is reverted at the end of the catch_Warnings block.
        warnings.simplefilter("always")
        with ignore_warnings(UserWarning):
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)
            warnings.warn('this is the other message', RuntimeWarning)
        assert len(w) == 3
        for wi in w:
            assert isinstance(wi.message, RuntimeWarning)
            assert str(wi.message) == 'this is the other message'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\masked\test_arrow_compat.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


pa = pytest.importorskip("pyarrow")

from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask

arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES]
arrays += [pd.array([0.1, 0.2, 0.3, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES]
arrays += [pd.array([True, False, True, None], dtype="boolean")]


@pytest.fixture(params=arrays, ids=[a.dtype.name for a in arrays])
def data(request):
    """
    Fixture returning parametrized array from given dtype, including integer,
    float and boolean
    """
    return request.param


def test_arrow_array(data):
    arr = pa.array(data)
    expected = pa.array(
        data.to_numpy(object, na_value=None),
        type=pa.from_numpy_dtype(data.dtype.numpy_dtype),
    )
    assert arr.equals(expected)


def test_arrow_roundtrip(data):
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    assert table.field("a").type == str(data.dtype.numpy_dtype)

    result = table.to_pandas()
    assert result["a"].dtype == data.dtype
    tm.assert_frame_equal(result, df)


def test_dataframe_from_arrow_types_mapper():
    def types_mapper(arrow_type):
        if pa.types.is_boolean(arrow_type):
            return pd.BooleanDtype()
        elif pa.types.is_integer(arrow_type):
            return pd.Int64Dtype()

    bools_array = pa.array([True, None, False], type=pa.bool_())
    ints_array = pa.array([1, None, 2], type=pa.int64())
    small_ints_array = pa.array([-1, 0, 7], type=pa.int8())
    record_batch = pa.RecordBatch.from_arrays(
        [bools_array, ints_array, small_ints_array], ["bools", "ints", "small_ints"]
    )
    result = record_batch.to_pandas(types_mapper=types_mapper)
    bools = pd.Series([True, None, False], dtype="boolean")
    ints = pd.Series([1, None, 2], dtype="Int64")
    small_ints = pd.Series([-1, 0, 7], dtype="Int64")
    expected = pd.DataFrame({"bools": bools, "ints": ints, "small_ints": small_ints})
    tm.assert_frame_equal(result, expected)


def test_arrow_load_from_zero_chunks(data):
    # GH-41040

    df = pd.DataFrame({"a": data[0:0]})
    table = pa.table(df)
    assert table.field("a").type == str(data.dtype.numpy_dtype)
    table = pa.table(
        [pa.chunked_array([], type=table.field("a").type)], schema=table.schema
    )
    result = table.to_pandas()
    assert result["a"].dtype == data.dtype
    tm.assert_frame_equal(result, df)


def test_arrow_from_arrow_uint():
    # https://github.com/pandas-dev/pandas/issues/31896
    # possible mismatch in types

    dtype = pd.UInt32Dtype()
    result = dtype.__from_arrow__(pa.array([1, 2, 3, 4, None], type="int64"))
    expected = pd.array([1, 2, 3, 4, None], dtype="UInt32")

    tm.assert_extension_array_equal(result, expected)


def test_arrow_sliced(data):
    # https://github.com/pandas-dev/pandas/issues/38525

    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    result = table.slice(2, None).to_pandas()
    expected = df.iloc[2:].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)

    # no missing values
    df2 = df.fillna(data[0])
    table = pa.table(df2)
    result = table.slice(2, None).to_pandas()
    expected = df2.iloc[2:].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.fixture
def np_dtype_to_arrays(any_real_numpy_dtype):
    """
    Fixture returning actual and expected dtype, pandas and numpy arrays and
    mask from a given numpy dtype
    """
    np_dtype = np.dtype(any_real_numpy_dtype)
    pa_type = pa.from_numpy_dtype(np_dtype)

    # None ensures the creation of a bitmask buffer.
    pa_array = pa.array([0, 1, 2, None], type=pa_type)
    # Since masked Arrow buffer slots are not required to contain a specific
    # value, assert only the first three values of the created np.array
    np_expected = np.array([0, 1, 2], dtype=np_dtype)
    mask_expected = np.array([True, True, True, False])
    return np_dtype, pa_array, np_expected, mask_expected


def test_pyarrow_array_to_numpy_and_mask(np_dtype_to_arrays):
    """
    Test conversion from pyarrow array to numpy array.

    Modifies the pyarrow buffer to contain padding and offset, which are
    considered valid buffers by pyarrow.

    Also tests empty pyarrow arrays with non empty buffers.
    See https://github.com/pandas-dev/pandas/issues/40896
    """
    np_dtype, pa_array, np_expected, mask_expected = np_dtype_to_arrays
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)

    mask_buffer = pa_array.buffers()[0]
    data_buffer = pa_array.buffers()[1]
    data_buffer_bytes = pa_array.buffers()[1].to_pybytes()

    # Add trailing padding to the buffer.
    data_buffer_trail = pa.py_buffer(data_buffer_bytes + b"\x00")
    pa_array_trail = pa.Array.from_buffers(
        type=pa_array.type,
        length=len(pa_array),
        buffers=[mask_buffer, data_buffer_trail],
        offset=pa_array.offset,
    )
    pa_array_trail.validate()
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_trail, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)

    # Add offset to the buffer.
    offset = b"\x00" * (pa_array.type.bit_width // 8)
    data_buffer_offset = pa.py_buffer(offset + data_buffer_bytes)
    mask_buffer_offset = pa.py_buffer(b"\x0E")
    pa_array_offset = pa.Array.from_buffers(
        type=pa_array.type,
        length=len(pa_array),
        buffers=[mask_buffer_offset, data_buffer_offset],
        offset=pa_array.offset + 1,
    )
    pa_array_offset.validate()
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected)
    tm.assert_numpy_array_equal(mask, mask_expected)

    # Empty array
    np_expected_empty = np.array([], dtype=np_dtype)
    mask_expected_empty = np.array([], dtype=np.bool_)

    pa_array_offset = pa.Array.from_buffers(
        type=pa_array.type,
        length=0,
        buffers=[mask_buffer, data_buffer],
        offset=pa_array.offset,
    )
    pa_array_offset.validate()
    data, mask = pyarrow_array_to_numpy_and_mask(pa_array_offset, np_dtype)
    tm.assert_numpy_array_equal(data[:3], np_expected_empty)
    tm.assert_numpy_array_equal(mask, mask_expected_empty)


@pytest.mark.parametrize(
    "arr", [pa.nulls(10), pa.chunked_array([pa.nulls(4), pa.nulls(6)])]
)
def test_from_arrow_null(data, arr):
    res = data.dtype.__from_arrow__(arr)
    assert res.isna().all()
    assert len(res) == 10


def test_from_arrow_type_error(data):
    # ensure that __from_arrow__ returns a TypeError when getting a wrong
    # array type

    arr = pa.array(data).cast("string")
    with pytest.raises(TypeError, match=None):
        # we don't test the exact error message, only the fact that it raises
        # a TypeError is relevant
        data.dtype.__from_arrow__(arr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_transpose.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    Series,
    Timestamp,
    bdate_range,
    date_range,
    timedelta_range,
)
import pandas._testing as tm


class TestTranspose:
    def test_transpose_td64_intervals(self):
        # GH#44917
        tdi = timedelta_range("0 Days", "3 Days")
        ii = IntervalIndex.from_breaks(tdi)
        ii = ii.insert(-1, np.nan)
        df = DataFrame(ii)

        result = df.T
        expected = DataFrame({i: ii[i : i + 1] for i in range(len(ii))})
        tm.assert_frame_equal(result, expected)

    def test_transpose_empty_preserves_datetimeindex(self):
        # GH#41382
        dti = DatetimeIndex([], dtype="M8[ns]")
        df = DataFrame(index=dti)

        expected = DatetimeIndex([], dtype="datetime64[ns]", freq=None)

        result1 = df.T.sum().index
        result2 = df.sum(axis=1).index

        tm.assert_index_equal(result1, expected)
        tm.assert_index_equal(result2, expected)

    def test_transpose_tzaware_1col_single_tz(self):
        # GH#26825
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")

        df = DataFrame(dti)
        assert (df.dtypes == dti.dtype).all()
        res = df.T
        assert (res.dtypes == dti.dtype).all()

    def test_transpose_tzaware_2col_single_tz(self):
        # GH#26825
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")

        df3 = DataFrame({"A": dti, "B": dti})
        assert (df3.dtypes == dti.dtype).all()
        res3 = df3.T
        assert (res3.dtypes == dti.dtype).all()

    def test_transpose_tzaware_2col_mixed_tz(self):
        # GH#26825
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        dti2 = dti.tz_convert("US/Pacific")

        df4 = DataFrame({"A": dti, "B": dti2})
        assert (df4.dtypes == [dti.dtype, dti2.dtype]).all()
        assert (df4.T.dtypes == object).all()
        tm.assert_frame_equal(df4.T.T, df4.astype(object))

    @pytest.mark.parametrize("tz", [None, "America/New_York"])
    def test_transpose_preserves_dtindex_equality_with_dst(self, tz):
        # GH#19970
        idx = date_range("20161101", "20161130", freq="4h", tz=tz)
        df = DataFrame({"a": range(len(idx)), "b": range(len(idx))}, index=idx)
        result = df.T == df.T
        expected = DataFrame(True, index=list("ab"), columns=idx)
        tm.assert_frame_equal(result, expected)

    def test_transpose_object_to_tzaware_mixed_tz(self):
        # GH#26825
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        dti2 = dti.tz_convert("US/Pacific")

        # mixed all-tzaware dtypes
        df2 = DataFrame([dti, dti2])
        assert (df2.dtypes == object).all()
        res2 = df2.T
        assert (res2.dtypes == object).all()

    def test_transpose_uint64(self):
        df = DataFrame(
            {"A": np.arange(3), "B": [2**63, 2**63 + 5, 2**63 + 10]},
            dtype=np.uint64,
        )
        result = df.T
        expected = DataFrame(df.values.T)
        expected.index = ["A", "B"]
        tm.assert_frame_equal(result, expected)

    def test_transpose_float(self, float_frame):
        frame = float_frame
        dft = frame.T
        for idx, series in dft.items():
            for col, value in series.items():
                if np.isnan(value):
                    assert np.isnan(frame[col][idx])
                else:
                    assert value == frame[col][idx]

    def test_transpose_mixed(self):
        # mixed type
        mixed = DataFrame(
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
                "D": bdate_range("1/1/2009", periods=5),
            },
            index=Index(["a", "b", "c", "d", "e"], dtype=object),
        )

        mixed_T = mixed.T
        for col, s in mixed_T.items():
            assert s.dtype == np.object_

    @td.skip_array_manager_invalid_test
    def test_transpose_get_view(self, float_frame, using_copy_on_write):
        dft = float_frame.T
        dft.iloc[:, 5:10] = 5

        if using_copy_on_write:
            assert (float_frame.values[5:10] != 5).all()
        else:
            assert (float_frame.values[5:10] == 5).all()

    @td.skip_array_manager_invalid_test
    def test_transpose_get_view_dt64tzget_view(self, using_copy_on_write):
        dti = date_range("2016-01-01", periods=6, tz="US/Pacific")
        arr = dti._data.reshape(3, 2)
        df = DataFrame(arr)
        assert df._mgr.nblocks == 1

        result = df.T
        assert result._mgr.nblocks == 1

        rtrip = result._mgr.blocks[0].values
        if using_copy_on_write:
            assert np.shares_memory(df._mgr.blocks[0].values._ndarray, rtrip._ndarray)
        else:
            assert np.shares_memory(arr._ndarray, rtrip._ndarray)

    def test_transpose_not_inferring_dt(self):
        # GH#51546
        df = DataFrame(
            {
                "a": [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
            },
            dtype=object,
        )
        result = df.T
        expected = DataFrame(
            [[Timestamp("2019-12-31"), Timestamp("2019-12-31")]],
            columns=[0, 1],
            index=["a"],
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)

    def test_transpose_not_inferring_dt_mixed_blocks(self):
        # GH#51546
        df = DataFrame(
            {
                "a": Series(
                    [Timestamp("2019-12-31"), Timestamp("2019-12-31")], dtype=object
                ),
                "b": [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
            }
        )
        result = df.T
        expected = DataFrame(
            [
                [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
                [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
            ],
            columns=[0, 1],
            index=["a", "b"],
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype1", ["Int64", "Float64"])
    @pytest.mark.parametrize("dtype2", ["Int64", "Float64"])
    def test_transpose(self, dtype1, dtype2):
        # GH#57315 - transpose should have F contiguous blocks
        df = DataFrame(
            {
                "a": pd.array([1, 1, 2], dtype=dtype1),
                "b": pd.array([3, 4, 5], dtype=dtype2),
            }
        )
        result = df.T
        for blk in result._mgr.blocks:
            # When dtypes are unequal, we get NumPy object array
            data = blk.values._data if dtype1 == dtype2 else blk.values
            assert data.flags["F_CONTIGUOUS"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\generic\test_frame.py ===
from copy import deepcopy
from operator import methodcaller

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrame:
    @pytest.mark.parametrize("func", ["_set_axis_name", "rename_axis"])
    def test_set_axis_name(self, func):
        df = DataFrame([[1, 2], [3, 4]])

        result = methodcaller(func, "foo")(df)
        assert df.index.name is None
        assert result.index.name == "foo"

        result = methodcaller(func, "cols", axis=1)(df)
        assert df.columns.name is None
        assert result.columns.name == "cols"

    @pytest.mark.parametrize("func", ["_set_axis_name", "rename_axis"])
    def test_set_axis_name_mi(self, func):
        df = DataFrame(
            np.empty((3, 3)),
            index=MultiIndex.from_tuples([("A", x) for x in list("aBc")]),
            columns=MultiIndex.from_tuples([("C", x) for x in list("xyz")]),
        )

        level_names = ["L1", "L2"]

        result = methodcaller(func, level_names)(df)
        assert result.index.names == level_names
        assert result.columns.names == [None, None]

        result = methodcaller(func, level_names, axis=1)(df)
        assert result.columns.names == ["L1", "L2"]
        assert result.index.names == [None, None]

    def test_nonzero_single_element(self):
        # allow single item via bool method
        msg_warn = (
            "DataFrame.bool is now deprecated and will be removed "
            "in future version of pandas"
        )
        df = DataFrame([[True]])
        df1 = DataFrame([[False]])
        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            assert df.bool()

        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            assert not df1.bool()

        df = DataFrame([[False, False]])
        msg_err = "The truth value of a DataFrame is ambiguous"
        with pytest.raises(ValueError, match=msg_err):
            bool(df)

        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            with pytest.raises(ValueError, match=msg_err):
                df.bool()

    def test_metadata_propagation_indiv_groupby(self):
        # groupby
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.default_rng(2).standard_normal(8),
                "D": np.random.default_rng(2).standard_normal(8),
            }
        )
        result = df.groupby("A").sum()
        tm.assert_metadata_equivalent(df, result)

    def test_metadata_propagation_indiv_resample(self):
        # resample
        df = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 2)),
            index=date_range("20130101", periods=1000, freq="s"),
        )
        result = df.resample("1min")
        tm.assert_metadata_equivalent(df, result)

    def test_metadata_propagation_indiv(self, monkeypatch):
        # merging with override
        # GH 6923

        def finalize(self, other, method=None, **kwargs):
            for name in self._metadata:
                if method == "merge":
                    left, right = other.left, other.right
                    value = getattr(left, name, "") + "|" + getattr(right, name, "")
                    object.__setattr__(self, name, value)
                elif method == "concat":
                    value = "+".join(
                        [getattr(o, name) for o in other.objs if getattr(o, name, None)]
                    )
                    object.__setattr__(self, name, value)
                else:
                    object.__setattr__(self, name, getattr(other, name, ""))

            return self

        with monkeypatch.context() as m:
            m.setattr(DataFrame, "_metadata", ["filename"])
            m.setattr(DataFrame, "__finalize__", finalize)

            df1 = DataFrame(
                np.random.default_rng(2).integers(0, 4, (3, 2)), columns=["a", "b"]
            )
            df2 = DataFrame(
                np.random.default_rng(2).integers(0, 4, (3, 2)), columns=["c", "d"]
            )
            DataFrame._metadata = ["filename"]
            df1.filename = "fname1.csv"
            df2.filename = "fname2.csv"

            result = df1.merge(df2, left_on=["a"], right_on=["c"], how="inner")
            assert result.filename == "fname1.csv|fname2.csv"

            # concat
            # GH#6927
            df1 = DataFrame(
                np.random.default_rng(2).integers(0, 4, (3, 2)), columns=list("ab")
            )
            df1.filename = "foo"

            result = pd.concat([df1, df1])
            assert result.filename == "foo+foo"

    def test_set_attribute(self):
        # Test for consistent setattr behavior when an attribute and a column
        # have the same name (Issue #8994)
        df = DataFrame({"x": [1, 2, 3]})

        df.y = 2
        df["y"] = [2, 4, 6]
        df.y = 5

        assert df.y == 5
        tm.assert_series_equal(df["y"], Series([2, 4, 6], name="y"))

    def test_deepcopy_empty(self):
        # This test covers empty frame copying with non-empty column sets
        # as reported in issue GH15370
        empty_frame = DataFrame(data=[], index=[], columns=["A"])
        empty_frame_copy = deepcopy(empty_frame)

        tm.assert_frame_equal(empty_frame_copy, empty_frame)


# formerly in Generic but only test DataFrame
class TestDataFrame2:
    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        msg = 'For argument "inplace" expected type bool, received type'
        with pytest.raises(ValueError, match=msg):
            df.copy().rename_axis(mapper={"a": "x", "b": "y"}, axis=1, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().drop("a", axis=1, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().fillna(value=0, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().replace(to_replace=1, value=7, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().interpolate(inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy()._where(cond=df.a > 2, inplace=value)

        with pytest.raises(ValueError, match=msg):
            df.copy().mask(cond=df.a > 2, inplace=value)

    def test_unexpected_keyword(self):
        # GH8597
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=["jim", "joe"]
        )
        ca = pd.Categorical([0, 0, 2, 2, 3, np.nan])
        ts = df["joe"].copy()
        ts[2] = np.nan

        msg = "unexpected keyword"
        with pytest.raises(TypeError, match=msg):
            df.drop("joe", axis=1, in_place=True)

        with pytest.raises(TypeError, match=msg):
            df.reindex([1, 0], inplace=True)

        with pytest.raises(TypeError, match=msg):
            ca.fillna(0, inplace=True)

        with pytest.raises(TypeError, match=msg):
            ts.fillna(0, in_place=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\categories\tests\test_baseclasses.py ===
from sympy.categories import (Object, Morphism, IdentityMorphism,
                              NamedMorphism, CompositeMorphism,
                              Diagram, Category)
from sympy.categories.baseclasses import Class
from sympy.testing.pytest import raises
from sympy.core.containers import (Dict, Tuple)
from sympy.sets import EmptySet
from sympy.sets.sets import FiniteSet


def test_morphisms():
    A = Object("A")
    B = Object("B")
    C = Object("C")
    D = Object("D")

    # Test the base morphism.
    f = NamedMorphism(A, B, "f")
    assert f.domain == A
    assert f.codomain == B
    assert f == NamedMorphism(A, B, "f")

    # Test identities.
    id_A = IdentityMorphism(A)
    id_B = IdentityMorphism(B)
    assert id_A.domain == A
    assert id_A.codomain == A
    assert id_A == IdentityMorphism(A)
    assert id_A != id_B

    # Test named morphisms.
    g = NamedMorphism(B, C, "g")
    assert g.name == "g"
    assert g != f
    assert g == NamedMorphism(B, C, "g")
    assert g != NamedMorphism(B, C, "f")

    # Test composite morphisms.
    assert f == CompositeMorphism(f)

    k = g.compose(f)
    assert k.domain == A
    assert k.codomain == C
    assert k.components == Tuple(f, g)
    assert g * f == k
    assert CompositeMorphism(f, g) == k

    assert CompositeMorphism(g * f) == g * f

    # Test the associativity of composition.
    h = NamedMorphism(C, D, "h")

    p = h * g
    u = h * g * f

    assert h * k == u
    assert p * f == u
    assert CompositeMorphism(f, g, h) == u

    # Test flattening.
    u2 = u.flatten("u")
    assert isinstance(u2, NamedMorphism)
    assert u2.name == "u"
    assert u2.domain == A
    assert u2.codomain == D

    # Test identities.
    assert f * id_A == f
    assert id_B * f == f
    assert id_A * id_A == id_A
    assert CompositeMorphism(id_A) == id_A

    # Test bad compositions.
    raises(ValueError, lambda: f * g)

    raises(TypeError, lambda: f.compose(None))
    raises(TypeError, lambda: id_A.compose(None))
    raises(TypeError, lambda: f * None)
    raises(TypeError, lambda: id_A * None)

    raises(TypeError, lambda: CompositeMorphism(f, None, 1))

    raises(ValueError, lambda: NamedMorphism(A, B, ""))
    raises(NotImplementedError, lambda: Morphism(A, B))


def test_diagram():
    A = Object("A")
    B = Object("B")
    C = Object("C")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    id_A = IdentityMorphism(A)
    id_B = IdentityMorphism(B)

    empty = EmptySet

    # Test the addition of identities.
    d1 = Diagram([f])

    assert d1.objects == FiniteSet(A, B)
    assert d1.hom(A, B) == (FiniteSet(f), empty)
    assert d1.hom(A, A) == (FiniteSet(id_A), empty)
    assert d1.hom(B, B) == (FiniteSet(id_B), empty)

    assert d1 == Diagram([id_A, f])
    assert d1 == Diagram([f, f])

    # Test the addition of composites.
    d2 = Diagram([f, g])
    homAC = d2.hom(A, C)[0]

    assert d2.objects == FiniteSet(A, B, C)
    assert g * f in d2.premises.keys()
    assert homAC == FiniteSet(g * f)

    # Test equality, inequality and hash.
    d11 = Diagram([f])

    assert d1 == d11
    assert d1 != d2
    assert hash(d1) == hash(d11)

    d11 = Diagram({f: "unique"})
    assert d1 != d11

    # Make sure that (re-)adding composites (with new properties)
    # works as expected.
    d = Diagram([f, g], {g * f: "unique"})
    assert d.conclusions == Dict({g * f: FiniteSet("unique")})

    # Check the hom-sets when there are premises and conclusions.
    assert d.hom(A, C) == (FiniteSet(g * f), FiniteSet(g * f))
    d = Diagram([f, g], [g * f])
    assert d.hom(A, C) == (FiniteSet(g * f), FiniteSet(g * f))

    # Check how the properties of composite morphisms are computed.
    d = Diagram({f: ["unique", "isomorphism"], g: "unique"})
    assert d.premises[g * f] == FiniteSet("unique")

    # Check that conclusion morphisms with new objects are not allowed.
    d = Diagram([f], [g])
    assert d.conclusions == Dict({})

    # Test an empty diagram.
    d = Diagram()
    assert d.premises == Dict({})
    assert d.conclusions == Dict({})
    assert d.objects == empty

    # Check a SymPy Dict object.
    d = Diagram(Dict({f: FiniteSet("unique", "isomorphism"), g: "unique"}))
    assert d.premises[g * f] == FiniteSet("unique")

    # Check the addition of components of composite morphisms.
    d = Diagram([g * f])
    assert f in d.premises
    assert g in d.premises

    # Check subdiagrams.
    d = Diagram([f, g], {g * f: "unique"})

    d1 = Diagram([f])
    assert d.is_subdiagram(d1)
    assert not d1.is_subdiagram(d)

    d = Diagram([NamedMorphism(B, A, "f'")])
    assert not d.is_subdiagram(d1)
    assert not d1.is_subdiagram(d)

    d1 = Diagram([f, g], {g * f: ["unique", "something"]})
    assert not d.is_subdiagram(d1)
    assert not d1.is_subdiagram(d)

    d = Diagram({f: "blooh"})
    d1 = Diagram({f: "bleeh"})
    assert not d.is_subdiagram(d1)
    assert not d1.is_subdiagram(d)

    d = Diagram([f, g], {f: "unique", g * f: "veryunique"})
    d1 = d.subdiagram_from_objects(FiniteSet(A, B))
    assert d1 == Diagram([f], {f: "unique"})
    raises(ValueError, lambda: d.subdiagram_from_objects(FiniteSet(A,
           Object("D"))))

    raises(ValueError, lambda: Diagram({IdentityMorphism(A): "unique"}))


def test_category():
    A = Object("A")
    B = Object("B")
    C = Object("C")

    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")

    d1 = Diagram([f, g])
    d2 = Diagram([f])

    objects = d1.objects | d2.objects

    K = Category("K", objects, commutative_diagrams=[d1, d2])

    assert K.name == "K"
    assert K.objects == Class(objects)
    assert K.commutative_diagrams == FiniteSet(d1, d2)

    raises(ValueError, lambda: Category(""))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_sym_expr.py ===
from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises
from sympy.external import import_module

lfortran = import_module('lfortran')
cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

if lfortran and cin:
    from sympy.codegen.ast import (Variable, IntBaseType, FloatBaseType, String,
                                   Declaration, FloatType)
    from sympy.core import Integer, Float
    from sympy.core.symbol import Symbol

    expr1 = SymPyExpression()
    src = """\
    integer :: a, b, c, d
    real :: p, q, r, s
    """

    def test_c_parse():
        src1 = """\
        int a, b = 4;
        float c, d = 2.4;
        """
        expr1.convert_to_expr(src1, 'c')
        ls = expr1.return_expr()

        assert ls[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc'))
            )
        )
        assert ls[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(4)
            )
        )
        assert ls[2] == Declaration(
            Variable(
                Symbol('c'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    )
            )
        )
        assert ls[3] == Declaration(
            Variable(
                Symbol('d'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.3999999999999999', precision=53)
            )
        )


    def test_fortran_parse():
        expr = SymPyExpression(src, 'f')
        ls = expr.return_expr()

        assert ls[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls[2] == Declaration(
            Variable(
                Symbol('c'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls[3] == Declaration(
            Variable(
                Symbol('d'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls[4] == Declaration(
            Variable(
                Symbol('p'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )
        assert ls[5] == Declaration(
            Variable(
                Symbol('q'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )
        assert ls[6] == Declaration(
            Variable(
                Symbol('r'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )
        assert ls[7] == Declaration(
            Variable(
                Symbol('s'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )


    def test_convert_py():
        src1 = (
            src +
            """\
            a = b + c
            s = p * q / r
            """
        )
        expr1.convert_to_expr(src1, 'f')
        exp_py = expr1.convert_to_python()
        assert exp_py == [
            'a = 0',
            'b = 0',
            'c = 0',
            'd = 0',
            'p = 0.0',
            'q = 0.0',
            'r = 0.0',
            's = 0.0',
            'a = b + c',
            's = p*q/r'
        ]


    def test_convert_fort():
        src1 = (
            src +
            """\
            a = b + c
            s = p * q / r
            """
        )
        expr1.convert_to_expr(src1, 'f')
        exp_fort = expr1.convert_to_fortran()
        assert exp_fort == [
            '      integer*4 a',
            '      integer*4 b',
            '      integer*4 c',
            '      integer*4 d',
            '      real*8 p',
            '      real*8 q',
            '      real*8 r',
            '      real*8 s',
            '      a = b + c',
            '      s = p*q/r'
        ]


    def test_convert_c():
        src1 = (
            src +
            """\
            a = b + c
            s = p * q / r
            """
        )
        expr1.convert_to_expr(src1, 'f')
        exp_c = expr1.convert_to_c()
        assert exp_c == [
            'int a = 0',
            'int b = 0',
            'int c = 0',
            'int d = 0',
            'double p = 0.0',
            'double q = 0.0',
            'double r = 0.0',
            'double s = 0.0',
            'a = b + c;',
            's = p*q/r;'
        ]


    def test_exceptions():
        src = 'int a;'
        raises(ValueError, lambda: SymPyExpression(src))
        raises(ValueError, lambda: SymPyExpression(mode = 'c'))
        raises(NotImplementedError, lambda: SymPyExpression(src, mode = 'd'))

elif not lfortran and not cin:
    def test_raise():
        raises(ImportError, lambda: SymPyExpression('int a;', 'c'))
        raises(ImportError, lambda: SymPyExpression('integer :: a', 'f'))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_nullspace.py ===
from sympy import ZZ, Matrix
from sympy.polys.matrices import DM, DomainMatrix
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM

import pytest

zeros = lambda shape, K: DomainMatrix.zeros(shape, K).to_dense()
eye = lambda n, K: DomainMatrix.eye(n, K).to_dense()


#
# DomainMatrix.nullspace can have a divided answer or can return an undivided
# uncanonical answer. The uncanonical answer is not unique but we can make it
# unique by making it primitive (remove gcd). The tests here all show the
# primitive form. We test two things:
#
#   A.nullspace().primitive()[1] == answer.
#   A.nullspace(divide_last=True) == _divide_last(answer).
#
# The nullspace as returned by DomainMatrix and related classes is the
# transpose of the nullspace as returned by Matrix. Matrix returns a list of
# of column vectors whereas DomainMatrix returns a matrix whose rows are the
# nullspace vectors.
#


NULLSPACE_EXAMPLES = [

    (
        'zz_1',
         DM([[ 1, 2, 3]], ZZ),
         DM([[-2, 1, 0],
             [-3, 0, 1]], ZZ),
    ),

    (
        'zz_2',
         zeros((0, 0), ZZ),
         zeros((0, 0), ZZ),
    ),

    (
        'zz_3',
        zeros((2, 0), ZZ),
        zeros((0, 0), ZZ),
    ),

    (
        'zz_4',
        zeros((0, 2), ZZ),
        eye(2, ZZ),
    ),

    (
        'zz_5',
        zeros((2, 2), ZZ),
        eye(2, ZZ),
    ),

    (
        'zz_6',
        DM([[1, 2],
            [3, 4]], ZZ),
        zeros((0, 2), ZZ),
    ),

    (
        'zz_7',
        DM([[1, 1],
            [1, 1]], ZZ),
        DM([[-1, 1]], ZZ),
    ),

    (
        'zz_8',
        DM([[1],
            [1]], ZZ),
        zeros((0, 1), ZZ),
    ),

    (
        'zz_9',
        DM([[1, 1]], ZZ),
        DM([[-1, 1]], ZZ),
    ),

    (
        'zz_10',
        DM([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]], ZZ),
        DM([[ 0,  0, 1,  0,  0, 0, 0, 0, 0, 0],
            [-1,  0, 0,  0,  0, 0, 1, 0, 0, 0],
            [ 0, -1, 0,  0,  0, 0, 0, 1, 0, 0],
            [ 0,  0, 0, -1,  0, 0, 0, 0, 1, 0],
            [ 0,  0, 0,  0, -1, 0, 0, 0, 0, 1]], ZZ),
    ),

]


def _to_DM(A, ans):
    """Convert the answer to DomainMatrix."""
    if isinstance(A, DomainMatrix):
        return A.to_dense()
    elif isinstance(A, DDM):
        return DomainMatrix(list(A), A.shape, A.domain).to_dense()
    elif isinstance(A, SDM):
        return DomainMatrix(dict(A), A.shape, A.domain).to_dense()
    else:
        assert False # pragma: no cover


def _divide_last(null):
    """Normalize the nullspace by the rightmost non-zero entry."""
    null = null.to_field()

    if null.is_zero_matrix:
        return null

    rows = []
    for i in range(null.shape[0]):
        for j in reversed(range(null.shape[1])):
            if null[i, j]:
                rows.append(null[i, :] / null[i, j])
                break
        else:
            assert False # pragma: no cover

    return DomainMatrix.vstack(*rows)


def _check_primitive(null, null_ans):
    """Check that the primitive of the answer matches."""
    null = _to_DM(null, null_ans)
    cont, null_prim = null.primitive()
    assert null_prim == null_ans


def _check_divided(null, null_ans):
    """Check the divided answer."""
    null = _to_DM(null, null_ans)
    null_ans_norm = _divide_last(null_ans)
    assert null == null_ans_norm


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_Matrix_nullspace(name, A, A_null):
    A = A.to_Matrix()

    A_null_cols = A.nullspace()

    # We have to patch up the case where the nullspace is empty
    if A_null_cols:
        A_null_found = Matrix.hstack(*A_null_cols)
    else:
        A_null_found = Matrix.zeros(A.cols, 0)

    A_null_found = A_null_found.to_DM().to_field().to_dense()

    # The Matrix result is the transpose of DomainMatrix result.
    A_null_found = A_null_found.transpose()

    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_dense_nullspace(name, A, A_null):
    A = A.to_field().to_dense()
    A_null_found = A.nullspace(divide_last=True)
    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_sparse_nullspace(name, A, A_null):
    A = A.to_field().to_sparse()
    A_null_found = A.nullspace(divide_last=True)
    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_ddm_nullspace(name, A, A_null):
    A = A.to_field().to_ddm()
    A_null_found, _ = A.nullspace()
    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_sdm_nullspace(name, A, A_null):
    A = A.to_field().to_sdm()
    A_null_found, _ = A.nullspace()
    _check_divided(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_dense_nullspace_fracfree(name, A, A_null):
    A = A.to_dense()
    A_null_found = A.nullspace()
    _check_primitive(A_null_found, A_null)


@pytest.mark.parametrize('name, A, A_null', NULLSPACE_EXAMPLES)
def test_dm_sparse_nullspace_fracfree(name, A, A_null):
    A = A.to_sparse()
    A_null_found = A.nullspace()
    _check_primitive(A_null_found, A_null)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\conftest.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)
from pandas.core.groupby.base import (
    reduction_kernels,
    transformation_kernels,
)


@pytest.fixture(params=[True, False])
def sort(request):
    return request.param


@pytest.fixture(params=[True, False])
def as_index(request):
    return request.param


@pytest.fixture(params=[True, False])
def dropna(request):
    return request.param


@pytest.fixture(params=[True, False])
def observed(request):
    return request.param


@pytest.fixture
def df():
    return DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )


@pytest.fixture
def ts():
    return Series(
        np.random.default_rng(2).standard_normal(30),
        index=date_range("2000-01-01", periods=30, freq="B"),
    )


@pytest.fixture
def tsframe():
    return DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=30, freq="B"),
    )


@pytest.fixture
def three_group():
    return DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )


@pytest.fixture()
def slice_test_df():
    data = [
        [0, "a", "a0_at_0"],
        [1, "b", "b0_at_1"],
        [2, "a", "a1_at_2"],
        [3, "b", "b1_at_3"],
        [4, "c", "c0_at_4"],
        [5, "a", "a2_at_5"],
        [6, "a", "a3_at_6"],
        [7, "a", "a4_at_7"],
    ]
    df = DataFrame(data, columns=["Index", "Group", "Value"])
    return df.set_index("Index")


@pytest.fixture()
def slice_test_grouped(slice_test_df):
    return slice_test_df.groupby("Group", as_index=False)


@pytest.fixture(params=sorted(reduction_kernels))
def reduction_func(request):
    """
    yields the string names of all groupby reduction functions, one at a time.
    """
    return request.param


@pytest.fixture(params=sorted(transformation_kernels))
def transformation_func(request):
    """yields the string names of all groupby transformation functions."""
    return request.param


@pytest.fixture(params=sorted(reduction_kernels) + sorted(transformation_kernels))
def groupby_func(request):
    """yields both aggregation and transformation functions."""
    return request.param


@pytest.fixture(params=[True, False])
def parallel(request):
    """parallel keyword argument for numba.jit"""
    return request.param


# Can parameterize nogil & nopython over True | False, but limiting per
# https://github.com/pandas-dev/pandas/pull/41971#issuecomment-860607472


@pytest.fixture(params=[False])
def nogil(request):
    """nogil keyword argument for numba.jit"""
    return request.param


@pytest.fixture(params=[True])
def nopython(request):
    """nopython keyword argument for numba.jit"""
    return request.param


@pytest.fixture(
    params=[
        ("mean", {}),
        ("var", {"ddof": 1}),
        ("var", {"ddof": 0}),
        ("std", {"ddof": 1}),
        ("std", {"ddof": 0}),
        ("sum", {}),
        ("min", {}),
        ("max", {}),
        ("sum", {"min_count": 2}),
        ("min", {"min_count": 2}),
        ("max", {"min_count": 2}),
    ],
    ids=[
        "mean",
        "var_1",
        "var_0",
        "std_1",
        "std_0",
        "sum",
        "min",
        "max",
        "sum-min_count",
        "min-min_count",
        "max-min_count",
    ],
)
def numba_supported_reductions(request):
    """reductions supported with engine='numba'"""
    return request.param

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_interval_tree.py ===
from itertools import permutations

import numpy as np
import pytest

from pandas._libs.interval import IntervalTree
from pandas.compat import IS64

import pandas._testing as tm


def skipif_32bit(param):
    """
    Skip parameters in a parametrize on 32bit systems. Specifically used
    here to skip leaf_size parameters related to GH 23440.
    """
    marks = pytest.mark.skipif(not IS64, reason="GH 23440: int type mismatch on 32bit")
    return pytest.param(param, marks=marks)


@pytest.fixture(params=["int64", "float64", "uint64"])
def dtype(request):
    return request.param


@pytest.fixture(params=[skipif_32bit(1), skipif_32bit(2), 10])
def leaf_size(request):
    """
    Fixture to specify IntervalTree leaf_size parameter; to be used with the
    tree fixture.
    """
    return request.param


@pytest.fixture(
    params=[
        np.arange(5, dtype="int64"),
        np.arange(5, dtype="uint64"),
        np.arange(5, dtype="float64"),
        np.array([0, 1, 2, 3, 4, np.nan], dtype="float64"),
    ]
)
def tree(request, leaf_size):
    left = request.param
    return IntervalTree(left, left + 2, leaf_size=leaf_size)


class TestIntervalTree:
    def test_get_indexer(self, tree):
        result = tree.get_indexer(np.array([1.0, 5.5, 6.5]))
        expected = np.array([0, 4, -1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

        with pytest.raises(
            KeyError, match="'indexer does not intersect a unique set of intervals'"
        ):
            tree.get_indexer(np.array([3.0]))

    @pytest.mark.parametrize(
        "dtype, target_value, target_dtype",
        [("int64", 2**63 + 1, "uint64"), ("uint64", -1, "int64")],
    )
    def test_get_indexer_overflow(self, dtype, target_value, target_dtype):
        left, right = np.array([0, 1], dtype=dtype), np.array([1, 2], dtype=dtype)
        tree = IntervalTree(left, right)

        result = tree.get_indexer(np.array([target_value], dtype=target_dtype))
        expected = np.array([-1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_non_unique(self, tree):
        indexer, missing = tree.get_indexer_non_unique(np.array([1.0, 2.0, 6.5]))

        result = indexer[:1]
        expected = np.array([0], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

        result = np.sort(indexer[1:3])
        expected = np.array([0, 1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

        result = np.sort(indexer[3:])
        expected = np.array([-1], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

        result = missing
        expected = np.array([2], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype, target_value, target_dtype",
        [("int64", 2**63 + 1, "uint64"), ("uint64", -1, "int64")],
    )
    def test_get_indexer_non_unique_overflow(self, dtype, target_value, target_dtype):
        left, right = np.array([0, 2], dtype=dtype), np.array([1, 3], dtype=dtype)
        tree = IntervalTree(left, right)
        target = np.array([target_value], dtype=target_dtype)

        result_indexer, result_missing = tree.get_indexer_non_unique(target)
        expected_indexer = np.array([-1], dtype="intp")
        tm.assert_numpy_array_equal(result_indexer, expected_indexer)

        expected_missing = np.array([0], dtype="intp")
        tm.assert_numpy_array_equal(result_missing, expected_missing)

    def test_duplicates(self, dtype):
        left = np.array([0, 0, 0], dtype=dtype)
        tree = IntervalTree(left, left + 1)

        with pytest.raises(
            KeyError, match="'indexer does not intersect a unique set of intervals'"
        ):
            tree.get_indexer(np.array([0.5]))

        indexer, missing = tree.get_indexer_non_unique(np.array([0.5]))
        result = np.sort(indexer)
        expected = np.array([0, 1, 2], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

        result = missing
        expected = np.array([], dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "leaf_size", [skipif_32bit(1), skipif_32bit(10), skipif_32bit(100), 10000]
    )
    def test_get_indexer_closed(self, closed, leaf_size):
        x = np.arange(1000, dtype="float64")
        found = x.astype("intp")
        not_found = (-1 * np.ones(1000)).astype("intp")

        tree = IntervalTree(x, x + 0.5, closed=closed, leaf_size=leaf_size)
        tm.assert_numpy_array_equal(found, tree.get_indexer(x + 0.25))

        expected = found if tree.closed_left else not_found
        tm.assert_numpy_array_equal(expected, tree.get_indexer(x + 0.0))

        expected = found if tree.closed_right else not_found
        tm.assert_numpy_array_equal(expected, tree.get_indexer(x + 0.5))

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (np.array([0, 1, 4], dtype="int64"), np.array([2, 3, 5]), True),
            (np.array([0, 1, 2], dtype="int64"), np.array([5, 4, 3]), True),
            (np.array([0, 1, np.nan]), np.array([5, 4, np.nan]), True),
            (np.array([0, 2, 4], dtype="int64"), np.array([1, 3, 5]), False),
            (np.array([0, 2, np.nan]), np.array([1, 3, np.nan]), False),
        ],
    )
    @pytest.mark.parametrize("order", (list(x) for x in permutations(range(3))))
    def test_is_overlapping(self, closed, order, left, right, expected):
        # GH 23309
        tree = IntervalTree(left[order], right[order], closed=closed)
        result = tree.is_overlapping
        assert result is expected

    @pytest.mark.parametrize("order", (list(x) for x in permutations(range(3))))
    def test_is_overlapping_endpoints(self, closed, order):
        """shared endpoints are marked as overlapping"""
        # GH 23309
        left, right = np.arange(3, dtype="int64"), np.arange(1, 4)
        tree = IntervalTree(left[order], right[order], closed=closed)
        result = tree.is_overlapping
        expected = closed == "both"
        assert result is expected

    @pytest.mark.parametrize(
        "left, right",
        [
            (np.array([], dtype="int64"), np.array([], dtype="int64")),
            (np.array([0], dtype="int64"), np.array([1], dtype="int64")),
            (np.array([np.nan]), np.array([np.nan])),
            (np.array([np.nan] * 3), np.array([np.nan] * 3)),
        ],
    )
    def test_is_overlapping_trivial(self, closed, left, right):
        # GH 23309
        tree = IntervalTree(left, right, closed=closed)
        assert tree.is_overlapping is False

    @pytest.mark.skipif(not IS64, reason="GH 23440")
    def test_construction_overflow(self):
        # GH 25485
        left, right = np.arange(101, dtype="int64"), [np.iinfo(np.int64).max] * 101
        tree = IntervalTree(left, right)

        # pivot should be average of left/right medians
        result = tree.root.pivot
        expected = (50 + np.iinfo(np.int64).max) / 2
        assert result == expected

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            ([-np.inf, 1.0], [1.0, 2.0], 0.0),
            ([-np.inf, -2.0], [-2.0, -1.0], -2.0),
            ([-2.0, -1.0], [-1.0, np.inf], 0.0),
            ([1.0, 2.0], [2.0, np.inf], 2.0),
        ],
    )
    def test_inf_bound_infinite_recursion(self, left, right, expected):
        # GH 46658

        tree = IntervalTree(left * 101, right * 101)

        result = tree.root.pivot
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_setops.py ===
import numpy as np
import pytest

from pandas import (
    Index,
    IntervalIndex,
    Timestamp,
    interval_range,
)
import pandas._testing as tm


def monotonic_index(start, end, dtype="int64", closed="right"):
    return IntervalIndex.from_breaks(np.arange(start, end, dtype=dtype), closed=closed)


def empty_index(dtype="int64", closed="right"):
    return IntervalIndex(np.array([], dtype=dtype), closed=closed)


class TestIntervalIndex:
    def test_union(self, closed, sort):
        index = monotonic_index(0, 11, closed=closed)
        other = monotonic_index(5, 13, closed=closed)

        expected = monotonic_index(0, 13, closed=closed)
        result = index[::-1].union(other, sort=sort)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        result = other[::-1].union(index, sort=sort)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        tm.assert_index_equal(index.union(index, sort=sort), index)
        tm.assert_index_equal(index.union(index[:1], sort=sort), index)

    def test_union_empty_result(self, closed, sort):
        # GH 19101: empty result, same dtype
        index = empty_index(dtype="int64", closed=closed)
        result = index.union(index, sort=sort)
        tm.assert_index_equal(result, index)

        # GH 19101: empty result, different numeric dtypes -> common dtype is f8
        other = empty_index(dtype="float64", closed=closed)
        result = index.union(other, sort=sort)
        expected = other
        tm.assert_index_equal(result, expected)

        other = index.union(index, sort=sort)
        tm.assert_index_equal(result, expected)

        other = empty_index(dtype="uint64", closed=closed)
        result = index.union(other, sort=sort)
        tm.assert_index_equal(result, expected)

        result = other.union(index, sort=sort)
        tm.assert_index_equal(result, expected)

    def test_intersection(self, closed, sort):
        index = monotonic_index(0, 11, closed=closed)
        other = monotonic_index(5, 13, closed=closed)

        expected = monotonic_index(5, 11, closed=closed)
        result = index[::-1].intersection(other, sort=sort)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        result = other[::-1].intersection(index, sort=sort)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        tm.assert_index_equal(index.intersection(index, sort=sort), index)

        # GH 26225: nested intervals
        index = IntervalIndex.from_tuples([(1, 2), (1, 3), (1, 4), (0, 2)])
        other = IntervalIndex.from_tuples([(1, 2), (1, 3)])
        expected = IntervalIndex.from_tuples([(1, 2), (1, 3)])
        result = index.intersection(other)
        tm.assert_index_equal(result, expected)

        # GH 26225
        index = IntervalIndex.from_tuples([(0, 3), (0, 2)])
        other = IntervalIndex.from_tuples([(0, 2), (1, 3)])
        expected = IntervalIndex.from_tuples([(0, 2)])
        result = index.intersection(other)
        tm.assert_index_equal(result, expected)

        # GH 26225: duplicate nan element
        index = IntervalIndex([np.nan, np.nan])
        other = IntervalIndex([np.nan])
        expected = IntervalIndex([np.nan])
        result = index.intersection(other)
        tm.assert_index_equal(result, expected)

    def test_intersection_empty_result(self, closed, sort):
        index = monotonic_index(0, 11, closed=closed)

        # GH 19101: empty result, same dtype
        other = monotonic_index(300, 314, closed=closed)
        expected = empty_index(dtype="int64", closed=closed)
        result = index.intersection(other, sort=sort)
        tm.assert_index_equal(result, expected)

        # GH 19101: empty result, different numeric dtypes -> common dtype is float64
        other = monotonic_index(300, 314, dtype="float64", closed=closed)
        result = index.intersection(other, sort=sort)
        expected = other[:0]
        tm.assert_index_equal(result, expected)

        other = monotonic_index(300, 314, dtype="uint64", closed=closed)
        result = index.intersection(other, sort=sort)
        tm.assert_index_equal(result, expected)

    def test_intersection_duplicates(self):
        # GH#38743
        index = IntervalIndex.from_tuples([(1, 2), (1, 2), (2, 3), (3, 4)])
        other = IntervalIndex.from_tuples([(1, 2), (2, 3)])
        expected = IntervalIndex.from_tuples([(1, 2), (2, 3)])
        result = index.intersection(other)
        tm.assert_index_equal(result, expected)

    def test_difference(self, closed, sort):
        index = IntervalIndex.from_arrays([1, 0, 3, 2], [1, 2, 3, 4], closed=closed)
        result = index.difference(index[:1], sort=sort)
        expected = index[1:]
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

        # GH 19101: empty result, same dtype
        result = index.difference(index, sort=sort)
        expected = empty_index(dtype="int64", closed=closed)
        tm.assert_index_equal(result, expected)

        # GH 19101: empty result, different dtypes
        other = IntervalIndex.from_arrays(
            index.left.astype("float64"), index.right, closed=closed
        )
        result = index.difference(other, sort=sort)
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference(self, closed, sort):
        index = monotonic_index(0, 11, closed=closed)
        result = index[1:].symmetric_difference(index[:-1], sort=sort)
        expected = IntervalIndex([index[0], index[-1]])
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        # GH 19101: empty result, same dtype
        result = index.symmetric_difference(index, sort=sort)
        expected = empty_index(dtype="int64", closed=closed)
        if sort in (None, True):
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result.sort_values(), expected)

        # GH 19101: empty result, different dtypes
        other = IntervalIndex.from_arrays(
            index.left.astype("float64"), index.right, closed=closed
        )
        result = index.symmetric_difference(other, sort=sort)
        expected = empty_index(dtype="float64", closed=closed)
        tm.assert_index_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:'<' not supported between:RuntimeWarning")
    @pytest.mark.parametrize(
        "op_name", ["union", "intersection", "difference", "symmetric_difference"]
    )
    def test_set_incompatible_types(self, closed, op_name, sort):
        index = monotonic_index(0, 11, closed=closed)
        set_op = getattr(index, op_name)

        # TODO: standardize return type of non-union setops type(self vs other)
        # non-IntervalIndex
        if op_name == "difference":
            expected = index
        else:
            expected = getattr(index.astype("O"), op_name)(Index([1, 2, 3]))
        result = set_op(Index([1, 2, 3]), sort=sort)
        tm.assert_index_equal(result, expected)

        # mixed closed -> cast to object
        for other_closed in {"right", "left", "both", "neither"} - {closed}:
            other = monotonic_index(0, 11, closed=other_closed)
            expected = getattr(index.astype(object), op_name)(other, sort=sort)
            if op_name == "difference":
                expected = index
            result = set_op(other, sort=sort)
            tm.assert_index_equal(result, expected)

        # GH 19016: incompatible dtypes -> cast to object
        other = interval_range(Timestamp("20180101"), periods=9, closed=closed)
        expected = getattr(index.astype(object), op_name)(other, sort=sort)
        if op_name == "difference":
            expected = index
        result = set_op(other, sort=sort)
        tm.assert_index_equal(result, expected)