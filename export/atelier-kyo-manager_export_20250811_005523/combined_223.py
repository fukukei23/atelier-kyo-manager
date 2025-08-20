
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\test_frame_transform.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames


def unpack_obj(obj, klass, axis):
    """
    Helper to ensure we have the right type of object for a test parametrized
    over frame_or_series.
    """
    if klass is not DataFrame:
        obj = obj["A"]
        if axis != 0:
            pytest.skip(f"Test is only for DataFrame with axis={axis}")
    return obj


def test_transform_ufunc(axis, float_frame, frame_or_series):
    # GH 35964
    obj = unpack_obj(float_frame, frame_or_series, axis)

    with np.errstate(all="ignore"):
        f_sqrt = np.sqrt(obj)

    # ufunc
    result = obj.transform(np.sqrt, axis=axis)
    expected = f_sqrt
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sqrt], ["sqrt"]),
        ([np.abs, np.sqrt], ["absolute", "sqrt"]),
        (np.array([np.sqrt]), ["sqrt"]),
        (np.array([np.abs, np.sqrt]), ["absolute", "sqrt"]),
    ],
)
def test_transform_listlike(axis, float_frame, ops, names):
    # GH 35964
    other_axis = 1 if axis in {0, "index"} else 0
    with np.errstate(all="ignore"):
        expected = zip_frames([op(float_frame) for op in ops], axis=other_axis)
    if axis in {0, "index"}:
        expected.columns = MultiIndex.from_product([float_frame.columns, names])
    else:
        expected.index = MultiIndex.from_product([float_frame.index, names])
    result = float_frame.transform(ops, axis=axis)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ops", [[], np.array([])])
def test_transform_empty_listlike(float_frame, ops, frame_or_series):
    obj = unpack_obj(float_frame, frame_or_series, 0)

    with pytest.raises(ValueError, match="No transform functions were provided"):
        obj.transform(ops)


def test_transform_listlike_func_with_args():
    # GH 50624
    df = DataFrame({"x": [1, 2, 3]})

    def foo1(x, a=1, c=0):
        return x + a + c

    def foo2(x, b=2, c=0):
        return x + b + c

    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        df.transform([foo1, foo2], 0, 3, b=3, c=4)

    result = df.transform([foo1, foo2], 0, 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]],
        columns=MultiIndex.from_tuples([("x", "foo1"), ("x", "foo2")]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("box", [dict, Series])
def test_transform_dictlike(axis, float_frame, box):
    # GH 35964
    if axis in (0, "index"):
        e = float_frame.columns[0]
        expected = float_frame[[e]].transform(np.abs)
    else:
        e = float_frame.index[0]
        expected = float_frame.iloc[[0]].transform(np.abs)
    result = float_frame.transform(box({e: np.abs}), axis=axis)
    tm.assert_frame_equal(result, expected)


def test_transform_dictlike_mixed():
    # GH 40018 - mix of lists and non-lists in values of a dictionary
    df = DataFrame({"a": [1, 2], "b": [1, 4], "c": [1, 4]})
    result = df.transform({"b": ["sqrt", "abs"], "c": "sqrt"})
    expected = DataFrame(
        [[1.0, 1, 1.0], [2.0, 4, 2.0]],
        columns=MultiIndex([("b", "c"), ("sqrt", "abs")], [(0, 0, 1), (0, 1, 0)]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {},
        {"A": []},
        {"A": [], "B": "cumsum"},
        {"A": "cumsum", "B": []},
        {"A": [], "B": ["cumsum"]},
        {"A": ["cumsum"], "B": []},
    ],
)
def test_transform_empty_dictlike(float_frame, ops, frame_or_series):
    obj = unpack_obj(float_frame, frame_or_series, 0)

    with pytest.raises(ValueError, match="No transform functions were provided"):
        obj.transform(ops)


@pytest.mark.parametrize("use_apply", [True, False])
def test_transform_udf(axis, float_frame, use_apply, frame_or_series):
    # GH 35964
    obj = unpack_obj(float_frame, frame_or_series, axis)

    # transform uses UDF either via apply or passing the entire DataFrame
    def func(x):
        # transform is using apply iff x is not a DataFrame
        if use_apply == isinstance(x, frame_or_series):
            # Force transform to fallback
            raise ValueError
        return x + 1

    result = obj.transform(func, axis=axis)
    expected = obj + 1
    tm.assert_equal(result, expected)


wont_fail = ["ffill", "bfill", "fillna", "pad", "backfill", "shift"]
frame_kernels_raise = [x for x in frame_transform_kernels if x not in wont_fail]


@pytest.mark.parametrize("op", [*frame_kernels_raise, lambda x: x + 1])
def test_transform_bad_dtype(op, frame_or_series, request):
    # GH 35964
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )

    obj = DataFrame({"A": 3 * [object]})  # DataFrame that will fail on most transforms
    obj = tm.get_obj(obj, frame_or_series)
    error = TypeError
    msg = "|".join(
        [
            "not supported between instances of 'type' and 'type'",
            "unsupported operand type",
        ]
    )

    with pytest.raises(error, match=msg):
        obj.transform(op)
    with pytest.raises(error, match=msg):
        obj.transform([op])
    with pytest.raises(error, match=msg):
        obj.transform({"A": op})
    with pytest.raises(error, match=msg):
        obj.transform({"A": [op]})


@pytest.mark.parametrize("op", frame_kernels_raise)
def test_transform_failure_typeerror(request, op):
    # GH 35964

    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )

    # Using object makes most transform kernels fail
    df = DataFrame({"A": 3 * [object], "B": [1, 2, 3]})
    error = TypeError
    msg = "|".join(
        [
            "not supported between instances of 'type' and 'type'",
            "unsupported operand type",
        ]
    )

    with pytest.raises(error, match=msg):
        df.transform([op])

    with pytest.raises(error, match=msg):
        df.transform({"A": op, "B": op})

    with pytest.raises(error, match=msg):
        df.transform({"A": [op], "B": [op]})

    with pytest.raises(error, match=msg):
        df.transform({"A": [op, "shift"], "B": [op]})


def test_transform_failure_valueerror():
    # GH 40211
    def op(x):
        if np.sum(np.sum(x)) < 10:
            raise ValueError
        return x

    df = DataFrame({"A": [1, 2, 3], "B": [400, 500, 600]})
    msg = "Transform function failed"

    with pytest.raises(ValueError, match=msg):
        df.transform([op])

    with pytest.raises(ValueError, match=msg):
        df.transform({"A": op, "B": op})

    with pytest.raises(ValueError, match=msg):
        df.transform({"A": [op], "B": [op]})

    with pytest.raises(ValueError, match=msg):
        df.transform({"A": [op, "shift"], "B": [op]})


@pytest.mark.parametrize("use_apply", [True, False])
def test_transform_passes_args(use_apply, frame_or_series):
    # GH 35964
    # transform uses UDF either via apply or passing the entire DataFrame
    expected_args = [1, 2]
    expected_kwargs = {"c": 3}

    def f(x, a, b, c):
        # transform is using apply iff x is not a DataFrame
        if use_apply == isinstance(x, frame_or_series):
            # Force transform to fallback
            raise ValueError
        assert [a, b] == expected_args
        assert c == expected_kwargs["c"]
        return x

    frame_or_series([1]).transform(f, 0, *expected_args, **expected_kwargs)


def test_transform_empty_dataframe():
    # https://github.com/pandas-dev/pandas/issues/39636
    df = DataFrame([], columns=["col1", "col2"])
    result = df.transform(lambda x: x + 10)
    tm.assert_frame_equal(result, df)

    result = df["col1"].transform(lambda x: x + 10)
    tm.assert_series_equal(result, df["col1"])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_qubit.py ===
import random

from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.qubit import (measure_all, measure_all_oneshot, measure_partial,
                                         matrix_to_qubit, matrix_to_density,
                                         qubit_to_matrix, IntQubit,
                                         IntQubitBra, QubitBra)
from sympy.physics.quantum.gate import (HadamardGate, CNOT, XGate, YGate,
                                        ZGate, PhaseGate)
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.shor import Qubit
from sympy.testing.pytest import raises
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr

x, y = symbols('x,y')

epsilon = .000001


def test_Qubit():
    array = [0, 0, 1, 1, 0]
    qb = Qubit('00110')
    assert qb.flip(0) == Qubit('00111')
    assert qb.flip(1) == Qubit('00100')
    assert qb.flip(4) == Qubit('10110')
    assert qb.qubit_values == (0, 0, 1, 1, 0)
    assert qb.dimension == 5
    for i in range(5):
        assert qb[i] == array[4 - i]
    assert len(qb) == 5
    qb = Qubit('110')


def test_QubitBra():
    qb = Qubit(0)
    qb_bra = QubitBra(0)
    assert qb.dual_class() == QubitBra
    assert qb_bra.dual_class() == Qubit

    qb = Qubit(1, 1, 0)
    qb_bra = QubitBra(1, 1, 0)
    assert represent(qb, nqubits=3).H == represent(qb_bra, nqubits=3)

    qb = Qubit(0, 1)
    qb_bra = QubitBra(1,0)
    assert qb._eval_innerproduct_QubitBra(qb_bra) == Integer(0)

    qb_bra = QubitBra(0, 1)
    assert qb._eval_innerproduct_QubitBra(qb_bra) == Integer(1)


def test_IntQubit():
    # issue 9136
    iqb = IntQubit(0, nqubits=1)
    assert qubit_to_matrix(Qubit('0')) == qubit_to_matrix(iqb)

    qb = Qubit('1010')
    assert qubit_to_matrix(IntQubit(qb)) == qubit_to_matrix(qb)

    iqb = IntQubit(1, nqubits=1)
    assert qubit_to_matrix(Qubit('1')) == qubit_to_matrix(iqb)
    assert qubit_to_matrix(IntQubit(1)) == qubit_to_matrix(iqb)

    iqb = IntQubit(7, nqubits=4)
    assert qubit_to_matrix(Qubit('0111')) == qubit_to_matrix(iqb)
    assert qubit_to_matrix(IntQubit(7, 4)) == qubit_to_matrix(iqb)

    iqb = IntQubit(8)
    assert iqb.as_int() == 8
    assert iqb.qubit_values == (1, 0, 0, 0)

    iqb = IntQubit(7, 4)
    assert iqb.qubit_values == (0, 1, 1, 1)
    assert IntQubit(3) == IntQubit(3, 2)

    #test Dual Classes
    iqb = IntQubit(3)
    iqb_bra = IntQubitBra(3)
    assert iqb.dual_class() == IntQubitBra
    assert iqb_bra.dual_class() == IntQubit

    iqb = IntQubit(5)
    iqb_bra = IntQubitBra(5)
    assert iqb._eval_innerproduct_IntQubitBra(iqb_bra) == Integer(1)

    iqb = IntQubit(4)
    iqb_bra = IntQubitBra(5)
    assert iqb._eval_innerproduct_IntQubitBra(iqb_bra) == Integer(0)
    raises(ValueError, lambda: IntQubit(4, 1))

    raises(ValueError, lambda: IntQubit('5'))
    raises(ValueError, lambda: IntQubit(5, '5'))
    raises(ValueError, lambda: IntQubit(5, nqubits='5'))
    raises(TypeError, lambda: IntQubit(5, bad_arg=True))

def test_superposition_of_states():
    state = 1/sqrt(2)*Qubit('01') + 1/sqrt(2)*Qubit('10')
    state_gate = CNOT(0, 1)*HadamardGate(0)*state
    state_expanded = Qubit('01')/2 + Qubit('00')/2 - Qubit('11')/2 + Qubit('10')/2
    assert qapply(state_gate).expand() == state_expanded
    assert matrix_to_qubit(represent(state_gate, nqubits=2)) == state_expanded


#test apply methods
def test_apply_represent_equality():
    gates = [HadamardGate(int(3*random.random())),
     XGate(int(3*random.random())), ZGate(int(3*random.random())),
        YGate(int(3*random.random())), ZGate(int(3*random.random())),
        PhaseGate(int(3*random.random()))]

    circuit = Qubit(int(random.random()*2), int(random.random()*2),
    int(random.random()*2), int(random.random()*2), int(random.random()*2),
        int(random.random()*2))
    for i in range(int(random.random()*6)):
        circuit = gates[int(random.random()*6)]*circuit

    mat = represent(circuit, nqubits=6)
    states = qapply(circuit)
    state_rep = matrix_to_qubit(mat)
    states = states.expand()
    state_rep = state_rep.expand()
    assert state_rep == states


def test_matrix_to_qubits():
    qb = Qubit(0, 0, 0, 0)
    mat = Matrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert matrix_to_qubit(mat) == qb
    assert qubit_to_matrix(qb) == mat

    state = 2*sqrt(2)*(Qubit(0, 0, 0) + Qubit(0, 0, 1) + Qubit(0, 1, 0) +
                       Qubit(0, 1, 1) + Qubit(1, 0, 0) + Qubit(1, 0, 1) +
                       Qubit(1, 1, 0) + Qubit(1, 1, 1))
    ones = sqrt(2)*2*Matrix([1, 1, 1, 1, 1, 1, 1, 1])
    assert matrix_to_qubit(ones) == state.expand()
    assert qubit_to_matrix(state) == ones


def test_measure_normalize():
    a, b = symbols('a b')
    state = a*Qubit('110') + b*Qubit('111')
    assert measure_partial(state, (0,), normalize=False) == \
        [(a*Qubit('110'), a*a.conjugate()), (b*Qubit('111'), b*b.conjugate())]
    assert measure_all(state, normalize=False) == \
        [(Qubit('110'), a*a.conjugate()), (Qubit('111'), b*b.conjugate())]


def test_measure_partial():
    #Basic test of collapse of entangled two qubits (Bell States)
    state = Qubit('01') + Qubit('10')
    assert measure_partial(state, (0,)) == \
        [(Qubit('10'), S.Half), (Qubit('01'), S.Half)]
    assert measure_partial(state, int(0)) == \
        [(Qubit('10'), S.Half), (Qubit('01'), S.Half)]
    assert measure_partial(state, (0,)) == \
        measure_partial(state, (1,))[::-1]

    #Test of more complex collapse and probability calculation
    state1 = sqrt(2)/sqrt(3)*Qubit('00001') + 1/sqrt(3)*Qubit('11111')
    assert measure_partial(state1, (0,)) == \
        [(sqrt(2)/sqrt(3)*Qubit('00001') + 1/sqrt(3)*Qubit('11111'), 1)]
    assert measure_partial(state1, (1, 2)) == measure_partial(state1, (3, 4))
    assert measure_partial(state1, (1, 2, 3)) == \
        [(Qubit('00001'), Rational(2, 3)), (Qubit('11111'), Rational(1, 3))]

    #test of measuring multiple bits at once
    state2 = Qubit('1111') + Qubit('1101') + Qubit('1011') + Qubit('1000')
    assert measure_partial(state2, (0, 1, 3)) == \
        [(Qubit('1000'), Rational(1, 4)), (Qubit('1101'), Rational(1, 4)),
         (Qubit('1011')/sqrt(2) + Qubit('1111')/sqrt(2), S.Half)]
    assert measure_partial(state2, (0,)) == \
        [(Qubit('1000'), Rational(1, 4)),
         (Qubit('1111')/sqrt(3) + Qubit('1101')/sqrt(3) +
          Qubit('1011')/sqrt(3), Rational(3, 4))]


def test_measure_all():
    assert measure_all(Qubit('11')) == [(Qubit('11'), 1)]
    state = Qubit('11') + Qubit('10')
    assert measure_all(state) == [(Qubit('10'), S.Half),
           (Qubit('11'), S.Half)]
    state2 = Qubit('11')/sqrt(5) + 2*Qubit('00')/sqrt(5)
    assert measure_all(state2) == \
        [(Qubit('00'), Rational(4, 5)), (Qubit('11'), Rational(1, 5))]

    # from issue #12585
    assert measure_all(qapply(Qubit('0'))) == [(Qubit('0'), 1)]


def test_measure_all_oneshot():
    random.seed(42)
    # for issue #27092
    assert measure_all_oneshot(Qubit('11')) == Qubit('11')
    assert measure_all_oneshot(Qubit('1')) == Qubit('1')
    assert measure_all_oneshot(Qubit('0')/sqrt(2) + Qubit('1')/sqrt(2)) == \
            Qubit('0')


def test_eval_trace():
    q1 = Qubit('10110')
    q2 = Qubit('01010')
    d = Density([q1, 0.6], [q2, 0.4])

    t = Tr(d)
    assert t.doit() == 1.0

    # extreme bits
    t = Tr(d, 0)
    assert t.doit() == (0.4*Density([Qubit('0101'), 1]) +
                        0.6*Density([Qubit('1011'), 1]))
    t = Tr(d, 4)
    assert t.doit() == (0.4*Density([Qubit('1010'), 1]) +
                        0.6*Density([Qubit('0110'), 1]))
    # index somewhere in between
    t = Tr(d, 2)
    assert t.doit() == (0.4*Density([Qubit('0110'), 1]) +
                        0.6*Density([Qubit('1010'), 1]))
    #trace all indices
    t = Tr(d, [0, 1, 2, 3, 4])
    assert t.doit() == 1.0

    # trace some indices, initialized in
    # non-canonical order
    t = Tr(d, [2, 1, 3])
    assert t.doit() == (0.4*Density([Qubit('00'), 1]) +
                        0.6*Density([Qubit('10'), 1]))

    # mixed states
    q = (1/sqrt(2)) * (Qubit('00') + Qubit('11'))
    d = Density( [q, 1.0] )
    t = Tr(d, 0)
    assert t.doit() == (0.5*Density([Qubit('0'), 1]) +
                        0.5*Density([Qubit('1'), 1]))


def test_matrix_to_density():
    mat = Matrix([[0, 0], [0, 1]])
    assert matrix_to_density(mat) == Density([Qubit('1'), 1])

    mat = Matrix([[1, 0], [0, 0]])
    assert matrix_to_density(mat) == Density([Qubit('0'), 1])

    mat = Matrix([[0, 0], [0, 0]])
    assert matrix_to_density(mat) == 0

    mat = Matrix([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]])

    assert matrix_to_density(mat) == Density([Qubit('10'), 1])

    mat = Matrix([[1, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    assert matrix_to_density(mat) == Density([Qubit('00'), 1])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_callback.py ===
import math
import platform
import sys
import textwrap
import threading
import time
import traceback

import pytest

import numpy as np
from numpy.testing import IS_PYPY

from . import util


class TestF77Callback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "foo.f")]

    @pytest.mark.parametrize("name", ["t", "t2"])
    @pytest.mark.slow
    def test_all(self, name):
        self.check_function(name)

    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = t(fun,[fun_extra_args])

        Wrapper for ``t``.

        Parameters
        ----------
        fun : call-back function

        Other Parameters
        ----------------
        fun_extra_args : input tuple, optional
            Default: ()

        Returns
        -------
        a : int

        Notes
        -----
        Call-back functions::

            def fun(): return a
            Return objects:
                a : int
        """)
        assert self.module.t.__doc__ == expected

    def check_function(self, name):
        t = getattr(self.module, name)
        r = t(lambda: 4)
        assert r == 4
        r = t(lambda a: 5, fun_extra_args=(6, ))
        assert r == 5
        r = t(lambda a: a, fun_extra_args=(6, ))
        assert r == 6
        r = t(lambda a: 5 + a, fun_extra_args=(7, ))
        assert r == 12
        r = t(math.degrees, fun_extra_args=(math.pi, ))
        assert r == 180
        r = t(math.degrees, fun_extra_args=(math.pi, ))
        assert r == 180

        r = t(self.module.func, fun_extra_args=(6, ))
        assert r == 17
        r = t(self.module.func0)
        assert r == 11
        r = t(self.module.func0._cpointer)
        assert r == 11

        class A:
            def __call__(self):
                return 7

            def mth(self):
                return 9

        a = A()
        r = t(a)
        assert r == 7
        r = t(a.mth)
        assert r == 9

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback(self):
        def callback(code):
            if code == "r":
                return 0
            else:
                return 1

        f = self.module.string_callback
        r = f(callback)
        assert r == 0

    @pytest.mark.skipif(sys.platform == 'win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback_array(self):
        # See gh-10027
        cu1 = np.zeros((1, ), "S8")
        cu2 = np.zeros((1, 8), "c")
        cu3 = np.array([""], "S8")

        def callback(cu, lencu):
            if cu.shape != (lencu,):
                return 1
            if cu.dtype != "S8":
                return 2
            if not np.all(cu == b""):
                return 3
            return 0

        f = self.module.string_callback_array
        for cu in [cu1, cu2, cu3]:
            res = f(callback, cu, cu.size)
            assert res == 0

    def test_threadsafety(self):
        # Segfaults if the callback handling is not threadsafe

        errors = []

        def cb():
            # Sleep here to make it more likely for another thread
            # to call their callback at the same time.
            time.sleep(1e-3)

            # Check reentrancy
            r = self.module.t(lambda: 123)
            assert r == 123

            return 42

        def runner(name):
            try:
                for j in range(50):
                    r = self.module.t(cb)
                    assert r == 42
                    self.check_function(name)
            except Exception:
                errors.append(traceback.format_exc())

        threads = [
            threading.Thread(target=runner, args=(arg, ))
            for arg in ("t", "t2") for n in range(20)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        errors = "\n\n".join(errors)
        if errors:
            raise AssertionError(errors)

    def test_hidden_callback(self):
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith("Callback global_f not defined")

        try:
            self.module.hidden_callback2(2)
        except Exception as msg:
            assert str(msg).startswith("cb: Callback global_f not defined")

        self.module.global_f = lambda x: x + 1
        r = self.module.hidden_callback(2)
        assert r == 3

        self.module.global_f = lambda x: x + 2
        r = self.module.hidden_callback(2)
        assert r == 4

        del self.module.global_f
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith("Callback global_f not defined")

        self.module.global_f = lambda x=0: x + 3
        r = self.module.hidden_callback(2)
        assert r == 5

        # reproducer of gh18341
        r = self.module.hidden_callback2(2)
        assert r == 3


class TestF77CallbackPythonTLS(TestF77Callback):
    """
    Callback tests using Python thread-local storage instead of
    compiler-provided
    """

    options = ["-DF2PY_USE_PYTHON_TLS"]


class TestF90Callback(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh17797.f90")]

    @pytest.mark.slow
    def test_gh17797(self):
        def incr(x):
            return x + 123

        y = np.array([1, 2, 3], dtype=np.int64)
        r = self.module.gh17797(incr, y)
        assert r == 123 + 1 + 2 + 3


class TestGH18335(util.F2PyTest):
    """The reproduction of the reported issue requires specific input that
    extensions may break the issue conditions, so the reproducer is
    implemented as a separate test class. Do not extend this test with
    other tests!
    """
    sources = [util.getpath("tests", "src", "callback", "gh18335.f90")]

    @pytest.mark.slow
    def test_gh18335(self):
        def foo(x):
            x[0] += 1

        r = self.module.gh18335(foo)
        assert r == 123 + 1


class TestGH25211(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh25211.f"),
               util.getpath("tests", "src", "callback", "gh25211.pyf")]
    module_name = "callback2"

    def test_gh25211(self):
        def bar(x):
            return x * x

        res = self.module.foo(bar)
        assert res == 110


@pytest.mark.slow
@pytest.mark.xfail(condition=(platform.system().lower() == 'darwin'),
                   run=False,
                   reason="Callback aborts cause CI failures on macOS")
class TestCBFortranCallstatement(util.F2PyTest):
    sources = [util.getpath("tests", "src", "callback", "gh26681.f90")]
    options = ['--lower']

    def test_callstatement_fortran(self):
        with pytest.raises(ValueError, match='helpme') as exc:
            self.module.mypy_abort = self.module.utils.my_abort
            self.module.utils.do_something('helpme')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_asfreq.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas._libs.tslibs.offsets import MonthEnd

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm

from pandas.tseries import offsets


class TestAsFreq:
    @pytest.fixture(params=["s", "ms", "us", "ns"])
    def unit(self, request):
        return request.param

    def test_asfreq2(self, frame_or_series):
        ts = frame_or_series(
            [0.0, 1.0, 2.0],
            index=DatetimeIndex(
                [
                    datetime(2009, 10, 30),
                    datetime(2009, 11, 30),
                    datetime(2009, 12, 31),
                ],
                dtype="M8[ns]",
                freq="BME",
            ),
        )

        daily_ts = ts.asfreq("B")
        monthly_ts = daily_ts.asfreq("BME")
        tm.assert_equal(monthly_ts, ts)

        daily_ts = ts.asfreq("B", method="pad")
        monthly_ts = daily_ts.asfreq("BME")
        tm.assert_equal(monthly_ts, ts)

        daily_ts = ts.asfreq(offsets.BDay())
        monthly_ts = daily_ts.asfreq(offsets.BMonthEnd())
        tm.assert_equal(monthly_ts, ts)

        result = ts[:0].asfreq("ME")
        assert len(result) == 0
        assert result is not ts

        if frame_or_series is Series:
            daily_ts = ts.asfreq("D", fill_value=-1)
            result = daily_ts.value_counts().sort_index()
            expected = Series(
                [60, 1, 1, 1], index=[-1.0, 2.0, 1.0, 0.0], name="count"
            ).sort_index()
            tm.assert_series_equal(result, expected)

    def test_asfreq_datetimeindex_empty(self, frame_or_series):
        # GH#14320
        index = DatetimeIndex(["2016-09-29 11:00"])
        expected = frame_or_series(index=index, dtype=object).asfreq("h")
        result = frame_or_series([3], index=index.copy()).asfreq("h")
        tm.assert_index_equal(expected.index, result.index)

    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_tz_aware_asfreq_smoke(self, tz, frame_or_series):
        dr = date_range("2011-12-01", "2012-07-20", freq="D", tz=tz)

        obj = frame_or_series(
            np.random.default_rng(2).standard_normal(len(dr)), index=dr
        )

        # it works!
        obj.asfreq("min")

    def test_asfreq_normalize(self, frame_or_series):
        rng = date_range("1/1/2000 09:30", periods=20)
        norm = date_range("1/1/2000", periods=20)

        vals = np.random.default_rng(2).standard_normal((20, 3))

        obj = DataFrame(vals, index=rng)
        expected = DataFrame(vals, index=norm)
        if frame_or_series is Series:
            obj = obj[0]
            expected = expected[0]

        result = obj.asfreq("D", normalize=True)
        tm.assert_equal(result, expected)

    def test_asfreq_keep_index_name(self, frame_or_series):
        # GH#9854
        index_name = "bar"
        index = date_range("20130101", periods=20, name=index_name)
        obj = DataFrame(list(range(20)), columns=["foo"], index=index)
        obj = tm.get_obj(obj, frame_or_series)

        assert index_name == obj.index.name
        assert index_name == obj.asfreq("10D").index.name

    def test_asfreq_ts(self, frame_or_series):
        index = period_range(freq="Y", start="1/1/2001", end="12/31/2010")
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 3)), index=index
        )
        obj = tm.get_obj(obj, frame_or_series)

        result = obj.asfreq("D", how="end")
        exp_index = index.asfreq("D", how="end")
        assert len(result) == len(obj)
        tm.assert_index_equal(result.index, exp_index)

        result = obj.asfreq("D", how="start")
        exp_index = index.asfreq("D", how="start")
        assert len(result) == len(obj)
        tm.assert_index_equal(result.index, exp_index)

    def test_asfreq_resample_set_correct_freq(self, frame_or_series):
        # GH#5613
        # we test if .asfreq() and .resample() set the correct value for .freq
        dti = to_datetime(["2012-01-01", "2012-01-02", "2012-01-03"])
        obj = DataFrame({"col": [1, 2, 3]}, index=dti)
        obj = tm.get_obj(obj, frame_or_series)

        # testing the settings before calling .asfreq() and .resample()
        assert obj.index.freq is None
        assert obj.index.inferred_freq == "D"

        # does .asfreq() set .freq correctly?
        assert obj.asfreq("D").index.freq == "D"

        # does .resample() set .freq correctly?
        assert obj.resample("D").asfreq().index.freq == "D"

    def test_asfreq_empty(self, datetime_frame):
        # test does not blow up on length-0 DataFrame
        zero_length = datetime_frame.reindex([])
        result = zero_length.asfreq("BME")
        assert result is not zero_length

    def test_asfreq(self, datetime_frame):
        offset_monthly = datetime_frame.asfreq(offsets.BMonthEnd())
        rule_monthly = datetime_frame.asfreq("BME")

        tm.assert_frame_equal(offset_monthly, rule_monthly)

        rule_monthly.asfreq("B", method="pad")
        # TODO: actually check that this worked.

        # don't forget!
        rule_monthly.asfreq("B", method="pad")

    def test_asfreq_datetimeindex(self):
        df = DataFrame(
            {"A": [1, 2, 3]},
            index=[datetime(2011, 11, 1), datetime(2011, 11, 2), datetime(2011, 11, 3)],
        )
        df = df.asfreq("B")
        assert isinstance(df.index, DatetimeIndex)

        ts = df["A"].asfreq("B")
        assert isinstance(ts.index, DatetimeIndex)

    def test_asfreq_fillvalue(self):
        # test for fill value during upsampling, related to issue 3715

        # setup
        rng = date_range("1/1/2016", periods=10, freq="2s")
        # Explicit cast to 'float' to avoid implicit cast when setting None
        ts = Series(np.arange(len(rng)), index=rng, dtype="float")
        df = DataFrame({"one": ts})

        # insert pre-existing missing value
        df.loc["2016-01-01 00:00:08", "one"] = None

        actual_df = df.asfreq(freq="1s", fill_value=9.0)
        expected_df = df.asfreq(freq="1s").fillna(9.0)
        expected_df.loc["2016-01-01 00:00:08", "one"] = None
        tm.assert_frame_equal(expected_df, actual_df)

        expected_series = ts.asfreq(freq="1s").fillna(9.0)
        actual_series = ts.asfreq(freq="1s", fill_value=9.0)
        tm.assert_series_equal(expected_series, actual_series)

    def test_asfreq_with_date_object_index(self, frame_or_series):
        rng = date_range("1/1/2000", periods=20)
        ts = frame_or_series(np.random.default_rng(2).standard_normal(20), index=rng)

        ts2 = ts.copy()
        ts2.index = [x.date() for x in ts2.index]

        result = ts2.asfreq("4h", method="ffill")
        expected = ts.asfreq("4h", method="ffill")
        tm.assert_equal(result, expected)

    def test_asfreq_with_unsorted_index(self, frame_or_series):
        # GH#39805
        # Test that rows are not dropped when the datetime index is out of order
        index = to_datetime(["2021-01-04", "2021-01-02", "2021-01-03", "2021-01-01"])
        result = frame_or_series(range(4), index=index)

        expected = result.reindex(sorted(index))
        expected.index = expected.index._with_freq("infer")

        result = result.asfreq("D")
        tm.assert_equal(result, expected)

    def test_asfreq_after_normalize(self, unit):
        # https://github.com/pandas-dev/pandas/issues/50727
        result = DatetimeIndex(
            date_range("2000", periods=2).as_unit(unit).normalize(), freq="D"
        )
        expected = DatetimeIndex(["2000-01-01", "2000-01-02"], freq="D").as_unit(unit)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, freq_half",
        [
            ("2ME", "ME"),
            (MonthEnd(2), MonthEnd(1)),
        ],
    )
    def test_asfreq_2ME(self, freq, freq_half):
        index = date_range("1/1/2000", periods=6, freq=freq_half)
        df = DataFrame({"s": Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=index)})
        expected = df.asfreq(freq=freq)

        index = date_range("1/1/2000", periods=3, freq=freq)
        result = DataFrame({"s": Series([0.0, 2.0, 4.0], index=index)})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, freq_depr",
        [
            ("2ME", "2M"),
            ("2QE", "2Q"),
            ("2QE-SEP", "2Q-SEP"),
            ("1BQE", "1BQ"),
            ("2BQE-SEP", "2BQ-SEP"),
            ("1YE", "1Y"),
            ("2YE-MAR", "2Y-MAR"),
            ("1YE", "1A"),
            ("2YE-MAR", "2A-MAR"),
            ("2BYE-MAR", "2BA-MAR"),
        ],
    )
    def test_asfreq_frequency_M_Q_Y_A_deprecated(self, freq, freq_depr):
        # GH#9586, #55978
        depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
        f"in a future version, please use '{freq[1:]}' instead."

        index = date_range("1/1/2000", periods=4, freq=f"{freq[1:]}")
        df = DataFrame({"s": Series([0.0, 1.0, 2.0, 3.0], index=index)})
        expected = df.asfreq(freq=freq)
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = df.asfreq(freq=freq_depr)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_analytics.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    date_range,
    period_range,
)
import pandas._testing as tm


def test_infer_objects(idx):
    with pytest.raises(NotImplementedError, match="to_frame"):
        idx.infer_objects()


def test_shift(idx):
    # GH8083 test the base class for shift
    msg = (
        "This method is only implemented for DatetimeIndex, PeriodIndex and "
        "TimedeltaIndex; Got type MultiIndex"
    )
    with pytest.raises(NotImplementedError, match=msg):
        idx.shift(1)
    with pytest.raises(NotImplementedError, match=msg):
        idx.shift(1, 2)


def test_groupby(idx):
    groups = idx.groupby(np.array([1, 1, 1, 2, 2, 2]))
    labels = idx.tolist()
    exp = {1: labels[:3], 2: labels[3:]}
    tm.assert_dict_equal(groups, exp)

    # GH5620
    groups = idx.groupby(idx)
    exp = {key: [key] for key in idx}
    tm.assert_dict_equal(groups, exp)


def test_truncate_multiindex():
    # GH 34564 for MultiIndex level names check
    major_axis = Index(list(range(4)))
    minor_axis = Index(list(range(2)))

    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])

    index = MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=["L1", "L2"],
    )

    result = index.truncate(before=1)
    assert "foo" not in result.levels[0]
    assert 1 in result.levels[0]
    assert index.names == result.names

    result = index.truncate(after=1)
    assert 2 not in result.levels[0]
    assert 1 in result.levels[0]
    assert index.names == result.names

    result = index.truncate(before=1, after=2)
    assert len(result.levels[0]) == 2
    assert index.names == result.names

    msg = "after < before"
    with pytest.raises(ValueError, match=msg):
        index.truncate(3, 1)


# TODO: reshape


def test_reorder_levels(idx):
    # this blows up
    with pytest.raises(IndexError, match="^Too many levels"):
        idx.reorder_levels([2, 1, 0])


def test_numpy_repeat():
    reps = 2
    numbers = [1, 2, 3]
    names = np.array(["foo", "bar"])

    m = MultiIndex.from_product([numbers, names], names=names)
    expected = MultiIndex.from_product([numbers, names.repeat(reps)], names=names)
    tm.assert_index_equal(np.repeat(m, reps), expected)

    msg = "the 'axis' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        np.repeat(m, reps, axis=1)


def test_append_mixed_dtypes():
    # GH 13660
    dti = date_range("2011-01-01", freq="ME", periods=3)
    dti_tz = date_range("2011-01-01", freq="ME", periods=3, tz="US/Eastern")
    pi = period_range("2011-01", freq="M", periods=3)

    mi = MultiIndex.from_arrays(
        [[1, 2, 3], [1.1, np.nan, 3.3], ["a", "b", "c"], dti, dti_tz, pi]
    )
    assert mi.nlevels == 6

    res = mi.append(mi)
    exp = MultiIndex.from_arrays(
        [
            [1, 2, 3, 1, 2, 3],
            [1.1, np.nan, 3.3, 1.1, np.nan, 3.3],
            ["a", "b", "c", "a", "b", "c"],
            dti.append(dti),
            dti_tz.append(dti_tz),
            pi.append(pi),
        ]
    )
    tm.assert_index_equal(res, exp)

    other = MultiIndex.from_arrays(
        [
            ["x", "y", "z"],
            ["x", "y", "z"],
            ["x", "y", "z"],
            ["x", "y", "z"],
            ["x", "y", "z"],
            ["x", "y", "z"],
        ]
    )

    res = mi.append(other)
    exp = MultiIndex.from_arrays(
        [
            [1, 2, 3, "x", "y", "z"],
            [1.1, np.nan, 3.3, "x", "y", "z"],
            ["a", "b", "c", "x", "y", "z"],
            dti.append(Index(["x", "y", "z"])),
            dti_tz.append(Index(["x", "y", "z"])),
            pi.append(Index(["x", "y", "z"])),
        ]
    )
    tm.assert_index_equal(res, exp)


def test_iter(idx):
    result = list(idx)
    expected = [
        ("foo", "one"),
        ("foo", "two"),
        ("bar", "one"),
        ("baz", "two"),
        ("qux", "one"),
        ("qux", "two"),
    ]
    assert result == expected


def test_sub(idx):
    first = idx

    # - now raises (previously was set op difference)
    msg = "cannot perform __sub__ with this index type: MultiIndex"
    with pytest.raises(TypeError, match=msg):
        first - idx[-3:]
    with pytest.raises(TypeError, match=msg):
        idx[-3:] - first
    with pytest.raises(TypeError, match=msg):
        idx[-3:] - first.tolist()
    msg = "cannot perform __rsub__ with this index type: MultiIndex"
    with pytest.raises(TypeError, match=msg):
        first.tolist() - idx[-3:]


def test_map(idx):
    # callable
    index = idx

    result = index.map(lambda x: x)
    tm.assert_index_equal(result, index)


@pytest.mark.parametrize(
    "mapper",
    [
        lambda values, idx: {i: e for e, i in zip(values, idx)},
        lambda values, idx: pd.Series(values, idx),
    ],
)
def test_map_dictlike(idx, mapper):
    identity = mapper(idx.values, idx)

    # we don't infer to uint64 dtype for a dict
    if idx.dtype == np.uint64 and isinstance(identity, dict):
        expected = idx.astype("int64")
    else:
        expected = idx

    result = idx.map(identity)
    tm.assert_index_equal(result, expected)

    # empty mappable
    expected = Index([np.nan] * len(idx))
    result = idx.map(mapper(expected, idx))
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "func",
    [
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
        np.sqrt,
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.deg2rad,
        np.rad2deg,
    ],
    ids=lambda func: func.__name__,
)
def test_numpy_ufuncs(idx, func):
    # test ufuncs of numpy. see:
    # https://numpy.org/doc/stable/reference/ufuncs.html

    expected_exception = TypeError
    msg = (
        "loop of ufunc does not support argument 0 of type tuple which "
        f"has no callable {func.__name__} method"
    )
    with pytest.raises(expected_exception, match=msg):
        func(idx)


@pytest.mark.parametrize(
    "func",
    [np.isfinite, np.isinf, np.isnan, np.signbit],
    ids=lambda func: func.__name__,
)
def test_numpy_type_funcs(idx, func):
    msg = (
        f"ufunc '{func.__name__}' not supported for the input types, and the inputs "
        "could not be safely coerced to any supported types according to "
        "the casting rule ''safe''"
    )
    with pytest.raises(TypeError, match=msg):
        func(idx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_converters.py ===
"""
Tests column conversion functionality during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO

from dateutil.parser import parse
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


def test_converters_type_must_be_dict(all_parsers):
    parser = all_parsers
    data = """index,A,B,C,D
foo,2,3,4,5
"""
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters=0)
        return
    with pytest.raises(TypeError, match="Type converters.+"):
        parser.read_csv(StringIO(data), converters=0)


@pytest.mark.parametrize("column", [3, "D"])
@pytest.mark.parametrize(
    "converter", [parse, lambda x: int(x.split("/")[2])]  # Produce integer.
)
def test_converters(all_parsers, column, converter):
    parser = all_parsers
    data = """A,B,C,D
a,1,2,01/01/2009
b,3,4,01/02/2009
c,4,5,01/03/2009
"""
    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters={column: converter})
        return

    result = parser.read_csv(StringIO(data), converters={column: converter})

    expected = parser.read_csv(StringIO(data))
    expected["D"] = expected["D"].map(converter)

    tm.assert_frame_equal(result, expected)


def test_converters_no_implicit_conv(all_parsers):
    # see gh-2184
    parser = all_parsers
    data = """000102,1.2,A\n001245,2,B"""

    converters = {0: lambda x: x.strip()}

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=None, converters=converters)
        return

    result = parser.read_csv(StringIO(data), header=None, converters=converters)

    # Column 0 should not be casted to numeric and should remain as object.
    expected = DataFrame([["000102", 1.2, "A"], ["001245", 2, "B"]])
    tm.assert_frame_equal(result, expected)


def test_converters_euro_decimal_format(all_parsers):
    # see gh-583
    converters = {}
    parser = all_parsers

    data = """Id;Number1;Number2;Text1;Text2;Number3
1;1521,1541;187101,9543;ABC;poi;4,7387
2;121,12;14897,76;DEF;uyt;0,3773
3;878,158;108013,434;GHI;rez;2,7356"""
    converters["Number1"] = converters["Number2"] = converters[
        "Number3"
    ] = lambda x: float(x.replace(",", "."))

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=";", converters=converters)
        return

    result = parser.read_csv(StringIO(data), sep=";", converters=converters)
    expected = DataFrame(
        [
            [1, 1521.1541, 187101.9543, "ABC", "poi", 4.7387],
            [2, 121.12, 14897.76, "DEF", "uyt", 0.3773],
            [3, 878.158, 108013.434, "GHI", "rez", 2.7356],
        ],
        columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],
    )
    tm.assert_frame_equal(result, expected)


def test_converters_corner_with_nans(all_parsers):
    parser = all_parsers
    data = """id,score,days
1,2,12
2,2-5,
3,,14+
4,6-12,2"""

    # Example converters.
    def convert_days(x):
        x = x.strip()

        if not x:
            return np.nan

        is_plus = x.endswith("+")

        if is_plus:
            x = int(x[:-1]) + 1
        else:
            x = int(x)

        return x

    def convert_days_sentinel(x):
        x = x.strip()

        if not x:
            return np.nan

        is_plus = x.endswith("+")

        if is_plus:
            x = int(x[:-1]) + 1
        else:
            x = int(x)

        return x

    def convert_score(x):
        x = x.strip()

        if not x:
            return np.nan

        if x.find("-") > 0:
            val_min, val_max = map(int, x.split("-"))
            val = 0.5 * (val_min + val_max)
        else:
            val = float(x)

        return val

    results = []

    for day_converter in [convert_days, convert_days_sentinel]:
        if parser.engine == "pyarrow":
            msg = "The 'converters' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(
                    StringIO(data),
                    converters={"score": convert_score, "days": day_converter},
                    na_values=["", None],
                )
            continue

        result = parser.read_csv(
            StringIO(data),
            converters={"score": convert_score, "days": day_converter},
            na_values=["", None],
        )
        assert pd.isna(result["days"][1])
        results.append(result)

    if parser.engine != "pyarrow":
        tm.assert_frame_equal(results[0], results[1])


@pytest.mark.parametrize("conv_f", [lambda x: x, str])
def test_converter_index_col_bug(all_parsers, conv_f):
    # see gh-1835 , GH#40589
    parser = all_parsers
    data = "A;B\n1;2\n3;4"

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), sep=";", index_col="A", converters={"A": conv_f}
            )
        return

    rs = parser.read_csv(
        StringIO(data), sep=";", index_col="A", converters={"A": conv_f}
    )

    xp = DataFrame({"B": [2, 4]}, index=Index(["1", "3"], name="A"))
    tm.assert_frame_equal(rs, xp)


def test_converter_identity_object(all_parsers):
    # GH#40589
    parser = all_parsers
    data = "A,B\n1,2\n3,4"

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters={"A": lambda x: x})
        return

    rs = parser.read_csv(StringIO(data), converters={"A": lambda x: x})

    xp = DataFrame({"A": ["1", "3"], "B": [2, 4]})
    tm.assert_frame_equal(rs, xp)


def test_converter_multi_index(all_parsers):
    # GH 42446
    parser = all_parsers
    data = "A,B,B\nX,Y,Z\n1,2,3"

    if parser.engine == "pyarrow":
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                header=list(range(2)),
                converters={
                    ("A", "X"): np.int32,
                    ("B", "Y"): np.int32,
                    ("B", "Z"): np.float32,
                },
            )
        return

    result = parser.read_csv(
        StringIO(data),
        header=list(range(2)),
        converters={
            ("A", "X"): np.int32,
            ("B", "Y"): np.int32,
            ("B", "Z"): np.float32,
        },
    )

    expected = DataFrame(
        {
            ("A", "X"): np.int32([1]),
            ("B", "Y"): np.int32([2]),
            ("B", "Z"): np.float32([3]),
        }
    )

    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_align.py ===
from datetime import timezone

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "first_slice,second_slice",
    [
        [[2, None], [None, -5]],
        [[None, 0], [None, -5]],
        [[None, -5], [None, 0]],
        [[None, 0], [None, 0]],
    ],
)
@pytest.mark.parametrize("fill", [None, -1])
def test_align(datetime_series, first_slice, second_slice, join_type, fill):
    a = datetime_series[slice(*first_slice)]
    b = datetime_series[slice(*second_slice)]

    aa, ab = a.align(b, join=join_type, fill_value=fill)

    join_index = a.index.join(b.index, how=join_type)
    if fill is not None:
        diff_a = aa.index.difference(join_index)
        diff_b = ab.index.difference(join_index)
        if len(diff_a) > 0:
            assert (aa.reindex(diff_a) == fill).all()
        if len(diff_b) > 0:
            assert (ab.reindex(diff_b) == fill).all()

    ea = a.reindex(join_index)
    eb = b.reindex(join_index)

    if fill is not None:
        ea = ea.fillna(fill)
        eb = eb.fillna(fill)

    tm.assert_series_equal(aa, ea)
    tm.assert_series_equal(ab, eb)
    assert aa.name == "ts"
    assert ea.name == "ts"
    assert ab.name == "ts"
    assert eb.name == "ts"


@pytest.mark.parametrize(
    "first_slice,second_slice",
    [
        [[2, None], [None, -5]],
        [[None, 0], [None, -5]],
        [[None, -5], [None, 0]],
        [[None, 0], [None, 0]],
    ],
)
@pytest.mark.parametrize("method", ["pad", "bfill"])
@pytest.mark.parametrize("limit", [None, 1])
def test_align_fill_method(
    datetime_series, first_slice, second_slice, join_type, method, limit
):
    a = datetime_series[slice(*first_slice)]
    b = datetime_series[slice(*second_slice)]

    msg = (
        "The 'method', 'limit', and 'fill_axis' keywords in Series.align "
        "are deprecated"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        aa, ab = a.align(b, join=join_type, method=method, limit=limit)

    join_index = a.index.join(b.index, how=join_type)
    ea = a.reindex(join_index)
    eb = b.reindex(join_index)

    msg2 = "Series.fillna with 'method' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        ea = ea.fillna(method=method, limit=limit)
        eb = eb.fillna(method=method, limit=limit)

    tm.assert_series_equal(aa, ea)
    tm.assert_series_equal(ab, eb)


def test_align_nocopy(datetime_series, using_copy_on_write):
    b = datetime_series[:5].copy()

    # do copy
    a = datetime_series.copy()
    ra, _ = a.align(b, join="left")
    ra[:5] = 5
    assert not (a[:5] == 5).any()

    # do not copy
    a = datetime_series.copy()
    ra, _ = a.align(b, join="left", copy=False)
    ra[:5] = 5
    if using_copy_on_write:
        assert not (a[:5] == 5).any()
    else:
        assert (a[:5] == 5).all()

    # do copy
    a = datetime_series.copy()
    b = datetime_series[:5].copy()
    _, rb = a.align(b, join="right")
    rb[:3] = 5
    assert not (b[:3] == 5).any()

    # do not copy
    a = datetime_series.copy()
    b = datetime_series[:5].copy()
    _, rb = a.align(b, join="right", copy=False)
    rb[:2] = 5
    if using_copy_on_write:
        assert not (b[:2] == 5).any()
    else:
        assert (b[:2] == 5).all()


def test_align_same_index(datetime_series, using_copy_on_write):
    a, b = datetime_series.align(datetime_series, copy=False)
    if not using_copy_on_write:
        assert a.index is datetime_series.index
        assert b.index is datetime_series.index
    else:
        assert a.index.is_(datetime_series.index)
        assert b.index.is_(datetime_series.index)

    a, b = datetime_series.align(datetime_series, copy=True)
    assert a.index is not datetime_series.index
    assert b.index is not datetime_series.index
    assert a.index.is_(datetime_series.index)
    assert b.index.is_(datetime_series.index)


def test_align_multiindex():
    # GH 10665

    midx = pd.MultiIndex.from_product(
        [range(2), range(3), range(2)], names=("a", "b", "c")
    )
    idx = pd.Index(range(2), name="b")
    s1 = Series(np.arange(12, dtype="int64"), index=midx)
    s2 = Series(np.arange(2, dtype="int64"), index=idx)

    # these must be the same results (but flipped)
    res1l, res1r = s1.align(s2, join="left")
    res2l, res2r = s2.align(s1, join="right")

    expl = s1
    tm.assert_series_equal(expl, res1l)
    tm.assert_series_equal(expl, res2r)
    expr = Series([0, 0, 1, 1, np.nan, np.nan] * 2, index=midx)
    tm.assert_series_equal(expr, res1r)
    tm.assert_series_equal(expr, res2l)

    res1l, res1r = s1.align(s2, join="right")
    res2l, res2r = s2.align(s1, join="left")

    exp_idx = pd.MultiIndex.from_product(
        [range(2), range(2), range(2)], names=("a", "b", "c")
    )
    expl = Series([0, 1, 2, 3, 6, 7, 8, 9], index=exp_idx)
    tm.assert_series_equal(expl, res1l)
    tm.assert_series_equal(expl, res2r)
    expr = Series([0, 0, 1, 1] * 2, index=exp_idx)
    tm.assert_series_equal(expr, res1r)
    tm.assert_series_equal(expr, res2l)


@pytest.mark.parametrize("method", ["backfill", "bfill", "pad", "ffill", None])
def test_align_with_dataframe_method(method):
    # GH31788
    ser = Series(range(3), index=range(3))
    df = pd.DataFrame(0.0, index=range(3), columns=range(3))

    msg = (
        "The 'method', 'limit', and 'fill_axis' keywords in Series.align "
        "are deprecated"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result_ser, result_df = ser.align(df, method=method)
    tm.assert_series_equal(result_ser, ser)
    tm.assert_frame_equal(result_df, df)


def test_align_dt64tzindex_mismatched_tzs():
    idx1 = date_range("2001", periods=5, freq="h", tz="US/Eastern")
    ser = Series(np.random.default_rng(2).standard_normal(len(idx1)), index=idx1)
    ser_central = ser.tz_convert("US/Central")
    # different timezones convert to UTC

    new1, new2 = ser.align(ser_central)
    assert new1.index.tz is timezone.utc
    assert new2.index.tz is timezone.utc


def test_align_periodindex(join_type):
    rng = period_range("1/1/2000", "1/1/2010", freq="Y")
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    # TODO: assert something?
    ts.align(ts[::2], join=join_type)


def test_align_stringindex(any_string_dtype):
    left = Series(range(3), index=pd.Index(["a", "b", "d"], dtype=any_string_dtype))
    right = Series(range(3), index=pd.Index(["a", "b", "c"], dtype=any_string_dtype))
    result_left, result_right = left.align(right)

    expected_idx = pd.Index(["a", "b", "c", "d"], dtype=any_string_dtype)
    expected_left = Series([0, 1, np.nan, 2], index=expected_idx)
    expected_right = Series([0, 1, 2, np.nan], index=expected_idx)

    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)


def test_align_left_fewer_levels():
    # GH#45224
    left = Series([2], index=pd.MultiIndex.from_tuples([(1, 3)], names=["a", "c"]))
    right = Series(
        [1], index=pd.MultiIndex.from_tuples([(1, 2, 3)], names=["a", "b", "c"])
    )
    result_left, result_right = left.align(right)

    expected_right = Series(
        [1], index=pd.MultiIndex.from_tuples([(1, 3, 2)], names=["a", "c", "b"])
    )
    expected_left = Series(
        [2], index=pd.MultiIndex.from_tuples([(1, 3, 2)], names=["a", "c", "b"])
    )
    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)


def test_align_left_different_named_levels():
    # GH#45224
    left = Series(
        [2], index=pd.MultiIndex.from_tuples([(1, 4, 3)], names=["a", "d", "c"])
    )
    right = Series(
        [1], index=pd.MultiIndex.from_tuples([(1, 2, 3)], names=["a", "b", "c"])
    )
    result_left, result_right = left.align(right)

    expected_left = Series(
        [2], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=["a", "d", "c", "b"])
    )
    expected_right = Series(
        [1], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=["a", "d", "c", "b"])
    )
    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_functions.py ===
from sympy import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
                                     RigidBody)
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
                                     kinetic_energy, linear_momentum,
                                     outer, potential_energy, msubs,
                                     find_dynamicsymbols, Lagrangian)

from sympy.physics.mechanics.functions import (
    center_of_mass, _validate_coordinates, _parse_linear_solver)
from sympy.testing.pytest import raises, warns_deprecated_sympy


q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
N = ReferenceFrame('N')
A = N.orientnew('A', 'Axis', [q1, N.z])
B = A.orientnew('B', 'Axis', [q2, A.x])
C = B.orientnew('C', 'Axis', [q3, B.y])


def test_linear_momentum():
    N = ReferenceFrame('N')
    Ac = Point('Ac')
    Ac.set_vel(N, 25 * N.y)
    I = outer(N.x, N.x)
    A = RigidBody('A', Ac, N, 20, (I, Ac))
    P = Point('P')
    Pa = Particle('Pa', P, 1)
    Pa.point.set_vel(N, 10 * N.x)
    raises(TypeError, lambda: linear_momentum(A, A, Pa))
    raises(TypeError, lambda: linear_momentum(N, N, Pa))
    assert linear_momentum(N, A, Pa) == 10 * N.x + 500 * N.y


def test_angular_momentum_and_linear_momentum():
    """A rod with length 2l, centroidal inertia I, and mass M along with a
    particle of mass m fixed to the end of the rod rotate with an angular rate
    of omega about point O which is fixed to the non-particle end of the rod.
    The rod's reference frame is A and the inertial frame is N."""
    m, M, l, I = symbols('m, M, l, I')
    omega = dynamicsymbols('omega')
    N = ReferenceFrame('N')
    a = ReferenceFrame('a')
    O = Point('O')
    Ac = O.locatenew('Ac', l * N.x)
    P = Ac.locatenew('P', l * N.x)
    O.set_vel(N, 0 * N.x)
    a.set_ang_vel(N, omega * N.z)
    Ac.v2pt_theory(O, N, a)
    P.v2pt_theory(O, N, a)
    Pa = Particle('Pa', P, m)
    A = RigidBody('A', Ac, a, M, (I * outer(N.z, N.z), Ac))
    expected = 2 * m * omega * l * N.y + M * l * omega * N.y
    assert linear_momentum(N, A, Pa) == expected
    raises(TypeError, lambda: angular_momentum(N, N, A, Pa))
    raises(TypeError, lambda: angular_momentum(O, O, A, Pa))
    raises(TypeError, lambda: angular_momentum(O, N, O, Pa))
    expected = (I + M * l**2 + 4 * m * l**2) * omega * N.z
    assert angular_momentum(O, N, A, Pa) == expected


def test_kinetic_energy():
    m, M, l1 = symbols('m M l1')
    omega = dynamicsymbols('omega')
    N = ReferenceFrame('N')
    O = Point('O')
    O.set_vel(N, 0 * N.x)
    Ac = O.locatenew('Ac', l1 * N.x)
    P = Ac.locatenew('P', l1 * N.x)
    a = ReferenceFrame('a')
    a.set_ang_vel(N, omega * N.z)
    Ac.v2pt_theory(O, N, a)
    P.v2pt_theory(O, N, a)
    Pa = Particle('Pa', P, m)
    I = outer(N.z, N.z)
    A = RigidBody('A', Ac, a, M, (I, Ac))
    raises(TypeError, lambda: kinetic_energy(Pa, Pa, A))
    raises(TypeError, lambda: kinetic_energy(N, N, A))
    assert 0 == (kinetic_energy(N, Pa, A) - (M*l1**2*omega**2/2
            + 2*l1**2*m*omega**2 + omega**2/2)).expand()


def test_potential_energy():
    m, M, l1, g, h, H = symbols('m M l1 g h H')
    omega = dynamicsymbols('omega')
    N = ReferenceFrame('N')
    O = Point('O')
    O.set_vel(N, 0 * N.x)
    Ac = O.locatenew('Ac', l1 * N.x)
    P = Ac.locatenew('P', l1 * N.x)
    a = ReferenceFrame('a')
    a.set_ang_vel(N, omega * N.z)
    Ac.v2pt_theory(O, N, a)
    P.v2pt_theory(O, N, a)
    Pa = Particle('Pa', P, m)
    I = outer(N.z, N.z)
    A = RigidBody('A', Ac, a, M, (I, Ac))
    Pa.potential_energy = m * g * h
    A.potential_energy = M * g * H
    assert potential_energy(A, Pa) == m * g * h + M * g * H


def test_Lagrangian():
    M, m, g, h = symbols('M m g h')
    N = ReferenceFrame('N')
    O = Point('O')
    O.set_vel(N, 0 * N.x)
    P = O.locatenew('P', 1 * N.x)
    P.set_vel(N, 10 * N.x)
    Pa = Particle('Pa', P, 1)
    Ac = O.locatenew('Ac', 2 * N.y)
    Ac.set_vel(N, 5 * N.y)
    a = ReferenceFrame('a')
    a.set_ang_vel(N, 10 * N.z)
    I = outer(N.z, N.z)
    A = RigidBody('A', Ac, a, 20, (I, Ac))
    Pa.potential_energy = m * g * h
    A.potential_energy = M * g * h
    raises(TypeError, lambda: Lagrangian(A, A, Pa))
    raises(TypeError, lambda: Lagrangian(N, N, Pa))


def test_msubs():
    a, b = symbols('a, b')
    x, y, z = dynamicsymbols('x, y, z')
    # Test simple substitution
    expr = Matrix([[a*x + b, x*y.diff() + y],
                   [x.diff().diff(), z + sin(z.diff())]])
    sol = Matrix([[a + b, y],
                  [x.diff().diff(), 1]])
    sd = {x: 1, z: 1, z.diff(): 0, y.diff(): 0}
    assert msubs(expr, sd) == sol
    # Test smart substitution
    expr = cos(x + y)*tan(x + y) + b*x.diff()
    sd = {x: 0, y: pi/2, x.diff(): 1}
    assert msubs(expr, sd, smart=True) == b + 1
    N = ReferenceFrame('N')
    v = x*N.x + y*N.y
    d = x*(N.x|N.x) + y*(N.y|N.y)
    v_sol = 1*N.y
    d_sol = 1*(N.y|N.y)
    sd = {x: 0, y: 1}
    assert msubs(v, sd) == v_sol
    assert msubs(d, sd) == d_sol


def test_find_dynamicsymbols():
    a, b = symbols('a, b')
    x, y, z = dynamicsymbols('x, y, z')
    expr = Matrix([[a*x + b, x*y.diff() + y],
                   [x.diff().diff(), z + sin(z.diff())]])
    # Test finding all dynamicsymbols
    sol = {x, y.diff(), y, x.diff().diff(), z, z.diff()}
    assert find_dynamicsymbols(expr) == sol
    # Test finding all but those in sym_list
    exclude_list = [x, y, z]
    sol = {y.diff(), x.diff().diff(), z.diff()}
    assert find_dynamicsymbols(expr, exclude=exclude_list) == sol
    # Test finding all dynamicsymbols in a vector with a given reference frame
    d, e, f = dynamicsymbols('d, e, f')
    A = ReferenceFrame('A')
    v = d * A.x + e * A.y + f * A.z
    sol = {d, e, f}
    assert find_dynamicsymbols(v, reference_frame=A) == sol
    # Test if a ValueError is raised on supplying only a vector as input
    raises(ValueError, lambda: find_dynamicsymbols(v))


# This function tests the center_of_mass() function
# that was added in PR #14758 to compute the center of
# mass of a system of bodies.
def test_center_of_mass():
    a = ReferenceFrame('a')
    m = symbols('m', real=True)
    p1 = Particle('p1', Point('p1_pt'), S.One)
    p2 = Particle('p2', Point('p2_pt'), S(2))
    p3 = Particle('p3', Point('p3_pt'), S(3))
    p4 = Particle('p4', Point('p4_pt'), m)
    b_f = ReferenceFrame('b_f')
    b_cm = Point('b_cm')
    mb = symbols('mb')
    b = RigidBody('b', b_cm, b_f, mb, (outer(b_f.x, b_f.x), b_cm))
    p2.point.set_pos(p1.point, a.x)
    p3.point.set_pos(p1.point, a.x + a.y)
    p4.point.set_pos(p1.point, a.y)
    b.masscenter.set_pos(p1.point, a.y + a.z)
    point_o=Point('o')
    point_o.set_pos(p1.point, center_of_mass(p1.point, p1, p2, p3, p4, b))
    expr = 5/(m + mb + 6)*a.x + (m + mb + 3)/(m + mb + 6)*a.y + mb/(m + mb + 6)*a.z
    assert point_o.pos_from(p1.point)-expr == 0


def test_validate_coordinates():
    q1, q2, q3, u1, u2, u3, ua1, ua2, ua3 = dynamicsymbols('q1:4 u1:4 ua1:4')
    s1, s2, s3 = symbols('s1:4')
    # Test normal
    _validate_coordinates([q1, q2, q3], [u1, u2, u3],
                          u_auxiliary=[ua1, ua2, ua3])
    # Test not equal number of coordinates and speeds
    _validate_coordinates([q1, q2])
    _validate_coordinates([q1, q2], [u1])
    _validate_coordinates(speeds=[u1, u2])
    # Test duplicate
    _validate_coordinates([q1, q2, q2], [u1, u2, u3], check_duplicates=False)
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q2], [u1, u2, u3]))
    _validate_coordinates([q1, q2, q3], [u1, u2, u2], check_duplicates=False)
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [u1, u2, u2], check_duplicates=True))
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [q1, u2, u3], check_duplicates=True))
    _validate_coordinates([q1, q2, q3], [u1, u2, u3], check_duplicates=False,
                          u_auxiliary=[u1, ua2, ua2])
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [u1, u2, u3], u_auxiliary=[u1, ua2, ua3]))
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [u1, u2, u3], u_auxiliary=[q1, ua2, ua3]))
    raises(ValueError, lambda: _validate_coordinates(
        [q1, q2, q3], [u1, u2, u3], u_auxiliary=[ua1, ua2, ua2]))
    # Test is_dynamicsymbols
    _validate_coordinates([q1 + q2, q3], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([q1 + q2, q3]))
    _validate_coordinates([s1, q1, q2], [0, u1, u2], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates(
        [s1, q1, q2], [0, u1, u2], is_dynamicsymbols=True))
    _validate_coordinates([s1 + s2 + s3, q1], [0, u1], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates(
        [s1 + s2 + s3, q1], [0, u1], is_dynamicsymbols=True))
    _validate_coordinates(u_auxiliary=[s1, ua1], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates(u_auxiliary=[s1, ua1]))
    # Test normal function
    t = dynamicsymbols._t
    a = symbols('a')
    f1, f2 = symbols('f1:3', cls=Function)
    _validate_coordinates([f1(a), f2(a)], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([f1(a), f2(a)]))
    raises(ValueError, lambda: _validate_coordinates(speeds=[f1(a), f2(a)]))
    dynamicsymbols._t = a
    _validate_coordinates([f1(a), f2(a)])
    raises(ValueError, lambda: _validate_coordinates([f1(t), f2(t)]))
    dynamicsymbols._t = t


def test_parse_linear_solver():
    A, b = Matrix(3, 3, symbols('a:9')), Matrix(3, 2, symbols('b:6'))
    assert _parse_linear_solver(Matrix.LUsolve) == Matrix.LUsolve  # Test callable
    assert _parse_linear_solver('LU')(A, b) == Matrix.LUsolve(A, b)


def test_deprecated_moved_functions():
    from sympy.physics.mechanics.functions import (
        inertia, inertia_of_point_mass, gravity)
    N = ReferenceFrame('N')
    with warns_deprecated_sympy():
        assert inertia(N, 0, 1, 0, 1) == (N.x | N.y) + (N.y | N.x) + (N.y | N.y)
    with warns_deprecated_sympy():
        assert inertia_of_point_mass(1, N.x + N.y, N) == (
            (N.x | N.x) + (N.y | N.y) + 2 * (N.z | N.z) -
            (N.x | N.y) - (N.y | N.x))
    p = Particle('P')
    with warns_deprecated_sympy():
        assert gravity(-2 * N.z, p) == [(p.masscenter, -2 * p.mass * N.z)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_dst.py ===
"""
Tests for DateOffset additions over Daylight Savings Time
"""
from datetime import timedelta

import pytest
import pytz

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    CBMonthBegin,
    CBMonthEnd,
    CustomBusinessDay,
    DateOffset,
    Day,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Week,
    YearBegin,
    YearEnd,
)
from pandas.errors import PerformanceWarning

from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.util.version import Version

# error: Module has no attribute "__version__"
pytz_version = Version(pytz.__version__)  # type: ignore[attr-defined]


def get_utc_offset_hours(ts):
    # take a Timestamp and compute total hours of utc offset
    o = ts.utcoffset()
    return (o.days * 24 * 3600 + o.seconds) / 3600.0


class TestDST:
    # one microsecond before the DST transition
    ts_pre_fallback = "2013-11-03 01:59:59.999999"
    ts_pre_springfwd = "2013-03-10 01:59:59.999999"

    # test both basic names and dateutil timezones
    timezone_utc_offsets = {
        "US/Eastern": {"utc_offset_daylight": -4, "utc_offset_standard": -5},
        "dateutil/US/Pacific": {"utc_offset_daylight": -7, "utc_offset_standard": -8},
    }
    valid_date_offsets_singular = [
        "weekday",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
    ]
    valid_date_offsets_plural = [
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
    ]

    def _test_all_offsets(self, n, **kwds):
        valid_offsets = (
            self.valid_date_offsets_plural
            if n > 1
            else self.valid_date_offsets_singular
        )

        for name in valid_offsets:
            self._test_offset(offset_name=name, offset_n=n, **kwds)

    def _test_offset(self, offset_name, offset_n, tstart, expected_utc_offset):
        offset = DateOffset(**{offset_name: offset_n})

        if (
            offset_name in ["hour", "minute", "second", "microsecond"]
            and offset_n == 1
            and tstart == Timestamp("2013-11-03 01:59:59.999999-0500", tz="US/Eastern")
        ):
            # This addition results in an ambiguous wall time
            err_msg = {
                "hour": "2013-11-03 01:59:59.999999",
                "minute": "2013-11-03 01:01:59.999999",
                "second": "2013-11-03 01:59:01.999999",
                "microsecond": "2013-11-03 01:59:59.000001",
            }[offset_name]
            with pytest.raises(pytz.AmbiguousTimeError, match=err_msg):
                tstart + offset
            # While we're here, let's check that we get the same behavior in a
            #  vectorized path
            dti = DatetimeIndex([tstart])
            warn_msg = "Non-vectorized DateOffset"
            with pytest.raises(pytz.AmbiguousTimeError, match=err_msg):
                with tm.assert_produces_warning(PerformanceWarning, match=warn_msg):
                    dti + offset
            return

        t = tstart + offset
        if expected_utc_offset is not None:
            assert get_utc_offset_hours(t) == expected_utc_offset

        if offset_name == "weeks":
            # dates should match
            assert t.date() == timedelta(days=7 * offset.kwds["weeks"]) + tstart.date()
            # expect the same day of week, hour of day, minute, second, ...
            assert (
                t.dayofweek == tstart.dayofweek
                and t.hour == tstart.hour
                and t.minute == tstart.minute
                and t.second == tstart.second
            )
        elif offset_name == "days":
            # dates should match
            assert timedelta(offset.kwds["days"]) + tstart.date() == t.date()
            # expect the same hour of day, minute, second, ...
            assert (
                t.hour == tstart.hour
                and t.minute == tstart.minute
                and t.second == tstart.second
            )
        elif offset_name in self.valid_date_offsets_singular:
            # expect the singular offset value to match between tstart and t
            datepart_offset = getattr(
                t, offset_name if offset_name != "weekday" else "dayofweek"
            )
            assert datepart_offset == offset.kwds[offset_name]
        else:
            # the offset should be the same as if it was done in UTC
            assert t == (tstart.tz_convert("UTC") + offset).tz_convert("US/Pacific")

    def _make_timestamp(self, string, hrs_offset, tz):
        if hrs_offset >= 0:
            offset_string = f"{hrs_offset:02d}00"
        else:
            offset_string = f"-{(hrs_offset * -1):02}00"
        return Timestamp(string + offset_string).tz_convert(tz)

    def test_springforward_plural(self):
        # test moving from standard to daylight savings
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            hrs_pre = utc_offsets["utc_offset_standard"]
            hrs_post = utc_offsets["utc_offset_daylight"]
            self._test_all_offsets(
                n=3,
                tstart=self._make_timestamp(self.ts_pre_springfwd, hrs_pre, tz),
                expected_utc_offset=hrs_post,
            )

    def test_fallback_singular(self):
        # in the case of singular offsets, we don't necessarily know which utc
        # offset the new Timestamp will wind up in (the tz for 1 month may be
        # different from 1 second) so we don't specify an expected_utc_offset
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            hrs_pre = utc_offsets["utc_offset_standard"]
            self._test_all_offsets(
                n=1,
                tstart=self._make_timestamp(self.ts_pre_fallback, hrs_pre, tz),
                expected_utc_offset=None,
            )

    def test_springforward_singular(self):
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            hrs_pre = utc_offsets["utc_offset_standard"]
            self._test_all_offsets(
                n=1,
                tstart=self._make_timestamp(self.ts_pre_springfwd, hrs_pre, tz),
                expected_utc_offset=None,
            )

    offset_classes = {
        MonthBegin: ["11/2/2012", "12/1/2012"],
        MonthEnd: ["11/2/2012", "11/30/2012"],
        BMonthBegin: ["11/2/2012", "12/3/2012"],
        BMonthEnd: ["11/2/2012", "11/30/2012"],
        CBMonthBegin: ["11/2/2012", "12/3/2012"],
        CBMonthEnd: ["11/2/2012", "11/30/2012"],
        SemiMonthBegin: ["11/2/2012", "11/15/2012"],
        SemiMonthEnd: ["11/2/2012", "11/15/2012"],
        Week: ["11/2/2012", "11/9/2012"],
        YearBegin: ["11/2/2012", "1/1/2013"],
        YearEnd: ["11/2/2012", "12/31/2012"],
        BYearBegin: ["11/2/2012", "1/1/2013"],
        BYearEnd: ["11/2/2012", "12/31/2012"],
        QuarterBegin: ["11/2/2012", "12/1/2012"],
        QuarterEnd: ["11/2/2012", "12/31/2012"],
        BQuarterBegin: ["11/2/2012", "12/3/2012"],
        BQuarterEnd: ["11/2/2012", "12/31/2012"],
        Day: ["11/4/2012", "11/4/2012 23:00"],
    }.items()

    @pytest.mark.parametrize("tup", offset_classes)
    def test_all_offset_classes(self, tup):
        offset, test_values = tup

        first = Timestamp(test_values[0], tz="US/Eastern") + offset()
        second = Timestamp(test_values[1], tz="US/Eastern")
        assert first == second


@pytest.mark.parametrize(
    "original_dt, target_dt, offset, tz",
    [
        pytest.param(
            Timestamp("1900-01-01"),
            Timestamp("1905-07-01"),
            MonthBegin(66),
            "Africa/Lagos",
            marks=pytest.mark.xfail(
                pytz_version < Version("2020.5") or pytz_version == Version("2022.2"),
                reason="GH#41906: pytz utc transition dates changed",
            ),
        ),
        (
            Timestamp("2021-10-01 01:15"),
            Timestamp("2021-10-31 01:15"),
            MonthEnd(1),
            "Europe/London",
        ),
        (
            Timestamp("2010-12-05 02:59"),
            Timestamp("2010-10-31 02:59"),
            SemiMonthEnd(-3),
            "Europe/Paris",
        ),
        (
            Timestamp("2021-10-31 01:20"),
            Timestamp("2021-11-07 01:20"),
            CustomBusinessDay(2, weekmask="Sun Mon"),
            "US/Eastern",
        ),
        (
            Timestamp("2020-04-03 01:30"),
            Timestamp("2020-11-01 01:30"),
            YearBegin(1, month=11),
            "America/Chicago",
        ),
    ],
)
def test_nontick_offset_with_ambiguous_time_error(original_dt, target_dt, offset, tz):
    # .apply for non-Tick offsets throws AmbiguousTimeError when the target dt
    # is dst-ambiguous
    localized_dt = original_dt.tz_localize(tz)

    msg = f"Cannot infer dst time from {target_dt}, try using the 'ambiguous' argument"
    with pytest.raises(pytz.AmbiguousTimeError, match=msg):
        localized_dt + offset

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\accessors\test_cat_accessor.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties


class TestCatAccessor:
    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.cat.set_categories([1, 2, 3]),
            lambda x: x.cat.reorder_categories([2, 3, 1], ordered=True),
            lambda x: x.cat.rename_categories([1, 2, 3]),
            lambda x: x.cat.remove_unused_categories(),
            lambda x: x.cat.remove_categories([2]),
            lambda x: x.cat.add_categories([4]),
            lambda x: x.cat.as_ordered(),
            lambda x: x.cat.as_unordered(),
        ],
    )
    def test_getname_categorical_accessor(self, method):
        # GH#17509
        ser = Series([1, 2, 3], name="A").astype("category")
        expected = "A"
        result = method(ser).name
        assert result == expected

    def test_cat_accessor(self):
        ser = Series(Categorical(["a", "b", np.nan, "a"]))
        tm.assert_index_equal(ser.cat.categories, Index(["a", "b"]))
        assert not ser.cat.ordered, False

        exp = Categorical(["a", "b", np.nan, "a"], categories=["b", "a"])

        res = ser.cat.set_categories(["b", "a"])
        tm.assert_categorical_equal(res.values, exp)

        ser[:] = "a"
        ser = ser.cat.remove_unused_categories()
        tm.assert_index_equal(ser.cat.categories, Index(["a"]))

    def test_cat_accessor_api(self):
        # GH#9322

        assert Series.cat is CategoricalAccessor
        ser = Series(list("aabbcde")).astype("category")
        assert isinstance(ser.cat, CategoricalAccessor)

        invalid = Series([1])
        with pytest.raises(AttributeError, match="only use .cat accessor"):
            invalid.cat
        assert not hasattr(invalid, "cat")

    def test_cat_accessor_no_new_attributes(self):
        # https://github.com/pandas-dev/pandas/issues/10673
        cat = Series(list("aabbcde")).astype("category")
        with pytest.raises(AttributeError, match="You cannot add any new attribute"):
            cat.cat.xlabel = "a"

    def test_categorical_delegations(self):
        # invalid accessor
        msg = r"Can only use \.cat accessor with a 'category' dtype"
        with pytest.raises(AttributeError, match=msg):
            Series([1, 2, 3]).cat
        with pytest.raises(AttributeError, match=msg):
            Series([1, 2, 3]).cat()
        with pytest.raises(AttributeError, match=msg):
            Series(["a", "b", "c"]).cat
        with pytest.raises(AttributeError, match=msg):
            Series(np.arange(5.0)).cat
        with pytest.raises(AttributeError, match=msg):
            Series([Timestamp("20130101")]).cat

        # Series should delegate calls to '.categories', '.codes', '.ordered'
        # and the methods '.set_categories()' 'drop_unused_categories()' to the
        # categorical
        ser = Series(Categorical(["a", "b", "c", "a"], ordered=True))
        exp_categories = Index(["a", "b", "c"])
        tm.assert_index_equal(ser.cat.categories, exp_categories)
        ser = ser.cat.rename_categories([1, 2, 3])
        exp_categories = Index([1, 2, 3])
        tm.assert_index_equal(ser.cat.categories, exp_categories)

        exp_codes = Series([0, 1, 2, 0], dtype="int8")
        tm.assert_series_equal(ser.cat.codes, exp_codes)

        assert ser.cat.ordered
        ser = ser.cat.as_unordered()
        assert not ser.cat.ordered

        ser = ser.cat.as_ordered()
        assert ser.cat.ordered

        # reorder
        ser = Series(Categorical(["a", "b", "c", "a"], ordered=True))
        exp_categories = Index(["c", "b", "a"])
        exp_values = np.array(["a", "b", "c", "a"], dtype=np.object_)
        ser = ser.cat.set_categories(["c", "b", "a"])
        tm.assert_index_equal(ser.cat.categories, exp_categories)
        tm.assert_numpy_array_equal(ser.values.__array__(), exp_values)
        tm.assert_numpy_array_equal(ser.__array__(), exp_values)

        # remove unused categories
        ser = Series(Categorical(["a", "b", "b", "a"], categories=["a", "b", "c"]))
        exp_categories = Index(["a", "b"])
        exp_values = np.array(["a", "b", "b", "a"], dtype=np.object_)
        ser = ser.cat.remove_unused_categories()
        tm.assert_index_equal(ser.cat.categories, exp_categories)
        tm.assert_numpy_array_equal(ser.values.__array__(), exp_values)
        tm.assert_numpy_array_equal(ser.__array__(), exp_values)

        # This method is likely to be confused, so test that it raises an error
        # on wrong inputs:
        msg = "'Series' object has no attribute 'set_categories'"
        with pytest.raises(AttributeError, match=msg):
            ser.set_categories([4, 3, 2, 1])

        # right: ser.cat.set_categories([4,3,2,1])

        # GH#18862 (let Series.cat.rename_categories take callables)
        ser = Series(Categorical(["a", "b", "c", "a"], ordered=True))
        result = ser.cat.rename_categories(lambda x: x.upper())
        expected = Series(
            Categorical(["A", "B", "C", "A"], categories=["A", "B", "C"], ordered=True)
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "idx",
        [
            date_range("1/1/2015", periods=5),
            date_range("1/1/2015", periods=5, tz="MET"),
            period_range("1/1/2015", freq="D", periods=5),
            timedelta_range("1 days", "10 days"),
        ],
    )
    def test_dt_accessor_api_for_categorical(self, idx):
        # https://github.com/pandas-dev/pandas/issues/10661

        ser = Series(idx)
        cat = ser.astype("category")

        # only testing field (like .day)
        # and bool (is_month_start)
        attr_names = type(ser._values)._datetimelike_ops

        assert isinstance(cat.dt, Properties)

        special_func_defs = [
            ("strftime", ("%Y-%m-%d",), {}),
            ("round", ("D",), {}),
            ("floor", ("D",), {}),
            ("ceil", ("D",), {}),
            ("asfreq", ("D",), {}),
            ("as_unit", ("s"), {}),
        ]
        if idx.dtype == "M8[ns]":
            # exclude dt64tz since that is already localized and would raise
            tup = ("tz_localize", ("UTC",), {})
            special_func_defs.append(tup)
        elif idx.dtype.kind == "M":
            # exclude dt64 since that is not localized so would raise
            tup = ("tz_convert", ("EST",), {})
            special_func_defs.append(tup)

        _special_func_names = [f[0] for f in special_func_defs]

        _ignore_names = ["components", "tz_localize", "tz_convert"]

        func_names = [
            fname
            for fname in dir(ser.dt)
            if not (
                fname.startswith("_")
                or fname in attr_names
                or fname in _special_func_names
                or fname in _ignore_names
            )
        ]

        func_defs = [(fname, (), {}) for fname in func_names]
        func_defs.extend(
            f_def for f_def in special_func_defs if f_def[0] in dir(ser.dt)
        )

        for func, args, kwargs in func_defs:
            warn_cls = []
            if func == "to_period" and getattr(idx, "tz", None) is not None:
                # dropping TZ
                warn_cls.append(UserWarning)
            if func == "to_pydatetime":
                # deprecated to return Index[object]
                warn_cls.append(FutureWarning)
            if warn_cls:
                warn_cls = tuple(warn_cls)
            else:
                warn_cls = None
            with tm.assert_produces_warning(warn_cls):
                res = getattr(cat.dt, func)(*args, **kwargs)
                exp = getattr(ser.dt, func)(*args, **kwargs)

            tm.assert_equal(res, exp)

        for attr in attr_names:
            res = getattr(cat.dt, attr)
            exp = getattr(ser.dt, attr)

            tm.assert_equal(res, exp)

    def test_dt_accessor_api_for_categorical_invalid(self):
        invalid = Series([1, 2, 3]).astype("category")
        msg = "Can only use .dt accessor with datetimelike"

        with pytest.raises(AttributeError, match=msg):
            invalid.dt
        assert not hasattr(invalid, "str")

    def test_set_categories_setitem(self):
        # GH#43334

        df = DataFrame({"Survived": [1, 0, 1], "Sex": [0, 1, 1]}, dtype="category")

        df["Survived"] = df["Survived"].cat.rename_categories(["No", "Yes"])
        df["Sex"] = df["Sex"].cat.rename_categories(["female", "male"])

        # values should not be coerced to NaN
        assert list(df["Sex"]) == ["female", "male", "male"]
        assert list(df["Survived"]) == ["Yes", "No", "Yes"]

        df["Sex"] = Categorical(df["Sex"], categories=["female", "male"], ordered=False)
        df["Survived"] = Categorical(
            df["Survived"], categories=["No", "Yes"], ordered=False
        )

        # values should not be coerced to NaN
        assert list(df["Sex"]) == ["female", "male", "male"]
        assert list(df["Survived"]) == ["Yes", "No", "Yes"]

    def test_categorical_of_booleans_is_boolean(self):
        # https://github.com/pandas-dev/pandas/issues/46313
        df = DataFrame(
            {"int_cat": [1, 2, 3], "bool_cat": [True, False, False]}, dtype="category"
        )
        value = df["bool_cat"].cat.categories.dtype
        expected = np.dtype(np.bool_)
        assert value is expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_at.py ===
from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas.errors import InvalidIndexError

from pandas import (
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm


def test_at_timezone():
    # https://github.com/pandas-dev/pandas/issues/33544
    result = DataFrame({"foo": [datetime(2000, 1, 1)]})
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        result.at[0, "foo"] = datetime(2000, 1, 2, tzinfo=timezone.utc)
    expected = DataFrame(
        {"foo": [datetime(2000, 1, 2, tzinfo=timezone.utc)]}, dtype=object
    )
    tm.assert_frame_equal(result, expected)


def test_selection_methods_of_assigned_col():
    # GH 29282
    df = DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = DataFrame(data={"c": [7, 8, 9]}, index=[2, 1, 0])
    df["c"] = df2["c"]
    df.at[1, "c"] = 11
    result = df
    expected = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [9, 11, 7]})
    tm.assert_frame_equal(result, expected)
    result = df.at[1, "c"]
    assert result == 11

    result = df["c"]
    expected = Series([9, 11, 7], name="c")
    tm.assert_series_equal(result, expected)

    result = df[["c"]]
    expected = DataFrame({"c": [9, 11, 7]})
    tm.assert_frame_equal(result, expected)


class TestAtSetItem:
    def test_at_setitem_item_cache_cleared(self):
        # GH#22372 Note the multi-step construction is necessary to trigger
        #  the original bug. pandas/issues/22372#issuecomment-413345309
        df = DataFrame(index=[0])
        df["x"] = 1
        df["cost"] = 2

        # accessing df["cost"] adds "cost" to the _item_cache
        df["cost"]

        # This loc[[0]] lookup used to call _consolidate_inplace at the
        #  BlockManager level, which failed to clear the _item_cache
        df.loc[[0]]

        df.at[0, "x"] = 4
        df.at[0, "cost"] = 789

        expected = DataFrame(
            {"x": [4], "cost": 789},
            index=[0],
            columns=Index(["x", "cost"], dtype=object),
        )
        tm.assert_frame_equal(df, expected)

        # And in particular, check that the _item_cache has updated correctly.
        tm.assert_series_equal(df["cost"], expected["cost"])

    def test_at_setitem_mixed_index_assignment(self):
        # GH#19860
        ser = Series([1, 2, 3, 4, 5], index=["a", "b", "c", 1, 2])
        ser.at["a"] = 11
        assert ser.iat[0] == 11
        ser.at[1] = 22
        assert ser.iat[3] == 22

    def test_at_setitem_categorical_missing(self):
        df = DataFrame(
            index=range(3), columns=range(3), dtype=CategoricalDtype(["foo", "bar"])
        )
        df.at[1, 1] = "foo"

        expected = DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, "foo", np.nan],
                [np.nan, np.nan, np.nan],
            ],
            dtype=CategoricalDtype(["foo", "bar"]),
        )

        tm.assert_frame_equal(df, expected)

    def test_at_setitem_multiindex(self):
        df = DataFrame(
            np.zeros((3, 2), dtype="int64"),
            columns=MultiIndex.from_tuples([("a", 0), ("a", 1)]),
        )
        df.at[0, "a"] = 10
        expected = DataFrame(
            [[10, 10], [0, 0], [0, 0]],
            columns=MultiIndex.from_tuples([("a", 0), ("a", 1)]),
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("row", (Timestamp("2019-01-01"), "2019-01-01"))
    def test_at_datetime_index(self, row):
        # Set float64 dtype to avoid upcast when setting .5
        df = DataFrame(
            data=[[1] * 2], index=DatetimeIndex(data=["2019-01-01", "2019-01-02"])
        ).astype({0: "float64"})
        expected = DataFrame(
            data=[[0.5, 1], [1.0, 1]],
            index=DatetimeIndex(data=["2019-01-01", "2019-01-02"]),
        )

        df.at[row, 0] = 0.5
        tm.assert_frame_equal(df, expected)


class TestAtSetItemWithExpansion:
    def test_at_setitem_expansion_series_dt64tz_value(self, tz_naive_fixture):
        # GH#25506
        ts = Timestamp("2017-08-05 00:00:00+0100", tz=tz_naive_fixture)
        result = Series(ts)
        result.at[1] = ts
        expected = Series([ts, ts])
        tm.assert_series_equal(result, expected)


class TestAtWithDuplicates:
    def test_at_with_duplicate_axes_requires_scalar_lookup(self):
        # GH#33041 check that falling back to loc doesn't allow non-scalar
        #  args to slip in

        arr = np.random.default_rng(2).standard_normal(6).reshape(3, 2)
        df = DataFrame(arr, columns=["A", "A"])

        msg = "Invalid call for scalar access"
        with pytest.raises(ValueError, match=msg):
            df.at[[1, 2]]
        with pytest.raises(ValueError, match=msg):
            df.at[1, ["A"]]
        with pytest.raises(ValueError, match=msg):
            df.at[:, "A"]

        with pytest.raises(ValueError, match=msg):
            df.at[[1, 2]] = 1
        with pytest.raises(ValueError, match=msg):
            df.at[1, ["A"]] = 1
        with pytest.raises(ValueError, match=msg):
            df.at[:, "A"] = 1


class TestAtErrors:
    # TODO: De-duplicate/parametrize
    #  test_at_series_raises_key_error2, test_at_frame_raises_key_error2

    def test_at_series_raises_key_error(self, indexer_al):
        # GH#31724 .at should match .loc

        ser = Series([1, 2, 3], index=[3, 2, 1])
        result = indexer_al(ser)[1]
        assert result == 3

        with pytest.raises(KeyError, match="a"):
            indexer_al(ser)["a"]

    def test_at_frame_raises_key_error(self, indexer_al):
        # GH#31724 .at should match .loc

        df = DataFrame({0: [1, 2, 3]}, index=[3, 2, 1])

        result = indexer_al(df)[1, 0]
        assert result == 3

        with pytest.raises(KeyError, match="a"):
            indexer_al(df)["a", 0]

        with pytest.raises(KeyError, match="a"):
            indexer_al(df)[1, "a"]

    def test_at_series_raises_key_error2(self, indexer_al):
        # at should not fallback
        # GH#7814
        # GH#31724 .at should match .loc
        ser = Series([1, 2, 3], index=list("abc"))
        result = indexer_al(ser)["a"]
        assert result == 1

        with pytest.raises(KeyError, match="^0$"):
            indexer_al(ser)[0]

    def test_at_frame_raises_key_error2(self, indexer_al):
        # GH#31724 .at should match .loc
        df = DataFrame({"A": [1, 2, 3]}, index=list("abc"))
        result = indexer_al(df)["a", "A"]
        assert result == 1

        with pytest.raises(KeyError, match="^0$"):
            indexer_al(df)["a", 0]

    def test_at_frame_multiple_columns(self):
        # GH#48296 - at shouldn't modify multiple columns
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        new_row = [6, 7]
        with pytest.raises(
            InvalidIndexError,
            match=f"You can only assign a scalar value not a \\{type(new_row)}",
        ):
            df.at[5] = new_row

    def test_at_getitem_mixed_index_no_fallback(self):
        # GH#19860
        ser = Series([1, 2, 3, 4, 5], index=["a", "b", "c", 1, 2])
        with pytest.raises(KeyError, match="^0$"):
            ser.at[0]
        with pytest.raises(KeyError, match="^4$"):
            ser.at[4]

    def test_at_categorical_integers(self):
        # CategoricalIndex with integer categories that don't happen to match
        #  the Categorical's codes
        ci = CategoricalIndex([3, 4])

        arr = np.arange(4).reshape(2, 2)
        frame = DataFrame(arr, index=ci)

        for df in [frame, frame.T]:
            for key in [0, 1]:
                with pytest.raises(KeyError, match=str(key)):
                    df.at[key, key]

    def test_at_applied_for_rows(self):
        # GH#48729 .at should raise InvalidIndexError when assigning rows
        df = DataFrame(index=["a"], columns=["col1", "col2"])
        new_row = [123, 15]
        with pytest.raises(
            InvalidIndexError,
            match=f"You can only assign a scalar value not a \\{type(new_row)}",
        ):
            df.at["a"] = new_row

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_fp_groups.py ===
from sympy.core.singleton import S
from sympy.combinatorics.fp_groups import (FpGroup, low_index_subgroups,
                                   reidemeister_presentation, FpSubgroup,
                                           simplify_presentation)
from sympy.combinatorics.free_groups import (free_group, FreeGroup)

from sympy.testing.pytest import slow

"""
References
==========

[1] Holt, D., Eick, B., O'Brien, E.
"Handbook of Computational Group Theory"

[2] John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490.
"Implementation and Analysis of the Todd-Coxeter Algorithm"

[3] PROC. SECOND  INTERNAT. CONF. THEORY OF GROUPS, CANBERRA 1973,
pp. 347-356. "A Reidemeister-Schreier program" by George Havas.
http://staff.itee.uq.edu.au/havas/1973cdhw.pdf

"""

def test_low_index_subgroups():
    F, x, y = free_group("x, y")

    # Example 5.10 from [1] Pg. 194
    f = FpGroup(F, [x**2, y**3, (x*y)**4])
    L = low_index_subgroups(f, 4)
    t1 = [[[0, 0, 0, 0]],
          [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 3, 3]],
          [[0, 0, 1, 2], [2, 2, 2, 0], [1, 1, 0, 1]],
          [[1, 1, 0, 0], [0, 0, 1, 1]]]
    for i in range(len(t1)):
        assert L[i].table == t1[i]

    f = FpGroup(F, [x**2, y**3, (x*y)**7])
    L = low_index_subgroups(f, 15)
    t2 = [[[0, 0, 0, 0]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [4, 4, 5, 3], [6, 6, 3, 4], [5, 5, 6, 6]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [6, 6, 5, 3], [5, 5, 3, 4], [4, 4, 6, 6]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [6, 6, 5, 3], [7, 7, 3, 4], [4, 4, 8, 9], [5, 5, 10, 11],
            [11, 11, 9, 6], [9, 9, 6, 8], [12, 12, 11, 7], [8, 8, 7, 10],
            [10, 10, 13, 14], [14, 14, 14, 12], [13, 13, 12, 13]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [6, 6, 5, 3], [7, 7, 3, 4], [4, 4, 8, 9], [5, 5, 10, 11],
            [11, 11, 9, 6], [12, 12, 6, 8], [10, 10, 11, 7], [8, 8, 7, 10],
            [9, 9, 13, 14], [14, 14, 14, 12], [13, 13, 12, 13]],
           [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5],
            [6, 6, 5, 3], [7, 7, 3, 4], [4, 4, 8, 9], [5, 5, 10, 11],
            [11, 11, 9, 6], [12, 12, 6, 8], [13, 13, 11, 7], [8, 8, 7, 10],
            [9, 9, 12, 12], [10, 10, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 3, 3], [2, 2, 5, 6]
            , [7, 7, 6, 4], [8, 8, 4, 5], [5, 5, 8, 9], [6, 6, 9, 7],
            [10, 10, 7, 8], [9, 9, 11, 12], [11, 11, 12, 10], [13, 13, 10, 11],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 3, 3], [2, 2, 5, 6]
            , [7, 7, 6, 4], [8, 8, 4, 5], [5, 5, 8, 9], [6, 6, 9, 7],
            [10, 10, 7, 8], [9, 9, 11, 12], [13, 13, 12, 10], [12, 12, 10, 11],
            [11, 11, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 4, 4]
            , [7, 7, 6, 3], [8, 8, 3, 5], [5, 5, 8, 9], [6, 6, 9, 7],
            [10, 10, 7, 8], [9, 9, 11, 12], [13, 13, 12, 10], [12, 12, 10, 11],
            [11, 11, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [5, 5, 6, 3], [9, 9, 3, 5], [10, 10, 8, 4], [8, 8, 4, 7],
            [6, 6, 10, 11], [7, 7, 11, 9], [12, 12, 9, 10], [11, 11, 13, 14],
            [14, 14, 14, 12], [13, 13, 12, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [6, 6, 6, 3], [5, 5, 3, 5], [8, 8, 8, 4], [7, 7, 4, 7]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [6, 6, 3, 5], [10, 10, 8, 4], [11, 11, 4, 7],
            [5, 5, 10, 12], [7, 7, 12, 9], [8, 8, 11, 11], [13, 13, 9, 10],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [6, 6, 3, 5], [10, 10, 8, 4], [11, 11, 4, 7],
            [5, 5, 12, 11], [7, 7, 10, 10], [8, 8, 9, 12], [13, 13, 11, 9],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [10, 10, 3, 5], [7, 7, 8, 4], [11, 11, 4, 7],
            [5, 5, 9, 9], [6, 6, 11, 12], [8, 8, 12, 10], [13, 13, 10, 11],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [10, 10, 3, 5], [7, 7, 8, 4], [11, 11, 4, 7],
            [5, 5, 12, 11], [6, 6, 10, 10], [8, 8, 9, 12], [13, 13, 11, 9],
            [12, 12, 13, 13]],
           [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8]
            , [9, 9, 6, 3], [10, 10, 3, 5], [11, 11, 8, 4], [12, 12, 4, 7],
            [5, 5, 9, 9], [6, 6, 12, 13], [7, 7, 11, 11], [8, 8, 13, 10],
            [13, 13, 10, 12]],
           [[1, 1, 0, 0], [0, 0, 2, 3], [4, 4, 3, 1], [5, 5, 1, 2], [2, 2, 4, 4]
            , [3, 3, 6, 7], [7, 7, 7, 5], [6, 6, 5, 6]]]
    for i  in range(len(t2)):
        assert L[i].table == t2[i]

    f = FpGroup(F, [x**2, y**3, (x*y)**7])
    L = low_index_subgroups(f, 10, [x])
    t3 = [[[0, 0, 0, 0]],
          [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5], [4, 4, 5, 3],
           [6, 6, 3, 4], [5, 5, 6, 6]],
          [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 4, 5], [6, 6, 5, 3],
           [5, 5, 3, 4], [4, 4, 6, 6]],
          [[0, 0, 1, 2], [3, 3, 2, 0], [4, 4, 0, 1], [1, 1, 5, 6], [2, 2, 7, 8],
           [6, 6, 6, 3], [5, 5, 3, 5], [8, 8, 8, 4], [7, 7, 4, 7]]]
    for i in range(len(t3)):
        assert L[i].table == t3[i]


def test_subgroup_presentations():
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x**3, y**5, (x*y)**2])
    H = [x*y, x**-1*y**-1*x*y*x]
    p1 = reidemeister_presentation(f, H)
    assert str(p1) == "((y_1, y_2), (y_1**2, y_2**3, y_2*y_1*y_2*y_1*y_2*y_1))"

    H = f.subgroup(H)
    assert (H.generators, H.relators) == p1

    f = FpGroup(F, [x**3, y**3, (x*y)**3])
    H = [x*y, x*y**-1]
    p2 = reidemeister_presentation(f, H)
    assert str(p2) == "((x_0, y_0), (x_0**3, y_0**3, x_0*y_0*x_0*y_0*x_0*y_0))"

    f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    H = [x]
    p3 = reidemeister_presentation(f, H)
    assert str(p3) == "((x_0,), (x_0**4,))"

    f = FpGroup(F, [x**3*y**-3, (x*y)**3, (x*y**-1)**2])
    H = [x]
    p4 = reidemeister_presentation(f, H)
    assert str(p4) == "((x_0,), (x_0**6,))"

    # this presentation can be improved, the most simplified form
    # of presentation is <a, b | a^11, b^2, (a*b)^3, (a^4*b*a^-5*b)^2>
    # See [2] Pg 474 group PSL_2(11)
    # This is the group PSL_2(11)
    F, a, b, c = free_group("a, b, c")
    f = FpGroup(F, [a**11, b**5, c**4, (b*c**2)**2, (a*b*c)**3, (a**4*c**2)**3, b**2*c**-1*b**-1*c, a**4*b**-1*a**-1*b])
    H = [a, b, c**2]
    gens, rels = reidemeister_presentation(f, H)
    assert str(gens) == "(b_1, c_3)"
    assert len(rels) == 18


@slow
def test_order():
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    assert f.order() == 8

    f = FpGroup(F, [x*y*x**-1*y**-1, y**2])
    assert f.order() is S.Infinity

    F, a, b, c = free_group("a, b, c")
    f = FpGroup(F, [a**250, b**2, c*b*c**-1*b, c**4, c**-1*a**-1*c*a, a**-1*b**-1*a*b])
    assert f.order() == 2000

    F, x = free_group("x")
    f = FpGroup(F, [])
    assert f.order() is S.Infinity

    f = FpGroup(free_group('')[0], [])
    assert f.order() == 1

def test_fp_subgroup():
    def _test_subgroup(K, T, S):
        _gens = T(K.generators)
        assert all(elem in S for elem in _gens)
        assert T.is_injective()
        assert T.image().order() == S.order()
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    S = FpSubgroup(f, [x*y])
    assert (x*y)**-3 in S
    K, T = f.subgroup([x*y], homomorphism=True)
    assert T(K.generators) == [y*x**-1]
    _test_subgroup(K, T, S)

    S = FpSubgroup(f, [x**-1*y*x])
    assert x**-1*y**4*x in S
    assert x**-1*y**4*x**2 not in S
    K, T = f.subgroup([x**-1*y*x], homomorphism=True)
    assert T(K.generators[0]**3) == y**3
    _test_subgroup(K, T, S)

    f = FpGroup(F, [x**3, y**5, (x*y)**2])
    H = [x*y, x**-1*y**-1*x*y*x]
    K, T = f.subgroup(H, homomorphism=True)
    S = FpSubgroup(f, H)
    _test_subgroup(K, T, S)

def test_permutation_methods():
    F, x, y = free_group("x, y")
    # DihedralGroup(8)
    G = FpGroup(F, [x**2, y**8, x*y*x**-1*y])
    T = G._to_perm_group()[1]
    assert T.is_isomorphism()
    assert G.center() == [y**4]

    # DiheadralGroup(4)
    G = FpGroup(F, [x**2, y**4, x*y*x**-1*y])
    S = FpSubgroup(G, G.normal_closure([x]))
    assert x in S
    assert y**-1*x*y in S

    # Z_5xZ_4
    G = FpGroup(F, [x*y*x**-1*y**-1, y**5, x**4])
    assert G.is_abelian
    assert G.is_solvable

    # AlternatingGroup(5)
    G = FpGroup(F, [x**3, y**2, (x*y)**5])
    assert not G.is_solvable

    # AlternatingGroup(4)
    G = FpGroup(F, [x**3, y**2, (x*y)**3])
    assert len(G.derived_series()) == 3
    S = FpSubgroup(G, G.derived_subgroup())
    assert S.order() == 4


def test_simplify_presentation():
    # ref #16083
    G = simplify_presentation(FpGroup(FreeGroup([]), []))
    assert not G.generators
    assert not G.relators

    # CyclicGroup(3)
    # The second generator in <x, y | x^2, x^5, y^3> is trivial due to relators {x^2, x^5}
    F, x, y = free_group("x, y")
    G = simplify_presentation(FpGroup(F, [x**2, x**5, y**3]))
    assert x in G.relators

def test_cyclic():
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x*y, x**-1*y**-1*x*y*x])
    assert f.is_cyclic
    f = FpGroup(F, [x*y, x*y**-1])
    assert f.is_cyclic
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    assert not f.is_cyclic


def test_abelian_invariants():
    F, x, y = free_group("x, y")
    f = FpGroup(F, [x*y, x**-1*y**-1*x*y*x])
    assert f.abelian_invariants() == []
    f = FpGroup(F, [x*y, x*y**-1])
    assert f.abelian_invariants() == [2]
    f = FpGroup(F, [x**4, y**2, x*y*x**-1*y])
    assert f.abelian_invariants() == [2, 4]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_errors.py ===
import datetime
from io import BytesIO
import re

import numpy as np
import pytest

from pandas import (
    CategoricalIndex,
    DataFrame,
    HDFStore,
    Index,
    MultiIndex,
    _testing as tm,
    date_range,
    read_hdf,
)
from pandas.tests.io.pytables.common import ensure_clean_store

from pandas.io.pytables import (
    Term,
    _maybe_adjust_name,
)

pytestmark = [pytest.mark.single_cpu]


def test_pass_spec_to_storer(setup_path):
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    with ensure_clean_store(setup_path) as store:
        store.put("df", df)
        msg = (
            "cannot pass a column specification when reading a Fixed format "
            "store. this store must be selected in its entirety"
        )
        with pytest.raises(TypeError, match=msg):
            store.select("df", columns=["A"])
        msg = (
            "cannot pass a where specification when reading from a Fixed "
            "format store. this store must be selected in its entirety"
        )
        with pytest.raises(TypeError, match=msg):
            store.select("df", where=[("columns=A")])


def test_table_index_incompatible_dtypes(setup_path):
    df1 = DataFrame({"a": [1, 2, 3]})
    df2 = DataFrame({"a": [4, 5, 6]}, index=date_range("1/1/2000", periods=3))

    with ensure_clean_store(setup_path) as store:
        store.put("frame", df1, format="table")
        msg = re.escape("incompatible kind in col [integer - datetime64[ns]]")
        with pytest.raises(TypeError, match=msg):
            store.put("frame", df2, format="table", append=True)


def test_unimplemented_dtypes_table_columns(setup_path):
    with ensure_clean_store(setup_path) as store:
        dtypes = [("date", datetime.date(2001, 1, 2))]

        # currently not supported dtypes ####
        for n, f in dtypes:
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )
            df[n] = f
            msg = re.escape(f"[{n}] is not implemented as a table column")
            with pytest.raises(TypeError, match=msg):
                store.append(f"df1_{n}", df)

    # frame
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    df["obj1"] = "foo"
    df["obj2"] = "bar"
    df["datetime1"] = datetime.date(2001, 1, 2)
    df = df._consolidate()

    with ensure_clean_store(setup_path) as store:
        # this fails because we have a date in the object block......
        msg = "|".join(
            [
                re.escape(
                    "Cannot serialize the column [datetime1]\nbecause its data "
                    "contents are not [string] but [date] object dtype"
                ),
                re.escape("[date] is not implemented as a table column"),
            ]
        )
        with pytest.raises(TypeError, match=msg):
            store.append("df_unimplemented", df)


def test_invalid_terms(tmp_path, setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df["string"] = "foo"
        df.loc[df.index[0:4], "string"] = "bar"

        store.put("df", df, format="table")

        # some invalid terms
        msg = re.escape("__init__() missing 1 required positional argument: 'where'")
        with pytest.raises(TypeError, match=msg):
            Term()

        # more invalid
        msg = re.escape(
            "cannot process expression [df.index[3]], "
            "[2000-01-06 00:00:00] is not a valid condition"
        )
        with pytest.raises(ValueError, match=msg):
            store.select("df", "df.index[3]")

        msg = "invalid syntax"
        with pytest.raises(SyntaxError, match=msg):
            store.select("df", "index>")

    # from the docs
    path = tmp_path / setup_path
    dfq = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=list("ABCD"),
        index=date_range("20130101", periods=10),
    )
    dfq.to_hdf(path, key="dfq", format="table", data_columns=True)

    # check ok
    read_hdf(path, "dfq", where="index>Timestamp('20130104') & columns=['A', 'B']")
    read_hdf(path, "dfq", where="A>0 or C>0")

    # catch the invalid reference
    path = tmp_path / setup_path
    dfq = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=list("ABCD"),
        index=date_range("20130101", periods=10),
    )
    dfq.to_hdf(path, key="dfq", format="table")

    msg = (
        r"The passed where expression: A>0 or C>0\n\s*"
        r"contains an invalid variable reference\n\s*"
        r"all of the variable references must be a reference to\n\s*"
        r"an axis \(e.g. 'index' or 'columns'\), or a data_column\n\s*"
        r"The currently defined references are: index,columns\n"
    )
    with pytest.raises(ValueError, match=msg):
        read_hdf(path, "dfq", where="A>0 or C>0")


def test_append_with_diff_col_name_types_raises_value_error(setup_path):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)))
    df2 = DataFrame({"a": np.random.default_rng(2).standard_normal(10)})
    df3 = DataFrame({(1, 2): np.random.default_rng(2).standard_normal(10)})
    df4 = DataFrame({("1", 2): np.random.default_rng(2).standard_normal(10)})
    df5 = DataFrame({("1", 2, object): np.random.default_rng(2).standard_normal(10)})

    with ensure_clean_store(setup_path) as store:
        name = "df_diff_valerror"
        store.append(name, df)

        for d in (df2, df3, df4, df5):
            msg = re.escape(
                "cannot match existing table structure for [0] on appending data"
            )
            with pytest.raises(ValueError, match=msg):
                store.append(name, d)


def test_invalid_complib(setup_path):
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    with tm.ensure_clean(setup_path) as path:
        msg = r"complib only supports \[.*\] compression."
        with pytest.raises(ValueError, match=msg):
            df.to_hdf(path, key="df", complib="foolib")


@pytest.mark.parametrize(
    "idx",
    [
        date_range("2019", freq="D", periods=3, tz="UTC"),
        CategoricalIndex(list("abc")),
    ],
)
def test_to_hdf_multiindex_extension_dtype(idx, tmp_path, setup_path):
    # GH 7775
    mi = MultiIndex.from_arrays([idx, idx])
    df = DataFrame(0, index=mi, columns=["a"])
    path = tmp_path / setup_path
    with pytest.raises(NotImplementedError, match="Saving a MultiIndex"):
        df.to_hdf(path, key="df")


def test_unsuppored_hdf_file_error(datapath):
    # GH 9539
    data_path = datapath("io", "data", "legacy_hdf/incompatible_dataset.h5")
    message = (
        r"Dataset\(s\) incompatible with Pandas data types, "
        "not table, or no datasets found in HDF5 file."
    )

    with pytest.raises(ValueError, match=message):
        read_hdf(data_path)


def test_read_hdf_errors(setup_path, tmp_path):
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    path = tmp_path / setup_path
    msg = r"File [\S]* does not exist"
    with pytest.raises(OSError, match=msg):
        read_hdf(path, "key")

    df.to_hdf(path, key="df")
    store = HDFStore(path, mode="r")
    store.close()

    msg = "The HDFStore must be open for reading."
    with pytest.raises(OSError, match=msg):
        read_hdf(store, "df")


def test_read_hdf_generic_buffer_errors():
    msg = "Support for generic buffers has not been implemented."
    with pytest.raises(NotImplementedError, match=msg):
        read_hdf(BytesIO(b""), "df")


@pytest.mark.parametrize("bad_version", [(1, 2), (1,), [], "12", "123"])
def test_maybe_adjust_name_bad_version_raises(bad_version):
    msg = "Version is incorrect, expected sequence of 3 integers"
    with pytest.raises(ValueError, match=msg):
        _maybe_adjust_name("values_block_0", version=bad_version)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\__init__.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.
from .initialise import init, deinit, reinit, colorama_text, just_fix_windows_console
from .ansi import Fore, Back, Style, Cursor
from .ansitowin32 import AnsiToWin32

__version__ = '0.4.6'


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\internal\__init__.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\_ufunc.py ===
from numpy import ufunc

_UFunc_Nin1_Nout1 = ufunc
_UFunc_Nin2_Nout1 = ufunc
_UFunc_Nin1_Nout2 = ufunc
_UFunc_Nin2_Nout2 = ufunc
_GUFunc_Nin2_Nout1 = ufunc

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\capi\_ld_preload.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This file can be modified by setup.py when building a manylinux2010 wheel
# When modified, it will preload some libraries needed for the python C extension

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\ArgType.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

class ArgType(object):
    INPUT = 0
    OUTPUT = 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\NodeType.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

class NodeType(object):
    Primitive = 0
    Fused = 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\outcome\_version.py ===
# This file is imported from __init__.py and exec'd from setup.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Final

__version__: 'Final[str]' = "1.3.0.post0"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\arrow\__init__.py ===
from pandas.core.arrays.arrow.accessors import (
    ListAccessor,
    StructAccessor,
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray

__all__ = ["ArrowExtensionArray", "StructAccessor", "ListAccessor"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_groupby_shift_diff.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


def test_group_shift_with_null_key():
    # This test is designed to replicate the segfault in issue #13813.
    n_rows = 1200

    # Generate a moderately large dataframe with occasional missing
    # values in column `B`, and then group by [`A`, `B`]. This should
    # force `-1` in `labels` array of `g._grouper.group_info` exactly
    # at those places, where the group-by key is partially missing.
    df = DataFrame(
        [(i % 12, i % 3 if i % 3 else np.nan, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    expected = DataFrame(
        [(i + 12 if i % 3 and i < n_rows - 12 else np.nan) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    result = g.shift(-1)

    tm.assert_frame_equal(result, expected)


def test_group_shift_with_fill_value():
    # GH #24128
    n_rows = 24
    df = DataFrame(
        [(i % 12, i % 3, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    expected = DataFrame(
        [(i + 12 if i < n_rows - 12 else 0) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    result = g.shift(-1, fill_value=0)

    tm.assert_frame_equal(result, expected)


def test_group_shift_lose_timezone():
    # GH 30134
    now_dt = Timestamp.utcnow().as_unit("ns")
    df = DataFrame({"a": [1, 1], "date": now_dt})
    result = df.groupby("a").shift(0).iloc[0]
    expected = Series({"date": now_dt}, name=result.name)
    tm.assert_series_equal(result, expected)


def test_group_diff_real_series(any_real_numpy_dtype):
    df = DataFrame(
        {"a": [1, 2, 3, 3, 2], "b": [1, 2, 3, 4, 5]},
        dtype=any_real_numpy_dtype,
    )
    result = df.groupby("a")["b"].diff()
    exp_dtype = "float"
    if any_real_numpy_dtype in ["int8", "int16", "float32"]:
        exp_dtype = "float32"
    expected = Series([np.nan, np.nan, np.nan, 1.0, 3.0], dtype=exp_dtype, name="b")
    tm.assert_series_equal(result, expected)


def test_group_diff_real_frame(any_real_numpy_dtype):
    df = DataFrame(
        {
            "a": [1, 2, 3, 3, 2],
            "b": [1, 2, 3, 4, 5],
            "c": [1, 2, 3, 4, 6],
        },
        dtype=any_real_numpy_dtype,
    )
    result = df.groupby("a").diff()
    exp_dtype = "float"
    if any_real_numpy_dtype in ["int8", "int16", "float32"]:
        exp_dtype = "float32"
    expected = DataFrame(
        {
            "b": [np.nan, np.nan, np.nan, 1.0, 3.0],
            "c": [np.nan, np.nan, np.nan, 1.0, 4.0],
        },
        dtype=exp_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [
            Timestamp("2013-01-01"),
            Timestamp("2013-01-02"),
            Timestamp("2013-01-03"),
        ],
        [Timedelta("5 days"), Timedelta("6 days"), Timedelta("7 days")],
    ],
)
def test_group_diff_datetimelike(data, unit):
    df = DataFrame({"a": [1, 2, 2], "b": data})
    df["b"] = df["b"].dt.as_unit(unit)
    result = df.groupby("a")["b"].diff()
    expected = Series([NaT, NaT, Timedelta("1 days")], name="b").dt.as_unit(unit)
    tm.assert_series_equal(result, expected)


def test_group_diff_bool():
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [True, True, False, False, True]})
    result = df.groupby("a")["b"].diff()
    expected = Series([np.nan, np.nan, np.nan, False, False], name="b")
    tm.assert_series_equal(result, expected)


def test_group_diff_object_raises(object_dtype):
    df = DataFrame(
        {"a": ["foo", "bar", "bar"], "b": ["baz", "foo", "foo"]}, dtype=object_dtype
    )
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for -"):
        df.groupby("a")["b"].diff()


def test_empty_shift_with_fill():
    # GH 41264, single-index check
    df = DataFrame(columns=["a", "b", "c"])
    shifted = df.groupby(["a"]).shift(1)
    shifted_with_fill = df.groupby(["a"]).shift(1, fill_value=0)
    tm.assert_frame_equal(shifted, shifted_with_fill)
    tm.assert_index_equal(shifted.index, shifted_with_fill.index)


def test_multindex_empty_shift_with_fill():
    # GH 41264, multi-index check
    df = DataFrame(columns=["a", "b", "c"])
    shifted = df.groupby(["a", "b"]).shift(1)
    shifted_with_fill = df.groupby(["a", "b"]).shift(1, fill_value=0)
    tm.assert_frame_equal(shifted, shifted_with_fill)
    tm.assert_index_equal(shifted.index, shifted_with_fill.index)


def test_shift_periods_freq():
    # GH 54093
    data = {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data, index=date_range(start="20100101", periods=6))
    result = df.groupby(df.index).shift(periods=-2, freq="D")
    expected = DataFrame(data, index=date_range(start="2009-12-30", periods=6))
    tm.assert_frame_equal(result, expected)


def test_shift_deprecate_freq_and_fill_value():
    # GH 53832
    data = {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data, index=date_range(start="20100101", periods=6))
    msg = (
        "Passing a 'freq' together with a 'fill_value' silently ignores the fill_value"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby(df.index).shift(periods=-2, freq="D", fill_value="1")


def test_shift_disallow_suffix_if_periods_is_int():
    # GH#44424
    data = {"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]}
    df = DataFrame(data)
    msg = "Cannot specify `suffix` if `periods` is an int."
    with pytest.raises(ValueError, match=msg):
        df.groupby("b").shift(1, suffix="fails")


def test_group_shift_with_multiple_periods():
    # GH#44424
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [True, True, False, False, True]})

    shifted_df = df.groupby("b")[["a"]].shift([0, 1])
    expected_df = DataFrame(
        {"a_0": [1, 2, 3, 3, 2], "a_1": [np.nan, 1.0, np.nan, 3.0, 2.0]}
    )
    tm.assert_frame_equal(shifted_df, expected_df)

    # series
    shifted_series = df.groupby("b")["a"].shift([0, 1])
    tm.assert_frame_equal(shifted_series, expected_df)


def test_group_shift_with_multiple_periods_and_freq():
    # GH#44424
    df = DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [True, True, False, False, True]},
        index=date_range("1/1/2000", periods=5, freq="h"),
    )
    shifted_df = df.groupby("b")[["a"]].shift(
        [0, 1],
        freq="h",
    )
    expected_df = DataFrame(
        {
            "a_0": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "a_1": [
                np.nan,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
        },
        index=date_range("1/1/2000", periods=6, freq="h"),
    )
    tm.assert_frame_equal(shifted_df, expected_df)


def test_group_shift_with_multiple_periods_and_fill_value():
    # GH#44424
    df = DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [True, True, False, False, True]},
    )
    shifted_df = df.groupby("b")[["a"]].shift([0, 1], fill_value=-1)
    expected_df = DataFrame(
        {"a_0": [1, 2, 3, 4, 5], "a_1": [-1, 1, -1, 3, 2]},
    )
    tm.assert_frame_equal(shifted_df, expected_df)


def test_group_shift_with_multiple_periods_and_both_fill_and_freq_deprecated():
    # GH#44424
    df = DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [True, True, False, False, True]},
        index=date_range("1/1/2000", periods=5, freq="h"),
    )
    msg = (
        "Passing a 'freq' together with a 'fill_value' silently ignores the "
        "fill_value"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby("b")[["a"]].shift([1, 2], fill_value=1, freq="h")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\modes\__init__.py ===
from . import emacs, notemacs, vi

__all__ = ["emacs", "notemacs", "vi"]

editingmodes = [emacs.EmacsMode, notemacs.NotEmacsMode, vi.ViMode]

# add check to ensure all modes have unique mode names

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\constants.py ===
import os
import sys

ROOT_FOLDER_NAME = ".wdm"
DEFAULT_PROJECT_ROOT_CACHE_PATH = os.path.join(sys.path[0], ROOT_FOLDER_NAME)
DEFAULT_USER_HOME_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ROOT_FOLDER_NAME)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_logical.py ===
import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
    kleene_and,
    kleene_or,
    kleene_xor,
)
from pandas.tests.extension.base import BaseOpsUtil


class TestLogicalOps(BaseOpsUtil):
    def test_numpy_scalars_ok(self, all_logical_operators):
        a = pd.array([True, False, None], dtype="boolean")
        op = getattr(a, all_logical_operators)

        tm.assert_extension_array_equal(op(True), op(np.bool_(True)))
        tm.assert_extension_array_equal(op(False), op(np.bool_(False)))

    def get_op_from_name(self, op_name):
        short_opname = op_name.strip("_")
        short_opname = short_opname if "xor" in short_opname else short_opname + "_"
        try:
            op = getattr(operator, short_opname)
        except AttributeError:
            # Assume it is the reverse operator
            rop = getattr(operator, short_opname[1:])
            op = lambda x, y: rop(y, x)

        return op

    def test_empty_ok(self, all_logical_operators):
        a = pd.array([], dtype="boolean")
        op_name = all_logical_operators
        result = getattr(a, op_name)(True)
        tm.assert_extension_array_equal(a, result)

        result = getattr(a, op_name)(False)
        tm.assert_extension_array_equal(a, result)

        result = getattr(a, op_name)(pd.NA)
        tm.assert_extension_array_equal(a, result)

    @pytest.mark.parametrize(
        "other", ["a", pd.Timestamp(2017, 1, 1, 12), np.timedelta64(4)]
    )
    def test_eq_mismatched_type(self, other):
        # GH-44499
        arr = pd.array([True, False])
        result = arr == other
        expected = pd.array([False, False])
        tm.assert_extension_array_equal(result, expected)

        result = arr != other
        expected = pd.array([True, True])
        tm.assert_extension_array_equal(result, expected)

    def test_logical_length_mismatch_raises(self, all_logical_operators):
        op_name = all_logical_operators
        a = pd.array([True, False, None], dtype="boolean")
        msg = "Lengths must match"

        with pytest.raises(ValueError, match=msg):
            getattr(a, op_name)([True, False])

        with pytest.raises(ValueError, match=msg):
            getattr(a, op_name)(np.array([True, False]))

        with pytest.raises(ValueError, match=msg):
            getattr(a, op_name)(pd.array([True, False], dtype="boolean"))

    def test_logical_nan_raises(self, all_logical_operators):
        op_name = all_logical_operators
        a = pd.array([True, False, None], dtype="boolean")
        msg = "Got float instead"

        with pytest.raises(TypeError, match=msg):
            getattr(a, op_name)(np.nan)

    @pytest.mark.parametrize("other", ["a", 1])
    def test_non_bool_or_na_other_raises(self, other, all_logical_operators):
        a = pd.array([True, False], dtype="boolean")
        with pytest.raises(TypeError, match=str(type(other).__name__)):
            getattr(a, all_logical_operators)(other)

    def test_kleene_or(self):
        # A clear test of behavior.
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")
        result = a | b
        expected = pd.array(
            [True, True, True, True, False, None, True, None, None], dtype="boolean"
        )
        tm.assert_extension_array_equal(result, expected)

        result = b | a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (pd.NA, [True, None, None]),
            (True, [True, True, True]),
            (np.bool_(True), [True, True, True]),
            (False, [True, False, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_or_scalar(self, other, expected):
        # TODO: test True & False
        a = pd.array([True, False, None], dtype="boolean")
        result = a | other
        expected = pd.array(expected, dtype="boolean")
        tm.assert_extension_array_equal(result, expected)

        result = other | a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )

    def test_kleene_and(self):
        # A clear test of behavior.
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")
        result = a & b
        expected = pd.array(
            [True, False, None, False, False, False, None, False, None], dtype="boolean"
        )
        tm.assert_extension_array_equal(result, expected)

        result = b & a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (pd.NA, [None, False, None]),
            (True, [True, False, None]),
            (False, [False, False, False]),
            (np.bool_(True), [True, False, None]),
            (np.bool_(False), [False, False, False]),
        ],
    )
    def test_kleene_and_scalar(self, other, expected):
        a = pd.array([True, False, None], dtype="boolean")
        result = a & other
        expected = pd.array(expected, dtype="boolean")
        tm.assert_extension_array_equal(result, expected)

        result = other & a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )

    def test_kleene_xor(self):
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")
        result = a ^ b
        expected = pd.array(
            [False, True, None, True, False, None, None, None, None], dtype="boolean"
        )
        tm.assert_extension_array_equal(result, expected)

        result = b ^ a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (pd.NA, [None, None, None]),
            (True, [False, True, None]),
            (np.bool_(True), [False, True, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_xor_scalar(self, other, expected):
        a = pd.array([True, False, None], dtype="boolean")
        result = a ^ other
        expected = pd.array(expected, dtype="boolean")
        tm.assert_extension_array_equal(result, expected)

        result = other ^ a
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )

    @pytest.mark.parametrize("other", [True, False, pd.NA, [True, False, None] * 3])
    def test_no_masked_assumptions(self, other, all_logical_operators):
        # The logical operations should not assume that masked values are False!
        a = pd.arrays.BooleanArray(
            np.array([True, True, True, False, False, False, True, False, True]),
            np.array([False] * 6 + [True, True, True]),
        )
        b = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        if isinstance(other, list):
            other = pd.array(other, dtype="boolean")

        result = getattr(a, all_logical_operators)(other)
        expected = getattr(b, all_logical_operators)(other)
        tm.assert_extension_array_equal(result, expected)

        if isinstance(other, BooleanArray):
            other._data[other._mask] = True
            a._data[a._mask] = False

            result = getattr(a, all_logical_operators)(other)
            expected = getattr(b, all_logical_operators)(other)
            tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("operation", [kleene_or, kleene_xor, kleene_and])
def test_error_both_scalar(operation):
    msg = r"Either `left` or `right` need to be a np\.ndarray."
    with pytest.raises(TypeError, match=msg):
        # masks need to be non-None, otherwise it ends up in an infinite recursion
        operation(True, True, np.zeros(1), np.zeros(1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_astype.py ===
import re

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)

from pandas import (
    CategoricalIndex,
    Index,
    IntervalIndex,
    NaT,
    Timedelta,
    Timestamp,
    interval_range,
)
import pandas._testing as tm


class AstypeTests:
    """Tests common to IntervalIndex with any subtype"""

    def test_astype_idempotent(self, index):
        result = index.astype("interval")
        tm.assert_index_equal(result, index)

        result = index.astype(index.dtype)
        tm.assert_index_equal(result, index)

    def test_astype_object(self, index):
        result = index.astype(object)
        expected = Index(index.values, dtype="object")
        tm.assert_index_equal(result, expected)
        assert not result.equals(index)

    def test_astype_category(self, index):
        result = index.astype("category")
        expected = CategoricalIndex(index.values)
        tm.assert_index_equal(result, expected)

        result = index.astype(CategoricalDtype())
        tm.assert_index_equal(result, expected)

        # non-default params
        categories = index.dropna().unique().values[:-1]
        dtype = CategoricalDtype(categories=categories, ordered=True)
        result = index.astype(dtype)
        expected = CategoricalIndex(index.values, categories=categories, ordered=True)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            "int64",
            "uint64",
            "float64",
            "complex128",
            "period[M]",
            "timedelta64",
            "timedelta64[ns]",
            "datetime64",
            "datetime64[ns]",
            "datetime64[ns, US/Eastern]",
        ],
    )
    def test_astype_cannot_cast(self, index, dtype):
        msg = "Cannot cast IntervalIndex to dtype"
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    def test_astype_invalid_dtype(self, index):
        msg = "data type [\"']fake_dtype[\"'] not understood"
        with pytest.raises(TypeError, match=msg):
            index.astype("fake_dtype")


class TestIntSubtype(AstypeTests):
    """Tests specific to IntervalIndex with integer-like subtype"""

    indexes = [
        IntervalIndex.from_breaks(np.arange(-10, 11, dtype="int64")),
        IntervalIndex.from_breaks(np.arange(100, dtype="uint64"), closed="left"),
    ]

    @pytest.fixture(params=indexes)
    def index(self, request):
        return request.param

    @pytest.mark.parametrize(
        "subtype", ["float64", "datetime64[ns]", "timedelta64[ns]"]
    )
    def test_subtype_conversion(self, index, subtype):
        dtype = IntervalDtype(subtype, index.closed)
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype), index.right.astype(subtype), closed=index.closed
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "subtype_start, subtype_end", [("int64", "uint64"), ("uint64", "int64")]
    )
    def test_subtype_integer(self, subtype_start, subtype_end):
        index = IntervalIndex.from_breaks(np.arange(100, dtype=subtype_start))
        dtype = IntervalDtype(subtype_end, index.closed)
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype_end),
            index.right.astype(subtype_end),
            closed=index.closed,
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.xfail(reason="GH#15832")
    def test_subtype_integer_errors(self):
        # int64 -> uint64 fails with negative values
        index = interval_range(-10, 10)
        dtype = IntervalDtype("uint64", "right")

        # Until we decide what the exception message _should_ be, we
        #  assert something that it should _not_ be.
        #  We should _not_ be getting a message suggesting that the -10
        #  has been wrapped around to a large-positive integer
        msg = "^(?!(left side of interval must be <= right side))"
        with pytest.raises(ValueError, match=msg):
            index.astype(dtype)


class TestFloatSubtype(AstypeTests):
    """Tests specific to IntervalIndex with float subtype"""

    indexes = [
        interval_range(-10.0, 10.0, closed="neither"),
        IntervalIndex.from_arrays(
            [-1.5, np.nan, 0.0, 0.0, 1.5], [-0.5, np.nan, 1.0, 1.0, 3.0], closed="both"
        ),
    ]

    @pytest.fixture(params=indexes)
    def index(self, request):
        return request.param

    @pytest.mark.parametrize("subtype", ["int64", "uint64"])
    def test_subtype_integer(self, subtype):
        index = interval_range(0.0, 10.0)
        dtype = IntervalDtype(subtype, "right")
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype), index.right.astype(subtype), closed=index.closed
        )
        tm.assert_index_equal(result, expected)

        # raises with NA
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(ValueError, match=msg):
            index.insert(0, np.nan).astype(dtype)

    @pytest.mark.parametrize("subtype", ["int64", "uint64"])
    def test_subtype_integer_with_non_integer_borders(self, subtype):
        index = interval_range(0.0, 3.0, freq=0.25)
        dtype = IntervalDtype(subtype, "right")
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype), index.right.astype(subtype), closed=index.closed
        )
        tm.assert_index_equal(result, expected)

    def test_subtype_integer_errors(self):
        # float64 -> uint64 fails with negative values
        index = interval_range(-10.0, 10.0)
        dtype = IntervalDtype("uint64", "right")
        msg = re.escape(
            "Cannot convert interval[float64, right] to interval[uint64, right]; "
            "subtypes are incompatible"
        )
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    @pytest.mark.parametrize("subtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_subtype_datetimelike(self, index, subtype):
        dtype = IntervalDtype(subtype, "right")
        msg = "Cannot convert .* to .*; subtypes are incompatible"
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_astype_category(self, index):
        super().test_astype_category(index)


class TestDatetimelikeSubtype(AstypeTests):
    """Tests specific to IntervalIndex with datetime-like subtype"""

    indexes = [
        interval_range(Timestamp("2018-01-01"), periods=10, closed="neither"),
        interval_range(Timestamp("2018-01-01"), periods=10).insert(2, NaT),
        interval_range(Timestamp("2018-01-01", tz="US/Eastern"), periods=10),
        interval_range(Timedelta("0 days"), periods=10, closed="both"),
        interval_range(Timedelta("0 days"), periods=10).insert(2, NaT),
    ]

    @pytest.fixture(params=indexes)
    def index(self, request):
        return request.param

    @pytest.mark.parametrize("subtype", ["int64", "uint64"])
    def test_subtype_integer(self, index, subtype):
        dtype = IntervalDtype(subtype, "right")

        if subtype != "int64":
            msg = (
                r"Cannot convert interval\[(timedelta64|datetime64)\[ns.*\], .*\] "
                r"to interval\[uint64, .*\]"
            )
            with pytest.raises(TypeError, match=msg):
                index.astype(dtype)
            return

        result = index.astype(dtype)
        new_left = index.left.astype(subtype)
        new_right = index.right.astype(subtype)

        expected = IntervalIndex.from_arrays(new_left, new_right, closed=index.closed)
        tm.assert_index_equal(result, expected)

    def test_subtype_float(self, index):
        dtype = IntervalDtype("float64", "right")
        msg = "Cannot convert .* to .*; subtypes are incompatible"
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    def test_subtype_datetimelike(self):
        # datetime -> timedelta raises
        dtype = IntervalDtype("timedelta64[ns]", "right")
        msg = "Cannot convert .* to .*; subtypes are incompatible"

        index = interval_range(Timestamp("2018-01-01"), periods=10)
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

        index = interval_range(Timestamp("2018-01-01", tz="CET"), periods=10)
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

        # timedelta -> datetime raises
        dtype = IntervalDtype("datetime64[ns]", "right")
        index = interval_range(Timedelta("0 days"), periods=10)
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_setops.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import Hour


class TestTimedeltaIndex:
    def test_union(self):
        i1 = timedelta_range("1day", periods=5)
        i2 = timedelta_range("3day", periods=5)
        result = i1.union(i2)
        expected = timedelta_range("1day", periods=7)
        tm.assert_index_equal(result, expected)

        i1 = Index(np.arange(0, 20, 2, dtype=np.int64))
        i2 = timedelta_range(start="1 day", periods=10, freq="D")
        i1.union(i2)  # Works
        i2.union(i1)  # Fails with "AttributeError: can't set attribute"

    def test_union_sort_false(self):
        tdi = timedelta_range("1day", periods=5)

        left = tdi[3:]
        right = tdi[:3]

        # Check that we are testing the desired code path
        assert left._can_fast_union(right)

        result = left.union(right)
        tm.assert_index_equal(result, tdi)

        result = left.union(right, sort=False)
        expected = TimedeltaIndex(["4 Days", "5 Days", "1 Days", "2 Day", "3 Days"])
        tm.assert_index_equal(result, expected)

    def test_union_coverage(self):
        idx = TimedeltaIndex(["3d", "1d", "2d"])
        ordered = TimedeltaIndex(idx.sort_values(), freq="infer")
        result = ordered.union(idx)
        tm.assert_index_equal(result, ordered)

        result = ordered[:0].union(ordered)
        tm.assert_index_equal(result, ordered)
        assert result.freq == ordered.freq

    def test_union_bug_1730(self):
        rng_a = timedelta_range("1 day", periods=4, freq="3h")
        rng_b = timedelta_range("1 day", periods=4, freq="4h")

        result = rng_a.union(rng_b)
        exp = TimedeltaIndex(sorted(set(rng_a) | set(rng_b)))
        tm.assert_index_equal(result, exp)

    def test_union_bug_1745(self):
        left = TimedeltaIndex(["1 day 15:19:49.695000"])
        right = TimedeltaIndex(
            ["2 day 13:04:21.322000", "1 day 15:27:24.873000", "1 day 15:31:05.350000"]
        )

        result = left.union(right)
        exp = TimedeltaIndex(sorted(set(left) | set(right)))
        tm.assert_index_equal(result, exp)

    def test_union_bug_4564(self):
        left = timedelta_range("1 day", "30d")
        right = left + pd.offsets.Minute(15)

        result = left.union(right)
        exp = TimedeltaIndex(sorted(set(left) | set(right)))
        tm.assert_index_equal(result, exp)

    def test_union_freq_infer(self):
        # When taking the union of two TimedeltaIndexes, we infer
        #  a freq even if the arguments don't have freq.  This matches
        #  DatetimeIndex behavior.
        tdi = timedelta_range("1 Day", periods=5)
        left = tdi[[0, 1, 3, 4]]
        right = tdi[[2, 3, 1]]

        assert left.freq is None
        assert right.freq is None

        result = left.union(right)
        tm.assert_index_equal(result, tdi)
        assert result.freq == "D"

    def test_intersection_bug_1708(self):
        index_1 = timedelta_range("1 day", periods=4, freq="h")
        index_2 = index_1 + pd.offsets.Hour(5)

        result = index_1.intersection(index_2)
        assert len(result) == 0

        index_1 = timedelta_range("1 day", periods=4, freq="h")
        index_2 = index_1 + pd.offsets.Hour(1)

        result = index_1.intersection(index_2)
        expected = timedelta_range("1 day 01:00:00", periods=3, freq="h")
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    def test_intersection_equal(self, sort):
        # GH 24471 Test intersection outcome given the sort keyword
        # for equal indices intersection should return the original index
        first = timedelta_range("1 day", periods=4, freq="h")
        second = timedelta_range("1 day", periods=4, freq="h")
        intersect = first.intersection(second, sort=sort)
        if sort is None:
            tm.assert_index_equal(intersect, second.sort_values())
        tm.assert_index_equal(intersect, second)

        # Corner cases
        inter = first.intersection(first, sort=sort)
        assert inter is first

    @pytest.mark.parametrize("period_1, period_2", [(0, 4), (4, 0)])
    def test_intersection_zero_length(self, period_1, period_2, sort):
        # GH 24471 test for non overlap the intersection should be zero length
        index_1 = timedelta_range("1 day", periods=period_1, freq="h")
        index_2 = timedelta_range("1 day", periods=period_2, freq="h")
        expected = timedelta_range("1 day", periods=0, freq="h")
        result = index_1.intersection(index_2, sort=sort)
        tm.assert_index_equal(result, expected)

    def test_zero_length_input_index(self, sort):
        # GH 24966 test for 0-len intersections are copied
        index_1 = timedelta_range("1 day", periods=0, freq="h")
        index_2 = timedelta_range("1 day", periods=3, freq="h")
        result = index_1.intersection(index_2, sort=sort)
        assert index_1 is not result
        assert index_2 is not result
        tm.assert_copy(result, index_1)

    @pytest.mark.parametrize(
        "rng, expected",
        # if target has the same name, it is preserved
        [
            (
                timedelta_range("1 day", periods=5, freq="h", name="idx"),
                timedelta_range("1 day", periods=4, freq="h", name="idx"),
            ),
            # if target name is different, it will be reset
            (
                timedelta_range("1 day", periods=5, freq="h", name="other"),
                timedelta_range("1 day", periods=4, freq="h", name=None),
            ),
            # if no overlap exists return empty index
            (
                timedelta_range("1 day", periods=10, freq="h", name="idx")[5:],
                TimedeltaIndex([], freq="h", name="idx"),
            ),
        ],
    )
    def test_intersection(self, rng, expected, sort):
        # GH 4690 (with tz)
        base = timedelta_range("1 day", periods=4, freq="h", name="idx")
        result = base.intersection(rng, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freq == expected.freq

    @pytest.mark.parametrize(
        "rng, expected",
        # part intersection works
        [
            (
                TimedeltaIndex(["5 hour", "2 hour", "4 hour", "9 hour"], name="idx"),
                TimedeltaIndex(["2 hour", "4 hour"], name="idx"),
            ),
            # reordered part intersection
            (
                TimedeltaIndex(["2 hour", "5 hour", "5 hour", "1 hour"], name="other"),
                TimedeltaIndex(["1 hour", "2 hour"], name=None),
            ),
            # reversed index
            (
                TimedeltaIndex(["1 hour", "2 hour", "4 hour", "3 hour"], name="idx")[
                    ::-1
                ],
                TimedeltaIndex(["1 hour", "2 hour", "4 hour", "3 hour"], name="idx"),
            ),
        ],
    )
    def test_intersection_non_monotonic(self, rng, expected, sort):
        # 24471 non-monotonic
        base = TimedeltaIndex(["1 hour", "2 hour", "4 hour", "3 hour"], name="idx")
        result = base.intersection(rng, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

        # if reversed order, frequency is still the same
        if all(base == rng[::-1]) and sort is None:
            assert isinstance(result.freq, Hour)
        else:
            assert result.freq is None


class TestTimedeltaIndexDifference:
    def test_difference_freq(self, sort):
        # GH14323: Difference of TimedeltaIndex should not preserve frequency

        index = timedelta_range("0 days", "5 days", freq="D")

        other = timedelta_range("1 days", "4 days", freq="D")
        expected = TimedeltaIndex(["0 days", "5 days"], freq=None)
        idx_diff = index.difference(other, sort)
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

        # preserve frequency when the difference is a contiguous
        # subset of the original range
        other = timedelta_range("2 days", "5 days", freq="D")
        idx_diff = index.difference(other, sort)
        expected = TimedeltaIndex(["0 days", "1 days"], freq="D")
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

    def test_difference_sort(self, sort):
        index = TimedeltaIndex(
            ["5 days", "3 days", "2 days", "4 days", "1 days", "0 days"]
        )

        other = timedelta_range("1 days", "4 days", freq="D")
        idx_diff = index.difference(other, sort)

        expected = TimedeltaIndex(["5 days", "0 days"], freq=None)

        if sort is None:
            expected = expected.sort_values()

        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

        other = timedelta_range("2 days", "5 days", freq="D")
        idx_diff = index.difference(other, sort)
        expected = TimedeltaIndex(["1 days", "0 days"], freq=None)

        if sort is None:
            expected = expected.sort_values()

        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_eng_formatting.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    reset_option,
    set_eng_float_format,
)

from pandas.io.formats.format import EngFormatter


@pytest.fixture(autouse=True)
def reset_float_format():
    yield
    reset_option("display.float_format")


class TestEngFormatter:
    def test_eng_float_formatter2(self, float_frame):
        df = float_frame
        df.loc[5] = 0

        set_eng_float_format()
        repr(df)

        set_eng_float_format(use_eng_prefix=True)
        repr(df)

        set_eng_float_format(accuracy=0)
        repr(df)

    def test_eng_float_formatter(self):
        df = DataFrame({"A": [1.41, 141.0, 14100, 1410000.0]})

        set_eng_float_format()
        result = df.to_string()
        expected = (
            "             A\n"
            "0    1.410E+00\n"
            "1  141.000E+00\n"
            "2   14.100E+03\n"
            "3    1.410E+06"
        )
        assert result == expected

        set_eng_float_format(use_eng_prefix=True)
        result = df.to_string()
        expected = "         A\n0    1.410\n1  141.000\n2  14.100k\n3   1.410M"
        assert result == expected

        set_eng_float_format(accuracy=0)
        result = df.to_string()
        expected = "         A\n0    1E+00\n1  141E+00\n2   14E+03\n3    1E+06"
        assert result == expected

    def compare(self, formatter, input, output):
        formatted_input = formatter(input)
        assert formatted_input == output

    def compare_all(self, formatter, in_out):
        """
        Parameters:
        -----------
        formatter: EngFormatter under test
        in_out: list of tuples. Each tuple = (number, expected_formatting)

        It is tested if 'formatter(number) == expected_formatting'.
        *number* should be >= 0 because formatter(-number) == fmt is also
        tested. *fmt* is derived from *expected_formatting*
        """
        for input, output in in_out:
            self.compare(formatter, input, output)
            self.compare(formatter, -input, "-" + output[1:])

    def test_exponents_with_eng_prefix(self):
        formatter = EngFormatter(accuracy=3, use_eng_prefix=True)
        f = np.sqrt(2)
        in_out = [
            (f * 10**-24, " 1.414y"),
            (f * 10**-23, " 14.142y"),
            (f * 10**-22, " 141.421y"),
            (f * 10**-21, " 1.414z"),
            (f * 10**-20, " 14.142z"),
            (f * 10**-19, " 141.421z"),
            (f * 10**-18, " 1.414a"),
            (f * 10**-17, " 14.142a"),
            (f * 10**-16, " 141.421a"),
            (f * 10**-15, " 1.414f"),
            (f * 10**-14, " 14.142f"),
            (f * 10**-13, " 141.421f"),
            (f * 10**-12, " 1.414p"),
            (f * 10**-11, " 14.142p"),
            (f * 10**-10, " 141.421p"),
            (f * 10**-9, " 1.414n"),
            (f * 10**-8, " 14.142n"),
            (f * 10**-7, " 141.421n"),
            (f * 10**-6, " 1.414u"),
            (f * 10**-5, " 14.142u"),
            (f * 10**-4, " 141.421u"),
            (f * 10**-3, " 1.414m"),
            (f * 10**-2, " 14.142m"),
            (f * 10**-1, " 141.421m"),
            (f * 10**0, " 1.414"),
            (f * 10**1, " 14.142"),
            (f * 10**2, " 141.421"),
            (f * 10**3, " 1.414k"),
            (f * 10**4, " 14.142k"),
            (f * 10**5, " 141.421k"),
            (f * 10**6, " 1.414M"),
            (f * 10**7, " 14.142M"),
            (f * 10**8, " 141.421M"),
            (f * 10**9, " 1.414G"),
            (f * 10**10, " 14.142G"),
            (f * 10**11, " 141.421G"),
            (f * 10**12, " 1.414T"),
            (f * 10**13, " 14.142T"),
            (f * 10**14, " 141.421T"),
            (f * 10**15, " 1.414P"),
            (f * 10**16, " 14.142P"),
            (f * 10**17, " 141.421P"),
            (f * 10**18, " 1.414E"),
            (f * 10**19, " 14.142E"),
            (f * 10**20, " 141.421E"),
            (f * 10**21, " 1.414Z"),
            (f * 10**22, " 14.142Z"),
            (f * 10**23, " 141.421Z"),
            (f * 10**24, " 1.414Y"),
            (f * 10**25, " 14.142Y"),
            (f * 10**26, " 141.421Y"),
        ]
        self.compare_all(formatter, in_out)

    def test_exponents_without_eng_prefix(self):
        formatter = EngFormatter(accuracy=4, use_eng_prefix=False)
        f = np.pi
        in_out = [
            (f * 10**-24, " 3.1416E-24"),
            (f * 10**-23, " 31.4159E-24"),
            (f * 10**-22, " 314.1593E-24"),
            (f * 10**-21, " 3.1416E-21"),
            (f * 10**-20, " 31.4159E-21"),
            (f * 10**-19, " 314.1593E-21"),
            (f * 10**-18, " 3.1416E-18"),
            (f * 10**-17, " 31.4159E-18"),
            (f * 10**-16, " 314.1593E-18"),
            (f * 10**-15, " 3.1416E-15"),
            (f * 10**-14, " 31.4159E-15"),
            (f * 10**-13, " 314.1593E-15"),
            (f * 10**-12, " 3.1416E-12"),
            (f * 10**-11, " 31.4159E-12"),
            (f * 10**-10, " 314.1593E-12"),
            (f * 10**-9, " 3.1416E-09"),
            (f * 10**-8, " 31.4159E-09"),
            (f * 10**-7, " 314.1593E-09"),
            (f * 10**-6, " 3.1416E-06"),
            (f * 10**-5, " 31.4159E-06"),
            (f * 10**-4, " 314.1593E-06"),
            (f * 10**-3, " 3.1416E-03"),
            (f * 10**-2, " 31.4159E-03"),
            (f * 10**-1, " 314.1593E-03"),
            (f * 10**0, " 3.1416E+00"),
            (f * 10**1, " 31.4159E+00"),
            (f * 10**2, " 314.1593E+00"),
            (f * 10**3, " 3.1416E+03"),
            (f * 10**4, " 31.4159E+03"),
            (f * 10**5, " 314.1593E+03"),
            (f * 10**6, " 3.1416E+06"),
            (f * 10**7, " 31.4159E+06"),
            (f * 10**8, " 314.1593E+06"),
            (f * 10**9, " 3.1416E+09"),
            (f * 10**10, " 31.4159E+09"),
            (f * 10**11, " 314.1593E+09"),
            (f * 10**12, " 3.1416E+12"),
            (f * 10**13, " 31.4159E+12"),
            (f * 10**14, " 314.1593E+12"),
            (f * 10**15, " 3.1416E+15"),
            (f * 10**16, " 31.4159E+15"),
            (f * 10**17, " 314.1593E+15"),
            (f * 10**18, " 3.1416E+18"),
            (f * 10**19, " 31.4159E+18"),
            (f * 10**20, " 314.1593E+18"),
            (f * 10**21, " 3.1416E+21"),
            (f * 10**22, " 31.4159E+21"),
            (f * 10**23, " 314.1593E+21"),
            (f * 10**24, " 3.1416E+24"),
            (f * 10**25, " 31.4159E+24"),
            (f * 10**26, " 314.1593E+24"),
        ]
        self.compare_all(formatter, in_out)

    def test_rounding(self):
        formatter = EngFormatter(accuracy=3, use_eng_prefix=True)
        in_out = [
            (5.55555, " 5.556"),
            (55.5555, " 55.556"),
            (555.555, " 555.555"),
            (5555.55, " 5.556k"),
            (55555.5, " 55.556k"),
            (555555, " 555.555k"),
        ]
        self.compare_all(formatter, in_out)

        formatter = EngFormatter(accuracy=1, use_eng_prefix=True)
        in_out = [
            (5.55555, " 5.6"),
            (55.5555, " 55.6"),
            (555.555, " 555.6"),
            (5555.55, " 5.6k"),
            (55555.5, " 55.6k"),
            (555555, " 555.6k"),
        ]
        self.compare_all(formatter, in_out)

        formatter = EngFormatter(accuracy=0, use_eng_prefix=True)
        in_out = [
            (5.55555, " 6"),
            (55.5555, " 56"),
            (555.555, " 556"),
            (5555.55, " 6k"),
            (55555.5, " 56k"),
            (555555, " 556k"),
        ]
        self.compare_all(formatter, in_out)

        formatter = EngFormatter(accuracy=3, use_eng_prefix=True)
        result = formatter(0)
        assert result == " 0.000"

    def test_nan(self):
        # Issue #11981

        formatter = EngFormatter(accuracy=1, use_eng_prefix=True)
        result = formatter(np.nan)
        assert result == "NaN"

        df = DataFrame(
            {
                "a": [1.5, 10.3, 20.5],
                "b": [50.3, 60.67, 70.12],
                "c": [100.2, 101.33, 120.33],
            }
        )
        pt = df.pivot_table(values="a", index="b", columns="c")
        set_eng_float_format(accuracy=1)
        result = pt.to_string()
        assert "NaN" in result

    def test_inf(self):
        # Issue #11981

        formatter = EngFormatter(accuracy=1, use_eng_prefix=True)
        result = formatter(np.inf)
        assert result == "inf"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\test_pivot_multilevel.py ===
import numpy as np
import pytest

from pandas._libs import lib

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "input_index, input_columns, input_values, "
    "expected_values, expected_columns, expected_index",
    [
        (
            ["lev4"],
            "lev3",
            "values",
            [
                [0.0, np.nan],
                [np.nan, 1.0],
                [2.0, np.nan],
                [np.nan, 3.0],
                [4.0, np.nan],
                [np.nan, 5.0],
                [6.0, np.nan],
                [np.nan, 7.0],
            ],
            Index([1, 2], name="lev3"),
            Index([1, 2, 3, 4, 5, 6, 7, 8], name="lev4"),
        ),
        (
            ["lev4"],
            "lev3",
            lib.no_default,
            [
                [1.0, np.nan, 1.0, np.nan, 0.0, np.nan],
                [np.nan, 1.0, np.nan, 1.0, np.nan, 1.0],
                [1.0, np.nan, 2.0, np.nan, 2.0, np.nan],
                [np.nan, 1.0, np.nan, 2.0, np.nan, 3.0],
                [2.0, np.nan, 1.0, np.nan, 4.0, np.nan],
                [np.nan, 2.0, np.nan, 1.0, np.nan, 5.0],
                [2.0, np.nan, 2.0, np.nan, 6.0, np.nan],
                [np.nan, 2.0, np.nan, 2.0, np.nan, 7.0],
            ],
            MultiIndex.from_tuples(
                [
                    ("lev1", 1),
                    ("lev1", 2),
                    ("lev2", 1),
                    ("lev2", 2),
                    ("values", 1),
                    ("values", 2),
                ],
                names=[None, "lev3"],
            ),
            Index([1, 2, 3, 4, 5, 6, 7, 8], name="lev4"),
        ),
        (
            ["lev1", "lev2"],
            "lev3",
            "values",
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            Index([1, 2], name="lev3"),
            MultiIndex.from_tuples(
                [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev1", "lev2"]
            ),
        ),
        (
            ["lev1", "lev2"],
            "lev3",
            lib.no_default,
            [[1, 2, 0, 1], [3, 4, 2, 3], [5, 6, 4, 5], [7, 8, 6, 7]],
            MultiIndex.from_tuples(
                [("lev4", 1), ("lev4", 2), ("values", 1), ("values", 2)],
                names=[None, "lev3"],
            ),
            MultiIndex.from_tuples(
                [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev1", "lev2"]
            ),
        ),
    ],
)
def test_pivot_list_like_index(
    input_index,
    input_columns,
    input_values,
    expected_values,
    expected_columns,
    expected_index,
):
    # GH 21425, test when index is given a list
    df = pd.DataFrame(
        {
            "lev1": [1, 1, 1, 1, 2, 2, 2, 2],
            "lev2": [1, 1, 2, 2, 1, 1, 2, 2],
            "lev3": [1, 2, 1, 2, 1, 2, 1, 2],
            "lev4": [1, 2, 3, 4, 5, 6, 7, 8],
            "values": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )

    result = df.pivot(index=input_index, columns=input_columns, values=input_values)
    expected = pd.DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "input_index, input_columns, input_values, "
    "expected_values, expected_columns, expected_index",
    [
        (
            "lev4",
            ["lev3"],
            "values",
            [
                [0.0, np.nan],
                [np.nan, 1.0],
                [2.0, np.nan],
                [np.nan, 3.0],
                [4.0, np.nan],
                [np.nan, 5.0],
                [6.0, np.nan],
                [np.nan, 7.0],
            ],
            Index([1, 2], name="lev3"),
            Index([1, 2, 3, 4, 5, 6, 7, 8], name="lev4"),
        ),
        (
            ["lev1", "lev2"],
            ["lev3"],
            "values",
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            Index([1, 2], name="lev3"),
            MultiIndex.from_tuples(
                [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev1", "lev2"]
            ),
        ),
        (
            ["lev1"],
            ["lev2", "lev3"],
            "values",
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            MultiIndex.from_tuples(
                [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev2", "lev3"]
            ),
            Index([1, 2], name="lev1"),
        ),
        (
            ["lev1", "lev2"],
            ["lev3", "lev4"],
            "values",
            [
                [0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0],
            ],
            MultiIndex.from_tuples(
                [(1, 1), (2, 2), (1, 3), (2, 4), (1, 5), (2, 6), (1, 7), (2, 8)],
                names=["lev3", "lev4"],
            ),
            MultiIndex.from_tuples(
                [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev1", "lev2"]
            ),
        ),
    ],
)
def test_pivot_list_like_columns(
    input_index,
    input_columns,
    input_values,
    expected_values,
    expected_columns,
    expected_index,
):
    # GH 21425, test when columns is given a list
    df = pd.DataFrame(
        {
            "lev1": [1, 1, 1, 1, 2, 2, 2, 2],
            "lev2": [1, 1, 2, 2, 1, 1, 2, 2],
            "lev3": [1, 2, 1, 2, 1, 2, 1, 2],
            "lev4": [1, 2, 3, 4, 5, 6, 7, 8],
            "values": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )

    result = df.pivot(index=input_index, columns=input_columns, values=input_values)
    expected = pd.DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    tm.assert_frame_equal(result, expected)


def test_pivot_multiindexed_rows_and_cols(using_array_manager):
    # GH 36360

    df = pd.DataFrame(
        data=np.arange(12).reshape(4, 3),
        columns=MultiIndex.from_tuples(
            [(0, 0), (0, 1), (0, 2)], names=["col_L0", "col_L1"]
        ),
        index=MultiIndex.from_tuples(
            [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 0, 0)],
            names=["idx_L0", "idx_L1", "idx_L2"],
        ),
    )

    res = df.pivot_table(
        index=["idx_L0"],
        columns=["idx_L1"],
        values=[(0, 1)],
        aggfunc=lambda col: col.values.sum(),
    )

    expected = pd.DataFrame(
        data=[[5, np.nan], [10, 7.0]],
        columns=MultiIndex.from_tuples(
            [(0, 1, 0), (0, 1, 1)], names=["col_L0", "col_L1", "idx_L1"]
        ),
        index=Index([0, 1], dtype="int64", name="idx_L0"),
    )
    if not using_array_manager:
        # BlockManager does not preserve the dtypes
        expected = expected.astype("float64")

    tm.assert_frame_equal(res, expected)


def test_pivot_df_multiindex_index_none():
    # GH 23955
    df = pd.DataFrame(
        [
            ["A", "A1", "label1", 1],
            ["A", "A2", "label2", 2],
            ["B", "A1", "label1", 3],
            ["B", "A2", "label2", 4],
        ],
        columns=["index_1", "index_2", "label", "value"],
    )
    df = df.set_index(["index_1", "index_2"])

    result = df.pivot(columns="label", values="value")
    expected = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 2.0], [3.0, np.nan], [np.nan, 4.0]],
        index=df.index,
        columns=Index(["label1", "label2"], name="label"),
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\test_simplex.py ===
from sympy.core.numbers import Rational
from sympy.core.relational import Eq, Ne
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.core.singleton import S
from sympy.core.random import random, choice
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import randprime
from sympy.matrices.dense import Matrix
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.solvers.simplex import (_lp as lp, _primal_dual,
    UnboundedLPError, InfeasibleLPError, lpmin, lpmax,
    _m, _abcd, _simplex, linprog)

from sympy.external.importtools import import_module

from sympy.testing.pytest import raises

from sympy.abc import x, y, z


np = import_module("numpy")
scipy = import_module("scipy")


def test_lp():
    r1 = y + 2*z <= 3
    r2 = -x - 3*z <= -2
    r3 = 2*x + y + 7*z <= 5
    constraints = [r1, r2, r3, x >= 0, y >= 0, z >= 0]
    objective = -x - y - 5 * z
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True

    r1 = x - y + 2*z <= 3
    r2 = -x + 2*y - 3*z <= -2
    r3 = 2*x + y - 7*z <= -5
    constraints = [r1, r2, r3, x >= 0, y >= 0, z >= 0]
    objective = -x - y - 5*z
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True

    r1 = x - y + 2*z <= -4
    r2 = -x + 2*y - 3*z <= 8
    r3 = 2*x + y - 7*z <= 10
    constraints = [r1, r2, r3, x >= 0, y >= 0, z >= 0]
    const = 2
    objective = -x-y-5*z+const # has constant term
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True

    # Section 4 Problem 1 from
    # http://web.tecnico.ulisboa.pt/mcasquilho/acad/or/ftp/FergusonUCLA_LP.pdf
    # answer on page 55
    v = x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
    r1 = x1 - x2 - 2*x3 - x4 <= 4
    r2 = 2*x1 + x3 -4*x4 <= 2
    r3 = -2*x1 + x2 + x4 <= 1
    objective, constraints = x1 - 2*x2 - 3*x3 - x4, [r1, r2, r3] + [
        i >= 0 for i in v]
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert ans == (4, {x1: 7, x2: 0, x3: 0, x4: 3})

    # input contains Floats
    r1 = x - y + 2.0*z <= -4
    r2 = -x + 2*y - 3.0*z <= 8
    r3 = 2*x + y - 7*z <= 10
    constraints = [r1, r2, r3] + [i >= 0 for i in (x, y, z)]
    objective = -x-y-5*z
    optimum, argmax = lp(max, objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True

    # input contains non-float or non-Rational
    r1 = x - y + sqrt(2) * z <= -4
    r2 = -x + 2*y - 3*z <= 8
    r3 = 2*x + y - 7*z <= 10
    raises(TypeError, lambda: lp(max, -x-y-5*z, [r1, r2, r3]))

    r1 = x >= 0
    raises(UnboundedLPError, lambda: lp(max, x, [r1]))
    r2 = x <= -1
    raises(InfeasibleLPError, lambda: lp(max, x, [r1, r2]))

    # strict inequalities are not allowed
    r1 = x > 0
    raises(TypeError, lambda: lp(max, x, [r1]))

    # not equals not allowed
    r1 = Ne(x, 0)
    raises(TypeError, lambda: lp(max, x, [r1]))

    def make_random_problem(nvar=2, num_constraints=2, sparsity=.1):
        def rand():
            if random() < sparsity:
                return sympify(0)
            int1, int2 = [randprime(0, 200) for _ in range(2)]
            return Rational(int1, int2)*choice([-1, 1])
        variables = symbols('x1:%s' % (nvar + 1))
        constraints = [(sum(rand()*x for x in variables) <= rand())
                       for _ in range(num_constraints)]
        objective = sum(rand() * x for x in variables)
        return objective, constraints, variables

    # equality
    r1 = Eq(x, y)
    r2 = Eq(y, z)
    r3 = z <= 3
    constraints = [r1, r2, r3]
    objective = x
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True


def test_simplex():
    L = [
        [[1, 1], [-1, 1], [0, 1], [-1, 0]],
        [5, 1, 2, -1],
        [[1, 1]],
        [-1]]
    A, B, C, D = _abcd(_m(*L), list=False)
    assert _simplex(A, B, -C, -D) == (-6, [3, 2], [1, 0, 0, 0])
    assert _simplex(A, B, -C, -D, dual=True) == (-6,
        [1, 0, 0, 0], [5, 0])

    assert _simplex([[]],[],[[1]],[0]) == (0, [0], [])

    # handling of Eq (or Eq-like x<=y, x>=y conditions)
    assert lpmax(x - y, [x <= y + 2, x >= y + 2, x >= 0, y >= 0]
        ) == (2, {x: 2, y: 0})
    assert lpmax(x - y, [x <= y + 2, Eq(x, y + 2), x >= 0, y >= 0]
        ) == (2, {x: 2, y: 0})
    assert lpmax(x - y, [x <= y + 2, Eq(x, 2)]) == (2, {x: 2, y: 0})
    assert lpmax(y, [Eq(y, 2)]) == (2, {y: 2})

    # the conditions are equivalent to Eq(x, y + 2)
    assert lpmin(y, [x <= y + 2, x >= y + 2, y >= 0]
        ) == (0, {x: 2, y: 0})
    # equivalent to Eq(y, -2)
    assert lpmax(y, [0 <= y + 2, 0 >= y + 2]) == (-2, {y: -2})
    assert lpmax(y, [0 <= y + 2, 0 >= y + 2, y <= 0]
        ) == (-2, {y: -2})

    # extra symbols symbols
    assert lpmin(x, [y >= 1, x >= y]) == (1, {x: 1, y: 1})
    assert lpmin(x, [y >= 1, x >= y + z, x >= 0, z >= 0]
        ) == (1, {x: 1, y: 1, z: 0})

    # detect oscillation
    # o1
    v = x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
    raises(InfeasibleLPError, lambda: lpmin(
        9*x2 - 8*x3 + 3*x4 + 6,
        [5*x2 - 2*x3 <= 0,
        -x1 - 8*x2 + 9*x3 <= -3,
        10*x1 - x2+ 9*x4 <= -4] + [i >= 0 for i in v]))
    # o2 - equations fed to lpmin are changed into a matrix
    # system that doesn't oscillate and has the same solution
    # as below
    M = linear_eq_to_matrix
    f = 5*x2 + x3 + 4*x4 - x1
    L = 5*x2 + 2*x3 + 5*x4 - (x1 + 5)
    cond = [L <= 0] + [Eq(3*x2 + x4, 2), Eq(-x1 + x3 + 2*x4, 1)]
    c, d = M(f, v)
    a, b = M(L, v)
    aeq, beq = M(cond[1:], v)
    ans = (S(9)/2, [0, S(1)/2, 0, S(1)/2])
    assert linprog(c, a, b, aeq, beq, bounds=(0, 1)) == ans
    lpans = lpmin(f, cond + [x1 >= 0, x1 <= 1,
        x2 >= 0, x2 <= 1, x3 >= 0, x3 <= 1, x4 >= 0, x4 <= 1])
    assert (lpans[0], list(lpans[1].values())) == ans


def test_lpmin_lpmax():
    v = x1, x2, y1, y2 = symbols('x1 x2 y1 y2')
    L = [[1, -1]], [1], [[1, 1]], [2]
    a, b, c, d = [Matrix(i) for i in L]
    m = Matrix([[a, b], [c, d]])
    f, constr = _primal_dual(m)[0]
    ans = lpmin(f, constr + [i >= 0 for i in v[:2]])
    assert ans == (-1, {x1: 1, x2: 0}),ans

    L = [[1, -1], [1, 1]], [1, 1], [[1, 1]], [2]
    a, b, c, d = [Matrix(i) for i in L]
    m = Matrix([[a, b], [c, d]])
    f, constr = _primal_dual(m)[1]
    ans = lpmax(f, constr + [i >= 0 for i in v[-2:]])
    assert ans == (-1, {y1: 1, y2: 0})


def test_linprog():
    for do in range(2):
        if not do:
            M = lambda a, b: linear_eq_to_matrix(a, b)
        else:
            # check matrices as list
            M = lambda a, b: tuple([
                i.tolist() for i in linear_eq_to_matrix(a, b)])

        v = x, y, z = symbols('x1:4')
        f = x + y - 2*z
        c = M(f, v)[0]
        ineq = [7*x + 4*y - 7*z <= 3,
            3*x - y + 10*z <= 6,
            x >= 0, y >= 0, z >= 0]
        ab = M([i.lts - i.gts for i in ineq], v)
        ans = (-S(6)/5, [0, 0, S(3)/5])
        assert lpmin(f, ineq) == (ans[0], dict(zip(v, ans[1])))
        assert linprog(c, *ab) == ans

        f += 1
        c = M(f, v)[0]
        eq = [Eq(y - 9*x, 1)]
        abeq = M([i.lhs - i.rhs for i in eq], v)
        ans = (1 - S(2)/5, [0, 1, S(7)/10])
        assert lpmin(f, ineq + eq) == (ans[0], dict(zip(v, ans[1])))
        assert linprog(c, *ab, *abeq) == (ans[0] - 1, ans[1])

        eq = [z - y <= S.Half]
        abeq = M([i.lhs - i.rhs for i in eq], v)
        ans = (1 - S(10)/9, [0, S(1)/9, S(11)/18])
        assert lpmin(f, ineq + eq) == (ans[0], dict(zip(v, ans[1])))
        assert linprog(c, *ab, *abeq) == (ans[0] - 1, ans[1])

        bounds = [(0, None), (0, None), (None, S.Half)]
        ans = (0, [0, 0, S.Half])
        assert lpmin(f, ineq + [z <= S.Half]) == (
            ans[0], dict(zip(v, ans[1])))
        assert linprog(c, *ab, bounds=bounds) == (ans[0] - 1, ans[1])
        assert linprog(c, *ab, bounds={v.index(z): bounds[-1]}
            ) == (ans[0] - 1, ans[1])
        eq = [z - y <= S.Half]

    assert linprog([[1]], [], [], bounds=(2, 3)) == (2, [2])
    assert linprog([1], [], [], bounds=(2, 3)) == (2, [2])
    assert linprog([1], bounds=(2, 3)) == (2, [2])
    assert linprog([1, -1], [[1, 1]], [2], bounds={1:(None, None)}
        ) == (-2, [0, 2])
    assert linprog([1, -1], [[1, 1]], [5], bounds={1:(3, None)}
        ) == (-5, [0, 5])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_matrices.py ===
import pytest
import sys
from mpmath import *

def test_matrix_basic():
    A1 = matrix(3)
    for i in range(3):
        A1[i,i] = 1
    assert A1 == eye(3)
    assert A1 == matrix(A1)
    A2 = matrix(3, 2)
    assert not A2._matrix__data
    A3 = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert list(A3) == list(range(1, 10))
    A3[1,1] = 0
    assert not (1, 1) in A3._matrix__data
    A4 = matrix([[1, 2, 3], [4, 5, 6]])
    A5 = matrix([[6, -1], [3, 2], [0, -3]])
    assert A4 * A5 == matrix([[12, -6], [39, -12]])
    assert A1 * A3 == A3 * A1 == A3
    pytest.raises(ValueError, lambda: A2*A2)
    l = [[10, 20, 30], [40, 0, 60], [70, 80, 90]]
    A6 = matrix(l)
    assert A6.tolist() == l
    assert A6 == eval(repr(A6))
    A6 = fp.matrix(A6)
    assert A6 == eval(repr(A6))
    assert A6*1j == eval(repr(A6*1j))
    assert A3 * 10 == 10 * A3 == A6
    assert A2.rows == 3
    assert A2.cols == 2
    A3.rows = 2
    A3.cols = 2
    assert len(A3._matrix__data) == 3
    assert A4 + A4 == 2*A4
    pytest.raises(ValueError, lambda: A4 + A2)
    assert sum(A1 - A1) == 0
    A7 = matrix([[1, 2], [3, 4], [5, 6], [7, 8]])
    x = matrix([10, -10])
    assert A7*x == matrix([-10, -10, -10, -10])
    A8 = ones(5)
    assert sum((A8 + 1) - (2 - zeros(5))) == 0
    assert (1 + ones(4)) / 2 - 1 == zeros(4)
    assert eye(3)**10 == eye(3)
    pytest.raises(ValueError, lambda: A7**2)
    A9 = randmatrix(3)
    A10 = matrix(A9)
    A9[0,0] = -100
    assert A9 != A10
    assert nstr(A9)

def test_matmul():
    """
    Test the PEP465 "@" matrix multiplication syntax.
    To avoid syntax errors when importing this file in Python 3.5 and below, we have to use exec() - sorry for that.
    """
    # TODO remove exec() wrapper as soon as we drop support for Python <= 3.5
    if sys.hexversion < 0x30500f0:
        # we are on Python < 3.5
        pytest.skip("'@' (__matmul__) is only supported in Python 3.5 or newer")
    A4 = matrix([[1, 2, 3], [4, 5, 6]])
    A5 = matrix([[6, -1], [3, 2], [0, -3]])
    exec("assert A4 @ A5 == A4 * A5")

def test_matrix_slices():
    A = matrix([    [1, 2, 3],
                        [4, 5 ,6],
                        [7, 8 ,9]])
    V = matrix([1,2,3,4,5])

    # Get slice
    assert A[:,:] == A
    assert A[:,1] == matrix([[2],[5],[8]])
    assert A[2,:] == matrix([[7, 8 ,9]])
    assert A[1:3,1:3] == matrix([[5,6],[8,9]])
    assert V[2:4] == matrix([3,4])
    pytest.raises(IndexError, lambda: A[:,1:6])

    # Assign slice with matrix
    A1 = matrix(3)
    A1[:,:] = A
    assert A1[:,:] == matrix([[1, 2, 3],
                                        [4, 5 ,6],
                                        [7, 8 ,9]])
    A1[0,:] = matrix([[10, 11, 12]])
    assert A1 == matrix([ [10, 11, 12],
                                    [4, 5 ,6],
                                    [7, 8 ,9]])
    A1[:,2] = matrix([[13], [14], [15]])
    assert A1 == matrix([ [10, 11, 13],
                                    [4, 5 ,14],
                                    [7, 8 ,15]])
    A1[:2,:2] = matrix([[16, 17], [18 , 19]])
    assert A1 == matrix([ [16, 17, 13],
                                    [18, 19 ,14],
                                    [7, 8 ,15]])
    V[1:3] = 10
    assert V == matrix([1,10,10,4,5])
    with pytest.raises(ValueError):
        A1[2,:] = A[:,1]

    with pytest.raises(IndexError):
        A1[2,1:20] = A[:,:]

    # Assign slice with scalar
    A1[:,2] = 10
    assert A1 == matrix([ [16, 17, 10],
                                    [18, 19 ,10],
                                    [7, 8 ,10]])
    A1[:,:] = 40
    for x in A1:
        assert x == 40


def test_matrix_power():
    A = matrix([[1, 2], [3, 4]])
    assert A**2 == A*A
    assert A**3 == A*A*A
    assert A**-1 == inverse(A)
    assert A**-2 == inverse(A*A)

def test_matrix_transform():
    A = matrix([[1, 2], [3, 4], [5, 6]])
    assert A.T == A.transpose() == matrix([[1, 3, 5], [2, 4, 6]])
    swap_row(A, 1, 2)
    assert A == matrix([[1, 2], [5, 6], [3, 4]])
    l = [1, 2]
    swap_row(l, 0, 1)
    assert l == [2, 1]
    assert extend(eye(3), [1,2,3]) == matrix([[1,0,0,1],[0,1,0,2],[0,0,1,3]])

def test_matrix_conjugate():
    A = matrix([[1 + j, 0], [2, j]])
    assert A.conjugate() == matrix([[mpc(1, -1), 0], [2, mpc(0, -1)]])
    assert A.transpose_conj() == A.H == matrix([[mpc(1, -1), 2],
                                                [0, mpc(0, -1)]])

def test_matrix_creation():
    assert diag([1, 2, 3]) == matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    A1 = ones(2, 3)
    assert A1.rows == 2 and A1.cols == 3
    for a in A1:
        assert a == 1
    A2 = zeros(3, 2)
    assert A2.rows == 3 and A2.cols == 2
    for a in A2:
        assert a == 0
    assert randmatrix(10) != randmatrix(10)
    one = mpf(1)
    assert hilbert(3) == matrix([[one, one/2, one/3],
                                 [one/2, one/3, one/4],
                                 [one/3, one/4, one/5]])

def test_norms():
    # matrix norms
    A = matrix([[1, -2], [-3, -1], [2, 1]])
    assert mnorm(A,1) == 6
    assert mnorm(A,inf) == 4
    assert mnorm(A,'F') == sqrt(20)
    # vector norms
    assert norm(-3) == 3
    x = [1, -2, 7, -12]
    assert norm(x, 1) == 22
    assert round(norm(x, 2), 10) == 14.0712472795
    assert round(norm(x, 10), 10) == 12.0054633727
    assert norm(x, inf) == 12

def test_vector():
    x = matrix([0, 1, 2, 3, 4])
    assert x == matrix([[0], [1], [2], [3], [4]])
    assert x[3] == 3
    assert len(x._matrix__data) == 4
    assert list(x) == list(range(5))
    x[0] = -10
    x[4] = 0
    assert x[0] == -10
    assert len(x) == len(x.T) == 5
    assert x.T*x == matrix([[114]])

def test_matrix_copy():
    A = ones(6)
    B = A.copy()
    C = +A
    assert A == B
    assert A == C
    B[0,0] = 0
    assert A != B
    C[0,0] = 42
    assert A != C

def test_matrix_numpy():
    try:
        import numpy
    except ImportError:
        return
    l = [[1, 2], [3, 4], [5, 6]]
    a = numpy.array(l)
    assert matrix(l) == matrix(a)

def test_interval_matrix_scalar_mult():
    """Multiplication of iv.matrix and any scalar type"""
    a = mpi(-1, 1)
    b = a + a * 2j
    c = mpf(42)
    d = c + c * 2j
    e = 1.234
    f = fp.convert(e)
    g = e + e * 3j
    h = fp.convert(g)
    M = iv.ones(1)
    for x in [a, b, c, d, e, f, g, h]:
        assert x * M == iv.matrix([x])
        assert M * x == iv.matrix([x])

@pytest.mark.xfail()
def test_interval_matrix_matrix_mult():
    """Multiplication of iv.matrix and other matrix types"""
    A = ones(1)
    B = fp.ones(1)
    M = iv.ones(1)
    for X in [A, B, M]:
        assert X * M == iv.matrix(X)
        assert X * M == X
        assert M * X == iv.matrix(X)
        assert M * X == X

def test_matrix_conversion_to_iv():
    # Test that matrices with foreign datatypes are properly converted
    for other_type_eye in [eye(3), fp.eye(3), iv.eye(3)]:
        A = iv.matrix(other_type_eye)
        B = iv.eye(3)
        assert type(A[0,0]) == type(B[0,0])
        assert A.tolist() == B.tolist()

def test_interval_matrix_mult_bug():
    # regression test for interval matrix multiplication:
    # result must be nonzero-width and contain the exact result
    x = convert('1.00000000000001') # note: this is implicitly rounded to some near mpf float value
    A = matrix([[x]])
    B = iv.matrix(A)
    C = iv.matrix([[x]])
    assert B == C
    B = B * B
    C = C * C
    assert B == C
    assert B[0, 0].delta > 1e-16
    assert B[0, 0].delta < 3e-16
    assert C[0, 0].delta > 1e-16
    assert C[0, 0].delta < 3e-16
    assert mp.mpf('1.00000000000001998401444325291756783368705994138804689654') in B[0, 0]
    assert mp.mpf('1.00000000000001998401444325291756783368705994138804689654') in C[0, 0]
    # the following caused an error before the bug was fixed
    assert iv.matrix(mp.eye(2)) * (iv.ones(2) + mpi(1, 2)) == iv.matrix([[mpi(2, 3), mpi(2, 3)], [mpi(2, 3), mpi(2, 3)]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\test_accessor.py ===
import string

import numpy as np
import pytest

import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


class TestSeriesAccessor:
    def test_to_dense(self):
        ser = pd.Series([0, 1, 0, 10], dtype="Sparse[int64]")
        result = ser.sparse.to_dense()
        expected = pd.Series([0, 1, 0, 10])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("attr", ["npoints", "density", "fill_value", "sp_values"])
    def test_get_attributes(self, attr):
        arr = SparseArray([0, 1])
        ser = pd.Series(arr)

        result = getattr(ser.sparse, attr)
        expected = getattr(arr, attr)
        assert result == expected

    def test_from_coo(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")

        row = [0, 3, 1, 0]
        col = [0, 3, 1, 2]
        data = [4, 5, 7, 9]

        sp_array = scipy_sparse.coo_matrix((data, (row, col)))
        result = pd.Series.sparse.from_coo(sp_array)

        index = pd.MultiIndex.from_arrays(
            [
                np.array([0, 0, 1, 3], dtype=np.int32),
                np.array([0, 2, 1, 3], dtype=np.int32),
            ],
        )
        expected = pd.Series([4, 9, 7, 5], index=index, dtype="Sparse[int]")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "sort_labels, expected_rows, expected_cols, expected_values_pos",
        [
            (
                False,
                [("b", 2), ("a", 2), ("b", 1), ("a", 1)],
                [("z", 1), ("z", 2), ("x", 2), ("z", 0)],
                {1: (1, 0), 3: (3, 3)},
            ),
            (
                True,
                [("a", 1), ("a", 2), ("b", 1), ("b", 2)],
                [("x", 2), ("z", 0), ("z", 1), ("z", 2)],
                {1: (1, 2), 3: (0, 1)},
            ),
        ],
    )
    def test_to_coo(
        self, sort_labels, expected_rows, expected_cols, expected_values_pos
    ):
        sp_sparse = pytest.importorskip("scipy.sparse")

        values = SparseArray([0, np.nan, 1, 0, None, 3], fill_value=0)
        index = pd.MultiIndex.from_tuples(
            [
                ("b", 2, "z", 1),
                ("a", 2, "z", 2),
                ("a", 2, "z", 1),
                ("a", 2, "x", 2),
                ("b", 1, "z", 1),
                ("a", 1, "z", 0),
            ]
        )
        ss = pd.Series(values, index=index)

        expected_A = np.zeros((4, 4))
        for value, (row, col) in expected_values_pos.items():
            expected_A[row, col] = value

        A, rows, cols = ss.sparse.to_coo(
            row_levels=(0, 1), column_levels=(2, 3), sort_labels=sort_labels
        )
        assert isinstance(A, sp_sparse.coo_matrix)
        tm.assert_numpy_array_equal(A.toarray(), expected_A)
        assert rows == expected_rows
        assert cols == expected_cols

    def test_non_sparse_raises(self):
        ser = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError, match=".sparse"):
            ser.sparse.density


class TestFrameAccessor:
    def test_accessor_raises(self):
        df = pd.DataFrame({"A": [0, 1]})
        with pytest.raises(AttributeError, match="sparse"):
            df.sparse

    @pytest.mark.parametrize("format", ["csc", "csr", "coo"])
    @pytest.mark.parametrize("labels", [None, list(string.ascii_letters[:10])])
    @pytest.mark.parametrize("dtype", ["float64", "int64"])
    def test_from_spmatrix(self, format, labels, dtype):
        sp_sparse = pytest.importorskip("scipy.sparse")

        sp_dtype = SparseDtype(dtype, np.array(0, dtype=dtype).item())

        mat = sp_sparse.eye(10, format=format, dtype=dtype)
        result = pd.DataFrame.sparse.from_spmatrix(mat, index=labels, columns=labels)
        expected = pd.DataFrame(
            np.eye(10, dtype=dtype), index=labels, columns=labels
        ).astype(sp_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("format", ["csc", "csr", "coo"])
    def test_from_spmatrix_including_explicit_zero(self, format):
        sp_sparse = pytest.importorskip("scipy.sparse")

        mat = sp_sparse.random(10, 2, density=0.5, format=format)
        mat.data[0] = 0
        result = pd.DataFrame.sparse.from_spmatrix(mat)
        dtype = SparseDtype("float64", 0.0)
        expected = pd.DataFrame(mat.todense()).astype(dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "columns",
        [["a", "b"], pd.MultiIndex.from_product([["A"], ["a", "b"]]), ["a", "a"]],
    )
    def test_from_spmatrix_columns(self, columns):
        sp_sparse = pytest.importorskip("scipy.sparse")

        dtype = SparseDtype("float64", 0.0)

        mat = sp_sparse.random(10, 2, density=0.5)
        result = pd.DataFrame.sparse.from_spmatrix(mat, columns=columns)
        expected = pd.DataFrame(mat.toarray(), columns=columns).astype(dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "colnames", [("A", "B"), (1, 2), (1, pd.NA), (0.1, 0.2), ("x", "x"), (0, 0)]
    )
    def test_to_coo(self, colnames):
        sp_sparse = pytest.importorskip("scipy.sparse")

        df = pd.DataFrame(
            {colnames[0]: [0, 1, 0], colnames[1]: [1, 0, 0]}, dtype="Sparse[int64, 0]"
        )
        result = df.sparse.to_coo()
        expected = sp_sparse.coo_matrix(np.asarray(df))
        assert (result != expected).nnz == 0

    @pytest.mark.parametrize("fill_value", [1, np.nan])
    def test_to_coo_nonzero_fill_val_raises(self, fill_value):
        pytest.importorskip("scipy")
        df = pd.DataFrame(
            {
                "A": SparseArray(
                    [fill_value, fill_value, fill_value, 2], fill_value=fill_value
                ),
                "B": SparseArray(
                    [fill_value, 2, fill_value, fill_value], fill_value=fill_value
                ),
            }
        )
        with pytest.raises(ValueError, match="fill value must be 0"):
            df.sparse.to_coo()

    def test_to_coo_midx_categorical(self):
        # GH#50996
        sp_sparse = pytest.importorskip("scipy.sparse")

        midx = pd.MultiIndex.from_arrays(
            [
                pd.CategoricalIndex(list("ab"), name="x"),
                pd.CategoricalIndex([0, 1], name="y"),
            ]
        )

        ser = pd.Series(1, index=midx, dtype="Sparse[int]")
        result = ser.sparse.to_coo(row_levels=["x"], column_levels=["y"])[0]
        expected = sp_sparse.coo_matrix(
            (np.array([1, 1]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2)
        )
        assert (result != expected).nnz == 0

    def test_to_dense(self):
        df = pd.DataFrame(
            {
                "A": SparseArray([1, 0], dtype=SparseDtype("int64", 0)),
                "B": SparseArray([1, 0], dtype=SparseDtype("int64", 1)),
                "C": SparseArray([1.0, 0.0], dtype=SparseDtype("float64", 0.0)),
            },
            index=["b", "a"],
        )
        result = df.sparse.to_dense()
        expected = pd.DataFrame(
            {"A": [1, 0], "B": [1, 0], "C": [1.0, 0.0]}, index=["b", "a"]
        )
        tm.assert_frame_equal(result, expected)

    def test_density(self):
        df = pd.DataFrame(
            {
                "A": SparseArray([1, 0, 2, 1], fill_value=0),
                "B": SparseArray([0, 1, 1, 1], fill_value=0),
            }
        )
        res = df.sparse.density
        expected = 0.75
        assert res == expected

    @pytest.mark.parametrize("dtype", ["int64", "float64"])
    @pytest.mark.parametrize("dense_index", [True, False])
    def test_series_from_coo(self, dtype, dense_index):
        sp_sparse = pytest.importorskip("scipy.sparse")

        A = sp_sparse.eye(3, format="coo", dtype=dtype)
        result = pd.Series.sparse.from_coo(A, dense_index=dense_index)

        index = pd.MultiIndex.from_tuples(
            [
                np.array([0, 0], dtype=np.int32),
                np.array([1, 1], dtype=np.int32),
                np.array([2, 2], dtype=np.int32),
            ],
        )
        expected = pd.Series(SparseArray(np.array([1, 1, 1], dtype=dtype)), index=index)
        if dense_index:
            expected = expected.reindex(pd.MultiIndex.from_product(index.levels))

        tm.assert_series_equal(result, expected)

    def test_series_from_coo_incorrect_format_raises(self):
        # gh-26554
        sp_sparse = pytest.importorskip("scipy.sparse")

        m = sp_sparse.csr_matrix(np.array([[0, 1], [0, 0]]))
        with pytest.raises(
            TypeError, match="Expected coo_matrix. Got csr_matrix instead."
        ):
            pd.Series.sparse.from_coo(m)

    def test_with_column_named_sparse(self):
        # https://github.com/pandas-dev/pandas/issues/30758
        df = pd.DataFrame({"sparse": pd.arrays.SparseArray([1, 2])})
        assert isinstance(df.sparse, pd.core.arrays.sparse.accessor.SparseFrameAccessor)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_isin.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray


class TestSeriesIsIn:
    def test_isin(self):
        s = Series(["A", "B", "C", "a", "B", "B", "A", "C"])

        result = s.isin(["A", "C"])
        expected = Series([True, False, True, False, False, False, True, True])
        tm.assert_series_equal(result, expected)

        # GH#16012
        # This specific issue has to have a series over 1e6 in len, but the
        # comparison array (in_list) must be large enough so that numpy doesn't
        # do a manual masking trick that will avoid this issue altogether
        s = Series(list("abcdefghijk" * 10**5))
        # If numpy doesn't do the manual comparison/mask, these
        # unorderable mixed types are what cause the exception in numpy
        in_list = [-1, "a", "b", "G", "Y", "Z", "E", "K", "E", "S", "I", "R", "R"] * 6

        assert s.isin(in_list).sum() == 200000

    def test_isin_with_string_scalar(self):
        # GH#4763
        s = Series(["A", "B", "C", "a", "B", "B", "A", "C"])
        msg = (
            r"only list-like objects are allowed to be passed to isin\(\), "
            r"you passed a `str`"
        )
        with pytest.raises(TypeError, match=msg):
            s.isin("a")

        s = Series(["aaa", "b", "c"])
        with pytest.raises(TypeError, match=msg):
            s.isin("aaa")

    def test_isin_datetimelike_mismatched_reso(self):
        expected = Series([True, True, False, False, False])

        ser = Series(date_range("jan-01-2013", "jan-05-2013"))

        # fails on dtype conversion in the first place
        day_values = np.asarray(ser[0:2].values).astype("datetime64[D]")
        result = ser.isin(day_values)
        tm.assert_series_equal(result, expected)

        dta = ser[:2]._values.astype("M8[s]")
        result = ser.isin(dta)
        tm.assert_series_equal(result, expected)

    def test_isin_datetimelike_mismatched_reso_list(self):
        expected = Series([True, True, False, False, False])

        ser = Series(date_range("jan-01-2013", "jan-05-2013"))

        dta = ser[:2]._values.astype("M8[s]")
        result = ser.isin(list(dta))
        tm.assert_series_equal(result, expected)

    def test_isin_with_i8(self):
        # GH#5021

        expected = Series([True, True, False, False, False])
        expected2 = Series([False, True, False, False, False])

        # datetime64[ns]
        s = Series(date_range("jan-01-2013", "jan-05-2013"))

        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

        result = s.isin(s[0:2].values)
        tm.assert_series_equal(result, expected)

        result = s.isin([s[1]])
        tm.assert_series_equal(result, expected2)

        result = s.isin([np.datetime64(s[1])])
        tm.assert_series_equal(result, expected2)

        result = s.isin(set(s[0:2]))
        tm.assert_series_equal(result, expected)

        # timedelta64[ns]
        s = Series(pd.to_timedelta(range(5), unit="d"))
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        # see GH#16991
        s = Series(["a", "b"])
        expected = Series([False, False])

        result = s.isin(empty)
        tm.assert_series_equal(expected, result)

    def test_isin_read_only(self):
        # https://github.com/pandas-dev/pandas/issues/37174
        arr = np.array([1, 2, 3])
        arr.setflags(write=False)
        s = Series([1, 2, 3])
        result = s.isin(arr)
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [object, None])
    def test_isin_dt64_values_vs_ints(self, dtype):
        # GH#36621 dont cast integers to datetimes for isin
        dti = date_range("2013-01-01", "2013-01-05")
        ser = Series(dti)

        comps = np.asarray([1356998400000000000], dtype=dtype)

        res = dti.isin(comps)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(comps)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, comps)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_tzawareness_mismatch(self):
        dti = date_range("2013-01-01", "2013-01-05")
        ser = Series(dti)

        other = dti.tz_localize("UTC")

        res = dti.isin(other)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_period_freq_mismatch(self):
        dti = date_range("2013-01-01", "2013-01-05")
        pi = dti.to_period("M")
        ser = Series(pi)

        # We construct another PeriodIndex with the same i8 values
        #  but different dtype
        dtype = dti.to_period("Y").dtype
        other = PeriodArray._simple_new(pi.asi8, dtype=dtype)

        res = pi.isin(other)
        expected = np.array([False] * len(pi), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize("values", [[-9.0, 0.0], [-9, 0]])
    def test_isin_float_in_int_series(self, values):
        # GH#19356 GH#21804
        ser = Series(values)
        result = ser.isin([-9, -0.5])
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
    @pytest.mark.parametrize(
        "data,values,expected",
        [
            ([0, 1, 0], [1], [False, True, False]),
            ([0, 1, 0], [1, pd.NA], [False, True, False]),
            ([0, pd.NA, 0], [1, 0], [True, False, True]),
            ([0, 1, pd.NA], [1, pd.NA], [False, True, True]),
            ([0, 1, pd.NA], [1, np.nan], [False, True, False]),
            ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None], [False, False, False]),
        ],
    )
    def test_isin_masked_types(self, dtype, data, values, expected):
        # GH#42405
        ser = Series(data, dtype=dtype)

        result = ser.isin(values)
        expected = Series(expected, dtype="boolean")

        tm.assert_series_equal(result, expected)


def test_isin_large_series_mixed_dtypes_and_nan(monkeypatch):
    # https://github.com/pandas-dev/pandas/issues/37094
    # combination of object dtype for the values
    # and > _MINIMUM_COMP_ARR_LEN elements
    min_isin_comp = 5
    ser = Series([1, 2, np.nan] * min_isin_comp)
    with monkeypatch.context() as m:
        m.setattr(algorithms, "_MINIMUM_COMP_ARR_LEN", min_isin_comp)
        result = ser.isin({"foo", "bar"})
    expected = Series([False] * 3 * min_isin_comp)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "array,expected",
    [
        (
            [0, 1j, 1j, 1, 1 + 1j, 1 + 2j, 1 + 1j],
            Series([False, True, True, False, True, True, True], dtype=bool),
        )
    ],
)
def test_isin_complex_numbers(array, expected):
    # GH 17927
    result = Series(array).isin([1j, 1 + 1j, 1 + 2j])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data,is_in",
    [([1, [2]], [1]), (["simple str", [{"values": 3}]], ["simple str"])],
)
def test_isin_filtering_with_mixed_object_types(data, is_in):
    # GH 20883

    ser = Series(data)
    result = ser.isin(is_in)
    expected = Series([True, False])

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("data", [[1, 2, 3], [1.0, 2.0, 3.0]])
@pytest.mark.parametrize("isin", [[1, 2], [1.0, 2.0]])
def test_isin_filtering_on_iterable(data, isin):
    # GH 50234

    ser = Series(data)
    result = ser.isin(i for i in isin)
    expected_result = Series([True, True, False])

    tm.assert_series_equal(result, expected_result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_timezones.py ===
"""
Tests for DatetimeIndex timezone-related methods
"""
from datetime import (
    datetime,
    timedelta,
    timezone,
    tzinfo,
)

from dateutil.tz import gettz
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import (
    conversion,
    timezones,
)

import pandas as pd
from pandas import (
    DatetimeIndex,
    Timestamp,
    bdate_range,
    date_range,
    isna,
    to_datetime,
)
import pandas._testing as tm


class FixedOffset(tzinfo):
    """Fixed offset in minutes east from UTC."""

    def __init__(self, offset, name) -> None:
        self.__offset = timedelta(minutes=offset)
        self.__name = name

    def utcoffset(self, dt):
        return self.__offset

    def tzname(self, dt):
        return self.__name

    def dst(self, dt):
        return timedelta(0)


fixed_off_no_name = FixedOffset(-330, None)


class TestDatetimeIndexTimezones:
    # -------------------------------------------------------------
    # Unsorted

    def test_dti_drop_dont_lose_tz(self):
        # GH#2621
        ind = date_range("2012-12-01", periods=10, tz="utc")
        ind = ind.drop(ind[-1])

        assert ind.tz is not None

    def test_dti_tz_conversion_freq(self, tz_naive_fixture):
        # GH25241
        t3 = DatetimeIndex(["2019-01-01 10:00"], freq="h")
        assert t3.tz_localize(tz=tz_naive_fixture).freq == t3.freq
        t4 = DatetimeIndex(["2019-01-02 12:00"], tz="UTC", freq="min")
        assert t4.tz_convert(tz="UTC").freq == t4.freq

    def test_drop_dst_boundary(self):
        # see gh-18031
        tz = "Europe/Brussels"
        freq = "15min"

        start = Timestamp("201710290100", tz=tz)
        end = Timestamp("201710290300", tz=tz)
        index = date_range(start=start, end=end, freq=freq)

        expected = DatetimeIndex(
            [
                "201710290115",
                "201710290130",
                "201710290145",
                "201710290200",
                "201710290215",
                "201710290230",
                "201710290245",
                "201710290200",
                "201710290215",
                "201710290230",
                "201710290245",
                "201710290300",
            ],
            dtype="M8[ns, Europe/Brussels]",
            freq=freq,
            ambiguous=[
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        )
        result = index.drop(index[0])
        tm.assert_index_equal(result, expected)

    def test_date_range_localize(self, unit):
        rng = date_range(
            "3/11/2012 03:00", periods=15, freq="h", tz="US/Eastern", unit=unit
        )
        rng2 = DatetimeIndex(
            ["3/11/2012 03:00", "3/11/2012 04:00"], dtype=f"M8[{unit}, US/Eastern]"
        )
        rng3 = date_range("3/11/2012 03:00", periods=15, freq="h", unit=unit)
        rng3 = rng3.tz_localize("US/Eastern")

        tm.assert_index_equal(rng._with_freq(None), rng3)

        # DST transition time
        val = rng[0]
        exp = Timestamp("3/11/2012 03:00", tz="US/Eastern")

        assert val.hour == 3
        assert exp.hour == 3
        assert val == exp  # same UTC value
        tm.assert_index_equal(rng[:2], rng2)

    def test_date_range_localize2(self, unit):
        # Right before the DST transition
        rng = date_range(
            "3/11/2012 00:00", periods=2, freq="h", tz="US/Eastern", unit=unit
        )
        rng2 = DatetimeIndex(
            ["3/11/2012 00:00", "3/11/2012 01:00"],
            dtype=f"M8[{unit}, US/Eastern]",
            freq="h",
        )
        tm.assert_index_equal(rng, rng2)
        exp = Timestamp("3/11/2012 00:00", tz="US/Eastern")
        assert exp.hour == 0
        assert rng[0] == exp
        exp = Timestamp("3/11/2012 01:00", tz="US/Eastern")
        assert exp.hour == 1
        assert rng[1] == exp

        rng = date_range(
            "3/11/2012 00:00", periods=10, freq="h", tz="US/Eastern", unit=unit
        )
        assert rng[2].hour == 3

    def test_timestamp_equality_different_timezones(self):
        utc_range = date_range("1/1/2000", periods=20, tz="UTC")
        eastern_range = utc_range.tz_convert("US/Eastern")
        berlin_range = utc_range.tz_convert("Europe/Berlin")

        for a, b, c in zip(utc_range, eastern_range, berlin_range):
            assert a == b
            assert b == c
            assert a == c

        assert (utc_range == eastern_range).all()
        assert (utc_range == berlin_range).all()
        assert (berlin_range == eastern_range).all()

    def test_dti_equals_with_tz(self):
        left = date_range("1/1/2011", periods=100, freq="h", tz="utc")
        right = date_range("1/1/2011", periods=100, freq="h", tz="US/Eastern")

        assert not left.equals(right)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_nat(self, tzstr):
        idx = DatetimeIndex([Timestamp("2013-1-1", tz=tzstr), pd.NaT])

        assert isna(idx[1])
        assert idx[0].tzinfo is not None

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_utc_box_timestamp_and_localize(self, tzstr):
        tz = timezones.maybe_get_tz(tzstr)

        rng = date_range("3/11/2012", "3/12/2012", freq="h", tz="utc")
        rng_eastern = rng.tz_convert(tzstr)

        expected = rng[-1].astimezone(tz)

        stamp = rng_eastern[-1]
        assert stamp == expected
        assert stamp.tzinfo == expected.tzinfo

        # right tzinfo
        rng = date_range("3/13/2012", "3/14/2012", freq="h", tz="utc")
        rng_eastern = rng.tz_convert(tzstr)
        # test not valid for dateutil timezones.
        # assert 'EDT' in repr(rng_eastern[0].tzinfo)
        assert "EDT" in repr(rng_eastern[0].tzinfo) or "tzfile" in repr(
            rng_eastern[0].tzinfo
        )

    @pytest.mark.parametrize("tz", [pytz.timezone("US/Central"), gettz("US/Central")])
    def test_with_tz(self, tz):
        # just want it to work
        start = datetime(2011, 3, 12, tzinfo=pytz.utc)
        dr = bdate_range(start, periods=50, freq=pd.offsets.Hour())
        assert dr.tz is pytz.utc

        # DateRange with naive datetimes
        dr = bdate_range("1/1/2005", "1/1/2009", tz=pytz.utc)
        dr = bdate_range("1/1/2005", "1/1/2009", tz=tz)

        # normalized
        central = dr.tz_convert(tz)
        assert central.tz is tz
        naive = central[0].to_pydatetime().replace(tzinfo=None)
        comp = conversion.localize_pydatetime(naive, tz).tzinfo
        assert central[0].tz is comp

        # compare vs a localized tz
        naive = dr[0].to_pydatetime().replace(tzinfo=None)
        comp = conversion.localize_pydatetime(naive, tz).tzinfo
        assert central[0].tz is comp

        # datetimes with tzinfo set
        dr = bdate_range(
            datetime(2005, 1, 1, tzinfo=pytz.utc), datetime(2009, 1, 1, tzinfo=pytz.utc)
        )
        msg = "Start and end cannot both be tz-aware with different timezones"
        with pytest.raises(Exception, match=msg):
            bdate_range(datetime(2005, 1, 1, tzinfo=pytz.utc), "1/1/2009", tz=tz)

    @pytest.mark.parametrize("tz", [pytz.timezone("US/Eastern"), gettz("US/Eastern")])
    def test_dti_convert_tz_aware_datetime_datetime(self, tz):
        # GH#1581
        dates = [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]

        dates_aware = [conversion.localize_pydatetime(x, tz) for x in dates]
        result = DatetimeIndex(dates_aware).as_unit("ns")
        assert timezones.tz_compare(result.tz, tz)

        converted = to_datetime(dates_aware, utc=True).as_unit("ns")
        ex_vals = np.array([Timestamp(x).as_unit("ns")._value for x in dates_aware])
        tm.assert_numpy_array_equal(converted.asi8, ex_vals)
        assert converted.tz is timezone.utc

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_nlargest.py ===
"""
Note: for naming purposes, most tests are title with as e.g. "test_nlargest_foo"
but are implicitly also testing nsmallest_foo.
"""
from string import ascii_lowercase

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version


@pytest.fixture
def df_duplicates():
    return pd.DataFrame(
        {"a": [1, 2, 3, 4, 4], "b": [1, 1, 1, 1, 1], "c": [0, 1, 2, 5, 4]},
        index=[0, 0, 1, 1, 1],
    )


@pytest.fixture
def df_strings():
    return pd.DataFrame(
        {
            "a": np.random.default_rng(2).permutation(10),
            "b": list(ascii_lowercase[:10]),
            "c": np.random.default_rng(2).permutation(10).astype("float64"),
        }
    )


@pytest.fixture
def df_main_dtypes():
    return pd.DataFrame(
        {
            "group": [1, 1, 2],
            "int": [1, 2, 3],
            "float": [4.0, 5.0, 6.0],
            "string": list("abc"),
            "category_string": pd.Series(list("abc")).astype("category"),
            "category_int": [7, 8, 9],
            "datetime": pd.date_range("20130101", periods=3),
            "datetimetz": pd.date_range("20130101", periods=3, tz="US/Eastern"),
            "timedelta": pd.timedelta_range("1 s", periods=3, freq="s"),
        },
        columns=[
            "group",
            "int",
            "float",
            "string",
            "category_string",
            "category_int",
            "datetime",
            "datetimetz",
            "timedelta",
        ],
    )


class TestNLargestNSmallest:
    # ----------------------------------------------------------------------
    # Top / bottom
    @pytest.mark.parametrize(
        "order",
        [
            ["a"],
            ["c"],
            ["a", "b"],
            ["a", "c"],
            ["b", "a"],
            ["b", "c"],
            ["a", "b", "c"],
            ["c", "a", "b"],
            ["c", "b", "a"],
            ["b", "c", "a"],
            ["b", "a", "c"],
            # dups!
            ["b", "c", "c"],
        ],
    )
    @pytest.mark.parametrize("n", range(1, 11))
    def test_nlargest_n(self, df_strings, nselect_method, n, order):
        # GH#10393
        df = df_strings
        if "b" in order:
            error_msg = (
                f"Column 'b' has dtype (object|str), "
                f"cannot use method '{nselect_method}' with this dtype"
            )
            with pytest.raises(TypeError, match=error_msg):
                getattr(df, nselect_method)(n, order)
        else:
            ascending = nselect_method == "nsmallest"
            result = getattr(df, nselect_method)(n, order)
            expected = df.sort_values(order, ascending=ascending).head(n)
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "columns", [["group", "category_string"], ["group", "string"]]
    )
    def test_nlargest_error(self, df_main_dtypes, nselect_method, columns):
        df = df_main_dtypes
        col = columns[1]
        error_msg = (
            f"Column '{col}' has dtype {df[col].dtype}, "
            f"cannot use method '{nselect_method}' with this dtype"
        )
        # escape some characters that may be in the repr
        error_msg = (
            error_msg.replace("(", "\\(")
            .replace(")", "\\)")
            .replace("[", "\\[")
            .replace("]", "\\]")
        )
        with pytest.raises(TypeError, match=error_msg):
            getattr(df, nselect_method)(2, columns)

    def test_nlargest_all_dtypes(self, df_main_dtypes):
        df = df_main_dtypes
        df.nsmallest(2, list(set(df) - {"category_string", "string"}))
        df.nlargest(2, list(set(df) - {"category_string", "string"}))

    def test_nlargest_duplicates_on_starter_columns(self):
        # regression test for GH#22752

        df = pd.DataFrame({"a": [2, 2, 2, 1, 1, 1], "b": [1, 2, 3, 3, 2, 1]})

        result = df.nlargest(4, columns=["a", "b"])
        expected = pd.DataFrame(
            {"a": [2, 2, 2, 1], "b": [3, 2, 1, 3]}, index=[2, 1, 0, 3]
        )
        tm.assert_frame_equal(result, expected)

        result = df.nsmallest(4, columns=["a", "b"])
        expected = pd.DataFrame(
            {"a": [1, 1, 1, 2], "b": [1, 2, 3, 1]}, index=[5, 4, 3, 0]
        )
        tm.assert_frame_equal(result, expected)

    def test_nlargest_n_identical_values(self):
        # GH#15297
        df = pd.DataFrame({"a": [1] * 5, "b": [1, 2, 3, 4, 5]})

        result = df.nlargest(3, "a")
        expected = pd.DataFrame({"a": [1] * 3, "b": [1, 2, 3]}, index=[0, 1, 2])
        tm.assert_frame_equal(result, expected)

        result = df.nsmallest(3, "a")
        expected = pd.DataFrame({"a": [1] * 3, "b": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "order",
        [["a", "b", "c"], ["c", "b", "a"], ["a"], ["b"], ["a", "b"], ["c", "b"]],
    )
    @pytest.mark.parametrize("n", range(1, 6))
    def test_nlargest_n_duplicate_index(self, df_duplicates, n, order, request):
        # GH#13412

        df = df_duplicates
        result = df.nsmallest(n, order)
        expected = df.sort_values(order).head(n)
        tm.assert_frame_equal(result, expected)

        result = df.nlargest(n, order)
        expected = df.sort_values(order, ascending=False).head(n)
        if Version(np.__version__) >= Version("1.25") and (
            (order == ["a"] and n in (1, 2, 3, 4)) or (order == ["a", "b"]) and n == 5
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "pandas default unstable sorting of duplicates"
                        "issue with numpy>=1.25 with AVX instructions"
                    ),
                    strict=False,
                )
            )
        tm.assert_frame_equal(result, expected)

    def test_nlargest_duplicate_keep_all_ties(self):
        # GH#16818
        df = pd.DataFrame(
            {"a": [5, 4, 4, 2, 3, 3, 3, 3], "b": [10, 9, 8, 7, 5, 50, 10, 20]}
        )
        result = df.nlargest(4, "a", keep="all")
        expected = pd.DataFrame(
            {
                "a": {0: 5, 1: 4, 2: 4, 4: 3, 5: 3, 6: 3, 7: 3},
                "b": {0: 10, 1: 9, 2: 8, 4: 5, 5: 50, 6: 10, 7: 20},
            }
        )
        tm.assert_frame_equal(result, expected)

        result = df.nsmallest(2, "a", keep="all")
        expected = pd.DataFrame(
            {
                "a": {3: 2, 4: 3, 5: 3, 6: 3, 7: 3},
                "b": {3: 7, 4: 5, 5: 50, 6: 10, 7: 20},
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_nlargest_multiindex_column_lookup(self):
        # Check whether tuples are correctly treated as multi-level lookups.
        # GH#23033
        df = pd.DataFrame(
            columns=pd.MultiIndex.from_product([["x"], ["a", "b"]]),
            data=[[0.33, 0.13], [0.86, 0.25], [0.25, 0.70], [0.85, 0.91]],
        )

        # nsmallest
        result = df.nsmallest(3, ("x", "a"))
        expected = df.iloc[[2, 0, 3]]
        tm.assert_frame_equal(result, expected)

        # nlargest
        result = df.nlargest(3, ("x", "b"))
        expected = df.iloc[[3, 2, 1]]
        tm.assert_frame_equal(result, expected)

    def test_nlargest_nan(self):
        # GH#43060
        df = pd.DataFrame([np.nan, np.nan, 0, 1, 2, 3])
        result = df.nlargest(5, 0)
        expected = df.sort_values(0, ascending=False).head(5)
        tm.assert_frame_equal(result, expected)

    def test_nsmallest_nan_after_n_element(self):
        # GH#46589
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, None, 7],
                "b": [7, 6, 5, 4, 3, 2, 1],
                "c": [1, 1, 2, 2, 3, 3, 3],
            },
            index=range(7),
        )
        result = df.nsmallest(5, columns=["a", "b"])
        expected = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [7, 6, 5, 4, 3],
                "c": [1, 1, 2, 2, 3],
            },
            index=range(5),
        ).astype({"a": "float"})
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_formats.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)
import pandas._testing as tm


def test_format(idx):
    msg = "MultiIndex.format is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        idx.format()
        idx[:0].format()


def test_format_integer_names():
    index = MultiIndex(
        levels=[[0, 1], [0, 1]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]], names=[0, 1]
    )
    msg = "MultiIndex.format is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        index.format(names=True)


def test_format_sparse_config(idx):
    # GH1538
    msg = "MultiIndex.format is deprecated"
    with pd.option_context("display.multi_sparse", False):
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = idx.format()
    assert result[1] == "foo  two"


def test_format_sparse_display():
    index = MultiIndex(
        levels=[[0, 1], [0, 1], [0, 1], [0]],
        codes=[
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    )
    msg = "MultiIndex.format is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = index.format()
    assert result[3] == "1  0  0  0"


def test_repr_with_unicode_data():
    with pd.option_context("display.encoding", "UTF-8"):
        d = {"a": ["\u05d0", 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
        index = pd.DataFrame(d).set_index(["a", "b"]).index
        assert "\\" not in repr(index)  # we don't want unicode-escaped


def test_repr_roundtrip_raises():
    mi = MultiIndex.from_product([list("ab"), range(3)], names=["first", "second"])
    msg = "Must pass both levels and codes"
    with pytest.raises(TypeError, match=msg):
        eval(repr(mi))


def test_unicode_string_with_unicode():
    d = {"a": ["\u05d0", 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    idx = pd.DataFrame(d).set_index(["a", "b"]).index
    str(idx)


def test_repr_max_seq_item_setting(idx):
    # GH10182
    idx = idx.repeat(50)
    with pd.option_context("display.max_seq_items", None):
        repr(idx)
        assert "..." not in str(idx)


class TestRepr:
    def test_unicode_repr_issues(self):
        levels = [Index(["a/\u03c3", "b/\u03c3", "c/\u03c3"]), Index([0, 1])]
        codes = [np.arange(3).repeat(2), np.tile(np.arange(2), 3)]
        index = MultiIndex(levels=levels, codes=codes)

        repr(index.levels)
        repr(index.get_level_values(1))

    def test_repr_max_seq_items_equal_to_n(self, idx):
        # display.max_seq_items == n
        with pd.option_context("display.max_seq_items", 6):
            result = idx.__repr__()
            expected = """\
MultiIndex([('foo', 'one'),
            ('foo', 'two'),
            ('bar', 'one'),
            ('baz', 'two'),
            ('qux', 'one'),
            ('qux', 'two')],
           names=['first', 'second'])"""
            assert result == expected

    def test_repr(self, idx):
        result = idx[:1].__repr__()
        expected = """\
MultiIndex([('foo', 'one')],
           names=['first', 'second'])"""
        assert result == expected

        result = idx.__repr__()
        expected = """\
MultiIndex([('foo', 'one'),
            ('foo', 'two'),
            ('bar', 'one'),
            ('baz', 'two'),
            ('qux', 'one'),
            ('qux', 'two')],
           names=['first', 'second'])"""
        assert result == expected

        with pd.option_context("display.max_seq_items", 5):
            result = idx.__repr__()
            expected = """\
MultiIndex([('foo', 'one'),
            ('foo', 'two'),
            ...
            ('qux', 'one'),
            ('qux', 'two')],
           names=['first', 'second'], length=6)"""
            assert result == expected

        # display.max_seq_items == 1
        with pd.option_context("display.max_seq_items", 1):
            result = idx.__repr__()
            expected = """\
MultiIndex([...
            ('qux', 'two')],
           names=['first', ...], length=6)"""
            assert result == expected

    def test_rjust(self):
        n = 1000
        ci = pd.CategoricalIndex(list("a" * n) + (["abc"] * n))
        dti = pd.date_range("2000-01-01", freq="s", periods=n * 2)
        mi = MultiIndex.from_arrays([ci, ci.codes + 9, dti], names=["a", "b", "dti"])
        result = mi[:1].__repr__()
        expected = """\
MultiIndex([('a', 9, '2000-01-01 00:00:00')],
           names=['a', 'b', 'dti'])"""
        assert result == expected

        result = mi[::500].__repr__()
        expected = """\
MultiIndex([(  'a',  9, '2000-01-01 00:00:00'),
            (  'a',  9, '2000-01-01 00:08:20'),
            ('abc', 10, '2000-01-01 00:16:40'),
            ('abc', 10, '2000-01-01 00:25:00')],
           names=['a', 'b', 'dti'])"""
        assert result == expected

        result = mi.__repr__()
        expected = """\
MultiIndex([(  'a',  9, '2000-01-01 00:00:00'),
            (  'a',  9, '2000-01-01 00:00:01'),
            (  'a',  9, '2000-01-01 00:00:02'),
            (  'a',  9, '2000-01-01 00:00:03'),
            (  'a',  9, '2000-01-01 00:00:04'),
            (  'a',  9, '2000-01-01 00:00:05'),
            (  'a',  9, '2000-01-01 00:00:06'),
            (  'a',  9, '2000-01-01 00:00:07'),
            (  'a',  9, '2000-01-01 00:00:08'),
            (  'a',  9, '2000-01-01 00:00:09'),
            ...
            ('abc', 10, '2000-01-01 00:33:10'),
            ('abc', 10, '2000-01-01 00:33:11'),
            ('abc', 10, '2000-01-01 00:33:12'),
            ('abc', 10, '2000-01-01 00:33:13'),
            ('abc', 10, '2000-01-01 00:33:14'),
            ('abc', 10, '2000-01-01 00:33:15'),
            ('abc', 10, '2000-01-01 00:33:16'),
            ('abc', 10, '2000-01-01 00:33:17'),
            ('abc', 10, '2000-01-01 00:33:18'),
            ('abc', 10, '2000-01-01 00:33:19')],
           names=['a', 'b', 'dti'], length=2000)"""
        assert result == expected

    def test_tuple_width(self):
        n = 1000
        ci = pd.CategoricalIndex(list("a" * n) + (["abc"] * n))
        dti = pd.date_range("2000-01-01", freq="s", periods=n * 2)
        levels = [ci, ci.codes + 9, dti, dti, dti]
        names = ["a", "b", "dti_1", "dti_2", "dti_3"]
        mi = MultiIndex.from_arrays(levels, names=names)
        result = mi[:1].__repr__()
        expected = """MultiIndex([('a', 9, '2000-01-01 00:00:00', '2000-01-01 00:00:00', ...)],
           names=['a', 'b', 'dti_1', 'dti_2', 'dti_3'])"""  # noqa: E501
        assert result == expected

        result = mi[:10].__repr__()
        expected = """\
MultiIndex([('a', 9, '2000-01-01 00:00:00', '2000-01-01 00:00:00', ...),
            ('a', 9, '2000-01-01 00:00:01', '2000-01-01 00:00:01', ...),
            ('a', 9, '2000-01-01 00:00:02', '2000-01-01 00:00:02', ...),
            ('a', 9, '2000-01-01 00:00:03', '2000-01-01 00:00:03', ...),
            ('a', 9, '2000-01-01 00:00:04', '2000-01-01 00:00:04', ...),
            ('a', 9, '2000-01-01 00:00:05', '2000-01-01 00:00:05', ...),
            ('a', 9, '2000-01-01 00:00:06', '2000-01-01 00:00:06', ...),
            ('a', 9, '2000-01-01 00:00:07', '2000-01-01 00:00:07', ...),
            ('a', 9, '2000-01-01 00:00:08', '2000-01-01 00:00:08', ...),
            ('a', 9, '2000-01-01 00:00:09', '2000-01-01 00:00:09', ...)],
           names=['a', 'b', 'dti_1', 'dti_2', 'dti_3'])"""
        assert result == expected

        result = mi.__repr__()
        expected = """\
MultiIndex([(  'a',  9, '2000-01-01 00:00:00', '2000-01-01 00:00:00', ...),
            (  'a',  9, '2000-01-01 00:00:01', '2000-01-01 00:00:01', ...),
            (  'a',  9, '2000-01-01 00:00:02', '2000-01-01 00:00:02', ...),
            (  'a',  9, '2000-01-01 00:00:03', '2000-01-01 00:00:03', ...),
            (  'a',  9, '2000-01-01 00:00:04', '2000-01-01 00:00:04', ...),
            (  'a',  9, '2000-01-01 00:00:05', '2000-01-01 00:00:05', ...),
            (  'a',  9, '2000-01-01 00:00:06', '2000-01-01 00:00:06', ...),
            (  'a',  9, '2000-01-01 00:00:07', '2000-01-01 00:00:07', ...),
            (  'a',  9, '2000-01-01 00:00:08', '2000-01-01 00:00:08', ...),
            (  'a',  9, '2000-01-01 00:00:09', '2000-01-01 00:00:09', ...),
            ...
            ('abc', 10, '2000-01-01 00:33:10', '2000-01-01 00:33:10', ...),
            ('abc', 10, '2000-01-01 00:33:11', '2000-01-01 00:33:11', ...),
            ('abc', 10, '2000-01-01 00:33:12', '2000-01-01 00:33:12', ...),
            ('abc', 10, '2000-01-01 00:33:13', '2000-01-01 00:33:13', ...),
            ('abc', 10, '2000-01-01 00:33:14', '2000-01-01 00:33:14', ...),
            ('abc', 10, '2000-01-01 00:33:15', '2000-01-01 00:33:15', ...),
            ('abc', 10, '2000-01-01 00:33:16', '2000-01-01 00:33:16', ...),
            ('abc', 10, '2000-01-01 00:33:17', '2000-01-01 00:33:17', ...),
            ('abc', 10, '2000-01-01 00:33:18', '2000-01-01 00:33:18', ...),
            ('abc', 10, '2000-01-01 00:33:19', '2000-01-01 00:33:19', ...)],
           names=['a', 'b', 'dti_1', 'dti_2', 'dti_3'], length=2000)"""
        assert result == expected

    def test_multiindex_long_element(self):
        # Non-regression test towards GH#52960
        data = MultiIndex.from_tuples([("c" * 62,)])

        expected = (
            "MultiIndex([('cccccccccccccccccccccccccccccccccccccccc"
            "cccccccccccccccccccccc',)],\n           )"
        )
        assert str(data) == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_jointsmethod.py ===
from sympy.core.function import expand
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.mechanics import (
    PinJoint, JointsMethod, RigidBody, Particle, Body, KanesMethod,
    PrismaticJoint, LagrangesMethod, inertia)
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy import zeros
from sympy.utilities.lambdify import lambdify
from sympy.solvers.solvers import solve


t = dynamicsymbols._t # type: ignore


def test_jointsmethod():
    with warns_deprecated_sympy():
        P = Body('P')
        C = Body('C')
    Pin = PinJoint('P1', P, C)
    C_ixx, g = symbols('C_ixx g')
    q, u = dynamicsymbols('q_P1, u_P1')
    P.apply_force(g*P.y)
    with warns_deprecated_sympy():
        method = JointsMethod(P, Pin)
    assert method.frame == P.frame
    assert method.bodies == [C, P]
    assert method.loads == [(P.masscenter, g*P.frame.y)]
    assert method.q == Matrix([q])
    assert method.u == Matrix([u])
    assert method.kdes == Matrix([u - q.diff()])
    soln = method.form_eoms()
    assert soln == Matrix([[-C_ixx*u.diff()]])
    assert method.forcing_full == Matrix([[u], [0]])
    assert method.mass_matrix_full == Matrix([[1, 0], [0, C_ixx]])
    assert isinstance(method.method, KanesMethod)


def test_rigid_body_particle_compatibility():
    l, m, g = symbols('l m g')
    C = RigidBody('C')
    b = Particle('b', mass=m)
    b_frame = ReferenceFrame('b_frame')
    q, u = dynamicsymbols('q u')
    P = PinJoint('P', C, b, coordinates=q, speeds=u, child_interframe=b_frame,
                 child_point=-l * b_frame.x, joint_axis=C.z)
    with warns_deprecated_sympy():
        method = JointsMethod(C, P)
    method.loads.append((b.masscenter, m * g * C.x))
    method.form_eoms()
    rhs = method.rhs()
    assert rhs[1] == -g*sin(q)/l


def test_jointmethod_duplicate_coordinates_speeds():
    with warns_deprecated_sympy():
        P = Body('P')
        C = Body('C')
        T = Body('T')
    q, u = dynamicsymbols('q u')
    P1 = PinJoint('P1', P, C, q)
    P2 = PrismaticJoint('P2', C, T, q)
    with warns_deprecated_sympy():
        raises(ValueError, lambda: JointsMethod(P, P1, P2))

    P1 = PinJoint('P1', P, C, speeds=u)
    P2 = PrismaticJoint('P2', C, T, speeds=u)
    with warns_deprecated_sympy():
        raises(ValueError, lambda: JointsMethod(P, P1, P2))

    P1 = PinJoint('P1', P, C, q, u)
    P2 = PrismaticJoint('P2', C, T, q, u)
    with warns_deprecated_sympy():
        raises(ValueError, lambda: JointsMethod(P, P1, P2))

def test_complete_simple_double_pendulum():
    q1, q2 = dynamicsymbols('q1 q2')
    u1, u2 = dynamicsymbols('u1 u2')
    m, l, g = symbols('m l g')
    with warns_deprecated_sympy():
        C = Body('C')  # ceiling
        PartP = Body('P', mass=m)
        PartR = Body('R', mass=m)
    J1 = PinJoint('J1', C, PartP, speeds=u1, coordinates=q1,
                  child_point=-l*PartP.x, joint_axis=C.z)
    J2 = PinJoint('J2', PartP, PartR, speeds=u2, coordinates=q2,
                  child_point=-l*PartR.x, joint_axis=PartP.z)

    PartP.apply_force(m*g*C.x)
    PartR.apply_force(m*g*C.x)

    with warns_deprecated_sympy():
        method = JointsMethod(C, J1, J2)
    method.form_eoms()

    assert expand(method.mass_matrix_full) == Matrix([[1, 0, 0, 0],
                                                      [0, 1, 0, 0],
                                                      [0, 0, 2*l**2*m*cos(q2) + 3*l**2*m, l**2*m*cos(q2) + l**2*m],
                                                      [0, 0, l**2*m*cos(q2) + l**2*m, l**2*m]])
    assert trigsimp(method.forcing_full) == trigsimp(Matrix([[u1], [u2], [-g*l*m*(sin(q1 + q2) + sin(q1)) -
                                           g*l*m*sin(q1) + l**2*m*(2*u1 + u2)*u2*sin(q2)],
                                          [-g*l*m*sin(q1 + q2) - l**2*m*u1**2*sin(q2)]]))

def test_two_dof_joints():
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    m, c1, c2, k1, k2 = symbols('m c1 c2 k1 k2')
    with warns_deprecated_sympy():
        W = Body('W')
        B1 = Body('B1', mass=m)
        B2 = Body('B2', mass=m)
    J1 = PrismaticJoint('J1', W, B1, coordinates=q1, speeds=u1)
    J2 = PrismaticJoint('J2', B1, B2, coordinates=q2, speeds=u2)
    W.apply_force(k1*q1*W.x, reaction_body=B1)
    W.apply_force(c1*u1*W.x, reaction_body=B1)
    B1.apply_force(k2*q2*W.x, reaction_body=B2)
    B1.apply_force(c2*u2*W.x, reaction_body=B2)
    with warns_deprecated_sympy():
        method = JointsMethod(W, J1, J2)
    method.form_eoms()
    MM = method.mass_matrix
    forcing = method.forcing
    rhs = MM.LUsolve(forcing)
    assert expand(rhs[0]) == expand((-k1 * q1 - c1 * u1 + k2 * q2 + c2 * u2)/m)
    assert expand(rhs[1]) == expand((k1 * q1 + c1 * u1 - 2 * k2 * q2 - 2 *
                                    c2 * u2) / m)

def test_simple_pedulum():
    l, m, g = symbols('l m g')
    with warns_deprecated_sympy():
        C = Body('C')
        b = Body('b', mass=m)
    q = dynamicsymbols('q')
    P = PinJoint('P', C, b, speeds=q.diff(t), coordinates=q,
                 child_point=-l * b.x, joint_axis=C.z)
    b.potential_energy = - m * g * l * cos(q)
    with warns_deprecated_sympy():
        method = JointsMethod(C, P)
    method.form_eoms(LagrangesMethod)
    rhs = method.rhs()
    assert rhs[1] == -g*sin(q)/l

def test_chaos_pendulum():
    #https://www.pydy.org/examples/chaos_pendulum.html
    mA, mB, lA, lB, IAxx, IBxx, IByy, IBzz, g = symbols('mA, mB, lA, lB, IAxx, IBxx, IByy, IBzz, g')
    theta, phi, omega, alpha = dynamicsymbols('theta phi omega alpha')

    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    with warns_deprecated_sympy():
        rod = Body('rod', mass=mA, frame=A,
                   central_inertia=inertia(A, IAxx, IAxx, 0))
        plate = Body('plate', mass=mB, frame=B,
                     central_inertia=inertia(B, IBxx, IByy, IBzz))
        C = Body('C')
    J1 = PinJoint('J1', C, rod, coordinates=theta, speeds=omega,
                  child_point=-lA * rod.z, joint_axis=C.y)
    J2 = PinJoint('J2', rod, plate, coordinates=phi, speeds=alpha,
                  parent_point=(lB - lA) * rod.z, joint_axis=rod.z)

    rod.apply_force(mA*g*C.z)
    plate.apply_force(mB*g*C.z)

    with warns_deprecated_sympy():
        method = JointsMethod(C, J1, J2)
    method.form_eoms()

    MM = method.mass_matrix
    forcing = method.forcing
    rhs = MM.LUsolve(forcing)
    xd = (-2 * IBxx * alpha * omega * sin(phi) * cos(phi) + 2 * IByy * alpha * omega * sin(phi) *
            cos(phi) - g * lA * mA * sin(theta) - g * lB * mB * sin(theta)) / (IAxx + IBxx *
                sin(phi)**2 + IByy * cos(phi)**2 + lA**2 * mA + lB**2 * mB)
    assert (rhs[0] - xd).simplify() == 0
    xd = (IBxx - IByy) * omega**2 * sin(phi) * cos(phi) / IBzz
    assert (rhs[1] - xd).simplify() == 0

def test_four_bar_linkage_with_manual_constraints():
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1:4, u1:4')
    l1, l2, l3, l4, rho = symbols('l1:5, rho')

    N = ReferenceFrame('N')
    inertias = [inertia(N, 0, 0, rho * l ** 3 / 12) for l in (l1, l2, l3, l4)]
    with warns_deprecated_sympy():
        link1 = Body('Link1', frame=N, mass=rho * l1,
                     central_inertia=inertias[0])
        link2 = Body('Link2', mass=rho * l2, central_inertia=inertias[1])
        link3 = Body('Link3', mass=rho * l3, central_inertia=inertias[2])
        link4 = Body('Link4', mass=rho * l4, central_inertia=inertias[3])

    joint1 = PinJoint(
        'J1', link1, link2, coordinates=q1, speeds=u1, joint_axis=link1.z,
        parent_point=l1 / 2 * link1.x, child_point=-l2 / 2 * link2.x)
    joint2 = PinJoint(
        'J2', link2, link3, coordinates=q2, speeds=u2, joint_axis=link2.z,
        parent_point=l2 / 2 * link2.x, child_point=-l3 / 2 * link3.x)
    joint3 = PinJoint(
        'J3', link3, link4, coordinates=q3, speeds=u3, joint_axis=link3.z,
        parent_point=l3 / 2 * link3.x, child_point=-l4 / 2 * link4.x)

    loop = link4.masscenter.pos_from(link1.masscenter) \
           + l1 / 2 * link1.x + l4 / 2 * link4.x

    fh = Matrix([loop.dot(link1.x), loop.dot(link1.y)])

    with warns_deprecated_sympy():
        method = JointsMethod(link1, joint1, joint2, joint3)

    t = dynamicsymbols._t
    qdots = solve(method.kdes, [q1.diff(t), q2.diff(t), q3.diff(t)])
    fhd = fh.diff(t).subs(qdots)

    kane = KanesMethod(method.frame, q_ind=[q1], u_ind=[u1],
                       q_dependent=[q2, q3], u_dependent=[u2, u3],
                       kd_eqs=method.kdes, configuration_constraints=fh,
                       velocity_constraints=fhd, forcelist=method.loads,
                       bodies=method.bodies)
    fr, frs = kane.kanes_equations()
    assert fr == zeros(1)

    # Numerically check the mass- and forcing-matrix
    p = Matrix([l1, l2, l3, l4, rho])
    q = Matrix([q1, q2, q3])
    u = Matrix([u1, u2, u3])
    eval_m = lambdify((q, p), kane.mass_matrix)
    eval_f = lambdify((q, u, p), kane.forcing)
    eval_fhd = lambdify((q, u, p), fhd)

    p_vals = [0.13, 0.24, 0.21, 0.34, 997]
    q_vals = [2.1, 0.6655470375077588, 2.527408138024188]  # Satisfies fh
    u_vals = [0.2, -0.17963733938852067, 0.1309060540601612]  # Satisfies fhd
    mass_check = Matrix([[3.452709815256506e+01, 7.003948798374735e+00,
                          -4.939690970641498e+00],
                         [-2.203792703880936e-14, 2.071702479957077e-01,
                          2.842917573033711e-01],
                         [-1.300000000000123e-01, -8.836934896046506e-03,
                          1.864891330060847e-01]])
    forcing_check = Matrix([[-0.031211821321648],
                            [-0.00066022608181],
                            [0.001813559741243]])
    eps = 1e-10
    assert all(abs(x) < eps for x in eval_fhd(q_vals, u_vals, p_vals))
    assert all(abs(x) < eps for x in
               (Matrix(eval_m(q_vals, p_vals)) - mass_check))
    assert all(abs(x) < eps for x in
               (Matrix(eval_f(q_vals, u_vals, p_vals)) - forcing_check))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_partfrac.py ===
"""Tests for algorithms for partial fraction decomposition of rational
functions. """

from sympy.polys.partfrac import (
    apart_undetermined_coeffs,
    apart,
    apart_list, assemble_partfrac_list
)

from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, pi, all_close)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x, y, a, b, c


def test_apart():
    assert apart(1) == 1
    assert apart(1, x) == 1

    f, g = (x**2 + 1)/(x + 1), 2/(x + 1) + x - 1

    assert apart(f, full=False) == g
    assert apart(f, full=True) == g

    f, g = 1/(x + 2)/(x + 1), 1/(1 + x) - 1/(2 + x)

    assert apart(f, full=False) == g
    assert apart(f, full=True) == g

    f, g = 1/(x + 1)/(x + 5), -1/(5 + x)/4 + 1/(1 + x)/4

    assert apart(f, full=False) == g
    assert apart(f, full=True) == g

    assert apart((E*x + 2)/(x - pi)*(x - 1), x) == \
        2 - E + E*pi + E*x + (E*pi + 2)*(pi - 1)/(x - pi)

    assert apart(Eq((x**2 + 1)/(x + 1), x), x) == Eq(x - 1 + 2/(x + 1), x)

    assert apart(x/2, y) == x/2

    f, g = (x+y)/(2*x - y), Rational(3, 2)*y/(2*x - y) + S.Half

    assert apart(f, x, full=False) == g
    assert apart(f, x, full=True) == g

    f, g = (x+y)/(2*x - y), 3*x/(2*x - y) - 1

    assert apart(f, y, full=False) == g
    assert apart(f, y, full=True) == g

    raises(NotImplementedError, lambda: apart(1/(x + 1)/(y + 2)))


def test_apart_matrix():
    M = Matrix(2, 2, lambda i, j: 1/(x + i + 1)/(x + j))

    assert apart(M) == Matrix([
        [1/x - 1/(x + 1), (x + 1)**(-2)],
        [1/(2*x) - (S.Half)/(x + 2), 1/(x + 1) - 1/(x + 2)],
    ])


def test_apart_symbolic():
    f = a*x**4 + (2*b + 2*a*c)*x**3 + (4*b*c - a**2 + a*c**2)*x**2 + \
        (-2*a*b + 2*b*c**2)*x - b**2
    g = a**2*x**4 + (2*a*b + 2*c*a**2)*x**3 + (4*a*b*c + b**2 +
        a**2*c**2)*x**2 + (2*c*b**2 + 2*a*b*c**2)*x + b**2*c**2

    assert apart(f/g, x) == 1/a - 1/(x + c)**2 - b**2/(a*(a*x + b)**2)

    assert apart(1/((x + a)*(x + b)*(x + c)), x) == \
        1/((a - c)*(b - c)*(c + x)) - 1/((a - b)*(b - c)*(b + x)) + \
        1/((a - b)*(a - c)*(a + x))


def _make_extension_example():
    # https://github.com/sympy/sympy/issues/18531
    from sympy.core import Mul
    def mul2(expr):
        # 2-arg mul hack...
        return Mul(2, expr, evaluate=False)

    f = ((x**2 + 1)**3/((x - 1)**2*(x + 1)**2*(-x**2 + 2*x + 1)*(x**2 + 2*x - 1)))
    g = (1/mul2(x - sqrt(2) + 1)
       - 1/mul2(x - sqrt(2) - 1)
       + 1/mul2(x + 1 + sqrt(2))
       - 1/mul2(x - 1 + sqrt(2))
       + 1/mul2((x + 1)**2)
       + 1/mul2((x - 1)**2))
    return f, g


def test_apart_extension():
    f = 2/(x**2 + 1)
    g = I/(x + I) - I/(x - I)

    assert apart(f, extension=I) == g
    assert apart(f, gaussian=True) == g

    f = x/((x - 2)*(x + I))

    assert factor(together(apart(f)).expand()) == f

    f, g = _make_extension_example()

    # XXX: Only works with dotprodsimp. See test_apart_extension_xfail below
    from sympy.matrices import dotprodsimp
    with dotprodsimp(True):
        assert apart(f, x, extension={sqrt(2)}) == g


def test_apart_extension_xfail():
    f, g = _make_extension_example()
    assert apart(f, x, extension={sqrt(2)}) == g


def test_apart_full():
    f = 1/(x**2 + 1)

    assert apart(f, full=False) == f
    assert apart(f, full=True).dummy_eq(
        -RootSum(x**2 + 1, Lambda(a, a/(x - a)), auto=False)/2)

    f = 1/(x**3 + x + 1)

    assert apart(f, full=False) == f
    assert apart(f, full=True).dummy_eq(
        RootSum(x**3 + x + 1,
        Lambda(a, (a**2*Rational(6, 31) - a*Rational(9, 31) + Rational(4, 31))/(x - a)), auto=False))

    f = 1/(x**5 + 1)

    assert apart(f, full=False) == \
        (Rational(-1, 5))*((x**3 - 2*x**2 + 3*x - 4)/(x**4 - x**3 + x**2 -
         x + 1)) + (Rational(1, 5))/(x + 1)
    assert apart(f, full=True).dummy_eq(
        -RootSum(x**4 - x**3 + x**2 - x + 1,
        Lambda(a, a/(x - a)), auto=False)/5 + (Rational(1, 5))/(x + 1))


def test_apart_full_floats():
    # https://github.com/sympy/sympy/issues/26648
    f = (
        6.43369157032015e-9*x**3 + 1.35203404799555e-5*x**2
        + 0.00357538393743079*x + 0.085
        )/(
        4.74334912634438e-11*x**4 + 4.09576274286244e-6*x**3
        + 0.00334241812250921*x**2 + 0.15406018058983*x + 1.0
    )

    expected = (
        133.599202650992/(x + 85524.0054884464)
        + 1.07757928431867/(x + 774.88576677949)
        + 0.395006955518971/(x + 40.7977016133126)
        + 0.564264854137341/(x + 7.79746609204661)
    )

    f_apart = apart(f, full=True).evalf()

    # There is a significant floating point error in this operation.
    assert all_close(f_apart, expected, rtol=1e-3, atol=1e-5)


def test_apart_undetermined_coeffs():
    p = Poly(2*x - 3)
    q = Poly(x**9 - x**8 - x**6 + x**5 - 2*x**2 + 3*x - 1)
    r = (-x**7 - x**6 - x**5 + 4)/(x**8 - x**5 - 2*x + 1) + 1/(x - 1)

    assert apart_undetermined_coeffs(p, q) == r

    p = Poly(1, x, domain='ZZ[a,b]')
    q = Poly((x + a)*(x + b), x, domain='ZZ[a,b]')
    r = 1/((a - b)*(b + x)) - 1/((a - b)*(a + x))

    assert apart_undetermined_coeffs(p, q) == r


def test_apart_list():
    from sympy.utilities.iterables import numbered_symbols
    def dummy_eq(i, j):
        if type(i) in (list, tuple):
            return all(dummy_eq(i, j) for i, j in zip(i, j))
        return i == j or i.dummy_eq(j)

    w0, w1, w2 = Symbol("w0"), Symbol("w1"), Symbol("w2")
    _a = Dummy("a")

    f = (-2*x - 2*x**2) / (3*x**2 - 6*x)
    got = apart_list(f, x, dummies=numbered_symbols("w"))
    ans = (-1, Poly(Rational(2, 3), x, domain='QQ'),
        [(Poly(w0 - 2, w0, domain='ZZ'), Lambda(_a, 2), Lambda(_a, -_a + x), 1)])
    assert dummy_eq(got, ans)

    got = apart_list(2/(x**2-2), x, dummies=numbered_symbols("w"))
    ans = (1, Poly(0, x, domain='ZZ'), [(Poly(w0**2 - 2, w0, domain='ZZ'),
        Lambda(_a, _a/2),
        Lambda(_a, -_a + x), 1)])
    assert dummy_eq(got, ans)

    f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)
    got = apart_list(f, x, dummies=numbered_symbols("w"))
    ans = (1, Poly(0, x, domain='ZZ'),
        [(Poly(w0 - 2, w0, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1),
        (Poly(w1**2 - 1, w1, domain='ZZ'), Lambda(_a, -3*_a - 6), Lambda(_a, -_a + x), 2),
        (Poly(w2 + 1, w2, domain='ZZ'), Lambda(_a, -4), Lambda(_a, -_a + x), 1)])
    assert dummy_eq(got, ans)


def test_assemble_partfrac_list():
    f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)
    pfd = apart_list(f)
    assert assemble_partfrac_list(pfd) == -4/(x + 1) - 3/(x + 1)**2 - 9/(x - 1)**2 + 4/(x - 2)

    a = Dummy("a")
    pfd = (1, Poly(0, x, domain='ZZ'), [([sqrt(2),-sqrt(2)], Lambda(a, a/2), Lambda(a, -a + x), 1)])
    assert assemble_partfrac_list(pfd) == -1/(sqrt(2)*(x + sqrt(2))) + 1/(sqrt(2)*(x - sqrt(2)))


@XFAIL
def test_noncommutative_pseudomultivariate():
    # apart doesn't go inside noncommutative expressions
    class foo(Expr):
        is_commutative=False
    e = x/(x + x*y)
    c = 1/(1 + y)
    assert apart(e + foo(e)) == c + foo(c)
    assert apart(e*foo(e)) == c*foo(c)

def test_noncommutative():
    class foo(Expr):
        is_commutative=False
    e = x/(x + x*y)
    c = 1/(1 + y)
    assert apart(e + foo()) == c + foo()

def test_issue_5798():
    assert apart(
        2*x/(x**2 + 1) - (x - 1)/(2*(x**2 + 1)) + 1/(2*(x + 1)) - 2/x) == \
        (3*x + 1)/(x**2 + 1)/2 + 1/(x + 1)/2 - 2/x

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\scalars.py ===
import datetime as dt

import pytest
import numpy as np

b = np.bool()
b_ = np.bool_()
u8 = np.uint64()
i8 = np.int64()
f8 = np.float64()
c16 = np.complex128()
U = np.str_()
S = np.bytes_()


# Construction
class D:
    def __index__(self) -> int:
        return 0


class C:
    def __complex__(self) -> complex:
        return 3j


class B:
    def __int__(self) -> int:
        return 4


class A:
    def __float__(self) -> float:
        return 4.0


np.complex64(3j)
np.complex64(A())
np.complex64(C())
np.complex128(3j)
np.complex128(C())
np.complex128(None)
np.complex64("1.2")
np.complex128(b"2j")

np.int8(4)
np.int16(3.4)
np.int32(4)
np.int64(-1)
np.uint8(B())
np.uint32()
np.int32("1")
np.int64(b"2")

np.float16(A())
np.float32(16)
np.float64(3.0)
np.float64(None)
np.float32("1")
np.float16(b"2.5")

np.uint64(D())
np.float32(D())
np.complex64(D())

np.bytes_(b"hello")
np.bytes_("hello", 'utf-8')
np.bytes_("hello", encoding='utf-8')
np.str_("hello")
np.str_(b"hello", 'utf-8')
np.str_(b"hello", encoding='utf-8')

# Array-ish semantics
np.int8().real
np.int16().imag
np.int32().data
np.int64().flags

np.uint8().itemsize * 2
np.uint16().ndim + 1
np.uint32().strides
np.uint64().shape

# Time structures
np.datetime64()
np.datetime64(0, "D")
np.datetime64(0, b"D")
np.datetime64(0, ('ms', 3))
np.datetime64("2019")
np.datetime64(b"2019")
np.datetime64("2019", "D")
np.datetime64("2019", "us")
np.datetime64("2019", "as")
np.datetime64(np.datetime64())
np.datetime64(np.datetime64())
np.datetime64(dt.datetime(2000, 5, 3))
np.datetime64(dt.datetime(2000, 5, 3), "D")
np.datetime64(dt.datetime(2000, 5, 3), "us")
np.datetime64(dt.datetime(2000, 5, 3), "as")
np.datetime64(dt.date(2000, 5, 3))
np.datetime64(dt.date(2000, 5, 3), "D")
np.datetime64(dt.date(2000, 5, 3), "us")
np.datetime64(dt.date(2000, 5, 3), "as")
np.datetime64(None)
np.datetime64(None, "D")

np.timedelta64()
np.timedelta64(0)
np.timedelta64(0, "D")
np.timedelta64(0, ('ms', 3))
np.timedelta64(0, b"D")
np.timedelta64("3")
np.timedelta64(b"5")
np.timedelta64(np.timedelta64(2))
np.timedelta64(dt.timedelta(2))
np.timedelta64(None)
np.timedelta64(None, "D")

np.void(1)
np.void(np.int64(1))
np.void(True)
np.void(np.bool(True))
np.void(b"test")
np.void(np.bytes_("test"))
np.void(object(), [("a", "O"), ("b", "O")])
np.void(object(), dtype=[("a", "O"), ("b", "O")])

# Protocols
i8 = np.int64()
u8 = np.uint64()
f8 = np.float64()
c16 = np.complex128()
b = np.bool()
td = np.timedelta64()
U = np.str_("1")
S = np.bytes_("1")
AR = np.array(1, dtype=np.float64)

int(i8)
int(u8)
int(f8)
int(b)
int(td)
int(U)
int(S)
int(AR)
with pytest.warns(np.exceptions.ComplexWarning):
    int(c16)

float(i8)
float(u8)
float(f8)
float(b_)
float(td)
float(U)
float(S)
float(AR)
with pytest.warns(np.exceptions.ComplexWarning):
    float(c16)

complex(i8)
complex(u8)
complex(f8)
complex(c16)
complex(b_)
complex(td)
complex(U)
complex(AR)


# Misc
c16.dtype
c16.real
c16.imag
c16.real.real
c16.real.imag
c16.ndim
c16.size
c16.itemsize
c16.shape
c16.strides
c16.squeeze()
c16.byteswap()
c16.transpose()

# Aliases
np.byte()
np.short()
np.intc()
np.intp()
np.int_()
np.longlong()

np.ubyte()
np.ushort()
np.uintc()
np.uintp()
np.uint()
np.ulonglong()

np.half()
np.single()
np.double()
np.longdouble()

np.csingle()
np.cdouble()
np.clongdouble()

b.item()
i8.item()
u8.item()
f8.item()
c16.item()
U.item()
S.item()

b.tolist()
i8.tolist()
u8.tolist()
f8.tolist()
c16.tolist()
U.tolist()
S.tolist()

b.ravel()
i8.ravel()
u8.ravel()
f8.ravel()
c16.ravel()
U.ravel()
S.ravel()

b.flatten()
i8.flatten()
u8.flatten()
f8.flatten()
c16.flatten()
U.flatten()
S.flatten()

b.reshape(1)
i8.reshape(1)
u8.reshape(1)
f8.reshape(1)
c16.reshape(1)
U.reshape(1)
S.reshape(1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\masked\test_arithmetic.py ===
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

# integer dtypes
arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES]
scalars: list[Any] = [2] * len(arrays)
# floating dtypes
arrays += [pd.array([0.1, 0.2, 0.3, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES]
scalars += [0.2, 0.2]
# boolean
arrays += [pd.array([True, False, True, None], dtype="boolean")]
scalars += [False]


@pytest.fixture(params=zip(arrays, scalars), ids=[a.dtype.name for a in arrays])
def data(request):
    """Fixture returning parametrized (array, scalar) tuple.

    Used to test equivalence of scalars, numpy arrays with array ops, and the
    equivalence of DataFrame and Series ops.
    """
    return request.param


def check_skip(data, op_name):
    if isinstance(data.dtype, pd.BooleanDtype) and "sub" in op_name:
        pytest.skip("subtract not implemented for boolean")


def is_bool_not_implemented(data, op_name):
    # match non-masked behavior
    return data.dtype.kind == "b" and op_name.strip("_").lstrip("r") in [
        "pow",
        "truediv",
        "floordiv",
    ]


# Test equivalence of scalars, numpy arrays with array ops
# -----------------------------------------------------------------------------


def test_array_scalar_like_equivalence(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    scalar_array = pd.array([scalar] * len(data), dtype=data.dtype)

    # TODO also add len-1 array (np.array([scalar], dtype=data.dtype.numpy_dtype))
    for scalar in [scalar, data.dtype.type(scalar)]:
        if is_bool_not_implemented(data, all_arithmetic_operators):
            msg = "operator '.*' not implemented for bool dtypes"
            with pytest.raises(NotImplementedError, match=msg):
                op(data, scalar)
            with pytest.raises(NotImplementedError, match=msg):
                op(data, scalar_array)
        else:
            result = op(data, scalar)
            expected = op(data, scalar_array)
            tm.assert_extension_array_equal(result, expected)


def test_array_NA(data, all_arithmetic_operators):
    data, _ = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    scalar = pd.NA
    scalar_array = pd.array([pd.NA] * len(data), dtype=data.dtype)

    mask = data._mask.copy()

    if is_bool_not_implemented(data, all_arithmetic_operators):
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            op(data, scalar)
        # GH#45421 check op doesn't alter data._mask inplace
        tm.assert_numpy_array_equal(mask, data._mask)
        return

    result = op(data, scalar)
    # GH#45421 check op doesn't alter data._mask inplace
    tm.assert_numpy_array_equal(mask, data._mask)

    expected = op(data, scalar_array)
    tm.assert_numpy_array_equal(mask, data._mask)

    tm.assert_extension_array_equal(result, expected)


def test_numpy_array_equivalence(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    numpy_array = np.array([scalar] * len(data), dtype=data.dtype.numpy_dtype)
    pd_array = pd.array(numpy_array, dtype=data.dtype)

    if is_bool_not_implemented(data, all_arithmetic_operators):
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            op(data, numpy_array)
        with pytest.raises(NotImplementedError, match=msg):
            op(data, pd_array)
        return

    result = op(data, numpy_array)
    expected = op(data, pd_array)
    tm.assert_extension_array_equal(result, expected)


# Test equivalence with Series and DataFrame ops
# -----------------------------------------------------------------------------


def test_frame(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    # DataFrame with scalar
    df = pd.DataFrame({"A": data})

    if is_bool_not_implemented(data, all_arithmetic_operators):
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            op(df, scalar)
        with pytest.raises(NotImplementedError, match=msg):
            op(data, scalar)
        return

    result = op(df, scalar)
    expected = pd.DataFrame({"A": op(data, scalar)})
    tm.assert_frame_equal(result, expected)


def test_series(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    ser = pd.Series(data)

    others = [
        scalar,
        np.array([scalar] * len(data), dtype=data.dtype.numpy_dtype),
        pd.array([scalar] * len(data), dtype=data.dtype),
        pd.Series([scalar] * len(data), dtype=data.dtype),
    ]

    for other in others:
        if is_bool_not_implemented(data, all_arithmetic_operators):
            msg = "operator '.*' not implemented for bool dtypes"
            with pytest.raises(NotImplementedError, match=msg):
                op(ser, other)

        else:
            result = op(ser, other)
            expected = pd.Series(op(data, other))
            tm.assert_series_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


def test_error_invalid_object(data, all_arithmetic_operators):
    data, _ = data

    op = all_arithmetic_operators
    opa = getattr(data, op)

    # 2d -> return NotImplemented
    result = opa(pd.DataFrame({"A": data}))
    assert result is NotImplemented

    msg = r"can only perform ops with 1-d structures"
    with pytest.raises(NotImplementedError, match=msg):
        opa(np.arange(len(data)).reshape(-1, len(data)))


def test_error_len_mismatch(data, all_arithmetic_operators):
    # operating with a list-like with non-matching length raises
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)

    other = [scalar] * (len(data) - 1)

    err = ValueError
    msg = "|".join(
        [
            r"operands could not be broadcast together with shapes \(3,\) \(4,\)",
            r"operands could not be broadcast together with shapes \(4,\) \(3,\)",
        ]
    )
    if data.dtype.kind == "b" and all_arithmetic_operators.strip("_") in [
        "sub",
        "rsub",
    ]:
        err = TypeError
        msg = (
            r"numpy boolean subtract, the `\-` operator, is not supported, use "
            r"the bitwise_xor, the `\^` operator, or the logical_xor function instead"
        )
    elif is_bool_not_implemented(data, all_arithmetic_operators):
        msg = "operator '.*' not implemented for bool dtypes"
        err = NotImplementedError

    for other in [other, np.array(other)]:
        with pytest.raises(err, match=msg):
            op(data, other)

        s = pd.Series(data)
        with pytest.raises(err, match=msg):
            op(s, other)


@pytest.mark.parametrize("op", ["__neg__", "__abs__", "__invert__"])
def test_unary_op_does_not_propagate_mask(data, op):
    # https://github.com/pandas-dev/pandas/issues/39943
    data, _ = data
    ser = pd.Series(data)

    if op == "__invert__" and data.dtype.kind == "f":
        # we follow numpy in raising
        msg = "ufunc 'invert' not supported for the input types"
        with pytest.raises(TypeError, match=msg):
            getattr(ser, op)()
        with pytest.raises(TypeError, match=msg):
            getattr(data, op)()
        with pytest.raises(TypeError, match=msg):
            # Check that this is still the numpy behavior
            getattr(data._data, op)()

        return

    result = getattr(ser, op)()
    expected = result.copy(deep=True)
    ser[0] = None
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_nlargest.py ===
"""
Note: for naming purposes, most tests are title with as e.g. "test_nlargest_foo"
but are implicitly also testing nsmallest_foo.
"""
from itertools import product

import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm

main_dtypes = [
    "datetime",
    "datetimetz",
    "timedelta",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


@pytest.fixture
def s_main_dtypes():
    """
    A DataFrame with many dtypes

    * datetime
    * datetimetz
    * timedelta
    * [u]int{8,16,32,64}
    * float{32,64}

    The columns are the name of the dtype.
    """
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2003", "2002", "2001", "2002", "2005"]),
            "datetimetz": pd.to_datetime(
                ["2003", "2002", "2001", "2002", "2005"]
            ).tz_localize("US/Eastern"),
            "timedelta": pd.to_timedelta(["3d", "2d", "1d", "2d", "5d"]),
        }
    )

    for dtype in [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]:
        df[dtype] = Series([3, 2, 1, 2, 5], dtype=dtype)

    return df


@pytest.fixture(params=main_dtypes)
def s_main_dtypes_split(request, s_main_dtypes):
    """Each series in s_main_dtypes."""
    return s_main_dtypes[request.param]


def assert_check_nselect_boundary(vals, dtype, method):
    # helper function for 'test_boundary_{dtype}' tests
    ser = Series(vals, dtype=dtype)
    result = getattr(ser, method)(3)
    expected_idxr = [0, 1, 2] if method == "nsmallest" else [3, 2, 1]
    expected = ser.loc[expected_idxr]
    tm.assert_series_equal(result, expected)


class TestSeriesNLargestNSmallest:
    @pytest.mark.parametrize(
        "r",
        [
            Series([3.0, 2, 1, 2, "5"], dtype="object"),
            Series([3.0, 2, 1, 2, 5], dtype="object"),
            # not supported on some archs
            # Series([3., 2, 1, 2, 5], dtype='complex256'),
            Series([3.0, 2, 1, 2, 5], dtype="complex128"),
            Series(list("abcde")),
            Series(list("abcde"), dtype="category"),
        ],
    )
    def test_nlargest_error(self, r):
        dt = r.dtype
        msg = f"Cannot use method 'n(largest|smallest)' with dtype {dt}"
        args = 2, len(r), 0, -1
        methods = r.nlargest, r.nsmallest
        for method, arg in product(methods, args):
            with pytest.raises(TypeError, match=msg):
                method(arg)

    def test_nsmallest_nlargest(self, s_main_dtypes_split):
        # float, int, datetime64 (use i8), timedelts64 (same),
        # object that are numbers, object that are strings
        ser = s_main_dtypes_split

        tm.assert_series_equal(ser.nsmallest(2), ser.iloc[[2, 1]])
        tm.assert_series_equal(ser.nsmallest(2, keep="last"), ser.iloc[[2, 3]])

        empty = ser.iloc[0:0]
        tm.assert_series_equal(ser.nsmallest(0), empty)
        tm.assert_series_equal(ser.nsmallest(-1), empty)
        tm.assert_series_equal(ser.nlargest(0), empty)
        tm.assert_series_equal(ser.nlargest(-1), empty)

        tm.assert_series_equal(ser.nsmallest(len(ser)), ser.sort_values())
        tm.assert_series_equal(ser.nsmallest(len(ser) + 1), ser.sort_values())
        tm.assert_series_equal(ser.nlargest(len(ser)), ser.iloc[[4, 0, 1, 3, 2]])
        tm.assert_series_equal(ser.nlargest(len(ser) + 1), ser.iloc[[4, 0, 1, 3, 2]])

    def test_nlargest_misc(self):
        ser = Series([3.0, np.nan, 1, 2, 5])
        result = ser.nlargest()
        expected = ser.iloc[[4, 0, 3, 2, 1]]
        tm.assert_series_equal(result, expected)
        result = ser.nsmallest()
        expected = ser.iloc[[2, 3, 0, 4, 1]]
        tm.assert_series_equal(result, expected)

        msg = 'keep must be either "first", "last"'
        with pytest.raises(ValueError, match=msg):
            ser.nsmallest(keep="invalid")
        with pytest.raises(ValueError, match=msg):
            ser.nlargest(keep="invalid")

        # GH#15297
        ser = Series([1] * 5, index=[1, 2, 3, 4, 5])
        expected_first = Series([1] * 3, index=[1, 2, 3])
        expected_last = Series([1] * 3, index=[5, 4, 3])

        result = ser.nsmallest(3)
        tm.assert_series_equal(result, expected_first)

        result = ser.nsmallest(3, keep="last")
        tm.assert_series_equal(result, expected_last)

        result = ser.nlargest(3)
        tm.assert_series_equal(result, expected_first)

        result = ser.nlargest(3, keep="last")
        tm.assert_series_equal(result, expected_last)

    @pytest.mark.parametrize("n", range(1, 5))
    def test_nlargest_n(self, n):
        # GH 13412
        ser = Series([1, 4, 3, 2], index=[0, 0, 1, 1])
        result = ser.nlargest(n)
        expected = ser.sort_values(ascending=False).head(n)
        tm.assert_series_equal(result, expected)

        result = ser.nsmallest(n)
        expected = ser.sort_values().head(n)
        tm.assert_series_equal(result, expected)

    def test_nlargest_boundary_integer(self, nselect_method, any_int_numpy_dtype):
        # GH#21426
        dtype_info = np.iinfo(any_int_numpy_dtype)
        min_val, max_val = dtype_info.min, dtype_info.max
        vals = [min_val, min_val + 1, max_val - 1, max_val]
        assert_check_nselect_boundary(vals, any_int_numpy_dtype, nselect_method)

    def test_nlargest_boundary_float(self, nselect_method, float_numpy_dtype):
        # GH#21426
        dtype_info = np.finfo(float_numpy_dtype)
        min_val, max_val = dtype_info.min, dtype_info.max
        min_2nd, max_2nd = np.nextafter([min_val, max_val], 0, dtype=float_numpy_dtype)
        vals = [min_val, min_2nd, max_2nd, max_val]
        assert_check_nselect_boundary(vals, float_numpy_dtype, nselect_method)

    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_nlargest_boundary_datetimelike(self, nselect_method, dtype):
        # GH#21426
        # use int64 bounds and +1 to min_val since true minimum is NaT
        # (include min_val/NaT at end to maintain same expected_idxr)
        dtype_info = np.iinfo("int64")
        min_val, max_val = dtype_info.min, dtype_info.max
        vals = [min_val + 1, min_val + 2, max_val - 1, max_val, min_val]
        assert_check_nselect_boundary(vals, dtype, nselect_method)

    def test_nlargest_duplicate_keep_all_ties(self):
        # see GH#16818
        ser = Series([10, 9, 8, 7, 7, 7, 7, 6])
        result = ser.nlargest(4, keep="all")
        expected = Series([10, 9, 8, 7, 7, 7, 7])
        tm.assert_series_equal(result, expected)

        result = ser.nsmallest(2, keep="all")
        expected = Series([6, 7, 7, 7, 7], index=[7, 3, 4, 5, 6])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data,expected", [([True, False], [True]), ([True, False, True, True], [True])]
    )
    def test_nlargest_boolean(self, data, expected):
        # GH#26154 : ensure True > False
        ser = Series(data)
        result = ser.nlargest(1)
        expected = Series(expected)
        tm.assert_series_equal(result, expected)

    def test_nlargest_nullable(self, any_numeric_ea_dtype):
        # GH#42816
        dtype = any_numeric_ea_dtype
        if dtype.startswith("UInt"):
            # Can't cast from negative float to uint on some platforms
            arr = np.random.default_rng(2).integers(1, 10, 10)
        else:
            arr = np.random.default_rng(2).standard_normal(10)
        arr = arr.astype(dtype.lower(), copy=False)

        ser = Series(arr.copy(), dtype=dtype)
        ser[1] = pd.NA
        result = ser.nlargest(5)

        expected = (
            Series(np.delete(arr, 1), index=ser.index.delete(1))
            .nlargest(5)
            .astype(dtype)
        )
        tm.assert_series_equal(result, expected)

    def test_nsmallest_nan_when_keep_is_all(self):
        # GH#46589
        s = Series([1, 2, 3, 3, 3, None])
        result = s.nsmallest(3, keep="all")
        expected = Series([1.0, 2.0, 3.0, 3.0, 3.0])
        tm.assert_series_equal(result, expected)

        s = Series([1, 2, None, None, None])
        result = s.nsmallest(3, keep="all")
        expected = Series([1, 2, None, None, None])
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_state.py ===
from sympy.core.add import Add
from sympy.core.function import diff
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import raises

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.state import (
    Ket, Bra, TimeDepKet, TimeDepBra,
    KetBase, BraBase, StateBase, Wavefunction,
    OrthogonalKet, OrthogonalBra
)
from sympy.physics.quantum.hilbert import HilbertSpace

x, y, t = symbols('x,y,t')


class CustomKet(Ket):
    @classmethod
    def default_args(self):
        return ("test",)


class CustomKetMultipleLabels(Ket):
    @classmethod
    def default_args(self):
        return ("r", "theta", "phi")


class CustomTimeDepKet(TimeDepKet):
    @classmethod
    def default_args(self):
        return ("test", "t")


class CustomTimeDepKetMultipleLabels(TimeDepKet):
    @classmethod
    def default_args(self):
        return ("r", "theta", "phi", "t")


def test_ket():
    k = Ket('0')

    assert isinstance(k, Ket)
    assert isinstance(k, KetBase)
    assert isinstance(k, StateBase)
    assert isinstance(k, QExpr)

    assert k.label == (Symbol('0'),)
    assert k.hilbert_space == HilbertSpace()
    assert k.is_commutative is False

    # Make sure this doesn't get converted to the number pi.
    k = Ket('pi')
    assert k.label == (Symbol('pi'),)

    k = Ket(x, y)
    assert k.label == (x, y)
    assert k.hilbert_space == HilbertSpace()
    assert k.is_commutative is False

    assert k.dual_class() == Bra
    assert k.dual == Bra(x, y)
    assert k.subs(x, y) == Ket(y, y)

    k = CustomKet()
    assert k == CustomKet("test")

    k = CustomKetMultipleLabels()
    assert k == CustomKetMultipleLabels("r", "theta", "phi")

    assert Ket() == Ket('psi')


def test_bra():
    b = Bra('0')

    assert isinstance(b, Bra)
    assert isinstance(b, BraBase)
    assert isinstance(b, StateBase)
    assert isinstance(b, QExpr)

    assert b.label == (Symbol('0'),)
    assert b.hilbert_space == HilbertSpace()
    assert b.is_commutative is False

    # Make sure this doesn't get converted to the number pi.
    b = Bra('pi')
    assert b.label == (Symbol('pi'),)

    b = Bra(x, y)
    assert b.label == (x, y)
    assert b.hilbert_space == HilbertSpace()
    assert b.is_commutative is False

    assert b.dual_class() == Ket
    assert b.dual == Ket(x, y)
    assert b.subs(x, y) == Bra(y, y)

    assert Bra() == Bra('psi')


def test_ops():
    k0 = Ket(0)
    k1 = Ket(1)
    k = 2*I*k0 - (x/sqrt(2))*k1
    assert k == Add(Mul(2, I, k0),
        Mul(Rational(-1, 2), x, Pow(2, S.Half), k1))


def test_time_dep_ket():
    k = TimeDepKet(0, t)

    assert isinstance(k, TimeDepKet)
    assert isinstance(k, KetBase)
    assert isinstance(k, StateBase)
    assert isinstance(k, QExpr)

    assert k.label == (Integer(0),)
    assert k.args == (Integer(0), t)
    assert k.time == t

    assert k.dual_class() == TimeDepBra
    assert k.dual == TimeDepBra(0, t)

    assert k.subs(t, 2) == TimeDepKet(0, 2)

    k = TimeDepKet(x, 0.5)
    assert k.label == (x,)
    assert k.args == (x, sympify(0.5))

    k = CustomTimeDepKet()
    assert k.label == (Symbol("test"),)
    assert k.time == Symbol("t")
    assert k == CustomTimeDepKet("test", "t")

    k = CustomTimeDepKetMultipleLabels()
    assert k.label == (Symbol("r"), Symbol("theta"), Symbol("phi"))
    assert k.time == Symbol("t")
    assert k == CustomTimeDepKetMultipleLabels("r", "theta", "phi", "t")

    assert TimeDepKet() == TimeDepKet("psi", "t")


def test_time_dep_bra():
    b = TimeDepBra(0, t)

    assert isinstance(b, TimeDepBra)
    assert isinstance(b, BraBase)
    assert isinstance(b, StateBase)
    assert isinstance(b, QExpr)

    assert b.label == (Integer(0),)
    assert b.args == (Integer(0), t)
    assert b.time == t

    assert b.dual_class() == TimeDepKet
    assert b.dual == TimeDepKet(0, t)

    k = TimeDepBra(x, 0.5)
    assert k.label == (x,)
    assert k.args == (x, sympify(0.5))

    assert TimeDepBra() == TimeDepBra("psi", "t")


def test_bra_ket_dagger():
    x = symbols('x', complex=True)
    k = Ket('k')
    b = Bra('b')
    assert Dagger(k) == Bra('k')
    assert Dagger(b) == Ket('b')
    assert Dagger(k).is_commutative is False

    k2 = Ket('k2')
    e = 2*I*k + x*k2
    assert Dagger(e) == conjugate(x)*Dagger(k2) - 2*I*Dagger(k)


def test_wavefunction():
    x, y = symbols('x y', real=True)
    L = symbols('L', positive=True)
    n = symbols('n', integer=True, positive=True)

    f = Wavefunction(x**2, x)
    p = f.prob()
    lims = f.limits

    assert f.is_normalized is False
    assert f.norm is oo
    assert f(10) == 100
    assert p(10) == 10000
    assert lims[x] == (-oo, oo)
    assert diff(f, x) == Wavefunction(2*x, x)
    raises(NotImplementedError, lambda: f.normalize())
    assert conjugate(f) == Wavefunction(conjugate(f.expr), x)
    assert conjugate(f) == Dagger(f)

    g = Wavefunction(x**2*y + y**2*x, (x, 0, 1), (y, 0, 2))
    lims_g = g.limits

    assert lims_g[x] == (0, 1)
    assert lims_g[y] == (0, 2)
    assert g.is_normalized is False
    assert g.norm == sqrt(42)/3
    assert g(2, 4) == 0
    assert g(1, 1) == 2
    assert diff(diff(g, x), y) == Wavefunction(2*x + 2*y, (x, 0, 1), (y, 0, 2))
    assert conjugate(g) == Wavefunction(conjugate(g.expr), *g.args[1:])
    assert conjugate(g) == Dagger(g)

    h = Wavefunction(sqrt(5)*x**2, (x, 0, 1))
    assert h.is_normalized is True
    assert h.normalize() == h
    assert conjugate(h) == Wavefunction(conjugate(h.expr), (x, 0, 1))
    assert conjugate(h) == Dagger(h)

    piab = Wavefunction(sin(n*pi*x/L), (x, 0, L))
    assert piab.norm == sqrt(L/2)
    assert piab(L + 1) == 0
    assert piab(0.5) == sin(0.5*n*pi/L)
    assert piab(0.5, n=1, L=1) == sin(0.5*pi)
    assert piab.normalize() == \
        Wavefunction(sqrt(2)/sqrt(L)*sin(n*pi*x/L), (x, 0, L))
    assert conjugate(piab) == Wavefunction(conjugate(piab.expr), (x, 0, L))
    assert conjugate(piab) == Dagger(piab)

    k = Wavefunction(x**2, 'x')
    assert type(k.variables[0]) == Symbol

def test_orthogonal_states():
    bracket = OrthogonalBra(x) * OrthogonalKet(x)
    assert bracket.doit() == 1

    bracket = OrthogonalBra(x) * OrthogonalKet(x+1)
    assert bracket.doit() == 0

    bracket = OrthogonalBra(x) * OrthogonalKet(y)
    assert bracket.doit() == bracket

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_quantile.py ===
import numpy as np
import pytest

from pandas.core.dtypes.common import is_integer

import pandas as pd
from pandas import (
    Index,
    Series,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import Timestamp


class TestSeriesQuantile:
    def test_quantile(self, datetime_series):
        q = datetime_series.quantile(0.1)
        assert q == np.percentile(datetime_series.dropna(), 10)

        q = datetime_series.quantile(0.9)
        assert q == np.percentile(datetime_series.dropna(), 90)

        # object dtype
        q = Series(datetime_series, dtype=object).quantile(0.9)
        assert q == np.percentile(datetime_series.dropna(), 90)

        # datetime64[ns] dtype
        dts = datetime_series.index.to_series()
        q = dts.quantile(0.2)
        assert q == Timestamp("2000-01-10 19:12:00")

        # timedelta64[ns] dtype
        tds = dts.diff()
        q = tds.quantile(0.25)
        assert q == pd.to_timedelta("24:00:00")

        # GH7661
        result = Series([np.timedelta64("NaT")]).sum()
        assert result == pd.Timedelta(0)

        msg = "percentiles should all be in the interval \\[0, 1\\]"
        for invalid in [-1, 2, [0.5, -1], [0.5, 2]]:
            with pytest.raises(ValueError, match=msg):
                datetime_series.quantile(invalid)

        s = Series(np.random.default_rng(2).standard_normal(100))
        percentile_array = [-0.5, 0.25, 1.5]
        with pytest.raises(ValueError, match=msg):
            s.quantile(percentile_array)

    def test_quantile_multi(self, datetime_series, unit):
        datetime_series.index = datetime_series.index.as_unit(unit)
        qs = [0.1, 0.9]
        result = datetime_series.quantile(qs)
        expected = Series(
            [
                np.percentile(datetime_series.dropna(), 10),
                np.percentile(datetime_series.dropna(), 90),
            ],
            index=qs,
            name=datetime_series.name,
        )
        tm.assert_series_equal(result, expected)

        dts = datetime_series.index.to_series()
        dts.name = "xxx"
        result = dts.quantile((0.2, 0.2))
        expected = Series(
            [Timestamp("2000-01-10 19:12:00"), Timestamp("2000-01-10 19:12:00")],
            index=[0.2, 0.2],
            name="xxx",
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(result, expected)

        result = datetime_series.quantile([])
        expected = Series(
            [], name=datetime_series.name, index=Index([], dtype=float), dtype="float64"
        )
        tm.assert_series_equal(result, expected)

    def test_quantile_interpolation(self, datetime_series):
        # see gh-10174

        # interpolation = linear (default case)
        q = datetime_series.quantile(0.1, interpolation="linear")
        assert q == np.percentile(datetime_series.dropna(), 10)
        q1 = datetime_series.quantile(0.1)
        assert q1 == np.percentile(datetime_series.dropna(), 10)

        # test with and without interpolation keyword
        assert q == q1

    def test_quantile_interpolation_dtype(self):
        # GH #10174

        # interpolation = linear (default case)
        q = Series([1, 3, 4]).quantile(0.5, interpolation="lower")
        assert q == np.percentile(np.array([1, 3, 4]), 50)
        assert is_integer(q)

        q = Series([1, 3, 4]).quantile(0.5, interpolation="higher")
        assert q == np.percentile(np.array([1, 3, 4]), 50)
        assert is_integer(q)

    def test_quantile_nan(self):
        # GH 13098
        ser = Series([1, 2, 3, 4, np.nan])
        result = ser.quantile(0.5)
        expected = 2.5
        assert result == expected

        # all nan/empty
        s1 = Series([], dtype=object)
        cases = [s1, Series([np.nan, np.nan])]

        for ser in cases:
            res = ser.quantile(0.5)
            assert np.isnan(res)

            res = ser.quantile([0.5])
            tm.assert_series_equal(res, Series([np.nan], index=[0.5]))

            res = ser.quantile([0.2, 0.3])
            tm.assert_series_equal(res, Series([np.nan, np.nan], index=[0.2, 0.3]))

    @pytest.mark.parametrize(
        "case",
        [
            [
                Timestamp("2011-01-01"),
                Timestamp("2011-01-02"),
                Timestamp("2011-01-03"),
            ],
            [
                Timestamp("2011-01-01", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                Timestamp("2011-01-03", tz="US/Eastern"),
            ],
            [pd.Timedelta("1 days"), pd.Timedelta("2 days"), pd.Timedelta("3 days")],
            # NaT
            [
                Timestamp("2011-01-01"),
                Timestamp("2011-01-02"),
                Timestamp("2011-01-03"),
                pd.NaT,
            ],
            [
                Timestamp("2011-01-01", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                Timestamp("2011-01-03", tz="US/Eastern"),
                pd.NaT,
            ],
            [
                pd.Timedelta("1 days"),
                pd.Timedelta("2 days"),
                pd.Timedelta("3 days"),
                pd.NaT,
            ],
        ],
    )
    def test_quantile_box(self, case):
        ser = Series(case, name="XXX")
        res = ser.quantile(0.5)
        assert res == case[1]

        res = ser.quantile([0.5])
        exp = Series([case[1]], index=[0.5], name="XXX")
        tm.assert_series_equal(res, exp)

    def test_datetime_timedelta_quantiles(self):
        # covers #9694
        assert pd.isna(Series([], dtype="M8[ns]").quantile(0.5))
        assert pd.isna(Series([], dtype="m8[ns]").quantile(0.5))

    def test_quantile_nat(self):
        res = Series([pd.NaT, pd.NaT]).quantile(0.5)
        assert res is pd.NaT

        res = Series([pd.NaT, pd.NaT]).quantile([0.5])
        tm.assert_series_equal(res, Series([pd.NaT], index=[0.5]))

    @pytest.mark.parametrize(
        "values, dtype",
        [([0, 0, 0, 1, 2, 3], "Sparse[int]"), ([0.0, None, 1.0, 2.0], "Sparse[float]")],
    )
    def test_quantile_sparse(self, values, dtype):
        ser = Series(values, dtype=dtype)
        result = ser.quantile([0.5])
        expected = Series(np.asarray(ser)).quantile([0.5]).astype("Sparse[float]")
        tm.assert_series_equal(result, expected)

    def test_quantile_empty_float64(self):
        # floats
        ser = Series([], dtype="float64")

        res = ser.quantile(0.5)
        assert np.isnan(res)

        res = ser.quantile([0.5])
        exp = Series([np.nan], index=[0.5])
        tm.assert_series_equal(res, exp)

    def test_quantile_empty_int64(self):
        # int
        ser = Series([], dtype="int64")

        res = ser.quantile(0.5)
        assert np.isnan(res)

        res = ser.quantile([0.5])
        exp = Series([np.nan], index=[0.5])
        tm.assert_series_equal(res, exp)

    def test_quantile_empty_dt64(self):
        # datetime
        ser = Series([], dtype="datetime64[ns]")

        res = ser.quantile(0.5)
        assert res is pd.NaT

        res = ser.quantile([0.5])
        exp = Series([pd.NaT], index=[0.5], dtype=ser.dtype)
        tm.assert_series_equal(res, exp)

    @pytest.mark.parametrize("dtype", [int, float, "Int64"])
    def test_quantile_dtypes(self, dtype):
        result = Series([1, 2, 3], dtype=dtype).quantile(np.arange(0, 1, 0.25))
        expected = Series(np.arange(1, 3, 0.5), index=np.arange(0, 1, 0.25))
        if dtype == "Int64":
            expected = expected.astype("Float64")
        tm.assert_series_equal(result, expected)

    def test_quantile_all_na(self, any_int_ea_dtype):
        # GH#50681
        ser = Series([pd.NA, pd.NA], dtype=any_int_ea_dtype)
        with tm.assert_produces_warning(None):
            result = ser.quantile([0.1, 0.5])
        expected = Series([pd.NA, pd.NA], dtype=any_int_ea_dtype, index=[0.1, 0.5])
        tm.assert_series_equal(result, expected)

    def test_quantile_dtype_size(self, any_int_ea_dtype):
        # GH#50681
        ser = Series([pd.NA, pd.NA, 1], dtype=any_int_ea_dtype)
        result = ser.quantile([0.1, 0.5])
        expected = Series([1, 1], dtype=any_int_ea_dtype, index=[0.1, 0.5])
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_lagrange.py ===
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
                                    RigidBody, LagrangesMethod, Particle,
                                    inertia, Lagrangian)
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises


def test_invalid_coordinates():
    # Simple pendulum, but use symbol instead of dynamicsymbol
    l, m, g = symbols('l m g')
    q = symbols('q')  # Generalized coordinate
    N, O = ReferenceFrame('N'), Point('O')
    O.set_vel(N, 0)
    P = Particle('P', Point('P'), m)
    P.point.set_pos(O, l * (sin(q) * N.x - cos(q) * N.y))
    P.potential_energy = m * g * P.point.pos_from(O).dot(N.y)
    L = Lagrangian(N, P)
    raises(ValueError, lambda: LagrangesMethod(L, [q], bodies=P))


def test_disc_on_an_incline_plane():
    # Disc rolling on an inclined plane
    # First the generalized coordinates are created. The mass center of the
    # disc is located from top vertex of the inclined plane by the generalized
    # coordinate 'y'. The orientation of the disc is defined by the angle
    # 'theta'. The mass of the disc is 'm' and its radius is 'R'. The length of
    # the inclined path is 'l', the angle of inclination is 'alpha'. 'g' is the
    # gravitational constant.
    y, theta = dynamicsymbols('y theta')
    yd, thetad = dynamicsymbols('y theta', 1)
    m, g, R, l, alpha = symbols('m g R l alpha')

    # Next, we create the inertial reference frame 'N'. A reference frame 'A'
    # is attached to the inclined plane. Finally a frame is created which is attached to the disk.
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [pi/2 - alpha, N.z])
    B = A.orientnew('B', 'Axis', [-theta, A.z])

    # Creating the disc 'D'; we create the point that represents the mass
    # center of the disc and set its velocity. The inertia dyadic of the disc
    # is created. Finally, we create the disc.
    Do = Point('Do')
    Do.set_vel(N, yd * A.x)
    I = m * R**2/2 * B.z | B.z
    D = RigidBody('D', Do, B, m, (I, Do))

    # To construct the Lagrangian, 'L', of the disc, we determine its kinetic
    # and potential energies, T and U, respectively. L is defined as the
    # difference between T and U.
    D.potential_energy = m * g * (l - y) * sin(alpha)
    L = Lagrangian(N, D)

    # We then create the list of generalized coordinates and constraint
    # equations. The constraint arises due to the disc rolling without slip on
    # on the inclined path. We then invoke the 'LagrangesMethod' class and
    # supply it the necessary arguments and generate the equations of motion.
    # The'rhs' method solves for the q_double_dots (i.e. the second derivative
    # with respect to time  of the generalized coordinates and the lagrange
    # multipliers.
    q = [y, theta]
    hol_coneqs = [y - R * theta]
    m = LagrangesMethod(L, q, hol_coneqs=hol_coneqs)
    m.form_lagranges_equations()
    rhs = m.rhs()
    rhs.simplify()
    assert rhs[2] == 2*g*sin(alpha)/3


def test_simp_pen():
    # This tests that the equations generated by LagrangesMethod are identical
    # to those obtained by hand calculations. The system under consideration is
    # the simple pendulum.
    # We begin by creating the generalized coordinates as per the requirements
    # of LagrangesMethod. Also we created the associate symbols
    # that characterize the system: 'm' is the mass of the bob, l is the length
    # of the massless rigid rod connecting the bob to a point O fixed in the
    # inertial frame.
    q, u = dynamicsymbols('q u')
    qd, ud = dynamicsymbols('q u ', 1)
    l, m, g = symbols('l m g')

    # We then create the inertial frame and a frame attached to the massless
    # string following which we define the inertial angular velocity of the
    # string.
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q, N.z])
    A.set_ang_vel(N, qd * N.z)

    # Next, we create the point O and fix it in the inertial frame. We then
    # locate the point P to which the bob is attached. Its corresponding
    # velocity is then determined by the 'two point formula'.
    O = Point('O')
    O.set_vel(N, 0)
    P = O.locatenew('P', l * A.x)
    P.v2pt_theory(O, N, A)

    # The 'Particle' which represents the bob is then created and its
    # Lagrangian generated.
    Pa = Particle('Pa', P, m)
    Pa.potential_energy = - m * g * l * cos(q)
    L = Lagrangian(N, Pa)

    # The 'LagrangesMethod' class is invoked to obtain equations of motion.
    lm = LagrangesMethod(L, [q])
    lm.form_lagranges_equations()
    RHS = lm.rhs()
    assert RHS[1] == -g*sin(q)/l


def test_nonminimal_pendulum():
    q1, q2 = dynamicsymbols('q1:3')
    q1d, q2d = dynamicsymbols('q1:3', level=1)
    L, m, t = symbols('L, m, t')
    g = 9.8
    # Compose World Frame
    N = ReferenceFrame('N')
    pN = Point('N*')
    pN.set_vel(N, 0)
    # Create point P, the pendulum mass
    P = pN.locatenew('P1', q1*N.x + q2*N.y)
    P.set_vel(N, P.pos_from(pN).dt(N))
    pP = Particle('pP', P, m)
    # Constraint Equations
    f_c = Matrix([q1**2 + q2**2 - L**2])
    # Calculate the lagrangian, and form the equations of motion
    Lag = Lagrangian(N, pP)
    LM = LagrangesMethod(Lag, [q1, q2], hol_coneqs=f_c,
            forcelist=[(P, m*g*N.x)], frame=N)
    LM.form_lagranges_equations()
    # Check solution
    lam1 = LM.lam_vec[0, 0]
    eom_sol = Matrix([[m*Derivative(q1, t, t) - 9.8*m + 2*lam1*q1],
                      [m*Derivative(q2, t, t) + 2*lam1*q2]])
    assert LM.eom == eom_sol
    # Check multiplier solution
    lam_sol = Matrix([(19.6*q1 + 2*q1d**2 + 2*q2d**2)/(4*q1**2/m + 4*q2**2/m)])
    assert simplify(LM.solve_multipliers(sol_type='Matrix')) == simplify(lam_sol)


def test_dub_pen():

    # The system considered is the double pendulum. Like in the
    # test of the simple pendulum above, we begin by creating the generalized
    # coordinates and the simple generalized speeds and accelerations which
    # will be used later. Following this we create frames and points necessary
    # for the kinematics. The procedure isn't explicitly explained as this is
    # similar to the simple  pendulum. Also this is documented on the pydy.org
    # website.
    q1, q2 = dynamicsymbols('q1 q2')
    q1d, q2d = dynamicsymbols('q1 q2', 1)
    q1dd, q2dd = dynamicsymbols('q1 q2', 2)
    u1, u2 = dynamicsymbols('u1 u2')
    u1d, u2d = dynamicsymbols('u1 u2', 1)
    l, m, g = symbols('l m g')

    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = N.orientnew('B', 'Axis', [q2, N.z])

    A.set_ang_vel(N, q1d * A.z)
    B.set_ang_vel(N, q2d * A.z)

    O = Point('O')
    P = O.locatenew('P', l * A.x)
    R = P.locatenew('R', l * B.x)

    O.set_vel(N, 0)
    P.v2pt_theory(O, N, A)
    R.v2pt_theory(P, N, B)

    ParP = Particle('ParP', P, m)
    ParR = Particle('ParR', R, m)

    ParP.potential_energy = - m * g * l * cos(q1)
    ParR.potential_energy = - m * g * l * cos(q1) - m * g * l * cos(q2)
    L = Lagrangian(N, ParP, ParR)
    lm = LagrangesMethod(L, [q1, q2], bodies=[ParP, ParR])
    lm.form_lagranges_equations()

    assert simplify(l*m*(2*g*sin(q1) + l*sin(q1)*sin(q2)*q2dd
        + l*sin(q1)*cos(q2)*q2d**2 - l*sin(q2)*cos(q1)*q2d**2
        + l*cos(q1)*cos(q2)*q2dd + 2*l*q1dd) - lm.eom[0]) == 0
    assert simplify(l*m*(g*sin(q2) + l*sin(q1)*sin(q2)*q1dd
        - l*sin(q1)*cos(q2)*q1d**2 + l*sin(q2)*cos(q1)*q1d**2
        + l*cos(q1)*cos(q2)*q1dd + l*q2dd) - lm.eom[1]) == 0
    assert lm.bodies == [ParP, ParR]


def test_rolling_disc():
    # Rolling Disc Example
    # Here the rolling disc is formed from the contact point up, removing the
    # need to introduce generalized speeds. Only 3 configuration and 3
    # speed variables are need to describe this system, along with the
    # disc's mass and radius, and the local gravity.
    q1, q2, q3 = dynamicsymbols('q1 q2 q3')
    q1d, q2d, q3d = dynamicsymbols('q1 q2 q3', 1)
    r, m, g = symbols('r m g')

    # The kinematics are formed by a series of simple rotations. Each simple
    # rotation creates a new frame, and the next rotation is defined by the new
    # frame's basis vectors. This example uses a 3-1-2 series of rotations, or
    # Z, X, Y series of rotations. Angular velocity for this is defined using
    # the second frame's basis (the lean frame).
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])

    # This is the translational kinematics. We create a point with no velocity
    # in N; this is the contact point between the disc and ground. Next we form
    # the position vector from the contact point to the disc's center of mass.
    # Finally we form the velocity and acceleration of the disc.
    C = Point('C')
    C.set_vel(N, 0)
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)

    # Forming the inertia dyadic.
    I = inertia(L, m/4 * r**2, m/2 * r**2, m/4 * r**2)
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))

    # Finally we form the equations of motion, using the same steps we did
    # before. Supply the Lagrangian, the generalized speeds.
    BodyD.potential_energy = - m * g * r * cos(q2)
    Lag = Lagrangian(N, BodyD)
    q = [q1, q2, q3]
    q1 = Function('q1')
    q2 = Function('q2')
    q3 = Function('q3')
    l = LagrangesMethod(Lag, q)
    l.form_lagranges_equations()
    RHS = l.rhs()
    RHS.simplify()
    t = symbols('t')

    assert (l.mass_matrix[3:6] == [0, 5*m*r**2/4, 0])
    assert RHS[4].simplify() == (
        (-8*g*sin(q2(t)) + r*(5*sin(2*q2(t))*Derivative(q1(t), t) +
        12*cos(q2(t))*Derivative(q3(t), t))*Derivative(q1(t), t))/(10*r))
    assert RHS[5] == (-5*cos(q2(t))*Derivative(q1(t), t) + 6*tan(q2(t)
        )*Derivative(q3(t), t) + 4*Derivative(q1(t), t)/cos(q2(t))
        )*Derivative(q2(t), t)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_memmap.py ===
import mmap
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile

import pytest

from numpy import (
    add,
    allclose,
    arange,
    asarray,
    average,
    isscalar,
    memmap,
    multiply,
    ndarray,
    prod,
    subtract,
    sum,
)
from numpy.testing import (
    IS_PYPY,
    assert_,
    assert_array_equal,
    assert_equal,
    break_cycles,
    suppress_warnings,
)


class TestMemmap:
    def setup_method(self):
        self.tmpfp = NamedTemporaryFile(prefix='mmap')
        self.shape = (3, 4)
        self.dtype = 'float32'
        self.data = arange(12, dtype=self.dtype)
        self.data.resize(self.shape)

    def teardown_method(self):
        self.tmpfp.close()
        self.data = None
        if IS_PYPY:
            break_cycles()
            break_cycles()

    def test_roundtrip(self):
        # Write data to file
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp[:] = self.data[:]
        del fp  # Test __del__ machinery, which handles cleanup

        # Read data back from file
        newfp = memmap(self.tmpfp, dtype=self.dtype, mode='r',
                       shape=self.shape)
        assert_(allclose(self.data, newfp))
        assert_array_equal(self.data, newfp)
        assert_equal(newfp.flags.writeable, False)

    def test_open_with_filename(self, tmp_path):
        tmpname = tmp_path / 'mmap'
        fp = memmap(tmpname, dtype=self.dtype, mode='w+',
                       shape=self.shape)
        fp[:] = self.data[:]
        del fp

    def test_unnamed_file(self):
        with TemporaryFile() as f:
            fp = memmap(f, dtype=self.dtype, shape=self.shape)
            del fp

    def test_attributes(self):
        offset = 1
        mode = "w+"
        fp = memmap(self.tmpfp, dtype=self.dtype, mode=mode,
                    shape=self.shape, offset=offset)
        assert_equal(offset, fp.offset)
        assert_equal(mode, fp.mode)
        del fp

    def test_filename(self, tmp_path):
        tmpname = tmp_path / "mmap"
        fp = memmap(tmpname, dtype=self.dtype, mode='w+',
                       shape=self.shape)
        abspath = Path(os.path.abspath(tmpname))
        fp[:] = self.data[:]
        assert_equal(abspath, fp.filename)
        b = fp[:1]
        assert_equal(abspath, b.filename)
        del b
        del fp

    def test_path(self, tmp_path):
        tmpname = tmp_path / "mmap"
        fp = memmap(Path(tmpname), dtype=self.dtype, mode='w+',
                       shape=self.shape)
        # os.path.realpath does not resolve symlinks on Windows
        # see: https://bugs.python.org/issue9949
        # use Path.resolve, just as memmap class does internally
        abspath = str(Path(tmpname).resolve())
        fp[:] = self.data[:]
        assert_equal(abspath, str(fp.filename.resolve()))
        b = fp[:1]
        assert_equal(abspath, str(b.filename.resolve()))
        del b
        del fp

    def test_filename_fileobj(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode="w+",
                    shape=self.shape)
        assert_equal(fp.filename, self.tmpfp.name)

    @pytest.mark.skipif(sys.platform == 'gnu0',
                        reason="Known to fail on hurd")
    def test_flush(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp[:] = self.data[:]
        assert_equal(fp[0], self.data[0])
        fp.flush()

    def test_del(self):
        # Make sure a view does not delete the underlying mmap
        fp_base = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp_base[0] = 5
        fp_view = fp_base[0:1]
        assert_equal(fp_view[0], 5)
        del fp_view
        # Should still be able to access and assign values after
        # deleting the view
        assert_equal(fp_base[0], 5)
        fp_base[0] = 6
        assert_equal(fp_base[0], 6)

    def test_arithmetic_drops_references(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        tmp = (fp + 10)
        if isinstance(tmp, memmap):
            assert_(tmp._mmap is not fp._mmap)

    def test_indexing_drops_references(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        tmp = fp[(1, 2), (2, 3)]
        if isinstance(tmp, memmap):
            assert_(tmp._mmap is not fp._mmap)

    def test_slicing_keeps_references(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        assert_(fp[:2, :2]._mmap is fp._mmap)

    def test_view(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
        new1 = fp.view()
        new2 = new1.view()
        assert_(new1.base is fp)
        assert_(new2.base is fp)
        new_array = asarray(fp)
        assert_(new_array.base is fp)

    def test_ufunc_return_ndarray(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
        fp[:] = self.data

        with suppress_warnings() as sup:
            sup.filter(FutureWarning, "np.average currently does not preserve")
            for unary_op in [sum, average, prod]:
                result = unary_op(fp)
                assert_(isscalar(result))
                assert_(result.__class__ is self.data[0, 0].__class__)

                assert_(unary_op(fp, axis=0).__class__ is ndarray)
                assert_(unary_op(fp, axis=1).__class__ is ndarray)

        for binary_op in [add, subtract, multiply]:
            assert_(binary_op(fp, self.data).__class__ is ndarray)
            assert_(binary_op(self.data, fp).__class__ is ndarray)
            assert_(binary_op(fp, fp).__class__ is ndarray)

        fp += 1
        assert fp.__class__ is memmap
        add(fp, 1, out=fp)
        assert fp.__class__ is memmap

    def test_getitem(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
        fp[:] = self.data

        assert_(fp[1:, :-1].__class__ is memmap)
        # Fancy indexing returns a copy that is not memmapped
        assert_(fp[[0, 1]].__class__ is ndarray)

    def test_memmap_subclass(self):
        class MemmapSubClass(memmap):
            pass

        fp = MemmapSubClass(self.tmpfp, dtype=self.dtype, shape=self.shape)
        fp[:] = self.data

        # We keep previous behavior for subclasses of memmap, i.e. the
        # ufunc and __getitem__ output is never turned into a ndarray
        assert_(sum(fp, axis=0).__class__ is MemmapSubClass)
        assert_(sum(fp).__class__ is MemmapSubClass)
        assert_(fp[1:, :-1].__class__ is MemmapSubClass)
        assert fp[[0, 1]].__class__ is MemmapSubClass

    def test_mmap_offset_greater_than_allocation_granularity(self):
        size = 5 * mmap.ALLOCATIONGRANULARITY
        offset = mmap.ALLOCATIONGRANULARITY + 1
        fp = memmap(self.tmpfp, shape=size, mode='w+', offset=offset)
        assert_(fp.offset == offset)

    def test_empty_array_with_offset_multiple_of_allocation_granularity(self):
        self.tmpfp.write(b'a' * mmap.ALLOCATIONGRANULARITY)
        size = 0
        offset = mmap.ALLOCATIONGRANULARITY
        fp = memmap(self.tmpfp, shape=size, mode='w+', offset=offset)
        assert_equal(fp.offset, offset)

    def test_no_shape(self):
        self.tmpfp.write(b'a' * 16)
        mm = memmap(self.tmpfp, dtype='float64')
        assert_equal(mm.shape, (2,))

    def test_empty_array(self):
        # gh-12653
        with pytest.raises(ValueError, match='empty file'):
            memmap(self.tmpfp, shape=(0, 4), mode='r')

        # gh-27723
        # empty memmap works with mode in ('w+','r+')
        memmap(self.tmpfp, shape=(0, 4), mode='w+')

        # ok now the file is not empty
        memmap(self.tmpfp, shape=(0, 4), mode='w+')

    def test_shape_type(self):
        memmap(self.tmpfp, shape=3, mode='w+')
        memmap(self.tmpfp, shape=self.shape, mode='w+')
        memmap(self.tmpfp, shape=list(self.shape), mode='w+')
        memmap(self.tmpfp, shape=asarray(self.shape), mode='w+')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_scalar_methods.py ===
"""
Test the scalar constructors, which also do type-coercion
"""
import fractions
import platform
import types
from typing import Any

import pytest

import numpy as np
from numpy._core import sctypes
from numpy.testing import assert_equal, assert_raises


class TestAsIntegerRatio:
    # derived in part from the cpython test "test_floatasratio"

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    @pytest.mark.parametrize("f, ratio", [
        (0.875, (7, 8)),
        (-0.875, (-7, 8)),
        (0.0, (0, 1)),
        (11.5, (23, 2)),
        ])
    def test_small(self, ftype, f, ratio):
        assert_equal(ftype(f).as_integer_ratio(), ratio)

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    def test_simple_fractions(self, ftype):
        R = fractions.Fraction
        assert_equal(R(0, 1),
                     R(*ftype(0.0).as_integer_ratio()))
        assert_equal(R(5, 2),
                     R(*ftype(2.5).as_integer_ratio()))
        assert_equal(R(1, 2),
                     R(*ftype(0.5).as_integer_ratio()))
        assert_equal(R(-2100, 1),
                     R(*ftype(-2100.0).as_integer_ratio()))

    @pytest.mark.parametrize("ftype", [
        np.half, np.single, np.double, np.longdouble])
    def test_errors(self, ftype):
        assert_raises(OverflowError, ftype('inf').as_integer_ratio)
        assert_raises(OverflowError, ftype('-inf').as_integer_ratio)
        assert_raises(ValueError, ftype('nan').as_integer_ratio)

    def test_against_known_values(self):
        R = fractions.Fraction
        assert_equal(R(1075, 512),
                     R(*np.half(2.1).as_integer_ratio()))
        assert_equal(R(-1075, 512),
                     R(*np.half(-2.1).as_integer_ratio()))
        assert_equal(R(4404019, 2097152),
                     R(*np.single(2.1).as_integer_ratio()))
        assert_equal(R(-4404019, 2097152),
                     R(*np.single(-2.1).as_integer_ratio()))
        assert_equal(R(4728779608739021, 2251799813685248),
                     R(*np.double(2.1).as_integer_ratio()))
        assert_equal(R(-4728779608739021, 2251799813685248),
                     R(*np.double(-2.1).as_integer_ratio()))
        # longdouble is platform dependent

    @pytest.mark.parametrize("ftype, frac_vals, exp_vals", [
        # dtype test cases generated using hypothesis
        # first five generated cases per dtype
        (np.half, [0.0, 0.01154830649280303, 0.31082276347447274,
                   0.527350517124794, 0.8308562335072596],
                  [0, 1, 0, -8, 12]),
        (np.single, [0.0, 0.09248576989263226, 0.8160498218131407,
                     0.17389442853722373, 0.7956044195067877],
                    [0, 12, 10, 17, -26]),
        (np.double, [0.0, 0.031066908499895136, 0.5214135908877832,
                     0.45780736035689296, 0.5906586745934036],
                    [0, -801, 51, 194, -653]),
        pytest.param(
            np.longdouble,
            [0.0, 0.20492557202724854, 0.4277180662199366, 0.9888085019891495,
             0.9620175814461964],
            [0, -7400, 14266, -7822, -8721],
            marks=[
                pytest.mark.skipif(
                    np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double"),
                pytest.mark.skipif(
                    platform.machine().startswith("ppc"),
                    reason="IBM double double"),
            ]
        )
    ])
    def test_roundtrip(self, ftype, frac_vals, exp_vals):
        for frac, exp in zip(frac_vals, exp_vals):
            f = np.ldexp(ftype(frac), exp)
            assert f.dtype == ftype
            n, d = f.as_integer_ratio()

            try:
                nf = np.longdouble(n)
                df = np.longdouble(d)
                if not np.isfinite(df):
                    raise OverflowError
            except (OverflowError, RuntimeWarning):
                # the values may not fit in any float type
                pytest.skip("longdouble too small on this platform")

            assert_equal(nf / df, f, f"{n}/{d}")


class TestIsInteger:
    @pytest.mark.parametrize("str_value", ["inf", "nan"])
    @pytest.mark.parametrize("code", np.typecodes["Float"])
    def test_special(self, code: str, str_value: str) -> None:
        cls = np.dtype(code).type
        value = cls(str_value)
        assert not value.is_integer()

    @pytest.mark.parametrize(
        "code", np.typecodes["Float"] + np.typecodes["AllInteger"]
    )
    def test_true(self, code: str) -> None:
        float_array = np.arange(-5, 5).astype(code)
        for value in float_array:
            assert value.is_integer()

    @pytest.mark.parametrize("code", np.typecodes["Float"])
    def test_false(self, code: str) -> None:
        float_array = np.arange(-5, 5).astype(code)
        float_array *= 1.1
        for value in float_array:
            if value == 0:
                continue
            assert not value.is_integer()


class TestClassGetItem:
    @pytest.mark.parametrize("cls", [
        np.number,
        np.integer,
        np.inexact,
        np.unsignedinteger,
        np.signedinteger,
        np.floating,
    ])
    def test_abc(self, cls: type[np.number]) -> None:
        alias = cls[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    def test_abc_complexfloating(self) -> None:
        alias = np.complexfloating[Any, Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is np.complexfloating

    @pytest.mark.parametrize("arg_len", range(4))
    def test_abc_complexfloating_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len in (1, 2):
            assert np.complexfloating[arg_tup]
        else:
            match = f"Too {'few' if arg_len == 0 else 'many'} arguments"
            with pytest.raises(TypeError, match=match):
                np.complexfloating[arg_tup]

    @pytest.mark.parametrize("cls", [np.generic, np.flexible, np.character])
    def test_abc_non_numeric(self, cls: type[np.generic]) -> None:
        with pytest.raises(TypeError):
            cls[Any]

    @pytest.mark.parametrize("code", np.typecodes["All"])
    def test_concrete(self, code: str) -> None:
        cls = np.dtype(code).type
        with pytest.raises(TypeError):
            cls[Any]

    @pytest.mark.parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.number[arg_tup]
        else:
            with pytest.raises(TypeError):
                np.number[arg_tup]

    def test_subscript_scalar(self) -> None:
        assert np.number[Any]


class TestBitCount:
    # derived in part from the cpython test "test_bit_count"

    @pytest.mark.parametrize("itype", sctypes['int'] + sctypes['uint'])
    def test_small(self, itype):
        for a in range(max(np.iinfo(itype).min, 0), 128):
            msg = f"Smoke test for {itype}({a}).bit_count()"
            assert itype(a).bit_count() == a.bit_count(), msg

    def test_bit_count(self):
        for exp in [10, 17, 63]:
            a = 2**exp
            assert np.uint64(a).bit_count() == 1
            assert np.uint64(a - 1).bit_count() == exp
            assert np.uint64(a ^ 63).bit_count() == 7
            assert np.uint64((a - 1) ^ 510).bit_count() == exp - 8


class TestDevice:
    """
    Test scalar.device attribute and scalar.to_device() method.
    """
    scalars = [np.bool(True), np.int64(1), np.uint64(1), np.float64(1.0),
               np.complex128(1 + 1j)]

    @pytest.mark.parametrize("scalar", scalars)
    def test_device(self, scalar):
        assert scalar.device == "cpu"

    @pytest.mark.parametrize("scalar", scalars)
    def test_to_device(self, scalar):
        assert scalar.to_device("cpu") is scalar

    @pytest.mark.parametrize("scalar", scalars)
    def test___array_namespace__(self, scalar):
        assert scalar.__array_namespace__() is np


@pytest.mark.parametrize("scalar", [np.bool(True), np.int8(1), np.float64(1)])
def test_array_wrap(scalar):
    # Test scalars array wrap as long as it exists.  NumPy itself should
    # probably not use it, so it may not be necessary to keep it around.

    arr0d = np.array(3, dtype=np.int8)
    # Third argument not passed, None, or True "decays" to scalar.
    # (I don't think NumPy would pass `None`, but it seems clear to support)
    assert type(scalar.__array_wrap__(arr0d)) is np.int8
    assert type(scalar.__array_wrap__(arr0d, None, None)) is np.int8
    assert type(scalar.__array_wrap__(arr0d, None, True)) is np.int8

    # Otherwise, result should be the input
    assert scalar.__array_wrap__(arr0d, None, False) is arr0d

    # An old bug.  A non 0-d array cannot be converted to scalar:
    arr1d = np.array([3], dtype=np.int8)
    assert scalar.__array_wrap__(arr1d) is arr1d
    assert scalar.__array_wrap__(arr1d, None, True) is arr1d

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_sort_values.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestSeriesSortValues:
    def test_sort_values(self, datetime_series, using_copy_on_write):
        # check indexes are reordered corresponding with the values
        ser = Series([3, 2, 4, 1], ["A", "B", "C", "D"])
        expected = Series([1, 2, 3, 4], ["D", "B", "A", "C"])
        result = ser.sort_values()
        tm.assert_series_equal(expected, result)

        ts = datetime_series.copy()
        ts[:5] = np.nan
        vals = ts.values

        result = ts.sort_values()
        assert np.isnan(result[-5:]).all()
        tm.assert_numpy_array_equal(result[:-5].values, np.sort(vals[5:]))

        # na_position
        result = ts.sort_values(na_position="first")
        assert np.isnan(result[:5]).all()
        tm.assert_numpy_array_equal(result[5:].values, np.sort(vals[5:]))

        # something object-type
        ser = Series(["A", "B"], [1, 2])
        # no failure
        ser.sort_values()

        # ascending=False
        ordered = ts.sort_values(ascending=False)
        expected = np.sort(ts.dropna().values)[::-1]
        tm.assert_almost_equal(expected, ordered.dropna().values)
        ordered = ts.sort_values(ascending=False, na_position="first")
        tm.assert_almost_equal(expected, ordered.dropna().values)

        # ascending=[False] should behave the same as ascending=False
        ordered = ts.sort_values(ascending=[False])
        expected = ts.sort_values(ascending=False)
        tm.assert_series_equal(expected, ordered)
        ordered = ts.sort_values(ascending=[False], na_position="first")
        expected = ts.sort_values(ascending=False, na_position="first")
        tm.assert_series_equal(expected, ordered)

        msg = 'For argument "ascending" expected type bool, received type NoneType.'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=None)
        msg = r"Length of ascending \(0\) must be 1 for Series"
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=[])
        msg = r"Length of ascending \(3\) must be 1 for Series"
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=[1, 2, 3])
        msg = r"Length of ascending \(2\) must be 1 for Series"
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=[False, False])
        msg = 'For argument "ascending" expected type bool, received type str.'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending="foobar")

        # inplace=True
        ts = datetime_series.copy()
        return_value = ts.sort_values(ascending=False, inplace=True)
        assert return_value is None
        tm.assert_series_equal(ts, datetime_series.sort_values(ascending=False))
        tm.assert_index_equal(
            ts.index, datetime_series.sort_values(ascending=False).index
        )

        # GH#5856/5853
        # Series.sort_values operating on a view
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        s = df.iloc[:, 0]

        msg = (
            "This Series is a view of some other array, to sort in-place "
            "you must create a copy"
        )
        if using_copy_on_write:
            s.sort_values(inplace=True)
            tm.assert_series_equal(s, df.iloc[:, 0].sort_values())
        else:
            with pytest.raises(ValueError, match=msg):
                s.sort_values(inplace=True)

    def test_sort_values_categorical(self):
        c = Categorical(["a", "b", "b", "a"], ordered=False)
        cat = Series(c.copy())

        # sort in the categories order
        expected = Series(
            Categorical(["a", "a", "b", "b"], ordered=False), index=[0, 3, 1, 2]
        )
        result = cat.sort_values()
        tm.assert_series_equal(result, expected)

        cat = Series(Categorical(["a", "c", "b", "d"], ordered=True))
        res = cat.sort_values()
        exp = np.array(["a", "b", "c", "d"], dtype=np.object_)
        tm.assert_numpy_array_equal(res.__array__(), exp)

        cat = Series(
            Categorical(
                ["a", "c", "b", "d"], categories=["a", "b", "c", "d"], ordered=True
            )
        )
        res = cat.sort_values()
        exp = np.array(["a", "b", "c", "d"], dtype=np.object_)
        tm.assert_numpy_array_equal(res.__array__(), exp)

        res = cat.sort_values(ascending=False)
        exp = np.array(["d", "c", "b", "a"], dtype=np.object_)
        tm.assert_numpy_array_equal(res.__array__(), exp)

        raw_cat1 = Categorical(
            ["a", "b", "c", "d"], categories=["a", "b", "c", "d"], ordered=False
        )
        raw_cat2 = Categorical(
            ["a", "b", "c", "d"], categories=["d", "c", "b", "a"], ordered=True
        )
        s = ["a", "b", "c", "d"]
        df = DataFrame(
            {"unsort": raw_cat1, "sort": raw_cat2, "string": s, "values": [1, 2, 3, 4]}
        )

        # Cats must be sorted in a dataframe
        res = df.sort_values(by=["string"], ascending=False)
        exp = np.array(["d", "c", "b", "a"], dtype=np.object_)
        tm.assert_numpy_array_equal(res["sort"].values.__array__(), exp)
        assert res["sort"].dtype == "category"

        res = df.sort_values(by=["sort"], ascending=False)
        exp = df.sort_values(by=["string"], ascending=True)
        tm.assert_series_equal(res["values"], exp["values"])
        assert res["sort"].dtype == "category"
        assert res["unsort"].dtype == "category"

        # unordered cat, but we allow this
        df.sort_values(by=["unsort"], ascending=False)

        # multi-columns sort
        # GH#7848
        df = DataFrame(
            {"id": [6, 5, 4, 3, 2, 1], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
        )
        df["grade"] = Categorical(df["raw_grade"], ordered=True)
        df["grade"] = df["grade"].cat.set_categories(["b", "e", "a"])

        # sorts 'grade' according to the order of the categories
        result = df.sort_values(by=["grade"])
        expected = df.iloc[[1, 2, 5, 0, 3, 4]]
        tm.assert_frame_equal(result, expected)

        # multi
        result = df.sort_values(by=["grade", "id"])
        expected = df.iloc[[2, 1, 5, 4, 3, 0]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_list, sorted_list, ignore_index, output_index",
        [
            ([2, 3, 6, 1], [6, 3, 2, 1], True, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [6, 3, 2, 1], False, [2, 1, 0, 3]),
        ],
    )
    def test_sort_values_ignore_index(
        self, inplace, original_list, sorted_list, ignore_index, output_index
    ):
        # GH 30114
        ser = Series(original_list)
        expected = Series(sorted_list, index=output_index)
        kwargs = {"ignore_index": ignore_index, "inplace": inplace}

        if inplace:
            result_ser = ser.copy()
            result_ser.sort_values(ascending=False, **kwargs)
        else:
            result_ser = ser.sort_values(ascending=False, **kwargs)

        tm.assert_series_equal(result_ser, expected)
        tm.assert_series_equal(ser, Series(original_list))

    def test_mergesort_descending_stability(self):
        # GH 28697
        s = Series([1, 2, 1, 3], ["first", "b", "second", "c"])
        result = s.sort_values(ascending=False, kind="mergesort")
        expected = Series([3, 2, 1, 1], ["c", "b", "first", "second"])
        tm.assert_series_equal(result, expected)

    def test_sort_values_validate_ascending_for_value_error(self):
        # GH41634
        ser = Series([23, 7, 21])

        msg = 'For argument "ascending" expected type bool, received type str.'
        with pytest.raises(ValueError, match=msg):
            ser.sort_values(ascending="False")

    @pytest.mark.parametrize("ascending", [False, 0, 1, True])
    def test_sort_values_validate_ascending_functional(self, ascending):
        # GH41634
        ser = Series([23, 7, 21])
        expected = np.sort(ser.values)

        sorted_ser = ser.sort_values(ascending=ascending)
        if not ascending:
            expected = expected[::-1]

        result = sorted_ser.values
        tm.assert_numpy_array_equal(result, expected)


class TestSeriesSortingKey:
    def test_sort_values_key(self):
        series = Series(np.array(["Hello", "goodbye"]))

        result = series.sort_values(axis=0)
        expected = series
        tm.assert_series_equal(result, expected)

        result = series.sort_values(axis=0, key=lambda x: x.str.lower())
        expected = series[::-1]
        tm.assert_series_equal(result, expected)

    def test_sort_values_key_nan(self):
        series = Series(np.array([0, 5, np.nan, 3, 2, np.nan]))

        result = series.sort_values(axis=0)
        expected = series.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_series_equal(result, expected)

        result = series.sort_values(axis=0, key=lambda x: x + 5)
        expected = series.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_series_equal(result, expected)

        result = series.sort_values(axis=0, key=lambda x: -x, ascending=False)
        expected = series.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_lambdarepr.py ===
from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises

from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.lambdarepr import lambdarepr, LambdaPrinter, NumExprPrinter


x, y, z = symbols("x,y,z")
i, a, b = symbols("i,a,b")
j, c, d = symbols("j,c,d")


def test_basic():
    assert lambdarepr(x*y) == "x*y"
    assert lambdarepr(x + y) in ["y + x", "x + y"]
    assert lambdarepr(x**y) == "x**y"


def test_matrix():
    # Test printing a Matrix that has an element that is printed differently
    # with the LambdaPrinter than with the StrPrinter.
    e = x % 2
    assert lambdarepr(e) != str(e)
    assert lambdarepr(Matrix([e])) == 'ImmutableDenseMatrix([[x % 2]])'


def test_piecewise():
    # In each case, test eval() the lambdarepr() to make sure there are a
    # correct number of parentheses. It will give a SyntaxError if there aren't.

    h = "lambda x: "

    p = Piecewise((x, x < 0))
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((x) if (x < 0) else None)"

    p = Piecewise(
        (1, x < 1),
        (2, x < 2),
        (0, True)
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((1) if (x < 1) else (2) if (x < 2) else (0))"

    p = Piecewise(
        (1, x < 1),
        (2, x < 2),
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((1) if (x < 1) else (2) if (x < 2) else None)"

    p = Piecewise(
        (x, x < 1),
        (x**2, Interval(3, 4, True, False).contains(x)),
        (0, True),
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((x) if (x < 1) else (x**2) if (((x <= 4)) and ((x > 3))) else (0))"

    p = Piecewise(
        (x**2, x < 0),
        (x, x < 1),
        (2 - x, x >= 1),
        (0, True), evaluate=False
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((x**2) if (x < 0) else (x) if (x < 1)"\
                                " else (2 - x) if (x >= 1) else (0))"

    p = Piecewise(
        (x**2, x < 0),
        (x, x < 1),
        (2 - x, x >= 1), evaluate=False
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((x**2) if (x < 0) else (x) if (x < 1)"\
                    " else (2 - x) if (x >= 1) else None)"

    p = Piecewise(
        (1, x >= 1),
        (2, x >= 2),
        (3, x >= 3),
        (4, x >= 4),
        (5, x >= 5),
        (6, True)
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((1) if (x >= 1) else (2) if (x >= 2) else (3) if (x >= 3)"\
                        " else (4) if (x >= 4) else (5) if (x >= 5) else (6))"

    p = Piecewise(
        (1, x <= 1),
        (2, x <= 2),
        (3, x <= 3),
        (4, x <= 4),
        (5, x <= 5),
        (6, True)
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((1) if (x <= 1) else (2) if (x <= 2) else (3) if (x <= 3)"\
                            " else (4) if (x <= 4) else (5) if (x <= 5) else (6))"

    p = Piecewise(
        (1, x > 1),
        (2, x > 2),
        (3, x > 3),
        (4, x > 4),
        (5, x > 5),
        (6, True)
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l =="((1) if (x > 1) else (2) if (x > 2) else (3) if (x > 3)"\
                            " else (4) if (x > 4) else (5) if (x > 5) else (6))"

    p = Piecewise(
        (1, x < 1),
        (2, x < 2),
        (3, x < 3),
        (4, x < 4),
        (5, x < 5),
        (6, True)
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((1) if (x < 1) else (2) if (x < 2) else (3) if (x < 3)"\
                            " else (4) if (x < 4) else (5) if (x < 5) else (6))"

    p = Piecewise(
        (Piecewise(
            (1, x > 0),
            (2, True)
        ), y > 0),
        (3, True)
    )
    l = lambdarepr(p)
    eval(h + l)
    assert l == "((((1) if (x > 0) else (2))) if (y > 0) else (3))"


def test_sum__1():
    # In each case, test eval() the lambdarepr() to make sure that
    # it evaluates to the same results as the symbolic expression
    s = Sum(x ** i, (i, a, b))
    l = lambdarepr(s)
    assert l == "(builtins.sum(x**i for i in range(a, b+1)))"

    args = x, a, b
    f = lambdify(args, s)
    v = 2, 3, 8
    assert f(*v) == s.subs(zip(args, v)).doit()

def test_sum__2():
    s = Sum(i * x, (i, a, b))
    l = lambdarepr(s)
    assert l == "(builtins.sum(i*x for i in range(a, b+1)))"

    args = x, a, b
    f = lambdify(args, s)
    v = 2, 3, 8
    assert f(*v) == s.subs(zip(args, v)).doit()


def test_multiple_sums():
    s = Sum(i * x + j, (i, a, b), (j, c, d))

    l = lambdarepr(s)
    assert l == "(builtins.sum(i*x + j for j in range(c, d+1) for i in range(a, b+1)))"

    args = x, a, b, c, d
    f = lambdify(args, s)
    vals = 2, 3, 4, 5, 6
    f_ref = s.subs(zip(args, vals)).doit()
    f_res = f(*vals)
    assert f_res == f_ref


def test_sqrt():
    prntr = LambdaPrinter({'standard' : 'python3'})
    assert prntr._print_Pow(sqrt(x), rational=False) == 'sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'


def test_settings():
    raises(TypeError, lambda: lambdarepr(sin(x), method="garbage"))


def test_numexpr():
    # test ITE rewrite as Piecewise
    from sympy.logic.boolalg import ITE
    expr = ITE(x > 0, True, False, evaluate=False)
    assert NumExprPrinter().doprint(expr) == \
           "numexpr.evaluate('where((x > 0), True, False)', truediv=True)"

    from sympy.codegen.ast import Return, FunctionDefinition, Variable, Assignment
    func_def = FunctionDefinition(None, 'foo', [Variable(x)], [Assignment(y,x), Return(y**2)])
    expected = "def foo(x):\n"\
               "    y = numexpr.evaluate('x', truediv=True)\n"\
               "    return numexpr.evaluate('y**2', truediv=True)"
    assert NumExprPrinter().doprint(func_def) == expected


class CustomPrintedObject(Expr):
    def _lambdacode(self, printer):
        return 'lambda'

    def _tensorflowcode(self, printer):
        return 'tensorflow'

    def _numpycode(self, printer):
        return 'numpy'

    def _numexprcode(self, printer):
        return 'numexpr'

    def _mpmathcode(self, printer):
        return 'mpmath'


def test_printmethod():
    # In each case, printmethod is called to test
    # its working

    obj = CustomPrintedObject()
    assert LambdaPrinter().doprint(obj) == 'lambda'
    assert TensorflowPrinter().doprint(obj) == 'tensorflow'
    assert NumExprPrinter().doprint(obj) == "numexpr.evaluate('numexpr', truediv=True)"

    assert NumExprPrinter().doprint(Piecewise((y, x >= 0), (z, x < 0))) == \
            "numexpr.evaluate('where((x >= 0), y, z)', truediv=True)"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\test_construction.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
    Int8Dtype,
    Int32Dtype,
    Int64Dtype,
)


@pytest.fixture(params=[pd.array, IntegerArray._from_sequence])
def constructor(request):
    """Fixture returning parametrized IntegerArray from given sequence.

    Used to test dtype conversions.
    """
    return request.param


def test_uses_pandas_na():
    a = pd.array([1, None], dtype=Int64Dtype())
    assert a[1] is pd.NA


def test_from_dtype_from_float(data):
    # construct from our dtype & string dtype
    dtype = data.dtype

    # from float
    expected = pd.Series(data)
    result = pd.Series(data.to_numpy(na_value=np.nan, dtype="float"), dtype=str(dtype))
    tm.assert_series_equal(result, expected)

    # from int / list
    expected = pd.Series(data)
    result = pd.Series(np.array(data).tolist(), dtype=str(dtype))
    tm.assert_series_equal(result, expected)

    # from int / array
    expected = pd.Series(data).dropna().reset_index(drop=True)
    dropped = np.array(data.dropna()).astype(np.dtype(dtype.type))
    result = pd.Series(dropped, dtype=str(dtype))
    tm.assert_series_equal(result, expected)


def test_conversions(data_missing):
    # astype to object series
    df = pd.DataFrame({"A": data_missing})
    result = df["A"].astype("object")
    expected = pd.Series(np.array([pd.NA, 1], dtype=object), name="A")
    tm.assert_series_equal(result, expected)

    # convert to object ndarray
    # we assert that we are exactly equal
    # including type conversions of scalars
    result = df["A"].astype("object").values
    expected = np.array([pd.NA, 1], dtype=object)
    tm.assert_numpy_array_equal(result, expected)

    for r, e in zip(result, expected):
        if pd.isnull(r):
            assert pd.isnull(e)
        elif is_integer(r):
            assert r == e
            assert is_integer(e)
        else:
            assert r == e
            assert type(r) == type(e)


def test_integer_array_constructor():
    values = np.array([1, 2, 3, 4], dtype="int64")
    mask = np.array([False, False, False, True], dtype="bool")

    result = IntegerArray(values, mask)
    expected = pd.array([1, 2, 3, np.nan], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    msg = r".* should be .* numpy array. Use the 'pd.array' function instead"
    with pytest.raises(TypeError, match=msg):
        IntegerArray(values.tolist(), mask)

    with pytest.raises(TypeError, match=msg):
        IntegerArray(values, mask.tolist())

    with pytest.raises(TypeError, match=msg):
        IntegerArray(values.astype(float), mask)
    msg = r"__init__\(\) missing 1 required positional argument: 'mask'"
    with pytest.raises(TypeError, match=msg):
        IntegerArray(values)


def test_integer_array_constructor_copy():
    values = np.array([1, 2, 3, 4], dtype="int64")
    mask = np.array([False, False, False, True], dtype="bool")

    result = IntegerArray(values, mask)
    assert result._data is values
    assert result._mask is mask

    result = IntegerArray(values, mask, copy=True)
    assert result._data is not values
    assert result._mask is not mask


@pytest.mark.parametrize(
    "a, b",
    [
        ([1, None], [1, np.nan]),
        ([None], [np.nan]),
        ([None, np.nan], [np.nan, np.nan]),
        ([np.nan, np.nan], [np.nan, np.nan]),
    ],
)
def test_to_integer_array_none_is_nan(a, b):
    result = pd.array(a, dtype="Int64")
    expected = pd.array(b, dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        ["foo", "bar"],
        "foo",
        1,
        1.0,
        pd.date_range("20130101", periods=2),
        np.array(["foo"]),
        [[1, 2], [3, 4]],
        [np.nan, {"a": 1}],
    ],
)
def test_to_integer_array_error(values):
    # error in converting existing arrays to IntegerArrays
    msg = "|".join(
        [
            r"cannot be converted to IntegerDtype",
            r"invalid literal for int\(\) with base 10:",
            r"values must be a 1D list-like",
            r"Cannot pass scalar",
            r"int\(\) argument must be a string",
        ]
    )
    with pytest.raises((ValueError, TypeError), match=msg):
        pd.array(values, dtype="Int64")

    with pytest.raises((ValueError, TypeError), match=msg):
        IntegerArray._from_sequence(values)


def test_to_integer_array_inferred_dtype(constructor):
    # if values has dtype -> respect it
    result = constructor(np.array([1, 2], dtype="int8"))
    assert result.dtype == Int8Dtype()
    result = constructor(np.array([1, 2], dtype="int32"))
    assert result.dtype == Int32Dtype()

    # if values have no dtype -> always int64
    result = constructor([1, 2])
    assert result.dtype == Int64Dtype()


def test_to_integer_array_dtype_keyword(constructor):
    result = constructor([1, 2], dtype="Int8")
    assert result.dtype == Int8Dtype()

    # if values has dtype -> override it
    result = constructor(np.array([1, 2], dtype="int8"), dtype="Int32")
    assert result.dtype == Int32Dtype()


def test_to_integer_array_float():
    result = IntegerArray._from_sequence([1.0, 2.0], dtype="Int64")
    expected = pd.array([1, 2], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    with pytest.raises(TypeError, match="cannot safely cast non-equivalent"):
        IntegerArray._from_sequence([1.5, 2.0], dtype="Int64")

    # for float dtypes, the itemsize is not preserved
    result = IntegerArray._from_sequence(
        np.array([1.0, 2.0], dtype="float32"), dtype="Int64"
    )
    assert result.dtype == Int64Dtype()


def test_to_integer_array_str():
    result = IntegerArray._from_sequence(["1", "2", None], dtype="Int64")
    expected = pd.array([1, 2, np.nan], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    with pytest.raises(
        ValueError, match=r"invalid literal for int\(\) with base 10: .*"
    ):
        IntegerArray._from_sequence(["1", "2", ""], dtype="Int64")

    with pytest.raises(
        ValueError, match=r"invalid literal for int\(\) with base 10: .*"
    ):
        IntegerArray._from_sequence(["1.5", "2.0"], dtype="Int64")


@pytest.mark.parametrize(
    "bool_values, int_values, target_dtype, expected_dtype",
    [
        ([False, True], [0, 1], Int64Dtype(), Int64Dtype()),
        ([False, True], [0, 1], "Int64", Int64Dtype()),
        ([False, True, np.nan], [0, 1, np.nan], Int64Dtype(), Int64Dtype()),
    ],
)
def test_to_integer_array_bool(
    constructor, bool_values, int_values, target_dtype, expected_dtype
):
    result = constructor(bool_values, dtype=target_dtype)
    assert result.dtype == expected_dtype
    expected = pd.array(int_values, dtype=target_dtype)
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "values, to_dtype, result_dtype",
    [
        (np.array([1], dtype="int64"), None, Int64Dtype),
        (np.array([1, np.nan]), None, Int64Dtype),
        (np.array([1, np.nan]), "int8", Int8Dtype),
    ],
)
def test_to_integer_array(values, to_dtype, result_dtype):
    # convert existing arrays to IntegerArrays
    result = IntegerArray._from_sequence(values, dtype=to_dtype)
    assert result.dtype == result_dtype()
    expected = pd.array(values, dtype=result_dtype())
    tm.assert_extension_array_equal(result, expected)


def test_integer_array_from_boolean():
    # GH31104
    expected = pd.array(np.array([True, False]), dtype="Int64")
    result = pd.array(np.array([True, False], dtype=object), dtype="Int64")
    tm.assert_extension_array_equal(result, expected)