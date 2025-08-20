
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_circuitutils.py ===
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import X, Y, Z, H, CNOT, CGate
from sympy.physics.quantum.identitysearch import bfs_identity_search
from sympy.physics.quantum.circuitutils import (kmp_table, find_subcircuit,
        replace_subcircuit, convert_to_symbolic_indices,
        convert_to_real_indices, random_reduce, random_insert,
        flatten_ids)
from sympy.testing.pytest import slow


def create_gate_sequence(qubit=0):
    gates = (X(qubit), Y(qubit), Z(qubit), H(qubit))
    return gates


def test_kmp_table():
    word = ('a', 'b', 'c', 'd', 'a', 'b', 'd')
    expected_table = [-1, 0, 0, 0, 0, 1, 2]
    assert expected_table == kmp_table(word)

    word = ('P', 'A', 'R', 'T', 'I', 'C', 'I', 'P', 'A', 'T', 'E', ' ',
            'I', 'N', ' ', 'P', 'A', 'R', 'A', 'C', 'H', 'U', 'T', 'E')
    expected_table = [-1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0,
                      0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0]
    assert expected_table == kmp_table(word)

    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    word = (x, y, y, x, z)
    expected_table = [-1, 0, 0, 0, 1]
    assert expected_table == kmp_table(word)

    word = (x, x, y, h, z)
    expected_table = [-1, 0, 1, 0, 0]
    assert expected_table == kmp_table(word)


def test_find_subcircuit():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    x1 = X(1)
    y1 = Y(1)

    i0 = Symbol('i0')
    x_i0 = X(i0)
    y_i0 = Y(i0)
    z_i0 = Z(i0)
    h_i0 = H(i0)

    circuit = (x, y, z)

    assert find_subcircuit(circuit, (x,)) == 0
    assert find_subcircuit(circuit, (x1,)) == -1
    assert find_subcircuit(circuit, (y,)) == 1
    assert find_subcircuit(circuit, (h,)) == -1
    assert find_subcircuit(circuit, Mul(x, h)) == -1
    assert find_subcircuit(circuit, Mul(x, y, z)) == 0
    assert find_subcircuit(circuit, Mul(y, z)) == 1
    assert find_subcircuit(Mul(*circuit), (x, y, z, h)) == -1
    assert find_subcircuit(Mul(*circuit), (z, y, x)) == -1
    assert find_subcircuit(circuit, (x,), start=2, end=1) == -1

    circuit = (x, y, x, y, z)
    assert find_subcircuit(Mul(*circuit), Mul(x, y, z)) == 2
    assert find_subcircuit(circuit, (x,), start=1) == 2
    assert find_subcircuit(circuit, (x, y), start=1, end=2) == -1
    assert find_subcircuit(Mul(*circuit), (x, y), start=1, end=3) == -1
    assert find_subcircuit(circuit, (x, y), start=1, end=4) == 2
    assert find_subcircuit(circuit, (x, y), start=2, end=4) == 2

    circuit = (x, y, z, x1, x, y, z, h, x, y, x1,
               x, y, z, h, y1, h)
    assert find_subcircuit(circuit, (x, y, z, h, y1)) == 11

    circuit = (x, y, x_i0, y_i0, z_i0, z)
    assert find_subcircuit(circuit, (x_i0, y_i0, z_i0)) == 2

    circuit = (x_i0, y_i0, z_i0, x_i0, y_i0, h_i0)
    subcircuit = (x_i0, y_i0, z_i0)
    result = find_subcircuit(circuit, subcircuit)
    assert result == 0


def test_replace_subcircuit():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))

    # Standard cases
    circuit = (z, y, x, x)
    remove = (z, y, x)
    assert replace_subcircuit(circuit, Mul(*remove)) == (x,)
    assert replace_subcircuit(circuit, remove + (x,)) == ()
    assert replace_subcircuit(circuit, remove, pos=1) == circuit
    assert replace_subcircuit(circuit, remove, pos=0) == (x,)
    assert replace_subcircuit(circuit, (x, x), pos=2) == (z, y)
    assert replace_subcircuit(circuit, (h,)) == circuit

    circuit = (x, y, x, y, z)
    remove = (x, y, z)
    assert replace_subcircuit(Mul(*circuit), Mul(*remove)) == (x, y)
    remove = (x, y, x, y)
    assert replace_subcircuit(circuit, remove) == (z,)

    circuit = (x, h, cgate_z, h, cnot)
    remove = (x, h, cgate_z)
    assert replace_subcircuit(circuit, Mul(*remove), pos=-1) == (h, cnot)
    assert replace_subcircuit(circuit, remove, pos=1) == circuit
    remove = (h, h)
    assert replace_subcircuit(circuit, remove) == circuit
    remove = (h, cgate_z, h, cnot)
    assert replace_subcircuit(circuit, remove) == (x,)

    replace = (h, x)
    actual = replace_subcircuit(circuit, remove,
                     replace=replace)
    assert actual == (x, h, x)

    circuit = (x, y, h, x, y, z)
    remove = (x, y)
    replace = (cnot, cgate_z)
    actual = replace_subcircuit(circuit, remove,
                     replace=Mul(*replace))
    assert actual == (cnot, cgate_z, h, x, y, z)

    actual = replace_subcircuit(circuit, remove,
                     replace=replace, pos=1)
    assert actual == (x, y, h, cnot, cgate_z, z)


def test_convert_to_symbolic_indices():
    (x, y, z, h) = create_gate_sequence()

    i0 = Symbol('i0')
    exp_map = {i0: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x,))
    assert actual == (X(i0),)
    assert act_map == exp_map

    expected = (X(i0), Y(i0), Z(i0), H(i0))
    exp_map = {i0: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x, y, z, h))
    assert actual == expected
    assert exp_map == act_map

    (x1, y1, z1, h1) = create_gate_sequence(1)
    i1 = Symbol('i1')

    expected = (X(i0), Y(i0), Z(i0), H(i0))
    exp_map = {i0: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x1, y1, z1, h1))
    assert actual == expected
    assert act_map == exp_map

    expected = (X(i0), Y(i0), Z(i0), H(i0), X(i1), Y(i1), Z(i1), H(i1))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x, y, z, h,
                                         x1, y1, z1, h1))
    assert actual == expected
    assert act_map == exp_map

    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(Mul(x1, y1,
                                         z1, h1, x, y, z, h))
    assert actual == expected
    assert act_map == exp_map

    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1), H(i0), H(i1))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(Mul(x, x1,
                                         y, y1, z, z1, h, h1))
    assert actual == expected
    assert act_map == exp_map

    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x1, x, y1, y,
                                         z1, z, h1, h))
    assert actual == expected
    assert act_map == exp_map

    cnot_10 = CNOT(1, 0)
    cnot_01 = CNOT(0, 1)
    cgate_z_10 = CGate(1, Z(0))
    cgate_z_01 = CGate(0, Z(1))

    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1),
                H(i0), H(i1), CNOT(i1, i0), CNOT(i0, i1),
                CGate(i1, Z(i0)), CGate(i0, Z(i1)))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    args = (x, x1, y, y1, z, z1, h, h1, cnot_10, cnot_01,
            cgate_z_10, cgate_z_01)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    args = (x1, x, y1, y, z1, z, h1, h, cnot_10, cnot_01,
            cgate_z_10, cgate_z_01)
    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1),
                H(i0), H(i1), CNOT(i0, i1), CNOT(i1, i0),
                CGate(i0, Z(i1)), CGate(i1, Z(i0)))
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    args = (cnot_10, h, cgate_z_01, h)
    expected = (CNOT(i0, i1), H(i1), CGate(i1, Z(i0)), H(i1))
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    args = (cnot_01, h1, cgate_z_10, h1)
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    args = (cnot_10, h1, cgate_z_01, h1)
    expected = (CNOT(i0, i1), H(i0), CGate(i1, Z(i0)), H(i0))
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    i2 = Symbol('i2')
    ccgate_z = CGate(0, CGate(1, Z(2)))
    ccgate_x = CGate(1, CGate(2, X(0)))
    args = (ccgate_z, ccgate_x)

    expected = (CGate(i0, CGate(i1, Z(i2))), CGate(i1, CGate(i2, X(i0))))
    exp_map = {i0: Integer(0), i1: Integer(1), i2: Integer(2)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    ndx_map = {i0: Integer(0)}
    index_gen = numbered_symbols(prefix='i', start=1)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args,
                                         qubit_map=ndx_map,
                                         start=i0,
                                         gen=index_gen)
    assert actual == expected
    assert act_map == exp_map

    i3 = Symbol('i3')
    cgate_x0_c321 = CGate((3, 2, 1), X(0))
    exp_map = {i0: Integer(3), i1: Integer(2),
               i2: Integer(1), i3: Integer(0)}
    expected = (CGate((i0, i1, i2), X(i3)),)
    args = (cgate_x0_c321,)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map


def test_convert_to_real_indices():
    i0 = Symbol('i0')
    i1 = Symbol('i1')

    (x, y, z, h) = create_gate_sequence()

    x_i0 = X(i0)
    y_i0 = Y(i0)
    z_i0 = Z(i0)

    qubit_map = {i0: 0}
    args = (z_i0, y_i0, x_i0)
    expected = (z, y, x)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    cnot_10 = CNOT(1, 0)
    cnot_01 = CNOT(0, 1)
    cgate_z_10 = CGate(1, Z(0))
    cgate_z_01 = CGate(0, Z(1))

    cnot_i1_i0 = CNOT(i1, i0)
    cnot_i0_i1 = CNOT(i0, i1)
    cgate_z_i1_i0 = CGate(i1, Z(i0))

    qubit_map = {i0: 0, i1: 1}
    args = (cnot_i1_i0,)
    expected = (cnot_10,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    args = (cgate_z_i1_i0,)
    expected = (cgate_z_10,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    args = (cnot_i0_i1,)
    expected = (cnot_01,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    qubit_map = {i0: 1, i1: 0}
    args = (cgate_z_i1_i0,)
    expected = (cgate_z_01,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    i2 = Symbol('i2')
    ccgate_z = CGate(i0, CGate(i1, Z(i2)))
    ccgate_x = CGate(i1, CGate(i2, X(i0)))

    qubit_map = {i0: 0, i1: 1, i2: 2}
    args = (ccgate_z, ccgate_x)
    expected = (CGate(0, CGate(1, Z(2))), CGate(1, CGate(2, X(0))))
    actual = convert_to_real_indices(Mul(*args), qubit_map)
    assert actual == expected

    qubit_map = {i0: 1, i2: 0, i1: 2}
    args = (ccgate_x, ccgate_z)
    expected = (CGate(2, CGate(0, X(1))), CGate(1, CGate(2, Z(0))))
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected


@slow
def test_random_reduce():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))

    gate_list = [x, y, z]
    ids = list(bfs_identity_search(gate_list, 1, max_depth=4))

    circuit = (x, y, h, z, cnot)
    assert random_reduce(circuit, []) == circuit
    assert random_reduce(circuit, ids) == circuit

    seq = [2, 11, 9, 3, 5]
    circuit = (x, y, z, x, y, h)
    assert random_reduce(circuit, ids, seed=seq) == (x, y, h)

    circuit = (x, x, y, y, z, z)
    assert random_reduce(circuit, ids, seed=seq) == (x, x, y, y)

    seq = [14, 13, 0]
    assert random_reduce(circuit, ids, seed=seq) == (y, y, z, z)

    gate_list = [x, y, z, h, cnot, cgate_z]
    ids = list(bfs_identity_search(gate_list, 2, max_depth=4))

    seq = [25]
    circuit = (x, y, z, y, h, y, h, cgate_z, h, cnot)
    expected = (x, y, z, cgate_z, h, cnot)
    assert random_reduce(circuit, ids, seed=seq) == expected
    circuit = Mul(*circuit)
    assert random_reduce(circuit, ids, seed=seq) == expected


@slow
def test_random_insert():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))

    choices = [(x, x)]
    circuit = (y, y)
    loc, choice = 0, 0
    actual = random_insert(circuit, choices, seed=[loc, choice])
    assert actual == (x, x, y, y)

    circuit = (x, y, z, h)
    choices = [(h, h), (x, y, z)]
    expected = (x, x, y, z, y, z, h)
    loc, choice = 1, 1
    actual = random_insert(circuit, choices, seed=[loc, choice])
    assert actual == expected

    gate_list = [x, y, z, h, cnot, cgate_z]
    ids = list(bfs_identity_search(gate_list, 2, max_depth=4))

    eq_ids = flatten_ids(ids)

    circuit = (x, y, h, cnot, cgate_z)
    expected = (x, z, x, z, x, y, h, cnot, cgate_z)
    loc, choice = 1, 30
    actual = random_insert(circuit, eq_ids, seed=[loc, choice])
    assert actual == expected
    circuit = Mul(*circuit)
    actual = random_insert(circuit, eq_ids, seed=[loc, choice])
    assert actual == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_codegen_rust.py ===
from io import StringIO

from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy


x, y, z = symbols('x,y,z')


def test_empty_rust_code():
    code_gen = RustCodeGen()
    output = StringIO()
    code_gen.dump_rs([], output, "file", header=False, empty=False)
    source = output.getvalue()
    assert source == ""


def test_simple_rust_code():
    name_expr = ("test", (x + y)*z)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    assert result[0] == "test.rs"
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64, z: f64) -> f64 {\n"
        "    let out1 = z*(x + y);\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected


def test_simple_code_with_header():
    name_expr = ("test", (x + y)*z)
    result, = codegen(name_expr, "Rust", header=True, empty=False)
    assert result[0] == "test.rs"
    source = result[1]
    version_str = "Code generated with SymPy %s" % sympy.__version__
    version_line = version_str.center(76).rstrip()
    expected = (
        "/*\n"
        " *%(version_line)s\n"
        " *\n"
        " *              See http://www.sympy.org/ for more information.\n"
        " *\n"
        " *                       This file is part of 'project'\n"
        " */\n"
        "fn test(x: f64, y: f64, z: f64) -> f64 {\n"
        "    let out1 = z*(x + y);\n"
        "    out1\n"
        "}\n"
    ) % {'version_line': version_line}
    assert source == expected


def test_simple_code_nameout():
    expr = Equality(z, (x + y))
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64) -> f64 {\n"
        "    let z = x + y;\n"
        "    z\n"
        "}\n"
    )
    assert source == expected


def test_numbersymbol():
    name_expr = ("test", pi**Catalan)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test() -> f64 {\n"
        "    const Catalan: f64 = %s;\n"
        "    let out1 = PI.powf(Catalan);\n"
        "    out1\n"
        "}\n"
    ) % Catalan.evalf(17)
    assert source == expected


@XFAIL
def test_numbersymbol_inline():
    # FIXME: how to pass inline to the RustCodePrinter?
    name_expr = ("test", [pi**Catalan, EulerGamma])
    result, = codegen(name_expr, "Rust", header=False,
                      empty=False, inline=True)
    source = result[1]
    expected = (
        "fn test() -> (f64, f64) {\n"
        "    const Catalan: f64 = %s;\n"
        "    const EulerGamma: f64 = %s;\n"
        "    let out1 = PI.powf(Catalan);\n"
        "    let out2 = EulerGamma);\n"
        "    (out1, out2)\n"
        "}\n"
    ) % (Catalan.evalf(17), EulerGamma.evalf(17))
    assert source == expected


def test_argument_order():
    expr = x + y
    routine = make_routine("test", expr, argument_sequence=[z, x, y], language="rust")
    code_gen = RustCodeGen()
    output = StringIO()
    code_gen.dump_rs([routine], output, "test", header=False, empty=False)
    source = output.getvalue()
    expected = (
        "fn test(z: f64, x: f64, y: f64) -> f64 {\n"
        "    let out1 = x + y;\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected


def test_multiple_results_rust():
    # Here the output order is the input order
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    name_expr = ("test", [expr1, expr2])
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64, z: f64) -> (f64, f64) {\n"
        "    let out1 = z*(x + y);\n"
        "    let out2 = z*(x - y);\n"
        "    (out1, out2)\n"
        "}\n"
    )
    assert source == expected


def test_results_named_unordered():
    # Here output order is based on name_expr
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64, z: f64) -> (f64, f64, f64) {\n"
        "    let C = z*(x + y);\n"
        "    let A = z*(x - y);\n"
        "    let B = 2*x;\n"
        "    (C, A, B)\n"
        "}\n"
    )
    assert source == expected


def test_results_named_ordered():
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result = codegen(name_expr, "Rust", header=False, empty=False,
                     argument_sequence=(x, z, y))
    assert result[0][0] == "test.rs"
    source = result[0][1]
    expected = (
        "fn test(x: f64, z: f64, y: f64) -> (f64, f64, f64) {\n"
        "    let C = z*(x + y);\n"
        "    let A = z*(x - y);\n"
        "    let B = 2*x;\n"
        "    (C, A, B)\n"
        "}\n"
    )
    assert source == expected


def test_complicated_rs_codegen():
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    name_expr = ("testlong",
            [ ((sin(x) + cos(y) + tan(z))**3).expand(),
            cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))
    ])
    result = codegen(name_expr, "Rust", header=False, empty=False)
    assert result[0][0] == "testlong.rs"
    source = result[0][1]
    expected = (
        "fn testlong(x: f64, y: f64, z: f64) -> (f64, f64) {\n"
        "    let out1 = x.sin().powi(3) + 3*x.sin().powi(2)*y.cos()"
        " + 3*x.sin().powi(2)*z.tan() + 3*x.sin()*y.cos().powi(2)"
        " + 6*x.sin()*y.cos()*z.tan() + 3*x.sin()*z.tan().powi(2)"
        " + y.cos().powi(3) + 3*y.cos().powi(2)*z.tan()"
        " + 3*y.cos()*z.tan().powi(2) + z.tan().powi(3);\n"
        "    let out2 = (x + y + z).cos().cos().cos().cos()"
        ".cos().cos().cos().cos();\n"
        "    (out1, out2)\n"
        "}\n"
    )
    assert source == expected


def test_output_arg_mixed_unordered():
    # named outputs are alphabetical, unnamed output appear in the given order
    from sympy.functions.elementary.trigonometric import (cos, sin)
    a = symbols("a")
    name_expr = ("foo", [cos(2*x), Equality(y, sin(x)), cos(x), Equality(a, sin(2*x))])
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    assert result[0] == "foo.rs"
    source = result[1]
    expected = (
        "fn foo(x: f64) -> (f64, f64, f64, f64) {\n"
        "    let out1 = (2*x).cos();\n"
        "    let y = x.sin();\n"
        "    let out3 = x.cos();\n"
        "    let a = (2*x).sin();\n"
        "    (out1, y, out3, a)\n"
        "}\n"
    )
    assert source == expected


def test_piecewise_():
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True), evaluate=False)
    name_expr = ("pwtest", pw)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn pwtest(x: f64) -> f64 {\n"
        "    let out1 = if (x < -1.0) {\n"
        "        0\n"
        "    } else if (x <= 1.0) {\n"
        "        x.powi(2)\n"
        "    } else if (x > 1.0) {\n"
        "        2 - x\n"
        "    } else {\n"
        "        1\n"
        "    };\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected


@XFAIL
def test_piecewise_inline():
    # FIXME: how to pass inline to the RustCodePrinter?
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True))
    name_expr = ("pwtest", pw)
    result, = codegen(name_expr, "Rust", header=False, empty=False,
                      inline=True)
    source = result[1]
    expected = (
        "fn pwtest(x: f64) -> f64 {\n"
        "    let out1 = if (x < -1) { 0 } else if (x <= 1) { x.powi(2) }"
        " else if (x > 1) { -x + 2 } else { 1 };\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected


def test_multifcns_per_file():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result = codegen(name_expr, "Rust", header=False, empty=False)
    assert result[0][0] == "foo.rs"
    source = result[0][1]
    expected = (
        "fn foo(x: f64, y: f64) -> (f64, f64) {\n"
        "    let out1 = 2*x;\n"
        "    let out2 = 3*y;\n"
        "    (out1, out2)\n"
        "}\n"
        "fn bar(y: f64) -> (f64, f64) {\n"
        "    let out1 = y.powi(2);\n"
        "    let out2 = 4*y;\n"
        "    (out1, out2)\n"
        "}\n"
    )
    assert source == expected


def test_multifcns_per_file_w_header():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result = codegen(name_expr, "Rust", header=True, empty=False)
    assert result[0][0] == "foo.rs"
    source = result[0][1]
    version_str = "Code generated with SymPy %s" % sympy.__version__
    version_line = version_str.center(76).rstrip()
    expected = (
        "/*\n"
        " *%(version_line)s\n"
        " *\n"
        " *              See http://www.sympy.org/ for more information.\n"
        " *\n"
        " *                       This file is part of 'project'\n"
        " */\n"
        "fn foo(x: f64, y: f64) -> (f64, f64) {\n"
        "    let out1 = 2*x;\n"
        "    let out2 = 3*y;\n"
        "    (out1, out2)\n"
        "}\n"
        "fn bar(y: f64) -> (f64, f64) {\n"
        "    let out1 = y.powi(2);\n"
        "    let out2 = 4*y;\n"
        "    (out1, out2)\n"
        "}\n"
    ) % {'version_line': version_line}
    assert source == expected


def test_filename_match_prefix():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result, = codegen(name_expr, "Rust", prefix="baz", header=False,
                     empty=False)
    assert result[0] == "baz.rs"


def test_InOutArgument():
    expr = Equality(x, x**2)
    name_expr = ("mysqr", expr)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn mysqr(x: f64) -> f64 {\n"
        "    let x = x.powi(2);\n"
        "    x\n"
        "}\n"
    )
    assert source == expected


def test_InOutArgument_order():
    # can specify the order as (x, y)
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Rust", header=False,
                      empty=False, argument_sequence=(x,y))
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64) -> f64 {\n"
        "    let x = x.powi(2) + y;\n"
        "    x\n"
        "}\n"
    )
    assert source == expected
    # make sure it gives (x, y) not (y, x)
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64) -> f64 {\n"
        "    let x = x.powi(2) + y;\n"
        "    x\n"
        "}\n"
    )
    assert source == expected


def test_not_supported():
    f = Function('f')
    name_expr = ("test", [f(x).diff(x), S.ComplexInfinity])
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64) -> (f64, f64) {\n"
        "    // unsupported: Derivative(f(x), x)\n"
        "    // unsupported: zoo\n"
        "    let out1 = Derivative(f(x), x);\n"
        "    let out2 = zoo;\n"
        "    (out1, out2)\n"
        "}\n"
    )
    assert source == expected


def test_global_vars_rust():
    x, y, z, t = symbols("x y z t")
    result = codegen(('f', x*y), "Rust", header=False, empty=False,
                     global_vars=(y,))
    source = result[0][1]
    expected = (
        "fn f(x: f64) -> f64 {\n"
        "    let out1 = x*y;\n"
        "    out1\n"
        "}\n"
        )
    assert source == expected

    result = codegen(('f', x*y+z), "Rust", header=False, empty=False,
                     argument_sequence=(x, y), global_vars=(z, t))
    source = result[0][1]
    expected = (
        "fn f(x: f64, y: f64) -> f64 {\n"
        "    let out1 = x*y + z;\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_api.py ===
import numpy as np
import pytest

from pandas.errors import (
    DataError,
    SpecificationError,
)

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    Timestamp,
    concat,
    date_range,
    timedelta_range,
)
import pandas._testing as tm


def test_getitem(step):
    frame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    r = frame.rolling(window=5, step=step)
    tm.assert_index_equal(r._selected_obj.columns, frame[::step].columns)

    r = frame.rolling(window=5, step=step)[1]
    assert r._selected_obj.name == frame[::step].columns[1]

    # technically this is allowed
    r = frame.rolling(window=5, step=step)[1, 3]
    tm.assert_index_equal(r._selected_obj.columns, frame[::step].columns[[1, 3]])

    r = frame.rolling(window=5, step=step)[[1, 3]]
    tm.assert_index_equal(r._selected_obj.columns, frame[::step].columns[[1, 3]])


def test_select_bad_cols():
    df = DataFrame([[1, 2]], columns=["A", "B"])
    g = df.rolling(window=5)
    with pytest.raises(KeyError, match="Columns not found: 'C'"):
        g[["C"]]
    with pytest.raises(KeyError, match="^[^A]+$"):
        # A should not be referenced as a bad column...
        # will have to rethink regex if you change message!
        g[["A", "C"]]


def test_attribute_access():
    df = DataFrame([[1, 2]], columns=["A", "B"])
    r = df.rolling(window=5)
    tm.assert_series_equal(r.A.sum(), r["A"].sum())
    msg = "'Rolling' object has no attribute 'F'"
    with pytest.raises(AttributeError, match=msg):
        r.F


def tests_skip_nuisance(step):
    df = DataFrame({"A": range(5), "B": range(5, 10), "C": "foo"})
    r = df.rolling(window=3, step=step)
    result = r[["A", "B"]].sum()
    expected = DataFrame(
        {"A": [np.nan, np.nan, 3, 6, 9], "B": [np.nan, np.nan, 18, 21, 24]},
        columns=list("AB"),
    )[::step]
    tm.assert_frame_equal(result, expected)


def test_sum_object_str_raises(step):
    df = DataFrame({"A": range(5), "B": range(5, 10), "C": "foo"})
    r = df.rolling(window=3, step=step)
    with pytest.raises(
        DataError, match="Cannot aggregate non-numeric type: object|str"
    ):
        # GH#42738, enforced in 2.0
        r.sum()


def test_agg(step):
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})

    r = df.rolling(window=3, step=step)
    a_mean = r["A"].mean()
    a_std = r["A"].std()
    a_sum = r["A"].sum()
    b_mean = r["B"].mean()
    b_std = r["B"].std()

    with tm.assert_produces_warning(FutureWarning, match="using Rolling.[mean|std]"):
        result = r.aggregate([np.mean, np.std])
    expected = concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = MultiIndex.from_product([["A", "B"], ["mean", "std"]])
    tm.assert_frame_equal(result, expected)

    with tm.assert_produces_warning(FutureWarning, match="using Rolling.[mean|std]"):
        result = r.aggregate({"A": np.mean, "B": np.std})

    expected = concat([a_mean, b_std], axis=1)
    tm.assert_frame_equal(result, expected, check_like=True)

    result = r.aggregate({"A": ["mean", "std"]})
    expected = concat([a_mean, a_std], axis=1)
    expected.columns = MultiIndex.from_tuples([("A", "mean"), ("A", "std")])
    tm.assert_frame_equal(result, expected)

    result = r["A"].aggregate(["mean", "sum"])
    expected = concat([a_mean, a_sum], axis=1)
    expected.columns = ["mean", "sum"]
    tm.assert_frame_equal(result, expected)

    msg = "nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        # using a dict with renaming
        r.aggregate({"A": {"mean": "mean", "sum": "sum"}})

    with pytest.raises(SpecificationError, match=msg):
        r.aggregate(
            {"A": {"mean": "mean", "sum": "sum"}, "B": {"mean2": "mean", "sum2": "sum"}}
        )

    result = r.aggregate({"A": ["mean", "std"], "B": ["mean", "std"]})
    expected = concat([a_mean, a_std, b_mean, b_std], axis=1)

    exp_cols = [("A", "mean"), ("A", "std"), ("B", "mean"), ("B", "std")]
    expected.columns = MultiIndex.from_tuples(exp_cols)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "func", [["min"], ["mean", "max"], {"b": "sum"}, {"b": "prod", "c": "median"}]
)
def test_multi_axis_1_raises(func):
    # GH#46904
    df = DataFrame({"a": [1, 1, 2], "b": [3, 4, 5], "c": [6, 7, 8]})
    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        r = df.rolling(window=3, axis=1)
    with pytest.raises(NotImplementedError, match="axis other than 0 is not supported"):
        r.agg(func)


def test_agg_apply(raw):
    # passed lambda
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})

    r = df.rolling(window=3)
    a_sum = r["A"].sum()

    with tm.assert_produces_warning(FutureWarning, match="using Rolling.[sum|std]"):
        result = r.agg({"A": np.sum, "B": lambda x: np.std(x, ddof=1)})
    rcustom = r["B"].apply(lambda x: np.std(x, ddof=1), raw=raw)
    expected = concat([a_sum, rcustom], axis=1)
    tm.assert_frame_equal(result, expected, check_like=True)


def test_agg_consistency(step):
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})
    r = df.rolling(window=3, step=step)

    with tm.assert_produces_warning(FutureWarning, match="using Rolling.[sum|mean]"):
        result = r.agg([np.sum, np.mean]).columns
    expected = MultiIndex.from_product([list("AB"), ["sum", "mean"]])
    tm.assert_index_equal(result, expected)

    with tm.assert_produces_warning(FutureWarning, match="using Rolling.[sum|mean]"):
        result = r["A"].agg([np.sum, np.mean]).columns
    expected = Index(["sum", "mean"])
    tm.assert_index_equal(result, expected)

    with tm.assert_produces_warning(FutureWarning, match="using Rolling.[sum|mean]"):
        result = r.agg({"A": [np.sum, np.mean]}).columns
    expected = MultiIndex.from_tuples([("A", "sum"), ("A", "mean")])
    tm.assert_index_equal(result, expected)


def test_agg_nested_dicts():
    # API change for disallowing these types of nested dicts
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})
    r = df.rolling(window=3)

    msg = "nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        r.aggregate({"r1": {"A": ["mean", "sum"]}, "r2": {"B": ["mean", "sum"]}})

    expected = concat(
        [r["A"].mean(), r["A"].std(), r["B"].mean(), r["B"].std()], axis=1
    )
    expected.columns = MultiIndex.from_tuples(
        [("ra", "mean"), ("ra", "std"), ("rb", "mean"), ("rb", "std")]
    )
    with pytest.raises(SpecificationError, match=msg):
        r[["A", "B"]].agg({"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}})

    with pytest.raises(SpecificationError, match=msg):
        r.agg({"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}})


def test_count_nonnumeric_types(step):
    # GH12541
    cols = [
        "int",
        "float",
        "string",
        "datetime",
        "timedelta",
        "periods",
        "fl_inf",
        "fl_nan",
        "str_nan",
        "dt_nat",
        "periods_nat",
    ]
    dt_nat_col = [Timestamp("20170101"), Timestamp("20170203"), Timestamp(None)]

    df = DataFrame(
        {
            "int": [1, 2, 3],
            "float": [4.0, 5.0, 6.0],
            "string": list("abc"),
            "datetime": date_range("20170101", periods=3),
            "timedelta": timedelta_range("1 s", periods=3, freq="s"),
            "periods": [
                Period("2012-01"),
                Period("2012-02"),
                Period("2012-03"),
            ],
            "fl_inf": [1.0, 2.0, np.inf],
            "fl_nan": [1.0, 2.0, np.nan],
            "str_nan": ["aa", "bb", np.nan],
            "dt_nat": dt_nat_col,
            "periods_nat": [
                Period("2012-01"),
                Period("2012-02"),
                Period(None),
            ],
        },
        columns=cols,
    )

    expected = DataFrame(
        {
            "int": [1.0, 2.0, 2.0],
            "float": [1.0, 2.0, 2.0],
            "string": [1.0, 2.0, 2.0],
            "datetime": [1.0, 2.0, 2.0],
            "timedelta": [1.0, 2.0, 2.0],
            "periods": [1.0, 2.0, 2.0],
            "fl_inf": [1.0, 2.0, 2.0],
            "fl_nan": [1.0, 2.0, 1.0],
            "str_nan": [1.0, 2.0, 1.0],
            "dt_nat": [1.0, 2.0, 1.0],
            "periods_nat": [1.0, 2.0, 1.0],
        },
        columns=cols,
    )[::step]

    result = df.rolling(window=2, min_periods=0, step=step).count()
    tm.assert_frame_equal(result, expected)

    result = df.rolling(1, min_periods=0, step=step).count()
    expected = df.notna().astype(float)[::step]
    tm.assert_frame_equal(result, expected)


def test_preserve_metadata():
    # GH 10565
    s = Series(np.arange(100), name="foo")

    s2 = s.rolling(30).sum()
    s3 = s.rolling(20).sum()
    assert s2.name == "foo"
    assert s3.name == "foo"


@pytest.mark.parametrize(
    "func,window_size,expected_vals",
    [
        (
            "rolling",
            2,
            [
                [np.nan, np.nan, np.nan, np.nan],
                [15.0, 20.0, 25.0, 20.0],
                [25.0, 30.0, 35.0, 30.0],
                [np.nan, np.nan, np.nan, np.nan],
                [20.0, 30.0, 35.0, 30.0],
                [35.0, 40.0, 60.0, 40.0],
                [60.0, 80.0, 85.0, 80],
            ],
        ),
        (
            "expanding",
            None,
            [
                [10.0, 10.0, 20.0, 20.0],
                [15.0, 20.0, 25.0, 20.0],
                [20.0, 30.0, 30.0, 20.0],
                [10.0, 10.0, 30.0, 30.0],
                [20.0, 30.0, 35.0, 30.0],
                [26.666667, 40.0, 50.0, 30.0],
                [40.0, 80.0, 60.0, 30.0],
            ],
        ),
    ],
)
def test_multiple_agg_funcs(func, window_size, expected_vals):
    # GH 15072
    df = DataFrame(
        [
            ["A", 10, 20],
            ["A", 20, 30],
            ["A", 30, 40],
            ["B", 10, 30],
            ["B", 30, 40],
            ["B", 40, 80],
            ["B", 80, 90],
        ],
        columns=["stock", "low", "high"],
    )

    f = getattr(df.groupby("stock"), func)
    if window_size:
        window = f(window_size)
    else:
        window = f()

    index = MultiIndex.from_tuples(
        [("A", 0), ("A", 1), ("A", 2), ("B", 3), ("B", 4), ("B", 5), ("B", 6)],
        names=["stock", None],
    )
    columns = MultiIndex.from_tuples(
        [("low", "mean"), ("low", "max"), ("high", "mean"), ("high", "min")]
    )
    expected = DataFrame(expected_vals, index=index, columns=columns)

    result = window.agg({"low": ["mean", "max"], "high": ["mean", "min"]})

    tm.assert_frame_equal(result, expected)


def test_dont_modify_attributes_after_methods(
    arithmetic_win_operators, closed, center, min_periods, step
):
    # GH 39554
    roll_obj = Series(range(1)).rolling(
        1, center=center, closed=closed, min_periods=min_periods, step=step
    )
    expected = {attr: getattr(roll_obj, attr) for attr in roll_obj._attributes}
    getattr(roll_obj, arithmetic_win_operators)()
    result = {attr: getattr(roll_obj, attr) for attr in roll_obj._attributes}
    assert result == expected


def test_centered_axis_validation(step):
    # ok
    msg = "The 'axis' keyword in Series.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        Series(np.ones(10)).rolling(window=3, center=True, axis=0, step=step).mean()

    # bad axis
    msg = "No axis named 1 for object type Series"
    with pytest.raises(ValueError, match=msg):
        Series(np.ones(10)).rolling(window=3, center=True, axis=1, step=step).mean()

    # ok ok
    df = DataFrame(np.ones((10, 10)))
    msg = "The 'axis' keyword in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.rolling(window=3, center=True, axis=0, step=step).mean()
    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.rolling(window=3, center=True, axis=1, step=step).mean()

    # bad axis
    msg = "No axis named 2 for object type DataFrame"
    with pytest.raises(ValueError, match=msg):
        (df.rolling(window=3, center=True, axis=2, step=step).mean())


def test_rolling_min_min_periods(step):
    a = Series([1, 2, 3, 4, 5])
    result = a.rolling(window=100, min_periods=1, step=step).min()
    expected = Series(np.ones(len(a)))[::step]
    tm.assert_series_equal(result, expected)
    msg = "min_periods 5 must be <= window 3"
    with pytest.raises(ValueError, match=msg):
        Series([1, 2, 3]).rolling(window=3, min_periods=5, step=step).min()


def test_rolling_max_min_periods(step):
    a = Series([1, 2, 3, 4, 5], dtype=np.float64)
    result = a.rolling(window=100, min_periods=1, step=step).max()
    expected = a[::step]
    tm.assert_almost_equal(result, expected)
    msg = "min_periods 5 must be <= window 3"
    with pytest.raises(ValueError, match=msg):
        Series([1, 2, 3]).rolling(window=3, min_periods=5, step=step).max()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\tests\test_inference.py ===
"""For more tests on satisfiability, see test_dimacs"""

from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.core.relational import Unequality
from sympy.logic.boolalg import And, Or, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
     pl_true, satisfiable, valid, entails, PropKB
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
    find_pure_symbol, find_unit_clause, unit_propagate, \
    find_pure_symbol_int_repr, find_unit_clause_int_repr, \
    unit_propagate_int_repr
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable

from sympy.logic.algorithms.z3_wrapper import z3_satisfiable
from sympy.assumptions.cnf import CNF, EncodedCNF
from sympy.logic.tests.test_lra_theory import make_random_problem
from sympy.core.random import randint

from sympy.testing.pytest import raises, skip
from sympy.external import import_module


def test_literal():
    A, B = symbols('A,B')
    assert literal_symbol(True) is True
    assert literal_symbol(False) is False
    assert literal_symbol(A) is A
    assert literal_symbol(~A) is A


def test_find_pure_symbol():
    A, B, C = symbols('A,B,C')
    assert find_pure_symbol([A], [A]) == (A, True)
    assert find_pure_symbol([A, B], [~A | B, ~B | A]) == (None, None)
    assert find_pure_symbol([A, B, C], [ A | ~B, ~B | ~C, C | A]) == (A, True)
    assert find_pure_symbol([A, B, C], [~A | B, B | ~C, C | A]) == (B, True)
    assert find_pure_symbol([A, B, C], [~A | ~B, ~B | ~C, C | A]) == (B, False)
    assert find_pure_symbol(
        [A, B, C], [~A | B, ~B | ~C, C | A]) == (None, None)


def test_find_pure_symbol_int_repr():
    assert find_pure_symbol_int_repr([1], [{1}]) == (1, True)
    assert find_pure_symbol_int_repr([1, 2],
                [{-1, 2}, {-2, 1}]) == (None, None)
    assert find_pure_symbol_int_repr([1, 2, 3],
                [{1, -2}, {-2, -3}, {3, 1}]) == (1, True)
    assert find_pure_symbol_int_repr([1, 2, 3],
                [{-1, 2}, {2, -3}, {3, 1}]) == (2, True)
    assert find_pure_symbol_int_repr([1, 2, 3],
                [{-1, -2}, {-2, -3}, {3, 1}]) == (2, False)
    assert find_pure_symbol_int_repr([1, 2, 3],
                [{-1, 2}, {-2, -3}, {3, 1}]) == (None, None)


def test_unit_clause():
    A, B, C = symbols('A,B,C')
    assert find_unit_clause([A], {}) == (A, True)
    assert find_unit_clause([A, ~A], {}) == (A, True)  # Wrong ??
    assert find_unit_clause([A | B], {A: True}) == (B, True)
    assert find_unit_clause([A | B], {B: True}) == (A, True)
    assert find_unit_clause(
        [A | B | C, B | ~C, A | ~B], {A: True}) == (B, False)
    assert find_unit_clause([A | B | C, B | ~C, A | B], {A: True}) == (B, True)
    assert find_unit_clause([A | B | C, B | ~C, A ], {}) == (A, True)


def test_unit_clause_int_repr():
    assert find_unit_clause_int_repr(map(set, [[1]]), {}) == (1, True)
    assert find_unit_clause_int_repr(map(set, [[1], [-1]]), {}) == (1, True)
    assert find_unit_clause_int_repr([{1, 2}], {1: True}) == (2, True)
    assert find_unit_clause_int_repr([{1, 2}], {2: True}) == (1, True)
    assert find_unit_clause_int_repr(map(set,
        [[1, 2, 3], [2, -3], [1, -2]]), {1: True}) == (2, False)
    assert find_unit_clause_int_repr(map(set,
        [[1, 2, 3], [3, -3], [1, 2]]), {1: True}) == (2, True)

    A, B, C = symbols('A,B,C')
    assert find_unit_clause([A | B | C, B | ~C, A ], {}) == (A, True)


def test_unit_propagate():
    A, B, C = symbols('A,B,C')
    assert unit_propagate([A | B], A) == []
    assert unit_propagate([A | B, ~A | C, ~C | B, A], A) == [C, ~C | B, A]


def test_unit_propagate_int_repr():
    assert unit_propagate_int_repr([{1, 2}], 1) == []
    assert unit_propagate_int_repr(map(set,
        [[1, 2], [-1, 3], [-3, 2], [1]]), 1) == [{3}, {-3, 2}]


def test_dpll():
    """This is also tested in test_dimacs"""
    A, B, C = symbols('A,B,C')
    assert dpll([A | B], [A, B], {A: True, B: True}) == {A: True, B: True}


def test_dpll_satisfiable():
    A, B, C = symbols('A,B,C')
    assert dpll_satisfiable( A & ~A ) is False
    assert dpll_satisfiable( A & ~B ) == {A: True, B: False}
    assert dpll_satisfiable(
        A | B ) in ({A: True}, {B: True}, {A: True, B: True})
    assert dpll_satisfiable(
        (~A | B) & (~B | A) ) in ({A: True, B: True}, {A: False, B: False})
    assert dpll_satisfiable( (A | B) & (~B | C) ) in ({A: True, B: False},
            {A: True, C: True}, {B: True, C: True})
    assert dpll_satisfiable( A & B & C  ) == {A: True, B: True, C: True}
    assert dpll_satisfiable( (A | B) & (A >> B) ) == {B: True}
    assert dpll_satisfiable( Equivalent(A, B) & A ) == {A: True, B: True}
    assert dpll_satisfiable( Equivalent(A, B) & ~A ) == {A: False, B: False}


def test_dpll2_satisfiable():
    A, B, C = symbols('A,B,C')
    assert dpll2_satisfiable( A & ~A ) is False
    assert dpll2_satisfiable( A & ~B ) == {A: True, B: False}
    assert dpll2_satisfiable(
        A | B ) in ({A: True}, {B: True}, {A: True, B: True})
    assert dpll2_satisfiable(
        (~A | B) & (~B | A) ) in ({A: True, B: True}, {A: False, B: False})
    assert dpll2_satisfiable( (A | B) & (~B | C) ) in ({A: True, B: False, C: True},
        {A: True, B: True, C: True})
    assert dpll2_satisfiable( A & B & C  ) == {A: True, B: True, C: True}
    assert dpll2_satisfiable( (A | B) & (A >> B) ) in ({B: True, A: False},
        {B: True, A: True})
    assert dpll2_satisfiable( Equivalent(A, B) & A ) == {A: True, B: True}
    assert dpll2_satisfiable( Equivalent(A, B) & ~A ) == {A: False, B: False}


def test_minisat22_satisfiable():
    A, B, C = symbols('A,B,C')
    minisat22_satisfiable = lambda expr: satisfiable(expr, algorithm="minisat22")
    assert minisat22_satisfiable( A & ~A ) is False
    assert minisat22_satisfiable( A & ~B ) == {A: True, B: False}
    assert minisat22_satisfiable(
        A | B ) in ({A: True}, {B: False}, {A: False, B: True}, {A: True, B: True}, {A: True, B: False})
    assert minisat22_satisfiable(
        (~A | B) & (~B | A) ) in ({A: True, B: True}, {A: False, B: False})
    assert minisat22_satisfiable( (A | B) & (~B | C) ) in ({A: True, B: False, C: True},
        {A: True, B: True, C: True}, {A: False, B: True, C: True}, {A: True, B: False, C: False})
    assert minisat22_satisfiable( A & B & C  ) == {A: True, B: True, C: True}
    assert minisat22_satisfiable( (A | B) & (A >> B) ) in ({B: True, A: False},
        {B: True, A: True})
    assert minisat22_satisfiable( Equivalent(A, B) & A ) == {A: True, B: True}
    assert minisat22_satisfiable( Equivalent(A, B) & ~A ) == {A: False, B: False}

def test_minisat22_minimal_satisfiable():
    A, B, C = symbols('A,B,C')
    minisat22_satisfiable = lambda expr, minimal=True: satisfiable(expr, algorithm="minisat22", minimal=True)
    assert minisat22_satisfiable( A & ~A ) is False
    assert minisat22_satisfiable( A & ~B ) == {A: True, B: False}
    assert minisat22_satisfiable(
        A | B ) in ({A: True}, {B: False}, {A: False, B: True}, {A: True, B: True}, {A: True, B: False})
    assert minisat22_satisfiable(
        (~A | B) & (~B | A) ) in ({A: True, B: True}, {A: False, B: False})
    assert minisat22_satisfiable( (A | B) & (~B | C) ) in ({A: True, B: False, C: True},
        {A: True, B: True, C: True}, {A: False, B: True, C: True}, {A: True, B: False, C: False})
    assert minisat22_satisfiable( A & B & C  ) == {A: True, B: True, C: True}
    assert minisat22_satisfiable( (A | B) & (A >> B) ) in ({B: True, A: False},
        {B: True, A: True})
    assert minisat22_satisfiable( Equivalent(A, B) & A ) == {A: True, B: True}
    assert minisat22_satisfiable( Equivalent(A, B) & ~A ) == {A: False, B: False}
    g = satisfiable((A | B | C),algorithm="minisat22",minimal=True,all_models=True)
    sol = next(g)
    first_solution = {key for key, value in sol.items() if value}
    sol=next(g)
    second_solution = {key for key, value in sol.items() if value}
    sol=next(g)
    third_solution = {key for key, value in sol.items() if value}
    assert not first_solution <= second_solution
    assert not second_solution <= third_solution
    assert not first_solution <= third_solution

def test_satisfiable():
    A, B, C = symbols('A,B,C')
    assert satisfiable(A & (A >> B) & ~B) is False


def test_valid():
    A, B, C = symbols('A,B,C')
    assert valid(A >> (B >> A)) is True
    assert valid((A >> (B >> C)) >> ((A >> B) >> (A >> C))) is True
    assert valid((~B >> ~A) >> (A >> B)) is True
    assert valid(A | B | C) is False
    assert valid(A >> B) is False


def test_pl_true():
    A, B, C = symbols('A,B,C')
    assert pl_true(True) is True
    assert pl_true( A & B, {A: True, B: True}) is True
    assert pl_true( A | B, {A: True}) is True
    assert pl_true( A | B, {B: True}) is True
    assert pl_true( A | B, {A: None, B: True}) is True
    assert pl_true( A >> B, {A: False}) is True
    assert pl_true( A | B | ~C, {A: False, B: True, C: True}) is True
    assert pl_true(Equivalent(A, B), {A: False, B: False}) is True

    # test for false
    assert pl_true(False) is False
    assert pl_true( A & B, {A: False, B: False}) is False
    assert pl_true( A & B, {A: False}) is False
    assert pl_true( A & B, {B: False}) is False
    assert pl_true( A | B, {A: False, B: False}) is False

    #test for None
    assert pl_true(B, {B: None}) is None
    assert pl_true( A & B, {A: True, B: None}) is None
    assert pl_true( A >> B, {A: True, B: None}) is None
    assert pl_true(Equivalent(A, B), {A: None}) is None
    assert pl_true(Equivalent(A, B), {A: True, B: None}) is None

    # Test for deep
    assert pl_true(A | B, {A: False}, deep=True) is None
    assert pl_true(~A & ~B, {A: False}, deep=True) is None
    assert pl_true(A | B, {A: False, B: False}, deep=True) is False
    assert pl_true(A & B & (~A | ~B), {A: True}, deep=True) is False
    assert pl_true((C >> A) >> (B >> A), {C: True}, deep=True) is True


def test_pl_true_wrong_input():
    from sympy.core.numbers import pi
    raises(ValueError, lambda: pl_true('John Cleese'))
    raises(ValueError, lambda: pl_true(42 + pi + pi ** 2))
    raises(ValueError, lambda: pl_true(42))


def test_entails():
    A, B, C = symbols('A, B, C')
    assert entails(A, [A >> B, ~B]) is False
    assert entails(B, [Equivalent(A, B), A]) is True
    assert entails((A >> B) >> (~A >> ~B)) is False
    assert entails((A >> B) >> (~B >> ~A)) is True


def test_PropKB():
    A, B, C = symbols('A,B,C')
    kb = PropKB()
    assert kb.ask(A >> B) is False
    assert kb.ask(A >> (B >> A)) is True
    kb.tell(A >> B)
    kb.tell(B >> C)
    assert kb.ask(A) is False
    assert kb.ask(B) is False
    assert kb.ask(C) is False
    assert kb.ask(~A) is False
    assert kb.ask(~B) is False
    assert kb.ask(~C) is False
    assert kb.ask(A >> C) is True
    kb.tell(A)
    assert kb.ask(A) is True
    assert kb.ask(B) is True
    assert kb.ask(C) is True
    assert kb.ask(~C) is False
    kb.retract(A)
    assert kb.ask(C) is False


def test_propKB_tolerant():
    """"tolerant to bad input"""
    kb = PropKB()
    A, B, C = symbols('A,B,C')
    assert kb.ask(B) is False

def test_satisfiable_non_symbols():
    x, y = symbols('x y')
    assumptions = Q.zero(x*y)
    facts = Implies(Q.zero(x*y), Q.zero(x) | Q.zero(y))
    query = ~Q.zero(x) & ~Q.zero(y)
    refutations = [
        {Q.zero(x): True, Q.zero(x*y): True},
        {Q.zero(y): True, Q.zero(x*y): True},
        {Q.zero(x): True, Q.zero(y): True, Q.zero(x*y): True},
        {Q.zero(x): True, Q.zero(y): False, Q.zero(x*y): True},
        {Q.zero(x): False, Q.zero(y): True, Q.zero(x*y): True}]
    assert not satisfiable(And(assumptions, facts, query), algorithm='dpll')
    assert satisfiable(And(assumptions, facts, ~query), algorithm='dpll') in refutations
    assert not satisfiable(And(assumptions, facts, query), algorithm='dpll2')
    assert satisfiable(And(assumptions, facts, ~query), algorithm='dpll2') in refutations

def test_satisfiable_bool():
    from sympy.core.singleton import S
    assert satisfiable(true) == {true: true}
    assert satisfiable(S.true) == {true: true}
    assert satisfiable(false) is False
    assert satisfiable(S.false) is False


def test_satisfiable_all_models():
    from sympy.abc import A, B
    assert next(satisfiable(False, all_models=True)) is False
    assert list(satisfiable((A >> ~A) & A, all_models=True)) == [False]
    assert list(satisfiable(True, all_models=True)) == [{true: true}]

    models = [{A: True, B: False}, {A: False, B: True}]
    result = satisfiable(A ^ B, all_models=True)
    models.remove(next(result))
    models.remove(next(result))
    raises(StopIteration, lambda: next(result))
    assert not models

    assert list(satisfiable(Equivalent(A, B), all_models=True)) == \
    [{A: False, B: False}, {A: True, B: True}]

    models = [{A: False, B: False}, {A: False, B: True}, {A: True, B: True}]
    for model in satisfiable(A >> B, all_models=True):
        models.remove(model)
    assert not models

    # This is a santiy test to check that only the required number
    # of solutions are generated. The expr below has 2**100 - 1 models
    # which would time out the test if all are generated at once.
    from sympy.utilities.iterables import numbered_symbols
    from sympy.logic.boolalg import Or
    sym = numbered_symbols()
    X = [next(sym) for i in range(100)]
    result = satisfiable(Or(*X), all_models=True)
    for i in range(10):
        assert next(result)


def test_z3():
    z3 = import_module("z3")

    if not z3:
        skip("z3 not installed.")
    A, B, C = symbols('A,B,C')
    x, y, z = symbols('x,y,z')
    assert z3_satisfiable((x >= 2) & (x < 1)) is False
    assert z3_satisfiable( A & ~A ) is False

    model = z3_satisfiable(A & (~A | B | C))
    assert bool(model) is True
    assert model[A] is True

    # test nonlinear function
    assert z3_satisfiable((x ** 2 >= 2) & (x < 1) & (x > -1)) is False


def test_z3_vs_lra_dpll2():
    z3 = import_module("z3")
    if z3 is None:
        skip("z3 not installed.")

    def boolean_formula_to_encoded_cnf(bf):
        cnf = CNF.from_prop(bf)
        enc = EncodedCNF()
        enc.from_cnf(cnf)
        return enc

    def make_random_cnf(num_clauses=5, num_constraints=10, num_var=2):
        assert num_clauses <= num_constraints
        constraints = make_random_problem(num_variables=num_var, num_constraints=num_constraints, rational=False)
        clauses = [[cons] for cons in constraints[:num_clauses]]
        for cons in constraints[num_clauses:]:
            if isinstance(cons, Unequality):
                cons = ~cons
            i = randint(0, num_clauses-1)
            clauses[i].append(cons)

        clauses = [Or(*clause) for clause in clauses]
        cnf = And(*clauses)
        return boolean_formula_to_encoded_cnf(cnf)

    lra_dpll2_satisfiable = lambda x: dpll2_satisfiable(x, use_lra_theory=True)

    for _ in range(50):
        cnf = make_random_cnf(num_clauses=10, num_constraints=15, num_var=2)

        try:
            z3_sat = z3_satisfiable(cnf)
        except z3.z3types.Z3Exception:
            continue

        lra_dpll2_sat = lra_dpll2_satisfiable(cnf) is not False

        assert z3_sat == lra_dpll2_sat

def test_issue_27733():
    x, y = symbols('x,y')
    clauses = [[1, -3, -2], [5, 7, -8, -6, -4], [-10, -9, 10, 11, -4], [-12, 13, 14], [-10, 9, -6, 11, -4],
               [16, -15, 18, -19, -17], [11, -6, 10, -9], [9, 11, -10, -9], [2, -3, -1], [-13, 12], [-15, 3, -17],
               [-16, -15, 19, -17], [-6, -9, 10, 11, -4], [20, -1, -2], [-23, -22, -21], [10, 11, -10, -9],
               [9, 11, -4, -10], [24, -6, -4], [-14, 12], [-10, -9, 9, -6, 11], [25, -27, -26], [-15, 19, -18, -17],
               [5, 8, -7, -6, -4], [-30, -29, 28], [12], [14]]

    encoding = {Q.gt(y, i): i for i in range(1, 31) if i != 11 and i != 12}
    encoding[Q.gt(x, 0)] = 11
    encoding[Q.lt(x, 0)] = 12

    cnf = EncodedCNF(clauses, encoding)
    assert satisfiable(cnf, use_lra_theory=True) is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_jscode.py ===
from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
                        EulerGamma, Catalan, Lambda, Dummy, S, Eq, Ne, Le,
                        Lt, Gt, Ge, Mod)
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
                             sinh, cosh, tanh, asin, acos, acosh, Max, Min)
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol

from sympy.printing.jscode import jscode

x, y, z = symbols('x,y,z')


def test_printmethod():
    assert jscode(Abs(x)) == "Math.abs(x)"


def test_jscode_sqrt():
    assert jscode(sqrt(x)) == "Math.sqrt(x)"
    assert jscode(x**0.5) == "Math.sqrt(x)"
    assert jscode(x**(S.One/3)) == "Math.cbrt(x)"


def test_jscode_Pow():
    g = implemented_function('g', Lambda(x, 2*x))
    assert jscode(x**3) == "Math.pow(x, 3)"
    assert jscode(x**(y**3)) == "Math.pow(x, Math.pow(y, 3))"
    assert jscode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "Math.pow(3.5*2*x, -x + Math.pow(y, x))/(Math.pow(x, 2) + y)"
    assert jscode(x**-1.0) == '1/x'


def test_jscode_constants_mathh():
    assert jscode(exp(1)) == "Math.E"
    assert jscode(pi) == "Math.PI"
    assert jscode(oo) == "Number.POSITIVE_INFINITY"
    assert jscode(-oo) == "Number.NEGATIVE_INFINITY"


def test_jscode_constants_other():
    assert jscode(
        2*GoldenRatio) == "var GoldenRatio = %s;\n2*GoldenRatio" % GoldenRatio.evalf(17)
    assert jscode(2*Catalan) == "var Catalan = %s;\n2*Catalan" % Catalan.evalf(17)
    assert jscode(
        2*EulerGamma) == "var EulerGamma = %s;\n2*EulerGamma" % EulerGamma.evalf(17)


def test_jscode_Rational():
    assert jscode(Rational(3, 7)) == "3/7"
    assert jscode(Rational(18, 9)) == "2"
    assert jscode(Rational(3, -7)) == "-3/7"
    assert jscode(Rational(-3, -7)) == "3/7"


def test_Relational():
    assert jscode(Eq(x, y)) == "x == y"
    assert jscode(Ne(x, y)) == "x != y"
    assert jscode(Le(x, y)) == "x <= y"
    assert jscode(Lt(x, y)) == "x < y"
    assert jscode(Gt(x, y)) == "x > y"
    assert jscode(Ge(x, y)) == "x >= y"


def test_Mod():
    assert jscode(Mod(x, y)) == '((x % y) + y) % y'
    assert jscode(Mod(x, x + y)) == '((x % (x + y)) + (x + y)) % (x + y)'
    p1, p2 = symbols('p1 p2', positive=True)
    assert jscode(Mod(p1, p2)) == 'p1 % p2'
    assert jscode(Mod(p1, p2 + 3)) == 'p1 % (p2 + 3)'
    assert jscode(Mod(-3, -7, evaluate=False)) == '(-3) % (-7)'
    assert jscode(-Mod(p1, p2)) == '-(p1 % p2)'
    assert jscode(x*Mod(p1, p2)) == 'x*(p1 % p2)'


def test_jscode_Integer():
    assert jscode(Integer(67)) == "67"
    assert jscode(Integer(-1)) == "-1"


def test_jscode_functions():
    assert jscode(sin(x) ** cos(x)) == "Math.pow(Math.sin(x), Math.cos(x))"
    assert jscode(sinh(x) * cosh(x)) == "Math.sinh(x)*Math.cosh(x)"
    assert jscode(Max(x, y) + Min(x, y)) == "Math.max(x, y) + Math.min(x, y)"
    assert jscode(tanh(x)*acosh(y)) == "Math.tanh(x)*Math.acosh(y)"
    assert jscode(asin(x)-acos(y)) == "-Math.acos(y) + Math.asin(x)"


def test_jscode_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert jscode(g(x)) == "2*x"
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert jscode(g(x)) == "var Catalan = %s;\n2*x/Catalan" % Catalan.evalf(17)
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert jscode(g(A[i]), assign_to=A[i]) == (
        "for (var i=0; i<n; i++){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )


def test_jscode_exceptions():
    assert jscode(ceiling(x)) == "Math.ceil(x)"
    assert jscode(Abs(x)) == "Math.abs(x)"


def test_jscode_boolean():
    assert jscode(x & y) == "x && y"
    assert jscode(x | y) == "x || y"
    assert jscode(~x) == "!x"
    assert jscode(x & y & z) == "x && y && z"
    assert jscode(x | y | z) == "x || y || z"
    assert jscode((x & y) | z) == "z || x && y"
    assert jscode((x | y) & z) == "z && (x || y)"


def test_jscode_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    p = jscode(expr)
    s = \
"""\
((x < 1) ? (
   x
)
: (
   Math.pow(x, 2)
))\
"""
    assert p == s
    assert jscode(expr, assign_to="c") == (
    "if (x < 1) {\n"
    "   c = x;\n"
    "}\n"
    "else {\n"
    "   c = Math.pow(x, 2);\n"
    "}")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: jscode(expr))


def test_jscode_Piecewise_deep():
    p = jscode(2*Piecewise((x, x < 1), (x**2, True)))
    s = \
"""\
2*((x < 1) ? (
   x
)
: (
   Math.pow(x, 2)
))\
"""
    assert p == s


def test_jscode_settings():
    raises(TypeError, lambda: jscode(sin(x), method="garbage"))


def test_jscode_Indexed():
    n, m, o = symbols('n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    p = JavascriptCodePrinter()
    p._not_c = set()

    x = IndexedBase('x')[j]
    assert p._print_Indexed(x) == 'x[j]'
    A = IndexedBase('A')[i, j]
    assert p._print_Indexed(A) == 'A[%s]' % (m*i+j)
    B = IndexedBase('B')[i, j, k]
    assert p._print_Indexed(B) == 'B[%s]' % (i*o*m+j*o+k)

    assert p._not_c == set()


def test_jscode_loops_matrix_vector():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (var i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (var i=0; i<m; i++){\n'
        '   for (var j=0; j<n; j++){\n'
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    c = jscode(A[i, j]*x[j], assign_to=y[i])
    assert c == s


def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    expected = (
        'for (var i_%(icount)i=0; i_%(icount)i<m_%(mcount)i; i_%(icount)i++){\n'
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    code = jscode(x[i], assign_to=y[i])
    assert code == expected


def test_jscode_loops_add():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (var i=0; i<m; i++){\n'
        '   y[i] = x[i] + z[i];\n'
        '}\n'
        'for (var i=0; i<m; i++){\n'
        '   for (var j=0; j<n; j++){\n'
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    c = jscode(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i])
    assert c == s


def test_jscode_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (var i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (var i=0; i<m; i++){\n'
        '   for (var j=0; j<n; j++){\n'
        '      for (var k=0; k<o; k++){\n'
        '         for (var l=0; l<p; l++){\n'
        '            y[i] = a[%s]*b[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    c = jscode(b[j, k, l]*a[i, j, k, l], assign_to=y[i])
    assert c == s


def test_jscode_loops_addfactor():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (var i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (var i=0; i<m; i++){\n'
        '   for (var j=0; j<n; j++){\n'
        '      for (var k=0; k<o; k++){\n'
        '         for (var l=0; l<p; l++){\n'
        '            y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    c = jscode((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    assert c == s


def test_jscode_loops_multiple_terms():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    s0 = (
        'for (var i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
    )
    s1 = (
        'for (var i=0; i<m; i++){\n'
        '   for (var j=0; j<n; j++){\n'
        '      for (var k=0; k<o; k++){\n'
        '         y[i] = b[j]*b[k]*c[%s] + y[i];\n' % (i*n*o + j*o + k) +\
        '      }\n'
        '   }\n'
        '}\n'
    )
    s2 = (
        'for (var i=0; i<m; i++){\n'
        '   for (var k=0; k<o; k++){\n'
        '      y[i] = a[%s]*b[k] + y[i];\n' % (i*o + k) +\
        '   }\n'
        '}\n'
    )
    s3 = (
        'for (var i=0; i<m; i++){\n'
        '   for (var j=0; j<n; j++){\n'
        '      y[i] = a[%s]*b[j] + y[i];\n' % (i*n + j) +\
        '   }\n'
        '}\n'
    )
    c = jscode(
        b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])
    assert (c == s0 + s1 + s2 + s3[:-1] or
            c == s0 + s1 + s3 + s2[:-1] or
            c == s0 + s2 + s1 + s3[:-1] or
            c == s0 + s2 + s3 + s1[:-1] or
            c == s0 + s3 + s1 + s2[:-1] or
            c == s0 + s3 + s2 + s1[:-1])


def test_Matrix_printing():
    # Test returning a Matrix
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    assert jscode(mat, A) == (
        "A[0] = x*y;\n"
        "if (y > 0) {\n"
        "   A[1] = x + 2;\n"
        "}\n"
        "else {\n"
        "   A[1] = y;\n"
        "}\n"
        "A[2] = Math.sin(z);")
    # Test using MatrixElements in expressions
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    assert jscode(expr) == (
        "((x > 0) ? (\n"
        "   2*A[2]\n"
        ")\n"
        ": (\n"
        "   A[2]\n"
        ")) + Math.sin(A[1]) + A[0]")
    # Test using MatrixElements in a Matrix
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    assert jscode(m, M) == (
        "M[0] = Math.sin(q[1]);\n"
        "M[1] = 0;\n"
        "M[2] = Math.cos(q[2]);\n"
        "M[3] = q[1] + q[2];\n"
        "M[4] = q[3];\n"
        "M[5] = 5;\n"
        "M[6] = 2*q[4]/q[1];\n"
        "M[7] = Math.sqrt(q[0]) + 4;\n"
        "M[8] = 0;")


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(jscode(A[0, 0]) == "A[0]")
    assert(jscode(3 * A[0, 0]) == "3*A[0]")

    F = C[0, 0].subs(C, A - B)
    assert(jscode(F) == "(A - B)[0]")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_api.py ===
from copy import deepcopy
import inspect
import pydoc

import numpy as np
import pytest

from pandas._config import using_string_dtype
from pandas._config.config import option_context

from pandas.compat import HAS_PYARROW

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
    timedelta_range,
)
import pandas._testing as tm


class TestDataFrameMisc:
    def test_getitem_pop_assign_name(self, float_frame):
        s = float_frame["A"]
        assert s.name == "A"

        s = float_frame.pop("A")
        assert s.name == "A"

        s = float_frame.loc[:, "B"]
        assert s.name == "B"

        s2 = s.loc[:]
        assert s2.name == "B"

    def test_get_axis(self, float_frame):
        f = float_frame
        assert f._get_axis_number(0) == 0
        assert f._get_axis_number(1) == 1
        assert f._get_axis_number("index") == 0
        assert f._get_axis_number("rows") == 0
        assert f._get_axis_number("columns") == 1

        assert f._get_axis_name(0) == "index"
        assert f._get_axis_name(1) == "columns"
        assert f._get_axis_name("index") == "index"
        assert f._get_axis_name("rows") == "index"
        assert f._get_axis_name("columns") == "columns"

        assert f._get_axis(0) is f.index
        assert f._get_axis(1) is f.columns

        with pytest.raises(ValueError, match="No axis named"):
            f._get_axis_number(2)

        with pytest.raises(ValueError, match="No axis.*foo"):
            f._get_axis_name("foo")

        with pytest.raises(ValueError, match="No axis.*None"):
            f._get_axis_name(None)

        with pytest.raises(ValueError, match="No axis named"):
            f._get_axis_number(None)

    def test_column_contains_raises(self, float_frame):
        with pytest.raises(TypeError, match="unhashable type: 'Index'"):
            float_frame.columns in float_frame

    def test_tab_completion(self):
        # DataFrame whose columns are identifiers shall have them in __dir__.
        df = DataFrame([list("abcd"), list("efgh")], columns=list("ABCD"))
        for key in list("ABCD"):
            assert key in dir(df)
        assert isinstance(df.__getitem__("A"), Series)

        # DataFrame whose first-level columns are identifiers shall have
        # them in __dir__.
        df = DataFrame(
            [list("abcd"), list("efgh")],
            columns=pd.MultiIndex.from_tuples(list(zip("ABCD", "EFGH"))),
        )
        for key in list("ABCD"):
            assert key in dir(df)
        for key in list("EFGH"):
            assert key not in dir(df)
        assert isinstance(df.__getitem__("A"), DataFrame)

    def test_display_max_dir_items(self):
        # display.max_dir_items increaes the number of columns that are in __dir__.
        columns = ["a" + str(i) for i in range(420)]
        values = [range(420), range(420)]
        df = DataFrame(values, columns=columns)

        # The default value for display.max_dir_items is 100
        assert "a99" in dir(df)
        assert "a100" not in dir(df)

        with option_context("display.max_dir_items", 300):
            df = DataFrame(values, columns=columns)
            assert "a299" in dir(df)
            assert "a300" not in dir(df)

        with option_context("display.max_dir_items", None):
            df = DataFrame(values, columns=columns)
            assert "a419" in dir(df)

    def test_not_hashable(self):
        empty_frame = DataFrame()

        df = DataFrame([1])
        msg = "unhashable type: 'DataFrame'"
        with pytest.raises(TypeError, match=msg):
            hash(df)
        with pytest.raises(TypeError, match=msg):
            hash(empty_frame)

    @pytest.mark.xfail(
        using_string_dtype() and HAS_PYARROW, reason="surrogates not allowed"
    )
    def test_column_name_contains_unicode_surrogate(self):
        # GH 25509
        colname = "\ud83d"
        df = DataFrame({colname: []})
        # this should not crash
        assert colname not in dir(df)
        assert df.columns[0] == colname

    def test_new_empty_index(self):
        df1 = DataFrame(np.random.default_rng(2).standard_normal((0, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((0, 3)))
        df1.index.name = "foo"
        assert df2.index.name is None

    def test_get_agg_axis(self, float_frame):
        cols = float_frame._get_agg_axis(0)
        assert cols is float_frame.columns

        idx = float_frame._get_agg_axis(1)
        assert idx is float_frame.index

        msg = r"Axis must be 0 or 1 \(got 2\)"
        with pytest.raises(ValueError, match=msg):
            float_frame._get_agg_axis(2)

    def test_empty(self, float_frame, float_string_frame):
        empty_frame = DataFrame()
        assert empty_frame.empty

        assert not float_frame.empty
        assert not float_string_frame.empty

        # corner case
        df = DataFrame({"A": [1.0, 2.0, 3.0], "B": ["a", "b", "c"]}, index=np.arange(3))
        del df["A"]
        assert not df.empty

    def test_len(self, float_frame):
        assert len(float_frame) == len(float_frame.index)

        # single block corner case
        arr = float_frame[["A", "B"]].values
        expected = float_frame.reindex(columns=["A", "B"]).values
        tm.assert_almost_equal(arr, expected)

    def test_axis_aliases(self, float_frame):
        f = float_frame

        # reg name
        expected = f.sum(axis=0)
        result = f.sum(axis="index")
        tm.assert_series_equal(result, expected)

        expected = f.sum(axis=1)
        result = f.sum(axis="columns")
        tm.assert_series_equal(result, expected)

    def test_class_axis(self):
        # GH 18147
        # no exception and no empty docstring
        assert pydoc.getdoc(DataFrame.index)
        assert pydoc.getdoc(DataFrame.columns)

    def test_series_put_names(self, float_string_frame):
        series = float_string_frame._series
        for k, v in series.items():
            assert v.name == k

    def test_empty_nonzero(self):
        df = DataFrame([1, 2, 3])
        assert not df.empty
        df = DataFrame(index=[1], columns=[1])
        assert not df.empty
        df = DataFrame(index=["a", "b"], columns=["c", "d"]).dropna()
        assert df.empty
        assert df.T.empty

    @pytest.mark.parametrize(
        "df",
        [
            DataFrame(),
            DataFrame(index=[1]),
            DataFrame(columns=[1]),
            DataFrame({1: []}),
        ],
    )
    def test_empty_like(self, df):
        assert df.empty
        assert df.T.empty

    def test_with_datetimelikes(self):
        df = DataFrame(
            {
                "A": date_range("20130101", periods=10),
                "B": timedelta_range("1 day", periods=10),
            }
        )
        t = df.T

        result = t.dtypes.value_counts()
        expected = Series({np.dtype("object"): 10}, name="count")
        tm.assert_series_equal(result, expected)

    def test_deepcopy(self, float_frame):
        cp = deepcopy(float_frame)
        cp.loc[0, "A"] = 10
        assert not float_frame.equals(cp)

    def test_inplace_return_self(self):
        # GH 1893

        data = DataFrame(
            {"a": ["foo", "bar", "baz", "qux"], "b": [0, 0, 1, 1], "c": [1, 2, 3, 4]}
        )

        def _check_f(base, f):
            result = f(base)
            assert result is None

        # -----DataFrame-----

        # set_index
        f = lambda x: x.set_index("a", inplace=True)
        _check_f(data.copy(), f)

        # reset_index
        f = lambda x: x.reset_index(inplace=True)
        _check_f(data.set_index("a"), f)

        # drop_duplicates
        f = lambda x: x.drop_duplicates(inplace=True)
        _check_f(data.copy(), f)

        # sort
        f = lambda x: x.sort_values("b", inplace=True)
        _check_f(data.copy(), f)

        # sort_index
        f = lambda x: x.sort_index(inplace=True)
        _check_f(data.copy(), f)

        # fillna
        f = lambda x: x.fillna(0, inplace=True)
        _check_f(data.copy(), f)

        # replace
        f = lambda x: x.replace(1, 0, inplace=True)
        _check_f(data.copy(), f)

        # rename
        f = lambda x: x.rename({1: "foo"}, inplace=True)
        _check_f(data.copy(), f)

        # -----Series-----
        d = data.copy()["c"]

        # reset_index
        f = lambda x: x.reset_index(inplace=True, drop=True)
        _check_f(data.set_index("a")["c"], f)

        # fillna
        f = lambda x: x.fillna(0, inplace=True)
        _check_f(d.copy(), f)

        # replace
        f = lambda x: x.replace(1, 0, inplace=True)
        _check_f(d.copy(), f)

        # rename
        f = lambda x: x.rename({1: "foo"}, inplace=True)
        _check_f(d.copy(), f)

    def test_tab_complete_warning(self, ip, frame_or_series):
        # GH 16409
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        if frame_or_series is DataFrame:
            code = "from pandas import DataFrame; obj = DataFrame()"
        else:
            code = "from pandas import Series; obj = Series(dtype=object)"

        ip.run_cell(code)
        # GH 31324 newer jedi version raises Deprecation warning;
        #  appears resolved 2021-02-02
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter("ignore"):
                list(ip.Completer.completions("obj.", 1))

    def test_attrs(self):
        df = DataFrame({"A": [2, 3]})
        assert df.attrs == {}
        df.attrs["version"] = 1

        result = df.rename(columns=str)
        assert result.attrs == {"version": 1}

    def test_attrs_deepcopy(self):
        df = DataFrame({"A": [2, 3]})
        assert df.attrs == {}
        df.attrs["tags"] = {"spam", "ham"}

        result = df.rename(columns=str)
        assert result.attrs == df.attrs
        assert result.attrs["tags"] is not df.attrs["tags"]

    @pytest.mark.parametrize("allows_duplicate_labels", [True, False, None])
    def test_set_flags(
        self,
        allows_duplicate_labels,
        frame_or_series,
        using_copy_on_write,
        warn_copy_on_write,
    ):
        obj = DataFrame({"A": [1, 2]})
        key = (0, 0)
        if frame_or_series is Series:
            obj = obj["A"]
            key = 0

        result = obj.set_flags(allows_duplicate_labels=allows_duplicate_labels)

        if allows_duplicate_labels is None:
            # We don't update when it's not provided
            assert result.flags.allows_duplicate_labels is True
        else:
            assert result.flags.allows_duplicate_labels is allows_duplicate_labels

        # We made a copy
        assert obj is not result

        # We didn't mutate obj
        assert obj.flags.allows_duplicate_labels is True

        # But we didn't copy data
        if frame_or_series is Series:
            assert np.may_share_memory(obj.values, result.values)
        else:
            assert np.may_share_memory(obj["A"].values, result["A"].values)

        with tm.assert_cow_warning(warn_copy_on_write):
            result.iloc[key] = 0
        if using_copy_on_write:
            assert obj.iloc[key] == 1
        else:
            assert obj.iloc[key] == 0
            # set back to 1 for test below
            with tm.assert_cow_warning(warn_copy_on_write):
                result.iloc[key] = 1

        # Now we do copy.
        result = obj.set_flags(
            copy=True, allows_duplicate_labels=allows_duplicate_labels
        )
        result.iloc[key] = 10
        assert obj.iloc[key] == 1

    def test_constructor_expanddim(self):
        # GH#33628 accessing _constructor_expanddim should not raise NotImplementedError
        # GH38782 pandas has no container higher than DataFrame (two-dim), so
        # DataFrame._constructor_expand_dim, doesn't make sense, so is removed.
        df = DataFrame()

        msg = "'DataFrame' object has no attribute '_constructor_expanddim'"
        with pytest.raises(AttributeError, match=msg):
            df._constructor_expanddim(np.arange(27).reshape(3, 3, 3))

    def test_inspect_getmembers(self):
        # GH38740
        df = DataFrame()
        msg = "DataFrame._data is deprecated"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            inspect.getmembers(df)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_counting.py ===
from itertools import product
from string import ascii_lowercase

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestCounting:
    def test_cumcount(self):
        df = DataFrame([["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"])
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 1, 2, 0, 3])

        tm.assert_series_equal(expected, g.cumcount())
        tm.assert_series_equal(expected, sg.cumcount())

    def test_cumcount_empty(self):
        ge = DataFrame().groupby(level=0)
        se = Series(dtype=object).groupby(level=0)

        # edge case, as this is usually considered float
        e = Series(dtype="int64")

        tm.assert_series_equal(e, ge.cumcount())
        tm.assert_series_equal(e, se.cumcount())

    def test_cumcount_dupe_index(self):
        df = DataFrame(
            [["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"], index=[0] * 5
        )
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 1, 2, 0, 3], index=[0] * 5)

        tm.assert_series_equal(expected, g.cumcount())
        tm.assert_series_equal(expected, sg.cumcount())

    def test_cumcount_mi(self):
        mi = MultiIndex.from_tuples([[0, 1], [1, 2], [2, 2], [2, 2], [1, 0]])
        df = DataFrame([["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"], index=mi)
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 1, 2, 0, 3], index=mi)

        tm.assert_series_equal(expected, g.cumcount())
        tm.assert_series_equal(expected, sg.cumcount())

    def test_cumcount_groupby_not_col(self):
        df = DataFrame(
            [["a"], ["a"], ["a"], ["b"], ["a"]], columns=["A"], index=[0] * 5
        )
        g = df.groupby([0, 0, 0, 1, 0])
        sg = g.A

        expected = Series([0, 1, 2, 0, 3], index=[0] * 5)

        tm.assert_series_equal(expected, g.cumcount())
        tm.assert_series_equal(expected, sg.cumcount())

    def test_ngroup(self):
        df = DataFrame({"A": list("aaaba")})
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 0, 0, 1, 0])

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_distinct(self):
        df = DataFrame({"A": list("abcde")})
        g = df.groupby("A")
        sg = g.A

        expected = Series(range(5), dtype="int64")

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_one_group(self):
        df = DataFrame({"A": [0] * 5})
        g = df.groupby("A")
        sg = g.A

        expected = Series([0] * 5)

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_empty(self):
        ge = DataFrame().groupby(level=0)
        se = Series(dtype=object).groupby(level=0)

        # edge case, as this is usually considered float
        e = Series(dtype="int64")

        tm.assert_series_equal(e, ge.ngroup())
        tm.assert_series_equal(e, se.ngroup())

    def test_ngroup_series_matches_frame(self):
        df = DataFrame({"A": list("aaaba")})
        s = Series(list("aaaba"))

        tm.assert_series_equal(df.groupby(s).ngroup(), s.groupby(s).ngroup())

    def test_ngroup_dupe_index(self):
        df = DataFrame({"A": list("aaaba")}, index=[0] * 5)
        g = df.groupby("A")
        sg = g.A

        expected = Series([0, 0, 0, 1, 0], index=[0] * 5)

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_mi(self):
        mi = MultiIndex.from_tuples([[0, 1], [1, 2], [2, 2], [2, 2], [1, 0]])
        df = DataFrame({"A": list("aaaba")}, index=mi)
        g = df.groupby("A")
        sg = g.A
        expected = Series([0, 0, 0, 1, 0], index=mi)

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_groupby_not_col(self):
        df = DataFrame({"A": list("aaaba")}, index=[0] * 5)
        g = df.groupby([0, 0, 0, 1, 0])
        sg = g.A

        expected = Series([0, 0, 0, 1, 0], index=[0] * 5)

        tm.assert_series_equal(expected, g.ngroup())
        tm.assert_series_equal(expected, sg.ngroup())

    def test_ngroup_descending(self):
        df = DataFrame(["a", "a", "b", "a", "b"], columns=["A"])
        g = df.groupby(["A"])

        ascending = Series([0, 0, 1, 0, 1])
        descending = Series([1, 1, 0, 1, 0])

        tm.assert_series_equal(descending, (g.ngroups - 1) - ascending)
        tm.assert_series_equal(ascending, g.ngroup(ascending=True))
        tm.assert_series_equal(descending, g.ngroup(ascending=False))

    def test_ngroup_matches_cumcount(self):
        # verify one manually-worked out case works
        df = DataFrame(
            [["a", "x"], ["a", "y"], ["b", "x"], ["a", "x"], ["b", "y"]],
            columns=["A", "X"],
        )
        g = df.groupby(["A", "X"])
        g_ngroup = g.ngroup()
        g_cumcount = g.cumcount()
        expected_ngroup = Series([0, 1, 2, 0, 3])
        expected_cumcount = Series([0, 0, 0, 1, 0])

        tm.assert_series_equal(g_ngroup, expected_ngroup)
        tm.assert_series_equal(g_cumcount, expected_cumcount)

    def test_ngroup_cumcount_pair(self):
        # brute force comparison for all small series
        for p in product(range(3), repeat=4):
            df = DataFrame({"a": p})
            g = df.groupby(["a"])

            order = sorted(set(p))
            ngroupd = [order.index(val) for val in p]
            cumcounted = [p[:i].count(val) for i, val in enumerate(p)]

            tm.assert_series_equal(g.ngroup(), Series(ngroupd))
            tm.assert_series_equal(g.cumcount(), Series(cumcounted))

    def test_ngroup_respects_groupby_order(self, sort):
        df = DataFrame({"a": np.random.default_rng(2).choice(list("abcdef"), 100)})
        g = df.groupby("a", sort=sort)
        df["group_id"] = -1
        df["group_index"] = -1

        for i, (_, group) in enumerate(g):
            df.loc[group.index, "group_id"] = i
            for j, ind in enumerate(group.index):
                df.loc[ind, "group_index"] = j

        tm.assert_series_equal(Series(df["group_id"].values), g.ngroup())
        tm.assert_series_equal(Series(df["group_index"].values), g.cumcount())

    @pytest.mark.parametrize(
        "datetimelike",
        [
            [Timestamp(f"2016-05-{i:02d} 20:09:25+00:00") for i in range(1, 4)],
            [Timestamp(f"2016-05-{i:02d} 20:09:25") for i in range(1, 4)],
            [Timestamp(f"2016-05-{i:02d} 20:09:25", tz="UTC") for i in range(1, 4)],
            [Timedelta(x, unit="h") for x in range(1, 4)],
            [Period(freq="2W", year=2017, month=x) for x in range(1, 4)],
        ],
    )
    def test_count_with_datetimelike(self, datetimelike):
        # test for #13393, where DataframeGroupBy.count() fails
        # when counting a datetimelike column.

        df = DataFrame({"x": ["a", "a", "b"], "y": datetimelike})
        res = df.groupby("x").count()
        expected = DataFrame({"y": [2, 1]}, index=["a", "b"])
        expected.index.name = "x"
        tm.assert_frame_equal(expected, res)

    def test_count_with_only_nans_in_first_group(self):
        # GH21956
        df = DataFrame({"A": [np.nan, np.nan], "B": ["a", "b"], "C": [1, 2]})
        result = df.groupby(["A", "B"]).C.count()
        mi = MultiIndex(levels=[[], ["a", "b"]], codes=[[], []], names=["A", "B"])
        expected = Series([], index=mi, dtype=np.int64, name="C")
        tm.assert_series_equal(result, expected, check_index_type=False)

    def test_count_groupby_column_with_nan_in_groupby_column(self):
        # https://github.com/pandas-dev/pandas/issues/32841
        df = DataFrame({"A": [1, 1, 1, 1, 1], "B": [5, 4, np.nan, 3, 0]})
        res = df.groupby(["B"]).count()
        expected = DataFrame(
            index=Index([0.0, 3.0, 4.0, 5.0], name="B"), data={"A": [1, 1, 1, 1]}
        )
        tm.assert_frame_equal(expected, res)

    def test_groupby_count_dateparseerror(self):
        dr = date_range(start="1/1/2012", freq="5min", periods=10)

        # BAD Example, datetimes first
        ser = Series(np.arange(10), index=[dr, np.arange(10)])
        grouped = ser.groupby(lambda x: x[1] % 2 == 0)
        result = grouped.count()

        ser = Series(np.arange(10), index=[np.arange(10), dr])
        grouped = ser.groupby(lambda x: x[0] % 2 == 0)
        expected = grouped.count()

        tm.assert_series_equal(result, expected)


def test_groupby_timedelta_cython_count():
    df = DataFrame(
        {"g": list("ab" * 2), "delta": np.arange(4).astype("timedelta64[ns]")}
    )
    expected = Series([2, 2], index=Index(["a", "b"], name="g"), name="delta")
    result = df.groupby("g").delta.count()
    tm.assert_series_equal(expected, result)


def test_count():
    n = 1 << 15
    dr = date_range("2015-08-30", periods=n // 10, freq="min")

    df = DataFrame(
        {
            "1st": np.random.default_rng(2).choice(list(ascii_lowercase), n),
            "2nd": np.random.default_rng(2).integers(0, 5, n),
            "3rd": np.random.default_rng(2).standard_normal(n).round(3),
            "4th": np.random.default_rng(2).integers(-10, 10, n),
            "5th": np.random.default_rng(2).choice(dr, n),
            "6th": np.random.default_rng(2).standard_normal(n).round(3),
            "7th": np.random.default_rng(2).standard_normal(n).round(3),
            "8th": np.random.default_rng(2).choice(dr, n)
            - np.random.default_rng(2).choice(dr, 1),
            "9th": np.random.default_rng(2).choice(list(ascii_lowercase), n),
        }
    )

    for col in df.columns.drop(["1st", "2nd", "4th"]):
        df.loc[np.random.default_rng(2).choice(n, n // 10), col] = np.nan

    df["9th"] = df["9th"].astype("category")

    for key in ["1st", "2nd", ["1st", "2nd"]]:
        left = df.groupby(key).count()
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            right = df.groupby(key).apply(DataFrame.count).drop(key, axis=1)
        tm.assert_frame_equal(left, right)


def test_count_non_nulls():
    # GH#5610
    # count counts non-nulls
    df = DataFrame(
        [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, np.nan]],
        columns=["A", "B", "C"],
    )

    count_as = df.groupby("A").count()
    count_not_as = df.groupby("A", as_index=False).count()

    expected = DataFrame([[1, 2], [0, 0]], columns=["B", "C"], index=[1, 3])
    expected.index.name = "A"
    tm.assert_frame_equal(count_not_as, expected.reset_index())
    tm.assert_frame_equal(count_as, expected)

    count_B = df.groupby("A")["B"].count()
    tm.assert_series_equal(count_B, expected["B"])


def test_count_object():
    df = DataFrame({"a": ["a"] * 3 + ["b"] * 3, "c": [2] * 3 + [3] * 3})
    result = df.groupby("c").a.count()
    expected = Series([3, 3], index=Index([2, 3], name="c"), name="a")
    tm.assert_series_equal(result, expected)

    df = DataFrame({"a": ["a", np.nan, np.nan] + ["b"] * 3, "c": [2] * 3 + [3] * 3})
    result = df.groupby("c").a.count()
    expected = Series([1, 3], index=Index([2, 3], name="c"), name="a")
    tm.assert_series_equal(result, expected)


def test_count_cross_type():
    # GH8169
    # Set float64 dtype to avoid upcast when setting nan below
    vals = np.hstack(
        (
            np.random.default_rng(2).integers(0, 5, (100, 2)),
            np.random.default_rng(2).integers(0, 2, (100, 2)),
        )
    ).astype("float64")

    df = DataFrame(vals, columns=["a", "b", "c", "d"])
    df[df == 2] = np.nan
    expected = df.groupby(["c", "d"]).count()

    for t in ["float32", "object"]:
        df["a"] = df["a"].astype(t)
        df["b"] = df["b"].astype(t)
        result = df.groupby(["c", "d"]).count()
        tm.assert_frame_equal(result, expected)


def test_lower_int_prec_count():
    df = DataFrame(
        {
            "a": np.array([0, 1, 2, 100], np.int8),
            "b": np.array([1, 2, 3, 6], np.uint32),
            "c": np.array([4, 5, 6, 8], np.int16),
            "grp": list("ab" * 2),
        }
    )
    result = df.groupby("grp").count()
    expected = DataFrame(
        {"a": [2, 2], "b": [2, 2], "c": [2, 2]}, index=Index(list("ab"), name="grp")
    )
    tm.assert_frame_equal(result, expected)


def test_count_uses_size_on_exception():
    class RaisingObjectException(Exception):
        pass

    class RaisingObject:
        def __init__(self, msg="I will raise inside Cython") -> None:
            super().__init__()
            self.msg = msg

        def __eq__(self, other):
            # gets called in Cython to check that raising calls the method
            raise RaisingObjectException(self.msg)

    df = DataFrame({"a": [RaisingObject() for _ in range(4)], "grp": list("ab" * 2)})
    result = df.groupby("grp").count()
    expected = DataFrame({"a": [2, 2]}, index=Index(list("ab"), name="grp"))
    tm.assert_frame_equal(result, expected)


def test_count_arrow_string_array(any_string_dtype):
    # GH#54751
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {"a": [1, 2, 3], "b": Series(["a", "b", "a"], dtype=any_string_dtype)}
    )
    result = df.groupby("a").count()
    expected = DataFrame({"b": 1}, index=Index([1, 2, 3], name="a"))
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_category.py ===
import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
)
import pandas._testing as tm
from pandas.core.indexes.api import (
    CategoricalIndex,
    Index,
)


class TestCategoricalIndex:
    @pytest.fixture
    def simple_index(self) -> CategoricalIndex:
        return CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)

    def test_can_hold_identifiers(self):
        idx = CategoricalIndex(list("aabbca"), categories=None, ordered=False)
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is True

    def test_insert(self, simple_index):
        ci = simple_index
        categories = ci.categories

        # test 0th element
        result = ci.insert(0, "a")
        expected = CategoricalIndex(list("aaabbca"), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)

        # test Nth element that follows Python list behavior
        result = ci.insert(-1, "a")
        expected = CategoricalIndex(list("aabbcaa"), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)

        # test empty
        result = CategoricalIndex([], categories=categories).insert(0, "a")
        expected = CategoricalIndex(["a"], categories=categories)
        tm.assert_index_equal(result, expected, exact=True)

        # invalid -> cast to object
        expected = ci.astype(object).insert(0, "d")
        result = ci.insert(0, "d").astype(object)
        tm.assert_index_equal(result, expected, exact=True)

        # GH 18295 (test missing)
        expected = CategoricalIndex(["a", np.nan, "a", "b", "c", "b"])
        for na in (np.nan, pd.NaT, None):
            result = CategoricalIndex(list("aabcb")).insert(1, na)
            tm.assert_index_equal(result, expected)

    def test_insert_na_mismatched_dtype(self):
        ci = CategoricalIndex([0, 1, 1])
        result = ci.insert(0, pd.NaT)
        expected = Index([pd.NaT, 0, 1, 1], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_delete(self, simple_index):
        ci = simple_index
        categories = ci.categories

        result = ci.delete(0)
        expected = CategoricalIndex(list("abbca"), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)

        result = ci.delete(-1)
        expected = CategoricalIndex(list("aabbc"), categories=categories)
        tm.assert_index_equal(result, expected, exact=True)

        with tm.external_error_raised((IndexError, ValueError)):
            # Either depending on NumPy version
            ci.delete(10)

    @pytest.mark.parametrize(
        "data, non_lexsorted_data",
        [[[1, 2, 3], [9, 0, 1, 2, 3]], [list("abc"), list("fabcd")]],
    )
    def test_is_monotonic(self, data, non_lexsorted_data):
        c = CategoricalIndex(data)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False

        c = CategoricalIndex(data, ordered=True)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False

        c = CategoricalIndex(data, categories=reversed(data))
        assert c.is_monotonic_increasing is False
        assert c.is_monotonic_decreasing is True

        c = CategoricalIndex(data, categories=reversed(data), ordered=True)
        assert c.is_monotonic_increasing is False
        assert c.is_monotonic_decreasing is True

        # test when data is neither monotonic increasing nor decreasing
        reordered_data = [data[0], data[2], data[1]]
        c = CategoricalIndex(reordered_data, categories=reversed(data))
        assert c.is_monotonic_increasing is False
        assert c.is_monotonic_decreasing is False

        # non lexsorted categories
        categories = non_lexsorted_data

        c = CategoricalIndex(categories[:2], categories=categories)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False

        c = CategoricalIndex(categories[1:3], categories=categories)
        assert c.is_monotonic_increasing is True
        assert c.is_monotonic_decreasing is False

    def test_has_duplicates(self):
        idx = CategoricalIndex([0, 0, 0], name="foo")
        assert idx.is_unique is False
        assert idx.has_duplicates is True

        idx = CategoricalIndex([0, 1], categories=[2, 3], name="foo")
        assert idx.is_unique is False
        assert idx.has_duplicates is True

        idx = CategoricalIndex([0, 1, 2, 3], categories=[1, 2, 3], name="foo")
        assert idx.is_unique is True
        assert idx.has_duplicates is False

    @pytest.mark.parametrize(
        "data, categories, expected",
        [
            (
                [1, 1, 1],
                [1, 2, 3],
                {
                    "first": np.array([False, True, True]),
                    "last": np.array([True, True, False]),
                    False: np.array([True, True, True]),
                },
            ),
            (
                [1, 1, 1],
                list("abc"),
                {
                    "first": np.array([False, True, True]),
                    "last": np.array([True, True, False]),
                    False: np.array([True, True, True]),
                },
            ),
            (
                [2, "a", "b"],
                list("abc"),
                {
                    "first": np.zeros(shape=(3), dtype=np.bool_),
                    "last": np.zeros(shape=(3), dtype=np.bool_),
                    False: np.zeros(shape=(3), dtype=np.bool_),
                },
            ),
            (
                list("abb"),
                list("abc"),
                {
                    "first": np.array([False, False, True]),
                    "last": np.array([False, True, False]),
                    False: np.array([False, True, True]),
                },
            ),
        ],
    )
    def test_drop_duplicates(self, data, categories, expected):
        idx = CategoricalIndex(data, categories=categories, name="foo")
        for keep, e in expected.items():
            tm.assert_numpy_array_equal(idx.duplicated(keep=keep), e)
            e = idx[~e]
            result = idx.drop_duplicates(keep=keep)
            tm.assert_index_equal(result, e)

    @pytest.mark.parametrize(
        "data, categories, expected_data",
        [
            ([1, 1, 1], [1, 2, 3], [1]),
            ([1, 1, 1], list("abc"), [np.nan]),
            ([1, 2, "a"], [1, 2, 3], [1, 2, np.nan]),
            ([2, "a", "b"], list("abc"), [np.nan, "a", "b"]),
        ],
    )
    def test_unique(self, data, categories, expected_data, ordered):
        dtype = CategoricalDtype(categories, ordered=ordered)

        idx = CategoricalIndex(data, dtype=dtype)
        expected = CategoricalIndex(expected_data, dtype=dtype)
        tm.assert_index_equal(idx.unique(), expected)

    @pytest.mark.xfail(using_string_dtype(), reason="repr doesn't roundtrip")
    def test_repr_roundtrip(self):
        ci = CategoricalIndex(["a", "b"], categories=["a", "b"], ordered=True)
        str(ci)
        tm.assert_index_equal(eval(repr(ci)), ci, exact=True)

        # formatting
        str(ci)

        # long format
        # this is not reprable
        ci = CategoricalIndex(np.random.default_rng(2).integers(0, 5, size=100))
        str(ci)

    def test_isin(self):
        ci = CategoricalIndex(list("aabca") + [np.nan], categories=["c", "a", "b"])
        tm.assert_numpy_array_equal(
            ci.isin(["c"]), np.array([False, False, False, True, False, False])
        )
        tm.assert_numpy_array_equal(
            ci.isin(["c", "a", "b"]), np.array([True] * 5 + [False])
        )
        tm.assert_numpy_array_equal(
            ci.isin(["c", "a", "b", np.nan]), np.array([True] * 6)
        )

        # mismatched categorical -> coerced to ndarray so doesn't matter
        result = ci.isin(ci.set_categories(list("abcdefghi")))
        expected = np.array([True] * 6)
        tm.assert_numpy_array_equal(result, expected)

        result = ci.isin(ci.set_categories(list("defghi")))
        expected = np.array([False] * 5 + [True])
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_overlapping_intervals(self):
        # GH 34974
        idx = pd.IntervalIndex([pd.Interval(0, 2), pd.Interval(0, 1)])
        result = CategoricalIndex(idx).isin(idx)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_identical(self):
        ci1 = CategoricalIndex(["a", "b"], categories=["a", "b"], ordered=True)
        ci2 = CategoricalIndex(["a", "b"], categories=["a", "b", "c"], ordered=True)
        assert ci1.identical(ci1)
        assert ci1.identical(ci1.copy())
        assert not ci1.identical(ci2)

    def test_ensure_copied_data(self):
        # gh-12309: Check the "copy" argument of each
        # Index.__new__ is honored.
        #
        # Must be tested separately from other indexes because
        # self.values is not an ndarray.
        index = CategoricalIndex(list("ab") * 5)

        result = CategoricalIndex(index.values, copy=True)
        tm.assert_index_equal(index, result)
        assert not np.shares_memory(result._data._codes, index._data._codes)

        result = CategoricalIndex(index.values, copy=False)
        assert result._data._codes is index._data._codes


class TestCategoricalIndex2:
    def test_view_i8(self):
        # GH#25464
        ci = CategoricalIndex(list("ab") * 50)
        msg = "When changing to a larger dtype, its size must be a divisor"
        with pytest.raises(ValueError, match=msg):
            ci.view("i8")
        with pytest.raises(ValueError, match=msg):
            ci._data.view("i8")

        ci = ci[:-4]  # length divisible by 8

        res = ci.view("i8")
        expected = ci._data.codes.view("i8")
        tm.assert_numpy_array_equal(res, expected)

        cat = ci._data
        tm.assert_numpy_array_equal(cat.view("i8"), expected)

    @pytest.mark.parametrize(
        "dtype, engine_type",
        [
            (np.int8, libindex.Int8Engine),
            (np.int16, libindex.Int16Engine),
            (np.int32, libindex.Int32Engine),
            (np.int64, libindex.Int64Engine),
        ],
    )
    def test_engine_type(self, dtype, engine_type):
        if dtype != np.int64:
            # num. of uniques required to push CategoricalIndex.codes to a
            # dtype (128 categories required for .codes dtype to be int16 etc.)
            num_uniques = {np.int8: 1, np.int16: 128, np.int32: 32768}[dtype]
            ci = CategoricalIndex(range(num_uniques))
        else:
            # having 2**32 - 2**31 categories would be very memory-intensive,
            # so we cheat a bit with the dtype
            ci = CategoricalIndex(range(32768))  # == 2**16 - 2**(16 - 1)
            arr = ci.values._ndarray.astype("int64")
            NDArrayBacked.__init__(ci._data, arr, ci.dtype)
        assert np.issubdtype(ci.codes.dtype, dtype)
        assert isinstance(ci._engine, engine_type)

    @pytest.mark.parametrize(
        "func,op_name",
        [
            (lambda idx: idx - idx, "__sub__"),
            (lambda idx: idx + idx, "__add__"),
            (lambda idx: idx - ["a", "b"], "__sub__"),
            (lambda idx: idx + ["a", "b"], "__add__"),
            (lambda idx: ["a", "b"] - idx, "__rsub__"),
            (lambda idx: ["a", "b"] + idx, "__radd__"),
        ],
    )
    def test_disallow_addsub_ops(self, func, op_name):
        # GH 10039
        # set ops (+/-) raise TypeError
        idx = Index(Categorical(["a", "b"]))
        cat_or_list = "'(Categorical|list)' and '(Categorical|list)'"
        msg = "|".join(
            [
                f"cannot perform {op_name} with this index type: CategoricalIndex",
                "can only concatenate list",
                rf"unsupported operand type\(s\) for [\+-]: {cat_or_list}",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            func(idx)

    def test_method_delegation(self):
        ci = CategoricalIndex(list("aabbca"), categories=list("cabdef"))
        result = ci.set_categories(list("cab"))
        tm.assert_index_equal(
            result, CategoricalIndex(list("aabbca"), categories=list("cab"))
        )

        ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
        result = ci.rename_categories(list("efg"))
        tm.assert_index_equal(
            result, CategoricalIndex(list("ffggef"), categories=list("efg"))
        )

        # GH18862 (let rename_categories take callables)
        result = ci.rename_categories(lambda x: x.upper())
        tm.assert_index_equal(
            result, CategoricalIndex(list("AABBCA"), categories=list("CAB"))
        )

        ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
        result = ci.add_categories(["d"])
        tm.assert_index_equal(
            result, CategoricalIndex(list("aabbca"), categories=list("cabd"))
        )

        ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
        result = ci.remove_categories(["c"])
        tm.assert_index_equal(
            result,
            CategoricalIndex(list("aabb") + [np.nan] + ["a"], categories=list("ab")),
        )

        ci = CategoricalIndex(list("aabbca"), categories=list("cabdef"))
        result = ci.as_unordered()
        tm.assert_index_equal(result, ci)

        ci = CategoricalIndex(list("aabbca"), categories=list("cabdef"))
        result = ci.as_ordered()
        tm.assert_index_equal(
            result,
            CategoricalIndex(list("aabbca"), categories=list("cabdef"), ordered=True),
        )

        # invalid
        msg = "cannot use inplace with CategoricalIndex"
        with pytest.raises(ValueError, match=msg):
            ci.set_categories(list("cab"), inplace=True)

    def test_remove_maintains_order(self):
        ci = CategoricalIndex(list("abcdda"), categories=list("abcd"))
        result = ci.reorder_categories(["d", "c", "b", "a"], ordered=True)
        tm.assert_index_equal(
            result,
            CategoricalIndex(list("abcdda"), categories=list("dcba"), ordered=True),
        )
        result = result.remove_categories(["c"])
        tm.assert_index_equal(
            result,
            CategoricalIndex(
                ["a", "b", np.nan, "d", "d", "a"], categories=list("dba"), ordered=True
            ),
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_frame_equal.py ===
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm


@pytest.fixture(params=[True, False])
def by_blocks_fixture(request):
    return request.param


@pytest.fixture(params=["DataFrame", "Series"])
def obj_fixture(request):
    return request.param


def _assert_frame_equal_both(a, b, **kwargs):
    """
    Check that two DataFrame equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : DataFrame
        The first DataFrame to compare.
    b : DataFrame
        The second DataFrame to compare.
    kwargs : dict
        The arguments passed to `tm.assert_frame_equal`.
    """
    tm.assert_frame_equal(a, b, **kwargs)
    tm.assert_frame_equal(b, a, **kwargs)


@pytest.mark.parametrize("check_like", [True, False])
def test_frame_equal_row_order_mismatch(check_like, obj_fixture):
    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = DataFrame({"A": [3, 2, 1], "B": [6, 5, 4]}, index=["c", "b", "a"])

    if not check_like:  # Do not ignore row-column orderings.
        msg = f"{obj_fixture}.index are different"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, check_like=check_like, obj=obj_fixture)
    else:
        _assert_frame_equal_both(df1, df2, check_like=check_like, obj=obj_fixture)


@pytest.mark.parametrize(
    "df1,df2",
    [
        (DataFrame({"A": [1, 2, 3]}), DataFrame({"A": [1, 2, 3, 4]})),
        (DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}), DataFrame({"A": [1, 2, 3]})),
    ],
)
def test_frame_equal_shape_mismatch(df1, df2, obj_fixture):
    msg = f"{obj_fixture} are different"

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, obj=obj_fixture)


@pytest.mark.parametrize(
    "df1,df2,msg",
    [
        # Index
        (
            DataFrame.from_records({"a": [1, 2], "c": ["l1", "l2"]}, index=["a"]),
            DataFrame.from_records({"a": [1.0, 2.0], "c": ["l1", "l2"]}, index=["a"]),
            "DataFrame\\.index are different",
        ),
        # MultiIndex
        (
            DataFrame.from_records(
                {"a": [1, 2], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ),
            DataFrame.from_records(
                {"a": [1.0, 2.0], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ),
            "MultiIndex level \\[0\\] are different",
        ),
    ],
)
def test_frame_equal_index_dtype_mismatch(df1, df2, msg, check_index_type):
    kwargs = {"check_index_type": check_index_type}

    if check_index_type:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, **kwargs)
    else:
        tm.assert_frame_equal(df1, df2, **kwargs)


def test_empty_dtypes(check_dtype):
    columns = ["col1", "col2"]
    df1 = DataFrame(columns=columns)
    df2 = DataFrame(columns=columns)

    kwargs = {"check_dtype": check_dtype}
    df1["col1"] = df1["col1"].astype("int64")

    if check_dtype:
        msg = r"Attributes of DataFrame\..* are different"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, **kwargs)
    else:
        tm.assert_frame_equal(df1, df2, **kwargs)


@pytest.mark.parametrize("check_like", [True, False])
def test_frame_equal_index_mismatch(check_like, obj_fixture, using_infer_string):
    if using_infer_string:
        dtype = "str"
    else:
        dtype = "object"
    msg = f"""{obj_fixture}\\.index are different

{obj_fixture}\\.index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\['a', 'b', 'c'\\], dtype='{dtype}'\\)
\\[right\\]: Index\\(\\['a', 'b', 'd'\\], dtype='{dtype}'\\)
At positional index 2, first diff: c != d"""

    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "d"])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, check_like=check_like, obj=obj_fixture)


@pytest.mark.parametrize("check_like", [True, False])
def test_frame_equal_columns_mismatch(check_like, obj_fixture, using_infer_string):
    if using_infer_string:
        dtype = "str"
    else:
        dtype = "object"
    msg = f"""{obj_fixture}\\.columns are different

{obj_fixture}\\.columns values are different \\(50\\.0 %\\)
\\[left\\]:  Index\\(\\['A', 'B'\\], dtype='{dtype}'\\)
\\[right\\]: Index\\(\\['A', 'b'\\], dtype='{dtype}'\\)"""

    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = DataFrame({"A": [1, 2, 3], "b": [4, 5, 6]}, index=["a", "b", "c"])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, check_like=check_like, obj=obj_fixture)


def test_frame_equal_block_mismatch(by_blocks_fixture, obj_fixture):
    obj = obj_fixture
    msg = f"""{obj}\\.iloc\\[:, 1\\] \\(column name="B"\\) are different

{obj}\\.iloc\\[:, 1\\] \\(column name="B"\\) values are different \\(33\\.33333 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[4, 5, 6\\]
\\[right\\]: \\[4, 5, 7\\]"""

    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 7]})

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, by_blocks=by_blocks_fixture, obj=obj_fixture)


@pytest.mark.parametrize(
    "df1,df2,msg",
    [
        (
            DataFrame({"A": ["á", "à", "ä"], "E": ["é", "è", "ë"]}),
            DataFrame({"A": ["á", "à", "ä"], "E": ["é", "è", "e̊"]}),
            """{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) are different

{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) values are different \\(33\\.33333 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[é, è, ë\\]
\\[right\\]: \\[é, è, e̊\\]""",
        ),
        (
            DataFrame({"A": ["á", "à", "ä"], "E": ["é", "è", "ë"]}),
            DataFrame({"A": ["a", "a", "a"], "E": ["e", "e", "e"]}),
            """{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) are different

{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) values are different \\(100\\.0 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[a, a, a\\]""",
        ),
    ],
)
def test_frame_equal_unicode(df1, df2, msg, by_blocks_fixture, obj_fixture):
    # see gh-20503
    #
    # Test ensures that `tm.assert_frame_equals` raises the right exception
    # when comparing DataFrames containing differing unicode objects.
    msg = msg.format(obj=obj_fixture)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, by_blocks=by_blocks_fixture, obj=obj_fixture)


def test_assert_frame_equal_extension_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/32747
    left = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    right = left.astype(int)

    msg = (
        "Attributes of DataFrame\\.iloc\\[:, 0\\] "
        '\\(column name="a"\\) are different\n\n'
        'Attribute "dtype" are different\n'
        "\\[left\\]:  Int64\n"
        "\\[right\\]: int[32|64]"
    )

    tm.assert_frame_equal(left, right, check_dtype=False)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(left, right, check_dtype=True)


def test_assert_frame_equal_interval_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/32747
    left = DataFrame({"a": [pd.Interval(0, 1)]}, dtype="interval")
    right = left.astype(object)

    msg = (
        "Attributes of DataFrame\\.iloc\\[:, 0\\] "
        '\\(column name="a"\\) are different\n\n'
        'Attribute "dtype" are different\n'
        "\\[left\\]:  interval\\[int64, right\\]\n"
        "\\[right\\]: object"
    )

    tm.assert_frame_equal(left, right, check_dtype=False)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(left, right, check_dtype=True)


def test_assert_frame_equal_ignore_extension_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/35715
    left = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    right = DataFrame({"a": [1, 2, 3]}, dtype="Int32")
    tm.assert_frame_equal(left, right, check_dtype=False)


def test_assert_frame_equal_ignore_extension_dtype_mismatch_cross_class():
    # https://github.com/pandas-dev/pandas/issues/35715
    left = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    right = DataFrame({"a": [1, 2, 3]}, dtype="int64")
    tm.assert_frame_equal(left, right, check_dtype=False)


@pytest.mark.parametrize(
    "dtype",
    [
        ("timedelta64[ns]"),
        ("datetime64[ns, UTC]"),
        ("Period[D]"),
    ],
)
def test_assert_frame_equal_datetime_like_dtype_mismatch(dtype):
    df1 = DataFrame({"a": []}, dtype=dtype)
    df2 = DataFrame({"a": []})
    tm.assert_frame_equal(df1, df2, check_dtype=False)


def test_allows_duplicate_labels():
    left = DataFrame()
    right = DataFrame().set_flags(allows_duplicate_labels=False)
    tm.assert_frame_equal(left, left)
    tm.assert_frame_equal(right, right)
    tm.assert_frame_equal(left, right, check_flags=False)
    tm.assert_frame_equal(right, left, check_flags=False)

    with pytest.raises(AssertionError, match="<Flags"):
        tm.assert_frame_equal(left, right)

    with pytest.raises(AssertionError, match="<Flags"):
        tm.assert_frame_equal(left, right)


def test_assert_frame_equal_columns_mixed_dtype():
    # GH#39168
    df = DataFrame([[0, 1, 2]], columns=["foo", "bar", 42], index=[1, "test", 2])
    tm.assert_frame_equal(df, df, check_like=True)


def test_frame_equal_extension_dtype(frame_or_series, any_numeric_ea_dtype):
    # GH#39410
    obj = frame_or_series([1, 2], dtype=any_numeric_ea_dtype)
    tm.assert_equal(obj, obj, check_exact=True)


@pytest.mark.parametrize("indexer", [(0, 1), (1, 0)])
def test_frame_equal_mixed_dtypes(frame_or_series, any_numeric_ea_dtype, indexer):
    dtypes = (any_numeric_ea_dtype, "int64")
    obj1 = frame_or_series([1, 2], dtype=dtypes[indexer[0]])
    obj2 = frame_or_series([1, 2], dtype=dtypes[indexer[1]])
    tm.assert_equal(obj1, obj2, check_exact=True, check_dtype=False)


def test_assert_frame_equal_check_like_different_indexes():
    # GH#39739
    df1 = DataFrame(index=pd.Index([], dtype="object"))
    df2 = DataFrame(index=pd.RangeIndex(start=0, stop=0, step=1))
    with pytest.raises(AssertionError, match="DataFrame.index are different"):
        tm.assert_frame_equal(df1, df2, check_like=True)


def test_assert_frame_equal_checking_allow_dups_flag():
    # GH#45554
    left = DataFrame([[1, 2], [3, 4]])
    left.flags.allows_duplicate_labels = False

    right = DataFrame([[1, 2], [3, 4]])
    right.flags.allows_duplicate_labels = True
    tm.assert_frame_equal(left, right, check_flags=False)

    with pytest.raises(AssertionError, match="allows_duplicate_labels"):
        tm.assert_frame_equal(left, right, check_flags=True)


def test_assert_frame_equal_check_like_categorical_midx():
    # GH#48975
    left = DataFrame(
        [[1], [2], [3]],
        index=pd.MultiIndex.from_arrays(
            [
                pd.Categorical(["a", "b", "c"]),
                pd.Categorical(["a", "b", "c"]),
            ]
        ),
    )
    right = DataFrame(
        [[3], [2], [1]],
        index=pd.MultiIndex.from_arrays(
            [
                pd.Categorical(["c", "b", "a"]),
                pd.Categorical(["c", "b", "a"]),
            ]
        ),
    )
    tm.assert_frame_equal(left, right, check_like=True)


def test_assert_frame_equal_ea_column_definition_in_exception_mask():
    # GH#50323
    df1 = DataFrame({"a": pd.Series([pd.NA, 1], dtype="Int64")})
    df2 = DataFrame({"a": pd.Series([1, 1], dtype="Int64")})

    msg = r'DataFrame.iloc\[:, 0\] \(column name="a"\) NA mask values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)


def test_assert_frame_equal_ea_column_definition_in_exception():
    # GH#50323
    df1 = DataFrame({"a": pd.Series([pd.NA, 1], dtype="Int64")})
    df2 = DataFrame({"a": pd.Series([pd.NA, 2], dtype="Int64")})

    msg = r'DataFrame.iloc\[:, 0\] \(column name="a"\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, check_exact=True)


def test_assert_frame_equal_ts_column():
    # GH#50323
    df1 = DataFrame({"a": [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-12-31")]})
    df2 = DataFrame({"a": [pd.Timestamp("2020-12-31"), pd.Timestamp("2020-12-31")]})

    msg = r'DataFrame.iloc\[:, 0\] \(column name="a"\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)


def test_assert_frame_equal_set():
    # GH#51727
    df1 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 6}]})
    df2 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 6}]})
    tm.assert_frame_equal(df1, df2)


def test_assert_frame_equal_set_mismatch():
    # GH#51727
    df1 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 6}]})
    df2 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 7}]})

    msg = r'DataFrame.iloc\[:, 0\] \(column name="set_column"\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\tests\test_util.py ===
from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import (Abs, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import frac
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
    cos, cot, csc, sec, sin, tan, asin, acos, atan, acot, asec, acsc)
from sympy.functions.elementary.hyperbolic import (sinh, cosh, tanh, coth,
    sech, csch, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.error_functions import expint
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.simplify import simplify
from sympy.calculus.util import (function_range, continuous_domain, not_empty_in,
                                 periodicity, lcim, is_convex,
                                 stationary_points, minimum, maximum)
from sympy.sets.sets import (Interval, FiniteSet, Complement, Union)
from sympy.sets.fancysets import ImageSet
from sympy.sets.conditionset import ConditionSet
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow, slow
from sympy.abc import x, y

a = Symbol('a', real=True)

def test_function_range():
    assert function_range(sin(x), x, Interval(-pi/2, pi/2)
        ) == Interval(-1, 1)
    assert function_range(sin(x), x, Interval(0, pi)
        ) == Interval(0, 1)
    assert function_range(tan(x), x, Interval(0, pi)
        ) == Interval(-oo, oo)
    assert function_range(tan(x), x, Interval(pi/2, pi)
        ) == Interval(-oo, 0)
    assert function_range((x + 3)/(x - 2), x, Interval(-5, 5)
        ) == Union(Interval(-oo, Rational(2, 7)), Interval(Rational(8, 3), oo))
    assert function_range(1/(x**2), x, Interval(-1, 1)
        ) == Interval(1, oo)
    assert function_range(exp(x), x, Interval(-1, 1)
        ) == Interval(exp(-1), exp(1))
    assert function_range(log(x) - x, x, S.Reals
        ) == Interval(-oo, -1)
    assert function_range(sqrt(3*x - 1), x, Interval(0, 2)
        ) == Interval(0, sqrt(5))
    assert function_range(x*(x - 1) - (x**2 - x), x, S.Reals
        ) == FiniteSet(0)
    assert function_range(x*(x - 1) - (x**2 - x) + y, x, S.Reals
        ) == FiniteSet(y)
    assert function_range(sin(x), x, Union(Interval(-5, -3), FiniteSet(4))
        ) == Union(Interval(-sin(3), 1), FiniteSet(sin(4)))
    assert function_range(cos(x), x, Interval(-oo, -4)
        ) == Interval(-1, 1)
    assert function_range(cos(x), x, S.EmptySet) == S.EmptySet
    assert function_range(x/sqrt(x**2+1), x, S.Reals) == Interval.open(-1,1)
    raises(NotImplementedError, lambda : function_range(
        exp(x)*(sin(x) - cos(x))/2 - x, x, S.Reals))
    raises(NotImplementedError, lambda : function_range(
        sin(x) + x, x, S.Reals)) # issue 13273
    raises(NotImplementedError, lambda : function_range(
        log(x), x, S.Integers))
    raises(NotImplementedError, lambda : function_range(
        sin(x)/2, x, S.Naturals))


@slow
def test_function_range1():
    assert function_range(tan(x)**2 + tan(3*x)**2 + 1, x, S.Reals) == Interval(1,oo)


def test_continuous_domain():
    assert continuous_domain(sin(x), x, Interval(0, 2*pi)) == Interval(0, 2*pi)
    assert continuous_domain(tan(x), x, Interval(0, 2*pi)) == \
        Union(Interval(0, pi/2, False, True), Interval(pi/2, pi*Rational(3, 2), True, True),
              Interval(pi*Rational(3, 2), 2*pi, True, False))
    assert continuous_domain(cot(x), x, Interval(0, 2*pi)) == Union(
        Interval.open(0, pi), Interval.open(pi, 2*pi))
    assert continuous_domain((x - 1)/((x - 1)**2), x, S.Reals) == \
        Union(Interval(-oo, 1, True, True), Interval(1, oo, True, True))
    assert continuous_domain(log(x) + log(4*x - 1), x, S.Reals) == \
        Interval(Rational(1, 4), oo, True, True)
    assert continuous_domain(1/sqrt(x - 3), x, S.Reals) == Interval(3, oo, True, True)
    assert continuous_domain(1/x - 2, x, S.Reals) == \
        Union(Interval.open(-oo, 0), Interval.open(0, oo))
    assert continuous_domain(1/(x**2 - 4) + 2, x, S.Reals) == \
        Union(Interval.open(-oo, -2), Interval.open(-2, 2), Interval.open(2, oo))
    assert continuous_domain((x+1)**pi, x, S.Reals) == Interval(-1, oo)
    assert continuous_domain((x+1)**(pi/2), x, S.Reals) == Interval(-1, oo)
    assert continuous_domain(x**x, x, S.Reals) == Interval(0, oo)
    assert continuous_domain((x+1)**log(x**2), x, S.Reals) == Union(
        Interval.Ropen(-1, 0), Interval.open(0, oo))
    domain = continuous_domain(log(tan(x)**2 + 1), x, S.Reals)
    assert not domain.contains(3*pi/2)
    assert domain.contains(5)
    d = Symbol('d', even=True, zero=False)
    assert continuous_domain(x**(1/d), x, S.Reals) == Interval(0, oo)
    n = Dummy('n')
    assert continuous_domain(1/sin(x), x, S.Reals).dummy_eq(Complement(
        S.Reals, Union(ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
                       ImageSet(Lambda(n, 2*n*pi), S.Integers))))
    assert continuous_domain(sin(x) + cos(x), x, S.Reals) == S.Reals
    assert continuous_domain(asin(x), x, S.Reals) == Interval(-1, 1) # issue #21786
    assert continuous_domain(1/acos(log(x)), x, S.Reals) == Interval.Ropen(exp(-1), E)
    assert continuous_domain(sinh(x)+cosh(x), x, S.Reals) == S.Reals
    assert continuous_domain(tanh(x)+sech(x), x, S.Reals) == S.Reals
    assert continuous_domain(atan(x)+asinh(x), x, S.Reals) == S.Reals
    assert continuous_domain(acosh(x), x, S.Reals) == Interval(1, oo)
    assert continuous_domain(atanh(x), x, S.Reals) == Interval.open(-1, 1)
    assert continuous_domain(atanh(x)+acosh(x), x, S.Reals) == S.EmptySet
    assert continuous_domain(asech(x), x, S.Reals) == Interval.Lopen(0, 1)
    assert continuous_domain(acoth(x), x, S.Reals) == Union(
        Interval.open(-oo, -1), Interval.open(1, oo))
    assert continuous_domain(asec(x), x, S.Reals) == Union(
        Interval(-oo, -1), Interval(1, oo))
    assert continuous_domain(acsc(x), x, S.Reals) == Union(
        Interval(-oo, -1), Interval(1, oo))
    for f in (coth, acsch, csch):
        assert continuous_domain(f(x), x, S.Reals) == Union(
            Interval.open(-oo, 0), Interval.open(0, oo))
    assert continuous_domain(acot(x), x, S.Reals).contains(0) == False
    assert continuous_domain(1/(exp(x) - x), x, S.Reals) == Complement(
        S.Reals, ConditionSet(x, Eq(-x + exp(x), 0), S.Reals))
    assert continuous_domain(frac(x**2), x, Interval(-2,-1)) == Union(
        Interval.open(-2, -sqrt(3)), Interval.open(-sqrt(2), -1),
        Interval.open(-sqrt(3), -sqrt(2)))
    assert continuous_domain(frac(x), x, S.Reals) == Complement(
        S.Reals, S.Integers)
    raises(NotImplementedError, lambda : continuous_domain(
        1/(x**2+1), x, S.Complexes))
    raises(NotImplementedError, lambda : continuous_domain(
        gamma(x), x, Interval(-5,0)))
    assert continuous_domain(x + gamma(pi), x, S.Reals) == S.Reals


@XFAIL
def test_continuous_domain_acot():
    acot_cont = Piecewise((pi+acot(x), x<0), (acot(x), True))
    assert continuous_domain(acot_cont, x, S.Reals) == S.Reals

@XFAIL
def test_continuous_domain_gamma():
    assert continuous_domain(gamma(x), x, S.Reals).contains(-1) == False

@XFAIL
def test_continuous_domain_neg_power():
    assert continuous_domain((x-2)**(1-x), x, S.Reals) == Interval.open(2, oo)


def test_not_empty_in():
    assert not_empty_in(FiniteSet(x, 2*x).intersect(Interval(1, 2, True, False)), x) == \
        Interval(S.Half, 2, True, False)
    assert not_empty_in(FiniteSet(x, x**2).intersect(Interval(1, 2)), x) == \
        Union(Interval(-sqrt(2), -1), Interval(1, 2))
    assert not_empty_in(FiniteSet(x**2 + x, x).intersect(Interval(2, 4)), x) == \
        Union(Interval(-sqrt(17)/2 - S.Half, -2),
              Interval(1, Rational(-1, 2) + sqrt(17)/2), Interval(2, 4))
    assert not_empty_in(FiniteSet(x/(x - 1)).intersect(S.Reals), x) == \
        Complement(S.Reals, FiniteSet(1))
    assert not_empty_in(FiniteSet(a/(a - 1)).intersect(S.Reals), a) == \
        Complement(S.Reals, FiniteSet(1))
    assert not_empty_in(FiniteSet((x**2 - 3*x + 2)/(x - 1)).intersect(S.Reals), x) == \
        Complement(S.Reals, FiniteSet(1))
    assert not_empty_in(FiniteSet(3, 4, x/(x - 1)).intersect(Interval(2, 3)), x) == \
        Interval(-oo, oo)
    assert not_empty_in(FiniteSet(4, x/(x - 1)).intersect(Interval(2, 3)), x) == \
        Interval(S(3)/2, 2)
    assert not_empty_in(FiniteSet(x/(x**2 - 1)).intersect(S.Reals), x) == \
        Complement(S.Reals, FiniteSet(-1, 1))
    assert not_empty_in(FiniteSet(x, x**2).intersect(Union(Interval(1, 3, True, True),
                                                           Interval(4, 5))), x) == \
        Union(Interval(-sqrt(5), -2), Interval(-sqrt(3), -1, True, True),
              Interval(1, 3, True, True), Interval(4, 5))
    assert not_empty_in(FiniteSet(1).intersect(Interval(3, 4)), x) == S.EmptySet
    assert not_empty_in(FiniteSet(x**2/(x + 2)).intersect(Interval(1, oo)), x) == \
        Union(Interval(-2, -1, True, False), Interval(2, oo))
    raises(ValueError, lambda: not_empty_in(x))
    raises(ValueError, lambda: not_empty_in(Interval(0, 1), x))
    raises(NotImplementedError,
           lambda: not_empty_in(FiniteSet(x).intersect(S.Reals), x, a))


@_both_exp_pow
def test_periodicity():
    assert periodicity(sin(2*x), x) == pi
    assert periodicity((-2)*tan(4*x), x) == pi/4
    assert periodicity(sin(x)**2, x) == 2*pi
    assert periodicity(3**tan(3*x), x) == pi/3
    assert periodicity(tan(x)*cos(x), x) == 2*pi
    assert periodicity(sin(x)**(tan(x)), x) == 2*pi
    assert periodicity(tan(x)*sec(x), x) == 2*pi
    assert periodicity(sin(2*x)*cos(2*x) - y, x) == pi/2
    assert periodicity(tan(x) + cot(x), x) == pi
    assert periodicity(sin(x) - cos(2*x), x) == 2*pi
    assert periodicity(sin(x) - 1, x) == 2*pi
    assert periodicity(sin(4*x) + sin(x)*cos(x), x) == pi
    assert periodicity(exp(sin(x)), x) == 2*pi
    assert periodicity(log(cot(2*x)) - sin(cos(2*x)), x) == pi
    assert periodicity(sin(2*x)*exp(tan(x) - csc(2*x)), x) == pi
    assert periodicity(cos(sec(x) - csc(2*x)), x) == 2*pi
    assert periodicity(tan(sin(2*x)), x) == pi
    assert periodicity(2*tan(x)**2, x) == pi
    assert periodicity(sin(x%4), x) == 4
    assert periodicity(sin(x)%4, x) == 2*pi
    assert periodicity(tan((3*x-2)%4), x) == Rational(4, 3)
    assert periodicity((sqrt(2)*(x+1)+x) % 3, x) == 3 / (sqrt(2)+1)
    assert periodicity((x**2+1) % x, x) is None
    assert periodicity(sin(re(x)), x) == 2*pi
    assert periodicity(sin(x)**2 + cos(x)**2, x) is S.Zero
    assert periodicity(tan(x), y) is S.Zero
    assert periodicity(sin(x) + I*cos(x), x) == 2*pi
    assert periodicity(x - sin(2*y), y) == pi

    assert periodicity(exp(x), x) is None
    assert periodicity(exp(I*x), x) == 2*pi
    assert periodicity(exp(I*a), a) == 2*pi
    assert periodicity(exp(a), a) is None
    assert periodicity(exp(log(sin(a) + I*cos(2*a)), evaluate=False), a) == 2*pi
    assert periodicity(exp(log(sin(2*a) + I*cos(a)), evaluate=False), a) == 2*pi
    assert periodicity(exp(sin(a)), a) == 2*pi
    assert periodicity(exp(2*I*a), a) == pi
    assert periodicity(exp(a + I*sin(a)), a) is None
    assert periodicity(exp(cos(a/2) + sin(a)), a) == 4*pi
    assert periodicity(log(x), x) is None
    assert periodicity(exp(x)**sin(x), x) is None
    assert periodicity(sin(x)**y, y) is None

    assert periodicity(Abs(sin(Abs(sin(x)))), x) == pi
    assert all(periodicity(Abs(f(x)), x) == pi for f in (
        cos, sin, sec, csc, tan, cot))
    assert periodicity(Abs(sin(tan(x))), x) == pi
    assert periodicity(Abs(sin(sin(x) + tan(x))), x) == 2*pi
    assert periodicity(sin(x) > S.Half, x) == 2*pi

    assert periodicity(x > 2, x) is None
    assert periodicity(x**3 - x**2 + 1, x) is None
    assert periodicity(Abs(x), x) is None
    assert periodicity(Abs(x**2 - 1), x) is None

    assert periodicity((x**2 + 4)%2, x) is None
    assert periodicity((E**x)%3, x) is None

    assert periodicity(sin(expint(1, x))/expint(1, x), x) is None
    # returning `None` for any Piecewise
    p = Piecewise((0, x < -1), (x**2, x <= 1), (log(x), True))
    assert periodicity(p, x) is None

    m = MatrixSymbol('m', 3, 3)
    raises(NotImplementedError, lambda: periodicity(sin(m), m))
    raises(NotImplementedError, lambda: periodicity(sin(m[0, 0]), m))
    raises(NotImplementedError, lambda: periodicity(sin(m), m[0, 0]))
    raises(NotImplementedError, lambda: periodicity(sin(m[0, 0]), m[0, 0]))


def test_periodicity_check():
    assert periodicity(tan(x), x, check=True) == pi
    assert periodicity(sin(x) + cos(x), x, check=True) == 2*pi
    assert periodicity(sec(x), x) == 2*pi
    assert periodicity(sin(x*y), x) == 2*pi/abs(y)
    assert periodicity(Abs(sec(sec(x))), x) == pi


def test_lcim():
    assert lcim([S.Half, S(2), S(3)]) == 6
    assert lcim([pi/2, pi/4, pi]) == pi
    assert lcim([2*pi, pi/2]) == 2*pi
    assert lcim([S.One, 2*pi]) is None
    assert lcim([S(2) + 2*E, E/3 + Rational(1, 3), S.One + E]) == S(2) + 2*E


def test_is_convex():
    assert is_convex(1/x, x, domain=Interval.open(0, oo)) == True
    assert is_convex(1/x, x, domain=Interval(-oo, 0)) == False
    assert is_convex(x**2, x, domain=Interval(0, oo)) == True
    assert is_convex(1/x**3, x, domain=Interval.Lopen(0, oo)) == True
    assert is_convex(-1/x**3, x, domain=Interval.Ropen(-oo, 0)) == True
    assert is_convex(log(x) ,x) == False
    assert is_convex(x**2+y**2, x, y) == True
    assert is_convex(cos(x) + cos(y), x) == False
    assert is_convex(8*x**2 - 2*y**2, x, y) == False


def test_stationary_points():
    assert stationary_points(sin(x), x, Interval(-pi/2, pi/2)
        ) == {-pi/2, pi/2}
    assert  stationary_points(sin(x), x, Interval.Ropen(0, pi/4)
        ) is S.EmptySet
    assert stationary_points(tan(x), x,
        ) is S.EmptySet
    assert stationary_points(sin(x)*cos(x), x, Interval(0, pi)
        ) == {pi/4, pi*Rational(3, 4)}
    assert stationary_points(sec(x), x, Interval(0, pi)
        ) == {0, pi}
    assert stationary_points((x+3)*(x-2), x
        ) == FiniteSet(Rational(-1, 2))
    assert stationary_points((x + 3)/(x - 2), x, Interval(-5, 5)
        ) is S.EmptySet
    assert stationary_points((x**2+3)/(x-2), x
        ) == {2 - sqrt(7), 2 + sqrt(7)}
    assert stationary_points((x**2+3)/(x-2), x, Interval(0, 5)
        ) == {2 + sqrt(7)}
    assert stationary_points(x**4 + x**3 - 5*x**2, x, S.Reals
        ) == FiniteSet(-2, 0, Rational(5, 4))
    assert stationary_points(exp(x), x
        ) is S.EmptySet
    assert stationary_points(log(x) - x, x, S.Reals
        ) == {1}
    assert stationary_points(cos(x), x, Union(Interval(0, 5), Interval(-6, -3))
        ) == {0, -pi, pi}
    assert stationary_points(y, x, S.Reals
        ) == S.Reals
    assert stationary_points(y, x, S.EmptySet) == S.EmptySet


def test_maximum():
    assert maximum(sin(x), x) is S.One
    assert maximum(sin(x), x, Interval(0, 1)) == sin(1)
    assert maximum(tan(x), x) is oo
    assert maximum(tan(x), x, Interval(-pi/4, pi/4)) is S.One
    assert maximum(sin(x)*cos(x), x, S.Reals) == S.Half
    assert simplify(maximum(sin(x)*cos(x), x, Interval(pi*Rational(3, 8), pi*Rational(5, 8)))
        ) == sqrt(2)/4
    assert maximum((x+3)*(x-2), x) is oo
    assert maximum((x+3)*(x-2), x, Interval(-5, 0)) == S(14)
    assert maximum((x+3)/(x-2), x, Interval(-5, 0)) == Rational(2, 7)
    assert simplify(maximum(-x**4-x**3+x**2+10, x)
        ) == 41*sqrt(41)/512 + Rational(5419, 512)
    assert maximum(exp(x), x, Interval(-oo, 2)) == exp(2)
    assert maximum(log(x) - x, x, S.Reals) is S.NegativeOne
    assert maximum(cos(x), x, Union(Interval(0, 5), Interval(-6, -3))
        ) is S.One
    assert maximum(cos(x)-sin(x), x, S.Reals) == sqrt(2)
    assert maximum(y, x, S.Reals) == y
    assert maximum(abs(a**3 + a), a, Interval(0, 2)) == 10
    assert maximum(abs(60*a**3 + 24*a), a, Interval(0, 2)) == 528
    assert maximum(abs(12*a*(5*a**2 + 2)), a, Interval(0, 2)) == 528
    assert maximum(x/sqrt(x**2+1), x, S.Reals) == 1

    raises(ValueError, lambda : maximum(sin(x), x, S.EmptySet))
    raises(ValueError, lambda : maximum(log(cos(x)), x, S.EmptySet))
    raises(ValueError, lambda : maximum(1/(x**2 + y**2 + 1), x, S.EmptySet))
    raises(ValueError, lambda : maximum(sin(x), sin(x)))
    raises(ValueError, lambda : maximum(sin(x), x*y, S.EmptySet))
    raises(ValueError, lambda : maximum(sin(x), S.One))


def test_minimum():
    assert minimum(sin(x), x) is S.NegativeOne
    assert minimum(sin(x), x, Interval(1, 4)) == sin(4)
    assert minimum(tan(x), x) is -oo
    assert minimum(tan(x), x, Interval(-pi/4, pi/4)) is S.NegativeOne
    assert minimum(sin(x)*cos(x), x, S.Reals) == Rational(-1, 2)
    assert simplify(minimum(sin(x)*cos(x), x, Interval(pi*Rational(3, 8), pi*Rational(5, 8)))
        ) == -sqrt(2)/4
    assert minimum((x+3)*(x-2), x) == Rational(-25, 4)
    assert minimum((x+3)/(x-2), x, Interval(-5, 0)) == Rational(-3, 2)
    assert minimum(x**4-x**3+x**2+10, x) == S(10)
    assert minimum(exp(x), x, Interval(-2, oo)) == exp(-2)
    assert minimum(log(x) - x, x, S.Reals) is -oo
    assert minimum(cos(x), x, Union(Interval(0, 5), Interval(-6, -3))
        ) is S.NegativeOne
    assert minimum(cos(x)-sin(x), x, S.Reals) == -sqrt(2)
    assert minimum(y, x, S.Reals) == y
    assert minimum(x/sqrt(x**2+1), x, S.Reals) == -1

    raises(ValueError, lambda : minimum(sin(x), x, S.EmptySet))
    raises(ValueError, lambda : minimum(log(cos(x)), x, S.EmptySet))
    raises(ValueError, lambda : minimum(1/(x**2 + y**2 + 1), x, S.EmptySet))
    raises(ValueError, lambda : minimum(sin(x), sin(x)))
    raises(ValueError, lambda : minimum(sin(x), x*y, S.EmptySet))
    raises(ValueError, lambda : minimum(sin(x), S.One))


def test_issue_19869():
    assert (maximum(sqrt(3)*(x - 1)/(3*sqrt(x**2 + 1)), x)
        ) == sqrt(3)/3


def test_issue_16469():
    f = abs(a)
    assert function_range(f, a, S.Reals) == Interval(0, oo, False, True)


@_both_exp_pow
def test_issue_18747():
    assert periodicity(exp(pi*I*(x/4 + S.Half/2)), x) == 8


def test_issue_25942():
    assert (acos(x) > pi/3).as_set() == Interval.Ropen(-1, S(1)/2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\discrete\tests\test_convolutions.py ===
from sympy.core.numbers import (E, Rational, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, symbols, I
from sympy.discrete.convolutions import (
    convolution, convolution_fft, convolution_ntt, convolution_fwht,
    convolution_subset, covering_product, intersecting_product,
    convolution_int)
from sympy.testing.pytest import raises
from sympy.abc import x, y

def test_convolution():
    # fft
    a = [1, Rational(5, 3), sqrt(3), Rational(7, 5)]
    b = [9, 5, 5, 4, 3, 2]
    c = [3, 5, 3, 7, 8]
    d = [1422, 6572, 3213, 5552]
    e = [-1, Rational(5, 3), Rational(7, 5)]

    assert convolution(a, b) == convolution_fft(a, b)
    assert convolution(a, b, dps=9) == convolution_fft(a, b, dps=9)
    assert convolution(a, d, dps=7) == convolution_fft(d, a, dps=7)
    assert convolution(a, d[1:], dps=3) == convolution_fft(d[1:], a, dps=3)

    # prime moduli of the form (m*2**k + 1), sequence length
    # should be a divisor of 2**k
    p = 7*17*2**23 + 1
    q = 19*2**10 + 1

    # ntt
    assert convolution(d, b, prime=q) == convolution_ntt(b, d, prime=q)
    assert convolution(c, b, prime=p) == convolution_ntt(b, c, prime=p)
    assert convolution(d, c, prime=p) == convolution_ntt(c, d, prime=p)
    raises(TypeError, lambda: convolution(b, d, dps=5, prime=q))
    raises(TypeError, lambda: convolution(b, d, dps=6, prime=q))

    # fwht
    assert convolution(a, b, dyadic=True) == convolution_fwht(a, b)
    assert convolution(a, b, dyadic=False) == convolution(a, b)
    raises(TypeError, lambda: convolution(b, d, dps=2, dyadic=True))
    raises(TypeError, lambda: convolution(b, d, prime=p, dyadic=True))
    raises(TypeError, lambda: convolution(a, b, dps=2, dyadic=True))
    raises(TypeError, lambda: convolution(b, c, prime=p, dyadic=True))

    # subset
    assert convolution(a, b, subset=True) == convolution_subset(a, b) == \
            convolution(a, b, subset=True, dyadic=False) == \
                convolution(a, b, subset=True)
    assert convolution(a, b, subset=False) == convolution(a, b)
    raises(TypeError, lambda: convolution(a, b, subset=True, dyadic=True))
    raises(TypeError, lambda: convolution(c, d, subset=True, dps=6))
    raises(TypeError, lambda: convolution(a, c, subset=True, prime=q))

    # integer
    assert convolution([0], [0]) == convolution_int([0], [0])
    assert convolution(b, c) == convolution_int(b, c)

    # rational
    assert convolution([Rational(1,2)], [Rational(1,2)]) == [Rational(1, 4)]
    assert convolution(b, e) == [-9, 10, Rational(239, 15), Rational(34, 3),
                                 Rational(32, 3), Rational(43, 5), Rational(113, 15),
                                 Rational(14, 5)]


def test_cyclic_convolution():
    # fft
    a = [1, Rational(5, 3), sqrt(3), Rational(7, 5)]
    b = [9, 5, 5, 4, 3, 2]

    assert convolution([1, 2, 3], [4, 5, 6], cycle=0) == \
            convolution([1, 2, 3], [4, 5, 6], cycle=5) == \
                convolution([1, 2, 3], [4, 5, 6])

    assert convolution([1, 2, 3], [4, 5, 6], cycle=3) == [31, 31, 28]

    a = [Rational(1, 3), Rational(7, 3), Rational(5, 9), Rational(2, 7), Rational(5, 8)]
    b = [Rational(3, 5), Rational(4, 7), Rational(7, 8), Rational(8, 9)]

    assert convolution(a, b, cycle=0) == \
            convolution(a, b, cycle=len(a) + len(b) - 1)

    assert convolution(a, b, cycle=4) == [Rational(87277, 26460), Rational(30521, 11340),
                            Rational(11125, 4032), Rational(3653, 1080)]

    assert convolution(a, b, cycle=6) == [Rational(20177, 20160), Rational(676, 315), Rational(47, 24),
                            Rational(3053, 1080), Rational(16397, 5292), Rational(2497, 2268)]

    assert convolution(a, b, cycle=9) == \
                convolution(a, b, cycle=0) + [S.Zero]

    # ntt
    a = [2313, 5323532, S(3232), 42142, 42242421]
    b = [S(33456), 56757, 45754, 432423]

    assert convolution(a, b, prime=19*2**10 + 1, cycle=0) == \
            convolution(a, b, prime=19*2**10 + 1, cycle=8) == \
                convolution(a, b, prime=19*2**10 + 1)

    assert convolution(a, b, prime=19*2**10 + 1, cycle=5) == [96, 17146, 2664,
                                                                    15534, 3517]

    assert convolution(a, b, prime=19*2**10 + 1, cycle=7) == [4643, 3458, 1260,
                                                        15534, 3517, 16314, 13688]

    assert convolution(a, b, prime=19*2**10 + 1, cycle=9) == \
            convolution(a, b, prime=19*2**10 + 1) + [0]

    # fwht
    u, v, w, x, y = symbols('u v w x y')
    p, q, r, s, t = symbols('p q r s t')
    c = [u, v, w, x, y]
    d = [p, q, r, s, t]

    assert convolution(a, b, dyadic=True, cycle=3) == \
                        [2499522285783, 19861417974796, 4702176579021]

    assert convolution(a, b, dyadic=True, cycle=5) == [2718149225143,
            2114320852171, 20571217906407, 246166418903, 1413262436976]

    assert convolution(c, d, dyadic=True, cycle=4) == \
            [p*u + p*y + q*v + r*w + s*x + t*u + t*y,
             p*v + q*u + q*y + r*x + s*w + t*v,
             p*w + q*x + r*u + r*y + s*v + t*w,
             p*x + q*w + r*v + s*u + s*y + t*x]

    assert convolution(c, d, dyadic=True, cycle=6) == \
            [p*u + q*v + r*w + r*y + s*x + t*w + t*y,
             p*v + q*u + r*x + s*w + s*y + t*x,
             p*w + q*x + r*u + s*v,
             p*x + q*w + r*v + s*u,
             p*y + t*u,
             q*y + t*v]

    # subset
    assert convolution(a, b, subset=True, cycle=7) == [18266671799811,
                    178235365533, 213958794, 246166418903, 1413262436976,
                    2397553088697, 1932759730434]

    assert convolution(a[1:], b, subset=True, cycle=4) == \
            [178104086592, 302255835516, 244982785880, 3717819845434]

    assert convolution(a, b[:-1], subset=True, cycle=6) == [1932837114162,
            178235365533, 213958794, 245166224504, 1413262436976, 2397553088697]

    assert convolution(c, d, subset=True, cycle=3) == \
            [p*u + p*x + q*w + r*v + r*y + s*u + t*w,
             p*v + p*y + q*u + s*y + t*u + t*x,
             p*w + q*y + r*u + t*v]

    assert convolution(c, d, subset=True, cycle=5) == \
            [p*u + q*y + t*v,
             p*v + q*u + r*y + t*w,
             p*w + r*u + s*y + t*x,
             p*x + q*w + r*v + s*u,
             p*y + t*u]

    raises(ValueError, lambda: convolution([1, 2, 3], [4, 5, 6], cycle=-1))


def test_convolution_fft():
    assert all(convolution_fft([], x, dps=y) == [] for x in ([], [1]) for y in (None, 3))
    assert convolution_fft([1, 2, 3], [4, 5, 6]) == [4, 13, 28, 27, 18]
    assert convolution_fft([1], [5, 6, 7]) == [5, 6, 7]
    assert convolution_fft([1, 3], [5, 6, 7]) == [5, 21, 25, 21]

    assert convolution_fft([1 + 2*I], [2 + 3*I]) == [-4 + 7*I]
    assert convolution_fft([1 + 2*I, 3 + 4*I, 5 + 3*I/5], [Rational(2, 5) + 4*I/7]) == \
            [Rational(-26, 35) + I*48/35, Rational(-38, 35) + I*116/35, Rational(58, 35) + I*542/175]

    assert convolution_fft([Rational(3, 4), Rational(5, 6)], [Rational(7, 8), Rational(1, 3), Rational(2, 5)]) == \
                                    [Rational(21, 32), Rational(47, 48), Rational(26, 45), Rational(1, 3)]

    assert convolution_fft([Rational(1, 9), Rational(2, 3), Rational(3, 5)], [Rational(2, 5), Rational(3, 7), Rational(4, 9)]) == \
                                [Rational(2, 45), Rational(11, 35), Rational(8152, 14175), Rational(523, 945), Rational(4, 15)]

    assert convolution_fft([pi, E, sqrt(2)], [sqrt(3), 1/pi, 1/E]) == \
                    [sqrt(3)*pi, 1 + sqrt(3)*E, E/pi + pi*exp(-1) + sqrt(6),
                                            sqrt(2)/pi + 1, sqrt(2)*exp(-1)]

    assert convolution_fft([2321, 33123], [5321, 6321, 71323]) == \
                        [12350041, 190918524, 374911166, 2362431729]

    assert convolution_fft([312313, 31278232], [32139631, 319631]) == \
                        [10037624576503, 1005370659728895, 9997492572392]

    raises(TypeError, lambda: convolution_fft(x, y))
    raises(ValueError, lambda: convolution_fft([x, y], [y, x]))


def test_convolution_ntt():
    # prime moduli of the form (m*2**k + 1), sequence length
    # should be a divisor of 2**k
    p = 7*17*2**23 + 1
    q = 19*2**10 + 1
    r = 2*500000003 + 1 # only for sequences of length 1 or 2
    # s = 2*3*5*7 # composite modulus

    assert all(convolution_ntt([], x, prime=y) == [] for x in ([], [1]) for y in (p, q, r))
    assert convolution_ntt([2], [3], r) == [6]
    assert convolution_ntt([2, 3], [4], r) == [8, 12]

    assert convolution_ntt([32121, 42144, 4214, 4241], [32132, 3232, 87242], p) == [33867619,
                                    459741727, 79180879, 831885249, 381344700, 369993322]
    assert convolution_ntt([121913, 3171831, 31888131, 12], [17882, 21292, 29921, 312], q) == \
                                                [8158, 3065, 3682, 7090, 1239, 2232, 3744]

    assert convolution_ntt([12, 19, 21, 98, 67], [2, 6, 7, 8, 9], p) == \
                    convolution_ntt([12, 19, 21, 98, 67], [2, 6, 7, 8, 9], q)
    assert convolution_ntt([12, 19, 21, 98, 67], [21, 76, 17, 78, 69], p) == \
                    convolution_ntt([12, 19, 21, 98, 67], [21, 76, 17, 78, 69], q)

    raises(ValueError, lambda: convolution_ntt([2, 3], [4, 5], r))
    raises(ValueError, lambda: convolution_ntt([x, y], [y, x], q))
    raises(TypeError, lambda: convolution_ntt(x, y, p))


def test_convolution_fwht():
    assert convolution_fwht([], []) == []
    assert convolution_fwht([], [1]) == []
    assert convolution_fwht([1, 2, 3], [4, 5, 6]) == [32, 13, 18, 27]

    assert convolution_fwht([Rational(5, 7), Rational(6, 8), Rational(7, 3)], [2, 4, Rational(6, 7)]) == \
                                    [Rational(45, 7), Rational(61, 14), Rational(776, 147), Rational(419, 42)]

    a = [1, Rational(5, 3), sqrt(3), Rational(7, 5), 4 + 5*I]
    b = [94, 51, 53, 45, 31, 27, 13]
    c = [3 + 4*I, 5 + 7*I, 3, Rational(7, 6), 8]

    assert convolution_fwht(a, b) == [53*sqrt(3) + 366 + 155*I,
                                    45*sqrt(3) + Rational(5848, 15) + 135*I,
                                    94*sqrt(3) + Rational(1257, 5) + 65*I,
                                    51*sqrt(3) + Rational(3974, 15),
                                    13*sqrt(3) + 452 + 470*I,
                                    Rational(4513, 15) + 255*I,
                                    31*sqrt(3) + Rational(1314, 5) + 265*I,
                                    27*sqrt(3) + Rational(3676, 15) + 225*I]

    assert convolution_fwht(b, c) == [Rational(1993, 2) + 733*I, Rational(6215, 6) + 862*I,
        Rational(1659, 2) + 527*I, Rational(1988, 3) + 551*I, 1019 + 313*I, Rational(3955, 6) + 325*I,
        Rational(1175, 2) + 52*I, Rational(3253, 6) + 91*I]

    assert convolution_fwht(a[3:], c) == [Rational(-54, 5) + I*293/5, -1 + I*204/5,
            Rational(133, 15) + I*35/6, Rational(409, 30) + 15*I, Rational(56, 5), 32 + 40*I, 0, 0]

    u, v, w, x, y, z = symbols('u v w x y z')

    assert convolution_fwht([u, v], [x, y]) == [u*x + v*y, u*y + v*x]

    assert convolution_fwht([u, v, w], [x, y]) == \
        [u*x + v*y, u*y + v*x, w*x, w*y]

    assert convolution_fwht([u, v, w], [x, y, z]) == \
        [u*x + v*y + w*z, u*y + v*x, u*z + w*x, v*z + w*y]

    raises(TypeError, lambda: convolution_fwht(x, y))
    raises(TypeError, lambda: convolution_fwht(x*y, u + v))


def test_convolution_subset():
    assert convolution_subset([], []) == []
    assert convolution_subset([], [Rational(1, 3)]) == []
    assert convolution_subset([6 + I*3/7], [Rational(2, 3)]) == [4 + I*2/7]

    a = [1, Rational(5, 3), sqrt(3), 4 + 5*I]
    b = [64, 71, 55, 47, 33, 29, 15]
    c = [3 + I*2/3, 5 + 7*I, 7, Rational(7, 5), 9]

    assert convolution_subset(a, b) == [64, Rational(533, 3), 55 + 64*sqrt(3),
                                        71*sqrt(3) + Rational(1184, 3) + 320*I, 33, 84,
                                        15 + 33*sqrt(3), 29*sqrt(3) + 157 + 165*I]

    assert convolution_subset(b, c) == [192 + I*128/3, 533 + I*1486/3,
                                        613 + I*110/3, Rational(5013, 5) + I*1249/3,
                                        675 + 22*I, 891 + I*751/3,
                                        771 + 10*I, Rational(3736, 5) + 105*I]

    assert convolution_subset(a, c) == convolution_subset(c, a)
    assert convolution_subset(a[:2], b) == \
            [64, Rational(533, 3), 55, Rational(416, 3), 33, 84, 15, 25]

    assert convolution_subset(a[:2], c) == \
            [3 + I*2/3, 10 + I*73/9, 7, Rational(196, 15), 9, 15, 0, 0]

    u, v, w, x, y, z = symbols('u v w x y z')

    assert convolution_subset([u, v, w], [x, y]) == [u*x, u*y + v*x, w*x, w*y]
    assert convolution_subset([u, v, w, x], [y, z]) == \
                            [u*y, u*z + v*y, w*y, w*z + x*y]

    assert convolution_subset([u, v], [x, y, z]) == \
                    convolution_subset([x, y, z], [u, v])

    raises(TypeError, lambda: convolution_subset(x, z))
    raises(TypeError, lambda: convolution_subset(Rational(7, 3), u))


def test_covering_product():
    assert covering_product([], []) == []
    assert covering_product([], [Rational(1, 3)]) == []
    assert covering_product([6 + I*3/7], [Rational(2, 3)]) == [4 + I*2/7]

    a = [1, Rational(5, 8), sqrt(7), 4 + 9*I]
    b = [66, 81, 95, 49, 37, 89, 17]
    c = [3 + I*2/3, 51 + 72*I, 7, Rational(7, 15), 91]

    assert covering_product(a, b) == [66, Rational(1383, 8), 95 + 161*sqrt(7),
                                        130*sqrt(7) + 1303 + 2619*I, 37,
                                        Rational(671, 4), 17 + 54*sqrt(7),
                                        89*sqrt(7) + Rational(4661, 8) + 1287*I]

    assert covering_product(b, c) == [198 + 44*I, 7740 + 10638*I,
                                        1412 + I*190/3, Rational(42684, 5) + I*31202/3,
                                        9484 + I*74/3, 22163 + I*27394/3,
                                        10621 + I*34/3, Rational(90236, 15) + 1224*I]

    assert covering_product(a, c) == covering_product(c, a)
    assert covering_product(b, c[:-1]) == [198 + 44*I, 7740 + 10638*I,
                                         1412 + I*190/3, Rational(42684, 5) + I*31202/3,
                                         111 + I*74/3, 6693 + I*27394/3,
                                         429 + I*34/3, Rational(23351, 15) + 1224*I]

    assert covering_product(a, c[:-1]) == [3 + I*2/3,
                            Rational(339, 4) + I*1409/12, 7 + 10*sqrt(7) + 2*sqrt(7)*I/3,
                            -403 + 772*sqrt(7)/15 + 72*sqrt(7)*I + I*12658/15]

    u, v, w, x, y, z = symbols('u v w x y z')

    assert covering_product([u, v, w], [x, y]) == \
                            [u*x, u*y + v*x + v*y, w*x, w*y]

    assert covering_product([u, v, w, x], [y, z]) == \
                            [u*y, u*z + v*y + v*z, w*y, w*z + x*y + x*z]

    assert covering_product([u, v], [x, y, z]) == \
                    covering_product([x, y, z], [u, v])

    raises(TypeError, lambda: covering_product(x, z))
    raises(TypeError, lambda: covering_product(Rational(7, 3), u))


def test_intersecting_product():
    assert intersecting_product([], []) == []
    assert intersecting_product([], [Rational(1, 3)]) == []
    assert intersecting_product([6 + I*3/7], [Rational(2, 3)]) == [4 + I*2/7]

    a = [1, sqrt(5), Rational(3, 8) + 5*I, 4 + 7*I]
    b = [67, 51, 65, 48, 36, 79, 27]
    c = [3 + I*2/5, 5 + 9*I, 7, Rational(7, 19), 13]

    assert intersecting_product(a, b) == [195*sqrt(5) + Rational(6979, 8) + 1886*I,
                                178*sqrt(5) + 520 + 910*I, Rational(841, 2) + 1344*I,
                                192 + 336*I, 0, 0, 0, 0]

    assert intersecting_product(b, c) == [Rational(128553, 19) + I*9521/5,
                Rational(17820, 19) + 1602*I, Rational(19264, 19), Rational(336, 19), 1846, 0, 0, 0]

    assert intersecting_product(a, c) == intersecting_product(c, a)
    assert intersecting_product(b[1:], c[:-1]) == [Rational(64788, 19) + I*8622/5,
                    Rational(12804, 19) + 1152*I, Rational(11508, 19), Rational(252, 19), 0, 0, 0, 0]

    assert intersecting_product(a, c[:-2]) == \
                    [Rational(-99, 5) + 10*sqrt(5) + 2*sqrt(5)*I/5 + I*3021/40,
                    -43 + 5*sqrt(5) + 9*sqrt(5)*I + 71*I, Rational(245, 8) + 84*I, 0]

    u, v, w, x, y, z = symbols('u v w x y z')

    assert intersecting_product([u, v, w], [x, y]) == \
                            [u*x + u*y + v*x + w*x + w*y, v*y, 0, 0]

    assert intersecting_product([u, v, w, x], [y, z]) == \
                        [u*y + u*z + v*y + w*y + w*z + x*y, v*z + x*z, 0, 0]

    assert intersecting_product([u, v], [x, y, z]) == \
                    intersecting_product([x, y, z], [u, v])

    raises(TypeError, lambda: intersecting_product(x, z))
    raises(TypeError, lambda: intersecting_product(u, Rational(8, 3)))


def test_convolution_int():
    assert convolution_int([1], [1]) == [1]
    assert convolution_int([1, 1], [0]) == [0]
    assert convolution_int([1, 2, 3], [4, 5, 6]) == [4, 13, 28, 27, 18]
    assert convolution_int([1], [5, 6, 7]) == [5, 6, 7]
    assert convolution_int([1, 3], [5, 6, 7]) == [5, 21, 25, 21]
    assert convolution_int([10, -5, 1, 3], [-5, 6, 7]) == [-50, 85, 35, -44, 25, 21]
    assert convolution_int([0, 1, 0, -1], [1, 0, -1, 0]) == [0, 1, 0, -2, 0, 1]
    assert convolution_int(
        [-341, -5, 1, 3, -71, -99, 43, 87],
        [5, 6, 7, 12, 345, 21, -78, -7, -89]
    ) == [-1705, -2071, -2412, -4106, -118035, -9774, 25998, 2981, 5509,
          -34317, 19228, 38870, 5485, 1724, -4436, -7743]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\libs\test_join.py ===
import numpy as np
import pytest

from pandas._libs import join as libjoin
from pandas._libs.join import (
    inner_join,
    left_outer_join,
)

import pandas._testing as tm


class TestIndexer:
    @pytest.mark.parametrize(
        "dtype", ["int32", "int64", "float32", "float64", "object"]
    )
    def test_outer_join_indexer(self, dtype):
        indexer = libjoin.outer_join_indexer

        left = np.arange(3, dtype=dtype)
        right = np.arange(2, 5, dtype=dtype)
        empty = np.array([], dtype=dtype)

        result, lindexer, rindexer = indexer(left, right)
        assert isinstance(result, np.ndarray)
        assert isinstance(lindexer, np.ndarray)
        assert isinstance(rindexer, np.ndarray)
        tm.assert_numpy_array_equal(result, np.arange(5, dtype=dtype))
        exp = np.array([0, 1, 2, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([-1, -1, 0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

        result, lindexer, rindexer = indexer(empty, right)
        tm.assert_numpy_array_equal(result, right)
        exp = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

        result, lindexer, rindexer = indexer(left, empty)
        tm.assert_numpy_array_equal(result, left)
        exp = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(lindexer, exp)
        exp = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(rindexer, exp)

    def test_cython_left_outer_join(self):
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1], dtype=np.intp)
        max_group = 5

        ls, rs = left_outer_join(left, right, max_group)

        exp_ls = left.argsort(kind="mergesort")
        exp_rs = right.argsort(kind="mergesort")

        exp_li = np.array([0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10])
        exp_ri = np.array(
            [0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 4, 5, -1, -1]
        )

        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1

        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1

        tm.assert_numpy_array_equal(ls, exp_ls, check_dtype=False)
        tm.assert_numpy_array_equal(rs, exp_rs, check_dtype=False)

    def test_cython_right_outer_join(self):
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1], dtype=np.intp)
        max_group = 5

        rs, ls = left_outer_join(right, left, max_group)

        exp_ls = left.argsort(kind="mergesort")
        exp_rs = right.argsort(kind="mergesort")

        #            0        1        1        1
        exp_li = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                3,
                4,
                5,
                3,
                4,
                5,
                #            2        2        4
                6,
                7,
                8,
                6,
                7,
                8,
                -1,
            ]
        )
        exp_ri = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6])

        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1

        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1

        tm.assert_numpy_array_equal(ls, exp_ls)
        tm.assert_numpy_array_equal(rs, exp_rs)

    def test_cython_inner_join(self):
        left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
        right = np.array([1, 1, 0, 4, 2, 2, 1, 4], dtype=np.intp)
        max_group = 5

        ls, rs = inner_join(left, right, max_group)

        exp_ls = left.argsort(kind="mergesort")
        exp_rs = right.argsort(kind="mergesort")

        exp_li = np.array([0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8])
        exp_ri = np.array([0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 4, 5])

        exp_ls = exp_ls.take(exp_li)
        exp_ls[exp_li == -1] = -1

        exp_rs = exp_rs.take(exp_ri)
        exp_rs[exp_ri == -1] = -1

        tm.assert_numpy_array_equal(ls, exp_ls)
        tm.assert_numpy_array_equal(rs, exp_rs)


@pytest.mark.parametrize("readonly", [True, False])
def test_left_join_indexer_unique(readonly):
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([2, 2, 3, 4, 4], dtype=np.int64)
    if readonly:
        # GH#37312, GH#37264
        a.setflags(write=False)
        b.setflags(write=False)

    result = libjoin.left_join_indexer_unique(b, a)
    expected = np.array([1, 1, 2, 3, 3], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)


def test_left_outer_join_bug():
    left = np.array(
        [
            0,
            1,
            0,
            1,
            1,
            2,
            3,
            1,
            0,
            2,
            1,
            2,
            0,
            1,
            1,
            2,
            3,
            2,
            3,
            2,
            1,
            1,
            3,
            0,
            3,
            2,
            3,
            0,
            0,
            2,
            3,
            2,
            0,
            3,
            1,
            3,
            0,
            1,
            3,
            0,
            0,
            1,
            0,
            3,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            2,
            2,
            2,
            2,
            2,
            0,
            3,
            1,
            2,
            0,
            0,
            3,
            1,
            3,
            2,
            2,
            0,
            1,
            3,
            0,
            2,
            3,
            2,
            3,
            3,
            2,
            3,
            3,
            1,
            3,
            2,
            0,
            0,
            3,
            1,
            1,
            1,
            0,
            2,
            3,
            3,
            1,
            2,
            0,
            3,
            1,
            2,
            0,
            2,
        ],
        dtype=np.intp,
    )

    right = np.array([3, 1], dtype=np.intp)
    max_groups = 4

    lidx, ridx = libjoin.left_outer_join(left, right, max_groups, sort=False)

    exp_lidx = np.arange(len(left), dtype=np.intp)
    exp_ridx = -np.ones(len(left), dtype=np.intp)

    exp_ridx[left == 1] = 1
    exp_ridx[left == 3] = 0

    tm.assert_numpy_array_equal(lidx, exp_lidx)
    tm.assert_numpy_array_equal(ridx, exp_ridx)


def test_inner_join_indexer():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)

    index, ares, bres = libjoin.inner_join_indexer(a, b)

    index_exp = np.array([3, 5], dtype=np.int64)
    tm.assert_almost_equal(index, index_exp)

    aexp = np.array([2, 4], dtype=np.intp)
    bexp = np.array([1, 2], dtype=np.intp)
    tm.assert_almost_equal(ares, aexp)
    tm.assert_almost_equal(bres, bexp)

    a = np.array([5], dtype=np.int64)
    b = np.array([5], dtype=np.int64)

    index, ares, bres = libjoin.inner_join_indexer(a, b)
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))


def test_outer_join_indexer():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)

    index, ares, bres = libjoin.outer_join_indexer(a, b)

    index_exp = np.array([0, 1, 2, 3, 4, 5, 7, 9], dtype=np.int64)
    tm.assert_almost_equal(index, index_exp)

    aexp = np.array([-1, 0, 1, 2, 3, 4, -1, -1], dtype=np.intp)
    bexp = np.array([0, -1, -1, 1, -1, 2, 3, 4], dtype=np.intp)
    tm.assert_almost_equal(ares, aexp)
    tm.assert_almost_equal(bres, bexp)

    a = np.array([5], dtype=np.int64)
    b = np.array([5], dtype=np.int64)

    index, ares, bres = libjoin.outer_join_indexer(a, b)
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))


def test_left_join_indexer():
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([0, 3, 5, 7, 9], dtype=np.int64)

    index, ares, bres = libjoin.left_join_indexer(a, b)

    tm.assert_almost_equal(index, a)

    aexp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
    bexp = np.array([-1, -1, 1, -1, 2], dtype=np.intp)
    tm.assert_almost_equal(ares, aexp)
    tm.assert_almost_equal(bres, bexp)

    a = np.array([5], dtype=np.int64)
    b = np.array([5], dtype=np.int64)

    index, ares, bres = libjoin.left_join_indexer(a, b)
    tm.assert_numpy_array_equal(index, np.array([5], dtype=np.int64))
    tm.assert_numpy_array_equal(ares, np.array([0], dtype=np.intp))
    tm.assert_numpy_array_equal(bres, np.array([0], dtype=np.intp))


def test_left_join_indexer2():
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)

    res, lidx, ridx = libjoin.left_join_indexer(idx2, idx)

    exp_res = np.array([1, 1, 2, 5, 7, 9], dtype=np.int64)
    tm.assert_almost_equal(res, exp_res)

    exp_lidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
    tm.assert_almost_equal(lidx, exp_lidx)

    exp_ridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
    tm.assert_almost_equal(ridx, exp_ridx)


def test_outer_join_indexer2():
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)

    res, lidx, ridx = libjoin.outer_join_indexer(idx2, idx)

    exp_res = np.array([1, 1, 2, 5, 7, 9], dtype=np.int64)
    tm.assert_almost_equal(res, exp_res)

    exp_lidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
    tm.assert_almost_equal(lidx, exp_lidx)

    exp_ridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
    tm.assert_almost_equal(ridx, exp_ridx)


def test_inner_join_indexer2():
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)

    res, lidx, ridx = libjoin.inner_join_indexer(idx2, idx)

    exp_res = np.array([1, 1, 2, 5], dtype=np.int64)
    tm.assert_almost_equal(res, exp_res)

    exp_lidx = np.array([0, 0, 1, 2], dtype=np.intp)
    tm.assert_almost_equal(lidx, exp_lidx)

    exp_ridx = np.array([0, 1, 2, 3], dtype=np.intp)
    tm.assert_almost_equal(ridx, exp_ridx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\resample\test_time_grouper.py ===
from datetime import datetime
from operator import methodcaller

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range


@pytest.fixture
def test_series():
    return Series(
        np.random.default_rng(2).standard_normal(1000),
        index=date_range("1/1/2000", periods=1000),
    )


def test_apply(test_series):
    grouper = Grouper(freq="YE", label="right", closed="right")

    grouped = test_series.groupby(grouper)

    def f(x):
        return x.sort_values()[-3:]

    applied = grouped.apply(f)
    expected = test_series.groupby(lambda x: x.year).apply(f)

    applied.index = applied.index.droplevel(0)
    expected.index = expected.index.droplevel(0)
    tm.assert_series_equal(applied, expected)


def test_count(test_series):
    test_series[::3] = np.nan

    expected = test_series.groupby(lambda x: x.year).count()

    grouper = Grouper(freq="YE", label="right", closed="right")
    result = test_series.groupby(grouper).count()
    expected.index = result.index
    tm.assert_series_equal(result, expected)

    result = test_series.resample("YE").count()
    expected.index = result.index
    tm.assert_series_equal(result, expected)


def test_numpy_reduction(test_series):
    result = test_series.resample("YE", closed="right").prod()

    msg = "using SeriesGroupBy.prod"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = test_series.groupby(lambda x: x.year).agg(np.prod)
    expected.index = result.index

    tm.assert_series_equal(result, expected)


def test_apply_iteration():
    # #2300
    N = 1000
    ind = date_range(start="2000-01-01", freq="D", periods=N)
    df = DataFrame({"open": 1, "close": 2}, index=ind)
    tg = Grouper(freq="ME")

    grouper, _ = tg._get_grouper(df)

    # Errors
    grouped = df.groupby(grouper, group_keys=False)

    def f(df):
        return df["close"] / df["open"]

    # it works!
    result = grouped.apply(f)
    tm.assert_index_equal(result.index, df.index)


@pytest.mark.parametrize(
    "index",
    [
        Index([1, 2]),
        Index(["a", "b"]),
        Index([1.1, 2.2]),
        pd.MultiIndex.from_arrays([[1, 2], ["a", "b"]]),
    ],
)
def test_fails_on_no_datetime_index(index):
    name = type(index).__name__
    df = DataFrame({"a": range(len(index))}, index=index)

    msg = (
        "Only valid with DatetimeIndex, TimedeltaIndex "
        f"or PeriodIndex, but got an instance of '{name}'"
    )
    with pytest.raises(TypeError, match=msg):
        df.groupby(Grouper(freq="D"))


def test_aaa_group_order():
    # GH 12840
    # check TimeGrouper perform stable sorts
    n = 20
    data = np.random.default_rng(2).standard_normal((n, 4))
    df = DataFrame(data, columns=["A", "B", "C", "D"])
    df["key"] = [
        datetime(2013, 1, 1),
        datetime(2013, 1, 2),
        datetime(2013, 1, 3),
        datetime(2013, 1, 4),
        datetime(2013, 1, 5),
    ] * 4
    grouped = df.groupby(Grouper(key="key", freq="D"))

    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 1)), df[::5])
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 2)), df[1::5])
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 3)), df[2::5])
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 4)), df[3::5])
    tm.assert_frame_equal(grouped.get_group(datetime(2013, 1, 5)), df[4::5])


def test_aggregate_normal(resample_method):
    """Check TimeGrouper's aggregation is identical as normal groupby."""

    data = np.random.default_rng(2).standard_normal((20, 4))
    normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
    normal_df["key"] = [1, 2, 3, 4, 5] * 4

    dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
    dt_df["key"] = Index(
        [
            datetime(2013, 1, 1),
            datetime(2013, 1, 2),
            datetime(2013, 1, 3),
            datetime(2013, 1, 4),
            datetime(2013, 1, 5),
        ]
        * 4,
        dtype="M8[ns]",
    )

    normal_grouped = normal_df.groupby("key")
    dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))

    expected = getattr(normal_grouped, resample_method)()
    dt_result = getattr(dt_grouped, resample_method)()
    expected.index = date_range(start="2013-01-01", freq="D", periods=5, name="key")
    tm.assert_equal(expected, dt_result)


@pytest.mark.xfail(reason="if TimeGrouper is used included, 'nth' doesn't work yet")
def test_aggregate_nth():
    """Check TimeGrouper's aggregation is identical as normal groupby."""

    data = np.random.default_rng(2).standard_normal((20, 4))
    normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
    normal_df["key"] = [1, 2, 3, 4, 5] * 4

    dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
    dt_df["key"] = [
        datetime(2013, 1, 1),
        datetime(2013, 1, 2),
        datetime(2013, 1, 3),
        datetime(2013, 1, 4),
        datetime(2013, 1, 5),
    ] * 4

    normal_grouped = normal_df.groupby("key")
    dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))

    expected = normal_grouped.nth(3)
    expected.index = date_range(start="2013-01-01", freq="D", periods=5, name="key")
    dt_result = dt_grouped.nth(3)
    tm.assert_frame_equal(expected, dt_result)


@pytest.mark.parametrize(
    "method, method_args, unit",
    [
        ("sum", {}, 0),
        ("sum", {"min_count": 0}, 0),
        ("sum", {"min_count": 1}, np.nan),
        ("prod", {}, 1),
        ("prod", {"min_count": 0}, 1),
        ("prod", {"min_count": 1}, np.nan),
    ],
)
def test_resample_entirely_nat_window(method, method_args, unit):
    ser = Series([0] * 2 + [np.nan] * 2, index=date_range("2017", periods=4))
    result = methodcaller(method, **method_args)(ser.resample("2d"))

    exp_dti = pd.DatetimeIndex(["2017-01-01", "2017-01-03"], dtype="M8[ns]", freq="2D")
    expected = Series([0.0, unit], index=exp_dti)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, fill_value",
    [("min", np.nan), ("max", np.nan), ("sum", 0), ("prod", 1), ("count", 0)],
)
def test_aggregate_with_nat(func, fill_value):
    # check TimeGrouper's aggregation is identical as normal groupby
    # if NaT is included, 'var', 'std', 'mean', 'first','last'
    # and 'nth' doesn't work yet

    n = 20
    data = np.random.default_rng(2).standard_normal((n, 4)).astype("int64")
    normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
    normal_df["key"] = [1, 2, np.nan, 4, 5] * 4

    dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
    dt_df["key"] = Index(
        [
            datetime(2013, 1, 1),
            datetime(2013, 1, 2),
            pd.NaT,
            datetime(2013, 1, 4),
            datetime(2013, 1, 5),
        ]
        * 4,
        dtype="M8[ns]",
    )

    normal_grouped = normal_df.groupby("key")
    dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))

    normal_result = getattr(normal_grouped, func)()
    dt_result = getattr(dt_grouped, func)()

    pad = DataFrame([[fill_value] * 4], index=[3], columns=["A", "B", "C", "D"])
    expected = pd.concat([normal_result, pad])
    expected = expected.sort_index()
    dti = date_range(
        start="2013-01-01",
        freq="D",
        periods=5,
        name="key",
        unit=dt_df["key"]._values.unit,
    )
    expected.index = dti._with_freq(None)  # TODO: is this desired?
    tm.assert_frame_equal(expected, dt_result)
    assert dt_result.index.name == "key"


def test_aggregate_with_nat_size():
    # GH 9925
    n = 20
    data = np.random.default_rng(2).standard_normal((n, 4)).astype("int64")
    normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
    normal_df["key"] = [1, 2, np.nan, 4, 5] * 4

    dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
    dt_df["key"] = Index(
        [
            datetime(2013, 1, 1),
            datetime(2013, 1, 2),
            pd.NaT,
            datetime(2013, 1, 4),
            datetime(2013, 1, 5),
        ]
        * 4,
        dtype="M8[ns]",
    )

    normal_grouped = normal_df.groupby("key")
    dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))

    normal_result = normal_grouped.size()
    dt_result = dt_grouped.size()

    pad = Series([0], index=[3])
    expected = pd.concat([normal_result, pad])
    expected = expected.sort_index()
    expected.index = date_range(
        start="2013-01-01",
        freq="D",
        periods=5,
        name="key",
        unit=dt_df["key"]._values.unit,
    )._with_freq(None)
    tm.assert_series_equal(expected, dt_result)
    assert dt_result.index.name == "key"


def test_repr():
    # GH18203
    result = repr(Grouper(key="A", freq="h"))
    expected = (
        "TimeGrouper(key='A', freq=<Hour>, axis=0, sort=True, dropna=True, "
        "closed='left', label='left', how='mean', "
        "convention='e', origin='start_day')"
    )
    assert result == expected

    result = repr(Grouper(key="A", freq="h", origin="2000-01-01"))
    expected = (
        "TimeGrouper(key='A', freq=<Hour>, axis=0, sort=True, dropna=True, "
        "closed='left', label='left', how='mean', "
        "convention='e', origin=Timestamp('2000-01-01 00:00:00'))"
    )
    assert result == expected


@pytest.mark.parametrize(
    "method, method_args, expected_values",
    [
        ("sum", {}, [1, 0, 1]),
        ("sum", {"min_count": 0}, [1, 0, 1]),
        ("sum", {"min_count": 1}, [1, np.nan, 1]),
        ("sum", {"min_count": 2}, [np.nan, np.nan, np.nan]),
        ("prod", {}, [1, 1, 1]),
        ("prod", {"min_count": 0}, [1, 1, 1]),
        ("prod", {"min_count": 1}, [1, np.nan, 1]),
        ("prod", {"min_count": 2}, [np.nan, np.nan, np.nan]),
    ],
)
def test_upsample_sum(method, method_args, expected_values):
    ser = Series(1, index=date_range("2017", periods=2, freq="h"))
    resampled = ser.resample("30min")
    index = pd.DatetimeIndex(
        ["2017-01-01T00:00:00", "2017-01-01T00:30:00", "2017-01-01T01:00:00"],
        dtype="M8[ns]",
        freq="30min",
    )
    result = methodcaller(method, **method_args)(resampled)
    expected = Series(expected_values, index=index)
    tm.assert_series_equal(result, expected)


def test_groupby_resample_interpolate():
    # GH 35325
    d = {"price": [10, 11, 9], "volume": [50, 60, 50]}

    df = DataFrame(d)

    df["week_starting"] = date_range("01/01/2018", periods=3, freq="W")

    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = (
            df.set_index("week_starting")
            .groupby("volume")
            .resample("1D")
            .interpolate(method="linear")
        )

    volume = [50] * 15 + [60]
    week_starting = list(date_range("2018-01-07", "2018-01-21")) + [
        Timestamp("2018-01-14")
    ]
    expected_ind = pd.MultiIndex.from_arrays(
        [volume, week_starting],
        names=["volume", "week_starting"],
    )

    expected = DataFrame(
        data={
            "price": [
                10.0,
                9.928571428571429,
                9.857142857142858,
                9.785714285714286,
                9.714285714285714,
                9.642857142857142,
                9.571428571428571,
                9.5,
                9.428571428571429,
                9.357142857142858,
                9.285714285714286,
                9.214285714285714,
                9.142857142857142,
                9.071428571428571,
                9.0,
                11.0,
            ],
            "volume": [50.0] * 15 + [60],
        },
        index=expected_ind,
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_julia.py ===
from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
                        Tuple, Symbol, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos, sinc
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
                            HadamardProduct, SparseMatrix)
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
                                            besselk, hankel1, hankel2, airyai,
                                            airybi, airyaiprime, airybiprime)
from sympy.testing.pytest import XFAIL

from sympy.printing.julia import julia_code

x, y, z = symbols('x,y,z')


def test_Integer():
    assert julia_code(Integer(67)) == "67"
    assert julia_code(Integer(-1)) == "-1"


def test_Rational():
    assert julia_code(Rational(3, 7)) == "3 // 7"
    assert julia_code(Rational(18, 9)) == "2"
    assert julia_code(Rational(3, -7)) == "-3 // 7"
    assert julia_code(Rational(-3, -7)) == "3 // 7"
    assert julia_code(x + Rational(3, 7)) == "x + 3 // 7"
    assert julia_code(Rational(3, 7)*x) == "(3 // 7) * x"


def test_Relational():
    assert julia_code(Eq(x, y)) == "x == y"
    assert julia_code(Ne(x, y)) == "x != y"
    assert julia_code(Le(x, y)) == "x <= y"
    assert julia_code(Lt(x, y)) == "x < y"
    assert julia_code(Gt(x, y)) == "x > y"
    assert julia_code(Ge(x, y)) == "x >= y"


def test_Function():
    assert julia_code(sin(x) ** cos(x)) == "sin(x) .^ cos(x)"
    assert julia_code(abs(x)) == "abs(x)"
    assert julia_code(ceiling(x)) == "ceil(x)"


def test_Pow():
    assert julia_code(x**3) == "x .^ 3"
    assert julia_code(x**(y**3)) == "x .^ (y .^ 3)"
    assert julia_code(x**Rational(2, 3)) == 'x .^ (2 // 3)'
    g = implemented_function('g', Lambda(x, 2*x))
    assert julia_code(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5 * 2 * x) .^ (-x + y .^ x) ./ (x .^ 2 + y)"
    # For issue 14160
    assert julia_code(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
                                                evaluate=False)) == '-2 * x ./ (y .* y)'


def test_basic_ops():
    assert julia_code(x*y) == "x .* y"
    assert julia_code(x + y) == "x + y"
    assert julia_code(x - y) == "x - y"
    assert julia_code(-x) == "-x"


def test_1_over_x_and_sqrt():
    # 1.0 and 0.5 would do something different in regular StrPrinter,
    # but these are exact in IEEE floating point so no different here.
    assert julia_code(1/x) == '1 ./ x'
    assert julia_code(x**-1) == julia_code(x**-1.0) == '1 ./ x'
    assert julia_code(1/sqrt(x)) == '1 ./ sqrt(x)'
    assert julia_code(x**-S.Half) == julia_code(x**-0.5) == '1 ./ sqrt(x)'
    assert julia_code(sqrt(x)) == 'sqrt(x)'
    assert julia_code(x**S.Half) == julia_code(x**0.5) == 'sqrt(x)'
    assert julia_code(1/pi) == '1 / pi'
    assert julia_code(pi**-1) == julia_code(pi**-1.0) == '1 / pi'
    assert julia_code(pi**-0.5) == '1 / sqrt(pi)'


def test_mix_number_mult_symbols():
    assert julia_code(3*x) == "3 * x"
    assert julia_code(pi*x) == "pi * x"
    assert julia_code(3/x) == "3 ./ x"
    assert julia_code(pi/x) == "pi ./ x"
    assert julia_code(x/3) == "x / 3"
    assert julia_code(x/pi) == "x / pi"
    assert julia_code(x*y) == "x .* y"
    assert julia_code(3*x*y) == "3 * x .* y"
    assert julia_code(3*pi*x*y) == "3 * pi * x .* y"
    assert julia_code(x/y) == "x ./ y"
    assert julia_code(3*x/y) == "3 * x ./ y"
    assert julia_code(x*y/z) == "x .* y ./ z"
    assert julia_code(x/y*z) == "x .* z ./ y"
    assert julia_code(1/x/y) == "1 ./ (x .* y)"
    assert julia_code(2*pi*x/y/z) == "2 * pi * x ./ (y .* z)"
    assert julia_code(3*pi/x) == "3 * pi ./ x"
    assert julia_code(S(3)/5) == "3 // 5"
    assert julia_code(S(3)/5*x) == "(3 // 5) * x"
    assert julia_code(x/y/z) == "x ./ (y .* z)"
    assert julia_code((x+y)/z) == "(x + y) ./ z"
    assert julia_code((x+y)/(z+x)) == "(x + y) ./ (x + z)"
    assert julia_code((x+y)/EulerGamma) == "(x + y) / eulergamma"
    assert julia_code(x/3/pi) == "x / (3 * pi)"
    assert julia_code(S(3)/5*x*y/pi) == "(3 // 5) * x .* y / pi"


def test_mix_number_pow_symbols():
    assert julia_code(pi**3) == 'pi ^ 3'
    assert julia_code(x**2) == 'x .^ 2'
    assert julia_code(x**(pi**3)) == 'x .^ (pi ^ 3)'
    assert julia_code(x**y) == 'x .^ y'
    assert julia_code(x**(y**z)) == 'x .^ (y .^ z)'
    assert julia_code((x**y)**z) == '(x .^ y) .^ z'


def test_imag():
    I = S('I')
    assert julia_code(I) == "im"
    assert julia_code(5*I) == "5im"
    assert julia_code((S(3)/2)*I) == "(3 // 2) * im"
    assert julia_code(3+4*I) == "3 + 4im"


def test_constants():
    assert julia_code(pi) == "pi"
    assert julia_code(oo) == "Inf"
    assert julia_code(-oo) == "-Inf"
    assert julia_code(S.NegativeInfinity) == "-Inf"
    assert julia_code(S.NaN) == "NaN"
    assert julia_code(S.Exp1) == "e"
    assert julia_code(exp(1)) == "e"


def test_constants_other():
    assert julia_code(2*GoldenRatio) == "2 * golden"
    assert julia_code(2*Catalan) == "2 * catalan"
    assert julia_code(2*EulerGamma) == "2 * eulergamma"


def test_boolean():
    assert julia_code(x & y) == "x && y"
    assert julia_code(x | y) == "x || y"
    assert julia_code(~x) == "!x"
    assert julia_code(x & y & z) == "x && y && z"
    assert julia_code(x | y | z) == "x || y || z"
    assert julia_code((x & y) | z) == "z || x && y"
    assert julia_code((x | y) & z) == "z && (x || y)"

def test_sinc():
    assert julia_code(sinc(x)) == 'sinc(x / pi)'
    assert julia_code(sinc(x + 3)) == 'sinc((x + 3) / pi)'
    assert julia_code(sinc(pi * (x + 3))) == 'sinc(x + 3)'

def test_Matrices():
    assert julia_code(Matrix(1, 1, [10])) == "[10]"
    A = Matrix([[1, sin(x/2), abs(x)],
                [0, 1, pi],
                [0, exp(1), ceiling(x)]])
    expected = ("[1 sin(x / 2)  abs(x);\n"
                "0          1      pi;\n"
                "0          e ceil(x)]")
    assert julia_code(A) == expected
    # row and columns
    assert julia_code(A[:,0]) == "[1, 0, 0]"
    assert julia_code(A[0,:]) == "[1 sin(x / 2) abs(x)]"
    # empty matrices
    assert julia_code(Matrix(0, 0, [])) == 'zeros(0, 0)'
    assert julia_code(Matrix(0, 3, [])) == 'zeros(0, 3)'
    # annoying to read but correct
    assert julia_code(Matrix([[x, x - y, -y]])) == "[x x - y -y]"


def test_vector_entries_hadamard():
    # For a row or column, user might to use the other dimension
    A = Matrix([[1, sin(2/x), 3*pi/x/5]])
    assert julia_code(A) == "[1 sin(2 ./ x) (3 // 5) * pi ./ x]"
    assert julia_code(A.T) == "[1, sin(2 ./ x), (3 // 5) * pi ./ x]"


@XFAIL
def test_Matrices_entries_not_hadamard():
    # For Matrix with col >= 2, row >= 2, they need to be scalars
    # FIXME: is it worth worrying about this?  Its not wrong, just
    # leave it user's responsibility to put scalar data for x.
    A = Matrix([[1, sin(2/x), 3*pi/x/5], [1, 2, x*y]])
    expected = ("[1 sin(2/x) 3*pi/(5*x);\n"
                "1        2        x*y]") # <- we give x.*y
    assert julia_code(A) == expected


def test_MatrixSymbol():
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    assert julia_code(A*B) == "A * B"
    assert julia_code(B*A) == "B * A"
    assert julia_code(2*A*B) == "2 * A * B"
    assert julia_code(B*2*A) == "2 * B * A"
    assert julia_code(A*(B + 3*Identity(n))) == "A * (3 * eye(n) + B)"
    assert julia_code(A**(x**2)) == "A ^ (x .^ 2)"
    assert julia_code(A**3) == "A ^ 3"
    assert julia_code(A**S.Half) == "A ^ (1 // 2)"


def test_special_matrices():
    assert julia_code(6*Identity(3)) == "6 * eye(3)"


def test_containers():
    assert julia_code([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
        "Any[1, 2, 3, Any[4, 5, Any[6, 7]], 8, Any[9, 10], 11]"
    assert julia_code((1, 2, (3, 4))) == "(1, 2, (3, 4))"
    assert julia_code([1]) == "Any[1]"
    assert julia_code((1,)) == "(1,)"
    assert julia_code(Tuple(*[1, 2, 3])) == "(1, 2, 3)"
    assert julia_code((1, x*y, (3, x**2))) == "(1, x .* y, (3, x .^ 2))"
    # scalar, matrix, empty matrix and empty list
    assert julia_code((1, eye(3), Matrix(0, 0, []), [])) == "(1, [1 0 0;\n0 1 0;\n0 0 1], zeros(0, 0), Any[])"


def test_julia_noninline():
    source = julia_code((x+y)/Catalan, assign_to='me', inline=False)
    expected = (
        "const Catalan = %s\n"
        "me = (x + y) / Catalan"
    ) % Catalan.evalf(17)
    assert source == expected


def test_julia_piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    assert julia_code(expr) == "((x < 1) ? (x) : (x .^ 2))"
    assert julia_code(expr, assign_to="r") == (
        "r = ((x < 1) ? (x) : (x .^ 2))")
    assert julia_code(expr, assign_to="r", inline=False) == (
        "if (x < 1)\n"
        "    r = x\n"
        "else\n"
        "    r = x .^ 2\n"
        "end")
    expr = Piecewise((x**2, x < 1), (x**3, x < 2), (x**4, x < 3), (x**5, True))
    expected = ("((x < 1) ? (x .^ 2) :\n"
                "(x < 2) ? (x .^ 3) :\n"
                "(x < 3) ? (x .^ 4) : (x .^ 5))")
    assert julia_code(expr) == expected
    assert julia_code(expr, assign_to="r") == "r = " + expected
    assert julia_code(expr, assign_to="r", inline=False) == (
        "if (x < 1)\n"
        "    r = x .^ 2\n"
        "elseif (x < 2)\n"
        "    r = x .^ 3\n"
        "elseif (x < 3)\n"
        "    r = x .^ 4\n"
        "else\n"
        "    r = x .^ 5\n"
        "end")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: julia_code(expr))


def test_julia_piecewise_times_const():
    pw = Piecewise((x, x < 1), (x**2, True))
    assert julia_code(2*pw) == "2 * ((x < 1) ? (x) : (x .^ 2))"
    assert julia_code(pw/x) == "((x < 1) ? (x) : (x .^ 2)) ./ x"
    assert julia_code(pw/(x*y)) == "((x < 1) ? (x) : (x .^ 2)) ./ (x .* y)"
    assert julia_code(pw/3) == "((x < 1) ? (x) : (x .^ 2)) / 3"


def test_julia_matrix_assign_to():
    A = Matrix([[1, 2, 3]])
    assert julia_code(A, assign_to='a') == "a = [1 2 3]"
    A = Matrix([[1, 2], [3, 4]])
    assert julia_code(A, assign_to='A') == "A = [1 2;\n3 4]"


def test_julia_matrix_assign_to_more():
    # assigning to Symbol or MatrixSymbol requires lhs/rhs match
    A = Matrix([[1, 2, 3]])
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 2, 3)
    assert julia_code(A, assign_to=B) == "B = [1 2 3]"
    raises(ValueError, lambda: julia_code(A, assign_to=x))
    raises(ValueError, lambda: julia_code(A, assign_to=C))


def test_julia_matrix_1x1():
    A = Matrix([[3]])
    B = MatrixSymbol('B', 1, 1)
    C = MatrixSymbol('C', 1, 2)
    assert julia_code(A, assign_to=B) == "B = [3]"
    # FIXME?
    #assert julia_code(A, assign_to=x) == "x = [3]"
    raises(ValueError, lambda: julia_code(A, assign_to=C))


def test_julia_matrix_elements():
    A = Matrix([[x, 2, x*y]])
    assert julia_code(A[0, 0]**2 + A[0, 1] + A[0, 2]) == "x .^ 2 + x .* y + 2"
    A = MatrixSymbol('AA', 1, 3)
    assert julia_code(A) == "AA"
    assert julia_code(A[0, 0]**2 + sin(A[0,1]) + A[0,2]) == \
           "sin(AA[1,2]) + AA[1,1] .^ 2 + AA[1,3]"
    assert julia_code(sum(A)) == "AA[1,1] + AA[1,2] + AA[1,3]"


def test_julia_boolean():
    assert julia_code(True) == "true"
    assert julia_code(S.true) == "true"
    assert julia_code(False) == "false"
    assert julia_code(S.false) == "false"


def test_julia_not_supported():
    with raises(NotImplementedError):
        julia_code(S.ComplexInfinity)

    f = Function('f')
    assert julia_code(f(x).diff(x), strict=False) == (
        "# Not supported in Julia:\n"
        "# Derivative\n"
        "Derivative(f(x), x)"
    )


def test_trick_indent_with_end_else_words():
    # words starting with "end" or "else" do not confuse the indenter
    t1 = S('endless')
    t2 = S('elsewhere')
    pw = Piecewise((t1, x < 0), (t2, x <= 1), (1, True))
    assert julia_code(pw, inline=False) == (
        "if (x < 0)\n"
        "    endless\n"
        "elseif (x <= 1)\n"
        "    elsewhere\n"
        "else\n"
        "    1\n"
        "end")


def test_haramard():
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    v = MatrixSymbol('v', 3, 1)
    h = MatrixSymbol('h', 1, 3)
    C = HadamardProduct(A, B)
    assert julia_code(C) == "A .* B"
    assert julia_code(C*v) == "(A .* B) * v"
    assert julia_code(h*C*v) == "h * (A .* B) * v"
    assert julia_code(C*A) == "(A .* B) * A"
    # mixing Hadamard and scalar strange b/c we vectorize scalars
    assert julia_code(C*x*y) == "(x .* y) * (A .* B)"


def test_sparse():
    M = SparseMatrix(5, 6, {})
    M[2, 2] = 10
    M[1, 2] = 20
    M[1, 3] = 22
    M[0, 3] = 30
    M[3, 0] = x*y
    assert julia_code(M) == (
        "sparse([4, 2, 3, 1, 2], [1, 3, 3, 4, 4], [x .* y, 20, 10, 30, 22], 5, 6)"
    )


def test_specfun():
    n = Symbol('n')
    for f in [besselj, bessely, besseli, besselk]:
        assert julia_code(f(n, x)) == f.__name__ + '(n, x)'
    for f in [airyai, airyaiprime, airybi, airybiprime]:
        assert julia_code(f(x)) == f.__name__ + '(x)'
    assert julia_code(hankel1(n, x)) == 'hankelh1(n, x)'
    assert julia_code(hankel2(n, x)) == 'hankelh2(n, x)'
    assert julia_code(jn(n, x)) == 'sqrt(2) * sqrt(pi) * sqrt(1 ./ x) .* besselj(n + 1 // 2, x) / 2'
    assert julia_code(yn(n, x)) == 'sqrt(2) * sqrt(pi) * sqrt(1 ./ x) .* bessely(n + 1 // 2, x) / 2'


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(julia_code(A[0, 0]) == "A[1,1]")
    assert(julia_code(3 * A[0, 0]) == "3 * A[1,1]")

    F = C[0, 0].subs(C, A - B)
    assert(julia_code(F) == "(A - B)[1,1]")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_append.py ===
import datetime as dt
from itertools import combinations

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    concat,
    isna,
)
import pandas._testing as tm


class TestAppend:
    def test_append(self, sort, float_frame):
        mixed_frame = float_frame.copy()
        mixed_frame["foo"] = "bar"

        begin_index = float_frame.index[:5]
        end_index = float_frame.index[5:]

        begin_frame = float_frame.reindex(begin_index)
        end_frame = float_frame.reindex(end_index)

        appended = begin_frame._append(end_frame)
        tm.assert_almost_equal(appended["A"], float_frame["A"])

        del end_frame["A"]
        partial_appended = begin_frame._append(end_frame, sort=sort)
        assert "A" in partial_appended

        partial_appended = end_frame._append(begin_frame, sort=sort)
        assert "A" in partial_appended

        # mixed type handling
        appended = mixed_frame[:5]._append(mixed_frame[5:])
        tm.assert_frame_equal(appended, mixed_frame)

        # what to test here
        mixed_appended = mixed_frame[:5]._append(float_frame[5:], sort=sort)
        mixed_appended2 = float_frame[:5]._append(mixed_frame[5:], sort=sort)

        # all equal except 'foo' column
        tm.assert_frame_equal(
            mixed_appended.reindex(columns=["A", "B", "C", "D"]),
            mixed_appended2.reindex(columns=["A", "B", "C", "D"]),
        )

    def test_append_empty(self, float_frame):
        empty = DataFrame()

        appended = float_frame._append(empty)
        tm.assert_frame_equal(float_frame, appended)
        assert appended is not float_frame

        appended = empty._append(float_frame)
        tm.assert_frame_equal(float_frame, appended)
        assert appended is not float_frame

    def test_append_overlap_raises(self, float_frame):
        msg = "Indexes have overlapping values"
        with pytest.raises(ValueError, match=msg):
            float_frame._append(float_frame, verify_integrity=True)

    def test_append_new_columns(self):
        # see gh-6129: new columns
        df = DataFrame({"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}})
        row = Series([5, 6, 7], index=["a", "b", "c"], name="z")
        expected = DataFrame(
            {
                "a": {"x": 1, "y": 2, "z": 5},
                "b": {"x": 3, "y": 4, "z": 6},
                "c": {"z": 7},
            }
        )
        result = df._append(row)
        tm.assert_frame_equal(result, expected)

    def test_append_length0_frame(self, sort):
        df = DataFrame(columns=["A", "B", "C"])
        df3 = DataFrame(index=[0, 1], columns=["A", "B"])
        df5 = df._append(df3, sort=sort)

        expected = DataFrame(index=[0, 1], columns=["A", "B", "C"])
        tm.assert_frame_equal(df5, expected)

    def test_append_records(self):
        arr1 = np.zeros((2,), dtype=("i4,f4,S10"))
        arr1[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]

        arr2 = np.zeros((3,), dtype=("i4,f4,S10"))
        arr2[:] = [(3, 4.0, "foo"), (5, 6.0, "bar"), (7.0, 8.0, "baz")]

        df1 = DataFrame(arr1)
        df2 = DataFrame(arr2)

        result = df1._append(df2, ignore_index=True)
        expected = DataFrame(np.concatenate((arr1, arr2)))
        tm.assert_frame_equal(result, expected)

    # rewrite sort fixture, since we also want to test default of None
    def test_append_sorts(self, sort):
        df1 = DataFrame({"a": [1, 2], "b": [1, 2]}, columns=["b", "a"])
        df2 = DataFrame({"a": [1, 2], "c": [3, 4]}, index=[2, 3])

        result = df1._append(df2, sort=sort)

        # for None / True
        expected = DataFrame(
            {"b": [1, 2, None, None], "a": [1, 2, 1, 2], "c": [None, None, 3, 4]},
            columns=["a", "b", "c"],
        )
        if sort is False:
            expected = expected[["b", "a", "c"]]
        tm.assert_frame_equal(result, expected)

    def test_append_different_columns(self, sort):
        df = DataFrame(
            {
                "bools": np.random.default_rng(2).standard_normal(10) > 0,
                "ints": np.random.default_rng(2).integers(0, 10, 10),
                "floats": np.random.default_rng(2).standard_normal(10),
                "strings": ["foo", "bar"] * 5,
            }
        )

        a = df[:5].loc[:, ["bools", "ints", "floats"]]
        b = df[5:].loc[:, ["strings", "ints", "floats"]]

        appended = a._append(b, sort=sort)
        assert isna(appended["strings"][0:4]).all()
        assert isna(appended["bools"][5:]).all()

    def test_append_many(self, sort, float_frame):
        chunks = [
            float_frame[:5],
            float_frame[5:10],
            float_frame[10:15],
            float_frame[15:],
        ]

        result = chunks[0]._append(chunks[1:])
        tm.assert_frame_equal(result, float_frame)

        chunks[-1] = chunks[-1].copy()
        chunks[-1]["foo"] = "bar"
        result = chunks[0]._append(chunks[1:], sort=sort)
        tm.assert_frame_equal(result.loc[:, float_frame.columns], float_frame)
        assert (result["foo"][15:] == "bar").all()
        assert result["foo"][:15].isna().all()

    def test_append_preserve_index_name(self):
        # #980
        df1 = DataFrame(columns=["A", "B", "C"])
        df1 = df1.set_index(["A"])
        df2 = DataFrame(data=[[1, 4, 7], [2, 5, 8], [3, 6, 9]], columns=["A", "B", "C"])
        df2 = df2.set_index(["A"])

        msg = "The behavior of array concatenation with empty entries is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df1._append(df2)
        assert result.index.name == "A"

    indexes_can_append = [
        pd.RangeIndex(3),
        Index([4, 5, 6]),
        Index([4.5, 5.5, 6.5]),
        Index(list("abc")),
        pd.CategoricalIndex("A B C".split()),
        pd.CategoricalIndex("D E F".split(), ordered=True),
        pd.IntervalIndex.from_breaks([7, 8, 9, 10]),
        pd.DatetimeIndex(
            [
                dt.datetime(2013, 1, 3, 0, 0),
                dt.datetime(2013, 1, 3, 6, 10),
                dt.datetime(2013, 1, 3, 7, 12),
            ]
        ),
        pd.MultiIndex.from_arrays(["A B C".split(), "D E F".split()]),
    ]

    @pytest.mark.parametrize(
        "index", indexes_can_append, ids=lambda x: type(x).__name__
    )
    def test_append_same_columns_type(self, index):
        # GH18359

        # df wider than ser
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=index)
        ser_index = index[:2]
        ser = Series([7, 8], index=ser_index, name=2)
        result = df._append(ser)
        expected = DataFrame(
            [[1, 2, 3.0], [4, 5, 6], [7, 8, np.nan]], index=[0, 1, 2], columns=index
        )
        # integer dtype is preserved for columns present in ser.index
        assert expected.dtypes.iloc[0].kind == "i"
        assert expected.dtypes.iloc[1].kind == "i"

        tm.assert_frame_equal(result, expected)

        # ser wider than df
        ser_index = index
        index = index[:2]
        df = DataFrame([[1, 2], [4, 5]], columns=index)
        ser = Series([7, 8, 9], index=ser_index, name=2)
        result = df._append(ser)
        expected = DataFrame(
            [[1, 2, np.nan], [4, 5, np.nan], [7, 8, 9]],
            index=[0, 1, 2],
            columns=ser_index,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "df_columns, series_index",
        combinations(indexes_can_append, r=2),
        ids=lambda x: type(x).__name__,
    )
    def test_append_different_columns_types(self, df_columns, series_index):
        # GH18359
        # See also test 'test_append_different_columns_types_raises' below
        # for errors raised when appending

        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=df_columns)
        ser = Series([7, 8, 9], index=series_index, name=2)

        result = df._append(ser)
        idx_diff = ser.index.difference(df_columns)
        combined_columns = Index(df_columns.tolist()).append(idx_diff)
        expected = DataFrame(
            [
                [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
                [4, 5, 6, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 7, 8, 9],
            ],
            index=[0, 1, 2],
            columns=combined_columns,
        )
        tm.assert_frame_equal(result, expected)

    def test_append_dtype_coerce(self, sort):
        # GH 4993
        # appending with datetime will incorrectly convert datetime64

        df1 = DataFrame(
            index=[1, 2],
            data=[dt.datetime(2013, 1, 1, 0, 0), dt.datetime(2013, 1, 2, 0, 0)],
            columns=["start_time"],
        )
        df2 = DataFrame(
            index=[4, 5],
            data=[
                [dt.datetime(2013, 1, 3, 0, 0), dt.datetime(2013, 1, 3, 6, 10)],
                [dt.datetime(2013, 1, 4, 0, 0), dt.datetime(2013, 1, 4, 7, 10)],
            ],
            columns=["start_time", "end_time"],
        )

        expected = concat(
            [
                Series(
                    [
                        pd.NaT,
                        pd.NaT,
                        dt.datetime(2013, 1, 3, 6, 10),
                        dt.datetime(2013, 1, 4, 7, 10),
                    ],
                    name="end_time",
                ),
                Series(
                    [
                        dt.datetime(2013, 1, 1, 0, 0),
                        dt.datetime(2013, 1, 2, 0, 0),
                        dt.datetime(2013, 1, 3, 0, 0),
                        dt.datetime(2013, 1, 4, 0, 0),
                    ],
                    name="start_time",
                ),
            ],
            axis=1,
            sort=sort,
        )
        result = df1._append(df2, ignore_index=True, sort=sort)
        if sort:
            expected = expected[["end_time", "start_time"]]
        else:
            expected = expected[["start_time", "end_time"]]

        tm.assert_frame_equal(result, expected)

    def test_append_missing_column_proper_upcast(self, sort):
        df1 = DataFrame({"A": np.array([1, 2, 3, 4], dtype="i8")})
        df2 = DataFrame({"B": np.array([True, False, True, False], dtype=bool)})

        appended = df1._append(df2, ignore_index=True, sort=sort)
        assert appended["A"].dtype == "f8"
        assert appended["B"].dtype == "O"

    def test_append_empty_frame_to_series_with_dateutil_tz(self):
        # GH 23682
        date = Timestamp("2018-10-24 07:30:00", tz=dateutil.tz.tzutc())
        ser = Series({"a": 1.0, "b": 2.0, "date": date})
        df = DataFrame(columns=["c", "d"])
        result_a = df._append(ser, ignore_index=True)
        expected = DataFrame(
            [[np.nan, np.nan, 1.0, 2.0, date]], columns=["c", "d", "a", "b", "date"]
        )
        # These columns get cast to object after append
        expected["c"] = expected["c"].astype(object)
        expected["d"] = expected["d"].astype(object)
        tm.assert_frame_equal(result_a, expected)

        expected = DataFrame(
            [[np.nan, np.nan, 1.0, 2.0, date]] * 2, columns=["c", "d", "a", "b", "date"]
        )
        expected["c"] = expected["c"].astype(object)
        expected["d"] = expected["d"].astype(object)
        result_b = result_a._append(ser, ignore_index=True)
        tm.assert_frame_equal(result_b, expected)

        result = df._append([ser, ser], ignore_index=True)
        tm.assert_frame_equal(result, expected)

    def test_append_empty_tz_frame_with_datetime64ns(self, using_array_manager):
        # https://github.com/pandas-dev/pandas/issues/35460
        df = DataFrame(columns=["a"]).astype("datetime64[ns, UTC]")

        # pd.NaT gets inferred as tz-naive, so append result is tz-naive
        result = df._append({"a": pd.NaT}, ignore_index=True)
        if using_array_manager:
            expected = DataFrame({"a": [pd.NaT]}, dtype=object)
        else:
            expected = DataFrame({"a": [np.nan]}, dtype=object)
        tm.assert_frame_equal(result, expected)

        # also test with typed value to append
        df = DataFrame(columns=["a"]).astype("datetime64[ns, UTC]")
        other = Series({"a": pd.NaT}, dtype="datetime64[ns]")
        result = df._append(other, ignore_index=True)
        tm.assert_frame_equal(result, expected)

        # mismatched tz
        other = Series({"a": pd.NaT}, dtype="datetime64[ns, US/Pacific]")
        result = df._append(other, ignore_index=True)
        expected = DataFrame({"a": [pd.NaT]}).astype(object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_str", ["datetime64[ns, UTC]", "datetime64[ns]", "Int64", "int64"]
    )
    @pytest.mark.parametrize("val", [1, "NaT"])
    def test_append_empty_frame_with_timedelta64ns_nat(
        self, dtype_str, val, using_array_manager
    ):
        # https://github.com/pandas-dev/pandas/issues/35460
        df = DataFrame(columns=["a"]).astype(dtype_str)

        other = DataFrame({"a": [np.timedelta64(val, "ns")]})
        result = df._append(other, ignore_index=True)

        expected = other.astype(object)
        if isinstance(val, str) and dtype_str != "int64" and not using_array_manager:
            # TODO: expected used to be `other.astype(object)` which is a more
            #  reasonable result.  This was changed when tightening
            #  assert_frame_equal's treatment of mismatched NAs to match the
            #  existing behavior.
            expected = DataFrame({"a": [np.nan]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_str", ["datetime64[ns, UTC]", "datetime64[ns]", "Int64", "int64"]
    )
    @pytest.mark.parametrize("val", [1, "NaT"])
    def test_append_frame_with_timedelta64ns_nat(self, dtype_str, val):
        # https://github.com/pandas-dev/pandas/issues/35460
        df = DataFrame({"a": pd.array([1], dtype=dtype_str)})

        other = DataFrame({"a": [np.timedelta64(val, "ns")]})
        result = df._append(other, ignore_index=True)

        expected = DataFrame({"a": [df.iloc[0, 0], other.iloc[0, 0]]}, dtype=object)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_indexing.py ===
import math

import numpy as np
import pytest

from pandas import (
    NA,
    Categorical,
    CategoricalIndex,
    Index,
    Interval,
    IntervalIndex,
    NaT,
    PeriodIndex,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm
import pandas.core.common as com


class TestCategoricalIndexingWithFactor:
    def test_getitem(self):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        assert factor[0] == "a"
        assert factor[-1] == "c"

        subf = factor[[0, 1, 2]]
        tm.assert_numpy_array_equal(subf._codes, np.array([0, 1, 1], dtype=np.int8))

        subf = factor[np.asarray(factor) == "c"]
        tm.assert_numpy_array_equal(subf._codes, np.array([2, 2, 2], dtype=np.int8))

    def test_setitem(self):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        # int/positional
        c = factor.copy()
        c[0] = "b"
        assert c[0] == "b"
        c[-1] = "a"
        assert c[-1] == "a"

        # boolean
        c = factor.copy()
        indexer = np.zeros(len(c), dtype="bool")
        indexer[0] = True
        indexer[-1] = True
        c[indexer] = "c"
        expected = Categorical(["c", "b", "b", "a", "a", "c", "c", "c"], ordered=True)

        tm.assert_categorical_equal(c, expected)

    @pytest.mark.parametrize(
        "other",
        [Categorical(["b", "a"]), Categorical(["b", "a"], categories=["b", "a"])],
    )
    def test_setitem_same_but_unordered(self, other):
        # GH-24142
        target = Categorical(["a", "b"], categories=["a", "b"])
        mask = np.array([True, False])
        target[mask] = other[mask]
        expected = Categorical(["b", "b"], categories=["a", "b"])
        tm.assert_categorical_equal(target, expected)

    @pytest.mark.parametrize(
        "other",
        [
            Categorical(["b", "a"], categories=["b", "a", "c"]),
            Categorical(["b", "a"], categories=["a", "b", "c"]),
            Categorical(["a", "a"], categories=["a"]),
            Categorical(["b", "b"], categories=["b"]),
        ],
    )
    def test_setitem_different_unordered_raises(self, other):
        # GH-24142
        target = Categorical(["a", "b"], categories=["a", "b"])
        mask = np.array([True, False])
        msg = "Cannot set a Categorical with another, without identical categories"
        with pytest.raises(TypeError, match=msg):
            target[mask] = other[mask]

    @pytest.mark.parametrize(
        "other",
        [
            Categorical(["b", "a"]),
            Categorical(["b", "a"], categories=["b", "a"], ordered=True),
            Categorical(["b", "a"], categories=["a", "b", "c"], ordered=True),
        ],
    )
    def test_setitem_same_ordered_raises(self, other):
        # Gh-24142
        target = Categorical(["a", "b"], categories=["a", "b"], ordered=True)
        mask = np.array([True, False])
        msg = "Cannot set a Categorical with another, without identical categories"
        with pytest.raises(TypeError, match=msg):
            target[mask] = other[mask]

    def test_setitem_tuple(self):
        # GH#20439
        cat = Categorical([(0, 1), (0, 2), (0, 1)])

        # This should not raise
        cat[1] = cat[0]
        assert cat[1] == (0, 1)

    def test_setitem_listlike(self):
        # GH#9469
        # properly coerce the input indexers

        cat = Categorical(
            np.random.default_rng(2).integers(0, 5, size=150000).astype(np.int8)
        ).add_categories([-1000])
        indexer = np.array([100000]).astype(np.int64)
        cat[indexer] = -1000

        # we are asserting the code result here
        # which maps to the -1000 category
        result = cat.codes[np.array([100000]).astype(np.int64)]
        tm.assert_numpy_array_equal(result, np.array([5], dtype="int8"))


class TestCategoricalIndexing:
    def test_getitem_slice(self):
        cat = Categorical(["a", "b", "c", "d", "a", "b", "c"])
        sliced = cat[3]
        assert sliced == "d"

        sliced = cat[3:5]
        expected = Categorical(["d", "a"], categories=["a", "b", "c", "d"])
        tm.assert_categorical_equal(sliced, expected)

    def test_getitem_listlike(self):
        # GH 9469
        # properly coerce the input indexers

        c = Categorical(
            np.random.default_rng(2).integers(0, 5, size=150000).astype(np.int8)
        )
        result = c.codes[np.array([100000]).astype(np.int64)]
        expected = c[np.array([100000]).astype(np.int64)].codes
        tm.assert_numpy_array_equal(result, expected)

    def test_periodindex(self):
        idx1 = PeriodIndex(
            ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"],
            freq="M",
        )

        cat1 = Categorical(idx1)
        str(cat1)
        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
        exp_idx = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")
        tm.assert_numpy_array_equal(cat1._codes, exp_arr)
        tm.assert_index_equal(cat1.categories, exp_idx)

        idx2 = PeriodIndex(
            ["2014-03", "2014-03", "2014-02", "2014-01", "2014-03", "2014-01"],
            freq="M",
        )
        cat2 = Categorical(idx2, ordered=True)
        str(cat2)
        exp_arr = np.array([2, 2, 1, 0, 2, 0], dtype=np.int8)
        exp_idx2 = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")
        tm.assert_numpy_array_equal(cat2._codes, exp_arr)
        tm.assert_index_equal(cat2.categories, exp_idx2)

        idx3 = PeriodIndex(
            [
                "2013-12",
                "2013-11",
                "2013-10",
                "2013-09",
                "2013-08",
                "2013-07",
                "2013-05",
            ],
            freq="M",
        )
        cat3 = Categorical(idx3, ordered=True)
        exp_arr = np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
        exp_idx = PeriodIndex(
            [
                "2013-05",
                "2013-07",
                "2013-08",
                "2013-09",
                "2013-10",
                "2013-11",
                "2013-12",
            ],
            freq="M",
        )
        tm.assert_numpy_array_equal(cat3._codes, exp_arr)
        tm.assert_index_equal(cat3.categories, exp_idx)

    @pytest.mark.parametrize(
        "null_val",
        [None, np.nan, NaT, NA, math.nan, "NaT", "nat", "NAT", "nan", "NaN", "NAN"],
    )
    def test_periodindex_on_null_types(self, null_val):
        # GH 46673
        result = PeriodIndex(["2022-04-06", "2022-04-07", null_val], freq="D")
        expected = PeriodIndex(["2022-04-06", "2022-04-07", "NaT"], dtype="period[D]")
        assert result[2] is NaT
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("new_categories", [[1, 2, 3, 4], [1, 2]])
    def test_categories_assignments_wrong_length_raises(self, new_categories):
        cat = Categorical(["a", "b", "c", "a"])
        msg = (
            "new categories need to have the same number of items "
            "as the old categories!"
        )
        with pytest.raises(ValueError, match=msg):
            cat.rename_categories(new_categories)

    # Combinations of sorted/unique:
    @pytest.mark.parametrize(
        "idx_values", [[1, 2, 3, 4], [1, 3, 2, 4], [1, 3, 3, 4], [1, 2, 2, 4]]
    )
    # Combinations of missing/unique
    @pytest.mark.parametrize("key_values", [[1, 2], [1, 5], [1, 1], [5, 5]])
    @pytest.mark.parametrize("key_class", [Categorical, CategoricalIndex])
    @pytest.mark.parametrize("dtype", [None, "category", "key"])
    def test_get_indexer_non_unique(self, idx_values, key_values, key_class, dtype):
        # GH 21448
        key = key_class(key_values, categories=range(1, 5))

        if dtype == "key":
            dtype = key.dtype

        # Test for flat index and CategoricalIndex with same/different cats:
        idx = Index(idx_values, dtype=dtype)
        expected, exp_miss = idx.get_indexer_non_unique(key_values)
        result, res_miss = idx.get_indexer_non_unique(key)

        tm.assert_numpy_array_equal(expected, result)
        tm.assert_numpy_array_equal(exp_miss, res_miss)

        exp_unique = idx.unique().get_indexer(key_values)
        res_unique = idx.unique().get_indexer(key)
        tm.assert_numpy_array_equal(res_unique, exp_unique)

    def test_where_unobserved_nan(self):
        ser = Series(Categorical(["a", "b"]))
        result = ser.where([True, False])
        expected = Series(Categorical(["a", None], categories=["a", "b"]))
        tm.assert_series_equal(result, expected)

        # all NA
        ser = Series(Categorical(["a", "b"]))
        result = ser.where([False, False])
        expected = Series(Categorical([None, None], categories=["a", "b"]))
        tm.assert_series_equal(result, expected)

    def test_where_unobserved_categories(self):
        ser = Series(Categorical(["a", "b", "c"], categories=["d", "c", "b", "a"]))
        result = ser.where([True, True, False], other="b")
        expected = Series(Categorical(["a", "b", "b"], categories=ser.cat.categories))
        tm.assert_series_equal(result, expected)

    def test_where_other_categorical(self):
        ser = Series(Categorical(["a", "b", "c"], categories=["d", "c", "b", "a"]))
        other = Categorical(["b", "c", "a"], categories=["a", "c", "b", "d"])
        result = ser.where([True, False, True], other)
        expected = Series(Categorical(["a", "c", "c"], dtype=ser.dtype))
        tm.assert_series_equal(result, expected)

    def test_where_new_category_raises(self):
        ser = Series(Categorical(["a", "b", "c"]))
        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(TypeError, match=msg):
            ser.where([True, False, True], "d")

    def test_where_ordered_differs_rasies(self):
        ser = Series(
            Categorical(["a", "b", "c"], categories=["d", "c", "b", "a"], ordered=True)
        )
        other = Categorical(
            ["b", "c", "a"], categories=["a", "c", "b", "d"], ordered=True
        )
        with pytest.raises(TypeError, match="without identical categories"):
            ser.where([True, False, True], other)


class TestContains:
    def test_contains(self):
        # GH#21508
        cat = Categorical(list("aabbca"), categories=list("cab"))

        assert "b" in cat
        assert "z" not in cat
        assert np.nan not in cat
        with pytest.raises(TypeError, match="unhashable type: 'list'"):
            assert [1] in cat

        # assert codes NOT in index
        assert 0 not in cat
        assert 1 not in cat

        cat = Categorical(list("aabbca") + [np.nan], categories=list("cab"))
        assert np.nan in cat

    @pytest.mark.parametrize(
        "item, expected",
        [
            (Interval(0, 1), True),
            (1.5, True),
            (Interval(0.5, 1.5), False),
            ("a", False),
            (Timestamp(1), False),
            (Timedelta(1), False),
        ],
        ids=str,
    )
    def test_contains_interval(self, item, expected):
        # GH#23705
        cat = Categorical(IntervalIndex.from_breaks(range(3)))
        result = item in cat
        assert result is expected

    def test_contains_list(self):
        # GH#21729
        cat = Categorical([1, 2, 3])

        assert "a" not in cat

        with pytest.raises(TypeError, match="unhashable type"):
            ["a"] in cat

        with pytest.raises(TypeError, match="unhashable type"):
            ["a", "b"] in cat


@pytest.mark.parametrize("index", [True, False])
def test_mask_with_boolean(index):
    ser = Series(range(3))
    idx = Categorical([True, False, True])
    if index:
        idx = CategoricalIndex(idx)

    assert com.is_bool_indexer(idx)
    result = ser[idx]
    expected = ser[idx.astype("object")]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("index", [True, False])
def test_mask_with_boolean_na_treated_as_false(index):
    # https://github.com/pandas-dev/pandas/issues/31503
    ser = Series(range(3))
    idx = Categorical([True, False, None])
    if index:
        idx = CategoricalIndex(idx)

    result = ser[idx]
    expected = ser[idx.fillna(False)]

    tm.assert_series_equal(result, expected)


@pytest.fixture
def non_coercible_categorical(monkeypatch):
    """
    Monkeypatch Categorical.__array__ to ensure no implicit conversion.

    Raises
    ------
    ValueError
        When Categorical.__array__ is called.
    """

    # TODO(Categorical): identify other places where this may be
    # useful and move to a conftest.py
    def array(self, dtype=None):
        raise ValueError("I cannot be converted.")

    with monkeypatch.context() as m:
        m.setattr(Categorical, "__array__", array)
        yield


def test_series_at():
    arr = Categorical(["a", "b", "c"])
    ser = Series(arr)
    result = ser.at[0]
    assert result == "a"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_get_set.py ===
import numpy as np
import pytest

from pandas.compat import PY311

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import (
    CategoricalIndex,
    MultiIndex,
)
import pandas._testing as tm


def assert_matching(actual, expected, check_dtype=False):
    # avoid specifying internal representation
    # as much as possible
    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        act = np.asarray(act)
        exp = np.asarray(exp)
        tm.assert_numpy_array_equal(act, exp, check_dtype=check_dtype)


def test_get_level_number_integer(idx):
    idx.names = [1, 0]
    assert idx._get_level_number(1) == 0
    assert idx._get_level_number(0) == 1
    msg = "Too many levels: Index has only 2 levels, not 3"
    with pytest.raises(IndexError, match=msg):
        idx._get_level_number(2)
    with pytest.raises(KeyError, match="Level fourth not found"):
        idx._get_level_number("fourth")


def test_get_dtypes(using_infer_string):
    # Test MultiIndex.dtypes (# Gh37062)
    idx_multitype = MultiIndex.from_product(
        [[1, 2, 3], ["a", "b", "c"], pd.date_range("20200101", periods=2, tz="UTC")],
        names=["int", "string", "dt"],
    )

    exp = "object" if not using_infer_string else pd.StringDtype(na_value=np.nan)
    expected = pd.Series(
        {
            "int": np.dtype("int64"),
            "string": exp,
            "dt": DatetimeTZDtype(tz="utc"),
        }
    )
    tm.assert_series_equal(expected, idx_multitype.dtypes)


def test_get_dtypes_no_level_name(using_infer_string):
    # Test MultiIndex.dtypes (# GH38580 )
    idx_multitype = MultiIndex.from_product(
        [
            [1, 2, 3],
            ["a", "b", "c"],
            pd.date_range("20200101", periods=2, tz="UTC"),
        ],
    )
    exp = "object" if not using_infer_string else pd.StringDtype(na_value=np.nan)
    expected = pd.Series(
        {
            "level_0": np.dtype("int64"),
            "level_1": exp,
            "level_2": DatetimeTZDtype(tz="utc"),
        }
    )
    tm.assert_series_equal(expected, idx_multitype.dtypes)


def test_get_dtypes_duplicate_level_names(using_infer_string):
    # Test MultiIndex.dtypes with non-unique level names (# GH45174)
    result = MultiIndex.from_product(
        [
            [1, 2, 3],
            ["a", "b", "c"],
            pd.date_range("20200101", periods=2, tz="UTC"),
        ],
        names=["A", "A", "A"],
    ).dtypes
    exp = "object" if not using_infer_string else pd.StringDtype(na_value=np.nan)
    expected = pd.Series(
        [np.dtype("int64"), exp, DatetimeTZDtype(tz="utc")],
        index=["A", "A", "A"],
    )
    tm.assert_series_equal(result, expected)


def test_get_level_number_out_of_bounds(multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data

    with pytest.raises(IndexError, match="Too many levels"):
        frame.index._get_level_number(2)
    with pytest.raises(IndexError, match="not a valid level number"):
        frame.index._get_level_number(-3)


def test_set_name_methods(idx):
    # so long as these are synonyms, we don't need to test set_names
    index_names = ["first", "second"]
    assert idx.rename == idx.set_names
    new_names = [name + "SUFFIX" for name in index_names]
    ind = idx.set_names(new_names)
    assert idx.names == index_names
    assert ind.names == new_names
    msg = "Length of names must match number of levels in MultiIndex"
    with pytest.raises(ValueError, match=msg):
        ind.set_names(new_names + new_names)
    new_names2 = [name + "SUFFIX2" for name in new_names]
    res = ind.set_names(new_names2, inplace=True)
    assert res is None
    assert ind.names == new_names2

    # set names for specific level (# GH7792)
    ind = idx.set_names(new_names[0], level=0)
    assert idx.names == index_names
    assert ind.names == [new_names[0], index_names[1]]

    res = ind.set_names(new_names2[0], level=0, inplace=True)
    assert res is None
    assert ind.names == [new_names2[0], index_names[1]]

    # set names for multiple levels
    ind = idx.set_names(new_names, level=[0, 1])
    assert idx.names == index_names
    assert ind.names == new_names

    res = ind.set_names(new_names2, level=[0, 1], inplace=True)
    assert res is None
    assert ind.names == new_names2


def test_set_levels_codes_directly(idx):
    # setting levels/codes directly raises AttributeError

    levels = idx.levels
    new_levels = [[lev + "a" for lev in level] for level in levels]

    codes = idx.codes
    major_codes, minor_codes = codes
    major_codes = [(x + 1) % 3 for x in major_codes]
    minor_codes = [(x + 1) % 1 for x in minor_codes]
    new_codes = [major_codes, minor_codes]

    msg = "Can't set attribute"
    with pytest.raises(AttributeError, match=msg):
        idx.levels = new_levels

    msg = (
        "property 'codes' of 'MultiIndex' object has no setter"
        if PY311
        else "can't set attribute"
    )
    with pytest.raises(AttributeError, match=msg):
        idx.codes = new_codes


def test_set_levels(idx):
    # side note - you probably wouldn't want to use levels and codes
    # directly like this - but it is possible.
    levels = idx.levels
    new_levels = [[lev + "a" for lev in level] for level in levels]

    # level changing [w/o mutation]
    ind2 = idx.set_levels(new_levels)
    assert_matching(ind2.levels, new_levels)
    assert_matching(idx.levels, levels)

    # level changing specific level [w/o mutation]
    ind2 = idx.set_levels(new_levels[0], level=0)
    assert_matching(ind2.levels, [new_levels[0], levels[1]])
    assert_matching(idx.levels, levels)

    ind2 = idx.set_levels(new_levels[1], level=1)
    assert_matching(ind2.levels, [levels[0], new_levels[1]])
    assert_matching(idx.levels, levels)

    # level changing multiple levels [w/o mutation]
    ind2 = idx.set_levels(new_levels, level=[0, 1])
    assert_matching(ind2.levels, new_levels)
    assert_matching(idx.levels, levels)

    # illegal level changing should not change levels
    # GH 13754
    original_index = idx.copy()
    with pytest.raises(ValueError, match="^On"):
        idx.set_levels(["c"], level=0)
    assert_matching(idx.levels, original_index.levels, check_dtype=True)

    with pytest.raises(ValueError, match="^On"):
        idx.set_codes([0, 1, 2, 3, 4, 5], level=0)
    assert_matching(idx.codes, original_index.codes, check_dtype=True)

    with pytest.raises(TypeError, match="^Levels"):
        idx.set_levels("c", level=0)
    assert_matching(idx.levels, original_index.levels, check_dtype=True)

    with pytest.raises(TypeError, match="^Codes"):
        idx.set_codes(1, level=0)
    assert_matching(idx.codes, original_index.codes, check_dtype=True)


def test_set_codes(idx):
    # side note - you probably wouldn't want to use levels and codes
    # directly like this - but it is possible.
    codes = idx.codes
    major_codes, minor_codes = codes
    major_codes = [(x + 1) % 3 for x in major_codes]
    minor_codes = [(x + 1) % 1 for x in minor_codes]
    new_codes = [major_codes, minor_codes]

    # changing codes w/o mutation
    ind2 = idx.set_codes(new_codes)
    assert_matching(ind2.codes, new_codes)
    assert_matching(idx.codes, codes)

    # codes changing specific level w/o mutation
    ind2 = idx.set_codes(new_codes[0], level=0)
    assert_matching(ind2.codes, [new_codes[0], codes[1]])
    assert_matching(idx.codes, codes)

    ind2 = idx.set_codes(new_codes[1], level=1)
    assert_matching(ind2.codes, [codes[0], new_codes[1]])
    assert_matching(idx.codes, codes)

    # codes changing multiple levels w/o mutation
    ind2 = idx.set_codes(new_codes, level=[0, 1])
    assert_matching(ind2.codes, new_codes)
    assert_matching(idx.codes, codes)

    # label changing for levels of different magnitude of categories
    ind = MultiIndex.from_tuples([(0, i) for i in range(130)])
    new_codes = range(129, -1, -1)
    expected = MultiIndex.from_tuples([(0, i) for i in new_codes])

    # [w/o mutation]
    result = ind.set_codes(codes=new_codes, level=1)
    assert result.equals(expected)


def test_set_levels_codes_names_bad_input(idx):
    levels, codes = idx.levels, idx.codes
    names = idx.names

    with pytest.raises(ValueError, match="Length of levels"):
        idx.set_levels([levels[0]])

    with pytest.raises(ValueError, match="Length of codes"):
        idx.set_codes([codes[0]])

    with pytest.raises(ValueError, match="Length of names"):
        idx.set_names([names[0]])

    # shouldn't scalar data error, instead should demand list-like
    with pytest.raises(TypeError, match="list of lists-like"):
        idx.set_levels(levels[0])

    # shouldn't scalar data error, instead should demand list-like
    with pytest.raises(TypeError, match="list of lists-like"):
        idx.set_codes(codes[0])

    # shouldn't scalar data error, instead should demand list-like
    with pytest.raises(TypeError, match="list-like"):
        idx.set_names(names[0])

    # should have equal lengths
    with pytest.raises(TypeError, match="list of lists-like"):
        idx.set_levels(levels[0], level=[0, 1])

    with pytest.raises(TypeError, match="list-like"):
        idx.set_levels(levels, level=0)

    # should have equal lengths
    with pytest.raises(TypeError, match="list of lists-like"):
        idx.set_codes(codes[0], level=[0, 1])

    with pytest.raises(TypeError, match="list-like"):
        idx.set_codes(codes, level=0)

    # should have equal lengths
    with pytest.raises(ValueError, match="Length of names"):
        idx.set_names(names[0], level=[0, 1])

    with pytest.raises(TypeError, match="Names must be a"):
        idx.set_names(names, level=0)


@pytest.mark.parametrize("inplace", [True, False])
def test_set_names_with_nlevel_1(inplace):
    # GH 21149
    # Ensure that .set_names for MultiIndex with
    # nlevels == 1 does not raise any errors
    expected = MultiIndex(levels=[[0, 1]], codes=[[0, 1]], names=["first"])
    m = MultiIndex.from_product([[0, 1]])
    result = m.set_names("first", level=0, inplace=inplace)

    if inplace:
        result = m

    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_set_levels_categorical(ordered):
    # GH13854
    index = MultiIndex.from_arrays([list("xyzx"), [0, 1, 2, 3]])

    cidx = CategoricalIndex(list("bac"), ordered=ordered)
    result = index.set_levels(cidx, level=0)
    expected = MultiIndex(levels=[cidx, [0, 1, 2, 3]], codes=index.codes)
    tm.assert_index_equal(result, expected)

    result_lvl = result.get_level_values(0)
    expected_lvl = CategoricalIndex(
        list("bacb"), categories=cidx.categories, ordered=cidx.ordered
    )
    tm.assert_index_equal(result_lvl, expected_lvl)


def test_set_value_keeps_names():
    # motivating example from #3742
    lev1 = ["hans", "hans", "hans", "grethe", "grethe", "grethe"]
    lev2 = ["1", "2", "3"] * 2
    idx = MultiIndex.from_arrays([lev1, lev2], names=["Name", "Number"])
    df = pd.DataFrame(
        np.random.default_rng(2).standard_normal((6, 4)),
        columns=["one", "two", "three", "four"],
        index=idx,
    )
    df = df.sort_index()
    assert df._is_copy is None
    assert df.index.names == ("Name", "Number")
    df.at[("grethe", "4"), "one"] = 99.34
    assert df._is_copy is None
    assert df.index.names == ("Name", "Number")


def test_set_levels_with_iterable():
    # GH23273
    sizes = [1, 2, 3]
    colors = ["black"] * 3
    index = MultiIndex.from_arrays([sizes, colors], names=["size", "color"])

    result = index.set_levels(map(int, ["3", "2", "1"]), level="size")

    expected_sizes = [3, 2, 1]
    expected = MultiIndex.from_arrays([expected_sizes, colors], names=["size", "color"])
    tm.assert_index_equal(result, expected)


def test_set_empty_level():
    # GH#48636
    midx = MultiIndex.from_arrays([[]], names=["A"])
    result = midx.set_levels(pd.DatetimeIndex([]), level=0)
    expected = MultiIndex.from_arrays([pd.DatetimeIndex([])], names=["A"])
    tm.assert_index_equal(result, expected)


def test_set_levels_pos_args_removal():
    # https://github.com/pandas-dev/pandas/issues/41485
    idx = MultiIndex.from_tuples(
        [
            (1, "one"),
            (3, "one"),
        ],
        names=["foo", "bar"],
    )
    with pytest.raises(TypeError, match="positional arguments"):
        idx.set_levels(["a", "b", "c"], 0)

    with pytest.raises(TypeError, match="positional arguments"):
        idx.set_codes([[0, 1], [1, 0]], 0)


def test_set_levels_categorical_keep_dtype():
    # GH#52125
    midx = MultiIndex.from_arrays([[5, 6]])
    result = midx.set_levels(levels=pd.Categorical([1, 2]), level=0)
    expected = MultiIndex.from_arrays([pd.Categorical([1, 2])])
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_wtf\_compat.py ===
import warnings


class FlaskWTFDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter("always", FlaskWTFDeprecationWarning)
warnings.filterwarnings(
    "ignore", category=FlaskWTFDeprecationWarning, module="wtforms|flask_wtf"
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\version.py ===

"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.3.0"
__version__ = version
full_version = version

git_revision = "0532af47d6a815298b7841de00bdbc547104b237"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\logger.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging


def get_logger(name, level=logging.DEBUG):
    logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\__init__.py ===
# Copyright (c) 2010-2024 openpyxl


from .alignment import Alignment
from .borders import Border, Side
from .colors import Color
from .fills import PatternFill, GradientFill, Fill
from .fonts import Font, DEFAULT_FONT
from .numbers import NumberFormatDescriptor, is_date_format, is_builtin
from .protection import Protection
from .named_styles import NamedStyle

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\api\test_api.py ===
from __future__ import annotations

import pytest

import pandas as pd
from pandas import api
import pandas._testing as tm
from pandas.api import (
    extensions as api_extensions,
    indexers as api_indexers,
    interchange as api_interchange,
    types as api_types,
    typing as api_typing,
)


class Base:
    def check(self, namespace, expected, ignored=None):
        # see which names are in the namespace, minus optional
        # ignored ones
        # compare vs the expected

        result = sorted(
            f for f in dir(namespace) if not f.startswith("__") and f != "annotations"
        )
        if ignored is not None:
            result = sorted(set(result) - set(ignored))

        expected = sorted(expected)
        tm.assert_almost_equal(result, expected)


class TestPDApi(Base):
    # these are optionally imported based on testing
    # & need to be ignored
    ignored = ["tests", "locale", "conftest", "_version_meson"]

    # top-level sub-packages
    public_lib = [
        "api",
        "arrays",
        "options",
        "test",
        "testing",
        "errors",
        "plotting",
        "io",
        "tseries",
    ]
    private_lib = ["compat", "core", "pandas", "util", "_built_with_meson"]

    # misc
    misc = ["IndexSlice", "NaT", "NA"]

    # top-level classes
    classes = [
        "ArrowDtype",
        "Categorical",
        "CategoricalIndex",
        "DataFrame",
        "DateOffset",
        "DatetimeIndex",
        "ExcelFile",
        "ExcelWriter",
        "Flags",
        "Grouper",
        "HDFStore",
        "Index",
        "MultiIndex",
        "Period",
        "PeriodIndex",
        "RangeIndex",
        "Series",
        "SparseDtype",
        "StringDtype",
        "Timedelta",
        "TimedeltaIndex",
        "Timestamp",
        "Interval",
        "IntervalIndex",
        "CategoricalDtype",
        "PeriodDtype",
        "IntervalDtype",
        "DatetimeTZDtype",
        "BooleanDtype",
        "Int8Dtype",
        "Int16Dtype",
        "Int32Dtype",
        "Int64Dtype",
        "UInt8Dtype",
        "UInt16Dtype",
        "UInt32Dtype",
        "UInt64Dtype",
        "Float32Dtype",
        "Float64Dtype",
        "NamedAgg",
    ]

    # these are already deprecated; awaiting removal
    deprecated_classes: list[str] = []

    # external modules exposed in pandas namespace
    modules: list[str] = []

    # top-level functions
    funcs = [
        "array",
        "bdate_range",
        "concat",
        "crosstab",
        "cut",
        "date_range",
        "interval_range",
        "eval",
        "factorize",
        "get_dummies",
        "from_dummies",
        "infer_freq",
        "isna",
        "isnull",
        "lreshape",
        "melt",
        "notna",
        "notnull",
        "offsets",
        "merge",
        "merge_ordered",
        "merge_asof",
        "period_range",
        "pivot",
        "pivot_table",
        "qcut",
        "show_versions",
        "timedelta_range",
        "unique",
        "value_counts",
        "wide_to_long",
    ]

    # top-level option funcs
    funcs_option = [
        "reset_option",
        "describe_option",
        "get_option",
        "option_context",
        "set_option",
        "set_eng_float_format",
    ]

    # top-level read_* funcs
    funcs_read = [
        "read_clipboard",
        "read_csv",
        "read_excel",
        "read_fwf",
        "read_gbq",
        "read_hdf",
        "read_html",
        "read_xml",
        "read_json",
        "read_pickle",
        "read_sas",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "read_stata",
        "read_table",
        "read_feather",
        "read_parquet",
        "read_orc",
        "read_spss",
    ]

    # top-level json funcs
    funcs_json = ["json_normalize"]

    # top-level to_* funcs
    funcs_to = ["to_datetime", "to_numeric", "to_pickle", "to_timedelta"]

    # top-level to deprecate in the future
    deprecated_funcs_in_future: list[str] = []

    # these are already deprecated; awaiting removal
    deprecated_funcs: list[str] = []

    # private modules in pandas namespace
    private_modules = [
        "_config",
        "_libs",
        "_is_numpy_dev",
        "_pandas_datetime_CAPI",
        "_pandas_parser_CAPI",
        "_testing",
        "_typing",
    ]
    if not pd._built_with_meson:
        private_modules.append("_version")

    def test_api(self):
        checkthese = (
            self.public_lib
            + self.private_lib
            + self.misc
            + self.modules
            + self.classes
            + self.funcs
            + self.funcs_option
            + self.funcs_read
            + self.funcs_json
            + self.funcs_to
            + self.private_modules
        )
        self.check(namespace=pd, expected=checkthese, ignored=self.ignored)

    def test_api_all(self):
        expected = set(
            self.public_lib
            + self.misc
            + self.modules
            + self.classes
            + self.funcs
            + self.funcs_option
            + self.funcs_read
            + self.funcs_json
            + self.funcs_to
        ) - set(self.deprecated_classes)
        actual = set(pd.__all__)

        extraneous = actual - expected
        assert not extraneous

        missing = expected - actual
        assert not missing

    def test_depr(self):
        deprecated_list = (
            self.deprecated_classes
            + self.deprecated_funcs
            + self.deprecated_funcs_in_future
        )
        for depr in deprecated_list:
            with tm.assert_produces_warning(FutureWarning):
                _ = getattr(pd, depr)


class TestApi(Base):
    allowed_api_dirs = [
        "types",
        "extensions",
        "indexers",
        "interchange",
        "typing",
    ]
    allowed_typing = [
        "DataFrameGroupBy",
        "DatetimeIndexResamplerGroupby",
        "Expanding",
        "ExpandingGroupby",
        "ExponentialMovingWindow",
        "ExponentialMovingWindowGroupby",
        "JsonReader",
        "NaTType",
        "NAType",
        "PeriodIndexResamplerGroupby",
        "Resampler",
        "Rolling",
        "RollingGroupby",
        "SeriesGroupBy",
        "StataReader",
        "TimedeltaIndexResamplerGroupby",
        "TimeGrouper",
        "Window",
    ]
    allowed_api_types = [
        "is_any_real_numeric_dtype",
        "is_array_like",
        "is_bool",
        "is_bool_dtype",
        "is_categorical_dtype",
        "is_complex",
        "is_complex_dtype",
        "is_datetime64_any_dtype",
        "is_datetime64_dtype",
        "is_datetime64_ns_dtype",
        "is_datetime64tz_dtype",
        "is_dict_like",
        "is_dtype_equal",
        "is_extension_array_dtype",
        "is_file_like",
        "is_float",
        "is_float_dtype",
        "is_hashable",
        "is_int64_dtype",
        "is_integer",
        "is_integer_dtype",
        "is_interval",
        "is_interval_dtype",
        "is_iterator",
        "is_list_like",
        "is_named_tuple",
        "is_number",
        "is_numeric_dtype",
        "is_object_dtype",
        "is_period_dtype",
        "is_re",
        "is_re_compilable",
        "is_scalar",
        "is_signed_integer_dtype",
        "is_sparse",
        "is_string_dtype",
        "is_timedelta64_dtype",
        "is_timedelta64_ns_dtype",
        "is_unsigned_integer_dtype",
        "pandas_dtype",
        "infer_dtype",
        "union_categoricals",
        "CategoricalDtype",
        "DatetimeTZDtype",
        "IntervalDtype",
        "PeriodDtype",
    ]
    allowed_api_interchange = ["from_dataframe", "DataFrame"]
    allowed_api_indexers = [
        "check_array_indexer",
        "BaseIndexer",
        "FixedForwardWindowIndexer",
        "VariableOffsetWindowIndexer",
    ]
    allowed_api_extensions = [
        "no_default",
        "ExtensionDtype",
        "register_extension_dtype",
        "register_dataframe_accessor",
        "register_index_accessor",
        "register_series_accessor",
        "take",
        "ExtensionArray",
        "ExtensionScalarOpsMixin",
    ]

    def test_api(self):
        self.check(api, self.allowed_api_dirs)

    def test_api_typing(self):
        self.check(api_typing, self.allowed_typing)

    def test_api_types(self):
        self.check(api_types, self.allowed_api_types)

    def test_api_interchange(self):
        self.check(api_interchange, self.allowed_api_interchange)

    def test_api_indexers(self):
        self.check(api_indexers, self.allowed_api_indexers)

    def test_api_extensions(self):
        self.check(api_extensions, self.allowed_api_extensions)


class TestTesting(Base):
    funcs = [
        "assert_frame_equal",
        "assert_series_equal",
        "assert_index_equal",
        "assert_extension_array_equal",
    ]

    def test_testing(self):
        from pandas import testing

        self.check(testing, self.funcs)

    def test_util_in_top_level(self):
        with pytest.raises(AttributeError, match="foo"):
            pd.util.foo


def test_pandas_array_alias():
    msg = "PandasArray has been renamed NumpyExtensionArray"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = pd.arrays.PandasArray

    assert res is pd.arrays.NumpyExtensionArray

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\test_round.py ===
from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest
import pytz

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    OutOfBoundsDatetime,
    Timedelta,
    Timestamp,
    iNaT,
    to_offset,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

import pandas._testing as tm


class TestTimestampRound:
    def test_round_division_by_zero_raises(self):
        ts = Timestamp("2016-01-01")

        msg = "Division by zero in rounding"
        with pytest.raises(ValueError, match=msg):
            ts.round("0ns")

    @pytest.mark.parametrize(
        "timestamp, freq, expected",
        [
            ("20130101 09:10:11", "D", "20130101"),
            ("20130101 19:10:11", "D", "20130102"),
            ("20130201 12:00:00", "D", "20130202"),
            ("20130104 12:00:00", "D", "20130105"),
            ("2000-01-05 05:09:15.13", "D", "2000-01-05 00:00:00"),
            ("2000-01-05 05:09:15.13", "h", "2000-01-05 05:00:00"),
            ("2000-01-05 05:09:15.13", "s", "2000-01-05 05:09:15"),
        ],
    )
    def test_round_frequencies(self, timestamp, freq, expected):
        dt = Timestamp(timestamp)
        result = dt.round(freq)
        expected = Timestamp(expected)
        assert result == expected

    def test_round_tzaware(self):
        dt = Timestamp("20130101 09:10:11", tz="US/Eastern")
        result = dt.round("D")
        expected = Timestamp("20130101", tz="US/Eastern")
        assert result == expected

        dt = Timestamp("20130101 09:10:11", tz="US/Eastern")
        result = dt.round("s")
        assert result == dt

    def test_round_30min(self):
        # round
        dt = Timestamp("20130104 12:32:00")
        result = dt.round("30Min")
        expected = Timestamp("20130104 12:30:00")
        assert result == expected

    def test_round_subsecond(self):
        # GH#14440 & GH#15578
        result = Timestamp("2016-10-17 12:00:00.0015").round("ms")
        expected = Timestamp("2016-10-17 12:00:00.002000")
        assert result == expected

        result = Timestamp("2016-10-17 12:00:00.00149").round("ms")
        expected = Timestamp("2016-10-17 12:00:00.001000")
        assert result == expected

        ts = Timestamp("2016-10-17 12:00:00.0015")
        for freq in ["us", "ns"]:
            assert ts == ts.round(freq)

        result = Timestamp("2016-10-17 12:00:00.001501031").round("10ns")
        expected = Timestamp("2016-10-17 12:00:00.001501030")
        assert result == expected

    def test_round_nonstandard_freq(self):
        with tm.assert_produces_warning(False):
            Timestamp("2016-10-17 12:00:00.001501031").round("1010ns")

    def test_round_invalid_arg(self):
        stamp = Timestamp("2000-01-05 05:09:15.13")
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            stamp.round("foo")

    @pytest.mark.parametrize(
        "test_input, rounder, freq, expected",
        [
            ("2117-01-01 00:00:45", "floor", "15s", "2117-01-01 00:00:45"),
            ("2117-01-01 00:00:45", "ceil", "15s", "2117-01-01 00:00:45"),
            (
                "2117-01-01 00:00:45.000000012",
                "floor",
                "10ns",
                "2117-01-01 00:00:45.000000010",
            ),
            (
                "1823-01-01 00:00:01.000000012",
                "ceil",
                "10ns",
                "1823-01-01 00:00:01.000000020",
            ),
            ("1823-01-01 00:00:01", "floor", "1s", "1823-01-01 00:00:01"),
            ("1823-01-01 00:00:01", "ceil", "1s", "1823-01-01 00:00:01"),
            ("NaT", "floor", "1s", "NaT"),
            ("NaT", "ceil", "1s", "NaT"),
        ],
    )
    def test_ceil_floor_edge(self, test_input, rounder, freq, expected):
        dt = Timestamp(test_input)
        func = getattr(dt, rounder)
        result = func(freq)

        if dt is NaT:
            assert result is NaT
        else:
            expected = Timestamp(expected)
            assert result == expected

    @pytest.mark.parametrize(
        "test_input, freq, expected",
        [
            ("2018-01-01 00:02:06", "2s", "2018-01-01 00:02:06"),
            ("2018-01-01 00:02:00", "2min", "2018-01-01 00:02:00"),
            ("2018-01-01 00:04:00", "4min", "2018-01-01 00:04:00"),
            ("2018-01-01 00:15:00", "15min", "2018-01-01 00:15:00"),
            ("2018-01-01 00:20:00", "20min", "2018-01-01 00:20:00"),
            ("2018-01-01 03:00:00", "3h", "2018-01-01 03:00:00"),
        ],
    )
    @pytest.mark.parametrize("rounder", ["ceil", "floor", "round"])
    def test_round_minute_freq(self, test_input, freq, expected, rounder):
        # Ensure timestamps that shouldn't round dont!
        # GH#21262

        dt = Timestamp(test_input)
        expected = Timestamp(expected)
        func = getattr(dt, rounder)
        result = func(freq)
        assert result == expected

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_ceil(self, unit):
        dt = Timestamp("20130101 09:10:11").as_unit(unit)
        result = dt.ceil("D")
        expected = Timestamp("20130102")
        assert result == expected
        assert result._creso == dt._creso

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_floor(self, unit):
        dt = Timestamp("20130101 09:10:11").as_unit(unit)
        result = dt.floor("D")
        expected = Timestamp("20130101")
        assert result == expected
        assert result._creso == dt._creso

    @pytest.mark.parametrize("method", ["ceil", "round", "floor"])
    @pytest.mark.parametrize(
        "unit",
        ["ns", "us", "ms", "s"],
    )
    def test_round_dst_border_ambiguous(self, method, unit):
        # GH 18946 round near "fall back" DST
        ts = Timestamp("2017-10-29 00:00:00", tz="UTC").tz_convert("Europe/Madrid")
        ts = ts.as_unit(unit)
        #
        result = getattr(ts, method)("h", ambiguous=True)
        assert result == ts
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        result = getattr(ts, method)("h", ambiguous=False)
        expected = Timestamp("2017-10-29 01:00:00", tz="UTC").tz_convert(
            "Europe/Madrid"
        )
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        result = getattr(ts, method)("h", ambiguous="NaT")
        assert result is NaT

        msg = "Cannot infer dst time"
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            getattr(ts, method)("h", ambiguous="raise")

    @pytest.mark.parametrize(
        "method, ts_str, freq",
        [
            ["ceil", "2018-03-11 01:59:00-0600", "5min"],
            ["round", "2018-03-11 01:59:00-0600", "5min"],
            ["floor", "2018-03-11 03:01:00-0500", "2h"],
        ],
    )
    @pytest.mark.parametrize(
        "unit",
        ["ns", "us", "ms", "s"],
    )
    def test_round_dst_border_nonexistent(self, method, ts_str, freq, unit):
        # GH 23324 round near "spring forward" DST
        ts = Timestamp(ts_str, tz="America/Chicago").as_unit(unit)
        result = getattr(ts, method)(freq, nonexistent="shift_forward")
        expected = Timestamp("2018-03-11 03:00:00", tz="America/Chicago")
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        result = getattr(ts, method)(freq, nonexistent="NaT")
        assert result is NaT

        msg = "2018-03-11 02:00:00"
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            getattr(ts, method)(freq, nonexistent="raise")

    @pytest.mark.parametrize(
        "timestamp",
        [
            "2018-01-01 0:0:0.124999360",
            "2018-01-01 0:0:0.125000367",
            "2018-01-01 0:0:0.125500",
            "2018-01-01 0:0:0.126500",
            "2018-01-01 12:00:00",
            "2019-01-01 12:00:00",
        ],
    )
    @pytest.mark.parametrize(
        "freq",
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
            "1D",
        ],
    )
    def test_round_int64(self, timestamp, freq):
        # check that all rounding modes are accurate to int64 precision
        # see GH#22591
        dt = Timestamp(timestamp).as_unit("ns")
        unit = to_offset(freq).nanos

        # test floor
        result = dt.floor(freq)
        assert result._value % unit == 0, f"floor not a {freq} multiple"
        assert 0 <= dt._value - result._value < unit, "floor error"

        # test ceil
        result = dt.ceil(freq)
        assert result._value % unit == 0, f"ceil not a {freq} multiple"
        assert 0 <= result._value - dt._value < unit, "ceil error"

        # test round
        result = dt.round(freq)
        assert result._value % unit == 0, f"round not a {freq} multiple"
        assert abs(result._value - dt._value) <= unit // 2, "round error"
        if unit % 2 == 0 and abs(result._value - dt._value) == unit // 2:
            # round half to even
            assert result._value // unit % 2 == 0, "round half to even error"

    def test_round_implementation_bounds(self):
        # See also: analogous test for Timedelta
        result = Timestamp.min.ceil("s")
        expected = Timestamp(1677, 9, 21, 0, 12, 44)
        assert result == expected

        result = Timestamp.max.floor("s")
        expected = Timestamp.max - Timedelta(854775807)
        assert result == expected

        msg = "Cannot round 1677-09-21 00:12:43.145224193 to freq=<Second>"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.floor("s")

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.round("s")

        msg = "Cannot round 2262-04-11 23:47:16.854775807 to freq=<Second>"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.ceil("s")

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.round("s")

    @given(val=st.integers(iNaT + 1, lib.i8max))
    @pytest.mark.parametrize(
        "method", [Timestamp.round, Timestamp.floor, Timestamp.ceil]
    )
    def test_round_sanity(self, val, method):
        cls = Timestamp
        err_cls = OutOfBoundsDatetime

        val = np.int64(val)
        ts = cls(val)

        def checker(ts, nanos, unit):
            # First check that we do raise in cases where we should
            if nanos == 1:
                pass
            else:
                div, mod = divmod(ts._value, nanos)
                diff = int(nanos - mod)
                lb = ts._value - mod
                assert lb <= ts._value  # i.e. no overflows with python ints
                ub = ts._value + diff
                assert ub > ts._value  # i.e. no overflows with python ints

                msg = "without overflow"
                if mod == 0:
                    # We should never be raising in this
                    pass
                elif method is cls.ceil:
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif method is cls.floor:
                    if lb < cls.min._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif mod >= diff:
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif lb < cls.min._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return

            res = method(ts, unit)

            td = res - ts
            diff = abs(td._value)
            assert diff < nanos
            assert res._value % nanos == 0

            if method is cls.round:
                assert diff <= nanos / 2
            elif method is cls.floor:
                assert res <= ts
            elif method is cls.ceil:
                assert res >= ts

        nanos = 1
        checker(ts, nanos, "ns")

        nanos = 1000
        checker(ts, nanos, "us")

        nanos = 1_000_000
        checker(ts, nanos, "ms")

        nanos = 1_000_000_000
        checker(ts, nanos, "s")

        nanos = 60 * 1_000_000_000
        checker(ts, nanos, "min")

        nanos = 60 * 60 * 1_000_000_000
        checker(ts, nanos, "h")

        nanos = 24 * 60 * 60 * 1_000_000_000
        checker(ts, nanos, "D")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\__init__.py ===
# ext/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from .. import util as _sa_util


_sa_util.preloaded.import_prefix("sqlalchemy.ext")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\pynodes.py ===
from .abstract_nodes import List as AbstractList
from .ast import Token


class List(AbstractList):
    pass


class NumExprEvaluate(Token):
    """represents a call to :class:`numexpr`s :func:`evaluate`"""
    __slots__ = _fields = ('expr',)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\benchmarks\bench_sympify.py ===
from sympy.core import sympify, Symbol

x = Symbol('x')


def timeit_sympify_1():
    sympify(1)


def timeit_sympify_x():
    sympify(x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\benchmarks\bench_exp.py ===
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp

x, y = symbols('x,y')

e = exp(2*x)
q = exp(3*x)


def timeit_exp_subs():
    e.subs(q, y)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\multipledispatch\__init__.py ===
from .core import dispatch
from .dispatcher import (Dispatcher, halt_ordering, restart_ordering,
    MDNotImplementedError)

__version__ = '0.4.9'

__all__ = [
    'dispatch',

    'Dispatcher', 'halt_ordering', 'restart_ordering', 'MDNotImplementedError',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_subprocess_platform\windows.py ===
from typing import TYPE_CHECKING

from .._wait_for_object import WaitForSingleObject

if TYPE_CHECKING:
    from .. import _subprocess


async def wait_child_exiting(process: "_subprocess.Process") -> None:
    # _handle is not in Popen stubs, though it is present on Windows.
    await WaitForSingleObject(int(process._proc._handle))  # type: ignore[attr-defined]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_constructors.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Period,
    PeriodIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

# -----------------------------------------------------------------------------
# Copy/view behaviour for Series / DataFrame constructors


@pytest.mark.parametrize("dtype", [None, "int64"])
def test_series_from_series(dtype, using_copy_on_write, warn_copy_on_write):
    # Case: constructing a Series from another Series object follows CoW rules:
    # a new object is returned and thus mutations are not propagated
    ser = Series([1, 2, 3], name="name")

    # default is copy=False -> new Series is a shallow copy / view of original
    result = Series(ser, dtype=dtype)

    # the shallow copy still shares memory
    assert np.shares_memory(get_array(ser), get_array(result))

    if using_copy_on_write:
        assert result._mgr.blocks[0].refs.has_reference()

    if using_copy_on_write:
        # mutating new series copy doesn't mutate original
        result.iloc[0] = 0
        assert ser.iloc[0] == 1
        # mutating triggered a copy-on-write -> no longer shares memory
        assert not np.shares_memory(get_array(ser), get_array(result))
    else:
        # mutating shallow copy does mutate original
        with tm.assert_cow_warning(warn_copy_on_write):
            result.iloc[0] = 0
        assert ser.iloc[0] == 0
        # and still shares memory
        assert np.shares_memory(get_array(ser), get_array(result))

    # the same when modifying the parent
    result = Series(ser, dtype=dtype)

    if using_copy_on_write:
        # mutating original doesn't mutate new series
        ser.iloc[0] = 0
        assert result.iloc[0] == 1
    else:
        # mutating original does mutate shallow copy
        with tm.assert_cow_warning(warn_copy_on_write):
            ser.iloc[0] = 0
        assert result.iloc[0] == 0


def test_series_from_series_with_reindex(using_copy_on_write, warn_copy_on_write):
    # Case: constructing a Series from another Series with specifying an index
    # that potentially requires a reindex of the values
    ser = Series([1, 2, 3], name="name")

    # passing an index that doesn't actually require a reindex of the values
    # -> without CoW we get an actual mutating view
    for index in [
        ser.index,
        ser.index.copy(),
        list(ser.index),
        ser.index.rename("idx"),
    ]:
        result = Series(ser, index=index)
        assert np.shares_memory(ser.values, result.values)
        with tm.assert_cow_warning(warn_copy_on_write):
            result.iloc[0] = 0
        if using_copy_on_write:
            assert ser.iloc[0] == 1
        else:
            assert ser.iloc[0] == 0

    # ensure that if an actual reindex is needed, we don't have any refs
    # (mutating the result wouldn't trigger CoW)
    result = Series(ser, index=[0, 1, 2, 3])
    assert not np.shares_memory(ser.values, result.values)
    if using_copy_on_write:
        assert not result._mgr.blocks[0].refs.has_reference()


@pytest.mark.parametrize("fastpath", [False, True])
@pytest.mark.parametrize("dtype", [None, "int64"])
@pytest.mark.parametrize("idx", [None, pd.RangeIndex(start=0, stop=3, step=1)])
@pytest.mark.parametrize(
    "arr", [np.array([1, 2, 3], dtype="int64"), pd.array([1, 2, 3], dtype="Int64")]
)
def test_series_from_array(using_copy_on_write, idx, dtype, fastpath, arr):
    if idx is None or dtype is not None:
        fastpath = False
    msg = "The 'fastpath' keyword in pd.Series is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser = Series(arr, dtype=dtype, index=idx, fastpath=fastpath)
    ser_orig = ser.copy()
    data = getattr(arr, "_data", arr)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(ser), data)
    else:
        assert np.shares_memory(get_array(ser), data)

    arr[0] = 100
    if using_copy_on_write:
        tm.assert_series_equal(ser, ser_orig)
    else:
        expected = Series([100, 2, 3], dtype=dtype if dtype is not None else arr.dtype)
        tm.assert_series_equal(ser, expected)


@pytest.mark.parametrize("copy", [True, False, None])
def test_series_from_array_different_dtype(using_copy_on_write, copy):
    arr = np.array([1, 2, 3], dtype="int64")
    ser = Series(arr, dtype="int32", copy=copy)
    assert not np.shares_memory(get_array(ser), arr)


@pytest.mark.parametrize(
    "idx",
    [
        Index([1, 2]),
        DatetimeIndex([Timestamp("2019-12-31"), Timestamp("2020-12-31")]),
        PeriodIndex([Period("2019-12-31"), Period("2020-12-31")]),
        TimedeltaIndex([Timedelta("1 days"), Timedelta("2 days")]),
    ],
)
def test_series_from_index(using_copy_on_write, idx):
    ser = Series(idx)
    expected = idx.copy(deep=True)
    if using_copy_on_write:
        assert np.shares_memory(get_array(ser), get_array(idx))
        assert not ser._mgr._has_no_reference(0)
    else:
        assert not np.shares_memory(get_array(ser), get_array(idx))
    ser.iloc[0] = ser.iloc[1]
    tm.assert_index_equal(idx, expected)


def test_series_from_index_different_dtypes(using_copy_on_write):
    idx = Index([1, 2, 3], dtype="int64")
    ser = Series(idx, dtype="int32")
    assert not np.shares_memory(get_array(ser), get_array(idx))
    if using_copy_on_write:
        assert ser._mgr._has_no_reference(0)


@pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
@pytest.mark.parametrize("fastpath", [False, True])
@pytest.mark.parametrize("dtype", [None, "int64"])
@pytest.mark.parametrize("idx", [None, pd.RangeIndex(start=0, stop=3, step=1)])
def test_series_from_block_manager(using_copy_on_write, idx, dtype, fastpath):
    ser = Series([1, 2, 3], dtype="int64")
    ser_orig = ser.copy()
    msg = "The 'fastpath' keyword in pd.Series is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser2 = Series(ser._mgr, dtype=dtype, fastpath=fastpath, index=idx)
    assert np.shares_memory(get_array(ser), get_array(ser2))
    if using_copy_on_write:
        assert not ser2._mgr._has_no_reference(0)

    ser2.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_series_equal(ser, ser_orig)
    else:
        expected = Series([100, 2, 3])
        tm.assert_series_equal(ser, expected)


def test_series_from_block_manager_different_dtype(using_copy_on_write):
    ser = Series([1, 2, 3], dtype="int64")
    msg = "Passing a SingleBlockManager to Series"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser2 = Series(ser._mgr, dtype="int32")
    assert not np.shares_memory(get_array(ser), get_array(ser2))
    if using_copy_on_write:
        assert ser2._mgr._has_no_reference(0)


@pytest.mark.parametrize("use_mgr", [True, False])
@pytest.mark.parametrize("columns", [None, ["a"]])
def test_dataframe_constructor_mgr_or_df(
    using_copy_on_write, warn_copy_on_write, columns, use_mgr
):
    df = DataFrame({"a": [1, 2, 3]})
    df_orig = df.copy()

    if use_mgr:
        data = df._mgr
        warn = DeprecationWarning
    else:
        data = df
        warn = None
    msg = "Passing a BlockManager to DataFrame"
    with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
        new_df = DataFrame(data)

    assert np.shares_memory(get_array(df, "a"), get_array(new_df, "a"))
    with tm.assert_cow_warning(warn_copy_on_write and not use_mgr):
        new_df.iloc[0] = 100

    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "a"), get_array(new_df, "a"))
        tm.assert_frame_equal(df, df_orig)
    else:
        assert np.shares_memory(get_array(df, "a"), get_array(new_df, "a"))
        tm.assert_frame_equal(df, new_df)


@pytest.mark.parametrize("dtype", [None, "int64", "Int64"])
@pytest.mark.parametrize("index", [None, [0, 1, 2]])
@pytest.mark.parametrize("columns", [None, ["a", "b"], ["a", "b", "c"]])
def test_dataframe_from_dict_of_series(
    request, using_copy_on_write, warn_copy_on_write, columns, index, dtype
):
    # Case: constructing a DataFrame from Series objects with copy=False
    # has to do a lazy following CoW rules
    # (the default for DataFrame(dict) is still to copy to ensure consolidation)
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    s1_orig = s1.copy()
    expected = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, index=index, columns=columns, dtype=dtype
    )

    result = DataFrame(
        {"a": s1, "b": s2}, index=index, columns=columns, dtype=dtype, copy=False
    )

    # the shallow copy still shares memory
    assert np.shares_memory(get_array(result, "a"), get_array(s1))

    # mutating the new dataframe doesn't mutate original
    with tm.assert_cow_warning(warn_copy_on_write):
        result.iloc[0, 0] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(s1))
        tm.assert_series_equal(s1, s1_orig)
    else:
        assert s1.iloc[0] == 10

    # the same when modifying the parent series
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    result = DataFrame(
        {"a": s1, "b": s2}, index=index, columns=columns, dtype=dtype, copy=False
    )
    with tm.assert_cow_warning(warn_copy_on_write):
        s1.iloc[0] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, "a"), get_array(s1))
        tm.assert_frame_equal(result, expected)
    else:
        assert result.iloc[0, 0] == 10


@pytest.mark.parametrize("dtype", [None, "int64"])
def test_dataframe_from_dict_of_series_with_reindex(dtype):
    # Case: constructing a DataFrame from Series objects with copy=False
    # and passing an index that requires an actual (no-view) reindex -> need
    # to ensure the result doesn't have refs set up to unnecessarily trigger
    # a copy on write
    s1 = Series([1, 2, 3])
    s2 = Series([4, 5, 6])
    df = DataFrame({"a": s1, "b": s2}, index=[1, 2, 3], dtype=dtype, copy=False)

    # df should own its memory, so mutating shouldn't trigger a copy
    arr_before = get_array(df, "a")
    assert not np.shares_memory(arr_before, get_array(s1))
    df.iloc[0, 0] = 100
    arr_after = get_array(df, "a")
    assert np.shares_memory(arr_before, arr_after)


@pytest.mark.parametrize("cons", [Series, Index])
@pytest.mark.parametrize(
    "data, dtype", [([1, 2], None), ([1, 2], "int64"), (["a", "b"], object)]
)
def test_dataframe_from_series_or_index(
    using_copy_on_write, warn_copy_on_write, data, dtype, cons
):
    obj = cons(data, dtype=dtype)
    obj_orig = obj.copy()
    df = DataFrame(obj, dtype=dtype)
    assert np.shares_memory(get_array(obj), get_array(df, 0))
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)

    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = data[-1]
    if using_copy_on_write:
        tm.assert_equal(obj, obj_orig)


@pytest.mark.parametrize("cons", [Series, Index])
def test_dataframe_from_series_or_index_different_dtype(using_copy_on_write, cons):
    obj = cons([1, 2], dtype="int64")
    df = DataFrame(obj, dtype="int32")
    assert not np.shares_memory(get_array(obj), get_array(df, 0))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)


def test_dataframe_from_series_infer_datetime(using_copy_on_write):
    ser = Series([Timestamp("2019-12-31"), Timestamp("2020-12-31")], dtype=object)
    with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
        df = DataFrame(ser)
    assert not np.shares_memory(get_array(ser), get_array(df, 0))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)


@pytest.mark.parametrize("index", [None, [0, 1, 2]])
def test_dataframe_from_dict_of_series_with_dtype(index):
    # Variant of above, but now passing a dtype that causes a copy
    # -> need to ensure the result doesn't have refs set up to unnecessarily
    # trigger a copy on write
    s1 = Series([1.0, 2.0, 3.0])
    s2 = Series([4, 5, 6])
    df = DataFrame({"a": s1, "b": s2}, index=index, dtype="int64", copy=False)

    # df should own its memory, so mutating shouldn't trigger a copy
    arr_before = get_array(df, "a")
    assert not np.shares_memory(arr_before, get_array(s1))
    df.iloc[0, 0] = 100
    arr_after = get_array(df, "a")
    assert np.shares_memory(arr_before, arr_after)


@pytest.mark.parametrize("copy", [False, None, True])
def test_frame_from_numpy_array(using_copy_on_write, copy, using_array_manager):
    arr = np.array([[1, 2], [3, 4]])
    df = DataFrame(arr, copy=copy)

    if (
        using_copy_on_write
        and copy is not False
        or copy is True
        or (using_array_manager and copy is None)
    ):
        assert not np.shares_memory(get_array(df, 0), arr)
    else:
        assert np.shares_memory(get_array(df, 0), arr)


def test_dataframe_from_records_with_dataframe(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3]})
    df_orig = df.copy()
    with tm.assert_produces_warning(FutureWarning):
        df2 = DataFrame.from_records(df)
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    with tm.assert_cow_warning(warn_copy_on_write):
        df2.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        tm.assert_frame_equal(df, df2)


def test_frame_from_dict_of_index(using_copy_on_write):
    idx = Index([1, 2, 3])
    expected = idx.copy(deep=True)
    df = DataFrame({"a": idx}, copy=False)
    assert np.shares_memory(get_array(df, "a"), idx._values)
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)

        df.iloc[0, 0] = 100
        tm.assert_index_equal(idx, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_chunksize.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import StringIO

import numpy as np
import pytest

from pandas._libs import parsers as libparsers
from pandas.errors import DtypeWarning

from pandas import (
    DataFrame,
    concat,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.mark.parametrize("index_col", [0, "index"])
def test_read_chunksize_with_index(all_parsers, index_col):
    parser = all_parsers
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    expected = DataFrame(
        [
            ["foo", 2, 3, 4, 5],
            ["bar", 7, 8, 9, 10],
            ["baz", 12, 13, 14, 15],
            ["qux", 12, 13, 14, 15],
            ["foo2", 12, 13, 14, 15],
            ["bar2", 12, 13, 14, 15],
        ],
        columns=["index", "A", "B", "C", "D"],
    )
    expected = expected.set_index("index")

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
                list(reader)
        return

    with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
        chunks = list(reader)
    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])


@pytest.mark.parametrize("chunksize", [1.3, "foo", 0])
def test_read_chunksize_bad(all_parsers, chunksize):
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    msg = r"'chunksize' must be an integer >=1"
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"

    with pytest.raises(ValueError, match=msg):
        with parser.read_csv(StringIO(data), chunksize=chunksize) as _:
            pass


@pytest.mark.parametrize("chunksize", [2, 8])
def test_read_chunksize_and_nrows(all_parsers, chunksize):
    # see gh-15755
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0, "nrows": 5}

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
        return

    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=chunksize, **kwargs) as reader:
        tm.assert_frame_equal(concat(reader), expected)


def test_read_chunksize_and_nrows_changing_size(all_parsers):
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0, "nrows": 5}

    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
        return

    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=8, **kwargs) as reader:
        tm.assert_frame_equal(reader.get_chunk(size=2), expected.iloc[:2])
        tm.assert_frame_equal(reader.get_chunk(size=4), expected.iloc[2:5])

        with pytest.raises(StopIteration, match=""):
            reader.get_chunk(size=3)


def test_get_chunk_passed_chunksize(all_parsers):
    parser = all_parsers
    data = """A,B,C
1,2,3
4,5,6
7,8,9
1,2,3"""

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), chunksize=2) as reader:
                reader.get_chunk()
        return

    with parser.read_csv(StringIO(data), chunksize=2) as reader:
        result = reader.get_chunk()

    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{}, {"index_col": 0}])
def test_read_chunksize_compat(all_parsers, kwargs):
    # see gh-12185
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
                concat(reader)
        return

    with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
        via_reader = concat(reader)
    tm.assert_frame_equal(via_reader, result)


def test_read_chunksize_jagged_names(all_parsers):
    # see gh-23509
    parser = all_parsers
    data = "\n".join(["0"] * 7 + [",".join(["0"] * 10)])

    expected = DataFrame([[0] + [np.nan] * 9] * 7 + [[0] * 10])

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(
                StringIO(data), names=range(10), chunksize=4
            ) as reader:
                concat(reader)
        return

    with parser.read_csv(StringIO(data), names=range(10), chunksize=4) as reader:
        result = concat(reader)
    tm.assert_frame_equal(result, expected)


def test_chunk_begins_with_newline_whitespace(all_parsers):
    # see gh-10022
    parser = all_parsers
    data = "\n hello\nworld\n"

    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([" hello", "world"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
def test_chunks_have_consistent_numerical_type(all_parsers, monkeypatch):
    # mainly an issue with the C parser
    heuristic = 2**3
    parser = all_parsers
    integers = [str(i) for i in range(heuristic - 1)]
    data = "a\n" + "\n".join(integers + ["1.0", "2.0"] + integers)

    # Coercions should work without warnings.
    with monkeypatch.context() as m:
        m.setattr(libparsers, "DEFAULT_BUFFER_HEURISTIC", heuristic)
        result = parser.read_csv(StringIO(data))

    assert type(result.a[0]) is np.float64
    assert result.a.dtype == float


def test_warn_if_chunks_have_mismatched_type(all_parsers, using_infer_string):
    warning_type = None
    parser = all_parsers
    size = 10000

    # see gh-3866: if chunks are different types and can't
    # be coerced using numerical types, then issue warning.
    if parser.engine == "c" and parser.low_memory:
        warning_type = DtypeWarning
        # Use larger size to hit warning path
        size = 499999

    integers = [str(i) for i in range(size)]
    data = "a\n" + "\n".join(integers + ["a", "b"] + integers)

    buf = StringIO(data)

    if parser.engine == "pyarrow":
        df = parser.read_csv(
            buf,
        )
    else:
        df = parser.read_csv_check_warnings(
            warning_type,
            r"Columns \(0\) have mixed types. "
            "Specify dtype option on import or set low_memory=False.",
            buf,
        )
    if parser.engine == "c" and parser.low_memory:
        assert df.a.dtype == object
    elif using_infer_string:
        assert df.a.dtype == "str"
    else:
        assert df.a.dtype == object


@pytest.mark.parametrize("iterator", [True, False])
def test_empty_with_nrows_chunksize(all_parsers, iterator):
    # see gh-9535
    parser = all_parsers
    expected = DataFrame(columns=["foo", "bar"])

    nrows = 10
    data = StringIO("foo,bar\n")

    if parser.engine == "pyarrow":
        msg = (
            "The '(nrows|chunksize)' option is not supported with the 'pyarrow' engine"
        )
        with pytest.raises(ValueError, match=msg):
            if iterator:
                with parser.read_csv(data, chunksize=nrows) as reader:
                    next(iter(reader))
            else:
                parser.read_csv(data, nrows=nrows)
        return

    if iterator:
        with parser.read_csv(data, chunksize=nrows) as reader:
            result = next(iter(reader))
    else:
        result = parser.read_csv(data, nrows=nrows)

    tm.assert_frame_equal(result, expected)


def test_read_csv_memory_growth_chunksize(all_parsers):
    # see gh-24805
    #
    # Let's just make sure that we don't crash
    # as we iteratively process all chunks.
    parser = all_parsers

    with tm.ensure_clean() as path:
        with open(path, "w", encoding="utf-8") as f:
            for i in range(1000):
                f.write(str(i) + "\n")

        if parser.engine == "pyarrow":
            msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                with parser.read_csv(path, chunksize=20) as result:
                    for _ in result:
                        pass
            return

        with parser.read_csv(path, chunksize=20) as result:
            for _ in result:
                pass


def test_chunksize_with_usecols_second_block_shorter(all_parsers):
    # GH#21211
    parser = all_parsers
    data = """1,2,3,4
5,6,7,8
9,10,11
"""

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                names=["a", "b"],
                chunksize=2,
                usecols=[0, 1],
                header=None,
            )
        return

    result_chunks = parser.read_csv(
        StringIO(data),
        names=["a", "b"],
        chunksize=2,
        usecols=[0, 1],
        header=None,
    )

    expected_frames = [
        DataFrame({"a": [1, 5], "b": [2, 6]}),
        DataFrame({"a": [9], "b": [10]}, index=[2]),
    ]

    for i, result in enumerate(result_chunks):
        tm.assert_frame_equal(result, expected_frames[i])


def test_chunksize_second_block_shorter(all_parsers):
    # GH#21211
    parser = all_parsers
    data = """a,b,c,d
1,2,3,4
5,6,7,8
9,10,11
"""

    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), chunksize=2)
        return

    result_chunks = parser.read_csv(StringIO(data), chunksize=2)

    expected_frames = [
        DataFrame({"a": [1, 5], "b": [2, 6], "c": [3, 7], "d": [4, 8]}),
        DataFrame({"a": [9], "b": [10], "c": [11], "d": [np.nan]}, index=[2]),
    ]

    for i, result in enumerate(result_chunks):
        tm.assert_frame_equal(result, expected_frames[i])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\tests\test_point.py ===
from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings

def test_point_v1pt_theorys():
    q, q2 = dynamicsymbols('q q2')
    qd, q2d = dynamicsymbols('q q2', 1)
    qdd, q2dd = dynamicsymbols('q q2', 2)
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    B.set_ang_vel(N, qd * B.z)
    O = Point('O')
    P = O.locatenew('P', B.x)
    P.set_vel(B, 0)
    O.set_vel(N, 0)
    assert P.v1pt_theory(O, N, B) == qd * B.y
    O.set_vel(N, N.x)
    assert P.v1pt_theory(O, N, B) == N.x + qd * B.y
    P.set_vel(B, B.z)
    assert P.v1pt_theory(O, N, B) == B.z + N.x + qd * B.y


def test_point_a1pt_theorys():
    q, q2 = dynamicsymbols('q q2')
    qd, q2d = dynamicsymbols('q q2', 1)
    qdd, q2dd = dynamicsymbols('q q2', 2)
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    B.set_ang_vel(N, qd * B.z)
    O = Point('O')
    P = O.locatenew('P', B.x)
    P.set_vel(B, 0)
    O.set_vel(N, 0)
    assert P.a1pt_theory(O, N, B) == -(qd**2) * B.x + qdd * B.y
    P.set_vel(B, q2d * B.z)
    assert P.a1pt_theory(O, N, B) == -(qd**2) * B.x + qdd * B.y + q2dd * B.z
    O.set_vel(N, q2d * B.x)
    assert P.a1pt_theory(O, N, B) == ((q2dd - qd**2) * B.x + (q2d * qd + qdd) * B.y +
                               q2dd * B.z)


def test_point_v2pt_theorys():
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    N = ReferenceFrame('N')
    B = N.orientnew('B', 'Axis', [q, N.z])
    O = Point('O')
    P = O.locatenew('P', 0)
    O.set_vel(N, 0)
    assert P.v2pt_theory(O, N, B) == 0
    P = O.locatenew('P', B.x)
    assert P.v2pt_theory(O, N, B) == (qd * B.z ^ B.x)
    O.set_vel(N, N.x)
    assert P.v2pt_theory(O, N, B) == N.x + qd * B.y


def test_point_a2pt_theorys():
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    qdd = dynamicsymbols('q', 2)
    N = ReferenceFrame('N')
    B = N.orientnew('B', 'Axis', [q, N.z])
    O = Point('O')
    P = O.locatenew('P', 0)
    O.set_vel(N, 0)
    assert P.a2pt_theory(O, N, B) == 0
    P.set_pos(O, B.x)
    assert P.a2pt_theory(O, N, B) == (-qd**2) * B.x + (qdd) * B.y


def test_point_funcs():
    q, q2 = dynamicsymbols('q q2')
    qd, q2d = dynamicsymbols('q q2', 1)
    qdd, q2dd = dynamicsymbols('q q2', 2)
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    B.set_ang_vel(N, 5 * B.y)
    O = Point('O')
    P = O.locatenew('P', q * B.x + q2 * B.y)
    assert P.pos_from(O) == q * B.x + q2 * B.y
    P.set_vel(B, qd * B.x + q2d * B.y)
    assert P.vel(B) == qd * B.x + q2d * B.y
    O.set_vel(N, 0)
    assert O.vel(N) == 0
    assert P.a1pt_theory(O, N, B) == ((-25 * q + qdd) * B.x + (q2dd) * B.y +
                               (-10 * qd) * B.z)

    B = N.orientnew('B', 'Axis', [q, N.z])
    O = Point('O')
    P = O.locatenew('P', 10 * B.x)
    O.set_vel(N, 5 * N.x)
    assert O.vel(N) == 5 * N.x
    assert P.a2pt_theory(O, N, B) == (-10 * qd**2) * B.x + (10 * qdd) * B.y

    B.set_ang_vel(N, 5 * B.y)
    O = Point('O')
    P = O.locatenew('P', q * B.x + q2 * B.y)
    P.set_vel(B, qd * B.x + q2d * B.y)
    O.set_vel(N, 0)
    assert P.v1pt_theory(O, N, B) == qd * B.x + q2d * B.y - 5 * q * B.z


def test_point_pos():
    q = dynamicsymbols('q')
    N = ReferenceFrame('N')
    B = N.orientnew('B', 'Axis', [q, N.z])
    O = Point('O')
    P = O.locatenew('P', 10 * N.x + 5 * B.x)
    assert P.pos_from(O) == 10 * N.x + 5 * B.x
    Q = P.locatenew('Q', 10 * N.y + 5 * B.y)
    assert Q.pos_from(P) == 10 * N.y + 5 * B.y
    assert Q.pos_from(O) == 10 * N.x + 10 * N.y + 5 * B.x + 5 * B.y
    assert O.pos_from(Q) == -10 * N.x - 10 * N.y - 5 * B.x - 5 * B.y

def test_point_partial_velocity():

    N = ReferenceFrame('N')
    A = ReferenceFrame('A')

    p = Point('p')

    u1, u2 = dynamicsymbols('u1, u2')

    p.set_vel(N, u1 * A.x + u2 * N.y)

    assert p.partial_velocity(N, u1) == A.x
    assert p.partial_velocity(N, u1, u2) == (A.x, N.y)
    raises(ValueError, lambda: p.partial_velocity(A, u1))

def test_point_vel(): #Basic functionality
    q1, q2 = dynamicsymbols('q1 q2')
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    Q = Point('Q')
    O = Point('O')
    Q.set_pos(O, q1 * N.x)
    raises(ValueError , lambda: Q.vel(N)) # Velocity of O in N is not defined
    O.set_vel(N, q2 * N.y)
    assert O.vel(N) == q2 * N.y
    raises(ValueError , lambda : O.vel(B)) #Velocity of O is not defined in B

def test_auto_point_vel():
    t = dynamicsymbols._t
    q1, q2 = dynamicsymbols('q1 q2')
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    O = Point('O')
    Q = Point('Q')
    Q.set_pos(O, q1 * N.x)
    O.set_vel(N, q2 * N.y)
    assert Q.vel(N) == q1.diff(t) * N.x + q2 * N.y  # Velocity of Q using O
    P1 = Point('P1')
    P1.set_pos(O, q1 * B.x)
    P2 = Point('P2')
    P2.set_pos(P1, q2 * B.z)
    raises(ValueError, lambda : P2.vel(B)) # O's velocity is defined in different frame, and no
    #point in between has its velocity defined
    raises(ValueError, lambda: P2.vel(N)) # Velocity of O not defined in N

def test_auto_point_vel_multiple_point_path():
    t = dynamicsymbols._t
    q1, q2 = dynamicsymbols('q1 q2')
    B = ReferenceFrame('B')
    P = Point('P')
    P.set_vel(B, q1 * B.x)
    P1 = Point('P1')
    P1.set_pos(P, q2 * B.y)
    P1.set_vel(B, q1 * B.z)
    P2 = Point('P2')
    P2.set_pos(P1, q1 * B.z)
    P3 = Point('P3')
    P3.set_pos(P2, 10 * q1 * B.y)
    assert P3.vel(B) == 10 * q1.diff(t) * B.y + (q1 + q1.diff(t)) * B.z

def test_auto_vel_dont_overwrite():
    t = dynamicsymbols._t
    q1, q2, u1 = dynamicsymbols('q1, q2, u1')
    N = ReferenceFrame('N')
    P = Point('P1')
    P.set_vel(N, u1 * N.x)
    P1 = Point('P1')
    P1.set_pos(P, q2 * N.y)
    assert P1.vel(N) == q2.diff(t) * N.y + u1 * N.x
    assert P.vel(N) == u1 * N.x
    P1.set_vel(N, u1 * N.z)
    assert P1.vel(N) == u1 * N.z

def test_auto_point_vel_if_tree_has_vel_but_inappropriate_pos_vector():
    q1, q2 = dynamicsymbols('q1 q2')
    B = ReferenceFrame('B')
    S = ReferenceFrame('S')
    P = Point('P')
    P.set_vel(B, q1 * B.x)
    P1 = Point('P1')
    P1.set_pos(P, S.y)
    raises(ValueError, lambda : P1.vel(B)) # P1.pos_from(P) can't be expressed in B
    raises(ValueError, lambda : P1.vel(S)) # P.vel(S) not defined

def test_auto_point_vel_shortest_path():
    t = dynamicsymbols._t
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    B = ReferenceFrame('B')
    P = Point('P')
    P.set_vel(B, u1 * B.x)
    P1 = Point('P1')
    P1.set_pos(P, q2 * B.y)
    P1.set_vel(B, q1 * B.z)
    P2 = Point('P2')
    P2.set_pos(P1, q1 * B.z)
    P3 = Point('P3')
    P3.set_pos(P2, 10 * q1 * B.y)
    P4 = Point('P4')
    P4.set_pos(P3, q1 * B.x)
    O = Point('O')
    O.set_vel(B, u2 * B.y)
    O1 = Point('O1')
    O1.set_pos(O, q2 * B.z)
    P4.set_pos(O1, q1 * B.x + q2 * B.z)
    with warnings.catch_warnings(): #There are two possible paths in this point tree, thus a warning is raised
        warnings.simplefilter('error')
        with ignore_warnings(UserWarning):
            assert P4.vel(B) == q1.diff(t) * B.x + u2 * B.y + 2 * q2.diff(t) * B.z

def test_auto_point_vel_connected_frames():
    t = dynamicsymbols._t
    q, q1, q2, u = dynamicsymbols('q q1 q2 u')
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    O = Point('O')
    O.set_vel(N, u * N.x)
    P = Point('P')
    P.set_pos(O, q1 * N.x + q2 * B.y)
    raises(ValueError, lambda: P.vel(N))
    N.orient(B, 'Axis', (q, B.x))
    assert P.vel(N) == (u + q1.diff(t)) * N.x + q2.diff(t) * B.y - q2 * q.diff(t) * B.z

def test_auto_point_vel_multiple_paths_warning_arises():
    q, u = dynamicsymbols('q u')
    N = ReferenceFrame('N')
    O = Point('O')
    P = Point('P')
    Q = Point('Q')
    R = Point('R')
    P.set_vel(N, u * N.x)
    Q.set_vel(N, u *N.y)
    R.set_vel(N, u * N.z)
    O.set_pos(P, q * N.z)
    O.set_pos(Q, q * N.y)
    O.set_pos(R, q * N.x)
    with warnings.catch_warnings(): #There are two possible paths in this point tree, thus a warning is raised
        warnings.simplefilter("error")
        raises(UserWarning ,lambda: O.vel(N))

def test_auto_vel_cyclic_warning_arises():
    P = Point('P')
    P1 = Point('P1')
    P2 = Point('P2')
    P3 = Point('P3')
    N = ReferenceFrame('N')
    P.set_vel(N, N.x)
    P1.set_pos(P, N.x)
    P2.set_pos(P1, N.y)
    P3.set_pos(P2, N.z)
    P1.set_pos(P3, N.x + N.y)
    with warnings.catch_warnings(): #The path is cyclic at P1, thus a warning is raised
        warnings.simplefilter("error")
        raises(UserWarning ,lambda: P2.vel(N))

def test_auto_vel_cyclic_warning_msg():
    P = Point('P')
    P1 = Point('P1')
    P2 = Point('P2')
    P3 = Point('P3')
    N = ReferenceFrame('N')
    P.set_vel(N, N.x)
    P1.set_pos(P, N.x)
    P2.set_pos(P1, N.y)
    P3.set_pos(P2, N.z)
    P1.set_pos(P3, N.x + N.y)
    with warnings.catch_warnings(record = True) as w: #The path is cyclic at P1, thus a warning is raised
        warnings.simplefilter("always")
        P2.vel(N)
        msg = str(w[-1].message).replace("\n", " ")
        assert issubclass(w[-1].category, UserWarning)
        assert 'Kinematic loops are defined among the positions of points. This is likely not desired and may cause errors in your calculations.' in msg

def test_auto_vel_multiple_path_warning_msg():
    N = ReferenceFrame('N')
    O = Point('O')
    P = Point('P')
    Q = Point('Q')
    P.set_vel(N, N.x)
    Q.set_vel(N, N.y)
    O.set_pos(P, N.z)
    O.set_pos(Q, N.y)
    with warnings.catch_warnings(record = True) as w: #There are two possible paths in this point tree, thus a warning is raised
        warnings.simplefilter("always")
        O.vel(N)
        msg = str(w[-1].message).replace("\n", " ")
        assert issubclass(w[-1].category, UserWarning)
        assert 'Velocity' in msg
        assert 'automatically calculated based on point' in msg
        assert 'Velocities from these points are not necessarily the same. This may cause errors in your calculations.' in msg

def test_auto_vel_derivative():
    q1, q2 = dynamicsymbols('q1:3')
    u1, u2 = dynamicsymbols('u1:3', 1)
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    B.orient_axis(A, A.z, q1)
    B.set_ang_vel(A, u1 * A.z)
    C.orient_axis(B, B.z, q2)
    C.set_ang_vel(B, u2 * B.z)

    Am = Point('Am')
    Am.set_vel(A, 0)
    Bm = Point('Bm')
    Bm.set_pos(Am, B.x)
    Bm.set_vel(B, 0)
    Bm.set_vel(C, 0)
    Cm = Point('Cm')
    Cm.set_pos(Bm, C.x)
    Cm.set_vel(C, 0)
    temp = Cm._vel_dict.copy()
    assert Cm.vel(A) == (u1 * B.y + (u1 + u2) * C.y)
    Cm._vel_dict = temp
    Cm.v2pt_theory(Bm, B, C)
    assert Cm.vel(A) == (u1 * B.y + (u1 + u2) * C.y)

def test_auto_point_acc_zero_vel():
    N = ReferenceFrame('N')
    O = Point('O')
    O.set_vel(N, 0)
    assert O.acc(N) == 0 * N.x

def test_auto_point_acc_compute_vel():
    t = dynamicsymbols._t
    q1 = dynamicsymbols('q1')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    A.orient_axis(N, N.z, q1)

    O = Point('O')
    O.set_vel(N, 0)
    P = Point('P')
    P.set_pos(O, A.x)
    assert P.acc(N) == -q1.diff(t) ** 2 * A.x + q1.diff(t, 2) * A.y

def test_auto_acc_derivative():
    # Tests whether the Point.acc method gives the correct acceleration of the
    # end point of two linkages in series, while getting minimal information.
    q1, q2 = dynamicsymbols('q1:3')
    u1, u2 = dynamicsymbols('q1:3', 1)
    v1, v2 = dynamicsymbols('q1:3', 2)
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    B.orient_axis(A, A.z, q1)
    C.orient_axis(B, B.z, q2)

    Am = Point('Am')
    Am.set_vel(A, 0)
    Bm = Point('Bm')
    Bm.set_pos(Am, B.x)
    Bm.set_vel(B, 0)
    Bm.set_vel(C, 0)
    Cm = Point('Cm')
    Cm.set_pos(Bm, C.x)
    Cm.set_vel(C, 0)

    # Copy dictionaries to later check the calculation using the 2pt_theories
    Bm_vel_dict, Cm_vel_dict = Bm._vel_dict.copy(), Cm._vel_dict.copy()
    Bm_acc_dict, Cm_acc_dict = Bm._acc_dict.copy(), Cm._acc_dict.copy()
    check = -u1 ** 2 * B.x + v1 * B.y - (u1 + u2) ** 2 * C.x + (v1 + v2) * C.y
    assert Cm.acc(A) == check
    Bm._vel_dict, Cm._vel_dict = Bm_vel_dict, Cm_vel_dict
    Bm._acc_dict, Cm._acc_dict = Bm_acc_dict, Cm_acc_dict
    Bm.v2pt_theory(Am, A, B)
    Cm.v2pt_theory(Bm, A, C)
    Bm.a2pt_theory(Am, A, B)
    assert Cm.a2pt_theory(Bm, A, C) == check

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_repr.py ===
from __future__ import annotations
from typing import Any

from sympy.external.gmpy import GROUND_TYPES
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.assumptions.ask import Q
from sympy.core.function import (Function, WildFunction)
from sympy.core.numbers import (AlgebraicNumber, Float, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.special.delta_functions import Heaviside
from sympy.logic.boolalg import (false, true)
from sympy.matrices.dense import (Matrix, ones)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.combinatorics import Cycle, Permutation
from sympy.core.symbol import Str
from sympy.geometry import Point, Ellipse
from sympy.printing import srepr
from sympy.polys import ring, field, ZZ, QQ, lex, grlex, Poly
from sympy.polys.polyclasses import DMP
from sympy.polys.agca.extensions import FiniteExtension

x, y = symbols('x,y')

# eval(srepr(expr)) == expr has to succeed in the right environment. The right
# environment is the scope of "from sympy import *" for most cases.
ENV: dict[str, Any] = {"Str": Str}
exec("from sympy import *", ENV)


def sT(expr, string, import_stmt=None, **kwargs):
    """
    sT := sreprTest

    Tests that srepr delivers the expected string and that
    the condition eval(srepr(expr))==expr holds.
    """
    if import_stmt is None:
        ENV2 = ENV
    else:
        ENV2 = ENV.copy()
        exec(import_stmt, ENV2)

    assert srepr(expr, **kwargs) == string
    assert eval(string, ENV2) == expr


def test_printmethod():
    class R(Abs):
        def _sympyrepr(self, printer):
            return "foo(%s)" % printer._print(self.args[0])
    assert srepr(R(x)) == "foo(Symbol('x'))"


def test_Add():
    sT(x + y, "Add(Symbol('x'), Symbol('y'))")
    assert srepr(x**2 + 1, order='lex') == "Add(Pow(Symbol('x'), Integer(2)), Integer(1))"
    assert srepr(x**2 + 1, order='old') == "Add(Integer(1), Pow(Symbol('x'), Integer(2)))"
    assert srepr(sympify('x + 3 - 2', evaluate=False), order='none') == "Add(Symbol('x'), Integer(3), Mul(Integer(-1), Integer(2)))"


def test_more_than_255_args_issue_10259():
    from sympy.core.add import Add
    from sympy.core.mul import Mul
    for op in (Add, Mul):
        expr = op(*symbols('x:256'))
        assert eval(srepr(expr)) == expr


def test_Function():
    sT(Function("f")(x), "Function('f')(Symbol('x'))")
    # test unapplied Function
    sT(Function('f'), "Function('f')")

    sT(sin(x), "sin(Symbol('x'))")
    sT(sin, "sin")


def test_Heaviside():
    sT(Heaviside(x), "Heaviside(Symbol('x'))")
    sT(Heaviside(x, 1), "Heaviside(Symbol('x'), Integer(1))")


def test_Geometry():
    sT(Point(0, 0), "Point2D(Integer(0), Integer(0))")
    sT(Ellipse(Point(0, 0), 5, 1),
       "Ellipse(Point2D(Integer(0), Integer(0)), Integer(5), Integer(1))")
    # TODO more tests


def test_Singletons():
    sT(S.Catalan, 'Catalan')
    sT(S.ComplexInfinity, 'zoo')
    sT(S.EulerGamma, 'EulerGamma')
    sT(S.Exp1, 'E')
    sT(S.GoldenRatio, 'GoldenRatio')
    sT(S.TribonacciConstant, 'TribonacciConstant')
    sT(S.Half, 'Rational(1, 2)')
    sT(S.ImaginaryUnit, 'I')
    sT(S.Infinity, 'oo')
    sT(S.NaN, 'nan')
    sT(S.NegativeInfinity, '-oo')
    sT(S.NegativeOne, 'Integer(-1)')
    sT(S.One, 'Integer(1)')
    sT(S.Pi, 'pi')
    sT(S.Zero, 'Integer(0)')
    sT(S.Complexes, 'Complexes')
    sT(S.EmptySequence, 'EmptySequence')
    sT(S.EmptySet, 'EmptySet')
    # sT(S.IdentityFunction, 'Lambda(_x, _x)')
    sT(S.Naturals, 'Naturals')
    sT(S.Naturals0, 'Naturals0')
    sT(S.Rationals, 'Rationals')
    sT(S.Reals, 'Reals')
    sT(S.UniversalSet, 'UniversalSet')


def test_Integer():
    sT(Integer(4), "Integer(4)")


def test_list():
    sT([x, Integer(4)], "[Symbol('x'), Integer(4)]")


def test_Matrix():
    for cls, name in [(Matrix, "MutableDenseMatrix"), (ImmutableDenseMatrix, "ImmutableDenseMatrix")]:
        sT(cls([[x**+1, 1], [y, x + y]]),
           "%s([[Symbol('x'), Integer(1)], [Symbol('y'), Add(Symbol('x'), Symbol('y'))]])" % name)

        sT(cls(), "%s([])" % name)

        sT(cls([[x**+1, 1], [y, x + y]]), "%s([[Symbol('x'), Integer(1)], [Symbol('y'), Add(Symbol('x'), Symbol('y'))]])" % name)


def test_empty_Matrix():
    sT(ones(0, 3), "MutableDenseMatrix(0, 3, [])")
    sT(ones(4, 0), "MutableDenseMatrix(4, 0, [])")
    sT(ones(0, 0), "MutableDenseMatrix([])")


def test_Rational():
    sT(Rational(1, 3), "Rational(1, 3)")
    sT(Rational(-1, 3), "Rational(-1, 3)")


def test_Float():
    sT(Float('1.23', dps=3), "Float('1.22998', precision=13)")
    sT(Float('1.23456789', dps=9), "Float('1.23456788994', precision=33)")
    sT(Float('1.234567890123456789', dps=19),
       "Float('1.234567890123456789013', precision=66)")
    sT(Float('0.60038617995049726', dps=15),
       "Float('0.60038617995049726', precision=53)")

    sT(Float('1.23', precision=13), "Float('1.22998', precision=13)")
    sT(Float('1.23456789', precision=33),
       "Float('1.23456788994', precision=33)")
    sT(Float('1.234567890123456789', precision=66),
       "Float('1.234567890123456789013', precision=66)")
    sT(Float('0.60038617995049726', precision=53),
       "Float('0.60038617995049726', precision=53)")

    sT(Float('0.60038617995049726', 15),
       "Float('0.60038617995049726', precision=53)")


def test_Symbol():
    sT(x, "Symbol('x')")
    sT(y, "Symbol('y')")
    sT(Symbol('x', negative=True), "Symbol('x', negative=True)")


def test_Symbol_two_assumptions():
    x = Symbol('x', negative=0, integer=1)
    # order could vary
    s1 = "Symbol('x', integer=True, negative=False)"
    s2 = "Symbol('x', negative=False, integer=True)"
    assert srepr(x) in (s1, s2)
    assert eval(srepr(x), ENV) == x


def test_Symbol_no_special_commutative_treatment():
    sT(Symbol('x'), "Symbol('x')")
    sT(Symbol('x', commutative=False), "Symbol('x', commutative=False)")
    sT(Symbol('x', commutative=0), "Symbol('x', commutative=False)")
    sT(Symbol('x', commutative=True), "Symbol('x', commutative=True)")
    sT(Symbol('x', commutative=1), "Symbol('x', commutative=True)")


def test_Wild():
    sT(Wild('x', even=True), "Wild('x', even=True)")


def test_Dummy():
    d = Dummy('d')
    sT(d, "Dummy('d', dummy_index=%s)" % str(d.dummy_index))


def test_Dummy_assumption():
    d = Dummy('d', nonzero=True)
    assert d == eval(srepr(d))
    s1 = "Dummy('d', dummy_index=%s, nonzero=True)" % str(d.dummy_index)
    s2 = "Dummy('d', nonzero=True, dummy_index=%s)" % str(d.dummy_index)
    assert srepr(d) in (s1, s2)


def test_Dummy_from_Symbol():
    # should not get the full dictionary of assumptions
    n = Symbol('n', integer=True)
    d = n.as_dummy()
    assert srepr(d
        ) == "Dummy('n', dummy_index=%s)" % str(d.dummy_index)


def test_tuple():
    sT((x,), "(Symbol('x'),)")
    sT((x, y), "(Symbol('x'), Symbol('y'))")


def test_WildFunction():
    sT(WildFunction('w'), "WildFunction('w')")


def test_settins():
    raises(TypeError, lambda: srepr(x, method="garbage"))


def test_Mul():
    sT(3*x**3*y, "Mul(Integer(3), Pow(Symbol('x'), Integer(3)), Symbol('y'))")
    assert srepr(3*x**3*y, order='old') == "Mul(Integer(3), Symbol('y'), Pow(Symbol('x'), Integer(3)))"
    assert srepr(sympify('(x+4)*2*x*7', evaluate=False), order='none') == "Mul(Add(Symbol('x'), Integer(4)), Integer(2), Symbol('x'), Integer(7))"


def test_AlgebraicNumber():
    a = AlgebraicNumber(sqrt(2))
    sT(a, "AlgebraicNumber(Pow(Integer(2), Rational(1, 2)), [Integer(1), Integer(0)])")
    a = AlgebraicNumber(root(-2, 3))
    sT(a, "AlgebraicNumber(Pow(Integer(-2), Rational(1, 3)), [Integer(1), Integer(0)])")


def test_PolyRing():
    assert srepr(ring("x", ZZ, lex)[0]) == "PolyRing((Symbol('x'),), ZZ, lex)"
    assert srepr(ring("x,y", QQ, grlex)[0]) == "PolyRing((Symbol('x'), Symbol('y')), QQ, grlex)"
    assert srepr(ring("x,y,z", ZZ["t"], lex)[0]) == "PolyRing((Symbol('x'), Symbol('y'), Symbol('z')), ZZ[t], lex)"


def test_FracField():
    assert srepr(field("x", ZZ, lex)[0]) == "FracField((Symbol('x'),), ZZ, lex)"
    assert srepr(field("x,y", QQ, grlex)[0]) == "FracField((Symbol('x'), Symbol('y')), QQ, grlex)"
    assert srepr(field("x,y,z", ZZ["t"], lex)[0]) == "FracField((Symbol('x'), Symbol('y'), Symbol('z')), ZZ[t], lex)"


def test_PolyElement():
    R, x, y = ring("x,y", ZZ)
    assert srepr(3*x**2*y + 1) == "PolyElement(PolyRing((Symbol('x'), Symbol('y')), ZZ, lex), [((2, 1), 3), ((0, 0), 1)])"


def test_FracElement():
    F, x, y = field("x,y", ZZ)
    assert srepr((3*x**2*y + 1)/(x - y**2)) == "FracElement(FracField((Symbol('x'), Symbol('y')), ZZ, lex), [((2, 1), 3), ((0, 0), 1)], [((1, 0), 1), ((0, 2), -1)])"


def test_FractionField():
    assert srepr(QQ.frac_field(x)) == \
        "FractionField(FracField((Symbol('x'),), QQ, lex))"
    assert srepr(QQ.frac_field(x, y, order=grlex)) == \
        "FractionField(FracField((Symbol('x'), Symbol('y')), QQ, grlex))"


def test_PolynomialRingBase():
    assert srepr(ZZ.old_poly_ring(x)) == \
        "GlobalPolynomialRing(ZZ, Symbol('x'))"
    assert srepr(ZZ[x].old_poly_ring(y)) == \
        "GlobalPolynomialRing(ZZ[x], Symbol('y'))"
    assert srepr(QQ.frac_field(x).old_poly_ring(y)) == \
        "GlobalPolynomialRing(FractionField(FracField((Symbol('x'),), QQ, lex)), Symbol('y'))"


def test_DMP():
    p1 = DMP([1, 2], ZZ)
    p2 = ZZ.old_poly_ring(x)([1, 2])
    if GROUND_TYPES != 'flint':
        assert srepr(p1) == "DMP_Python([1, 2], ZZ)"
        assert srepr(p2) == "DMP_Python([1, 2], ZZ)"
    else:
        assert srepr(p1) == "DUP_Flint([1, 2], ZZ)"
        assert srepr(p2) == "DUP_Flint([1, 2], ZZ)"


def test_FiniteExtension():
    assert srepr(FiniteExtension(Poly(x**2 + 1, x))) == \
        "FiniteExtension(Poly(x**2 + 1, x, domain='ZZ'))"


def test_ExtensionElement():
    A = FiniteExtension(Poly(x**2 + 1, x))
    if GROUND_TYPES != 'flint':
        ans = "ExtElem(DMP_Python([1, 0], ZZ), FiniteExtension(Poly(x**2 + 1, x, domain='ZZ')))"
    else:
        ans = "ExtElem(DUP_Flint([1, 0], ZZ), FiniteExtension(Poly(x**2 + 1, x, domain='ZZ')))"
    assert srepr(A.generator) == ans

def test_BooleanAtom():
    assert srepr(true) == "true"
    assert srepr(false) == "false"


def test_Integers():
    sT(S.Integers, "Integers")


def test_Naturals():
    sT(S.Naturals, "Naturals")


def test_Naturals0():
    sT(S.Naturals0, "Naturals0")


def test_Reals():
    sT(S.Reals, "Reals")


def test_matrix_expressions():
    n = symbols('n', integer=True)
    A = MatrixSymbol("A", n, n)
    B = MatrixSymbol("B", n, n)
    sT(A, "MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True))")
    sT(A*B, "MatMul(MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Str('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")
    sT(A + B, "MatAdd(MatrixSymbol(Str('A'), Symbol('n', integer=True), Symbol('n', integer=True)), MatrixSymbol(Str('B'), Symbol('n', integer=True), Symbol('n', integer=True)))")


def test_Cycle():
    # FIXME: sT fails because Cycle is not immutable and calling srepr(Cycle(1, 2))
    # adds keys to the Cycle dict (GH-17661)
    #import_stmt = "from sympy.combinatorics import Cycle"
    #sT(Cycle(1, 2), "Cycle(1, 2)", import_stmt)
    assert srepr(Cycle(1, 2)) == "Cycle(1, 2)"


def test_Permutation():
    import_stmt = "from sympy.combinatorics import Permutation"
    sT(Permutation(1, 2)(3, 4), "Permutation([0, 2, 1, 4, 3])", import_stmt, perm_cyclic=False)
    sT(Permutation(1, 2)(3, 4), "Permutation(1, 2)(3, 4)", import_stmt, perm_cyclic=True)

    with warns_deprecated_sympy():
        old_print_cyclic = Permutation.print_cyclic
        Permutation.print_cyclic = False
        sT(Permutation(1, 2)(3, 4), "Permutation([0, 2, 1, 4, 3])", import_stmt)
        Permutation.print_cyclic = old_print_cyclic

def test_dict():
    from sympy.abc import x, y, z
    d = {}
    assert srepr(d) == "{}"
    d = {x: y}
    assert srepr(d) == "{Symbol('x'): Symbol('y')}"
    d = {x: y, y: z}
    assert srepr(d) in (
        "{Symbol('x'): Symbol('y'), Symbol('y'): Symbol('z')}",
        "{Symbol('y'): Symbol('z'), Symbol('x'): Symbol('y')}",
    )
    d = {x: {y: z}}
    assert srepr(d) == "{Symbol('x'): {Symbol('y'): Symbol('z')}}"

def test_set():
    from sympy.abc import x, y
    s = set()
    assert srepr(s) == "set()"
    s = {x, y}
    assert srepr(s) in ("{Symbol('x'), Symbol('y')}", "{Symbol('y'), Symbol('x')}")

def test_Predicate():
    sT(Q.even, "Q.even")

def test_AppliedPredicate():
    sT(Q.even(Symbol('z')), "AppliedPredicate(Q.even, Symbol('z'))")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_maple.py ===
from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
                        Tuple, Symbol, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos, sinc, lucas
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
                            HadamardProduct, SparseMatrix)
from sympy.functions.special.bessel import besseli

from sympy.printing.maple import maple_code

x, y, z = symbols('x,y,z')


def test_Integer():
    assert maple_code(Integer(67)) == "67"
    assert maple_code(Integer(-1)) == "-1"


def test_Rational():
    assert maple_code(Rational(3, 7)) == "3/7"
    assert maple_code(Rational(18, 9)) == "2"
    assert maple_code(Rational(3, -7)) == "-3/7"
    assert maple_code(Rational(-3, -7)) == "3/7"
    assert maple_code(x + Rational(3, 7)) == "x + 3/7"
    assert maple_code(Rational(3, 7) * x) == '(3/7)*x'


def test_Relational():
    assert maple_code(Eq(x, y)) == "x = y"
    assert maple_code(Ne(x, y)) == "x <> y"
    assert maple_code(Le(x, y)) == "x <= y"
    assert maple_code(Lt(x, y)) == "x < y"
    assert maple_code(Gt(x, y)) == "x > y"
    assert maple_code(Ge(x, y)) == "x >= y"


def test_Function():
    assert maple_code(sin(x) ** cos(x)) == "sin(x)^cos(x)"
    assert maple_code(abs(x)) == "abs(x)"
    assert maple_code(ceiling(x)) == "ceil(x)"


def test_Pow():
    assert maple_code(x ** 3) == "x^3"
    assert maple_code(x ** (y ** 3)) == "x^(y^3)"

    assert maple_code((x ** 3) ** y) == "(x^3)^y"
    assert maple_code(x ** Rational(2, 3)) == 'x^(2/3)'

    g = implemented_function('g', Lambda(x, 2 * x))
    assert maple_code(1 / (g(x) * 3.5) ** (x - y ** x) / (x ** 2 + y)) == \
           "(3.5*2*x)^(-x + y^x)/(x^2 + y)"
    # For issue 14160
    assert maple_code(Mul(-2, x, Pow(Mul(y, y, evaluate=False), -1, evaluate=False),
                          evaluate=False)) == '-2*x/(y*y)'


def test_basic_ops():
    assert maple_code(x * y) == "x*y"
    assert maple_code(x + y) == "x + y"
    assert maple_code(x - y) == "x - y"
    assert maple_code(-x) == "-x"


def test_1_over_x_and_sqrt():
    # 1.0 and 0.5 would do something different in regular StrPrinter,
    # but these are exact in IEEE floating point so no different here.
    assert maple_code(1 / x) == '1/x'
    assert maple_code(x ** -1) == maple_code(x ** -1.0) == '1/x'
    assert maple_code(1 / sqrt(x)) == '1/sqrt(x)'
    assert maple_code(x ** -S.Half) == maple_code(x ** -0.5) == '1/sqrt(x)'
    assert maple_code(sqrt(x)) == 'sqrt(x)'
    assert maple_code(x ** S.Half) == maple_code(x ** 0.5) == 'sqrt(x)'
    assert maple_code(1 / pi) == '1/Pi'
    assert maple_code(pi ** -1) == maple_code(pi ** -1.0) == '1/Pi'
    assert maple_code(pi ** -0.5) == '1/sqrt(Pi)'


def test_mix_number_mult_symbols():
    assert maple_code(3 * x) == "3*x"
    assert maple_code(pi * x) == "Pi*x"
    assert maple_code(3 / x) == "3/x"
    assert maple_code(pi / x) == "Pi/x"
    assert maple_code(x / 3) == '(1/3)*x'
    assert maple_code(x / pi) == "x/Pi"
    assert maple_code(x * y) == "x*y"
    assert maple_code(3 * x * y) == "3*x*y"
    assert maple_code(3 * pi * x * y) == "3*Pi*x*y"
    assert maple_code(x / y) == "x/y"
    assert maple_code(3 * x / y) == "3*x/y"
    assert maple_code(x * y / z) == "x*y/z"
    assert maple_code(x / y * z) == "x*z/y"
    assert maple_code(1 / x / y) == "1/(x*y)"
    assert maple_code(2 * pi * x / y / z) == "2*Pi*x/(y*z)"
    assert maple_code(3 * pi / x) == "3*Pi/x"
    assert maple_code(S(3) / 5) == "3/5"
    assert maple_code(S(3) / 5 * x) == '(3/5)*x'
    assert maple_code(x / y / z) == "x/(y*z)"
    assert maple_code((x + y) / z) == "(x + y)/z"
    assert maple_code((x + y) / (z + x)) == "(x + y)/(x + z)"
    assert maple_code((x + y) / EulerGamma) == '(x + y)/gamma'
    assert maple_code(x / 3 / pi) == '(1/3)*x/Pi'
    assert maple_code(S(3) / 5 * x * y / pi) == '(3/5)*x*y/Pi'


def test_mix_number_pow_symbols():
    assert maple_code(pi ** 3) == 'Pi^3'
    assert maple_code(x ** 2) == 'x^2'

    assert maple_code(x ** (pi ** 3)) == 'x^(Pi^3)'
    assert maple_code(x ** y) == 'x^y'

    assert maple_code(x ** (y ** z)) == 'x^(y^z)'
    assert maple_code((x ** y) ** z) == '(x^y)^z'


def test_imag():
    I = S('I')
    assert maple_code(I) == "I"
    assert maple_code(5 * I) == "5*I"

    assert maple_code((S(3) / 2) * I) == "(3/2)*I"
    assert maple_code(3 + 4 * I) == "3 + 4*I"


def test_constants():
    assert maple_code(pi) == "Pi"
    assert maple_code(oo) == "infinity"
    assert maple_code(-oo) == "-infinity"
    assert maple_code(S.NegativeInfinity) == "-infinity"
    assert maple_code(S.NaN) == "undefined"
    assert maple_code(S.Exp1) == "exp(1)"
    assert maple_code(exp(1)) == "exp(1)"


def test_constants_other():
    assert maple_code(2 * GoldenRatio) == '2*(1/2 + (1/2)*sqrt(5))'
    assert maple_code(2 * Catalan) == '2*Catalan'
    assert maple_code(2 * EulerGamma) == "2*gamma"


def test_boolean():
    assert maple_code(x & y) == "x and y"
    assert maple_code(x | y) == "x or y"
    assert maple_code(~x) == "not x"
    assert maple_code(x & y & z) == "x and y and z"
    assert maple_code(x | y | z) == "x or y or z"
    assert maple_code((x & y) | z) == "z or x and y"
    assert maple_code((x | y) & z) == "z and (x or y)"


def test_Matrices():
    assert maple_code(Matrix(1, 1, [10])) == \
           'Matrix([[10]], storage = rectangular)'

    A = Matrix([[1, sin(x / 2), abs(x)],
                [0, 1, pi],
                [0, exp(1), ceiling(x)]])
    expected = \
        'Matrix(' \
        '[[1, sin((1/2)*x), abs(x)],' \
        ' [0, 1, Pi],' \
        ' [0, exp(1), ceil(x)]], ' \
        'storage = rectangular)'
    assert maple_code(A) == expected

    # row and columns
    assert maple_code(A[:, 0]) == \
           'Matrix([[1], [0], [0]], storage = rectangular)'
    assert maple_code(A[0, :]) == \
           'Matrix([[1, sin((1/2)*x), abs(x)]], storage = rectangular)'
    assert maple_code(Matrix([[x, x - y, -y]])) == \
           'Matrix([[x, x - y, -y]], storage = rectangular)'

    # empty matrices
    assert maple_code(Matrix(0, 0, [])) == \
           'Matrix([], storage = rectangular)'
    assert maple_code(Matrix(0, 3, [])) == \
           'Matrix([], storage = rectangular)'

def test_SparseMatrices():
    assert maple_code(SparseMatrix(Identity(2))) == 'Matrix([[1, 0], [0, 1]], storage = sparse)'


def test_vector_entries_hadamard():
    # For a row or column, user might to use the other dimension
    A = Matrix([[1, sin(2 / x), 3 * pi / x / 5]])
    assert maple_code(A) == \
           'Matrix([[1, sin(2/x), (3/5)*Pi/x]], storage = rectangular)'
    assert maple_code(A.T) == \
           'Matrix([[1], [sin(2/x)], [(3/5)*Pi/x]], storage = rectangular)'


def test_Matrices_entries_not_hadamard():
    A = Matrix([[1, sin(2 / x), 3 * pi / x / 5], [1, 2, x * y]])
    expected = \
        'Matrix([[1, sin(2/x), (3/5)*Pi/x], [1, 2, x*y]], ' \
        'storage = rectangular)'
    assert maple_code(A) == expected


def test_MatrixSymbol():
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    assert maple_code(A * B) == "A.B"
    assert maple_code(B * A) == "B.A"
    assert maple_code(2 * A * B) == "2*A.B"
    assert maple_code(B * 2 * A) == "2*B.A"

    assert maple_code(
        A * (B + 3 * Identity(n))) == "A.(3*Matrix(n, shape = identity) + B)"

    assert maple_code(A ** (x ** 2)) == "MatrixPower(A, x^2)"
    assert maple_code(A ** 3) == "MatrixPower(A, 3)"
    assert maple_code(A ** (S.Half)) == "MatrixPower(A, 1/2)"


def test_special_matrices():
    assert maple_code(6 * Identity(3)) == "6*Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], storage = sparse)"
    assert maple_code(Identity(x)) == 'Matrix(x, shape = identity)'


def test_containers():
    assert maple_code([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
           "[1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]"

    assert maple_code((1, 2, (3, 4))) == "[1, 2, [3, 4]]"
    assert maple_code([1]) == "[1]"
    assert maple_code((1,)) == "[1]"
    assert maple_code(Tuple(*[1, 2, 3])) == "[1, 2, 3]"
    assert maple_code((1, x * y, (3, x ** 2))) == "[1, x*y, [3, x^2]]"
    # scalar, matrix, empty matrix and empty list

    assert maple_code((1, eye(3), Matrix(0, 0, []), [])) == \
           "[1, Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], storage = rectangular), Matrix([], storage = rectangular), []]"


def test_maple_noninline():
    source = maple_code((x + y)/Catalan, assign_to='me', inline=False)
    expected = "me := (x + y)/Catalan"

    assert source == expected


def test_maple_matrix_assign_to():
    A = Matrix([[1, 2, 3]])
    assert maple_code(A, assign_to='a') == "a := Matrix([[1, 2, 3]], storage = rectangular)"
    A = Matrix([[1, 2], [3, 4]])
    assert maple_code(A, assign_to='A') == "A := Matrix([[1, 2], [3, 4]], storage = rectangular)"


def test_maple_matrix_assign_to_more():
    # assigning to Symbol or MatrixSymbol requires lhs/rhs match
    A = Matrix([[1, 2, 3]])
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 2, 3)
    assert maple_code(A, assign_to=B) == "B := Matrix([[1, 2, 3]], storage = rectangular)"
    raises(ValueError, lambda: maple_code(A, assign_to=x))
    raises(ValueError, lambda: maple_code(A, assign_to=C))


def test_maple_matrix_1x1():
    A = Matrix([[3]])
    assert maple_code(A, assign_to='B') == "B := Matrix([[3]], storage = rectangular)"


def test_maple_matrix_elements():
    A = Matrix([[x, 2, x * y]])

    assert maple_code(A[0, 0] ** 2 + A[0, 1] + A[0, 2]) == "x^2 + x*y + 2"
    AA = MatrixSymbol('AA', 1, 3)
    assert maple_code(AA) == "AA"

    assert maple_code(AA[0, 0] ** 2 + sin(AA[0, 1]) + AA[0, 2]) == \
           "sin(AA[1, 2]) + AA[1, 1]^2 + AA[1, 3]"
    assert maple_code(sum(AA)) == "AA[1, 1] + AA[1, 2] + AA[1, 3]"


def test_maple_boolean():
    assert maple_code(True) == "true"
    assert maple_code(S.true) == "true"
    assert maple_code(False) == "false"
    assert maple_code(S.false) == "false"


def test_sparse():
    M = SparseMatrix(5, 6, {})
    M[2, 2] = 10
    M[1, 2] = 20
    M[1, 3] = 22
    M[0, 3] = 30
    M[3, 0] = x * y
    assert maple_code(M) == \
           'Matrix([[0, 0, 0, 30, 0, 0],' \
           ' [0, 0, 20, 22, 0, 0],' \
           ' [0, 0, 10, 0, 0, 0],' \
           ' [x*y, 0, 0, 0, 0, 0],' \
           ' [0, 0, 0, 0, 0, 0]], ' \
           'storage = sparse)'

# Not an important point.
def test_maple_not_supported():
    with raises(NotImplementedError):
        maple_code(S.ComplexInfinity)


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)

    assert (maple_code(A[0, 0]) == "A[1, 1]")
    assert (maple_code(3 * A[0, 0]) == "3*A[1, 1]")

    F = A-B

    assert (maple_code(F[0,0]) == "A[1, 1] - B[1, 1]")


def test_hadamard():
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    v = MatrixSymbol('v', 3, 1)
    h = MatrixSymbol('h', 1, 3)
    C = HadamardProduct(A, B)
    assert maple_code(C) == "A*B"

    assert maple_code(C * v) == "(A*B).v"
    # HadamardProduct is higher than dot product.

    assert maple_code(h * C * v) == "h.(A*B).v"

    assert maple_code(C * A) == "(A*B).A"
    # mixing Hadamard and scalar strange b/c we vectorize scalars

    assert maple_code(C * x * y) == "x*y*(A*B)"


def test_maple_piecewise():
    expr = Piecewise((x, x < 1), (x ** 2, True))

    assert maple_code(expr) == "piecewise(x < 1, x, x^2)"
    assert maple_code(expr, assign_to="r") == (
        "r := piecewise(x < 1, x, x^2)")

    expr = Piecewise((x ** 2, x < 1), (x ** 3, x < 2), (x ** 4, x < 3), (x ** 5, True))
    expected = "piecewise(x < 1, x^2, x < 2, x^3, x < 3, x^4, x^5)"
    assert maple_code(expr) == expected
    assert maple_code(expr, assign_to="r") == "r := " + expected

    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: maple_code(expr))


def test_maple_piecewise_times_const():
    pw = Piecewise((x, x < 1), (x ** 2, True))

    assert maple_code(2 * pw) == "2*piecewise(x < 1, x, x^2)"
    assert maple_code(pw / x) == "piecewise(x < 1, x, x^2)/x"
    assert maple_code(pw / (x * y)) == "piecewise(x < 1, x, x^2)/(x*y)"
    assert maple_code(pw / 3) == "(1/3)*piecewise(x < 1, x, x^2)"


def test_maple_derivatives():
    f = Function('f')
    assert maple_code(f(x).diff(x)) == 'diff(f(x), x)'
    assert maple_code(f(x).diff(x, 2)) == 'diff(f(x), x$2)'


def test_automatic_rewrites():
    assert maple_code(lucas(x)) == '(2^(-x)*((1 - sqrt(5))^x + (1 + sqrt(5))^x))'
    assert maple_code(sinc(x)) == '(piecewise(x <> 0, sin(x)/x, 1))'


def test_specfun():
    assert maple_code('asin(x)') == 'arcsin(x)'
    assert maple_code(besseli(x, y)) == 'BesselI(x, y)'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_numpy.py ===
from sympy.concrete.summations import Sum
from sympy.core.mod import Mod
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import polygamma
from sympy.functions.special.error_functions import (Si, Ci)
from sympy.matrices import Matrix
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.utilities.lambdify import lambdify
from sympy import symbols, Min, Max

from sympy.abc import x, i, j, a, b, c, d
from sympy.core import Pow
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.cfunctions import log1p, expm1, hypot, log10, exp2, log2, Sqrt
from sympy.tensor.array import Array
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
    PermuteDims, ArrayDiagonal
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter, _numpy_known_constants, \
    _numpy_known_functions, _scipy_known_constants, _scipy_known_functions
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array

from sympy.testing.pytest import skip, raises
from sympy.external import import_module

np = import_module('numpy')
jax = import_module('jax')

if np:
    deafult_float_info = np.finfo(np.array([]).dtype)
    NUMPY_DEFAULT_EPSILON = deafult_float_info.eps

def test_numpy_piecewise_regression():
    """
    NumPyPrinter needs to print Piecewise()'s choicelist as a list to avoid
    breaking compatibility with numpy 1.8. This is not necessary in numpy 1.9+.
    See gh-9747 and gh-9749 for details.
    """
    printer = NumPyPrinter()
    p = Piecewise((1, x < 0), (0, True))
    assert printer.doprint(p) == \
        'numpy.select([numpy.less(x, 0),True], [1,0], default=numpy.nan)'
    assert printer.module_imports == {'numpy': {'select', 'less', 'nan'}}

def test_numpy_logaddexp():
    lae = logaddexp(a, b)
    assert NumPyPrinter().doprint(lae) == 'numpy.logaddexp(a, b)'
    lae2 = logaddexp2(a, b)
    assert NumPyPrinter().doprint(lae2) == 'numpy.logaddexp2(a, b)'


def test_sum():
    if not np:
        skip("NumPy not installed")

    s = Sum(x ** i, (i, a, b))
    f = lambdify((a, b, x), s, 'numpy')

    a_, b_ = 0, 10
    x_ = np.linspace(-1, +1, 10)
    assert np.allclose(f(a_, b_, x_), sum(x_ ** i_ for i_ in range(a_, b_ + 1)))

    s = Sum(i * x, (i, a, b))
    f = lambdify((a, b, x), s, 'numpy')

    a_, b_ = 0, 10
    x_ = np.linspace(-1, +1, 10)
    assert np.allclose(f(a_, b_, x_), sum(i_ * x_ for i_ in range(a_, b_ + 1)))


def test_multiple_sums():
    if not np:
        skip("NumPy not installed")

    s = Sum((x + j) * i, (i, a, b), (j, c, d))
    f = lambdify((a, b, c, d, x), s, 'numpy')

    a_, b_ = 0, 10
    c_, d_ = 11, 21
    x_ = np.linspace(-1, +1, 10)
    assert np.allclose(f(a_, b_, c_, d_, x_),
                       sum((x_ + j_) * i_ for i_ in range(a_, b_ + 1) for j_ in range(c_, d_ + 1)))


def test_codegen_einsum():
    if not np:
        skip("NumPy not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)

    cg = convert_matrix_to_array(M * N)
    f = lambdify((M, N), cg, 'numpy')

    ma = np.array([[1, 2], [3, 4]])
    mb = np.array([[1,-2], [-1, 3]])
    assert (f(ma, mb) == np.matmul(ma, mb)).all()


def test_codegen_extra():
    if not np:
        skip("NumPy not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)
    P = MatrixSymbol("P", 2, 2)
    Q = MatrixSymbol("Q", 2, 2)
    ma = np.array([[1, 2], [3, 4]])
    mb = np.array([[1,-2], [-1, 3]])
    mc = np.array([[2, 0], [1, 2]])
    md = np.array([[1,-1], [4, 7]])

    cg = ArrayTensorProduct(M, N)
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.einsum(ma, [0, 1], mb, [2, 3])).all()

    cg = ArrayAdd(M, N)
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == ma+mb).all()

    cg = ArrayAdd(M, N, P)
    f = lambdify((M, N, P), cg, 'numpy')
    assert (f(ma, mb, mc) == ma+mb+mc).all()

    cg = ArrayAdd(M, N, P, Q)
    f = lambdify((M, N, P, Q), cg, 'numpy')
    assert (f(ma, mb, mc, md) == ma+mb+mc+md).all()

    cg = PermuteDims(M, [1, 0])
    f = lambdify((M,), cg, 'numpy')
    assert (f(ma) == ma.T).all()

    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.transpose(np.einsum(ma, [0, 1], mb, [2, 3]), (1, 2, 3, 0))).all()

    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.diagonal(np.einsum(ma, [0, 1], mb, [2, 3]), axis1=1, axis2=2)).all()


def test_relational():
    if not np:
        skip("NumPy not installed")

    e = Equality(x, 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [False, True, False])

    e = Unequality(x, 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [True, False, True])

    e = (x < 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [True, False, False])

    e = (x <= 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [True, True, False])

    e = (x > 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [False, False, True])

    e = (x >= 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [False, True, True])


def test_mod():
    if not np:
        skip("NumPy not installed")

    e = Mod(a, b)
    f = lambdify((a, b), e)

    a_ = np.array([0, 1, 2, 3])
    b_ = 2
    assert np.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = np.array([0, 1, 2, 3])
    b_ = np.array([2, 2, 2, 2])
    assert np.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = np.array([2, 3, 4, 5])
    b_ = np.array([2, 3, 4, 5])
    assert np.array_equal(f(a_, b_), [0, 0, 0, 0])


def test_pow():
    if not np:
        skip('NumPy not installed')

    expr = Pow(2, -1, evaluate=False)
    f = lambdify([], expr, 'numpy')
    assert f() == 0.5


def test_expm1():
    if not np:
        skip("NumPy not installed")

    f = lambdify((a,), expm1(a), 'numpy')
    assert abs(f(1e-10) - 1e-10 - 5e-21) <= 1e-10 * NUMPY_DEFAULT_EPSILON


def test_log1p():
    if not np:
        skip("NumPy not installed")

    f = lambdify((a,), log1p(a), 'numpy')
    assert abs(f(1e-99) - 1e-99) <= 1e-99 * NUMPY_DEFAULT_EPSILON

def test_hypot():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a, b), hypot(a, b), 'numpy')(3, 4) - 5) <= NUMPY_DEFAULT_EPSILON

def test_log10():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), log10(a), 'numpy')(100) - 2) <= NUMPY_DEFAULT_EPSILON


def test_exp2():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), exp2(a), 'numpy')(5) - 32) <= NUMPY_DEFAULT_EPSILON


def test_log2():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), log2(a), 'numpy')(256) - 8) <= NUMPY_DEFAULT_EPSILON


def test_Sqrt():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), Sqrt(a), 'numpy')(4) - 2) <= NUMPY_DEFAULT_EPSILON


def test_sqrt():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), sqrt(a), 'numpy')(4) - 2) <= NUMPY_DEFAULT_EPSILON


def test_matsolve():
    if not np:
        skip("NumPy not installed")

    M = MatrixSymbol("M", 3, 3)
    x = MatrixSymbol("x", 3, 1)

    expr = M**(-1) * x + x
    matsolve_expr = MatrixSolve(M, x) + x

    f = lambdify((M, x), expr)
    f_matsolve = lambdify((M, x), matsolve_expr)

    m0 = np.array([[1, 2, 3], [3, 2, 5], [5, 6, 7]])
    assert np.linalg.matrix_rank(m0) == 3

    x0 = np.array([3, 4, 5])

    assert np.allclose(f_matsolve(m0, x0), f(m0, x0))


def test_16857():
    if not np:
        skip("NumPy not installed")

    a_1 = MatrixSymbol('a_1', 10, 3)
    a_2 = MatrixSymbol('a_2', 10, 3)
    a_3 = MatrixSymbol('a_3', 10, 3)
    a_4 = MatrixSymbol('a_4', 10, 3)
    A = BlockMatrix([[a_1, a_2], [a_3, a_4]])
    assert A.shape == (20, 6)

    printer = NumPyPrinter()
    assert printer.doprint(A) == 'numpy.block([[a_1, a_2], [a_3, a_4]])'


def test_issue_17006():
    if not np:
        skip("NumPy not installed")

    M = MatrixSymbol("M", 2, 2)

    f = lambdify(M, M + Identity(2))
    ma = np.array([[1, 2], [3, 4]])
    mr = np.array([[2, 2], [3, 5]])

    assert (f(ma) == mr).all()

    from sympy.core.symbol import symbols
    n = symbols('n', integer=True)
    N = MatrixSymbol("M", n, n)
    raises(NotImplementedError, lambda: lambdify(N, N + Identity(n)))

def test_jax_tuple_compatibility():
    if not jax:
        skip("Jax not installed")

    x, y, z = symbols('x y z')
    expr = Max(x, y, z) + Min(x, y, z)
    func = lambdify((x, y, z), expr, 'jax')
    input_tuple1, input_tuple2 = (1, 2, 3), (4, 5, 6)
    input_array1, input_array2 = jax.numpy.asarray(input_tuple1), jax.numpy.asarray(input_tuple2)
    assert np.allclose(func(*input_tuple1), func(*input_array1))
    assert np.allclose(func(*input_tuple2), func(*input_array2))

def test_numpy_array():
    p = NumPyPrinter()
    assert p.doprint(Array([[1, 2], [3, 5]])) == 'numpy.array([[1, 2], [3, 5]])'
    assert p.doprint(Array([1, 2])) == 'numpy.array([1, 2])'
    assert p.doprint(Array([[[1, 2, 3]]])) == 'numpy.array([[[1, 2, 3]]])'
    assert p.doprint(Array([], (0,))) == 'numpy.zeros((0,))'
    assert p.doprint(Array([], (0, 0))) == 'numpy.zeros((0, 0))'
    assert p.doprint(Array([], (0, 1))) == 'numpy.zeros((0, 1))'
    assert p.doprint(Array([], (1, 0))) == 'numpy.zeros((1, 0))'
    assert p.doprint(Array([1], ())) == 'numpy.array(1)'

def test_numpy_matrix():
    p = NumPyPrinter()
    assert p.doprint(Matrix([[1, 2], [3, 5]])) == 'numpy.array([[1, 2], [3, 5]])'
    assert p.doprint(Matrix([1, 2])) == 'numpy.array([[1], [2]])'
    assert p.doprint(Matrix(0, 0, [])) == 'numpy.zeros((0, 0))'
    assert p.doprint(Matrix(0, 1, [])) == 'numpy.zeros((0, 1))'
    assert p.doprint(Matrix(1, 0, [])) == 'numpy.zeros((1, 0))'

def test_numpy_known_funcs_consts():
    assert _numpy_known_constants['NaN'] == 'numpy.nan'
    assert _numpy_known_constants['EulerGamma'] == 'numpy.euler_gamma'

    assert _numpy_known_functions['acos'] == 'numpy.arccos'
    assert _numpy_known_functions['log'] == 'numpy.log'

def test_scipy_known_funcs_consts():
    assert _scipy_known_constants['GoldenRatio'] == 'scipy.constants.golden_ratio'
    assert _scipy_known_constants['Pi'] == 'scipy.constants.pi'

    assert _scipy_known_functions['erf'] == 'scipy.special.erf'
    assert _scipy_known_functions['factorial'] == 'scipy.special.factorial'

def test_numpy_print_methods():
    prntr = NumPyPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')

def test_scipy_print_methods():
    prntr = SciPyPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')
    assert hasattr(prntr, '_print_erf')
    assert hasattr(prntr, '_print_factorial')
    assert hasattr(prntr, '_print_chebyshevt')
    k = Symbol('k', integer=True, nonnegative=True)
    x = Symbol('x', real=True)
    assert prntr.doprint(polygamma(k, x)) == "scipy.special.polygamma(k, x)"
    assert prntr.doprint(Si(x)) == "scipy.special.sici(x)[0]"
    assert prntr.doprint(Ci(x)) == "scipy.special.sici(x)[1]"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\numeric\test_join.py ===
import numpy as np
import pytest

import pandas._testing as tm
from pandas.core.indexes.api import Index


class TestJoinInt64Index:
    def test_join_non_unique(self):
        left = Index([4, 4, 3, 3])

        joined, lidx, ridx = left.join(left, return_indexers=True)

        exp_joined = Index([4, 4, 4, 4, 3, 3, 3, 3])
        tm.assert_index_equal(joined, exp_joined)

        exp_lidx = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(lidx, exp_lidx)

        exp_ridx = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(ridx, exp_ridx)

    def test_join_inner(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64)
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64)

        # not monotonic
        res, lidx, ridx = index.join(other, how="inner", return_indexers=True)

        # no guarantee of sortedness, so sort for comparison purposes
        ind = res.argsort()
        res = res.take(ind)
        lidx = lidx.take(ind)
        ridx = ridx.take(ind)

        eres = Index([2, 12], dtype=np.int64)
        elidx = np.array([1, 6], dtype=np.intp)
        eridx = np.array([4, 1], dtype=np.intp)

        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

        # monotonic
        res, lidx, ridx = index.join(other_mono, how="inner", return_indexers=True)

        res2 = index.intersection(other_mono)
        tm.assert_index_equal(res, res2)

        elidx = np.array([1, 6], dtype=np.intp)
        eridx = np.array([1, 4], dtype=np.intp)
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_left(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64)
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64)

        # not monotonic
        res, lidx, ridx = index.join(other, how="left", return_indexers=True)
        eres = index
        eridx = np.array([-1, 4, -1, -1, -1, -1, 1, -1, -1, -1], dtype=np.intp)

        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)

        # monotonic
        res, lidx, ridx = index.join(other_mono, how="left", return_indexers=True)
        eridx = np.array([-1, 1, -1, -1, -1, -1, 4, -1, -1, -1], dtype=np.intp)
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)

        # non-unique
        idx = Index([1, 1, 2, 5])
        idx2 = Index([1, 2, 5, 7, 9])
        res, lidx, ridx = idx2.join(idx, how="left", return_indexers=True)
        eres = Index([1, 1, 2, 5, 7, 9])  # 1 is in idx2, so it should be x2
        eridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        elidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_right(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64)
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64)

        # not monotonic
        res, lidx, ridx = index.join(other, how="right", return_indexers=True)
        eres = other
        elidx = np.array([-1, 6, -1, -1, 1, -1], dtype=np.intp)

        assert isinstance(other, Index) and other.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        assert ridx is None

        # monotonic
        res, lidx, ridx = index.join(other_mono, how="right", return_indexers=True)
        eres = other_mono
        elidx = np.array([-1, 1, -1, -1, 6, -1], dtype=np.intp)
        assert isinstance(other, Index) and other.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        assert ridx is None

        # non-unique
        idx = Index([1, 1, 2, 5])
        idx2 = Index([1, 2, 5, 7, 9])
        res, lidx, ridx = idx.join(idx2, how="right", return_indexers=True)
        eres = Index([1, 1, 2, 5, 7, 9])  # 1 is in idx2, so it should be x2
        elidx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        eridx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_non_int_index(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([3, 6, 7, 8, 10], dtype=object)

        outer = index.join(other, how="outer")
        outer2 = other.join(index, how="outer")
        expected = Index([0, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 18])
        tm.assert_index_equal(outer, outer2)
        tm.assert_index_equal(outer, expected)

        inner = index.join(other, how="inner")
        inner2 = other.join(index, how="inner")
        expected = Index([6, 8, 10])
        tm.assert_index_equal(inner, inner2)
        tm.assert_index_equal(inner, expected)

        left = index.join(other, how="left")
        tm.assert_index_equal(left, index.astype(object))

        left2 = other.join(index, how="left")
        tm.assert_index_equal(left2, other)

        right = index.join(other, how="right")
        tm.assert_index_equal(right, other)

        right2 = other.join(index, how="right")
        tm.assert_index_equal(right2, index.astype(object))

    def test_join_outer(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        other = Index([7, 12, 25, 1, 2, 5], dtype=np.int64)
        other_mono = Index([1, 2, 5, 7, 12, 25], dtype=np.int64)

        # not monotonic
        # guarantee of sortedness
        res, lidx, ridx = index.join(other, how="outer", return_indexers=True)
        noidx_res = index.join(other, how="outer")
        tm.assert_index_equal(res, noidx_res)

        eres = Index([0, 1, 2, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 25], dtype=np.int64)
        elidx = np.array([0, -1, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, 9, -1], dtype=np.intp)
        eridx = np.array(
            [-1, 3, 4, -1, 5, -1, 0, -1, -1, 1, -1, -1, -1, 2], dtype=np.intp
        )

        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

        # monotonic
        res, lidx, ridx = index.join(other_mono, how="outer", return_indexers=True)
        noidx_res = index.join(other_mono, how="outer")
        tm.assert_index_equal(res, noidx_res)

        elidx = np.array([0, -1, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, 9, -1], dtype=np.intp)
        eridx = np.array(
            [-1, 0, 1, -1, 2, -1, 3, -1, -1, 4, -1, -1, -1, 5], dtype=np.intp
        )
        assert isinstance(res, Index) and res.dtype == np.int64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)


class TestJoinUInt64Index:
    @pytest.fixture
    def index_large(self):
        # large values used in TestUInt64Index where no compat needed with int64/float64
        large = [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25]
        return Index(large, dtype=np.uint64)

    def test_join_inner(self, index_large):
        other = Index(2**63 + np.array([7, 12, 25, 1, 2, 10], dtype="uint64"))
        other_mono = Index(2**63 + np.array([1, 2, 7, 10, 12, 25], dtype="uint64"))

        # not monotonic
        res, lidx, ridx = index_large.join(other, how="inner", return_indexers=True)

        # no guarantee of sortedness, so sort for comparison purposes
        ind = res.argsort()
        res = res.take(ind)
        lidx = lidx.take(ind)
        ridx = ridx.take(ind)

        eres = Index(2**63 + np.array([10, 25], dtype="uint64"))
        elidx = np.array([1, 4], dtype=np.intp)
        eridx = np.array([5, 2], dtype=np.intp)

        assert isinstance(res, Index) and res.dtype == np.uint64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

        # monotonic
        res, lidx, ridx = index_large.join(
            other_mono, how="inner", return_indexers=True
        )

        res2 = index_large.intersection(other_mono)
        tm.assert_index_equal(res, res2)

        elidx = np.array([1, 4], dtype=np.intp)
        eridx = np.array([3, 5], dtype=np.intp)

        assert isinstance(res, Index) and res.dtype == np.uint64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_left(self, index_large):
        other = Index(2**63 + np.array([7, 12, 25, 1, 2, 10], dtype="uint64"))
        other_mono = Index(2**63 + np.array([1, 2, 7, 10, 12, 25], dtype="uint64"))

        # not monotonic
        res, lidx, ridx = index_large.join(other, how="left", return_indexers=True)
        eres = index_large
        eridx = np.array([-1, 5, -1, -1, 2], dtype=np.intp)

        assert isinstance(res, Index) and res.dtype == np.uint64
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)

        # monotonic
        res, lidx, ridx = index_large.join(other_mono, how="left", return_indexers=True)
        eridx = np.array([-1, 3, -1, -1, 5], dtype=np.intp)

        assert isinstance(res, Index) and res.dtype == np.uint64
        tm.assert_index_equal(res, eres)
        assert lidx is None
        tm.assert_numpy_array_equal(ridx, eridx)

        # non-unique
        idx = Index(2**63 + np.array([1, 1, 2, 5], dtype="uint64"))
        idx2 = Index(2**63 + np.array([1, 2, 5, 7, 9], dtype="uint64"))
        res, lidx, ridx = idx2.join(idx, how="left", return_indexers=True)

        # 1 is in idx2, so it should be x2
        eres = Index(2**63 + np.array([1, 1, 2, 5, 7, 9], dtype="uint64"))
        eridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        elidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)

        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_right(self, index_large):
        other = Index(2**63 + np.array([7, 12, 25, 1, 2, 10], dtype="uint64"))
        other_mono = Index(2**63 + np.array([1, 2, 7, 10, 12, 25], dtype="uint64"))

        # not monotonic
        res, lidx, ridx = index_large.join(other, how="right", return_indexers=True)
        eres = other
        elidx = np.array([-1, -1, 4, -1, -1, 1], dtype=np.intp)

        tm.assert_numpy_array_equal(lidx, elidx)
        assert isinstance(other, Index) and other.dtype == np.uint64
        tm.assert_index_equal(res, eres)
        assert ridx is None

        # monotonic
        res, lidx, ridx = index_large.join(
            other_mono, how="right", return_indexers=True
        )
        eres = other_mono
        elidx = np.array([-1, -1, -1, 1, -1, 4], dtype=np.intp)

        assert isinstance(other, Index) and other.dtype == np.uint64
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_index_equal(res, eres)
        assert ridx is None

        # non-unique
        idx = Index(2**63 + np.array([1, 1, 2, 5], dtype="uint64"))
        idx2 = Index(2**63 + np.array([1, 2, 5, 7, 9], dtype="uint64"))
        res, lidx, ridx = idx.join(idx2, how="right", return_indexers=True)

        # 1 is in idx2, so it should be x2
        eres = Index(2**63 + np.array([1, 1, 2, 5, 7, 9], dtype="uint64"))
        elidx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
        eridx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)

        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

    def test_join_non_int_index(self, index_large):
        other = Index(
            2**63 + np.array([1, 5, 7, 10, 20], dtype="uint64"), dtype=object
        )

        outer = index_large.join(other, how="outer")
        outer2 = other.join(index_large, how="outer")
        expected = Index(
            2**63 + np.array([0, 1, 5, 7, 10, 15, 20, 25], dtype="uint64")
        )
        tm.assert_index_equal(outer, outer2)
        tm.assert_index_equal(outer, expected)

        inner = index_large.join(other, how="inner")
        inner2 = other.join(index_large, how="inner")
        expected = Index(2**63 + np.array([10, 20], dtype="uint64"))
        tm.assert_index_equal(inner, inner2)
        tm.assert_index_equal(inner, expected)

        left = index_large.join(other, how="left")
        tm.assert_index_equal(left, index_large.astype(object))

        left2 = other.join(index_large, how="left")
        tm.assert_index_equal(left2, other)

        right = index_large.join(other, how="right")
        tm.assert_index_equal(right, other)

        right2 = other.join(index_large, how="right")
        tm.assert_index_equal(right2, index_large.astype(object))

    def test_join_outer(self, index_large):
        other = Index(2**63 + np.array([7, 12, 25, 1, 2, 10], dtype="uint64"))
        other_mono = Index(2**63 + np.array([1, 2, 7, 10, 12, 25], dtype="uint64"))

        # not monotonic
        # guarantee of sortedness
        res, lidx, ridx = index_large.join(other, how="outer", return_indexers=True)
        noidx_res = index_large.join(other, how="outer")
        tm.assert_index_equal(res, noidx_res)

        eres = Index(
            2**63 + np.array([0, 1, 2, 7, 10, 12, 15, 20, 25], dtype="uint64")
        )
        elidx = np.array([0, -1, -1, -1, 1, -1, 2, 3, 4], dtype=np.intp)
        eridx = np.array([-1, 3, 4, 0, 5, 1, -1, -1, 2], dtype=np.intp)

        assert isinstance(res, Index) and res.dtype == np.uint64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)

        # monotonic
        res, lidx, ridx = index_large.join(
            other_mono, how="outer", return_indexers=True
        )
        noidx_res = index_large.join(other_mono, how="outer")
        tm.assert_index_equal(res, noidx_res)

        elidx = np.array([0, -1, -1, -1, 1, -1, 2, 3, 4], dtype=np.intp)
        eridx = np.array([-1, 0, 1, 2, 3, 4, -1, -1, 5], dtype=np.intp)

        assert isinstance(res, Index) and res.dtype == np.uint64
        tm.assert_index_equal(res, eres)
        tm.assert_numpy_array_equal(lidx, elidx)
        tm.assert_numpy_array_equal(ridx, eridx)