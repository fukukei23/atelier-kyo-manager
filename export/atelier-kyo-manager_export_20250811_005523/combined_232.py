
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\internals\test_managers.py ===
"""
Testing interaction between the different managers (BlockManager, ArrayManager)
"""
import os
import subprocess
import sys

import pytest

from pandas.core.dtypes.missing import array_equivalent

import pandas as pd
import pandas._testing as tm
from pandas.core.internals import (
    ArrayManager,
    BlockManager,
    SingleArrayManager,
    SingleBlockManager,
)


def test_dataframe_creation():
    msg = "data_manager option is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context("mode.data_manager", "block"):
            df_block = pd.DataFrame(
                {"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": [4, 5, 6]}
            )
    assert isinstance(df_block._mgr, BlockManager)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context("mode.data_manager", "array"):
            df_array = pd.DataFrame(
                {"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": [4, 5, 6]}
            )
    assert isinstance(df_array._mgr, ArrayManager)

    # also ensure both are seen as equal
    tm.assert_frame_equal(df_block, df_array)

    # conversion from one manager to the other
    result = df_block._as_manager("block")
    assert isinstance(result._mgr, BlockManager)
    result = df_block._as_manager("array")
    assert isinstance(result._mgr, ArrayManager)
    tm.assert_frame_equal(result, df_block)
    assert all(
        array_equivalent(left, right)
        for left, right in zip(result._mgr.arrays, df_array._mgr.arrays)
    )

    result = df_array._as_manager("array")
    assert isinstance(result._mgr, ArrayManager)
    result = df_array._as_manager("block")
    assert isinstance(result._mgr, BlockManager)
    tm.assert_frame_equal(result, df_array)
    assert len(result._mgr.blocks) == 2


def test_series_creation():
    msg = "data_manager option is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context("mode.data_manager", "block"):
            s_block = pd.Series([1, 2, 3], name="A", index=["a", "b", "c"])
    assert isinstance(s_block._mgr, SingleBlockManager)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context("mode.data_manager", "array"):
            s_array = pd.Series([1, 2, 3], name="A", index=["a", "b", "c"])
    assert isinstance(s_array._mgr, SingleArrayManager)

    # also ensure both are seen as equal
    tm.assert_series_equal(s_block, s_array)

    # conversion from one manager to the other
    result = s_block._as_manager("block")
    assert isinstance(result._mgr, SingleBlockManager)
    result = s_block._as_manager("array")
    assert isinstance(result._mgr, SingleArrayManager)
    tm.assert_series_equal(result, s_block)

    result = s_array._as_manager("array")
    assert isinstance(result._mgr, SingleArrayManager)
    result = s_array._as_manager("block")
    assert isinstance(result._mgr, SingleBlockManager)
    tm.assert_series_equal(result, s_array)


@pytest.mark.single_cpu
@pytest.mark.parametrize("manager", ["block", "array"])
def test_array_manager_depr_env_var(manager):
    # GH#55043
    test_env = os.environ.copy()
    test_env["PANDAS_DATA_MANAGER"] = manager
    response = subprocess.run(
        [sys.executable, "-c", "import pandas"],
        capture_output=True,
        env=test_env,
        check=True,
    )
    msg = "FutureWarning: The env variable PANDAS_DATA_MANAGER is set"
    stderr_msg = response.stderr.decode("utf-8")
    assert msg in stderr_msg, stderr_msg

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_dagger.py ===
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import conjugate
from sympy.matrices.dense import Matrix

from sympy.physics.quantum.dagger import adjoint, Dagger
from sympy.external import import_module
from sympy.testing.pytest import skip, warns_deprecated_sympy
from sympy.physics.quantum.operator import Operator, IdentityOperator


def test_scalars():
    x = symbols('x', complex=True)
    assert Dagger(x) == conjugate(x)
    assert Dagger(I*x) == -I*conjugate(x)

    i = symbols('i', real=True)
    assert Dagger(i) == i

    p = symbols('p')
    assert isinstance(Dagger(p), conjugate)

    i = Integer(3)
    assert Dagger(i) == i

    A = symbols('A', commutative=False)
    assert Dagger(A).is_commutative is False


def test_matrix():
    x = symbols('x')
    m = Matrix([[I, x*I], [2, 4]])
    assert Dagger(m) == m.H


def test_dagger_mul():
    O = Operator('O')
    assert Dagger(O)*O == Dagger(O)*O
    with warns_deprecated_sympy():
        I = IdentityOperator()
        assert Dagger(O)*O*I == Mul(Dagger(O), O)*I
    assert Dagger(O)*Dagger(O) == Dagger(O)**2
    assert Dagger(O)*Dagger(I) == Dagger(O)


class Foo(Expr):

    def _eval_adjoint(self):
        return I


def test_eval_adjoint():
    f = Foo()
    d = Dagger(f)
    assert d == I

np = import_module('numpy')


def test_numpy_dagger():
    if not np:
        skip("numpy not installed.")

    a = np.array([[1.0, 2.0j], [-1.0j, 2.0]])
    adag = a.copy().transpose().conjugate()
    assert (Dagger(a) == adag).all()


scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})


def test_scipy_sparse_dagger():
    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")
    else:
        sparse = scipy.sparse

    a = sparse.csr_matrix([[1.0 + 0.0j, 2.0j], [-1.0j, 2.0 + 0.0j]])
    adag = a.copy().transpose().conjugate()
    assert np.linalg.norm((Dagger(a) - adag).todense()) == 0.0


def test_unknown():
    """Check treatment of unknown objects.
    Objects without adjoint or conjugate/transpose methods
    are sympified and wrapped in dagger.
    """
    x = symbols("x", commutative=False)
    result = Dagger(x)
    assert result.args == (x,) and isinstance(result, adjoint)


def test_unevaluated():
    """Check that evaluate=False returns unevaluated Dagger.
    """
    x = symbols("x", real=True)
    assert Dagger(x) == x
    result = Dagger(x, evaluate=False)
    assert result.args == (x,) and isinstance(result, adjoint)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\test_runtime.py ===
"""Test the runtime usage of `numpy.typing`."""

from typing import (
    Any,
    NamedTuple,
    Union,  # pyright: ignore[reportDeprecated]
    get_args,
    get_origin,
    get_type_hints,
)

import pytest

import numpy as np
import numpy._typing as _npt
import numpy.typing as npt


class TypeTup(NamedTuple):
    typ: type
    args: tuple[type, ...]
    origin: type | None


NDArrayTup = TypeTup(npt.NDArray, npt.NDArray.__args__, np.ndarray)

TYPES = {
    "ArrayLike": TypeTup(npt.ArrayLike, npt.ArrayLike.__args__, Union),
    "DTypeLike": TypeTup(npt.DTypeLike, npt.DTypeLike.__args__, Union),
    "NBitBase": TypeTup(npt.NBitBase, (), None),
    "NDArray": NDArrayTup,
}


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_args(name: type, tup: TypeTup) -> None:
    """Test `typing.get_args`."""
    typ, ref = tup.typ, tup.args
    out = get_args(typ)
    assert out == ref


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_origin(name: type, tup: TypeTup) -> None:
    """Test `typing.get_origin`."""
    typ, ref = tup.typ, tup.origin
    out = get_origin(typ)
    assert out == ref


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_type_hints(name: type, tup: TypeTup) -> None:
    """Test `typing.get_type_hints`."""
    typ = tup.typ

    def func(a: typ) -> None: pass

    out = get_type_hints(func)
    ref = {"a": typ, "return": type(None)}
    assert out == ref


@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_type_hints_str(name: type, tup: TypeTup) -> None:
    """Test `typing.get_type_hints` with string-representation of types."""
    typ_str, typ = f"npt.{name}", tup.typ

    def func(a: typ_str) -> None: pass

    out = get_type_hints(func)
    ref = {"a": typ, "return": type(None)}
    assert out == ref


def test_keys() -> None:
    """Test that ``TYPES.keys()`` and ``numpy.typing.__all__`` are synced."""
    keys = TYPES.keys()
    ref = set(npt.__all__)
    assert keys == ref


PROTOCOLS: dict[str, tuple[type[Any], object]] = {
    "_SupportsDType": (_npt._SupportsDType, np.int64(1)),
    "_SupportsArray": (_npt._SupportsArray, np.arange(10)),
    "_SupportsArrayFunc": (_npt._SupportsArrayFunc, np.arange(10)),
    "_NestedSequence": (_npt._NestedSequence, [1]),
}


@pytest.mark.parametrize("cls,obj", PROTOCOLS.values(), ids=PROTOCOLS.keys())
class TestRuntimeProtocol:
    def test_isinstance(self, cls: type[Any], obj: object) -> None:
        assert isinstance(obj, cls)
        assert not isinstance(None, cls)

    def test_issubclass(self, cls: type[Any], obj: object) -> None:
        if cls is _npt._SupportsDType:
            pytest.xfail(
                "Protocols with non-method members don't support issubclass()"
            )
        assert issubclass(type(obj), cls)
        assert not issubclass(type(None), cls)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_limited_api.py ===
import os
import subprocess
import sys
import sysconfig

import pytest

from numpy.testing import IS_EDITABLE, IS_PYPY, IS_WASM, NOGIL_BUILD

# This import is copied from random.tests.test_extending
try:
    import cython
    from Cython.Compiler.Version import version as cython_version
except ImportError:
    cython = None
else:
    from numpy._utils import _pep440

    # Note: keep in sync with the one in pyproject.toml
    required_version = "3.0.6"
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # too old or wrong cython, skip the test
        cython = None

pytestmark = pytest.mark.skipif(cython is None, reason="requires cython")


if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )


@pytest.fixture(scope='module')
def install_temp(tmpdir_factory):
    # Based in part on test_cython from random.tests.test_extending
    if IS_WASM:
        pytest.skip("No subprocess")

    srcdir = os.path.join(os.path.dirname(__file__), 'examples', 'limited_api')
    build_dir = tmpdir_factory.mktemp("limited_api") / "build"
    os.makedirs(build_dir, exist_ok=True)
    # Ensure we use the correct Python interpreter even when `meson` is
    # installed in a different Python environment (see gh-24956)
    native_file = str(build_dir / 'interpreter-native-file.ini')
    with open(native_file, 'w') as f:
        f.write("[binaries]\n")
        f.write(f"python = '{sys.executable}'\n")
        f.write(f"python3 = '{sys.executable}'")

    try:
        subprocess.check_call(["meson", "--version"])
    except FileNotFoundError:
        pytest.skip("No usable 'meson' found")
    if sysconfig.get_platform() == "win-arm64":
        pytest.skip("Meson unable to find MSVC linker on win-arm64")
    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--werror",
                               "--buildtype=release",
                               "--vsenv", "--native-file", native_file,
                               str(srcdir)],
                              cwd=build_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup", "--werror",
                               "--native-file", native_file, str(srcdir)],
                              cwd=build_dir
                              )
    try:
        subprocess.check_call(
            ["meson", "compile", "-vv"], cwd=build_dir)
    except subprocess.CalledProcessError as p:
        print(f"{p.stdout=}")
        print(f"{p.stderr=}")
        raise

    sys.path.append(str(build_dir))


@pytest.mark.skipif(IS_WASM, reason="Can't start subprocess")
@pytest.mark.xfail(
    sysconfig.get_config_var("Py_DEBUG"),
    reason=(
        "Py_LIMITED_API is incompatible with Py_DEBUG, Py_TRACE_REFS, "
        "and Py_REF_DEBUG"
    ),
)
@pytest.mark.xfail(
    NOGIL_BUILD,
    reason="Py_GIL_DISABLED builds do not currently support the limited API",
)
@pytest.mark.skipif(IS_PYPY, reason="no support for limited API in PyPy")
def test_limited_api(install_temp):
    """Test building a third-party C extension with the limited API
    and building a cython extension with the limited API
    """

    import limited_api1  # Earliest (3.6)  # noqa: F401
    import limited_api2  # cython  # noqa: F401
    import limited_api_latest  # Latest version (current Python)  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_upcast.py ===
import numpy as np
import pytest

from pandas._libs.parsers import (
    _maybe_upcast,
    na_values,
)

import pandas as pd
from pandas import NA
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowStringArray,
    BooleanArray,
    FloatingArray,
    IntegerArray,
    StringArray,
)


def test_maybe_upcast(any_real_numpy_dtype):
    # GH#36712

    dtype = np.dtype(any_real_numpy_dtype)
    na_value = na_values[dtype]
    arr = np.array([1, 2, na_value], dtype=dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([False, False, True])
    if issubclass(dtype.type, np.integer):
        expected = IntegerArray(arr, mask=expected_mask)
    else:
        expected = FloatingArray(arr, mask=expected_mask)

    tm.assert_extension_array_equal(result, expected)


def test_maybe_upcast_no_na(any_real_numpy_dtype):
    # GH#36712
    arr = np.array([1, 2, 3], dtype=any_real_numpy_dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([False, False, False])
    if issubclass(np.dtype(any_real_numpy_dtype).type, np.integer):
        expected = IntegerArray(arr, mask=expected_mask)
    else:
        expected = FloatingArray(arr, mask=expected_mask)

    tm.assert_extension_array_equal(result, expected)


def test_maybe_upcaste_bool():
    # GH#36712
    dtype = np.bool_
    na_value = na_values[dtype]
    arr = np.array([True, False, na_value], dtype="uint8").view(dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([False, False, True])
    expected = BooleanArray(arr, mask=expected_mask)
    tm.assert_extension_array_equal(result, expected)


def test_maybe_upcaste_bool_no_nan():
    # GH#36712
    dtype = np.bool_
    arr = np.array([True, False, False], dtype="uint8").view(dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([False, False, False])
    expected = BooleanArray(arr, mask=expected_mask)
    tm.assert_extension_array_equal(result, expected)


def test_maybe_upcaste_all_nan():
    # GH#36712
    dtype = np.int64
    na_value = na_values[dtype]
    arr = np.array([na_value, na_value], dtype=dtype)
    result = _maybe_upcast(arr, use_dtype_backend=True)

    expected_mask = np.array([True, True])
    expected = IntegerArray(arr, mask=expected_mask)
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("val", [na_values[np.object_], "c"])
def test_maybe_upcast_object(val, string_storage):
    # GH#36712
    pa = pytest.importorskip("pyarrow")

    with pd.option_context("mode.string_storage", string_storage):
        arr = np.array(["a", "b", val], dtype=np.object_)
        result = _maybe_upcast(arr, use_dtype_backend=True)

        if string_storage == "python":
            exp_val = "c" if val == "c" else NA
            expected = StringArray(np.array(["a", "b", exp_val], dtype=np.object_))
        else:
            exp_val = "c" if val == "c" else None
            expected = ArrowStringArray(pa.array(["a", "b", exp_val]))
        tm.assert_extension_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\tests\test_gaussopt.py ===
from sympy.core.evalf import N
from sympy.core.numbers import (Float, I, oo, pi)
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor

from sympy.physics.optics import (BeamParameter, CurvedMirror,
  CurvedRefraction, FlatMirror, FlatRefraction, FreeSpace, GeometricRay,
  RayTransferMatrix, ThinLens, conjugate_gauss_beams,
  gaussian_conj, geometric_conj_ab, geometric_conj_af, geometric_conj_bf,
  rayleigh2waist, waist2rayleigh)


def streq(a, b):
    return str(a) == str(b)


def test_gauss_opt():
    mat = RayTransferMatrix(1, 2, 3, 4)
    assert mat == Matrix([[1, 2], [3, 4]])
    assert mat == RayTransferMatrix( Matrix([[1, 2], [3, 4]]) )
    assert [mat.A, mat.B, mat.C, mat.D] == [1, 2, 3, 4]

    d, f, h, n1, n2, R = symbols('d f h n1 n2 R')
    lens = ThinLens(f)
    assert lens == Matrix([[   1, 0], [-1/f, 1]])
    assert lens.C == -1/f
    assert FreeSpace(d) == Matrix([[ 1, d], [0, 1]])
    assert FlatRefraction(n1, n2) == Matrix([[1, 0], [0, n1/n2]])
    assert CurvedRefraction(
        R, n1, n2) == Matrix([[1, 0], [(n1 - n2)/(R*n2), n1/n2]])
    assert FlatMirror() == Matrix([[1, 0], [0, 1]])
    assert CurvedMirror(R) == Matrix([[   1, 0], [-2/R, 1]])
    assert ThinLens(f) == Matrix([[   1, 0], [-1/f, 1]])

    mul = CurvedMirror(R)*FreeSpace(d)
    mul_mat = Matrix([[   1, 0], [-2/R, 1]])*Matrix([[ 1, d], [0, 1]])
    assert mul.A == mul_mat[0, 0]
    assert mul.B == mul_mat[0, 1]
    assert mul.C == mul_mat[1, 0]
    assert mul.D == mul_mat[1, 1]

    angle = symbols('angle')
    assert GeometricRay(h, angle) == Matrix([[    h], [angle]])
    assert FreeSpace(
        d)*GeometricRay(h, angle) == Matrix([[angle*d + h], [angle]])
    assert GeometricRay( Matrix( ((h,), (angle,)) ) ) == Matrix([[h], [angle]])
    assert (FreeSpace(d)*GeometricRay(h, angle)).height == angle*d + h
    assert (FreeSpace(d)*GeometricRay(h, angle)).angle == angle

    p = BeamParameter(530e-9, 1, w=1e-3)
    assert streq(p.q, 1 + 1.88679245283019*I*pi)
    assert streq(N(p.q), 1.0 + 5.92753330865999*I)
    assert streq(N(p.w_0), Float(0.00100000000000000))
    assert streq(N(p.z_r), Float(5.92753330865999))
    fs = FreeSpace(10)
    p1 = fs*p
    assert streq(N(p.w), Float(0.00101413072159615))
    assert streq(N(p1.w), Float(0.00210803120913829))

    w, wavelen = symbols('w wavelen')
    assert waist2rayleigh(w, wavelen) == pi*w**2/wavelen
    z_r, wavelen = symbols('z_r wavelen')
    assert rayleigh2waist(z_r, wavelen) == sqrt(wavelen*z_r)/sqrt(pi)

    a, b, f = symbols('a b f')
    assert geometric_conj_ab(a, b) == a*b/(a + b)
    assert geometric_conj_af(a, f) == a*f/(a - f)
    assert geometric_conj_bf(b, f) == b*f/(b - f)
    assert geometric_conj_ab(oo, b) == b
    assert geometric_conj_ab(a, oo) == a

    s_in, z_r_in, f = symbols('s_in z_r_in f')
    assert gaussian_conj(
        s_in, z_r_in, f)[0] == 1/(-1/(s_in + z_r_in**2/(-f + s_in)) + 1/f)
    assert gaussian_conj(
        s_in, z_r_in, f)[1] == z_r_in/(1 - s_in**2/f**2 + z_r_in**2/f**2)
    assert gaussian_conj(
        s_in, z_r_in, f)[2] == 1/sqrt(1 - s_in**2/f**2 + z_r_in**2/f**2)

    l, w_i, w_o, f = symbols('l w_i w_o f')
    assert conjugate_gauss_beams(l, w_i, w_o, f=f)[0] == f*(
        -sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)) + 1)
    assert factor(conjugate_gauss_beams(l, w_i, w_o, f=f)[1]) == f*w_o**2*(
        w_i**2/w_o**2 - sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)))/w_i**2
    assert conjugate_gauss_beams(l, w_i, w_o, f=f)[2] == f

    z, l, w_0 = symbols('z l w_0', positive=True)
    p = BeamParameter(l, z, w=w_0)
    assert p.radius == z*(pi**2*w_0**4/(l**2*z**2) + 1)
    assert p.w == w_0*sqrt(l**2*z**2/(pi**2*w_0**4) + 1)
    assert p.w_0 == w_0
    assert p.divergence == l/(pi*w_0)
    assert p.gouy == atan2(z, pi*w_0**2/l)
    assert p.waist_approximation_limit == 2*l/pi

    p = BeamParameter(530e-9, 1, w=1e-3, n=2)
    assert streq(p.q, 1 + 3.77358490566038*I*pi)
    assert streq(N(p.z_r), Float(11.8550666173200))
    assert streq(N(p.w_0), Float(0.00100000000000000))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_clip.py ===
import numpy as np

from pandas import (
    DataFrame,
    option_context,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def test_clip_inplace_reference(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1.5, 2, 3]})
    df_copy = df.copy()
    arr_a = get_array(df, "a")
    view = df[:]
    if warn_copy_on_write:
        with tm.assert_cow_warning():
            df.clip(lower=2, inplace=True)
    else:
        df.clip(lower=2, inplace=True)

    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "a"), arr_a)
        assert df._mgr._has_no_reference(0)
        assert view._mgr._has_no_reference(0)
        tm.assert_frame_equal(df_copy, view)
    else:
        assert np.shares_memory(get_array(df, "a"), arr_a)


def test_clip_inplace_reference_no_op(using_copy_on_write):
    df = DataFrame({"a": [1.5, 2, 3]})
    df_copy = df.copy()
    arr_a = get_array(df, "a")
    view = df[:]
    df.clip(lower=0, inplace=True)

    assert np.shares_memory(get_array(df, "a"), arr_a)

    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
        assert not view._mgr._has_no_reference(0)
        tm.assert_frame_equal(df_copy, view)


def test_clip_inplace(using_copy_on_write):
    df = DataFrame({"a": [1.5, 2, 3]})
    arr_a = get_array(df, "a")
    df.clip(lower=2, inplace=True)

    assert np.shares_memory(get_array(df, "a"), arr_a)

    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)


def test_clip(using_copy_on_write):
    df = DataFrame({"a": [1.5, 2, 3]})
    df_orig = df.copy()
    df2 = df.clip(lower=2)

    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
    tm.assert_frame_equal(df_orig, df)


def test_clip_no_op(using_copy_on_write):
    df = DataFrame({"a": [1.5, 2, 3]})
    df2 = df.clip(lower=0)

    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))


def test_clip_chained_inplace(using_copy_on_write):
    df = DataFrame({"a": [1, 4, 2], "b": 1})
    df_orig = df.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df["a"].clip(1, 2, inplace=True)
        tm.assert_frame_equal(df, df_orig)

        with tm.raises_chained_assignment_error():
            df[["a"]].clip(1, 2, inplace=True)
        tm.assert_frame_equal(df, df_orig)
    else:
        with tm.assert_produces_warning(FutureWarning, match="inplace method"):
            df["a"].clip(1, 2, inplace=True)

        with tm.assert_produces_warning(None):
            with option_context("mode.chained_assignment", None):
                df[["a"]].clip(1, 2, inplace=True)

        with tm.assert_produces_warning(None):
            with option_context("mode.chained_assignment", None):
                df[df["a"] > 1].clip(1, 2, inplace=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_residues.py ===
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cot, sin, tan)
from sympy.series.residues import residue
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, z, a, s, k


def test_basic1():
    assert residue(1/x, x, 0) == 1
    assert residue(-2/x, x, 0) == -2
    assert residue(81/x, x, 0) == 81
    assert residue(1/x**2, x, 0) == 0
    assert residue(0, x, 0) == 0
    assert residue(5, x, 0) == 0
    assert residue(x, x, 0) == 0
    assert residue(x**2, x, 0) == 0


def test_basic2():
    assert residue(1/x, x, 1) == 0
    assert residue(-2/x, x, 1) == 0
    assert residue(81/x, x, -1) == 0
    assert residue(1/x**2, x, 1) == 0
    assert residue(0, x, 1) == 0
    assert residue(5, x, 1) == 0
    assert residue(x, x, 1) == 0
    assert residue(x**2, x, 5) == 0


def test_f():
    f = Function("f")
    assert residue(f(x)/x**5, x, 0) == f(x).diff(x, 4).subs(x, 0)/24


def test_functions():
    assert residue(1/sin(x), x, 0) == 1
    assert residue(2/sin(x), x, 0) == 2
    assert residue(1/sin(x)**2, x, 0) == 0
    assert residue(1/sin(x)**5, x, 0) == Rational(3, 8)


def test_expressions():
    assert residue(1/(x + 1), x, 0) == 0
    assert residue(1/(x + 1), x, -1) == 1
    assert residue(1/(x**2 + 1), x, -1) == 0
    assert residue(1/(x**2 + 1), x, I) == -I/2
    assert residue(1/(x**2 + 1), x, -I) == I/2
    assert residue(1/(x**4 + 1), x, 0) == 0
    assert residue(1/(x**4 + 1), x, exp(I*pi/4)).equals(-(Rational(1, 4) + I/4)/sqrt(2))
    assert residue(1/(x**2 + a**2)**2, x, a*I) == -I/4/a**3


@XFAIL
def test_expressions_failing():
    n = Symbol('n', integer=True, positive=True)
    assert residue(exp(z)/(z - pi*I/4*a)**n, z, I*pi*a) == \
        exp(I*pi*a/4)/factorial(n - 1)


def test_NotImplemented():
    raises(NotImplementedError, lambda: residue(exp(1/z), z, 0))


def test_bug():
    assert residue(2**(z)*(s + z)*(1 - s - z)/z**2, z, 0) == \
        1 + s*log(2) - s**2*log(2) - 2*s


def test_issue_5654():
    assert residue(1/(x**2 + a**2)**2, x, a*I) == -I/(4*a**3)
    assert residue(1/s*1/(z - exp(s)), s, 0) == 1/(z - 1)
    assert residue((1 + k)/s*1/(z - exp(s)), s, 0) == k/(z - 1) + 1/(z - 1)


def test_issue_6499():
    assert residue(1/(exp(z) - 1), z, 0) == 1


def test_issue_14037():
    assert residue(sin(x**50)/x**51, x, 0) == 1


def test_issue_21176():
    f = x**2*cot(pi*x)/(x**4 + 1)
    assert residue(f, x, -sqrt(2)/2 - sqrt(2)*I/2).cancel().together(deep=True)\
        == sqrt(2)*(1 - I)/(8*tan(sqrt(2)*pi*(1 + I)/2))


def test_issue_21177():
    r = -sqrt(3)*tanh(sqrt(3)*pi/2)/3
    a = residue(cot(pi*x)/((x - 1)*(x - 2) + 1), x, S(3)/2 - sqrt(3)*I/2)
    b = residue(cot(pi*x)/(x**2 - 3*x + 3), x, S(3)/2 - sqrt(3)*I/2)
    assert a == r
    assert (b - a).cancel() == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_string.py ===
import pytest

import numpy as np

from . import util


class TestString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "string", "char.f90")]

    @pytest.mark.slow
    def test_char(self):
        strings = np.array(["ab", "cd", "ef"], dtype="c").T
        inp, out = self.module.char_test.change_strings(
            strings, strings.shape[1])
        assert inp == pytest.approx(strings)
        expected = strings.copy()
        expected[1, :] = "AAA"
        assert out == pytest.approx(expected)


class TestDocStringArguments(util.F2PyTest):
    sources = [util.getpath("tests", "src", "string", "string.f")]

    def test_example(self):
        a = np.array(b"123\0\0")
        b = np.array(b"123\0\0")
        c = np.array(b"123")
        d = np.array(b"123")

        self.module.foo(a, b, c, d)

        assert a.tobytes() == b"123\0\0"
        assert b.tobytes() == b"B23\0\0"
        assert c.tobytes() == b"123"
        assert d.tobytes() == b"D23"


class TestFixedString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "string", "fixed_string.f90")]

    @staticmethod
    def _sint(s, start=0, end=None):
        """Return the content of a string buffer as integer value.

        For example:
          _sint('1234') -> 4321
          _sint('123A') -> 17321
        """
        if isinstance(s, np.ndarray):
            s = s.tobytes()
        elif isinstance(s, str):
            s = s.encode()
        assert isinstance(s, bytes)
        if end is None:
            end = len(s)
        i = 0
        for j in range(start, min(end, len(s))):
            i += s[j] * 10**j
        return i

    def _get_input(self, intent="in"):
        if intent in ["in"]:
            yield ""
            yield "1"
            yield "1234"
            yield "12345"
            yield b""
            yield b"\0"
            yield b"1"
            yield b"\01"
            yield b"1\0"
            yield b"1234"
            yield b"12345"
        yield np.ndarray((), np.bytes_, buffer=b"")  # array(b'', dtype='|S0')
        yield np.array(b"")  # array(b'', dtype='|S1')
        yield np.array(b"\0")
        yield np.array(b"1")
        yield np.array(b"1\0")
        yield np.array(b"\01")
        yield np.array(b"1234")
        yield np.array(b"123\0")
        yield np.array(b"12345")

    def test_intent_in(self):
        for s in self._get_input():
            r = self.module.test_in_bytes4(s)
            # also checks that s is not changed inplace
            expected = self._sint(s, end=4)
            assert r == expected, s

    def test_intent_inout(self):
        for s in self._get_input(intent="inout"):
            rest = self._sint(s, start=4)
            r = self.module.test_inout_bytes4(s)
            expected = self._sint(s, end=4)
            assert r == expected

            # check that the rest of input string is preserved
            assert rest == self._sint(s, start=4)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\tests\test_regression.py ===
import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_array_equal,
    suppress_warnings,
)


class TestRegression:
    def test_masked_array_create(self):
        # Ticket #17
        x = np.ma.masked_array([0, 1, 2, 3, 0, 4, 5, 6],
                               mask=[0, 0, 0, 1, 1, 1, 0, 0])
        assert_array_equal(np.ma.nonzero(x), [[1, 2, 6, 7]])

    def test_masked_array(self):
        # Ticket #61
        np.ma.array(1, mask=[1])

    def test_mem_masked_where(self):
        # Ticket #62
        from numpy.ma import MaskType, masked_where
        a = np.zeros((1, 1))
        b = np.zeros(a.shape, MaskType)
        c = masked_where(b, a)
        a - c

    def test_masked_array_multiply(self):
        # Ticket #254
        a = np.ma.zeros((4, 1))
        a[2, 0] = np.ma.masked
        b = np.zeros((4, 2))
        a * b
        b * a

    def test_masked_array_repeat(self):
        # Ticket #271
        np.ma.array([1], mask=False).repeat(10)

    def test_masked_array_repr_unicode(self):
        # Ticket #1256
        repr(np.ma.array("Unicode"))

    def test_atleast_2d(self):
        # Ticket #1559
        a = np.ma.masked_array([0.0, 1.2, 3.5], mask=[False, True, False])
        b = np.atleast_2d(a)
        assert_(a.mask.ndim == 1)
        assert_(b.mask.ndim == 2)

    def test_set_fill_value_unicode_py3(self):
        # Ticket #2733
        a = np.ma.masked_array(['a', 'b', 'c'], mask=[1, 0, 0])
        a.fill_value = 'X'
        assert_(a.fill_value == 'X')

    def test_var_sets_maskedarray_scalar(self):
        # Issue gh-2757
        a = np.ma.array(np.arange(5), mask=True)
        mout = np.ma.array(-1, dtype=float)
        a.var(out=mout)
        assert_(mout._data == 0)

    def test_ddof_corrcoef(self):
        # See gh-3336
        x = np.ma.masked_equal([1, 2, 3, 4, 5], 4)
        y = np.array([2, 2.5, 3.1, 3, 5])
        # this test can be removed after deprecation.
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            r0 = np.ma.corrcoef(x, y, ddof=0)
            r1 = np.ma.corrcoef(x, y, ddof=1)
            # ddof should not have an effect (it gets cancelled out)
            assert_allclose(r0.data, r1.data)

    def test_mask_not_backmangled(self):
        # See gh-10314.  Test case taken from gh-3140.
        a = np.ma.MaskedArray([1., 2.], mask=[False, False])
        assert_(a.mask.shape == (2,))
        b = np.tile(a, (2, 1))
        # Check that the above no longer changes a.shape to (1, 2)
        assert_(a.mask.shape == (2,))
        assert_(b.shape == (2, 2))
        assert_(b.mask.shape == (2, 2))

    def test_empty_list_on_structured(self):
        # See gh-12464. Indexing with empty list should give empty result.
        ma = np.ma.MaskedArray([(1, 1.), (2, 2.), (3, 3.)], dtype='i4,f4')
        assert_array_equal(ma[[]], ma[:0])

    def test_masked_array_tobytes_fortran(self):
        ma = np.ma.arange(4).reshape((2, 2))
        assert_array_equal(ma.tobytes(order='F'), ma.T.tobytes())

    def test_structured_array(self):
        # see gh-22041
        np.ma.array((1, (b"", b"")),
                    dtype=[("x", np.int_),
                          ("y", [("i", np.void), ("j", np.void)])])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_optional_dependency.py ===
import sys
import types

import pytest

from pandas.compat._optional import (
    VERSIONS,
    import_optional_dependency,
)

import pandas._testing as tm


def test_import_optional():
    match = "Missing .*notapackage.* pip .* conda .* notapackage"
    with pytest.raises(ImportError, match=match) as exc_info:
        import_optional_dependency("notapackage")
    # The original exception should be there as context:
    assert isinstance(exc_info.value.__context__, ImportError)

    result = import_optional_dependency("notapackage", errors="ignore")
    assert result is None


def test_xlrd_version_fallback():
    pytest.importorskip("xlrd")
    import_optional_dependency("xlrd")


def test_bad_version(monkeypatch):
    name = "fakemodule"
    module = types.ModuleType(name)
    module.__version__ = "0.9.0"
    sys.modules[name] = module
    monkeypatch.setitem(VERSIONS, name, "1.0.0")

    match = "Pandas requires .*1.0.0.* of .fakemodule.*'0.9.0'"
    with pytest.raises(ImportError, match=match):
        import_optional_dependency("fakemodule")

    # Test min_version parameter
    result = import_optional_dependency("fakemodule", min_version="0.8")
    assert result is module

    with tm.assert_produces_warning(UserWarning):
        result = import_optional_dependency("fakemodule", errors="warn")
    assert result is None

    module.__version__ = "1.0.0"  # exact match is OK
    result = import_optional_dependency("fakemodule")
    assert result is module

    with pytest.raises(ImportError, match="Pandas requires version '1.1.0'"):
        import_optional_dependency("fakemodule", min_version="1.1.0")

    with tm.assert_produces_warning(UserWarning):
        result = import_optional_dependency(
            "fakemodule", errors="warn", min_version="1.1.0"
        )
    assert result is None

    result = import_optional_dependency(
        "fakemodule", errors="ignore", min_version="1.1.0"
    )
    assert result is None


def test_submodule(monkeypatch):
    # Create a fake module with a submodule
    name = "fakemodule"
    module = types.ModuleType(name)
    module.__version__ = "0.9.0"
    sys.modules[name] = module
    sub_name = "submodule"
    submodule = types.ModuleType(sub_name)
    setattr(module, sub_name, submodule)
    sys.modules[f"{name}.{sub_name}"] = submodule
    monkeypatch.setitem(VERSIONS, name, "1.0.0")

    match = "Pandas requires .*1.0.0.* of .fakemodule.*'0.9.0'"
    with pytest.raises(ImportError, match=match):
        import_optional_dependency("fakemodule.submodule")

    with tm.assert_produces_warning(UserWarning):
        result = import_optional_dependency("fakemodule.submodule", errors="warn")
    assert result is None

    module.__version__ = "1.0.0"  # exact match is OK
    result = import_optional_dependency("fakemodule.submodule")
    assert result is submodule


def test_no_version_raises(monkeypatch):
    name = "fakemodule"
    module = types.ModuleType(name)
    sys.modules[name] = module
    monkeypatch.setitem(VERSIONS, name, "1.0.0")

    with pytest.raises(ImportError, match="Can't determine .* fakemodule"):
        import_optional_dependency(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\conftest.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    NaT,
    date_range,
)


@pytest.fixture
def datetime_frame() -> DataFrame:
    """
    Fixture for DataFrame of floats with DatetimeIndex

    Columns are ['A', 'B', 'C', 'D']
    """
    return DataFrame(
        np.random.default_rng(2).standard_normal((100, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=100, freq="B"),
    )


@pytest.fixture
def float_string_frame():
    """
    Fixture for DataFrame of floats and strings with index of unique strings

    Columns are ['A', 'B', 'C', 'D', 'foo'].
    """
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD")),
    )
    df["foo"] = "bar"
    return df


@pytest.fixture
def mixed_float_frame():
    """
    Fixture for DataFrame of different float types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    df = DataFrame(
        {
            col: np.random.default_rng(2).random(30, dtype=dtype)
            for col, dtype in zip(
                list("ABCD"), ["float32", "float32", "float32", "float64"]
            )
        },
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
    )
    # not supported by numpy random
    df["C"] = df["C"].astype("float16")
    return df


@pytest.fixture
def mixed_int_frame():
    """
    Fixture for DataFrame of different int types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    return DataFrame(
        {
            col: np.ones(30, dtype=dtype)
            for col, dtype in zip(list("ABCD"), ["int32", "uint64", "uint8", "int64"])
        },
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
    )


@pytest.fixture
def timezone_frame():
    """
    Fixture for DataFrame of date_range Series with different time zones

    Columns are ['A', 'B', 'C']; some entries are missing

               A                         B                         C
    0 2013-01-01 2013-01-01 00:00:00-05:00 2013-01-01 00:00:00+01:00
    1 2013-01-02                       NaT                       NaT
    2 2013-01-03 2013-01-03 00:00:00-05:00 2013-01-03 00:00:00+01:00
    """
    df = DataFrame(
        {
            "A": date_range("20130101", periods=3),
            "B": date_range("20130101", periods=3, tz="US/Eastern"),
            "C": date_range("20130101", periods=3, tz="CET"),
        }
    )
    df.iloc[1, 1] = NaT
    df.iloc[1, 2] = NaT
    return df

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\continuum_mechanics\tests\test_truss.py ===
from sympy.core.symbol import Symbol, symbols
from sympy.physics.continuum_mechanics.truss import Truss
from sympy import sqrt


def test_truss():
    A = Symbol('A')
    B = Symbol('B')
    C = Symbol('C')
    AB, BC, AC = symbols('AB, BC, AC')
    P = Symbol('P')

    t = Truss()
    assert t.nodes == []
    assert t.node_labels == []
    assert t.node_positions == []
    assert t.members == {}
    assert t.loads == {}
    assert t.supports == {}
    assert t.reaction_loads == {}
    assert t.internal_forces == {}

    # testing the add_node method
    t.add_node((A, 0, 0), (B, 2, 2), (C, 3, 0))
    assert t.nodes == [(A, 0, 0), (B, 2, 2), (C, 3, 0)]
    assert t.node_labels == [A, B, C]
    assert t.node_positions == [(0, 0), (2, 2), (3, 0)]
    assert t.loads == {}
    assert t.supports == {}
    assert t.reaction_loads == {}

    # testing the remove_node method
    t.remove_node(C)
    assert t.nodes == [(A, 0, 0), (B, 2, 2)]
    assert t.node_labels == [A, B]
    assert t.node_positions == [(0, 0), (2, 2)]
    assert t.loads == {}
    assert t.supports == {}

    t.add_node((C, 3, 0))

    # testing the add_member method
    t.add_member((AB, A, B), (BC, B, C), (AC, A, C))
    assert t.members == {AB: [A, B], BC: [B, C], AC: [A, C]}
    assert t.internal_forces == {AB: 0, BC: 0, AC: 0}

    # testing the remove_member method
    t.remove_member(BC)
    assert t.members == {AB: [A, B], AC: [A, C]}
    assert t.internal_forces == {AB: 0, AC: 0}

    t.add_member((BC, B, C))

    D, CD = symbols('D, CD')

    # testing the change_label methods
    t.change_node_label((B, D))
    assert t.nodes == [(A, 0, 0), (D, 2, 2), (C, 3, 0)]
    assert t.node_labels == [A, D, C]
    assert t.loads == {}
    assert t.supports == {}
    assert t.members == {AB: [A, D], BC: [D, C], AC: [A, C]}

    t.change_member_label((BC, CD))
    assert t.members == {AB: [A, D], CD: [D, C], AC: [A, C]}
    assert t.internal_forces == {AB: 0, CD: 0, AC: 0}


    # testing the apply_load method
    t.apply_load((A, P, 90), (A, P/4, 90), (A, 2*P,45), (D, P/2, 90))
    assert t.loads == {A: [[P, 90], [P/4, 90], [2*P, 45]], D: [[P/2, 90]]}
    assert t.loads[A] == [[P, 90], [P/4, 90], [2*P, 45]]

    # testing the remove_load method
    t.remove_load((A, P/4, 90))
    assert t.loads == {A: [[P, 90], [2*P, 45]], D: [[P/2, 90]]}
    assert t.loads[A] == [[P, 90], [2*P, 45]]

    # testing the apply_support method
    t.apply_support((A, "pinned"), (D, "roller"))
    assert t.supports == {A: 'pinned', D: 'roller'}
    assert t.reaction_loads == {}
    assert t.loads == {A: [[P, 90], [2*P, 45], [Symbol('R_A_x'), 0], [Symbol('R_A_y'), 90]],  D: [[P/2, 90], [Symbol('R_D_y'), 90]]}

    # testing the remove_support method
    t.remove_support(A)
    assert t.supports == {D: 'roller'}
    assert t.reaction_loads == {}
    assert t.loads == {A: [[P, 90], [2*P, 45]], D: [[P/2, 90], [Symbol('R_D_y'), 90]]}

    t.apply_support((A, "pinned"))

    # testing the solve method
    t.solve()
    assert t.reaction_loads['R_A_x'] == -sqrt(2)*P
    assert t.reaction_loads['R_A_y'] == -sqrt(2)*P - P
    assert t.reaction_loads['R_D_y'] == -P/2
    assert t.internal_forces[AB]/P == 0
    assert t.internal_forces[CD] == 0
    assert t.internal_forces[AC] == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_drop.py ===
import pytest

from pandas import (
    Index,
    Series,
)
import pandas._testing as tm
from pandas.api.types import is_bool_dtype


@pytest.mark.parametrize(
    "data, index, drop_labels, axis, expected_data, expected_index",
    [
        # Unique Index
        ([1, 2], ["one", "two"], ["two"], 0, [1], ["one"]),
        ([1, 2], ["one", "two"], ["two"], "rows", [1], ["one"]),
        ([1, 1, 2], ["one", "two", "one"], ["two"], 0, [1, 2], ["one", "one"]),
        # GH 5248 Non-Unique Index
        ([1, 1, 2], ["one", "two", "one"], "two", 0, [1, 2], ["one", "one"]),
        ([1, 1, 2], ["one", "two", "one"], ["one"], 0, [1], ["two"]),
        ([1, 1, 2], ["one", "two", "one"], "one", 0, [1], ["two"]),
    ],
)
def test_drop_unique_and_non_unique_index(
    data, index, axis, drop_labels, expected_data, expected_index
):
    ser = Series(data=data, index=index)
    result = ser.drop(drop_labels, axis=axis)
    expected = Series(data=expected_data, index=expected_index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, index, drop_labels, axis, error_type, error_desc",
    [
        # single string/tuple-like
        (range(3), list("abc"), "bc", 0, KeyError, "not found in axis"),
        # bad axis
        (range(3), list("abc"), ("a",), 0, KeyError, "not found in axis"),
        (range(3), list("abc"), "one", "columns", ValueError, "No axis named columns"),
    ],
)
def test_drop_exception_raised(data, index, drop_labels, axis, error_type, error_desc):
    ser = Series(data, index=index)
    with pytest.raises(error_type, match=error_desc):
        ser.drop(drop_labels, axis=axis)


def test_drop_with_ignore_errors():
    # errors='ignore'
    ser = Series(range(3), index=list("abc"))
    result = ser.drop("bc", errors="ignore")
    tm.assert_series_equal(result, ser)
    result = ser.drop(["a", "d"], errors="ignore")
    expected = ser.iloc[1:]
    tm.assert_series_equal(result, expected)

    # GH 8522
    ser = Series([2, 3], index=[True, False])
    assert is_bool_dtype(ser.index)
    assert ser.index.dtype == bool
    result = ser.drop(True)
    expected = Series([3], index=[False])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("index", [[1, 2, 3], [1, 1, 3]])
@pytest.mark.parametrize("drop_labels", [[], [1], [3]])
def test_drop_empty_list(index, drop_labels):
    # GH 21494
    expected_index = [i for i in index if i not in drop_labels]
    series = Series(index=index, dtype=object).drop(drop_labels)
    expected = Series(index=expected_index, dtype=object)
    tm.assert_series_equal(series, expected)


@pytest.mark.parametrize(
    "data, index, drop_labels",
    [
        (None, [1, 2, 3], [1, 4]),
        (None, [1, 2, 2], [1, 4]),
        ([2, 3], [0, 1], [False, True]),
    ],
)
def test_drop_non_empty_list(data, index, drop_labels):
    # GH 21494 and GH 16877
    dtype = object if data is None else None
    ser = Series(data=data, index=index, dtype=dtype)
    with pytest.raises(KeyError, match="not found in axis"):
        ser.drop(drop_labels)


def test_drop_index_ea_dtype(any_numeric_ea_dtype):
    # GH#45860
    df = Series(100, index=Index([1, 2, 2], dtype=any_numeric_ea_dtype))
    idx = Index([df.index[1]])
    result = df.drop(idx)
    expected = Series(100, index=Index([1], dtype=any_numeric_ea_dtype))
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_matmul.py ===
import operator

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


class TestMatMul:
    def test_matmul(self):
        # matmul test is for GH#10259
        a = DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=["a", "b", "c"],
            columns=["p", "q", "r", "s"],
        )
        b = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)),
            index=["p", "q", "r", "s"],
            columns=["one", "two"],
        )

        # DataFrame @ DataFrame
        result = operator.matmul(a, b)
        expected = DataFrame(
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_frame_equal(result, expected)

        # DataFrame @ Series
        result = operator.matmul(a, b.one)
        expected = Series(np.dot(a.values, b.one.values), index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        # np.array @ DataFrame
        result = operator.matmul(a.values, b)
        assert isinstance(result, DataFrame)
        assert result.columns.equals(b.columns)
        assert result.index.equals(Index(range(3)))
        expected = np.dot(a.values, b.values)
        tm.assert_almost_equal(result.values, expected)

        # nested list @ DataFrame (__rmatmul__)
        result = operator.matmul(a.values.tolist(), b)
        expected = DataFrame(
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_almost_equal(result.values, expected.values)

        # mixed dtype DataFrame @ DataFrame
        a["q"] = a.q.round().astype(int)
        result = operator.matmul(a, b)
        expected = DataFrame(
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_frame_equal(result, expected)

        # different dtypes DataFrame @ DataFrame
        a = a.astype(int)
        result = operator.matmul(a, b)
        expected = DataFrame(
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_frame_equal(result, expected)

        # unaligned
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=[1, 2, 3],
            columns=range(4),
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=range(5),
            columns=[1, 2, 3],
        )

        with pytest.raises(ValueError, match="aligned"):
            operator.matmul(df, df2)

    def test_matmul_message_shapes(self):
        # GH#21581 exception message should reflect original shapes,
        #  not transposed shapes
        a = np.random.default_rng(2).random((10, 4))
        b = np.random.default_rng(2).random((5, 3))

        df = DataFrame(b)

        msg = r"shapes \(10, 4\) and \(5, 3\) not aligned"
        with pytest.raises(ValueError, match=msg):
            a @ df
        with pytest.raises(ValueError, match=msg):
            a.tolist() @ df

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_backend.py ===
import sys
import types

import pytest

import pandas.util._test_decorators as td

import pandas


@pytest.fixture
def dummy_backend():
    db = types.ModuleType("pandas_dummy_backend")
    setattr(db, "plot", lambda *args, **kwargs: "used_dummy")
    return db


@pytest.fixture
def restore_backend():
    """Restore the plotting backend to matplotlib"""
    with pandas.option_context("plotting.backend", "matplotlib"):
        yield


def test_backend_is_not_module():
    msg = "Could not find plotting backend 'not_an_existing_module'."
    with pytest.raises(ValueError, match=msg):
        pandas.set_option("plotting.backend", "not_an_existing_module")

    assert pandas.options.plotting.backend == "matplotlib"


def test_backend_is_correct(monkeypatch, restore_backend, dummy_backend):
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)

    pandas.set_option("plotting.backend", "pandas_dummy_backend")
    assert pandas.get_option("plotting.backend") == "pandas_dummy_backend"
    assert (
        pandas.plotting._core._get_plot_backend("pandas_dummy_backend") is dummy_backend
    )


def test_backend_can_be_set_in_plot_call(monkeypatch, restore_backend, dummy_backend):
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)
    df = pandas.DataFrame([1, 2, 3])

    assert pandas.get_option("plotting.backend") == "matplotlib"
    assert df.plot(backend="pandas_dummy_backend") == "used_dummy"


def test_register_entrypoint(restore_backend, tmp_path, monkeypatch, dummy_backend):
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)

    dist_info = tmp_path / "my_backend-0.0.0.dist-info"
    dist_info.mkdir()
    # entry_point name should not match module name - otherwise pandas will
    # fall back to backend lookup by module name
    (dist_info / "entry_points.txt").write_bytes(
        b"[pandas_plotting_backends]\nmy_ep_backend = pandas_dummy_backend\n"
    )

    assert pandas.plotting._core._get_plot_backend("my_ep_backend") is dummy_backend

    with pandas.option_context("plotting.backend", "my_ep_backend"):
        assert pandas.plotting._core._get_plot_backend() is dummy_backend


def test_setting_backend_without_plot_raises(monkeypatch):
    # GH-28163
    module = types.ModuleType("pandas_plot_backend")
    monkeypatch.setitem(sys.modules, "pandas_plot_backend", module)

    assert pandas.options.plotting.backend == "matplotlib"
    with pytest.raises(
        ValueError, match="Could not find plotting backend 'pandas_plot_backend'."
    ):
        pandas.set_option("plotting.backend", "pandas_plot_backend")

    assert pandas.options.plotting.backend == "matplotlib"


@td.skip_if_installed("matplotlib")
def test_no_matplotlib_ok():
    msg = (
        'matplotlib is required for plotting when the default backend "matplotlib" is '
        "selected."
    )
    with pytest.raises(ImportError, match=msg):
        pandas.plotting._core._get_plot_backend("matplotlib")


def test_extra_kinds_ok(monkeypatch, restore_backend, dummy_backend):
    # https://github.com/pandas-dev/pandas/pull/28647
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)
    pandas.set_option("plotting.backend", "pandas_dummy_backend")
    df = pandas.DataFrame({"A": [1, 2, 3]})
    df.plot(kind="not a real kind")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_custom_business_day.py ===
"""
Tests for offsets.CustomBusinessDay / CDay
"""
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs.offsets import CDay

from pandas import (
    _testing as tm,
    read_pickle,
)
from pandas.tests.tseries.offsets.common import assert_offset_equal

from pandas.tseries.holiday import USFederalHolidayCalendar


@pytest.fixture
def offset():
    return CDay()


@pytest.fixture
def offset2():
    return CDay(2)


class TestCustomBusinessDay:
    def test_repr(self, offset, offset2):
        assert repr(offset) == "<CustomBusinessDay>"
        assert repr(offset2) == "<2 * CustomBusinessDays>"

        expected = "<BusinessDay: offset=datetime.timedelta(days=1)>"
        assert repr(offset + timedelta(1)) == expected

    def test_holidays(self):
        # Define a TradingDay offset
        holidays = ["2012-05-01", datetime(2013, 5, 1), np.datetime64("2014-05-01")]
        tday = CDay(holidays=holidays)
        for year in range(2012, 2015):
            dt = datetime(year, 4, 30)
            xp = datetime(year, 5, 2)
            rs = dt + tday
            assert rs == xp

    def test_weekmask(self):
        weekmask_saudi = "Sat Sun Mon Tue Wed"  # Thu-Fri Weekend
        weekmask_uae = "1111001"  # Fri-Sat Weekend
        weekmask_egypt = [1, 1, 1, 1, 0, 0, 1]  # Fri-Sat Weekend
        bday_saudi = CDay(weekmask=weekmask_saudi)
        bday_uae = CDay(weekmask=weekmask_uae)
        bday_egypt = CDay(weekmask=weekmask_egypt)
        dt = datetime(2013, 5, 1)
        xp_saudi = datetime(2013, 5, 4)
        xp_uae = datetime(2013, 5, 2)
        xp_egypt = datetime(2013, 5, 2)
        assert xp_saudi == dt + bday_saudi
        assert xp_uae == dt + bday_uae
        assert xp_egypt == dt + bday_egypt
        xp2 = datetime(2013, 5, 5)
        assert xp2 == dt + 2 * bday_saudi
        assert xp2 == dt + 2 * bday_uae
        assert xp2 == dt + 2 * bday_egypt

    def test_weekmask_and_holidays(self):
        weekmask_egypt = "Sun Mon Tue Wed Thu"  # Fri-Sat Weekend
        holidays = ["2012-05-01", datetime(2013, 5, 1), np.datetime64("2014-05-01")]
        bday_egypt = CDay(holidays=holidays, weekmask=weekmask_egypt)
        dt = datetime(2013, 4, 30)
        xp_egypt = datetime(2013, 5, 5)
        assert xp_egypt == dt + 2 * bday_egypt

    @pytest.mark.filterwarnings("ignore:Non:pandas.errors.PerformanceWarning")
    def test_calendar(self):
        calendar = USFederalHolidayCalendar()
        dt = datetime(2014, 1, 17)
        assert_offset_equal(CDay(calendar=calendar), dt, datetime(2014, 1, 21))

    def test_roundtrip_pickle(self, offset, offset2):
        def _check_roundtrip(obj):
            unpickled = tm.round_trip_pickle(obj)
            assert unpickled == obj

        _check_roundtrip(offset)
        _check_roundtrip(offset2)
        _check_roundtrip(offset * 2)

    def test_pickle_compat_0_14_1(self, datapath):
        hdays = [datetime(2013, 1, 1) for ele in range(4)]
        pth = datapath("tseries", "offsets", "data", "cday-0.14.1.pickle")
        cday0_14_1 = read_pickle(pth)
        cday = CDay(holidays=hdays)
        assert cday == cday0_14_1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_trigonometry.py ===
from sympy.core import Ne, Rational, Symbol
from sympy.functions import sin, cos, tan, csc, sec, cot, log, Piecewise
from sympy.integrals.trigonometry import trigintegrate

x = Symbol('x')


def test_trigintegrate_odd():
    assert trigintegrate(Rational(1), x) == x
    assert trigintegrate(x, x) is None
    assert trigintegrate(x**2, x) is None

    assert trigintegrate(sin(x), x) == -cos(x)
    assert trigintegrate(cos(x), x) == sin(x)

    assert trigintegrate(sin(3*x), x) == -cos(3*x)/3
    assert trigintegrate(cos(3*x), x) == sin(3*x)/3

    y = Symbol('y')
    assert trigintegrate(sin(y*x), x) == Piecewise(
        (-cos(y*x)/y, Ne(y, 0)), (0, True))
    assert trigintegrate(cos(y*x), x) == Piecewise(
        (sin(y*x)/y, Ne(y, 0)), (x, True))
    assert trigintegrate(sin(y*x)**2, x) == Piecewise(
        ((x*y/2 - sin(x*y)*cos(x*y)/2)/y, Ne(y, 0)), (0, True))
    assert trigintegrate(sin(y*x)*cos(y*x), x) == Piecewise(
        (sin(x*y)**2/(2*y), Ne(y, 0)), (0, True))
    assert trigintegrate(cos(y*x)**2, x) == Piecewise(
        ((x*y/2 + sin(x*y)*cos(x*y)/2)/y, Ne(y, 0)), (x, True))

    y = Symbol('y', positive=True)
    # TODO: remove conds='none' below. For this to work we would have to rule
    #       out (e.g. by trying solve) the condition y = 0, incompatible with
    #       y.is_positive being True.
    assert trigintegrate(sin(y*x), x, conds='none') == -cos(y*x)/y
    assert trigintegrate(cos(y*x), x, conds='none') == sin(y*x)/y

    assert trigintegrate(sin(x)*cos(x), x) == sin(x)**2/2
    assert trigintegrate(sin(x)*cos(x)**2, x) == -cos(x)**3/3
    assert trigintegrate(sin(x)**2*cos(x), x) == sin(x)**3/3

    # check if it selects right function to substitute,
    # so the result is kept simple
    assert trigintegrate(sin(x)**7 * cos(x), x) == sin(x)**8/8
    assert trigintegrate(sin(x) * cos(x)**7, x) == -cos(x)**8/8

    assert trigintegrate(sin(x)**7 * cos(x)**3, x) == \
        -sin(x)**10/10 + sin(x)**8/8
    assert trigintegrate(sin(x)**3 * cos(x)**7, x) == \
        cos(x)**10/10 - cos(x)**8/8

    # both n, m are odd and -ve, and not necessarily equal
    assert trigintegrate(sin(x)**-1*cos(x)**-1, x) == \
        -log(sin(x)**2 - 1)/2 + log(sin(x))


def test_trigintegrate_even():
    assert trigintegrate(sin(x)**2, x) == x/2 - cos(x)*sin(x)/2
    assert trigintegrate(cos(x)**2, x) == x/2 + cos(x)*sin(x)/2

    assert trigintegrate(sin(3*x)**2, x) == x/2 - cos(3*x)*sin(3*x)/6
    assert trigintegrate(cos(3*x)**2, x) == x/2 + cos(3*x)*sin(3*x)/6
    assert trigintegrate(sin(x)**2 * cos(x)**2, x) == \
        x/8 - sin(2*x)*cos(2*x)/16

    assert trigintegrate(sin(x)**4 * cos(x)**2, x) == \
        x/16 - sin(x) *cos(x)/16 - sin(x)**3*cos(x)/24 + \
        sin(x)**5*cos(x)/6

    assert trigintegrate(sin(x)**2 * cos(x)**4, x) == \
        x/16 + cos(x) *sin(x)/16 + cos(x)**3*sin(x)/24 - \
        cos(x)**5*sin(x)/6

    assert trigintegrate(sin(x)**(-4), x) == -2*cos(x)/(3*sin(x)) \
        - cos(x)/(3*sin(x)**3)

    assert trigintegrate(cos(x)**(-6), x) == sin(x)/(5*cos(x)**5) \
        + 4*sin(x)/(15*cos(x)**3) + 8*sin(x)/(15*cos(x))


def test_trigintegrate_mixed():
    assert trigintegrate(sin(x)*sec(x), x) == -log(cos(x))
    assert trigintegrate(sin(x)*csc(x), x) == x
    assert trigintegrate(sin(x)*cot(x), x) == sin(x)

    assert trigintegrate(cos(x)*sec(x), x) == x
    assert trigintegrate(cos(x)*csc(x), x) == log(sin(x))
    assert trigintegrate(cos(x)*tan(x), x) == -cos(x)
    assert trigintegrate(cos(x)*cot(x), x) == log(cos(x) - 1)/2 \
        - log(cos(x) + 1)/2 + cos(x)
    assert trigintegrate(cot(x)*cos(x)**2, x) == log(sin(x)) - sin(x)**2/2


def test_trigintegrate_symbolic():
    n = Symbol('n', integer=True)
    assert trigintegrate(cos(x)**n, x) is None
    assert trigintegrate(sin(x)**n, x) is None
    assert trigintegrate(cot(x)**n, x) is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_downcast.py ===
import decimal

import numpy as np
import pytest

from pandas.core.dtypes.cast import maybe_downcast_to_dtype

from pandas import (
    Series,
    Timedelta,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "arr,dtype,expected",
    [
        (
            np.array([8.5, 8.6, 8.7, 8.8, 8.9999999999995]),
            "infer",
            np.array([8.5, 8.6, 8.7, 8.8, 8.9999999999995]),
        ),
        (
            np.array([8.0, 8.0, 8.0, 8.0, 8.9999999999995]),
            "infer",
            np.array([8, 8, 8, 8, 9], dtype=np.int64),
        ),
        (
            np.array([8.0, 8.0, 8.0, 8.0, 9.0000000000005]),
            "infer",
            np.array([8, 8, 8, 8, 9], dtype=np.int64),
        ),
        (
            # This is a judgement call, but we do _not_ downcast Decimal
            #  objects
            np.array([decimal.Decimal(0.0)]),
            "int64",
            np.array([decimal.Decimal(0.0)]),
        ),
        (
            # GH#45837
            np.array([Timedelta(days=1), Timedelta(days=2)], dtype=object),
            "infer",
            np.array([1, 2], dtype="m8[D]").astype("m8[ns]"),
        ),
        # TODO: similar for dt64, dt64tz, Period, Interval?
    ],
)
def test_downcast(arr, expected, dtype):
    result = maybe_downcast_to_dtype(arr, dtype)
    tm.assert_numpy_array_equal(result, expected)


def test_downcast_booleans():
    # see gh-16875: coercing of booleans.
    ser = Series([True, True, False])
    result = maybe_downcast_to_dtype(ser, np.dtype(np.float64))

    expected = ser.values
    tm.assert_numpy_array_equal(result, expected)


def test_downcast_conversion_no_nan(any_real_numpy_dtype):
    dtype = any_real_numpy_dtype
    expected = np.array([1, 2])
    arr = np.array([1.0, 2.0], dtype=dtype)

    result = maybe_downcast_to_dtype(arr, "infer")
    tm.assert_almost_equal(result, expected, check_dtype=False)


def test_downcast_conversion_nan(float_numpy_dtype):
    dtype = float_numpy_dtype
    data = [1.0, 2.0, np.nan]

    expected = np.array(data, dtype=dtype)
    arr = np.array(data, dtype=dtype)

    result = maybe_downcast_to_dtype(arr, "infer")
    tm.assert_almost_equal(result, expected)


def test_downcast_conversion_empty(any_real_numpy_dtype):
    dtype = any_real_numpy_dtype
    arr = np.array([], dtype=dtype)
    result = maybe_downcast_to_dtype(arr, np.dtype("int64"))
    tm.assert_numpy_array_equal(result, np.array([], dtype=np.int64))


@pytest.mark.parametrize("klass", [np.datetime64, np.timedelta64])
def test_datetime_likes_nan(klass):
    dtype = klass.__name__ + "[ns]"
    arr = np.array([1, 2, np.nan])

    exp = np.array([1, 2, klass("NaT")], dtype)
    res = maybe_downcast_to_dtype(arr, dtype)
    tm.assert_numpy_array_equal(res, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\base_class\test_reshape.py ===
"""
Tests for ndarray-like method on the base Index class
"""
import numpy as np
import pytest

import pandas as pd
from pandas import Index
import pandas._testing as tm


class TestReshape:
    def test_repeat(self):
        repeats = 2
        index = Index([1, 2, 3])
        expected = Index([1, 1, 2, 2, 3, 3])

        result = index.repeat(repeats)
        tm.assert_index_equal(result, expected)

    def test_insert(self):
        # GH 7256
        # validate neg/pos inserts
        result = Index(["b", "c", "d"])

        # test 0th element
        tm.assert_index_equal(Index(["a", "b", "c", "d"]), result.insert(0, "a"))

        # test Nth element that follows Python list behavior
        tm.assert_index_equal(Index(["b", "c", "e", "d"]), result.insert(-1, "e"))

        # test loc +/- neq (0, -1)
        tm.assert_index_equal(result.insert(1, "z"), result.insert(-2, "z"))

        # test empty
        null_index = Index([])
        tm.assert_index_equal(Index(["a"], dtype=object), null_index.insert(0, "a"))

    def test_insert_missing(self, request, nulls_fixture, using_infer_string):
        if using_infer_string and nulls_fixture is pd.NA:
            request.applymarker(pytest.mark.xfail(reason="TODO(infer_string)"))
        # GH#22295
        # test there is no mangling of NA values
        expected = Index(["a", nulls_fixture, "b", "c"], dtype=object)
        result = Index(list("abc"), dtype=object).insert(
            1, Index([nulls_fixture], dtype=object)
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "val", [(1, 2), np.datetime64("2019-12-31"), np.timedelta64(1, "D")]
    )
    @pytest.mark.parametrize("loc", [-1, 2])
    def test_insert_datetime_into_object(self, loc, val):
        # GH#44509
        idx = Index(["1", "2", "3"])
        result = idx.insert(loc, val)
        expected = Index(["1", "2", val, "3"])
        tm.assert_index_equal(result, expected)
        assert type(expected[2]) is type(val)

    def test_insert_none_into_string_numpy(self, string_dtype_no_object):
        # GH#55365
        index = Index(["a", "b", "c"], dtype=string_dtype_no_object)
        result = index.insert(-1, None)
        expected = Index(["a", "b", None, "c"], dtype=string_dtype_no_object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "pos,expected",
        [
            (0, Index(["b", "c", "d"], name="index")),
            (-1, Index(["a", "b", "c"], name="index")),
        ],
    )
    def test_delete(self, pos, expected):
        index = Index(["a", "b", "c", "d"], name="index")
        result = index.delete(pos)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

    def test_delete_raises(self):
        index = Index(["a", "b", "c", "d"], name="index")
        msg = "index 5 is out of bounds for axis 0 with size 4"
        with pytest.raises(IndexError, match=msg):
            index.delete(5)

    def test_append_multiple(self):
        index = Index(["a", "b", "c", "d", "e", "f"])

        foos = [index[:2], index[2:4], index[4:]]
        result = foos[0].append(foos[1:])
        tm.assert_index_equal(result, index)

        # empty
        result = index.append([])
        tm.assert_index_equal(result, index)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_parametricregion.py ===
from sympy.core.numbers import pi
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.parametricregion import ParametricRegion, parametric_region_list
from sympy.geometry import Point, Segment, Curve, Ellipse, Line, Parabola, Polygon
from sympy.testing.pytest import raises
from sympy.abc import a, b, r, t, x, y, z, theta, phi


C = CoordSys3D('C')

def test_ParametricRegion():

    point = ParametricRegion((3, 4))
    assert point.definition == (3, 4)
    assert point.parameters == ()
    assert point.limits == {}
    assert point.dimensions == 0

    # line x = y
    line_xy = ParametricRegion((y, y), (y, 1, 5))
    assert line_xy .definition == (y, y)
    assert line_xy.parameters == (y,)
    assert line_xy.dimensions == 1

    # line y = z
    line_yz = ParametricRegion((x,t,t), x, (t, 1, 2))
    assert line_yz.definition == (x,t,t)
    assert line_yz.parameters == (x, t)
    assert line_yz.limits == {t: (1, 2)}
    assert line_yz.dimensions == 1

    p1 = ParametricRegion((9*a, -16*b), (a, 0, 2), (b, -1, 5))
    assert p1.definition == (9*a, -16*b)
    assert p1.parameters == (a, b)
    assert p1.limits == {a: (0, 2), b: (-1, 5)}
    assert p1.dimensions == 2

    p2 = ParametricRegion((t, t**3), t)
    assert p2.parameters == (t,)
    assert p2.limits == {}
    assert p2.dimensions == 0

    circle = ParametricRegion((r*cos(theta), r*sin(theta)), r, (theta, 0, 2*pi))
    assert circle.definition == (r*cos(theta), r*sin(theta))
    assert circle.dimensions == 1

    halfdisc = ParametricRegion((r*cos(theta), r*sin(theta)), (r, -2, 2), (theta, 0, pi))
    assert halfdisc.definition == (r*cos(theta), r*sin(theta))
    assert halfdisc.parameters == (r, theta)
    assert halfdisc.limits == {r: (-2, 2), theta: (0, pi)}
    assert halfdisc.dimensions == 2

    ellipse = ParametricRegion((a*cos(t), b*sin(t)), (t, 0, 8))
    assert ellipse.parameters == (t,)
    assert ellipse.limits == {t: (0, 8)}
    assert ellipse.dimensions == 1

    cylinder = ParametricRegion((r*cos(theta), r*sin(theta), z), (r, 0, 1), (theta, 0, 2*pi), (z, 0, 4))
    assert cylinder.parameters == (r, theta, z)
    assert cylinder.dimensions == 3

    sphere = ParametricRegion((r*sin(phi)*cos(theta),r*sin(phi)*sin(theta), r*cos(phi)),
                                r, (theta, 0, 2*pi), (phi, 0, pi))
    assert sphere.definition == (r*sin(phi)*cos(theta),r*sin(phi)*sin(theta), r*cos(phi))
    assert sphere.parameters == (r, theta, phi)
    assert sphere.dimensions == 2

    raises(ValueError, lambda: ParametricRegion((a*t**2, 2*a*t), (a, -2)))
    raises(ValueError, lambda: ParametricRegion((a, b), (a**2, sin(b)), (a, 2, 4, 6)))


def test_parametric_region_list():

    point = Point(-5, 12)
    assert parametric_region_list(point) == [ParametricRegion((-5, 12))]

    e = Ellipse(Point(2, 8), 2, 6)
    assert parametric_region_list(e, t) == [ParametricRegion((2*cos(t) + 2, 6*sin(t) + 8), (t, 0, 2*pi))]

    c = Curve((t, t**3), (t, 5, 3))
    assert parametric_region_list(c) == [ParametricRegion((t, t**3), (t, 5, 3))]

    s = Segment(Point(2, 11, -6), Point(0, 2, 5))
    assert parametric_region_list(s, t) == [ParametricRegion((2 - 2*t, 11 - 9*t, 11*t - 6), (t, 0, 1))]
    s1 = Segment(Point(0, 0), (1, 0))
    assert parametric_region_list(s1, t) == [ParametricRegion((t, 0), (t, 0, 1))]
    s2 = Segment(Point(1, 2, 3), Point(1, 2, 5))
    assert parametric_region_list(s2, t) == [ParametricRegion((1, 2, 2*t + 3), (t, 0, 1))]
    s3 = Segment(Point(12, 56), Point(12, 56))
    assert parametric_region_list(s3) == [ParametricRegion((12, 56))]

    poly = Polygon((1,3), (-3, 8), (2, 4))
    assert parametric_region_list(poly, t) == [ParametricRegion((1 - 4*t, 5*t + 3), (t, 0, 1)), ParametricRegion((5*t - 3, 8 - 4*t), (t, 0, 1)), ParametricRegion((2 - t, 4 - t), (t, 0, 1))]

    p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7,8)))
    raises(ValueError, lambda: parametric_region_list(p1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_equals.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    MultiIndex,
)


class TestEquals:
    def test_equals_categorical(self):
        ci1 = CategoricalIndex(["a", "b"], categories=["a", "b"], ordered=True)
        ci2 = CategoricalIndex(["a", "b"], categories=["a", "b", "c"], ordered=True)

        assert ci1.equals(ci1)
        assert not ci1.equals(ci2)
        assert ci1.equals(ci1.astype(object))
        assert ci1.astype(object).equals(ci1)

        assert (ci1 == ci1).all()
        assert not (ci1 != ci1).all()
        assert not (ci1 > ci1).all()
        assert not (ci1 < ci1).all()
        assert (ci1 <= ci1).all()
        assert (ci1 >= ci1).all()

        assert not (ci1 == 1).all()
        assert (ci1 == Index(["a", "b"])).all()
        assert (ci1 == ci1.values).all()

        # invalid comparisons
        with pytest.raises(ValueError, match="Lengths must match"):
            ci1 == Index(["a", "b", "c"])

        msg = "Categoricals can only be compared if 'categories' are the same"
        with pytest.raises(TypeError, match=msg):
            ci1 == ci2
        with pytest.raises(TypeError, match=msg):
            ci1 == Categorical(ci1.values, ordered=False)
        with pytest.raises(TypeError, match=msg):
            ci1 == Categorical(ci1.values, categories=list("abc"))

        # tests
        # make sure that we are testing for category inclusion properly
        ci = CategoricalIndex(list("aabca"), categories=["c", "a", "b"])
        assert not ci.equals(list("aabca"))
        # Same categories, but different order
        # Unordered
        assert ci.equals(CategoricalIndex(list("aabca")))
        # Ordered
        assert not ci.equals(CategoricalIndex(list("aabca"), ordered=True))
        assert ci.equals(ci.copy())

        ci = CategoricalIndex(list("aabca") + [np.nan], categories=["c", "a", "b"])
        assert not ci.equals(list("aabca"))
        assert not ci.equals(CategoricalIndex(list("aabca")))
        assert ci.equals(ci.copy())

        ci = CategoricalIndex(list("aabca") + [np.nan], categories=["c", "a", "b"])
        assert not ci.equals(list("aabca") + [np.nan])
        assert ci.equals(CategoricalIndex(list("aabca") + [np.nan]))
        assert not ci.equals(CategoricalIndex(list("aabca") + [np.nan], ordered=True))
        assert ci.equals(ci.copy())

    def test_equals_categorical_unordered(self):
        # https://github.com/pandas-dev/pandas/issues/16603
        a = CategoricalIndex(["A"], categories=["A", "B"])
        b = CategoricalIndex(["A"], categories=["B", "A"])
        c = CategoricalIndex(["C"], categories=["B", "A"])
        assert a.equals(b)
        assert not a.equals(c)
        assert not b.equals(c)

    def test_equals_non_category(self):
        # GH#37667 Case where other contains a value not among ci's
        #  categories ("D") and also contains np.nan
        ci = CategoricalIndex(["A", "B", np.nan, np.nan])
        other = Index(["A", "B", "D", np.nan])

        assert not ci.equals(other)

    def test_equals_multiindex(self):
        # dont raise NotImplementedError when calling is_dtype_compat

        mi = MultiIndex.from_arrays([["A", "B", "C", "D"], range(4)])
        ci = mi.to_flat_index().astype("category")

        assert not ci.equals(mi)

    def test_equals_string_dtype(self, any_string_dtype):
        # GH#55364
        idx = CategoricalIndex(list("abc"), name="B")
        other = Index(["a", "b", "c"], name="B", dtype=any_string_dtype)
        assert idx.equals(other)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_copy.py ===
from copy import (
    copy,
    deepcopy,
)

import pytest

from pandas import MultiIndex
import pandas._testing as tm


def assert_multiindex_copied(copy, original):
    # Levels should be (at least, shallow copied)
    tm.assert_copy(copy.levels, original.levels)
    tm.assert_almost_equal(copy.codes, original.codes)

    # Labels doesn't matter which way copied
    tm.assert_almost_equal(copy.codes, original.codes)
    assert copy.codes is not original.codes

    # Names doesn't matter which way copied
    assert copy.names == original.names
    assert copy.names is not original.names

    # Sort order should be copied
    assert copy.sortorder == original.sortorder


def test_copy(idx):
    i_copy = idx.copy()

    assert_multiindex_copied(i_copy, idx)


def test_shallow_copy(idx):
    i_copy = idx._view()

    assert_multiindex_copied(i_copy, idx)


def test_view(idx):
    i_view = idx.view()
    assert_multiindex_copied(i_view, idx)


@pytest.mark.parametrize("func", [copy, deepcopy])
def test_copy_and_deepcopy(func):
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )
    idx_copy = func(idx)
    assert idx_copy is not idx
    assert idx_copy.equals(idx)


@pytest.mark.parametrize("deep", [True, False])
def test_copy_method(deep):
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )
    idx_copy = idx.copy(deep=deep)
    assert idx_copy.equals(idx)


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize(
    "kwarg, value",
    [
        ("names", ["third", "fourth"]),
    ],
)
def test_copy_method_kwargs(deep, kwarg, value):
    # gh-12309: Check that the "name" argument as well other kwargs are honored
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )
    idx_copy = idx.copy(**{kwarg: value, "deep": deep})
    assert getattr(idx_copy, kwarg) == value


def test_copy_deep_false_retains_id():
    # GH#47878
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )

    res = idx.copy(deep=False)
    assert res._id is idx._id

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_to_string.py ===
from textwrap import dedent

import pytest

from pandas import (
    DataFrame,
    Series,
)

pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler


@pytest.fixture
def df():
    return DataFrame(
        {"A": [0, 1], "B": [-0.61, -1.22], "C": Series(["ab", "cd"], dtype=object)}
    )


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0, precision=2)


def test_basic_string(styler):
    result = styler.to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    """
    )
    assert result == expected


def test_string_delimiter(styler):
    result = styler.to_string(delimiter=";")
    expected = dedent(
        """\
    ;A;B;C
    0;0;-0.61;ab
    1;1;-1.22;cd
    """
    )
    assert result == expected


def test_concat(styler):
    result = styler.concat(styler.data.agg(["sum"]).style).to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    sum 1 -1.830000 abcd
    """
    )
    assert result == expected


def test_concat_recursion(styler):
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["sum"]), uuid_len=0, precision=3)
    styler3 = Styler(df.agg(["sum"]), uuid_len=0, precision=4)
    result = styler1.concat(styler2.concat(styler3)).to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    sum 1 -1.830 abcd
    sum 1 -1.8300 abcd
    """
    )
    assert result == expected


def test_concat_chain(styler):
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["sum"]), uuid_len=0, precision=3)
    styler3 = Styler(df.agg(["sum"]), uuid_len=0, precision=4)
    result = styler1.concat(styler2).concat(styler3).to_string()
    expected = dedent(
        """\
     A B C
    0 0 -0.61 ab
    1 1 -1.22 cd
    sum 1 -1.830 abcd
    sum 1 -1.8300 abcd
    """
    )
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\usecols\test_strings.py ===
"""
Tests the usecols functionality during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO

import pytest

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


def test_usecols_with_unicode_strings(all_parsers):
    # see gh-13219
    data = """AAA,BBB,CCC,DDD
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers

    exp_data = {
        "AAA": {
            0: 0.056674972999999997,
            1: 2.6132309819999997,
            2: 3.5689350380000002,
        },
        "BBB": {0: 8, 1: 2, 2: 7},
    }
    expected = DataFrame(exp_data)

    result = parser.read_csv(StringIO(data), usecols=["AAA", "BBB"])
    tm.assert_frame_equal(result, expected)


def test_usecols_with_single_byte_unicode_strings(all_parsers):
    # see gh-13219
    data = """A,B,C,D
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers

    exp_data = {
        "A": {
            0: 0.056674972999999997,
            1: 2.6132309819999997,
            2: 3.5689350380000002,
        },
        "B": {0: 8, 1: 2, 2: 7},
    }
    expected = DataFrame(exp_data)

    result = parser.read_csv(StringIO(data), usecols=["A", "B"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("usecols", [["AAA", b"BBB"], [b"AAA", "BBB"]])
def test_usecols_with_mixed_encoding_strings(all_parsers, usecols):
    data = """AAA,BBB,CCC,DDD
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers
    _msg_validate_usecols_arg = (
        "'usecols' must either be list-like "
        "of all strings, all unicode, all "
        "integers or a callable."
    )
    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols=usecols)


@pytest.mark.parametrize("usecols", [["あああ", "いい"], ["あああ", "いい"]])
def test_usecols_with_multi_byte_characters(all_parsers, usecols):
    data = """あああ,いい,ううう,ええええ
0.056674973,8,True,a
2.613230982,2,False,b
3.568935038,7,False,a"""
    parser = all_parsers

    exp_data = {
        "あああ": {
            0: 0.056674972999999997,
            1: 2.6132309819999997,
            2: 3.5689350380000002,
        },
        "いい": {0: 8, 1: 2, 2: 7},
    }
    expected = DataFrame(exp_data)

    result = parser.read_csv(StringIO(data), usecols=usecols)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\tzwin.py ===
# tzwin has moved to dateutil.tz.win
from .tz.win import *

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\matrices\__init__.py ===
from . import eigen           # to set methods
from . import eigen_symmetric # to set methods

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_quad.py ===
import pytest
from mpmath import *

def ae(a, b):
    return abs(a-b) < 10**(-mp.dps+5)

def test_basic_integrals():
    for prec in [15, 30, 100]:
        mp.dps = prec
        assert ae(quadts(lambda x: x**3 - 3*x**2, [-2, 4]), -12)
        assert ae(quadgl(lambda x: x**3 - 3*x**2, [-2, 4]), -12)
        assert ae(quadts(sin, [0, pi]), 2)
        assert ae(quadts(sin, [0, 2*pi]), 0)
        assert ae(quadts(exp, [-inf, -1]), 1/e)
        assert ae(quadts(lambda x: exp(-x), [0, inf]), 1)
        assert ae(quadts(lambda x: exp(-x*x), [-inf, inf]), sqrt(pi))
        assert ae(quadts(lambda x: 1/(1+x*x), [-1, 1]), pi/2)
        assert ae(quadts(lambda x: 1/(1+x*x), [-inf, inf]), pi)
        assert ae(quadts(lambda x: 2*sqrt(1-x*x), [-1, 1]), pi)
    mp.dps = 15

def test_multiple_intervals():
    y,err = quad(lambda x: sign(x), [-0.5, 0.9, 1], maxdegree=2, error=True)
    assert abs(y-0.5) < 2*err

def test_quad_symmetry():
    assert quadts(sin, [-1, 1]) == 0
    assert quadgl(sin, [-1, 1]) == 0

def test_quad_infinite_mirror():
    # Check mirrored infinite interval
    assert ae(quad(lambda x: exp(-x*x), [inf,-inf]), -sqrt(pi))
    assert ae(quad(lambda x: exp(x), [0,-inf]), -1)

def test_quadgl_linear():
    assert quadgl(lambda x: x, [0, 1], maxdegree=1).ae(0.5)

def test_complex_integration():
    assert quadts(lambda x: x, [0, 1+j]).ae(j)

def test_quadosc():
    mp.dps = 15
    assert quadosc(lambda x: sin(x)/x, [0, inf], period=2*pi).ae(pi/2)

# Double integrals
def test_double_trivial():
    assert ae(quadts(lambda x, y: x, [0, 1], [0, 1]), 0.5)
    assert ae(quadts(lambda x, y: x, [-1, 1], [-1, 1]), 0.0)

def test_double_1():
    assert ae(quadts(lambda x, y: cos(x+y/2), [-pi/2, pi/2], [0, pi]), 4)

def test_double_2():
    assert ae(quadts(lambda x, y: (x-1)/((1-x*y)*log(x*y)), [0, 1], [0, 1]), euler)

def test_double_3():
    assert ae(quadts(lambda x, y: 1/sqrt(1+x*x+y*y), [-1, 1], [-1, 1]), 4*log(2+sqrt(3))-2*pi/3)

def test_double_4():
    assert ae(quadts(lambda x, y: 1/(1-x*x * y*y), [0, 1], [0, 1]), pi**2 / 8)

def test_double_5():
    assert ae(quadts(lambda x, y: 1/(1-x*y), [0, 1], [0, 1]), pi**2 / 6)

def test_double_6():
    assert ae(quadts(lambda x, y: exp(-(x+y)), [0, inf], [0, inf]), 1)

def test_double_7():
    assert ae(quadts(lambda x, y: exp(-x*x-y*y), [-inf, inf], [-inf, inf]), pi)


# Test integrals from "Experimentation in Mathematics" by Borwein,
# Bailey & Girgensohn
def test_expmath_integrals():
    for prec in [15, 30, 50]:
        mp.dps = prec
        assert ae(quadts(lambda x: x/sinh(x), [0, inf]),                    pi**2 / 4)
        assert ae(quadts(lambda x: log(x)**2 / (1+x**2), [0, inf]),         pi**3 / 8)
        assert ae(quadts(lambda x: (1+x**2)/(1+x**4), [0, inf]),            pi/sqrt(2))
        assert ae(quadts(lambda x: log(x)/cosh(x)**2, [0, inf]),            log(pi)-2*log(2)-euler)
        assert ae(quadts(lambda x: log(1+x**3)/(1-x+x**2), [0, inf]),       2*pi*log(3)/sqrt(3))
        assert ae(quadts(lambda x: log(x)**2 / (x**2+x+1), [0, 1]),         8*pi**3 / (81*sqrt(3)))
        assert ae(quadts(lambda x: log(cos(x))**2, [0, pi/2]),              pi/2 * (log(2)**2+pi**2/12))
        assert ae(quadts(lambda x: x**2 / sin(x)**2, [0, pi/2]),            pi*log(2))
        assert ae(quadts(lambda x: x**2/sqrt(exp(x)-1), [0, inf]),          4*pi*(log(2)**2 + pi**2/12))
        assert ae(quadts(lambda x: x*exp(-x)*sqrt(1-exp(-2*x)), [0, inf]),  pi*(1+2*log(2))/8)
    mp.dps = 15

# Do not reach full accuracy
@pytest.mark.xfail
def test_expmath_fail():
    assert ae(quadts(lambda x: sqrt(tan(x)), [0, pi/2]),          pi*sqrt(2)/2)
    assert ae(quadts(lambda x: atan(x)/(x*sqrt(1-x**2)), [0, 1]), pi*log(1+sqrt(2))/2)
    assert ae(quadts(lambda x: log(1+x**2)/x**2, [0, 1]),         pi/2-log(2))
    assert ae(quadts(lambda x: x**2/((1+x**4)*sqrt(1-x**4)), [0, 1]),     pi/8)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\char\__init__.py ===
from numpy._core.defchararray import *
from numpy._core.defchararray import __all__, __doc__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\rec\__init__.py ===
from numpy._core.records import *
from numpy._core.records import __all__, __doc__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\strings\__init__.py ===
from numpy._core.strings import *
from numpy._core.strings import __all__, __doc__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\numeric.py ===
"""
Tests for :mod:`numpy._core.numeric`.

Does not include tests which fall under ``array_constructors``.

"""

from __future__ import annotations
from typing import cast

import numpy as np
import numpy.typing as npt

class SubClass(npt.NDArray[np.float64]): ...


i8 = np.int64(1)

A = cast(
    np.ndarray[tuple[int, int, int], np.dtype[np.intp]],
    np.arange(27).reshape(3, 3, 3),
)
B: list[list[list[int]]] = A.tolist()
C = np.empty((27, 27)).view(SubClass)

np.count_nonzero(i8)
np.count_nonzero(A)
np.count_nonzero(B)
np.count_nonzero(A, keepdims=True)
np.count_nonzero(A, axis=0)

np.isfortran(i8)
np.isfortran(A)

np.argwhere(i8)
np.argwhere(A)

np.flatnonzero(i8)
np.flatnonzero(A)

np.correlate(B[0][0], A.ravel(), mode="valid")
np.correlate(A.ravel(), A.ravel(), mode="same")

np.convolve(B[0][0], A.ravel(), mode="valid")
np.convolve(A.ravel(), A.ravel(), mode="same")

np.outer(i8, A)
np.outer(B, A)
np.outer(A, A)
np.outer(A, A, out=C)

np.tensordot(B, A)
np.tensordot(A, A)
np.tensordot(A, A, axes=0)
np.tensordot(A, A, axes=(0, 1))

np.isscalar(i8)
np.isscalar(A)
np.isscalar(B)

np.roll(A, 1)
np.roll(A, (1, 2))
np.roll(B, 1)

np.rollaxis(A, 0, 1)

np.moveaxis(A, 0, 1)
np.moveaxis(A, (0, 1), (1, 2))

np.cross(B, A)
np.cross(A, A)

np.indices([0, 1, 2])
np.indices([0, 1, 2], sparse=False)
np.indices([0, 1, 2], sparse=True)

np.binary_repr(1)

np.base_repr(1)

np.allclose(i8, A)
np.allclose(B, A)
np.allclose(A, A)

np.isclose(i8, A)
np.isclose(B, A)
np.isclose(A, A)

np.array_equal(i8, A)
np.array_equal(B, A)
np.array_equal(A, A)

np.array_equiv(i8, A)
np.array_equiv(B, A)
np.array_equiv(A, A)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\capi\build_and_package_info.py ===
package_name = 'onnxruntime'
__version__ = '1.22.0'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\capi\version_info.py ===
use_cuda = False
vs2019 = False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\execution_providers\qnn\__init__.py ===
from .preprocess import qnn_preprocess_model  # noqa: F401
from .quant_config import get_qnn_qdq_config  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\__init__.py ===
# from .base_operator import QuantOperatorBase
# from .matmul import MatMulInteger

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_version_meson.py ===
__version__="2.3.0"
__git_version__="2cc37625532045f4ac55b27176454bbbc9baf213"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\api.py ===
__all__ = ["eval"]
from pandas.core.computation.eval import eval

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_normalize.py ===
from dateutil.tz import tzlocal
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DatetimeIndex,
    NaT,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestNormalize:
    def test_normalize(self):
        rng = date_range("1/1/2000 9:30", periods=10, freq="D")

        result = rng.normalize()
        expected = date_range("1/1/2000", periods=10, freq="D")
        tm.assert_index_equal(result, expected)

        arr_ns = np.array([1380585623454345752, 1380585612343234312]).astype(
            "datetime64[ns]"
        )
        rng_ns = DatetimeIndex(arr_ns)
        rng_ns_normalized = rng_ns.normalize()

        arr_ns = np.array([1380585600000000000, 1380585600000000000]).astype(
            "datetime64[ns]"
        )
        expected = DatetimeIndex(arr_ns)
        tm.assert_index_equal(rng_ns_normalized, expected)

        assert result.is_normalized
        assert not rng.is_normalized

    def test_normalize_nat(self):
        dti = DatetimeIndex([NaT, Timestamp("2018-01-01 01:00:00")])
        result = dti.normalize()
        expected = DatetimeIndex([NaT, Timestamp("2018-01-01")])
        tm.assert_index_equal(result, expected)

    def test_normalize_tz(self):
        rng = date_range("1/1/2000 9:30", periods=10, freq="D", tz="US/Eastern")

        result = rng.normalize()  # does not preserve freq
        expected = date_range("1/1/2000", periods=10, freq="D", tz="US/Eastern")
        tm.assert_index_equal(result, expected._with_freq(None))

        assert result.is_normalized
        assert not rng.is_normalized

        rng = date_range("1/1/2000 9:30", periods=10, freq="D", tz="UTC")

        result = rng.normalize()
        expected = date_range("1/1/2000", periods=10, freq="D", tz="UTC")
        tm.assert_index_equal(result, expected)

        assert result.is_normalized
        assert not rng.is_normalized

        rng = date_range("1/1/2000 9:30", periods=10, freq="D", tz=tzlocal())
        result = rng.normalize()  # does not preserve freq
        expected = date_range("1/1/2000", periods=10, freq="D", tz=tzlocal())
        tm.assert_index_equal(result, expected._with_freq(None))

        assert result.is_normalized
        assert not rng.is_normalized

    @td.skip_if_windows
    @pytest.mark.parametrize(
        "timezone",
        [
            "US/Pacific",
            "US/Eastern",
            "UTC",
            "Asia/Kolkata",
            "Asia/Shanghai",
            "Australia/Canberra",
        ],
    )
    def test_normalize_tz_local(self, timezone):
        # GH#13459
        with tm.set_timezone(timezone):
            rng = date_range("1/1/2000 9:30", periods=10, freq="D", tz=tzlocal())

            result = rng.normalize()
            expected = date_range("1/1/2000", periods=10, freq="D", tz=tzlocal())
            expected = expected._with_freq(None)
            tm.assert_index_equal(result, expected)

            assert result.is_normalized
            assert not rng.is_normalized

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\numeric\test_astype.py ===
import numpy as np
import pytest

from pandas import (
    Index,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm


class TestAstype:
    def test_astype_float64_to_uint64(self):
        # GH#45309 used to incorrectly return Index with int64 dtype
        idx = Index([0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64)
        result = idx.astype("u8")
        expected = Index([0, 5, 10, 15, 20], dtype=np.uint64)
        tm.assert_index_equal(result, expected, exact=True)

        idx_with_negatives = idx - 10
        with pytest.raises(ValueError, match="losslessly"):
            idx_with_negatives.astype(np.uint64)

    def test_astype_float64_to_object(self):
        float_index = Index([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float64)
        result = float_index.astype(object)
        assert result.equals(float_index)
        assert float_index.equals(result)
        assert isinstance(result, Index) and result.dtype == object

    def test_astype_float64_mixed_to_object(self):
        # mixed int-float
        idx = Index([1.5, 2, 3, 4, 5], dtype=np.float64)
        idx.name = "foo"
        result = idx.astype(object)
        assert result.equals(idx)
        assert idx.equals(result)
        assert isinstance(result, Index) and result.dtype == object

    @pytest.mark.parametrize("dtype", ["int16", "int32", "int64"])
    def test_astype_float64_to_int_dtype(self, dtype):
        # GH#12881
        # a float astype int
        idx = Index([0, 1, 2], dtype=np.float64)
        result = idx.astype(dtype)
        expected = Index([0, 1, 2], dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

        idx = Index([0, 1.1, 2], dtype=np.float64)
        result = idx.astype(dtype)
        expected = Index([0, 1, 2], dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_astype_float64_to_float_dtype(self, dtype):
        # GH#12881
        # a float astype int
        idx = Index([0, 1, 2], dtype=np.float64)
        result = idx.astype(dtype)
        assert isinstance(result, Index) and result.dtype == dtype

    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_astype_float_to_datetimelike(self, dtype):
        # GH#49660 pre-2.0 Index.astype from floating to M8/m8/Period raised,
        #  inconsistent with Series.astype
        idx = Index([0, 1.1, 2], dtype=np.float64)

        result = idx.astype(dtype)
        if dtype[0] == "M":
            expected = to_datetime(idx.values)
        else:
            expected = to_timedelta(idx.values)
        tm.assert_index_equal(result, expected)

        # check that we match Series behavior
        result = idx.to_series().set_axis(range(3)).astype(dtype)
        expected = expected.to_series().set_axis(range(3))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [int, "int16", "int32", "int64"])
    @pytest.mark.parametrize("non_finite", [np.inf, np.nan])
    def test_cannot_cast_inf_to_int(self, non_finite, dtype):
        # GH#13149
        idx = Index([1, 2, non_finite], dtype=np.float64)

        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(ValueError, match=msg):
            idx.astype(dtype)

    def test_astype_from_object(self):
        index = Index([1.0, np.nan, 0.2], dtype="object")
        result = index.astype(float)
        expected = Index([1.0, np.nan, 0.2], dtype=np.float64)
        assert result.dtype == expected.dtype
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\_version.py ===
# This file is protected via CODEOWNERS
__version__ = "1.26.20"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_eval.py ===
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, tan)
from sympy.testing.pytest import XFAIL


def test_add_eval():
    a = Symbol("a")
    b = Symbol("b")
    c = Rational(1)
    p = Rational(5)
    assert a*b + c + p == a*b + 6
    assert c + a + p == a + 6
    assert c + a - p == a + (-4)
    assert a + a == 2*a
    assert a + p + a == 2*a + 5
    assert c + p == Rational(6)
    assert b + a - b == a


def test_addmul_eval():
    a = Symbol("a")
    b = Symbol("b")
    c = Rational(1)
    p = Rational(5)
    assert c + a + b*c + a - p == 2*a + b + (-4)
    assert a*2 + p + a == a*2 + 5 + a
    assert a*2 + p + a == 3*a + 5
    assert a*2 + a == 3*a


def test_pow_eval():
    # XXX Pow does not fully support conversion of negative numbers
    #     to their complex equivalent

    assert sqrt(-1) == I

    assert sqrt(-4) == 2*I
    assert sqrt( 4) == 2
    assert (8)**Rational(1, 3) == 2
    assert (-8)**Rational(1, 3) == 2*((-1)**Rational(1, 3))

    assert sqrt(-2) == I*sqrt(2)
    assert (-1)**Rational(1, 3) != I
    assert (-10)**Rational(1, 3) != I*((10)**Rational(1, 3))
    assert (-2)**Rational(1, 4) != (2)**Rational(1, 4)

    assert 64**Rational(1, 3) == 4
    assert 64**Rational(2, 3) == 16
    assert 24/sqrt(64) == 3
    assert (-27)**Rational(1, 3) == 3*(-1)**Rational(1, 3)

    assert (cos(2) / tan(2))**2 == (cos(2) / tan(2))**2


@XFAIL
def test_pow_eval_X1():
    assert (-1)**Rational(1, 3) == S.Half + S.Half*I*sqrt(3)


def test_mulpow_eval():
    x = Symbol('x')
    assert sqrt(50)/(sqrt(2)*x) == 5/x
    assert sqrt(27)/sqrt(3) == 3


def test_evalpow_bug():
    x = Symbol("x")
    assert 1/(1/x) == x
    assert 1/(-1/x) == -x


def test_symbol_expand():
    x = Symbol('x')
    y = Symbol('y')

    f = x**4*y**4
    assert f == x**4*y**4
    assert f == f.expand()

    g = (x*y)**4
    assert g == f
    assert g.expand() == f
    assert g.expand() == g.expand().expand()


def test_function():
    f, l = map(Function, 'fl')
    x = Symbol('x')
    assert exp(l(x))*l(x)/exp(l(x)) == l(x)
    assert exp(f(x))*f(x)/exp(f(x)) == f(x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\latex\errors.py ===
class LaTeXParsingError(Exception):
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\latex\lark\__init__.py ===
from .latex_parser import parse_latex_lark, LarkLaTeXParser # noqa
from .transformer import TransformToSymPyExpr # noqa

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\tests\test_dimensionsystem.py ===
from sympy.core.symbol import symbols
from sympy.matrices.dense import (Matrix, eye)
from sympy.physics.units.definitions.dimension_definitions import (
    action, current, length, mass, time,
    velocity)
from sympy.physics.units.dimensions import DimensionSystem


def test_extend():
    ms = DimensionSystem((length, time), (velocity,))

    mks = ms.extend((mass,), (action,))

    res = DimensionSystem((length, time, mass), (velocity, action))
    assert mks.base_dims == res.base_dims
    assert mks.derived_dims == res.derived_dims


def test_list_dims():
    dimsys = DimensionSystem((length, time, mass))

    assert dimsys.list_can_dims == (length, mass, time)


def test_dim_can_vector():
    dimsys = DimensionSystem(
        [length, mass, time],
        [velocity, action],
        {
            velocity: {length: 1, time: -1}
        }
    )

    assert dimsys.dim_can_vector(length) == Matrix([1, 0, 0])
    assert dimsys.dim_can_vector(velocity) == Matrix([1, 0, -1])

    dimsys = DimensionSystem(
        (length, velocity, action),
        (mass, time),
        {
            time: {length: 1, velocity: -1}
        }
    )

    assert dimsys.dim_can_vector(length) == Matrix([0, 1, 0])
    assert dimsys.dim_can_vector(velocity) == Matrix([0, 0, 1])
    assert dimsys.dim_can_vector(time) == Matrix([0, 1, -1])

    dimsys = DimensionSystem(
        (length, mass, time),
        (velocity, action),
        {velocity: {length: 1, time: -1},
         action: {mass: 1, length: 2, time: -1}})

    assert dimsys.dim_vector(length) == Matrix([1, 0, 0])
    assert dimsys.dim_vector(velocity) == Matrix([1, 0, -1])


def test_inv_can_transf_matrix():
    dimsys = DimensionSystem((length, mass, time))
    assert dimsys.inv_can_transf_matrix == eye(3)


def test_can_transf_matrix():
    dimsys = DimensionSystem((length, mass, time))
    assert dimsys.can_transf_matrix == eye(3)

    dimsys = DimensionSystem((length, velocity, action))
    assert dimsys.can_transf_matrix == eye(3)

    dimsys = DimensionSystem((length, time), (velocity,), {velocity: {length: 1, time: -1}})
    assert dimsys.can_transf_matrix == eye(2)


def test_is_consistent():
    assert DimensionSystem((length, time)).is_consistent is True


def test_print_dim_base():
    mksa = DimensionSystem(
        (length, time, mass, current),
        (action,),
        {action: {mass: 1, length: 2, time: -1}})
    L, M, T = symbols("L M T")
    assert mksa.print_dim_base(action) == L**2*M/T


def test_dim():
    dimsys = DimensionSystem(
        (length, mass, time),
        (velocity, action),
        {velocity: {length: 1, time: -1},
         action: {mass: 1, length: 2, time: -1}}
    )
    assert dimsys.dim == 3

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_dispersion.py ===
from sympy.core import Symbol, S, oo
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import poly
from sympy.polys.dispersion import dispersion, dispersionset


def test_dispersion():
    x = Symbol("x")
    a = Symbol("a")

    fp = poly(S.Zero, x)
    assert sorted(dispersionset(fp)) == [0]

    fp = poly(S(2), x)
    assert sorted(dispersionset(fp)) == [0]

    fp = poly(x + 1, x)
    assert sorted(dispersionset(fp)) == [0]
    assert dispersion(fp) == 0

    fp = poly((x + 1)*(x + 2), x)
    assert sorted(dispersionset(fp)) == [0, 1]
    assert dispersion(fp) == 1

    fp = poly(x*(x + 3), x)
    assert sorted(dispersionset(fp)) == [0, 3]
    assert dispersion(fp) == 3

    fp = poly((x - 3)*(x + 3), x)
    assert sorted(dispersionset(fp)) == [0, 6]
    assert dispersion(fp) == 6

    fp = poly(x**4 - 3*x**2 + 1, x)
    gp = fp.shift(-3)
    assert sorted(dispersionset(fp, gp)) == [2, 3, 4]
    assert dispersion(fp, gp) == 4
    assert sorted(dispersionset(gp, fp)) == []
    assert dispersion(gp, fp) is -oo

    fp = poly(x*(3*x**2+a)*(x-2536)*(x**3+a), x)
    gp = fp.as_expr().subs(x, x-345).as_poly(x)
    assert sorted(dispersionset(fp, gp)) == [345, 2881]
    assert sorted(dispersionset(gp, fp)) == [2191]

    gp = poly((x-2)**2*(x-3)**3*(x-5)**3, x)
    assert sorted(dispersionset(gp)) == [0, 1, 2, 3]
    assert sorted(dispersionset(gp, (gp+4)**2)) == [1, 2]

    fp = poly(x*(x+2)*(x-1), x)
    assert sorted(dispersionset(fp)) == [0, 1, 2, 3]

    fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
    gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
    assert sorted(dispersionset(fp, gp)) == [2]
    assert sorted(dispersionset(gp, fp)) == [1, 4]

    # There are some difficulties if we compute over Z[a]
    # and alpha happens to lie in Z[a] instead of simply Z.
    # Hence we can not decide if alpha is indeed integral
    # in general.

    fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
    assert sorted(dispersionset(fp)) == [0, 1]

    # For any specific value of a, the dispersion is 3*a
    # but the algorithm can not find this in general.
    # This is the point where the resultant based Ansatz
    # is superior to the current one.
    fp = poly(a**2*x**3 + (a**3 + a**2 + a + 1)*x, x)
    gp = fp.as_expr().subs(x, x - 3*a).as_poly(x)
    assert sorted(dispersionset(fp, gp)) == []

    fpa = fp.as_expr().subs(a, 2).as_poly(x)
    gpa = gp.as_expr().subs(a, 2).as_poly(x)
    assert sorted(dispersionset(fpa, gpa)) == [6]

    # Work with Expr instead of Poly
    f = (x + 1)*(x + 2)
    assert sorted(dispersionset(f)) == [0, 1]
    assert dispersion(f) == 1

    f = x**4 - 3*x**2 + 1
    g = x**4 - 12*x**3 + 51*x**2 - 90*x + 55
    assert sorted(dispersionset(f, g)) == [2, 3, 4]
    assert dispersion(f, g) == 4

    # Work with Expr and specify a generator
    f = (x + 1)*(x + 2)
    assert sorted(dispersionset(f, None, x)) == [0, 1]
    assert dispersion(f, None, x) == 1

    f = x**4 - 3*x**2 + 1
    g = x**4 - 12*x**3 + 51*x**2 - 90*x + 55
    assert sorted(dispersionset(f, g, x)) == [2, 3, 4]
    assert dispersion(f, g, x) == 4

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\sampling\tests\test_sample_finite_rv.py ===
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.external import import_module
from sympy.stats import Binomial, sample, Die, FiniteRV, DiscreteUniform, Bernoulli, BetaBinomial, Hypergeometric, \
    Rademacher
from sympy.testing.pytest import skip, raises

def test_given_sample():
    X = Die('X', 6)
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    assert sample(X, X > 5) == 6

def test_sample_numpy():
    distribs_numpy = [
        Binomial("B", 5, 0.4),
        Hypergeometric("H", 2, 1, 1)
    ]
    size = 3
    numpy = import_module('numpy')
    if not numpy:
        skip('Numpy is not installed. Abort tests for _sample_numpy.')
    else:
        for X in distribs_numpy:
            samps = sample(X, size=size, library='numpy')
            for sam in samps:
                assert sam in X.pspace.domain.set
        raises(NotImplementedError,
               lambda: sample(Die("D"), library='numpy'))
    raises(NotImplementedError,
           lambda: Die("D").pspace.sample(library='tensorflow'))


def test_sample_scipy():
    distribs_scipy = [
        FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)}),
        DiscreteUniform("Y", list(range(5))),
        Die("D"),
        Bernoulli("Be", 0.3),
        Binomial("Bi", 5, 0.4),
        BetaBinomial("Bb", 2, 1, 1),
        Hypergeometric("H", 1, 1, 1),
        Rademacher("R")
    ]

    size = 3
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            samps = sample(X, size=size)
            samps2 = sample(X, size=(2, 2))
            for sam in samps:
                assert sam in X.pspace.domain.set
            for i in range(2):
                for j in range(2):
                    assert samps2[i][j] in X.pspace.domain.set


def test_sample_pymc():
    distribs_pymc = [
        Bernoulli('B', 0.2),
        Binomial('N', 5, 0.4)
    ]
    size = 3
    pymc = import_module('pymc')
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        for X in distribs_pymc:
            samps = sample(X, size=size, library='pymc')
            for sam in samps:
                assert sam in X.pspace.domain.set
        raises(NotImplementedError,
                          lambda: (sample(Die("D"), library='pymc')))


def test_sample_seed():
    F = FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)})
    size = 10
    libraries = ['scipy', 'numpy', 'pymc']
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0 = sample(F, size=size, library=lib, seed=0)
                s1 = sample(F, size=size, library=lib, seed=0)
                s2 = sample(F, size=size, library=lib, seed=1)
                assert all(s0 == s1)
                assert not all(s1 == s2)
        except NotImplementedError:
            continue

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_aggregation.py ===
import numpy as np
import pytest

from pandas.core.apply import (
    _make_unique_kwarg_list,
    maybe_mangle_lambdas,
)


def test_maybe_mangle_lambdas_passthrough():
    assert maybe_mangle_lambdas("mean") == "mean"
    assert maybe_mangle_lambdas(lambda x: x).__name__ == "<lambda>"
    # don't mangel single lambda.
    assert maybe_mangle_lambdas([lambda x: x])[0].__name__ == "<lambda>"


def test_maybe_mangle_lambdas_listlike():
    aggfuncs = [lambda x: 1, lambda x: 2]
    result = maybe_mangle_lambdas(aggfuncs)
    assert result[0].__name__ == "<lambda_0>"
    assert result[1].__name__ == "<lambda_1>"
    assert aggfuncs[0](None) == result[0](None)
    assert aggfuncs[1](None) == result[1](None)


def test_maybe_mangle_lambdas():
    func = {"A": [lambda x: 0, lambda x: 1]}
    result = maybe_mangle_lambdas(func)
    assert result["A"][0].__name__ == "<lambda_0>"
    assert result["A"][1].__name__ == "<lambda_1>"


def test_maybe_mangle_lambdas_args():
    func = {"A": [lambda x, a, b=1: (0, a, b), lambda x: 1]}
    result = maybe_mangle_lambdas(func)
    assert result["A"][0].__name__ == "<lambda_0>"
    assert result["A"][1].__name__ == "<lambda_1>"

    assert func["A"][0](0, 1) == (0, 1, 1)
    assert func["A"][0](0, 1, 2) == (0, 1, 2)
    assert func["A"][0](0, 2, b=3) == (0, 2, 3)


def test_maybe_mangle_lambdas_named():
    func = {"C": np.mean, "D": {"foo": np.mean, "bar": np.mean}}
    result = maybe_mangle_lambdas(func)
    assert result == func


@pytest.mark.parametrize(
    "order, expected_reorder",
    [
        (
            [
                ("height", "<lambda>"),
                ("height", "max"),
                ("weight", "max"),
                ("height", "<lambda>"),
                ("weight", "<lambda>"),
            ],
            [
                ("height", "<lambda>_0"),
                ("height", "max"),
                ("weight", "max"),
                ("height", "<lambda>_1"),
                ("weight", "<lambda>"),
            ],
        ),
        (
            [
                ("col2", "min"),
                ("col1", "<lambda>"),
                ("col1", "<lambda>"),
                ("col1", "<lambda>"),
            ],
            [
                ("col2", "min"),
                ("col1", "<lambda>_0"),
                ("col1", "<lambda>_1"),
                ("col1", "<lambda>_2"),
            ],
        ),
        (
            [("col", "<lambda>"), ("col", "<lambda>"), ("col", "<lambda>")],
            [("col", "<lambda>_0"), ("col", "<lambda>_1"), ("col", "<lambda>_2")],
        ),
    ],
)
def test_make_unique(order, expected_reorder):
    # GH 27519, test if make_unique function reorders correctly
    result = _make_unique_kwarg_list(order)

    assert result == expected_reorder

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\interval\test_overlaps.py ===
"""Tests for Interval-Interval operations, such as overlaps, contains, etc."""
import numpy as np
import pytest

from pandas import (
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


@pytest.fixture(params=[IntervalArray, IntervalIndex])
def constructor(request):
    """
    Fixture for testing both interval container classes.
    """
    return request.param


@pytest.fixture(
    params=[
        (Timedelta("0 days"), Timedelta("1 day")),
        (Timestamp("2018-01-01"), Timedelta("1 day")),
        (0, 1),
    ],
    ids=lambda x: type(x[0]).__name__,
)
def start_shift(request):
    """
    Fixture for generating intervals of different types from a start value
    and a shift value that can be added to start to generate an endpoint.
    """
    return request.param


class TestOverlaps:
    def test_overlaps_interval(self, constructor, start_shift, closed, other_closed):
        start, shift = start_shift
        interval = Interval(start, start + 3 * shift, other_closed)

        # intervals: identical, nested, spanning, partial, adjacent, disjoint
        tuples = [
            (start, start + 3 * shift),
            (start + shift, start + 2 * shift),
            (start - shift, start + 4 * shift),
            (start + 2 * shift, start + 4 * shift),
            (start + 3 * shift, start + 4 * shift),
            (start + 4 * shift, start + 5 * shift),
        ]
        interval_container = constructor.from_tuples(tuples, closed)

        adjacent = interval.closed_right and interval_container.closed_left
        expected = np.array([True, True, True, True, adjacent, False])
        result = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("other_constructor", [IntervalArray, IntervalIndex])
    def test_overlaps_interval_container(self, constructor, other_constructor):
        # TODO: modify this test when implemented
        interval_container = constructor.from_breaks(range(5))
        other_container = other_constructor.from_breaks(range(5))
        with pytest.raises(NotImplementedError, match="^$"):
            interval_container.overlaps(other_container)

    def test_overlaps_na(self, constructor, start_shift):
        """NA values are marked as False"""
        start, shift = start_shift
        interval = Interval(start, start + shift)

        tuples = [
            (start, start + shift),
            np.nan,
            (start + 2 * shift, start + 3 * shift),
        ]
        interval_container = constructor.from_tuples(tuples)

        expected = np.array([True, False, False])
        result = interval_container.overlaps(interval)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [10, True, "foo", Timedelta("1 day"), Timestamp("2018-01-01")],
        ids=lambda x: type(x).__name__,
    )
    def test_overlaps_invalid_type(self, constructor, other):
        interval_container = constructor.from_breaks(range(5))
        msg = f"`other` must be Interval-like, got {type(other).__name__}"
        with pytest.raises(TypeError, match=msg):
            interval_container.overlaps(other)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\tests\test_polynomialring.py ===
"""Tests for the PolynomialRing classes. """

from sympy.polys.domains import QQ, ZZ
from sympy.polys.polyerrors import ExactQuotientFailed, CoercionFailed, NotReversible

from sympy.abc import x, y

from sympy.testing.pytest import raises


def test_build_order():
    R = QQ.old_poly_ring(x, y, order=(("lex", x), ("ilex", y)))
    assert R.order((1, 5)) == ((1,), (-5,))


def test_globalring():
    Qxy = QQ.old_frac_field(x, y)
    R = QQ.old_poly_ring(x, y)
    X = R.convert(x)
    Y = R.convert(y)

    assert x in R
    assert 1/x not in R
    assert 1/(1 + x) not in R
    assert Y in R
    assert X * (Y**2 + 1) == R.convert(x * (y**2 + 1))
    assert X + 1 == R.convert(x + 1)
    raises(ExactQuotientFailed, lambda: X/Y)
    raises(TypeError, lambda: x/Y)
    raises(TypeError, lambda: X/y)
    assert X**2 / X == X

    assert R.from_GlobalPolynomialRing(ZZ.old_poly_ring(x, y).convert(x), ZZ.old_poly_ring(x, y)) == X
    assert R.from_FractionField(Qxy.convert(x), Qxy) == X
    assert R.from_FractionField(Qxy.convert(x/y), Qxy) is None

    assert R._sdm_to_vector(R._vector_to_sdm([X, Y], R.order), 2) == [X, Y]


def test_localring():
    Qxy = QQ.old_frac_field(x, y)
    R = QQ.old_poly_ring(x, y, order="ilex")
    X = R.convert(x)
    Y = R.convert(y)

    assert x in R
    assert 1/x not in R
    assert 1/(1 + x) in R
    assert Y in R
    assert X*(Y**2 + 1)/(1 + X) == R.convert(x*(y**2 + 1)/(1 + x))
    raises(TypeError, lambda: x/Y)
    raises(TypeError, lambda: X/y)
    assert X + 1 == R.convert(x + 1)
    assert X**2 / X == X

    assert R.from_GlobalPolynomialRing(ZZ.old_poly_ring(x, y).convert(x), ZZ.old_poly_ring(x, y)) == X
    assert R.from_FractionField(Qxy.convert(x), Qxy) == X
    raises(CoercionFailed, lambda: R.from_FractionField(Qxy.convert(x/y), Qxy))
    raises(ExactQuotientFailed, lambda: R.exquo(X, Y))
    raises(NotReversible, lambda: R.revert(X))

    assert R._sdm_to_vector(
        R._vector_to_sdm([X/(X + 1), Y/(1 + X*Y)], R.order), 2) == \
        [X*(1 + X*Y), Y*(1 + X)]


def test_conversion():
    L = QQ.old_poly_ring(x, y, order="ilex")
    G = QQ.old_poly_ring(x, y)

    assert L.convert(x) == L.convert(G.convert(x), G)
    assert G.convert(x) == G.convert(L.convert(x), L)
    raises(CoercionFailed, lambda: G.convert(L.convert(1/(1 + x)), L))


def test_units():
    R = QQ.old_poly_ring(x)
    assert R.is_unit(R.convert(1))
    assert R.is_unit(R.convert(2))
    assert not R.is_unit(R.convert(x))
    assert not R.is_unit(R.convert(1 + x))

    R = QQ.old_poly_ring(x, order='ilex')
    assert R.is_unit(R.convert(1))
    assert R.is_unit(R.convert(2))
    assert not R.is_unit(R.convert(x))
    assert R.is_unit(R.convert(1 + x))

    R = ZZ.old_poly_ring(x)
    assert R.is_unit(R.convert(1))
    assert not R.is_unit(R.convert(2))
    assert not R.is_unit(R.convert(x))
    assert not R.is_unit(R.convert(1 + x))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_argparse.py ===
"""
Tests for the private NumPy argument parsing functionality.
They mainly exists to ensure good test coverage without having to try the
weirder cases on actual numpy functions but test them in one place.

The test function is defined in C to be equivalent to (errors may not always
match exactly, and could be adjusted):

    def func(arg1, /, arg2, *, arg3):
        i = integer(arg1)  # reproducing the 'i' parsing in Python.
        return None
"""

import threading

import pytest
from numpy._core._multiarray_tests import (
    argparse_example_function as func,
)
from numpy._core._multiarray_tests import (
    threaded_argparse_example_function as thread_func,
)

import numpy as np
from numpy.testing import IS_WASM


@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for threads")
def test_thread_safe_argparse_cache():
    b = threading.Barrier(8)

    def call_thread_func():
        b.wait()
        thread_func(arg1=3, arg2=None)

    tasks = [threading.Thread(target=call_thread_func) for _ in range(8)]
    [t.start() for t in tasks]
    [t.join() for t in tasks]


def test_invalid_integers():
    with pytest.raises(TypeError,
            match="integer argument expected, got float"):
        func(1.)
    with pytest.raises(OverflowError):
        func(2**100)


def test_missing_arguments():
    with pytest.raises(TypeError,
            match="missing required positional argument 0"):
        func()
    with pytest.raises(TypeError,
            match="missing required positional argument 0"):
        func(arg2=1, arg3=4)
    with pytest.raises(TypeError,
            match=r"missing required argument \'arg2\' \(pos 1\)"):
        func(1, arg3=5)


def test_too_many_positional():
    # the second argument is positional but can be passed as keyword.
    with pytest.raises(TypeError,
            match="takes from 2 to 3 positional arguments but 4 were given"):
        func(1, 2, 3, 4)


def test_multiple_values():
    with pytest.raises(TypeError,
            match=r"given by name \('arg2'\) and position \(position 1\)"):
        func(1, 2, arg2=3)


def test_string_fallbacks():
    # We can (currently?) use numpy strings to test the "slow" fallbacks
    # that should normally not be taken due to string interning.
    arg2 = np.str_("arg2")
    missing_arg = np.str_("missing_arg")
    func(1, **{arg2: 3})
    with pytest.raises(TypeError,
            match="got an unexpected keyword argument 'missing_arg'"):
        func(2, **{missing_arg: 3})


def test_too_many_arguments_method_forwarding():
    # Not directly related to the standard argument parsing, but we sometimes
    # forward methods to Python: arr.mean() calls np._core._methods._mean()
    # This adds code coverage for this `npy_forward_method`.
    arr = np.arange(3)
    args = range(1000)
    with pytest.raises(TypeError):
        arr.mean(*args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_take.py ===
import pytest

import pandas._testing as tm


class TestDataFrameTake:
    def test_take_slices_deprecated(self, float_frame):
        # GH#51539
        df = float_frame

        slc = slice(0, 4, 1)
        with tm.assert_produces_warning(FutureWarning):
            df.take(slc, axis=0)
        with tm.assert_produces_warning(FutureWarning):
            df.take(slc, axis=1)

    def test_take(self, float_frame):
        # homogeneous
        order = [3, 1, 2, 0]
        for df in [float_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["D", "B", "C", "A"]]
            tm.assert_frame_equal(result, expected, check_names=False)

        # negative indices
        order = [2, 1, -1]
        for df in [float_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            result = df.take(order, axis=0)
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["C", "B", "D"]]
            tm.assert_frame_equal(result, expected, check_names=False)

        # illegal indices
        msg = "indices are out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, 30], axis=0)
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, -31], axis=0)
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, 5], axis=1)
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, -5], axis=1)

    def test_take_mixed_type(self, float_string_frame):
        # mixed-dtype
        order = [4, 1, 2, 0, 3]
        for df in [float_string_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["foo", "B", "C", "A", "D"]]
            tm.assert_frame_equal(result, expected)

        # negative indices
        order = [4, 1, -2]
        for df in [float_string_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["foo", "B", "D"]]
            tm.assert_frame_equal(result, expected)

    def test_take_mixed_numeric(self, mixed_float_frame, mixed_int_frame):
        # by dtype
        order = [1, 2, 0, 3]
        for df in [mixed_float_frame, mixed_int_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["B", "C", "A", "D"]]
            tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_retain_attributes.py ===
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    _testing as tm,
    date_range,
    errors,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)

pytestmark = pytest.mark.single_cpu


def test_retain_index_attributes(setup_path, unit):
    # GH 3499, losing frequency info on index recreation
    dti = date_range("2000-1-1", periods=3, freq="h", unit=unit)
    df = DataFrame({"A": Series(range(3), index=dti)})

    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, "data")
        store.put("data", df, format="table")

        result = store.get("data")
        tm.assert_frame_equal(df, result)

        for attr in ["freq", "tz", "name"]:
            for idx in ["index", "columns"]:
                assert getattr(getattr(df, idx), attr, None) == getattr(
                    getattr(result, idx), attr, None
                )

        dti2 = date_range("2002-1-1", periods=3, freq="D", unit=unit)
        # try to append a table with a different frequency
        with tm.assert_produces_warning(errors.AttributeConflictWarning):
            df2 = DataFrame({"A": Series(range(3), index=dti2)})
            store.append("data", df2)

        assert store.get_storer("data").info["index"]["freq"] is None

        # this is ok
        _maybe_remove(store, "df2")
        dti3 = DatetimeIndex(
            ["2001-01-01", "2001-01-02", "2002-01-01"], dtype=f"M8[{unit}]"
        )
        df2 = DataFrame(
            {
                "A": Series(
                    range(3),
                    index=dti3,
                )
            }
        )
        store.append("df2", df2)
        dti4 = date_range("2002-1-1", periods=3, freq="D", unit=unit)
        df3 = DataFrame({"A": Series(range(3), index=dti4)})
        store.append("df2", df3)


def test_retain_index_attributes2(tmp_path, setup_path):
    path = tmp_path / setup_path

    with tm.assert_produces_warning(errors.AttributeConflictWarning):
        df = DataFrame(
            {"A": Series(range(3), index=date_range("2000-1-1", periods=3, freq="h"))}
        )
        df.to_hdf(path, key="data", mode="w", append=True)
        df2 = DataFrame(
            {"A": Series(range(3), index=date_range("2002-1-1", periods=3, freq="D"))}
        )

        df2.to_hdf(path, key="data", append=True)

        idx = date_range("2000-1-1", periods=3, freq="h")
        idx.name = "foo"
        df = DataFrame({"A": Series(range(3), index=idx)})
        df.to_hdf(path, key="data", mode="w", append=True)

    assert read_hdf(path, key="data").index.name == "foo"

    with tm.assert_produces_warning(errors.AttributeConflictWarning):
        idx2 = date_range("2001-1-1", periods=3, freq="h")
        idx2.name = "bar"
        df2 = DataFrame({"A": Series(range(3), index=idx2)})
        df2.to_hdf(path, key="data", append=True)

    assert read_hdf(path, "data").index.name is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_grover.py ===
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import IntQubit
from sympy.physics.quantum.grover import (apply_grover, superposition_basis,
        OracleGate, grover_iteration, WGate)


def return_one_on_two(qubits):
    return qubits == IntQubit(2, qubits.nqubits)


def return_one_on_one(qubits):
    return qubits == IntQubit(1, nqubits=qubits.nqubits)


def test_superposition_basis():
    nbits = 2
    first_half_state = IntQubit(0, nqubits=nbits)/2 + IntQubit(1, nqubits=nbits)/2
    second_half_state = IntQubit(2, nbits)/2 + IntQubit(3, nbits)/2
    assert first_half_state + second_half_state == superposition_basis(nbits)

    nbits = 3
    firstq = (1/sqrt(8))*IntQubit(0, nqubits=nbits) + (1/sqrt(8))*IntQubit(1, nqubits=nbits)
    secondq = (1/sqrt(8))*IntQubit(2, nbits) + (1/sqrt(8))*IntQubit(3, nbits)
    thirdq = (1/sqrt(8))*IntQubit(4, nbits) + (1/sqrt(8))*IntQubit(5, nbits)
    fourthq = (1/sqrt(8))*IntQubit(6, nbits) + (1/sqrt(8))*IntQubit(7, nbits)
    assert firstq + secondq + thirdq + fourthq == superposition_basis(nbits)


def test_OracleGate():
    v = OracleGate(1, lambda qubits: qubits == IntQubit(0))
    assert qapply(v*IntQubit(0)) == -IntQubit(0)
    assert qapply(v*IntQubit(1)) == IntQubit(1)

    nbits = 2
    v = OracleGate(2, return_one_on_two)
    assert qapply(v*IntQubit(0, nbits)) == IntQubit(0, nqubits=nbits)
    assert qapply(v*IntQubit(1, nbits)) == IntQubit(1, nqubits=nbits)
    assert qapply(v*IntQubit(2, nbits)) == -IntQubit(2, nbits)
    assert qapply(v*IntQubit(3, nbits)) == IntQubit(3, nbits)

    assert represent(OracleGate(1, lambda qubits: qubits == IntQubit(0)), nqubits=1) == \
           Matrix([[-1, 0], [0, 1]])
    assert represent(v, nqubits=2) == Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def test_WGate():
    nqubits = 2
    basis_states = superposition_basis(nqubits)
    assert qapply(WGate(nqubits)*basis_states) == basis_states

    expected = ((2/sqrt(pow(2, nqubits)))*basis_states) - IntQubit(1, nqubits=nqubits)
    assert qapply(WGate(nqubits)*IntQubit(1, nqubits=nqubits)) == expected


def test_grover_iteration_1():
    numqubits = 2
    basis_states = superposition_basis(numqubits)
    v = OracleGate(numqubits, return_one_on_one)
    expected = IntQubit(1, nqubits=numqubits)
    assert qapply(grover_iteration(basis_states, v)) == expected


def test_grover_iteration_2():
    numqubits = 4
    basis_states = superposition_basis(numqubits)
    v = OracleGate(numqubits, return_one_on_two)
    # After (pi/4)sqrt(pow(2, n)), IntQubit(2) should have highest prob
    # In this case, after around pi times (3 or 4)
    iterated = grover_iteration(basis_states, v)
    iterated = qapply(iterated)
    iterated = grover_iteration(iterated, v)
    iterated = qapply(iterated)
    iterated = grover_iteration(iterated, v)
    iterated = qapply(iterated)
    # In this case, probability was highest after 3 iterations
    # Probability of Qubit('0010') was 251/256 (3) vs 781/1024 (4)
    # Ask about measurement
    expected = (-13*basis_states)/64 + 264*IntQubit(2, numqubits)/256
    assert qapply(expected) == iterated


def test_grover():
    nqubits = 2
    assert apply_grover(return_one_on_one, nqubits) == IntQubit(1, nqubits=nqubits)

    nqubits = 4
    basis_states = superposition_basis(nqubits)
    expected = (-13*basis_states)/64 + 264*IntQubit(2, nqubits)/256
    assert apply_grover(return_one_on_two, 4) == qapply(expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\tests\test_tree.py ===
from sympy.strategies.tree import treeapply, greedy, allresults, brute
from functools import partial, reduce


def inc(x):
    return x + 1


def dec(x):
    return x - 1


def double(x):
    return 2 * x


def square(x):
    return x**2


def add(*args):
    return sum(args)


def mul(*args):
    return reduce(lambda a, b: a * b, args, 1)


def test_treeapply():
    tree = ([3, 3], [4, 1], 2)
    assert treeapply(tree, {list: min, tuple: max}) == 3
    assert treeapply(tree, {list: add, tuple: mul}) == 60


def test_treeapply_leaf():
    assert treeapply(3, {}, leaf=lambda x: x**2) == 9
    tree = ([3, 3], [4, 1], 2)
    treep1 = ([4, 4], [5, 2], 3)
    assert treeapply(tree, {list: min, tuple: max}, leaf=lambda x: x + 1) == \
           treeapply(treep1, {list: min, tuple: max})


def test_treeapply_strategies():
    from sympy.strategies import chain, minimize
    join = {list: chain, tuple: minimize}

    assert treeapply(inc, join) == inc
    assert treeapply((inc, dec), join)(5) == minimize(inc, dec)(5)
    assert treeapply([inc, dec], join)(5) == chain(inc, dec)(5)
    tree = (inc, [dec, double])  # either inc or dec-then-double
    assert treeapply(tree, join)(5) == 6
    assert treeapply(tree, join)(1) == 0

    maximize = partial(minimize, objective=lambda x: -x)
    join = {list: chain, tuple: maximize}
    fn = treeapply(tree, join)
    assert fn(4) == 6  # highest value comes from the dec then double
    assert fn(1) == 2  # highest value comes from the inc


def test_greedy():
    tree = [inc, (dec, double)]  # either inc or dec-then-double

    fn = greedy(tree, objective=lambda x: -x)
    assert fn(4) == 6  # highest value comes from the dec then double
    assert fn(1) == 2  # highest value comes from the inc

    tree = [inc, dec, [inc, dec, [(inc, inc), (dec, dec)]]]
    lowest = greedy(tree)
    assert lowest(10) == 8

    highest = greedy(tree, objective=lambda x: -x)
    assert highest(10) == 12


def test_allresults():
    # square = lambda x: x**2

    assert set(allresults(inc)(3)) == {inc(3)}
    assert set(allresults([inc, dec])(3)) == {2, 4}
    assert set(allresults((inc, dec))(3)) == {3}
    assert set(allresults([inc, (dec, double)])(4)) == {5, 6}


def test_brute():
    tree = ([inc, dec], square)
    fn = brute(tree, lambda x: -x)

    assert fn(2) == (2 + 1)**2
    assert fn(-2) == (-2 - 1)**2

    assert brute(inc)(1) == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_rootfinding.py ===
import pytest
from mpmath import *
from mpmath.calculus.optimization import Secant, Muller, Bisection, Illinois, \
    Pegasus, Anderson, Ridder, ANewton, Newton, MNewton, MDNewton

def test_findroot():
    # old tests, assuming secant
    mp.dps = 15
    assert findroot(lambda x: 4*x-3, mpf(5)).ae(0.75)
    assert findroot(sin, mpf(3)).ae(pi)
    assert findroot(sin, (mpf(3), mpf(3.14))).ae(pi)
    assert findroot(lambda x: x*x+1, mpc(2+2j)).ae(1j)
    # test all solvers with 1 starting point
    f = lambda x: cos(x)
    for solver in [Newton, Secant, MNewton, Muller, ANewton]:
        x = findroot(f, 2., solver=solver)
        assert abs(f(x)) < eps
    # test all solvers with interval of 2 points
    for solver in [Secant, Muller, Bisection, Illinois, Pegasus, Anderson,
                   Ridder]:
        x = findroot(f, (1., 2.), solver=solver)
        assert abs(f(x)) < eps
    # test types
    f = lambda x: (x - 2)**2

    assert isinstance(findroot(f, 1, tol=1e-10), mpf)
    assert isinstance(iv.findroot(f, 1., tol=1e-10), iv.mpf)
    assert isinstance(fp.findroot(f, 1, tol=1e-10), float)
    assert isinstance(fp.findroot(f, 1+0j, tol=1e-10), complex)

    # issue 401
    with pytest.raises(ValueError):
        with workprec(2):
            findroot(lambda x: x**2 - 4456178*x + 60372201703370,
                     mpc(real='5.278e+13', imag='-5.278e+13'))

    # issue 192
    with pytest.raises(ValueError):
        findroot(lambda x: -1, 0)

    # issue 387
    with pytest.raises(ValueError):
        findroot(lambda p: (1 - p)**30 - 1, 0.9)

def test_bisection():
    # issue 273
    assert findroot(lambda x: x**2-1,(0,2),solver='bisect') == 1

def test_mnewton():
    f = lambda x: polyval([1,3,3,1],x)
    x = findroot(f, -0.9, solver='mnewton')
    assert abs(f(x)) < eps

def test_anewton():
    f = lambda x: (x - 2)**100
    x = findroot(f, 1., solver=ANewton)
    assert abs(f(x)) < eps

def test_muller():
    f = lambda x: (2 + x)**3 + 2
    x = findroot(f, 1., solver=Muller)
    assert abs(f(x)) < eps

def test_multiplicity():
    for i in range(1, 5):
        assert multiplicity(lambda x: (x - 1)**i, 1) == i
    assert multiplicity(lambda x: x**2, 1) == 0

def test_multidimensional():
    def f(*x):
        return [3*x[0]**2-2*x[1]**2-1, x[0]**2-2*x[0]+x[1]**2+2*x[1]-8]
    assert mnorm(jacobian(f, (1,-2)) - matrix([[6,8],[0,-2]]),1) < 1.e-7
    for x, error in MDNewton(mp, f, (1,-2), verbose=0,
                             norm=lambda x: norm(x, inf)):
        pass
    assert norm(f(*x), 2) < 1e-14
    # The Chinese mathematician Zhu Shijie was the very first to solve this
    # nonlinear system 700 years ago
    f1 = lambda x, y: -x + 2*y
    f2 = lambda x, y: (x**2 + x*(y**2 - 2) - 4*y)  /  (x + 4)
    f3 = lambda x, y: sqrt(x**2 + y**2)
    def f(x, y):
        f1x = f1(x, y)
        return (f2(x, y) - f1x, f3(x, y) - f1x)
    x = findroot(f, (10, 10))
    assert [int(round(i)) for i in x] == [3, 4]

def test_trivial():
    assert findroot(lambda x: 0, 1) == 1
    assert findroot(lambda x: x, 0) == 0
    #assert findroot(lambda x, y: x + y, (1, -1)) == (1, -1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_data_list.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
import csv
from io import StringIO

import pytest

from pandas import DataFrame
import pandas._testing as tm

from pandas.io.parsers import TextParser

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


@xfail_pyarrow
def test_read_data_list(all_parsers):
    parser = all_parsers
    kwargs = {"index_col": 0}
    data = "A,B,C\nfoo,1,2,3\nbar,4,5,6"

    data_list = [["A", "B", "C"], ["foo", "1", "2", "3"], ["bar", "4", "5", "6"]]
    expected = parser.read_csv(StringIO(data), **kwargs)

    with TextParser(data_list, chunksize=2, **kwargs) as parser:
        result = parser.read()

    tm.assert_frame_equal(result, expected)


def test_reader_list(all_parsers):
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0}

    lines = list(csv.reader(StringIO(data)))
    with TextParser(lines, chunksize=2, **kwargs) as reader:
        chunks = list(reader)

    expected = parser.read_csv(StringIO(data), **kwargs)

    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])


def test_reader_list_skiprows(all_parsers):
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0}

    lines = list(csv.reader(StringIO(data)))
    with TextParser(lines, chunksize=2, skiprows=[1], **kwargs) as reader:
        chunks = list(reader)

    expected = parser.read_csv(StringIO(data), **kwargs)

    tm.assert_frame_equal(chunks[0], expected[1:3])


def test_read_csv_parse_simple_list(all_parsers):
    parser = all_parsers
    data = """foo
bar baz
qux foo
foo
bar"""

    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame(["foo", "bar baz", "qux foo", "foo", "bar"])
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_copy.py ===
import numpy as np
import pytest

from pandas import (
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestCopy:
    @pytest.mark.parametrize("deep", ["default", None, False, True])
    def test_copy(self, deep, using_copy_on_write, warn_copy_on_write):
        ser = Series(np.arange(10), dtype="float64")

        # default deep is True
        if deep == "default":
            ser2 = ser.copy()
        else:
            ser2 = ser.copy(deep=deep)

        if using_copy_on_write:
            # INFO(CoW) a shallow copy doesn't yet copy the data
            # but parent will not be modified (CoW)
            if deep is None or deep is False:
                assert np.may_share_memory(ser.values, ser2.values)
            else:
                assert not np.may_share_memory(ser.values, ser2.values)

        with tm.assert_cow_warning(warn_copy_on_write and deep is False):
            ser2[::2] = np.nan

        if deep is not False or using_copy_on_write:
            # Did not modify original Series
            assert np.isnan(ser2[0])
            assert not np.isnan(ser[0])
        else:
            # we DID modify the original Series
            assert np.isnan(ser2[0])
            assert np.isnan(ser[0])

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    @pytest.mark.parametrize("deep", ["default", None, False, True])
    def test_copy_tzaware(self, deep, using_copy_on_write):
        # GH#11794
        # copy of tz-aware
        expected = Series([Timestamp("2012/01/01", tz="UTC")])
        expected2 = Series([Timestamp("1999/01/01", tz="UTC")])

        ser = Series([Timestamp("2012/01/01", tz="UTC")])

        if deep == "default":
            ser2 = ser.copy()
        else:
            ser2 = ser.copy(deep=deep)

        if using_copy_on_write:
            # INFO(CoW) a shallow copy doesn't yet copy the data
            # but parent will not be modified (CoW)
            if deep is None or deep is False:
                assert np.may_share_memory(ser.values, ser2.values)
            else:
                assert not np.may_share_memory(ser.values, ser2.values)

        ser2[0] = Timestamp("1999/01/01", tz="UTC")

        # default deep is True
        if deep is not False or using_copy_on_write:
            # Did not modify original Series
            tm.assert_series_equal(ser2, expected2)
            tm.assert_series_equal(ser, expected)
        else:
            # we DID modify the original Series
            tm.assert_series_equal(ser2, expected2)
            tm.assert_series_equal(ser, expected2)

    def test_copy_name(self, datetime_series):
        result = datetime_series.copy()
        assert result.name == datetime_series.name

    def test_copy_index_name_checking(self, datetime_series):
        # don't want to be able to modify the index stored elsewhere after
        # making a copy

        datetime_series.index.name = None
        assert datetime_series.index.name is None
        assert datetime_series is datetime_series

        cp = datetime_series.copy()
        cp.index.name = "foo"
        assert datetime_series.index.name is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_cache.py ===
import sys
from sympy.core.cache import cacheit, cached_property, lazy_function
from sympy.testing.pytest import raises

def test_cacheit_doc():
    @cacheit
    def testfn():
        "test docstring"
        pass

    assert testfn.__doc__ == "test docstring"
    assert testfn.__name__ == "testfn"

def test_cacheit_unhashable():
    @cacheit
    def testit(x):
        return x

    assert testit(1) == 1
    assert testit(1) == 1
    a = {}
    assert testit(a) == {}
    a[1] = 2
    assert testit(a) == {1: 2}

def test_cachit_exception():
    # Make sure the cache doesn't call functions multiple times when they
    # raise TypeError

    a = []

    @cacheit
    def testf(x):
        a.append(0)
        raise TypeError

    raises(TypeError, lambda: testf(1))
    assert len(a) == 1

    a.clear()
    # Unhashable type
    raises(TypeError, lambda: testf([]))
    assert len(a) == 1

    @cacheit
    def testf2(x):
        a.append(0)
        raise TypeError("Error")

    a.clear()
    raises(TypeError, lambda: testf2(1))
    assert len(a) == 1

    a.clear()
    # Unhashable type
    raises(TypeError, lambda: testf2([]))
    assert len(a) == 1

def test_cached_property():
    class A:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        @cached_property
        def prop(self):
            self.calls = self.calls + 1
            return self.value

    a = A(2)
    assert a.calls == 0
    assert a.prop == 2
    assert a.calls == 1
    assert a.prop == 2
    assert a.calls == 1
    b = A(None)
    assert b.prop == None


def test_lazy_function():
    module_name='xmlrpc.client'
    function_name = 'gzip_decode'
    lazy = lazy_function(module_name, function_name)
    assert lazy(b'') == b''
    assert module_name in sys.modules
    assert function_name in str(lazy)
    repr_lazy = repr(lazy)
    assert 'LazyFunction' in repr_lazy
    assert function_name in repr_lazy

    lazy = lazy_function('sympy.core.cache', 'cheap')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\diffgeom\tests\test_hyperbolic_space.py ===
r'''
unit test describing the hyperbolic half-plane with the Poincare metric. This
is a basic model of hyperbolic geometry on the (positive) half-space

{(x,y) \in R^2 | y > 0}

with the Riemannian metric

ds^2 = (dx^2 + dy^2)/y^2

It has constant negative scalar curvature = -2

https://en.wikipedia.org/wiki/Poincare_half-plane_model
'''
from sympy.matrices.dense import diag
from sympy.diffgeom import (twoform_to_matrix,
                            metric_to_Christoffel_1st, metric_to_Christoffel_2nd,
                            metric_to_Riemann_components, metric_to_Ricci_components)
import sympy.diffgeom.rn
from sympy.tensor.array import ImmutableDenseNDimArray


def test_H2():
    TP = sympy.diffgeom.TensorProduct
    R2 = sympy.diffgeom.rn.R2
    y = R2.y
    dy = R2.dy
    dx = R2.dx
    g = (TP(dx, dx) + TP(dy, dy))*y**(-2)
    automat = twoform_to_matrix(g)
    mat = diag(y**(-2), y**(-2))
    assert mat == automat

    gamma1 = metric_to_Christoffel_1st(g)
    assert gamma1[0, 0, 0] == 0
    assert gamma1[0, 0, 1] == -y**(-3)
    assert gamma1[0, 1, 0] == -y**(-3)
    assert gamma1[0, 1, 1] == 0

    assert gamma1[1, 1, 1] == -y**(-3)
    assert gamma1[1, 1, 0] == 0
    assert gamma1[1, 0, 1] == 0
    assert gamma1[1, 0, 0] == y**(-3)

    gamma2 = metric_to_Christoffel_2nd(g)
    assert gamma2[0, 0, 0] == 0
    assert gamma2[0, 0, 1] == -y**(-1)
    assert gamma2[0, 1, 0] == -y**(-1)
    assert gamma2[0, 1, 1] == 0

    assert gamma2[1, 1, 1] == -y**(-1)
    assert gamma2[1, 1, 0] == 0
    assert gamma2[1, 0, 1] == 0
    assert gamma2[1, 0, 0] == y**(-1)

    Rm = metric_to_Riemann_components(g)
    assert Rm[0, 0, 0, 0] == 0
    assert Rm[0, 0, 0, 1] == 0
    assert Rm[0, 0, 1, 0] == 0
    assert Rm[0, 0, 1, 1] == 0

    assert Rm[0, 1, 0, 0] == 0
    assert Rm[0, 1, 0, 1] == -y**(-2)
    assert Rm[0, 1, 1, 0] == y**(-2)
    assert Rm[0, 1, 1, 1] == 0

    assert Rm[1, 0, 0, 0] == 0
    assert Rm[1, 0, 0, 1] == y**(-2)
    assert Rm[1, 0, 1, 0] == -y**(-2)
    assert Rm[1, 0, 1, 1] == 0

    assert Rm[1, 1, 0, 0] == 0
    assert Rm[1, 1, 0, 1] == 0
    assert Rm[1, 1, 1, 0] == 0
    assert Rm[1, 1, 1, 1] == 0

    Ric = metric_to_Ricci_components(g)
    assert Ric[0, 0] == -y**(-2)
    assert Ric[0, 1] == 0
    assert Ric[1, 0] == 0
    assert Ric[0, 0] == -y**(-2)

    assert Ric == ImmutableDenseNDimArray([-y**(-2), 0, 0, -y**(-2)], (2, 2))

    ## scalar curvature is -2
    #TODO - it would be nice to have index contraction built-in
    R = (Ric[0, 0] + Ric[1, 1])*y**2
    assert R == -2

    ## Gauss curvature is -1
    assert R/2 == -1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_appellseqs.py ===
"""Tests for efficient functions for generating Appell sequences."""
from sympy.core.numbers import Rational as Q
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.appellseqs import (bernoulli_poly, bernoulli_c_poly,
    euler_poly, genocchi_poly, andre_poly)
from sympy.abc import x

def test_bernoulli_poly():
    raises(ValueError, lambda: bernoulli_poly(-1, x))
    assert bernoulli_poly(1, x, polys=True) == Poly(x - Q(1,2))

    assert bernoulli_poly(0, x) == 1
    assert bernoulli_poly(1, x) == x - Q(1,2)
    assert bernoulli_poly(2, x) == x**2 - x + Q(1,6)
    assert bernoulli_poly(3, x) == x**3 - Q(3,2)*x**2 + Q(1,2)*x
    assert bernoulli_poly(4, x) == x**4 - 2*x**3 + x**2 - Q(1,30)
    assert bernoulli_poly(5, x) == x**5 - Q(5,2)*x**4 + Q(5,3)*x**3 - Q(1,6)*x
    assert bernoulli_poly(6, x) == x**6 - 3*x**5 + Q(5,2)*x**4 - Q(1,2)*x**2 + Q(1,42)

    assert bernoulli_poly(1).dummy_eq(x - Q(1,2))
    assert bernoulli_poly(1, polys=True) == Poly(x - Q(1,2))

def test_bernoulli_c_poly():
    raises(ValueError, lambda: bernoulli_c_poly(-1, x))
    assert bernoulli_c_poly(1, x, polys=True) == Poly(x, domain='QQ')

    assert bernoulli_c_poly(0, x) == 1
    assert bernoulli_c_poly(1, x) == x
    assert bernoulli_c_poly(2, x) == x**2 - Q(1,3)
    assert bernoulli_c_poly(3, x) == x**3 - x
    assert bernoulli_c_poly(4, x) == x**4 - 2*x**2 + Q(7,15)
    assert bernoulli_c_poly(5, x) == x**5 - Q(10,3)*x**3 + Q(7,3)*x
    assert bernoulli_c_poly(6, x) == x**6 - 5*x**4 + 7*x**2 - Q(31,21)

    assert bernoulli_c_poly(1).dummy_eq(x)
    assert bernoulli_c_poly(1, polys=True) == Poly(x, domain='QQ')

    assert 2**8 * bernoulli_poly(8, (x+1)/2).expand() == bernoulli_c_poly(8, x)
    assert 2**9 * bernoulli_poly(9, (x+1)/2).expand() == bernoulli_c_poly(9, x)

def test_genocchi_poly():
    raises(ValueError, lambda: genocchi_poly(-1, x))
    assert genocchi_poly(2, x, polys=True) == Poly(-2*x + 1)

    assert genocchi_poly(0, x) == 0
    assert genocchi_poly(1, x) == -1
    assert genocchi_poly(2, x) == 1 - 2*x
    assert genocchi_poly(3, x) == 3*x - 3*x**2
    assert genocchi_poly(4, x) == -1 + 6*x**2 - 4*x**3
    assert genocchi_poly(5, x) == -5*x + 10*x**3 - 5*x**4
    assert genocchi_poly(6, x) == 3 - 15*x**2 + 15*x**4 - 6*x**5

    assert genocchi_poly(2).dummy_eq(-2*x + 1)
    assert genocchi_poly(2, polys=True) == Poly(-2*x + 1)

    assert 2 * (bernoulli_poly(8, x) - bernoulli_c_poly(8, x)) == genocchi_poly(8, x)
    assert 2 * (bernoulli_poly(9, x) - bernoulli_c_poly(9, x)) == genocchi_poly(9, x)

def test_euler_poly():
    raises(ValueError, lambda: euler_poly(-1, x))
    assert euler_poly(1, x, polys=True) == Poly(x - Q(1,2))

    assert euler_poly(0, x) == 1
    assert euler_poly(1, x) == x - Q(1,2)
    assert euler_poly(2, x) == x**2 - x
    assert euler_poly(3, x) == x**3 - Q(3,2)*x**2 + Q(1,4)
    assert euler_poly(4, x) == x**4 - 2*x**3 + x
    assert euler_poly(5, x) == x**5 - Q(5,2)*x**4 + Q(5,2)*x**2 - Q(1,2)
    assert euler_poly(6, x) == x**6 - 3*x**5 + 5*x**3 - 3*x

    assert euler_poly(1).dummy_eq(x - Q(1,2))
    assert euler_poly(1, polys=True) == Poly(x - Q(1,2))

    assert genocchi_poly(9, x) == euler_poly(8, x) * -9
    assert genocchi_poly(10, x) == euler_poly(9, x) * -10

def test_andre_poly():
    raises(ValueError, lambda: andre_poly(-1, x))
    assert andre_poly(1, x, polys=True) == Poly(x)

    assert andre_poly(0, x) == 1
    assert andre_poly(1, x) == x
    assert andre_poly(2, x) == x**2 - 1
    assert andre_poly(3, x) == x**3 - 3*x
    assert andre_poly(4, x) == x**4 - 6*x**2 + 5
    assert andre_poly(5, x) == x**5 - 10*x**3 + 25*x
    assert andre_poly(6, x) == x**6 - 15*x**4 + 75*x**2 - 61

    assert andre_poly(1).dummy_eq(x)
    assert andre_poly(1, polys=True) == Poly(x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test__exceptions.py ===
"""
Tests of the ._exceptions module. Primarily for exercising the __str__ methods.
"""

import pickle

import pytest

import numpy as np
from numpy.exceptions import AxisError

_ArrayMemoryError = np._core._exceptions._ArrayMemoryError
_UFuncNoLoopError = np._core._exceptions._UFuncNoLoopError

class TestArrayMemoryError:
    def test_pickling(self):
        """ Test that _ArrayMemoryError can be pickled """
        error = _ArrayMemoryError((1023,), np.dtype(np.uint8))
        res = pickle.loads(pickle.dumps(error))
        assert res._total_size == error._total_size

    def test_str(self):
        e = _ArrayMemoryError((1023,), np.dtype(np.uint8))
        str(e)  # not crashing is enough

    # testing these properties is easier than testing the full string repr
    def test__size_to_string(self):
        """ Test e._size_to_string """
        f = _ArrayMemoryError._size_to_string
        Ki = 1024
        assert f(0) == '0 bytes'
        assert f(1) == '1 bytes'
        assert f(1023) == '1023 bytes'
        assert f(Ki) == '1.00 KiB'
        assert f(Ki + 1) == '1.00 KiB'
        assert f(10 * Ki) == '10.0 KiB'
        assert f(int(999.4 * Ki)) == '999. KiB'
        assert f(int(1023.4 * Ki)) == '1023. KiB'
        assert f(int(1023.5 * Ki)) == '1.00 MiB'
        assert f(Ki * Ki) == '1.00 MiB'

        # 1023.9999 Mib should round to 1 GiB
        assert f(int(Ki * Ki * Ki * 0.9999)) == '1.00 GiB'
        assert f(Ki * Ki * Ki * Ki * Ki * Ki) == '1.00 EiB'
        # larger than sys.maxsize, adding larger prefixes isn't going to help
        # anyway.
        assert f(Ki * Ki * Ki * Ki * Ki * Ki * 123456) == '123456. EiB'

    def test__total_size(self):
        """ Test e._total_size """
        e = _ArrayMemoryError((1,), np.dtype(np.uint8))
        assert e._total_size == 1

        e = _ArrayMemoryError((2, 4), np.dtype((np.uint64, 16)))
        assert e._total_size == 1024


class TestUFuncNoLoopError:
    def test_pickling(self):
        """ Test that _UFuncNoLoopError can be pickled """
        assert isinstance(pickle.dumps(_UFuncNoLoopError), bytes)


@pytest.mark.parametrize("args", [
    (2, 1, None),
    (2, 1, "test_prefix"),
    ("test message",),
])
class TestAxisError:
    def test_attr(self, args):
        """Validate attribute types."""
        exc = AxisError(*args)
        if len(args) == 1:
            assert exc.axis is None
            assert exc.ndim is None
        else:
            axis, ndim, *_ = args
            assert exc.axis == axis
            assert exc.ndim == ndim

    def test_pickling(self, args):
        """Test that `AxisError` can be pickled."""
        exc = AxisError(*args)
        exc2 = pickle.loads(pickle.dumps(exc))

        assert type(exc) is type(exc2)
        for name in ("axis", "ndim", "args"):
            attr1 = getattr(exc, name)
            attr2 = getattr(exc2, name)
            assert attr1 == attr2, name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_astype.py ===
from datetime import date

import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    IntervalIndex,
)
import pandas._testing as tm


class TestAstype:
    def test_astype(self):
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)

        result = ci.astype(object)
        tm.assert_index_equal(result, Index(np.array(ci), dtype=object))

        # this IS equal, but not the same class
        assert result.equals(ci)
        assert isinstance(result, Index)
        assert not isinstance(result, CategoricalIndex)

        # interval
        ii = IntervalIndex.from_arrays(left=[-0.001, 2.0], right=[2, 4], closed="right")

        ci = CategoricalIndex(
            Categorical.from_codes([0, 1, -1], categories=ii, ordered=True)
        )

        result = ci.astype("interval")
        expected = ii.take([0, 1, -1], allow_fill=True, fill_value=np.nan)
        tm.assert_index_equal(result, expected)

        result = IntervalIndex(result.values)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("dtype_ordered", [True, False])
    @pytest.mark.parametrize("index_ordered", [True, False])
    def test_astype_category(self, name, dtype_ordered, index_ordered):
        # GH#18630
        index = CategoricalIndex(
            list("aabbca"), categories=list("cab"), ordered=index_ordered
        )
        if name:
            index = index.rename(name)

        # standard categories
        dtype = CategoricalDtype(ordered=dtype_ordered)
        result = index.astype(dtype)
        expected = CategoricalIndex(
            index.tolist(),
            name=name,
            categories=index.categories,
            ordered=dtype_ordered,
        )
        tm.assert_index_equal(result, expected)

        # non-standard categories
        dtype = CategoricalDtype(index.unique().tolist()[:-1], dtype_ordered)
        result = index.astype(dtype)
        expected = CategoricalIndex(index.tolist(), name=name, dtype=dtype)
        tm.assert_index_equal(result, expected)

        if dtype_ordered is False:
            # dtype='category' can't specify ordered, so only test once
            result = index.astype("category")
            expected = index
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("box", [True, False])
    def test_categorical_date_roundtrip(self, box):
        # astype to categorical and back should preserve date objects
        v = date.today()

        obj = Index([v, v])
        assert obj.dtype == object
        if box:
            obj = obj.array

        cat = obj.astype("category")

        rtrip = cat.astype(object)
        assert rtrip.dtype == object
        assert type(rtrip[0]) is date

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_ipython_compat.py ===
import numpy as np

import pandas._config.config as cf

from pandas import (
    DataFrame,
    MultiIndex,
)


class TestTableSchemaRepr:
    def test_publishes(self, ip):
        ipython = ip.instance(config=ip.config)
        df = DataFrame({"A": [1, 2]})
        objects = [df["A"], df]  # dataframe / series
        expected_keys = [
            {"text/plain", "application/vnd.dataresource+json"},
            {"text/plain", "text/html", "application/vnd.dataresource+json"},
        ]

        opt = cf.option_context("display.html.table_schema", True)
        last_obj = None
        for obj, expected in zip(objects, expected_keys):
            last_obj = obj
            with opt:
                formatted = ipython.display_formatter.format(obj)
            assert set(formatted[0].keys()) == expected

        with_latex = cf.option_context("styler.render.repr", "latex")

        with opt, with_latex:
            formatted = ipython.display_formatter.format(last_obj)

        expected = {
            "text/plain",
            "text/html",
            "text/latex",
            "application/vnd.dataresource+json",
        }
        assert set(formatted[0].keys()) == expected

    def test_publishes_not_implemented(self, ip):
        # column MultiIndex
        # GH#15996
        midx = MultiIndex.from_product([["A", "B"], ["a", "b", "c"]])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, len(midx))), columns=midx
        )

        opt = cf.option_context("display.html.table_schema", True)

        with opt:
            formatted = ip.instance(config=ip.config).display_formatter.format(df)

        expected = {"text/plain", "text/html"}
        assert set(formatted[0].keys()) == expected

    def test_config_on(self):
        df = DataFrame({"A": [1, 2]})
        with cf.option_context("display.html.table_schema", True):
            result = df._repr_data_resource_()

        assert result is not None

    def test_config_default_off(self):
        df = DataFrame({"A": [1, 2]})
        with cf.option_context("display.html.table_schema", False):
            result = df._repr_data_resource_()

        assert result is None

    def test_enable_data_resource_formatter(self, ip):
        # GH#10491
        formatters = ip.instance(config=ip.config).display_formatter.formatters
        mimetype = "application/vnd.dataresource+json"

        with cf.option_context("display.html.table_schema", True):
            assert "application/vnd.dataresource+json" in formatters
            assert formatters[mimetype].enabled

        # still there, just disabled
        assert "application/vnd.dataresource+json" in formatters
        assert not formatters[mimetype].enabled

        # able to re-set
        with cf.option_context("display.html.table_schema", True):
            assert "application/vnd.dataresource+json" in formatters
            assert formatters[mimetype].enabled
            # smoke test that it works
            ip.instance(config=ip.config).display_formatter.format(cf)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_categorical_equal.py ===
import pytest

from pandas import Categorical
import pandas._testing as tm


@pytest.mark.parametrize(
    "c",
    [Categorical([1, 2, 3, 4]), Categorical([1, 2, 3, 4], categories=[1, 2, 3, 4, 5])],
)
def test_categorical_equal(c):
    tm.assert_categorical_equal(c, c)


@pytest.mark.parametrize("check_category_order", [True, False])
def test_categorical_equal_order_mismatch(check_category_order):
    c1 = Categorical([1, 2, 3, 4], categories=[1, 2, 3, 4])
    c2 = Categorical([1, 2, 3, 4], categories=[4, 3, 2, 1])
    kwargs = {"check_category_order": check_category_order}

    if check_category_order:
        msg = """Categorical\\.categories are different

Categorical\\.categories values are different \\(100\\.0 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[4, 3, 2, 1\\], dtype='int64'\\)"""
        with pytest.raises(AssertionError, match=msg):
            tm.assert_categorical_equal(c1, c2, **kwargs)
    else:
        tm.assert_categorical_equal(c1, c2, **kwargs)


def test_categorical_equal_categories_mismatch():
    msg = """Categorical\\.categories are different

Categorical\\.categories values are different \\(25\\.0 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 3, 5\\], dtype='int64'\\)"""

    c1 = Categorical([1, 2, 3, 4])
    c2 = Categorical([1, 2, 3, 5])

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)


def test_categorical_equal_codes_mismatch():
    categories = [1, 2, 3, 4]
    msg = """Categorical\\.codes are different

Categorical\\.codes values are different \\(50\\.0 %\\)
\\[left\\]:  \\[0, 1, 3, 2\\]
\\[right\\]: \\[0, 1, 2, 3\\]"""

    c1 = Categorical([1, 2, 4, 3], categories=categories)
    c2 = Categorical([1, 2, 3, 4], categories=categories)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)


def test_categorical_equal_ordered_mismatch():
    data = [1, 2, 3, 4]
    msg = """Categorical are different

Attribute "ordered" are different
\\[left\\]:  False
\\[right\\]: True"""

    c1 = Categorical(data, ordered=False)
    c2 = Categorical(data, ordered=True)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)


@pytest.mark.parametrize("obj", ["index", "foo", "pandas"])
def test_categorical_equal_object_override(obj):
    data = [1, 2, 3, 4]
    msg = f"""{obj} are different

Attribute "ordered" are different
\\[left\\]:  False
\\[right\\]: True"""

    c1 = Categorical(data, ordered=False)
    c2 = Categorical(data, ordered=True)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2, obj=obj)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_deprecate_kwarg.py ===
import pytest

from pandas.util._decorators import deprecate_kwarg

import pandas._testing as tm


@deprecate_kwarg("old", "new")
def _f1(new=False):
    return new


_f2_mappings = {"yes": True, "no": False}


@deprecate_kwarg("old", "new", _f2_mappings)
def _f2(new=False):
    return new


def _f3_mapping(x):
    return x + 1


@deprecate_kwarg("old", "new", _f3_mapping)
def _f3(new=0):
    return new


@pytest.mark.parametrize("key,klass", [("old", FutureWarning), ("new", None)])
def test_deprecate_kwarg(key, klass):
    x = 78

    with tm.assert_produces_warning(klass):
        assert _f1(**{key: x}) == x


@pytest.mark.parametrize("key", list(_f2_mappings.keys()))
def test_dict_deprecate_kwarg(key):
    with tm.assert_produces_warning(FutureWarning):
        assert _f2(old=key) == _f2_mappings[key]


@pytest.mark.parametrize("key", ["bogus", 12345, -1.23])
def test_missing_deprecate_kwarg(key):
    with tm.assert_produces_warning(FutureWarning):
        assert _f2(old=key) == key


@pytest.mark.parametrize("x", [1, -1.4, 0])
def test_callable_deprecate_kwarg(x):
    with tm.assert_produces_warning(FutureWarning):
        assert _f3(old=x) == _f3_mapping(x)


def test_callable_deprecate_kwarg_fail():
    msg = "((can only|cannot) concatenate)|(must be str)|(Can't convert)"

    with pytest.raises(TypeError, match=msg):
        _f3(old="hello")


def test_bad_deprecate_kwarg():
    msg = "mapping from old to new argument values must be dict or callable!"

    with pytest.raises(TypeError, match=msg):

        @deprecate_kwarg("old", "new", 0)
        def f4(new=None):
            return new


@deprecate_kwarg("old", None)
def _f4(old=True, unchanged=True):
    return old, unchanged


@pytest.mark.parametrize("key", ["old", "unchanged"])
def test_deprecate_keyword(key):
    x = 9

    if key == "old":
        klass = FutureWarning
        expected = (x, True)
    else:
        klass = None
        expected = (True, x)

    with tm.assert_produces_warning(klass):
        assert _f4(**{key: x}) == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_doc.py ===
from textwrap import dedent

from pandas.util._decorators import doc


@doc(method="cumsum", operation="sum")
def cumsum(whatever):
    """
    This is the {method} method.

    It computes the cumulative {operation}.
    """


@doc(
    cumsum,
    dedent(
        """
        Examples
        --------

        >>> cumavg([1, 2, 3])
        2
        """
    ),
    method="cumavg",
    operation="average",
)
def cumavg(whatever):
    pass


@doc(cumsum, method="cummax", operation="maximum")
def cummax(whatever):
    pass


@doc(cummax, method="cummin", operation="minimum")
def cummin(whatever):
    pass


def test_docstring_formatting():
    docstr = dedent(
        """
        This is the cumsum method.

        It computes the cumulative sum.
        """
    )
    assert cumsum.__doc__ == docstr


def test_docstring_appending():
    docstr = dedent(
        """
        This is the cumavg method.

        It computes the cumulative average.

        Examples
        --------

        >>> cumavg([1, 2, 3])
        2
        """
    )
    assert cumavg.__doc__ == docstr


def test_doc_template_from_func():
    docstr = dedent(
        """
        This is the cummax method.

        It computes the cumulative maximum.
        """
    )
    assert cummax.__doc__ == docstr


def test_inherit_doc_template():
    docstr = dedent(
        """
        This is the cummin method.

        It computes the cumulative minimum.
        """
    )
    assert cummin.__doc__ == docstr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_eigen.py ===
"""
Tests for the sympy.polys.matrices.eigen module
"""

from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix

from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.polys.matrices.domainmatrix import DomainMatrix

from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy


def test_dom_eigenvects_rational():
    # Rational eigenvalues
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(1), QQ(2)]], (2, 2), QQ)
    rational_eigenvects = [
        (QQ, QQ(3), 1, DomainMatrix([[QQ(1), QQ(1)]], (1, 2), QQ)),
        (QQ, QQ(0), 1, DomainMatrix([[QQ(-2), QQ(1)]], (1, 2), QQ)),
    ]
    assert dom_eigenvects(A) == (rational_eigenvects, [])

    # Test converting to Expr:
    sympy_eigenvects = [
        (S(3), 1, [Matrix([1, 1])]),
        (S(0), 1, [Matrix([-2, 1])]),
    ]
    assert dom_eigenvects_to_sympy(rational_eigenvects, [], Matrix) == sympy_eigenvects


def test_dom_eigenvects_algebraic():
    # Algebraic eigenvalues
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Avects = dom_eigenvects(A)

    # Extract the dummy to build the expected result:
    lamda = Avects[1][0][1].gens[0]
    irreducible = Poly(lamda**2 - 5*lamda - 2, lamda, domain=QQ)
    K = FiniteExtension(irreducible)
    KK = K.from_sympy
    algebraic_eigenvects = [
        (K, irreducible, 1, DomainMatrix([[KK((lamda-4)/3), KK(1)]], (1, 2), K)),
    ]
    assert Avects == ([], algebraic_eigenvects)

    # Test converting to Expr:
    sympy_eigenvects = [
        (S(5)/2 - sqrt(33)/2, 1, [Matrix([[-sqrt(33)/6 - S(1)/2], [1]])]),
        (S(5)/2 + sqrt(33)/2, 1, [Matrix([[-S(1)/2 + sqrt(33)/6], [1]])]),
    ]
    assert dom_eigenvects_to_sympy([], algebraic_eigenvects, Matrix) == sympy_eigenvects


def test_dom_eigenvects_rootof():
    # Algebraic eigenvalues
    A = DomainMatrix([
        [0, 0, 0, 0, -1],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]], (5, 5), QQ)
    Avects = dom_eigenvects(A)

    # Extract the dummy to build the expected result:
    lamda = Avects[1][0][1].gens[0]
    irreducible = Poly(lamda**5 - lamda + 1, lamda, domain=QQ)
    K = FiniteExtension(irreducible)
    KK = K.from_sympy
    algebraic_eigenvects = [
        (K, irreducible, 1,
            DomainMatrix([
                [KK(lamda**4-1), KK(lamda**3), KK(lamda**2), KK(lamda), KK(1)]
            ], (1, 5), K)),
    ]
    assert Avects == ([], algebraic_eigenvects)

    # Test converting to Expr (slow):
    l0, l1, l2, l3, l4 = [CRootOf(lamda**5 - lamda + 1, i) for i in range(5)]
    sympy_eigenvects = [
        (l0, 1, [Matrix([-1 + l0**4, l0**3, l0**2, l0, 1])]),
        (l1, 1, [Matrix([-1 + l1**4, l1**3, l1**2, l1, 1])]),
        (l2, 1, [Matrix([-1 + l2**4, l2**3, l2**2, l2, 1])]),
        (l3, 1, [Matrix([-1 + l3**4, l3**3, l3**2, l3, 1])]),
        (l4, 1, [Matrix([-1 + l4**4, l4**3, l4**2, l4, 1])]),
    ]
    assert dom_eigenvects_to_sympy([], algebraic_eigenvects, Matrix) == sympy_eigenvects

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_epathtools.py ===
"""Tests for tools for manipulation of expressions using paths. """

from sympy.simplify.epathtools import epath, EPath
from sympy.testing.pytest import raises

from sympy.core.numbers import E
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.abc import x, y, z, t


def test_epath_select():
    expr = [((x, 1, t), 2), ((3, y, 4), z)]

    assert epath("/*", expr) == [((x, 1, t), 2), ((3, y, 4), z)]
    assert epath("/*/*", expr) == [(x, 1, t), 2, (3, y, 4), z]
    assert epath("/*/*/*", expr) == [x, 1, t, 3, y, 4]
    assert epath("/*/*/*/*", expr) == []

    assert epath("/[:]", expr) == [((x, 1, t), 2), ((3, y, 4), z)]
    assert epath("/[:]/[:]", expr) == [(x, 1, t), 2, (3, y, 4), z]
    assert epath("/[:]/[:]/[:]", expr) == [x, 1, t, 3, y, 4]
    assert epath("/[:]/[:]/[:]/[:]", expr) == []

    assert epath("/*/[:]", expr) == [(x, 1, t), 2, (3, y, 4), z]

    assert epath("/*/[0]", expr) == [(x, 1, t), (3, y, 4)]
    assert epath("/*/[1]", expr) == [2, z]
    assert epath("/*/[2]", expr) == []

    assert epath("/*/int", expr) == [2]
    assert epath("/*/Symbol", expr) == [z]
    assert epath("/*/tuple", expr) == [(x, 1, t), (3, y, 4)]
    assert epath("/*/__iter__?", expr) == [(x, 1, t), (3, y, 4)]

    assert epath("/*/int|tuple", expr) == [(x, 1, t), 2, (3, y, 4)]
    assert epath("/*/Symbol|tuple", expr) == [(x, 1, t), (3, y, 4), z]
    assert epath("/*/int|Symbol|tuple", expr) == [(x, 1, t), 2, (3, y, 4), z]

    assert epath("/*/int|__iter__?", expr) == [(x, 1, t), 2, (3, y, 4)]
    assert epath("/*/Symbol|__iter__?", expr) == [(x, 1, t), (3, y, 4), z]
    assert epath(
        "/*/int|Symbol|__iter__?", expr) == [(x, 1, t), 2, (3, y, 4), z]

    assert epath("/*/[0]/int", expr) == [1, 3, 4]
    assert epath("/*/[0]/Symbol", expr) == [x, t, y]

    assert epath("/*/[0]/int[1:]", expr) == [1, 4]
    assert epath("/*/[0]/Symbol[1:]", expr) == [t, y]

    assert epath("/Symbol", x + y + z + 1) == [x, y, z]
    assert epath("/*/*/Symbol", t + sin(x + 1) + cos(x + y + E)) == [x, x, y]


def test_epath_apply():
    expr = [((x, 1, t), 2), ((3, y, 4), z)]
    func = lambda expr: expr**2

    assert epath("/*", expr, list) == [[(x, 1, t), 2], [(3, y, 4), z]]

    assert epath("/*/[0]", expr, list) == [([x, 1, t], 2), ([3, y, 4], z)]
    assert epath("/*/[1]", expr, func) == [((x, 1, t), 4), ((3, y, 4), z**2)]
    assert epath("/*/[2]", expr, list) == expr

    assert epath("/*/[0]/int", expr, func) == [((x, 1, t), 2), ((9, y, 16), z)]
    assert epath("/*/[0]/Symbol", expr, func) == [((x**2, 1, t**2), 2),
                 ((3, y**2, 4), z)]
    assert epath(
        "/*/[0]/int[1:]", expr, func) == [((x, 1, t), 2), ((3, y, 16), z)]
    assert epath("/*/[0]/Symbol[1:]", expr, func) == [((x, 1, t**2),
                 2), ((3, y**2, 4), z)]

    assert epath("/Symbol", x + y + z + 1, func) == x**2 + y**2 + z**2 + 1
    assert epath("/*/*/Symbol", t + sin(x + 1) + cos(x + y + E), func) == \
        t + sin(x**2 + 1) + cos(x**2 + y**2 + E)


def test_EPath():
    assert EPath("/*/[0]")._path == "/*/[0]"
    assert EPath(EPath("/*/[0]"))._path == "/*/[0]"
    assert isinstance(epath("/*/[0]"), EPath) is True

    assert repr(EPath("/*/[0]")) == "EPath('/*/[0]')"

    raises(ValueError, lambda: EPath(""))
    raises(ValueError, lambda: EPath("/"))
    raises(ValueError, lambda: EPath("/|x"))
    raises(ValueError, lambda: EPath("/["))
    raises(ValueError, lambda: EPath("/[0]%"))

    raises(NotImplementedError, lambda: EPath("Symbol"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_implicitregion.py ===
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.abc import x, y, z, s, t
from sympy.sets import FiniteSet, EmptySet
from sympy.geometry import Point
from sympy.vector import ImplicitRegion
from sympy.testing.pytest import raises


def test_ImplicitRegion():
    ellipse = ImplicitRegion((x, y), (x**2/4 + y**2/16 - 1))
    assert ellipse.equation == x**2/4 + y**2/16 - 1
    assert ellipse.variables == (x, y)
    assert ellipse.degree == 2
    r = ImplicitRegion((x, y, z), Eq(x**4 + y**2 - x*y, 6))
    assert r.equation == x**4 + y**2 - x*y - 6
    assert r.variables == (x, y, z)
    assert r.degree == 4


def test_regular_point():
    r1 = ImplicitRegion((x,), x**2 - 16)
    assert r1.regular_point() == (-4,)
    c1 = ImplicitRegion((x, y), x**2 + y**2 - 4)
    assert c1.regular_point() == (0, -2)
    c2 = ImplicitRegion((x, y), (x - S(5)/2)**2 + y**2 - (S(1)/4)**2)
    assert c2.regular_point() == (S(5)/2, -S(1)/4)
    c3 = ImplicitRegion((x, y), (y - 5)**2  - 16*(x - 5))
    assert c3.regular_point() == (5, 5)
    r2 = ImplicitRegion((x, y), x**2 - 4*x*y - 3*y**2 + 4*x + 8*y - 5)
    assert r2.regular_point() == (S(4)/7, S(9)/7)
    r3 = ImplicitRegion((x, y), x**2 - 2*x*y + 3*y**2 - 2*x - 5*y + 3/2)
    raises(ValueError, lambda: r3.regular_point())


def test_singular_points_and_multiplicty():
    r1 = ImplicitRegion((x, y, z), Eq(x + y + z, 0))
    assert r1.singular_points() == EmptySet
    r2 = ImplicitRegion((x, y, z), x*y*z + y**4 -x**2*z**2)
    assert r2.singular_points() == FiniteSet((0, 0, z), (x, 0, 0))
    assert r2.multiplicity((0, 0, 0)) == 3
    assert r2.multiplicity((0, 0, 6)) == 2
    r3 = ImplicitRegion((x, y, z), z**2 - x**2 - y**2)
    assert r3.singular_points() == FiniteSet((0, 0, 0))
    assert r3.multiplicity((0, 0, 0)) == 2
    r4 = ImplicitRegion((x, y), x**2 + y**2 - 2*x)
    assert r4.singular_points() == EmptySet
    assert r4.multiplicity(Point(1, 3)) == 0


def test_rational_parametrization():
    p = ImplicitRegion((x,), x - 2)
    assert p.rational_parametrization() == (x - 2,)

    line = ImplicitRegion((x, y), Eq(y, 3*x + 2))
    assert line.rational_parametrization() == (x, 3*x + 2)

    circle1 = ImplicitRegion((x, y), (x-2)**2 + (y+3)**2 - 4)
    assert circle1.rational_parametrization(parameters=t) == (4*t/(t**2 + 1) + 2, 4*t**2/(t**2 + 1) - 5)
    circle2 = ImplicitRegion((x, y), (x - S.Half)**2 + y**2 - (S(1)/2)**2)

    assert circle2.rational_parametrization(parameters=t) == (t/(t**2 + 1) + S(1)/2, t**2/(t**2 + 1) - S(1)/2)
    circle3 = ImplicitRegion((x, y), Eq(x**2 + y**2, 2*x))
    assert circle3.rational_parametrization(parameters=(t,)) == (2*t/(t**2 + 1) + 1, 2*t**2/(t**2 + 1) - 1)

    parabola = ImplicitRegion((x, y), (y - 3)**2 - 4*(x + 6))
    assert parabola.rational_parametrization(t) == (-6 + 4/t**2, 3 + 4/t)

    rect_hyperbola = ImplicitRegion((x, y), x*y - 1)
    assert rect_hyperbola.rational_parametrization(t) == (-1 + (t + 1)/t, t)

    cubic_curve = ImplicitRegion((x, y), x**3 + x**2 - y**2)
    assert cubic_curve.rational_parametrization(parameters=(t)) == (t**2 - 1, t*(t**2 - 1))
    cuspidal = ImplicitRegion((x, y), (x**3 - y**2))
    assert cuspidal.rational_parametrization(t) == (t**2, t**3)

    I = ImplicitRegion((x, y), x**3 + x**2 - y**2)
    assert I.rational_parametrization(t) == (t**2 - 1, t*(t**2 - 1))

    sphere = ImplicitRegion((x, y, z), Eq(x**2 + y**2 + z**2, 2*x))
    assert sphere.rational_parametrization(parameters=(s, t)) == (2/(s**2 + t**2 + 1), 2*t/(s**2 + t**2 + 1), 2*s/(s**2 + t**2 + 1))

    conic = ImplicitRegion((x, y), Eq(x**2 + 4*x*y + 3*y**2 + x - y + 10, 0))
    assert conic.rational_parametrization(t) == (
        S(17)/2 + 4/(3*t**2 + 4*t + 1), 4*t/(3*t**2 + 4*t + 1) - S(11)/2)

    r1 = ImplicitRegion((x, y), y**2 - x**3 + x)
    raises(NotImplementedError, lambda: r1.rational_parametrization())
    r2 = ImplicitRegion((x, y), y**2 - x**3 - x**2 + 1)
    raises(NotImplementedError, lambda: r2.rational_parametrization())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_algos.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("categories", [["b", "a", "c"], ["a", "b", "c", "d"]])
def test_factorize(categories, ordered):
    cat = pd.Categorical(
        ["b", "b", "a", "c", None], categories=categories, ordered=ordered
    )
    codes, uniques = pd.factorize(cat)
    expected_codes = np.array([0, 0, 1, 2, -1], dtype=np.intp)
    expected_uniques = pd.Categorical(
        ["b", "a", "c"], categories=categories, ordered=ordered
    )

    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_categorical_equal(uniques, expected_uniques)


def test_factorized_sort():
    cat = pd.Categorical(["b", "b", None, "a"])
    codes, uniques = pd.factorize(cat, sort=True)
    expected_codes = np.array([1, 1, -1, 0], dtype=np.intp)
    expected_uniques = pd.Categorical(["a", "b"])

    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_categorical_equal(uniques, expected_uniques)


def test_factorized_sort_ordered():
    cat = pd.Categorical(
        ["b", "b", None, "a"], categories=["c", "b", "a"], ordered=True
    )

    codes, uniques = pd.factorize(cat, sort=True)
    expected_codes = np.array([0, 0, -1, 1], dtype=np.intp)
    expected_uniques = pd.Categorical(
        ["b", "a"], categories=["c", "b", "a"], ordered=True
    )

    tm.assert_numpy_array_equal(codes, expected_codes)
    tm.assert_categorical_equal(uniques, expected_uniques)


def test_isin_cats():
    # GH2003
    cat = pd.Categorical(["a", "b", np.nan])

    result = cat.isin(["a", np.nan])
    expected = np.array([True, False, True], dtype=bool)
    tm.assert_numpy_array_equal(expected, result)

    result = cat.isin(["a", "c"])
    expected = np.array([True, False, False], dtype=bool)
    tm.assert_numpy_array_equal(expected, result)


@pytest.mark.parametrize("value", [[""], [None, ""], [pd.NaT, ""]])
def test_isin_cats_corner_cases(value):
    # GH36550
    cat = pd.Categorical([""])
    result = cat.isin(value)
    expected = np.array([True], dtype=bool)
    tm.assert_numpy_array_equal(expected, result)


@pytest.mark.parametrize("empty", [[], pd.Series(dtype=object), np.array([])])
def test_isin_empty(empty):
    s = pd.Categorical(["a", "b"])
    expected = np.array([False, False], dtype=bool)

    result = s.isin(empty)
    tm.assert_numpy_array_equal(expected, result)


def test_diff():
    ser = pd.Series([1, 2, 3], dtype="category")

    msg = "Convert to a suitable dtype"
    with pytest.raises(TypeError, match=msg):
        ser.diff()

    df = ser.to_frame(name="A")
    with pytest.raises(TypeError, match=msg):
        df.diff()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_take.py ===
import numpy as np
import pytest

from pandas import Categorical
import pandas._testing as tm


@pytest.fixture(params=[True, False])
def allow_fill(request):
    """Boolean 'allow_fill' parameter for Categorical.take"""
    return request.param


class TestTake:
    # https://github.com/pandas-dev/pandas/issues/20664

    def test_take_default_allow_fill(self):
        cat = Categorical(["a", "b"])
        with tm.assert_produces_warning(None):
            result = cat.take([0, -1])

        assert result.equals(cat)

    def test_take_positive_no_warning(self):
        cat = Categorical(["a", "b"])
        with tm.assert_produces_warning(None):
            cat.take([0, 0])

    def test_take_bounds(self, allow_fill):
        # https://github.com/pandas-dev/pandas/issues/20664
        cat = Categorical(["a", "b", "a"])
        if allow_fill:
            msg = "indices are out-of-bounds"
        else:
            msg = "index 4 is out of bounds for( axis 0 with)? size 3"
        with pytest.raises(IndexError, match=msg):
            cat.take([4, 5], allow_fill=allow_fill)

    def test_take_empty(self, allow_fill):
        # https://github.com/pandas-dev/pandas/issues/20664
        cat = Categorical([], categories=["a", "b"])
        if allow_fill:
            msg = "indices are out-of-bounds"
        else:
            msg = "cannot do a non-empty take from an empty axes"
        with pytest.raises(IndexError, match=msg):
            cat.take([0], allow_fill=allow_fill)

    def test_positional_take(self, ordered):
        cat = Categorical(["a", "a", "b", "b"], categories=["b", "a"], ordered=ordered)
        result = cat.take([0, 1, 2], allow_fill=False)
        expected = Categorical(
            ["a", "a", "b"], categories=cat.categories, ordered=ordered
        )
        tm.assert_categorical_equal(result, expected)

    def test_positional_take_unobserved(self, ordered):
        cat = Categorical(["a", "b"], categories=["a", "b", "c"], ordered=ordered)
        result = cat.take([1, 0], allow_fill=False)
        expected = Categorical(["b", "a"], categories=cat.categories, ordered=ordered)
        tm.assert_categorical_equal(result, expected)

    def test_take_allow_fill(self):
        # https://github.com/pandas-dev/pandas/issues/23296
        cat = Categorical(["a", "a", "b"])
        result = cat.take([0, -1, -1], allow_fill=True)
        expected = Categorical(["a", np.nan, np.nan], categories=["a", "b"])
        tm.assert_categorical_equal(result, expected)

    def test_take_fill_with_negative_one(self):
        # -1 was a category
        cat = Categorical([-1, 0, 1])
        result = cat.take([0, -1, 1], allow_fill=True, fill_value=-1)
        expected = Categorical([-1, -1, 0], categories=[-1, 0, 1])
        tm.assert_categorical_equal(result, expected)

    def test_take_fill_value(self):
        # https://github.com/pandas-dev/pandas/issues/23296
        cat = Categorical(["a", "b", "c"])
        result = cat.take([0, 1, -1], fill_value="a", allow_fill=True)
        expected = Categorical(["a", "b", "a"], categories=["a", "b", "c"])
        tm.assert_categorical_equal(result, expected)

    def test_take_fill_value_new_raises(self):
        # https://github.com/pandas-dev/pandas/issues/23296
        cat = Categorical(["a", "b", "c"])
        xpr = r"Cannot setitem on a Categorical with a new category \(d\)"
        with pytest.raises(TypeError, match=xpr):
            cat.take([0, 1, -1], fill_value="d", allow_fill=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\array_with_attr\array.py ===
"""
Test extension array that has custom attribute information (not stored on the dtype).

"""
from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.base import ExtensionDtype

import pandas as pd
from pandas.core.arrays import ExtensionArray

if TYPE_CHECKING:
    from pandas._typing import type_t


class FloatAttrDtype(ExtensionDtype):
    type = float
    name = "float_attr"
    na_value = np.nan

    @classmethod
    def construct_array_type(cls) -> type_t[FloatAttrArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return FloatAttrArray


class FloatAttrArray(ExtensionArray):
    dtype = FloatAttrDtype()
    __array_priority__ = 1000

    def __init__(self, values, attr=None) -> None:
        if not isinstance(values, np.ndarray):
            raise TypeError("Need to pass a numpy array of float64 dtype as values")
        if not values.dtype == "float64":
            raise TypeError("Need to pass a numpy array of float64 dtype as values")
        self.data = values
        self.attr = attr

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if not copy:
            data = np.asarray(scalars, dtype="float64")
        else:
            data = np.array(scalars, dtype="float64", copy=copy)
        return cls(data)

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]
        else:
            # slice, list-like, mask
            item = pd.api.indexers.check_array_indexer(self, item)
            return type(self)(self.data[item], self.attr)

    def __len__(self) -> int:
        return len(self.data)

    def isna(self):
        return np.isnan(self.data)

    def take(self, indexer, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        data = self.data
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        return type(self)(result, self.attr)

    def copy(self):
        return type(self)(self.data.copy(), self.attr)

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = np.concatenate([x.data for x in to_concat])
        attr = to_concat[0].attr if len(to_concat) else None
        return cls(data, attr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_npfuncs.py ===
"""
Tests for np.foo applied to DataFrame, not necessarily ufuncs.
"""
import numpy as np

from pandas import (
    Categorical,
    DataFrame,
)
import pandas._testing as tm


class TestAsArray:
    def test_asarray_homogeneous(self):
        df = DataFrame({"A": Categorical([1, 2]), "B": Categorical([1, 2])})
        result = np.asarray(df)
        # may change from object in the future
        expected = np.array([[1, 1], [2, 2]], dtype="object")
        tm.assert_numpy_array_equal(result, expected)

    def test_np_sqrt(self, float_frame):
        with np.errstate(all="ignore"):
            result = np.sqrt(float_frame)
        assert isinstance(result, type(float_frame))
        assert result.index.is_(float_frame.index)
        assert result.columns.is_(float_frame.columns)

        tm.assert_frame_equal(result, float_frame.apply(np.sqrt))

    def test_sum_deprecated_axis_behavior(self):
        # GH#52042 deprecated behavior of df.sum(axis=None), which gets
        #  called when we do np.sum(df)

        arr = np.random.default_rng(2).standard_normal((4, 3))
        df = DataFrame(arr)

        msg = "The behavior of DataFrame.sum with axis=None is deprecated"
        with tm.assert_produces_warning(
            FutureWarning, match=msg, check_stacklevel=False
        ):
            res = np.sum(df)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df.sum(axis=None)
        tm.assert_series_equal(res, expected)

    def test_np_ravel(self):
        # GH26247
        arr = np.array(
            [
                [0.11197053, 0.44361564, -0.92589452],
                [0.05883648, -0.00948922, -0.26469934],
            ]
        )

        result = np.ravel([DataFrame(batch.reshape(1, 3)) for batch in arr])
        expected = np.array(
            [
                0.11197053,
                0.44361564,
                -0.92589452,
                0.05883648,
                -0.00948922,
                -0.26469934,
            ]
        )
        tm.assert_numpy_array_equal(result, expected)

        result = np.ravel(DataFrame(arr[0].reshape(1, 3), columns=["x1", "x2", "x3"]))
        expected = np.array([0.11197053, 0.44361564, -0.92589452])
        tm.assert_numpy_array_equal(result, expected)

        result = np.ravel(
            [
                DataFrame(batch.reshape(1, 3), columns=["x1", "x2", "x3"])
                for batch in arr
            ]
        )
        expected = np.array(
            [
                0.11197053,
                0.44361564,
                -0.92589452,
                0.05883648,
                -0.00948922,
                -0.26469934,
            ]
        )
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_to_period.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    PeriodIndex,
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestToPeriod:
    def test_to_period(self, frame_or_series):
        K = 5

        dr = date_range("1/1/2000", "1/1/2001", freq="D")
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(dr), K)),
            index=dr,
            columns=["A", "B", "C", "D", "E"],
        )
        obj["mix"] = "a"
        obj = tm.get_obj(obj, frame_or_series)

        pts = obj.to_period()
        exp = obj.copy()
        exp.index = period_range("1/1/2000", "1/1/2001")
        tm.assert_equal(pts, exp)

        pts = obj.to_period("M")
        exp.index = exp.index.asfreq("M")
        tm.assert_equal(pts, exp)

    def test_to_period_without_freq(self, frame_or_series):
        # GH#7606 without freq
        idx = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"])
        exp_idx = PeriodIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"], freq="D"
        )

        obj = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), index=idx, columns=idx
        )
        obj = tm.get_obj(obj, frame_or_series)
        expected = obj.copy()
        expected.index = exp_idx
        tm.assert_equal(obj.to_period(), expected)

        if frame_or_series is DataFrame:
            expected = obj.copy()
            expected.columns = exp_idx
            tm.assert_frame_equal(obj.to_period(axis=1), expected)

    def test_to_period_columns(self):
        dr = date_range("1/1/2000", "1/1/2001")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(dr), 5)), index=dr)
        df["mix"] = "a"

        df = df.T
        pts = df.to_period(axis=1)
        exp = df.copy()
        exp.columns = period_range("1/1/2000", "1/1/2001")
        tm.assert_frame_equal(pts, exp)

        pts = df.to_period("M", axis=1)
        tm.assert_index_equal(pts.columns, exp.columns.asfreq("M"))

    def test_to_period_invalid_axis(self):
        dr = date_range("1/1/2000", "1/1/2001")
        df = DataFrame(np.random.default_rng(2).standard_normal((len(dr), 5)), index=dr)
        df["mix"] = "a"

        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.to_period(axis=2)

    def test_to_period_raises(self, index, frame_or_series):
        # https://github.com/pandas-dev/pandas/issues/33327
        obj = Series(index=index, dtype=object)
        if frame_or_series is DataFrame:
            obj = obj.to_frame()

        if not isinstance(index, DatetimeIndex):
            msg = f"unsupported Type {type(index).__name__}"
            with pytest.raises(TypeError, match=msg):
                obj.to_period()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_numba.py ===
import pytest

from pandas.compat import is_platform_arm

from pandas import (
    DataFrame,
    Series,
    option_context,
)
import pandas._testing as tm
from pandas.util.version import Version

pytestmark = [pytest.mark.single_cpu]

numba = pytest.importorskip("numba")
pytestmark.append(
    pytest.mark.skipif(
        Version(numba.__version__) == Version("0.61") and is_platform_arm(),
        reason=f"Segfaults on ARM platforms with numba {numba.__version__}",
    )
)


@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestEngine:
    def test_cython_vs_numba_frame(
        self, sort, nogil, parallel, nopython, numba_supported_reductions
    ):
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        gb = df.groupby("a", sort=sort)
        result = getattr(gb, func)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(gb, func)(**kwargs)
        tm.assert_frame_equal(result, expected)

    def test_cython_vs_numba_getitem(
        self, sort, nogil, parallel, nopython, numba_supported_reductions
    ):
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        gb = df.groupby("a", sort=sort)["c"]
        result = getattr(gb, func)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(gb, func)(**kwargs)
        tm.assert_series_equal(result, expected)

    def test_cython_vs_numba_series(
        self, sort, nogil, parallel, nopython, numba_supported_reductions
    ):
        func, kwargs = numba_supported_reductions
        ser = Series(range(3), index=[1, 2, 1], name="foo")
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        gb = ser.groupby(level=0, sort=sort)
        result = getattr(gb, func)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        expected = getattr(gb, func)(**kwargs)
        tm.assert_series_equal(result, expected)

    def test_as_index_false_unsupported(self, numba_supported_reductions):
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        gb = df.groupby("a", as_index=False)
        with pytest.raises(NotImplementedError, match="as_index=False"):
            getattr(gb, func)(engine="numba", **kwargs)

    def test_axis_1_unsupported(self, numba_supported_reductions):
        func, kwargs = numba_supported_reductions
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        gb = df.groupby("a", axis=1)
        with pytest.raises(NotImplementedError, match="axis=1"):
            getattr(gb, func)(engine="numba", **kwargs)

    def test_no_engine_doesnt_raise(self):
        # GH55520
        df = DataFrame({"a": [3, 2, 3, 2], "b": range(4), "c": range(1, 5)})
        gb = df.groupby("a")
        # Make sure behavior of functions w/out engine argument don't raise
        # when the global use_numba option is set
        with option_context("compute.use_numba", True):
            res = gb.agg({"b": "first"})
        expected = gb.agg({"b": "first"})
        tm.assert_frame_equal(res, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimelike_\test_drop_duplicates.py ===
import numpy as np
import pytest

from pandas import (
    PeriodIndex,
    Series,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class DropDuplicates:
    def test_drop_duplicates_metadata(self, idx):
        # GH#10115
        result = idx.drop_duplicates()
        tm.assert_index_equal(idx, result)
        assert idx.freq == result.freq

        idx_dup = idx.append(idx)
        result = idx_dup.drop_duplicates()

        expected = idx
        if not isinstance(idx, PeriodIndex):
            # freq is reset except for PeriodIndex
            assert idx_dup.freq is None
            assert result.freq is None
            expected = idx._with_freq(None)
        else:
            assert result.freq == expected.freq

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "keep, expected, index",
        [
            (
                "first",
                np.concatenate(([False] * 10, [True] * 5)),
                np.arange(0, 10, dtype=np.int64),
            ),
            (
                "last",
                np.concatenate(([True] * 5, [False] * 10)),
                np.arange(5, 15, dtype=np.int64),
            ),
            (
                False,
                np.concatenate(([True] * 5, [False] * 5, [True] * 5)),
                np.arange(5, 10, dtype=np.int64),
            ),
        ],
    )
    def test_drop_duplicates(self, keep, expected, index, idx):
        # to check Index/Series compat
        idx = idx.append(idx[:5])

        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected)
        expected = idx[~expected]

        result = idx.drop_duplicates(keep=keep)
        tm.assert_index_equal(result, expected)

        result = Series(idx).drop_duplicates(keep=keep)
        expected = Series(expected, index=index)
        tm.assert_series_equal(result, expected)


class TestDropDuplicatesPeriodIndex(DropDuplicates):
    @pytest.fixture(params=["D", "3D", "h", "2h", "min", "2min", "s", "3s"])
    def freq(self, request):
        return request.param

    @pytest.fixture
    def idx(self, freq):
        return period_range("2011-01-01", periods=10, freq=freq, name="idx")


class TestDropDuplicatesDatetimeIndex(DropDuplicates):
    @pytest.fixture
    def idx(self, freq_sample):
        return date_range("2011-01-01", freq=freq_sample, periods=10, name="idx")


class TestDropDuplicatesTimedeltaIndex(DropDuplicates):
    @pytest.fixture
    def idx(self, freq_sample):
        return timedelta_range("1 day", periods=10, freq=freq_sample, name="idx")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\interchange\test_utils.py ===
import numpy as np
import pytest

import pandas as pd
from pandas.core.interchange.utils import dtype_to_arrow_c_fmt

# TODO: use ArrowSchema to get reference C-string.
# At the time, there is no way to access ArrowSchema holding a type format string
# from python. The only way to access it is to export the structure to a C-pointer,
# see DataType._export_to_c() method defined in
# https://github.com/apache/arrow/blob/master/python/pyarrow/types.pxi


@pytest.mark.parametrize(
    "pandas_dtype, c_string",
    [
        (np.dtype("bool"), "b"),
        (np.dtype("int8"), "c"),
        (np.dtype("uint8"), "C"),
        (np.dtype("int16"), "s"),
        (np.dtype("uint16"), "S"),
        (np.dtype("int32"), "i"),
        (np.dtype("uint32"), "I"),
        (np.dtype("int64"), "l"),
        (np.dtype("uint64"), "L"),
        (np.dtype("float16"), "e"),
        (np.dtype("float32"), "f"),
        (np.dtype("float64"), "g"),
        (pd.Series(["a"]).dtype, "u"),
        (
            pd.Series([0]).astype("datetime64[ns]").dtype,
            "tsn:",
        ),
        (pd.CategoricalDtype(["a"]), "l"),
        (np.dtype("O"), "u"),
    ],
)
def test_dtype_to_arrow_c_fmt(pandas_dtype, c_string):  # PR01
    """Test ``dtype_to_arrow_c_fmt`` utility function."""
    assert dtype_to_arrow_c_fmt(pandas_dtype) == c_string


@pytest.mark.parametrize(
    "pa_dtype, args_kwargs, c_string",
    [
        ["null", {}, "n"],
        ["bool_", {}, "b"],
        ["uint8", {}, "C"],
        ["uint16", {}, "S"],
        ["uint32", {}, "I"],
        ["uint64", {}, "L"],
        ["int8", {}, "c"],
        ["int16", {}, "S"],
        ["int32", {}, "i"],
        ["int64", {}, "l"],
        ["float16", {}, "e"],
        ["float32", {}, "f"],
        ["float64", {}, "g"],
        ["string", {}, "u"],
        ["binary", {}, "z"],
        ["time32", ("s",), "tts"],
        ["time32", ("ms",), "ttm"],
        ["time64", ("us",), "ttu"],
        ["time64", ("ns",), "ttn"],
        ["date32", {}, "tdD"],
        ["date64", {}, "tdm"],
        ["timestamp", {"unit": "s"}, "tss:"],
        ["timestamp", {"unit": "ms"}, "tsm:"],
        ["timestamp", {"unit": "us"}, "tsu:"],
        ["timestamp", {"unit": "ns"}, "tsn:"],
        ["timestamp", {"unit": "ns", "tz": "UTC"}, "tsn:UTC"],
        ["duration", ("s",), "tDs"],
        ["duration", ("ms",), "tDm"],
        ["duration", ("us",), "tDu"],
        ["duration", ("ns",), "tDn"],
        ["decimal128", {"precision": 4, "scale": 2}, "d:4,2"],
    ],
)
def test_dtype_to_arrow_c_fmt_arrowdtype(pa_dtype, args_kwargs, c_string):
    # GH 52323
    pa = pytest.importorskip("pyarrow")
    if not args_kwargs:
        pa_type = getattr(pa, pa_dtype)()
    elif isinstance(args_kwargs, tuple):
        pa_type = getattr(pa, pa_dtype)(*args_kwargs)
    else:
        pa_type = getattr(pa, pa_dtype)(**args_kwargs)
    arrow_type = pd.ArrowDtype(pa_type)
    assert dtype_to_arrow_c_fmt(arrow_type) == c_string

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_equal.py ===
from sympy.core.numbers import Rational
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.exponential import exp


def test_equal():
    b = Symbol("b")
    a = Symbol("a")
    e1 = a + b
    e2 = 2*a*b
    e3 = a**3*b**2
    e4 = a*b + b*a
    assert not e1 == e2
    assert not e1 == e2
    assert e1 != e2
    assert e2 == e4
    assert e2 != e3
    assert not e2 == e3

    x = Symbol("x")
    e1 = exp(x + 1/x)
    y = Symbol("x")
    e2 = exp(y + 1/y)
    assert e1 == e2
    assert not e1 != e2
    y = Symbol("y")
    e2 = exp(y + 1/y)
    assert not e1 == e2
    assert e1 != e2

    e5 = Rational(3) + 2*x - x - x
    assert e5 == 3
    assert 3 == e5
    assert e5 != 4
    assert 4 != e5
    assert e5 != 3 + x
    assert 3 + x != e5


def test_expevalbug():
    x = Symbol("x")
    e1 = exp(1*x)
    e3 = exp(x)
    assert e1 == e3


def test_cmp_bug1():
    class T:
        pass

    t = T()
    x = Symbol("x")

    assert not (x == t)
    assert (x != t)


def test_cmp_bug2():
    class T:
        pass

    t = T()

    assert not (Symbol == t)
    assert (Symbol != t)


def test_cmp_issue_4357():
    """ Check that Basic subclasses can be compared with sympifiable objects.

    https://github.com/sympy/sympy/issues/4357
    """
    assert not (Symbol == 1)
    assert (Symbol != 1)
    assert not (Symbol == 'x')
    assert (Symbol != 'x')


def test_dummy_eq():
    x = Symbol('x')
    y = Symbol('y')

    u = Dummy('u')

    assert (u**2 + 1).dummy_eq(x**2 + 1) is True
    assert ((u**2 + 1) == (x**2 + 1)) is False

    assert (u**2 + y).dummy_eq(x**2 + y, x) is True
    assert (u**2 + y).dummy_eq(x**2 + y, y) is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_beta_functions.py ===
from sympy.core.function import (diff, expand_func)
from sympy.core.numbers import I, Rational, pi
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.beta_functions import (beta, betainc, betainc_regularized)
from sympy.functions.special.gamma_functions import gamma, polygamma
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.core.function import ArgumentIndexError
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises


def test_beta():
    x, y = symbols('x y')
    t = Dummy('t')

    assert unchanged(beta, x, y)
    assert unchanged(beta, x, x)

    assert beta(5, -3).is_real == True
    assert beta(3, y).is_real is None

    assert expand_func(beta(x, y)) == gamma(x)*gamma(y)/gamma(x + y)
    assert expand_func(beta(x, y) - beta(y, x)) == 0  # Symmetric
    assert expand_func(beta(x, y)) == expand_func(beta(x, y + 1) + beta(x + 1, y)).simplify()

    assert diff(beta(x, y), x) == beta(x, y)*(polygamma(0, x) - polygamma(0, x + y))
    assert diff(beta(x, y), y) == beta(x, y)*(polygamma(0, y) - polygamma(0, x + y))

    assert conjugate(beta(x, y)) == beta(conjugate(x), conjugate(y))

    raises(ArgumentIndexError, lambda: beta(x, y).fdiff(3))

    assert beta(x, y).rewrite(gamma) == gamma(x)*gamma(y)/gamma(x + y)
    assert beta(x).rewrite(gamma) == gamma(x)**2/gamma(2*x)
    assert beta(x, y).rewrite(Integral).dummy_eq(Integral(t**(x - 1) * (1 - t)**(y - 1), (t, 0, 1)))
    assert beta(Rational(-19, 10), Rational(-1, 10)) == S.Zero
    assert beta(Rational(-19, 10), Rational(-9, 10)) == \
        800*2**(S(4)/5)*sqrt(pi)*gamma(S.One/10)/(171*gamma(-S(7)/5))
    assert beta(Rational(19, 10), Rational(29, 10)) == 100/(551*catalan(Rational(19, 10)))
    assert beta(1, 0) == S.ComplexInfinity
    assert beta(0, 1) == S.ComplexInfinity
    assert beta(2, 3) == S.One/12
    assert unchanged(beta, x, x + 1)
    assert unchanged(beta, x, 1)
    assert unchanged(beta, 1, y)
    assert beta(x, x + 1).doit() == 1/(x*(x+1)*catalan(x))
    assert beta(1, y).doit() == 1/y
    assert beta(x, 1).doit() == 1/x
    assert beta(Rational(-19, 10), Rational(-1, 10), evaluate=False).doit() == S.Zero
    assert beta(2) == beta(2, 2)
    assert beta(x, evaluate=False) != beta(x, x)
    assert beta(x, evaluate=False).doit() == beta(x, x)


def test_betainc():
    a, b, x1, x2 = symbols('a b x1 x2')

    assert unchanged(betainc, a, b, x1, x2)
    assert unchanged(betainc, a, b, 0, x1)

    assert betainc(1, 2, 0, -5).is_real == True
    assert betainc(1, 2, 0, x2).is_real is None
    assert conjugate(betainc(I, 2, 3 - I, 1 + 4*I)) == betainc(-I, 2, 3 + I, 1 - 4*I)

    assert betainc(a, b, 0, 1).rewrite(Integral).dummy_eq(beta(a, b).rewrite(Integral))
    assert betainc(1, 2, 0, x2).rewrite(hyper) == x2*hyper((1, -1), (2,), x2)

    assert betainc(1, 2, 3, 3).evalf() == 0


def test_betainc_regularized():
    a, b, x1, x2 = symbols('a b x1 x2')

    assert unchanged(betainc_regularized, a, b, x1, x2)
    assert unchanged(betainc_regularized, a, b, 0, x1)

    assert betainc_regularized(3, 5, 0, -1).is_real == True
    assert betainc_regularized(3, 5, 0, x2).is_real is None
    assert conjugate(betainc_regularized(3*I, 1, 2 + I, 1 + 2*I)) == betainc_regularized(-3*I, 1, 2 - I, 1 - 2*I)

    assert betainc_regularized(a, b, 0, 1).rewrite(Integral) == 1
    assert betainc_regularized(1, 2, x1, x2).rewrite(hyper) == 2*x2*hyper((1, -1), (2,), x2) - 2*x1*hyper((1, -1), (2,), x1)

    assert betainc_regularized(4, 1, 5, 5).evalf() == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_qasm.py ===
from sympy.physics.quantum.qasm import Qasm, flip_index, trim,\
     get_index, nonblank, fullsplit, fixcommand, stripquotes, read_qasm
from sympy.physics.quantum.gate import X, Z, H, S, T
from sympy.physics.quantum.gate import CNOT, SWAP, CPHASE, CGate, CGateS
from sympy.physics.quantum.circuitplot import Mz

def test_qasm_readqasm():
    qasm_lines = """\
    qubit q_0
    qubit q_1
    h q_0
    cnot q_0,q_1
    """
    q = read_qasm(qasm_lines)
    assert q.get_circuit() == CNOT(1,0)*H(1)

def test_qasm_ex1():
    q = Qasm('qubit q0', 'qubit q1', 'h q0', 'cnot q0,q1')
    assert q.get_circuit() == CNOT(1,0)*H(1)

def test_qasm_ex1_methodcalls():
    q = Qasm()
    q.qubit('q_0')
    q.qubit('q_1')
    q.h('q_0')
    q.cnot('q_0', 'q_1')
    assert q.get_circuit() == CNOT(1,0)*H(1)

def test_qasm_swap():
    q = Qasm('qubit q0', 'qubit q1', 'cnot q0,q1', 'cnot q1,q0', 'cnot q0,q1')
    assert q.get_circuit() == CNOT(1,0)*CNOT(0,1)*CNOT(1,0)


def test_qasm_ex2():
    q = Qasm('qubit q_0', 'qubit q_1', 'qubit q_2', 'h  q_1',
             'cnot q_1,q_2', 'cnot q_0,q_1', 'h q_0',
             'measure q_1', 'measure q_0',
             'c-x q_1,q_2', 'c-z q_0,q_2')
    assert q.get_circuit() == CGate(2,Z(0))*CGate(1,X(0))*Mz(2)*Mz(1)*H(2)*CNOT(2,1)*CNOT(1,0)*H(1)

def test_qasm_1q():
    for symbol, gate in [('x', X), ('z', Z), ('h', H), ('s', S), ('t', T), ('measure', Mz)]:
        q = Qasm('qubit q_0', '%s q_0' % symbol)
        assert q.get_circuit() == gate(0)

def test_qasm_2q():
    for symbol, gate in [('cnot', CNOT), ('swap', SWAP), ('cphase', CPHASE)]:
        q = Qasm('qubit q_0', 'qubit q_1', '%s q_0,q_1' % symbol)
        assert q.get_circuit() == gate(1,0)

def test_qasm_3q():
    q = Qasm('qubit q0', 'qubit q1', 'qubit q2', 'toffoli q2,q1,q0')
    assert q.get_circuit() == CGateS((0,1),X(2))

def test_qasm_flip_index():
    assert flip_index(0, 2) == 1
    assert flip_index(1, 2) == 0

def test_qasm_trim():
    assert trim('nothing happens here') == 'nothing happens here'
    assert trim("Something #happens here") == "Something "

def test_qasm_get_index():
    assert get_index('q0', ['q0', 'q1']) == 1
    assert get_index('q1', ['q0', 'q1']) == 0

def test_qasm_nonblank():
    assert list(nonblank('abcd')) == list('abcd')
    assert list(nonblank('abc ')) == list('abc')

def test_qasm_fullsplit():
    assert fullsplit('g q0,q1,q2,  q3') == ('g', ['q0', 'q1', 'q2', 'q3'])

def test_qasm_fixcommand():
    assert fixcommand('foo') == 'foo'
    assert fixcommand('def') == 'qdef'

def test_qasm_stripquotes():
    assert stripquotes("'S'") == 'S'
    assert stripquotes('"S"') == 'S'
    assert stripquotes('S') == 'S'

def test_qasm_qdef():
    # weaker test condition (str) since we don't have access to the actual class
    q = Qasm("def Q,0,Q",'qubit q0','Q q0')
    assert str(q.get_circuit()) == 'Q(0)'

    q = Qasm("def CQ,1,Q", 'qubit q0', 'qubit q1', 'CQ q0,q1')
    assert str(q.get_circuit()) == 'C((1),Q(0))'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_diff.py ===
import numpy as np
import pytest

from pandas import (
    Series,
    TimedeltaIndex,
    date_range,
)
import pandas._testing as tm


class TestSeriesDiff:
    def test_diff_np(self):
        # TODO(__array_function__): could make np.diff return a Series
        #  matching ser.diff()

        ser = Series(np.arange(5))

        res = np.diff(ser)
        expected = np.array([1, 1, 1, 1])
        tm.assert_numpy_array_equal(res, expected)

    def test_diff_int(self):
        # int dtype
        a = 10000000000000000
        b = a + 1
        ser = Series([a, b])

        result = ser.diff()
        assert result[1] == 1

    def test_diff_tz(self):
        # Combined datetime diff, normal diff and boolean diff test
        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        ts.diff()

        # neg n
        result = ts.diff(-1)
        expected = ts - ts.shift(-1)
        tm.assert_series_equal(result, expected)

        # 0
        result = ts.diff(0)
        expected = ts - ts
        tm.assert_series_equal(result, expected)

    def test_diff_dt64(self):
        # datetime diff (GH#3100)
        ser = Series(date_range("20130102", periods=5))
        result = ser.diff()
        expected = ser - ser.shift(1)
        tm.assert_series_equal(result, expected)

        # timedelta diff
        result = result - result.shift(1)  # previous result
        expected = expected.diff()  # previously expected
        tm.assert_series_equal(result, expected)

    def test_diff_dt64tz(self):
        # with tz
        ser = Series(
            date_range("2000-01-01 09:00:00", periods=5, tz="US/Eastern"), name="foo"
        )
        result = ser.diff()
        expected = Series(TimedeltaIndex(["NaT"] + ["1 days"] * 4), name="foo")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input,output,diff",
        [([False, True, True, False, False], [np.nan, True, False, True, False], 1)],
    )
    def test_diff_bool(self, input, output, diff):
        # boolean series (test for fixing #17294)
        ser = Series(input)
        result = ser.diff()
        expected = Series(output)
        tm.assert_series_equal(result, expected)

    def test_diff_object_dtype(self):
        # object series
        ser = Series([False, True, 5.0, np.nan, True, False])
        result = ser.diff()
        expected = ser - ser.shift(1)
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\tests\test_plotting.py ===
from sympy.external.importtools import import_module

disabled = False

# if pyglet.gl fails to import, e.g. opengl is missing, we disable the tests
pyglet_gl = import_module("pyglet.gl", catch=(OSError,))
pyglet_window = import_module("pyglet.window", catch=(OSError,))
if not pyglet_gl or not pyglet_window:
    disabled = True


from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
x, y, z = symbols('x, y, z')


def test_plot_2d():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(x, [x, -5, 5, 4], visible=False)
    p.wait_for_calculations()


def test_plot_2d_discontinuous():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(1/x, [x, -1, 1, 2], visible=False)
    p.wait_for_calculations()


def test_plot_3d():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(x*y, [x, -5, 5, 5], [y, -5, 5, 5], visible=False)
    p.wait_for_calculations()


def test_plot_3d_discontinuous():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(1/x, [x, -3, 3, 6], [y, -1, 1, 1], visible=False)
    p.wait_for_calculations()


def test_plot_2d_polar():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(1/x, [x, -1, 1, 4], 'mode=polar', visible=False)
    p.wait_for_calculations()


def test_plot_3d_cylinder():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(
        1/y, [x, 0, 6.282, 4], [y, -1, 1, 4], 'mode=polar;style=solid',
        visible=False)
    p.wait_for_calculations()


def test_plot_3d_spherical():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(
        1, [x, 0, 6.282, 4], [y, 0, 3.141,
            4], 'mode=spherical;style=wireframe',
        visible=False)
    p.wait_for_calculations()


def test_plot_2d_parametric():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(sin(x), cos(x), [x, 0, 6.282, 4], visible=False)
    p.wait_for_calculations()


def test_plot_3d_parametric():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(sin(x), cos(x), x/5.0, [x, 0, 6.282, 4], visible=False)
    p.wait_for_calculations()


def _test_plot_log():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(log(x), [x, 0, 6.282, 4], 'mode=polar', visible=False)
    p.wait_for_calculations()


def test_plot_integral():
    # Make sure it doesn't treat x as an independent variable
    from sympy.plotting.pygletplot import PygletPlot
    from sympy.integrals.integrals import Integral
    p = PygletPlot(Integral(z*x, (x, 1, z), (z, 1, y)), visible=False)
    p.wait_for_calculations()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\unify\tests\test_unify.py ===
from sympy.unify.core import Compound, Variable, CondVariable, allcombinations
from sympy.unify import core

a,b,c = 'a', 'b', 'c'
w,x,y,z = map(Variable, 'wxyz')

C = Compound

def is_associative(x):
    return isinstance(x, Compound) and (x.op in ('Add', 'Mul', 'CAdd', 'CMul'))
def is_commutative(x):
    return isinstance(x, Compound) and (x.op in ('CAdd', 'CMul'))


def unify(a, b, s={}):
    return core.unify(a, b, s=s, is_associative=is_associative,
                          is_commutative=is_commutative)

def test_basic():
    assert list(unify(a, x, {})) == [{x: a}]
    assert list(unify(a, x, {x: 10})) == []
    assert list(unify(1, x, {})) == [{x: 1}]
    assert list(unify(a, a, {})) == [{}]
    assert list(unify((w, x), (y, z), {})) == [{w: y, x: z}]
    assert list(unify(x, (a, b), {})) == [{x: (a, b)}]

    assert list(unify((a, b), (x, x), {})) == []
    assert list(unify((y, z), (x, x), {}))!= []
    assert list(unify((a, (b, c)), (a, (x, y)), {})) == [{x: b, y: c}]

def test_ops():
    assert list(unify(C('Add', (a,b,c)), C('Add', (a,x,y)), {})) == \
            [{x:b, y:c}]
    assert list(unify(C('Add', (C('Mul', (1,2)), b,c)), C('Add', (x,y,c)), {})) == \
            [{x: C('Mul', (1,2)), y:b}]

def test_associative():
    c1 = C('Add', (1,2,3))
    c2 = C('Add', (x,y))
    assert tuple(unify(c1, c2, {})) == ({x: 1, y: C('Add', (2, 3))},
                                         {x: C('Add', (1, 2)), y: 3})

def test_commutative():
    c1 = C('CAdd', (1,2,3))
    c2 = C('CAdd', (x,y))
    result = list(unify(c1, c2, {}))
    assert  {x: 1, y: C('CAdd', (2, 3))} in result
    assert ({x: 2, y: C('CAdd', (1, 3))} in result or
            {x: 2, y: C('CAdd', (3, 1))} in result)

def _test_combinations_assoc():
    assert set(allcombinations((1,2,3), (a,b), True)) == \
        {(((1, 2), (3,)), (a, b)), (((1,), (2, 3)), (a, b))}

def _test_combinations_comm():
    assert set(allcombinations((1,2,3), (a,b), None)) == \
        {(((1,), (2, 3)), ('a', 'b')), (((2,), (3, 1)), ('a', 'b')),
             (((3,), (1, 2)), ('a', 'b')), (((1, 2), (3,)), ('a', 'b')),
             (((2, 3), (1,)), ('a', 'b')), (((3, 1), (2,)), ('a', 'b'))}

def test_allcombinations():
    assert set(allcombinations((1,2), (1,2), 'commutative')) ==\
        {(((1,),(2,)), ((1,),(2,))), (((1,),(2,)), ((2,),(1,)))}


def test_commutativity():
    c1 = Compound('CAdd', (a, b))
    c2 = Compound('CAdd', (x, y))
    assert is_commutative(c1) and is_commutative(c2)
    assert len(list(unify(c1, c2, {}))) == 2


def test_CondVariable():
    expr = C('CAdd', (1, 2))
    x = Variable('x')
    y = CondVariable('y', lambda a: a % 2 == 0)
    z = CondVariable('z', lambda a: a > 3)
    pattern = C('CAdd', (x, y))
    assert list(unify(expr, pattern, {})) == \
            [{x: 1, y: 2}]

    z = CondVariable('z', lambda a: a > 3)
    pattern = C('CAdd', (z, y))

    assert list(unify(expr, pattern, {})) == []

def test_defaultdict():
    assert next(unify(Variable('x'), 'foo')) == {Variable('x'): 'foo'}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\tests\test_deprecations.py ===
"""Test deprecation and future warnings.

"""
import io
import textwrap

import pytest

import numpy as np
from numpy.ma.core import MaskedArrayFutureWarning
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_warns


class TestArgsort:
    """ gh-8701 """
    def _test_base(self, argsort, cls):
        arr_0d = np.array(1).view(cls)
        argsort(arr_0d)

        arr_1d = np.array([1, 2, 3]).view(cls)
        argsort(arr_1d)

        # argsort has a bad default for >1d arrays
        arr_2d = np.array([[1, 2], [3, 4]]).view(cls)
        result = assert_warns(
            np.ma.core.MaskedArrayFutureWarning, argsort, arr_2d)
        assert_equal(result, argsort(arr_2d, axis=None))

        # should be no warnings for explicitly specifying it
        argsort(arr_2d, axis=None)
        argsort(arr_2d, axis=-1)

    def test_function_ndarray(self):
        return self._test_base(np.ma.argsort, np.ndarray)

    def test_function_maskedarray(self):
        return self._test_base(np.ma.argsort, np.ma.MaskedArray)

    def test_method(self):
        return self._test_base(np.ma.MaskedArray.argsort, np.ma.MaskedArray)


class TestMinimumMaximum:

    def test_axis_default(self):
        # NumPy 1.13, 2017-05-06

        data1d = np.ma.arange(6)
        data2d = data1d.reshape(2, 3)

        ma_min = np.ma.minimum.reduce
        ma_max = np.ma.maximum.reduce

        # check that the default axis is still None, but warns on 2d arrays
        result = assert_warns(MaskedArrayFutureWarning, ma_max, data2d)
        assert_equal(result, ma_max(data2d, axis=None))

        result = assert_warns(MaskedArrayFutureWarning, ma_min, data2d)
        assert_equal(result, ma_min(data2d, axis=None))

        # no warnings on 1d, as both new and old defaults are equivalent
        result = ma_min(data1d)
        assert_equal(result, ma_min(data1d, axis=None))
        assert_equal(result, ma_min(data1d, axis=0))

        result = ma_max(data1d)
        assert_equal(result, ma_max(data1d, axis=None))
        assert_equal(result, ma_max(data1d, axis=0))


class TestFromtextfile:
    def test_fromtextfile_delimitor(self):
        # NumPy 1.22.0, 2021-09-23

        textfile = io.StringIO(textwrap.dedent(
            """
            A,B,C,D
            'string 1';1;1.0;'mixed column'
            'string 2';2;2.0;
            'string 3';3;3.0;123
            'string 4';4;4.0;3.14
            """
        ))

        with pytest.warns(DeprecationWarning):
            result = np.ma.mrecords.fromtextfile(textfile, delimitor=';')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\ndarray_conversion.py ===
import os
import tempfile

import numpy as np

nd = np.array([[1, 2], [3, 4]])
scalar_array = np.array(1)

# item
scalar_array.item()
nd.item(1)
nd.item(0, 1)
nd.item((0, 1))

# tobytes
nd.tobytes()
nd.tobytes("C")
nd.tobytes(None)

# tofile
if os.name != "nt":
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        nd.tofile(tmp.name)
        nd.tofile(tmp.name, "")
        nd.tofile(tmp.name, sep="")

        nd.tofile(tmp.name, "", "%s")
        nd.tofile(tmp.name, format="%s")

        nd.tofile(tmp)

# dump is pretty simple
# dumps is pretty simple

# astype
nd.astype("float")
nd.astype(float)

nd.astype(float, "K")
nd.astype(float, order="K")

nd.astype(float, "K", "unsafe")
nd.astype(float, casting="unsafe")

nd.astype(float, "K", "unsafe", True)
nd.astype(float, subok=True)

nd.astype(float, "K", "unsafe", True, True)
nd.astype(float, copy=True)

# byteswap
nd.byteswap()
nd.byteswap(True)

# copy
nd.copy()
nd.copy("C")

# view
nd.view()
nd.view(np.int64)
nd.view(dtype=np.int64)
nd.view(np.int64, np.matrix)
nd.view(type=np.matrix)

# getfield
complex_array = np.array([[1 + 1j, 0], [0, 1 - 1j]], dtype=np.complex128)

complex_array.getfield("float")
complex_array.getfield(float)

complex_array.getfield("float", 8)
complex_array.getfield(float, offset=8)

# setflags
nd.setflags()

nd.setflags(True)
nd.setflags(write=True)

nd.setflags(True, True)
nd.setflags(write=True, align=True)

nd.setflags(True, True, False)
nd.setflags(write=True, align=True, uic=False)

# fill is pretty simple

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\casting.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock


class BaseCastingTests:
    """Casting to and from ExtensionDtypes"""

    def test_astype_object_series(self, all_data):
        ser = pd.Series(all_data, name="A")
        result = ser.astype(object)
        assert result.dtype == np.dtype(object)
        if hasattr(result._mgr, "blocks"):
            blk = result._mgr.blocks[0]
            assert isinstance(blk, NumpyBlock)
            assert blk.is_object
        assert isinstance(result._mgr.array, np.ndarray)
        assert result._mgr.array.dtype == np.dtype(object)

    def test_astype_object_frame(self, all_data):
        df = pd.DataFrame({"A": all_data})

        result = df.astype(object)
        if hasattr(result._mgr, "blocks"):
            blk = result._mgr.blocks[0]
            assert isinstance(blk, NumpyBlock), type(blk)
            assert blk.is_object
        assert isinstance(result._mgr.arrays[0], np.ndarray)
        assert result._mgr.arrays[0].dtype == np.dtype(object)

        # check that we can compare the dtypes
        comp = result.dtypes == df.dtypes
        assert not comp.any()

    def test_tolist(self, data):
        result = pd.Series(data).tolist()
        expected = list(data)
        assert result == expected

    def test_astype_str(self, data):
        result = pd.Series(data[:2]).astype(str)
        expected = pd.Series([str(x) for x in data[:2]], dtype=str)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "nullable_string_dtype",
        [
            "string[python]",
            pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_astype_string(self, data, nullable_string_dtype):
        # GH-33465, GH#45326 as of 2.0 we decode bytes instead of calling str(obj)
        result = pd.Series(data[:5]).astype(nullable_string_dtype)
        expected = pd.Series(
            [str(x) if not isinstance(x, bytes) else x.decode() for x in data[:5]],
            dtype=nullable_string_dtype,
        )
        tm.assert_series_equal(result, expected)

    def test_to_numpy(self, data):
        expected = np.asarray(data)

        result = data.to_numpy()
        tm.assert_equal(result, expected)

        result = pd.Series(data).to_numpy()
        tm.assert_equal(result, expected)

    def test_astype_empty_dataframe(self, dtype):
        # https://github.com/pandas-dev/pandas/issues/33113
        df = pd.DataFrame()
        result = df.astype(dtype)
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("copy", [True, False])
    def test_astype_own_type(self, data, copy):
        # ensure that astype returns the original object for equal dtype and copy=False
        # https://github.com/pandas-dev/pandas/issues/28488
        result = data.astype(data.dtype, copy=copy)
        assert (result is data) is (not copy)
        tm.assert_extension_array_equal(result, data)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_chaining_and_caching.py ===
import numpy as np
import pytest

from pandas._libs import index as libindex
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


def test_detect_chained_assignment(using_copy_on_write, warn_copy_on_write):
    # Inplace ops, originally from:
    # https://stackoverflow.com/questions/20508968/series-fillna-in-a-multiindex-dataframe-does-not-fill-is-this-a-bug
    a = [12, 23]
    b = [123, None]
    c = [1234, 2345]
    d = [12345, 23456]
    tuples = [("eyes", "left"), ("eyes", "right"), ("ears", "left"), ("ears", "right")]
    events = {
        ("eyes", "left"): a,
        ("eyes", "right"): b,
        ("ears", "left"): c,
        ("ears", "right"): d,
    }
    multiind = MultiIndex.from_tuples(tuples, names=["part", "side"])
    zed = DataFrame(events, index=["a", "b"], columns=multiind)

    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            zed["eyes"]["right"].fillna(value=555, inplace=True)
    elif warn_copy_on_write:
        with tm.assert_produces_warning(None):
            zed["eyes"]["right"].fillna(value=555, inplace=True)
    else:
        msg = "A value is trying to be set on a copy of a slice from a DataFrame"
        with pytest.raises(SettingWithCopyError, match=msg):
            with tm.assert_produces_warning(None):
                zed["eyes"]["right"].fillna(value=555, inplace=True)


@td.skip_array_manager_invalid_test  # with ArrayManager df.loc[0] is not a view
def test_cache_updating(using_copy_on_write, warn_copy_on_write):
    # 5216
    # make sure that we don't try to set a dead cache
    a = np.random.default_rng(2).random((10, 3))
    df = DataFrame(a, columns=["x", "y", "z"])
    df_original = df.copy()
    tuples = [(i, j) for i in range(5) for j in range(2)]
    index = MultiIndex.from_tuples(tuples)
    df.index = index

    # setting via chained assignment
    # but actually works, since everything is a view

    with tm.raises_chained_assignment_error():
        df.loc[0]["z"].iloc[0] = 1.0

    if using_copy_on_write:
        assert df.loc[(0, 0), "z"] == df_original.loc[0, "z"]
    else:
        result = df.loc[(0, 0), "z"]
        assert result == 1

    # correct setting
    df.loc[(0, 0), "z"] = 2
    result = df.loc[(0, 0), "z"]
    assert result == 2


def test_indexer_caching(monkeypatch):
    # GH5727
    # make sure that indexers are in the _internal_names_set
    size_cutoff = 20
    with monkeypatch.context():
        monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
        index = MultiIndex.from_arrays([np.arange(size_cutoff), np.arange(size_cutoff)])
        s = Series(np.zeros(size_cutoff), index=index)

        # setitem
        s[s == 0] = 1
    expected = Series(np.ones(size_cutoff), index=index)
    tm.assert_series_equal(s, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_keys.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    HDFStore,
    Index,
    Series,
    date_range,
)
from pandas.tests.io.pytables.common import (
    ensure_clean_store,
    tables,
)

pytestmark = [pytest.mark.single_cpu]


def test_keys(setup_path):
    with ensure_clean_store(setup_path) as store:
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        store["b"] = Series(
            range(10), dtype="float64", index=[f"i_{i}" for i in range(10)]
        )
        store["c"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )

        assert len(store) == 3
        expected = {"/a", "/b", "/c"}
        assert set(store.keys()) == expected
        assert set(store) == expected


def test_non_pandas_keys(tmp_path, setup_path):
    class Table1(tables.IsDescription):
        value1 = tables.Float32Col()

    class Table2(tables.IsDescription):
        value2 = tables.Float32Col()

    class Table3(tables.IsDescription):
        value3 = tables.Float32Col()

    path = tmp_path / setup_path
    with tables.open_file(path, mode="w") as h5file:
        group = h5file.create_group("/", "group")
        h5file.create_table(group, "table1", Table1, "Table 1")
        h5file.create_table(group, "table2", Table2, "Table 2")
        h5file.create_table(group, "table3", Table3, "Table 3")
    with HDFStore(path) as store:
        assert len(store.keys(include="native")) == 3
        expected = {"/group/table1", "/group/table2", "/group/table3"}
        assert set(store.keys(include="native")) == expected
        assert set(store.keys(include="pandas")) == set()
        for name in expected:
            df = store.get(name)
            assert len(df.columns) == 1


def test_keys_illegal_include_keyword_value(setup_path):
    with ensure_clean_store(setup_path) as store:
        with pytest.raises(
            ValueError,
            match="`include` should be either 'pandas' or 'native' but is 'illegal'",
        ):
            store.keys(include="illegal")


def test_keys_ignore_hdf_softlink(setup_path):
    # GH 20523
    # Puts a softlink into HDF file and rereads

    with ensure_clean_store(setup_path) as store:
        df = DataFrame({"A": range(5), "B": range(5)})
        store.put("df", df)

        assert store.keys() == ["/df"]

        store._handle.create_soft_link(store._handle.root, "symlink", "df")

        # Should ignore the softlink
        assert store.keys() == ["/df"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\interval\test_interval.py ===
import numpy as np
import pytest

from pandas import (
    Interval,
    Timedelta,
    Timestamp,
)


@pytest.fixture
def interval():
    return Interval(0, 1)


class TestInterval:
    def test_properties(self, interval):
        assert interval.closed == "right"
        assert interval.left == 0
        assert interval.right == 1
        assert interval.mid == 0.5

    def test_hash(self, interval):
        # should not raise
        hash(interval)

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (0, 5, 5),
            (-2, 5.5, 7.5),
            (10, 10, 0),
            (10, np.inf, np.inf),
            (-np.inf, -5, np.inf),
            (-np.inf, np.inf, np.inf),
            (Timedelta("0 days"), Timedelta("5 days"), Timedelta("5 days")),
            (Timedelta("10 days"), Timedelta("10 days"), Timedelta("0 days")),
            (Timedelta("1h10min"), Timedelta("5h5min"), Timedelta("3h55min")),
            (Timedelta("5s"), Timedelta("1h"), Timedelta("59min55s")),
        ],
    )
    def test_length(self, left, right, expected):
        # GH 18789
        iv = Interval(left, right)
        result = iv.length
        assert result == expected

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            ("2017-01-01", "2017-01-06", "5 days"),
            ("2017-01-01", "2017-01-01 12:00:00", "12 hours"),
            ("2017-01-01 12:00", "2017-01-01 12:00:00", "0 days"),
            ("2017-01-01 12:01", "2017-01-05 17:31:00", "4 days 5 hours 30 min"),
        ],
    )
    @pytest.mark.parametrize("tz", (None, "UTC", "CET", "US/Eastern"))
    def test_length_timestamp(self, tz, left, right, expected):
        # GH 18789
        iv = Interval(Timestamp(left, tz=tz), Timestamp(right, tz=tz))
        result = iv.length
        expected = Timedelta(expected)
        assert result == expected

    @pytest.mark.parametrize(
        "left, right",
        [
            (0, 1),
            (Timedelta("0 days"), Timedelta("1 day")),
            (Timestamp("2018-01-01"), Timestamp("2018-01-02")),
            (
                Timestamp("2018-01-01", tz="US/Eastern"),
                Timestamp("2018-01-02", tz="US/Eastern"),
            ),
        ],
    )
    def test_is_empty(self, left, right, closed):
        # GH27219
        # non-empty always return False
        iv = Interval(left, right, closed)
        assert iv.is_empty is False

        # same endpoint is empty except when closed='both' (contains one point)
        iv = Interval(left, left, closed)
        result = iv.is_empty
        expected = closed != "both"
        assert result is expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_pc_groups.py ===
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup, DihedralGroup
from sympy.matrices import Matrix

def test_pc_presentation():
    Groups = [SymmetricGroup(3), SymmetricGroup(4), SymmetricGroup(9).sylow_subgroup(3),
         SymmetricGroup(9).sylow_subgroup(2), SymmetricGroup(8).sylow_subgroup(2), DihedralGroup(10)]

    S = SymmetricGroup(125).sylow_subgroup(5)
    G = S.derived_series()[2]
    Groups.append(G)

    G = SymmetricGroup(25).sylow_subgroup(5)
    Groups.append(G)

    S = SymmetricGroup(11**2).sylow_subgroup(11)
    G = S.derived_series()[2]
    Groups.append(G)

    for G in Groups:
        PcGroup = G.polycyclic_group()
        collector = PcGroup.collector
        pc_presentation = collector.pc_presentation

        pcgs = PcGroup.pcgs
        free_group = collector.free_group
        free_to_perm = {}
        for s, g in zip(free_group.symbols, pcgs):
            free_to_perm[s] = g

        for k, v in pc_presentation.items():
            k_array = k.array_form
            if v != ():
                v_array = v.array_form

            lhs = Permutation()
            for gen in k_array:
                s = gen[0]
                e = gen[1]
                lhs = lhs*free_to_perm[s]**e

            if v == ():
                assert lhs.is_identity
                continue

            rhs = Permutation()
            for gen in v_array:
                s = gen[0]
                e = gen[1]
                rhs = rhs*free_to_perm[s]**e

            assert lhs == rhs


def test_exponent_vector():

    Groups = [SymmetricGroup(3), SymmetricGroup(4), SymmetricGroup(9).sylow_subgroup(3),
         SymmetricGroup(9).sylow_subgroup(2), SymmetricGroup(8).sylow_subgroup(2)]

    for G in Groups:
        PcGroup = G.polycyclic_group()
        collector = PcGroup.collector

        pcgs = PcGroup.pcgs
        # free_group = collector.free_group

        for gen in G.generators:
            exp = collector.exponent_vector(gen)
            g = Permutation()
            for i in range(len(exp)):
                g = g*pcgs[i]**exp[i] if exp[i] else g
            assert g == gen


def test_induced_pcgs():
    G = [SymmetricGroup(9).sylow_subgroup(3), SymmetricGroup(20).sylow_subgroup(2), AlternatingGroup(4),
    DihedralGroup(4), DihedralGroup(10), DihedralGroup(9), SymmetricGroup(3), SymmetricGroup(4)]

    for g in G:
        PcGroup = g.polycyclic_group()
        collector = PcGroup.collector
        gens = list(g.generators)
        ipcgs = collector.induced_pcgs(gens)
        m = []
        for i in ipcgs:
            m.append(collector.exponent_vector(i))
        assert Matrix(m).is_upper

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_constructor_postprocessor.py ===
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import (Symbol, symbols)

from sympy.testing.pytest import XFAIL

class SymbolInMulOnce(Symbol):
    # Test class for a symbol that can only appear once in a `Mul` expression.
    pass


Basic._constructor_postprocessor_mapping[SymbolInMulOnce] = {
    "Mul": [lambda x: x],
    "Pow": [lambda x: x.base if isinstance(x.base, SymbolInMulOnce) else x],
    "Add": [lambda x: x],
}


def _postprocess_SymbolRemovesOtherSymbols(expr):
    args = tuple(i for i in expr.args if not isinstance(i, Symbol) or isinstance(i, SymbolRemovesOtherSymbols))
    if args == expr.args:
        return expr
    return Mul.fromiter(args)


class SymbolRemovesOtherSymbols(Symbol):
    # Test class for a symbol that removes other symbols in `Mul`.
    pass

Basic._constructor_postprocessor_mapping[SymbolRemovesOtherSymbols] = {
    "Mul": [_postprocess_SymbolRemovesOtherSymbols],
}

class SubclassSymbolInMulOnce(SymbolInMulOnce):
    pass

class SubclassSymbolRemovesOtherSymbols(SymbolRemovesOtherSymbols):
    pass


def test_constructor_postprocessors1():
    x = SymbolInMulOnce("x")
    y = SymbolInMulOnce("y")
    assert isinstance(3*x, Mul)
    assert (3*x).args == (3, x)
    assert x*x == x
    assert 3*x*x == 3*x
    assert 2*x*x + x == 3*x
    assert x**3*y*y == x*y
    assert x**5 + y*x**3 == x + x*y

    w = SymbolRemovesOtherSymbols("w")
    assert x*w == w
    assert (3*w).args == (3, w)
    assert set((w + x).args) == {x, w}

def test_constructor_postprocessors2():
    x = SubclassSymbolInMulOnce("x")
    y = SubclassSymbolInMulOnce("y")
    assert isinstance(3*x, Mul)
    assert (3*x).args == (3, x)
    assert x*x == x
    assert 3*x*x == 3*x
    assert 2*x*x + x == 3*x
    assert x**3*y*y == x*y
    assert x**5 + y*x**3 == x + x*y

    w = SubclassSymbolRemovesOtherSymbols("w")
    assert x*w == w
    assert (3*w).args == (3, w)
    assert set((w + x).args) == {x, w}


@XFAIL
def test_subexpression_postprocessors():
    # The postprocessors used to work with subexpressions, but the
    # functionality was removed. See #15948.
    a = symbols("a")
    x = SymbolInMulOnce("x")
    w = SymbolRemovesOtherSymbols("w")
    assert 3*a*w**2 == 3*w**2
    assert 3*a*x**3*w**2 == 3*w**2

    x = SubclassSymbolInMulOnce("x")
    w = SubclassSymbolRemovesOtherSymbols("w")
    assert 3*a*w**2 == 3*w**2
    assert 3*a*x**3*w**2 == 3*w**2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_gc.py ===
import gc

import weakref

import greenlet


from . import TestCase
from .leakcheck import fails_leakcheck
# These only work with greenlet gc support
# which is no longer optional.
assert greenlet.GREENLET_USE_GC

class GCTests(TestCase):
    def test_dead_circular_ref(self):
        o = weakref.ref(greenlet.greenlet(greenlet.getcurrent).switch())
        gc.collect()
        if o() is not None:
            import sys
            print("O IS NOT NONE.", sys.getrefcount(o()))
        self.assertIsNone(o())
        self.assertFalse(gc.garbage, gc.garbage)

    def test_circular_greenlet(self):
        class circular_greenlet(greenlet.greenlet):
            self = None
        o = circular_greenlet()
        o.self = o
        o = weakref.ref(o)
        gc.collect()
        self.assertIsNone(o())
        self.assertFalse(gc.garbage, gc.garbage)

    def test_inactive_ref(self):
        class inactive_greenlet(greenlet.greenlet):
            def __init__(self):
                greenlet.greenlet.__init__(self, run=self.run)

            def run(self):
                pass
        o = inactive_greenlet()
        o = weakref.ref(o)
        gc.collect()
        self.assertIsNone(o())
        self.assertFalse(gc.garbage, gc.garbage)

    @fails_leakcheck
    def test_finalizer_crash(self):
        # This test is designed to crash when active greenlets
        # are made garbage collectable, until the underlying
        # problem is resolved. How does it work:
        # - order of object creation is important
        # - array is created first, so it is moved to unreachable first
        # - we create a cycle between a greenlet and this array
        # - we create an object that participates in gc, is only
        #   referenced by a greenlet, and would corrupt gc lists
        #   on destruction, the easiest is to use an object with
        #   a finalizer
        # - because array is the first object in unreachable it is
        #   cleared first, which causes all references to greenlet
        #   to disappear and causes greenlet to be destroyed, but since
        #   it is still live it causes a switch during gc, which causes
        #   an object with finalizer to be destroyed, which causes stack
        #   corruption and then a crash

        class object_with_finalizer(object):
            def __del__(self):
                pass
        array = []
        parent = greenlet.getcurrent()
        def greenlet_body():
            greenlet.getcurrent().object = object_with_finalizer()
            try:
                parent.switch()
            except greenlet.GreenletExit:
                print("Got greenlet exit!")
            finally:
                del greenlet.getcurrent().object
        g = greenlet.greenlet(greenlet_body)
        g.array = array
        array.append(g)
        g.switch()
        del array
        del g
        greenlet.getcurrent()
        gc.collect()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\internals\test_api.py ===
"""
Tests for the pseudo-public API implemented in internals/api.py and exposed
in core.internals
"""

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core import internals
from pandas.core.internals import api


def test_internals_api():
    assert internals.make_block is api.make_block


def test_namespace():
    # SUBJECT TO CHANGE

    modules = [
        "blocks",
        "concat",
        "managers",
        "construction",
        "array_manager",
        "base",
        "api",
        "ops",
    ]
    expected = [
        "make_block",
        "DataManager",
        "ArrayManager",
        "BlockManager",
        "SingleDataManager",
        "SingleBlockManager",
        "SingleArrayManager",
        "concatenate_managers",
    ]

    result = [x for x in dir(internals) if not x.startswith("__")]
    assert set(result) == set(expected + modules)


@pytest.mark.parametrize(
    "name",
    [
        "NumericBlock",
        "ObjectBlock",
        "Block",
        "ExtensionBlock",
        "DatetimeTZBlock",
    ],
)
def test_deprecations(name):
    # GH#55139
    msg = f"{name} is deprecated.* Use public APIs instead"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        getattr(internals, name)

    if name not in ["NumericBlock", "ObjectBlock"]:
        # NumericBlock and ObjectBlock are not in the internals.api namespace
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            getattr(api, name)


def test_make_block_2d_with_dti():
    # GH#41168
    dti = pd.date_range("2012", periods=3, tz="UTC")
    blk = api.make_block(dti, placement=[0])

    assert blk.shape == (1, 3)
    assert blk.values.shape == (1, 3)


def test_create_block_manager_from_blocks_deprecated():
    # GH#33892
    # If they must, downstream packages should get this from internals.api,
    #  not internals.
    msg = (
        "create_block_manager_from_blocks is deprecated and will be "
        "removed in a future version. Use public APIs instead"
    )
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        internals.create_block_manager_from_blocks

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\excel\test_xlsxwriter.py ===
import contextlib

import pytest

from pandas.compat import is_platform_windows

from pandas import DataFrame
import pandas._testing as tm

from pandas.io.excel import ExcelWriter

xlsxwriter = pytest.importorskip("xlsxwriter")

if is_platform_windows():
    pytestmark = pytest.mark.single_cpu


@pytest.fixture
def ext():
    return ".xlsx"


def test_column_format(ext):
    # Test that column formats are applied to cells. Test for issue #9167.
    # Applicable to xlsxwriter only.
    openpyxl = pytest.importorskip("openpyxl")

    with tm.ensure_clean(ext) as path:
        frame = DataFrame({"A": [123456, 123456], "B": [123456, 123456]})

        with ExcelWriter(path) as writer:
            frame.to_excel(writer)

            # Add a number format to col B and ensure it is applied to cells.
            num_format = "#,##0"
            write_workbook = writer.book
            write_worksheet = write_workbook.worksheets()[0]
            col_format = write_workbook.add_format({"num_format": num_format})
            write_worksheet.set_column("B:B", None, col_format)

        with contextlib.closing(openpyxl.load_workbook(path)) as read_workbook:
            try:
                read_worksheet = read_workbook["Sheet1"]
            except TypeError:
                # compat
                read_worksheet = read_workbook.get_sheet_by_name(name="Sheet1")

        # Get the number format from the cell.
        try:
            cell = read_worksheet["B2"]
        except TypeError:
            # compat
            cell = read_worksheet.cell("B2")

        try:
            read_num_format = cell.number_format
        except AttributeError:
            read_num_format = cell.style.number_format._format_code

        assert read_num_format == num_format


def test_write_append_mode_raises(ext):
    msg = "Append mode is not supported with xlsxwriter!"

    with tm.ensure_clean(ext) as f:
        with pytest.raises(ValueError, match=msg):
            ExcelWriter(f, engine="xlsxwriter", mode="a")


@pytest.mark.parametrize("nan_inf_to_errors", [True, False])
def test_engine_kwargs(ext, nan_inf_to_errors):
    # GH 42286
    engine_kwargs = {"options": {"nan_inf_to_errors": nan_inf_to_errors}}
    with tm.ensure_clean(ext) as f:
        with ExcelWriter(f, engine="xlsxwriter", engine_kwargs=engine_kwargs) as writer:
            assert writer.book.nan_inf_to_errors == nan_inf_to_errors


def test_book_and_sheets_consistent(ext):
    # GH#45687 - Ensure sheets is updated if user modifies book
    with tm.ensure_clean(ext) as f:
        with ExcelWriter(f, engine="xlsxwriter") as writer:
            assert writer.sheets == {}
            sheet = writer.book.add_worksheet("test_name")
            assert writer.sheets == {"test_name": sheet}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\test_as_unit.py ===
import pytest

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime

from pandas import Timestamp


class TestTimestampAsUnit:
    def test_as_unit(self):
        ts = Timestamp("1970-01-01").as_unit("ns")
        assert ts.unit == "ns"

        assert ts.as_unit("ns") is ts

        res = ts.as_unit("us")
        assert res._value == ts._value // 1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_us.value

        rt = res.as_unit("ns")
        assert rt._value == ts._value
        assert rt._creso == ts._creso

        res = ts.as_unit("ms")
        assert res._value == ts._value // 1_000_000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value

        rt = res.as_unit("ns")
        assert rt._value == ts._value
        assert rt._creso == ts._creso

        res = ts.as_unit("s")
        assert res._value == ts._value // 1_000_000_000
        assert res._creso == NpyDatetimeUnit.NPY_FR_s.value

        rt = res.as_unit("ns")
        assert rt._value == ts._value
        assert rt._creso == ts._creso

    def test_as_unit_overflows(self):
        # microsecond that would be just out of bounds for nano
        us = 9223372800000000
        ts = Timestamp._from_value_and_reso(us, NpyDatetimeUnit.NPY_FR_us.value, None)

        msg = "Cannot cast 2262-04-12 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            ts.as_unit("ns")

        res = ts.as_unit("ms")
        assert res._value == us // 1000
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value

    def test_as_unit_rounding(self):
        ts = Timestamp(1_500_000)  # i.e. 1500 microseconds
        res = ts.as_unit("ms")

        expected = Timestamp(1_000_000)  # i.e. 1 millisecond
        assert res == expected

        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
        assert res._value == 1

        with pytest.raises(ValueError, match="Cannot losslessly convert units"):
            ts.as_unit("ms", round_ok=False)

    def test_as_unit_non_nano(self):
        # case where we are going neither to nor from nano
        ts = Timestamp("1970-01-02").as_unit("ms")
        assert ts.year == 1970
        assert ts.month == 1
        assert ts.day == 2
        assert ts.hour == ts.minute == ts.second == ts.microsecond == ts.nanosecond == 0

        res = ts.as_unit("s")
        assert res._value == 24 * 3600
        assert res.year == 1970
        assert res.month == 1
        assert res.day == 2
        assert (
            res.hour
            == res.minute
            == res.second
            == res.microsecond
            == res.nanosecond
            == 0
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_loads.py ===
from pytest import raises

from sympy import symbols
from sympy.physics.mechanics import (RigidBody, Particle, ReferenceFrame, Point,
                                     outer, dynamicsymbols, Force, Torque)
from sympy.physics.mechanics.loads import gravity, _parse_load


def test_force_default():
    N = ReferenceFrame('N')
    Po = Point('Po')
    f1 = Force(Po, N.x)
    assert f1.point == Po
    assert f1.force == N.x
    assert f1.__repr__() == 'Force(point=Po, force=N.x)'
    # Test tuple behaviour
    assert isinstance(f1, tuple)
    assert f1[0] == Po
    assert f1[1] == N.x
    assert f1 == (Po, N.x)
    assert f1 != (N.x, Po)
    assert f1 != (Po, N.x + N.y)
    assert f1 != (Point('Co'), N.x)
    # Test body as input
    P = Particle('P', Po)
    f2 = Force(P, N.x)
    assert f1 == f2


def test_torque_default():
    N = ReferenceFrame('N')
    f1 = Torque(N, N.x)
    assert f1.frame == N
    assert f1.torque == N.x
    assert f1.__repr__() == 'Torque(frame=N, torque=N.x)'
    # Test tuple behaviour
    assert isinstance(f1, tuple)
    assert f1[0] == N
    assert f1[1] == N.x
    assert f1 == (N, N.x)
    assert f1 != (N.x, N)
    assert f1 != (N, N.x + N.y)
    assert f1 != (ReferenceFrame('A'), N.x)
    # Test body as input
    rb = RigidBody('P', frame=N)
    f2 = Torque(rb, N.x)
    assert f1 == f2


def test_gravity():
    N = ReferenceFrame('N')
    m, M, g = symbols('m M g')
    F1, F2 = dynamicsymbols('F1 F2')
    po = Point('po')
    pa = Particle('pa', po, m)
    A = ReferenceFrame('A')
    P = Point('P')
    I = outer(A.x, A.x)
    B = RigidBody('B', P, A, M, (I, P))
    forceList = [(po, F1), (P, F2)]
    forceList.extend(gravity(g * N.y, pa, B))
    l = [(po, F1), (P, F2), (po, g * m * N.y), (P, g * M * N.y)]

    for i in range(len(l)):
        for j in range(len(l[i])):
            assert forceList[i][j] == l[i][j]


def test_parse_loads():
    N = ReferenceFrame('N')
    po = Point('po')
    assert _parse_load(Force(po, N.z)) == (po, N.z)
    assert _parse_load(Torque(N, N.x)) == (N, N.x)
    f1 = _parse_load((po, N.x))  # Test whether a force is recognized
    assert isinstance(f1, Force)
    assert f1 == Force(po, N.x)
    t1 = _parse_load((N, N.y))  # Test whether a torque is recognized
    assert isinstance(t1, Torque)
    assert t1 == Torque(N, N.y)
    # Bodies should be undetermined (even in case of a Particle)
    raises(ValueError, lambda: _parse_load((Particle('pa', po), N.x)))
    raises(ValueError, lambda: _parse_load((RigidBody('pa', po, N), N.x)))
    # Invalid tuple length
    raises(ValueError, lambda: _parse_load((po, N.x, po, N.x)))
    # Invalid type
    raises(TypeError, lambda: _parse_load([po, N.x]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\tests\test_prefixes.py ===
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.physics.units import Quantity, length, meter, W
from sympy.physics.units.prefixes import PREFIXES, Prefix, prefix_unit, kilo, \
    kibi
from sympy.physics.units.systems import SI

x = Symbol('x')


def test_prefix_operations():
    m = PREFIXES['m']
    k = PREFIXES['k']
    M = PREFIXES['M']

    dodeca = Prefix('dodeca', 'dd', 1, base=12)

    assert m * k is S.One
    assert m * W == W / 1000
    assert k * k == M
    assert 1 / m == k
    assert k / m == M

    assert dodeca * dodeca == 144
    assert 1 / dodeca == S.One / 12
    assert k / dodeca == S(1000) / 12
    assert dodeca / dodeca is S.One

    m = Quantity("fake_meter")
    SI.set_quantity_dimension(m, S.One)
    SI.set_quantity_scale_factor(m, S.One)

    assert dodeca * m == 12 * m
    assert dodeca / m == 12 / m

    expr1 = kilo * 3
    assert isinstance(expr1, Mul)
    assert expr1.args == (3, kilo)

    expr2 = kilo * x
    assert isinstance(expr2, Mul)
    assert expr2.args == (x, kilo)

    expr3 = kilo / 3
    assert isinstance(expr3, Mul)
    assert expr3.args == (Rational(1, 3), kilo)
    assert expr3.args == (S.One/3, kilo)

    expr4 = kilo / x
    assert isinstance(expr4, Mul)
    assert expr4.args == (1/x, kilo)


def test_prefix_unit():
    m = Quantity("fake_meter", abbrev="m")
    m.set_global_relative_scale_factor(1, meter)

    pref = {"m": PREFIXES["m"], "c": PREFIXES["c"], "d": PREFIXES["d"]}

    q1 = Quantity("millifake_meter", abbrev="mm")
    q2 = Quantity("centifake_meter", abbrev="cm")
    q3 = Quantity("decifake_meter", abbrev="dm")

    SI.set_quantity_dimension(q1, length)

    SI.set_quantity_scale_factor(q1, PREFIXES["m"])
    SI.set_quantity_scale_factor(q1, PREFIXES["c"])
    SI.set_quantity_scale_factor(q1, PREFIXES["d"])

    res = [q1, q2, q3]

    prefs = prefix_unit(m, pref)
    assert set(prefs) == set(res)
    assert {v.abbrev for v in prefs} == set(symbols("mm,cm,dm"))


def test_bases():
    assert kilo.base == 10
    assert kibi.base == 2


def test_repr():
    assert eval(repr(kilo)) == kilo
    assert eval(repr(kibi)) == kibi

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\tests\test_unitsystem.py ===
from sympy.physics.units import DimensionSystem, joule, second, ampere

from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.physics.units.definitions import c, kg, m, s
from sympy.physics.units.definitions.dimension_definitions import length, time
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.physics.units.util import convert_to


def test_definition():
    # want to test if the system can have several units of the same dimension
    dm = Quantity("dm")
    base = (m, s)
    # base_dim = (m.dimension, s.dimension)
    ms = UnitSystem(base, (c, dm), "MS", "MS system")
    ms.set_quantity_dimension(dm, length)
    ms.set_quantity_scale_factor(dm, Rational(1, 10))

    assert set(ms._base_units) == set(base)
    assert set(ms._units) == {m, s, c, dm}
    # assert ms._units == DimensionSystem._sort_dims(base + (velocity,))
    assert ms.name == "MS"
    assert ms.descr == "MS system"


def test_str_repr():
    assert str(UnitSystem((m, s), name="MS")) == "MS"
    assert str(UnitSystem((m, s))) == "UnitSystem((meter, second))"

    assert repr(UnitSystem((m, s))) == "<UnitSystem: (%s, %s)>" % (m, s)


def test_convert_to():
    A = Quantity("A")
    A.set_global_relative_scale_factor(S.One, ampere)

    Js = Quantity("Js")
    Js.set_global_relative_scale_factor(S.One, joule*second)

    mksa = UnitSystem((m, kg, s, A), (Js,))
    assert convert_to(Js, mksa._base_units) == m**2*kg*s**-1/1000


def test_extend():
    ms = UnitSystem((m, s), (c,))
    Js = Quantity("Js")
    Js.set_global_relative_scale_factor(1, joule*second)
    mks = ms.extend((kg,), (Js,))

    res = UnitSystem((m, s, kg), (c, Js))
    assert set(mks._base_units) == set(res._base_units)
    assert set(mks._units) == set(res._units)


def test_dim():
    dimsys = UnitSystem((m, kg, s), (c,))
    assert dimsys.dim == 3


def test_is_consistent():
    dimension_system = DimensionSystem([length, time])
    us = UnitSystem([m, s], dimension_system=dimension_system)
    assert us.is_consistent == True


def test_get_units_non_prefixed():
    from sympy.physics.units import volt, ohm
    unit_system = UnitSystem.get_unit_system("SI")
    units = unit_system.get_units_non_prefixed()
    for prefix in ["giga", "tera", "peta", "exa", "zetta", "yotta", "kilo", "hecto", "deca", "deci", "centi", "milli", "micro", "nano", "pico", "femto", "atto", "zepto", "yocto"]:
        for unit in units:
            assert isinstance(unit, Quantity), f"{unit} must be a Quantity, not {type(unit)}"
            assert not unit.is_prefixed, f"{unit} is marked as prefixed"
            assert not unit.is_physical_constant, f"{unit} is marked as physics constant"
            assert not unit.name.name.startswith(prefix), f"Unit {unit.name} has prefix {prefix}"
    assert volt in units
    assert ohm in units

def test_derived_units_must_exist_in_unit_system():
    for unit_system in UnitSystem._unit_systems.values():
        for preferred_unit in unit_system.derived_units.values():
            units = preferred_unit.atoms(Quantity)
            for unit in units:
                assert unit in unit_system._units, f"Unit {unit} is not in unit system {unit_system}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_cxx.py ===
from sympy.core.numbers import Float, Integer, Rational
from sympy.core.symbol import symbols
from sympy.functions import beta, Ei, zeta, Max, Min, sqrt, riemann_xi, frac
from sympy.printing.cxx import CXX98CodePrinter, CXX11CodePrinter, CXX17CodePrinter, cxxcode
from sympy.codegen.cfunctions import log1p


x, y, u, v = symbols('x y u v')


def test_CXX98CodePrinter():
    assert CXX98CodePrinter().doprint(Max(x, 3)) in ('std::max(x, 3)', 'std::max(3, x)')
    assert CXX98CodePrinter().doprint(Min(x, 3, sqrt(x))) == 'std::min(3, std::min(x, std::sqrt(x)))'
    cxx98printer = CXX98CodePrinter()
    assert cxx98printer.language == 'C++'
    assert cxx98printer.standard == 'C++98'
    assert 'template' in cxx98printer.reserved_words
    assert 'alignas' not in cxx98printer.reserved_words


def test_CXX11CodePrinter():
    assert CXX11CodePrinter().doprint(log1p(x)) == 'std::log1p(x)'

    cxx11printer = CXX11CodePrinter()
    assert cxx11printer.language == 'C++'
    assert cxx11printer.standard == 'C++11'
    assert 'operator' in cxx11printer.reserved_words
    assert 'noexcept' in cxx11printer.reserved_words
    assert 'concept' not in cxx11printer.reserved_words


def test_subclass_print_method():
    class MyPrinter(CXX11CodePrinter):
        def _print_log1p(self, expr):
            return 'my_library::log1p(%s)' % ', '.join(map(self._print, expr.args))

    assert MyPrinter().doprint(log1p(x)) == 'my_library::log1p(x)'


def test_subclass_print_method__ns():
    class MyPrinter(CXX11CodePrinter):
        _ns = 'my_library::'

    p = CXX11CodePrinter()
    myp = MyPrinter()

    assert p.doprint(log1p(x)) == 'std::log1p(x)'
    assert myp.doprint(log1p(x)) == 'my_library::log1p(x)'


def test_CXX17CodePrinter():
    assert CXX17CodePrinter().doprint(beta(x, y)) == 'std::beta(x, y)'
    assert CXX17CodePrinter().doprint(Ei(x)) == 'std::expint(x)'
    assert CXX17CodePrinter().doprint(zeta(x)) == 'std::riemann_zeta(x)'

    # Automatic rewrite
    assert CXX17CodePrinter().doprint(frac(x)) == '(x - std::floor(x))'
    assert CXX17CodePrinter().doprint(riemann_xi(x)) == '((1.0/2.0)*std::pow(M_PI, -1.0/2.0*x)*x*(x - 1)*std::tgamma((1.0/2.0)*x)*std::riemann_zeta(x))'


def test_cxxcode():
    assert sorted(cxxcode(sqrt(x)*.5).split('*')) == sorted(['0.5', 'std::sqrt(x)'])

def test_cxxcode_nested_minmax():
    assert cxxcode(Max(Min(x, y), Min(u, v))) \
        == 'std::max(std::min(u, v), std::min(x, y))'
    assert cxxcode(Min(Max(x, y), Max(u, v))) \
        == 'std::min(std::max(u, v), std::max(x, y))'

def test_subclass_Integer_Float():
    class MyPrinter(CXX17CodePrinter):
        def _print_Integer(self, arg):
            return 'bigInt("%s")' % super()._print_Integer(arg)

        def _print_Float(self, arg):
            rat = Rational(arg)
            return 'bigFloat(%s, %s)' % (
                self._print(Integer(rat.p)),
                self._print(Integer(rat.q))
            )

    p = MyPrinter()
    for i in range(13):
        assert p.doprint(i) == 'bigInt("%d")' % i
    assert p.doprint(Float(0.5)) == 'bigFloat(bigInt("1"), bigInt("2"))'
    assert p.doprint(x**-1.0) == 'bigFloat(bigInt("1"), bigInt("1"))/x'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_equals.py ===
import numpy as np

from pandas import (
    DataFrame,
    date_range,
)
import pandas._testing as tm


class TestEquals:
    def test_dataframe_not_equal(self):
        # see GH#28839
        df1 = DataFrame({"a": [1, 2], "b": ["s", "d"]})
        df2 = DataFrame({"a": ["s", "d"], "b": [1, 2]})
        assert df1.equals(df2) is False

    def test_equals_different_blocks(self, using_array_manager, using_infer_string):
        # GH#9330
        df0 = DataFrame({"A": ["x", "y"], "B": [1, 2], "C": ["w", "z"]})
        df1 = df0.reset_index()[["A", "B", "C"]]
        if not using_array_manager and not using_infer_string:
            # this assert verifies that the above operations have
            # induced a block rearrangement
            assert df0._mgr.blocks[0].dtype != df1._mgr.blocks[0].dtype

        # do the real tests
        tm.assert_frame_equal(df0, df1)
        assert df0.equals(df1)
        assert df1.equals(df0)

    def test_equals(self):
        # Add object dtype column with nans
        index = np.random.default_rng(2).random(10)
        df1 = DataFrame(
            np.random.default_rng(2).random(10), index=index, columns=["floats"]
        )
        df1["text"] = "the sky is so blue. we could use more chocolate.".split()
        df1["start"] = date_range("2000-1-1", periods=10, freq="min")
        df1["end"] = date_range("2000-1-1", periods=10, freq="D")
        df1["diff"] = df1["end"] - df1["start"]
        # Explicitly cast to object, to avoid implicit cast when setting np.nan
        df1["bool"] = (np.arange(10) % 3 == 0).astype(object)
        df1.loc[::2] = np.nan
        df2 = df1.copy()
        assert df1["text"].equals(df2["text"])
        assert df1["start"].equals(df2["start"])
        assert df1["end"].equals(df2["end"])
        assert df1["diff"].equals(df2["diff"])
        assert df1["bool"].equals(df2["bool"])
        assert df1.equals(df2)
        assert not df1.equals(object)

        # different dtype
        different = df1.copy()
        different["floats"] = different["floats"].astype("float32")
        assert not df1.equals(different)

        # different index
        different_index = -index
        different = df2.set_index(different_index)
        assert not df1.equals(different)

        # different columns
        different = df2.copy()
        different.columns = df2.columns[::-1]
        assert not df1.equals(different)

        # DatetimeIndex
        index = date_range("2000-1-1", periods=10, freq="min")
        df1 = df1.set_index(index)
        df2 = df1.copy()
        assert df1.equals(df2)

        # MultiIndex
        df3 = df1.set_index(["text"], append=True)
        df2 = df1.set_index(["text"], append=True)
        assert df3.equals(df2)

        df2 = df1.set_index(["floats"], append=True)
        assert not df3.equals(df2)

        # NaN in index
        df3 = df1.set_index(["floats"], append=True)
        df2 = df1.set_index(["floats"], append=True)
        assert df3.equals(df2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_index_as_string.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.fixture(params=[["inner"], ["inner", "outer"]])
def frame(request):
    levels = request.param
    df = pd.DataFrame(
        {
            "outer": ["a", "a", "a", "b", "b", "b"],
            "inner": [1, 2, 3, 1, 2, 3],
            "A": np.arange(6),
            "B": ["one", "one", "two", "two", "one", "one"],
        }
    )
    if levels:
        df = df.set_index(levels)

    return df


@pytest.fixture()
def series():
    df = pd.DataFrame(
        {
            "outer": ["a", "a", "a", "b", "b", "b"],
            "inner": [1, 2, 3, 1, 2, 3],
            "A": np.arange(6),
            "B": ["one", "one", "two", "two", "one", "one"],
        }
    )
    s = df.set_index(["outer", "inner", "B"])["A"]

    return s


@pytest.mark.parametrize(
    "key_strs,groupers",
    [
        ("inner", pd.Grouper(level="inner")),  # Index name
        (["inner"], [pd.Grouper(level="inner")]),  # List of index name
        (["B", "inner"], ["B", pd.Grouper(level="inner")]),  # Column and index
        (["inner", "B"], [pd.Grouper(level="inner"), "B"]),  # Index and column
    ],
)
def test_grouper_index_level_as_string(frame, key_strs, groupers):
    if "B" not in key_strs or "outer" in frame.columns:
        result = frame.groupby(key_strs).mean(numeric_only=True)
        expected = frame.groupby(groupers).mean(numeric_only=True)
    else:
        result = frame.groupby(key_strs).mean()
        expected = frame.groupby(groupers).mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "levels",
    [
        "inner",
        "outer",
        "B",
        ["inner"],
        ["outer"],
        ["B"],
        ["inner", "outer"],
        ["outer", "inner"],
        ["inner", "outer", "B"],
        ["B", "outer", "inner"],
    ],
)
def test_grouper_index_level_as_string_series(series, levels):
    # Compute expected result
    if isinstance(levels, list):
        groupers = [pd.Grouper(level=lv) for lv in levels]
    else:
        groupers = pd.Grouper(level=levels)

    expected = series.groupby(groupers).mean()

    # Compute and check result
    result = series.groupby(levels).mean()
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_tooltip.py ===
import numpy as np
import pytest

from pandas import DataFrame

pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler


@pytest.fixture
def df():
    return DataFrame(
        data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        columns=["A", "B", "C"],
        index=["x", "y", "z"],
    )


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0)


@pytest.mark.parametrize(
    "ttips",
    [
        DataFrame(  # Test basic reindex and ignoring blank
            data=[["Min", "Max"], [np.nan, ""]],
            columns=["A", "C"],
            index=["x", "y"],
        ),
        DataFrame(  # Test non-referenced columns, reversed col names, short index
            data=[["Max", "Min", "Bad-Col"]], columns=["C", "A", "D"], index=["x"]
        ),
    ],
)
def test_tooltip_render(ttips, styler):
    # GH 21266
    result = styler.set_tooltips(ttips).to_html()

    # test tooltip table level class
    assert "#T_ .pd-t {\n  visibility: hidden;\n" in result

    # test 'Min' tooltip added
    assert "#T_ #T__row0_col0:hover .pd-t {\n  visibility: visible;\n}" in result
    assert '#T_ #T__row0_col0 .pd-t::after {\n  content: "Min";\n}' in result
    assert 'class="data row0 col0" >0<span class="pd-t"></span></td>' in result

    # test 'Max' tooltip added
    assert "#T_ #T__row0_col2:hover .pd-t {\n  visibility: visible;\n}" in result
    assert '#T_ #T__row0_col2 .pd-t::after {\n  content: "Max";\n}' in result
    assert 'class="data row0 col2" >2<span class="pd-t"></span></td>' in result

    # test Nan, empty string and bad column ignored
    assert "#T_ #T__row1_col0:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row0_col1:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "#T_ #T__row1_col2:hover .pd-t {\n  visibility: visible;\n}" not in result
    assert "Bad-Col" not in result


def test_tooltip_ignored(styler):
    # GH 21266
    result = styler.to_html()  # no set_tooltips() creates no <span>
    assert '<style type="text/css">\n</style>' in result
    assert '<span class="pd-t"></span>' not in result


def test_tooltip_css_class(styler):
    # GH 21266
    result = styler.set_tooltips(
        DataFrame([["tooltip"]], index=["x"], columns=["A"]),
        css_class="other-class",
        props=[("color", "green")],
    ).to_html()
    assert "#T_ .other-class {\n  color: green;\n" in result
    assert '#T_ #T__row0_col0 .other-class::after {\n  content: "tooltip";\n' in result

    # GH 39563
    result = styler.set_tooltips(  # set_tooltips overwrites previous
        DataFrame([["tooltip"]], index=["x"], columns=["A"]),
        css_class="another-class",
        props="color:green;color:red;",
    ).to_html()
    assert "#T_ .another-class {\n  color: green;\n  color: red;\n}" in result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\tests\test_basis.py ===
from sympy.abc import x
from sympy.core import S
from sympy.core.numbers import AlgebraicNumber
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.basis import round_two
from sympy.testing.pytest import raises


def test_round_two():
    # Poly must be irreducible, and over ZZ or QQ:
    raises(ValueError, lambda: round_two(Poly(x ** 2 - 1)))
    raises(ValueError, lambda: round_two(Poly(x ** 2 + sqrt(2))))

    # Test on many fields:
    cases = (
        # A couple of cyclotomic fields:
        (cyclotomic_poly(5), DomainMatrix.eye(4, QQ), 125),
        (cyclotomic_poly(7), DomainMatrix.eye(6, QQ), -16807),
        # A couple of quadratic fields (one 1 mod 4, one 3 mod 4):
        (x ** 2 - 5, DM([[1, (1, 2)], [0, (1, 2)]], QQ), 5),
        (x ** 2 - 7, DM([[1, 0], [0, 1]], QQ), 28),
        # Dedekind's example of a field with 2 as essential disc divisor:
        (x ** 3 + x ** 2 - 2 * x + 8, DM([[1, 0, 0], [0, 1, 0], [0, (1, 2), (1, 2)]], QQ).transpose(), -503),
        # A bunch of cubics with various forms for F -- all of these require
        # second or third enlargements. (Five of them require a third, while the rest require just a second.)
        # F = 2^2
        (x**3 + 3 * x**2 - 4 * x + 4, DM([((1, 2), (1, 4), (1, 4)), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), -83),
        # F = 2^2 * 3
        (x**3 + 3 * x**2 + 3 * x - 3, DM([((1, 2), 0, (1, 2)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -108),
        # F = 2^3
        (x**3 + 5 * x**2 - x + 3, DM([((1, 4), 0, (3, 4)), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), -31),
        # F = 2^2 * 5
        (x**3 + 5 * x**2 - 5 * x - 5, DM([((1, 2), 0, (1, 2)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), 1300),
        # F = 3^2
        (x**3 + 3 * x**2 + 5, DM([((1, 3), (1, 3), (1, 3)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -135),
        # F = 3^3
        (x**3 + 6 * x**2 + 3 * x - 1, DM([((1, 3), (1, 3), (1, 3)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), 81),
        # F = 2^2 * 3^2
        (x**3 + 6 * x**2 + 4, DM([((1, 3), (2, 3), (1, 3)), (0, 1, 0), (0, 0, (1, 2))], QQ).transpose(), -108),
        # F = 2^3 * 7
        (x**3 + 7 * x**2 + 7 * x - 7, DM([((1, 4), 0, (3, 4)), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), 49),
        # F = 2^2 * 13
        (x**3 + 7 * x**2 - x + 5, DM([((1, 2), 0, (1, 2)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -2028),
        # F = 2^4
        (x**3 + 7 * x**2 - 5 * x + 5, DM([((1, 4), 0, (3, 4)), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), -140),
        # F = 5^2
        (x**3 + 4 * x**2 - 3 * x + 7, DM([((1, 5), (4, 5), (4, 5)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -175),
        # F = 7^2
        (x**3 + 8 * x**2 + 5 * x - 1, DM([((1, 7), (6, 7), (2, 7)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), 49),
        # F = 2 * 5 * 7
        (x**3 + 8 * x**2 - 2 * x + 6, DM([(1, 0, 0), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -14700),
        # F = 2^2 * 3 * 5
        (x**3 + 6 * x**2 - 3 * x + 8, DM([(1, 0, 0), (0, (1, 4), (1, 4)), (0, 0, 1)], QQ).transpose(), -675),
        # F = 2 * 3^2 * 7
        (x**3 + 9 * x**2 + 6 * x - 8, DM([(1, 0, 0), (0, (1, 2), (1, 2)), (0, 0, 1)], QQ).transpose(), 3969),
        # F = 2^2 * 3^2 * 7
        (x**3 + 15 * x**2 - 9 * x + 13, DM([((1, 6), (1, 3), (1, 6)), (0, 1, 0), (0, 0, 1)], QQ).transpose(), -5292),
        # Polynomial need not be monic
        (5*x**3 + 5*x**2 - 10 * x + 40, DM([[1, 0, 0], [0, 1, 0], [0, (1, 2), (1, 2)]], QQ).transpose(), -503),
        # Polynomial can have non-integer rational coeffs
        (QQ(5, 3)*x**3 + QQ(5, 3)*x**2 - QQ(10, 3)*x + QQ(40, 3), DM([[1, 0, 0], [0, 1, 0], [0, (1, 2), (1, 2)]], QQ).transpose(), -503),
    )
    for f, B_exp, d_exp in cases:
        K = QQ.alg_field_from_poly(f)
        B = K.maximal_order().QQ_matrix
        d = K.discriminant()
        assert d == d_exp
        # The computed basis need not equal the expected one, but their quotient
        # must be unimodular:
        assert (B.inv()*B_exp).det()**2 == 1


def test_AlgebraicField_integral_basis():
    alpha = AlgebraicNumber(sqrt(5), alias='alpha')
    k = QQ.algebraic_field(alpha)
    B0 = k.integral_basis()
    B1 = k.integral_basis(fmt='sympy')
    B2 = k.integral_basis(fmt='alg')
    assert B0 == [k([1]), k([S.Half, S.Half])]
    assert B1 == [1, S.Half + alpha/2]
    assert B2 == [k.ext.field_element([1]),
                  k.ext.field_element([S.Half, S.Half])]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_arraymethod.py ===
"""
This file tests the generic aspects of ArrayMethod.  At the time of writing
this is private API, but when added, public API may be added here.
"""

import types
from typing import Any

import pytest
from numpy._core._multiarray_umath import _get_castingimpl as get_castingimpl

import numpy as np


class TestResolveDescriptors:
    # Test mainly error paths of the resolve_descriptors function,
    # note that the `casting_unittests` tests exercise this non-error paths.

    # Casting implementations are the main/only current user:
    method = get_castingimpl(type(np.dtype("d")), type(np.dtype("f")))

    @pytest.mark.parametrize("args", [
        (True,),  # Not a tuple.
        ((None,)),  # Too few elements
        ((None, None, None),),  # Too many
        ((None, None),),  # Input dtype is None, which is invalid.
        ((np.dtype("d"), True),),  # Output dtype is not a dtype
        ((np.dtype("f"), None),),  # Input dtype does not match method
    ])
    def test_invalid_arguments(self, args):
        with pytest.raises(TypeError):
            self.method._resolve_descriptors(*args)


class TestSimpleStridedCall:
    # Test mainly error paths of the resolve_descriptors function,
    # note that the `casting_unittests` tests exercise this non-error paths.

    # Casting implementations are the main/only current user:
    method = get_castingimpl(type(np.dtype("d")), type(np.dtype("f")))

    @pytest.mark.parametrize(["args", "error"], [
        ((True,), TypeError),  # Not a tuple
        (((None,),), TypeError),  # Too few elements
        ((None, None), TypeError),  # Inputs are not arrays.
        (((None, None, None),), TypeError),  # Too many
        (((np.arange(3), np.arange(3)),), TypeError),  # Incorrect dtypes
        (((np.ones(3, dtype=">d"), np.ones(3, dtype="<f")),),
         TypeError),  # Does not support byte-swapping
        (((np.ones((2, 2), dtype="d"), np.ones((2, 2), dtype="f")),),
         ValueError),  # not 1-D
        (((np.ones(3, dtype="d"), np.ones(4, dtype="f")),),
          ValueError),  # different length
        (((np.frombuffer(b"\0x00" * 3 * 2, dtype="d"),
           np.frombuffer(b"\0x00" * 3, dtype="f")),),
         ValueError),  # output not writeable
    ])
    def test_invalid_arguments(self, args, error):
        # This is private API, which may be modified freely
        with pytest.raises(error):
            self.method._simple_strided_call(*args)


@pytest.mark.parametrize(
    "cls", [
        np.ndarray, np.recarray, np.char.chararray, np.matrix, np.memmap
    ]
)
class TestClassGetItem:
    def test_class_getitem(self, cls: type[np.ndarray]) -> None:
        """Test `ndarray.__class_getitem__`."""
        alias = cls[Any, Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    @pytest.mark.parametrize("arg_len", range(4))
    def test_subscript_tup(self, cls: type[np.ndarray], arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len in (1, 2):
            assert cls[arg_tup]
        else:
            match = f"Too {'few' if arg_len == 0 else 'many'} arguments"
            with pytest.raises(TypeError, match=match):
                cls[arg_tup]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\test_series_transform.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "args, kwargs, increment",
    [((), {}, 0), ((), {"a": 1}, 1), ((2, 3), {}, 32), ((1,), {"c": 2}, 201)],
)
def test_agg_args(args, kwargs, increment):
    # GH 43357
    def f(x, a=0, b=0, c=0):
        return x + a + 10 * b + 100 * c

    s = Series([1, 2])
    result = s.transform(f, 0, *args, **kwargs)
    expected = s + increment
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sqrt], ["sqrt"]),
        ([np.abs, np.sqrt], ["absolute", "sqrt"]),
        (np.array([np.sqrt]), ["sqrt"]),
        (np.array([np.abs, np.sqrt]), ["absolute", "sqrt"]),
    ],
)
def test_transform_listlike(string_series, ops, names):
    # GH 35964
    with np.errstate(all="ignore"):
        expected = concat([op(string_series) for op in ops], axis=1)
        expected.columns = names
        result = string_series.transform(ops)
        tm.assert_frame_equal(result, expected)


def test_transform_listlike_func_with_args():
    # GH 50624

    s = Series([1, 2, 3])

    def foo1(x, a=1, c=0):
        return x + a + c

    def foo2(x, b=2, c=0):
        return x + b + c

    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        s.transform([foo1, foo2], 0, 3, b=3, c=4)

    result = s.transform([foo1, foo2], 0, 3, c=4)
    expected = DataFrame({"foo1": [8, 9, 10], "foo2": [8, 9, 10]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("box", [dict, Series])
def test_transform_dictlike(string_series, box):
    # GH 35964
    with np.errstate(all="ignore"):
        expected = concat([np.sqrt(string_series), np.abs(string_series)], axis=1)
    expected.columns = ["foo", "bar"]
    result = string_series.transform(box({"foo": np.sqrt, "bar": np.abs}))
    tm.assert_frame_equal(result, expected)


def test_transform_dictlike_mixed():
    # GH 40018 - mix of lists and non-lists in values of a dictionary
    df = Series([1, 4])
    result = df.transform({"b": ["sqrt", "abs"], "c": "sqrt"})
    expected = DataFrame(
        [[1.0, 1, 1.0], [2.0, 4, 2.0]],
        columns=MultiIndex([("b", "c"), ("sqrt", "abs")], [(0, 0, 1), (0, 1, 0)]),
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_assign.py ===
import pytest

from pandas import DataFrame
import pandas._testing as tm


class TestAssign:
    def test_assign(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        original = df.copy()
        result = df.assign(C=df.B / df.A)
        expected = df.copy()
        expected["C"] = [4, 2.5, 2]
        tm.assert_frame_equal(result, expected)

        # lambda syntax
        result = df.assign(C=lambda x: x.B / x.A)
        tm.assert_frame_equal(result, expected)

        # original is unmodified
        tm.assert_frame_equal(df, original)

        # Non-Series array-like
        result = df.assign(C=[4, 2.5, 2])
        tm.assert_frame_equal(result, expected)
        # original is unmodified
        tm.assert_frame_equal(df, original)

        result = df.assign(B=df.B / df.A)
        expected = expected.drop("B", axis=1).rename(columns={"C": "B"})
        tm.assert_frame_equal(result, expected)

        # overwrite
        result = df.assign(A=df.A + df.B)
        expected = df.copy()
        expected["A"] = [5, 7, 9]
        tm.assert_frame_equal(result, expected)

        # lambda
        result = df.assign(A=lambda x: x.A + x.B)
        tm.assert_frame_equal(result, expected)

    def test_assign_multiple(self):
        df = DataFrame([[1, 4], [2, 5], [3, 6]], columns=["A", "B"])
        result = df.assign(C=[7, 8, 9], D=df.A, E=lambda x: x.B)
        expected = DataFrame(
            [[1, 4, 7, 1, 4], [2, 5, 8, 2, 5], [3, 6, 9, 3, 6]], columns=list("ABCDE")
        )
        tm.assert_frame_equal(result, expected)

    def test_assign_order(self):
        # GH 9818
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        result = df.assign(D=df.A + df.B, C=df.A - df.B)

        expected = DataFrame([[1, 2, 3, -1], [3, 4, 7, -1]], columns=list("ABDC"))
        tm.assert_frame_equal(result, expected)
        result = df.assign(C=df.A - df.B, D=df.A + df.B)

        expected = DataFrame([[1, 2, -1, 3], [3, 4, -1, 7]], columns=list("ABCD"))

        tm.assert_frame_equal(result, expected)

    def test_assign_bad(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # non-keyword argument
        msg = r"assign\(\) takes 1 positional argument but 2 were given"
        with pytest.raises(TypeError, match=msg):
            df.assign(lambda x: x.A)
        msg = "'DataFrame' object has no attribute 'C'"
        with pytest.raises(AttributeError, match=msg):
            df.assign(C=df.A, D=df.A + df.C)

    def test_assign_dependent(self):
        df = DataFrame({"A": [1, 2], "B": [3, 4]})

        result = df.assign(C=df.A, D=lambda x: x["A"] + x["C"])
        expected = DataFrame([[1, 3, 1, 2], [2, 4, 2, 4]], columns=list("ABCD"))
        tm.assert_frame_equal(result, expected)

        result = df.assign(C=lambda df: df.A, D=lambda df: df["A"] + df["C"])
        expected = DataFrame([[1, 3, 1, 2], [2, 4, 2, 4]], columns=list("ABCD"))
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_argsort.py ===
import numpy as np
import pytest

from pandas import (
    Series,
    Timestamp,
    isna,
)
import pandas._testing as tm


class TestSeriesArgsort:
    def test_argsort_axis(self):
        # GH#54257
        ser = Series(range(3))

        msg = "No axis named 2 for object type Series"
        with pytest.raises(ValueError, match=msg):
            ser.argsort(axis=2)

    def test_argsort_numpy(self, datetime_series):
        ser = datetime_series

        res = np.argsort(ser).values
        expected = np.argsort(np.array(ser))
        tm.assert_numpy_array_equal(res, expected)

        # with missing values
        ts = ser.copy()
        ts[::2] = np.nan

        msg = "The behavior of Series.argsort in the presence of NA values"
        with tm.assert_produces_warning(
            FutureWarning, match=msg, check_stacklevel=False
        ):
            result = np.argsort(ts)[1::2]
        expected = np.argsort(np.array(ts.dropna()))

        tm.assert_numpy_array_equal(result.values, expected)

    def test_argsort(self, datetime_series):
        argsorted = datetime_series.argsort()
        assert issubclass(argsorted.dtype.type, np.integer)

    def test_argsort_dt64(self, unit):
        # GH#2967 (introduced bug in 0.11-dev I think)
        ser = Series(
            [Timestamp(f"201301{i:02d}") for i in range(1, 6)], dtype=f"M8[{unit}]"
        )
        assert ser.dtype == f"datetime64[{unit}]"
        shifted = ser.shift(-1)
        assert shifted.dtype == f"datetime64[{unit}]"
        assert isna(shifted[4])

        result = ser.argsort()
        expected = Series(range(5), dtype=np.intp)
        tm.assert_series_equal(result, expected)

        msg = "The behavior of Series.argsort in the presence of NA values"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = shifted.argsort()
        expected = Series(list(range(4)) + [-1], dtype=np.intp)
        tm.assert_series_equal(result, expected)

    def test_argsort_stable(self):
        ser = Series(np.random.default_rng(2).integers(0, 100, size=10000))
        mindexer = ser.argsort(kind="mergesort")
        qindexer = ser.argsort()

        mexpected = np.argsort(ser.values, kind="mergesort")
        qexpected = np.argsort(ser.values, kind="quicksort")

        tm.assert_series_equal(mindexer.astype(np.intp), Series(mexpected))
        tm.assert_series_equal(qindexer.astype(np.intp), Series(qexpected))
        msg = (
            r"ndarray Expected type <class 'numpy\.ndarray'>, "
            r"found <class 'pandas\.core\.series\.Series'> instead"
        )
        with pytest.raises(AssertionError, match=msg):
            tm.assert_numpy_array_equal(qindexer, mindexer)

    def test_argsort_preserve_name(self, datetime_series):
        result = datetime_series.argsort()
        assert result.name == datetime_series.name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_validate_args_and_kwargs.py ===
import pytest

from pandas.util._validators import validate_args_and_kwargs


@pytest.fixture
def _fname():
    return "func"


def test_invalid_total_length_max_length_one(_fname):
    compat_args = ("foo",)
    kwargs = {"foo": "FOO"}
    args = ("FoO", "BaZ")

    min_fname_arg_count = 0
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(kwargs) + len(args) + min_fname_arg_count

    msg = (
        rf"{_fname}\(\) takes at most {max_length} "
        rf"argument \({actual_length} given\)"
    )

    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)


def test_invalid_total_length_max_length_multiple(_fname):
    compat_args = ("foo", "bar", "baz")
    kwargs = {"foo": "FOO", "bar": "BAR"}
    args = ("FoO", "BaZ")

    min_fname_arg_count = 2
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(kwargs) + len(args) + min_fname_arg_count

    msg = (
        rf"{_fname}\(\) takes at most {max_length} "
        rf"arguments \({actual_length} given\)"
    )

    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)


@pytest.mark.parametrize("args,kwargs", [((), {"foo": -5, "bar": 2}), ((-5, 2), {})])
def test_missing_args_or_kwargs(args, kwargs, _fname):
    bad_arg = "bar"
    min_fname_arg_count = 2

    compat_args = {"foo": -5, bad_arg: 1}

    msg = (
        rf"the '{bad_arg}' parameter is not supported "
        rf"in the pandas implementation of {_fname}\(\)"
    )

    with pytest.raises(ValueError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)


def test_duplicate_argument(_fname):
    min_fname_arg_count = 2

    compat_args = {"foo": None, "bar": None, "baz": None}
    kwargs = {"foo": None, "bar": None}
    args = (None,)  # duplicate value for "foo"

    msg = rf"{_fname}\(\) got multiple values for keyword argument 'foo'"

    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)


def test_validation(_fname):
    # No exceptions should be raised.
    compat_args = {"foo": 1, "bar": None, "baz": -2}
    kwargs = {"baz": -2}

    args = (1, None)
    min_fname_arg_count = 2

    validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\tests\test_physics_matrices.py ===
from sympy.physics.matrices import msigma, mgamma, minkowski_tensor, pat_matrix, mdft
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.testing.pytest import warns_deprecated_sympy


def test_parallel_axis_theorem():
    # This tests the parallel axis theorem matrix by comparing to test
    # matrices.

    # First case, 1 in all directions.
    mat1 = Matrix(((2, -1, -1), (-1, 2, -1), (-1, -1, 2)))
    assert pat_matrix(1, 1, 1, 1) == mat1
    assert pat_matrix(2, 1, 1, 1) == 2*mat1

    # Second case, 1 in x, 0 in all others
    mat2 = Matrix(((0, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert pat_matrix(1, 1, 0, 0) == mat2
    assert pat_matrix(2, 1, 0, 0) == 2*mat2

    # Third case, 1 in y, 0 in all others
    mat3 = Matrix(((1, 0, 0), (0, 0, 0), (0, 0, 1)))
    assert pat_matrix(1, 0, 1, 0) == mat3
    assert pat_matrix(2, 0, 1, 0) == 2*mat3

    # Fourth case, 1 in z, 0 in all others
    mat4 = Matrix(((1, 0, 0), (0, 1, 0), (0, 0, 0)))
    assert pat_matrix(1, 0, 0, 1) == mat4
    assert pat_matrix(2, 0, 0, 1) == 2*mat4


def test_Pauli():
    #this and the following test are testing both Pauli and Dirac matrices
    #and also that the general Matrix class works correctly in a real world
    #situation
    sigma1 = msigma(1)
    sigma2 = msigma(2)
    sigma3 = msigma(3)

    assert sigma1 == sigma1
    assert sigma1 != sigma2

    # sigma*I -> I*sigma    (see #354)
    assert sigma1*sigma2 == sigma3*I
    assert sigma3*sigma1 == sigma2*I
    assert sigma2*sigma3 == sigma1*I

    assert sigma1*sigma1 == eye(2)
    assert sigma2*sigma2 == eye(2)
    assert sigma3*sigma3 == eye(2)

    assert sigma1*2*sigma1 == 2*eye(2)
    assert sigma1*sigma3*sigma1 == -sigma3


def test_Dirac():
    gamma0 = mgamma(0)
    gamma1 = mgamma(1)
    gamma2 = mgamma(2)
    gamma3 = mgamma(3)
    gamma5 = mgamma(5)

    # gamma*I -> I*gamma    (see #354)
    assert gamma5 == gamma0 * gamma1 * gamma2 * gamma3 * I
    assert gamma1 * gamma2 + gamma2 * gamma1 == zeros(4)
    assert gamma0 * gamma0 == eye(4) * minkowski_tensor[0, 0]
    assert gamma2 * gamma2 != eye(4) * minkowski_tensor[0, 0]
    assert gamma2 * gamma2 == eye(4) * minkowski_tensor[2, 2]

    assert mgamma(5, True) == \
        mgamma(0, True)*mgamma(1, True)*mgamma(2, True)*mgamma(3, True)*I

def test_mdft():
    with warns_deprecated_sympy():
        assert mdft(1) == Matrix([[1]])
    with warns_deprecated_sympy():
        assert mdft(2) == 1/sqrt(2)*Matrix([[1,1],[1,-1]])
    with warns_deprecated_sympy():
        assert mdft(4) == Matrix([[S.Half,  S.Half,  S.Half, S.Half],
                                  [S.Half, -I/2, Rational(-1,2),  I/2],
                                  [S.Half, Rational(-1,2),  S.Half, Rational(-1,2)],
                                  [S.Half,  I/2, Rational(-1,2), -I/2]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\tests\test_traverse.py ===
from sympy.strategies.traverse import (
    top_down, bottom_up, sall, top_down_once, bottom_up_once, basic_fns)
from sympy.strategies.rl import rebuild
from sympy.strategies.util import expr_fns
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Str, Symbol
from sympy.abc import x, y, z


def zero_symbols(expression):
    return S.Zero if isinstance(expression, Symbol) else expression


def test_sall():
    zero_onelevel = sall(zero_symbols)

    assert zero_onelevel(Basic(x, y, Basic(x, z))) == \
        Basic(S(0), S(0), Basic(x, z))


def test_bottom_up():
    _test_global_traversal(bottom_up)
    _test_stop_on_non_basics(bottom_up)


def test_top_down():
    _test_global_traversal(top_down)
    _test_stop_on_non_basics(top_down)


def _test_global_traversal(trav):
    zero_all_symbols = trav(zero_symbols)

    assert zero_all_symbols(Basic(x, y, Basic(x, z))) == \
        Basic(S(0), S(0), Basic(S(0), S(0)))


def _test_stop_on_non_basics(trav):
    def add_one_if_can(expr):
        try:
            return expr + 1
        except TypeError:
            return expr

    expr = Basic(S(1), Str('a'), Basic(S(2), Str('b')))
    expected = Basic(S(2), Str('a'), Basic(S(3), Str('b')))
    rl = trav(add_one_if_can)

    assert rl(expr) == expected


class Basic2(Basic):
    pass


def rl(x):
    if x.args and not isinstance(x.args[0], Integer):
        return Basic2(*x.args)
    return x


def test_top_down_once():
    top_rl = top_down_once(rl)

    assert top_rl(Basic(S(1.0), S(2.0), Basic(S(3), S(4)))) == \
        Basic2(S(1.0), S(2.0), Basic(S(3), S(4)))


def test_bottom_up_once():
    bottom_rl = bottom_up_once(rl)

    assert bottom_rl(Basic(S(1), S(2), Basic(S(3.0), S(4.0)))) == \
        Basic(S(1), S(2), Basic2(S(3.0), S(4.0)))


def test_expr_fns():
    expr = x + y**3
    e = bottom_up(lambda v: v + 1, expr_fns)(expr)
    b = bottom_up(lambda v: Basic.__new__(Add, v, S(1)), basic_fns)(expr)

    assert rebuild(b) == e

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_modules.py ===
import textwrap

import pytest

from numpy.testing import IS_PYPY

from . import util


@pytest.mark.slow
class TestModuleFilterPublicEntities(util.F2PyTest):
    sources = [
        util.getpath(
            "tests", "src", "modules", "gh26920",
            "two_mods_with_one_public_routine.f90"
        )
    ]
    # we filter the only public function mod2
    only = ["mod1_func1", ]

    def test_gh26920(self):
        # if it compiles and can be loaded, things are fine
        pass


@pytest.mark.slow
class TestModuleWithoutPublicEntities(util.F2PyTest):
    sources = [
        util.getpath(
            "tests", "src", "modules", "gh26920",
            "two_mods_with_no_public_entities.f90"
        )
    ]
    only = ["mod1_func1", ]

    def test_gh26920(self):
        # if it compiles and can be loaded, things are fine
        pass


@pytest.mark.slow
class TestModuleDocString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "modules", "module_data_docstring.f90")]

    @pytest.mark.xfail(IS_PYPY, reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_module_docstring(self):
        assert self.module.mod.__doc__ == textwrap.dedent(
            """\
                     i : 'i'-scalar
                     x : 'i'-array(4)
                     a : 'f'-array(2,3)
                     b : 'f'-array(-1,-1), not allocated\x00
                     foo()\n
                     Wrapper for ``foo``.\n\n"""
        )


@pytest.mark.slow
class TestModuleAndSubroutine(util.F2PyTest):
    module_name = "example"
    sources = [
        util.getpath("tests", "src", "modules", "gh25337", "data.f90"),
        util.getpath("tests", "src", "modules", "gh25337", "use_data.f90"),
    ]

    def test_gh25337(self):
        self.module.data.set_shift(3)
        assert "data" in dir(self.module)


@pytest.mark.slow
class TestUsedModule(util.F2PyTest):
    module_name = "fmath"
    sources = [
        util.getpath("tests", "src", "modules", "use_modules.f90"),
    ]

    def test_gh25867(self):
        compiled_mods = [x for x in dir(self.module) if "__" not in x]
        assert "useops" in compiled_mods
        assert self.module.useops.sum_and_double(3, 7) == 20
        assert "mathops" in compiled_mods
        assert self.module.mathops.add(3, 7) == 10

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_all_methods.py ===
"""
Tests that apply to all groupby operation methods.

The only tests that should appear here are those that use the `groupby_func` fixture.
Even if it does use that fixture, prefer a more specific test file if it available
such as:

 - test_categorical
 - test_groupby_dropna
 - test_groupby_subclass
 - test_raises
"""

import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


def test_multiindex_group_all_columns_when_empty(groupby_func):
    # GH 32464
    df = DataFrame({"a": [], "b": [], "c": []}).set_index(["a", "b", "c"])
    gb = df.groupby(["a", "b", "c"], group_keys=False)
    method = getattr(gb, groupby_func)
    args = get_groupby_method_args(groupby_func, df)

    warn = FutureWarning if groupby_func == "fillna" else None
    warn_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=warn_msg):
        result = method(*args).index
    expected = df.index
    tm.assert_index_equal(result, expected)


def test_duplicate_columns(request, groupby_func, as_index):
    # GH#50806
    if groupby_func == "corrwith":
        msg = "GH#50845 - corrwith fails when there are duplicate columns"
        request.applymarker(pytest.mark.xfail(reason=msg))
    df = DataFrame([[1, 3, 6], [1, 4, 7], [2, 5, 8]], columns=list("abb"))
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby("a", as_index=as_index)
    warn = FutureWarning if groupby_func == "fillna" else None
    warn_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=warn_msg):
        result = getattr(gb, groupby_func)(*args)

    expected_df = df.set_axis(["a", "b", "c"], axis=1)
    expected_args = get_groupby_method_args(groupby_func, expected_df)
    expected_gb = expected_df.groupby("a", as_index=as_index)
    warn = FutureWarning if groupby_func == "fillna" else None
    warn_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=warn_msg):
        expected = getattr(expected_gb, groupby_func)(*expected_args)
    if groupby_func not in ("size", "ngroup", "cumcount"):
        expected = expected.rename(columns={"c": "b"})
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index(["a", "a"], name="foo"),
        pd.MultiIndex.from_tuples((("a", "a"), ("a", "a")), names=["foo", "bar"]),
    ],
)
def test_dup_labels_output_shape(groupby_func, idx):
    if groupby_func in {"size", "ngroup", "cumcount"}:
        pytest.skip(f"Not applicable for {groupby_func}")

    df = DataFrame([[1, 1]], columns=idx)
    grp_by = df.groupby([0])

    args = get_groupby_method_args(groupby_func, df)
    warn = FutureWarning if groupby_func == "fillna" else None
    warn_msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=warn_msg):
        result = getattr(grp_by, groupby_func)(*args)

    assert result.shape == (1, 2)
    tm.assert_index_equal(result.columns, idx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_repeat.py ===
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestRepeat:
    def test_repeat_range(self, tz_naive_fixture):
        rng = date_range("1/1/2000", "1/1/2001")

        result = rng.repeat(5)
        assert result.freq is None
        assert len(result) == 5 * len(rng)

    def test_repeat_range2(self, tz_naive_fixture, unit):
        tz = tz_naive_fixture
        index = date_range("2001-01-01", periods=2, freq="D", tz=tz, unit=unit)
        exp = DatetimeIndex(
            ["2001-01-01", "2001-01-01", "2001-01-02", "2001-01-02"], tz=tz
        ).as_unit(unit)
        for res in [index.repeat(2), np.repeat(index, 2)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

    def test_repeat_range3(self, tz_naive_fixture, unit):
        tz = tz_naive_fixture
        index = date_range("2001-01-01", periods=2, freq="2D", tz=tz, unit=unit)
        exp = DatetimeIndex(
            ["2001-01-01", "2001-01-01", "2001-01-03", "2001-01-03"], tz=tz
        ).as_unit(unit)
        for res in [index.repeat(2), np.repeat(index, 2)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

    def test_repeat_range4(self, tz_naive_fixture, unit):
        tz = tz_naive_fixture
        index = DatetimeIndex(["2001-01-01", "NaT", "2003-01-01"], tz=tz).as_unit(unit)
        exp = DatetimeIndex(
            [
                "2001-01-01",
                "2001-01-01",
                "2001-01-01",
                "NaT",
                "NaT",
                "NaT",
                "2003-01-01",
                "2003-01-01",
                "2003-01-01",
            ],
            tz=tz,
        ).as_unit(unit)
        for res in [index.repeat(3), np.repeat(index, 3)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

    def test_repeat(self, tz_naive_fixture, unit):
        tz = tz_naive_fixture
        reps = 2
        msg = "the 'axis' parameter is not supported"

        rng = date_range(start="2016-01-01", periods=2, freq="30Min", tz=tz, unit=unit)

        expected_rng = DatetimeIndex(
            [
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:00:00", tz=tz),
                Timestamp("2016-01-01 00:30:00", tz=tz),
                Timestamp("2016-01-01 00:30:00", tz=tz),
            ]
        ).as_unit(unit)

        res = rng.repeat(reps)
        tm.assert_index_equal(res, expected_rng)
        assert res.freq is None

        tm.assert_index_equal(np.repeat(rng, reps), expected_rng)
        with pytest.raises(ValueError, match=msg):
            np.repeat(rng, reps, axis=1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\continuum_mechanics\tests\test_cable.py ===
from sympy.physics.continuum_mechanics.cable import Cable
from sympy.core.symbol import Symbol


def test_cable():
    c = Cable(('A', 0, 10), ('B', 10, 10))
    assert c.supports == {'A': [0, 10], 'B': [10, 10]}
    assert c.left_support == [0, 10]
    assert c.right_support == [10, 10]
    assert c.loads == {'distributed': {}, 'point_load': {}}
    assert c.loads_position == {}
    assert c.length == 0
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    # tests for change_support method
    c.change_support('A', ('C', 12, 3))
    assert c.supports == {'B': [10, 10], 'C': [12, 3]}
    assert c.left_support == [10, 10]
    assert c.right_support == [12, 3]
    assert c.reaction_loads == {Symbol("R_B_x"): 0, Symbol("R_B_y"): 0, Symbol("R_C_x"): 0, Symbol("R_C_y"): 0}

    c.change_support('C', ('A', 0, 10))

    # tests for apply_load method for point loads
    c.apply_load(-1, ('X', 2, 5, 3, 30))
    c.apply_load(-1, ('Y', 5, 8, 5, 60))
    assert c.loads == {'distributed': {}, 'point_load': {'X': [3, 30], 'Y': [5, 60]}}
    assert c.loads_position == {'X': [2, 5], 'Y': [5, 8]}
    assert c.length == 0
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    # tests for remove_loads method
    c.remove_loads('X')
    assert c.loads == {'distributed': {}, 'point_load': {'Y': [5, 60]}}
    assert c.loads_position == {'Y': [5, 8]}
    assert c.length == 0
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    c.remove_loads('Y')

    #tests for apply_load method for distributed load
    c.apply_load(0, ('Z', 9))
    assert c.loads == {'distributed': {'Z': 9}, 'point_load': {}}
    assert c.loads_position == {}
    assert c.length == 0
    assert c.reaction_loads == {Symbol("R_A_x"): 0, Symbol("R_A_y"): 0, Symbol("R_B_x"): 0, Symbol("R_B_y"): 0}

    # tests for apply_length method
    c.apply_length(20)
    assert c.length == 20

    del c
    # tests for solve method
    # for point loads
    c = Cable(("A", 0, 10), ("B", 5.5, 8))
    c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
    c.apply_load(-1, ('X', 4, 6, 8, 270))
    c.solve()
    #assert c.tension == {Symbol("Z_X"): 4.79150773600774, Symbol("X_B"): 6.78571428571429, Symbol("A_Z"): 6.89488895397307}
    assert abs(c.tension[Symbol("A_Z")] - 6.89488895397307) < 10e-12
    assert abs(c.tension[Symbol("Z_X")] - 4.79150773600774) < 10e-12
    assert abs(c.tension[Symbol("X_B")] - 6.78571428571429) < 10e-12
    #assert c.reaction_loads == {Symbol("R_A_x"): -4.06504065040650, Symbol("R_A_y"): 5.56910569105691, Symbol("R_B_x"): 4.06504065040650, Symbol("R_B_y"): 5.43089430894309}
    assert abs(c.reaction_loads[Symbol("R_A_x")] + 4.06504065040650) < 10e-12
    assert abs(c.reaction_loads[Symbol("R_A_y")] - 5.56910569105691) < 10e-12
    assert abs(c.reaction_loads[Symbol("R_B_x")] - 4.06504065040650) < 10e-12
    assert abs(c.reaction_loads[Symbol("R_B_y")] - 5.43089430894309) < 10e-12
    assert abs(c.length - 8.25609584845190) < 10e-12

    del c
    # tests for solve method
    # for distributed loads
    c=Cable(("A", 0, 40),("B", 100, 20))
    c.apply_load(0, ("X", 850))
    c.solve(58.58, 0)

    # assert c.tension['distributed'] == 36456.8485*sqrt(0.000543529004799705*(X + 0.00135624381275735)**2 + 1)
    assert abs(c.tension_at(0) - 61717.4130533677) < 10e-11
    assert abs(c.tension_at(40) - 39738.0809048449) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_A_x")] - 36465.0000000000) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_A_y")] + 49793.0000000000) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_B_x")] - 44399.9537590861) < 10e-11
    assert abs(c.reaction_loads[Symbol("R_B_y")] - 42868.2071025955 ) < 10e-11

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_subclass.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)


class TestSeriesSubclassing:
    @pytest.mark.parametrize(
        "idx_method, indexer, exp_data, exp_idx",
        [
            ["loc", ["a", "b"], [1, 2], "ab"],
            ["iloc", [2, 3], [3, 4], "cd"],
        ],
    )
    def test_indexing_sliced(self, idx_method, indexer, exp_data, exp_idx):
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list("abcd"))
        res = getattr(s, idx_method)[indexer]
        exp = tm.SubclassedSeries(exp_data, index=list(exp_idx))
        tm.assert_series_equal(res, exp)

    def test_to_frame(self):
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list("abcd"), name="xxx")
        res = s.to_frame()
        exp = tm.SubclassedDataFrame({"xxx": [1, 2, 3, 4]}, index=list("abcd"))
        tm.assert_frame_equal(res, exp)

    def test_subclass_unstack(self):
        # GH 15564
        s = tm.SubclassedSeries([1, 2, 3, 4], index=[list("aabb"), list("xyxy")])

        res = s.unstack()
        exp = tm.SubclassedDataFrame({"x": [1, 3], "y": [2, 4]}, index=["a", "b"])

        tm.assert_frame_equal(res, exp)

    def test_subclass_empty_repr(self):
        sub_series = tm.SubclassedSeries()
        assert "SubclassedSeries" in repr(sub_series)

    def test_asof(self):
        N = 3
        rng = pd.date_range("1/1/1990", periods=N, freq="53s")
        s = tm.SubclassedSeries({"A": [np.nan, np.nan, np.nan]}, index=rng)

        result = s.asof(rng[-2:])
        assert isinstance(result, tm.SubclassedSeries)

    def test_explode(self):
        s = tm.SubclassedSeries([[1, 2, 3], "foo", [], [3, 4]])
        result = s.explode()
        assert isinstance(result, tm.SubclassedSeries)

    def test_equals(self):
        # https://github.com/pandas-dev/pandas/pull/34402
        # allow subclass in both directions
        s1 = pd.Series([1, 2, 3])
        s2 = tm.SubclassedSeries([1, 2, 3])
        assert s1.equals(s2)
        assert s2.equals(s1)


class SubclassedSeries(pd.Series):
    @property
    def _constructor(self):
        def _new(*args, **kwargs):
            # some constructor logic that accesses the Series' name
            if self.name == "test":
                return pd.Series(*args, **kwargs)
            return SubclassedSeries(*args, **kwargs)

        return _new


def test_constructor_from_dict():
    # https://github.com/pandas-dev/pandas/issues/52445
    result = SubclassedSeries({"a": 1, "b": 2, "c": 3})
    assert isinstance(result, SubclassedSeries)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_xs.py ===
import numpy as np
import pytest

from pandas import (
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


def test_xs_datetimelike_wrapping():
    # GH#31630 a case where we shouldn't wrap datetime64 in Timestamp
    arr = date_range("2016-01-01", periods=3)._data._ndarray

    ser = Series(arr, dtype=object)
    for i in range(len(ser)):
        ser.iloc[i] = arr[i]
    assert ser.dtype == object
    assert isinstance(ser[0], np.datetime64)

    result = ser.xs(0)
    assert isinstance(result, np.datetime64)


class TestXSWithMultiIndex:
    def test_xs_level_series(self, multiindex_dataframe_random_data):
        df = multiindex_dataframe_random_data
        ser = df["A"]
        expected = ser[:, "two"]
        result = df.xs("two", level=1)["A"]
        tm.assert_series_equal(result, expected)

    def test_series_getitem_multiindex_xs_by_label(self):
        # GH#5684
        idx = MultiIndex.from_tuples(
            [("a", "one"), ("a", "two"), ("b", "one"), ("b", "two")]
        )
        ser = Series([1, 2, 3, 4], index=idx)
        return_value = ser.index.set_names(["L1", "L2"], inplace=True)
        assert return_value is None
        expected = Series([1, 3], index=["a", "b"])
        return_value = expected.index.set_names(["L1"], inplace=True)
        assert return_value is None

        result = ser.xs("one", level="L2")
        tm.assert_series_equal(result, expected)

    def test_series_getitem_multiindex_xs(self):
        # GH#6258
        dt = list(date_range("20130903", periods=3))
        idx = MultiIndex.from_product([list("AB"), dt])
        ser = Series([1, 3, 4, 1, 3, 4], index=idx)
        expected = Series([1, 1], index=list("AB"))

        result = ser.xs("20130903", level=1)
        tm.assert_series_equal(result, expected)

    def test_series_xs_droplevel_false(self):
        # GH: 19056
        mi = MultiIndex.from_tuples(
            [("a", "x"), ("a", "y"), ("b", "x")], names=["level1", "level2"]
        )
        ser = Series([1, 1, 1], index=mi)
        result = ser.xs("a", axis=0, drop_level=False)
        expected = Series(
            [1, 1],
            index=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y")], names=["level1", "level2"]
            ),
        )
        tm.assert_series_equal(result, expected)

    def test_xs_key_as_list(self):
        # GH#41760
        mi = MultiIndex.from_tuples([("a", "x")], names=["level1", "level2"])
        ser = Series([1], index=mi)
        with pytest.raises(TypeError, match="list keys are not supported"):
            ser.xs(["a", "x"], axis=0, drop_level=False)

        with pytest.raises(TypeError, match="list keys are not supported"):
            ser.xs(["a"], axis=0, drop_level=False)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_matmul.py ===
import operator

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestMatmul:
    def test_matmul(self):
        # matmul test is for GH#10259
        a = Series(
            np.random.default_rng(2).standard_normal(4), index=["p", "q", "r", "s"]
        )
        b = DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=["1", "2", "3"],
            columns=["p", "q", "r", "s"],
        ).T

        # Series @ DataFrame -> Series
        result = operator.matmul(a, b)
        expected = Series(np.dot(a.values, b.values), index=["1", "2", "3"])
        tm.assert_series_equal(result, expected)

        # DataFrame @ Series -> Series
        result = operator.matmul(b.T, a)
        expected = Series(np.dot(b.T.values, a.T.values), index=["1", "2", "3"])
        tm.assert_series_equal(result, expected)

        # Series @ Series -> scalar
        result = operator.matmul(a, a)
        expected = np.dot(a.values, a.values)
        tm.assert_almost_equal(result, expected)

        # GH#21530
        # vector (1D np.array) @ Series (__rmatmul__)
        result = operator.matmul(a.values, a)
        expected = np.dot(a.values, a.values)
        tm.assert_almost_equal(result, expected)

        # GH#21530
        # vector (1D list) @ Series (__rmatmul__)
        result = operator.matmul(a.values.tolist(), a)
        expected = np.dot(a.values, a.values)
        tm.assert_almost_equal(result, expected)

        # GH#21530
        # matrix (2D np.array) @ Series (__rmatmul__)
        result = operator.matmul(b.T.values, a)
        expected = np.dot(b.T.values, a.values)
        tm.assert_almost_equal(result, expected)

        # GH#21530
        # matrix (2D nested lists) @ Series (__rmatmul__)
        result = operator.matmul(b.T.values.tolist(), a)
        expected = np.dot(b.T.values, a.values)
        tm.assert_almost_equal(result, expected)

        # mixed dtype DataFrame @ Series
        a["p"] = int(a.p)
        result = operator.matmul(b.T, a)
        expected = Series(np.dot(b.T.values, a.T.values), index=["1", "2", "3"])
        tm.assert_series_equal(result, expected)

        # different dtypes DataFrame @ Series
        a = a.astype(int)
        result = operator.matmul(b.T, a)
        expected = Series(np.dot(b.T.values, a.T.values), index=["1", "2", "3"])
        tm.assert_series_equal(result, expected)

        msg = r"Dot product shape mismatch, \(4,\) vs \(3,\)"
        # exception raised is of type Exception
        with pytest.raises(Exception, match=msg):
            a.dot(a.values[:3])
        msg = "matrices are not aligned"
        with pytest.raises(ValueError, match=msg):
            a.dot(b.T)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_galois.py ===
"""Test groups defined by the galois module. """

from sympy.combinatorics.galois import (
    S4TransitiveSubgroups, S5TransitiveSubgroups, S6TransitiveSubgroups,
    find_transitive_subgroups_of_S6,
)
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import (
    SymmetricGroup, AlternatingGroup, CyclicGroup,
)


def test_four_group():
    G = S4TransitiveSubgroups.V.get_perm_group()
    A4 = AlternatingGroup(4)
    assert G.is_subgroup(A4)
    assert G.degree == 4
    assert G.is_transitive()
    assert G.order() == 4
    assert not G.is_cyclic


def test_M20():
    G = S5TransitiveSubgroups.M20.get_perm_group()
    S5 = SymmetricGroup(5)
    A5 = AlternatingGroup(5)
    assert G.is_subgroup(S5)
    assert not G.is_subgroup(A5)
    assert G.degree == 5
    assert G.is_transitive()
    assert G.order() == 20


# Setting this True means that for each of the transitive subgroups of S6,
# we run a test not only on the fixed representation, but also on one freshly
# generated by the search procedure.
INCLUDE_SEARCH_REPS = False
S6_randomized = {}
if INCLUDE_SEARCH_REPS:
    S6_randomized = find_transitive_subgroups_of_S6(*list(S6TransitiveSubgroups))


def get_versions_of_S6_subgroup(name):
    vers = [name.get_perm_group()]
    if INCLUDE_SEARCH_REPS:
        vers.append(S6_randomized[name])
    return vers


def test_S6_transitive_subgroups():
    """
    Test enough characteristics to distinguish all 16 transitive subgroups.
    """
    ts = S6TransitiveSubgroups
    A6 = AlternatingGroup(6)
    for name, alt, order, is_isom, not_isom in [
        (ts.C6,     False,    6, CyclicGroup(6), None),
        (ts.S3,     False,    6, SymmetricGroup(3), None),
        (ts.D6,     False,   12, None, None),
        (ts.A4,     True,    12, None, None),
        (ts.G18,    False,   18, None, None),
        (ts.A4xC2,  False,   24, None, SymmetricGroup(4)),
        (ts.S4m,    False,   24, SymmetricGroup(4), None),
        (ts.S4p,    True,    24, None, None),
        (ts.G36m,   False,   36, None, None),
        (ts.G36p,   True,    36, None, None),
        (ts.S4xC2,  False,   48, None, None),
        (ts.PSL2F5, True,    60, None, None),
        (ts.G72,    False,   72, None, None),
        (ts.PGL2F5, False,  120, None, None),
        (ts.A6,     True,   360, None, None),
        (ts.S6,     False,  720, None, None),
    ]:
        for G in get_versions_of_S6_subgroup(name):
            assert G.is_transitive()
            assert G.degree == 6
            assert G.is_subgroup(A6) is alt
            assert G.order() == order
            if is_isom:
                assert is_isomorphic(G, is_isom)
            if not_isom:
                assert not is_isomorphic(G, not_isom)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\tests\test_guess.py ===
from sympy.concrete.guess import (
            find_simple_recurrence_vector,
            find_simple_recurrence,
            rationalize,
            guess_generating_function_rational,
            guess_generating_function,
            guess
        )
from sympy.concrete.products import Product
from sympy.core.function import Function
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.exponential import exp


def test_find_simple_recurrence_vector():
    assert find_simple_recurrence_vector(
            [fibonacci(k) for k in range(12)]) == [1, -1, -1]


def test_find_simple_recurrence():
    a = Function('a')
    n = Symbol('n')
    assert find_simple_recurrence([fibonacci(k) for k in range(12)]) == (
        -a(n) - a(n + 1) + a(n + 2))

    f = Function('a')
    i = Symbol('n')
    a = [1, 1, 1]
    for k in range(15): a.append(5*a[-1]-3*a[-2]+8*a[-3])
    assert find_simple_recurrence(a, A=f, N=i) == (
        -8*f(i) + 3*f(i + 1) - 5*f(i + 2) + f(i + 3))
    assert find_simple_recurrence([0, 2, 15, 74, 12, 3, 0,
                                    1, 2, 85, 4, 5, 63]) == 0


def test_rationalize():
    from mpmath import cos, pi, mpf
    assert rationalize(cos(pi/3)) == S.Half
    assert rationalize(mpf("0.333333333333333")) == Rational(1, 3)
    assert rationalize(mpf("-0.333333333333333")) == Rational(-1, 3)
    assert rationalize(pi, maxcoeff = 250) == Rational(355, 113)


def test_guess_generating_function_rational():
    x = Symbol('x')
    assert guess_generating_function_rational([fibonacci(k)
        for k in range(5, 15)]) == ((3*x + 5)/(-x**2 - x + 1))


def test_guess_generating_function():
    x = Symbol('x')
    assert guess_generating_function([fibonacci(k)
        for k in range(5, 15)])['ogf'] == ((3*x + 5)/(-x**2 - x + 1))
    assert guess_generating_function(
        [1, 2, 5, 14, 41, 124, 383, 1200, 3799, 12122, 38919])['ogf'] == (
        (1/(x**4 + 2*x**2 - 4*x + 1))**S.Half)
    assert guess_generating_function(sympify(
       "[3/2, 11/2, 0, -121/2, -363/2, 121, 4719/2, 11495/2, -8712, -178717/2]")
       )['ogf'] == (x + Rational(3, 2))/(11*x**2 - 3*x + 1)
    assert guess_generating_function([factorial(k) for k in range(12)],
       types=['egf'])['egf'] == 1/(-x + 1)
    assert guess_generating_function([k+1 for k in range(12)],
       types=['egf']) == {'egf': (x + 1)*exp(x), 'lgdegf': (x + 2)/(x + 1)}


def test_guess():
    i0, i1 = symbols('i0 i1')
    assert guess([1, 2, 6, 24, 120], evaluate=False) == [Product(i1 + 1, (i1, 1, i0 - 1))]
    assert guess([1, 2, 6, 24, 120]) == [RisingFactorial(2, i0 - 1)]
    assert guess([1, 2, 7, 42, 429, 7436, 218348, 10850216], niter=4) == [
        2**(i0 - 1)*(Rational(27, 16))**(i0**2/2 - 3*i0/2 +
        1)*Product(RisingFactorial(Rational(5, 3), i1 - 1)*RisingFactorial(Rational(7, 3), i1
        - 1)/(RisingFactorial(Rational(3, 2), i1 - 1)*RisingFactorial(Rational(5, 2), i1 -
        1)), (i1, 1, i0 - 1))]
    assert guess([1, 0, 2]) == []
    x, y = symbols('x y')
    assert guess([1, 2, 6, 24, 120], variables=[x, y]) == [RisingFactorial(2, x - 1)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\tests\test_interface.py ===
# This test file tests the SymPy function interface, that people use to create
# their own new functions. It should be as easy as possible.
#
# We test that it works with both Function and DefinedFunction. New code should
# use DefinedFunction because it has better type inference. Old code still
# using Function should continue to work though.
from sympy.core.function import Function, DefinedFunction
from sympy.core.sympify import sympify
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.limits import limit
from sympy.abc import x


def test_function_series1():
    """Create our new "sin" function."""

    for F in [Function, DefinedFunction]:

        class my_function(F):

            def fdiff(self, argindex=1):
                return cos(self.args[0])

            @classmethod
            def eval(cls, arg):
                arg = sympify(arg)
                if arg == 0:
                    return sympify(0)

        #Test that the taylor series is correct
        assert my_function(x).series(x, 0, 10) == sin(x).series(x, 0, 10)
        assert limit(my_function(x)/x, x, 0) == 1


def test_function_series2():
    """Create our new "cos" function."""

    for F in [Function, DefinedFunction]:

        class my_function2(F):

            def fdiff(self, argindex=1):
                return -sin(self.args[0])

            @classmethod
            def eval(cls, arg):
                arg = sympify(arg)
                if arg == 0:
                    return sympify(1)

        #Test that the taylor series is correct
        assert my_function2(x).series(x, 0, 10) == cos(x).series(x, 0, 10)


def test_function_series3():
    """
    Test our easy "tanh" function.

    This test tests two things:
      * that the Function interface works as expected and it's easy to use
      * that the general algorithm for the series expansion works even when the
        derivative is defined recursively in terms of the original function,
        since tanh(x).diff(x) == 1-tanh(x)**2
    """

    for F in [Function, DefinedFunction]:

        class mytanh(F):

            def fdiff(self, argindex=1):
                return 1 - mytanh(self.args[0])**2

            @classmethod
            def eval(cls, arg):
                arg = sympify(arg)
                if arg == 0:
                    return sympify(0)

        e = tanh(x)
        f = mytanh(x)
        assert e.series(x, 0, 6) == f.series(x, 0, 6)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\tests\test_waves.py ===
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import (I, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.simplify.simplify import simplify
from sympy.abc import epsilon, mu
from sympy.functions.elementary.exponential import exp
from sympy.physics.units import speed_of_light, m, s
from sympy.physics.optics import TWave

from sympy.testing.pytest import raises

c = speed_of_light.convert_to(m/s)

def test_twave():
    A1, phi1, A2, phi2, f = symbols('A1, phi1, A2, phi2, f')
    n = Symbol('n')  # Refractive index
    t = Symbol('t')  # Time
    x = Symbol('x')  # Spatial variable
    E = Function('E')
    w1 = TWave(A1, f, phi1)
    w2 = TWave(A2, f, phi2)
    assert w1.amplitude == A1
    assert w1.frequency == f
    assert w1.phase == phi1
    assert w1.wavelength == c/(f*n)
    assert w1.time_period == 1/f
    assert w1.angular_velocity == 2*pi*f
    assert w1.wavenumber == 2*pi*f*n/c
    assert w1.speed == c/n

    w3 = w1 + w2
    assert w3.amplitude == sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2)
    assert w3.frequency == f
    assert w3.phase == atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2))
    assert w3.wavelength == c/(f*n)
    assert w3.time_period == 1/f
    assert w3.angular_velocity == 2*pi*f
    assert w3.wavenumber == 2*pi*f*n/c
    assert w3.speed == c/n
    assert simplify(w3.rewrite(sin) - w2.rewrite(sin) - w1.rewrite(sin)) == 0
    assert w3.rewrite('pde') == epsilon*mu*Derivative(E(x, t), t, t) + Derivative(E(x, t), x, x)
    assert w3.rewrite(cos) == sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2)
        + A2**2)*cos(pi*f*n*x*s/(149896229*m) - 2*pi*f*t + atan2(A1*sin(phi1)
        + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2)))
    assert w3.rewrite(exp) == sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2)
        + A2**2)*exp(I*(-2*pi*f*t + atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1)
        + A2*cos(phi2)) + pi*s*f*n*x/(149896229*m)))

    w4 = TWave(A1, None, 0, 1/f)
    assert w4.frequency == f

    w5 = w1 - w2
    assert w5.amplitude == sqrt(A1**2 - 2*A1*A2*cos(phi1 - phi2) + A2**2)
    assert w5.frequency == f
    assert w5.phase == atan2(A1*sin(phi1) - A2*sin(phi2), A1*cos(phi1) - A2*cos(phi2))
    assert w5.wavelength == c/(f*n)
    assert w5.time_period == 1/f
    assert w5.angular_velocity == 2*pi*f
    assert w5.wavenumber == 2*pi*f*n/c
    assert w5.speed == c/n
    assert simplify(w5.rewrite(sin) - w1.rewrite(sin) + w2.rewrite(sin)) == 0
    assert w5.rewrite('pde') == epsilon*mu*Derivative(E(x, t), t, t) + Derivative(E(x, t), x, x)
    assert w5.rewrite(cos) == sqrt(A1**2 - 2*A1*A2*cos(phi1 - phi2)
        + A2**2)*cos(-2*pi*f*t + atan2(A1*sin(phi1) - A2*sin(phi2), A1*cos(phi1)
        - A2*cos(phi2)) + pi*s*f*n*x/(149896229*m))
    assert w5.rewrite(exp) == sqrt(A1**2 - 2*A1*A2*cos(phi1 - phi2)
        + A2**2)*exp(I*(-2*pi*f*t + atan2(A1*sin(phi1) - A2*sin(phi2), A1*cos(phi1)
        - A2*cos(phi2)) + pi*s*f*n*x/(149896229*m)))

    w6 = 2*w1
    assert w6.amplitude == 2*A1
    assert w6.frequency == f
    assert w6.phase == phi1
    w7 = -w6
    assert w7.amplitude == -2*A1
    assert w7.frequency == f
    assert w7.phase == phi1

    raises(ValueError, lambda:TWave(A1))
    raises(ValueError, lambda:TWave(A1, f, phi1, t))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_mix.py ===
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ne)
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.stats import (Poisson, Beta, Exponential, P,
                        Multinomial, MultivariateBeta)
from sympy.stats.crv_types import Normal
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
from sympy.stats.joint_rv import MarginalDistribution
from sympy.stats.rv import pspace, density
from sympy.testing.pytest import ignore_warnings

def test_density():
    x = Symbol('x')
    l = Symbol('l', positive=True)
    rate = Beta(l, 2, 3)
    X = Poisson(x, rate)
    assert isinstance(pspace(X), CompoundPSpace)
    assert density(X, Eq(rate, rate.symbol)) == PoissonDistribution(l)
    N1 = Normal('N1', 0, 1)
    N2 = Normal('N2', N1, 2)
    assert density(N2)(0).doit() == sqrt(10)/(10*sqrt(pi))
    assert simplify(density(N2, Eq(N1, 1))(x)) == \
        sqrt(2)*exp(-(x - 1)**2/8)/(4*sqrt(pi))
    assert simplify(density(N2)(x)) == sqrt(10)*exp(-x**2/10)/(10*sqrt(pi))

def test_MarginalDistribution():
    a1, p1, p2 = symbols('a1 p1 p2', positive=True)
    C = Multinomial('C', 2, p1, p2)
    B = MultivariateBeta('B', a1, C[0])
    MGR = MarginalDistribution(B, (C[0],))
    mgrc = Mul(Symbol('B'), Piecewise(ExprCondPair(Mul(Integer(2),
    Pow(Symbol('p1', positive=True), Indexed(IndexedBase(Symbol('C')),
    Integer(0))), Pow(Symbol('p2', positive=True),
    Indexed(IndexedBase(Symbol('C')), Integer(1))),
    Pow(factorial(Indexed(IndexedBase(Symbol('C')), Integer(0))), Integer(-1)),
    Pow(factorial(Indexed(IndexedBase(Symbol('C')), Integer(1))), Integer(-1))),
    Eq(Add(Indexed(IndexedBase(Symbol('C')), Integer(0)),
    Indexed(IndexedBase(Symbol('C')), Integer(1))), Integer(2))),
    ExprCondPair(Integer(0), True)), Pow(gamma(Symbol('a1', positive=True)),
    Integer(-1)), gamma(Add(Symbol('a1', positive=True),
    Indexed(IndexedBase(Symbol('C')), Integer(0)))),
    Pow(gamma(Indexed(IndexedBase(Symbol('C')), Integer(0))), Integer(-1)),
    Pow(Indexed(IndexedBase(Symbol('B')), Integer(0)),
    Add(Symbol('a1', positive=True), Integer(-1))),
    Pow(Indexed(IndexedBase(Symbol('B')), Integer(1)),
    Add(Indexed(IndexedBase(Symbol('C')), Integer(0)), Integer(-1))))
    assert MGR(C) == mgrc

def test_compound_distribution():
    Y = Poisson('Y', 1)
    Z = Poisson('Z', Y)
    assert isinstance(pspace(Z), CompoundPSpace)
    assert isinstance(pspace(Z).distribution, CompoundDistribution)
    assert Z.pspace.distribution.pdf(1).doit() == exp(-2)*exp(exp(-1))

def test_mix_expression():
    Y, E = Poisson('Y', 1), Exponential('E', 1)
    k = Dummy('k')
    expr1 = Integral(Sum(exp(-1)*Integral(exp(-k)*DiracDelta(k - 2), (k, 0, oo)
    )/factorial(k), (k, 0, oo)), (k, -oo, 0))
    expr2 = Integral(Sum(exp(-1)*Integral(exp(-k)*DiracDelta(k - 2), (k, 0, oo)
    )/factorial(k), (k, 0, oo)), (k, 0, oo))
    assert P(Eq(Y + E, 1)) == 0
    assert P(Ne(Y + E, 2)) == 1
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert P(E + Y < 2, evaluate=False).rewrite(Integral).dummy_eq(expr1)
        assert P(E + Y > 2, evaluate=False).rewrite(Integral).dummy_eq(expr2)