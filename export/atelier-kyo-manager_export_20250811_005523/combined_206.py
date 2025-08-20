
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_api.py ===
import sys

import pytest
from numpy._core._rational_tests import rational

import numpy as np
import numpy._core.umath as ncu
from numpy.testing import (
    HAS_REFCOUNT,
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_warns,
)


def test_array_array():
    tobj = type(object)
    ones11 = np.ones((1, 1), np.float64)
    tndarray = type(ones11)
    # Test is_ndarray
    assert_equal(np.array(ones11, dtype=np.float64), ones11)
    if HAS_REFCOUNT:
        old_refcount = sys.getrefcount(tndarray)
        np.array(ones11)
        assert_equal(old_refcount, sys.getrefcount(tndarray))

    # test None
    assert_equal(np.array(None, dtype=np.float64),
                 np.array(np.nan, dtype=np.float64))
    if HAS_REFCOUNT:
        old_refcount = sys.getrefcount(tobj)
        np.array(None, dtype=np.float64)
        assert_equal(old_refcount, sys.getrefcount(tobj))

    # test scalar
    assert_equal(np.array(1.0, dtype=np.float64),
                 np.ones((), dtype=np.float64))
    if HAS_REFCOUNT:
        old_refcount = sys.getrefcount(np.float64)
        np.array(np.array(1.0, dtype=np.float64), dtype=np.float64)
        assert_equal(old_refcount, sys.getrefcount(np.float64))

    # test string
    S2 = np.dtype((bytes, 2))
    S3 = np.dtype((bytes, 3))
    S5 = np.dtype((bytes, 5))
    assert_equal(np.array(b"1.0", dtype=np.float64),
                 np.ones((), dtype=np.float64))
    assert_equal(np.array(b"1.0").dtype, S3)
    assert_equal(np.array(b"1.0", dtype=bytes).dtype, S3)
    assert_equal(np.array(b"1.0", dtype=S2), np.array(b"1."))
    assert_equal(np.array(b"1", dtype=S5), np.ones((), dtype=S5))

    # test string
    U2 = np.dtype((str, 2))
    U3 = np.dtype((str, 3))
    U5 = np.dtype((str, 5))
    assert_equal(np.array("1.0", dtype=np.float64),
                 np.ones((), dtype=np.float64))
    assert_equal(np.array("1.0").dtype, U3)
    assert_equal(np.array("1.0", dtype=str).dtype, U3)
    assert_equal(np.array("1.0", dtype=U2), np.array("1."))
    assert_equal(np.array("1", dtype=U5), np.ones((), dtype=U5))

    builtins = getattr(__builtins__, '__dict__', __builtins__)
    assert_(hasattr(builtins, 'get'))

    # test memoryview
    dat = np.array(memoryview(b'1.0'), dtype=np.float64)
    assert_equal(dat, [49.0, 46.0, 48.0])
    assert_(dat.dtype.type is np.float64)

    dat = np.array(memoryview(b'1.0'))
    assert_equal(dat, [49, 46, 48])
    assert_(dat.dtype.type is np.uint8)

    # test array interface
    a = np.array(100.0, dtype=np.float64)
    o = type("o", (object,),
             {"__array_interface__": a.__array_interface__})
    assert_equal(np.array(o, dtype=np.float64), a)

    # test array_struct interface
    a = np.array([(1, 4.0, 'Hello'), (2, 6.0, 'World')],
                 dtype=[('f0', int), ('f1', float), ('f2', str)])
    o = type("o", (object,),
             {"__array_struct__": a.__array_struct__})
    # wasn't what I expected... is np.array(o) supposed to equal a ?
    # instead we get a array([...], dtype=">V18")
    assert_equal(bytes(np.array(o).data), bytes(a.data))

    # test array
    def custom__array__(self, dtype=None, copy=None):
        return np.array(100.0, dtype=dtype, copy=copy)

    o = type("o", (object,), {"__array__": custom__array__})()
    assert_equal(np.array(o, dtype=np.float64), np.array(100.0, np.float64))

    # test recursion
    nested = 1.5
    for i in range(ncu.MAXDIMS):
        nested = [nested]

    # no error
    np.array(nested)

    # Exceeds recursion limit
    assert_raises(ValueError, np.array, [nested], dtype=np.float64)

    # Try with lists...
    # float32
    assert_equal(np.array([None] * 10, dtype=np.float32),
                 np.full((10,), np.nan, dtype=np.float32))
    assert_equal(np.array([[None]] * 10, dtype=np.float32),
                 np.full((10, 1), np.nan, dtype=np.float32))
    assert_equal(np.array([[None] * 10], dtype=np.float32),
                 np.full((1, 10), np.nan, dtype=np.float32))
    assert_equal(np.array([[None] * 10] * 10, dtype=np.float32),
                 np.full((10, 10), np.nan, dtype=np.float32))
    # float64
    assert_equal(np.array([None] * 10, dtype=np.float64),
                 np.full((10,), np.nan, dtype=np.float64))
    assert_equal(np.array([[None]] * 10, dtype=np.float64),
                 np.full((10, 1), np.nan, dtype=np.float64))
    assert_equal(np.array([[None] * 10], dtype=np.float64),
                 np.full((1, 10), np.nan, dtype=np.float64))
    assert_equal(np.array([[None] * 10] * 10, dtype=np.float64),
                 np.full((10, 10), np.nan, dtype=np.float64))

    assert_equal(np.array([1.0] * 10, dtype=np.float64),
                 np.ones((10,), dtype=np.float64))
    assert_equal(np.array([[1.0]] * 10, dtype=np.float64),
                 np.ones((10, 1), dtype=np.float64))
    assert_equal(np.array([[1.0] * 10], dtype=np.float64),
                 np.ones((1, 10), dtype=np.float64))
    assert_equal(np.array([[1.0] * 10] * 10, dtype=np.float64),
                 np.ones((10, 10), dtype=np.float64))

    # Try with tuples
    assert_equal(np.array((None,) * 10, dtype=np.float64),
                 np.full((10,), np.nan, dtype=np.float64))
    assert_equal(np.array([(None,)] * 10, dtype=np.float64),
                 np.full((10, 1), np.nan, dtype=np.float64))
    assert_equal(np.array([(None,) * 10], dtype=np.float64),
                 np.full((1, 10), np.nan, dtype=np.float64))
    assert_equal(np.array([(None,) * 10] * 10, dtype=np.float64),
                 np.full((10, 10), np.nan, dtype=np.float64))

    assert_equal(np.array((1.0,) * 10, dtype=np.float64),
                 np.ones((10,), dtype=np.float64))
    assert_equal(np.array([(1.0,)] * 10, dtype=np.float64),
                 np.ones((10, 1), dtype=np.float64))
    assert_equal(np.array([(1.0,) * 10], dtype=np.float64),
                 np.ones((1, 10), dtype=np.float64))
    assert_equal(np.array([(1.0,) * 10] * 10, dtype=np.float64),
                 np.ones((10, 10), dtype=np.float64))

@pytest.mark.parametrize("array", [True, False])
def test_array_impossible_casts(array):
    # All builtin types can be forcibly cast, at least theoretically,
    # but user dtypes cannot necessarily.
    rt = rational(1, 2)
    if array:
        rt = np.array(rt)
    with assert_raises(TypeError):
        np.array(rt, dtype="M8")


def test_array_astype():
    a = np.arange(6, dtype='f4').reshape(2, 3)
    # Default behavior: allows unsafe casts, keeps memory layout,
    #                   always copies.
    b = a.astype('i4')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('i4'))
    assert_equal(a.strides, b.strides)
    b = a.T.astype('i4')
    assert_equal(a.T, b)
    assert_equal(b.dtype, np.dtype('i4'))
    assert_equal(a.T.strides, b.strides)
    b = a.astype('f4')
    assert_equal(a, b)
    assert_(not (a is b))

    # copy=False parameter skips a copy
    b = a.astype('f4', copy=False)
    assert_(a is b)

    # order parameter allows overriding of the memory layout,
    # forcing a copy if the layout is wrong
    b = a.astype('f4', order='F', copy=False)
    assert_equal(a, b)
    assert_(not (a is b))
    assert_(b.flags.f_contiguous)

    b = a.astype('f4', order='C', copy=False)
    assert_equal(a, b)
    assert_(a is b)
    assert_(b.flags.c_contiguous)

    # casting parameter allows catching bad casts
    b = a.astype('c8', casting='safe')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('c8'))

    assert_raises(TypeError, a.astype, 'i4', casting='safe')

    # subok=False passes through a non-subclassed array
    b = a.astype('f4', subok=0, copy=False)
    assert_(a is b)

    class MyNDArray(np.ndarray):
        pass

    a = np.array([[0, 1, 2], [3, 4, 5]], dtype='f4').view(MyNDArray)

    # subok=True passes through a subclass
    b = a.astype('f4', subok=True, copy=False)
    assert_(a is b)

    # subok=True is default, and creates a subtype on a cast
    b = a.astype('i4', copy=False)
    assert_equal(a, b)
    assert_equal(type(b), MyNDArray)

    # subok=False never returns a subclass
    b = a.astype('f4', subok=False, copy=False)
    assert_equal(a, b)
    assert_(not (a is b))
    assert_(type(b) is not MyNDArray)

    # Make sure converting from string object to fixed length string
    # does not truncate.
    a = np.array([b'a' * 100], dtype='O')
    b = a.astype('S')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('S100'))
    a = np.array(['a' * 100], dtype='O')
    b = a.astype('U')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('U100'))

    # Same test as above but for strings shorter than 64 characters
    a = np.array([b'a' * 10], dtype='O')
    b = a.astype('S')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('S10'))
    a = np.array(['a' * 10], dtype='O')
    b = a.astype('U')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('U10'))

    a = np.array(123456789012345678901234567890, dtype='O').astype('S')
    assert_array_equal(a, np.array(b'1234567890' * 3, dtype='S30'))
    a = np.array(123456789012345678901234567890, dtype='O').astype('U')
    assert_array_equal(a, np.array('1234567890' * 3, dtype='U30'))

    a = np.array([123456789012345678901234567890], dtype='O').astype('S')
    assert_array_equal(a, np.array(b'1234567890' * 3, dtype='S30'))
    a = np.array([123456789012345678901234567890], dtype='O').astype('U')
    assert_array_equal(a, np.array('1234567890' * 3, dtype='U30'))

    a = np.array(123456789012345678901234567890, dtype='S')
    assert_array_equal(a, np.array(b'1234567890' * 3, dtype='S30'))
    a = np.array(123456789012345678901234567890, dtype='U')
    assert_array_equal(a, np.array('1234567890' * 3, dtype='U30'))

    a = np.array('a\u0140', dtype='U')
    b = np.ndarray(buffer=a, dtype='uint32', shape=2)
    assert_(b.size == 2)

    a = np.array([1000], dtype='i4')
    assert_raises(TypeError, a.astype, 'S1', casting='safe')

    a = np.array(1000, dtype='i4')
    assert_raises(TypeError, a.astype, 'U1', casting='safe')

    # gh-24023
    assert_raises(TypeError, a.astype)

@pytest.mark.parametrize("dt", ["S", "U"])
def test_array_astype_to_string_discovery_empty(dt):
    # See also gh-19085
    arr = np.array([""], dtype=object)
    # Note, the itemsize is the `0 -> 1` logic, which should change.
    # The important part the test is rather that it does not error.
    assert arr.astype(dt).dtype.itemsize == np.dtype(f"{dt}1").itemsize

    # check the same thing for `np.can_cast` (since it accepts arrays)
    assert np.can_cast(arr, dt, casting="unsafe")
    assert not np.can_cast(arr, dt, casting="same_kind")
    # as well as for the object as a descriptor:
    assert np.can_cast("O", dt, casting="unsafe")

@pytest.mark.parametrize("dt", ["d", "f", "S13", "U32"])
def test_array_astype_to_void(dt):
    dt = np.dtype(dt)
    arr = np.array([], dtype=dt)
    assert arr.astype("V").dtype.itemsize == dt.itemsize

def test_object_array_astype_to_void():
    # This is different to `test_array_astype_to_void` as object arrays
    # are inspected.  The default void is "V8" (8 is the length of double)
    arr = np.array([], dtype="O").astype("V")
    assert arr.dtype == "V8"

@pytest.mark.parametrize("t",
    np._core.sctypes['uint'] +
    np._core.sctypes['int'] +
    np._core.sctypes['float']
)
def test_array_astype_warning(t):
    # test ComplexWarning when casting from complex to float or int
    a = np.array(10, dtype=np.complex128)
    assert_warns(np.exceptions.ComplexWarning, a.astype, t)

@pytest.mark.parametrize(["dtype", "out_dtype"],
        [(np.bytes_, np.bool),
         (np.str_, np.bool),
         (np.dtype("S10,S9"), np.dtype("?,?")),
         # The following also checks unaligned unicode access:
         (np.dtype("S7,U9"), np.dtype("?,?"))])
def test_string_to_boolean_cast(dtype, out_dtype):
    # Only the last two (empty) strings are falsy (the `\0` is stripped):
    arr = np.array(
            ["10", "10\0\0\0", "0\0\0", "0", "False", " ", "", "\0"],
            dtype=dtype)
    expected = np.array(
            [True, True, True, True, True, True, False, False],
            dtype=out_dtype)
    assert_array_equal(arr.astype(out_dtype), expected)
    # As it's similar, check that nonzero behaves the same (structs are
    # nonzero if all entries are)
    assert_array_equal(np.nonzero(arr), np.nonzero(expected))

@pytest.mark.parametrize("str_type", [str, bytes, np.str_])
@pytest.mark.parametrize("scalar_type",
        [np.complex64, np.complex128, np.clongdouble])
def test_string_to_complex_cast(str_type, scalar_type):
    value = scalar_type(b"1+3j")
    assert scalar_type(value) == 1 + 3j
    assert np.array([value], dtype=object).astype(scalar_type)[()] == 1 + 3j
    assert np.array(value).astype(scalar_type)[()] == 1 + 3j
    arr = np.zeros(1, dtype=scalar_type)
    arr[0] = value
    assert arr[0] == 1 + 3j

@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
def test_none_to_nan_cast(dtype):
    # Note that at the time of writing this test, the scalar constructors
    # reject None
    arr = np.zeros(1, dtype=dtype)
    arr[0] = None
    assert np.isnan(arr)[0]
    assert np.isnan(np.array(None, dtype=dtype))[()]
    assert np.isnan(np.array([None], dtype=dtype))[0]
    assert np.isnan(np.array(None).astype(dtype))[()]

def test_copyto_fromscalar():
    a = np.arange(6, dtype='f4').reshape(2, 3)

    # Simple copy
    np.copyto(a, 1.5)
    assert_equal(a, 1.5)
    np.copyto(a.T, 2.5)
    assert_equal(a, 2.5)

    # Where-masked copy
    mask = np.array([[0, 1, 0], [0, 0, 1]], dtype='?')
    np.copyto(a, 3.5, where=mask)
    assert_equal(a, [[2.5, 3.5, 2.5], [2.5, 2.5, 3.5]])
    mask = np.array([[0, 1], [1, 1], [1, 0]], dtype='?')
    np.copyto(a.T, 4.5, where=mask)
    assert_equal(a, [[2.5, 4.5, 4.5], [4.5, 4.5, 3.5]])

def test_copyto():
    a = np.arange(6, dtype='i4').reshape(2, 3)

    # Simple copy
    np.copyto(a, [[3, 1, 5], [6, 2, 1]])
    assert_equal(a, [[3, 1, 5], [6, 2, 1]])

    # Overlapping copy should work
    np.copyto(a[:, :2], a[::-1, 1::-1])
    assert_equal(a, [[2, 6, 5], [1, 3, 1]])

    # Defaults to 'same_kind' casting
    assert_raises(TypeError, np.copyto, a, 1.5)

    # Force a copy with 'unsafe' casting, truncating 1.5 to 1
    np.copyto(a, 1.5, casting='unsafe')
    assert_equal(a, 1)

    # Copying with a mask
    np.copyto(a, 3, where=[True, False, True])
    assert_equal(a, [[3, 1, 3], [3, 1, 3]])

    # Casting rule still applies with a mask
    assert_raises(TypeError, np.copyto, a, 3.5, where=[True, False, True])

    # Lists of integer 0's and 1's is ok too
    np.copyto(a, 4.0, casting='unsafe', where=[[0, 1, 1], [1, 0, 0]])
    assert_equal(a, [[3, 4, 4], [4, 1, 3]])

    # Overlapping copy with mask should work
    np.copyto(a[:, :2], a[::-1, 1::-1], where=[[0, 1], [1, 1]])
    assert_equal(a, [[3, 4, 4], [4, 3, 3]])

    # 'dst' must be an array
    assert_raises(TypeError, np.copyto, [1, 2, 3], [2, 3, 4])


def test_copyto_cast_safety():
    with pytest.raises(TypeError):
        np.copyto(np.arange(3), 3., casting="safe")

    # Can put integer and float scalars safely (and equiv):
    np.copyto(np.arange(3), 3, casting="equiv")
    np.copyto(np.arange(3.), 3., casting="equiv")
    # And also with less precision safely:
    np.copyto(np.arange(3, dtype="uint8"), 3, casting="safe")
    np.copyto(np.arange(3., dtype="float32"), 3., casting="safe")

    # But not equiv:
    with pytest.raises(TypeError):
        np.copyto(np.arange(3, dtype="uint8"), 3, casting="equiv")

    with pytest.raises(TypeError):
        np.copyto(np.arange(3., dtype="float32"), 3., casting="equiv")

    # As a special thing, object is equiv currently:
    np.copyto(np.arange(3, dtype=object), 3, casting="equiv")

    # The following raises an overflow error/gives a warning but not
    # type error (due to casting), though:
    with pytest.raises(OverflowError):
        np.copyto(np.arange(3), 2**80, casting="safe")

    with pytest.warns(RuntimeWarning):
        np.copyto(np.arange(3, dtype=np.float32), 2e300, casting="safe")


def test_copyto_permut():
    # test explicit overflow case
    pad = 500
    l = [True] * pad + [True, True, True, True]
    r = np.zeros(len(l) - pad)
    d = np.ones(len(l) - pad)
    mask = np.array(l)[pad:]
    np.copyto(r, d, where=mask[::-1])

    # test all permutation of possible masks, 9 should be sufficient for
    # current 4 byte unrolled code
    power = 9
    d = np.ones(power)
    for i in range(2**power):
        r = np.zeros(power)
        l = [(i & x) != 0 for x in range(power)]
        mask = np.array(l)
        np.copyto(r, d, where=mask)
        assert_array_equal(r == 1, l)
        assert_equal(r.sum(), sum(l))

        r = np.zeros(power)
        np.copyto(r, d, where=mask[::-1])
        assert_array_equal(r == 1, l[::-1])
        assert_equal(r.sum(), sum(l))

        r = np.zeros(power)
        np.copyto(r[::2], d[::2], where=mask[::2])
        assert_array_equal(r[::2] == 1, l[::2])
        assert_equal(r[::2].sum(), sum(l[::2]))

        r = np.zeros(power)
        np.copyto(r[::2], d[::2], where=mask[::-2])
        assert_array_equal(r[::2] == 1, l[::-2])
        assert_equal(r[::2].sum(), sum(l[::-2]))

        for c in [0xFF, 0x7F, 0x02, 0x10]:
            r = np.zeros(power)
            mask = np.array(l)
            imask = np.array(l).view(np.uint8)
            imask[mask != 0] = c
            np.copyto(r, d, where=mask)
            assert_array_equal(r == 1, l)
            assert_equal(r.sum(), sum(l))

    r = np.zeros(power)
    np.copyto(r, d, where=True)
    assert_equal(r.sum(), r.size)
    r = np.ones(power)
    d = np.zeros(power)
    np.copyto(r, d, where=False)
    assert_equal(r.sum(), r.size)

def test_copy_order():
    a = np.arange(24).reshape(2, 1, 3, 4)
    b = a.copy(order='F')
    c = np.arange(24).reshape(2, 1, 4, 3).swapaxes(2, 3)

    def check_copy_result(x, y, ccontig, fcontig, strides=False):
        assert_(not (x is y))
        assert_equal(x, y)
        assert_equal(res.flags.c_contiguous, ccontig)
        assert_equal(res.flags.f_contiguous, fcontig)

    # Validate the initial state of a, b, and c
    assert_(a.flags.c_contiguous)
    assert_(not a.flags.f_contiguous)
    assert_(not b.flags.c_contiguous)
    assert_(b.flags.f_contiguous)
    assert_(not c.flags.c_contiguous)
    assert_(not c.flags.f_contiguous)

    # Copy with order='C'
    res = a.copy(order='C')
    check_copy_result(res, a, ccontig=True, fcontig=False, strides=True)
    res = b.copy(order='C')
    check_copy_result(res, b, ccontig=True, fcontig=False, strides=False)
    res = c.copy(order='C')
    check_copy_result(res, c, ccontig=True, fcontig=False, strides=False)
    res = np.copy(a, order='C')
    check_copy_result(res, a, ccontig=True, fcontig=False, strides=True)
    res = np.copy(b, order='C')
    check_copy_result(res, b, ccontig=True, fcontig=False, strides=False)
    res = np.copy(c, order='C')
    check_copy_result(res, c, ccontig=True, fcontig=False, strides=False)

    # Copy with order='F'
    res = a.copy(order='F')
    check_copy_result(res, a, ccontig=False, fcontig=True, strides=False)
    res = b.copy(order='F')
    check_copy_result(res, b, ccontig=False, fcontig=True, strides=True)
    res = c.copy(order='F')
    check_copy_result(res, c, ccontig=False, fcontig=True, strides=False)
    res = np.copy(a, order='F')
    check_copy_result(res, a, ccontig=False, fcontig=True, strides=False)
    res = np.copy(b, order='F')
    check_copy_result(res, b, ccontig=False, fcontig=True, strides=True)
    res = np.copy(c, order='F')
    check_copy_result(res, c, ccontig=False, fcontig=True, strides=False)

    # Copy with order='K'
    res = a.copy(order='K')
    check_copy_result(res, a, ccontig=True, fcontig=False, strides=True)
    res = b.copy(order='K')
    check_copy_result(res, b, ccontig=False, fcontig=True, strides=True)
    res = c.copy(order='K')
    check_copy_result(res, c, ccontig=False, fcontig=False, strides=True)
    res = np.copy(a, order='K')
    check_copy_result(res, a, ccontig=True, fcontig=False, strides=True)
    res = np.copy(b, order='K')
    check_copy_result(res, b, ccontig=False, fcontig=True, strides=True)
    res = np.copy(c, order='K')
    check_copy_result(res, c, ccontig=False, fcontig=False, strides=True)

def test_contiguous_flags():
    a = np.ones((4, 4, 1))[::2, :, :]
    a.strides = a.strides[:2] + (-123,)
    b = np.ones((2, 2, 1, 2, 2)).swapaxes(3, 4)

    def check_contig(a, ccontig, fcontig):
        assert_(a.flags.c_contiguous == ccontig)
        assert_(a.flags.f_contiguous == fcontig)

    # Check if new arrays are correct:
    check_contig(a, False, False)
    check_contig(b, False, False)
    check_contig(np.empty((2, 2, 0, 2, 2)), True, True)
    check_contig(np.array([[[1], [2]]], order='F'), True, True)
    check_contig(np.empty((2, 2)), True, False)
    check_contig(np.empty((2, 2), order='F'), False, True)

    # Check that np.array creates correct contiguous flags:
    check_contig(np.array(a, copy=None), False, False)
    check_contig(np.array(a, copy=None, order='C'), True, False)
    check_contig(np.array(a, ndmin=4, copy=None, order='F'), False, True)

    # Check slicing update of flags and :
    check_contig(a[0], True, True)
    check_contig(a[None, ::4, ..., None], True, True)
    check_contig(b[0, 0, ...], False, True)
    check_contig(b[:, :, 0:0, :, :], True, True)

    # Test ravel and squeeze.
    check_contig(a.ravel(), True, True)
    check_contig(np.ones((1, 3, 1)).squeeze(), True, True)

def test_broadcast_arrays():
    # Test user defined dtypes
    a = np.array([(1, 2, 3)], dtype='u4,u4,u4')
    b = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype='u4,u4,u4')
    result = np.broadcast_arrays(a, b)
    assert_equal(result[0], np.array([(1, 2, 3), (1, 2, 3), (1, 2, 3)], dtype='u4,u4,u4'))
    assert_equal(result[1], np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype='u4,u4,u4'))

@pytest.mark.parametrize(["shape", "fill_value", "expected_output"],
        [((2, 2), [5.0,  6.0], np.array([[5.0, 6.0], [5.0, 6.0]])),
         ((3, 2), [1.0,  2.0], np.array([[1.0, 2.0], [1.0, 2.0], [1.0,  2.0]]))])
def test_full_from_list(shape, fill_value, expected_output):
    output = np.full(shape, fill_value)
    assert_equal(output, expected_output)

def test_astype_copyflag():
    # test the various copyflag options
    arr = np.arange(10, dtype=np.intp)

    res_true = arr.astype(np.intp, copy=True)
    assert not np.shares_memory(arr, res_true)

    res_false = arr.astype(np.intp, copy=False)
    assert np.shares_memory(arr, res_false)

    res_false_float = arr.astype(np.float64, copy=False)
    assert not np.shares_memory(arr, res_false_float)

    # _CopyMode enum isn't allowed
    assert_raises(ValueError, arr.astype, np.float64,
                  copy=np._CopyMode.NEVER)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_codegen_julia.py ===
from io import StringIO

from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy


x, y, z = symbols('x,y,z')


def test_empty_jl_code():
    code_gen = JuliaCodeGen()
    output = StringIO()
    code_gen.dump_jl([], output, "file", header=False, empty=False)
    source = output.getvalue()
    assert source == ""


def test_jl_simple_code():
    name_expr = ("test", (x + y)*z)
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    assert result[0] == "test.jl"
    source = result[1]
    expected = (
        "function test(x, y, z)\n"
        "    out1 = z .* (x + y)\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected


def test_jl_simple_code_with_header():
    name_expr = ("test", (x + y)*z)
    result, = codegen(name_expr, "Julia", header=True, empty=False)
    assert result[0] == "test.jl"
    source = result[1]
    expected = (
        "#   Code generated with SymPy " + sympy.__version__ + "\n"
        "#\n"
        "#   See http://www.sympy.org/ for more information.\n"
        "#\n"
        "#   This file is part of 'project'\n"
        "function test(x, y, z)\n"
        "    out1 = z .* (x + y)\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected


def test_jl_simple_code_nameout():
    expr = Equality(z, (x + y))
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(x, y)\n"
        "    z = x + y\n"
        "    return z\n"
        "end\n"
    )
    assert source == expected


def test_jl_numbersymbol():
    name_expr = ("test", pi**Catalan)
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test()\n"
        "    out1 = pi ^ catalan\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected


@XFAIL
def test_jl_numbersymbol_no_inline():
    # FIXME: how to pass inline=False to the JuliaCodePrinter?
    name_expr = ("test", [pi**Catalan, EulerGamma])
    result, = codegen(name_expr, "Julia", header=False,
                      empty=False, inline=False)
    source = result[1]
    expected = (
        "function test()\n"
        "    Catalan = 0.915965594177219\n"
        "    EulerGamma = 0.5772156649015329\n"
        "    out1 = pi ^ Catalan\n"
        "    out2 = EulerGamma\n"
        "    return out1, out2\n"
        "end\n"
    )
    assert source == expected


def test_jl_code_argument_order():
    expr = x + y
    routine = make_routine("test", expr, argument_sequence=[z, x, y], language="julia")
    code_gen = JuliaCodeGen()
    output = StringIO()
    code_gen.dump_jl([routine], output, "test", header=False, empty=False)
    source = output.getvalue()
    expected = (
        "function test(z, x, y)\n"
        "    out1 = x + y\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected


def test_multiple_results_m():
    # Here the output order is the input order
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    name_expr = ("test", [expr1, expr2])
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(x, y, z)\n"
        "    out1 = z .* (x + y)\n"
        "    out2 = z .* (x - y)\n"
        "    return out1, out2\n"
        "end\n"
    )
    assert source == expected


def test_results_named_unordered():
    # Here output order is based on name_expr
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(x, y, z)\n"
        "    C = z .* (x + y)\n"
        "    A = z .* (x - y)\n"
        "    B = 2 * x\n"
        "    return C, A, B\n"
        "end\n"
    )
    assert source == expected


def test_results_named_ordered():
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result = codegen(name_expr, "Julia", header=False, empty=False,
                     argument_sequence=(x, z, y))
    assert result[0][0] == "test.jl"
    source = result[0][1]
    expected = (
        "function test(x, z, y)\n"
        "    C = z .* (x + y)\n"
        "    A = z .* (x - y)\n"
        "    B = 2 * x\n"
        "    return C, A, B\n"
        "end\n"
    )
    assert source == expected


def test_complicated_jl_codegen():
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    name_expr = ("testlong",
            [ ((sin(x) + cos(y) + tan(z))**3).expand(),
            cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))
    ])
    result = codegen(name_expr, "Julia", header=False, empty=False)
    assert result[0][0] == "testlong.jl"
    source = result[0][1]
    expected = (
        "function testlong(x, y, z)\n"
        "    out1 = sin(x) .^ 3 + 3 * sin(x) .^ 2 .* cos(y) + 3 * sin(x) .^ 2 .* tan(z)"
        " + 3 * sin(x) .* cos(y) .^ 2 + 6 * sin(x) .* cos(y) .* tan(z) + 3 * sin(x) .* tan(z) .^ 2"
        " + cos(y) .^ 3 + 3 * cos(y) .^ 2 .* tan(z) + 3 * cos(y) .* tan(z) .^ 2 + tan(z) .^ 3\n"
        "    out2 = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))\n"
        "    return out1, out2\n"
        "end\n"
    )
    assert source == expected


def test_jl_output_arg_mixed_unordered():
    # named outputs are alphabetical, unnamed output appear in the given order
    from sympy.functions.elementary.trigonometric import (cos, sin)
    a = symbols("a")
    name_expr = ("foo", [cos(2*x), Equality(y, sin(x)), cos(x), Equality(a, sin(2*x))])
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    assert result[0] == "foo.jl"
    source = result[1]
    expected = (
        'function foo(x)\n'
        '    out1 = cos(2 * x)\n'
        '    y = sin(x)\n'
        '    out3 = cos(x)\n'
        '    a = sin(2 * x)\n'
        '    return out1, y, out3, a\n'
        'end\n'
    )
    assert source == expected


def test_jl_piecewise_():
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True), evaluate=False)
    name_expr = ("pwtest", pw)
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function pwtest(x)\n"
        "    out1 = ((x < -1) ? (0) :\n"
        "    (x <= 1) ? (x .^ 2) :\n"
        "    (x > 1) ? (2 - x) : (1))\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected


@XFAIL
def test_jl_piecewise_no_inline():
    # FIXME: how to pass inline=False to the JuliaCodePrinter?
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True))
    name_expr = ("pwtest", pw)
    result, = codegen(name_expr, "Julia", header=False, empty=False,
                      inline=False)
    source = result[1]
    expected = (
        "function pwtest(x)\n"
        "    if (x < -1)\n"
        "        out1 = 0\n"
        "    elseif (x <= 1)\n"
        "        out1 = x .^ 2\n"
        "    elseif (x > 1)\n"
        "        out1 = -x + 2\n"
        "    else\n"
        "        out1 = 1\n"
        "    end\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected


def test_jl_multifcns_per_file():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result = codegen(name_expr, "Julia", header=False, empty=False)
    assert result[0][0] == "foo.jl"
    source = result[0][1]
    expected = (
        "function foo(x, y)\n"
        "    out1 = 2 * x\n"
        "    out2 = 3 * y\n"
        "    return out1, out2\n"
        "end\n"
        "function bar(y)\n"
        "    out1 = y .^ 2\n"
        "    out2 = 4 * y\n"
        "    return out1, out2\n"
        "end\n"
    )
    assert source == expected


def test_jl_multifcns_per_file_w_header():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result = codegen(name_expr, "Julia", header=True, empty=False)
    assert result[0][0] == "foo.jl"
    source = result[0][1]
    expected = (
        "#   Code generated with SymPy " + sympy.__version__ + "\n"
        "#\n"
        "#   See http://www.sympy.org/ for more information.\n"
        "#\n"
        "#   This file is part of 'project'\n"
        "function foo(x, y)\n"
        "    out1 = 2 * x\n"
        "    out2 = 3 * y\n"
        "    return out1, out2\n"
        "end\n"
        "function bar(y)\n"
        "    out1 = y .^ 2\n"
        "    out2 = 4 * y\n"
        "    return out1, out2\n"
        "end\n"
    )
    assert source == expected


def test_jl_filename_match_prefix():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result, = codegen(name_expr, "Julia", prefix="baz", header=False,
                     empty=False)
    assert result[0] == "baz.jl"


def test_jl_matrix_named():
    e2 = Matrix([[x, 2*y, pi*z]])
    name_expr = ("test", Equality(MatrixSymbol('myout1', 1, 3), e2))
    result = codegen(name_expr, "Julia", header=False, empty=False)
    assert result[0][0] == "test.jl"
    source = result[0][1]
    expected = (
        "function test(x, y, z)\n"
        "    myout1 = [x 2 * y pi * z]\n"
        "    return myout1\n"
        "end\n"
    )
    assert source == expected


def test_jl_matrix_named_matsym():
    myout1 = MatrixSymbol('myout1', 1, 3)
    e2 = Matrix([[x, 2*y, pi*z]])
    name_expr = ("test", Equality(myout1, e2, evaluate=False))
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(x, y, z)\n"
        "    myout1 = [x 2 * y pi * z]\n"
        "    return myout1\n"
        "end\n"
    )
    assert source == expected


def test_jl_matrix_output_autoname():
    expr = Matrix([[x, x+y, 3]])
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(x, y)\n"
        "    out1 = [x x + y 3]\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected


def test_jl_matrix_output_autoname_2():
    e1 = (x + y)
    e2 = Matrix([[2*x, 2*y, 2*z]])
    e3 = Matrix([[x], [y], [z]])
    e4 = Matrix([[x, y], [z, 16]])
    name_expr = ("test", (e1, e2, e3, e4))
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(x, y, z)\n"
        "    out1 = x + y\n"
        "    out2 = [2 * x 2 * y 2 * z]\n"
        "    out3 = [x, y, z]\n"
        "    out4 = [x  y;\n"
        "    z 16]\n"
        "    return out1, out2, out3, out4\n"
        "end\n"
    )
    assert source == expected


def test_jl_results_matrix_named_ordered():
    B, C = symbols('B,C')
    A = MatrixSymbol('A', 1, 3)
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, Matrix([[1, 2, x]]))
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result, = codegen(name_expr, "Julia", header=False, empty=False,
                     argument_sequence=(x, z, y))
    source = result[1]
    expected = (
        "function test(x, z, y)\n"
        "    C = z .* (x + y)\n"
        "    A = [1 2 x]\n"
        "    B = 2 * x\n"
        "    return C, A, B\n"
        "end\n"
    )
    assert source == expected


def test_jl_matrixsymbol_slice():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 2, 1)
    name_expr = ("test", [Equality(B, A[0, :]),
                          Equality(C, A[1, :]),
                          Equality(D, A[:, 2])])
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(A)\n"
        "    B = A[1,:]\n"
        "    C = A[2,:]\n"
        "    D = A[:,3]\n"
        "    return B, C, D\n"
        "end\n"
    )
    assert source == expected


def test_jl_matrixsymbol_slice2():
    A = MatrixSymbol('A', 3, 4)
    B = MatrixSymbol('B', 2, 2)
    C = MatrixSymbol('C', 2, 2)
    name_expr = ("test", [Equality(B, A[0:2, 0:2]),
                          Equality(C, A[0:2, 1:3])])
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(A)\n"
        "    B = A[1:2,1:2]\n"
        "    C = A[1:2,2:3]\n"
        "    return B, C\n"
        "end\n"
    )
    assert source == expected


def test_jl_matrixsymbol_slice3():
    A = MatrixSymbol('A', 8, 7)
    B = MatrixSymbol('B', 2, 2)
    C = MatrixSymbol('C', 4, 2)
    name_expr = ("test", [Equality(B, A[6:, 1::3]),
                          Equality(C, A[::2, ::3])])
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(A)\n"
        "    B = A[7:end,2:3:end]\n"
        "    C = A[1:2:end,1:3:end]\n"
        "    return B, C\n"
        "end\n"
    )
    assert source == expected


def test_jl_matrixsymbol_slice_autoname():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    name_expr = ("test", [Equality(B, A[0,:]), A[1,:], A[:,0], A[:,1]])
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(A)\n"
        "    B = A[1,:]\n"
        "    out2 = A[2,:]\n"
        "    out3 = A[:,1]\n"
        "    out4 = A[:,2]\n"
        "    return B, out2, out3, out4\n"
        "end\n"
    )
    assert source == expected


def test_jl_loops():
    # Note: an Julia programmer would probably vectorize this across one or
    # more dimensions.  Also, size(A) would be used rather than passing in m
    # and n.  Perhaps users would expect us to vectorize automatically here?
    # Or is it possible to represent such things using IndexedBase?
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    result, = codegen(('mat_vec_mult', Eq(y[i], A[i, j]*x[j])), "Julia",
                      header=False, empty=False)
    source = result[1]
    expected = (
        'function mat_vec_mult(y, A, m, n, x)\n'
        '    for i = 1:m\n'
        '        y[i] = 0\n'
        '    end\n'
        '    for i = 1:m\n'
        '        for j = 1:n\n'
        '            y[i] = %(rhs)s + y[i]\n'
        '        end\n'
        '    end\n'
        '    return y\n'
        'end\n'
    )
    assert (source == expected % {'rhs': 'A[%s,%s] .* x[j]' % (i, j)} or
            source == expected % {'rhs': 'x[j] .* A[%s,%s]' % (i, j)})


def test_jl_tensor_loops_multiple_contractions():
    # see comments in previous test about vectorizing
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m, o, p = symbols('n m o p', integer=True)
    A = IndexedBase('A')
    B = IndexedBase('B')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)
    result, = codegen(('tensorthing', Eq(y[i], B[j, k, l]*A[i, j, k, l])),
                      "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        'function tensorthing(y, A, B, m, n, o, p)\n'
        '    for i = 1:m\n'
        '        y[i] = 0\n'
        '    end\n'
        '    for i = 1:m\n'
        '        for j = 1:n\n'
        '            for k = 1:o\n'
        '                for l = 1:p\n'
        '                    y[i] = A[i,j,k,l] .* B[j,k,l] + y[i]\n'
        '                end\n'
        '            end\n'
        '        end\n'
        '    end\n'
        '    return y\n'
        'end\n'
    )
    assert source == expected


def test_jl_InOutArgument():
    expr = Equality(x, x**2)
    name_expr = ("mysqr", expr)
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function mysqr(x)\n"
        "    x = x .^ 2\n"
        "    return x\n"
        "end\n"
    )
    assert source == expected


def test_jl_InOutArgument_order():
    # can specify the order as (x, y)
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Julia", header=False,
                      empty=False, argument_sequence=(x,y))
    source = result[1]
    expected = (
        "function test(x, y)\n"
        "    x = x .^ 2 + y\n"
        "    return x\n"
        "end\n"
    )
    assert source == expected
    # make sure it gives (x, y) not (y, x)
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(x, y)\n"
        "    x = x .^ 2 + y\n"
        "    return x\n"
        "end\n"
    )
    assert source == expected


def test_jl_not_supported():
    f = Function('f')
    name_expr = ("test", [f(x).diff(x), S.ComplexInfinity])
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    source = result[1]
    expected = (
        "function test(x)\n"
        "    # unsupported: Derivative(f(x), x)\n"
        "    # unsupported: zoo\n"
        "    out1 = Derivative(f(x), x)\n"
        "    out2 = zoo\n"
        "    return out1, out2\n"
        "end\n"
    )
    assert source == expected


def test_global_vars_octave():
    x, y, z, t = symbols("x y z t")
    result = codegen(('f', x*y), "Julia", header=False, empty=False,
                     global_vars=(y,))
    source = result[0][1]
    expected = (
        "function f(x)\n"
        "    out1 = x .* y\n"
        "    return out1\n"
        "end\n"
        )
    assert source == expected

    result = codegen(('f', x*y+z), "Julia", header=False, empty=False,
                     argument_sequence=(x, y), global_vars=(z, t))
    source = result[0][1]
    expected = (
        "function f(x, y)\n"
        "    out1 = x .* y + z\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_classes.py ===
"""Test inter-conversion of different polynomial classes.

This tests the convert and cast methods of all the polynomial classes.

"""
import operator as op
from numbers import Number

import pytest

import numpy as np
from numpy.exceptions import RankWarning
from numpy.polynomial import (
    Chebyshev,
    Hermite,
    HermiteE,
    Laguerre,
    Legendre,
    Polynomial,
)
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

#
# fixtures
#

classes = (
    Polynomial, Legendre, Chebyshev, Laguerre,
    Hermite, HermiteE
    )
classids = tuple(cls.__name__ for cls in classes)

@pytest.fixture(params=classes, ids=classids)
def Poly(request):
    return request.param


#
# helper functions
#
random = np.random.random


def assert_poly_almost_equal(p1, p2, msg=""):
    try:
        assert_(np.all(p1.domain == p2.domain))
        assert_(np.all(p1.window == p2.window))
        assert_almost_equal(p1.coef, p2.coef)
    except AssertionError:
        msg = f"Result: {p1}\nTarget: {p2}"
        raise AssertionError(msg)


#
# Test conversion methods that depend on combinations of two classes.
#

Poly1 = Poly
Poly2 = Poly


def test_conversion(Poly1, Poly2):
    x = np.linspace(0, 1, 10)
    coef = random((3,))

    d1 = Poly1.domain + random((2,)) * .25
    w1 = Poly1.window + random((2,)) * .25
    p1 = Poly1(coef, domain=d1, window=w1)

    d2 = Poly2.domain + random((2,)) * .25
    w2 = Poly2.window + random((2,)) * .25
    p2 = p1.convert(kind=Poly2, domain=d2, window=w2)

    assert_almost_equal(p2.domain, d2)
    assert_almost_equal(p2.window, w2)
    assert_almost_equal(p2(x), p1(x))


def test_cast(Poly1, Poly2):
    x = np.linspace(0, 1, 10)
    coef = random((3,))

    d1 = Poly1.domain + random((2,)) * .25
    w1 = Poly1.window + random((2,)) * .25
    p1 = Poly1(coef, domain=d1, window=w1)

    d2 = Poly2.domain + random((2,)) * .25
    w2 = Poly2.window + random((2,)) * .25
    p2 = Poly2.cast(p1, domain=d2, window=w2)

    assert_almost_equal(p2.domain, d2)
    assert_almost_equal(p2.window, w2)
    assert_almost_equal(p2(x), p1(x))


#
# test methods that depend on one class
#


def test_identity(Poly):
    d = Poly.domain + random((2,)) * .25
    w = Poly.window + random((2,)) * .25
    x = np.linspace(d[0], d[1], 11)
    p = Poly.identity(domain=d, window=w)
    assert_equal(p.domain, d)
    assert_equal(p.window, w)
    assert_almost_equal(p(x), x)


def test_basis(Poly):
    d = Poly.domain + random((2,)) * .25
    w = Poly.window + random((2,)) * .25
    p = Poly.basis(5, domain=d, window=w)
    assert_equal(p.domain, d)
    assert_equal(p.window, w)
    assert_equal(p.coef, [0] * 5 + [1])


def test_fromroots(Poly):
    # check that requested roots are zeros of a polynomial
    # of correct degree, domain, and window.
    d = Poly.domain + random((2,)) * .25
    w = Poly.window + random((2,)) * .25
    r = random((5,))
    p1 = Poly.fromroots(r, domain=d, window=w)
    assert_equal(p1.degree(), len(r))
    assert_equal(p1.domain, d)
    assert_equal(p1.window, w)
    assert_almost_equal(p1(r), 0)

    # check that polynomial is monic
    pdom = Polynomial.domain
    pwin = Polynomial.window
    p2 = Polynomial.cast(p1, domain=pdom, window=pwin)
    assert_almost_equal(p2.coef[-1], 1)


def test_bad_conditioned_fit(Poly):

    x = [0., 0., 1.]
    y = [1., 2., 3.]

    # check RankWarning is raised
    with pytest.warns(RankWarning) as record:
        Poly.fit(x, y, 2)
    assert record[0].message.args[0] == "The fit may be poorly conditioned"


def test_fit(Poly):

    def f(x):
        return x * (x - 1) * (x - 2)
    x = np.linspace(0, 3)
    y = f(x)

    # check default value of domain and window
    p = Poly.fit(x, y, 3)
    assert_almost_equal(p.domain, [0, 3])
    assert_almost_equal(p(x), y)
    assert_equal(p.degree(), 3)

    # check with given domains and window
    d = Poly.domain + random((2,)) * .25
    w = Poly.window + random((2,)) * .25
    p = Poly.fit(x, y, 3, domain=d, window=w)
    assert_almost_equal(p(x), y)
    assert_almost_equal(p.domain, d)
    assert_almost_equal(p.window, w)
    p = Poly.fit(x, y, [0, 1, 2, 3], domain=d, window=w)
    assert_almost_equal(p(x), y)
    assert_almost_equal(p.domain, d)
    assert_almost_equal(p.window, w)

    # check with class domain default
    p = Poly.fit(x, y, 3, [])
    assert_equal(p.domain, Poly.domain)
    assert_equal(p.window, Poly.window)
    p = Poly.fit(x, y, [0, 1, 2, 3], [])
    assert_equal(p.domain, Poly.domain)
    assert_equal(p.window, Poly.window)

    # check that fit accepts weights.
    w = np.zeros_like(x)
    z = y + random(y.shape) * .25
    w[::2] = 1
    p1 = Poly.fit(x[::2], z[::2], 3)
    p2 = Poly.fit(x, z, 3, w=w)
    p3 = Poly.fit(x, z, [0, 1, 2, 3], w=w)
    assert_almost_equal(p1(x), p2(x))
    assert_almost_equal(p2(x), p3(x))


def test_equal(Poly):
    p1 = Poly([1, 2, 3], domain=[0, 1], window=[2, 3])
    p2 = Poly([1, 1, 1], domain=[0, 1], window=[2, 3])
    p3 = Poly([1, 2, 3], domain=[1, 2], window=[2, 3])
    p4 = Poly([1, 2, 3], domain=[0, 1], window=[1, 2])
    assert_(p1 == p1)
    assert_(not p1 == p2)
    assert_(not p1 == p3)
    assert_(not p1 == p4)


def test_not_equal(Poly):
    p1 = Poly([1, 2, 3], domain=[0, 1], window=[2, 3])
    p2 = Poly([1, 1, 1], domain=[0, 1], window=[2, 3])
    p3 = Poly([1, 2, 3], domain=[1, 2], window=[2, 3])
    p4 = Poly([1, 2, 3], domain=[0, 1], window=[1, 2])
    assert_(not p1 != p1)
    assert_(p1 != p2)
    assert_(p1 != p3)
    assert_(p1 != p4)


def test_add(Poly):
    # This checks commutation, not numerical correctness
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = p1 + p2
    assert_poly_almost_equal(p2 + p1, p3)
    assert_poly_almost_equal(p1 + c2, p3)
    assert_poly_almost_equal(c2 + p1, p3)
    assert_poly_almost_equal(p1 + tuple(c2), p3)
    assert_poly_almost_equal(tuple(c2) + p1, p3)
    assert_poly_almost_equal(p1 + np.array(c2), p3)
    assert_poly_almost_equal(np.array(c2) + p1, p3)
    assert_raises(TypeError, op.add, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.add, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.add, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.add, p1, Polynomial([0]))


def test_sub(Poly):
    # This checks commutation, not numerical correctness
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = p1 - p2
    assert_poly_almost_equal(p2 - p1, -p3)
    assert_poly_almost_equal(p1 - c2, p3)
    assert_poly_almost_equal(c2 - p1, -p3)
    assert_poly_almost_equal(p1 - tuple(c2), p3)
    assert_poly_almost_equal(tuple(c2) - p1, -p3)
    assert_poly_almost_equal(p1 - np.array(c2), p3)
    assert_poly_almost_equal(np.array(c2) - p1, -p3)
    assert_raises(TypeError, op.sub, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.sub, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.sub, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.sub, p1, Polynomial([0]))


def test_mul(Poly):
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = p1 * p2
    assert_poly_almost_equal(p2 * p1, p3)
    assert_poly_almost_equal(p1 * c2, p3)
    assert_poly_almost_equal(c2 * p1, p3)
    assert_poly_almost_equal(p1 * tuple(c2), p3)
    assert_poly_almost_equal(tuple(c2) * p1, p3)
    assert_poly_almost_equal(p1 * np.array(c2), p3)
    assert_poly_almost_equal(np.array(c2) * p1, p3)
    assert_poly_almost_equal(p1 * 2, p1 * Poly([2]))
    assert_poly_almost_equal(2 * p1, p1 * Poly([2]))
    assert_raises(TypeError, op.mul, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.mul, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.mul, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.mul, p1, Polynomial([0]))


def test_floordiv(Poly):
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    c3 = list(random((2,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = Poly(c3)
    p4 = p1 * p2 + p3
    c4 = list(p4.coef)
    assert_poly_almost_equal(p4 // p2, p1)
    assert_poly_almost_equal(p4 // c2, p1)
    assert_poly_almost_equal(c4 // p2, p1)
    assert_poly_almost_equal(p4 // tuple(c2), p1)
    assert_poly_almost_equal(tuple(c4) // p2, p1)
    assert_poly_almost_equal(p4 // np.array(c2), p1)
    assert_poly_almost_equal(np.array(c4) // p2, p1)
    assert_poly_almost_equal(2 // p2, Poly([0]))
    assert_poly_almost_equal(p2 // 2, 0.5 * p2)
    assert_raises(
        TypeError, op.floordiv, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(
        TypeError, op.floordiv, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.floordiv, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.floordiv, p1, Polynomial([0]))


def test_truediv(Poly):
    # true division is valid only if the denominator is a Number and
    # not a python bool.
    p1 = Poly([1, 2, 3])
    p2 = p1 * 5

    for stype in np.ScalarType:
        if not issubclass(stype, Number) or issubclass(stype, bool):
            continue
        s = stype(5)
        assert_poly_almost_equal(op.truediv(p2, s), p1)
        assert_raises(TypeError, op.truediv, s, p2)
    for stype in (int, float):
        s = stype(5)
        assert_poly_almost_equal(op.truediv(p2, s), p1)
        assert_raises(TypeError, op.truediv, s, p2)
    for stype in [complex]:
        s = stype(5, 0)
        assert_poly_almost_equal(op.truediv(p2, s), p1)
        assert_raises(TypeError, op.truediv, s, p2)
    for s in [(), [], {}, False, np.array([1])]:
        assert_raises(TypeError, op.truediv, p2, s)
        assert_raises(TypeError, op.truediv, s, p2)
    for ptype in classes:
        assert_raises(TypeError, op.truediv, p2, ptype(1))


def test_mod(Poly):
    # This checks commutation, not numerical correctness
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    c3 = list(random((2,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = Poly(c3)
    p4 = p1 * p2 + p3
    c4 = list(p4.coef)
    assert_poly_almost_equal(p4 % p2, p3)
    assert_poly_almost_equal(p4 % c2, p3)
    assert_poly_almost_equal(c4 % p2, p3)
    assert_poly_almost_equal(p4 % tuple(c2), p3)
    assert_poly_almost_equal(tuple(c4) % p2, p3)
    assert_poly_almost_equal(p4 % np.array(c2), p3)
    assert_poly_almost_equal(np.array(c4) % p2, p3)
    assert_poly_almost_equal(2 % p2, Poly([2]))
    assert_poly_almost_equal(p2 % 2, Poly([0]))
    assert_raises(TypeError, op.mod, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.mod, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.mod, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.mod, p1, Polynomial([0]))


def test_divmod(Poly):
    # This checks commutation, not numerical correctness
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    c3 = list(random((2,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = Poly(c3)
    p4 = p1 * p2 + p3
    c4 = list(p4.coef)
    quo, rem = divmod(p4, p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p4, c2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(c4, p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p4, tuple(c2))
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(tuple(c4), p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p4, np.array(c2))
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(np.array(c4), p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p2, 2)
    assert_poly_almost_equal(quo, 0.5 * p2)
    assert_poly_almost_equal(rem, Poly([0]))
    quo, rem = divmod(2, p2)
    assert_poly_almost_equal(quo, Poly([0]))
    assert_poly_almost_equal(rem, Poly([2]))
    assert_raises(TypeError, divmod, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, divmod, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, divmod, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, divmod, p1, Polynomial([0]))


def test_roots(Poly):
    d = Poly.domain * 1.25 + .25
    w = Poly.window
    tgt = np.linspace(d[0], d[1], 5)
    res = np.sort(Poly.fromroots(tgt, domain=d, window=w).roots())
    assert_almost_equal(res, tgt)
    # default domain and window
    res = np.sort(Poly.fromroots(tgt).roots())
    assert_almost_equal(res, tgt)


def test_degree(Poly):
    p = Poly.basis(5)
    assert_equal(p.degree(), 5)


def test_copy(Poly):
    p1 = Poly.basis(5)
    p2 = p1.copy()
    assert_(p1 == p2)
    assert_(p1 is not p2)
    assert_(p1.coef is not p2.coef)
    assert_(p1.domain is not p2.domain)
    assert_(p1.window is not p2.window)


def test_integ(Poly):
    P = Polynomial
    # Check defaults
    p0 = Poly.cast(P([1 * 2, 2 * 3, 3 * 4]))
    p1 = P.cast(p0.integ())
    p2 = P.cast(p0.integ(2))
    assert_poly_almost_equal(p1, P([0, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([0, 0, 1, 1, 1]))
    # Check with k
    p0 = Poly.cast(P([1 * 2, 2 * 3, 3 * 4]))
    p1 = P.cast(p0.integ(k=1))
    p2 = P.cast(p0.integ(2, k=[1, 1]))
    assert_poly_almost_equal(p1, P([1, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([1, 1, 1, 1, 1]))
    # Check with lbnd
    p0 = Poly.cast(P([1 * 2, 2 * 3, 3 * 4]))
    p1 = P.cast(p0.integ(lbnd=1))
    p2 = P.cast(p0.integ(2, lbnd=1))
    assert_poly_almost_equal(p1, P([-9, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([6, -9, 1, 1, 1]))
    # Check scaling
    d = 2 * Poly.domain
    p0 = Poly.cast(P([1 * 2, 2 * 3, 3 * 4]), domain=d)
    p1 = P.cast(p0.integ())
    p2 = P.cast(p0.integ(2))
    assert_poly_almost_equal(p1, P([0, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([0, 0, 1, 1, 1]))


def test_deriv(Poly):
    # Check that the derivative is the inverse of integration. It is
    # assumes that the integration has been checked elsewhere.
    d = Poly.domain + random((2,)) * .25
    w = Poly.window + random((2,)) * .25
    p1 = Poly([1, 2, 3], domain=d, window=w)
    p2 = p1.integ(2, k=[1, 2])
    p3 = p1.integ(1, k=[1])
    assert_almost_equal(p2.deriv(1).coef, p3.coef)
    assert_almost_equal(p2.deriv(2).coef, p1.coef)
    # default domain and window
    p1 = Poly([1, 2, 3])
    p2 = p1.integ(2, k=[1, 2])
    p3 = p1.integ(1, k=[1])
    assert_almost_equal(p2.deriv(1).coef, p3.coef)
    assert_almost_equal(p2.deriv(2).coef, p1.coef)


def test_linspace(Poly):
    d = Poly.domain + random((2,)) * .25
    w = Poly.window + random((2,)) * .25
    p = Poly([1, 2, 3], domain=d, window=w)
    # check default domain
    xtgt = np.linspace(d[0], d[1], 20)
    ytgt = p(xtgt)
    xres, yres = p.linspace(20)
    assert_almost_equal(xres, xtgt)
    assert_almost_equal(yres, ytgt)
    # check specified domain
    xtgt = np.linspace(0, 2, 20)
    ytgt = p(xtgt)
    xres, yres = p.linspace(20, domain=[0, 2])
    assert_almost_equal(xres, xtgt)
    assert_almost_equal(yres, ytgt)


def test_pow(Poly):
    d = Poly.domain + random((2,)) * .25
    w = Poly.window + random((2,)) * .25
    tgt = Poly([1], domain=d, window=w)
    tst = Poly([1, 2, 3], domain=d, window=w)
    for i in range(5):
        assert_poly_almost_equal(tst**i, tgt)
        tgt = tgt * tst
    # default domain and window
    tgt = Poly([1])
    tst = Poly([1, 2, 3])
    for i in range(5):
        assert_poly_almost_equal(tst**i, tgt)
        tgt = tgt * tst
    # check error for invalid powers
    assert_raises(ValueError, op.pow, tgt, 1.5)
    assert_raises(ValueError, op.pow, tgt, -1)


def test_call(Poly):
    P = Polynomial
    d = Poly.domain
    x = np.linspace(d[0], d[1], 11)

    # Check defaults
    p = Poly.cast(P([1, 2, 3]))
    tgt = 1 + x * (2 + 3 * x)
    res = p(x)
    assert_almost_equal(res, tgt)


def test_call_with_list(Poly):
    p = Poly([1, 2, 3])
    x = [-1, 0, 2]
    res = p(x)
    assert_equal(res, p(np.array(x)))


def test_cutdeg(Poly):
    p = Poly([1, 2, 3])
    assert_raises(ValueError, p.cutdeg, .5)
    assert_raises(ValueError, p.cutdeg, -1)
    assert_equal(len(p.cutdeg(3)), 3)
    assert_equal(len(p.cutdeg(2)), 3)
    assert_equal(len(p.cutdeg(1)), 2)
    assert_equal(len(p.cutdeg(0)), 1)


def test_truncate(Poly):
    p = Poly([1, 2, 3])
    assert_raises(ValueError, p.truncate, .5)
    assert_raises(ValueError, p.truncate, 0)
    assert_equal(len(p.truncate(4)), 3)
    assert_equal(len(p.truncate(3)), 3)
    assert_equal(len(p.truncate(2)), 2)
    assert_equal(len(p.truncate(1)), 1)


def test_trim(Poly):
    c = [1, 1e-6, 1e-12, 0]
    p = Poly(c)
    assert_equal(p.trim().coef, c[:3])
    assert_equal(p.trim(1e-10).coef, c[:2])
    assert_equal(p.trim(1e-5).coef, c[:1])


def test_mapparms(Poly):
    # check with defaults. Should be identity.
    d = Poly.domain
    w = Poly.window
    p = Poly([1], domain=d, window=w)
    assert_almost_equal([0, 1], p.mapparms())
    #
    w = 2 * d + 1
    p = Poly([1], domain=d, window=w)
    assert_almost_equal([1, 2], p.mapparms())


def test_ufunc_override(Poly):
    p = Poly([1, 2, 3])
    x = np.ones(3)
    assert_raises(TypeError, np.add, p, x)
    assert_raises(TypeError, np.add, x, p)


#
# Test class method that only exists for some classes
#


class TestInterpolate:

    def f(self, x):
        return x * (x - 1) * (x - 2)

    def test_raises(self):
        assert_raises(ValueError, Chebyshev.interpolate, self.f, -1)
        assert_raises(TypeError, Chebyshev.interpolate, self.f, 10.)

    def test_dimensions(self):
        for deg in range(1, 5):
            assert_(Chebyshev.interpolate(self.f, deg).degree() == deg)

    def test_approximation(self):

        def powx(x, p):
            return x**p

        x = np.linspace(0, 2, 10)
        for deg in range(10):
            for t in range(deg + 1):
                p = Chebyshev.interpolate(powx, deg, domain=[0, 2], args=(t,))
                assert_almost_equal(p(x), powx(x, t), decimal=11)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_formal.py ===
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, asech)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin)
from sympy.functions.special.bessel import airyai
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.series.formal import fps
from sympy.series.order import O
from sympy.series.formal import (rational_algorithm, FormalPowerSeries,
                                 FormalPowerSeriesProduct, FormalPowerSeriesCompose,
                                 FormalPowerSeriesInverse, simpleDE,
                                 rational_independent, exp_re, hyper_re)
from sympy.testing.pytest import raises, XFAIL, slow

x, y, z = symbols('x y z')
n, m, k = symbols('n m k', integer=True)
f, r = Function('f'), Function('r')


def test_rational_algorithm():
    f = 1 / ((x - 1)**2 * (x - 2))
    assert rational_algorithm(f, x, k) == \
        (-2**(-k - 1) + 1 - (factorial(k + 1) / factorial(k)), 0, 0)

    f = (1 + x + x**2 + x**3) / ((x - 1) * (x - 2))
    assert rational_algorithm(f, x, k) == \
        (-15*2**(-k - 1) + 4, x + 4, 0)

    f = z / (y*m - m*x - y*x + x**2)
    assert rational_algorithm(f, x, k) == \
        (((-y**(-k - 1)*z) / (y - m)) + ((m**(-k - 1)*z) / (y - m)), 0, 0)

    f = x / (1 - x - x**2)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        (((Rational(-1, 2) + sqrt(5)/2)**(-k - 1) *
         (-sqrt(5)/10 + S.Half)) +
         ((-sqrt(5)/2 - S.Half)**(-k - 1) *
         (sqrt(5)/10 + S.Half)), 0, 0)

    f = 1 / (x**2 + 2*x + 2)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        ((I*(-1 + I)**(-k - 1)) / 2 - (I*(-1 - I)**(-k - 1)) / 2, 0, 0)

    f = log(1 + x)
    assert rational_algorithm(f, x, k) == \
        (-(-1)**(-k) / k, 0, 1)

    f = atan(x)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        (((I*I**(-k)) / 2 - (I*(-I)**(-k)) / 2) / k, 0, 1)

    f = x*atan(x) - log(1 + x**2) / 2
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        (((I*I**(-k + 1)) / 2 - (I*(-I)**(-k + 1)) / 2) /
         (k*(k - 1)), 0, 2)

    f = log((1 + x) / (1 - x)) / 2 - atan(x)
    assert rational_algorithm(f, x, k) is None
    assert rational_algorithm(f, x, k, full=True) == \
        ((-(-1)**(-k) / 2 - (I*I**(-k)) / 2 + (I*(-I)**(-k)) / 2 +
          S.Half) / k, 0, 1)

    assert rational_algorithm(cos(x), x, k) is None


def test_rational_independent():
    ri = rational_independent
    assert ri([], x) == []
    assert ri([cos(x), sin(x)], x) == [cos(x), sin(x)]
    assert ri([x**2, sin(x), x*sin(x), x**3], x) == \
        [x**3 + x**2, x*sin(x) + sin(x)]
    assert ri([S.One, x*log(x), log(x), sin(x)/x, cos(x), sin(x), x], x) == \
        [x + 1, x*log(x) + log(x), sin(x)/x + sin(x), cos(x)]


def test_simpleDE():
    # Tests just the first valid DE
    for DE in simpleDE(exp(x), x, f):
        assert DE == (-f(x) + Derivative(f(x), x), 1)
        break
    for DE in simpleDE(sin(x), x, f):
        assert DE == (f(x) + Derivative(f(x), x, x), 2)
        break
    for DE in simpleDE(log(1 + x), x, f):
        assert DE == ((x + 1)*Derivative(f(x), x, 2) + Derivative(f(x), x), 2)
        break
    for DE in simpleDE(asin(x), x, f):
        assert DE == (x*Derivative(f(x), x) + (x**2 - 1)*Derivative(f(x), x, x),
                      2)
        break
    for DE in simpleDE(exp(x)*sin(x), x, f):
        assert DE == (2*f(x) - 2*Derivative(f(x)) + Derivative(f(x), x, x), 2)
        break
    for DE in simpleDE(((1 + x)/(1 - x))**n, x, f):
        assert DE == (2*n*f(x) + (x**2 - 1)*Derivative(f(x), x), 1)
        break
    for DE in simpleDE(airyai(x), x, f):
        assert DE == (-x*f(x) + Derivative(f(x), x, x), 2)
        break


def test_exp_re():
    d = -f(x) + Derivative(f(x), x)
    assert exp_re(d, r, k) == -r(k) + r(k + 1)

    d = f(x) + Derivative(f(x), x, x)
    assert exp_re(d, r, k) == r(k) + r(k + 2)

    d = f(x) + Derivative(f(x), x) + Derivative(f(x), x, x)
    assert exp_re(d, r, k) == r(k) + r(k + 1) + r(k + 2)

    d = Derivative(f(x), x) + Derivative(f(x), x, x)
    assert exp_re(d, r, k) == r(k) + r(k + 1)

    d = Derivative(f(x), x, 3) + Derivative(f(x), x, 4) + Derivative(f(x))
    assert exp_re(d, r, k) == r(k) + r(k + 2) + r(k + 3)


def test_hyper_re():
    d = f(x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == r(k) + (k+1)*(k+2)*r(k + 2)

    d = -x*f(x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == (k + 2)*(k + 3)*r(k + 3) - r(k)

    d = 2*f(x) - 2*Derivative(f(x), x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == \
        (-2*k - 2)*r(k + 1) + (k + 1)*(k + 2)*r(k + 2) + 2*r(k)

    d = 2*n*f(x) + (x**2 - 1)*Derivative(f(x), x)
    assert hyper_re(d, r, k) == \
        k*r(k) + 2*n*r(k + 1) + (-k - 2)*r(k + 2)

    d = (x**10 + 4)*Derivative(f(x), x) + x*(x**10 - 1)*Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == \
        (k*(k - 1) + k)*r(k) + (4*k - (k + 9)*(k + 10) + 40)*r(k + 10)

    d = ((x**2 - 1)*Derivative(f(x), x, 3) + 3*x*Derivative(f(x), x, x) +
         Derivative(f(x), x))
    assert hyper_re(d, r, k) == \
        ((k*(k - 2)*(k - 1) + 3*k*(k - 1) + k)*r(k) +
         (-k*(k + 1)*(k + 2))*r(k + 2))


def test_fps():
    assert fps(1) == 1
    assert fps(2, x) == 2
    assert fps(2, x, dir='+') == 2
    assert fps(2, x, dir='-') == 2
    assert fps(1/x + 1/x**2) == 1/x + 1/x**2
    assert fps(log(1 + x), hyper=False, rational=False) == log(1 + x)

    f = fps(x**2 + x + 1)
    assert isinstance(f, FormalPowerSeries)
    assert f.function == x**2 + x + 1
    assert f[0] == 1
    assert f[2] == x**2
    assert f.truncate(4) == x**2 + x + 1 + O(x**4)
    assert f.polynomial() == x**2 + x + 1

    f = fps(log(1 + x))
    assert isinstance(f, FormalPowerSeries)
    assert f.function == log(1 + x)
    assert f.subs(x, y) == f
    assert f[:5] == [0, x, -x**2/2, x**3/3, -x**4/4]
    assert f.as_leading_term(x) == x
    assert f.polynomial(6) == x - x**2/2 + x**3/3 - x**4/4 + x**5/5

    k = f.ak.variables[0]
    assert f.infinite == Sum((-(-1)**(-k)*x**k)/k, (k, 1, oo))

    ft, s = f.truncate(n=None), f[:5]
    for i, t in enumerate(ft):
        if i == 5:
            break
        assert s[i] == t

    f = sin(x).fps(x)
    assert isinstance(f, FormalPowerSeries)
    assert f.truncate() == x - x**3/6 + x**5/120 + O(x**6)

    raises(NotImplementedError, lambda: fps(y*x))
    raises(ValueError, lambda: fps(x, dir=0))


@slow
def test_fps__rational():
    assert fps(1/x) == (1/x)
    assert fps((x**2 + x + 1) / x**3, dir=-1) == (x**2 + x + 1) / x**3

    f = 1 / ((x - 1)**2 * (x - 2))
    assert fps(f, x).truncate() == \
        (Rational(-1, 2) - x*Rational(5, 4) - 17*x**2/8 - 49*x**3/16 - 129*x**4/32 -
         321*x**5/64 + O(x**6))

    f = (1 + x + x**2 + x**3) / ((x - 1) * (x - 2))
    assert fps(f, x).truncate() == \
        (S.Half + x*Rational(5, 4) + 17*x**2/8 + 49*x**3/16 + 113*x**4/32 +
         241*x**5/64 + O(x**6))

    f = x / (1 - x - x**2)
    assert fps(f, x, full=True).truncate() == \
        x + x**2 + 2*x**3 + 3*x**4 + 5*x**5 + O(x**6)

    f = 1 / (x**2 + 2*x + 2)
    assert fps(f, x, full=True).truncate() == \
        S.Half - x/2 + x**2/4 - x**4/8 + x**5/8 + O(x**6)

    f = log(1 + x)
    assert fps(f, x).truncate() == \
        x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    assert fps(f, x, dir=1).truncate() == fps(f, x, dir=-1).truncate()
    assert fps(f, x, 2).truncate() == \
        (log(3) - Rational(2, 3) - (x - 2)**2/18 + (x - 2)**3/81 -
         (x - 2)**4/324 + (x - 2)**5/1215 + x/3 + O((x - 2)**6, (x, 2)))
    assert fps(f, x, 2, dir=-1).truncate() == \
        (log(3) - Rational(2, 3) - (-x + 2)**2/18 - (-x + 2)**3/81 -
         (-x + 2)**4/324 - (-x + 2)**5/1215 + x/3 + O((x - 2)**6, (x, 2)))

    f = atan(x)
    assert fps(f, x, full=True).truncate() == x - x**3/3 + x**5/5 + O(x**6)
    assert fps(f, x, full=True, dir=1).truncate() == \
        fps(f, x, full=True, dir=-1).truncate()
    assert fps(f, x, 2, full=True).truncate() == \
        (atan(2) - Rational(2, 5) - 2*(x - 2)**2/25 + 11*(x - 2)**3/375 -
         6*(x - 2)**4/625 + 41*(x - 2)**5/15625 + x/5 + O((x - 2)**6, (x, 2)))
    assert fps(f, x, 2, full=True, dir=-1).truncate() == \
        (atan(2) - Rational(2, 5) - 2*(-x + 2)**2/25 - 11*(-x + 2)**3/375 -
         6*(-x + 2)**4/625 - 41*(-x + 2)**5/15625 + x/5 + O((x - 2)**6, (x, 2)))

    f = x*atan(x) - log(1 + x**2) / 2
    assert fps(f, x, full=True).truncate() == x**2/2 - x**4/12 + O(x**6)

    f = log((1 + x) / (1 - x)) / 2 - atan(x)
    assert fps(f, x, full=True).truncate(n=10) == 2*x**3/3 + 2*x**7/7 + O(x**10)


@slow
def test_fps__hyper():
    f = sin(x)
    assert fps(f, x).truncate() == x - x**3/6 + x**5/120 + O(x**6)

    f = cos(x)
    assert fps(f, x).truncate() == 1 - x**2/2 + x**4/24 + O(x**6)

    f = exp(x)
    assert fps(f, x).truncate() == \
        1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)

    f = atan(x)
    assert fps(f, x).truncate() == x - x**3/3 + x**5/5 + O(x**6)

    f = exp(acos(x))
    assert fps(f, x).truncate() == \
        (exp(pi/2) - x*exp(pi/2) + x**2*exp(pi/2)/2 - x**3*exp(pi/2)/3 +
         5*x**4*exp(pi/2)/24 - x**5*exp(pi/2)/6 + O(x**6))

    f = exp(acosh(x))
    assert fps(f, x).truncate() == I + x - I*x**2/2 - I*x**4/8 + O(x**6)

    f = atan(1/x)
    assert fps(f, x).truncate() == pi/2 - x + x**3/3 - x**5/5 + O(x**6)

    f = x*atan(x) - log(1 + x**2) / 2
    assert fps(f, x, rational=False).truncate() == x**2/2 - x**4/12 + O(x**6)

    f = log(1 + x)
    assert fps(f, x, rational=False).truncate() == \
        x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)

    f = airyai(x**2)
    assert fps(f, x).truncate() == \
        (3**Rational(5, 6)*gamma(Rational(1, 3))/(6*pi) -
         3**Rational(2, 3)*x**2/(3*gamma(Rational(1, 3))) + O(x**6))

    f = exp(x)*sin(x)
    assert fps(f, x).truncate() == x + x**2 + x**3/3 - x**5/30 + O(x**6)

    f = exp(x)*sin(x)/x
    assert fps(f, x).truncate() == 1 + x + x**2/3 - x**4/30 - x**5/90 + O(x**6)

    f = sin(x) * cos(x)
    assert fps(f, x).truncate() == x - 2*x**3/3 + 2*x**5/15 + O(x**6)


def test_fps_shift():
    f = x**-5*sin(x)
    assert fps(f, x).truncate() == \
        1/x**4 - 1/(6*x**2) + Rational(1, 120) - x**2/5040 + x**4/362880 + O(x**6)

    f = x**2*atan(x)
    assert fps(f, x, rational=False).truncate() == \
        x**3 - x**5/3 + O(x**6)

    f = cos(sqrt(x))*x
    assert fps(f, x).truncate() == \
        x - x**2/2 + x**3/24 - x**4/720 + x**5/40320 + O(x**6)

    f = x**2*cos(sqrt(x))
    assert fps(f, x).truncate() == \
        x**2 - x**3/2 + x**4/24 - x**5/720 + O(x**6)


def test_fps__Add_expr():
    f = x*atan(x) - log(1 + x**2) / 2
    assert fps(f, x).truncate() == x**2/2 - x**4/12 + O(x**6)

    f = sin(x) + cos(x) - exp(x) + log(1 + x)
    assert fps(f, x).truncate() == x - 3*x**2/2 - x**4/4 + x**5/5 + O(x**6)

    f = 1/x + sin(x)
    assert fps(f, x).truncate() == 1/x + x - x**3/6 + x**5/120 + O(x**6)

    f = sin(x) - cos(x) + 1/(x - 1)
    assert fps(f, x).truncate() == \
        -2 - x**2/2 - 7*x**3/6 - 25*x**4/24 - 119*x**5/120 + O(x**6)


def test_fps__asymptotic():
    f = exp(x)
    assert fps(f, x, oo) == f
    assert fps(f, x, -oo).truncate() == O(1/x**6, (x, oo))

    f = erf(x)
    assert fps(f, x, oo).truncate() == 1 + O(1/x**6, (x, oo))
    assert fps(f, x, -oo).truncate() == -1 + O(1/x**6, (x, oo))

    f = atan(x)
    assert fps(f, x, oo, full=True).truncate() == \
        -1/(5*x**5) + 1/(3*x**3) - 1/x + pi/2 + O(1/x**6, (x, oo))
    assert fps(f, x, -oo, full=True).truncate() == \
        -1/(5*x**5) + 1/(3*x**3) - 1/x - pi/2 + O(1/x**6, (x, oo))

    f = log(1 + x)
    assert fps(f, x, oo) != \
        (-1/(5*x**5) - 1/(4*x**4) + 1/(3*x**3) - 1/(2*x**2) + 1/x - log(1/x) +
         O(1/x**6, (x, oo)))
    assert fps(f, x, -oo) != \
        (-1/(5*x**5) - 1/(4*x**4) + 1/(3*x**3) - 1/(2*x**2) + 1/x + I*pi -
         log(-1/x) + O(1/x**6, (x, oo)))


def test_fps__fractional():
    f = sin(sqrt(x)) / x
    assert fps(f, x).truncate() == \
        (1/sqrt(x) - sqrt(x)/6 + x**Rational(3, 2)/120 -
         x**Rational(5, 2)/5040 + x**Rational(7, 2)/362880 -
         x**Rational(9, 2)/39916800 + x**Rational(11, 2)/6227020800 + O(x**6))

    f = sin(sqrt(x)) * x
    assert fps(f, x).truncate() == \
        (x**Rational(3, 2) - x**Rational(5, 2)/6 + x**Rational(7, 2)/120 -
         x**Rational(9, 2)/5040 + x**Rational(11, 2)/362880 + O(x**6))

    f = atan(sqrt(x)) / x**2
    assert fps(f, x).truncate() == \
        (x**Rational(-3, 2) - x**Rational(-1, 2)/3 + x**S.Half/5 -
         x**Rational(3, 2)/7 + x**Rational(5, 2)/9 - x**Rational(7, 2)/11 +
         x**Rational(9, 2)/13 - x**Rational(11, 2)/15 + O(x**6))

    f = exp(sqrt(x))
    assert fps(f, x).truncate().expand() == \
        (1 + x/2 + x**2/24 + x**3/720 + x**4/40320 + x**5/3628800 + sqrt(x) +
         x**Rational(3, 2)/6 + x**Rational(5, 2)/120 + x**Rational(7, 2)/5040 +
         x**Rational(9, 2)/362880 + x**Rational(11, 2)/39916800 + O(x**6))

    f = exp(sqrt(x))*x
    assert fps(f, x).truncate().expand() == \
        (x + x**2/2 + x**3/24 + x**4/720 + x**5/40320 + x**Rational(3, 2) +
         x**Rational(5, 2)/6 + x**Rational(7, 2)/120 + x**Rational(9, 2)/5040 +
         x**Rational(11, 2)/362880 + O(x**6))


def test_fps__logarithmic_singularity():
    f = log(1 + 1/x)
    assert fps(f, x) != \
        -log(x) + x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)
    assert fps(f, x, rational=False) != \
        -log(x) + x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)


@XFAIL
def test_fps__logarithmic_singularity_fail():
    f = asech(x)  # Algorithms for computing limits probably needs improvements
    assert fps(f, x) == log(2) - log(x) - x**2/4 - 3*x**4/64 + O(x**6)


def test_fps_symbolic():
    f = x**n*sin(x**2)
    assert fps(f, x).truncate(8) == x**(n + 2) - x**(n + 6)/6 + O(x**(n + 8), x)

    f = x**n*log(1 + x)
    fp = fps(f, x)
    k = fp.ak.variables[0]
    assert fp.infinite == \
        Sum((-(-1)**(-k)*x**(k + n))/k, (k, 1, oo))

    f = (x - 2)**n*log(1 + x)
    assert fps(f, x, 2).truncate() == \
        ((x - 2)**n*log(3) + (x - 2)**(n + 1)/3 - (x - 2)**(n + 2)/18 + (x - 2)**(n + 3)/81 -
         (x - 2)**(n + 4)/324 + (x - 2)**(n + 5)/1215 + O((x - 2)**(n + 6), (x, 2)))

    f = x**(n - 2)*cos(x)
    assert fps(f, x).truncate() == \
        (x**(n - 2) - x**n/2 + x**(n + 2)/24 + O(x**(n + 4), x))

    f = x**(n - 2)*sin(x) + x**n*exp(x)
    assert fps(f, x).truncate() == \
        (x**(n - 1) + x**(n + 1) + x**(n + 2)/2 + x**n +
         x**(n + 4)/24 + x**(n + 5)/60 + O(x**(n + 6), x))

    f = x**n*atan(x)
    assert fps(f, x, oo).truncate() == \
        (-x**(n - 5)/5 + x**(n - 3)/3 + x**n*(pi/2 - 1/x) +
         O((1/x)**(-n)/x**6, (x, oo)))

    f = x**(n/2)*cos(x)
    assert fps(f, x).truncate() == \
        x**(n/2) - x**(n/2 + 2)/2 + x**(n/2 + 4)/24 + O(x**(n/2 + 6), x)

    f = x**(n + m)*sin(x)
    assert fps(f, x).truncate() == \
        x**(m + n + 1) - x**(m + n + 3)/6 + x**(m + n + 5)/120 + O(x**(m + n + 6), x)


def test_fps__slow():
    f = x*exp(x)*sin(2*x)  # TODO: rsolve needs improvement
    assert fps(f, x).truncate() == 2*x**2 + 2*x**3 - x**4/3 - x**5 + O(x**6)


def test_fps__operations():
    f1, f2 = fps(sin(x)), fps(cos(x))

    fsum = f1 + f2
    assert fsum.function == sin(x) + cos(x)
    assert fsum.truncate() == \
        1 + x - x**2/2 - x**3/6 + x**4/24 + x**5/120 + O(x**6)

    fsum = f1 + 1
    assert fsum.function == sin(x) + 1
    assert fsum.truncate() == 1 + x - x**3/6 + x**5/120 + O(x**6)

    fsum = 1 + f2
    assert fsum.function == cos(x) + 1
    assert fsum.truncate() == 2 - x**2/2 + x**4/24 + O(x**6)

    assert (f1 + x) == Add(f1, x)

    assert -f2.truncate() == -1 + x**2/2 - x**4/24 + O(x**6)
    assert (f1 - f1) is S.Zero

    fsub = f1 - f2
    assert fsub.function == sin(x) - cos(x)
    assert fsub.truncate() == \
        -1 + x + x**2/2 - x**3/6 - x**4/24 + x**5/120 + O(x**6)

    fsub = f1 - 1
    assert fsub.function == sin(x) - 1
    assert fsub.truncate() == -1 + x - x**3/6 + x**5/120 + O(x**6)

    fsub = 1 - f2
    assert fsub.function == -cos(x) + 1
    assert fsub.truncate() == x**2/2 - x**4/24 + O(x**6)

    raises(ValueError, lambda: f1 + fps(exp(x), dir=-1))
    raises(ValueError, lambda: f1 + fps(exp(x), x0=1))

    fm = f1 * 3

    assert fm.function == 3*sin(x)
    assert fm.truncate() == 3*x - x**3/2 + x**5/40 + O(x**6)

    fm = 3 * f2

    assert fm.function == 3*cos(x)
    assert fm.truncate() == 3 - 3*x**2/2 + x**4/8 + O(x**6)

    assert (f1 * f2) == Mul(f1, f2)
    assert (f1 * x) == Mul(f1, x)

    fd = f1.diff()
    assert fd.function == cos(x)
    assert fd.truncate() == 1 - x**2/2 + x**4/24 + O(x**6)

    fd = f2.diff()
    assert fd.function == -sin(x)
    assert fd.truncate() == -x + x**3/6 - x**5/120 + O(x**6)

    fd = f2.diff().diff()
    assert fd.function == -cos(x)
    assert fd.truncate() == -1 + x**2/2 - x**4/24 + O(x**6)

    f3 = fps(exp(sqrt(x)))
    fd = f3.diff()
    assert fd.truncate().expand() == \
        (1/(2*sqrt(x)) + S.Half + x/12 + x**2/240 + x**3/10080 + x**4/725760 +
         x**5/79833600 + sqrt(x)/4 + x**Rational(3, 2)/48 + x**Rational(5, 2)/1440 +
         x**Rational(7, 2)/80640 + x**Rational(9, 2)/7257600 + x**Rational(11, 2)/958003200 +
         O(x**6))

    assert f1.integrate((x, 0, 1)) == -cos(1) + 1
    assert integrate(f1, (x, 0, 1)) == -cos(1) + 1

    fi = integrate(f1, x)
    assert fi.function == -cos(x)
    assert fi.truncate() == -1 + x**2/2 - x**4/24 + O(x**6)

    fi = f2.integrate(x)
    assert fi.function == sin(x)
    assert fi.truncate() == x - x**3/6 + x**5/120 + O(x**6)

def test_fps__product():
    f1, f2, f3 = fps(sin(x)), fps(exp(x)), fps(cos(x))

    raises(ValueError, lambda: f1.product(exp(x), x))
    raises(ValueError, lambda: f1.product(fps(exp(x), dir=-1), x, 4))
    raises(ValueError, lambda: f1.product(fps(exp(x), x0=1), x, 4))
    raises(ValueError, lambda: f1.product(fps(exp(y)), x, 4))

    fprod = f1.product(f2, x)
    assert isinstance(fprod, FormalPowerSeriesProduct)
    assert isinstance(fprod.ffps, FormalPowerSeries)
    assert isinstance(fprod.gfps, FormalPowerSeries)
    assert fprod.f == sin(x)
    assert fprod.g == exp(x)
    assert fprod.function == sin(x) * exp(x)
    assert fprod._eval_terms(4) == x + x**2 + x**3/3
    assert fprod.truncate(4) == x + x**2 + x**3/3 + O(x**4)
    assert fprod.polynomial(4) == x + x**2 + x**3/3

    raises(NotImplementedError, lambda: fprod._eval_term(5))
    raises(NotImplementedError, lambda: fprod.infinite)
    raises(NotImplementedError, lambda: fprod._eval_derivative(x))
    raises(NotImplementedError, lambda: fprod.integrate(x))

    assert f1.product(f3, x)._eval_terms(4) == x - 2*x**3/3
    assert f1.product(f3, x).truncate(4) == x - 2*x**3/3 + O(x**4)


def test_fps__compose():
    f1, f2, f3 = fps(exp(x)), fps(sin(x)), fps(cos(x))

    raises(ValueError, lambda: f1.compose(sin(x), x))
    raises(ValueError, lambda: f1.compose(fps(sin(x), dir=-1), x, 4))
    raises(ValueError, lambda: f1.compose(fps(sin(x), x0=1), x, 4))
    raises(ValueError, lambda: f1.compose(fps(sin(y)), x, 4))

    raises(ValueError, lambda: f1.compose(f3, x))
    raises(ValueError, lambda: f2.compose(f3, x))

    fcomp = f1.compose(f2, x)
    assert isinstance(fcomp, FormalPowerSeriesCompose)
    assert isinstance(fcomp.ffps, FormalPowerSeries)
    assert isinstance(fcomp.gfps, FormalPowerSeries)
    assert fcomp.f == exp(x)
    assert fcomp.g == sin(x)
    assert fcomp.function == exp(sin(x))
    assert fcomp._eval_terms(6) == 1 + x + x**2/2 - x**4/8 - x**5/15
    assert fcomp.truncate() == 1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)
    assert fcomp.truncate(5) == 1 + x + x**2/2 - x**4/8 + O(x**5)

    raises(NotImplementedError, lambda: fcomp._eval_term(5))
    raises(NotImplementedError, lambda: fcomp.infinite)
    raises(NotImplementedError, lambda: fcomp._eval_derivative(x))
    raises(NotImplementedError, lambda: fcomp.integrate(x))

    assert f1.compose(f2, x).truncate(4) == 1 + x + x**2/2 + O(x**4)
    assert f1.compose(f2, x).truncate(8) == \
        1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8)
    assert f1.compose(f2, x).truncate(6) == \
        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)

    assert f2.compose(f2, x).truncate(4) == x - x**3/3 + O(x**4)
    assert f2.compose(f2, x).truncate(8) == x - x**3/3 + x**5/10 - 8*x**7/315 + O(x**8)
    assert f2.compose(f2, x).truncate(6) == x - x**3/3 + x**5/10 + O(x**6)


def test_fps__inverse():
    f1, f2, f3 = fps(sin(x)), fps(exp(x)), fps(cos(x))

    raises(ValueError, lambda: f1.inverse(x))

    finv = f2.inverse(x)
    assert isinstance(finv, FormalPowerSeriesInverse)
    assert isinstance(finv.ffps, FormalPowerSeries)
    raises(ValueError, lambda: finv.gfps)

    assert finv.f == exp(x)
    assert finv.function == exp(-x)
    assert finv._eval_terms(5) == 1 - x + x**2/2 - x**3/6 + x**4/24
    assert finv.truncate() == 1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6)
    assert finv.truncate(5) == 1 - x + x**2/2 - x**3/6 + x**4/24 + O(x**5)

    raises(NotImplementedError, lambda: finv._eval_term(5))
    raises(ValueError, lambda: finv.g)
    raises(NotImplementedError, lambda: finv.infinite)
    raises(NotImplementedError, lambda: finv._eval_derivative(x))
    raises(NotImplementedError, lambda: finv.integrate(x))

    assert f2.inverse(x).truncate(8) == \
        1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + x**6/720 - x**7/5040 + O(x**8)

    assert f3.inverse(x).truncate() == 1 + x**2/2 + 5*x**4/24 + O(x**6)
    assert f3.inverse(x).truncate(8) == 1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\interchange\test_impl.py ===
from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas.compat import (
    is_ci_environment,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_lt1p23

import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
    ColumnNullType,
    DtypeKind,
)
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes


@pytest.fixture
def data_categorical():
    return {
        "ordered": pd.Categorical(list("testdata") * 30, ordered=True),
        "unordered": pd.Categorical(list("testdata") * 30, ordered=False),
    }


@pytest.fixture
def string_data():
    return {
        "separator data": [
            "abC|DeF,Hik",
            "234,3245.67",
            "gSaf,qWer|Gre",
            "asd3,4sad|",
            np.nan,
        ]
    }


@pytest.mark.parametrize("data", [("ordered", True), ("unordered", False)])
def test_categorical_dtype(data, data_categorical):
    df = pd.DataFrame({"A": (data_categorical[data[0]])})

    col = df.__dataframe__().get_column_by_name("A")
    assert col.dtype[0] == DtypeKind.CATEGORICAL
    assert col.null_count == 0
    assert col.describe_null == (ColumnNullType.USE_SENTINEL, -1)
    assert col.num_chunks() == 1
    desc_cat = col.describe_categorical
    assert desc_cat["is_ordered"] == data[1]
    assert desc_cat["is_dictionary"] is True
    assert isinstance(desc_cat["categories"], PandasColumn)
    tm.assert_series_equal(
        desc_cat["categories"]._col, pd.Series(["a", "d", "e", "s", "t"])
    )

    tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))


def test_categorical_pyarrow():
    # GH 49889
    pa = pytest.importorskip("pyarrow", "11.0.0")

    arr = ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", "Sat", "Sun"]
    table = pa.table({"weekday": pa.array(arr).dictionary_encode()})
    exchange_df = table.__dataframe__()
    result = from_dataframe(exchange_df)
    weekday = pd.Categorical(
        arr, categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    expected = pd.DataFrame({"weekday": weekday})
    tm.assert_frame_equal(result, expected)


def test_empty_categorical_pyarrow():
    # https://github.com/pandas-dev/pandas/issues/53077
    pa = pytest.importorskip("pyarrow", "11.0.0")

    arr = [None]
    table = pa.table({"arr": pa.array(arr, "float64").dictionary_encode()})
    exchange_df = table.__dataframe__()
    result = pd.api.interchange.from_dataframe(exchange_df)
    expected = pd.DataFrame({"arr": pd.Categorical([np.nan])})
    tm.assert_frame_equal(result, expected)


def test_large_string_pyarrow():
    # GH 52795
    pa = pytest.importorskip("pyarrow", "11.0.0")

    arr = ["Mon", "Tue"]
    table = pa.table({"weekday": pa.array(arr, "large_string")})
    exchange_df = table.__dataframe__()
    result = from_dataframe(exchange_df)
    expected = pd.DataFrame({"weekday": ["Mon", "Tue"]})
    tm.assert_frame_equal(result, expected)

    # check round-trip
    assert pa.Table.equals(pa.interchange.from_dataframe(result), table)


@pytest.mark.parametrize(
    ("offset", "length", "expected_values"),
    [
        (0, None, [3.3, float("nan"), 2.1]),
        (1, None, [float("nan"), 2.1]),
        (2, None, [2.1]),
        (0, 2, [3.3, float("nan")]),
        (0, 1, [3.3]),
        (1, 1, [float("nan")]),
    ],
)
def test_bitmasks_pyarrow(offset, length, expected_values):
    # GH 52795
    pa = pytest.importorskip("pyarrow", "11.0.0")

    arr = [3.3, None, 2.1]
    table = pa.table({"arr": arr}).slice(offset, length)
    exchange_df = table.__dataframe__()
    result = from_dataframe(exchange_df)
    expected = pd.DataFrame({"arr": expected_values})
    tm.assert_frame_equal(result, expected)

    # check round-trip
    assert pa.Table.equals(pa.interchange.from_dataframe(result), table)


@pytest.mark.parametrize(
    "data",
    [
        lambda: np.random.default_rng(2).integers(-100, 100),
        lambda: np.random.default_rng(2).integers(1, 100),
        lambda: np.random.default_rng(2).random(),
        lambda: np.random.default_rng(2).choice([True, False]),
        lambda: datetime(
            year=np.random.default_rng(2).integers(1900, 2100),
            month=np.random.default_rng(2).integers(1, 12),
            day=np.random.default_rng(2).integers(1, 20),
        ),
    ],
)
def test_dataframe(data):
    NCOLS, NROWS = 10, 20
    data = {
        f"col{int((i - NCOLS / 2) % NCOLS + 1)}": [data() for _ in range(NROWS)]
        for i in range(NCOLS)
    }
    df = pd.DataFrame(data)

    df2 = df.__dataframe__()

    assert df2.num_columns() == NCOLS
    assert df2.num_rows() == NROWS

    assert list(df2.column_names()) == list(data.keys())

    indices = (0, 2)
    names = tuple(list(data.keys())[idx] for idx in indices)

    result = from_dataframe(df2.select_columns(indices))
    expected = from_dataframe(df2.select_columns_by_name(names))
    tm.assert_frame_equal(result, expected)

    assert isinstance(result.attrs["_INTERCHANGE_PROTOCOL_BUFFERS"], list)
    assert isinstance(expected.attrs["_INTERCHANGE_PROTOCOL_BUFFERS"], list)


def test_missing_from_masked():
    df = pd.DataFrame(
        {
            "x": np.array([1.0, 2.0, 3.0, 4.0, 0.0]),
            "y": np.array([1.5, 2.5, 3.5, 4.5, 0]),
            "z": np.array([1.0, 0.0, 1.0, 1.0, 1.0]),
        }
    )

    rng = np.random.default_rng(2)
    dict_null = {col: rng.integers(low=0, high=len(df)) for col in df.columns}
    for col, num_nulls in dict_null.items():
        null_idx = df.index[
            rng.choice(np.arange(len(df)), size=num_nulls, replace=False)
        ]
        df.loc[null_idx, col] = None

    df2 = df.__dataframe__()

    assert df2.get_column_by_name("x").null_count == dict_null["x"]
    assert df2.get_column_by_name("y").null_count == dict_null["y"]
    assert df2.get_column_by_name("z").null_count == dict_null["z"]


@pytest.mark.parametrize(
    "data",
    [
        {"x": [1.5, 2.5, 3.5], "y": [9.2, 10.5, 11.8]},
        {"x": [1, 2, 0], "y": [9.2, 10.5, 11.8]},
        {
            "x": np.array([True, True, False]),
            "y": np.array([1, 2, 0]),
            "z": np.array([9.2, 10.5, 11.8]),
        },
    ],
)
def test_mixed_data(data):
    df = pd.DataFrame(data)
    df2 = df.__dataframe__()

    for col_name in df.columns:
        assert df2.get_column_by_name(col_name).null_count == 0


def test_mixed_missing():
    df = pd.DataFrame(
        {
            "x": np.array([True, None, False, None, True]),
            "y": np.array([None, 2, None, 1, 2]),
            "z": np.array([9.2, 10.5, None, 11.8, None]),
        }
    )

    df2 = df.__dataframe__()

    for col_name in df.columns:
        assert df2.get_column_by_name(col_name).null_count == 2


def test_string(string_data):
    test_str_data = string_data["separator data"] + [""]
    df = pd.DataFrame({"A": test_str_data})
    col = df.__dataframe__().get_column_by_name("A")

    assert col.size() == 6
    assert col.null_count == 1
    assert col.dtype[0] == DtypeKind.STRING
    assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)

    df_sliced = df[1:]
    col = df_sliced.__dataframe__().get_column_by_name("A")
    assert col.size() == 5
    assert col.null_count == 1
    assert col.dtype[0] == DtypeKind.STRING
    assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)


def test_nonstring_object():
    df = pd.DataFrame({"A": ["a", 10, 1.0, ()]})
    col = df.__dataframe__().get_column_by_name("A")
    with pytest.raises(NotImplementedError, match="not supported yet"):
        col.dtype


def test_datetime():
    df = pd.DataFrame({"A": [pd.Timestamp("2022-01-01"), pd.NaT]})
    col = df.__dataframe__().get_column_by_name("A")

    assert col.size() == 2
    assert col.null_count == 1
    assert col.dtype[0] == DtypeKind.DATETIME
    assert col.describe_null == (ColumnNullType.USE_SENTINEL, iNaT)

    tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))


@pytest.mark.skipif(np_version_lt1p23, reason="Numpy > 1.23 required")
def test_categorical_to_numpy_dlpack():
    # https://github.com/pandas-dev/pandas/issues/48393
    df = pd.DataFrame({"A": pd.Categorical(["a", "b", "a"])})
    col = df.__dataframe__().get_column_by_name("A")
    result = np.from_dlpack(col.get_buffers()["data"][0])
    expected = np.array([0, 1, 0], dtype="int8")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("data", [{}, {"a": []}])
def test_empty_pyarrow(data):
    # GH 53155
    pytest.importorskip("pyarrow", "11.0.0")
    from pyarrow.interchange import from_dataframe as pa_from_dataframe

    expected = pd.DataFrame(data)
    arrow_df = pa_from_dataframe(expected)
    result = from_dataframe(arrow_df)
    tm.assert_frame_equal(result, expected, check_column_type=False)


def test_multi_chunk_pyarrow() -> None:
    pa = pytest.importorskip("pyarrow", "11.0.0")
    n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
    names = ["n_legs"]
    table = pa.table([n_legs], names=names)
    with pytest.raises(
        RuntimeError,
        match="Cannot do zero copy conversion into multi-column DataFrame block",
    ):
        pd.api.interchange.from_dataframe(table, allow_copy=False)


def test_multi_chunk_column() -> None:
    pytest.importorskip("pyarrow", "11.0.0")
    ser = pd.Series([1, 2, None], dtype="Int64[pyarrow]")
    df = pd.concat([ser, ser], ignore_index=True).to_frame("a")
    df_orig = df.copy()
    with pytest.raises(
        RuntimeError, match="Found multi-chunk pyarrow array, but `allow_copy` is False"
    ):
        pd.api.interchange.from_dataframe(df.__dataframe__(allow_copy=False))
    result = pd.api.interchange.from_dataframe(df.__dataframe__(allow_copy=True))
    # Interchange protocol defaults to creating numpy-backed columns, so currently this
    # is 'float64'.
    expected = pd.DataFrame({"a": [1.0, 2.0, None, 1.0, 2.0, None]}, dtype="float64")
    tm.assert_frame_equal(result, expected)

    # Check that the rechunking we did didn't modify the original DataFrame.
    tm.assert_frame_equal(df, df_orig)
    assert len(df["a"].array._pa_array.chunks) == 2
    assert len(df_orig["a"].array._pa_array.chunks) == 2


def test_timestamp_ns_pyarrow():
    # GH 56712
    pytest.importorskip("pyarrow", "11.0.0")
    timestamp_args = {
        "year": 2000,
        "month": 1,
        "day": 1,
        "hour": 1,
        "minute": 1,
        "second": 1,
    }
    df = pd.Series(
        [datetime(**timestamp_args)],
        dtype="timestamp[ns][pyarrow]",
        name="col0",
    ).to_frame()

    dfi = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(dfi)["col0"].item()

    expected = pd.Timestamp(**timestamp_args)
    assert result == expected


@pytest.mark.parametrize("tz", ["UTC", "US/Pacific"])
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_datetimetzdtype(tz, unit):
    # GH 54239
    tz_data = (
        pd.date_range("2018-01-01", periods=5, freq="D").tz_localize(tz).as_unit(unit)
    )
    df = pd.DataFrame({"ts_tz": tz_data})
    tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))


def test_interchange_from_non_pandas_tz_aware(request):
    # GH 54239, 54287
    pa = pytest.importorskip("pyarrow", "11.0.0")
    import pyarrow.compute as pc

    if is_platform_windows() and is_ci_environment():
        mark = pytest.mark.xfail(
            raises=pa.ArrowInvalid,
            reason=(
                "TODO: Set ARROW_TIMEZONE_DATABASE environment variable "
                "on CI to path to the tzdata for pyarrow."
            ),
        )
        request.applymarker(mark)

    arr = pa.array([datetime(2020, 1, 1), None, datetime(2020, 1, 2)])
    arr = pc.assume_timezone(arr, "Asia/Kathmandu")
    table = pa.table({"arr": arr})
    exchange_df = table.__dataframe__()
    result = from_dataframe(exchange_df)

    expected = pd.DataFrame(
        ["2020-01-01 00:00:00+05:45", "NaT", "2020-01-02 00:00:00+05:45"],
        columns=["arr"],
        dtype="datetime64[us, Asia/Kathmandu]",
    )
    tm.assert_frame_equal(expected, result)


def test_interchange_from_corrected_buffer_dtypes(monkeypatch) -> None:
    # https://github.com/pandas-dev/pandas/issues/54781
    df = pd.DataFrame({"a": ["foo", "bar"]}).__dataframe__()
    interchange = df.__dataframe__()
    column = interchange.get_column_by_name("a")
    buffers = column.get_buffers()
    buffers_data = buffers["data"]
    buffer_dtype = buffers_data[1]
    buffer_dtype = (
        DtypeKind.UINT,
        8,
        ArrowCTypes.UINT8,
        buffer_dtype[3],
    )
    buffers["data"] = (buffers_data[0], buffer_dtype)
    column.get_buffers = lambda: buffers
    interchange.get_column_by_name = lambda _: column
    monkeypatch.setattr(df, "__dataframe__", lambda allow_copy: interchange)
    pd.api.interchange.from_dataframe(df)


def test_empty_string_column():
    # https://github.com/pandas-dev/pandas/issues/56703
    df = pd.DataFrame({"a": []}, dtype=str)
    df2 = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(df2)
    tm.assert_frame_equal(df, result)


def test_large_string():
    # GH#56702
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": ["x"]}, dtype="large_string[pyarrow]")
    result = pd.api.interchange.from_dataframe(df.__dataframe__())
    expected = pd.DataFrame({"a": ["x"]}, dtype="str")
    tm.assert_frame_equal(result, expected)


def test_non_str_names():
    # https://github.com/pandas-dev/pandas/issues/56701
    df = pd.Series([1, 2, 3], name=0).to_frame()
    names = df.__dataframe__().column_names()
    assert names == ["0"]


def test_non_str_names_w_duplicates():
    # https://github.com/pandas-dev/pandas/issues/56701
    df = pd.DataFrame({"0": [1, 2, 3], 0: [4, 5, 6]})
    dfi = df.__dataframe__()
    with pytest.raises(
        TypeError,
        match=(
            "Expected a Series, got a DataFrame. This likely happened because you "
            "called __dataframe__ on a DataFrame which, after converting column "
            r"names to string, resulted in duplicated names: Index\(\['0', '0'\], "
            r"dtype='(str|object)'\). Please rename these columns before using the "
            "interchange protocol."
        ),
    ):
        pd.api.interchange.from_dataframe(dfi, allow_copy=False)


@pytest.mark.parametrize(
    ("data", "dtype", "expected_dtype"),
    [
        ([1, 2, None], "Int64", "int64"),
        ([1, 2, None], "Int64[pyarrow]", "int64"),
        ([1, 2, None], "Int8", "int8"),
        ([1, 2, None], "Int8[pyarrow]", "int8"),
        (
            [1, 2, None],
            "UInt64",
            "uint64",
        ),
        (
            [1, 2, None],
            "UInt64[pyarrow]",
            "uint64",
        ),
        ([1.0, 2.25, None], "Float32", "float32"),
        ([1.0, 2.25, None], "Float32[pyarrow]", "float32"),
        ([True, False, None], "boolean", "bool"),
        ([True, False, None], "boolean[pyarrow]", "bool"),
        (["much ado", "about", None], pd.StringDtype(na_value=np.nan), "large_string"),
        (["much ado", "about", None], "string[pyarrow]", "large_string"),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), None],
            "timestamp[ns][pyarrow]",
            "timestamp[ns]",
        ),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), None],
            "timestamp[us][pyarrow]",
            "timestamp[us]",
        ),
        (
            [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                None,
            ],
            "timestamp[us, Asia/Kathmandu][pyarrow]",
            "timestamp[us, tz=Asia/Kathmandu]",
        ),
    ],
)
def test_pandas_nullable_with_missing_values(
    data: list, dtype: str, expected_dtype: str
) -> None:
    # https://github.com/pandas-dev/pandas/issues/57643
    # https://github.com/pandas-dev/pandas/issues/57664
    pa = pytest.importorskip("pyarrow", "11.0.0")
    import pyarrow.interchange as pai

    if expected_dtype == "timestamp[us, tz=Asia/Kathmandu]":
        expected_dtype = pa.timestamp("us", "Asia/Kathmandu")

    df = pd.DataFrame({"a": data}, dtype=dtype)
    result = pai.from_dataframe(df.__dataframe__())["a"]
    assert result.type == expected_dtype
    assert result[0].as_py() == data[0]
    assert result[1].as_py() == data[1]
    assert result[2].as_py() is None


@pytest.mark.parametrize(
    ("data", "dtype", "expected_dtype"),
    [
        ([1, 2, 3], "Int64", "int64"),
        ([1, 2, 3], "Int64[pyarrow]", "int64"),
        ([1, 2, 3], "Int8", "int8"),
        ([1, 2, 3], "Int8[pyarrow]", "int8"),
        (
            [1, 2, 3],
            "UInt64",
            "uint64",
        ),
        (
            [1, 2, 3],
            "UInt64[pyarrow]",
            "uint64",
        ),
        ([1.0, 2.25, 5.0], "Float32", "float32"),
        ([1.0, 2.25, 5.0], "Float32[pyarrow]", "float32"),
        ([True, False, False], "boolean", "bool"),
        ([True, False, False], "boolean[pyarrow]", "bool"),
        (
            ["much ado", "about", "nothing"],
            pd.StringDtype(na_value=np.nan),
            "large_string",
        ),
        (["much ado", "about", "nothing"], "string[pyarrow]", "large_string"),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "timestamp[ns][pyarrow]",
            "timestamp[ns]",
        ),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "timestamp[us][pyarrow]",
            "timestamp[us]",
        ),
        (
            [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                datetime(2020, 1, 3, tzinfo=timezone.utc),
            ],
            "timestamp[us, Asia/Kathmandu][pyarrow]",
            "timestamp[us, tz=Asia/Kathmandu]",
        ),
    ],
)
def test_pandas_nullable_without_missing_values(
    data: list, dtype: str, expected_dtype: str
) -> None:
    # https://github.com/pandas-dev/pandas/issues/57643
    pa = pytest.importorskip("pyarrow", "11.0.0")
    import pyarrow.interchange as pai

    if expected_dtype == "timestamp[us, tz=Asia/Kathmandu]":
        expected_dtype = pa.timestamp("us", "Asia/Kathmandu")

    df = pd.DataFrame({"a": data}, dtype=dtype)
    result = pai.from_dataframe(df.__dataframe__())["a"]
    assert result.type == expected_dtype
    assert result[0].as_py() == data[0]
    assert result[1].as_py() == data[1]
    assert result[2].as_py() == data[2]


def test_string_validity_buffer() -> None:
    # https://github.com/pandas-dev/pandas/issues/57761
    pytest.importorskip("pyarrow", "11.0.0")
    df = pd.DataFrame({"a": ["x"]}, dtype="large_string[pyarrow]")
    result = df.__dataframe__().get_column_by_name("a").get_buffers()["validity"]
    assert result is None


def test_string_validity_buffer_no_missing() -> None:
    # https://github.com/pandas-dev/pandas/issues/57762
    pytest.importorskip("pyarrow", "11.0.0")
    df = pd.DataFrame({"a": ["x", None]}, dtype="large_string[pyarrow]")
    validity = df.__dataframe__().get_column_by_name("a").get_buffers()["validity"]
    assert validity is not None
    result = validity[1]
    expected = (DtypeKind.BOOL, 1, ArrowCTypes.BOOL, "=")
    assert result == expected


def test_empty_dataframe():
    # https://github.com/pandas-dev/pandas/issues/56700
    df = pd.DataFrame({"a": []}, dtype="int8")
    dfi = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(dfi, allow_copy=False)
    expected = pd.DataFrame({"a": []}, dtype="int8")
    tm.assert_frame_equal(result, expected)


def test_from_dataframe_list_dtype():
    pa = pytest.importorskip("pyarrow", "14.0.0")
    data = {"a": [[1, 2], [4, 5, 6]]}
    tbl = pa.table(data)
    result = from_dataframe(tbl)
    expected = pd.DataFrame(data)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_solvers.py ===
import pytest
from sympy.core.function import expand_mul
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.matrices.exceptions import (ShapeError, NonSquareMatrixError)
from sympy.matrices import (
    ImmutableMatrix, Matrix, eye, ones, ImmutableDenseMatrix, dotprodsimp)
from sympy.matrices.determinant import _det_laplace
from sympy.testing.pytest import raises
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.polys.matrices.exceptions import DMShapeError
from sympy.solvers.solveset import linsolve
from sympy.abc import x, y

def test_issue_17247_expression_blowup_29():
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.gauss_jordan_solve(ones(4, 1)) == (Matrix(S('''[
            [                          -32549314808672/3306971225785 - 17397006745216*I/3306971225785],
            [                               67439348256/3306971225785 - 9167503335872*I/3306971225785],
            [-15091965363354518272/21217636514687010905 + 16890163109293858304*I/21217636514687010905],
            [                                                          -11328/952745 + 87616*I/952745]]''')), Matrix(0, 1, []))

def test_issue_17247_expression_blowup_30():
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.cholesky_solve(ones(4, 1)) == Matrix(S('''[
            [                          -32549314808672/3306971225785 - 17397006745216*I/3306971225785],
            [                               67439348256/3306971225785 - 9167503335872*I/3306971225785],
            [-15091965363354518272/21217636514687010905 + 16890163109293858304*I/21217636514687010905],
            [                                                          -11328/952745 + 87616*I/952745]]'''))

# @XFAIL # This calculation hangs with dotprodsimp.
# def test_issue_17247_expression_blowup_31():
#     M = Matrix([
#         [x + 1, 1 - x,     0,     0],
#         [1 - x, x + 1,     0, x + 1],
#         [    0, 1 - x, x + 1,     0],
#         [    0,     0,     0, x + 1]])
#     with dotprodsimp(True):
#         assert M.LDLsolve(ones(4, 1)) == Matrix([
#             [(x + 1)/(4*x)],
#             [(x - 1)/(4*x)],
#             [(x + 1)/(4*x)],
#             [    1/(x + 1)]])


def test_LUsolve_iszerofunc():
    # taken from https://github.com/sympy/sympy/issues/24679

    M = Matrix([[(x + 1)**2 - (x**2 + 2*x + 1), x], [x, 0]])
    b = Matrix([1, 1])
    is_zero_func = lambda e: False if e._random() else True

    x_exp = Matrix([1/x, (1-(-x**2 - 2*x + (x+1)**2 - 1)/x)/x])

    assert (x_exp - M.LUsolve(b, iszerofunc=is_zero_func)) == Matrix([0, 0])


def test_issue_17247_expression_blowup_32():
    M = Matrix([
        [x + 1, 1 - x,     0,     0],
        [1 - x, x + 1,     0, x + 1],
        [    0, 1 - x, x + 1,     0],
        [    0,     0,     0, x + 1]])
    with dotprodsimp(True):
        assert M.LUsolve(ones(4, 1)) == Matrix([
            [(x + 1)/(4*x)],
            [(x - 1)/(4*x)],
            [(x + 1)/(4*x)],
            [    1/(x + 1)]])

def test_LUsolve():
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.LUsolve(b)
    assert soln == x
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.LUsolve(b)
    assert soln == x
    A = Matrix([[2, 1], [1, 0], [1, 0]])   # issue 14548
    b = Matrix([3, 1, 1])
    assert A.LUsolve(b) == Matrix([1, 1])
    b = Matrix([3, 1, 2])                  # inconsistent
    raises(ValueError, lambda: A.LUsolve(b))
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4],
                [2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix([2, 1, -4])
    b = A*x
    soln = A.LUsolve(b)
    assert soln == x
    A = Matrix([[0, -1, 2], [5, 10, 7]])  # underdetermined
    x = Matrix([-1, 2, 0])
    b = A*x
    raises(NotImplementedError, lambda: A.LUsolve(b))

    A = Matrix(4, 4, lambda i, j: 1/(i+j+1) if i != 3 else 0)
    b = Matrix.zeros(4, 1)
    raises(NonInvertibleMatrixError, lambda: A.LUsolve(b))


def test_LUsolve_noncommutative():
    a0, a1, a2, a3 = symbols("a:4", commutative=False)
    b0, b1 = symbols("b:2", commutative=False)
    A = Matrix([[a0, a1], [a2, a3]])
    check = A * A.LUsolve(Matrix([b0, b1]))
    assert check[0, 0].expand() == b0
    # Because sympy simplification is very limited with noncommutative expressions,
    # perform an explicit check with the second element
    assert check[1, 0] == (
        a2*a0**(-1)*(-a1*(-a2*a0**(-1)*a1 + a3)**(-1)*(-a2*a0**(-1)*b0 + b1) + b0)
        + a3*(-a2*a0**(-1)*a1 + a3)**(-1)*(-a2*a0**(-1)*b0 + b1)
    )


def test_QRsolve():
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.QRsolve(b)
    assert soln == x
    x = Matrix([[1, 2], [3, 4], [5, 6]])
    b = A*x
    soln = A.QRsolve(b)
    assert soln == x

    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.QRsolve(b)
    assert soln == x
    x = Matrix([[7, 8], [9, 10], [11, 12]])
    b = A*x
    soln = A.QRsolve(b)
    assert soln == x

def test_errors():
    raises(ShapeError, lambda: Matrix([1]).LUsolve(Matrix([[1, 2], [3, 4]])))

def test_cholesky_solve():
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.cholesky_solve(b)
    assert soln == x
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.cholesky_solve(b)
    assert soln == x
    A = Matrix(((1, 5), (5, 1)))
    x = Matrix((4, -3))
    b = A*x
    soln = A.cholesky_solve(b)
    assert soln == x
    A = Matrix(((9, 3*I), (-3*I, 5)))
    x = Matrix((-2, 1))
    b = A*x
    soln = A.cholesky_solve(b)
    assert expand_mul(soln) == x
    A = Matrix(((9*I, 3), (-3 + I, 5)))
    x = Matrix((2 + 3*I, -1))
    b = A*x
    soln = A.cholesky_solve(b)
    assert expand_mul(soln) == x
    a00, a01, a11, b0, b1 = symbols('a00, a01, a11, b0, b1')
    A = Matrix(((a00, a01), (a01, a11)))
    b = Matrix((b0, b1))
    x = A.cholesky_solve(b)
    assert simplify(A*x) == b


def test_LDLsolve():
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.LDLsolve(b)
    assert soln == x

    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.LDLsolve(b)
    assert soln == x

    A = Matrix(((9, 3*I), (-3*I, 5)))
    x = Matrix((-2, 1))
    b = A*x
    soln = A.LDLsolve(b)
    assert expand_mul(soln) == x

    A = Matrix(((9*I, 3), (-3 + I, 5)))
    x = Matrix((2 + 3*I, -1))
    b = A*x
    soln = A.LDLsolve(b)
    assert expand_mul(soln) == x

    A = Matrix(((9, 3), (3, 9)))
    x = Matrix((1, 1))
    b = A * x
    soln = A.LDLsolve(b)
    assert expand_mul(soln) == x

    A = Matrix([[-5, -3, -4], [-3, -7, 7]])
    x = Matrix([[8], [7], [-2]])
    b = A * x
    raises(NotImplementedError, lambda: A.LDLsolve(b))


def test_lower_triangular_solve():

    raises(NonSquareMatrixError,
        lambda: Matrix([1, 0]).lower_triangular_solve(Matrix([0, 1])))
    raises(ShapeError,
        lambda: Matrix([[1, 0], [0, 1]]).lower_triangular_solve(Matrix([1])))
    raises(ValueError,
        lambda: Matrix([[2, 1], [1, 2]]).lower_triangular_solve(
            Matrix([[1, 0], [0, 1]])))

    A = Matrix([[1, 0], [0, 1]])
    B = Matrix([[x, y], [y, x]])
    C = Matrix([[4, 8], [2, 9]])

    assert A.lower_triangular_solve(B) == B
    assert A.lower_triangular_solve(C) == C


def test_upper_triangular_solve():

    raises(NonSquareMatrixError,
        lambda: Matrix([1, 0]).upper_triangular_solve(Matrix([0, 1])))
    raises(ShapeError,
        lambda: Matrix([[1, 0], [0, 1]]).upper_triangular_solve(Matrix([1])))
    raises(TypeError,
        lambda: Matrix([[2, 1], [1, 2]]).upper_triangular_solve(
            Matrix([[1, 0], [0, 1]])))

    A = Matrix([[1, 0], [0, 1]])
    B = Matrix([[x, y], [y, x]])
    C = Matrix([[2, 4], [3, 8]])

    assert A.upper_triangular_solve(B) == B
    assert A.upper_triangular_solve(C) == C


def test_diagonal_solve():
    raises(TypeError, lambda: Matrix([1, 1]).diagonal_solve(Matrix([1])))
    A = Matrix([[1, 0], [0, 1]])*2
    B = Matrix([[x, y], [y, x]])
    assert A.diagonal_solve(B) == B/2

    A = Matrix([[1, 0], [1, 2]])
    raises(TypeError, lambda: A.diagonal_solve(B))

def test_pinv_solve():
    # Fully determined system (unique result, identical to other solvers).
    A = Matrix([[1, 5], [7, 9]])
    B = Matrix([12, 13])
    assert A.pinv_solve(B) == A.cholesky_solve(B)
    assert A.pinv_solve(B) == A.LDLsolve(B)
    assert A.pinv_solve(B) == Matrix([sympify('-43/26'), sympify('71/26')])
    assert A * A.pinv() * B == B
    # Fully determined, with two-dimensional B matrix.
    B = Matrix([[12, 13, 14], [15, 16, 17]])
    assert A.pinv_solve(B) == A.cholesky_solve(B)
    assert A.pinv_solve(B) == A.LDLsolve(B)
    assert A.pinv_solve(B) == Matrix([[-33, -37, -41], [69, 75, 81]]) / 26
    assert A * A.pinv() * B == B
    # Underdetermined system (infinite results).
    A = Matrix([[1, 0, 1], [0, 1, 1]])
    B = Matrix([5, 7])
    solution = A.pinv_solve(B)
    w = {}
    for s in solution.atoms(Symbol):
        # Extract dummy symbols used in the solution.
        w[s.name] = s
    assert solution == Matrix([[w['w0_0']/3 + w['w1_0']/3 - w['w2_0']/3 + 1],
                               [w['w0_0']/3 + w['w1_0']/3 - w['w2_0']/3 + 3],
                               [-w['w0_0']/3 - w['w1_0']/3 + w['w2_0']/3 + 4]])
    assert A * A.pinv() * B == B
    # Overdetermined system (least squares results).
    A = Matrix([[1, 0], [0, 0], [0, 1]])
    B = Matrix([3, 2, 1])
    assert A.pinv_solve(B) == Matrix([3, 1])
    # Proof the solution is not exact.
    assert A * A.pinv() * B != B

def test_pinv_rank_deficient():
    # Test the four properties of the pseudoinverse for various matrices.
    As = [Matrix([[1, 1, 1], [2, 2, 2]]),
          Matrix([[1, 0], [0, 0]]),
          Matrix([[1, 2], [2, 4], [3, 6]])]

    for A in As:
        A_pinv = A.pinv(method="RD")
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA

    for A in As:
        A_pinv = A.pinv(method="ED")
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA

    # Test solving with rank-deficient matrices.
    A = Matrix([[1, 0], [0, 0]])
    # Exact, non-unique solution.
    B = Matrix([3, 0])
    solution = A.pinv_solve(B)
    w1 = solution.atoms(Symbol).pop()
    assert w1.name == 'w1_0'
    assert solution == Matrix([3, w1])
    assert A * A.pinv() * B == B
    # Least squares, non-unique solution.
    B = Matrix([3, 1])
    solution = A.pinv_solve(B)
    w1 = solution.atoms(Symbol).pop()
    assert w1.name == 'w1_0'
    assert solution == Matrix([3, w1])
    assert A * A.pinv() * B != B

def test_gauss_jordan_solve():

    # Square, full rank, unique solution
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    b = Matrix([3, 6, 9])
    sol, params = A.gauss_jordan_solve(b)
    assert sol == Matrix([[-1], [2], [0]])
    assert params == Matrix(0, 1, [])

    # Square, full rank, unique solution, B has more columns than rows
    A = eye(3)
    B = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    sol, params = A.gauss_jordan_solve(B)
    assert sol == B
    assert params == Matrix(0, 4, [])

    # Square, reduced rank, parametrized solution
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = Matrix([3, 6, 9])
    sol, params, freevar = A.gauss_jordan_solve(b, freevar=True)
    w = {}
    for s in sol.atoms(Symbol):
        # Extract dummy symbols used in the solution.
        w[s.name] = s
    assert sol == Matrix([[w['tau0'] - 1], [-2*w['tau0'] + 2], [w['tau0']]])
    assert params == Matrix([[w['tau0']]])
    assert freevar == [2]

    # Square, reduced rank, parametrized solution, B has two columns
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = Matrix([[3, 4], [6, 8], [9, 12]])
    sol, params, freevar = A.gauss_jordan_solve(B, freevar=True)
    w = {}
    for s in sol.atoms(Symbol):
        # Extract dummy symbols used in the solution.
        w[s.name] = s
    assert sol == Matrix([[w['tau0'] - 1, w['tau1'] - Rational(4, 3)],
                          [-2*w['tau0'] + 2, -2*w['tau1'] + Rational(8, 3)],
                          [w['tau0'], w['tau1']],])
    assert params == Matrix([[w['tau0'], w['tau1']]])
    assert freevar == [2]

    # Square, reduced rank, parametrized solution
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    b = Matrix([0, 0, 0])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[-2*w['tau0'] - 3*w['tau1']],
                         [w['tau0']], [w['tau1']]])
    assert params == Matrix([[w['tau0']], [w['tau1']]])

    # Square, reduced rank, parametrized solution
    A = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    b = Matrix([0, 0, 0])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[w['tau0']], [w['tau1']], [w['tau2']]])
    assert params == Matrix([[w['tau0']], [w['tau1']], [w['tau2']]])

    # Square, reduced rank, no solution
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    b = Matrix([0, 0, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Rectangular, tall, full rank, unique solution
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    b = Matrix([0, 0, 1, 0])
    sol, params = A.gauss_jordan_solve(b)
    assert sol == Matrix([[Rational(-1, 2)], [0], [Rational(1, 6)]])
    assert params == Matrix(0, 1, [])

    # Rectangular, tall, full rank, unique solution, B has less columns than rows
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0,0], [0, 0], [1, 2], [0, 0]])
    sol, params = A.gauss_jordan_solve(B)
    assert sol == Matrix([[Rational(-1, 2), Rational(-2, 2)], [0, 0], [Rational(1, 6), Rational(2, 6)]])
    assert params == Matrix(0, 2, [])

    # Rectangular, tall, full rank, no solution
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    b = Matrix([0, 0, 0, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Rectangular, tall, full rank, no solution, B has two columns (2nd has no solution)
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0,0], [0, 0], [1, 0], [0, 1]])
    raises(ValueError, lambda: A.gauss_jordan_solve(B))

    # Rectangular, tall, full rank, no solution, B has two columns (1st has no solution)
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0,0], [0, 0], [0, 1], [1, 0]])
    raises(ValueError, lambda: A.gauss_jordan_solve(B))

    # Rectangular, tall, reduced rank, parametrized solution
    A = Matrix([[1, 5, 3], [2, 10, 6], [3, 15, 9], [1, 4, 3]])
    b = Matrix([0, 0, 0, 1])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[-3*w['tau0'] + 5], [-1], [w['tau0']]])
    assert params == Matrix([[w['tau0']]])

    # Rectangular, tall, reduced rank, no solution
    A = Matrix([[1, 5, 3], [2, 10, 6], [3, 15, 9], [1, 4, 3]])
    b = Matrix([0, 0, 1, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Rectangular, wide, full rank, parametrized solution
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 1, 12]])
    b = Matrix([1, 1, 1])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[2*w['tau0'] - 1], [-3*w['tau0'] + 1], [0],
                         [w['tau0']]])
    assert params == Matrix([[w['tau0']]])

    # Rectangular, wide, reduced rank, parametrized solution
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]])
    b = Matrix([0, 1, 0])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[w['tau0'] + 2*w['tau1'] + S.Half],
                         [-2*w['tau0'] - 3*w['tau1'] - Rational(1, 4)],
                         [w['tau0']], [w['tau1']]])
    assert params == Matrix([[w['tau0']], [w['tau1']]])
    # watch out for clashing symbols
    x0, x1, x2, _x0 = symbols('_tau0 _tau1 _tau2 tau1')
    M = Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
    A = M[:, :-1]
    b = M[:, -1:]
    sol, params = A.gauss_jordan_solve(b)
    assert params == Matrix(3, 1, [x0, x1, x2])
    assert sol == Matrix(5, 1, [x0, 0, x1, _x0, x2])

    # Rectangular, wide, reduced rank, no solution
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]])
    b = Matrix([1, 1, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Test for immutable matrix
    A = ImmutableMatrix([[1, 0], [0, 1]])
    B = ImmutableMatrix([1, 2])
    sol, params = A.gauss_jordan_solve(B)
    assert sol == ImmutableMatrix([1, 2])
    assert params == ImmutableMatrix(0, 1, [])
    assert sol.__class__ == ImmutableDenseMatrix
    assert params.__class__ == ImmutableDenseMatrix

    # Test placement of free variables
    A = Matrix([[1, 0, 0, 0], [0, 0, 0, 1]])
    b = Matrix([1, 1])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[1], [w['tau0']], [w['tau1']], [1]])
    assert params == Matrix([[w['tau0']], [w['tau1']]])


def test_linsolve_underdetermined_AND_gauss_jordan_solve():
    #Test placement of free variables as per issue 19815
    A = Matrix([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    B  = Matrix([1, 2, 1, 1, 1, 1, 1, 2])
    sol, params = A.gauss_jordan_solve(B)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert params == Matrix([[w['tau0']], [w['tau1']], [w['tau2']],
                             [w['tau3']], [w['tau4']], [w['tau5']]])
    assert sol == Matrix([[1 - 1*w['tau2']],
                          [w['tau2']],
                          [1 - 1*w['tau0'] + w['tau1']],
                          [w['tau0']],
                          [w['tau3'] + w['tau4']],
                          [-1*w['tau3'] - 1*w['tau4'] - 1*w['tau1']],
                          [1 - 1*w['tau2']],
                          [w['tau1']],
                          [w['tau2']],
                          [w['tau3']],
                          [w['tau4']],
                          [1 - 1*w['tau5']],
                          [w['tau5']],
                          [1]])

    from sympy.abc import j,f
    # https://github.com/sympy/sympy/issues/20046
    A = Matrix([
    [1,  1, 1,  1, 1,  1, 1,  1,  1],
    [0, -1, 0, -1, 0, -1, 0, -1, -j],
    [0,  0, 0,  0, 1,  1, 1,  1,  f]
    ])

    sol_1=Matrix(list(linsolve(A))[0])

    tau0, tau1, tau2, tau3, tau4 = symbols('tau:5')

    assert sol_1 == Matrix([[-f - j - tau0 + tau2 + tau4 + 1],
                          [j - tau1 - tau2 - tau4],
                          [tau0],
                          [tau1],
                          [f - tau2 - tau3 - tau4],
                          [tau2],
                          [tau3],
                          [tau4]])

    # https://github.com/sympy/sympy/issues/19815
    sol_2 = A[:, : -1 ] * sol_1 - A[:, -1 ]
    assert sol_2 == Matrix([[0], [0], [0]])


@pytest.mark.parametrize("det_method", ["bird", "laplace"])
@pytest.mark.parametrize("M, rhs", [
    (Matrix([[2, 3, 5], [3, 6, 2], [8, 3, 6]]), Matrix(3, 1, [3, 7, 5])),
    (Matrix([[2, 3, 5], [3, 6, 2], [8, 3, 6]]),
     Matrix([[1, 2], [3, 4], [5, 6]])),
    (Matrix(2, 2, symbols("a:4")), Matrix(2, 1, symbols("b:2"))),
])
def test_cramer_solve(det_method, M, rhs):
    assert simplify(M.cramer_solve(rhs, det_method=det_method) - M.LUsolve(rhs)
                    ) == Matrix.zeros(M.rows, rhs.cols)


@pytest.mark.parametrize("det_method, error", [
    ("bird", DMShapeError), (_det_laplace, NonSquareMatrixError)])
def test_cramer_solve_errors(det_method, error):
    # Non-square matrix
    A = Matrix([[0, -1, 2], [5, 10, 7]])
    b = Matrix([-2, 15])
    raises(error, lambda: A.cramer_solve(b, det_method=det_method))


def test_solve():
    A = Matrix([[1,2], [2,4]])
    b = Matrix([[3], [4]])
    raises(ValueError, lambda: A.solve(b)) #no solution
    b = Matrix([[ 4], [8]])
    raises(ValueError, lambda: A.solve(b)) #infinite solution

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_ellipse.py ===
from sympy.core import expand
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sec
from sympy.geometry.line import Segment2D
from sympy.geometry.point import Point2D
from sympy.geometry import (Circle, Ellipse, GeometryError, Line, Point,
                            Polygon, Ray, RegularPolygon, Segment,
                            Triangle, intersection)
from sympy.testing.pytest import raises, slow
from sympy.integrals.integrals import integrate
from sympy.functions.special.elliptic_integrals import elliptic_e
from sympy.functions.elementary.miscellaneous import Max


def test_ellipse_equation_using_slope():
    from sympy.abc import x, y

    e1 = Ellipse(Point(1, 0), 3, 2)
    assert str(e1.equation(_slope=1)) == str((-x + y + 1)**2/8 + (x + y - 1)**2/18 - 1)

    e2 = Ellipse(Point(0, 0), 4, 1)
    assert str(e2.equation(_slope=1)) == str((-x + y)**2/2 + (x + y)**2/32 - 1)

    e3 = Ellipse(Point(1, 5), 6, 2)
    assert str(e3.equation(_slope=2)) == str((-2*x + y - 3)**2/20 + (x + 2*y - 11)**2/180 - 1)


def test_object_from_equation():
    from sympy.abc import x, y, a, b, c, d, e
    assert Circle(x**2 + y**2 + 3*x + 4*y - 8) == Circle(Point2D(S(-3) / 2, -2), sqrt(57) / 2)
    assert Circle(x**2 + y**2 + 6*x + 8*y + 25) == Circle(Point2D(-3, -4), 0)
    assert Circle(a**2 + b**2 + 6*a + 8*b + 25, x='a', y='b') == Circle(Point2D(-3, -4), 0)
    assert Circle(x**2 + y**2 - 25) == Circle(Point2D(0, 0), 5)
    assert Circle(x**2 + y**2) == Circle(Point2D(0, 0), 0)
    assert Circle(a**2 + b**2, x='a', y='b') == Circle(Point2D(0, 0), 0)
    assert Circle(x**2 + y**2 + 6*x + 8) == Circle(Point2D(-3, 0), 1)
    assert Circle(x**2 + y**2 + 6*y + 8) == Circle(Point2D(0, -3), 1)
    assert Circle((x - 1)**2 + y**2 - 9) == Circle(Point2D(1, 0), 3)
    assert Circle(6*(x**2) + 6*(y**2) + 6*x + 8*y - 25) == Circle(Point2D(Rational(-1, 2), Rational(-2, 3)), 5*sqrt(7)/6)
    assert Circle(Eq(a**2 + b**2, 25), x='a', y=b) == Circle(Point2D(0, 0), 5)
    raises(GeometryError, lambda: Circle(x**2 + y**2 + 3*x + 4*y + 26))
    raises(GeometryError, lambda: Circle(x**2 + y**2 + 25))
    raises(GeometryError, lambda: Circle(a**2 + b**2 + 25, x='a', y='b'))
    raises(GeometryError, lambda: Circle(x**2 + 6*y + 8))
    raises(GeometryError, lambda: Circle(6*(x ** 2) + 4*(y**2) + 6*x + 8*y + 25))
    raises(ValueError, lambda: Circle(a**2 + b**2 + 3*a + 4*b - 8))
    # .equation() adds 'real=True' assumption; '==' would fail if assumptions differed
    x, y = symbols('x y', real=True)
    eq = a*x**2 + a*y**2 + c*x + d*y + e
    assert expand(Circle(eq).equation()*a) == eq


@slow
def test_ellipse_geom():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    t = Symbol('t', real=True)
    y1 = Symbol('y1', real=True)
    half = S.Half
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p4 = Point(0, 1)

    e1 = Ellipse(p1, 1, 1)
    e2 = Ellipse(p2, half, 1)
    e3 = Ellipse(p1, y1, y1)
    c1 = Circle(p1, 1)
    c2 = Circle(p2, 1)
    c3 = Circle(Point(sqrt(2), sqrt(2)), 1)
    l1 = Line(p1, p2)

    # Test creation with three points
    cen, rad = Point(3*half, 2), 5*half
    assert Circle(Point(0, 0), Point(3, 0), Point(0, 4)) == Circle(cen, rad)
    assert Circle(Point(0, 0), Point(1, 1), Point(2, 2)) == Segment2D(Point2D(0, 0), Point2D(2, 2))

    raises(ValueError, lambda: Ellipse(None, None, None, 1))
    raises(ValueError, lambda: Ellipse())
    raises(GeometryError, lambda: Circle(Point(0, 0)))
    raises(GeometryError, lambda: Circle(Symbol('x')*Symbol('y')))

    # Basic Stuff
    assert Ellipse(None, 1, 1).center == Point(0, 0)
    assert e1 == c1
    assert e1 != e2
    assert e1 != l1
    assert p4 in e1
    assert e1 in e1
    assert e2 in e2
    assert 1 not in e2
    assert p2 not in e2
    assert e1.area == pi
    assert e2.area == pi/2
    assert e3.area == pi*y1*abs(y1)
    assert c1.area == e1.area
    assert c1.circumference == e1.circumference
    assert e3.circumference == 2*pi*y1
    assert e1.plot_interval() == e2.plot_interval() == [t, -pi, pi]
    assert e1.plot_interval(x) == e2.plot_interval(x) == [x, -pi, pi]

    assert c1.minor == 1
    assert c1.major == 1
    assert c1.hradius == 1
    assert c1.vradius == 1

    assert Ellipse((1, 1), 0, 0) == Point(1, 1)
    assert Ellipse((1, 1), 1, 0) == Segment(Point(0, 1), Point(2, 1))
    assert Ellipse((1, 1), 0, 1) == Segment(Point(1, 0), Point(1, 2))

    # Private Functions
    assert hash(c1) == hash(Circle(Point(1, 0), Point(0, 1), Point(0, -1)))
    assert c1 in e1
    assert (Line(p1, p2) in e1) is False
    assert e1.__cmp__(e1) == 0
    assert e1.__cmp__(Point(0, 0)) > 0

    # Encloses
    assert e1.encloses(Segment(Point(-0.5, -0.5), Point(0.5, 0.5))) is True
    assert e1.encloses(Line(p1, p2)) is False
    assert e1.encloses(Ray(p1, p2)) is False
    assert e1.encloses(e1) is False
    assert e1.encloses(
        Polygon(Point(-0.5, -0.5), Point(-0.5, 0.5), Point(0.5, 0.5))) is True
    assert e1.encloses(RegularPolygon(p1, 0.5, 3)) is True
    assert e1.encloses(RegularPolygon(p1, 5, 3)) is False
    assert e1.encloses(RegularPolygon(p2, 5, 3)) is False

    assert e2.arbitrary_point() in e2
    raises(ValueError, lambda: Ellipse(Point(x, y), 1, 1).arbitrary_point(parameter='x'))

    # Foci
    f1, f2 = Point(sqrt(12), 0), Point(-sqrt(12), 0)
    ef = Ellipse(Point(0, 0), 4, 2)
    assert ef.foci in [(f1, f2), (f2, f1)]

    # Tangents
    v = sqrt(2) / 2
    p1_1 = Point(v, v)
    p1_2 = p2 + Point(half, 0)
    p1_3 = p2 + Point(0, 1)
    assert e1.tangent_lines(p4) == c1.tangent_lines(p4)
    assert e2.tangent_lines(p1_2) == [Line(Point(Rational(3, 2), 1), Point(Rational(3, 2), S.Half))]
    assert e2.tangent_lines(p1_3) == [Line(Point(1, 2), Point(Rational(5, 4), 2))]
    assert c1.tangent_lines(p1_1) != [Line(p1_1, Point(0, sqrt(2)))]
    assert c1.tangent_lines(p1) == []
    assert e2.is_tangent(Line(p1_2, p2 + Point(half, 1)))
    assert e2.is_tangent(Line(p1_3, p2 + Point(half, 1)))
    assert c1.is_tangent(Line(p1_1, Point(0, sqrt(2))))
    assert e1.is_tangent(Line(Point(0, 0), Point(1, 1))) is False
    assert c1.is_tangent(e1) is True
    assert c1.is_tangent(Ellipse(Point(2, 0), 1, 1)) is True
    assert c1.is_tangent(
        Polygon(Point(1, 1), Point(1, -1), Point(2, 0))) is False
    assert c1.is_tangent(
        Polygon(Point(1, 1), Point(1, 0), Point(2, 0))) is False
    assert Circle(Point(5, 5), 3).is_tangent(Circle(Point(0, 5), 1)) is False

    assert Ellipse(Point(5, 5), 2, 1).tangent_lines(Point(0, 0)) == \
        [Line(Point(0, 0), Point(Rational(77, 25), Rational(132, 25))),
     Line(Point(0, 0), Point(Rational(33, 5), Rational(22, 5)))]
    assert Ellipse(Point(5, 5), 2, 1).tangent_lines(Point(3, 4)) == \
        [Line(Point(3, 4), Point(4, 4)), Line(Point(3, 4), Point(3, 5))]
    assert Circle(Point(5, 5), 2).tangent_lines(Point(3, 3)) == \
        [Line(Point(3, 3), Point(4, 3)), Line(Point(3, 3), Point(3, 4))]
    assert Circle(Point(5, 5), 2).tangent_lines(Point(5 - 2*sqrt(2), 5)) == \
        [Line(Point(5 - 2*sqrt(2), 5), Point(5 - sqrt(2), 5 - sqrt(2))),
     Line(Point(5 - 2*sqrt(2), 5), Point(5 - sqrt(2), 5 + sqrt(2))), ]
    assert Circle(Point(5, 5), 5).tangent_lines(Point(4, 0)) == \
        [Line(Point(4, 0), Point(Rational(40, 13), Rational(5, 13))),
     Line(Point(4, 0), Point(5, 0))]
    assert Circle(Point(5, 5), 5).tangent_lines(Point(0, 6)) == \
        [Line(Point(0, 6), Point(0, 7)),
        Line(Point(0, 6), Point(Rational(5, 13), Rational(90, 13)))]

    # for numerical calculations, we shouldn't demand exact equality,
    # so only test up to the desired precision
    def lines_close(l1, l2, prec):
        """ tests whether l1 and 12 are within 10**(-prec)
        of each other """
        return abs(l1.p1 - l2.p1) < 10**(-prec) and abs(l1.p2 - l2.p2) < 10**(-prec)
    def line_list_close(ll1, ll2, prec):
        return all(lines_close(l1, l2, prec) for l1, l2 in zip(ll1, ll2))

    e = Ellipse(Point(0, 0), 2, 1)
    assert e.normal_lines(Point(0, 0)) == \
        [Line(Point(0, 0), Point(0, 1)), Line(Point(0, 0), Point(1, 0))]
    assert e.normal_lines(Point(1, 0)) == \
        [Line(Point(0, 0), Point(1, 0))]
    assert e.normal_lines((0, 1)) == \
        [Line(Point(0, 0), Point(0, 1))]
    assert line_list_close(e.normal_lines(Point(1, 1), 2), [
        Line(Point(Rational(-51, 26), Rational(-1, 5)), Point(Rational(-25, 26), Rational(17, 83))),
        Line(Point(Rational(28, 29), Rational(-7, 8)), Point(Rational(57, 29), Rational(-9, 2)))], 2)
    # test the failure of Poly.intervals and checks a point on the boundary
    p = Point(sqrt(3), S.Half)
    assert p in e
    assert line_list_close(e.normal_lines(p, 2), [
        Line(Point(Rational(-341, 171), Rational(-1, 13)), Point(Rational(-170, 171), Rational(5, 64))),
        Line(Point(Rational(26, 15), Rational(-1, 2)), Point(Rational(41, 15), Rational(-43, 26)))], 2)
    # be sure to use the slope that isn't undefined on boundary
    e = Ellipse((0, 0), 2, 2*sqrt(3)/3)
    assert line_list_close(e.normal_lines((1, 1), 2), [
        Line(Point(Rational(-64, 33), Rational(-20, 71)), Point(Rational(-31, 33), Rational(2, 13))),
        Line(Point(1, -1), Point(2, -4))], 2)
    # general ellipse fails except under certain conditions
    e = Ellipse((0, 0), x, 1)
    assert e.normal_lines((x + 1, 0)) == [Line(Point(0, 0), Point(1, 0))]
    raises(NotImplementedError, lambda: e.normal_lines((x + 1, 1)))
    # Properties
    major = 3
    minor = 1
    e4 = Ellipse(p2, minor, major)
    assert e4.focus_distance == sqrt(major**2 - minor**2)
    ecc = e4.focus_distance / major
    assert e4.eccentricity == ecc
    assert e4.periapsis == major*(1 - ecc)
    assert e4.apoapsis == major*(1 + ecc)
    assert e4.semilatus_rectum == major*(1 - ecc ** 2)
    # independent of orientation
    e4 = Ellipse(p2, major, minor)
    assert e4.focus_distance == sqrt(major**2 - minor**2)
    ecc = e4.focus_distance / major
    assert e4.eccentricity == ecc
    assert e4.periapsis == major*(1 - ecc)
    assert e4.apoapsis == major*(1 + ecc)

    # Intersection
    l1 = Line(Point(1, -5), Point(1, 5))
    l2 = Line(Point(-5, -1), Point(5, -1))
    l3 = Line(Point(-1, -1), Point(1, 1))
    l4 = Line(Point(-10, 0), Point(0, 10))
    pts_c1_l3 = [Point(sqrt(2)/2, sqrt(2)/2), Point(-sqrt(2)/2, -sqrt(2)/2)]

    assert intersection(e2, l4) == []
    assert intersection(c1, Point(1, 0)) == [Point(1, 0)]
    assert intersection(c1, l1) == [Point(1, 0)]
    assert intersection(c1, l2) == [Point(0, -1)]
    assert intersection(c1, l3) in [pts_c1_l3, [pts_c1_l3[1], pts_c1_l3[0]]]
    assert intersection(c1, c2) == [Point(0, 1), Point(1, 0)]
    assert intersection(c1, c3) == [Point(sqrt(2)/2, sqrt(2)/2)]
    assert e1.intersection(l1) == [Point(1, 0)]
    assert e2.intersection(l4) == []
    assert e1.intersection(Circle(Point(0, 2), 1)) == [Point(0, 1)]
    assert e1.intersection(Circle(Point(5, 0), 1)) == []
    assert e1.intersection(Ellipse(Point(2, 0), 1, 1)) == [Point(1, 0)]
    assert e1.intersection(Ellipse(Point(5, 0), 1, 1)) == []
    assert e1.intersection(Point(2, 0)) == []
    assert e1.intersection(e1) == e1
    assert intersection(Ellipse(Point(0, 0), 2, 1), Ellipse(Point(3, 0), 1, 2)) == [Point(2, 0)]
    assert intersection(Circle(Point(0, 0), 2), Circle(Point(3, 0), 1)) == [Point(2, 0)]
    assert intersection(Circle(Point(0, 0), 2), Circle(Point(7, 0), 1)) == []
    assert intersection(Ellipse(Point(0, 0), 5, 17), Ellipse(Point(4, 0), 1, 0.2)
        ) == [Point(5.0, 0, evaluate=False)]
    assert intersection(Ellipse(Point(0, 0), 5, 17), Ellipse(Point(4, 0), 0.999, 0.2)) == []
    assert Circle((0, 0), S.Half).intersection(
        Triangle((-1, 0), (1, 0), (0, 1))) == [
        Point(Rational(-1, 2), 0), Point(S.Half, 0)]
    raises(TypeError, lambda: intersection(e2, Line((0, 0, 0), (0, 0, 1))))
    raises(TypeError, lambda: intersection(e2, Rational(12)))
    raises(TypeError, lambda: Ellipse.intersection(e2, 1))
    # some special case intersections
    csmall = Circle(p1, 3)
    cbig = Circle(p1, 5)
    cout = Circle(Point(5, 5), 1)
    # one circle inside of another
    assert csmall.intersection(cbig) == []
    # separate circles
    assert csmall.intersection(cout) == []
    # coincident circles
    assert csmall.intersection(csmall) == csmall

    v = sqrt(2)
    t1 = Triangle(Point(0, v), Point(0, -v), Point(v, 0))
    points = intersection(t1, c1)
    assert len(points) == 4
    assert Point(0, 1) in points
    assert Point(0, -1) in points
    assert Point(v/2, v/2) in points
    assert Point(v/2, -v/2) in points

    circ = Circle(Point(0, 0), 5)
    elip = Ellipse(Point(0, 0), 5, 20)
    assert intersection(circ, elip) in \
        [[Point(5, 0), Point(-5, 0)], [Point(-5, 0), Point(5, 0)]]
    assert elip.tangent_lines(Point(0, 0)) == []
    elip = Ellipse(Point(0, 0), 3, 2)
    assert elip.tangent_lines(Point(3, 0)) == \
        [Line(Point(3, 0), Point(3, -12))]

    e1 = Ellipse(Point(0, 0), 5, 10)
    e2 = Ellipse(Point(2, 1), 4, 8)
    a = Rational(53, 17)
    c = 2*sqrt(3991)/17
    ans = [Point(a - c/8, a/2 + c), Point(a + c/8, a/2 - c)]
    assert e1.intersection(e2) == ans
    e2 = Ellipse(Point(x, y), 4, 8)
    c = sqrt(3991)
    ans = [Point(-c/68 + a, c*Rational(2, 17) + a/2), Point(c/68 + a, c*Rational(-2, 17) + a/2)]
    assert [p.subs({x: 2, y:1}) for p in e1.intersection(e2)] == ans

    # Combinations of above
    assert e3.is_tangent(e3.tangent_lines(p1 + Point(y1, 0))[0])

    e = Ellipse((1, 2), 3, 2)
    assert e.tangent_lines(Point(10, 0)) == \
        [Line(Point(10, 0), Point(1, 0)),
        Line(Point(10, 0), Point(Rational(14, 5), Rational(18, 5)))]

    # encloses_point
    e = Ellipse((0, 0), 1, 2)
    assert e.encloses_point(e.center)
    assert e.encloses_point(e.center + Point(0, e.vradius - Rational(1, 10)))
    assert e.encloses_point(e.center + Point(e.hradius - Rational(1, 10), 0))
    assert e.encloses_point(e.center + Point(e.hradius, 0)) is False
    assert e.encloses_point(
        e.center + Point(e.hradius + Rational(1, 10), 0)) is False
    e = Ellipse((0, 0), 2, 1)
    assert e.encloses_point(e.center)
    assert e.encloses_point(e.center + Point(0, e.vradius - Rational(1, 10)))
    assert e.encloses_point(e.center + Point(e.hradius - Rational(1, 10), 0))
    assert e.encloses_point(e.center + Point(e.hradius, 0)) is False
    assert e.encloses_point(
        e.center + Point(e.hradius + Rational(1, 10), 0)) is False
    assert c1.encloses_point(Point(1, 0)) is False
    assert c1.encloses_point(Point(0.3, 0.4)) is True

    assert e.scale(2, 3) == Ellipse((0, 0), 4, 3)
    assert e.scale(3, 6) == Ellipse((0, 0), 6, 6)
    assert e.rotate(pi) == e
    assert e.rotate(pi, (1, 2)) == Ellipse(Point(2, 4), 2, 1)
    raises(NotImplementedError, lambda: e.rotate(pi/3))

    # Circle rotation tests (Issue #11743)
    # Link - https://github.com/sympy/sympy/issues/11743
    cir = Circle(Point(1, 0), 1)
    assert cir.rotate(pi/2) == Circle(Point(0, 1), 1)
    assert cir.rotate(pi/3) == Circle(Point(S.Half, sqrt(3)/2), 1)
    assert cir.rotate(pi/3, Point(1, 0)) == Circle(Point(1, 0), 1)
    assert cir.rotate(pi/3, Point(0, 1)) == Circle(Point(S.Half + sqrt(3)/2, S.Half + sqrt(3)/2), 1)


def test_construction():
    e1 = Ellipse(hradius=2, vradius=1, eccentricity=None)
    assert e1.eccentricity == sqrt(3)/2

    e2 = Ellipse(hradius=2, vradius=None, eccentricity=sqrt(3)/2)
    assert e2.vradius == 1

    e3 = Ellipse(hradius=None, vradius=1, eccentricity=sqrt(3)/2)
    assert e3.hradius == 2

    # filter(None, iterator) filters out anything falsey, including 0
    # eccentricity would be filtered out in this case and the constructor would throw an error
    e4 = Ellipse(Point(0, 0), hradius=1, eccentricity=0)
    assert e4.vradius == 1

    #tests for eccentricity > 1
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity = S(3)/2))
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity=sec(5)))
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity=S.Pi-S(2)))

    #tests for eccentricity = 1
    #if vradius is not defined
    assert Ellipse(None, 1, None, 1).length == 2
    #if hradius is not defined
    raises(GeometryError, lambda: Ellipse(None, None, 1, eccentricity = 1))

    #tests for eccentricity < 0
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity = -3))
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity = -0.5))

def test_ellipse_random_point():
    y1 = Symbol('y1', real=True)
    e3 = Ellipse(Point(0, 0), y1, y1)
    rx, ry = Symbol('rx'), Symbol('ry')
    for ind in range(0, 5):
        r = e3.random_point()
        # substitution should give zero*y1**2
        assert e3.equation(rx, ry).subs(zip((rx, ry), r.args)).equals(0)
    # test for the case with seed
    r = e3.random_point(seed=1)
    assert e3.equation(rx, ry).subs(zip((rx, ry), r.args)).equals(0)


def test_repr():
    assert repr(Circle((0, 1), 2)) == 'Circle(Point2D(0, 1), 2)'


def test_transform():
    c = Circle((1, 1), 2)
    assert c.scale(-1) == Circle((-1, 1), 2)
    assert c.scale(y=-1) == Circle((1, -1), 2)
    assert c.scale(2) == Ellipse((2, 1), 4, 2)

    assert Ellipse((0, 0), 2, 3).scale(2, 3, (4, 5)) == \
        Ellipse(Point(-4, -10), 4, 9)
    assert Circle((0, 0), 2).scale(2, 3, (4, 5)) == \
        Ellipse(Point(-4, -10), 4, 6)
    assert Ellipse((0, 0), 2, 3).scale(3, 3, (4, 5)) == \
        Ellipse(Point(-8, -10), 6, 9)
    assert Circle((0, 0), 2).scale(3, 3, (4, 5)) == \
        Circle(Point(-8, -10), 6)
    assert Circle(Point(-8, -10), 6).scale(Rational(1, 3), Rational(1, 3), (4, 5)) == \
        Circle((0, 0), 2)
    assert Circle((0, 0), 2).translate(4, 5) == \
        Circle((4, 5), 2)
    assert Circle((0, 0), 2).scale(3, 3) == \
        Circle((0, 0), 6)


def test_bounds():
    e1 = Ellipse(Point(0, 0), 3, 5)
    e2 = Ellipse(Point(2, -2), 7, 7)
    c1 = Circle(Point(2, -2), 7)
    c2 = Circle(Point(-2, 0), Point(0, 2), Point(2, 0))
    assert e1.bounds == (-3, -5, 3, 5)
    assert e2.bounds == (-5, -9, 9, 5)
    assert c1.bounds == (-5, -9, 9, 5)
    assert c2.bounds == (-2, -2, 2, 2)


def test_reflect():
    b = Symbol('b')
    m = Symbol('m')
    l = Line((0, b), slope=m)
    t1 = Triangle((0, 0), (1, 0), (2, 3))
    assert t1.area == -t1.reflect(l).area
    e = Ellipse((1, 0), 1, 2)
    assert e.area == -e.reflect(Line((1, 0), slope=0)).area
    assert e.area == -e.reflect(Line((1, 0), slope=oo)).area
    raises(NotImplementedError, lambda: e.reflect(Line((1, 0), slope=m)))
    assert Circle((0, 1), 1).reflect(Line((0, 0), (1, 1))) == Circle(Point2D(1, 0), -1)


def test_is_tangent():
    e1 = Ellipse(Point(0, 0), 3, 5)
    c1 = Circle(Point(2, -2), 7)
    assert e1.is_tangent(Point(0, 0)) is False
    assert e1.is_tangent(Point(3, 0)) is False
    assert e1.is_tangent(e1) is True
    assert e1.is_tangent(Ellipse((0, 0), 1, 2)) is False
    assert e1.is_tangent(Ellipse((0, 0), 3, 2)) is True
    assert c1.is_tangent(Ellipse((2, -2), 7, 1)) is True
    assert c1.is_tangent(Circle((11, -2), 2)) is True
    assert c1.is_tangent(Circle((7, -2), 2)) is True
    assert c1.is_tangent(Ray((-5, -2), (-15, -20))) is False
    assert c1.is_tangent(Ray((-3, -2), (-15, -20))) is False
    assert c1.is_tangent(Ray((-3, -22), (15, 20))) is False
    assert c1.is_tangent(Ray((9, 20), (9, -20))) is True
    assert c1.is_tangent(Ray((2, 5), (9, 5))) is True
    assert c1.is_tangent(Segment((2, 5), (9, 5))) is True
    assert e1.is_tangent(Segment((2, 2), (-7, 7))) is False
    assert e1.is_tangent(Segment((0, 0), (1, 2))) is False
    assert c1.is_tangent(Segment((0, 0), (-5, -2))) is False
    assert e1.is_tangent(Segment((3, 0), (12, 12))) is False
    assert e1.is_tangent(Segment((12, 12), (3, 0))) is False
    assert e1.is_tangent(Segment((-3, 0), (3, 0))) is False
    assert e1.is_tangent(Segment((-3, 5), (3, 5))) is True
    assert e1.is_tangent(Line((10, 0), (10, 10))) is False
    assert e1.is_tangent(Line((0, 0), (1, 1))) is False
    assert e1.is_tangent(Line((-3, 0), (-2.99, -0.001))) is False
    assert e1.is_tangent(Line((-3, 0), (-3, 1))) is True
    assert e1.is_tangent(Polygon((0, 0), (5, 5), (5, -5))) is False
    assert e1.is_tangent(Polygon((-100, -50), (-40, -334), (-70, -52))) is False
    assert e1.is_tangent(Polygon((-3, 0), (3, 0), (0, 1))) is False
    assert e1.is_tangent(Polygon((-3, 0), (3, 0), (0, 5))) is False
    assert e1.is_tangent(Polygon((-3, 0), (0, -5), (3, 0), (0, 5))) is False
    assert e1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is True
    assert c1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is False
    assert e1.is_tangent(Polygon((0, 0), (3, 0), (7, 7), (0, 5))) is False
    assert e1.is_tangent(Polygon((3, 12), (3, -12), (6, 5))) is False
    assert e1.is_tangent(Polygon((3, 12), (3, -12), (0, -5), (0, 5))) is False
    assert e1.is_tangent(Polygon((3, 0), (5, 7), (6, -5))) is False
    assert c1.is_tangent(Segment((0, 0), (-5, -2))) is False
    assert e1.is_tangent(Segment((-3, 0), (3, 0))) is False
    assert e1.is_tangent(Segment((-3, 5), (3, 5))) is True
    assert e1.is_tangent(Polygon((0, 0), (5, 5), (5, -5))) is False
    assert e1.is_tangent(Polygon((-100, -50), (-40, -334), (-70, -52))) is False
    assert e1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is True
    assert c1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is False
    assert e1.is_tangent(Polygon((3, 12), (3, -12), (0, -5), (0, 5))) is False
    assert e1.is_tangent(Polygon((3, 0), (5, 7), (6, -5))) is False
    raises(TypeError, lambda: e1.is_tangent(Point(0, 0, 0)))
    raises(TypeError, lambda: e1.is_tangent(Rational(5)))


def test_parameter_value():
    t = Symbol('t')
    e = Ellipse(Point(0, 0), 3, 5)
    assert e.parameter_value((3, 0), t) == {t: 0}
    raises(ValueError, lambda: e.parameter_value((4, 0), t))


@slow
def test_second_moment_of_area():
    x, y = symbols('x, y')
    e = Ellipse(Point(0, 0), 5, 4)
    I_yy = 2*4*integrate(sqrt(25 - x**2)*x**2, (x, -5, 5))/5
    I_xx = 2*5*integrate(sqrt(16 - y**2)*y**2, (y, -4, 4))/4
    Y = 3*sqrt(1 - x**2/5**2)
    I_xy = integrate(integrate(y, (y, -Y, Y))*x, (x, -5, 5))
    assert I_yy == e.second_moment_of_area()[1]
    assert I_xx == e.second_moment_of_area()[0]
    assert I_xy == e.second_moment_of_area()[2]
    #checking for other point
    t1 = e.second_moment_of_area(Point(6,5))
    t2 = (580*pi, 845*pi, 600*pi)
    assert t1==t2


def test_section_modulus_and_polar_second_moment_of_area():
    d = Symbol('d', positive=True)
    c = Circle((3, 7), 8)
    assert c.polar_second_moment_of_area() == 2048*pi
    assert c.section_modulus() == (128*pi, 128*pi)
    c = Circle((2, 9), d/2)
    assert c.polar_second_moment_of_area() == pi*d**3*Abs(d)/64 + pi*d*Abs(d)**3/64
    assert c.section_modulus() == (pi*d**3/S(32), pi*d**3/S(32))

    a, b = symbols('a, b', positive=True)
    e = Ellipse((4, 6), a, b)
    assert e.section_modulus() == (pi*a*b**2/S(4), pi*a**2*b/S(4))
    assert e.polar_second_moment_of_area() == pi*a**3*b/S(4) + pi*a*b**3/S(4)
    e = e.rotate(pi/2) # no change in polar and section modulus
    assert e.section_modulus() == (pi*a**2*b/S(4), pi*a*b**2/S(4))
    assert e.polar_second_moment_of_area() == pi*a**3*b/S(4) + pi*a*b**3/S(4)

    e = Ellipse((a, b), 2, 6)
    assert e.section_modulus() == (18*pi, 6*pi)
    assert e.polar_second_moment_of_area() == 120*pi

    e = Ellipse(Point(0, 0), 2, 2)
    assert e.section_modulus() == (2*pi, 2*pi)
    assert e.section_modulus(Point(2, 2)) == (2*pi, 2*pi)
    assert e.section_modulus((2, 2)) == (2*pi, 2*pi)


def test_circumference():
    M = Symbol('M')
    m = Symbol('m')
    assert Ellipse(Point(0, 0), M, m).circumference == 4 * M * elliptic_e((M ** 2 - m ** 2) / M**2)

    assert Ellipse(Point(0, 0), 5, 4).circumference == 20 * elliptic_e(S(9) / 25)

    # circle
    assert Ellipse(None, 1, None, 0).circumference == 2*pi

    # test numerically
    assert abs(Ellipse(None, hradius=5, vradius=3).circumference.evalf(16) - 25.52699886339813) < 1e-10


def test_issue_15259():
    assert Circle((1, 2), 0) == Point(1, 2)


def test_issue_15797_equals():
    Ri = 0.024127189424130748
    Ci = (0.0864931002830291, 0.0819863295239654)
    A = Point(0, 0.0578591400998346)
    c = Circle(Ci, Ri)  # evaluated
    assert c.is_tangent(c.tangent_lines(A)[0]) == True
    assert c.center.x.is_Rational
    assert c.center.y.is_Rational
    assert c.radius.is_Rational
    u = Circle(Ci, Ri, evaluate=False)  # unevaluated
    assert u.center.x.is_Float
    assert u.center.y.is_Float
    assert u.radius.is_Float


def test_auxiliary_circle():
    x, y, a, b = symbols('x y a b')
    e = Ellipse((x, y), a, b)
    # the general result
    assert e.auxiliary_circle() == Circle((x, y), Max(a, b))
    # a special case where Ellipse is a Circle
    assert Circle((3, 4), 8).auxiliary_circle() == Circle((3, 4), 8)


def test_director_circle():
    x, y, a, b = symbols('x y a b')
    e = Ellipse((x, y), a, b)
    # the general result
    assert e.director_circle() == Circle((x, y), sqrt(a**2 + b**2))
    # a special case where Ellipse is a Circle
    assert Circle((3, 4), 8).director_circle() == Circle((3, 4), 8*sqrt(2))


def test_evolute():
    #ellipse centered at h,k
    x, y, h, k = symbols('x y h k',real = True)
    a, b = symbols('a b')
    e = Ellipse(Point(h, k), a, b)
    t1 = (e.hradius*(x - e.center.x))**Rational(2, 3)
    t2 = (e.vradius*(y - e.center.y))**Rational(2, 3)
    E = t1 + t2 - (e.hradius**2 - e.vradius**2)**Rational(2, 3)
    assert e.evolute() == E
    #Numerical Example
    e = Ellipse(Point(1, 1), 6, 3)
    t1 = (6*(x - 1))**Rational(2, 3)
    t2 = (3*(y - 1))**Rational(2, 3)
    E = t1 + t2 - (27)**Rational(2, 3)
    assert e.evolute() == E


def test_svg():
    e1 = Ellipse(Point(1, 0), 3, 2)
    assert e1._svg(2, "#FFAAFF") == '<ellipse fill="#FFAAFF" stroke="#555555" stroke-width="4.0" opacity="0.6" cx="1.00000000000000" cy="0" rx="3.00000000000000" ry="2.00000000000000"/>'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\arithmetic.py ===
from __future__ import annotations

from typing import Any, cast
import numpy as np
import numpy.typing as npt
import pytest

c16 = np.complex128(1)
f8 = np.float64(1)
i8 = np.int64(1)
u8 = np.uint64(1)

c8 = np.complex64(1)
f4 = np.float32(1)
i4 = np.int32(1)
u4 = np.uint32(1)

dt = np.datetime64(1, "D")
td = np.timedelta64(1, "D")

b_ = np.bool(1)

b = bool(1)
c = complex(1)
f = float(1)
i = int(1)


class Object:
    def __array__(self, dtype: np.typing.DTypeLike = None,
                  copy: bool | None = None) -> np.ndarray[Any, np.dtype[np.object_]]:
        ret = np.empty((), dtype=object)
        ret[()] = self
        return ret

    def __sub__(self, value: Any) -> Object:
        return self

    def __rsub__(self, value: Any) -> Object:
        return self

    def __floordiv__(self, value: Any) -> Object:
        return self

    def __rfloordiv__(self, value: Any) -> Object:
        return self

    def __mul__(self, value: Any) -> Object:
        return self

    def __rmul__(self, value: Any) -> Object:
        return self

    def __pow__(self, value: Any) -> Object:
        return self

    def __rpow__(self, value: Any) -> Object:
        return self


AR_b: npt.NDArray[np.bool] = np.array([True])
AR_u: npt.NDArray[np.uint32] = np.array([1], dtype=np.uint32)
AR_i: npt.NDArray[np.int64] = np.array([1])
AR_integer: npt.NDArray[np.integer] = cast(npt.NDArray[np.integer], AR_i)
AR_f: npt.NDArray[np.float64] = np.array([1.0])
AR_c: npt.NDArray[np.complex128] = np.array([1j])
AR_m: npt.NDArray[np.timedelta64] = np.array([np.timedelta64(1, "D")])
AR_M: npt.NDArray[np.datetime64] = np.array([np.datetime64(1, "D")])
AR_O: npt.NDArray[np.object_] = np.array([Object()])

AR_LIKE_b = [True]
AR_LIKE_u = [np.uint32(1)]
AR_LIKE_i = [1]
AR_LIKE_f = [1.0]
AR_LIKE_c = [1j]
AR_LIKE_m = [np.timedelta64(1, "D")]
AR_LIKE_M = [np.datetime64(1, "D")]
AR_LIKE_O = [Object()]

# Array subtractions

AR_b - AR_LIKE_u
AR_b - AR_LIKE_i
AR_b - AR_LIKE_f
AR_b - AR_LIKE_c
AR_b - AR_LIKE_m
AR_b - AR_LIKE_O

AR_LIKE_u - AR_b
AR_LIKE_i - AR_b
AR_LIKE_f - AR_b
AR_LIKE_c - AR_b
AR_LIKE_m - AR_b
AR_LIKE_M - AR_b
AR_LIKE_O - AR_b

AR_u - AR_LIKE_b
AR_u - AR_LIKE_u
AR_u - AR_LIKE_i
AR_u - AR_LIKE_f
AR_u - AR_LIKE_c
AR_u - AR_LIKE_m
AR_u - AR_LIKE_O

AR_LIKE_b - AR_u
AR_LIKE_u - AR_u
AR_LIKE_i - AR_u
AR_LIKE_f - AR_u
AR_LIKE_c - AR_u
AR_LIKE_m - AR_u
AR_LIKE_M - AR_u
AR_LIKE_O - AR_u

AR_i - AR_LIKE_b
AR_i - AR_LIKE_u
AR_i - AR_LIKE_i
AR_i - AR_LIKE_f
AR_i - AR_LIKE_c
AR_i - AR_LIKE_m
AR_i - AR_LIKE_O

AR_LIKE_b - AR_i
AR_LIKE_u - AR_i
AR_LIKE_i - AR_i
AR_LIKE_f - AR_i
AR_LIKE_c - AR_i
AR_LIKE_m - AR_i
AR_LIKE_M - AR_i
AR_LIKE_O - AR_i

AR_f - AR_LIKE_b
AR_f - AR_LIKE_u
AR_f - AR_LIKE_i
AR_f - AR_LIKE_f
AR_f - AR_LIKE_c
AR_f - AR_LIKE_O

AR_LIKE_b - AR_f
AR_LIKE_u - AR_f
AR_LIKE_i - AR_f
AR_LIKE_f - AR_f
AR_LIKE_c - AR_f
AR_LIKE_O - AR_f

AR_c - AR_LIKE_b
AR_c - AR_LIKE_u
AR_c - AR_LIKE_i
AR_c - AR_LIKE_f
AR_c - AR_LIKE_c
AR_c - AR_LIKE_O

AR_LIKE_b - AR_c
AR_LIKE_u - AR_c
AR_LIKE_i - AR_c
AR_LIKE_f - AR_c
AR_LIKE_c - AR_c
AR_LIKE_O - AR_c

AR_m - AR_LIKE_b
AR_m - AR_LIKE_u
AR_m - AR_LIKE_i
AR_m - AR_LIKE_m

AR_LIKE_b - AR_m
AR_LIKE_u - AR_m
AR_LIKE_i - AR_m
AR_LIKE_m - AR_m
AR_LIKE_M - AR_m

AR_M - AR_LIKE_b
AR_M - AR_LIKE_u
AR_M - AR_LIKE_i
AR_M - AR_LIKE_m
AR_M - AR_LIKE_M

AR_LIKE_M - AR_M

AR_O - AR_LIKE_b
AR_O - AR_LIKE_u
AR_O - AR_LIKE_i
AR_O - AR_LIKE_f
AR_O - AR_LIKE_c
AR_O - AR_LIKE_O

AR_LIKE_b - AR_O
AR_LIKE_u - AR_O
AR_LIKE_i - AR_O
AR_LIKE_f - AR_O
AR_LIKE_c - AR_O
AR_LIKE_O - AR_O

AR_u += AR_b
AR_u += AR_u
AR_u += 1  # Allowed during runtime as long as the object is 0D and >=0

# Array floor division

AR_b // AR_LIKE_b
AR_b // AR_LIKE_u
AR_b // AR_LIKE_i
AR_b // AR_LIKE_f
AR_b // AR_LIKE_O

AR_LIKE_b // AR_b
AR_LIKE_u // AR_b
AR_LIKE_i // AR_b
AR_LIKE_f // AR_b
AR_LIKE_O // AR_b

AR_u // AR_LIKE_b
AR_u // AR_LIKE_u
AR_u // AR_LIKE_i
AR_u // AR_LIKE_f
AR_u // AR_LIKE_O

AR_LIKE_b // AR_u
AR_LIKE_u // AR_u
AR_LIKE_i // AR_u
AR_LIKE_f // AR_u
AR_LIKE_m // AR_u
AR_LIKE_O // AR_u

AR_i // AR_LIKE_b
AR_i // AR_LIKE_u
AR_i // AR_LIKE_i
AR_i // AR_LIKE_f
AR_i // AR_LIKE_O

AR_LIKE_b // AR_i
AR_LIKE_u // AR_i
AR_LIKE_i // AR_i
AR_LIKE_f // AR_i
AR_LIKE_m // AR_i
AR_LIKE_O // AR_i

AR_f // AR_LIKE_b
AR_f // AR_LIKE_u
AR_f // AR_LIKE_i
AR_f // AR_LIKE_f
AR_f // AR_LIKE_O

AR_LIKE_b // AR_f
AR_LIKE_u // AR_f
AR_LIKE_i // AR_f
AR_LIKE_f // AR_f
AR_LIKE_m // AR_f
AR_LIKE_O // AR_f

AR_m // AR_LIKE_u
AR_m // AR_LIKE_i
AR_m // AR_LIKE_f
AR_m // AR_LIKE_m

AR_LIKE_m // AR_m

AR_m /= f
AR_m //= f
AR_m /= AR_f
AR_m /= AR_LIKE_f
AR_m //= AR_f
AR_m //= AR_LIKE_f

AR_O // AR_LIKE_b
AR_O // AR_LIKE_u
AR_O // AR_LIKE_i
AR_O // AR_LIKE_f
AR_O // AR_LIKE_O

AR_LIKE_b // AR_O
AR_LIKE_u // AR_O
AR_LIKE_i // AR_O
AR_LIKE_f // AR_O
AR_LIKE_O // AR_O

# Inplace multiplication

AR_b *= AR_LIKE_b

AR_u *= AR_LIKE_b
AR_u *= AR_LIKE_u

AR_i *= AR_LIKE_b
AR_i *= AR_LIKE_u
AR_i *= AR_LIKE_i

AR_integer *= AR_LIKE_b
AR_integer *= AR_LIKE_u
AR_integer *= AR_LIKE_i

AR_f *= AR_LIKE_b
AR_f *= AR_LIKE_u
AR_f *= AR_LIKE_i
AR_f *= AR_LIKE_f

AR_c *= AR_LIKE_b
AR_c *= AR_LIKE_u
AR_c *= AR_LIKE_i
AR_c *= AR_LIKE_f
AR_c *= AR_LIKE_c

AR_m *= AR_LIKE_b
AR_m *= AR_LIKE_u
AR_m *= AR_LIKE_i
AR_m *= AR_LIKE_f

AR_O *= AR_LIKE_b
AR_O *= AR_LIKE_u
AR_O *= AR_LIKE_i
AR_O *= AR_LIKE_f
AR_O *= AR_LIKE_c
AR_O *= AR_LIKE_O

# Inplace power

AR_u **= AR_LIKE_b
AR_u **= AR_LIKE_u

AR_i **= AR_LIKE_b
AR_i **= AR_LIKE_u
AR_i **= AR_LIKE_i

AR_integer **= AR_LIKE_b
AR_integer **= AR_LIKE_u
AR_integer **= AR_LIKE_i

AR_f **= AR_LIKE_b
AR_f **= AR_LIKE_u
AR_f **= AR_LIKE_i
AR_f **= AR_LIKE_f

AR_c **= AR_LIKE_b
AR_c **= AR_LIKE_u
AR_c **= AR_LIKE_i
AR_c **= AR_LIKE_f
AR_c **= AR_LIKE_c

AR_O **= AR_LIKE_b
AR_O **= AR_LIKE_u
AR_O **= AR_LIKE_i
AR_O **= AR_LIKE_f
AR_O **= AR_LIKE_c
AR_O **= AR_LIKE_O

# unary ops

-c16
-c8
-f8
-f4
-i8
-i4
with pytest.warns(RuntimeWarning):
    -u8
    -u4
-td
-AR_f

+c16
+c8
+f8
+f4
+i8
+i4
+u8
+u4
+td
+AR_f

abs(c16)
abs(c8)
abs(f8)
abs(f4)
abs(i8)
abs(i4)
abs(u8)
abs(u4)
abs(td)
abs(b_)
abs(AR_f)

# Time structures

dt + td
dt + i
dt + i4
dt + i8
dt - dt
dt - i
dt - i4
dt - i8

td + td
td + i
td + i4
td + i8
td - td
td - i
td - i4
td - i8
td / f
td / f4
td / f8
td / td
td // td
td % td


# boolean

b_ / b
b_ / b_
b_ / i
b_ / i8
b_ / i4
b_ / u8
b_ / u4
b_ / f
b_ / f8
b_ / f4
b_ / c
b_ / c16
b_ / c8

b / b_
b_ / b_
i / b_
i8 / b_
i4 / b_
u8 / b_
u4 / b_
f / b_
f8 / b_
f4 / b_
c / b_
c16 / b_
c8 / b_

# Complex

c16 + c16
c16 + f8
c16 + i8
c16 + c8
c16 + f4
c16 + i4
c16 + b_
c16 + b
c16 + c
c16 + f
c16 + i
c16 + AR_f

c16 + c16
f8 + c16
i8 + c16
c8 + c16
f4 + c16
i4 + c16
b_ + c16
b + c16
c + c16
f + c16
i + c16
AR_f + c16

c8 + c16
c8 + f8
c8 + i8
c8 + c8
c8 + f4
c8 + i4
c8 + b_
c8 + b
c8 + c
c8 + f
c8 + i
c8 + AR_f

c16 + c8
f8 + c8
i8 + c8
c8 + c8
f4 + c8
i4 + c8
b_ + c8
b + c8
c + c8
f + c8
i + c8
AR_f + c8

# Float

f8 + f8
f8 + i8
f8 + f4
f8 + i4
f8 + b_
f8 + b
f8 + c
f8 + f
f8 + i
f8 + AR_f

f8 + f8
i8 + f8
f4 + f8
i4 + f8
b_ + f8
b + f8
c + f8
f + f8
i + f8
AR_f + f8

f4 + f8
f4 + i8
f4 + f4
f4 + i4
f4 + b_
f4 + b
f4 + c
f4 + f
f4 + i
f4 + AR_f

f8 + f4
i8 + f4
f4 + f4
i4 + f4
b_ + f4
b + f4
c + f4
f + f4
i + f4
AR_f + f4

# Int

i8 + i8
i8 + u8
i8 + i4
i8 + u4
i8 + b_
i8 + b
i8 + c
i8 + f
i8 + i
i8 + AR_f

u8 + u8
u8 + i4
u8 + u4
u8 + b_
u8 + b
u8 + c
u8 + f
u8 + i
u8 + AR_f

i8 + i8
u8 + i8
i4 + i8
u4 + i8
b_ + i8
b + i8
c + i8
f + i8
i + i8
AR_f + i8

u8 + u8
i4 + u8
u4 + u8
b_ + u8
b + u8
c + u8
f + u8
i + u8
AR_f + u8

i4 + i8
i4 + i4
i4 + i
i4 + b_
i4 + b
i4 + AR_f

u4 + i8
u4 + i4
u4 + u8
u4 + u4
u4 + i
u4 + b_
u4 + b
u4 + AR_f

i8 + i4
i4 + i4
i + i4
b_ + i4
b + i4
AR_f + i4

i8 + u4
i4 + u4
u8 + u4
u4 + u4
b_ + u4
b + u4
i + u4
AR_f + u4

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\numeric\test_indexing.py ===
import numpy as np
import pytest

from pandas.errors import InvalidIndexError

from pandas import (
    NA,
    Index,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowExtensionArray,
    FloatingArray,
)


@pytest.fixture
def index_large():
    # large values used in Index[uint64] tests where no compat needed with Int64/Float64
    large = [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25]
    return Index(large, dtype=np.uint64)


class TestGetLoc:
    def test_get_loc(self):
        index = Index([0, 1, 2])
        assert index.get_loc(1) == 1

    def test_get_loc_raises_bad_label(self):
        index = Index([0, 1, 2])
        with pytest.raises(InvalidIndexError, match=r"\[1, 2\]"):
            index.get_loc([1, 2])

    def test_get_loc_float64(self):
        idx = Index([0.0, 1.0, 2.0], dtype=np.float64)

        with pytest.raises(KeyError, match="^'foo'$"):
            idx.get_loc("foo")
        with pytest.raises(KeyError, match=r"^1\.5$"):
            idx.get_loc(1.5)
        with pytest.raises(KeyError, match="^True$"):
            idx.get_loc(True)
        with pytest.raises(KeyError, match="^False$"):
            idx.get_loc(False)

    def test_get_loc_na(self):
        idx = Index([np.nan, 1, 2], dtype=np.float64)
        assert idx.get_loc(1) == 1
        assert idx.get_loc(np.nan) == 0

        idx = Index([np.nan, 1, np.nan], dtype=np.float64)
        assert idx.get_loc(1) == 1

        # representable by slice [0:2:2]
        msg = "'Cannot get left slice bound for non-unique label: nan'"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)
        # not representable by slice
        idx = Index([np.nan, 1, np.nan, np.nan], dtype=np.float64)
        assert idx.get_loc(1) == 1
        msg = "'Cannot get left slice bound for non-unique label: nan"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)

    def test_get_loc_missing_nan(self):
        # GH#8569
        idx = Index([1, 2], dtype=np.float64)
        assert idx.get_loc(1) == 0
        with pytest.raises(KeyError, match=r"^3$"):
            idx.get_loc(3)
        with pytest.raises(KeyError, match="^nan$"):
            idx.get_loc(np.nan)
        with pytest.raises(InvalidIndexError, match=r"\[nan\]"):
            # listlike/non-hashable raises TypeError
            idx.get_loc([np.nan])

    @pytest.mark.parametrize("vals", [[1], [1.0], [Timestamp("2019-12-31")], ["test"]])
    def test_get_loc_float_index_nan_with_method(self, vals):
        # GH#39382
        idx = Index(vals)
        with pytest.raises(KeyError, match="nan"):
            idx.get_loc(np.nan)

    @pytest.mark.parametrize("dtype", ["f8", "i8", "u8"])
    def test_get_loc_numericindex_none_raises(self, dtype):
        # case that goes through searchsorted and key is non-comparable to values
        arr = np.arange(10**7, dtype=dtype)
        idx = Index(arr)
        with pytest.raises(KeyError, match="None"):
            idx.get_loc(None)

    def test_get_loc_overflows(self):
        # unique but non-monotonic goes through IndexEngine.mapping.get_item
        idx = Index([0, 2, 1])

        val = np.iinfo(np.int64).max + 1

        with pytest.raises(KeyError, match=str(val)):
            idx.get_loc(val)
        with pytest.raises(KeyError, match=str(val)):
            idx._engine.get_loc(val)


class TestGetIndexer:
    def test_get_indexer(self):
        index1 = Index([1, 2, 3, 4, 5])
        index2 = Index([2, 4, 6])

        r1 = index1.get_indexer(index2)
        e1 = np.array([1, 3, -1], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize(
        "expected,method",
        [
            (np.array([-1, 0, 0, 1, 1], dtype=np.intp), "pad"),
            (np.array([-1, 0, 0, 1, 1], dtype=np.intp), "ffill"),
            (np.array([0, 0, 1, 1, 2], dtype=np.intp), "backfill"),
            (np.array([0, 0, 1, 1, 2], dtype=np.intp), "bfill"),
        ],
    )
    def test_get_indexer_methods(self, reverse, expected, method):
        index1 = Index([1, 2, 3, 4, 5])
        index2 = Index([2, 4, 6])

        if reverse:
            index1 = index1[::-1]
            expected = expected[::-1]

        result = index2.get_indexer(index1, method=method)
        tm.assert_almost_equal(result, expected)

    def test_get_indexer_invalid(self):
        # GH10411
        index = Index(np.arange(10))

        with pytest.raises(ValueError, match="tolerance argument"):
            index.get_indexer([1, 0], tolerance=1)

        with pytest.raises(ValueError, match="limit argument"):
            index.get_indexer([1, 0], limit=1)

    @pytest.mark.parametrize(
        "method, tolerance, indexer, expected",
        [
            ("pad", None, [0, 5, 9], [0, 5, 9]),
            ("backfill", None, [0, 5, 9], [0, 5, 9]),
            ("nearest", None, [0, 5, 9], [0, 5, 9]),
            ("pad", 0, [0, 5, 9], [0, 5, 9]),
            ("backfill", 0, [0, 5, 9], [0, 5, 9]),
            ("nearest", 0, [0, 5, 9], [0, 5, 9]),
            ("pad", None, [0.2, 1.8, 8.5], [0, 1, 8]),
            ("backfill", None, [0.2, 1.8, 8.5], [1, 2, 9]),
            ("nearest", None, [0.2, 1.8, 8.5], [0, 2, 9]),
            ("pad", 1, [0.2, 1.8, 8.5], [0, 1, 8]),
            ("backfill", 1, [0.2, 1.8, 8.5], [1, 2, 9]),
            ("nearest", 1, [0.2, 1.8, 8.5], [0, 2, 9]),
            ("pad", 0.2, [0.2, 1.8, 8.5], [0, -1, -1]),
            ("backfill", 0.2, [0.2, 1.8, 8.5], [-1, 2, -1]),
            ("nearest", 0.2, [0.2, 1.8, 8.5], [0, 2, -1]),
        ],
    )
    def test_get_indexer_nearest(self, method, tolerance, indexer, expected):
        index = Index(np.arange(10))

        actual = index.get_indexer(indexer, method=method, tolerance=tolerance)
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize("listtype", [list, tuple, Series, np.array])
    @pytest.mark.parametrize(
        "tolerance, expected",
        list(
            zip(
                [[0.3, 0.3, 0.1], [0.2, 0.1, 0.1], [0.1, 0.5, 0.5]],
                [[0, 2, -1], [0, -1, -1], [-1, 2, 9]],
            )
        ),
    )
    def test_get_indexer_nearest_listlike_tolerance(
        self, tolerance, expected, listtype
    ):
        index = Index(np.arange(10))

        actual = index.get_indexer(
            [0.2, 1.8, 8.5], method="nearest", tolerance=listtype(tolerance)
        )
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    def test_get_indexer_nearest_error(self):
        index = Index(np.arange(10))
        with pytest.raises(ValueError, match="limit argument"):
            index.get_indexer([1, 0], method="nearest", limit=1)

        with pytest.raises(ValueError, match="tolerance size must match"):
            index.get_indexer([1, 0], method="nearest", tolerance=[1, 2, 3])

    @pytest.mark.parametrize(
        "method,expected",
        [("pad", [8, 7, 0]), ("backfill", [9, 8, 1]), ("nearest", [9, 7, 0])],
    )
    def test_get_indexer_nearest_decreasing(self, method, expected):
        index = Index(np.arange(10))[::-1]

        actual = index.get_indexer([0, 5, 9], method=method)
        tm.assert_numpy_array_equal(actual, np.array([9, 4, 0], dtype=np.intp))

        actual = index.get_indexer([0.2, 1.8, 8.5], method=method)
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize("idx_dtype", ["int64", "float64", "uint64", "range"])
    @pytest.mark.parametrize("method", ["get_indexer", "get_indexer_non_unique"])
    def test_get_indexer_numeric_index_boolean_target(self, method, idx_dtype):
        # GH 16877

        if idx_dtype == "range":
            numeric_index = RangeIndex(4)
        else:
            numeric_index = Index(np.arange(4, dtype=idx_dtype))

        other = Index([True, False, True])

        result = getattr(numeric_index, method)(other)
        expected = np.array([-1, -1, -1], dtype=np.intp)
        if method == "get_indexer":
            tm.assert_numpy_array_equal(result, expected)
        else:
            missing = np.arange(3, dtype=np.intp)
            tm.assert_numpy_array_equal(result[0], expected)
            tm.assert_numpy_array_equal(result[1], missing)

    @pytest.mark.parametrize("method", ["pad", "backfill", "nearest"])
    def test_get_indexer_with_method_numeric_vs_bool(self, method):
        left = Index([1, 2, 3])
        right = Index([True, False])

        with pytest.raises(TypeError, match="Cannot compare"):
            left.get_indexer(right, method=method)

        with pytest.raises(TypeError, match="Cannot compare"):
            right.get_indexer(left, method=method)

    def test_get_indexer_numeric_vs_bool(self):
        left = Index([1, 2, 3])
        right = Index([True, False])

        res = left.get_indexer(right)
        expected = -1 * np.ones(len(right), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

        res = right.get_indexer(left)
        expected = -1 * np.ones(len(left), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

        res = left.get_indexer_non_unique(right)[0]
        expected = -1 * np.ones(len(right), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

        res = right.get_indexer_non_unique(left)[0]
        expected = -1 * np.ones(len(left), dtype=np.intp)
        tm.assert_numpy_array_equal(res, expected)

    def test_get_indexer_float64(self):
        idx = Index([0.0, 1.0, 2.0], dtype=np.float64)
        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        target = [-0.1, 0.5, 1.1]
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )

    def test_get_indexer_nan(self):
        # GH#7820
        result = Index([1, 2, np.nan], dtype=np.float64).get_indexer([np.nan])
        expected = np.array([2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_int64(self):
        index = Index(range(0, 20, 2), dtype=np.int64)
        target = Index(np.arange(10), dtype=np.int64)
        indexer = index.get_indexer(target)
        expected = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Index(np.arange(10), dtype=np.int64)
        indexer = index.get_indexer(target, method="pad")
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Index(np.arange(10), dtype=np.int64)
        indexer = index.get_indexer(target, method="backfill")
        expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_uint64(self, index_large):
        target = Index(np.arange(10).astype("uint64") * 5 + 2**63)
        indexer = index_large.get_indexer(target)
        expected = np.array([0, -1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Index(np.arange(10).astype("uint64") * 5 + 2**63)
        indexer = index_large.get_indexer(target, method="pad")
        expected = np.array([0, 0, 1, 2, 3, 4, 4, 4, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

        target = Index(np.arange(10).astype("uint64") * 5 + 2**63)
        indexer = index_large.get_indexer(target, method="backfill")
        expected = np.array([0, 1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    @pytest.mark.parametrize("val, val2", [(4, 5), (4, 4), (4, NA), (NA, NA)])
    def test_get_loc_masked(self, val, val2, any_numeric_ea_and_arrow_dtype):
        # GH#39133
        idx = Index([1, 2, 3, val, val2], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.get_loc(2)
        assert result == 1

        with pytest.raises(KeyError, match="9"):
            idx.get_loc(9)

    def test_get_loc_masked_na(self, any_numeric_ea_and_arrow_dtype):
        # GH#39133
        idx = Index([1, 2, NA], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.get_loc(NA)
        assert result == 2

        idx = Index([1, 2, NA, NA], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.get_loc(NA)
        tm.assert_numpy_array_equal(result, np.array([False, False, True, True]))

        idx = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
        with pytest.raises(KeyError, match="NA"):
            idx.get_loc(NA)

    def test_get_loc_masked_na_and_nan(self):
        # GH#39133
        idx = Index(
            FloatingArray(
                np.array([1, 2, 1, np.nan]), mask=np.array([False, False, True, False])
            )
        )
        result = idx.get_loc(NA)
        assert result == 2
        result = idx.get_loc(np.nan)
        assert result == 3

        idx = Index(
            FloatingArray(np.array([1, 2, 1.0]), mask=np.array([False, False, True]))
        )
        result = idx.get_loc(NA)
        assert result == 2
        with pytest.raises(KeyError, match="nan"):
            idx.get_loc(np.nan)

        idx = Index(
            FloatingArray(
                np.array([1, 2, np.nan]), mask=np.array([False, False, False])
            )
        )
        result = idx.get_loc(np.nan)
        assert result == 2
        with pytest.raises(KeyError, match="NA"):
            idx.get_loc(NA)

    @pytest.mark.parametrize("val", [4, 2])
    def test_get_indexer_masked_na(self, any_numeric_ea_and_arrow_dtype, val):
        # GH#39133
        idx = Index([1, 2, NA, 3, val], dtype=any_numeric_ea_and_arrow_dtype)
        result = idx.get_indexer_for([1, NA, 5])
        expected = np.array([0, 2, -1])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("dtype", ["boolean", "bool[pyarrow]"])
    def test_get_indexer_masked_na_boolean(self, dtype):
        # GH#39133
        if dtype == "bool[pyarrow]":
            pytest.importorskip("pyarrow")
        idx = Index([True, False, NA], dtype=dtype)
        result = idx.get_loc(False)
        assert result == 1
        result = idx.get_loc(NA)
        assert result == 2

    def test_get_indexer_arrow_dictionary_target(self):
        pa = pytest.importorskip("pyarrow")
        target = Index(
            ArrowExtensionArray(
                pa.array([1, 2], type=pa.dictionary(pa.int8(), pa.int8()))
            )
        )
        idx = Index([1])

        result = idx.get_indexer(target)
        expected = np.array([0, -1], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

        result_1, result_2 = idx.get_indexer_non_unique(target)
        expected_1, expected_2 = np.array([0, -1], dtype=np.int64), np.array(
            [1], dtype=np.int64
        )
        tm.assert_numpy_array_equal(result_1, expected_1)
        tm.assert_numpy_array_equal(result_2, expected_2)


class TestWhere:
    @pytest.mark.parametrize(
        "index",
        [
            Index(np.arange(5, dtype="float64")),
            Index(range(0, 20, 2), dtype=np.int64),
            Index(np.arange(5, dtype="uint64")),
        ],
    )
    def test_where(self, listlike_box, index):
        cond = [True] * len(index)
        expected = index
        result = index.where(listlike_box(cond))

        cond = [False] + [True] * (len(index) - 1)
        expected = Index([index._na_value] + index[1:].tolist(), dtype=np.float64)
        result = index.where(listlike_box(cond))
        tm.assert_index_equal(result, expected)

    def test_where_uint64(self):
        idx = Index([0, 6, 2], dtype=np.uint64)
        mask = np.array([False, True, False])
        other = np.array([1], dtype=np.int64)

        expected = Index([1, 6, 1], dtype=np.uint64)

        result = idx.where(mask, other)
        tm.assert_index_equal(result, expected)

        result = idx.putmask(~mask, other)
        tm.assert_index_equal(result, expected)

    def test_where_infers_type_instead_of_trying_to_convert_string_to_float(self):
        # GH 32413
        index = Index([1, np.nan])
        cond = index.notna()
        other = Index(["a", "b"], dtype="string")

        expected = Index([1.0, "b"])
        result = index.where(cond, other)

        tm.assert_index_equal(result, expected)


class TestTake:
    @pytest.mark.parametrize("idx_dtype", [np.float64, np.int64, np.uint64])
    def test_take_preserve_name(self, idx_dtype):
        index = Index([1, 2, 3, 4], dtype=idx_dtype, name="foo")
        taken = index.take([3, 0, 1])
        assert index.name == taken.name

    def test_take_fill_value_float64(self):
        # GH 12631
        idx = Index([1.0, 2.0, 3.0], name="xxx", dtype=np.float64)
        result = idx.take(np.array([1, 0, -1]))
        expected = Index([2.0, 1.0, 3.0], dtype=np.float64, name="xxx")
        tm.assert_index_equal(result, expected)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = Index([2.0, 1.0, np.nan], dtype=np.float64, name="xxx")
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index([2.0, 1.0, 3.0], dtype=np.float64, name="xxx")
        tm.assert_index_equal(result, expected)

        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_take_fill_value_ints(self, dtype):
        # see gh-12631
        idx = Index([1, 2, 3], dtype=dtype, name="xxx")
        result = idx.take(np.array([1, 0, -1]))
        expected = Index([2, 1, 3], dtype=dtype, name="xxx")
        tm.assert_index_equal(result, expected)

        name = type(idx).__name__
        msg = f"Unable to fill values because {name} cannot contain NA"

        # fill_value=True
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -1]), fill_value=True)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index([2, 1, 3], dtype=dtype, name="xxx")
        tm.assert_index_equal(result, expected)

        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))


class TestContains:
    @pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64])
    def test_contains_none(self, dtype):
        # GH#35788 should return False, not raise TypeError
        index = Index([0, 1, 2, 3, 4], dtype=dtype)
        assert None not in index

    def test_contains_float64_nans(self):
        index = Index([1.0, 2.0, np.nan], dtype=np.float64)
        assert np.nan in index

    def test_contains_float64_not_nans(self):
        index = Index([1.0, 2.0, np.nan], dtype=np.float64)
        assert 1.0 in index


class TestSliceLocs:
    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs(self, dtype):
        index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
        n = len(index)

        assert index.slice_locs(start=2) == (2, n)
        assert index.slice_locs(start=3) == (3, n)
        assert index.slice_locs(3, 8) == (3, 6)
        assert index.slice_locs(5, 10) == (3, n)
        assert index.slice_locs(end=8) == (0, 6)
        assert index.slice_locs(end=9) == (0, 7)

        # reversed
        index2 = index[::-1]
        assert index2.slice_locs(8, 2) == (2, 6)
        assert index2.slice_locs(7, 3) == (2, 5)

    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs_float_locs(self, dtype):
        index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
        n = len(index)
        assert index.slice_locs(5.0, 10.0) == (3, n)
        assert index.slice_locs(4.5, 10.5) == (3, 8)

        index2 = index[::-1]
        assert index2.slice_locs(8.5, 1.5) == (2, 6)
        assert index2.slice_locs(10.5, -1) == (0, n)

    @pytest.mark.parametrize("dtype", [int, float])
    def test_slice_locs_dup_numeric(self, dtype):
        index = Index(np.array([10, 12, 12, 14], dtype=dtype))
        assert index.slice_locs(12, 12) == (1, 3)
        assert index.slice_locs(11, 13) == (1, 3)

        index2 = index[::-1]
        assert index2.slice_locs(12, 12) == (1, 3)
        assert index2.slice_locs(13, 11) == (1, 3)

    def test_slice_locs_na(self):
        index = Index([np.nan, 1, 2])
        assert index.slice_locs(1) == (1, 3)
        assert index.slice_locs(np.nan) == (0, 3)

        index = Index([0, np.nan, np.nan, 1, 2])
        assert index.slice_locs(np.nan) == (1, 5)

    def test_slice_locs_na_raises(self):
        index = Index([np.nan, 1, 2])
        with pytest.raises(KeyError, match=""):
            index.slice_locs(start=1.5)

        with pytest.raises(KeyError, match=""):
            index.slice_locs(end=1.5)


class TestGetSliceBounds:
    @pytest.mark.parametrize("side, expected", [("left", 4), ("right", 5)])
    def test_get_slice_bounds_within(self, side, expected):
        index = Index(range(6))
        result = index.get_slice_bound(4, side=side)
        assert result == expected

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("bound, expected", [(-1, 0), (10, 6)])
    def test_get_slice_bounds_outside(self, side, expected, bound):
        index = Index(range(6))
        result = index.get_slice_bound(bound, side=side)
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\itsdangerous\_json.py ===
from __future__ import annotations

import json as _json
import typing as t


class _CompactJSON:
    """Wrapper around json module that strips whitespace."""

    @staticmethod
    def loads(payload: str | bytes) -> t.Any:
        return _json.loads(payload)

    @staticmethod
    def dumps(obj: t.Any, **kwargs: t.Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("separators", (",", ":"))
        return _json.dumps(obj, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_utils\_convertions.py ===
"""
A set of methods retained from np.compat module that
are still used across codebase.
"""

__all__ = ["asunicode", "asbytes"]


def asunicode(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\datasets\__init__.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Short examples used in the documentation.
"""

import os


def get_example(name):
    """
    Retrieves the absolute file name of an example.
    """
    this = os.path.abspath(os.path.dirname(__file__))
    full = os.path.join(this, name)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Unable to find example '{name}'")
    return full

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\argmax.py ===
from .base_operator import QuantOperatorBase


# Use the quantized tensor as input without DQ.
class QArgMax(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        quantized_input_value = self.quantizer.find_quantized_value(node.input[0])
        if quantized_input_value is None:
            self.quantizer.new_nodes += [node]
            return

        node.input[0] = quantized_input_value.q_name
        self.quantizer.new_nodes += [node]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\AttributeType.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

class AttributeType(object):
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSOR = 11
    SPARSE_TENSORS = 12

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\descriptors\slots.py ===
# Metaclass for mixing slots and descriptors
# From "Programming in Python 3" by Mark Summerfield Ch.8 p. 383

class AutoSlotProperties(type):

    def __new__(mcl, classname, bases, dictionary):
        slots = list(dictionary.get("__slots__", []))
        for getter_name in [key for key in dictionary if key.startswith("get_")]:
            name = getter_name
            slots.append("__" + name)
            getter = dictionary.pop(getter_name)
            setter = dictionary.get(setter_name, None)
            if (setter is not None
                and isinstance(setter, collections.Callable)):
                del dictionary[setter_name]
            dictionary[name] = property(getter. setter)
            dictionary["__slots__"] = tuple(slots)
            return super().__new__(mcl, classname, bases, dictionary)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\workbook\external_reference.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Sequence
)
from openpyxl.descriptors.excel import (
    Relation,
)

class ExternalReference(Serialisable):

    tagname = "externalReference"

    id = Relation()

    def __init__(self, id):
        self.id = id

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\testing.py ===
"""
Public testing utility functions.
"""


from pandas._testing import (
    assert_extension_array_equal,
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)

__all__ = [
    "assert_extension_array_equal",
    "assert_frame_equal",
    "assert_series_equal",
    "assert_index_equal",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\__init__.py ===
from typing import List, Optional

from pip._internal.utils import _log

# init_logging() must be called before any call to logging.getLogger()
# which happens at import of most modules.
_log.init_logging()


def main(args: Optional[List[str]] = None) -> int:
    """This is preserved for old console scripts that may still be referencing
    it.

    For additional details, see https://github.com/pypa/pip/issues/7498.
    """
    from pip._internal.utils.entrypoints import _wrapper

    return _wrapper(args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\logger\log.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from pyreadline3.unicode_helper import ensure_str

from .logger import LOGGER


def log(record: str) -> None:
    s = ensure_str(record)

    LOGGER.debug(s)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\connectors\__init__.py ===
# connectors/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php


from ..engine.interfaces import Dialect


class Connector(Dialect):
    """Base class for dialect mixins, for DBAPIs that work
    across entirely different database backends.

    Currently the only such mixin is pyodbc.

    """

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\__init__.py ===
"""
A module to implement logical predicates and assumption system.
"""

from .assume import (
    AppliedPredicate, Predicate, AssumptionsContext, assuming,
    global_assumptions
)
from .ask import Q, ask, register_handler, remove_handler
from .refine import refine
from .relation import BinaryRelation, AppliedBinaryRelation

__all__ = [
    'AppliedPredicate', 'Predicate', 'AssumptionsContext', 'assuming',
    'global_assumptions', 'Q', 'ask', 'register_handler', 'remove_handler',
    'refine',
    'BinaryRelation', 'AppliedBinaryRelation'
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\abstract_nodes.py ===
"""This module provides containers for python objects that are valid
printing targets but are not a subclass of SymPy's Printable.
"""


from sympy.core.containers import Tuple


class List(Tuple):
    """Represents a (frozen) (Python) list (for code printing purposes)."""
    def __eq__(self, other):
        if isinstance(other, list):
            return self == List(*other)
        else:
            return self.args == other

    def __hash__(self):
        return super().__hash__()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\holonomic\__init__.py ===
r"""
The :py:mod:`~sympy.holonomic` module is intended to deal with holonomic functions along
with various operations on them like addition, multiplication, composition,
integration and differentiation. The module also implements various kinds of
conversions such as converting holonomic functions to a different form and the
other way around.
"""

from .holonomic import (DifferentialOperator, HolonomicFunction, DifferentialOperators,
    from_hyper, from_meijerg, expr_to_holonomic)
from .recurrence import RecurrenceOperators, RecurrenceOperator, HolonomicSequence

__all__ = [
    'DifferentialOperator', 'HolonomicFunction', 'DifferentialOperators',
    'from_hyper', 'from_meijerg', 'expr_to_holonomic',

    'RecurrenceOperators', 'RecurrenceOperator', 'HolonomicSequence',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\hyperexpand_doc.py ===
""" This module cooks up a docstring when imported. Its only purpose is to
    be displayed in the sphinx documentation. """

from sympy.core.relational import Eq
from sympy.functions.special.hyper import hyper
from sympy.printing.latex import latex
from sympy.simplify.hyperexpand import FormulaCollection

c = FormulaCollection()

doc = ""

for f in c.formulae:
    obj = Eq(hyper(f.func.ap, f.func.bq, f.z),
             f.closed_form.rewrite('nonrepsmall'))
    doc += ".. math::\n  %s\n" % latex(obj)

__doc__ = doc

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_datetimes.py ===
import datetime as dt
from datetime import datetime

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    to_timedelta,
)
import pandas._testing as tm


class TestDatetimeConcat:
    def test_concat_datetime64_block(self):
        rng = date_range("1/1/2000", periods=10)

        df = DataFrame({"time": rng})

        result = concat([df, df])
        assert (result.iloc[:10]["time"] == rng).all()
        assert (result.iloc[10:]["time"] == rng).all()

    def test_concat_datetime_datetime64_frame(self):
        # GH#2624
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), "hi"])

        df2_obj = DataFrame.from_records(rows, columns=["date", "test"])

        ind = date_range(start="2000/1/1", freq="D", periods=10)
        df1 = DataFrame({"date": ind, "test": range(10)})

        # it works!
        concat([df1, df2_obj])

    def test_concat_datetime_timezone(self):
        # GH 18523
        idx1 = date_range("2011-01-01", periods=3, freq="h", tz="Europe/Paris")
        idx2 = date_range(start=idx1[0], end=idx1[-1], freq="h")
        df1 = DataFrame({"a": [1, 2, 3]}, index=idx1)
        df2 = DataFrame({"b": [1, 2, 3]}, index=idx2)
        result = concat([df1, df2], axis=1)

        exp_idx = DatetimeIndex(
            [
                "2011-01-01 00:00:00+01:00",
                "2011-01-01 01:00:00+01:00",
                "2011-01-01 02:00:00+01:00",
            ],
            dtype="M8[ns, Europe/Paris]",
            freq="h",
        )
        expected = DataFrame(
            [[1, 1], [2, 2], [3, 3]], index=exp_idx, columns=["a", "b"]
        )

        tm.assert_frame_equal(result, expected)

        idx3 = date_range("2011-01-01", periods=3, freq="h", tz="Asia/Tokyo")
        df3 = DataFrame({"b": [1, 2, 3]}, index=idx3)
        result = concat([df1, df3], axis=1)

        exp_idx = DatetimeIndex(
            [
                "2010-12-31 15:00:00+00:00",
                "2010-12-31 16:00:00+00:00",
                "2010-12-31 17:00:00+00:00",
                "2010-12-31 23:00:00+00:00",
                "2011-01-01 00:00:00+00:00",
                "2011-01-01 01:00:00+00:00",
            ]
        ).as_unit("ns")

        expected = DataFrame(
            [
                [np.nan, 1],
                [np.nan, 2],
                [np.nan, 3],
                [1, np.nan],
                [2, np.nan],
                [3, np.nan],
            ],
            index=exp_idx,
            columns=["a", "b"],
        )

        tm.assert_frame_equal(result, expected)

        # GH 13783: Concat after resample
        result = concat([df1.resample("h").mean(), df2.resample("h").mean()], sort=True)
        expected = DataFrame(
            {"a": [1, 2, 3] + [np.nan] * 3, "b": [np.nan] * 3 + [1, 2, 3]},
            index=idx1.append(idx1),
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_datetimeindex_freq(self):
        # GH 3232
        # Monotonic index result
        dr = date_range("01-Jan-2013", periods=100, freq="50ms", tz="UTC")
        data = list(range(100))
        expected = DataFrame(data, index=dr)
        result = concat([expected[:50], expected[50:]])
        tm.assert_frame_equal(result, expected)

        # Non-monotonic index result
        result = concat([expected[50:], expected[:50]])
        expected = DataFrame(data[50:] + data[:50], index=dr[50:].append(dr[:50]))
        expected.index._data.freq = None
        tm.assert_frame_equal(result, expected)

    def test_concat_multiindex_datetime_object_index(self):
        # https://github.com/pandas-dev/pandas/issues/11058
        idx = Index(
            [dt.date(2013, 1, 1), dt.date(2014, 1, 1), dt.date(2015, 1, 1)],
            dtype="object",
        )

        s = Series(
            ["a", "b"],
            index=MultiIndex.from_arrays(
                [
                    [1, 2],
                    idx[:-1],
                ],
                names=["first", "second"],
            ),
        )
        s2 = Series(
            ["a", "b"],
            index=MultiIndex.from_arrays(
                [[1, 2], idx[::2]],
                names=["first", "second"],
            ),
        )
        mi = MultiIndex.from_arrays(
            [[1, 2, 2], idx],
            names=["first", "second"],
        )
        assert mi.levels[1].dtype == object

        expected = DataFrame(
            [["a", "a"], ["b", np.nan], [np.nan, "b"]],
            index=mi,
        )
        result = concat([s, s2], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_NaT_series(self):
        # GH 11693
        # test for merging NaT series with datetime series.
        x = Series(
            date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="US/Eastern")
        )
        y = Series(pd.NaT, index=[0, 1], dtype="datetime64[ns, US/Eastern]")
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])

        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # all NaT with tz
        expected = Series(pd.NaT, index=range(4), dtype="datetime64[ns, US/Eastern]")
        result = concat([y, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_NaT_series2(self):
        # without tz
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h"))
        y = Series(date_range("20151124 10:00", "20151124 11:00", freq="1h"))
        y[:] = pd.NaT
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # all NaT without tz
        x[:] = pd.NaT
        expected = Series(pd.NaT, index=range(4), dtype="datetime64[ns]")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_concat_NaT_dataframes(self, tz):
        # GH 12396

        dti = DatetimeIndex([pd.NaT, pd.NaT], tz=tz)
        first = DataFrame({0: dti})
        second = DataFrame(
            [[Timestamp("2015/01/01", tz=tz)], [Timestamp("2016/01/01", tz=tz)]],
            index=[2, 3],
        )
        expected = DataFrame(
            [
                pd.NaT,
                pd.NaT,
                Timestamp("2015/01/01", tz=tz),
                Timestamp("2016/01/01", tz=tz),
            ]
        )

        result = concat([first, second], axis=0)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    @pytest.mark.parametrize("item", [pd.NaT, Timestamp("20150101")])
    def test_concat_NaT_dataframes_all_NaT_axis_0(
        self, tz1, tz2, item, using_array_manager
    ):
        # GH 12396

        # tz-naive
        first = DataFrame([[pd.NaT], [pd.NaT]]).apply(lambda x: x.dt.tz_localize(tz1))
        second = DataFrame([item]).apply(lambda x: x.dt.tz_localize(tz2))

        result = concat([first, second], axis=0)
        expected = DataFrame(Series([pd.NaT, pd.NaT, item], index=[0, 1, 0]))
        expected = expected.apply(lambda x: x.dt.tz_localize(tz2))
        if tz1 != tz2:
            expected = expected.astype(object)
            if item is pd.NaT and not using_array_manager:
                # GH#18463
                # TODO: setting nan here is to keep the test passing as we
                #  make assert_frame_equal stricter, but is nan really the
                #  ideal behavior here?
                if tz1 is not None:
                    expected.iloc[-1, 0] = np.nan
                else:
                    expected.iloc[:-1, 0] = np.nan

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    def test_concat_NaT_dataframes_all_NaT_axis_1(self, tz1, tz2):
        # GH 12396

        first = DataFrame(Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1))
        second = DataFrame(Series([pd.NaT]).dt.tz_localize(tz2), columns=[1])
        expected = DataFrame(
            {
                0: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1),
                1: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz2),
            }
        )
        result = concat([first, second], axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    def test_concat_NaT_series_dataframe_all_NaT(self, tz1, tz2):
        # GH 12396

        # tz-naive
        first = Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1)
        second = DataFrame(
            [
                [Timestamp("2015/01/01", tz=tz2)],
                [Timestamp("2016/01/01", tz=tz2)],
            ],
            index=[2, 3],
        )

        expected = DataFrame(
            [
                pd.NaT,
                pd.NaT,
                Timestamp("2015/01/01", tz=tz2),
                Timestamp("2016/01/01", tz=tz2),
            ]
        )
        if tz1 != tz2:
            expected = expected.astype(object)

        result = concat([first, second])
        tm.assert_frame_equal(result, expected)


class TestTimezoneConcat:
    def test_concat_tz_series(self):
        # gh-11755: tz and no tz
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="UTC"))
        y = Series(date_range("2012-01-01", "2012-01-02"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_tz_series2(self):
        # gh-11887: concat tz and object
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="UTC"))
        y = Series(["a", "b"])
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_tz_series3(self, unit, unit2):
        # see gh-12217 and gh-12306
        # Concatenating two UTC times
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f"M8[{unit}]")
        first[0] = first[0].dt.tz_localize("UTC")

        second = DataFrame([[datetime(2016, 1, 2)]], dtype=f"M8[{unit2}]")
        second[0] = second[0].dt.tz_localize("UTC")

        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f"datetime64[{exp_unit}, UTC]"

    def test_concat_tz_series4(self, unit, unit2):
        # Concatenating two London times
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f"M8[{unit}]")
        first[0] = first[0].dt.tz_localize("Europe/London")

        second = DataFrame([[datetime(2016, 1, 2)]], dtype=f"M8[{unit2}]")
        second[0] = second[0].dt.tz_localize("Europe/London")

        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f"datetime64[{exp_unit}, Europe/London]"

    def test_concat_tz_series5(self, unit, unit2):
        # Concatenating 2+1 London times
        first = DataFrame(
            [[datetime(2016, 1, 1)], [datetime(2016, 1, 2)]], dtype=f"M8[{unit}]"
        )
        first[0] = first[0].dt.tz_localize("Europe/London")

        second = DataFrame([[datetime(2016, 1, 3)]], dtype=f"M8[{unit2}]")
        second[0] = second[0].dt.tz_localize("Europe/London")

        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f"datetime64[{exp_unit}, Europe/London]"

    def test_concat_tz_series6(self, unit, unit2):
        # Concatenating 1+2 London times
        first = DataFrame([[datetime(2016, 1, 1)]], dtype=f"M8[{unit}]")
        first[0] = first[0].dt.tz_localize("Europe/London")

        second = DataFrame(
            [[datetime(2016, 1, 2)], [datetime(2016, 1, 3)]], dtype=f"M8[{unit2}]"
        )
        second[0] = second[0].dt.tz_localize("Europe/London")

        result = concat([first, second])
        exp_unit = tm.get_finest_unit(unit, unit2)
        assert result[0].dtype == f"datetime64[{exp_unit}, Europe/London]"

    def test_concat_tz_series_tzlocal(self):
        # see gh-13583
        x = [
            Timestamp("2011-01-01", tz=dateutil.tz.tzlocal()),
            Timestamp("2011-02-01", tz=dateutil.tz.tzlocal()),
        ]
        y = [
            Timestamp("2012-01-01", tz=dateutil.tz.tzlocal()),
            Timestamp("2012-02-01", tz=dateutil.tz.tzlocal()),
        ]

        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y))
        assert result.dtype == "datetime64[ns, tzlocal()]"

    def test_concat_tz_series_with_datetimelike(self):
        # see gh-12620: tz and timedelta
        x = [
            Timestamp("2011-01-01", tz="US/Eastern"),
            Timestamp("2011-02-01", tz="US/Eastern"),
        ]
        y = [pd.Timedelta("1 day"), pd.Timedelta("2 day")]
        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y, dtype="object"))

        # tz and period
        y = [pd.Period("2011-03", freq="M"), pd.Period("2011-04", freq="M")]
        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y, dtype="object"))

    def test_concat_tz_frame(self):
        df2 = DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
            },
            index=range(5),
        )

        # concat
        df3 = concat([df2.A.to_frame(), df2.B.to_frame()], axis=1)
        tm.assert_frame_equal(df2, df3)

    def test_concat_multiple_tzs(self):
        # GH#12467
        # combining datetime tz-aware and naive DataFrames
        ts1 = Timestamp("2015-01-01", tz=None)
        ts2 = Timestamp("2015-01-01", tz="UTC")
        ts3 = Timestamp("2015-01-01", tz="EST")

        df1 = DataFrame({"time": [ts1]})
        df2 = DataFrame({"time": [ts2]})
        df3 = DataFrame({"time": [ts3]})

        results = concat([df1, df2]).reset_index(drop=True)
        expected = DataFrame({"time": [ts1, ts2]}, dtype=object)
        tm.assert_frame_equal(results, expected)

        results = concat([df1, df3]).reset_index(drop=True)
        expected = DataFrame({"time": [ts1, ts3]}, dtype=object)
        tm.assert_frame_equal(results, expected)

        results = concat([df2, df3]).reset_index(drop=True)
        expected = DataFrame({"time": [ts2, ts3]})
        tm.assert_frame_equal(results, expected)

    def test_concat_multiindex_with_tz(self):
        # GH 6606
        df = DataFrame(
            {
                "dt": DatetimeIndex(
                    [
                        datetime(2014, 1, 1),
                        datetime(2014, 1, 2),
                        datetime(2014, 1, 3),
                    ],
                    dtype="M8[ns, US/Pacific]",
                ),
                "b": ["A", "B", "C"],
                "c": [1, 2, 3],
                "d": [4, 5, 6],
            }
        )
        df = df.set_index(["dt", "b"])

        exp_idx1 = DatetimeIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03"] * 2,
            dtype="M8[ns, US/Pacific]",
            name="dt",
        )
        exp_idx2 = Index(["A", "B", "C"] * 2, name="b")
        exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
        expected = DataFrame(
            {"c": [1, 2, 3] * 2, "d": [4, 5, 6] * 2}, index=exp_idx, columns=["c", "d"]
        )

        result = concat([df, df])
        tm.assert_frame_equal(result, expected)

    def test_concat_tz_not_aligned(self):
        # GH#22796
        ts = pd.to_datetime([1, 2]).tz_localize("UTC")
        a = DataFrame({"A": ts})
        b = DataFrame({"A": ts, "B": ts})
        result = concat([a, b], sort=True, ignore_index=True)
        expected = DataFrame(
            {"A": list(ts) + list(ts), "B": [pd.NaT, pd.NaT] + list(ts)}
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "t1",
        [
            "2015-01-01",
            pytest.param(
                pd.NaT,
                marks=pytest.mark.xfail(
                    reason="GH23037 incorrect dtype when concatenating"
                ),
            ),
        ],
    )
    def test_concat_tz_NaT(self, t1):
        # GH#22796
        # Concatenating tz-aware multicolumn DataFrames
        ts1 = Timestamp(t1, tz="UTC")
        ts2 = Timestamp("2015-01-01", tz="UTC")
        ts3 = Timestamp("2015-01-01", tz="UTC")

        df1 = DataFrame([[ts1, ts2]])
        df2 = DataFrame([[ts3]])

        result = concat([df1, df2])
        expected = DataFrame([[ts1, ts2], [ts3, pd.NaT]], index=[0, 0])

        tm.assert_frame_equal(result, expected)

    def test_concat_tz_with_empty(self):
        # GH 9188
        result = concat(
            [DataFrame(date_range("2000", periods=1, tz="UTC")), DataFrame()]
        )
        expected = DataFrame(date_range("2000", periods=1, tz="UTC"))
        tm.assert_frame_equal(result, expected)


class TestPeriodConcat:
    def test_concat_period_series(self):
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-10-01", "2016-01-01"], freq="D"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="Period[D]")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_period_multiple_freq_series(self):
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-10-01", "2016-01-01"], freq="M"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"

    def test_concat_period_other_series(self):
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="M"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"

    def test_concat_period_other_series2(self):
        # non-period
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(DatetimeIndex(["2015-11-01", "2015-12-01"]))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"

    def test_concat_period_other_series3(self):
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(["A", "B"])
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"


def test_concat_timedelta64_block():
    rng = to_timedelta(np.arange(10), unit="s")

    df = DataFrame({"time": rng})

    result = concat([df, df])
    tm.assert_frame_equal(result.iloc[:10], df)
    tm.assert_frame_equal(result.iloc[10:], df)


def test_concat_multiindex_datetime_nat():
    # GH#44900
    left = DataFrame({"a": 1}, index=MultiIndex.from_tuples([(1, pd.NaT)]))
    right = DataFrame(
        {"b": 2}, index=MultiIndex.from_tuples([(1, pd.NaT), (2, pd.NaT)])
    )
    result = concat([left, right], axis="columns")
    expected = DataFrame(
        {"a": [1.0, np.nan], "b": 2}, MultiIndex.from_tuples([(1, pd.NaT), (2, pd.NaT)])
    )
    tm.assert_frame_equal(result, expected)


def test_concat_float_datetime64(using_array_manager):
    # GH#32934
    df_time = DataFrame({"A": pd.array(["2000"], dtype="datetime64[ns]")})
    df_float = DataFrame({"A": pd.array([1.0], dtype="float64")})

    expected = DataFrame(
        {
            "A": [
                pd.array(["2000"], dtype="datetime64[ns]")[0],
                pd.array([1.0], dtype="float64")[0],
            ]
        },
        index=[0, 0],
    )
    result = concat([df_time, df_float])
    tm.assert_frame_equal(result, expected)

    expected = DataFrame({"A": pd.array([], dtype="object")})
    result = concat([df_time.iloc[:0], df_float.iloc[:0]])
    tm.assert_frame_equal(result, expected)

    expected = DataFrame({"A": pd.array([1.0], dtype="object")})
    result = concat([df_time.iloc[:0], df_float])
    tm.assert_frame_equal(result, expected)

    if not using_array_manager:
        expected = DataFrame({"A": pd.array(["2000"], dtype="datetime64[ns]")})
        msg = "The behavior of DataFrame concatenation with empty or all-NA entries"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = concat([df_time, df_float.iloc[:0]])
        tm.assert_frame_equal(result, expected)
    else:
        expected = DataFrame({"A": pd.array(["2000"], dtype="datetime64[ns]")}).astype(
            {"A": "object"}
        )
        result = concat([df_time, df_float.iloc[:0]])
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_map.py ===
from collections import (
    Counter,
    defaultdict,
)
from decimal import Decimal
import math

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    bdate_range,
    date_range,
    isna,
    timedelta_range,
)
import pandas._testing as tm


def test_series_map_box_timedelta():
    # GH#11349
    ser = Series(timedelta_range("1 day 1 s", periods=5, freq="h"))

    def f(x):
        return x.total_seconds()

    ser.map(f)


def test_map_callable(datetime_series):
    with np.errstate(all="ignore"):
        tm.assert_series_equal(datetime_series.map(np.sqrt), np.sqrt(datetime_series))

    # map function element-wise
    tm.assert_series_equal(datetime_series.map(math.exp), np.exp(datetime_series))

    # empty series
    s = Series(dtype=object, name="foo", index=Index([], name="bar"))
    rs = s.map(lambda x: x)
    tm.assert_series_equal(s, rs)

    # check all metadata (GH 9322)
    assert s is not rs
    assert s.index is rs.index
    assert s.dtype == rs.dtype
    assert s.name == rs.name

    # index but no data
    s = Series(index=[1, 2, 3], dtype=np.float64)
    rs = s.map(lambda x: x)
    tm.assert_series_equal(s, rs)


def test_map_same_length_inference_bug():
    s = Series([1, 2])

    def f(x):
        return (x, x + 1)

    s = Series([1, 2, 3])
    result = s.map(f)
    expected = Series([(1, 2), (2, 3), (3, 4)])
    tm.assert_series_equal(result, expected)

    s = Series(["foo,bar"])
    result = s.map(lambda x: x.split(","))
    expected = Series([("foo", "bar")])
    tm.assert_series_equal(result, expected)


def test_series_map_box_timestamps():
    # GH#2689, GH#2627
    ser = Series(date_range("1/1/2000", periods=3))

    def func(x):
        return (x.hour, x.day, x.month)

    result = ser.map(func)
    expected = Series([(0, 1, 1), (0, 2, 1), (0, 3, 1)])
    tm.assert_series_equal(result, expected)


def test_map_series_stringdtype(any_string_dtype, using_infer_string):
    # map test on StringDType, GH#40823
    ser1 = Series(
        data=["cat", "dog", "rabbit"],
        index=["id1", "id2", "id3"],
        dtype=any_string_dtype,
    )
    ser2 = Series(["id3", "id2", "id1", "id7000"], dtype=any_string_dtype)
    result = ser2.map(ser1)

    item = pd.NA
    if ser2.dtype == object:
        item = np.nan

    expected = Series(data=["rabbit", "dog", "cat", item], dtype=any_string_dtype)
    if using_infer_string and any_string_dtype == "object":
        expected = expected.astype("str")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected_dtype",
    [(["1-1", "1-1", np.nan], "category"), (["1-1", "1-2", np.nan], "str")],
)
def test_map_categorical_with_nan_values(data, expected_dtype):
    # GH 20714 bug fixed in: GH 24275
    def func(val):
        return val.split("-")[0]

    s = Series(data, dtype="category")

    result = s.map(func, na_action="ignore")
    expected = Series(["1", "1", np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_map_empty_integer_series():
    # GH52384
    s = Series([], dtype=int)
    result = s.map(lambda x: x)
    tm.assert_series_equal(result, s)


def test_map_empty_integer_series_with_datetime_index():
    # GH 21245
    s = Series([], index=date_range(start="2018-01-01", periods=0), dtype=int)
    result = s.map(lambda x: x)
    tm.assert_series_equal(result, s)


@pytest.mark.parametrize("func", [str, lambda x: str(x)])
def test_map_simple_str_callables_same_as_astype(
    string_series, func, using_infer_string
):
    # test that we are evaluating row-by-row first
    # before vectorized evaluation
    result = string_series.map(func)
    expected = string_series.astype(str if not using_infer_string else "str")
    tm.assert_series_equal(result, expected)


def test_list_raises(string_series):
    with pytest.raises(TypeError, match="'list' object is not callable"):
        string_series.map([lambda x: x])


def test_map():
    data = {
        "A": [0.0, 1.0, 2.0, 3.0, 4.0],
        "B": [0.0, 1.0, 0.0, 1.0, 0.0],
        "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
        "D": bdate_range("1/1/2009", periods=5),
    }

    source = Series(data["B"], index=data["C"])
    target = Series(data["C"][:4], index=data["D"][:4])

    merged = target.map(source)

    for k, v in merged.items():
        assert v == source[target[k]]

    # input could be a dict
    merged = target.map(source.to_dict())

    for k, v in merged.items():
        assert v == source[target[k]]


def test_map_datetime(datetime_series):
    # function
    result = datetime_series.map(lambda x: x * 2)
    tm.assert_series_equal(result, datetime_series * 2)


def test_map_category():
    # GH 10324
    a = Series([1, 2, 3, 4])
    b = Series(["even", "odd", "even", "odd"], dtype="category")
    c = Series(["even", "odd", "even", "odd"])

    exp = Series(["odd", "even", "odd", np.nan], dtype="category")
    tm.assert_series_equal(a.map(b), exp)
    exp = Series(["odd", "even", "odd", np.nan])
    tm.assert_series_equal(a.map(c), exp)


def test_map_category_numeric():
    a = Series(["a", "b", "c", "d"])
    b = Series([1, 2, 3, 4], index=pd.CategoricalIndex(["b", "c", "d", "e"]))
    c = Series([1, 2, 3, 4], index=Index(["b", "c", "d", "e"]))

    exp = Series([np.nan, 1, 2, 3])
    tm.assert_series_equal(a.map(b), exp)
    exp = Series([np.nan, 1, 2, 3])
    tm.assert_series_equal(a.map(c), exp)


def test_map_category_string():
    a = Series(["a", "b", "c", "d"])
    b = Series(
        ["B", "C", "D", "E"],
        dtype="category",
        index=pd.CategoricalIndex(["b", "c", "d", "e"]),
    )
    c = Series(["B", "C", "D", "E"], index=Index(["b", "c", "d", "e"]))

    exp = Series(
        pd.Categorical([np.nan, "B", "C", "D"], categories=["B", "C", "D", "E"])
    )
    tm.assert_series_equal(a.map(b), exp)
    exp = Series([np.nan, "B", "C", "D"])
    tm.assert_series_equal(a.map(c), exp)


@pytest.mark.filterwarnings(r"ignore:Dtype inference:FutureWarning")
def test_map_empty(request, index):
    if isinstance(index, MultiIndex):
        request.applymarker(
            pytest.mark.xfail(
                reason="Initializing a Series from a MultiIndex is not supported"
            )
        )

    s = Series(index)
    result = s.map({})

    expected = Series(np.nan, index=s.index)
    tm.assert_series_equal(result, expected)


def test_map_compat():
    # related GH 8024
    s = Series([True, True, False], index=[1, 2, 3])
    result = s.map({True: "foo", False: "bar"})
    expected = Series(["foo", "foo", "bar"], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)


def test_map_int():
    left = Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4})
    right = Series({1: 11, 2: 22, 3: 33})

    assert left.dtype == np.float64
    assert issubclass(right.dtype.type, np.integer)

    merged = left.map(right)
    assert merged.dtype == np.float64
    assert isna(merged["d"])
    assert not isna(merged["c"])


def test_map_type_inference():
    s = Series(range(3))
    s2 = s.map(lambda x: np.where(x == 0, 0, 1))
    assert issubclass(s2.dtype.type, np.integer)


def test_map_decimal(string_series):
    result = string_series.map(lambda x: Decimal(str(x)))
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)


def test_map_na_exclusion():
    s = Series([1.5, np.nan, 3, np.nan, 5])

    result = s.map(lambda x: x * 2, na_action="ignore")
    exp = s * 2
    tm.assert_series_equal(result, exp)


def test_map_dict_with_tuple_keys():
    """
    Due to new MultiIndex-ing behaviour in v0.14.0,
    dicts with tuple keys passed to map were being
    converted to a multi-index, preventing tuple values
    from being mapped properly.
    """
    # GH 18496
    df = DataFrame({"a": [(1,), (2,), (3, 4), (5, 6)]})
    label_mappings = {(1,): "A", (2,): "B", (3, 4): "A", (5, 6): "B"}

    df["labels"] = df["a"].map(label_mappings)
    df["expected_labels"] = Series(["A", "B", "A", "B"], index=df.index)
    # All labels should be filled now
    tm.assert_series_equal(df["labels"], df["expected_labels"], check_names=False)


def test_map_counter():
    s = Series(["a", "b", "c"], index=[1, 2, 3])
    counter = Counter()
    counter["b"] = 5
    counter["c"] += 1
    result = s.map(counter)
    expected = Series([0, 5, 1], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)


def test_map_defaultdict():
    s = Series([1, 2, 3], index=["a", "b", "c"])
    default_dict = defaultdict(lambda: "blank")
    default_dict[1] = "stuff"
    result = s.map(default_dict)
    expected = Series(["stuff", "blank", "blank"], index=["a", "b", "c"])
    tm.assert_series_equal(result, expected)


def test_map_dict_na_key():
    # https://github.com/pandas-dev/pandas/issues/17648
    # Checks that np.nan key is appropriately mapped
    s = Series([1, 2, np.nan])
    expected = Series(["a", "b", "c"])
    result = s.map({1: "a", 2: "b", np.nan: "c"})
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("na_action", [None, "ignore"])
def test_map_defaultdict_na_key(na_action):
    # GH 48813
    s = Series([1, 2, np.nan])
    default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", np.nan: "c"})
    result = s.map(default_map, na_action=na_action)
    expected = Series({0: "a", 1: "b", 2: "c" if na_action is None else np.nan})
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("na_action", [None, "ignore"])
def test_map_defaultdict_missing_key(na_action):
    # GH 48813
    s = Series([1, 2, np.nan])
    default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", 3: "c"})
    result = s.map(default_map, na_action=na_action)
    expected = Series({0: "a", 1: "b", 2: "missing" if na_action is None else np.nan})
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("na_action", [None, "ignore"])
def test_map_defaultdict_unmutated(na_action):
    # GH 48813
    s = Series([1, 2, np.nan])
    default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", np.nan: "c"})
    expected_default_map = default_map.copy()
    s.map(default_map, na_action=na_action)
    assert default_map == expected_default_map


@pytest.mark.parametrize("arg_func", [dict, Series])
def test_map_dict_ignore_na(arg_func):
    # GH#47527
    mapping = arg_func({1: 10, np.nan: 42})
    ser = Series([1, np.nan, 2])
    result = ser.map(mapping, na_action="ignore")
    expected = Series([10, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_map_defaultdict_ignore_na():
    # GH#47527
    mapping = defaultdict(int, {1: 10, np.nan: 42})
    ser = Series([1, np.nan, 2])
    result = ser.map(mapping)
    expected = Series([10, 42, 0])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "na_action, expected",
    [(None, Series([10.0, 42.0, np.nan])), ("ignore", Series([10, np.nan, np.nan]))],
)
def test_map_categorical_na_ignore(na_action, expected):
    # GH#47527
    values = pd.Categorical([1, np.nan, 2], categories=[10, 1, 2])
    ser = Series(values)
    result = ser.map({1: 10, np.nan: 42}, na_action=na_action)
    tm.assert_series_equal(result, expected)


def test_map_dict_subclass_with_missing():
    """
    Test Series.map with a dictionary subclass that defines __missing__,
    i.e. sets a default value (GH #15999).
    """

    class DictWithMissing(dict):
        def __missing__(self, key):
            return "missing"

    s = Series([1, 2, 3])
    dictionary = DictWithMissing({3: "three"})
    result = s.map(dictionary)
    expected = Series(["missing", "missing", "three"])
    tm.assert_series_equal(result, expected)


def test_map_dict_subclass_without_missing():
    class DictWithoutMissing(dict):
        pass

    s = Series([1, 2, 3])
    dictionary = DictWithoutMissing({3: "three"})
    result = s.map(dictionary)
    expected = Series([np.nan, np.nan, "three"])
    tm.assert_series_equal(result, expected)


def test_map_abc_mapping(non_dict_mapping_subclass):
    # https://github.com/pandas-dev/pandas/issues/29733
    # Check collections.abc.Mapping support as mapper for Series.map
    s = Series([1, 2, 3])
    not_a_dictionary = non_dict_mapping_subclass({3: "three"})
    result = s.map(not_a_dictionary)
    expected = Series([np.nan, np.nan, "three"])
    tm.assert_series_equal(result, expected)


def test_map_abc_mapping_with_missing(non_dict_mapping_subclass):
    # https://github.com/pandas-dev/pandas/issues/29733
    # Check collections.abc.Mapping support as mapper for Series.map
    class NonDictMappingWithMissing(non_dict_mapping_subclass):
        def __missing__(self, key):
            return "missing"

    s = Series([1, 2, 3])
    not_a_dictionary = NonDictMappingWithMissing({3: "three"})
    result = s.map(not_a_dictionary)
    # __missing__ is a dict concept, not a Mapping concept,
    # so it should not change the result!
    expected = Series([np.nan, np.nan, "three"])
    tm.assert_series_equal(result, expected)


def test_map_box_dt64(unit):
    vals = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]
    ser = Series(vals).dt.as_unit(unit)
    assert ser.dtype == f"datetime64[{unit}]"
    # boxed value must be Timestamp instance
    res = ser.map(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    exp = Series(["Timestamp_1_None", "Timestamp_2_None"])
    tm.assert_series_equal(res, exp)


def test_map_box_dt64tz(unit):
    vals = [
        pd.Timestamp("2011-01-01", tz="US/Eastern"),
        pd.Timestamp("2011-01-02", tz="US/Eastern"),
    ]
    ser = Series(vals).dt.as_unit(unit)
    assert ser.dtype == f"datetime64[{unit}, US/Eastern]"
    res = ser.map(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    exp = Series(["Timestamp_1_US/Eastern", "Timestamp_2_US/Eastern"])
    tm.assert_series_equal(res, exp)


def test_map_box_td64(unit):
    # timedelta
    vals = [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
    ser = Series(vals).dt.as_unit(unit)
    assert ser.dtype == f"timedelta64[{unit}]"
    res = ser.map(lambda x: f"{type(x).__name__}_{x.days}")
    exp = Series(["Timedelta_1", "Timedelta_2"])
    tm.assert_series_equal(res, exp)


def test_map_box_period():
    # period
    vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
    ser = Series(vals)
    assert ser.dtype == "Period[M]"
    res = ser.map(lambda x: f"{type(x).__name__}_{x.freqstr}")
    exp = Series(["Period_M", "Period_M"])
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize("na_action", [None, "ignore"])
def test_map_categorical(na_action, using_infer_string):
    values = pd.Categorical(list("ABBABCD"), categories=list("DCBA"), ordered=True)
    s = Series(values, name="XX", index=list("abcdefg"))

    result = s.map(lambda x: x.lower(), na_action=na_action)
    exp_values = pd.Categorical(list("abbabcd"), categories=list("dcba"), ordered=True)
    exp = Series(exp_values, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    tm.assert_categorical_equal(result.values, exp_values)

    result = s.map(lambda x: "A", na_action=na_action)
    exp = Series(["A"] * 7, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    assert result.dtype == object if not using_infer_string else "str"


@pytest.mark.parametrize(
    "na_action, expected",
    (
        [None, Series(["A", "B", "nan"], name="XX")],
        [
            "ignore",
            Series(
                ["A", "B", np.nan],
                name="XX",
                dtype=pd.CategoricalDtype(list("DCBA"), True),
            ),
        ],
    ),
)
def test_map_categorical_na_action(na_action, expected):
    dtype = pd.CategoricalDtype(list("DCBA"), ordered=True)
    values = pd.Categorical(list("AB") + [np.nan], dtype=dtype)
    s = Series(values, name="XX")
    result = s.map(str, na_action=na_action)
    tm.assert_series_equal(result, expected)


def test_map_datetimetz():
    values = date_range("2011-01-01", "2011-01-02", freq="h").tz_localize("Asia/Tokyo")
    s = Series(values, name="XX")

    # keep tz
    result = s.map(lambda x: x + pd.offsets.Day())
    exp_values = date_range("2011-01-02", "2011-01-03", freq="h").tz_localize(
        "Asia/Tokyo"
    )
    exp = Series(exp_values, name="XX")
    tm.assert_series_equal(result, exp)

    result = s.map(lambda x: x.hour)
    exp = Series(list(range(24)) + [0], name="XX", dtype=np.int64)
    tm.assert_series_equal(result, exp)

    # not vectorized
    def f(x):
        if not isinstance(x, pd.Timestamp):
            raise ValueError
        return str(x.tz)

    result = s.map(f)
    exp = Series(["Asia/Tokyo"] * 25, name="XX")
    tm.assert_series_equal(result, exp)


@pytest.mark.parametrize(
    "vals,mapping,exp",
    [
        (list("abc"), {np.nan: "not NaN"}, [np.nan] * 3 + ["not NaN"]),
        (list("abc"), {"a": "a letter"}, ["a letter"] + [np.nan] * 3),
        (list(range(3)), {0: 42}, [42] + [np.nan] * 3),
    ],
)
def test_map_missing_mixed(vals, mapping, exp):
    # GH20495
    s = Series(vals + [np.nan])
    result = s.map(mapping)
    exp = Series(exp)
    tm.assert_series_equal(result, exp)


def test_map_scalar_on_date_time_index_aware_series():
    # GH 25959
    # Calling map on a localized time series should not cause an error
    series = Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10, tz="UTC"),
        name="ts",
    )
    result = Series(series.index).map(lambda x: 1)
    tm.assert_series_equal(result, Series(np.ones(len(series)), dtype="int64"))


def test_map_float_to_string_precision():
    # GH 13228
    ser = Series(1 / 3)
    result = ser.map(lambda val: str(val)).to_dict()
    expected = {0: "0.3333333333333333"}
    assert result == expected


def test_map_to_timedelta():
    list_of_valid_strings = ["00:00:01", "00:00:02"]
    a = pd.to_timedelta(list_of_valid_strings)
    b = Series(list_of_valid_strings).map(pd.to_timedelta)
    tm.assert_series_equal(Series(a), b)

    list_of_strings = ["00:00:01", np.nan, pd.NaT, pd.NaT]

    a = pd.to_timedelta(list_of_strings)
    ser = Series(list_of_strings)
    b = ser.map(pd.to_timedelta)
    tm.assert_series_equal(Series(a), b)


def test_map_type():
    # GH 46719
    s = Series([3, "string", float], index=["a", "b", "c"])
    result = s.map(type)
    expected = Series([int, str, type], index=["a", "b", "c"])
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_soup.py ===
# -*- coding: utf-8 -*-
"""Tests of Beautiful Soup as a whole."""

import logging
import pickle
import pytest
from typing import Iterable

from bs4 import (
    BeautifulSoup,
    GuessedAtParserWarning,
    dammit,
)
from bs4.builder import (
    TreeBuilder,
)
from bs4.element import (
    AttributeValueList,
    XMLAttributeDict,
    Comment,
    PYTHON_SPECIFIC_ENCODINGS,
    Tag,
    NavigableString,
)
from bs4.filter import SoupStrainer
from bs4.exceptions import (
    ParserRejectedMarkup,
)
from bs4._warnings import (
    MarkupResemblesLocatorWarning,
)


from . import (
    default_builder,
    LXML_PRESENT,
    SoupTest,
)
import warnings
from typing import Type


class TestConstructor(SoupTest):
    def test_short_unicode_input(self):
        data = "<h1>éé</h1>"
        soup = self.soup(data)
        assert "éé" == soup.h1.string

    def test_embedded_null(self):
        data = "<h1>foo\0bar</h1>"
        soup = self.soup(data)
        assert "foo\0bar" == soup.h1.string

    def test_exclude_encodings(self):
        utf8_data = "Räksmörgås".encode("utf-8")
        soup = self.soup(utf8_data, exclude_encodings=["utf-8"])
        assert "windows-1252" == soup.original_encoding

    def test_custom_builder_class(self):
        # Verify that you can pass in a custom Builder class and
        # it'll be instantiated with the appropriate keyword arguments.
        class Mock(object):
            def __init__(self, **kwargs):
                self.called_with = kwargs
                self.is_xml = True
                self.store_line_numbers = False
                self.cdata_list_attributes = []
                self.preserve_whitespace_tags = []
                self.string_containers = {}
                self.attribute_dict_class = XMLAttributeDict
                self.attribute_value_list_class = AttributeValueList

            def initialize_soup(self, soup):
                pass

            def feed(self, markup):
                self.fed = markup

            def reset(self):
                pass

            def ignore(self, ignore):
                pass

            set_up_substitutions = can_be_empty_element = ignore

            def prepare_markup(self, *args, **kwargs):
                yield (
                    "prepared markup",
                    "original encoding",
                    "declared encoding",
                    "contains replacement characters",
                )

        kwargs = dict(
            var="value",
            # This is a deprecated BS3-era keyword argument, which
            # will be stripped out.
            convertEntities=True,
        )
        with warnings.catch_warnings(record=True):
            soup = BeautifulSoup("", builder=Mock, **kwargs)
        assert isinstance(soup.builder, Mock)
        assert dict(var="value") == soup.builder.called_with
        assert "prepared markup" == soup.builder.fed

        # You can also instantiate the TreeBuilder yourself. In this
        # case, that specific object is used and any keyword arguments
        # to the BeautifulSoup constructor are ignored.
        builder = Mock(**kwargs)
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup(
                "",
                builder=builder,
                ignored_value=True,
            )
        msg = str(w[0].message)
        assert msg.startswith(
            "Keyword arguments to the BeautifulSoup constructor will be ignored."
        )
        assert builder == soup.builder
        assert kwargs == builder.called_with

    def test_parser_markup_rejection(self):
        # If markup is completely rejected by the parser, an
        # explanatory ParserRejectedMarkup exception is raised.
        class Mock(TreeBuilder):
            def feed(self, *args, **kwargs):
                raise ParserRejectedMarkup("Nope.")

        def prepare_markup(self, markup, *args, **kwargs):
            # We're going to try two different ways of preparing this markup,
            # but feed() will reject both of them.
            yield markup, None, None, False
            yield markup, None, None, False


        with pytest.raises(ParserRejectedMarkup) as exc_info:
            BeautifulSoup("", builder=Mock)
        assert (
            "The markup you provided was rejected by the parser. Trying a different parser or a different encoding may help."
            in str(exc_info.value)
        )

    def test_cdata_list_attributes(self):
        # Most attribute values are represented as scalars, but the
        # HTML standard says that some attributes, like 'class' have
        # space-separated lists as values.
        markup = '<a id=" an id " class=" a class "></a>'
        soup = self.soup(markup)

        # Note that the spaces are stripped for 'class' but not for 'id'.
        a = soup.a
        assert " an id " == a["id"]
        assert ["a", "class"] == a["class"]

        # TreeBuilder takes an argument called 'multi_valued_attributes'  which lets
        # you customize or disable this. As always, you can customize the TreeBuilder
        # by passing in a keyword argument to the BeautifulSoup constructor.
        soup = self.soup(markup, builder=default_builder, multi_valued_attributes=None)
        assert " a class " == soup.a["class"]

        # Here are two ways of saying that `id` is a multi-valued
        # attribute in this context, but 'class' is not.
        for switcheroo in ({"*": "id"}, {"a": "id"}):
            with warnings.catch_warnings(record=True):
                # This will create a warning about not explicitly
                # specifying a parser, but we'll ignore it.
                soup = self.soup(
                    markup, builder=None, multi_valued_attributes=switcheroo
                )
            a = soup.a
            assert ["an", "id"] == a["id"]
            assert " a class " == a["class"]

    def test_replacement_classes(self):
        # Test the ability to pass in replacements for element classes
        # which will be used when building the tree.
        class TagPlus(Tag):
            pass

        class StringPlus(NavigableString):
            pass

        class CommentPlus(Comment):
            pass

        soup = self.soup(
            "<a><b>foo</b>bar</a><!--whee-->",
            element_classes={
                Tag: TagPlus,
                NavigableString: StringPlus,
                Comment: CommentPlus,
            },
        )

        # The tree was built with TagPlus, StringPlus, and CommentPlus objects,
        # rather than Tag, String, and Comment objects.
        assert all(
            isinstance(x, (TagPlus, StringPlus, CommentPlus)) for x in soup.descendants
        )

    def test_alternate_string_containers(self):
        # Test the ability to customize the string containers for
        # different types of tags.
        class PString(NavigableString):
            pass

        class BString(NavigableString):
            pass

        soup = self.soup(
            "<div>Hello.<p>Here is <b>some <i>bolded</i></b> text",
            string_containers={
                "b": BString,
                "p": PString,
            },
        )

        # The string before the <p> tag is a regular NavigableString.
        assert isinstance(soup.div.contents[0], NavigableString)

        # The string inside the <p> tag, but not inside the <i> tag,
        # is a PString.
        assert isinstance(soup.p.contents[0], PString)

        # Every string inside the <b> tag is a BString, even the one that
        # was also inside an <i> tag.
        for s in soup.b.strings:
            assert isinstance(s, BString)

        # Now that parsing was complete, the string_container_stack
        # (where this information was kept) has been cleared out.
        assert [] == soup.string_container_stack

    @pytest.mark.parametrize("bad_markup", [1, False, lambda x: False])
    def test_invalid_markup_type(self, bad_markup):
        with pytest.raises(TypeError) as exc_info:
            BeautifulSoup(bad_markup, "html.parser")
        assert (
            f"Incoming markup is of an invalid type: {bad_markup!r}. Markup must be a string, a bytestring, or an open filehandle."
            in str(exc_info.value)
        )


class TestOutput(SoupTest):
    @pytest.mark.parametrize(
        "eventual_encoding,actual_encoding",
        [
            ("utf-8", "utf-8"),
            ("utf-16", "utf-16"),
        ],
    )
    def test_decode_xml_declaration(self, eventual_encoding, actual_encoding):
        # Most of the time, calling decode() on an XML document will
        # give you a document declaration that mentions the encoding
        # you intend to use when encoding the document as a
        # bytestring.
        soup = self.soup("<tag></tag>")
        soup.is_xml = True
        assert (
            f'<?xml version="1.0" encoding="{actual_encoding}"?>\n<tag></tag>'
            == soup.decode(eventual_encoding=eventual_encoding)
        )

    @pytest.mark.parametrize(
        "eventual_encoding", [x for x in PYTHON_SPECIFIC_ENCODINGS] + [None]
    )
    def test_decode_xml_declaration_with_missing_or_python_internal_eventual_encoding(
        self, eventual_encoding
    ):
        # But if you pass a Python internal encoding into decode(), or
        # omit the eventual_encoding altogether, the document
        # declaration won't mention any particular encoding.
        soup = BeautifulSoup("<tag></tag>", "html.parser")
        soup.is_xml = True
        assert '<?xml version="1.0"?>\n<tag></tag>' == soup.decode(
            eventual_encoding=eventual_encoding
        )

    def test(self):
        # BeautifulSoup subclasses Tag and extends the decode() method.
        # Make sure the other Tag methods which call decode() call
        # it correctly.
        soup = self.soup("<tag></tag>")
        assert b"<tag></tag>" == soup.encode(encoding="utf-8")
        assert b"<tag></tag>" == soup.encode_contents(encoding="utf-8")
        assert "<tag></tag>" == soup.decode_contents()
        assert "<tag>\n</tag>\n" == soup.prettify()


class TestWarnings(SoupTest):
    # Note that some of the tests in this class create BeautifulSoup
    # objects directly rather than using self.soup(). That's
    # because SoupTest.soup is defined in a different file,
    # which will throw off the assertion in _assert_warning
    # that the code that triggered the warning is in the same
    # file as the test.

    def _assert_warning(
        self, warnings: Iterable[warnings.WarningMessage], cls: Type[Warning]
    ) -> warnings.WarningMessage:
        for w in warnings:
            if isinstance(w.message, cls):
                assert w.filename == __file__
                return w
        raise Exception("%s warning not found in %r" % (cls, warnings))

    def _assert_no_parser_specified(self, w: Iterable[warnings.WarningMessage]) -> None:
        warning = self._assert_warning(w, GuessedAtParserWarning)
        message = str(warning.message)
        assert message.startswith(GuessedAtParserWarning.MESSAGE[:60])

    def test_warning_if_no_parser_specified(self):
        with warnings.catch_warnings(record=True) as w:
            BeautifulSoup("<a><b></b></a>")
        self._assert_no_parser_specified(w)

    def test_warning_if_parser_specified_too_vague(self):
        with warnings.catch_warnings(record=True) as w:
            BeautifulSoup("<a><b></b></a>", "html")
        self._assert_no_parser_specified(w)

    def test_no_warning_if_explicit_parser_specified(self):
        with warnings.catch_warnings(record=True) as w:
            self.soup("<a><b></b></a>")
        assert [] == w

    def test_warning_if_strainer_filters_everything(self):
        strainer = SoupStrainer(name="a", string="b")
        with warnings.catch_warnings(record=True) as w:
            self.soup("<a><b></b></a>", parse_only=strainer)
        warning = self._assert_warning(w, UserWarning)
        msg = str(warning.message)
        assert msg.startswith("The given value for parse_only will exclude everything:")

    def test_parseOnlyThese_renamed_to_parse_only(self):
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup(
                "<a><b></b></a>",
                "html.parser",
                parseOnlyThese=SoupStrainer("b"),
            )
        warning = self._assert_warning(w, DeprecationWarning)
        msg = str(warning.message)
        assert "parseOnlyThese" in msg
        assert "parse_only" in msg
        assert b"<b></b>" == soup.encode()

    def test_fromEncoding_renamed_to_from_encoding(self):
        with warnings.catch_warnings(record=True) as w:
            utf8 = b"\xc3\xa9"
            soup = BeautifulSoup(utf8, "html.parser", fromEncoding="utf8")
        warning = self._assert_warning(w, DeprecationWarning)
        msg = str(warning.message)
        assert "fromEncoding" in msg
        assert "from_encoding" in msg
        assert "utf8" == soup.original_encoding

    def test_unrecognized_keyword_argument(self):
        with pytest.raises(TypeError):
            self.soup("<a>", no_such_argument=True)

    @pytest.mark.parametrize(
        "markup",
        [
            "markup.html",
            "markup.htm",
            "markup.HTML",
            "markup.txt",
            "markup.xhtml",
            "markup.xml",
            "/home/user/file.txt",
            r"c:\user\file.html" r"\\server\share\path\file.XhTml",
        ],
    )
    def test_resembles_filename_warning(self, markup):
        # A warning is issued if the "markup" looks like the name of
        # an HTML or text file, or a full path to a file on disk.
        with warnings.catch_warnings(record=True) as w:
            BeautifulSoup(markup, "html.parser")
            warning = self._assert_warning(w, MarkupResemblesLocatorWarning)
            assert "looks more like a filename" in str(warning.message)

    @pytest.mark.parametrize(
        "markup",
        [
            "filename",
            "markuphtml",
            "markup.com",
            "",
            # Excluded due to an irrelevant file extension.
            "markup.js",
            "markup.jpg",
            "markup.markup",
            # Excluded due to the lack of any file extension.
            "/home/user/file",
            r"c:\user\file.html" r"\\server\share\path\file",
            # Excluded because of two consecutive slashes _and_ the
            # colon.
            "log message containing a url http://www.url.com/ right there.html",
            # Excluded for containing various characters or combinations
            # not usually found in filenames.
            "two  consecutive  spaces.html",
            "two//consecutive//slashes.html",
            "looks/like/a/filename/but/oops/theres/a#comment.html",
            "two\nlines.html",
            "contains?.html",
            "contains*.html",
            "contains#.html",
            "contains&.html",
            "contains;.html",
            "contains>.html",
            "contains<.html",
            "contains$.html",
            "contains|.html",
            "contains:.html",
            ":-at-the-front.html",
        ],
    )
    def test_resembles_filename_no_warning(self, markup):
        # The 'looks more like a filename' warning is not issued if
        # the markup looks like a bare string, a domain name, or a
        # file that's not an HTML file.
        with warnings.catch_warnings(record=True) as w:
            self.soup(markup)
        assert [] == w

    def test_url_warning_with_bytes_url(self):
        url = b"http://www.crummybytes.com/"
        with warnings.catch_warnings(record=True) as warning_list:
            BeautifulSoup(url, "html.parser")
        warning = self._assert_warning(warning_list, MarkupResemblesLocatorWarning)
        assert "looks more like a URL" in str(warning.message)
        assert url not in str(warning.message).encode("utf8")

    def test_url_warning_with_unicode_url(self):
        url = "http://www.crummyunicode.com/"
        with warnings.catch_warnings(record=True) as warning_list:
            # note - this url must differ from the bytes one otherwise
            # python's warnings system swallows the second warning
            BeautifulSoup(url, "html.parser")
        warning = self._assert_warning(warning_list, MarkupResemblesLocatorWarning)
        assert "looks more like a URL" in str(warning.message)
        assert url not in str(warning.message)

    def test_url_warning_with_bytes_and_space(self):
        # Here the markup contains something besides a URL, so no warning
        # is issued.
        with warnings.catch_warnings(record=True) as warning_list:
            self.soup(b"http://www.crummybytes.com/ is great")
        assert not any("looks more like a URL" in str(w.message) for w in warning_list)

    def test_url_warning_with_unicode_and_space(self):
        with warnings.catch_warnings(record=True) as warning_list:
            self.soup("http://www.crummyunicode.com/ is great")
        assert not any("looks more like a URL" in str(w.message) for w in warning_list)


class TestSelectiveParsing(SoupTest):
    def test_parse_with_soupstrainer(self):
        markup = "No<b>Yes</b><a>No<b>Yes <c>Yes</c></b>"
        strainer = SoupStrainer("b")
        soup = self.soup(markup, parse_only=strainer)
        assert soup.encode() == b"<b>Yes</b><b>Yes <c>Yes</c></b>"


class TestNewTag(SoupTest):
    """Test the BeautifulSoup.new_tag() method."""

    def test_new_tag(self):
        soup = self.soup("")
        new_tag = soup.new_tag("foo", string="txt", bar="baz", attrs={"name": "a name"})
        assert isinstance(new_tag, Tag)
        assert "foo" == new_tag.name
        assert new_tag.string == "txt"
        assert dict(bar="baz", name="a name") == new_tag.attrs
        assert None is new_tag.parent

        # string can be null
        new_tag = soup.new_tag("foo")
        assert None is new_tag.string
        new_tag = soup.new_tag("foo", string=None)
        assert None is new_tag.string

        # Or the empty string
        new_tag = soup.new_tag("foo", string="")
        assert "" == new_tag.string

    @pytest.mark.skipif(
        not LXML_PRESENT, reason="lxml not installed, cannot parse XML document"
    )
    def test_xml_tag_inherits_self_closing_rules_from_builder(self):
        xml_soup = BeautifulSoup("", "xml")
        xml_br = xml_soup.new_tag("br")
        xml_p = xml_soup.new_tag("p")

        # Both the <br> and <p> tag are empty-element, just because
        # they have no contents.
        assert b"<br/>" == xml_br.encode()
        assert b"<p/>" == xml_p.encode()

    def test_tag_inherits_self_closing_rules_from_builder(self):
        html_soup = BeautifulSoup("", "html.parser")
        html_br = html_soup.new_tag("br")
        html_p = html_soup.new_tag("p")

        # The HTML builder users HTML's rules about which tags are
        # empty-element tags, and the new tags reflect these rules.
        assert b"<br/>" == html_br.encode()
        assert b"<p></p>" == html_p.encode()


class TestNewString(SoupTest):
    """Test the BeautifulSoup.new_string() method."""

    def test_new_string_creates_navigablestring(self):
        soup = self.soup("")
        s = soup.new_string("foo")
        assert "foo" == s
        assert isinstance(s, NavigableString)

    def test_new_string_can_create_navigablestring_subclass(self):
        soup = self.soup("")
        s = soup.new_string("foo", Comment)
        assert "foo" == s
        assert isinstance(s, Comment)


class TestPickle(SoupTest):
    # Test our ability to pickle the BeautifulSoup object itself.

    def test_normal_pickle(self):
        soup = self.soup("<a>some markup</a>")
        pickled = pickle.dumps(soup)
        unpickled = pickle.loads(pickled)
        assert "some markup" == unpickled.a.string

    def test_pickle_with_no_builder(self):
        # We had a bug that prevented pickling from working if
        # the builder wasn't set.
        soup = self.soup("some markup")
        soup.builder = None
        pickled = pickle.dumps(soup)
        unpickled = pickle.loads(pickled)
        assert "some markup" == unpickled.string


class TestEncodingConversion(SoupTest):
    # Test Beautiful Soup's ability to decode and encode from various
    # encodings.

    def setup_method(self):
        self.unicode_data = '<html><head><meta charset="utf-8"/></head><body><foo>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</foo></body></html>'
        self.utf8_data = self.unicode_data.encode("utf-8")
        # Just so you know what it looks like.
        assert (
            self.utf8_data
            == b'<html><head><meta charset="utf-8"/></head><body><foo>Sacr\xc3\xa9 bleu!</foo></body></html>'
        )

    def test_ascii_in_unicode_out(self):
        # ASCII input is converted to Unicode. The original_encoding
        # attribute is set to 'utf-8', a superset of ASCII.
        chardet = dammit._chardet_dammit
        logging.disable(logging.WARNING)
        try:

            def noop(str):
                return None

            # Disable chardet, which will realize that the ASCII is ASCII.
            dammit._chardet_dammit = noop
            ascii = b"<foo>a</foo>"
            soup_from_ascii = self.soup(ascii)
            unicode_output = soup_from_ascii.decode()
            assert isinstance(unicode_output, str)
            assert unicode_output == self.document_for(ascii.decode())
            assert soup_from_ascii.original_encoding.lower() == "utf-8"
        finally:
            logging.disable(logging.NOTSET)
            dammit._chardet_dammit = chardet

    def test_unicode_in_unicode_out(self):
        # Unicode input is left alone. The original_encoding attribute
        # is not set.
        soup_from_unicode = self.soup(self.unicode_data)
        assert soup_from_unicode.decode() == self.unicode_data
        assert soup_from_unicode.foo.string == "Sacr\xe9 bleu!"
        assert soup_from_unicode.original_encoding is None

    def test_utf8_in_unicode_out(self):
        # UTF-8 input is converted to Unicode. The original_encoding
        # attribute is set.
        soup_from_utf8 = self.soup(self.utf8_data)
        assert soup_from_utf8.decode() == self.unicode_data
        assert soup_from_utf8.foo.string == "Sacr\xe9 bleu!"

    def test_utf8_out(self):
        # The internal data structures can be encoded as UTF-8.
        soup_from_unicode = self.soup(self.unicode_data)
        assert soup_from_unicode.encode("utf-8") == self.utf8_data

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_quadrature.py ===
from sympy.core import S, Rational
from sympy.integrals.quadrature import (gauss_legendre, gauss_laguerre,
                                        gauss_hermite, gauss_gen_laguerre,
                                        gauss_chebyshev_t, gauss_chebyshev_u,
                                        gauss_jacobi, gauss_lobatto)


def test_legendre():
    x, w = gauss_legendre(1, 17)
    assert [str(r) for r in x] == ['0']
    assert [str(r) for r in w] == ['2.0000000000000000']

    x, w = gauss_legendre(2, 17)
    assert [str(r) for r in x] == [
            '-0.57735026918962576',
            '0.57735026918962576']
    assert [str(r) for r in w] == [
            '1.0000000000000000',
            '1.0000000000000000']

    x, w = gauss_legendre(3, 17)
    assert [str(r) for r in x] == [
            '-0.77459666924148338',
            '0',
            '0.77459666924148338']
    assert [str(r) for r in w] == [
            '0.55555555555555556',
            '0.88888888888888889',
            '0.55555555555555556']

    x, w = gauss_legendre(4, 17)
    assert [str(r) for r in x] == [
            '-0.86113631159405258',
            '-0.33998104358485626',
            '0.33998104358485626',
            '0.86113631159405258']
    assert [str(r) for r in w] == [
            '0.34785484513745386',
            '0.65214515486254614',
            '0.65214515486254614',
            '0.34785484513745386']


def test_legendre_precise():
    x, w = gauss_legendre(3, 40)
    assert [str(r) for r in x] == [
            '-0.7745966692414833770358530799564799221666',
            '0',
            '0.7745966692414833770358530799564799221666']
    assert [str(r) for r in w] == [
            '0.5555555555555555555555555555555555555556',
            '0.8888888888888888888888888888888888888889',
            '0.5555555555555555555555555555555555555556']


def test_laguerre():
    x, w = gauss_laguerre(1, 17)
    assert [str(r) for r in x] == ['1.0000000000000000']
    assert [str(r) for r in w] == ['1.0000000000000000']

    x, w = gauss_laguerre(2, 17)
    assert [str(r) for r in x] == [
            '0.58578643762690495',
            '3.4142135623730950']
    assert [str(r) for r in w] == [
            '0.85355339059327376',
            '0.14644660940672624']

    x, w = gauss_laguerre(3, 17)
    assert [str(r) for r in x] == [
            '0.41577455678347908',
            '2.2942803602790417',
            '6.2899450829374792',
            ]
    assert [str(r) for r in w] == [
            '0.71109300992917302',
            '0.27851773356924085',
            '0.010389256501586136',
            ]

    x, w = gauss_laguerre(4, 17)
    assert [str(r) for r in x] == [
            '0.32254768961939231',
            '1.7457611011583466',
            '4.5366202969211280',
            '9.3950709123011331']
    assert [str(r) for r in w] == [
            '0.60315410434163360',
            '0.35741869243779969',
            '0.038887908515005384',
            '0.00053929470556132745']

    x, w = gauss_laguerre(5, 17)
    assert [str(r) for r in x] == [
            '0.26356031971814091',
            '1.4134030591065168',
            '3.5964257710407221',
            '7.0858100058588376',
            '12.640800844275783']
    assert [str(r) for r in w] == [
            '0.52175561058280865',
            '0.39866681108317593',
            '0.075942449681707595',
            '0.0036117586799220485',
            '2.3369972385776228e-5']


def test_laguerre_precise():
    x, w = gauss_laguerre(3, 40)
    assert [str(r) for r in x] == [
            '0.4157745567834790833115338731282744735466',
            '2.294280360279041719822050361359593868960',
            '6.289945082937479196866415765512131657493']
    assert [str(r) for r in w] == [
            '0.7110930099291730154495901911425944313094',
            '0.2785177335692408488014448884567264810349',
            '0.01038925650158613574896492040067908765572']


def test_hermite():
    x, w = gauss_hermite(1, 17)
    assert [str(r) for r in x] == ['0']
    assert [str(r) for r in w] == ['1.7724538509055160']

    x, w = gauss_hermite(2, 17)
    assert [str(r) for r in x] == [
            '-0.70710678118654752',
            '0.70710678118654752']
    assert [str(r) for r in w] == [
            '0.88622692545275801',
            '0.88622692545275801']

    x, w = gauss_hermite(3, 17)
    assert [str(r) for r in x] == [
            '-1.2247448713915890',
            '0',
            '1.2247448713915890']
    assert [str(r) for r in w] == [
            '0.29540897515091934',
            '1.1816359006036774',
            '0.29540897515091934']

    x, w = gauss_hermite(4, 17)
    assert [str(r) for r in x] == [
            '-1.6506801238857846',
            '-0.52464762327529032',
            '0.52464762327529032',
            '1.6506801238857846']
    assert [str(r) for r in w] == [
            '0.081312835447245177',
            '0.80491409000551284',
            '0.80491409000551284',
            '0.081312835447245177']

    x, w = gauss_hermite(5, 17)
    assert [str(r) for r in x] == [
            '-2.0201828704560856',
            '-0.95857246461381851',
            '0',
            '0.95857246461381851',
            '2.0201828704560856']
    assert [str(r) for r in w] == [
            '0.019953242059045913',
            '0.39361932315224116',
            '0.94530872048294188',
            '0.39361932315224116',
            '0.019953242059045913']


def test_hermite_precise():
    x, w = gauss_hermite(3, 40)
    assert [str(r) for r in x] == [
        '-1.224744871391589049098642037352945695983',
        '0',
        '1.224744871391589049098642037352945695983']
    assert [str(r) for r in w] == [
        '0.2954089751509193378830279138901908637996',
        '1.181635900603677351532111655560763455198',
        '0.2954089751509193378830279138901908637996']


def test_gen_laguerre():
    x, w = gauss_gen_laguerre(1, Rational(-1, 2), 17)
    assert [str(r) for r in x] == ['0.50000000000000000']
    assert [str(r) for r in w] == ['1.7724538509055160']

    x, w = gauss_gen_laguerre(2, Rational(-1, 2), 17)
    assert [str(r) for r in x] == [
            '0.27525512860841095',
            '2.7247448713915890']
    assert [str(r) for r in w] == [
            '1.6098281800110257',
            '0.16262567089449035']

    x, w = gauss_gen_laguerre(3, Rational(-1, 2), 17)
    assert [str(r) for r in x] == [
            '0.19016350919348813',
            '1.7844927485432516',
            '5.5253437422632603']
    assert [str(r) for r in w] == [
            '1.4492591904487850',
            '0.31413464064571329',
            '0.0090600198110176913']

    x, w = gauss_gen_laguerre(4, Rational(-1, 2), 17)
    assert [str(r) for r in x] == [
            '0.14530352150331709',
            '1.3390972881263614',
            '3.9269635013582872',
            '8.5886356890120343']
    assert [str(r) for r in w] == [
            '1.3222940251164826',
            '0.41560465162978376',
            '0.034155966014826951',
            '0.00039920814442273524']

    x, w = gauss_gen_laguerre(5, Rational(-1, 2), 17)
    assert [str(r) for r in x] == [
            '0.11758132021177814',
            '1.0745620124369040',
            '3.0859374437175500',
            '6.4147297336620305',
            '11.807189489971737']
    assert [str(r) for r in w] == [
            '1.2217252674706516',
            '0.48027722216462937',
            '0.067748788910962126',
            '0.0026872914935624654',
            '1.5280865710465241e-5']

    x, w = gauss_gen_laguerre(1, 2, 17)
    assert [str(r) for r in x] == ['3.0000000000000000']
    assert [str(r) for r in w] == ['2.0000000000000000']

    x, w = gauss_gen_laguerre(2, 2, 17)
    assert [str(r) for r in x] == [
            '2.0000000000000000',
            '6.0000000000000000']
    assert [str(r) for r in w] == [
            '1.5000000000000000',
            '0.50000000000000000']

    x, w = gauss_gen_laguerre(3, 2, 17)
    assert [str(r) for r in x] == [
            '1.5173870806774125',
            '4.3115831337195203',
            '9.1710297856030672']
    assert [str(r) for r in w] == [
            '1.0374949614904253',
            '0.90575000470306537',
            '0.056755033806509347']

    x, w = gauss_gen_laguerre(4, 2, 17)
    assert [str(r) for r in x] == [
            '1.2267632635003021',
            '3.4125073586969460',
            '6.9026926058516134',
            '12.458036771951139']
    assert [str(r) for r in w] == [
            '0.72552499769865438',
            '1.0634242919791946',
            '0.20669613102835355',
            '0.0043545792937974889']

    x, w = gauss_gen_laguerre(5, 2, 17)
    assert [str(r) for r in x] == [
            '1.0311091440933816',
            '2.8372128239538217',
            '5.6202942725987079',
            '9.6829098376640271',
            '15.828473921690062']
    assert [str(r) for r in w] == [
            '0.52091739683509184',
            '1.0667059331592211',
            '0.38354972366693113',
            '0.028564233532974658',
            '0.00026271280578124935']


def test_gen_laguerre_precise():
    x, w = gauss_gen_laguerre(3, Rational(-1, 2), 40)
    assert [str(r) for r in x] == [
            '0.1901635091934881328718554276203028970878',
            '1.784492748543251591186722461957367638500',
            '5.525343742263260275941422110422329464413']
    assert [str(r) for r in w] == [
            '1.449259190448785048183829411195134343108',
            '0.3141346406457132878326231270167565378246',
            '0.009060019811017691281714945129254301865020']

    x, w = gauss_gen_laguerre(3, 2, 40)
    assert [str(r) for r in x] == [
            '1.517387080677412495020323111016672547482',
            '4.311583133719520302881184669723530562299',
            '9.171029785603067202098492219259796890218']
    assert [str(r) for r in w] == [
            '1.037494961490425285817554606541269153041',
            '0.9057500047030653669269785048806009945254',
            '0.05675503380650934725546688857812985243312']


def test_chebyshev_t():
    x, w = gauss_chebyshev_t(1, 17)
    assert [str(r) for r in x] == ['0']
    assert [str(r) for r in w] == ['3.1415926535897932']

    x, w = gauss_chebyshev_t(2, 17)
    assert [str(r) for r in x] == [
            '0.70710678118654752',
            '-0.70710678118654752']
    assert [str(r) for r in w] == [
            '1.5707963267948966',
            '1.5707963267948966']

    x, w = gauss_chebyshev_t(3, 17)
    assert [str(r) for r in x] == [
            '0.86602540378443865',
            '0',
            '-0.86602540378443865']
    assert [str(r) for r in w] == [
            '1.0471975511965977',
            '1.0471975511965977',
            '1.0471975511965977']

    x, w = gauss_chebyshev_t(4, 17)
    assert [str(r) for r in x] == [
            '0.92387953251128676',
            '0.38268343236508977',
            '-0.38268343236508977',
            '-0.92387953251128676']
    assert [str(r) for r in w] == [
            '0.78539816339744831',
            '0.78539816339744831',
            '0.78539816339744831',
            '0.78539816339744831']

    x, w = gauss_chebyshev_t(5, 17)
    assert [str(r) for r in x] == [
            '0.95105651629515357',
            '0.58778525229247313',
            '0',
            '-0.58778525229247313',
            '-0.95105651629515357']
    assert [str(r) for r in w] == [
            '0.62831853071795865',
            '0.62831853071795865',
            '0.62831853071795865',
            '0.62831853071795865',
            '0.62831853071795865']


def test_chebyshev_t_precise():
    x, w = gauss_chebyshev_t(3, 40)
    assert [str(r) for r in x] == [
            '0.8660254037844386467637231707529361834714',
            '0',
            '-0.8660254037844386467637231707529361834714']
    assert [str(r) for r in w] == [
            '1.047197551196597746154214461093167628066',
            '1.047197551196597746154214461093167628066',
            '1.047197551196597746154214461093167628066']


def test_chebyshev_u():
    x, w = gauss_chebyshev_u(1, 17)
    assert [str(r) for r in x] == ['0']
    assert [str(r) for r in w] == ['1.5707963267948966']

    x, w = gauss_chebyshev_u(2, 17)
    assert [str(r) for r in x] == [
            '0.50000000000000000',
            '-0.50000000000000000']
    assert [str(r) for r in w] == [
            '0.78539816339744831',
            '0.78539816339744831']

    x, w = gauss_chebyshev_u(3, 17)
    assert [str(r) for r in x] == [
            '0.70710678118654752',
            '0',
            '-0.70710678118654752']
    assert [str(r) for r in w] == [
            '0.39269908169872415',
            '0.78539816339744831',
            '0.39269908169872415']

    x, w = gauss_chebyshev_u(4, 17)
    assert [str(r) for r in x] == [
            '0.80901699437494742',
            '0.30901699437494742',
            '-0.30901699437494742',
            '-0.80901699437494742']
    assert [str(r) for r in w] == [
            '0.21707871342270599',
            '0.56831944997474231',
            '0.56831944997474231',
            '0.21707871342270599']

    x, w = gauss_chebyshev_u(5, 17)
    assert [str(r) for r in x] == [
            '0.86602540378443865',
            '0.50000000000000000',
            '0',
            '-0.50000000000000000',
            '-0.86602540378443865']
    assert [str(r) for r in w] == [
            '0.13089969389957472',
            '0.39269908169872415',
            '0.52359877559829887',
            '0.39269908169872415',
            '0.13089969389957472']


def test_chebyshev_u_precise():
    x, w = gauss_chebyshev_u(3, 40)
    assert [str(r) for r in x] == [
            '0.7071067811865475244008443621048490392848',
            '0',
            '-0.7071067811865475244008443621048490392848']
    assert [str(r) for r in w] == [
            '0.3926990816987241548078304229099378605246',
            '0.7853981633974483096156608458198757210493',
            '0.3926990816987241548078304229099378605246']


def test_jacobi():
    x, w = gauss_jacobi(1, Rational(-1, 2), S.Half, 17)
    assert [str(r) for r in x] == ['0.50000000000000000']
    assert [str(r) for r in w] == ['3.1415926535897932']

    x, w = gauss_jacobi(2, Rational(-1, 2), S.Half, 17)
    assert [str(r) for r in x] == [
            '-0.30901699437494742',
            '0.80901699437494742']
    assert [str(r) for r in w] == [
            '0.86831485369082398',
            '2.2732777998989693']

    x, w = gauss_jacobi(3, Rational(-1, 2), S.Half, 17)
    assert [str(r) for r in x] == [
            '-0.62348980185873353',
            '0.22252093395631440',
            '0.90096886790241913']
    assert [str(r) for r in w] == [
            '0.33795476356635433',
            '1.0973322242791115',
            '1.7063056657443274']

    x, w = gauss_jacobi(4, Rational(-1, 2), S.Half, 17)
    assert [str(r) for r in x] == [
            '-0.76604444311897804',
            '-0.17364817766693035',
            '0.50000000000000000',
            '0.93969262078590838']
    assert [str(r) for r in w] == [
            '0.16333179083642836',
            '0.57690240318269103',
            '1.0471975511965977',
            '1.3541609083740761']

    x, w = gauss_jacobi(5, Rational(-1, 2), S.Half, 17)
    assert [str(r) for r in x] == [
            '-0.84125353283118117',
            '-0.41541501300188643',
            '0.14231483827328514',
            '0.65486073394528506',
            '0.95949297361449739']
    assert [str(r) for r in w] == [
            '0.090675770007435372',
            '0.33391416373675607',
            '0.65248870981926643',
            '0.94525424081394926',
            '1.1192597692123861']

    x, w = gauss_jacobi(1, 2, 3, 17)
    assert [str(r) for r in x] == ['0.14285714285714286']
    assert [str(r) for r in w] == ['1.0666666666666667']

    x, w = gauss_jacobi(2, 2, 3, 17)
    assert [str(r) for r in x] == [
            '-0.24025307335204215',
            '0.46247529557426437']
    assert [str(r) for r in w] == [
            '0.48514624517838660',
            '0.58152042148828007']

    x, w = gauss_jacobi(3, 2, 3, 17)
    assert [str(r) for r in x] == [
            '-0.46115870378089762',
            '0.10438533038323902',
            '0.62950064612493132']
    assert [str(r) for r in w] == [
            '0.17937613502213266',
            '0.61595640991147154',
            '0.27133412173306246']

    x, w = gauss_jacobi(4, 2, 3, 17)
    assert [str(r) for r in x] == [
            '-0.59903470850824782',
            '-0.14761105199952565',
            '0.32554377081188859',
            '0.72879429738819258']
    assert [str(r) for r in w] == [
            '0.067809641836772187',
            '0.38956404952032481',
            '0.47995970868024150',
            '0.12933326662932816']

    x, w = gauss_jacobi(5, 2, 3, 17)
    assert [str(r) for r in x] == [
            '-0.69045775012676106',
            '-0.32651993134900065',
            '0.082337849552034905',
            '0.47517887061283164',
            '0.79279429464422850']
    assert [str(r) for r in w] == [
            '0.027410178066337099',
            '0.21291786060364828',
            '0.43908437944395081',
            '0.32220656547221822',
            '0.065047683080512268']


def test_jacobi_precise():
    x, w = gauss_jacobi(3, Rational(-1, 2), S.Half, 40)
    assert [str(r) for r in x] == [
            '-0.6234898018587335305250048840042398106323',
            '0.2225209339563144042889025644967947594664',
            '0.9009688679024191262361023195074450511659']
    assert [str(r) for r in w] == [
            '0.3379547635663543330553835737094171534907',
            '1.097332224279111467485302294320899710461',
            '1.706305665744327437921957515249186020246']

    x, w = gauss_jacobi(3, 2, 3, 40)
    assert [str(r) for r in x] == [
            '-0.4611587037808976179121958105554375981274',
            '0.1043853303832390210914918407615869143233',
            '0.6295006461249313240934312425211234110769']
    assert [str(r) for r in w] == [
            '0.1793761350221326596137764371503859752628',
            '0.6159564099114715430909548532229749439714',
            '0.2713341217330624639619353762933057474325']


def test_lobatto():
    x, w = gauss_lobatto(2, 17)
    assert [str(r) for r in x] == [
            '-1',
            '1']
    assert [str(r) for r in w] == [
            '1.0000000000000000',
            '1.0000000000000000']

    x, w = gauss_lobatto(3, 17)
    assert [str(r) for r in x] == [
            '-1',
            '0',
            '1']
    assert [str(r) for r in w] == [
            '0.33333333333333333',
            '1.3333333333333333',
            '0.33333333333333333']

    x, w = gauss_lobatto(4, 17)
    assert [str(r) for r in x] == [
            '-1',
            '-0.44721359549995794',
            '0.44721359549995794',
            '1']
    assert [str(r) for r in w] == [
            '0.16666666666666667',
            '0.83333333333333333',
            '0.83333333333333333',
            '0.16666666666666667']

    x, w = gauss_lobatto(5, 17)
    assert [str(r) for r in x] == [
            '-1',
            '-0.65465367070797714',
            '0',
            '0.65465367070797714',
            '1']
    assert [str(r) for r in w] == [
            '0.10000000000000000',
            '0.54444444444444444',
            '0.71111111111111111',
            '0.54444444444444444',
            '0.10000000000000000']


def test_lobatto_precise():
    x, w = gauss_lobatto(3, 40)
    assert [str(r) for r in x] == [
            '-1',
            '0',
            '1']
    assert [str(r) for r in w] == [
            '0.3333333333333333333333333333333333333333',
            '1.333333333333333333333333333333333333333',
            '0.3333333333333333333333333333333333333333']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_polyclasses.py ===
"""Tests for OO layer of several polynomial representations. """

from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
                                    NotInvertible)
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises, warns_deprecated_sympy

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]

def test_DMP___init__():
    f = DMP([[ZZ(0)], [], [ZZ(0), ZZ(1), ZZ(2)], [ZZ(3)]], ZZ)

    assert f._rep == [[1, 2], [3]]
    assert f.dom == ZZ
    assert f.lev == 1

    f = DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ, 1)

    assert f._rep == [[1, 2], [3]]
    assert f.dom == ZZ
    assert f.lev == 1

    f = DMP.from_dict({(1, 1): ZZ(1), (0, 0): ZZ(2)}, 1, ZZ)

    assert f._rep == [[1, 0], [2]]
    assert f.dom == ZZ
    assert f.lev == 1


def test_DMP_rep_deprecation():
    f = DMP([1, 2, 3], ZZ)

    with warns_deprecated_sympy():
        assert f.rep == [1, 2, 3]


def test_DMP___eq__():
    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ) == \
        DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ)

    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ) == \
        DMP([[QQ(1), QQ(2)], [QQ(3)]], QQ)
    assert DMP([[QQ(1), QQ(2)], [QQ(3)]], QQ) == \
        DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ)

    assert DMP([[[ZZ(1)]]], ZZ) != DMP([[ZZ(1)]], ZZ)
    assert DMP([[ZZ(1)]], ZZ) != DMP([[[ZZ(1)]]], ZZ)


def test_DMP___bool__():
    assert bool(DMP([[]], ZZ)) is False
    assert bool(DMP([[ZZ(1)]], ZZ)) is True


def test_DMP_to_dict():
    f = DMP([[ZZ(3)], [], [ZZ(2)], [], [ZZ(8)]], ZZ)

    assert f.to_dict() == \
        {(4, 0): 3, (2, 0): 2, (0, 0): 8}
    assert f.to_sympy_dict() == \
        {(4, 0): ZZ.to_sympy(3), (2, 0): ZZ.to_sympy(2), (0, 0):
         ZZ.to_sympy(8)}


def test_DMP_properties():
    assert DMP([[]], ZZ).is_zero is True
    assert DMP([[ZZ(1)]], ZZ).is_zero is False

    assert DMP([[ZZ(1)]], ZZ).is_one is True
    assert DMP([[ZZ(2)]], ZZ).is_one is False

    assert DMP([[ZZ(1)]], ZZ).is_ground is True
    assert DMP([[ZZ(1)], [ZZ(2)], [ZZ(1)]], ZZ).is_ground is False

    assert DMP([[ZZ(1)], [ZZ(2), ZZ(0)], [ZZ(1), ZZ(0)]], ZZ).is_sqf is True
    assert DMP([[ZZ(1)], [ZZ(2), ZZ(0)], [ZZ(1), ZZ(0), ZZ(0)]], ZZ).is_sqf is False

    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ).is_monic is True
    assert DMP([[ZZ(2), ZZ(2)], [ZZ(3)]], ZZ).is_monic is False

    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ).is_primitive is True
    assert DMP([[ZZ(2), ZZ(4)], [ZZ(6)]], ZZ).is_primitive is False


def test_DMP_arithmetics():
    f = DMP([[ZZ(2)], [ZZ(2), ZZ(0)]], ZZ)

    assert f.mul_ground(2) == DMP([[ZZ(4)], [ZZ(4), ZZ(0)]], ZZ)
    assert f.quo_ground(2) == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)

    raises(ExactQuotientFailed, lambda: f.exquo_ground(3))

    f = DMP([[ZZ(-5)]], ZZ)
    g = DMP([[ZZ(5)]], ZZ)

    assert f.abs() == g
    assert abs(f) == g

    assert g.neg() == f
    assert -g == f

    h = DMP([[]], ZZ)

    assert f.add(g) == h
    assert f + g == h
    assert g + f == h
    assert f + 5 == h
    assert 5 + f == h

    h = DMP([[ZZ(-10)]], ZZ)

    assert f.sub(g) == h
    assert f - g == h
    assert g - f == -h
    assert f - 5 == h
    assert 5 - f == -h

    h = DMP([[ZZ(-25)]], ZZ)

    assert f.mul(g) == h
    assert f * g == h
    assert g * f == h
    assert f * 5 == h
    assert 5 * f == h

    h = DMP([[ZZ(25)]], ZZ)

    assert f.sqr() == h
    assert f.pow(2) == h
    assert f**2 == h

    raises(TypeError, lambda: f.pow('x'))

    f = DMP([[ZZ(1)], [], [ZZ(1), ZZ(0), ZZ(0)]], ZZ)
    g = DMP([[ZZ(2)], [ZZ(-2), ZZ(0)]], ZZ)

    q = DMP([[ZZ(2)], [ZZ(2), ZZ(0)]], ZZ)
    r = DMP([[ZZ(8), ZZ(0), ZZ(0)]], ZZ)

    assert f.pdiv(g) == (q, r)
    assert f.pquo(g) == q
    assert f.prem(g) == r

    raises(ExactQuotientFailed, lambda: f.pexquo(g))

    f = DMP([[ZZ(1)], [], [ZZ(1), ZZ(0), ZZ(0)]], ZZ)
    g = DMP([[ZZ(1)], [ZZ(-1), ZZ(0)]], ZZ)

    q = DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    r = DMP([[ZZ(2), ZZ(0), ZZ(0)]], ZZ)

    assert f.div(g) == (q, r)
    assert f.quo(g) == q
    assert f.rem(g) == r

    assert divmod(f, g) == (q, r)
    assert f // g == q
    assert f % g == r

    raises(ExactQuotientFailed, lambda: f.exquo(g))

    f = DMP([ZZ(1), ZZ(0), ZZ(-1)], ZZ)
    g = DMP([ZZ(2), ZZ(-2)], ZZ)

    q = DMP([], ZZ)
    r = f

    pq = DMP([ZZ(2), ZZ(2)], ZZ)
    pr = DMP([], ZZ)

    assert f.div(g) == (q, r)
    assert f.quo(g) == q
    assert f.rem(g) == r

    assert divmod(f, g) == (q, r)
    assert f // g == q
    assert f % g == r

    raises(ExactQuotientFailed, lambda: f.exquo(g))

    assert f.pdiv(g) == (pq, pr)
    assert f.pquo(g) == pq
    assert f.prem(g) == pr
    assert f.pexquo(g) == pq


def test_DMP_functionality():
    f = DMP([[ZZ(1)], [ZZ(2), ZZ(0)], [ZZ(1), ZZ(0), ZZ(0)]], ZZ)
    g = DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    h = DMP([[ZZ(1)]], ZZ)

    assert f.degree() == 2
    assert f.degree_list() == (2, 2)
    assert f.total_degree() == 2

    assert f.LC() == ZZ(1)
    assert f.TC() == ZZ(0)
    assert f.nth(1, 1) == ZZ(2)

    raises(TypeError, lambda: f.nth(0, 'x'))

    assert f.max_norm() == 2
    assert f.l1_norm() == 4

    u = DMP([[ZZ(2)], [ZZ(2), ZZ(0)]], ZZ)

    assert f.diff(m=1, j=0) == u
    assert f.diff(m=1, j=1) == u

    raises(TypeError, lambda: f.diff(m='x', j=0))

    u = DMP([ZZ(1), ZZ(2), ZZ(1)], ZZ)
    v = DMP([ZZ(1), ZZ(2), ZZ(1)], ZZ)

    assert f.eval(a=1, j=0) == u
    assert f.eval(a=1, j=1) == v

    assert f.eval(1).eval(1) == ZZ(4)

    assert f.cofactors(g) == (g, g, h)
    assert f.gcd(g) == g
    assert f.lcm(g) == f

    u = DMP([[QQ(45), QQ(30), QQ(5)]], QQ)
    v = DMP([[QQ(1), QQ(2, 3), QQ(1, 9)]], QQ)

    assert u.monic() == v

    assert (4*f).content() == ZZ(4)
    assert (4*f).primitive() == (ZZ(4), f)

    f = DMP([QQ(1,3), QQ(1)], QQ)
    g = DMP([QQ(1,7), QQ(1)], QQ)

    assert f.cancel(g) == f.cancel(g, include=True) == (
        DMP([QQ(7), QQ(21)], QQ),
        DMP([QQ(3), QQ(21)], QQ)
    )
    assert f.cancel(g, include=False) == (
        QQ(7),
        QQ(3),
        DMP([QQ(1), QQ(3)], QQ),
        DMP([QQ(1), QQ(7)], QQ)
    )

    f = DMP([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)], [ZZ(6)]], ZZ)

    assert f.trunc(3) == DMP([[ZZ(1)], [ZZ(-1)], [], [ZZ(1)], [ZZ(-1)], []], ZZ)

    f = DMP(f_4, ZZ)

    assert f.sqf_part() == -f
    assert f.sqf_list() == (ZZ(-1), [(-f, 1)])

    f = DMP([[ZZ(-1)], [], [], [ZZ(5)]], ZZ)
    g = DMP([[ZZ(3), ZZ(1)], [], []], ZZ)
    h = DMP([[ZZ(45), ZZ(30), ZZ(5)]], ZZ)

    r = DMP([ZZ(675), ZZ(675), ZZ(225), ZZ(25)], ZZ)

    assert f.subresultants(g) == [f, g, h]
    assert f.resultant(g) == r

    f = DMP([ZZ(1), ZZ(3), ZZ(9), ZZ(-13)], ZZ)

    assert f.discriminant() == -11664

    f = DMP([QQ(2), QQ(0)], QQ)
    g = DMP([QQ(1), QQ(0), QQ(-16)], QQ)

    s = DMP([QQ(1, 32), QQ(0)], QQ)
    t = DMP([QQ(-1, 16)], QQ)
    h = DMP([QQ(1)], QQ)

    assert f.half_gcdex(g) == (s, h)
    assert f.gcdex(g) == (s, t, h)

    assert f.invert(g) == s

    f = DMP([[QQ(1)], [QQ(2)], [QQ(3)]], QQ)

    raises(ValueError, lambda: f.half_gcdex(f))
    raises(ValueError, lambda: f.gcdex(f))

    raises(ValueError, lambda: f.invert(f))

    f = DMP(ZZ.map([1, 0, 20, 0, 150, 0, 500, 0, 625, -2, 0, -10, 9]), ZZ)
    g = DMP([ZZ(1), ZZ(0), ZZ(0), ZZ(-2), ZZ(9)], ZZ)
    h = DMP([ZZ(1), ZZ(0), ZZ(5), ZZ(0)], ZZ)

    assert g.compose(h) == f
    assert f.decompose() == [g, h]

    f = DMP([[QQ(1)], [QQ(2)], [QQ(3)]], QQ)

    raises(ValueError, lambda: f.decompose())
    raises(ValueError, lambda: f.sturm())


def test_DMP_exclude():
    f = [[[[[[[[[[[[[[[[[[[[[[[[[[ZZ(1)]], [[]]]]]]]]]]]]]]]]]]]]]]]]]]
    J = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 24, 25]

    assert DMP(f, ZZ).exclude() == (J, DMP([ZZ(1), ZZ(0)], ZZ))
    assert DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ).exclude() ==\
            ([], DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ))


def test_DMF__init__():
    f = DMF(([[0], [], [0, 1, 2], [3]], [[1, 2, 3]]), ZZ)

    assert f.num == [[1, 2], [3]]
    assert f.den == [[1, 2, 3]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF(([[1, 2], [3]], [[1, 2, 3]]), ZZ, 1)

    assert f.num == [[1, 2], [3]]
    assert f.den == [[1, 2, 3]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF(([[-1], [-2]], [[3], [-4]]), ZZ)

    assert f.num == [[-1], [-2]]
    assert f.den == [[3], [-4]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF(([[1], [2]], [[-3], [4]]), ZZ)

    assert f.num == [[-1], [-2]]
    assert f.den == [[3], [-4]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF(([[1], [2]], [[-3], [4]]), ZZ)

    assert f.num == [[-1], [-2]]
    assert f.den == [[3], [-4]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF(([[]], [[-3], [4]]), ZZ)

    assert f.num == [[]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF(17, ZZ, 1)

    assert f.num == [[17]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF(([[1], [2]]), ZZ)

    assert f.num == [[1], [2]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF([[0], [], [0, 1, 2], [3]], ZZ)

    assert f.num == [[1, 2], [3]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF({(1, 1): 1, (0, 0): 2}, ZZ, 1)

    assert f.num == [[1, 0], [2]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    f = DMF(([[QQ(1)], [QQ(2)]], [[-QQ(3)], [QQ(4)]]), QQ)

    assert f.num == [[-QQ(1)], [-QQ(2)]]
    assert f.den == [[QQ(3)], [-QQ(4)]]
    assert f.lev == 1
    assert f.dom == QQ

    f = DMF(([[QQ(1, 5)], [QQ(2, 5)]], [[-QQ(3, 7)], [QQ(4, 7)]]), QQ)

    assert f.num == [[-QQ(7)], [-QQ(14)]]
    assert f.den == [[QQ(15)], [-QQ(20)]]
    assert f.lev == 1
    assert f.dom == QQ

    raises(ValueError, lambda: DMF(([1], [[1]]), ZZ))
    raises(ZeroDivisionError, lambda: DMF(([1], []), ZZ))


def test_DMF__bool__():
    assert bool(DMF([[]], ZZ)) is False
    assert bool(DMF([[1]], ZZ)) is True


def test_DMF_properties():
    assert DMF([[]], ZZ).is_zero is True
    assert DMF([[]], ZZ).is_one is False

    assert DMF([[1]], ZZ).is_zero is False
    assert DMF([[1]], ZZ).is_one is True

    assert DMF(([[1]], [[2]]), ZZ).is_one is False


def test_DMF_arithmetics():
    f = DMF([[7], [-9]], ZZ)
    g = DMF([[-7], [9]], ZZ)

    assert f.neg() == -f == g

    f = DMF(([[1]], [[1], []]), ZZ)
    g = DMF(([[1]], [[1, 0]]), ZZ)

    h = DMF(([[1], [1, 0]], [[1, 0], []]), ZZ)

    assert f.add(g) == f + g == h
    assert g.add(f) == g + f == h

    h = DMF(([[-1], [1, 0]], [[1, 0], []]), ZZ)

    assert f.sub(g) == f - g == h

    h = DMF(([[1]], [[1, 0], []]), ZZ)

    assert f.mul(g) == f*g == h
    assert g.mul(f) == g*f == h

    h = DMF(([[1, 0]], [[1], []]), ZZ)

    assert f.quo(g) == f/g == h

    h = DMF(([[1]], [[1], [], [], []]), ZZ)

    assert f.pow(3) == f**3 == h

    h = DMF(([[1]], [[1, 0, 0, 0]]), ZZ)

    assert g.pow(3) == g**3 == h

    h = DMF(([[1, 0]], [[1]]), ZZ)

    assert g.pow(-1) == g**-1 == h


def test_ANP___init__():
    rep = [QQ(1), QQ(1)]
    mod = [QQ(1), QQ(0), QQ(1)]

    f = ANP(rep, mod, QQ)

    assert f.to_list() == [QQ(1), QQ(1)]
    assert f.mod_to_list() == [QQ(1), QQ(0), QQ(1)]
    assert f.dom == QQ

    rep = {1: QQ(1), 0: QQ(1)}
    mod = {2: QQ(1), 0: QQ(1)}

    f = ANP(rep, mod, QQ)

    assert f.to_list() == [QQ(1), QQ(1)]
    assert f.mod_to_list() == [QQ(1), QQ(0), QQ(1)]
    assert f.dom == QQ

    f = ANP(1, mod, QQ)

    assert f.to_list() == [QQ(1)]
    assert f.mod_to_list() == [QQ(1), QQ(0), QQ(1)]
    assert f.dom == QQ

    f = ANP([1, 0.5], mod, QQ)

    assert all(QQ.of_type(a) for a in f.to_list())

    raises(CoercionFailed, lambda: ANP([sqrt(2)], mod, QQ))


def test_ANP___eq__():
    a = ANP([QQ(1), QQ(1)], [QQ(1), QQ(0), QQ(1)], QQ)
    b = ANP([QQ(1), QQ(1)], [QQ(1), QQ(0), QQ(2)], QQ)

    assert (a == a) is True
    assert (a != a) is False

    assert (a == b) is False
    assert (a != b) is True

    b = ANP([QQ(1), QQ(2)], [QQ(1), QQ(0), QQ(1)], QQ)

    assert (a == b) is False
    assert (a != b) is True


def test_ANP___bool__():
    assert bool(ANP([], [QQ(1), QQ(0), QQ(1)], QQ)) is False
    assert bool(ANP([QQ(1)], [QQ(1), QQ(0), QQ(1)], QQ)) is True


def test_ANP_properties():
    mod = [QQ(1), QQ(0), QQ(1)]

    assert ANP([QQ(0)], mod, QQ).is_zero is True
    assert ANP([QQ(1)], mod, QQ).is_zero is False

    assert ANP([QQ(1)], mod, QQ).is_one is True
    assert ANP([QQ(2)], mod, QQ).is_one is False


def test_ANP_arithmetics():
    mod = [QQ(1), QQ(0), QQ(0), QQ(-2)]

    a = ANP([QQ(2), QQ(-1), QQ(1)], mod, QQ)
    b = ANP([QQ(1), QQ(2)], mod, QQ)

    c = ANP([QQ(-2), QQ(1), QQ(-1)], mod, QQ)

    assert a.neg() == -a == c

    c = ANP([QQ(2), QQ(0), QQ(3)], mod, QQ)

    assert a.add(b) == a + b == c
    assert b.add(a) == b + a == c

    c = ANP([QQ(2), QQ(-2), QQ(-1)], mod, QQ)

    assert a.sub(b) == a - b == c

    c = ANP([QQ(-2), QQ(2), QQ(1)], mod, QQ)

    assert b.sub(a) == b - a == c

    c = ANP([QQ(3), QQ(-1), QQ(6)], mod, QQ)

    assert a.mul(b) == a*b == c
    assert b.mul(a) == b*a == c

    c = ANP([QQ(-1, 43), QQ(9, 43), QQ(5, 43)], mod, QQ)

    assert a.pow(0) == a**(0) == ANP(1, mod, QQ)
    assert a.pow(1) == a**(1) == a

    assert a.pow(-1) == a**(-1) == c

    assert a.quo(a) == a.mul(a.pow(-1)) == a*a**(-1) == ANP(1, mod, QQ)

    c = ANP([], [1, 0, 0, -2], QQ)
    r1 = a.rem(b)

    (q, r2) = a.div(b)

    assert r1 == r2 == c == a % b

    raises(NotInvertible, lambda: a.div(c))
    raises(NotInvertible, lambda: a.rem(c))

    # Comparison with "hard-coded" value fails despite looking identical
    # from sympy import Rational
    # c = ANP([Rational(11, 10), Rational(-1, 5), Rational(-3, 5)], [1, 0, 0, -2], QQ)

    assert q == a/b # == c

def test_ANP_unify():
    mod_z = [ZZ(1), ZZ(0), ZZ(-2)]
    mod_q = [QQ(1), QQ(0), QQ(-2)]

    a = ANP([QQ(1)], mod_q, QQ)
    b = ANP([ZZ(1)], mod_z, ZZ)

    assert a.unify(b)[0] == QQ
    assert b.unify(a)[0] == QQ
    assert a.unify(a)[0] == QQ
    assert b.unify(b)[0] == ZZ

    assert a.unify_ANP(b)[-1] == QQ
    assert b.unify_ANP(a)[-1] == QQ
    assert a.unify_ANP(a)[-1] == QQ
    assert b.unify_ANP(b)[-1] == ZZ


def test_zero_poly():
    from sympy import Symbol
    x = Symbol('x')

    R_old = ZZ.old_poly_ring(x)
    zero_poly_old = R_old(0)
    cont_old, prim_old = zero_poly_old.primitive()

    assert cont_old == 0
    assert prim_old == zero_poly_old
    assert prim_old.is_primitive is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\base\test_conversion.py ===
import numpy as np
import pytest

from pandas.compat import HAS_PYARROW
from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import (
    CategoricalIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import StringArrayNumpySemantics
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics


class TestToIterable:
    # test that we convert an iterable to python types

    dtypes = [
        ("int8", int),
        ("int16", int),
        ("int32", int),
        ("int64", int),
        ("uint8", int),
        ("uint16", int),
        ("uint32", int),
        ("uint64", int),
        ("float16", float),
        ("float32", float),
        ("float64", float),
        ("datetime64[ns]", Timestamp),
        ("datetime64[ns, US/Eastern]", Timestamp),
        ("timedelta64[ns]", Timedelta),
    ]

    @pytest.mark.parametrize("dtype, rdtype", dtypes)
    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    def test_iterable(self, index_or_series, method, dtype, rdtype):
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        typ = index_or_series
        if dtype == "float16" and issubclass(typ, pd.Index):
            with pytest.raises(NotImplementedError, match="float16 indexes are not "):
                typ([1], dtype=dtype)
            return
        s = typ([1], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        "dtype, rdtype, obj",
        [
            ("object", object, "a"),
            ("object", int, 1),
            ("category", object, "a"),
            ("category", int, 1),
        ],
    )
    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    def test_iterable_object_and_category(
        self, index_or_series, method, dtype, rdtype, obj
    ):
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        typ = index_or_series
        s = typ([obj], dtype=dtype)
        result = method(s)[0]
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize("dtype, rdtype", dtypes)
    def test_iterable_items(self, dtype, rdtype):
        # gh-13258
        # test if items yields the correct boxed scalars
        # this only applies to series
        s = Series([1], dtype=dtype)
        _, result = next(iter(s.items()))
        assert isinstance(result, rdtype)

        _, result = next(iter(s.items()))
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        "dtype, rdtype", dtypes + [("object", int), ("category", int)]
    )
    def test_iterable_map(self, index_or_series, dtype, rdtype):
        # gh-13236
        # coerce iteration to underlying python / pandas types
        typ = index_or_series
        if dtype == "float16" and issubclass(typ, pd.Index):
            with pytest.raises(NotImplementedError, match="float16 indexes are not "):
                typ([1], dtype=dtype)
            return
        s = typ([1], dtype=dtype)
        result = s.map(type)[0]
        if not isinstance(rdtype, tuple):
            rdtype = (rdtype,)
        assert result in rdtype

    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    def test_categorial_datetimelike(self, method):
        i = CategoricalIndex([Timestamp("1999-12-31"), Timestamp("2000-12-31")])

        result = method(i)[0]
        assert isinstance(result, Timestamp)

    def test_iter_box_dt64(self, unit):
        vals = [Timestamp("2011-01-01"), Timestamp("2011-01-02")]
        ser = Series(vals).dt.as_unit(unit)
        assert ser.dtype == f"datetime64[{unit}]"
        for res, exp in zip(ser, vals):
            assert isinstance(res, Timestamp)
            assert res.tz is None
            assert res == exp
            assert res.unit == unit

    def test_iter_box_dt64tz(self, unit):
        vals = [
            Timestamp("2011-01-01", tz="US/Eastern"),
            Timestamp("2011-01-02", tz="US/Eastern"),
        ]
        ser = Series(vals).dt.as_unit(unit)

        assert ser.dtype == f"datetime64[{unit}, US/Eastern]"
        for res, exp in zip(ser, vals):
            assert isinstance(res, Timestamp)
            assert res.tz == exp.tz
            assert res == exp
            assert res.unit == unit

    def test_iter_box_timedelta64(self, unit):
        # timedelta
        vals = [Timedelta("1 days"), Timedelta("2 days")]
        ser = Series(vals).dt.as_unit(unit)
        assert ser.dtype == f"timedelta64[{unit}]"
        for res, exp in zip(ser, vals):
            assert isinstance(res, Timedelta)
            assert res == exp
            assert res.unit == unit

    def test_iter_box_period(self):
        # period
        vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
        s = Series(vals)
        assert s.dtype == "Period[M]"
        for res, exp in zip(s, vals):
            assert isinstance(res, pd.Period)
            assert res.freq == "ME"
            assert res == exp


@pytest.mark.parametrize(
    "arr, expected_type, dtype",
    [
        (np.array([0, 1], dtype=np.int64), np.ndarray, "int64"),
        (np.array(["a", "b"]), np.ndarray, "object"),
        (pd.Categorical(["a", "b"]), pd.Categorical, "category"),
        (
            pd.DatetimeIndex(["2017", "2018"], tz="US/Central"),
            DatetimeArray,
            "datetime64[ns, US/Central]",
        ),
        (
            pd.PeriodIndex([2018, 2019], freq="Y"),
            PeriodArray,
            pd.core.dtypes.dtypes.PeriodDtype("Y-DEC"),
        ),
        (pd.IntervalIndex.from_breaks([0, 1, 2]), IntervalArray, "interval"),
        (
            pd.DatetimeIndex(["2017", "2018"]),
            DatetimeArray,
            "datetime64[ns]",
        ),
        (
            pd.TimedeltaIndex([10**10]),
            TimedeltaArray,
            "m8[ns]",
        ),
    ],
)
def test_values_consistent(arr, expected_type, dtype, using_infer_string):
    if using_infer_string and dtype == "object":
        expected_type = (
            ArrowStringArrayNumpySemantics if HAS_PYARROW else StringArrayNumpySemantics
        )
    l_values = Series(arr)._values
    r_values = pd.Index(arr)._values
    assert type(l_values) is expected_type
    assert type(l_values) is type(r_values)

    tm.assert_equal(l_values, r_values)


@pytest.mark.parametrize("arr", [np.array([1, 2, 3])])
def test_numpy_array(arr):
    ser = Series(arr)
    result = ser.array
    expected = NumpyExtensionArray(arr)
    tm.assert_extension_array_equal(result, expected)


def test_numpy_array_all_dtypes(any_numpy_dtype):
    ser = Series(dtype=any_numpy_dtype)
    result = ser.array
    if np.dtype(any_numpy_dtype).kind == "M":
        assert isinstance(result, DatetimeArray)
    elif np.dtype(any_numpy_dtype).kind == "m":
        assert isinstance(result, TimedeltaArray)
    else:
        assert isinstance(result, NumpyExtensionArray)


@pytest.mark.parametrize(
    "arr, attr",
    [
        (pd.Categorical(["a", "b"]), "_codes"),
        (PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]"), "_ndarray"),
        (pd.array([0, np.nan], dtype="Int64"), "_data"),
        (IntervalArray.from_breaks([0, 1]), "_left"),
        (SparseArray([0, 1]), "_sparse_values"),
        (
            DatetimeArray._from_sequence(np.array([1, 2], dtype="datetime64[ns]")),
            "_ndarray",
        ),
        # tz-aware Datetime
        (
            DatetimeArray._from_sequence(
                np.array(
                    ["2000-01-01T12:00:00", "2000-01-02T12:00:00"], dtype="M8[ns]"
                ),
                dtype=DatetimeTZDtype(tz="US/Central"),
            ),
            "_ndarray",
        ),
    ],
)
def test_array(arr, attr, index_or_series, request):
    box = index_or_series

    result = box(arr, copy=False).array

    if attr:
        arr = getattr(arr, attr)
        result = getattr(result, attr)

    assert result is arr


def test_array_multiindex_raises():
    idx = pd.MultiIndex.from_product([["A"], ["a", "b"]])
    msg = "MultiIndex has no single backing array"
    with pytest.raises(ValueError, match=msg):
        idx.array


@pytest.mark.parametrize(
    "arr, expected, zero_copy",
    [
        (np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64), True),
        (pd.Categorical(["a", "b"]), np.array(["a", "b"], dtype=object), False),
        (
            pd.core.arrays.period_array(["2000", "2001"], freq="D"),
            np.array([pd.Period("2000", freq="D"), pd.Period("2001", freq="D")]),
            False,
        ),
        (pd.array([0, np.nan], dtype="Int64"), np.array([0, np.nan]), False),
        (
            IntervalArray.from_breaks([0, 1, 2]),
            np.array([pd.Interval(0, 1), pd.Interval(1, 2)], dtype=object),
            False,
        ),
        (SparseArray([0, 1]), np.array([0, 1], dtype=np.int64), False),
        # tz-naive datetime
        (
            DatetimeArray._from_sequence(np.array(["2000", "2001"], dtype="M8[ns]")),
            np.array(["2000", "2001"], dtype="M8[ns]"),
            True,
        ),
        # tz-aware stays tz`-aware
        (
            DatetimeArray._from_sequence(
                np.array(["2000-01-01T06:00:00", "2000-01-02T06:00:00"], dtype="M8[ns]")
            )
            .tz_localize("UTC")
            .tz_convert("US/Central"),
            np.array(
                [
                    Timestamp("2000-01-01", tz="US/Central"),
                    Timestamp("2000-01-02", tz="US/Central"),
                ]
            ),
            False,
        ),
        # Timedelta
        (
            TimedeltaArray._from_sequence(
                np.array([0, 3600000000000], dtype="i8").view("m8[ns]")
            ),
            np.array([0, 3600000000000], dtype="m8[ns]"),
            True,
        ),
        # GH#26406 tz is preserved in Categorical[dt64tz]
        (
            pd.Categorical(date_range("2016-01-01", periods=2, tz="US/Pacific")),
            np.array(
                [
                    Timestamp("2016-01-01", tz="US/Pacific"),
                    Timestamp("2016-01-02", tz="US/Pacific"),
                ]
            ),
            False,
        ),
    ],
)
def test_to_numpy(arr, expected, zero_copy, index_or_series_or_array):
    box = index_or_series_or_array

    with tm.assert_produces_warning(None):
        thing = box(arr)

    result = thing.to_numpy()
    tm.assert_numpy_array_equal(result, expected)

    result = np.asarray(thing)
    tm.assert_numpy_array_equal(result, expected)

    # Additionally, we check the `copy=` semantics for array/asarray
    # (these are implemented by us via `__array__`).
    result_cp1 = np.array(thing, copy=True)
    result_cp2 = np.array(thing, copy=True)
    # When called with `copy=True` NumPy/we should ensure a copy was made
    assert not np.may_share_memory(result_cp1, result_cp2)

    if not np_version_gt2:
        # copy=False semantics are only supported in NumPy>=2.
        return

    if not zero_copy:
        msg = "Starting with NumPy 2.0, the behavior of the 'copy' keyword has changed"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            np.array(thing, copy=False)

    else:
        result_nocopy1 = np.array(thing, copy=False)
        result_nocopy2 = np.array(thing, copy=False)
        # If copy=False was given, these must share the same data
        assert np.may_share_memory(result_nocopy1, result_nocopy2)


@pytest.mark.parametrize("as_series", [True, False])
@pytest.mark.parametrize(
    "arr", [np.array([1, 2, 3], dtype="int64"), np.array(["a", "b", "c"], dtype=object)]
)
def test_to_numpy_copy(arr, as_series, using_infer_string):
    obj = pd.Index(arr, copy=False)
    if as_series:
        obj = Series(obj.values, copy=False)

    # no copy by default
    result = obj.to_numpy()
    if using_infer_string and arr.dtype == object and obj.dtype.storage == "pyarrow":
        assert np.shares_memory(arr, result) is False
    else:
        assert np.shares_memory(arr, result) is True

    result = obj.to_numpy(copy=False)
    if using_infer_string and arr.dtype == object and obj.dtype.storage == "pyarrow":
        assert np.shares_memory(arr, result) is False
    else:
        assert np.shares_memory(arr, result) is True

    # copy=True
    result = obj.to_numpy(copy=True)
    assert np.shares_memory(arr, result) is False


@pytest.mark.parametrize("as_series", [True, False])
def test_to_numpy_dtype(as_series, unit):
    tz = "US/Eastern"
    obj = pd.DatetimeIndex(["2000", "2001"], tz=tz)
    if as_series:
        obj = Series(obj)

    # preserve tz by default
    result = obj.to_numpy()
    expected = np.array(
        [Timestamp("2000", tz=tz), Timestamp("2001", tz=tz)], dtype=object
    )
    tm.assert_numpy_array_equal(result, expected)

    result = obj.to_numpy(dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    result = obj.to_numpy(dtype="M8[ns]")
    expected = np.array(["2000-01-01T05", "2001-01-01T05"], dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "values, dtype, na_value, expected",
    [
        ([1, 2, None], "float64", 0, [1.0, 2.0, 0.0]),
        (
            [Timestamp("2000"), Timestamp("2000"), pd.NaT],
            None,
            Timestamp("2000"),
            [np.datetime64("2000-01-01T00:00:00.000000000")] * 3,
        ),
    ],
)
def test_to_numpy_na_value_numpy_dtype(
    index_or_series, values, dtype, na_value, expected
):
    obj = index_or_series(values)
    result = obj.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array(expected)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, multiindex, dtype, na_value, expected",
    [
        (
            [1, 2, None, 4],
            [(0, "a"), (0, "b"), (1, "b"), (1, "c")],
            float,
            None,
            [1.0, 2.0, np.nan, 4.0],
        ),
        (
            [1, 2, None, 4],
            [(0, "a"), (0, "b"), (1, "b"), (1, "c")],
            float,
            np.nan,
            [1.0, 2.0, np.nan, 4.0],
        ),
        (
            [1.0, 2.0, np.nan, 4.0],
            [("a", 0), ("a", 1), ("a", 2), ("b", 0)],
            int,
            0,
            [1, 2, 0, 4],
        ),
        (
            [Timestamp("2000"), Timestamp("2000"), pd.NaT],
            [(0, Timestamp("2021")), (0, Timestamp("2022")), (1, Timestamp("2000"))],
            None,
            Timestamp("2000"),
            [np.datetime64("2000-01-01T00:00:00.000000000")] * 3,
        ),
    ],
)
def test_to_numpy_multiindex_series_na_value(
    data, multiindex, dtype, na_value, expected
):
    index = pd.MultiIndex.from_tuples(multiindex)
    series = Series(data, index=index)
    result = series.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array(expected)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_kwargs_raises():
    # numpy
    s = Series([1, 2, 3])
    msg = r"to_numpy\(\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg):
        s.to_numpy(foo=True)

    # extension
    s = Series([1, 2, 3], dtype="Int64")
    with pytest.raises(TypeError, match=msg):
        s.to_numpy(foo=True)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [1, 2, None]},
        {"a": np.array([1, 2, 3]), "b": np.array([1, 2, np.nan])},
        {"a": pd.array([1, 2, 3]), "b": pd.array([1, 2, None])},
    ],
)
@pytest.mark.parametrize("dtype, na_value", [(float, np.nan), (object, None)])
def test_to_numpy_dataframe_na_value(data, dtype, na_value):
    # https://github.com/pandas-dev/pandas/issues/33820
    df = pd.DataFrame(data)
    result = df.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array([[1, 1], [2, 2], [3, na_value]], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            {"a": pd.array([1, 2, None])},
            np.array([[1.0], [2.0], [np.nan]], dtype=float),
        ),
        (
            {"a": [1, 2, 3], "b": [1, 2, 3]},
            np.array([[1, 1], [2, 2], [3, 3]], dtype=float),
        ),
    ],
)
def test_to_numpy_dataframe_single_block(data, expected):
    # https://github.com/pandas-dev/pandas/issues/33820
    df = pd.DataFrame(data)
    result = df.to_numpy(dtype=float, na_value=np.nan)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_dataframe_single_block_no_mutate():
    # https://github.com/pandas-dev/pandas/issues/33820
    result = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
    expected = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
    result.to_numpy(na_value=0.0)
    tm.assert_frame_equal(result, expected)


class TestAsArray:
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    def test_asarray_object_dt64(self, tz):
        ser = Series(date_range("2000", periods=2, tz=tz))

        with tm.assert_produces_warning(None):
            # Future behavior (for tzaware case) with no warning
            result = np.asarray(ser, dtype=object)

        expected = np.array(
            [Timestamp("2000-01-01", tz=tz), Timestamp("2000-01-02", tz=tz)]
        )
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_naive(self):
        # This shouldn't produce a warning.
        ser = Series(date_range("2000", periods=2))
        expected = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        result = np.asarray(ser)

        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_aware(self):
        tz = "US/Central"
        ser = Series(date_range("2000", periods=2, tz=tz))
        expected = np.array(["2000-01-01T06", "2000-01-02T06"], dtype="M8[ns]")
        result = np.asarray(ser, dtype="datetime64[ns]")

        tm.assert_numpy_array_equal(result, expected)

        # Old behavior with no warning
        result = np.asarray(ser, dtype="M8[ns]")

        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_direct.py ===
import os
import sys
from os.path import join

import pytest

import numpy as np
from numpy.random import (
    MT19937,
    PCG64,
    PCG64DXSM,
    SFC64,
    Generator,
    Philox,
    RandomState,
    SeedSequence,
    default_rng,
)
from numpy.random._common import interface
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

try:
    import cffi  # noqa: F401

    MISSING_CFFI = False
except ImportError:
    MISSING_CFFI = True

try:
    import ctypes  # noqa: F401

    MISSING_CTYPES = False
except ImportError:
    MISSING_CTYPES = False

if sys.flags.optimize > 1:
    # no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1
    # cffi cannot succeed
    MISSING_CFFI = True


pwd = os.path.dirname(os.path.abspath(__file__))


def assert_state_equal(actual, target):
    for key in actual:
        if isinstance(actual[key], dict):
            assert_state_equal(actual[key], target[key])
        elif isinstance(actual[key], np.ndarray):
            assert_array_equal(actual[key], target[key])
        else:
            assert actual[key] == target[key]


def uint32_to_float32(u):
    return ((u >> np.uint32(8)) * (1.0 / 2**24)).astype(np.float32)


def uniform32_from_uint64(x):
    x = np.uint64(x)
    upper = np.array(x >> np.uint64(32), dtype=np.uint32)
    lower = np.uint64(0xffffffff)
    lower = np.array(x & lower, dtype=np.uint32)
    joined = np.column_stack([lower, upper]).ravel()
    return uint32_to_float32(joined)


def uniform32_from_uint53(x):
    x = np.uint64(x) >> np.uint64(16)
    x = np.uint32(x & np.uint64(0xffffffff))
    return uint32_to_float32(x)


def uniform32_from_uint32(x):
    return uint32_to_float32(x)


def uniform32_from_uint(x, bits):
    if bits == 64:
        return uniform32_from_uint64(x)
    elif bits == 53:
        return uniform32_from_uint53(x)
    elif bits == 32:
        return uniform32_from_uint32(x)
    else:
        raise NotImplementedError


def uniform_from_uint(x, bits):
    if bits in (64, 63, 53):
        return uniform_from_uint64(x)
    elif bits == 32:
        return uniform_from_uint32(x)


def uniform_from_uint64(x):
    return (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def uniform_from_uint32(x):
    out = np.empty(len(x) // 2)
    for i in range(0, len(x), 2):
        a = x[i] >> 5
        b = x[i + 1] >> 6
        out[i // 2] = (a * 67108864.0 + b) / 9007199254740992.0
    return out


def uniform_from_dsfmt(x):
    return x.view(np.double) - 1.0


def gauss_from_uint(x, n, bits):
    if bits in (64, 63):
        doubles = uniform_from_uint64(x)
    elif bits == 32:
        doubles = uniform_from_uint32(x)
    else:  # bits == 'dsfmt'
        doubles = uniform_from_dsfmt(x)
    gauss = []
    loc = 0
    x1 = x2 = 0.0
    while len(gauss) < n:
        r2 = 2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * doubles[loc] - 1.0
            x2 = 2.0 * doubles[loc + 1] - 1.0
            r2 = x1 * x1 + x2 * x2
            loc += 2

        f = np.sqrt(-2.0 * np.log(r2) / r2)
        gauss.append(f * x2)
        gauss.append(f * x1)

    return gauss[:n]


def test_seedsequence():
    from numpy.random.bit_generator import (
        ISeedSequence,
        ISpawnableSeedSequence,
        SeedlessSeedSequence,
    )

    s1 = SeedSequence(range(10), spawn_key=(1, 2), pool_size=6)
    s1.spawn(10)
    s2 = SeedSequence(**s1.state)
    assert_equal(s1.state, s2.state)
    assert_equal(s1.n_children_spawned, s2.n_children_spawned)

    # The interfaces cannot be instantiated themselves.
    assert_raises(TypeError, ISeedSequence)
    assert_raises(TypeError, ISpawnableSeedSequence)
    dummy = SeedlessSeedSequence()
    assert_raises(NotImplementedError, dummy.generate_state, 10)
    assert len(dummy.spawn(10)) == 10


def test_generator_spawning():
    """ Test spawning new generators and bit_generators directly.
    """
    rng = np.random.default_rng()
    seq = rng.bit_generator.seed_seq
    new_ss = seq.spawn(5)
    expected_keys = [seq.spawn_key + (i,) for i in range(5)]
    assert [c.spawn_key for c in new_ss] == expected_keys

    new_bgs = rng.bit_generator.spawn(5)
    expected_keys = [seq.spawn_key + (i,) for i in range(5, 10)]
    assert [bg.seed_seq.spawn_key for bg in new_bgs] == expected_keys

    new_rngs = rng.spawn(5)
    expected_keys = [seq.spawn_key + (i,) for i in range(10, 15)]
    found_keys = [rng.bit_generator.seed_seq.spawn_key for rng in new_rngs]
    assert found_keys == expected_keys

    # Sanity check that streams are actually different:
    assert new_rngs[0].uniform() != new_rngs[1].uniform()


def test_non_spawnable():
    from numpy.random.bit_generator import ISeedSequence

    class FakeSeedSequence:
        def generate_state(self, n_words, dtype=np.uint32):
            return np.zeros(n_words, dtype=dtype)

    ISeedSequence.register(FakeSeedSequence)

    rng = np.random.default_rng(FakeSeedSequence())

    with pytest.raises(TypeError, match="The underlying SeedSequence"):
        rng.spawn(5)

    with pytest.raises(TypeError, match="The underlying SeedSequence"):
        rng.bit_generator.spawn(5)


class Base:
    dtype = np.uint64
    data2 = data1 = {}

    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.seed_error_type = TypeError
        cls.invalid_init_types = []
        cls.invalid_init_values = []

    @classmethod
    def _read_csv(cls, filename):
        with open(filename) as csv:
            seed = csv.readline()
            seed = seed.split(',')
            seed = [int(s.strip(), 0) for s in seed[1:]]
            data = []
            for line in csv:
                data.append(int(line.split(',')[-1].strip(), 0))
            return {'seed': seed, 'data': np.array(data, dtype=cls.dtype)}

    def test_raw(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        uints = bit_generator.random_raw(1000)
        assert_equal(uints, self.data1['data'])

        bit_generator = self.bit_generator(*self.data1['seed'])
        uints = bit_generator.random_raw()
        assert_equal(uints, self.data1['data'][0])

        bit_generator = self.bit_generator(*self.data2['seed'])
        uints = bit_generator.random_raw(1000)
        assert_equal(uints, self.data2['data'])

    def test_random_raw(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        uints = bit_generator.random_raw(output=False)
        assert uints is None
        uints = bit_generator.random_raw(1000, output=False)
        assert uints is None

    def test_gauss_inv(self):
        n = 25
        rs = RandomState(self.bit_generator(*self.data1['seed']))
        gauss = rs.standard_normal(n)
        assert_allclose(gauss,
                        gauss_from_uint(self.data1['data'], n, self.bits))

        rs = RandomState(self.bit_generator(*self.data2['seed']))
        gauss = rs.standard_normal(25)
        assert_allclose(gauss,
                        gauss_from_uint(self.data2['data'], n, self.bits))

    def test_uniform_double(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        vals = uniform_from_uint(self.data1['data'], self.bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float64)

        rs = Generator(self.bit_generator(*self.data2['seed']))
        vals = uniform_from_uint(self.data2['data'], self.bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float64)

    def test_uniform_float(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        vals = uniform32_from_uint(self.data1['data'], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

        rs = Generator(self.bit_generator(*self.data2['seed']))
        vals = uniform32_from_uint(self.data2['data'], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

    def test_repr(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert 'Generator' in repr(rs)
        assert f'{id(rs):#x}'.upper().replace('X', 'x') in repr(rs)

    def test_str(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert 'Generator' in str(rs)
        assert str(self.bit_generator.__name__) in str(rs)
        assert f'{id(rs):#x}'.upper().replace('X', 'x') not in str(rs)

    def test_pickle(self):
        import pickle

        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        bitgen_pkl = pickle.dumps(bit_generator)
        reloaded = pickle.loads(bitgen_pkl)
        reloaded_state = reloaded.state
        assert_array_equal(Generator(bit_generator).standard_normal(1000),
                           Generator(reloaded).standard_normal(1000))
        assert bit_generator is not reloaded
        assert_state_equal(reloaded_state, state)

        ss = SeedSequence(100)
        aa = pickle.loads(pickle.dumps(ss))
        assert_equal(ss.state, aa.state)

    def test_pickle_preserves_seed_sequence(self):
        # GH 26234
        # Add explicit test that bit generators preserve seed sequences
        import pickle

        bit_generator = self.bit_generator(*self.data1['seed'])
        ss = bit_generator.seed_seq
        bg_plk = pickle.loads(pickle.dumps(bit_generator))
        ss_plk = bg_plk.seed_seq
        assert_equal(ss.state, ss_plk.state)
        assert_equal(ss.pool, ss_plk.pool)

        bit_generator.seed_seq.spawn(10)
        bg_plk = pickle.loads(pickle.dumps(bit_generator))
        ss_plk = bg_plk.seed_seq
        assert_equal(ss.state, ss_plk.state)
        assert_equal(ss.n_children_spawned, ss_plk.n_children_spawned)

    def test_invalid_state_type(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        with pytest.raises(TypeError):
            bit_generator.state = {'1'}

    def test_invalid_state_value(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        state['bit_generator'] = 'otherBitGenerator'
        with pytest.raises(ValueError):
            bit_generator.state = state

    def test_invalid_init_type(self):
        bit_generator = self.bit_generator
        for st in self.invalid_init_types:
            with pytest.raises(TypeError):
                bit_generator(*st)

    def test_invalid_init_values(self):
        bit_generator = self.bit_generator
        for st in self.invalid_init_values:
            with pytest.raises((ValueError, OverflowError)):
                bit_generator(*st)

    def test_benchmark(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        bit_generator._benchmark(1)
        bit_generator._benchmark(1, 'double')
        with pytest.raises(ValueError):
            bit_generator._benchmark(1, 'int32')

    @pytest.mark.skipif(MISSING_CFFI, reason='cffi not available')
    def test_cffi(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        cffi_interface = bit_generator.cffi
        assert isinstance(cffi_interface, interface)
        other_cffi_interface = bit_generator.cffi
        assert other_cffi_interface is cffi_interface

    @pytest.mark.skipif(MISSING_CTYPES, reason='ctypes not available')
    def test_ctypes(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        ctypes_interface = bit_generator.ctypes
        assert isinstance(ctypes_interface, interface)
        other_ctypes_interface = bit_generator.ctypes
        assert other_ctypes_interface is ctypes_interface

    def test_getstate(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        alt_state = bit_generator.__getstate__()
        assert isinstance(alt_state, tuple)
        assert_state_equal(state, alt_state[0])
        assert isinstance(alt_state[1], SeedSequence)

class TestPhilox(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = Philox
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, './data/philox-testset-1.csv'))
        cls.data2 = cls._read_csv(
            join(pwd, './data/philox-testset-2.csv'))
        cls.seed_error_type = TypeError
        cls.invalid_init_types = []
        cls.invalid_init_values = [(1, None, 1), (-1,), (None, None, 2 ** 257 + 1)]

    def test_set_key(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        keyed = self.bit_generator(counter=state['state']['counter'],
                                   key=state['state']['key'])
        assert_state_equal(bit_generator.state, keyed.state)


class TestPCG64(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, './data/pcg64-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/pcg64-testset-2.csv'))
        cls.seed_error_type = (ValueError, TypeError)
        cls.invalid_init_types = [(3.2,), ([None],), (1, None)]
        cls.invalid_init_values = [(-1,)]

    def test_advance_symmetry(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        state = rs.bit_generator.state
        step = -0x9e3779b97f4a7c150000000000000000
        rs.bit_generator.advance(step)
        val_neg = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(2**128 + step)
        val_pos = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(10 * 2**128 + step)
        val_big = rs.integers(10)
        assert val_neg == val_pos
        assert val_big == val_pos

    def test_advange_large(self):
        rs = Generator(self.bit_generator(38219308213743))
        pcg = rs.bit_generator
        state = pcg.state["state"]
        initial_state = 287608843259529770491897792873167516365
        assert state["state"] == initial_state
        pcg.advance(sum(2**i for i in (96, 64, 32, 16, 8, 4, 2, 1)))
        state = pcg.state["state"]
        advanced_state = 135275564607035429730177404003164635391
        assert state["state"] == advanced_state


class TestPCG64DXSM(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64DXSM
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, './data/pcg64dxsm-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/pcg64dxsm-testset-2.csv'))
        cls.seed_error_type = (ValueError, TypeError)
        cls.invalid_init_types = [(3.2,), ([None],), (1, None)]
        cls.invalid_init_values = [(-1,)]

    def test_advance_symmetry(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        state = rs.bit_generator.state
        step = -0x9e3779b97f4a7c150000000000000000
        rs.bit_generator.advance(step)
        val_neg = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(2**128 + step)
        val_pos = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(10 * 2**128 + step)
        val_big = rs.integers(10)
        assert val_neg == val_pos
        assert val_big == val_pos

    def test_advange_large(self):
        rs = Generator(self.bit_generator(38219308213743))
        pcg = rs.bit_generator
        state = pcg.state
        initial_state = 287608843259529770491897792873167516365
        assert state["state"]["state"] == initial_state
        pcg.advance(sum(2**i for i in (96, 64, 32, 16, 8, 4, 2, 1)))
        state = pcg.state["state"]
        advanced_state = 277778083536782149546677086420637664879
        assert state["state"] == advanced_state


class TestMT19937(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = MT19937
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, './data/mt19937-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/mt19937-testset-2.csv'))
        cls.seed_error_type = ValueError
        cls.invalid_init_types = []
        cls.invalid_init_values = [(-1,)]

    def test_seed_float_array(self):
        assert_raises(TypeError, self.bit_generator, np.array([np.pi]))
        assert_raises(TypeError, self.bit_generator, np.array([-np.pi]))
        assert_raises(TypeError, self.bit_generator, np.array([np.pi, -np.pi]))
        assert_raises(TypeError, self.bit_generator, np.array([0, np.pi]))
        assert_raises(TypeError, self.bit_generator, [np.pi])
        assert_raises(TypeError, self.bit_generator, [0, np.pi])

    def test_state_tuple(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        bit_generator = rs.bit_generator
        state = bit_generator.state
        desired = rs.integers(2 ** 16)
        tup = (state['bit_generator'], state['state']['key'],
               state['state']['pos'])
        bit_generator.state = tup
        actual = rs.integers(2 ** 16)
        assert_equal(actual, desired)
        tup = tup + (0, 0.0)
        bit_generator.state = tup
        actual = rs.integers(2 ** 16)
        assert_equal(actual, desired)


class TestSFC64(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = SFC64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, './data/sfc64-testset-1.csv'))
        cls.data2 = cls._read_csv(
            join(pwd, './data/sfc64-testset-2.csv'))
        cls.seed_error_type = (ValueError, TypeError)
        cls.invalid_init_types = [(3.2,), ([None],), (1, None)]
        cls.invalid_init_values = [(-1,)]

    def test_legacy_pickle(self):
        # Pickling format was changed in 2.0.x
        import gzip
        import pickle

        expected_state = np.array(
            [
                9957867060933711493,
                532597980065565856,
                14769588338631205282,
                13
            ],
            dtype=np.uint64
        )

        base_path = os.path.split(os.path.abspath(__file__))[0]
        pkl_file = os.path.join(base_path, "data", "sfc64_np126.pkl.gz")
        with gzip.open(pkl_file) as gz:
            sfc = pickle.load(gz)

        assert isinstance(sfc, SFC64)
        assert_equal(sfc.state["state"]["state"], expected_state)


class TestDefaultRNG:
    def test_seed(self):
        for args in [(), (None,), (1234,), ([1234, 5678],)]:
            rg = default_rng(*args)
            assert isinstance(rg.bit_generator, PCG64)

    def test_passthrough(self):
        bg = Philox()
        rg = default_rng(bg)
        assert rg.bit_generator is bg
        rg2 = default_rng(rg)
        assert rg2 is rg
        assert rg2.bit_generator is bg

    def test_coercion_RandomState_Generator(self):
        # use default_rng to coerce RandomState to Generator
        rs = RandomState(1234)
        rg = default_rng(rs)
        assert isinstance(rg.bit_generator, MT19937)
        assert rg.bit_generator is rs._bit_generator

        # RandomState with a non MT19937 bit generator
        _original = np.random.get_bit_generator()
        bg = PCG64(12342298)
        np.random.set_bit_generator(bg)
        rs = np.random.mtrand._rand
        rg = default_rng(rs)
        assert rg.bit_generator is bg

        # vital to get global state back to original, otherwise
        # other tests start to fail.
        np.random.set_bit_generator(_original)