
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_indexing.py ===
import functools
import operator
import sys
import warnings
from itertools import product

import pytest
from numpy._core._multiarray_tests import array_indexing

import numpy as np
from numpy.exceptions import ComplexWarning, VisibleDeprecationWarning
from numpy.testing import (
    HAS_REFCOUNT,
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
)


class TestIndexing:
    def test_index_no_floats(self):
        a = np.array([[[5]]])

        assert_raises(IndexError, lambda: a[0.0])
        assert_raises(IndexError, lambda: a[0, 0.0])
        assert_raises(IndexError, lambda: a[0.0, 0])
        assert_raises(IndexError, lambda: a[0.0, :])
        assert_raises(IndexError, lambda: a[:, 0.0])
        assert_raises(IndexError, lambda: a[:, 0.0, :])
        assert_raises(IndexError, lambda: a[0.0, :, :])
        assert_raises(IndexError, lambda: a[0, 0, 0.0])
        assert_raises(IndexError, lambda: a[0.0, 0, 0])
        assert_raises(IndexError, lambda: a[0, 0.0, 0])
        assert_raises(IndexError, lambda: a[-1.4])
        assert_raises(IndexError, lambda: a[0, -1.4])
        assert_raises(IndexError, lambda: a[-1.4, 0])
        assert_raises(IndexError, lambda: a[-1.4, :])
        assert_raises(IndexError, lambda: a[:, -1.4])
        assert_raises(IndexError, lambda: a[:, -1.4, :])
        assert_raises(IndexError, lambda: a[-1.4, :, :])
        assert_raises(IndexError, lambda: a[0, 0, -1.4])
        assert_raises(IndexError, lambda: a[-1.4, 0, 0])
        assert_raises(IndexError, lambda: a[0, -1.4, 0])
        assert_raises(IndexError, lambda: a[0.0:, 0.0])
        assert_raises(IndexError, lambda: a[0.0:, 0.0, :])

    def test_slicing_no_floats(self):
        a = np.array([[5]])

        # start as float.
        assert_raises(TypeError, lambda: a[0.0:])
        assert_raises(TypeError, lambda: a[0:, 0.0:2])
        assert_raises(TypeError, lambda: a[0.0::2, :0])
        assert_raises(TypeError, lambda: a[0.0:1:2, :])
        assert_raises(TypeError, lambda: a[:, 0.0:])
        # stop as float.
        assert_raises(TypeError, lambda: a[:0.0])
        assert_raises(TypeError, lambda: a[:0, 1:2.0])
        assert_raises(TypeError, lambda: a[:0.0:2, :0])
        assert_raises(TypeError, lambda: a[:0.0, :])
        assert_raises(TypeError, lambda: a[:, 0:4.0:2])
        # step as float.
        assert_raises(TypeError, lambda: a[::1.0])
        assert_raises(TypeError, lambda: a[0:, :2:2.0])
        assert_raises(TypeError, lambda: a[1::4.0, :0])
        assert_raises(TypeError, lambda: a[::5.0, :])
        assert_raises(TypeError, lambda: a[:, 0:4:2.0])
        # mixed.
        assert_raises(TypeError, lambda: a[1.0:2:2.0])
        assert_raises(TypeError, lambda: a[1.0::2.0])
        assert_raises(TypeError, lambda: a[0:, :2.0:2.0])
        assert_raises(TypeError, lambda: a[1.0:1:4.0, :0])
        assert_raises(TypeError, lambda: a[1.0:5.0:5.0, :])
        assert_raises(TypeError, lambda: a[:, 0.4:4.0:2.0])
        # should still get the DeprecationWarning if step = 0.
        assert_raises(TypeError, lambda: a[::0.0])

    def test_index_no_array_to_index(self):
        # No non-scalar arrays.
        a = np.array([[[1]]])

        assert_raises(TypeError, lambda: a[a:a:a])

    def test_none_index(self):
        # `None` index adds newaxis
        a = np.array([1, 2, 3])
        assert_equal(a[None], a[np.newaxis])
        assert_equal(a[None].ndim, a.ndim + 1)

    def test_empty_tuple_index(self):
        # Empty tuple index creates a view
        a = np.array([1, 2, 3])
        assert_equal(a[()], a)
        assert_(a[()].base is a)
        a = np.array(0)
        assert_(isinstance(a[()], np.int_))

    def test_void_scalar_empty_tuple(self):
        s = np.zeros((), dtype='V4')
        assert_equal(s[()].dtype, s.dtype)
        assert_equal(s[()], s)
        assert_equal(type(s[...]), np.ndarray)

    def test_same_kind_index_casting(self):
        # Indexes should be cast with same-kind and not safe, even if that
        # is somewhat unsafe. So test various different code paths.
        index = np.arange(5)
        u_index = index.astype(np.uintp)
        arr = np.arange(10)

        assert_array_equal(arr[index], arr[u_index])
        arr[u_index] = np.arange(5)
        assert_array_equal(arr, np.arange(10))

        arr = np.arange(10).reshape(5, 2)
        assert_array_equal(arr[index], arr[u_index])

        arr[u_index] = np.arange(5)[:, None]
        assert_array_equal(arr, np.arange(5)[:, None].repeat(2, axis=1))

        arr = np.arange(25).reshape(5, 5)
        assert_array_equal(arr[u_index, u_index], arr[index, index])

    def test_empty_fancy_index(self):
        # Empty list index creates an empty array
        # with the same dtype (but with weird shape)
        a = np.array([1, 2, 3])
        assert_equal(a[[]], [])
        assert_equal(a[[]].dtype, a.dtype)

        b = np.array([], dtype=np.intp)
        assert_equal(a[[]], [])
        assert_equal(a[[]].dtype, a.dtype)

        b = np.array([])
        assert_raises(IndexError, a.__getitem__, b)

    def test_gh_26542(self):
        a = np.array([0, 1, 2])
        idx = np.array([2, 1, 0])
        a[idx] = a
        expected = np.array([2, 1, 0])
        assert_equal(a, expected)

    def test_gh_26542_2d(self):
        a = np.array([[0, 1, 2]])
        idx_row = np.zeros(3, dtype=int)
        idx_col = np.array([2, 1, 0])
        a[idx_row, idx_col] = a
        expected = np.array([[2, 1, 0]])
        assert_equal(a, expected)

    def test_gh_26542_index_overlap(self):
        arr = np.arange(100)
        expected_vals = np.copy(arr[:-10])
        arr[10:] = arr[:-10]
        actual_vals = arr[10:]
        assert_equal(actual_vals, expected_vals)

    def test_gh_26844(self):
        expected = [0, 1, 3, 3, 3]
        a = np.arange(5)
        a[2:][a[:-2]] = 3
        assert_equal(a, expected)

    def test_gh_26844_segfault(self):
        # check for absence of segfault for:
        # https://github.com/numpy/numpy/pull/26958/files#r1854589178
        a = np.arange(5)
        expected = [0, 1, 3, 3, 3]
        a[2:][None, a[:-2]] = 3
        assert_equal(a, expected)

    def test_ellipsis_index(self):
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        assert_(a[...] is not a)
        assert_equal(a[...], a)
        # `a[...]` was `a` in numpy <1.9.
        assert_(a[...].base is a)

        # Slicing with ellipsis can skip an
        # arbitrary number of dimensions
        assert_equal(a[0, ...], a[0])
        assert_equal(a[0, ...], a[0, :])
        assert_equal(a[..., 0], a[:, 0])

        # Slicing with ellipsis always results
        # in an array, not a scalar
        assert_equal(a[0, ..., 1], np.array(2))

        # Assignment with `(Ellipsis,)` on 0-d arrays
        b = np.array(1)
        b[(Ellipsis,)] = 2
        assert_equal(b, 2)

    def test_single_int_index(self):
        # Single integer index selects one row
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        assert_equal(a[0], [1, 2, 3])
        assert_equal(a[-1], [7, 8, 9])

        # Index out of bounds produces IndexError
        assert_raises(IndexError, a.__getitem__, 1 << 30)
        # Index overflow produces IndexError
        assert_raises(IndexError, a.__getitem__, 1 << 64)

    def test_single_bool_index(self):
        # Single boolean index
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        assert_equal(a[np.array(True)], a[None])
        assert_equal(a[np.array(False)], a[None][0:0])

    def test_boolean_shape_mismatch(self):
        arr = np.ones((5, 4, 3))

        index = np.array([True])
        assert_raises(IndexError, arr.__getitem__, index)

        index = np.array([False] * 6)
        assert_raises(IndexError, arr.__getitem__, index)

        index = np.zeros((4, 4), dtype=bool)
        assert_raises(IndexError, arr.__getitem__, index)

        assert_raises(IndexError, arr.__getitem__, (slice(None), index))

    def test_boolean_indexing_onedim(self):
        # Indexing a 2-dimensional array with
        # boolean array of length one
        a = np.array([[0., 0., 0.]])
        b = np.array([True], dtype=bool)
        assert_equal(a[b], a)
        # boolean assignment
        a[b] = 1.
        assert_equal(a, [[1., 1., 1.]])

    def test_boolean_assignment_value_mismatch(self):
        # A boolean assignment should fail when the shape of the values
        # cannot be broadcast to the subscription. (see also gh-3458)
        a = np.arange(4)

        def f(a, v):
            a[a > -1] = v

        assert_raises(ValueError, f, a, [])
        assert_raises(ValueError, f, a, [1, 2, 3])
        assert_raises(ValueError, f, a[:1], [1, 2, 3])

    def test_boolean_assignment_needs_api(self):
        # See also gh-7666
        # This caused a segfault on Python 2 due to the GIL not being
        # held when the iterator does not need it, but the transfer function
        # does
        arr = np.zeros(1000)
        indx = np.zeros(1000, dtype=bool)
        indx[:100] = True
        arr[indx] = np.ones(100, dtype=object)

        expected = np.zeros(1000)
        expected[:100] = 1
        assert_array_equal(arr, expected)

    def test_boolean_indexing_twodim(self):
        # Indexing a 2-dimensional array with
        # 2-dimensional boolean array
        a = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        b = np.array([[ True, False, True],
                      [False, True, False],
                      [ True, False, True]])
        assert_equal(a[b], [1, 3, 5, 7, 9])
        assert_equal(a[b[1]], [[4, 5, 6]])
        assert_equal(a[b[0]], a[b[2]])

        # boolean assignment
        a[b] = 0
        assert_equal(a, [[0, 2, 0],
                         [4, 0, 6],
                         [0, 8, 0]])

    def test_boolean_indexing_list(self):
        # Regression test for #13715. It's a use-after-free bug which the
        # test won't directly catch, but it will show up in valgrind.
        a = np.array([1, 2, 3])
        b = [True, False, True]
        # Two variants of the test because the first takes a fast path
        assert_equal(a[b], [1, 3])
        assert_equal(a[None, b], [[1, 3]])

    def test_reverse_strides_and_subspace_bufferinit(self):
        # This tests that the strides are not reversed for simple and
        # subspace fancy indexing.
        a = np.ones(5)
        b = np.zeros(5, dtype=np.intp)[::-1]
        c = np.arange(5)[::-1]

        a[b] = c
        # If the strides are not reversed, the 0 in the arange comes last.
        assert_equal(a[0], 0)

        # This also tests that the subspace buffer is initialized:
        a = np.ones((5, 2))
        c = np.arange(10).reshape(5, 2)[::-1]
        a[b, :] = c
        assert_equal(a[0], [0, 1])

    def test_reversed_strides_result_allocation(self):
        # Test a bug when calculating the output strides for a result array
        # when the subspace size was 1 (and test other cases as well)
        a = np.arange(10)[:, None]
        i = np.arange(10)[::-1]
        assert_array_equal(a[i], a[i.copy('C')])

        a = np.arange(20).reshape(-1, 2)

    def test_uncontiguous_subspace_assignment(self):
        # During development there was a bug activating a skip logic
        # based on ndim instead of size.
        a = np.full((3, 4, 2), -1)
        b = np.full((3, 4, 2), -1)

        a[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T
        b[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T.copy()

        assert_equal(a, b)

    def test_too_many_fancy_indices_special_case(self):
        # Just documents behaviour, this is a small limitation.
        a = np.ones((1,) * 64)  # 64 is NPY_MAXDIMS
        assert_raises(IndexError, a.__getitem__, (np.array([0]),) * 64)

    def test_scalar_array_bool(self):
        # NumPy bools can be used as boolean index (python ones as of yet not)
        a = np.array(1)
        assert_equal(a[np.bool(True)], a[np.array(True)])
        assert_equal(a[np.bool(False)], a[np.array(False)])

        # After deprecating bools as integers:
        #a = np.array([0,1,2])
        #assert_equal(a[True, :], a[None, :])
        #assert_equal(a[:, True], a[:, None])
        #
        #assert_(not np.may_share_memory(a, a[True, :]))

    def test_everything_returns_views(self):
        # Before `...` would return a itself.
        a = np.arange(5)

        assert_(a is not a[()])
        assert_(a is not a[...])
        assert_(a is not a[:])

    def test_broaderrors_indexing(self):
        a = np.zeros((5, 5))
        assert_raises(IndexError, a.__getitem__, ([0, 1], [0, 1, 2]))
        assert_raises(IndexError, a.__setitem__, ([0, 1], [0, 1, 2]), 0)

    def test_trivial_fancy_out_of_bounds(self):
        a = np.zeros(5)
        ind = np.ones(20, dtype=np.intp)
        ind[-1] = 10
        assert_raises(IndexError, a.__getitem__, ind)
        assert_raises(IndexError, a.__setitem__, ind, 0)
        ind = np.ones(20, dtype=np.intp)
        ind[0] = 11
        assert_raises(IndexError, a.__getitem__, ind)
        assert_raises(IndexError, a.__setitem__, ind, 0)

    def test_trivial_fancy_not_possible(self):
        # Test that the fast path for trivial assignment is not incorrectly
        # used when the index is not contiguous or 1D, see also gh-11467.
        a = np.arange(6)
        idx = np.arange(6, dtype=np.intp).reshape(2, 1, 3)[:, :, 0]
        assert_array_equal(a[idx], idx)

        # this case must not go into the fast path, note that idx is
        # a non-contiguous none 1D array here.
        a[idx] = -1
        res = np.arange(6)
        res[0] = -1
        res[3] = -1
        assert_array_equal(a, res)

    def test_nonbaseclass_values(self):
        class SubClass(np.ndarray):
            def __array_finalize__(self, old):
                # Have array finalize do funny things
                self.fill(99)

        a = np.zeros((5, 5))
        s = a.copy().view(type=SubClass)
        s.fill(1)

        a[[0, 1, 2, 3, 4], :] = s
        assert_((a == 1).all())

        # Subspace is last, so transposing might want to finalize
        a[:, [0, 1, 2, 3, 4]] = s
        assert_((a == 1).all())

        a.fill(0)
        a[...] = s
        assert_((a == 1).all())

    def test_array_like_values(self):
        # Similar to the above test, but use a memoryview instead
        a = np.zeros((5, 5))
        s = np.arange(25, dtype=np.float64).reshape(5, 5)

        a[[0, 1, 2, 3, 4], :] = memoryview(s)
        assert_array_equal(a, s)

        a[:, [0, 1, 2, 3, 4]] = memoryview(s)
        assert_array_equal(a, s)

        a[...] = memoryview(s)
        assert_array_equal(a, s)

    @pytest.mark.parametrize("writeable", [True, False])
    def test_subclass_writeable(self, writeable):
        d = np.rec.array([('NGC1001', 11), ('NGC1002', 1.), ('NGC1003', 1.)],
                         dtype=[('target', 'S20'), ('V_mag', '>f4')])
        d.flags.writeable = writeable
        # Advanced indexing results are always writeable:
        ind = np.array([False, True, True], dtype=bool)
        assert d[ind].flags.writeable
        ind = np.array([0, 1])
        assert d[ind].flags.writeable
        # Views should be writeable if the original array is:
        assert d[...].flags.writeable == writeable
        assert d[0].flags.writeable == writeable

    def test_memory_order(self):
        # This is not necessary to preserve. Memory layouts for
        # more complex indices are not as simple.
        a = np.arange(10)
        b = np.arange(10).reshape(5, 2).T
        assert_(a[b].flags.f_contiguous)

        # Takes a different implementation branch:
        a = a.reshape(-1, 1)
        assert_(a[b, 0].flags.f_contiguous)

    def test_scalar_return_type(self):
        # Full scalar indices should return scalars and object
        # arrays should not call PyArray_Return on their items
        class Zero:
            # The most basic valid indexing
            def __index__(self):
                return 0

        z = Zero()

        class ArrayLike:
            # Simple array, should behave like the array
            def __array__(self, dtype=None, copy=None):
                return np.array(0)

        a = np.zeros(())
        assert_(isinstance(a[()], np.float64))
        a = np.zeros(1)
        assert_(isinstance(a[z], np.float64))
        a = np.zeros((1, 1))
        assert_(isinstance(a[z, np.array(0)], np.float64))
        assert_(isinstance(a[z, ArrayLike()], np.float64))

        # And object arrays do not call it too often:
        b = np.array(0)
        a = np.array(0, dtype=object)
        a[()] = b
        assert_(isinstance(a[()], np.ndarray))
        a = np.array([b, None])
        assert_(isinstance(a[z], np.ndarray))
        a = np.array([[b, None]])
        assert_(isinstance(a[z, np.array(0)], np.ndarray))
        assert_(isinstance(a[z, ArrayLike()], np.ndarray))

    def test_small_regressions(self):
        # Reference count of intp for index checks
        a = np.array([0])
        if HAS_REFCOUNT:
            refcount = sys.getrefcount(np.dtype(np.intp))
        # item setting always checks indices in separate function:
        a[np.array([0], dtype=np.intp)] = 1
        a[np.array([0], dtype=np.uint8)] = 1
        assert_raises(IndexError, a.__setitem__,
                      np.array([1], dtype=np.intp), 1)
        assert_raises(IndexError, a.__setitem__,
                      np.array([1], dtype=np.uint8), 1)

        if HAS_REFCOUNT:
            assert_equal(sys.getrefcount(np.dtype(np.intp)), refcount)

    def test_unaligned(self):
        v = (np.zeros(64, dtype=np.int8) + ord('a'))[1:-7]
        d = v.view(np.dtype("S8"))
        # unaligned source
        x = (np.zeros(16, dtype=np.int8) + ord('a'))[1:-7]
        x = x.view(np.dtype("S8"))
        x[...] = np.array("b" * 8, dtype="S")
        b = np.arange(d.size)
        # trivial
        assert_equal(d[b], d)
        d[b] = x
        # nontrivial
        # unaligned index array
        b = np.zeros(d.size + 1).view(np.int8)[1:-(np.intp(0).itemsize - 1)]
        b = b.view(np.intp)[:d.size]
        b[...] = np.arange(d.size)
        assert_equal(d[b.astype(np.int16)], d)
        d[b.astype(np.int16)] = x
        # boolean
        d[b % 2 == 0]
        d[b % 2 == 0] = x[::2]

    def test_tuple_subclass(self):
        arr = np.ones((5, 5))

        # A tuple subclass should also be an nd-index
        class TupleSubclass(tuple):
            pass
        index = ([1], [1])
        index = TupleSubclass(index)
        assert_(arr[index].shape == (1,))
        # Unlike the non nd-index:
        assert_(arr[index,].shape != (1,))

    def test_broken_sequence_not_nd_index(self):
        # See gh-5063:
        # If we have an object which claims to be a sequence, but fails
        # on item getting, this should not be converted to an nd-index (tuple)
        # If this object happens to be a valid index otherwise, it should work
        # This object here is very dubious and probably bad though:
        class SequenceLike:
            def __index__(self):
                return 0

            def __len__(self):
                return 1

            def __getitem__(self, item):
                raise IndexError('Not possible')

        arr = np.arange(10)
        assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])

        # also test that field indexing does not segfault
        # for a similar reason, by indexing a structured array
        arr = np.zeros((1,), dtype=[('f1', 'i8'), ('f2', 'i8')])
        assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])

    def test_indexing_array_weird_strides(self):
        # See also gh-6221
        # the shapes used here come from the issue and create the correct
        # size for the iterator buffering size.
        x = np.ones(10)
        x2 = np.ones((10, 2))
        ind = np.arange(10)[:, None, None, None]
        ind = np.broadcast_to(ind, (10, 55, 4, 4))

        # single advanced index case
        assert_array_equal(x[ind], x[ind.copy()])
        # higher dimensional advanced index
        zind = np.zeros(4, dtype=np.intp)
        assert_array_equal(x2[ind, zind], x2[ind.copy(), zind])

    def test_indexing_array_negative_strides(self):
        # From gh-8264,
        # core dumps if negative strides are used in iteration
        arro = np.zeros((4, 4))
        arr = arro[::-1, ::-1]

        slices = (slice(None), [0, 1, 2, 3])
        arr[slices] = 10
        assert_array_equal(arr, 10.)

    def test_character_assignment(self):
        # This is an example a function going through CopyObject which
        # used to have an untested special path for scalars
        # (the character special dtype case, should be deprecated probably)
        arr = np.zeros((1, 5), dtype="c")
        arr[0] = np.str_("asdfg")  # must assign as a sequence
        assert_array_equal(arr[0], np.array("asdfg", dtype="c"))
        assert arr[0, 1] == b"s"  # make sure not all were set to "a" for both

    @pytest.mark.parametrize("index",
            [True, False, np.array([0])])
    @pytest.mark.parametrize("num", [64, 80])
    @pytest.mark.parametrize("original_ndim", [1, 64])
    def test_too_many_advanced_indices(self, index, num, original_ndim):
        # These are limitations based on the number of arguments we can process.
        # For `num=32` (and all boolean cases), the result is actually define;
        # but the use of NpyIter (NPY_MAXARGS) limits it for technical reasons.
        arr = np.ones((1,) * original_ndim)
        with pytest.raises(IndexError):
            arr[(index,) * num]
        with pytest.raises(IndexError):
            arr[(index,) * num] = 1.

    def test_nontuple_ndindex(self):
        a = np.arange(25).reshape((5, 5))
        assert_equal(a[[0, 1]], np.array([a[0], a[1]]))
        assert_equal(a[[0, 1], [0, 1]], np.array([0, 6]))
        assert_raises(IndexError, a.__getitem__, [slice(None)])

    def test_flat_index_on_flatiter(self):
        a = np.arange(9).reshape((3, 3))
        b = np.array([0, 5, 6])
        assert_equal(a.flat[b.flat], np.array([0, 5, 6]))

    def test_empty_string_flat_index_on_flatiter(self):
        a = np.arange(9).reshape((3, 3))
        b = np.array([], dtype="S")
        assert_equal(a.flat[b.flat], np.array([]))

    def test_nonempty_string_flat_index_on_flatiter(self):
        a = np.arange(9).reshape((3, 3))
        b = np.array(["a"], dtype="S")
        with pytest.raises(IndexError, match="unsupported iterator index"):
            a.flat[b.flat]


class TestFieldIndexing:
    def test_scalar_return_type(self):
        # Field access on an array should return an array, even if it
        # is 0-d.
        a = np.zeros((), [('a', 'f8')])
        assert_(isinstance(a['a'], np.ndarray))
        assert_(isinstance(a[['a']], np.ndarray))


class TestBroadcastedAssignments:
    def assign(self, a, ind, val):
        a[ind] = val
        return a

    def test_prepending_ones(self):
        a = np.zeros((3, 2))

        a[...] = np.ones((1, 3, 2))
        # Fancy with subspace with and without transpose
        a[[0, 1, 2], :] = np.ones((1, 3, 2))
        a[:, [0, 1]] = np.ones((1, 3, 2))
        # Fancy without subspace (with broadcasting)
        a[[[0], [1], [2]], [0, 1]] = np.ones((1, 3, 2))

    def test_prepend_not_one(self):
        assign = self.assign
        s_ = np.s_
        a = np.zeros(5)

        # Too large and not only ones.
        assert_raises(ValueError, assign, a, s_[...], np.ones((2, 1)))
        assert_raises(ValueError, assign, a, s_[[1, 2, 3],], np.ones((2, 1)))
        assert_raises(ValueError, assign, a, s_[[[1], [2]],], np.ones((2, 2, 1)))

    def test_simple_broadcasting_errors(self):
        assign = self.assign
        s_ = np.s_
        a = np.zeros((5, 1))

        assert_raises(ValueError, assign, a, s_[...], np.zeros((5, 2)))
        assert_raises(ValueError, assign, a, s_[...], np.zeros((5, 0)))
        assert_raises(ValueError, assign, a, s_[:, [0]], np.zeros((5, 2)))
        assert_raises(ValueError, assign, a, s_[:, [0]], np.zeros((5, 0)))
        assert_raises(ValueError, assign, a, s_[[0], :], np.zeros((2, 1)))

    @pytest.mark.parametrize("index", [
            (..., [1, 2], slice(None)),
            ([0, 1], ..., 0),
            (..., [1, 2], [1, 2])])
    def test_broadcast_error_reports_correct_shape(self, index):
        values = np.zeros((100, 100))  # will never broadcast below

        arr = np.zeros((3, 4, 5, 6, 7))
        # We currently report without any spaces (could be changed)
        shape_str = str(arr[index].shape).replace(" ", "")

        with pytest.raises(ValueError) as e:
            arr[index] = values

        assert str(e.value).endswith(shape_str)

    def test_index_is_larger(self):
        # Simple case of fancy index broadcasting of the index.
        a = np.zeros((5, 5))
        a[[[0], [1], [2]], [0, 1, 2]] = [2, 3, 4]

        assert_((a[:3, :3] == [2, 3, 4]).all())

    def test_broadcast_subspace(self):
        a = np.zeros((100, 100))
        v = np.arange(100)[:, None]
        b = np.arange(100)[::-1]
        a[b] = v
        assert_((a[::-1] == v).all())


class TestSubclasses:
    def test_basic(self):
        # Test that indexing in various ways produces SubClass instances,
        # and that the base is set up correctly: the original subclass
        # instance for views, and a new ndarray for advanced/boolean indexing
        # where a copy was made (latter a regression test for gh-11983).
        class SubClass(np.ndarray):
            pass

        a = np.arange(5)
        s = a.view(SubClass)
        s_slice = s[:3]
        assert_(type(s_slice) is SubClass)
        assert_(s_slice.base is s)
        assert_array_equal(s_slice, a[:3])

        s_fancy = s[[0, 1, 2]]
        assert_(type(s_fancy) is SubClass)
        assert_(s_fancy.base is not s)
        assert_(type(s_fancy.base) is np.ndarray)
        assert_array_equal(s_fancy, a[[0, 1, 2]])
        assert_array_equal(s_fancy.base, a[[0, 1, 2]])

        s_bool = s[s > 0]
        assert_(type(s_bool) is SubClass)
        assert_(s_bool.base is not s)
        assert_(type(s_bool.base) is np.ndarray)
        assert_array_equal(s_bool, a[a > 0])
        assert_array_equal(s_bool.base, a[a > 0])

    def test_fancy_on_read_only(self):
        # Test that fancy indexing on read-only SubClass does not make a
        # read-only copy (gh-14132)
        class SubClass(np.ndarray):
            pass

        a = np.arange(5)
        s = a.view(SubClass)
        s.flags.writeable = False
        s_fancy = s[[0, 1, 2]]
        assert_(s_fancy.flags.writeable)

    def test_finalize_gets_full_info(self):
        # Array finalize should be called on the filled array.
        class SubClass(np.ndarray):
            def __array_finalize__(self, old):
                self.finalize_status = np.array(self)
                self.old = old

        s = np.arange(10).view(SubClass)
        new_s = s[:3]
        assert_array_equal(new_s.finalize_status, new_s)
        assert_array_equal(new_s.old, s)

        new_s = s[[0, 1, 2, 3]]
        assert_array_equal(new_s.finalize_status, new_s)
        assert_array_equal(new_s.old, s)

        new_s = s[s > 0]
        assert_array_equal(new_s.finalize_status, new_s)
        assert_array_equal(new_s.old, s)


class TestFancyIndexingCast:
    def test_boolean_index_cast_assign(self):
        # Setup the boolean index and float arrays.
        shape = (8, 63)
        bool_index = np.zeros(shape).astype(bool)
        bool_index[0, 1] = True
        zero_array = np.zeros(shape)

        # Assigning float is fine.
        zero_array[bool_index] = np.array([1])
        assert_equal(zero_array[0, 1], 1)

        # Fancy indexing works, although we get a cast warning.
        assert_warns(ComplexWarning,
                     zero_array.__setitem__, ([0], [1]), np.array([2 + 1j]))
        assert_equal(zero_array[0, 1], 2)  # No complex part

        # Cast complex to float, throwing away the imaginary portion.
        assert_warns(ComplexWarning,
                     zero_array.__setitem__, bool_index, np.array([1j]))
        assert_equal(zero_array[0, 1], 0)

class TestFancyIndexingEquivalence:
    def test_object_assign(self):
        # Check that the field and object special case using copyto is active.
        # The right hand side cannot be converted to an array here.
        a = np.arange(5, dtype=object)
        b = a.copy()
        a[:3] = [1, (1, 2), 3]
        b[[0, 1, 2]] = [1, (1, 2), 3]
        assert_array_equal(a, b)

        # test same for subspace fancy indexing
        b = np.arange(5, dtype=object)[None, :]
        b[[0], :3] = [[1, (1, 2), 3]]
        assert_array_equal(a, b[0])

        # Check that swapping of axes works.
        # There was a bug that made the later assignment throw a ValueError
        # do to an incorrectly transposed temporary right hand side (gh-5714)
        b = b.T
        b[:3, [0]] = [[1], [(1, 2)], [3]]
        assert_array_equal(a, b[:, 0])

        # Another test for the memory order of the subspace
        arr = np.ones((3, 4, 5), dtype=object)
        # Equivalent slicing assignment for comparison
        cmp_arr = arr.copy()
        cmp_arr[:1, ...] = [[[1], [2], [3], [4]]]
        arr[[0], ...] = [[[1], [2], [3], [4]]]
        assert_array_equal(arr, cmp_arr)
        arr = arr.copy('F')
        arr[[0], ...] = [[[1], [2], [3], [4]]]
        assert_array_equal(arr, cmp_arr)

    def test_cast_equivalence(self):
        # Yes, normal slicing uses unsafe casting.
        a = np.arange(5)
        b = a.copy()

        a[:3] = np.array(['2', '-3', '-1'])
        b[[0, 2, 1]] = np.array(['2', '-1', '-3'])
        assert_array_equal(a, b)

        # test the same for subspace fancy indexing
        b = np.arange(5)[None, :]
        b[[0], :3] = np.array([['2', '-3', '-1']])
        assert_array_equal(a, b[0])


class TestMultiIndexingAutomated:
    """
    These tests use code to mimic the C-Code indexing for selection.

    NOTE:

        * This still lacks tests for complex item setting.
        * If you change behavior of indexing, you might want to modify
          these tests to try more combinations.
        * Behavior was written to match numpy version 1.8. (though a
          first version matched 1.7.)
        * Only tuple indices are supported by the mimicking code.
          (and tested as of writing this)
        * Error types should match most of the time as long as there
          is only one error. For multiple errors, what gets raised
          will usually not be the same one. They are *not* tested.

    Update 2016-11-30: It is probably not worth maintaining this test
    indefinitely and it can be dropped if maintenance becomes a burden.

    """

    def setup_method(self):
        self.a = np.arange(np.prod([3, 1, 5, 6])).reshape(3, 1, 5, 6)
        self.b = np.empty((3, 0, 5, 6))
        self.complex_indices = ['skip', Ellipsis,
            0,
            # Boolean indices, up to 3-d for some special cases of eating up
            # dimensions, also need to test all False
            np.array([True, False, False]),
            np.array([[True, False], [False, True]]),
            np.array([[[False, False], [False, False]]]),
            # Some slices:
            slice(-5, 5, 2),
            slice(1, 1, 100),
            slice(4, -1, -2),
            slice(None, None, -3),
            # Some Fancy indexes:
            np.empty((0, 1, 1), dtype=np.intp),  # empty and can be broadcast
            np.array([0, 1, -2]),
            np.array([[2], [0], [1]]),
            np.array([[0, -1], [0, 1]], dtype=np.dtype('intp').newbyteorder()),
            np.array([2, -1], dtype=np.int8),
            np.zeros([1] * 31, dtype=int),  # trigger too large array.
            np.array([0., 1.])]  # invalid datatype
        # Some simpler indices that still cover a bit more
        self.simple_indices = [Ellipsis, None, -1, [1], np.array([True]),
                               'skip']
        # Very simple ones to fill the rest:
        self.fill_indices = [slice(None, None), 0]

    def _get_multi_index(self, arr, indices):
        """Mimic multi dimensional indexing.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed.
        indices : tuple of index objects

        Returns
        -------
        out : ndarray
            An array equivalent to the indexing operation (but always a copy).
            `arr[indices]` should be identical.
        no_copy : bool
            Whether the indexing operation requires a copy. If this is `True`,
            `np.may_share_memory(arr, arr[indices])` should be `True` (with
            some exceptions for scalars and possibly 0-d arrays).

        Notes
        -----
        While the function may mostly match the errors of normal indexing this
        is generally not the case.
        """
        in_indices = list(indices)
        indices = []
        # if False, this is a fancy or boolean index
        no_copy = True
        # number of fancy/scalar indexes that are not consecutive
        num_fancy = 0
        # number of dimensions indexed by a "fancy" index
        fancy_dim = 0
        # NOTE: This is a funny twist (and probably OK to change).
        # The boolean array has illegal indexes, but this is
        # allowed if the broadcast fancy-indices are 0-sized.
        # This variable is to catch that case.
        error_unless_broadcast_to_empty = False

        # We need to handle Ellipsis and make arrays from indices, also
        # check if this is fancy indexing (set no_copy).
        ndim = 0
        ellipsis_pos = None  # define here mostly to replace all but first.
        for i, indx in enumerate(in_indices):
            if indx is None:
                continue
            if isinstance(indx, np.ndarray) and indx.dtype == bool:
                no_copy = False
                if indx.ndim == 0:
                    raise IndexError
                # boolean indices can have higher dimensions
                ndim += indx.ndim
                fancy_dim += indx.ndim
                continue
            if indx is Ellipsis:
                if ellipsis_pos is None:
                    ellipsis_pos = i
                    continue  # do not increment ndim counter
                raise IndexError
            if isinstance(indx, slice):
                ndim += 1
                continue
            if not isinstance(indx, np.ndarray):
                # This could be open for changes in numpy.
                # numpy should maybe raise an error if casting to intp
                # is not safe. It rejects np.array([1., 2.]) but not
                # [1., 2.] as index (same for ie. np.take).
                # (Note the importance of empty lists if changing this here)
                try:
                    indx = np.array(indx, dtype=np.intp)
                except ValueError:
                    raise IndexError
                in_indices[i] = indx
            elif indx.dtype.kind not in 'bi':
                raise IndexError('arrays used as indices must be of '
                                 'integer (or boolean) type')
            if indx.ndim != 0:
                no_copy = False
            ndim += 1
            fancy_dim += 1

        if arr.ndim - ndim < 0:
            # we can't take more dimensions then we have, not even for 0-d
            # arrays.  since a[()] makes sense, but not a[(),]. We will
            # raise an error later on, unless a broadcasting error occurs
            # first.
            raise IndexError

        if ndim == 0 and None not in in_indices:
            # Well we have no indexes or one Ellipsis. This is legal.
            return arr.copy(), no_copy

        if ellipsis_pos is not None:
            in_indices[ellipsis_pos:ellipsis_pos + 1] = ([slice(None, None)] *
                                                       (arr.ndim - ndim))

        for ax, indx in enumerate(in_indices):
            if isinstance(indx, slice):
                # convert to an index array
                indx = np.arange(*indx.indices(arr.shape[ax]))
                indices.append(['s', indx])
                continue
            elif indx is None:
                # this is like taking a slice with one element from a new axis:
                indices.append(['n', np.array([0], dtype=np.intp)])
                arr = arr.reshape(arr.shape[:ax] + (1,) + arr.shape[ax:])
                continue
            if isinstance(indx, np.ndarray) and indx.dtype == bool:
                if indx.shape != arr.shape[ax:ax + indx.ndim]:
                    raise IndexError

                try:
                    flat_indx = np.ravel_multi_index(np.nonzero(indx),
                                    arr.shape[ax:ax + indx.ndim], mode='raise')
                except Exception:
                    error_unless_broadcast_to_empty = True
                    # fill with 0s instead, and raise error later
                    flat_indx = np.array([0] * indx.sum(), dtype=np.intp)
                # concatenate axis into a single one:
                if indx.ndim != 0:
                    arr = arr.reshape(arr.shape[:ax]
                                  + (np.prod(arr.shape[ax:ax + indx.ndim]),)
                                  + arr.shape[ax + indx.ndim:])
                    indx = flat_indx
                else:
                    # This could be changed, a 0-d boolean index can
                    # make sense (even outside the 0-d indexed array case)
                    # Note that originally this is could be interpreted as
                    # integer in the full integer special case.
                    raise IndexError
            # If the index is a singleton, the bounds check is done
            # before the broadcasting. This used to be different in <1.9
            elif indx.ndim == 0 and not (
                -arr.shape[ax] <= indx < arr.shape[ax]
            ):
                raise IndexError
            if indx.ndim == 0:
                # The index is a scalar. This used to be two fold, but if
                # fancy indexing was active, the check was done later,
                # possibly after broadcasting it away (1.7. or earlier).
                # Now it is always done.
                if indx >= arr.shape[ax] or indx < - arr.shape[ax]:
                    raise IndexError
            if (len(indices) > 0 and
                    indices[-1][0] == 'f' and
                    ax != ellipsis_pos):
                # NOTE: There could still have been a 0-sized Ellipsis
                # between them. Checked that with ellipsis_pos.
                indices[-1].append(indx)
            else:
                # We have a fancy index that is not after an existing one.
                # NOTE: A 0-d array triggers this as well, while one may
                # expect it to not trigger it, since a scalar would not be
                # considered fancy indexing.
                num_fancy += 1
                indices.append(['f', indx])

        if num_fancy > 1 and not no_copy:
            # We have to flush the fancy indexes left
            new_indices = indices[:]
            axes = list(range(arr.ndim))
            fancy_axes = []
            new_indices.insert(0, ['f'])
            ni = 0
            ai = 0
            for indx in indices:
                ni += 1
                if indx[0] == 'f':
                    new_indices[0].extend(indx[1:])
                    del new_indices[ni]
                    ni -= 1
                    for ax in range(ai, ai + len(indx[1:])):
                        fancy_axes.append(ax)
                        axes.remove(ax)
                ai += len(indx) - 1  # axis we are at
            indices = new_indices
            # and now we need to transpose arr:
            arr = arr.transpose(*(fancy_axes + axes))

        # We only have one 'f' index now and arr is transposed accordingly.
        # Now handle newaxis by reshaping...
        ax = 0
        for indx in indices:
            if indx[0] == 'f':
                if len(indx) == 1:
                    continue
                # First of all, reshape arr to combine fancy axes into one:
                orig_shape = arr.shape
                orig_slice = orig_shape[ax:ax + len(indx[1:])]
                arr = arr.reshape(arr.shape[:ax]
                                    + (np.prod(orig_slice).astype(int),)
                                    + arr.shape[ax + len(indx[1:]):])

                # Check if broadcasting works
                res = np.broadcast(*indx[1:])
                # unfortunately the indices might be out of bounds. So check
                # that first, and use mode='wrap' then. However only if
                # there are any indices...
                if res.size != 0:
                    if error_unless_broadcast_to_empty:
                        raise IndexError
                    for _indx, _size in zip(indx[1:], orig_slice):
                        if _indx.size == 0:
                            continue
                        if np.any(_indx >= _size) or np.any(_indx < -_size):
                            raise IndexError
                if len(indx[1:]) == len(orig_slice):
                    if np.prod(orig_slice) == 0:
                        # Work around for a crash or IndexError with 'wrap'
                        # in some 0-sized cases.
                        try:
                            mi = np.ravel_multi_index(indx[1:], orig_slice,
                                                      mode='raise')
                        except Exception:
                            # This happens with 0-sized orig_slice (sometimes?)
                            # here it is a ValueError, but indexing gives a:
                            raise IndexError('invalid index into 0-sized')
                    else:
                        mi = np.ravel_multi_index(indx[1:], orig_slice,
                                                  mode='wrap')
                else:
                    # Maybe never happens...
                    raise ValueError
                arr = arr.take(mi.ravel(), axis=ax)
                try:
                    arr = arr.reshape(arr.shape[:ax]
                                        + mi.shape
                                        + arr.shape[ax + 1:])
                except ValueError:
                    # too many dimensions, probably
                    raise IndexError
                ax += mi.ndim
                continue

            # If we are here, we have a 1D array for take:
            arr = arr.take(indx[1], axis=ax)
            ax += 1

        return arr, no_copy

    def _check_multi_index(self, arr, index):
        """Check a multi index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed, must be a reshaped arange.
        index : tuple of indexing objects
            Index being tested.
        """
        # Test item getting
        try:
            mimic_get, no_copy = self._get_multi_index(arr, index)
        except Exception as e:
            if HAS_REFCOUNT:
                prev_refcount = sys.getrefcount(arr)
            assert_raises(type(e), arr.__getitem__, index)
            assert_raises(type(e), arr.__setitem__, index, 0)
            if HAS_REFCOUNT:
                assert_equal(prev_refcount, sys.getrefcount(arr))
            return

        self._compare_index_result(arr, index, mimic_get, no_copy)

    def _check_single_index(self, arr, index):
        """Check a single index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed, must be an arange.
        index : indexing object
            Index being tested. Must be a single index and not a tuple
            of indexing objects (see also `_check_multi_index`).
        """
        try:
            mimic_get, no_copy = self._get_multi_index(arr, (index,))
        except Exception as e:
            if HAS_REFCOUNT:
                prev_refcount = sys.getrefcount(arr)
            assert_raises(type(e), arr.__getitem__, index)
            assert_raises(type(e), arr.__setitem__, index, 0)
            if HAS_REFCOUNT:
                assert_equal(prev_refcount, sys.getrefcount(arr))
            return

        self._compare_index_result(arr, index, mimic_get, no_copy)

    def _compare_index_result(self, arr, index, mimic_get, no_copy):
        """Compare mimicked result to indexing result.
        """
        arr = arr.copy()
        if HAS_REFCOUNT:
            startcount = sys.getrefcount(arr)
        indexed_arr = arr[index]
        assert_array_equal(indexed_arr, mimic_get)
        # Check if we got a view, unless its a 0-sized or 0-d array.
        # (then its not a view, and that does not matter)
        if indexed_arr.size != 0 and indexed_arr.ndim != 0:
            assert_(np.may_share_memory(indexed_arr, arr) == no_copy)
            # Check reference count of the original array
            if HAS_REFCOUNT:
                if no_copy:
                    # refcount increases by one:
                    assert_equal(sys.getrefcount(arr), startcount + 1)
                else:
                    assert_equal(sys.getrefcount(arr), startcount)

        # Test non-broadcast setitem:
        b = arr.copy()
        b[index] = mimic_get + 1000
        if b.size == 0:
            return  # nothing to compare here...
        if no_copy and indexed_arr.ndim != 0:
            # change indexed_arr in-place to manipulate original:
            indexed_arr += 1000
            assert_array_equal(arr, b)
            return
        # Use the fact that the array is originally an arange:
        arr.flat[indexed_arr.ravel()] += 1000
        assert_array_equal(arr, b)

    def test_boolean(self):
        a = np.array(5)
        assert_equal(a[np.array(True)], 5)
        a[np.array(True)] = 1
        assert_equal(a, 1)
        # NOTE: This is different from normal broadcasting, as
        # arr[boolean_array] works like in a multi index. Which means
        # it is aligned to the left. This is probably correct for
        # consistency with arr[boolean_array,] also no broadcasting
        # is done at all
        self._check_multi_index(
            self.a, (np.zeros_like(self.a, dtype=bool),))
        self._check_multi_index(
            self.a, (np.zeros_like(self.a, dtype=bool)[..., 0],))
        self._check_multi_index(
            self.a, (np.zeros_like(self.a, dtype=bool)[None, ...],))

    def test_multidim(self):
        # Automatically test combinations with complex indexes on 2nd (or 1st)
        # spot and the simple ones in one other spot.
        with warnings.catch_warnings():
            # This is so that np.array(True) is not accepted in a full integer
            # index, when running the file separately.
            warnings.filterwarnings('error', '', DeprecationWarning)
            warnings.filterwarnings('error', '', VisibleDeprecationWarning)

            def isskip(idx):
                return isinstance(idx, str) and idx == "skip"

            for simple_pos in [0, 2, 3]:
                tocheck = [self.fill_indices, self.complex_indices,
                           self.fill_indices, self.fill_indices]
                tocheck[simple_pos] = self.simple_indices
                for index in product(*tocheck):
                    index = tuple(i for i in index if not isskip(i))
                    self._check_multi_index(self.a, index)
                    self._check_multi_index(self.b, index)

        # Check very simple item getting:
        self._check_multi_index(self.a, (0, 0, 0, 0))
        self._check_multi_index(self.b, (0, 0, 0, 0))
        # Also check (simple cases of) too many indices:
        assert_raises(IndexError, self.a.__getitem__, (0, 0, 0, 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, 0, 0, 0), 0)
        assert_raises(IndexError, self.a.__getitem__, (0, 0, [1], 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, [1], 0, 0), 0)

    def test_1d(self):
        a = np.arange(10)
        for index in self.complex_indices:
            self._check_single_index(a, index)

class TestFloatNonIntegerArgument:
    """
    These test that ``TypeError`` is raised when you try to use
    non-integers as arguments to for indexing and slicing e.g. ``a[0.0:5]``
    and ``a[0.5]``, or other functions like ``array.reshape(1., -1)``.

    """
    def test_valid_indexing(self):
        # These should raise no errors.
        a = np.array([[[5]]])

        a[np.array([0])]
        a[[0, 0]]
        a[:, [0, 0]]
        a[:, 0, :]
        a[:, :, :]

    def test_valid_slicing(self):
        # These should raise no errors.
        a = np.array([[[5]]])

        a[::]
        a[0:]
        a[:2]
        a[0:2]
        a[::2]
        a[1::2]
        a[:2:2]
        a[1:2:2]

    def test_non_integer_argument_errors(self):
        a = np.array([[5]])

        assert_raises(TypeError, np.reshape, a, (1., 1., -1))
        assert_raises(TypeError, np.reshape, a, (np.array(1.), -1))
        assert_raises(TypeError, np.take, a, [0], 1.)
        assert_raises(TypeError, np.take, a, [0], np.float64(1.))

    def test_non_integer_sequence_multiplication(self):
        # NumPy scalar sequence multiply should not work with non-integers
        def mult(a, b):
            return a * b

        assert_raises(TypeError, mult, [1], np.float64(3))
        # following should be OK
        mult([1], np.int_(3))

    def test_reduce_axis_float_index(self):
        d = np.zeros((3, 3, 3))
        assert_raises(TypeError, np.min, d, 0.5)
        assert_raises(TypeError, np.min, d, (0.5, 1))
        assert_raises(TypeError, np.min, d, (1, 2.2))
        assert_raises(TypeError, np.min, d, (.2, 1.2))


class TestBooleanIndexing:
    # Using a boolean as integer argument/indexing is an error.
    def test_bool_as_int_argument_errors(self):
        a = np.array([[[1]]])

        assert_raises(TypeError, np.reshape, a, (True, -1))
        assert_raises(TypeError, np.reshape, a, (np.bool(True), -1))
        # Note that operator.index(np.array(True)) does not work, a boolean
        # array is thus also deprecated, but not with the same message:
        assert_raises(TypeError, operator.index, np.array(True))
        assert_raises(TypeError, operator.index, np.True_)
        assert_raises(TypeError, np.take, args=(a, [0], False))

    def test_boolean_indexing_weirdness(self):
        # Weird boolean indexing things
        a = np.ones((2, 3, 4))
        assert a[False, True, ...].shape == (0, 2, 3, 4)
        assert a[True, [0, 1], True, True, [1], [[2]]].shape == (1, 2)
        assert_raises(IndexError, lambda: a[False, [0, 1], ...])

    def test_boolean_indexing_fast_path(self):
        # These used to either give the wrong error, or incorrectly give no
        # error.
        a = np.ones((3, 3))

        # This used to incorrectly work (and give an array of shape (0,))
        idx1 = np.array([[False] * 9])
        assert_raises_regex(IndexError,
            "boolean index did not match indexed array along axis 0; "
            "size of axis is 3 but size of corresponding boolean axis is 1",
            lambda: a[idx1])

        # This used to incorrectly give a ValueError: operands could not be broadcast together
        idx2 = np.array([[False] * 8 + [True]])
        assert_raises_regex(IndexError,
            "boolean index did not match indexed array along axis 0; "
            "size of axis is 3 but size of corresponding boolean axis is 1",
            lambda: a[idx2])

        # This is the same as it used to be. The above two should work like this.
        idx3 = np.array([[False] * 10])
        assert_raises_regex(IndexError,
            "boolean index did not match indexed array along axis 0; "
            "size of axis is 3 but size of corresponding boolean axis is 1",
            lambda: a[idx3])

        # This used to give ValueError: non-broadcastable operand
        a = np.ones((1, 1, 2))
        idx = np.array([[[True], [False]]])
        assert_raises_regex(IndexError,
            "boolean index did not match indexed array along axis 1; "
            "size of axis is 1 but size of corresponding boolean axis is 2",
            lambda: a[idx])


class TestArrayToIndexDeprecation:
    """Creating an index from array not 0-D is an error.

    """
    def test_array_to_index_error(self):
        # so no exception is expected. The raising is effectively tested above.
        a = np.array([[[1]]])

        assert_raises(TypeError, operator.index, np.array([1]))
        assert_raises(TypeError, np.reshape, a, (a, -1))
        assert_raises(TypeError, np.take, a, [0], a)


class TestNonIntegerArrayLike:
    """Tests that array_likes only valid if can safely cast to integer.

    For instance, lists give IndexError when they cannot be safely cast to
    an integer.

    """
    def test_basic(self):
        a = np.arange(10)

        assert_raises(IndexError, a.__getitem__, [0.5, 1.5])
        assert_raises(IndexError, a.__getitem__, (['1', '2'],))

        # The following is valid
        a.__getitem__([])


class TestMultipleEllipsisError:
    """An index can only have a single ellipsis.

    """
    def test_basic(self):
        a = np.arange(10)
        assert_raises(IndexError, lambda: a[..., ...])
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 2,))
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 3,))


class TestCApiAccess:
    def test_getitem(self):
        subscript = functools.partial(array_indexing, 0)

        # 0-d arrays don't work:
        assert_raises(IndexError, subscript, np.ones(()), 0)
        # Out of bound values:
        assert_raises(IndexError, subscript, np.ones(10), 11)
        assert_raises(IndexError, subscript, np.ones(10), -11)
        assert_raises(IndexError, subscript, np.ones((10, 10)), 11)
        assert_raises(IndexError, subscript, np.ones((10, 10)), -11)

        a = np.arange(10)
        assert_array_equal(a[4], subscript(a, 4))
        a = a.reshape(5, 2)
        assert_array_equal(a[-4], subscript(a, -4))

    def test_setitem(self):
        assign = functools.partial(array_indexing, 1)

        # Deletion is impossible:
        assert_raises(ValueError, assign, np.ones(10), 0)
        # 0-d arrays don't work:
        assert_raises(IndexError, assign, np.ones(()), 0, 0)
        # Out of bound values:
        assert_raises(IndexError, assign, np.ones(10), 11, 0)
        assert_raises(IndexError, assign, np.ones(10), -11, 0)
        assert_raises(IndexError, assign, np.ones((10, 10)), 11, 0)
        assert_raises(IndexError, assign, np.ones((10, 10)), -11, 0)

        a = np.arange(10)
        assign(a, 4, 10)
        assert_(a[4] == 10)

        a = a.reshape(5, 2)
        assign(a, 4, 10)
        assert_array_equal(a[-1], [10, 10])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_strings.py ===
import operator
import sys

import pytest

import numpy as np
from numpy.testing import IS_PYPY, assert_array_equal, assert_raises
from numpy.testing._private.utils import requires_memory

COMPARISONS = [
    (operator.eq, np.equal, "=="),
    (operator.ne, np.not_equal, "!="),
    (operator.lt, np.less, "<"),
    (operator.le, np.less_equal, "<="),
    (operator.gt, np.greater, ">"),
    (operator.ge, np.greater_equal, ">="),
]

MAX = np.iinfo(np.int64).max

IS_PYPY_LT_7_3_16 = IS_PYPY and sys.implementation.version < (7, 3, 16)

@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
def test_mixed_string_comparison_ufuncs_fail(op, ufunc, sym):
    arr_string = np.array(["a", "b"], dtype="S")
    arr_unicode = np.array(["a", "c"], dtype="U")

    with pytest.raises(TypeError, match="did not contain a loop"):
        ufunc(arr_string, arr_unicode)

    with pytest.raises(TypeError, match="did not contain a loop"):
        ufunc(arr_unicode, arr_string)

@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
def test_mixed_string_comparisons_ufuncs_with_cast(op, ufunc, sym):
    arr_string = np.array(["a", "b"], dtype="S")
    arr_unicode = np.array(["a", "c"], dtype="U")

    # While there is no loop, manual casting is acceptable:
    res1 = ufunc(arr_string, arr_unicode, signature="UU->?", casting="unsafe")
    res2 = ufunc(arr_string, arr_unicode, signature="SS->?", casting="unsafe")

    expected = op(arr_string.astype("U"), arr_unicode)
    assert_array_equal(res1, expected)
    assert_array_equal(res2, expected)


@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
@pytest.mark.parametrize("dtypes", [
        ("S2", "S2"), ("S2", "S10"),
        ("<U1", "<U1"), ("<U1", ">U1"), (">U1", ">U1"),
        ("<U1", "<U10"), ("<U1", ">U10")])
@pytest.mark.parametrize("aligned", [True, False])
def test_string_comparisons(op, ufunc, sym, dtypes, aligned):
    # ensure native byte-order for the first view to stay within unicode range
    native_dt = np.dtype(dtypes[0]).newbyteorder("=")
    arr = np.arange(2**15).view(native_dt).astype(dtypes[0])
    if not aligned:
        # Make `arr` unaligned:
        new = np.zeros(arr.nbytes + 1, dtype=np.uint8)[1:].view(dtypes[0])
        new[...] = arr
        arr = new

    arr2 = arr.astype(dtypes[1], copy=True)
    np.random.shuffle(arr2)
    arr[0] = arr2[0]  # make sure one matches

    expected = [op(d1, d2) for d1, d2 in zip(arr.tolist(), arr2.tolist())]
    assert_array_equal(op(arr, arr2), expected)
    assert_array_equal(ufunc(arr, arr2), expected)
    assert_array_equal(
        np.char.compare_chararrays(arr, arr2, sym, False), expected
    )

    expected = [op(d2, d1) for d1, d2 in zip(arr.tolist(), arr2.tolist())]
    assert_array_equal(op(arr2, arr), expected)
    assert_array_equal(ufunc(arr2, arr), expected)
    assert_array_equal(
        np.char.compare_chararrays(arr2, arr, sym, False), expected
    )


@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
@pytest.mark.parametrize("dtypes", [
        ("S2", "S2"), ("S2", "S10"), ("<U1", "<U1"), ("<U1", ">U10")])
def test_string_comparisons_empty(op, ufunc, sym, dtypes):
    arr = np.empty((1, 0, 1, 5), dtype=dtypes[0])
    arr2 = np.empty((100, 1, 0, 1), dtype=dtypes[1])

    expected = np.empty(np.broadcast_shapes(arr.shape, arr2.shape), dtype=bool)
    assert_array_equal(op(arr, arr2), expected)
    assert_array_equal(ufunc(arr, arr2), expected)
    assert_array_equal(
        np.char.compare_chararrays(arr, arr2, sym, False), expected
    )


@pytest.mark.parametrize("str_dt", ["S", "U"])
@pytest.mark.parametrize("float_dt", np.typecodes["AllFloat"])
def test_float_to_string_cast(str_dt, float_dt):
    float_dt = np.dtype(float_dt)
    fi = np.finfo(float_dt)
    arr = np.array([np.nan, np.inf, -np.inf, fi.max, fi.min], dtype=float_dt)
    expected = ["nan", "inf", "-inf", str(fi.max), str(fi.min)]
    if float_dt.kind == "c":
        expected = [f"({r}+0j)" for r in expected]

    res = arr.astype(str_dt)
    assert_array_equal(res, np.array(expected, dtype=str_dt))


@pytest.mark.parametrize("str_dt", "US")
@pytest.mark.parametrize("size", [-1, np.iinfo(np.intc).max])
def test_string_size_dtype_errors(str_dt, size):
    if size > 0:
        size = size // np.dtype(f"{str_dt}1").itemsize + 1

    with pytest.raises(ValueError):
        np.dtype((str_dt, size))
    with pytest.raises(TypeError):
        np.dtype(f"{str_dt}{size}")


@pytest.mark.parametrize("str_dt", "US")
def test_string_size_dtype_large_repr(str_dt):
    size = np.iinfo(np.intc).max // np.dtype(f"{str_dt}1").itemsize
    size_str = str(size)

    dtype = np.dtype((str_dt, size))
    assert size_str in dtype.str
    assert size_str in str(dtype)
    assert size_str in repr(dtype)


@pytest.mark.slow
@requires_memory(2 * np.iinfo(np.intc).max)
@pytest.mark.parametrize("str_dt", "US")
def test_large_string_coercion_error(str_dt):
    very_large = np.iinfo(np.intc).max // np.dtype(f"{str_dt}1").itemsize
    try:
        large_string = "A" * (very_large + 1)
    except Exception:
        # We may not be able to create this Python string on 32bit.
        pytest.skip("python failed to create huge string")

    class MyStr:
        def __str__(self):
            return large_string

    try:
        # TypeError from NumPy, or OverflowError from 32bit Python.
        with pytest.raises((TypeError, OverflowError)):
            np.array([large_string], dtype=str_dt)

        # Same as above, but input has to be converted to a string.
        with pytest.raises((TypeError, OverflowError)):
            np.array([MyStr()], dtype=str_dt)
    except MemoryError:
        # Catch memory errors, because `requires_memory` would do so.
        raise AssertionError("Ops should raise before any large allocation.")

@pytest.mark.slow
@requires_memory(2 * np.iinfo(np.intc).max)
@pytest.mark.parametrize("str_dt", "US")
def test_large_string_addition_error(str_dt):
    very_large = np.iinfo(np.intc).max // np.dtype(f"{str_dt}1").itemsize

    a = np.array(["A" * very_large], dtype=str_dt)
    b = np.array("B", dtype=str_dt)
    try:
        with pytest.raises(TypeError):
            np.add(a, b)
        with pytest.raises(TypeError):
            np.add(a, a)
    except MemoryError:
        # Catch memory errors, because `requires_memory` would do so.
        raise AssertionError("Ops should raise before any large allocation.")


def test_large_string_cast():
    very_large = np.iinfo(np.intc).max // 4
    # Could be nice to test very large path, but it makes too many huge
    # allocations right now (need non-legacy cast loops for this).
    # a = np.array([], dtype=np.dtype(("S", very_large)))
    # assert a.astype("U").dtype.itemsize == very_large * 4

    a = np.array([], dtype=np.dtype(("S", very_large + 1)))
    # It is not perfect but OK if this raises a MemoryError during setup
    # (this happens due clunky code and/or buffer setup.)
    with pytest.raises((TypeError, MemoryError)):
        a.astype("U")


@pytest.mark.parametrize("dt", ["S", "U", "T"])
class TestMethods:

    @pytest.mark.parametrize("in1,in2,out", [
        ("", "", ""),
        ("abc", "abc", "abcabc"),
        ("12345", "12345", "1234512345"),
        ("MixedCase", "MixedCase", "MixedCaseMixedCase"),
        ("12345 \0 ", "12345 \0 ", "12345 \0 12345 \0 "),
        ("UPPER", "UPPER", "UPPERUPPER"),
        (["abc", "def"], ["hello", "world"], ["abchello", "defworld"]),
    ])
    def test_add(self, in1, in2, out, dt):
        in1 = np.array(in1, dtype=dt)
        in2 = np.array(in2, dtype=dt)
        out = np.array(out, dtype=dt)
        assert_array_equal(np.strings.add(in1, in2), out)

    @pytest.mark.parametrize("in1,in2,out", [
        ("abc", 3, "abcabcabc"),
        ("abc", 0, ""),
        ("abc", -1, ""),
        (["abc", "def"], [1, 4], ["abc", "defdefdefdef"]),
    ])
    def test_multiply(self, in1, in2, out, dt):
        in1 = np.array(in1, dtype=dt)
        out = np.array(out, dtype=dt)
        assert_array_equal(np.strings.multiply(in1, in2), out)

    def test_multiply_raises(self, dt):
        with pytest.raises(TypeError, match="unsupported type"):
            np.strings.multiply(np.array("abc", dtype=dt), 3.14)

        with pytest.raises(OverflowError):
            np.strings.multiply(np.array("abc", dtype=dt), sys.maxsize)

    def test_inplace_multiply(self, dt):
        arr = np.array(['foo ', 'bar'], dtype=dt)
        arr *= 2
        if dt != "T":
            assert_array_equal(arr, np.array(['foo ', 'barb'], dtype=dt))
        else:
            assert_array_equal(arr, ['foo foo ', 'barbar'])

        with pytest.raises(OverflowError):
            arr *= sys.maxsize

    @pytest.mark.parametrize("i_dt", [np.int8, np.int16, np.int32,
                                      np.int64, np.int_])
    def test_multiply_integer_dtypes(self, i_dt, dt):
        a = np.array("abc", dtype=dt)
        i = np.array(3, dtype=i_dt)
        res = np.array("abcabcabc", dtype=dt)
        assert_array_equal(np.strings.multiply(a, i), res)

    @pytest.mark.parametrize("in_,out", [
        ("", False),
        ("a", True),
        ("A", True),
        ("\n", False),
        ("abc", True),
        ("aBc123", False),
        ("abc\n", False),
        (["abc", "aBc123"], [True, False]),
    ])
    def test_isalpha(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isalpha(in_), out)

    @pytest.mark.parametrize("in_,out", [
        ('', False),
        ('a', True),
        ('A', True),
        ('\n', False),
        ('123abc456', True),
        ('a1b3c', True),
        ('aBc000 ', False),
        ('abc\n', False),
    ])
    def test_isalnum(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isalnum(in_), out)

    @pytest.mark.parametrize("in_,out", [
        ("", False),
        ("a", False),
        ("0", True),
        ("012345", True),
        ("012345a", False),
        (["a", "012345"], [False, True]),
    ])
    def test_isdigit(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isdigit(in_), out)

    @pytest.mark.parametrize("in_,out", [
        ("", False),
        ("a", False),
        ("1", False),
        (" ", True),
        ("\t", True),
        ("\r", True),
        ("\n", True),
        (" \t\r \n", True),
        (" \t\r\na", False),
        (["\t1", " \t\r \n"], [False, True])
    ])
    def test_isspace(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isspace(in_), out)

    @pytest.mark.parametrize("in_,out", [
        ('', False),
        ('a', True),
        ('A', False),
        ('\n', False),
        ('abc', True),
        ('aBc', False),
        ('abc\n', True),
    ])
    def test_islower(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.islower(in_), out)

    @pytest.mark.parametrize("in_,out", [
        ('', False),
        ('a', False),
        ('A', True),
        ('\n', False),
        ('ABC', True),
        ('AbC', False),
        ('ABC\n', True),
    ])
    def test_isupper(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isupper(in_), out)

    @pytest.mark.parametrize("in_,out", [
        ('', False),
        ('a', False),
        ('A', True),
        ('\n', False),
        ('A Titlecased Line', True),
        ('A\nTitlecased Line', True),
        ('A Titlecased, Line', True),
        ('Not a capitalized String', False),
        ('Not\ta Titlecase String', False),
        ('Not--a Titlecase String', False),
        ('NOT', False),
    ])
    def test_istitle(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.istitle(in_), out)

    @pytest.mark.parametrize("in_,out", [
        ("", 0),
        ("abc", 3),
        ("12345", 5),
        ("MixedCase", 9),
        ("12345 \x00 ", 8),
        ("UPPER", 5),
        (["abc", "12345 \x00 "], [3, 8]),
    ])
    def test_str_len(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.str_len(in_), out)

    @pytest.mark.parametrize("a,sub,start,end,out", [
        ("abcdefghiabc", "abc", 0, None, 0),
        ("abcdefghiabc", "abc", 1, None, 9),
        ("abcdefghiabc", "def", 4, None, -1),
        ("abc", "", 0, None, 0),
        ("abc", "", 3, None, 3),
        ("abc", "", 4, None, -1),
        ("rrarrrrrrrrra", "a", 0, None, 2),
        ("rrarrrrrrrrra", "a", 4, None, 12),
        ("rrarrrrrrrrra", "a", 4, 6, -1),
        ("", "", 0, None, 0),
        ("", "", 1, 1, -1),
        ("", "", MAX, 0, -1),
        ("", "xx", 0, None, -1),
        ("", "xx", 1, 1, -1),
        ("", "xx", MAX, 0, -1),
        pytest.param(99 * "a" + "b", "b", 0, None, 99,
                     id="99*a+b-b-0-None-99"),
        pytest.param(98 * "a" + "ba", "ba", 0, None, 98,
                     id="98*a+ba-ba-0-None-98"),
        pytest.param(100 * "a", "b", 0, None, -1,
                     id="100*a-b-0-None--1"),
        pytest.param(30000 * "a" + 100 * "b", 100 * "b", 0, None, 30000,
                     id="30000*a+100*b-100*b-0-None-30000"),
        pytest.param(30000 * "a", 100 * "b", 0, None, -1,
                     id="30000*a-100*b-0-None--1"),
        pytest.param(15000 * "a" + 15000 * "b", 15000 * "b", 0, None, 15000,
                     id="15000*a+15000*b-15000*b-0-None-15000"),
        pytest.param(15000 * "a" + 15000 * "b", 15000 * "c", 0, None, -1,
                     id="15000*a+15000*b-15000*c-0-None--1"),
        (["abcdefghiabc", "rrarrrrrrrrra"], ["def", "arr"], [0, 3],
         None, [3, -1]),
        ("Ae¢☃€ 😊" * 2, "😊", 0, None, 6),
        ("Ae¢☃€ 😊" * 2, "😊", 7, None, 13),
        pytest.param("A" * (2 ** 17), r"[\w]+\Z", 0, None, -1,
                     id=r"A*2**17-[\w]+\Z-0-None--1"),
    ])
    def test_find(self, a, sub, start, end, out, dt):
        if "😊" in a and dt == "S":
            pytest.skip("Bytes dtype does not support non-ascii input")
        a = np.array(a, dtype=dt)
        sub = np.array(sub, dtype=dt)
        assert_array_equal(np.strings.find(a, sub, start, end), out)

    @pytest.mark.parametrize("a,sub,start,end,out", [
        ("abcdefghiabc", "abc", 0, None, 9),
        ("abcdefghiabc", "", 0, None, 12),
        ("abcdefghiabc", "abcd", 0, None, 0),
        ("abcdefghiabc", "abcz", 0, None, -1),
        ("abc", "", 0, None, 3),
        ("abc", "", 3, None, 3),
        ("abc", "", 4, None, -1),
        ("rrarrrrrrrrra", "a", 0, None, 12),
        ("rrarrrrrrrrra", "a", 4, None, 12),
        ("rrarrrrrrrrra", "a", 4, 6, -1),
        (["abcdefghiabc", "rrarrrrrrrrra"], ["abc", "a"], [0, 0],
         None, [9, 12]),
        ("Ae¢☃€ 😊" * 2, "😊", 0, None, 13),
        ("Ae¢☃€ 😊" * 2, "😊", 0, 7, 6),
    ])
    def test_rfind(self, a, sub, start, end, out, dt):
        if "😊" in a and dt == "S":
            pytest.skip("Bytes dtype does not support non-ascii input")
        a = np.array(a, dtype=dt)
        sub = np.array(sub, dtype=dt)
        assert_array_equal(np.strings.rfind(a, sub, start, end), out)

    @pytest.mark.parametrize("a,sub,start,end,out", [
        ("aaa", "a", 0, None, 3),
        ("aaa", "b", 0, None, 0),
        ("aaa", "a", 1, None, 2),
        ("aaa", "a", 10, None, 0),
        ("aaa", "a", -1, None, 1),
        ("aaa", "a", -10, None, 3),
        ("aaa", "a", 0, 1, 1),
        ("aaa", "a", 0, 10, 3),
        ("aaa", "a", 0, -1, 2),
        ("aaa", "a", 0, -10, 0),
        ("aaa", "", 1, None, 3),
        ("aaa", "", 3, None, 1),
        ("aaa", "", 10, None, 0),
        ("aaa", "", -1, None, 2),
        ("aaa", "", -10, None, 4),
        ("aaa", "aaaa", 0, None, 0),
        pytest.param(98 * "a" + "ba", "ba", 0, None, 1,
                     id="98*a+ba-ba-0-None-1"),
        pytest.param(30000 * "a" + 100 * "b", 100 * "b", 0, None, 1,
                     id="30000*a+100*b-100*b-0-None-1"),
        pytest.param(30000 * "a", 100 * "b", 0, None, 0,
                     id="30000*a-100*b-0-None-0"),
        pytest.param(30000 * "a" + 100 * "ab", "ab", 0, None, 100,
                     id="30000*a+100*ab-ab-0-None-100"),
        pytest.param(15000 * "a" + 15000 * "b", 15000 * "b", 0, None, 1,
                     id="15000*a+15000*b-15000*b-0-None-1"),
        pytest.param(15000 * "a" + 15000 * "b", 15000 * "c", 0, None, 0,
                     id="15000*a+15000*b-15000*c-0-None-0"),
        ("", "", 0, None, 1),
        ("", "", 1, 1, 0),
        ("", "", MAX, 0, 0),
        ("", "xx", 0, None, 0),
        ("", "xx", 1, 1, 0),
        ("", "xx", MAX, 0, 0),
        (["aaa", ""], ["a", ""], [0, 0], None, [3, 1]),
        ("Ae¢☃€ 😊" * 100, "😊", 0, None, 100),
    ])
    def test_count(self, a, sub, start, end, out, dt):
        if "😊" in a and dt == "S":
            pytest.skip("Bytes dtype does not support non-ascii input")
        a = np.array(a, dtype=dt)
        sub = np.array(sub, dtype=dt)
        assert_array_equal(np.strings.count(a, sub, start, end), out)

    @pytest.mark.parametrize("a,prefix,start,end,out", [
        ("hello", "he", 0, None, True),
        ("hello", "hello", 0, None, True),
        ("hello", "hello world", 0, None, False),
        ("hello", "", 0, None, True),
        ("hello", "ello", 0, None, False),
        ("hello", "ello", 1, None, True),
        ("hello", "o", 4, None, True),
        ("hello", "o", 5, None, False),
        ("hello", "", 5, None, True),
        ("hello", "lo", 6, None, False),
        ("helloworld", "lowo", 3, None, True),
        ("helloworld", "lowo", 3, 7, True),
        ("helloworld", "lowo", 3, 6, False),
        ("", "", 0, 1, True),
        ("", "", 0, 0, True),
        ("", "", 1, 0, False),
        ("hello", "he", 0, -1, True),
        ("hello", "he", -53, -1, True),
        ("hello", "hello", 0, -1, False),
        ("hello", "hello world", -1, -10, False),
        ("hello", "ello", -5, None, False),
        ("hello", "ello", -4, None, True),
        ("hello", "o", -2, None, False),
        ("hello", "o", -1, None, True),
        ("hello", "", -3, -3, True),
        ("hello", "lo", -9, None, False),
        (["hello", ""], ["he", ""], [0, 0], None, [True, True]),
    ])
    def test_startswith(self, a, prefix, start, end, out, dt):
        a = np.array(a, dtype=dt)
        prefix = np.array(prefix, dtype=dt)
        assert_array_equal(np.strings.startswith(a, prefix, start, end), out)

    @pytest.mark.parametrize("a,suffix,start,end,out", [
        ("hello", "lo", 0, None, True),
        ("hello", "he", 0, None, False),
        ("hello", "", 0, None, True),
        ("hello", "hello world", 0, None, False),
        ("helloworld", "worl", 0, None, False),
        ("helloworld", "worl", 3, 9, True),
        ("helloworld", "world", 3, 12, True),
        ("helloworld", "lowo", 1, 7, True),
        ("helloworld", "lowo", 2, 7, True),
        ("helloworld", "lowo", 3, 7, True),
        ("helloworld", "lowo", 4, 7, False),
        ("helloworld", "lowo", 3, 8, False),
        ("ab", "ab", 0, 1, False),
        ("ab", "ab", 0, 0, False),
        ("", "", 0, 1, True),
        ("", "", 0, 0, True),
        ("", "", 1, 0, False),
        ("hello", "lo", -2, None, True),
        ("hello", "he", -2, None, False),
        ("hello", "", -3, -3, True),
        ("hello", "hello world", -10, -2, False),
        ("helloworld", "worl", -6, None, False),
        ("helloworld", "worl", -5, -1, True),
        ("helloworld", "worl", -5, 9, True),
        ("helloworld", "world", -7, 12, True),
        ("helloworld", "lowo", -99, -3, True),
        ("helloworld", "lowo", -8, -3, True),
        ("helloworld", "lowo", -7, -3, True),
        ("helloworld", "lowo", 3, -4, False),
        ("helloworld", "lowo", -8, -2, False),
        (["hello", "helloworld"], ["lo", "worl"], [0, -6], None,
         [True, False]),
    ])
    def test_endswith(self, a, suffix, start, end, out, dt):
        a = np.array(a, dtype=dt)
        suffix = np.array(suffix, dtype=dt)
        assert_array_equal(np.strings.endswith(a, suffix, start, end), out)

    @pytest.mark.parametrize("a,chars,out", [
        ("", None, ""),
        ("   hello   ", None, "hello   "),
        ("hello", None, "hello"),
        (" \t\n\r\f\vabc \t\n\r\f\v", None, "abc \t\n\r\f\v"),
        (["   hello   ", "hello"], None, ["hello   ", "hello"]),
        ("", "", ""),
        ("", "xyz", ""),
        ("hello", "", "hello"),
        ("xyzzyhelloxyzzy", "xyz", "helloxyzzy"),
        ("hello", "xyz", "hello"),
        ("xyxz", "xyxz", ""),
        ("xyxzx", "x", "yxzx"),
        (["xyzzyhelloxyzzy", "hello"], ["xyz", "xyz"],
         ["helloxyzzy", "hello"]),
        (["ba", "ac", "baa", "bba"], "b", ["a", "ac", "aa", "a"]),
    ])
    def test_lstrip(self, a, chars, out, dt):
        a = np.array(a, dtype=dt)
        out = np.array(out, dtype=dt)
        if chars is not None:
            chars = np.array(chars, dtype=dt)
            assert_array_equal(np.strings.lstrip(a, chars), out)
        else:
            assert_array_equal(np.strings.lstrip(a), out)

    @pytest.mark.parametrize("a,chars,out", [
        ("", None, ""),
        ("   hello   ", None, "   hello"),
        ("hello", None, "hello"),
        (" \t\n\r\f\vabc \t\n\r\f\v", None, " \t\n\r\f\vabc"),
        (["   hello   ", "hello"], None, ["   hello", "hello"]),
        ("", "", ""),
        ("", "xyz", ""),
        ("hello", "", "hello"),
        (["hello    ", "abcdefghijklmnop"], None,
         ["hello", "abcdefghijklmnop"]),
        ("xyzzyhelloxyzzy", "xyz", "xyzzyhello"),
        ("hello", "xyz", "hello"),
        ("xyxz", "xyxz", ""),
        ("    ", None, ""),
        ("xyxzx", "x", "xyxz"),
        (["xyzzyhelloxyzzy", "hello"], ["xyz", "xyz"],
         ["xyzzyhello", "hello"]),
        (["ab", "ac", "aab", "abb"], "b", ["a", "ac", "aa", "a"]),
    ])
    def test_rstrip(self, a, chars, out, dt):
        a = np.array(a, dtype=dt)
        out = np.array(out, dtype=dt)
        if chars is not None:
            chars = np.array(chars, dtype=dt)
            assert_array_equal(np.strings.rstrip(a, chars), out)
        else:
            assert_array_equal(np.strings.rstrip(a), out)

    @pytest.mark.parametrize("a,chars,out", [
        ("", None, ""),
        ("   hello   ", None, "hello"),
        ("hello", None, "hello"),
        (" \t\n\r\f\vabc \t\n\r\f\v", None, "abc"),
        (["   hello   ", "hello"], None, ["hello", "hello"]),
        ("", "", ""),
        ("", "xyz", ""),
        ("hello", "", "hello"),
        ("xyzzyhelloxyzzy", "xyz", "hello"),
        ("hello", "xyz", "hello"),
        ("xyxz", "xyxz", ""),
        ("xyxzx", "x", "yxz"),
        (["xyzzyhelloxyzzy", "hello"], ["xyz", "xyz"],
         ["hello", "hello"]),
        (["bab", "ac", "baab", "bbabb"], "b", ["a", "ac", "aa", "a"]),
    ])
    def test_strip(self, a, chars, out, dt):
        a = np.array(a, dtype=dt)
        if chars is not None:
            chars = np.array(chars, dtype=dt)
        out = np.array(out, dtype=dt)
        assert_array_equal(np.strings.strip(a, chars), out)

    @pytest.mark.parametrize("buf,old,new,count,res", [
        ("", "", "", -1, ""),
        ("", "", "A", -1, "A"),
        ("", "A", "", -1, ""),
        ("", "A", "A", -1, ""),
        ("", "", "", 100, ""),
        ("", "", "A", 100, "A"),
        ("A", "", "", -1, "A"),
        ("A", "", "*", -1, "*A*"),
        ("A", "", "*1", -1, "*1A*1"),
        ("A", "", "*-#", -1, "*-#A*-#"),
        ("AA", "", "*-", -1, "*-A*-A*-"),
        ("AA", "", "*-", -1, "*-A*-A*-"),
        ("AA", "", "*-", 4, "*-A*-A*-"),
        ("AA", "", "*-", 3, "*-A*-A*-"),
        ("AA", "", "*-", 2, "*-A*-A"),
        ("AA", "", "*-", 1, "*-AA"),
        ("AA", "", "*-", 0, "AA"),
        ("A", "A", "", -1, ""),
        ("AAA", "A", "", -1, ""),
        ("AAA", "A", "", -1, ""),
        ("AAA", "A", "", 4, ""),
        ("AAA", "A", "", 3, ""),
        ("AAA", "A", "", 2, "A"),
        ("AAA", "A", "", 1, "AA"),
        ("AAA", "A", "", 0, "AAA"),
        ("AAAAAAAAAA", "A", "", -1, ""),
        ("ABACADA", "A", "", -1, "BCD"),
        ("ABACADA", "A", "", -1, "BCD"),
        ("ABACADA", "A", "", 5, "BCD"),
        ("ABACADA", "A", "", 4, "BCD"),
        ("ABACADA", "A", "", 3, "BCDA"),
        ("ABACADA", "A", "", 2, "BCADA"),
        ("ABACADA", "A", "", 1, "BACADA"),
        ("ABACADA", "A", "", 0, "ABACADA"),
        ("ABCAD", "A", "", -1, "BCD"),
        ("ABCADAA", "A", "", -1, "BCD"),
        ("BCD", "A", "", -1, "BCD"),
        ("*************", "A", "", -1, "*************"),
        ("^" + "A" * 1000 + "^", "A", "", 999, "^A^"),
        ("the", "the", "", -1, ""),
        ("theater", "the", "", -1, "ater"),
        ("thethe", "the", "", -1, ""),
        ("thethethethe", "the", "", -1, ""),
        ("theatheatheathea", "the", "", -1, "aaaa"),
        ("that", "the", "", -1, "that"),
        ("thaet", "the", "", -1, "thaet"),
        ("here and there", "the", "", -1, "here and re"),
        ("here and there and there", "the", "", -1, "here and re and re"),
        ("here and there and there", "the", "", 3, "here and re and re"),
        ("here and there and there", "the", "", 2, "here and re and re"),
        ("here and there and there", "the", "", 1, "here and re and there"),
        ("here and there and there", "the", "", 0, "here and there and there"),
        ("here and there and there", "the", "", -1, "here and re and re"),
        ("abc", "the", "", -1, "abc"),
        ("abcdefg", "the", "", -1, "abcdefg"),
        ("bbobob", "bob", "", -1, "bob"),
        ("bbobobXbbobob", "bob", "", -1, "bobXbob"),
        ("aaaaaaabob", "bob", "", -1, "aaaaaaa"),
        ("aaaaaaa", "bob", "", -1, "aaaaaaa"),
        ("Who goes there?", "o", "o", -1, "Who goes there?"),
        ("Who goes there?", "o", "O", -1, "WhO gOes there?"),
        ("Who goes there?", "o", "O", -1, "WhO gOes there?"),
        ("Who goes there?", "o", "O", 3, "WhO gOes there?"),
        ("Who goes there?", "o", "O", 2, "WhO gOes there?"),
        ("Who goes there?", "o", "O", 1, "WhO goes there?"),
        ("Who goes there?", "o", "O", 0, "Who goes there?"),
        ("Who goes there?", "a", "q", -1, "Who goes there?"),
        ("Who goes there?", "W", "w", -1, "who goes there?"),
        ("WWho goes there?WW", "W", "w", -1, "wwho goes there?ww"),
        ("Who goes there?", "?", "!", -1, "Who goes there!"),
        ("Who goes there??", "?", "!", -1, "Who goes there!!"),
        ("Who goes there?", ".", "!", -1, "Who goes there?"),
        ("This is a tissue", "is", "**", -1, "Th** ** a t**sue"),
        ("This is a tissue", "is", "**", -1, "Th** ** a t**sue"),
        ("This is a tissue", "is", "**", 4, "Th** ** a t**sue"),
        ("This is a tissue", "is", "**", 3, "Th** ** a t**sue"),
        ("This is a tissue", "is", "**", 2, "Th** ** a tissue"),
        ("This is a tissue", "is", "**", 1, "Th** is a tissue"),
        ("This is a tissue", "is", "**", 0, "This is a tissue"),
        ("bobob", "bob", "cob", -1, "cobob"),
        ("bobobXbobobob", "bob", "cob", -1, "cobobXcobocob"),
        ("bobob", "bot", "bot", -1, "bobob"),
        ("Reykjavik", "k", "KK", -1, "ReyKKjaviKK"),
        ("Reykjavik", "k", "KK", -1, "ReyKKjaviKK"),
        ("Reykjavik", "k", "KK", 2, "ReyKKjaviKK"),
        ("Reykjavik", "k", "KK", 1, "ReyKKjavik"),
        ("Reykjavik", "k", "KK", 0, "Reykjavik"),
        ("A.B.C.", ".", "----", -1, "A----B----C----"),
        ("Reykjavik", "q", "KK", -1, "Reykjavik"),
        ("spam, spam, eggs and spam", "spam", "ham", -1,
            "ham, ham, eggs and ham"),
        ("spam, spam, eggs and spam", "spam", "ham", -1,
            "ham, ham, eggs and ham"),
        ("spam, spam, eggs and spam", "spam", "ham", 4,
            "ham, ham, eggs and ham"),
        ("spam, spam, eggs and spam", "spam", "ham", 3,
            "ham, ham, eggs and ham"),
        ("spam, spam, eggs and spam", "spam", "ham", 2,
            "ham, ham, eggs and spam"),
        ("spam, spam, eggs and spam", "spam", "ham", 1,
            "ham, spam, eggs and spam"),
        ("spam, spam, eggs and spam", "spam", "ham", 0,
            "spam, spam, eggs and spam"),
        ("bobobob", "bobob", "bob", -1, "bobob"),
        ("bobobobXbobobob", "bobob", "bob", -1, "bobobXbobob"),
        ("BOBOBOB", "bob", "bobby", -1, "BOBOBOB"),
        ("one!two!three!", "!", "@", 1, "one@two!three!"),
        ("one!two!three!", "!", "", -1, "onetwothree"),
        ("one!two!three!", "!", "@", 2, "one@two@three!"),
        ("one!two!three!", "!", "@", 3, "one@two@three@"),
        ("one!two!three!", "!", "@", 4, "one@two@three@"),
        ("one!two!three!", "!", "@", 0, "one!two!three!"),
        ("one!two!three!", "!", "@", -1, "one@two@three@"),
        ("one!two!three!", "x", "@", -1, "one!two!three!"),
        ("one!two!three!", "x", "@", 2, "one!two!three!"),
        ("abc", "", "-", -1, "-a-b-c-"),
        ("abc", "", "-", 3, "-a-b-c"),
        ("abc", "", "-", 0, "abc"),
        ("abc", "ab", "--", 0, "abc"),
        ("abc", "xy", "--", -1, "abc"),
        (["abbc", "abbd"], "b", "z", [1, 2], ["azbc", "azzd"]),
    ])
    def test_replace(self, buf, old, new, count, res, dt):
        if "😊" in buf and dt == "S":
            pytest.skip("Bytes dtype does not support non-ascii input")
        buf = np.array(buf, dtype=dt)
        old = np.array(old, dtype=dt)
        new = np.array(new, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.replace(buf, old, new, count), res)

    @pytest.mark.parametrize("buf,sub,start,end,res", [
        ("abcdefghiabc", "", 0, None, 0),
        ("abcdefghiabc", "def", 0, None, 3),
        ("abcdefghiabc", "abc", 0, None, 0),
        ("abcdefghiabc", "abc", 1, None, 9),
    ])
    def test_index(self, buf, sub, start, end, res, dt):
        buf = np.array(buf, dtype=dt)
        sub = np.array(sub, dtype=dt)
        assert_array_equal(np.strings.index(buf, sub, start, end), res)

    @pytest.mark.parametrize("buf,sub,start,end", [
        ("abcdefghiabc", "hib", 0, None),
        ("abcdefghiab", "abc", 1, None),
        ("abcdefghi", "ghi", 8, None),
        ("abcdefghi", "ghi", -1, None),
        ("rrarrrrrrrrra", "a", 4, 6),
    ])
    def test_index_raises(self, buf, sub, start, end, dt):
        buf = np.array(buf, dtype=dt)
        sub = np.array(sub, dtype=dt)
        with pytest.raises(ValueError, match="substring not found"):
            np.strings.index(buf, sub, start, end)

    @pytest.mark.parametrize("buf,sub,start,end,res", [
        ("abcdefghiabc", "", 0, None, 12),
        ("abcdefghiabc", "def", 0, None, 3),
        ("abcdefghiabc", "abc", 0, None, 9),
        ("abcdefghiabc", "abc", 0, -1, 0),
    ])
    def test_rindex(self, buf, sub, start, end, res, dt):
        buf = np.array(buf, dtype=dt)
        sub = np.array(sub, dtype=dt)
        assert_array_equal(np.strings.rindex(buf, sub, start, end), res)

    @pytest.mark.parametrize("buf,sub,start,end", [
        ("abcdefghiabc", "hib", 0, None),
        ("defghiabc", "def", 1, None),
        ("defghiabc", "abc", 0, -1),
        ("abcdefghi", "ghi", 0, 8),
        ("abcdefghi", "ghi", 0, -1),
        ("rrarrrrrrrrra", "a", 4, 6),
    ])
    def test_rindex_raises(self, buf, sub, start, end, dt):
        buf = np.array(buf, dtype=dt)
        sub = np.array(sub, dtype=dt)
        with pytest.raises(ValueError, match="substring not found"):
            np.strings.rindex(buf, sub, start, end)

    @pytest.mark.parametrize("buf,tabsize,res", [
        ("abc\rab\tdef\ng\thi", 8, "abc\rab      def\ng       hi"),
        ("abc\rab\tdef\ng\thi", 4, "abc\rab  def\ng   hi"),
        ("abc\r\nab\tdef\ng\thi", 8, "abc\r\nab      def\ng       hi"),
        ("abc\r\nab\tdef\ng\thi", 4, "abc\r\nab  def\ng   hi"),
        ("abc\r\nab\r\ndef\ng\r\nhi", 4, "abc\r\nab\r\ndef\ng\r\nhi"),
        (" \ta\n\tb", 1, "  a\n b"),
    ])
    def test_expandtabs(self, buf, tabsize, res, dt):
        buf = np.array(buf, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.expandtabs(buf, tabsize), res)

    def test_expandtabs_raises_overflow(self, dt):
        with pytest.raises(OverflowError, match="new string is too long"):
            np.strings.expandtabs(np.array("\ta\n\tb", dtype=dt), sys.maxsize)
            np.strings.expandtabs(np.array("\ta\n\tb", dtype=dt), 2**61)

    FILL_ERROR = "The fill character must be exactly one character long"

    def test_center_raises_multiple_character_fill(self, dt):
        buf = np.array("abc", dtype=dt)
        fill = np.array("**", dtype=dt)
        with pytest.raises(TypeError, match=self.FILL_ERROR):
            np.strings.center(buf, 10, fill)

    def test_ljust_raises_multiple_character_fill(self, dt):
        buf = np.array("abc", dtype=dt)
        fill = np.array("**", dtype=dt)
        with pytest.raises(TypeError, match=self.FILL_ERROR):
            np.strings.ljust(buf, 10, fill)

    def test_rjust_raises_multiple_character_fill(self, dt):
        buf = np.array("abc", dtype=dt)
        fill = np.array("**", dtype=dt)
        with pytest.raises(TypeError, match=self.FILL_ERROR):
            np.strings.rjust(buf, 10, fill)

    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('abc', 10, ' ', '   abc    '),
        ('abc', 6, ' ', ' abc  '),
        ('abc', 3, ' ', 'abc'),
        ('abc', 2, ' ', 'abc'),
        ('abc', 10, '*', '***abc****'),
    ])
    def test_center(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.center(buf, width, fillchar), res)

    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('abc', 10, ' ', 'abc       '),
        ('abc', 6, ' ', 'abc   '),
        ('abc', 3, ' ', 'abc'),
        ('abc', 2, ' ', 'abc'),
        ('abc', 10, '*', 'abc*******'),
    ])
    def test_ljust(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.ljust(buf, width, fillchar), res)

    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('abc', 10, ' ', '       abc'),
        ('abc', 6, ' ', '   abc'),
        ('abc', 3, ' ', 'abc'),
        ('abc', 2, ' ', 'abc'),
        ('abc', 10, '*', '*******abc'),
    ])
    def test_rjust(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.rjust(buf, width, fillchar), res)

    @pytest.mark.parametrize("buf,width,res", [
        ('123', 2, '123'),
        ('123', 3, '123'),
        ('0123', 4, '0123'),
        ('+123', 3, '+123'),
        ('+123', 4, '+123'),
        ('+123', 5, '+0123'),
        ('+0123', 5, '+0123'),
        ('-123', 3, '-123'),
        ('-123', 4, '-123'),
        ('-0123', 5, '-0123'),
        ('000', 3, '000'),
        ('34', 1, '34'),
        ('0034', 4, '0034'),
    ])
    def test_zfill(self, buf, width, res, dt):
        buf = np.array(buf, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.zfill(buf, width), res)

    @pytest.mark.parametrize("buf,sep,res1,res2,res3", [
        ("this is the partition method", "ti", "this is the par",
            "ti", "tion method"),
        ("http://www.python.org", "://", "http", "://", "www.python.org"),
        ("http://www.python.org", "?", "http://www.python.org", "", ""),
        ("http://www.python.org", "http://", "", "http://", "www.python.org"),
        ("http://www.python.org", "org", "http://www.python.", "org", ""),
        ("http://www.python.org", ["://", "?", "http://", "org"],
            ["http", "http://www.python.org", "", "http://www.python."],
            ["://", "", "http://", "org"],
            ["www.python.org", "", "www.python.org", ""]),
        ("mississippi", "ss", "mi", "ss", "issippi"),
        ("mississippi", "i", "m", "i", "ssissippi"),
        ("mississippi", "w", "mississippi", "", ""),
    ])
    def test_partition(self, buf, sep, res1, res2, res3, dt):
        buf = np.array(buf, dtype=dt)
        sep = np.array(sep, dtype=dt)
        res1 = np.array(res1, dtype=dt)
        res2 = np.array(res2, dtype=dt)
        res3 = np.array(res3, dtype=dt)
        act1, act2, act3 = np.strings.partition(buf, sep)
        assert_array_equal(act1, res1)
        assert_array_equal(act2, res2)
        assert_array_equal(act3, res3)
        assert_array_equal(act1 + act2 + act3, buf)

    @pytest.mark.parametrize("buf,sep,res1,res2,res3", [
        ("this is the partition method", "ti", "this is the parti",
            "ti", "on method"),
        ("http://www.python.org", "://", "http", "://", "www.python.org"),
        ("http://www.python.org", "?", "", "", "http://www.python.org"),
        ("http://www.python.org", "http://", "", "http://", "www.python.org"),
        ("http://www.python.org", "org", "http://www.python.", "org", ""),
        ("http://www.python.org", ["://", "?", "http://", "org"],
            ["http", "", "", "http://www.python."],
            ["://", "", "http://", "org"],
            ["www.python.org", "http://www.python.org", "www.python.org", ""]),
        ("mississippi", "ss", "missi", "ss", "ippi"),
        ("mississippi", "i", "mississipp", "i", ""),
        ("mississippi", "w", "", "", "mississippi"),
    ])
    def test_rpartition(self, buf, sep, res1, res2, res3, dt):
        buf = np.array(buf, dtype=dt)
        sep = np.array(sep, dtype=dt)
        res1 = np.array(res1, dtype=dt)
        res2 = np.array(res2, dtype=dt)
        res3 = np.array(res3, dtype=dt)
        act1, act2, act3 = np.strings.rpartition(buf, sep)
        assert_array_equal(act1, res1)
        assert_array_equal(act2, res2)
        assert_array_equal(act3, res3)
        assert_array_equal(act1 + act2 + act3, buf)

    @pytest.mark.parametrize("args", [
        (None,),
        (0,),
        (1,),
        (3,),
        (5,),
        (6,),  # test index past the end
        (-1,),
        (-3,),
        ([3, 4],),
        ([2, 4],),
        ([-3, 5],),
        ([0, -5],),
        (1, 4),
        (-3, 5),
        (None, -1),
        (0, [4, 2]),
        ([1, 2], [-1, -2]),
        (1, 5, 2),
        (None, None, -1),
        ([0, 6], [-1, 0], [2, -1]),
    ])
    def test_slice(self, args, dt):
        buf = np.array(["hello", "world"], dtype=dt)
        act = np.strings.slice(buf, *args)
        bcast_args = tuple(np.broadcast_to(arg, buf.shape) for arg in args)
        res = np.array([s[slice(*arg)]
                        for s, arg in zip(buf, zip(*bcast_args))],
                       dtype=dt)
        assert_array_equal(act, res)

    def test_slice_unsupported(self, dt):
        with pytest.raises(TypeError, match="did not contain a loop"):
            np.strings.slice(np.array([1, 2, 3]), 4)

        with pytest.raises(TypeError, match=r"Cannot cast ufunc '_slice' input .* from .* to dtype\('int(64|32)'\)"):
            np.strings.slice(np.array(['foo', 'bar'], dtype=dt), np.array(['foo', 'bar'], dtype=dt))

    @pytest.mark.parametrize("int_dt", [np.int8, np.int16, np.int32, np.int64,
                                        np.uint8, np.uint16, np.uint32, np.uint64])
    def test_slice_int_type_promotion(self, int_dt, dt):
        buf = np.array(["hello", "world"], dtype=dt)

        assert_array_equal(np.strings.slice(buf, int_dt(4)), np.array(["hell", "worl"], dtype=dt))
        assert_array_equal(np.strings.slice(buf, np.array([4, 4], dtype=int_dt)), np.array(["hell", "worl"], dtype=dt))

        assert_array_equal(np.strings.slice(buf, int_dt(2), int_dt(4)), np.array(["ll", "rl"], dtype=dt))
        assert_array_equal(np.strings.slice(buf, np.array([2, 2], dtype=int_dt), np.array([4, 4], dtype=int_dt)), np.array(["ll", "rl"], dtype=dt))

        assert_array_equal(np.strings.slice(buf, int_dt(0), int_dt(4), int_dt(2)), np.array(["hl", "wr"], dtype=dt))
        assert_array_equal(np.strings.slice(buf, np.array([0, 0], dtype=int_dt), np.array([4, 4], dtype=int_dt), np.array([2, 2], dtype=int_dt)), np.array(["hl", "wr"], dtype=dt))

@pytest.mark.parametrize("dt", ["U", "T"])
class TestMethodsWithUnicode:
    @pytest.mark.parametrize("in_,out", [
        ("", False),
        ("a", False),
        ("0", True),
        ("\u2460", False),  # CIRCLED DIGIT 1
        ("\xbc", False),  # VULGAR FRACTION ONE QUARTER
        ("\u0660", True),  # ARABIC_INDIC DIGIT ZERO
        ("012345", True),
        ("012345a", False),
        (["0", "a"], [True, False]),
    ])
    def test_isdecimal_unicode(self, in_, out, dt):
        buf = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isdecimal(buf), out)

    @pytest.mark.parametrize("in_,out", [
        ("", False),
        ("a", False),
        ("0", True),
        ("\u2460", True),  # CIRCLED DIGIT 1
        ("\xbc", True),  # VULGAR FRACTION ONE QUARTER
        ("\u0660", True),  # ARABIC_INDIC DIGIT ZERO
        ("012345", True),
        ("012345a", False),
        (["0", "a"], [True, False]),
    ])
    def test_isnumeric_unicode(self, in_, out, dt):
        buf = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isnumeric(buf), out)

    @pytest.mark.parametrize("buf,old,new,count,res", [
        ("...\u043c......<", "<", "&lt;", -1, "...\u043c......&lt;"),
        ("Ae¢☃€ 😊" * 2, "A", "B", -1, "Be¢☃€ 😊Be¢☃€ 😊"),
        ("Ae¢☃€ 😊" * 2, "😊", "B", -1, "Ae¢☃€ BAe¢☃€ B"),
    ])
    def test_replace_unicode(self, buf, old, new, count, res, dt):
        buf = np.array(buf, dtype=dt)
        old = np.array(old, dtype=dt)
        new = np.array(new, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.replace(buf, old, new, count), res)

    @pytest.mark.parametrize("in_", [
        '\U00010401',
        '\U00010427',
        '\U00010429',
        '\U0001044E',
        '\U0001D7F6',
        '\U00011066',
        '\U000104A0',
        pytest.param('\U0001F107', marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISALNUM",
            strict=True)),
    ])
    def test_isalnum_unicode(self, in_, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isalnum(in_), True)

    @pytest.mark.parametrize("in_,out", [
        ('\u1FFc', False),
        ('\u2167', False),
        ('\U00010401', False),
        ('\U00010427', False),
        ('\U0001F40D', False),
        ('\U0001F46F', False),
        ('\u2177', True),
        pytest.param('\U00010429', True, marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISLOWER",
            strict=True)),
        ('\U0001044E', True),
    ])
    def test_islower_unicode(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.islower(in_), out)

    @pytest.mark.parametrize("in_,out", [
        ('\u1FFc', False),
        ('\u2167', True),
        ('\U00010401', True),
        ('\U00010427', True),
        ('\U0001F40D', False),
        ('\U0001F46F', False),
        ('\u2177', False),
        pytest.param('\U00010429', False, marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISUPPER",
            strict=True)),
        ('\U0001044E', False),
    ])
    def test_isupper_unicode(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isupper(in_), out)

    @pytest.mark.parametrize("in_,out", [
        ('\u1FFc', True),
        ('Greek \u1FFcitlecases ...', True),
        pytest.param('\U00010401\U00010429', True, marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISISTITLE",
            strict=True)),
        ('\U00010427\U0001044E', True),
        pytest.param('\U00010429', False, marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISISTITLE",
            strict=True)),
        ('\U0001044E', False),
        ('\U0001F40D', False),
        ('\U0001F46F', False),
    ])
    def test_istitle_unicode(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.istitle(in_), out)

    @pytest.mark.parametrize("buf,sub,start,end,res", [
        ("Ae¢☃€ 😊" * 2, "😊", 0, None, 6),
        ("Ae¢☃€ 😊" * 2, "😊", 7, None, 13),
    ])
    def test_index_unicode(self, buf, sub, start, end, res, dt):
        buf = np.array(buf, dtype=dt)
        sub = np.array(sub, dtype=dt)
        assert_array_equal(np.strings.index(buf, sub, start, end), res)

    def test_index_raises_unicode(self, dt):
        with pytest.raises(ValueError, match="substring not found"):
            np.strings.index("Ae¢☃€ 😊", "😀")

    @pytest.mark.parametrize("buf,res", [
        ("Ae¢☃€ \t 😊", "Ae¢☃€    😊"),
        ("\t\U0001044E", "        \U0001044E"),
    ])
    def test_expandtabs(self, buf, res, dt):
        buf = np.array(buf, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.expandtabs(buf), res)

    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('x', 2, '\U0001044E', 'x\U0001044E'),
        ('x', 3, '\U0001044E', '\U0001044Ex\U0001044E'),
        ('x', 4, '\U0001044E', '\U0001044Ex\U0001044E\U0001044E'),
    ])
    def test_center(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.center(buf, width, fillchar), res)

    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('x', 2, '\U0001044E', 'x\U0001044E'),
        ('x', 3, '\U0001044E', 'x\U0001044E\U0001044E'),
        ('x', 4, '\U0001044E', 'x\U0001044E\U0001044E\U0001044E'),
    ])
    def test_ljust(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.ljust(buf, width, fillchar), res)

    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('x', 2, '\U0001044E', '\U0001044Ex'),
        ('x', 3, '\U0001044E', '\U0001044E\U0001044Ex'),
        ('x', 4, '\U0001044E', '\U0001044E\U0001044E\U0001044Ex'),
    ])
    def test_rjust(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.rjust(buf, width, fillchar), res)

    @pytest.mark.parametrize("buf,sep,res1,res2,res3", [
        ("āāāāĀĀĀĀ", "Ă", "āāāāĀĀĀĀ", "", ""),
        ("āāāāĂĀĀĀĀ", "Ă", "āāāā", "Ă", "ĀĀĀĀ"),
        ("āāāāĂĂĀĀĀĀ", "ĂĂ", "āāāā", "ĂĂ", "ĀĀĀĀ"),
        ("𐌁𐌁𐌁𐌁𐌀𐌀𐌀𐌀", "𐌂", "𐌁𐌁𐌁𐌁𐌀𐌀𐌀𐌀", "", ""),
        ("𐌁𐌁𐌁𐌁𐌂𐌀𐌀𐌀𐌀", "𐌂", "𐌁𐌁𐌁𐌁", "𐌂", "𐌀𐌀𐌀𐌀"),
        ("𐌁𐌁𐌁𐌁𐌂𐌂𐌀𐌀𐌀𐌀", "𐌂𐌂", "𐌁𐌁𐌁𐌁", "𐌂𐌂", "𐌀𐌀𐌀𐌀"),
        ("𐌁𐌁𐌁𐌁𐌂𐌂𐌂𐌂𐌀𐌀𐌀𐌀", "𐌂𐌂𐌂𐌂", "𐌁𐌁𐌁𐌁", "𐌂𐌂𐌂𐌂", "𐌀𐌀𐌀𐌀"),
    ])
    def test_partition(self, buf, sep, res1, res2, res3, dt):
        buf = np.array(buf, dtype=dt)
        sep = np.array(sep, dtype=dt)
        res1 = np.array(res1, dtype=dt)
        res2 = np.array(res2, dtype=dt)
        res3 = np.array(res3, dtype=dt)
        act1, act2, act3 = np.strings.partition(buf, sep)
        assert_array_equal(act1, res1)
        assert_array_equal(act2, res2)
        assert_array_equal(act3, res3)
        assert_array_equal(act1 + act2 + act3, buf)

    @pytest.mark.parametrize("buf,sep,res1,res2,res3", [
        ("āāāāĀĀĀĀ", "Ă", "", "", "āāāāĀĀĀĀ"),
        ("āāāāĂĀĀĀĀ", "Ă", "āāāā", "Ă", "ĀĀĀĀ"),
        ("āāāāĂĂĀĀĀĀ", "ĂĂ", "āāāā", "ĂĂ", "ĀĀĀĀ"),
        ("𐌁𐌁𐌁𐌁𐌀𐌀𐌀𐌀", "𐌂", "", "", "𐌁𐌁𐌁𐌁𐌀𐌀𐌀𐌀"),
        ("𐌁𐌁𐌁𐌁𐌂𐌀𐌀𐌀𐌀", "𐌂", "𐌁𐌁𐌁𐌁", "𐌂", "𐌀𐌀𐌀𐌀"),
        ("𐌁𐌁𐌁𐌁𐌂𐌂𐌀𐌀𐌀𐌀", "𐌂𐌂", "𐌁𐌁𐌁𐌁", "𐌂𐌂", "𐌀𐌀𐌀𐌀"),
    ])
    def test_rpartition(self, buf, sep, res1, res2, res3, dt):
        buf = np.array(buf, dtype=dt)
        sep = np.array(sep, dtype=dt)
        res1 = np.array(res1, dtype=dt)
        res2 = np.array(res2, dtype=dt)
        res3 = np.array(res3, dtype=dt)
        act1, act2, act3 = np.strings.rpartition(buf, sep)
        assert_array_equal(act1, res1)
        assert_array_equal(act2, res2)
        assert_array_equal(act3, res3)
        assert_array_equal(act1 + act2 + act3, buf)

    @pytest.mark.parametrize("method", ["strip", "lstrip", "rstrip"])
    @pytest.mark.parametrize(
        "source,strip",
        [
            ("λμ", "μ"),
            ("λμ", "λ"),
            ("λ" * 5 + "μ" * 2, "μ"),
            ("λ" * 5 + "μ" * 2, "λ"),
            ("λ" * 5 + "A" + "μ" * 2, "μλ"),
            ("λμ" * 5, "μ"),
            ("λμ" * 5, "λ"),
    ])
    def test_strip_functions_unicode(self, source, strip, method, dt):
        src_array = np.array([source], dtype=dt)

        npy_func = getattr(np.strings, method)
        py_func = getattr(str, method)

        expected = np.array([py_func(source, strip)], dtype=dt)
        actual = npy_func(src_array, strip)

        assert_array_equal(actual, expected)

    @pytest.mark.parametrize("args", [
        (None,),
        (0,),
        (1,),
        (5,),
        (15,),
        (22,),
        (-1,),
        (-3,),
        ([3, 4],),
        ([-5, 5],),
        ([0, -8],),
        (1, 12),
        (-12, 15),
        (None, -1),
        (0, [17, 6]),
        ([1, 2], [-1, -2]),
        (1, 11, 2),
        (None, None, -1),
        ([0, 10], [-1, 0], [2, -1]),
    ])
    def test_slice(self, args, dt):
        buf = np.array(["Приве́т नमस्ते שָׁלוֹם", "😀😃😄😁😆😅🤣😂🙂🙃"],
                       dtype=dt)
        act = np.strings.slice(buf, *args)
        bcast_args = tuple(np.broadcast_to(arg, buf.shape) for arg in args)
        res = np.array([s[slice(*arg)]
                        for s, arg in zip(buf, zip(*bcast_args))],
                       dtype=dt)
        assert_array_equal(act, res)


class TestMixedTypeMethods:
    def test_center(self):
        buf = np.array("😊", dtype="U")
        fill = np.array("*", dtype="S")
        res = np.array("*😊*", dtype="U")
        assert_array_equal(np.strings.center(buf, 3, fill), res)

        buf = np.array("s", dtype="S")
        fill = np.array("*", dtype="U")
        res = np.array("*s*", dtype="S")
        assert_array_equal(np.strings.center(buf, 3, fill), res)

        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            buf = np.array("s", dtype="S")
            fill = np.array("😊", dtype="U")
            np.strings.center(buf, 3, fill)

    def test_ljust(self):
        buf = np.array("😊", dtype="U")
        fill = np.array("*", dtype="S")
        res = np.array("😊**", dtype="U")
        assert_array_equal(np.strings.ljust(buf, 3, fill), res)

        buf = np.array("s", dtype="S")
        fill = np.array("*", dtype="U")
        res = np.array("s**", dtype="S")
        assert_array_equal(np.strings.ljust(buf, 3, fill), res)

        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            buf = np.array("s", dtype="S")
            fill = np.array("😊", dtype="U")
            np.strings.ljust(buf, 3, fill)

    def test_rjust(self):
        buf = np.array("😊", dtype="U")
        fill = np.array("*", dtype="S")
        res = np.array("**😊", dtype="U")
        assert_array_equal(np.strings.rjust(buf, 3, fill), res)

        buf = np.array("s", dtype="S")
        fill = np.array("*", dtype="U")
        res = np.array("**s", dtype="S")
        assert_array_equal(np.strings.rjust(buf, 3, fill), res)

        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            buf = np.array("s", dtype="S")
            fill = np.array("😊", dtype="U")
            np.strings.rjust(buf, 3, fill)


class TestUnicodeOnlyMethodsRaiseWithBytes:
    def test_isdecimal_raises(self):
        in_ = np.array(b"1")
        with assert_raises(TypeError):
            np.strings.isdecimal(in_)

    def test_isnumeric_bytes(self):
        in_ = np.array(b"1")
        with assert_raises(TypeError):
            np.strings.isnumeric(in_)


def check_itemsize(n_elem, dt):
    if dt == "T":
        return np.dtype(dt).itemsize
    if dt == "S":
        return n_elem
    if dt == "U":
        return n_elem * 4

@pytest.mark.parametrize("dt", ["S", "U", "T"])
class TestReplaceOnArrays:

    def test_replace_count_and_size(self, dt):
        a = np.array(["0123456789" * i for i in range(4)], dtype=dt)
        r1 = np.strings.replace(a, "5", "ABCDE")
        assert r1.dtype.itemsize == check_itemsize(3 * 10 + 3 * 4, dt)
        r1_res = np.array(["01234ABCDE6789" * i for i in range(4)], dtype=dt)
        assert_array_equal(r1, r1_res)
        r2 = np.strings.replace(a, "5", "ABCDE", 1)
        assert r2.dtype.itemsize == check_itemsize(3 * 10 + 4, dt)
        r3 = np.strings.replace(a, "5", "ABCDE", 0)
        assert r3.dtype.itemsize == a.dtype.itemsize
        assert_array_equal(r3, a)
        # Negative values mean to replace all.
        r4 = np.strings.replace(a, "5", "ABCDE", -1)
        assert r4.dtype.itemsize == check_itemsize(3 * 10 + 3 * 4, dt)
        assert_array_equal(r4, r1)
        # We can do count on an element-by-element basis.
        r5 = np.strings.replace(a, "5", "ABCDE", [-1, -1, -1, 1])
        assert r5.dtype.itemsize == check_itemsize(3 * 10 + 4, dt)
        assert_array_equal(r5, np.array(
            ["01234ABCDE6789" * i for i in range(3)]
            + ["01234ABCDE6789" + "0123456789" * 2], dtype=dt))

    def test_replace_broadcasting(self, dt):
        a = np.array("0,0,0", dtype=dt)
        r1 = np.strings.replace(a, "0", "1", np.arange(3))
        assert r1.dtype == a.dtype
        assert_array_equal(r1, np.array(["0,0,0", "1,0,0", "1,1,0"], dtype=dt))
        r2 = np.strings.replace(a, "0", [["1"], ["2"]], np.arange(1, 4))
        assert_array_equal(r2, np.array([["1,0,0", "1,1,0", "1,1,1"],
                                         ["2,0,0", "2,2,0", "2,2,2"]],
                                        dtype=dt))
        r3 = np.strings.replace(a, ["0", "0,0", "0,0,0"], "X")
        assert_array_equal(r3, np.array(["X,X,X", "X,0", "X"], dtype=dt))


class TestOverride:
    @classmethod
    def setup_class(cls):
        class Override:

            def __array_function__(self, *args, **kwargs):
                return "function"

            def __array_ufunc__(self, *args, **kwargs):
                return "ufunc"

        cls.override = Override()

    @pytest.mark.parametrize("func, kwargs", [
        (np.strings.center, dict(width=10)),
        (np.strings.capitalize, {}),
        (np.strings.decode, {}),
        (np.strings.encode, {}),
        (np.strings.expandtabs, {}),
        (np.strings.ljust, dict(width=10)),
        (np.strings.lower, {}),
        (np.strings.mod, dict(values=2)),
        (np.strings.multiply, dict(i=2)),
        (np.strings.partition, dict(sep="foo")),
        (np.strings.rjust, dict(width=10)),
        (np.strings.rpartition, dict(sep="foo")),
        (np.strings.swapcase, {}),
        (np.strings.title, {}),
        (np.strings.translate, dict(table=None)),
        (np.strings.upper, {}),
        (np.strings.zfill, dict(width=10)),
    ])
    def test_override_function(self, func, kwargs):
        assert func(self.override, **kwargs) == "function"

    @pytest.mark.parametrize("func, args, kwargs", [
        (np.strings.add, (None, ), {}),
        (np.strings.lstrip, (), {}),
        (np.strings.rstrip, (), {}),
        (np.strings.strip, (), {}),
        (np.strings.equal, (None, ), {}),
        (np.strings.not_equal, (None, ), {}),
        (np.strings.greater_equal, (None, ), {}),
        (np.strings.less_equal, (None, ), {}),
        (np.strings.greater, (None, ), {}),
        (np.strings.less, (None, ), {}),
        (np.strings.count, ("foo", ), {}),
        (np.strings.endswith, ("foo", ), {}),
        (np.strings.find, ("foo", ), {}),
        (np.strings.index, ("foo", ), {}),
        (np.strings.isalnum, (), {}),
        (np.strings.isalpha, (), {}),
        (np.strings.isdecimal, (), {}),
        (np.strings.isdigit, (), {}),
        (np.strings.islower, (), {}),
        (np.strings.isnumeric, (), {}),
        (np.strings.isspace, (), {}),
        (np.strings.istitle, (), {}),
        (np.strings.isupper, (), {}),
        (np.strings.rfind, ("foo", ), {}),
        (np.strings.rindex, ("foo", ), {}),
        (np.strings.startswith, ("foo", ), {}),
        (np.strings.str_len, (), {}),
    ])
    def test_override_ufunc(self, func, args, kwargs):
        assert func(self.override, *args, **kwargs) == "ufunc"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_business_hour.py ===
"""
Tests for offsets.BusinessHour
"""
from __future__ import annotations

from datetime import (
    datetime,
    time as dt_time,
)

import pytest

from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
)
from pandas._libs.tslibs.offsets import (
    BDay,
    BusinessHour,
    Nano,
)

from pandas import (
    DatetimeIndex,
    _testing as tm,
    date_range,
)
from pandas.tests.tseries.offsets.common import assert_offset_equal


@pytest.fixture
def dt():
    return datetime(2014, 7, 1, 10, 00)


@pytest.fixture
def _offset():
    return BusinessHour


@pytest.fixture
def offset1():
    return BusinessHour()


@pytest.fixture
def offset2():
    return BusinessHour(n=3)


@pytest.fixture
def offset3():
    return BusinessHour(n=-1)


@pytest.fixture
def offset4():
    return BusinessHour(n=-4)


@pytest.fixture
def offset5():
    return BusinessHour(start=dt_time(11, 0), end=dt_time(14, 30))


@pytest.fixture
def offset6():
    return BusinessHour(start="20:00", end="05:00")


@pytest.fixture
def offset7():
    return BusinessHour(n=-2, start=dt_time(21, 30), end=dt_time(6, 30))


@pytest.fixture
def offset8():
    return BusinessHour(start=["09:00", "13:00"], end=["12:00", "17:00"])


@pytest.fixture
def offset9():
    return BusinessHour(n=3, start=["09:00", "22:00"], end=["13:00", "03:00"])


@pytest.fixture
def offset10():
    return BusinessHour(n=-1, start=["23:00", "13:00"], end=["02:00", "17:00"])


class TestBusinessHour:
    @pytest.mark.parametrize(
        "start,end,match",
        [
            (
                dt_time(11, 0, 5),
                "17:00",
                "time data must be specified only with hour and minute",
            ),
            ("AAA", "17:00", "time data must match '%H:%M' format"),
            ("14:00:05", "17:00", "time data must match '%H:%M' format"),
            ([], "17:00", "Must include at least 1 start time"),
            ("09:00", [], "Must include at least 1 end time"),
            (
                ["09:00", "11:00"],
                "17:00",
                "number of starting time and ending time must be the same",
            ),
            (
                ["09:00", "11:00"],
                ["10:00"],
                "number of starting time and ending time must be the same",
            ),
            (
                ["09:00", "11:00"],
                ["12:00", "20:00"],
                r"invalid starting and ending time\(s\): opening hours should not "
                "touch or overlap with one another",
            ),
            (
                ["12:00", "20:00"],
                ["09:00", "11:00"],
                r"invalid starting and ending time\(s\): opening hours should not "
                "touch or overlap with one another",
            ),
        ],
    )
    def test_constructor_errors(self, start, end, match):
        with pytest.raises(ValueError, match=match):
            BusinessHour(start=start, end=end)

    def test_different_normalize_equals(self, _offset):
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    def test_repr(
        self,
        offset1,
        offset2,
        offset3,
        offset4,
        offset5,
        offset6,
        offset7,
        offset8,
        offset9,
        offset10,
    ):
        assert repr(offset1) == "<BusinessHour: bh=09:00-17:00>"
        assert repr(offset2) == "<3 * BusinessHours: bh=09:00-17:00>"
        assert repr(offset3) == "<-1 * BusinessHour: bh=09:00-17:00>"
        assert repr(offset4) == "<-4 * BusinessHours: bh=09:00-17:00>"

        assert repr(offset5) == "<BusinessHour: bh=11:00-14:30>"
        assert repr(offset6) == "<BusinessHour: bh=20:00-05:00>"
        assert repr(offset7) == "<-2 * BusinessHours: bh=21:30-06:30>"
        assert repr(offset8) == "<BusinessHour: bh=09:00-12:00,13:00-17:00>"
        assert repr(offset9) == "<3 * BusinessHours: bh=09:00-13:00,22:00-03:00>"
        assert repr(offset10) == "<-1 * BusinessHour: bh=13:00-17:00,23:00-02:00>"

    def test_with_offset(self, dt):
        expected = Timestamp("2014-07-01 13:00")

        assert dt + BusinessHour() * 3 == expected
        assert dt + BusinessHour(n=3) == expected

    @pytest.mark.parametrize(
        "offset_name",
        ["offset1", "offset2", "offset3", "offset4", "offset8", "offset9", "offset10"],
    )
    def test_eq_attribute(self, offset_name, request):
        offset = request.getfixturevalue(offset_name)
        assert offset == offset

    @pytest.mark.parametrize(
        "offset1,offset2",
        [
            (BusinessHour(start="09:00"), BusinessHour()),
            (
                BusinessHour(start=["23:00", "13:00"], end=["12:00", "17:00"]),
                BusinessHour(start=["13:00", "23:00"], end=["17:00", "12:00"]),
            ),
        ],
    )
    def test_eq(self, offset1, offset2):
        assert offset1 == offset2

    @pytest.mark.parametrize(
        "offset1,offset2",
        [
            (BusinessHour(), BusinessHour(-1)),
            (BusinessHour(start="09:00"), BusinessHour(start="09:01")),
            (
                BusinessHour(start="09:00", end="17:00"),
                BusinessHour(start="17:00", end="09:01"),
            ),
            (
                BusinessHour(start=["13:00", "23:00"], end=["18:00", "07:00"]),
                BusinessHour(start=["13:00", "23:00"], end=["17:00", "12:00"]),
            ),
        ],
    )
    def test_neq(self, offset1, offset2):
        assert offset1 != offset2

    @pytest.mark.parametrize(
        "offset_name",
        ["offset1", "offset2", "offset3", "offset4", "offset8", "offset9", "offset10"],
    )
    def test_hash(self, offset_name, request):
        offset = request.getfixturevalue(offset_name)
        assert offset == offset

    def test_add_datetime(
        self,
        dt,
        offset1,
        offset2,
        offset3,
        offset4,
        offset8,
        offset9,
        offset10,
    ):
        assert offset1 + dt == datetime(2014, 7, 1, 11)
        assert offset2 + dt == datetime(2014, 7, 1, 13)
        assert offset3 + dt == datetime(2014, 6, 30, 17)
        assert offset4 + dt == datetime(2014, 6, 30, 14)
        assert offset8 + dt == datetime(2014, 7, 1, 11)
        assert offset9 + dt == datetime(2014, 7, 1, 22)
        assert offset10 + dt == datetime(2014, 7, 1, 1)

    def test_sub(self, dt, offset2, _offset):
        off = offset2
        msg = "Cannot subtract datetime from offset"
        with pytest.raises(TypeError, match=msg):
            off - dt
        assert 2 * off - off == off

        assert dt - offset2 == dt + _offset(-3)

    def test_multiply_by_zero(self, dt, offset1, offset2):
        assert dt - 0 * offset1 == dt
        assert dt + 0 * offset1 == dt
        assert dt - 0 * offset2 == dt
        assert dt + 0 * offset2 == dt

    def testRollback1(
        self,
        dt,
        _offset,
        offset1,
        offset2,
        offset3,
        offset4,
        offset5,
        offset6,
        offset7,
        offset8,
        offset9,
        offset10,
    ):
        assert offset1.rollback(dt) == dt
        assert offset2.rollback(dt) == dt
        assert offset3.rollback(dt) == dt
        assert offset4.rollback(dt) == dt
        assert offset5.rollback(dt) == datetime(2014, 6, 30, 14, 30)
        assert offset6.rollback(dt) == datetime(2014, 7, 1, 5, 0)
        assert offset7.rollback(dt) == datetime(2014, 7, 1, 6, 30)
        assert offset8.rollback(dt) == dt
        assert offset9.rollback(dt) == dt
        assert offset10.rollback(dt) == datetime(2014, 7, 1, 2)

        datet = datetime(2014, 7, 1, 0)
        assert offset1.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset2.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset3.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset4.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset5.rollback(datet) == datetime(2014, 6, 30, 14, 30)
        assert offset6.rollback(datet) == datet
        assert offset7.rollback(datet) == datet
        assert offset8.rollback(datet) == datetime(2014, 6, 30, 17)
        assert offset9.rollback(datet) == datet
        assert offset10.rollback(datet) == datet

        assert _offset(5).rollback(dt) == dt

    def testRollback2(self, _offset):
        assert _offset(-3).rollback(datetime(2014, 7, 5, 15, 0)) == datetime(
            2014, 7, 4, 17, 0
        )

    def testRollforward1(
        self,
        dt,
        _offset,
        offset1,
        offset2,
        offset3,
        offset4,
        offset5,
        offset6,
        offset7,
        offset8,
        offset9,
        offset10,
    ):
        assert offset1.rollforward(dt) == dt
        assert offset2.rollforward(dt) == dt
        assert offset3.rollforward(dt) == dt
        assert offset4.rollforward(dt) == dt
        assert offset5.rollforward(dt) == datetime(2014, 7, 1, 11, 0)
        assert offset6.rollforward(dt) == datetime(2014, 7, 1, 20, 0)
        assert offset7.rollforward(dt) == datetime(2014, 7, 1, 21, 30)
        assert offset8.rollforward(dt) == dt
        assert offset9.rollforward(dt) == dt
        assert offset10.rollforward(dt) == datetime(2014, 7, 1, 13)

        datet = datetime(2014, 7, 1, 0)
        assert offset1.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset2.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset3.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset4.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset5.rollforward(datet) == datetime(2014, 7, 1, 11)
        assert offset6.rollforward(datet) == datet
        assert offset7.rollforward(datet) == datet
        assert offset8.rollforward(datet) == datetime(2014, 7, 1, 9)
        assert offset9.rollforward(datet) == datet
        assert offset10.rollforward(datet) == datet

        assert _offset(5).rollforward(dt) == dt

    def testRollforward2(self, _offset):
        assert _offset(-3).rollforward(datetime(2014, 7, 5, 16, 0)) == datetime(
            2014, 7, 7, 9
        )

    def test_roll_date_object(self):
        offset = BusinessHour()

        dt = datetime(2014, 7, 6, 15, 0)

        result = offset.rollback(dt)
        assert result == datetime(2014, 7, 4, 17)

        result = offset.rollforward(dt)
        assert result == datetime(2014, 7, 7, 9)

    normalize_cases = []
    normalize_cases.append(
        (
            BusinessHour(normalize=True),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 1, 0): datetime(2014, 7, 1),
                datetime(2014, 7, 4, 15): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 15, 59): datetime(2014, 7, 4),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 7),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 7),
            },
        )
    )

    normalize_cases.append(
        (
            BusinessHour(-1, normalize=True),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 6, 30),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 30),
                datetime(2014, 7, 1, 0): datetime(2014, 6, 30),
                datetime(2014, 7, 7, 10): datetime(2014, 7, 4),
                datetime(2014, 7, 7, 10, 1): datetime(2014, 7, 7),
                datetime(2014, 7, 5, 23): datetime(2014, 7, 4),
                datetime(2014, 7, 6, 10): datetime(2014, 7, 4),
            },
        )
    )

    normalize_cases.append(
        (
            BusinessHour(1, normalize=True, start="17:00", end="04:00"),
            {
                datetime(2014, 7, 1, 8): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 2),
                datetime(2014, 7, 2, 3): datetime(2014, 7, 2),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5),
                datetime(2014, 7, 5, 2): datetime(2014, 7, 5),
                datetime(2014, 7, 7, 2): datetime(2014, 7, 7),
                datetime(2014, 7, 7, 17): datetime(2014, 7, 7),
            },
        )
    )

    @pytest.mark.parametrize("case", normalize_cases)
    def test_normalize(self, case):
        offset, cases = case
        for dt, expected in cases.items():
            assert offset._apply(dt) == expected

    on_offset_cases = []
    on_offset_cases.append(
        (
            BusinessHour(),
            {
                datetime(2014, 7, 1, 9): True,
                datetime(2014, 7, 1, 8, 59): False,
                datetime(2014, 7, 1, 8): False,
                datetime(2014, 7, 1, 17): True,
                datetime(2014, 7, 1, 17, 1): False,
                datetime(2014, 7, 1, 18): False,
                datetime(2014, 7, 5, 9): False,
                datetime(2014, 7, 6, 12): False,
            },
        )
    )

    on_offset_cases.append(
        (
            BusinessHour(start="10:00", end="15:00"),
            {
                datetime(2014, 7, 1, 9): False,
                datetime(2014, 7, 1, 10): True,
                datetime(2014, 7, 1, 15): True,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12): False,
                datetime(2014, 7, 6, 12): False,
            },
        )
    )

    on_offset_cases.append(
        (
            BusinessHour(start="19:00", end="05:00"),
            {
                datetime(2014, 7, 1, 9, 0): False,
                datetime(2014, 7, 1, 10, 0): False,
                datetime(2014, 7, 1, 15): False,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12, 0): False,
                datetime(2014, 7, 6, 12, 0): False,
                datetime(2014, 7, 1, 19, 0): True,
                datetime(2014, 7, 2, 0, 0): True,
                datetime(2014, 7, 4, 23): True,
                datetime(2014, 7, 5, 1): True,
                datetime(2014, 7, 5, 5, 0): True,
                datetime(2014, 7, 6, 23, 0): False,
                datetime(2014, 7, 7, 3, 0): False,
            },
        )
    )

    on_offset_cases.append(
        (
            BusinessHour(start=["09:00", "13:00"], end=["12:00", "17:00"]),
            {
                datetime(2014, 7, 1, 9): True,
                datetime(2014, 7, 1, 8, 59): False,
                datetime(2014, 7, 1, 8): False,
                datetime(2014, 7, 1, 17): True,
                datetime(2014, 7, 1, 17, 1): False,
                datetime(2014, 7, 1, 18): False,
                datetime(2014, 7, 5, 9): False,
                datetime(2014, 7, 6, 12): False,
                datetime(2014, 7, 1, 12, 30): False,
            },
        )
    )

    on_offset_cases.append(
        (
            BusinessHour(start=["19:00", "23:00"], end=["21:00", "05:00"]),
            {
                datetime(2014, 7, 1, 9, 0): False,
                datetime(2014, 7, 1, 10, 0): False,
                datetime(2014, 7, 1, 15): False,
                datetime(2014, 7, 1, 15, 1): False,
                datetime(2014, 7, 5, 12, 0): False,
                datetime(2014, 7, 6, 12, 0): False,
                datetime(2014, 7, 1, 19, 0): True,
                datetime(2014, 7, 2, 0, 0): True,
                datetime(2014, 7, 4, 23): True,
                datetime(2014, 7, 5, 1): True,
                datetime(2014, 7, 5, 5, 0): True,
                datetime(2014, 7, 6, 23, 0): False,
                datetime(2014, 7, 7, 3, 0): False,
                datetime(2014, 7, 4, 22): False,
            },
        )
    )

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, cases = case
        for dt, expected in cases.items():
            assert offset.is_on_offset(dt) == expected

    apply_cases = [
        (
            BusinessHour(),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 2, 9, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 12),
                # out of business hours
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 10),
                # saturday
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 9, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 9, 30, 30),
            },
        ),
        (
            BusinessHour(4),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 2, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 13),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 12, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 12, 30, 30),
            },
        ),
        (
            BusinessHour(-1),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 10),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 30, 17),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 1, 15, 30, 15),
                datetime(2014, 7, 1, 9, 30, 15): datetime(2014, 6, 30, 16, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 5): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 10),
                # out of business hours
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 16),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 2, 16),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 16),
                # saturday
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 16),
                datetime(2014, 7, 7, 9): datetime(2014, 7, 4, 16),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 4, 16, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 4, 16, 30, 30),
            },
        ),
        (
            BusinessHour(-4),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 30, 15),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 30, 17),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 11),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 4, 18): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 4, 13, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 4, 13, 30, 30),
            },
        ),
        (
            BusinessHour(start="13:00", end="16:00"),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 1, 15, 30, 15): datetime(2014, 7, 2, 13, 30, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 14),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 14),
            },
        ),
        (
            BusinessHour(n=2, start="13:00", end="16:00"),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 2, 14, 30): datetime(2014, 7, 3, 13, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 15),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 15),
                datetime(2014, 7, 4, 14, 30): datetime(2014, 7, 7, 13, 30),
                datetime(2014, 7, 4, 14, 30, 30): datetime(2014, 7, 7, 13, 30, 30),
            },
        ),
        (
            BusinessHour(n=-1, start="13:00", end="16:00"),
            {
                datetime(2014, 7, 2, 11): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 15): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 16): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 13, 30, 15): datetime(2014, 7, 1, 15, 30, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 15),
                datetime(2014, 7, 7, 11): datetime(2014, 7, 4, 15),
            },
        ),
        (
            BusinessHour(n=-3, start="10:00", end="16:00"),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 11),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 13),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 2, 11, 30): datetime(2014, 7, 1, 14, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 13),
                datetime(2014, 7, 4, 10): datetime(2014, 7, 3, 13),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 4, 16): datetime(2014, 7, 4, 13),
                datetime(2014, 7, 4, 12, 30): datetime(2014, 7, 3, 15, 30),
                datetime(2014, 7, 4, 12, 30, 30): datetime(2014, 7, 3, 15, 30, 30),
            },
        ),
        (
            BusinessHour(start="19:00", end="05:00"),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 20),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 20),
                datetime(2014, 7, 2, 4, 30): datetime(2014, 7, 2, 19, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 1),
                datetime(2014, 7, 4, 10): datetime(2014, 7, 4, 20),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 5, 0),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 5, 1),
                datetime(2014, 7, 5, 4): datetime(2014, 7, 7, 19),
                datetime(2014, 7, 5, 4, 30): datetime(2014, 7, 7, 19, 30),
                datetime(2014, 7, 5, 4, 30, 30): datetime(2014, 7, 7, 19, 30, 30),
            },
        ),
        (
            BusinessHour(n=-1, start="19:00", end="05:00"),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 4),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 20): datetime(2014, 7, 2, 5),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 19, 30): datetime(2014, 7, 2, 4, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 23),
                datetime(2014, 7, 3, 6): datetime(2014, 7, 3, 4),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 4, 22),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 4, 23),
                datetime(2014, 7, 5, 4): datetime(2014, 7, 5, 3),
                datetime(2014, 7, 7, 19, 30): datetime(2014, 7, 5, 4, 30),
                datetime(2014, 7, 7, 19, 30, 30): datetime(2014, 7, 5, 4, 30, 30),
            },
        ),
        (
            BusinessHour(n=4, start="00:00", end="23:00"),
            {
                datetime(2014, 7, 3, 22): datetime(2014, 7, 4, 3),
                datetime(2014, 7, 4, 22): datetime(2014, 7, 7, 3),
                datetime(2014, 7, 3, 22, 30): datetime(2014, 7, 4, 3, 30),
                datetime(2014, 7, 3, 22, 20): datetime(2014, 7, 4, 3, 20),
                datetime(2014, 7, 4, 22, 30, 30): datetime(2014, 7, 7, 3, 30, 30),
                datetime(2014, 7, 4, 22, 30, 20): datetime(2014, 7, 7, 3, 30, 20),
            },
        ),
        (
            BusinessHour(n=-4, start="00:00", end="23:00"),
            {
                datetime(2014, 7, 4, 3): datetime(2014, 7, 3, 22),
                datetime(2014, 7, 7, 3): datetime(2014, 7, 4, 22),
                datetime(2014, 7, 4, 3, 30): datetime(2014, 7, 3, 22, 30),
                datetime(2014, 7, 4, 3, 20): datetime(2014, 7, 3, 22, 20),
                datetime(2014, 7, 7, 3, 30, 30): datetime(2014, 7, 4, 22, 30, 30),
                datetime(2014, 7, 7, 3, 30, 20): datetime(2014, 7, 4, 22, 30, 20),
            },
        ),
        (
            BusinessHour(start=["09:00", "14:00"], end=["12:00", "18:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 14),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 1, 19): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 17),
                datetime(2014, 7, 1, 16, 30, 15): datetime(2014, 7, 1, 17, 30, 15),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 14),
                # out of business hours
                datetime(2014, 7, 1, 13): datetime(2014, 7, 1, 15),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 10),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 10),
                # saturday
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 10),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 9),
                datetime(2014, 7, 4, 17, 30): datetime(2014, 7, 7, 9, 30),
                datetime(2014, 7, 4, 17, 30, 30): datetime(2014, 7, 7, 9, 30, 30),
            },
        ),
        (
            BusinessHour(n=4, start=["09:00", "14:00"], end=["12:00", "18:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 1, 17),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 2, 9),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 2, 10),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 2, 11),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 2, 14),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 2, 17),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 15),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 3, 15),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 7, 15),
                datetime(2014, 7, 4, 17): datetime(2014, 7, 7, 14),
                datetime(2014, 7, 4, 16, 30): datetime(2014, 7, 7, 11, 30),
                datetime(2014, 7, 4, 16, 30, 30): datetime(2014, 7, 7, 11, 30, 30),
            },
        ),
        (
            BusinessHour(n=-4, start=["09:00", "14:00"], end=["12:00", "18:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 30, 17),
                datetime(2014, 7, 1, 15): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 1, 10),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 11),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 1, 12),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 12),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 4, 12),
                datetime(2014, 7, 4, 18): datetime(2014, 7, 4, 12),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 4, 14, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 4, 14, 30, 30),
            },
        ),
        (
            BusinessHour(n=-1, start=["19:00", "03:00"], end=["01:00", "05:00"]),
            {
                datetime(2014, 7, 1, 17): datetime(2014, 7, 1, 4),
                datetime(2014, 7, 2, 14): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 13): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 20): datetime(2014, 7, 2, 5),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 2, 4),
                datetime(2014, 7, 2, 4): datetime(2014, 7, 2, 1),
                datetime(2014, 7, 2, 19, 30): datetime(2014, 7, 2, 4, 30),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 2, 23),
                datetime(2014, 7, 3, 6): datetime(2014, 7, 3, 4),
                datetime(2014, 7, 4, 23): datetime(2014, 7, 4, 22),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 4, 23),
                datetime(2014, 7, 5, 4): datetime(2014, 7, 5, 0),
                datetime(2014, 7, 7, 3, 30): datetime(2014, 7, 5, 0, 30),
                datetime(2014, 7, 7, 19, 30): datetime(2014, 7, 7, 4, 30),
                datetime(2014, 7, 7, 19, 30, 30): datetime(2014, 7, 7, 4, 30, 30),
            },
        ),
    ]

    # long business hours (see gh-26381)

    # multiple business hours

    @pytest.mark.parametrize("case", apply_cases)
    def test_apply(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    apply_large_n_cases = [
        (
            # A week later
            BusinessHour(40),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 8, 11),
                datetime(2014, 7, 1, 13): datetime(2014, 7, 8, 13),
                datetime(2014, 7, 1, 15): datetime(2014, 7, 8, 15),
                datetime(2014, 7, 1, 16): datetime(2014, 7, 8, 16),
                datetime(2014, 7, 1, 17): datetime(2014, 7, 9, 9),
                datetime(2014, 7, 2, 11): datetime(2014, 7, 9, 11),
                datetime(2014, 7, 2, 8): datetime(2014, 7, 9, 9),
                datetime(2014, 7, 2, 19): datetime(2014, 7, 10, 9),
                datetime(2014, 7, 2, 23): datetime(2014, 7, 10, 9),
                datetime(2014, 7, 3, 0): datetime(2014, 7, 10, 9),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 14, 9),
                datetime(2014, 7, 4, 18): datetime(2014, 7, 14, 9),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 14, 9, 30),
                datetime(2014, 7, 7, 9, 30, 30): datetime(2014, 7, 14, 9, 30, 30),
            },
        ),
        (
            # 3 days and 1 hour before
            BusinessHour(-25),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 26, 10),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 26, 12),
                datetime(2014, 7, 1, 9): datetime(2014, 6, 25, 16),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 25, 17),
                datetime(2014, 7, 3, 11): datetime(2014, 6, 30, 10),
                datetime(2014, 7, 3, 8): datetime(2014, 6, 27, 16),
                datetime(2014, 7, 3, 19): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 3, 23): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 4, 9): datetime(2014, 6, 30, 16),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 1, 16),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 1, 16, 30),
                datetime(2014, 7, 7, 10, 30, 30): datetime(2014, 7, 2, 9, 30, 30),
            },
        ),
        (
            # 5 days and 3 hours later
            BusinessHour(28, start="21:00", end="02:00"),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 9, 0),
                datetime(2014, 7, 1, 22): datetime(2014, 7, 9, 1),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 9, 21),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 10, 0),
                datetime(2014, 7, 3, 21): datetime(2014, 7, 11, 0),
                datetime(2014, 7, 4, 1): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 2): datetime(2014, 7, 12, 0),
                datetime(2014, 7, 4, 3): datetime(2014, 7, 12, 0),
                datetime(2014, 7, 5, 1): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 15, 0),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 15, 0),
                datetime(2014, 7, 7, 1): datetime(2014, 7, 15, 0),
                datetime(2014, 7, 7, 23, 30): datetime(2014, 7, 15, 21, 30),
            },
        ),
        (
            # large n for multiple opening hours (3 days and 1 hour before)
            BusinessHour(n=-25, start=["09:00", "14:00"], end=["12:00", "19:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 6, 26, 10),
                datetime(2014, 7, 1, 13): datetime(2014, 6, 26, 11),
                datetime(2014, 7, 1, 9): datetime(2014, 6, 25, 18),
                datetime(2014, 7, 1, 10): datetime(2014, 6, 25, 19),
                datetime(2014, 7, 3, 11): datetime(2014, 6, 30, 10),
                datetime(2014, 7, 3, 8): datetime(2014, 6, 27, 18),
                datetime(2014, 7, 3, 19): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 3, 23): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 4, 9): datetime(2014, 6, 30, 18),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 1, 18),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 1, 18),
                datetime(2014, 7, 7, 9, 30): datetime(2014, 7, 1, 18, 30),
                datetime(2014, 7, 7, 10, 30, 30): datetime(2014, 7, 2, 9, 30, 30),
            },
        ),
        (
            # 5 days and 3 hours later
            BusinessHour(28, start=["21:00", "03:00"], end=["01:00", "04:00"]),
            {
                datetime(2014, 7, 1, 11): datetime(2014, 7, 9, 0),
                datetime(2014, 7, 1, 22): datetime(2014, 7, 9, 3),
                datetime(2014, 7, 1, 23): datetime(2014, 7, 9, 21),
                datetime(2014, 7, 2, 2): datetime(2014, 7, 9, 23),
                datetime(2014, 7, 3, 21): datetime(2014, 7, 11, 0),
                datetime(2014, 7, 4, 1): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 2): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 3): datetime(2014, 7, 11, 23),
                datetime(2014, 7, 4, 21): datetime(2014, 7, 12, 0),
                datetime(2014, 7, 5, 0): datetime(2014, 7, 14, 22),
                datetime(2014, 7, 5, 1): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 5, 15): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 6, 18): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 7, 1): datetime(2014, 7, 14, 23),
                datetime(2014, 7, 7, 23, 30): datetime(2014, 7, 15, 21, 30),
            },
        ),
    ]

    @pytest.mark.parametrize("case", apply_large_n_cases)
    def test_apply_large_n(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_apply_nanoseconds(self):
        tests = [
            (
                BusinessHour(),
                {
                    Timestamp("2014-07-04 15:00")
                    + Nano(5): Timestamp("2014-07-04 16:00")
                    + Nano(5),
                    Timestamp("2014-07-04 16:00")
                    + Nano(5): Timestamp("2014-07-07 09:00")
                    + Nano(5),
                    Timestamp("2014-07-04 16:00")
                    - Nano(5): Timestamp("2014-07-04 17:00")
                    - Nano(5),
                },
            ),
            (
                BusinessHour(-1),
                {
                    Timestamp("2014-07-04 15:00")
                    + Nano(5): Timestamp("2014-07-04 14:00")
                    + Nano(5),
                    Timestamp("2014-07-04 10:00")
                    + Nano(5): Timestamp("2014-07-04 09:00")
                    + Nano(5),
                    Timestamp("2014-07-04 10:00")
                    - Nano(5): Timestamp("2014-07-03 17:00")
                    - Nano(5),
                },
            ),
        ]

        for offset, cases in tests:
            for base, expected in cases.items():
                assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("td_unit", ["s", "ms", "us", "ns"])
    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_bday_ignores_timedeltas(self, unit, td_unit):
        # GH#55608
        idx = date_range("2010/02/01", "2010/02/10", freq="12h", unit=unit)
        td = Timedelta(3, unit="h").as_unit(td_unit)
        off = BDay(offset=td)
        t1 = idx + off

        exp_unit = tm.get_finest_unit(td.unit, idx.unit)

        expected = DatetimeIndex(
            [
                "2010-02-02 03:00:00",
                "2010-02-02 15:00:00",
                "2010-02-03 03:00:00",
                "2010-02-03 15:00:00",
                "2010-02-04 03:00:00",
                "2010-02-04 15:00:00",
                "2010-02-05 03:00:00",
                "2010-02-05 15:00:00",
                "2010-02-08 03:00:00",
                "2010-02-08 15:00:00",
                "2010-02-08 03:00:00",
                "2010-02-08 15:00:00",
                "2010-02-08 03:00:00",
                "2010-02-08 15:00:00",
                "2010-02-09 03:00:00",
                "2010-02-09 15:00:00",
                "2010-02-10 03:00:00",
                "2010-02-10 15:00:00",
                "2010-02-11 03:00:00",
            ],
            freq=None,
        ).as_unit(exp_unit)
        tm.assert_index_equal(t1, expected)

        # TODO(GH#55564): as_unit will be unnecessary
        pointwise = DatetimeIndex([x + off for x in idx]).as_unit(exp_unit)
        tm.assert_index_equal(pointwise, expected)

    def test_add_bday_offset_nanos(self):
        # GH#55608
        idx = date_range("2010/02/01", "2010/02/10", freq="12h", unit="ns")
        off = BDay(offset=Timedelta(3, unit="ns"))

        result = idx + off
        expected = DatetimeIndex([x + off for x in idx])
        tm.assert_index_equal(result, expected)


class TestOpeningTimes:
    # opening time should be affected by sign of n, not by n's value and end
    opening_time_cases = [
        (
            [
                BusinessHour(),
                BusinessHour(n=2),
                BusinessHour(n=4),
                BusinessHour(end="10:00"),
                BusinessHour(n=2, end="4:00"),
                BusinessHour(n=4, end="15:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 1, 9),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 1, 9),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 1, 9),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 1, 9),
                ),
                # if timestamp is on opening time, next opening time is
                # as it is
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 2, 10): (
                    datetime(2014, 7, 3, 9),
                    datetime(2014, 7, 2, 9),
                ),
                # 2014-07-05 is saturday
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 4, 9),
                ),
                datetime(2014, 7, 7, 9, 1): (
                    datetime(2014, 7, 8, 9),
                    datetime(2014, 7, 7, 9),
                ),
            },
        ),
        (
            [
                BusinessHour(start="11:15"),
                BusinessHour(n=2, start="11:15"),
                BusinessHour(n=3, start="11:15"),
                BusinessHour(start="11:15", end="10:00"),
                BusinessHour(n=2, start="11:15", end="4:00"),
                BusinessHour(n=3, start="11:15", end="15:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 11, 15),
                    datetime(2014, 6, 30, 11, 15),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 2, 10): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 11, 15),
                ),
                datetime(2014, 7, 2, 11, 15): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 2, 11, 15),
                ),
                datetime(2014, 7, 2, 11, 15, 1): (
                    datetime(2014, 7, 3, 11, 15),
                    datetime(2014, 7, 2, 11, 15),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 11, 15),
                    datetime(2014, 7, 3, 11, 15),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
                datetime(2014, 7, 7, 9, 1): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 11, 15),
                ),
            },
        ),
        (
            [
                BusinessHour(-1),
                BusinessHour(n=-2),
                BusinessHour(n=-4),
                BusinessHour(n=-1, end="10:00"),
                BusinessHour(n=-2, end="4:00"),
                BusinessHour(n=-4, end="15:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 1, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 1, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 1, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 2, 9),
                ),
                datetime(2014, 7, 2, 10): (
                    datetime(2014, 7, 2, 9),
                    datetime(2014, 7, 3, 9),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 4, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 7, 9): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 7, 9),
                ),
                datetime(2014, 7, 7, 9, 1): (
                    datetime(2014, 7, 7, 9),
                    datetime(2014, 7, 8, 9),
                ),
            },
        ),
        (
            [
                BusinessHour(start="17:00", end="05:00"),
                BusinessHour(n=3, start="17:00", end="03:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 6, 30, 17),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 2, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 2, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 4, 17): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 3, 17),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 7, 17, 1): (
                    datetime(2014, 7, 8, 17),
                    datetime(2014, 7, 7, 17),
                ),
            },
        ),
        (
            [
                BusinessHour(-1, start="17:00", end="05:00"),
                BusinessHour(n=-2, start="17:00", end="03:00"),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 6, 30, 17),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 2, 16, 59): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 17),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 3, 17),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 17),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 17),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 17),
                ),
                datetime(2014, 7, 7, 18): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 8, 17),
                ),
            },
        ),
        (
            [
                BusinessHour(start=["11:15", "15:00"], end=["13:00", "20:00"]),
                BusinessHour(n=3, start=["11:15", "15:00"], end=["12:00", "20:00"]),
                BusinessHour(start=["11:15", "15:00"], end=["13:00", "17:00"]),
                BusinessHour(n=2, start=["11:15", "15:00"], end=["12:00", "03:00"]),
                BusinessHour(n=3, start=["11:15", "15:00"], end=["13:00", "16:00"]),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 11, 15),
                    datetime(2014, 6, 30, 15),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 2, 10): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 1, 15),
                ),
                datetime(2014, 7, 2, 11, 15): (
                    datetime(2014, 7, 2, 11, 15),
                    datetime(2014, 7, 2, 11, 15),
                ),
                datetime(2014, 7, 2, 11, 15, 1): (
                    datetime(2014, 7, 2, 15),
                    datetime(2014, 7, 2, 11, 15),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 11, 15),
                    datetime(2014, 7, 3, 15),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 7, 9, 1): (
                    datetime(2014, 7, 7, 11, 15),
                    datetime(2014, 7, 4, 15),
                ),
                datetime(2014, 7, 7, 12): (
                    datetime(2014, 7, 7, 15),
                    datetime(2014, 7, 7, 11, 15),
                ),
            },
        ),
        (
            [
                BusinessHour(n=-1, start=["17:00", "08:00"], end=["05:00", "10:00"]),
                BusinessHour(n=-2, start=["08:00", "17:00"], end=["10:00", "03:00"]),
            ],
            {
                datetime(2014, 7, 1, 11): (
                    datetime(2014, 7, 1, 8),
                    datetime(2014, 7, 1, 17),
                ),
                datetime(2014, 7, 1, 18): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 8),
                ),
                datetime(2014, 7, 1, 23): (
                    datetime(2014, 7, 1, 17),
                    datetime(2014, 7, 2, 8),
                ),
                datetime(2014, 7, 2, 8): (
                    datetime(2014, 7, 2, 8),
                    datetime(2014, 7, 2, 8),
                ),
                datetime(2014, 7, 2, 9): (
                    datetime(2014, 7, 2, 8),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 2, 16, 59): (
                    datetime(2014, 7, 2, 8),
                    datetime(2014, 7, 2, 17),
                ),
                datetime(2014, 7, 5, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 8),
                ),
                datetime(2014, 7, 4, 10): (
                    datetime(2014, 7, 4, 8),
                    datetime(2014, 7, 4, 17),
                ),
                datetime(2014, 7, 4, 23): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 8),
                ),
                datetime(2014, 7, 6, 10): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 8),
                ),
                datetime(2014, 7, 7, 5): (
                    datetime(2014, 7, 4, 17),
                    datetime(2014, 7, 7, 8),
                ),
                datetime(2014, 7, 7, 18): (
                    datetime(2014, 7, 7, 17),
                    datetime(2014, 7, 8, 8),
                ),
            },
        ),
    ]

    @pytest.mark.parametrize("case", opening_time_cases)
    def test_opening_time(self, case):
        _offsets, cases = case
        for offset in _offsets:
            for dt, (exp_next, exp_prev) in cases.items():
                assert offset._next_opening_time(dt) == exp_next
                assert offset._prev_opening_time(dt) == exp_prev

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_limits.py ===
from itertools import product

from sympy.concrete.summations import Sum
from sympy.core.function import (Function, diff)
from sympy.core import EulerGamma, GoldenRatio
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
from sympy.functions.elementary.complexes import (Abs, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, asinh, acosh, acoth, acsch, asech, tanh, sinh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin,
                                                      atan, cos, cot, csc, sec, sin, tan)
from sympy.functions.special.bessel import (besseli, bessely, besselj, besselk)
from sympy.functions.special.error_functions import (Ei, erf, erfc, erfi, fresnelc, fresnels)
from sympy.functions.special.gamma_functions import (digamma, gamma, uppergamma)
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import (Integral, integrate)
from sympy.series.limits import (Limit, limit)
from sympy.simplify.simplify import (logcombine, simplify)
from sympy.simplify.hyperexpand import hyperexpand

from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.mul import Mul
from sympy.series.limits import heuristics
from sympy.series.order import Order
from sympy.testing.pytest import XFAIL, raises

from sympy import elliptic_e, elliptic_k

from sympy.abc import x, y, z, k
n = Symbol('n', integer=True, positive=True)


def test_basic1():
    assert limit(x, x, oo) is oo
    assert limit(x, x, -oo) is -oo
    assert limit(-x, x, oo) is -oo
    assert limit(x**2, x, -oo) is oo
    assert limit(-x**2, x, oo) is -oo
    assert limit(x*log(x), x, 0, dir="+") == 0
    assert limit(1/x, x, oo) == 0
    assert limit(exp(x), x, oo) is oo
    assert limit(-exp(x), x, oo) is -oo
    assert limit(exp(x)/x, x, oo) is oo
    assert limit(1/x - exp(-x), x, oo) == 0
    assert limit(x + 1/x, x, oo) is oo
    assert limit(x - x**2, x, oo) is -oo
    assert limit((1 + x)**(1 + sqrt(2)), x, 0) == 1
    assert limit((1 + x)**oo, x, 0) == Limit((x + 1)**oo, x, 0)
    assert limit((1 + x)**oo, x, 0, dir='-') == Limit((x + 1)**oo, x, 0, dir='-')
    assert limit((1 + x + y)**oo, x, 0, dir='-') == Limit((1 + x + y)**oo, x, 0, dir='-')
    assert limit(y/x/log(x), x, 0) == -oo*sign(y)
    assert limit(cos(x + y)/x, x, 0) == sign(cos(y))*oo
    assert limit(gamma(1/x + 3), x, oo) == 2
    assert limit(S.NaN, x, -oo) is S.NaN
    assert limit(Order(2)*x, x, S.NaN) is S.NaN
    assert limit(1/(x - 1), x, 1, dir="+") is oo
    assert limit(1/(x - 1), x, 1, dir="-") is -oo
    assert limit(1/(5 - x)**3, x, 5, dir="+") is -oo
    assert limit(1/(5 - x)**3, x, 5, dir="-") is oo
    assert limit(1/sin(x), x, pi, dir="+") is -oo
    assert limit(1/sin(x), x, pi, dir="-") is oo
    assert limit(1/cos(x), x, pi/2, dir="+") is -oo
    assert limit(1/cos(x), x, pi/2, dir="-") is oo
    assert limit(1/tan(x**3), x, (2*pi)**Rational(1, 3), dir="+") is oo
    assert limit(1/tan(x**3), x, (2*pi)**Rational(1, 3), dir="-") is -oo
    assert limit(1/cot(x)**3, x, (pi*Rational(3, 2)), dir="+") is -oo
    assert limit(1/cot(x)**3, x, (pi*Rational(3, 2)), dir="-") is oo
    assert limit(tan(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(cot(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(sec(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(csc(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)

    # test bi-directional limits
    assert limit(sin(x)/x, x, 0, dir="+-") == 1
    assert limit(x**2, x, 0, dir="+-") == 0
    assert limit(1/x**2, x, 0, dir="+-") is oo

    # test failing bi-directional limits
    assert limit(1/x, x, 0, dir="+-") is zoo
    # approaching 0
    # from dir="+"
    assert limit(1 + 1/x, x, 0) is oo
    # from dir='-'
    # Add
    assert limit(1 + 1/x, x, 0, dir='-') is -oo
    # Pow
    assert limit(x**(-2), x, 0, dir='-') is oo
    assert limit(x**(-3), x, 0, dir='-') is -oo
    assert limit(1/sqrt(x), x, 0, dir='-') == (-oo)*I
    assert limit(x**2, x, 0, dir='-') == 0
    assert limit(sqrt(x), x, 0, dir='-') == 0
    assert limit(x**-pi, x, 0, dir='-') == -oo*(-1)**(1 - pi)
    assert limit((1 + cos(x))**oo, x, 0) == Limit((cos(x) + 1)**oo, x, 0)

    # test pull request 22491
    assert limit(1/asin(x), x, 0, dir = '+') == oo
    assert limit(1/asin(x), x, 0, dir = '-') == -oo
    assert limit(1/sinh(x), x, 0, dir = '+') == oo
    assert limit(1/sinh(x), x, 0, dir = '-') == -oo
    assert limit(log(1/x) + 1/sin(x), x, 0, dir = '+') == oo
    assert limit(log(1/x) + 1/x, x, 0, dir = '+') == oo


def test_basic2():
    assert limit(x**x, x, 0, dir="+") == 1
    assert limit((exp(x) - 1)/x, x, 0) == 1
    assert limit(1 + 1/x, x, oo) == 1
    assert limit(-exp(1/x), x, oo) == -1
    assert limit(x + exp(-x), x, oo) is oo
    assert limit(x + exp(-x**2), x, oo) is oo
    assert limit(x + exp(-exp(x)), x, oo) is oo
    assert limit(13 + 1/x - exp(-x), x, oo) == 13


def test_basic3():
    assert limit(1/x, x, 0, dir="+") is oo
    assert limit(1/x, x, 0, dir="-") is -oo


def test_basic4():
    assert limit(2*x + y*x, x, 0) == 0
    assert limit(2*x + y*x, x, 1) == 2 + y
    assert limit(2*x**8 + y*x**(-3), x, -2) == 512 - y/8
    assert limit(sqrt(x + 1) - sqrt(x), x, oo) == 0
    assert integrate(1/(x**3 + 1), (x, 0, oo)) == 2*pi*sqrt(3)/9


def test_log():
    # https://github.com/sympy/sympy/issues/21598
    a, b, c = symbols('a b c', positive=True)
    A = log(a/b) - (log(a) - log(b))
    assert A.limit(a, oo) == 0
    assert (A * c).limit(a, oo) == 0

    tau, x = symbols('tau x', positive=True)
    # The value of manualintegrate in the issue
    expr = tau**2*((tau - 1)*(tau + 1)*log(x + 1)/(tau**2 + 1)**2 + 1/((tau**2\
            + 1)*(x + 1)) - (-2*tau*atan(x/tau) + (tau**2/2 - 1/2)*log(tau**2\
            + x**2))/(tau**2 + 1)**2)
    assert limit(expr, x, oo) == pi*tau**3/(tau**2 + 1)**2


def test_piecewise():
    # https://github.com/sympy/sympy/issues/18363
    assert limit((real_root(x - 6, 3) + 2)/(x + 2), x, -2, '+') == Rational(1, 12)


def test_piecewise2():
    func1 = 2*sqrt(x)*Piecewise(((4*x - 2)/Abs(sqrt(4 - 4*(2*x - 1)**2)), 4*x - 2\
            >= 0), ((2 - 4*x)/Abs(sqrt(4 - 4*(2*x - 1)**2)), True))
    func2 = Piecewise((x**2/2, x <= 0.5), (x/2 - 0.125, True))
    func3 = Piecewise(((x - 9) / 5, x < -1), ((x - 9) / 5, x > 4), (sqrt(Abs(x - 3)), True))
    assert limit(func1, x, 0) == 1
    assert limit(func2, x, 0) == 0
    assert limit(func3, x, -1) == 2


def test_basic5():
    class my(Function):
        @classmethod
        def eval(cls, arg):
            if arg is S.Infinity:
                return S.NaN
    assert limit(my(x), x, oo) == Limit(my(x), x, oo)


def test_issue_3885():
    assert limit(x*y + x*z, z, 2) == x*y + 2*x


def test_Limit():
    assert Limit(sin(x)/x, x, 0) != 1
    assert Limit(sin(x)/x, x, 0).doit() == 1
    assert Limit(x, x, 0, dir='+-').args == (x, x, 0, Symbol('+-'))


def test_floor():
    assert limit(floor(x), x, -2, "+") == -2
    assert limit(floor(x), x, -2, "-") == -3
    assert limit(floor(x), x, -1, "+") == -1
    assert limit(floor(x), x, -1, "-") == -2
    assert limit(floor(x), x, 0, "+") == 0
    assert limit(floor(x), x, 0, "-") == -1
    assert limit(floor(x), x, 1, "+") == 1
    assert limit(floor(x), x, 1, "-") == 0
    assert limit(floor(x), x, 2, "+") == 2
    assert limit(floor(x), x, 2, "-") == 1
    assert limit(floor(x), x, 248, "+") == 248
    assert limit(floor(x), x, 248, "-") == 247

    # https://github.com/sympy/sympy/issues/14478
    assert limit(x*floor(3/x)/2, x, 0, '+') == Rational(3, 2)
    assert limit(floor(x + 1/2) - floor(x), x, oo) == AccumBounds(-S.Half, S(3)/2)

    # test issue 9158
    assert limit(floor(atan(x)), x, oo) == 1
    assert limit(floor(atan(x)), x, -oo) == -2
    assert limit(ceiling(atan(x)), x, oo) == 2
    assert limit(ceiling(atan(x)), x, -oo) == -1


def test_floor_requires_robust_assumptions():
    assert limit(floor(sin(x)), x, 0, "+") == 0
    assert limit(floor(sin(x)), x, 0, "-") == -1
    assert limit(floor(cos(x)), x, 0, "+") == 0
    assert limit(floor(cos(x)), x, 0, "-") == 0
    assert limit(floor(5 + sin(x)), x, 0, "+") == 5
    assert limit(floor(5 + sin(x)), x, 0, "-") == 4
    assert limit(floor(5 + cos(x)), x, 0, "+") == 5
    assert limit(floor(5 + cos(x)), x, 0, "-") == 5


def test_ceiling():
    assert limit(ceiling(x), x, -2, "+") == -1
    assert limit(ceiling(x), x, -2, "-") == -2
    assert limit(ceiling(x), x, -1, "+") == 0
    assert limit(ceiling(x), x, -1, "-") == -1
    assert limit(ceiling(x), x, 0, "+") == 1
    assert limit(ceiling(x), x, 0, "-") == 0
    assert limit(ceiling(x), x, 1, "+") == 2
    assert limit(ceiling(x), x, 1, "-") == 1
    assert limit(ceiling(x), x, 2, "+") == 3
    assert limit(ceiling(x), x, 2, "-") == 2
    assert limit(ceiling(x), x, 248, "+") == 249
    assert limit(ceiling(x), x, 248, "-") == 248

    # https://github.com/sympy/sympy/issues/14478
    assert limit(x*ceiling(3/x)/2, x, 0, '+') == Rational(3, 2)
    assert limit(ceiling(x + 1/2) - ceiling(x), x, oo) == AccumBounds(-S.Half, S(3)/2)


def test_ceiling_requires_robust_assumptions():
    assert limit(ceiling(sin(x)), x, 0, "+") == 1
    assert limit(ceiling(sin(x)), x, 0, "-") == 0
    assert limit(ceiling(cos(x)), x, 0, "+") == 1
    assert limit(ceiling(cos(x)), x, 0, "-") == 1
    assert limit(ceiling(5 + sin(x)), x, 0, "+") == 6
    assert limit(ceiling(5 + sin(x)), x, 0, "-") == 5
    assert limit(ceiling(5 + cos(x)), x, 0, "+") == 6
    assert limit(ceiling(5 + cos(x)), x, 0, "-") == 6


def test_frac():
    assert limit(frac(x), x, oo) == AccumBounds(0, 1)
    assert limit(frac(x)**(1/x), x, oo) == AccumBounds(0, 1)
    assert limit(frac(x)**(1/x), x, -oo) == AccumBounds(1, oo)
    assert limit(frac(x)**x, x, oo) == AccumBounds(0, oo)  # wolfram gives (0, 1)
    assert limit(frac(sin(x)), x, 0, "+") == 0
    assert limit(frac(sin(x)), x, 0, "-") == 1
    assert limit(frac(cos(x)), x, 0, "+-") == 1
    assert limit(frac(x**2), x, 0, "+-") == 0
    raises(ValueError, lambda: limit(frac(x), x, 0, '+-'))
    assert limit(frac(-2*x + 1), x, 0, "+") == 1
    assert limit(frac(-2*x + 1), x, 0, "-") == 0
    assert limit(frac(x + S.Half), x, 0, "+-") == S(1)/2
    assert limit(frac(1/x), x, 0) == AccumBounds(0, 1)


def test_issue_14355():
    assert limit(floor(sin(x)/x), x, 0, '+') == 0
    assert limit(floor(sin(x)/x), x, 0, '-') == 0
    # test comment https://github.com/sympy/sympy/issues/14355#issuecomment-372121314
    assert limit(floor(-tan(x)/x), x, 0, '+') == -2
    assert limit(floor(-tan(x)/x), x, 0, '-') == -2


def test_atan():
    x = Symbol("x", real=True)
    assert limit(atan(x)*sin(1/x), x, 0) == 0
    assert limit(atan(x) + sqrt(x + 1) - sqrt(x), x, oo) == pi/2


def test_set_signs():
    assert limit(abs(x), x, 0) == 0
    assert limit(abs(sin(x)), x, 0) == 0
    assert limit(abs(cos(x)), x, 0) == 1
    assert limit(abs(sin(x + 1)), x, 0) == sin(1)

    # https://github.com/sympy/sympy/issues/9449
    assert limit((Abs(x + y) - Abs(x - y))/(2*x), x, 0) == sign(y)

    # https://github.com/sympy/sympy/issues/12398
    assert limit(Abs(log(x)/x**3), x, oo) == 0
    assert limit(x*(Abs(log(x)/x**3)/Abs(log(x + 1)/(x + 1)**3) - 1), x, oo) == 3

    # https://github.com/sympy/sympy/issues/18501
    assert limit(Abs(log(x - 1)**3 - 1), x, 1, '+') == oo

    # https://github.com/sympy/sympy/issues/18997
    assert limit(Abs(log(x)), x, 0) == oo
    assert limit(Abs(log(Abs(x))), x, 0) == oo

    # https://github.com/sympy/sympy/issues/19026
    z = Symbol('z', positive=True)
    assert limit(Abs(log(z) + 1)/log(z), z, oo) == 1

    # https://github.com/sympy/sympy/issues/20704
    assert limit(z*(Abs(1/z + y) - Abs(y - 1/z))/2, z, 0) == 0

    # https://github.com/sympy/sympy/issues/21606
    assert limit(cos(z)/sign(z), z, pi, '-') == -1


def test_heuristic():
    x = Symbol("x", real=True)
    assert heuristics(sin(1/x) + atan(x), x, 0, '+') == AccumBounds(-1, 1)
    assert limit(log(2 + sqrt(atan(x))*sqrt(sin(1/x))), x, 0) == log(2)


def test_issue_3871():
    z = Symbol("z", positive=True)
    f = -1/z*exp(-z*x)
    assert limit(f, x, oo) == 0
    assert f.limit(x, oo) == 0


def test_exponential():
    n = Symbol('n')
    x = Symbol('x', real=True)
    assert limit((1 + x/n)**n, n, oo) == exp(x)
    assert limit((1 + x/(2*n))**n, n, oo) == exp(x/2)
    assert limit((1 + x/(2*n + 1))**n, n, oo) == exp(x/2)
    assert limit(((x - 1)/(x + 1))**x, x, oo) == exp(-2)
    assert limit(1 + (1 + 1/x)**x, x, oo) == 1 + S.Exp1
    assert limit((2 + 6*x)**x/(6*x)**x, x, oo) == exp(S('1/3'))


def test_exponential2():
    n = Symbol('n')
    assert limit((1 + x/(n + sin(n)))**n, n, oo) == exp(x)


def test_doit():
    f = Integral(2 * x, x)
    l = Limit(f, x, oo)
    assert l.doit() is oo


def test_series_AccumBounds():
    assert limit(sin(k) - sin(k + 1), k, oo) == AccumBounds(-2, 2)
    assert limit(cos(k) - cos(k + 1) + 1, k, oo) == AccumBounds(-1, 3)

    # not the exact bound
    assert limit(sin(k) - sin(k)*cos(k), k, oo) == AccumBounds(-2, 2)

    # test for issue #9934
    lo = (-3 + cos(1))/2
    hi = (1 + cos(1))/2
    t1 = Mul(AccumBounds(lo, hi), 1/(-1 + cos(1)), evaluate=False)
    assert limit(simplify(Sum(cos(n).rewrite(exp), (n, 0, k)).doit().rewrite(sin)), k, oo) == t1

    t2 = Mul(AccumBounds(-1 + sin(1)/2, sin(1)/2 + 1), 1/(1 - cos(1)))
    assert limit(simplify(Sum(sin(n).rewrite(exp), (n, 0, k)).doit().rewrite(sin)), k, oo) == t2

    assert limit(((sin(x) + 1)/2)**x, x, oo) == AccumBounds(0, oo)  # wolfram says 0

    # https://github.com/sympy/sympy/issues/12312
    e = 2**(-x)*(sin(x) + 1)**x
    assert limit(e, x, oo) == AccumBounds(0, oo)


def test_bessel_functions_at_infinity():
    # Pull Request 23844 implements limits for all bessel and modified bessel
    # functions approaching infinity along any direction i.e. abs(z0) tends to oo

    assert limit(besselj(1, x), x, oo) == 0
    assert limit(besselj(1, x), x, -oo) == 0
    assert limit(besselj(1, x), x, I*oo) == oo*I
    assert limit(besselj(1, x), x, -I*oo) == -oo*I
    assert limit(bessely(1, x), x, oo) == 0
    assert limit(bessely(1, x), x, -oo) == 0
    assert limit(bessely(1, x), x, I*oo) == -oo
    assert limit(bessely(1, x), x, -I*oo) == -oo
    assert limit(besseli(1, x), x, oo) == oo
    assert limit(besseli(1, x), x, -oo) == -oo
    assert limit(besseli(1, x), x, I*oo) == 0
    assert limit(besseli(1, x), x, -I*oo) == 0
    assert limit(besselk(1, x), x, oo) == 0
    assert limit(besselk(1, x), x, -oo) == -oo*I
    assert limit(besselk(1, x), x, I*oo) == 0
    assert limit(besselk(1, x), x, -I*oo) == 0

    # test issue 14874
    assert limit(besselk(0, x), x, oo) == 0


@XFAIL
def test_doit2():
    f = Integral(2 * x, x)
    l = Limit(f, x, oo)
    # limit() breaks on the contained Integral.
    assert l.doit(deep=False) == l


def test_issue_2929():
    assert limit((x * exp(x))/(exp(x) - 1), x, -oo) == 0


def test_issue_3792():
    assert limit((1 - cos(x))/x**2, x, S.Half) == 4 - 4*cos(S.Half)
    assert limit(sin(sin(x + 1) + 1), x, 0) == sin(1 + sin(1))
    assert limit(abs(sin(x + 1) + 1), x, 0) == 1 + sin(1)


def test_issue_4090():
    assert limit(1/(x + 3), x, 2) == Rational(1, 5)
    assert limit(1/(x + pi), x, 2) == S.One/(2 + pi)
    assert limit(log(x)/(x**2 + 3), x, 2) == log(2)/7
    assert limit(log(x)/(x**2 + pi), x, 2) == log(2)/(4 + pi)


def test_issue_4547():
    assert limit(cot(x), x, 0, dir='+') is oo
    assert limit(cot(x), x, pi/2, dir='+') == 0


def test_issue_5164():
    assert limit(x**0.5, x, oo) == oo**0.5 is oo
    assert limit(x**0.5, x, 16) == 4 # Should this be a float?
    assert limit(x**0.5, x, 0) == 0
    assert limit(x**(-0.5), x, oo) == 0
    assert limit(x**(-0.5), x, 4) == S.Half # Should this be a float?


def test_issue_5383():
    func = (1.0 * 1 + 1.0 * x)**(1.0 * 1 / x)
    assert limit(func, x, 0) == E


def test_issue_14793():
    expr = ((x + S(1)/2) * log(x) - x + log(2*pi)/2 - \
        log(factorial(x)) + S(1)/(12*x))*x**3
    assert limit(expr, x, oo) == S(1)/360


def test_issue_5183():
    # using list(...) so py.test can recalculate values
    tests = list(product([x, -x],
                         [-1, 1],
                         [2, 3, S.Half, Rational(2, 3)],
                         ['-', '+']))
    results = (oo, oo, -oo, oo, -oo*I, oo, -oo*(-1)**Rational(1, 3), oo,
               0, 0, 0, 0, 0, 0, 0, 0,
               oo, oo, oo, -oo, oo, -oo*I, oo, -oo*(-1)**Rational(1, 3),
               0, 0, 0, 0, 0, 0, 0, 0)
    assert len(tests) == len(results)
    for i, (args, res) in enumerate(zip(tests, results)):
        y, s, e, d = args
        eq = y**(s*e)
        try:
            assert limit(eq, x, 0, dir=d) == res
        except AssertionError:
            if 0:  # change to 1 if you want to see the failing tests
                print()
                print(i, res, eq, d, limit(eq, x, 0, dir=d))
            else:
                assert None


def test_issue_5184():
    assert limit(sin(x)/x, x, oo) == 0
    assert limit(atan(x), x, oo) == pi/2
    assert limit(gamma(x), x, oo) is oo
    assert limit(cos(x)/x, x, oo) == 0
    assert limit(gamma(x), x, S.Half) == sqrt(pi)

    r = Symbol('r', real=True)
    assert limit(r*sin(1/r), r, 0) == 0


def test_issue_5229():
    assert limit((1 + y)**(1/y) - S.Exp1, y, 0) == 0


def test_issue_4546():
    # using list(...) so py.test can recalculate values
    tests = list(product([cot, tan],
                         [-pi/2, 0, pi/2, pi, pi*Rational(3, 2)],
                         ['-', '+']))
    results = (0, 0, -oo, oo, 0, 0, -oo, oo, 0, 0,
               oo, -oo, 0, 0, oo, -oo, 0, 0, oo, -oo)
    assert len(tests) == len(results)
    for i, (args, res) in enumerate(zip(tests, results)):
        f, l, d = args
        eq = f(x)
        try:
            assert limit(eq, x, l, dir=d) == res
        except AssertionError:
            if 0:  # change to 1 if you want to see the failing tests
                print()
                print(i, res, eq, l, d, limit(eq, x, l, dir=d))
            else:
                assert None


def test_issue_3934():
    assert limit((1 + x**log(3))**(1/x), x, 0) == 1
    assert limit((5**(1/x) + 3**(1/x))**x, x, 0) == 5


def test_issue_5955():
    assert limit((x**16)/(1 + x**16), x, oo) == 1
    assert limit((x**100)/(1 + x**100), x, oo) == 1
    assert limit((x**1885)/(1 + x**1885), x, oo) == 1
    assert limit((x**1000/((x + 1)**1000 + exp(-x))), x, oo) == 1


def test_newissue():
    assert limit(exp(1/sin(x))/exp(cot(x)), x, 0) == 1


def test_extended_real_line():
    assert limit(x - oo, x, oo) == Limit(x - oo, x, oo)
    assert limit(1/(x + sin(x)) - oo, x, 0) == Limit(1/(x + sin(x)) - oo, x, 0)
    assert limit(oo/x, x, oo) == Limit(oo/x, x, oo)
    assert limit(x - oo + 1/x, x, oo) == Limit(x - oo + 1/x, x, oo)


@XFAIL
def test_order_oo():
    x = Symbol('x', positive=True)
    assert Order(x)*oo != Order(1, x)
    assert limit(oo/(x**2 - 4), x, oo) is oo


def test_issue_5436():
    raises(NotImplementedError, lambda: limit(exp(x*y), x, oo))
    raises(NotImplementedError, lambda: limit(exp(-x*y), x, oo))


def test_Limit_dir():
    raises(TypeError, lambda: Limit(x, x, 0, dir=0))
    raises(ValueError, lambda: Limit(x, x, 0, dir='0'))


def test_polynomial():
    assert limit((x + 1)**1000/((x + 1)**1000 + 1), x, oo) == 1
    assert limit((x + 1)**1000/((x + 1)**1000 + 1), x, -oo) == 1
    assert limit(x ** Rational(77, 3) / (1 + x ** Rational(77, 3)), x, oo) == 1
    assert limit(x ** 101.1 / (1 + x ** 101.1), x, oo) == 1


def test_rational():
    assert limit(1/y - (1/(y + x) + x/(y + x)/y)/z, x, oo) == (z - 1)/(y*z)
    assert limit(1/y - (1/(y + x) + x/(y + x)/y)/z, x, -oo) == (z - 1)/(y*z)


def test_issue_5740():
    assert limit(log(x)*z - log(2*x)*y, x, 0) == oo*sign(y - z)


def test_issue_6366():
    n = Symbol('n', integer=True, positive=True)
    r = (n + 1)*x**(n + 1)/(x**(n + 1) - 1) - x/(x - 1)
    assert limit(r, x, 1).cancel() == n/2


def test_factorial():
    f = factorial(x)
    assert limit(f, x, oo) is oo
    assert limit(x/f, x, oo) == 0
    # see Stirling's approximation:
    # https://en.wikipedia.org/wiki/Stirling's_approximation
    assert limit(f/(sqrt(2*pi*x)*(x/E)**x), x, oo) == 1
    assert limit(f, x, -oo) == gamma(-oo)


def test_issue_6560():
    e = (5*x**3/4 - x*Rational(3, 4) + (y*(3*x**2/2 - S.Half) +
                             35*x**4/8 - 15*x**2/4 + Rational(3, 8))/(2*(y + 1)))
    assert limit(e, y, oo) == 5*x**3/4 + 3*x**2/4 - 3*x/4 - Rational(1, 4)

@XFAIL
def test_issue_5172():
    n = Symbol('n')
    r = Symbol('r', positive=True)
    c = Symbol('c')
    p = Symbol('p', positive=True)
    m = Symbol('m', negative=True)
    expr = ((2*n*(n - r + 1)/(n + r*(n - r + 1)))**c +
            (r - 1)*(n*(n - r + 2)/(n + r*(n - r + 1)))**c - n)/(n**c - n)
    expr = expr.subs(c, c + 1)
    raises(NotImplementedError, lambda: limit(expr, n, oo))
    assert limit(expr.subs(c, m), n, oo) == 1
    assert limit(expr.subs(c, p), n, oo).simplify() == \
        (2**(p + 1) + r - 1)/(r + 1)**(p + 1)


def test_issue_7088():
    a = Symbol('a')
    assert limit(sqrt(x/(x + a)), x, oo) == 1


def test_branch_cuts():
    assert limit(asin(I*x + 2), x, 0) == pi - asin(2)
    assert limit(asin(I*x + 2), x, 0, '-') == asin(2)
    assert limit(asin(I*x - 2), x, 0) == -asin(2)
    assert limit(asin(I*x - 2), x, 0, '-') == -pi + asin(2)
    assert limit(acos(I*x + 2), x, 0) == -acos(2)
    assert limit(acos(I*x + 2), x, 0, '-') == acos(2)
    assert limit(acos(I*x - 2), x, 0) == acos(-2)
    assert limit(acos(I*x - 2), x, 0, '-') == 2*pi - acos(-2)
    assert limit(atan(x + 2*I), x, 0) == I*atanh(2)
    assert limit(atan(x + 2*I), x, 0, '-') == -pi + I*atanh(2)
    assert limit(atan(x - 2*I), x, 0) == pi - I*atanh(2)
    assert limit(atan(x - 2*I), x, 0, '-') == -I*atanh(2)
    assert limit(atan(1/x), x, 0) == pi/2
    assert limit(atan(1/x), x, 0, '-') == -pi/2
    assert limit(atan(x), x, oo) == pi/2
    assert limit(atan(x), x, -oo) == -pi/2
    assert limit(acot(x + S(1)/2*I), x, 0) == pi - I*acoth(S(1)/2)
    assert limit(acot(x + S(1)/2*I), x, 0, '-') == -I*acoth(S(1)/2)
    assert limit(acot(x - S(1)/2*I), x, 0) == I*acoth(S(1)/2)
    assert limit(acot(x - S(1)/2*I), x, 0, '-') == -pi + I*acoth(S(1)/2)
    assert limit(acot(x), x, 0) == pi/2
    assert limit(acot(x), x, 0, '-') == -pi/2
    assert limit(asec(I*x + S(1)/2), x, 0) == asec(S(1)/2)
    assert limit(asec(I*x + S(1)/2), x, 0, '-') == -asec(S(1)/2)
    assert limit(asec(I*x - S(1)/2), x, 0) == 2*pi - asec(-S(1)/2)
    assert limit(asec(I*x - S(1)/2), x, 0, '-') == asec(-S(1)/2)
    assert limit(acsc(I*x + S(1)/2), x, 0) == acsc(S(1)/2)
    assert limit(acsc(I*x + S(1)/2), x, 0, '-') == pi - acsc(S(1)/2)
    assert limit(acsc(I*x - S(1)/2), x, 0) == -pi + acsc(S(1)/2)
    assert limit(acsc(I*x - S(1)/2), x, 0, '-') == -acsc(S(1)/2)

    assert limit(log(I*x - 1), x, 0) == I*pi
    assert limit(log(I*x - 1), x, 0, '-') == -I*pi
    assert limit(log(-I*x - 1), x, 0) == -I*pi
    assert limit(log(-I*x - 1), x, 0, '-') == I*pi

    assert limit(sqrt(I*x - 1), x, 0) == I
    assert limit(sqrt(I*x - 1), x, 0, '-') == -I
    assert limit(sqrt(-I*x - 1), x, 0) == -I
    assert limit(sqrt(-I*x - 1), x, 0, '-') == I

    assert limit(cbrt(I*x - 1), x, 0) == (-1)**(S(1)/3)
    assert limit(cbrt(I*x - 1), x, 0, '-') == -(-1)**(S(2)/3)
    assert limit(cbrt(-I*x - 1), x, 0) == -(-1)**(S(2)/3)
    assert limit(cbrt(-I*x - 1), x, 0, '-') == (-1)**(S(1)/3)


def test_issue_6364():
    a = Symbol('a')
    e = z/(1 - sqrt(1 + z)*sin(a)**2 - sqrt(1 - z)*cos(a)**2)
    assert limit(e, z, 0) == 1/(cos(a)**2 - S.Half)


def test_issue_6682():
    assert limit(exp(2*Ei(-x))/x**2, x, 0) == exp(2*EulerGamma)


def test_issue_4099():
    a = Symbol('a')
    assert limit(a/x, x, 0) == oo*sign(a)
    assert limit(-a/x, x, 0) == -oo*sign(a)
    assert limit(-a*x, x, oo) == -oo*sign(a)
    assert limit(a*x, x, oo) == oo*sign(a)


def test_issue_4503():
    dx = Symbol('dx')
    assert limit((sqrt(1 + exp(x + dx)) - sqrt(1 + exp(x)))/dx, dx, 0) == \
        exp(x)/(2*sqrt(exp(x) + 1))


def test_issue_6052():
    G = meijerg((), (), (1,), (0,), -x)
    g = hyperexpand(G)
    assert limit(g, x, 0, '+-') == 0
    assert limit(g, x, oo) == -oo


def test_issue_7224():
    expr = sqrt(x)*besseli(1,sqrt(8*x))
    assert limit(x*diff(expr, x, x)/expr, x, 0) == 2
    assert limit(x*diff(expr, x, x)/expr, x, 1).evalf() == 2.0


def test_issue_7391_8166():
    f = Function('f')
    # limit should depend on the continuity of the expression at the point passed
    assert limit(f(x), x, 4) == Limit(f(x), x, 4, dir='+')
    assert limit(x*f(x)**2/(x**2 + f(x)**4), x, 0) == Limit(x*f(x)**2/(x**2 + f(x)**4), x, 0, dir='+')


def test_issue_8208():
    assert limit(n**(Rational(1, 1e9) - 1), n, oo) == 0


def test_issue_8229():
    assert limit((x**Rational(1, 4) - 2)/(sqrt(x) - 4)**Rational(2, 3), x, 16) == 0


def test_issue_8433():
    d, t = symbols('d t', positive=True)
    assert limit(erf(1 - t/d), t, oo) == -1


def test_issue_8481():
    k = Symbol('k', integer=True, nonnegative=True)
    lamda = Symbol('lamda', positive=True)
    assert limit(lamda**k * exp(-lamda) / factorial(k), k, oo) == 0


def test_issue_8462():
    assert limit(binomial(n, n/2), n, oo) == oo
    assert limit(binomial(n, n/2) * 3 ** (-n), n, oo) == 0


def test_issue_8634():
    n = Symbol('n', integer=True, positive=True)
    x = Symbol('x')
    assert limit(x**n, x, -oo) == oo*sign((-1)**n)


def test_issue_8635_18176():
    x = Symbol('x', real=True)
    k = Symbol('k', positive=True)
    assert limit(x**n - x**(n - 0), x, oo) == 0
    assert limit(x**n - x**(n - 5), x, oo) == oo
    assert limit(x**n - x**(n - 2.5), x, oo) == oo
    assert limit(x**n - x**(n - k - 1), x, oo) == oo
    x = Symbol('x', positive=True)
    assert limit(x**n - x**(n - 1), x, oo) == oo
    assert limit(x**n - x**(n + 2), x, oo) == -oo


def test_issue_8730():
    assert limit(subfactorial(x), x, oo) is oo


def test_issue_9252():
    n = Symbol('n', integer=True)
    c = Symbol('c', positive=True)
    assert limit((log(n))**(n/log(n)) / (1 + c)**n, n, oo) == 0
    # limit should depend on the value of c
    raises(NotImplementedError, lambda: limit((log(n))**(n/log(n)) / c**n, n, oo))


def test_issue_9558():
    assert limit(sin(x)**15, x, 0, '-') == 0


def test_issue_10801():
    # make sure limits work with binomial
    assert limit(16**k / (k * binomial(2*k, k)**2), k, oo) == pi


def test_issue_10976():
    s, x = symbols('s x', real=True)
    assert limit(erf(s*x)/erf(s), s, 0) == x


def test_issue_9041():
    assert limit(factorial(n) / ((n/exp(1))**n * sqrt(2*pi*n)), n, oo) == 1


def test_issue_9205():
    x, y, a = symbols('x, y, a')
    assert Limit(x, x, a).free_symbols == {a}
    assert Limit(x, x, a, '-').free_symbols == {a}
    assert Limit(x + y, x + y, a).free_symbols == {a}
    assert Limit(-x**2 + y, x**2, a).free_symbols == {y, a}


def test_issue_9471():
    assert limit(((27**(log(n,3)))/n**3),n,oo) == 1
    assert limit(((27**(log(n,3)+1))/n**3),n,oo) == 27


def test_issue_10382():
    assert limit(fibonacci(n + 1)/fibonacci(n), n, oo) == GoldenRatio


def test_issue_11496():
    assert limit(erfc(log(1/x)), x, oo) == 2


def test_issue_11879():
    assert simplify(limit(((x+y)**n-x**n)/y, y, 0)) == n*x**(n-1)


def test_limit_with_Float():
    k = symbols("k")
    assert limit(1.0 ** k, k, oo) == 1
    assert limit(0.3*1.0**k, k, oo) == Rational(3, 10)


def test_issue_10610():
    assert limit(3**x*3**(-x - 1)*(x + 1)**2/x**2, x, oo) == Rational(1, 3)


def test_issue_10868():
    assert limit(log(x) + asech(x), x, 0, '+') == log(2)
    assert limit(log(x) + asech(x), x, 0, '-') == log(2) + 2*I*pi
    raises(ValueError, lambda: limit(log(x) + asech(x), x, 0, '+-'))
    assert limit(log(x) + asech(x), x, oo) == oo
    assert limit(log(x) + acsch(x), x, 0, '+') == log(2)
    assert limit(log(x) + acsch(x), x, 0, '-') == -oo
    raises(ValueError, lambda: limit(log(x) + acsch(x), x, 0, '+-'))
    assert limit(log(x) + acsch(x), x, oo) == oo


def test_issue_6599():
    assert limit((n + cos(n))/n, n, oo) == 1


def test_issue_12555():
    assert limit((3**x + 2* x**10) / (x**10 + exp(x)), x, -oo) == 2
    assert limit((3**x + 2* x**10) / (x**10 + exp(x)), x, oo) is oo


def test_issue_12769():
    r, z, x = symbols('r z x', real=True)
    a, b, s0, K, F0, s, T = symbols('a b s0 K F0 s T', positive=True, real=True)
    fx = (F0**b*K**b*r*s0 - sqrt((F0**2*K**(2*b)*a**2*(b - 1) + \
        F0**(2*b)*K**2*a**2*(b - 1) + F0**(2*b)*K**(2*b)*s0**2*(b - 1)*(b**2 - 2*b + 1) - \
        2*F0**(2*b)*K**(b + 1)*a*r*s0*(b**2 - 2*b +  1) + \
        2*F0**(b + 1)*K**(2*b)*a*r*s0*(b**2 - 2*b + 1) - \
        2*F0**(b + 1)*K**(b + 1)*a**2*(b - 1))/((b - 1)*(b**2 - 2*b + 1))))*(b*r -  b - r + 1)

    assert fx.subs(K, F0).factor(deep=True) == limit(fx, K, F0).factor(deep=True)


def test_issue_13332():
    assert limit(sqrt(30)*5**(-5*x - 1)*(46656*x)**x*(5*x + 2)**(5*x + 5*S.Half) *
                (6*x + 2)**(-6*x - 5*S.Half), x, oo) == Rational(25, 36)


def test_issue_12564():
    assert limit(x**2 + x*sin(x) + cos(x), x, -oo) is oo
    assert limit(x**2 + x*sin(x) + cos(x), x, oo) is oo
    assert limit(((x + cos(x))**2).expand(), x, oo) is oo
    assert limit(((x + sin(x))**2).expand(), x, oo) is oo
    assert limit(((x + cos(x))**2).expand(), x, -oo) is oo
    assert limit(((x + sin(x))**2).expand(), x, -oo) is oo


def test_issue_14456():
    raises(NotImplementedError, lambda: Limit(exp(x), x, zoo).doit())
    raises(NotImplementedError, lambda: Limit(x**2/(x+1), x, zoo).doit())


def test_issue_14411():
    assert limit(3*sec(4*pi*x - x/3), x, 3*pi/(24*pi - 2)) is -oo


def test_issue_13382():
    assert limit(x*(((x + 1)**2 + 1)/(x**2 + 1) - 1), x, oo) == 2


def test_issue_13403():
    assert limit(x*(-1 + (x + log(x + 1) + 1)/(x + log(x))), x, oo) == 1


def test_issue_13416():
    assert limit((-x**3*log(x)**3 + (x - 1)*(x + 1)**2*log(x + 1)**3)/(x**2*log(x)**3), x, oo) == 1


def test_issue_13462():
    assert limit(n**2*(2*n*(-(1 - 1/(2*n))**x + 1) - x - (-x**2/4 + x/4)/n), n, oo) == x**3/24 - x**2/8 + x/12


def test_issue_13750():
    a = Symbol('a')
    assert limit(erf(a - x), x, oo) == -1
    assert limit(erf(sqrt(x) - x), x, oo) == -1


def test_issue_14276():
    assert isinstance(limit(sin(x)**log(x), x, oo), Limit)
    assert isinstance(limit(sin(x)**cos(x), x, oo), Limit)
    assert isinstance(limit(sin(log(cos(x))), x, oo), Limit)
    assert limit((1 + 1/(x**2 + cos(x)))**(x**2 + x), x, oo) == E


def test_issue_14514():
    assert limit((1/(log(x)**log(x)))**(1/x), x, oo) == 1


def test_issues_14525():
    assert limit(sin(x)**2 - cos(x) + tan(x)*csc(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(sin(x)**2 - cos(x) + sin(x)*cot(x), x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(cot(x) - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.Infinity)
    assert limit(cos(x) - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.One)
    assert limit(sin(x) - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.One)
    assert limit(cos(x)**2 - tan(x)**2, x, oo) == AccumBounds(S.NegativeInfinity, S.One)
    assert limit(tan(x)**2 + sin(x)**2 - cos(x), x, oo) == AccumBounds(-S.One, S.Infinity)


def test_issue_14574():
    assert limit(sqrt(x)*cos(x - x**2) / (x + 1), x, oo) == 0


def test_issue_10102():
    assert limit(fresnels(x), x, oo) == S.Half
    assert limit(3 + fresnels(x), x, oo) == 3 + S.Half
    assert limit(5*fresnels(x), x, oo) == Rational(5, 2)
    assert limit(fresnelc(x), x, oo) == S.Half
    assert limit(fresnels(x), x, -oo) == Rational(-1, 2)
    assert limit(4*fresnelc(x), x, -oo) == -2


def test_issue_14377():
    raises(NotImplementedError, lambda: limit(exp(I*x)*sin(pi*x), x, oo))


def test_issue_15146():
    e = (x/2) * (-2*x**3 - 2*(x**3 - 1) * x**2 * digamma(x**3 + 1) + \
        2*(x**3 - 1) * x**2 * digamma(x**3 + x + 1) + x + 3)
    assert limit(e, x, oo) == S(1)/3


def test_issue_15202():
    e = (2**x*(2 + 2**(-x)*(-2*2**x + x + 2))/(x + 1))**(x + 1)
    assert limit(e, x, oo) == exp(1)

    e = (log(x, 2)**7 + 10*x*factorial(x) + 5**x) / (factorial(x + 1) + 3*factorial(x) + 10**x)
    assert limit(e, x, oo) == 10


def test_issue_15282():
    assert limit((x**2000 - (x + 1)**2000) / x**1999, x, oo) == -2000


def test_issue_15984():
    assert limit((-x + log(exp(x) + 1))/x, x, oo, dir='-') == 0


def test_issue_13571():
    assert limit(uppergamma(x, 1) / gamma(x), x, oo) == 1


def test_issue_13575():
    assert limit(acos(erfi(x)), x, 1) == acos(erfi(S.One))


def test_issue_17325():
    assert Limit(sin(x)/x, x, 0, dir="+-").doit() == 1
    assert Limit(x**2, x, 0, dir="+-").doit() == 0
    assert Limit(1/x**2, x, 0, dir="+-").doit() is oo
    assert Limit(1/x, x, 0, dir="+-").doit() is zoo


def test_issue_10978():
    assert LambertW(x).limit(x, 0) == 0


def test_issue_14313_comment():
    assert limit(floor(n/2), n, oo) is oo


def test_issue_15323():
    d = ((1 - 1/x)**x).diff(x)
    assert limit(d, x, 1, dir='+') == 1


def test_issue_12571():
    assert limit(-LambertW(-log(x))/log(x), x, 1) == 1


def test_issue_14590():
    assert limit((x**3*((x + 1)/x)**x)/((x + 1)*(x + 2)*(x + 3)), x, oo) == exp(1)


def test_issue_14393():
    a, b = symbols('a b')
    assert limit((x**b - y**b)/(x**a - y**a), x, y) == b*y**(-a + b)/a


def test_issue_14556():
    assert limit(factorial(n + 1)**(1/(n + 1)) - factorial(n)**(1/n), n, oo) == exp(-1)


def test_issue_14811():
    assert limit(((1 + ((S(2)/3)**(x + 1)))**(2**x))/(2**((S(4)/3)**(x - 1))), x, oo) == oo


def test_issue_16222():
    assert limit(exp(x), x, 1000000000) == exp(1000000000)


def test_issue_16714():
    assert limit(((x**(x + 1) + (x + 1)**x) / x**(x + 1))**x, x, oo) == exp(exp(1))


def test_issue_16722():
    z = symbols('z', positive=True)
    assert limit(binomial(n + z, n)*n**-z, n, oo) == 1/gamma(z + 1)
    z = symbols('z', positive=True, integer=True)
    assert limit(binomial(n + z, n)*n**-z, n, oo) == 1/gamma(z + 1)


def test_issue_17431():
    assert limit(((n + 1) + 1) / (((n + 1) + 2) * factorial(n + 1)) *
                 (n + 2) * factorial(n) / (n + 1), n, oo) == 0
    assert limit((n + 2)**2*factorial(n)/((n + 1)*(n + 3)*factorial(n + 1))
                 , n, oo) == 0
    assert limit((n + 1) * factorial(n) / (n * factorial(n + 1)), n, oo) == 0


def test_issue_17671():
    assert limit(Ei(-log(x)) - log(log(x))/x, x, 1) == EulerGamma


def test_issue_17751():
    a, b, c, x = symbols('a b c x', positive=True)
    assert limit((a + 1)*x - sqrt((a + 1)**2*x**2 + b*x + c), x, oo) == -b/(2*a + 2)


def test_issue_17792():
    assert limit(factorial(n)/sqrt(n)*(exp(1)/n)**n, n, oo) == sqrt(2)*sqrt(pi)


def test_issue_18118():
    assert limit(sign(sin(x)), x, 0, "-") == -1
    assert limit(sign(sin(x)), x, 0, "+") == 1


def test_issue_18306():
    assert limit(sin(sqrt(x))/sqrt(sin(x)), x, 0, '+') == 1


def test_issue_18378():
    assert limit(log(exp(3*x) + x)/log(exp(x) + x**100), x, oo) == 3


def test_issue_18399():
    assert limit((1 - S(1)/2*x)**(3*x), x, oo) is zoo
    assert limit((-x)**x, x, oo) is zoo


def test_issue_18442():
    assert limit(tan(x)**(2**(sqrt(pi))), x, oo, dir='-') == Limit(tan(x)**(2**(sqrt(pi))), x, oo, dir='-')


def test_issue_18452():
    assert limit(abs(log(x))**x, x, 0) == 1
    assert limit(abs(log(x))**x, x, 0, "-") == 1


def test_issue_18473():
    assert limit(sin(x)**(1/x), x, oo) == Limit(sin(x)**(1/x), x, oo, dir='-')
    assert limit(cos(x)**(1/x), x, oo) == Limit(cos(x)**(1/x), x, oo, dir='-')
    assert limit(tan(x)**(1/x), x, oo) == Limit(tan(x)**(1/x), x, oo, dir='-')
    assert limit((cos(x) + 2)**(1/x), x, oo) == 1
    assert limit((sin(x) + 10)**(1/x), x, oo) == 1
    assert limit((cos(x) - 2)**(1/x), x, oo) == Limit((cos(x) - 2)**(1/x), x, oo, dir='-')
    assert limit((cos(x) + 1)**(1/x), x, oo) == AccumBounds(0, 1)
    assert limit((tan(x)**2)**(2/x) , x, oo) == AccumBounds(0, oo)
    assert limit((sin(x)**2)**(1/x), x, oo) == AccumBounds(0, 1)
    # Tests for issue #23751
    assert limit((cos(x) + 1)**(1/x), x, -oo) == AccumBounds(1, oo)
    assert limit((sin(x)**2)**(1/x), x, -oo) == AccumBounds(1, oo)
    assert limit((tan(x)**2)**(2/x) , x, -oo) == AccumBounds(0, oo)


def test_issue_18482():
    assert limit((2*exp(3*x)/(exp(2*x) + 1))**(1/x), x, oo) == exp(1)


def test_issue_18508():
    assert limit(sin(x)/sqrt(1-cos(x)), x, 0) == sqrt(2)
    assert limit(sin(x)/sqrt(1-cos(x)), x, 0, dir='+') == sqrt(2)
    assert limit(sin(x)/sqrt(1-cos(x)), x, 0, dir='-') == -sqrt(2)


def test_issue_18521():
    raises(NotImplementedError, lambda: limit(exp((2 - n) * x), x, oo))


def test_issue_18969():
    a, b = symbols('a b', positive=True)
    assert limit(LambertW(a), a, b) == LambertW(b)
    assert limit(exp(LambertW(a)), a, b) == exp(LambertW(b))


def test_issue_18992():
    assert limit(n/(factorial(n)**(1/n)), n, oo) == exp(1)


def test_issue_19067():
    x = Symbol('x')
    assert limit(gamma(x)/(gamma(x - 1)*gamma(x + 2)), x, 0) == -1


def test_issue_19586():
    assert limit(x**(2**x*3**(-x)), x, oo) == 1


def test_issue_13715():
    n = Symbol('n')
    p = Symbol('p', zero=True)
    assert limit(n + p, n, 0) == 0


def test_issue_15055():
    assert limit(n**3*((-n - 1)*sin(1/n) + (n + 2)*sin(1/(n + 1)))/(-n + 1), n, oo) == 1


def test_issue_16708():
    m, vi = symbols('m vi', positive=True)
    B, ti, d = symbols('B ti d')
    assert limit((B*ti*vi - sqrt(m)*sqrt(-2*B*d*vi + m*(vi)**2) + m*vi)/(B*vi), B, 0) == (d + ti*vi)/vi


def test_issue_19154():
    assert limit(besseli(1, 3 *x)/(x *besseli(1, x)**3), x , oo) == 2*sqrt(3)*pi/3
    assert limit(besseli(1, 3 *x)/(x *besseli(1, x)**3), x , -oo) == -2*sqrt(3)*pi/3


def test_issue_19453():
    beta = Symbol("beta", positive=True)
    h = Symbol("h", positive=True)
    m = Symbol("m", positive=True)
    w = Symbol("omega", positive=True)
    g = Symbol("g", positive=True)

    e = exp(1)
    q = 3*h**2*beta*g*e**(0.5*h*beta*w)
    p = m**2*w**2
    s = e**(h*beta*w) - 1
    Z = -q/(4*p*s) - q/(2*p*s**2) - q*(e**(h*beta*w) + 1)/(2*p*s**3)\
            + e**(0.5*h*beta*w)/s
    E = -diff(log(Z), beta)

    assert limit(E - 0.5*h*w, beta, oo) == 0
    assert limit(E.simplify() - 0.5*h*w, beta, oo) == 0


def test_issue_19739():
    assert limit((-S(1)/4)**x, x, oo) == 0


def test_issue_19766():
    assert limit(2**(-x)*sqrt(4**(x + 1) + 1), x, oo) == 2


def test_issue_19770():
    m = Symbol('m')
    # the result is not 0 for non-real m
    assert limit(cos(m*x)/x, x, oo) == Limit(cos(m*x)/x, x, oo, dir='-')
    m = Symbol('m', real=True)
    # can be improved to give the correct result 0
    assert limit(cos(m*x)/x, x, oo) == Limit(cos(m*x)/x, x, oo, dir='-')
    m = Symbol('m', nonzero=True)
    assert limit(cos(m*x), x, oo) == AccumBounds(-1, 1)
    assert limit(cos(m*x)/x, x, oo) == 0


def test_issue_7535():
    assert limit(tan(x)/sin(tan(x)), x, pi/2) == Limit(tan(x)/sin(tan(x)), x, pi/2, dir='+')
    assert limit(tan(x)/sin(tan(x)), x, pi/2, dir='-') == Limit(tan(x)/sin(tan(x)), x, pi/2, dir='-')
    assert limit(tan(x)/sin(tan(x)), x, pi/2, dir='+-') == Limit(tan(x)/sin(tan(x)), x, pi/2, dir='+-')
    assert limit(sin(tan(x)),x,pi/2) == AccumBounds(-1, 1)
    assert -oo*(1/sin(-oo)) == AccumBounds(-oo, oo)
    assert oo*(1/sin(oo)) == AccumBounds(-oo, oo)
    assert oo*(1/sin(-oo)) == AccumBounds(-oo, oo)
    assert -oo*(1/sin(oo)) == AccumBounds(-oo, oo)


def test_issue_20365():
    assert limit(((x + 1)**(1/x) - E)/x, x, 0) == -E/2


def test_issue_21031():
    assert limit(((1 + x)**(1/x) - (1 + 2*x)**(1/(2*x)))/asin(x), x, 0) == E/2


def test_issue_21038():
    assert limit(sin(pi*x)/(3*x - 12), x, 4) == pi/3


def test_issue_20578():
    expr = abs(x) * sin(1/x)
    assert limit(expr,x,0,'+') == 0
    assert limit(expr,x,0,'-') == 0
    assert limit(expr,x,0,'+-') == 0


def test_issue_21227():
    f = log(x)

    assert f.nseries(x, logx=y) == y
    assert f.nseries(x, logx=-x) == -x

    f = log(-log(x))

    assert f.nseries(x, logx=y) == log(-y)
    assert f.nseries(x, logx=-x) == log(x)

    f = log(log(x))

    assert f.nseries(x, logx=y) == log(y)
    assert f.nseries(x, logx=-x) == log(-x)
    assert f.nseries(x, logx=x) == log(x)

    f = log(log(log(1/x)))

    assert f.nseries(x, logx=y) == log(log(-y))
    assert f.nseries(x, logx=-y) == log(log(y))
    assert f.nseries(x, logx=x) == log(log(-x))
    assert f.nseries(x, logx=-x) == log(log(x))


def test_issue_21415():
    exp = (x-1)*cos(1/(x-1))
    assert exp.limit(x,1) == 0
    assert exp.expand().limit(x,1) == 0


def test_issue_21530():
    assert limit(sinh(n + 1)/sinh(n), n, oo) == E


def test_issue_21550():
    r = (sqrt(5) - 1)/2
    assert limit((x - r)/(x**2 + x - 1), x, r) == sqrt(5)/5


def test_issue_21661():
    out = limit((x**(x + 1) * (log(x) + 1) + 1) / x, x, 11)
    assert out == S(3138428376722)/11 + 285311670611*log(11)


def test_issue_21701():
    assert limit((besselj(z, x)/x**z).subs(z, 7), x, 0) == S(1)/645120


def test_issue_21721():
    a = Symbol('a', real=True)
    I = integrate(1/(pi*(1 + (x - a)**2)), x)
    assert I.limit(x, oo) == S.Half


def test_issue_21756():
    term = (1 - exp(-2*I*pi*z))/(1 - exp(-2*I*pi*z/5))
    assert term.limit(z, 0) == 5
    assert re(term).limit(z, 0) == 5


def test_issue_21785():
    a = Symbol('a')
    assert sqrt((-a**2 + x**2)/(1 - x**2)).limit(a, 1, '-') == I


def test_issue_22181():
    assert limit((-1)**x * 2**(-x), x, oo) == 0


def test_issue_22220():
    e1 = sqrt(30)*atan(sqrt(30)*tan(x/2)/6)/30
    e2 = sqrt(30)*I*(-log(sqrt(2)*tan(x/2) - 2*sqrt(15)*I/5) +
                     +log(sqrt(2)*tan(x/2) + 2*sqrt(15)*I/5))/60

    assert limit(e1, x, -pi) == -sqrt(30)*pi/60
    assert limit(e2, x, -pi) == -sqrt(30)*pi/30

    assert limit(e1, x, -pi, '-') == sqrt(30)*pi/60
    assert limit(e2, x, -pi, '-') == 0

    # test https://github.com/sympy/sympy/issues/22220#issuecomment-972727694
    expr = log(x - I) - log(-x - I)
    expr2 = logcombine(expr, force=True)
    assert limit(expr, x, oo) == limit(expr2, x, oo) == I*pi

    # test https://github.com/sympy/sympy/issues/22220#issuecomment-1077618340
    expr = expr = (-log(tan(x/2) - I) +log(tan(x/2) + I))
    assert limit(expr, x, pi, '+') == 2*I*pi
    assert limit(expr, x, pi, '-') == 0


def test_issue_22334():
    k, n  = symbols('k, n', positive=True)
    assert limit((n+1)**k/((n+1)**(k+1) - (n)**(k+1)), n, oo) == 1/(k + 1)
    assert limit((n+1)**k/((n+1)**(k+1) - (n)**(k+1)).expand(), n, oo) == 1/(k + 1)
    assert limit((n+1)**k/(n*(-n**k + (n + 1)**k) + (n + 1)**k), n, oo) == 1/(k + 1)


def test_issue_22836_limit():
    assert limit(2**(1/x)/factorial(1/(x)), x, 0) == S.Zero


def test_sympyissue_22986():
    assert limit(acosh(1 + 1/x)*sqrt(x), x, oo) == sqrt(2)


def test_issue_23231():
    f = (2**x - 2**(-x))/(2**x + 2**(-x))
    assert limit(f, x, -oo) == -1


def test_issue_23596():
    assert integrate(((1 + x)/x**2)*exp(-1/x), (x, 0, oo)) == oo


def test_issue_23752():
    expr1 = sqrt(-I*x**2 + x - 3)
    expr2 = sqrt(-I*x**2 + I*x - 3)
    assert limit(expr1, x, 0, '+') == -sqrt(3)*I
    assert limit(expr1, x, 0, '-') == -sqrt(3)*I
    assert limit(expr2, x, 0, '+') == sqrt(3)*I
    assert limit(expr2, x, 0, '-') == -sqrt(3)*I


def test_issue_24276():
    fx = log(tan(pi/2*tanh(x))).diff(x)
    assert fx.limit(x, oo) == 2
    assert fx.simplify().limit(x, oo) == 2
    assert fx.rewrite(sin).limit(x, oo) == 2
    assert fx.rewrite(sin).simplify().limit(x, oo) == 2

def test_issue_25230():
    a = Symbol('a', real = True)
    b = Symbol('b', positive = True)
    c = Symbol('c', negative = True)
    n = Symbol('n', integer = True)
    raises(NotImplementedError, lambda: limit(Mod(x, a), x, a))
    assert limit(Mod(x, b), x, n*b, '+') == 0
    assert limit(Mod(x, b), x, n*b, '-') == b
    assert limit(Mod(x, c), x, n*c, '+') == c
    assert limit(Mod(x, c), x, n*c, '-') == 0


def test_issue_25582():

    assert limit(asin(exp(x)), x, oo, '-') == -oo*I
    assert limit(acos(exp(x)), x, oo, '-') == oo*I
    assert limit(atan(exp(x)), x, oo, '-') == pi/2
    assert limit(acot(exp(x)), x, oo, '-') == 0
    assert limit(asec(exp(x)), x, oo, '-') == pi/2
    assert limit(acsc(exp(x)), x, oo, '-') == 0


def test_issue_25847():
    #atan
    assert limit(atan(sin(x)/x), x, 0, '+-') == pi/4
    assert limit(atan(exp(1/x)), x, 0, '+') == pi/2
    assert limit(atan(exp(1/x)), x, 0, '-') == 0

    #asin
    assert limit(asin(sin(x)/x), x, 0, '+-') == pi/2
    assert limit(asin(exp(1/x)), x, 0, '+') == -oo*I
    assert limit(asin(exp(1/x)), x, 0, '-') == 0

    #acos
    assert limit(acos(sin(x)/x), x, 0, '+-') == 0
    assert limit(acos(exp(1/x)), x, 0, '+') == oo*I
    assert limit(acos(exp(1/x)), x, 0, '-') == pi/2

    #acot
    assert limit(acot(sin(x)/x), x, 0, '+-') == pi/4
    assert limit(acot(exp(1/x)), x, 0, '+') == 0
    assert limit(acot(exp(1/x)), x, 0, '-') == pi/2

    #asec
    assert limit(asec(sin(x)/x), x, 0, '+-') == 0
    assert limit(asec(exp(1/x)), x, 0, '+') == pi/2
    assert limit(asec(exp(1/x)), x, 0, '-') == oo*I

    #acsc
    assert limit(acsc(sin(x)/x), x, 0, '+-') == pi/2
    assert limit(acsc(exp(1/x)), x, 0, '+') == 0
    assert limit(acsc(exp(1/x)), x, 0, '-') == -oo*I

    #atanh
    assert limit(atanh(sin(x)/x), x, 0, '+-') == oo
    assert limit(atanh(exp(1/x)), x, 0, '+') == -I*pi/2
    assert limit(atanh(exp(1/x)), x, 0, '-') == 0

    #asinh
    assert limit(asinh(sin(x)/x), x, 0, '+-') == log(1 + sqrt(2))
    assert limit(asinh(exp(1/x)), x, 0, '+') == oo
    assert limit(asinh(exp(1/x)), x, 0, '-') == 0

    #acosh
    assert limit(acosh(sin(x)/x), x, 0, '+-') == 0
    assert limit(acosh(exp(1/x)), x, 0, '+') == oo
    assert limit(acosh(exp(1/x)), x, 0, '-') == I*pi/2

    #acoth
    assert limit(acoth(sin(x)/x), x, 0, '+-') == oo
    assert limit(acoth(exp(1/x)), x, 0, '+') == 0
    assert limit(acoth(exp(1/x)), x, 0, '-') == -I*pi/2

    #asech
    assert limit(asech(sin(x)/x), x, 0, '+-') == 0
    assert limit(asech(exp(1/x)), x, 0, '+') == I*pi/2
    assert limit(asech(exp(1/x)), x, 0, '-') == oo

    #acsch
    assert limit(acsch(sin(x)/x), x, 0, '+-') == log(1 + sqrt(2))
    assert limit(acsch(exp(1/x)), x, 0, '+') == 0
    assert limit(acsch(exp(1/x)), x, 0, '-') == oo


def test_issue_26040():
    assert limit(besseli(0, x + 1)/besseli(0, x), x, oo) == S.Exp1


def test_issue_26250():
    e = elliptic_e(4*x/(x**2 + 2*x + 1))
    k = elliptic_k(4*x/(x**2 + 2*x + 1))
    e1 = ((1-3*x**2)*e**2/2 - (x**2-2*x+1)*e*k/2)
    e2 = pi**2*(x**8 - 2*x**7 - x**6 + 4*x**5 - x**4 - 2*x**3 + x**2)
    assert limit(e1/e2, x, 0) == -S(1)/8


def test_issue_26513():
    assert limit(abs((-x/(x+1))**x), x ,oo) == exp(-1)
    assert limit((x/(x + 1))**x, x, oo) == exp(-1)
    raises (NotImplementedError, lambda: limit((-x/(x+1))**x, x, oo))


def test_issue_26916():
    assert limit(Ei(x)*exp(-x), x, +oo) == 0
    assert limit(Ei(x)*exp(-x), x, -oo) == 0


def test_issue_22982_15323():
    assert limit((log(E + 1/x) - 1)**(1 - sqrt(E + 1/x)), x, oo) == oo
    assert limit((1 - 1/x)**x*(log(1 - 1/x) + 1/(x*(1 - 1/x))), x, 1, dir='+') == 1
    assert limit((log(E + 1/x) )**(1 - sqrt(E + 1/x)), x, oo) == 1
    assert limit((log(E + 1/x) - 1)**(- sqrt(E + 1/x)), x, oo) == oo


def test_issue_26991():
    assert limit(x/((x - 6)*sinh(tanh(0.03*x)) + tanh(x) - 0.5), x, oo) == 1/sinh(1)

def test_issue_27278():
    expr = (1/(x*log((x + 3)/x)))**x*((x + 1)*log((x + 4)/(x + 1)))**(x + 1)/3
    assert limit(expr, x, oo) == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\_backends\_backend.py ===
from abc import ABC, abstractmethod


class Backend(ABC):
    def __init__(
        self,
        modulename,
        sources,
        extra_objects,
        build_dir,
        include_dirs,
        library_dirs,
        libraries,
        define_macros,
        undef_macros,
        f2py_flags,
        sysinfo_flags,
        fc_flags,
        flib_flags,
        setup_flags,
        remove_build_dir,
        extra_dat,
    ):
        self.modulename = modulename
        self.sources = sources
        self.extra_objects = extra_objects
        self.build_dir = build_dir
        self.include_dirs = include_dirs
        self.library_dirs = library_dirs
        self.libraries = libraries
        self.define_macros = define_macros
        self.undef_macros = undef_macros
        self.f2py_flags = f2py_flags
        self.sysinfo_flags = sysinfo_flags
        self.fc_flags = fc_flags
        self.flib_flags = flib_flags
        self.setup_flags = setup_flags
        self.remove_build_dir = remove_build_dir
        self.extra_dat = extra_dat

    @abstractmethod
    def compile(self) -> None:
        """Compile the wrapper."""
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\_examples\cffi\extending.py ===
"""
Use cffi to access any of the underlying C functions from distributions.h
"""
import os

import cffi

import numpy as np

from .parse import parse_distributions_h

ffi = cffi.FFI()

inc_dir = os.path.join(np.get_include(), 'numpy')

# Basic numpy types
ffi.cdef('''
    typedef intptr_t npy_intp;
    typedef unsigned char npy_bool;

''')

parse_distributions_h(ffi, inc_dir)

lib = ffi.dlopen(np.random._generator.__file__)

# Compare the distributions.h random_standard_normal_fill to
# Generator.standard_random
bit_gen = np.random.PCG64()
rng = np.random.Generator(bit_gen)
state = bit_gen.state

interface = rng.bit_generator.cffi
n = 100
vals_cffi = ffi.new('double[%d]' % n)
lib.random_standard_normal_fill(interface.bit_generator, n, vals_cffi)

# reset the state
bit_gen.state = state

vals = rng.standard_normal(n)

for i in range(n):
    assert vals[i] == vals_cffi[i]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\reader\strings.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.cell.text import Text

from openpyxl.xml.functions import iterparse
from openpyxl.xml.constants import SHEET_MAIN_NS
from openpyxl.cell.rich_text import CellRichText


def read_string_table(xml_source):
    """Read in all shared strings in the table"""

    strings = []
    STRING_TAG = '{%s}si' % SHEET_MAIN_NS

    for _, node in iterparse(xml_source):
        if node.tag == STRING_TAG:
            text = Text.from_tree(node).content
            text = text.replace('x005F_', '')
            node.clear()

            strings.append(text)

    return strings


def read_rich_text(xml_source):
    """Read in all shared strings in the table"""

    strings = []
    STRING_TAG = '{%s}si' % SHEET_MAIN_NS

    for _, node in iterparse(xml_source):
        if node.tag == STRING_TAG:
            text = CellRichText.from_tree(node)
            if len(text) == 0:
                text = ''
            elif len(text) == 1 and isinstance(text[0], str):
                text = text[0]
            node.clear()

            strings.append(text)

    return strings

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\pool\__init__.py ===
# pool/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php


"""Connection pooling for DB-API connections.

Provides a number of connection pool implementations for a variety of
usage scenarios and thread behavior requirements imposed by the
application, DB-API or database itself.

Also provides a DB-API 2.0 connection proxying mechanism allowing
regular DB-API connect() methods to be transparently managed by a
SQLAlchemy connection pool.
"""

from . import events
from .base import _AdhocProxiedConnection as _AdhocProxiedConnection
from .base import _ConnectionFairy as _ConnectionFairy
from .base import _ConnectionRecord
from .base import _CreatorFnType as _CreatorFnType
from .base import _CreatorWRecFnType as _CreatorWRecFnType
from .base import _finalize_fairy
from .base import _ResetStyleArgType as _ResetStyleArgType
from .base import ConnectionPoolEntry as ConnectionPoolEntry
from .base import ManagesConnection as ManagesConnection
from .base import Pool as Pool
from .base import PoolProxiedConnection as PoolProxiedConnection
from .base import PoolResetState as PoolResetState
from .base import reset_commit as reset_commit
from .base import reset_none as reset_none
from .base import reset_rollback as reset_rollback
from .impl import AssertionPool as AssertionPool
from .impl import AsyncAdaptedQueuePool as AsyncAdaptedQueuePool
from .impl import (
    FallbackAsyncAdaptedQueuePool as FallbackAsyncAdaptedQueuePool,
)
from .impl import NullPool as NullPool
from .impl import QueuePool as QueuePool
from .impl import SingletonThreadPool as SingletonThreadPool
from .impl import StaticPool as StaticPool

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_query_eval.py ===
import operator

import numpy as np
import pytest

from pandas.errors import (
    NumExprClobberingError,
    UndefinedVariableError,
)
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED


@pytest.fixture(params=["python", "pandas"], ids=lambda x: x)
def parser(request):
    return request.param


@pytest.fixture(
    params=["python", pytest.param("numexpr", marks=td.skip_if_no("numexpr"))],
    ids=lambda x: x,
)
def engine(request):
    return request.param


def skip_if_no_pandas_parser(parser):
    if parser != "pandas":
        pytest.skip(f"cannot evaluate with parser={parser}")


class TestCompat:
    @pytest.fixture
    def df(self):
        return DataFrame({"A": [1, 2, 3]})

    @pytest.fixture
    def expected1(self, df):
        return df[df.A > 0]

    @pytest.fixture
    def expected2(self, df):
        return df.A + 1

    def test_query_default(self, df, expected1, expected2):
        # GH 12749
        # this should always work, whether NUMEXPR_INSTALLED or not
        result = df.query("A>0")
        tm.assert_frame_equal(result, expected1)
        result = df.eval("A+1")
        tm.assert_series_equal(result, expected2, check_names=False)

    def test_query_None(self, df, expected1, expected2):
        result = df.query("A>0", engine=None)
        tm.assert_frame_equal(result, expected1)
        result = df.eval("A+1", engine=None)
        tm.assert_series_equal(result, expected2, check_names=False)

    def test_query_python(self, df, expected1, expected2):
        result = df.query("A>0", engine="python")
        tm.assert_frame_equal(result, expected1)
        result = df.eval("A+1", engine="python")
        tm.assert_series_equal(result, expected2, check_names=False)

    def test_query_numexpr(self, df, expected1, expected2):
        if NUMEXPR_INSTALLED:
            result = df.query("A>0", engine="numexpr")
            tm.assert_frame_equal(result, expected1)
            result = df.eval("A+1", engine="numexpr")
            tm.assert_series_equal(result, expected2, check_names=False)
        else:
            msg = (
                r"'numexpr' is not installed or an unsupported version. "
                r"Cannot use engine='numexpr' for query/eval if 'numexpr' is "
                r"not installed"
            )
            with pytest.raises(ImportError, match=msg):
                df.query("A>0", engine="numexpr")
            with pytest.raises(ImportError, match=msg):
                df.eval("A+1", engine="numexpr")


class TestDataFrameEval:
    # smaller hits python, larger hits numexpr
    @pytest.mark.parametrize("n", [4, 4000])
    @pytest.mark.parametrize(
        "op_str,op,rop",
        [
            ("+", "__add__", "__radd__"),
            ("-", "__sub__", "__rsub__"),
            ("*", "__mul__", "__rmul__"),
            ("/", "__truediv__", "__rtruediv__"),
        ],
    )
    def test_ops(self, op_str, op, rop, n):
        # tst ops and reversed ops in evaluation
        # GH7198

        df = DataFrame(1, index=range(n), columns=list("abcd"))
        df.iloc[0] = 2
        m = df.mean()

        base = DataFrame(  # noqa: F841
            np.tile(m.values, n).reshape(n, -1), columns=list("abcd")
        )

        expected = eval(f"base {op_str} df")

        # ops as strings
        result = eval(f"m {op_str} df")
        tm.assert_frame_equal(result, expected)

        # these are commutative
        if op in ["+", "*"]:
            result = getattr(df, op)(m)
            tm.assert_frame_equal(result, expected)

        # these are not
        elif op in ["-", "/"]:
            result = getattr(df, rop)(m)
            tm.assert_frame_equal(result, expected)

    def test_dataframe_sub_numexpr_path(self):
        # GH7192: Note we need a large number of rows to ensure this
        #  goes through the numexpr path
        df = DataFrame({"A": np.random.default_rng(2).standard_normal(25000)})
        df.iloc[0:5] = np.nan
        expected = 1 - np.isnan(df.iloc[0:25])
        result = (1 - np.isnan(df)).iloc[0:25]
        tm.assert_frame_equal(result, expected)

    def test_query_non_str(self):
        # GH 11485
        df = DataFrame({"A": [1, 2, 3], "B": ["a", "b", "b"]})

        msg = "expr must be a string to be evaluated"
        with pytest.raises(ValueError, match=msg):
            df.query(lambda x: x.B == "b")

        with pytest.raises(ValueError, match=msg):
            df.query(111)

    def test_query_empty_string(self):
        # GH 13139
        df = DataFrame({"A": [1, 2, 3]})

        msg = "expr cannot be an empty string"
        with pytest.raises(ValueError, match=msg):
            df.query("")

    def test_eval_resolvers_as_list(self):
        # GH 14095
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)), columns=list("ab")
        )
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        assert df.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]
        assert pd.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]

    def test_eval_resolvers_combined(self):
        # GH 34966
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)), columns=list("ab")
        )
        dict1 = {"c": 2}

        # Both input and default index/column resolvers should be usable
        result = df.eval("a + b * c", resolvers=[dict1])

        expected = df["a"] + df["b"] * dict1["c"]
        tm.assert_series_equal(result, expected)

    def test_eval_object_dtype_binop(self):
        # GH#24883
        df = DataFrame({"a1": ["Y", "N"]})
        res = df.eval("c = ((a1 == 'Y') & True)")
        expected = DataFrame({"a1": ["Y", "N"], "c": [True, False]})
        tm.assert_frame_equal(res, expected)

    def test_extension_array_eval(self, engine, parser, request):
        # GH#58748
        if engine == "numexpr":
            mark = pytest.mark.xfail(
                reason="numexpr does not support extension array dtypes"
            )
            request.applymarker(mark)
        df = DataFrame({"a": pd.array([1, 2, 3]), "b": pd.array([4, 5, 6])})
        result = df.eval("a / b", engine=engine, parser=parser)
        expected = Series(pd.array([0.25, 0.40, 0.50]))
        tm.assert_series_equal(result, expected)

    def test_complex_eval(self, engine, parser):
        # GH#21374
        df = DataFrame({"a": [1 + 2j], "b": [1 + 1j]})
        result = df.eval("a/b", engine=engine, parser=parser)
        expected = Series([1.5 + 0.5j])
        tm.assert_series_equal(result, expected)


class TestDataFrameQueryWithMultiIndex:
    def test_query_with_named_multiindex(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.random.default_rng(2).choice(["eggs", "ham"], size=10)
        index = MultiIndex.from_arrays([a, b], names=["color", "food"])
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind = Series(
            df.index.get_level_values("color").values, index=index, name="color"
        )

        # equality
        res1 = df.query('color == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == color', parser=parser, engine=engine)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('color != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != color', parser=parser, engine=engine)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('color == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('color != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["red"] in color', parser=parser, engine=engine)
        res2 = df.query('"red" in color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["red"] not in color', parser=parser, engine=engine)
        res2 = df.query('"red" not in color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_unnamed_multiindex(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.random.default_rng(2).choice(["eggs", "ham"], size=10)
        index = MultiIndex.from_arrays([a, b])
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        ind = Series(df.index.get_level_values(0).values, index=index)

        res1 = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == ilevel_0', parser=parser, engine=engine)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != ilevel_0', parser=parser, engine=engine)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('ilevel_0 == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('ilevel_0 != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["red"] in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" in ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["red"] not in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" not in ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # ## LEVEL 1
        ind = Series(df.index.get_level_values(1).values, index=index)
        res1 = df.query('ilevel_1 == "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" == ilevel_1', parser=parser, engine=engine)
        exp = df[ind == "eggs"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('ilevel_1 != "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" != ilevel_1', parser=parser, engine=engine)
        exp = df[ind != "eggs"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('ilevel_1 == ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] == ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('ilevel_1 != ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] != ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["eggs"] in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" in ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["eggs"] not in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" not in ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_partially_named_multiindex(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.arange(10)
        index = MultiIndex.from_arrays([a, b])
        index.names = [None, "rating"]
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        res = df.query("rating == 1", parser=parser, engine=engine)
        ind = Series(
            df.index.get_level_values("rating").values, index=index, name="rating"
        )
        exp = df[ind == 1]
        tm.assert_frame_equal(res, exp)

        res = df.query("rating != 1", parser=parser, engine=engine)
        ind = Series(
            df.index.get_level_values("rating").values, index=index, name="rating"
        )
        exp = df[ind != 1]
        tm.assert_frame_equal(res, exp)

        res = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        ind = Series(df.index.get_level_values(0).values, index=index)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res, exp)

        res = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        ind = Series(df.index.get_level_values(0).values, index=index)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res, exp)

    def test_query_multiindex_get_index_resolvers(self):
        df = DataFrame(
            np.ones((10, 3)),
            index=MultiIndex.from_arrays(
                [range(10) for _ in range(2)], names=["spam", "eggs"]
            ),
        )
        resolvers = df._get_index_resolvers()

        def to_series(mi, level):
            level_values = mi.get_level_values(level)
            s = level_values.to_series()
            s.index = mi
            return s

        col_series = df.columns.to_series()
        expected = {
            "index": df.index,
            "columns": col_series,
            "spam": to_series(df.index, "spam"),
            "eggs": to_series(df.index, "eggs"),
            "clevel_0": col_series,
        }
        for k, v in resolvers.items():
            if isinstance(v, Index):
                assert v.is_(expected[k])
            elif isinstance(v, Series):
                tm.assert_series_equal(v, expected[k])
            else:
                raise AssertionError("object must be a Series or Index")


@td.skip_if_no("numexpr")
class TestDataFrameQueryNumExprPandas:
    @pytest.fixture
    def engine(self):
        return "numexpr"

    @pytest.fixture
    def parser(self):
        return "pandas"

    def test_date_query_with_attribute_access(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        res = df.query(
            "@df.dates1 < 20130101 < @df.dates3", engine=engine, parser=parser
        )
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_no_attribute_access(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates2"] = date_range("1/1/2013", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates3"] = pd.NaT
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query("index < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self, engine, parser):
        n = 10
        # Cast to object to avoid implicit cast when setting entry to pd.NaT below
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype(
            {0: object}
        )
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query("index < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self, engine, parser):
        n = 10
        d = {}
        d["dates1"] = date_range("1/1/2012", periods=n)
        d["dates3"] = date_range("1/1/2014", periods=n)
        df = DataFrame(d)
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.index.to_series() < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_non_date(self, engine, parser):
        n = 10
        df = DataFrame(
            {"dates": date_range("1/1/2012", periods=n), "nondate": np.arange(n)}
        )

        result = df.query("dates == nondate", parser=parser, engine=engine)
        assert len(result) == 0

        result = df.query("dates != nondate", parser=parser, engine=engine)
        tm.assert_frame_equal(result, df)

        msg = r"Invalid comparison between dtype=datetime64\[ns\] and ndarray"
        for op in ["<", ">", "<=", ">="]:
            with pytest.raises(TypeError, match=msg):
                df.query(f"dates {op} nondate", parser=parser, engine=engine)

    def test_query_syntax_error(self, engine, parser):
        df = DataFrame({"i": range(10), "+": range(3, 13), "r": range(4, 14)})
        msg = "invalid syntax"
        with pytest.raises(SyntaxError, match=msg):
            df.query("i - +", engine=engine, parser=parser)

    def test_query_scope(self, engine, parser):
        skip_if_no_pandas_parser(parser)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 2)), columns=list("ab")
        )

        a, b = 1, 2  # noqa: F841
        res = df.query("a > b", engine=engine, parser=parser)
        expected = df[df.a > df.b]
        tm.assert_frame_equal(res, expected)

        res = df.query("@a > b", engine=engine, parser=parser)
        expected = df[a > df.b]
        tm.assert_frame_equal(res, expected)

        # no local variable c
        with pytest.raises(
            UndefinedVariableError, match="local variable 'c' is not defined"
        ):
            df.query("@a > b > @c", engine=engine, parser=parser)

        # no column named 'c'
        with pytest.raises(UndefinedVariableError, match="name 'c' is not defined"):
            df.query("@a > b > c", engine=engine, parser=parser)

    def test_query_doesnt_pickup_local(self, engine, parser):
        n = m = 10
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        # we don't pick up the local 'sin'
        with pytest.raises(UndefinedVariableError, match="name 'sin' is not defined"):
            df.query("sin > 5", engine=engine, parser=parser)

    def test_query_builtin(self, engine, parser):
        n = m = 10
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        df.index.name = "sin"
        msg = "Variables in expression.+"
        with pytest.raises(NumExprClobberingError, match=msg):
            df.query("sin > 5", engine=engine, parser=parser)

    def test_query(self, engine, parser):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )

        tm.assert_frame_equal(
            df.query("a < b", engine=engine, parser=parser), df[df.a < df.b]
        )
        tm.assert_frame_equal(
            df.query("a + b > b * c", engine=engine, parser=parser),
            df[df.a + df.b > df.b * df.c],
        )

    def test_query_index_with_name(self, engine, parser):
        df = DataFrame(
            np.random.default_rng(2).integers(10, size=(10, 3)),
            index=Index(range(10), name="blob"),
            columns=["a", "b", "c"],
        )
        res = df.query("(blob < 5) & (a < b)", engine=engine, parser=parser)
        expec = df[(df.index < 5) & (df.a < df.b)]
        tm.assert_frame_equal(res, expec)

        res = df.query("blob < b", engine=engine, parser=parser)
        expec = df[df.index < df.b]

        tm.assert_frame_equal(res, expec)

    def test_query_index_without_name(self, engine, parser):
        df = DataFrame(
            np.random.default_rng(2).integers(10, size=(10, 3)),
            index=range(10),
            columns=["a", "b", "c"],
        )

        # "index" should refer to the index
        res = df.query("index < b", engine=engine, parser=parser)
        expec = df[df.index < df.b]
        tm.assert_frame_equal(res, expec)

        # test against a scalar
        res = df.query("index < 5", engine=engine, parser=parser)
        expec = df[df.index < 5]
        tm.assert_frame_equal(res, expec)

    def test_nested_scope(self, engine, parser):
        skip_if_no_pandas_parser(parser)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        expected = df[(df > 0) & (df2 > 0)]

        result = df.query("(@df > 0) & (@df2 > 0)", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

        result = pd.eval("df[df > 0 and df2 > 0]", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

        result = pd.eval(
            "df[df > 0 and df2 > 0 and df[df > 0] > 0]", engine=engine, parser=parser
        )
        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        tm.assert_frame_equal(result, expected)

        result = pd.eval("df[(df>0) & (df2>0)]", engine=engine, parser=parser)
        expected = df.query("(@df>0) & (@df2>0)", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

    def test_nested_raises_on_local_self_reference(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))

        # can't reference ourself b/c we're a local so @ is necessary
        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query("df > 0", engine=engine, parser=parser)

    def test_local_syntax(self, engine, parser):
        skip_if_no_pandas_parser(parser)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((100, 10)),
            columns=list("abcdefghij"),
        )
        b = 1
        expect = df[df.a < b]
        result = df.query("a < @b", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)

        expect = df[df.a < df.b]
        result = df.query("a < b", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)

    def test_chained_cmp_and_in(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        cols = list("abc")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((100, len(cols))), columns=cols
        )
        res = df.query(
            "a < b < c and a not in b not in c", engine=engine, parser=parser
        )
        ind = (df.a < df.b) & (df.b < df.c) & ~df.b.isin(df.a) & ~df.c.isin(df.b)
        expec = df[ind]
        tm.assert_frame_equal(res, expec)

    def test_local_variable_with_in(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        a = Series(np.random.default_rng(2).integers(3, size=15), name="a")
        b = Series(np.random.default_rng(2).integers(10, size=15), name="b")
        df = DataFrame({"a": a, "b": b})

        expected = df.loc[(df.b - 1).isin(a)]
        result = df.query("b - 1 in a", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

        b = Series(np.random.default_rng(2).integers(10, size=15), name="b")
        expected = df.loc[(b - 1).isin(a)]
        result = df.query("@b - 1 in a", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

    def test_at_inside_string(self, engine, parser):
        skip_if_no_pandas_parser(parser)
        c = 1  # noqa: F841
        df = DataFrame({"a": ["a", "a", "b", "b", "@c", "@c"]})
        result = df.query('a == "@c"', engine=engine, parser=parser)
        expected = df[df.a == "@c"]
        tm.assert_frame_equal(result, expected)

    def test_query_undefined_local(self):
        engine, parser = self.engine, self.parser
        skip_if_no_pandas_parser(parser)

        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list("ab"))
        with pytest.raises(
            UndefinedVariableError, match="local variable 'c' is not defined"
        ):
            df.query("a == @c", engine=engine, parser=parser)

    def test_index_resolvers_come_after_columns_with_the_same_name(
        self, engine, parser
    ):
        n = 1  # noqa: F841
        a = np.r_[20:101:20]

        df = DataFrame(
            {"index": a, "b": np.random.default_rng(2).standard_normal(a.size)}
        )
        df.index.name = "index"
        result = df.query("index > 5", engine=engine, parser=parser)
        expected = df[df["index"] > 5]
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {"index": a, "b": np.random.default_rng(2).standard_normal(a.size)}
        )
        result = df.query("ilevel_0 > 5", engine=engine, parser=parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": a, "b": np.random.default_rng(2).standard_normal(a.size)})
        df.index.name = "a"
        result = df.query("a > 5", engine=engine, parser=parser)
        expected = df[df.a > 5]
        tm.assert_frame_equal(result, expected)

        result = df.query("index > 5", engine=engine, parser=parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op, f", [["==", operator.eq], ["!=", operator.ne]])
    def test_inf(self, op, f, engine, parser):
        n = 10
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(n),
                "b": np.random.default_rng(2).random(n),
            }
        )
        df.loc[::2, 0] = np.inf
        q = f"a {op} inf"
        expected = df[f(df.a, np.inf)]
        result = df.query(q, engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

    def test_check_tz_aware_index_query(self, tz_aware_fixture):
        # https://github.com/pandas-dev/pandas/issues/29463
        tz = tz_aware_fixture
        df_index = date_range(
            start="2019-01-01", freq="1d", periods=10, tz=tz, name="time"
        )
        expected = DataFrame(index=df_index)
        df = DataFrame(index=df_index)
        result = df.query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(df_index)
        expected.columns = expected.columns.astype(object)
        result = df.reset_index().query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)

    def test_method_calls_in_query(self, engine, parser):
        # https://github.com/pandas-dev/pandas/issues/22435
        n = 10
        df = DataFrame(
            {
                "a": 2 * np.random.default_rng(2).random(n),
                "b": np.random.default_rng(2).random(n),
            }
        )
        expected = df[df["a"].astype("int") == 0]
        result = df.query("a.astype('int') == 0", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {
                "a": np.where(
                    np.random.default_rng(2).random(n) < 0.5,
                    np.nan,
                    np.random.default_rng(2).standard_normal(n),
                ),
                "b": np.random.default_rng(2).standard_normal(n),
            }
        )
        expected = df[df["a"].notnull()]
        result = df.query("a.notnull()", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numexpr")
class TestDataFrameQueryNumExprPython(TestDataFrameQueryNumExprPandas):
    @pytest.fixture
    def engine(self):
        return "numexpr"

    @pytest.fixture
    def parser(self):
        return "python"

    def test_date_query_no_attribute_access(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        res = df.query(
            "(dates1 < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates2"] = date_range("1/1/2013", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates3"] = pd.NaT
        res = df.query(
            "(dates1 < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query(
            "(index < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self, engine, parser):
        n = 10
        # Cast to object to avoid implicit cast when setting entry to pd.NaT below
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype(
            {0: object}
        )
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query(
            "(index < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        msg = r"'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query("index < 20130101 < dates3", engine=engine, parser=parser)

    def test_nested_scope(self, engine, parser):
        # smoke test
        x = 1  # noqa: F841
        result = pd.eval("x + 1", engine=engine, parser=parser)
        assert result == 2

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))

        # don't have the pandas parser
        msg = r"The '@' prefix is only supported by the pandas parser"
        with pytest.raises(SyntaxError, match=msg):
            df.query("(@df>0) & (@df2>0)", engine=engine, parser=parser)

        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query("(df>0) & (df2>0)", engine=engine, parser=parser)

        expected = df[(df > 0) & (df2 > 0)]
        result = pd.eval("df[(df > 0) & (df2 > 0)]", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        result = pd.eval(
            "df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]", engine=engine, parser=parser
        )
        tm.assert_frame_equal(expected, result)

    def test_query_numexpr_with_min_and_max_columns(self):
        df = DataFrame({"min": [1, 2, 3], "max": [4, 5, 6]})
        regex_to_match = (
            r"Variables in expression \"\(min\) == \(1\)\" "
            r"overlap with builtins: \('min'\)"
        )
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            df.query("min == 1")

        regex_to_match = (
            r"Variables in expression \"\(max\) == \(1\)\" "
            r"overlap with builtins: \('max'\)"
        )
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            df.query("max == 1")


class TestDataFrameQueryPythonPandas(TestDataFrameQueryNumExprPandas):
    @pytest.fixture
    def engine(self):
        return "python"

    @pytest.fixture
    def parser(self):
        return "pandas"

    def test_query_builtin(self, engine, parser):
        n = m = 10
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        df.index.name = "sin"
        expected = df[df.index > 5]
        result = df.query("sin > 5", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)


class TestDataFrameQueryPythonPython(TestDataFrameQueryNumExprPython):
    @pytest.fixture
    def engine(self):
        return "python"

    @pytest.fixture
    def parser(self):
        return "python"

    def test_query_builtin(self, engine, parser):
        n = m = 10
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        df.index.name = "sin"
        expected = df[df.index > 5]
        result = df.query("sin > 5", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)


class TestDataFrameQueryStrings:
    def test_str_query_method(self, parser, engine):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=["b"])
        df["strings"] = Series(list("aabbccddee"))
        expect = df[df.strings == "a"]

        if parser != "pandas":
            col = "strings"
            lst = '"a"'

            lhs = [col] * 2 + [lst] * 2
            rhs = lhs[::-1]

            eq, ne = "==", "!="
            ops = 2 * ([eq] + [ne])
            msg = r"'(Not)?In' nodes are not implemented"

            for lhs, op, rhs in zip(lhs, ops, rhs):
                ex = f"{lhs} {op} {rhs}"
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(
                        ex,
                        engine=engine,
                        parser=parser,
                        local_dict={"strings": df.strings},
                    )
        else:
            res = df.query('"a" == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('strings == "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            tm.assert_frame_equal(res, df[df.strings.isin(["a"])])

            expect = df[df.strings != "a"]
            res = df.query('strings != "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('"a" != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            tm.assert_frame_equal(res, df[~df.strings.isin(["a"])])

    def test_str_list_query_method(self, parser, engine):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=["b"])
        df["strings"] = Series(list("aabbccddee"))
        expect = df[df.strings.isin(["a", "b"])]

        if parser != "pandas":
            col = "strings"
            lst = '["a", "b"]'

            lhs = [col] * 2 + [lst] * 2
            rhs = lhs[::-1]

            eq, ne = "==", "!="
            ops = 2 * ([eq] + [ne])
            msg = r"'(Not)?In' nodes are not implemented"

            for lhs, op, rhs in zip(lhs, ops, rhs):
                ex = f"{lhs} {op} {rhs}"
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(ex, engine=engine, parser=parser)
        else:
            res = df.query('strings == ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('["a", "b"] == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            expect = df[~df.strings.isin(["a", "b"])]

            res = df.query('strings != ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('["a", "b"] != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

    def test_query_with_string_columns(self, parser, engine):
        df = DataFrame(
            {
                "a": list("aaaabbbbcccc"),
                "b": list("aabbccddeeff"),
                "c": np.random.default_rng(2).integers(5, size=12),
                "d": np.random.default_rng(2).integers(9, size=12),
            }
        )
        if parser == "pandas":
            res = df.query("a in b", parser=parser, engine=engine)
            expec = df[df.a.isin(df.b)]
            tm.assert_frame_equal(res, expec)

            res = df.query("a in b and c < d", parser=parser, engine=engine)
            expec = df[df.a.isin(df.b) & (df.c < df.d)]
            tm.assert_frame_equal(res, expec)
        else:
            msg = r"'(Not)?In' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query("a in b", parser=parser, engine=engine)

            msg = r"'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query("a in b and c < d", parser=parser, engine=engine)

    def test_object_array_eq_ne(self, parser, engine):
        df = DataFrame(
            {
                "a": list("aaaabbbbcccc"),
                "b": list("aabbccddeeff"),
                "c": np.random.default_rng(2).integers(5, size=12),
                "d": np.random.default_rng(2).integers(9, size=12),
            }
        )
        res = df.query("a == b", parser=parser, engine=engine)
        exp = df[df.a == df.b]
        tm.assert_frame_equal(res, exp)

        res = df.query("a != b", parser=parser, engine=engine)
        exp = df[df.a != df.b]
        tm.assert_frame_equal(res, exp)

    def test_query_with_nested_strings(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        events = [
            f"page {n} {act}" for n in range(1, 4) for act in ["load", "exit"]
        ] * 2
        stamps1 = date_range("2014-01-01 0:00:01", freq="30s", periods=6)
        stamps2 = date_range("2014-02-01 1:00:01", freq="30s", periods=6)
        df = DataFrame(
            {
                "id": np.arange(1, 7).repeat(2),
                "event": events,
                "timestamp": stamps1.append(stamps2),
            }
        )

        expected = df[df.event == '"page 1 load"']
        res = df.query("""'"page 1 load"' in event""", parser=parser, engine=engine)
        tm.assert_frame_equal(expected, res)

    def test_query_with_nested_special_character(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        df = DataFrame({"a": ["a", "b", "test & test"], "b": [1, 2, 3]})
        res = df.query('a == "test & test"', parser=parser, engine=engine)
        expec = df[df.a == "test & test"]
        tm.assert_frame_equal(res, expec)

    @pytest.mark.parametrize(
        "op, func",
        [
            ["<", operator.lt],
            [">", operator.gt],
            ["<=", operator.le],
            [">=", operator.ge],
        ],
    )
    def test_query_lex_compare_strings(self, parser, engine, op, func):
        a = Series(np.random.default_rng(2).choice(list("abcde"), 20))
        b = Series(np.arange(a.size))
        df = DataFrame({"X": a, "Y": b})

        res = df.query(f'X {op} "d"', engine=engine, parser=parser)
        expected = df[func(df.X, "d")]
        tm.assert_frame_equal(res, expected)

    def test_query_single_element_booleans(self, parser, engine):
        columns = "bid", "bidsize", "ask", "asksize"
        data = np.random.default_rng(2).integers(2, size=(1, len(columns))).astype(bool)
        df = DataFrame(data, columns=columns)
        res = df.query("bid & ask", engine=engine, parser=parser)
        expected = df[df.bid & df.ask]
        tm.assert_frame_equal(res, expected)

    def test_query_string_scalar_variable(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        df = DataFrame(
            {
                "Symbol": ["BUD US", "BUD US", "IBM US", "IBM US"],
                "Price": [109.70, 109.72, 183.30, 183.35],
            }
        )
        e = df[df.Symbol == "BUD US"]
        symb = "BUD US"  # noqa: F841
        r = df.query("Symbol == @symb", parser=parser, engine=engine)
        tm.assert_frame_equal(e, r)

    @pytest.mark.parametrize(
        "in_list",
        [
            [None, "asdf", "ghjk"],
            ["asdf", None, "ghjk"],
            ["asdf", "ghjk", None],
            [None, None, "asdf"],
            ["asdf", None, None],
            [None, None, None],
        ],
    )
    def test_query_string_null_elements(self, in_list):
        # GITHUB ISSUE #31516
        parser = "pandas"
        engine = "python"
        expected = {i: value for i, value in enumerate(in_list) if value == "asdf"}

        df_expected = DataFrame({"a": expected}, dtype="string")
        df_expected.index = df_expected.index.astype("int64")
        df = DataFrame({"a": in_list}, dtype="string")
        res1 = df.query("a == 'asdf'", parser=parser, engine=engine)
        res2 = df[df["a"] == "asdf"]
        res3 = df.query("a <= 'asdf'", parser=parser, engine=engine)
        tm.assert_frame_equal(res1, df_expected)
        tm.assert_frame_equal(res1, res2)
        tm.assert_frame_equal(res1, res3)
        tm.assert_frame_equal(res2, res3)


class TestDataFrameEvalWithFrame:
    @pytest.fixture
    def frame(self):
        return DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )

    def test_simple_expr(self, frame, parser, engine):
        res = frame.eval("a + b", engine=engine, parser=parser)
        expect = frame.a + frame.b
        tm.assert_series_equal(res, expect)

    def test_bool_arith_expr(self, frame, parser, engine):
        res = frame.eval("a[a < 1] + b", engine=engine, parser=parser)
        expect = frame.a[frame.a < 1] + frame.b
        tm.assert_series_equal(res, expect)

    @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    def test_invalid_type_for_operator_raises(self, parser, engine, op):
        df = DataFrame({"a": [1, 2], "b": ["c", "d"]})
        msg = r"unsupported operand type\(s\) for .+: '.+' and '.+'|Cannot"

        with pytest.raises(TypeError, match=msg):
            df.eval(f"a {op} b", engine=engine, parser=parser)


class TestDataFrameQueryBacktickQuoting:
    @pytest.fixture
    def df(self):
        """
        Yields a dataframe with strings that may or may not need escaping
        by backticks. The last two columns cannot be escaped by backticks
        and should raise a ValueError.
        """
        yield DataFrame(
            {
                "A": [1, 2, 3],
                "B B": [3, 2, 1],
                "C C": [4, 5, 6],
                "C  C": [7, 4, 3],
                "C_C": [8, 9, 10],
                "D_D D": [11, 1, 101],
                "E.E": [6, 3, 5],
                "F-F": [8, 1, 10],
                "1e1": [2, 4, 8],
                "def": [10, 11, 2],
                "A (x)": [4, 1, 3],
                "B(x)": [1, 1, 5],
                "B (x)": [2, 7, 4],
                "  &^ :!€$?(} >    <++*''  ": [2, 5, 6],
                "": [10, 11, 1],
                " A": [4, 7, 9],
                "  ": [1, 2, 1],
                "it's": [6, 3, 1],
                "that's": [9, 1, 8],
                "☺": [8, 7, 6],
                "foo#bar": [2, 4, 5],
                1: [5, 7, 9],
            }
        )

    def test_single_backtick_variable_query(self, df):
        res = df.query("1 < `B B`")
        expect = df[1 < df["B B"]]
        tm.assert_frame_equal(res, expect)

    def test_two_backtick_variables_query(self, df):
        res = df.query("1 < `B B` and 4 < `C C`")
        expect = df[(1 < df["B B"]) & (4 < df["C C"])]
        tm.assert_frame_equal(res, expect)

    def test_single_backtick_variable_expr(self, df):
        res = df.eval("A + `B B`")
        expect = df["A"] + df["B B"]
        tm.assert_series_equal(res, expect)

    def test_two_backtick_variables_expr(self, df):
        res = df.eval("`B B` + `C C`")
        expect = df["B B"] + df["C C"]
        tm.assert_series_equal(res, expect)

    def test_already_underscore_variable(self, df):
        res = df.eval("`C_C` + A")
        expect = df["C_C"] + df["A"]
        tm.assert_series_equal(res, expect)

    def test_same_name_but_underscores(self, df):
        res = df.eval("C_C + `C C`")
        expect = df["C_C"] + df["C C"]
        tm.assert_series_equal(res, expect)

    def test_mixed_underscores_and_spaces(self, df):
        res = df.eval("A + `D_D D`")
        expect = df["A"] + df["D_D D"]
        tm.assert_series_equal(res, expect)

    def test_backtick_quote_name_with_no_spaces(self, df):
        res = df.eval("A + `C_C`")
        expect = df["A"] + df["C_C"]
        tm.assert_series_equal(res, expect)

    def test_special_characters(self, df):
        res = df.eval("`E.E` + `F-F` - A")
        expect = df["E.E"] + df["F-F"] - df["A"]
        tm.assert_series_equal(res, expect)

    def test_start_with_digit(self, df):
        res = df.eval("A + `1e1`")
        expect = df["A"] + df["1e1"]
        tm.assert_series_equal(res, expect)

    def test_keyword(self, df):
        res = df.eval("A + `def`")
        expect = df["A"] + df["def"]
        tm.assert_series_equal(res, expect)

    def test_unneeded_quoting(self, df):
        res = df.query("`A` > 2")
        expect = df[df["A"] > 2]
        tm.assert_frame_equal(res, expect)

    def test_parenthesis(self, df):
        res = df.query("`A (x)` > 2")
        expect = df[df["A (x)"] > 2]
        tm.assert_frame_equal(res, expect)

    def test_empty_string(self, df):
        res = df.query("`` > 5")
        expect = df[df[""] > 5]
        tm.assert_frame_equal(res, expect)

    def test_multiple_spaces(self, df):
        res = df.query("`C  C` > 5")
        expect = df[df["C  C"] > 5]
        tm.assert_frame_equal(res, expect)

    def test_start_with_spaces(self, df):
        res = df.eval("` A` + `  `")
        expect = df[" A"] + df["  "]
        tm.assert_series_equal(res, expect)

    def test_lots_of_operators_string(self, df):
        res = df.query("`  &^ :!€$?(} >    <++*''  ` > 4")
        expect = df[df["  &^ :!€$?(} >    <++*''  "] > 4]
        tm.assert_frame_equal(res, expect)

    def test_missing_attribute(self, df):
        message = "module 'pandas' has no attribute 'thing'"
        with pytest.raises(AttributeError, match=message):
            df.eval("@pd.thing")

    def test_failing_quote(self, df):
        msg = r"(Could not convert ).*( to a valid Python identifier.)"
        with pytest.raises(SyntaxError, match=msg):
            df.query("`it's` > `that's`")

    def test_failing_character_outside_range(self, df):
        msg = r"(Could not convert ).*( to a valid Python identifier.)"
        with pytest.raises(SyntaxError, match=msg):
            df.query("`☺` > 4")

    def test_failing_hashtag(self, df):
        msg = "Failed to parse backticks"
        with pytest.raises(SyntaxError, match=msg):
            df.query("`foo#bar` > 4")

    def test_call_non_named_expression(self, df):
        """
        Only attributes and variables ('named functions') can be called.
        .__call__() is not an allowed attribute because that would allow
        calling anything.
        https://github.com/pandas-dev/pandas/pull/32460
        """

        def func(*_):
            return 1

        funcs = [func]  # noqa: F841

        df.eval("@func()")

        with pytest.raises(TypeError, match="Only named functions are supported"):
            df.eval("@funcs[0]()")

        with pytest.raises(TypeError, match="Only named functions are supported"):
            df.eval("@funcs[0].__call__()")

    def test_ea_dtypes(self, any_numeric_ea_and_arrow_dtype):
        # GH#29618
        df = DataFrame(
            [[1, 2], [3, 4]], columns=["a", "b"], dtype=any_numeric_ea_and_arrow_dtype
        )
        warning = RuntimeWarning if NUMEXPR_INSTALLED else None
        with tm.assert_produces_warning(warning):
            result = df.eval("c = b - a")
        expected = DataFrame(
            [[1, 2, 1], [3, 4, 1]],
            columns=["a", "b", "c"],
            dtype=any_numeric_ea_and_arrow_dtype,
        )
        tm.assert_frame_equal(result, expected)

    def test_ea_dtypes_and_scalar(self):
        # GH#29618
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"], dtype="Float64")
        warning = RuntimeWarning if NUMEXPR_INSTALLED else None
        with tm.assert_produces_warning(warning):
            result = df.eval("c = b - 1")
        expected = DataFrame(
            [[1, 2, 1], [3, 4, 3]], columns=["a", "b", "c"], dtype="Float64"
        )
        tm.assert_frame_equal(result, expected)

    def test_ea_dtypes_and_scalar_operation(self, any_numeric_ea_and_arrow_dtype):
        # GH#29618
        df = DataFrame(
            [[1, 2], [3, 4]], columns=["a", "b"], dtype=any_numeric_ea_and_arrow_dtype
        )
        result = df.eval("c = 2 - 1")
        expected = DataFrame(
            {
                "a": Series([1, 3], dtype=any_numeric_ea_and_arrow_dtype),
                "b": Series([2, 4], dtype=any_numeric_ea_and_arrow_dtype),
                "c": Series([1, 1], dtype=result["c"].dtype),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["int64", "Int64", "int64[pyarrow]"])
    def test_query_ea_dtypes(self, dtype):
        if dtype == "int64[pyarrow]":
            pytest.importorskip("pyarrow")
        # GH#50261
        df = DataFrame({"a": Series([1, 2], dtype=dtype)})
        ref = {2}  # noqa: F841
        warning = RuntimeWarning if dtype == "Int64" and NUMEXPR_INSTALLED else None
        with tm.assert_produces_warning(warning):
            result = df.query("a in @ref")
        expected = DataFrame({"a": Series([2], dtype=dtype, index=[1])})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("engine", ["python", "numexpr"])
    @pytest.mark.parametrize("dtype", ["int64", "Int64", "int64[pyarrow]"])
    def test_query_ea_equality_comparison(self, dtype, engine):
        # GH#50261
        warning = RuntimeWarning if engine == "numexpr" else None
        if engine == "numexpr" and not NUMEXPR_INSTALLED:
            pytest.skip("numexpr not installed")
        if dtype == "int64[pyarrow]":
            pytest.importorskip("pyarrow")
        df = DataFrame(
            {"A": Series([1, 1, 2], dtype="Int64"), "B": Series([1, 2, 2], dtype=dtype)}
        )
        with tm.assert_produces_warning(warning):
            result = df.query("A == B", engine=engine)
        expected = DataFrame(
            {
                "A": Series([1, 2], dtype="Int64", index=[0, 2]),
                "B": Series([1, 2], dtype=dtype, index=[0, 2]),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_all_nat_in_object(self):
        # GH#57068
        now = pd.Timestamp.now("UTC")  # noqa: F841
        df = DataFrame({"a": pd.to_datetime([None, None], utc=True)}, dtype=object)
        result = df.query("a > @now")
        expected = DataFrame({"a": []}, dtype=object)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\tests\test_domains.py ===
"""Tests for classes defining properties of ground domains, e.g. ZZ, QQ, ZZ[x] ... """

from sympy.external.gmpy import GROUND_TYPES

from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Integer,
    Rational, oo, pi, _illegal)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.abc import x, y, z

from sympy.polys.domains import (ZZ, QQ, RR, CC, FF, GF, EX, EXRAW, ZZ_gmpy,
    ZZ_python, QQ_gmpy, QQ_python)
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.gaussiandomains import ZZ_I, QQ_I
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.domains.realfield import RealField

from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.rings import ring, PolyElement
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.polys.fields import field, FracElement

from sympy.polys.agca.extensions import FiniteExtension

from sympy.polys.polyerrors import (
    UnificationFailed,
    GeneratorsError,
    CoercionFailed,
    NotInvertible,
    DomainError)

from sympy.testing.pytest import raises, warns_deprecated_sympy

from itertools import product

ALG = QQ.algebraic_field(sqrt(2), sqrt(3))

def unify(K0, K1):
    return K0.unify(K1)

def test_Domain_unify():
    F3 = GF(3)
    F5 = GF(5)

    assert unify(F3, F3) == F3
    raises(UnificationFailed, lambda: unify(F3, ZZ))
    raises(UnificationFailed, lambda: unify(F3, QQ))
    raises(UnificationFailed, lambda: unify(F3, ZZ_I))
    raises(UnificationFailed, lambda: unify(F3, QQ_I))
    raises(UnificationFailed, lambda: unify(F3, ALG))
    raises(UnificationFailed, lambda: unify(F3, RR))
    raises(UnificationFailed, lambda: unify(F3, CC))
    raises(UnificationFailed, lambda: unify(F3, ZZ[x]))
    raises(UnificationFailed, lambda: unify(F3, ZZ.frac_field(x)))
    raises(UnificationFailed, lambda: unify(F3, EX))

    assert unify(F5, F5) == F5
    raises(UnificationFailed, lambda: unify(F5, F3))
    raises(UnificationFailed, lambda: unify(F5, F3[x]))
    raises(UnificationFailed, lambda: unify(F5, F3.frac_field(x)))

    raises(UnificationFailed, lambda: unify(ZZ, F3))
    assert unify(ZZ, ZZ) == ZZ
    assert unify(ZZ, QQ) == QQ
    assert unify(ZZ, ALG) == ALG
    assert unify(ZZ, RR) == RR
    assert unify(ZZ, CC) == CC
    assert unify(ZZ, ZZ[x]) == ZZ[x]
    assert unify(ZZ, ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ, EX) == EX

    raises(UnificationFailed, lambda: unify(QQ, F3))
    assert unify(QQ, ZZ) == QQ
    assert unify(QQ, QQ) == QQ
    assert unify(QQ, ALG) == ALG
    assert unify(QQ, RR) == RR
    assert unify(QQ, CC) == CC
    assert unify(QQ, ZZ[x]) == QQ[x]
    assert unify(QQ, ZZ.frac_field(x)) == QQ.frac_field(x)
    assert unify(QQ, EX) == EX

    raises(UnificationFailed, lambda: unify(ZZ_I, F3))
    assert unify(ZZ_I, ZZ) == ZZ_I
    assert unify(ZZ_I, ZZ_I) == ZZ_I
    assert unify(ZZ_I, QQ) == QQ_I
    assert unify(ZZ_I, ALG) == QQ.algebraic_field(I, sqrt(2), sqrt(3))
    assert unify(ZZ_I, RR) == CC
    assert unify(ZZ_I, CC) == CC
    assert unify(ZZ_I, ZZ[x]) == ZZ_I[x]
    assert unify(ZZ_I, ZZ_I[x]) == ZZ_I[x]
    assert unify(ZZ_I, ZZ.frac_field(x)) == ZZ_I.frac_field(x)
    assert unify(ZZ_I, ZZ_I.frac_field(x)) == ZZ_I.frac_field(x)
    assert unify(ZZ_I, EX) == EX

    raises(UnificationFailed, lambda: unify(QQ_I, F3))
    assert unify(QQ_I, ZZ) == QQ_I
    assert unify(QQ_I, ZZ_I) == QQ_I
    assert unify(QQ_I, QQ) == QQ_I
    assert unify(QQ_I, ALG) == QQ.algebraic_field(I, sqrt(2), sqrt(3))
    assert unify(QQ_I, RR) == CC
    assert unify(QQ_I, CC) == CC
    assert unify(QQ_I, ZZ[x]) == QQ_I[x]
    assert unify(QQ_I, ZZ_I[x]) == QQ_I[x]
    assert unify(QQ_I, QQ[x]) == QQ_I[x]
    assert unify(QQ_I, QQ_I[x]) == QQ_I[x]
    assert unify(QQ_I, ZZ.frac_field(x)) == QQ_I.frac_field(x)
    assert unify(QQ_I, ZZ_I.frac_field(x)) == QQ_I.frac_field(x)
    assert unify(QQ_I, QQ.frac_field(x)) == QQ_I.frac_field(x)
    assert unify(QQ_I, QQ_I.frac_field(x)) == QQ_I.frac_field(x)
    assert unify(QQ_I, EX) == EX

    raises(UnificationFailed, lambda: unify(RR, F3))
    assert unify(RR, ZZ) == RR
    assert unify(RR, QQ) == RR
    assert unify(RR, ALG) == RR
    assert unify(RR, RR) == RR
    assert unify(RR, CC) == CC
    assert unify(RR, ZZ[x]) == RR[x]
    assert unify(RR, ZZ.frac_field(x)) == RR.frac_field(x)
    assert unify(RR, EX) == EX
    assert RR[x].unify(ZZ.frac_field(y)) == RR.frac_field(x, y)

    raises(UnificationFailed, lambda: unify(CC, F3))
    assert unify(CC, ZZ) == CC
    assert unify(CC, QQ) == CC
    assert unify(CC, ALG) == CC
    assert unify(CC, RR) == CC
    assert unify(CC, CC) == CC
    assert unify(CC, ZZ[x]) == CC[x]
    assert unify(CC, ZZ.frac_field(x)) == CC.frac_field(x)
    assert unify(CC, EX) == EX

    raises(UnificationFailed, lambda: unify(ZZ[x], F3))
    assert unify(ZZ[x], ZZ) == ZZ[x]
    assert unify(ZZ[x], QQ) == QQ[x]
    assert unify(ZZ[x], ALG) == ALG[x]
    assert unify(ZZ[x], RR) == RR[x]
    assert unify(ZZ[x], CC) == CC[x]
    assert unify(ZZ[x], ZZ[x]) == ZZ[x]
    assert unify(ZZ[x], ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ[x], EX) == EX

    raises(UnificationFailed, lambda: unify(ZZ.frac_field(x), F3))
    assert unify(ZZ.frac_field(x), ZZ) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), QQ) == QQ.frac_field(x)
    assert unify(ZZ.frac_field(x), ALG) == ALG.frac_field(x)
    assert unify(ZZ.frac_field(x), RR) == RR.frac_field(x)
    assert unify(ZZ.frac_field(x), CC) == CC.frac_field(x)
    assert unify(ZZ.frac_field(x), ZZ[x]) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), EX) == EX

    raises(UnificationFailed, lambda: unify(EX, F3))
    assert unify(EX, ZZ) == EX
    assert unify(EX, QQ) == EX
    assert unify(EX, ALG) == EX
    assert unify(EX, RR) == EX
    assert unify(EX, CC) == EX
    assert unify(EX, ZZ[x]) == EX
    assert unify(EX, ZZ.frac_field(x)) == EX
    assert unify(EX, EX) == EX

def test_Domain_unify_composite():
    assert unify(ZZ.poly_ring(x), ZZ) == ZZ.poly_ring(x)
    assert unify(ZZ.poly_ring(x), QQ) == QQ.poly_ring(x)
    assert unify(QQ.poly_ring(x), ZZ) == QQ.poly_ring(x)
    assert unify(QQ.poly_ring(x), QQ) == QQ.poly_ring(x)

    assert unify(ZZ, ZZ.poly_ring(x)) == ZZ.poly_ring(x)
    assert unify(QQ, ZZ.poly_ring(x)) == QQ.poly_ring(x)
    assert unify(ZZ, QQ.poly_ring(x)) == QQ.poly_ring(x)
    assert unify(QQ, QQ.poly_ring(x)) == QQ.poly_ring(x)

    assert unify(ZZ.poly_ring(x, y), ZZ) == ZZ.poly_ring(x, y)
    assert unify(ZZ.poly_ring(x, y), QQ) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x, y), ZZ) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x, y), QQ) == QQ.poly_ring(x, y)

    assert unify(ZZ, ZZ.poly_ring(x, y)) == ZZ.poly_ring(x, y)
    assert unify(QQ, ZZ.poly_ring(x, y)) == QQ.poly_ring(x, y)
    assert unify(ZZ, QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)
    assert unify(QQ, QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)

    assert unify(ZZ.frac_field(x), ZZ) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), QQ) == QQ.frac_field(x)
    assert unify(QQ.frac_field(x), ZZ) == QQ.frac_field(x)
    assert unify(QQ.frac_field(x), QQ) == QQ.frac_field(x)

    assert unify(ZZ, ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(QQ, ZZ.frac_field(x)) == QQ.frac_field(x)
    assert unify(ZZ, QQ.frac_field(x)) == QQ.frac_field(x)
    assert unify(QQ, QQ.frac_field(x)) == QQ.frac_field(x)

    assert unify(ZZ.frac_field(x, y), ZZ) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), QQ) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), ZZ) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), QQ) == QQ.frac_field(x, y)

    assert unify(ZZ, ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ, ZZ.frac_field(x, y)) == QQ.frac_field(x, y)
    assert unify(ZZ, QQ.frac_field(x, y)) == QQ.frac_field(x, y)
    assert unify(QQ, QQ.frac_field(x, y)) == QQ.frac_field(x, y)

    assert unify(ZZ.poly_ring(x), ZZ.poly_ring(x)) == ZZ.poly_ring(x)
    assert unify(ZZ.poly_ring(x), QQ.poly_ring(x)) == QQ.poly_ring(x)
    assert unify(QQ.poly_ring(x), ZZ.poly_ring(x)) == QQ.poly_ring(x)
    assert unify(QQ.poly_ring(x), QQ.poly_ring(x)) == QQ.poly_ring(x)

    assert unify(ZZ.poly_ring(x, y), ZZ.poly_ring(x)) == ZZ.poly_ring(x, y)
    assert unify(ZZ.poly_ring(x, y), QQ.poly_ring(x)) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x, y), ZZ.poly_ring(x)) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x, y), QQ.poly_ring(x)) == QQ.poly_ring(x, y)

    assert unify(ZZ.poly_ring(x), ZZ.poly_ring(x, y)) == ZZ.poly_ring(x, y)
    assert unify(ZZ.poly_ring(x), QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x), ZZ.poly_ring(x, y)) == QQ.poly_ring(x, y)
    assert unify(QQ.poly_ring(x), QQ.poly_ring(x, y)) == QQ.poly_ring(x, y)

    assert unify(ZZ.poly_ring(x, y), ZZ.poly_ring(x, z)) == ZZ.poly_ring(x, y, z)
    assert unify(ZZ.poly_ring(x, y), QQ.poly_ring(x, z)) == QQ.poly_ring(x, y, z)
    assert unify(QQ.poly_ring(x, y), ZZ.poly_ring(x, z)) == QQ.poly_ring(x, y, z)
    assert unify(QQ.poly_ring(x, y), QQ.poly_ring(x, z)) == QQ.poly_ring(x, y, z)

    assert unify(ZZ.frac_field(x), ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), QQ.frac_field(x)) == QQ.frac_field(x)
    assert unify(QQ.frac_field(x), ZZ.frac_field(x)) == QQ.frac_field(x)
    assert unify(QQ.frac_field(x), QQ.frac_field(x)) == QQ.frac_field(x)

    assert unify(ZZ.frac_field(x, y), ZZ.frac_field(x)) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), QQ.frac_field(x)) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), ZZ.frac_field(x)) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), QQ.frac_field(x)) == QQ.frac_field(x, y)

    assert unify(ZZ.frac_field(x), ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x), QQ.frac_field(x, y)) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x), ZZ.frac_field(x, y)) == QQ.frac_field(x, y)
    assert unify(QQ.frac_field(x), QQ.frac_field(x, y)) == QQ.frac_field(x, y)

    assert unify(ZZ.frac_field(x, y), ZZ.frac_field(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(ZZ.frac_field(x, y), QQ.frac_field(x, z)) == QQ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), ZZ.frac_field(x, z)) == QQ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), QQ.frac_field(x, z)) == QQ.frac_field(x, y, z)

    assert unify(ZZ.poly_ring(x), ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(ZZ.poly_ring(x), QQ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(QQ.poly_ring(x), ZZ.frac_field(x)) == ZZ.frac_field(x)
    assert unify(QQ.poly_ring(x), QQ.frac_field(x)) == QQ.frac_field(x)

    assert unify(ZZ.poly_ring(x, y), ZZ.frac_field(x)) == ZZ.frac_field(x, y)
    assert unify(ZZ.poly_ring(x, y), QQ.frac_field(x)) == ZZ.frac_field(x, y)
    assert unify(QQ.poly_ring(x, y), ZZ.frac_field(x)) == ZZ.frac_field(x, y)
    assert unify(QQ.poly_ring(x, y), QQ.frac_field(x)) == QQ.frac_field(x, y)

    assert unify(ZZ.poly_ring(x), ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(ZZ.poly_ring(x), QQ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ.poly_ring(x), ZZ.frac_field(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ.poly_ring(x), QQ.frac_field(x, y)) == QQ.frac_field(x, y)

    assert unify(ZZ.poly_ring(x, y), ZZ.frac_field(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(ZZ.poly_ring(x, y), QQ.frac_field(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(QQ.poly_ring(x, y), ZZ.frac_field(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(QQ.poly_ring(x, y), QQ.frac_field(x, z)) == QQ.frac_field(x, y, z)

    assert unify(ZZ.frac_field(x), ZZ.poly_ring(x)) == ZZ.frac_field(x)
    assert unify(ZZ.frac_field(x), QQ.poly_ring(x)) == ZZ.frac_field(x)
    assert unify(QQ.frac_field(x), ZZ.poly_ring(x)) == ZZ.frac_field(x)
    assert unify(QQ.frac_field(x), QQ.poly_ring(x)) == QQ.frac_field(x)

    assert unify(ZZ.frac_field(x, y), ZZ.poly_ring(x)) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x, y), QQ.poly_ring(x)) == ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), ZZ.poly_ring(x)) == ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x, y), QQ.poly_ring(x)) == QQ.frac_field(x, y)

    assert unify(ZZ.frac_field(x), ZZ.poly_ring(x, y)) == ZZ.frac_field(x, y)
    assert unify(ZZ.frac_field(x), QQ.poly_ring(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x), ZZ.poly_ring(x, y)) == ZZ.frac_field(x, y)
    assert unify(QQ.frac_field(x), QQ.poly_ring(x, y)) == QQ.frac_field(x, y)

    assert unify(ZZ.frac_field(x, y), ZZ.poly_ring(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(ZZ.frac_field(x, y), QQ.poly_ring(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), ZZ.poly_ring(x, z)) == ZZ.frac_field(x, y, z)
    assert unify(QQ.frac_field(x, y), QQ.poly_ring(x, z)) == QQ.frac_field(x, y, z)

def test_Domain_unify_algebraic():
    sqrt5 = QQ.algebraic_field(sqrt(5))
    sqrt7 = QQ.algebraic_field(sqrt(7))
    sqrt57 = QQ.algebraic_field(sqrt(5), sqrt(7))

    assert sqrt5.unify(sqrt7) == sqrt57

    assert sqrt5.unify(sqrt5[x, y]) == sqrt5[x, y]
    assert sqrt5[x, y].unify(sqrt5) == sqrt5[x, y]

    assert sqrt5.unify(sqrt5.frac_field(x, y)) == sqrt5.frac_field(x, y)
    assert sqrt5.frac_field(x, y).unify(sqrt5) == sqrt5.frac_field(x, y)

    assert sqrt5.unify(sqrt7[x, y]) == sqrt57[x, y]
    assert sqrt5[x, y].unify(sqrt7) == sqrt57[x, y]

    assert sqrt5.unify(sqrt7.frac_field(x, y)) == sqrt57.frac_field(x, y)
    assert sqrt5.frac_field(x, y).unify(sqrt7) == sqrt57.frac_field(x, y)

def test_Domain_unify_FiniteExtension():
    KxZZ = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ))
    KxQQ = FiniteExtension(Poly(x**2 - 2, x, domain=QQ))
    KxZZy = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ[y]))
    KxQQy = FiniteExtension(Poly(x**2 - 2, x, domain=QQ[y]))

    assert KxZZ.unify(KxZZ) == KxZZ
    assert KxQQ.unify(KxQQ) == KxQQ
    assert KxZZy.unify(KxZZy) == KxZZy
    assert KxQQy.unify(KxQQy) == KxQQy

    assert KxZZ.unify(ZZ) == KxZZ
    assert KxZZ.unify(QQ) == KxQQ
    assert KxQQ.unify(ZZ) == KxQQ
    assert KxQQ.unify(QQ) == KxQQ

    assert KxZZ.unify(ZZ[y]) == KxZZy
    assert KxZZ.unify(QQ[y]) == KxQQy
    assert KxQQ.unify(ZZ[y]) == KxQQy
    assert KxQQ.unify(QQ[y]) == KxQQy

    assert KxZZy.unify(ZZ) == KxZZy
    assert KxZZy.unify(QQ) == KxQQy
    assert KxQQy.unify(ZZ) == KxQQy
    assert KxQQy.unify(QQ) == KxQQy

    assert KxZZy.unify(ZZ[y]) == KxZZy
    assert KxZZy.unify(QQ[y]) == KxQQy
    assert KxQQy.unify(ZZ[y]) == KxQQy
    assert KxQQy.unify(QQ[y]) == KxQQy

    K = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ[y]))
    assert K.unify(ZZ) == K
    assert K.unify(ZZ[x]) == K
    assert K.unify(ZZ[y]) == K
    assert K.unify(ZZ[x, y]) == K

    Kz = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ[y, z]))
    assert K.unify(ZZ[z]) == Kz
    assert K.unify(ZZ[x, z]) == Kz
    assert K.unify(ZZ[y, z]) == Kz
    assert K.unify(ZZ[x, y, z]) == Kz

    Kx = FiniteExtension(Poly(x**2 - 2, x, domain=ZZ))
    Ky = FiniteExtension(Poly(y**2 - 2, y, domain=ZZ))
    Kxy = FiniteExtension(Poly(y**2 - 2, y, domain=Kx))
    assert Kx.unify(Kx) == Kx
    assert Ky.unify(Ky) == Ky
    assert Kx.unify(Ky) == Kxy
    assert Ky.unify(Kx) == Kxy

def test_Domain_unify_with_symbols():
    raises(UnificationFailed, lambda: ZZ[x, y].unify_with_symbols(ZZ, (y, z)))
    raises(UnificationFailed, lambda: ZZ.unify_with_symbols(ZZ[x, y], (y, z)))

def test_Domain__contains__():
    assert (0 in EX) is True
    assert (0 in ZZ) is True
    assert (0 in QQ) is True
    assert (0 in RR) is True
    assert (0 in CC) is True
    assert (0 in ALG) is True
    assert (0 in ZZ[x, y]) is True
    assert (0 in QQ[x, y]) is True
    assert (0 in RR[x, y]) is True

    assert (-7 in EX) is True
    assert (-7 in ZZ) is True
    assert (-7 in QQ) is True
    assert (-7 in RR) is True
    assert (-7 in CC) is True
    assert (-7 in ALG) is True
    assert (-7 in ZZ[x, y]) is True
    assert (-7 in QQ[x, y]) is True
    assert (-7 in RR[x, y]) is True

    assert (17 in EX) is True
    assert (17 in ZZ) is True
    assert (17 in QQ) is True
    assert (17 in RR) is True
    assert (17 in CC) is True
    assert (17 in ALG) is True
    assert (17 in ZZ[x, y]) is True
    assert (17 in QQ[x, y]) is True
    assert (17 in RR[x, y]) is True

    assert (Rational(-1, 7) in EX) is True
    assert (Rational(-1, 7) in ZZ) is False
    assert (Rational(-1, 7) in QQ) is True
    assert (Rational(-1, 7) in RR) is True
    assert (Rational(-1, 7) in CC) is True
    assert (Rational(-1, 7) in ALG) is True
    assert (Rational(-1, 7) in ZZ[x, y]) is False
    assert (Rational(-1, 7) in QQ[x, y]) is True
    assert (Rational(-1, 7) in RR[x, y]) is True

    assert (Rational(3, 5) in EX) is True
    assert (Rational(3, 5) in ZZ) is False
    assert (Rational(3, 5) in QQ) is True
    assert (Rational(3, 5) in RR) is True
    assert (Rational(3, 5) in CC) is True
    assert (Rational(3, 5) in ALG) is True
    assert (Rational(3, 5) in ZZ[x, y]) is False
    assert (Rational(3, 5) in QQ[x, y]) is True
    assert (Rational(3, 5) in RR[x, y]) is True

    assert (3.0 in EX) is True
    assert (3.0 in ZZ) is True
    assert (3.0 in QQ) is True
    assert (3.0 in RR) is True
    assert (3.0 in CC) is True
    assert (3.0 in ALG) is True
    assert (3.0 in ZZ[x, y]) is True
    assert (3.0 in QQ[x, y]) is True
    assert (3.0 in RR[x, y]) is True

    assert (3.14 in EX) is True
    assert (3.14 in ZZ) is False
    assert (3.14 in QQ) is True
    assert (3.14 in RR) is True
    assert (3.14 in CC) is True
    assert (3.14 in ALG) is True
    assert (3.14 in ZZ[x, y]) is False
    assert (3.14 in QQ[x, y]) is True
    assert (3.14 in RR[x, y]) is True

    assert (oo in ALG) is False
    assert (oo in ZZ[x, y]) is False
    assert (oo in QQ[x, y]) is False

    assert (-oo in ZZ) is False
    assert (-oo in QQ) is False
    assert (-oo in ALG) is False
    assert (-oo in ZZ[x, y]) is False
    assert (-oo in QQ[x, y]) is False

    assert (sqrt(7) in EX) is True
    assert (sqrt(7) in ZZ) is False
    assert (sqrt(7) in QQ) is False
    assert (sqrt(7) in RR) is True
    assert (sqrt(7) in CC) is True
    assert (sqrt(7) in ALG) is False
    assert (sqrt(7) in ZZ[x, y]) is False
    assert (sqrt(7) in QQ[x, y]) is False
    assert (sqrt(7) in RR[x, y]) is True

    assert (2*sqrt(3) + 1 in EX) is True
    assert (2*sqrt(3) + 1 in ZZ) is False
    assert (2*sqrt(3) + 1 in QQ) is False
    assert (2*sqrt(3) + 1 in RR) is True
    assert (2*sqrt(3) + 1 in CC) is True
    assert (2*sqrt(3) + 1 in ALG) is True
    assert (2*sqrt(3) + 1 in ZZ[x, y]) is False
    assert (2*sqrt(3) + 1 in QQ[x, y]) is False
    assert (2*sqrt(3) + 1 in RR[x, y]) is True

    assert (sin(1) in EX) is True
    assert (sin(1) in ZZ) is False
    assert (sin(1) in QQ) is False
    assert (sin(1) in RR) is True
    assert (sin(1) in CC) is True
    assert (sin(1) in ALG) is False
    assert (sin(1) in ZZ[x, y]) is False
    assert (sin(1) in QQ[x, y]) is False
    assert (sin(1) in RR[x, y]) is True

    assert (x**2 + 1 in EX) is True
    assert (x**2 + 1 in ZZ) is False
    assert (x**2 + 1 in QQ) is False
    assert (x**2 + 1 in RR) is False
    assert (x**2 + 1 in CC) is False
    assert (x**2 + 1 in ALG) is False
    assert (x**2 + 1 in ZZ[x]) is True
    assert (x**2 + 1 in QQ[x]) is True
    assert (x**2 + 1 in RR[x]) is True
    assert (x**2 + 1 in ZZ[x, y]) is True
    assert (x**2 + 1 in QQ[x, y]) is True
    assert (x**2 + 1 in RR[x, y]) is True

    assert (x**2 + y**2 in EX) is True
    assert (x**2 + y**2 in ZZ) is False
    assert (x**2 + y**2 in QQ) is False
    assert (x**2 + y**2 in RR) is False
    assert (x**2 + y**2 in CC) is False
    assert (x**2 + y**2 in ALG) is False
    assert (x**2 + y**2 in ZZ[x]) is False
    assert (x**2 + y**2 in QQ[x]) is False
    assert (x**2 + y**2 in RR[x]) is False
    assert (x**2 + y**2 in ZZ[x, y]) is True
    assert (x**2 + y**2 in QQ[x, y]) is True
    assert (x**2 + y**2 in RR[x, y]) is True

    assert (Rational(3, 2)*x/(y + 1) - z in QQ[x, y, z]) is False


def test_issue_14433():
    assert (Rational(2, 3)*x in QQ.frac_field(1/x)) is True
    assert (1/x in QQ.frac_field(x)) is True
    assert ((x**2 + y**2) in QQ.frac_field(1/x, 1/y)) is True
    assert ((x + y) in QQ.frac_field(1/x, y)) is True
    assert ((x - y) in QQ.frac_field(x, 1/y)) is True


def test_Domain_is_field():
    assert ZZ.is_Field is False
    assert GF(5).is_Field is True
    assert GF(6).is_Field is False
    assert QQ.is_Field is True
    assert RR.is_Field is True
    assert CC.is_Field is True
    assert EX.is_Field is True
    assert ALG.is_Field is True
    assert QQ[x].is_Field is False
    assert ZZ.frac_field(x).is_Field is True


def test_Domain_get_ring():
    assert ZZ.has_assoc_Ring is True
    assert QQ.has_assoc_Ring is True
    assert ZZ[x].has_assoc_Ring is True
    assert QQ[x].has_assoc_Ring is True
    assert ZZ[x, y].has_assoc_Ring is True
    assert QQ[x, y].has_assoc_Ring is True
    assert ZZ.frac_field(x).has_assoc_Ring is True
    assert QQ.frac_field(x).has_assoc_Ring is True
    assert ZZ.frac_field(x, y).has_assoc_Ring is True
    assert QQ.frac_field(x, y).has_assoc_Ring is True

    assert EX.has_assoc_Ring is False
    assert RR.has_assoc_Ring is False
    assert ALG.has_assoc_Ring is False

    assert ZZ.get_ring() == ZZ
    assert QQ.get_ring() == ZZ
    assert ZZ[x].get_ring() == ZZ[x]
    assert QQ[x].get_ring() == QQ[x]
    assert ZZ[x, y].get_ring() == ZZ[x, y]
    assert QQ[x, y].get_ring() == QQ[x, y]
    assert ZZ.frac_field(x).get_ring() == ZZ[x]
    assert QQ.frac_field(x).get_ring() == QQ[x]
    assert ZZ.frac_field(x, y).get_ring() == ZZ[x, y]
    assert QQ.frac_field(x, y).get_ring() == QQ[x, y]

    assert EX.get_ring() == EX

    assert RR.get_ring() == RR
    # XXX: This should also be like RR
    raises(DomainError, lambda: ALG.get_ring())


def test_Domain_get_field():
    assert EX.has_assoc_Field is True
    assert ZZ.has_assoc_Field is True
    assert QQ.has_assoc_Field is True
    assert RR.has_assoc_Field is True
    assert ALG.has_assoc_Field is True
    assert ZZ[x].has_assoc_Field is True
    assert QQ[x].has_assoc_Field is True
    assert ZZ[x, y].has_assoc_Field is True
    assert QQ[x, y].has_assoc_Field is True

    assert EX.get_field() == EX
    assert ZZ.get_field() == QQ
    assert QQ.get_field() == QQ
    assert RR.get_field() == RR
    assert ALG.get_field() == ALG
    assert ZZ[x].get_field() == ZZ.frac_field(x)
    assert QQ[x].get_field() == QQ.frac_field(x)
    assert ZZ[x, y].get_field() == ZZ.frac_field(x, y)
    assert QQ[x, y].get_field() == QQ.frac_field(x, y)


def test_Domain_set_domain():
    doms = [GF(5), ZZ, QQ, ALG, RR, CC, EX, ZZ[z], QQ[z], RR[z], CC[z], EX[z]]
    for D1 in doms:
        for D2 in doms:
            assert D1[x].set_domain(D2) == D2[x]
            assert D1[x, y].set_domain(D2) == D2[x, y]
            assert D1.frac_field(x).set_domain(D2) == D2.frac_field(x)
            assert D1.frac_field(x, y).set_domain(D2) == D2.frac_field(x, y)
            assert D1.old_poly_ring(x).set_domain(D2) == D2.old_poly_ring(x)
            assert D1.old_poly_ring(x, y).set_domain(D2) == D2.old_poly_ring(x, y)
            assert D1.old_frac_field(x).set_domain(D2) == D2.old_frac_field(x)
            assert D1.old_frac_field(x, y).set_domain(D2) == D2.old_frac_field(x, y)


def test_Domain_is_Exact():
    exact = [GF(5), ZZ, QQ, ALG, EX]
    inexact = [RR, CC]
    for D in exact + inexact:
        for R in D, D[x], D.frac_field(x), D.old_poly_ring(x), D.old_frac_field(x):
            if D in exact:
                assert R.is_Exact is True
            else:
                assert R.is_Exact is False


def test_Domain_get_exact():
    assert EX.get_exact() == EX
    assert ZZ.get_exact() == ZZ
    assert QQ.get_exact() == QQ
    assert RR.get_exact() == QQ
    assert CC.get_exact() == QQ_I
    assert ALG.get_exact() == ALG
    assert ZZ[x].get_exact() == ZZ[x]
    assert QQ[x].get_exact() == QQ[x]
    assert RR[x].get_exact() == QQ[x]
    assert CC[x].get_exact() == QQ_I[x]
    assert ZZ[x, y].get_exact() == ZZ[x, y]
    assert QQ[x, y].get_exact() == QQ[x, y]
    assert RR[x, y].get_exact() == QQ[x, y]
    assert CC[x, y].get_exact() == QQ_I[x, y]
    assert ZZ.frac_field(x).get_exact() == ZZ.frac_field(x)
    assert QQ.frac_field(x).get_exact() == QQ.frac_field(x)
    assert RR.frac_field(x).get_exact() == QQ.frac_field(x)
    assert CC.frac_field(x).get_exact() == QQ_I.frac_field(x)
    assert ZZ.frac_field(x, y).get_exact() == ZZ.frac_field(x, y)
    assert QQ.frac_field(x, y).get_exact() == QQ.frac_field(x, y)
    assert RR.frac_field(x, y).get_exact() == QQ.frac_field(x, y)
    assert CC.frac_field(x, y).get_exact() == QQ_I.frac_field(x, y)
    assert ZZ.old_poly_ring(x).get_exact() == ZZ.old_poly_ring(x)
    assert QQ.old_poly_ring(x).get_exact() == QQ.old_poly_ring(x)
    assert RR.old_poly_ring(x).get_exact() == QQ.old_poly_ring(x)
    assert CC.old_poly_ring(x).get_exact() == QQ_I.old_poly_ring(x)
    assert ZZ.old_poly_ring(x, y).get_exact() == ZZ.old_poly_ring(x, y)
    assert QQ.old_poly_ring(x, y).get_exact() == QQ.old_poly_ring(x, y)
    assert RR.old_poly_ring(x, y).get_exact() == QQ.old_poly_ring(x, y)
    assert CC.old_poly_ring(x, y).get_exact() == QQ_I.old_poly_ring(x, y)
    assert ZZ.old_frac_field(x).get_exact() == ZZ.old_frac_field(x)
    assert QQ.old_frac_field(x).get_exact() == QQ.old_frac_field(x)
    assert RR.old_frac_field(x).get_exact() == QQ.old_frac_field(x)
    assert CC.old_frac_field(x).get_exact() == QQ_I.old_frac_field(x)
    assert ZZ.old_frac_field(x, y).get_exact() == ZZ.old_frac_field(x, y)
    assert QQ.old_frac_field(x, y).get_exact() == QQ.old_frac_field(x, y)
    assert RR.old_frac_field(x, y).get_exact() == QQ.old_frac_field(x, y)
    assert CC.old_frac_field(x, y).get_exact() == QQ_I.old_frac_field(x, y)


def test_Domain_characteristic():
    for F, c in [(FF(3), 3), (FF(5), 5), (FF(7), 7)]:
        for R in F, F[x], F.frac_field(x), F.old_poly_ring(x), F.old_frac_field(x):
            assert R.has_CharacteristicZero is False
            assert R.characteristic() == c
    for D in ZZ, QQ, ZZ_I, QQ_I, ALG:
        for R in D, D[x], D.frac_field(x), D.old_poly_ring(x), D.old_frac_field(x):
            assert R.has_CharacteristicZero is True
            assert R.characteristic() == 0


def test_Domain_is_unit():
    nums = [-2, -1, 0, 1, 2]
    invring = [False, True, False, True, False]
    invfield = [True, True, False, True, True]
    ZZx, QQx, QQxf = ZZ[x], QQ[x], QQ.frac_field(x)
    assert [ZZ.is_unit(ZZ(n)) for n in nums] == invring
    assert [QQ.is_unit(QQ(n)) for n in nums] == invfield
    assert [ZZx.is_unit(ZZx(n)) for n in nums] == invring
    assert [QQx.is_unit(QQx(n)) for n in nums] == invfield
    assert [QQxf.is_unit(QQxf(n)) for n in nums] == invfield
    assert ZZx.is_unit(ZZx(x)) is False
    assert QQx.is_unit(QQx(x)) is False
    assert QQxf.is_unit(QQxf(x)) is True


def test_Domain_convert():

    def check_element(e1, e2, K1, K2, K3):
        if isinstance(e1, PolyElement):
            assert isinstance(e2, PolyElement) and e1.ring == e2.ring
        elif isinstance(e1, FracElement):
            assert isinstance(e2, FracElement) and e1.field == e2.field
        else:
            assert type(e1) is type(e2), '%s, %s: %s %s -> %s' % (e1, e2, K1, K2, K3)
        assert e1 == e2, '%s, %s: %s %s -> %s' % (e1, e2, K1, K2, K3)

    def check_domains(K1, K2):
        K3 = K1.unify(K2)
        check_element(K3.convert_from(K1.one, K1),  K3.one,  K1, K2, K3)
        check_element(K3.convert_from(K2.one, K2),  K3.one,  K1, K2, K3)
        check_element(K3.convert_from(K1.zero, K1), K3.zero, K1, K2, K3)
        check_element(K3.convert_from(K2.zero, K2), K3.zero, K1, K2, K3)

    def composite_domains(K):
        domains = [
            K,
            K[y], K[z], K[y, z],
            K.frac_field(y), K.frac_field(z), K.frac_field(y, z),
            # XXX: These should be tested and made to work...
            # K.old_poly_ring(y), K.old_frac_field(y),
        ]
        return domains

    QQ2 = QQ.algebraic_field(sqrt(2))
    QQ3 = QQ.algebraic_field(sqrt(3))
    doms = [ZZ, QQ, QQ2, QQ3, QQ_I, ZZ_I, RR, CC]

    for i, K1 in enumerate(doms):
        for K2 in doms[i:]:
            for K3 in composite_domains(K1):
                for K4 in composite_domains(K2):
                    check_domains(K3, K4)

    assert QQ.convert(10e-52) == QQ(1684996666696915, 1684996666696914987166688442938726917102321526408785780068975640576)

    R, xr = ring("x", ZZ)
    assert ZZ.convert(xr - xr) == 0
    assert ZZ.convert(xr - xr, R.to_domain()) == 0

    assert CC.convert(ZZ_I(1, 2)) == CC(1, 2)
    assert CC.convert(QQ_I(1, 2)) == CC(1, 2)

    assert QQ.convert_from(RR(0.5), RR) == QQ(1, 2)
    assert RR.convert_from(QQ(1, 2), QQ) == RR(0.5)
    assert QQ_I.convert_from(CC(0.5, 0.75), CC) == QQ_I(QQ(1, 2), QQ(3, 4))
    assert CC.convert_from(QQ_I(QQ(1, 2), QQ(3, 4)), QQ_I) == CC(0.5, 0.75)

    K1 = QQ.frac_field(x)
    K2 = ZZ.frac_field(x)
    K3 = QQ[x]
    K4 = ZZ[x]
    Ks = [K1, K2, K3, K4]
    for Ka, Kb in product(Ks, Ks):
        assert Ka.convert_from(Kb.from_sympy(x), Kb) == Ka.from_sympy(x)

    assert K2.convert_from(QQ(1, 2), QQ) == K2(QQ(1, 2))


def test_EX_convert():

    elements = [
        (ZZ, ZZ(3)),
        (QQ, QQ(1,2)),
        (ZZ_I, ZZ_I(1,2)),
        (QQ_I, QQ_I(1,2)),
        (RR, RR(3)),
        (CC, CC(1,2)),
        (EX, EX(3)),
        (EXRAW, EXRAW(3)),
        (ALG, ALG.from_sympy(sqrt(2))),
    ]

    for R, e in elements:
        for EE in EX, EXRAW:
            elem = EE.from_sympy(R.to_sympy(e))
            assert EE.convert_from(e, R) == elem
            assert R.convert_from(elem, EE) == e


def test_GlobalPolynomialRing_convert():
    K1 = QQ.old_poly_ring(x)
    K2 = QQ[x]
    assert K1.convert(x) == K1.convert(K2.convert(x), K2)
    assert K2.convert(x) == K2.convert(K1.convert(x), K1)

    K1 = QQ.old_poly_ring(x, y)
    K2 = QQ[x]
    assert K1.convert(x) == K1.convert(K2.convert(x), K2)
    #assert K2.convert(x) == K2.convert(K1.convert(x), K1)

    K1 = ZZ.old_poly_ring(x, y)
    K2 = QQ[x]
    assert K1.convert(x) == K1.convert(K2.convert(x), K2)
    #assert K2.convert(x) == K2.convert(K1.convert(x), K1)


def test_PolynomialRing__init():
    R, = ring("", ZZ)
    assert ZZ.poly_ring() == R.to_domain()


def test_FractionField__init():
    F, = field("", ZZ)
    assert ZZ.frac_field() == F.to_domain()


def test_FractionField_convert():
    K = QQ.frac_field(x)
    assert K.convert(QQ(2, 3), QQ) == K.from_sympy(Rational(2, 3))
    K = QQ.frac_field(x)
    assert K.convert(ZZ(2), ZZ) == K.from_sympy(Integer(2))


def test_inject():
    assert ZZ.inject(x, y, z) == ZZ[x, y, z]
    assert ZZ[x].inject(y, z) == ZZ[x, y, z]
    assert ZZ.frac_field(x).inject(y, z) == ZZ.frac_field(x, y, z)
    raises(GeneratorsError, lambda: ZZ[x].inject(x))


def test_drop():
    assert ZZ.drop(x) == ZZ
    assert ZZ[x].drop(x) == ZZ
    assert ZZ[x, y].drop(x) == ZZ[y]
    assert ZZ.frac_field(x).drop(x) == ZZ
    assert ZZ.frac_field(x, y).drop(x) == ZZ.frac_field(y)
    assert ZZ[x][y].drop(y) == ZZ[x]
    assert ZZ[x][y].drop(x) == ZZ[y]
    assert ZZ.frac_field(x)[y].drop(x) == ZZ[y]
    assert ZZ.frac_field(x)[y].drop(y) == ZZ.frac_field(x)
    Ky = FiniteExtension(Poly(x**2-1, x, domain=ZZ[y]))
    K = FiniteExtension(Poly(x**2-1, x, domain=ZZ))
    assert Ky.drop(y) == K
    raises(GeneratorsError, lambda: Ky.drop(x))


def test_Domain_map():
    seq = ZZ.map([1, 2, 3, 4])

    assert all(ZZ.of_type(elt) for elt in seq)

    seq = ZZ.map([[1, 2, 3, 4]])

    assert all(ZZ.of_type(elt) for elt in seq[0]) and len(seq) == 1


def test_Domain___eq__():
    assert (ZZ[x, y] == ZZ[x, y]) is True
    assert (QQ[x, y] == QQ[x, y]) is True

    assert (ZZ[x, y] == QQ[x, y]) is False
    assert (QQ[x, y] == ZZ[x, y]) is False

    assert (ZZ.frac_field(x, y) == ZZ.frac_field(x, y)) is True
    assert (QQ.frac_field(x, y) == QQ.frac_field(x, y)) is True

    assert (ZZ.frac_field(x, y) == QQ.frac_field(x, y)) is False
    assert (QQ.frac_field(x, y) == ZZ.frac_field(x, y)) is False

    assert RealField()[x] == RR[x]


def test_Domain__algebraic_field():
    alg = ZZ.algebraic_field(sqrt(2))
    assert alg.ext.minpoly == Poly(x**2 - 2)
    assert alg.dom == QQ

    alg = QQ.algebraic_field(sqrt(2))
    assert alg.ext.minpoly == Poly(x**2 - 2)
    assert alg.dom == QQ

    alg = alg.algebraic_field(sqrt(3))
    assert alg.ext.minpoly == Poly(x**4 - 10*x**2 + 1)
    assert alg.dom == QQ


def test_Domain_alg_field_from_poly():
    f = Poly(x**2 - 2)
    g = Poly(x**2 - 3)
    h = Poly(x**4 - 10*x**2 + 1)

    alg = ZZ.alg_field_from_poly(f)
    assert alg.ext.minpoly == f
    assert alg.dom == QQ

    alg = QQ.alg_field_from_poly(f)
    assert alg.ext.minpoly == f
    assert alg.dom == QQ

    alg = alg.alg_field_from_poly(g)
    assert alg.ext.minpoly == h
    assert alg.dom == QQ


def test_Domain_cyclotomic_field():
    K = ZZ.cyclotomic_field(12)
    assert K.ext.minpoly == Poly(cyclotomic_poly(12))
    assert K.dom == QQ

    F = QQ.cyclotomic_field(3)
    assert F.ext.minpoly == Poly(cyclotomic_poly(3))
    assert F.dom == QQ

    E = F.cyclotomic_field(4)
    assert field_isomorphism(E.ext, K.ext) is not None
    assert E.dom == QQ


def test_PolynomialRing_from_FractionField():
    F, x,y = field("x,y", ZZ)
    R, X,Y = ring("x,y", ZZ)

    f = (x**2 + y**2)/(x + 1)
    g = (x**2 + y**2)/4
    h =  x**2 + y**2

    assert R.to_domain().from_FractionField(f, F.to_domain()) is None
    assert R.to_domain().from_FractionField(g, F.to_domain()) == X**2/4 + Y**2/4
    assert R.to_domain().from_FractionField(h, F.to_domain()) == X**2 + Y**2

    F, x,y = field("x,y", QQ)
    R, X,Y = ring("x,y", QQ)

    f = (x**2 + y**2)/(x + 1)
    g = (x**2 + y**2)/4
    h =  x**2 + y**2

    assert R.to_domain().from_FractionField(f, F.to_domain()) is None
    assert R.to_domain().from_FractionField(g, F.to_domain()) == X**2/4 + Y**2/4
    assert R.to_domain().from_FractionField(h, F.to_domain()) == X**2 + Y**2


def test_FractionField_from_PolynomialRing():
    R, x,y = ring("x,y", QQ)
    F, X,Y = field("x,y", ZZ)

    f = 3*x**2 + 5*y**2
    g = x**2/3 + y**2/5

    assert F.to_domain().from_PolynomialRing(f, R.to_domain()) == 3*X**2 + 5*Y**2
    assert F.to_domain().from_PolynomialRing(g, R.to_domain()) == (5*X**2 + 3*Y**2)/15


def test_FF_of_type():
    # XXX: of_type is not very useful here because in the case of ground types
    # = flint all elements are of type nmod.
    assert FF(3).of_type(FF(3)(1)) is True
    assert FF(5).of_type(FF(5)(3)) is True


def test___eq__():
    assert not QQ[x] == ZZ[x]
    assert not QQ.frac_field(x) == ZZ.frac_field(x)


def test_RealField_from_sympy():
    assert RR.convert(S.Zero) == RR.dtype(0)
    assert RR.convert(S(0.0)) == RR.dtype(0.0)
    assert RR.convert(S.One) == RR.dtype(1)
    assert RR.convert(S(1.0)) == RR.dtype(1.0)
    assert RR.convert(sin(1)) == RR.dtype(sin(1).evalf())


def test_not_in_any_domain():
    check = list(_illegal) + [x] + [
        float(i) for i in _illegal[:3]]
    for dom in (ZZ, QQ, RR, CC, EX):
        for i in check:
            if i == x and dom == EX:
                continue
            assert i not in dom, (i, dom)
            raises(CoercionFailed, lambda: dom.convert(i))


def test_ModularInteger():
    F3 = FF(3)

    a = F3(0)
    assert F3.of_type(a) and a == 0
    a = F3(1)
    assert F3.of_type(a) and a == 1
    a = F3(2)
    assert F3.of_type(a) and a == 2
    a = F3(3)
    assert F3.of_type(a) and a == 0
    a = F3(4)
    assert F3.of_type(a) and a == 1

    a = F3(F3(0))
    assert F3.of_type(a) and a == 0
    a = F3(F3(1))
    assert F3.of_type(a) and a == 1
    a = F3(F3(2))
    assert F3.of_type(a) and a == 2
    a = F3(F3(3))
    assert F3.of_type(a) and a == 0
    a = F3(F3(4))
    assert F3.of_type(a) and a == 1

    a = -F3(1)
    assert F3.of_type(a) and a == 2
    a = -F3(2)
    assert F3.of_type(a) and a == 1

    a = 2 + F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2) + 2
    assert F3.of_type(a) and a == 1
    a = F3(2) + F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2) + F3(2)
    assert F3.of_type(a) and a == 1

    a = 3 - F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(3) - 2
    assert F3.of_type(a) and a == 1
    a = F3(3) - F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(3) - F3(2)
    assert F3.of_type(a) and a == 1

    a = 2*F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2)*2
    assert F3.of_type(a) and a == 1
    a = F3(2)*F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2)*F3(2)
    assert F3.of_type(a) and a == 1

    a = 2/F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2)/2
    assert F3.of_type(a) and a == 1
    a = F3(2)/F3(2)
    assert F3.of_type(a) and a == 1
    a = F3(2)/F3(2)
    assert F3.of_type(a) and a == 1

    a = F3(2)**0
    assert F3.of_type(a) and a == 1
    a = F3(2)**1
    assert F3.of_type(a) and a == 2
    a = F3(2)**2
    assert F3.of_type(a) and a == 1

    F7 = FF(7)

    a = F7(3)**100000000000
    assert F7.of_type(a) and a == 4
    a = F7(3)**-100000000000
    assert F7.of_type(a) and a == 2

    assert bool(F3(3)) is False
    assert bool(F3(4)) is True

    F5 = FF(5)

    a = F5(1)**(-1)
    assert F5.of_type(a) and a == 1
    a = F5(2)**(-1)
    assert F5.of_type(a) and a == 3
    a = F5(3)**(-1)
    assert F5.of_type(a) and a == 2
    a = F5(4)**(-1)
    assert F5.of_type(a) and a == 4

    if GROUND_TYPES != 'flint':
        # XXX: This gives a core dump with python-flint...
        raises(NotInvertible, lambda: F5(0)**(-1))
        raises(NotInvertible, lambda: F5(5)**(-1))

    raises(ValueError, lambda: FF(0))
    raises(ValueError, lambda: FF(2.1))

    for n1 in range(5):
        for n2 in range(5):
            if GROUND_TYPES != 'flint':
                with warns_deprecated_sympy():
                    assert (F5(n1) < F5(n2)) is (n1 < n2)
                with warns_deprecated_sympy():
                    assert (F5(n1) <= F5(n2)) is (n1 <= n2)
                with warns_deprecated_sympy():
                    assert (F5(n1) > F5(n2)) is (n1 > n2)
                with warns_deprecated_sympy():
                    assert (F5(n1) >= F5(n2)) is (n1 >= n2)
            else:
                raises(TypeError, lambda: F5(n1) < F5(n2))
                raises(TypeError, lambda: F5(n1) <= F5(n2))
                raises(TypeError, lambda: F5(n1) > F5(n2))
                raises(TypeError, lambda: F5(n1) >= F5(n2))

    # https://github.com/sympy/sympy/issues/26789
    assert GF(Integer(5)) == F5
    assert F5(Integer(3)) == F5(3)


def test_QQ_int():
    assert int(QQ(2**2000, 3**1250)) == 455431
    assert int(QQ(2**100, 3)) == 422550200076076467165567735125


def test_RR_double():
    assert RR(3.14) > 1e-50
    assert RR(1e-13) > 1e-50
    assert RR(1e-14) > 1e-50
    assert RR(1e-15) > 1e-50
    assert RR(1e-20) > 1e-50
    assert RR(1e-40) > 1e-50


def test_RR_Float():
    f1 = Float("1.01")
    f2 = Float("1.0000000000000000000001")
    assert f1._prec == 53
    assert f2._prec == 80
    assert RR(f1)-1 > 1e-50
    assert RR(f2)-1 < 1e-50 # RR's precision is lower than f2's

    RR2 = RealField(prec=f2._prec)
    assert RR2(f1)-1 > 1e-50
    assert RR2(f2)-1 > 1e-50 # RR's precision is equal to f2's


def test_CC_double():
    assert CC(3.14).real > 1e-50
    assert CC(1e-13).real > 1e-50
    assert CC(1e-14).real > 1e-50
    assert CC(1e-15).real > 1e-50
    assert CC(1e-20).real > 1e-50
    assert CC(1e-40).real > 1e-50

    assert CC(3.14j).imag > 1e-50
    assert CC(1e-13j).imag > 1e-50
    assert CC(1e-14j).imag > 1e-50
    assert CC(1e-15j).imag > 1e-50
    assert CC(1e-20j).imag > 1e-50
    assert CC(1e-40j).imag > 1e-50


def test_gaussian_domains():
    I = S.ImaginaryUnit
    a, b, c, d = [ZZ_I.convert(x) for x in (5, 2 + I, 3 - I, 5 - 5*I)]
    assert ZZ_I.gcd(a, b) == b
    assert ZZ_I.gcd(a, c) == b
    assert ZZ_I.lcm(a, b) == a
    assert ZZ_I.lcm(a, c) == d
    assert ZZ_I(3, 4) != QQ_I(3, 4)  # XXX is this right or should QQ->ZZ if possible?
    assert ZZ_I(3, 0) != 3           # and should this go to Integer?
    assert QQ_I(S(3)/4, 0) != S(3)/4 # and this to Rational?
    assert ZZ_I(0, 0).quadrant() == 0
    assert ZZ_I(-1, 0).quadrant() == 2

    assert QQ_I.convert(QQ(3, 2)) == QQ_I(QQ(3, 2), QQ(0))
    assert QQ_I.convert(QQ(3, 2), QQ) == QQ_I(QQ(3, 2), QQ(0))

    for G in (QQ_I, ZZ_I):

        q = G(3, 4)
        assert str(q) == '3 + 4*I'
        assert q.parent() == G
        assert q._get_xy(pi) == (None, None)
        assert q._get_xy(2) == (2, 0)
        assert q._get_xy(2*I) == (0, 2)

        assert hash(q) == hash((3, 4))
        assert G(1, 2) == G(1, 2)
        assert G(1, 2) != G(1, 3)
        assert G(3, 0) == G(3)

        assert q + q == G(6, 8)
        assert q - q == G(0, 0)
        assert 3 - q  == -q + 3 == G(0, -4)
        assert 3 + q == q + 3 == G(6, 4)
        assert q * q == G(-7, 24)
        assert 3 * q == q * 3 == G(9, 12)
        assert q ** 0 == G(1, 0)
        assert q ** 1 == q
        assert q ** 2 == q * q == G(-7, 24)
        assert q ** 3 == q * q * q == G(-117, 44)
        assert 1 / q == q ** -1 == QQ_I(S(3)/25, - S(4)/25)
        assert q / 1 == QQ_I(3, 4)
        assert q / 2 == QQ_I(S(3)/2, 2)
        assert q/3 == QQ_I(1, S(4)/3)
        assert 3/q == QQ_I(S(9)/25, -S(12)/25)
        i, r = divmod(q, 2)
        assert 2*i + r == q
        i, r = divmod(2, q)
        assert q*i + r == G(2, 0)

        a, b = G(2, 0), G(1, -1)
        c, d, g = G.gcdex(a, b)
        assert g == G.gcd(a, b)
        assert c * a + d * b == g

        raises(ZeroDivisionError, lambda: q % 0)
        raises(ZeroDivisionError, lambda: q / 0)
        raises(ZeroDivisionError, lambda: q // 0)
        raises(ZeroDivisionError, lambda: divmod(q, 0))
        raises(ZeroDivisionError, lambda: divmod(q, 0))
        raises(TypeError, lambda: q + x)
        raises(TypeError, lambda: q - x)
        raises(TypeError, lambda: x + q)
        raises(TypeError, lambda: x - q)
        raises(TypeError, lambda: q * x)
        raises(TypeError, lambda: x * q)
        raises(TypeError, lambda: q / x)
        raises(TypeError, lambda: x / q)
        raises(TypeError, lambda: q // x)
        raises(TypeError, lambda: x // q)

        assert G.from_sympy(S(2)) == G(2, 0)
        assert G.to_sympy(G(2, 0)) == S(2)
        raises(CoercionFailed, lambda: G.from_sympy(pi))

        PR = G.inject(x)
        assert isinstance(PR, PolynomialRing)
        assert PR.domain == G
        assert len(PR.gens) == 1 and PR.gens[0].as_expr() == x

        if G is QQ_I:
            AF = G.as_AlgebraicField()
            assert isinstance(AF, AlgebraicField)
            assert AF.domain == QQ
            assert AF.ext.args[0] == I

        for qi in [G(-1, 0), G(1, 0), G(0, -1), G(0, 1)]:
            assert G.is_negative(qi) is False
            assert G.is_positive(qi) is False
            assert G.is_nonnegative(qi) is False
            assert G.is_nonpositive(qi) is False

        domains = [ZZ, QQ, AlgebraicField(QQ, I)]

        # XXX: These domains are all obsolete because ZZ/QQ with MPZ/MPQ
        # already use either gmpy, flint or python depending on the
        # availability of these libraries. We can keep these tests for now but
        # ideally we should remove these alternate domains entirely.
        domains += [ZZ_python(), QQ_python()]
        if GROUND_TYPES == 'gmpy':
            domains += [ZZ_gmpy(), QQ_gmpy()]

        for K in domains:
            assert G.convert(K(2)) == G(2, 0)
            assert G.convert(K(2), K) == G(2, 0)

        for K in ZZ_I, QQ_I:
            assert G.convert(K(1, 1)) == G(1, 1)
            assert G.convert(K(1, 1), K) == G(1, 1)

        if G == ZZ_I:
            assert repr(q) == 'ZZ_I(3, 4)'
            assert q//3 == G(1, 1)
            assert 12//q == G(1, -2)
            assert 12 % q == G(1, 2)
            assert q % 2 == G(-1, 0)
            assert i == G(0, 0)
            assert r == G(2, 0)
            assert G.get_ring() == G
            assert G.get_field() == QQ_I
        else:
            assert repr(q) == 'QQ_I(3, 4)'
            assert G.get_ring() == ZZ_I
            assert G.get_field() == G
            assert q//3 == G(1, S(4)/3)
            assert 12//q == G(S(36)/25, -S(48)/25)
            assert 12 % q == G(0, 0)
            assert q % 2 == G(0, 0)
            assert i == G(S(6)/25, -S(8)/25), (G,i)
            assert r == G(0, 0)
            q2 = G(S(3)/2, S(5)/3)
            assert G.numer(q2) == ZZ_I(9, 10)
            assert G.denom(q2) == ZZ_I(6)


def test_EX_EXRAW():
    assert EXRAW.zero is S.Zero
    assert EXRAW.one is S.One

    assert EX(1) == EX.Expression(1)
    assert EX(1).ex is S.One
    assert EXRAW(1) is S.One

    # EX has cancelling but EXRAW does not
    assert 2*EX((x + y*x)/x) == EX(2 + 2*y) != 2*((x + y*x)/x)
    assert 2*EXRAW((x + y*x)/x) == 2*((x + y*x)/x) != (1 + y)

    assert EXRAW.convert_from(EX(1), EX) is EXRAW.one
    assert EX.convert_from(EXRAW(1), EXRAW) == EX.one

    assert EXRAW.from_sympy(S.One) is S.One
    assert EXRAW.to_sympy(EXRAW.one) is S.One
    raises(CoercionFailed, lambda: EXRAW.from_sympy([]))

    assert EXRAW.get_field() == EXRAW

    assert EXRAW.unify(EX) == EXRAW
    assert EX.unify(EXRAW) == EXRAW


def test_EX_ordering():
    elements = [EX(1), EX(x), EX(3)]
    assert sorted(elements) == [EX(1), EX(3), EX(x)]


def test_canonical_unit():

    for K in [ZZ, QQ, RR]: # CC?
        assert K.canonical_unit(K(2)) == K(1)
        assert K.canonical_unit(K(-2)) == K(-1)

    for K in [ZZ_I, QQ_I]:
        i = K.from_sympy(I)
        assert K.canonical_unit(K(2)) == K(1)
        assert K.canonical_unit(K(2)*i) == -i
        assert K.canonical_unit(-K(2)) == K(-1)
        assert K.canonical_unit(-K(2)*i) == i

    K = ZZ[x]
    assert K.canonical_unit(K(x + 1)) == K(1)
    assert K.canonical_unit(K(-x + 1)) == K(-1)

    K = ZZ_I[x]
    assert K.canonical_unit(K.from_sympy(I*x)) == ZZ_I(0, -1)

    K = ZZ_I.frac_field(x, y)
    i = K.from_sympy(I)
    assert i / i == K.one
    assert (K.one + i)/(i - K.one) == -i


def test_Domain_is_negative():
    I = S.ImaginaryUnit
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    assert CC.is_negative(a) == False
    assert CC.is_negative(b) == False


def test_Domain_is_positive():
    I = S.ImaginaryUnit
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    assert CC.is_positive(a) == False
    assert CC.is_positive(b) == False


def test_Domain_is_nonnegative():
    I = S.ImaginaryUnit
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    assert CC.is_nonnegative(a) == False
    assert CC.is_nonnegative(b) == False


def test_Domain_is_nonpositive():
    I = S.ImaginaryUnit
    a, b = [CC.convert(x) for x in (2 + I, 5)]
    assert CC.is_nonpositive(a) == False
    assert CC.is_nonpositive(b) == False


def test_exponential_domain():
    K = ZZ[E]
    eK = K.from_sympy(E)
    assert K.from_sympy(exp(3)) == eK ** 3
    assert K.convert(exp(3)) == eK ** 3


def test_AlgebraicField_alias():
    # No default alias:
    k = QQ.algebraic_field(sqrt(2))
    assert k.ext.alias is None

    # For a single extension, its alias is used:
    alpha = AlgebraicNumber(sqrt(2), alias='alpha')
    k = QQ.algebraic_field(alpha)
    assert k.ext.alias.name == 'alpha'

    # Can override the alias of a single extension:
    k = QQ.algebraic_field(alpha, alias='theta')
    assert k.ext.alias.name == 'theta'

    # With multiple extensions, no default alias:
    k = QQ.algebraic_field(sqrt(2), sqrt(3))
    assert k.ext.alias is None

    # With multiple extensions, no default alias, even if one of
    # the extensions has one:
    k = QQ.algebraic_field(alpha, sqrt(3))
    assert k.ext.alias is None

    # With multiple extensions, may set an alias:
    k = QQ.algebraic_field(sqrt(2), sqrt(3), alias='theta')
    assert k.ext.alias.name == 'theta'

    # Alias is passed to constructed field elements:
    k = QQ.algebraic_field(alpha)
    beta = k.to_alg_num(k([1, 2, 3]))
    assert beta.alias is alpha.alias


def test_exsqrt():
    assert ZZ.is_square(ZZ(4)) is True
    assert ZZ.exsqrt(ZZ(4)) == ZZ(2)
    assert ZZ.is_square(ZZ(42)) is False
    assert ZZ.exsqrt(ZZ(42)) is None
    assert ZZ.is_square(ZZ(0)) is True
    assert ZZ.exsqrt(ZZ(0)) == ZZ(0)
    assert ZZ.is_square(ZZ(-1)) is False
    assert ZZ.exsqrt(ZZ(-1)) is None

    assert QQ.is_square(QQ(9, 4)) is True
    assert QQ.exsqrt(QQ(9, 4)) == QQ(3, 2)
    assert QQ.is_square(QQ(18, 8)) is True
    assert QQ.exsqrt(QQ(18, 8)) == QQ(3, 2)
    assert QQ.is_square(QQ(-9, -4)) is True
    assert QQ.exsqrt(QQ(-9, -4)) == QQ(3, 2)
    assert QQ.is_square(QQ(11, 4)) is False
    assert QQ.exsqrt(QQ(11, 4)) is None
    assert QQ.is_square(QQ(9, 5)) is False
    assert QQ.exsqrt(QQ(9, 5)) is None
    assert QQ.is_square(QQ(4)) is True
    assert QQ.exsqrt(QQ(4)) == QQ(2)
    assert QQ.is_square(QQ(0)) is True
    assert QQ.exsqrt(QQ(0)) == QQ(0)
    assert QQ.is_square(QQ(-16, 9)) is False
    assert QQ.exsqrt(QQ(-16, 9)) is None

    assert RR.is_square(RR(6.25)) is True
    assert RR.exsqrt(RR(6.25)) == RR(2.5)
    assert RR.is_square(RR(2)) is True
    assert RR.almosteq(RR.exsqrt(RR(2)), RR(1.4142135623730951), tolerance=1e-15)
    assert RR.is_square(RR(0)) is True
    assert RR.exsqrt(RR(0)) == RR(0)
    assert RR.is_square(RR(-1)) is False
    assert RR.exsqrt(RR(-1)) is None

    assert CC.is_square(CC(2)) is True
    assert CC.almosteq(CC.exsqrt(CC(2)), CC(1.4142135623730951), tolerance=1e-15)
    assert CC.is_square(CC(0)) is True
    assert CC.exsqrt(CC(0)) == CC(0)
    assert CC.is_square(CC(-1)) is True
    assert CC.exsqrt(CC(-1)) == CC(0, 1)
    assert CC.is_square(CC(0, 2)) is True
    assert CC.exsqrt(CC(0, 2)) == CC(1, 1)
    assert CC.is_square(CC(-3, -4)) is True
    assert CC.exsqrt(CC(-3, -4)) == CC(1, -2)

    F2 = FF(2)
    assert F2.is_square(F2(1)) is True
    assert F2.exsqrt(F2(1)) == F2(1)
    assert F2.is_square(F2(0)) is True
    assert F2.exsqrt(F2(0)) == F2(0)

    F7 = FF(7)
    assert F7.is_square(F7(2)) is True
    assert F7.exsqrt(F7(2)) == F7(3)
    assert F7.is_square(F7(3)) is False
    assert F7.exsqrt(F7(3)) is None
    assert F7.is_square(F7(0)) is True
    assert F7.exsqrt(F7(0)) == F7(0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_setitem.py ===
from datetime import datetime

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    PeriodIndex,
    Series,
    Timestamp,
    cut,
    date_range,
    notna,
    period_range,
)
import pandas._testing as tm
from pandas.core.arrays import SparseArray

from pandas.tseries.offsets import BDay


class TestDataFrameSetItem:
    def test_setitem_str_subclass(self):
        # GH#37366
        class mystring(str):
            pass

        data = ["2020-10-22 01:21:00+00:00"]
        index = DatetimeIndex(data)
        df = DataFrame({"a": [1]}, index=index)
        df["b"] = 2
        df[mystring("c")] = 3
        expected = DataFrame({"a": [1], "b": [2], mystring("c"): [3]}, index=index)
        tm.assert_equal(df, expected)

    @pytest.mark.parametrize(
        "dtype", ["int32", "int64", "uint32", "uint64", "float32", "float64"]
    )
    def test_setitem_dtype(self, dtype, float_frame):
        # Use integers since casting negative floats to uints is undefined
        arr = np.random.default_rng(2).integers(1, 10, len(float_frame))

        float_frame[dtype] = np.array(arr, dtype=dtype)
        assert float_frame[dtype].dtype.name == dtype

    def test_setitem_list_not_dataframe(self, float_frame):
        data = np.random.default_rng(2).standard_normal((len(float_frame), 2))
        float_frame[["A", "B"]] = data
        tm.assert_almost_equal(float_frame[["A", "B"]].values, data)

    def test_setitem_error_msmgs(self):
        # GH 7432
        df = DataFrame(
            {"bar": [1, 2, 3], "baz": ["d", "e", "f"]},
            index=Index(["a", "b", "c"], name="foo"),
        )
        ser = Series(
            ["g", "h", "i", "j"],
            index=Index(["a", "b", "c", "a"], name="foo"),
            name="fiz",
        )
        msg = "cannot reindex on an axis with duplicate labels"
        with pytest.raises(ValueError, match=msg):
            df["newcol"] = ser

        # GH 4107, more descriptive error message
        df = DataFrame(
            np.random.default_rng(2).integers(0, 2, (4, 4)),
            columns=["a", "b", "c", "d"],
        )

        msg = "Cannot set a DataFrame with multiple columns to the single column gr"
        with pytest.raises(ValueError, match=msg):
            df["gr"] = df.groupby(["b", "c"]).count()

        # GH 55956, specific message for zero columns
        msg = "Cannot set a DataFrame without columns to the column gr"
        with pytest.raises(ValueError, match=msg):
            df["gr"] = DataFrame()

    def test_setitem_benchmark(self):
        # from the vb_suite/frame_methods/frame_insert_columns
        N = 10
        K = 5
        df = DataFrame(index=range(N))
        new_col = np.random.default_rng(2).standard_normal(N)
        for i in range(K):
            df[i] = new_col
        expected = DataFrame(np.repeat(new_col, K).reshape(N, K), index=range(N))
        tm.assert_frame_equal(df, expected)

    def test_setitem_different_dtype(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=np.arange(5),
            columns=["c", "b", "a"],
        )
        df.insert(0, "foo", df["a"])
        df.insert(2, "bar", df["c"])

        # diff dtype

        # new item
        df["x"] = df["a"].astype("float32")
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")] * 5 + [np.dtype("float32")],
            index=["foo", "c", "bar", "b", "a", "x"],
        )
        tm.assert_series_equal(result, expected)

        # replacing current (in different block)
        df["a"] = df["a"].astype("float32")
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")] * 4 + [np.dtype("float32")] * 2,
            index=["foo", "c", "bar", "b", "a", "x"],
        )
        tm.assert_series_equal(result, expected)

        df["y"] = df["a"].astype("int32")
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")] * 4 + [np.dtype("float32")] * 2 + [np.dtype("int32")],
            index=["foo", "c", "bar", "b", "a", "x", "y"],
        )
        tm.assert_series_equal(result, expected)

    def test_setitem_empty_columns(self):
        # GH 13522
        df = DataFrame(index=["A", "B", "C"])
        df["X"] = df.index
        df["X"] = ["x", "y", "z"]
        exp = DataFrame(
            data={"X": ["x", "y", "z"]},
            index=["A", "B", "C"],
            columns=Index(["X"], dtype=object),
        )
        tm.assert_frame_equal(df, exp)

    def test_setitem_dt64_index_empty_columns(self):
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        df = DataFrame(index=np.arange(len(rng)))

        df["A"] = rng
        assert df["A"].dtype == np.dtype("M8[ns]")

    def test_setitem_timestamp_empty_columns(self):
        # GH#19843
        df = DataFrame(index=range(3))
        df["now"] = Timestamp("20130101", tz="UTC").as_unit("ns")

        expected = DataFrame(
            [[Timestamp("20130101", tz="UTC")]] * 3,
            index=range(3),
            columns=Index(["now"], dtype=object),
        )
        tm.assert_frame_equal(df, expected)

    def test_setitem_wrong_length_categorical_dtype_raises(self):
        # GH#29523
        cat = Categorical.from_codes([0, 1, 1, 0, 1, 2], ["a", "b", "c"])
        df = DataFrame(range(10), columns=["bar"])

        msg = (
            rf"Length of values \({len(cat)}\) "
            rf"does not match length of index \({len(df)}\)"
        )
        with pytest.raises(ValueError, match=msg):
            df["foo"] = cat

    def test_setitem_with_sparse_value(self):
        # GH#8131
        df = DataFrame({"c_1": ["a", "b", "c"], "n_1": [1.0, 2.0, 3.0]})
        sp_array = SparseArray([0, 0, 1])
        df["new_column"] = sp_array

        expected = Series(sp_array, name="new_column")
        tm.assert_series_equal(df["new_column"], expected)

    def test_setitem_with_unaligned_sparse_value(self):
        df = DataFrame({"c_1": ["a", "b", "c"], "n_1": [1.0, 2.0, 3.0]})
        sp_series = Series(SparseArray([0, 0, 1]), index=[2, 1, 0])

        df["new_column"] = sp_series
        expected = Series(SparseArray([1, 0, 0]), name="new_column")
        tm.assert_series_equal(df["new_column"], expected)

    def test_setitem_period_preserves_dtype(self):
        # GH: 26861
        data = [Period("2003-12", "D")]
        result = DataFrame([])
        result["a"] = data

        expected = DataFrame({"a": data}, columns=Index(["a"], dtype=object))

        tm.assert_frame_equal(result, expected)

    def test_setitem_dict_preserves_dtypes(self):
        # https://github.com/pandas-dev/pandas/issues/34573
        expected = DataFrame(
            {
                "a": Series([0, 1, 2], dtype="int64"),
                "b": Series([1, 2, 3], dtype=float),
                "c": Series([1, 2, 3], dtype=float),
                "d": Series([1, 2, 3], dtype="uint32"),
            }
        )
        df = DataFrame(
            {
                "a": Series([], dtype="int64"),
                "b": Series([], dtype=float),
                "c": Series([], dtype=float),
                "d": Series([], dtype="uint32"),
            }
        )
        for idx, b in enumerate([1, 2, 3]):
            df.loc[df.shape[0]] = {
                "a": int(idx),
                "b": float(b),
                "c": float(b),
                "d": np.uint32(b),
            }
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "obj,dtype",
        [
            (Period("2020-01"), PeriodDtype("M")),
            (Interval(left=0, right=5), IntervalDtype("int64", "right")),
            (
                Timestamp("2011-01-01", tz="US/Eastern"),
                DatetimeTZDtype(unit="s", tz="US/Eastern"),
            ),
        ],
    )
    def test_setitem_extension_types(self, obj, dtype):
        # GH: 34832
        expected = DataFrame({"idx": [1, 2, 3], "obj": Series([obj] * 3, dtype=dtype)})

        df = DataFrame({"idx": [1, 2, 3]})
        df["obj"] = obj

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "ea_name",
        [
            dtype.name
            for dtype in ea_registry.dtypes
            # property would require instantiation
            if not isinstance(dtype.name, property)
        ]
        + ["datetime64[ns, UTC]", "period[D]"],
    )
    def test_setitem_with_ea_name(self, ea_name):
        # GH 38386
        result = DataFrame([0])
        result[ea_name] = [1]
        expected = DataFrame({0: [0], ea_name: [1]})
        tm.assert_frame_equal(result, expected)

    def test_setitem_dt64_ndarray_with_NaT_and_diff_time_units(self):
        # GH#7492
        data_ns = np.array([1, "nat"], dtype="datetime64[ns]")
        result = Series(data_ns).to_frame()
        result["new"] = data_ns
        expected = DataFrame({0: [1, None], "new": [1, None]}, dtype="datetime64[ns]")
        tm.assert_frame_equal(result, expected)

        # OutOfBoundsDatetime error shouldn't occur; as of 2.0 we preserve "M8[s]"
        data_s = np.array([1, "nat"], dtype="datetime64[s]")
        result["new"] = data_s
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_numpy_array_equal(result["new"].to_numpy(), data_s)

    @pytest.mark.parametrize("unit", ["h", "m", "s", "ms", "D", "M", "Y"])
    def test_frame_setitem_datetime64_col_other_units(self, unit):
        # Check that non-nano dt64 values get cast to dt64 on setitem
        #  into a not-yet-existing column
        n = 100

        dtype = np.dtype(f"M8[{unit}]")
        vals = np.arange(n, dtype=np.int64).view(dtype)
        if unit in ["s", "ms"]:
            # supported unit
            ex_vals = vals
        else:
            # we get the nearest supported units, i.e. "s"
            ex_vals = vals.astype("datetime64[s]")

        df = DataFrame({"ints": np.arange(n)}, index=np.arange(n))
        df[unit] = vals

        assert df[unit].dtype == ex_vals.dtype
        assert (df[unit].values == ex_vals).all()

    @pytest.mark.parametrize("unit", ["h", "m", "s", "ms", "D", "M", "Y"])
    def test_frame_setitem_existing_datetime64_col_other_units(self, unit):
        # Check that non-nano dt64 values get cast to dt64 on setitem
        #  into an already-existing dt64 column
        n = 100

        dtype = np.dtype(f"M8[{unit}]")
        vals = np.arange(n, dtype=np.int64).view(dtype)
        ex_vals = vals.astype("datetime64[ns]")

        df = DataFrame({"ints": np.arange(n)}, index=np.arange(n))
        df["dates"] = np.arange(n, dtype=np.int64).view("M8[ns]")

        # We overwrite existing dt64 column with new, non-nano dt64 vals
        df["dates"] = vals
        assert (df["dates"].values == ex_vals).all()

    def test_setitem_dt64tz(self, timezone_frame, using_copy_on_write):
        df = timezone_frame
        idx = df["B"].rename("foo")

        # setitem
        df["C"] = idx
        tm.assert_series_equal(df["C"], Series(idx, name="C"))

        df["D"] = "foo"
        df["D"] = idx
        tm.assert_series_equal(df["D"], Series(idx, name="D"))
        del df["D"]

        # assert that A & C are not sharing the same base (e.g. they
        # are copies)
        # Note: This does not hold with Copy on Write (because of lazy copying)
        v1 = df._mgr.arrays[1]
        v2 = df._mgr.arrays[2]
        tm.assert_extension_array_equal(v1, v2)
        v1base = v1._ndarray.base
        v2base = v2._ndarray.base
        if not using_copy_on_write:
            assert v1base is None or (id(v1base) != id(v2base))
        else:
            assert id(v1base) == id(v2base)

        # with nan
        df2 = df.copy()
        df2.iloc[1, 1] = NaT
        df2.iloc[1, 2] = NaT
        result = df2["B"]
        tm.assert_series_equal(notna(result), Series([True, False, True], name="B"))
        tm.assert_series_equal(df2.dtypes, df.dtypes)

    def test_setitem_periodindex(self):
        rng = period_range("1/1/2000", periods=5, name="index")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=rng)

        df["Index"] = rng
        rs = Index(df["Index"])
        tm.assert_index_equal(rs, rng, check_names=False)
        assert rs.name == "Index"
        assert rng.name == "index"

        rs = df.reset_index().set_index("index")
        assert isinstance(rs.index, PeriodIndex)
        tm.assert_index_equal(rs.index, rng)

    def test_setitem_complete_column_with_array(self):
        # GH#37954
        df = DataFrame({"a": ["one", "two", "three"], "b": [1, 2, 3]})
        arr = np.array([[1, 1], [3, 1], [5, 1]])
        df[["c", "d"]] = arr
        expected = DataFrame(
            {
                "a": ["one", "two", "three"],
                "b": [1, 2, 3],
                "c": [1, 3, 5],
                "d": [1, 1, 1],
            }
        )
        expected["c"] = expected["c"].astype(arr.dtype)
        expected["d"] = expected["d"].astype(arr.dtype)
        assert expected["c"].dtype == arr.dtype
        assert expected["d"].dtype == arr.dtype
        tm.assert_frame_equal(df, expected)

    def test_setitem_period_d_dtype(self):
        # GH 39763
        rng = period_range("2016-01-01", periods=9, freq="D", name="A")
        result = DataFrame(rng)
        expected = DataFrame(
            {"A": ["NaT", "NaT", "NaT", "NaT", "NaT", "NaT", "NaT", "NaT", "NaT"]},
            dtype="period[D]",
        )
        result.iloc[:] = rng._na_value
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["f8", "i8", "u8"])
    def test_setitem_bool_with_numeric_index(self, dtype):
        # GH#36319
        cols = Index([1, 2, 3], dtype=dtype)
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=cols)

        df[False] = ["a", "b", "c"]

        expected_cols = Index([1, 2, 3, False], dtype=object)
        if dtype == "f8":
            expected_cols = Index([1.0, 2.0, 3.0, False], dtype=object)

        tm.assert_index_equal(df.columns, expected_cols)

    @pytest.mark.parametrize("indexer", ["B", ["B"]])
    def test_setitem_frame_length_0_str_key(self, indexer):
        # GH#38831
        df = DataFrame(columns=["A", "B"])
        other = DataFrame({"B": [1, 2]})
        df[indexer] = other
        expected = DataFrame({"A": [np.nan] * 2, "B": [1, 2]})
        expected["A"] = expected["A"].astype("object")
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_duplicate_columns(self):
        # GH#15695
        cols = ["A", "B", "C"] * 2
        df = DataFrame(index=range(3), columns=cols)
        df.loc[0, "A"] = (0, 3)
        df.loc[:, "B"] = (1, 4)
        df["C"] = (2, 5)
        expected = DataFrame(
            [
                [0, 1, 2, 3, 4, 5],
                [np.nan, 1, 2, np.nan, 4, 5],
                [np.nan, 1, 2, np.nan, 4, 5],
            ],
            dtype="object",
        )

        # set these with unique columns to be extra-unambiguous
        expected[2] = expected[2].astype(np.int64)
        expected[5] = expected[5].astype(np.int64)
        expected.columns = cols

        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_duplicate_columns_size_mismatch(self):
        # GH#39510
        cols = ["A", "B", "C"] * 2
        df = DataFrame(index=range(3), columns=cols)
        with pytest.raises(ValueError, match="Columns must be same length as key"):
            df[["A"]] = (0, 3, 5)

        df2 = df.iloc[:, :3]  # unique columns
        with pytest.raises(ValueError, match="Columns must be same length as key"):
            df2[["A"]] = (0, 3, 5)

    @pytest.mark.parametrize("cols", [["a", "b", "c"], ["a", "a", "a"]])
    def test_setitem_df_wrong_column_number(self, cols):
        # GH#38604
        df = DataFrame([[1, 2, 3]], columns=cols)
        rhs = DataFrame([[10, 11]], columns=["d", "e"])
        msg = "Columns must be same length as key"
        with pytest.raises(ValueError, match=msg):
            df["a"] = rhs

    def test_setitem_listlike_indexer_duplicate_columns(self):
        # GH#38604
        df = DataFrame([[1, 2, 3]], columns=["a", "b", "b"])
        rhs = DataFrame([[10, 11, 12]], columns=["a", "b", "b"])
        df[["a", "b"]] = rhs
        expected = DataFrame([[10, 11, 12]], columns=["a", "b", "b"])
        tm.assert_frame_equal(df, expected)

        df[["c", "b"]] = rhs
        expected = DataFrame([[10, 11, 12, 10]], columns=["a", "b", "b", "c"])
        tm.assert_frame_equal(df, expected)

    def test_setitem_listlike_indexer_duplicate_columns_not_equal_length(self):
        # GH#39403
        df = DataFrame([[1, 2, 3]], columns=["a", "b", "b"])
        rhs = DataFrame([[10, 11]], columns=["a", "b"])
        msg = "Columns must be same length as key"
        with pytest.raises(ValueError, match=msg):
            df[["a", "b"]] = rhs

    def test_setitem_intervals(self):
        df = DataFrame({"A": range(10)})
        ser = cut(df["A"], 5)
        assert isinstance(ser.cat.categories, IntervalIndex)

        # B & D end up as Categoricals
        # the remainder are converted to in-line objects
        # containing an IntervalIndex.values
        df["B"] = ser
        df["C"] = np.array(ser)
        df["D"] = ser.values
        df["E"] = np.array(ser.values)
        df["F"] = ser.astype(object)

        assert isinstance(df["B"].dtype, CategoricalDtype)
        assert isinstance(df["B"].cat.categories.dtype, IntervalDtype)
        assert isinstance(df["D"].dtype, CategoricalDtype)
        assert isinstance(df["D"].cat.categories.dtype, IntervalDtype)

        # These go through the Series constructor and so get inferred back
        #  to IntervalDtype
        assert isinstance(df["C"].dtype, IntervalDtype)
        assert isinstance(df["E"].dtype, IntervalDtype)

        # But the Series constructor doesn't do inference on Series objects,
        #  so setting df["F"] doesn't get cast back to IntervalDtype
        assert is_object_dtype(df["F"])

        # they compare equal as Index
        # when converted to numpy objects
        c = lambda x: Index(np.array(x))
        tm.assert_index_equal(c(df.B), c(df.B))
        tm.assert_index_equal(c(df.B), c(df.C), check_names=False)
        tm.assert_index_equal(c(df.B), c(df.D), check_names=False)
        tm.assert_index_equal(c(df.C), c(df.D), check_names=False)

        # B & D are the same Series
        tm.assert_series_equal(df["B"], df["B"])
        tm.assert_series_equal(df["B"], df["D"], check_names=False)

        # C & E are the same Series
        tm.assert_series_equal(df["C"], df["C"])
        tm.assert_series_equal(df["C"], df["E"], check_names=False)

    def test_setitem_categorical(self):
        # GH#35369
        df = DataFrame({"h": Series(list("mn")).astype("category")})
        df.h = df.h.cat.reorder_categories(["n", "m"])
        expected = DataFrame(
            {"h": Categorical(["m", "n"]).reorder_categories(["n", "m"])}
        )
        tm.assert_frame_equal(df, expected)

    def test_setitem_with_empty_listlike(self):
        # GH#17101
        index = Index([], name="idx")
        result = DataFrame(columns=["A"], index=index)
        result["A"] = []
        expected = DataFrame(columns=["A"], index=index)
        tm.assert_index_equal(result.index, expected.index)

    @pytest.mark.parametrize(
        "cols, values, expected",
        [
            (["C", "D", "D", "a"], [1, 2, 3, 4], 4),  # with duplicates
            (["D", "C", "D", "a"], [1, 2, 3, 4], 4),  # mixed order
            (["C", "B", "B", "a"], [1, 2, 3, 4], 4),  # other duplicate cols
            (["C", "B", "a"], [1, 2, 3], 3),  # no duplicates
            (["B", "C", "a"], [3, 2, 1], 1),  # alphabetical order
            (["C", "a", "B"], [3, 2, 1], 2),  # in the middle
        ],
    )
    def test_setitem_same_column(self, cols, values, expected):
        # GH#23239
        df = DataFrame([values], columns=cols)
        df["a"] = df["a"]
        result = df["a"].values[0]
        assert result == expected

    def test_setitem_multi_index(self):
        # GH#7655, test that assigning to a sub-frame of a frame
        # with multi-index columns aligns both rows and columns
        it = ["jim", "joe", "jolie"], ["first", "last"], ["left", "center", "right"]

        cols = MultiIndex.from_product(it)
        index = date_range("20141006", periods=20)
        vals = np.random.default_rng(2).integers(1, 1000, (len(index), len(cols)))
        df = DataFrame(vals, columns=cols, index=index)

        i, j = df.index.values.copy(), it[-1][:]

        np.random.default_rng(2).shuffle(i)
        df["jim"] = df["jolie"].loc[i, ::-1]
        tm.assert_frame_equal(df["jim"], df["jolie"])

        np.random.default_rng(2).shuffle(j)
        df[("joe", "first")] = df[("jolie", "last")].loc[i, j]
        tm.assert_frame_equal(df[("joe", "first")], df[("jolie", "last")])

        np.random.default_rng(2).shuffle(j)
        df[("joe", "last")] = df[("jolie", "first")].loc[i, j]
        tm.assert_frame_equal(df[("joe", "last")], df[("jolie", "first")])

    @pytest.mark.parametrize(
        "columns,box,expected",
        [
            (
                ["A", "B", "C", "D"],
                7,
                DataFrame(
                    [[7, 7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                ["C", "D"],
                [7, 8],
                DataFrame(
                    [[1, 2, 7, 8], [3, 4, 7, 8], [5, 6, 7, 8]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                ["A", "B", "C"],
                np.array([7, 8, 9], dtype=np.int64),
                DataFrame([[7, 8, 9], [7, 8, 9], [7, 8, 9]], columns=["A", "B", "C"]),
            ),
            (
                ["B", "C", "D"],
                [[7, 8, 9], [10, 11, 12], [13, 14, 15]],
                DataFrame(
                    [[1, 7, 8, 9], [3, 10, 11, 12], [5, 13, 14, 15]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                ["C", "A", "D"],
                np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.int64),
                DataFrame(
                    [[8, 2, 7, 9], [11, 4, 10, 12], [14, 6, 13, 15]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                ["A", "C"],
                DataFrame([[7, 8], [9, 10], [11, 12]], columns=["A", "C"]),
                DataFrame(
                    [[7, 2, 8], [9, 4, 10], [11, 6, 12]], columns=["A", "B", "C"]
                ),
            ),
        ],
    )
    def test_setitem_list_missing_columns(self, columns, box, expected):
        # GH#29334
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=["A", "B"])
        df[columns] = box
        tm.assert_frame_equal(df, expected)

    def test_setitem_list_of_tuples(self, float_frame):
        tuples = list(zip(float_frame["A"], float_frame["B"]))
        float_frame["tuples"] = tuples

        result = float_frame["tuples"]
        expected = Series(tuples, index=float_frame.index, name="tuples")
        tm.assert_series_equal(result, expected)

    def test_setitem_iloc_generator(self):
        # GH#39614
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        indexer = (x for x in [1, 2])
        df.iloc[indexer] = 1
        expected = DataFrame({"a": [1, 1, 1], "b": [4, 1, 1]})
        tm.assert_frame_equal(df, expected)

    def test_setitem_iloc_two_dimensional_generator(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        indexer = (x for x in [1, 2])
        df.iloc[indexer, 1] = 1
        expected = DataFrame({"a": [1, 2, 3], "b": [4, 1, 1]})
        tm.assert_frame_equal(df, expected)

    def test_setitem_dtypes_bytes_type_to_object(self):
        # GH 20734
        index = Series(name="id", dtype="S24")
        df = DataFrame(index=index, columns=Index([], dtype="str"))
        df["a"] = Series(name="a", index=index, dtype=np.uint32)
        df["b"] = Series(name="b", index=index, dtype="S64")
        df["c"] = Series(name="c", index=index, dtype="S64")
        df["d"] = Series(name="d", index=index, dtype=np.uint8)
        result = df.dtypes
        expected = Series([np.uint32, object, object, np.uint8], index=list("abcd"))
        tm.assert_series_equal(result, expected)

    def test_boolean_mask_nullable_int64(self):
        # GH 28928
        result = DataFrame({"a": [3, 4], "b": [5, 6]}).astype(
            {"a": "int64", "b": "Int64"}
        )
        mask = Series(False, index=result.index)
        result.loc[mask, "a"] = result["a"]
        result.loc[mask, "b"] = result["b"]
        expected = DataFrame({"a": [3, 4], "b": [5, 6]}).astype(
            {"a": "int64", "b": "Int64"}
        )
        tm.assert_frame_equal(result, expected)

    def test_setitem_ea_dtype_rhs_series(self):
        # GH#47425
        df = DataFrame({"a": [1, 2]})
        df["a"] = Series([1, 2], dtype="Int64")
        expected = DataFrame({"a": [1, 2]}, dtype="Int64")
        tm.assert_frame_equal(df, expected)

    # TODO(ArrayManager) set column with 2d column array, see #44788
    @td.skip_array_manager_not_yet_implemented
    def test_setitem_npmatrix_2d(self):
        # GH#42376
        # for use-case df["x"] = sparse.random((10, 10)).mean(axis=1)
        expected = DataFrame(
            {"np-array": np.ones(10), "np-matrix": np.ones(10)}, index=np.arange(10)
        )

        a = np.ones((10, 1))
        df = DataFrame(index=np.arange(10), columns=Index([], dtype="str"))
        df["np-array"] = a

        # Instantiation of `np.matrix` gives PendingDeprecationWarning
        with tm.assert_produces_warning(PendingDeprecationWarning):
            df["np-matrix"] = np.matrix(a)

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("vals", [{}, {"d": "a"}])
    def test_setitem_aligning_dict_with_index(self, vals):
        # GH#47216
        df = DataFrame({"a": [1, 2], "b": [3, 4], **vals})
        df.loc[:, "a"] = {1: 100, 0: 200}
        df.loc[:, "c"] = {0: 5, 1: 6}
        df.loc[:, "e"] = {1: 5}
        expected = DataFrame(
            {"a": [200, 100], "b": [3, 4], **vals, "c": [5, 6], "e": [np.nan, 5]}
        )
        tm.assert_frame_equal(df, expected)

    def test_setitem_rhs_dataframe(self):
        # GH#47578
        df = DataFrame({"a": [1, 2]})
        df["a"] = DataFrame({"a": [10, 11]}, index=[1, 2])
        expected = DataFrame({"a": [np.nan, 10]})
        tm.assert_frame_equal(df, expected)

        df = DataFrame({"a": [1, 2]})
        df.isetitem(0, DataFrame({"a": [10, 11]}, index=[1, 2]))
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_overwrite_with_ea_dtype(self, any_numeric_ea_dtype):
        # GH#46896
        df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        df["a"] = DataFrame({"a": [10, 11]}, dtype=any_numeric_ea_dtype)
        expected = DataFrame(
            {
                "a": Series([10, 11], dtype=any_numeric_ea_dtype),
                "b": [2, 4],
            }
        )
        tm.assert_frame_equal(df, expected)

    def test_setitem_string_option_object_index(self):
        # GH#55638
        pytest.importorskip("pyarrow")
        df = DataFrame({"a": [1, 2]})
        with pd.option_context("future.infer_string", True):
            df["b"] = Index(["a", "b"], dtype=object)
        expected = DataFrame({"a": [1, 2], "b": Series(["a", "b"], dtype=object)})
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_midx_columns(self):
        # GH#49121
        df = DataFrame({("a", "b"): [10]})
        expected = df.copy()
        col_name = ("a", "b")
        df[col_name] = df[[col_name]]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_ea_dtype(self):
        # GH#55604
        df = DataFrame({"a": np.array([10], dtype="i8")})
        df.loc[:, "a"] = Series([11], dtype="Int64")
        expected = DataFrame({"a": np.array([11], dtype="i8")})
        tm.assert_frame_equal(df, expected)

        df = DataFrame({"a": np.array([10], dtype="i8")})
        df.iloc[:, 0] = Series([11], dtype="Int64")
        tm.assert_frame_equal(df, expected)

    def test_setitem_object_inferring(self):
        # GH#56102
        idx = Index([Timestamp("2019-12-31")], dtype=object)
        df = DataFrame({"a": [1]})
        with tm.assert_produces_warning(FutureWarning, match="infer"):
            df.loc[:, "b"] = idx
        with tm.assert_produces_warning(FutureWarning, match="infer"):
            df["c"] = idx

        expected = DataFrame(
            {
                "a": [1],
                "b": Series([Timestamp("2019-12-31")], dtype="datetime64[ns]"),
                "c": Series([Timestamp("2019-12-31")], dtype="datetime64[ns]"),
            }
        )
        tm.assert_frame_equal(df, expected)


class TestSetitemTZAwareValues:
    @pytest.fixture
    def idx(self):
        naive = DatetimeIndex(["2013-1-1 13:00", "2013-1-2 14:00"], name="B")
        idx = naive.tz_localize("US/Pacific")
        return idx

    @pytest.fixture
    def expected(self, idx):
        expected = Series(np.array(idx.tolist(), dtype="object"), name="B")
        assert expected.dtype == idx.dtype
        return expected

    def test_setitem_dt64series(self, idx, expected):
        # convert to utc
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 1)), columns=["A"])
        df["B"] = idx
        df["B"] = idx.to_series(index=[0, 1]).dt.tz_convert(None)

        result = df["B"]
        comp = Series(idx.tz_convert("UTC").tz_localize(None), name="B")
        tm.assert_series_equal(result, comp)

    def test_setitem_datetimeindex(self, idx, expected):
        # setting a DataFrame column with a tzaware DTI retains the dtype
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 1)), columns=["A"])

        # assign to frame
        df["B"] = idx
        result = df["B"]
        tm.assert_series_equal(result, expected)

    def test_setitem_object_array_of_tzaware_datetimes(self, idx, expected):
        # setting a DataFrame column with a tzaware DTI retains the dtype
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 1)), columns=["A"])

        # object array of datetimes with a tz
        df["B"] = idx.to_pydatetime()
        result = df["B"]
        tm.assert_series_equal(result, expected)


class TestDataFrameSetItemWithExpansion:
    def test_setitem_listlike_views(self, using_copy_on_write, warn_copy_on_write):
        # GH#38148
        df = DataFrame({"a": [1, 2, 3], "b": [4, 4, 6]})

        # get one column as a view of df
        ser = df["a"]

        # add columns with list-like indexer
        df[["c", "d"]] = np.array([[0.1, 0.2], [0.3, 0.4], [0.4, 0.5]])

        # edit in place the first column to check view semantics
        with tm.assert_cow_warning(warn_copy_on_write):
            df.iloc[0, 0] = 100

        if using_copy_on_write:
            expected = Series([1, 2, 3], name="a")
        else:
            expected = Series([100, 2, 3], name="a")
        tm.assert_series_equal(ser, expected)

    def test_setitem_string_column_numpy_dtype_raising(self):
        # GH#39010
        df = DataFrame([[1, 2], [3, 4]])
        df["0 - Name"] = [5, 6]
        expected = DataFrame([[1, 2, 5], [3, 4, 6]], columns=[0, 1, "0 - Name"])
        tm.assert_frame_equal(df, expected)

    def test_setitem_empty_df_duplicate_columns(self, using_copy_on_write):
        # GH#38521
        df = DataFrame(columns=["a", "b", "b"], dtype="float64")
        df.loc[:, "a"] = list(range(2))
        expected = DataFrame(
            [[0, np.nan, np.nan], [1, np.nan, np.nan]], columns=["a", "b", "b"]
        )
        tm.assert_frame_equal(df, expected)

    def test_setitem_with_expansion_categorical_dtype(self):
        # assignment
        df = DataFrame(
            {
                "value": np.array(
                    np.random.default_rng(2).integers(0, 10000, 100), dtype="int32"
                )
            }
        )
        labels = Categorical([f"{i} - {i + 499}" for i in range(0, 10000, 500)])

        df = df.sort_values(by=["value"], ascending=True)
        ser = cut(df.value, range(0, 10500, 500), right=False, labels=labels)
        cat = ser.values

        # setting with a Categorical
        df["D"] = cat
        result = df.dtypes
        expected = Series(
            [np.dtype("int32"), CategoricalDtype(categories=labels, ordered=False)],
            index=["value", "D"],
        )
        tm.assert_series_equal(result, expected)

        # setting with a Series
        df["E"] = ser
        result = df.dtypes
        expected = Series(
            [
                np.dtype("int32"),
                CategoricalDtype(categories=labels, ordered=False),
                CategoricalDtype(categories=labels, ordered=False),
            ],
            index=["value", "D", "E"],
        )
        tm.assert_series_equal(result, expected)

        result1 = df["D"]
        result2 = df["E"]
        tm.assert_categorical_equal(result1._mgr.array, cat)

        # sorting
        ser.name = "E"
        tm.assert_series_equal(result2.sort_index(), ser.sort_index())

    def test_setitem_scalars_no_index(self):
        # GH#16823 / GH#17894
        df = DataFrame()
        df["foo"] = 1
        expected = DataFrame(columns=Index(["foo"], dtype=object)).astype(np.int64)
        tm.assert_frame_equal(df, expected)

    def test_setitem_newcol_tuple_key(self, float_frame):
        assert (
            "A",
            "B",
        ) not in float_frame.columns
        float_frame["A", "B"] = float_frame["A"]
        assert ("A", "B") in float_frame.columns

        result = float_frame["A", "B"]
        expected = float_frame["A"]
        tm.assert_series_equal(result, expected, check_names=False)

    def test_frame_setitem_newcol_timestamp(self):
        # GH#2155
        columns = date_range(start="1/1/2012", end="2/1/2012", freq=BDay())
        data = DataFrame(columns=columns, index=range(10))
        t = datetime(2012, 11, 1)
        ts = Timestamp(t)
        data[ts] = np.nan  # works, mostly a smoke-test
        assert np.isnan(data[ts]).all()

    def test_frame_setitem_rangeindex_into_new_col(self):
        # GH#47128
        df = DataFrame({"a": ["a", "b"]})
        df["b"] = df.index
        df.loc[[False, True], "b"] = 100
        result = df.loc[[1], :]
        expected = DataFrame({"a": ["b"], "b": [100]}, index=[1])
        tm.assert_frame_equal(result, expected)

    def test_setitem_frame_keep_ea_dtype(self, any_numeric_ea_dtype):
        # GH#46896
        df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        df["c"] = DataFrame({"a": [10, 11]}, dtype=any_numeric_ea_dtype)
        expected = DataFrame(
            {
                "a": [1, 3],
                "b": [2, 4],
                "c": Series([10, 11], dtype=any_numeric_ea_dtype),
            }
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_expansion_with_timedelta_type(self):
        result = DataFrame(columns=list("abc"))
        result.loc[0] = {
            "a": pd.to_timedelta(5, unit="s"),
            "b": pd.to_timedelta(72, unit="s"),
            "c": "23",
        }
        expected = DataFrame(
            [[pd.Timedelta("0 days 00:00:05"), pd.Timedelta("0 days 00:01:12"), "23"]],
            index=Index([0]),
            columns=(["a", "b", "c"]),
        )
        tm.assert_frame_equal(result, expected)


class TestDataFrameSetItemSlicing:
    def test_setitem_slice_position(self):
        # GH#31469
        df = DataFrame(np.zeros((100, 1)))
        df[-4:] = 1
        arr = np.zeros((100, 1))
        arr[-4:] = 1
        expected = DataFrame(arr)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer", [tm.setitem, tm.iloc])
    @pytest.mark.parametrize("box", [Series, np.array, list, pd.array])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_setitem_slice_indexer_broadcasting_rhs(self, n, box, indexer):
        # GH#40440
        df = DataFrame([[1, 3, 5]] + [[2, 4, 6]] * n, columns=["a", "b", "c"])
        indexer(df)[1:] = box([10, 11, 12])
        expected = DataFrame([[1, 3, 5]] + [[10, 11, 12]] * n, columns=["a", "b", "c"])
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("box", [Series, np.array, list, pd.array])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_setitem_list_indexer_broadcasting_rhs(self, n, box):
        # GH#40440
        df = DataFrame([[1, 3, 5]] + [[2, 4, 6]] * n, columns=["a", "b", "c"])
        df.iloc[list(range(1, n + 1))] = box([10, 11, 12])
        expected = DataFrame([[1, 3, 5]] + [[10, 11, 12]] * n, columns=["a", "b", "c"])
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer", [tm.setitem, tm.iloc])
    @pytest.mark.parametrize("box", [Series, np.array, list, pd.array])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_setitem_slice_broadcasting_rhs_mixed_dtypes(self, n, box, indexer):
        # GH#40440
        df = DataFrame(
            [[1, 3, 5], ["x", "y", "z"]] + [[2, 4, 6]] * n, columns=["a", "b", "c"]
        )
        indexer(df)[1:] = box([10, 11, 12])
        expected = DataFrame(
            [[1, 3, 5]] + [[10, 11, 12]] * (n + 1),
            columns=["a", "b", "c"],
            dtype="object",
        )
        tm.assert_frame_equal(df, expected)


class TestDataFrameSetItemCallable:
    def test_setitem_callable(self):
        # GH#12533
        df = DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
        df[lambda x: "A"] = [11, 12, 13, 14]

        exp = DataFrame({"A": [11, 12, 13, 14], "B": [5, 6, 7, 8]})
        tm.assert_frame_equal(df, exp)

    def test_setitem_other_callable(self):
        # GH#13299
        def inc(x):
            return x + 1

        # Set dtype object straight away to avoid upcast when setting inc below
        df = DataFrame([[-1, 1], [1, -1]], dtype=object)
        df[df > 0] = inc

        expected = DataFrame([[-1, inc], [inc, -1]])
        tm.assert_frame_equal(df, expected)


class TestDataFrameSetItemBooleanMask:
    @td.skip_array_manager_invalid_test  # TODO(ArrayManager) rewrite not using .values
    @pytest.mark.parametrize(
        "mask_type",
        [lambda df: df > np.abs(df) / 2, lambda df: (df > np.abs(df) / 2).values],
        ids=["dataframe", "array"],
    )
    def test_setitem_boolean_mask(self, mask_type, float_frame):
        # Test for issue #18582
        df = float_frame.copy()
        mask = mask_type(df)

        # index with boolean mask
        result = df.copy()
        result[mask] = np.nan

        expected = df.values.copy()
        expected[np.array(mask)] = np.nan
        expected = DataFrame(expected, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(reason="Currently empty indexers are treated as all False")
    @pytest.mark.parametrize("box", [list, np.array, Series])
    def test_setitem_loc_empty_indexer_raises_with_non_empty_value(self, box):
        # GH#37672
        df = DataFrame({"a": ["a"], "b": [1], "c": [1]})
        if box == Series:
            indexer = box([], dtype="object")
        else:
            indexer = box([])
        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df.loc[indexer, ["b"]] = [1]

    @pytest.mark.parametrize("box", [list, np.array, Series])
    def test_setitem_loc_only_false_indexer_dtype_changed(self, box):
        # GH#37550
        # Dtype is only changed when value to set is a Series and indexer is
        # empty/bool all False
        df = DataFrame({"a": ["a"], "b": [1], "c": [1]})
        indexer = box([False])
        df.loc[indexer, ["b"]] = 10 - df["c"]
        expected = DataFrame({"a": ["a"], "b": [1], "c": [1]})
        tm.assert_frame_equal(df, expected)

        df.loc[indexer, ["b"]] = 9
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer", [tm.setitem, tm.loc])
    def test_setitem_boolean_mask_aligning(self, indexer):
        # GH#39931
        df = DataFrame({"a": [1, 4, 2, 3], "b": [5, 6, 7, 8]})
        expected = df.copy()
        mask = df["a"] >= 3
        indexer(df)[mask] = indexer(df)[mask].sort_values("a")
        tm.assert_frame_equal(df, expected)

    def test_setitem_mask_categorical(self):
        # assign multiple rows (mixed values) (-> array) -> exp_multi_row
        # changed multiple rows
        cats2 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        idx2 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values2 = [1, 1, 2, 2, 1, 1, 1]
        exp_multi_row = DataFrame({"cats": cats2, "values": values2}, index=idx2)

        catsf = Categorical(
            ["a", "a", "c", "c", "a", "a", "a"], categories=["a", "b", "c"]
        )
        idxf = Index(["h", "i", "j", "k", "l", "m", "n"])
        valuesf = [1, 1, 3, 3, 1, 1, 1]
        df = DataFrame({"cats": catsf, "values": valuesf}, index=idxf)

        exp_fancy = exp_multi_row.copy()
        exp_fancy["cats"] = exp_fancy["cats"].cat.set_categories(["a", "b", "c"])

        mask = df["cats"] == "c"
        df[mask] = ["b", 2]
        # category c is kept in .categories
        tm.assert_frame_equal(df, exp_fancy)

    @pytest.mark.parametrize("dtype", ["float", "int64"])
    @pytest.mark.parametrize("kwargs", [{}, {"index": [1]}, {"columns": ["A"]}])
    def test_setitem_empty_frame_with_boolean(self, dtype, kwargs):
        # see GH#10126
        kwargs["dtype"] = dtype
        df = DataFrame(**kwargs)

        df2 = df.copy()
        df[df > df2] = 47
        tm.assert_frame_equal(df, df2)

    def test_setitem_boolean_indexing(self):
        idx = list(range(3))
        cols = ["A", "B", "C"]
        df1 = DataFrame(
            index=idx,
            columns=cols,
            data=np.array(
                [[0.0, 0.5, 1.0], [1.5, 2.0, 2.5], [3.0, 3.5, 4.0]], dtype=float
            ),
        )
        df2 = DataFrame(index=idx, columns=cols, data=np.ones((len(idx), len(cols))))

        expected = DataFrame(
            index=idx,
            columns=cols,
            data=np.array([[0.0, 0.5, 1.0], [1.5, 2.0, -1], [-1, -1, -1]], dtype=float),
        )

        df1[df1 > 2.0 * df2] = -1
        tm.assert_frame_equal(df1, expected)
        with pytest.raises(ValueError, match="Item wrong length"):
            df1[df1.index[:-1] > 2] = -1

    def test_loc_setitem_all_false_boolean_two_blocks(self):
        # GH#40885
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": "a"})
        expected = df.copy()
        indexer = Series([False, False], name="c")
        df.loc[indexer, ["b"]] = DataFrame({"b": [5, 6]}, index=[0, 1])
        tm.assert_frame_equal(df, expected)

    def test_setitem_ea_boolean_mask(self):
        # GH#47125
        df = DataFrame([[-1, 2], [3, -4]])
        expected = DataFrame([[0, 2], [3, 0]])
        boolean_indexer = DataFrame(
            {
                0: Series([True, False], dtype="boolean"),
                1: Series([pd.NA, True], dtype="boolean"),
            }
        )
        df[boolean_indexer] = 0
        tm.assert_frame_equal(df, expected)


class TestDataFrameSetitemCopyViewSemantics:
    def test_setitem_always_copy(self, float_frame):
        assert "E" not in float_frame.columns
        s = float_frame["A"].copy()
        float_frame["E"] = s

        float_frame.iloc[5:10, float_frame.columns.get_loc("E")] = np.nan
        assert notna(s[5:10]).all()

    @pytest.mark.parametrize("consolidate", [True, False])
    def test_setitem_partial_column_inplace(
        self, consolidate, using_array_manager, using_copy_on_write
    ):
        # This setting should be in-place, regardless of whether frame is
        #  single-block or multi-block
        # GH#304 this used to be incorrectly not-inplace, in which case
        #  we needed to ensure _item_cache was cleared.

        df = DataFrame(
            {"x": [1.1, 2.1, 3.1, 4.1], "y": [5.1, 6.1, 7.1, 8.1]}, index=[0, 1, 2, 3]
        )
        df.insert(2, "z", np.nan)
        if not using_array_manager:
            if consolidate:
                df._consolidate_inplace()
                assert len(df._mgr.blocks) == 1
            else:
                assert len(df._mgr.blocks) == 2

        zvals = df["z"]._values

        df.loc[2:, "z"] = 42

        expected = Series([np.nan, np.nan, 42, 42], index=df.index, name="z")
        tm.assert_series_equal(df["z"], expected)

        # check setting occurred in-place
        if not using_copy_on_write:
            tm.assert_numpy_array_equal(zvals, expected.values)
            assert np.shares_memory(zvals, df["z"]._values)

    def test_setitem_duplicate_columns_not_inplace(self):
        # GH#39510
        cols = ["A", "B"] * 2
        df = DataFrame(0.0, index=[0], columns=cols)
        df_copy = df.copy()
        df_view = df[:]
        df["B"] = (2, 5)

        expected = DataFrame([[0.0, 2, 0.0, 5]], columns=cols)
        tm.assert_frame_equal(df_view, df_copy)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "value", [1, np.array([[1], [1]], dtype="int64"), [[1], [1]]]
    )
    def test_setitem_same_dtype_not_inplace(self, value, using_array_manager):
        # GH#39510
        cols = ["A", "B"]
        df = DataFrame(0, index=[0, 1], columns=cols)
        df_copy = df.copy()
        df_view = df[:]
        df[["B"]] = value

        expected = DataFrame([[0, 1], [0, 1]], columns=cols)
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df_view, df_copy)

    @pytest.mark.parametrize("value", [1.0, np.array([[1.0], [1.0]]), [[1.0], [1.0]]])
    def test_setitem_listlike_key_scalar_value_not_inplace(self, value):
        # GH#39510
        cols = ["A", "B"]
        df = DataFrame(0, index=[0, 1], columns=cols)
        df_copy = df.copy()
        df_view = df[:]
        df[["B"]] = value

        expected = DataFrame([[0, 1.0], [0, 1.0]], columns=cols)
        tm.assert_frame_equal(df_view, df_copy)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "indexer",
        [
            "a",
            ["a"],
            pytest.param(
                [True, False],
                marks=pytest.mark.xfail(
                    reason="Boolean indexer incorrectly setting inplace",
                    strict=False,  # passing on some builds, no obvious pattern
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "value, set_value",
        [
            (1, 5),
            (1.0, 5.0),
            (Timestamp("2020-12-31"), Timestamp("2021-12-31")),
            ("a", "b"),
        ],
    )
    def test_setitem_not_operating_inplace(self, value, set_value, indexer):
        # GH#43406
        df = DataFrame({"a": value}, index=[0, 1])
        expected = df.copy()
        view = df[:]
        df[indexer] = set_value
        tm.assert_frame_equal(view, expected)

    @td.skip_array_manager_invalid_test
    def test_setitem_column_update_inplace(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # https://github.com/pandas-dev/pandas/issues/47172

        labels = [f"c{i}" for i in range(10)]
        df = DataFrame({col: np.zeros(len(labels)) for col in labels}, index=labels)
        values = df._mgr.blocks[0].values

        with tm.raises_chained_assignment_error():
            for label in df.columns:
                df[label][label] = 1
        if not using_copy_on_write:
            # diagonal values all updated
            assert np.all(values[np.arange(10), np.arange(10)] == 1)
        else:
            # original dataframe not updated
            assert np.all(values[np.arange(10), np.arange(10)] == 0)

    def test_setitem_column_frame_as_category(self):
        # GH31581
        df = DataFrame([1, 2, 3])
        df["col1"] = DataFrame([1, 2, 3], dtype="category")
        df["col2"] = Series([1, 2, 3], dtype="category")

        expected_types = Series(
            ["int64", "category", "category"], index=[0, "col1", "col2"], dtype=object
        )
        tm.assert_series_equal(df.dtypes, expected_types)

    @pytest.mark.parametrize("dtype", ["int64", "Int64"])
    def test_setitem_iloc_with_numpy_array(self, dtype):
        # GH-33828
        df = DataFrame({"a": np.ones(3)}, dtype=dtype)
        df.iloc[np.array([0]), np.array([0])] = np.array([[2]])

        expected = DataFrame({"a": [2, 1, 1]}, dtype=dtype)
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_dup_cols_dtype(self):
        # GH#53143
        df = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=["a", "b", "a", "c"])
        rhs = DataFrame([[0, 1.5], [2, 2.5]], columns=["a", "a"])
        df["a"] = rhs
        expected = DataFrame(
            [[0, 2, 1.5, 4], [2, 5, 2.5, 7]], columns=["a", "b", "a", "c"]
        )
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
        rhs = DataFrame([[0, 1.5], [2, 2.5]], columns=["a", "a"])
        df["a"] = rhs
        expected = DataFrame([[0, 1.5, 3], [2, 2.5, 6]], columns=["a", "a", "b"])
        tm.assert_frame_equal(df, expected)

    def test_frame_setitem_empty_dataframe(self):
        # GH#28871
        dti = DatetimeIndex(["2000-01-01"], dtype="M8[ns]", name="date")
        df = DataFrame({"date": dti}).set_index("date")
        df = df[0:0].copy()

        df["3010"] = None
        df["2010"] = None

        expected = DataFrame(
            [],
            columns=["3010", "2010"],
            index=dti[:0],
        )
        tm.assert_frame_equal(df, expected)


def test_full_setter_loc_incompatible_dtype():
    # https://github.com/pandas-dev/pandas/issues/55791
    df = DataFrame({"a": [1, 2]})
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        df.loc[:, "a"] = True
    expected = DataFrame({"a": [True, True]})
    tm.assert_frame_equal(df, expected)

    df = DataFrame({"a": [1, 2]})
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        df.loc[:, "a"] = {0: 3.5, 1: 4.5}
    expected = DataFrame({"a": [3.5, 4.5]})
    tm.assert_frame_equal(df, expected)

    df = DataFrame({"a": [1, 2]})
    df.loc[:, "a"] = {0: 3, 1: 4}
    expected = DataFrame({"a": [3, 4]})
    tm.assert_frame_equal(df, expected)


def test_setitem_partial_row_multiple_columns():
    # https://github.com/pandas-dev/pandas/issues/56503
    df = DataFrame({"A": [1, 2, 3], "B": [4.0, 5, 6]})
    # should not warn
    df.loc[df.index <= 1, ["F", "G"]] = (1, "abc")
    expected = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4.0, 5, 6],
            "F": [1.0, 1, float("nan")],
            "G": ["abc", "abc", float("nan")],
        }
    )
    tm.assert_frame_equal(df, expected)