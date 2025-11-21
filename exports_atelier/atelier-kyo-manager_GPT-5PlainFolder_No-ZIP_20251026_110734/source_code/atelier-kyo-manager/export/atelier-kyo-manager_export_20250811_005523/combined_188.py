
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_einsum.py ===
import itertools

import pytest

import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    suppress_warnings,
)

# Setup for optimize einsum
chars = 'abcdefghij'
sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3])
global_size_dict = dict(zip(chars, sizes))


class TestEinsum:
    @pytest.mark.parametrize("do_opt", [True, False])
    @pytest.mark.parametrize("einsum_fn", [np.einsum, np.einsum_path])
    def test_einsum_errors(self, do_opt, einsum_fn):
        # Need enough arguments
        assert_raises(ValueError, einsum_fn, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "", optimize=do_opt)

        # subscripts must be a string
        assert_raises(TypeError, einsum_fn, 0, 0, optimize=do_opt)

        # issue 4528 revealed a segfault with this call
        assert_raises(TypeError, einsum_fn, *(None,) * 63, optimize=do_opt)

        # number of operands must match count in subscripts string
        assert_raises(ValueError, einsum_fn, "", 0, 0, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, ",", 0, [0], [0],
                      optimize=do_opt)
        assert_raises(ValueError, einsum_fn, ",", [0], optimize=do_opt)

        # can't have more subscripts than dimensions in the operand
        assert_raises(ValueError, einsum_fn, "i", 0, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "ij", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "...i", 0, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "i...j", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "i...", 0, optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "ij...", [0, 0], optimize=do_opt)

        # invalid ellipsis
        assert_raises(ValueError, einsum_fn, "i..", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, ".i...", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "j->..j", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "j->.j...", [0, 0],
                      optimize=do_opt)

        # invalid subscript character
        assert_raises(ValueError, einsum_fn, "i%...", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "...j$", [0, 0], optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "i->&", [0, 0], optimize=do_opt)

        # output subscripts must appear in input
        assert_raises(ValueError, einsum_fn, "i->ij", [0, 0], optimize=do_opt)

        # output subscripts may only be specified once
        assert_raises(ValueError, einsum_fn, "ij->jij", [[0, 0], [0, 0]],
                      optimize=do_opt)

        # dimensions must match when being collapsed
        assert_raises(ValueError, einsum_fn, "ii",
                      np.arange(6).reshape(2, 3), optimize=do_opt)
        assert_raises(ValueError, einsum_fn, "ii->i",
                      np.arange(6).reshape(2, 3), optimize=do_opt)

        with assert_raises_regex(ValueError, "'b'"):
            # gh-11221 - 'c' erroneously appeared in the error message
            a = np.ones((3, 3, 4, 5, 6))
            b = np.ones((3, 4, 5))
            einsum_fn('aabcb,abc', a, b)

    def test_einsum_sorting_behavior(self):
        # Case 1: 26 dimensions (all lowercase indices)
        n1 = 26
        x1 = np.random.random((1,) * n1)
        path1 = np.einsum_path(x1, range(n1))[1]  # Get einsum path details
        output_indices1 = path1.split("->")[-1].strip()  # Extract output indices
        # Assert indices are only uppercase letters and sorted correctly
        assert all(c.isupper() for c in output_indices1), (
            "Output indices for n=26 should use uppercase letters only: "
            f"{output_indices1}"
        )
        assert_equal(
            output_indices1,
            ''.join(sorted(output_indices1)),
            err_msg=(
                "Output indices for n=26 are not lexicographically sorted: "
                f"{output_indices1}"
            )
        )

        # Case 2: 27 dimensions (includes uppercase indices)
        n2 = 27
        x2 = np.random.random((1,) * n2)
        path2 = np.einsum_path(x2, range(n2))[1]
        output_indices2 = path2.split("->")[-1].strip()
        # Assert indices include both uppercase and lowercase letters
        assert any(c.islower() for c in output_indices2), (
            "Output indices for n=27 should include uppercase letters: "
            f"{output_indices2}"
        )
        # Assert output indices are sorted uppercase before lowercase
        assert_equal(
            output_indices2,
            ''.join(sorted(output_indices2)),
            err_msg=(
                "Output indices for n=27 are not lexicographically sorted: "
                f"{output_indices2}"
            )
        )

        # Additional Check: Ensure dimensions correspond correctly to indices
        # Generate expected mapping of dimensions to indices
        expected_indices = [
            chr(i + ord('A')) if i < 26 else chr(i - 26 + ord('a'))
            for i in range(n2)
        ]
        assert_equal(
            output_indices2,
            ''.join(expected_indices),
            err_msg=(
                "Output indices do not map to the correct dimensions. Expected: "
                f"{''.join(expected_indices)}, Got: {output_indices2}"
            )
        )

    @pytest.mark.parametrize("do_opt", [True, False])
    def test_einsum_specific_errors(self, do_opt):
        # out parameter must be an array
        assert_raises(TypeError, np.einsum, "", 0, out='test',
                      optimize=do_opt)

        # order parameter must be a valid order
        assert_raises(ValueError, np.einsum, "", 0, order='W',
                      optimize=do_opt)

        # casting parameter must be a valid casting
        assert_raises(ValueError, np.einsum, "", 0, casting='blah',
                      optimize=do_opt)

        # dtype parameter must be a valid dtype
        assert_raises(TypeError, np.einsum, "", 0, dtype='bad_data_type',
                      optimize=do_opt)

        # other keyword arguments are rejected
        assert_raises(TypeError, np.einsum, "", 0, bad_arg=0, optimize=do_opt)

        # broadcasting to new dimensions must be enabled explicitly
        assert_raises(ValueError, np.einsum, "i", np.arange(6).reshape(2, 3),
                      optimize=do_opt)
        assert_raises(ValueError, np.einsum, "i->i", [[0, 1], [0, 1]],
                      out=np.arange(4).reshape(2, 2), optimize=do_opt)

        # Check order kwarg, asanyarray allows 1d to pass through
        assert_raises(ValueError, np.einsum, "i->i",
                      np.arange(6).reshape(-1, 1), optimize=do_opt, order='d')

    def test_einsum_object_errors(self):
        # Exceptions created by object arithmetic should
        # successfully propagate

        class CustomException(Exception):
            pass

        class DestructoBox:

            def __init__(self, value, destruct):
                self._val = value
                self._destruct = destruct

            def __add__(self, other):
                tmp = self._val + other._val
                if tmp >= self._destruct:
                    raise CustomException
                else:
                    self._val = tmp
                    return self

            def __radd__(self, other):
                if other == 0:
                    return self
                else:
                    return self.__add__(other)

            def __mul__(self, other):
                tmp = self._val * other._val
                if tmp >= self._destruct:
                    raise CustomException
                else:
                    self._val = tmp
                    return self

            def __rmul__(self, other):
                if other == 0:
                    return self
                else:
                    return self.__mul__(other)

        a = np.array([DestructoBox(i, 5) for i in range(1, 10)],
                     dtype='object').reshape(3, 3)

        # raised from unbuffered_loop_nop1_ndim2
        assert_raises(CustomException, np.einsum, "ij->i", a)

        # raised from unbuffered_loop_nop1_ndim3
        b = np.array([DestructoBox(i, 100) for i in range(27)],
                     dtype='object').reshape(3, 3, 3)
        assert_raises(CustomException, np.einsum, "i...k->...", b)

        # raised from unbuffered_loop_nop2_ndim2
        b = np.array([DestructoBox(i, 55) for i in range(1, 4)],
                     dtype='object')
        assert_raises(CustomException, np.einsum, "ij, j", a, b)

        # raised from unbuffered_loop_nop2_ndim3
        assert_raises(CustomException, np.einsum, "ij, jh", a, a)

        # raised from PyArray_EinsteinSum
        assert_raises(CustomException, np.einsum, "ij->", a)

    def test_einsum_views(self):
        # pass-through
        for do_opt in [True, False]:
            a = np.arange(6)
            a.shape = (2, 3)

            b = np.einsum("...", a, optimize=do_opt)
            assert_(b.base is a)

            b = np.einsum(a, [Ellipsis], optimize=do_opt)
            assert_(b.base is a)

            b = np.einsum("ij", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a)

            b = np.einsum(a, [0, 1], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a)

            # output is writeable whenever input is writeable
            b = np.einsum("...", a, optimize=do_opt)
            assert_(b.flags['WRITEABLE'])
            a.flags['WRITEABLE'] = False
            b = np.einsum("...", a, optimize=do_opt)
            assert_(not b.flags['WRITEABLE'])

            # transpose
            a = np.arange(6)
            a.shape = (2, 3)

            b = np.einsum("ji", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a.T)

            b = np.einsum(a, [1, 0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a.T)

            # diagonal
            a = np.arange(9)
            a.shape = (3, 3)

            b = np.einsum("ii->i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[i, i] for i in range(3)])

            b = np.einsum(a, [0, 0], [0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[i, i] for i in range(3)])

            # diagonal with various ways of broadcasting an additional dimension
            a = np.arange(27)
            a.shape = (3, 3, 3)

            b = np.einsum("...ii->...i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)] for x in a])

            b = np.einsum(a, [Ellipsis, 0, 0], [Ellipsis, 0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)] for x in a])

            b = np.einsum("ii...->...i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)]
                             for x in a.transpose(2, 0, 1)])

            b = np.einsum(a, [0, 0, Ellipsis], [Ellipsis, 0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)]
                             for x in a.transpose(2, 0, 1)])

            b = np.einsum("...ii->i...", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[:, i, i] for i in range(3)])

            b = np.einsum(a, [Ellipsis, 0, 0], [0, Ellipsis], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[:, i, i] for i in range(3)])

            b = np.einsum("jii->ij", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[:, i, i] for i in range(3)])

            b = np.einsum(a, [1, 0, 0], [0, 1], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[:, i, i] for i in range(3)])

            b = np.einsum("ii...->i...", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a.transpose(2, 0, 1)[:, i, i] for i in range(3)])

            b = np.einsum(a, [0, 0, Ellipsis], [0, Ellipsis], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a.transpose(2, 0, 1)[:, i, i] for i in range(3)])

            b = np.einsum("i...i->i...", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a.transpose(1, 0, 2)[:, i, i] for i in range(3)])

            b = np.einsum(a, [0, Ellipsis, 0], [0, Ellipsis], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a.transpose(1, 0, 2)[:, i, i] for i in range(3)])

            b = np.einsum("i...i->...i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)]
                             for x in a.transpose(1, 0, 2)])

            b = np.einsum(a, [0, Ellipsis, 0], [Ellipsis, 0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [[x[i, i] for i in range(3)]
                             for x in a.transpose(1, 0, 2)])

            # triple diagonal
            a = np.arange(27)
            a.shape = (3, 3, 3)

            b = np.einsum("iii->i", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[i, i, i] for i in range(3)])

            b = np.einsum(a, [0, 0, 0], [0], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, [a[i, i, i] for i in range(3)])

            # swap axes
            a = np.arange(24)
            a.shape = (2, 3, 4)

            b = np.einsum("ijk->jik", a, optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a.swapaxes(0, 1))

            b = np.einsum(a, [0, 1, 2], [1, 0, 2], optimize=do_opt)
            assert_(b.base is a)
            assert_equal(b, a.swapaxes(0, 1))

    def check_einsum_sums(self, dtype, do_opt=False):
        dtype = np.dtype(dtype)
        # Check various sums.  Does many sizes to exercise unrolled loops.

        # sum(a, axis=-1)
        for n in range(1, 17):
            a = np.arange(n, dtype=dtype)
            b = np.sum(a, axis=-1)
            if hasattr(b, 'astype'):
                b = b.astype(dtype)
            assert_equal(np.einsum("i->", a, optimize=do_opt), b)
            assert_equal(np.einsum(a, [0], [], optimize=do_opt), b)

        for n in range(1, 17):
            a = np.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            b = np.sum(a, axis=-1)
            if hasattr(b, 'astype'):
                b = b.astype(dtype)
            assert_equal(np.einsum("...i->...", a, optimize=do_opt), b)
            assert_equal(np.einsum(a, [Ellipsis, 0], [Ellipsis], optimize=do_opt), b)

        # sum(a, axis=0)
        for n in range(1, 17):
            a = np.arange(2 * n, dtype=dtype).reshape(2, n)
            b = np.sum(a, axis=0)
            if hasattr(b, 'astype'):
                b = b.astype(dtype)
            assert_equal(np.einsum("i...->...", a, optimize=do_opt), b)
            assert_equal(np.einsum(a, [0, Ellipsis], [Ellipsis], optimize=do_opt), b)

        for n in range(1, 17):
            a = np.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            b = np.sum(a, axis=0)
            if hasattr(b, 'astype'):
                b = b.astype(dtype)
            assert_equal(np.einsum("i...->...", a, optimize=do_opt), b)
            assert_equal(np.einsum(a, [0, Ellipsis], [Ellipsis], optimize=do_opt), b)

        # trace(a)
        for n in range(1, 17):
            a = np.arange(n * n, dtype=dtype).reshape(n, n)
            b = np.trace(a)
            if hasattr(b, 'astype'):
                b = b.astype(dtype)
            assert_equal(np.einsum("ii", a, optimize=do_opt), b)
            assert_equal(np.einsum(a, [0, 0], optimize=do_opt), b)

            # gh-15961: should accept numpy int64 type in subscript list
            np_array = np.asarray([0, 0])
            assert_equal(np.einsum(a, np_array, optimize=do_opt), b)
            assert_equal(np.einsum(a, list(np_array), optimize=do_opt), b)

        # multiply(a, b)
        assert_equal(np.einsum("..., ...", 3, 4), 12)  # scalar case
        for n in range(1, 17):
            a = np.arange(3 * n, dtype=dtype).reshape(3, n)
            b = np.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            assert_equal(np.einsum("..., ...", a, b, optimize=do_opt),
                         np.multiply(a, b))
            assert_equal(np.einsum(a, [Ellipsis], b, [Ellipsis], optimize=do_opt),
                         np.multiply(a, b))

        # inner(a,b)
        for n in range(1, 17):
            a = np.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
            b = np.arange(n, dtype=dtype)
            assert_equal(np.einsum("...i, ...i", a, b, optimize=do_opt), np.inner(a, b))
            assert_equal(np.einsum(a, [Ellipsis, 0], b, [Ellipsis, 0], optimize=do_opt),
                         np.inner(a, b))

        for n in range(1, 11):
            a = np.arange(n * 3 * 2, dtype=dtype).reshape(n, 3, 2)
            b = np.arange(n, dtype=dtype)
            assert_equal(np.einsum("i..., i...", a, b, optimize=do_opt),
                         np.inner(a.T, b.T).T)
            assert_equal(np.einsum(a, [0, Ellipsis], b, [0, Ellipsis], optimize=do_opt),
                         np.inner(a.T, b.T).T)

        # outer(a,b)
        for n in range(1, 17):
            a = np.arange(3, dtype=dtype) + 1
            b = np.arange(n, dtype=dtype) + 1
            assert_equal(np.einsum("i,j", a, b, optimize=do_opt),
                         np.outer(a, b))
            assert_equal(np.einsum(a, [0], b, [1], optimize=do_opt),
                         np.outer(a, b))

        # Suppress the complex warnings for the 'as f8' tests
        with suppress_warnings() as sup:
            sup.filter(np.exceptions.ComplexWarning)

            # matvec(a,b) / a.dot(b) where a is matrix, b is vector
            for n in range(1, 17):
                a = np.arange(4 * n, dtype=dtype).reshape(4, n)
                b = np.arange(n, dtype=dtype)
                assert_equal(np.einsum("ij, j", a, b, optimize=do_opt),
                             np.dot(a, b))
                assert_equal(np.einsum(a, [0, 1], b, [1], optimize=do_opt),
                             np.dot(a, b))

                c = np.arange(4, dtype=dtype)
                np.einsum("ij,j", a, b, out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(a.astype('f8'),
                                    b.astype('f8')).astype(dtype))
                c[...] = 0
                np.einsum(a, [0, 1], b, [1], out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(a.astype('f8'),
                                    b.astype('f8')).astype(dtype))

            for n in range(1, 17):
                a = np.arange(4 * n, dtype=dtype).reshape(4, n)
                b = np.arange(n, dtype=dtype)
                assert_equal(np.einsum("ji,j", a.T, b.T, optimize=do_opt),
                             np.dot(b.T, a.T))
                assert_equal(np.einsum(a.T, [1, 0], b.T, [1], optimize=do_opt),
                             np.dot(b.T, a.T))

                c = np.arange(4, dtype=dtype)
                np.einsum("ji,j", a.T, b.T, out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(b.T.astype('f8'),
                                    a.T.astype('f8')).astype(dtype))
                c[...] = 0
                np.einsum(a.T, [1, 0], b.T, [1], out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(b.T.astype('f8'),
                                    a.T.astype('f8')).astype(dtype))

            # matmat(a,b) / a.dot(b) where a is matrix, b is matrix
            for n in range(1, 17):
                if n < 8 or dtype != 'f2':
                    a = np.arange(4 * n, dtype=dtype).reshape(4, n)
                    b = np.arange(n * 6, dtype=dtype).reshape(n, 6)
                    assert_equal(np.einsum("ij,jk", a, b, optimize=do_opt),
                                 np.dot(a, b))
                    assert_equal(np.einsum(a, [0, 1], b, [1, 2], optimize=do_opt),
                                 np.dot(a, b))

            for n in range(1, 17):
                a = np.arange(4 * n, dtype=dtype).reshape(4, n)
                b = np.arange(n * 6, dtype=dtype).reshape(n, 6)
                c = np.arange(24, dtype=dtype).reshape(4, 6)
                np.einsum("ij,jk", a, b, out=c, dtype='f8', casting='unsafe',
                          optimize=do_opt)
                assert_equal(c,
                             np.dot(a.astype('f8'),
                                    b.astype('f8')).astype(dtype))
                c[...] = 0
                np.einsum(a, [0, 1], b, [1, 2], out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c,
                             np.dot(a.astype('f8'),
                                    b.astype('f8')).astype(dtype))

            # matrix triple product (note this is not currently an efficient
            # way to multiply 3 matrices)
            a = np.arange(12, dtype=dtype).reshape(3, 4)
            b = np.arange(20, dtype=dtype).reshape(4, 5)
            c = np.arange(30, dtype=dtype).reshape(5, 6)
            if dtype != 'f2':
                assert_equal(np.einsum("ij,jk,kl", a, b, c, optimize=do_opt),
                             a.dot(b).dot(c))
                assert_equal(np.einsum(a, [0, 1], b, [1, 2], c, [2, 3],
                                       optimize=do_opt), a.dot(b).dot(c))

            d = np.arange(18, dtype=dtype).reshape(3, 6)
            np.einsum("ij,jk,kl", a, b, c, out=d,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            tgt = a.astype('f8').dot(b.astype('f8'))
            tgt = tgt.dot(c.astype('f8')).astype(dtype)
            assert_equal(d, tgt)

            d[...] = 0
            np.einsum(a, [0, 1], b, [1, 2], c, [2, 3], out=d,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            tgt = a.astype('f8').dot(b.astype('f8'))
            tgt = tgt.dot(c.astype('f8')).astype(dtype)
            assert_equal(d, tgt)

            # tensordot(a, b)
            if np.dtype(dtype) != np.dtype('f2'):
                a = np.arange(60, dtype=dtype).reshape(3, 4, 5)
                b = np.arange(24, dtype=dtype).reshape(4, 3, 2)
                assert_equal(np.einsum("ijk, jil -> kl", a, b),
                             np.tensordot(a, b, axes=([1, 0], [0, 1])))
                assert_equal(np.einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3]),
                             np.tensordot(a, b, axes=([1, 0], [0, 1])))

                c = np.arange(10, dtype=dtype).reshape(5, 2)
                np.einsum("ijk,jil->kl", a, b, out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c, np.tensordot(a.astype('f8'), b.astype('f8'),
                             axes=([1, 0], [0, 1])).astype(dtype))
                c[...] = 0
                np.einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3], out=c,
                          dtype='f8', casting='unsafe', optimize=do_opt)
                assert_equal(c, np.tensordot(a.astype('f8'), b.astype('f8'),
                             axes=([1, 0], [0, 1])).astype(dtype))

        # logical_and(logical_and(a!=0, b!=0), c!=0)
        neg_val = -2 if dtype.kind != "u" else np.iinfo(dtype).max - 1
        a = np.array([1,   3,   neg_val, 0,  12,  13,   0,   1], dtype=dtype)
        b = np.array([0,   3.5, 0., neg_val,  0,   1,    3,   12], dtype=dtype)
        c = np.array([True, True, False, True, True, False, True, True])

        assert_equal(np.einsum("i,i,i->i", a, b, c,
                     dtype='?', casting='unsafe', optimize=do_opt),
                     np.logical_and(np.logical_and(a != 0, b != 0), c != 0))
        assert_equal(np.einsum(a, [0], b, [0], c, [0], [0],
                     dtype='?', casting='unsafe'),
                     np.logical_and(np.logical_and(a != 0, b != 0), c != 0))

        a = np.arange(9, dtype=dtype)
        assert_equal(np.einsum(",i->", 3, a), 3 * np.sum(a))
        assert_equal(np.einsum(3, [], a, [0], []), 3 * np.sum(a))
        assert_equal(np.einsum("i,->", a, 3), 3 * np.sum(a))
        assert_equal(np.einsum(a, [0], 3, [], []), 3 * np.sum(a))

        # Various stride0, contiguous, and SSE aligned variants
        for n in range(1, 25):
            a = np.arange(n, dtype=dtype)
            if np.dtype(dtype).itemsize > 1:
                assert_equal(np.einsum("...,...", a, a, optimize=do_opt),
                             np.multiply(a, a))
                assert_equal(np.einsum("i,i", a, a, optimize=do_opt), np.dot(a, a))
                assert_equal(np.einsum("i,->i", a, 2, optimize=do_opt), 2 * a)
                assert_equal(np.einsum(",i->i", 2, a, optimize=do_opt), 2 * a)
                assert_equal(np.einsum("i,->", a, 2, optimize=do_opt), 2 * np.sum(a))
                assert_equal(np.einsum(",i->", 2, a, optimize=do_opt), 2 * np.sum(a))

                assert_equal(np.einsum("...,...", a[1:], a[:-1], optimize=do_opt),
                             np.multiply(a[1:], a[:-1]))
                assert_equal(np.einsum("i,i", a[1:], a[:-1], optimize=do_opt),
                             np.dot(a[1:], a[:-1]))
                assert_equal(np.einsum("i,->i", a[1:], 2, optimize=do_opt), 2 * a[1:])
                assert_equal(np.einsum(",i->i", 2, a[1:], optimize=do_opt), 2 * a[1:])
                assert_equal(np.einsum("i,->", a[1:], 2, optimize=do_opt),
                             2 * np.sum(a[1:]))
                assert_equal(np.einsum(",i->", 2, a[1:], optimize=do_opt),
                             2 * np.sum(a[1:]))

        # An object array, summed as the data type
        a = np.arange(9, dtype=object)

        b = np.einsum("i->", a, dtype=dtype, casting='unsafe')
        assert_equal(b, np.sum(a))
        if hasattr(b, "dtype"):
            # Can be a python object when dtype is object
            assert_equal(b.dtype, np.dtype(dtype))

        b = np.einsum(a, [0], [], dtype=dtype, casting='unsafe')
        assert_equal(b, np.sum(a))
        if hasattr(b, "dtype"):
            # Can be a python object when dtype is object
            assert_equal(b.dtype, np.dtype(dtype))

        # A case which was failing (ticket #1885)
        p = np.arange(2) + 1
        q = np.arange(4).reshape(2, 2) + 3
        r = np.arange(4).reshape(2, 2) + 7
        assert_equal(np.einsum('z,mz,zm->', p, q, r), 253)

        # singleton dimensions broadcast (gh-10343)
        p = np.ones((10, 2))
        q = np.ones((1, 2))
        assert_array_equal(np.einsum('ij,ij->j', p, q, optimize=True),
                           np.einsum('ij,ij->j', p, q, optimize=False))
        assert_array_equal(np.einsum('ij,ij->j', p, q, optimize=True),
                           [10.] * 2)

        # a blas-compatible contraction broadcasting case which was failing
        # for optimize=True (ticket #10930)
        x = np.array([2., 3.])
        y = np.array([4.])
        assert_array_equal(np.einsum("i, i", x, y, optimize=False), 20.)
        assert_array_equal(np.einsum("i, i", x, y, optimize=True), 20.)

        # all-ones array was bypassing bug (ticket #10930)
        p = np.ones((1, 5)) / 2
        q = np.ones((5, 5)) / 2
        for optimize in (True, False):
            assert_array_equal(np.einsum("...ij,...jk->...ik", p, p,
                                         optimize=optimize),
                               np.einsum("...ij,...jk->...ik", p, q,
                                         optimize=optimize))
            assert_array_equal(np.einsum("...ij,...jk->...ik", p, q,
                                         optimize=optimize),
                               np.full((1, 5), 1.25))

        # Cases which were failing (gh-10899)
        x = np.eye(2, dtype=dtype)
        y = np.ones(2, dtype=dtype)
        assert_array_equal(np.einsum("ji,i->", x, y, optimize=optimize),
                           [2.])  # contig_contig_outstride0_two
        assert_array_equal(np.einsum("i,ij->", y, x, optimize=optimize),
                           [2.])  # stride0_contig_outstride0_two
        assert_array_equal(np.einsum("ij,i->", x, y, optimize=optimize),
                           [2.])  # contig_stride0_outstride0_two

    def test_einsum_sums_int8(self):
        self.check_einsum_sums('i1')

    def test_einsum_sums_uint8(self):
        self.check_einsum_sums('u1')

    def test_einsum_sums_int16(self):
        self.check_einsum_sums('i2')

    def test_einsum_sums_uint16(self):
        self.check_einsum_sums('u2')

    def test_einsum_sums_int32(self):
        self.check_einsum_sums('i4')
        self.check_einsum_sums('i4', True)

    def test_einsum_sums_uint32(self):
        self.check_einsum_sums('u4')
        self.check_einsum_sums('u4', True)

    def test_einsum_sums_int64(self):
        self.check_einsum_sums('i8')

    def test_einsum_sums_uint64(self):
        self.check_einsum_sums('u8')

    def test_einsum_sums_float16(self):
        self.check_einsum_sums('f2')

    def test_einsum_sums_float32(self):
        self.check_einsum_sums('f4')

    def test_einsum_sums_float64(self):
        self.check_einsum_sums('f8')
        self.check_einsum_sums('f8', True)

    def test_einsum_sums_longdouble(self):
        self.check_einsum_sums(np.longdouble)

    def test_einsum_sums_cfloat64(self):
        self.check_einsum_sums('c8')
        self.check_einsum_sums('c8', True)

    def test_einsum_sums_cfloat128(self):
        self.check_einsum_sums('c16')

    def test_einsum_sums_clongdouble(self):
        self.check_einsum_sums(np.clongdouble)

    def test_einsum_sums_object(self):
        self.check_einsum_sums('object')
        self.check_einsum_sums('object', True)

    def test_einsum_misc(self):
        # This call used to crash because of a bug in
        # PyArray_AssignZero
        a = np.ones((1, 2))
        b = np.ones((2, 2, 1))
        assert_equal(np.einsum('ij...,j...->i...', a, b), [[[2], [2]]])
        assert_equal(np.einsum('ij...,j...->i...', a, b, optimize=True), [[[2], [2]]])

        # Regression test for issue #10369 (test unicode inputs with Python 2)
        assert_equal(np.einsum('ij...,j...->i...', a, b), [[[2], [2]]])
        assert_equal(np.einsum('...i,...i', [1, 2, 3], [2, 3, 4]), 20)
        assert_equal(np.einsum('...i,...i', [1, 2, 3], [2, 3, 4],
                               optimize='greedy'), 20)

        # The iterator had an issue with buffering this reduction
        a = np.ones((5, 12, 4, 2, 3), np.int64)
        b = np.ones((5, 12, 11), np.int64)
        assert_equal(np.einsum('ijklm,ijn,ijn->', a, b, b),
                     np.einsum('ijklm,ijn->', a, b))
        assert_equal(np.einsum('ijklm,ijn,ijn->', a, b, b, optimize=True),
                     np.einsum('ijklm,ijn->', a, b, optimize=True))

        # Issue #2027, was a problem in the contiguous 3-argument
        # inner loop implementation
        a = np.arange(1, 3)
        b = np.arange(1, 5).reshape(2, 2)
        c = np.arange(1, 9).reshape(4, 2)
        assert_equal(np.einsum('x,yx,zx->xzy', a, b, c),
                     [[[1,  3], [3,  9], [5, 15], [7, 21]],
                     [[8, 16], [16, 32], [24, 48], [32, 64]]])
        assert_equal(np.einsum('x,yx,zx->xzy', a, b, c, optimize=True),
                     [[[1,  3], [3,  9], [5, 15], [7, 21]],
                     [[8, 16], [16, 32], [24, 48], [32, 64]]])

        # Ensure explicitly setting out=None does not cause an error
        # see issue gh-15776 and issue gh-15256
        assert_equal(np.einsum('i,j', [1], [2], out=None), [[2]])

    def test_object_loop(self):

        class Mult:
            def __mul__(self, other):
                return 42

        objMult = np.array([Mult()])
        objNULL = np.ndarray(buffer=b'\0' * np.intp(0).itemsize, shape=1, dtype=object)

        with pytest.raises(TypeError):
            np.einsum("i,j", [1], objNULL)
        with pytest.raises(TypeError):
            np.einsum("i,j", objNULL, [1])
        assert np.einsum("i,j", objMult, objMult) == 42

    def test_subscript_range(self):
        # Issue #7741, make sure that all letters of Latin alphabet (both uppercase & lowercase) can be used
        # when creating a subscript from arrays
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        np.einsum(a, [0, 20], b, [20, 2], [0, 2], optimize=False)
        np.einsum(a, [0, 27], b, [27, 2], [0, 2], optimize=False)
        np.einsum(a, [0, 51], b, [51, 2], [0, 2], optimize=False)
        assert_raises(ValueError, lambda: np.einsum(a, [0, 52], b, [52, 2], [0, 2], optimize=False))
        assert_raises(ValueError, lambda: np.einsum(a, [-1, 5], b, [5, 2], [-1, 2], optimize=False))

    def test_einsum_broadcast(self):
        # Issue #2455 change in handling ellipsis
        # remove the 'middle broadcast' error
        # only use the 'RIGHT' iteration in prepare_op_axes
        # adds auto broadcast on left where it belongs
        # broadcast on right has to be explicit
        # We need to test the optimized parsing as well

        A = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        B = np.arange(3)
        ref = np.einsum('ijk,j->ijk', A, B, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum('ij...,j...->ij...', A, B, optimize=opt), ref)
            assert_equal(np.einsum('ij...,...j->ij...', A, B, optimize=opt), ref)
            assert_equal(np.einsum('ij...,j->ij...', A, B, optimize=opt), ref)  # used to raise error

        A = np.arange(12).reshape((4, 3))
        B = np.arange(6).reshape((3, 2))
        ref = np.einsum('ik,kj->ij', A, B, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum('ik...,k...->i...', A, B, optimize=opt), ref)
            assert_equal(np.einsum('ik...,...kj->i...j', A, B, optimize=opt), ref)
            assert_equal(np.einsum('...k,kj', A, B, optimize=opt), ref)  # used to raise error
            assert_equal(np.einsum('ik,k...->i...', A, B, optimize=opt), ref)  # used to raise error

        dims = [2, 3, 4, 5]
        a = np.arange(np.prod(dims)).reshape(dims)
        v = np.arange(dims[2])
        ref = np.einsum('ijkl,k->ijl', a, v, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum('ijkl,k', a, v, optimize=opt), ref)
            assert_equal(np.einsum('...kl,k', a, v, optimize=opt), ref)  # used to raise error
            assert_equal(np.einsum('...kl,k...', a, v, optimize=opt), ref)

        J, K, M = 160, 160, 120
        A = np.arange(J * K * M).reshape(1, 1, 1, J, K, M)
        B = np.arange(J * K * M * 3).reshape(J, K, M, 3)
        ref = np.einsum('...lmn,...lmno->...o', A, B, optimize=False)
        for opt in [True, False]:
            assert_equal(np.einsum('...lmn,lmno->...o', A, B,
                                   optimize=opt), ref)  # used to raise error

    def test_einsum_fixedstridebug(self):
        # Issue #4485 obscure einsum bug
        # This case revealed a bug in nditer where it reported a stride
        # as 'fixed' (0) when it was in fact not fixed during processing
        # (0 or 4). The reason for the bug was that the check for a fixed
        # stride was using the information from the 2D inner loop reuse
        # to restrict the iteration dimensions it had to validate to be
        # the same, but that 2D inner loop reuse logic is only triggered
        # during the buffer copying step, and hence it was invalid to
        # rely on those values. The fix is to check all the dimensions
        # of the stride in question, which in the test case reveals that
        # the stride is not fixed.
        #
        # NOTE: This test is triggered by the fact that the default buffersize,
        #       used by einsum, is 8192, and 3*2731 = 8193, is larger than that
        #       and results in a mismatch between the buffering and the
        #       striding for operand A.
        A = np.arange(2 * 3).reshape(2, 3).astype(np.float32)
        B = np.arange(2 * 3 * 2731).reshape(2, 3, 2731).astype(np.int16)
        es = np.einsum('cl, cpx->lpx',  A,  B)
        tp = np.tensordot(A,  B,  axes=(0,  0))
        assert_equal(es,  tp)
        # The following is the original test case from the bug report,
        # made repeatable by changing random arrays to aranges.
        A = np.arange(3 * 3).reshape(3, 3).astype(np.float64)
        B = np.arange(3 * 3 * 64 * 64).reshape(3, 3, 64, 64).astype(np.float32)
        es = np.einsum('cl, cpxy->lpxy',  A, B)
        tp = np.tensordot(A, B,  axes=(0, 0))
        assert_equal(es, tp)

    def test_einsum_fixed_collapsingbug(self):
        # Issue #5147.
        # The bug only occurred when output argument of einssum was used.
        x = np.random.normal(0, 1, (5, 5, 5, 5))
        y1 = np.zeros((5, 5))
        np.einsum('aabb->ab', x, out=y1)
        idx = np.arange(5)
        y2 = x[idx[:, None], idx[:, None], idx, idx]
        assert_equal(y1, y2)

    def test_einsum_failed_on_p9_and_s390x(self):
        # Issues gh-14692 and gh-12689
        # Bug with signed vs unsigned char errored on power9 and s390x Linux
        tensor = np.random.random_sample((10, 10, 10, 10))
        x = np.einsum('ijij->', tensor)
        y = tensor.trace(axis1=0, axis2=2).trace()
        assert_allclose(x, y)

    def test_einsum_all_contig_non_contig_output(self):
        # Issue gh-5907, tests that the all contiguous special case
        # actually checks the contiguity of the output
        x = np.ones((5, 5))
        out = np.ones(10)[::2]
        correct_base = np.ones(10)
        correct_base[::2] = 5
        # Always worked (inner iteration is done with 0-stride):
        np.einsum('mi,mi,mi->m', x, x, x, out=out)
        assert_array_equal(out.base, correct_base)
        # Example 1:
        out = np.ones(10)[::2]
        np.einsum('im,im,im->m', x, x, x, out=out)
        assert_array_equal(out.base, correct_base)
        # Example 2, buffering causes x to be contiguous but
        # special cases do not catch the operation before:
        out = np.ones((2, 2, 2))[..., 0]
        correct_base = np.ones((2, 2, 2))
        correct_base[..., 0] = 2
        x = np.ones((2, 2), np.float32)
        np.einsum('ij,jk->ik', x, x, out=out)
        assert_array_equal(out.base, correct_base)

    @pytest.mark.parametrize("dtype",
             np.typecodes["AllFloat"] + np.typecodes["AllInteger"])
    def test_different_paths(self, dtype):
        # Test originally added to cover broken float16 path: gh-20305
        # Likely most are covered elsewhere, at least partially.
        dtype = np.dtype(dtype)
        # Simple test, designed to exercise most specialized code paths,
        # note the +0.5 for floats.  This makes sure we use a float value
        # where the results must be exact.
        arr = (np.arange(7) + 0.5).astype(dtype)
        scalar = np.array(2, dtype=dtype)

        # contig -> scalar:
        res = np.einsum('i->', arr)
        assert res == arr.sum()
        # contig, contig -> contig:
        res = np.einsum('i,i->i', arr, arr)
        assert_array_equal(res, arr * arr)
        # noncontig, noncontig -> contig:
        res = np.einsum('i,i->i', arr.repeat(2)[::2], arr.repeat(2)[::2])
        assert_array_equal(res, arr * arr)
        # contig + contig -> scalar
        assert np.einsum('i,i->', arr, arr) == (arr * arr).sum()
        # contig + scalar -> contig (with out)
        out = np.ones(7, dtype=dtype)
        res = np.einsum('i,->i', arr, dtype.type(2), out=out)
        assert_array_equal(res, arr * dtype.type(2))
        # scalar + contig -> contig (with out)
        res = np.einsum(',i->i', scalar, arr)
        assert_array_equal(res, arr * dtype.type(2))
        # scalar + contig -> scalar
        res = np.einsum(',i->', scalar, arr)
        # Use einsum to compare to not have difference due to sum round-offs:
        assert res == np.einsum('i->', scalar * arr)
        # contig + scalar -> scalar
        res = np.einsum('i,->', arr, scalar)
        # Use einsum to compare to not have difference due to sum round-offs:
        assert res == np.einsum('i->', scalar * arr)
        # contig + contig + contig -> scalar
        arr = np.array([0.5, 0.5, 0.25, 4.5, 3.], dtype=dtype)
        res = np.einsum('i,i,i->', arr, arr, arr)
        assert_array_equal(res, (arr * arr * arr).sum())
        # four arrays:
        res = np.einsum('i,i,i,i->', arr, arr, arr, arr)
        assert_array_equal(res, (arr * arr * arr * arr).sum())

    def test_small_boolean_arrays(self):
        # See gh-5946.
        # Use array of True embedded in False.
        a = np.zeros((16, 1, 1), dtype=np.bool)[:2]
        a[...] = True
        out = np.zeros((16, 1, 1), dtype=np.bool)[:2]
        tgt = np.ones((2, 1, 1), dtype=np.bool)
        res = np.einsum('...ij,...jk->...ik', a, a, out=out)
        assert_equal(res, tgt)

    def test_out_is_res(self):
        a = np.arange(9).reshape(3, 3)
        res = np.einsum('...ij,...jk->...ik', a, a, out=a)
        assert res is a

    def optimize_compare(self, subscripts, operands=None):
        # Tests all paths of the optimization function against
        # conventional einsum
        if operands is None:
            args = [subscripts]
            terms = subscripts.split('->')[0].split(',')
            for term in terms:
                dims = [global_size_dict[x] for x in term]
                args.append(np.random.rand(*dims))
        else:
            args = [subscripts] + operands

        noopt = np.einsum(*args, optimize=False)
        opt = np.einsum(*args, optimize='greedy')
        assert_almost_equal(opt, noopt)
        opt = np.einsum(*args, optimize='optimal')
        assert_almost_equal(opt, noopt)

    def test_hadamard_like_products(self):
        # Hadamard outer products
        self.optimize_compare('a,ab,abc->abc')
        self.optimize_compare('a,b,ab->ab')

    def test_index_transformations(self):
        # Simple index transformation cases
        self.optimize_compare('ea,fb,gc,hd,abcd->efgh')
        self.optimize_compare('ea,fb,abcd,gc,hd->efgh')
        self.optimize_compare('abcd,ea,fb,gc,hd->efgh')

    def test_complex(self):
        # Long test cases
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('cd,bdhe,aidb,hgca,gc,hgibcd,hgac')
        self.optimize_compare('abhe,hidj,jgba,hiab,gab')
        self.optimize_compare('bde,cdh,agdb,hica,ibd,hgicd,hiac')
        self.optimize_compare('chd,bde,agbc,hiad,hgc,hgi,hiad')
        self.optimize_compare('chd,bde,agbc,hiad,bdi,cgh,agdb')
        self.optimize_compare('bdhe,acad,hiab,agac,hibd')

    def test_collapse(self):
        # Inner products
        self.optimize_compare('ab,ab,c->')
        self.optimize_compare('ab,ab,c->c')
        self.optimize_compare('ab,ab,cd,cd->')
        self.optimize_compare('ab,ab,cd,cd->ac')
        self.optimize_compare('ab,ab,cd,cd->cd')
        self.optimize_compare('ab,ab,cd,cd,ef,ef->')

    def test_expand(self):
        # Outer products
        self.optimize_compare('ab,cd,ef->abcdef')
        self.optimize_compare('ab,cd,ef->acdf')
        self.optimize_compare('ab,cd,de->abcde')
        self.optimize_compare('ab,cd,de->be')
        self.optimize_compare('ab,bcd,cd->abcd')
        self.optimize_compare('ab,bcd,cd->abd')

    def test_edge_cases(self):
        # Difficult edge cases for optimization
        self.optimize_compare('eb,cb,fb->cef')
        self.optimize_compare('dd,fb,be,cdb->cef')
        self.optimize_compare('bca,cdb,dbf,afc->')
        self.optimize_compare('dcc,fce,ea,dbf->ab')
        self.optimize_compare('fdf,cdd,ccd,afe->ae')
        self.optimize_compare('abcd,ad')
        self.optimize_compare('ed,fcd,ff,bcf->be')
        self.optimize_compare('baa,dcf,af,cde->be')
        self.optimize_compare('bd,db,eac->ace')
        self.optimize_compare('fff,fae,bef,def->abd')
        self.optimize_compare('efc,dbc,acf,fd->abe')
        self.optimize_compare('ba,ac,da->bcd')

    def test_inner_product(self):
        # Inner products
        self.optimize_compare('ab,ab')
        self.optimize_compare('ab,ba')
        self.optimize_compare('abc,abc')
        self.optimize_compare('abc,bac')
        self.optimize_compare('abc,cba')

    def test_random_cases(self):
        # Randomly built test cases
        self.optimize_compare('aab,fa,df,ecc->bde')
        self.optimize_compare('ecb,fef,bad,ed->ac')
        self.optimize_compare('bcf,bbb,fbf,fc->')
        self.optimize_compare('bb,ff,be->e')
        self.optimize_compare('bcb,bb,fc,fff->')
        self.optimize_compare('fbb,dfd,fc,fc->')
        self.optimize_compare('afd,ba,cc,dc->bf')
        self.optimize_compare('adb,bc,fa,cfc->d')
        self.optimize_compare('bbd,bda,fc,db->acf')
        self.optimize_compare('dba,ead,cad->bce')
        self.optimize_compare('aef,fbc,dca->bde')

    def test_combined_views_mapping(self):
        # gh-10792
        a = np.arange(9).reshape(1, 1, 3, 1, 3)
        b = np.einsum('bbcdc->d', a)
        assert_equal(b, [12])

    def test_broadcasting_dot_cases(self):
        # Ensures broadcasting cases are not mistaken for GEMM

        a = np.random.rand(1, 5, 4)
        b = np.random.rand(4, 6)
        c = np.random.rand(5, 6)
        d = np.random.rand(10)

        self.optimize_compare('ijk,kl,jl', operands=[a, b, c])
        self.optimize_compare('ijk,kl,jl,i->i', operands=[a, b, c, d])

        e = np.random.rand(1, 1, 5, 4)
        f = np.random.rand(7, 7)
        self.optimize_compare('abjk,kl,jl', operands=[e, b, c])
        self.optimize_compare('abjk,kl,jl,ab->ab', operands=[e, b, c, f])

        # Edge case found in gh-11308
        g = np.arange(64).reshape(2, 4, 8)
        self.optimize_compare('obk,ijk->ioj', operands=[g, g])

    def test_output_order(self):
        # Ensure output order is respected for optimize cases, the below
        # contraction should yield a reshaped tensor view
        # gh-16415

        a = np.ones((2, 3, 5), order='F')
        b = np.ones((4, 3), order='F')

        for opt in [True, False]:
            tmp = np.einsum('...ft,mf->...mt', a, b, order='a', optimize=opt)
            assert_(tmp.flags.f_contiguous)

            tmp = np.einsum('...ft,mf->...mt', a, b, order='f', optimize=opt)
            assert_(tmp.flags.f_contiguous)

            tmp = np.einsum('...ft,mf->...mt', a, b, order='c', optimize=opt)
            assert_(tmp.flags.c_contiguous)

            tmp = np.einsum('...ft,mf->...mt', a, b, order='k', optimize=opt)
            assert_(tmp.flags.c_contiguous is False)
            assert_(tmp.flags.f_contiguous is False)

            tmp = np.einsum('...ft,mf->...mt', a, b, optimize=opt)
            assert_(tmp.flags.c_contiguous is False)
            assert_(tmp.flags.f_contiguous is False)

        c = np.ones((4, 3), order='C')
        for opt in [True, False]:
            tmp = np.einsum('...ft,mf->...mt', a, c, order='a', optimize=opt)
            assert_(tmp.flags.c_contiguous)

        d = np.ones((2, 3, 5), order='C')
        for opt in [True, False]:
            tmp = np.einsum('...ft,mf->...mt', d, c, order='a', optimize=opt)
            assert_(tmp.flags.c_contiguous)

class TestEinsumPath:
    def build_operands(self, string, size_dict=global_size_dict):

        # Builds views based off initial operands
        operands = [string]
        terms = string.split('->')[0].split(',')
        for term in terms:
            dims = [size_dict[x] for x in term]
            operands.append(np.random.rand(*dims))

        return operands

    def assert_path_equal(self, comp, benchmark):
        # Checks if list of tuples are equivalent
        ret = (len(comp) == len(benchmark))
        assert_(ret)
        for pos in range(len(comp) - 1):
            ret &= isinstance(comp[pos + 1], tuple)
            ret &= (comp[pos + 1] == benchmark[pos + 1])
        assert_(ret)

    def test_memory_contraints(self):
        # Ensure memory constraints are satisfied

        outer_test = self.build_operands('a,b,c->abc')

        path, path_str = np.einsum_path(*outer_test, optimize=('greedy', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])

        path, path_str = np.einsum_path(*outer_test, optimize=('optimal', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])

        long_test = self.build_operands('acdf,jbje,gihb,hfac')
        path, path_str = np.einsum_path(*long_test, optimize=('greedy', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

        path, path_str = np.einsum_path(*long_test, optimize=('optimal', 0))
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

    def test_long_paths(self):
        # Long complex cases

        # Long test 1
        long_test1 = self.build_operands('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        path, path_str = np.einsum_path(*long_test1, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])

        path, path_str = np.einsum_path(*long_test1, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])

        # Long test 2
        long_test2 = self.build_operands('chd,bde,agbc,hiad,bdi,cgh,agdb')
        path, path_str = np.einsum_path(*long_test2, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 4), (0, 3), (3, 4), (1, 3), (1, 2), (0, 1)])

        path, path_str = np.einsum_path(*long_test2, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path',
                                      (0, 5), (1, 4), (3, 4), (1, 3), (1, 2), (0, 1)])

    def test_edge_paths(self):
        # Difficult edge cases

        # Edge test1
        edge_test1 = self.build_operands('eb,cb,fb->cef')
        path, path_str = np.einsum_path(*edge_test1, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])

        path, path_str = np.einsum_path(*edge_test1, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])

        # Edge test2
        edge_test2 = self.build_operands('dd,fb,be,cdb->cef')
        path, path_str = np.einsum_path(*edge_test2, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])

        path, path_str = np.einsum_path(*edge_test2, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])

        # Edge test3
        edge_test3 = self.build_operands('bca,cdb,dbf,afc->')
        path, path_str = np.einsum_path(*edge_test3, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        path, path_str = np.einsum_path(*edge_test3, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        # Edge test4
        edge_test4 = self.build_operands('dcc,fce,ea,dbf->ab')
        path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])

        path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        # Edge test5
        edge_test4 = self.build_operands('a,ac,ab,ad,cd,bd,bc->',
                                         size_dict={"a": 20, "b": 20, "c": 20, "d": 20})
        path, path_str = np.einsum_path(*edge_test4, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])

        path, path_str = np.einsum_path(*edge_test4, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])

    def test_path_type_input(self):
        # Test explicit path handling
        path_test = self.build_operands('dcc,fce,ea,dbf->ab')

        path, path_str = np.einsum_path(*path_test, optimize=False)
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

        path, path_str = np.einsum_path(*path_test, optimize=True)
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 1), (0, 1)])

        exp_path = ['einsum_path', (0, 2), (0, 2), (0, 1)]
        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        self.assert_path_equal(path, exp_path)

        # Double check einsum works on the input path
        noopt = np.einsum(*path_test, optimize=False)
        opt = np.einsum(*path_test, optimize=exp_path)
        assert_almost_equal(noopt, opt)

    def test_path_type_input_internal_trace(self):
        # gh-20962
        path_test = self.build_operands('cab,cdd->ab')
        exp_path = ['einsum_path', (1,), (0, 1)]

        path, path_str = np.einsum_path(*path_test, optimize=exp_path)
        self.assert_path_equal(path, exp_path)

        # Double check einsum works on the input path
        noopt = np.einsum(*path_test, optimize=False)
        opt = np.einsum(*path_test, optimize=exp_path)
        assert_almost_equal(noopt, opt)

    def test_path_type_input_invalid(self):
        path_test = self.build_operands('ab,bc,cd,de->ae')
        exp_path = ['einsum_path', (2, 3), (0, 1)]
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        assert_raises(
            RuntimeError, np.einsum_path, *path_test, optimize=exp_path)

        path_test = self.build_operands('a,a,a->a')
        exp_path = ['einsum_path', (1,), (0, 1)]
        assert_raises(RuntimeError, np.einsum, *path_test, optimize=exp_path)
        assert_raises(
            RuntimeError, np.einsum_path, *path_test, optimize=exp_path)

    def test_spaces(self):
        # gh-10794
        arr = np.array([[1]])
        for sp in itertools.product(['', ' '], repeat=4):
            # no error for any spacing
            np.einsum('{}...a{}->{}...a{}'.format(*sp), arr)

def test_overlap():
    a = np.arange(9, dtype=int).reshape(3, 3)
    b = np.arange(9, dtype=int).reshape(3, 3)
    d = np.dot(a, b)
    # sanity check
    c = np.einsum('ij,jk->ik', a, b)
    assert_equal(c, d)
    # gh-10080, out overlaps one of the operands
    c = np.einsum('ij,jk->ik', a, b, out=b)
    assert_equal(c, d)

def test_einsum_chunking_precision():
    """Most einsum operations are reductions and until NumPy 2.3 reductions
    never (or almost never?) used the `GROWINNER` mechanism to increase the
    inner loop size when no buffers are needed.
    Because einsum reductions work roughly:

        def inner(*inputs, out):
            accumulate = 0
            for vals in zip(*inputs):
                accumulate += prod(vals)
            out[0] += accumulate

    Calling the inner-loop more often actually improves accuracy slightly
    (same effect as pairwise summation but much less).
    Without adding pairwise summation to the inner-loop it seems best to just
    not use GROWINNER, a quick tests suggest that is maybe 1% slowdown for
    the simplest `einsum("i,i->i", x, x)` case.

    (It is not clear that we should guarantee precision to this extend.)
    """
    num = 1_000_000
    value = 1. + np.finfo(np.float64).eps * 8196
    res = np.einsum("i->", np.broadcast_to(np.array(value), num)) / num

    # At with GROWINNER 11 decimals succeed (larger will be less)
    assert_almost_equal(res, value, decimal=15)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\tests\test_fancysets.py ===

from sympy.core.expr import unchanged
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range, normalize_theta_set,
                                  ComplexRegion)
from sympy.sets.sets import (FiniteSet, Interval, Union, imageset,
                             Intersection, ProductSet, SetKind)
from sympy.sets.conditionset import ConditionSet
from sympy.simplify.simplify import simplify
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import And
from sympy.matrices.dense import eye
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, t, z
from sympy.core.mod import Mod

import itertools


def test_naturals():
    N = S.Naturals
    assert 5 in N
    assert -5 not in N
    assert 5.5 not in N
    ni = iter(N)
    a, b, c, d = next(ni), next(ni), next(ni), next(ni)
    assert (a, b, c, d) == (1, 2, 3, 4)
    assert isinstance(a, Basic)

    assert N.intersect(Interval(-5, 5)) == Range(1, 6)
    assert N.intersect(Interval(-5, 5, True, True)) == Range(1, 5)

    assert N.boundary == N
    assert N.is_open == False
    assert N.is_closed == True

    assert N.inf == 1
    assert N.sup is oo
    assert not N.contains(oo)
    for s in (S.Naturals0, S.Naturals):
        assert s.intersection(S.Reals) is s
        assert s.is_subset(S.Reals)

    assert N.as_relational(x) == And(Eq(floor(x), x), x >= 1, x < oo)


def test_naturals0():
    N = S.Naturals0
    assert 0 in N
    assert -1 not in N
    assert next(iter(N)) == 0
    assert not N.contains(oo)
    assert N.contains(sin(x)) == Contains(sin(x), N)


def test_integers():
    Z = S.Integers
    assert 5 in Z
    assert -5 in Z
    assert 5.5 not in Z
    assert not Z.contains(oo)
    assert not Z.contains(-oo)

    zi = iter(Z)
    a, b, c, d = next(zi), next(zi), next(zi), next(zi)
    assert (a, b, c, d) == (0, 1, -1, 2)
    assert isinstance(a, Basic)

    assert Z.intersect(Interval(-5, 5)) == Range(-5, 6)
    assert Z.intersect(Interval(-5, 5, True, True)) == Range(-4, 5)
    assert Z.intersect(Interval(5, S.Infinity)) == Range(5, S.Infinity)
    assert Z.intersect(Interval.Lopen(5, S.Infinity)) == Range(6, S.Infinity)

    assert Z.inf is -oo
    assert Z.sup is oo

    assert Z.boundary == Z
    assert Z.is_open == False
    assert Z.is_closed == True

    assert Z.as_relational(x) == And(Eq(floor(x), x), -oo < x, x < oo)


def test_ImageSet():
    raises(ValueError, lambda: ImageSet(x, S.Integers))
    assert ImageSet(Lambda(x, 1), S.Integers) == FiniteSet(1)
    assert ImageSet(Lambda(x, y), S.Integers) == {y}
    assert ImageSet(Lambda(x, 1), S.EmptySet) == S.EmptySet
    empty = Intersection(FiniteSet(log(2)/pi), S.Integers)
    assert unchanged(ImageSet, Lambda(x, 1), empty)  # issue #17471
    squares = ImageSet(Lambda(x, x**2), S.Naturals)
    assert 4 in squares
    assert 5 not in squares
    assert FiniteSet(*range(10)).intersect(squares) == FiniteSet(1, 4, 9)

    assert 16 not in squares.intersect(Interval(0, 10))

    si = iter(squares)
    a, b, c, d = next(si), next(si), next(si), next(si)
    assert (a, b, c, d) == (1, 4, 9, 16)

    harmonics = ImageSet(Lambda(x, 1/x), S.Naturals)
    assert Rational(1, 5) in harmonics
    assert Rational(.25) in harmonics
    assert harmonics.contains(.25) == Contains(
        0.25, ImageSet(Lambda(x, 1/x), S.Naturals), evaluate=False)
    assert Rational(.3) not in harmonics
    assert (1, 2) not in harmonics

    assert harmonics.is_iterable

    assert imageset(x, -x, Interval(0, 1)) == Interval(-1, 0)

    assert ImageSet(Lambda(x, x**2), Interval(0, 2)).doit() == Interval(0, 4)
    assert ImageSet(Lambda((x, y), 2*x), {4}, {3}).doit() == FiniteSet(8)
    assert (ImageSet(Lambda((x, y), x+y), {1, 2, 3}, {10, 20, 30}).doit() ==
                FiniteSet(11, 12, 13, 21, 22, 23, 31, 32, 33))

    c = Interval(1, 3) * Interval(1, 3)
    assert Tuple(2, 6) in ImageSet(Lambda(((x, y),), (x, 2*y)), c)
    assert Tuple(2, S.Half) in ImageSet(Lambda(((x, y),), (x, 1/y)), c)
    assert Tuple(2, -2) not in ImageSet(Lambda(((x, y),), (x, y**2)), c)
    assert Tuple(2, -2) in ImageSet(Lambda(((x, y),), (x, -2)), c)
    c3 = ProductSet(Interval(3, 7), Interval(8, 11), Interval(5, 9))
    assert Tuple(8, 3, 9) in ImageSet(Lambda(((t, y, x),), (y, t, x)), c3)
    assert Tuple(Rational(1, 8), 3, 9) in ImageSet(Lambda(((t, y, x),), (1/y, t, x)), c3)
    assert 2/pi not in ImageSet(Lambda(((x, y),), 2/x), c)
    assert 2/S(100) not in ImageSet(Lambda(((x, y),), 2/x), c)
    assert Rational(2, 3) in ImageSet(Lambda(((x, y),), 2/x), c)

    S1 = imageset(lambda x, y: x + y, S.Integers, S.Naturals)
    assert S1.base_pset == ProductSet(S.Integers, S.Naturals)
    assert S1.base_sets == (S.Integers, S.Naturals)

    # Passing a set instead of a FiniteSet shouldn't raise
    assert unchanged(ImageSet, Lambda(x, x**2), {1, 2, 3})

    S2 = ImageSet(Lambda(((x, y),), x+y), {(1, 2), (3, 4)})
    assert 3 in S2.doit()
    # FIXME: This doesn't yet work:
    #assert 3 in S2
    assert S2._contains(3) is None

    raises(TypeError, lambda: ImageSet(Lambda(x, x**2), 1))


def test_image_is_ImageSet():
    assert isinstance(imageset(x, sqrt(sin(x)), Range(5)), ImageSet)


def test_halfcircle():
    r, th = symbols('r, theta', real=True)
    L = Lambda(((r, th),), (r*cos(th), r*sin(th)))
    halfcircle = ImageSet(L, Interval(0, 1)*Interval(0, pi))

    assert (1, 0) in halfcircle
    assert (0, -1) not in halfcircle
    assert (0, 0) in halfcircle
    assert halfcircle._contains((r, 0)) is None
    assert not halfcircle.is_iterable


@XFAIL
def test_halfcircle_fail():
    r, th = symbols('r, theta', real=True)
    L = Lambda(((r, th),), (r*cos(th), r*sin(th)))
    halfcircle = ImageSet(L, Interval(0, 1)*Interval(0, pi))
    assert (r, 2*pi) not in halfcircle


def test_ImageSet_iterator_not_injective():
    L = Lambda(x, x - x % 2)  # produces 0, 2, 2, 4, 4, 6, 6, ...
    evens = ImageSet(L, S.Naturals)
    i = iter(evens)
    # No repeats here
    assert (next(i), next(i), next(i), next(i)) == (0, 2, 4, 6)


def test_inf_Range_len():
    raises(ValueError, lambda: len(Range(0, oo, 2)))
    assert Range(0, oo, 2).size is S.Infinity
    assert Range(0, -oo, -2).size is S.Infinity
    assert Range(oo, 0, -2).size is S.Infinity
    assert Range(-oo, 0, 2).size is S.Infinity


def test_Range_set():
    empty = Range(0)

    assert Range(5) == Range(0, 5) == Range(0, 5, 1)

    r = Range(10, 20, 2)
    assert 12 in r
    assert 8 not in r
    assert 11 not in r
    assert 30 not in r

    assert list(Range(0, 5)) == list(range(5))
    assert list(Range(5, 0, -1)) == list(range(5, 0, -1))


    assert Range(5, 15).sup == 14
    assert Range(5, 15).inf == 5
    assert Range(15, 5, -1).sup == 15
    assert Range(15, 5, -1).inf == 6
    assert Range(10, 67, 10).sup == 60
    assert Range(60, 7, -10).inf == 10

    assert len(Range(10, 38, 10)) == 3

    assert Range(0, 0, 5) == empty
    assert Range(oo, oo, 1) == empty
    assert Range(oo, 1, 1) == empty
    assert Range(-oo, 1, -1) == empty
    assert Range(1, oo, -1) == empty
    assert Range(1, -oo, 1) == empty
    assert Range(1, -4, oo) == empty
    ip = symbols('ip', positive=True)
    assert Range(0, ip, -1) == empty
    assert Range(0, -ip, 1) == empty
    assert Range(1, -4, -oo) == Range(1, 2)
    assert Range(1, 4, oo) == Range(1, 2)
    assert Range(-oo, oo).size == oo
    assert Range(oo, -oo, -1).size == oo
    raises(ValueError, lambda: Range(-oo, oo, 2))
    raises(ValueError, lambda: Range(x, pi, y))
    raises(ValueError, lambda: Range(x, y, 0))

    assert 5 in Range(0, oo, 5)
    assert -5 in Range(-oo, 0, 5)
    assert oo not in Range(0, oo)
    ni = symbols('ni', integer=False)
    assert ni not in Range(oo)
    u = symbols('u', integer=None)
    assert Range(oo).contains(u) is not False
    inf = symbols('inf', infinite=True)
    assert inf not in Range(-oo, oo)
    raises(ValueError, lambda: Range(0, oo, 2)[-1])
    raises(ValueError, lambda: Range(0, -oo, -2)[-1])
    assert Range(-oo, 1, 1)[-1] is S.Zero
    assert Range(oo, 1, -1)[-1] == 2
    assert inf not in Range(oo)
    assert Range(1, 10, 1)[-1] == 9
    assert all(i.is_Integer for i in Range(0, -1, 1))
    it = iter(Range(-oo, 0, 2))
    raises(TypeError, lambda: next(it))

    assert empty.intersect(S.Integers) == empty
    assert Range(-1, 10, 1).intersect(S.Complexes) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Reals) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Rationals) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Integers) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Naturals) == Range(1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Naturals0) == Range(0, 10, 1)

    # test slicing
    assert Range(1, 10, 1)[5] == 6
    assert Range(1, 12, 2)[5] == 11
    assert Range(1, 10, 1)[-1] == 9
    assert Range(1, 10, 3)[-1] == 7
    raises(ValueError, lambda: Range(oo,0,-1)[1:3:0])
    raises(ValueError, lambda: Range(oo,0,-1)[:1])
    raises(ValueError, lambda: Range(1, oo)[-2])
    raises(ValueError, lambda: Range(-oo, 1)[2])
    raises(IndexError, lambda: Range(10)[-20])
    raises(IndexError, lambda: Range(10)[20])
    raises(ValueError, lambda: Range(2, -oo, -2)[2:2:0])
    assert Range(2, -oo, -2)[2:2:2] == empty
    assert Range(2, -oo, -2)[:2:2] == Range(2, -2, -4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[:2:2])
    assert Range(-oo, 4, 2)[::-2] == Range(2, -oo, -4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[::2])
    assert Range(oo, 2, -2)[::] == Range(oo, 2, -2)
    assert Range(-oo, 4, 2)[:-2:-2] == Range(2, 0, -4)
    assert Range(-oo, 4, 2)[:-2:2] == Range(-oo, 0, 4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[:0:-2])
    raises(ValueError, lambda: Range(-oo, 4, 2)[:2:-2])
    assert Range(-oo, 4, 2)[-2::-2] == Range(0, -oo, -4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[-2:0:-2])
    raises(ValueError, lambda: Range(-oo, 4, 2)[0::2])
    assert Range(oo, 2, -2)[0::] == Range(oo, 2, -2)
    raises(ValueError, lambda: Range(-oo, 4, 2)[0:-2:2])
    assert Range(oo, 2, -2)[0:-2:] == Range(oo, 6, -2)
    raises(ValueError, lambda: Range(oo, 2, -2)[0:2:])
    raises(ValueError, lambda: Range(-oo, 4, 2)[2::-1])
    assert Range(-oo, 4, 2)[-2::2] == Range(0, 4, 4)
    assert Range(oo, 0, -2)[-10:0:2] == empty
    raises(ValueError, lambda: Range(oo, 0, -2)[0])
    raises(ValueError, lambda: Range(oo, 0, -2)[-10:10:2])
    raises(ValueError, lambda: Range(oo, 0, -2)[0::-2])
    assert Range(oo, 0, -2)[0:-4:-2] == empty
    assert Range(oo, 0, -2)[:0:2] == empty
    raises(ValueError, lambda: Range(oo, 0, -2)[:1:-1])

    # test empty Range
    assert Range(x, x, y) == empty
    assert empty.reversed == empty
    assert 0 not in empty
    assert list(empty) == []
    assert len(empty) == 0
    assert empty.size is S.Zero
    assert empty.intersect(FiniteSet(0)) is S.EmptySet
    assert bool(empty) is False
    raises(IndexError, lambda: empty[0])
    assert empty[:0] == empty
    raises(NotImplementedError, lambda: empty.inf)
    raises(NotImplementedError, lambda: empty.sup)
    assert empty.as_relational(x) is S.false

    AB = [None] + list(range(12))
    for R in [
            Range(1, 10),
            Range(1, 10, 2),
        ]:
        r = list(R)
        for a, b, c in itertools.product(AB, AB, [-3, -1, None, 1, 3]):
            for reverse in range(2):
                r = list(reversed(r))
                R = R.reversed
                result = list(R[a:b:c])
                ans = r[a:b:c]
                txt = ('\n%s[%s:%s:%s] = %s -> %s' % (
                R, a, b, c, result, ans))
                check = ans == result
                assert check, txt

    assert Range(1, 10, 1).boundary == Range(1, 10, 1)

    for r in (Range(1, 10, 2), Range(1, oo, 2)):
        rev = r.reversed
        assert r.inf == rev.inf and r.sup == rev.sup
        assert r.step == -rev.step

    builtin_range = range

    raises(TypeError, lambda: Range(builtin_range(1)))
    assert S(builtin_range(10)) == Range(10)
    assert S(builtin_range(1000000000000)) == Range(1000000000000)

    # test Range.as_relational
    assert Range(1, 4).as_relational(x) == (x >= 1) & (x <= 3)  & Eq(Mod(x, 1), 0)
    assert Range(oo, 1, -2).as_relational(x) == (x >= 3) & (x < oo)  & Eq(Mod(x + 1, -2), 0)


def test_Range_symbolic():
    # symbolic Range
    xr = Range(x, x + 4, 5)
    sr = Range(x, y, t)
    i = Symbol('i', integer=True)
    ip = Symbol('i', integer=True, positive=True)
    ipr = Range(ip)
    inr = Range(0, -ip, -1)
    ir = Range(i, i + 19, 2)
    ir2 = Range(i, i*8, 3*i)
    i = Symbol('i', integer=True)
    inf = symbols('inf', infinite=True)
    raises(ValueError, lambda: Range(inf))
    raises(ValueError, lambda: Range(inf, 0, -1))
    raises(ValueError, lambda: Range(inf, inf, 1))
    raises(ValueError, lambda: Range(1, 1, inf))
    # args
    assert xr.args == (x, x + 5, 5)
    assert sr.args == (x, y, t)
    assert ir.args == (i, i + 20, 2)
    assert ir2.args == (i, 10*i, 3*i)
    # reversed
    raises(ValueError, lambda: xr.reversed)
    raises(ValueError, lambda: sr.reversed)
    assert ipr.reversed.args == (ip - 1, -1, -1)
    assert inr.reversed.args == (-ip + 1, 1, 1)
    assert ir.reversed.args == (i + 18, i - 2, -2)
    assert ir2.reversed.args == (7*i, -2*i, -3*i)
    # contains
    assert inf not in sr
    assert inf not in ir
    assert 0 in ipr
    assert 0 in inr
    raises(TypeError, lambda: 1 in ipr)
    raises(TypeError, lambda: -1 in inr)
    assert .1 not in sr
    assert .1 not in ir
    assert i + 1 not in ir
    assert i + 2 in ir
    raises(TypeError, lambda: x in xr)  # XXX is this what contains is supposed to do?
    raises(TypeError, lambda: 1 in sr)  # XXX is this what contains is supposed to do?
    # iter
    raises(ValueError, lambda: next(iter(xr)))
    raises(ValueError, lambda: next(iter(sr)))
    assert next(iter(ir)) == i
    assert next(iter(ir2)) == i
    assert sr.intersect(S.Integers) == sr
    assert sr.intersect(FiniteSet(x)) == Intersection({x}, sr)
    raises(ValueError, lambda: sr[:2])
    raises(ValueError, lambda: xr[0])
    raises(ValueError, lambda: sr[0])
    # len
    assert len(ir) == ir.size == 10
    assert len(ir2) == ir2.size == 3
    raises(ValueError, lambda: len(xr))
    raises(ValueError, lambda: xr.size)
    raises(ValueError, lambda: len(sr))
    raises(ValueError, lambda: sr.size)
    # bool
    assert bool(Range(0)) == False
    assert bool(xr)
    assert bool(ir)
    assert bool(ipr)
    assert bool(inr)
    raises(ValueError, lambda: bool(sr))
    raises(ValueError, lambda: bool(ir2))
    # inf
    raises(ValueError, lambda: xr.inf)
    raises(ValueError, lambda: sr.inf)
    assert ipr.inf == 0
    assert inr.inf == -ip + 1
    assert ir.inf == i
    raises(ValueError, lambda: ir2.inf)
    # sup
    raises(ValueError, lambda: xr.sup)
    raises(ValueError, lambda: sr.sup)
    assert ipr.sup == ip - 1
    assert inr.sup == 0
    assert ir.inf == i
    raises(ValueError, lambda: ir2.sup)
    # getitem
    raises(ValueError, lambda: xr[0])
    raises(ValueError, lambda: sr[0])
    raises(ValueError, lambda: sr[-1])
    raises(ValueError, lambda: sr[:2])
    assert ir[:2] == Range(i, i + 4, 2)
    assert ir[0] == i
    assert ir[-2] == i + 16
    assert ir[-1] == i + 18
    assert ir2[:2] == Range(i, 7*i, 3*i)
    assert ir2[0] == i
    assert ir2[-2] == 4*i
    assert ir2[-1] == 7*i
    raises(ValueError, lambda: Range(i)[-1])
    assert ipr[0] == ipr.inf == 0
    assert ipr[-1] == ipr.sup == ip - 1
    assert inr[0] == inr.sup == 0
    assert inr[-1] == inr.inf == -ip + 1
    raises(ValueError, lambda: ipr[-2])
    assert ir.inf == i
    assert ir.sup == i + 18
    raises(ValueError, lambda: Range(i).inf)
    # as_relational
    assert ir.as_relational(x) == ((x >= i) & (x <= i + 18) &
        Eq(Mod(-i + x, 2), 0))
    assert ir2.as_relational(x) == Eq(
        Mod(-i + x, 3*i), 0) & (((x >= i) & (x <= 7*i) & (3*i >= 1)) |
        ((x <= i) & (x >= 7*i) & (3*i <= -1)))
    assert Range(i, i + 1).as_relational(x) == Eq(x, i)
    assert sr.as_relational(z) == Eq(
        Mod(t, 1), 0) & Eq(Mod(x, 1), 0) & Eq(Mod(-x + z, t), 0
        ) & (((z >= x) & (z <= -t + y) & (t >= 1)) |
        ((z <= x) & (z >= -t + y) & (t <= -1)))
    assert xr.as_relational(z) == Eq(z, x) & Eq(Mod(x, 1), 0)
    # symbols can clash if user wants (but it must be integer)
    assert xr.as_relational(x) == Eq(Mod(x, 1), 0)
    # contains() for symbolic values (issue #18146)
    e = Symbol('e', integer=True, even=True)
    o = Symbol('o', integer=True, odd=True)
    assert Range(5).contains(i) == And(i >= 0, i <= 4)
    assert Range(1).contains(i) == Eq(i, 0)
    assert Range(-oo, 5, 1).contains(i) == (i <= 4)
    assert Range(-oo, oo).contains(i) == True
    assert Range(0, 8, 2).contains(i) == Contains(i, Range(0, 8, 2))
    assert Range(0, 8, 2).contains(e) == And(e >= 0, e <= 6)
    assert Range(0, 8, 2).contains(2*i) == And(2*i >= 0, 2*i <= 6)
    assert Range(0, 8, 2).contains(o) == False
    assert Range(1, 9, 2).contains(e) == False
    assert Range(1, 9, 2).contains(o) == And(o >= 1, o <= 7)
    assert Range(8, 0, -2).contains(o) == False
    assert Range(9, 1, -2).contains(o) == And(o >= 3, o <= 9)
    assert Range(-oo, 8, 2).contains(i) == Contains(i, Range(-oo, 8, 2))


def test_range_range_intersection():
    for a, b, r in [
            (Range(0), Range(1), S.EmptySet),
            (Range(3), Range(4, oo), S.EmptySet),
            (Range(3), Range(-3, -1), S.EmptySet),
            (Range(1, 3), Range(0, 3), Range(1, 3)),
            (Range(1, 3), Range(1, 4), Range(1, 3)),
            (Range(1, oo, 2), Range(2, oo, 2), S.EmptySet),
            (Range(0, oo, 2), Range(oo), Range(0, oo, 2)),
            (Range(0, oo, 2), Range(100), Range(0, 100, 2)),
            (Range(2, oo, 2), Range(oo), Range(2, oo, 2)),
            (Range(0, oo, 2), Range(5, 6), S.EmptySet),
            (Range(2, 80, 1), Range(55, 71, 4), Range(55, 71, 4)),
            (Range(0, 6, 3), Range(-oo, 5, 3), S.EmptySet),
            (Range(0, oo, 2), Range(5, oo, 3), Range(8, oo, 6)),
            (Range(4, 6, 2), Range(2, 16, 7), S.EmptySet),]:
        assert a.intersect(b) == r
        assert a.intersect(b.reversed) == r
        assert a.reversed.intersect(b) == r
        assert a.reversed.intersect(b.reversed) == r
        a, b = b, a
        assert a.intersect(b) == r
        assert a.intersect(b.reversed) == r
        assert a.reversed.intersect(b) == r
        assert a.reversed.intersect(b.reversed) == r


def test_range_interval_intersection():
    p = symbols('p', positive=True)
    assert isinstance(Range(3).intersect(Interval(p, p + 2)), Intersection)
    assert Range(4).intersect(Interval(0, 3)) == Range(4)
    assert Range(4).intersect(Interval(-oo, oo)) == Range(4)
    assert Range(4).intersect(Interval(1, oo)) == Range(1, 4)
    assert Range(4).intersect(Interval(1.1, oo)) == Range(2, 4)
    assert Range(4).intersect(Interval(0.1, 3)) == Range(1, 4)
    assert Range(4).intersect(Interval(0.1, 3.1)) == Range(1, 4)
    assert Range(4).intersect(Interval.open(0, 3)) == Range(1, 3)
    assert Range(4).intersect(Interval.open(0.1, 0.5)) is S.EmptySet
    assert Interval(-1, 5).intersect(S.Complexes) == Interval(-1, 5)
    assert Interval(-1, 5).intersect(S.Reals) == Interval(-1, 5)
    assert Interval(-1, 5).intersect(S.Integers) == Range(-1, 6)
    assert Interval(-1, 5).intersect(S.Naturals) == Range(1, 6)
    assert Interval(-1, 5).intersect(S.Naturals0) == Range(0, 6)

    # Null Range intersections
    assert Range(0).intersect(Interval(0.2, 0.8)) is S.EmptySet
    assert Range(0).intersect(Interval(-oo, oo)) is S.EmptySet


def test_range_is_finite_set():
    assert Range(-100, 100).is_finite_set is True
    assert Range(2, oo).is_finite_set is False
    assert Range(-oo, 50).is_finite_set is False
    assert Range(-oo, oo).is_finite_set is False
    assert Range(oo, -oo).is_finite_set is True
    assert Range(0, 0).is_finite_set is True
    assert Range(oo, oo).is_finite_set is True
    assert Range(-oo, -oo).is_finite_set is True
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    assert Range(n, n + 49).is_finite_set is True
    assert Range(n, 0).is_finite_set is True
    assert Range(-3, n + 7).is_finite_set is True
    assert Range(n, m).is_finite_set is True
    assert Range(n + m, m - n).is_finite_set is True
    assert Range(n, n + m + n).is_finite_set is True
    assert Range(n, oo).is_finite_set is False
    assert Range(-oo, n).is_finite_set is False
    assert Range(n, -oo).is_finite_set is True
    assert Range(oo, n).is_finite_set is True


def test_Range_is_iterable():
    assert Range(-100, 100).is_iterable is True
    assert Range(2, oo).is_iterable is False
    assert Range(-oo, 50).is_iterable is False
    assert Range(-oo, oo).is_iterable is False
    assert Range(oo, -oo).is_iterable is True
    assert Range(0, 0).is_iterable is True
    assert Range(oo, oo).is_iterable is True
    assert Range(-oo, -oo).is_iterable is True
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    p = Symbol('p', integer=True, positive=True)
    assert Range(n, n + 49).is_iterable is True
    assert Range(n, 0).is_iterable is False
    assert Range(-3, n + 7).is_iterable is False
    assert Range(-3, p + 7).is_iterable is False # Should work with better __iter__
    assert Range(n, m).is_iterable is False
    assert Range(n + m, m - n).is_iterable is False
    assert Range(n, n + m + n).is_iterable is False
    assert Range(n, oo).is_iterable is False
    assert Range(-oo, n).is_iterable is False
    x = Symbol('x')
    assert Range(x, x + 49).is_iterable is False
    assert Range(x, 0).is_iterable is False
    assert Range(-3, x + 7).is_iterable is False
    assert Range(x, m).is_iterable is False
    assert Range(x + m, m - x).is_iterable is False
    assert Range(x, x + m + x).is_iterable is False
    assert Range(x, oo).is_iterable is False
    assert Range(-oo, x).is_iterable is False


def test_Integers_eval_imageset():
    ans = ImageSet(Lambda(x, 2*x + Rational(3, 7)), S.Integers)
    im = imageset(Lambda(x, -2*x + Rational(3, 7)), S.Integers)
    assert im == ans
    im = imageset(Lambda(x, -2*x - Rational(11, 7)), S.Integers)
    assert im == ans
    y = Symbol('y')
    L = imageset(x, 2*x + y, S.Integers)
    assert y + 4 in L
    a, b, c = 0.092, 0.433, 0.341
    assert a in imageset(x, a + c*x, S.Integers)
    assert b in imageset(x, b + c*x, S.Integers)

    _x = symbols('x', negative=True)
    eq = _x**2 - _x + 1
    assert imageset(_x, eq, S.Integers).lamda.expr == _x**2 + _x + 1
    eq = 3*_x - 1
    assert imageset(_x, eq, S.Integers).lamda.expr == 3*_x + 2

    assert imageset(x, (x, 1/x), S.Integers) == \
        ImageSet(Lambda(x, (x, 1/x)), S.Integers)


def test_Range_eval_imageset():
    a, b, c = symbols('a b c')
    assert imageset(x, a*(x + b) + c, Range(3)) == \
        imageset(x, a*x + a*b + c, Range(3))
    eq = (x + 1)**2
    assert imageset(x, eq, Range(3)).lamda.expr == eq
    eq = a*(x + b) + c
    r = Range(3, -3, -2)
    imset = imageset(x, eq, r)
    assert imset.lamda.expr != eq
    assert list(imset) == [eq.subs(x, i).expand() for i in list(r)]


def test_fun():
    assert (FiniteSet(*ImageSet(Lambda(x, sin(pi*x/4)),
        Range(-10, 11))) == FiniteSet(-1, -sqrt(2)/2, 0, sqrt(2)/2, 1))


def test_Range_is_empty():
    i = Symbol('i', integer=True)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)

    assert Range(0).is_empty
    assert not Range(1).is_empty
    assert Range(1, 0).is_empty
    assert not Range(-1, 0).is_empty
    assert Range(i).is_empty is None
    assert Range(n).is_empty
    assert Range(p).is_empty is False
    assert Range(n, 0).is_empty is False
    assert Range(n, p).is_empty is False
    assert Range(p, n).is_empty
    assert Range(n, -1).is_empty is None
    assert Range(p, n, -1).is_empty is False


def test_Reals():
    assert 5 in S.Reals
    assert S.Pi in S.Reals
    assert -sqrt(2) in S.Reals
    assert (2, 5) not in S.Reals
    assert sqrt(-1) not in S.Reals
    assert S.Reals == Interval(-oo, oo)
    assert S.Reals != Interval(0, oo)
    assert S.Reals.is_subset(Interval(-oo, oo))
    assert S.Reals.intersect(Range(-oo, oo)) == Range(-oo, oo)
    assert S.ComplexInfinity not in S.Reals
    assert S.NaN not in S.Reals
    assert x + S.ComplexInfinity not in S.Reals


def test_Complex():
    assert 5 in S.Complexes
    assert 5 + 4*I in S.Complexes
    assert S.Pi in S.Complexes
    assert -sqrt(2) in S.Complexes
    assert -I in S.Complexes
    assert sqrt(-1) in S.Complexes
    assert S.Complexes.intersect(S.Reals) == S.Reals
    assert S.Complexes.union(S.Reals) == S.Complexes
    assert S.Complexes == ComplexRegion(S.Reals*S.Reals)
    assert (S.Complexes == ComplexRegion(Interval(1, 2)*Interval(3, 4))) == False
    assert str(S.Complexes) == "Complexes"
    assert repr(S.Complexes) == "Complexes"


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def test_intersections():
    assert S.Integers.intersect(S.Reals) == S.Integers
    assert 5 in S.Integers.intersect(S.Reals)
    assert 5 in S.Integers.intersect(S.Reals)
    assert -5 not in S.Naturals.intersect(S.Reals)
    assert 5.5 not in S.Integers.intersect(S.Reals)
    assert 5 in S.Integers.intersect(Interval(3, oo))
    assert -5 in S.Integers.intersect(Interval(-oo, 3))
    assert all(x.is_Integer
            for x in take(10, S.Integers.intersect(Interval(3, oo)) ))


def test_infinitely_indexed_set_1():
    from sympy.abc import n, m
    assert imageset(Lambda(n, n), S.Integers) == imageset(Lambda(m, m), S.Integers)

    assert imageset(Lambda(n, 2*n), S.Integers).intersect(
            imageset(Lambda(m, 2*m + 1), S.Integers)) is S.EmptySet

    assert imageset(Lambda(n, 2*n), S.Integers).intersect(
            imageset(Lambda(n, 2*n + 1), S.Integers)) is S.EmptySet

    assert imageset(Lambda(m, 2*m), S.Integers).intersect(
                imageset(Lambda(n, 3*n), S.Integers)).dummy_eq(
            ImageSet(Lambda(t, 6*t), S.Integers))

    assert imageset(x, x/2 + Rational(1, 3), S.Integers).intersect(S.Integers) is S.EmptySet
    assert imageset(x, x/2 + S.Half, S.Integers).intersect(S.Integers) is S.Integers

    # https://github.com/sympy/sympy/issues/17355
    S53 = ImageSet(Lambda(n, 5*n + 3), S.Integers)
    assert S53.intersect(S.Integers) == S53


def test_infinitely_indexed_set_2():
    from sympy.abc import n
    a = Symbol('a', integer=True)
    assert imageset(Lambda(n, n), S.Integers) == \
        imageset(Lambda(n, n + a), S.Integers)
    assert imageset(Lambda(n, n + pi), S.Integers) == \
        imageset(Lambda(n, n + a + pi), S.Integers)
    assert imageset(Lambda(n, n), S.Integers) == \
        imageset(Lambda(n, -n + a), S.Integers)
    assert imageset(Lambda(n, -6*n), S.Integers) == \
        ImageSet(Lambda(n, 6*n), S.Integers)
    assert imageset(Lambda(n, 2*n + pi), S.Integers) == \
        ImageSet(Lambda(n, 2*n + pi - 2), S.Integers)


def test_imageset_intersect_real():
    from sympy.abc import n
    assert imageset(Lambda(n, n + (n - 1)*(n + 1)*I), S.Integers).intersect(S.Reals) == FiniteSet(-1, 1)
    im = (n - 1)*(n + S.Half)
    assert imageset(Lambda(n, n + im*I), S.Integers
        ).intersect(S.Reals) == FiniteSet(1)
    assert imageset(Lambda(n, n + im*(n + 1)*I), S.Naturals0
        ).intersect(S.Reals) == FiniteSet(1)
    assert imageset(Lambda(n, n/2 + im.expand()*I), S.Integers
        ).intersect(S.Reals) == ImageSet(Lambda(x, x/2), ConditionSet(
        n, Eq(n**2 - n/2 - S(1)/2, 0), S.Integers))
    assert imageset(Lambda(n, n/(1/n - 1) + im*(n + 1)*I), S.Integers
        ).intersect(S.Reals) == FiniteSet(S.Half)
    assert imageset(Lambda(n, n/(n - 6) +
        (n - 3)*(n + 1)*I/(2*n + 2)), S.Integers).intersect(
        S.Reals) == FiniteSet(-1)
    assert imageset(Lambda(n, n/(n**2 - 9) +
        (n - 3)*(n + 1)*I/(2*n + 2)), S.Integers).intersect(
        S.Reals) is S.EmptySet
    s = ImageSet(
        Lambda(n, -I*(I*(2*pi*n - pi/4) + log(Abs(sqrt(-I))))),
        S.Integers)
    # s is unevaluated, but after intersection the result
    # should be canonical
    assert s.intersect(S.Reals) == imageset(
        Lambda(n, 2*n*pi - pi/4), S.Integers) == ImageSet(
        Lambda(n, 2*pi*n + pi*Rational(7, 4)), S.Integers)


def test_imageset_intersect_interval():
    from sympy.abc import n
    f1 = ImageSet(Lambda(n, n*pi), S.Integers)
    f2 = ImageSet(Lambda(n, 2*n), Interval(0, pi))
    f3 = ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers)
    # complex expressions
    f4 = ImageSet(Lambda(n, n*I*pi), S.Integers)
    f5 = ImageSet(Lambda(n, 2*I*n*pi + pi/2), S.Integers)
    # non-linear expressions
    f6 = ImageSet(Lambda(n, log(n)), S.Integers)
    f7 = ImageSet(Lambda(n, n**2), S.Integers)
    f8 = ImageSet(Lambda(n, Abs(n)), S.Integers)
    f9 = ImageSet(Lambda(n, exp(n)), S.Naturals0)

    assert f1.intersect(Interval(-1, 1)) == FiniteSet(0)
    assert f1.intersect(Interval(0, 2*pi, False, True)) == FiniteSet(0, pi)
    assert f2.intersect(Interval(1, 2)) == Interval(1, 2)
    assert f3.intersect(Interval(-1, 1)) == S.EmptySet
    assert f3.intersect(Interval(-5, 5)) == FiniteSet(pi*Rational(-3, 2), pi/2)
    assert f4.intersect(Interval(-1, 1)) == FiniteSet(0)
    assert f4.intersect(Interval(1, 2)) == S.EmptySet
    assert f5.intersect(Interval(0, 1)) == S.EmptySet
    assert f6.intersect(Interval(0, 1)) == FiniteSet(S.Zero, log(2))
    assert f7.intersect(Interval(0, 10)) == Intersection(f7, Interval(0, 10))
    assert f8.intersect(Interval(0, 2)) == Intersection(f8, Interval(0, 2))
    assert f9.intersect(Interval(1, 2)) == Intersection(f9, Interval(1, 2))


def test_imageset_intersect_diophantine():
    from sympy.abc import m, n
    # Check that same lambda variable for both ImageSets is handled correctly
    img1 = ImageSet(Lambda(n, 2*n + 1), S.Integers)
    img2 = ImageSet(Lambda(n, 4*n + 1), S.Integers)
    assert img1.intersect(img2) == img2
    # Empty solution set returned by diophantine:
    assert ImageSet(Lambda(n, 2*n), S.Integers).intersect(
            ImageSet(Lambda(n, 2*n + 1), S.Integers)) == S.EmptySet
    # Check intersection with S.Integers:
    assert ImageSet(Lambda(n, 9/n + 20*n/3), S.Integers).intersect(
            S.Integers) == FiniteSet(-61, -23, 23, 61)
    # Single solution (2, 3) for diophantine solution:
    assert ImageSet(Lambda(n, (n - 2)**2), S.Integers).intersect(
            ImageSet(Lambda(n, -(n - 3)**2), S.Integers)) == FiniteSet(0)
    # Single parametric solution for diophantine solution:
    assert ImageSet(Lambda(n, n**2 + 5), S.Integers).intersect(
            ImageSet(Lambda(m, 2*m), S.Integers)).dummy_eq(ImageSet(
            Lambda(n, 4*n**2 + 4*n + 6), S.Integers))
    # 4 non-parametric solution couples for dioph. equation:
    assert ImageSet(Lambda(n, n**2 - 9), S.Integers).intersect(
            ImageSet(Lambda(m, -m**2), S.Integers)) == FiniteSet(-9, 0)
    # Double parametric solution for diophantine solution:
    assert ImageSet(Lambda(m, m**2 + 40), S.Integers).intersect(
            ImageSet(Lambda(n, 41*n), S.Integers)).dummy_eq(Intersection(
            ImageSet(Lambda(m, m**2 + 40), S.Integers),
            ImageSet(Lambda(n, 41*n), S.Integers)))
    # Check that diophantine returns *all* (8) solutions (permute=True)
    assert ImageSet(Lambda(n, n**4 - 2**4), S.Integers).intersect(
            ImageSet(Lambda(m, -m**4 + 3**4), S.Integers)) == FiniteSet(0, 65)
    assert ImageSet(Lambda(n, pi/12 + n*5*pi/12), S.Integers).intersect(
            ImageSet(Lambda(n, 7*pi/12 + n*11*pi/12), S.Integers)).dummy_eq(ImageSet(
            Lambda(n, 55*pi*n/12 + 17*pi/4), S.Integers))
    # TypeError raised by diophantine (#18081)
    assert ImageSet(Lambda(n, n*log(2)), S.Integers).intersection(
        S.Integers).dummy_eq(Intersection(ImageSet(
        Lambda(n, n*log(2)), S.Integers), S.Integers))
    # NotImplementedError raised by diophantine (no solver for cubic_thue)
    assert ImageSet(Lambda(n, n**3 + 1), S.Integers).intersect(
            ImageSet(Lambda(n, n**3), S.Integers)).dummy_eq(Intersection(
            ImageSet(Lambda(n, n**3 + 1), S.Integers),
            ImageSet(Lambda(n, n**3), S.Integers)))


def test_infinitely_indexed_set_3():
    from sympy.abc import n, m
    assert imageset(Lambda(m, 2*pi*m), S.Integers).intersect(
            imageset(Lambda(n, 3*pi*n), S.Integers)).dummy_eq(
        ImageSet(Lambda(t, 6*pi*t), S.Integers))
    assert imageset(Lambda(n, 2*n + 1), S.Integers) == \
        imageset(Lambda(n, 2*n - 1), S.Integers)
    assert imageset(Lambda(n, 3*n + 2), S.Integers) == \
        imageset(Lambda(n, 3*n - 1), S.Integers)


def test_ImageSet_simplification():
    from sympy.abc import n, m
    assert imageset(Lambda(n, n), S.Integers) == S.Integers
    assert imageset(Lambda(n, sin(n)),
                    imageset(Lambda(m, tan(m)), S.Integers)) == \
            imageset(Lambda(m, sin(tan(m))), S.Integers)
    assert imageset(n, 1 + 2*n, S.Naturals) == Range(3, oo, 2)
    assert imageset(n, 1 + 2*n, S.Naturals0) == Range(1, oo, 2)
    assert imageset(n, 1 - 2*n, S.Naturals) == Range(-1, -oo, -2)


def test_ImageSet_contains():
    assert (2, S.Half) in imageset(x, (x, 1/x), S.Integers)
    assert imageset(x, x + I*3, S.Integers).intersection(S.Reals) is S.EmptySet
    i = Dummy(integer=True)
    q = imageset(x, x + I*y, S.Integers).intersection(S.Reals)
    assert q.subs(y, I*i).intersection(S.Integers) is S.Integers
    q = imageset(x, x + I*y/x, S.Integers).intersection(S.Reals)
    assert q.subs(y, 0) is S.Integers
    assert q.subs(y, I*i*x).intersection(S.Integers) is S.Integers
    z = cos(1)**2 + sin(1)**2 - 1
    q = imageset(x, x + I*z, S.Integers).intersection(S.Reals)
    assert q is not S.EmptySet


def test_ComplexRegion_contains():
    r = Symbol('r', real=True)
    # contains in ComplexRegion
    a = Interval(2, 3)
    b = Interval(4, 6)
    c = Interval(7, 9)
    c1 = ComplexRegion(a*b)
    c2 = ComplexRegion(Union(a*b, c*a))
    assert 2.5 + 4.5*I in c1
    assert 2 + 4*I in c1
    assert 3 + 4*I in c1
    assert 8 + 2.5*I in c2
    assert 2.5 + 6.1*I not in c1
    assert 4.5 + 3.2*I not in c1
    assert c1.contains(x) == Contains(x, c1, evaluate=False)
    assert c1.contains(r) == False
    assert c2.contains(x) == Contains(x, c2, evaluate=False)
    assert c2.contains(r) == False

    r1 = Interval(0, 1)
    theta1 = Interval(0, 2*S.Pi)
    c3 = ComplexRegion(r1*theta1, polar=True)
    assert (0.5 + I*6/10) in c3
    assert (S.Half + I*6/10) in c3
    assert (S.Half + .6*I) in c3
    assert (0.5 + .6*I) in c3
    assert I in c3
    assert 1 in c3
    assert 0 in c3
    assert 1 + I not in c3
    assert 1 - I not in c3
    assert c3.contains(x) == Contains(x, c3, evaluate=False)
    assert c3.contains(r + 2*I) == Contains(
        r + 2*I, c3, evaluate=False)  # is in fact False
    assert c3.contains(1/(1 + r**2)) == Contains(
        1/(1 + r**2), c3, evaluate=False)  # is in fact True

    r2 = Interval(0, 3)
    theta2 = Interval(pi, 2*pi, left_open=True)
    c4 = ComplexRegion(r2*theta2, polar=True)
    assert c4.contains(0) == True
    assert c4.contains(2 + I) == False
    assert c4.contains(-2 + I) == False
    assert c4.contains(-2 - I) == True
    assert c4.contains(2 - I) == True
    assert c4.contains(-2) == False
    assert c4.contains(2) == True
    assert c4.contains(x) == Contains(x, c4, evaluate=False)
    assert c4.contains(3/(1 + r**2)) == Contains(
        3/(1 + r**2), c4, evaluate=False)  # is in fact True

    raises(ValueError, lambda: ComplexRegion(r1*theta1, polar=2))


def test_symbolic_Range():
    n = Symbol('n')
    raises(ValueError, lambda: Range(n)[0])
    raises(IndexError, lambda: Range(n, n)[0])
    raises(ValueError, lambda: Range(n, n+1)[0])
    raises(ValueError, lambda: Range(n).size)

    n = Symbol('n', integer=True)
    raises(ValueError, lambda: Range(n)[0])
    raises(IndexError, lambda: Range(n, n)[0])
    assert Range(n, n+1)[0] == n
    raises(ValueError, lambda: Range(n).size)
    assert Range(n, n+1).size == 1

    n = Symbol('n', integer=True, nonnegative=True)
    raises(ValueError, lambda: Range(n)[0])
    raises(IndexError, lambda: Range(n, n)[0])
    assert Range(n+1)[0] == 0
    assert Range(n, n+1)[0] == n
    assert Range(n).size == n
    assert Range(n+1).size == n+1
    assert Range(n, n+1).size == 1

    n = Symbol('n', integer=True, positive=True)
    assert Range(n)[0] == 0
    assert Range(n, n+1)[0] == n
    assert Range(n).size == n
    assert Range(n, n+1).size == 1

    m = Symbol('m', integer=True, positive=True)

    assert Range(n, n+m)[0] == n
    assert Range(n, n+m).size == m
    assert Range(n, n+1).size == 1
    assert Range(n, n+m, 2).size == floor(m/2)

    m = Symbol('m', integer=True, positive=True, even=True)
    assert Range(n, n+m, 2).size == m/2


def test_issue_18400():
    n = Symbol('n', integer=True)
    raises(ValueError, lambda: imageset(lambda x: x*2, Range(n)))

    n = Symbol('n', integer=True, positive=True)
    # No exception
    assert imageset(lambda x: x*2, Range(n)) == imageset(lambda x: x*2, Range(n))


def test_ComplexRegion_intersect():
    # Polar form
    X_axis = ComplexRegion(Interval(0, oo)*FiniteSet(0, S.Pi), polar=True)

    unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    upper_half_unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    upper_half_disk = ComplexRegion(Interval(0, oo)*Interval(0, S.Pi), polar=True)
    lower_half_disk = ComplexRegion(Interval(0, oo)*Interval(S.Pi, 2*S.Pi), polar=True)
    right_half_disk = ComplexRegion(Interval(0, oo)*Interval(-S.Pi/2, S.Pi/2), polar=True)
    first_quad_disk = ComplexRegion(Interval(0, oo)*Interval(0, S.Pi/2), polar=True)

    assert upper_half_disk.intersect(unit_disk) == upper_half_unit_disk
    assert right_half_disk.intersect(first_quad_disk) == first_quad_disk
    assert upper_half_disk.intersect(right_half_disk) == first_quad_disk
    assert upper_half_disk.intersect(lower_half_disk) == X_axis

    c1 = ComplexRegion(Interval(0, 4)*Interval(0, 2*S.Pi), polar=True)
    assert c1.intersect(Interval(1, 5)) == Interval(1, 4)
    assert c1.intersect(Interval(4, 9)) == FiniteSet(4)
    assert c1.intersect(Interval(5, 12)) is S.EmptySet

    # Rectangular form
    X_axis = ComplexRegion(Interval(-oo, oo)*FiniteSet(0))

    unit_square = ComplexRegion(Interval(-1, 1)*Interval(-1, 1))
    upper_half_unit_square = ComplexRegion(Interval(-1, 1)*Interval(0, 1))
    upper_half_plane = ComplexRegion(Interval(-oo, oo)*Interval(0, oo))
    lower_half_plane = ComplexRegion(Interval(-oo, oo)*Interval(-oo, 0))
    right_half_plane = ComplexRegion(Interval(0, oo)*Interval(-oo, oo))
    first_quad_plane = ComplexRegion(Interval(0, oo)*Interval(0, oo))

    assert upper_half_plane.intersect(unit_square) == upper_half_unit_square
    assert right_half_plane.intersect(first_quad_plane) == first_quad_plane
    assert upper_half_plane.intersect(right_half_plane) == first_quad_plane
    assert upper_half_plane.intersect(lower_half_plane) == X_axis

    c1 = ComplexRegion(Interval(-5, 5)*Interval(-10, 10))
    assert c1.intersect(Interval(2, 7)) == Interval(2, 5)
    assert c1.intersect(Interval(5, 7)) == FiniteSet(5)
    assert c1.intersect(Interval(6, 9)) is S.EmptySet

    # unevaluated object
    C1 = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    C2 = ComplexRegion(Interval(-1, 1)*Interval(-1, 1))
    assert C1.intersect(C2) == Intersection(C1, C2, evaluate=False)


def test_ComplexRegion_union():
    # Polar form
    c1 = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    c2 = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    c3 = ComplexRegion(Interval(0, oo)*Interval(0, S.Pi), polar=True)
    c4 = ComplexRegion(Interval(0, oo)*Interval(S.Pi, 2*S.Pi), polar=True)

    p1 = Union(Interval(0, 1)*Interval(0, 2*S.Pi), Interval(0, 1)*Interval(0, S.Pi))
    p2 = Union(Interval(0, oo)*Interval(0, S.Pi), Interval(0, oo)*Interval(S.Pi, 2*S.Pi))

    assert c1.union(c2) == ComplexRegion(p1, polar=True)
    assert c3.union(c4) == ComplexRegion(p2, polar=True)

    # Rectangular form
    c5 = ComplexRegion(Interval(2, 5)*Interval(6, 9))
    c6 = ComplexRegion(Interval(4, 6)*Interval(10, 12))
    c7 = ComplexRegion(Interval(0, 10)*Interval(-10, 0))
    c8 = ComplexRegion(Interval(12, 16)*Interval(14, 20))

    p3 = Union(Interval(2, 5)*Interval(6, 9), Interval(4, 6)*Interval(10, 12))
    p4 = Union(Interval(0, 10)*Interval(-10, 0), Interval(12, 16)*Interval(14, 20))

    assert c5.union(c6) == ComplexRegion(p3)
    assert c7.union(c8) == ComplexRegion(p4)

    assert c1.union(Interval(2, 4)) == Union(c1, Interval(2, 4), evaluate=False)
    assert c5.union(Interval(2, 4)) == Union(c5, ComplexRegion.from_real(Interval(2, 4)))


def test_ComplexRegion_from_real():
    c1 = ComplexRegion(Interval(0, 1) * Interval(0, 2 * S.Pi), polar=True)

    raises(ValueError, lambda: c1.from_real(c1))
    assert c1.from_real(Interval(-1, 1)) == ComplexRegion(Interval(-1, 1) * FiniteSet(0), False)


def test_ComplexRegion_measure():
    a, b = Interval(2, 5), Interval(4, 8)
    theta1, theta2 = Interval(0, 2*S.Pi), Interval(0, S.Pi)
    c1 = ComplexRegion(a*b)
    c2 = ComplexRegion(Union(a*theta1, b*theta2), polar=True)

    assert c1.measure == 12
    assert c2.measure == 9*pi


def test_normalize_theta_set():
    # Interval
    assert normalize_theta_set(Interval(pi, 2*pi)) == \
        Union(FiniteSet(0), Interval.Ropen(pi, 2*pi))
    assert normalize_theta_set(Interval(pi*Rational(9, 2), 5*pi)) == Interval(pi/2, pi)
    assert normalize_theta_set(Interval(pi*Rational(-3, 2), pi/2)) == Interval.Ropen(0, 2*pi)
    assert normalize_theta_set(Interval.open(pi*Rational(-3, 2), pi/2)) == \
        Union(Interval.Ropen(0, pi/2), Interval.open(pi/2, 2*pi))
    assert normalize_theta_set(Interval.open(pi*Rational(-7, 2), pi*Rational(-3, 2))) == \
        Union(Interval.Ropen(0, pi/2), Interval.open(pi/2, 2*pi))
    assert normalize_theta_set(Interval(-pi/2, pi/2)) == \
        Union(Interval(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval.open(-pi/2, pi/2)) == \
        Union(Interval.Ropen(0, pi/2), Interval.open(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval(-4*pi, 3*pi)) == Interval.Ropen(0, 2*pi)
    assert normalize_theta_set(Interval(pi*Rational(-3, 2), -pi/2)) == Interval(pi/2, pi*Rational(3, 2))
    assert normalize_theta_set(Interval.open(0, 2*pi)) == Interval.open(0, 2*pi)
    assert normalize_theta_set(Interval.Ropen(-pi/2, pi/2)) == \
        Union(Interval.Ropen(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval.Lopen(-pi/2, pi/2)) == \
        Union(Interval(0, pi/2), Interval.open(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval(-pi/2, pi/2)) == \
        Union(Interval(0, pi/2), Interval.Ropen(pi*Rational(3, 2), 2*pi))
    assert normalize_theta_set(Interval.open(4*pi, pi*Rational(9, 2))) == Interval.open(0, pi/2)
    assert normalize_theta_set(Interval.Lopen(4*pi, pi*Rational(9, 2))) == Interval.Lopen(0, pi/2)
    assert normalize_theta_set(Interval.Ropen(4*pi, pi*Rational(9, 2))) == Interval.Ropen(0, pi/2)
    assert normalize_theta_set(Interval.open(3*pi, 5*pi)) == \
        Union(Interval.Ropen(0, pi), Interval.open(pi, 2*pi))

    # FiniteSet
    assert normalize_theta_set(FiniteSet(0, pi, 3*pi)) == FiniteSet(0, pi)
    assert normalize_theta_set(FiniteSet(0, pi/2, pi, 2*pi)) == FiniteSet(0, pi/2, pi)
    assert normalize_theta_set(FiniteSet(0, -pi/2, -pi, -2*pi)) == FiniteSet(0, pi, pi*Rational(3, 2))
    assert normalize_theta_set(FiniteSet(pi*Rational(-3, 2), pi/2)) == \
        FiniteSet(pi/2)
    assert normalize_theta_set(FiniteSet(2*pi)) == FiniteSet(0)

    # Unions
    assert normalize_theta_set(Union(Interval(0, pi/3), Interval(pi/2, pi))) == \
        Union(Interval(0, pi/3), Interval(pi/2, pi))
    assert normalize_theta_set(Union(Interval(0, pi), Interval(2*pi, pi*Rational(7, 3)))) == \
        Interval(0, pi)

    # ValueError for non-real sets
    raises(ValueError, lambda: normalize_theta_set(S.Complexes))

    # NotImplementedError for subset of reals
    raises(NotImplementedError, lambda: normalize_theta_set(Interval(0, 1)))

    # NotImplementedError without pi as coefficient
    raises(NotImplementedError, lambda: normalize_theta_set(Interval(1, 2*pi)))
    raises(NotImplementedError, lambda: normalize_theta_set(Interval(2*pi, 10)))
    raises(NotImplementedError, lambda: normalize_theta_set(FiniteSet(0, 3, 3*pi)))


def test_ComplexRegion_FiniteSet():
    x, y, z, a, b, c = symbols('x y z a b c')

    # Issue #9669
    assert ComplexRegion(FiniteSet(a, b, c)*FiniteSet(x, y, z)) == \
        FiniteSet(a + I*x, a + I*y, a + I*z, b + I*x, b + I*y,
                  b + I*z, c + I*x, c + I*y, c + I*z)
    assert ComplexRegion(FiniteSet(2)*FiniteSet(3)) == FiniteSet(2 + 3*I)


def test_union_RealSubSet():
    assert (S.Complexes).union(Interval(1, 2)) == S.Complexes
    assert (S.Complexes).union(S.Integers) == S.Complexes


def test_SetKind_fancySet():
    G = lambda *args: ImageSet(Lambda(x, x ** 2), *args)
    assert G(Interval(1, 4)).kind is SetKind(NumberKind)
    assert G(FiniteSet(1, 4)).kind is SetKind(NumberKind)
    assert S.Rationals.kind is SetKind(NumberKind)
    assert S.Naturals.kind is SetKind(NumberKind)
    assert S.Integers.kind is SetKind(NumberKind)
    assert Range(3).kind is SetKind(NumberKind)
    a = Interval(2, 3)
    b = Interval(4, 6)
    c1 = ComplexRegion(a*b)
    assert c1.kind is SetKind(TupleKind(NumberKind, NumberKind))


def test_issue_9980():
    c1 = ComplexRegion(Interval(1, 2)*Interval(2, 3))
    c2 = ComplexRegion(Interval(1, 5)*Interval(1, 3))
    R = Union(c1, c2)
    assert simplify(R) == ComplexRegion(Union(Interval(1, 2)*Interval(2, 3), \
                                    Interval(1, 5)*Interval(1, 3)), False)
    assert c1.func(*c1.args) == c1
    assert R.func(*R.args) == R


def test_issue_11732():
    interval12 = Interval(1, 2)
    finiteset1234 = FiniteSet(1, 2, 3, 4)
    pointComplex = Tuple(1, 5)

    assert (interval12 in S.Naturals) == False
    assert (interval12 in S.Naturals0) == False
    assert (interval12 in S.Integers) == False
    assert (interval12 in S.Complexes) == False

    assert (finiteset1234 in S.Naturals) == False
    assert (finiteset1234 in S.Naturals0) == False
    assert (finiteset1234 in S.Integers) == False
    assert (finiteset1234 in S.Complexes) == False

    assert (pointComplex in S.Naturals) == False
    assert (pointComplex in S.Naturals0) == False
    assert (pointComplex in S.Integers) == False
    assert (pointComplex in S.Complexes) == True


def test_issue_11730():
    unit = Interval(0, 1)
    square = ComplexRegion(unit ** 2)

    assert Union(S.Complexes, FiniteSet(oo)) != S.Complexes
    assert Union(S.Complexes, FiniteSet(eye(4))) != S.Complexes
    assert Union(unit, square) == square
    assert Intersection(S.Reals, square) == unit


def test_issue_11938():
    unit = Interval(0, 1)
    ival = Interval(1, 2)
    cr1 = ComplexRegion(ival * unit)

    assert Intersection(cr1, S.Reals) == ival
    assert Intersection(cr1, unit) == FiniteSet(1)

    arg1 = Interval(0, S.Pi)
    arg2 = FiniteSet(S.Pi)
    arg3 = Interval(S.Pi / 4, 3 * S.Pi / 4)
    cp1 = ComplexRegion(unit * arg1, polar=True)
    cp2 = ComplexRegion(unit * arg2, polar=True)
    cp3 = ComplexRegion(unit * arg3, polar=True)

    assert Intersection(cp1, S.Reals) == Interval(-1, 1)
    assert Intersection(cp2, S.Reals) == Interval(-1, 0)
    assert Intersection(cp3, S.Reals) == FiniteSet(0)


def test_issue_11914():
    a, b = Interval(0, 1), Interval(0, pi)
    c, d = Interval(2, 3), Interval(pi, 3 * pi / 2)
    cp1 = ComplexRegion(a * b, polar=True)
    cp2 = ComplexRegion(c * d, polar=True)

    assert -3 in cp1.union(cp2)
    assert -3 in cp2.union(cp1)
    assert -5 not in cp1.union(cp2)


def test_issue_9543():
    assert ImageSet(Lambda(x, x**2), S.Naturals).is_subset(S.Reals)


def test_issue_16871():
    assert ImageSet(Lambda(x, x), FiniteSet(1)) == {1}
    assert ImageSet(Lambda(x, x - 3), S.Integers
        ).intersection(S.Integers) is S.Integers


@XFAIL
def test_issue_16871b():
    assert ImageSet(Lambda(x, x - 3), S.Integers).is_subset(S.Integers)


def test_issue_18050():
    assert imageset(Lambda(x, I*x + 1), S.Integers
        ) == ImageSet(Lambda(x, I*x + 1), S.Integers)
    assert imageset(Lambda(x, 3*I*x + 4 + 8*I), S.Integers
        ) == ImageSet(Lambda(x, 3*I*x + 4 + 2*I), S.Integers)
    # no 'Mod' for next 2 tests:
    assert imageset(Lambda(x, 2*x + 3*I), S.Integers
        ) == ImageSet(Lambda(x, 2*x + 3*I), S.Integers)
    r = Symbol('r', positive=True)
    assert imageset(Lambda(x, r*x + 10), S.Integers
        ) == ImageSet(Lambda(x, r*x + 10), S.Integers)
    # reduce real part:
    assert imageset(Lambda(x, 3*x + 8 + 5*I), S.Integers
        ) == ImageSet(Lambda(x, 3*x + 2 + 5*I), S.Integers)


def test_Rationals():
    assert S.Integers.is_subset(S.Rationals)
    assert S.Naturals.is_subset(S.Rationals)
    assert S.Naturals0.is_subset(S.Rationals)
    assert S.Rationals.is_subset(S.Reals)
    assert S.Rationals.inf is -oo
    assert S.Rationals.sup is oo
    it = iter(S.Rationals)
    assert [next(it) for i in range(12)] == [
        0, 1, -1, S.Half, 2, Rational(-1, 2), -2,
        Rational(1, 3), 3, Rational(-1, 3), -3, Rational(2, 3)]
    assert Basic() not in S.Rationals
    assert S.Half in S.Rationals
    assert S.Rationals.contains(0.5) == Contains(
        0.5, S.Rationals, evaluate=False)
    assert 2 in S.Rationals
    r = symbols('r', rational=True)
    assert r in S.Rationals
    raises(TypeError, lambda: x in S.Rationals)
    # issue #18134:
    assert S.Rationals.boundary == S.Reals
    assert S.Rationals.closure == S.Reals
    assert S.Rationals.is_open == False
    assert S.Rationals.is_closed == False


def test_NZQRC_unions():
    # check that all trivial number set unions are simplified:
    nbrsets = (S.Naturals, S.Naturals0, S.Integers, S.Rationals,
        S.Reals, S.Complexes)
    unions = (Union(a, b) for a in nbrsets for b in nbrsets)
    assert all(u.is_Union is False for u in unions)


def test_imageset_intersection():
    n = Dummy()
    s = ImageSet(Lambda(n, -I*(I*(2*pi*n - pi/4) +
        log(Abs(sqrt(-I))))), S.Integers)
    assert s.intersect(S.Reals) == ImageSet(
        Lambda(n, 2*pi*n + pi*Rational(7, 4)), S.Integers)


def test_issue_17858():
    assert 1 in Range(-oo, oo)
    assert 0 in Range(oo, -oo, -1)
    assert oo not in Range(-oo, oo)
    assert -oo not in Range(-oo, oo)

def test_issue_17859():
    r = Range(-oo,oo)
    raises(ValueError,lambda: r[::2])
    raises(ValueError, lambda: r[::-2])
    r = Range(oo,-oo,-1)
    raises(ValueError,lambda: r[::2])
    raises(ValueError, lambda: r[::-2])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\norm.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from .qdq_base_operator import QDQOperatorBase


class QDQNormalization(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type in {"InstanceNormalization", "LayerNormalization", "BatchNormalization"}

        # Input
        self.quantizer.quantize_activation_tensor(node.input[0])

        # Scale
        scale_is_initializer = self.quantizer.is_input_a_initializer(node.input[1])
        scale_is_per_channel, scale_channel_axis = self.quantizer.is_tensor_per_channel(
            node.input[1], default_axis=1, op_type=node.op_type
        )

        if scale_is_per_channel:
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], axis=scale_channel_axis)
        elif scale_is_initializer:
            self.quantizer.quantize_weight_tensor(node.input[1])
        else:
            self.quantizer.quantize_activation_tensor(node.input[1])

        # Bias
        if len(node.input) > 2 and node.input[2]:
            self.quantizer.quantize_bias_tensor(node.name, node.input[2], node.input[0], node.input[1])

        # Output
        if not self.disable_qdq_for_node_output:
            for output_name in node.output:
                self.quantizer.quantize_activation_tensor(output_name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\affinity_helper.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Get/Set cpu affinity. Currently only support part of Unix system
import logging
import os

logger = logging.getLogger(__name__)


class AffinitySetting:
    def __init__(self):
        self.pid = os.getpid()
        self.affinity = None
        self.is_os_supported = hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")
        if not self.is_os_supported:
            logger.warning("Current OS does not support os.get_affinity() and os.set_affinity()")

    def get_affinity(self):
        if self.is_os_supported:
            self.affinity = os.sched_getaffinity(self.pid)

    def set_affinity(self):
        if self.is_os_supported:
            current_affinity = os.sched_getaffinity(self.pid)
            if self.affinity != current_affinity:
                logger.warning(
                    "Replacing affinity setting %s with %s",
                    str(current_affinity),
                    str(self.affinity),
                )
                os.sched_setaffinity(self.pid, self.affinity)


if __name__ == "__main__":
    affi_helper = AffinitySetting()
    affi_helper.get_affinity()
    affi_helper.set_affinity()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\compat\singleton.py ===
# Copyright (c) 2010-2024 openpyxl

import weakref


class Singleton(type):
    """
    Singleton metaclass
    Based on Python Cookbook 3rd Edition Recipe 9.13
    Only one instance of a class can exist. Does not work with __slots__
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.__instance = None

    def __call__(self, *args, **kw):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kw)
        return self.__instance


class Cached(type):
    """
    Caching metaclass
    Child classes will only create new instances of themselves if
    one doesn't already exist. Does not work with __slots__
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.__cache = weakref.WeakValueDictionary()

    def __call__(self, *args):
        if args in self.__cache:
            return self.__cache[args]

        obj = super().__call__(*args)
        self.__cache[args] = obj
        return obj

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\_build_tables.py ===
#-----------------------------------------------------------------
# pycparser: _build_tables.py
#
# A dummy for generating the lexing/parsing tables and and
# compiling them into .pyc for faster execution in optimized mode.
# Also generates AST code from the configuration file.
# Should be called from the pycparser directory.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#-----------------------------------------------------------------

# Insert '.' and '..' as first entries to the search path for modules.
# Restricted environments like embeddable python do not include the
# current working directory on startup.
import importlib
import sys
sys.path[0:0] = ['.', '..']

# Generate c_ast.py
from _ast_gen import ASTCodeGenerator
ast_gen = ASTCodeGenerator('_c_ast.cfg')
ast_gen.generate(open('c_ast.py', 'w'))

from pycparser import c_parser

# Generates the tables
#
c_parser.CParser(
    lex_optimize=True,
    yacc_debug=False,
    yacc_optimize=True)

# Load to compile into .pyc
#
importlib.invalidate_caches()

import lextab
import yacctab
import c_ast

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\_has_cy.py ===
# util/_has_cy.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

import os
import typing


def _import_cy_extensions():
    # all cython extension extension modules are treated as optional by the
    # setup, so to ensure that all are compiled, all should be imported here
    from ..cyextension import collections
    from ..cyextension import immutabledict
    from ..cyextension import processors
    from ..cyextension import resultproxy
    from ..cyextension import util

    return (collections, immutabledict, processors, resultproxy, util)


_CYEXTENSION_MSG: str
if not typing.TYPE_CHECKING:
    if os.environ.get("DISABLE_SQLALCHEMY_CEXT_RUNTIME"):
        HAS_CYEXTENSION = False
        _CYEXTENSION_MSG = "DISABLE_SQLALCHEMY_CEXT_RUNTIME is set"
    else:
        try:
            _import_cy_extensions()
        except ImportError as err:
            HAS_CYEXTENSION = False
            _CYEXTENSION_MSG = str(err)
        else:
            _CYEXTENSION_MSG = "Loaded"
            HAS_CYEXTENSION = True
else:
    HAS_CYEXTENSION = False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\futils.py ===
from itertools import chain
from sympy.codegen.fnodes import Module
from sympy.core.symbol import Dummy
from sympy.printing.fortran import FCodePrinter

""" This module collects utilities for rendering Fortran code. """


def render_as_module(definitions, name, declarations=(), printer_settings=None):
    """ Creates a ``Module`` instance and renders it as a string.

    This generates Fortran source code for a module with the correct ``use`` statements.

    Parameters
    ==========

    definitions : iterable
        Passed to :class:`sympy.codegen.fnodes.Module`.
    name : str
        Passed to :class:`sympy.codegen.fnodes.Module`.
    declarations : iterable
        Passed to :class:`sympy.codegen.fnodes.Module`. It will be extended with
        use statements, 'implicit none' and public list generated from ``definitions``.
    printer_settings : dict
        Passed to ``FCodePrinter`` (default: ``{'standard': 2003, 'source_format': 'free'}``).

    """
    printer_settings = printer_settings or {'standard': 2003, 'source_format': 'free'}
    printer = FCodePrinter(printer_settings)
    dummy = Dummy()
    if isinstance(definitions, Module):
        raise ValueError("This function expects to construct a module on its own.")
    mod = Module(name, chain(declarations, [dummy]), definitions)
    fstr = printer.doprint(mod)
    module_use_str = '   %s\n' % '   \n'.join(['use %s, only: %s' % (k, ', '.join(v)) for
                                                k, v in printer.module_uses.items()])
    module_use_str += '   implicit none\n'
    module_use_str += '   private\n'
    module_use_str += '   public %s\n' % ', '.join([str(node.name) for node in definitions if getattr(node, 'name', None)])
    return fstr.replace(printer.doprint(dummy), module_use_str)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\source.py ===
"""
This module adds several functions for interactive source code inspection.
"""


def get_class(lookup_view):
    """
    Convert a string version of a class name to the object.

    For example, get_class('sympy.core.Basic') will return
    class Basic located in module sympy.core
    """
    if isinstance(lookup_view, str):
        mod_name, func_name = get_mod_func(lookup_view)
        if func_name != '':
            lookup_view = getattr(
                __import__(mod_name, {}, {}, ['*']), func_name)
            if not callable(lookup_view):
                raise AttributeError(
                    "'%s.%s' is not a callable." % (mod_name, func_name))
    return lookup_view


def get_mod_func(callback):
    """
    splits the string path to a class into a string path to the module
    and the name of the class.

    Examples
    ========

    >>> from sympy.utilities.source import get_mod_func
    >>> get_mod_func('sympy.core.basic.Basic')
    ('sympy.core.basic', 'Basic')

    """
    dot = callback.rfind('.')
    if dot == -1:
        return callback, ''
    return callback[:dot], callback[dot + 1:]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\tests\test_secondquant.py ===
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.physics.secondquant import (
    Dagger, Bd, VarBosonicBasis, BBra, B, BKet, FixedBosonicBasis,
    matrix_rep, apply_operators, InnerProduct, Commutator, KroneckerDelta,
    AnnihilateBoson, CreateBoson, BosonicOperator,
    F, Fd, FKet, BosonState, CreateFermion, AnnihilateFermion,
    evaluate_deltas, AntiSymmetricTensor, contraction, NO, wicks,
    PermutationOperator, simplify_index_permutations,
    _sort_anticommuting_fermions, _get_ordered_dummies,
    substitute_dummies, FockStateBosonKet,
    ContractionAppliesOnlyToFermions
)

from sympy.concrete.summations import Sum
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.repr import srepr
from sympy.simplify.simplify import simplify

from sympy.testing.pytest import slow, raises
from sympy.printing.latex import latex


def test_PermutationOperator():
    p, q, r, s = symbols('p,q,r,s')
    f, g, h, i = map(Function, 'fghi')
    P = PermutationOperator
    assert P(p, q).get_permuted(f(p)*g(q)) == -f(q)*g(p)
    assert P(p, q).get_permuted(f(p, q)) == -f(q, p)
    assert P(p, q).get_permuted(f(p)) == f(p)
    expr = (f(p)*g(q)*h(r)*i(s)
        - f(q)*g(p)*h(r)*i(s)
        - f(p)*g(q)*h(s)*i(r)
        + f(q)*g(p)*h(s)*i(r))
    perms = [P(p, q), P(r, s)]
    assert (simplify_index_permutations(expr, perms) ==
        P(p, q)*P(r, s)*f(p)*g(q)*h(r)*i(s))
    assert latex(P(p, q)) == 'P(pq)'

    p1, p2 = symbols('p1,p2')
    assert latex(P(p1,p2) == 'P(p_{1}p_{2})')

def test_index_permutations_with_dummies():
    a, b, c, d = symbols('a b c d')
    p, q, r, s = symbols('p q r s', cls=Dummy)
    f, g = map(Function, 'fg')
    P = PermutationOperator

    # No dummy substitution necessary
    expr = f(a, b, p, q) - f(b, a, p, q)
    assert simplify_index_permutations(
        expr, [P(a, b)]) == P(a, b)*f(a, b, p, q)

    # Cases where dummy substitution is needed
    expected = P(a, b)*substitute_dummies(f(a, b, p, q))

    expr = f(a, b, p, q) - f(b, a, q, p)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expected == substitute_dummies(result)

    expr = f(a, b, q, p) - f(b, a, p, q)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expected == substitute_dummies(result)

    # A case where nothing can be done
    expr = f(a, b, q, p) - g(b, a, p, q)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expr == result


def test_dagger():
    i, j, n, m = symbols('i,j,n,m')
    assert Dagger(1) == 1
    assert Dagger(1.0) == 1.0
    assert Dagger(2*I) == -2*I
    assert Dagger(S.Half*I/3.0) == I*Rational(-1, 2)/3.0
    assert Dagger(BKet([n])) == BBra([n])
    assert Dagger(B(0)) == Bd(0)
    assert Dagger(Bd(0)) == B(0)
    assert Dagger(B(n)) == Bd(n)
    assert Dagger(Bd(n)) == B(n)
    assert Dagger(B(0) + B(1)) == Bd(0) + Bd(1)
    assert Dagger(n*m) == Dagger(n)*Dagger(m)  # n, m commute
    assert Dagger(B(n)*B(m)) == Bd(m)*Bd(n)
    assert Dagger(B(n)**10) == Dagger(B(n))**10
    assert Dagger('a') == Dagger(Symbol('a'))
    assert Dagger(Dagger('a')) == Symbol('a')
    assert Dagger(exp(2 * I)) == exp(-2 * I)
    assert Dagger(i) == conjugate(i)


def test_operator():
    i, j = symbols('i,j')
    o = BosonicOperator(i)
    assert o.state == i
    assert o.is_symbolic
    o = BosonicOperator(1)
    assert o.state == 1
    assert not o.is_symbolic


def test_create():
    i, j, n, m, p1 = symbols('i,j,n,m,p1')
    o = Bd(i)
    assert latex(o) == "{b^\\dagger_{i}}"
    assert latex(Bd(p1)) == "{b^\\dagger_{p_{1}}}"
    assert isinstance(o, CreateBoson)
    o = o.subs(i, j)
    assert o.atoms(Symbol) == {j}
    o = Bd(0)
    assert o.apply_operator(BKet([n])) == sqrt(n + 1)*BKet([n + 1])
    o = Bd(n)
    assert o.apply_operator(BKet([n])) == o*BKet([n])


def test_annihilate():
    i, j, n, m, p1 = symbols('i,j,n,m,p1')
    o = B(i)
    assert latex(o) == "b_{i}"
    assert latex(B(p1)) == "b_{p_{1}}"
    assert isinstance(o, AnnihilateBoson)
    o = o.subs(i, j)
    assert o.atoms(Symbol) == {j}
    o = B(0)
    assert o.apply_operator(BKet([n])) == sqrt(n)*BKet([n - 1])
    o = B(n)
    assert o.apply_operator(BKet([n])) == o*BKet([n])


def test_basic_state():
    i, j, n, m = symbols('i,j,n,m')
    s = BosonState([0, 1, 2, 3, 4])
    assert len(s) == 5
    assert s.args[0] == tuple(range(5))
    assert s.up(0) == BosonState([1, 1, 2, 3, 4])
    assert s.down(4) == BosonState([0, 1, 2, 3, 3])
    for i in range(5):
        assert s.up(i).down(i) == s
    assert s.down(0) == 0
    for i in range(5):
        assert s[i] == i
    s = BosonState([n, m])
    assert s.down(0) == BosonState([n - 1, m])
    assert s.up(0) == BosonState([n + 1, m])


def test_basic_apply():
    n = symbols("n")
    e = B(0)*BKet([n])
    assert apply_operators(e) == sqrt(n)*BKet([n - 1])
    e = Bd(0)*BKet([n])
    assert apply_operators(e) == sqrt(n + 1)*BKet([n + 1])


def test_complex_apply():
    n, m = symbols("n,m")
    o = Bd(0)*B(0)*Bd(1)*B(0)
    e = apply_operators(o*BKet([n, m]))
    answer = sqrt(n)*sqrt(m + 1)*(-1 + n)*BKet([-1 + n, 1 + m])
    assert expand(e) == expand(answer)


def test_number_operator():
    n = symbols("n")
    o = Bd(0)*B(0)
    e = apply_operators(o*BKet([n]))
    assert e == n*BKet([n])


def test_inner_product():
    i, j, k, l = symbols('i,j,k,l')
    s1 = BBra([0])
    s2 = BKet([1])
    assert InnerProduct(s1, Dagger(s1)) == 1
    assert InnerProduct(s1, s2) == 0
    s1 = BBra([i, j])
    s2 = BKet([k, l])
    r = InnerProduct(s1, s2)
    assert r == KroneckerDelta(i, k)*KroneckerDelta(j, l)


def test_symbolic_matrix_elements():
    n, m = symbols('n,m')
    s1 = BBra([n])
    s2 = BKet([m])
    o = B(0)
    e = apply_operators(s1*o*s2)
    assert e == sqrt(m)*KroneckerDelta(n, m - 1)


def test_matrix_elements():
    b = VarBosonicBasis(5)
    o = B(0)
    m = matrix_rep(o, b)
    for i in range(4):
        assert m[i, i + 1] == sqrt(i + 1)
    o = Bd(0)
    m = matrix_rep(o, b)
    for i in range(4):
        assert m[i + 1, i] == sqrt(i + 1)


def test_fixed_bosonic_basis():
    b = FixedBosonicBasis(2, 2)
    # assert b == [FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]
    state = b.state(1)
    assert state == FockStateBosonKet((1, 1))
    assert b.index(state) == 1
    assert b.state(1) == b[1]
    assert len(b) == 3
    assert str(b) == '[FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]'
    assert repr(b) == '[FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]'
    assert srepr(b) == '[FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]'


@slow
def test_sho():
    n, m = symbols('n,m')
    h_n = Bd(n)*B(n)*(n + S.Half)
    H = Sum(h_n, (n, 0, 5))
    o = H.doit(deep=False)
    b = FixedBosonicBasis(2, 6)
    m = matrix_rep(o, b)
    # We need to double check these energy values to make sure that they
    # are correct and have the proper degeneracies!
    diag = [1, 2, 3, 3, 4, 5, 4, 5, 6, 7, 5, 6, 7, 8, 9, 6, 7, 8, 9, 10, 11]
    for i in range(len(diag)):
        assert diag[i] == m[i, i]


def test_commutation():
    n, m = symbols("n,m", above_fermi=True)
    c = Commutator(B(0), Bd(0))
    assert c == 1
    c = Commutator(Bd(0), B(0))
    assert c == -1
    c = Commutator(B(n), Bd(0))
    assert c == KroneckerDelta(n, 0)
    c = Commutator(B(0), B(0))
    assert c == 0
    c = Commutator(B(0), Bd(0))
    e = simplify(apply_operators(c*BKet([n])))
    assert e == BKet([n])
    c = Commutator(B(0), B(1))
    e = simplify(apply_operators(c*BKet([n, m])))
    assert e == 0

    c = Commutator(F(m), Fd(m))
    assert c == +1 - 2*NO(Fd(m)*F(m))
    c = Commutator(Fd(m), F(m))
    assert c.expand() == -1 + 2*NO(Fd(m)*F(m))

    C = Commutator
    X, Y, Z = symbols('X,Y,Z', commutative=False)
    assert C(C(X, Y), Z) != 0
    assert C(C(X, Z), Y) != 0
    assert C(Y, C(X, Z)) != 0

    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    p, q, r, s = symbols('p,q,r,s')
    D = KroneckerDelta

    assert C(Fd(a), F(i)) == -2*NO(F(i)*Fd(a))
    assert C(Fd(j), NO(Fd(a)*F(i))).doit(wicks=True) == -D(j, i)*Fd(a)
    assert C(Fd(a)*F(i), Fd(b)*F(j)).doit(wicks=True) == 0

    c1 = Commutator(F(a), Fd(a))
    assert Commutator.eval(c1, c1) == 0
    c = Commutator(Fd(a)*F(i),Fd(b)*F(j))
    assert latex(c) == r'\left[{a^\dagger_{a}} a_{i},{a^\dagger_{b}} a_{j}\right]'
    assert repr(c) == 'Commutator(CreateFermion(a)*AnnihilateFermion(i),CreateFermion(b)*AnnihilateFermion(j))'
    assert str(c) == '[CreateFermion(a)*AnnihilateFermion(i),CreateFermion(b)*AnnihilateFermion(j)]'


def test_create_f():
    i, j, n, m = symbols('i,j,n,m')
    o = Fd(i)
    assert isinstance(o, CreateFermion)
    o = o.subs(i, j)
    assert o.atoms(Symbol) == {j}
    o = Fd(1)
    assert o.apply_operator(FKet([n])) == FKet([1, n])
    assert o.apply_operator(FKet([n])) == -FKet([n, 1])
    o = Fd(n)
    assert o.apply_operator(FKet([])) == FKet([n])

    vacuum = FKet([], fermi_level=4)
    assert vacuum == FKet([], fermi_level=4)

    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    p, q, r, s = symbols('p,q,r,s')
    p1 = symbols("p1")

    assert Fd(i).apply_operator(FKet([i, j, k], 4)) == FKet([j, k], 4)
    assert Fd(a).apply_operator(FKet([i, b, k], 4)) == FKet([a, i, b, k], 4)

    assert Dagger(B(p)).apply_operator(q) == q*CreateBoson(p)
    assert repr(Fd(p)) == 'CreateFermion(p)'
    assert srepr(Fd(p)) == "CreateFermion(Symbol('p'))"
    assert latex(Fd(p)) == r'{a^\dagger_{p}}'
    assert latex(Fd(p1)) == r'{a^\dagger_{p_{1}}}'
    assert latex(FKet([a,i], 1)) == r"\left|\left( a, \  i\right)\right\rangle"
    assert latex(FKet([j,i,b,a], 2)) == r"\left|\left( a, \  b, \  i, \  j\right)\right\rangle"


def test_annihilate_f():
    i, j, n, m = symbols('i,j,n,m')
    o = F(i)
    assert isinstance(o, AnnihilateFermion)
    o = o.subs(i, j)
    assert o.atoms(Symbol) == {j}
    o = F(1)
    assert o.apply_operator(FKet([1, n])) == FKet([n])
    assert o.apply_operator(FKet([n, 1])) == -FKet([n])
    o = F(n)
    assert o.apply_operator(FKet([n])) == FKet([])

    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    p, q, r, s = symbols('p,q,r,s')
    p1 = symbols('p1')

    assert F(i).apply_operator(FKet([i, j, k], 4)) == 0
    assert F(a).apply_operator(FKet([i, b, k], 4)) == 0
    assert F(l).apply_operator(FKet([i, j, k], 3)) == 0
    assert F(l).apply_operator(FKet([i, j, k], 4)) == FKet([l, i, j, k], 4)
    assert str(F(p)) == 'f(p)'
    assert repr(F(p)) == 'AnnihilateFermion(p)'
    assert srepr(F(p)) == "AnnihilateFermion(Symbol('p'))"
    assert latex(F(p)) == 'a_{p}'
    assert latex(F(p1)) == 'a_{p_{1}}'


def test_create_b():
    i, j, n, m = symbols('i,j,n,m')
    o = Bd(i)
    assert isinstance(o, CreateBoson)
    o = o.subs(i, j)
    assert o.atoms(Symbol) == {j}
    o = Bd(0)
    assert o.apply_operator(BKet([n])) == sqrt(n + 1)*BKet([n + 1])
    o = Bd(n)
    assert o.apply_operator(BKet([n])) == o*BKet([n])


def test_annihilate_b():
    i, j, n, m = symbols('i,j,n,m')
    o = B(i)
    assert isinstance(o, AnnihilateBoson)
    o = o.subs(i, j)
    assert o.atoms(Symbol) == {j}
    o = B(0)


def test_wicks():
    p, q, r, s = symbols('p,q,r,s', above_fermi=True)

    # Testing for particles only

    str = F(p)*Fd(q)
    assert wicks(str) == NO(F(p)*Fd(q)) + KroneckerDelta(p, q)
    str = Fd(p)*F(q)
    assert wicks(str) == NO(Fd(p)*F(q))

    str = F(p)*Fd(q)*F(r)*Fd(s)
    nstr = wicks(str)
    fasit = NO(
        KroneckerDelta(p, q)*KroneckerDelta(r, s)
        + KroneckerDelta(p, q)*AnnihilateFermion(r)*CreateFermion(s)
        + KroneckerDelta(r, s)*AnnihilateFermion(p)*CreateFermion(q)
        - KroneckerDelta(p, s)*AnnihilateFermion(r)*CreateFermion(q)
        - AnnihilateFermion(p)*AnnihilateFermion(r)*CreateFermion(q)*CreateFermion(s))
    assert nstr == fasit

    assert (p*q*nstr).expand() == wicks(p*q*str)
    assert (nstr*p*q*2).expand() == wicks(str*p*q*2)

    # Testing CC equations particles and holes
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)
    p, q, r, s = symbols('p q r s', cls=Dummy)

    assert (wicks(F(a)*NO(F(i)*F(j))*Fd(b)) ==
            NO(F(a)*F(i)*F(j)*Fd(b)) +
            KroneckerDelta(a, b)*NO(F(i)*F(j)))
    assert (wicks(F(a)*NO(F(i)*F(j)*F(k))*Fd(b)) ==
            NO(F(a)*F(i)*F(j)*F(k)*Fd(b)) -
            KroneckerDelta(a, b)*NO(F(i)*F(j)*F(k)))

    expr = wicks(Fd(i)*NO(Fd(j)*F(k))*F(l))
    assert (expr ==
           -KroneckerDelta(i, k)*NO(Fd(j)*F(l)) -
            KroneckerDelta(j, l)*NO(Fd(i)*F(k)) -
            KroneckerDelta(i, k)*KroneckerDelta(j, l) +
            KroneckerDelta(i, l)*NO(Fd(j)*F(k)) +
            NO(Fd(i)*Fd(j)*F(k)*F(l)))
    expr = wicks(F(a)*NO(F(b)*Fd(c))*Fd(d))
    assert (expr ==
           -KroneckerDelta(a, c)*NO(F(b)*Fd(d)) -
            KroneckerDelta(b, d)*NO(F(a)*Fd(c)) -
            KroneckerDelta(a, c)*KroneckerDelta(b, d) +
            KroneckerDelta(a, d)*NO(F(b)*Fd(c)) +
            NO(F(a)*F(b)*Fd(c)*Fd(d)))


def test_NO():
    i, j, k, l = symbols('i j k l', below_fermi=True)
    a, b, c, d = symbols('a b c d', above_fermi=True)
    p, q, r, s = symbols('p q r s', cls=Dummy)

    assert (NO(Fd(p)*F(q) + Fd(a)*F(b)) ==
       NO(Fd(p)*F(q)) + NO(Fd(a)*F(b)))
    assert (NO(Fd(i)*NO(F(j)*Fd(a))) ==
       NO(Fd(i)*F(j)*Fd(a)))
    assert NO(1) == 1
    assert NO(i) == i
    assert (NO(Fd(a)*Fd(b)*(F(c) + F(d))) ==
            NO(Fd(a)*Fd(b)*F(c)) +
            NO(Fd(a)*Fd(b)*F(d)))

    assert NO(Fd(a)*F(b))._remove_brackets() == Fd(a)*F(b)
    assert NO(F(j)*Fd(i))._remove_brackets() == F(j)*Fd(i)

    assert (NO(Fd(p)*F(q)).subs(Fd(p), Fd(a) + Fd(i)) ==
            NO(Fd(a)*F(q)) + NO(Fd(i)*F(q)))
    assert (NO(Fd(p)*F(q)).subs(F(q), F(a) + F(i)) ==
            NO(Fd(p)*F(a)) + NO(Fd(p)*F(i)))

    expr = NO(Fd(p)*F(q))._remove_brackets()
    assert wicks(expr) == NO(expr)

    assert NO(Fd(a)*F(b)) == - NO(F(b)*Fd(a))

    no = NO(Fd(a)*F(i)*F(b)*Fd(j))
    l1 = list(no.iter_q_creators())
    assert l1 == [0, 1]
    l2 = list(no.iter_q_annihilators())
    assert l2 == [3, 2]
    no = NO(Fd(a)*Fd(i))
    assert no.has_q_creators == 1
    assert no.has_q_annihilators == -1
    assert str(no) == ':CreateFermion(a)*CreateFermion(i):'
    assert repr(no) == 'NO(CreateFermion(a)*CreateFermion(i))'
    assert latex(no) == r'\left\{{a^\dagger_{a}} {a^\dagger_{i}}\right\}'
    raises(NotImplementedError, lambda:  NO(Bd(p)*F(q)))


def test_sorting():
    i, j = symbols('i,j', below_fermi=True)
    a, b = symbols('a,b', above_fermi=True)
    p, q = symbols('p,q')

    # p, q
    assert _sort_anticommuting_fermions([Fd(p), F(q)]) == ([Fd(p), F(q)], 0)
    assert _sort_anticommuting_fermions([F(p), Fd(q)]) == ([Fd(q), F(p)], 1)

    # i, p
    assert _sort_anticommuting_fermions([F(p), Fd(i)]) == ([F(p), Fd(i)], 0)
    assert _sort_anticommuting_fermions([Fd(i), F(p)]) == ([F(p), Fd(i)], 1)
    assert _sort_anticommuting_fermions([Fd(p), Fd(i)]) == ([Fd(p), Fd(i)], 0)
    assert _sort_anticommuting_fermions([Fd(i), Fd(p)]) == ([Fd(p), Fd(i)], 1)
    assert _sort_anticommuting_fermions([F(p), F(i)]) == ([F(i), F(p)], 1)
    assert _sort_anticommuting_fermions([F(i), F(p)]) == ([F(i), F(p)], 0)
    assert _sort_anticommuting_fermions([Fd(p), F(i)]) == ([F(i), Fd(p)], 1)
    assert _sort_anticommuting_fermions([F(i), Fd(p)]) == ([F(i), Fd(p)], 0)

    # a, p
    assert _sort_anticommuting_fermions([F(p), Fd(a)]) == ([Fd(a), F(p)], 1)
    assert _sort_anticommuting_fermions([Fd(a), F(p)]) == ([Fd(a), F(p)], 0)
    assert _sort_anticommuting_fermions([Fd(p), Fd(a)]) == ([Fd(a), Fd(p)], 1)
    assert _sort_anticommuting_fermions([Fd(a), Fd(p)]) == ([Fd(a), Fd(p)], 0)
    assert _sort_anticommuting_fermions([F(p), F(a)]) == ([F(p), F(a)], 0)
    assert _sort_anticommuting_fermions([F(a), F(p)]) == ([F(p), F(a)], 1)
    assert _sort_anticommuting_fermions([Fd(p), F(a)]) == ([Fd(p), F(a)], 0)
    assert _sort_anticommuting_fermions([F(a), Fd(p)]) == ([Fd(p), F(a)], 1)

    # i, a
    assert _sort_anticommuting_fermions([F(i), Fd(j)]) == ([F(i), Fd(j)], 0)
    assert _sort_anticommuting_fermions([Fd(j), F(i)]) == ([F(i), Fd(j)], 1)
    assert _sort_anticommuting_fermions([Fd(a), Fd(i)]) == ([Fd(a), Fd(i)], 0)
    assert _sort_anticommuting_fermions([Fd(i), Fd(a)]) == ([Fd(a), Fd(i)], 1)
    assert _sort_anticommuting_fermions([F(a), F(i)]) == ([F(i), F(a)], 1)
    assert _sort_anticommuting_fermions([F(i), F(a)]) == ([F(i), F(a)], 0)


def test_contraction():
    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    p, q, r, s = symbols('p,q,r,s')
    assert contraction(Fd(i), F(j)) == KroneckerDelta(i, j)
    assert contraction(F(a), Fd(b)) == KroneckerDelta(a, b)
    assert contraction(F(a), Fd(i)) == 0
    assert contraction(Fd(a), F(i)) == 0
    assert contraction(F(i), Fd(a)) == 0
    assert contraction(Fd(i), F(a)) == 0
    assert contraction(Fd(i), F(p)) == KroneckerDelta(i, p)
    restr = evaluate_deltas(contraction(Fd(p), F(q)))
    assert restr.is_only_below_fermi
    restr = evaluate_deltas(contraction(F(p), Fd(q)))
    assert restr.is_only_above_fermi
    raises(ContractionAppliesOnlyToFermions, lambda: contraction(B(a), Fd(b)))


def test_evaluate_deltas():
    i, j, k = symbols('i,j,k')

    r = KroneckerDelta(i, j) * KroneckerDelta(j, k)
    assert evaluate_deltas(r) == KroneckerDelta(i, k)

    r = KroneckerDelta(i, 0) * KroneckerDelta(j, k)
    assert evaluate_deltas(r) == KroneckerDelta(i, 0) * KroneckerDelta(j, k)

    r = KroneckerDelta(1, j) * KroneckerDelta(j, k)
    assert evaluate_deltas(r) == KroneckerDelta(1, k)

    r = KroneckerDelta(j, 2) * KroneckerDelta(k, j)
    assert evaluate_deltas(r) == KroneckerDelta(2, k)

    r = KroneckerDelta(i, 0) * KroneckerDelta(i, j) * KroneckerDelta(j, 1)
    assert evaluate_deltas(r) == 0

    r = (KroneckerDelta(0, i) * KroneckerDelta(0, j)
         * KroneckerDelta(1, j) * KroneckerDelta(1, j))
    assert evaluate_deltas(r) == 0


def test_Tensors():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)
    p, q, r, s = symbols('p q r s')

    AT = AntiSymmetricTensor
    assert AT('t', (a, b), (i, j)) == -AT('t', (b, a), (i, j))
    assert AT('t', (a, b), (i, j)) == AT('t', (b, a), (j, i))
    assert AT('t', (a, b), (i, j)) == -AT('t', (a, b), (j, i))
    assert AT('t', (a, a), (i, j)) == 0
    assert AT('t', (a, b), (i, i)) == 0
    assert AT('t', (a, b, c), (i, j)) == -AT('t', (b, a, c), (i, j))
    assert AT('t', (a, b, c), (i, j, k)) == AT('t', (b, a, c), (i, k, j))

    tabij = AT('t', (a, b), (i, j))
    assert tabij.has(a)
    assert tabij.has(b)
    assert tabij.has(i)
    assert tabij.has(j)
    assert tabij.subs(b, c) == AT('t', (a, c), (i, j))
    assert (2*tabij).subs(i, c) == 2*AT('t', (a, b), (c, j))
    assert tabij.symbol == Symbol('t')
    assert latex(tabij) == '{t^{ab}_{ij}}'
    assert str(tabij) == 't((_a, _b),(_i, _j))'

    assert AT('t', (a, a), (i, j)).subs(a, b) == AT('t', (b, b), (i, j))
    assert AT('t', (a, i), (a, j)).subs(a, b) == AT('t', (b, i), (b, j))

    a1, a2, a3, a4 = symbols('alpha1:5')
    u_alpha1234 = AntiSymmetricTensor("u", (a1, a2), (a3, a4))

    assert latex(u_alpha1234) == r'{u^{\alpha_{1}\alpha_{2}}_{\alpha_{3}\alpha_{4}}}'
    assert str(u_alpha1234) == 'u((alpha1, alpha2),(alpha3, alpha4))'


def test_fully_contracted():
    i, j, k, l = symbols('i j k l', below_fermi=True)
    a, b, c, d = symbols('a b c d', above_fermi=True)
    p, q, r, s = symbols('p q r s', cls=Dummy)

    Fock = (AntiSymmetricTensor('f', (p,), (q,))*
            NO(Fd(p)*F(q)))
    V = (AntiSymmetricTensor('v', (p, q), (r, s))*
         NO(Fd(p)*Fd(q)*F(s)*F(r)))/4

    Fai = wicks(NO(Fd(i)*F(a))*Fock,
            keep_only_fully_contracted=True,
            simplify_kronecker_deltas=True)
    assert Fai == AntiSymmetricTensor('f', (a,), (i,))
    Vabij = wicks(NO(Fd(i)*Fd(j)*F(b)*F(a))*V,
            keep_only_fully_contracted=True,
            simplify_kronecker_deltas=True)
    assert Vabij == AntiSymmetricTensor('v', (a, b), (i, j))


def test_substitute_dummies_without_dummies():
    i, j = symbols('i,j')
    assert substitute_dummies(att(i, j) + 2) == att(i, j) + 2
    assert substitute_dummies(att(i, j) + 1) == att(i, j) + 1


def test_substitute_dummies_NO_operator():
    i, j = symbols('i j', cls=Dummy)
    assert substitute_dummies(att(i, j)*NO(Fd(i)*F(j))
                - att(j, i)*NO(Fd(j)*F(i))) == 0


def test_substitute_dummies_SQ_operator():
    i, j = symbols('i j', cls=Dummy)
    assert substitute_dummies(att(i, j)*Fd(i)*F(j)
                - att(j, i)*Fd(j)*F(i)) == 0


def test_substitute_dummies_new_indices():
    i, j = symbols('i j', below_fermi=True, cls=Dummy)
    a, b = symbols('a b', above_fermi=True, cls=Dummy)
    p, q = symbols('p q', cls=Dummy)
    f = Function('f')
    assert substitute_dummies(f(i, a, p) - f(j, b, q), new_indices=True) == 0


def test_substitute_dummies_substitution_order():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    f = Function('f')
    from sympy.utilities.iterables import variations
    for permut in variations([i, j, k, l], 4):
        assert substitute_dummies(f(*permut) - f(i, j, k, l)) == 0


def test_dummy_order_inner_outer_lines_VT1T1T1():
    ii = symbols('i', below_fermi=True)
    aa = symbols('a', above_fermi=True)
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    # Coupled-Cluster T1 terms with V*T1*T1*T1
    # t^{a}_{k} t^{c}_{i} t^{d}_{l} v^{lk}_{dc}
    exprs = [
        # permut v and t <=> swapping internal lines, equivalent
        # irrespective of symmetries in v
        v(k, l, c, d)*t(c, ii)*t(d, l)*t(aa, k),
        v(l, k, c, d)*t(c, ii)*t(d, k)*t(aa, l),
        v(k, l, d, c)*t(d, ii)*t(c, l)*t(aa, k),
        v(l, k, d, c)*t(d, ii)*t(c, k)*t(aa, l),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_dummy_order_inner_outer_lines_VT1T1T1T1():
    ii, jj = symbols('i j', below_fermi=True)
    aa, bb = symbols('a b', above_fermi=True)
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    # Coupled-Cluster T2 terms with V*T1*T1*T1*T1
    exprs = [
        # permut t <=> swapping external lines, not equivalent
        # except if v has certain symmetries.
        v(k, l, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
        v(k, l, c, d)*t(c, jj)*t(d, ii)*t(aa, k)*t(bb, l),
        v(k, l, c, d)*t(c, ii)*t(d, jj)*t(bb, k)*t(aa, l),
        v(k, l, c, d)*t(c, jj)*t(d, ii)*t(bb, k)*t(aa, l),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [
        # permut v <=> swapping external lines, not equivalent
        # except if v has certain symmetries.
        #
        # Note that in contrast to above, these permutations have identical
        # dummy order.  That is because the proximity to external indices
        # has higher influence on the canonical dummy ordering than the
        # position of a dummy on the factors.  In fact, the terms here are
        # similar in structure as the result of the dummy substitutions above.
        v(k, l, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
        v(l, k, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
        v(k, l, d, c)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
        v(l, k, d, c)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) == dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [
        # permut t and v <=> swapping internal lines, equivalent.
        # Canonical dummy order is different, and a consistent
        # substitution reveals the equivalence.
        v(k, l, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
        v(k, l, d, c)*t(c, jj)*t(d, ii)*t(aa, k)*t(bb, l),
        v(l, k, c, d)*t(c, ii)*t(d, jj)*t(bb, k)*t(aa, l),
        v(l, k, d, c)*t(c, jj)*t(d, ii)*t(bb, k)*t(aa, l),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_get_subNO():
    p, q, r = symbols('p,q,r')
    assert NO(F(p)*F(q)*F(r)).get_subNO(1) == NO(F(p)*F(r))
    assert NO(F(p)*F(q)*F(r)).get_subNO(0) == NO(F(q)*F(r))
    assert NO(F(p)*F(q)*F(r)).get_subNO(2) == NO(F(p)*F(q))


def test_equivalent_internal_lines_VT1T1():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    exprs = [  # permute v.  Different dummy order. Not equivalent.
        v(i, j, a, b)*t(a, i)*t(b, j),
        v(j, i, a, b)*t(a, i)*t(b, j),
        v(i, j, b, a)*t(a, i)*t(b, j),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [  # permute v.  Different dummy order. Equivalent
        v(i, j, a, b)*t(a, i)*t(b, j),
        v(j, i, b, a)*t(a, i)*t(b, j),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

    exprs = [  # permute t.  Same dummy order, not equivalent.
        v(i, j, a, b)*t(a, i)*t(b, j),
        v(i, j, a, b)*t(b, i)*t(a, j),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) == dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [  # permute v and t.  Different dummy order, equivalent
        v(i, j, a, b)*t(a, i)*t(b, j),
        v(j, i, a, b)*t(a, j)*t(b, i),
        v(i, j, b, a)*t(b, i)*t(a, j),
        v(j, i, b, a)*t(b, j)*t(a, i),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_equivalent_internal_lines_VT2conjT2():
    # this diagram requires special handling in TCE
    i, j, k, l, m, n = symbols('i j k l m n', below_fermi=True, cls=Dummy)
    a, b, c, d, e, f = symbols('a b c d e f', above_fermi=True, cls=Dummy)
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    # v(abcd)t(abij)t(ijcd)
    template = v(p1, p2, p3, p4)*t(p1, p2, i, j)*t(i, j, p3, p4)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = v(p1, p2, p3, p4)*t(p1, p2, j, i)*t(j, i, p3, p4)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)

    # v(abcd)t(abij)t(jicd)
    template = v(p1, p2, p3, p4)*t(p1, p2, i, j)*t(j, i, p3, p4)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = v(p1, p2, p3, p4)*t(p1, p2, j, i)*t(i, j, p3, p4)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)


def test_equivalent_internal_lines_VT2conjT2_ambiguous_order():
    # These diagrams invokes _determine_ambiguous() because the
    # dummies can not be ordered unambiguously by the key alone
    i, j, k, l, m, n = symbols('i j k l m n', below_fermi=True, cls=Dummy)
    a, b, c, d, e, f = symbols('a b c d e f', above_fermi=True, cls=Dummy)
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    # v(abcd)t(abij)t(cdij)
    template = v(p1, p2, p3, p4)*t(p1, p2, i, j)*t(p3, p4, i, j)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = v(p1, p2, p3, p4)*t(p1, p2, j, i)*t(p3, p4, i, j)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert dums(base) != dums(expr)
        assert substitute_dummies(expr) == substitute_dummies(base)


def test_equivalent_internal_lines_VT2():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies
    exprs = [
        # permute v. Same dummy order, not equivalent.
        #
        # This test show that the dummy order may not be sensitive to all
        # index permutations.  The following expressions have identical
        # structure as the resulting terms from of the dummy substitutions
        # in the test above.  Here, all expressions have the same dummy
        # order, so they cannot be simplified by means of dummy
        # substitution.  In order to simplify further, it is necessary to
        # exploit symmetries in the objects, for instance if t or v is
        # antisymmetric.
        v(i, j, a, b)*t(a, b, i, j),
        v(j, i, a, b)*t(a, b, i, j),
        v(i, j, b, a)*t(a, b, i, j),
        v(j, i, b, a)*t(a, b, i, j),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) == dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [
        # permute t.
        v(i, j, a, b)*t(a, b, i, j),
        v(i, j, a, b)*t(b, a, i, j),
        v(i, j, a, b)*t(a, b, j, i),
        v(i, j, a, b)*t(b, a, j, i),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [  # permute v and t.  Relabelling of dummies should be equivalent.
        v(i, j, a, b)*t(a, b, i, j),
        v(j, i, a, b)*t(a, b, j, i),
        v(i, j, b, a)*t(b, a, i, j),
        v(j, i, b, a)*t(b, a, j, i),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_internal_external_VT2T2():
    ii, jj = symbols('i j', below_fermi=True)
    aa, bb = symbols('a b', above_fermi=True)
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    exprs = [
        v(k, l, c, d)*t(aa, c, ii, k)*t(bb, d, jj, l),
        v(l, k, c, d)*t(aa, c, ii, l)*t(bb, d, jj, k),
        v(k, l, d, c)*t(aa, d, ii, k)*t(bb, c, jj, l),
        v(l, k, d, c)*t(aa, d, ii, l)*t(bb, c, jj, k),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
    exprs = [
        v(k, l, c, d)*t(aa, c, ii, k)*t(d, bb, jj, l),
        v(l, k, c, d)*t(aa, c, ii, l)*t(d, bb, jj, k),
        v(k, l, d, c)*t(aa, d, ii, k)*t(c, bb, jj, l),
        v(l, k, d, c)*t(aa, d, ii, l)*t(c, bb, jj, k),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
    exprs = [
        v(k, l, c, d)*t(c, aa, ii, k)*t(bb, d, jj, l),
        v(l, k, c, d)*t(c, aa, ii, l)*t(bb, d, jj, k),
        v(k, l, d, c)*t(d, aa, ii, k)*t(bb, c, jj, l),
        v(l, k, d, c)*t(d, aa, ii, l)*t(bb, c, jj, k),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_internal_external_pqrs():
    ii, jj = symbols('i j')
    aa, bb = symbols('a b')
    k, l = symbols('k l', cls=Dummy)
    c, d = symbols('c d', cls=Dummy)

    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    exprs = [
        v(k, l, c, d)*t(aa, c, ii, k)*t(bb, d, jj, l),
        v(l, k, c, d)*t(aa, c, ii, l)*t(bb, d, jj, k),
        v(k, l, d, c)*t(aa, d, ii, k)*t(bb, c, jj, l),
        v(l, k, d, c)*t(aa, d, ii, l)*t(bb, c, jj, k),
    ]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_dummy_order_well_defined():
    aa, bb = symbols('a b', above_fermi=True)
    k, l, m = symbols('k l m', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)
    p, q = symbols('p q', cls=Dummy)

    A = Function('A')
    B = Function('B')
    C = Function('C')
    dums = _get_ordered_dummies

    # We go through all key components in the order of increasing priority,
    # and consider only fully orderable expressions.  Non-orderable expressions
    # are tested elsewhere.

    # pos in first factor determines sort order
    assert dums(A(k, l)*B(l, k)) == [k, l]
    assert dums(A(l, k)*B(l, k)) == [l, k]
    assert dums(A(k, l)*B(k, l)) == [k, l]
    assert dums(A(l, k)*B(k, l)) == [l, k]

    # factors involving the index
    assert dums(A(k, l)*B(l, m)*C(k, m)) == [l, k, m]
    assert dums(A(k, l)*B(l, m)*C(m, k)) == [l, k, m]
    assert dums(A(l, k)*B(l, m)*C(k, m)) == [l, k, m]
    assert dums(A(l, k)*B(l, m)*C(m, k)) == [l, k, m]
    assert dums(A(k, l)*B(m, l)*C(k, m)) == [l, k, m]
    assert dums(A(k, l)*B(m, l)*C(m, k)) == [l, k, m]
    assert dums(A(l, k)*B(m, l)*C(k, m)) == [l, k, m]
    assert dums(A(l, k)*B(m, l)*C(m, k)) == [l, k, m]

    # same, but with factor order determined by non-dummies
    assert dums(A(k, aa, l)*A(l, bb, m)*A(bb, k, m)) == [l, k, m]
    assert dums(A(k, aa, l)*A(l, bb, m)*A(bb, m, k)) == [l, k, m]
    assert dums(A(k, aa, l)*A(m, bb, l)*A(bb, k, m)) == [l, k, m]
    assert dums(A(k, aa, l)*A(m, bb, l)*A(bb, m, k)) == [l, k, m]
    assert dums(A(l, aa, k)*A(l, bb, m)*A(bb, k, m)) == [l, k, m]
    assert dums(A(l, aa, k)*A(l, bb, m)*A(bb, m, k)) == [l, k, m]
    assert dums(A(l, aa, k)*A(m, bb, l)*A(bb, k, m)) == [l, k, m]
    assert dums(A(l, aa, k)*A(m, bb, l)*A(bb, m, k)) == [l, k, m]

    # index range
    assert dums(A(p, c, k)*B(p, c, k)) == [k, c, p]
    assert dums(A(p, k, c)*B(p, c, k)) == [k, c, p]
    assert dums(A(c, k, p)*B(p, c, k)) == [k, c, p]
    assert dums(A(c, p, k)*B(p, c, k)) == [k, c, p]
    assert dums(A(k, c, p)*B(p, c, k)) == [k, c, p]
    assert dums(A(k, p, c)*B(p, c, k)) == [k, c, p]
    assert dums(B(p, c, k)*A(p, c, k)) == [k, c, p]
    assert dums(B(p, k, c)*A(p, c, k)) == [k, c, p]
    assert dums(B(c, k, p)*A(p, c, k)) == [k, c, p]
    assert dums(B(c, p, k)*A(p, c, k)) == [k, c, p]
    assert dums(B(k, c, p)*A(p, c, k)) == [k, c, p]
    assert dums(B(k, p, c)*A(p, c, k)) == [k, c, p]


def test_dummy_order_ambiguous():
    aa, bb = symbols('a b', above_fermi=True)
    i, j, k, l, m = symbols('i j k l m', below_fermi=True, cls=Dummy)
    a, b, c, d, e = symbols('a b c d e', above_fermi=True, cls=Dummy)
    p, q = symbols('p q', cls=Dummy)
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    p5, p6, p7, p8 = symbols('p5 p6 p7 p8', above_fermi=True, cls=Dummy)
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)
    h5, h6, h7, h8 = symbols('h5 h6 h7 h8', below_fermi=True, cls=Dummy)

    A = Function('A')
    B = Function('B')

    from sympy.utilities.iterables import variations

    # A*A*A*A*B  --  ordering of p5 and p4 is used to figure out the rest
    template = A(p1, p2)*A(p4, p1)*A(p2, p3)*A(p3, p5)*B(p5, p4)
    permutator = variations([a, b, c, d, e], 5)
    base = template.subs(zip([p1, p2, p3, p4, p5], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4, p5], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

    # A*A*A*A*A  --  an arbitrary index is assigned and the rest are figured out
    template = A(p1, p2)*A(p4, p1)*A(p2, p3)*A(p3, p5)*A(p5, p4)
    permutator = variations([a, b, c, d, e], 5)
    base = template.subs(zip([p1, p2, p3, p4, p5], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4, p5], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

    # A*A*A  --  ordering of p5 and p4 is used to figure out the rest
    template = A(p1, p2, p4, p1)*A(p2, p3, p3, p5)*A(p5, p4)
    permutator = variations([a, b, c, d, e], 5)
    base = template.subs(zip([p1, p2, p3, p4, p5], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4, p5], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)


def atv(*args):
    return AntiSymmetricTensor('v', args[:2], args[2:] )


def att(*args):
    if len(args) == 4:
        return AntiSymmetricTensor('t', args[:2], args[2:] )
    elif len(args) == 2:
        return AntiSymmetricTensor('t', (args[0],), (args[1],))


def test_dummy_order_inner_outer_lines_VT1T1T1_AT():
    ii = symbols('i', below_fermi=True)
    aa = symbols('a', above_fermi=True)
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    # Coupled-Cluster T1 terms with V*T1*T1*T1
    # t^{a}_{k} t^{c}_{i} t^{d}_{l} v^{lk}_{dc}
    exprs = [
        # permut v and t <=> swapping internal lines, equivalent
        # irrespective of symmetries in v
        atv(k, l, c, d)*att(c, ii)*att(d, l)*att(aa, k),
        atv(l, k, c, d)*att(c, ii)*att(d, k)*att(aa, l),
        atv(k, l, d, c)*att(d, ii)*att(c, l)*att(aa, k),
        atv(l, k, d, c)*att(d, ii)*att(c, k)*att(aa, l),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_dummy_order_inner_outer_lines_VT1T1T1T1_AT():
    ii, jj = symbols('i j', below_fermi=True)
    aa, bb = symbols('a b', above_fermi=True)
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    # Coupled-Cluster T2 terms with V*T1*T1*T1*T1
    # non-equivalent substitutions (change of sign)
    exprs = [
        # permut t <=> swapping external lines
        atv(k, l, c, d)*att(c, ii)*att(d, jj)*att(aa, k)*att(bb, l),
        atv(k, l, c, d)*att(c, jj)*att(d, ii)*att(aa, k)*att(bb, l),
        atv(k, l, c, d)*att(c, ii)*att(d, jj)*att(bb, k)*att(aa, l),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == -substitute_dummies(permut)

    # equivalent substitutions
    exprs = [
        atv(k, l, c, d)*att(c, ii)*att(d, jj)*att(aa, k)*att(bb, l),
        # permut t <=> swapping external lines
        atv(k, l, c, d)*att(c, jj)*att(d, ii)*att(bb, k)*att(aa, l),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_equivalent_internal_lines_VT1T1_AT():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)

    exprs = [  # permute v.  Different dummy order. Not equivalent.
        atv(i, j, a, b)*att(a, i)*att(b, j),
        atv(j, i, a, b)*att(a, i)*att(b, j),
        atv(i, j, b, a)*att(a, i)*att(b, j),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [  # permute v.  Different dummy order. Equivalent
        atv(i, j, a, b)*att(a, i)*att(b, j),
        atv(j, i, b, a)*att(a, i)*att(b, j),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

    exprs = [  # permute t.  Same dummy order, not equivalent.
        atv(i, j, a, b)*att(a, i)*att(b, j),
        atv(i, j, a, b)*att(b, i)*att(a, j),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [  # permute v and t.  Different dummy order, equivalent
        atv(i, j, a, b)*att(a, i)*att(b, j),
        atv(j, i, a, b)*att(a, j)*att(b, i),
        atv(i, j, b, a)*att(b, i)*att(a, j),
        atv(j, i, b, a)*att(b, j)*att(a, i),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_equivalent_internal_lines_VT2conjT2_AT():
    # this diagram requires special handling in TCE
    i, j, k, l, m, n = symbols('i j k l m n', below_fermi=True, cls=Dummy)
    a, b, c, d, e, f = symbols('a b c d e f', above_fermi=True, cls=Dummy)
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations

    # atv(abcd)att(abij)att(ijcd)
    template = atv(p1, p2, p3, p4)*att(p1, p2, i, j)*att(i, j, p3, p4)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = atv(p1, p2, p3, p4)*att(p1, p2, j, i)*att(j, i, p3, p4)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

    # atv(abcd)att(abij)att(jicd)
    template = atv(p1, p2, p3, p4)*att(p1, p2, i, j)*att(j, i, p3, p4)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = atv(p1, p2, p3, p4)*att(p1, p2, j, i)*att(i, j, p3, p4)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)


def test_equivalent_internal_lines_VT2conjT2_ambiguous_order_AT():
    # These diagrams invokes _determine_ambiguous() because the
    # dummies can not be ordered unambiguously by the key alone
    i, j, k, l, m, n = symbols('i j k l m n', below_fermi=True, cls=Dummy)
    a, b, c, d, e, f = symbols('a b c d e f', above_fermi=True, cls=Dummy)
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations

    # atv(abcd)att(abij)att(cdij)
    template = atv(p1, p2, p3, p4)*att(p1, p2, i, j)*att(p3, p4, i, j)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)
    template = atv(p1, p2, p3, p4)*att(p1, p2, j, i)*att(p3, p4, i, j)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)


def test_equivalent_internal_lines_VT2_AT():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)

    exprs = [
        # permute v. Same dummy order, not equivalent.
        atv(i, j, a, b)*att(a, b, i, j),
        atv(j, i, a, b)*att(a, b, i, j),
        atv(i, j, b, a)*att(a, b, i, j),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [
        # permute t.
        atv(i, j, a, b)*att(a, b, i, j),
        atv(i, j, a, b)*att(b, a, i, j),
        atv(i, j, a, b)*att(a, b, j, i),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    exprs = [  # permute v and t.  Relabelling of dummies should be equivalent.
        atv(i, j, a, b)*att(a, b, i, j),
        atv(j, i, a, b)*att(a, b, j, i),
        atv(i, j, b, a)*att(b, a, i, j),
        atv(j, i, b, a)*att(b, a, j, i),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_internal_external_VT2T2_AT():
    ii, jj = symbols('i j', below_fermi=True)
    aa, bb = symbols('a b', above_fermi=True)
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    exprs = [
        atv(k, l, c, d)*att(aa, c, ii, k)*att(bb, d, jj, l),
        atv(l, k, c, d)*att(aa, c, ii, l)*att(bb, d, jj, k),
        atv(k, l, d, c)*att(aa, d, ii, k)*att(bb, c, jj, l),
        atv(l, k, d, c)*att(aa, d, ii, l)*att(bb, c, jj, k),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
    exprs = [
        atv(k, l, c, d)*att(aa, c, ii, k)*att(d, bb, jj, l),
        atv(l, k, c, d)*att(aa, c, ii, l)*att(d, bb, jj, k),
        atv(k, l, d, c)*att(aa, d, ii, k)*att(c, bb, jj, l),
        atv(l, k, d, c)*att(aa, d, ii, l)*att(c, bb, jj, k),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
    exprs = [
        atv(k, l, c, d)*att(c, aa, ii, k)*att(bb, d, jj, l),
        atv(l, k, c, d)*att(c, aa, ii, l)*att(bb, d, jj, k),
        atv(k, l, d, c)*att(d, aa, ii, k)*att(bb, c, jj, l),
        atv(l, k, d, c)*att(d, aa, ii, l)*att(bb, c, jj, k),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_internal_external_pqrs_AT():
    ii, jj = symbols('i j')
    aa, bb = symbols('a b')
    k, l = symbols('k l', cls=Dummy)
    c, d = symbols('c d', cls=Dummy)

    exprs = [
        atv(k, l, c, d)*att(aa, c, ii, k)*att(bb, d, jj, l),
        atv(l, k, c, d)*att(aa, c, ii, l)*att(bb, d, jj, k),
        atv(k, l, d, c)*att(aa, d, ii, k)*att(bb, c, jj, l),
        atv(l, k, d, c)*att(aa, d, ii, l)*att(bb, c, jj, k),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_issue_19661():
    a = Symbol('0')
    assert latex(Commutator(Bd(a)**2, B(a))
                 ) == '- \\left[b_{0},{b^\\dagger_{0}}^{2}\\right]'


def test_canonical_ordering_AntiSymmetricTensor():
    v = symbols("v")

    c, d = symbols(('c','d'), above_fermi=True,
                                   cls=Dummy)
    k, l = symbols(('k','l'), below_fermi=True,
                                   cls=Dummy)

    # formerly, the left gave either the left or the right
    assert AntiSymmetricTensor(v, (k, l), (d, c)
        ) == -AntiSymmetricTensor(v, (l, k), (d, c))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dotenv\ipython.py ===
from IPython.core.magic import Magics, line_magic, magics_class  # type: ignore
from IPython.core.magic_arguments import (argument, magic_arguments,  # type: ignore
                                          parse_argstring)  # type: ignore

from .main import find_dotenv, load_dotenv


@magics_class
class IPythonDotEnv(Magics):

    @magic_arguments()
    @argument(
        '-o', '--override', action='store_true',
        help="Indicate to override existing variables"
    )
    @argument(
        '-v', '--verbose', action='store_true',
        help="Indicate function calls to be verbose"
    )
    @argument('dotenv_path', nargs='?', type=str, default='.env',
              help='Search in increasingly higher folders for the `dotenv_path`')
    @line_magic
    def dotenv(self, line):
        args = parse_argstring(self.dotenv, line)
        # Locate the .env file
        dotenv_path = args.dotenv_path
        try:
            dotenv_path = find_dotenv(dotenv_path, True, True)
        except IOError:
            print("cannot find .env file")
            return

        # Load the .env file
        load_dotenv(dotenv_path, verbose=args.verbose, override=args.override)


def load_ipython_extension(ipython):
    """Register the %dotenv magic."""
    ipython.register_magics(IPythonDotEnv)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_sqlalchemy\table.py ===
from __future__ import annotations

import typing as t

import sqlalchemy as sa
import sqlalchemy.sql.schema as sa_sql_schema


class _Table(sa.Table):
    @t.overload
    def __init__(
        self,
        name: str,
        *args: sa_sql_schema.SchemaItem,
        bind_key: str | None = None,
        **kwargs: t.Any,
    ) -> None:
        ...

    @t.overload
    def __init__(
        self,
        name: str,
        metadata: sa.MetaData,
        *args: sa_sql_schema.SchemaItem,
        **kwargs: t.Any,
    ) -> None:
        ...

    @t.overload
    def __init__(
        self, name: str, *args: sa_sql_schema.SchemaItem, **kwargs: t.Any
    ) -> None:
        ...

    def __init__(
        self, name: str, *args: sa_sql_schema.SchemaItem, **kwargs: t.Any
    ) -> None:
        super().__init__(name, *args, **kwargs)  # type: ignore[arg-type]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_configtool.py ===
import argparse
import sys
from pathlib import Path

from .lib._utils_impl import get_include
from .version import __version__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the version and exit.",
    )
    parser.add_argument(
        "--cflags",
        action="store_true",
        help="Compile flag needed when using the NumPy headers.",
    )
    parser.add_argument(
        "--pkgconfigdir",
        action="store_true",
        help=("Print the pkgconfig directory in which `numpy.pc` is stored "
              "(useful for setting $PKG_CONFIG_PATH)."),
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.cflags:
        print("-I" + get_include())
    if args.pkgconfigdir:
        _path = Path(get_include()) / '..' / 'lib' / 'pkgconfig'
        print(_path.resolve())


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\pydy-example-repo\double_pendulum.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

q1, q2, u1, u2 = _me.dynamicsymbols('q1 q2 u1 u2')
q1_d, q2_d, u1_d, u2_d = _me.dynamicsymbols('q1_ q2_ u1_ u2_', 1)
l, m, g = _sm.symbols('l m g', real=True)
frame_n = _me.ReferenceFrame('n')
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_a.orient(frame_n, 'Axis', [q1, frame_n.z])
frame_b.orient(frame_n, 'Axis', [q2, frame_n.z])
frame_a.set_ang_vel(frame_n, u1*frame_n.z)
frame_b.set_ang_vel(frame_n, u2*frame_n.z)
point_o = _me.Point('o')
particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))
particle_r = _me.Particle('r', _me.Point('r_pt'), _sm.Symbol('m'))
particle_p.point.set_pos(point_o, l*frame_a.x)
particle_r.point.set_pos(particle_p.point, l*frame_b.x)
point_o.set_vel(frame_n, 0)
particle_p.point.v2pt_theory(point_o,frame_n,frame_a)
particle_r.point.v2pt_theory(particle_p.point,frame_n,frame_b)
particle_p.mass = m
particle_r.mass = m
force_p = particle_p.mass*(g*frame_n.x)
force_r = particle_r.mass*(g*frame_n.x)
kd_eqs = [q1_d - u1, q2_d - u2]
forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x)), (particle_r.point,particle_r.mass*(g*frame_n.x))]
kane = _me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u1, u2], kd_eqs = kd_eqs)
fr, frstar = kane.kanes_equations([particle_p, particle_r], forceList)
zero = fr+frstar
from pydy.system import System
sys = System(kane, constants = {l:1, m:1, g:9.81},
specifieds={},
initial_conditions={q1:.1, q2:.2, u1:0, u2:0},
times = _np.linspace(0.0, 10, 10/.01))

y=sys.integrate()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\method.py ===
from abc import ABC, abstractmethod

class _Methods(ABC):
    """Abstract Base Class for all methods."""

    @abstractmethod
    def q(self):
        pass

    @abstractmethod
    def u(self):
        pass

    @abstractmethod
    def bodies(self):
        pass

    @abstractmethod
    def loads(self):
        pass

    @abstractmethod
    def mass_matrix(self):
        pass

    @abstractmethod
    def forcing(self):
        pass

    @abstractmethod
    def mass_matrix_full(self):
        pass

    @abstractmethod
    def forcing_full(self):
        pass

    def _form_eoms(self):
        raise NotImplementedError("Subclasses must implement this.")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\testing\__init__.py ===
# Uses `from x import y as y` for compatibility with `pyright --verifytypes` (#2625)

from .._core import (
    MockClock as MockClock,
    wait_all_tasks_blocked as wait_all_tasks_blocked,
)
from .._threads import (
    active_thread_count as active_thread_count,
    wait_all_threads_completed as wait_all_threads_completed,
)
from .._util import fixup_module_metadata
from ._check_streams import (
    check_half_closeable_stream as check_half_closeable_stream,
    check_one_way_stream as check_one_way_stream,
    check_two_way_stream as check_two_way_stream,
)
from ._checkpoints import (
    assert_checkpoints as assert_checkpoints,
    assert_no_checkpoints as assert_no_checkpoints,
)
from ._memory_streams import (
    MemoryReceiveStream as MemoryReceiveStream,
    MemorySendStream as MemorySendStream,
    lockstep_stream_one_way_pair as lockstep_stream_one_way_pair,
    lockstep_stream_pair as lockstep_stream_pair,
    memory_stream_one_way_pair as memory_stream_one_way_pair,
    memory_stream_pair as memory_stream_pair,
    memory_stream_pump as memory_stream_pump,
)
from ._network import open_stream_to_socket_listener as open_stream_to_socket_listener
from ._raises_group import Matcher as Matcher, RaisesGroup as RaisesGroup
from ._sequencer import Sequencer as Sequencer
from ._trio_test import trio_test as trio_test

################################################################


fixup_module_metadata(__name__, globals())
del fixup_module_metadata

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_nanops.py ===
from functools import partial

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
from pandas import (
    Series,
    isna,
)
import pandas._testing as tm
from pandas.core import nanops

use_bn = nanops._USE_BOTTLENECK


@pytest.fixture
def disable_bottleneck(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(nanops, "_USE_BOTTLENECK", False)
        yield


@pytest.fixture
def arr_shape():
    return 11, 7


@pytest.fixture
def arr_float(arr_shape):
    return np.random.default_rng(2).standard_normal(arr_shape)


@pytest.fixture
def arr_complex(arr_float):
    return arr_float + arr_float * 1j


@pytest.fixture
def arr_int(arr_shape):
    return np.random.default_rng(2).integers(-10, 10, arr_shape)


@pytest.fixture
def arr_bool(arr_shape):
    return np.random.default_rng(2).integers(0, 2, arr_shape) == 0


@pytest.fixture
def arr_str(arr_float):
    return np.abs(arr_float).astype("S")


@pytest.fixture
def arr_utf(arr_float):
    return np.abs(arr_float).astype("U")


@pytest.fixture
def arr_date(arr_shape):
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype("M8[ns]")


@pytest.fixture
def arr_tdelta(arr_shape):
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype("m8[ns]")


@pytest.fixture
def arr_nan(arr_shape):
    return np.tile(np.nan, arr_shape)


@pytest.fixture
def arr_float_nan(arr_float, arr_nan):
    return np.vstack([arr_float, arr_nan])


@pytest.fixture
def arr_nan_float1(arr_nan, arr_float):
    return np.vstack([arr_nan, arr_float])


@pytest.fixture
def arr_nan_nan(arr_nan):
    return np.vstack([arr_nan, arr_nan])


@pytest.fixture
def arr_inf(arr_float):
    return arr_float * np.inf


@pytest.fixture
def arr_float_inf(arr_float, arr_inf):
    return np.vstack([arr_float, arr_inf])


@pytest.fixture
def arr_nan_inf(arr_nan, arr_inf):
    return np.vstack([arr_nan, arr_inf])


@pytest.fixture
def arr_float_nan_inf(arr_float, arr_nan, arr_inf):
    return np.vstack([arr_float, arr_nan, arr_inf])


@pytest.fixture
def arr_nan_nan_inf(arr_nan, arr_inf):
    return np.vstack([arr_nan, arr_nan, arr_inf])


@pytest.fixture
def arr_obj(
    arr_float, arr_int, arr_bool, arr_complex, arr_str, arr_utf, arr_date, arr_tdelta
):
    return np.vstack(
        [
            arr_float.astype("O"),
            arr_int.astype("O"),
            arr_bool.astype("O"),
            arr_complex.astype("O"),
            arr_str.astype("O"),
            arr_utf.astype("O"),
            arr_date.astype("O"),
            arr_tdelta.astype("O"),
        ]
    )


@pytest.fixture
def arr_nan_nanj(arr_nan):
    with np.errstate(invalid="ignore"):
        return arr_nan + arr_nan * 1j


@pytest.fixture
def arr_complex_nan(arr_complex, arr_nan_nanj):
    with np.errstate(invalid="ignore"):
        return np.vstack([arr_complex, arr_nan_nanj])


@pytest.fixture
def arr_nan_infj(arr_inf):
    with np.errstate(invalid="ignore"):
        return arr_inf * 1j


@pytest.fixture
def arr_complex_nan_infj(arr_complex, arr_nan_infj):
    with np.errstate(invalid="ignore"):
        return np.vstack([arr_complex, arr_nan_infj])


@pytest.fixture
def arr_float_1d(arr_float):
    return arr_float[:, 0]


@pytest.fixture
def arr_nan_1d(arr_nan):
    return arr_nan[:, 0]


@pytest.fixture
def arr_float_nan_1d(arr_float_nan):
    return arr_float_nan[:, 0]


@pytest.fixture
def arr_float1_nan_1d(arr_float1_nan):
    return arr_float1_nan[:, 0]


@pytest.fixture
def arr_nan_float1_1d(arr_nan_float1):
    return arr_nan_float1[:, 0]


class TestnanopsDataFrame:
    def setup_method(self):
        nanops._USE_BOTTLENECK = False

        arr_shape = (11, 7)

        self.arr_float = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_float1 = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_complex = self.arr_float + self.arr_float1 * 1j
        self.arr_int = np.random.default_rng(2).integers(-10, 10, arr_shape)
        self.arr_bool = np.random.default_rng(2).integers(0, 2, arr_shape) == 0
        self.arr_str = np.abs(self.arr_float).astype("S")
        self.arr_utf = np.abs(self.arr_float).astype("U")
        self.arr_date = (
            np.random.default_rng(2).integers(0, 20000, arr_shape).astype("M8[ns]")
        )
        self.arr_tdelta = (
            np.random.default_rng(2).integers(0, 20000, arr_shape).astype("m8[ns]")
        )

        self.arr_nan = np.tile(np.nan, arr_shape)
        self.arr_float_nan = np.vstack([self.arr_float, self.arr_nan])
        self.arr_float1_nan = np.vstack([self.arr_float1, self.arr_nan])
        self.arr_nan_float1 = np.vstack([self.arr_nan, self.arr_float1])
        self.arr_nan_nan = np.vstack([self.arr_nan, self.arr_nan])

        self.arr_inf = self.arr_float * np.inf
        self.arr_float_inf = np.vstack([self.arr_float, self.arr_inf])

        self.arr_nan_inf = np.vstack([self.arr_nan, self.arr_inf])
        self.arr_float_nan_inf = np.vstack([self.arr_float, self.arr_nan, self.arr_inf])
        self.arr_nan_nan_inf = np.vstack([self.arr_nan, self.arr_nan, self.arr_inf])
        self.arr_obj = np.vstack(
            [
                self.arr_float.astype("O"),
                self.arr_int.astype("O"),
                self.arr_bool.astype("O"),
                self.arr_complex.astype("O"),
                self.arr_str.astype("O"),
                self.arr_utf.astype("O"),
                self.arr_date.astype("O"),
                self.arr_tdelta.astype("O"),
            ]
        )

        with np.errstate(invalid="ignore"):
            self.arr_nan_nanj = self.arr_nan + self.arr_nan * 1j
            self.arr_complex_nan = np.vstack([self.arr_complex, self.arr_nan_nanj])

            self.arr_nan_infj = self.arr_inf * 1j
            self.arr_complex_nan_infj = np.vstack([self.arr_complex, self.arr_nan_infj])

        self.arr_float_2d = self.arr_float
        self.arr_float1_2d = self.arr_float1

        self.arr_nan_2d = self.arr_nan
        self.arr_float_nan_2d = self.arr_float_nan
        self.arr_float1_nan_2d = self.arr_float1_nan
        self.arr_nan_float1_2d = self.arr_nan_float1

        self.arr_float_1d = self.arr_float[:, 0]
        self.arr_float1_1d = self.arr_float1[:, 0]

        self.arr_nan_1d = self.arr_nan[:, 0]
        self.arr_float_nan_1d = self.arr_float_nan[:, 0]
        self.arr_float1_nan_1d = self.arr_float1_nan[:, 0]
        self.arr_nan_float1_1d = self.arr_nan_float1[:, 0]

    def teardown_method(self):
        nanops._USE_BOTTLENECK = use_bn

    def check_results(self, targ, res, axis, check_dtype=True):
        res = getattr(res, "asm8", res)

        if (
            axis != 0
            and hasattr(targ, "shape")
            and targ.ndim
            and targ.shape != res.shape
        ):
            res = np.split(res, [targ.shape[0]], axis=0)[0]

        try:
            tm.assert_almost_equal(targ, res, check_dtype=check_dtype)
        except AssertionError:
            # handle timedelta dtypes
            if hasattr(targ, "dtype") and targ.dtype == "m8[ns]":
                raise

            # There are sometimes rounding errors with
            # complex and object dtypes.
            # If it isn't one of those, re-raise the error.
            if not hasattr(res, "dtype") or res.dtype.kind not in ["c", "O"]:
                raise
            # convert object dtypes to something that can be split into
            # real and imaginary parts
            if res.dtype.kind == "O":
                if targ.dtype.kind != "O":
                    res = res.astype(targ.dtype)
                else:
                    cast_dtype = "c16" if hasattr(np, "complex128") else "f8"
                    res = res.astype(cast_dtype)
                    targ = targ.astype(cast_dtype)
            # there should never be a case where numpy returns an object
            # but nanops doesn't, so make that an exception
            elif targ.dtype.kind == "O":
                raise
            tm.assert_almost_equal(np.real(targ), np.real(res), check_dtype=check_dtype)
            tm.assert_almost_equal(np.imag(targ), np.imag(res), check_dtype=check_dtype)

    def check_fun_data(
        self,
        testfunc,
        targfunc,
        testarval,
        targarval,
        skipna,
        check_dtype=True,
        empty_targfunc=None,
        **kwargs,
    ):
        for axis in list(range(targarval.ndim)) + [None]:
            targartempval = targarval if skipna else testarval
            if skipna and empty_targfunc and isna(targartempval).all():
                targ = empty_targfunc(targartempval, axis=axis, **kwargs)
            else:
                targ = targfunc(targartempval, axis=axis, **kwargs)

            if targartempval.dtype == object and (
                targfunc is np.any or targfunc is np.all
            ):
                # GH#12863 the numpy functions will retain e.g. floatiness
                if isinstance(targ, np.ndarray):
                    targ = targ.astype(bool)
                else:
                    targ = bool(targ)

            res = testfunc(testarval, axis=axis, skipna=skipna, **kwargs)

            if (
                isinstance(targ, np.complex128)
                and isinstance(res, float)
                and np.isnan(targ)
                and np.isnan(res)
            ):
                # GH#18463
                targ = res

            self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna:
                res = testfunc(testarval, axis=axis, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if axis is None:
                res = testfunc(testarval, skipna=skipna, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna and axis is None:
                res = testfunc(testarval, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)

        if testarval.ndim <= 1:
            return

        # Recurse on lower-dimension
        testarval2 = np.take(testarval, 0, axis=-1)
        targarval2 = np.take(targarval, 0, axis=-1)
        self.check_fun_data(
            testfunc,
            targfunc,
            testarval2,
            targarval2,
            skipna=skipna,
            check_dtype=check_dtype,
            empty_targfunc=empty_targfunc,
            **kwargs,
        )

    def check_fun(
        self, testfunc, targfunc, testar, skipna, empty_targfunc=None, **kwargs
    ):
        targar = testar
        if testar.endswith("_nan") and hasattr(self, testar[:-4]):
            targar = testar[:-4]

        testarval = getattr(self, testar)
        targarval = getattr(self, targar)
        self.check_fun_data(
            testfunc,
            targfunc,
            testarval,
            targarval,
            skipna=skipna,
            empty_targfunc=empty_targfunc,
            **kwargs,
        )

    def check_funs(
        self,
        testfunc,
        targfunc,
        skipna,
        allow_complex=True,
        allow_all_nan=True,
        allow_date=True,
        allow_tdelta=True,
        allow_obj=True,
        **kwargs,
    ):
        self.check_fun(testfunc, targfunc, "arr_float", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_float_nan", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_int", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_bool", skipna, **kwargs)
        objs = [
            self.arr_float.astype("O"),
            self.arr_int.astype("O"),
            self.arr_bool.astype("O"),
        ]

        if allow_all_nan:
            self.check_fun(testfunc, targfunc, "arr_nan", skipna, **kwargs)

        if allow_complex:
            self.check_fun(testfunc, targfunc, "arr_complex", skipna, **kwargs)
            self.check_fun(testfunc, targfunc, "arr_complex_nan", skipna, **kwargs)
            if allow_all_nan:
                self.check_fun(testfunc, targfunc, "arr_nan_nanj", skipna, **kwargs)
            objs += [self.arr_complex.astype("O")]

        if allow_date:
            targfunc(self.arr_date)
            self.check_fun(testfunc, targfunc, "arr_date", skipna, **kwargs)
            objs += [self.arr_date.astype("O")]

        if allow_tdelta:
            try:
                targfunc(self.arr_tdelta)
            except TypeError:
                pass
            else:
                self.check_fun(testfunc, targfunc, "arr_tdelta", skipna, **kwargs)
                objs += [self.arr_tdelta.astype("O")]

        if allow_obj:
            self.arr_obj = np.vstack(objs)
            # some nanops handle object dtypes better than their numpy
            # counterparts, so the numpy functions need to be given something
            # else
            if allow_obj == "convert":
                targfunc = partial(
                    self._badobj_wrap, func=targfunc, allow_complex=allow_complex
                )
            self.check_fun(testfunc, targfunc, "arr_obj", skipna, **kwargs)

    def _badobj_wrap(self, value, func, allow_complex=True, **kwargs):
        if value.dtype.kind == "O":
            if allow_complex:
                value = value.astype("c16")
            else:
                value = value.astype("f8")
        return func(value, **kwargs)

    @pytest.mark.parametrize(
        "nan_op,np_op", [(nanops.nanany, np.any), (nanops.nanall, np.all)]
    )
    def test_nan_funcs(self, nan_op, np_op, skipna):
        self.check_funs(nan_op, np_op, skipna, allow_all_nan=False, allow_date=False)

    def test_nansum(self, skipna):
        self.check_funs(
            nanops.nansum,
            np.sum,
            skipna,
            allow_date=False,
            check_dtype=False,
            empty_targfunc=np.nansum,
        )

    def test_nanmean(self, skipna):
        self.check_funs(
            nanops.nanmean, np.mean, skipna, allow_obj=False, allow_date=False
        )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanmedian(self, skipna):
        self.check_funs(
            nanops.nanmedian,
            np.median,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
        )

    @pytest.mark.parametrize("ddof", range(3))
    def test_nanvar(self, ddof, skipna):
        self.check_funs(
            nanops.nanvar,
            np.var,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
            ddof=ddof,
        )

    @pytest.mark.parametrize("ddof", range(3))
    def test_nanstd(self, ddof, skipna):
        self.check_funs(
            nanops.nanstd,
            np.std,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
            ddof=ddof,
        )

    @pytest.mark.parametrize("ddof", range(3))
    def test_nansem(self, ddof, skipna):
        sp_stats = pytest.importorskip("scipy.stats")

        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nansem,
                sp_stats.sem,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
                allow_obj="convert",
                ddof=ddof,
            )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize(
        "nan_op,np_op", [(nanops.nanmin, np.min), (nanops.nanmax, np.max)]
    )
    def test_nanops_with_warnings(self, nan_op, np_op, skipna):
        self.check_funs(nan_op, np_op, skipna, allow_obj=False)

    def _argminmax_wrap(self, value, axis=None, func=None):
        res = func(value, axis)
        nans = np.min(value, axis)
        nullnan = isna(nans)
        if res.ndim:
            res[nullnan] = -1
        elif (
            hasattr(nullnan, "all")
            and nullnan.all()
            or not hasattr(nullnan, "all")
            and nullnan
        ):
            res = -1
        return res

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanargmax(self, skipna):
        func = partial(self._argminmax_wrap, func=np.argmax)
        self.check_funs(nanops.nanargmax, func, skipna, allow_obj=False)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanargmin(self, skipna):
        func = partial(self._argminmax_wrap, func=np.argmin)
        self.check_funs(nanops.nanargmin, func, skipna, allow_obj=False)

    def _skew_kurt_wrap(self, values, axis=None, func=None):
        if not isinstance(values.dtype.type, np.floating):
            values = values.astype("f8")
        result = func(values, axis=axis, bias=False)
        # fix for handling cases where all elements in an axis are the same
        if isinstance(result, np.ndarray):
            result[np.max(values, axis=axis) == np.min(values, axis=axis)] = 0
            return result
        elif np.max(values) == np.min(values):
            return 0.0
        return result

    def test_nanskew(self, skipna):
        sp_stats = pytest.importorskip("scipy.stats")

        func = partial(self._skew_kurt_wrap, func=sp_stats.skew)
        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nanskew,
                func,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
            )

    def test_nankurt(self, skipna):
        sp_stats = pytest.importorskip("scipy.stats")

        func1 = partial(sp_stats.kurtosis, fisher=True)
        func = partial(self._skew_kurt_wrap, func=func1)
        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nankurt,
                func,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
            )

    def test_nanprod(self, skipna):
        self.check_funs(
            nanops.nanprod,
            np.prod,
            skipna,
            allow_date=False,
            allow_tdelta=False,
            empty_targfunc=np.nanprod,
        )

    def check_nancorr_nancov_2d(self, checkfun, targ0, targ1, **kwargs):
        res00 = checkfun(self.arr_float_2d, self.arr_float1_2d, **kwargs)
        res01 = checkfun(
            self.arr_float_2d,
            self.arr_float1_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)

        res10 = checkfun(self.arr_float_nan_2d, self.arr_float1_nan_2d, **kwargs)
        res11 = checkfun(
            self.arr_float_nan_2d,
            self.arr_float1_nan_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)

        targ2 = np.nan
        res20 = checkfun(self.arr_nan_2d, self.arr_float1_2d, **kwargs)
        res21 = checkfun(self.arr_float_2d, self.arr_nan_2d, **kwargs)
        res22 = checkfun(self.arr_nan_2d, self.arr_nan_2d, **kwargs)
        res23 = checkfun(self.arr_float_nan_2d, self.arr_nan_float1_2d, **kwargs)
        res24 = checkfun(
            self.arr_float_nan_2d,
            self.arr_nan_float1_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        res25 = checkfun(
            self.arr_float_2d,
            self.arr_float1_2d,
            min_periods=len(self.arr_float_2d) + 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def check_nancorr_nancov_1d(self, checkfun, targ0, targ1, **kwargs):
        res00 = checkfun(self.arr_float_1d, self.arr_float1_1d, **kwargs)
        res01 = checkfun(
            self.arr_float_1d,
            self.arr_float1_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)

        res10 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, **kwargs)
        res11 = checkfun(
            self.arr_float_nan_1d,
            self.arr_float1_nan_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)

        targ2 = np.nan
        res20 = checkfun(self.arr_nan_1d, self.arr_float1_1d, **kwargs)
        res21 = checkfun(self.arr_float_1d, self.arr_nan_1d, **kwargs)
        res22 = checkfun(self.arr_nan_1d, self.arr_nan_1d, **kwargs)
        res23 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, **kwargs)
        res24 = checkfun(
            self.arr_float_nan_1d,
            self.arr_nan_float1_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        res25 = checkfun(
            self.arr_float_1d,
            self.arr_float1_1d,
            min_periods=len(self.arr_float_1d) + 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def test_nancorr(self):
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1)
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="pearson")

    def test_nancorr_pearson(self):
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="pearson")
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="pearson")

    def test_nancorr_kendall(self):
        sp_stats = pytest.importorskip("scipy.stats")

        targ0 = sp_stats.kendalltau(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="kendall")
        targ0 = sp_stats.kendalltau(self.arr_float_1d, self.arr_float1_1d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="kendall")

    def test_nancorr_spearman(self):
        sp_stats = pytest.importorskip("scipy.stats")

        targ0 = sp_stats.spearmanr(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.spearmanr(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="spearman")
        targ0 = sp_stats.spearmanr(self.arr_float_1d, self.arr_float1_1d)[0]
        targ1 = sp_stats.spearmanr(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="spearman")

    def test_invalid_method(self):
        pytest.importorskip("scipy")
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        msg = "Unknown method 'foo', expected one of 'kendall', 'spearman'"
        with pytest.raises(ValueError, match=msg):
            self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="foo")

    def test_nancov(self):
        targ0 = np.cov(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.cov(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancov, targ0, targ1)
        targ0 = np.cov(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.cov(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancov, targ0, targ1)


@pytest.mark.parametrize(
    "arr, correct",
    [
        ("arr_complex", False),
        ("arr_int", False),
        ("arr_bool", False),
        ("arr_str", False),
        ("arr_utf", False),
        ("arr_complex", False),
        ("arr_complex_nan", False),
        ("arr_nan_nanj", False),
        ("arr_nan_infj", True),
        ("arr_complex_nan_infj", True),
    ],
)
def test_has_infs_non_float(request, arr, correct, disable_bottleneck):
    val = request.getfixturevalue(arr)
    while getattr(val, "ndim", True):
        res0 = nanops._has_infs(val)
        if correct:
            assert res0
        else:
            assert not res0

        if not hasattr(val, "ndim"):
            break

        # Reduce dimension for next step in the loop
        val = np.take(val, 0, axis=-1)


@pytest.mark.parametrize(
    "arr, correct",
    [
        ("arr_float", False),
        ("arr_nan", False),
        ("arr_float_nan", False),
        ("arr_nan_nan", False),
        ("arr_float_inf", True),
        ("arr_inf", True),
        ("arr_nan_inf", True),
        ("arr_float_nan_inf", True),
        ("arr_nan_nan_inf", True),
    ],
)
@pytest.mark.parametrize("astype", [None, "f4", "f2"])
def test_has_infs_floats(request, arr, correct, astype, disable_bottleneck):
    val = request.getfixturevalue(arr)
    if astype is not None:
        val = val.astype(astype)
    while getattr(val, "ndim", True):
        res0 = nanops._has_infs(val)
        if correct:
            assert res0
        else:
            assert not res0

        if not hasattr(val, "ndim"):
            break

        # Reduce dimension for next step in the loop
        val = np.take(val, 0, axis=-1)


@pytest.mark.parametrize(
    "fixture", ["arr_float", "arr_complex", "arr_int", "arr_bool", "arr_str", "arr_utf"]
)
def test_bn_ok_dtype(fixture, request, disable_bottleneck):
    obj = request.getfixturevalue(fixture)
    assert nanops._bn_ok_dtype(obj.dtype, "test")


@pytest.mark.parametrize(
    "fixture",
    [
        "arr_date",
        "arr_tdelta",
        "arr_obj",
    ],
)
def test_bn_not_ok_dtype(fixture, request, disable_bottleneck):
    obj = request.getfixturevalue(fixture)
    assert not nanops._bn_ok_dtype(obj.dtype, "test")


class TestEnsureNumeric:
    def test_numeric_values(self):
        # Test integer
        assert nanops._ensure_numeric(1) == 1

        # Test float
        assert nanops._ensure_numeric(1.1) == 1.1

        # Test complex
        assert nanops._ensure_numeric(1 + 2j) == 1 + 2j

    def test_ndarray(self):
        # Test numeric ndarray
        values = np.array([1, 2, 3])
        assert np.allclose(nanops._ensure_numeric(values), values)

        # Test object ndarray
        o_values = values.astype(object)
        assert np.allclose(nanops._ensure_numeric(o_values), values)

        # Test convertible string ndarray
        s_values = np.array(["1", "2", "3"], dtype=object)
        msg = r"Could not convert \['1' '2' '3'\] to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric(s_values)

        # Test non-convertible string ndarray
        s_values = np.array(["foo", "bar", "baz"], dtype=object)
        msg = r"Could not convert .* to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric(s_values)

    def test_convertable_values(self):
        with pytest.raises(TypeError, match="Could not convert string '1' to numeric"):
            nanops._ensure_numeric("1")
        with pytest.raises(
            TypeError, match="Could not convert string '1.1' to numeric"
        ):
            nanops._ensure_numeric("1.1")
        with pytest.raises(
            TypeError, match=r"Could not convert string '1\+1j' to numeric"
        ):
            nanops._ensure_numeric("1+1j")

    def test_non_convertable_values(self):
        msg = "Could not convert string 'foo' to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric("foo")

        # with the wrong type, python raises TypeError for us
        msg = "argument must be a string or a number"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric({})
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric([])


class TestNanvarFixedValues:
    # xref GH10242
    # Samples from a normal distribution.
    @pytest.fixture
    def variance(self):
        return 3.0

    @pytest.fixture
    def samples(self, variance):
        return self.prng.normal(scale=variance**0.5, size=100000)

    def test_nanvar_all_finite(self, samples, variance):
        actual_variance = nanops.nanvar(samples)
        tm.assert_almost_equal(actual_variance, variance, rtol=1e-2)

    def test_nanvar_nans(self, samples, variance):
        samples_test = np.nan * np.ones(2 * samples.shape[0])
        samples_test[::2] = samples

        actual_variance = nanops.nanvar(samples_test, skipna=True)
        tm.assert_almost_equal(actual_variance, variance, rtol=1e-2)

        actual_variance = nanops.nanvar(samples_test, skipna=False)
        tm.assert_almost_equal(actual_variance, np.nan, rtol=1e-2)

    def test_nanstd_nans(self, samples, variance):
        samples_test = np.nan * np.ones(2 * samples.shape[0])
        samples_test[::2] = samples

        actual_std = nanops.nanstd(samples_test, skipna=True)
        tm.assert_almost_equal(actual_std, variance**0.5, rtol=1e-2)

        actual_std = nanops.nanvar(samples_test, skipna=False)
        tm.assert_almost_equal(actual_std, np.nan, rtol=1e-2)

    def test_nanvar_axis(self, samples, variance):
        # Generate some sample data.
        samples_unif = self.prng.uniform(size=samples.shape[0])
        samples = np.vstack([samples, samples_unif])

        actual_variance = nanops.nanvar(samples, axis=1)
        tm.assert_almost_equal(
            actual_variance, np.array([variance, 1.0 / 12]), rtol=1e-2
        )

    def test_nanvar_ddof(self):
        n = 5
        samples = self.prng.uniform(size=(10000, n + 1))
        samples[:, -1] = np.nan  # Force use of our own algorithm.

        variance_0 = nanops.nanvar(samples, axis=1, skipna=True, ddof=0).mean()
        variance_1 = nanops.nanvar(samples, axis=1, skipna=True, ddof=1).mean()
        variance_2 = nanops.nanvar(samples, axis=1, skipna=True, ddof=2).mean()

        # The unbiased estimate.
        var = 1.0 / 12
        tm.assert_almost_equal(variance_1, var, rtol=1e-2)

        # The underestimated variance.
        tm.assert_almost_equal(variance_0, (n - 1.0) / n * var, rtol=1e-2)

        # The overestimated variance.
        tm.assert_almost_equal(variance_2, (n - 1.0) / (n - 2.0) * var, rtol=1e-2)

    @pytest.mark.parametrize("axis", range(2))
    @pytest.mark.parametrize("ddof", range(3))
    def test_ground_truth(self, axis, ddof):
        # Test against values that were precomputed with Numpy.
        samples = np.empty((4, 4))
        samples[:3, :3] = np.array(
            [
                [0.97303362, 0.21869576, 0.55560287],
                [0.72980153, 0.03109364, 0.99155171],
                [0.09317602, 0.60078248, 0.15871292],
            ]
        )
        samples[3] = samples[:, 3] = np.nan

        # Actual variances along axis=0, 1 for ddof=0, 1, 2
        variance = np.array(
            [
                [
                    [0.13762259, 0.05619224, 0.11568816],
                    [0.20643388, 0.08428837, 0.17353224],
                    [0.41286776, 0.16857673, 0.34706449],
                ],
                [
                    [0.09519783, 0.16435395, 0.05082054],
                    [0.14279674, 0.24653093, 0.07623082],
                    [0.28559348, 0.49306186, 0.15246163],
                ],
            ]
        )

        # Test nanvar.
        var = nanops.nanvar(samples, skipna=True, axis=axis, ddof=ddof)
        tm.assert_almost_equal(var[:3], variance[axis, ddof])
        assert np.isnan(var[3])

        # Test nanstd.
        std = nanops.nanstd(samples, skipna=True, axis=axis, ddof=ddof)
        tm.assert_almost_equal(std[:3], variance[axis, ddof] ** 0.5)
        assert np.isnan(std[3])

    @pytest.mark.parametrize("ddof", range(3))
    def test_nanstd_roundoff(self, ddof):
        # Regression test for GH 10242 (test data taken from GH 10489). Ensure
        # that variance is stable.
        data = Series(766897346 * np.ones(10))
        result = data.std(ddof=ddof)
        assert result == 0.0

    @property
    def prng(self):
        return np.random.default_rng(2)


class TestNanskewFixedValues:
    # xref GH 11974
    # Test data + skewness value (computed with scipy.stats.skew)
    @pytest.fixture
    def samples(self):
        return np.sin(np.linspace(0, 1, 200))

    @pytest.fixture
    def actual_skew(self):
        return -0.1875895205961754

    @pytest.mark.parametrize("val", [3075.2, 3075.3, 3075.5])
    def test_constant_series(self, val):
        # xref GH 11974
        data = val * np.ones(300)
        skew = nanops.nanskew(data)
        assert skew == 0.0

    def test_all_finite(self):
        alpha, beta = 0.3, 0.1
        left_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nanskew(left_tailed) < 0

        alpha, beta = 0.1, 0.3
        right_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nanskew(right_tailed) > 0

    def test_ground_truth(self, samples, actual_skew):
        skew = nanops.nanskew(samples)
        tm.assert_almost_equal(skew, actual_skew)

    def test_axis(self, samples, actual_skew):
        samples = np.vstack([samples, np.nan * np.ones(len(samples))])
        skew = nanops.nanskew(samples, axis=1)
        tm.assert_almost_equal(skew, np.array([actual_skew, np.nan]))

    def test_nans(self, samples):
        samples = np.hstack([samples, np.nan])
        skew = nanops.nanskew(samples, skipna=False)
        assert np.isnan(skew)

    def test_nans_skipna(self, samples, actual_skew):
        samples = np.hstack([samples, np.nan])
        skew = nanops.nanskew(samples, skipna=True)
        tm.assert_almost_equal(skew, actual_skew)

    @property
    def prng(self):
        return np.random.default_rng(2)


class TestNankurtFixedValues:
    # xref GH 11974
    # Test data + kurtosis value (computed with scipy.stats.kurtosis)
    @pytest.fixture
    def samples(self):
        return np.sin(np.linspace(0, 1, 200))

    @pytest.fixture
    def actual_kurt(self):
        return -1.2058303433799713

    @pytest.mark.parametrize("val", [3075.2, 3075.3, 3075.5])
    def test_constant_series(self, val):
        # xref GH 11974
        data = val * np.ones(300)
        kurt = nanops.nankurt(data)
        assert kurt == 0.0

    def test_all_finite(self):
        alpha, beta = 0.3, 0.1
        left_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nankurt(left_tailed) < 2

        alpha, beta = 0.1, 0.3
        right_tailed = self.prng.beta(alpha, beta, size=100)
        assert nanops.nankurt(right_tailed) < 0

    def test_ground_truth(self, samples, actual_kurt):
        kurt = nanops.nankurt(samples)
        tm.assert_almost_equal(kurt, actual_kurt)

    def test_axis(self, samples, actual_kurt):
        samples = np.vstack([samples, np.nan * np.ones(len(samples))])
        kurt = nanops.nankurt(samples, axis=1)
        tm.assert_almost_equal(kurt, np.array([actual_kurt, np.nan]))

    def test_nans(self, samples):
        samples = np.hstack([samples, np.nan])
        kurt = nanops.nankurt(samples, skipna=False)
        assert np.isnan(kurt)

    def test_nans_skipna(self, samples, actual_kurt):
        samples = np.hstack([samples, np.nan])
        kurt = nanops.nankurt(samples, skipna=True)
        tm.assert_almost_equal(kurt, actual_kurt)

    @property
    def prng(self):
        return np.random.default_rng(2)


class TestDatetime64NaNOps:
    @pytest.fixture(params=["s", "ms", "us", "ns"])
    def unit(self, request):
        return request.param

    # Enabling mean changes the behavior of DataFrame.mean
    # See https://github.com/pandas-dev/pandas/issues/24752
    def test_nanmean(self, unit):
        dti = pd.date_range("2016-01-01", periods=3).as_unit(unit)
        expected = dti[1]

        for obj in [dti, dti._data]:
            result = nanops.nanmean(obj)
            assert result == expected

        dti2 = dti.insert(1, pd.NaT)

        for obj in [dti2, dti2._data]:
            result = nanops.nanmean(obj)
            assert result == expected

    @pytest.mark.parametrize("constructor", ["M8", "m8"])
    def test_nanmean_skipna_false(self, constructor, unit):
        dtype = f"{constructor}[{unit}]"
        arr = np.arange(12).astype(np.int64).view(dtype).reshape(4, 3)

        arr[-1, -1] = "NaT"

        result = nanops.nanmean(arr, skipna=False)
        assert np.isnat(result)
        assert result.dtype == dtype

        result = nanops.nanmean(arr, axis=0, skipna=False)
        expected = np.array([4, 5, "NaT"], dtype=arr.dtype)
        tm.assert_numpy_array_equal(result, expected)

        result = nanops.nanmean(arr, axis=1, skipna=False)
        expected = np.array([arr[0, 1], arr[1, 1], arr[2, 1], arr[-1, -1]])
        tm.assert_numpy_array_equal(result, expected)


def test_use_bottleneck():
    if nanops._BOTTLENECK_INSTALLED:
        with pd.option_context("use_bottleneck", True):
            assert pd.get_option("use_bottleneck")

        with pd.option_context("use_bottleneck", False):
            assert not pd.get_option("use_bottleneck")


@pytest.mark.parametrize(
    "numpy_op, expected",
    [
        (np.sum, 10),
        (np.nansum, 10),
        (np.mean, 2.5),
        (np.nanmean, 2.5),
        (np.median, 2.5),
        (np.nanmedian, 2.5),
        (np.min, 1),
        (np.max, 4),
        (np.nanmin, 1),
        (np.nanmax, 4),
    ],
)
def test_numpy_ops(numpy_op, expected):
    # GH8383
    result = numpy_op(Series([1, 2, 3, 4]))
    assert result == expected


@pytest.mark.parametrize(
    "operation",
    [
        nanops.nanany,
        nanops.nanall,
        nanops.nansum,
        nanops.nanmean,
        nanops.nanmedian,
        nanops.nanstd,
        nanops.nanvar,
        nanops.nansem,
        nanops.nanargmax,
        nanops.nanargmin,
        nanops.nanmax,
        nanops.nanmin,
        nanops.nanskew,
        nanops.nankurt,
        nanops.nanprod,
    ],
)
def test_nanops_independent_of_mask_param(operation):
    # GH22764
    ser = Series([1, 2, np.nan, 3, np.nan, 4])
    mask = ser.isna()
    median_expected = operation(ser._values)
    median_result = operation(ser._values, mask=mask)
    assert median_expected == median_result


@pytest.mark.parametrize("min_count", [-1, 0])
def test_check_below_min_count_negative_or_zero_min_count(min_count):
    # GH35227
    result = nanops.check_below_min_count((21, 37), None, min_count)
    expected_result = False
    assert result == expected_result


@pytest.mark.parametrize(
    "mask", [None, np.array([False, False, True]), np.array([True] + 9 * [False])]
)
@pytest.mark.parametrize("min_count, expected_result", [(1, False), (101, True)])
def test_check_below_min_count_positive_min_count(mask, min_count, expected_result):
    # GH35227
    shape = (10, 10)
    result = nanops.check_below_min_count(shape, mask, min_count)
    assert result == expected_result


@td.skip_if_windows
@td.skip_if_32bit
@pytest.mark.parametrize("min_count, expected_result", [(1, False), (2812191852, True)])
def test_check_below_min_count_large_shape(min_count, expected_result):
    # GH35227 large shape used to show that the issue is fixed
    shape = (2244367, 1253)
    result = nanops.check_below_min_count(shape, mask=None, min_count=min_count)
    assert result == expected_result


@pytest.mark.parametrize("func", ["nanmean", "nansum"])
def test_check_bottleneck_disallow(any_real_numpy_dtype, func):
    # GH 42878 bottleneck sometimes produces unreliable results for mean and sum
    assert not nanops._bn_ok_dtype(np.dtype(any_real_numpy_dtype).type, func)


@pytest.mark.parametrize("val", [2**55, -(2**55), 20150515061816532])
def test_nanmean_overflow(disable_bottleneck, val):
    # GH 10155
    # In the previous implementation mean can overflow for int dtypes, it
    # is now consistent with numpy

    ser = Series(val, index=range(500), dtype=np.int64)
    result = ser.mean()
    np_result = ser.values.mean()
    assert result == val
    assert result == np_result
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "dtype",
    [
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        getattr(np, "float128", None),
    ],
)
@pytest.mark.parametrize("method", ["mean", "std", "var", "skew", "kurt", "min", "max"])
def test_returned_dtype(disable_bottleneck, dtype, method):
    if dtype is None:
        pytest.skip("np.float128 not available")

    ser = Series(range(10), dtype=dtype)
    result = getattr(ser, method)()
    if is_integer_dtype(dtype) and method not in ["min", "max"]:
        assert result.dtype == np.float64
    else:
        assert result.dtype == dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_relational.py ===
from sympy.core.logic import fuzzy_and
from sympy.core.sympify import _sympify
from sympy.multipledispatch import dispatch
from sympy.testing.pytest import XFAIL, raises
from sympy.assumptions.ask import Q
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr, unchanged
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import sign, Abs
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (And, Implies, Not, Or, Xor)
from sympy.sets import Reals
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.core.relational import (Relational, Equality, Unequality,
                                   GreaterThan, LessThan, StrictGreaterThan,
                                   StrictLessThan, Rel, Eq, Lt, Le,
                                   Gt, Ge, Ne, is_le, is_gt, is_ge, is_lt, is_eq, is_neq)
from sympy.sets.sets import Interval, FiniteSet

from itertools import combinations

x, y, z, t = symbols('x,y,z,t')


def rel_check(a, b):
    from sympy.testing.pytest import raises
    assert a.is_number and b.is_number
    for do in range(len({type(a), type(b)})):
        if S.NaN in (a, b):
            v = [(a == b), (a != b)]
            assert len(set(v)) == 1 and v[0] == False
            assert not (a != b) and not (a == b)
            assert raises(TypeError, lambda: a < b)
            assert raises(TypeError, lambda: a <= b)
            assert raises(TypeError, lambda: a > b)
            assert raises(TypeError, lambda: a >= b)
        else:
            E = [(a == b), (a != b)]
            assert len(set(E)) == 2
            v = [
            (a < b), (a <= b), (a > b), (a >= b)]
            i = [
            [True,    True,     False,   False],
            [False,   True,     False,   True], # <-- i == 1
            [False,   False,    True,    True]].index(v)
            if i == 1:
                assert E[0] or (a.is_Float != b.is_Float) # ugh
            else:
                assert E[1]
        a, b = b, a
    return True


def test_rel_ne():
    assert Relational(x, y, '!=') == Ne(x, y)

    # issue 6116
    p = Symbol('p', positive=True)
    assert Ne(p, 0) is S.true


def test_rel_subs():
    e = Relational(x, y, '==')
    e = e.subs(x, z)

    assert isinstance(e, Equality)
    assert e.lhs == z
    assert e.rhs == y

    e = Relational(x, y, '>=')
    e = e.subs(x, z)

    assert isinstance(e, GreaterThan)
    assert e.lhs == z
    assert e.rhs == y

    e = Relational(x, y, '<=')
    e = e.subs(x, z)

    assert isinstance(e, LessThan)
    assert e.lhs == z
    assert e.rhs == y

    e = Relational(x, y, '>')
    e = e.subs(x, z)

    assert isinstance(e, StrictGreaterThan)
    assert e.lhs == z
    assert e.rhs == y

    e = Relational(x, y, '<')
    e = e.subs(x, z)

    assert isinstance(e, StrictLessThan)
    assert e.lhs == z
    assert e.rhs == y

    e = Eq(x, 0)
    assert e.subs(x, 0) is S.true
    assert e.subs(x, 1) is S.false


def test_wrappers():
    e = x + x**2

    res = Relational(y, e, '==')
    assert Rel(y, x + x**2, '==') == res
    assert Eq(y, x + x**2) == res

    res = Relational(y, e, '<')
    assert Lt(y, x + x**2) == res

    res = Relational(y, e, '<=')
    assert Le(y, x + x**2) == res

    res = Relational(y, e, '>')
    assert Gt(y, x + x**2) == res

    res = Relational(y, e, '>=')
    assert Ge(y, x + x**2) == res

    res = Relational(y, e, '!=')
    assert Ne(y, x + x**2) == res


def test_Eq_Ne():

    assert Eq(x, x)  # issue 5719

    # issue 6116
    p = Symbol('p', positive=True)
    assert Eq(p, 0) is S.false

    # issue 13348; 19048
    # SymPy is strict about 0 and 1 not being
    # interpreted as Booleans
    assert Eq(True, 1) is S.false
    assert Eq(False, 0) is S.false
    assert Eq(~x, 0) is S.false
    assert Eq(~x, 1) is S.false
    assert Ne(True, 1) is S.true
    assert Ne(False, 0) is S.true
    assert Ne(~x, 0) is S.true
    assert Ne(~x, 1) is S.true

    assert Eq((), 1) is S.false
    assert Ne((), 1) is S.true


def test_as_poly():
    from sympy.polys.polytools import Poly
    # Only Eq should have an as_poly method:
    assert Eq(x, 1).as_poly() == Poly(x - 1, x, domain='ZZ')
    raises(AttributeError, lambda: Ne(x, 1).as_poly())
    raises(AttributeError, lambda: Ge(x, 1).as_poly())
    raises(AttributeError, lambda: Gt(x, 1).as_poly())
    raises(AttributeError, lambda: Le(x, 1).as_poly())
    raises(AttributeError, lambda: Lt(x, 1).as_poly())


def test_rel_Infinity():
    # NOTE: All of these are actually handled by sympy.core.Number, and do
    # not create Relational objects.
    assert (oo > oo) is S.false
    assert (oo > -oo) is S.true
    assert (oo > 1) is S.true
    assert (oo < oo) is S.false
    assert (oo < -oo) is S.false
    assert (oo < 1) is S.false
    assert (oo >= oo) is S.true
    assert (oo >= -oo) is S.true
    assert (oo >= 1) is S.true
    assert (oo <= oo) is S.true
    assert (oo <= -oo) is S.false
    assert (oo <= 1) is S.false
    assert (-oo > oo) is S.false
    assert (-oo > -oo) is S.false
    assert (-oo > 1) is S.false
    assert (-oo < oo) is S.true
    assert (-oo < -oo) is S.false
    assert (-oo < 1) is S.true
    assert (-oo >= oo) is S.false
    assert (-oo >= -oo) is S.true
    assert (-oo >= 1) is S.false
    assert (-oo <= oo) is S.true
    assert (-oo <= -oo) is S.true
    assert (-oo <= 1) is S.true


def test_infinite_symbol_inequalities():
    x = Symbol('x', extended_positive=True, infinite=True)
    y = Symbol('y', extended_positive=True, infinite=True)
    z = Symbol('z', extended_negative=True, infinite=True)
    w = Symbol('w', extended_negative=True, infinite=True)

    inf_set = (x, y, oo)
    ninf_set = (z, w, -oo)

    for inf1 in inf_set:
        assert (inf1 < 1) is S.false
        assert (inf1 > 1) is S.true
        assert (inf1 <= 1) is S.false
        assert (inf1 >= 1) is S.true

        for inf2 in inf_set:
            assert (inf1 < inf2) is S.false
            assert (inf1 > inf2) is S.false
            assert (inf1 <= inf2) is S.true
            assert (inf1 >= inf2) is S.true

        for ninf1 in ninf_set:
            assert (inf1 < ninf1) is S.false
            assert (inf1 > ninf1) is S.true
            assert (inf1 <= ninf1) is S.false
            assert (inf1 >= ninf1) is S.true
            assert (ninf1 < inf1) is S.true
            assert (ninf1 > inf1) is S.false
            assert (ninf1 <= inf1) is S.true
            assert (ninf1 >= inf1) is S.false

    for ninf1 in ninf_set:
        assert (ninf1 < 1) is S.true
        assert (ninf1 > 1) is S.false
        assert (ninf1 <= 1) is S.true
        assert (ninf1 >= 1) is S.false

        for ninf2 in ninf_set:
            assert (ninf1 < ninf2) is S.false
            assert (ninf1 > ninf2) is S.false
            assert (ninf1 <= ninf2) is S.true
            assert (ninf1 >= ninf2) is S.true


def test_bool():
    assert Eq(0, 0) is S.true
    assert Eq(1, 0) is S.false
    assert Ne(0, 0) is S.false
    assert Ne(1, 0) is S.true
    assert Lt(0, 1) is S.true
    assert Lt(1, 0) is S.false
    assert Le(0, 1) is S.true
    assert Le(1, 0) is S.false
    assert Le(0, 0) is S.true
    assert Gt(1, 0) is S.true
    assert Gt(0, 1) is S.false
    assert Ge(1, 0) is S.true
    assert Ge(0, 1) is S.false
    assert Ge(1, 1) is S.true
    assert Eq(I, 2) is S.false
    assert Ne(I, 2) is S.true
    raises(TypeError, lambda: Gt(I, 2))
    raises(TypeError, lambda: Ge(I, 2))
    raises(TypeError, lambda: Lt(I, 2))
    raises(TypeError, lambda: Le(I, 2))
    a = Float('.000000000000000000001', '')
    b = Float('.0000000000000000000001', '')
    assert Eq(pi + a, pi + b) is S.false


def test_rich_cmp():
    assert (x < y) == Lt(x, y)
    assert (x <= y) == Le(x, y)
    assert (x > y) == Gt(x, y)
    assert (x >= y) == Ge(x, y)


def test_doit():
    from sympy.core.symbol import Symbol
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    np = Symbol('np', nonpositive=True)
    nn = Symbol('nn', nonnegative=True)

    assert Gt(p, 0).doit() is S.true
    assert Gt(p, 1).doit() == Gt(p, 1)
    assert Ge(p, 0).doit() is S.true
    assert Le(p, 0).doit() is S.false
    assert Lt(n, 0).doit() is S.true
    assert Le(np, 0).doit() is S.true
    assert Gt(nn, 0).doit() == Gt(nn, 0)
    assert Lt(nn, 0).doit() is S.false

    assert Eq(x, 0).doit() == Eq(x, 0)


def test_new_relational():
    x = Symbol('x')

    assert Eq(x, 0) == Relational(x, 0)       # None ==> Equality
    assert Eq(x, 0) == Relational(x, 0, '==')
    assert Eq(x, 0) == Relational(x, 0, 'eq')
    assert Eq(x, 0) == Equality(x, 0)

    assert Eq(x, 0) != Relational(x, 1)       # None ==> Equality
    assert Eq(x, 0) != Relational(x, 1, '==')
    assert Eq(x, 0) != Relational(x, 1, 'eq')
    assert Eq(x, 0) != Equality(x, 1)

    assert Eq(x, -1) == Relational(x, -1)       # None ==> Equality
    assert Eq(x, -1) == Relational(x, -1, '==')
    assert Eq(x, -1) == Relational(x, -1, 'eq')
    assert Eq(x, -1) == Equality(x, -1)
    assert Eq(x, -1) != Relational(x, 1)       # None ==> Equality
    assert Eq(x, -1) != Relational(x, 1, '==')
    assert Eq(x, -1) != Relational(x, 1, 'eq')
    assert Eq(x, -1) != Equality(x, 1)

    assert Ne(x, 0) == Relational(x, 0, '!=')
    assert Ne(x, 0) == Relational(x, 0, '<>')
    assert Ne(x, 0) == Relational(x, 0, 'ne')
    assert Ne(x, 0) == Unequality(x, 0)
    assert Ne(x, 0) != Relational(x, 1, '!=')
    assert Ne(x, 0) != Relational(x, 1, '<>')
    assert Ne(x, 0) != Relational(x, 1, 'ne')
    assert Ne(x, 0) != Unequality(x, 1)

    assert Ge(x, 0) == Relational(x, 0, '>=')
    assert Ge(x, 0) == Relational(x, 0, 'ge')
    assert Ge(x, 0) == GreaterThan(x, 0)
    assert Ge(x, 1) != Relational(x, 0, '>=')
    assert Ge(x, 1) != Relational(x, 0, 'ge')
    assert Ge(x, 1) != GreaterThan(x, 0)
    assert (x >= 1) == Relational(x, 1, '>=')
    assert (x >= 1) == Relational(x, 1, 'ge')
    assert (x >= 1) == GreaterThan(x, 1)
    assert (x >= 0) != Relational(x, 1, '>=')
    assert (x >= 0) != Relational(x, 1, 'ge')
    assert (x >= 0) != GreaterThan(x, 1)

    assert Le(x, 0) == Relational(x, 0, '<=')
    assert Le(x, 0) == Relational(x, 0, 'le')
    assert Le(x, 0) == LessThan(x, 0)
    assert Le(x, 1) != Relational(x, 0, '<=')
    assert Le(x, 1) != Relational(x, 0, 'le')
    assert Le(x, 1) != LessThan(x, 0)
    assert (x <= 1) == Relational(x, 1, '<=')
    assert (x <= 1) == Relational(x, 1, 'le')
    assert (x <= 1) == LessThan(x, 1)
    assert (x <= 0) != Relational(x, 1, '<=')
    assert (x <= 0) != Relational(x, 1, 'le')
    assert (x <= 0) != LessThan(x, 1)

    assert Gt(x, 0) == Relational(x, 0, '>')
    assert Gt(x, 0) == Relational(x, 0, 'gt')
    assert Gt(x, 0) == StrictGreaterThan(x, 0)
    assert Gt(x, 1) != Relational(x, 0, '>')
    assert Gt(x, 1) != Relational(x, 0, 'gt')
    assert Gt(x, 1) != StrictGreaterThan(x, 0)
    assert (x > 1) == Relational(x, 1, '>')
    assert (x > 1) == Relational(x, 1, 'gt')
    assert (x > 1) == StrictGreaterThan(x, 1)
    assert (x > 0) != Relational(x, 1, '>')
    assert (x > 0) != Relational(x, 1, 'gt')
    assert (x > 0) != StrictGreaterThan(x, 1)

    assert Lt(x, 0) == Relational(x, 0, '<')
    assert Lt(x, 0) == Relational(x, 0, 'lt')
    assert Lt(x, 0) == StrictLessThan(x, 0)
    assert Lt(x, 1) != Relational(x, 0, '<')
    assert Lt(x, 1) != Relational(x, 0, 'lt')
    assert Lt(x, 1) != StrictLessThan(x, 0)
    assert (x < 1) == Relational(x, 1, '<')
    assert (x < 1) == Relational(x, 1, 'lt')
    assert (x < 1) == StrictLessThan(x, 1)
    assert (x < 0) != Relational(x, 1, '<')
    assert (x < 0) != Relational(x, 1, 'lt')
    assert (x < 0) != StrictLessThan(x, 1)

    # finally, some fuzz testing
    from sympy.core.random import randint
    for i in range(100):
        while 1:
            strtype, length = (chr, 65535) if randint(0, 1) else (chr, 255)
            relation_type = strtype(randint(0, length))
            if randint(0, 1):
                relation_type += strtype(randint(0, length))
            if relation_type not in ('==', 'eq', '!=', '<>', 'ne', '>=', 'ge',
                                     '<=', 'le', '>', 'gt', '<', 'lt', ':=',
                                     '+=', '-=', '*=', '/=', '%='):
                break

        raises(ValueError, lambda: Relational(x, 1, relation_type))
    assert all(Relational(x, 0, op).rel_op == '==' for op in ('eq', '=='))
    assert all(Relational(x, 0, op).rel_op == '!='
               for op in ('ne', '<>', '!='))
    assert all(Relational(x, 0, op).rel_op == '>' for op in ('gt', '>'))
    assert all(Relational(x, 0, op).rel_op == '<' for op in ('lt', '<'))
    assert all(Relational(x, 0, op).rel_op == '>=' for op in ('ge', '>='))
    assert all(Relational(x, 0, op).rel_op == '<=' for op in ('le', '<='))


def test_relational_arithmetic():
    for cls in [Eq, Ne, Le, Lt, Ge, Gt]:
        rel = cls(x, y)
        raises(TypeError, lambda: 0+rel)
        raises(TypeError, lambda: 1*rel)
        raises(TypeError, lambda: 1**rel)
        raises(TypeError, lambda: rel**1)
        raises(TypeError, lambda: Add(0, rel))
        raises(TypeError, lambda: Mul(1, rel))
        raises(TypeError, lambda: Pow(1, rel))
        raises(TypeError, lambda: Pow(rel, 1))


def test_relational_bool_output():
    # https://github.com/sympy/sympy/issues/5931
    raises(TypeError, lambda: bool(x > 3))
    raises(TypeError, lambda: bool(x >= 3))
    raises(TypeError, lambda: bool(x < 3))
    raises(TypeError, lambda: bool(x <= 3))
    raises(TypeError, lambda: bool(Eq(x, 3)))
    raises(TypeError, lambda: bool(Ne(x, 3)))


def test_relational_logic_symbols():
    # See issue 6204
    assert (x < y) & (z < t) == And(x < y, z < t)
    assert (x < y) | (z < t) == Or(x < y, z < t)
    assert ~(x < y) == Not(x < y)
    assert (x < y) >> (z < t) == Implies(x < y, z < t)
    assert (x < y) << (z < t) == Implies(z < t, x < y)
    assert (x < y) ^ (z < t) == Xor(x < y, z < t)

    assert isinstance((x < y) & (z < t), And)
    assert isinstance((x < y) | (z < t), Or)
    assert isinstance(~(x < y), GreaterThan)
    assert isinstance((x < y) >> (z < t), Implies)
    assert isinstance((x < y) << (z < t), Implies)
    assert isinstance((x < y) ^ (z < t), (Or, Xor))


def test_univariate_relational_as_set():
    assert (x > 0).as_set() == Interval(0, oo, True, True)
    assert (x >= 0).as_set() == Interval(0, oo)
    assert (x < 0).as_set() == Interval(-oo, 0, True, True)
    assert (x <= 0).as_set() == Interval(-oo, 0)
    assert Eq(x, 0).as_set() == FiniteSet(0)
    assert Ne(x, 0).as_set() == Interval(-oo, 0, True, True) + \
        Interval(0, oo, True, True)

    assert (x**2 >= 4).as_set() == Interval(-oo, -2) + Interval(2, oo)


@XFAIL
def test_multivariate_relational_as_set():
    assert (x*y >= 0).as_set() == Interval(0, oo)*Interval(0, oo) + \
        Interval(-oo, 0)*Interval(-oo, 0)


def test_Not():
    assert Not(Equality(x, y)) == Unequality(x, y)
    assert Not(Unequality(x, y)) == Equality(x, y)
    assert Not(StrictGreaterThan(x, y)) == LessThan(x, y)
    assert Not(StrictLessThan(x, y)) == GreaterThan(x, y)
    assert Not(GreaterThan(x, y)) == StrictLessThan(x, y)
    assert Not(LessThan(x, y)) == StrictGreaterThan(x, y)


def test_evaluate():
    assert str(Eq(x, x, evaluate=False)) == 'Eq(x, x)'
    assert Eq(x, x, evaluate=False).doit() == S.true
    assert str(Ne(x, x, evaluate=False)) == 'Ne(x, x)'
    assert Ne(x, x, evaluate=False).doit() == S.false

    assert str(Ge(x, x, evaluate=False)) == 'x >= x'
    assert str(Le(x, x, evaluate=False)) == 'x <= x'
    assert str(Gt(x, x, evaluate=False)) == 'x > x'
    assert str(Lt(x, x, evaluate=False)) == 'x < x'


def assert_all_ineq_raise_TypeError(a, b):
    raises(TypeError, lambda: a > b)
    raises(TypeError, lambda: a >= b)
    raises(TypeError, lambda: a < b)
    raises(TypeError, lambda: a <= b)
    raises(TypeError, lambda: b > a)
    raises(TypeError, lambda: b >= a)
    raises(TypeError, lambda: b < a)
    raises(TypeError, lambda: b <= a)


def assert_all_ineq_give_class_Inequality(a, b):
    """All inequality operations on `a` and `b` result in class Inequality."""
    from sympy.core.relational import _Inequality as Inequality
    assert isinstance(a > b,  Inequality)
    assert isinstance(a >= b, Inequality)
    assert isinstance(a < b,  Inequality)
    assert isinstance(a <= b, Inequality)
    assert isinstance(b > a,  Inequality)
    assert isinstance(b >= a, Inequality)
    assert isinstance(b < a,  Inequality)
    assert isinstance(b <= a, Inequality)


def test_imaginary_compare_raises_TypeError():
    # See issue #5724
    assert_all_ineq_raise_TypeError(I, x)


def test_complex_compare_not_real():
    # two cases which are not real
    y = Symbol('y', imaginary=True)
    z = Symbol('z', complex=True, extended_real=False)
    for w in (y, z):
        assert_all_ineq_raise_TypeError(2, w)
    # some cases which should remain un-evaluated
    t = Symbol('t')
    x = Symbol('x', real=True)
    z = Symbol('z', complex=True)
    for w in (x, z, t):
        assert_all_ineq_give_class_Inequality(2, w)


def test_imaginary_and_inf_compare_raises_TypeError():
    # See pull request #7835
    y = Symbol('y', imaginary=True)
    assert_all_ineq_raise_TypeError(oo, y)
    assert_all_ineq_raise_TypeError(-oo, y)


def test_complex_pure_imag_not_ordered():
    raises(TypeError, lambda: 2*I < 3*I)

    # more generally
    x = Symbol('x', real=True, nonzero=True)
    y = Symbol('y', imaginary=True)
    z = Symbol('z', complex=True)
    assert_all_ineq_raise_TypeError(I, y)

    t = I*x   # an imaginary number, should raise errors
    assert_all_ineq_raise_TypeError(2, t)

    t = -I*y   # a real number, so no errors
    assert_all_ineq_give_class_Inequality(2, t)

    t = I*z   # unknown, should be unevaluated
    assert_all_ineq_give_class_Inequality(2, t)


def test_x_minus_y_not_same_as_x_lt_y():
    """
    A consequence of pull request #7792 is that `x - y < 0` and `x < y`
    are not synonymous.
    """
    x = I + 2
    y = I + 3
    raises(TypeError, lambda: x < y)
    assert x - y < 0

    ineq = Lt(x, y, evaluate=False)
    raises(TypeError, lambda: ineq.doit())
    assert ineq.lhs - ineq.rhs < 0

    t = Symbol('t', imaginary=True)
    x = 2 + t
    y = 3 + t
    ineq = Lt(x, y, evaluate=False)
    raises(TypeError, lambda: ineq.doit())
    assert ineq.lhs - ineq.rhs < 0

    # this one should give error either way
    x = I + 2
    y = 2*I + 3
    raises(TypeError, lambda: x < y)
    raises(TypeError, lambda: x - y < 0)


def test_nan_equality_exceptions():
    # See issue #7774
    import random
    assert Equality(nan, nan) is S.false
    assert Unequality(nan, nan) is S.true

    # See issue #7773
    A = (x, S.Zero, S.One/3, pi, oo, -oo)
    assert Equality(nan, random.choice(A)) is S.false
    assert Equality(random.choice(A), nan) is S.false
    assert Unequality(nan, random.choice(A)) is S.true
    assert Unequality(random.choice(A), nan) is S.true


def test_nan_inequality_raise_errors():
    # See discussion in pull request #7776.  We test inequalities with
    # a set including examples of various classes.
    for q in (x, S.Zero, S(10), S.One/3, pi, S(1.3), oo, -oo, nan):
        assert_all_ineq_raise_TypeError(q, nan)


def test_nan_complex_inequalities():
    # Comparisons of NaN with non-real raise errors, we're not too
    # fussy whether its the NaN error or complex error.
    for r in (I, zoo, Symbol('z', imaginary=True)):
        assert_all_ineq_raise_TypeError(r, nan)


def test_complex_infinity_inequalities():
    raises(TypeError, lambda: zoo > 0)
    raises(TypeError, lambda: zoo >= 0)
    raises(TypeError, lambda: zoo < 0)
    raises(TypeError, lambda: zoo <= 0)


def test_inequalities_symbol_name_same():
    """Using the operator and functional forms should give same results."""
    # We test all combinations from a set
    # FIXME: could replace with random selection after test passes
    A = (x, y, S.Zero, S.One/3, pi, oo, -oo)
    for a in A:
        for b in A:
            assert Gt(a, b) == (a > b)
            assert Lt(a, b) == (a < b)
            assert Ge(a, b) == (a >= b)
            assert Le(a, b) == (a <= b)

    for b in (y, S.Zero, S.One/3, pi, oo, -oo):
        assert Gt(x, b, evaluate=False) == (x > b)
        assert Lt(x, b, evaluate=False) == (x < b)
        assert Ge(x, b, evaluate=False) == (x >= b)
        assert Le(x, b, evaluate=False) == (x <= b)

    for b in (y, S.Zero, S.One/3, pi, oo, -oo):
        assert Gt(b, x, evaluate=False) == (b > x)
        assert Lt(b, x, evaluate=False) == (b < x)
        assert Ge(b, x, evaluate=False) == (b >= x)
        assert Le(b, x, evaluate=False) == (b <= x)


def test_inequalities_symbol_name_same_complex():
    """Using the operator and functional forms should give same results.
    With complex non-real numbers, both should raise errors.
    """
    # FIXME: could replace with random selection after test passes
    for a in (x, S.Zero, S.One/3, pi, oo, Rational(1, 3)):
        raises(TypeError, lambda: Gt(a, I))
        raises(TypeError, lambda: a > I)
        raises(TypeError, lambda: Lt(a, I))
        raises(TypeError, lambda: a < I)
        raises(TypeError, lambda: Ge(a, I))
        raises(TypeError, lambda: a >= I)
        raises(TypeError, lambda: Le(a, I))
        raises(TypeError, lambda: a <= I)


def test_inequalities_cant_sympify_other():
    # see issue 7833
    from operator import gt, lt, ge, le

    bar = "foo"

    for a in (x, S.Zero, S.One/3, pi, I, zoo, oo, -oo, nan, Rational(1, 3)):
        for op in (lt, gt, le, ge):
            raises(TypeError, lambda: op(a, bar))


def test_ineq_avoid_wild_symbol_flip():
    # see issue #7951, we try to avoid this internally, e.g., by using
    # __lt__ instead of "<".
    from sympy.core.symbol import Wild
    p = symbols('p', cls=Wild)
    # x > p might flip, but Gt should not:
    assert Gt(x, p) == Gt(x, p, evaluate=False)
    # Previously failed as 'p > x':
    e = Lt(x, y).subs({y: p})
    assert e == Lt(x, p, evaluate=False)
    # Previously failed as 'p <= x':
    e = Ge(x, p).doit()
    assert e == Ge(x, p, evaluate=False)


def test_issue_8245():
    a = S("6506833320952669167898688709329/5070602400912917605986812821504")
    assert rel_check(a, a.n(10))
    assert rel_check(a, a.n(20))
    assert rel_check(a, a.n())
    # prec of 31 is enough to fully capture a as mpf
    assert Float(a, 31) == Float(str(a.p), '')/Float(str(a.q), '')
    for i in range(31):
        r = Rational(Float(a, i))
        f = Float(r)
        assert (f < a) == (Rational(f) < a)
    # test sign handling
    assert (-f < -a) == (Rational(-f) < -a)
    # test equivalence handling
    isa = Float(a.p,'')/Float(a.q,'')
    assert isa <= a
    assert not isa < a
    assert isa >= a
    assert not isa > a
    assert isa > 0

    a = sqrt(2)
    r = Rational(str(a.n(30)))
    assert rel_check(a, r)

    a = sqrt(2)
    r = Rational(str(a.n(29)))
    assert rel_check(a, r)

    assert Eq(log(cos(2)**2 + sin(2)**2), 0) is S.true


def test_issue_8449():
    p = Symbol('p', nonnegative=True)
    assert Lt(-oo, p)
    assert Ge(-oo, p) is S.false
    assert Gt(oo, -p)
    assert Le(oo, -p) is S.false


def test_simplify_relational():
    assert simplify(x*(y + 1) - x*y - x + 1 < x) == (x > 1)
    assert simplify(x*(y + 1) - x*y - x - 1 < x) == (x > -1)
    assert simplify(x < x*(y + 1) - x*y - x + 1) == (x < 1)
    q, r = symbols("q r")
    assert (((-q + r) - (q - r)) <= 0).simplify() == (q >= r)
    root2 = sqrt(2)
    equation = ((root2 * (-q + r) - root2 * (q - r)) <= 0).simplify()
    assert equation == (q >= r)
    r = S.One < x
    # canonical operations are not the same as simplification,
    # so if there is no simplification, canonicalization will
    # be done unless the measure forbids it
    assert simplify(r) == r.canonical
    assert simplify(r, ratio=0) != r.canonical
    # this is not a random test; in _eval_simplify
    # this will simplify to S.false and that is the
    # reason for the 'if r.is_Relational' in Relational's
    # _eval_simplify routine
    assert simplify(-(2**(pi*Rational(3, 2)) + 6**pi)**(1/pi) +
                    2*(2**(pi/2) + 3**pi)**(1/pi) < 0) is S.false
    # canonical at least
    assert Eq(y, x).simplify() == Eq(x, y)
    assert Eq(x - 1, 0).simplify() == Eq(x, 1)
    assert Eq(x - 1, x).simplify() == S.false
    assert Eq(2*x - 1, x).simplify() == Eq(x, 1)
    assert Eq(2*x, 4).simplify() == Eq(x, 2)
    z = cos(1)**2 + sin(1)**2 - 1  # z.is_zero is None
    assert Eq(z*x, 0).simplify() == S.true

    assert Ne(y, x).simplify() == Ne(x, y)
    assert Ne(x - 1, 0).simplify() == Ne(x, 1)
    assert Ne(x - 1, x).simplify() == S.true
    assert Ne(2*x - 1, x).simplify() == Ne(x, 1)
    assert Ne(2*x, 4).simplify() == Ne(x, 2)
    assert Ne(z*x, 0).simplify() == S.false

    # No real-valued assumptions
    assert Ge(y, x).simplify() == Le(x, y)
    assert Ge(x - 1, 0).simplify() == Ge(x, 1)
    assert Ge(x - 1, x).simplify() == S.false
    assert Ge(2*x - 1, x).simplify() == Ge(x, 1)
    assert Ge(2*x, 4).simplify() == Ge(x, 2)
    assert Ge(z*x, 0).simplify() == S.true
    assert Ge(x, -2).simplify() == Ge(x, -2)
    assert Ge(-x, -2).simplify() == Le(x, 2)
    assert Ge(x, 2).simplify() == Ge(x, 2)
    assert Ge(-x, 2).simplify() == Le(x, -2)

    assert Le(y, x).simplify() == Ge(x, y)
    assert Le(x - 1, 0).simplify() == Le(x, 1)
    assert Le(x - 1, x).simplify() == S.true
    assert Le(2*x - 1, x).simplify() == Le(x, 1)
    assert Le(2*x, 4).simplify() == Le(x, 2)
    assert Le(z*x, 0).simplify() == S.true
    assert Le(x, -2).simplify() == Le(x, -2)
    assert Le(-x, -2).simplify() == Ge(x, 2)
    assert Le(x, 2).simplify() == Le(x, 2)
    assert Le(-x, 2).simplify() == Ge(x, -2)

    assert Gt(y, x).simplify() == Lt(x, y)
    assert Gt(x - 1, 0).simplify() == Gt(x, 1)
    assert Gt(x - 1, x).simplify() == S.false
    assert Gt(2*x - 1, x).simplify() == Gt(x, 1)
    assert Gt(2*x, 4).simplify() == Gt(x, 2)
    assert Gt(z*x, 0).simplify() == S.false
    assert Gt(x, -2).simplify() == Gt(x, -2)
    assert Gt(-x, -2).simplify() == Lt(x, 2)
    assert Gt(x, 2).simplify() == Gt(x, 2)
    assert Gt(-x, 2).simplify() == Lt(x, -2)

    assert Lt(y, x).simplify() == Gt(x, y)
    assert Lt(x - 1, 0).simplify() == Lt(x, 1)
    assert Lt(x - 1, x).simplify() == S.true
    assert Lt(2*x - 1, x).simplify() == Lt(x, 1)
    assert Lt(2*x, 4).simplify() == Lt(x, 2)
    assert Lt(z*x, 0).simplify() == S.false
    assert Lt(x, -2).simplify() == Lt(x, -2)
    assert Lt(-x, -2).simplify() == Gt(x, 2)
    assert Lt(x, 2).simplify() == Lt(x, 2)
    assert Lt(-x, 2).simplify() == Gt(x, -2)

    # Test particular branches of _eval_simplify
    m = exp(1) - exp_polar(1)
    assert simplify(m*x > 1) is S.false
    # These two test the same branch
    assert simplify(m*x + 2*m*y > 1) is S.false
    assert simplify(m*x + y > 1 + y) is S.false


def test_equals():
    w, x, y, z = symbols('w:z')
    f = Function('f')
    assert Eq(x, 1).equals(Eq(x*(y + 1) - x*y - x + 1, x))
    assert Eq(x, y).equals(x < y, True) == False
    assert Eq(x, f(1)).equals(Eq(x, f(2)), True) == f(1) - f(2)
    assert Eq(f(1), y).equals(Eq(f(2), y), True) == f(1) - f(2)
    assert Eq(x, f(1)).equals(Eq(f(2), x), True) == f(1) - f(2)
    assert Eq(f(1), x).equals(Eq(x, f(2)), True) == f(1) - f(2)
    assert Eq(w, x).equals(Eq(y, z), True) == False
    assert Eq(f(1), f(2)).equals(Eq(f(3), f(4)), True) == f(1) - f(3)
    assert (x < y).equals(y > x, True) == True
    assert (x < y).equals(y >= x, True) == False
    assert (x < y).equals(z < y, True) == False
    assert (x < y).equals(x < z, True) == False
    assert (x < f(1)).equals(x < f(2), True) == f(1) - f(2)
    assert (f(1) < x).equals(f(2) < x, True) == f(1) - f(2)


def test_reversed():
    assert (x < y).reversed == (y > x)
    assert (x <= y).reversed == (y >= x)
    assert Eq(x, y, evaluate=False).reversed == Eq(y, x, evaluate=False)
    assert Ne(x, y, evaluate=False).reversed == Ne(y, x, evaluate=False)
    assert (x >= y).reversed == (y <= x)
    assert (x > y).reversed == (y < x)


def test_canonical():
    c = [i.canonical for i in (
        x + y < z,
        x + 2 > 3,
        x < 2,
        S(2) > x,
        x**2 > -x/y,
        Gt(3, 2, evaluate=False)
        )]
    assert [i.canonical for i in c] == c
    assert [i.reversed.canonical for i in c] == c
    assert not any(i.lhs.is_Number and not i.rhs.is_Number for i in c)

    c = [i.reversed.func(i.rhs, i.lhs, evaluate=False).canonical for i in c]
    assert [i.canonical for i in c] == c
    assert [i.reversed.canonical for i in c] == c
    assert not any(i.lhs.is_Number and not i.rhs.is_Number for i in c)
    assert Eq(y < x, x > y).canonical is S.true


@XFAIL
def test_issue_8444_nonworkingtests():
    x = symbols('x', real=True)
    assert (x <= oo) == (x >= -oo) == True

    x = symbols('x')
    assert x >= floor(x)
    assert (x < floor(x)) == False
    assert x <= ceiling(x)
    assert (x > ceiling(x)) == False


def test_issue_8444_workingtests():
    x = symbols('x')
    assert Gt(x, floor(x)) == Gt(x, floor(x), evaluate=False)
    assert Ge(x, floor(x)) == Ge(x, floor(x), evaluate=False)
    assert Lt(x, ceiling(x)) == Lt(x, ceiling(x), evaluate=False)
    assert Le(x, ceiling(x)) == Le(x, ceiling(x), evaluate=False)
    i = symbols('i', integer=True)
    assert (i > floor(i)) == False
    assert (i < ceiling(i)) == False


def test_issue_10304():
    d = cos(1)**2 + sin(1)**2 - 1
    assert d.is_comparable is False  # if this fails, find a new d
    e = 1 + d*I
    assert simplify(Eq(e, 0)) is S.false


def test_issue_18412():
    d = (Rational(1, 6) + z / 4 / y)
    assert Eq(x, pi * y**3 * d).replace(y**3, z) == Eq(x, pi * z * d)


def test_issue_10401():
    x = symbols('x')
    fin = symbols('inf', finite=True)
    inf = symbols('inf', infinite=True)
    inf2 = symbols('inf2', infinite=True)
    infx = symbols('infx', infinite=True, extended_real=True)
    # Used in the commented tests below:
    #infx2 = symbols('infx2', infinite=True, extended_real=True)
    infnx = symbols('inf~x', infinite=True, extended_real=False)
    infnx2 = symbols('inf~x2', infinite=True, extended_real=False)
    infp = symbols('infp', infinite=True, extended_positive=True)
    infp1 = symbols('infp1', infinite=True, extended_positive=True)
    infn = symbols('infn', infinite=True, extended_negative=True)
    zero = symbols('z', zero=True)
    nonzero = symbols('nz', zero=False, finite=True)

    assert Eq(1/(1/x + 1), 1).func is Eq
    assert Eq(1/(1/x + 1), 1).subs(x, S.ComplexInfinity) is S.true
    assert Eq(1/(1/fin + 1), 1) is S.false

    T, F = S.true, S.false
    assert Eq(fin, inf) is F
    assert Eq(inf, inf2) not in (T, F) and inf != inf2
    assert Eq(1 + inf, 2 + inf2) not in (T, F) and inf != inf2
    assert Eq(infp, infp1) is T
    assert Eq(infp, infn) is F
    assert Eq(1 + I*oo, I*oo) is F
    assert Eq(I*oo, 1 + I*oo) is F
    assert Eq(1 + I*oo, 2 + I*oo) is F
    assert Eq(1 + I*oo, 2 + I*infx) is F
    assert Eq(1 + I*oo, 2 + infx) is F
    # FIXME: The test below fails because (-infx).is_extended_positive is True
    # (should be None)
    #assert Eq(1 + I*infx, 1 + I*infx2) not in (T, F) and infx != infx2
    #
    assert Eq(zoo, sqrt(2) + I*oo) is F
    assert Eq(zoo, oo) is F
    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)
    assert Eq(i*I, r) not in (T, F)
    assert Eq(infx, infnx) is F
    assert Eq(infnx, infnx2) not in (T, F) and infnx != infnx2
    assert Eq(zoo, oo) is F
    assert Eq(inf/inf2, 0) is F
    assert Eq(inf/fin, 0) is F
    assert Eq(fin/inf, 0) is T
    assert Eq(zero/nonzero, 0) is T and ((zero/nonzero) != 0)
    # The commented out test below is incorrect because:
    assert zoo == -zoo
    assert Eq(zoo, -zoo) is T
    assert Eq(oo, -oo) is F
    assert Eq(inf, -inf) not in (T, F)

    assert Eq(fin/(fin + 1), 1) is S.false

    o = symbols('o', odd=True)
    assert Eq(o, 2*o) is S.false

    p = symbols('p', positive=True)
    assert Eq(p/(p - 1), 1) is F


def test_issue_10633():
    assert Eq(True, False) == False
    assert Eq(False, True) == False
    assert Eq(True, True) == True
    assert Eq(False, False) == True


def test_issue_10927():
    x = symbols('x')
    assert str(Eq(x, oo)) == 'Eq(x, oo)'
    assert str(Eq(x, -oo)) == 'Eq(x, -oo)'


def test_issues_13081_12583_12534():
    # 13081
    r = Rational('905502432259640373/288230376151711744')
    assert (r < pi) is S.false
    assert (r > pi) is S.true
    # 12583
    v = sqrt(2)
    u = sqrt(v) + 2/sqrt(10 - 8/sqrt(2 - v) + 4*v*(1/sqrt(2 - v) - 1))
    assert (u >= 0) is S.true
    # 12534; Rational vs NumberSymbol
    # here are some precisions for which Rational forms
    # at a lower and higher precision bracket the value of pi
    # e.g. for p = 20:
    # Rational(pi.n(p + 1)).n(25) = 3.14159265358979323846 2834
    #                    pi.n(25) = 3.14159265358979323846 2643
    # Rational(pi.n(p    )).n(25) = 3.14159265358979323846 1987
    assert [p for p in range(20, 50) if
            (Rational(pi.n(p)) < pi) and
            (pi < Rational(pi.n(p + 1)))] == [20, 24, 27, 33, 37, 43, 48]
    # pick one such precision and affirm that the reversed operation
    # gives the opposite result, i.e. if x < y is true then x > y
    # must be false
    for i in (20, 21):
        v = pi.n(i)
        assert rel_check(Rational(v), pi)
        assert rel_check(v, pi)
    assert rel_check(pi.n(20), pi.n(21))
    # Float vs Rational
    # the rational form is less than the floating representation
    # at the same precision
    assert [i for i in range(15, 50) if Rational(pi.n(i)) > pi.n(i)] == []
    # this should be the same if we reverse the relational
    assert [i for i in range(15, 50) if pi.n(i) < Rational(pi.n(i))] == []

def test_issue_18188():
    from sympy.sets.conditionset import ConditionSet
    result1 = Eq(x*cos(x) - 3*sin(x), 0)
    assert result1.as_set() == ConditionSet(x, Eq(x*cos(x) - 3*sin(x), 0), Reals)

    result2 = Eq(x**2 + sqrt(x*2) + sin(x), 0)
    assert result2.as_set() == ConditionSet(x, Eq(sqrt(2)*sqrt(x) + x**2 + sin(x), 0), Reals)

def test_binary_symbols():
    ans = {x}
    for f in Eq, Ne:
        for t in S.true, S.false:
            eq = f(x, S.true)
            assert eq.binary_symbols == ans
            assert eq.reversed.binary_symbols == ans
        assert f(x, 1).binary_symbols == set()


def test_rel_args():
    # can't have Boolean args; this is automatic for True/False
    # with Python 3 and we confirm that SymPy does the same
    # for true/false
    for op in ['<', '<=', '>', '>=']:
        for b in (S.true, x < 1, And(x, y)):
            for v in (0.1, 1, 2**32, t, S.One):
                raises(TypeError, lambda: Relational(b, v, op))


def test_nothing_happens_to_Eq_condition_during_simplify():
    # issue 25701
    r = symbols('r', real=True)
    assert Eq(2*sign(r + 3)/(5*Abs(r + 3)**Rational(3, 5)), 0
        ).simplify() == Eq(Piecewise(
        (0, Eq(r, -3)), ((r + 3)/(5*Abs((r + 3)**Rational(8, 5)))*2, True)), 0)


def test_issue_15847():
    a = Ne(x*(x + y), x**2 + x*y)
    assert simplify(a) == False


def test_negated_property():
    eq = Eq(x, y)
    assert eq.negated == Ne(x, y)

    eq = Ne(x, y)
    assert eq.negated == Eq(x, y)

    eq = Ge(x + y, y - x)
    assert eq.negated == Lt(x + y, y - x)

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, y).negated.negated == f(x, y)


def test_reversedsign_property():
    eq = Eq(x, y)
    assert eq.reversedsign == Eq(-x, -y)

    eq = Ne(x, y)
    assert eq.reversedsign == Ne(-x, -y)

    eq = Ge(x + y, y - x)
    assert eq.reversedsign == Le(-x - y, x - y)

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, y).reversedsign.reversedsign == f(x, y)

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(-x, y).reversedsign.reversedsign == f(-x, y)

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, -y).reversedsign.reversedsign == f(x, -y)

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(-x, -y).reversedsign.reversedsign == f(-x, -y)


def test_reversed_reversedsign_property():
    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, y).reversed.reversedsign == f(x, y).reversedsign.reversed

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(-x, y).reversed.reversedsign == f(-x, y).reversedsign.reversed

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(x, -y).reversed.reversedsign == f(x, -y).reversedsign.reversed

    for f in (Eq, Ne, Ge, Gt, Le, Lt):
        assert f(-x, -y).reversed.reversedsign == \
            f(-x, -y).reversedsign.reversed


def test_improved_canonical():
    def test_different_forms(listofforms):
        for form1, form2 in combinations(listofforms, 2):
            assert form1.canonical == form2.canonical

    def generate_forms(expr):
        return [expr, expr.reversed, expr.reversedsign,
                expr.reversed.reversedsign]

    test_different_forms(generate_forms(x > -y))
    test_different_forms(generate_forms(x >= -y))
    test_different_forms(generate_forms(Eq(x, -y)))
    test_different_forms(generate_forms(Ne(x, -y)))
    test_different_forms(generate_forms(pi < x))
    test_different_forms(generate_forms(pi - 5*y < -x + 2*y**2 - 7))

    assert (pi >= x).canonical == (x <= pi)


def test_set_equality_canonical():
    a, b, c = symbols('a b c')

    A = Eq(FiniteSet(a, b, c), FiniteSet(1, 2, 3))
    B = Ne(FiniteSet(a, b, c), FiniteSet(4, 5, 6))

    assert A.canonical == A.reversed
    assert B.canonical == B.reversed


def test_trigsimp():
    # issue 16736
    s, c = sin(2*x), cos(2*x)
    eq = Eq(s, c)
    assert trigsimp(eq) == eq  # no rearrangement of sides
    # simplification of sides might result in
    # an unevaluated Eq
    changed = trigsimp(Eq(s + c, sqrt(2)))
    assert isinstance(changed, Eq)
    assert changed.subs(x, pi/8) is S.true
    # or an evaluated one
    assert trigsimp(Eq(cos(x)**2 + sin(x)**2, 1)) is S.true


def test_polynomial_relation_simplification():
    assert Ge(3*x*(x + 1) + 4, 3*x).simplify() in [Ge(x**2, -Rational(4,3)), Le(-x**2, Rational(4, 3))]
    assert Le(-(3*x*(x + 1) + 4), -3*x).simplify() in [Ge(x**2, -Rational(4,3)), Le(-x**2, Rational(4, 3))]
    assert ((x**2+3)*(x**2-1)+3*x >= 2*x**2).simplify() in [(x**4 + 3*x >= 3), (-x**4 - 3*x <= -3)]


def test_multivariate_linear_function_simplification():
    assert Ge(x + y, x - y).simplify() == Ge(y, 0)
    assert Le(-x + y, -x - y).simplify() == Le(y, 0)
    assert Eq(2*x + y, 2*x + y - 3).simplify() == False
    assert (2*x + y > 2*x + y - 3).simplify() == True
    assert (2*x + y < 2*x + y - 3).simplify() == False
    assert (2*x + y < 2*x + y + 3).simplify() == True
    a, b, c, d, e, f, g = symbols('a b c d e f g')
    assert Lt(a + b + c + 2*d, 3*d - f + g). simplify() == Lt(a, -b - c + d - f + g)


def test_nonpolymonial_relations():
    assert Eq(cos(x), 0).simplify() == Eq(cos(x), 0)

def test_18778():
    raises(TypeError, lambda: is_le(Basic(), Basic()))
    raises(TypeError, lambda: is_gt(Basic(), Basic()))
    raises(TypeError, lambda: is_ge(Basic(), Basic()))
    raises(TypeError, lambda: is_lt(Basic(), Basic()))

def test_EvalEq():
    """

    This test exists to ensure backwards compatibility.
    The method to use is _eval_is_eq
    """
    from sympy.core.expr import Expr

    class PowTest(Expr):
        def __new__(cls, base, exp):
            return Basic.__new__(PowTest, _sympify(base), _sympify(exp))

        def _eval_Eq(lhs, rhs):
            if type(lhs) == PowTest and type(rhs) == PowTest:
                return lhs.args[0] == rhs.args[0] and lhs.args[1] == rhs.args[1]

    assert is_eq(PowTest(3, 4), PowTest(3,4))
    assert is_eq(PowTest(3, 4), _sympify(4)) is None
    assert is_neq(PowTest(3, 4), PowTest(3,7))


def test_is_eq():
    # test assumptions
    assert is_eq(x, y, Q.infinite(x) & Q.finite(y)) is False
    assert is_eq(x, y, Q.infinite(x) & Q.infinite(y) & Q.extended_real(x) & ~Q.extended_real(y)) is False
    assert is_eq(x, y, Q.infinite(x) & Q.infinite(y) & Q.extended_positive(x) & Q.extended_negative(y)) is False

    assert is_eq(x+I, y+I, Q.infinite(x) & Q.finite(y)) is False
    assert is_eq(1+x*I, 1+y*I, Q.infinite(x) & Q.finite(y)) is False

    assert is_eq(x, S(0), assumptions=Q.zero(x))
    assert is_eq(x, S(0), assumptions=~Q.zero(x)) is False
    assert is_eq(x, S(0), assumptions=Q.nonzero(x)) is False
    assert is_neq(x, S(0), assumptions=Q.zero(x)) is False
    assert is_neq(x, S(0), assumptions=~Q.zero(x))
    assert is_neq(x, S(0), assumptions=Q.nonzero(x))

    # test registration
    class PowTest(Expr):
        def __new__(cls, base, exp):
            return Basic.__new__(cls, _sympify(base), _sympify(exp))

    @dispatch(PowTest, PowTest)
    def _eval_is_eq(lhs, rhs):
        if type(lhs) == PowTest and type(rhs) == PowTest:
            return fuzzy_and([is_eq(lhs.args[0], rhs.args[0]), is_eq(lhs.args[1], rhs.args[1])])

    assert is_eq(PowTest(3, 4), PowTest(3,4))
    assert is_eq(PowTest(3, 4), _sympify(4)) is None
    assert is_neq(PowTest(3, 4), PowTest(3,7))


def test_is_ge_le():
    # test assumptions
    assert is_ge(x, S(0), Q.nonnegative(x)) is True
    assert is_ge(x, S(0), Q.negative(x)) is False

    # test registration
    class PowTest(Expr):
        def __new__(cls, base, exp):
            return Basic.__new__(cls, _sympify(base), _sympify(exp))

    @dispatch(PowTest, PowTest)
    def _eval_is_ge(lhs, rhs):
        if type(lhs) == PowTest and type(rhs) == PowTest:
            return fuzzy_and([is_ge(lhs.args[0], rhs.args[0]), is_ge(lhs.args[1], rhs.args[1])])

    assert is_ge(PowTest(3, 9), PowTest(3,2))
    assert is_gt(PowTest(3, 9), PowTest(3,2))
    assert is_le(PowTest(3, 2), PowTest(3,9))
    assert is_lt(PowTest(3, 2), PowTest(3,9))


def test_weak_strict():
    for func in (Eq, Ne):
        eq = func(x, 1)
        assert eq.strict == eq.weak == eq
    eq = Gt(x, 1)
    assert eq.weak == Ge(x, 1)
    assert eq.strict == eq
    eq = Lt(x, 1)
    assert eq.weak == Le(x, 1)
    assert eq.strict == eq
    eq = Ge(x, 1)
    assert eq.strict == Gt(x, 1)
    assert eq.weak == eq
    eq = Le(x, 1)
    assert eq.strict == Lt(x, 1)
    assert eq.weak == eq


def test_issue_23731():
    i = symbols('i', integer=True)
    assert unchanged(Eq, i, 1.0)
    assert unchanged(Eq, i/2, 0.5)
    ni = symbols('ni', integer=False)
    assert Eq(ni, 1) == False
    assert unchanged(Eq, ni, .1)
    assert Eq(ni, 1.0) == False
    nr = symbols('nr', rational=False)
    assert Eq(nr, .1) == False


def test_rewrite_Add():
    from sympy.testing.pytest import warns_deprecated_sympy
    with warns_deprecated_sympy():
        assert Eq(x, y).rewrite(Add) == x - y

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_indexing.py ===
import numpy as np
import pytest

from pandas.errors import SettingWithCopyWarning

from pandas.core.dtypes.common import is_float_dtype

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


@pytest.fixture(params=["numpy", "nullable"])
def backend(request):
    if request.param == "numpy":

        def make_dataframe(*args, **kwargs):
            return DataFrame(*args, **kwargs)

        def make_series(*args, **kwargs):
            return Series(*args, **kwargs)

    elif request.param == "nullable":

        def make_dataframe(*args, **kwargs):
            df = DataFrame(*args, **kwargs)
            df_nullable = df.convert_dtypes()
            # convert_dtypes will try to cast float to int if there is no loss in
            # precision -> undo that change
            for col in df.columns:
                if is_float_dtype(df[col].dtype) and not is_float_dtype(
                    df_nullable[col].dtype
                ):
                    df_nullable[col] = df_nullable[col].astype("Float64")
            # copy final result to ensure we start with a fully self-owning DataFrame
            return df_nullable.copy()

        def make_series(*args, **kwargs):
            ser = Series(*args, **kwargs)
            return ser.convert_dtypes().copy()

    return request.param, make_dataframe, make_series


# -----------------------------------------------------------------------------
# Indexing operations taking subset + modifying the subset/parent


def test_subset_column_selection(backend, using_copy_on_write):
    # Case: taking a subset of the columns of a DataFrame
    # + afterwards modifying the subset
    _, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    subset = df[["a", "c"]]

    if using_copy_on_write:
        # the subset shares memory ...
        assert np.shares_memory(get_array(subset, "a"), get_array(df, "a"))
        # ... but uses CoW when being modified
        subset.iloc[0, 0] = 0
    else:
        assert not np.shares_memory(get_array(subset, "a"), get_array(df, "a"))
        # INFO this no longer raise warning since pandas 1.4
        # with pd.option_context("chained_assignment", "warn"):
        #     with tm.assert_produces_warning(SettingWithCopyWarning):
        subset.iloc[0, 0] = 0

    assert not np.shares_memory(get_array(subset, "a"), get_array(df, "a"))

    expected = DataFrame({"a": [0, 2, 3], "c": [0.1, 0.2, 0.3]})
    tm.assert_frame_equal(subset, expected)
    tm.assert_frame_equal(df, df_orig)


def test_subset_column_selection_modify_parent(backend, using_copy_on_write):
    # Case: taking a subset of the columns of a DataFrame
    # + afterwards modifying the parent
    _, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})

    subset = df[["a", "c"]]

    if using_copy_on_write:
        # the subset shares memory ...
        assert np.shares_memory(get_array(subset, "a"), get_array(df, "a"))
        # ... but parent uses CoW parent when it is modified
    df.iloc[0, 0] = 0

    assert not np.shares_memory(get_array(subset, "a"), get_array(df, "a"))
    if using_copy_on_write:
        # different column/block still shares memory
        assert np.shares_memory(get_array(subset, "c"), get_array(df, "c"))

    expected = DataFrame({"a": [1, 2, 3], "c": [0.1, 0.2, 0.3]})
    tm.assert_frame_equal(subset, expected)


def test_subset_row_slice(backend, using_copy_on_write, warn_copy_on_write):
    # Case: taking a subset of the rows of a DataFrame using a slice
    # + afterwards modifying the subset
    _, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    subset = df[1:3]
    subset._mgr._verify_integrity()

    assert np.shares_memory(get_array(subset, "a"), get_array(df, "a"))

    if using_copy_on_write:
        subset.iloc[0, 0] = 0
        assert not np.shares_memory(get_array(subset, "a"), get_array(df, "a"))

    else:
        # INFO this no longer raise warning since pandas 1.4
        # with pd.option_context("chained_assignment", "warn"):
        #     with tm.assert_produces_warning(SettingWithCopyWarning):
        with tm.assert_cow_warning(warn_copy_on_write):
            subset.iloc[0, 0] = 0

    subset._mgr._verify_integrity()

    expected = DataFrame({"a": [0, 3], "b": [5, 6], "c": [0.2, 0.3]}, index=range(1, 3))
    tm.assert_frame_equal(subset, expected)
    if using_copy_on_write:
        # original parent dataframe is not modified (CoW)
        tm.assert_frame_equal(df, df_orig)
    else:
        # original parent dataframe is actually updated
        df_orig.iloc[1, 0] = 0
        tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_subset_column_slice(
    backend, using_copy_on_write, warn_copy_on_write, using_array_manager, dtype
):
    # Case: taking a subset of the columns of a DataFrame using a slice
    # + afterwards modifying the subset
    dtype_backend, DataFrame, _ = backend
    single_block = (
        dtype == "int64" and dtype_backend == "numpy"
    ) and not using_array_manager
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    df_orig = df.copy()

    subset = df.iloc[:, 1:]
    subset._mgr._verify_integrity()

    if using_copy_on_write:
        assert np.shares_memory(get_array(subset, "b"), get_array(df, "b"))

        subset.iloc[0, 0] = 0
        assert not np.shares_memory(get_array(subset, "b"), get_array(df, "b"))
    elif warn_copy_on_write:
        with tm.assert_cow_warning(single_block):
            subset.iloc[0, 0] = 0
    else:
        # we only get a warning in case of a single block
        warn = SettingWithCopyWarning if single_block else None
        with pd.option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(warn):
                subset.iloc[0, 0] = 0

    expected = DataFrame({"b": [0, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)})
    tm.assert_frame_equal(subset, expected)
    # original parent dataframe is not modified (also not for BlockManager case,
    # except for single block)
    if not using_copy_on_write and (using_array_manager or single_block):
        df_orig.iloc[0, 1] = 0
        tm.assert_frame_equal(df, df_orig)
    else:
        tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
@pytest.mark.parametrize(
    "row_indexer",
    [slice(1, 2), np.array([False, True, True]), np.array([1, 2])],
    ids=["slice", "mask", "array"],
)
@pytest.mark.parametrize(
    "column_indexer",
    [slice("b", "c"), np.array([False, True, True]), ["b", "c"]],
    ids=["slice", "mask", "array"],
)
def test_subset_loc_rows_columns(
    backend,
    dtype,
    row_indexer,
    column_indexer,
    using_array_manager,
    using_copy_on_write,
    warn_copy_on_write,
):
    # Case: taking a subset of the rows+columns of a DataFrame using .loc
    # + afterwards modifying the subset
    # Generic test for several combinations of row/column indexers, not all
    # of those could actually return a view / need CoW (so this test is not
    # checking memory sharing, only ensuring subsequent mutation doesn't
    # affect the parent dataframe)
    dtype_backend, DataFrame, _ = backend
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    df_orig = df.copy()

    subset = df.loc[row_indexer, column_indexer]

    # a few corner cases _do_ actually modify the parent (with both row and column
    # slice, and in case of ArrayManager or BlockManager with single block)
    mutate_parent = (
        isinstance(row_indexer, slice)
        and isinstance(column_indexer, slice)
        and (
            using_array_manager
            or (
                dtype == "int64"
                and dtype_backend == "numpy"
                and not using_copy_on_write
            )
        )
    )

    # modifying the subset never modifies the parent
    with tm.assert_cow_warning(warn_copy_on_write and mutate_parent):
        subset.iloc[0, 0] = 0

    expected = DataFrame(
        {"b": [0, 6], "c": np.array([8, 9], dtype=dtype)}, index=range(1, 3)
    )
    tm.assert_frame_equal(subset, expected)
    if mutate_parent:
        df_orig.iloc[1, 1] = 0
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
@pytest.mark.parametrize(
    "row_indexer",
    [slice(1, 3), np.array([False, True, True]), np.array([1, 2])],
    ids=["slice", "mask", "array"],
)
@pytest.mark.parametrize(
    "column_indexer",
    [slice(1, 3), np.array([False, True, True]), [1, 2]],
    ids=["slice", "mask", "array"],
)
def test_subset_iloc_rows_columns(
    backend,
    dtype,
    row_indexer,
    column_indexer,
    using_array_manager,
    using_copy_on_write,
    warn_copy_on_write,
):
    # Case: taking a subset of the rows+columns of a DataFrame using .iloc
    # + afterwards modifying the subset
    # Generic test for several combinations of row/column indexers, not all
    # of those could actually return a view / need CoW (so this test is not
    # checking memory sharing, only ensuring subsequent mutation doesn't
    # affect the parent dataframe)
    dtype_backend, DataFrame, _ = backend
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    df_orig = df.copy()

    subset = df.iloc[row_indexer, column_indexer]

    # a few corner cases _do_ actually modify the parent (with both row and column
    # slice, and in case of ArrayManager or BlockManager with single block)
    mutate_parent = (
        isinstance(row_indexer, slice)
        and isinstance(column_indexer, slice)
        and (
            using_array_manager
            or (
                dtype == "int64"
                and dtype_backend == "numpy"
                and not using_copy_on_write
            )
        )
    )

    # modifying the subset never modifies the parent
    with tm.assert_cow_warning(warn_copy_on_write and mutate_parent):
        subset.iloc[0, 0] = 0

    expected = DataFrame(
        {"b": [0, 6], "c": np.array([8, 9], dtype=dtype)}, index=range(1, 3)
    )
    tm.assert_frame_equal(subset, expected)
    if mutate_parent:
        df_orig.iloc[1, 1] = 0
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "indexer",
    [slice(0, 2), np.array([True, True, False]), np.array([0, 1])],
    ids=["slice", "mask", "array"],
)
def test_subset_set_with_row_indexer(
    backend, indexer_si, indexer, using_copy_on_write, warn_copy_on_write
):
    # Case: setting values with a row indexer on a viewing subset
    # subset[indexer] = value and subset.iloc[indexer] = value
    _, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7], "c": [0.1, 0.2, 0.3, 0.4]})
    df_orig = df.copy()
    subset = df[1:4]

    if (
        indexer_si is tm.setitem
        and isinstance(indexer, np.ndarray)
        and indexer.dtype == "int"
    ):
        pytest.skip("setitem with labels selects on columns")

    if using_copy_on_write:
        indexer_si(subset)[indexer] = 0
    elif warn_copy_on_write:
        with tm.assert_cow_warning():
            indexer_si(subset)[indexer] = 0
    else:
        # INFO iloc no longer raises warning since pandas 1.4
        warn = SettingWithCopyWarning if indexer_si is tm.setitem else None
        with pd.option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(warn):
                indexer_si(subset)[indexer] = 0

    expected = DataFrame(
        {"a": [0, 0, 4], "b": [0, 0, 7], "c": [0.0, 0.0, 0.4]}, index=range(1, 4)
    )
    tm.assert_frame_equal(subset, expected)
    if using_copy_on_write:
        # original parent dataframe is not modified (CoW)
        tm.assert_frame_equal(df, df_orig)
    else:
        # original parent dataframe is actually updated
        df_orig[1:3] = 0
        tm.assert_frame_equal(df, df_orig)


def test_subset_set_with_mask(backend, using_copy_on_write, warn_copy_on_write):
    # Case: setting values with a mask on a viewing subset: subset[mask] = value
    _, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7], "c": [0.1, 0.2, 0.3, 0.4]})
    df_orig = df.copy()
    subset = df[1:4]

    mask = subset > 3

    if using_copy_on_write:
        subset[mask] = 0
    elif warn_copy_on_write:
        with tm.assert_cow_warning():
            subset[mask] = 0
    else:
        with pd.option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(SettingWithCopyWarning):
                subset[mask] = 0

    expected = DataFrame(
        {"a": [2, 3, 0], "b": [0, 0, 0], "c": [0.20, 0.3, 0.4]}, index=range(1, 4)
    )
    tm.assert_frame_equal(subset, expected)
    if using_copy_on_write:
        # original parent dataframe is not modified (CoW)
        tm.assert_frame_equal(df, df_orig)
    else:
        # original parent dataframe is actually updated
        df_orig.loc[3, "a"] = 0
        df_orig.loc[1:3, "b"] = 0
        tm.assert_frame_equal(df, df_orig)


def test_subset_set_column(backend, using_copy_on_write, warn_copy_on_write):
    # Case: setting a single column on a viewing subset -> subset[col] = value
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    subset = df[1:3]

    if dtype_backend == "numpy":
        arr = np.array([10, 11], dtype="int64")
    else:
        arr = pd.array([10, 11], dtype="Int64")

    if using_copy_on_write or warn_copy_on_write:
        subset["a"] = arr
    else:
        with pd.option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(SettingWithCopyWarning):
                subset["a"] = arr

    subset._mgr._verify_integrity()
    expected = DataFrame(
        {"a": [10, 11], "b": [5, 6], "c": [0.2, 0.3]}, index=range(1, 3)
    )
    tm.assert_frame_equal(subset, expected)
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_subset_set_column_with_loc(
    backend, using_copy_on_write, warn_copy_on_write, using_array_manager, dtype
):
    # Case: setting a single column with loc on a viewing subset
    # -> subset.loc[:, col] = value
    _, DataFrame, _ = backend
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    df_orig = df.copy()
    subset = df[1:3]

    if using_copy_on_write:
        subset.loc[:, "a"] = np.array([10, 11], dtype="int64")
    elif warn_copy_on_write:
        with tm.assert_cow_warning():
            subset.loc[:, "a"] = np.array([10, 11], dtype="int64")
    else:
        with pd.option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(
                None,
                raise_on_extra_warnings=not using_array_manager,
            ):
                subset.loc[:, "a"] = np.array([10, 11], dtype="int64")

    subset._mgr._verify_integrity()
    expected = DataFrame(
        {"a": [10, 11], "b": [5, 6], "c": np.array([8, 9], dtype=dtype)},
        index=range(1, 3),
    )
    tm.assert_frame_equal(subset, expected)
    if using_copy_on_write:
        # original parent dataframe is not modified (CoW)
        tm.assert_frame_equal(df, df_orig)
    else:
        # original parent dataframe is actually updated
        df_orig.loc[1:3, "a"] = np.array([10, 11], dtype="int64")
        tm.assert_frame_equal(df, df_orig)


def test_subset_set_column_with_loc2(
    backend, using_copy_on_write, warn_copy_on_write, using_array_manager
):
    # Case: setting a single column with loc on a viewing subset
    # -> subset.loc[:, col] = value
    # separate test for case of DataFrame of a single column -> takes a separate
    # code path
    _, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3]})
    df_orig = df.copy()
    subset = df[1:3]

    if using_copy_on_write:
        subset.loc[:, "a"] = 0
    elif warn_copy_on_write:
        with tm.assert_cow_warning():
            subset.loc[:, "a"] = 0
    else:
        with pd.option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(
                None,
                raise_on_extra_warnings=not using_array_manager,
            ):
                subset.loc[:, "a"] = 0

    subset._mgr._verify_integrity()
    expected = DataFrame({"a": [0, 0]}, index=range(1, 3))
    tm.assert_frame_equal(subset, expected)
    if using_copy_on_write:
        # original parent dataframe is not modified (CoW)
        tm.assert_frame_equal(df, df_orig)
    else:
        # original parent dataframe is actually updated
        df_orig.loc[1:3, "a"] = 0
        tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_subset_set_columns(backend, using_copy_on_write, warn_copy_on_write, dtype):
    # Case: setting multiple columns on a viewing subset
    # -> subset[[col1, col2]] = value
    dtype_backend, DataFrame, _ = backend
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    df_orig = df.copy()
    subset = df[1:3]

    if using_copy_on_write or warn_copy_on_write:
        subset[["a", "c"]] = 0
    else:
        with pd.option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(SettingWithCopyWarning):
                subset[["a", "c"]] = 0

    subset._mgr._verify_integrity()
    if using_copy_on_write:
        # first and third column should certainly have no references anymore
        assert all(subset._mgr._has_no_reference(i) for i in [0, 2])
    expected = DataFrame({"a": [0, 0], "b": [5, 6], "c": [0, 0]}, index=range(1, 3))
    if dtype_backend == "nullable":
        # there is not yet a global option, so overriding a column by setting a scalar
        # defaults to numpy dtype even if original column was nullable
        expected["a"] = expected["a"].astype("int64")
        expected["c"] = expected["c"].astype("int64")

    tm.assert_frame_equal(subset, expected)
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "indexer",
    [slice("a", "b"), np.array([True, True, False]), ["a", "b"]],
    ids=["slice", "mask", "array"],
)
def test_subset_set_with_column_indexer(
    backend, indexer, using_copy_on_write, warn_copy_on_write
):
    # Case: setting multiple columns with a column indexer on a viewing subset
    # -> subset.loc[:, [col1, col2]] = value
    _, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": [4, 5, 6]})
    df_orig = df.copy()
    subset = df[1:3]

    if using_copy_on_write:
        subset.loc[:, indexer] = 0
    elif warn_copy_on_write:
        with tm.assert_cow_warning():
            subset.loc[:, indexer] = 0
    else:
        with pd.option_context("chained_assignment", "warn"):
            # As of 2.0, this setitem attempts (successfully) to set values
            #  inplace, so the assignment is not chained.
            subset.loc[:, indexer] = 0

    subset._mgr._verify_integrity()
    expected = DataFrame({"a": [0, 0], "b": [0.0, 0.0], "c": [5, 6]}, index=range(1, 3))
    tm.assert_frame_equal(subset, expected)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        # pre-2.0, in the mixed case with BlockManager, only column "a"
        #  would be mutated in the parent frame. this changed with the
        #  enforcement of GH#45333
        df_orig.loc[1:2, ["a", "b"]] = 0
        tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "method",
    [
        lambda df: df[["a", "b"]][0:2],
        lambda df: df[0:2][["a", "b"]],
        lambda df: df[["a", "b"]].iloc[0:2],
        lambda df: df[["a", "b"]].loc[0:1],
        lambda df: df[0:2].iloc[:, 0:2],
        lambda df: df[0:2].loc[:, "a":"b"],  # type: ignore[misc]
    ],
    ids=[
        "row-getitem-slice",
        "column-getitem",
        "row-iloc-slice",
        "row-loc-slice",
        "column-iloc-slice",
        "column-loc-slice",
    ],
)
@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_subset_chained_getitem(
    request,
    backend,
    method,
    dtype,
    using_copy_on_write,
    using_array_manager,
    warn_copy_on_write,
):
    # Case: creating a subset using multiple, chained getitem calls using views
    # still needs to guarantee proper CoW behaviour
    _, DataFrame, _ = backend
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    df_orig = df.copy()

    # when not using CoW, it depends on whether we have a single block or not
    # and whether we are slicing the columns -> in that case we have a view
    test_callspec = request.node.callspec.id
    if not using_array_manager:
        subset_is_view = test_callspec in (
            "numpy-single-block-column-iloc-slice",
            "numpy-single-block-column-loc-slice",
        )
    else:
        # with ArrayManager, it doesn't matter whether we have
        # single vs mixed block or numpy vs nullable dtypes
        subset_is_view = test_callspec.endswith(
            ("column-iloc-slice", "column-loc-slice")
        )

    # modify subset -> don't modify parent
    subset = method(df)

    with tm.assert_cow_warning(warn_copy_on_write and subset_is_view):
        subset.iloc[0, 0] = 0
    if using_copy_on_write or (not subset_is_view):
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.iloc[0, 0] == 0

    # modify parent -> don't modify subset
    subset = method(df)
    with tm.assert_cow_warning(warn_copy_on_write and subset_is_view):
        df.iloc[0, 0] = 0
    expected = DataFrame({"a": [1, 2], "b": [4, 5]})
    if using_copy_on_write or not subset_is_view:
        tm.assert_frame_equal(subset, expected)
    else:
        assert subset.iloc[0, 0] == 0


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_subset_chained_getitem_column(
    backend, dtype, using_copy_on_write, warn_copy_on_write
):
    # Case: creating a subset using multiple, chained getitem calls using views
    # still needs to guarantee proper CoW behaviour
    dtype_backend, DataFrame, Series = backend
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    df_orig = df.copy()

    # modify subset -> don't modify parent
    subset = df[:]["a"][0:2]
    df._clear_item_cache()
    with tm.assert_cow_warning(warn_copy_on_write):
        subset.iloc[0] = 0
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.iloc[0, 0] == 0

    # modify parent -> don't modify subset
    subset = df[:]["a"][0:2]
    df._clear_item_cache()
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 0
    expected = Series([1, 2], name="a")
    if using_copy_on_write:
        tm.assert_series_equal(subset, expected)
    else:
        assert subset.iloc[0] == 0


@pytest.mark.parametrize(
    "method",
    [
        lambda s: s["a":"c"]["a":"b"],  # type: ignore[misc]
        lambda s: s.iloc[0:3].iloc[0:2],
        lambda s: s.loc["a":"c"].loc["a":"b"],  # type: ignore[misc]
        lambda s: s.loc["a":"c"]  # type: ignore[misc]
        .iloc[0:3]
        .iloc[0:2]
        .loc["a":"b"]  # type: ignore[misc]
        .iloc[0:1],
    ],
    ids=["getitem", "iloc", "loc", "long-chain"],
)
def test_subset_chained_getitem_series(
    backend, method, using_copy_on_write, warn_copy_on_write
):
    # Case: creating a subset using multiple, chained getitem calls using views
    # still needs to guarantee proper CoW behaviour
    _, _, Series = backend
    s = Series([1, 2, 3], index=["a", "b", "c"])
    s_orig = s.copy()

    # modify subset -> don't modify parent
    subset = method(s)
    with tm.assert_cow_warning(warn_copy_on_write):
        subset.iloc[0] = 0
    if using_copy_on_write:
        tm.assert_series_equal(s, s_orig)
    else:
        assert s.iloc[0] == 0

    # modify parent -> don't modify subset
    subset = s.iloc[0:3].iloc[0:2]
    with tm.assert_cow_warning(warn_copy_on_write):
        s.iloc[0] = 0
    expected = Series([1, 2], index=["a", "b"])
    if using_copy_on_write:
        tm.assert_series_equal(subset, expected)
    else:
        assert subset.iloc[0] == 0


def test_subset_chained_single_block_row(
    using_copy_on_write, using_array_manager, warn_copy_on_write
):
    # not parametrizing this for dtype backend, since this explicitly tests single block
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df_orig = df.copy()

    # modify subset -> don't modify parent
    subset = df[:].iloc[0].iloc[0:2]
    with tm.assert_cow_warning(warn_copy_on_write):
        subset.iloc[0] = 0
    if using_copy_on_write or using_array_manager:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.iloc[0, 0] == 0

    # modify parent -> don't modify subset
    subset = df[:].iloc[0].iloc[0:2]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 0
    expected = Series([1, 4], index=["a", "b"], name=0)
    if using_copy_on_write or using_array_manager:
        tm.assert_series_equal(subset, expected)
    else:
        assert subset.iloc[0] == 0


@pytest.mark.parametrize(
    "method",
    [
        lambda df: df[:],
        lambda df: df.loc[:, :],
        lambda df: df.loc[:],
        lambda df: df.iloc[:, :],
        lambda df: df.iloc[:],
    ],
    ids=["getitem", "loc", "loc-rows", "iloc", "iloc-rows"],
)
def test_null_slice(backend, method, using_copy_on_write, warn_copy_on_write):
    # Case: also all variants of indexing with a null slice (:) should return
    # new objects to ensure we correctly use CoW for the results
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df_orig = df.copy()

    df2 = method(df)

    # we always return new objects (shallow copy), regardless of CoW or not
    assert df2 is not df

    # and those trigger CoW when mutated
    with tm.assert_cow_warning(warn_copy_on_write):
        df2.iloc[0, 0] = 0
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.iloc[0, 0] == 0


@pytest.mark.parametrize(
    "method",
    [
        lambda s: s[:],
        lambda s: s.loc[:],
        lambda s: s.iloc[:],
    ],
    ids=["getitem", "loc", "iloc"],
)
def test_null_slice_series(backend, method, using_copy_on_write, warn_copy_on_write):
    _, _, Series = backend
    s = Series([1, 2, 3], index=["a", "b", "c"])
    s_orig = s.copy()

    s2 = method(s)

    # we always return new objects, regardless of CoW or not
    assert s2 is not s

    # and those trigger CoW when mutated
    with tm.assert_cow_warning(warn_copy_on_write):
        s2.iloc[0] = 0
    if using_copy_on_write:
        tm.assert_series_equal(s, s_orig)
    else:
        assert s.iloc[0] == 0


# TODO add more tests modifying the parent


# -----------------------------------------------------------------------------
# Series -- Indexing operations taking subset + modifying the subset/parent


def test_series_getitem_slice(backend, using_copy_on_write, warn_copy_on_write):
    # Case: taking a slice of a Series + afterwards modifying the subset
    _, _, Series = backend
    s = Series([1, 2, 3], index=["a", "b", "c"])
    s_orig = s.copy()

    subset = s[:]
    assert np.shares_memory(get_array(subset), get_array(s))

    with tm.assert_cow_warning(warn_copy_on_write):
        subset.iloc[0] = 0

    if using_copy_on_write:
        assert not np.shares_memory(get_array(subset), get_array(s))

    expected = Series([0, 2, 3], index=["a", "b", "c"])
    tm.assert_series_equal(subset, expected)

    if using_copy_on_write:
        # original parent series is not modified (CoW)
        tm.assert_series_equal(s, s_orig)
    else:
        # original parent series is actually updated
        assert s.iloc[0] == 0


def test_series_getitem_ellipsis(using_copy_on_write, warn_copy_on_write):
    # Case: taking a view of a Series using Ellipsis + afterwards modifying the subset
    s = Series([1, 2, 3])
    s_orig = s.copy()

    subset = s[...]
    assert np.shares_memory(get_array(subset), get_array(s))

    with tm.assert_cow_warning(warn_copy_on_write):
        subset.iloc[0] = 0

    if using_copy_on_write:
        assert not np.shares_memory(get_array(subset), get_array(s))

    expected = Series([0, 2, 3])
    tm.assert_series_equal(subset, expected)

    if using_copy_on_write:
        # original parent series is not modified (CoW)
        tm.assert_series_equal(s, s_orig)
    else:
        # original parent series is actually updated
        assert s.iloc[0] == 0


@pytest.mark.parametrize(
    "indexer",
    [slice(0, 2), np.array([True, True, False]), np.array([0, 1])],
    ids=["slice", "mask", "array"],
)
def test_series_subset_set_with_indexer(
    backend, indexer_si, indexer, using_copy_on_write, warn_copy_on_write
):
    # Case: setting values in a viewing Series with an indexer
    _, _, Series = backend
    s = Series([1, 2, 3], index=["a", "b", "c"])
    s_orig = s.copy()
    subset = s[:]

    warn = None
    msg = "Series.__setitem__ treating keys as positions is deprecated"
    if (
        indexer_si is tm.setitem
        and isinstance(indexer, np.ndarray)
        and indexer.dtype.kind == "i"
    ):
        warn = FutureWarning
    if warn_copy_on_write:
        with tm.assert_cow_warning(raise_on_extra_warnings=warn is not None):
            indexer_si(subset)[indexer] = 0
    else:
        with tm.assert_produces_warning(warn, match=msg):
            indexer_si(subset)[indexer] = 0
    expected = Series([0, 0, 3], index=["a", "b", "c"])
    tm.assert_series_equal(subset, expected)

    if using_copy_on_write:
        tm.assert_series_equal(s, s_orig)
    else:
        tm.assert_series_equal(s, expected)


# -----------------------------------------------------------------------------
# del operator


def test_del_frame(backend, using_copy_on_write, warn_copy_on_write):
    # Case: deleting a column with `del` on a viewing child dataframe should
    # not modify parent + update the references
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df[:]

    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    del df2["b"]

    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    tm.assert_frame_equal(df, df_orig)
    tm.assert_frame_equal(df2, df_orig[["a", "c"]])
    df2._mgr._verify_integrity()

    with tm.assert_cow_warning(warn_copy_on_write and dtype_backend == "numpy"):
        df.loc[0, "b"] = 200
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    df_orig = df.copy()

    with tm.assert_cow_warning(warn_copy_on_write):
        df2.loc[0, "a"] = 100
    if using_copy_on_write:
        # modifying child after deleting a column still doesn't update parent
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.loc[0, "a"] == 100


def test_del_series(backend):
    _, _, Series = backend
    s = Series([1, 2, 3], index=["a", "b", "c"])
    s_orig = s.copy()
    s2 = s[:]

    assert np.shares_memory(get_array(s), get_array(s2))

    del s2["a"]

    assert not np.shares_memory(get_array(s), get_array(s2))
    tm.assert_series_equal(s, s_orig)
    tm.assert_series_equal(s2, s_orig[["b", "c"]])

    # modifying s2 doesn't need copy on write (due to `del`, s2 is backed by new array)
    values = s2.values
    s2.loc["b"] = 100
    assert values[0] == 100


# -----------------------------------------------------------------------------
# Accessing column as Series


def test_column_as_series(
    backend, using_copy_on_write, warn_copy_on_write, using_array_manager
):
    # Case: selecting a single column now also uses Copy-on-Write
    dtype_backend, DataFrame, Series = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    s = df["a"]

    assert np.shares_memory(get_array(s, "a"), get_array(df, "a"))

    if using_copy_on_write or using_array_manager:
        s[0] = 0
    else:
        if warn_copy_on_write:
            with tm.assert_cow_warning():
                s[0] = 0
        else:
            warn = SettingWithCopyWarning if dtype_backend == "numpy" else None
            with pd.option_context("chained_assignment", "warn"):
                with tm.assert_produces_warning(warn):
                    s[0] = 0

    expected = Series([0, 2, 3], name="a")
    tm.assert_series_equal(s, expected)
    if using_copy_on_write:
        # assert not np.shares_memory(s.values, get_array(df, "a"))
        tm.assert_frame_equal(df, df_orig)
        # ensure cached series on getitem is not the changed series
        tm.assert_series_equal(df["a"], df_orig["a"])
    else:
        df_orig.iloc[0, 0] = 0
        tm.assert_frame_equal(df, df_orig)


def test_column_as_series_set_with_upcast(
    backend, using_copy_on_write, using_array_manager, warn_copy_on_write
):
    # Case: selecting a single column now also uses Copy-on-Write -> when
    # setting a value causes an upcast, we don't need to update the parent
    # DataFrame through the cache mechanism
    dtype_backend, DataFrame, Series = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    s = df["a"]
    if dtype_backend == "nullable":
        with tm.assert_cow_warning(warn_copy_on_write):
            with pytest.raises(TypeError, match="Invalid value"):
                s[0] = "foo"
        expected = Series([1, 2, 3], name="a")
    elif using_copy_on_write or warn_copy_on_write or using_array_manager:
        # TODO(CoW-warn) assert the FutureWarning for CoW is also raised
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            s[0] = "foo"
        expected = Series(["foo", 2, 3], dtype=object, name="a")
    else:
        with pd.option_context("chained_assignment", "warn"):
            msg = "|".join(
                [
                    "A value is trying to be set on a copy of a slice from a DataFrame",
                    "Setting an item of incompatible dtype is deprecated",
                ]
            )
            with tm.assert_produces_warning(
                (SettingWithCopyWarning, FutureWarning), match=msg
            ):
                s[0] = "foo"
        expected = Series(["foo", 2, 3], dtype=object, name="a")

    tm.assert_series_equal(s, expected)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
        # ensure cached series on getitem is not the changed series
        tm.assert_series_equal(df["a"], df_orig["a"])
    else:
        df_orig["a"] = expected
        tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "method",
    [
        lambda df: df["a"],
        lambda df: df.loc[:, "a"],
        lambda df: df.iloc[:, 0],
    ],
    ids=["getitem", "loc", "iloc"],
)
def test_column_as_series_no_item_cache(
    request,
    backend,
    method,
    using_copy_on_write,
    warn_copy_on_write,
    using_array_manager,
):
    # Case: selecting a single column (which now also uses Copy-on-Write to protect
    # the view) should always give a new object (i.e. not make use of a cache)
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    s1 = method(df)
    s2 = method(df)

    is_iloc = "iloc" in request.node.name
    if using_copy_on_write or warn_copy_on_write or is_iloc:
        assert s1 is not s2
    else:
        assert s1 is s2

    if using_copy_on_write or using_array_manager:
        s1.iloc[0] = 0
    elif warn_copy_on_write:
        with tm.assert_cow_warning():
            s1.iloc[0] = 0
    else:
        warn = SettingWithCopyWarning if dtype_backend == "numpy" else None
        with pd.option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(warn):
                s1.iloc[0] = 0

    if using_copy_on_write:
        tm.assert_series_equal(s2, df_orig["a"])
        tm.assert_frame_equal(df, df_orig)
    else:
        assert s2.iloc[0] == 0


# TODO add tests for other indexing methods on the Series


def test_dataframe_add_column_from_series(backend, using_copy_on_write):
    # Case: adding a new column to a DataFrame from an existing column/series
    # -> delays copy under CoW
    _, DataFrame, Series = backend
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})

    s = Series([10, 11, 12])
    df["new"] = s
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "new"), get_array(s))
    else:
        assert not np.shares_memory(get_array(df, "new"), get_array(s))

    # editing series -> doesn't modify column in frame
    s[0] = 0
    expected = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "new": [10, 11, 12]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("val", [100, "a"])
@pytest.mark.parametrize(
    "indexer_func, indexer",
    [
        (tm.loc, (0, "a")),
        (tm.iloc, (0, 0)),
        (tm.loc, ([0], "a")),
        (tm.iloc, ([0], 0)),
        (tm.loc, (slice(None), "a")),
        (tm.iloc, (slice(None), 0)),
    ],
)
@pytest.mark.parametrize(
    "col", [[0.1, 0.2, 0.3], [7, 8, 9]], ids=["mixed-block", "single-block"]
)
def test_set_value_copy_only_necessary_column(
    using_copy_on_write, warn_copy_on_write, indexer_func, indexer, val, col
):
    # When setting inplace, only copy column that is modified instead of the whole
    # block (by splitting the block)
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": col})
    df_orig = df.copy()
    view = df[:]

    if val == "a" and not warn_copy_on_write:
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype is deprecated"
        ):
            indexer_func(df)[indexer] = val
    if val == "a" and warn_copy_on_write:
        with tm.assert_produces_warning(
            FutureWarning, match="incompatible dtype|Setting a value on a view"
        ):
            indexer_func(df)[indexer] = val
    else:
        with tm.assert_cow_warning(warn_copy_on_write and val == 100):
            indexer_func(df)[indexer] = val

    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "b"), get_array(view, "b"))
        assert not np.shares_memory(get_array(df, "a"), get_array(view, "a"))
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(get_array(df, "c"), get_array(view, "c"))
        if val == "a":
            assert not np.shares_memory(get_array(df, "a"), get_array(view, "a"))
        else:
            assert np.shares_memory(get_array(df, "a"), get_array(view, "a"))


def test_series_midx_slice(using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2, 3], index=pd.MultiIndex.from_arrays([[1, 1, 2], [3, 4, 5]]))
    ser_orig = ser.copy()
    result = ser[1]
    assert np.shares_memory(get_array(ser), get_array(result))
    with tm.assert_cow_warning(warn_copy_on_write):
        result.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_series_equal(ser, ser_orig)
    else:
        expected = Series(
            [100, 2, 3], index=pd.MultiIndex.from_arrays([[1, 1, 2], [3, 4, 5]])
        )
        tm.assert_series_equal(ser, expected)


def test_getitem_midx_slice(
    using_copy_on_write, warn_copy_on_write, using_array_manager
):
    df = DataFrame({("a", "x"): [1, 2], ("a", "y"): 1, ("b", "x"): 2})
    df_orig = df.copy()
    new_df = df[("a",)]

    if using_copy_on_write:
        assert not new_df._mgr._has_no_reference(0)

    if not using_array_manager:
        assert np.shares_memory(get_array(df, ("a", "x")), get_array(new_df, "x"))
    if using_copy_on_write:
        new_df.iloc[0, 0] = 100
        tm.assert_frame_equal(df_orig, df)
    else:
        if warn_copy_on_write:
            with tm.assert_cow_warning():
                new_df.iloc[0, 0] = 100
        else:
            with pd.option_context("chained_assignment", "warn"):
                with tm.assert_produces_warning(SettingWithCopyWarning):
                    new_df.iloc[0, 0] = 100
        assert df.iloc[0, 0] == 100


def test_series_midx_tuples_slice(using_copy_on_write, warn_copy_on_write):
    ser = Series(
        [1, 2, 3],
        index=pd.MultiIndex.from_tuples([((1, 2), 3), ((1, 2), 4), ((2, 3), 4)]),
    )
    result = ser[(1, 2)]
    assert np.shares_memory(get_array(ser), get_array(result))
    with tm.assert_cow_warning(warn_copy_on_write):
        result.iloc[0] = 100
    if using_copy_on_write:
        expected = Series(
            [1, 2, 3],
            index=pd.MultiIndex.from_tuples([((1, 2), 3), ((1, 2), 4), ((2, 3), 4)]),
        )
        tm.assert_series_equal(ser, expected)


def test_midx_read_only_bool_indexer():
    # GH#56635
    def mklbl(prefix, n):
        return [f"{prefix}{i}" for i in range(n)]

    idx = pd.MultiIndex.from_product(
        [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)]
    )
    cols = pd.MultiIndex.from_tuples(
        [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")], names=["lvl0", "lvl1"]
    )
    df = DataFrame(1, index=idx, columns=cols).sort_index().sort_index(axis=1)

    mask = df[("a", "foo")] == 1
    expected_mask = mask.copy()
    result = df.loc[pd.IndexSlice[mask, :, ["C1", "C3"]], :]
    expected = df.loc[pd.IndexSlice[:, :, ["C1", "C3"]], :]
    tm.assert_frame_equal(result, expected)
    tm.assert_series_equal(mask, expected_mask)


def test_loc_enlarging_with_dataframe(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3]})
    rhs = DataFrame({"b": [1, 2, 3], "c": [4, 5, 6]})
    rhs_orig = rhs.copy()
    df.loc[:, ["b", "c"]] = rhs
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "b"), get_array(rhs, "b"))
        assert np.shares_memory(get_array(df, "c"), get_array(rhs, "c"))
        assert not df._mgr._has_no_reference(1)
    else:
        assert not np.shares_memory(get_array(df, "b"), get_array(rhs, "b"))

    df.iloc[0, 1] = 100
    tm.assert_frame_equal(rhs, rhs_orig)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_commonmatrix.py ===
#
# Code for testing deprecated matrix classes. New test code should not be added
# here. Instead, add it to test_matrixbase.py.
#
# This entire test module and the corresponding sympy/matrices/common.py
# module will be removed in a future release.
#
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy

from sympy.assumptions import Q
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.numbers import I, Integer, oo, pi, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.exceptions import ShapeError, NonSquareMatrixError
from sympy.matrices.kind import MatrixKind
from sympy.matrices.common import (
    _MinimalMatrix, _CastableMatrix, MatrixShaping, MatrixProperties,
    MatrixOperations, MatrixArithmetic, MatrixSpecial)
from sympy.matrices.matrices import MatrixCalculus
from sympy.matrices import (Matrix, diag, eye,
    matrix_multiply_elementwise, ones, zeros, SparseMatrix, banded,
    MutableDenseMatrix, MutableSparseMatrix, ImmutableDenseMatrix,
    ImmutableSparseMatrix)
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray as Array

from sympy.abc import x, y, z


def test_matrix_deprecated_isinstance():

    # Test that e.g. isinstance(M, MatrixCommon) still gives True when M is a
    # Matrix for each of the deprecated matrix classes.

    from sympy.matrices.common import (
        MatrixRequired,
        MatrixShaping,
        MatrixSpecial,
        MatrixProperties,
        MatrixOperations,
        MatrixArithmetic,
        MatrixCommon
    )
    from sympy.matrices.matrices import (
        MatrixDeterminant,
        MatrixReductions,
        MatrixSubspaces,
        MatrixEigen,
        MatrixCalculus,
        MatrixDeprecated
    )
    from sympy import (
        Matrix,
        ImmutableMatrix,
        SparseMatrix,
        ImmutableSparseMatrix
    )
    all_mixins = (
        MatrixRequired,
        MatrixShaping,
        MatrixSpecial,
        MatrixProperties,
        MatrixOperations,
        MatrixArithmetic,
        MatrixCommon,
        MatrixDeterminant,
        MatrixReductions,
        MatrixSubspaces,
        MatrixEigen,
        MatrixCalculus,
        MatrixDeprecated
    )
    all_matrices = (
        Matrix,
        ImmutableMatrix,
        SparseMatrix,
        ImmutableSparseMatrix
    )

    Ms = [M([[1, 2], [3, 4]]) for M in all_matrices]
    t = ()

    for mixin in all_mixins:
        for M in Ms:
            with warns_deprecated_sympy():
                assert isinstance(M, mixin) is True
        with warns_deprecated_sympy():
            assert isinstance(t, mixin) is False


# classes to test the deprecated matrix classes. We use warns_deprecated_sympy
# to suppress the deprecation warnings because subclassing the deprecated
# classes causes a warning to be raised.

with warns_deprecated_sympy():
    class ShapingOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixShaping):
        pass


def eye_Shaping(n):
    return ShapingOnlyMatrix(n, n, lambda i, j: int(i == j))


def zeros_Shaping(n):
    return ShapingOnlyMatrix(n, n, lambda i, j: 0)


with warns_deprecated_sympy():
    class PropertiesOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixProperties):
        pass


def eye_Properties(n):
    return PropertiesOnlyMatrix(n, n, lambda i, j: int(i == j))


def zeros_Properties(n):
    return PropertiesOnlyMatrix(n, n, lambda i, j: 0)


with warns_deprecated_sympy():
    class OperationsOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixOperations):
        pass


def eye_Operations(n):
    return OperationsOnlyMatrix(n, n, lambda i, j: int(i == j))


def zeros_Operations(n):
    return OperationsOnlyMatrix(n, n, lambda i, j: 0)


with warns_deprecated_sympy():
    class ArithmeticOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixArithmetic):
        pass


def eye_Arithmetic(n):
    return ArithmeticOnlyMatrix(n, n, lambda i, j: int(i == j))


def zeros_Arithmetic(n):
    return ArithmeticOnlyMatrix(n, n, lambda i, j: 0)


with warns_deprecated_sympy():
    class SpecialOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixSpecial):
        pass


with warns_deprecated_sympy():
    class CalculusOnlyMatrix(_MinimalMatrix, _CastableMatrix, MatrixCalculus):
        pass


def test__MinimalMatrix():
    x = _MinimalMatrix(2, 3, [1, 2, 3, 4, 5, 6])
    assert x.rows == 2
    assert x.cols == 3
    assert x[2] == 3
    assert x[1, 1] == 5
    assert list(x) == [1, 2, 3, 4, 5, 6]
    assert list(x[1, :]) == [4, 5, 6]
    assert list(x[:, 1]) == [2, 5]
    assert list(x[:, :]) == list(x)
    assert x[:, :] == x
    assert _MinimalMatrix(x) == x
    assert _MinimalMatrix([[1, 2, 3], [4, 5, 6]]) == x
    assert _MinimalMatrix(([1, 2, 3], [4, 5, 6])) == x
    assert _MinimalMatrix([(1, 2, 3), (4, 5, 6)]) == x
    assert _MinimalMatrix(((1, 2, 3), (4, 5, 6))) == x
    assert not (_MinimalMatrix([[1, 2], [3, 4], [5, 6]]) == x)


def test_kind():
    assert Matrix([[1, 2], [3, 4]]).kind == MatrixKind(NumberKind)
    assert Matrix([[0, 0], [0, 0]]).kind == MatrixKind(NumberKind)
    assert Matrix(0, 0, []).kind == MatrixKind(NumberKind)
    assert Matrix([[x]]).kind == MatrixKind(NumberKind)
    assert Matrix([[1, Matrix([[1]])]]).kind == MatrixKind(UndefinedKind)
    assert SparseMatrix([[1]]).kind == MatrixKind(NumberKind)
    assert SparseMatrix([[1, Matrix([[1]])]]).kind == MatrixKind(UndefinedKind)


# ShapingOnlyMatrix tests
def test_vec():
    m = ShapingOnlyMatrix(2, 2, [1, 3, 2, 4])
    m_vec = m.vec()
    assert m_vec.cols == 1
    for i in range(4):
        assert m_vec[i] == i + 1


def test_todok():
    a, b, c, d = symbols('a:d')
    m1 = MutableDenseMatrix([[a, b], [c, d]])
    m2 = ImmutableDenseMatrix([[a, b], [c, d]])
    m3 = MutableSparseMatrix([[a, b], [c, d]])
    m4 = ImmutableSparseMatrix([[a, b], [c, d]])
    assert m1.todok() == m2.todok() == m3.todok() == m4.todok() == \
        {(0, 0): a, (0, 1): b, (1, 0): c, (1, 1): d}


def test_tolist():
    lst = [[S.One, S.Half, x*y, S.Zero], [x, y, z, x**2], [y, -S.One, z*x, 3]]
    flat_lst = [S.One, S.Half, x*y, S.Zero, x, y, z, x**2, y, -S.One, z*x, 3]
    m = ShapingOnlyMatrix(3, 4, flat_lst)
    assert m.tolist() == lst

def test_todod():
    m = ShapingOnlyMatrix(3, 2, [[S.One, 0], [0, S.Half], [x, 0]])
    dict = {0: {0: S.One}, 1: {1: S.Half}, 2: {0: x}}
    assert m.todod() == dict

def test_row_col_del():
    e = ShapingOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    raises(IndexError, lambda: e.row_del(5))
    raises(IndexError, lambda: e.row_del(-5))
    raises(IndexError, lambda: e.col_del(5))
    raises(IndexError, lambda: e.col_del(-5))

    assert e.row_del(2) == e.row_del(-1) == Matrix([[1, 2, 3], [4, 5, 6]])
    assert e.col_del(2) == e.col_del(-1) == Matrix([[1, 2], [4, 5], [7, 8]])

    assert e.row_del(1) == e.row_del(-2) == Matrix([[1, 2, 3], [7, 8, 9]])
    assert e.col_del(1) == e.col_del(-2) == Matrix([[1, 3], [4, 6], [7, 9]])


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
    A, B, C, D = diag(a, b, b), diag(a, b, c), diag(a, c, b), diag(c, c, b)
    A = ShapingOnlyMatrix(A.rows, A.cols, A)
    B = ShapingOnlyMatrix(B.rows, B.cols, B)
    C = ShapingOnlyMatrix(C.rows, C.cols, C)
    D = ShapingOnlyMatrix(D.rows, D.cols, D)

    assert A.get_diag_blocks() == [a, b, b]
    assert B.get_diag_blocks() == [a, b, c]
    assert C.get_diag_blocks() == [a, c, b]
    assert D.get_diag_blocks() == [c, c, b]


def test_shape():
    m = ShapingOnlyMatrix(1, 2, [0, 0])
    assert m.shape == (1, 2)


def test_reshape():
    m0 = eye_Shaping(3)
    assert m0.reshape(1, 9) == Matrix(1, 9, (1, 0, 0, 0, 1, 0, 0, 0, 1))
    m1 = ShapingOnlyMatrix(3, 4, lambda i, j: i + j)
    assert m1.reshape(
        4, 3) == Matrix(((0, 1, 2), (3, 1, 2), (3, 4, 2), (3, 4, 5)))
    assert m1.reshape(2, 6) == Matrix(((0, 1, 2, 3, 1, 2), (3, 4, 2, 3, 4, 5)))


def test_row_col():
    m = ShapingOnlyMatrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert m.row(0) == Matrix(1, 3, [1, 2, 3])
    assert m.col(0) == Matrix(3, 1, [1, 4, 7])


def test_row_join():
    assert eye_Shaping(3).row_join(Matrix([7, 7, 7])) == \
           Matrix([[1, 0, 0, 7],
                   [0, 1, 0, 7],
                   [0, 0, 1, 7]])


def test_col_join():
    assert eye_Shaping(3).col_join(Matrix([[7, 7, 7]])) == \
           Matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [7, 7, 7]])


def test_row_insert():
    r4 = Matrix([[4, 4, 4]])
    for i in range(-4, 5):
        l = [1, 0, 0]
        l.insert(i, 4)
        assert flatten(eye_Shaping(3).row_insert(i, r4).col(0).tolist()) == l


def test_col_insert():
    c4 = Matrix([4, 4, 4])
    for i in range(-4, 5):
        l = [0, 0, 0]
        l.insert(i, 4)
        assert flatten(zeros_Shaping(3).col_insert(i, c4).row(0).tolist()) == l
    # issue 13643
    assert eye_Shaping(6).col_insert(3, Matrix([[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])) == \
           Matrix([[1, 0, 0, 2, 2, 0, 0, 0],
                   [0, 1, 0, 2, 2, 0, 0, 0],
                   [0, 0, 1, 2, 2, 0, 0, 0],
                   [0, 0, 0, 2, 2, 1, 0, 0],
                   [0, 0, 0, 2, 2, 0, 1, 0],
                   [0, 0, 0, 2, 2, 0, 0, 1]])


def test_extract():
    m = ShapingOnlyMatrix(4, 3, lambda i, j: i*3 + j)
    assert m.extract([0, 1, 3], [0, 1]) == Matrix(3, 2, [0, 1, 3, 4, 9, 10])
    assert m.extract([0, 3], [0, 0, 2]) == Matrix(2, 3, [0, 0, 2, 9, 9, 11])
    assert m.extract(range(4), range(3)) == m
    raises(IndexError, lambda: m.extract([4], [0]))
    raises(IndexError, lambda: m.extract([0], [3]))


def test_hstack():
    m = ShapingOnlyMatrix(4, 3, lambda i, j: i*3 + j)
    m2 = ShapingOnlyMatrix(3, 4, lambda i, j: i*3 + j)
    assert m == m.hstack(m)
    assert m.hstack(m, m, m) == ShapingOnlyMatrix.hstack(m, m, m) == Matrix([
                [0,  1,  2, 0,  1,  2, 0,  1,  2],
                [3,  4,  5, 3,  4,  5, 3,  4,  5],
                [6,  7,  8, 6,  7,  8, 6,  7,  8],
                [9, 10, 11, 9, 10, 11, 9, 10, 11]])
    raises(ShapeError, lambda: m.hstack(m, m2))
    assert Matrix.hstack() == Matrix()

    # test regression #12938
    M1 = Matrix.zeros(0, 0)
    M2 = Matrix.zeros(0, 1)
    M3 = Matrix.zeros(0, 2)
    M4 = Matrix.zeros(0, 3)
    m = ShapingOnlyMatrix.hstack(M1, M2, M3, M4)
    assert m.rows == 0 and m.cols == 6


def test_vstack():
    m = ShapingOnlyMatrix(4, 3, lambda i, j: i*3 + j)
    m2 = ShapingOnlyMatrix(3, 4, lambda i, j: i*3 + j)
    assert m == m.vstack(m)
    assert m.vstack(m, m, m) == ShapingOnlyMatrix.vstack(m, m, m) == Matrix([
                                [0,  1,  2],
                                [3,  4,  5],
                                [6,  7,  8],
                                [9, 10, 11],
                                [0,  1,  2],
                                [3,  4,  5],
                                [6,  7,  8],
                                [9, 10, 11],
                                [0,  1,  2],
                                [3,  4,  5],
                                [6,  7,  8],
                                [9, 10, 11]])
    raises(ShapeError, lambda: m.vstack(m, m2))
    assert Matrix.vstack() == Matrix()


# PropertiesOnlyMatrix tests
def test_atoms():
    m = PropertiesOnlyMatrix(2, 2, [1, 2, x, 1 - 1/x])
    assert m.atoms() == {S.One, S(2), S.NegativeOne, x}
    assert m.atoms(Symbol) == {x}


def test_free_symbols():
    assert PropertiesOnlyMatrix([[x], [0]]).free_symbols == {x}


def test_has():
    A = PropertiesOnlyMatrix(((x, y), (2, 3)))
    assert A.has(x)
    assert not A.has(z)
    assert A.has(Symbol)

    A = PropertiesOnlyMatrix(((2, y), (2, 3)))
    assert not A.has(x)


def test_is_anti_symmetric():
    x = symbols('x')
    assert PropertiesOnlyMatrix(2, 1, [1, 2]).is_anti_symmetric() is False
    m = PropertiesOnlyMatrix(3, 3, [0, x**2 + 2*x + 1, y, -(x + 1)**2, 0, x*y, -y, -x*y, 0])
    assert m.is_anti_symmetric() is True
    assert m.is_anti_symmetric(simplify=False) is False
    assert m.is_anti_symmetric(simplify=lambda x: x) is False

    m = PropertiesOnlyMatrix(3, 3, [x.expand() for x in m])
    assert m.is_anti_symmetric(simplify=False) is True
    m = PropertiesOnlyMatrix(3, 3, [x.expand() for x in [S.One] + list(m)[1:]])
    assert m.is_anti_symmetric() is False


def test_diagonal_symmetrical():
    m = PropertiesOnlyMatrix(2, 2, [0, 1, 1, 0])
    assert not m.is_diagonal()
    assert m.is_symmetric()
    assert m.is_symmetric(simplify=False)

    m = PropertiesOnlyMatrix(2, 2, [1, 0, 0, 1])
    assert m.is_diagonal()

    m = PropertiesOnlyMatrix(3, 3, diag(1, 2, 3))
    assert m.is_diagonal()
    assert m.is_symmetric()

    m = PropertiesOnlyMatrix(3, 3, [1, 0, 0, 0, 2, 0, 0, 0, 3])
    assert m == diag(1, 2, 3)

    m = PropertiesOnlyMatrix(2, 3, zeros(2, 3))
    assert not m.is_symmetric()
    assert m.is_diagonal()

    m = PropertiesOnlyMatrix(((5, 0), (0, 6), (0, 0)))
    assert m.is_diagonal()

    m = PropertiesOnlyMatrix(((5, 0, 0), (0, 6, 0)))
    assert m.is_diagonal()

    m = Matrix(3, 3, [1, x**2 + 2*x + 1, y, (x + 1)**2, 2, 0, y, 0, 3])
    assert m.is_symmetric()
    assert not m.is_symmetric(simplify=False)
    assert m.expand().is_symmetric(simplify=False)


def test_is_hermitian():
    a = PropertiesOnlyMatrix([[1, I], [-I, 1]])
    assert a.is_hermitian
    a = PropertiesOnlyMatrix([[2*I, I], [-I, 1]])
    assert a.is_hermitian is False
    a = PropertiesOnlyMatrix([[x, I], [-I, 1]])
    assert a.is_hermitian is None
    a = PropertiesOnlyMatrix([[x, 1], [-I, 1]])
    assert a.is_hermitian is False


def test_is_Identity():
    assert eye_Properties(3).is_Identity
    assert not PropertiesOnlyMatrix(zeros(3)).is_Identity
    assert not PropertiesOnlyMatrix(ones(3)).is_Identity
    # issue 6242
    assert not PropertiesOnlyMatrix([[1, 0, 0]]).is_Identity


def test_is_symbolic():
    a = PropertiesOnlyMatrix([[x, x], [x, x]])
    assert a.is_symbolic() is True
    a = PropertiesOnlyMatrix([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert a.is_symbolic() is False
    a = PropertiesOnlyMatrix([[1, 2, 3, 4], [5, 6, x, 8]])
    assert a.is_symbolic() is True
    a = PropertiesOnlyMatrix([[1, x, 3]])
    assert a.is_symbolic() is True
    a = PropertiesOnlyMatrix([[1, 2, 3]])
    assert a.is_symbolic() is False
    a = PropertiesOnlyMatrix([[1], [x], [3]])
    assert a.is_symbolic() is True
    a = PropertiesOnlyMatrix([[1], [2], [3]])
    assert a.is_symbolic() is False


def test_is_upper():
    a = PropertiesOnlyMatrix([[1, 2, 3]])
    assert a.is_upper is True
    a = PropertiesOnlyMatrix([[1], [2], [3]])
    assert a.is_upper is False


def test_is_lower():
    a = PropertiesOnlyMatrix([[1, 2, 3]])
    assert a.is_lower is False
    a = PropertiesOnlyMatrix([[1], [2], [3]])
    assert a.is_lower is True


def test_is_square():
    m = PropertiesOnlyMatrix([[1], [1]])
    m2 = PropertiesOnlyMatrix([[2, 2], [2, 2]])
    assert not m.is_square
    assert m2.is_square


def test_is_symmetric():
    m = PropertiesOnlyMatrix(2, 2, [0, 1, 1, 0])
    assert m.is_symmetric()
    m = PropertiesOnlyMatrix(2, 2, [0, 1, 0, 1])
    assert not m.is_symmetric()


def test_is_hessenberg():
    A = PropertiesOnlyMatrix([[3, 4, 1], [2, 4, 5], [0, 1, 2]])
    assert A.is_upper_hessenberg
    A = PropertiesOnlyMatrix(3, 3, [3, 2, 0, 4, 4, 1, 1, 5, 2])
    assert A.is_lower_hessenberg
    A = PropertiesOnlyMatrix(3, 3, [3, 2, -1, 4, 4, 1, 1, 5, 2])
    assert A.is_lower_hessenberg is False
    assert A.is_upper_hessenberg is False

    A = PropertiesOnlyMatrix([[3, 4, 1], [2, 4, 5], [3, 1, 2]])
    assert not A.is_upper_hessenberg


def test_is_zero():
    assert PropertiesOnlyMatrix(0, 0, []).is_zero_matrix
    assert PropertiesOnlyMatrix([[0, 0], [0, 0]]).is_zero_matrix
    assert PropertiesOnlyMatrix(zeros(3, 4)).is_zero_matrix
    assert not PropertiesOnlyMatrix(eye(3)).is_zero_matrix
    assert PropertiesOnlyMatrix([[x, 0], [0, 0]]).is_zero_matrix == None
    assert PropertiesOnlyMatrix([[x, 1], [0, 0]]).is_zero_matrix == False
    a = Symbol('a', nonzero=True)
    assert PropertiesOnlyMatrix([[a, 0], [0, 0]]).is_zero_matrix == False


def test_values():
    assert set(PropertiesOnlyMatrix(2, 2, [0, 1, 2, 3]
        ).values()) == {1, 2, 3}
    x = Symbol('x', real=True)
    assert set(PropertiesOnlyMatrix(2, 2, [x, 0, 0, 1]
        ).values()) == {x, 1}


# OperationsOnlyMatrix tests
def test_applyfunc():
    m0 = OperationsOnlyMatrix(eye(3))
    assert m0.applyfunc(lambda x: 2*x) == eye(3)*2
    assert m0.applyfunc(lambda x: 0) == zeros(3)
    assert m0.applyfunc(lambda x: 1) == ones(3)


def test_adjoint():
    dat = [[0, I], [1, 0]]
    ans = OperationsOnlyMatrix([[0, 1], [-I, 0]])
    assert ans.adjoint() == Matrix(dat)


def test_as_real_imag():
    m1 = OperationsOnlyMatrix(2, 2, [1, 2, 3, 4])
    m3 = OperationsOnlyMatrix(2, 2,
        [1 + S.ImaginaryUnit, 2 + 2*S.ImaginaryUnit,
        3 + 3*S.ImaginaryUnit, 4 + 4*S.ImaginaryUnit])

    a, b = m3.as_real_imag()
    assert a == m1
    assert b == m1


def test_conjugate():
    M = OperationsOnlyMatrix([[0, I, 5],
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


def test_doit():
    a = OperationsOnlyMatrix([[Add(x, x, evaluate=False)]])
    assert a[0] != 2*x
    assert a.doit() == Matrix([[2*x]])


def test_evalf():
    a = OperationsOnlyMatrix(2, 1, [sqrt(5), 6])
    assert all(a.evalf()[i] == a[i].evalf() for i in range(2))
    assert all(a.evalf(2)[i] == a[i].evalf(2) for i in range(2))
    assert all(a.n(2)[i] == a[i].n(2) for i in range(2))


def test_expand():
    m0 = OperationsOnlyMatrix([[x*(x + y), 2], [((x + y)*y)*x, x*(y + x*(x + y))]])
    # Test if expand() returns a matrix
    m1 = m0.expand()
    assert m1 == Matrix(
        [[x*y + x**2, 2], [x*y**2 + y*x**2, x*y + y*x**2 + x**3]])

    a = Symbol('a', real=True)

    assert OperationsOnlyMatrix(1, 1, [exp(I*a)]).expand(complex=True) == \
           Matrix([cos(a) + I*sin(a)])


def test_refine():
    m0 = OperationsOnlyMatrix([[Abs(x)**2, sqrt(x**2)],
                 [sqrt(x**2)*Abs(y)**2, sqrt(y**2)*Abs(x)**2]])
    m1 = m0.refine(Q.real(x) & Q.real(y))
    assert m1 == Matrix([[x**2, Abs(x)], [y**2*Abs(x), x**2*Abs(y)]])

    m1 = m0.refine(Q.positive(x) & Q.positive(y))
    assert m1 == Matrix([[x**2, x], [x*y**2, x**2*y]])

    m1 = m0.refine(Q.negative(x) & Q.negative(y))
    assert m1 == Matrix([[x**2, -x], [-x*y**2, -x**2*y]])


def test_replace():
    F, G = symbols('F, G', cls=Function)
    K = OperationsOnlyMatrix(2, 2, lambda i, j: G(i+j))
    M = OperationsOnlyMatrix(2, 2, lambda i, j: F(i+j))
    N = M.replace(F, G)
    assert N == K


def test_replace_map():
    F, G = symbols('F, G', cls=Function)
    K = OperationsOnlyMatrix(2, 2, [(G(0), {F(0): G(0)}), (G(1), {F(1): G(1)}), (G(1), {F(1) \
                                                                              : G(1)}), (G(2), {F(2): G(2)})])
    M = OperationsOnlyMatrix(2, 2, lambda i, j: F(i+j))
    N = M.replace(F, G, True)
    assert N == K


def test_rot90():
    A = Matrix([[1, 2], [3, 4]])
    assert A == A.rot90(0) == A.rot90(4)
    assert A.rot90(2) == A.rot90(-2) == A.rot90(6) == Matrix(((4, 3), (2, 1)))
    assert A.rot90(3) == A.rot90(-1) == A.rot90(7) == Matrix(((2, 4), (1, 3)))
    assert A.rot90() == A.rot90(-7) == A.rot90(-3) == Matrix(((3, 1), (4, 2)))

def test_simplify():
    n = Symbol('n')
    f = Function('f')

    M = OperationsOnlyMatrix([[            1/x + 1/y,                 (x + x*y) / x  ],
                [ (f(x) + y*f(x))/f(x), 2 * (1/n - cos(n * pi)/n) / pi ]])
    assert M.simplify() == Matrix([[ (x + y)/(x * y),                        1 + y ],
                        [           1 + y, 2*((1 - 1*cos(pi*n))/(pi*n)) ]])
    eq = (1 + x)**2
    M = OperationsOnlyMatrix([[eq]])
    assert M.simplify() == Matrix([[eq]])
    assert M.simplify(ratio=oo) == Matrix([[eq.simplify(ratio=oo)]])

    # https://github.com/sympy/sympy/issues/19353
    m = Matrix([[30, 2], [3, 4]])
    assert (1/(m.trace())).simplify() == Rational(1, 34)


def test_subs():
    assert OperationsOnlyMatrix([[1, x], [x, 4]]).subs(x, 5) == Matrix([[1, 5], [5, 4]])
    assert OperationsOnlyMatrix([[x, 2], [x + y, 4]]).subs([[x, -1], [y, -2]]) == \
           Matrix([[-1, 2], [-3, 4]])
    assert OperationsOnlyMatrix([[x, 2], [x + y, 4]]).subs([(x, -1), (y, -2)]) == \
           Matrix([[-1, 2], [-3, 4]])
    assert OperationsOnlyMatrix([[x, 2], [x + y, 4]]).subs({x: -1, y: -2}) == \
           Matrix([[-1, 2], [-3, 4]])
    assert OperationsOnlyMatrix([[x*y]]).subs({x: y - 1, y: x - 1}, simultaneous=True) == \
           Matrix([[(x - 1)*(y - 1)]])


def test_trace():
    M = OperationsOnlyMatrix([[1, 0, 0],
                [0, 5, 0],
                [0, 0, 8]])
    assert M.trace() == 14


def test_xreplace():
    assert OperationsOnlyMatrix([[1, x], [x, 4]]).xreplace({x: 5}) == \
           Matrix([[1, 5], [5, 4]])
    assert OperationsOnlyMatrix([[x, 2], [x + y, 4]]).xreplace({x: -1, y: -2}) == \
           Matrix([[-1, 2], [-3, 4]])


def test_permute():
    a = OperationsOnlyMatrix(3, 4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    raises(IndexError, lambda: a.permute([[0, 5]]))
    raises(ValueError, lambda: a.permute(Symbol('x')))
    b = a.permute_rows([[0, 2], [0, 1]])
    assert a.permute([[0, 2], [0, 1]]) == b == Matrix([
                                            [5,  6,  7,  8],
                                            [9, 10, 11, 12],
                                            [1,  2,  3,  4]])

    b = a.permute_cols([[0, 2], [0, 1]])
    assert a.permute([[0, 2], [0, 1]], orientation='cols') == b ==\
                            Matrix([
                            [ 2,  3, 1,  4],
                            [ 6,  7, 5,  8],
                            [10, 11, 9, 12]])

    b = a.permute_cols([[0, 2], [0, 1]], direction='backward')
    assert a.permute([[0, 2], [0, 1]], orientation='cols', direction='backward') == b ==\
                            Matrix([
                            [ 3, 1,  2,  4],
                            [ 7, 5,  6,  8],
                            [11, 9, 10, 12]])

    assert a.permute([1, 2, 0, 3]) == Matrix([
                                            [5,  6,  7,  8],
                                            [9, 10, 11, 12],
                                            [1,  2,  3,  4]])

    from sympy.combinatorics import Permutation
    assert a.permute(Permutation([1, 2, 0, 3])) == Matrix([
                                            [5,  6,  7,  8],
                                            [9, 10, 11, 12],
                                            [1,  2,  3,  4]])

def test_upper_triangular():

    A = OperationsOnlyMatrix([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ])

    R = A.upper_triangular(2)
    assert R == OperationsOnlyMatrix([
                        [0, 0, 1, 1],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]
                    ])

    R = A.upper_triangular(-2)
    assert R == OperationsOnlyMatrix([
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 1, 1, 1]
                    ])

    R = A.upper_triangular()
    assert R == OperationsOnlyMatrix([
                        [1, 1, 1, 1],
                        [0, 1, 1, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]
                    ])

def test_lower_triangular():
    A = OperationsOnlyMatrix([
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]
                    ])

    L = A.lower_triangular()
    assert L == ArithmeticOnlyMatrix([
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1]])

    L = A.lower_triangular(2)
    assert L == ArithmeticOnlyMatrix([
                        [1, 1, 1, 0],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]
                    ])

    L = A.lower_triangular(-2)
    assert L == ArithmeticOnlyMatrix([
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 0, 0]
                    ])


# ArithmeticOnlyMatrix tests
def test_abs():
    m = ArithmeticOnlyMatrix([[1, -2], [x, y]])
    assert abs(m) == ArithmeticOnlyMatrix([[1, 2], [Abs(x), Abs(y)]])


def test_add():
    m = ArithmeticOnlyMatrix([[1, 2, 3], [x, y, x], [2*y, -50, z*x]])
    assert m + m == ArithmeticOnlyMatrix([[2, 4, 6], [2*x, 2*y, 2*x], [4*y, -100, 2*z*x]])
    n = ArithmeticOnlyMatrix(1, 2, [1, 2])
    raises(ShapeError, lambda: m + n)


def test_multiplication():
    a = ArithmeticOnlyMatrix((
        (1, 2),
        (3, 1),
        (0, 6),
    ))

    b = ArithmeticOnlyMatrix((
        (1, 2),
        (3, 0),
    ))

    raises(ShapeError, lambda: b*a)
    raises(TypeError, lambda: a*{})

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

    h = a.multiply_elementwise(c)
    assert h == matrix_multiply_elementwise(a, c)
    assert h[0, 0] == 7
    assert h[0, 1] == 4
    assert h[1, 0] == 18
    assert h[1, 1] == 6
    assert h[2, 0] == 0
    assert h[2, 1] == 0
    raises(ShapeError, lambda: a.multiply_elementwise(b))

    c = b * Symbol("x")
    assert isinstance(c, ArithmeticOnlyMatrix)
    assert c[0, 0] == x
    assert c[0, 1] == 2*x
    assert c[1, 0] == 3*x
    assert c[1, 1] == 0

    c2 = x * b
    assert c == c2

    c = 5 * b
    assert isinstance(c, ArithmeticOnlyMatrix)
    assert c[0, 0] == 5
    assert c[0, 1] == 2*5
    assert c[1, 0] == 3*5
    assert c[1, 1] == 0

    try:
        eval('c = 5 @ b')
    except SyntaxError:
        pass
    else:
        assert isinstance(c, ArithmeticOnlyMatrix)
        assert c[0, 0] == 5
        assert c[0, 1] == 2*5
        assert c[1, 0] == 3*5
        assert c[1, 1] == 0

    # https://github.com/sympy/sympy/issues/22353
    A = Matrix(ones(3, 1))
    _h = -Rational(1, 2)
    B = Matrix([_h, _h, _h])
    assert A.multiply_elementwise(B) == Matrix([
        [_h],
        [_h],
        [_h]])


def test_matmul():
    a = Matrix([[1, 2], [3, 4]])

    assert a.__matmul__(2) == NotImplemented

    assert a.__rmatmul__(2) == NotImplemented

    #This is done this way because @ is only supported in Python 3.5+
    #To check 2@a case
    try:
        eval('2 @ a')
    except SyntaxError:
        pass
    except TypeError:  #TypeError is raised in case of NotImplemented is returned
        pass

    #Check a@2 case
    try:
        eval('a @ 2')
    except SyntaxError:
        pass
    except TypeError:  #TypeError is raised in case of NotImplemented is returned
        pass


def test_non_matmul():
    """
    Test that if explicitly specified as non-matrix, mul reverts
    to scalar multiplication.
    """
    class foo(Expr):
        is_Matrix=False
        is_MatrixLike=False
        shape = (1, 1)

    A = Matrix([[1, 2], [3, 4]])
    b = foo()
    assert b*A == Matrix([[b, 2*b], [3*b, 4*b]])
    assert A*b == Matrix([[b, 2*b], [3*b, 4*b]])


def test_power():
    raises(NonSquareMatrixError, lambda: Matrix((1, 2))**2)

    A = ArithmeticOnlyMatrix([[2, 3], [4, 5]])
    assert (A**5)[:] == (6140, 8097, 10796, 14237)
    A = ArithmeticOnlyMatrix([[2, 1, 3], [4, 2, 4], [6, 12, 1]])
    assert (A**3)[:] == (290, 262, 251, 448, 440, 368, 702, 954, 433)
    assert A**0 == eye(3)
    assert A**1 == A
    assert (ArithmeticOnlyMatrix([[2]]) ** 100)[0, 0] == 2**100
    assert ArithmeticOnlyMatrix([[1, 2], [3, 4]])**Integer(2) == ArithmeticOnlyMatrix([[7, 10], [15, 22]])
    A = Matrix([[1,2],[4,5]])
    assert A.pow(20, method='cayley') == A.pow(20, method='multiply')

def test_neg():
    n = ArithmeticOnlyMatrix(1, 2, [1, 2])
    assert -n == ArithmeticOnlyMatrix(1, 2, [-1, -2])


def test_sub():
    n = ArithmeticOnlyMatrix(1, 2, [1, 2])
    assert n - n == ArithmeticOnlyMatrix(1, 2, [0, 0])


def test_div():
    n = ArithmeticOnlyMatrix(1, 2, [1, 2])
    assert n/2 == ArithmeticOnlyMatrix(1, 2, [S.Half, S(2)/2])

# SpecialOnlyMatrix tests
def test_eye():
    assert list(SpecialOnlyMatrix.eye(2, 2)) == [1, 0, 0, 1]
    assert list(SpecialOnlyMatrix.eye(2)) == [1, 0, 0, 1]
    assert type(SpecialOnlyMatrix.eye(2)) == SpecialOnlyMatrix
    assert type(SpecialOnlyMatrix.eye(2, cls=Matrix)) == Matrix


def test_ones():
    assert list(SpecialOnlyMatrix.ones(2, 2)) == [1, 1, 1, 1]
    assert list(SpecialOnlyMatrix.ones(2)) == [1, 1, 1, 1]
    assert SpecialOnlyMatrix.ones(2, 3) == Matrix([[1, 1, 1], [1, 1, 1]])
    assert type(SpecialOnlyMatrix.ones(2)) == SpecialOnlyMatrix
    assert type(SpecialOnlyMatrix.ones(2, cls=Matrix)) == Matrix


def test_zeros():
    assert list(SpecialOnlyMatrix.zeros(2, 2)) == [0, 0, 0, 0]
    assert list(SpecialOnlyMatrix.zeros(2)) == [0, 0, 0, 0]
    assert SpecialOnlyMatrix.zeros(2, 3) == Matrix([[0, 0, 0], [0, 0, 0]])
    assert type(SpecialOnlyMatrix.zeros(2)) == SpecialOnlyMatrix
    assert type(SpecialOnlyMatrix.zeros(2, cls=Matrix)) == Matrix


def test_diag_make():
    diag = SpecialOnlyMatrix.diag
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    assert diag(a, b, b) == Matrix([
        [1, 2, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0],
        [0, 0, 3, x, 0, 0],
        [0, 0, y, 3, 0, 0],
        [0, 0, 0, 0, 3, x],
        [0, 0, 0, 0, y, 3],
    ])
    assert diag(a, b, c) == Matrix([
        [1, 2, 0, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0, 0],
        [0, 0, 3, x, 0, 0, 0],
        [0, 0, y, 3, 0, 0, 0],
        [0, 0, 0, 0, 3, x, 3],
        [0, 0, 0, 0, y, 3, z],
        [0, 0, 0, 0, x, y, z],
    ])
    assert diag(a, c, b) == Matrix([
        [1, 2, 0, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0, 0],
        [0, 0, 3, x, 3, 0, 0],
        [0, 0, y, 3, z, 0, 0],
        [0, 0, x, y, z, 0, 0],
        [0, 0, 0, 0, 0, 3, x],
        [0, 0, 0, 0, 0, y, 3],
    ])
    a = Matrix([x, y, z])
    b = Matrix([[1, 2], [3, 4]])
    c = Matrix([[5, 6]])
    # this "wandering diagonal" is what makes this
    # a block diagonal where each block is independent
    # of the others
    assert diag(a, 7, b, c) == Matrix([
        [x, 0, 0, 0, 0, 0],
        [y, 0, 0, 0, 0, 0],
        [z, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 3, 4, 0, 0],
        [0, 0, 0, 0, 5, 6]])
    raises(ValueError, lambda: diag(a, 7, b, c, rows=5))
    assert diag(1) == Matrix([[1]])
    assert diag(1, rows=2) == Matrix([[1, 0], [0, 0]])
    assert diag(1, cols=2) == Matrix([[1, 0], [0, 0]])
    assert diag(1, rows=3, cols=2) == Matrix([[1, 0], [0, 0], [0, 0]])
    assert diag(*[2, 3]) == Matrix([
        [2, 0],
        [0, 3]])
    assert diag(Matrix([2, 3])) == Matrix([
        [2],
        [3]])
    assert diag([1, [2, 3], 4], unpack=False) == \
            diag([[1], [2, 3], [4]], unpack=False) == Matrix([
        [1, 0],
        [2, 3],
        [4, 0]])
    assert type(diag(1)) == SpecialOnlyMatrix
    assert type(diag(1, cls=Matrix)) == Matrix
    assert Matrix.diag([1, 2, 3]) == Matrix.diag(1, 2, 3)
    assert Matrix.diag([1, 2, 3], unpack=False).shape == (3, 1)
    assert Matrix.diag([[1, 2, 3]]).shape == (3, 1)
    assert Matrix.diag([[1, 2, 3]], unpack=False).shape == (1, 3)
    assert Matrix.diag([[[1, 2, 3]]]).shape == (1, 3)
    # kerning can be used to move the starting point
    assert Matrix.diag(ones(0, 2), 1, 2) == Matrix([
        [0, 0, 1, 0],
        [0, 0, 0, 2]])
    assert Matrix.diag(ones(2, 0), 1, 2) == Matrix([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 2]])


def test_diagonal():
    m = Matrix(3, 3, range(9))
    d = m.diagonal()
    assert d == m.diagonal(0)
    assert tuple(d) == (0, 4, 8)
    assert tuple(m.diagonal(1)) == (1, 5)
    assert tuple(m.diagonal(-1)) == (3, 7)
    assert tuple(m.diagonal(2)) == (2,)
    assert type(m.diagonal()) == type(m)
    s = SparseMatrix(3, 3, {(1, 1): 1})
    assert type(s.diagonal()) == type(s)
    assert type(m) != type(s)
    raises(ValueError, lambda: m.diagonal(3))
    raises(ValueError, lambda: m.diagonal(-3))
    raises(ValueError, lambda: m.diagonal(pi))
    M = ones(2, 3)
    assert banded({i: list(M.diagonal(i))
        for i in range(1-M.rows, M.cols)}) == M


def test_jordan_block():
    assert SpecialOnlyMatrix.jordan_block(3, 2) == SpecialOnlyMatrix.jordan_block(3, eigenvalue=2) \
            == SpecialOnlyMatrix.jordan_block(size=3, eigenvalue=2) \
            == SpecialOnlyMatrix.jordan_block(3, 2, band='upper') \
            == SpecialOnlyMatrix.jordan_block(
                size=3, eigenval=2, eigenvalue=2) \
            == Matrix([
                [2, 1, 0],
                [0, 2, 1],
                [0, 0, 2]])

    assert SpecialOnlyMatrix.jordan_block(3, 2, band='lower') == Matrix([
                    [2, 0, 0],
                    [1, 2, 0],
                    [0, 1, 2]])
    # missing eigenvalue
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(2))
    # non-integral size
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(3.5, 2))
    # size not specified
    raises(ValueError, lambda: SpecialOnlyMatrix.jordan_block(eigenvalue=2))
    # inconsistent eigenvalue
    raises(ValueError,
    lambda: SpecialOnlyMatrix.jordan_block(
        eigenvalue=2, eigenval=4))

    # Using alias keyword
    assert SpecialOnlyMatrix.jordan_block(size=3, eigenvalue=2) == \
        SpecialOnlyMatrix.jordan_block(size=3, eigenval=2)


def test_orthogonalize():
    m = Matrix([[1, 2], [3, 4]])
    assert m.orthogonalize(Matrix([[2], [1]])) == [Matrix([[2], [1]])]
    assert m.orthogonalize(Matrix([[2], [1]]), normalize=True) == \
        [Matrix([[2*sqrt(5)/5], [sqrt(5)/5]])]
    assert m.orthogonalize(Matrix([[1], [2]]), Matrix([[-1], [4]])) == \
        [Matrix([[1], [2]]), Matrix([[Rational(-12, 5)], [Rational(6, 5)]])]
    assert m.orthogonalize(Matrix([[0], [0]]), Matrix([[-1], [4]])) == \
        [Matrix([[-1], [4]])]
    assert m.orthogonalize(Matrix([[0], [0]])) == []

    n = Matrix([[9, 1, 9], [3, 6, 10], [8, 5, 2]])
    vecs = [Matrix([[-5], [1]]), Matrix([[-5], [2]]), Matrix([[-5], [-2]])]
    assert n.orthogonalize(*vecs) == \
        [Matrix([[-5], [1]]), Matrix([[Rational(5, 26)], [Rational(25, 26)]])]

    vecs = [Matrix([0, 0, 0]), Matrix([1, 2, 3]), Matrix([1, 4, 5])]
    raises(ValueError, lambda: Matrix.orthogonalize(*vecs, rankcheck=True))

    vecs = [Matrix([1, 2, 3]), Matrix([4, 5, 6]), Matrix([7, 8, 9])]
    raises(ValueError, lambda: Matrix.orthogonalize(*vecs, rankcheck=True))

def test_wilkinson():

    wminus, wplus = Matrix.wilkinson(1)
    assert wminus == Matrix([
                                [-1, 1, 0],
                                [1, 0, 1],
                                [0, 1, 1]])
    assert wplus == Matrix([
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1]])

    wminus, wplus = Matrix.wilkinson(3)
    assert wminus == Matrix([
                                [-3,  1,  0, 0, 0, 0, 0],
                                [1, -2,  1, 0, 0, 0, 0],
                                [0,  1, -1, 1, 0, 0, 0],
                                [0,  0,  1, 0, 1, 0, 0],
                                [0,  0,  0, 1, 1, 1, 0],
                                [0,  0,  0, 0, 1, 2, 1],

      [0,  0,  0, 0, 0, 1, 3]])

    assert wplus == Matrix([
                            [3, 1, 0, 0, 0, 0, 0],
                            [1, 2, 1, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 1, 2, 1],
                            [0, 0, 0, 0, 0, 1, 3]])


# CalculusOnlyMatrix tests
@XFAIL
def test_diff():
    x, y = symbols('x y')
    m = CalculusOnlyMatrix(2, 1, [x, y])
    # TODO: currently not working as ``_MinimalMatrix`` cannot be sympified:
    assert m.diff(x) == Matrix(2, 1, [1, 0])


def test_integrate():
    x, y = symbols('x y')
    m = CalculusOnlyMatrix(2, 1, [x, y])
    assert m.integrate(x) == Matrix(2, 1, [x**2/2, y*x])


def test_jacobian2():
    rho, phi = symbols("rho,phi")
    X = CalculusOnlyMatrix(3, 1, [rho*cos(phi), rho*sin(phi), rho**2])
    Y = CalculusOnlyMatrix(2, 1, [rho, phi])
    J = Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)],
        [   2*rho,             0],
    ])
    assert X.jacobian(Y) == J

    m = CalculusOnlyMatrix(2, 2, [1, 2, 3, 4])
    m2 = CalculusOnlyMatrix(4, 1, [1, 2, 3, 4])
    raises(TypeError, lambda: m.jacobian(Matrix([1, 2])))
    raises(TypeError, lambda: m2.jacobian(m))


def test_limit():
    x, y = symbols('x y')
    m = CalculusOnlyMatrix(2, 1, [1/x, y])
    assert m.limit(x, 5) == Matrix(2, 1, [Rational(1, 5), y])


def test_issue_13774():
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    v = [1, 1, 1]
    raises(TypeError, lambda: M*v)
    raises(TypeError, lambda: v*M)

def test_companion():
    x = Symbol('x')
    y = Symbol('y')
    raises(ValueError, lambda: Matrix.companion(1))
    raises(ValueError, lambda: Matrix.companion(Poly([1], x)))
    raises(ValueError, lambda: Matrix.companion(Poly([2, 1], x)))
    raises(ValueError, lambda: Matrix.companion(Poly(x*y, [x, y])))

    c0, c1, c2 = symbols('c0:3')
    assert Matrix.companion(Poly([1, c0], x)) == Matrix([-c0])
    assert Matrix.companion(Poly([1, c1, c0], x)) == \
        Matrix([[0, -c0], [1, -c1]])
    assert Matrix.companion(Poly([1, c2, c1, c0], x)) == \
        Matrix([[0, 0, -c0], [1, 0, -c1], [0, 1, -c2]])

def test_issue_10589():
    x, y, z = symbols("x, y z")
    M1 = Matrix([x, y, z])
    M1 = M1.subs(zip([x, y, z], [1, 2, 3]))
    assert M1 == Matrix([[1], [2], [3]])

    M2 = Matrix([[x, x, x, x, x], [x, x, x, x, x], [x, x, x, x, x]])
    M2 = M2.subs(zip([x], [1]))
    assert M2 == Matrix([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

def test_rmul_pr19860():
    class Foo(ImmutableDenseMatrix):
        _op_priority = MutableDenseMatrix._op_priority + 0.01

    a = Matrix(2, 2, [1, 2, 3, 4])
    b = Foo(2, 2, [1, 2, 3, 4])

    # This would throw a RecursionError: maximum recursion depth
    # since b always has higher priority even after a.as_mutable()
    c = a*b

    assert isinstance(c, Foo)
    assert c == Matrix([[7, 10], [15, 22]])


def test_issue_18956():
    A = Array([[1, 2], [3, 4]])
    B = Matrix([[1,2],[3,4]])
    raises(TypeError, lambda: B + A)
    raises(TypeError, lambda: A + B)


def test__eq__():
    class My(object):
        def __iter__(self):
            yield 1
            yield 2
            return
        def __getitem__(self, i):
            return list(self)[i]
    a = Matrix(2, 1, [1, 2])
    assert a != My()
    class My_sympy(My):
        def _sympy_(self):
            return Matrix(self)
    assert a == My_sympy()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\test_melt.py ===
import re

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    date_range,
    lreshape,
    melt,
    wide_to_long,
)
import pandas._testing as tm


@pytest.fixture
def df():
    res = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    res["id1"] = (res["A"] > 0).astype(np.int64)
    res["id2"] = (res["B"] > 0).astype(np.int64)
    return res


@pytest.fixture
def df1():
    res = DataFrame(
        [
            [1.067683, -1.110463, 0.20867],
            [-1.321405, 0.368915, -1.055342],
            [-0.807333, 0.08298, -0.873361],
        ]
    )
    res.columns = [list("ABC"), list("abc")]
    res.columns.names = ["CAP", "low"]
    return res


@pytest.fixture
def var_name():
    return "var"


@pytest.fixture
def value_name():
    return "val"


class TestMelt:
    def test_top_level_method(self, df):
        result = melt(df)
        assert result.columns.tolist() == ["variable", "value"]

    def test_method_signatures(self, df, df1, var_name, value_name):
        tm.assert_frame_equal(df.melt(), melt(df))

        tm.assert_frame_equal(
            df.melt(id_vars=["id1", "id2"], value_vars=["A", "B"]),
            melt(df, id_vars=["id1", "id2"], value_vars=["A", "B"]),
        )

        tm.assert_frame_equal(
            df.melt(var_name=var_name, value_name=value_name),
            melt(df, var_name=var_name, value_name=value_name),
        )

        tm.assert_frame_equal(df1.melt(col_level=0), melt(df1, col_level=0))

    def test_default_col_names(self, df):
        result = df.melt()
        assert result.columns.tolist() == ["variable", "value"]

        result1 = df.melt(id_vars=["id1"])
        assert result1.columns.tolist() == ["id1", "variable", "value"]

        result2 = df.melt(id_vars=["id1", "id2"])
        assert result2.columns.tolist() == ["id1", "id2", "variable", "value"]

    def test_value_vars(self, df):
        result3 = df.melt(id_vars=["id1", "id2"], value_vars="A")
        assert len(result3) == 10

        result4 = df.melt(id_vars=["id1", "id2"], value_vars=["A", "B"])
        expected4 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                "variable": ["A"] * 10 + ["B"] * 10,
                "value": (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", "variable", "value"],
        )
        tm.assert_frame_equal(result4, expected4)

    @pytest.mark.parametrize("type_", (tuple, list, np.array))
    def test_value_vars_types(self, type_, df):
        # GH 15348
        expected = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                "variable": ["A"] * 10 + ["B"] * 10,
                "value": (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", "variable", "value"],
        )
        result = df.melt(id_vars=["id1", "id2"], value_vars=type_(("A", "B")))
        tm.assert_frame_equal(result, expected)

    def test_vars_work_with_multiindex(self, df1):
        expected = DataFrame(
            {
                ("A", "a"): df1[("A", "a")],
                "CAP": ["B"] * len(df1),
                "low": ["b"] * len(df1),
                "value": df1[("B", "b")],
            },
            columns=[("A", "a"), "CAP", "low", "value"],
        )

        result = df1.melt(id_vars=[("A", "a")], value_vars=[("B", "b")])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "id_vars, value_vars, col_level, expected",
        [
            (
                ["A"],
                ["B"],
                0,
                DataFrame(
                    {
                        "A": {0: 1.067683, 1: -1.321405, 2: -0.807333},
                        "CAP": {0: "B", 1: "B", 2: "B"},
                        "value": {0: -1.110463, 1: 0.368915, 2: 0.08298},
                    }
                ),
            ),
            (
                ["a"],
                ["b"],
                1,
                DataFrame(
                    {
                        "a": {0: 1.067683, 1: -1.321405, 2: -0.807333},
                        "low": {0: "b", 1: "b", 2: "b"},
                        "value": {0: -1.110463, 1: 0.368915, 2: 0.08298},
                    }
                ),
            ),
        ],
    )
    def test_single_vars_work_with_multiindex(
        self, id_vars, value_vars, col_level, expected, df1
    ):
        result = df1.melt(id_vars, value_vars, col_level=col_level)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "id_vars, value_vars",
        [
            [("A", "a"), [("B", "b")]],
            [[("A", "a")], ("B", "b")],
            [("A", "a"), ("B", "b")],
        ],
    )
    def test_tuple_vars_fail_with_multiindex(self, id_vars, value_vars, df1):
        # melt should fail with an informative error message if
        # the columns have a MultiIndex and a tuple is passed
        # for id_vars or value_vars.
        msg = r"(id|value)_vars must be a list of tuples when columns are a MultiIndex"
        with pytest.raises(ValueError, match=msg):
            df1.melt(id_vars=id_vars, value_vars=value_vars)

    def test_custom_var_name(self, df, var_name):
        result5 = df.melt(var_name=var_name)
        assert result5.columns.tolist() == ["var", "value"]

        result6 = df.melt(id_vars=["id1"], var_name=var_name)
        assert result6.columns.tolist() == ["id1", "var", "value"]

        result7 = df.melt(id_vars=["id1", "id2"], var_name=var_name)
        assert result7.columns.tolist() == ["id1", "id2", "var", "value"]

        result8 = df.melt(id_vars=["id1", "id2"], value_vars="A", var_name=var_name)
        assert result8.columns.tolist() == ["id1", "id2", "var", "value"]

        result9 = df.melt(
            id_vars=["id1", "id2"], value_vars=["A", "B"], var_name=var_name
        )
        expected9 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                var_name: ["A"] * 10 + ["B"] * 10,
                "value": (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", var_name, "value"],
        )
        tm.assert_frame_equal(result9, expected9)

    def test_custom_value_name(self, df, value_name):
        result10 = df.melt(value_name=value_name)
        assert result10.columns.tolist() == ["variable", "val"]

        result11 = df.melt(id_vars=["id1"], value_name=value_name)
        assert result11.columns.tolist() == ["id1", "variable", "val"]

        result12 = df.melt(id_vars=["id1", "id2"], value_name=value_name)
        assert result12.columns.tolist() == ["id1", "id2", "variable", "val"]

        result13 = df.melt(
            id_vars=["id1", "id2"], value_vars="A", value_name=value_name
        )
        assert result13.columns.tolist() == ["id1", "id2", "variable", "val"]

        result14 = df.melt(
            id_vars=["id1", "id2"], value_vars=["A", "B"], value_name=value_name
        )
        expected14 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                "variable": ["A"] * 10 + ["B"] * 10,
                value_name: (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", "variable", value_name],
        )
        tm.assert_frame_equal(result14, expected14)

    def test_custom_var_and_value_name(self, df, value_name, var_name):
        result15 = df.melt(var_name=var_name, value_name=value_name)
        assert result15.columns.tolist() == ["var", "val"]

        result16 = df.melt(id_vars=["id1"], var_name=var_name, value_name=value_name)
        assert result16.columns.tolist() == ["id1", "var", "val"]

        result17 = df.melt(
            id_vars=["id1", "id2"], var_name=var_name, value_name=value_name
        )
        assert result17.columns.tolist() == ["id1", "id2", "var", "val"]

        result18 = df.melt(
            id_vars=["id1", "id2"],
            value_vars="A",
            var_name=var_name,
            value_name=value_name,
        )
        assert result18.columns.tolist() == ["id1", "id2", "var", "val"]

        result19 = df.melt(
            id_vars=["id1", "id2"],
            value_vars=["A", "B"],
            var_name=var_name,
            value_name=value_name,
        )
        expected19 = DataFrame(
            {
                "id1": df["id1"].tolist() * 2,
                "id2": df["id2"].tolist() * 2,
                var_name: ["A"] * 10 + ["B"] * 10,
                value_name: (df["A"].tolist() + df["B"].tolist()),
            },
            columns=["id1", "id2", var_name, value_name],
        )
        tm.assert_frame_equal(result19, expected19)

        df20 = df.copy()
        df20.columns.name = "foo"
        result20 = df20.melt()
        assert result20.columns.tolist() == ["foo", "value"]

    @pytest.mark.parametrize("col_level", [0, "CAP"])
    def test_col_level(self, col_level, df1):
        res = df1.melt(col_level=col_level)
        assert res.columns.tolist() == ["CAP", "value"]

    def test_multiindex(self, df1):
        res = df1.melt()
        assert res.columns.tolist() == ["CAP", "low", "value"]

    @pytest.mark.parametrize(
        "col",
        [
            pd.Series(date_range("2010", periods=5, tz="US/Pacific")),
            pd.Series(["a", "b", "c", "a", "d"], dtype="category"),
            pd.Series([0, 1, 0, 0, 0]),
        ],
    )
    def test_pandas_dtypes(self, col):
        # GH 15785
        df = DataFrame(
            {"klass": range(5), "col": col, "attr1": [1, 0, 0, 0, 0], "attr2": col}
        )
        expected_value = pd.concat([pd.Series([1, 0, 0, 0, 0]), col], ignore_index=True)
        result = melt(
            df, id_vars=["klass", "col"], var_name="attribute", value_name="value"
        )
        expected = DataFrame(
            {
                0: list(range(5)) * 2,
                1: pd.concat([col] * 2, ignore_index=True),
                2: ["attr1"] * 5 + ["attr2"] * 5,
                3: expected_value,
            }
        )
        expected.columns = ["klass", "col", "attribute", "value"]
        tm.assert_frame_equal(result, expected)

    def test_preserve_category(self):
        # GH 15853
        data = DataFrame({"A": [1, 2], "B": pd.Categorical(["X", "Y"])})
        result = melt(data, ["B"], ["A"])
        expected = DataFrame(
            {"B": pd.Categorical(["X", "Y"]), "variable": ["A", "A"], "value": [1, 2]}
        )

        tm.assert_frame_equal(result, expected)

    def test_melt_missing_columns_raises(self):
        # GH-23575
        # This test is to ensure that pandas raises an error if melting is
        # attempted with column names absent from the dataframe

        # Generate data
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)), columns=list("abcd")
        )

        # Try to melt with missing `value_vars` column name
        msg = "The following id_vars or value_vars are not present in the DataFrame:"
        with pytest.raises(KeyError, match=msg):
            df.melt(["a", "b"], ["C", "d"])

        # Try to melt with missing `id_vars` column name
        with pytest.raises(KeyError, match=msg):
            df.melt(["A", "b"], ["c", "d"])

        # Multiple missing
        with pytest.raises(
            KeyError,
            match=msg,
        ):
            df.melt(["a", "b", "not_here", "or_there"], ["c", "d"])

        # Multiindex melt fails if column is missing from multilevel melt
        multi = df.copy()
        multi.columns = [list("ABCD"), list("abcd")]
        with pytest.raises(KeyError, match=msg):
            multi.melt([("E", "a")], [("B", "b")])
        # Multiindex fails if column is missing from single level melt
        with pytest.raises(KeyError, match=msg):
            multi.melt(["A"], ["F"], col_level=0)

    def test_melt_mixed_int_str_id_vars(self):
        # GH 29718
        df = DataFrame({0: ["foo"], "a": ["bar"], "b": [1], "d": [2]})
        result = melt(df, id_vars=[0, "a"], value_vars=["b", "d"])
        expected = DataFrame(
            {0: ["foo"] * 2, "a": ["bar"] * 2, "variable": list("bd"), "value": [1, 2]}
        )
        # the df's columns are mixed type and thus object -> preserves object dtype
        expected["variable"] = expected["variable"].astype(object)
        tm.assert_frame_equal(result, expected)

    def test_melt_mixed_int_str_value_vars(self):
        # GH 29718
        df = DataFrame({0: ["foo"], "a": ["bar"]})
        result = melt(df, value_vars=[0, "a"])
        expected = DataFrame({"variable": [0, "a"], "value": ["foo", "bar"]})
        tm.assert_frame_equal(result, expected)

    def test_ignore_index(self):
        # GH 17440
        df = DataFrame({"foo": [0], "bar": [1]}, index=["first"])
        result = melt(df, ignore_index=False)
        expected = DataFrame(
            {"variable": ["foo", "bar"], "value": [0, 1]}, index=["first", "first"]
        )
        tm.assert_frame_equal(result, expected)

    def test_ignore_multiindex(self):
        # GH 17440
        index = pd.MultiIndex.from_tuples(
            [("first", "second"), ("first", "third")], names=["baz", "foobar"]
        )
        df = DataFrame({"foo": [0, 1], "bar": [2, 3]}, index=index)
        result = melt(df, ignore_index=False)

        expected_index = pd.MultiIndex.from_tuples(
            [("first", "second"), ("first", "third")] * 2, names=["baz", "foobar"]
        )
        expected = DataFrame(
            {"variable": ["foo"] * 2 + ["bar"] * 2, "value": [0, 1, 2, 3]},
            index=expected_index,
        )

        tm.assert_frame_equal(result, expected)

    def test_ignore_index_name_and_type(self):
        # GH 17440
        index = Index(["foo", "bar"], dtype="category", name="baz")
        df = DataFrame({"x": [0, 1], "y": [2, 3]}, index=index)
        result = melt(df, ignore_index=False)

        expected_index = Index(["foo", "bar"] * 2, dtype="category", name="baz")
        expected = DataFrame(
            {"variable": ["x", "x", "y", "y"], "value": [0, 1, 2, 3]},
            index=expected_index,
        )

        tm.assert_frame_equal(result, expected)

    def test_melt_with_duplicate_columns(self):
        # GH#41951
        df = DataFrame([["id", 2, 3]], columns=["a", "b", "b"])
        result = df.melt(id_vars=["a"], value_vars=["b"])
        expected = DataFrame(
            [["id", "b", 2], ["id", "b", 3]], columns=["a", "variable", "value"]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["Int8", "Int64"])
    def test_melt_ea_dtype(self, dtype):
        # GH#41570
        df = DataFrame(
            {
                "a": pd.Series([1, 2], dtype="Int8"),
                "b": pd.Series([3, 4], dtype=dtype),
            }
        )
        result = df.melt()
        expected = DataFrame(
            {
                "variable": ["a", "a", "b", "b"],
                "value": pd.Series([1, 2, 3, 4], dtype=dtype),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_melt_ea_columns(self):
        # GH 54297
        df = DataFrame(
            {
                "A": {0: "a", 1: "b", 2: "c"},
                "B": {0: 1, 1: 3, 2: 5},
                "C": {0: 2, 1: 4, 2: 6},
            }
        )
        df.columns = df.columns.astype("string[python]")
        result = df.melt(id_vars=["A"], value_vars=["B"])
        expected = DataFrame(
            {
                "A": list("abc"),
                "variable": pd.Series(["B"] * 3, dtype="string[python]"),
                "value": [1, 3, 5],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_melt_preserves_datetime(self):
        df = DataFrame(
            data=[
                {
                    "type": "A0",
                    "start_date": pd.Timestamp("2023/03/01", tz="Asia/Tokyo"),
                    "end_date": pd.Timestamp("2023/03/10", tz="Asia/Tokyo"),
                },
                {
                    "type": "A1",
                    "start_date": pd.Timestamp("2023/03/01", tz="Asia/Tokyo"),
                    "end_date": pd.Timestamp("2023/03/11", tz="Asia/Tokyo"),
                },
            ],
            index=["aaaa", "bbbb"],
        )
        result = df.melt(
            id_vars=["type"],
            value_vars=["start_date", "end_date"],
            var_name="start/end",
            value_name="date",
        )
        expected = DataFrame(
            {
                "type": {0: "A0", 1: "A1", 2: "A0", 3: "A1"},
                "start/end": {
                    0: "start_date",
                    1: "start_date",
                    2: "end_date",
                    3: "end_date",
                },
                "date": {
                    0: pd.Timestamp("2023-03-01 00:00:00+0900", tz="Asia/Tokyo"),
                    1: pd.Timestamp("2023-03-01 00:00:00+0900", tz="Asia/Tokyo"),
                    2: pd.Timestamp("2023-03-10 00:00:00+0900", tz="Asia/Tokyo"),
                    3: pd.Timestamp("2023-03-11 00:00:00+0900", tz="Asia/Tokyo"),
                },
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_melt_allows_non_scalar_id_vars(self):
        df = DataFrame(
            data={"a": [1, 2, 3], "b": [4, 5, 6]},
            index=["11", "22", "33"],
        )
        result = df.melt(
            id_vars="a",
            var_name=0,
            value_name=1,
        )
        expected = DataFrame({"a": [1, 2, 3], 0: ["b"] * 3, 1: [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_melt_allows_non_string_var_name(self):
        df = DataFrame(
            data={"a": [1, 2, 3], "b": [4, 5, 6]},
            index=["11", "22", "33"],
        )
        result = df.melt(
            id_vars=["a"],
            var_name=0,
            value_name=1,
        )
        expected = DataFrame({"a": [1, 2, 3], 0: ["b"] * 3, 1: [4, 5, 6]})
        tm.assert_frame_equal(result, expected)

    def test_melt_non_scalar_var_name_raises(self):
        df = DataFrame(
            data={"a": [1, 2, 3], "b": [4, 5, 6]},
            index=["11", "22", "33"],
        )
        with pytest.raises(ValueError, match=r".* must be a scalar."):
            df.melt(id_vars=["a"], var_name=[1, 2])


class TestLreshape:
    def test_pairs(self):
        data = {
            "birthdt": [
                "08jan2009",
                "20dec2008",
                "30dec2008",
                "21dec2008",
                "11jan2009",
            ],
            "birthwt": [1766, 3301, 1454, 3139, 4133],
            "id": [101, 102, 103, 104, 105],
            "sex": ["Male", "Female", "Female", "Female", "Female"],
            "visitdt1": [
                "11jan2009",
                "22dec2008",
                "04jan2009",
                "29dec2008",
                "20jan2009",
            ],
            "visitdt2": ["21jan2009", np.nan, "22jan2009", "31dec2008", "03feb2009"],
            "visitdt3": ["05feb2009", np.nan, np.nan, "02jan2009", "15feb2009"],
            "wt1": [1823, 3338, 1549, 3298, 4306],
            "wt2": [2011.0, np.nan, 1892.0, 3338.0, 4575.0],
            "wt3": [2293.0, np.nan, np.nan, 3377.0, 4805.0],
        }

        df = DataFrame(data)

        spec = {
            "visitdt": [f"visitdt{i:d}" for i in range(1, 4)],
            "wt": [f"wt{i:d}" for i in range(1, 4)],
        }
        result = lreshape(df, spec)

        exp_data = {
            "birthdt": [
                "08jan2009",
                "20dec2008",
                "30dec2008",
                "21dec2008",
                "11jan2009",
                "08jan2009",
                "30dec2008",
                "21dec2008",
                "11jan2009",
                "08jan2009",
                "21dec2008",
                "11jan2009",
            ],
            "birthwt": [
                1766,
                3301,
                1454,
                3139,
                4133,
                1766,
                1454,
                3139,
                4133,
                1766,
                3139,
                4133,
            ],
            "id": [101, 102, 103, 104, 105, 101, 103, 104, 105, 101, 104, 105],
            "sex": [
                "Male",
                "Female",
                "Female",
                "Female",
                "Female",
                "Male",
                "Female",
                "Female",
                "Female",
                "Male",
                "Female",
                "Female",
            ],
            "visitdt": [
                "11jan2009",
                "22dec2008",
                "04jan2009",
                "29dec2008",
                "20jan2009",
                "21jan2009",
                "22jan2009",
                "31dec2008",
                "03feb2009",
                "05feb2009",
                "02jan2009",
                "15feb2009",
            ],
            "wt": [
                1823.0,
                3338.0,
                1549.0,
                3298.0,
                4306.0,
                2011.0,
                1892.0,
                3338.0,
                4575.0,
                2293.0,
                3377.0,
                4805.0,
            ],
        }
        exp = DataFrame(exp_data, columns=result.columns)
        tm.assert_frame_equal(result, exp)

        result = lreshape(df, spec, dropna=False)
        exp_data = {
            "birthdt": [
                "08jan2009",
                "20dec2008",
                "30dec2008",
                "21dec2008",
                "11jan2009",
                "08jan2009",
                "20dec2008",
                "30dec2008",
                "21dec2008",
                "11jan2009",
                "08jan2009",
                "20dec2008",
                "30dec2008",
                "21dec2008",
                "11jan2009",
            ],
            "birthwt": [
                1766,
                3301,
                1454,
                3139,
                4133,
                1766,
                3301,
                1454,
                3139,
                4133,
                1766,
                3301,
                1454,
                3139,
                4133,
            ],
            "id": [
                101,
                102,
                103,
                104,
                105,
                101,
                102,
                103,
                104,
                105,
                101,
                102,
                103,
                104,
                105,
            ],
            "sex": [
                "Male",
                "Female",
                "Female",
                "Female",
                "Female",
                "Male",
                "Female",
                "Female",
                "Female",
                "Female",
                "Male",
                "Female",
                "Female",
                "Female",
                "Female",
            ],
            "visitdt": [
                "11jan2009",
                "22dec2008",
                "04jan2009",
                "29dec2008",
                "20jan2009",
                "21jan2009",
                np.nan,
                "22jan2009",
                "31dec2008",
                "03feb2009",
                "05feb2009",
                np.nan,
                np.nan,
                "02jan2009",
                "15feb2009",
            ],
            "wt": [
                1823.0,
                3338.0,
                1549.0,
                3298.0,
                4306.0,
                2011.0,
                np.nan,
                1892.0,
                3338.0,
                4575.0,
                2293.0,
                np.nan,
                np.nan,
                3377.0,
                4805.0,
            ],
        }
        exp = DataFrame(exp_data, columns=result.columns)
        tm.assert_frame_equal(result, exp)

        spec = {
            "visitdt": [f"visitdt{i:d}" for i in range(1, 3)],
            "wt": [f"wt{i:d}" for i in range(1, 4)],
        }
        msg = "All column lists must be same length"
        with pytest.raises(ValueError, match=msg):
            lreshape(df, spec)


class TestWideToLong:
    def test_simple(self):
        x = np.random.default_rng(2).standard_normal(3)
        df = DataFrame(
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
        expected = DataFrame(exp_data)
        expected = expected.set_index(["id", "year"])[["X", "A", "B"]]
        result = wide_to_long(df, ["A", "B"], i="id", j="year")
        tm.assert_frame_equal(result, expected)

    def test_stubs(self):
        # GH9204 wide_to_long call should not modify 'stubs' list
        df = DataFrame([[0, 1, 2, 3, 8], [4, 5, 6, 7, 9]])
        df.columns = ["id", "inc1", "inc2", "edu1", "edu2"]
        stubs = ["inc", "edu"]

        wide_to_long(df, stubs, i="id", j="age")

        assert stubs == ["inc", "edu"]

    def test_separating_character(self):
        # GH14779

        x = np.random.default_rng(2).standard_normal(3)
        df = DataFrame(
            {
                "A.1970": {0: "a", 1: "b", 2: "c"},
                "A.1980": {0: "d", 1: "e", 2: "f"},
                "B.1970": {0: 2.5, 1: 1.2, 2: 0.7},
                "B.1980": {0: 3.2, 1: 1.3, 2: 0.1},
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
        expected = DataFrame(exp_data)
        expected = expected.set_index(["id", "year"])[["X", "A", "B"]]
        result = wide_to_long(df, ["A", "B"], i="id", j="year", sep=".")
        tm.assert_frame_equal(result, expected)

    def test_escapable_characters(self):
        x = np.random.default_rng(2).standard_normal(3)
        df = DataFrame(
            {
                "A(quarterly)1970": {0: "a", 1: "b", 2: "c"},
                "A(quarterly)1980": {0: "d", 1: "e", 2: "f"},
                "B(quarterly)1970": {0: 2.5, 1: 1.2, 2: 0.7},
                "B(quarterly)1980": {0: 3.2, 1: 1.3, 2: 0.1},
                "X": dict(zip(range(3), x)),
            }
        )
        df["id"] = df.index
        exp_data = {
            "X": x.tolist() + x.tolist(),
            "A(quarterly)": ["a", "b", "c", "d", "e", "f"],
            "B(quarterly)": [2.5, 1.2, 0.7, 3.2, 1.3, 0.1],
            "year": [1970, 1970, 1970, 1980, 1980, 1980],
            "id": [0, 1, 2, 0, 1, 2],
        }
        expected = DataFrame(exp_data)
        expected = expected.set_index(["id", "year"])[
            ["X", "A(quarterly)", "B(quarterly)"]
        ]
        result = wide_to_long(df, ["A(quarterly)", "B(quarterly)"], i="id", j="year")
        tm.assert_frame_equal(result, expected)

    def test_unbalanced(self):
        # test that we can have a varying amount of time variables
        df = DataFrame(
            {
                "A2010": [1.0, 2.0],
                "A2011": [3.0, 4.0],
                "B2010": [5.0, 6.0],
                "X": ["X1", "X2"],
            }
        )
        df["id"] = df.index
        exp_data = {
            "X": ["X1", "X2", "X1", "X2"],
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [5.0, 6.0, np.nan, np.nan],
            "id": [0, 1, 0, 1],
            "year": [2010, 2010, 2011, 2011],
        }
        expected = DataFrame(exp_data)
        expected = expected.set_index(["id", "year"])[["X", "A", "B"]]
        result = wide_to_long(df, ["A", "B"], i="id", j="year")
        tm.assert_frame_equal(result, expected)

    def test_character_overlap(self):
        # Test we handle overlapping characters in both id_vars and value_vars
        df = DataFrame(
            {
                "A11": ["a11", "a22", "a33"],
                "A12": ["a21", "a22", "a23"],
                "B11": ["b11", "b12", "b13"],
                "B12": ["b21", "b22", "b23"],
                "BB11": [1, 2, 3],
                "BB12": [4, 5, 6],
                "BBBX": [91, 92, 93],
                "BBBZ": [91, 92, 93],
            }
        )
        df["id"] = df.index
        expected = DataFrame(
            {
                "BBBX": [91, 92, 93, 91, 92, 93],
                "BBBZ": [91, 92, 93, 91, 92, 93],
                "A": ["a11", "a22", "a33", "a21", "a22", "a23"],
                "B": ["b11", "b12", "b13", "b21", "b22", "b23"],
                "BB": [1, 2, 3, 4, 5, 6],
                "id": [0, 1, 2, 0, 1, 2],
                "year": [11, 11, 11, 12, 12, 12],
            }
        )
        expected = expected.set_index(["id", "year"])[["BBBX", "BBBZ", "A", "B", "BB"]]
        result = wide_to_long(df, ["A", "B", "BB"], i="id", j="year")
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_invalid_separator(self):
        # if an invalid separator is supplied a empty data frame is returned
        sep = "nope!"
        df = DataFrame(
            {
                "A2010": [1.0, 2.0],
                "A2011": [3.0, 4.0],
                "B2010": [5.0, 6.0],
                "X": ["X1", "X2"],
            }
        )
        df["id"] = df.index
        exp_data = {
            "X": "",
            "A2010": [],
            "A2011": [],
            "B2010": [],
            "id": [],
            "year": [],
            "A": [],
            "B": [],
        }
        expected = DataFrame(exp_data).astype({"year": np.int64})
        expected = expected.set_index(["id", "year"])[
            ["X", "A2010", "A2011", "B2010", "A", "B"]
        ]
        expected.index = expected.index.set_levels([0, 1], level=0)
        result = wide_to_long(df, ["A", "B"], i="id", j="year", sep=sep)
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_num_string_disambiguation(self):
        # Test that we can disambiguate number value_vars from
        # string value_vars
        df = DataFrame(
            {
                "A11": ["a11", "a22", "a33"],
                "A12": ["a21", "a22", "a23"],
                "B11": ["b11", "b12", "b13"],
                "B12": ["b21", "b22", "b23"],
                "BB11": [1, 2, 3],
                "BB12": [4, 5, 6],
                "Arating": [91, 92, 93],
                "Arating_old": [91, 92, 93],
            }
        )
        df["id"] = df.index
        expected = DataFrame(
            {
                "Arating": [91, 92, 93, 91, 92, 93],
                "Arating_old": [91, 92, 93, 91, 92, 93],
                "A": ["a11", "a22", "a33", "a21", "a22", "a23"],
                "B": ["b11", "b12", "b13", "b21", "b22", "b23"],
                "BB": [1, 2, 3, 4, 5, 6],
                "id": [0, 1, 2, 0, 1, 2],
                "year": [11, 11, 11, 12, 12, 12],
            }
        )
        expected = expected.set_index(["id", "year"])[
            ["Arating", "Arating_old", "A", "B", "BB"]
        ]
        result = wide_to_long(df, ["A", "B", "BB"], i="id", j="year")
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_invalid_suffixtype(self):
        # If all stubs names end with a string, but a numeric suffix is
        # assumed,  an empty data frame is returned
        df = DataFrame(
            {
                "Aone": [1.0, 2.0],
                "Atwo": [3.0, 4.0],
                "Bone": [5.0, 6.0],
                "X": ["X1", "X2"],
            }
        )
        df["id"] = df.index
        exp_data = {
            "X": "",
            "Aone": [],
            "Atwo": [],
            "Bone": [],
            "id": [],
            "year": [],
            "A": [],
            "B": [],
        }
        expected = DataFrame(exp_data).astype({"year": np.int64})

        expected = expected.set_index(["id", "year"])
        expected.index = expected.index.set_levels([0, 1], level=0)
        result = wide_to_long(df, ["A", "B"], i="id", j="year")
        tm.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))

    def test_multiple_id_columns(self):
        # Taken from http://www.ats.ucla.edu/stat/stata/modules/reshapel.htm
        df = DataFrame(
            {
                "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
                "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
            }
        )
        expected = DataFrame(
            {
                "ht": [
                    2.8,
                    3.4,
                    2.9,
                    3.8,
                    2.2,
                    2.9,
                    2.0,
                    3.2,
                    1.8,
                    2.8,
                    1.9,
                    2.4,
                    2.2,
                    3.3,
                    2.3,
                    3.4,
                    2.1,
                    2.9,
                ],
                "famid": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                "birth": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
                "age": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            }
        )
        expected = expected.set_index(["famid", "birth", "age"])[["ht"]]
        result = wide_to_long(df, "ht", i=["famid", "birth"], j="age")
        tm.assert_frame_equal(result, expected)

    def test_non_unique_idvars(self):
        # GH16382
        # Raise an error message if non unique id vars (i) are passed
        df = DataFrame(
            {"A_A1": [1, 2, 3, 4, 5], "B_B1": [1, 2, 3, 4, 5], "x": [1, 1, 1, 1, 1]}
        )
        msg = "the id variables need to uniquely identify each row"
        with pytest.raises(ValueError, match=msg):
            wide_to_long(df, ["A_A", "B_B"], i="x", j="colname")

    def test_cast_j_int(self):
        df = DataFrame(
            {
                "actor_1": ["CCH Pounder", "Johnny Depp", "Christoph Waltz"],
                "actor_2": ["Joel David Moore", "Orlando Bloom", "Rory Kinnear"],
                "actor_fb_likes_1": [1000.0, 40000.0, 11000.0],
                "actor_fb_likes_2": [936.0, 5000.0, 393.0],
                "title": ["Avatar", "Pirates of the Caribbean", "Spectre"],
            }
        )

        expected = DataFrame(
            {
                "actor": [
                    "CCH Pounder",
                    "Johnny Depp",
                    "Christoph Waltz",
                    "Joel David Moore",
                    "Orlando Bloom",
                    "Rory Kinnear",
                ],
                "actor_fb_likes": [1000.0, 40000.0, 11000.0, 936.0, 5000.0, 393.0],
                "num": [1, 1, 1, 2, 2, 2],
                "title": [
                    "Avatar",
                    "Pirates of the Caribbean",
                    "Spectre",
                    "Avatar",
                    "Pirates of the Caribbean",
                    "Spectre",
                ],
            }
        ).set_index(["title", "num"])
        result = wide_to_long(
            df, ["actor", "actor_fb_likes"], i="title", j="num", sep="_"
        )

        tm.assert_frame_equal(result, expected)

    def test_identical_stubnames(self):
        df = DataFrame(
            {
                "A2010": [1.0, 2.0],
                "A2011": [3.0, 4.0],
                "B2010": [5.0, 6.0],
                "A": ["X1", "X2"],
            }
        )
        msg = "stubname can't be identical to a column name"
        with pytest.raises(ValueError, match=msg):
            wide_to_long(df, ["A", "B"], i="A", j="colname")

    def test_nonnumeric_suffix(self):
        df = DataFrame(
            {
                "treatment_placebo": [1.0, 2.0],
                "treatment_test": [3.0, 4.0],
                "result_placebo": [5.0, 6.0],
                "A": ["X1", "X2"],
            }
        )
        expected = DataFrame(
            {
                "A": ["X1", "X2", "X1", "X2"],
                "colname": ["placebo", "placebo", "test", "test"],
                "result": [5.0, 6.0, np.nan, np.nan],
                "treatment": [1.0, 2.0, 3.0, 4.0],
            }
        )
        expected = expected.set_index(["A", "colname"])
        result = wide_to_long(
            df, ["result", "treatment"], i="A", j="colname", suffix="[a-z]+", sep="_"
        )
        tm.assert_frame_equal(result, expected)

    def test_mixed_type_suffix(self):
        df = DataFrame(
            {
                "A": ["X1", "X2"],
                "result_1": [0, 9],
                "result_foo": [5.0, 6.0],
                "treatment_1": [1.0, 2.0],
                "treatment_foo": [3.0, 4.0],
            }
        )
        expected = DataFrame(
            {
                "A": ["X1", "X2", "X1", "X2"],
                "colname": ["1", "1", "foo", "foo"],
                "result": [0.0, 9.0, 5.0, 6.0],
                "treatment": [1.0, 2.0, 3.0, 4.0],
            }
        ).set_index(["A", "colname"])
        result = wide_to_long(
            df, ["result", "treatment"], i="A", j="colname", suffix=".+", sep="_"
        )
        tm.assert_frame_equal(result, expected)

    def test_float_suffix(self):
        df = DataFrame(
            {
                "treatment_1.1": [1.0, 2.0],
                "treatment_2.1": [3.0, 4.0],
                "result_1.2": [5.0, 6.0],
                "result_1": [0, 9],
                "A": ["X1", "X2"],
            }
        )
        expected = DataFrame(
            {
                "A": ["X1", "X2", "X1", "X2", "X1", "X2", "X1", "X2"],
                "colname": [1.2, 1.2, 1.0, 1.0, 1.1, 1.1, 2.1, 2.1],
                "result": [5.0, 6.0, 0.0, 9.0, np.nan, np.nan, np.nan, np.nan],
                "treatment": [np.nan, np.nan, np.nan, np.nan, 1.0, 2.0, 3.0, 4.0],
            }
        )
        expected = expected.set_index(["A", "colname"])
        result = wide_to_long(
            df, ["result", "treatment"], i="A", j="colname", suffix="[0-9.]+", sep="_"
        )
        tm.assert_frame_equal(result, expected)

    def test_col_substring_of_stubname(self):
        # GH22468
        # Don't raise ValueError when a column name is a substring
        # of a stubname that's been passed as a string
        wide_data = {
            "node_id": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            "A": {0: 0.80, 1: 0.0, 2: 0.25, 3: 1.0, 4: 0.81},
            "PA0": {0: 0.74, 1: 0.56, 2: 0.56, 3: 0.98, 4: 0.6},
            "PA1": {0: 0.77, 1: 0.64, 2: 0.52, 3: 0.98, 4: 0.67},
            "PA3": {0: 0.34, 1: 0.70, 2: 0.52, 3: 0.98, 4: 0.67},
        }
        wide_df = DataFrame.from_dict(wide_data)
        expected = wide_to_long(wide_df, stubnames=["PA"], i=["node_id", "A"], j="time")
        result = wide_to_long(wide_df, stubnames="PA", i=["node_id", "A"], j="time")
        tm.assert_frame_equal(result, expected)

    def test_raise_of_column_name_value(self):
        # GH34731, enforced in 2.0
        # raise a ValueError if the resultant value column name matches
        # a name in the dataframe already (default name is "value")
        df = DataFrame({"col": list("ABC"), "value": range(10, 16, 2)})

        with pytest.raises(
            ValueError, match=re.escape("value_name (value) cannot match")
        ):
            df.melt(id_vars="value", value_name="value")

    def test_missing_stubname(self, request, any_string_dtype, using_infer_string):
        if using_infer_string and any_string_dtype == "object":
            # triggers object dtype inference warning of dtype=object
            request.applymarker(pytest.mark.xfail(reason="TODO(infer_string)"))
        # GH46044
        df = DataFrame({"id": ["1", "2"], "a-1": [100, 200], "a-2": [300, 400]})
        df = df.astype({"id": any_string_dtype})
        result = wide_to_long(
            df,
            stubnames=["a", "b"],
            i="id",
            j="num",
            sep="-",
        )
        index = Index(
            [("1", 1), ("2", 1), ("1", 2), ("2", 2)],
            name=("id", "num"),
        )
        expected = DataFrame(
            {"a": [100, 200, 300, 400], "b": [np.nan] * 4},
            index=index,
        )
        new_level = expected.index.levels[0].astype(any_string_dtype)
        if any_string_dtype == "object":
            new_level = expected.index.levels[0].astype("str")
        expected.index = expected.index.set_levels(new_level, level=0)
        tm.assert_frame_equal(result, expected)


def test_wide_to_long_string_columns(string_storage):
    # GH 57066
    string_dtype = pd.StringDtype(string_storage, na_value=np.nan)
    df = DataFrame(
        {
            "ID": {0: 1},
            "R_test1": {0: 1},
            "R_test2": {0: 1},
            "R_test3": {0: 2},
            "D": {0: 1},
        }
    )
    df.columns = df.columns.astype(string_dtype)
    result = wide_to_long(
        df, stubnames="R", i="ID", j="UNPIVOTED", sep="_", suffix=".*"
    )
    expected = DataFrame(
        [[1, 1], [1, 1], [1, 2]],
        columns=Index(["D", "R"]),
        index=pd.MultiIndex.from_arrays(
            [
                [1, 1, 1],
                Index(["test1", "test2", "test3"], dtype=string_dtype),
            ],
            names=["ID", "UNPIVOTED"],
        ),
    )
    tm.assert_frame_equal(result, expected)