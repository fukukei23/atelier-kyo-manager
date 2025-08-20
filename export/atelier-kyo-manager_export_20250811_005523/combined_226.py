
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_distributedmodules.py ===
"""Tests for sparse distributed modules. """

from sympy.polys.distributedmodules import (
    sdm_monomial_mul, sdm_monomial_deg, sdm_monomial_divides,
    sdm_add, sdm_LM, sdm_LT, sdm_mul_term, sdm_zero, sdm_deg,
    sdm_LC, sdm_from_dict,
    sdm_spoly, sdm_ecart, sdm_nf_mora, sdm_groebner,
    sdm_from_vector, sdm_to_vector, sdm_monomial_lcm
)

from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ

from sympy.abc import x, y, z


def test_sdm_monomial_mul():
    assert sdm_monomial_mul((1, 1, 0), (1, 3)) == (1, 2, 3)


def test_sdm_monomial_deg():
    assert sdm_monomial_deg((5, 2, 1)) == 3


def test_sdm_monomial_lcm():
    assert sdm_monomial_lcm((1, 2, 3), (1, 5, 0)) == (1, 5, 3)


def test_sdm_monomial_divides():
    assert sdm_monomial_divides((1, 0, 0), (1, 0, 0)) is True
    assert sdm_monomial_divides((1, 0, 0), (1, 2, 1)) is True
    assert sdm_monomial_divides((5, 1, 1), (5, 2, 1)) is True

    assert sdm_monomial_divides((1, 0, 0), (2, 0, 0)) is False
    assert sdm_monomial_divides((1, 1, 0), (1, 0, 0)) is False
    assert sdm_monomial_divides((5, 1, 2), (5, 0, 1)) is False


def test_sdm_LC():
    assert sdm_LC([((1, 2, 3), QQ(5))], QQ) == QQ(5)


def test_sdm_from_dict():
    dic = {(1, 2, 1, 1): QQ(1), (1, 1, 2, 1): QQ(1), (1, 0, 2, 1): QQ(1),
           (1, 0, 0, 3): QQ(1), (1, 1, 1, 0): QQ(1)}
    assert sdm_from_dict(dic, grlex) == \
        [((1, 2, 1, 1), QQ(1)), ((1, 1, 2, 1), QQ(1)),
         ((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1))]

# TODO test to_dict?


def test_sdm_add():
    assert sdm_add([((1, 1, 1), QQ(1))], [((2, 0, 0), QQ(1))], lex, QQ) == \
        [((2, 0, 0), QQ(1)), ((1, 1, 1), QQ(1))]
    assert sdm_add([((1, 1, 1), QQ(1))], [((1, 1, 1), QQ(-1))], lex, QQ) == []
    assert sdm_add([((1, 0, 0), QQ(1))], [((1, 0, 0), QQ(2))], lex, QQ) == \
        [((1, 0, 0), QQ(3))]
    assert sdm_add([((1, 0, 1), QQ(1))], [((1, 1, 0), QQ(1))], lex, QQ) == \
        [((1, 1, 0), QQ(1)), ((1, 0, 1), QQ(1))]


def test_sdm_LM():
    dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(1), (4, 0, 1): QQ(1)}
    assert sdm_LM(sdm_from_dict(dic, lex)) == (4, 0, 1)


def test_sdm_LT():
    dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(2), (4, 0, 1): QQ(3)}
    assert sdm_LT(sdm_from_dict(dic, lex)) == ((4, 0, 1), QQ(3))


def test_sdm_mul_term():
    assert sdm_mul_term([((1, 0, 0), QQ(1))], ((0, 0), QQ(0)), lex, QQ) == []
    assert sdm_mul_term([], ((1, 0), QQ(1)), lex, QQ) == []
    assert sdm_mul_term([((1, 0, 0), QQ(1))], ((1, 0), QQ(1)), lex, QQ) == \
        [((1, 1, 0), QQ(1))]
    f = [((2, 0, 1), QQ(4)), ((1, 1, 0), QQ(3))]
    assert sdm_mul_term(f, ((1, 1), QQ(2)), lex, QQ) == \
        [((2, 1, 2), QQ(8)), ((1, 2, 1), QQ(6))]


def test_sdm_zero():
    assert sdm_zero() == []


def test_sdm_deg():
    assert sdm_deg([((1, 2, 3), 1), ((10, 0, 1), 1), ((2, 3, 4), 4)]) == 7


def test_sdm_spoly():
    f = [((2, 1, 1), QQ(1)), ((1, 0, 1), QQ(1))]
    g = [((2, 3, 0), QQ(1))]
    h = [((1, 2, 3), QQ(1))]
    assert sdm_spoly(f, h, lex, QQ) == []
    assert sdm_spoly(f, g, lex, QQ) == [((1, 2, 1), QQ(1))]


def test_sdm_ecart():
    assert sdm_ecart([((1, 2, 3), 1), ((1, 0, 1), 1)]) == 0
    assert sdm_ecart([((2, 2, 1), 1), ((1, 5, 1), 1)]) == 3


def test_sdm_nf_mora():
    f = sdm_from_dict({(1, 2, 1, 1): QQ(1), (1, 1, 2, 1): QQ(1),
                (1, 0, 2, 1): QQ(1), (1, 0, 0, 3): QQ(1), (1, 1, 1, 0): QQ(1)},
        grlex)
    f1 = sdm_from_dict({(1, 1, 1, 0): QQ(1), (1, 0, 2, 0): QQ(1),
                        (1, 0, 0, 0): QQ(-1)}, grlex)
    f2 = sdm_from_dict({(1, 1, 1, 0): QQ(1)}, grlex)
    (id0, id1, id2) = [sdm_from_dict({(i, 0, 0, 0): QQ(1)}, grlex)
                       for i in range(3)]

    assert sdm_nf_mora(f, [f1, f2], grlex, QQ, phantom=(id0, [id1, id2])) == \
        ([((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1)),
          ((1, 1, 0, 1), QQ(1))],
         [((1, 1, 0, 1), QQ(-1)), ((0, 0, 0, 0), QQ(1))])
    assert sdm_nf_mora(f, [f2, f1], grlex, QQ, phantom=(id0, [id2, id1])) == \
        ([((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1))],
         [((2, 1, 0, 1), QQ(-1)), ((2, 0, 1, 1), QQ(-1)), ((0, 0, 0, 0), QQ(1))])

    f = sdm_from_vector([x*z, y**2 + y*z - z, y], lex, QQ, gens=[x, y, z])
    f1 = sdm_from_vector([x, y, 1], lex, QQ, gens=[x, y, z])
    f2 = sdm_from_vector([x*y, z, z**2], lex, QQ, gens=[x, y, z])
    assert sdm_nf_mora(f, [f1, f2], lex, QQ) == \
        sdm_nf_mora(f, [f2, f1], lex, QQ) == \
        [((1, 0, 1, 1), QQ(1)), ((1, 0, 0, 1), QQ(-1)), ((0, 1, 1, 0), QQ(-1)),
         ((0, 1, 0, 1), QQ(1))]


def test_conversion():
    f = [x**2 + y**2, 2*z]
    g = [((1, 0, 0, 1), QQ(2)), ((0, 2, 0, 0), QQ(1)), ((0, 0, 2, 0), QQ(1))]
    assert sdm_to_vector(g, [x, y, z], QQ) == f
    assert sdm_from_vector(f, lex, QQ) == g
    assert sdm_from_vector(
        [x, 1], lex, QQ) == [((1, 0), QQ(1)), ((0, 1), QQ(1))]
    assert sdm_to_vector([((1, 1, 0, 0), 1)], [x, y, z], QQ, n=3) == [0, x, 0]
    assert sdm_from_vector([0, 0], lex, QQ, gens=[x, y]) == sdm_zero()


def test_nontrivial():
    gens = [x, y, z]

    def contains(I, f):
        S = [sdm_from_vector([g], lex, QQ, gens=gens) for g in I]
        G = sdm_groebner(S, sdm_nf_mora, lex, QQ)
        return sdm_nf_mora(sdm_from_vector([f], lex, QQ, gens=gens),
                           G, lex, QQ) == sdm_zero()

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**3)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4 + y**3 + 2*z*y*x)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y*z)
    assert contains([x, 1 + x + y, 5 - 7*y], 1)
    assert contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**3)
    assert not contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**2 + y**2)

    # compare local order
    assert not contains([x*(1 + x + y), y*(1 + z)], x)
    assert not contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_local():
    igrlex = InverseOrder(grlex)
    gens = [x, y, z]

    def contains(I, f):
        S = [sdm_from_vector([g], igrlex, QQ, gens=gens) for g in I]
        G = sdm_groebner(S, sdm_nf_mora, igrlex, QQ)
        return sdm_nf_mora(sdm_from_vector([f], lex, QQ, gens=gens),
                           G, lex, QQ) == sdm_zero()
    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_uncovered_line():
    gens = [x, y]
    f1 = sdm_zero()
    f2 = sdm_from_vector([x, 0], lex, QQ, gens=gens)
    f3 = sdm_from_vector([0, y], lex, QQ, gens=gens)

    assert sdm_spoly(f1, f2, lex, QQ) == sdm_zero()
    assert sdm_spoly(f3, f2, lex, QQ) == sdm_zero()


def test_chain_criterion():
    gens = [x]
    f1 = sdm_from_vector([1, x], grlex, QQ, gens=gens)
    f2 = sdm_from_vector([0, x - 2], grlex, QQ, gens=gens)
    assert len(sdm_groebner([f1, f2], sdm_nf_mora, grlex, QQ)) == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_generator_mt19937_regressions.py ===
import pytest

import numpy as np
from numpy.random import MT19937, Generator
from numpy.testing import assert_, assert_array_equal


class TestRegression:

    def setup_method(self):
        self.mt19937 = Generator(MT19937(121263137472525314065))

    def test_vonmises_range(self):
        # Make sure generated random variables are in [-pi, pi].
        # Regression test for ticket #986.
        for mu in np.linspace(-7., 7., 5):
            r = self.mt19937.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_hypergeometric_range(self):
        # Test for ticket #921
        assert_(np.all(self.mt19937.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(self.mt19937.hypergeometric(18, 3, 11, size=10) > 0))

        # Test for ticket #5623
        args = (2**20 - 2, 2**20 - 2, 2**20 - 2)  # Check for 32-bit systems
        assert_(self.mt19937.hypergeometric(*args) > 0)

    def test_logseries_convergence(self):
        # Test for ticket #923
        N = 1000
        rvsn = self.mt19937.logseries(0.8, size=N)
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
            mt19937 = Generator(MT19937(12345))
            shuffled = np.array(t, dtype=object)
            mt19937.shuffle(shuffled)
            expected = np.array([t[2], t[0], t[3], t[1]], dtype=object)
            assert_array_equal(np.array(shuffled, dtype=object), expected)

    def test_call_within_randomstate(self):
        # Check that custom BitGenerator does not call into global state
        res = np.array([1, 8, 0, 1, 5, 3, 3, 8, 1, 4])
        for i in range(3):
            mt19937 = Generator(MT19937(i))
            m = Generator(MT19937(4321))
            # If m.state is not honored, the result will change
            assert_array_equal(m.choice(10, size=10, p=np.ones(10) / 10.), res)

    def test_multivariate_normal_size_types(self):
        # Test for multivariate_normal issue with 'size' argument.
        # Check that the multivariate_normal size argument can be a
        # numpy integer.
        self.mt19937.multivariate_normal([0], [[0]], size=1)
        self.mt19937.multivariate_normal([0], [[0]], size=np.int_(1))
        self.mt19937.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        # Test that beta with small a and b parameters does not produce
        # NaNs due to roundoff errors causing 0 / 0, gh-5851
        x = self.mt19937.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in mt19937.beta')

    def test_beta_very_small_parameters(self):
        # gh-24203: beta would hang with very small parameters.
        self.mt19937.beta(1e-49, 1e-40)

    def test_beta_ridiculously_small_parameters(self):
        # gh-24266: beta would generate nan when the parameters
        # were subnormal or a small multiple of the smallest normal.
        tiny = np.finfo(1.0).tiny
        x = self.mt19937.beta(tiny / 32, tiny / 40, size=50)
        assert not np.any(np.isnan(x))

    def test_beta_expected_zero_frequency(self):
        # gh-24475: For small a and b (e.g. a=0.0025, b=0.0025), beta
        # would generate too many zeros.
        a = 0.0025
        b = 0.0025
        n = 1000000
        x = self.mt19937.beta(a, b, size=n)
        nzeros = np.count_nonzero(x == 0)
        # beta CDF at x = np.finfo(np.double).smallest_subnormal/2
        # is p = 0.0776169083131899, e.g,
        #
        #    import numpy as np
        #    from mpmath import mp
        #    mp.dps = 160
        #    x = mp.mpf(np.finfo(np.float64).smallest_subnormal)/2
        #    # CDF of the beta distribution at x:
        #    p = mp.betainc(a, b, x1=0, x2=x, regularized=True)
        #    n = 1000000
        #    exprected_freq = float(n*p)
        #
        expected_freq = 77616.90831318991
        assert 0.95 * expected_freq < nzeros < 1.05 * expected_freq

    def test_choice_sum_of_probs_tolerance(self):
        # The sum of probs should be 1.0 with some tolerance.
        # For low precision dtypes the tolerance was too tight.
        # See numpy github issue 6123.
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = self.mt19937.choice(a, p=probs)
            assert_(c in a)
            with pytest.raises(ValueError):
                self.mt19937.choice(a, p=probs * 0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # Test that permuting an array of different length strings
        # will not cause a segfault on garbage collection
        # Tests gh-7710

        a = np.array(['a', 'a' * 1000])

        for _ in range(100):
            self.mt19937.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        # Test that permuting an array of objects will not cause
        # a segfault on garbage collection.
        # See gh-7719
        a = np.array([np.arange(1), np.arange(4)], dtype=object)

        for _ in range(1000):
            self.mt19937.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_permutation_subclass(self):

        class N(np.ndarray):
            pass

        mt19937 = Generator(MT19937(1))
        orig = np.arange(3).view(N)
        perm = mt19937.permutation(orig)
        assert_array_equal(perm, np.array([2, 0, 1]))
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self, dtype=None, copy=None):
                return self.a

        mt19937 = Generator(MT19937(1))
        m = M()
        perm = mt19937.permutation(m)
        assert_array_equal(perm, np.array([4, 1, 3, 0, 2]))
        assert_array_equal(m.__array__(), np.arange(5))

    def test_gamma_0(self):
        assert self.mt19937.standard_gamma(0.0) == 0.0
        assert_array_equal(self.mt19937.standard_gamma([0.0]), 0.0)

        actual = self.mt19937.standard_gamma([0.0], dtype='float')
        expected = np.array([0.], dtype=np.float32)
        assert_array_equal(actual, expected)

    def test_geometric_tiny_prob(self):
        # Regression test for gh-17007.
        # When p = 1e-30, the probability that a sample will exceed 2**63-1
        # is 0.9999999999907766, so we expect the result to be all 2**63-1.
        assert_array_equal(self.mt19937.geometric(p=1e-30, size=3),
                           np.iinfo(np.int64).max)

    def test_zipf_large_parameter(self):
        # Regression test for part of gh-9829: a call such as rng.zipf(10000)
        # would hang.
        n = 8
        sample = self.mt19937.zipf(10000, size=n)
        assert_array_equal(sample, np.ones(n, dtype=np.int64))

    def test_zipf_a_near_1(self):
        # Regression test for gh-9829: a call such as rng.zipf(1.0000000000001)
        # would hang.
        n = 100000
        sample = self.mt19937.zipf(1.0000000000001, size=n)
        # Not much of a test, but let's do something more than verify that
        # it doesn't hang.  Certainly for a monotonically decreasing
        # discrete distribution truncated to signed 64 bit integers, more
        # than half should be less than 2**62.
        assert np.count_nonzero(sample < 2**62) > n / 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_scalar_ctors.py ===
"""
Test the scalar constructors, which also do type-coercion
"""
import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_warns,
)


class TestFromString:
    def test_floating(self):
        # Ticket #640, floats from string
        fsingle = np.single('1.234')
        fdouble = np.double('1.234')
        flongdouble = np.longdouble('1.234')
        assert_almost_equal(fsingle, 1.234)
        assert_almost_equal(fdouble, 1.234)
        assert_almost_equal(flongdouble, 1.234)

    def test_floating_overflow(self):
        """ Strings containing an unrepresentable float overflow """
        fhalf = np.half('1e10000')
        assert_equal(fhalf, np.inf)
        fsingle = np.single('1e10000')
        assert_equal(fsingle, np.inf)
        fdouble = np.double('1e10000')
        assert_equal(fdouble, np.inf)
        flongdouble = assert_warns(RuntimeWarning, np.longdouble, '1e10000')
        assert_equal(flongdouble, np.inf)

        fhalf = np.half('-1e10000')
        assert_equal(fhalf, -np.inf)
        fsingle = np.single('-1e10000')
        assert_equal(fsingle, -np.inf)
        fdouble = np.double('-1e10000')
        assert_equal(fdouble, -np.inf)
        flongdouble = assert_warns(RuntimeWarning, np.longdouble, '-1e10000')
        assert_equal(flongdouble, -np.inf)


class TestExtraArgs:
    def test_superclass(self):
        # try both positional and keyword arguments
        s = np.str_(b'\\x61', encoding='unicode-escape')
        assert s == 'a'
        s = np.str_(b'\\x61', 'unicode-escape')
        assert s == 'a'

        # previously this would return '\\xx'
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', encoding='unicode-escape')
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', 'unicode-escape')

        # superclass fails, but numpy succeeds
        assert np.bytes_(-2) == b'-2'

    def test_datetime(self):
        dt = np.datetime64('2000-01', ('M', 2))
        assert np.datetime_data(dt) == ('M', 2)

        with pytest.raises(TypeError):
            np.datetime64('2000', garbage=True)

    def test_bool(self):
        with pytest.raises(TypeError):
            np.bool(False, garbage=True)

    def test_void(self):
        with pytest.raises(TypeError):
            np.void(b'test', garbage=True)


class TestFromInt:
    def test_intp(self):
        # Ticket #99
        assert_equal(1024, np.intp(1024))

    def test_uint64_from_negative(self):
        with pytest.raises(OverflowError):
            np.uint64(-2)


int_types = [np.byte, np.short, np.intc, np.long, np.longlong]
uint_types = [np.ubyte, np.ushort, np.uintc, np.ulong, np.ulonglong]
float_types = [np.half, np.single, np.double, np.longdouble]
cfloat_types = [np.csingle, np.cdouble, np.clongdouble]


class TestArrayFromScalar:
    """ gh-15467 and gh-19125 """

    def _do_test(self, t1, t2, arg=2):
        if arg is None:
            x = t1()
        elif isinstance(arg, tuple):
            if t1 is np.clongdouble:
                pytest.xfail("creating a clongdouble from real and "
                             "imaginary parts isn't supported")
            x = t1(*arg)
        else:
            x = t1(arg)
        arr = np.array(x, dtype=t2)
        # type should be preserved exactly
        if t2 is None:
            assert arr.dtype.type is t1
        else:
            assert arr.dtype.type is t2

    @pytest.mark.parametrize('t1', int_types + uint_types)
    @pytest.mark.parametrize('t2', int_types + uint_types + [None])
    def test_integers(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', float_types)
    @pytest.mark.parametrize('t2', float_types + [None])
    def test_reals(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', cfloat_types)
    @pytest.mark.parametrize('t2', cfloat_types + [None])
    @pytest.mark.parametrize('arg', [2, 1 + 3j, (1, 2), None])
    def test_complex(self, t1, t2, arg):
        self._do_test(t1, t2, arg)

    @pytest.mark.parametrize('t', cfloat_types)
    def test_complex_errors(self, t):
        with pytest.raises(TypeError):
            t(1j, 1j)
        with pytest.raises(TypeError):
            t(1, None)
        with pytest.raises(TypeError):
            t(None, 1)


@pytest.mark.parametrize("length",
        [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_via_length(length):
    res = np.void(length)
    assert type(res) is np.void
    assert res.item() == b"\0" * 5
    assert res.dtype == "V5"

@pytest.mark.parametrize("bytes_",
        [b"spam", np.array(567.)])
def test_void_from_byteslike(bytes_):
    res = np.void(bytes_)
    expected = bytes(bytes_)
    assert type(res) is np.void
    assert res.item() == expected

    # Passing dtype can extend it (this is how filling works)
    res = np.void(bytes_, dtype="V100")
    assert type(res) is np.void
    assert res.item()[:len(expected)] == expected
    assert res.item()[len(expected):] == b"\0" * (res.nbytes - len(expected))
    # As well as shorten:
    res = np.void(bytes_, dtype="V4")
    assert type(res) is np.void
    assert res.item() == expected[:4]

def test_void_arraylike_trumps_byteslike():
    # The memoryview is converted as an array-like of shape (18,)
    # rather than a single bytes-like of that length.
    m = memoryview(b"just one mintleaf?")
    res = np.void(m)
    assert type(res) is np.ndarray
    assert res.dtype == "V1"
    assert res.shape == (18,)

def test_void_dtype_arg():
    # Basic test for the dtype argument (positional and keyword)
    res = np.void((1, 2), dtype="i,i")
    assert res.item() == (1, 2)
    res = np.void((2, 3), "i,i")
    assert res.item() == (2, 3)

@pytest.mark.parametrize("data",
        [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_from_integer_with_dtype(data):
    # The "length" meaning is ignored, rather data is used:
    res = np.void(data, dtype="i,i")
    assert type(res) is np.void
    assert res.dtype == "i,i"
    assert res["f0"] == 5 and res["f1"] == 5

def test_void_from_structure():
    dtype = np.dtype([('s', [('f', 'f8'), ('u', 'U1')]), ('i', 'i2')])
    data = np.array(((1., 'a'), 2), dtype=dtype)
    res = np.void(data[()], dtype=dtype)
    assert type(res) is np.void
    assert res.dtype == dtype
    assert res == data[()]

def test_void_bad_dtype():
    with pytest.raises(TypeError,
            match="void: descr must be a `void.*int64"):
        np.void(4, dtype="i8")

    # Subarray dtype (with shape `(4,)` is rejected):
    with pytest.raises(TypeError,
            match=r"void: descr must be a `void.*\(4,\)"):
        np.void(4, dtype="4i")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_conversion_utils.py ===
"""
Tests for numpy/_core/src/multiarray/conversion_utils.c
"""
import re

import numpy._core._multiarray_tests as mt
import pytest

from numpy._core.multiarray import CLIP, RAISE, WRAP
from numpy.testing import assert_raises


class StringConverterTestCase:
    allow_bytes = True
    case_insensitive = True
    exact_match = False
    warn = True

    def _check_value_error(self, val):
        pattern = fr'\(got {re.escape(repr(val))}\)'
        with pytest.raises(ValueError, match=pattern) as exc:
            self.conv(val)

    def _check_conv_assert_warn(self, val, expected):
        if self.warn:
            with assert_raises(ValueError) as exc:
                assert self.conv(val) == expected
        else:
            assert self.conv(val) == expected

    def _check(self, val, expected):
        """Takes valid non-deprecated inputs for converters,
        runs converters on inputs, checks correctness of outputs,
        warnings and errors"""
        assert self.conv(val) == expected

        if self.allow_bytes:
            assert self.conv(val.encode('ascii')) == expected
        else:
            with pytest.raises(TypeError):
                self.conv(val.encode('ascii'))

        if len(val) != 1:
            if self.exact_match:
                self._check_value_error(val[:1])
                self._check_value_error(val + '\0')
            else:
                self._check_conv_assert_warn(val[:1], expected)

        if self.case_insensitive:
            if val != val.lower():
                self._check_conv_assert_warn(val.lower(), expected)
            if val != val.upper():
                self._check_conv_assert_warn(val.upper(), expected)
        else:
            if val != val.lower():
                self._check_value_error(val.lower())
            if val != val.upper():
                self._check_value_error(val.upper())

    def test_wrong_type(self):
        # common cases which apply to all the below
        with pytest.raises(TypeError):
            self.conv({})
        with pytest.raises(TypeError):
            self.conv([])

    def test_wrong_value(self):
        # nonsense strings
        self._check_value_error('')
        self._check_value_error('\N{greek small letter pi}')

        if self.allow_bytes:
            self._check_value_error(b'')
            # bytes which can't be converted to strings via utf8
            self._check_value_error(b"\xFF")
        if self.exact_match:
            self._check_value_error("there's no way this is supported")


class TestByteorderConverter(StringConverterTestCase):
    """ Tests of PyArray_ByteorderConverter """
    conv = mt.run_byteorder_converter
    warn = False

    def test_valid(self):
        for s in ['big', '>']:
            self._check(s, 'NPY_BIG')
        for s in ['little', '<']:
            self._check(s, 'NPY_LITTLE')
        for s in ['native', '=']:
            self._check(s, 'NPY_NATIVE')
        for s in ['ignore', '|']:
            self._check(s, 'NPY_IGNORE')
        for s in ['swap']:
            self._check(s, 'NPY_SWAP')


class TestSortkindConverter(StringConverterTestCase):
    """ Tests of PyArray_SortkindConverter """
    conv = mt.run_sortkind_converter
    warn = False

    def test_valid(self):
        self._check('quicksort', 'NPY_QUICKSORT')
        self._check('heapsort', 'NPY_HEAPSORT')
        self._check('mergesort', 'NPY_STABLESORT')  # alias
        self._check('stable', 'NPY_STABLESORT')


class TestSelectkindConverter(StringConverterTestCase):
    """ Tests of PyArray_SelectkindConverter """
    conv = mt.run_selectkind_converter
    case_insensitive = False
    exact_match = True

    def test_valid(self):
        self._check('introselect', 'NPY_INTROSELECT')


class TestSearchsideConverter(StringConverterTestCase):
    """ Tests of PyArray_SearchsideConverter """
    conv = mt.run_searchside_converter

    def test_valid(self):
        self._check('left', 'NPY_SEARCHLEFT')
        self._check('right', 'NPY_SEARCHRIGHT')


class TestOrderConverter(StringConverterTestCase):
    """ Tests of PyArray_OrderConverter """
    conv = mt.run_order_converter
    warn = False

    def test_valid(self):
        self._check('c', 'NPY_CORDER')
        self._check('f', 'NPY_FORTRANORDER')
        self._check('a', 'NPY_ANYORDER')
        self._check('k', 'NPY_KEEPORDER')

    def test_flatten_invalid_order(self):
        # invalid after gh-14596
        with pytest.raises(ValueError):
            self.conv('Z')
        for order in [False, True, 0, 8]:
            with pytest.raises(TypeError):
                self.conv(order)


class TestClipmodeConverter(StringConverterTestCase):
    """ Tests of PyArray_ClipmodeConverter """
    conv = mt.run_clipmode_converter

    def test_valid(self):
        self._check('clip', 'NPY_CLIP')
        self._check('wrap', 'NPY_WRAP')
        self._check('raise', 'NPY_RAISE')

        # integer values allowed here
        assert self.conv(CLIP) == 'NPY_CLIP'
        assert self.conv(WRAP) == 'NPY_WRAP'
        assert self.conv(RAISE) == 'NPY_RAISE'


class TestCastingConverter(StringConverterTestCase):
    """ Tests of PyArray_CastingConverter """
    conv = mt.run_casting_converter
    case_insensitive = False
    exact_match = True

    def test_valid(self):
        self._check("no", "NPY_NO_CASTING")
        self._check("equiv", "NPY_EQUIV_CASTING")
        self._check("safe", "NPY_SAFE_CASTING")
        self._check("same_kind", "NPY_SAME_KIND_CASTING")
        self._check("unsafe", "NPY_UNSAFE_CASTING")


class TestIntpConverter:
    """ Tests of PyArray_IntpConverter """
    conv = mt.run_intp_converter

    def test_basic(self):
        assert self.conv(1) == (1,)
        assert self.conv((1, 2)) == (1, 2)
        assert self.conv([1, 2]) == (1, 2)
        assert self.conv(()) == ()

    def test_none(self):
        with pytest.raises(TypeError):
            assert self.conv(None) == ()

    def test_float(self):
        with pytest.raises(TypeError):
            self.conv(1.0)
        with pytest.raises(TypeError):
            self.conv([1, 1.0])

    def test_too_large(self):
        with pytest.raises(ValueError):
            self.conv(2**64)

    def test_too_many_dims(self):
        assert self.conv([1] * 64) == (1,) * 64
        with pytest.raises(ValueError):
            self.conv([1] * 65)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_cse_diff.py ===
"""Tests for the ``sympy.simplify._cse_diff.py`` module."""

import pytest

from sympy.core.symbol import (Symbol, symbols)
from sympy.core.numbers import Integer
from sympy.core.function import Function
from sympy.core import Derivative
from sympy.functions.elementary.exponential import exp
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.physics.mechanics import dynamicsymbols
from sympy.simplify._cse_diff import (_forward_jacobian,
                                      _remove_cse_from_derivative,
                                      _forward_jacobian_cse,
                                      _forward_jacobian_norm_in_cse_out)
from sympy.simplify.simplify import simplify
from sympy.matrices import Matrix, eye

from sympy.testing.pytest import raises
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.simplify.trigsimp import trigsimp

from sympy import cse


w = Symbol('w')
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

q1, q2, q3 = dynamicsymbols('q1 q2 q3')

# Define the custom functions
k = Function('k')(x, y)
f = Function('f')(k, z)

zero = Integer(0)
one = Integer(1)
two = Integer(2)
neg_one = Integer(-1)


@pytest.mark.parametrize(
    'expr, wrt',
    [
        ([zero], [x]),
        ([one], [x]),
        ([two], [x]),
        ([neg_one], [x]),
        ([x], [x]),
        ([y], [x]),
        ([x + y], [x]),
        ([x*y], [x]),
        ([x**2], [x]),
        ([x**y], [x]),
        ([exp(x)], [x]),
        ([sin(x)], [x]),
        ([tan(x)], [x]),
        ([zero, one, x, y, x*y, x + y], [x, y]),
        ([((x/y) + sin(x/y) - exp(y))*((x/y) - exp(y))], [x, y]),
        ([w*tan(y*z)/(x - tan(y*z)), w*x*tan(y*z)/(x - tan(y*z))], [w, x, y, z]),
        ([q1**2 + q2, q2**2 + q3, q3**2 + q1], [q1, q2, q3]),
        ([f + Derivative(f, x) + k + 2*x], [x])
    ]
)


def test_forward_jacobian(expr, wrt):
    expr = ImmutableDenseMatrix([expr]).T
    wrt = ImmutableDenseMatrix([wrt]).T
    jacobian = _forward_jacobian(expr, wrt)
    zeros = ImmutableDenseMatrix.zeros(*jacobian.shape)
    assert simplify(jacobian - expr.jacobian(wrt)) == zeros


def test_process_cse():
    x, y, z = symbols('x y z')
    f = Function('f')
    k = Function('k')
    expr = Matrix([f(k(x,y), z) + Derivative(f(k(x,y), z), x) + k(x,y) + 2*x])
    repl, reduced = cse(expr)
    p_repl, p_reduced = _remove_cse_from_derivative(repl, reduced)

    x0 = symbols('x0')
    x1 = symbols('x1')

    expected_output = (
        [(x0, k(x, y)), (x1, f(x0, z))],
        [Matrix([2 * x + x0 + x1 + Derivative(f(k(x, y), z), x)])]
    )

    assert p_repl == expected_output[0], f"Expected {expected_output[0]}, but got {p_repl}"
    assert p_reduced == expected_output[1], f"Expected {expected_output[1]}, but got {p_reduced}"


def test_io_matrix_type():
    x, y, z = symbols('x y z')
    expr = ImmutableDenseMatrix([
        x * y + y * z + x * y * z,
        x ** 2 + y ** 2 + z ** 2,
        x * y + x * z + y * z
    ])
    wrt = ImmutableDenseMatrix([x, y, z])

    replacements, reduced_expr = cse(expr)

    # Test _forward_jacobian_core
    replacements_core, jacobian_core, precomputed_fs_core = _forward_jacobian_cse(replacements, reduced_expr, wrt)
    assert isinstance(jacobian_core[0], type(reduced_expr[0])), "Jacobian should be a Matrix of the same type as the input"

    # Test _forward_jacobian_norm_in_dag_out
    replacements_norm, jacobian_norm, precomputed_fs_norm = _forward_jacobian_norm_in_cse_out(
        expr, wrt)
    assert isinstance(jacobian_norm[0], type(reduced_expr[0])), "Jacobian should be a Matrix of the same type as the input"

    # Test _forward_jacobian
    jacobian = _forward_jacobian(expr, wrt)
    assert isinstance(jacobian, type(expr)), "Jacobian should be a Matrix of the same type as the input"


def test_forward_jacobian_input_output():
    x, y, z = symbols('x y z')
    expr = Matrix([
        x * y + y * z + x * y * z,
        x ** 2 + y ** 2 + z ** 2,
        x * y + x * z + y * z
    ])
    wrt = Matrix([x, y, z])

    replacements, reduced_expr = cse(expr)

    # Test _forward_jacobian_core
    replacements_core, jacobian_core, precomputed_fs_core = _forward_jacobian_cse(replacements, reduced_expr, wrt)
    assert isinstance(replacements_core, type(replacements)), "Replacements should be a list"
    assert isinstance(jacobian_core, type(reduced_expr)), "Jacobian should be a list"
    assert isinstance(precomputed_fs_core, list), "Precomputed free symbols should be a list"
    assert len(replacements_core) == len(replacements), "Length of replacements does not match"
    assert len(jacobian_core) == 1, "Jacobian should have one element"
    assert len(precomputed_fs_core) == len(replacements), "Length of precomputed free symbols does not match"

    # Test _forward_jacobian_norm_in_dag_out
    replacements_norm, jacobian_norm, precomputed_fs_norm = _forward_jacobian_norm_in_cse_out(expr, wrt)
    assert isinstance(replacements_norm, type(replacements)), "Replacements should be a list"
    assert isinstance(jacobian_norm, type(reduced_expr)), "Jacobian should be a list"
    assert isinstance(precomputed_fs_norm, list), "Precomputed free symbols should be a list"
    assert len(replacements_norm) == len(replacements), "Length of replacements does not match"
    assert len(jacobian_norm) == 1, "Jacobian should have one element"
    assert len(precomputed_fs_norm) == len(replacements), "Length of precomputed free symbols does not match"


def test_jacobian_hessian():
    L = Matrix(1, 2, [x**2*y, 2*y**2 + x*y])
    syms = [x, y]
    assert _forward_jacobian(L, syms) == Matrix([[2*x*y, x**2], [y, 4*y + x]])

    L = Matrix(1, 2, [x, x**2*y**3])
    assert _forward_jacobian(L, syms) == Matrix([[1, 0], [2*x*y**3, x**2*3*y**2]])


def test_jacobian_metrics():
    rho, phi = symbols("rho,phi")
    X = Matrix([rho * cos(phi), rho * sin(phi)])
    Y = Matrix([rho, phi])
    J = _forward_jacobian(X, Y)
    assert J == X.jacobian(Y.T)
    assert J == (X.T).jacobian(Y)
    assert J == (X.T).jacobian(Y.T)
    g = J.T * eye(J.shape[0]) * J
    g = g.applyfunc(trigsimp)
    assert g == Matrix([[1, 0], [0, rho ** 2]])


def test_jacobian2():
    rho, phi = symbols("rho,phi")
    X = Matrix([rho * cos(phi), rho * sin(phi), rho ** 2])
    Y = Matrix([rho, phi])
    J = Matrix([
        [cos(phi), -rho * sin(phi)],
        [sin(phi), rho * cos(phi)],
        [2 * rho, 0],
    ])
    assert _forward_jacobian(X, Y) == J


def test_issue_4564():
    X = Matrix([exp(x + y + z), exp(x + y + z), exp(x + y + z)])
    Y = Matrix([x, y, z])
    for i in range(1, 3):
        for j in range(1, 3):
            X_slice = X[:i, :]
            Y_slice = Y[:j, :]
            J = _forward_jacobian(X_slice, Y_slice)
            assert J.rows == i
            assert J.cols == j
            for k in range(j):
                assert J[:, k] == X_slice


def test_nonvectorJacobian():
    X = Matrix([[exp(x + y + z), exp(x + y + z)],
                [exp(x + y + z), exp(x + y + z)]])
    raises(TypeError, lambda: _forward_jacobian(X, Matrix([x, y, z])))
    X = X[0, :]
    Y = Matrix([[x, y], [x, z]])
    raises(TypeError, lambda: _forward_jacobian(X, Y))
    raises(TypeError, lambda: _forward_jacobian(X, Matrix([[x, y], [x, z]])))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\test_typing.py ===
import importlib.util
import os
import re
import shutil
import textwrap
from collections import defaultdict
from typing import TYPE_CHECKING

import pytest

# Only trigger a full `mypy` run if this environment variable is set
# Note that these tests tend to take over a minute even on a macOS M1 CPU,
# and more than that in CI.
RUN_MYPY = "NPY_RUN_MYPY_IN_TESTSUITE" in os.environ
if RUN_MYPY and RUN_MYPY not in ('0', '', 'false'):
    RUN_MYPY = True

# Skips all functions in this file
pytestmark = pytest.mark.skipif(
    not RUN_MYPY,
    reason="`NPY_RUN_MYPY_IN_TESTSUITE` not set"
)


try:
    from mypy import api
except ImportError:
    NO_MYPY = True
else:
    NO_MYPY = False

if TYPE_CHECKING:
    from collections.abc import Iterator

    # We need this as annotation, but it's located in a private namespace.
    # As a compromise, do *not* import it during runtime
    from _pytest.mark.structures import ParameterSet

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PASS_DIR = os.path.join(DATA_DIR, "pass")
FAIL_DIR = os.path.join(DATA_DIR, "fail")
REVEAL_DIR = os.path.join(DATA_DIR, "reveal")
MISC_DIR = os.path.join(DATA_DIR, "misc")
MYPY_INI = os.path.join(DATA_DIR, "mypy.ini")
CACHE_DIR = os.path.join(DATA_DIR, ".mypy_cache")

#: A dictionary with file names as keys and lists of the mypy stdout as values.
#: To-be populated by `run_mypy`.
OUTPUT_MYPY: defaultdict[str, list[str]] = defaultdict(list)


def _key_func(key: str) -> str:
    """Split at the first occurrence of the ``:`` character.

    Windows drive-letters (*e.g.* ``C:``) are ignored herein.
    """
    drive, tail = os.path.splitdrive(key)
    return os.path.join(drive, tail.split(":", 1)[0])


def _strip_filename(msg: str) -> tuple[int, str]:
    """Strip the filename and line number from a mypy message."""
    _, tail = os.path.splitdrive(msg)
    _, lineno, msg = tail.split(":", 2)
    return int(lineno), msg.strip()


def strip_func(match: re.Match[str]) -> str:
    """`re.sub` helper function for stripping module names."""
    return match.groups()[1]


@pytest.fixture(scope="module", autouse=True)
def run_mypy() -> None:
    """Clears the cache and run mypy before running any of the typing tests.

    The mypy results are cached in `OUTPUT_MYPY` for further use.

    The cache refresh can be skipped using

    NUMPY_TYPING_TEST_CLEAR_CACHE=0 pytest numpy/typing/tests
    """
    if (
        os.path.isdir(CACHE_DIR)
        and bool(os.environ.get("NUMPY_TYPING_TEST_CLEAR_CACHE", True))  # noqa: PLW1508
    ):
        shutil.rmtree(CACHE_DIR)

    split_pattern = re.compile(r"(\s+)?\^(\~+)?")
    for directory in (PASS_DIR, REVEAL_DIR, FAIL_DIR, MISC_DIR):
        # Run mypy
        stdout, stderr, exit_code = api.run([
            "--config-file",
            MYPY_INI,
            "--cache-dir",
            CACHE_DIR,
            directory,
        ])
        if stderr:
            pytest.fail(f"Unexpected mypy standard error\n\n{stderr}", False)
        elif exit_code not in {0, 1}:
            pytest.fail(f"Unexpected mypy exit code: {exit_code}\n\n{stdout}", False)

        str_concat = ""
        filename: str | None = None
        for i in stdout.split("\n"):
            if "note:" in i:
                continue
            if filename is None:
                filename = _key_func(i)

            str_concat += f"{i}\n"
            if split_pattern.match(i) is not None:
                OUTPUT_MYPY[filename].append(str_concat)
                str_concat = ""
                filename = None


def get_test_cases(*directories: str) -> "Iterator[ParameterSet]":
    for directory in directories:
        for root, _, files in os.walk(directory):
            for fname in files:
                short_fname, ext = os.path.splitext(fname)
                if ext not in (".pyi", ".py"):
                    continue

                fullpath = os.path.join(root, fname)
                yield pytest.param(fullpath, id=short_fname)


_FAIL_INDENT = " " * 4
_FAIL_SEP = "\n" + "_" * 79 + "\n\n"

_FAIL_MSG_REVEAL = """{}:{} - reveal mismatch:

{}"""


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(PASS_DIR, FAIL_DIR))
def test_pass(path) -> None:
    # Alias `OUTPUT_MYPY` so that it appears in the local namespace
    output_mypy = OUTPUT_MYPY

    if path not in output_mypy:
        return

    relpath = os.path.relpath(path)

    # collect any reported errors, and clean up the output
    messages = []
    for message in output_mypy[path]:
        lineno, content = _strip_filename(message)
        content = content.removeprefix("error:").lstrip()
        messages.append(f"{relpath}:{lineno} - {content}")

    if messages:
        pytest.fail("\n".join(messages), pytrace=False)


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(REVEAL_DIR))
def test_reveal(path: str) -> None:
    """Validate that mypy correctly infers the return-types of
    the expressions in `path`.
    """
    __tracebackhide__ = True

    output_mypy = OUTPUT_MYPY
    if path not in output_mypy:
        return

    relpath = os.path.relpath(path)

    # collect any reported errors, and clean up the output
    failures = []
    for error_line in output_mypy[path]:
        lineno, error_msg = _strip_filename(error_line)
        error_msg = textwrap.indent(error_msg, _FAIL_INDENT)
        reason = _FAIL_MSG_REVEAL.format(relpath, lineno, error_msg)
        failures.append(reason)

    if failures:
        reasons = _FAIL_SEP.join(failures)
        pytest.fail(reasons, pytrace=False)


@pytest.mark.slow
@pytest.mark.skipif(NO_MYPY, reason="Mypy is not installed")
@pytest.mark.parametrize("path", get_test_cases(PASS_DIR))
def test_code_runs(path: str) -> None:
    """Validate that the code in `path` properly during runtime."""
    path_without_extension, _ = os.path.splitext(path)
    dirname, filename = path.split(os.sep)[-2:]

    spec = importlib.util.spec_from_file_location(
        f"{dirname}.{filename}", path
    )
    assert spec is not None
    assert spec.loader is not None

    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_getlimits.py ===
""" Test functions for limits module.

"""
import types
import warnings

import pytest

import numpy as np
from numpy import double, half, longdouble, single
from numpy._core import finfo, iinfo
from numpy._core.getlimits import _discovered_machar, _float_ma
from numpy.testing import assert_, assert_equal, assert_raises

##################################################

class TestPythonFloat:
    def test_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype), id(ftype2))

class TestHalf:
    def test_singleton(self):
        ftype = finfo(half)
        ftype2 = finfo(half)
        assert_equal(id(ftype), id(ftype2))

class TestSingle:
    def test_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype), id(ftype2))

class TestDouble:
    def test_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype), id(ftype2))

class TestLongdouble:
    def test_singleton(self):
        ftype = finfo(longdouble)
        ftype2 = finfo(longdouble)
        assert_equal(id(ftype), id(ftype2))

def assert_finfo_equal(f1, f2):
    # assert two finfo instances have the same attributes
    for attr in ('bits', 'eps', 'epsneg', 'iexp', 'machep',
                 'max', 'maxexp', 'min', 'minexp', 'negep', 'nexp',
                 'nmant', 'precision', 'resolution', 'tiny',
                 'smallest_normal', 'smallest_subnormal'):
        assert_equal(getattr(f1, attr), getattr(f2, attr),
                     f'finfo instances {f1} and {f2} differ on {attr}')

def assert_iinfo_equal(i1, i2):
    # assert two iinfo instances have the same attributes
    for attr in ('bits', 'min', 'max'):
        assert_equal(getattr(i1, attr), getattr(i2, attr),
                     f'iinfo instances {i1} and {i2} differ on {attr}')

class TestFinfo:
    def test_basic(self):
        dts = list(zip(['f2', 'f4', 'f8', 'c8', 'c16'],
                       [np.float16, np.float32, np.float64, np.complex64,
                        np.complex128]))
        for dt1, dt2 in dts:
            assert_finfo_equal(finfo(dt1), finfo(dt2))

        assert_raises(ValueError, finfo, 'i4')

    def test_regression_gh23108(self):
        # np.float32(1.0) and np.float64(1.0) have the same hash and are
        # equal under the == operator
        f1 = np.finfo(np.float32(1.0))
        f2 = np.finfo(np.float64(1.0))
        assert f1 != f2

    def test_regression_gh23867(self):
        class NonHashableWithDtype:
            __hash__ = None
            dtype = np.dtype('float32')

        x = NonHashableWithDtype()
        assert np.finfo(x) == np.finfo(x.dtype)


class TestIinfo:
    def test_basic(self):
        dts = list(zip(['i1', 'i2', 'i4', 'i8',
                   'u1', 'u2', 'u4', 'u8'],
                  [np.int8, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32, np.uint64]))
        for dt1, dt2 in dts:
            assert_iinfo_equal(iinfo(dt1), iinfo(dt2))

        assert_raises(ValueError, iinfo, 'f4')

    def test_unsigned_max(self):
        types = np._core.sctypes['uint']
        for T in types:
            with np.errstate(over="ignore"):
                max_calculated = T(0) - T(1)
            assert_equal(iinfo(T).max, max_calculated)

class TestRepr:
    def test_iinfo_repr(self):
        expected = "iinfo(min=-32768, max=32767, dtype=int16)"
        assert_equal(repr(np.iinfo(np.int16)), expected)

    def test_finfo_repr(self):
        expected = "finfo(resolution=1e-06, min=-3.4028235e+38,"\
                   " max=3.4028235e+38, dtype=float32)"
        assert_equal(repr(np.finfo(np.float32)), expected)


def test_instances():
    # Test the finfo and iinfo results on numeric instances agree with
    # the results on the corresponding types

    for c in [int, np.int16, np.int32, np.int64]:
        class_iinfo = iinfo(c)
        instance_iinfo = iinfo(c(12))

        assert_iinfo_equal(class_iinfo, instance_iinfo)

    for c in [float, np.float16, np.float32, np.float64]:
        class_finfo = finfo(c)
        instance_finfo = finfo(c(1.2))
        assert_finfo_equal(class_finfo, instance_finfo)

    with pytest.raises(ValueError):
        iinfo(10.)

    with pytest.raises(ValueError):
        iinfo('hi')

    with pytest.raises(ValueError):
        finfo(np.int64(1))


def assert_ma_equal(discovered, ma_like):
    # Check MachAr-like objects same as calculated MachAr instances
    for key, value in discovered.__dict__.items():
        assert_equal(value, getattr(ma_like, key))
        if hasattr(value, 'shape'):
            assert_equal(value.shape, getattr(ma_like, key).shape)
            assert_equal(value.dtype, getattr(ma_like, key).dtype)


def test_known_types():
    # Test we are correctly compiling parameters for known types
    for ftype, ma_like in ((np.float16, _float_ma[16]),
                           (np.float32, _float_ma[32]),
                           (np.float64, _float_ma[64])):
        assert_ma_equal(_discovered_machar(ftype), ma_like)
    # Suppress warning for broken discovery of double double on PPC
    with np.errstate(all='ignore'):
        ld_ma = _discovered_machar(np.longdouble)
    bytes = np.dtype(np.longdouble).itemsize
    if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
        # 80-bit extended precision
        assert_ma_equal(ld_ma, _float_ma[80])
    elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
        # IEE 754 128-bit
        assert_ma_equal(ld_ma, _float_ma[128])


def test_subnormal_warning():
    """Test that the subnormal is zero warning is not being raised."""
    with np.errstate(all='ignore'):
        ld_ma = _discovered_machar(np.longdouble)
    bytes = np.dtype(np.longdouble).itemsize
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
            # 80-bit extended precision
            ld_ma.smallest_subnormal
            assert len(w) == 0
        elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
            # IEE 754 128-bit
            ld_ma.smallest_subnormal
            assert len(w) == 0
        else:
            # Double double
            ld_ma.smallest_subnormal
            # This test may fail on some platforms
            assert len(w) == 0


def test_plausible_finfo():
    # Assert that finfo returns reasonable results for all types
    for ftype in np._core.sctypes['float'] + np._core.sctypes['complex']:
        info = np.finfo(ftype)
        assert_(info.nmant > 1)
        assert_(info.minexp < -1)
        assert_(info.maxexp > 1)


class TestRuntimeSubscriptable:
    def test_finfo_generic(self):
        assert isinstance(np.finfo[np.float64], types.GenericAlias)

    def test_iinfo_generic(self):
        assert isinstance(np.iinfo[np.int_], types.GenericAlias)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\_natype.py ===
# Vendored implementation of pandas.NA, adapted from pandas/_libs/missing.pyx
#
# This is vendored to avoid adding pandas as a test dependency.

__all__ = ["pd_NA"]

import numbers

import numpy as np


def _create_binary_propagating_op(name, is_divmod=False):
    is_cmp = name.strip("_") in ["eq", "ne", "le", "lt", "ge", "gt"]

    def method(self, other):
        if (
            other is pd_NA
            or isinstance(other, (str, bytes, numbers.Number, np.bool))
            or (isinstance(other, np.ndarray) and not other.shape)
        ):
            # Need the other.shape clause to handle NumPy scalars,
            # since we do a setitem on `out` below, which
            # won't work for NumPy scalars.
            if is_divmod:
                return pd_NA, pd_NA
            else:
                return pd_NA

        elif isinstance(other, np.ndarray):
            out = np.empty(other.shape, dtype=object)
            out[:] = pd_NA

            if is_divmod:
                return out, out.copy()
            else:
                return out

        elif is_cmp and isinstance(other, (np.datetime64, np.timedelta64)):
            return pd_NA

        elif isinstance(other, np.datetime64):
            if name in ["__sub__", "__rsub__"]:
                return pd_NA

        elif isinstance(other, np.timedelta64):
            if name in ["__sub__", "__rsub__", "__add__", "__radd__"]:
                return pd_NA

        return NotImplemented

    method.__name__ = name
    return method


def _create_unary_propagating_op(name: str):
    def method(self):
        return pd_NA

    method.__name__ = name
    return method


class NAType:
    def __repr__(self) -> str:
        return "<NA>"

    def __format__(self, format_spec) -> str:
        try:
            return self.__repr__().__format__(format_spec)
        except ValueError:
            return self.__repr__()

    def __bool__(self):
        raise TypeError("boolean value of NA is ambiguous")

    def __hash__(self):
        exponent = 31 if is_32bit else 61
        return 2**exponent - 1

    def __reduce__(self):
        return "pd_NA"

    # Binary arithmetic and comparison ops -> propagate

    __add__ = _create_binary_propagating_op("__add__")
    __radd__ = _create_binary_propagating_op("__radd__")
    __sub__ = _create_binary_propagating_op("__sub__")
    __rsub__ = _create_binary_propagating_op("__rsub__")
    __mul__ = _create_binary_propagating_op("__mul__")
    __rmul__ = _create_binary_propagating_op("__rmul__")
    __matmul__ = _create_binary_propagating_op("__matmul__")
    __rmatmul__ = _create_binary_propagating_op("__rmatmul__")
    __truediv__ = _create_binary_propagating_op("__truediv__")
    __rtruediv__ = _create_binary_propagating_op("__rtruediv__")
    __floordiv__ = _create_binary_propagating_op("__floordiv__")
    __rfloordiv__ = _create_binary_propagating_op("__rfloordiv__")
    __mod__ = _create_binary_propagating_op("__mod__")
    __rmod__ = _create_binary_propagating_op("__rmod__")
    __divmod__ = _create_binary_propagating_op("__divmod__", is_divmod=True)
    __rdivmod__ = _create_binary_propagating_op("__rdivmod__", is_divmod=True)
    # __lshift__ and __rshift__ are not implemented

    __eq__ = _create_binary_propagating_op("__eq__")
    __ne__ = _create_binary_propagating_op("__ne__")
    __le__ = _create_binary_propagating_op("__le__")
    __lt__ = _create_binary_propagating_op("__lt__")
    __gt__ = _create_binary_propagating_op("__gt__")
    __ge__ = _create_binary_propagating_op("__ge__")

    # Unary ops

    __neg__ = _create_unary_propagating_op("__neg__")
    __pos__ = _create_unary_propagating_op("__pos__")
    __abs__ = _create_unary_propagating_op("__abs__")
    __invert__ = _create_unary_propagating_op("__invert__")

    # pow has special
    def __pow__(self, other):
        if other is pd_NA:
            return pd_NA
        elif isinstance(other, (numbers.Number, np.bool)):
            if other == 0:
                # returning positive is correct for +/- 0.
                return type(other)(1)
            else:
                return pd_NA
        elif util.is_array(other):
            return np.where(other == 0, other.dtype.type(1), pd_NA)

        return NotImplemented

    def __rpow__(self, other):
        if other is pd_NA:
            return pd_NA
        elif isinstance(other, (numbers.Number, np.bool)):
            if other == 1:
                return other
            else:
                return pd_NA
        elif util.is_array(other):
            return np.where(other == 1, other, pd_NA)
        return NotImplemented

    # Logical ops using Kleene logic

    def __and__(self, other):
        if other is False:
            return False
        elif other is True or other is pd_NA:
            return pd_NA
        return NotImplemented

    __rand__ = __and__

    def __or__(self, other):
        if other is True:
            return True
        elif other is False or other is pd_NA:
            return pd_NA
        return NotImplemented

    __ror__ = __or__

    def __xor__(self, other):
        if other is False or other is True or other is pd_NA:
            return pd_NA
        return NotImplemented

    __rxor__ = __xor__

    __array_priority__ = 1000
    _HANDLED_TYPES = (np.ndarray, numbers.Number, str, np.bool)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        types = self._HANDLED_TYPES + (NAType,)
        for x in inputs:
            if not isinstance(x, types):
                return NotImplemented

        if method != "__call__":
            raise ValueError(f"ufunc method '{method}' not supported for NA")
        result = maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is NotImplemented:
            # For a NumPy ufunc that's not a binop, like np.logaddexp
            index = next(i for i, x in enumerate(inputs) if x is pd_NA)
            result = np.broadcast_arrays(*inputs)[index]
            if result.ndim == 0:
                result = result.item()
            if ufunc.nout > 1:
                result = (pd_NA,) * ufunc.nout

        return result


pd_NA = NAType()


def get_stringdtype_dtype(na_object, coerce=True):
    # explicit is check for pd_NA because != with pd_NA returns pd_NA
    if na_object is pd_NA or na_object != "unset":
        return np.dtypes.StringDType(na_object=na_object, coerce=coerce)
    else:
        return np.dtypes.StringDType(coerce=coerce)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_value_counts.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_data_frame_value_counts_unsorted():
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts(sort=False)
    expected = pd.Series(
        data=[1, 2, 1],
        index=pd.MultiIndex.from_arrays(
            [(2, 4, 6), (2, 0, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_ascending():
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts(ascending=True)
    expected = pd.Series(
        data=[1, 1, 2],
        index=pd.MultiIndex.from_arrays(
            [(2, 6, 4), (2, 0, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_default():
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts()
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.MultiIndex.from_arrays(
            [(4, 2, 6), (0, 2, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_normalize():
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    result = df.value_counts(normalize=True)
    expected = pd.Series(
        data=[0.5, 0.25, 0.25],
        index=pd.MultiIndex.from_arrays(
            [(4, 2, 6), (0, 2, 0)], names=["num_legs", "num_wings"]
        ),
        name="proportion",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_single_col_default():
    df = pd.DataFrame({"num_legs": [2, 4, 4, 6]})

    result = df.value_counts()
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.MultiIndex.from_arrays([[4, 2, 6]], names=["num_legs"]),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_empty():
    df_no_cols = pd.DataFrame()

    result = df_no_cols.value_counts()
    expected = pd.Series(
        [], dtype=np.int64, name="count", index=np.array([], dtype=np.intp)
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_empty_normalize():
    df_no_cols = pd.DataFrame()

    result = df_no_cols.value_counts(normalize=True)
    expected = pd.Series(
        [], dtype=np.float64, name="proportion", index=np.array([], dtype=np.intp)
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_dropna_true(nulls_fixture):
    # GH 41334
    df = pd.DataFrame(
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )
    result = df.value_counts()
    expected = pd.Series(
        data=[1, 1],
        index=pd.MultiIndex.from_arrays(
            [("Beth", "John"), ("Louise", "Smith")], names=["first_name", "middle_name"]
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_data_frame_value_counts_dropna_false(nulls_fixture):
    # GH 41334
    df = pd.DataFrame(
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )

    result = df.value_counts(dropna=False)
    expected = pd.Series(
        data=[1, 1, 1, 1],
        index=pd.MultiIndex(
            levels=[
                pd.Index(["Anne", "Beth", "John"]),
                pd.Index(["Louise", "Smith", np.nan]),
            ],
            codes=[[0, 1, 2, 2], [2, 0, 1, 2]],
            names=["first_name", "middle_name"],
        ),
        name="count",
    )

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", (["first_name", "middle_name"], [0, 1]))
def test_data_frame_value_counts_subset(nulls_fixture, columns):
    # GH 50829
    df = pd.DataFrame(
        {
            columns[0]: ["John", "Anne", "John", "Beth"],
            columns[1]: ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )
    result = df.value_counts(columns[0])
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.Index(["John", "Anne", "Beth"], name=columns[0]),
        name="count",
    )

    tm.assert_series_equal(result, expected)


def test_value_counts_categorical_future_warning():
    # GH#54775
    df = pd.DataFrame({"a": [1, 2, 3]}, dtype="category")
    result = df.value_counts()
    expected = pd.Series(
        1,
        index=pd.MultiIndex.from_arrays(
            [pd.Index([1, 2, 3], name="a", dtype="category")]
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)


def test_value_counts_with_missing_category():
    # GH-54836
    df = pd.DataFrame({"a": pd.Categorical([1, 2, 4], categories=[1, 2, 3, 4])})
    result = df.value_counts()
    expected = pd.Series(
        [1, 1, 1, 0],
        index=pd.MultiIndex.from_arrays(
            [pd.CategoricalIndex([1, 2, 4, 3], categories=[1, 2, 3, 4], name="a")]
        ),
        name="count",
    )
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_asof.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import IncompatibleFrequency

from pandas import (
    DatetimeIndex,
    PeriodIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    offsets,
    period_range,
)
import pandas._testing as tm


class TestSeriesAsof:
    def test_asof_nanosecond_index_access(self):
        ts = Timestamp("20130101").as_unit("ns")._value
        dti = DatetimeIndex([ts + 50 + i for i in range(100)])
        ser = Series(np.random.default_rng(2).standard_normal(100), index=dti)

        first_value = ser.asof(ser.index[0])

        # GH#46903 previously incorrectly was "day"
        assert dti.resolution == "nanosecond"

        # this used to not work bc parsing was done by dateutil that didn't
        #  handle nanoseconds
        assert first_value == ser["2013-01-01 00:00:00.000000050"]

        expected_ts = np.datetime64("2013-01-01 00:00:00.000000050", "ns")
        assert first_value == ser[Timestamp(expected_ts)]

    def test_basic(self):
        # array or list or dates
        N = 50
        rng = date_range("1/1/1990", periods=N, freq="53s")
        ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)
        ts.iloc[15:30] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")

        result = ts.asof(dates)
        assert notna(result).all()
        lb = ts.index[14]
        ub = ts.index[30]

        result = ts.asof(list(dates))
        assert notna(result).all()
        lb = ts.index[14]
        ub = ts.index[30]

        mask = (result.index >= lb) & (result.index < ub)
        rs = result[mask]
        assert (rs == ts[lb]).all()

        val = result[result.index[result.index >= ub][0]]
        assert ts[ub] == val

    def test_scalar(self):
        N = 30
        rng = date_range("1/1/1990", periods=N, freq="53s")
        # Explicit cast to float avoid implicit cast when setting nan
        ts = Series(np.arange(N), index=rng, dtype="float")
        ts.iloc[5:10] = np.nan
        ts.iloc[15:20] = np.nan

        val1 = ts.asof(ts.index[7])
        val2 = ts.asof(ts.index[19])

        assert val1 == ts.iloc[4]
        assert val2 == ts.iloc[14]

        # accepts strings
        val1 = ts.asof(str(ts.index[7]))
        assert val1 == ts.iloc[4]

        # in there
        result = ts.asof(ts.index[3])
        assert result == ts.iloc[3]

        # no as of value
        d = ts.index[0] - offsets.BDay()
        assert np.isnan(ts.asof(d))

    def test_with_nan(self):
        # basic asof test
        rng = date_range("1/1/2000", "1/2/2000", freq="4h")
        s = Series(np.arange(len(rng)), index=rng)
        r = s.resample("2h").mean()

        result = r.asof(r.index)
        expected = Series(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6.0],
            index=date_range("1/1/2000", "1/2/2000", freq="2h"),
        )
        tm.assert_series_equal(result, expected)

        r.iloc[3:5] = np.nan
        result = r.asof(r.index)
        expected = Series(
            [0, 0, 1, 1, 1, 1, 3, 3, 4, 4, 5, 5, 6.0],
            index=date_range("1/1/2000", "1/2/2000", freq="2h"),
        )
        tm.assert_series_equal(result, expected)

        r.iloc[-3:] = np.nan
        result = r.asof(r.index)
        expected = Series(
            [0, 0, 1, 1, 1, 1, 3, 3, 4, 4, 4, 4, 4.0],
            index=date_range("1/1/2000", "1/2/2000", freq="2h"),
        )
        tm.assert_series_equal(result, expected)

    def test_periodindex(self):
        # array or list or dates
        N = 50
        rng = period_range("1/1/1990", periods=N, freq="h")
        ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)
        ts.iloc[15:30] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="37min")

        result = ts.asof(dates)
        assert notna(result).all()
        lb = ts.index[14]
        ub = ts.index[30]

        result = ts.asof(list(dates))
        assert notna(result).all()
        lb = ts.index[14]
        ub = ts.index[30]

        pix = PeriodIndex(result.index.values, freq="h")
        mask = (pix >= lb) & (pix < ub)
        rs = result[mask]
        assert (rs == ts[lb]).all()

        ts.iloc[5:10] = np.nan
        ts.iloc[15:20] = np.nan

        val1 = ts.asof(ts.index[7])
        val2 = ts.asof(ts.index[19])

        assert val1 == ts.iloc[4]
        assert val2 == ts.iloc[14]

        # accepts strings
        val1 = ts.asof(str(ts.index[7]))
        assert val1 == ts.iloc[4]

        # in there
        assert ts.asof(ts.index[3]) == ts.iloc[3]

        # no as of value
        d = ts.index[0].to_timestamp() - offsets.BDay()
        assert isna(ts.asof(d))

        # Mismatched freq
        msg = "Input has different freq"
        with pytest.raises(IncompatibleFrequency, match=msg):
            ts.asof(rng.asfreq("D"))

    def test_errors(self):
        s = Series(
            [1, 2, 3],
            index=[Timestamp("20130101"), Timestamp("20130103"), Timestamp("20130102")],
        )

        # non-monotonic
        assert not s.index.is_monotonic_increasing
        with pytest.raises(ValueError, match="requires a sorted index"):
            s.asof(s.index[0])

        # subset with Series
        N = 10
        rng = date_range("1/1/1990", periods=N, freq="53s")
        s = Series(np.random.default_rng(2).standard_normal(N), index=rng)
        with pytest.raises(ValueError, match="not valid for Series"):
            s.asof(s.index[0], subset="foo")

    def test_all_nans(self):
        # GH 15713
        # series is all nans

        # testing non-default indexes
        N = 50
        rng = date_range("1/1/1990", periods=N, freq="53s")

        dates = date_range("1/1/1990", periods=N * 3, freq="25s")
        result = Series(np.nan, index=rng).asof(dates)
        expected = Series(np.nan, index=dates)
        tm.assert_series_equal(result, expected)

        # testing scalar input
        date = date_range("1/1/1990", periods=N * 3, freq="25s")[0]
        result = Series(np.nan, index=rng).asof(date)
        assert isna(result)

        # test name is propagated
        result = Series(np.nan, index=[1, 2, 3, 4], name="test").asof([4, 5])
        expected = Series(np.nan, index=[4, 5], name="test")
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\strings\test_api.py ===
import numpy as np
import pytest

from pandas import (
    CategoricalDtype,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    option_context,
)
from pandas.core.strings.accessor import StringMethods

# subset of the full set from pandas/conftest.py
_any_allowed_skipna_inferred_dtype = [
    ("string", ["a", np.nan, "c"]),
    ("bytes", [b"a", np.nan, b"c"]),
    ("empty", [np.nan, np.nan, np.nan]),
    ("empty", []),
    ("mixed-integer", ["a", np.nan, 2]),
]
ids, _ = zip(*_any_allowed_skipna_inferred_dtype)  # use inferred type as id


@pytest.fixture(params=_any_allowed_skipna_inferred_dtype, ids=ids)
def any_allowed_skipna_inferred_dtype(request):
    """
    Fixture for all (inferred) dtypes allowed in StringMethods.__init__

    The covered (inferred) types are:
    * 'string'
    * 'empty'
    * 'bytes'
    * 'mixed'
    * 'mixed-integer'

    Returns
    -------
    inferred_dtype : str
        The string for the inferred dtype from _libs.lib.infer_dtype
    values : np.ndarray
        An array of object dtype that will be inferred to have
        `inferred_dtype`

    Examples
    --------
    >>> from pandas._libs import lib
    >>>
    >>> def test_something(any_allowed_skipna_inferred_dtype):
    ...     inferred_dtype, values = any_allowed_skipna_inferred_dtype
    ...     # will pass
    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype
    ...
    ...     # constructor for .str-accessor will also pass
    ...     Series(values).str
    """
    inferred_dtype, values = request.param
    values = np.array(values, dtype=object)  # object dtype to avoid casting

    # correctness of inference tested in tests/dtypes/test_inference.py
    return inferred_dtype, values


def test_api(any_string_dtype):
    # GH 6106, GH 9322
    assert Series.str is StringMethods
    assert isinstance(Series([""], dtype=any_string_dtype).str, StringMethods)


def test_api_mi_raises():
    # GH 23679
    mi = MultiIndex.from_arrays([["a", "b", "c"]])
    msg = "Can only use .str accessor with Index, not MultiIndex"
    with pytest.raises(AttributeError, match=msg):
        mi.str
    assert not hasattr(mi, "str")


@pytest.mark.parametrize("dtype", [object, "category"])
def test_api_per_dtype(index_or_series, dtype, any_skipna_inferred_dtype):
    # one instance of parametrized fixture
    box = index_or_series
    inferred_dtype, values = any_skipna_inferred_dtype

    t = box(values, dtype=dtype)  # explicit dtype to avoid casting

    types_passing_constructor = [
        "string",
        "unicode",
        "empty",
        "bytes",
        "mixed",
        "mixed-integer",
    ]
    if inferred_dtype in types_passing_constructor:
        # GH 6106
        assert isinstance(t.str, StringMethods)
    else:
        # GH 9184, GH 23011, GH 23163
        msg = "Can only use .str accessor with string values.*"
        with pytest.raises(AttributeError, match=msg):
            t.str
        assert not hasattr(t, "str")


@pytest.mark.parametrize("dtype", [object, "category"])
def test_api_per_method(
    index_or_series,
    dtype,
    any_allowed_skipna_inferred_dtype,
    any_string_method,
    request,
    using_infer_string,
):
    # this test does not check correctness of the different methods,
    # just that the methods work on the specified (inferred) dtypes,
    # and raise on all others
    box = index_or_series

    # one instance of each parametrized fixture
    inferred_dtype, values = any_allowed_skipna_inferred_dtype
    method_name, args, kwargs = any_string_method

    reason = None
    if box is Index and values.size == 0:
        if method_name in ["partition", "rpartition"] and kwargs.get("expand", True):
            raises = TypeError
            reason = "Method cannot deal with empty Index"
        elif method_name == "split" and kwargs.get("expand", None):
            raises = TypeError
            reason = "Split fails on empty Series when expand=True"
        elif method_name == "get_dummies":
            raises = ValueError
            reason = "Need to fortify get_dummies corner cases"

    elif (
        box is Index
        and inferred_dtype == "empty"
        and dtype == object
        and method_name == "get_dummies"
    ):
        raises = ValueError
        reason = "Need to fortify get_dummies corner cases"

    if reason is not None:
        mark = pytest.mark.xfail(raises=raises, reason=reason)
        request.applymarker(mark)

    t = box(values, dtype=dtype)  # explicit dtype to avoid casting
    method = getattr(t.str, method_name)

    if using_infer_string and dtype == "category":
        string_allowed = method_name not in ["decode"]
    else:
        string_allowed = True
    bytes_allowed = method_name in ["decode", "get", "len", "slice"]
    # as of v0.23.4, all methods except 'cat' are very lenient with the
    # allowed data types, just returning NaN for entries that error.
    # This could be changed with an 'errors'-kwarg to the `str`-accessor,
    # see discussion in GH 13877
    mixed_allowed = method_name not in ["cat"]

    allowed_types = (
        ["empty"]
        + ["string", "unicode"] * string_allowed
        + ["bytes"] * bytes_allowed
        + ["mixed", "mixed-integer"] * mixed_allowed
    )

    if inferred_dtype in allowed_types:
        # xref GH 23555, GH 23556
        with option_context("future.no_silent_downcasting", True):
            method(*args, **kwargs)  # works!
    else:
        # GH 23011, GH 23163
        msg = (
            f"Cannot use .str.{method_name} with values of "
            f"inferred dtype {repr(inferred_dtype)}."
            "|a bytes-like object is required, not 'str'"
        )
        with pytest.raises(TypeError, match=msg):
            method(*args, **kwargs)


def test_api_for_categorical(any_string_method, any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/10661
    s = Series(list("aabb"), dtype=any_string_dtype)
    s = s + " " + s
    c = s.astype("category")
    c = c.astype(CategoricalDtype(c.dtype.categories.astype("object")))
    assert isinstance(c.str, StringMethods)

    method_name, args, kwargs = any_string_method

    result = getattr(c.str, method_name)(*args, **kwargs)
    expected = getattr(s.astype("object").str, method_name)(*args, **kwargs)

    if isinstance(result, DataFrame):
        tm.assert_frame_equal(result, expected)
    elif isinstance(result, Series):
        tm.assert_series_equal(result, expected)
    else:
        # str.cat(others=None) returns string, for example
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\tests\test_convert_indexed_to_array.py ===
from sympy import tanh
from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.tensor.array.expressions import ArrayElementwiseApplyFunc
from sympy.tensor.indexed import IndexedBase
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ArrayContraction, ArrayTensorProduct, \
    ArrayDiagonal, ArrayAdd, PermuteDims, ArrayElement, _array_tensor_product, _array_contraction, _array_diagonal, \
    _array_add, _permute_dims, ArraySymbol, OneArray
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array, _convert_indexed_to_array
from sympy.testing.pytest import raises


A, B = symbols("A B", cls=IndexedBase)
i, j, k, l, m, n = symbols("i j k l m n")
d0, d1, d2, d3 = symbols("d0:4")

I = Identity(k)

M = MatrixSymbol("M", k, k)
N = MatrixSymbol("N", k, k)
P = MatrixSymbol("P", k, k)
Q = MatrixSymbol("Q", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)
c = MatrixSymbol("c", k, 1)
d = MatrixSymbol("d", k, 1)


def test_arrayexpr_convert_index_to_array_support_function():
    expr = M[i, j]
    assert _convert_indexed_to_array(expr) == (M, (i, j))
    expr = M[i, j]*N[k, l]
    assert _convert_indexed_to_array(expr) == (ArrayTensorProduct(M, N), (i, j, k, l))
    expr = M[i, j]*N[j, k]
    assert _convert_indexed_to_array(expr) == (ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2)), (i, k, j))
    expr = Sum(M[i, j]*N[j, k], (j, 0, k-1))
    assert _convert_indexed_to_array(expr) == (ArrayContraction(ArrayTensorProduct(M, N), (1, 2)), (i, k))
    expr = M[i, j] + N[i, j]
    assert _convert_indexed_to_array(expr) == (ArrayAdd(M, N), (i, j))
    expr = M[i, j] + N[j, i]
    assert _convert_indexed_to_array(expr) == (ArrayAdd(M, PermuteDims(N, Permutation([1, 0]))), (i, j))
    expr = M[i, j] + M[j, i]
    assert _convert_indexed_to_array(expr) == (ArrayAdd(M, PermuteDims(M, Permutation([1, 0]))), (i, j))
    expr = (M*N*P)[i, j]
    assert _convert_indexed_to_array(expr) == (_array_contraction(ArrayTensorProduct(M, N, P), (1, 2), (3, 4)), (i, j))
    expr = expr.function  # Disregard summation in previous expression
    ret1, ret2 = _convert_indexed_to_array(expr)
    assert ret1 == ArrayDiagonal(ArrayTensorProduct(M, N, P), (1, 2), (3, 4))
    assert str(ret2) == "(i, j, _i_1, _i_2)"
    expr = KroneckerDelta(i, j)*M[i, k]
    assert _convert_indexed_to_array(expr) == (M, ({i, j}, k))
    expr = KroneckerDelta(i, j)*KroneckerDelta(j, k)*M[i, l]
    assert _convert_indexed_to_array(expr) == (M, ({i, j, k}, l))
    expr = KroneckerDelta(j, k)*(M[i, j]*N[k, l] + N[i, j]*M[k, l])
    assert _convert_indexed_to_array(expr) == (_array_diagonal(_array_add(
            ArrayTensorProduct(M, N),
            _permute_dims(ArrayTensorProduct(M, N), Permutation(0, 2)(1, 3))
        ), (1, 2)), (i, l, frozenset({j, k})))
    expr = KroneckerDelta(j, m)*KroneckerDelta(m, k)*(M[i, j]*N[k, l] + N[i, j]*M[k, l])
    assert _convert_indexed_to_array(expr) == (_array_diagonal(_array_add(
            ArrayTensorProduct(M, N),
            _permute_dims(ArrayTensorProduct(M, N), Permutation(0, 2)(1, 3))
        ), (1, 2)), (i, l, frozenset({j, m, k})))
    expr = KroneckerDelta(i, j)*KroneckerDelta(j, k)*KroneckerDelta(k,m)*M[i, 0]*KroneckerDelta(m, n)
    assert _convert_indexed_to_array(expr) == (M, ({i, j, k, m, n}, 0))
    expr = M[i, i]
    assert _convert_indexed_to_array(expr) == (ArrayDiagonal(M, (0, 1)), (i,))


def test_arrayexpr_convert_indexed_to_array_expression():

    s = Sum(A[i]*B[i], (i, 0, 3))
    cg = convert_indexed_to_array(s)
    assert cg == ArrayContraction(ArrayTensorProduct(A, B), (0, 1))

    expr = M*N
    result = ArrayContraction(ArrayTensorProduct(M, N), (1, 2))
    elem = expr[i, j]
    assert convert_indexed_to_array(elem) == result

    expr = M*N*M
    elem = expr[i, j]
    result = _array_contraction(_array_tensor_product(M, M, N), (1, 4), (2, 5))
    cg = convert_indexed_to_array(elem)
    assert cg == result

    cg = convert_indexed_to_array((M * N * P)[i, j])
    assert cg == _array_contraction(ArrayTensorProduct(M, N, P), (1, 2), (3, 4))

    cg = convert_indexed_to_array((M * N.T * P)[i, j])
    assert cg == _array_contraction(ArrayTensorProduct(M, N, P), (1, 3), (2, 4))

    expr = -2*M*N
    elem = expr[i, j]
    cg = convert_indexed_to_array(elem)
    assert cg == ArrayContraction(ArrayTensorProduct(-2, M, N), (1, 2))


def test_arrayexpr_convert_array_element_to_array_expression():
    A = ArraySymbol("A", (k,))
    B = ArraySymbol("B", (k,))

    s = Sum(A[i]*B[i], (i, 0, k-1))
    cg = convert_indexed_to_array(s)
    assert cg == ArrayContraction(ArrayTensorProduct(A, B), (0, 1))

    s = A[i]*B[i]
    cg = convert_indexed_to_array(s)
    assert cg == ArrayDiagonal(ArrayTensorProduct(A, B), (0, 1))

    s = A[i]*B[j]
    cg = convert_indexed_to_array(s, [i, j])
    assert cg == ArrayTensorProduct(A, B)
    cg = convert_indexed_to_array(s, [j, i])
    assert cg == ArrayTensorProduct(B, A)

    s = tanh(A[i]*B[j])
    cg = convert_indexed_to_array(s, [i, j])
    assert cg.dummy_eq(ArrayElementwiseApplyFunc(tanh, ArrayTensorProduct(A, B)))


def test_arrayexpr_convert_indexed_to_array_and_back_to_matrix():

    expr = a.T*b
    elem = expr[0, 0]
    cg = convert_indexed_to_array(elem)
    assert cg == ArrayElement(ArrayContraction(ArrayTensorProduct(a, b), (0, 2)), [0, 0])

    expr = M[i,j] + N[i,j]
    p1, p2 = _convert_indexed_to_array(expr)
    assert convert_array_to_matrix(p1) == M + N

    expr = M[i,j] + N[j,i]
    p1, p2 = _convert_indexed_to_array(expr)
    assert convert_array_to_matrix(p1) == M + N.T

    expr = M[i,j]*N[k,l] + N[i,j]*M[k,l]
    p1, p2 = _convert_indexed_to_array(expr)
    assert convert_array_to_matrix(p1) == ArrayAdd(
        ArrayTensorProduct(M, N),
        ArrayTensorProduct(N, M))

    expr = (M*N*P)[i, j]
    p1, p2 = _convert_indexed_to_array(expr)
    assert convert_array_to_matrix(p1) == M * N * P

    expr = Sum(M[i,j]*(N*P)[j,m], (j, 0, k-1))
    p1, p2 = _convert_indexed_to_array(expr)
    assert convert_array_to_matrix(p1) == M * N * P

    expr = Sum((P[j, m] + P[m, j])*(M[i,j]*N[m,n] + N[i,j]*M[m,n]), (j, 0, k-1), (m, 0, k-1))
    p1, p2 = _convert_indexed_to_array(expr)
    assert convert_array_to_matrix(p1) == M * P * N + M * P.T * N + N * P * M + N * P.T * M


def test_arrayexpr_convert_indexed_to_array_out_of_bounds():

    expr = Sum(M[i, i], (i, 0, 4))
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    expr = Sum(M[i, i], (i, 0, k))
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    expr = Sum(M[i, i], (i, 1, k-1))
    raises(ValueError, lambda: convert_indexed_to_array(expr))

    expr = Sum(M[i, j]*N[j,m], (j, 0, 4))
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    expr = Sum(M[i, j]*N[j,m], (j, 0, k))
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    expr = Sum(M[i, j]*N[j,m], (j, 1, k-1))
    raises(ValueError, lambda: convert_indexed_to_array(expr))


def test_arrayexpr_convert_indexed_to_array_broadcast():
    A = ArraySymbol("A", (3, 3))
    B = ArraySymbol("B", (3, 3))

    expr = A[i, j] + B[k, l]
    O2 = OneArray(3, 3)
    expected = ArrayAdd(ArrayTensorProduct(A, O2), ArrayTensorProduct(O2, B))
    assert convert_indexed_to_array(expr) == expected
    assert convert_indexed_to_array(expr, [i, j, k, l]) == expected
    assert convert_indexed_to_array(expr, [l, k, i, j]) == ArrayAdd(PermuteDims(ArrayTensorProduct(O2, A), [1, 0, 2, 3]), PermuteDims(ArrayTensorProduct(B, O2), [1, 0, 2, 3]))

    expr = A[i, j] + B[j, k]
    O1 = OneArray(3)
    assert convert_indexed_to_array(expr, [i, j, k]) == ArrayAdd(ArrayTensorProduct(A, O1), ArrayTensorProduct(O1, B))

    C = ArraySymbol("C", (d0, d1))
    D = ArraySymbol("D", (d3, d1))

    expr = C[i, j] + D[k, j]
    assert convert_indexed_to_array(expr, [i, j, k]) == ArrayAdd(ArrayTensorProduct(C, OneArray(d3)), PermuteDims(ArrayTensorProduct(OneArray(d0), D), [0, 2, 1]))

    X = ArraySymbol("X", (5, 3))

    expr = X[i, n] - X[j, n]
    assert convert_indexed_to_array(expr, [i, j, n]) == ArrayAdd(ArrayTensorProduct(-1, OneArray(5), X), PermuteDims(ArrayTensorProduct(X, OneArray(5)), [0, 2, 1]))

    raises(ValueError, lambda: convert_indexed_to_array(C[i, j] + D[i, j]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\test_construction.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
    Float32Dtype,
    Float64Dtype,
)


def test_uses_pandas_na():
    a = pd.array([1, None], dtype=Float64Dtype())
    assert a[1] is pd.NA


def test_floating_array_constructor():
    values = np.array([1, 2, 3, 4], dtype="float64")
    mask = np.array([False, False, False, True], dtype="bool")

    result = FloatingArray(values, mask)
    expected = pd.array([1, 2, 3, np.nan], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)
    tm.assert_numpy_array_equal(result._data, values)
    tm.assert_numpy_array_equal(result._mask, mask)

    msg = r".* should be .* numpy array. Use the 'pd.array' function instead"
    with pytest.raises(TypeError, match=msg):
        FloatingArray(values.tolist(), mask)

    with pytest.raises(TypeError, match=msg):
        FloatingArray(values, mask.tolist())

    with pytest.raises(TypeError, match=msg):
        FloatingArray(values.astype(int), mask)

    msg = r"__init__\(\) missing 1 required positional argument: 'mask'"
    with pytest.raises(TypeError, match=msg):
        FloatingArray(values)


def test_floating_array_disallows_float16():
    # GH#44715
    arr = np.array([1, 2], dtype=np.float16)
    mask = np.array([False, False])

    msg = "FloatingArray does not support np.float16 dtype"
    with pytest.raises(TypeError, match=msg):
        FloatingArray(arr, mask)


def test_floating_array_disallows_Float16_dtype(request):
    # GH#44715
    with pytest.raises(TypeError, match="data type 'Float16' not understood"):
        pd.array([1.0, 2.0], dtype="Float16")


def test_floating_array_constructor_copy():
    values = np.array([1, 2, 3, 4], dtype="float64")
    mask = np.array([False, False, False, True], dtype="bool")

    result = FloatingArray(values, mask)
    assert result._data is values
    assert result._mask is mask

    result = FloatingArray(values, mask, copy=True)
    assert result._data is not values
    assert result._mask is not mask


def test_to_array():
    result = pd.array([0.1, 0.2, 0.3, 0.4])
    expected = pd.array([0.1, 0.2, 0.3, 0.4], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b",
    [
        ([1, None], [1, pd.NA]),
        ([None], [pd.NA]),
        ([None, np.nan], [pd.NA, pd.NA]),
        ([1, np.nan], [1, pd.NA]),
        ([np.nan], [pd.NA]),
    ],
)
def test_to_array_none_is_nan(a, b):
    result = pd.array(a, dtype="Float64")
    expected = pd.array(b, dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


def test_to_array_mixed_integer_float():
    result = pd.array([1, 2.0])
    expected = pd.array([1.0, 2.0], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    result = pd.array([1, None, 2.0])
    expected = pd.array([1.0, None, 2.0], dtype="Float64")
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
        # GH#44514 all-NA case used to get quietly swapped out before checking ndim
        np.array([pd.NA] * 6, dtype=object).reshape(3, 2),
    ],
)
def test_to_array_error(values):
    # error in converting existing arrays to FloatingArray
    msg = "|".join(
        [
            "cannot be converted to FloatingDtype",
            "values must be a 1D list-like",
            "Cannot pass scalar",
            r"float\(\) argument must be a string or a (real )?number, not 'dict'",
            "could not convert string to float: 'foo'",
            r"could not convert string to float: np\.str_\('foo'\)",
        ]
    )
    with pytest.raises((TypeError, ValueError), match=msg):
        pd.array(values, dtype="Float64")


@pytest.mark.parametrize("values", [["1", "2", None], ["1.5", "2", None]])
def test_construct_from_float_strings(values):
    # see also test_to_integer_array_str
    expected = pd.array([float(values[0]), 2, None], dtype="Float64")

    res = pd.array(values, dtype="Float64")
    tm.assert_extension_array_equal(res, expected)

    res = FloatingArray._from_sequence(values)
    tm.assert_extension_array_equal(res, expected)


def test_to_array_inferred_dtype():
    # if values has dtype -> respect it
    result = pd.array(np.array([1, 2], dtype="float32"))
    assert result.dtype == Float32Dtype()

    # if values have no dtype -> always float64
    result = pd.array([1.0, 2.0])
    assert result.dtype == Float64Dtype()


def test_to_array_dtype_keyword():
    result = pd.array([1, 2], dtype="Float32")
    assert result.dtype == Float32Dtype()

    # if values has dtype -> override it
    result = pd.array(np.array([1, 2], dtype="float32"), dtype="Float64")
    assert result.dtype == Float64Dtype()


def test_to_array_integer():
    result = pd.array([1, 2], dtype="Float64")
    expected = pd.array([1.0, 2.0], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    # for integer dtypes, the itemsize is not preserved
    # TODO can we specify "floating" in general?
    result = pd.array(np.array([1, 2], dtype="int32"), dtype="Float64")
    assert result.dtype == Float64Dtype()


@pytest.mark.parametrize(
    "bool_values, values, target_dtype, expected_dtype",
    [
        ([False, True], [0, 1], Float64Dtype(), Float64Dtype()),
        ([False, True], [0, 1], "Float64", Float64Dtype()),
        ([False, True, np.nan], [0, 1, np.nan], Float64Dtype(), Float64Dtype()),
    ],
)
def test_to_array_bool(bool_values, values, target_dtype, expected_dtype):
    result = pd.array(bool_values, dtype=target_dtype)
    assert result.dtype == expected_dtype
    expected = pd.array(values, dtype=target_dtype)
    tm.assert_extension_array_equal(result, expected)


def test_series_from_float(data):
    # construct from our dtype & string dtype
    dtype = data.dtype

    # from float
    expected = pd.Series(data)
    result = pd.Series(data.to_numpy(na_value=np.nan, dtype="float"), dtype=str(dtype))
    tm.assert_series_equal(result, expected)

    # from list
    expected = pd.Series(data)
    result = pd.Series(np.array(data).tolist(), dtype=str(dtype))
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_update.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameUpdate:
    def test_update_nan(self):
        # #15593 #15617
        # test 1
        df1 = DataFrame({"A": [1.0, 2, 3], "B": date_range("2000", periods=3)})
        df2 = DataFrame({"A": [None, 2, 3]})
        expected = df1.copy()
        df1.update(df2, overwrite=False)

        tm.assert_frame_equal(df1, expected)

        # test 2
        df1 = DataFrame({"A": [1.0, None, 3], "B": date_range("2000", periods=3)})
        df2 = DataFrame({"A": [None, 2, 3]})
        expected = DataFrame({"A": [1.0, 2, 3], "B": date_range("2000", periods=3)})
        df1.update(df2, overwrite=False)

        tm.assert_frame_equal(df1, expected)

    def test_update(self):
        df = DataFrame(
            [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

        df.update(other)

        expected = DataFrame(
            [[1.5, np.nan, 3], [3.6, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
        )
        tm.assert_frame_equal(df, expected)

    def test_update_dtypes(self):
        # gh 3016
        df = DataFrame(
            [[1.0, 2.0, 1, False, True], [4.0, 5.0, 2, True, False]],
            columns=["A", "B", "int", "bool1", "bool2"],
        )

        other = DataFrame(
            [[45, 45, 3, True]], index=[0], columns=["A", "B", "int", "bool1"]
        )
        df.update(other)

        expected = DataFrame(
            [[45.0, 45.0, 3, True, True], [4.0, 5.0, 2, True, False]],
            columns=["A", "B", "int", "bool1", "bool2"],
        )
        tm.assert_frame_equal(df, expected)

    def test_update_nooverwrite(self):
        df = DataFrame(
            [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

        df.update(other, overwrite=False)

        expected = DataFrame(
            [[1.5, np.nan, 3], [1.5, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 3.0]]
        )
        tm.assert_frame_equal(df, expected)

    def test_update_filtered(self):
        df = DataFrame(
            [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

        df.update(other, filter_func=lambda x: x > 2)

        expected = DataFrame(
            [[1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "bad_kwarg, exception, msg",
        [
            # errors must be 'ignore' or 'raise'
            ({"errors": "something"}, ValueError, "The parameter errors must.*"),
            ({"join": "inner"}, NotImplementedError, "Only left join is supported"),
        ],
    )
    def test_update_raise_bad_parameter(self, bad_kwarg, exception, msg):
        df = DataFrame([[1.5, 1, 3.0]])
        with pytest.raises(exception, match=msg):
            df.update(df, **bad_kwarg)

    def test_update_raise_on_overlap(self):
        df = DataFrame(
            [[1.5, 1, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[2.0, np.nan], [np.nan, 7]], index=[1, 3], columns=[1, 2])
        with pytest.raises(ValueError, match="Data overlaps"):
            df.update(other, errors="raise")

    def test_update_from_non_df(self):
        d = {"a": Series([1, 2, 3, 4]), "b": Series([5, 6, 7, 8])}
        df = DataFrame(d)

        d["a"] = Series([5, 6, 7, 8])
        df.update(d)

        expected = DataFrame(d)

        tm.assert_frame_equal(df, expected)

        d = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
        df = DataFrame(d)

        d["a"] = [5, 6, 7, 8]
        df.update(d)

        expected = DataFrame(d)

        tm.assert_frame_equal(df, expected)

    def test_update_datetime_tz(self):
        # GH 25807
        result = DataFrame([pd.Timestamp("2019", tz="UTC")])
        with tm.assert_produces_warning(None):
            result.update(result)
        expected = DataFrame([pd.Timestamp("2019", tz="UTC")])
        tm.assert_frame_equal(result, expected)

    def test_update_datetime_tz_in_place(self, using_copy_on_write, warn_copy_on_write):
        # https://github.com/pandas-dev/pandas/issues/56227
        result = DataFrame([pd.Timestamp("2019", tz="UTC")])
        orig = result.copy()
        view = result[:]
        with tm.assert_produces_warning(
            FutureWarning if warn_copy_on_write else None, match="Setting a value"
        ):
            result.update(result + pd.Timedelta(days=1))
        expected = DataFrame([pd.Timestamp("2019-01-02", tz="UTC")])
        tm.assert_frame_equal(result, expected)
        if not using_copy_on_write:
            tm.assert_frame_equal(view, expected)
        else:
            tm.assert_frame_equal(view, orig)

    def test_update_with_different_dtype(self, using_copy_on_write):
        # GH#3217
        df = DataFrame({"a": [1, 3], "b": [np.nan, 2]})
        df["c"] = np.nan
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.update({"c": Series(["foo"], index=[0])})

        expected = DataFrame(
            {
                "a": [1, 3],
                "b": [np.nan, 2],
                "c": Series(["foo", np.nan]),
            }
        )
        tm.assert_frame_equal(df, expected)

    @td.skip_array_manager_invalid_test
    def test_update_modify_view(
        self, using_copy_on_write, warn_copy_on_write, using_infer_string
    ):
        # GH#47188
        df = DataFrame({"A": ["1", np.nan], "B": ["100", np.nan]})
        df2 = DataFrame({"A": ["a", "x"], "B": ["100", "200"]})
        df2_orig = df2.copy()
        result_view = df2[:]
        # TODO(CoW-warn) better warning message
        with tm.assert_cow_warning(warn_copy_on_write):
            df2.update(df)
        expected = DataFrame({"A": ["1", "x"], "B": ["100", "200"]})
        tm.assert_frame_equal(df2, expected)
        if using_copy_on_write or using_infer_string:
            tm.assert_frame_equal(result_view, df2_orig)
        else:
            tm.assert_frame_equal(result_view, expected)

    def test_update_dt_column_with_NaT_create_column(self):
        # GH#16713
        df = DataFrame({"A": [1, None], "B": [pd.NaT, pd.to_datetime("2016-01-01")]})
        df2 = DataFrame({"A": [2, 3]})
        df.update(df2, overwrite=False)
        expected = DataFrame(
            {"A": [1.0, 3.0], "B": [pd.NaT, pd.to_datetime("2016-01-01")]}
        )
        tm.assert_frame_equal(df, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\tests\test_gosper.py ===
"""Tests for Gosper's algorithm for hypergeometric summation. """

from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.polytools import Poly
from sympy.simplify.simplify import simplify
from sympy.concrete.gosper import gosper_normal, gosper_sum, gosper_term
from sympy.abc import a, b, j, k, m, n, r, x


def test_gosper_normal():
    eq = 4*n + 5, 2*(4*n + 1)*(2*n + 3), n
    assert gosper_normal(*eq) == \
        (Poly(Rational(1, 4), n), Poly(n + Rational(3, 2)), Poly(n + Rational(1, 4)))
    assert gosper_normal(*eq, polys=False) == \
        (Rational(1, 4), n + Rational(3, 2), n + Rational(1, 4))


def test_gosper_term():
    assert gosper_term((4*k + 1)*factorial(
        k)/factorial(2*k + 1), k) == (-k - S.Half)/(k + Rational(1, 4))


def test_gosper_sum():
    assert gosper_sum(1, (k, 0, n)) == 1 + n
    assert gosper_sum(k, (k, 0, n)) == n*(1 + n)/2
    assert gosper_sum(k**2, (k, 0, n)) == n*(1 + n)*(1 + 2*n)/6
    assert gosper_sum(k**3, (k, 0, n)) == n**2*(1 + n)**2/4

    assert gosper_sum(2**k, (k, 0, n)) == 2*2**n - 1

    assert gosper_sum(factorial(k), (k, 0, n)) is None
    assert gosper_sum(binomial(n, k), (k, 0, n)) is None

    assert gosper_sum(factorial(k)/k**2, (k, 0, n)) is None
    assert gosper_sum((k - 3)*factorial(k), (k, 0, n)) is None

    assert gosper_sum(k*factorial(k), k) == factorial(k)
    assert gosper_sum(
        k*factorial(k), (k, 0, n)) == n*factorial(n) + factorial(n) - 1

    assert gosper_sum((-1)**k*binomial(n, k), (k, 0, n)) == 0
    assert gosper_sum((
        -1)**k*binomial(n, k), (k, 0, m)) == -(-1)**m*(m - n)*binomial(n, m)/n

    assert gosper_sum((4*k + 1)*factorial(k)/factorial(2*k + 1), (k, 0, n)) == \
        (2*factorial(2*n + 1) - factorial(n))/factorial(2*n + 1)

    # issue 6033:
    assert gosper_sum(
        n*(n + a + b)*a**n*b**n/(factorial(n + a)*factorial(n + b)), \
        (n, 0, m)).simplify() == -exp(m*log(a) + m*log(b))*gamma(a + 1) \
        *gamma(b + 1)/(gamma(a)*gamma(b)*gamma(a + m + 1)*gamma(b + m + 1)) \
        + 1/(gamma(a)*gamma(b))


def test_gosper_sum_indefinite():
    assert gosper_sum(k, k) == k*(k - 1)/2
    assert gosper_sum(k**2, k) == k*(k - 1)*(2*k - 1)/6

    assert gosper_sum(1/(k*(k + 1)), k) == -1/k
    assert gosper_sum(-(27*k**4 + 158*k**3 + 430*k**2 + 678*k + 445)*gamma(2*k
                      + 4)/(3*(3*k + 7)*gamma(3*k + 6)), k) == \
        (3*k + 5)*(k**2 + 2*k + 5)*gamma(2*k + 4)/gamma(3*k + 6)


def test_gosper_sum_parametric():
    assert gosper_sum(binomial(S.Half, m - j + 1)*binomial(S.Half, m + j), (j, 1, n)) == \
        n*(1 + m - n)*(-1 + 2*m + 2*n)*binomial(S.Half, 1 + m - n)* \
        binomial(S.Half, m + n)/(m*(1 + 2*m))


def test_gosper_sum_algebraic():
    assert gosper_sum(
        n**2 + sqrt(2), (n, 0, m)) == (m + 1)*(2*m**2 + m + 6*sqrt(2))/6


def test_gosper_sum_iterated():
    f1 = binomial(2*k, k)/4**k
    f2 = (1 + 2*n)*binomial(2*n, n)/4**n
    f3 = (1 + 2*n)*(3 + 2*n)*binomial(2*n, n)/(3*4**n)
    f4 = (1 + 2*n)*(3 + 2*n)*(5 + 2*n)*binomial(2*n, n)/(15*4**n)
    f5 = (1 + 2*n)*(3 + 2*n)*(5 + 2*n)*(7 + 2*n)*binomial(2*n, n)/(105*4**n)

    assert gosper_sum(f1, (k, 0, n)) == f2
    assert gosper_sum(f2, (n, 0, n)) == f3
    assert gosper_sum(f3, (n, 0, n)) == f4
    assert gosper_sum(f4, (n, 0, n)) == f5

# the AeqB tests test expressions given in
# www.math.upenn.edu/~wilf/AeqB.pdf


def test_gosper_sum_AeqB_part1():
    f1a = n**4
    f1b = n**3*2**n
    f1c = 1/(n**2 + sqrt(5)*n - 1)
    f1d = n**4*4**n/binomial(2*n, n)
    f1e = factorial(3*n)/(factorial(n)*factorial(n + 1)*factorial(n + 2)*27**n)
    f1f = binomial(2*n, n)**2/((n + 1)*4**(2*n))
    f1g = (4*n - 1)*binomial(2*n, n)**2/((2*n - 1)**2*4**(2*n))
    f1h = n*factorial(n - S.Half)**2/factorial(n + 1)**2

    g1a = m*(m + 1)*(2*m + 1)*(3*m**2 + 3*m - 1)/30
    g1b = 26 + 2**(m + 1)*(m**3 - 3*m**2 + 9*m - 13)
    g1c = (m + 1)*(m*(m**2 - 7*m + 3)*sqrt(5) - (
        3*m**3 - 7*m**2 + 19*m - 6))/(2*m**3*sqrt(5) + m**4 + 5*m**2 - 1)/6
    g1d = Rational(-2, 231) + 2*4**m*(m + 1)*(63*m**4 + 112*m**3 + 18*m**2 -
             22*m + 3)/(693*binomial(2*m, m))
    g1e = Rational(-9, 2) + (81*m**2 + 261*m + 200)*factorial(
        3*m + 2)/(40*27**m*factorial(m)*factorial(m + 1)*factorial(m + 2))
    g1f = (2*m + 1)**2*binomial(2*m, m)**2/(4**(2*m)*(m + 1))
    g1g = -binomial(2*m, m)**2/4**(2*m)
    g1h = 4*pi -(2*m + 1)**2*(3*m + 4)*factorial(m - S.Half)**2/factorial(m + 1)**2

    g = gosper_sum(f1a, (n, 0, m))
    assert g is not None and simplify(g - g1a) == 0
    g = gosper_sum(f1b, (n, 0, m))
    assert g is not None and simplify(g - g1b) == 0
    g = gosper_sum(f1c, (n, 0, m))
    assert g is not None and simplify(g - g1c) == 0
    g = gosper_sum(f1d, (n, 0, m))
    assert g is not None and simplify(g - g1d) == 0
    g = gosper_sum(f1e, (n, 0, m))
    assert g is not None and simplify(g - g1e) == 0
    g = gosper_sum(f1f, (n, 0, m))
    assert g is not None and simplify(g - g1f) == 0
    g = gosper_sum(f1g, (n, 0, m))
    assert g is not None and simplify(g - g1g) == 0
    g = gosper_sum(f1h, (n, 0, m))
    # need to call rewrite(gamma) here because we have terms involving
    # factorial(1/2)
    assert g is not None and simplify(g - g1h).rewrite(gamma) == 0


def test_gosper_sum_AeqB_part2():
    f2a = n**2*a**n
    f2b = (n - r/2)*binomial(r, n)
    f2c = factorial(n - 1)**2/(factorial(n - x)*factorial(n + x))

    g2a = -a*(a + 1)/(a - 1)**3 + a**(
        m + 1)*(a**2*m**2 - 2*a*m**2 + m**2 - 2*a*m + 2*m + a + 1)/(a - 1)**3
    g2b = (m - r)*binomial(r, m)/2
    ff = factorial(1 - x)*factorial(1 + x)
    g2c = 1/ff*(
        1 - 1/x**2) + factorial(m)**2/(x**2*factorial(m - x)*factorial(m + x))

    g = gosper_sum(f2a, (n, 0, m))
    assert g is not None and simplify(g - g2a) == 0
    g = gosper_sum(f2b, (n, 0, m))
    assert g is not None and simplify(g - g2b) == 0
    g = gosper_sum(f2c, (n, 1, m))
    assert g is not None and simplify(g - g2c) == 0


def test_gosper_nan():
    a = Symbol('a', positive=True)
    b = Symbol('b', positive=True)
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    f2d = n*(n + a + b)*a**n*b**n/(factorial(n + a)*factorial(n + b))
    g2d = 1/(factorial(a - 1)*factorial(
        b - 1)) - a**(m + 1)*b**(m + 1)/(factorial(a + m)*factorial(b + m))
    g = gosper_sum(f2d, (n, 0, m))
    assert simplify(g - g2d) == 0


def test_gosper_sum_AeqB_part3():
    f3a = 1/n**4
    f3b = (6*n + 3)/(4*n**4 + 8*n**3 + 8*n**2 + 4*n + 3)
    f3c = 2**n*(n**2 - 2*n - 1)/(n**2*(n + 1)**2)
    f3d = n**2*4**n/((n + 1)*(n + 2))
    f3e = 2**n/(n + 1)
    f3f = 4*(n - 1)*(n**2 - 2*n - 1)/(n**2*(n + 1)**2*(n - 2)**2*(n - 3)**2)
    f3g = (n**4 - 14*n**2 - 24*n - 9)*2**n/(n**2*(n + 1)**2*(n + 2)**2*
           (n + 3)**2)

    # g3a -> no closed form
    g3b = m*(m + 2)/(2*m**2 + 4*m + 3)
    g3c = 2**m/m**2 - 2
    g3d = Rational(2, 3) + 4**(m + 1)*(m - 1)/(m + 2)/3
    # g3e -> no closed form
    g3f = -(Rational(-1, 16) + 1/((m - 2)**2*(m + 1)**2))  # the AeqB key is wrong
    g3g = Rational(-2, 9) + 2**(m + 1)/((m + 1)**2*(m + 3)**2)

    g = gosper_sum(f3a, (n, 1, m))
    assert g is None
    g = gosper_sum(f3b, (n, 1, m))
    assert g is not None and simplify(g - g3b) == 0
    g = gosper_sum(f3c, (n, 1, m - 1))
    assert g is not None and simplify(g - g3c) == 0
    g = gosper_sum(f3d, (n, 1, m))
    assert g is not None and simplify(g - g3d) == 0
    g = gosper_sum(f3e, (n, 0, m - 1))
    assert g is None
    g = gosper_sum(f3f, (n, 4, m))
    assert g is not None and simplify(g - g3f) == 0
    g = gosper_sum(f3g, (n, 1, m))
    assert g is not None and simplify(g - g3g) == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_puiseux.py ===
#
# Tests for PuiseuxRing and PuiseuxPoly
#

from sympy.testing.pytest import raises

from sympy import ZZ, QQ, ring
from sympy.polys.puiseux import PuiseuxRing, PuiseuxPoly, puiseux_ring

from sympy.abc import x, y


def test_puiseux_ring():
    R, px = puiseux_ring('x', QQ)
    R2, px2 = puiseux_ring([x], QQ)
    assert isinstance(R, PuiseuxRing)
    assert isinstance(px, PuiseuxPoly)
    assert R == R2
    assert px == px2
    assert R == PuiseuxRing('x', QQ)
    assert R == PuiseuxRing([x], QQ)
    assert R != PuiseuxRing('y', QQ)
    assert R != PuiseuxRing('x', ZZ)
    assert R != PuiseuxRing('x, y', QQ)
    assert R != QQ
    assert str(R) == 'PuiseuxRing((x,), QQ)'


def test_puiseux_ring_attributes():
    R1, px1, py1 = ring('x, y', QQ)
    R2, px2, py2 = puiseux_ring('x, y', QQ)
    assert R2.domain == QQ
    assert R2.symbols == (x, y)
    assert R2.gens == (px2, py2)
    assert R2.ngens == 2
    assert R2.poly_ring == R1
    assert R2.zero == PuiseuxPoly(R1.zero, R2)
    assert R2.one == PuiseuxPoly(R1.one, R2)
    assert R2.zero_monom == R1.zero_monom == (0, 0) # type: ignore
    assert R2.monomial_mul((1, 2), (3, 4)) == (4, 6)


def test_puiseux_ring_methods():
    R1, px1, py1 = ring('x, y', QQ)
    R2, px2, py2 = puiseux_ring('x, y', QQ)
    assert R2({(1, 2): 3}) == 3*px2*py2**2
    assert R2(px1) == px2
    assert R2(1) == R2.one
    assert R2(QQ(1,2)) == QQ(1,2)*R2.one
    assert R2.from_poly(px1) == px2
    assert R2.from_poly(px1) != py2
    assert R2.from_dict({(1, 2): QQ(3)}) == 3*px2*py2**2
    assert R2.from_dict({(QQ(1,2), 2): QQ(3)}) == 3*px2**QQ(1,2)*py2**2
    assert R2.from_int(3) == 3*R2.one
    assert R2.domain_new(3) == QQ(3)
    assert QQ.of_type(R2.domain_new(3))
    assert R2.ground_new(3) == 3*R2.one
    assert isinstance(R2.ground_new(3), PuiseuxPoly)
    assert R2.index(px2) == 0
    assert R2.index(py2) == 1


def test_puiseux_poly():
    R1, px1 = ring('x', QQ)
    R2, px2 = puiseux_ring('x', QQ)
    assert PuiseuxPoly(px1, R2) == px2
    assert px2.ring == R2
    assert px2.as_expr() == px1.as_expr() == x
    assert px1 != px2
    assert R2.one == px2**0 == 1
    assert px2 == px1
    assert px2 != 2.0
    assert px2**QQ(1,2) != px1


def test_puiseux_poly_normalization():
    R, x = puiseux_ring('x', QQ)
    assert (x**2 + 1) / x == x + 1/x == R({(1,): 1, (-1,): 1})
    assert (x**QQ(1,6))**2 == x**QQ(1,3) == R({(QQ(1,3),): 1})
    assert (x**QQ(1,6))**(-2) == x**(-QQ(1,3)) == R({(-QQ(1,3),): 1})
    assert (x**QQ(1,6))**QQ(1,2) == x**QQ(1,12) == R({(QQ(1,12),): 1})
    assert (x**QQ(1,6))**6 == x == R({(1,): 1})
    assert x**QQ(1,6) * x**QQ(1,3) == x**QQ(1,2) == R({(QQ(1,2),): 1})
    assert 1/x * x**2 == x == R({(1,): 1})
    assert 1/x**QQ(1,3) * x**QQ(1,3) == 1 == R({(0,): 1})


def test_puiseux_poly_monoms():
    R, x = puiseux_ring('x', QQ)
    assert x.monoms() == [(1,)]
    assert list(x) == [(1,)]
    assert (x**2 + 1).monoms() == [(2,), (0,)]
    assert R({(1,): 1, (-1,): 1}).monoms() == [(1,), (-1,)]
    assert R({(QQ(1,3),): 1}).monoms() == [(QQ(1,3),)]
    assert R({(-QQ(1,3),): 1}).monoms() == [(-QQ(1,3),)]
    p = x**QQ(1,6)
    assert p[(QQ(1,6),)] == 1
    raises(KeyError, lambda: p[(1,)])
    assert p.to_dict() == {(QQ(1,6),): 1}
    assert R(p.to_dict()) == p
    assert PuiseuxPoly.from_dict({(QQ(1,6),): 1}, R) == p


def test_puiseux_poly_repr():
    R, x = puiseux_ring('x', QQ)
    assert repr(x) == 'x'
    assert repr(x**QQ(1,2)) == 'x**(1/2)'
    assert repr(1/x) == 'x**(-1)'
    assert repr(2*x**2 + 1) == '1 + 2*x**2'
    assert repr(R.one) == '1'
    assert repr(2*R.one) == '2'


def test_puiseux_poly_unify():
    R, x = puiseux_ring('x', QQ)
    assert 1/x + x == x + 1/x == R({(1,): 1, (-1,): 1})
    assert repr(1/x + x) == 'x**(-1) + x'
    assert 1/x + 1/x == 2/x == R({(-1,): 2})
    assert repr(1/x + 1/x) == '2*x**(-1)'
    assert x**QQ(1,2) + x**QQ(1,2) == 2*x**QQ(1,2) == R({(QQ(1,2),): 2})
    assert repr(x**QQ(1,2) + x**QQ(1,2)) == '2*x**(1/2)'
    assert x**QQ(1,2) + x**QQ(1,3) == R({(QQ(1,2),): 1, (QQ(1,3),): 1})
    assert repr(x**QQ(1,2) + x**QQ(1,3)) == 'x**(1/3) + x**(1/2)'
    assert x + x**QQ(1,2) == R({(1,): 1, (QQ(1,2),): 1})
    assert repr(x + x**QQ(1,2)) == 'x**(1/2) + x'
    assert 1/x**QQ(1,2) + 1/x**QQ(1,3) == R({(-QQ(1,2),): 1, (-QQ(1,3),): 1})
    assert repr(1/x**QQ(1,2) + 1/x**QQ(1,3)) == 'x**(-1/2) + x**(-1/3)'
    assert 1/x + x**QQ(1,2) == x**QQ(1,2) + 1/x == R({(-1,): 1, (QQ(1,2),): 1})
    assert repr(1/x + x**QQ(1,2)) == 'x**(-1) + x**(1/2)'


def test_puiseux_poly_arit():
    R, x = puiseux_ring('x', QQ)
    R2, y = puiseux_ring('y', QQ)
    p = x**2 + 1
    assert +p == p
    assert -p == -1 - x**2
    assert p + p == 2*p == 2*x**2 + 2
    assert p + 1 == 1 + p == x**2 + 2
    assert p + QQ(1,2) == QQ(1,2) + p == x**2 + QQ(3,2)
    assert p - p == 0
    assert p - 1 == -1 + p == x**2
    assert p - QQ(1,2) == -QQ(1,2) + p == x**2 + QQ(1,2)
    assert 1 - p == -p + 1 == -x**2
    assert QQ(1,2) - p == -p + QQ(1,2) == -x**2 - QQ(1,2)
    assert p * p == x**4 + 2*x**2 + 1
    assert p * 1 == 1 * p == p
    assert 2 * p == p * 2 == 2*x**2 + 2
    assert p * QQ(1,2) == QQ(1,2) * p == QQ(1,2)*x**2 + QQ(1,2)
    assert x**QQ(1,2) * x**QQ(1,2) == x
    raises(ValueError, lambda: x + y)
    raises(ValueError, lambda: x - y)
    raises(ValueError, lambda: x * y)
    raises(TypeError, lambda: x + None)
    raises(TypeError, lambda: x - None)
    raises(TypeError, lambda: x * None)
    raises(TypeError, lambda: None + x)
    raises(TypeError, lambda: None - x)
    raises(TypeError, lambda: None * x)


def test_puiseux_poly_div():
    R, x = puiseux_ring('x', QQ)
    R2, y = puiseux_ring('y', QQ)
    p = x**2 - 1
    assert p / 1 == p
    assert p / QQ(1,2) == 2*p == 2*x**2 - 2
    assert p / x == x - 1/x == R({(1,): 1, (-1,): -1})
    assert 2 / x == 2*x**-1 == R({(-1,): 2})
    assert QQ(1,2) / x == QQ(1,2)*x**-1 == 1/(2*x) == 1/x/2 == R({(-1,): QQ(1,2)})
    raises(ZeroDivisionError, lambda: p / 0)
    raises(ValueError, lambda: (x + 1) / (x + 2))
    raises(ValueError, lambda: (x + 1) / (x + 1))
    raises(ValueError, lambda: x / y)
    raises(TypeError, lambda: x / None)
    raises(TypeError, lambda: None / x)


def test_puiseux_poly_pow():
    R, x = puiseux_ring('x', QQ)
    Rz, xz = puiseux_ring('x', ZZ)
    assert x**0 == 1 == R({(0,): 1})
    assert x**1 == x == R({(1,): 1})
    assert x**2 == x*x == R({(2,): 1})
    assert x**QQ(1,2) == R({(QQ(1,2),): 1})
    assert x**-1 == 1/x == R({(-1,): 1})
    assert x**-QQ(1,2) == 1/x**QQ(1,2) == R({(-QQ(1,2),): 1})
    assert (2*x)**-1 == 1/(2*x) == QQ(1,2)/x == QQ(1,2)*x**-1 == R({(-1,): QQ(1,2)})
    assert 2/x**2 == 2*x**-2 == R({(-2,): 2})
    assert 2/xz**2 == 2*xz**-2 == Rz({(-2,): 2})
    raises(TypeError, lambda: x**None)
    raises(ValueError, lambda: (x + 1)**-1)
    raises(ValueError, lambda: (x + 1)**QQ(1,2))
    raises(ValueError, lambda: (2*x)**QQ(1,2))
    raises(ValueError, lambda: (2*xz)**-1)


def test_puiseux_poly_diff():
    R, x, y = puiseux_ring('x, y', QQ)
    assert (x**2 + 1).diff(x) == 2*x
    assert (x**2 + 1).diff(y) == 0
    assert (x**2 + y**2).diff(x) == 2*x
    assert (x**QQ(1,2) + y**QQ(1,2)).diff(x) == QQ(1,2)*x**-QQ(1,2)
    assert ((x*y)**QQ(1,2)).diff(x) == QQ(1,2)*y**QQ(1,2)*x**-QQ(1,2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_sqrtdenest.py ===
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
    _subsets as subsets, _sqrt_numeric_denest)

r2, r3, r5, r6, r7, r10, r15, r29 = [sqrt(x) for x in (2, 3, 5, 6, 7, 10,
                                          15, 29)]


def test_sqrtdenest():
    d = {sqrt(5 + 2 * r6): r2 + r3,
        sqrt(5. + 2 * r6): sqrt(5. + 2 * r6),
        sqrt(5. + 4*sqrt(5 + 2 * r6)): sqrt(5.0 + 4*r2 + 4*r3),
        sqrt(r2): sqrt(r2),
        sqrt(5 + r7): sqrt(5 + r7),
        sqrt(3 + sqrt(5 + 2*r7)):
         3*r2*(5 + 2*r7)**Rational(1, 4)/(2*sqrt(6 + 3*r7)) +
         r2*sqrt(6 + 3*r7)/(2*(5 + 2*r7)**Rational(1, 4)),
        sqrt(3 + 2*r3): 3**Rational(3, 4)*(r6/2 + 3*r2/2)/3}
    for i in d:
        assert sqrtdenest(i) == d[i], i


def test_sqrtdenest2():
    assert sqrtdenest(sqrt(16 - 2*r29 + 2*sqrt(55 - 10*r29))) == \
        r5 + sqrt(11 - 2*r29)
    e = sqrt(-r5 + sqrt(-2*r29 + 2*sqrt(-10*r29 + 55) + 16))
    assert sqrtdenest(e) == root(-2*r29 + 11, 4)
    r = sqrt(1 + r7)
    assert sqrtdenest(sqrt(1 + r)) == sqrt(1 + r)
    e = sqrt(((1 + sqrt(1 + 2*sqrt(3 + r2 + r5)))**2).expand())
    assert sqrtdenest(e) == 1 + sqrt(1 + 2*sqrt(r2 + r5 + 3))

    assert sqrtdenest(sqrt(5*r3 + 6*r2)) == \
        sqrt(2)*root(3, 4) + root(3, 4)**3

    assert sqrtdenest(sqrt(((1 + r5 + sqrt(1 + r3))**2).expand())) == \
        1 + r5 + sqrt(1 + r3)

    assert sqrtdenest(sqrt(((1 + r5 + r7 + sqrt(1 + r3))**2).expand())) == \
        1 + sqrt(1 + r3) + r5 + r7

    e = sqrt(((1 + cos(2) + cos(3) + sqrt(1 + r3))**2).expand())
    assert sqrtdenest(e) == cos(3) + cos(2) + 1 + sqrt(1 + r3)

    e = sqrt(-2*r10 + 2*r2*sqrt(-2*r10 + 11) + 14)
    assert sqrtdenest(e) == sqrt(-2*r10 - 2*r2 + 4*r5 + 14)

    # check that the result is not more complicated than the input
    z = sqrt(-2*r29 + cos(2) + 2*sqrt(-10*r29 + 55) + 16)
    assert sqrtdenest(z) == z

    assert sqrtdenest(sqrt(r6 + sqrt(15))) == sqrt(r6 + sqrt(15))

    z = sqrt(15 - 2*sqrt(31) + 2*sqrt(55 - 10*r29))
    assert sqrtdenest(z) == z


def test_sqrtdenest_rec():
    assert sqrtdenest(sqrt(-4*sqrt(14) - 2*r6 + 4*sqrt(21) + 33)) == \
        -r2 + r3 + 2*r7
    assert sqrtdenest(sqrt(-28*r7 - 14*r5 + 4*sqrt(35) + 82)) == \
        -7 + r5 + 2*r7
    assert sqrtdenest(sqrt(6*r2/11 + 2*sqrt(22)/11 + 6*sqrt(11)/11 + 2)) == \
        sqrt(11)*(r2 + 3 + sqrt(11))/11
    assert sqrtdenest(sqrt(468*r3 + 3024*r2 + 2912*r6 + 19735)) == \
        9*r3 + 26 + 56*r6
    z = sqrt(-490*r3 - 98*sqrt(115) - 98*sqrt(345) - 2107)
    assert sqrtdenest(z) == sqrt(-1)*(7*r5 + 7*r15 + 7*sqrt(23))
    z = sqrt(-4*sqrt(14) - 2*r6 + 4*sqrt(21) + 34)
    assert sqrtdenest(z) == z
    assert sqrtdenest(sqrt(-8*r2 - 2*r5 + 18)) == -r10 + 1 + r2 + r5
    assert sqrtdenest(sqrt(8*r2 + 2*r5 - 18)) == \
        sqrt(-1)*(-r10 + 1 + r2 + r5)
    assert sqrtdenest(sqrt(8*r2/3 + 14*r5/3 + Rational(154, 9))) == \
        -r10/3 + r2 + r5 + 3
    assert sqrtdenest(sqrt(sqrt(2*r6 + 5) + sqrt(2*r7 + 8))) == \
        sqrt(1 + r2 + r3 + r7)
    assert sqrtdenest(sqrt(4*r15 + 8*r5 + 12*r3 + 24)) == 1 + r3 + r5 + r15

    w = 1 + r2 + r3 + r5 + r7
    assert sqrtdenest(sqrt((w**2).expand())) == w
    z = sqrt((w**2).expand() + 1)
    assert sqrtdenest(z) == z

    z = sqrt(2*r10 + 6*r2 + 4*r5 + 12 + 10*r15 + 30*r3)
    assert sqrtdenest(z) == z


def test_issue_6241():
    z = sqrt( -320 + 32*sqrt(5) + 64*r15)
    assert sqrtdenest(z) == z


def test_sqrtdenest3():
    z = sqrt(13 - 2*r10 + 2*r2*sqrt(-2*r10 + 11))
    assert sqrtdenest(z) == -1 + r2 + r10
    assert sqrtdenest(z, max_iter=1) == -1 + sqrt(2) + sqrt(10)
    z = sqrt(sqrt(r2 + 2) + 2)
    assert sqrtdenest(z) == z
    assert sqrtdenest(sqrt(-2*r10 + 4*r2*sqrt(-2*r10 + 11) + 20)) == \
        sqrt(-2*r10 - 4*r2 + 8*r5 + 20)
    assert sqrtdenest(sqrt((112 + 70*r2) + (46 + 34*r2)*r5)) == \
        r10 + 5 + 4*r2 + 3*r5
    z = sqrt(5 + sqrt(2*r6 + 5)*sqrt(-2*r29 + 2*sqrt(-10*r29 + 55) + 16))
    r = sqrt(-2*r29 + 11)
    assert sqrtdenest(z) == sqrt(r2*r + r3*r + r10 + r15 + 5)

    n = sqrt(2*r6/7 + 2*r7/7 + 2*sqrt(42)/7 + 2)
    d = sqrt(16 - 2*r29 + 2*sqrt(55 - 10*r29))
    assert sqrtdenest(n/d) == r7*(1 + r6 + r7)/(Mul(7, (sqrt(-2*r29 + 11) + r5),
                                                    evaluate=False))


def test_sqrtdenest4():
    # see Denest_en.pdf in https://github.com/sympy/sympy/issues/3192
    z = sqrt(8 - r2*sqrt(5 - r5) - sqrt(3)*(1 + r5))
    z1 = sqrtdenest(z)
    c = sqrt(-r5 + 5)
    z1 = ((-r15*c - r3*c + c + r5*c - r6 - r2 + r10 + sqrt(30))/4).expand()
    assert sqrtdenest(z) == z1

    z = sqrt(2*r2*sqrt(r2 + 2) + 5*r2 + 4*sqrt(r2 + 2) + 8)
    assert sqrtdenest(z) == r2 + sqrt(r2 + 2) + 2

    w = 2 + r2 + r3 + (1 + r3)*sqrt(2 + r2 + 5*r3)
    z = sqrt((w**2).expand())
    assert sqrtdenest(z) == w.expand()


def test_sqrt_symbolic_denest():
    x = Symbol('x')
    z = sqrt(((1 + sqrt(sqrt(2 + x) + 3))**2).expand())
    assert sqrtdenest(z) == sqrt((1 + sqrt(sqrt(2 + x) + 3))**2)
    z = sqrt(((1 + sqrt(sqrt(2 + cos(1)) + 3))**2).expand())
    assert sqrtdenest(z) == 1 + sqrt(sqrt(2 + cos(1)) + 3)
    z = ((1 + cos(2))**4 + 1).expand()
    assert sqrtdenest(z) == z
    z = sqrt(((1 + sqrt(sqrt(2 + cos(3*x)) + 3))**2 + 1).expand())
    assert sqrtdenest(z) == z
    c = cos(3)
    c2 = c**2
    assert sqrtdenest(sqrt(2*sqrt(1 + r3)*c + c2 + 1 + r3*c2)) == \
        -1 - sqrt(1 + r3)*c
    ra = sqrt(1 + r3)
    z = sqrt(20*ra*sqrt(3 + 3*r3) + 12*r3*ra*sqrt(3 + 3*r3) + 64*r3 + 112)
    assert sqrtdenest(z) == z


def test_issue_5857():
    from sympy.abc import x, y
    z = sqrt(1/(4*r3 + 7) + 1)
    ans = (r2 + r6)/(r3 + 2)
    assert sqrtdenest(z) == ans
    assert sqrtdenest(1 + z) == 1 + ans
    assert sqrtdenest(Integral(z + 1, (x, 1, 2))) == \
        Integral(1 + ans, (x, 1, 2))
    assert sqrtdenest(x + sqrt(y)) == x + sqrt(y)
    ans = (r2 + r6)/(r3 + 2)
    assert sqrtdenest(z) == ans
    assert sqrtdenest(1 + z) == 1 + ans
    assert sqrtdenest(Integral(z + 1, (x, 1, 2))) == \
        Integral(1 + ans, (x, 1, 2))
    assert sqrtdenest(x + sqrt(y)) == x + sqrt(y)


def test_subsets():
    assert subsets(1) == [[1]]
    assert subsets(4) == [
        [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0],
        [0, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1],
        [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]]


def test_issue_5653():
    assert sqrtdenest(
        sqrt(2 + sqrt(2 + sqrt(2)))) == sqrt(2 + sqrt(2 + sqrt(2)))

def test_issue_12420():
    assert sqrtdenest((3 - sqrt(2)*sqrt(4 + 3*I) + 3*I)/2) == I
    e = 3 - sqrt(2)*sqrt(4 + I) + 3*I
    assert sqrtdenest(e) == e

def test_sqrt_ratcomb():
    assert sqrtdenest(sqrt(1 + r3) + sqrt(3 + 3*r3) - sqrt(10 + 6*r3)) == 0

def test_issue_18041():
    e = -sqrt(-2 + 2*sqrt(3)*I)
    assert sqrtdenest(e) == -1 - sqrt(3)*I

def test_issue_19914():
    a = Integer(-8)
    b = Integer(-1)
    r = Integer(63)
    d2 = a*a - b*b*r

    assert _sqrt_numeric_denest(a, b, r, d2) == \
        sqrt(14)*I/2 + 3*sqrt(2)*I/2
    assert sqrtdenest(sqrt(-8-sqrt(63))) == sqrt(14)*I/2 + 3*sqrt(2)*I/2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\test_function.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray


@pytest.mark.parametrize("ufunc", [np.abs, np.sign])
# np.sign emits a warning with nans, <https://github.com/numpy/numpy/issues/15127>
@pytest.mark.filterwarnings("ignore:invalid value encountered in sign:RuntimeWarning")
def test_ufuncs_single_int(ufunc):
    a = pd.array([1, 2, -3, np.nan])
    result = ufunc(a)
    expected = pd.array(ufunc(a.astype(float)), dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    result = ufunc(s)
    expected = pd.Series(pd.array(ufunc(a.astype(float)), dtype="Int64"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.log, np.exp, np.sin, np.cos, np.sqrt])
def test_ufuncs_single_float(ufunc):
    a = pd.array([1, 2, -3, np.nan])
    with np.errstate(invalid="ignore"):
        result = ufunc(a)
        expected = FloatingArray(ufunc(a.astype(float)), mask=a._mask)
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    with np.errstate(invalid="ignore"):
        result = ufunc(s)
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.add, np.subtract])
def test_ufuncs_binary_int(ufunc):
    # two IntegerArrays
    a = pd.array([1, 2, -3, np.nan])
    result = ufunc(a, a)
    expected = pd.array(ufunc(a.astype(float), a.astype(float)), dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    # IntegerArray with numpy array
    arr = np.array([1, 2, 3, 4])
    result = ufunc(a, arr)
    expected = pd.array(ufunc(a.astype(float), arr), dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(arr, a)
    expected = pd.array(ufunc(arr, a.astype(float)), dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    # IntegerArray with scalar
    result = ufunc(a, 1)
    expected = pd.array(ufunc(a.astype(float), 1), dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(1, a)
    expected = pd.array(ufunc(1, a.astype(float)), dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


def test_ufunc_binary_output():
    a = pd.array([1, 2, np.nan])
    result = np.modf(a)
    expected = np.modf(a.to_numpy(na_value=np.nan, dtype="float"))
    expected = (pd.array(expected[0]), pd.array(expected[1]))

    assert isinstance(result, tuple)
    assert len(result) == 2

    for x, y in zip(result, expected):
        tm.assert_extension_array_equal(x, y)


@pytest.mark.parametrize("values", [[0, 1], [0, None]])
def test_ufunc_reduce_raises(values):
    arr = pd.array(values)

    res = np.add.reduce(arr)
    expected = arr.sum(skipna=False)
    tm.assert_almost_equal(res, expected)


@pytest.mark.parametrize(
    "pandasmethname, kwargs",
    [
        ("var", {"ddof": 0}),
        ("var", {"ddof": 1}),
        ("std", {"ddof": 0}),
        ("std", {"ddof": 1}),
        ("kurtosis", {}),
        ("skew", {}),
        ("sem", {}),
    ],
)
def test_stat_method(pandasmethname, kwargs):
    s = pd.Series(data=[1, 2, 3, 4, 5, 6, np.nan, np.nan], dtype="Int64")
    pandasmeth = getattr(s, pandasmethname)
    result = pandasmeth(**kwargs)
    s2 = pd.Series(data=[1, 2, 3, 4, 5, 6], dtype="Int64")
    pandasmeth = getattr(s2, pandasmethname)
    expected = pandasmeth(**kwargs)
    assert expected == result


def test_value_counts_na():
    arr = pd.array([1, 2, 1, pd.NA], dtype="Int64")
    result = arr.value_counts(dropna=False)
    ex_index = pd.Index([1, 2, pd.NA], dtype="Int64")
    assert ex_index.dtype == "Int64"
    expected = pd.Series([2, 1, 1], index=ex_index, dtype="Int64", name="count")
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([2, 1], index=arr[:2], dtype="Int64", name="count")
    assert expected.index.dtype == arr.dtype
    tm.assert_series_equal(result, expected)


def test_value_counts_empty():
    # https://github.com/pandas-dev/pandas/issues/33317
    ser = pd.Series([], dtype="Int64")
    result = ser.value_counts()
    idx = pd.Index([], dtype=ser.dtype)
    assert idx.dtype == ser.dtype
    expected = pd.Series([], index=idx, dtype="Int64", name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize():
    # GH 33172
    ser = pd.Series([1, 2, 1, pd.NA], dtype="Int64")
    result = ser.value_counts(normalize=True)
    expected = pd.Series([2, 1], index=ser[:2], dtype="Float64", name="proportion") / 3
    assert expected.index.dtype == ser.dtype
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("min_count", [0, 4])
def test_integer_array_sum(skipna, min_count, any_int_ea_dtype):
    dtype = any_int_ea_dtype
    arr = pd.array([1, 2, 3, None], dtype=dtype)
    result = arr.sum(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 6
    else:
        assert result is pd.NA


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("method", ["min", "max"])
def test_integer_array_min_max(skipna, method, any_int_ea_dtype):
    dtype = any_int_ea_dtype
    arr = pd.array([0, 1, None], dtype=dtype)
    func = getattr(arr, method)
    result = func(skipna=skipna)
    if skipna:
        assert result == (0 if method == "min" else 1)
    else:
        assert result is pd.NA


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("min_count", [0, 9])
def test_integer_array_prod(skipna, min_count, any_int_ea_dtype):
    dtype = any_int_ea_dtype
    arr = pd.array([1, 2, None], dtype=dtype)
    result = arr.prod(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 2
    else:
        assert result is pd.NA


@pytest.mark.parametrize(
    "values, expected", [([1, 2, 3], 6), ([1, 2, 3, None], 6), ([None], 0)]
)
def test_integer_array_numpy_sum(values, expected):
    arr = pd.array(values, dtype="Int64")
    result = np.sum(arr)
    assert result == expected


@pytest.mark.parametrize("op", ["sum", "prod", "min", "max"])
def test_dataframe_reductions(op):
    # https://github.com/pandas-dev/pandas/pull/32867
    # ensure the integers are not cast to float during reductions
    df = pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")})
    result = df.max()
    assert isinstance(result["a"], np.int64)


# TODO(jreback) - these need testing / are broken

# shift

# set_index (destroys type)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_describe.py ===
import numpy as np
import pytest

from pandas.compat.numpy import np_version_gte1p25

from pandas.core.dtypes.common import (
    is_complex_dtype,
    is_extension_array_dtype,
)

from pandas import (
    NA,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestSeriesDescribe:
    def test_describe_ints(self):
        ser = Series([0, 1, 2, 3, 4], name="int_data")
        result = ser.describe()
        expected = Series(
            [5, 2, ser.std(), 0, 1, 2, 3, 4],
            name="int_data",
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    def test_describe_bools(self):
        ser = Series([True, True, False, False, False], name="bool_data")
        result = ser.describe()
        expected = Series(
            [5, 2, False, 3], name="bool_data", index=["count", "unique", "top", "freq"]
        )
        tm.assert_series_equal(result, expected)

    def test_describe_strs(self):
        ser = Series(["a", "a", "b", "c", "d"], name="str_data")
        result = ser.describe()
        expected = Series(
            [5, 4, "a", 2], name="str_data", index=["count", "unique", "top", "freq"]
        )
        tm.assert_series_equal(result, expected)

    def test_describe_timedelta64(self):
        ser = Series(
            [
                Timedelta("1 days"),
                Timedelta("2 days"),
                Timedelta("3 days"),
                Timedelta("4 days"),
                Timedelta("5 days"),
            ],
            name="timedelta_data",
        )
        result = ser.describe()
        expected = Series(
            [5, ser[2], ser.std(), ser[0], ser[1], ser[2], ser[3], ser[4]],
            name="timedelta_data",
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    def test_describe_period(self):
        ser = Series(
            [Period("2020-01", "M"), Period("2020-01", "M"), Period("2019-12", "M")],
            name="period_data",
        )
        result = ser.describe()
        expected = Series(
            [3, 2, ser[0], 2],
            name="period_data",
            index=["count", "unique", "top", "freq"],
        )
        tm.assert_series_equal(result, expected)

    def test_describe_empty_object(self):
        # https://github.com/pandas-dev/pandas/issues/27183
        s = Series([None, None], dtype=object)
        result = s.describe()
        expected = Series(
            [0, 0, np.nan, np.nan],
            dtype=object,
            index=["count", "unique", "top", "freq"],
        )
        tm.assert_series_equal(result, expected)

        result = s[:0].describe()
        tm.assert_series_equal(result, expected)
        # ensure NaN, not None
        assert np.isnan(result.iloc[2])
        assert np.isnan(result.iloc[3])

    def test_describe_with_tz(self, tz_naive_fixture):
        # GH 21332
        tz = tz_naive_fixture
        name = str(tz_naive_fixture)
        start = Timestamp(2018, 1, 1)
        end = Timestamp(2018, 1, 5)
        s = Series(date_range(start, end, tz=tz), name=name)
        result = s.describe()
        expected = Series(
            [
                5,
                Timestamp(2018, 1, 3).tz_localize(tz),
                start.tz_localize(tz),
                s[1],
                s[2],
                s[3],
                end.tz_localize(tz),
            ],
            name=name,
            index=["count", "mean", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    def test_describe_with_tz_numeric(self):
        name = tz = "CET"
        start = Timestamp(2018, 1, 1)
        end = Timestamp(2018, 1, 5)
        s = Series(date_range(start, end, tz=tz), name=name)

        result = s.describe()

        expected = Series(
            [
                5,
                Timestamp("2018-01-03 00:00:00", tz=tz),
                Timestamp("2018-01-01 00:00:00", tz=tz),
                Timestamp("2018-01-02 00:00:00", tz=tz),
                Timestamp("2018-01-03 00:00:00", tz=tz),
                Timestamp("2018-01-04 00:00:00", tz=tz),
                Timestamp("2018-01-05 00:00:00", tz=tz),
            ],
            name=name,
            index=["count", "mean", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    def test_datetime_is_numeric_includes_datetime(self):
        s = Series(date_range("2012", periods=3))
        result = s.describe()
        expected = Series(
            [
                3,
                Timestamp("2012-01-02"),
                Timestamp("2012-01-01"),
                Timestamp("2012-01-01T12:00:00"),
                Timestamp("2012-01-02"),
                Timestamp("2012-01-02T12:00:00"),
                Timestamp("2012-01-03"),
            ],
            index=["count", "mean", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:Casting complex values to real discards")
    def test_numeric_result_dtype(self, any_numeric_dtype):
        # GH#48340 - describe should always return float on non-complex numeric input
        if is_extension_array_dtype(any_numeric_dtype):
            dtype = "Float64"
        else:
            dtype = "complex128" if is_complex_dtype(any_numeric_dtype) else None

        ser = Series([0, 1], dtype=any_numeric_dtype)
        if dtype == "complex128" and np_version_gte1p25:
            with pytest.raises(
                TypeError, match=r"^a must be an array of real numbers$"
            ):
                ser.describe()
            return
        result = ser.describe()
        expected = Series(
            [
                2.0,
                0.5,
                ser.std(),
                0,
                0.25,
                0.5,
                0.75,
                1.0,
            ],
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype=dtype,
        )
        tm.assert_series_equal(result, expected)

    def test_describe_one_element_ea(self):
        # GH#52515
        ser = Series([0.0], dtype="Float64")
        with tm.assert_produces_warning(None):
            result = ser.describe()
        expected = Series(
            [1, 0, NA, 0, 0, 0, 0, 0],
            dtype="Float64",
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\tests\test_textplot.py ===
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.plotting.textplot import textplot_str

from sympy.utilities.exceptions import ignore_warnings


def test_axes_alignment():
    x = Symbol('x')
    lines = [
        '      1 |                                                     ..',
        '        |                                                  ...  ',
        '        |                                                ..     ',
        '        |                                             ...       ',
        '        |                                          ...          ',
        '        |                                        ..             ',
        '        |                                     ...               ',
        '        |                                  ...                  ',
        '        |                                ..                     ',
        '        |                             ...                       ',
        '      0 |--------------------------...--------------------------',
        '        |                       ...                             ',
        '        |                     ..                                ',
        '        |                  ...                                  ',
        '        |               ...                                     ',
        '        |             ..                                        ',
        '        |          ...                                          ',
        '        |       ...                                             ',
        '        |     ..                                                ',
        '        |  ...                                                  ',
        '     -1 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    assert lines == list(textplot_str(x, -1, 1))

    lines = [
        '      1 |                                                     ..',
        '        |                                                 ....  ',
        '        |                                              ...      ',
        '        |                                           ...         ',
        '        |                                       ....            ',
        '        |                                    ...                ',
        '        |                                 ...                   ',
        '        |                             ....                      ',
        '      0 |--------------------------...--------------------------',
        '        |                      ....                             ',
        '        |                   ...                                 ',
        '        |                ...                                    ',
        '        |            ....                                       ',
        '        |         ...                                           ',
        '        |      ...                                              ',
        '        |  ....                                                 ',
        '     -1 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    assert lines == list(textplot_str(x, -1, 1, H=17))


def test_singularity():
    x = Symbol('x')
    lines = [
        '     54 | .                                                     ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ','        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '   27.5 |--.----------------------------------------------------',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |   .                                                   ',
        '        |    \\                                                  ',
        '        |     \\                                                 ',
        '        |      ..                                               ',
        '        |        ...                                            ',
        '        |           .............                               ',
        '      1 |_______________________________________________________',
        '         0                          0.5                        1'
    ]
    assert lines == list(textplot_str(1/x, 0, 1))

    lines = [
        '      0 |                                                 ......',
        '        |                                         ........      ',
        '        |                                 ........              ',
        '        |                           ......                      ',
        '        |                      .....                            ',
        '        |                  ....                                 ',
        '        |               ...                                     ',
        '        |             ..                                        ',
        '        |          ...                                          ',
        '        |         /                                             ',
        '     -2 |-------..----------------------------------------------',
        '        |      /                                                ',
        '        |     /                                                 ',
        '        |    /                                                  ',
        '        |   .                                                   ',
        '        |                                                       ',
        '        |  .                                                    ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '     -4 |_______________________________________________________',
        '         0                          0.5                        1'
    ]
    # RuntimeWarning: divide by zero encountered in log
    with ignore_warnings(RuntimeWarning):
        assert lines == list(textplot_str(log(x), 0, 1))


def test_sinc():
    x = Symbol('x')
    lines = [
        '      1 |                          . .                          ',
        '        |                         .   .                         ',
        '        |                                                       ',
        '        |                        .     .                        ',
        '        |                                                       ',
        '        |                       .       .                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                      .         .                      ',
        '        |                                                       ',
        '    0.4 |-------------------------------------------------------',
        '        |                     .           .                     ',
        '        |                                                       ',
        '        |                    .             .                    ',
        '        |                                                       ',
        '        |    .....                                     .....    ',
        '        |  ..     \\         .               .         /     ..  ',
        '        | /        \\                                 /        \\ ',
        '        |/          \\      .                 .      /          \\',
        '        |            \\    /                   \\    /            ',
        '   -0.2 |_______________________________________________________',
        '         -10                        0                          10'
    ]
    # RuntimeWarning: invalid value encountered in double_scalars
    with ignore_warnings(RuntimeWarning):
        assert lines == list(textplot_str(sin(x)/x, -10, 10))


def test_imaginary():
    x = Symbol('x')
    lines = [
        '      1 |                                                     ..',
        '        |                                                   ..  ',
        '        |                                                ...    ',
        '        |                                              ..       ',
        '        |                                            ..         ',
        '        |                                          ..           ',
        '        |                                        ..             ',
        '        |                                      ..               ',
        '        |                                    ..                 ',
        '        |                                   /                   ',
        '    0.5 |----------------------------------/--------------------',
        '        |                                ..                     ',
        '        |                               /                       ',
        '        |                              .                        ',
        '        |                                                       ',
        '        |                             .                         ',
        '        |                            .                          ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '      0 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    # RuntimeWarning: invalid value encountered in sqrt
    with ignore_warnings(RuntimeWarning):
        assert list(textplot_str(sqrt(x), -1, 1)) == lines

    lines = [
        '      1 |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '      0 |-------------------------------------------------------',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '     -1 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    assert list(textplot_str(S.ImaginaryUnit, -1, 1)) == lines

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_python.py ===
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.series.limits import limit

from sympy.printing.python import python

from sympy.testing.pytest import raises, XFAIL

x, y = symbols('x,y')
th = Symbol('theta')
ph = Symbol('phi')


def test_python_basic():
    # Simple numbers/symbols
    assert python(-Rational(1)/2) == "e = Rational(-1, 2)"
    assert python(-Rational(13)/22) == "e = Rational(-13, 22)"
    assert python(oo) == "e = oo"

    # Powers
    assert python(x**2) == "x = Symbol(\'x\')\ne = x**2"
    assert python(1/x) == "x = Symbol('x')\ne = 1/x"
    assert python(y*x**-2) == "y = Symbol('y')\nx = Symbol('x')\ne = y/x**2"
    assert python(
        x**Rational(-5, 2)) == "x = Symbol('x')\ne = x**Rational(-5, 2)"

    # Sums of terms
    assert python(x**2 + x + 1) in [
        "x = Symbol('x')\ne = 1 + x + x**2",
        "x = Symbol('x')\ne = x + x**2 + 1",
        "x = Symbol('x')\ne = x**2 + x + 1", ]
    assert python(1 - x) in [
        "x = Symbol('x')\ne = 1 - x",
        "x = Symbol('x')\ne = -x + 1"]
    assert python(1 - 2*x) in [
        "x = Symbol('x')\ne = 1 - 2*x",
        "x = Symbol('x')\ne = -2*x + 1"]
    assert python(1 - Rational(3, 2)*y/x) in [
        "y = Symbol('y')\nx = Symbol('x')\ne = 1 - 3/2*y/x",
        "y = Symbol('y')\nx = Symbol('x')\ne = -3/2*y/x + 1",
        "y = Symbol('y')\nx = Symbol('x')\ne = 1 - 3*y/(2*x)"]

    # Multiplication
    assert python(x/y) == "x = Symbol('x')\ny = Symbol('y')\ne = x/y"
    assert python(-x/y) == "x = Symbol('x')\ny = Symbol('y')\ne = -x/y"
    assert python((x + 2)/y) in [
        "y = Symbol('y')\nx = Symbol('x')\ne = 1/y*(2 + x)",
        "y = Symbol('y')\nx = Symbol('x')\ne = 1/y*(x + 2)",
        "x = Symbol('x')\ny = Symbol('y')\ne = 1/y*(2 + x)",
        "x = Symbol('x')\ny = Symbol('y')\ne = (2 + x)/y",
        "x = Symbol('x')\ny = Symbol('y')\ne = (x + 2)/y"]
    assert python((1 + x)*y) in [
        "y = Symbol('y')\nx = Symbol('x')\ne = y*(1 + x)",
        "y = Symbol('y')\nx = Symbol('x')\ne = y*(x + 1)", ]

    # Check for proper placement of negative sign
    assert python(-5*x/(x + 10)) == "x = Symbol('x')\ne = -5*x/(x + 10)"
    assert python(1 - Rational(3, 2)*(x + 1)) in [
        "x = Symbol('x')\ne = Rational(-3, 2)*x + Rational(-1, 2)",
        "x = Symbol('x')\ne = -3*x/2 + Rational(-1, 2)",
        "x = Symbol('x')\ne = -3*x/2 + Rational(-1, 2)"
    ]


def test_python_keyword_symbol_name_escaping():
    # Check for escaping of keywords
    assert python(
        5*Symbol("lambda")) == "lambda_ = Symbol('lambda')\ne = 5*lambda_"
    assert (python(5*Symbol("lambda") + 7*Symbol("lambda_")) ==
            "lambda__ = Symbol('lambda')\nlambda_ = Symbol('lambda_')\ne = 7*lambda_ + 5*lambda__")
    assert (python(5*Symbol("for") + Function("for_")(8)) ==
            "for__ = Symbol('for')\nfor_ = Function('for_')\ne = 5*for__ + for_(8)")


def test_python_keyword_function_name_escaping():
    assert python(
        5*Function("for")(8)) == "for_ = Function('for')\ne = 5*for_(8)"


def test_python_relational():
    assert python(Eq(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = Eq(x, y)"
    assert python(Ge(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x >= y"
    assert python(Le(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x <= y"
    assert python(Gt(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x > y"
    assert python(Lt(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x < y"
    assert python(Ne(x/(y + 1), y**2)) in [
        "x = Symbol('x')\ny = Symbol('y')\ne = Ne(x/(1 + y), y**2)",
        "x = Symbol('x')\ny = Symbol('y')\ne = Ne(x/(y + 1), y**2)"]


def test_python_functions():
    # Simple
    assert python(2*x + exp(x)) in "x = Symbol('x')\ne = 2*x + exp(x)"
    assert python(sqrt(2)) == 'e = sqrt(2)'
    assert python(2**Rational(1, 3)) == 'e = 2**Rational(1, 3)'
    assert python(sqrt(2 + pi)) == 'e = sqrt(2 + pi)'
    assert python((2 + pi)**Rational(1, 3)) == 'e = (2 + pi)**Rational(1, 3)'
    assert python(2**Rational(1, 4)) == 'e = 2**Rational(1, 4)'
    assert python(Abs(x)) == "x = Symbol('x')\ne = Abs(x)"
    assert python(
        Abs(x/(x**2 + 1))) in ["x = Symbol('x')\ne = Abs(x/(1 + x**2))",
            "x = Symbol('x')\ne = Abs(x/(x**2 + 1))"]

    # Univariate/Multivariate functions
    f = Function('f')
    assert python(f(x)) == "x = Symbol('x')\nf = Function('f')\ne = f(x)"
    assert python(f(x, y)) == "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x, y)"
    assert python(f(x/(y + 1), y)) in [
        "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x/(1 + y), y)",
        "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x/(y + 1), y)"]

    # Nesting of square roots
    assert python(sqrt((sqrt(x + 1)) + 1)) in [
        "x = Symbol('x')\ne = sqrt(1 + sqrt(1 + x))",
        "x = Symbol('x')\ne = sqrt(sqrt(x + 1) + 1)"]

    # Nesting of powers
    assert python((((x + 1)**Rational(1, 3)) + 1)**Rational(1, 3)) in [
        "x = Symbol('x')\ne = (1 + (1 + x)**Rational(1, 3))**Rational(1, 3)",
        "x = Symbol('x')\ne = ((x + 1)**Rational(1, 3) + 1)**Rational(1, 3)"]

    # Function powers
    assert python(sin(x)**2) == "x = Symbol('x')\ne = sin(x)**2"


@XFAIL
def test_python_functions_conjugates():
    a, b = map(Symbol, 'ab')
    assert python( conjugate(a + b*I) ) == '_     _\na - I*b'
    assert python( conjugate(exp(a + b*I)) ) == ' _     _\n a - I*b\ne       '


def test_python_derivatives():
    # Simple
    f_1 = Derivative(log(x), x, evaluate=False)
    assert python(f_1) == "x = Symbol('x')\ne = Derivative(log(x), x)"

    f_2 = Derivative(log(x), x, evaluate=False) + x
    assert python(f_2) == "x = Symbol('x')\ne = x + Derivative(log(x), x)"

    # Multiple symbols
    f_3 = Derivative(log(x) + x**2, x, y, evaluate=False)
    assert python(f_3) == \
        "x = Symbol('x')\ny = Symbol('y')\ne = Derivative(x**2 + log(x), x, y)"

    f_4 = Derivative(2*x*y, y, x, evaluate=False) + x**2
    assert python(f_4) in [
        "x = Symbol('x')\ny = Symbol('y')\ne = x**2 + Derivative(2*x*y, y, x)",
        "x = Symbol('x')\ny = Symbol('y')\ne = Derivative(2*x*y, y, x) + x**2"]


def test_python_integrals():
    # Simple
    f_1 = Integral(log(x), x)
    assert python(f_1) == "x = Symbol('x')\ne = Integral(log(x), x)"

    f_2 = Integral(x**2, x)
    assert python(f_2) == "x = Symbol('x')\ne = Integral(x**2, x)"

    # Double nesting of pow
    f_3 = Integral(x**(2**x), x)
    assert python(f_3) == "x = Symbol('x')\ne = Integral(x**(2**x), x)"

    # Definite integrals
    f_4 = Integral(x**2, (x, 1, 2))
    assert python(f_4) == "x = Symbol('x')\ne = Integral(x**2, (x, 1, 2))"

    f_5 = Integral(x**2, (x, Rational(1, 2), 10))
    assert python(
        f_5) == "x = Symbol('x')\ne = Integral(x**2, (x, Rational(1, 2), 10))"

    # Nested integrals
    f_6 = Integral(x**2*y**2, x, y)
    assert python(f_6) == "x = Symbol('x')\ny = Symbol('y')\ne = Integral(x**2*y**2, x, y)"


def test_python_matrix():
    p = python(Matrix([[x**2+1, 1], [y, x+y]]))
    s = "x = Symbol('x')\ny = Symbol('y')\ne = MutableDenseMatrix([[x**2 + 1, 1], [y, x + y]])"
    assert p == s

def test_python_limits():
    assert python(limit(x, x, oo)) == 'e = oo'
    assert python(limit(x**2, x, 0)) == 'e = 0'

def test_issue_20762():
    # Make sure Python removes curly braces from subscripted variables
    a_b = Symbol('a_{b}')
    b = Symbol('b')
    expr = a_b*b
    assert python(expr) == "a_b = Symbol('a_{b}')\nb = Symbol('b')\ne = a_b*b"


def test_settings():
    raises(TypeError, lambda: python(x, method="garbage"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\tests\test_subscheck.py ===
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.error_functions import (Ei, erf, erfi)
from sympy.integrals.integrals import Integral

from sympy.solvers.ode.subscheck import checkodesol, checksysodesol

from sympy.functions import besselj, bessely

from sympy.testing.pytest import raises, slow


C0, C1, C2, C3, C4 = symbols('C0:5')
u, x, y, z = symbols('u,x:z', real=True)
f = Function('f')
g = Function('g')
h = Function('h')


@slow
def test_checkodesol():
    # For the most part, checkodesol is well tested in the tests below.
    # These tests only handle cases not checked below.
    raises(ValueError, lambda: checkodesol(f(x, y).diff(x), Eq(f(x, y), x)))
    raises(ValueError, lambda: checkodesol(f(x).diff(x), Eq(f(x, y),
           x), f(x, y)))
    assert checkodesol(f(x).diff(x), Eq(f(x, y), x)) == \
        (False, -f(x).diff(x) + f(x, y).diff(x) - 1)
    assert checkodesol(f(x).diff(x), Eq(f(x), x)) is not True
    assert checkodesol(f(x).diff(x), Eq(f(x), x)) == (False, 1)
    sol1 = Eq(f(x)**5 + 11*f(x) - 2*f(x) + x, 0)
    assert checkodesol(diff(sol1.lhs, x), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x)*exp(f(x)), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 2), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 2)*exp(f(x)), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 3), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 3)*exp(f(x)), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 3), Eq(f(x), x*log(x))) == \
        (False, 60*x**4*((log(x) + 1)**2 + log(x))*(
        log(x) + 1)*log(x)**2 - 5*x**4*log(x)**4 - 9)
    assert checkodesol(diff(exp(f(x)) + x, x)*x, Eq(exp(f(x)) + x, 0)) == \
        (True, 0)
    assert checkodesol(diff(exp(f(x)) + x, x)*x, Eq(exp(f(x)) + x, 0),
        solve_for_func=False) == (True, 0)
    assert checkodesol(f(x).diff(x, 2), [Eq(f(x), C1 + C2*x),
        Eq(f(x), C2 + C1*x), Eq(f(x), C1*x + C2*x**2)]) == \
        [(True, 0), (True, 0), (False, C2)]
    assert checkodesol(f(x).diff(x, 2), {Eq(f(x), C1 + C2*x),
        Eq(f(x), C2 + C1*x), Eq(f(x), C1*x + C2*x**2)}) == \
        {(True, 0), (True, 0), (False, C2)}
    assert checkodesol(f(x).diff(x) - 1/f(x)/2, Eq(f(x)**2, x)) == \
        [(True, 0), (True, 0)]
    assert checkodesol(f(x).diff(x) - f(x), Eq(C1*exp(x), f(x))) == (True, 0)
    # Based on test_1st_homogeneous_coeff_ode2_eq3sol.  Make sure that
    # checkodesol tries back substituting f(x) when it can.
    eq3 = x*exp(f(x)/x) + f(x) - x*f(x).diff(x)
    sol3 = Eq(f(x), log(log(C1/x)**(-x)))
    assert not checkodesol(eq3, sol3)[1].has(f(x))
    # This case was failing intermittently depending on hash-seed:
    eqn = Eq(Derivative(x*Derivative(f(x), x), x)/x, exp(x))
    sol = Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))
    assert checkodesol(eqn, sol, order=2, solve_for_func=False)[0]
    eq = x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (2*x**2 +25)*f(x)
    sol = Eq(f(x), C1*besselj(5*I, sqrt(2)*x) + C2*bessely(5*I, sqrt(2)*x))
    assert checkodesol(eq, sol) == (True, 0)

    eqs = [Eq(f(x).diff(x), f(x) + g(x)), Eq(g(x).diff(x), f(x) + g(x))]
    sol = [Eq(f(x), -C1 + C2*exp(2*x)), Eq(g(x), C1 + C2*exp(2*x))]
    assert checkodesol(eqs, sol) == (True, [0, 0])


def test_checksysodesol():
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')
    eq = (Eq(diff(x(t),t), 9*y(t)), Eq(diff(y(t),t), 12*x(t)))
    sol = [Eq(x(t), 9*C1*exp(-6*sqrt(3)*t) + 9*C2*exp(6*sqrt(3)*t)), \
    Eq(y(t), -6*sqrt(3)*C1*exp(-6*sqrt(3)*t) + 6*sqrt(3)*C2*exp(6*sqrt(3)*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 2*x(t) + 4*y(t)), Eq(diff(y(t),t), 12*x(t) + 41*y(t)))
    sol = [Eq(x(t), 4*C1*exp(t*(-sqrt(1713)/2 + Rational(43, 2))) + 4*C2*exp(t*(sqrt(1713)/2 + \
    Rational(43, 2)))), Eq(y(t), C1*(-sqrt(1713)/2 + Rational(39, 2))*exp(t*(-sqrt(1713)/2 + \
    Rational(43, 2))) + C2*(Rational(39, 2) + sqrt(1713)/2)*exp(t*(sqrt(1713)/2 + Rational(43, 2))))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), x(t) + y(t)), Eq(diff(y(t),t), -2*x(t) + 2*y(t)))
    sol = [Eq(x(t), (C1*sin(sqrt(7)*t/2) + C2*cos(sqrt(7)*t/2))*exp(t*Rational(3, 2))), \
    Eq(y(t), ((C1/2 - sqrt(7)*C2/2)*sin(sqrt(7)*t/2) + (sqrt(7)*C1/2 + \
    C2/2)*cos(sqrt(7)*t/2))*exp(t*Rational(3, 2)))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), x(t) + y(t) + 9), Eq(diff(y(t),t), 2*x(t) + 5*y(t) + 23))
    sol = [Eq(x(t), C1*exp(t*(-sqrt(6) + 3)) + C2*exp(t*(sqrt(6) + 3)) - \
    Rational(22, 3)), Eq(y(t), C1*(-sqrt(6) + 2)*exp(t*(-sqrt(6) + 3)) + C2*(2 + \
    sqrt(6))*exp(t*(sqrt(6) + 3)) - Rational(5, 3))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), x(t) + y(t) + 81), Eq(diff(y(t),t), -2*x(t) + y(t) + 23))
    sol = [Eq(x(t), (C1*sin(sqrt(2)*t) + C2*cos(sqrt(2)*t))*exp(t) - Rational(58, 3)), \
    Eq(y(t), (sqrt(2)*C1*cos(sqrt(2)*t) - sqrt(2)*C2*sin(sqrt(2)*t))*exp(t) - Rational(185, 3))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 5*t*x(t) + 2*y(t)), Eq(diff(y(t),t), 2*x(t) + 5*t*y(t)))
    sol = [Eq(x(t), (C1*exp(Integral(2, t).doit()) + C2*exp(-(Integral(2, t)).doit()))*\
    exp((Integral(5*t, t)).doit())), Eq(y(t), (C1*exp((Integral(2, t)).doit()) - \
    C2*exp(-(Integral(2, t)).doit()))*exp((Integral(5*t, t)).doit()))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + 5*t*y(t)))
    sol = [Eq(x(t), (C1*cos((Integral(t**2, t)).doit()) + C2*sin((Integral(t**2, t)).doit()))*\
    exp((Integral(5*t, t)).doit())), Eq(y(t), (-C1*sin((Integral(t**2, t)).doit()) + \
    C2*cos((Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit()))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + (5*t+9*t**2)*y(t)))
    sol = [Eq(x(t), (C1*exp((-sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()) + \
    C2*exp((sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit())), \
    Eq(y(t), (C1*(-sqrt(77)/2 + Rational(9, 2))*exp((-sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()) + \
    C2*(sqrt(77)/2 + Rational(9, 2))*exp((sqrt(77)/2 + Rational(9, 2))*(Integral(t**2, t)).doit()))*exp((Integral(5*t, t)).doit()))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t,t), 5*x(t) + 43*y(t)), Eq(diff(y(t),t,t), x(t) + 9*y(t)))
    root0 = -sqrt(-sqrt(47) + 7)
    root1 = sqrt(-sqrt(47) + 7)
    root2 = -sqrt(sqrt(47) + 7)
    root3 = sqrt(sqrt(47) + 7)
    sol = [Eq(x(t), 43*C1*exp(t*root0) + 43*C2*exp(t*root1) + 43*C3*exp(t*root2) + 43*C4*exp(t*root3)), \
    Eq(y(t), C1*(root0**2 - 5)*exp(t*root0) + C2*(root1**2 - 5)*exp(t*root1) + \
    C3*(root2**2 - 5)*exp(t*root2) + C4*(root3**2 - 5)*exp(t*root3))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t,t), 8*x(t)+3*y(t)+31), Eq(diff(y(t),t,t), 9*x(t)+7*y(t)+12))
    root0 = -sqrt(-sqrt(109)/2 + Rational(15, 2))
    root1 = sqrt(-sqrt(109)/2 + Rational(15, 2))
    root2 = -sqrt(sqrt(109)/2 + Rational(15, 2))
    root3 = sqrt(sqrt(109)/2 + Rational(15, 2))
    sol = [Eq(x(t), 3*C1*exp(t*root0) + 3*C2*exp(t*root1) + 3*C3*exp(t*root2) + 3*C4*exp(t*root3) - Rational(181, 29)), \
    Eq(y(t), C1*(root0**2 - 8)*exp(t*root0) + C2*(root1**2 - 8)*exp(t*root1) + \
    C3*(root2**2 - 8)*exp(t*root2) + C4*(root3**2 - 8)*exp(t*root3) + Rational(183, 29))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t,t) - 9*diff(y(t),t) + 7*x(t),0), Eq(diff(y(t),t,t) + 9*diff(x(t),t) + 7*y(t),0))
    sol = [Eq(x(t), C1*cos(t*(Rational(9, 2) + sqrt(109)/2)) + C2*sin(t*(Rational(9, 2) + sqrt(109)/2)) + \
    C3*cos(t*(-sqrt(109)/2 + Rational(9, 2))) + C4*sin(t*(-sqrt(109)/2 + Rational(9, 2)))), Eq(y(t), -C1*sin(t*(Rational(9, 2) + sqrt(109)/2)) \
    + C2*cos(t*(Rational(9, 2) + sqrt(109)/2)) - C3*sin(t*(-sqrt(109)/2 + Rational(9, 2))) + C4*cos(t*(-sqrt(109)/2 + Rational(9, 2))))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t,t), 9*t*diff(y(t),t)-9*y(t)), Eq(diff(y(t),t,t),7*t*diff(x(t),t)-7*x(t)))
    I1 = sqrt(6)*7**Rational(1, 4)*sqrt(pi)*erfi(sqrt(6)*7**Rational(1, 4)*t/2)/2 - exp(3*sqrt(7)*t**2/2)/t
    I2 = -sqrt(6)*7**Rational(1, 4)*sqrt(pi)*erf(sqrt(6)*7**Rational(1, 4)*t/2)/2 - exp(-3*sqrt(7)*t**2/2)/t
    sol = [Eq(x(t), C3*t + t*(9*C1*I1 + 9*C2*I2)), Eq(y(t), C4*t + t*(3*sqrt(7)*C1*I1 - 3*sqrt(7)*C2*I2))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), 21*x(t)), Eq(diff(y(t),t), 17*x(t)+3*y(t)), Eq(diff(z(t),t), 5*x(t)+7*y(t)+9*z(t)))
    sol = [Eq(x(t), C1*exp(21*t)), Eq(y(t), 17*C1*exp(21*t)/18 + C2*exp(3*t)), \
    Eq(z(t), 209*C1*exp(21*t)/216 - 7*C2*exp(3*t)/6 + C3*exp(9*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(diff(x(t),t),3*y(t)-11*z(t)),Eq(diff(y(t),t),7*z(t)-3*x(t)),Eq(diff(z(t),t),11*x(t)-7*y(t)))
    sol = [Eq(x(t), 7*C0 + sqrt(179)*C1*cos(sqrt(179)*t) + (77*C1/3 + 130*C2/3)*sin(sqrt(179)*t)), \
    Eq(y(t), 11*C0 + sqrt(179)*C2*cos(sqrt(179)*t) + (-58*C1/3 - 77*C2/3)*sin(sqrt(179)*t)), \
    Eq(z(t), 3*C0 + sqrt(179)*(-7*C1/3 - 11*C2/3)*cos(sqrt(179)*t) + (11*C1 - 7*C2)*sin(sqrt(179)*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(3*diff(x(t),t),4*5*(y(t)-z(t))),Eq(4*diff(y(t),t),3*5*(z(t)-x(t))),Eq(5*diff(z(t),t),3*4*(x(t)-y(t))))
    sol = [Eq(x(t), C0 + 5*sqrt(2)*C1*cos(5*sqrt(2)*t) + (12*C1/5 + 164*C2/15)*sin(5*sqrt(2)*t)), \
    Eq(y(t), C0 + 5*sqrt(2)*C2*cos(5*sqrt(2)*t) + (-51*C1/10 - 12*C2/5)*sin(5*sqrt(2)*t)), \
    Eq(z(t), C0 + 5*sqrt(2)*(-9*C1/25 - 16*C2/25)*cos(5*sqrt(2)*t) + (12*C1/5 - 12*C2/5)*sin(5*sqrt(2)*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(diff(x(t),t),4*x(t) - z(t)),Eq(diff(y(t),t),2*x(t)+2*y(t)-z(t)),Eq(diff(z(t),t),3*x(t)+y(t)))
    sol = [Eq(x(t), C1*exp(2*t) + C2*t*exp(2*t) + C2*exp(2*t) + C3*t**2*exp(2*t)/2 + C3*t*exp(2*t) + C3*exp(2*t)), \
    Eq(y(t), C1*exp(2*t) + C2*t*exp(2*t) + C2*exp(2*t) + C3*t**2*exp(2*t)/2 + C3*t*exp(2*t)), \
    Eq(z(t), 2*C1*exp(2*t) + 2*C2*t*exp(2*t) + C2*exp(2*t) + C3*t**2*exp(2*t) + C3*t*exp(2*t) + C3*exp(2*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(diff(x(t),t),4*x(t) - y(t) - 2*z(t)),Eq(diff(y(t),t),2*x(t) + y(t)- 2*z(t)),Eq(diff(z(t),t),5*x(t)-3*z(t)))
    sol = [Eq(x(t), C1*exp(2*t) + C2*(-sin(t) + 3*cos(t)) + C3*(3*sin(t) + cos(t))), \
    Eq(y(t), C2*(-sin(t) + 3*cos(t)) + C3*(3*sin(t) + cos(t))), Eq(z(t), C1*exp(2*t) + 5*C2*cos(t) + 5*C3*sin(t))]
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])

    eq = (Eq(diff(x(t),t),x(t)*y(t)**3), Eq(diff(y(t),t),y(t)**5))
    sol = [Eq(x(t), C1*exp((-1/(4*C2 + 4*t))**(Rational(-1, 4)))), Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), C1*exp(-1/(-1/(4*C2 + 4*t))**Rational(1, 4))), Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), C1*exp(-I/(-1/(4*C2 + 4*t))**Rational(1, 4))), Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), C1*exp(I/(-1/(4*C2 + 4*t))**Rational(1, 4))), Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(diff(x(t),t), exp(3*x(t))*y(t)**3),Eq(diff(y(t),t), y(t)**5))
    sol = [Eq(x(t), -log(C1 - 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), -log(C1 + 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), -log(C1 + 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)), \
    Eq(x(t), -log(C1 - 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3), Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    assert checksysodesol(eq, sol) == (True, [0, 0])

    eq = (Eq(x(t),t*diff(x(t),t)+diff(x(t),t)*diff(y(t),t)), Eq(y(t),t*diff(y(t),t)+diff(y(t),t)**2))
    sol = {Eq(x(t), C1*C2 + C1*t), Eq(y(t), C2**2 + C2*t)}
    assert checksysodesol(eq, sol) == (True, [0, 0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_rde.py ===
"""Most of these tests come from the examples in Bronstein's book."""
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
    NonElementaryIntegralException)
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
    normal_denom, special_denom, bound_degree, spde, solve_poly_rde,
    no_cancel_equal, cancel_primitive, cancel_exp, rischDE)

from sympy.testing.pytest import raises
from sympy.abc import x, t, z, n

t0, t1, t2, k = symbols('t:3 k')


def test_order_at():
    a = Poly(t**4, t)
    b = Poly((t**2 + 1)**3*t, t)
    c = Poly((t**2 + 1)**6*t, t)
    d = Poly((t**2 + 1)**10*t**10, t)
    e = Poly((t**2 + 1)**100*t**37, t)
    p1 = Poly(t, t)
    p2 = Poly(1 + t**2, t)
    assert order_at(a, p1, t) == 4
    assert order_at(b, p1, t) == 1
    assert order_at(c, p1, t) == 1
    assert order_at(d, p1, t) == 10
    assert order_at(e, p1, t) == 37
    assert order_at(a, p2, t) == 0
    assert order_at(b, p2, t) == 3
    assert order_at(c, p2, t) == 6
    assert order_at(d, p1, t) == 10
    assert order_at(e, p2, t) == 100
    assert order_at(Poly(0, t), Poly(t, t), t) is oo
    assert order_at_oo(Poly(t**2 - 1, t), Poly(t + 1), t) == \
        order_at_oo(Poly(t - 1, t), Poly(1, t), t) == -1
    assert order_at_oo(Poly(0, t), Poly(1, t), t) is oo

def test_weak_normalizer():
    a = Poly((1 + x)*t**5 + 4*t**4 + (-1 - 3*x)*t**3 - 4*t**2 + (-2 + 2*x)*t, t)
    d = Poly(t**4 - 3*t**2 + 2, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    r = weak_normalizer(a, d, DE, z)
    assert r == (Poly(t**5 - t**4 - 4*t**3 + 4*t**2 + 4*t - 4, t, domain='ZZ[x]'),
        (Poly((1 + x)*t**2 + x*t, t, domain='ZZ[x]'),
         Poly(t + 1, t, domain='ZZ[x]')))
    assert weak_normalizer(r[1][0], r[1][1], DE) == (Poly(1, t), r[1])
    r = weak_normalizer(Poly(1 + t**2), Poly(t**2 - 1, t), DE, z)
    assert r == (Poly(t**4 - 2*t**2 + 1, t), (Poly(-3*t**2 + 1, t), Poly(t**2 - 1, t)))
    assert weak_normalizer(r[1][0], r[1][1], DE, z) == (Poly(1, t), r[1])
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2)]})
    r = weak_normalizer(Poly(1 + t**2), Poly(t, t), DE, z)
    assert r == (Poly(t, t), (Poly(0, t), Poly(1, t)))
    assert weak_normalizer(r[1][0], r[1][1], DE, z) == (Poly(1, t), r[1])


def test_normal_denom():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    raises(NonElementaryIntegralException, lambda: normal_denom(Poly(1, x), Poly(1, x),
    Poly(1, x), Poly(x, x), DE))
    fa, fd = Poly(t**2 + 1, t), Poly(1, t)
    ga, gd = Poly(1, t), Poly(t**2, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert normal_denom(fa, fd, ga, gd, DE) == \
        (Poly(t, t), (Poly(t**3 - t**2 + t - 1, t), Poly(1, t)), (Poly(1, t),
        Poly(1, t)), Poly(t, t))


def test_special_denom():
    # TODO: add more tests here
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert special_denom(Poly(1, t), Poly(t**2, t), Poly(1, t), Poly(t**2 - 1, t),
    Poly(t, t), DE) == \
        (Poly(1, t), Poly(t**2 - 1, t), Poly(t**2 - 1, t), Poly(t, t))
#    assert special_denom(Poly(1, t), Poly(2*x, t), Poly((1 + 2*x)*t, t), DE) == 1

    # issue 3940
    # Note, this isn't a very good test, because the denominator is just 1,
    # but at least it tests the exp cancellation case
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-2*x*t0, t0),
        Poly(I*k*t1, t1)]})
    DE.decrement_level()
    assert special_denom(Poly(1, t0), Poly(I*k, t0), Poly(1, t0), Poly(t0, t0),
    Poly(1, t0), DE) == \
        (Poly(1, t0, domain='ZZ'), Poly(I*k, t0, domain='ZZ_I[k,x]'),
                Poly(t0, t0, domain='ZZ'), Poly(1, t0, domain='ZZ'))


    assert special_denom(Poly(1, t), Poly(t**2, t), Poly(1, t), Poly(t**2 - 1, t),
    Poly(t, t), DE, case='tan') == \
           (Poly(1, t, t0, domain='ZZ'), Poly(t**2, t0, t, domain='ZZ[x]'),
            Poly(t, t, t0, domain='ZZ'), Poly(1, t0, domain='ZZ'))

    raises(ValueError, lambda: special_denom(Poly(1, t), Poly(t**2, t), Poly(1, t), Poly(t**2 - 1, t),
    Poly(t, t), DE, case='unrecognized_case'))


def test_bound_degree_fail():
    # Primitive
    DE = DifferentialExtension(extension={'D': [Poly(1, x),
        Poly(t0/x**2, t0), Poly(1/x, t)]})
    assert bound_degree(Poly(t**2, t), Poly(-(1/x**2*t**2 + 1/x), t),
        Poly((2*x - 1)*t**4 + (t0 + x)/x*t**3 - (t0 + 4*x**2)/2*x*t**2 + x*t,
        t), DE) == 3


def test_bound_degree():
    # Base
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert bound_degree(Poly(1, x), Poly(-2*x, x), Poly(1, x), DE) == 0

    # Primitive (see above test_bound_degree_fail)
    # TODO: Add test for when the degree bound becomes larger after limited_integrate
    # TODO: Add test for db == da - 1 case

    # Exp
    # TODO: Add tests
    # TODO: Add test for when the degree becomes larger after parametric_log_deriv()

    # Nonlinear
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert bound_degree(Poly(t, t), Poly((t - 1)*(t**2 + 1), t), Poly(1, t), DE) == 0


def test_spde():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    raises(NonElementaryIntegralException, lambda: spde(Poly(t, t), Poly((t - 1)*(t**2 + 1), t), Poly(1, t), 0, DE))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert spde(Poly(t**2 + x*t*2 + x**2, t), Poly(t**2/x**2 + (2/x - 1)*t, t),
        Poly(t**2/x**2 + (2/x - 1)*t, t), 0, DE) == \
        (Poly(0, t), Poly(0, t), 0, Poly(0, t), Poly(1, t, domain='ZZ(x)'))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0/x**2, t0), Poly(1/x, t)]})
    assert spde(Poly(t**2, t), Poly(-t**2/x**2 - 1/x, t),
    Poly((2*x - 1)*t**4 + (t0 + x)/x*t**3 - (t0 + 4*x**2)/(2*x)*t**2 + x*t, t), 3, DE) == \
        (Poly(0, t), Poly(0, t), 0, Poly(0, t),
        Poly(t0*t**2/2 + x**2*t**2 - x**2*t, t, domain='ZZ(x,t0)'))
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert spde(Poly(x**2 + x + 1, x), Poly(-2*x - 1, x), Poly(x**5/2 +
    3*x**4/4 + x**3 - x**2 + 1, x), 4, DE) == \
        (Poly(0, x, domain='QQ'), Poly(x/2 - Rational(1, 4), x), 2, Poly(x**2 + x + 1, x), Poly(x*Rational(5, 4), x))
    assert spde(Poly(x**2 + x + 1, x), Poly(-2*x - 1, x), Poly(x**5/2 +
    3*x**4/4 + x**3 - x**2 + 1, x), n, DE) == \
        (Poly(0, x, domain='QQ'), Poly(x/2 - Rational(1, 4), x), -2 + n, Poly(x**2 + x + 1, x), Poly(x*Rational(5, 4), x))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1, t)]})
    raises(NonElementaryIntegralException, lambda: spde(Poly((t - 1)*(t**2 + 1)**2, t), Poly((t - 1)*(t**2 + 1), t), Poly(1, t), 0, DE))
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert spde(Poly(x**2 - x, x), Poly(1, x), Poly(9*x**4 - 10*x**3 + 2*x**2, x), 4, DE) == \
        (Poly(0, x, domain='ZZ'), Poly(0, x), 0, Poly(0, x), Poly(3*x**3 - 2*x**2, x, domain='QQ'))
    assert spde(Poly(x**2 - x, x), Poly(x**2 - 5*x + 3, x), Poly(x**7 - x**6 - 2*x**4 + 3*x**3 - x**2, x), 5, DE) == \
        (Poly(1, x, domain='QQ'), Poly(x + 1, x, domain='QQ'), 1, Poly(x**4 - x**3, x), Poly(x**3 - x**2, x, domain='QQ'))

def test_solve_poly_rde_no_cancel():
    # deg(b) large
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    assert solve_poly_rde(Poly(t**2 + 1, t), Poly(t**3 + (x + 1)*t**2 + t + x + 2, t),
    oo, DE) == Poly(t + x, t)
    # deg(b) small
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert solve_poly_rde(Poly(0, x), Poly(x/2 - Rational(1, 4), x), oo, DE) == \
        Poly(x**2/4 - x/4, x)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert solve_poly_rde(Poly(2, t), Poly(t**2 + 2*t + 3, t), 1, DE) == \
        Poly(t + 1, t, x)
    # deg(b) == deg(D) - 1
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert no_cancel_equal(Poly(1 - t, t),
    Poly(t**3 + t**2 - 2*x*t - 2*x, t), oo, DE) == \
        (Poly(t**2, t), 1, Poly((-2 - 2*x)*t - 2*x, t))


def test_solve_poly_rde_cancel():
    # exp
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert cancel_exp(Poly(2*x, t), Poly(2*x, t), 0, DE) == \
        Poly(1, t)
    assert cancel_exp(Poly(2*x, t), Poly((1 + 2*x)*t, t), 1, DE) == \
        Poly(t, t)
    # TODO: Add more exp tests, including tests that require is_deriv_in_field()

    # primitive
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})

    # If the DecrementLevel context manager is working correctly, this shouldn't
    # cause any problems with the further tests.
    raises(NonElementaryIntegralException, lambda: cancel_primitive(Poly(1, t), Poly(t, t), oo, DE))

    assert cancel_primitive(Poly(1, t), Poly(t + 1/x, t), 2, DE) == \
        Poly(t, t)
    assert cancel_primitive(Poly(4*x, t), Poly(4*x*t**2 + 2*t/x, t), 3, DE) == \
        Poly(t**2, t)

    # TODO: Add more primitive tests, including tests that require is_deriv_in_field()


def test_rischDE():
    # TODO: Add more tests for rischDE, including ones from the text
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    DE.decrement_level()
    assert rischDE(Poly(-2*x, x), Poly(1, x), Poly(1 - 2*x - 2*x**2, x),
    Poly(1, x), DE) == \
        (Poly(x + 1, x), Poly(1, x))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\tests\test_utils.py ===
from sympy.core.numbers import comp, Rational
from sympy.physics.optics.utils import (refraction_angle, fresnel_coefficients,
        deviation, brewster_angle, critical_angle, lens_makers_formula,
        mirror_formula, lens_formula, hyperfocal_distance,
        transverse_magnification)
from sympy.physics.optics.medium import Medium
from sympy.physics.units import e0

from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.geometry.point import Point3D
from sympy.geometry.line import Ray3D
from sympy.geometry.plane import Plane

from sympy.testing.pytest import raises


ae = lambda a, b, n: comp(a, b, 10**-n)


def test_refraction_angle():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1')
    m2 = Medium('m2')
    r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    i = Matrix([1, 1, 1])
    n = Matrix([0, 0, 1])
    normal_ray = Ray3D(Point3D(0, 0, 0), Point3D(0, 0, 1))
    P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    assert refraction_angle(r1, 1, 1, n) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle([1, 1, 1], 1, 1, n) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle((1, 1, 1), 1, 1, n) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, [0, 0, 1]) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, (0, 0, 1)) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, normal_ray) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, plane=P) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(r1, 1, 1, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))
    assert refraction_angle(r1, m1, 1.33, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(Rational(100, 133), Rational(100, 133), -789378201649271*sqrt(3)/1000000000000000))
    assert refraction_angle(r1, 1, m2, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))
    assert refraction_angle(r1, n1, n2, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(n1/n2, n1/n2, -sqrt(3)*sqrt(-2*n1**2/(3*n2**2) + 1)))
    assert refraction_angle(r1, 1.33, 1, plane=P) == 0  # TIR
    assert refraction_angle(r1, 1, 1, normal_ray) == \
        Ray3D(Point3D(0, 0, 0), direction_ratio=[1, 1, -1])
    assert ae(refraction_angle(0.5, 1, 2), 0.24207, 5)
    assert ae(refraction_angle(0.5, 2, 1), 1.28293, 5)
    raises(ValueError, lambda: refraction_angle(r1, m1, m2, normal_ray, P))
    raises(TypeError, lambda: refraction_angle(m1, m1, m2)) # can add other values for arg[0]
    raises(TypeError, lambda: refraction_angle(r1, m1, m2, None, i))
    raises(TypeError, lambda: refraction_angle(r1, m1, m2, m2))


def test_fresnel_coefficients():
    assert all(ae(i, j, 5) for i, j in zip(
        fresnel_coefficients(0.5, 1, 1.33),
        [0.11163, -0.17138, 0.83581, 0.82862]))
    assert all(ae(i, j, 5) for i, j in zip(
        fresnel_coefficients(0.5, 1.33, 1),
        [-0.07726, 0.20482, 1.22724, 1.20482]))
    m1 = Medium('m1')
    m2 = Medium('m2', n=2)
    assert all(ae(i, j, 5) for i, j in zip(
        fresnel_coefficients(0.3, m1, m2),
        [0.31784, -0.34865, 0.65892, 0.65135]))
    ans = [[-0.23563, -0.97184], [0.81648, -0.57738]]
    got = fresnel_coefficients(0.6, m2, m1)
    for i, j in zip(got, ans):
        for a, b in zip(i.as_real_imag(), j):
            assert ae(a, b, 5)


def test_deviation():
    n1, n2 = symbols('n1, n2')
    r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    n = Matrix([0, 0, 1])
    i = Matrix([-1, -1, -1])
    normal_ray = Ray3D(Point3D(0, 0, 0), Point3D(0, 0, 1))
    P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    assert deviation(r1, 1, 1, normal=n) == 0
    assert deviation(r1, 1, 1, plane=P) == 0
    assert deviation(r1, 1, 1.1, plane=P).evalf(3) + 0.119 < 1e-3
    assert deviation(i, 1, 1.1, normal=normal_ray).evalf(3) + 0.119 < 1e-3
    assert deviation(r1, 1.33, 1, plane=P) is None  # TIR
    assert deviation(r1, 1, 1, normal=[0, 0, 1]) == 0
    assert deviation([-1, -1, -1], 1, 1, normal=[0, 0, 1]) == 0
    assert ae(deviation(0.5, 1, 2), -0.25793, 5)
    assert ae(deviation(0.5, 2, 1), 0.78293, 5)


def test_brewster_angle():
    m1 = Medium('m1', n=1)
    m2 = Medium('m2', n=1.33)
    assert ae(brewster_angle(m1, m2), 0.93, 2)
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    assert ae(brewster_angle(m1, m2), 0.93, 2)
    assert ae(brewster_angle(1, 1.33), 0.93, 2)


def test_critical_angle():
    m1 = Medium('m1', n=1)
    m2 = Medium('m2', n=1.33)
    assert ae(critical_angle(m2, m1), 0.85, 2)


def test_lens_makers_formula():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    assert lens_makers_formula(n1, n2, 10, -10) == 5.0*n2/(n1 - n2)
    assert ae(lens_makers_formula(m1, m2, 10, -10), -20.15, 2)
    assert ae(lens_makers_formula(1.33, 1, 10, -10),  15.15, 2)


def test_mirror_formula():
    u, v, f = symbols('u, v, f')
    assert mirror_formula(focal_length=f, u=u) == f*u/(-f + u)
    assert mirror_formula(focal_length=f, v=v) == f*v/(-f + v)
    assert mirror_formula(u=u, v=v) == u*v/(u + v)
    assert mirror_formula(u=oo, v=v) == v
    assert mirror_formula(u=oo, v=oo) is oo
    assert mirror_formula(focal_length=oo, u=u) == -u
    assert mirror_formula(u=u, v=oo) == u
    assert mirror_formula(focal_length=oo, v=oo) is oo
    assert mirror_formula(focal_length=f, v=oo) == f
    assert mirror_formula(focal_length=oo, v=v) == -v
    assert mirror_formula(focal_length=oo, u=oo) is oo
    assert mirror_formula(focal_length=f, u=oo) == f
    assert mirror_formula(focal_length=oo, u=u) == -u
    raises(ValueError, lambda: mirror_formula(focal_length=f, u=u, v=v))


def test_lens_formula():
    u, v, f = symbols('u, v, f')
    assert lens_formula(focal_length=f, u=u) == f*u/(f + u)
    assert lens_formula(focal_length=f, v=v) == f*v/(f - v)
    assert lens_formula(u=u, v=v) == u*v/(u - v)
    assert lens_formula(u=oo, v=v) == v
    assert lens_formula(u=oo, v=oo) is oo
    assert lens_formula(focal_length=oo, u=u) == u
    assert lens_formula(u=u, v=oo) == -u
    assert lens_formula(focal_length=oo, v=oo) is -oo
    assert lens_formula(focal_length=oo, v=v) == v
    assert lens_formula(focal_length=f, v=oo) == -f
    assert lens_formula(focal_length=oo, u=oo) is oo
    assert lens_formula(focal_length=oo, u=u) == u
    assert lens_formula(focal_length=f, u=oo) == f
    raises(ValueError, lambda: lens_formula(focal_length=f, u=u, v=v))


def test_hyperfocal_distance():
    f, N, c = symbols('f, N, c')
    assert hyperfocal_distance(f=f, N=N, c=c) == f**2/(N*c)
    assert ae(hyperfocal_distance(f=0.5, N=8, c=0.0033), 9.47, 2)


def test_transverse_magnification():
    si, so = symbols('si, so')
    assert transverse_magnification(si, so) == -si/so
    assert transverse_magnification(30, 15) == -2


def test_lens_makers_formula_thick_lens():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    assert ae(lens_makers_formula(m1, m2, 10, -10, d=1), -19.82, 2)
    assert lens_makers_formula(n1, n2, 1, -1, d=0.1) == n2/((2.0 - (0.1*n1 - 0.1*n2)/n1)*(n1 - n2))


def test_lens_makers_formula_plano_lens():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    assert ae(lens_makers_formula(m1, m2, 10, oo), -40.30, 2)
    assert lens_makers_formula(n1, n2, 10, oo) == 10.0*n2/(n1 - n2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\tests\test_numbers.py ===
"""Tests on algebraic numbers. """

from sympy.core.containers import Tuple
from sympy.core.numbers import (AlgebraicNumber, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import Poly
from sympy.polys.numberfields.subfield import to_number_field
from sympy.polys.polyclasses import DMP
from sympy.polys.domains import QQ
from sympy.polys.rootoftools import CRootOf
from sympy.abc import x, y


def test_AlgebraicNumber():
    minpoly, root = x**2 - 2, sqrt(2)

    a = AlgebraicNumber(root, gen=x)

    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is False

    assert a.coeffs() == [S.One, S.Zero]
    assert a.native_coeffs() == [QQ(1), QQ(0)]

    a = AlgebraicNumber(root, gen=x, alias='y')

    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    assert a.root == root
    assert a.alias == Symbol('y')
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is True

    a = AlgebraicNumber(root, gen=x, alias=Symbol('y'))

    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    assert a.root == root
    assert a.alias == Symbol('y')
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is True

    assert AlgebraicNumber(sqrt(2), []).rep == DMP([], QQ)
    assert AlgebraicNumber(sqrt(2), ()).rep == DMP([], QQ)
    assert AlgebraicNumber(sqrt(2), (0, 0)).rep == DMP([], QQ)

    assert AlgebraicNumber(sqrt(2), [8]).rep == DMP([QQ(8)], QQ)
    assert AlgebraicNumber(sqrt(2), [Rational(8, 3)]).rep == DMP([QQ(8, 3)], QQ)

    assert AlgebraicNumber(sqrt(2), [7, 3]).rep == DMP([QQ(7), QQ(3)], QQ)
    assert AlgebraicNumber(
        sqrt(2), [Rational(7, 9), Rational(3, 2)]).rep == DMP([QQ(7, 9), QQ(3, 2)], QQ)

    assert AlgebraicNumber(sqrt(2), [1, 2, 3]).rep == DMP([QQ(2), QQ(5)], QQ)

    a = AlgebraicNumber(AlgebraicNumber(root, gen=x), [1, 2])

    assert a.rep == DMP([QQ(1), QQ(2)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is False

    assert a.coeffs() == [S.One, S(2)]
    assert a.native_coeffs() == [QQ(1), QQ(2)]

    a = AlgebraicNumber((minpoly, root), [1, 2])

    assert a.rep == DMP([QQ(1), QQ(2)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is False

    a = AlgebraicNumber((Poly(minpoly), root), [1, 2])

    assert a.rep == DMP([QQ(1), QQ(2)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    assert a.is_aliased is False

    assert AlgebraicNumber( sqrt(3)).rep == DMP([ QQ(1), QQ(0)], QQ)
    assert AlgebraicNumber(-sqrt(3)).rep == DMP([ QQ(1), QQ(0)], QQ)

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(2))

    assert a == b

    c = AlgebraicNumber(sqrt(2), gen=x)

    assert a == b
    assert a == c

    a = AlgebraicNumber(sqrt(2), [1, 2])
    b = AlgebraicNumber(sqrt(2), [1, 3])

    assert a != b and a != sqrt(2) + 3

    assert (a == x) is False and (a != x) is True

    a = AlgebraicNumber(sqrt(2), [1, 0])
    b = AlgebraicNumber(sqrt(2), [1, 0], alias=y)

    assert a.as_poly(x) == Poly(x, domain='QQ')
    assert b.as_poly() == Poly(y, domain='QQ')

    assert a.as_expr() == sqrt(2)
    assert a.as_expr(x) == x
    assert b.as_expr() == sqrt(2)
    assert b.as_expr(x) == x

    a = AlgebraicNumber(sqrt(2), [2, 3])
    b = AlgebraicNumber(sqrt(2), [2, 3], alias=y)

    p = a.as_poly()

    assert p == Poly(2*p.gen + 3)

    assert a.as_poly(x) == Poly(2*x + 3, domain='QQ')
    assert b.as_poly() == Poly(2*y + 3, domain='QQ')

    assert a.as_expr() == 2*sqrt(2) + 3
    assert a.as_expr(x) == 2*x + 3
    assert b.as_expr() == 2*sqrt(2) + 3
    assert b.as_expr(x) == 2*x + 3

    a = AlgebraicNumber(sqrt(2))
    b = to_number_field(sqrt(2))
    assert a.args == b.args == (sqrt(2), Tuple(1, 0))
    b = AlgebraicNumber(sqrt(2), alias='alpha')
    assert b.args == (sqrt(2), Tuple(1, 0), Symbol('alpha'))

    a = AlgebraicNumber(sqrt(2), [1, 2, 3])
    assert a.args == (sqrt(2), Tuple(1, 2, 3))

    a = AlgebraicNumber(sqrt(2), [1, 2], "alpha")
    b = AlgebraicNumber(a)
    c = AlgebraicNumber(a, alias="gamma")
    assert a == b
    assert c.alias.name == "gamma"

    a = AlgebraicNumber(sqrt(2) + sqrt(3), [S(1)/2, 0, S(-9)/2, 0])
    b = AlgebraicNumber(a, [1, 0, 0])
    assert b.root == a.root
    assert a.to_root() == sqrt(2)
    assert b.to_root() == 2

    a = AlgebraicNumber(2)
    assert a.is_primitive_element is True


def test_to_algebraic_integer():
    a = AlgebraicNumber(sqrt(3), gen=x).to_algebraic_integer()

    assert a.minpoly == x**2 - 3
    assert a.root == sqrt(3)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)

    a = AlgebraicNumber(2*sqrt(3), gen=x).to_algebraic_integer()
    assert a.minpoly == x**2 - 12
    assert a.root == 2*sqrt(3)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)

    a = AlgebraicNumber(sqrt(3)/2, gen=x).to_algebraic_integer()

    assert a.minpoly == x**2 - 12
    assert a.root == 2*sqrt(3)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)

    a = AlgebraicNumber(sqrt(3)/2, [Rational(7, 19), 3], gen=x).to_algebraic_integer()

    assert a.minpoly == x**2 - 12
    assert a.root == 2*sqrt(3)
    assert a.rep == DMP([QQ(7, 19), QQ(3)], QQ)


def test_AlgebraicNumber_to_root():
    assert AlgebraicNumber(sqrt(2)).to_root() == sqrt(2)

    zeta5_squared = AlgebraicNumber(CRootOf(x**5 - 1, 4), coeffs=[1, 0, 0])
    assert zeta5_squared.to_root() == CRootOf(x**4 + x**3 + x**2 + x + 1, 1)

    zeta3_squared = AlgebraicNumber(CRootOf(x**3 - 1, 2), coeffs=[1, 0, 0])
    assert zeta3_squared.to_root() == -S(1)/2 - sqrt(3)*I/2
    assert zeta3_squared.to_root(radicals=False) == CRootOf(x**2 + x + 1, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_conversion.py ===
import numpy as np
import pytest

from pandas.compat.numpy import np_version_gt2

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
)
import pandas._testing as tm


def test_to_numpy(idx):
    result = idx.to_numpy()
    exp = idx.values
    tm.assert_numpy_array_equal(result, exp)


def test_array_interface(idx):
    # https://github.com/pandas-dev/pandas/pull/60046
    result = np.asarray(idx)
    expected = np.empty((6,), dtype=object)
    expected[:] = [
        ("foo", "one"),
        ("foo", "two"),
        ("bar", "one"),
        ("baz", "two"),
        ("qux", "one"),
        ("qux", "two"),
    ]
    tm.assert_numpy_array_equal(result, expected)

    # it always gives a copy by default, but the values are cached, so results
    # are still sharing memory
    result_copy1 = np.asarray(idx)
    result_copy2 = np.asarray(idx)
    assert np.may_share_memory(result_copy1, result_copy2)

    # with explicit copy=True, then it is an actual copy
    result_copy1 = np.array(idx, copy=True)
    result_copy2 = np.array(idx, copy=True)
    assert not np.may_share_memory(result_copy1, result_copy2)

    if not np_version_gt2:
        # copy=False semantics are only supported in NumPy>=2.
        return

    # for MultiIndex, copy=False is never allowed
    msg = "Starting with NumPy 2.0, the behavior of the 'copy' keyword has changed"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        np.array(idx, copy=False)


def test_to_frame():
    tuples = [(1, "one"), (1, "two"), (2, "one"), (2, "two")]

    index = MultiIndex.from_tuples(tuples)
    result = index.to_frame(index=False)
    expected = DataFrame(tuples)
    tm.assert_frame_equal(result, expected)

    result = index.to_frame()
    expected.index = index
    tm.assert_frame_equal(result, expected)

    tuples = [(1, "one"), (1, "two"), (2, "one"), (2, "two")]
    index = MultiIndex.from_tuples(tuples, names=["first", "second"])
    result = index.to_frame(index=False)
    expected = DataFrame(tuples)
    expected.columns = ["first", "second"]
    tm.assert_frame_equal(result, expected)

    result = index.to_frame()
    expected.index = index
    tm.assert_frame_equal(result, expected)

    # See GH-22580
    index = MultiIndex.from_tuples(tuples)
    result = index.to_frame(index=False, name=["first", "second"])
    expected = DataFrame(tuples)
    expected.columns = ["first", "second"]
    tm.assert_frame_equal(result, expected)

    result = index.to_frame(name=["first", "second"])
    expected.index = index
    expected.columns = ["first", "second"]
    tm.assert_frame_equal(result, expected)

    msg = "'name' must be a list / sequence of column names."
    with pytest.raises(TypeError, match=msg):
        index.to_frame(name="first")

    msg = "'name' should have same length as number of levels on index."
    with pytest.raises(ValueError, match=msg):
        index.to_frame(name=["first"])

    # Tests for datetime index
    index = MultiIndex.from_product([range(5), pd.date_range("20130101", periods=3)])
    result = index.to_frame(index=False)
    expected = DataFrame(
        {
            0: np.repeat(np.arange(5, dtype="int64"), 3),
            1: np.tile(pd.date_range("20130101", periods=3), 5),
        }
    )
    tm.assert_frame_equal(result, expected)

    result = index.to_frame()
    expected.index = index
    tm.assert_frame_equal(result, expected)

    # See GH-22580
    result = index.to_frame(index=False, name=["first", "second"])
    expected = DataFrame(
        {
            "first": np.repeat(np.arange(5, dtype="int64"), 3),
            "second": np.tile(pd.date_range("20130101", periods=3), 5),
        }
    )
    tm.assert_frame_equal(result, expected)

    result = index.to_frame(name=["first", "second"])
    expected.index = index
    tm.assert_frame_equal(result, expected)


def test_to_frame_dtype_fidelity():
    # GH 22420
    mi = MultiIndex.from_arrays(
        [
            pd.date_range("19910905", periods=6, tz="US/Eastern"),
            [1, 1, 1, 2, 2, 2],
            pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),
            ["x", "x", "y", "z", "x", "y"],
        ],
        names=["dates", "a", "b", "c"],
    )
    original_dtypes = {name: mi.levels[i].dtype for i, name in enumerate(mi.names)}

    expected_df = DataFrame(
        {
            "dates": pd.date_range("19910905", periods=6, tz="US/Eastern"),
            "a": [1, 1, 1, 2, 2, 2],
            "b": pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),
            "c": ["x", "x", "y", "z", "x", "y"],
        }
    )
    df = mi.to_frame(index=False)
    df_dtypes = df.dtypes.to_dict()

    tm.assert_frame_equal(df, expected_df)
    assert original_dtypes == df_dtypes


def test_to_frame_resulting_column_order():
    # GH 22420
    expected = ["z", 0, "a"]
    mi = MultiIndex.from_arrays(
        [["a", "b", "c"], ["x", "y", "z"], ["q", "w", "e"]], names=expected
    )
    result = mi.to_frame().columns.tolist()
    assert result == expected


def test_to_frame_duplicate_labels():
    # GH 45245
    data = [(1, 2), (3, 4)]
    names = ["a", "a"]
    index = MultiIndex.from_tuples(data, names=names)
    with pytest.raises(ValueError, match="Cannot create duplicate column labels"):
        index.to_frame()

    result = index.to_frame(allow_duplicates=True)
    expected = DataFrame(data, index=index, columns=names)
    tm.assert_frame_equal(result, expected)

    names = [None, 0]
    index = MultiIndex.from_tuples(data, names=names)
    with pytest.raises(ValueError, match="Cannot create duplicate column labels"):
        index.to_frame()

    result = index.to_frame(allow_duplicates=True)
    expected = DataFrame(data, index=index, columns=[0, 0])
    tm.assert_frame_equal(result, expected)


def test_to_flat_index(idx):
    expected = pd.Index(
        (
            ("foo", "one"),
            ("foo", "two"),
            ("bar", "one"),
            ("baz", "two"),
            ("qux", "one"),
            ("qux", "two"),
        ),
        tupleize_cols=False,
    )
    result = idx.to_flat_index()
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_names.py ===
import pytest

import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm


def check_level_names(index, names):
    assert [level.name for level in index.levels] == list(names)


def test_slice_keep_name():
    x = MultiIndex.from_tuples([("a", "b"), (1, 2), ("c", "d")], names=["x", "y"])
    assert x[1:].names == x.names


def test_index_name_retained():
    # GH9857
    result = pd.DataFrame({"x": [1, 2, 6], "y": [2, 2, 8], "z": [-5, 0, 5]})
    result = result.set_index("z")
    result.loc[10] = [9, 10]
    df_expected = pd.DataFrame(
        {"x": [1, 2, 6, 9], "y": [2, 2, 8, 10], "z": [-5, 0, 5, 10]}
    )
    df_expected = df_expected.set_index("z")
    tm.assert_frame_equal(result, df_expected)


def test_changing_names(idx):
    assert [level.name for level in idx.levels] == ["first", "second"]

    view = idx.view()
    copy = idx.copy()
    shallow_copy = idx._view()

    # changing names should not change level names on object
    new_names = [name + "a" for name in idx.names]
    idx.names = new_names
    check_level_names(idx, ["firsta", "seconda"])

    # and not on copies
    check_level_names(view, ["first", "second"])
    check_level_names(copy, ["first", "second"])
    check_level_names(shallow_copy, ["first", "second"])

    # and copies shouldn't change original
    shallow_copy.names = [name + "c" for name in shallow_copy.names]
    check_level_names(idx, ["firsta", "seconda"])


def test_take_preserve_name(idx):
    taken = idx.take([3, 0, 1])
    assert taken.names == idx.names


def test_copy_names():
    # Check that adding a "names" parameter to the copy is honored
    # GH14302
    multi_idx = MultiIndex.from_tuples([(1, 2), (3, 4)], names=["MyName1", "MyName2"])
    multi_idx1 = multi_idx.copy()

    assert multi_idx.equals(multi_idx1)
    assert multi_idx.names == ["MyName1", "MyName2"]
    assert multi_idx1.names == ["MyName1", "MyName2"]

    multi_idx2 = multi_idx.copy(names=["NewName1", "NewName2"])

    assert multi_idx.equals(multi_idx2)
    assert multi_idx.names == ["MyName1", "MyName2"]
    assert multi_idx2.names == ["NewName1", "NewName2"]

    multi_idx3 = multi_idx.copy(name=["NewName1", "NewName2"])

    assert multi_idx.equals(multi_idx3)
    assert multi_idx.names == ["MyName1", "MyName2"]
    assert multi_idx3.names == ["NewName1", "NewName2"]

    # gh-35592
    with pytest.raises(ValueError, match="Length of new names must be 2, got 1"):
        multi_idx.copy(names=["mario"])

    with pytest.raises(TypeError, match="MultiIndex.name must be a hashable type"):
        multi_idx.copy(names=[["mario"], ["luigi"]])


def test_names(idx):
    # names are assigned in setup
    assert idx.names == ["first", "second"]
    level_names = [level.name for level in idx.levels]
    assert level_names == idx.names

    # setting bad names on existing
    index = idx
    with pytest.raises(ValueError, match="^Length of names"):
        setattr(index, "names", list(index.names) + ["third"])
    with pytest.raises(ValueError, match="^Length of names"):
        setattr(index, "names", [])

    # initializing with bad names (should always be equivalent)
    major_axis, minor_axis = idx.levels
    major_codes, minor_codes = idx.codes
    with pytest.raises(ValueError, match="^Length of names"):
        MultiIndex(
            levels=[major_axis, minor_axis],
            codes=[major_codes, minor_codes],
            names=["first"],
        )
    with pytest.raises(ValueError, match="^Length of names"):
        MultiIndex(
            levels=[major_axis, minor_axis],
            codes=[major_codes, minor_codes],
            names=["first", "second", "third"],
        )

    # names are assigned on index, but not transferred to the levels
    index.names = ["a", "b"]
    level_names = [level.name for level in index.levels]
    assert level_names == ["a", "b"]


def test_duplicate_level_names_access_raises(idx):
    # GH19029
    idx.names = ["foo", "foo"]
    with pytest.raises(ValueError, match="name foo occurs multiple times"):
        idx._get_level_number("foo")


def test_get_names_from_levels():
    idx = MultiIndex.from_product([["a"], [1, 2]], names=["a", "b"])

    assert idx.levels[0].name == "a"
    assert idx.levels[1].name == "b"


def test_setting_names_from_levels_raises():
    idx = MultiIndex.from_product([["a"], [1, 2]], names=["a", "b"])
    with pytest.raises(RuntimeError, match="set_names"):
        idx.levels[0].name = "foo"

    with pytest.raises(RuntimeError, match="set_names"):
        idx.levels[1].name = "foo"

    new = pd.Series(1, index=idx.levels[0])
    with pytest.raises(RuntimeError, match="set_names"):
        new.index.name = "bar"

    assert pd.Index._no_setting_name is False
    assert pd.RangeIndex._no_setting_name is False


@pytest.mark.parametrize("func", ["rename", "set_names"])
@pytest.mark.parametrize(
    "rename_dict, exp_names",
    [
        ({"x": "z"}, ["z", "y", "z"]),
        ({"x": "z", "y": "x"}, ["z", "x", "z"]),
        ({"y": "z"}, ["x", "z", "x"]),
        ({}, ["x", "y", "x"]),
        ({"z": "a"}, ["x", "y", "x"]),
        ({"y": "z", "a": "b"}, ["x", "z", "x"]),
    ],
)
def test_name_mi_with_dict_like_duplicate_names(func, rename_dict, exp_names):
    # GH#20421
    mi = MultiIndex.from_arrays([[1, 2], [3, 4], [5, 6]], names=["x", "y", "x"])
    result = getattr(mi, func)(rename_dict)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4], [5, 6]], names=exp_names)
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("func", ["rename", "set_names"])
@pytest.mark.parametrize(
    "rename_dict, exp_names",
    [
        ({"x": "z"}, ["z", "y"]),
        ({"x": "z", "y": "x"}, ["z", "x"]),
        ({"a": "z"}, ["x", "y"]),
        ({}, ["x", "y"]),
    ],
)
def test_name_mi_with_dict_like(func, rename_dict, exp_names):
    # GH#20421
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["x", "y"])
    result = getattr(mi, func)(rename_dict)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]], names=exp_names)
    tm.assert_index_equal(result, expected)


def test_index_name_with_dict_like_raising():
    # GH#20421
    ix = pd.Index([1, 2])
    msg = "Can only pass dict-like as `names` for MultiIndex."
    with pytest.raises(TypeError, match=msg):
        ix.set_names({"x": "z"})


def test_multiindex_name_and_level_raising():
    # GH#20421
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["x", "y"])
    with pytest.raises(TypeError, match="Can not pass level for dictlike `names`."):
        mi.set_names(names={"x": "z"}, level={"x": "z"})

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\test_formats.py ===
from datetime import datetime
import pprint

import dateutil.tz
import pytest
import pytz  # a test below uses pytz but only inside a `eval` call

from pandas import Timestamp

ts_no_ns = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
)
ts_no_ns_year1 = Timestamp(
    year=1,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
)
ts_ns = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
    nanosecond=123,
)
ts_ns_tz = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=132263,
    nanosecond=123,
    tz="UTC",
)
ts_no_us = Timestamp(
    year=2019,
    month=5,
    day=18,
    hour=15,
    minute=17,
    second=8,
    microsecond=0,
    nanosecond=123,
)


@pytest.mark.parametrize(
    "ts, timespec, expected_iso",
    [
        (ts_no_ns, "auto", "2019-05-18T15:17:08.132263"),
        (ts_no_ns, "seconds", "2019-05-18T15:17:08"),
        (ts_no_ns, "nanoseconds", "2019-05-18T15:17:08.132263000"),
        (ts_no_ns_year1, "seconds", "0001-05-18T15:17:08"),
        (ts_no_ns_year1, "nanoseconds", "0001-05-18T15:17:08.132263000"),
        (ts_ns, "auto", "2019-05-18T15:17:08.132263123"),
        (ts_ns, "hours", "2019-05-18T15"),
        (ts_ns, "minutes", "2019-05-18T15:17"),
        (ts_ns, "seconds", "2019-05-18T15:17:08"),
        (ts_ns, "milliseconds", "2019-05-18T15:17:08.132"),
        (ts_ns, "microseconds", "2019-05-18T15:17:08.132263"),
        (ts_ns, "nanoseconds", "2019-05-18T15:17:08.132263123"),
        (ts_ns_tz, "auto", "2019-05-18T15:17:08.132263123+00:00"),
        (ts_ns_tz, "hours", "2019-05-18T15+00:00"),
        (ts_ns_tz, "minutes", "2019-05-18T15:17+00:00"),
        (ts_ns_tz, "seconds", "2019-05-18T15:17:08+00:00"),
        (ts_ns_tz, "milliseconds", "2019-05-18T15:17:08.132+00:00"),
        (ts_ns_tz, "microseconds", "2019-05-18T15:17:08.132263+00:00"),
        (ts_ns_tz, "nanoseconds", "2019-05-18T15:17:08.132263123+00:00"),
        (ts_no_us, "auto", "2019-05-18T15:17:08.000000123"),
    ],
)
def test_isoformat(ts, timespec, expected_iso):
    assert ts.isoformat(timespec=timespec) == expected_iso


class TestTimestampRendering:
    timezones = ["UTC", "Asia/Tokyo", "US/Eastern", "dateutil/America/Los_Angeles"]

    @pytest.mark.parametrize("tz", timezones)
    @pytest.mark.parametrize("freq", ["D", "M", "S", "N"])
    @pytest.mark.parametrize(
        "date", ["2014-03-07", "2014-01-01 09:00", "2014-01-01 00:00:00.000000001"]
    )
    def test_repr(self, date, freq, tz):
        # avoid to match with timezone name
        freq_repr = f"'{freq}'"
        if tz.startswith("dateutil"):
            tz_repr = tz.replace("dateutil", "")
        else:
            tz_repr = tz

        date_only = Timestamp(date)
        assert date in repr(date_only)
        assert tz_repr not in repr(date_only)
        assert freq_repr not in repr(date_only)
        assert date_only == eval(repr(date_only))

        date_tz = Timestamp(date, tz=tz)
        assert date in repr(date_tz)
        assert tz_repr in repr(date_tz)
        assert freq_repr not in repr(date_tz)
        assert date_tz == eval(repr(date_tz))

    def test_repr_utcoffset(self):
        # This can cause the tz field to be populated, but it's redundant to
        # include this information in the date-string.
        date_with_utc_offset = Timestamp("2014-03-13 00:00:00-0400", tz=None)
        assert "2014-03-13 00:00:00-0400" in repr(date_with_utc_offset)
        assert "tzoffset" not in repr(date_with_utc_offset)
        assert "UTC-04:00" in repr(date_with_utc_offset)
        expr = repr(date_with_utc_offset)
        assert date_with_utc_offset == eval(expr)

    def test_timestamp_repr_pre1900(self):
        # pre-1900
        stamp = Timestamp("1850-01-01", tz="US/Eastern")
        repr(stamp)

        iso8601 = "1850-01-01 01:23:45.012345"
        stamp = Timestamp(iso8601, tz="US/Eastern")
        result = repr(stamp)
        assert iso8601 in result

    def test_pprint(self):
        # GH#12622
        nested_obj = {"foo": 1, "bar": [{"w": {"a": Timestamp("2011-01-01")}}] * 10}
        result = pprint.pformat(nested_obj, width=50)
        expected = r"""{'bar': [{'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
         {'w': {'a': Timestamp('2011-01-01 00:00:00')}}],
 'foo': 1}"""
        assert result == expected

    def test_to_timestamp_repr_is_code(self):
        zs = [
            Timestamp("99-04-17 00:00:00", tz="UTC"),
            Timestamp("2001-04-17 00:00:00", tz="UTC"),
            Timestamp("2001-04-17 00:00:00", tz="America/Los_Angeles"),
            Timestamp("2001-04-17 00:00:00", tz=None),
        ]
        for z in zs:
            assert eval(repr(z)) == z

    def test_repr_matches_pydatetime_no_tz(self):
        dt_date = datetime(2013, 1, 2)
        assert str(dt_date) == str(Timestamp(dt_date))

        dt_datetime = datetime(2013, 1, 2, 12, 1, 3)
        assert str(dt_datetime) == str(Timestamp(dt_datetime))

        dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45)
        assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))

        ts_nanos_only = Timestamp(200)
        assert str(ts_nanos_only) == "1970-01-01 00:00:00.000000200"

        ts_nanos_micros = Timestamp(1200)
        assert str(ts_nanos_micros) == "1970-01-01 00:00:00.000001200"

    def test_repr_matches_pydatetime_tz_pytz(self):
        dt_date = datetime(2013, 1, 2, tzinfo=pytz.utc)
        assert str(dt_date) == str(Timestamp(dt_date))

        dt_datetime = datetime(2013, 1, 2, 12, 1, 3, tzinfo=pytz.utc)
        assert str(dt_datetime) == str(Timestamp(dt_datetime))

        dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45, tzinfo=pytz.utc)
        assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))

    def test_repr_matches_pydatetime_tz_dateutil(self):
        utc = dateutil.tz.tzutc()

        dt_date = datetime(2013, 1, 2, tzinfo=utc)
        assert str(dt_date) == str(Timestamp(dt_date))

        dt_datetime = datetime(2013, 1, 2, 12, 1, 3, tzinfo=utc)
        assert str(dt_datetime) == str(Timestamp(dt_datetime))

        dt_datetime_us = datetime(2013, 1, 2, 12, 1, 3, 45, tzinfo=utc)
        assert str(dt_datetime_us) == str(Timestamp(dt_datetime_us))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_print.py ===
import sys
from io import StringIO

import pytest

import numpy as np
from numpy._core.tests._locales import CommaDecimalPointLocale
from numpy.testing import IS_MUSL, assert_, assert_equal

_REF = {np.inf: 'inf', -np.inf: '-inf', np.nan: 'nan'}


@pytest.mark.parametrize('tp', [np.float32, np.double, np.longdouble])
def test_float_types(tp):
    """ Check formatting.

        This is only for the str function, and only for simple types.
        The precision of np.float32 and np.longdouble aren't the same as the
        python float precision.

    """
    for x in [0, 1, -1, 1e20]:
        assert_equal(str(tp(x)), str(float(x)),
                     err_msg=f'Failed str formatting for type {tp}')

    if tp(1e16).itemsize > 4:
        assert_equal(str(tp(1e16)), str(float('1e16')),
                     err_msg=f'Failed str formatting for type {tp}')
    else:
        ref = '1e+16'
        assert_equal(str(tp(1e16)), ref,
                     err_msg=f'Failed str formatting for type {tp}')


@pytest.mark.parametrize('tp', [np.float32, np.double, np.longdouble])
def test_nan_inf_float(tp):
    """ Check formatting of nan & inf.

        This is only for the str function, and only for simple types.
        The precision of np.float32 and np.longdouble aren't the same as the
        python float precision.

    """
    for x in [np.inf, -np.inf, np.nan]:
        assert_equal(str(tp(x)), _REF[x],
                     err_msg=f'Failed str formatting for type {tp}')


@pytest.mark.parametrize('tp', [np.complex64, np.cdouble, np.clongdouble])
def test_complex_types(tp):
    """Check formatting of complex types.

        This is only for the str function, and only for simple types.
        The precision of np.float32 and np.longdouble aren't the same as the
        python float precision.

    """
    for x in [0, 1, -1, 1e20]:
        assert_equal(str(tp(x)), str(complex(x)),
                     err_msg=f'Failed str formatting for type {tp}')
        assert_equal(str(tp(x * 1j)), str(complex(x * 1j)),
                     err_msg=f'Failed str formatting for type {tp}')
        assert_equal(str(tp(x + x * 1j)), str(complex(x + x * 1j)),
                     err_msg=f'Failed str formatting for type {tp}')

    if tp(1e16).itemsize > 8:
        assert_equal(str(tp(1e16)), str(complex(1e16)),
                     err_msg=f'Failed str formatting for type {tp}')
    else:
        ref = '(1e+16+0j)'
        assert_equal(str(tp(1e16)), ref,
                     err_msg=f'Failed str formatting for type {tp}')


@pytest.mark.parametrize('dtype', [np.complex64, np.cdouble, np.clongdouble])
def test_complex_inf_nan(dtype):
    """Check inf/nan formatting of complex types."""
    TESTS = {
        complex(np.inf, 0): "(inf+0j)",
        complex(0, np.inf): "infj",
        complex(-np.inf, 0): "(-inf+0j)",
        complex(0, -np.inf): "-infj",
        complex(np.inf, 1): "(inf+1j)",
        complex(1, np.inf): "(1+infj)",
        complex(-np.inf, 1): "(-inf+1j)",
        complex(1, -np.inf): "(1-infj)",
        complex(np.nan, 0): "(nan+0j)",
        complex(0, np.nan): "nanj",
        complex(-np.nan, 0): "(nan+0j)",
        complex(0, -np.nan): "nanj",
        complex(np.nan, 1): "(nan+1j)",
        complex(1, np.nan): "(1+nanj)",
        complex(-np.nan, 1): "(nan+1j)",
        complex(1, -np.nan): "(1+nanj)",
    }
    for c, s in TESTS.items():
        assert_equal(str(dtype(c)), s)


# print tests
def _test_redirected_print(x, tp, ref=None):
    file = StringIO()
    file_tp = StringIO()
    stdout = sys.stdout
    try:
        sys.stdout = file_tp
        print(tp(x))
        sys.stdout = file
        if ref:
            print(ref)
        else:
            print(x)
    finally:
        sys.stdout = stdout

    assert_equal(file.getvalue(), file_tp.getvalue(),
                 err_msg=f'print failed for type{tp}')


@pytest.mark.parametrize('tp', [np.float32, np.double, np.longdouble])
def test_float_type_print(tp):
    """Check formatting when using print """
    for x in [0, 1, -1, 1e20]:
        _test_redirected_print(float(x), tp)

    for x in [np.inf, -np.inf, np.nan]:
        _test_redirected_print(float(x), tp, _REF[x])

    if tp(1e16).itemsize > 4:
        _test_redirected_print(1e16, tp)
    else:
        ref = '1e+16'
        _test_redirected_print(1e16, tp, ref)


@pytest.mark.parametrize('tp', [np.complex64, np.cdouble, np.clongdouble])
def test_complex_type_print(tp):
    """Check formatting when using print """
    # We do not create complex with inf/nan directly because the feature is
    # missing in python < 2.6
    for x in [0, 1, -1, 1e20]:
        _test_redirected_print(complex(x), tp)

    if tp(1e16).itemsize > 8:
        _test_redirected_print(complex(1e16), tp)
    else:
        ref = '(1e+16+0j)'
        _test_redirected_print(complex(1e16), tp, ref)

    _test_redirected_print(complex(np.inf, 1), tp, '(inf+1j)')
    _test_redirected_print(complex(-np.inf, 1), tp, '(-inf+1j)')
    _test_redirected_print(complex(-np.nan, 1), tp, '(nan+1j)')


def test_scalar_format():
    """Test the str.format method with NumPy scalar types"""
    tests = [('{0}', True, np.bool),
            ('{0}', False, np.bool),
            ('{0:d}', 130, np.uint8),
            ('{0:d}', 50000, np.uint16),
            ('{0:d}', 3000000000, np.uint32),
            ('{0:d}', 15000000000000000000, np.uint64),
            ('{0:d}', -120, np.int8),
            ('{0:d}', -30000, np.int16),
            ('{0:d}', -2000000000, np.int32),
            ('{0:d}', -7000000000000000000, np.int64),
            ('{0:g}', 1.5, np.float16),
            ('{0:g}', 1.5, np.float32),
            ('{0:g}', 1.5, np.float64),
            ('{0:g}', 1.5, np.longdouble),
            ('{0:g}', 1.5 + 0.5j, np.complex64),
            ('{0:g}', 1.5 + 0.5j, np.complex128),
            ('{0:g}', 1.5 + 0.5j, np.clongdouble)]

    for (fmat, val, valtype) in tests:
        try:
            assert_equal(fmat.format(val), fmat.format(valtype(val)),
                    f"failed with val {val}, type {valtype}")
        except ValueError as e:
            assert_(False,
               "format raised exception (fmt='%s', val=%s, type=%s, exc='%s')" %
                            (fmat, repr(val), repr(valtype), str(e)))


#
# Locale tests: scalar types formatting should be independent of the locale
#

class TestCommaDecimalPointLocale(CommaDecimalPointLocale):

    def test_locale_single(self):
        assert_equal(str(np.float32(1.2)), str(1.2))

    def test_locale_double(self):
        assert_equal(str(np.double(1.2)), str(1.2))

    @pytest.mark.skipif(IS_MUSL,
                        reason="test flaky on musllinux")
    def test_locale_longdouble(self):
        assert_equal(str(np.longdouble('1.2')), str(1.2))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_categorical.py ===
"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
import string

import numpy as np
import pytest

from pandas._config import using_string_dtype

import pandas as pd
from pandas import Categorical
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tests.extension import base


def make_data():
    while True:
        values = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)
        # ensure we meet the requirements
        # 1. first two not null
        # 2. first and second are different
        if values[0] != values[1]:
            break
    return values


@pytest.fixture
def dtype():
    return CategoricalDtype()


@pytest.fixture
def data():
    """Length-100 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return Categorical(make_data())


@pytest.fixture
def data_missing():
    """Length 2 array with [NA, Valid]"""
    return Categorical([np.nan, "A"])


@pytest.fixture
def data_for_sorting():
    return Categorical(["A", "B", "C"], categories=["C", "A", "B"], ordered=True)


@pytest.fixture
def data_missing_for_sorting():
    return Categorical(["A", None, "B"], categories=["B", "A"], ordered=True)


@pytest.fixture
def data_for_grouping():
    return Categorical(["a", "a", None, None, "b", "b", "a", "c"])


class TestCategorical(base.ExtensionTests):
    @pytest.mark.xfail(reason="Memory usage doesn't match")
    def test_memory_usage(self, data):
        # TODO: Is this deliberate?
        super().test_memory_usage(data)

    def test_contains(self, data, data_missing):
        # GH-37867
        # na value handling in Categorical.__contains__ is deprecated.
        # See base.BaseInterFaceTests.test_contains for more details.

        na_value = data.dtype.na_value
        # ensure data without missing values
        data = data[~data.isna()]

        # first elements are non-missing
        assert data[0] in data
        assert data_missing[0] in data_missing

        # check the presence of na_value
        assert na_value in data_missing
        assert na_value not in data

        # Categoricals can contain other nan-likes than na_value
        for na_value_obj in tm.NULL_OBJECTS:
            if na_value_obj is na_value:
                continue
            assert na_value_obj not in data
            # this section suffers from super method
            if not using_string_dtype():
                assert na_value_obj in data_missing

    def test_empty(self, dtype):
        cls = dtype.construct_array_type()
        result = cls._empty((4,), dtype=dtype)

        assert isinstance(result, cls)
        # the dtype we passed is not initialized, so will not match the
        #  dtype on our result.
        assert result.dtype == CategoricalDtype([])

    @pytest.mark.skip(reason="Backwards compatibility")
    def test_getitem_scalar(self, data):
        # CategoricalDtype.type isn't "correct" since it should
        # be a parent of the elements (object). But don't want
        # to break things by changing.
        super().test_getitem_scalar(data)

    @pytest.mark.xfail(reason="Unobserved categories included")
    def test_value_counts(self, all_data, dropna):
        return super().test_value_counts(all_data, dropna)

    def test_combine_add(self, data_repeated):
        # GH 20825
        # When adding categoricals in combine, result is a string
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 + x2)
        expected = pd.Series(
            [a + b for (a, b) in zip(list(orig_data1), list(orig_data2))]
        )
        tm.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 + x2)
        expected = pd.Series([a + val for a in list(orig_data1)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data, na_action):
        result = data.map(lambda x: x, na_action=na_action)
        tm.assert_extension_array_equal(result, data)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        # frame & scalar
        op_name = all_arithmetic_operators
        if op_name == "__rmod__":
            request.applymarker(
                pytest.mark.xfail(
                    reason="rmod never called when string is first argument"
                )
            )
        super().test_arith_frame_with_scalar(data, op_name)

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        op_name = all_arithmetic_operators
        if op_name == "__rmod__":
            request.applymarker(
                pytest.mark.xfail(
                    reason="rmod never called when string is first argument"
                )
            )
        super().test_arith_series_with_scalar(data, op_name)

    def _compare_other(self, ser: pd.Series, data, op, other):
        op_name = f"__{op.__name__}__"
        if op_name not in ["__eq__", "__ne__"]:
            msg = "Unordered Categoricals can only compare equality or not"
            with pytest.raises(TypeError, match=msg):
                op(data, other)
        else:
            return super()._compare_other(ser, data, op, other)

    @pytest.mark.xfail(reason="Categorical overrides __repr__")
    @pytest.mark.parametrize("size", ["big", "small"])
    def test_array_repr(self, data, size):
        super().test_array_repr(data, size)

    @pytest.mark.xfail(reason="TBD")
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        super().test_groupby_extension_agg(as_index, data_for_grouping)


class Test2DCompat(base.NDArrayBacked2DTests):
    def test_repr_2d(self, data):
        # Categorical __repr__ doesn't include "Categorical", so we need
        #  to special-case
        res = repr(data.reshape(1, -1))
        assert res.count("\nCategories") == 1

        res = repr(data.reshape(-1, 1))
        assert res.count("\nCategories") == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_clip.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestDataFrameClip:
    def test_clip(self, float_frame):
        median = float_frame.median().median()
        original = float_frame.copy()

        double = float_frame.clip(upper=median, lower=median)
        assert not (double.values != median).any()

        # Verify that float_frame was not changed inplace
        assert (float_frame.values == original.values).all()

    def test_inplace_clip(self, float_frame):
        # GH#15388
        median = float_frame.median().median()
        frame_copy = float_frame.copy()

        return_value = frame_copy.clip(upper=median, lower=median, inplace=True)
        assert return_value is None
        assert not (frame_copy.values != median).any()

    def test_dataframe_clip(self):
        # GH#2747
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))

        for lb, ub in [(-1, 1), (1, -1)]:
            clipped_df = df.clip(lb, ub)

            lb, ub = min(lb, ub), max(ub, lb)
            lb_mask = df.values <= lb
            ub_mask = df.values >= ub
            mask = ~lb_mask & ~ub_mask
            assert (clipped_df.values[lb_mask] == lb).all()
            assert (clipped_df.values[ub_mask] == ub).all()
            assert (clipped_df.values[mask] == df.values[mask]).all()

    def test_clip_mixed_numeric(self):
        # clip on mixed integer or floats
        # GH#24162, clipping now preserves numeric types per column
        df = DataFrame({"A": [1, 2, 3], "B": [1.0, np.nan, 3.0]})
        result = df.clip(1, 2)
        expected = DataFrame({"A": [1, 2, 2], "B": [1.0, np.nan, 2.0]})
        tm.assert_frame_equal(result, expected)

        df = DataFrame([[1, 2, 3.4], [3, 4, 5.6]], columns=["foo", "bar", "baz"])
        expected = df.dtypes
        result = df.clip(upper=3).dtypes
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_clip_against_series(self, inplace):
        # GH#6966

        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
        lb = Series(np.random.default_rng(2).standard_normal(1000))
        ub = lb + 1

        original = df.copy()
        clipped_df = df.clip(lb, ub, axis=0, inplace=inplace)

        if inplace:
            clipped_df = df

        for i in range(2):
            lb_mask = original.iloc[:, i] <= lb
            ub_mask = original.iloc[:, i] >= ub
            mask = ~lb_mask & ~ub_mask

            result = clipped_df.loc[lb_mask, i]
            tm.assert_series_equal(result, lb[lb_mask], check_names=False)
            assert result.name == i

            result = clipped_df.loc[ub_mask, i]
            tm.assert_series_equal(result, ub[ub_mask], check_names=False)
            assert result.name == i

            tm.assert_series_equal(clipped_df.loc[mask, i], df.loc[mask, i])

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("lower", [[2, 3, 4], np.asarray([2, 3, 4])])
    @pytest.mark.parametrize(
        "axis,res",
        [
            (0, [[2.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 7.0, 7.0]]),
            (1, [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]]),
        ],
    )
    def test_clip_against_list_like(self, inplace, lower, axis, res):
        # GH#15390
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        original = DataFrame(
            arr, columns=["one", "two", "three"], index=["a", "b", "c"]
        )

        result = original.clip(lower=lower, upper=[5, 6, 7], axis=axis, inplace=inplace)

        expected = DataFrame(res, columns=original.columns, index=original.index)
        if inplace:
            result = original
        tm.assert_frame_equal(result, expected, check_exact=True)

    @pytest.mark.parametrize("axis", [0, 1, None])
    def test_clip_against_frame(self, axis):
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
        lb = DataFrame(np.random.default_rng(2).standard_normal((1000, 2)))
        ub = lb + 1

        clipped_df = df.clip(lb, ub, axis=axis)

        lb_mask = df <= lb
        ub_mask = df >= ub
        mask = ~lb_mask & ~ub_mask

        tm.assert_frame_equal(clipped_df[lb_mask], lb[lb_mask])
        tm.assert_frame_equal(clipped_df[ub_mask], ub[ub_mask])
        tm.assert_frame_equal(clipped_df[mask], df[mask])

    def test_clip_against_unordered_columns(self):
        # GH#20911
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 4)),
            columns=["A", "B", "C", "D"],
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 4)),
            columns=["D", "A", "B", "C"],
        )
        df3 = DataFrame(df2.values - 1, columns=["B", "D", "C", "A"])
        result_upper = df1.clip(lower=0, upper=df2)
        expected_upper = df1.clip(lower=0, upper=df2[df1.columns])
        result_lower = df1.clip(lower=df3, upper=3)
        expected_lower = df1.clip(lower=df3[df1.columns], upper=3)
        result_lower_upper = df1.clip(lower=df3, upper=df2)
        expected_lower_upper = df1.clip(lower=df3[df1.columns], upper=df2[df1.columns])
        tm.assert_frame_equal(result_upper, expected_upper)
        tm.assert_frame_equal(result_lower, expected_lower)
        tm.assert_frame_equal(result_lower_upper, expected_lower_upper)

    def test_clip_with_na_args(self, float_frame):
        """Should process np.nan argument as None"""
        # GH#17276
        tm.assert_frame_equal(float_frame.clip(np.nan), float_frame)
        tm.assert_frame_equal(float_frame.clip(upper=np.nan, lower=np.nan), float_frame)

        # GH#19992 and adjusted in GH#40420
        df = DataFrame({"col_0": [1, 2, 3], "col_1": [4, 5, 6], "col_2": [7, 8, 9]})

        msg = "Downcasting behavior in Series and DataFrame methods 'where'"
        # TODO: avoid this warning here?  seems like we should never be upcasting
        #  in the first place?
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.clip(lower=[4, 5, np.nan], axis=0)
        expected = DataFrame(
            {"col_0": [4, 5, 3], "col_1": [4, 5, 6], "col_2": [7, 8, 9]}
        )
        tm.assert_frame_equal(result, expected)

        result = df.clip(lower=[4, 5, np.nan], axis=1)
        expected = DataFrame(
            {"col_0": [4, 4, 4], "col_1": [5, 5, 6], "col_2": [7, 8, 9]}
        )
        tm.assert_frame_equal(result, expected)

        # GH#40420
        data = {"col_0": [9, -3, 0, -1, 5], "col_1": [-2, -7, 6, 8, -5]}
        df = DataFrame(data)
        t = Series([2, -4, np.nan, 6, 3])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.clip(lower=t, axis=0)
        expected = DataFrame({"col_0": [9, -3, 0, 6, 5], "col_1": [2, -4, 6, 8, 3]})
        tm.assert_frame_equal(result, expected)

    def test_clip_int_data_with_float_bound(self):
        # GH51472
        df = DataFrame({"a": [1, 2, 3]})
        result = df.clip(lower=1.5)
        expected = DataFrame({"a": [1.5, 2.0, 3.0]})
        tm.assert_frame_equal(result, expected)

    def test_clip_with_list_bound(self):
        # GH#54817
        df = DataFrame([1, 5])
        expected = DataFrame([3, 5])
        result = df.clip([3])
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([1, 3])
        result = df.clip(upper=[3])
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\string\test_indexing.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import Index
import pandas._testing as tm


def _isnan(val):
    try:
        return val is not pd.NA and np.isnan(val)
    except TypeError:
        return False


def _equivalent_na(dtype, null):
    if dtype.na_value is pd.NA and null is pd.NA:
        return True
    elif _isnan(dtype.na_value) and _isnan(null):
        return True
    else:
        return False


class TestGetLoc:
    def test_get_loc(self, any_string_dtype):
        index = Index(["a", "b", "c"], dtype=any_string_dtype)
        assert index.get_loc("b") == 1

    def test_get_loc_raises(self, any_string_dtype):
        index = Index(["a", "b", "c"], dtype=any_string_dtype)
        with pytest.raises(KeyError, match="d"):
            index.get_loc("d")

    def test_get_loc_invalid_value(self, any_string_dtype):
        index = Index(["a", "b", "c"], dtype=any_string_dtype)
        with pytest.raises(KeyError, match="1"):
            index.get_loc(1)

    def test_get_loc_non_unique(self, any_string_dtype):
        index = Index(["a", "b", "a"], dtype=any_string_dtype)
        result = index.get_loc("a")
        expected = np.array([True, False, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_get_loc_non_missing(self, any_string_dtype, nulls_fixture):
        index = Index(["a", "b", "c"], dtype=any_string_dtype)
        with pytest.raises(KeyError):
            index.get_loc(nulls_fixture)

    def test_get_loc_missing(self, any_string_dtype, nulls_fixture):
        index = Index(["a", "b", nulls_fixture], dtype=any_string_dtype)
        assert index.get_loc(nulls_fixture) == 2


class TestGetIndexer:
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("pad", [-1, 0, 1, 1]),
            ("backfill", [0, 0, 1, -1]),
        ],
    )
    def test_get_indexer_strings(self, any_string_dtype, method, expected):
        expected = np.array(expected, dtype=np.intp)
        index = Index(["b", "c"], dtype=any_string_dtype)
        actual = index.get_indexer(["a", "b", "c", "d"], method=method)

        tm.assert_numpy_array_equal(actual, expected)

    def test_get_indexer_strings_raises(self, any_string_dtype):
        index = Index(["b", "c"], dtype=any_string_dtype)

        msg = "|".join(
            [
                "operation 'sub' not supported for dtype 'str",
                r"unsupported operand type\(s\) for -: 'str' and 'str'",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            index.get_indexer(["a", "b", "c", "d"], method="nearest")

        with pytest.raises(TypeError, match=msg):
            index.get_indexer(["a", "b", "c", "d"], method="pad", tolerance=2)

        with pytest.raises(TypeError, match=msg):
            index.get_indexer(
                ["a", "b", "c", "d"], method="pad", tolerance=[2, 2, 2, 2]
            )

    @pytest.mark.parametrize("null", [None, np.nan, float("nan"), pd.NA])
    def test_get_indexer_missing(self, any_string_dtype, null, using_infer_string):
        # NaT and Decimal("NaN") from null_fixture are not supported for string dtype
        index = Index(["a", "b", null], dtype=any_string_dtype)
        result = index.get_indexer(["a", null, "c"])
        if using_infer_string:
            expected = np.array([0, 2, -1], dtype=np.intp)
        elif any_string_dtype == "string" and not _equivalent_na(
            any_string_dtype, null
        ):
            expected = np.array([0, -1, -1], dtype=np.intp)
        else:
            expected = np.array([0, 2, -1], dtype=np.intp)

        tm.assert_numpy_array_equal(result, expected)


class TestGetIndexerNonUnique:
    @pytest.mark.parametrize("null", [None, np.nan, float("nan"), pd.NA])
    def test_get_indexer_non_unique_nas(
        self, any_string_dtype, null, using_infer_string
    ):
        index = Index(["a", "b", null], dtype=any_string_dtype)
        indexer, missing = index.get_indexer_non_unique(["a", null])

        if using_infer_string:
            expected_indexer = np.array([0, 2], dtype=np.intp)
            expected_missing = np.array([], dtype=np.intp)
        elif any_string_dtype == "string" and not _equivalent_na(
            any_string_dtype, null
        ):
            expected_indexer = np.array([0, -1], dtype=np.intp)
            expected_missing = np.array([1], dtype=np.intp)
        else:
            expected_indexer = np.array([0, 2], dtype=np.intp)
            expected_missing = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)

        # actually non-unique
        index = Index(["a", null, "b", null], dtype=any_string_dtype)
        indexer, missing = index.get_indexer_non_unique(["a", null])

        if using_infer_string:
            expected_indexer = np.array([0, 1, 3], dtype=np.intp)
        elif any_string_dtype == "string" and not _equivalent_na(
            any_string_dtype, null
        ):
            pass
        else:
            expected_indexer = np.array([0, 1, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)


class TestSliceLocs:
    @pytest.mark.parametrize(
        "in_slice,expected",
        [
            # error: Slice index must be an integer or None
            (pd.IndexSlice[::-1], "yxdcb"),
            (pd.IndexSlice["b":"y":-1], ""),  # type: ignore[misc]
            (pd.IndexSlice["b"::-1], "b"),  # type: ignore[misc]
            (pd.IndexSlice[:"b":-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice[:"y":-1], "y"),  # type: ignore[misc]
            (pd.IndexSlice["y"::-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice["y"::-4], "yb"),  # type: ignore[misc]
            # absent labels
            (pd.IndexSlice[:"a":-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice[:"a":-2], "ydb"),  # type: ignore[misc]
            (pd.IndexSlice["z"::-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice["z"::-3], "yc"),  # type: ignore[misc]
            (pd.IndexSlice["m"::-1], "dcb"),  # type: ignore[misc]
            (pd.IndexSlice[:"m":-1], "yx"),  # type: ignore[misc]
            (pd.IndexSlice["a":"a":-1], ""),  # type: ignore[misc]
            (pd.IndexSlice["z":"z":-1], ""),  # type: ignore[misc]
            (pd.IndexSlice["m":"m":-1], ""),  # type: ignore[misc]
        ],
    )
    def test_slice_locs_negative_step(self, in_slice, expected, any_string_dtype):
        index = Index(list("bcdxy"), dtype=any_string_dtype)

        s_start, s_stop = index.slice_locs(in_slice.start, in_slice.stop, in_slice.step)
        result = index[s_start : s_stop : in_slice.step]
        expected = Index(list(expected), dtype=any_string_dtype)
        tm.assert_index_equal(result, expected)

    def test_slice_locs_negative_step_oob(self, any_string_dtype):
        index = Index(list("bcdxy"), dtype=any_string_dtype)

        result = index[-10:5:1]
        tm.assert_index_equal(result, index)

        result = index[4:-10:-1]
        expected = Index(list("yxdcb"), dtype=any_string_dtype)
        tm.assert_index_equal(result, expected)

    def test_slice_locs_dup(self, any_string_dtype):
        index = Index(["a", "a", "b", "c", "d", "d"], dtype=any_string_dtype)
        assert index.slice_locs("a", "d") == (0, 6)
        assert index.slice_locs(end="d") == (0, 6)
        assert index.slice_locs("a", "c") == (0, 4)
        assert index.slice_locs("b", "d") == (2, 6)

        index2 = index[::-1]
        assert index2.slice_locs("d", "a") == (0, 6)
        assert index2.slice_locs(end="a") == (0, 6)
        assert index2.slice_locs("d", "b") == (0, 4)
        assert index2.slice_locs("c", "a") == (2, 6)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\ndarray_misc.py ===
"""
Tests for miscellaneous (non-magic) ``np.ndarray``/``np.generic`` methods.

More extensive tests are performed for the methods'
function-based counterpart in `../from_numeric.py`.

"""

from __future__ import annotations

import operator
from typing import cast, Any

import numpy as np
import numpy.typing as npt

class SubClass(npt.NDArray[np.float64]): ...
class IntSubClass(npt.NDArray[np.intp]): ...

i4 = np.int32(1)
A: np.ndarray[Any, np.dtype[np.int32]] = np.array([[1]], dtype=np.int32)
B0 = np.empty((), dtype=np.int32).view(SubClass)
B1 = np.empty((1,), dtype=np.int32).view(SubClass)
B2 = np.empty((1, 1), dtype=np.int32).view(SubClass)
B_int0: IntSubClass = np.empty((), dtype=np.intp).view(IntSubClass)
C: np.ndarray[Any, np.dtype[np.int32]] = np.array([0, 1, 2], dtype=np.int32)
D = np.ones(3).view(SubClass)

ctypes_obj = A.ctypes

i4.all()
A.all()
A.all(axis=0)
A.all(keepdims=True)
A.all(out=B0)

i4.any()
A.any()
A.any(axis=0)
A.any(keepdims=True)
A.any(out=B0)

i4.argmax()
A.argmax()
A.argmax(axis=0)
A.argmax(out=B_int0)

i4.argmin()
A.argmin()
A.argmin(axis=0)
A.argmin(out=B_int0)

i4.argsort()
A.argsort()

i4.choose([()])
_choices = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
C.choose(_choices)
C.choose(_choices, out=D)

i4.clip(1)
A.clip(1)
A.clip(None, 1)
A.clip(1, out=B2)
A.clip(None, 1, out=B2)

i4.compress([1])
A.compress([1])
A.compress([1], out=B1)

i4.conj()
A.conj()
B0.conj()

i4.conjugate()
A.conjugate()
B0.conjugate()

i4.cumprod()
A.cumprod()
A.cumprod(out=B1)

i4.cumsum()
A.cumsum()
A.cumsum(out=B1)

i4.max()
A.max()
A.max(axis=0)
A.max(keepdims=True)
A.max(out=B0)

i4.mean()
A.mean()
A.mean(axis=0)
A.mean(keepdims=True)
A.mean(out=B0)

i4.min()
A.min()
A.min(axis=0)
A.min(keepdims=True)
A.min(out=B0)

i4.prod()
A.prod()
A.prod(axis=0)
A.prod(keepdims=True)
A.prod(out=B0)

i4.round()
A.round()
A.round(out=B2)

i4.repeat(1)
A.repeat(1)
B0.repeat(1)

i4.std()
A.std()
A.std(axis=0)
A.std(keepdims=True)
A.std(out=B0.astype(np.float64))

i4.sum()
A.sum()
A.sum(axis=0)
A.sum(keepdims=True)
A.sum(out=B0)

i4.take(0)
A.take(0)
A.take([0])
A.take(0, out=B0)
A.take([0], out=B1)

i4.var()
A.var()
A.var(axis=0)
A.var(keepdims=True)
A.var(out=B0)

A.argpartition([0])

A.diagonal()

A.dot(1)
A.dot(1, out=B2)

A.nonzero()

C.searchsorted(1)

A.trace()
A.trace(out=B0)

void = cast(np.void, np.array(1, dtype=[("f", np.float64)]).take(0))
void.setfield(10, np.float64)

A.item(0)
C.item(0)

A.ravel()
C.ravel()

A.flatten()
C.flatten()

A.reshape(1)
C.reshape(3)

int(np.array(1.0, dtype=np.float64))
int(np.array("1", dtype=np.str_))

float(np.array(1.0, dtype=np.float64))
float(np.array("1", dtype=np.str_))

complex(np.array(1.0, dtype=np.float64))

operator.index(np.array(1, dtype=np.int64))

# this fails on numpy 2.2.1
# https://github.com/scipy/scipy/blob/a755ee77ec47a64849abe42c349936475a6c2f24/scipy/io/arff/tests/test_arffread.py#L41-L44
A_float = np.array([[1, 5], [2, 4], [np.nan, np.nan]])
A_void: npt.NDArray[np.void] = np.empty(3, [("yop", float), ("yap", float)])
A_void["yop"] = A_float[:, 0]
A_void["yap"] = A_float[:, 1]

# deprecated

with np.testing.assert_warns(DeprecationWarning):
    ctypes_obj.get_data()  # type: ignore[deprecated]  # pyright: ignore[reportDeprecated]
with np.testing.assert_warns(DeprecationWarning):
    ctypes_obj.get_shape()  # type: ignore[deprecated]  # pyright: ignore[reportDeprecated]
with np.testing.assert_warns(DeprecationWarning):
    ctypes_obj.get_strides()  # type: ignore[deprecated]  # pyright: ignore[reportDeprecated]
with np.testing.assert_warns(DeprecationWarning):
    ctypes_obj.get_as_parameter()  # type: ignore[deprecated]  # pyright: ignore[reportDeprecated]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_asof.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import IncompatibleFrequency

from pandas import (
    DataFrame,
    Period,
    Series,
    Timestamp,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm


@pytest.fixture
def date_range_frame():
    """
    Fixture for DataFrame of ints with date_range index

    Columns are ['A', 'B'].
    """
    N = 50
    rng = date_range("1/1/1990", periods=N, freq="53s")
    return DataFrame({"A": np.arange(N), "B": np.arange(N)}, index=rng)


class TestFrameAsof:
    def test_basic(self, date_range_frame):
        # Explicitly cast to float to avoid implicit cast when setting np.nan
        df = date_range_frame.astype({"A": "float"})
        N = 50
        df.loc[df.index[15:30], "A"] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")

        result = df.asof(dates)
        assert result.notna().all(1).all()
        lb = df.index[14]
        ub = df.index[30]

        dates = list(dates)

        result = df.asof(dates)
        assert result.notna().all(1).all()

        mask = (result.index >= lb) & (result.index < ub)
        rs = result[mask]
        assert (rs == 14).all(1).all()

    def test_subset(self, date_range_frame):
        N = 10
        # explicitly cast to float to avoid implicit upcast when setting to np.nan
        df = date_range_frame.iloc[:N].copy().astype({"A": "float"})
        df.loc[df.index[4:8], "A"] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")

        # with a subset of A should be the same
        result = df.asof(dates, subset="A")
        expected = df.asof(dates)
        tm.assert_frame_equal(result, expected)

        # same with A/B
        result = df.asof(dates, subset=["A", "B"])
        expected = df.asof(dates)
        tm.assert_frame_equal(result, expected)

        # B gives df.asof
        result = df.asof(dates, subset="B")
        expected = df.resample("25s", closed="right").ffill().reindex(dates)
        expected.iloc[20:] = 9
        # no "missing", so "B" can retain int dtype (df["A"].dtype platform-dependent)
        expected["B"] = expected["B"].astype(df["B"].dtype)

        tm.assert_frame_equal(result, expected)

    def test_missing(self, date_range_frame):
        # GH 15118
        # no match found - `where` value before earliest date in index
        N = 10
        # Cast to 'float64' to avoid upcast when introducing nan in df.asof
        df = date_range_frame.iloc[:N].copy().astype("float64")

        result = df.asof("1989-12-31")

        expected = Series(
            index=["A", "B"], name=Timestamp("1989-12-31"), dtype=np.float64
        )
        tm.assert_series_equal(result, expected)

        result = df.asof(to_datetime(["1989-12-31"]))
        expected = DataFrame(
            index=to_datetime(["1989-12-31"]), columns=["A", "B"], dtype="float64"
        )
        tm.assert_frame_equal(result, expected)

        # Check that we handle PeriodIndex correctly, dont end up with
        #  period.ordinal for series name
        df = df.to_period("D")
        result = df.asof("1989-12-31")
        assert isinstance(result.name, Period)

    def test_asof_all_nans(self, frame_or_series):
        # GH 15713
        # DataFrame/Series is all nans
        result = frame_or_series([np.nan]).asof([0])
        expected = frame_or_series([np.nan])
        tm.assert_equal(result, expected)

    def test_all_nans(self, date_range_frame):
        # GH 15713
        # DataFrame is all nans

        # testing non-default indexes, multiple inputs
        N = 150
        rng = date_range_frame.index
        dates = date_range("1/1/1990", periods=N, freq="25s")
        result = DataFrame(np.nan, index=rng, columns=["A"]).asof(dates)
        expected = DataFrame(np.nan, index=dates, columns=["A"])
        tm.assert_frame_equal(result, expected)

        # testing multiple columns
        dates = date_range("1/1/1990", periods=N, freq="25s")
        result = DataFrame(np.nan, index=rng, columns=["A", "B", "C"]).asof(dates)
        expected = DataFrame(np.nan, index=dates, columns=["A", "B", "C"])
        tm.assert_frame_equal(result, expected)

        # testing scalar input
        result = DataFrame(np.nan, index=[1, 2], columns=["A", "B"]).asof([3])
        expected = DataFrame(np.nan, index=[3], columns=["A", "B"])
        tm.assert_frame_equal(result, expected)

        result = DataFrame(np.nan, index=[1, 2], columns=["A", "B"]).asof(3)
        expected = Series(np.nan, index=["A", "B"], name=3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "stamp,expected",
        [
            (
                Timestamp("2018-01-01 23:22:43.325+00:00"),
                Series(2, name=Timestamp("2018-01-01 23:22:43.325+00:00")),
            ),
            (
                Timestamp("2018-01-01 22:33:20.682+01:00"),
                Series(1, name=Timestamp("2018-01-01 22:33:20.682+01:00")),
            ),
        ],
    )
    def test_time_zone_aware_index(self, stamp, expected):
        # GH21194
        # Testing awareness of DataFrame index considering different
        # UTC and timezone
        df = DataFrame(
            data=[1, 2],
            index=[
                Timestamp("2018-01-01 21:00:05.001+00:00"),
                Timestamp("2018-01-01 22:35:10.550+00:00"),
            ],
        )

        result = df.asof(stamp)
        tm.assert_series_equal(result, expected)

    def test_is_copy(self, date_range_frame):
        # GH-27357, GH-30784: ensure the result of asof is an actual copy and
        # doesn't track the parent dataframe / doesn't give SettingWithCopy warnings
        df = date_range_frame.astype({"A": "float"})
        N = 50
        df.loc[df.index[15:30], "A"] = np.nan
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")

        result = df.asof(dates)

        with tm.assert_produces_warning(None):
            result["C"] = 1

    def test_asof_periodindex_mismatched_freq(self):
        N = 50
        rng = period_range("1/1/1990", periods=N, freq="h")
        df = DataFrame(np.random.default_rng(2).standard_normal(N), index=rng)

        # Mismatched freq
        msg = "Input has different freq"
        with pytest.raises(IncompatibleFrequency, match=msg):
            df.asof(rng.asfreq("D"))

    def test_asof_preserves_bool_dtype(self):
        # GH#16063 was casting bools to floats
        dti = date_range("2017-01-01", freq="MS", periods=4)
        ser = Series([True, False, True], index=dti[:-1])

        ts = dti[-1]
        res = ser.asof([ts])

        expected = Series([True], index=[ts])
        tm.assert_series_equal(res, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_convert_dtypes.py ===
import datetime

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class TestConvertDtypes:
    @pytest.mark.parametrize(
        "convert_integer, expected", [(False, np.dtype("int32")), (True, "Int32")]
    )
    def test_convert_dtypes(self, convert_integer, expected, string_storage):
        # Specific types are tested in tests/series/test_dtypes.py
        # Just check that it works for DataFrame here
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
                "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),
            }
        )
        with pd.option_context("string_storage", string_storage):
            result = df.convert_dtypes(True, True, convert_integer, False)
        expected = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype=expected),
                "b": pd.Series(["x", "y", "z"], dtype=f"string[{string_storage}]"),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_convert_empty(self):
        # Empty DataFrame can pass convert_dtypes, see GH#40393
        empty_df = pd.DataFrame()
        tm.assert_frame_equal(empty_df, empty_df.convert_dtypes())

    def test_convert_dtypes_retain_column_names(self):
        # GH#41435
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.columns.name = "cols"

        result = df.convert_dtypes()
        tm.assert_index_equal(result.columns, df.columns)
        assert result.columns.name == "cols"

    def test_pyarrow_dtype_backend(self):
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
                "b": pd.Series(["x", "y", None], dtype=np.dtype("O")),
                "c": pd.Series([True, False, None], dtype=np.dtype("O")),
                "d": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
                "e": pd.Series(pd.date_range("2022", periods=3)),
                "f": pd.Series(pd.date_range("2022", periods=3, tz="UTC").as_unit("s")),
                "g": pd.Series(pd.timedelta_range("1D", periods=3)),
            }
        )
        result = df.convert_dtypes(dtype_backend="pyarrow")
        expected = pd.DataFrame(
            {
                "a": pd.arrays.ArrowExtensionArray(
                    pa.array([1, 2, 3], type=pa.int32())
                ),
                "b": pd.arrays.ArrowExtensionArray(pa.array(["x", "y", None])),
                "c": pd.arrays.ArrowExtensionArray(pa.array([True, False, None])),
                "d": pd.arrays.ArrowExtensionArray(pa.array([None, 100.5, 200.0])),
                "e": pd.arrays.ArrowExtensionArray(
                    pa.array(
                        [
                            datetime.datetime(2022, 1, 1),
                            datetime.datetime(2022, 1, 2),
                            datetime.datetime(2022, 1, 3),
                        ],
                        type=pa.timestamp(unit="ns"),
                    )
                ),
                "f": pd.arrays.ArrowExtensionArray(
                    pa.array(
                        [
                            datetime.datetime(2022, 1, 1),
                            datetime.datetime(2022, 1, 2),
                            datetime.datetime(2022, 1, 3),
                        ],
                        type=pa.timestamp(unit="s", tz="UTC"),
                    )
                ),
                "g": pd.arrays.ArrowExtensionArray(
                    pa.array(
                        [
                            datetime.timedelta(1),
                            datetime.timedelta(2),
                            datetime.timedelta(3),
                        ],
                        type=pa.duration("ns"),
                    )
                ),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_dtype_backend_already_pyarrow(self):
        pytest.importorskip("pyarrow")
        expected = pd.DataFrame([1, 2, 3], dtype="int64[pyarrow]")
        result = expected.convert_dtypes(dtype_backend="pyarrow")
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_dtype_backend_from_pandas_nullable(self):
        pa = pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, None], dtype="Int32"),
                "b": pd.Series(["x", "y", None], dtype="string[python]"),
                "c": pd.Series([True, False, None], dtype="boolean"),
                "d": pd.Series([None, 100.5, 200], dtype="Float64"),
            }
        )
        result = df.convert_dtypes(dtype_backend="pyarrow")
        expected = pd.DataFrame(
            {
                "a": pd.arrays.ArrowExtensionArray(
                    pa.array([1, 2, None], type=pa.int32())
                ),
                "b": pd.arrays.ArrowExtensionArray(pa.array(["x", "y", None])),
                "c": pd.arrays.ArrowExtensionArray(pa.array([True, False, None])),
                "d": pd.arrays.ArrowExtensionArray(pa.array([None, 100.5, 200.0])),
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_dtype_empty_object(self):
        # GH 50970
        pytest.importorskip("pyarrow")
        expected = pd.DataFrame(columns=[0])
        result = expected.convert_dtypes(dtype_backend="pyarrow")
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_engine_lines_false(self):
        # GH 48893
        df = pd.DataFrame({"a": [1, 2, 3]})
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        with pytest.raises(ValueError, match=msg):
            df.convert_dtypes(dtype_backend="numpy")

    def test_pyarrow_backend_no_conversion(self):
        # GH#52872
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"a": [1, 2], "b": 1.5, "c": True, "d": "x"})
        expected = df.copy()
        result = df.convert_dtypes(
            convert_floating=False,
            convert_integer=False,
            convert_boolean=False,
            convert_string=False,
            dtype_backend="pyarrow",
        )
        tm.assert_frame_equal(result, expected)

    def test_convert_dtypes_pyarrow_to_np_nullable(self):
        # GH 53648
        pytest.importorskip("pyarrow")
        ser = pd.DataFrame(range(2), dtype="int32[pyarrow]")
        result = ser.convert_dtypes(dtype_backend="numpy_nullable")
        expected = pd.DataFrame(range(2), dtype="Int32")
        tm.assert_frame_equal(result, expected)

    def test_convert_dtypes_pyarrow_timestamp(self):
        # GH 54191
        pytest.importorskip("pyarrow")
        ser = pd.Series(pd.date_range("2020-01-01", "2020-01-02", freq="1min"))
        expected = ser.astype("timestamp[ms][pyarrow]")
        result = expected.convert_dtypes(dtype_backend="pyarrow")
        tm.assert_series_equal(result, expected)

    def test_convert_dtypes_avoid_block_splitting(self):
        # GH#55341
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": "a"})
        result = df.convert_dtypes(convert_integer=False)
        expected = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": pd.Series(["a"] * 3, dtype="string[python]"),
            }
        )
        tm.assert_frame_equal(result, expected)
        assert result._mgr.nblocks == 2

    def test_convert_dtypes_from_arrow(self):
        # GH#56581
        df = pd.DataFrame([["a", datetime.time(18, 12)]], columns=["a", "b"])
        result = df.convert_dtypes()
        expected = df.astype({"a": "string[python]"})
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_partial_slicing.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    PeriodIndex,
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestPeriodIndex:
    def test_getitem_periodindex_duplicates_string_slice(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # monotonic
        idx = PeriodIndex([2000, 2007, 2007, 2009, 2009], freq="Y-JUN")
        ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)
        original = ts.copy()

        result = ts["2007"]
        expected = ts[1:3]
        tm.assert_series_equal(result, expected)
        with tm.assert_cow_warning(warn_copy_on_write):
            result[:] = 1
        if using_copy_on_write:
            tm.assert_series_equal(ts, original)
        else:
            assert (ts[1:3] == 1).all()

        # not monotonic
        idx = PeriodIndex([2000, 2007, 2007, 2009, 2007], freq="Y-JUN")
        ts = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)

        result = ts["2007"]
        expected = ts[idx == "2007"]
        tm.assert_series_equal(result, expected)

    def test_getitem_periodindex_quarter_string(self):
        pi = PeriodIndex(["2Q05", "3Q05", "4Q05", "1Q06", "2Q06"], freq="Q")
        ser = Series(np.random.default_rng(2).random(len(pi)), index=pi).cumsum()
        # Todo: fix these accessors!
        assert ser["05Q4"] == ser.iloc[2]

    def test_pindex_slice_index(self):
        pi = period_range(start="1/1/10", end="12/31/12", freq="M")
        s = Series(np.random.default_rng(2).random(len(pi)), index=pi)
        res = s["2010"]
        exp = s[0:12]
        tm.assert_series_equal(res, exp)
        res = s["2011"]
        exp = s[12:24]
        tm.assert_series_equal(res, exp)

    @pytest.mark.parametrize("make_range", [date_range, period_range])
    def test_range_slice_day(self, make_range):
        # GH#6716
        idx = make_range(start="2013/01/01", freq="D", periods=400)

        msg = "slice indices must be integers or None or have an __index__ method"
        # slices against index should raise IndexError
        values = [
            "2014",
            "2013/02",
            "2013/01/02",
            "2013/02/01 9H",
            "2013/02/01 09:00",
        ]
        for v in values:
            with pytest.raises(TypeError, match=msg):
                idx[v:]

        s = Series(np.random.default_rng(2).random(len(idx)), index=idx)

        tm.assert_series_equal(s["2013/01/02":], s[1:])
        tm.assert_series_equal(s["2013/01/02":"2013/01/05"], s[1:5])
        tm.assert_series_equal(s["2013/02":], s[31:])
        tm.assert_series_equal(s["2014":], s[365:])

        invalid = ["2013/02/01 9H", "2013/02/01 09:00"]
        for v in invalid:
            with pytest.raises(TypeError, match=msg):
                idx[v:]

    @pytest.mark.parametrize("make_range", [date_range, period_range])
    def test_range_slice_seconds(self, make_range):
        # GH#6716
        idx = make_range(start="2013/01/01 09:00:00", freq="s", periods=4000)
        msg = "slice indices must be integers or None or have an __index__ method"

        # slices against index should raise IndexError
        values = [
            "2014",
            "2013/02",
            "2013/01/02",
            "2013/02/01 9H",
            "2013/02/01 09:00",
        ]
        for v in values:
            with pytest.raises(TypeError, match=msg):
                idx[v:]

        s = Series(np.random.default_rng(2).random(len(idx)), index=idx)

        tm.assert_series_equal(s["2013/01/01 09:05":"2013/01/01 09:10"], s[300:660])
        tm.assert_series_equal(s["2013/01/01 10:00":"2013/01/01 10:05"], s[3600:3960])
        tm.assert_series_equal(s["2013/01/01 10H":], s[3600:])
        tm.assert_series_equal(s[:"2013/01/01 09:30"], s[:1860])
        for d in ["2013/01/01", "2013/01", "2013"]:
            tm.assert_series_equal(s[d:], s)

    @pytest.mark.parametrize("make_range", [date_range, period_range])
    def test_range_slice_outofbounds(self, make_range):
        # GH#5407
        idx = make_range(start="2013/10/01", freq="D", periods=10)

        df = DataFrame({"units": [100 + i for i in range(10)]}, index=idx)
        empty = DataFrame(index=idx[:0], columns=["units"])
        empty["units"] = empty["units"].astype("int64")

        tm.assert_frame_equal(df["2013/09/01":"2013/09/30"], empty)
        tm.assert_frame_equal(df["2013/09/30":"2013/10/02"], df.iloc[:2])
        tm.assert_frame_equal(df["2013/10/01":"2013/10/02"], df.iloc[:2])
        tm.assert_frame_equal(df["2013/10/02":"2013/09/30"], empty)
        tm.assert_frame_equal(df["2013/10/15":"2013/10/17"], empty)
        tm.assert_frame_equal(df["2013-06":"2013-09"], empty)
        tm.assert_frame_equal(df["2013-11":"2013-12"], empty)

    @pytest.mark.parametrize("make_range", [date_range, period_range])
    def test_maybe_cast_slice_bound(self, make_range, frame_or_series):
        idx = make_range(start="2013/10/01", freq="D", periods=10)

        obj = DataFrame({"units": [100 + i for i in range(10)]}, index=idx)
        obj = tm.get_obj(obj, frame_or_series)

        msg = (
            f"cannot do slice indexing on {type(idx).__name__} with "
            r"these indexers \[foo\] of type str"
        )

        # Check the lower-level calls are raising where expected.
        with pytest.raises(TypeError, match=msg):
            idx._maybe_cast_slice_bound("foo", "left")
        with pytest.raises(TypeError, match=msg):
            idx.get_slice_bound("foo", "left")

        with pytest.raises(TypeError, match=msg):
            obj["2013/09/30":"foo"]
        with pytest.raises(TypeError, match=msg):
            obj["foo":"2013/09/30"]
        with pytest.raises(TypeError, match=msg):
            obj.loc["2013/09/30":"foo"]
        with pytest.raises(TypeError, match=msg):
            obj.loc["foo":"2013/09/30"]

    def test_partial_slice_doesnt_require_monotonicity(self):
        # See also: DatetimeIndex test ofm the same name
        dti = date_range("2014-01-01", periods=30, freq="30D")
        pi = dti.to_period("D")

        ser_montonic = Series(np.arange(30), index=pi)

        shuffler = list(range(0, 30, 2)) + list(range(1, 31, 2))
        ser = ser_montonic.iloc[shuffler]
        nidx = ser.index

        # Manually identified locations of year==2014
        indexer_2014 = np.array(
            [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20], dtype=np.intp
        )
        assert (nidx[indexer_2014].year == 2014).all()
        assert not (nidx[~indexer_2014].year == 2014).any()

        result = nidx.get_loc("2014")
        tm.assert_numpy_array_equal(result, indexer_2014)

        expected = ser.iloc[indexer_2014]
        result = ser.loc["2014"]
        tm.assert_series_equal(result, expected)

        result = ser["2014"]
        tm.assert_series_equal(result, expected)

        # Manually identified locations where ser.index is within Mat 2015
        indexer_may2015 = np.array([23], dtype=np.intp)
        assert nidx[23].year == 2015 and nidx[23].month == 5

        result = nidx.get_loc("May 2015")
        tm.assert_numpy_array_equal(result, indexer_may2015)

        expected = ser.iloc[indexer_may2015]
        result = ser.loc["May 2015"]
        tm.assert_series_equal(result, expected)

        result = ser["May 2015"]
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_logic.py ===
from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
    fuzzy_or, _fuzzy_group, _torf, fuzzy_nand, fuzzy_xor)
from sympy.testing.pytest import raises

from itertools import product

T = True
F = False
U = None



def test_torf():
    v = [T, F, U]
    for i in product(*[v]*3):
        assert _torf(i) is (True if all(j for j in i) else
                            (False if all(j is False for j in i) else None))


def test_fuzzy_group():
    v = [T, F, U]
    for i in product(*[v]*3):
        assert _fuzzy_group(i) is (None if None in i else
                                   (True if all(j for j in i) else False))
        assert _fuzzy_group(i, quick_exit=True) is \
            (None if (i.count(False) > 1) else
             (None if None in i else (True if all(j for j in i) else False)))
    it = (True if (i == 0) else None for i in range(2))
    assert _torf(it) is None
    it = (True if (i == 1) else None for i in range(2))
    assert _torf(it) is None


def test_fuzzy_not():
    assert fuzzy_not(T) == F
    assert fuzzy_not(F) == T
    assert fuzzy_not(U) == U


def test_fuzzy_and():
    assert fuzzy_and([T, T]) == T
    assert fuzzy_and([T, F]) == F
    assert fuzzy_and([T, U]) == U
    assert fuzzy_and([F, F]) == F
    assert fuzzy_and([F, U]) == F
    assert fuzzy_and([U, U]) == U
    assert [fuzzy_and([w]) for w in [U, T, F]] == [U, T, F]
    assert fuzzy_and([T, F, U]) == F
    assert fuzzy_and([]) == T
    raises(TypeError, lambda: fuzzy_and())


def test_fuzzy_or():
    assert fuzzy_or([T, T]) == T
    assert fuzzy_or([T, F]) == T
    assert fuzzy_or([T, U]) == T
    assert fuzzy_or([F, F]) == F
    assert fuzzy_or([F, U]) == U
    assert fuzzy_or([U, U]) == U
    assert [fuzzy_or([w]) for w in [U, T, F]] == [U, T, F]
    assert fuzzy_or([T, F, U]) == T
    assert fuzzy_or([]) == F
    raises(TypeError, lambda: fuzzy_or())


def test_logic_cmp():
    l1 = And('a', Not('b'))
    l2 = And('a', Not('b'))

    assert hash(l1) == hash(l2)
    assert (l1 == l2) == T
    assert (l1 != l2) == F

    assert And('a', 'b', 'c') == And('b', 'a', 'c')
    assert And('a', 'b', 'c') == And('c', 'b', 'a')
    assert And('a', 'b', 'c') == And('c', 'a', 'b')

    assert Not('a') < Not('b')
    assert (Not('b') < Not('a')) is False
    assert (Not('a') < 2) is False


def test_logic_onearg():
    assert And() is True
    assert Or() is False

    assert And(T) == T
    assert And(F) == F
    assert Or(T) == T
    assert Or(F) == F

    assert And('a') == 'a'
    assert Or('a') == 'a'


def test_logic_xnotx():
    assert And('a', Not('a')) == F
    assert Or('a', Not('a')) == T


def test_logic_eval_TF():
    assert And(F, F) == F
    assert And(F, T) == F
    assert And(T, F) == F
    assert And(T, T) == T

    assert Or(F, F) == F
    assert Or(F, T) == T
    assert Or(T, F) == T
    assert Or(T, T) == T

    assert And('a', T) == 'a'
    assert And('a', F) == F
    assert Or('a', T) == T
    assert Or('a', F) == 'a'


def test_logic_combine_args():
    assert And('a', 'b', 'a') == And('a', 'b')
    assert Or('a', 'b', 'a') == Or('a', 'b')

    assert And(And('a', 'b'), And('c', 'd')) == And('a', 'b', 'c', 'd')
    assert Or(Or('a', 'b'), Or('c', 'd')) == Or('a', 'b', 'c', 'd')

    assert Or('t', And('n', 'p', 'r'), And('n', 'r'), And('n', 'p', 'r'), 't',
              And('n', 'r')) == Or('t', And('n', 'p', 'r'), And('n', 'r'))


def test_logic_expand():
    t = And(Or('a', 'b'), 'c')
    assert t.expand() == Or(And('a', 'c'), And('b', 'c'))

    t = And(Or('a', Not('b')), 'b')
    assert t.expand() == And('a', 'b')

    t = And(Or('a', 'b'), Or('c', 'd'))
    assert t.expand() == \
        Or(And('a', 'c'), And('a', 'd'), And('b', 'c'), And('b', 'd'))


def test_logic_fromstring():
    S = Logic.fromstring

    assert S('a') == 'a'
    assert S('!a') == Not('a')
    assert S('a & b') == And('a', 'b')
    assert S('a | b') == Or('a', 'b')
    assert S('a | b & c') == And(Or('a', 'b'), 'c')
    assert S('a & b | c') == Or(And('a', 'b'), 'c')
    assert S('a & b & c') == And('a', 'b', 'c')
    assert S('a | b | c') == Or('a', 'b', 'c')

    raises(ValueError, lambda: S('| a'))
    raises(ValueError, lambda: S('& a'))
    raises(ValueError, lambda: S('a | | b'))
    raises(ValueError, lambda: S('a | & b'))
    raises(ValueError, lambda: S('a & & b'))
    raises(ValueError, lambda: S('a |'))
    raises(ValueError, lambda: S('a|b'))
    raises(ValueError, lambda: S('!'))
    raises(ValueError, lambda: S('! a'))
    raises(ValueError, lambda: S('!(a + 1)'))
    raises(ValueError, lambda: S(''))


def test_logic_not():
    assert Not('a') != '!a'
    assert Not('!a') != 'a'
    assert Not(True) == False
    assert Not(False) == True

    # NOTE: we may want to change default Not behaviour and put this
    # functionality into some method.
    assert Not(And('a', 'b')) == Or(Not('a'), Not('b'))
    assert Not(Or('a', 'b')) == And(Not('a'), Not('b'))

    raises(ValueError, lambda: Not(1))


def test_formatting():
    S = Logic.fromstring
    raises(ValueError, lambda: S('a&b'))
    raises(ValueError, lambda: S('a|b'))
    raises(ValueError, lambda: S('! a'))


def test_fuzzy_xor():
    assert fuzzy_xor((None,)) is None
    assert fuzzy_xor((None, True)) is None
    assert fuzzy_xor((None, False)) is None
    assert fuzzy_xor((True, False)) is True
    assert fuzzy_xor((True, True)) is False
    assert fuzzy_xor((True, True, False)) is False
    assert fuzzy_xor((True, True, False, True)) is True

def test_fuzzy_nand():
    for args in [(1, 0), (1, 1), (0, 0)]:
        assert fuzzy_nand(args) == fuzzy_not(fuzzy_and(args))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_lxml.py ===
"""Tests to ensure that the lxml tree builder generates good trees."""

import pickle
import pytest
import warnings
from . import LXML_PRESENT, LXML_VERSION

if LXML_PRESENT:
    from bs4.builder._lxml import LXMLTreeBuilder, LXMLTreeBuilderForXML

from bs4 import (
    BeautifulStoneSoup,
)
from . import (
    HTMLTreeBuilderSmokeTest,
    XMLTreeBuilderSmokeTest,
    SOUP_SIEVE_PRESENT,
)


@pytest.mark.skipif(
    not LXML_PRESENT,
    reason="lxml seems not to be present, not testing its tree builder.",
)
class TestLXMLTreeBuilder(HTMLTreeBuilderSmokeTest):
    """See ``HTMLTreeBuilderSmokeTest``."""

    @property
    def default_builder(self):
        return LXMLTreeBuilder

    def test_out_of_range_entity(self):
        self.assert_soup("<p>foo&#10000000000000;bar</p>", "<p>foobar</p>")
        self.assert_soup("<p>foo&#x10000000000000;bar</p>", "<p>foobar</p>")
        self.assert_soup("<p>foo&#1000000000;bar</p>", "<p>foobar</p>")

    def test_entities_in_foreign_document_encoding(self):
        # We can't implement this case correctly because by the time we
        # hear about markup like "&#147;", it's been (incorrectly) converted into
        # a string like u'\x93'
        pass

    # In lxml < 2.3.5, an empty doctype causes a segfault. Skip this
    # test if an old version of lxml is installed.

    @pytest.mark.skipif(
        not LXML_PRESENT or LXML_VERSION < (2, 3, 5, 0),
        reason="Skipping doctype test for old version of lxml to avoid segfault.",
    )
    def test_empty_doctype(self):
        soup = self.soup("<!DOCTYPE>")
        doctype = soup.contents[0]
        assert "" == doctype.strip()

    def test_beautifulstonesoup_is_xml_parser(self):
        # Make sure that the deprecated BSS class uses an xml builder
        # if one is installed.
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulStoneSoup("<b />")
        assert "<b/>" == str(soup.b)
        [warning] = w
        assert warning.filename == __file__
        assert "The BeautifulStoneSoup class was deprecated" in str(warning.message)

    def test_tracking_line_numbers(self):
        # The lxml TreeBuilder cannot keep track of line numbers from
        # the original markup. Even if you ask for line numbers, we
        # don't have 'em.
        #
        # However, for consistency with other parsers, Tag.sourceline
        # and Tag.sourcepos are always set to None, rather than being
        # available as an alias for find().
        soup = self.soup(
            "\n   <p>\n\n<sourceline>\n<b>text</b></sourceline><sourcepos></p>",
            store_line_numbers=True,
        )
        assert None is soup.p.sourceline
        assert None is soup.p.sourcepos


@pytest.mark.skipif(
    not LXML_PRESENT,
    reason="lxml seems not to be present, not testing its XML tree builder.",
)
class TestLXMLXMLTreeBuilder(XMLTreeBuilderSmokeTest):
    """See ``HTMLTreeBuilderSmokeTest``."""

    @property
    def default_builder(self):
        return LXMLTreeBuilderForXML

    def test_namespace_indexing(self):
        soup = self.soup(
            '<?xml version="1.1"?>\n'
            "<root>"
            '<tag xmlns="http://unprefixed-namespace.com">content</tag>'
            '<prefix:tag2 xmlns:prefix="http://prefixed-namespace.com">content</prefix:tag2>'
            '<prefix2:tag3 xmlns:prefix2="http://another-namespace.com">'
            '<subtag xmlns="http://another-unprefixed-namespace.com">'
            '<subsubtag xmlns="http://yet-another-unprefixed-namespace.com">'
            "</prefix2:tag3>"
            "</root>"
        )

        # The BeautifulSoup object includes every namespace prefix
        # defined in the entire document. This is the default set of
        # namespaces used by soupsieve.
        #
        # Un-prefixed namespaces are not included, and if a given
        # prefix is defined twice, only the first prefix encountered
        # in the document shows up here.
        assert soup._namespaces == {
            "xml": "http://www.w3.org/XML/1998/namespace",
            "prefix": "http://prefixed-namespace.com",
            "prefix2": "http://another-namespace.com",
        }

        # A Tag object includes only the namespace prefixes
        # that were in scope when it was parsed.

        # We do not track un-prefixed namespaces as we can only hold
        # one (the first one), and it will be recognized as the
        # default namespace by soupsieve, even when operating from a
        # tag with a different un-prefixed namespace.
        assert soup.tag._namespaces == {
            "xml": "http://www.w3.org/XML/1998/namespace",
        }

        assert soup.tag2._namespaces == {
            "prefix": "http://prefixed-namespace.com",
            "xml": "http://www.w3.org/XML/1998/namespace",
        }

        assert soup.subtag._namespaces == {
            "prefix2": "http://another-namespace.com",
            "xml": "http://www.w3.org/XML/1998/namespace",
        }

        assert soup.subsubtag._namespaces == {
            "prefix2": "http://another-namespace.com",
            "xml": "http://www.w3.org/XML/1998/namespace",
        }

    @pytest.mark.skipif(not SOUP_SIEVE_PRESENT, reason="Soup Sieve not installed")
    def test_namespace_interaction_with_select_and_find(self):
        # Demonstrate how namespaces interact with select* and
        # find* methods.

        soup = self.soup(
            '<?xml version="1.1"?>\n'
            "<root>"
            '<tag xmlns="http://unprefixed-namespace.com">content</tag>'
            '<prefix:tag2 xmlns:prefix="http://prefixed-namespace.com">content</tag>'
            '<subtag xmlns:prefix="http://another-namespace-same-prefix.com">'
            "<prefix:tag3>"
            "</subtag>"
            "</root>"
        )

        # soupselect uses namespace URIs.
        assert soup.select_one("tag").name == "tag"
        assert soup.select_one("prefix|tag2").name == "tag2"

        # If a prefix is declared more than once, only the first usage
        # is registered with the BeautifulSoup object.
        assert soup.select_one("prefix|tag3") is None

        # But you can always explicitly specify a namespace dictionary.
        assert (
            soup.select_one("prefix|tag3", namespaces=soup.subtag._namespaces).name
            == "tag3"
        )

        # And a Tag (as opposed to the BeautifulSoup object) will
        # have a set of default namespaces scoped to that Tag.
        assert soup.subtag.select_one("prefix|tag3").name == "tag3"

        # the find() methods aren't fully namespace-aware; they just
        # look at prefixes.
        assert soup.find("tag").name == "tag"
        assert soup.find("prefix:tag2").name == "tag2"
        assert soup.find("prefix:tag3").name == "tag3"
        assert soup.subtag.find("prefix:tag3").name == "tag3"

    def test_pickle_restores_builder(self):
        # The lxml TreeBuilder is not picklable, so when unpickling
        # a document created with it, a new TreeBuilder of the
        # appropriate class is created.
        soup = self.soup("<a>some markup</a>")
        assert isinstance(soup.builder, self.default_builder)
        pickled = pickle.dumps(soup)
        unpickled = pickle.loads(pickled)

        assert "some markup" == unpickled.a.string
        assert unpickled.builder != soup.builder
        assert isinstance(unpickled.builder, self.default_builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\accessors\test_struct_accessor.py ===
import re

import pytest

from pandas.compat.pyarrow import (
    pa_version_under11p0,
    pa_version_under13p0,
)

from pandas import (
    ArrowDtype,
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")


def test_struct_accessor_dtypes():
    ser = Series(
        [],
        dtype=ArrowDtype(
            pa.struct(
                [
                    ("int_col", pa.int64()),
                    ("string_col", pa.string()),
                    (
                        "struct_col",
                        pa.struct(
                            [
                                ("int_col", pa.int64()),
                                ("float_col", pa.float64()),
                            ]
                        ),
                    ),
                ]
            )
        ),
    )
    actual = ser.struct.dtypes
    expected = Series(
        [
            ArrowDtype(pa.int64()),
            ArrowDtype(pa.string()),
            ArrowDtype(
                pa.struct(
                    [
                        ("int_col", pa.int64()),
                        ("float_col", pa.float64()),
                    ]
                )
            ),
        ],
        index=Index(["int_col", "string_col", "struct_col"]),
    )
    tm.assert_series_equal(actual, expected)


@pytest.mark.skipif(pa_version_under13p0, reason="pyarrow>=13.0.0 required")
def test_struct_accessor_field():
    index = Index([-100, 42, 123])
    ser = Series(
        [
            {"rice": 1.0, "maize": -1, "wheat": "a"},
            {"rice": 2.0, "maize": 0, "wheat": "b"},
            {"rice": 3.0, "maize": 1, "wheat": "c"},
        ],
        dtype=ArrowDtype(
            pa.struct(
                [
                    ("rice", pa.float64()),
                    ("maize", pa.int64()),
                    ("wheat", pa.string()),
                ]
            )
        ),
        index=index,
    )
    by_name = ser.struct.field("maize")
    by_name_expected = Series(
        [-1, 0, 1],
        dtype=ArrowDtype(pa.int64()),
        index=index,
        name="maize",
    )
    tm.assert_series_equal(by_name, by_name_expected)

    by_index = ser.struct.field(2)
    by_index_expected = Series(
        ["a", "b", "c"],
        dtype=ArrowDtype(pa.string()),
        index=index,
        name="wheat",
    )
    tm.assert_series_equal(by_index, by_index_expected)


def test_struct_accessor_field_with_invalid_name_or_index():
    ser = Series([], dtype=ArrowDtype(pa.struct([("field", pa.int64())])))

    with pytest.raises(ValueError, match="name_or_index must be an int, str,"):
        ser.struct.field(1.1)


@pytest.mark.skipif(pa_version_under11p0, reason="pyarrow>=11.0.0 required")
def test_struct_accessor_explode():
    index = Index([-100, 42, 123])
    ser = Series(
        [
            {"painted": 1, "snapping": {"sea": "green"}},
            {"painted": 2, "snapping": {"sea": "leatherback"}},
            {"painted": 3, "snapping": {"sea": "hawksbill"}},
        ],
        dtype=ArrowDtype(
            pa.struct(
                [
                    ("painted", pa.int64()),
                    ("snapping", pa.struct([("sea", pa.string())])),
                ]
            )
        ),
        index=index,
    )
    actual = ser.struct.explode()
    expected = DataFrame(
        {
            "painted": Series([1, 2, 3], index=index, dtype=ArrowDtype(pa.int64())),
            "snapping": Series(
                [{"sea": "green"}, {"sea": "leatherback"}, {"sea": "hawksbill"}],
                index=index,
                dtype=ArrowDtype(pa.struct([("sea", pa.string())])),
            ),
        },
    )
    tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(Series([1, 2, 3], dtype="int64"), id="int64"),
        pytest.param(
            Series(["a", "b", "c"], dtype="string[pyarrow]"), id="string-pyarrow"
        ),
    ],
)
def test_struct_accessor_api_for_invalid(invalid):
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "Can only use the '.struct' accessor with 'struct[pyarrow]' dtype, "
            f"not {invalid.dtype}."
        ),
    ):
        invalid.struct


@pytest.mark.parametrize(
    ["indices", "name"],
    [
        (0, "int_col"),
        ([1, 2], "str_col"),
        (pc.field("int_col"), "int_col"),
        ("int_col", "int_col"),
        (b"string_col", b"string_col"),
        ([b"string_col"], "string_col"),
    ],
)
@pytest.mark.skipif(pa_version_under13p0, reason="pyarrow>=13.0.0 required")
def test_struct_accessor_field_expanded(indices, name):
    arrow_type = pa.struct(
        [
            ("int_col", pa.int64()),
            (
                "struct_col",
                pa.struct(
                    [
                        ("int_col", pa.int64()),
                        ("float_col", pa.float64()),
                        ("str_col", pa.string()),
                    ]
                ),
            ),
            (b"string_col", pa.string()),
        ]
    )

    data = pa.array([], type=arrow_type)
    ser = Series(data, dtype=ArrowDtype(arrow_type))
    expected = pc.struct_field(data, indices)
    result = ser.struct.field(indices)
    tm.assert_equal(result.array._pa_array.combine_chunks(), expected)
    assert result.name == name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\agca\tests\test_extensions.py ===
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix

from sympy.testing.pytest import raises

from sympy.abc import x, y, t


def test_FiniteExtension():
    # Gaussian integers
    A = FiniteExtension(Poly(x**2 + 1, x))
    assert A.rank == 2
    assert str(A) == 'ZZ[x]/(x**2 + 1)'
    i = A.generator
    assert i.parent() is A

    assert i*i == A(-1)
    raises(TypeError, lambda: i*())

    assert A.basis == (A.one, i)
    assert A(1) == A.one
    assert i**2 == A(-1)
    assert i**2 != -1  # no coercion
    assert (2 + i)*(1 - i) == 3 - i
    assert (1 + i)**8 == A(16)
    assert A(1).inverse() == A(1)
    raises(NotImplementedError, lambda: A(2).inverse())

    # Finite field of order 27
    F = FiniteExtension(Poly(x**3 - x + 1, x, modulus=3))
    assert F.rank == 3
    a = F.generator  # also generates the cyclic group F - {0}
    assert F.basis == (F(1), a, a**2)
    assert a**27 == a
    assert a**26 == F(1)
    assert a**13 == F(-1)
    assert a**9 == a + 1
    assert a**3 == a - 1
    assert a**6 == a**2 + a + 1
    assert F(x**2 + x).inverse() == 1 - a
    assert F(x + 2)**(-1) == F(x + 2).inverse()
    assert a**19 * a**(-19) == F(1)
    assert (a - 1) / (2*a**2 - 1) == a**2 + 1
    assert (a - 1) // (2*a**2 - 1) == a**2 + 1
    assert 2/(a**2 + 1) == a**2 - a + 1
    assert (a**2 + 1)/2 == -a**2 - 1
    raises(NotInvertible, lambda: F(0).inverse())

    # Function field of an elliptic curve
    K = FiniteExtension(Poly(t**2 - x**3 - x + 1, t, field=True))
    assert K.rank == 2
    assert str(K) == 'ZZ(x)[t]/(t**2 - x**3 - x + 1)'
    y = K.generator
    c = 1/(x**3 - x**2 + x - 1)
    assert ((y + x)*(y - x)).inverse() == K(c)
    assert (y + x)*(y - x)*c == K(1)  # explicit inverse of y + x


def test_FiniteExtension_eq_hash():
    # Test eq and hash
    p1 = Poly(x**2 - 2, x, domain=ZZ)
    p2 = Poly(x**2 - 2, x, domain=QQ)
    K1 = FiniteExtension(p1)
    K2 = FiniteExtension(p2)
    assert K1 == FiniteExtension(Poly(x**2 - 2))
    assert K2 != FiniteExtension(Poly(x**2 - 2))
    assert len({K1, K2, FiniteExtension(p1)}) == 2


def test_FiniteExtension_mod():
    # Test mod
    K = FiniteExtension(Poly(x**3 + 1, x, domain=QQ))
    xf = K(x)
    assert (xf**2 - 1) % 1 == K.zero
    assert 1 % (xf**2 - 1) == K.zero
    assert (xf**2 - 1) / (xf - 1) == xf + 1
    assert (xf**2 - 1) // (xf - 1) == xf + 1
    assert (xf**2 - 1) % (xf - 1) == K.zero
    raises(ZeroDivisionError, lambda: (xf**2 - 1) % 0)
    raises(TypeError, lambda: xf % [])
    raises(TypeError, lambda: [] % xf)

    # Test mod over ring
    K = FiniteExtension(Poly(x**3 + 1, x, domain=ZZ))
    xf = K(x)
    assert (xf**2 - 1) % 1 == K.zero
    raises(NotImplementedError, lambda: (xf**2 - 1) % (xf - 1))


def test_FiniteExtension_from_sympy():
    # Test to_sympy/from_sympy
    K = FiniteExtension(Poly(x**3 + 1, x, domain=ZZ))
    xf = K(x)
    assert K.from_sympy(x) == xf
    assert K.to_sympy(xf) == x


def test_FiniteExtension_set_domain():
    KZ = FiniteExtension(Poly(x**2 + 1, x, domain='ZZ'))
    KQ = FiniteExtension(Poly(x**2 + 1, x, domain='QQ'))
    assert KZ.set_domain(QQ) == KQ


def test_FiniteExtension_exquo():
    # Test exquo
    K = FiniteExtension(Poly(x**4 + 1))
    xf = K(x)
    assert K.exquo(xf**2 - 1, xf - 1) == xf + 1


def test_FiniteExtension_convert():
    # Test from_MonogenicFiniteExtension
    K1 = FiniteExtension(Poly(x**2 + 1))
    K2 = QQ[x]
    x1, x2 = K1(x), K2(x)
    assert K1.convert(x2) == x1
    assert K2.convert(x1) == x2

    K = FiniteExtension(Poly(x**2 - 1, domain=QQ))
    assert K.convert_from(QQ(1, 2), QQ) == K.one/2


def test_FiniteExtension_division_ring():
    # Test division in FiniteExtension over a ring
    KQ = FiniteExtension(Poly(x**2 - 1, x, domain=QQ))
    KZ = FiniteExtension(Poly(x**2 - 1, x, domain=ZZ))
    KQt = FiniteExtension(Poly(x**2 - 1, x, domain=QQ[t]))
    KQtf = FiniteExtension(Poly(x**2 - 1, x, domain=QQ.frac_field(t)))
    assert KQ.is_Field is True
    assert KZ.is_Field is False
    assert KQt.is_Field is False
    assert KQtf.is_Field is True
    for K in KQ, KZ, KQt, KQtf:
        xK = K.convert(x)
        assert xK / K.one == xK
        assert xK // K.one == xK
        assert xK % K.one == K.zero
        raises(ZeroDivisionError, lambda: xK / K.zero)
        raises(ZeroDivisionError, lambda: xK // K.zero)
        raises(ZeroDivisionError, lambda: xK % K.zero)
        if K.is_Field:
            assert xK / xK == K.one
            assert xK // xK == K.one
            assert xK % xK == K.zero
        else:
            raises(NotImplementedError, lambda: xK / xK)
            raises(NotImplementedError, lambda: xK // xK)
            raises(NotImplementedError, lambda: xK % xK)


def test_FiniteExtension_Poly():
    K = FiniteExtension(Poly(x**2 - 2))
    p = Poly(x, y, domain=K)
    assert p.domain == K
    assert p.as_expr() == x
    assert (p**2).as_expr() == 2

    K = FiniteExtension(Poly(x**2 - 2, x, domain=QQ))
    K2 = FiniteExtension(Poly(t**2 - 2, t, domain=K))
    assert str(K2) == 'QQ[x]/(x**2 - 2)[t]/(t**2 - 2)'

    eK = K2.convert(x + t)
    assert K2.to_sympy(eK) == x + t
    assert K2.to_sympy(eK ** 2) == 4 + 2*x*t
    p = Poly(x + t, y, domain=K2)
    assert p**2 == Poly(4 + 2*x*t, y, domain=K2)


def test_FiniteExtension_sincos_jacobian():
    # Use FiniteExtensino to compute the Jacobian of a matrix involving sin
    # and cos of different symbols.
    r, p, t = symbols('rho, phi, theta')
    elements = [
        [sin(p)*cos(t), r*cos(p)*cos(t), -r*sin(p)*sin(t)],
        [sin(p)*sin(t), r*cos(p)*sin(t),  r*sin(p)*cos(t)],
        [       cos(p),       -r*sin(p),                0],
    ]

    def make_extension(K):
        K = FiniteExtension(Poly(sin(p)**2+cos(p)**2-1, sin(p), domain=K[cos(p)]))
        K = FiniteExtension(Poly(sin(t)**2+cos(t)**2-1, sin(t), domain=K[cos(t)]))
        return K

    Ksc1 = make_extension(ZZ[r])
    Ksc2 = make_extension(ZZ)[r]

    for K in [Ksc1, Ksc2]:
        elements_K = [[K.convert(e) for e in row] for row in elements]
        J = DomainMatrix(elements_K, (3, 3), K)
        det = J.charpoly()[-1] * (-K.one)**3
        assert det == K.convert(r**2*sin(p))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_tree.py ===
from sympy.printing.tree import tree
from sympy.testing.pytest import XFAIL


# Remove this flag after making _assumptions cache deterministic.
@XFAIL
def test_print_tree_MatAdd():
    from sympy.matrices.expressions import MatrixSymbol
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)

    test_str = [
        'MatAdd: A + B\n',
        'algebraic: False\n',
        'commutative: False\n',
        'complex: False\n',
        'composite: False\n',
        'even: False\n',
        'extended_negative: False\n',
        'extended_nonnegative: False\n',
        'extended_nonpositive: False\n',
        'extended_nonzero: False\n',
        'extended_positive: False\n',
        'extended_real: False\n',
        'imaginary: False\n',
        'integer: False\n',
        'irrational: False\n',
        'negative: False\n',
        'noninteger: False\n',
        'nonnegative: False\n',
        'nonpositive: False\n',
        'nonzero: False\n',
        'odd: False\n',
        'positive: False\n',
        'prime: False\n',
        'rational: False\n',
        'real: False\n',
        'transcendental: False\n',
        'zero: False\n',
        '+-MatrixSymbol: A\n',
        '| algebraic: False\n',
        '| commutative: False\n',
        '| complex: False\n',
        '| composite: False\n',
        '| even: False\n',
        '| extended_negative: False\n',
        '| extended_nonnegative: False\n',
        '| extended_nonpositive: False\n',
        '| extended_nonzero: False\n',
        '| extended_positive: False\n',
        '| extended_real: False\n',
        '| imaginary: False\n',
        '| integer: False\n',
        '| irrational: False\n',
        '| negative: False\n',
        '| noninteger: False\n',
        '| nonnegative: False\n',
        '| nonpositive: False\n',
        '| nonzero: False\n',
        '| odd: False\n',
        '| positive: False\n',
        '| prime: False\n',
        '| rational: False\n',
        '| real: False\n',
        '| transcendental: False\n',
        '| zero: False\n',
        '| +-Symbol: A\n',
        '| | commutative: True\n',
        '| +-Integer: 3\n',
        '| | algebraic: True\n',
        '| | commutative: True\n',
        '| | complex: True\n',
        '| | extended_negative: False\n',
        '| | extended_nonnegative: True\n',
        '| | extended_real: True\n',
        '| | finite: True\n',
        '| | hermitian: True\n',
        '| | imaginary: False\n',
        '| | infinite: False\n',
        '| | integer: True\n',
        '| | irrational: False\n',
        '| | negative: False\n',
        '| | noninteger: False\n',
        '| | nonnegative: True\n',
        '| | rational: True\n',
        '| | real: True\n',
        '| | transcendental: False\n',
        '| +-Integer: 3\n',
        '|   algebraic: True\n',
        '|   commutative: True\n',
        '|   complex: True\n',
        '|   extended_negative: False\n',
        '|   extended_nonnegative: True\n',
        '|   extended_real: True\n',
        '|   finite: True\n',
        '|   hermitian: True\n',
        '|   imaginary: False\n',
        '|   infinite: False\n',
        '|   integer: True\n',
        '|   irrational: False\n',
        '|   negative: False\n',
        '|   noninteger: False\n',
        '|   nonnegative: True\n',
        '|   rational: True\n',
        '|   real: True\n',
        '|   transcendental: False\n',
        '+-MatrixSymbol: B\n',
        '  algebraic: False\n',
        '  commutative: False\n',
        '  complex: False\n',
        '  composite: False\n',
        '  even: False\n',
        '  extended_negative: False\n',
        '  extended_nonnegative: False\n',
        '  extended_nonpositive: False\n',
        '  extended_nonzero: False\n',
        '  extended_positive: False\n',
        '  extended_real: False\n',
        '  imaginary: False\n',
        '  integer: False\n',
        '  irrational: False\n',
        '  negative: False\n',
        '  noninteger: False\n',
        '  nonnegative: False\n',
        '  nonpositive: False\n',
        '  nonzero: False\n',
        '  odd: False\n',
        '  positive: False\n',
        '  prime: False\n',
        '  rational: False\n',
        '  real: False\n',
        '  transcendental: False\n',
        '  zero: False\n',
        '  +-Symbol: B\n',
        '  | commutative: True\n',
        '  +-Integer: 3\n',
        '  | algebraic: True\n',
        '  | commutative: True\n',
        '  | complex: True\n',
        '  | extended_negative: False\n',
        '  | extended_nonnegative: True\n',
        '  | extended_real: True\n',
        '  | finite: True\n',
        '  | hermitian: True\n',
        '  | imaginary: False\n',
        '  | infinite: False\n',
        '  | integer: True\n',
        '  | irrational: False\n',
        '  | negative: False\n',
        '  | noninteger: False\n',
        '  | nonnegative: True\n',
        '  | rational: True\n',
        '  | real: True\n',
        '  | transcendental: False\n',
        '  +-Integer: 3\n',
        '    algebraic: True\n',
        '    commutative: True\n',
        '    complex: True\n',
        '    extended_negative: False\n',
        '    extended_nonnegative: True\n',
        '    extended_real: True\n',
        '    finite: True\n',
        '    hermitian: True\n',
        '    imaginary: False\n',
        '    infinite: False\n',
        '    integer: True\n',
        '    irrational: False\n',
        '    negative: False\n',
        '    noninteger: False\n',
        '    nonnegative: True\n',
        '    rational: True\n',
        '    real: True\n',
        '    transcendental: False\n'
    ]

    assert tree(A + B) == "".join(test_str)


def test_print_tree_MatAdd_noassumptions():
    from sympy.matrices.expressions import MatrixSymbol
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)

    test_str = \
"""MatAdd: A + B
+-MatrixSymbol: A
| +-Str: A
| +-Integer: 3
| +-Integer: 3
+-MatrixSymbol: B
  +-Str: B
  +-Integer: 3
  +-Integer: 3
"""

    assert tree(A + B, assumptions=False) == test_str

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_unary.py ===
from decimal import Decimal

import numpy as np
import pytest

from pandas.compat.numpy import np_version_gte1p25

import pandas as pd
import pandas._testing as tm


class TestDataFrameUnaryOperators:
    # __pos__, __neg__, __invert__

    @pytest.mark.parametrize(
        "df,expected",
        [
            (pd.DataFrame({"a": [-1, 1]}), pd.DataFrame({"a": [1, -1]})),
            (pd.DataFrame({"a": [False, True]}), pd.DataFrame({"a": [True, False]})),
            (
                pd.DataFrame({"a": pd.Series(pd.to_timedelta([-1, 1]))}),
                pd.DataFrame({"a": pd.Series(pd.to_timedelta([1, -1]))}),
            ),
        ],
    )
    def test_neg_numeric(self, df, expected):
        tm.assert_frame_equal(-df, expected)
        tm.assert_series_equal(-df["a"], expected["a"])

    @pytest.mark.parametrize(
        "df, expected",
        [
            (np.array([1, 2], dtype=object), np.array([-1, -2], dtype=object)),
            ([Decimal("1.0"), Decimal("2.0")], [Decimal("-1.0"), Decimal("-2.0")]),
        ],
    )
    def test_neg_object(self, df, expected):
        # GH#21380
        df = pd.DataFrame({"a": df})
        expected = pd.DataFrame({"a": expected})
        tm.assert_frame_equal(-df, expected)
        tm.assert_series_equal(-df["a"], expected["a"])

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame({"a": ["a", "b"]}),
            pd.DataFrame({"a": pd.to_datetime(["2017-01-22", "1970-01-01"])}),
        ],
    )
    def test_neg_raises(self, df, using_infer_string):
        msg = (
            "bad operand type for unary -: 'str'|"
            r"bad operand type for unary -: 'DatetimeArray'|"
            "unary '-' not supported for dtype"
        )
        with pytest.raises(TypeError, match=msg):
            (-df)
        with pytest.raises(TypeError, match=msg):
            (-df["a"])

    def test_invert(self, float_frame):
        df = float_frame

        tm.assert_frame_equal(-(df < 0), ~(df < 0))

    def test_invert_mixed(self):
        shape = (10, 5)
        df = pd.concat(
            [
                pd.DataFrame(np.zeros(shape, dtype="bool")),
                pd.DataFrame(np.zeros(shape, dtype=int)),
            ],
            axis=1,
            ignore_index=True,
        )
        result = ~df
        expected = pd.concat(
            [
                pd.DataFrame(np.ones(shape, dtype="bool")),
                pd.DataFrame(-np.ones(shape, dtype=int)),
            ],
            axis=1,
            ignore_index=True,
        )
        tm.assert_frame_equal(result, expected)

    def test_invert_empty_not_input(self):
        # GH#51032
        df = pd.DataFrame()
        result = ~df
        tm.assert_frame_equal(df, result)
        assert df is not result

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame({"a": [-1, 1]}),
            pd.DataFrame({"a": [False, True]}),
            pd.DataFrame({"a": pd.Series(pd.to_timedelta([-1, 1]))}),
        ],
    )
    def test_pos_numeric(self, df):
        # GH#16073
        tm.assert_frame_equal(+df, df)
        tm.assert_series_equal(+df["a"], df["a"])

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame({"a": np.array([-1, 2], dtype=object)}),
            pd.DataFrame({"a": [Decimal("-1.0"), Decimal("2.0")]}),
        ],
    )
    def test_pos_object(self, df):
        # GH#21380
        tm.assert_frame_equal(+df, df)
        tm.assert_series_equal(+df["a"], df["a"])

    @pytest.mark.parametrize(
        "df",
        [
            pytest.param(
                pd.DataFrame({"a": ["a", "b"]}),
                # filterwarnings removable once min numpy version is 1.25
                marks=[
                    pytest.mark.filterwarnings("ignore:Applying:DeprecationWarning")
                ],
            ),
        ],
    )
    def test_pos_object_raises(self, df):
        # GH#21380
        if np_version_gte1p25:
            with pytest.raises(
                TypeError, match=r"^bad operand type for unary \+: \'str\'$"
            ):
                tm.assert_frame_equal(+df, df)
        else:
            tm.assert_series_equal(+df["a"], df["a"])

    @pytest.mark.parametrize(
        "df", [pd.DataFrame({"a": pd.to_datetime(["2017-01-22", "1970-01-01"])})]
    )
    def test_pos_raises(self, df):
        msg = r"bad operand type for unary \+: 'DatetimeArray'"
        with pytest.raises(TypeError, match=msg):
            (+df)
        with pytest.raises(TypeError, match=msg):
            (+df["a"])

    def test_unary_nullable(self):
        df = pd.DataFrame(
            {
                "a": pd.array([1, -2, 3, pd.NA], dtype="Int64"),
                "b": pd.array([4.0, -5.0, 6.0, pd.NA], dtype="Float32"),
                "c": pd.array([True, False, False, pd.NA], dtype="boolean"),
                # include numpy bool to make sure bool-vs-boolean behavior
                #  is consistent in non-NA locations
                "d": np.array([True, False, False, True]),
            }
        )

        result = +df
        res_ufunc = np.positive(df)
        expected = df
        # TODO: assert that we have copies?
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(res_ufunc, expected)

        result = -df
        res_ufunc = np.negative(df)
        expected = pd.DataFrame(
            {
                "a": pd.array([-1, 2, -3, pd.NA], dtype="Int64"),
                "b": pd.array([-4.0, 5.0, -6.0, pd.NA], dtype="Float32"),
                "c": pd.array([False, True, True, pd.NA], dtype="boolean"),
                "d": np.array([False, True, True, False]),
            }
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(res_ufunc, expected)

        result = abs(df)
        res_ufunc = np.abs(df)
        expected = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3, pd.NA], dtype="Int64"),
                "b": pd.array([4.0, 5.0, 6.0, pd.NA], dtype="Float32"),
                "c": pd.array([True, False, False, pd.NA], dtype="boolean"),
                "d": np.array([True, False, False, True]),
            }
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(res_ufunc, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_dialect.py ===
"""
Tests that dialects are properly handled during parsing
for all of the parsers defined in parsers.py
"""

import csv
from io import StringIO

import pytest

from pandas.errors import ParserWarning

from pandas import DataFrame
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def custom_dialect():
    dialect_name = "weird"
    dialect_kwargs = {
        "doublequote": False,
        "escapechar": "~",
        "delimiter": ":",
        "skipinitialspace": False,
        "quotechar": "`",
        "quoting": 3,
    }
    return dialect_name, dialect_kwargs


def test_dialect(all_parsers):
    parser = all_parsers
    data = """\
label1,label2,label3
index1,"a,c,e
index2,b,d,f
"""

    dia = csv.excel()
    dia.quoting = csv.QUOTE_NONE

    if parser.engine == "pyarrow":
        msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dialect=dia)
        return

    df = parser.read_csv(StringIO(data), dialect=dia)

    data = """\
label1,label2,label3
index1,a,c,e
index2,b,d,f
"""
    exp = parser.read_csv(StringIO(data))
    exp.replace("a", '"a', inplace=True)
    tm.assert_frame_equal(df, exp)


def test_dialect_str(all_parsers):
    dialect_name = "mydialect"
    parser = all_parsers
    data = """\
fruit:vegetable
apple:broccoli
pear:tomato
"""
    exp = DataFrame({"fruit": ["apple", "pear"], "vegetable": ["broccoli", "tomato"]})

    with tm.with_csv_dialect(dialect_name, delimiter=":"):
        if parser.engine == "pyarrow":
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), dialect=dialect_name)
            return

        df = parser.read_csv(StringIO(data), dialect=dialect_name)
        tm.assert_frame_equal(df, exp)


def test_invalid_dialect(all_parsers):
    class InvalidDialect:
        pass

    data = "a\n1"
    parser = all_parsers
    msg = "Invalid dialect"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), dialect=InvalidDialect)


@pytest.mark.parametrize(
    "arg",
    [None, "doublequote", "escapechar", "skipinitialspace", "quotechar", "quoting"],
)
@pytest.mark.parametrize("value", ["dialect", "default", "other"])
def test_dialect_conflict_except_delimiter(all_parsers, custom_dialect, arg, value):
    # see gh-23761.
    dialect_name, dialect_kwargs = custom_dialect
    parser = all_parsers

    expected = DataFrame({"a": [1], "b": [2]})
    data = "a:b\n1:2"

    warning_klass = None
    kwds = {}

    # arg=None tests when we pass in the dialect without any other arguments.
    if arg is not None:
        if value == "dialect":  # No conflict --> no warning.
            kwds[arg] = dialect_kwargs[arg]
        elif value == "default":  # Default --> no warning.
            from pandas.io.parsers.base_parser import parser_defaults

            kwds[arg] = parser_defaults[arg]
        else:  # Non-default + conflict with dialect --> warning.
            warning_klass = ParserWarning
            kwds[arg] = "blah"

    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        if parser.engine == "pyarrow":
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv_check_warnings(
                    # No warning bc we raise
                    None,
                    "Conflicting values for",
                    StringIO(data),
                    dialect=dialect_name,
                    **kwds,
                )
            return
        result = parser.read_csv_check_warnings(
            warning_klass,
            "Conflicting values for",
            StringIO(data),
            dialect=dialect_name,
            **kwds,
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,warning_klass",
    [
        ({"sep": ","}, None),  # sep is default --> sep_override=True
        ({"sep": "."}, ParserWarning),  # sep isn't default --> sep_override=False
        ({"delimiter": ":"}, None),  # No conflict
        ({"delimiter": None}, None),  # Default arguments --> sep_override=True
        ({"delimiter": ","}, ParserWarning),  # Conflict
        ({"delimiter": "."}, ParserWarning),  # Conflict
    ],
    ids=[
        "sep-override-true",
        "sep-override-false",
        "delimiter-no-conflict",
        "delimiter-default-arg",
        "delimiter-conflict",
        "delimiter-conflict2",
    ],
)
def test_dialect_conflict_delimiter(all_parsers, custom_dialect, kwargs, warning_klass):
    # see gh-23761.
    dialect_name, dialect_kwargs = custom_dialect
    parser = all_parsers

    expected = DataFrame({"a": [1], "b": [2]})
    data = "a:b\n1:2"

    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        if parser.engine == "pyarrow":
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv_check_warnings(
                    # no warning bc we raise
                    None,
                    "Conflicting values for 'delimiter'",
                    StringIO(data),
                    dialect=dialect_name,
                    **kwargs,
                )
            return
        result = parser.read_csv_check_warnings(
            warning_klass,
            "Conflicting values for 'delimiter'",
            StringIO(data),
            dialect=dialect_name,
            **kwargs,
        )
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_complex.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.tests.io.pytables.common import ensure_clean_store

from pandas.io.pytables import read_hdf


def test_complex_fixed(tmp_path, setup_path):
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)).astype(np.complex64),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    path = tmp_path / setup_path
    df.to_hdf(path, key="df")
    reread = read_hdf(path, "df")
    tm.assert_frame_equal(df, reread)

    df = DataFrame(
        np.random.default_rng(2).random((4, 5)).astype(np.complex128),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    path = tmp_path / setup_path
    df.to_hdf(path, key="df")
    reread = read_hdf(path, "df")
    tm.assert_frame_equal(df, reread)


def test_complex_table(tmp_path, setup_path):
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)).astype(np.complex64),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table")
    reread = read_hdf(path, key="df")
    tm.assert_frame_equal(df, reread)

    df = DataFrame(
        np.random.default_rng(2).random((4, 5)).astype(np.complex128),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table", mode="w")
    reread = read_hdf(path, "df")
    tm.assert_frame_equal(df, reread)


def test_complex_mixed_fixed(tmp_path, setup_path):
    complex64 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex64
    )
    complex128 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex128
    )
    df = DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "d"],
            "C": complex64,
            "D": complex128,
            "E": [1.0, 2.0, 3.0, 4.0],
        },
        index=list("abcd"),
    )
    path = tmp_path / setup_path
    df.to_hdf(path, key="df")
    reread = read_hdf(path, "df")
    tm.assert_frame_equal(df, reread)


def test_complex_mixed_table(tmp_path, setup_path):
    complex64 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex64
    )
    complex128 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex128
    )
    df = DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "d"],
            "C": complex64,
            "D": complex128,
            "E": [1.0, 2.0, 3.0, 4.0],
        },
        index=list("abcd"),
    )

    with ensure_clean_store(setup_path) as store:
        store.append("df", df, data_columns=["A", "B"])
        result = store.select("df", where="A>2")
        tm.assert_frame_equal(df.loc[df.A > 2], result)

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table")
    reread = read_hdf(path, "df")
    tm.assert_frame_equal(df, reread)


def test_complex_across_dimensions_fixed(tmp_path, setup_path):
    complex128 = np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
    s = Series(complex128, index=list("abcd"))
    df = DataFrame({"A": s, "B": s})

    objs = [s, df]
    comps = [tm.assert_series_equal, tm.assert_frame_equal]
    for obj, comp in zip(objs, comps):
        path = tmp_path / setup_path
        obj.to_hdf(path, key="obj", format="fixed")
        reread = read_hdf(path, "obj")
        comp(obj, reread)


def test_complex_across_dimensions(tmp_path, setup_path):
    complex128 = np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
    s = Series(complex128, index=list("abcd"))
    df = DataFrame({"A": s, "B": s})

    path = tmp_path / setup_path
    df.to_hdf(path, key="obj", format="table")
    reread = read_hdf(path, "obj")
    tm.assert_frame_equal(df, reread)


def test_complex_indexing_error(setup_path):
    complex128 = np.array(
        [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex128
    )
    df = DataFrame(
        {"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"], "C": complex128},
        index=list("abcd"),
    )

    msg = (
        "Columns containing complex values can be stored "
        "but cannot be indexed when using table format. "
        "Either use fixed format, set index=False, "
        "or do not include the columns containing complex "
        "values to data_columns when initializing the table."
    )

    with ensure_clean_store(setup_path) as store:
        with pytest.raises(TypeError, match=msg):
            store.append("df", df, data_columns=["C"])


def test_complex_series_error(tmp_path, setup_path):
    complex128 = np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
    s = Series(complex128, index=list("abcd"))

    msg = (
        "Columns containing complex values can be stored "
        "but cannot be indexed when using table format. "
        "Either use fixed format, set index=False, "
        "or do not include the columns containing complex "
        "values to data_columns when initializing the table."
    )

    path = tmp_path / setup_path
    with pytest.raises(TypeError, match=msg):
        s.to_hdf(path, key="obj", format="t")

    path = tmp_path / setup_path
    s.to_hdf(path, key="obj", format="t", index=False)
    reread = read_hdf(path, "obj")
    tm.assert_series_equal(s, reread)


def test_complex_append(setup_path):
    df = DataFrame(
        {
            "a": np.random.default_rng(2).standard_normal(100).astype(np.complex128),
            "b": np.random.default_rng(2).standard_normal(100),
        }
    )

    with ensure_clean_store(setup_path) as store:
        store.append("df", df, data_columns=["b"])
        store.append("df", df)
        result = store.select("df")
        tm.assert_frame_equal(pd.concat([df, df], axis=0), result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_implicit_multiplication_application.py ===
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
    implicit_multiplication_application,
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
    split_symbols,
    split_symbols_custom,
    _token_splittable
)
from sympy.testing.pytest import raises


def test_implicit_multiplication():
    cases = {
        '5x': '5*x',
        'abc': 'a*b*c',
        '3sin(x)': '3*sin(x)',
        '(x+1)(x+2)': '(x+1)*(x+2)',
        '(5 x**2)sin(x)': '(5*x**2)*sin(x)',
        '2 sin(x) cos(x)': '2*sin(x)*cos(x)',
        'pi x': 'pi*x',
        'x pi': 'x*pi',
        'E x': 'E*x',
        'EulerGamma y': 'EulerGamma*y',
        'E pi': 'E*pi',
        'pi (x + 2)': 'pi*(x+2)',
        '(x + 2) pi': '(x+2)*pi',
        'pi sin(x)': 'pi*sin(x)',
    }
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (split_symbols,
                                          implicit_multiplication)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert(implicit == normal)

    application = ['sin x', 'cos 2*x', 'sin cos x']
    for case in application:
        raises(SyntaxError,
               lambda: parse_expr(case, transformations=transformations2))
    raises(TypeError,
           lambda: parse_expr('sin**2(x)', transformations=transformations2))


def test_implicit_application():
    cases = {
        'factorial': 'factorial',
        'sin x': 'sin(x)',
        'tan y**3': 'tan(y**3)',
        'cos 2*x': 'cos(2*x)',
        '(cot)': 'cot',
        'sin cos tan x': 'sin(cos(tan(x)))'
    }
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (implicit_application,)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert(implicit == normal), (implicit, normal)

    multiplication = ['x y', 'x sin x', '2x']
    for case in multiplication:
        raises(SyntaxError,
               lambda: parse_expr(case, transformations=transformations2))
    raises(TypeError,
           lambda: parse_expr('sin**2(x)', transformations=transformations2))


def test_function_exponentiation():
    cases = {
        'sin**2(x)': 'sin(x)**2',
        'exp^y(z)': 'exp(z)^y',
        'sin**2(E^(x))': 'sin(E^(x))**2'
    }
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (function_exponentiation,)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert(implicit == normal)

    other_implicit = ['x y', 'x sin x', '2x', 'sin x',
                      'cos 2*x', 'sin cos x']
    for case in other_implicit:
        raises(SyntaxError,
               lambda: parse_expr(case, transformations=transformations2))

    assert parse_expr('x**2', local_dict={ 'x': sympy.Symbol('x') },
                      transformations=transformations2) == parse_expr('x**2')


def test_symbol_splitting():
    # By default Greek letter names should not be split (lambda is a keyword
    # so skip it)
    transformations = standard_transformations + (split_symbols,)
    greek_letters = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
                     'eta', 'theta', 'iota', 'kappa', 'mu', 'nu', 'xi',
                     'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon',
                     'phi', 'chi', 'psi', 'omega')

    for letter in greek_letters:
        assert(parse_expr(letter, transformations=transformations) ==
               parse_expr(letter))

    # Make sure symbol splitting resolves names
    transformations += (implicit_multiplication,)
    local_dict = { 'e': sympy.E }
    cases = {
        'xe': 'E*x',
        'Iy': 'I*y',
        'ee': 'E*E',
    }
    for case, expected in cases.items():
        assert(parse_expr(case, local_dict=local_dict,
                          transformations=transformations) ==
               parse_expr(expected))

    # Make sure custom splitting works
    def can_split(symbol):
        if symbol not in ('unsplittable', 'names'):
            return _token_splittable(symbol)
        return False
    transformations = standard_transformations
    transformations += (split_symbols_custom(can_split),
                        implicit_multiplication)

    assert(parse_expr('unsplittable', transformations=transformations) ==
           parse_expr('unsplittable'))
    assert(parse_expr('names', transformations=transformations) ==
           parse_expr('names'))
    assert(parse_expr('xy', transformations=transformations) ==
           parse_expr('x*y'))
    for letter in greek_letters:
        assert(parse_expr(letter, transformations=transformations) ==
               parse_expr(letter))


def test_all_implicit_steps():
    cases = {
        '2x': '2*x',  # implicit multiplication
        'x y': 'x*y',
        'xy': 'x*y',
        'sin x': 'sin(x)',  # add parentheses
        '2sin x': '2*sin(x)',
        'x y z': 'x*y*z',
        'sin(2 * 3x)': 'sin(2 * 3 * x)',
        'sin(x) (1 + cos(x))': 'sin(x) * (1 + cos(x))',
        '(x + 2) sin(x)': '(x + 2) * sin(x)',
        '(x + 2) sin x': '(x + 2) * sin(x)',
        'sin(sin x)': 'sin(sin(x))',
        'sin x!': 'sin(factorial(x))',
        'sin x!!': 'sin(factorial2(x))',
        'factorial': 'factorial',  # don't apply a bare function
        'x sin x': 'x * sin(x)',  # both application and multiplication
        'xy sin x': 'x * y * sin(x)',
        '(x+2)(x+3)': '(x + 2) * (x+3)',
        'x**2 + 2xy + y**2': 'x**2 + 2 * x * y + y**2',  # split the xy
        'pi': 'pi',  # don't mess with constants
        'None': 'None',
        'ln sin x': 'ln(sin(x))',  # multiple implicit function applications
        'sin x**2': 'sin(x**2)',  # implicit application to an exponential
        'alpha': 'Symbol("alpha")',  # don't split Greek letters/subscripts
        'x_2': 'Symbol("x_2")',
        'sin^2 x**2': 'sin(x**2)**2',  # function raised to a power
        'sin**3(x)': 'sin(x)**3',
        '(factorial)': 'factorial',
        'tan 3x': 'tan(3*x)',
        'sin^2(3*E^(x))': 'sin(3*E**(x))**2',
        'sin**2(E^(3x))': 'sin(E**(3*x))**2',
        'sin^2 (3x*E^(x))': 'sin(3*x*E^x)**2',
        'pi sin x': 'pi*sin(x)',
    }
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (implicit_multiplication_application,)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert(implicit == normal)


def test_no_methods_implicit_multiplication():
    # Issue 21020
    u = sympy.Symbol('u')
    transformations = standard_transformations + \
                      (implicit_multiplication,)
    expr = parse_expr('x.is_polynomial(x)', transformations=transformations)
    assert expr == True
    expr = parse_expr('(exp(x) / (1 + exp(2x))).subs(exp(x), u)',
                      transformations=transformations)
    assert expr == u/(u**2 + 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\test_function.py ===
import numpy as np
import pytest

from pandas.compat import IS64

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize("ufunc", [np.abs, np.sign])
# np.sign emits a warning with nans, <https://github.com/numpy/numpy/issues/15127>
@pytest.mark.filterwarnings("ignore:invalid value encountered in sign:RuntimeWarning")
def test_ufuncs_single(ufunc):
    a = pd.array([1, 2, -3, np.nan], dtype="Float64")
    result = ufunc(a)
    expected = pd.array(ufunc(a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    result = ufunc(s)
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.log, np.exp, np.sin, np.cos, np.sqrt])
def test_ufuncs_single_float(ufunc):
    a = pd.array([1.0, 0.2, 3.0, np.nan], dtype="Float64")
    with np.errstate(invalid="ignore"):
        result = ufunc(a)
        expected = pd.array(ufunc(a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    with np.errstate(invalid="ignore"):
        result = ufunc(s)
        expected = pd.Series(ufunc(s.astype(float)), dtype="Float64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.add, np.subtract])
def test_ufuncs_binary_float(ufunc):
    # two FloatingArrays
    a = pd.array([1, 0.2, -3, np.nan], dtype="Float64")
    result = ufunc(a, a)
    expected = pd.array(ufunc(a.astype(float), a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    # FloatingArray with numpy array
    arr = np.array([1, 2, 3, 4])
    result = ufunc(a, arr)
    expected = pd.array(ufunc(a.astype(float), arr), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(arr, a)
    expected = pd.array(ufunc(arr, a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    # FloatingArray with scalar
    result = ufunc(a, 1)
    expected = pd.array(ufunc(a.astype(float), 1), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(1, a)
    expected = pd.array(ufunc(1, a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("values", [[0, 1], [0, None]])
def test_ufunc_reduce_raises(values):
    arr = pd.array(values, dtype="Float64")

    res = np.add.reduce(arr)
    expected = arr.sum(skipna=False)
    tm.assert_almost_equal(res, expected)


@pytest.mark.skipif(not IS64, reason="GH 36579: fail on 32-bit system")
@pytest.mark.parametrize(
    "pandasmethname, kwargs",
    [
        ("var", {"ddof": 0}),
        ("var", {"ddof": 1}),
        ("std", {"ddof": 0}),
        ("std", {"ddof": 1}),
        ("kurtosis", {}),
        ("skew", {}),
        ("sem", {}),
    ],
)
def test_stat_method(pandasmethname, kwargs):
    s = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, np.nan, np.nan], dtype="Float64")
    pandasmeth = getattr(s, pandasmethname)
    result = pandasmeth(**kwargs)
    s2 = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype="float64")
    pandasmeth = getattr(s2, pandasmethname)
    expected = pandasmeth(**kwargs)
    assert expected == result


def test_value_counts_na():
    arr = pd.array([0.1, 0.2, 0.1, pd.NA], dtype="Float64")
    result = arr.value_counts(dropna=False)
    idx = pd.Index([0.1, 0.2, pd.NA], dtype=arr.dtype)
    assert idx.dtype == arr.dtype
    expected = pd.Series([2, 1, 1], index=idx, dtype="Int64", name="count")
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([2, 1], index=idx[:-1], dtype="Int64", name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_empty():
    ser = pd.Series([], dtype="Float64")
    result = ser.value_counts()
    idx = pd.Index([], dtype="Float64")
    assert idx.dtype == "Float64"
    expected = pd.Series([], index=idx, dtype="Int64", name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize():
    ser = pd.Series([0.1, 0.2, 0.1, pd.NA], dtype="Float64")
    result = ser.value_counts(normalize=True)
    expected = pd.Series([2, 1], index=ser[:2], dtype="Float64", name="proportion") / 3
    assert expected.index.dtype == ser.dtype
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("min_count", [0, 4])
def test_floating_array_sum(skipna, min_count, dtype):
    arr = pd.array([1, 2, 3, None], dtype=dtype)
    result = arr.sum(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 6.0
    else:
        assert result is pd.NA


@pytest.mark.parametrize(
    "values, expected", [([1, 2, 3], 6.0), ([1, 2, 3, None], 6.0), ([None], 0.0)]
)
def test_floating_array_numpy_sum(values, expected):
    arr = pd.array(values, dtype="Float64")
    result = np.sum(arr)
    assert result == expected


@pytest.mark.parametrize("op", ["sum", "min", "max", "prod"])
def test_preserve_dtypes(op):
    df = pd.DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": pd.array([0.1, None, 3.0], dtype="Float64"),
        }
    )

    # op
    result = getattr(df.C, op)()
    assert isinstance(result, np.float64)

    # groupby
    result = getattr(df.groupby("A"), op)()

    expected = pd.DataFrame(
        {"B": np.array([1.0, 3.0]), "C": pd.array([0.1, 3], dtype="Float64")},
        index=pd.Index(["a", "b"], name="A"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("method", ["min", "max"])
def test_floating_array_min_max(skipna, method, dtype):
    arr = pd.array([0.0, 1.0, None], dtype=dtype)
    func = getattr(arr, method)
    result = func(skipna=skipna)
    if skipna:
        assert result == (0 if method == "min" else 1)
    else:
        assert result is pd.NA


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("min_count", [0, 9])
def test_floating_array_prod(skipna, min_count, dtype):
    arr = pd.array([1.0, 2.0, None], dtype=dtype)
    result = arr.prod(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 2
    else:
        assert result is pd.NA

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\usecols\test_parse_dates.py ===
"""
Tests the usecols functionality during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO

import pytest

from pandas import (
    DataFrame,
    Index,
    Timestamp,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")

_msg_pyarrow_requires_names = (
    "The pyarrow engine does not allow 'usecols' to be integer column "
    "positions. Pass a list of string column names instead."
)


@pytest.mark.parametrize("usecols", [[0, 2, 3], [3, 0, 2]])
def test_usecols_with_parse_dates(all_parsers, usecols):
    # see gh-9755
    data = """a,b,c,d,e
0,1,2014-01-01,09:00,4
0,1,2014-01-02,10:00,4"""
    parser = all_parsers
    parse_dates = [[1, 2]]

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )

    cols = {
        "a": [0, 0],
        "c_d": [Timestamp("2014-01-01 09:00:00"), Timestamp("2014-01-02 10:00:00")],
    }
    expected = DataFrame(cols, columns=["c_d", "a"])
    if parser.engine == "pyarrow":
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            with tm.assert_produces_warning(
                FutureWarning, match=depr_msg, check_stacklevel=False
            ):
                parser.read_csv(
                    StringIO(data), usecols=usecols, parse_dates=parse_dates
                )
        return
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data), usecols=usecols, parse_dates=parse_dates
        )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # pyarrow.lib.ArrowKeyError: Column 'fdate' in include_columns
def test_usecols_with_parse_dates2(all_parsers):
    # see gh-13604
    parser = all_parsers
    data = """2008-02-07 09:40,1032.43
2008-02-07 09:50,1042.54
2008-02-07 10:00,1051.65"""

    names = ["date", "values"]
    usecols = names[:]
    parse_dates = [0]

    index = Index(
        [
            Timestamp("2008-02-07 09:40"),
            Timestamp("2008-02-07 09:50"),
            Timestamp("2008-02-07 10:00"),
        ],
        name="date",
    )
    cols = {"values": [1032.43, 1042.54, 1051.65]}
    expected = DataFrame(cols, index=index)

    result = parser.read_csv(
        StringIO(data),
        parse_dates=parse_dates,
        index_col=0,
        usecols=usecols,
        header=None,
        names=names,
    )
    tm.assert_frame_equal(result, expected)


def test_usecols_with_parse_dates3(all_parsers):
    # see gh-14792
    parser = all_parsers
    data = """a,b,c,d,e,f,g,h,i,j
2016/09/21,1,1,2,3,4,5,6,7,8"""

    usecols = list("abcdefghij")
    parse_dates = [0]

    cols = {
        "a": Timestamp("2016-09-21").as_unit("ns"),
        "b": [1],
        "c": [1],
        "d": [2],
        "e": [3],
        "f": [4],
        "g": [5],
        "h": [6],
        "i": [7],
        "j": [8],
    }
    expected = DataFrame(cols, columns=usecols)

    result = parser.read_csv(StringIO(data), usecols=usecols, parse_dates=parse_dates)
    tm.assert_frame_equal(result, expected)


def test_usecols_with_parse_dates4(all_parsers):
    data = "a,b,c,d,e,f,g,h,i,j\n2016/09/21,1,1,2,3,4,5,6,7,8"
    usecols = list("abcdefghij")
    parse_dates = [[0, 1]]
    parser = all_parsers

    cols = {
        "a_b": "2016/09/21 1",
        "c": [1],
        "d": [2],
        "e": [3],
        "f": [4],
        "g": [5],
        "h": [6],
        "i": [7],
        "j": [8],
    }
    expected = DataFrame(cols, columns=["a_b"] + list("cdefghij"))

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        (FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data),
            usecols=usecols,
            parse_dates=parse_dates,
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("usecols", [[0, 2, 3], [3, 0, 2]])
@pytest.mark.parametrize(
    "names",
    [
        list("abcde"),  # Names span all columns in original data.
        list("acd"),  # Names span only the selected columns.
    ],
)
def test_usecols_with_parse_dates_and_names(all_parsers, usecols, names, request):
    # see gh-9755
    s = """0,1,2014-01-01,09:00,4
0,1,2014-01-02,10:00,4"""
    parse_dates = [[1, 2]]
    parser = all_parsers

    if parser.engine == "pyarrow" and not (len(names) == 3 and usecols[0] == 0):
        mark = pytest.mark.xfail(
            reason="Length mismatch in some cases, UserWarning in other"
        )
        request.applymarker(mark)

    cols = {
        "a": [0, 0],
        "c_d": [Timestamp("2014-01-01 09:00:00"), Timestamp("2014-01-02 10:00:00")],
    }
    expected = DataFrame(cols, columns=["c_d", "a"])

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        (FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(s), names=names, parse_dates=parse_dates, usecols=usecols
        )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\test_replace.py ===
from datetime import datetime

from dateutil.tz import gettz
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timestamp,
    conversion,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
import pandas.util._test_decorators as td

import pandas._testing as tm


class TestTimestampReplace:
    def test_replace_out_of_pydatetime_bounds(self):
        # GH#50348
        ts = Timestamp("2016-01-01").as_unit("ns")

        msg = "Out of bounds timestamp: 99999-01-01 00:00:00 with frequency 'ns'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            ts.replace(year=99_999)

        ts = ts.as_unit("ms")
        result = ts.replace(year=99_999)
        assert result.year == 99_999
        assert result._value == Timestamp(np.datetime64("99999-01-01", "ms"))._value

    def test_replace_non_nano(self):
        ts = Timestamp._from_value_and_reso(
            91514880000000000, NpyDatetimeUnit.NPY_FR_us.value, None
        )
        assert ts.to_pydatetime() == datetime(4869, 12, 28)

        result = ts.replace(year=4900)
        assert result._creso == ts._creso
        assert result.to_pydatetime() == datetime(4900, 12, 28)

    def test_replace_naive(self):
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00")
        result = ts.replace(hour=0)
        expected = Timestamp("2016-01-01 00:00:00")
        assert result == expected

    def test_replace_aware(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        # replacing datetime components with and w/o presence of a timezone
        ts = Timestamp("2016-01-01 09:00:00", tz=tz)
        result = ts.replace(hour=0)
        expected = Timestamp("2016-01-01 00:00:00", tz=tz)
        assert result == expected

    def test_replace_preserves_nanos(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        result = ts.replace(hour=0)
        expected = Timestamp("2016-01-01 00:00:00.000000123", tz=tz)
        assert result == expected

    def test_replace_multiple(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        # replacing datetime components with and w/o presence of a timezone
        # test all
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        result = ts.replace(
            year=2015,
            month=2,
            day=2,
            hour=0,
            minute=5,
            second=5,
            microsecond=5,
            nanosecond=5,
        )
        expected = Timestamp("2015-02-02 00:05:05.000005005", tz=tz)
        assert result == expected

    def test_replace_invalid_kwarg(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        msg = r"replace\(\) got an unexpected keyword argument"
        with pytest.raises(TypeError, match=msg):
            ts.replace(foo=5)

    def test_replace_integer_args(self, tz_aware_fixture):
        tz = tz_aware_fixture
        # GH#14621, GH#7825
        ts = Timestamp("2016-01-01 09:00:00.000000123", tz=tz)
        msg = "value must be an integer, received <class 'float'> for hour"
        with pytest.raises(ValueError, match=msg):
            ts.replace(hour=0.1)

    def test_replace_tzinfo_equiv_tz_localize_none(self):
        # GH#14621, GH#7825
        # assert conversion to naive is the same as replacing tzinfo with None
        ts = Timestamp("2013-11-03 01:59:59.999999-0400", tz="US/Eastern")
        assert ts.tz_localize(None) == ts.replace(tzinfo=None)

    @td.skip_if_windows
    def test_replace_tzinfo(self):
        # GH#15683
        dt = datetime(2016, 3, 27, 1)
        tzinfo = pytz.timezone("CET").localize(dt, is_dst=False).tzinfo

        result_dt = dt.replace(tzinfo=tzinfo)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo)

        # datetime.timestamp() converts in the local timezone
        with tm.set_timezone("UTC"):
            assert result_dt.timestamp() == result_pd.timestamp()

        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()

        result_dt = dt.replace(tzinfo=tzinfo).replace(tzinfo=None)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo).replace(tzinfo=None)

        # datetime.timestamp() converts in the local timezone
        with tm.set_timezone("UTC"):
            assert result_dt.timestamp() == result_pd.timestamp()

        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()

    @pytest.mark.parametrize(
        "tz, normalize",
        [
            (pytz.timezone("US/Eastern"), lambda x: x.tzinfo.normalize(x)),
            (gettz("US/Eastern"), lambda x: x),
        ],
    )
    def test_replace_across_dst(self, tz, normalize):
        # GH#18319 check that 1) timezone is correctly normalized and
        # 2) that hour is not incorrectly changed by this normalization
        ts_naive = Timestamp("2017-12-03 16:03:30")
        ts_aware = conversion.localize_pydatetime(ts_naive, tz)

        # Preliminary sanity-check
        assert ts_aware == normalize(ts_aware)

        # Replace across DST boundary
        ts2 = ts_aware.replace(month=6)

        # Check that `replace` preserves hour literal
        assert (ts2.hour, ts2.minute) == (ts_aware.hour, ts_aware.minute)

        # Check that post-replace object is appropriately normalized
        ts2b = normalize(ts2)
        assert ts2 == ts2b

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_replace_dst_border(self, unit):
        # Gh 7825
        t = Timestamp("2013-11-3", tz="America/Chicago").as_unit(unit)
        result = t.replace(hour=3)
        expected = Timestamp("2013-11-3 03:00:00", tz="America/Chicago")
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    @pytest.mark.parametrize("fold", [0, 1])
    @pytest.mark.parametrize("tz", ["dateutil/Europe/London", "Europe/London"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_replace_dst_fold(self, fold, tz, unit):
        # GH 25017
        d = datetime(2019, 10, 27, 2, 30)
        ts = Timestamp(d, tz=tz).as_unit(unit)
        result = ts.replace(hour=1, fold=fold)
        expected = Timestamp(datetime(2019, 10, 27, 1, 30)).tz_localize(
            tz, ambiguous=not fold
        )
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    @pytest.mark.parametrize("fold", [0, 1])
    def test_replace_preserves_fold(self, fold):
        # GH#37610. Check that replace preserves Timestamp fold property
        tz = gettz("Europe/Moscow")

        ts = Timestamp(
            year=2009, month=10, day=25, hour=2, minute=30, fold=fold, tzinfo=tz
        )
        ts_replaced = ts.replace(second=1)

        assert ts_replaced.fold == fold

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_matmul.py ===
from sympy.core import I, symbols, Basic, Mul, S
from sympy.core.mul import mul
from sympy.functions import adjoint, transpose
from sympy.matrices.exceptions import ShapeError
from sympy.matrices import (Identity, Inverse, Matrix, MatrixSymbol, ZeroMatrix,
        eye, ImmutableMatrix)
from sympy.matrices.expressions import Adjoint, Transpose, det, MatPow
from sympy.matrices.expressions.special import GenericIdentity
from sympy.matrices.expressions.matmul import (factor_in_front, remove_ids,
        MatMul, combine_powers, any_zeros, unpack, only_squares)
from sympy.strategies import null_safe
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.symbol import Symbol

from sympy.testing.pytest import XFAIL, raises

n, m, l, k = symbols('n m l k', integer=True)
x = symbols('x')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
E = MatrixSymbol('E', m, n)

def test_evaluate():
    assert MatMul(C, C, evaluate=True) == MatMul(C, C).doit()

def test_adjoint():
    assert adjoint(A*B) == Adjoint(B)*Adjoint(A)
    assert adjoint(2*A*B) == 2*Adjoint(B)*Adjoint(A)
    assert adjoint(2*I*C) == -2*I*Adjoint(C)

    M = Matrix(2, 2, [1, 2 + I, 3, 4])
    MA = Matrix(2, 2, [1, 3, 2 - I, 4])
    assert adjoint(M) == MA
    assert adjoint(2*M) == 2*MA
    assert adjoint(MatMul(2, M)) == MatMul(2, MA).doit()


def test_transpose():
    assert transpose(A*B) == Transpose(B)*Transpose(A)
    assert transpose(2*A*B) == 2*Transpose(B)*Transpose(A)
    assert transpose(2*I*C) == 2*I*Transpose(C)

    M = Matrix(2, 2, [1, 2 + I, 3, 4])
    MT = Matrix(2, 2, [1, 3, 2 + I, 4])
    assert transpose(M) == MT
    assert transpose(2*M) == 2*MT
    assert transpose(x*M) == x*MT
    assert transpose(MatMul(2, M)) == MatMul(2, MT).doit()


def test_factor_in_front():
    assert factor_in_front(MatMul(A, 2, B, evaluate=False)) ==\
                           MatMul(2, A, B, evaluate=False)


def test_remove_ids():
    assert remove_ids(MatMul(A, Identity(m), B, evaluate=False)) == \
                      MatMul(A, B, evaluate=False)
    assert null_safe(remove_ids)(MatMul(Identity(n), evaluate=False)) == \
                                 MatMul(Identity(n), evaluate=False)


def test_combine_powers():
    assert combine_powers(MatMul(D, Inverse(D), D, evaluate=False)) == \
                 MatMul(Identity(n), D, evaluate=False)
    assert combine_powers(MatMul(B.T, Inverse(E*A), E, A, B, evaluate=False)) == \
        MatMul(B.T, Identity(m), B, evaluate=False)
    assert combine_powers(MatMul(A, E, Inverse(A*E), D, evaluate=False)) == \
        MatMul(Identity(n), D, evaluate=False)


def test_any_zeros():
    assert any_zeros(MatMul(A, ZeroMatrix(m, k), evaluate=False)) == \
                     ZeroMatrix(n, k)


def test_unpack():
    assert unpack(MatMul(A, evaluate=False)) == A
    x = MatMul(A, B)
    assert unpack(x) == x


def test_only_squares():
    assert only_squares(C) == [C]
    assert only_squares(C, D) == [C, D]
    assert only_squares(C, A, A.T, D) == [C, A*A.T, D]


def test_determinant():
    assert det(2*C) == 2**n*det(C)
    assert det(2*C*D) == 2**n*det(C)*det(D)
    assert det(3*C*A*A.T*D) == 3**n*det(C)*det(A*A.T)*det(D)


def test_doit():
    assert MatMul(C, 2, D).args == (C, 2, D)
    assert MatMul(C, 2, D).doit().args == (2, C, D)
    assert MatMul(C, Transpose(D*C)).args == (C, Transpose(D*C))
    assert MatMul(C, Transpose(D*C)).doit(deep=True).args == (C, C.T, D.T)


def test_doit_drills_down():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[2, 3], [4, 5]])
    assert MatMul(X, MatPow(Y, 2)).doit() == X*Y**2
    assert MatMul(C, Transpose(D*C)).doit().args == (C, C.T, D.T)


def test_doit_deep_false_still_canonical():
    assert (MatMul(C, Transpose(D*C), 2).doit(deep=False).args ==
            (2, C, Transpose(D*C)))


def test_matmul_scalar_Matrix_doit():
    # Issue 9053
    X = Matrix([[1, 2], [3, 4]])
    assert MatMul(2, X).doit() == 2*X


def test_matmul_sympify():
    assert isinstance(MatMul(eye(1), eye(1)).args[0], Basic)


def test_collapse_MatrixBase():
    A = Matrix([[1, 1], [1, 1]])
    B = Matrix([[1, 2], [3, 4]])
    assert MatMul(A, B).doit() == ImmutableMatrix([[4, 6], [4, 6]])


def test_refine():
    assert refine(C*C.T*D, Q.orthogonal(C)).doit() == D

    kC = k*C
    assert refine(kC*C.T, Q.orthogonal(C)).doit() == k*Identity(n)
    assert refine(kC* kC.T, Q.orthogonal(C)).doit() == (k**2)*Identity(n)

def test_matmul_no_matrices():
    assert MatMul(1) == 1
    assert MatMul(n, m) == n*m
    assert not isinstance(MatMul(n, m), MatMul)

def test_matmul_args_cnc():
    assert MatMul(n, A, A.T).args_cnc() == [[n], [A, A.T]]
    assert MatMul(A, A.T).args_cnc() == [[], [A, A.T]]

@XFAIL
def test_matmul_args_cnc_symbols():
    # Not currently supported
    a, b = symbols('a b', commutative=False)
    assert MatMul(n, a, b, A, A.T).args_cnc() == [[n], [a, b, A, A.T]]
    assert MatMul(n, a, A, b, A.T).args_cnc() == [[n], [a, A, b, A.T]]

def test_issue_12950():
    M = Matrix([[Symbol("x")]]) * MatrixSymbol("A", 1, 1)
    assert MatrixSymbol("A", 1, 1).as_explicit()[0]*Symbol('x') == M.as_explicit()[0]

def test_construction_with_Mul():
    assert Mul(C, D) == MatMul(C, D)
    assert Mul(D, C) == MatMul(D, C)

def test_construction_with_mul():
    assert mul(C, D) == MatMul(C, D)
    assert mul(D, C) == MatMul(D, C)
    assert mul(C, D) != MatMul(D, C)

def test_generic_identity():
    assert MatMul.identity == GenericIdentity()
    assert MatMul.identity != S.One


def test_issue_23519():
    N = Symbol("N", integer=True)
    M1 = MatrixSymbol("M1", N, N)
    M2 = MatrixSymbol("M2", N, N)
    I = Identity(N)
    z = (M2 + 2 * (M2 + I) * M1 + I)
    assert z.coeff(M1) == 2*I + 2*M2


def test_shape_error():
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 3, 3)
    raises(ShapeError, lambda: MatMul(A, B))


def test_matmul_transpose():
    # https://github.com/sympy/sympy/issues/9503
    M = Matrix(2, 2, [1, 2 + I, 3, 4])
    a = Symbol('a')
    assert (MatMul(a, M).T).expand() == (a*Matrix([[1, 3],[2 + I, 4]])).expand()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_inverse.py ===
from sympy import ZZ, Matrix
from sympy.polys.matrices import DM, DomainMatrix
from sympy.polys.matrices.dense import ddm_iinv
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.matrices.exceptions import NonInvertibleMatrixError

import pytest
from sympy.testing.pytest import raises
from sympy.core.numbers import all_close

from sympy.abc import x


# Examples are given as adjugate matrix and determinant adj_det should match
# these exactly but inv_den only matches after cancel_denom.


INVERSE_EXAMPLES = [

    (
        'zz_1',
        DomainMatrix([], (0, 0), ZZ),
        DomainMatrix([], (0, 0), ZZ),
        ZZ(1),
    ),

    (
        'zz_2',
        DM([[2]], ZZ),
        DM([[1]], ZZ),
        ZZ(2),
    ),

    (
        'zz_3',
        DM([[2, 0],
            [0, 2]], ZZ),
        DM([[2, 0],
            [0, 2]], ZZ),
        ZZ(4),
    ),

    (
        'zz_4',
        DM([[1, 2],
            [3, 4]], ZZ),
        DM([[ 4, -2],
            [-3,  1]], ZZ),
        ZZ(-2),
    ),

    (
        'zz_5',
        DM([[2, 2, 0],
            [0, 2, 2],
            [0, 0, 2]], ZZ),
        DM([[4, -4, 4],
            [0, 4, -4],
            [0, 0,  4]], ZZ),
        ZZ(8),
    ),

    (
        'zz_6',
        DM([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], ZZ),
        DM([[-3,   6, -3],
            [ 6, -12,  6],
            [-3,   6, -3]], ZZ),
        ZZ(0),
    ),
]


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_Matrix_inv(name, A, A_inv, den):

    def _check(**kwargs):
        if den != 0:
            assert A.inv(**kwargs) == A_inv
        else:
            raises(NonInvertibleMatrixError, lambda: A.inv(**kwargs))

    K = A.domain
    A = A.to_Matrix()
    A_inv = A_inv.to_Matrix() / K.to_sympy(den)
    _check()
    for method in ['GE', 'LU', 'ADJ', 'CH', 'LDL', 'QR']:
        _check(method=method)


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dm_inv_den(name, A, A_inv, den):
    if den != 0:
        A_inv_f, den_f = A.inv_den()
        assert A_inv_f.cancel_denom(den_f) == A_inv.cancel_denom(den)
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv_den())


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dm_inv(name, A, A_inv, den):
    A = A.to_field()
    if den != 0:
        A_inv = A_inv.to_field() / den
        assert A.inv() == A_inv
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv())


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_ddm_inv(name, A, A_inv, den):
    A = A.to_field().to_ddm()
    if den != 0:
        A_inv = (A_inv.to_field() / den).to_ddm()
        assert A.inv() == A_inv
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv())


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_sdm_inv(name, A, A_inv, den):
    A = A.to_field().to_sdm()
    if den != 0:
        A_inv = (A_inv.to_field() / den).to_sdm()
        assert A.inv() == A_inv
    else:
        raises(DMNonInvertibleMatrixError, lambda: A.inv())


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dense_ddm_iinv(name, A, A_inv, den):
    A = A.to_field().to_ddm().copy()
    K = A.domain
    A_result = A.copy()
    if den != 0:
        A_inv = (A_inv.to_field() / den).to_ddm()
        ddm_iinv(A_result, A, K)
        assert A_result == A_inv
    else:
        raises(DMNonInvertibleMatrixError, lambda: ddm_iinv(A_result, A, K))


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_Matrix_adjugate(name, A, A_inv, den):
    A = A.to_Matrix()
    A_inv = A_inv.to_Matrix()
    assert A.adjugate() == A_inv
    for method in ["bareiss", "berkowitz", "bird", "laplace", "lu"]:
        assert A.adjugate(method=method) == A_inv


@pytest.mark.parametrize('name, A, A_inv, den', INVERSE_EXAMPLES)
def test_dm_adj_det(name, A, A_inv, den):
    assert A.adj_det() == (A_inv, den)


def test_inverse_inexact():

    M = Matrix([[x-0.3, -0.06, -0.22],
                [-0.46, x-0.48, -0.41],
                [-0.14, -0.39, x-0.64]])

    Mn = Matrix([[1.0*x**2 - 1.12*x + 0.1473, 0.06*x + 0.0474, 0.22*x - 0.081],
                 [0.46*x - 0.237, 1.0*x**2 - 0.94*x + 0.1612, 0.41*x - 0.0218],
                 [0.14*x + 0.1122, 0.39*x - 0.1086, 1.0*x**2 - 0.78*x + 0.1164]])

    d = 1.0*x**3 - 1.42*x**2 + 0.4249*x - 0.0546540000000002

    Mi = Mn / d

    M_dm = M.to_DM()
    M_dmd = M_dm.to_dense()
    M_dm_num, M_dm_den = M_dm.inv_den()
    M_dmd_num, M_dmd_den = M_dmd.inv_den()

    # XXX: We don't check M_dm().to_field().inv() which currently uses division
    # and produces a more complicate result from gcd cancellation failing.
    # DomainMatrix.inv() over RR(x) should be changed to clear denominators and
    # use DomainMatrix.inv_den().

    Minvs = [
        M.inv(),
        (M_dm_num.to_field() / M_dm_den).to_Matrix(),
        (M_dmd_num.to_field() / M_dmd_den).to_Matrix(),
        M_dm_num.to_Matrix() / M_dm_den.as_expr(),
        M_dmd_num.to_Matrix() / M_dmd_den.as_expr(),
    ]

    for Minv in Minvs:
        for Mi1, Mi2 in zip(Minv.flat(), Mi.flat()):
            assert all_close(Mi2, Mi1)