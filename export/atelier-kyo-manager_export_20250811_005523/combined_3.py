
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_nanfunctions.py ===
import inspect
import warnings
from functools import partial

import pytest

import numpy as np
from numpy._core.numeric import normalize_axis_tuple
from numpy.exceptions import AxisError, ComplexWarning
from numpy.lib._nanfunctions_impl import _nan_mask, _replace_nan
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    suppress_warnings,
)

# Test data
_ndat = np.array([[0.6244, np.nan, 0.2692, 0.0116, np.nan, 0.1170],
                  [0.5351, -0.9403, np.nan, 0.2100, 0.4759, 0.2833],
                  [np.nan, np.nan, np.nan, 0.1042, np.nan, -0.5954],
                  [0.1610, np.nan, np.nan, 0.1859, 0.3146, np.nan]])


# Rows of _ndat with nans removed
_rdat = [np.array([0.6244, 0.2692, 0.0116, 0.1170]),
         np.array([0.5351, -0.9403, 0.2100, 0.4759, 0.2833]),
         np.array([0.1042, -0.5954]),
         np.array([0.1610, 0.1859, 0.3146])]

# Rows of _ndat with nans converted to ones
_ndat_ones = np.array([[0.6244, 1.0, 0.2692, 0.0116, 1.0, 0.1170],
                       [0.5351, -0.9403, 1.0, 0.2100, 0.4759, 0.2833],
                       [1.0, 1.0, 1.0, 0.1042, 1.0, -0.5954],
                       [0.1610, 1.0, 1.0, 0.1859, 0.3146, 1.0]])

# Rows of _ndat with nans converted to zeros
_ndat_zeros = np.array([[0.6244, 0.0, 0.2692, 0.0116, 0.0, 0.1170],
                        [0.5351, -0.9403, 0.0, 0.2100, 0.4759, 0.2833],
                        [0.0, 0.0, 0.0, 0.1042, 0.0, -0.5954],
                        [0.1610, 0.0, 0.0, 0.1859, 0.3146, 0.0]])


class TestSignatureMatch:
    NANFUNCS = {
        np.nanmin: np.amin,
        np.nanmax: np.amax,
        np.nanargmin: np.argmin,
        np.nanargmax: np.argmax,
        np.nansum: np.sum,
        np.nanprod: np.prod,
        np.nancumsum: np.cumsum,
        np.nancumprod: np.cumprod,
        np.nanmean: np.mean,
        np.nanmedian: np.median,
        np.nanpercentile: np.percentile,
        np.nanquantile: np.quantile,
        np.nanvar: np.var,
        np.nanstd: np.std,
    }
    IDS = [k.__name__ for k in NANFUNCS]

    @staticmethod
    def get_signature(func, default="..."):
        """Construct a signature and replace all default parameter-values."""
        prm_list = []
        signature = inspect.signature(func)
        for prm in signature.parameters.values():
            if prm.default is inspect.Parameter.empty:
                prm_list.append(prm)
            else:
                prm_list.append(prm.replace(default=default))
        return inspect.Signature(prm_list)

    @pytest.mark.parametrize("nan_func,func", NANFUNCS.items(), ids=IDS)
    def test_signature_match(self, nan_func, func):
        # Ignore the default parameter-values as they can sometimes differ
        # between the two functions (*e.g.* one has `False` while the other
        # has `np._NoValue`)
        signature = self.get_signature(func)
        nan_signature = self.get_signature(nan_func)
        np.testing.assert_equal(signature, nan_signature)

    def test_exhaustiveness(self):
        """Validate that all nan functions are actually tested."""
        np.testing.assert_equal(
            set(self.IDS), set(np.lib._nanfunctions_impl.__all__)
        )


class TestNanFunctions_MinMax:

    nanfuncs = [np.nanmin, np.nanmax]
    stdfuncs = [np.min, np.max]

    def test_mutation(self):
        # Check that passed array is not modified.
        ndat = _ndat.copy()
        for f in self.nanfuncs:
            f(ndat)
            assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for axis in [None, 0, 1]:
                tgt = rf(mat, axis=axis, keepdims=True)
                res = nf(mat, axis=axis, keepdims=True)
                assert_(res.ndim == tgt.ndim)

    def test_out(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            resout = np.zeros(3)
            tgt = rf(mat, axis=1)
            res = nf(mat, axis=1, out=resout)
            assert_almost_equal(res, resout)
            assert_almost_equal(res, tgt)

    def test_dtype_from_input(self):
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                mat = np.eye(3, dtype=c)
                tgt = rf(mat, axis=1).dtype.type
                res = nf(mat, axis=1).dtype.type
                assert_(res is tgt)
                # scalar case
                tgt = rf(mat, axis=None).dtype.type
                res = nf(mat, axis=None).dtype.type
                assert_(res is tgt)

    def test_result_values(self):
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            tgt = [rf(d) for d in _rdat]
            res = nf(_ndat, axis=1)
            assert_almost_equal(res, tgt)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip("`axis != None` not supported for 0d arrays")

        array = array.astype(dtype)
        match = "All-NaN slice encountered"
        for func in self.nanfuncs:
            with pytest.warns(RuntimeWarning, match=match):
                out = func(array, axis=axis)
            assert np.isnan(out).all()
            assert out.dtype == array.dtype

    def test_masked(self):
        mat = np.ma.fix_invalid(_ndat)
        msk = mat._mask.copy()
        for f in [np.nanmin]:
            res = f(mat, axis=1)
            tgt = f(_ndat, axis=1)
            assert_equal(res, tgt)
            assert_equal(mat._mask, msk)
            assert_(not np.isinf(mat).any())

    def test_scalar(self):
        for f in self.nanfuncs:
            assert_(f(0.) == 0.)

    def test_subclass(self):
        class MyNDArray(np.ndarray):
            pass

        # Check that it works and that type and
        # shape are preserved
        mine = np.eye(3).view(MyNDArray)
        for f in self.nanfuncs:
            res = f(mine, axis=0)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))
            res = f(mine, axis=1)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))
            res = f(mine)
            assert_(res.shape == ())

        # check that rows of nan are dealt with for subclasses (#4628)
        mine[1] = np.nan
        for f in self.nanfuncs:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                res = f(mine, axis=0)
                assert_(isinstance(res, MyNDArray))
                assert_(not np.any(np.isnan(res)))
                assert_(len(w) == 0)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                res = f(mine, axis=1)
                assert_(isinstance(res, MyNDArray))
                assert_(np.isnan(res[1]) and not np.isnan(res[0])
                        and not np.isnan(res[2]))
                assert_(len(w) == 1, 'no warning raised')
                assert_(issubclass(w[0].category, RuntimeWarning))

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                res = f(mine)
                assert_(res.shape == ())
                assert_(res != np.nan)
                assert_(len(w) == 0)

    def test_object_array(self):
        arr = np.array([[1.0, 2.0], [np.nan, 4.0], [np.nan, np.nan]], dtype=object)
        assert_equal(np.nanmin(arr), 1.0)
        assert_equal(np.nanmin(arr, axis=0), [1.0, 2.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # assert_equal does not work on object arrays of nan
            assert_equal(list(np.nanmin(arr, axis=1)), [1.0, 4.0, np.nan])
            assert_(len(w) == 1, 'no warning raised')
            assert_(issubclass(w[0].category, RuntimeWarning))

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_initial(self, dtype):
        class MyNDArray(np.ndarray):
            pass

        ar = np.arange(9).astype(dtype)
        ar[:5] = np.nan

        for f in self.nanfuncs:
            initial = 100 if f is np.nanmax else 0

            ret1 = f(ar, initial=initial)
            assert ret1.dtype == dtype
            assert ret1 == initial

            ret2 = f(ar.view(MyNDArray), initial=initial)
            assert ret2.dtype == dtype
            assert ret2 == initial

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_where(self, dtype):
        class MyNDArray(np.ndarray):
            pass

        ar = np.arange(9).reshape(3, 3).astype(dtype)
        ar[0, :] = np.nan
        where = np.ones_like(ar, dtype=np.bool)
        where[:, 0] = False

        for f in self.nanfuncs:
            reference = 4 if f is np.nanmin else 8

            ret1 = f(ar, where=where, initial=5)
            assert ret1.dtype == dtype
            assert ret1 == reference

            ret2 = f(ar.view(MyNDArray), where=where, initial=5)
            assert ret2.dtype == dtype
            assert ret2 == reference


class TestNanFunctions_ArgminArgmax:

    nanfuncs = [np.nanargmin, np.nanargmax]

    def test_mutation(self):
        # Check that passed array is not modified.
        ndat = _ndat.copy()
        for f in self.nanfuncs:
            f(ndat)
            assert_equal(ndat, _ndat)

    def test_result_values(self):
        for f, fcmp in zip(self.nanfuncs, [np.greater, np.less]):
            for row in _ndat:
                with suppress_warnings() as sup:
                    sup.filter(RuntimeWarning, "invalid value encountered in")
                    ind = f(row)
                    val = row[ind]
                    # comparing with NaN is tricky as the result
                    # is always false except for NaN != NaN
                    assert_(not np.isnan(val))
                    assert_(not fcmp(val, row).any())
                    assert_(not np.equal(val, row[:ind]).any())

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip("`axis != None` not supported for 0d arrays")

        array = array.astype(dtype)
        for func in self.nanfuncs:
            with pytest.raises(ValueError, match="All-NaN slice encountered"):
                func(array, axis=axis)

    def test_empty(self):
        mat = np.zeros((0, 3))
        for f in self.nanfuncs:
            for axis in [0, None]:
                assert_raises_regex(
                        ValueError,
                        "attempt to get argm.. of an empty sequence",
                        f, mat, axis=axis)
            for axis in [1]:
                res = f(mat, axis=axis)
                assert_equal(res, np.zeros(0))

    def test_scalar(self):
        for f in self.nanfuncs:
            assert_(f(0.) == 0.)

    def test_subclass(self):
        class MyNDArray(np.ndarray):
            pass

        # Check that it works and that type and
        # shape are preserved
        mine = np.eye(3).view(MyNDArray)
        for f in self.nanfuncs:
            res = f(mine, axis=0)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))
            res = f(mine, axis=1)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == (3,))
            res = f(mine)
            assert_(res.shape == ())

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_keepdims(self, dtype):
        ar = np.arange(9).astype(dtype)
        ar[:5] = np.nan

        for f in self.nanfuncs:
            reference = 5 if f is np.nanargmin else 8
            ret = f(ar, keepdims=True)
            assert ret.ndim == ar.ndim
            assert ret == reference

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_out(self, dtype):
        ar = np.arange(9).astype(dtype)
        ar[:5] = np.nan

        for f in self.nanfuncs:
            out = np.zeros((), dtype=np.intp)
            reference = 5 if f is np.nanargmin else 8
            ret = f(ar, out=out)
            assert ret is out
            assert ret == reference


_TEST_ARRAYS = {
    "0d": np.array(5),
    "1d": np.array([127, 39, 93, 87, 46])
}
for _v in _TEST_ARRAYS.values():
    _v.setflags(write=False)


@pytest.mark.parametrize(
    "dtype",
    np.typecodes["AllInteger"] + np.typecodes["AllFloat"] + "O",
)
@pytest.mark.parametrize("mat", _TEST_ARRAYS.values(), ids=_TEST_ARRAYS.keys())
class TestNanFunctions_NumberTypes:
    nanfuncs = {
        np.nanmin: np.min,
        np.nanmax: np.max,
        np.nanargmin: np.argmin,
        np.nanargmax: np.argmax,
        np.nansum: np.sum,
        np.nanprod: np.prod,
        np.nancumsum: np.cumsum,
        np.nancumprod: np.cumprod,
        np.nanmean: np.mean,
        np.nanmedian: np.median,
        np.nanvar: np.var,
        np.nanstd: np.std,
    }
    nanfunc_ids = [i.__name__ for i in nanfuncs]

    @pytest.mark.parametrize("nanfunc,func", nanfuncs.items(), ids=nanfunc_ids)
    @np.errstate(over="ignore")
    def test_nanfunc(self, mat, dtype, nanfunc, func):
        mat = mat.astype(dtype)
        tgt = func(mat)
        out = nanfunc(mat)

        assert_almost_equal(out, tgt)
        if dtype == "O":
            assert type(out) is type(tgt)
        else:
            assert out.dtype == tgt.dtype

    @pytest.mark.parametrize(
        "nanfunc,func",
        [(np.nanquantile, np.quantile), (np.nanpercentile, np.percentile)],
        ids=["nanquantile", "nanpercentile"],
    )
    def test_nanfunc_q(self, mat, dtype, nanfunc, func):
        mat = mat.astype(dtype)
        if mat.dtype.kind == "c":
            assert_raises(TypeError, func, mat, q=1)
            assert_raises(TypeError, nanfunc, mat, q=1)

        else:
            tgt = func(mat, q=1)
            out = nanfunc(mat, q=1)

            assert_almost_equal(out, tgt)

            if dtype == "O":
                assert type(out) is type(tgt)
            else:
                assert out.dtype == tgt.dtype

    @pytest.mark.parametrize(
        "nanfunc,func",
        [(np.nanvar, np.var), (np.nanstd, np.std)],
        ids=["nanvar", "nanstd"],
    )
    def test_nanfunc_ddof(self, mat, dtype, nanfunc, func):
        mat = mat.astype(dtype)
        tgt = func(mat, ddof=0.5)
        out = nanfunc(mat, ddof=0.5)

        assert_almost_equal(out, tgt)
        if dtype == "O":
            assert type(out) is type(tgt)
        else:
            assert out.dtype == tgt.dtype

    @pytest.mark.parametrize(
        "nanfunc", [np.nanvar, np.nanstd]
    )
    def test_nanfunc_correction(self, mat, dtype, nanfunc):
        mat = mat.astype(dtype)
        assert_almost_equal(
            nanfunc(mat, correction=0.5), nanfunc(mat, ddof=0.5)
        )

        err_msg = "ddof and correction can't be provided simultaneously."
        with assert_raises_regex(ValueError, err_msg):
            nanfunc(mat, ddof=0.5, correction=0.5)

        with assert_raises_regex(ValueError, err_msg):
            nanfunc(mat, ddof=1, correction=0)


class SharedNanFunctionsTestsMixin:
    def test_mutation(self):
        # Check that passed array is not modified.
        ndat = _ndat.copy()
        for f in self.nanfuncs:
            f(ndat)
            assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for axis in [None, 0, 1]:
                tgt = rf(mat, axis=axis, keepdims=True)
                res = nf(mat, axis=axis, keepdims=True)
                assert_(res.ndim == tgt.ndim)

    def test_out(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            resout = np.zeros(3)
            tgt = rf(mat, axis=1)
            res = nf(mat, axis=1, out=resout)
            assert_almost_equal(res, resout)
            assert_almost_equal(res, tgt)

    def test_dtype_from_dtype(self):
        mat = np.eye(3)
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                with suppress_warnings() as sup:
                    if nf in {np.nanstd, np.nanvar} and c in 'FDG':
                        # Giving the warning is a small bug, see gh-8000
                        sup.filter(ComplexWarning)
                    tgt = rf(mat, dtype=np.dtype(c), axis=1).dtype.type
                    res = nf(mat, dtype=np.dtype(c), axis=1).dtype.type
                    assert_(res is tgt)
                    # scalar case
                    tgt = rf(mat, dtype=np.dtype(c), axis=None).dtype.type
                    res = nf(mat, dtype=np.dtype(c), axis=None).dtype.type
                    assert_(res is tgt)

    def test_dtype_from_char(self):
        mat = np.eye(3)
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                with suppress_warnings() as sup:
                    if nf in {np.nanstd, np.nanvar} and c in 'FDG':
                        # Giving the warning is a small bug, see gh-8000
                        sup.filter(ComplexWarning)
                    tgt = rf(mat, dtype=c, axis=1).dtype.type
                    res = nf(mat, dtype=c, axis=1).dtype.type
                    assert_(res is tgt)
                    # scalar case
                    tgt = rf(mat, dtype=c, axis=None).dtype.type
                    res = nf(mat, dtype=c, axis=None).dtype.type
                    assert_(res is tgt)

    def test_dtype_from_input(self):
        codes = 'efdgFDG'
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            for c in codes:
                mat = np.eye(3, dtype=c)
                tgt = rf(mat, axis=1).dtype.type
                res = nf(mat, axis=1).dtype.type
                assert_(res is tgt, f"res {res}, tgt {tgt}")
                # scalar case
                tgt = rf(mat, axis=None).dtype.type
                res = nf(mat, axis=None).dtype.type
                assert_(res is tgt)

    def test_result_values(self):
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            tgt = [rf(d) for d in _rdat]
            res = nf(_ndat, axis=1)
            assert_almost_equal(res, tgt)

    def test_scalar(self):
        for f in self.nanfuncs:
            assert_(f(0.) == 0.)

    def test_subclass(self):
        class MyNDArray(np.ndarray):
            pass

        # Check that it works and that type and
        # shape are preserved
        array = np.eye(3)
        mine = array.view(MyNDArray)
        for f in self.nanfuncs:
            expected_shape = f(array, axis=0).shape
            res = f(mine, axis=0)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == expected_shape)
            expected_shape = f(array, axis=1).shape
            res = f(mine, axis=1)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == expected_shape)
            expected_shape = f(array).shape
            res = f(mine)
            assert_(isinstance(res, MyNDArray))
            assert_(res.shape == expected_shape)


class TestNanFunctions_SumProd(SharedNanFunctionsTestsMixin):

    nanfuncs = [np.nansum, np.nanprod]
    stdfuncs = [np.sum, np.prod]

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip("`axis != None` not supported for 0d arrays")

        array = array.astype(dtype)
        for func, identity in zip(self.nanfuncs, [0, 1]):
            out = func(array, axis=axis)
            assert np.all(out == identity)
            assert out.dtype == array.dtype

    def test_empty(self):
        for f, tgt_value in zip([np.nansum, np.nanprod], [0, 1]):
            mat = np.zeros((0, 3))
            tgt = [tgt_value] * 3
            res = f(mat, axis=0)
            assert_equal(res, tgt)
            tgt = []
            res = f(mat, axis=1)
            assert_equal(res, tgt)
            tgt = tgt_value
            res = f(mat, axis=None)
            assert_equal(res, tgt)

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_initial(self, dtype):
        ar = np.arange(9).astype(dtype)
        ar[:5] = np.nan

        for f in self.nanfuncs:
            reference = 28 if f is np.nansum else 3360
            ret = f(ar, initial=2)
            assert ret.dtype == dtype
            assert ret == reference

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_where(self, dtype):
        ar = np.arange(9).reshape(3, 3).astype(dtype)
        ar[0, :] = np.nan
        where = np.ones_like(ar, dtype=np.bool)
        where[:, 0] = False

        for f in self.nanfuncs:
            reference = 26 if f is np.nansum else 2240
            ret = f(ar, where=where, initial=2)
            assert ret.dtype == dtype
            assert ret == reference


class TestNanFunctions_CumSumProd(SharedNanFunctionsTestsMixin):

    nanfuncs = [np.nancumsum, np.nancumprod]
    stdfuncs = [np.cumsum, np.cumprod]

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan)
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip("`axis != None` not supported for 0d arrays")

        array = array.astype(dtype)
        for func, identity in zip(self.nanfuncs, [0, 1]):
            out = func(array)
            assert np.all(out == identity)
            assert out.dtype == array.dtype

    def test_empty(self):
        for f, tgt_value in zip(self.nanfuncs, [0, 1]):
            mat = np.zeros((0, 3))
            tgt = tgt_value * np.ones((0, 3))
            res = f(mat, axis=0)
            assert_equal(res, tgt)
            tgt = mat
            res = f(mat, axis=1)
            assert_equal(res, tgt)
            tgt = np.zeros(0)
            res = f(mat, axis=None)
            assert_equal(res, tgt)

    def test_keepdims(self):
        for f, g in zip(self.nanfuncs, self.stdfuncs):
            mat = np.eye(3)
            for axis in [None, 0, 1]:
                tgt = f(mat, axis=axis, out=None)
                res = g(mat, axis=axis, out=None)
                assert_(res.ndim == tgt.ndim)

        for f in self.nanfuncs:
            d = np.ones((3, 5, 7, 11))
            # Randomly set some elements to NaN:
            rs = np.random.RandomState(0)
            d[rs.rand(*d.shape) < 0.5] = np.nan
            res = f(d, axis=None)
            assert_equal(res.shape, (1155,))
            for axis in np.arange(4):
                res = f(d, axis=axis)
                assert_equal(res.shape, (3, 5, 7, 11))

    def test_result_values(self):
        for axis in (-2, -1, 0, 1, None):
            tgt = np.cumprod(_ndat_ones, axis=axis)
            res = np.nancumprod(_ndat, axis=axis)
            assert_almost_equal(res, tgt)
            tgt = np.cumsum(_ndat_zeros, axis=axis)
            res = np.nancumsum(_ndat, axis=axis)
            assert_almost_equal(res, tgt)

    def test_out(self):
        mat = np.eye(3)
        for nf, rf in zip(self.nanfuncs, self.stdfuncs):
            resout = np.eye(3)
            for axis in (-2, -1, 0, 1):
                tgt = rf(mat, axis=axis)
                res = nf(mat, axis=axis, out=resout)
                assert_almost_equal(res, resout)
                assert_almost_equal(res, tgt)


class TestNanFunctions_MeanVarStd(SharedNanFunctionsTestsMixin):

    nanfuncs = [np.nanmean, np.nanvar, np.nanstd]
    stdfuncs = [np.mean, np.var, np.std]

    def test_dtype_error(self):
        for f in self.nanfuncs:
            for dtype in [np.bool, np.int_, np.object_]:
                assert_raises(TypeError, f, _ndat, axis=1, dtype=dtype)

    def test_out_dtype_error(self):
        for f in self.nanfuncs:
            for dtype in [np.bool, np.int_, np.object_]:
                out = np.empty(_ndat.shape[0], dtype=dtype)
                assert_raises(TypeError, f, _ndat, axis=1, out=out)

    def test_ddof(self):
        nanfuncs = [np.nanvar, np.nanstd]
        stdfuncs = [np.var, np.std]
        for nf, rf in zip(nanfuncs, stdfuncs):
            for ddof in [0, 1]:
                tgt = [rf(d, ddof=ddof) for d in _rdat]
                res = nf(_ndat, axis=1, ddof=ddof)
                assert_almost_equal(res, tgt)

    def test_ddof_too_big(self):
        nanfuncs = [np.nanvar, np.nanstd]
        stdfuncs = [np.var, np.std]
        dsize = [len(d) for d in _rdat]
        for nf, rf in zip(nanfuncs, stdfuncs):
            for ddof in range(5):
                with suppress_warnings() as sup:
                    sup.record(RuntimeWarning)
                    sup.filter(ComplexWarning)
                    tgt = [ddof >= d for d in dsize]
                    res = nf(_ndat, axis=1, ddof=ddof)
                    assert_equal(np.isnan(res), tgt)
                    if any(tgt):
                        assert_(len(sup.log) == 1)
                    else:
                        assert_(len(sup.log) == 0)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip("`axis != None` not supported for 0d arrays")

        array = array.astype(dtype)
        match = "(Degrees of freedom <= 0 for slice.)|(Mean of empty slice)"
        for func in self.nanfuncs:
            with pytest.warns(RuntimeWarning, match=match):
                out = func(array, axis=axis)
            assert np.isnan(out).all()

            # `nanvar` and `nanstd` convert complex inputs to their
            # corresponding floating dtype
            if func is np.nanmean:
                assert out.dtype == array.dtype
            else:
                assert out.dtype == np.abs(array).dtype

    def test_empty(self):
        mat = np.zeros((0, 3))
        for f in self.nanfuncs:
            for axis in [0, None]:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    assert_(np.isnan(f(mat, axis=axis)).all())
                    assert_(len(w) == 1)
                    assert_(issubclass(w[0].category, RuntimeWarning))
            for axis in [1]:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    assert_equal(f(mat, axis=axis), np.zeros([]))
                    assert_(len(w) == 0)

    @pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
    def test_where(self, dtype):
        ar = np.arange(9).reshape(3, 3).astype(dtype)
        ar[0, :] = np.nan
        where = np.ones_like(ar, dtype=np.bool)
        where[:, 0] = False

        for f, f_std in zip(self.nanfuncs, self.stdfuncs):
            reference = f_std(ar[where][2:])
            dtype_reference = dtype if f is np.nanmean else ar.real.dtype

            ret = f(ar, where=where)
            assert ret.dtype == dtype_reference
            np.testing.assert_allclose(ret, reference)

    def test_nanstd_with_mean_keyword(self):
        # Setting the seed to make the test reproducible
        rng = np.random.RandomState(1234)
        A = rng.randn(10, 20, 5) + 0.5
        A[:, 5, :] = np.nan

        mean_out = np.zeros((10, 1, 5))
        std_out = np.zeros((10, 1, 5))

        mean = np.nanmean(A,
                       out=mean_out,
                       axis=1,
                       keepdims=True)

        # The returned  object should be the object specified during calling
        assert mean_out is mean

        std = np.nanstd(A,
                     out=std_out,
                     axis=1,
                     keepdims=True,
                     mean=mean)

        # The returned  object should be the object specified during calling
        assert std_out is std

        # Shape of returned mean and std should be same
        assert std.shape == mean.shape
        assert std.shape == (10, 1, 5)

        # Output should be the same as from the individual algorithms
        std_old = np.nanstd(A, axis=1, keepdims=True)

        assert std_old.shape == mean.shape
        assert_almost_equal(std, std_old)


_TIME_UNITS = (
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"
)

# All `inexact` + `timdelta64` type codes
_TYPE_CODES = list(np.typecodes["AllFloat"])
_TYPE_CODES += [f"m8[{unit}]" for unit in _TIME_UNITS]


class TestNanFunctions_Median:

    def test_mutation(self):
        # Check that passed array is not modified.
        ndat = _ndat.copy()
        np.nanmedian(ndat)
        assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for axis in [None, 0, 1]:
            tgt = np.median(mat, axis=axis, out=None, overwrite_input=False)
            res = np.nanmedian(mat, axis=axis, out=None, overwrite_input=False)
            assert_(res.ndim == tgt.ndim)

        d = np.ones((3, 5, 7, 11))
        # Randomly set some elements to NaN:
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            res = np.nanmedian(d, axis=None, keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanmedian(d, axis=(0, 1), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 11))
            res = np.nanmedian(d, axis=(0, 3), keepdims=True)
            assert_equal(res.shape, (1, 5, 7, 1))
            res = np.nanmedian(d, axis=(1,), keepdims=True)
            assert_equal(res.shape, (3, 1, 7, 11))
            res = np.nanmedian(d, axis=(0, 1, 2, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanmedian(d, axis=(0, 1, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 1))

    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,
            1,
            (1, ),
            (0, 1),
            (-3, -1),
        ]
    )
    @pytest.mark.filterwarnings("ignore:All-NaN slice:RuntimeWarning")
    def test_keepdims_out(self, axis):
        d = np.ones((3, 5, 7, 11))
        # Randomly set some elements to NaN:
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        out = np.empty(shape_out)
        result = np.nanmedian(d, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    def test_out(self):
        mat = np.random.rand(3, 3)
        nan_mat = np.insert(mat, [0, 2], np.nan, axis=1)
        resout = np.zeros(3)
        tgt = np.median(mat, axis=1)
        res = np.nanmedian(nan_mat, axis=1, out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        # 0-d output:
        resout = np.zeros(())
        tgt = np.median(mat, axis=None)
        res = np.nanmedian(nan_mat, axis=None, out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        res = np.nanmedian(nan_mat, axis=(0, 1), out=resout)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)

    def test_small_large(self):
        # test the small and large code paths, current cutoff 400 elements
        for s in [5, 20, 51, 200, 1000]:
            d = np.random.randn(4, s)
            # Randomly set some elements to NaN:
            w = np.random.randint(0, d.size, size=d.size // 5)
            d.ravel()[w] = np.nan
            d[:, 0] = 1.  # ensure at least one good value
            # use normal median without nans to compare
            tgt = []
            for x in d:
                nonan = np.compress(~np.isnan(x), x)
                tgt.append(np.median(nonan, overwrite_input=True))

            assert_array_equal(np.nanmedian(d, axis=-1), tgt)

    def test_result_values(self):
        tgt = [np.median(d) for d in _rdat]
        res = np.nanmedian(_ndat, axis=1)
        assert_almost_equal(res, tgt)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", _TYPE_CODES)
    def test_allnans(self, dtype, axis):
        mat = np.full((3, 3), np.nan).astype(dtype)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)

            output = np.nanmedian(mat, axis=axis)
            assert output.dtype == mat.dtype
            assert np.isnan(output).all()

            if axis is None:
                assert_(len(sup.log) == 1)
            else:
                assert_(len(sup.log) == 3)

            # Check scalar
            scalar = np.array(np.nan).astype(dtype)[()]
            output_scalar = np.nanmedian(scalar)
            assert output_scalar.dtype == scalar.dtype
            assert np.isnan(output_scalar)

            if axis is None:
                assert_(len(sup.log) == 2)
            else:
                assert_(len(sup.log) == 4)

    def test_empty(self):
        mat = np.zeros((0, 3))
        for axis in [0, None]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                assert_(np.isnan(np.nanmedian(mat, axis=axis)).all())
                assert_(len(w) == 1)
                assert_(issubclass(w[0].category, RuntimeWarning))
        for axis in [1]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                assert_equal(np.nanmedian(mat, axis=axis), np.zeros([]))
                assert_(len(w) == 0)

    def test_scalar(self):
        assert_(np.nanmedian(0.) == 0.)

    def test_extended_axis_invalid(self):
        d = np.ones((3, 5, 7, 11))
        assert_raises(AxisError, np.nanmedian, d, axis=-5)
        assert_raises(AxisError, np.nanmedian, d, axis=(0, -5))
        assert_raises(AxisError, np.nanmedian, d, axis=4)
        assert_raises(AxisError, np.nanmedian, d, axis=(0, 4))
        assert_raises(ValueError, np.nanmedian, d, axis=(1, 1))

    def test_float_special(self):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            for inf in [np.inf, -np.inf]:
                a = np.array([[inf,  np.nan], [np.nan, np.nan]])
                assert_equal(np.nanmedian(a, axis=0), [inf,  np.nan])
                assert_equal(np.nanmedian(a, axis=1), [inf,  np.nan])
                assert_equal(np.nanmedian(a), inf)

                # minimum fill value check
                a = np.array([[np.nan, np.nan, inf],
                             [np.nan, np.nan, inf]])
                assert_equal(np.nanmedian(a), inf)
                assert_equal(np.nanmedian(a, axis=0), [np.nan, np.nan, inf])
                assert_equal(np.nanmedian(a, axis=1), inf)

                # no mask path
                a = np.array([[inf, inf], [inf, inf]])
                assert_equal(np.nanmedian(a, axis=1), inf)

                a = np.array([[inf, 7, -inf, -9],
                              [-10, np.nan, np.nan, 5],
                              [4, np.nan, np.nan, inf]],
                              dtype=np.float32)
                if inf > 0:
                    assert_equal(np.nanmedian(a, axis=0), [4., 7., -inf, 5.])
                    assert_equal(np.nanmedian(a), 4.5)
                else:
                    assert_equal(np.nanmedian(a, axis=0), [-10., 7., -inf, -9.])
                    assert_equal(np.nanmedian(a), -2.5)
                assert_equal(np.nanmedian(a, axis=-1), [-1., -2.5, inf])

                for i in range(10):
                    for j in range(1, 10):
                        a = np.array([([np.nan] * i) + ([inf] * j)] * 2)
                        assert_equal(np.nanmedian(a), inf)
                        assert_equal(np.nanmedian(a, axis=1), inf)
                        assert_equal(np.nanmedian(a, axis=0),
                                     ([np.nan] * i) + [inf] * j)

                        a = np.array([([np.nan] * i) + ([-inf] * j)] * 2)
                        assert_equal(np.nanmedian(a), -inf)
                        assert_equal(np.nanmedian(a, axis=1), -inf)
                        assert_equal(np.nanmedian(a, axis=0),
                                     ([np.nan] * i) + [-inf] * j)


class TestNanFunctions_Percentile:

    def test_mutation(self):
        # Check that passed array is not modified.
        ndat = _ndat.copy()
        np.nanpercentile(ndat, 30)
        assert_equal(ndat, _ndat)

    def test_keepdims(self):
        mat = np.eye(3)
        for axis in [None, 0, 1]:
            tgt = np.percentile(mat, 70, axis=axis, out=None,
                                overwrite_input=False)
            res = np.nanpercentile(mat, 70, axis=axis, out=None,
                                   overwrite_input=False)
            assert_(res.ndim == tgt.ndim)

        d = np.ones((3, 5, 7, 11))
        # Randomly set some elements to NaN:
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            res = np.nanpercentile(d, 90, axis=None, keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanpercentile(d, 90, axis=(0, 1), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 11))
            res = np.nanpercentile(d, 90, axis=(0, 3), keepdims=True)
            assert_equal(res.shape, (1, 5, 7, 1))
            res = np.nanpercentile(d, 90, axis=(1,), keepdims=True)
            assert_equal(res.shape, (3, 1, 7, 11))
            res = np.nanpercentile(d, 90, axis=(0, 1, 2, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 1, 1))
            res = np.nanpercentile(d, 90, axis=(0, 1, 3), keepdims=True)
            assert_equal(res.shape, (1, 1, 7, 1))

    @pytest.mark.parametrize('q', [7, [1, 7]])
    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,
            1,
            (1,),
            (0, 1),
            (-3, -1),
        ]
    )
    @pytest.mark.filterwarnings("ignore:All-NaN slice:RuntimeWarning")
    def test_keepdims_out(self, q, axis):
        d = np.ones((3, 5, 7, 11))
        # Randomly set some elements to NaN:
        w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
        w = w.astype(np.intp)
        d[tuple(w)] = np.nan
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        shape_out = np.shape(q) + shape_out

        out = np.empty(shape_out)
        result = np.nanpercentile(d, q, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_out(self, weighted):
        mat = np.random.rand(3, 3)
        nan_mat = np.insert(mat, [0, 2], np.nan, axis=1)
        resout = np.zeros(3)
        if weighted:
            w_args = {"weights": np.ones_like(mat), "method": "inverted_cdf"}
            nan_w_args = {
                "weights": np.ones_like(nan_mat), "method": "inverted_cdf"
            }
        else:
            w_args = {}
            nan_w_args = {}
        tgt = np.percentile(mat, 42, axis=1, **w_args)
        res = np.nanpercentile(nan_mat, 42, axis=1, out=resout, **nan_w_args)
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        # 0-d output:
        resout = np.zeros(())
        tgt = np.percentile(mat, 42, axis=None, **w_args)
        res = np.nanpercentile(
            nan_mat, 42, axis=None, out=resout, **nan_w_args
        )
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)
        res = np.nanpercentile(
            nan_mat, 42, axis=(0, 1), out=resout, **nan_w_args
        )
        assert_almost_equal(res, resout)
        assert_almost_equal(res, tgt)

    def test_complex(self):
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='G')
        assert_raises(TypeError, np.nanpercentile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='D')
        assert_raises(TypeError, np.nanpercentile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='F')
        assert_raises(TypeError, np.nanpercentile, arr_c, 0.5)

    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("use_out", [False, True])
    def test_result_values(self, weighted, use_out):
        if weighted:
            percentile = partial(np.percentile, method="inverted_cdf")
            nanpercentile = partial(np.nanpercentile, method="inverted_cdf")

            def gen_weights(d):
                return np.ones_like(d)

        else:
            percentile = np.percentile
            nanpercentile = np.nanpercentile

            def gen_weights(d):
                return None

        tgt = [percentile(d, 28, weights=gen_weights(d)) for d in _rdat]
        out = np.empty_like(tgt) if use_out else None
        res = nanpercentile(_ndat, 28, axis=1,
                            weights=gen_weights(_ndat), out=out)
        assert_almost_equal(res, tgt)
        # Transpose the array to fit the output convention of numpy.percentile
        tgt = np.transpose([percentile(d, (28, 98), weights=gen_weights(d))
                            for d in _rdat])
        out = np.empty_like(tgt) if use_out else None
        res = nanpercentile(_ndat, (28, 98), axis=1,
                            weights=gen_weights(_ndat), out=out)
        assert_almost_equal(res, tgt)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["Float"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip("`axis != None` not supported for 0d arrays")

        array = array.astype(dtype)
        with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
            out = np.nanpercentile(array, 60, axis=axis)
        assert np.isnan(out).all()
        assert out.dtype == array.dtype

    def test_empty(self):
        mat = np.zeros((0, 3))
        for axis in [0, None]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                assert_(np.isnan(np.nanpercentile(mat, 40, axis=axis)).all())
                assert_(len(w) == 1)
                assert_(issubclass(w[0].category, RuntimeWarning))
        for axis in [1]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                assert_equal(np.nanpercentile(mat, 40, axis=axis), np.zeros([]))
                assert_(len(w) == 0)

    def test_scalar(self):
        assert_equal(np.nanpercentile(0., 100), 0.)
        a = np.arange(6)
        r = np.nanpercentile(a, 50, axis=0)
        assert_equal(r, 2.5)
        assert_(np.isscalar(r))

    def test_extended_axis_invalid(self):
        d = np.ones((3, 5, 7, 11))
        assert_raises(AxisError, np.nanpercentile, d, q=5, axis=-5)
        assert_raises(AxisError, np.nanpercentile, d, q=5, axis=(0, -5))
        assert_raises(AxisError, np.nanpercentile, d, q=5, axis=4)
        assert_raises(AxisError, np.nanpercentile, d, q=5, axis=(0, 4))
        assert_raises(ValueError, np.nanpercentile, d, q=5, axis=(1, 1))

    def test_multiple_percentiles(self):
        perc = [50, 100]
        mat = np.ones((4, 3))
        nan_mat = np.nan * mat
        # For checking consistency in higher dimensional case
        large_mat = np.ones((3, 4, 5))
        large_mat[:, 0:2:4, :] = 0
        large_mat[:, :, 3:] *= 2
        for axis in [None, 0, 1]:
            for keepdim in [False, True]:
                with suppress_warnings() as sup:
                    sup.filter(RuntimeWarning, "All-NaN slice encountered")
                    val = np.percentile(mat, perc, axis=axis, keepdims=keepdim)
                    nan_val = np.nanpercentile(nan_mat, perc, axis=axis,
                                               keepdims=keepdim)
                    assert_equal(nan_val.shape, val.shape)

                    val = np.percentile(large_mat, perc, axis=axis,
                                        keepdims=keepdim)
                    nan_val = np.nanpercentile(large_mat, perc, axis=axis,
                                               keepdims=keepdim)
                    assert_equal(nan_val, val)

        megamat = np.ones((3, 4, 5, 6))
        assert_equal(
            np.nanpercentile(megamat, perc, axis=(1, 2)).shape, (2, 3, 6)
        )

    @pytest.mark.parametrize("nan_weight", [0, 1, 2, 3, 1e200])
    def test_nan_value_with_weight(self, nan_weight):
        x = [1, np.nan, 2, 3]
        result = np.float64(2.0)
        q_unweighted = np.nanpercentile(x, 50, method="inverted_cdf")
        assert_equal(q_unweighted, result)

        # The weight value at the nan position should not matter.
        w = [1.0, nan_weight, 1.0, 1.0]
        q_weighted = np.nanpercentile(x, 50, weights=w, method="inverted_cdf")
        assert_equal(q_weighted, result)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_nan_value_with_weight_ndim(self, axis):
        # Create a multi-dimensional array to test
        np.random.seed(1)
        x_no_nan = np.random.random(size=(100, 99, 2))
        # Set some places to NaN (not particularly smart) so there is always
        # some non-Nan.
        x = x_no_nan.copy()
        x[np.arange(99), np.arange(99), 0] = np.nan

        p = np.array([[20., 50., 30], [70, 33, 80]])

        # We just use ones as weights, but replace it with 0 or 1e200 at the
        # NaN positions below.
        weights = np.ones_like(x)

        # For comparison use weighted normal percentile with nan weights at
        # 0 (and no NaNs); not sure this is strictly identical but should be
        # sufficiently so (if a percentile lies exactly on a 0 value).
        weights[np.isnan(x)] = 0
        p_expected = np.percentile(
            x_no_nan, p, axis=axis, weights=weights, method="inverted_cdf")

        p_unweighted = np.nanpercentile(
            x, p, axis=axis, method="inverted_cdf")
        # The normal and unweighted versions should be identical:
        assert_equal(p_unweighted, p_expected)

        weights[np.isnan(x)] = 1e200  # huge value, shouldn't matter
        p_weighted = np.nanpercentile(
            x, p, axis=axis, weights=weights, method="inverted_cdf")
        assert_equal(p_weighted, p_expected)
        # Also check with out passed:
        out = np.empty_like(p_weighted)
        res = np.nanpercentile(
            x, p, axis=axis, weights=weights, out=out, method="inverted_cdf")

        assert res is out
        assert_equal(out, p_expected)


class TestNanFunctions_Quantile:
    # most of this is already tested by TestPercentile

    @pytest.mark.parametrize("weighted", [False, True])
    def test_regression(self, weighted):
        ar = np.arange(24).reshape(2, 3, 4).astype(float)
        ar[0][1] = np.nan
        if weighted:
            w_args = {"weights": np.ones_like(ar), "method": "inverted_cdf"}
        else:
            w_args = {}

        assert_equal(np.nanquantile(ar, q=0.5, **w_args),
                     np.nanpercentile(ar, q=50, **w_args))
        assert_equal(np.nanquantile(ar, q=0.5, axis=0, **w_args),
                     np.nanpercentile(ar, q=50, axis=0, **w_args))
        assert_equal(np.nanquantile(ar, q=0.5, axis=1, **w_args),
                     np.nanpercentile(ar, q=50, axis=1, **w_args))
        assert_equal(np.nanquantile(ar, q=[0.5], axis=1, **w_args),
                     np.nanpercentile(ar, q=[50], axis=1, **w_args))
        assert_equal(np.nanquantile(ar, q=[0.25, 0.5, 0.75], axis=1, **w_args),
                     np.nanpercentile(ar, q=[25, 50, 75], axis=1, **w_args))

    def test_basic(self):
        x = np.arange(8) * 0.5
        assert_equal(np.nanquantile(x, 0), 0.)
        assert_equal(np.nanquantile(x, 1), 3.5)
        assert_equal(np.nanquantile(x, 0.5), 1.75)

    def test_complex(self):
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='G')
        assert_raises(TypeError, np.nanquantile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='D')
        assert_raises(TypeError, np.nanquantile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype='F')
        assert_raises(TypeError, np.nanquantile, arr_c, 0.5)

    def test_no_p_overwrite(self):
        # this is worth retesting, because quantile does not make a copy
        p0 = np.array([0, 0.75, 0.25, 0.5, 1.0])
        p = p0.copy()
        np.nanquantile(np.arange(100.), p, method="midpoint")
        assert_array_equal(p, p0)

        p0 = p0.tolist()
        p = p.tolist()
        np.nanquantile(np.arange(100.), p, method="midpoint")
        assert_array_equal(p, p0)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", np.typecodes["Float"])
    @pytest.mark.parametrize("array", [
        np.array(np.nan),
        np.full((3, 3), np.nan),
    ], ids=["0d", "2d"])
    def test_allnans(self, axis, dtype, array):
        if axis is not None and array.ndim == 0:
            pytest.skip("`axis != None` not supported for 0d arrays")

        array = array.astype(dtype)
        with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
            out = np.nanquantile(array, 1, axis=axis)
        assert np.isnan(out).all()
        assert out.dtype == array.dtype

@pytest.mark.parametrize("arr, expected", [
    # array of floats with some nans
    (np.array([np.nan, 5.0, np.nan, np.inf]),
     np.array([False, True, False, True])),
    # int64 array that can't possibly have nans
    (np.array([1, 5, 7, 9], dtype=np.int64),
     True),
    # bool array that can't possibly have nans
    (np.array([False, True, False, True]),
     True),
    # 2-D complex array with nans
    (np.array([[np.nan, 5.0],
               [np.nan, np.inf]], dtype=np.complex64),
     np.array([[False, True],
               [False, True]])),
    ])
def test__nan_mask(arr, expected):
    for out in [None, np.empty(arr.shape, dtype=np.bool)]:
        actual = _nan_mask(arr, out=out)
        assert_equal(actual, expected)
        # the above won't distinguish between True proper
        # and an array of True values; we want True proper
        # for types that can't possibly contain NaN
        if type(expected) is not np.ndarray:
            assert actual is True


def test__replace_nan():
    """ Test that _replace_nan returns the original array if there are no
    NaNs, not a copy.
    """
    for dtype in [np.bool, np.int32, np.int64]:
        arr = np.array([0, 1], dtype=dtype)
        result, mask = _replace_nan(arr, 0)
        assert mask is None
        # do not make a copy if there are no nans
        assert result is arr

    for dtype in [np.float32, np.float64]:
        arr = np.array([0, 1], dtype=dtype)
        result, mask = _replace_nan(arr, 2)
        assert (mask == False).all()
        # mask is not None, so we make a copy
        assert result is not arr
        assert_equal(result, arr)

        arr_nan = np.array([0, 1, np.nan], dtype=dtype)
        result_nan, mask_nan = _replace_nan(arr_nan, 2)
        assert_equal(mask_nan, np.array([False, False, True]))
        assert result_nan is not arr_nan
        assert_equal(result_nan, np.array([0, 1, 2]))
        assert np.isnan(arr_nan[-1])


def test_memmap_takes_fast_route(tmpdir):
    # We want memory mapped arrays to take the fast route through nanmax,
    # which avoids creating a mask by using fmax.reduce (see gh-28721). So we
    # check that on bad input, the error is from fmax (rather than maximum).
    a = np.arange(10., dtype=float)
    with open(tmpdir.join("data.bin"), "w+b") as fh:
        fh.write(a.tobytes())
        mm = np.memmap(fh, dtype=a.dtype, shape=a.shape)
        with pytest.raises(ValueError, match="reduction operation fmax"):
            np.nanmax(mm, out=np.zeros(2))
        # For completeness, same for nanmin.
        with pytest.raises(ValueError, match="reduction operation fmin"):
            np.nanmin(mm, out=np.zeros(2))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_arraypad.py ===
"""Tests for the array padding functions.

"""
import pytest

import numpy as np
from numpy.lib._arraypad_impl import _as_pairs
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

_numeric_dtypes = (
    np._core.sctypes["uint"]
    + np._core.sctypes["int"]
    + np._core.sctypes["float"]
    + np._core.sctypes["complex"]
)
_all_modes = {
    'constant': {'constant_values': 0},
    'edge': {},
    'linear_ramp': {'end_values': 0},
    'maximum': {'stat_length': None},
    'mean': {'stat_length': None},
    'median': {'stat_length': None},
    'minimum': {'stat_length': None},
    'reflect': {'reflect_type': 'even'},
    'symmetric': {'reflect_type': 'even'},
    'wrap': {},
    'empty': {}
}


class TestAsPairs:
    def test_single_value(self):
        """Test casting for a single value."""
        expected = np.array([[3, 3]] * 10)
        for x in (3, [3], [[3]]):
            result = _as_pairs(x, 10)
            assert_equal(result, expected)
        # Test with dtype=object
        obj = object()
        assert_equal(
            _as_pairs(obj, 10),
            np.array([[obj, obj]] * 10)
        )

    def test_two_values(self):
        """Test proper casting for two different values."""
        # Broadcasting in the first dimension with numbers
        expected = np.array([[3, 4]] * 10)
        for x in ([3, 4], [[3, 4]]):
            result = _as_pairs(x, 10)
            assert_equal(result, expected)
        # and with dtype=object
        obj = object()
        assert_equal(
            _as_pairs(["a", obj], 10),
            np.array([["a", obj]] * 10)
        )

        # Broadcasting in the second / last dimension with numbers
        assert_equal(
            _as_pairs([[3], [4]], 2),
            np.array([[3, 3], [4, 4]])
        )
        # and with dtype=object
        assert_equal(
            _as_pairs([["a"], [obj]], 2),
            np.array([["a", "a"], [obj, obj]])
        )

    def test_with_none(self):
        expected = ((None, None), (None, None), (None, None))
        assert_equal(
            _as_pairs(None, 3, as_index=False),
            expected
        )
        assert_equal(
            _as_pairs(None, 3, as_index=True),
            expected
        )

    def test_pass_through(self):
        """Test if `x` already matching desired output are passed through."""
        expected = np.arange(12).reshape((6, 2))
        assert_equal(
            _as_pairs(expected, 6),
            expected
        )

    def test_as_index(self):
        """Test results if `as_index=True`."""
        assert_equal(
            _as_pairs([2.6, 3.3], 10, as_index=True),
            np.array([[3, 3]] * 10, dtype=np.intp)
        )
        assert_equal(
            _as_pairs([2.6, 4.49], 10, as_index=True),
            np.array([[3, 4]] * 10, dtype=np.intp)
        )
        for x in (-3, [-3], [[-3]], [-3, 4], [3, -4], [[-3, 4]], [[4, -3]],
                  [[1, 2]] * 9 + [[1, -2]]):
            with pytest.raises(ValueError, match="negative values"):
                _as_pairs(x, 10, as_index=True)

    def test_exceptions(self):
        """Ensure faulty usage is discovered."""
        with pytest.raises(ValueError, match="more dimensions than allowed"):
            _as_pairs([[[3]]], 10)
        with pytest.raises(ValueError, match="could not be broadcast"):
            _as_pairs([[1, 2], [3, 4]], 3)
        with pytest.raises(ValueError, match="could not be broadcast"):
            _as_pairs(np.ones((2, 3)), 3)


class TestConditionalShortcuts:
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_zero_padding_shortcuts(self, mode):
        test = np.arange(120).reshape(4, 5, 6)
        pad_amt = [(0, 0) for _ in test.shape]
        assert_array_equal(test, np.pad(test, pad_amt, mode=mode))

    @pytest.mark.parametrize("mode", ['maximum', 'mean', 'median', 'minimum',])
    def test_shallow_statistic_range(self, mode):
        test = np.arange(120).reshape(4, 5, 6)
        pad_amt = [(1, 1) for _ in test.shape]
        assert_array_equal(np.pad(test, pad_amt, mode='edge'),
                           np.pad(test, pad_amt, mode=mode, stat_length=1))

    @pytest.mark.parametrize("mode", ['maximum', 'mean', 'median', 'minimum',])
    def test_clip_statistic_range(self, mode):
        test = np.arange(30).reshape(5, 6)
        pad_amt = [(3, 3) for _ in test.shape]
        assert_array_equal(np.pad(test, pad_amt, mode=mode),
                           np.pad(test, pad_amt, mode=mode, stat_length=30))


class TestStatistic:
    def test_check_mean_stat_length(self):
        a = np.arange(100).astype('f')
        a = np.pad(a, ((25, 20), ), 'mean', stat_length=((2, 3), ))
        b = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
             0.5, 0.5, 0.5, 0.5, 0.5,

             0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
             10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
             20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
             30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
             40., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
             50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
             60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
             70., 71., 72., 73., 74., 75., 76., 77., 78., 79.,
             80., 81., 82., 83., 84., 85., 86., 87., 88., 89.,
             90., 91., 92., 93., 94., 95., 96., 97., 98., 99.,

             98., 98., 98., 98., 98., 98., 98., 98., 98., 98.,
             98., 98., 98., 98., 98., 98., 98., 98., 98., 98.
             ])
        assert_array_equal(a, b)

    def test_check_maximum_1(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), 'maximum')
        b = np.array(
            [99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
            )
        assert_array_equal(a, b)

    def test_check_maximum_2(self):
        a = np.arange(100) + 1
        a = np.pad(a, (25, 20), 'maximum')
        b = np.array(
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100,

             1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
             61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
             71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
             81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
             91, 92, 93, 94, 95, 96, 97, 98, 99, 100,

             100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
            )
        assert_array_equal(a, b)

    def test_check_maximum_stat_length(self):
        a = np.arange(100) + 1
        a = np.pad(a, (25, 20), 'maximum', stat_length=10)
        b = np.array(
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10,

              1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
             61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
             71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
             81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
             91, 92, 93, 94, 95, 96, 97, 98, 99, 100,

             100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
            )
        assert_array_equal(a, b)

    def test_check_minimum_1(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), 'minimum')
        b = np.array(
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,

              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )
        assert_array_equal(a, b)

    def test_check_minimum_2(self):
        a = np.arange(100) + 2
        a = np.pad(a, (25, 20), 'minimum')
        b = np.array(
            [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
              2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
              2,  2,  2,  2,  2,

              2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
             22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
             42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
             52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
             62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
             72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
             82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
             92, 93, 94, 95, 96, 97, 98, 99, 100, 101,

             2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            )
        assert_array_equal(a, b)

    def test_check_minimum_stat_length(self):
        a = np.arange(100) + 1
        a = np.pad(a, (25, 20), 'minimum', stat_length=10)
        b = np.array(
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,

              1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
             61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
             71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
             81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
             91, 92, 93, 94, 95, 96, 97, 98, 99, 100,

             91, 91, 91, 91, 91, 91, 91, 91, 91, 91,
             91, 91, 91, 91, 91, 91, 91, 91, 91, 91]
            )
        assert_array_equal(a, b)

    def test_check_median(self):
        a = np.arange(100).astype('f')
        a = np.pad(a, (25, 20), 'median')
        b = np.array(
            [49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5,

             0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
             10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
             20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
             30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
             40., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
             50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
             60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
             70., 71., 72., 73., 74., 75., 76., 77., 78., 79.,
             80., 81., 82., 83., 84., 85., 86., 87., 88., 89.,
             90., 91., 92., 93., 94., 95., 96., 97., 98., 99.,

             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5]
            )
        assert_array_equal(a, b)

    def test_check_median_01(self):
        a = np.array([[3, 1, 4], [4, 5, 9], [9, 8, 2]])
        a = np.pad(a, 1, 'median')
        b = np.array(
            [[4, 4, 5, 4, 4],

             [3, 3, 1, 4, 3],
             [5, 4, 5, 9, 5],
             [8, 9, 8, 2, 8],

             [4, 4, 5, 4, 4]]
            )
        assert_array_equal(a, b)

    def test_check_median_02(self):
        a = np.array([[3, 1, 4], [4, 5, 9], [9, 8, 2]])
        a = np.pad(a.T, 1, 'median').T
        b = np.array(
            [[5, 4, 5, 4, 5],

             [3, 3, 1, 4, 3],
             [5, 4, 5, 9, 5],
             [8, 9, 8, 2, 8],

             [5, 4, 5, 4, 5]]
            )
        assert_array_equal(a, b)

    def test_check_median_stat_length(self):
        a = np.arange(100).astype('f')
        a[1] = 2.
        a[97] = 96.
        a = np.pad(a, (25, 20), 'median', stat_length=(3, 5))
        b = np.array(
            [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
              2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
              2.,  2.,  2.,  2.,  2.,

              0.,  2.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,
             10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
             20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
             30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
             40., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
             50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
             60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
             70., 71., 72., 73., 74., 75., 76., 77., 78., 79.,
             80., 81., 82., 83., 84., 85., 86., 87., 88., 89.,
             90., 91., 92., 93., 94., 95., 96., 96., 98., 99.,

             96., 96., 96., 96., 96., 96., 96., 96., 96., 96.,
             96., 96., 96., 96., 96., 96., 96., 96., 96., 96.]
            )
        assert_array_equal(a, b)

    def test_check_mean_shape_one(self):
        a = [[4, 5, 6]]
        a = np.pad(a, (5, 7), 'mean', stat_length=2)
        b = np.array(
            [[4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],

             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],

             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6]]
            )
        assert_array_equal(a, b)

    def test_check_mean_2(self):
        a = np.arange(100).astype('f')
        a = np.pad(a, (25, 20), 'mean')
        b = np.array(
            [49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5,

             0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
             10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
             20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
             30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
             40., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
             50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
             60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
             70., 71., 72., 73., 74., 75., 76., 77., 78., 79.,
             80., 81., 82., 83., 84., 85., 86., 87., 88., 89.,
             90., 91., 92., 93., 94., 95., 96., 97., 98., 99.,

             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5]
            )
        assert_array_equal(a, b)

    @pytest.mark.parametrize("mode", [
        "mean",
        "median",
        "minimum",
        "maximum"
    ])
    def test_same_prepend_append(self, mode):
        """ Test that appended and prepended values are equal """
        # This test is constructed to trigger floating point rounding errors in
        # a way that caused gh-11216 for mode=='mean'
        a = np.array([-1, 2, -1]) + np.array([0, 1e-12, 0], dtype=np.float64)
        a = np.pad(a, (1, 1), mode)
        assert_equal(a[0], a[-1])

    @pytest.mark.parametrize("mode", ["mean", "median", "minimum", "maximum"])
    @pytest.mark.parametrize(
        "stat_length", [-2, (-2,), (3, -1), ((5, 2), (-2, 3)), ((-4,), (2,))]
    )
    def test_check_negative_stat_length(self, mode, stat_length):
        arr = np.arange(30).reshape((6, 5))
        match = "index can't contain negative values"
        with pytest.raises(ValueError, match=match):
            np.pad(arr, 2, mode, stat_length=stat_length)

    def test_simple_stat_length(self):
        a = np.arange(30)
        a = np.reshape(a, (6, 5))
        a = np.pad(a, ((2, 3), (3, 2)), mode='mean', stat_length=(3,))
        b = np.array(
            [[6, 6, 6, 5, 6, 7, 8, 9, 8, 8],
             [6, 6, 6, 5, 6, 7, 8, 9, 8, 8],

             [1, 1, 1, 0, 1, 2, 3, 4, 3, 3],
             [6, 6, 6, 5, 6, 7, 8, 9, 8, 8],
             [11, 11, 11, 10, 11, 12, 13, 14, 13, 13],
             [16, 16, 16, 15, 16, 17, 18, 19, 18, 18],
             [21, 21, 21, 20, 21, 22, 23, 24, 23, 23],
             [26, 26, 26, 25, 26, 27, 28, 29, 28, 28],

             [21, 21, 21, 20, 21, 22, 23, 24, 23, 23],
             [21, 21, 21, 20, 21, 22, 23, 24, 23, 23],
             [21, 21, 21, 20, 21, 22, 23, 24, 23, 23]]
            )
        assert_array_equal(a, b)

    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in( scalar)? divide:RuntimeWarning"
    )
    @pytest.mark.parametrize("mode", ["mean", "median"])
    def test_zero_stat_length_valid(self, mode):
        arr = np.pad([1., 2.], (1, 2), mode, stat_length=0)
        expected = np.array([np.nan, 1., 2., np.nan, np.nan])
        assert_equal(arr, expected)

    @pytest.mark.parametrize("mode", ["minimum", "maximum"])
    def test_zero_stat_length_invalid(self, mode):
        match = "stat_length of 0 yields no value for padding"
        with pytest.raises(ValueError, match=match):
            np.pad([1., 2.], 0, mode, stat_length=0)
        with pytest.raises(ValueError, match=match):
            np.pad([1., 2.], 0, mode, stat_length=(1, 0))
        with pytest.raises(ValueError, match=match):
            np.pad([1., 2.], 1, mode, stat_length=0)
        with pytest.raises(ValueError, match=match):
            np.pad([1., 2.], 1, mode, stat_length=(1, 0))


class TestConstant:
    def test_check_constant(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), 'constant', constant_values=(10, 20))
        b = np.array(
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
             20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
            )
        assert_array_equal(a, b)

    def test_check_constant_zeros(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), 'constant')
        b = np.array(
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
            )
        assert_array_equal(a, b)

    def test_check_constant_float(self):
        # If input array is int, but constant_values are float, the dtype of
        # the array to be padded is kept
        arr = np.arange(30).reshape(5, 6)
        test = np.pad(arr, (1, 2), mode='constant',
                   constant_values=1.1)
        expected = np.array(
            [[1,  1,  1,  1,  1,  1,  1,  1,  1],

             [1,  0,  1,  2,  3,  4,  5,  1,  1],
             [1,  6,  7,  8,  9, 10, 11,  1,  1],
             [1, 12, 13, 14, 15, 16, 17,  1,  1],
             [1, 18, 19, 20, 21, 22, 23,  1,  1],
             [1, 24, 25, 26, 27, 28, 29,  1,  1],

             [1,  1,  1,  1,  1,  1,  1,  1,  1],
             [1,  1,  1,  1,  1,  1,  1,  1,  1]]
            )
        assert_allclose(test, expected)

    def test_check_constant_float2(self):
        # If input array is float, and constant_values are float, the dtype of
        # the array to be padded is kept - here retaining the float constants
        arr = np.arange(30).reshape(5, 6)
        arr_float = arr.astype(np.float64)
        test = np.pad(arr_float, ((1, 2), (1, 2)), mode='constant',
                   constant_values=1.1)
        expected = np.array(
            [[1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1],

             [1.1,   0. ,   1. ,   2. ,   3. ,   4. ,   5. ,   1.1,   1.1],  # noqa: E203
             [1.1,   6. ,   7. ,   8. ,   9. ,  10. ,  11. ,   1.1,   1.1],  # noqa: E203
             [1.1,  12. ,  13. ,  14. ,  15. ,  16. ,  17. ,   1.1,   1.1],  # noqa: E203
             [1.1,  18. ,  19. ,  20. ,  21. ,  22. ,  23. ,   1.1,   1.1],  # noqa: E203
             [1.1,  24. ,  25. ,  26. ,  27. ,  28. ,  29. ,   1.1,   1.1],  # noqa: E203

             [1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1],
             [1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1]]
            )
        assert_allclose(test, expected)

    def test_check_constant_float3(self):
        a = np.arange(100, dtype=float)
        a = np.pad(a, (25, 20), 'constant', constant_values=(-1.1, -1.2))
        b = np.array(
            [-1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1,
             -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1,
             -1.1, -1.1, -1.1, -1.1, -1.1,

             0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2,
             -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2]
            )
        assert_allclose(a, b)

    def test_check_constant_odd_pad_amount(self):
        arr = np.arange(30).reshape(5, 6)
        test = np.pad(arr, ((1,), (2,)), mode='constant',
                   constant_values=3)
        expected = np.array(
            [[3,  3,  3,  3,  3,  3,  3,  3,  3,  3],

             [3,  3,  0,  1,  2,  3,  4,  5,  3,  3],
             [3,  3,  6,  7,  8,  9, 10, 11,  3,  3],
             [3,  3, 12, 13, 14, 15, 16, 17,  3,  3],
             [3,  3, 18, 19, 20, 21, 22, 23,  3,  3],
             [3,  3, 24, 25, 26, 27, 28, 29,  3,  3],

             [3,  3,  3,  3,  3,  3,  3,  3,  3,  3]]
            )
        assert_allclose(test, expected)

    def test_check_constant_pad_2d(self):
        arr = np.arange(4).reshape(2, 2)
        test = np.pad(arr, ((1, 2), (1, 3)), mode='constant',
                          constant_values=((1, 2), (3, 4)))
        expected = np.array(
            [[3, 1, 1, 4, 4, 4],
             [3, 0, 1, 4, 4, 4],
             [3, 2, 3, 4, 4, 4],
             [3, 2, 2, 4, 4, 4],
             [3, 2, 2, 4, 4, 4]]
        )
        assert_allclose(test, expected)

    def test_check_large_integers(self):
        uint64_max = 2 ** 64 - 1
        arr = np.full(5, uint64_max, dtype=np.uint64)
        test = np.pad(arr, 1, mode="constant", constant_values=arr.min())
        expected = np.full(7, uint64_max, dtype=np.uint64)
        assert_array_equal(test, expected)

        int64_max = 2 ** 63 - 1
        arr = np.full(5, int64_max, dtype=np.int64)
        test = np.pad(arr, 1, mode="constant", constant_values=arr.min())
        expected = np.full(7, int64_max, dtype=np.int64)
        assert_array_equal(test, expected)

    def test_check_object_array(self):
        arr = np.empty(1, dtype=object)
        obj_a = object()
        arr[0] = obj_a
        obj_b = object()
        obj_c = object()
        arr = np.pad(arr, pad_width=1, mode='constant',
                     constant_values=(obj_b, obj_c))

        expected = np.empty((3,), dtype=object)
        expected[0] = obj_b
        expected[1] = obj_a
        expected[2] = obj_c

        assert_array_equal(arr, expected)

    def test_pad_empty_dimension(self):
        arr = np.zeros((3, 0, 2))
        result = np.pad(arr, [(0,), (2,), (1,)], mode="constant")
        assert result.shape == (3, 4, 4)


class TestLinearRamp:
    def test_check_simple(self):
        a = np.arange(100).astype('f')
        a = np.pad(a, (25, 20), 'linear_ramp', end_values=(4, 5))
        b = np.array(
            [4.00, 3.84, 3.68, 3.52, 3.36, 3.20, 3.04, 2.88, 2.72, 2.56,
             2.40, 2.24, 2.08, 1.92, 1.76, 1.60, 1.44, 1.28, 1.12, 0.96,
             0.80, 0.64, 0.48, 0.32, 0.16,

             0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00,
             10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
             20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
             30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
             40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0,
             50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0,
             60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0,
             70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
             80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0,
             90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,

             94.3, 89.6, 84.9, 80.2, 75.5, 70.8, 66.1, 61.4, 56.7, 52.0,
             47.3, 42.6, 37.9, 33.2, 28.5, 23.8, 19.1, 14.4, 9.7, 5.]
            )
        assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    def test_check_2d(self):
        arr = np.arange(20).reshape(4, 5).astype(np.float64)
        test = np.pad(arr, (2, 2), mode='linear_ramp', end_values=(0, 0))
        expected = np.array(
            [[0.,   0.,   0.,   0.,   0.,   0.,   0.,    0.,   0.],
             [0.,   0.,   0.,  0.5,   1.,  1.5,   2.,    1.,   0.],
             [0.,   0.,   0.,   1.,   2.,   3.,   4.,    2.,   0.],
             [0.,  2.5,   5.,   6.,   7.,   8.,   9.,   4.5,   0.],
             [0.,   5.,  10.,  11.,  12.,  13.,  14.,    7.,   0.],
             [0.,  7.5,  15.,  16.,  17.,  18.,  19.,   9.5,   0.],
             [0., 3.75,  7.5,   8.,  8.5,   9.,  9.5,  4.75,   0.],
             [0.,   0.,   0.,   0.,   0.,   0.,   0.,    0.,   0.]])
        assert_allclose(test, expected)

    @pytest.mark.xfail(exceptions=(AssertionError,))
    def test_object_array(self):
        from fractions import Fraction
        arr = np.array([Fraction(1, 2), Fraction(-1, 2)])
        actual = np.pad(arr, (2, 3), mode='linear_ramp', end_values=0)

        # deliberately chosen to have a non-power-of-2 denominator such that
        # rounding to floats causes a failure.
        expected = np.array([
            Fraction( 0, 12),
            Fraction( 3, 12),
            Fraction( 6, 12),
            Fraction(-6, 12),
            Fraction(-4, 12),
            Fraction(-2, 12),
            Fraction(-0, 12),
        ])
        assert_equal(actual, expected)

    def test_end_values(self):
        """Ensure that end values are exact."""
        a = np.pad(np.ones(10).reshape(2, 5), (223, 123), mode="linear_ramp")
        assert_equal(a[:, 0], 0.)
        assert_equal(a[:, -1], 0.)
        assert_equal(a[0, :], 0.)
        assert_equal(a[-1, :], 0.)

    @pytest.mark.parametrize("dtype", _numeric_dtypes)
    def test_negative_difference(self, dtype):
        """
        Check correct behavior of unsigned dtypes if there is a negative
        difference between the edge to pad and `end_values`. Check both cases
        to be independent of implementation. Test behavior for all other dtypes
        in case dtype casting interferes with complex dtypes. See gh-14191.
        """
        x = np.array([3], dtype=dtype)
        result = np.pad(x, 3, mode="linear_ramp", end_values=0)
        expected = np.array([0, 1, 2, 3, 2, 1, 0], dtype=dtype)
        assert_equal(result, expected)

        x = np.array([0], dtype=dtype)
        result = np.pad(x, 3, mode="linear_ramp", end_values=3)
        expected = np.array([3, 2, 1, 0, 1, 2, 3], dtype=dtype)
        assert_equal(result, expected)


class TestReflect:
    def test_check_simple(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), 'reflect')
        b = np.array(
            [25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
             15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
             5, 4, 3, 2, 1,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             98, 97, 96, 95, 94, 93, 92, 91, 90, 89,
             88, 87, 86, 85, 84, 83, 82, 81, 80, 79]
            )
        assert_array_equal(a, b)

    def test_check_odd_method(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), 'reflect', reflect_type='odd')
        b = np.array(
            [-25, -24, -23, -22, -21, -20, -19, -18, -17, -16,
             -15, -14, -13, -12, -11, -10, -9, -8, -7, -6,
             -5, -4, -3, -2, -1,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
             110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
            )
        assert_array_equal(a, b)

    def test_check_large_pad(self):
        a = [[4, 5, 6], [6, 7, 8]]
        a = np.pad(a, (5, 7), 'reflect')
        b = np.array(
            [[7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],

             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],

             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5]]
            )
        assert_array_equal(a, b)

    def test_check_shape(self):
        a = [[4, 5, 6]]
        a = np.pad(a, (5, 7), 'reflect')
        b = np.array(
            [[5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],

             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],

             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5]]
            )
        assert_array_equal(a, b)

    def test_check_01(self):
        a = np.pad([1, 2, 3], 2, 'reflect')
        b = np.array([3, 2, 1, 2, 3, 2, 1])
        assert_array_equal(a, b)

    def test_check_02(self):
        a = np.pad([1, 2, 3], 3, 'reflect')
        b = np.array([2, 3, 2, 1, 2, 3, 2, 1, 2])
        assert_array_equal(a, b)

    def test_check_03(self):
        a = np.pad([1, 2, 3], 4, 'reflect')
        b = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])
        assert_array_equal(a, b)

    def test_check_04(self):
        a = np.pad([1, 2, 3], [1, 10], 'reflect')
        b = np.array([2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1])
        assert_array_equal(a, b)

    def test_check_05(self):
        a = np.pad([1, 2, 3, 4], [45, 10], 'reflect')
        b = np.array(
            [4, 3, 2, 1, 2, 3, 4, 3, 2, 1,
             2, 3, 4, 3, 2, 1, 2, 3, 4, 3,
             2, 1, 2, 3, 4, 3, 2, 1, 2, 3,
             4, 3, 2, 1, 2, 3, 4, 3, 2, 1,
             2, 3, 4, 3, 2, 1, 2, 3, 4, 3,
             2, 1, 2, 3, 4, 3, 2, 1, 2])
        assert_array_equal(a, b)

    def test_check_06(self):
        a = np.pad([1, 2, 3, 4], [15, 2], 'symmetric')
        b = np.array(
            [2, 3, 4, 4, 3, 2, 1, 1, 2, 3,
             4, 4, 3, 2, 1, 1, 2, 3, 4, 4,
             3]
        )
        assert_array_equal(a, b)

    def test_check_07(self):
        a = np.pad([1, 2, 3, 4, 5, 6], [45, 3], 'symmetric')
        b = np.array(
            [4, 5, 6, 6, 5, 4, 3, 2, 1, 1,
             2, 3, 4, 5, 6, 6, 5, 4, 3, 2,
             1, 1, 2, 3, 4, 5, 6, 6, 5, 4,
             3, 2, 1, 1, 2, 3, 4, 5, 6, 6,
             5, 4, 3, 2, 1, 1, 2, 3, 4, 5,
             6, 6, 5, 4])
        assert_array_equal(a, b)


class TestEmptyArray:
    """Check how padding behaves on arrays with an empty dimension."""

    @pytest.mark.parametrize(
        # Keep parametrization ordered, otherwise pytest-xdist might believe
        # that different tests were collected during parallelization
        "mode", sorted(_all_modes.keys() - {"constant", "empty"})
    )
    def test_pad_empty_dimension(self, mode):
        match = ("can't extend empty axis 0 using modes other than 'constant' "
                 "or 'empty'")
        with pytest.raises(ValueError, match=match):
            np.pad([], 4, mode=mode)
        with pytest.raises(ValueError, match=match):
            np.pad(np.ndarray(0), 4, mode=mode)
        with pytest.raises(ValueError, match=match):
            np.pad(np.zeros((0, 3)), ((1,), (0,)), mode=mode)

    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_pad_non_empty_dimension(self, mode):
        result = np.pad(np.ones((2, 0, 2)), ((3,), (0,), (1,)), mode=mode)
        assert result.shape == (8, 0, 4)


class TestSymmetric:
    def test_check_simple(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), 'symmetric')
        b = np.array(
            [24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
             14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
             4, 3, 2, 1, 0,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             99, 98, 97, 96, 95, 94, 93, 92, 91, 90,
             89, 88, 87, 86, 85, 84, 83, 82, 81, 80]
            )
        assert_array_equal(a, b)

    def test_check_odd_method(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), 'symmetric', reflect_type='odd')
        b = np.array(
            [-24, -23, -22, -21, -20, -19, -18, -17, -16, -15,
             -14, -13, -12, -11, -10, -9, -8, -7, -6, -5,
             -4, -3, -2, -1, 0,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
             109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
            )
        assert_array_equal(a, b)

    def test_check_large_pad(self):
        a = [[4, 5, 6], [6, 7, 8]]
        a = np.pad(a, (5, 7), 'symmetric')
        b = np.array(
            [[5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],

             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],

             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6]]
            )

        assert_array_equal(a, b)

    def test_check_large_pad_odd(self):
        a = [[4, 5, 6], [6, 7, 8]]
        a = np.pad(a, (5, 7), 'symmetric', reflect_type='odd')
        b = np.array(
            [[-3, -2, -2, -1,  0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6],
             [-3, -2, -2, -1,  0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6],
             [-1,  0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8],
             [-1,  0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8],
             [ 1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8,  9, 10, 10],

             [ 1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8,  9, 10, 10],
             [ 3,  4,  4,  5,  6,  6,  7,  8,  8,  9, 10, 10, 11, 12, 12],

             [ 3,  4,  4,  5,  6,  6,  7,  8,  8,  9, 10, 10, 11, 12, 12],
             [ 5,  6,  6,  7,  8,  8,  9, 10, 10, 11, 12, 12, 13, 14, 14],
             [ 5,  6,  6,  7,  8,  8,  9, 10, 10, 11, 12, 12, 13, 14, 14],
             [ 7,  8,  8,  9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16],
             [ 7,  8,  8,  9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16],
             [ 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18],
             [ 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18]]
            )
        assert_array_equal(a, b)

    def test_check_shape(self):
        a = [[4, 5, 6]]
        a = np.pad(a, (5, 7), 'symmetric')
        b = np.array(
            [[5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],

             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],

             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6]]
            )
        assert_array_equal(a, b)

    def test_check_01(self):
        a = np.pad([1, 2, 3], 2, 'symmetric')
        b = np.array([2, 1, 1, 2, 3, 3, 2])
        assert_array_equal(a, b)

    def test_check_02(self):
        a = np.pad([1, 2, 3], 3, 'symmetric')
        b = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1])
        assert_array_equal(a, b)

    def test_check_03(self):
        a = np.pad([1, 2, 3], 6, 'symmetric')
        b = np.array([1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3])
        assert_array_equal(a, b)


class TestWrap:
    def test_check_simple(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), 'wrap')
        b = np.array(
            [75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
             85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
             95, 96, 97, 98, 99,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            )
        assert_array_equal(a, b)

    def test_check_large_pad(self):
        a = np.arange(12)
        a = np.reshape(a, (3, 4))
        a = np.pad(a, (10, 12), 'wrap')
        b = np.array(
            [[10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10,
              11, 8, 9, 10, 11, 8, 9, 10, 11],
             [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
              3, 0, 1, 2, 3, 0, 1, 2, 3],
             [6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6,
              7, 4, 5, 6, 7, 4, 5, 6, 7],
             [10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10,
              11, 8, 9, 10, 11, 8, 9, 10, 11],
             [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
              3, 0, 1, 2, 3, 0, 1, 2, 3],
             [6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6,
              7, 4, 5, 6, 7, 4, 5, 6, 7],
             [10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10,
              11, 8, 9, 10, 11, 8, 9, 10, 11],
             [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
              3, 0, 1, 2, 3, 0, 1, 2, 3],
             [6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6,
              7, 4, 5, 6, 7, 4, 5, 6, 7],
             [10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10,
              11, 8, 9, 10, 11, 8, 9, 10, 11],

             [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
              3, 0, 1, 2, 3, 0, 1, 2, 3],
             [6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6,
              7, 4, 5, 6, 7, 4, 5, 6, 7],
             [10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10,
              11, 8, 9, 10, 11, 8, 9, 10, 11],

             [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
              3, 0, 1, 2, 3, 0, 1, 2, 3],
             [6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6,
              7, 4, 5, 6, 7, 4, 5, 6, 7],
             [10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10,
              11, 8, 9, 10, 11, 8, 9, 10, 11],
             [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
              3, 0, 1, 2, 3, 0, 1, 2, 3],
             [6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6,
              7, 4, 5, 6, 7, 4, 5, 6, 7],
             [10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10,
              11, 8, 9, 10, 11, 8, 9, 10, 11],
             [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
              3, 0, 1, 2, 3, 0, 1, 2, 3],
             [6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6,
              7, 4, 5, 6, 7, 4, 5, 6, 7],
             [10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10,
              11, 8, 9, 10, 11, 8, 9, 10, 11],
             [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
              3, 0, 1, 2, 3, 0, 1, 2, 3],
             [6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6,
              7, 4, 5, 6, 7, 4, 5, 6, 7],
             [10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10,
              11, 8, 9, 10, 11, 8, 9, 10, 11]]
            )
        assert_array_equal(a, b)

    def test_check_01(self):
        a = np.pad([1, 2, 3], 3, 'wrap')
        b = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        assert_array_equal(a, b)

    def test_check_02(self):
        a = np.pad([1, 2, 3], 4, 'wrap')
        b = np.array([3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
        assert_array_equal(a, b)

    def test_pad_with_zero(self):
        a = np.ones((3, 5))
        b = np.pad(a, (0, 5), mode="wrap")
        assert_array_equal(a, b[:-5, :-5])

    def test_repeated_wrapping(self):
        """
        Check wrapping on each side individually if the wrapped area is longer
        than the original array.
        """
        a = np.arange(5)
        b = np.pad(a, (12, 0), mode="wrap")
        assert_array_equal(np.r_[a, a, a, a][3:], b)

        a = np.arange(5)
        b = np.pad(a, (0, 12), mode="wrap")
        assert_array_equal(np.r_[a, a, a, a][:-3], b)

    def test_repeated_wrapping_multiple_origin(self):
        """
        Assert that 'wrap' pads only with multiples of the original area if
        the pad width is larger than the original array.
        """
        a = np.arange(4).reshape(2, 2)
        a = np.pad(a, [(1, 3), (3, 1)], mode='wrap')
        b = np.array(
            [[3, 2, 3, 2, 3, 2],
             [1, 0, 1, 0, 1, 0],
             [3, 2, 3, 2, 3, 2],
             [1, 0, 1, 0, 1, 0],
             [3, 2, 3, 2, 3, 2],
             [1, 0, 1, 0, 1, 0]]
        )
        assert_array_equal(a, b)


class TestEdge:
    def test_check_simple(self):
        a = np.arange(12)
        a = np.reshape(a, (4, 3))
        a = np.pad(a, ((2, 3), (3, 2)), 'edge')
        b = np.array(
            [[0, 0, 0, 0, 1, 2, 2, 2],
             [0, 0, 0, 0, 1, 2, 2, 2],

             [0, 0, 0, 0, 1, 2, 2, 2],
             [3, 3, 3, 3, 4, 5, 5, 5],
             [6, 6, 6, 6, 7, 8, 8, 8],
             [9, 9, 9, 9, 10, 11, 11, 11],

             [9, 9, 9, 9, 10, 11, 11, 11],
             [9, 9, 9, 9, 10, 11, 11, 11],
             [9, 9, 9, 9, 10, 11, 11, 11]]
            )
        assert_array_equal(a, b)

    def test_check_width_shape_1_2(self):
        # Check a pad_width of the form ((1, 2),).
        # Regression test for issue gh-7808.
        a = np.array([1, 2, 3])
        padded = np.pad(a, ((1, 2),), 'edge')
        expected = np.array([1, 1, 2, 3, 3, 3])
        assert_array_equal(padded, expected)

        a = np.array([[1, 2, 3], [4, 5, 6]])
        padded = np.pad(a, ((1, 2),), 'edge')
        expected = np.pad(a, ((1, 2), (1, 2)), 'edge')
        assert_array_equal(padded, expected)

        a = np.arange(24).reshape(2, 3, 4)
        padded = np.pad(a, ((1, 2),), 'edge')
        expected = np.pad(a, ((1, 2), (1, 2), (1, 2)), 'edge')
        assert_array_equal(padded, expected)


class TestEmpty:
    def test_simple(self):
        arr = np.arange(24).reshape(4, 6)
        result = np.pad(arr, [(2, 3), (3, 1)], mode="empty")
        assert result.shape == (9, 10)
        assert_equal(arr, result[2:-3, 3:-1])

    def test_pad_empty_dimension(self):
        arr = np.zeros((3, 0, 2))
        result = np.pad(arr, [(0,), (2,), (1,)], mode="empty")
        assert result.shape == (3, 4, 4)


def test_legacy_vector_functionality():
    def _padwithtens(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 10
        vector[-pad_width[1]:] = 10

    a = np.arange(6).reshape(2, 3)
    a = np.pad(a, 2, _padwithtens)
    b = np.array(
        [[10, 10, 10, 10, 10, 10, 10],
         [10, 10, 10, 10, 10, 10, 10],

         [10, 10,  0,  1,  2, 10, 10],
         [10, 10,  3,  4,  5, 10, 10],

         [10, 10, 10, 10, 10, 10, 10],
         [10, 10, 10, 10, 10, 10, 10]]
        )
    assert_array_equal(a, b)


def test_unicode_mode():
    a = np.pad([1], 2, mode='constant')
    b = np.array([0, 0, 1, 0, 0])
    assert_array_equal(a, b)


@pytest.mark.parametrize("mode", ["edge", "symmetric", "reflect", "wrap"])
def test_object_input(mode):
    # Regression test for issue gh-11395.
    a = np.full((4, 3), fill_value=None)
    pad_amt = ((2, 3), (3, 2))
    b = np.full((9, 8), fill_value=None)
    assert_array_equal(np.pad(a, pad_amt, mode=mode), b)


class TestPadWidth:
    @pytest.mark.parametrize("pad_width", [
        (4, 5, 6, 7),
        ((1,), (2,), (3,)),
        ((1, 2), (3, 4), (5, 6)),
        ((3, 4, 5), (0, 1, 2)),
    ])
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_misshaped_pad_width(self, pad_width, mode):
        arr = np.arange(30).reshape((6, 5))
        match = "operands could not be broadcast together"
        with pytest.raises(ValueError, match=match):
            np.pad(arr, pad_width, mode)

    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_misshaped_pad_width_2(self, mode):
        arr = np.arange(30).reshape((6, 5))
        match = ("input operand has more dimensions than allowed by the axis "
                 "remapping")
        with pytest.raises(ValueError, match=match):
            np.pad(arr, (((3,), (4,), (5,)), ((0,), (1,), (2,))), mode)

    @pytest.mark.parametrize(
        "pad_width", [-2, (-2,), (3, -1), ((5, 2), (-2, 3)), ((-4,), (2,))])
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_negative_pad_width(self, pad_width, mode):
        arr = np.arange(30).reshape((6, 5))
        match = "index can't contain negative values"
        with pytest.raises(ValueError, match=match):
            np.pad(arr, pad_width, mode)

    @pytest.mark.parametrize("pad_width, dtype", [
        ("3", None),
        ("word", None),
        (None, None),
        (object(), None),
        (3.4, None),
        (((2, 3, 4), (3, 2)), object),
        (complex(1, -1), None),
        (((-2.1, 3), (3, 2)), None),
    ])
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_bad_type(self, pad_width, dtype, mode):
        arr = np.arange(30).reshape((6, 5))
        match = "`pad_width` must be of integral type."
        if dtype is not None:
            # avoid DeprecationWarning when not specifying dtype
            with pytest.raises(TypeError, match=match):
                np.pad(arr, np.array(pad_width, dtype=dtype), mode)
        else:
            with pytest.raises(TypeError, match=match):
                np.pad(arr, pad_width, mode)
            with pytest.raises(TypeError, match=match):
                np.pad(arr, np.array(pad_width), mode)

    def test_pad_width_as_ndarray(self):
        a = np.arange(12)
        a = np.reshape(a, (4, 3))
        a = np.pad(a, np.array(((2, 3), (3, 2))), 'edge')
        b = np.array(
            [[0,  0,  0,    0,  1,  2,    2,  2],
             [0,  0,  0,    0,  1,  2,    2,  2],

             [0,  0,  0,    0,  1,  2,    2,  2],
             [3,  3,  3,    3,  4,  5,    5,  5],
             [6,  6,  6,    6,  7,  8,    8,  8],
             [9,  9,  9,    9, 10, 11,   11, 11],

             [9,  9,  9,    9, 10, 11,   11, 11],
             [9,  9,  9,    9, 10, 11,   11, 11],
             [9,  9,  9,    9, 10, 11,   11, 11]]
            )
        assert_array_equal(a, b)

    @pytest.mark.parametrize("pad_width", [0, (0, 0), ((0, 0), (0, 0))])
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_zero_pad_width(self, pad_width, mode):
        arr = np.arange(30).reshape(6, 5)
        assert_array_equal(arr, np.pad(arr, pad_width, mode=mode))


@pytest.mark.parametrize("mode", _all_modes.keys())
def test_kwargs(mode):
    """Test behavior of pad's kwargs for the given mode."""
    allowed = _all_modes[mode]
    not_allowed = {}
    for kwargs in _all_modes.values():
        if kwargs != allowed:
            not_allowed.update(kwargs)
    # Test if allowed keyword arguments pass
    np.pad([1, 2, 3], 1, mode, **allowed)
    # Test if prohibited keyword arguments of other modes raise an error
    for key, value in not_allowed.items():
        match = f"unsupported keyword arguments for mode '{mode}'"
        with pytest.raises(ValueError, match=match):
            np.pad([1, 2, 3], 1, mode, **{key: value})


def test_constant_zero_default():
    arr = np.array([1, 1])
    assert_array_equal(np.pad(arr, 2), [0, 0, 1, 1, 0, 0])


@pytest.mark.parametrize("mode", [1, "const", object(), None, True, False])
def test_unsupported_mode(mode):
    match = f"mode '{mode}' is not supported"
    with pytest.raises(ValueError, match=match):
        np.pad([1, 2, 3], 4, mode=mode)


@pytest.mark.parametrize("mode", _all_modes.keys())
def test_non_contiguous_array(mode):
    arr = np.arange(24).reshape(4, 6)[::2, ::2]
    result = np.pad(arr, (2, 3), mode)
    assert result.shape == (7, 8)
    assert_equal(result[2:-3, 2:-3], arr)


@pytest.mark.parametrize("mode", _all_modes.keys())
def test_memory_layout_persistence(mode):
    """Test if C and F order is preserved for all pad modes."""
    x = np.ones((5, 10), order='C')
    assert np.pad(x, 5, mode).flags["C_CONTIGUOUS"]
    x = np.ones((5, 10), order='F')
    assert np.pad(x, 5, mode).flags["F_CONTIGUOUS"]


@pytest.mark.parametrize("dtype", _numeric_dtypes)
@pytest.mark.parametrize("mode", _all_modes.keys())
def test_dtype_persistence(dtype, mode):
    arr = np.zeros((3, 2, 1), dtype=dtype)
    result = np.pad(arr, 1, mode=mode)
    assert result.dtype == dtype

# === atelier-kyo-manager/app\forms.py ===
# app/forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, BooleanField, SubmitField, URLField
from wtforms.validators import DataRequired, NumberRange, URL, Optional

class ProductForm(FlaskForm):
    # 基本情報
    name = StringField('商品名', validators=[DataRequired()])
    brand = StringField('ブランド')
    purchase_price = FloatField('仕入価格', validators=[
        DataRequired(),
        NumberRange(min=0, message="0以上の数値を入力してください")
    ])
    selling_price = FloatField('販売価格', validators=[
        DataRequired(),
        NumberRange(min=0, message="0以上の数値を入力してください")
    ])

    # 経費項目（追加部分）
    transaction_fee = FloatField('取引手数料', validators=[Optional()])
    shipping_cost = FloatField('送料・梱包費', validators=[Optional()])
    customs_duty = FloatField('関税・輸入消費税', validators=[Optional()])
    procurement_fee = FloatField('買付代行料', validators=[Optional()])

    # URL関連
    supplier_url = URLField('仕入先URL', validators=[
        URL(message='有効なURLを入力してください', require_tld=False)
    ])
    image_url = URLField('画像URL', validators=[
        URL(message='有効なURLを入力してください', require_tld=False)
    ])

    # 在庫
    stock_status = BooleanField('在庫あり')

    submit = SubmitField('登録/更新')

class ProfitFilterForm(FlaskForm):
    min_profit = FloatField('最低利益額', validators=[
        NumberRange(min=0, message="0以上の数値を入力してください")
    ])
    submit = SubmitField('絞り込み')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_shape_base_impl.py ===
import functools
import warnings

import numpy._core.numeric as _nx
from numpy._core import atleast_3d, overrides, vstack
from numpy._core._multiarray_umath import _array_converter
from numpy._core.fromnumeric import reshape, transpose
from numpy._core.multiarray import normalize_axis_index
from numpy._core.numeric import (
    array,
    asanyarray,
    asarray,
    normalize_axis_tuple,
    zeros,
    zeros_like,
)
from numpy._core.overrides import set_module
from numpy._core.shape_base import _arrays_for_stack_dispatcher
from numpy.lib._index_tricks_impl import ndindex
from numpy.matrixlib.defmatrix import matrix  # this raises all the right alarm bells

__all__ = [
    'column_stack', 'row_stack', 'dstack', 'array_split', 'split',
    'hsplit', 'vsplit', 'dsplit', 'apply_over_axes', 'expand_dims',
    'apply_along_axis', 'kron', 'tile', 'take_along_axis',
    'put_along_axis'
    ]


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over
    if not _nx.issubdtype(indices.dtype, _nx.integer):
        raise IndexError('`indices` must be an integer array')
    if len(arr_shape) != indices.ndim:
        raise ValueError(
            "`indices` and `arr` must have the same number of dimensions")
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1:]
            fancy_index.append(_nx.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def _take_along_axis_dispatcher(arr, indices, axis=None):
    return (arr, indices)


@array_function_dispatch(_take_along_axis_dispatcher)
def take_along_axis(arr, indices, axis=-1):
    """
    Take values from the input array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to look up values in the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like `argsort` and
    `argpartition`, produce suitable indices for this function.

    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        Source array
    indices : ndarray (Ni..., J, Nk...)
        Indices to take along each 1d slice of ``arr``. This must match the
        dimension of ``arr``, but dimensions Ni and Nj only need to broadcast
        against ``arr``.
    axis : int or None, optional
        The axis to take 1d slices along. If axis is None, the input array is
        treated as if it had first been flattened to 1d, for consistency with
        `sort` and `argsort`.

        .. versionchanged:: 2.3
            The default value is now ``-1``.

    Returns
    -------
    out: ndarray (Ni..., J, Nk...)
        The indexed result.

    Notes
    -----
    This is equivalent to (but faster than) the following use of `ndindex` and
    `s_`, which sets each of ``ii`` and ``kk`` to a tuple of indices::

        Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis+1:]
        J = indices.shape[axis]  # Need not equal M
        out = np.empty(Ni + (J,) + Nk)

        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                a_1d       = a      [ii + s_[:,] + kk]
                indices_1d = indices[ii + s_[:,] + kk]
                out_1d     = out    [ii + s_[:,] + kk]
                for j in range(J):
                    out_1d[j] = a_1d[indices_1d[j]]

    Equivalently, eliminating the inner loop, the last two lines would be::

                out_1d[:] = a_1d[indices_1d]

    See Also
    --------
    take : Take along an axis, using the same indices for every 1d slice
    put_along_axis :
        Put values into the destination array by matching 1d index and data slices

    Examples
    --------
    >>> import numpy as np

    For this sample array

    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    We can sort either by using sort directly, or argsort and this function

    >>> np.sort(a, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])
    >>> ai = np.argsort(a, axis=1)
    >>> ai
    array([[0, 2, 1],
           [1, 2, 0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])

    The same works for max and min, if you maintain the trivial dimension
    with ``keepdims``:

    >>> np.max(a, axis=1, keepdims=True)
    array([[30],
           [60]])
    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> ai
    array([[1],
           [0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[30],
           [60]])

    If we want to get the max and min at the same time, we can stack the
    indices first

    >>> ai_min = np.argmin(a, axis=1, keepdims=True)
    >>> ai_max = np.argmax(a, axis=1, keepdims=True)
    >>> ai = np.concatenate([ai_min, ai_max], axis=1)
    >>> ai
    array([[0, 1],
           [1, 0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 30],
           [40, 60]])
    """
    # normalize inputs
    if axis is None:
        if indices.ndim != 1:
            raise ValueError(
                'when axis=None, `indices` must have a single dimension.')
        arr = arr.flat
        arr_shape = (len(arr),)  # flatiter has no .shape
        axis = 0
    else:
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

    # use the fancy index
    return arr[_make_along_axis_idx(arr_shape, indices, axis)]


def _put_along_axis_dispatcher(arr, indices, values, axis):
    return (arr, indices, values)


@array_function_dispatch(_put_along_axis_dispatcher)
def put_along_axis(arr, indices, values, axis):
    """
    Put values into the destination array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to place values into the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like `argsort` and
    `argpartition`, produce suitable indices for this function.

    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        Destination array.
    indices : ndarray (Ni..., J, Nk...)
        Indices to change along each 1d slice of `arr`. This must match the
        dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
        against `arr`.
    values : array_like (Ni..., J, Nk...)
        values to insert at those indices. Its shape and dimension are
        broadcast to match that of `indices`.
    axis : int
        The axis to take 1d slices along. If axis is None, the destination
        array is treated as if a flattened 1d view had been created of it.

    Notes
    -----
    This is equivalent to (but faster than) the following use of `ndindex` and
    `s_`, which sets each of ``ii`` and ``kk`` to a tuple of indices::

        Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis+1:]
        J = indices.shape[axis]  # Need not equal M

        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                a_1d       = a      [ii + s_[:,] + kk]
                indices_1d = indices[ii + s_[:,] + kk]
                values_1d  = values [ii + s_[:,] + kk]
                for j in range(J):
                    a_1d[indices_1d[j]] = values_1d[j]

    Equivalently, eliminating the inner loop, the last two lines would be::

                a_1d[indices_1d] = values_1d

    See Also
    --------
    take_along_axis :
        Take values from the input array by matching 1d index and data slices

    Examples
    --------
    >>> import numpy as np

    For this sample array

    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    We can replace the maximum values with:

    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> ai
    array([[1],
           [0]])
    >>> np.put_along_axis(a, ai, 99, axis=1)
    >>> a
    array([[10, 99, 20],
           [99, 40, 50]])

    """
    # normalize inputs
    if axis is None:
        if indices.ndim != 1:
            raise ValueError(
                'when axis=None, `indices` must have a single dimension.')
        arr = arr.flat
        axis = 0
        arr_shape = (len(arr),)  # flatiter has no .shape
    else:
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

    # use the fancy index
    arr[_make_along_axis_idx(arr_shape, indices, axis)] = values


def _apply_along_axis_dispatcher(func1d, axis, arr, *args, **kwargs):
    return (arr,)


@array_function_dispatch(_apply_along_axis_dispatcher)
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Apply a function to 1-D slices along the given axis.

    Execute `func1d(a, *args, **kwargs)` where `func1d` operates on 1-D arrays
    and `a` is a 1-D slice of `arr` along `axis`.

    This is equivalent to (but faster than) the following use of `ndindex` and
    `s_`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                f = func1d(arr[ii + s_[:,] + kk])
                Nj = f.shape
                for jj in ndindex(Nj):
                    out[ii + jj + kk] = f[jj]

    Equivalently, eliminating the inner loop, this can be expressed as::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                out[ii + s_[...,] + kk] = func1d(arr[ii + s_[:,] + kk])

    Parameters
    ----------
    func1d : function (M,) -> (Nj...)
        This function should accept 1-D arrays. It is applied to 1-D
        slices of `arr` along the specified axis.
    axis : integer
        Axis along which `arr` is sliced.
    arr : ndarray (Ni..., M, Nk...)
        Input array.
    args : any
        Additional arguments to `func1d`.
    kwargs : any
        Additional named arguments to `func1d`.

    Returns
    -------
    out : ndarray  (Ni..., Nj..., Nk...)
        The output array. The shape of `out` is identical to the shape of
        `arr`, except along the `axis` dimension. This axis is removed, and
        replaced with new dimensions equal to the shape of the return value
        of `func1d`. So if `func1d` returns a scalar `out` will have one
        fewer dimensions than `arr`.

    See Also
    --------
    apply_over_axes : Apply a function repeatedly over multiple axes.

    Examples
    --------
    >>> import numpy as np
    >>> def my_func(a):
    ...     \"\"\"Average first and last element of a 1-D array\"\"\"
    ...     return (a[0] + a[-1]) * 0.5
    >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> np.apply_along_axis(my_func, 0, b)
    array([4., 5., 6.])
    >>> np.apply_along_axis(my_func, 1, b)
    array([2.,  5.,  8.])

    For a function that returns a 1D array, the number of dimensions in
    `outarr` is the same as `arr`.

    >>> b = np.array([[8,1,7], [4,3,9], [5,2,6]])
    >>> np.apply_along_axis(sorted, 1, b)
    array([[1, 7, 8],
           [3, 4, 9],
           [2, 5, 6]])

    For a function that returns a higher dimensional array, those dimensions
    are inserted in place of the `axis` dimension.

    >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> np.apply_along_axis(np.diag, -1, b)
    array([[[1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]],
           [[4, 0, 0],
            [0, 5, 0],
            [0, 0, 6]],
           [[7, 0, 0],
            [0, 8, 0],
            [0, 0, 9]]])
    """
    # handle negative axes
    conv = _array_converter(arr)
    arr = conv[0]

    nd = arr.ndim
    axis = normalize_axis_index(axis, nd)

    # arr, with the iteration axis at the end
    in_dims = list(range(nd))
    inarr_view = transpose(arr, in_dims[:axis] + in_dims[axis + 1:] + [axis])

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars, which fixes gh-8642
    inds = ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError(
            'Cannot apply_along_axis when any iteration dimensions are 0'
        ) from None
    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    if not isinstance(res, matrix):
        buff = zeros_like(res, shape=inarr_view.shape[:-1] + res.shape)
    else:
        # Matrices are nasty with reshaping, so do not preserve them here.
        buff = zeros(inarr_view.shape[:-1] + res.shape, dtype=res.dtype)

    # permutation of axes such that out = buff.transpose(buff_permute)
    buff_dims = list(range(buff.ndim))
    buff_permute = (
        buff_dims[0 : axis] +
        buff_dims[buff.ndim - res.ndim : buff.ndim] +
        buff_dims[axis : buff.ndim - res.ndim]
    )

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = asanyarray(func1d(inarr_view[ind], *args, **kwargs))

    res = transpose(buff, buff_permute)
    return conv.wrap(res)


def _apply_over_axes_dispatcher(func, a, axes):
    return (a,)


@array_function_dispatch(_apply_over_axes_dispatcher)
def apply_over_axes(func, a, axes):
    """
    Apply a function repeatedly over multiple axes.

    `func` is called as `res = func(a, axis)`, where `axis` is the first
    element of `axes`.  The result `res` of the function call must have
    either the same dimensions as `a` or one less dimension.  If `res`
    has one less dimension than `a`, a dimension is inserted before
    `axis`.  The call to `func` is then repeated for each axis in `axes`,
    with `res` as the first argument.

    Parameters
    ----------
    func : function
        This function must take two arguments, `func(a, axis)`.
    a : array_like
        Input array.
    axes : array_like
        Axes over which `func` is applied; the elements must be integers.

    Returns
    -------
    apply_over_axis : ndarray
        The output array.  The number of dimensions is the same as `a`,
        but the shape can be different.  This depends on whether `func`
        changes the shape of its output with respect to its input.

    See Also
    --------
    apply_along_axis :
        Apply a function to 1-D slices of an array along the given axis.

    Notes
    -----
    This function is equivalent to tuple axis arguments to reorderable ufuncs
    with keepdims=True. Tuple axis arguments to ufuncs have been available since
    version 1.7.0.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(24).reshape(2,3,4)
    >>> a
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])

    Sum over axes 0 and 2. The result has same number of dimensions
    as the original array:

    >>> np.apply_over_axes(np.sum, a, [0,2])
    array([[[ 60],
            [ 92],
            [124]]])

    Tuple axis arguments to ufuncs are equivalent:

    >>> np.sum(a, axis=(0,2), keepdims=True)
    array([[[ 60],
            [ 92],
            [124]]])

    """
    val = asarray(a)
    N = a.ndim
    if array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning "
                                 "an array of the correct shape")
    return val


def _expand_dims_dispatcher(a, axis):
    return (a,)


@array_function_dispatch(_expand_dims_dispatcher)
def expand_dims(a, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or tuple of ints
        Position in the expanded axes where the new axis (or axes) is placed.

        .. deprecated:: 1.13.0
            Passing an axis where ``axis > a.ndim`` will be treated as
            ``axis == a.ndim``, and passing ``axis < -a.ndim - 1`` will
            be treated as ``axis == 0``. This behavior is deprecated.

    Returns
    -------
    result : ndarray
        View of `a` with the number of dimensions increased.

    See Also
    --------
    squeeze : The inverse operation, removing singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones
    atleast_1d, atleast_2d, atleast_3d

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2])
    >>> x.shape
    (2,)

    The following is equivalent to ``x[np.newaxis, :]`` or ``x[np.newaxis]``:

    >>> y = np.expand_dims(x, axis=0)
    >>> y
    array([[1, 2]])
    >>> y.shape
    (1, 2)

    The following is equivalent to ``x[:, np.newaxis]``:

    >>> y = np.expand_dims(x, axis=1)
    >>> y
    array([[1],
           [2]])
    >>> y.shape
    (2, 1)

    ``axis`` may also be a tuple:

    >>> y = np.expand_dims(x, axis=(0, 1))
    >>> y
    array([[[1, 2]]])

    >>> y = np.expand_dims(x, axis=(2, 0))
    >>> y
    array([[[1],
            [2]]])

    Note that some examples may use ``None`` instead of ``np.newaxis``.  These
    are the same objects:

    >>> np.newaxis is None
    True

    """
    if isinstance(a, matrix):
        a = asarray(a)
    else:
        a = asanyarray(a)

    if not isinstance(axis, (tuple, list)):
        axis = (axis,)

    out_ndim = len(axis) + a.ndim
    axis = normalize_axis_tuple(axis, out_ndim)

    shape_it = iter(a.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

    return a.reshape(shape)


# NOTE: Remove once deprecation period passes
@set_module("numpy")
def row_stack(tup, *, dtype=None, casting="same_kind"):
    # Deprecated in NumPy 2.0, 2023-08-18
    warnings.warn(
        "`row_stack` alias is deprecated. "
        "Use `np.vstack` directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return vstack(tup, dtype=dtype, casting=casting)


row_stack.__doc__ = vstack.__doc__


def _column_stack_dispatcher(tup):
    return _arrays_for_stack_dispatcher(tup)


@array_function_dispatch(_column_stack_dispatcher)
def column_stack(tup):
    """
    Stack 1-D arrays as columns into a 2-D array.

    Take a sequence of 1-D arrays and stack them as columns
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `hstack`.  1-D arrays are turned into 2-D columns
    first.

    Parameters
    ----------
    tup : sequence of 1-D or 2-D arrays.
        Arrays to stack. All of them must have the same first dimension.

    Returns
    -------
    stacked : 2-D array
        The array formed by stacking the given arrays.

    See Also
    --------
    stack, hstack, vstack, concatenate

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.column_stack((a,b))
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    arrays = []
    for v in tup:
        arr = asanyarray(v)
        if arr.ndim < 2:
            arr = array(arr, copy=None, subok=True, ndmin=2).T
        arrays.append(arr)
    return _nx.concatenate(arrays, 1)


def _dstack_dispatcher(tup):
    return _arrays_for_stack_dispatcher(tup)


@array_function_dispatch(_dstack_dispatcher)
def dstack(tup):
    """
    Stack arrays in sequence depth wise (along third axis).

    This is equivalent to concatenation along the third axis after 2-D arrays
    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by
    `dsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of arrays
        The arrays must have the same shape along all but the third axis.
        1-D or 2-D arrays must have the same shape.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 3-D.

    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    block : Assemble an nd-array from nested lists of blocks.
    vstack : Stack arrays in sequence vertically (row wise).
    hstack : Stack arrays in sequence horizontally (column wise).
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    dsplit : Split array along third axis.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.dstack((a,b))
    array([[[1, 2],
            [2, 3],
            [3, 4]]])

    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[2],[3],[4]])
    >>> np.dstack((a,b))
    array([[[1, 2]],
           [[2, 3]],
           [[3, 4]]])

    """
    arrs = atleast_3d(*tup)
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    return _nx.concatenate(arrs, 2)


def _replace_zero_by_x_arrays(sub_arys):
    for i in range(len(sub_arys)):
        if _nx.ndim(sub_arys[i]) == 0:
            sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
        elif _nx.sometrue(_nx.equal(_nx.shape(sub_arys[i]), 0)):
            sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
    return sub_arys


def _array_split_dispatcher(ary, indices_or_sections, axis=None):
    return (ary, indices_or_sections)


@array_function_dispatch(_array_split_dispatcher)
def array_split(ary, indices_or_sections, axis=0):
    """
    Split an array into multiple sub-arrays.

    Please refer to the ``split`` documentation.  The only difference
    between these functions is that ``array_split`` allows
    `indices_or_sections` to be an integer that does *not* equally
    divide the axis. For an array of length l that should be split
    into n sections, it returns l % n sub-arrays of size l//n + 1
    and the rest of size l//n.

    See Also
    --------
    split : Split array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(8.0)
    >>> np.array_split(x, 3)
    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.])]

    >>> x = np.arange(9)
    >>> np.array_split(x, 4)
    [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]

    """
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] +
                         extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

    sub_arys = []
    sary = _nx.swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))

    return sub_arys


def _split_dispatcher(ary, indices_or_sections, axis=None):
    return (ary, indices_or_sections)


@array_function_dispatch(_split_dispatcher)
def split(ary, indices_or_sections, axis=0):
    """
    Split an array into multiple sub-arrays as views into `ary`.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.

        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in

        - ary[:2]
        - ary[2:3]
        - ary[3:]

        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays as views into `ary`.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division.

    See Also
    --------
    array_split : Split an array into multiple sub-arrays of equal or
                  near-equal size.  Does not raise an exception if
                  an equal division cannot be made.
    hsplit : Split array into multiple sub-arrays horizontally (column-wise).
    vsplit : Split array into multiple sub-arrays vertically (row wise).
    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
    concatenate : Join a sequence of arrays along an existing axis.
    stack : Join a sequence of arrays along a new axis.
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(9.0)
    >>> np.split(x, 3)
    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.,  8.])]

    >>> x = np.arange(8.0)
    >>> np.split(x, [3, 5, 6, 10])
    [array([0.,  1.,  2.]),
     array([3.,  4.]),
     array([5.]),
     array([6.,  7.]),
     array([], dtype=float64)]

    """
    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = ary.shape[axis]
        if N % sections:
            raise ValueError(
                'array split does not result in an equal division') from None
    return array_split(ary, indices_or_sections, axis)


def _hvdsplit_dispatcher(ary, indices_or_sections):
    return (ary, indices_or_sections)


@array_function_dispatch(_hvdsplit_dispatcher)
def hsplit(ary, indices_or_sections):
    """
    Split an array into multiple sub-arrays horizontally (column-wise).

    Please refer to the `split` documentation.  `hsplit` is equivalent
    to `split` with ``axis=1``, the array is always split along the second
    axis except for 1-D arrays, where it is split at ``axis=0``.

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,   1.,   2.,   3.],
           [ 4.,   5.,   6.,   7.],
           [ 8.,   9.,  10.,  11.],
           [12.,  13.,  14.,  15.]])
    >>> np.hsplit(x, 2)
    [array([[  0.,   1.],
           [  4.,   5.],
           [  8.,   9.],
           [12.,  13.]]),
     array([[  2.,   3.],
           [  6.,   7.],
           [10.,  11.],
           [14.,  15.]])]
    >>> np.hsplit(x, np.array([3, 6]))
    [array([[ 0.,   1.,   2.],
           [ 4.,   5.,   6.],
           [ 8.,   9.,  10.],
           [12.,  13.,  14.]]),
     array([[ 3.],
           [ 7.],
           [11.],
           [15.]]),
     array([], shape=(4, 0), dtype=float64)]

    With a higher dimensional array the split is still along the second axis.

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[0.,  1.],
            [2.,  3.]],
           [[4.,  5.],
            [6.,  7.]]])
    >>> np.hsplit(x, 2)
    [array([[[0.,  1.]],
           [[4.,  5.]]]),
     array([[[2.,  3.]],
           [[6.,  7.]]])]

    With a 1-D array, the split is along axis 0.

    >>> x = np.array([0, 1, 2, 3, 4, 5])
    >>> np.hsplit(x, 2)
    [array([0, 1, 2]), array([3, 4, 5])]

    """
    if _nx.ndim(ary) == 0:
        raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    if ary.ndim > 1:
        return split(ary, indices_or_sections, 1)
    else:
        return split(ary, indices_or_sections, 0)


@array_function_dispatch(_hvdsplit_dispatcher)
def vsplit(ary, indices_or_sections):
    """
    Split an array into multiple sub-arrays vertically (row-wise).

    Please refer to the ``split`` documentation.  ``vsplit`` is equivalent
    to ``split`` with `axis=0` (default), the array is always split along the
    first axis regardless of the array dimension.

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,   1.,   2.,   3.],
           [ 4.,   5.,   6.,   7.],
           [ 8.,   9.,  10.,  11.],
           [12.,  13.,  14.,  15.]])
    >>> np.vsplit(x, 2)
    [array([[0., 1., 2., 3.],
            [4., 5., 6., 7.]]),
     array([[ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])]
    >>> np.vsplit(x, np.array([3, 6]))
    [array([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]]),
     array([[12., 13., 14., 15.]]),
     array([], shape=(0, 4), dtype=float64)]

    With a higher dimensional array the split is still along the first axis.

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[0.,  1.],
            [2.,  3.]],
           [[4.,  5.],
            [6.,  7.]]])
    >>> np.vsplit(x, 2)
    [array([[[0., 1.],
             [2., 3.]]]),
     array([[[4., 5.],
             [6., 7.]]])]

    """
    if _nx.ndim(ary) < 2:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    return split(ary, indices_or_sections, 0)


@array_function_dispatch(_hvdsplit_dispatcher)
def dsplit(ary, indices_or_sections):
    """
    Split array into multiple sub-arrays along the 3rd axis (depth).

    Please refer to the `split` documentation.  `dsplit` is equivalent
    to `split` with ``axis=2``, the array is always split along the third
    axis provided the array dimension is greater than or equal to 3.

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(16.0).reshape(2, 2, 4)
    >>> x
    array([[[ 0.,   1.,   2.,   3.],
            [ 4.,   5.,   6.,   7.]],
           [[ 8.,   9.,  10.,  11.],
            [12.,  13.,  14.,  15.]]])
    >>> np.dsplit(x, 2)
    [array([[[ 0.,  1.],
            [ 4.,  5.]],
           [[ 8.,  9.],
            [12., 13.]]]), array([[[ 2.,  3.],
            [ 6.,  7.]],
           [[10., 11.],
            [14., 15.]]])]
    >>> np.dsplit(x, np.array([3, 6]))
    [array([[[ 0.,   1.,   2.],
            [ 4.,   5.,   6.]],
           [[ 8.,   9.,  10.],
            [12.,  13.,  14.]]]),
     array([[[ 3.],
            [ 7.]],
           [[11.],
            [15.]]]),
    array([], shape=(2, 2, 0), dtype=float64)]
    """
    if _nx.ndim(ary) < 3:
        raise ValueError('dsplit only works on arrays of 3 or more dimensions')
    return split(ary, indices_or_sections, 2)


def get_array_wrap(*args):
    """Find the wrapper for the array with the highest priority.

    In case of ties, leftmost wins. If no wrapper is found, return None.

    .. deprecated:: 2.0
    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`get_array_wrap` is deprecated. "
        "(deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    wrappers = sorted((getattr(x, '__array_priority__', 0), -i,
                 x.__array_wrap__) for i, x in enumerate(args)
                                   if hasattr(x, '__array_wrap__'))
    if wrappers:
        return wrappers[-1][-1]
    return None


def _kron_dispatcher(a, b):
    return (a, b)


@array_function_dispatch(_kron_dispatcher)
def kron(a, b):
    """
    Kronecker product of two arrays.

    Computes the Kronecker product, a composite array made of blocks of the
    second array scaled by the first.

    Parameters
    ----------
    a, b : array_like

    Returns
    -------
    out : ndarray

    See Also
    --------
    outer : The outer product

    Notes
    -----
    The function assumes that the number of dimensions of `a` and `b`
    are the same, if necessary prepending the smallest with ones.
    If ``a.shape = (r0,r1,..,rN)`` and ``b.shape = (s0,s1,...,sN)``,
    the Kronecker product has shape ``(r0*s0, r1*s1, ..., rN*SN)``.
    The elements are products of elements from `a` and `b`, organized
    explicitly by::

        kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]

    where::

        kt = it * st + jt,  t = 0,...,N

    In the common 2-D case (N=1), the block structure can be visualized::

        [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],
         [  ...                              ...   ],
         [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]


    Examples
    --------
    >>> import numpy as np
    >>> np.kron([1,10,100], [5,6,7])
    array([  5,   6,   7, ..., 500, 600, 700])
    >>> np.kron([5,6,7], [1,10,100])
    array([  5,  50, 500, ...,   7,  70, 700])

    >>> np.kron(np.eye(2), np.ones((2,2)))
    array([[1.,  1.,  0.,  0.],
           [1.,  1.,  0.,  0.],
           [0.,  0.,  1.,  1.],
           [0.,  0.,  1.,  1.]])

    >>> a = np.arange(100).reshape((2,5,2,5))
    >>> b = np.arange(24).reshape((2,3,4))
    >>> c = np.kron(a,b)
    >>> c.shape
    (2, 10, 6, 20)
    >>> I = (1,3,0,2)
    >>> J = (0,2,1)
    >>> J1 = (0,) + J             # extend to ndim=4
    >>> S1 = (1,) + b.shape
    >>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))
    >>> c[K] == a[I]*b[J]
    True

    """
    # Working:
    # 1. Equalise the shapes by prepending smaller array with 1s
    # 2. Expand shapes of both the arrays by adding new axes at
    #    odd positions for 1st array and even positions for 2nd
    # 3. Compute the product of the modified array
    # 4. The inner most array elements now contain the rows of
    #    the Kronecker product
    # 5. Reshape the result to kron's shape, which is same as
    #    product of shapes of the two arrays.
    b = asanyarray(b)
    a = array(a, copy=None, subok=True, ndmin=b.ndim)
    is_any_mat = isinstance(a, matrix) or isinstance(b, matrix)
    ndb, nda = b.ndim, a.ndim
    nd = max(ndb, nda)

    if (nda == 0 or ndb == 0):
        return _nx.multiply(a, b)

    as_ = a.shape
    bs = b.shape
    if not a.flags.contiguous:
        a = reshape(a, as_)
    if not b.flags.contiguous:
        b = reshape(b, bs)

    # Equalise the shapes by prepending smaller one with 1s
    as_ = (1,) * max(0, ndb - nda) + as_
    bs = (1,) * max(0, nda - ndb) + bs

    # Insert empty dimensions
    a_arr = expand_dims(a, axis=tuple(range(ndb - nda)))
    b_arr = expand_dims(b, axis=tuple(range(nda - ndb)))

    # Compute the product
    a_arr = expand_dims(a_arr, axis=tuple(range(1, nd * 2, 2)))
    b_arr = expand_dims(b_arr, axis=tuple(range(0, nd * 2, 2)))
    # In case of `mat`, convert result to `array`
    result = _nx.multiply(a_arr, b_arr, subok=(not is_any_mat))

    # Reshape back
    result = result.reshape(_nx.multiply(as_, bs))

    return result if not is_any_mat else matrix(result, copy=False)


def _tile_dispatcher(A, reps):
    return (A, reps)


@array_function_dispatch(_tile_dispatcher)
def tile(A, reps):
    """
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by prepending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use numpy's broadcasting operations and functions.

    Parameters
    ----------
    A : array_like
        The input array.
    reps : array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.

    See Also
    --------
    repeat : Repeat elements of an array.
    broadcast_to : Broadcast an array to a new shape

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])
    >>> np.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])

    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])
    >>> np.tile(b, (2, 1))
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> c = np.array([1,2,3,4])
    >>> np.tile(c,(4,1))
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
    """
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    d = len(tup)
    if all(x == 1 for x in tup) and isinstance(A, _nx.ndarray):
        # Fixes the problem that the function does not make a copy if A is a
        # numpy array and the repetitions are 1 in all dimensions
        return _nx.array(A, copy=True, subok=True, ndmin=d)
    else:
        # Note that no copy of zero-sized arrays is made. However since they
        # have no data there is no risk of an inadvertent overwrite.
        c = _nx.array(A, copy=None, subok=True, ndmin=d)
    if (d < c.ndim):
        tup = (1,) * (c.ndim - d) + tup
    shape_out = tuple(s * t for s, t in zip(c.shape, tup))
    n = c.size
    if n > 0:
        for dim_in, nrep in zip(c.shape, tup):
            if nrep != 1:
                c = c.reshape(-1, n).repeat(nrep, 0)
            n //= dim_in
    return c.reshape(shape_out)

# === atelier-kyo-manager/app\utils\csv_handler.py ===
# app/utils/csv_handler.py
import csv
from io import TextIOWrapper
from app import db # dbオブジェクトをインポート
from app.models import Product # Productモデルをインポート

def import_products_from_csv(file):
    # ファイルをUTF-8で開く
    csv_file = TextIOWrapper(file, encoding='utf-8')
    reader = csv.DictReader(csv_file)

    # 既存のデータをクリアするか、更新するかは要検討
    # ここでは単純に追加する例

    for row in reader:
        try:
            # CSVの列名とProductモデルの属性名を合わせる
            # CSVファイルが以下の列を持つと仮定: '商品名', 'ブランド', '仕入価格', '販売価格', '仕入先URL', '画像URL', '在庫状況'
            product = Product(
                name=row.get('商品名', ''),
                brand=row.get('ブランド', ''),
                purchase_price=float(row.get('仕入価格', 0.0)),
                selling_price=float(row.get('販売価格', 0.0)),
                supplier_url=row.get('仕入先URL', ''),
                image_url=row.get('画像URL', ''),
                stock_status=(row.get('在庫状況', 'True').lower() == 'true') # 'True'/'False'文字列を真偽値に変換
            )
            # 利益率を計算して設定
            product.profit_margin = product.calculate_profit()
            db.session.add(product)
        except ValueError as e:
            print(f"CSVインポートエラー: 数値変換失敗 - {row}, エラー: {e}")
            # エラー処理。スキップするか、ログに記録するかなど
            continue
        except KeyError as e:
            print(f"CSVインポートエラー: 必須列が見つかりません - {e}, 行データ: {row}")
            continue

    db.session.commit()

# === atelier-kyo-manager/app\utils\utils\csv_handler.py ===
# app/utils/csv_handler.py
import csv
from io import TextIOWrapper
from app import db # dbオブジェクトをインポート
from app.models import Product # Productモデルをインポート

def import_products_from_csv(file):
    # ファイルをUTF-8で開く
    csv_file = TextIOWrapper(file, encoding='utf-8')
    reader = csv.DictReader(csv_file)

    # 既存のデータをクリアするか、更新するかは要検討
    # ここでは単純に追加する例

    for row in reader:
        try:
            # CSVの列名とProductモデルの属性名を合わせる
            # CSVファイルが以下の列を持つと仮定: '商品名', 'ブランド', '仕入価格', '販売価格', '仕入先URL', '画像URL', '在庫状況'
            product = Product(
                name=row.get('商品名', ''),
                brand=row.get('ブランド', ''),
                purchase_price=float(row.get('仕入価格', 0.0)),
                selling_price=float(row.get('販売価格', 0.0)),
                supplier_url=row.get('仕入先URL', ''),
                image_url=row.get('画像URL', ''),
                stock_status=(row.get('在庫状況', 'True').lower() == 'true') # 'True'/'False'文字列を真偽値に変換
            )
            # 利益率を計算して設定
            product.profit_margin = product.calculate_profit()
            db.session.add(product)
        except ValueError as e:
            print(f"CSVインポートエラー: 数値変換失敗 - {row}, エラー: {e}")
            # エラー処理。スキップするか、ログに記録するかなど
            continue
        except KeyError as e:
            print(f"CSVインポートエラー: 必須列が見つかりません - {e}, 行データ: {row}")
            continue

    db.session.commit()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_arraysetops_impl.py ===
"""
Set operations for arrays based on sorting.

Notes
-----

For floating point arrays, inaccurate results may appear due to usual round-off
and floating point comparison issues.

Speed could be gained in some operations by an implementation of
`numpy.sort`, that can provide directly the permutation vectors, thus avoiding
calls to `numpy.argsort`.

Original author: Robert Cimrman

"""
import functools
import warnings
from typing import NamedTuple

import numpy as np
from numpy._core import overrides
from numpy._core._multiarray_umath import _array_converter, _unique_hash

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    "ediff1d", "in1d", "intersect1d", "isin", "setdiff1d", "setxor1d",
    "union1d", "unique", "unique_all", "unique_counts", "unique_inverse",
    "unique_values"
]


def _ediff1d_dispatcher(ary, to_end=None, to_begin=None):
    return (ary, to_end, to_begin)


@array_function_dispatch(_ediff1d_dispatcher)
def ediff1d(ary, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    Parameters
    ----------
    ary : array_like
        If necessary, will be flattened before the differences are taken.
    to_end : array_like, optional
        Number(s) to append at the end of the returned differences.
    to_begin : array_like, optional
        Number(s) to prepend at the beginning of the returned differences.

    Returns
    -------
    ediff1d : ndarray
        The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.

    See Also
    --------
    diff, gradient

    Notes
    -----
    When applied to masked arrays, this function drops the mask information
    if the `to_begin` and/or `to_end` parameters are used.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.ediff1d(x)
    array([ 1,  2,  3, -7])

    >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
    array([-99,   1,   2, ...,  -7,  88,  99])

    The returned array is always 1D.

    >>> y = [[1, 2, 4], [1, 6, 24]]
    >>> np.ediff1d(y)
    array([ 1,  2, -3,  5, 18])

    """
    conv = _array_converter(ary)
    # Convert to (any) array and ravel:
    ary = conv[0].ravel()

    # enforce that the dtype of `ary` is used for the output
    dtype_req = ary.dtype

    # fast track default case
    if to_begin is None and to_end is None:
        return ary[1:] - ary[:-1]

    if to_begin is None:
        l_begin = 0
    else:
        to_begin = np.asanyarray(to_begin)
        if not np.can_cast(to_begin, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_begin` must be compatible "
                            "with input `ary` under the `same_kind` rule.")

        to_begin = to_begin.ravel()
        l_begin = len(to_begin)

    if to_end is None:
        l_end = 0
    else:
        to_end = np.asanyarray(to_end)
        if not np.can_cast(to_end, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_end` must be compatible "
                            "with input `ary` under the `same_kind` rule.")

        to_end = to_end.ravel()
        l_end = len(to_end)

    # do the calculation in place and copy to_begin and to_end
    l_diff = max(len(ary) - 1, 0)
    result = np.empty_like(ary, shape=l_diff + l_begin + l_end)

    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff:] = to_end
    np.subtract(ary[1:], ary[:-1], result[l_begin:l_begin + l_diff])

    return conv.wrap(result)


def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x


def _unique_dispatcher(ar, return_index=None, return_inverse=None,
                       return_counts=None, axis=None, *, equal_nan=None,
                       sorted=True):
    return (ar,)


@array_function_dispatch(_unique_dispatcher)
def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None, *, equal_nan=True,
           sorted=True):
    """
    Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details.  Object arrays or structured arrays
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.

    equal_nan : bool, optional
        If True, collapses multiple NaN values in the return array into one.

        .. versionadded:: 1.24

    sorted : bool, optional
        If True, the unique elements are sorted. Elements may be sorted in
        practice even if ``sorted=False``, but this could change without
        notice.

        .. versionadded:: 2.3

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    See Also
    --------
    repeat : Repeat elements of an array.
    sort : Return a sorted copy of an array.

    Notes
    -----
    When an axis is specified the subarrays indexed by the axis are sorted.
    This is done by making the specified axis the first dimension of the array
    (move the axis to the first dimension to keep the order of the other axes)
    and then flattening the subarrays in C order. The flattened subarrays are
    then viewed as a structured type with each element given a label, with the
    effect that we end up with a 1-D array of structured types that can be
    treated in the same way as any other 1-D array. The result is that the
    flattened subarrays are sorted in lexicographic order starting with the
    first element.

    .. versionchanged:: 1.21
        Like np.sort, NaN will sort to the end of the values.
        For complex arrays all NaN values are considered equivalent
        (no matter whether the NaN is in the real or imaginary part).
        As the representant for the returned array the smallest one in the
        lexicographical order is chosen - see np.sort for how the lexicographical
        order is defined for complex arrays.

    .. versionchanged:: 2.0
        For multi-dimensional inputs, ``unique_inverse`` is reshaped
        such that the input can be reconstructed using
        ``np.take(unique, unique_inverse, axis=axis)``. The result is
        now not 1-dimensional when ``axis=None``.

        Note that in NumPy 2.0.0 a higher dimensional array was returned also
        when ``axis`` was not ``None``.  This was reverted, but
        ``inverse.reshape(-1)`` can be used to ensure compatibility with both
        versions.

    Examples
    --------
    >>> import numpy as np
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])

    Return the unique rows of a 2D array

    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0], [2, 3, 4]])

    Return the indices of the original array that give the unique values:

    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'], dtype='<U1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'], dtype='<U1')

    Reconstruct the input array from the unique values and inverse:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])

    Reconstruct the input values from the unique values and counts:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> values, counts = np.unique(a, return_counts=True)
    >>> values
    array([1, 2, 3, 4, 6])
    >>> counts
    array([1, 3, 1, 1, 1])
    >>> np.repeat(values, counts)
    array([1, 2, 2, 2, 3, 4, 6])    # original order not preserved

    """
    ar = np.asanyarray(ar)
    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts,
                        equal_nan=equal_nan, inverse_shape=ar.shape, axis=None,
                        sorted=sorted)
        return _unpack_tuple(ret)

    # axis was specified and not None
    try:
        ar = np.moveaxis(ar, axis, 0)
    except np.exceptions.AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise np.exceptions.AxisError(axis, ar.ndim) from None
    inverse_shape = [1] * ar.ndim
    inverse_shape[axis] = ar.shape[0]

    # Must reshape to a contiguous 2D array for this to work...
    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
    ar = np.ascontiguousarray(ar)
    dtype = [(f'f{i}', ar.dtype) for i in range(ar.shape[1])]

    # At this point, `ar` has shape `(n, m)`, and `dtype` is a structured
    # data type with `m` fields where each field has the data type of `ar`.
    # In the following, we create the array `consolidated`, which has
    # shape `(n,)` with data type `dtype`.
    try:
        if ar.shape[1] > 0:
            consolidated = ar.view(dtype)
        else:
            # If ar.shape[1] == 0, then dtype will be `np.dtype([])`, which is
            # a data type with itemsize 0, and the call `ar.view(dtype)` will
            # fail.  Instead, we'll use `np.empty` to explicitly create the
            # array with shape `(len(ar),)`.  Because `dtype` in this case has
            # itemsize 0, the total size of the result is still 0 bytes.
            consolidated = np.empty(len(ar), dtype=dtype)
    except TypeError as e:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype)) from e

    def reshape_uniq(uniq):
        n = len(uniq)
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(n, *orig_shape[1:])
        uniq = np.moveaxis(uniq, 0, axis)
        return uniq

    output = _unique1d(consolidated, return_index,
                       return_inverse, return_counts,
                       equal_nan=equal_nan, inverse_shape=inverse_shape,
                       axis=axis, sorted=sorted)
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)


def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False, *, equal_nan=True, inverse_shape=None,
              axis=None, sorted=True):
    """
    Find the unique elements of an array, ignoring shape.

    Uses a hash table to find the unique elements if possible.
    """
    ar = np.asanyarray(ar).flatten()
    if len(ar.shape) != 1:
        # np.matrix, and maybe some other array subclasses, insist on keeping
        # two dimensions for all operations. Coerce to an ndarray in such cases.
        ar = np.asarray(ar).flatten()

    optional_indices = return_index or return_inverse

    # masked arrays are not supported yet.
    if not optional_indices and not return_counts and not np.ma.is_masked(ar):
        # First we convert the array to a numpy array, later we wrap it back
        # in case it was a subclass of numpy.ndarray.
        conv = _array_converter(ar)
        ar_, = conv

        if (hash_unique := _unique_hash(ar_)) is not NotImplemented:
            if sorted:
                hash_unique.sort()
            # We wrap the result back in case it was a subclass of numpy.ndarray.
            return (conv.wrap(hash_unique),)

    # If we don't use the hash map, we use the slower sorting method.
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool)
    mask[:1] = True
    if (equal_nan and aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and
            np.isnan(aux[-1])):
        if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side='left')
        if aux_firstnan > 0:
            mask[1:aux_firstnan] = (
                aux[1:aux_firstnan] != aux[:aux_firstnan - 1])
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1:] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx.reshape(inverse_shape) if axis is None else inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret


# Array API set functions

class UniqueAllResult(NamedTuple):
    values: np.ndarray
    indices: np.ndarray
    inverse_indices: np.ndarray
    counts: np.ndarray


class UniqueCountsResult(NamedTuple):
    values: np.ndarray
    counts: np.ndarray


class UniqueInverseResult(NamedTuple):
    values: np.ndarray
    inverse_indices: np.ndarray


def _unique_all_dispatcher(x, /):
    return (x,)


@array_function_dispatch(_unique_all_dispatcher)
def unique_all(x):
    """
    Find the unique elements of an array, and counts, inverse, and indices.

    This function is an Array API compatible alternative to::

        np.unique(x, return_index=True, return_inverse=True,
                  return_counts=True, equal_nan=False, sorted=False)

    but returns a namedtuple for easier access to each output.

    .. note::
        This function currently always returns a sorted result, however,
        this could change in any NumPy minor release.

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * indices - The first occurring indices for each unique element.
        * inverse_indices - The indices from the set of unique elements
          that reconstruct `x`.
        * counts - The corresponding counts for each unique element.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> x = [1, 1, 2]
    >>> uniq = np.unique_all(x)
    >>> uniq.values
    array([1, 2])
    >>> uniq.indices
    array([0, 2])
    >>> uniq.inverse_indices
    array([0, 0, 1])
    >>> uniq.counts
    array([2, 1])
    """
    result = unique(
        x,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        equal_nan=False,
    )
    return UniqueAllResult(*result)


def _unique_counts_dispatcher(x, /):
    return (x,)


@array_function_dispatch(_unique_counts_dispatcher)
def unique_counts(x):
    """
    Find the unique elements and counts of an input array `x`.

    This function is an Array API compatible alternative to::

        np.unique(x, return_counts=True, equal_nan=False, sorted=False)

    but returns a namedtuple for easier access to each output.

    .. note::
        This function currently always returns a sorted result, however,
        this could change in any NumPy minor release.

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * counts - The corresponding counts for each unique element.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> x = [1, 1, 2]
    >>> uniq = np.unique_counts(x)
    >>> uniq.values
    array([1, 2])
    >>> uniq.counts
    array([2, 1])
    """
    result = unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=True,
        equal_nan=False,
    )
    return UniqueCountsResult(*result)


def _unique_inverse_dispatcher(x, /):
    return (x,)


@array_function_dispatch(_unique_inverse_dispatcher)
def unique_inverse(x):
    """
    Find the unique elements of `x` and indices to reconstruct `x`.

    This function is an Array API compatible alternative to::

        np.unique(x, return_inverse=True, equal_nan=False, sorted=False)

    but returns a namedtuple for easier access to each output.

    .. note::
        This function currently always returns a sorted result, however,
        this could change in any NumPy minor release.

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * inverse_indices - The indices from the set of unique elements
          that reconstruct `x`.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> x = [1, 1, 2]
    >>> uniq = np.unique_inverse(x)
    >>> uniq.values
    array([1, 2])
    >>> uniq.inverse_indices
    array([0, 0, 1])
    """
    result = unique(
        x,
        return_index=False,
        return_inverse=True,
        return_counts=False,
        equal_nan=False,
    )
    return UniqueInverseResult(*result)


def _unique_values_dispatcher(x, /):
    return (x,)


@array_function_dispatch(_unique_values_dispatcher)
def unique_values(x):
    """
    Returns the unique elements of an input array `x`.

    This function is an Array API compatible alternative to::

        np.unique(x, equal_nan=False, sorted=False)

    .. versionchanged:: 2.3
       The algorithm was changed to a faster one that does not rely on
       sorting, and hence the results are no longer implicitly sorted.

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : ndarray
        The unique elements of an input array.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> np.unique_values([1, 1, 2])
    array([1, 2])  # may vary

    """
    return unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=False,
        equal_nan=False,
        sorted=False,
    )


def _intersect1d_dispatcher(
        ar1, ar2, assume_unique=None, return_indices=None):
    return (ar1, ar2)


@array_function_dispatch(_intersect1d_dispatcher)
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """
    Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays. Will be flattened if not already 1D.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  If True but ``ar1`` or ``ar2`` are not
        unique, incorrect results and out-of-bounds indices could result.
        Default is False.
    return_indices : bool
        If True, the indices which correspond to the intersection of the two
        arrays are returned. The first instance of a value is used if there are
        multiple. Default is False.

    Returns
    -------
    intersect1d : ndarray
        Sorted 1D array of common and unique elements.
    comm1 : ndarray
        The indices of the first occurrences of the common values in `ar1`.
        Only provided if `return_indices` is True.
    comm2 : ndarray
        The indices of the first occurrences of the common values in `ar2`.
        Only provided if `return_indices` is True.

    Examples
    --------
    >>> import numpy as np
    >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])

    To intersect more than two arrays, use functools.reduce:

    >>> from functools import reduce
    >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([3])

    To return the indices of the values common to the input arrays
    along with the intersected values:

    >>> x = np.array([1, 1, 2, 3, 4])
    >>> y = np.array([2, 1, 4, 6])
    >>> xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
    >>> x_ind, y_ind
    (array([0, 2, 4]), array([1, 0, 2]))
    >>> xy, x[x_ind], y[y_ind]
    (array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))

    """
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if not assume_unique:
        if return_indices:
            ar1, ind1 = unique(ar1, return_index=True)
            ar2, ind2 = unique(ar2, return_index=True)
        else:
            ar1 = unique(ar1)
            ar2 = unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    aux = np.concatenate((ar1, ar2))
    if return_indices:
        aux_sort_indices = np.argsort(aux, kind='mergesort')
        aux = aux[aux_sort_indices]
    else:
        aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d


def _setxor1d_dispatcher(ar1, ar2, assume_unique=None):
    return (ar1, ar2)


@array_function_dispatch(_setxor1d_dispatcher)
def setxor1d(ar1, ar2, assume_unique=False):
    """
    Find the set exclusive-or of two arrays.

    Return the sorted, unique values that are in only one (not both) of the
    input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation. Default is False.

    Returns
    -------
    setxor1d : ndarray
        Sorted 1D array of unique values that are in only one of the input
        arrays.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 2, 4])
    >>> b = np.array([2, 3, 5, 7, 5])
    >>> np.setxor1d(a,b)
    array([1, 4, 5, 7])

    """
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)

    aux = np.concatenate((ar1, ar2), axis=None)
    if aux.size == 0:
        return aux

    aux.sort()
    flag = np.concatenate(([True], aux[1:] != aux[:-1], [True]))
    return aux[flag[1:] & flag[:-1]]


def _in1d_dispatcher(ar1, ar2, assume_unique=None, invert=None, *,
                     kind=None):
    return (ar1, ar2)


@array_function_dispatch(_in1d_dispatcher)
def in1d(ar1, ar2, assume_unique=False, invert=False, *, kind=None):
    """
    Test whether each element of a 1-D array is also present in a second array.

    .. deprecated:: 2.0
        Use :func:`isin` instead of `in1d` for new code.

    Returns a boolean array the same length as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    Parameters
    ----------
    ar1 : (M,) array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of `ar1`.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of `ar1` is in `ar2` and True otherwise).
        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``np.invert(in1d(a, b))``.
    kind : {None, 'sort', 'table'}, optional
        The algorithm to use. This will not affect the final result,
        but will affect the speed and memory use. The default, None,
        will select automatically based on memory considerations.

        * If 'sort', will use a mergesort-based approach. This will have
          a memory usage of roughly 6 times the sum of the sizes of
          `ar1` and `ar2`, not accounting for size of dtypes.
        * If 'table', will use a lookup table approach similar
          to a counting sort. This is only available for boolean and
          integer arrays. This will have a memory usage of the
          size of `ar1` plus the max-min value of `ar2`. `assume_unique`
          has no effect when the 'table' option is used.
        * If None, will automatically choose 'table' if
          the required memory allocation is less than or equal to
          6 times the sum of the sizes of `ar1` and `ar2`,
          otherwise will use 'sort'. This is done to not use
          a large amount of memory by default, even though
          'table' may be faster in most cases. If 'table' is chosen,
          `assume_unique` will have no effect.

    Returns
    -------
    in1d : (M,) ndarray, bool
        The values `ar1[in1d]` are in `ar2`.

    See Also
    --------
    isin                  : Version of this function that preserves the
                            shape of ar1.

    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
    equivalent to ``np.array([item in b for item in a])``.
    However, this idea fails if `ar2` is a set, or similar (non-sequence)
    container:  As ``ar2`` is converted to an array, in those cases
    ``asarray(ar2)`` is an object array rather than the expected array of
    contained values.

    Using ``kind='table'`` tends to be faster than `kind='sort'` if the
    following relationship is true:
    ``log10(len(ar2)) > (log10(max(ar2)-min(ar2)) - 2.27) / 0.927``,
    but may use greater memory. The default value for `kind` will
    be automatically selected based only on memory usage, so one may
    manually set ``kind='table'`` if memory constraints can be relaxed.

    Examples
    --------
    >>> import numpy as np
    >>> test = np.array([0, 1, 2, 5, 0])
    >>> states = [0, 2]
    >>> mask = np.in1d(test, states)
    >>> mask
    array([ True, False,  True, False,  True])
    >>> test[mask]
    array([0, 2, 0])
    >>> mask = np.in1d(test, states, invert=True)
    >>> mask
    array([False,  True, False,  True, False])
    >>> test[mask]
    array([1, 5])
    """

    # Deprecated in NumPy 2.0, 2023-08-18
    warnings.warn(
        "`in1d` is deprecated. Use `np.isin` instead.",
        DeprecationWarning,
        stacklevel=2
    )

    return _in1d(ar1, ar2, assume_unique, invert, kind=kind)


def _in1d(ar1, ar2, assume_unique=False, invert=False, *, kind=None):
    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # Ensure that iteration through object arrays yields size-1 arrays
    if ar2.dtype == object:
        ar2 = ar2.reshape(-1, 1)

    if kind not in {None, 'sort', 'table'}:
        raise ValueError(
            f"Invalid kind: '{kind}'. Please use None, 'sort' or 'table'.")

    # Can use the table method if all arrays are integers or boolean:
    is_int_arrays = all(ar.dtype.kind in ("u", "i", "b") for ar in (ar1, ar2))
    use_table_method = is_int_arrays and kind in {None, 'table'}

    if use_table_method:
        if ar2.size == 0:
            if invert:
                return np.ones_like(ar1, dtype=bool)
            else:
                return np.zeros_like(ar1, dtype=bool)

        # Convert booleans to uint8 so we can use the fast integer algorithm
        if ar1.dtype == bool:
            ar1 = ar1.astype(np.uint8)
        if ar2.dtype == bool:
            ar2 = ar2.astype(np.uint8)

        ar2_min = int(np.min(ar2))
        ar2_max = int(np.max(ar2))

        ar2_range = ar2_max - ar2_min

        # Constraints on whether we can actually use the table method:
        #  1. Assert memory usage is not too large
        below_memory_constraint = ar2_range <= 6 * (ar1.size + ar2.size)
        #  2. Check overflows for (ar2 - ar2_min); dtype=ar2.dtype
        range_safe_from_overflow = ar2_range <= np.iinfo(ar2.dtype).max

        # Optimal performance is for approximately
        # log10(size) > (log10(range) - 2.27) / 0.927.
        # However, here we set the requirement that by default
        # the intermediate array can only be 6x
        # the combined memory allocation of the original
        # arrays. See discussion on
        # https://github.com/numpy/numpy/pull/12065.

        if (
            range_safe_from_overflow and
            (below_memory_constraint or kind == 'table')
        ):

            if invert:
                outgoing_array = np.ones_like(ar1, dtype=bool)
            else:
                outgoing_array = np.zeros_like(ar1, dtype=bool)

            # Make elements 1 where the integer exists in ar2
            if invert:
                isin_helper_ar = np.ones(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 0
            else:
                isin_helper_ar = np.zeros(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 1

            # Mask out elements we know won't work
            basic_mask = (ar1 <= ar2_max) & (ar1 >= ar2_min)
            in_range_ar1 = ar1[basic_mask]
            if in_range_ar1.size == 0:
                # Nothing more to do, since all values are out of range.
                return outgoing_array

            # Unfortunately, ar2_min can be out of range for `intp` even
            # if the calculation result must fit in range (and be positive).
            # In that case, use ar2.dtype which must work for all unmasked
            # values.
            try:
                ar2_min = np.array(ar2_min, dtype=np.intp)
                dtype = np.intp
            except OverflowError:
                dtype = ar2.dtype

            out = np.empty_like(in_range_ar1, dtype=np.intp)
            outgoing_array[basic_mask] = isin_helper_ar[
                    np.subtract(in_range_ar1, ar2_min, dtype=dtype,
                                out=out, casting="unsafe")]

            return outgoing_array
        elif kind == 'table':  # not range_safe_from_overflow
            raise RuntimeError(
                "You have specified kind='table', "
                "but the range of values in `ar2` or `ar1` exceed the "
                "maximum integer of the datatype. "
                "Please set `kind` to None or 'sort'."
            )
    elif kind == 'table':
        raise ValueError(
            "The 'table' method is only "
            "supported for boolean or integer arrays. "
            "Please select 'sort' or None for kind."
        )

    # Check if one of the arrays may contain arbitrary objects
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # This code is run when
    # a) the first condition is true, making the code significantly faster
    # b) the second condition is true (i.e. `ar1` or `ar2` may contain
    #    arbitrary objects), since then sorting is not guaranteed to work
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)

    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]


def _isin_dispatcher(element, test_elements, assume_unique=None, invert=None,
                     *, kind=None):
    return (element, test_elements)


@array_function_dispatch(_isin_dispatcher)
def isin(element, test_elements, assume_unique=False, invert=False, *,
         kind=None):
    """
    Calculates ``element in test_elements``, broadcasting over `element` only.
    Returns a boolean array of the same shape as `element` that is True
    where an element of `element` is in `test_elements` and False otherwise.

    Parameters
    ----------
    element : array_like
        Input array.
    test_elements : array_like
        The values against which to test each value of `element`.
        This argument is flattened if it is an array or array_like.
        See notes for behavior with non-array-like parameters.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted, as if
        calculating `element not in test_elements`. Default is False.
        ``np.isin(a, b, invert=True)`` is equivalent to (but faster
        than) ``np.invert(np.isin(a, b))``.
    kind : {None, 'sort', 'table'}, optional
        The algorithm to use. This will not affect the final result,
        but will affect the speed and memory use. The default, None,
        will select automatically based on memory considerations.

        * If 'sort', will use a mergesort-based approach. This will have
          a memory usage of roughly 6 times the sum of the sizes of
          `element` and `test_elements`, not accounting for size of dtypes.
        * If 'table', will use a lookup table approach similar
          to a counting sort. This is only available for boolean and
          integer arrays. This will have a memory usage of the
          size of `element` plus the max-min value of `test_elements`.
          `assume_unique` has no effect when the 'table' option is used.
        * If None, will automatically choose 'table' if
          the required memory allocation is less than or equal to
          6 times the sum of the sizes of `element` and `test_elements`,
          otherwise will use 'sort'. This is done to not use
          a large amount of memory by default, even though
          'table' may be faster in most cases. If 'table' is chosen,
          `assume_unique` will have no effect.


    Returns
    -------
    isin : ndarray, bool
        Has the same shape as `element`. The values `element[isin]`
        are in `test_elements`.

    Notes
    -----
    `isin` is an element-wise function version of the python keyword `in`.
    ``isin(a, b)`` is roughly equivalent to
    ``np.array([item in b for item in a])`` if `a` and `b` are 1-D sequences.

    `element` and `test_elements` are converted to arrays if they are not
    already. If `test_elements` is a set (or other non-sequence collection)
    it will be converted to an object array with one element, rather than an
    array of the values contained in `test_elements`. This is a consequence
    of the `array` constructor's way of handling non-sequence collections.
    Converting the set to a list usually gives the desired behavior.

    Using ``kind='table'`` tends to be faster than `kind='sort'` if the
    following relationship is true:
    ``log10(len(test_elements)) >
    (log10(max(test_elements)-min(test_elements)) - 2.27) / 0.927``,
    but may use greater memory. The default value for `kind` will
    be automatically selected based only on memory usage, so one may
    manually set ``kind='table'`` if memory constraints can be relaxed.

    Examples
    --------
    >>> import numpy as np
    >>> element = 2*np.arange(4).reshape((2, 2))
    >>> element
    array([[0, 2],
           [4, 6]])
    >>> test_elements = [1, 2, 4, 8]
    >>> mask = np.isin(element, test_elements)
    >>> mask
    array([[False,  True],
           [ True, False]])
    >>> element[mask]
    array([2, 4])

    The indices of the matched values can be obtained with `nonzero`:

    >>> np.nonzero(mask)
    (array([0, 1]), array([1, 0]))

    The test can also be inverted:

    >>> mask = np.isin(element, test_elements, invert=True)
    >>> mask
    array([[ True, False],
           [False,  True]])
    >>> element[mask]
    array([0, 6])

    Because of how `array` handles sets, the following does not
    work as expected:

    >>> test_set = {1, 2, 4, 8}
    >>> np.isin(element, test_set)
    array([[False, False],
           [False, False]])

    Casting the set to a list gives the expected result:

    >>> np.isin(element, list(test_set))
    array([[False,  True],
           [ True, False]])
    """
    element = np.asarray(element)
    return _in1d(element, test_elements, assume_unique=assume_unique,
                 invert=invert, kind=kind).reshape(element.shape)


def _union1d_dispatcher(ar1, ar2):
    return (ar1, ar2)


@array_function_dispatch(_union1d_dispatcher)
def union1d(ar1, ar2):
    """
    Find the union of two arrays.

    Return the unique, sorted array of values that are in either of the two
    input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays. They are flattened if they are not already 1D.

    Returns
    -------
    union1d : ndarray
        Unique, sorted union of the input arrays.

    Examples
    --------
    >>> import numpy as np
    >>> np.union1d([-1, 0, 1], [-2, 0, 2])
    array([-2, -1,  0,  1,  2])

    To find the union of more than two arrays, use functools.reduce:

    >>> from functools import reduce
    >>> reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([1, 2, 3, 4, 6])
    """
    return unique(np.concatenate((ar1, ar2), axis=None))


def _setdiff1d_dispatcher(ar1, ar2, assume_unique=None):
    return (ar1, ar2)


@array_function_dispatch(_setdiff1d_dispatcher)
def setdiff1d(ar1, ar2, assume_unique=False):
    """
    Find the set difference of two arrays.

    Return the unique values in `ar1` that are not in `ar2`.

    Parameters
    ----------
    ar1 : array_like
        Input array.
    ar2 : array_like
        Input comparison array.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    setdiff1d : ndarray
        1D array of values in `ar1` that are not in `ar2`. The result
        is sorted when `assume_unique=False`, but otherwise only sorted
        if the input is sorted.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 2, 4, 1])
    >>> b = np.array([3, 4, 5, 6])
    >>> np.setdiff1d(a, b)
    array([1, 2])

    """
    if assume_unique:
        ar1 = np.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[_in1d(ar1, ar2, assume_unique=True, invert=True)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_twodim_base_impl.py ===
""" Basic functions for manipulating 2d arrays

"""
import functools
import operator

from numpy._core import iinfo, overrides
from numpy._core._multiarray_umath import _array_converter
from numpy._core.numeric import (
    arange,
    asanyarray,
    asarray,
    diagonal,
    empty,
    greater_equal,
    indices,
    int8,
    int16,
    int32,
    int64,
    intp,
    multiply,
    nonzero,
    ones,
    promote_types,
    where,
    zeros,
)
from numpy._core.overrides import finalize_array_function_like, set_module
from numpy.lib._stride_tricks_impl import broadcast_to

__all__ = [
    'diag', 'diagflat', 'eye', 'fliplr', 'flipud', 'tri', 'triu',
    'tril', 'vander', 'histogram2d', 'mask_indices', 'tril_indices',
    'tril_indices_from', 'triu_indices', 'triu_indices_from', ]


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


i1 = iinfo(int8)
i2 = iinfo(int16)
i4 = iinfo(int32)


def _min_int(low, high):
    """ get small int that fits the range """
    if high <= i1.max and low >= i1.min:
        return int8
    if high <= i2.max and low >= i2.min:
        return int16
    if high <= i4.max and low >= i4.min:
        return int32
    return int64


def _flip_dispatcher(m):
    return (m,)


@array_function_dispatch(_flip_dispatcher)
def fliplr(m):
    """
    Reverse the order of elements along axis 1 (left/right).

    For a 2-D array, this flips the entries in each row in the left/right
    direction. Columns are preserved, but appear in a different order than
    before.

    Parameters
    ----------
    m : array_like
        Input array, must be at least 2-D.

    Returns
    -------
    f : ndarray
        A view of `m` with the columns reversed.  Since a view
        is returned, this operation is :math:`\\mathcal O(1)`.

    See Also
    --------
    flipud : Flip array in the up/down direction.
    flip : Flip array in one or more dimensions.
    rot90 : Rotate array counterclockwise.

    Notes
    -----
    Equivalent to ``m[:,::-1]`` or ``np.flip(m, axis=1)``.
    Requires the array to be at least 2-D.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.diag([1.,2.,3.])
    >>> A
    array([[1.,  0.,  0.],
           [0.,  2.,  0.],
           [0.,  0.,  3.]])
    >>> np.fliplr(A)
    array([[0.,  0.,  1.],
           [0.,  2.,  0.],
           [3.,  0.,  0.]])

    >>> rng = np.random.default_rng()
    >>> A = rng.normal(size=(2,3,5))
    >>> np.all(np.fliplr(A) == A[:,::-1,...])
    True

    """
    m = asanyarray(m)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return m[:, ::-1]


@array_function_dispatch(_flip_dispatcher)
def flipud(m):
    """
    Reverse the order of elements along axis 0 (up/down).

    For a 2-D array, this flips the entries in each column in the up/down
    direction. Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array.

    Returns
    -------
    out : array_like
        A view of `m` with the rows reversed.  Since a view is
        returned, this operation is :math:`\\mathcal O(1)`.

    See Also
    --------
    fliplr : Flip array in the left/right direction.
    flip : Flip array in one or more dimensions.
    rot90 : Rotate array counterclockwise.

    Notes
    -----
    Equivalent to ``m[::-1, ...]`` or ``np.flip(m, axis=0)``.
    Requires the array to be at least 1-D.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.diag([1.0, 2, 3])
    >>> A
    array([[1.,  0.,  0.],
           [0.,  2.,  0.],
           [0.,  0.,  3.]])
    >>> np.flipud(A)
    array([[0.,  0.,  3.],
           [0.,  2.,  0.],
           [1.,  0.,  0.]])

    >>> rng = np.random.default_rng()
    >>> A = rng.normal(size=(2,3,5))
    >>> np.all(np.flipud(A) == A[::-1,...])
    True

    >>> np.flipud([1,2])
    array([2, 1])

    """
    m = asanyarray(m)
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return m[::-1, ...]


@finalize_array_function_like
@set_module('numpy')
def eye(N, M=None, k=0, dtype=float, order='C', *, device=None, like=None):
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned array.
    order : {'C', 'F'}, optional
        Whether the output should be stored in row-major (C-style) or
        column-major (Fortran-style) order in memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    I : ndarray of shape (N,M)
      An array where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.

    See Also
    --------
    identity : (almost) equivalent function
    diag : diagonal 2-D array from a 1-D array specified by the user.

    Examples
    --------
    >>> import numpy as np
    >>> np.eye(2, dtype=int)
    array([[1, 0],
           [0, 1]])
    >>> np.eye(3, k=1)
    array([[0.,  1.,  0.],
           [0.,  0.,  1.],
           [0.,  0.,  0.]])

    """
    if like is not None:
        return _eye_with_like(
            like, N, M=M, k=k, dtype=dtype, order=order, device=device
        )
    if M is None:
        M = N
    m = zeros((N, M), dtype=dtype, order=order, device=device)
    if k >= M:
        return m
    # Ensure M and k are integers, so we don't get any surprise casting
    # results in the expressions `M-k` and `M+1` used below.  This avoids
    # a problem with inputs with type (for example) np.uint64.
    M = operator.index(M)
    k = operator.index(k)
    if k >= 0:
        i = k
    else:
        i = (-k) * M
    m[:M - k].flat[i::M + 1] = 1
    return m


_eye_with_like = array_function_dispatch()(eye)


def _diag_dispatcher(v, k=None):
    return (v,)


@array_function_dispatch(_diag_dispatcher)
def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    See the more detailed documentation for ``numpy.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting array;
    whether it returns a copy or a view depends on what version of numpy you
    are using.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    diagonal : Return specified diagonals.
    diagflat : Create a 2-D array with the flattened input as a diagonal.
    trace : Sum along diagonals.
    triu : Upper triangle of an array.
    tril : Lower triangle of an array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> np.diag(x)
    array([0, 4, 8])
    >>> np.diag(x, k=1)
    array([1, 5])
    >>> np.diag(x, k=-1)
    array([3, 7])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])

    """
    v = asanyarray(v)
    s = v.shape
    if len(s) == 1:
        n = s[0] + abs(k)
        res = zeros((n, n), v.dtype)
        if k >= 0:
            i = k
        else:
            i = (-k) * n
        res[:n - k].flat[i::n + 1] = v
        return res
    elif len(s) == 2:
        return diagonal(v, k)
    else:
        raise ValueError("Input must be 1- or 2-d.")


@array_function_dispatch(_diag_dispatcher)
def diagflat(v, k=0):
    """
    Create a two-dimensional array with the flattened input as a diagonal.

    Parameters
    ----------
    v : array_like
        Input data, which is flattened and set as the `k`-th
        diagonal of the output.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal,
        a positive (negative) `k` giving the number of the diagonal above
        (below) the main.

    Returns
    -------
    out : ndarray
        The 2-D output array.

    See Also
    --------
    diag : MATLAB work-alike for 1-D and 2-D arrays.
    diagonal : Return specified diagonals.
    trace : Sum along diagonals.

    Examples
    --------
    >>> import numpy as np
    >>> np.diagflat([[1,2], [3,4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> np.diagflat([1,2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])

    """
    conv = _array_converter(v)
    v, = conv.as_arrays(subok=False)
    v = v.ravel()
    s = len(v)
    n = s + abs(k)
    res = zeros((n, n), v.dtype)
    if (k >= 0):
        i = arange(0, n - k, dtype=intp)
        fi = i + k + i * n
    else:
        i = arange(0, n + k, dtype=intp)
        fi = i + (i - k) * n
    res.flat[fi] = v

    return conv.wrap(res)


@finalize_array_function_like
@set_module('numpy')
def tri(N, M=None, k=0, dtype=float, *, like=None):
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is float.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    tri : ndarray of shape (N, M)
        Array with its lower triangle filled with ones and zero elsewhere;
        in other words ``T[i,j] == 1`` for ``j <= i + k``, 0 otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> np.tri(3, 5, 2, dtype=int)
    array([[1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1]])

    >>> np.tri(3, 5, -1)
    array([[0.,  0.,  0.,  0.,  0.],
           [1.,  0.,  0.,  0.,  0.],
           [1.,  1.,  0.,  0.,  0.]])

    """
    if like is not None:
        return _tri_with_like(like, N, M=M, k=k, dtype=dtype)

    if M is None:
        M = N

    m = greater_equal.outer(arange(N, dtype=_min_int(0, N)),
                            arange(-k, M - k, dtype=_min_int(-k, M - k)))

    # Avoid making a copy if the requested type is already bool
    m = m.astype(dtype, copy=False)

    return m


_tri_with_like = array_function_dispatch()(tri)


def _trilu_dispatcher(m, k=None):
    return (m,)


@array_function_dispatch(_trilu_dispatcher)
def tril(m, k=0):
    """
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.
    For arrays with ``ndim`` exceeding 2, `tril` will apply to the final two
    axes.

    Parameters
    ----------
    m : array_like, shape (..., M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril : ndarray, shape (..., M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : same thing, only for the upper triangle

    Examples
    --------
    >>> import numpy as np
    >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])

    >>> np.tril(np.arange(3*4*5).reshape(3, 4, 5))
    array([[[ 0,  0,  0,  0,  0],
            [ 5,  6,  0,  0,  0],
            [10, 11, 12,  0,  0],
            [15, 16, 17, 18,  0]],
           [[20,  0,  0,  0,  0],
            [25, 26,  0,  0,  0],
            [30, 31, 32,  0,  0],
            [35, 36, 37, 38,  0]],
           [[40,  0,  0,  0,  0],
            [45, 46,  0,  0,  0],
            [50, 51, 52,  0,  0],
            [55, 56, 57, 58,  0]]])

    """
    m = asanyarray(m)
    mask = tri(*m.shape[-2:], k=k, dtype=bool)

    return where(mask, m, zeros(1, m.dtype))


@array_function_dispatch(_trilu_dispatcher)
def triu(m, k=0):
    """
    Upper triangle of an array.

    Return a copy of an array with the elements below the `k`-th diagonal
    zeroed. For arrays with ``ndim`` exceeding 2, `triu` will apply to the
    final two axes.

    Please refer to the documentation for `tril` for further details.

    See Also
    --------
    tril : lower triangle of an array

    Examples
    --------
    >>> import numpy as np
    >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])

    >>> np.triu(np.arange(3*4*5).reshape(3, 4, 5))
    array([[[ 0,  1,  2,  3,  4],
            [ 0,  6,  7,  8,  9],
            [ 0,  0, 12, 13, 14],
            [ 0,  0,  0, 18, 19]],
           [[20, 21, 22, 23, 24],
            [ 0, 26, 27, 28, 29],
            [ 0,  0, 32, 33, 34],
            [ 0,  0,  0, 38, 39]],
           [[40, 41, 42, 43, 44],
            [ 0, 46, 47, 48, 49],
            [ 0,  0, 52, 53, 54],
            [ 0,  0,  0, 58, 59]]])

    """
    m = asanyarray(m)
    mask = tri(*m.shape[-2:], k=k - 1, dtype=bool)

    return where(mask, zeros(1, m.dtype), m)


def _vander_dispatcher(x, N=None, increasing=None):
    return (x,)


# Originally borrowed from John Hunter and matplotlib
@array_function_dispatch(_vander_dispatcher)
def vander(x, N=None, increasing=False):
    """
    Generate a Vandermonde matrix.

    The columns of the output matrix are powers of the input vector. The
    order of the powers is determined by the `increasing` boolean argument.
    Specifically, when `increasing` is False, the `i`-th output column is
    the input vector raised element-wise to the power of ``N - i - 1``. Such
    a matrix with a geometric progression in each row is named for Alexandre-
    Theophile Vandermonde.

    Parameters
    ----------
    x : array_like
        1-D input array.
    N : int, optional
        Number of columns in the output.  If `N` is not specified, a square
        array is returned (``N = len(x)``).
    increasing : bool, optional
        Order of the powers of the columns.  If True, the powers increase
        from left to right, if False (the default) they are reversed.

    Returns
    -------
    out : ndarray
        Vandermonde matrix.  If `increasing` is False, the first column is
        ``x^(N-1)``, the second ``x^(N-2)`` and so forth. If `increasing` is
        True, the columns are ``x^0, x^1, ..., x^(N-1)``.

    See Also
    --------
    polynomial.polynomial.polyvander

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 5])
    >>> N = 3
    >>> np.vander(x, N)
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])

    >>> np.column_stack([x**(N-1-i) for i in range(N)])
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])

    >>> x = np.array([1, 2, 3, 5])
    >>> np.vander(x)
    array([[  1,   1,   1,   1],
           [  8,   4,   2,   1],
           [ 27,   9,   3,   1],
           [125,  25,   5,   1]])
    >>> np.vander(x, increasing=True)
    array([[  1,   1,   1,   1],
           [  1,   2,   4,   8],
           [  1,   3,   9,  27],
           [  1,   5,  25, 125]])

    The determinant of a square Vandermonde matrix is the product
    of the differences between the values of the input vector:

    >>> np.linalg.det(np.vander(x))
    48.000000000000043 # may vary
    >>> (5-3)*(5-2)*(5-1)*(3-2)*(3-1)*(2-1)
    48

    """
    x = asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array or sequence.")
    if N is None:
        N = len(x)

    v = empty((len(x), N), dtype=promote_types(x.dtype, int))
    tmp = v[:, ::-1] if not increasing else v

    if N > 0:
        tmp[:, 0] = 1
    if N > 1:
        tmp[:, 1:] = x[:, None]
        multiply.accumulate(tmp[:, 1:], out=tmp[:, 1:], axis=1)

    return v


def _histogram2d_dispatcher(x, y, bins=None, range=None, density=None,
                            weights=None):
    yield x
    yield y

    # This terrible logic is adapted from the checks in histogram2d
    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N == 2:
        yield from bins  # bins=[x, y]
    else:
        yield bins

    yield weights


@array_function_dispatch(_histogram2d_dispatcher)
def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    """
    Compute the bi-dimensional histogram of two data samples.

    Parameters
    ----------
    x : array_like, shape (N,)
        An array containing the x coordinates of the points to be
        histogrammed.
    y : array_like, shape (N,)
        An array containing the y coordinates of the points to be
        histogrammed.
    bins : int or array_like or [int, int] or [array, array], optional
        The bin specification:

        * If int, the number of bins for the two dimensions (nx=ny=bins).
        * If array_like, the bin edges for the two dimensions
          (x_edges=y_edges=bins).
        * If [int, int], the number of bins in each dimension
          (nx, ny = bins).
        * If [array, array], the bin edges in each dimension
          (x_edges, y_edges = bins).
        * A combination [int, array] or [array, int], where int
          is the number of bins and array is the bin edges.

    range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range
        will be considered outliers and not tallied in the histogram.
    density : bool, optional
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_area``.
    weights : array_like, shape(N,), optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
        Weights are normalized to 1 if `density` is True. If `density` is
        False, the values of the returned histogram are equal to the sum of
        the weights belonging to the samples falling into each bin.

    Returns
    -------
    H : ndarray, shape(nx, ny)
        The bi-dimensional histogram of samples `x` and `y`. Values in `x`
        are histogrammed along the first dimension and values in `y` are
        histogrammed along the second dimension.
    xedges : ndarray, shape(nx+1,)
        The bin edges along the first dimension.
    yedges : ndarray, shape(ny+1,)
        The bin edges along the second dimension.

    See Also
    --------
    histogram : 1D histogram
    histogramdd : Multidimensional histogram

    Notes
    -----
    When `density` is True, then the returned histogram is the sample
    density, defined such that the sum over bins of the product
    ``bin_value * bin_area`` is 1.

    Please note that the histogram does not follow the Cartesian convention
    where `x` values are on the abscissa and `y` values on the ordinate
    axis.  Rather, `x` is histogrammed along the first dimension of the
    array (vertical), and `y` along the second dimension of the array
    (horizontal).  This ensures compatibility with `histogramdd`.

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib.image import NonUniformImage
    >>> import matplotlib.pyplot as plt

    Construct a 2-D histogram with variable bin width. First define the bin
    edges:

    >>> xedges = [0, 1, 3, 5]
    >>> yedges = [0, 2, 3, 4, 6]

    Next we create a histogram H with random bin content:

    >>> x = np.random.normal(2, 1, 100)
    >>> y = np.random.normal(1, 1, 100)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    >>> # Histogram does not follow Cartesian convention (see Notes),
    >>> # therefore transpose H for visualization purposes.
    >>> H = H.T

    :func:`imshow <matplotlib.pyplot.imshow>` can only display square bins:

    >>> fig = plt.figure(figsize=(7, 3))
    >>> ax = fig.add_subplot(131, title='imshow: square bins')
    >>> plt.imshow(H, interpolation='nearest', origin='lower',
    ...         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    <matplotlib.image.AxesImage object at 0x...>

    :func:`pcolormesh <matplotlib.pyplot.pcolormesh>` can display actual edges:

    >>> ax = fig.add_subplot(132, title='pcolormesh: actual edges',
    ...         aspect='equal')
    >>> X, Y = np.meshgrid(xedges, yedges)
    >>> ax.pcolormesh(X, Y, H)
    <matplotlib.collections.QuadMesh object at 0x...>

    :class:`NonUniformImage <matplotlib.image.NonUniformImage>` can be used to
    display actual bin edges with interpolation:

    >>> ax = fig.add_subplot(133, title='NonUniformImage: interpolated',
    ...         aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    >>> im = NonUniformImage(ax, interpolation='bilinear')
    >>> xcenters = (xedges[:-1] + xedges[1:]) / 2
    >>> ycenters = (yedges[:-1] + yedges[1:]) / 2
    >>> im.set_data(xcenters, ycenters, H)
    >>> ax.add_image(im)
    >>> plt.show()

    It is also possible to construct a 2-D histogram without specifying bin
    edges:

    >>> # Generate non-symmetric test data
    >>> n = 10000
    >>> x = np.linspace(1, 100, n)
    >>> y = 2*np.log(x) + np.random.rand(n) - 0.5
    >>> # Compute 2d histogram. Note the order of x/y and xedges/yedges
    >>> H, yedges, xedges = np.histogram2d(y, x, bins=20)

    Now we can plot the histogram using
    :func:`pcolormesh <matplotlib.pyplot.pcolormesh>`, and a
    :func:`hexbin <matplotlib.pyplot.hexbin>` for comparison.

    >>> # Plot histogram using pcolormesh
    >>> fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    >>> ax1.pcolormesh(xedges, yedges, H, cmap='rainbow')
    >>> ax1.plot(x, 2*np.log(x), 'k-')
    >>> ax1.set_xlim(x.min(), x.max())
    >>> ax1.set_ylim(y.min(), y.max())
    >>> ax1.set_xlabel('x')
    >>> ax1.set_ylabel('y')
    >>> ax1.set_title('histogram2d')
    >>> ax1.grid()

    >>> # Create hexbin plot for comparison
    >>> ax2.hexbin(x, y, gridsize=20, cmap='rainbow')
    >>> ax2.plot(x, 2*np.log(x), 'k-')
    >>> ax2.set_title('hexbin')
    >>> ax2.set_xlim(x.min(), x.max())
    >>> ax2.set_xlabel('x')
    >>> ax2.grid()

    >>> plt.show()
    """
    from numpy import histogramdd

    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')

    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N not in {1, 2}:
        xedges = yedges = asarray(bins)
        bins = [xedges, yedges]
    hist, edges = histogramdd([x, y], bins, range, density, weights)
    return hist, edges[0], edges[1]


@set_module('numpy')
def mask_indices(n, mask_func, k=0):
    """
    Return the indices to access (n, n) arrays, given a masking function.

    Assume `mask_func` is a function that, for a square array a of size
    ``(n, n)`` with a possible offset argument `k`, when called as
    ``mask_func(a, k)`` returns a new array with zeros in certain locations
    (functions like `triu` or `tril` do precisely this). Then this function
    returns the indices where the non-zero values would be located.

    Parameters
    ----------
    n : int
        The returned indices will be valid to access arrays of shape (n, n).
    mask_func : callable
        A function whose call signature is similar to that of `triu`, `tril`.
        That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.
        `k` is an optional argument to the function.
    k : scalar
        An optional argument which is passed through to `mask_func`. Functions
        like `triu`, `tril` take a second argument that is interpreted as an
        offset.

    Returns
    -------
    indices : tuple of arrays.
        The `n` arrays of indices corresponding to the locations where
        ``mask_func(np.ones((n, n)), k)`` is True.

    See Also
    --------
    triu, tril, triu_indices, tril_indices

    Examples
    --------
    >>> import numpy as np

    These are the indices that would allow you to access the upper triangular
    part of any 3x3 array:

    >>> iu = np.mask_indices(3, np.triu)

    For example, if `a` is a 3x3 array:

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> a[iu]
    array([0, 1, 2, 4, 5, 8])

    An offset can be passed also to the masking function.  This gets us the
    indices starting on the first diagonal right of the main one:

    >>> iu1 = np.mask_indices(3, np.triu, 1)

    with which we now extract only three elements:

    >>> a[iu1]
    array([1, 2, 5])

    """
    m = ones((n, n), int)
    a = mask_func(m, k)
    return nonzero(a != 0)


@set_module('numpy')
def tril_indices(n, k=0, m=None):
    """
    Return the indices for the lower-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The row dimension of the arrays for which the returned
        indices will be valid.
    k : int, optional
        Diagonal offset (see `tril` for details).
    m : int, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.


    Returns
    -------
    inds : tuple of arrays
        The row and column indices, respectively. The row indices are sorted
        in non-decreasing order, and the correspdonding column indices are
        strictly increasing for each row.

    See also
    --------
    triu_indices : similar function, for upper-triangular.
    mask_indices : generic function accepting an arbitrary mask function.
    tril, triu

    Examples
    --------
    >>> import numpy as np

    Compute two different sets of indices to access 4x4 arrays, one for the
    lower triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> il1 = np.tril_indices(4)
    >>> il1
    (array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]), array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    Note that row indices (first array) are non-decreasing, and the corresponding
    column indices (second array) are strictly increasing for each row.
    Here is how they can be used with a sample array:

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Both for indexing:

    >>> a[il1]
    array([ 0,  4,  5, ..., 13, 14, 15])

    And for assigning values:

    >>> a[il1] = -1
    >>> a
    array([[-1,  1,  2,  3],
           [-1, -1,  6,  7],
           [-1, -1, -1, 11],
           [-1, -1, -1, -1]])

    These cover almost the whole array (two diagonals right of the main one):

    >>> il2 = np.tril_indices(4, 2)
    >>> a[il2] = -10
    >>> a
    array([[-10, -10, -10,   3],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10]])

    """
    tri_ = tri(n, m, k=k, dtype=bool)

    return tuple(broadcast_to(inds, tri_.shape)[tri_]
                 for inds in indices(tri_.shape, sparse=True))


def _trilu_indices_form_dispatcher(arr, k=None):
    return (arr,)


@array_function_dispatch(_trilu_indices_form_dispatcher)
def tril_indices_from(arr, k=0):
    """
    Return the indices for the lower-triangle of arr.

    See `tril_indices` for full details.

    Parameters
    ----------
    arr : array_like
        The indices will be valid for square arrays whose dimensions are
        the same as arr.
    k : int, optional
        Diagonal offset (see `tril` for details).

    Examples
    --------
    >>> import numpy as np

    Create a 4 by 4 array

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Pass the array to get the indices of the lower triangular elements.

    >>> trili = np.tril_indices_from(a)
    >>> trili
    (array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]), array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    >>> a[trili]
    array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])

    This is syntactic sugar for tril_indices().

    >>> np.tril_indices(a.shape[0])
    (array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]), array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    Use the `k` parameter to return the indices for the lower triangular array
    up to the k-th diagonal.

    >>> trili1 = np.tril_indices_from(a, k=1)
    >>> a[trili1]
    array([ 0,  1,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15])

    See Also
    --------
    tril_indices, tril, triu_indices_from
    """
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])


@set_module('numpy')
def triu_indices(n, k=0, m=None):
    """
    Return the indices for the upper-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The size of the arrays for which the returned indices will
        be valid.
    k : int, optional
        Diagonal offset (see `triu` for details).
    m : int, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.


    Returns
    -------
    inds : tuple, shape(2) of ndarrays, shape(`n`)
        The row and column indices, respectively. The row indices are sorted
        in non-decreasing order, and the correspdonding column indices are
        strictly increasing for each row.

    See also
    --------
    tril_indices : similar function, for lower-triangular.
    mask_indices : generic function accepting an arbitrary mask function.
    triu, tril

    Examples
    --------
    >>> import numpy as np

    Compute two different sets of indices to access 4x4 arrays, one for the
    upper triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> iu1 = np.triu_indices(4)
    >>> iu1
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]), array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    Note that row indices (first array) are non-decreasing, and the corresponding
    column indices (second array) are strictly increasing for each row.

    Here is how they can be used with a sample array:

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Both for indexing:

    >>> a[iu1]
    array([ 0,  1,  2, ..., 10, 11, 15])

    And for assigning values:

    >>> a[iu1] = -1
    >>> a
    array([[-1, -1, -1, -1],
           [ 4, -1, -1, -1],
           [ 8,  9, -1, -1],
           [12, 13, 14, -1]])

    These cover only a small part of the whole array (two diagonals right
    of the main one):

    >>> iu2 = np.triu_indices(4, 2)
    >>> a[iu2] = -10
    >>> a
    array([[ -1,  -1, -10, -10],
           [  4,  -1,  -1, -10],
           [  8,   9,  -1,  -1],
           [ 12,  13,  14,  -1]])

    """
    tri_ = ~tri(n, m, k=k - 1, dtype=bool)

    return tuple(broadcast_to(inds, tri_.shape)[tri_]
                 for inds in indices(tri_.shape, sparse=True))


@array_function_dispatch(_trilu_indices_form_dispatcher)
def triu_indices_from(arr, k=0):
    """
    Return the indices for the upper-triangle of arr.

    See `triu_indices` for full details.

    Parameters
    ----------
    arr : ndarray, shape(N, N)
        The indices will be valid for square arrays.
    k : int, optional
        Diagonal offset (see `triu` for details).

    Returns
    -------
    triu_indices_from : tuple, shape(2) of ndarray, shape(N)
        Indices for the upper-triangle of `arr`.

    Examples
    --------
    >>> import numpy as np

    Create a 4 by 4 array

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Pass the array to get the indices of the upper triangular elements.

    >>> triui = np.triu_indices_from(a)
    >>> triui
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]), array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    >>> a[triui]
    array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])

    This is syntactic sugar for triu_indices().

    >>> np.triu_indices(a.shape[0])
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]), array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    Use the `k` parameter to return the indices for the upper triangular array
    from the k-th diagonal.

    >>> triuim1 = np.triu_indices_from(a, k=1)
    >>> a[triuim1]
    array([ 1,  2,  3,  6,  7, 11])


    See Also
    --------
    triu_indices, triu, tril_indices_from
    """
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_loadtxt.py ===
"""
Tests specific to `np.loadtxt` added during the move of loadtxt to be backed
by C code.
These tests complement those found in `test_io.py`.
"""

import os
import sys
from io import StringIO
from tempfile import NamedTemporaryFile, mkstemp

import pytest

import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import HAS_REFCOUNT, IS_PYPY, assert_array_equal


def test_scientific_notation():
    """Test that both 'e' and 'E' are parsed correctly."""
    data = StringIO(

            "1.0e-1,2.0E1,3.0\n"
            "4.0e-2,5.0E-1,6.0\n"
            "7.0e-3,8.0E1,9.0\n"
            "0.0e-4,1.0E-1,2.0"

    )
    expected = np.array(
        [[0.1, 20., 3.0], [0.04, 0.5, 6], [0.007, 80., 9], [0, 0.1, 2]]
    )
    assert_array_equal(np.loadtxt(data, delimiter=","), expected)


@pytest.mark.parametrize("comment", ["..", "//", "@-", "this is a comment:"])
def test_comment_multiple_chars(comment):
    content = "# IGNORE\n1.5, 2.5# ABC\n3.0,4.0# XXX\n5.5,6.0\n"
    txt = StringIO(content.replace("#", comment))
    a = np.loadtxt(txt, delimiter=",", comments=comment)
    assert_equal(a, [[1.5, 2.5], [3.0, 4.0], [5.5, 6.0]])


@pytest.fixture
def mixed_types_structured():
    """
    Fixture providing heterogeneous input data with a structured dtype, along
    with the associated structured array.
    """
    data = StringIO(

            "1000;2.4;alpha;-34\n"
            "2000;3.1;beta;29\n"
            "3500;9.9;gamma;120\n"
            "4090;8.1;delta;0\n"
            "5001;4.4;epsilon;-99\n"
            "6543;7.8;omega;-1\n"

    )
    dtype = np.dtype(
        [('f0', np.uint16), ('f1', np.float64), ('f2', 'S7'), ('f3', np.int8)]
    )
    expected = np.array(
        [
            (1000, 2.4, "alpha", -34),
            (2000, 3.1, "beta", 29),
            (3500, 9.9, "gamma", 120),
            (4090, 8.1, "delta", 0),
            (5001, 4.4, "epsilon", -99),
            (6543, 7.8, "omega", -1)
        ],
        dtype=dtype
    )
    return data, dtype, expected


@pytest.mark.parametrize('skiprows', [0, 1, 2, 3])
def test_structured_dtype_and_skiprows_no_empty_lines(
        skiprows, mixed_types_structured):
    data, dtype, expected = mixed_types_structured
    a = np.loadtxt(data, dtype=dtype, delimiter=";", skiprows=skiprows)
    assert_array_equal(a, expected[skiprows:])


def test_unpack_structured(mixed_types_structured):
    data, dtype, expected = mixed_types_structured

    a, b, c, d = np.loadtxt(data, dtype=dtype, delimiter=";", unpack=True)
    assert_array_equal(a, expected["f0"])
    assert_array_equal(b, expected["f1"])
    assert_array_equal(c, expected["f2"])
    assert_array_equal(d, expected["f3"])


def test_structured_dtype_with_shape():
    dtype = np.dtype([("a", "u1", 2), ("b", "u1", 2)])
    data = StringIO("0,1,2,3\n6,7,8,9\n")
    expected = np.array([((0, 1), (2, 3)), ((6, 7), (8, 9))], dtype=dtype)
    assert_array_equal(np.loadtxt(data, delimiter=",", dtype=dtype), expected)


def test_structured_dtype_with_multi_shape():
    dtype = np.dtype([("a", "u1", (2, 2))])
    data = StringIO("0 1 2 3\n")
    expected = np.array([(((0, 1), (2, 3)),)], dtype=dtype)
    assert_array_equal(np.loadtxt(data, dtype=dtype), expected)


def test_nested_structured_subarray():
    # Test from gh-16678
    point = np.dtype([('x', float), ('y', float)])
    dt = np.dtype([('code', int), ('points', point, (2,))])
    data = StringIO("100,1,2,3,4\n200,5,6,7,8\n")
    expected = np.array(
        [
            (100, [(1., 2.), (3., 4.)]),
            (200, [(5., 6.), (7., 8.)]),
        ],
        dtype=dt
    )
    assert_array_equal(np.loadtxt(data, dtype=dt, delimiter=","), expected)


def test_structured_dtype_offsets():
    # An aligned structured dtype will have additional padding
    dt = np.dtype("i1, i4, i1, i4, i1, i4", align=True)
    data = StringIO("1,2,3,4,5,6\n7,8,9,10,11,12\n")
    expected = np.array([(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)], dtype=dt)
    assert_array_equal(np.loadtxt(data, delimiter=",", dtype=dt), expected)


@pytest.mark.parametrize("param", ("skiprows", "max_rows"))
def test_exception_negative_row_limits(param):
    """skiprows and max_rows should raise for negative parameters."""
    with pytest.raises(ValueError, match="argument must be nonnegative"):
        np.loadtxt("foo.bar", **{param: -3})


@pytest.mark.parametrize("param", ("skiprows", "max_rows"))
def test_exception_noninteger_row_limits(param):
    with pytest.raises(TypeError, match="argument must be an integer"):
        np.loadtxt("foo.bar", **{param: 1.0})


@pytest.mark.parametrize(
    "data, shape",
    [
        ("1 2 3 4 5\n", (1, 5)),  # Single row
        ("1\n2\n3\n4\n5\n", (5, 1)),  # Single column
    ]
)
def test_ndmin_single_row_or_col(data, shape):
    arr = np.array([1, 2, 3, 4, 5])
    arr2d = arr.reshape(shape)

    assert_array_equal(np.loadtxt(StringIO(data), dtype=int), arr)
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int, ndmin=0), arr)
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int, ndmin=1), arr)
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int, ndmin=2), arr2d)


@pytest.mark.parametrize("badval", [-1, 3, None, "plate of shrimp"])
def test_bad_ndmin(badval):
    with pytest.raises(ValueError, match="Illegal value of ndmin keyword"):
        np.loadtxt("foo.bar", ndmin=badval)


@pytest.mark.parametrize(
    "ws",
    (
            " ",  # space
            "\t",  # tab
            "\u2003",  # em
            "\u00A0",  # non-break
            "\u3000",  # ideographic space
    )
)
def test_blank_lines_spaces_delimit(ws):
    txt = StringIO(
        f"1 2{ws}30\n\n{ws}\n"
        f"4 5 60{ws}\n  {ws}  \n"
        f"7 8 {ws} 90\n  # comment\n"
        f"3 2 1"
    )
    # NOTE: It is unclear that the `  # comment` should succeed. Except
    #       for delimiter=None, which should use any whitespace (and maybe
    #       should just be implemented closer to Python
    expected = np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]])
    assert_equal(
        np.loadtxt(txt, dtype=int, delimiter=None, comments="#"), expected
    )


def test_blank_lines_normal_delimiter():
    txt = StringIO('1,2,30\n\n4,5,60\n\n7,8,90\n# comment\n3,2,1')
    expected = np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]])
    assert_equal(
        np.loadtxt(txt, dtype=int, delimiter=',', comments="#"), expected
    )


@pytest.mark.parametrize("dtype", (float, object))
def test_maxrows_no_blank_lines(dtype):
    txt = StringIO("1.5,2.5\n3.0,4.0\n5.5,6.0")
    res = np.loadtxt(txt, dtype=dtype, delimiter=",", max_rows=2)
    assert_equal(res.dtype, dtype)
    assert_equal(res, np.array([["1.5", "2.5"], ["3.0", "4.0"]], dtype=dtype))


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", (np.dtype("f8"), np.dtype("i2")))
def test_exception_message_bad_values(dtype):
    txt = StringIO("1,2\n3,XXX\n5,6")
    msg = f"could not convert string 'XXX' to {dtype} at row 1, column 2"
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(txt, dtype=dtype, delimiter=",")


def test_converters_negative_indices():
    txt = StringIO('1.5,2.5\n3.0,XXX\n5.5,6.0')
    conv = {-1: lambda s: np.nan if s == 'XXX' else float(s)}
    expected = np.array([[1.5, 2.5], [3.0, np.nan], [5.5, 6.0]])
    res = np.loadtxt(txt, dtype=np.float64, delimiter=",", converters=conv)
    assert_equal(res, expected)


def test_converters_negative_indices_with_usecols():
    txt = StringIO('1.5,2.5,3.5\n3.0,4.0,XXX\n5.5,6.0,7.5\n')
    conv = {-1: lambda s: np.nan if s == 'XXX' else float(s)}
    expected = np.array([[1.5, 3.5], [3.0, np.nan], [5.5, 7.5]])
    res = np.loadtxt(
        txt,
        dtype=np.float64,
        delimiter=",",
        converters=conv,
        usecols=[0, -1],
    )
    assert_equal(res, expected)

    # Second test with variable number of rows:
    res = np.loadtxt(StringIO('''0,1,2\n0,1,2,3,4'''), delimiter=",",
                     usecols=[0, -1], converters={-1: (lambda x: -1)})
    assert_array_equal(res, [[0, -1], [0, -1]])


def test_ragged_error():
    rows = ["1,2,3", "1,2,3", "4,3,2,1"]
    with pytest.raises(ValueError,
            match="the number of columns changed from 3 to 4 at row 3"):
        np.loadtxt(rows, delimiter=",")


def test_ragged_usecols():
    # usecols, and negative ones, work even with varying number of columns.
    txt = StringIO("0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n")
    expected = np.array([[0, 0], [0, 0], [0, 0]])
    res = np.loadtxt(txt, dtype=float, delimiter=",", usecols=[0, -2])
    assert_equal(res, expected)

    txt = StringIO("0,0,XXX\n0\n0,XXX,XXX,0,XXX\n")
    with pytest.raises(ValueError,
                match="invalid column index -2 at row 2 with 1 columns"):
        # There is no -2 column in the second row:
        np.loadtxt(txt, dtype=float, delimiter=",", usecols=[0, -2])


def test_empty_usecols():
    txt = StringIO("0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n")
    res = np.loadtxt(txt, dtype=np.dtype([]), delimiter=",", usecols=[])
    assert res.shape == (3,)
    assert res.dtype == np.dtype([])


@pytest.mark.parametrize("c1", ["a", "の", "🫕"])
@pytest.mark.parametrize("c2", ["a", "の", "🫕"])
def test_large_unicode_characters(c1, c2):
    # c1 and c2 span ascii, 16bit and 32bit range.
    txt = StringIO(f"a,{c1},c,1.0\ne,{c2},2.0,g")
    res = np.loadtxt(txt, dtype=np.dtype('U12'), delimiter=",")
    expected = np.array(
        [f"a,{c1},c,1.0".split(","), f"e,{c2},2.0,g".split(",")],
        dtype=np.dtype('U12')
    )
    assert_equal(res, expected)


def test_unicode_with_converter():
    txt = StringIO("cat,dog\nαβγ,δεζ\nabc,def\n")
    conv = {0: lambda s: s.upper()}
    res = np.loadtxt(
        txt,
        dtype=np.dtype("U12"),
        converters=conv,
        delimiter=",",
        encoding=None
    )
    expected = np.array([['CAT', 'dog'], ['ΑΒΓ', 'δεζ'], ['ABC', 'def']])
    assert_equal(res, expected)


def test_converter_with_structured_dtype():
    txt = StringIO('1.5,2.5,Abc\n3.0,4.0,dEf\n5.5,6.0,ghI\n')
    dt = np.dtype([('m', np.int32), ('r', np.float32), ('code', 'U8')])
    conv = {0: lambda s: int(10 * float(s)), -1: lambda s: s.upper()}
    res = np.loadtxt(txt, dtype=dt, delimiter=",", converters=conv)
    expected = np.array(
        [(15, 2.5, 'ABC'), (30, 4.0, 'DEF'), (55, 6.0, 'GHI')], dtype=dt
    )
    assert_equal(res, expected)


def test_converter_with_unicode_dtype():
    """
    With the 'bytes' encoding, tokens are encoded prior to being
    passed to the converter. This means that the output of the converter may
    be bytes instead of unicode as expected by `read_rows`.

    This test checks that outputs from the above scenario are properly decoded
    prior to parsing by `read_rows`.
    """
    txt = StringIO('abc,def\nrst,xyz')
    conv = bytes.upper
    res = np.loadtxt(
            txt, dtype=np.dtype("U3"), converters=conv, delimiter=",",
            encoding="bytes")
    expected = np.array([['ABC', 'DEF'], ['RST', 'XYZ']])
    assert_equal(res, expected)


def test_read_huge_row():
    row = "1.5, 2.5," * 50000
    row = row[:-1] + "\n"
    txt = StringIO(row * 2)
    res = np.loadtxt(txt, delimiter=",", dtype=float)
    assert_equal(res, np.tile([1.5, 2.5], (2, 50000)))


@pytest.mark.parametrize("dtype", "edfgFDG")
def test_huge_float(dtype):
    # Covers a non-optimized path that is rarely taken:
    field = "0" * 1000 + ".123456789"
    dtype = np.dtype(dtype)
    value = np.loadtxt([field], dtype=dtype)[()]
    assert value == dtype.type("0.123456789")


@pytest.mark.parametrize(
    ("given_dtype", "expected_dtype"),
    [
        ("S", np.dtype("S5")),
        ("U", np.dtype("U5")),
    ],
)
def test_string_no_length_given(given_dtype, expected_dtype):
    """
    The given dtype is just 'S' or 'U' with no length. In these cases, the
    length of the resulting dtype is determined by the longest string found
    in the file.
    """
    txt = StringIO("AAA,5-1\nBBBBB,0-3\nC,4-9\n")
    res = np.loadtxt(txt, dtype=given_dtype, delimiter=",")
    expected = np.array(
        [['AAA', '5-1'], ['BBBBB', '0-3'], ['C', '4-9']], dtype=expected_dtype
    )
    assert_equal(res, expected)
    assert_equal(res.dtype, expected_dtype)


def test_float_conversion():
    """
    Some tests that the conversion to float64 works as accurately as the
    Python built-in `float` function. In a naive version of the float parser,
    these strings resulted in values that were off by an ULP or two.
    """
    strings = [
        '0.9999999999999999',
        '9876543210.123456',
        '5.43215432154321e+300',
        '0.901',
        '0.333',
    ]
    txt = StringIO('\n'.join(strings))
    res = np.loadtxt(txt)
    expected = np.array([float(s) for s in strings])
    assert_equal(res, expected)


def test_bool():
    # Simple test for bool via integer
    txt = StringIO("1, 0\n10, -1")
    res = np.loadtxt(txt, dtype=bool, delimiter=",")
    assert res.dtype == bool
    assert_array_equal(res, [[True, False], [True, True]])
    # Make sure we use only 1 and 0 on the byte level:
    assert_array_equal(res.view(np.uint8), [[1, 0], [1, 1]])


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.filterwarnings("error:.*integer via a float.*:DeprecationWarning")
def test_integer_signs(dtype):
    dtype = np.dtype(dtype)
    assert np.loadtxt(["+2"], dtype=dtype) == 2
    if dtype.kind == "u":
        with pytest.raises(ValueError):
            np.loadtxt(["-1\n"], dtype=dtype)
    else:
        assert np.loadtxt(["-2\n"], dtype=dtype) == -2

    for sign in ["++", "+-", "--", "-+"]:
        with pytest.raises(ValueError):
            np.loadtxt([f"{sign}2\n"], dtype=dtype)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.filterwarnings("error:.*integer via a float.*:DeprecationWarning")
def test_implicit_cast_float_to_int_fails(dtype):
    txt = StringIO("1.0, 2.1, 3.7\n4, 5, 6")
    with pytest.raises(ValueError):
        np.loadtxt(txt, dtype=dtype, delimiter=",")

@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("with_parens", (False, True))
def test_complex_parsing(dtype, with_parens):
    s = "(1.0-2.5j),3.75,(7+-5.0j)\n(4),(-19e2j),(0)"
    if not with_parens:
        s = s.replace("(", "").replace(")", "")

    res = np.loadtxt(StringIO(s), dtype=dtype, delimiter=",")
    expected = np.array(
        [[1.0 - 2.5j, 3.75, 7 - 5j], [4.0, -1900j, 0]], dtype=dtype
    )
    assert_equal(res, expected)


def test_read_from_generator():
    def gen():
        for i in range(4):
            yield f"{i},{2 * i},{i**2}"

    res = np.loadtxt(gen(), dtype=int, delimiter=",")
    expected = np.array([[0, 0, 0], [1, 2, 1], [2, 4, 4], [3, 6, 9]])
    assert_equal(res, expected)


def test_read_from_generator_multitype():
    def gen():
        for i in range(3):
            yield f"{i} {i / 4}"

    res = np.loadtxt(gen(), dtype="i, d", delimiter=" ")
    expected = np.array([(0, 0.0), (1, 0.25), (2, 0.5)], dtype="i, d")
    assert_equal(res, expected)


def test_read_from_bad_generator():
    def gen():
        yield from ["1,2", b"3, 5", 12738]

    with pytest.raises(
            TypeError, match=r"non-string returned while reading data"):
        np.loadtxt(gen(), dtype="i, i", delimiter=",")


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_object_cleanup_on_read_error():
    sentinel = object()
    already_read = 0

    def conv(x):
        nonlocal already_read
        if already_read > 4999:
            raise ValueError("failed half-way through!")
        already_read += 1
        return sentinel

    txt = StringIO("x\n" * 10000)

    with pytest.raises(ValueError, match="at row 5000, column 1"):
        np.loadtxt(txt, dtype=object, converters={0: conv})

    assert sys.getrefcount(sentinel) == 2


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_character_not_bytes_compatible():
    """Test exception when a character cannot be encoded as 'S'."""
    data = StringIO("–")  # == \u2013
    with pytest.raises(ValueError):
        np.loadtxt(data, dtype="S5")


@pytest.mark.parametrize("conv", (0, [float], ""))
def test_invalid_converter(conv):
    msg = (
        "converters must be a dictionary mapping columns to converter "
        "functions or a single callable."
    )
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(StringIO("1 2\n3 4"), converters=conv)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_converters_dict_raises_non_integer_key():
    with pytest.raises(TypeError, match="keys of the converters dict"):
        np.loadtxt(StringIO("1 2\n3 4"), converters={"a": int})
    with pytest.raises(TypeError, match="keys of the converters dict"):
        np.loadtxt(StringIO("1 2\n3 4"), converters={"a": int}, usecols=0)


@pytest.mark.parametrize("bad_col_ind", (3, -3))
def test_converters_dict_raises_non_col_key(bad_col_ind):
    data = StringIO("1 2\n3 4")
    with pytest.raises(ValueError, match="converter specified for column"):
        np.loadtxt(data, converters={bad_col_ind: int})


def test_converters_dict_raises_val_not_callable():
    with pytest.raises(TypeError,
                match="values of the converters dictionary must be callable"):
        np.loadtxt(StringIO("1 2\n3 4"), converters={0: 1})


@pytest.mark.parametrize("q", ('"', "'", "`"))
def test_quoted_field(q):
    txt = StringIO(
        f"{q}alpha, x{q}, 2.5\n{q}beta, y{q}, 4.5\n{q}gamma, z{q}, 5.0\n"
    )
    dtype = np.dtype([('f0', 'U8'), ('f1', np.float64)])
    expected = np.array(
        [("alpha, x", 2.5), ("beta, y", 4.5), ("gamma, z", 5.0)], dtype=dtype
    )

    res = np.loadtxt(txt, dtype=dtype, delimiter=",", quotechar=q)
    assert_array_equal(res, expected)


@pytest.mark.parametrize("q", ('"', "'", "`"))
def test_quoted_field_with_whitepace_delimiter(q):
    txt = StringIO(
        f"{q}alpha, x{q}     2.5\n{q}beta, y{q} 4.5\n{q}gamma, z{q}   5.0\n"
    )
    dtype = np.dtype([('f0', 'U8'), ('f1', np.float64)])
    expected = np.array(
        [("alpha, x", 2.5), ("beta, y", 4.5), ("gamma, z", 5.0)], dtype=dtype
    )

    res = np.loadtxt(txt, dtype=dtype, delimiter=None, quotechar=q)
    assert_array_equal(res, expected)


def test_quote_support_default():
    """Support for quoted fields is disabled by default."""
    txt = StringIO('"lat,long", 45, 30\n')
    dtype = np.dtype([('f0', 'U24'), ('f1', np.float64), ('f2', np.float64)])

    with pytest.raises(ValueError,
            match="the dtype passed requires 3 columns but 4 were"):
        np.loadtxt(txt, dtype=dtype, delimiter=",")

    # Enable quoting support with non-None value for quotechar param
    txt.seek(0)
    expected = np.array([("lat,long", 45., 30.)], dtype=dtype)

    res = np.loadtxt(txt, dtype=dtype, delimiter=",", quotechar='"')
    assert_array_equal(res, expected)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_quotechar_multichar_error():
    txt = StringIO("1,2\n3,4")
    msg = r".*must be a single unicode character or None"
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, delimiter=",", quotechar="''")


def test_comment_multichar_error_with_quote():
    txt = StringIO("1,2\n3,4")
    msg = (
        "when multiple comments or a multi-character comment is given, "
        "quotes are not supported."
    )
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(txt, delimiter=",", comments="123", quotechar='"')
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(txt, delimiter=",", comments=["#", "%"], quotechar='"')

    # A single character string in a tuple is unpacked though:
    res = np.loadtxt(txt, delimiter=",", comments=("#",), quotechar="'")
    assert_equal(res, [[1, 2], [3, 4]])


def test_structured_dtype_with_quotes():
    data = StringIO(

            "1000;2.4;'alpha';-34\n"
            "2000;3.1;'beta';29\n"
            "3500;9.9;'gamma';120\n"
            "4090;8.1;'delta';0\n"
            "5001;4.4;'epsilon';-99\n"
            "6543;7.8;'omega';-1\n"

    )
    dtype = np.dtype(
        [('f0', np.uint16), ('f1', np.float64), ('f2', 'S7'), ('f3', np.int8)]
    )
    expected = np.array(
        [
            (1000, 2.4, "alpha", -34),
            (2000, 3.1, "beta", 29),
            (3500, 9.9, "gamma", 120),
            (4090, 8.1, "delta", 0),
            (5001, 4.4, "epsilon", -99),
            (6543, 7.8, "omega", -1)
        ],
        dtype=dtype
    )
    res = np.loadtxt(data, dtype=dtype, delimiter=";", quotechar="'")
    assert_array_equal(res, expected)


def test_quoted_field_is_not_empty():
    txt = StringIO('1\n\n"4"\n""')
    expected = np.array(["1", "4", ""], dtype="U1")
    res = np.loadtxt(txt, delimiter=",", dtype="U1", quotechar='"')
    assert_equal(res, expected)

def test_quoted_field_is_not_empty_nonstrict():
    # Same as test_quoted_field_is_not_empty but check that we are not strict
    # about missing closing quote (this is the `csv.reader` default also)
    txt = StringIO('1\n\n"4"\n"')
    expected = np.array(["1", "4", ""], dtype="U1")
    res = np.loadtxt(txt, delimiter=",", dtype="U1", quotechar='"')
    assert_equal(res, expected)

def test_consecutive_quotechar_escaped():
    txt = StringIO('"Hello, my name is ""Monty""!"')
    expected = np.array('Hello, my name is "Monty"!', dtype="U40")
    res = np.loadtxt(txt, dtype="U40", delimiter=",", quotechar='"')
    assert_equal(res, expected)


@pytest.mark.parametrize("data", ("", "\n\n\n", "# 1 2 3\n# 4 5 6\n"))
@pytest.mark.parametrize("ndmin", (0, 1, 2))
@pytest.mark.parametrize("usecols", [None, (1, 2, 3)])
def test_warn_on_no_data(data, ndmin, usecols):
    """Check that a UserWarning is emitted when no data is read from input."""
    if usecols is not None:
        expected_shape = (0, 3)
    elif ndmin == 2:
        expected_shape = (0, 1)  # guess a single column?!
    else:
        expected_shape = (0,)

    txt = StringIO(data)
    with pytest.warns(UserWarning, match="input contained no data"):
        res = np.loadtxt(txt, ndmin=ndmin, usecols=usecols)
    assert res.shape == expected_shape

    with NamedTemporaryFile(mode="w") as fh:
        fh.write(data)
        fh.seek(0)
        with pytest.warns(UserWarning, match="input contained no data"):
            res = np.loadtxt(txt, ndmin=ndmin, usecols=usecols)
        assert res.shape == expected_shape

@pytest.mark.parametrize("skiprows", (2, 3))
def test_warn_on_skipped_data(skiprows):
    data = "1 2 3\n4 5 6"
    txt = StringIO(data)
    with pytest.warns(UserWarning, match="input contained no data"):
        np.loadtxt(txt, skiprows=skiprows)


@pytest.mark.parametrize(["dtype", "value"], [
        ("i2", 0x0001), ("u2", 0x0001),
        ("i4", 0x00010203), ("u4", 0x00010203),
        ("i8", 0x0001020304050607), ("u8", 0x0001020304050607),
        # The following values are constructed to lead to unique bytes:
        ("float16", 3.07e-05),
        ("float32", 9.2557e-41), ("complex64", 9.2557e-41 + 2.8622554e-29j),
        ("float64", -1.758571353180402e-24),
        # Here and below, the repr side-steps a small loss of precision in
        # complex `str` in PyPy (which is probably fine, as repr works):
        ("complex128", repr(5.406409232372729e-29 - 1.758571353180402e-24j)),
        # Use integer values that fit into double.  Everything else leads to
        # problems due to longdoubles going via double and decimal strings
        # causing rounding errors.
        ("longdouble", 0x01020304050607),
        ("clongdouble", repr(0x01020304050607 + (0x00121314151617 * 1j))),
        ("U2", "\U00010203\U000a0b0c")])
@pytest.mark.parametrize("swap", [True, False])
def test_byteswapping_and_unaligned(dtype, value, swap):
    # Try to create "interesting" values within the valid unicode range:
    dtype = np.dtype(dtype)
    data = [f"x,{value}\n"]  # repr as PyPy `str` truncates some
    if swap:
        dtype = dtype.newbyteorder()
    full_dt = np.dtype([("a", "S1"), ("b", dtype)], align=False)
    # The above ensures that the interesting "b" field is unaligned:
    assert full_dt.fields["b"][1] == 1
    res = np.loadtxt(data, dtype=full_dt, delimiter=",",
                     max_rows=1)  # max-rows prevents over-allocation
    assert res["b"] == dtype.type(value)


@pytest.mark.parametrize("dtype",
        np.typecodes["AllInteger"] + "efdFD" + "?")
def test_unicode_whitespace_stripping(dtype):
    # Test that all numeric types (and bool) strip whitespace correctly
    # \u202F is a narrow no-break space, `\n` is just a whitespace if quoted.
    # Currently, skip float128 as it did not always support this and has no
    # "custom" parsing:
    txt = StringIO(' 3 ,"\u202F2\n"')
    res = np.loadtxt(txt, dtype=dtype, delimiter=",", quotechar='"')
    assert_array_equal(res, np.array([3, 2]).astype(dtype))


@pytest.mark.parametrize("dtype", "FD")
def test_unicode_whitespace_stripping_complex(dtype):
    # Complex has a few extra cases since it has two components and
    # parentheses
    line = " 1 , 2+3j , ( 4+5j ), ( 6+-7j )  , 8j , ( 9j ) \n"
    data = [line, line.replace(" ", "\u202F")]
    res = np.loadtxt(data, dtype=dtype, delimiter=',')
    assert_array_equal(res, np.array([[1, 2 + 3j, 4 + 5j, 6 - 7j, 8j, 9j]] * 2))


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", "FD")
@pytest.mark.parametrize("field",
        ["1 +2j", "1+ 2j", "1+2 j", "1+-+3", "(1j", "(1", "(1+2j", "1+2j)"])
def test_bad_complex(dtype, field):
    with pytest.raises(ValueError):
        np.loadtxt([field + "\n"], dtype=dtype, delimiter=",")


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype",
            np.typecodes["AllInteger"] + "efgdFDG" + "?")
def test_nul_character_error(dtype):
    # Test that a \0 character is correctly recognized as an error even if
    # what comes before is valid (not everything gets parsed internally).
    if dtype.lower() == "g":
        pytest.xfail("longdouble/clongdouble assignment may misbehave.")
    with pytest.raises(ValueError):
        np.loadtxt(["1\000"], dtype=dtype, delimiter=",", quotechar='"')


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype",
        np.typecodes["AllInteger"] + "efgdFDG" + "?")
def test_no_thousands_support(dtype):
    # Mainly to document behaviour, Python supports thousands like 1_1.
    # (e and G may end up using different conversion and support it, this is
    # a bug but happens...)
    if dtype == "e":
        pytest.skip("half assignment currently uses Python float converter")
    if dtype in "eG":
        pytest.xfail("clongdouble assignment is buggy (uses `complex`?).")

    assert int("1_1") == float("1_1") == complex("1_1") == 11
    with pytest.raises(ValueError):
        np.loadtxt(["1_1\n"], dtype=dtype)


@pytest.mark.parametrize("data", [
    ["1,2\n", "2\n,3\n"],
    ["1,2\n", "2\r,3\n"]])
def test_bad_newline_in_iterator(data):
    # In NumPy <=1.22 this was accepted, because newlines were completely
    # ignored when the input was an iterable.  This could be changed, but right
    # now, we raise an error.
    msg = "Found an unquoted embedded newline within a single line"
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(data, delimiter=",")


@pytest.mark.parametrize("data", [
    ["1,2\n", "2,3\r\n"],  # a universal newline
    ["1,2\n", "'2\n',3\n"],  # a quoted newline
    ["1,2\n", "'2\r',3\n"],
    ["1,2\n", "'2\r\n',3\n"],
])
def test_good_newline_in_iterator(data):
    # The quoted newlines will be untransformed here, but are just whitespace.
    res = np.loadtxt(data, delimiter=",", quotechar="'")
    assert_array_equal(res, [[1., 2.], [2., 3.]])


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n"])
def test_universal_newlines_quoted(newline):
    # Check that universal newline support within the tokenizer is not applied
    # to quoted fields.  (note that lines must end in newline or quoted
    # fields will not include a newline at all)
    data = ['1,"2\n"\n', '3,"4\n', '1"\n']
    data = [row.replace("\n", newline) for row in data]
    res = np.loadtxt(data, dtype=object, delimiter=",", quotechar='"')
    assert_array_equal(res, [['1', f'2{newline}'], ['3', f'4{newline}1']])


def test_null_character():
    # Basic tests to check that the NUL character is not special:
    res = np.loadtxt(["1\0002\0003\n", "4\0005\0006"], delimiter="\000")
    assert_array_equal(res, [[1, 2, 3], [4, 5, 6]])

    # Also not as part of a field (avoid unicode/arrays as unicode strips \0)
    res = np.loadtxt(["1\000,2\000,3\n", "4\000,5\000,6"],
                     delimiter=",", dtype=object)
    assert res.tolist() == [["1\000", "2\000", "3"], ["4\000", "5\000", "6"]]


def test_iterator_fails_getting_next_line():
    class BadSequence:
        def __len__(self):
            return 100

        def __getitem__(self, item):
            if item == 50:
                raise RuntimeError("Bad things happened!")
            return f"{item}, {item + 1}"

    with pytest.raises(RuntimeError, match="Bad things happened!"):
        np.loadtxt(BadSequence(), dtype=int, delimiter=",")


class TestCReaderUnitTests:
    # These are internal tests for path that should not be possible to hit
    # unless things go very very wrong somewhere.
    def test_not_an_filelike(self):
        with pytest.raises(AttributeError, match=".*read"):
            np._core._multiarray_umath._load_from_filelike(
                object(), dtype=np.dtype("i"), filelike=True)

    def test_filelike_read_fails(self):
        # Can only be reached if loadtxt opens the file, so it is hard to do
        # via the public interface (although maybe not impossible considering
        # the current "DataClass" backing).
        class BadFileLike:
            counter = 0

            def read(self, size):
                self.counter += 1
                if self.counter > 20:
                    raise RuntimeError("Bad bad bad!")
                return "1,2,3\n"

        with pytest.raises(RuntimeError, match="Bad bad bad!"):
            np._core._multiarray_umath._load_from_filelike(
                BadFileLike(), dtype=np.dtype("i"), filelike=True)

    def test_filelike_bad_read(self):
        # Can only be reached if loadtxt opens the file, so it is hard to do
        # via the public interface (although maybe not impossible considering
        # the current "DataClass" backing).

        class BadFileLike:
            counter = 0

            def read(self, size):
                return 1234  # not a string!

        with pytest.raises(TypeError,
                    match="non-string returned while reading data"):
            np._core._multiarray_umath._load_from_filelike(
                BadFileLike(), dtype=np.dtype("i"), filelike=True)

    def test_not_an_iter(self):
        with pytest.raises(TypeError,
                    match="error reading from object, expected an iterable"):
            np._core._multiarray_umath._load_from_filelike(
                object(), dtype=np.dtype("i"), filelike=False)

    def test_bad_type(self):
        with pytest.raises(TypeError, match="internal error: dtype must"):
            np._core._multiarray_umath._load_from_filelike(
                object(), dtype="i", filelike=False)

    def test_bad_encoding(self):
        with pytest.raises(TypeError, match="encoding must be a unicode"):
            np._core._multiarray_umath._load_from_filelike(
                object(), dtype=np.dtype("i"), filelike=False, encoding=123)

    @pytest.mark.parametrize("newline", ["\r", "\n", "\r\n"])
    def test_manual_universal_newlines(self, newline):
        # This is currently not available to users, because we should always
        # open files with universal newlines enabled `newlines=None`.
        # (And reading from an iterator uses slightly different code paths.)
        # We have no real support for `newline="\r"` or `newline="\n" as the
        # user cannot specify those options.
        data = StringIO('0\n1\n"2\n"\n3\n4 #\n'.replace("\n", newline),
                        newline="")

        res = np._core._multiarray_umath._load_from_filelike(
            data, dtype=np.dtype("U10"), filelike=True,
            quote='"', comment="#", skiplines=1)
        assert_array_equal(res[:, 0], ["1", f"2{newline}", "3", "4 "])


def test_delimiter_comment_collision_raises():
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO("1, 2, 3"), delimiter=",", comments=",")


def test_delimiter_quotechar_collision_raises():
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO("1, 2, 3"), delimiter=",", quotechar=",")


def test_comment_quotechar_collision_raises():
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO("1 2 3"), comments="#", quotechar="#")


def test_delimiter_and_multiple_comments_collision_raises():
    with pytest.raises(
        TypeError, match="Comment characters.*cannot include the delimiter"
    ):
        np.loadtxt(StringIO("1, 2, 3"), delimiter=",", comments=["#", ","])


@pytest.mark.parametrize(
    "ws",
    (
        " ",  # space
        "\t",  # tab
        "\u2003",  # em
        "\u00A0",  # non-break
        "\u3000",  # ideographic space
    )
)
def test_collision_with_default_delimiter_raises(ws):
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO(f"1{ws}2{ws}3\n4{ws}5{ws}6\n"), comments=ws)
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO(f"1{ws}2{ws}3\n4{ws}5{ws}6\n"), quotechar=ws)


@pytest.mark.parametrize("nl", ("\n", "\r"))
def test_control_character_newline_raises(nl):
    txt = StringIO(f"1{nl}2{nl}3{nl}{nl}4{nl}5{nl}6{nl}{nl}")
    msg = "control character.*cannot be a newline"
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, delimiter=nl)
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, comments=nl)
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, quotechar=nl)


@pytest.mark.parametrize(
    ("generic_data", "long_datum", "unitless_dtype", "expected_dtype"),
    [
        ("2012-03", "2013-01-15", "M8", "M8[D]"),  # Datetimes
        ("spam-a-lot", "tis_but_a_scratch", "U", "U17"),  # str
    ],
)
@pytest.mark.parametrize("nrows", (10, 50000, 60000))  # lt, eq, gt chunksize
def test_parametric_unit_discovery(
    generic_data, long_datum, unitless_dtype, expected_dtype, nrows
):
    """Check that the correct unit (e.g. month, day, second) is discovered from
    the data when a user specifies a unitless datetime."""
    # Unit should be "D" (days) due to last entry
    data = [generic_data] * nrows + [long_datum]
    expected = np.array(data, dtype=expected_dtype)
    assert len(data) == nrows + 1
    assert len(data) == len(expected)

    # file-like path
    txt = StringIO("\n".join(data))
    a = np.loadtxt(txt, dtype=unitless_dtype)
    assert len(a) == len(expected)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)

    # file-obj path
    fd, fname = mkstemp()
    os.close(fd)
    with open(fname, "w") as fh:
        fh.write("\n".join(data) + "\n")
    # loading the full file...
    a = np.loadtxt(fname, dtype=unitless_dtype)
    assert len(a) == len(expected)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)
    # loading half of the file...
    a = np.loadtxt(fname, dtype=unitless_dtype, max_rows=int(nrows / 2))
    os.remove(fname)
    assert len(a) == int(nrows / 2)
    assert_equal(a, expected[:int(nrows / 2)])


def test_str_dtype_unit_discovery_with_converter():
    data = ["spam-a-lot"] * 60000 + ["XXXtis_but_a_scratch"]
    expected = np.array(
        ["spam-a-lot"] * 60000 + ["tis_but_a_scratch"], dtype="U17"
    )
    conv = lambda s: s.removeprefix("XXX")

    # file-like path
    txt = StringIO("\n".join(data))
    a = np.loadtxt(txt, dtype="U", converters=conv)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)

    # file-obj path
    fd, fname = mkstemp()
    os.close(fd)
    with open(fname, "w") as fh:
        fh.write("\n".join(data))
    a = np.loadtxt(fname, dtype="U", converters=conv)
    os.remove(fname)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_control_character_empty():
    with pytest.raises(TypeError, match="Text reading control character must"):
        np.loadtxt(StringIO("1 2 3"), delimiter="")
    with pytest.raises(TypeError, match="Text reading control character must"):
        np.loadtxt(StringIO("1 2 3"), quotechar="")
    with pytest.raises(ValueError, match="comments cannot be an empty string"):
        np.loadtxt(StringIO("1 2 3"), comments="")
    with pytest.raises(ValueError, match="comments cannot be an empty string"):
        np.loadtxt(StringIO("1 2 3"), comments=["#", ""])


def test_control_characters_as_bytes():
    """Byte control characters (comments, delimiter) are supported."""
    a = np.loadtxt(StringIO("#header\n1,2,3"), comments=b"#", delimiter=b",")
    assert_equal(a, [1, 2, 3])


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_field_growing_cases():
    # Test empty field appending/growing (each field still takes 1 character)
    # to see if the final field appending does not create issues.
    res = np.loadtxt([""], delimiter=",", dtype=bytes)
    assert len(res) == 0

    for i in range(1, 1024):
        res = np.loadtxt(["," * i], delimiter=",", dtype=bytes, max_rows=10)
        assert len(res) == i + 1

@pytest.mark.parametrize("nmax", (10000, 50000, 55000, 60000))
def test_maxrows_exceeding_chunksize(nmax):
    # tries to read all of the file,
    # or less, equal, greater than _loadtxt_chunksize
    file_length = 60000

    # file-like path
    data = ["a 0.5 1"] * file_length
    txt = StringIO("\n".join(data))
    res = np.loadtxt(txt, dtype=str, delimiter=" ", max_rows=nmax)
    assert len(res) == nmax

    # file-obj path
    fd, fname = mkstemp()
    os.close(fd)
    with open(fname, "w") as fh:
        fh.write("\n".join(data))
    res = np.loadtxt(fname, dtype=str, delimiter=" ", max_rows=nmax)
    os.remove(fname)
    assert len(res) == nmax

@pytest.mark.parametrize("nskip", (0, 10000, 12345, 50000, 67891, 100000))
def test_skiprow_exceeding_maxrows_exceeding_chunksize(tmpdir, nskip):
    # tries to read a file in chunks by skipping a variable amount of lines,
    # less, equal, greater than max_rows
    file_length = 110000
    data = "\n".join(f"{i} a 0.5 1" for i in range(1, file_length + 1))
    expected_length = min(60000, file_length - nskip)
    expected = np.arange(nskip + 1, nskip + 1 + expected_length).astype(str)

    # file-like path
    txt = StringIO(data)
    res = np.loadtxt(txt, dtype='str', delimiter=" ", skiprows=nskip, max_rows=60000)
    assert len(res) == expected_length
    # are the right lines read in res?
    assert_array_equal(expected, res[:, 0])

    # file-obj path
    tmp_file = tmpdir / "test_data.txt"
    tmp_file.write(data)
    fname = str(tmp_file)
    res = np.loadtxt(fname, dtype='str', delimiter=" ", skiprows=nskip, max_rows=60000)
    assert len(res) == expected_length
    # are the right lines read in res?
    assert_array_equal(expected, res[:, 0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_histograms_impl.py ===
"""
Histogram-related functions
"""
import contextlib
import functools
import operator
import warnings

import numpy as np
from numpy._core import overrides

__all__ = ['histogram', 'histogramdd', 'histogram_bin_edges']

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# range is a keyword argument to many functions, so save the builtin so they can
# use it.
_range = range


def _ptp(x):
    """Peak-to-peak value of x.

    This implementation avoids the problem of signed integer arrays having a
    peak-to-peak value that cannot be represented with the array's data type.
    This function returns an unsigned value for signed integer arrays.
    """
    return _unsigned_subtract(x.max(), x.min())


def _hist_bin_sqrt(x, range):
    """
    Square root histogram bin estimator.

    Bin width is inversely proportional to the data size. Used by many
    programs for its simplicity.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    return _ptp(x) / np.sqrt(x.size)


def _hist_bin_sturges(x, range):
    """
    Sturges histogram bin estimator.

    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    return _ptp(x) / (np.log2(x.size) + 1.0)


def _hist_bin_rice(x, range):
    """
    Rice histogram bin estimator.

    Another simple estimator with no normality assumption. It has better
    performance for large data than Sturges, but tends to overestimate
    the number of bins. The number of bins is proportional to the cube
    root of data size (asymptotically optimal). The estimate depends
    only on size of the data.

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    return _ptp(x) / (2.0 * x.size ** (1.0 / 3))


def _hist_bin_scott(x, range):
    """
    Scott histogram bin estimator.

    The binwidth is proportional to the standard deviation of the data
    and inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    return (24.0 * np.pi**0.5 / x.size)**(1.0 / 3.0) * np.std(x)


def _hist_bin_stone(x, range):
    """
    Histogram bin estimator based on minimizing the estimated integrated squared error (ISE).

    The number of bins is chosen by minimizing the estimated ISE against the unknown
    true distribution. The ISE is estimated using cross-validation and can be regarded
    as a generalization of Scott's rule.
    https://en.wikipedia.org/wiki/Histogram#Scott.27s_normal_reference_rule

    This paper by Stone appears to be the origination of this rule.
    https://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    range : (float, float)
        The lower and upper range of the bins.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """  # noqa: E501

    n = x.size
    ptp_x = _ptp(x)
    if n <= 1 or ptp_x == 0:
        return 0

    def jhat(nbins):
        hh = ptp_x / nbins
        p_k = np.histogram(x, bins=nbins, range=range)[0] / n
        return (2 - (n + 1) * p_k.dot(p_k)) / hh

    nbins_upper_bound = max(100, int(np.sqrt(n)))
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    if nbins == nbins_upper_bound:
        warnings.warn("The number of bins estimated may be suboptimal.",
                      RuntimeWarning, stacklevel=3)
    return ptp_x / nbins


def _hist_bin_doane(x, range):
    """
    Doane's histogram bin estimator.

    Improved version of Sturges' formula which works better for
    non-normal data. See
    stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    if x.size > 2:
        sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = np.std(x)
        if sigma > 0.0:
            # These three operations add up to
            # g1 = np.mean(((x - np.mean(x)) / sigma)**3)
            # but use only one temp array instead of three
            temp = x - np.mean(x)
            np.true_divide(temp, sigma, temp)
            np.power(temp, 3, temp)
            g1 = np.mean(temp)
            return _ptp(x) / (1.0 + np.log2(x.size) +
                                    np.log2(1.0 + np.absolute(g1) / sg1))
    return 0.0


def _hist_bin_fd(x, range):
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 0 for the bin width.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    del range  # unused
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def _hist_bin_auto(x, range):
    """
    Histogram bin estimator that uses the minimum width of a relaxed
    Freedman-Diaconis and Sturges estimators if the FD bin width does
    not result in a large number of bins. The relaxed Freedman-Diaconis estimator
    limits the bin width to half the sqrt estimated to avoid small bins.

    The FD estimator is usually the most robust method, but its width
    estimate tends to be too large for small `x` and bad for data with limited
    variance. The Sturges estimator is quite good for small (<1000) datasets
    and is the default in the R language. This method gives good off-the-shelf
    behaviour.


    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.
    range : Tuple with range for the histogram

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.

    See Also
    --------
    _hist_bin_fd, _hist_bin_sturges
    """
    fd_bw = _hist_bin_fd(x, range)
    sturges_bw = _hist_bin_sturges(x, range)
    sqrt_bw = _hist_bin_sqrt(x, range)
    # heuristic to limit the maximal number of bins
    fd_bw_corrected = max(fd_bw, sqrt_bw / 2)
    return min(fd_bw_corrected, sturges_bw)


# Private dict initialized at module load time
_hist_bin_selectors = {'stone': _hist_bin_stone,
                       'auto': _hist_bin_auto,
                       'doane': _hist_bin_doane,
                       'fd': _hist_bin_fd,
                       'rice': _hist_bin_rice,
                       'scott': _hist_bin_scott,
                       'sqrt': _hist_bin_sqrt,
                       'sturges': _hist_bin_sturges}


def _ravel_and_check_weights(a, weights):
    """ Check a and weights have matching shapes, and ravel both """
    a = np.asarray(a)

    # Ensure that the array is a "subtractable" dtype
    if a.dtype == np.bool:
        msg = f"Converting input from {a.dtype} to {np.uint8} for compatibility."
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        a = a.astype(np.uint8)

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError(
                'weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"supplied range of [{first_edge}, {last_edge}] is not finite")
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"autodetected range of [{first_edge}, {last_edge}] is not finite")

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _unsigned_subtract(a, b):
    """
    Subtract two values where a >= b, and produce an unsigned result

    This is needed when finding the difference between the upper and lower
    bound of an int16 histogram
    """
    # coerce to a single type
    signed_to_unsigned = {
        np.byte: np.ubyte,
        np.short: np.ushort,
        np.intc: np.uintc,
        np.int_: np.uint,
        np.longlong: np.ulonglong
    }
    dt = np.result_type(a, b)
    try:
        unsigned_dt = signed_to_unsigned[dt.type]
    except KeyError:
        return np.subtract(a, b, dtype=dt)
    else:
        # we know the inputs are integers, and we are deliberately casting
        # signed to unsigned.  The input may be negative python integers so
        # ensure we pass in arrays with the initial dtype (related to NEP 50).
        return np.subtract(np.asarray(a, dtype=dt), np.asarray(b, dtype=dt),
                           casting='unsafe', dtype=unsigned_dt)


def _get_bin_edges(a, bins, range, weights):
    """
    Computes the bins used internally by `histogram`.

    Parameters
    ==========
    a : ndarray
        Ravelled data array
    bins, range
        Forwarded arguments from `histogram`.
    weights : ndarray, optional
        Ravelled weights array, or None

    Returns
    =======
    bin_edges : ndarray
        Array of bin edges
    uniform_bins : (Number, Number, int):
        The upper bound, lowerbound, and number of bins, used in the optimized
        implementation of `histogram` that works on uniform bins.
    """
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None

    if isinstance(bins, str):
        bin_name = bins
        # if `bins` is a string for an automatic method,
        # this will replace it with the number of bins calculated
        if bin_name not in _hist_bin_selectors:
            raise ValueError(
                f"{bin_name!r} is not a valid estimator for `bins`")
        if weights is not None:
            raise TypeError("Automated estimation of the number of "
                            "bins is not supported for weighted data")

        first_edge, last_edge = _get_outer_edges(a, range)

        # truncate the range if needed
        if range is not None:
            keep = (a >= first_edge)
            keep &= (a <= last_edge)
            if not np.logical_and.reduce(keep):
                a = a[keep]

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge))
            if width:
                if np.issubdtype(a.dtype, np.integer) and width < 1:
                    width = 1
                delta = _unsigned_subtract(last_edge, first_edge)
                n_equal_bins = int(np.ceil(delta / width))
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif np.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError as e:
            raise TypeError(
                '`bins` must be an integer, a string, or an array') from e
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')

        first_edge, last_edge = _get_outer_edges(a, range)

    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError(
                '`bins` must increase monotonically, when an array')

    else:
        raise ValueError('`bins` must be 1d, when an array')

    if n_equal_bins is not None:
        # gh-10322 means that type resolution rules are dependent on array
        # shapes. To avoid this causing problems, we pick a type now and stick
        # with it throughout.
        bin_type = np.result_type(first_edge, last_edge, a)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)

        # bin edges must be computed
        bin_edges = np.linspace(
            first_edge, last_edge, n_equal_bins + 1,
            endpoint=True, dtype=bin_type)
        if np.any(bin_edges[:-1] >= bin_edges[1:]):
            raise ValueError(
                f'Too many bins for data range. Cannot create {n_equal_bins} '
                f'finite-sized bins.')
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        return bin_edges, None


def _search_sorted_inclusive(a, v):
    """
    Like `searchsorted`, but where the last item in `v` is placed on the right.

    In the context of a histogram, this makes the last bin edge inclusive
    """
    return np.concatenate((
        a.searchsorted(v[:-1], 'left'),
        a.searchsorted(v[-1:], 'right')
    ))


def _histogram_bin_edges_dispatcher(a, bins=None, range=None, weights=None):
    return (a, bins, weights)


@array_function_dispatch(_histogram_bin_edges_dispatcher)
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    r"""
    Function to calculate only the edges of the bins used by the `histogram`
    function.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines the bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.

        If `bins` is a string from the list below, `histogram_bin_edges` will
        use the method chosen to calculate the optimal bin width and
        consequently the number of bins (see the Notes section for more detail
        on the estimators) from the data that falls within the requested range.
        While the bin width will be optimal for the actual data
        in the range, the number of bins will be computed to fill the
        entire range, including the empty portions. For visualisation,
        using the 'auto' option is suggested. Weighted data is not
        supported for automated bin size selection.

        'auto'
            Minimum bin width between the 'sturges' and 'fd' estimators.
            Provides good all-around performance.

        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.

        'doane'
            An improved version of Sturges' estimator that works better
            with non-normal datasets.

        'scott'
            Less robust estimator that takes into account data variability
            and data size.

        'stone'
            Estimator based on leave-one-out cross-validation estimate of
            the integrated squared error. Can be regarded as a generalization
            of Scott's rule.

        'rice'
            Estimator does not take variability into account, only data
            size. Commonly overestimates number of bins required.

        'sturges'
            R's default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins
            for large non-gaussian datasets.

        'sqrt'
            Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.

    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). This is currently not used by any of the bin estimators,
        but may be in the future.

    Returns
    -------
    bin_edges : array of dtype float
        The edges to pass into `histogram`

    See Also
    --------
    histogram

    Notes
    -----
    The methods to estimate the optimal number of bins are well founded
    in literature, and are inspired by the choices R provides for
    histogram visualisation. Note that having the number of bins
    proportional to :math:`n^{1/3}` is asymptotically optimal, which is
    why it appears in most estimators. These are simply plug-in methods
    that give good starting points for number of bins. In the equations
    below, :math:`h` is the binwidth and :math:`n_h` is the number of
    bins. All estimators that compute bin counts are recast to bin width
    using the `ptp` of the data. The final bin count is obtained from
    ``np.round(np.ceil(range / h))``. The final bin width is often less
    than what is returned by the estimators below.

    'auto' (minimum bin width of the 'sturges' and 'fd' estimators)
        A compromise to get a good value. For small datasets the Sturges
        value will usually be chosen, while larger datasets will usually
        default to FD.  Avoids the overly conservative behaviour of FD
        and Sturges for small and large datasets respectively.
        Switchover point is usually :math:`a.size \approx 1000`.

    'fd' (Freedman Diaconis Estimator)
        .. math:: h = 2 \frac{IQR}{n^{1/3}}

        The binwidth is proportional to the interquartile range (IQR)
        and inversely proportional to cube root of a.size. Can be too
        conservative for small datasets, but is quite good for large
        datasets. The IQR is very robust to outliers.

    'scott'
        .. math:: h = \sigma \sqrt[3]{\frac{24 \sqrt{\pi}}{n}}

        The binwidth is proportional to the standard deviation of the
        data and inversely proportional to cube root of ``x.size``. Can
        be too conservative for small datasets, but is quite good for
        large datasets. The standard deviation is not very robust to
        outliers. Values are very similar to the Freedman-Diaconis
        estimator in the absence of outliers.

    'rice'
        .. math:: n_h = 2n^{1/3}

        The number of bins is only proportional to cube root of
        ``a.size``. It tends to overestimate the number of bins and it
        does not take into account data variability.

    'sturges'
        .. math:: n_h = \log _{2}(n) + 1

        The number of bins is the base 2 log of ``a.size``.  This
        estimator assumes normality of data and is too conservative for
        larger, non-normal datasets. This is the default method in R's
        ``hist`` method.

    'doane'
        .. math:: n_h = 1 + \log_{2}(n) +
                        \log_{2}\left(1 + \frac{|g_1|}{\sigma_{g_1}}\right)

            g_1 = mean\left[\left(\frac{x - \mu}{\sigma}\right)^3\right]

            \sigma_{g_1} = \sqrt{\frac{6(n - 2)}{(n + 1)(n + 3)}}

        An improved version of Sturges' formula that produces better
        estimates for non-normal datasets. This estimator attempts to
        account for the skew of the data.

    'sqrt'
        .. math:: n_h = \sqrt n

        The simplest and fastest estimator. Only takes into account the
        data size.

    Additionally, if the data is of integer dtype, then the binwidth will never
    be less than 1.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
    >>> np.histogram_bin_edges(arr, bins='auto', range=(0, 1))
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> np.histogram_bin_edges(arr, bins=2)
    array([0. , 2.5, 5. ])

    For consistency with histogram, an array of pre-computed bins is
    passed through unmodified:

    >>> np.histogram_bin_edges(arr, [1, 2])
    array([1, 2])

    This function allows one set of bins to be computed, and reused across
    multiple histograms:

    >>> shared_bins = np.histogram_bin_edges(arr, bins='auto')
    >>> shared_bins
    array([0., 1., 2., 3., 4., 5.])

    >>> group_id = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    >>> hist_0, _ = np.histogram(arr[group_id == 0], bins=shared_bins)
    >>> hist_1, _ = np.histogram(arr[group_id == 1], bins=shared_bins)

    >>> hist_0; hist_1
    array([1, 1, 0, 1, 0])
    array([2, 0, 1, 1, 2])

    Which gives more easily comparable results than using separate bins for
    each histogram:

    >>> hist_0, bins_0 = np.histogram(arr[group_id == 0], bins='auto')
    >>> hist_1, bins_1 = np.histogram(arr[group_id == 1], bins='auto')
    >>> hist_0; hist_1
    array([1, 1, 1])
    array([2, 1, 1, 2])
    >>> bins_0; bins_1
    array([0., 1., 2., 3.])
    array([0.  , 1.25, 2.5 , 3.75, 5.  ])

    """
    a, weights = _ravel_and_check_weights(a, weights)
    bin_edges, _ = _get_bin_edges(a, bins, range, weights)
    return bin_edges


def _histogram_dispatcher(
        a, bins=None, range=None, density=None, weights=None):
    return (a, bins, weights)


@array_function_dispatch(_histogram_dispatcher)
def histogram(a, bins=10, range=None, density=None, weights=None):
    r"""
    Compute the histogram of a dataset.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.

        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.
    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). If `density` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.
        Please note that the ``dtype`` of `weights` will also become the
        ``dtype`` of the returned accumulator (`hist`), so it must be
        large enough to hold accumulated values as well.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.

    Returns
    -------
    hist : array
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.  If `weights` are given,
        ``hist.dtype`` will be taken from `weights`.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.


    See Also
    --------
    histogramdd, bincount, searchsorted, digitize, histogram_bin_edges

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words,
    if `bins` is::

      [1, 2, 3, 4]

    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
    the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
    *includes* 4.


    Examples
    --------
    >>> import numpy as np
    >>> np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> np.histogram(np.arange(4), bins=np.arange(5), density=True)
    (array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4]))
    >>> np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))

    >>> a = np.arange(5)
    >>> hist, bin_edges = np.histogram(a, density=True)
    >>> hist
    array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])
    >>> hist.sum()
    2.4999999999999996
    >>> np.sum(hist * np.diff(bin_edges))
    1.0

    Automated Bin Selection Methods example, using 2 peak random data
    with 2000 points.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np

        rng = np.random.RandomState(10)  # deterministic random data
        a = np.hstack((rng.normal(size=1000),
                       rng.normal(loc=5, scale=2, size=1000)))
        plt.hist(a, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()

    """
    a, weights = _ravel_and_check_weights(a, weights)

    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)

    # Histogram is an integer or a float array depending on the weights.
    if weights is None:
        ntype = np.dtype(np.intp)
    else:
        ntype = weights.dtype

    # We set a block size, as this allows us to iterate over chunks when
    # computing histograms, to minimize memory usage.
    BLOCK = 65536

    # The fast path uses bincount, but that only works for certain types
    # of weight
    simple_weights = (
        weights is None or
        np.can_cast(weights.dtype, np.double) or
        np.can_cast(weights.dtype, complex)
    )

    if uniform_bins is not None and simple_weights:
        # Fast algorithm for equal bins
        # We now convert values of a to bin indices, under the assumption of
        # equal bin widths (which is valid here).
        first_edge, last_edge, n_equal_bins = uniform_bins

        # Initialize empty histogram
        n = np.zeros(n_equal_bins, ntype)

        # Pre-compute histogram scaling factor
        norm_numerator = n_equal_bins
        norm_denom = _unsigned_subtract(last_edge, first_edge)

        # We iterate over blocks here for two reasons: the first is that for
        # large arrays, it is actually faster (for example for a 10^8 array it
        # is 2x as fast) and it results in a memory footprint 3x lower in the
        # limit of large arrays.
        for i in _range(0, len(a), BLOCK):
            tmp_a = a[i:i + BLOCK]
            if weights is None:
                tmp_w = None
            else:
                tmp_w = weights[i:i + BLOCK]

            # Only include values in the right range
            keep = (tmp_a >= first_edge)
            keep &= (tmp_a <= last_edge)
            if not np.logical_and.reduce(keep):
                tmp_a = tmp_a[keep]
                if tmp_w is not None:
                    tmp_w = tmp_w[keep]

            # This cast ensures no type promotions occur below, which gh-10322
            # make unpredictable. Getting it wrong leads to precision errors
            # like gh-8123.
            tmp_a = tmp_a.astype(bin_edges.dtype, copy=False)

            # Compute the bin indices, and for values that lie exactly on
            # last_edge we need to subtract one
            f_indices = ((_unsigned_subtract(tmp_a, first_edge) / norm_denom)
                         * norm_numerator)
            indices = f_indices.astype(np.intp)
            indices[indices == n_equal_bins] -= 1

            # The index computation is not guaranteed to give exactly
            # consistent results within ~1 ULP of the bin edges.
            decrement = tmp_a < bin_edges[indices]
            indices[decrement] -= 1
            # The last bin includes the right edge. The other bins do not.
            increment = ((tmp_a >= bin_edges[indices + 1])
                         & (indices != n_equal_bins - 1))
            indices[increment] += 1

            # We now compute the histogram using bincount
            if ntype.kind == 'c':
                n.real += np.bincount(indices, weights=tmp_w.real,
                                      minlength=n_equal_bins)
                n.imag += np.bincount(indices, weights=tmp_w.imag,
                                      minlength=n_equal_bins)
            else:
                n += np.bincount(indices, weights=tmp_w,
                                 minlength=n_equal_bins).astype(ntype)
    else:
        # Compute via cumulative histogram
        cum_n = np.zeros(bin_edges.shape, ntype)
        if weights is None:
            for i in _range(0, len(a), BLOCK):
                sa = np.sort(a[i:i + BLOCK])
                cum_n += _search_sorted_inclusive(sa, bin_edges)
        else:
            zero = np.zeros(1, dtype=ntype)
            for i in _range(0, len(a), BLOCK):
                tmp_a = a[i:i + BLOCK]
                tmp_w = weights[i:i + BLOCK]
                sorting_index = np.argsort(tmp_a)
                sa = tmp_a[sorting_index]
                sw = tmp_w[sorting_index]
                cw = np.concatenate((zero, sw.cumsum()))
                bin_index = _search_sorted_inclusive(sa, bin_edges)
                cum_n += cw[bin_index]

        n = np.diff(cum_n)

    if density:
        db = np.array(np.diff(bin_edges), float)
        return n / db / n.sum(), bin_edges

    return n, bin_edges


def _histogramdd_dispatcher(sample, bins=None, range=None, density=None,
                            weights=None):
    if hasattr(sample, 'shape'):  # same condition as used in histogramdd
        yield sample
    else:
        yield from sample
    with contextlib.suppress(TypeError):
        yield from bins
    yield weights


@array_function_dispatch(_histogramdd_dispatcher)
def histogramdd(sample, bins=10, range=None, density=None, weights=None):
    """
    Compute the multidimensional histogram of some data.

    Parameters
    ----------
    sample : (N, D) array, or (N, D) array_like
        The data to be histogrammed.

        Note the unusual interpretation of sample when an array_like:

        * When an array, each row is a coordinate in a D-dimensional space -
          such as ``histogramdd(np.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single
          coordinate - such as ``histogramdd((X, Y, Z))``.

        The first form should be preferred.

    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
        A sequence of length D, each an optional (lower, upper) tuple giving
        the outer bin edges to be used if the edges are not given explicitly in
        `bins`.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of D None values.
    density : bool, optional
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_volume``.
    weights : (N,) array_like, optional
        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
        Weights are normalized to 1 if density is True. If density is False,
        the values of the returned histogram are equal to the sum of the
        weights belonging to the samples falling into each bin.

    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. See density and weights
        for the different possible semantics.
    edges : tuple of ndarrays
        A tuple of D arrays describing the bin edges for each dimension.

    See Also
    --------
    histogram: 1-D histogram
    histogram2d: 2-D histogram

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> r = rng.normal(size=(100,3))
    >>> H, edges = np.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)

    """

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, np.intp)
    edges = D * [None]
    dedges = D * [None]
    if weights is not None:
        weights = np.asarray(weights)

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                'sample x.')
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    # normalize the range argument
    if range is None:
        range = (None,) * D
    elif len(range) != D:
        raise ValueError('range argument must have one entry per dimension')

    # Create edge arrays
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    f'`bins[{i}]` must be positive, when an integer')
            smin, smax = _get_outer_edges(sample[:, i], range[i])
            try:
                n = operator.index(bins[i])

            except TypeError as e:
                raise TypeError(
                    f"`bins[{i}]` must be an integer, when a scalar"
                ) from e

            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    f'`bins[{i}]` must be monotonically increasing, when an array')
        else:
            raise ValueError(
                f'`bins[{i}]` must be a scalar or 1d array')

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = np.diff(edges[i])

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = np.bincount(xy, weights, minlength=nbin.prod())

    # Shape into a proper matrix
    hist = hist.reshape(nbin)

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float, casting='safe')

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D * (slice(1, -1),)
    hist = hist[core]

    if density:
        # calculate the probability density function
        s = hist.sum()
        for i in _range(D):
            shape = np.ones(D, int)
            shape[i] = nbin[i] - 2
            hist = hist / dedges[i].reshape(shape)
        hist /= s

    if (hist.shape != nbin - 2).any():
        raise RuntimeError(
            "Internal Shape Error")
    return hist, edges

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_arraysetops.py ===
"""Test functions for 1D array set operations.

"""
import pytest

import numpy as np
from numpy import ediff1d, intersect1d, isin, setdiff1d, setxor1d, union1d, unique
from numpy.exceptions import AxisError
from numpy.testing import (
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)


class TestSetOps:

    def test_intersect1d(self):
        # unique inputs
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])

        ec = np.array([1, 2, 5])
        c = intersect1d(a, b, assume_unique=True)
        assert_array_equal(c, ec)

        # non-unique inputs
        a = np.array([5, 5, 7, 1, 2])
        b = np.array([2, 1, 4, 3, 3, 1, 5])

        ed = np.array([1, 2, 5])
        c = intersect1d(a, b)
        assert_array_equal(c, ed)
        assert_array_equal([], intersect1d([], []))

    def test_intersect1d_array_like(self):
        # See gh-11772
        class Test:
            def __array__(self, dtype=None, copy=None):
                return np.arange(3)

        a = Test()
        res = intersect1d(a, a)
        assert_array_equal(res, a)
        res = intersect1d([1, 2, 3], [1, 2, 3])
        assert_array_equal(res, [1, 2, 3])

    def test_intersect1d_indices(self):
        # unique inputs
        a = np.array([1, 2, 3, 4])
        b = np.array([2, 1, 4, 6])
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        ee = np.array([1, 2, 4])
        assert_array_equal(c, ee)
        assert_array_equal(a[i1], ee)
        assert_array_equal(b[i2], ee)

        # non-unique inputs
        a = np.array([1, 2, 2, 3, 4, 3, 2])
        b = np.array([1, 8, 4, 2, 2, 3, 2, 3])
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        ef = np.array([1, 2, 3, 4])
        assert_array_equal(c, ef)
        assert_array_equal(a[i1], ef)
        assert_array_equal(b[i2], ef)

        # non1d, unique inputs
        a = np.array([[2, 4, 5, 6], [7, 8, 1, 15]])
        b = np.array([[3, 2, 7, 6], [10, 12, 8, 9]])
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        ea = np.array([2, 6, 7, 8])
        assert_array_equal(ea, a[ui1])
        assert_array_equal(ea, b[ui2])

        # non1d, not assumed to be uniqueinputs
        a = np.array([[2, 4, 5, 6, 6], [4, 7, 8, 7, 2]])
        b = np.array([[3, 2, 7, 7], [10, 12, 8, 7]])
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        ea = np.array([2, 7, 8])
        assert_array_equal(ea, a[ui1])
        assert_array_equal(ea, b[ui2])

    def test_setxor1d(self):
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])

        ec = np.array([3, 4, 7])
        c = setxor1d(a, b)
        assert_array_equal(c, ec)

        a = np.array([1, 2, 3])
        b = np.array([6, 5, 4])

        ec = np.array([1, 2, 3, 4, 5, 6])
        c = setxor1d(a, b)
        assert_array_equal(c, ec)

        a = np.array([1, 8, 2, 3])
        b = np.array([6, 5, 4, 8])

        ec = np.array([1, 2, 3, 4, 5, 6])
        c = setxor1d(a, b)
        assert_array_equal(c, ec)

        assert_array_equal([], setxor1d([], []))

    def test_setxor1d_unique(self):
        a = np.array([1, 8, 2, 3])
        b = np.array([6, 5, 4, 8])

        ec = np.array([1, 2, 3, 4, 5, 6])
        c = setxor1d(a, b, assume_unique=True)
        assert_array_equal(c, ec)

        a = np.array([[1], [8], [2], [3]])
        b = np.array([[6, 5], [4, 8]])

        ec = np.array([1, 2, 3, 4, 5, 6])
        c = setxor1d(a, b, assume_unique=True)
        assert_array_equal(c, ec)

    def test_ediff1d(self):
        zero_elem = np.array([])
        one_elem = np.array([1])
        two_elem = np.array([1, 2])

        assert_array_equal([], ediff1d(zero_elem))
        assert_array_equal([0], ediff1d(zero_elem, to_begin=0))
        assert_array_equal([0], ediff1d(zero_elem, to_end=0))
        assert_array_equal([-1, 0], ediff1d(zero_elem, to_begin=-1, to_end=0))
        assert_array_equal([], ediff1d(one_elem))
        assert_array_equal([1], ediff1d(two_elem))
        assert_array_equal([7, 1, 9], ediff1d(two_elem, to_begin=7, to_end=9))
        assert_array_equal([5, 6, 1, 7, 8],
                           ediff1d(two_elem, to_begin=[5, 6], to_end=[7, 8]))
        assert_array_equal([1, 9], ediff1d(two_elem, to_end=9))
        assert_array_equal([1, 7, 8], ediff1d(two_elem, to_end=[7, 8]))
        assert_array_equal([7, 1], ediff1d(two_elem, to_begin=7))
        assert_array_equal([5, 6, 1], ediff1d(two_elem, to_begin=[5, 6]))

    @pytest.mark.parametrize("ary, prepend, append, expected", [
        # should fail because trying to cast
        # np.nan standard floating point value
        # into an integer array:
        (np.array([1, 2, 3], dtype=np.int64),
         None,
         np.nan,
         'to_end'),
        # should fail because attempting
        # to downcast to int type:
        (np.array([1, 2, 3], dtype=np.int64),
         np.array([5, 7, 2], dtype=np.float32),
         None,
         'to_begin'),
        # should fail because attempting to cast
        # two special floating point values
        # to integers (on both sides of ary),
        # `to_begin` is in the error message as the impl checks this first:
        (np.array([1., 3., 9.], dtype=np.int8),
         np.nan,
         np.nan,
         'to_begin'),
         ])
    def test_ediff1d_forbidden_type_casts(self, ary, prepend, append, expected):
        # verify resolution of gh-11490

        # specifically, raise an appropriate
        # Exception when attempting to append or
        # prepend with an incompatible type
        msg = f'dtype of `{expected}` must be compatible'
        with assert_raises_regex(TypeError, msg):
            ediff1d(ary=ary,
                    to_end=append,
                    to_begin=prepend)

    @pytest.mark.parametrize(
        "ary,prepend,append,expected",
        [
         (np.array([1, 2, 3], dtype=np.int16),
          2**16,  # will be cast to int16 under same kind rule.
          2**16 + 4,
          np.array([0, 1, 1, 4], dtype=np.int16)),
         (np.array([1, 2, 3], dtype=np.float32),
          np.array([5], dtype=np.float64),
          None,
          np.array([5, 1, 1], dtype=np.float32)),
         (np.array([1, 2, 3], dtype=np.int32),
          0,
          0,
          np.array([0, 1, 1, 0], dtype=np.int32)),
         (np.array([1, 2, 3], dtype=np.int64),
          3,
          -9,
          np.array([3, 1, 1, -9], dtype=np.int64)),
        ]
    )
    def test_ediff1d_scalar_handling(self,
                                     ary,
                                     prepend,
                                     append,
                                     expected):
        # maintain backwards-compatibility
        # of scalar prepend / append behavior
        # in ediff1d following fix for gh-11490
        actual = np.ediff1d(ary=ary,
                            to_end=append,
                            to_begin=prepend)
        assert_equal(actual, expected)
        assert actual.dtype == expected.dtype

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin(self, kind):
        def _isin_slow(a, b):
            b = np.asarray(b).flatten().tolist()
            return a in b
        isin_slow = np.vectorize(_isin_slow, otypes=[bool], excluded={1})

        def assert_isin_equal(a, b):
            x = isin(a, b, kind=kind)
            y = isin_slow(a, b)
            assert_array_equal(x, y)

        # multidimensional arrays in both arguments
        a = np.arange(24).reshape([2, 3, 4])
        b = np.array([[10, 20, 30], [0, 1, 3], [11, 22, 33]])
        assert_isin_equal(a, b)

        # array-likes as both arguments
        c = [(9, 8), (7, 6)]
        d = (9, 7)
        assert_isin_equal(c, d)

        # zero-d array:
        f = np.array(3)
        assert_isin_equal(f, b)
        assert_isin_equal(a, f)
        assert_isin_equal(f, f)

        # scalar:
        assert_isin_equal(5, b)
        assert_isin_equal(a, 6)
        assert_isin_equal(5, 6)

        # empty array-like:
        if kind != "table":
            # An empty list will become float64,
            # which is invalid for kind="table"
            x = []
            assert_isin_equal(x, b)
            assert_isin_equal(a, x)
            assert_isin_equal(x, x)

        # empty array with various types:
        for dtype in [bool, np.int64, np.float64]:
            if kind == "table" and dtype == np.float64:
                continue

            if dtype in {np.int64, np.float64}:
                ar = np.array([10, 20, 30], dtype=dtype)
            elif dtype in {bool}:
                ar = np.array([True, False, False])

            empty_array = np.array([], dtype=dtype)

            assert_isin_equal(empty_array, ar)
            assert_isin_equal(ar, empty_array)
            assert_isin_equal(empty_array, empty_array)

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin_additional(self, kind):
        # we use two different sizes for the b array here to test the
        # two different paths in isin().
        for mult in (1, 10):
            # One check without np.array to make sure lists are handled correct
            a = [5, 7, 1, 2]
            b = [2, 4, 3, 1, 5] * mult
            ec = np.array([True, False, True, True])
            c = isin(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            a[0] = 8
            ec = np.array([False, False, True, True])
            c = isin(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            a[0], a[3] = 4, 8
            ec = np.array([True, False, True, False])
            c = isin(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            b = [2, 3, 4] * mult
            ec = [False, True, False, True, True, True, True, True, True,
                  False, True, False, False, False]
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

            b = b + [5, 5, 4] * mult
            ec = [True, True, True, True, True, True, True, True, True, True,
                  True, False, True, True]
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

            a = np.array([5, 7, 1, 2])
            b = np.array([2, 4, 3, 1, 5] * mult)
            ec = np.array([True, False, True, True])
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

            a = np.array([5, 7, 1, 1, 2])
            b = np.array([2, 4, 3, 3, 1, 5] * mult)
            ec = np.array([True, False, True, True, True])
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

            a = np.array([5, 5])
            b = np.array([2, 2] * mult)
            ec = np.array([False, False])
            c = isin(a, b, kind=kind)
            assert_array_equal(c, ec)

        a = np.array([5])
        b = np.array([2])
        ec = np.array([False])
        c = isin(a, b, kind=kind)
        assert_array_equal(c, ec)

        if kind in {None, "sort"}:
            assert_array_equal(isin([], [], kind=kind), [])

    def test_isin_char_array(self):
        a = np.array(['a', 'b', 'c', 'd', 'e', 'c', 'e', 'b'])
        b = np.array(['a', 'c'])

        ec = np.array([True, False, True, False, False, True, False, False])
        c = isin(a, b)

        assert_array_equal(c, ec)

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin_invert(self, kind):
        "Test isin's invert parameter"
        # We use two different sizes for the b array here to test the
        # two different paths in isin().
        for mult in (1, 10):
            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            b = [2, 3, 4] * mult
            assert_array_equal(np.invert(isin(a, b, kind=kind)),
                               isin(a, b, invert=True, kind=kind))

        # float:
        if kind in {None, "sort"}:
            for mult in (1, 10):
                a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5],
                            dtype=np.float32)
                b = [2, 3, 4] * mult
                b = np.array(b, dtype=np.float32)
                assert_array_equal(np.invert(isin(a, b, kind=kind)),
                                   isin(a, b, invert=True, kind=kind))

    def test_isin_hit_alternate_algorithm(self):
        """Hit the standard isin code with integers"""
        # Need extreme range to hit standard code
        # This hits it without the use of kind='table'
        a = np.array([5, 4, 5, 3, 4, 4, 1e9], dtype=np.int64)
        b = np.array([2, 3, 4, 1e9], dtype=np.int64)
        expected = np.array([0, 1, 0, 1, 1, 1, 1], dtype=bool)
        assert_array_equal(expected, isin(a, b))
        assert_array_equal(np.invert(expected), isin(a, b, invert=True))

        a = np.array([5, 7, 1, 2], dtype=np.int64)
        b = np.array([2, 4, 3, 1, 5, 1e9], dtype=np.int64)
        ec = np.array([True, False, True, True])
        c = isin(a, b, assume_unique=True)
        assert_array_equal(c, ec)

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin_boolean(self, kind):
        """Test that isin works for boolean input"""
        a = np.array([True, False])
        b = np.array([False, False, False])
        expected = np.array([False, True])
        assert_array_equal(expected,
                           isin(a, b, kind=kind))
        assert_array_equal(np.invert(expected),
                           isin(a, b, invert=True, kind=kind))

    @pytest.mark.parametrize("kind", [None, "sort"])
    def test_isin_timedelta(self, kind):
        """Test that isin works for timedelta input"""
        rstate = np.random.RandomState(0)
        a = rstate.randint(0, 100, size=10)
        b = rstate.randint(0, 100, size=10)
        truth = isin(a, b)
        a_timedelta = a.astype("timedelta64[s]")
        b_timedelta = b.astype("timedelta64[s]")
        assert_array_equal(truth, isin(a_timedelta, b_timedelta, kind=kind))

    def test_isin_table_timedelta_fails(self):
        a = np.array([0, 1, 2], dtype="timedelta64[s]")
        b = a
        # Make sure it raises a value error:
        with pytest.raises(ValueError):
            isin(a, b, kind="table")

    @pytest.mark.parametrize(
        "dtype1,dtype2",
        [
            (np.int8, np.int16),
            (np.int16, np.int8),
            (np.uint8, np.uint16),
            (np.uint16, np.uint8),
            (np.uint8, np.int16),
            (np.int16, np.uint8),
            (np.uint64, np.int64),
        ]
    )
    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin_mixed_dtype(self, dtype1, dtype2, kind):
        """Test that isin works as expected for mixed dtype input."""
        is_dtype2_signed = np.issubdtype(dtype2, np.signedinteger)
        ar1 = np.array([0, 0, 1, 1], dtype=dtype1)

        if is_dtype2_signed:
            ar2 = np.array([-128, 0, 127], dtype=dtype2)
        else:
            ar2 = np.array([127, 0, 255], dtype=dtype2)

        expected = np.array([True, True, False, False])

        expect_failure = kind == "table" and (
            dtype1 == np.int16 and dtype2 == np.int8)

        if expect_failure:
            with pytest.raises(RuntimeError, match="exceed the maximum"):
                isin(ar1, ar2, kind=kind)
        else:
            assert_array_equal(isin(ar1, ar2, kind=kind), expected)

    @pytest.mark.parametrize("data", [
        np.array([2**63, 2**63 + 1], dtype=np.uint64),
        np.array([-2**62, -2**62 - 1], dtype=np.int64),
    ])
    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin_mixed_huge_vals(self, kind, data):
        """Test values outside intp range (negative ones if 32bit system)"""
        query = data[1]
        res = np.isin(data, query, kind=kind)
        assert_array_equal(res, [False, True])
        # Also check that nothing weird happens for values can't possibly
        # in range.
        data = data.astype(np.int32)  # clearly different values
        res = np.isin(data, query, kind=kind)
        assert_array_equal(res, [False, False])

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin_mixed_boolean(self, kind):
        """Test that isin works as expected for bool/int input."""
        for dtype in np.typecodes["AllInteger"]:
            a = np.array([True, False, False], dtype=bool)
            b = np.array([0, 0, 0, 0], dtype=dtype)
            expected = np.array([False, True, True], dtype=bool)
            assert_array_equal(isin(a, b, kind=kind), expected)

            a, b = b, a
            expected = np.array([True, True, True, True], dtype=bool)
            assert_array_equal(isin(a, b, kind=kind), expected)

    def test_isin_first_array_is_object(self):
        ar1 = [None]
        ar2 = np.array([1] * 10)
        expected = np.array([False])
        result = np.isin(ar1, ar2)
        assert_array_equal(result, expected)

    def test_isin_second_array_is_object(self):
        ar1 = 1
        ar2 = np.array([None] * 10)
        expected = np.array([False])
        result = np.isin(ar1, ar2)
        assert_array_equal(result, expected)

    def test_isin_both_arrays_are_object(self):
        ar1 = [None]
        ar2 = np.array([None] * 10)
        expected = np.array([True])
        result = np.isin(ar1, ar2)
        assert_array_equal(result, expected)

    def test_isin_both_arrays_have_structured_dtype(self):
        # Test arrays of a structured data type containing an integer field
        # and a field of dtype `object` allowing for arbitrary Python objects
        dt = np.dtype([('field1', int), ('field2', object)])
        ar1 = np.array([(1, None)], dtype=dt)
        ar2 = np.array([(1, None)] * 10, dtype=dt)
        expected = np.array([True])
        result = np.isin(ar1, ar2)
        assert_array_equal(result, expected)

    def test_isin_with_arrays_containing_tuples(self):
        ar1 = np.array([(1,), 2], dtype=object)
        ar2 = np.array([(1,), 2], dtype=object)
        expected = np.array([True, True])
        result = np.isin(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.isin(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))

        # An integer is added at the end of the array to make sure
        # that the array builder will create the array with tuples
        # and after it's created the integer is removed.
        # There's a bug in the array constructor that doesn't handle
        # tuples properly and adding the integer fixes that.
        ar1 = np.array([(1,), (2, 1), 1], dtype=object)
        ar1 = ar1[:-1]
        ar2 = np.array([(1,), (2, 1), 1], dtype=object)
        ar2 = ar2[:-1]
        expected = np.array([True, True])
        result = np.isin(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.isin(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))

        ar1 = np.array([(1,), (2, 3), 1], dtype=object)
        ar1 = ar1[:-1]
        ar2 = np.array([(1,), 2], dtype=object)
        expected = np.array([True, False])
        result = np.isin(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.isin(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))

    def test_isin_errors(self):
        """Test that isin raises expected errors."""

        # Error 1: `kind` is not one of 'sort' 'table' or None.
        ar1 = np.array([1, 2, 3, 4, 5])
        ar2 = np.array([2, 4, 6, 8, 10])
        assert_raises(ValueError, isin, ar1, ar2, kind='quicksort')

        # Error 2: `kind="table"` does not work for non-integral arrays.
        obj_ar1 = np.array([1, 'a', 3, 'b', 5], dtype=object)
        obj_ar2 = np.array([1, 'a', 3, 'b', 5], dtype=object)
        assert_raises(ValueError, isin, obj_ar1, obj_ar2, kind='table')

        for dtype in [np.int32, np.int64]:
            ar1 = np.array([-1, 2, 3, 4, 5], dtype=dtype)
            # The range of this array will overflow:
            overflow_ar2 = np.array([-1, np.iinfo(dtype).max], dtype=dtype)

            # Error 3: `kind="table"` will trigger a runtime error
            #  if there is an integer overflow expected when computing the
            #  range of ar2
            assert_raises(
                RuntimeError,
                isin, ar1, overflow_ar2, kind='table'
            )

            # Non-error: `kind=None` will *not* trigger a runtime error
            #  if there is an integer overflow, it will switch to
            #  the `sort` algorithm.
            result = np.isin(ar1, overflow_ar2, kind=None)
            assert_array_equal(result, [True] + [False] * 4)
            result = np.isin(ar1, overflow_ar2, kind='sort')
            assert_array_equal(result, [True] + [False] * 4)

    def test_union1d(self):
        a = np.array([5, 4, 7, 1, 2])
        b = np.array([2, 4, 3, 3, 2, 1, 5])

        ec = np.array([1, 2, 3, 4, 5, 7])
        c = union1d(a, b)
        assert_array_equal(c, ec)

        # Tests gh-10340, arguments to union1d should be
        # flattened if they are not already 1D
        x = np.array([[0, 1, 2], [3, 4, 5]])
        y = np.array([0, 1, 2, 3, 4])
        ez = np.array([0, 1, 2, 3, 4, 5])
        z = union1d(x, y)
        assert_array_equal(z, ez)

        assert_array_equal([], union1d([], []))

    def test_setdiff1d(self):
        a = np.array([6, 5, 4, 7, 1, 2, 7, 4])
        b = np.array([2, 4, 3, 3, 2, 1, 5])

        ec = np.array([6, 7])
        c = setdiff1d(a, b)
        assert_array_equal(c, ec)

        a = np.arange(21)
        b = np.arange(19)
        ec = np.array([19, 20])
        c = setdiff1d(a, b)
        assert_array_equal(c, ec)

        assert_array_equal([], setdiff1d([], []))
        a = np.array((), np.uint32)
        assert_equal(setdiff1d(a, []).dtype, np.uint32)

    def test_setdiff1d_unique(self):
        a = np.array([3, 2, 1])
        b = np.array([7, 5, 2])
        expected = np.array([3, 1])
        actual = setdiff1d(a, b, assume_unique=True)
        assert_equal(actual, expected)

    def test_setdiff1d_char_array(self):
        a = np.array(['a', 'b', 'c'])
        b = np.array(['a', 'b', 's'])
        assert_array_equal(setdiff1d(a, b), np.array(['c']))

    def test_manyways(self):
        a = np.array([5, 7, 1, 2, 8])
        b = np.array([9, 8, 2, 4, 3, 1, 5])

        c1 = setxor1d(a, b)
        aux1 = intersect1d(a, b)
        aux2 = union1d(a, b)
        c2 = setdiff1d(aux2, aux1)
        assert_array_equal(c1, c2)


class TestUnique:

    def check_all(self, a, b, i1, i2, c, dt):
        base_msg = 'check {0} failed for type {1}'

        msg = base_msg.format('values', dt)
        v = unique(a)
        assert_array_equal(v, b, msg)
        assert type(v) == type(b)

        msg = base_msg.format('return_index', dt)
        v, j = unique(a, True, False, False)
        assert_array_equal(v, b, msg)
        assert_array_equal(j, i1, msg)
        assert type(v) == type(b)

        msg = base_msg.format('return_inverse', dt)
        v, j = unique(a, False, True, False)
        assert_array_equal(v, b, msg)
        assert_array_equal(j, i2, msg)
        assert type(v) == type(b)

        msg = base_msg.format('return_counts', dt)
        v, j = unique(a, False, False, True)
        assert_array_equal(v, b, msg)
        assert_array_equal(j, c, msg)
        assert type(v) == type(b)

        msg = base_msg.format('return_index and return_inverse', dt)
        v, j1, j2 = unique(a, True, True, False)
        assert_array_equal(v, b, msg)
        assert_array_equal(j1, i1, msg)
        assert_array_equal(j2, i2, msg)
        assert type(v) == type(b)

        msg = base_msg.format('return_index and return_counts', dt)
        v, j1, j2 = unique(a, True, False, True)
        assert_array_equal(v, b, msg)
        assert_array_equal(j1, i1, msg)
        assert_array_equal(j2, c, msg)
        assert type(v) == type(b)

        msg = base_msg.format('return_inverse and return_counts', dt)
        v, j1, j2 = unique(a, False, True, True)
        assert_array_equal(v, b, msg)
        assert_array_equal(j1, i2, msg)
        assert_array_equal(j2, c, msg)
        assert type(v) == type(b)

        msg = base_msg.format(('return_index, return_inverse '
                                'and return_counts'), dt)
        v, j1, j2, j3 = unique(a, True, True, True)
        assert_array_equal(v, b, msg)
        assert_array_equal(j1, i1, msg)
        assert_array_equal(j2, i2, msg)
        assert_array_equal(j3, c, msg)
        assert type(v) == type(b)

    def get_types(self):
        types = []
        types.extend(np.typecodes['AllInteger'])
        types.extend(np.typecodes['AllFloat'])
        types.append('datetime64[D]')
        types.append('timedelta64[D]')
        return types

    def test_unique_1d(self):

        a = [5, 7, 1, 2, 1, 5, 7] * 10
        b = [1, 2, 5, 7]
        i1 = [2, 3, 0, 1]
        i2 = [2, 3, 0, 1, 0, 2, 3] * 10
        c = np.multiply([2, 1, 2, 2], 10)

        # test for numeric arrays
        types = self.get_types()
        for dt in types:
            aa = np.array(a, dt)
            bb = np.array(b, dt)
            self.check_all(aa, bb, i1, i2, c, dt)

        # test for object arrays
        dt = 'O'
        aa = np.empty(len(a), dt)
        aa[:] = a
        bb = np.empty(len(b), dt)
        bb[:] = b
        self.check_all(aa, bb, i1, i2, c, dt)

        # test for structured arrays
        dt = [('', 'i'), ('', 'i')]
        aa = np.array(list(zip(a, a)), dt)
        bb = np.array(list(zip(b, b)), dt)
        self.check_all(aa, bb, i1, i2, c, dt)

        # test for ticket #2799
        aa = [1. + 0.j, 1 - 1.j, 1]
        assert_array_equal(np.unique(aa), [1. - 1.j, 1. + 0.j])

        # test for ticket #4785
        a = [(1, 2), (1, 2), (2, 3)]
        unq = [1, 2, 3]
        inv = [[0, 1], [0, 1], [1, 2]]
        a1 = unique(a)
        assert_array_equal(a1, unq)
        a2, a2_inv = unique(a, return_inverse=True)
        assert_array_equal(a2, unq)
        assert_array_equal(a2_inv, inv)

        # test for chararrays with return_inverse (gh-5099)
        a = np.char.chararray(5)
        a[...] = ''
        a2, a2_inv = np.unique(a, return_inverse=True)
        assert_array_equal(a2_inv, np.zeros(5))

        # test for ticket #9137
        a = []
        a1_idx = np.unique(a, return_index=True)[1]
        a2_inv = np.unique(a, return_inverse=True)[1]
        a3_idx, a3_inv = np.unique(a, return_index=True,
                                   return_inverse=True)[1:]
        assert_equal(a1_idx.dtype, np.intp)
        assert_equal(a2_inv.dtype, np.intp)
        assert_equal(a3_idx.dtype, np.intp)
        assert_equal(a3_inv.dtype, np.intp)

        # test for ticket 2111 - float
        a = [2.0, np.nan, 1.0, np.nan]
        ua = [1.0, 2.0, np.nan]
        ua_idx = [2, 0, 1]
        ua_inv = [1, 2, 0, 2]
        ua_cnt = [1, 1, 2]
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for ticket 2111 - complex
        a = [2.0 - 1j, np.nan, 1.0 + 1j, complex(0.0, np.nan), complex(1.0, np.nan)]
        ua = [1.0 + 1j, 2.0 - 1j, complex(0.0, np.nan)]
        ua_idx = [2, 0, 3]
        ua_inv = [1, 2, 0, 2, 2]
        ua_cnt = [1, 1, 3]
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for ticket 2111 - datetime64
        nat = np.datetime64('nat')
        a = [np.datetime64('2020-12-26'), nat, np.datetime64('2020-12-24'), nat]
        ua = [np.datetime64('2020-12-24'), np.datetime64('2020-12-26'), nat]
        ua_idx = [2, 0, 1]
        ua_inv = [1, 2, 0, 2]
        ua_cnt = [1, 1, 2]
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for ticket 2111 - timedelta
        nat = np.timedelta64('nat')
        a = [np.timedelta64(1, 'D'), nat, np.timedelta64(1, 'h'), nat]
        ua = [np.timedelta64(1, 'h'), np.timedelta64(1, 'D'), nat]
        ua_idx = [2, 0, 1]
        ua_inv = [1, 2, 0, 2]
        ua_cnt = [1, 1, 2]
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for gh-19300
        all_nans = [np.nan] * 4
        ua = [np.nan]
        ua_idx = [0]
        ua_inv = [0, 0, 0, 0]
        ua_cnt = [4]
        assert_equal(np.unique(all_nans), ua)
        assert_equal(np.unique(all_nans, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(all_nans, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(all_nans, return_counts=True), (ua, ua_cnt))

    def test_unique_zero_sized(self):
        # test for zero-sized arrays
        for dt in self.get_types():
            a = np.array([], dt)
            b = np.array([], dt)
            i1 = np.array([], np.int64)
            i2 = np.array([], np.int64)
            c = np.array([], np.int64)
            self.check_all(a, b, i1, i2, c, dt)

    def test_unique_subclass(self):
        class Subclass(np.ndarray):
            pass

        i1 = [2, 3, 0, 1]
        i2 = [2, 3, 0, 1, 0, 2, 3] * 10
        c = np.multiply([2, 1, 2, 2], 10)

        # test for numeric arrays
        types = self.get_types()
        for dt in types:
            a = np.array([5, 7, 1, 2, 1, 5, 7] * 10, dtype=dt)
            b = np.array([1, 2, 5, 7], dtype=dt)
            aa = Subclass(a.shape, dtype=dt, buffer=a)
            bb = Subclass(b.shape, dtype=dt, buffer=b)
            self.check_all(aa, bb, i1, i2, c, dt)

    @pytest.mark.parametrize("arg", ["return_index", "return_inverse", "return_counts"])
    def test_unsupported_hash_based(self, arg):
        """These currently never use the hash-based solution.  However,
        it seems easier to just allow it.

        When the hash-based solution is added, this test should fail and be
        replaced with something more comprehensive.
        """
        a = np.array([1, 5, 2, 3, 4, 8, 199, 1, 3, 5])

        res_not_sorted = np.unique([1, 1], sorted=False, **{arg: True})
        res_sorted = np.unique([1, 1], sorted=True, **{arg: True})
        # The following should fail without first sorting `res_not_sorted`.
        for arr, expected in zip(res_not_sorted, res_sorted):
            assert_array_equal(arr, expected)

    def test_unique_axis_errors(self):
        assert_raises(TypeError, self._run_axis_tests, object)
        assert_raises(TypeError, self._run_axis_tests,
                      [('a', int), ('b', object)])

        assert_raises(AxisError, unique, np.arange(10), axis=2)
        assert_raises(AxisError, unique, np.arange(10), axis=-2)

    def test_unique_axis_list(self):
        msg = "Unique failed on list of lists"
        inp = [[0, 1, 0], [0, 1, 0]]
        inp_arr = np.asarray(inp)
        assert_array_equal(unique(inp, axis=0), unique(inp_arr, axis=0), msg)
        assert_array_equal(unique(inp, axis=1), unique(inp_arr, axis=1), msg)

    def test_unique_axis(self):
        types = []
        types.extend(np.typecodes['AllInteger'])
        types.extend(np.typecodes['AllFloat'])
        types.append('datetime64[D]')
        types.append('timedelta64[D]')
        types.append([('a', int), ('b', int)])
        types.append([('a', int), ('b', float)])

        for dtype in types:
            self._run_axis_tests(dtype)

        msg = 'Non-bitwise-equal booleans test failed'
        data = np.arange(10, dtype=np.uint8).reshape(-1, 2).view(bool)
        result = np.array([[False, True], [True, True]], dtype=bool)
        assert_array_equal(unique(data, axis=0), result, msg)

        msg = 'Negative zero equality test failed'
        data = np.array([[-0.0, 0.0], [0.0, -0.0], [-0.0, 0.0], [0.0, -0.0]])
        result = np.array([[-0.0, 0.0]])
        assert_array_equal(unique(data, axis=0), result, msg)

    @pytest.mark.parametrize("axis", [0, -1])
    def test_unique_1d_with_axis(self, axis):
        x = np.array([4, 3, 2, 3, 2, 1, 2, 2])
        uniq = unique(x, axis=axis)
        assert_array_equal(uniq, [1, 2, 3, 4])

    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_unique_inverse_with_axis(self, axis):
        x = np.array([[4, 4, 3], [2, 2, 1], [2, 2, 1], [4, 4, 3]])
        uniq, inv = unique(x, return_inverse=True, axis=axis)
        assert_equal(inv.ndim, x.ndim if axis is None else 1)
        assert_array_equal(x, np.take(uniq, inv, axis=axis))

    def test_unique_axis_zeros(self):
        # issue 15559
        single_zero = np.empty(shape=(2, 0), dtype=np.int8)
        uniq, idx, inv, cnt = unique(single_zero, axis=0, return_index=True,
                                     return_inverse=True, return_counts=True)

        # there's 1 element of shape (0,) along axis 0
        assert_equal(uniq.dtype, single_zero.dtype)
        assert_array_equal(uniq, np.empty(shape=(1, 0)))
        assert_array_equal(idx, np.array([0]))
        assert_array_equal(inv, np.array([0, 0]))
        assert_array_equal(cnt, np.array([2]))

        # there's 0 elements of shape (2,) along axis 1
        uniq, idx, inv, cnt = unique(single_zero, axis=1, return_index=True,
                                     return_inverse=True, return_counts=True)

        assert_equal(uniq.dtype, single_zero.dtype)
        assert_array_equal(uniq, np.empty(shape=(2, 0)))
        assert_array_equal(idx, np.array([]))
        assert_array_equal(inv, np.array([]))
        assert_array_equal(cnt, np.array([]))

        # test a "complicated" shape
        shape = (0, 2, 0, 3, 0, 4, 0)
        multiple_zeros = np.empty(shape=shape)
        for axis in range(len(shape)):
            expected_shape = list(shape)
            if shape[axis] == 0:
                expected_shape[axis] = 0
            else:
                expected_shape[axis] = 1

            assert_array_equal(unique(multiple_zeros, axis=axis),
                               np.empty(shape=expected_shape))

    def test_unique_masked(self):
        # issue 8664
        x = np.array([64, 0, 1, 2, 3, 63, 63, 0, 0, 0, 1, 2, 0, 63, 0],
                     dtype='uint8')
        y = np.ma.masked_equal(x, 0)

        v = np.unique(y)
        v2, i, c = np.unique(y, return_index=True, return_counts=True)

        msg = 'Unique returned different results when asked for index'
        assert_array_equal(v.data, v2.data, msg)
        assert_array_equal(v.mask, v2.mask, msg)

    def test_unique_sort_order_with_axis(self):
        # These tests fail if sorting along axis is done by treating subarrays
        # as unsigned byte strings.  See gh-10495.
        fmt = "sort order incorrect for integer type '%s'"
        for dt in 'bhilq':
            a = np.array([[-1], [0]], dt)
            b = np.unique(a, axis=0)
            assert_array_equal(a, b, fmt % dt)

    def _run_axis_tests(self, dtype):
        data = np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [1, 0, 0, 0]]).astype(dtype)

        msg = 'Unique with 1d array and axis=0 failed'
        result = np.array([0, 1])
        assert_array_equal(unique(data), result.astype(dtype), msg)

        msg = 'Unique with 2d array and axis=0 failed'
        result = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
        assert_array_equal(unique(data, axis=0), result.astype(dtype), msg)

        msg = 'Unique with 2d array and axis=1 failed'
        result = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        assert_array_equal(unique(data, axis=1), result.astype(dtype), msg)

        msg = 'Unique with 3d array and axis=2 failed'
        data3d = np.array([[[1, 1],
                            [1, 0]],
                           [[0, 1],
                            [0, 0]]]).astype(dtype)
        result = np.take(data3d, [1, 0], axis=2)
        assert_array_equal(unique(data3d, axis=2), result, msg)

        uniq, idx, inv, cnt = unique(data, axis=0, return_index=True,
                                     return_inverse=True, return_counts=True)
        msg = "Unique's return_index=True failed with axis=0"
        assert_array_equal(data[idx], uniq, msg)
        msg = "Unique's return_inverse=True failed with axis=0"
        assert_array_equal(np.take(uniq, inv, axis=0), data)
        msg = "Unique's return_counts=True failed with axis=0"
        assert_array_equal(cnt, np.array([2, 2]), msg)

        uniq, idx, inv, cnt = unique(data, axis=1, return_index=True,
                                     return_inverse=True, return_counts=True)
        msg = "Unique's return_index=True failed with axis=1"
        assert_array_equal(data[:, idx], uniq)
        msg = "Unique's return_inverse=True failed with axis=1"
        assert_array_equal(np.take(uniq, inv, axis=1), data)
        msg = "Unique's return_counts=True failed with axis=1"
        assert_array_equal(cnt, np.array([2, 1, 1]), msg)

    def test_unique_nanequals(self):
        # issue 20326
        a = np.array([1, 1, np.nan, np.nan, np.nan])
        unq = np.unique(a)
        not_unq = np.unique(a, equal_nan=False)
        assert_array_equal(unq, np.array([1, np.nan]))
        assert_array_equal(not_unq, np.array([1, np.nan, np.nan, np.nan]))

    def test_unique_array_api_functions(self):
        arr = np.array([np.nan, 1, 4, 1, 3, 4, np.nan, 5, 1])

        for res_unique_array_api, res_unique in [
            (
                np.unique_values(arr),
                np.unique(arr, equal_nan=False)
            ),
            (
                np.unique_counts(arr),
                np.unique(arr, return_counts=True, equal_nan=False)
            ),
            (
                np.unique_inverse(arr),
                np.unique(arr, return_inverse=True, equal_nan=False)
            ),
            (
                np.unique_all(arr),
                np.unique(
                    arr,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                    equal_nan=False
                )
            )
        ]:
            assert len(res_unique_array_api) == len(res_unique)
            for actual, expected in zip(res_unique_array_api, res_unique):
                assert_array_equal(actual, expected)

    def test_unique_inverse_shape(self):
        # Regression test for https://github.com/numpy/numpy/issues/25552
        arr = np.array([[1, 2, 3], [2, 3, 1]])
        expected_values, expected_inverse = np.unique(arr, return_inverse=True)
        expected_inverse = expected_inverse.reshape(arr.shape)
        for func in np.unique_inverse, np.unique_all:
            result = func(arr)
            assert_array_equal(expected_values, result.values)
            assert_array_equal(expected_inverse, result.inverse_indices)
            assert_array_equal(arr, result.values[result.inverse_indices])

    @pytest.mark.parametrize(
        'data',
        [[[1, 1, 1],
          [1, 1, 1]],
         [1, 3, 2],
         1],
    )
    @pytest.mark.parametrize('transpose', [False, True])
    @pytest.mark.parametrize('dtype', [np.int32, np.float64])
    def test_unique_with_matrix(self, data, transpose, dtype):
        mat = np.matrix(data).astype(dtype)
        if transpose:
            mat = mat.T
        u = np.unique(mat)
        expected = np.unique(np.asarray(mat))
        assert_array_equal(u, expected, strict=True)