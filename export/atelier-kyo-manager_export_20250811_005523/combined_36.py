
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\categorical.py ===
from __future__ import annotations

from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    NaT,
    algos as libalgos,
    lib,
)
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.cast import (
    coerce_indexer_dtype,
    find_common_type,
)
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
    is_any_real_numeric_dtype,
    is_bool_dtype,
    is_dict_like,
    is_hashable,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    CategoricalDtypeType,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
)

from pandas.core import (
    algorithms,
    arraylike,
    ops,
)
from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.algorithms import (
    factorize,
    take_nd,
)
from pandas.core.arrays._mixins import (
    NDArrayBackedExtensionArray,
    ravel_compat,
)
from pandas.core.base import (
    ExtensionArray,
    NoNewAttributesMixin,
    PandasObject,
)
import pandas.core.common as com
from pandas.core.construction import (
    extract_array,
    sanitize_array,
)
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin

from pandas.io.formats import console

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        AxisInt,
        Dtype,
        DtypeObj,
        NpDtype,
        Ordered,
        Self,
        Shape,
        SortKind,
        npt,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )


def _cat_compare_op(op):
    opname = f"__{op.__name__}__"
    fill_value = op is operator.ne

    @unpack_zerodim_and_defer(opname)
    def func(self, other):
        hashable = is_hashable(other)
        if is_list_like(other) and len(other) != len(self) and not hashable:
            # in hashable case we may have a tuple that is itself a category
            raise ValueError("Lengths must match.")

        if not self.ordered:
            if opname in ["__lt__", "__gt__", "__le__", "__ge__"]:
                raise TypeError(
                    "Unordered Categoricals can only compare equality or not"
                )
        if isinstance(other, Categorical):
            # Two Categoricals can only be compared if the categories are
            # the same (maybe up to ordering, depending on ordered)

            msg = "Categoricals can only be compared if 'categories' are the same."
            if not self._categories_match_up_to_permutation(other):
                raise TypeError(msg)

            if not self.ordered and not self.categories.equals(other.categories):
                # both unordered and different order
                other_codes = recode_for_categories(
                    other.codes, other.categories, self.categories, copy=False
                )
            else:
                other_codes = other._codes

            ret = op(self._codes, other_codes)
            mask = (self._codes == -1) | (other_codes == -1)
            if mask.any():
                ret[mask] = fill_value
            return ret

        if hashable:
            if other in self.categories:
                i = self._unbox_scalar(other)
                ret = op(self._codes, i)

                if opname not in {"__eq__", "__ge__", "__gt__"}:
                    # GH#29820 performance trick; get_loc will always give i>=0,
                    #  so in the cases (__ne__, __le__, __lt__) the setting
                    #  here is a no-op, so can be skipped.
                    mask = self._codes == -1
                    ret[mask] = fill_value
                return ret
            else:
                return ops.invalid_comparison(self, other, op)
        else:
            # allow categorical vs object dtype array comparisons for equality
            # these are only positional comparisons
            if opname not in ["__eq__", "__ne__"]:
                raise TypeError(
                    f"Cannot compare a Categorical for op {opname} with "
                    f"type {type(other)}.\nIf you want to compare values, "
                    "use 'np.asarray(cat) <op> other'."
                )

            if isinstance(other, ExtensionArray) and needs_i8_conversion(other.dtype):
                # We would return NotImplemented here, but that messes up
                #  ExtensionIndex's wrapped methods
                return op(other, self)
            return getattr(np.array(self), opname)(np.array(other))

    func.__name__ = opname

    return func


def contains(cat, key, container) -> bool:
    """
    Helper for membership check for ``key`` in ``cat``.

    This is a helper method for :method:`__contains__`
    and :class:`CategoricalIndex.__contains__`.

    Returns True if ``key`` is in ``cat.categories`` and the
    location of ``key`` in ``categories`` is in ``container``.

    Parameters
    ----------
    cat : :class:`Categorical`or :class:`categoricalIndex`
    key : a hashable object
        The key to check membership for.
    container : Container (e.g. list-like or mapping)
        The container to check for membership in.

    Returns
    -------
    is_in : bool
        True if ``key`` is in ``self.categories`` and location of
        ``key`` in ``categories`` is in ``container``, else False.

    Notes
    -----
    This method does not check for NaN values. Do that separately
    before calling this method.
    """
    hash(key)

    # get location of key in categories.
    # If a KeyError, the key isn't in categories, so logically
    #  can't be in container either.
    try:
        loc = cat.categories.get_loc(key)
    except (KeyError, TypeError):
        return False

    # loc is the location of key in categories, but also the *value*
    # for key in container. So, `key` may be in categories,
    # but still not in `container`. Example ('b' in categories,
    # but not in values):
    # 'b' in Categorical(['a'], categories=['a', 'b'])  # False
    if is_scalar(loc):
        return loc in container
    else:
        # if categories is an IntervalIndex, loc is an array.
        return any(loc_ in container for loc_ in loc)


class Categorical(NDArrayBackedExtensionArray, PandasObject, ObjectStringArrayMixin):
    """
    Represent a categorical variable in classic R / S-plus fashion.

    `Categoricals` can only take on a limited, and usually fixed, number
    of possible values (`categories`). In contrast to statistical categorical
    variables, a `Categorical` might have an order, but numerical operations
    (additions, divisions, ...) are not possible.

    All values of the `Categorical` are either in `categories` or `np.nan`.
    Assigning values outside of `categories` will raise a `ValueError`. Order
    is defined by the order of the `categories`, not lexical order of the
    values.

    Parameters
    ----------
    values : list-like
        The values of the categorical. If categories are given, values not in
        categories will be replaced with NaN.
    categories : Index-like (unique), optional
        The unique categories for this categorical. If not given, the
        categories are assumed to be the unique values of `values` (sorted, if
        possible, otherwise in the order in which they appear).
    ordered : bool, default False
        Whether or not this categorical is treated as a ordered categorical.
        If True, the resulting categorical will be ordered.
        An ordered categorical respects, when sorted, the order of its
        `categories` attribute (which in turn is the `categories` argument, if
        provided).
    dtype : CategoricalDtype
        An instance of ``CategoricalDtype`` to use for this categorical.

    Attributes
    ----------
    categories : Index
        The categories of this categorical.
    codes : ndarray
        The codes (integer positions, which point to the categories) of this
        categorical, read only.
    ordered : bool
        Whether or not this Categorical is ordered.
    dtype : CategoricalDtype
        The instance of ``CategoricalDtype`` storing the ``categories``
        and ``ordered``.

    Methods
    -------
    from_codes
    __array__

    Raises
    ------
    ValueError
        If the categories do not validate.
    TypeError
        If an explicit ``ordered=True`` is given but no `categories` and the
        `values` are not sortable.

    See Also
    --------
    CategoricalDtype : Type for categorical data.
    CategoricalIndex : An Index with an underlying ``Categorical``.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`__
    for more.

    Examples
    --------
    >>> pd.Categorical([1, 2, 3, 1, 2, 3])
    [1, 2, 3, 1, 2, 3]
    Categories (3, int64): [1, 2, 3]

    >>> pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Missing values are not included as a category.

    >>> c = pd.Categorical([1, 2, 3, 1, 2, 3, np.nan])
    >>> c
    [1, 2, 3, 1, 2, 3, NaN]
    Categories (3, int64): [1, 2, 3]

    However, their presence is indicated in the `codes` attribute
    by code `-1`.

    >>> c.codes
    array([ 0,  1,  2,  0,  1,  2, -1], dtype=int8)

    Ordered `Categoricals` can be sorted according to the custom order
    of the categories and can have a min and max value.

    >>> c = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ordered=True,
    ...                    categories=['c', 'b', 'a'])
    >>> c
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['c' < 'b' < 'a']
    >>> c.min()
    'c'
    """

    # For comparisons, so that numpy uses our implementation if the compare
    # ops, which raise
    __array_priority__ = 1000
    # tolist is not actually deprecated, just suppressed in the __dir__
    _hidden_attrs = PandasObject._hidden_attrs | frozenset(["tolist"])
    _typ = "categorical"

    _dtype: CategoricalDtype

    @classmethod
    # error: Argument 2 of "_simple_new" is incompatible with supertype
    # "NDArrayBacked"; supertype defines the argument type as
    # "Union[dtype[Any], ExtensionDtype]"
    def _simple_new(  # type: ignore[override]
        cls, codes: np.ndarray, dtype: CategoricalDtype
    ) -> Self:
        # NB: This is not _quite_ as simple as the "usual" _simple_new
        codes = coerce_indexer_dtype(codes, dtype.categories)
        dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
        return super()._simple_new(codes, dtype)

    def __init__(
        self,
        values,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        fastpath: bool | lib.NoDefault = lib.no_default,
        copy: bool = True,
    ) -> None:
        if fastpath is not lib.no_default:
            # GH#20110
            warnings.warn(
                "The 'fastpath' keyword in Categorical is deprecated and will "
                "be removed in a future version. Use Categorical.from_codes instead",
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
        else:
            fastpath = False

        dtype = CategoricalDtype._from_values_or_dtype(
            values, categories, ordered, dtype
        )
        # At this point, dtype is always a CategoricalDtype, but
        # we may have dtype.categories be None, and we need to
        # infer categories in a factorization step further below

        if fastpath:
            codes = coerce_indexer_dtype(values, dtype.categories)
            dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
            super().__init__(codes, dtype)
            return

        if not is_list_like(values):
            # GH#38433
            raise TypeError("Categorical input must be list-like")

        # null_mask indicates missing values we want to exclude from inference.
        # This means: only missing values in list-likes (not arrays/ndframes).
        null_mask = np.array(False)

        # sanitize input
        vdtype = getattr(values, "dtype", None)
        if isinstance(vdtype, CategoricalDtype):
            if dtype.categories is None:
                dtype = CategoricalDtype(values.categories, dtype.ordered)
        elif not isinstance(values, (ABCIndex, ABCSeries, ExtensionArray)):
            values = com.convert_to_list_like(values)
            if isinstance(values, list) and len(values) == 0:
                # By convention, empty lists result in object dtype:
                values = np.array([], dtype=object)
            elif isinstance(values, np.ndarray):
                if values.ndim > 1:
                    # preempt sanitize_array from raising ValueError
                    raise NotImplementedError(
                        "> 1 ndim Categorical are not supported at this time"
                    )
                values = sanitize_array(values, None)
            else:
                # i.e. must be a list
                arr = sanitize_array(values, None)
                null_mask = isna(arr)
                if null_mask.any():
                    # We remove null values here, then below will re-insert
                    #  them, grep "full_codes"
                    arr_list = [values[idx] for idx in np.where(~null_mask)[0]]

                    # GH#44900 Do not cast to float if we have only missing values
                    if arr_list or arr.dtype == "object":
                        sanitize_dtype = None
                    else:
                        sanitize_dtype = arr.dtype

                    arr = sanitize_array(arr_list, None, dtype=sanitize_dtype)
                values = arr

        if dtype.categories is None:
            if isinstance(values.dtype, ArrowDtype) and issubclass(
                values.dtype.type, CategoricalDtypeType
            ):
                arr = values._pa_array.combine_chunks()
                categories = arr.dictionary.to_pandas(types_mapper=ArrowDtype)
                codes = arr.indices.to_numpy()
                dtype = CategoricalDtype(categories, values.dtype.pyarrow_dtype.ordered)
            else:
                if not isinstance(values, ABCIndex):
                    # in particular RangeIndex xref test_index_equal_range_categories
                    values = sanitize_array(values, None)
                try:
                    codes, categories = factorize(values, sort=True)
                except TypeError as err:
                    codes, categories = factorize(values, sort=False)
                    if dtype.ordered:
                        # raise, as we don't have a sortable data structure and so
                        # the user should give us one by specifying categories
                        raise TypeError(
                            "'values' is not ordered, please "
                            "explicitly specify the categories order "
                            "by passing in a categories argument."
                        ) from err

                # we're inferring from values
                dtype = CategoricalDtype(categories, dtype.ordered)

        elif isinstance(values.dtype, CategoricalDtype):
            old_codes = extract_array(values)._codes
            codes = recode_for_categories(
                old_codes, values.dtype.categories, dtype.categories, copy=copy
            )

        else:
            codes = _get_codes_for_values(values, dtype.categories)

        if null_mask.any():
            # Reinsert -1 placeholders for previously removed missing values
            full_codes = -np.ones(null_mask.shape, dtype=codes.dtype)
            full_codes[~null_mask] = codes
            codes = full_codes

        dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
        arr = coerce_indexer_dtype(codes, dtype.categories)
        super().__init__(arr, dtype)

    @property
    def dtype(self) -> CategoricalDtype:
        """
        The :class:`~pandas.api.types.CategoricalDtype` for this instance.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat
        ['a', 'b']
        Categories (2, object): ['a' < 'b']
        >>> cat.dtype
        CategoricalDtype(categories=['a', 'b'], ordered=True, categories_dtype=object)
        """
        return self._dtype

    @property
    def _internal_fill_value(self) -> int:
        # using the specific numpy integer instead of python int to get
        #  the correct dtype back from _quantile in the all-NA case
        dtype = self._ndarray.dtype
        return dtype.type(-1)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self:
        if dtype is None:
            # The _from_scalars strictness doesn't make much sense in this case.
            raise NotImplementedError

        res = cls._from_sequence(scalars, dtype=dtype)

        # if there are any non-category elements in scalars, these will be
        #  converted to NAs in res.
        mask = isna(scalars)
        if not (mask == res.isna()).all():
            # Some non-category element in scalars got converted to NA in res.
            raise ValueError
        return res

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray:
        ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray:
        ...

    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike:
        ...

    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        """
        Coerce this type to another dtype

        Parameters
        ----------
        dtype : numpy dtype or pandas type
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and dtype is categorical, the original
            object is returned.
        """
        dtype = pandas_dtype(dtype)
        if self.dtype is dtype:
            result = self.copy() if copy else self

        elif isinstance(dtype, CategoricalDtype):
            # GH 10696/18593/18630
            dtype = self.dtype.update_dtype(dtype)
            self = self.copy() if copy else self
            result = self._set_dtype(dtype)

        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)

        elif dtype.kind in "iu" and self.isna().any():
            raise ValueError("Cannot convert float NaN to integer")

        elif len(self.codes) == 0 or len(self.categories) == 0:
            # For NumPy 1.x compatibility we cannot use copy=None.  And
            # `copy=False` has the meaning of `copy=None` here:
            if not copy:
                result = np.asarray(self, dtype=dtype)
            else:
                result = np.array(self, dtype=dtype)

        else:
            # GH8628 (PERF): astype category codes instead of astyping array
            new_cats = self.categories._values

            try:
                new_cats = new_cats.astype(dtype=dtype, copy=copy)
                fill_value = self.categories._na_value
                if not is_valid_na_for_dtype(fill_value, dtype):
                    fill_value = lib.item_from_zerodim(
                        np.array(self.categories._na_value).astype(dtype)
                    )
            except (
                TypeError,  # downstream error msg for CategoricalIndex is misleading
                ValueError,
            ):
                msg = f"Cannot cast {self.categories.dtype} dtype to {dtype}"
                raise ValueError(msg)

            result = take_nd(
                new_cats, ensure_platform_int(self._codes), fill_value=fill_value
            )

        return result

    def to_list(self):
        """
        Alias for tolist.
        """
        # GH#51254
        warnings.warn(
            "Categorical.to_list is deprecated and will be removed in a future "
            "version. Use obj.tolist() instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.tolist()

    @classmethod
    def _from_inferred_categories(
        cls, inferred_categories, inferred_codes, dtype, true_values=None
    ) -> Self:
        """
        Construct a Categorical from inferred values.

        For inferred categories (`dtype` is None) the categories are sorted.
        For explicit `dtype`, the `inferred_categories` are cast to the
        appropriate type.

        Parameters
        ----------
        inferred_categories : Index
        inferred_codes : Index
        dtype : CategoricalDtype or 'category'
        true_values : list, optional
            If none are provided, the default ones are
            "True", "TRUE", and "true."

        Returns
        -------
        Categorical
        """
        from pandas import (
            Index,
            to_datetime,
            to_numeric,
            to_timedelta,
        )

        cats = Index(inferred_categories)
        known_categories = (
            isinstance(dtype, CategoricalDtype) and dtype.categories is not None
        )

        if known_categories:
            # Convert to a specialized type with `dtype` if specified.
            if is_any_real_numeric_dtype(dtype.categories.dtype):
                cats = to_numeric(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "M"):
                cats = to_datetime(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "m"):
                cats = to_timedelta(inferred_categories, errors="coerce")
            elif is_bool_dtype(dtype.categories.dtype):
                if true_values is None:
                    true_values = ["True", "TRUE", "true"]

                # error: Incompatible types in assignment (expression has type
                # "ndarray", variable has type "Index")
                cats = cats.isin(true_values)  # type: ignore[assignment]

        if known_categories:
            # Recode from observation order to dtype.categories order.
            categories = dtype.categories
            codes = recode_for_categories(inferred_codes, cats, categories)
        elif not cats.is_monotonic_increasing:
            # Sort categories and recode for unknown categories.
            unsorted = cats.copy()
            categories = cats.sort_values()

            codes = recode_for_categories(inferred_codes, unsorted, categories)
            dtype = CategoricalDtype(categories, ordered=False)
        else:
            dtype = CategoricalDtype(cats, ordered=False)
            codes = inferred_codes

        return cls._simple_new(codes, dtype=dtype)

    @classmethod
    def from_codes(
        cls,
        codes,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        validate: bool = True,
    ) -> Self:
        """
        Make a Categorical type from codes and categories or dtype.

        This constructor is useful if you already have codes and
        categories/dtype and so do not need the (computation intensive)
        factorization step, which is usually done on the constructor.

        If your data does not follow this convention, please use the normal
        constructor.

        Parameters
        ----------
        codes : array-like of int
            An integer array, where each integer points to a category in
            categories or dtype.categories, or else is -1 for NaN.
        categories : index-like, optional
            The categories for the categorical. Items need to be unique.
            If the categories are not given here, then they must be provided
            in `dtype`.
        ordered : bool, optional
            Whether or not this categorical is treated as an ordered
            categorical. If not given here or in `dtype`, the resulting
            categorical will be unordered.
        dtype : CategoricalDtype or "category", optional
            If :class:`CategoricalDtype`, cannot be used together with
            `categories` or `ordered`.
        validate : bool, default True
            If True, validate that the codes are valid for the dtype.
            If False, don't validate that the codes are valid. Be careful about skipping
            validation, as invalid codes can lead to severe problems, such as segfaults.

            .. versionadded:: 2.1.0

        Returns
        -------
        Categorical

        Examples
        --------
        >>> dtype = pd.CategoricalDtype(['a', 'b'], ordered=True)
        >>> pd.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=dtype)
        ['a', 'b', 'a', 'b']
        Categories (2, object): ['a' < 'b']
        """
        dtype = CategoricalDtype._from_values_or_dtype(
            categories=categories, ordered=ordered, dtype=dtype
        )
        if dtype.categories is None:
            msg = (
                "The categories must be provided in 'categories' or "
                "'dtype'. Both were None."
            )
            raise ValueError(msg)

        if validate:
            # beware: non-valid codes may segfault
            codes = cls._validate_codes_for_dtype(codes, dtype=dtype)

        return cls._simple_new(codes, dtype=dtype)

    # ------------------------------------------------------------------
    # Categories/Codes/Ordered

    @property
    def categories(self) -> Index:
        """
        The categories of this categorical.

        Setting assigns new values to each category (effectively a rename of
        each individual category).

        The assigned value has to be a list-like object. All items must be
        unique and the number of items in the new categories must be the same
        as the number of items in the old categories.

        Raises
        ------
        ValueError
            If the new categories do not validate as categories or if the
            number of new categories is unequal the number of old categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], categories=['b', 'c', 'd'])
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.categories
        Index(['b', 'c', 'd'], dtype='object')

        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.categories
        Index(['a', 'b'], dtype='object')

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'c', 'b', 'a', 'c', 'b'])
        >>> ci.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> ci = pd.CategoricalIndex(['a', 'c'], categories=['c', 'b', 'a'])
        >>> ci.categories
        Index(['c', 'b', 'a'], dtype='object')
        """
        return self.dtype.categories

    @property
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.ordered
        False

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.ordered
        True

        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.ordered
        True

        >>> cat = pd.Categorical(['a', 'b'], ordered=False)
        >>> cat.ordered
        False

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b'], ordered=True)
        >>> ci.ordered
        True

        >>> ci = pd.CategoricalIndex(['a', 'b'], ordered=False)
        >>> ci.ordered
        False
        """
        return self.dtype.ordered

    @property
    def codes(self) -> np.ndarray:
        """
        The category codes of this categorical index.

        Codes are an array of integers which are the positions of the actual
        values in the categories array.

        There is no setter, use the other categorical methods and the normal item
        setter to change values in the categorical.

        Returns
        -------
        ndarray[int]
            A non-writable view of the ``codes`` array.

        Examples
        --------
        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.codes
        array([0, 1], dtype=int8)

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'])
        >>> ci.codes
        array([0, 1, 2, 0, 1, 2], dtype=int8)

        >>> ci = pd.CategoricalIndex(['a', 'c'], categories=['c', 'b', 'a'])
        >>> ci.codes
        array([2, 0], dtype=int8)
        """
        v = self._codes.view()
        v.flags.writeable = False
        return v

    def _set_categories(self, categories, fastpath: bool = False) -> None:
        """
        Sets new categories inplace

        Parameters
        ----------
        fastpath : bool, default False
           Don't perform validation of the categories for uniqueness or nulls

        Examples
        --------
        >>> c = pd.Categorical(['a', 'b'])
        >>> c
        ['a', 'b']
        Categories (2, object): ['a', 'b']

        >>> c._set_categories(pd.Index(['a', 'c']))
        >>> c
        ['a', 'c']
        Categories (2, object): ['a', 'c']
        """
        if fastpath:
            new_dtype = CategoricalDtype._from_fastpath(categories, self.ordered)
        else:
            new_dtype = CategoricalDtype(categories, ordered=self.ordered)
        if (
            not fastpath
            and self.dtype.categories is not None
            and len(new_dtype.categories) != len(self.dtype.categories)
        ):
            raise ValueError(
                "new categories need to have the same number of "
                "items as the old categories!"
            )

        super().__init__(self._ndarray, new_dtype)

    def _set_dtype(self, dtype: CategoricalDtype) -> Self:
        """
        Internal method for directly updating the CategoricalDtype

        Parameters
        ----------
        dtype : CategoricalDtype

        Notes
        -----
        We don't do any validation here. It's assumed that the dtype is
        a (valid) instance of `CategoricalDtype`.
        """
        codes = recode_for_categories(self.codes, self.categories, dtype.categories)
        return type(self)._simple_new(codes, dtype=dtype)

    def set_ordered(self, value: bool) -> Self:
        """
        Set the ordered attribute to the boolean value.

        Parameters
        ----------
        value : bool
           Set whether this categorical is ordered (True) or not (False).
        """
        new_dtype = CategoricalDtype(self.categories, ordered=value)
        cat = self.copy()
        NDArrayBacked.__init__(cat, cat._ndarray, new_dtype)
        return cat

    def as_ordered(self) -> Self:
        """
        Set the Categorical to be ordered.

        Returns
        -------
        Categorical
            Ordered Categorical.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.ordered
        False
        >>> ser = ser.cat.as_ordered()
        >>> ser.cat.ordered
        True

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a'])
        >>> ci.ordered
        False
        >>> ci = ci.as_ordered()
        >>> ci.ordered
        True
        """
        return self.set_ordered(True)

    def as_unordered(self) -> Self:
        """
        Set the Categorical to be unordered.

        Returns
        -------
        Categorical
            Unordered Categorical.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.ordered
        True
        >>> ser = ser.cat.as_unordered()
        >>> ser.cat.ordered
        False

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a'], ordered=True)
        >>> ci.ordered
        True
        >>> ci = ci.as_unordered()
        >>> ci.ordered
        False
        """
        return self.set_ordered(False)

    def set_categories(self, new_categories, ordered=None, rename: bool = False):
        """
        Set the categories to the specified new categories.

        ``new_categories`` can include new categories (which will result in
        unused categories) or remove old categories (which results in values
        set to ``NaN``). If ``rename=True``, the categories will simply be renamed
        (less or more items than in old categories will result in values set to
        ``NaN`` or in unused categories respectively).

        This method can be used to perform more than one action of adding,
        removing, and reordering simultaneously and is therefore faster than
        performing the individual steps via the more specialised methods.

        On the other hand this methods does not do checks (e.g., whether the
        old categories are included in the new categories on a reorder), which
        can result in surprising changes, for example when using special string
        dtypes, which does not considers a S1 string equal to a single char
        python string.

        Parameters
        ----------
        new_categories : Index-like
           The categories in new order.
        ordered : bool, default False
           Whether or not the categorical is treated as a ordered categorical.
           If not given, do not change the ordered information.
        rename : bool, default False
           Whether or not the new_categories should be considered as a rename
           of the old categories or as reordered categories.

        Returns
        -------
        Categorical with reordered categories.

        Raises
        ------
        ValueError
            If new_categories does not validate as categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'A'],
        ...                           categories=['a', 'b', 'c'], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser
        0   a
        1   b
        2   c
        3   NaN
        dtype: category
        Categories (3, object): ['a' < 'b' < 'c']

        >>> ser.cat.set_categories(['A', 'B', 'C'], rename=True)
        0   A
        1   B
        2   C
        3   NaN
        dtype: category
        Categories (3, object): ['A' < 'B' < 'C']

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'A'],
        ...                          categories=['a', 'b', 'c'], ordered=True)
        >>> ci
        CategoricalIndex(['a', 'b', 'c', nan], categories=['a', 'b', 'c'],
                         ordered=True, dtype='category')

        >>> ci.set_categories(['A', 'b', 'c'])
        CategoricalIndex([nan, 'b', 'c', nan], categories=['A', 'b', 'c'],
                         ordered=True, dtype='category')
        >>> ci.set_categories(['A', 'b', 'c'], rename=True)
        CategoricalIndex(['A', 'b', 'c', nan], categories=['A', 'b', 'c'],
                         ordered=True, dtype='category')
        """

        if ordered is None:
            ordered = self.dtype.ordered
        new_dtype = CategoricalDtype(new_categories, ordered=ordered)

        cat = self.copy()
        if rename:
            if cat.dtype.categories is not None and len(new_dtype.categories) < len(
                cat.dtype.categories
            ):
                # remove all _codes which are larger and set to -1/NaN
                cat._codes[cat._codes >= len(new_dtype.categories)] = -1
            codes = cat._codes
        else:
            codes = recode_for_categories(
                cat.codes, cat.categories, new_dtype.categories
            )
        NDArrayBacked.__init__(cat, codes, new_dtype)
        return cat

    def rename_categories(self, new_categories) -> Self:
        """
        Rename categories.

        Parameters
        ----------
        new_categories : list-like, dict-like or callable

            New categories which will replace old categories.

            * list-like: all items must be unique and the number of items in
              the new categories must match the existing number of categories.

            * dict-like: specifies a mapping from
              old categories to new. Categories not contained in the mapping
              are passed through and extra categories in the mapping are
              ignored.

            * callable : a callable that is called on all items in the old
              categories and whose return values comprise the new categories.

        Returns
        -------
        Categorical
            Categorical with renamed categories.

        Raises
        ------
        ValueError
            If new categories are list-like and do not have the same number of
            items than the current categories or do not validate as categories

        See Also
        --------
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['a', 'a', 'b'])
        >>> c.rename_categories([0, 1])
        [0, 0, 1]
        Categories (2, int64): [0, 1]

        For dict-like ``new_categories``, extra keys are ignored and
        categories not in the dictionary are passed through

        >>> c.rename_categories({'a': 'A', 'c': 'C'})
        ['A', 'A', 'b']
        Categories (2, object): ['A', 'b']

        You may also provide a callable to create the new categories

        >>> c.rename_categories(lambda x: x.upper())
        ['A', 'A', 'B']
        Categories (2, object): ['A', 'B']
        """

        if is_dict_like(new_categories):
            new_categories = [
                new_categories.get(item, item) for item in self.categories
            ]
        elif callable(new_categories):
            new_categories = [new_categories(item) for item in self.categories]

        cat = self.copy()
        cat._set_categories(new_categories)
        return cat

    def reorder_categories(self, new_categories, ordered=None) -> Self:
        """
        Reorder categories as specified in new_categories.

        ``new_categories`` need to include all old categories and no new category
        items.

        Parameters
        ----------
        new_categories : Index-like
           The categories in new order.
        ordered : bool, optional
           Whether or not the categorical is treated as a ordered categorical.
           If not given, do not change the ordered information.

        Returns
        -------
        Categorical
            Categorical with reordered categories.

        Raises
        ------
        ValueError
            If the new categories do not contain all old category items or any
            new ones

        See Also
        --------
        rename_categories : Rename categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser = ser.cat.reorder_categories(['c', 'b', 'a'], ordered=True)
        >>> ser
        0   a
        1   b
        2   c
        3   a
        dtype: category
        Categories (3, object): ['c' < 'b' < 'a']

        >>> ser.sort_values()
        2   c
        1   b
        0   a
        3   a
        dtype: category
        Categories (3, object): ['c' < 'b' < 'a']

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a'])
        >>> ci
        CategoricalIndex(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c'],
                         ordered=False, dtype='category')
        >>> ci.reorder_categories(['c', 'b', 'a'], ordered=True)
        CategoricalIndex(['a', 'b', 'c', 'a'], categories=['c', 'b', 'a'],
                         ordered=True, dtype='category')
        """
        if (
            len(self.categories) != len(new_categories)
            or not self.categories.difference(new_categories).empty
        ):
            raise ValueError(
                "items in new_categories are not the same as in old categories"
            )
        return self.set_categories(new_categories, ordered=ordered)

    def add_categories(self, new_categories) -> Self:
        """
        Add new categories.

        `new_categories` will be included at the last/highest place in the
        categories and will be unused directly after this call.

        Parameters
        ----------
        new_categories : category or list-like of category
           The new categories to be included.

        Returns
        -------
        Categorical
            Categorical with new categories added.

        Raises
        ------
        ValueError
            If the new categories include old categories or do not validate as
            categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['c', 'b', 'c'])
        >>> c
        ['c', 'b', 'c']
        Categories (2, object): ['b', 'c']

        >>> c.add_categories(['d', 'a'])
        ['c', 'b', 'c']
        Categories (4, object): ['b', 'c', 'd', 'a']
        """

        if not is_list_like(new_categories):
            new_categories = [new_categories]
        already_included = set(new_categories) & set(self.dtype.categories)
        if len(already_included) != 0:
            raise ValueError(
                f"new categories must not include old categories: {already_included}"
            )

        if hasattr(new_categories, "dtype"):
            from pandas import Series

            dtype = find_common_type(
                [self.dtype.categories.dtype, new_categories.dtype]
            )
            new_categories = Series(
                list(self.dtype.categories) + list(new_categories), dtype=dtype
            )
        else:
            new_categories = list(self.dtype.categories) + list(new_categories)

        new_dtype = CategoricalDtype(new_categories, self.ordered)
        cat = self.copy()
        codes = coerce_indexer_dtype(cat._ndarray, new_dtype.categories)
        NDArrayBacked.__init__(cat, codes, new_dtype)
        return cat

    def remove_categories(self, removals) -> Self:
        """
        Remove the specified categories.

        `removals` must be included in the old categories. Values which were in
        the removed categories will be set to NaN

        Parameters
        ----------
        removals : category or list of categories
           The categories which should be removed.

        Returns
        -------
        Categorical
            Categorical with removed categories.

        Raises
        ------
        ValueError
            If the removals are not contained in the categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['a', 'c', 'b', 'c', 'd'])
        >>> c
        ['a', 'c', 'b', 'c', 'd']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c.remove_categories(['d', 'a'])
        [NaN, 'c', 'b', 'c', NaN]
        Categories (2, object): ['b', 'c']
        """
        from pandas import Index

        if not is_list_like(removals):
            removals = [removals]

        removals = Index(removals).unique().dropna()
        new_categories = (
            self.dtype.categories.difference(removals, sort=False)
            if self.dtype.ordered is True
            else self.dtype.categories.difference(removals)
        )
        not_included = removals.difference(self.dtype.categories)

        if len(not_included) != 0:
            not_included = set(not_included)
            raise ValueError(f"removals must all be in old categories: {not_included}")

        return self.set_categories(new_categories, ordered=self.ordered, rename=False)

    def remove_unused_categories(self) -> Self:
        """
        Remove categories which are not used.

        Returns
        -------
        Categorical
            Categorical with unused categories dropped.

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['a', 'c', 'b', 'c', 'd'])
        >>> c
        ['a', 'c', 'b', 'c', 'd']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c[2] = 'a'
        >>> c[4] = 'c'
        >>> c
        ['a', 'c', 'a', 'c', 'c']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c.remove_unused_categories()
        ['a', 'c', 'a', 'c', 'c']
        Categories (2, object): ['a', 'c']
        """
        idx, inv = np.unique(self._codes, return_inverse=True)

        if idx.size != 0 and idx[0] == -1:  # na sentinel
            idx, inv = idx[1:], inv - 1

        new_categories = self.dtype.categories.take(idx)
        new_dtype = CategoricalDtype._from_fastpath(
            new_categories, ordered=self.ordered
        )
        new_codes = coerce_indexer_dtype(inv, new_dtype.categories)

        cat = self.copy()
        NDArrayBacked.__init__(cat, new_codes, new_dtype)
        return cat

    # ------------------------------------------------------------------

    def map(
        self,
        mapper,
        na_action: Literal["ignore"] | None | lib.NoDefault = lib.no_default,
    ):
        """
        Map categories using an input mapping or function.

        Maps the categories to new categories. If the mapping correspondence is
        one-to-one the result is a :class:`~pandas.Categorical` which has the
        same order property as the original, otherwise a :class:`~pandas.Index`
        is returned. NaN values are unaffected.

        If a `dict` or :class:`~pandas.Series` is used any unmapped category is
        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`
        will be returned.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default 'ignore'
            If 'ignore', propagate NaN values, without passing them to the
            mapping correspondence.

            .. deprecated:: 2.1.0

               The default value of 'ignore' has been deprecated and will be changed to
               None in the future.

        Returns
        -------
        pandas.Categorical or pandas.Index
            Mapped categorical.

        See Also
        --------
        CategoricalIndex.map : Apply a mapping correspondence on a
            :class:`~pandas.CategoricalIndex`.
        Index.map : Apply a mapping correspondence on an
            :class:`~pandas.Index`.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.
        Series.apply : Apply more complex functions on a
            :class:`~pandas.Series`.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b', 'c'])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.map(lambda x: x.upper(), na_action=None)
        ['A', 'B', 'C']
        Categories (3, object): ['A', 'B', 'C']
        >>> cat.map({'a': 'first', 'b': 'second', 'c': 'third'}, na_action=None)
        ['first', 'second', 'third']
        Categories (3, object): ['first', 'second', 'third']

        If the mapping is one-to-one the ordering of the categories is
        preserved:

        >>> cat = pd.Categorical(['a', 'b', 'c'], ordered=True)
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a' < 'b' < 'c']
        >>> cat.map({'a': 3, 'b': 2, 'c': 1}, na_action=None)
        [3, 2, 1]
        Categories (3, int64): [3 < 2 < 1]

        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:

        >>> cat.map({'a': 'first', 'b': 'second', 'c': 'first'}, na_action=None)
        Index(['first', 'second', 'first'], dtype='object')

        If a `dict` is used, all unmapped categories are mapped to `NaN` and
        the result is an :class:`~pandas.Index`:

        >>> cat.map({'a': 'first', 'b': 'second'}, na_action=None)
        Index(['first', 'second', nan], dtype='object')
        """
        if na_action is lib.no_default:
            warnings.warn(
                "The default value of 'ignore' for the `na_action` parameter in "
                "pandas.Categorical.map is deprecated and will be "
                "changed to 'None' in a future version. Please set na_action to the "
                "desired value to avoid seeing this warning",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            na_action = "ignore"

        assert callable(mapper) or is_dict_like(mapper)

        new_categories = self.categories.map(mapper)

        has_nans = np.any(self._codes == -1)

        na_val = np.nan
        if na_action is None and has_nans:
            na_val = mapper(np.nan) if callable(mapper) else mapper.get(np.nan, np.nan)

        if new_categories.is_unique and not new_categories.hasnans and na_val is np.nan:
            new_dtype = CategoricalDtype(new_categories, ordered=self.ordered)
            return self.from_codes(self._codes.copy(), dtype=new_dtype, validate=False)

        if has_nans:
            new_categories = new_categories.insert(len(new_categories), na_val)

        return np.take(new_categories, self._codes)

    __eq__ = _cat_compare_op(operator.eq)
    __ne__ = _cat_compare_op(operator.ne)
    __lt__ = _cat_compare_op(operator.lt)
    __gt__ = _cat_compare_op(operator.gt)
    __le__ = _cat_compare_op(operator.le)
    __ge__ = _cat_compare_op(operator.ge)

    # -------------------------------------------------------------
    # Validators; ideally these can be de-duplicated

    def _validate_setitem_value(self, value):
        if not is_hashable(value):
            # wrap scalars and hashable-listlikes in list
            return self._validate_listlike(value)
        else:
            return self._validate_scalar(value)

    def _validate_scalar(self, fill_value):
        """
        Convert a user-facing fill_value to a representation to use with our
        underlying ndarray, raising TypeError if this is not possible.

        Parameters
        ----------
        fill_value : object

        Returns
        -------
        fill_value : int

        Raises
        ------
        TypeError
        """

        if is_valid_na_for_dtype(fill_value, self.categories.dtype):
            fill_value = -1
        elif fill_value in self.categories:
            fill_value = self._unbox_scalar(fill_value)
        else:
            raise TypeError(
                "Cannot setitem on a Categorical with a new "
                f"category ({fill_value}), set the categories first"
            ) from None
        return fill_value

    @classmethod
    def _validate_codes_for_dtype(cls, codes, *, dtype: CategoricalDtype) -> np.ndarray:
        if isinstance(codes, ExtensionArray) and is_integer_dtype(codes.dtype):
            # Avoid the implicit conversion of Int to object
            if isna(codes).any():
                raise ValueError("codes cannot contain NA values")
            codes = codes.to_numpy(dtype=np.int64)
        else:
            codes = np.asarray(codes)
        if len(codes) and codes.dtype.kind not in "iu":
            raise ValueError("codes need to be array-like integers")

        if len(codes) and (codes.max() >= len(dtype.categories) or codes.min() < -1):
            raise ValueError("codes need to be between -1 and len(categories)-1")
        return codes

    # -------------------------------------------------------------

    @ravel_compat
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        The numpy array interface.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or None
            Specifies the the dtype for the array.

        copy : bool or None, optional
            See :func:`numpy.asarray`.

        Returns
        -------
        numpy.array
            A numpy array of either the specified dtype or,
            if dtype==None (default), the same dtype as
            categorical.categories.dtype.

        Examples
        --------

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)

        The following calls ``cat.__array__``

        >>> np.asarray(cat)
        array(['a', 'b'], dtype=object)
        """
        if copy is False:
            warnings.warn(
                "Starting with NumPy 2.0, the behavior of the 'copy' keyword has "
                "changed and passing 'copy=False' raises an error when returning "
                "a zero-copy NumPy array is not possible. pandas will follow "
                "this behavior starting with pandas 3.0.\nThis conversion to "
                "NumPy requires a copy, but 'copy=False' was passed. Consider "
                "using 'np.asarray(..)' instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        ret = take_nd(self.categories._values, self._codes)
        # When we're a Categorical[ExtensionArray], like Interval,
        # we need to ensure __array__ gets all the way to an
        # ndarray.

        # `take_nd` should already make a copy, so don't force again.
        return np.asarray(ret, dtype=dtype)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # for binary ops, use our custom dunder methods
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. test_numpy_ufuncs_out
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            # e.g. TestCategoricalAnalytics::test_min_max_ordered
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        # for all other cases, raise for now (similarly as what happens in
        # Series.__array_prepare__)
        raise TypeError(
            f"Object with dtype {self.dtype} cannot perform "
            f"the numpy op {ufunc.__name__}"
        )

    def __setstate__(self, state) -> None:
        """Necessary for making this object picklable"""
        if not isinstance(state, dict):
            return super().__setstate__(state)

        if "_dtype" not in state:
            state["_dtype"] = CategoricalDtype(state["_categories"], state["_ordered"])

        if "_codes" in state and "_ndarray" not in state:
            # backward compat, changed what is property vs attribute
            state["_ndarray"] = state.pop("_codes")

        super().__setstate__(state)

    @property
    def nbytes(self) -> int:
        return self._codes.nbytes + self.dtype.categories.values.nbytes

    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self._codes.nbytes + self.dtype.categories.memory_usage(deep=deep)

    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Detect missing values

        Missing values (-1 in .codes) are detected.

        Returns
        -------
        np.ndarray[bool] of whether my values are null

        See Also
        --------
        isna : Top-level isna.
        isnull : Alias of isna.
        Categorical.notna : Boolean inverse of Categorical.isna.

        """
        return self._codes == -1

    isnull = isna

    def notna(self) -> npt.NDArray[np.bool_]:
        """
        Inverse of isna

        Both missing values (-1 in .codes) and NA as a category are detected as
        null.

        Returns
        -------
        np.ndarray[bool] of whether my values are not null

        See Also
        --------
        notna : Top-level notna.
        notnull : Alias of notna.
        Categorical.isna : Boolean inverse of Categorical.notna.

        """
        return ~self.isna()

    notnull = notna

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of each category.

        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        from pandas import (
            CategoricalIndex,
            Series,
        )

        code, cat = self._codes, self.categories
        ncat, mask = (len(cat), code >= 0)
        ix, clean = np.arange(ncat), mask.all()

        if dropna or clean:
            obs = code if clean else code[mask]
            count = np.bincount(obs, minlength=ncat or 0)
        else:
            count = np.bincount(np.where(mask, code, ncat))
            ix = np.append(ix, -1)

        ix = coerce_indexer_dtype(ix, self.dtype.categories)
        ix = self._from_backing_data(ix)

        return Series(
            count, index=CategoricalIndex(ix), dtype="int64", name="count", copy=False
        )

    # error: Argument 2 of "_empty" is incompatible with supertype
    # "NDArrayBackedExtensionArray"; supertype defines the argument type as
    # "ExtensionDtype"
    @classmethod
    def _empty(  # type: ignore[override]
        cls, shape: Shape, dtype: CategoricalDtype
    ) -> Self:
        """
        Analogous to np.empty(shape, dtype=dtype)

        Parameters
        ----------
        shape : tuple[int]
        dtype : CategoricalDtype
        """
        arr = cls._from_sequence([], dtype=dtype)

        # We have to use np.zeros instead of np.empty otherwise the resulting
        #  ndarray may contain codes not supported by this dtype, in which
        #  case repr(result) could segfault.
        backing = np.zeros(shape, dtype=arr._ndarray.dtype)

        return arr._from_backing_data(backing)

    def _internal_get_values(self) -> ArrayLike:
        """
        Return the values.

        For internal compatibility with pandas formatting.

        Returns
        -------
        np.ndarray or ExtensionArray
            A numpy array or ExtensionArray of the same dtype as
            categorical.categories.dtype.
        """
        # if we are a datetime and period index, return Index to keep metadata
        if needs_i8_conversion(self.categories.dtype):
            return self.categories.take(self._codes, fill_value=NaT)._values
        elif is_integer_dtype(self.categories.dtype) and -1 in self._codes:
            return (
                self.categories.astype("object")
                .take(self._codes, fill_value=np.nan)
                ._values
            )
        return np.array(self)

    def check_for_ordered(self, op) -> None:
        """assert that we are ordered"""
        if not self.ordered:
            raise TypeError(
                f"Categorical is not ordered for operation {op}\n"
                "you can use .as_ordered() to change the "
                "Categorical to an ordered one\n"
            )

    def argsort(
        self, *, ascending: bool = True, kind: SortKind = "quicksort", **kwargs
    ):
        """
        Return the indices that would sort the Categorical.

        Missing values are sorted at the end.

        Parameters
        ----------
        ascending : bool, default True
            Whether the indices should result in an ascending
            or descending sort.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm.
        **kwargs:
            passed through to :func:`numpy.argsort`.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort

        Notes
        -----
        While an ordering is applied to the category values, arg-sorting
        in this context refers more to organizing and grouping together
        based on matching category values. Thus, this function can be
        called on an unordered Categorical instance unlike the functions
        'Categorical.min' and 'Categorical.max'.

        Examples
        --------
        >>> pd.Categorical(['b', 'b', 'a', 'c']).argsort()
        array([2, 0, 1, 3])

        >>> cat = pd.Categorical(['b', 'b', 'a', 'c'],
        ...                      categories=['c', 'b', 'a'],
        ...                      ordered=True)
        >>> cat.argsort()
        array([3, 0, 1, 2])

        Missing values are placed at the end

        >>> cat = pd.Categorical([2, None, 1])
        >>> cat.argsort()
        array([2, 0, 1])
        """
        return super().argsort(ascending=ascending, kind=kind, **kwargs)

    @overload
    def sort_values(
        self,
        *,
        inplace: Literal[False] = ...,
        ascending: bool = ...,
        na_position: str = ...,
    ) -> Self:
        ...

    @overload
    def sort_values(
        self, *, inplace: Literal[True], ascending: bool = ..., na_position: str = ...
    ) -> None:
        ...

    def sort_values(
        self,
        *,
        inplace: bool = False,
        ascending: bool = True,
        na_position: str = "last",
    ) -> Self | None:
        """
        Sort the Categorical by category value returning a new
        Categorical by default.

        While an ordering is applied to the category values, sorting in this
        context refers more to organizing and grouping together based on
        matching category values. Thus, this function can be called on an
        unordered Categorical instance unlike the functions 'Categorical.min'
        and 'Categorical.max'.

        Parameters
        ----------
        inplace : bool, default False
            Do operation in place.
        ascending : bool, default True
            Order ascending. Passing False orders descending. The
            ordering parameter provides the method by which the
            category values are organized.
        na_position : {'first', 'last'} (optional, default='last')
            'first' puts NaNs at the beginning
            'last' puts NaNs at the end

        Returns
        -------
        Categorical or None

        See Also
        --------
        Categorical.sort
        Series.sort_values

        Examples
        --------
        >>> c = pd.Categorical([1, 2, 2, 1, 5])
        >>> c
        [1, 2, 2, 1, 5]
        Categories (3, int64): [1, 2, 5]
        >>> c.sort_values()
        [1, 1, 2, 2, 5]
        Categories (3, int64): [1, 2, 5]
        >>> c.sort_values(ascending=False)
        [5, 2, 2, 1, 1]
        Categories (3, int64): [1, 2, 5]

        >>> c = pd.Categorical([1, 2, 2, 1, 5])

        'sort_values' behaviour with NaNs. Note that 'na_position'
        is independent of the 'ascending' parameter:

        >>> c = pd.Categorical([np.nan, 2, 2, np.nan, 5])
        >>> c
        [NaN, 2, 2, NaN, 5]
        Categories (2, int64): [2, 5]
        >>> c.sort_values()
        [2, 2, 5, NaN, NaN]
        Categories (2, int64): [2, 5]
        >>> c.sort_values(ascending=False)
        [5, 2, 2, NaN, NaN]
        Categories (2, int64): [2, 5]
        >>> c.sort_values(na_position='first')
        [NaN, NaN, 2, 2, 5]
        Categories (2, int64): [2, 5]
        >>> c.sort_values(ascending=False, na_position='first')
        [NaN, NaN, 5, 2, 2]
        Categories (2, int64): [2, 5]
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if na_position not in ["last", "first"]:
            raise ValueError(f"invalid na_position: {repr(na_position)}")

        sorted_idx = nargsort(self, ascending=ascending, na_position=na_position)

        if not inplace:
            codes = self._codes[sorted_idx]
            return self._from_backing_data(codes)
        self._codes[:] = self._codes[sorted_idx]
        return None

    def _rank(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ):
        """
        See Series.rank.__doc__.
        """
        if axis != 0:
            raise NotImplementedError
        vff = self._values_for_rank()
        return algorithms.rank(
            vff,
            axis=axis,
            method=method,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        )

    def _values_for_rank(self) -> np.ndarray:
        """
        For correctly ranking ordered categorical data. See GH#15420

        Ordered categorical data should be ranked on the basis of
        codes with -1 translated to NaN.

        Returns
        -------
        numpy.array

        """
        from pandas import Series

        if self.ordered:
            values = self.codes
            mask = values == -1
            if mask.any():
                values = values.astype("float64")
                values[mask] = np.nan
        elif is_any_real_numeric_dtype(self.categories.dtype):
            values = np.array(self)
        else:
            #  reorder the categories (so rank can use the float codes)
            #  instead of passing an object array to rank
            values = np.array(
                self.rename_categories(
                    Series(self.categories, copy=False).rank().values
                )
            )
        return values

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        """
        Hash a Categorical by hashing its categories, and then mapping the codes
        to the hashes.

        Parameters
        ----------
        encoding : str
        hash_key : str
        categorize : bool
            Ignored for Categorical.

        Returns
        -------
        np.ndarray[uint64]
        """
        # Note we ignore categorize, as we are already Categorical.
        from pandas.core.util.hashing import hash_array

        # Convert ExtensionArrays to ndarrays
        values = np.asarray(self.categories._values)
        hashed = hash_array(values, encoding, hash_key, categorize=False)

        # we have uint64, as we don't directly support missing values
        # we don't want to use take_nd which will coerce to float
        # instead, directly construct the result with a
        # max(np.uint64) as the missing value indicator
        #
        # TODO: GH#15362

        mask = self.isna()
        if len(hashed):
            result = hashed.take(self._codes)
        else:
            result = np.zeros(len(mask), dtype="uint64")

        if mask.any():
            result[mask] = lib.u8max

        return result

    # ------------------------------------------------------------------
    # NDArrayBackedExtensionArray compat

    @property
    def _codes(self) -> np.ndarray:
        return self._ndarray

    def _box_func(self, i: int):
        if i == -1:
            return np.nan
        return self.categories[i]

    def _unbox_scalar(self, key) -> int:
        # searchsorted is very performance sensitive. By converting codes
        # to same dtype as self.codes, we get much faster performance.
        code = self.categories.get_loc(key)
        code = self._ndarray.dtype.type(code)
        return code

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator:
        """
        Returns an Iterator over the values of this Categorical.
        """
        if self.ndim == 1:
            return iter(self._internal_get_values().tolist())
        else:
            return (self[n] for n in range(len(self)))

    def __contains__(self, key) -> bool:
        """
        Returns True if `key` is in this Categorical.
        """
        # if key is a NaN, check if any NaN is in self.
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return bool(self.isna().any())

        return contains(self, key, container=self._codes)

    # ------------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        # Returning None here will cause format_array to do inference.
        return None

    def _repr_categories(self) -> list[str]:
        """
        return the base repr for the categories
        """
        max_categories = (
            10
            if get_option("display.max_categories") == 0
            else get_option("display.max_categories")
        )
        from pandas.io.formats import format as fmt

        format_array = partial(
            fmt.format_array, formatter=None, quoting=QUOTE_NONNUMERIC
        )
        if len(self.categories) > max_categories:
            num = max_categories // 2
            head = format_array(self.categories[:num]._values)
            tail = format_array(self.categories[-num:]._values)
            category_strs = head + ["..."] + tail
        else:
            category_strs = format_array(self.categories._values)

        # Strip all leading spaces, which format_array adds for columns...
        category_strs = [x.strip() for x in category_strs]
        return category_strs

    def _get_repr_footer(self) -> str:
        """
        Returns a string representation of the footer.
        """
        category_strs = self._repr_categories()
        dtype = str(self.categories.dtype)
        levheader = f"Categories ({len(self.categories)}, {dtype}): "
        width, _ = get_terminal_size()
        max_width = get_option("display.width") or width
        if console.in_ipython_frontend():
            # 0 = no breaks
            max_width = 0
        levstring = ""
        start = True
        cur_col_len = len(levheader)  # header
        sep_len, sep = (3, " < ") if self.ordered else (2, ", ")
        linesep = f"{sep.rstrip()}\n"  # remove whitespace
        for val in category_strs:
            if max_width != 0 and cur_col_len + sep_len + len(val) > max_width:
                levstring += linesep + (" " * (len(levheader) + 1))
                cur_col_len = len(levheader) + 1  # header + a whitespace
            elif not start:
                levstring += sep
                cur_col_len += len(val)
            levstring += val
            start = False
        # replace to simple save space by
        return f"{levheader}[{levstring.replace(' < ... < ', ' ... ')}]"

    def _get_values_repr(self) -> str:
        from pandas.io.formats import format as fmt

        assert len(self) > 0

        vals = self._internal_get_values()
        fmt_values = fmt.format_array(
            vals,
            None,
            float_format=None,
            na_rep="NaN",
            quoting=QUOTE_NONNUMERIC,
        )

        fmt_values = [i.strip() for i in fmt_values]
        joined = ", ".join(fmt_values)
        result = "[" + joined + "]"
        return result

    def __repr__(self) -> str:
        """
        String representation.
        """
        footer = self._get_repr_footer()
        length = len(self)
        max_len = 10
        if length > max_len:
            # In long cases we do not display all entries, so we add Length
            #  information to the __repr__.
            num = max_len // 2
            head = self[:num]._get_values_repr()
            tail = self[-(max_len - num) :]._get_values_repr()
            body = f"{head[:-1]}, ..., {tail[1:]}"
            length_info = f"Length: {len(self)}"
            result = f"{body}\n{length_info}\n{footer}"
        elif length > 0:
            body = self._get_values_repr()
            result = f"{body}\n{footer}"
        else:
            # In the empty case we use a comma instead of newline to get
            #  a more compact __repr__
            body = "[]"
            result = f"{body}, {footer}"

        return result

    # ------------------------------------------------------------------

    def _validate_listlike(self, value):
        # NB: here we assume scalar-like tuples have already been excluded
        value = extract_array(value, extract_numpy=True)

        # require identical categories set
        if isinstance(value, Categorical):
            if self.dtype != value.dtype:
                raise TypeError(
                    "Cannot set a Categorical with another, "
                    "without identical categories"
                )
            # dtype equality implies categories_match_up_to_permutation
            value = self._encode_with_my_categories(value)
            return value._codes

        from pandas import Index

        # tupleize_cols=False for e.g. test_fillna_iterable_category GH#41914
        to_add = Index._with_infer(value, tupleize_cols=False).difference(
            self.categories
        )

        # no assignments of values not in categories, but it's always ok to set
        # something to np.nan
        if len(to_add) and not isna(to_add).all():
            raise TypeError(
                "Cannot setitem on a Categorical with a new "
                "category, set the categories first"
            )

        codes = self.categories.get_indexer(value)
        return codes.astype(self._ndarray.dtype, copy=False)

    def _reverse_indexer(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """
        Compute the inverse of a categorical, returning
        a dict of categories -> indexers.

        *This is an internal function*

        Returns
        -------
        Dict[Hashable, np.ndarray[np.intp]]
            dict of categories -> indexers

        Examples
        --------
        >>> c = pd.Categorical(list('aabca'))
        >>> c
        ['a', 'a', 'b', 'c', 'a']
        Categories (3, object): ['a', 'b', 'c']
        >>> c.categories
        Index(['a', 'b', 'c'], dtype='object')
        >>> c.codes
        array([0, 0, 1, 2, 0], dtype=int8)
        >>> c._reverse_indexer()
        {'a': array([0, 1, 4]), 'b': array([2]), 'c': array([3])}

        """
        categories = self.categories
        r, counts = libalgos.groupsort_indexer(
            ensure_platform_int(self.codes), categories.size
        )
        counts = ensure_int64(counts).cumsum()
        _result = (r[start:end] for start, end in zip(counts, counts[1:]))
        return dict(zip(categories, _result))

    # ------------------------------------------------------------------
    # Reductions

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        result = super()._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)
        if name in ["argmax", "argmin"]:
            # don't wrap in Categorical!
            return result
        if keepdims:
            return type(self)(result, dtype=self.dtype)
        else:
            return result

    def min(self, *, skipna: bool = True, **kwargs):
        """
        The minimum value of the object.

        Only ordered `Categoricals` have a minimum!

        Raises
        ------
        TypeError
            If the `Categorical` is not `ordered`.

        Returns
        -------
        min : the minimum of this `Categorical`, NA value if empty
        """
        nv.validate_minmax_axis(kwargs.get("axis", 0))
        nv.validate_min((), kwargs)
        self.check_for_ordered("min")

        if not len(self._codes):
            return self.dtype.na_value

        good = self._codes != -1
        if not good.all():
            if skipna and good.any():
                pointer = self._codes[good].min()
            else:
                return np.nan
        else:
            pointer = self._codes.min()
        return self._wrap_reduction_result(None, pointer)

    def max(self, *, skipna: bool = True, **kwargs):
        """
        The maximum value of the object.

        Only ordered `Categoricals` have a maximum!

        Raises
        ------
        TypeError
            If the `Categorical` is not `ordered`.

        Returns
        -------
        max : the maximum of this `Categorical`, NA if array is empty
        """
        nv.validate_minmax_axis(kwargs.get("axis", 0))
        nv.validate_max((), kwargs)
        self.check_for_ordered("max")

        if not len(self._codes):
            return self.dtype.na_value

        good = self._codes != -1
        if not good.all():
            if skipna and good.any():
                pointer = self._codes[good].max()
            else:
                return np.nan
        else:
            pointer = self._codes.max()
        return self._wrap_reduction_result(None, pointer)

    def _mode(self, dropna: bool = True) -> Categorical:
        codes = self._codes
        mask = None
        if dropna:
            mask = self.isna()

        res_codes = algorithms.mode(codes, mask=mask)
        res_codes = cast(np.ndarray, res_codes)
        assert res_codes.dtype == codes.dtype
        res = self._from_backing_data(res_codes)
        return res

    # ------------------------------------------------------------------
    # ExtensionArray Interface

    def unique(self) -> Self:
        """
        Return the ``Categorical`` which ``categories`` and ``codes`` are
        unique.

        .. versionchanged:: 1.3.0

            Previously, unused categories were dropped from the new categories.

        Returns
        -------
        Categorical

        See Also
        --------
        pandas.unique
        CategoricalIndex.unique
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> pd.Categorical(list("baabc")).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> pd.Categorical(list("baab"), categories=list("abc"), ordered=True).unique()
        ['b', 'a']
        Categories (3, object): ['a' < 'b' < 'c']
        """
        # pylint: disable=useless-parent-delegation
        return super().unique()

    def equals(self, other: object) -> bool:
        """
        Returns True if categorical arrays are equal.

        Parameters
        ----------
        other : `Categorical`

        Returns
        -------
        bool
        """
        if not isinstance(other, Categorical):
            return False
        elif self._categories_match_up_to_permutation(other):
            other = self._encode_with_my_categories(other)
            return np.array_equal(self._codes, other._codes)
        return False

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt = 0) -> Self:
        from pandas.core.dtypes.concat import union_categoricals

        first = to_concat[0]
        if axis >= first.ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {first.ndim}"
            )

        if axis == 1:
            # Flatten, concatenate then reshape
            if not all(x.ndim == 2 for x in to_concat):
                raise ValueError

            # pass correctly-shaped to union_categoricals
            tc_flat = []
            for obj in to_concat:
                tc_flat.extend([obj[:, i] for i in range(obj.shape[1])])

            res_flat = cls._concat_same_type(tc_flat, axis=0)

            result = res_flat.reshape(len(first), -1, order="F")
            return result

        result = union_categoricals(to_concat)
        return result

    # ------------------------------------------------------------------

    def _encode_with_my_categories(self, other: Categorical) -> Categorical:
        """
        Re-encode another categorical using this Categorical's categories.

        Notes
        -----
        This assumes we have already checked
        self._categories_match_up_to_permutation(other).
        """
        # Indexing on codes is more efficient if categories are the same,
        #  so we can apply some optimizations based on the degree of
        #  dtype-matching.
        codes = recode_for_categories(
            other.codes, other.categories, self.categories, copy=False
        )
        return self._from_backing_data(codes)

    def _categories_match_up_to_permutation(self, other: Categorical) -> bool:
        """
        Returns True if categoricals are the same dtype
          same categories, and same ordered

        Parameters
        ----------
        other : Categorical

        Returns
        -------
        bool
        """
        return hash(self.dtype) == hash(other.dtype)

    def describe(self) -> DataFrame:
        """
        Describes this Categorical

        Returns
        -------
        description: `DataFrame`
            A dataframe with frequency and counts by category.
        """
        counts = self.value_counts(dropna=False)
        freqs = counts / counts.sum()

        from pandas import Index
        from pandas.core.reshape.concat import concat

        result = concat([counts, freqs], axis=1)
        result.columns = Index(["counts", "freqs"])
        result.index.name = "categories"

        return result

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Check whether `values` are contained in Categorical.

        Return a boolean NumPy Array showing whether each element in
        the Categorical matches an element in the passed sequence of
        `values` exactly.

        Parameters
        ----------
        values : np.ndarray or ExtensionArray
            The sequence of values to test. Passing in a single string will
            raise a ``TypeError``. Instead, turn a single string into a
            list of one element.

        Returns
        -------
        np.ndarray[bool]

        Raises
        ------
        TypeError
          * If `values` is not a set or list-like

        See Also
        --------
        pandas.Series.isin : Equivalent method on Series.

        Examples
        --------
        >>> s = pd.Categorical(['lama', 'cow', 'lama', 'beetle', 'lama',
        ...                'hippo'])
        >>> s.isin(['cow', 'lama'])
        array([ True,  True,  True, False,  True, False])

        Passing a single string as ``s.isin('lama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(['lama'])
        array([ True, False,  True, False,  True, False])
        """
        null_mask = np.asarray(isna(values))
        code_values = self.categories.get_indexer_for(values)
        code_values = code_values[null_mask | (code_values >= 0)]
        return algorithms.isin(self.codes, code_values)

    def _replace(self, *, to_replace, value, inplace: bool = False):
        from pandas import Index

        orig_dtype = self.dtype

        inplace = validate_bool_kwarg(inplace, "inplace")
        cat = self if inplace else self.copy()

        mask = isna(np.asarray(value))
        if mask.any():
            removals = np.asarray(to_replace)[mask]
            removals = cat.categories[cat.categories.isin(removals)]
            new_cat = cat.remove_categories(removals)
            NDArrayBacked.__init__(cat, new_cat.codes, new_cat.dtype)

        ser = cat.categories.to_series()
        ser = ser.replace(to_replace=to_replace, value=value)

        all_values = Index(ser)

        # GH51016: maintain order of existing categories
        idxr = cat.categories.get_indexer_for(all_values)
        locs = np.arange(len(ser))
        locs = np.where(idxr == -1, locs, idxr)
        locs = locs.argsort()

        new_categories = ser.take(locs)
        new_categories = new_categories.drop_duplicates(keep="first")
        new_categories = Index(new_categories)
        new_codes = recode_for_categories(
            cat._codes, all_values, new_categories, copy=False
        )
        new_dtype = CategoricalDtype(new_categories, ordered=self.dtype.ordered)
        NDArrayBacked.__init__(cat, new_codes, new_dtype)

        if new_dtype != orig_dtype:
            warnings.warn(
                # GH#55147
                "The behavior of Series.replace (and DataFrame.replace) with "
                "CategoricalDtype is deprecated. In a future version, replace "
                "will only be used for cases that preserve the categories. "
                "To change the categories, use ser.cat.rename_categories "
                "instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if not inplace:
            return cat

    # ------------------------------------------------------------------------
    # String methods interface
    def _str_map(
        self, f, na_value=lib.no_default, dtype=np.dtype("object"), convert: bool = True
    ):
        # Optimization to apply the callable `f` to the categories once
        # and rebuild the result by `take`ing from the result with the codes.
        # Returns the same type as the object-dtype implementation though.
        categories = self.categories
        codes = self.codes
        if categories.dtype == "string":
            result = categories.array._str_map(f, na_value, dtype)  # type: ignore[attr-defined]
            if (
                categories.dtype.na_value is np.nan  # type: ignore[union-attr]
                and is_bool_dtype(dtype)
                and (na_value is lib.no_default or isna(na_value))
            ):
                # NaN propagates as False for functions with boolean return type
                na_value = False
        else:
            from pandas.core.arrays import NumpyExtensionArray

            result = NumpyExtensionArray(categories.to_numpy())._str_map(
                f, na_value, dtype
            )
        return take_nd(result, codes, fill_value=na_value)

    def _str_get_dummies(self, sep: str = "|"):
        # sep may not be in categories. Just bail on this.
        from pandas.core.arrays import NumpyExtensionArray

        return NumpyExtensionArray(self.to_numpy(str, na_value="NaN"))._str_get_dummies(
            sep
        )

    # ------------------------------------------------------------------------
    # GroupBy Methods

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
    ):
        from pandas.core.groupby.ops import WrappedCythonOp

        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        dtype = self.dtype
        if how in ["sum", "prod", "cumsum", "cumprod", "skew"]:
            raise TypeError(f"{dtype} type does not support {how} operations")
        if how in ["min", "max", "rank", "idxmin", "idxmax"] and not dtype.ordered:
            # raise TypeError instead of NotImplementedError to ensure we
            #  don't go down a group-by-group path, since in the empty-groups
            #  case that would fail to raise
            raise TypeError(f"Cannot perform {how} with non-ordered Categorical")
        if how not in [
            "rank",
            "any",
            "all",
            "first",
            "last",
            "min",
            "max",
            "idxmin",
            "idxmax",
        ]:
            if kind == "transform":
                raise TypeError(f"{dtype} type does not support {how} operations")
            raise TypeError(f"{dtype} dtype does not support aggregation '{how}'")

        result_mask = None
        mask = self.isna()
        if how == "rank":
            assert self.ordered  # checked earlier
            npvalues = self._ndarray
        elif how in ["first", "last", "min", "max", "idxmin", "idxmax"]:
            npvalues = self._ndarray
            result_mask = np.zeros(ngroups, dtype=bool)
        else:
            # any/all
            npvalues = self.astype(bool)

        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=mask,
            result_mask=result_mask,
            **kwargs,
        )

        if how in op.cast_blocklist:
            return res_values
        elif how in ["first", "last", "min", "max"]:
            res_values[result_mask == 1] = -1
        return self._from_backing_data(res_values)


# The Series.cat accessor


@delegate_names(
    delegate=Categorical, accessors=["categories", "ordered"], typ="property"
)
@delegate_names(
    delegate=Categorical,
    accessors=[
        "rename_categories",
        "reorder_categories",
        "add_categories",
        "remove_categories",
        "remove_unused_categories",
        "set_categories",
        "as_ordered",
        "as_unordered",
    ],
    typ="method",
)
class CategoricalAccessor(PandasDelegate, PandasObject, NoNewAttributesMixin):
    """
    Accessor object for categorical properties of the Series values.

    Parameters
    ----------
    data : Series or CategoricalIndex

    Examples
    --------
    >>> s = pd.Series(list("abbccc")).astype("category")
    >>> s
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']

    >>> s.cat.categories
    Index(['a', 'b', 'c'], dtype='object')

    >>> s.cat.rename_categories(list("cba"))
    0    c
    1    b
    2    b
    3    a
    4    a
    5    a
    dtype: category
    Categories (3, object): ['c', 'b', 'a']

    >>> s.cat.reorder_categories(list("cba"))
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['c', 'b', 'a']

    >>> s.cat.add_categories(["d", "e"])
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (5, object): ['a', 'b', 'c', 'd', 'e']

    >>> s.cat.remove_categories(["a", "c"])
    0    NaN
    1      b
    2      b
    3    NaN
    4    NaN
    5    NaN
    dtype: category
    Categories (1, object): ['b']

    >>> s1 = s.cat.add_categories(["d", "e"])
    >>> s1.cat.remove_unused_categories()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']

    >>> s.cat.set_categories(list("abcde"))
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (5, object): ['a', 'b', 'c', 'd', 'e']

    >>> s.cat.as_ordered()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a' < 'b' < 'c']

    >>> s.cat.as_unordered()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']
    """

    def __init__(self, data) -> None:
        self._validate(data)
        self._parent = data.values
        self._index = data.index
        self._name = data.name
        self._freeze()

    @staticmethod
    def _validate(data):
        if not isinstance(data.dtype, CategoricalDtype):
            raise AttributeError("Can only use .cat accessor with a 'category' dtype")

    def _delegate_property_get(self, name: str):
        return getattr(self._parent, name)

    # error: Signature of "_delegate_property_set" incompatible with supertype
    # "PandasDelegate"
    def _delegate_property_set(self, name: str, new_values):  # type: ignore[override]
        return setattr(self._parent, name, new_values)

    @property
    def codes(self) -> Series:
        """
        Return Series of codes as well as the index.

        Examples
        --------
        >>> raw_cate = pd.Categorical(["a", "b", "c", "a"], categories=["a", "b"])
        >>> ser = pd.Series(raw_cate)
        >>> ser.cat.codes
        0   0
        1   1
        2  -1
        3   0
        dtype: int8
        """
        from pandas import Series

        return Series(self._parent.codes, index=self._index)

    def _delegate_method(self, name: str, *args, **kwargs):
        from pandas import Series

        method = getattr(self._parent, name)
        res = method(*args, **kwargs)
        if res is not None:
            return Series(res, index=self._index, name=self._name)


# utility routines


def _get_codes_for_values(
    values: Index | Series | ExtensionArray | np.ndarray,
    categories: Index,
) -> np.ndarray:
    """
    utility routine to turn values into codes given the specified categories

    If `values` is known to be a Categorical, use recode_for_categories instead.
    """
    codes = categories.get_indexer_for(values)
    return coerce_indexer_dtype(codes, categories)


def recode_for_categories(
    codes: np.ndarray, old_categories, new_categories, copy: bool = True
) -> np.ndarray:
    """
    Convert a set of codes for to a new set of categories

    Parameters
    ----------
    codes : np.ndarray
    old_categories, new_categories : Index
    copy: bool, default True
        Whether to copy if the codes are unchanged.

    Returns
    -------
    new_codes : np.ndarray[np.int64]

    Examples
    --------
    >>> old_cat = pd.Index(['b', 'a', 'c'])
    >>> new_cat = pd.Index(['a', 'b'])
    >>> codes = np.array([0, 1, 1, 2])
    >>> recode_for_categories(codes, old_cat, new_cat)
    array([ 1,  0,  0, -1], dtype=int8)
    """
    if len(old_categories) == 0:
        # All null anyway, so just retain the nulls
        if copy:
            return codes.copy()
        return codes
    elif new_categories.equals(old_categories):
        # Same categories, so no need to actually recode
        if copy:
            return codes.copy()
        return codes

    indexer = coerce_indexer_dtype(
        new_categories.get_indexer_for(old_categories), new_categories
    )
    new_codes = take_nd(indexer, codes, fill_value=-1)
    return new_codes


def factorize_from_iterable(values) -> tuple[np.ndarray, Index]:
    """
    Factorize an input `values` into `categories` and `codes`. Preserves
    categorical dtype in `categories`.

    Parameters
    ----------
    values : list-like

    Returns
    -------
    codes : ndarray
    categories : Index
        If `values` has a categorical dtype, then `categories` is
        a CategoricalIndex keeping the categories and order of `values`.
    """
    from pandas import CategoricalIndex

    if not is_list_like(values):
        raise TypeError("Input must be list-like")

    categories: Index

    vdtype = getattr(values, "dtype", None)
    if isinstance(vdtype, CategoricalDtype):
        values = extract_array(values)
        # The Categorical we want to build has the same categories
        # as values but its codes are by def [0, ..., len(n_categories) - 1]
        cat_codes = np.arange(len(values.categories), dtype=values.codes.dtype)
        cat = Categorical.from_codes(cat_codes, dtype=values.dtype, validate=False)

        categories = CategoricalIndex(cat)
        codes = values.codes
    else:
        # The value of ordered is irrelevant since we don't use cat as such,
        # but only the resulting categories, the order of which is independent
        # from ordered. Set ordered to False as default. See GH #15457
        cat = Categorical(values, ordered=False)
        categories = cat.categories
        codes = cat.codes
    return codes, categories


def factorize_from_iterables(iterables) -> tuple[list[np.ndarray], list[Index]]:
    """
    A higher-level wrapper over `factorize_from_iterable`.

    Parameters
    ----------
    iterables : list-like of list-likes

    Returns
    -------
    codes : list of ndarrays
    categories : list of Indexes

    Notes
    -----
    See `factorize_from_iterable` for more info.
    """
    if len(iterables) == 0:
        # For consistency, it should return two empty lists.
        return [], []

    codes, categories = zip(*(factorize_from_iterable(it) for it in iterables))
    return list(codes), list(categories)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\rings.py ===
"""Sparse polynomial rings. """

from __future__ import annotations

from operator import add, mul, lt, le, gt, ge
from functools import reduce
from types import GeneratorType

from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.intfunc import igcd
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify, sympify
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.compatibility import IPolys
from sympy.polys.constructor import construct_domain
from sympy.polys.densebasic import ninf, dmp_to_dict, dmp_from_dict
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.heuristicgcd import heugcd
from sympy.polys.monomials import MonomialOps
from sympy.polys.orderings import lex, MonomialOrder
from sympy.polys.polyerrors import (
    CoercionFailed, GeneratorsError,
    ExactQuotientFailed, MultivariatePolynomialError)
from sympy.polys.polyoptions import (Domain as DomainOpt,
                                     Order as OrderOpt, build_options)
from sympy.polys.polyutils import (expr_from_dict, _dict_reorder,
                                   _parallel_dict_from_expr)
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public, subsets
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute

@public
def ring(symbols, domain, order: MonomialOrder|str = lex):
    """Construct a polynomial ring returning ``(ring, x_1, ..., x_n)``.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
    domain : :class:`~.Domain` or coercible
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> R, x, y, z = ring("x,y,z", ZZ, lex)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)
    return (_ring,) + _ring.gens

@public
def xring(symbols, domain, order=lex):
    """Construct a polynomial ring returning ``(ring, (x_1, ..., x_n))``.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
    domain : :class:`~.Domain` or coercible
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import xring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> R, (x, y, z) = xring("x,y,z", ZZ, lex)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)
    return (_ring, _ring.gens)

@public
def vring(symbols, domain, order=lex):
    """Construct a polynomial ring and inject ``x_1, ..., x_n`` into the global namespace.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
    domain : :class:`~.Domain` or coercible
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import vring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> vring("x,y,z", ZZ, lex)
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z # noqa:
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)
    pollute([ sym.name for sym in _ring.symbols ], _ring.gens)
    return _ring

@public
def sring(exprs, *symbols, **options):
    """Construct a ring deriving generators and domain from options and input expressions.

    Parameters
    ==========

    exprs : :class:`~.Expr` or sequence of :class:`~.Expr` (sympifiable)
    symbols : sequence of :class:`~.Symbol`/:class:`~.Expr`
    options : keyword arguments understood by :class:`~.Options`

    Examples
    ========

    >>> from sympy import sring, symbols

    >>> x, y, z = symbols("x,y,z")
    >>> R, f = sring(x + 2*y + 3*z)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> f
    x + 2*y + 3*z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    single = False

    if not is_sequence(exprs):
        exprs, single = [exprs], True

    exprs = list(map(sympify, exprs))
    opt = build_options(symbols, options)

    # TODO: rewrite this so that it doesn't use expand() (see poly()).
    reps, opt = _parallel_dict_from_expr(exprs, opt)

    if opt.domain is None:
        coeffs = sum([ list(rep.values()) for rep in reps ], [])

        opt.domain, coeffs_dom = construct_domain(coeffs, opt=opt)

        coeff_map = dict(zip(coeffs, coeffs_dom))
        reps = [{m: coeff_map[c] for m, c in rep.items()} for rep in reps]

    _ring = PolyRing(opt.gens, opt.domain, opt.order)
    polys = list(map(_ring.from_dict, reps))

    if single:
        return (_ring, polys[0])
    else:
        return (_ring, polys)

def _parse_symbols(symbols):
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True) if symbols else ()
    elif isinstance(symbols, Expr):
        return (symbols,)
    elif is_sequence(symbols):
        if all(isinstance(s, str) for s in symbols):
            return _symbols(symbols)
        elif all(isinstance(s, Expr) for s in symbols):
            return symbols

    raise GeneratorsError("expected a string, Symbol or expression or a non-empty sequence of strings, Symbols or expressions")


class PolyRing(DefaultPrinting, IPolys):
    """Multivariate distributed polynomial ring. """

    gens: tuple[PolyElement, ...]
    symbols: tuple[Expr, ...]
    ngens: int
    domain: Domain
    order: MonomialOrder

    def __new__(cls, symbols, domain, order=lex):
        symbols = tuple(_parse_symbols(symbols))
        ngens = len(symbols)
        domain = DomainOpt.preprocess(domain)
        order = OrderOpt.preprocess(order)

        _hash_tuple = (cls.__name__, symbols, ngens, domain, order)

        if domain.is_Composite and set(symbols) & set(domain.symbols):
            raise GeneratorsError("polynomial ring and it's ground domain share generators")

        obj = object.__new__(cls)
        obj._hash_tuple = _hash_tuple
        obj._hash = hash(_hash_tuple)
        obj.symbols = symbols
        obj.ngens = ngens
        obj.domain = domain
        obj.order = order

        obj.dtype = PolyElement(obj, ()).new

        obj.zero_monom = (0,)*ngens
        obj.gens = obj._gens()
        obj._gens_set = set(obj.gens)

        obj._one = [(obj.zero_monom, domain.one)]

        if ngens:
            # These expect monomials in at least one variable
            codegen = MonomialOps(ngens)
            obj.monomial_mul = codegen.mul()
            obj.monomial_pow = codegen.pow()
            obj.monomial_mulpow = codegen.mulpow()
            obj.monomial_ldiv = codegen.ldiv()
            obj.monomial_div = codegen.div()
            obj.monomial_lcm = codegen.lcm()
            obj.monomial_gcd = codegen.gcd()
        else:
            monunit = lambda a, b: ()
            obj.monomial_mul = monunit
            obj.monomial_pow = monunit
            obj.monomial_mulpow = lambda a, b, c: ()
            obj.monomial_ldiv = monunit
            obj.monomial_div = monunit
            obj.monomial_lcm = monunit
            obj.monomial_gcd = monunit


        if order is lex:
            obj.leading_expv = max
        else:
            obj.leading_expv = lambda f: max(f, key=order)

        for symbol, generator in zip(obj.symbols, obj.gens):
            if isinstance(symbol, Symbol):
                name = symbol.name

                if not hasattr(obj, name):
                    setattr(obj, name, generator)

        return obj

    def _gens(self):
        """Return a list of polynomial generators. """
        one = self.domain.one
        _gens = []
        for i in range(self.ngens):
            expv = self.monomial_basis(i)
            poly = self.zero
            poly[expv] = one
            _gens.append(poly)
        return tuple(_gens)

    def __getnewargs__(self):
        return (self.symbols, self.domain, self.order)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["leading_expv"]

        for key in state:
            if key.startswith("monomial_"):
                del state[key]

        return state

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, PolyRing) and \
            (self.symbols, self.domain, self.ngens, self.order) == \
            (other.symbols, other.domain, other.ngens, other.order)

    def __ne__(self, other):
        return not self == other

    def clone(self, symbols=None, domain=None, order=None):
        # Need a hashable tuple for cacheit to work
        if symbols is not None and isinstance(symbols, list):
            symbols = tuple(symbols)
        return self._clone(symbols, domain, order)

    @cacheit
    def _clone(self, symbols, domain, order):
        return self.__class__(symbols or self.symbols, domain or self.domain, order or self.order)

    def monomial_basis(self, i):
        """Return the ith-basis element. """
        basis = [0]*self.ngens
        basis[i] = 1
        return tuple(basis)

    @property
    def zero(self):
        return self.dtype([])

    @property
    def one(self):
        return self.dtype(self._one)

    def is_element(self, element):
        """True if ``element`` is an element of this ring. False otherwise. """
        return isinstance(element, PolyElement) and element.ring == self

    def domain_new(self, element, orig_domain=None):
        return self.domain.convert(element, orig_domain)

    def ground_new(self, coeff):
        return self.term_new(self.zero_monom, coeff)

    def term_new(self, monom, coeff):
        coeff = self.domain_new(coeff)
        poly = self.zero
        if coeff:
            poly[monom] = coeff
        return poly

    def ring_new(self, element):
        if isinstance(element, PolyElement):
            if self == element.ring:
                return element
            elif isinstance(self.domain, PolynomialRing) and self.domain.ring == element.ring:
                return self.ground_new(element)
            else:
                raise NotImplementedError("conversion")
        elif isinstance(element, str):
            raise NotImplementedError("parsing")
        elif isinstance(element, dict):
            return self.from_dict(element)
        elif isinstance(element, list):
            try:
                return self.from_terms(element)
            except ValueError:
                return self.from_list(element)
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)

    __call__ = ring_new

    def from_dict(self, element, orig_domain=None):
        domain_new = self.domain_new
        poly = self.zero

        for monom, coeff in element.items():
            coeff = domain_new(coeff, orig_domain)
            if coeff:
                poly[monom] = coeff

        return poly

    def from_terms(self, element, orig_domain=None):
        return self.from_dict(dict(element), orig_domain)

    def from_list(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain))

    def _rebuild_expr(self, expr, mapping):
        domain = self.domain

        def _rebuild(expr):
            generator = mapping.get(expr)

            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            else:
                # XXX: Use as_base_exp() to handle Pow(x, n) and also exp(n)
                # XXX: E can be a generator e.g. sring([exp(2)]) -> ZZ[E]
                base, exp = expr.as_base_exp()
                if exp.is_Integer and exp > 1:
                    return _rebuild(base)**int(exp)
                else:
                    return self.ground_new(domain.convert(expr))

        return _rebuild(sympify(expr))

    def from_expr(self, expr):
        mapping = dict(list(zip(self.symbols, self.gens)))

        try:
            poly = self._rebuild_expr(expr, mapping)
        except CoercionFailed:
            raise ValueError("expected an expression convertible to a polynomial in %s, got %s" % (self, expr))
        else:
            return self.ring_new(poly)

    def index(self, gen):
        """Compute index of ``gen`` in ``self.gens``. """
        if gen is None:
            if self.ngens:
                i = 0
            else:
                i = -1  # indicate impossible choice
        elif isinstance(gen, int):
            i = gen

            if 0 <= i and i < self.ngens:
                pass
            elif -self.ngens <= i and i <= -1:
                i = -i - 1
            else:
                raise ValueError("invalid generator index: %s" % gen)
        elif self.is_element(gen):
            try:
                i = self.gens.index(gen)
            except ValueError:
                raise ValueError("invalid generator: %s" % gen)
        elif isinstance(gen, str):
            try:
                i = self.symbols.index(gen)
            except ValueError:
                raise ValueError("invalid generator: %s" % gen)
        else:
            raise ValueError("expected a polynomial generator, an integer, a string or None, got %s" % gen)

        return i

    def drop(self, *gens):
        """Remove specified generators from this ring. """
        indices = set(map(self.index, gens))
        symbols = [ s for i, s in enumerate(self.symbols) if i not in indices ]

        if not symbols:
            return self.domain
        else:
            return self.clone(symbols=symbols)

    def __getitem__(self, key):
        symbols = self.symbols[key]

        if not symbols:
            return self.domain
        else:
            return self.clone(symbols=symbols)

    def to_ground(self):
        # TODO: should AlgebraicField be a Composite domain?
        if self.domain.is_Composite or hasattr(self.domain, 'domain'):
            return self.clone(domain=self.domain.domain)
        else:
            raise ValueError("%s is not a composite domain" % self.domain)

    def to_domain(self):
        return PolynomialRing(self)

    def to_field(self):
        from sympy.polys.fields import FracField
        return FracField(self.symbols, self.domain, self.order)

    @property
    def is_univariate(self):
        return len(self.gens) == 1

    @property
    def is_multivariate(self):
        return len(self.gens) > 1

    def add(self, *objs):
        """
        Add a sequence of polynomials or containers of polynomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> R, x = ring("x", ZZ)
        >>> R.add([ x**2 + 2*i + 3 for i in range(4) ])
        4*x**2 + 24
        >>> _.factor_list()
        (4, [(x**2 + 6, 1)])

        """
        p = self.zero

        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                p += self.add(*obj)
            else:
                p += obj

        return p

    def mul(self, *objs):
        """
        Multiply a sequence of polynomials or containers of polynomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> R, x = ring("x", ZZ)
        >>> R.mul([ x**2 + 2*i + 3 for i in range(4) ])
        x**8 + 24*x**6 + 206*x**4 + 744*x**2 + 945
        >>> _.factor_list()
        (1, [(x**2 + 3, 1), (x**2 + 5, 1), (x**2 + 7, 1), (x**2 + 9, 1)])

        """
        p = self.one

        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                p *= self.mul(*obj)
            else:
                p *= obj

        return p

    def drop_to_ground(self, *gens):
        r"""
        Remove specified generators from the ring and inject them into
        its domain.
        """
        indices = set(map(self.index, gens))
        symbols = [s for i, s in enumerate(self.symbols) if i not in indices]
        gens = [gen for i, gen in enumerate(self.gens) if i not in indices]

        if not symbols:
            return self
        else:
            return self.clone(symbols=symbols, domain=self.drop(*gens))

    def compose(self, other):
        """Add the generators of ``other`` to ``self``"""
        if self != other:
            syms = set(self.symbols).union(set(other.symbols))
            return self.clone(symbols=list(syms))
        else:
            return self

    def add_gens(self, symbols):
        """Add the elements of ``symbols`` as generators to ``self``"""
        syms = set(self.symbols).union(set(symbols))
        return self.clone(symbols=list(syms))

    def symmetric_poly(self, n):
        """
        Return the elementary symmetric polynomial of degree *n* over
        this ring's generators.
        """
        if n < 0 or n > self.ngens:
            raise ValueError("Cannot generate symmetric polynomial of order %s for %s" % (n, self.gens))
        elif not n:
            return self.one
        else:
            poly = self.zero
            for s in subsets(range(self.ngens), int(n)):
                monom = tuple(int(i in s) for i in range(self.ngens))
                poly += self.term_new(monom, self.domain.one)
            return poly


class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):
    """Element of multivariate distributed polynomial ring. """

    def __init__(self, ring, init):
        super().__init__(init)
        self.ring = ring
        # This check would be too slow to run every time:
        # self._check()

    def _check(self):
        assert isinstance(self, PolyElement)
        assert isinstance(self.ring, PolyRing)
        dom = self.ring.domain
        assert isinstance(dom, Domain)
        for monom, coeff in self.items():
            assert dom.of_type(coeff)
            assert len(monom) == self.ring.ngens
            assert all(isinstance(exp, int) and exp >= 0 for exp in monom)

    def new(self, init):
        return self.__class__(self.ring, init)

    def parent(self):
        return self.ring.to_domain()

    def __getnewargs__(self):
        return (self.ring, list(self.iterterms()))

    _hash = None

    def __hash__(self):
        # XXX: This computes a hash of a dictionary, but currently we don't
        # protect dictionary from being changed so any use site modifications
        # will make hashing go wrong. Use this feature with caution until we
        # figure out how to make a safe API without compromising speed of this
        # low-level class.
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.ring, frozenset(self.items())))
        return _hash

    def copy(self):
        """Return a copy of polynomial self.

        Polynomials are mutable; if one is interested in preserving
        a polynomial, and one plans to use inplace operations, one
        can copy the polynomial. This method makes a shallow copy.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> R, x, y = ring('x, y', ZZ)
        >>> p = (x + y)**2
        >>> p1 = p.copy()
        >>> p2 = p
        >>> p[R.zero_monom] = 3
        >>> p
        x**2 + 2*x*y + y**2 + 3
        >>> p1
        x**2 + 2*x*y + y**2
        >>> p2
        x**2 + 2*x*y + y**2 + 3

        """
        return self.new(self)

    def set_ring(self, new_ring):
        if self.ring == new_ring:
            return self
        elif self.ring.symbols != new_ring.symbols:
            terms = list(zip(*_dict_reorder(self, self.ring.symbols, new_ring.symbols)))
            return new_ring.from_terms(terms, self.ring.domain)
        else:
            return new_ring.from_dict(self, self.ring.domain)

    def as_expr(self, *symbols):
        if not symbols:
            symbols = self.ring.symbols
        elif len(symbols) != self.ring.ngens:
            raise ValueError(
                "Wrong number of symbols, expected %s got %s" %
                (self.ring.ngens, len(symbols))
            )

        return expr_from_dict(self.as_expr_dict(), *symbols)

    def as_expr_dict(self):
        to_sympy = self.ring.domain.to_sympy
        return {monom: to_sympy(coeff) for monom, coeff in self.iterterms()}

    def clear_denoms(self):
        domain = self.ring.domain

        if not domain.is_Field or not domain.has_assoc_Ring:
            return domain.one, self

        ground_ring = domain.get_ring()
        common = ground_ring.one
        lcm = ground_ring.lcm
        denom = domain.denom

        for coeff in self.values():
            common = lcm(common, denom(coeff))

        poly = self.new([ (k, v*common) for k, v in self.items() ])
        return common, poly

    def strip_zero(self):
        """Eliminate monomials with zero coefficient. """
        for k, v in list(self.items()):
            if not v:
                del self[k]

    def __eq__(p1, p2):
        """Equality test for polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = (x + y)**2 + (x - y)**2
        >>> p1 == 4*x*y
        False
        >>> p1 == 2*(x**2 + y**2)
        True

        """
        if not p2:
            return not p1
        elif p1.ring.is_element(p2):
            return dict.__eq__(p1, p2)
        elif len(p1) > 1:
            return False
        else:
            return p1.get(p1.ring.zero_monom) == p2

    def __ne__(p1, p2):
        return not p1 == p2

    def almosteq(p1, p2, tolerance=None):
        """Approximate equality test for polynomials. """
        ring = p1.ring

        if ring.is_element(p2):
            if set(p1.keys()) != set(p2.keys()):
                return False

            almosteq = ring.domain.almosteq

            for k in p1.keys():
                if not almosteq(p1[k], p2[k], tolerance):
                    return False
            return True
        elif len(p1) > 1:
            return False
        else:
            try:
                p2 = ring.domain.convert(p2)
            except CoercionFailed:
                return False
            else:
                return ring.domain.almosteq(p1.const(), p2, tolerance)

    def sort_key(self):
        return (len(self), self.terms())

    def _cmp(p1, p2, op):
        if p1.ring.is_element(p2):
            return op(p1.sort_key(), p2.sort_key())
        else:
            return NotImplemented

    def __lt__(p1, p2):
        return p1._cmp(p2, lt)
    def __le__(p1, p2):
        return p1._cmp(p2, le)
    def __gt__(p1, p2):
        return p1._cmp(p2, gt)
    def __ge__(p1, p2):
        return p1._cmp(p2, ge)

    def _drop(self, gen):
        ring = self.ring
        i = ring.index(gen)

        if ring.ngens == 1:
            return i, ring.domain
        else:
            symbols = list(ring.symbols)
            del symbols[i]
            return i, ring.clone(symbols=symbols)

    def drop(self, gen):
        i, ring = self._drop(gen)

        if self.ring.ngens == 1:
            if self.is_ground:
                return self.coeff(1)
            else:
                raise ValueError("Cannot drop %s" % gen)
        else:
            poly = ring.zero

            for k, v in self.items():
                if k[i] == 0:
                    K = list(k)
                    del K[i]
                    poly[tuple(K)] = v
                else:
                    raise ValueError("Cannot drop %s" % gen)

            return poly

    def _drop_to_ground(self, gen):
        ring = self.ring
        i = ring.index(gen)

        symbols = list(ring.symbols)
        del symbols[i]
        return i, ring.clone(symbols=symbols, domain=ring[i])

    def drop_to_ground(self, gen):
        if self.ring.ngens == 1:
            raise ValueError("Cannot drop only generator to ground")

        i, ring = self._drop_to_ground(gen)
        poly = ring.zero
        gen = ring.domain.gens[0]

        for monom, coeff in self.iterterms():
            mon = monom[:i] + monom[i+1:]
            if mon not in poly:
                poly[mon] = (gen**monom[i]).mul_ground(coeff)
            else:
                poly[mon] += (gen**monom[i]).mul_ground(coeff)

        return poly

    def to_dense(self):
        return dmp_from_dict(self, self.ring.ngens-1, self.ring.domain)

    def to_dict(self):
        return dict(self)

    def str(self, printer, precedence, exp_pattern, mul_symbol):
        if not self:
            return printer._print(self.ring.domain.zero)
        prec_mul = precedence["Mul"]
        prec_atom = precedence["Atom"]
        ring = self.ring
        symbols = ring.symbols
        ngens = ring.ngens
        zm = ring.zero_monom
        sexpvs = []
        for expv, coeff in self.terms():
            negative = ring.domain.is_negative(coeff)
            sign = " - " if negative else " + "
            sexpvs.append(sign)
            if expv == zm:
                scoeff = printer._print(coeff)
                if negative and scoeff.startswith("-"):
                    scoeff = scoeff[1:]
            else:
                if negative:
                    coeff = -coeff
                if coeff != self.ring.domain.one:
                    scoeff = printer.parenthesize(coeff, prec_mul, strict=True)
                else:
                    scoeff = ''
            sexpv = []
            for i in range(ngens):
                exp = expv[i]
                if not exp:
                    continue
                symbol = printer.parenthesize(symbols[i], prec_atom, strict=True)
                if exp != 1:
                    if exp != int(exp) or exp < 0:
                        sexp = printer.parenthesize(exp, prec_atom, strict=False)
                    else:
                        sexp = exp
                    sexpv.append(exp_pattern % (symbol, sexp))
                else:
                    sexpv.append('%s' % symbol)
            if scoeff:
                sexpv = [scoeff] + sexpv
            sexpvs.append(mul_symbol.join(sexpv))
        if sexpvs[0] in [" + ", " - "]:
            head = sexpvs.pop(0)
            if head == " - ":
                sexpvs.insert(0, "-")
        return "".join(sexpvs)

    @property
    def is_generator(self):
        return self in self.ring._gens_set

    @property
    def is_ground(self):
        return not self or (len(self) == 1 and self.ring.zero_monom in self)

    @property
    def is_monomial(self):
        return not self or (len(self) == 1 and self.LC == 1)

    @property
    def is_term(self):
        return len(self) <= 1

    @property
    def is_negative(self):
        return self.ring.domain.is_negative(self.LC)

    @property
    def is_positive(self):
        return self.ring.domain.is_positive(self.LC)

    @property
    def is_nonnegative(self):
        return self.ring.domain.is_nonnegative(self.LC)

    @property
    def is_nonpositive(self):
        return self.ring.domain.is_nonpositive(self.LC)

    @property
    def is_zero(f):
        return not f

    @property
    def is_one(f):
        return f == f.ring.one

    @property
    def is_monic(f):
        return f.ring.domain.is_one(f.LC)

    @property
    def is_primitive(f):
        return f.ring.domain.is_one(f.content())

    @property
    def is_linear(f):
        return all(sum(monom) <= 1 for monom in f.itermonoms())

    @property
    def is_quadratic(f):
        return all(sum(monom) <= 2 for monom in f.itermonoms())

    @property
    def is_squarefree(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_sqf_p(f)

    @property
    def is_irreducible(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_irreducible_p(f)

    @property
    def is_cyclotomic(f):
        if f.ring.is_univariate:
            return f.ring.dup_cyclotomic_p(f)
        else:
            raise MultivariatePolynomialError("cyclotomic polynomial")

    def __neg__(self):
        return self.new([ (monom, -coeff) for monom, coeff in self.iterterms() ])

    def __pos__(self):
        return self

    def __add__(p1, p2):
        """Add two polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> (x + y)**2 + (x - y)**2
        2*x**2 + 2*y**2

        """
        if not p2:
            return p1.copy()
        ring = p1.ring
        if ring.is_element(p2):
            p = p1.copy()
            get = p.get
            zero = ring.domain.zero
            for k, v in p2.items():
                v = get(k, zero) + v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__radd__(p1)
            else:
                return NotImplemented

        try:
            cp2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()
            if not cp2:
                return p
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = cp2
            else:
                if p2 == -p[zm]:
                    del p[zm]
                else:
                    p[zm] += cp2
            return p

    def __radd__(p1, n):
        p = p1.copy()
        if not n:
            return p
        ring = p1.ring
        try:
            n = ring.domain_new(n)
        except CoercionFailed:
            return NotImplemented
        else:
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = n
            else:
                if n == -p[zm]:
                    del p[zm]
                else:
                    p[zm] += n
            return p

    def __sub__(p1, p2):
        """Subtract polynomial p2 from p1.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = x + y**2
        >>> p2 = x*y + y**2
        >>> p1 - p2
        -x*y + x

        """
        if not p2:
            return p1.copy()
        ring = p1.ring
        if ring.is_element(p2):
            p = p1.copy()
            get = p.get
            zero = ring.domain.zero
            for k, v in p2.items():
                v = get(k, zero) - v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rsub__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = -p2
            else:
                if p2 == p[zm]:
                    del p[zm]
                else:
                    p[zm] -= p2
            return p

    def __rsub__(p1, n):
        """n - p1 with n convertible to the coefficient domain.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y
        >>> 4 - p
        -x - y + 4

        """
        ring = p1.ring
        try:
            n = ring.domain_new(n)
        except CoercionFailed:
            return NotImplemented
        else:
            p = ring.zero
            for expv in p1:
                p[expv] = -p1[expv]
            p += n
            # p._check()
            return p

    def __mul__(p1, p2):
        """Multiply two polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', QQ)
        >>> p1 = x + y
        >>> p2 = x - y
        >>> p1*p2
        x**2 - y**2

        """
        ring = p1.ring
        p = ring.zero
        if not p1 or not p2:
            return p
        elif ring.is_element(p2):
            get = p.get
            zero = ring.domain.zero
            monomial_mul = ring.monomial_mul
            p2it = list(p2.items())
            for exp1, v1 in p1.items():
                for exp2, v2 in p2it:
                    exp = monomial_mul(exp1, exp2)
                    p[exp] = get(exp, zero) + v1*v2
            p.strip_zero()
            # p._check()
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmul__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            for exp1, v1 in p1.items():
                v = v1*p2
                if v:
                    p[exp1] = v
            # p._check()
            return p

    def __rmul__(p1, p2):
        """p2 * p1 with p2 in the coefficient domain of p1.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y
        >>> 4 * p
        4*x + 4*y

        """
        p = p1.ring.zero
        if not p2:
            return p
        try:
            p2 = p.ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            for exp1, v1 in p1.items():
                v = p2*v1
                if v:
                    p[exp1] = v
            return p

    def __pow__(self, n):
        """raise polynomial to power `n`

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p**3
        x**3 + 3*x**2*y**2 + 3*x*y**4 + y**6

        """
        if not isinstance(n, int):
            raise TypeError("exponent must be an integer, got %s" % n)
        elif n < 0:
            raise ValueError("exponent must be a non-negative integer, got %s" % n)

        ring = self.ring

        if not n:
            if self:
                return ring.one
            else:
                raise ValueError("0**0")
        elif len(self) == 1:
            monom, coeff = list(self.items())[0]
            p = ring.zero
            if coeff == ring.domain.one:
                p[ring.monomial_pow(monom, n)] = coeff
            else:
                p[ring.monomial_pow(monom, n)] = coeff**n
            # p._check()
            return p

        # For ring series, we need negative and rational exponent support only
        # with monomials.
        n = int(n)
        if n < 0:
            raise ValueError("Negative exponent")

        elif n == 1:
            return self.copy()
        elif n == 2:
            return self.square()
        elif n == 3:
            return self*self.square()
        elif len(self) <= 5: # TODO: use an actual density measure
            return self._pow_multinomial(n)
        else:
            return self._pow_generic(n)

    def _pow_generic(self, n):
        p = self.ring.one
        c = self

        while True:
            if n & 1:
                p = p*c
                n -= 1
                if not n:
                    break

            c = c.square()
            n = n // 2

        return p

    def _pow_multinomial(self, n):
        multinomials = multinomial_coefficients(len(self), n).items()
        monomial_mulpow = self.ring.monomial_mulpow
        zero_monom = self.ring.zero_monom
        terms = self.items()
        zero = self.ring.domain.zero
        poly = self.ring.zero

        for multinomial, multinomial_coeff in multinomials:
            product_monom = zero_monom
            product_coeff = multinomial_coeff

            for exp, (monom, coeff) in zip(multinomial, terms):
                if exp:
                    product_monom = monomial_mulpow(product_monom, monom, exp)
                    product_coeff *= coeff**exp

            monom = tuple(product_monom)
            coeff = product_coeff

            coeff = poly.get(monom, zero) + coeff

            if coeff:
                poly[monom] = coeff
            elif monom in poly:
                del poly[monom]

        return poly

    def square(self):
        """square of a polynomial

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p.square()
        x**2 + 2*x*y**2 + y**4

        """
        ring = self.ring
        p = ring.zero
        get = p.get
        keys = list(self.keys())
        zero = ring.domain.zero
        monomial_mul = ring.monomial_mul
        for i in range(len(keys)):
            k1 = keys[i]
            pk = self[k1]
            for j in range(i):
                k2 = keys[j]
                exp = monomial_mul(k1, k2)
                p[exp] = get(exp, zero) + pk*self[k2]
        p = p.imul_num(2)
        get = p.get
        for k, v in self.items():
            k2 = monomial_mul(k, k)
            p[k2] = get(k2, zero) + v**2
        p.strip_zero()
        # p._check()
        return p

    def __divmod__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(p2):
            return p1.div(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rdivmod__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return (p1.quo_ground(p2), p1.rem_ground(p2))

    def __rdivmod__(p1, p2):
        ring = p1.ring
        try:
            p2 = ring.ground_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p2.div(p1)

    def __mod__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(p2):
            return p1.rem(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmod__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.rem_ground(p2)

    def __rmod__(p1, p2):
        ring = p1.ring
        try:
            p2 = ring.ground_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p2.rem(p1)

    def __floordiv__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(p2):
            return p1.quo(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rtruediv__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.quo_ground(p2)

    def __rfloordiv__(p1, p2):
        ring = p1.ring
        try:
            p2 = ring.ground_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p2.quo(p1)

    def __truediv__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(p2):
            return p1.exquo(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rtruediv__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.quo_ground(p2)

    def __rtruediv__(p1, p2):
        ring = p1.ring
        try:
            p2 = ring.ground_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p2.exquo(p1)

    def _term_div(self):
        zm = self.ring.zero_monom
        domain = self.ring.domain
        domain_quo = domain.quo
        monomial_div = self.ring.monomial_div

        if domain.is_Field:
            def term_div(a_lm_a_lc, b_lm_b_lc):
                a_lm, a_lc = a_lm_a_lc
                b_lm, b_lc = b_lm_b_lc
                if b_lm == zm: # apparently this is a very common case
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if monom is not None:
                    return monom, domain_quo(a_lc, b_lc)
                else:
                    return None
        else:
            def term_div(a_lm_a_lc, b_lm_b_lc):
                a_lm, a_lc = a_lm_a_lc
                b_lm, b_lc = b_lm_b_lc
                if b_lm == zm: # apparently this is a very common case
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if not (monom is None or a_lc % b_lc):
                    return monom, domain_quo(a_lc, b_lc)
                else:
                    return None

        return term_div

    def div(self, fv):
        """Division algorithm, see [CLO] p64.

        fv array of polynomials
           return qv, r such that
           self = sum(fv[i]*qv[i]) + r

        All polynomials are required not to be Laurent polynomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> f = x**3
        >>> f0 = x - y**2
        >>> f1 = x - y
        >>> qv, r = f.div((f0, f1))
        >>> qv[0]
        x**2 + x*y**2 + y**4
        >>> qv[1]
        0
        >>> r
        y**6

        """
        ring = self.ring
        ret_single = False
        if isinstance(fv, PolyElement):
            ret_single = True
            fv = [fv]
        if not all(fv):
            raise ZeroDivisionError("polynomial division")
        if not self:
            if ret_single:
                return ring.zero, ring.zero
            else:
                return [], ring.zero
        for f in fv:
            if f.ring != ring:
                raise ValueError('self and f must have the same ring')
        s = len(fv)
        qv = [ring.zero for i in range(s)]
        p = self.copy()
        r = ring.zero
        term_div = self._term_div()
        expvs = [fx.leading_expv() for fx in fv]
        while p:
            i = 0
            divoccurred = 0
            while i < s and divoccurred == 0:
                expv = p.leading_expv()
                term = term_div((expv, p[expv]), (expvs[i], fv[i][expvs[i]]))
                if term is not None:
                    expv1, c = term
                    qv[i] = qv[i]._iadd_monom((expv1, c))
                    p = p._iadd_poly_monom(fv[i], (expv1, -c))
                    divoccurred = 1
                else:
                    i += 1
            if not divoccurred:
                expv =  p.leading_expv()
                r = r._iadd_monom((expv, p[expv]))
                del p[expv]
        if expv == ring.zero_monom:
            r += p
        if ret_single:
            if not qv:
                return ring.zero, r
            else:
                return qv[0], r
        else:
            return qv, r

    def rem(self, G):
        f = self
        if isinstance(G, PolyElement):
            G = [G]
        if not all(G):
            raise ZeroDivisionError("polynomial division")
        ring = f.ring
        domain = ring.domain
        zero = domain.zero
        monomial_mul = ring.monomial_mul
        r = ring.zero
        term_div = f._term_div()
        ltf = f.LT
        f = f.copy()
        get = f.get
        while f:
            for g in G:
                tq = term_div(ltf, g.LT)
                if tq is not None:
                    m, c = tq
                    for mg, cg in g.iterterms():
                        m1 = monomial_mul(mg, m)
                        c1 = get(m1, zero) - c*cg
                        if not c1:
                            del f[m1]
                        else:
                            f[m1] = c1
                    ltm = f.leading_expv()
                    if ltm is not None:
                        ltf = ltm, f[ltm]

                    break
            else:
                ltm, ltc = ltf
                if ltm in r:
                    r[ltm] += ltc
                else:
                    r[ltm] = ltc
                del f[ltm]
                ltm = f.leading_expv()
                if ltm is not None:
                    ltf = ltm, f[ltm]

        return r

    def quo(f, G):
        return f.div(G)[0]

    def exquo(f, G):
        q, r = f.div(G)

        if not r:
            return q
        else:
            raise ExactQuotientFailed(f, G)

    def _iadd_monom(self, mc):
        """add to self the monomial coeff*x0**i0*x1**i1*...
        unless self is a generator -- then just return the sum of the two.

        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x**4 + 2*y
        >>> m = (1, 2)
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        x**4 + 5*x*y**2 + 2*y
        >>> p1 is p
        True
        >>> p = x
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        5*x*y**2 + x
        >>> p1 is p
        False

        """
        if self in self.ring._gens_set:
            cpself = self.copy()
        else:
            cpself = self
        expv, coeff = mc
        c = cpself.get(expv)
        if c is None:
            cpself[expv] = coeff
        else:
            c += coeff
            if c:
                cpself[expv] = c
            else:
                del cpself[expv]
        return cpself

    def _iadd_poly_monom(self, p2, mc):
        """add to self the product of (p)*(coeff*x0**i0*x1**i1*...)
        unless self is a generator -- then just return the sum of the two.

        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p1 = x**4 + 2*y
        >>> p2 = y + z
        >>> m = (1, 2, 3)
        >>> p1 = p1._iadd_poly_monom(p2, (m, 3))
        >>> p1
        x**4 + 3*x*y**3*z**3 + 3*x*y**2*z**4 + 2*y

        """
        p1 = self
        if p1 in p1.ring._gens_set:
            p1 = p1.copy()
        (m, c) = mc
        get = p1.get
        zero = p1.ring.domain.zero
        monomial_mul = p1.ring.monomial_mul
        for k, v in p2.items():
            ka = monomial_mul(k, m)
            coeff = get(ka, zero) + v*c
            if coeff:
                p1[ka] = coeff
            else:
                del p1[ka]
        return p1

    def degree(f, x=None):
        """
        The leading degree in ``x`` or the main variable.

        Note that the degree of 0 is negative infinity (``float('-inf')``)

        """
        i = f.ring.index(x)

        if not f:
            return ninf
        elif i < 0:
            return 0
        else:
            return max(monom[i] for monom in f.itermonoms())

    def degrees(f):
        """
        A tuple containing leading degrees in all variables.

        Note that the degree of 0 is negative infinity (``float('-inf')``)

        """
        if not f:
            return (ninf,)*f.ring.ngens
        else:
            return tuple(map(max, list(zip(*f.itermonoms()))))

    def tail_degree(f, x=None):
        """
        The tail degree in ``x`` or the main variable.

        Note that the degree of 0 is negative infinity (``float('-inf')``)

        """
        i = f.ring.index(x)

        if not f:
            return ninf
        elif i < 0:
            return 0
        else:
            return min(monom[i] for monom in f.itermonoms())

    def tail_degrees(f):
        """
        A tuple containing tail degrees in all variables.

        Note that the degree of 0 is negative infinity (``float('-inf')``)

        """
        if not f:
            return (ninf,)*f.ring.ngens
        else:
            return tuple(map(min, list(zip(*f.itermonoms()))))

    def leading_expv(self):
        """Leading monomial tuple according to the monomial ordering.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p = x**4 + x**3*y + x**2*z**2 + z**7
        >>> p.leading_expv()
        (4, 0, 0)

        """
        if self:
            return self.ring.leading_expv(self)
        else:
            return None

    def _get_coeff(self, expv):
        return self.get(expv, self.ring.domain.zero)

    def coeff(self, element):
        """
        Returns the coefficient that stands next to the given monomial.

        Parameters
        ==========

        element : PolyElement (with ``is_monomial = True``) or 1

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring("x,y,z", ZZ)
        >>> f = 3*x**2*y - x*y*z + 7*z**3 + 23

        >>> f.coeff(x**2*y)
        3
        >>> f.coeff(x*y)
        0
        >>> f.coeff(1)
        23

        """
        if element == 1:
            return self._get_coeff(self.ring.zero_monom)
        elif self.ring.is_element(element):
            terms = list(element.iterterms())
            if len(terms) == 1:
                monom, coeff = terms[0]
                if coeff == self.ring.domain.one:
                    return self._get_coeff(monom)

        raise ValueError("expected a monomial, got %s" % element)

    def const(self):
        """Returns the constant coefficient. """
        return self._get_coeff(self.ring.zero_monom)

    @property
    def LC(self):
        return self._get_coeff(self.leading_expv())

    @property
    def LM(self):
        expv = self.leading_expv()
        if expv is None:
            return self.ring.zero_monom
        else:
            return expv

    def leading_monom(self):
        """
        Leading monomial as a polynomial element.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> (3*x*y + y**2).leading_monom()
        x*y

        """
        p = self.ring.zero
        expv = self.leading_expv()
        if expv:
            p[expv] = self.ring.domain.one
        return p

    @property
    def LT(self):
        expv = self.leading_expv()
        if expv is None:
            return (self.ring.zero_monom, self.ring.domain.zero)
        else:
            return (expv, self._get_coeff(expv))

    def leading_term(self):
        """Leading term as a polynomial element.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> (3*x*y + y**2).leading_term()
        3*x*y

        """
        p = self.ring.zero
        expv = self.leading_expv()
        if expv is not None:
            p[expv] = self[expv]
        return p

    def _sorted(self, seq, order):
        if order is None:
            order = self.ring.order
        else:
            order = OrderOpt.preprocess(order)

        if order is lex:
            return sorted(seq, key=lambda monom: monom[0], reverse=True)
        else:
            return sorted(seq, key=lambda monom: order(monom[0]), reverse=True)

    def coeffs(self, order=None):
        """Ordered list of polynomial coefficients.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.coeffs()
        [2, 1]
        >>> f.coeffs(grlex)
        [1, 2]

        """
        return [ coeff for _, coeff in self.terms(order) ]

    def monoms(self, order=None):
        """Ordered list of polynomial monomials.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.monoms()
        [(2, 3), (1, 7)]
        >>> f.monoms(grlex)
        [(1, 7), (2, 3)]

        """
        return [ monom for monom, _ in self.terms(order) ]

    def terms(self, order=None):
        """Ordered list of polynomial terms.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.terms()
        [((2, 3), 2), ((1, 7), 1)]
        >>> f.terms(grlex)
        [((1, 7), 1), ((2, 3), 2)]

        """
        return self._sorted(list(self.items()), order)

    def itercoeffs(self):
        """Iterator over coefficients of a polynomial. """
        return iter(self.values())

    def itermonoms(self):
        """Iterator over monomials of a polynomial. """
        return iter(self.keys())

    def iterterms(self):
        """Iterator over terms of a polynomial. """
        return iter(self.items())

    def listcoeffs(self):
        """Unordered list of polynomial coefficients. """
        return list(self.values())

    def listmonoms(self):
        """Unordered list of polynomial monomials. """
        return list(self.keys())

    def listterms(self):
        """Unordered list of polynomial terms. """
        return list(self.items())

    def imul_num(p, c):
        """multiply inplace the polynomial p by an element in the
        coefficient ring, provided p is not one of the generators;
        else multiply not inplace

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p1 = p.imul_num(3)
        >>> p1
        3*x + 3*y**2
        >>> p1 is p
        True
        >>> p = x
        >>> p1 = p.imul_num(3)
        >>> p1
        3*x
        >>> p1 is p
        False

        """
        if p in p.ring._gens_set:
            return p*c
        if not c:
            p.clear()
            return
        for exp in p:
            p[exp] *= c
        return p

    def content(f):
        """Returns GCD of polynomial's coefficients. """
        domain = f.ring.domain
        cont = domain.zero
        gcd = domain.gcd

        for coeff in f.itercoeffs():
            cont = gcd(cont, coeff)

        return cont

    def primitive(f):
        """Returns content and a primitive polynomial. """
        cont = f.content()
        if cont == f.ring.domain.zero:
            return (cont, f)
        return cont, f.quo_ground(cont)

    def monic(f):
        """Divides all coefficients by the leading coefficient. """
        if not f:
            return f
        else:
            return f.quo_ground(f.LC)

    def mul_ground(f, x):
        if not x:
            return f.ring.zero

        terms = [ (monom, coeff*x) for monom, coeff in f.iterterms() ]
        return f.new(terms)

    def mul_monom(f, monom):
        monomial_mul = f.ring.monomial_mul
        terms = [ (monomial_mul(f_monom, monom), f_coeff) for f_monom, f_coeff in f.items() ]
        return f.new(terms)

    def mul_term(f, term):
        monom, coeff = term

        if not f or not coeff:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.mul_ground(coeff)

        monomial_mul = f.ring.monomial_mul
        terms = [ (monomial_mul(f_monom, monom), f_coeff*coeff) for f_monom, f_coeff in f.items() ]
        return f.new(terms)

    def quo_ground(f, x):
        domain = f.ring.domain

        if not x:
            raise ZeroDivisionError('polynomial division')
        if not f or x == domain.one:
            return f

        if domain.is_Field:
            quo = domain.quo
            terms = [ (monom, quo(coeff, x)) for monom, coeff in f.iterterms() ]
        else:
            terms = [ (monom, coeff // x) for monom, coeff in f.iterterms() if not (coeff % x) ]

        return f.new(terms)

    def quo_term(f, term):
        monom, coeff = term

        if not coeff:
            raise ZeroDivisionError("polynomial division")
        elif not f:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.quo_ground(coeff)

        term_div = f._term_div()

        terms = [ term_div(t, term) for t in f.iterterms() ]
        return f.new([ t for t in terms if t is not None ])

    def trunc_ground(f, p):
        if f.ring.domain.is_ZZ:
            terms = []

            for monom, coeff in f.iterterms():
                coeff = coeff % p

                if coeff > p // 2:
                    coeff = coeff - p

                terms.append((monom, coeff))
        else:
            terms = [ (monom, coeff % p) for monom, coeff in f.iterterms() ]

        poly = f.new(terms)
        poly.strip_zero()
        return poly

    rem_ground = trunc_ground

    def extract_ground(self, g):
        f = self
        fc = f.content()
        gc = g.content()

        gcd = f.ring.domain.gcd(fc, gc)

        f = f.quo_ground(gcd)
        g = g.quo_ground(gcd)

        return gcd, f, g

    def _norm(f, norm_func):
        if not f:
            return f.ring.domain.zero
        else:
            ground_abs = f.ring.domain.abs
            return norm_func([ ground_abs(coeff) for coeff in f.itercoeffs() ])

    def max_norm(f):
        return f._norm(max)

    def l1_norm(f):
        return f._norm(sum)

    def deflate(f, *G):
        ring = f.ring
        polys = [f] + list(G)

        J = [0]*ring.ngens

        for p in polys:
            for monom in p.itermonoms():
                for i, m in enumerate(monom):
                    J[i] = igcd(J[i], m)

        for i, b in enumerate(J):
            if not b:
                J[i] = 1

        J = tuple(J)

        if all(b == 1 for b in J):
            return J, polys

        H = []

        for p in polys:
            h = ring.zero

            for I, coeff in p.iterterms():
                N = [ i // j for i, j in zip(I, J) ]
                h[tuple(N)] = coeff

            H.append(h)

        return J, H

    def inflate(f, J):
        poly = f.ring.zero

        for I, coeff in f.iterterms():
            N = [ i*j for i, j in zip(I, J) ]
            poly[tuple(N)] = coeff

        return poly

    def lcm(self, g):
        f = self
        domain = f.ring.domain

        if not domain.is_Field:
            fc, f = f.primitive()
            gc, g = g.primitive()
            c = domain.lcm(fc, gc)

        h = (f*g).quo(f.gcd(g))

        if not domain.is_Field:
            return h.mul_ground(c)
        else:
            return h.monic()

    def gcd(f, g):
        return f.cofactors(g)[0]

    def cofactors(f, g):
        if not f and not g:
            zero = f.ring.zero
            return zero, zero, zero
        elif not f:
            h, cff, cfg = f._gcd_zero(g)
            return h, cff, cfg
        elif not g:
            h, cfg, cff = g._gcd_zero(f)
            return h, cff, cfg
        elif len(f) == 1:
            h, cff, cfg = f._gcd_monom(g)
            return h, cff, cfg
        elif len(g) == 1:
            h, cfg, cff = g._gcd_monom(f)
            return h, cff, cfg

        J, (f, g) = f.deflate(g)
        h, cff, cfg = f._gcd(g)

        return (h.inflate(J), cff.inflate(J), cfg.inflate(J))

    def _gcd_zero(f, g):
        one, zero = f.ring.one, f.ring.zero
        if g.is_nonnegative:
            return g, zero, one
        else:
            return -g, zero, -one

    def _gcd_monom(f, g):
        ring = f.ring
        ground_gcd = ring.domain.gcd
        ground_quo = ring.domain.quo
        monomial_gcd = ring.monomial_gcd
        monomial_ldiv = ring.monomial_ldiv
        mf, cf = list(f.iterterms())[0]
        _mgcd, _cgcd = mf, cf
        for mg, cg in g.iterterms():
            _mgcd = monomial_gcd(_mgcd, mg)
            _cgcd = ground_gcd(_cgcd, cg)
        h = f.new([(_mgcd, _cgcd)])
        cff = f.new([(monomial_ldiv(mf, _mgcd), ground_quo(cf, _cgcd))])
        cfg = f.new([(monomial_ldiv(mg, _mgcd), ground_quo(cg, _cgcd)) for mg, cg in g.iterterms()])
        return h, cff, cfg

    def _gcd(f, g):
        ring = f.ring

        if ring.domain.is_QQ:
            return f._gcd_QQ(g)
        elif ring.domain.is_ZZ:
            return f._gcd_ZZ(g)
        else: # TODO: don't use dense representation (port PRS algorithms)
            return ring.dmp_inner_gcd(f, g)

    def _gcd_ZZ(f, g):
        return heugcd(f, g)

    def _gcd_QQ(self, g):
        f = self
        ring = f.ring
        new_ring = ring.clone(domain=ring.domain.get_ring())

        cf, f = f.clear_denoms()
        cg, g = g.clear_denoms()

        f = f.set_ring(new_ring)
        g = g.set_ring(new_ring)

        h, cff, cfg = f._gcd_ZZ(g)

        h = h.set_ring(ring)
        c, h = h.LC, h.monic()

        cff = cff.set_ring(ring).mul_ground(ring.domain.quo(c, cf))
        cfg = cfg.set_ring(ring).mul_ground(ring.domain.quo(c, cg))

        return h, cff, cfg

    def cancel(self, g):
        """
        Cancel common factors in a rational function ``f/g``.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> (2*x**2 - 2).cancel(x**2 - 2*x + 1)
        (2*x + 2, x - 1)

        """
        f = self
        ring = f.ring

        if not f:
            return f, ring.one

        domain = ring.domain

        if not (domain.is_Field and domain.has_assoc_Ring):
            _, p, q = f.cofactors(g)
        else:
            new_ring = ring.clone(domain=domain.get_ring())

            cq, f = f.clear_denoms()
            cp, g = g.clear_denoms()

            f = f.set_ring(new_ring)
            g = g.set_ring(new_ring)

            _, p, q = f.cofactors(g)
            _, cp, cq = new_ring.domain.cofactors(cp, cq)

            p = p.set_ring(ring)
            q = q.set_ring(ring)

            p = p.mul_ground(cp)
            q = q.mul_ground(cq)

        # Make canonical with respect to sign or quadrant in the case of ZZ_I
        # or QQ_I. This ensures that the LC of the denominator is canonical by
        # multiplying top and bottom by a unit of the ring.
        u = q.canonical_unit()
        if u == domain.one:
            pass
        elif u == -domain.one:
            p, q = -p, -q
        else:
            p = p.mul_ground(u)
            q = q.mul_ground(u)

        return p, q

    def canonical_unit(f):
        domain = f.ring.domain
        return domain.canonical_unit(f.LC)

    def diff(f, x):
        """Computes partial derivative in ``x``.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring("x,y", ZZ)
        >>> p = x + x**2*y**3
        >>> p.diff(x)
        2*x*y**3 + 1

        """
        ring = f.ring
        i = ring.index(x)
        m = ring.monomial_basis(i)
        g = ring.zero
        for expv, coeff in f.iterterms():
            if expv[i]:
                e = ring.monomial_ldiv(expv, m)
                g[e] = ring.domain_new(coeff*expv[i])
        return g

    def __call__(f, *values):
        if 0 < len(values) <= f.ring.ngens:
            return f.evaluate(list(zip(f.ring.gens, values)))
        else:
            raise ValueError("expected at least 1 and at most %s values, got %s" % (f.ring.ngens, len(values)))

    def evaluate(self, x, a=None):
        f = self

        if isinstance(x, list) and a is None:
            (X, a), x = x[0], x[1:]
            f = f.evaluate(X, a)

            if not x:
                return f
            else:
                x = [ (Y.drop(X), a) for (Y, a) in x ]
                return f.evaluate(x)

        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)

        if ring.ngens == 1:
            result = ring.domain.zero

            for (n,), coeff in f.iterterms():
                result += coeff*a**n

            return result
        else:
            poly = ring.drop(x).zero

            for monom, coeff in f.iterterms():
                n, monom = monom[i], monom[:i] + monom[i+1:]
                coeff = coeff*a**n

                if monom in poly:
                    coeff = coeff + poly[monom]

                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                else:
                    if coeff:
                        poly[monom] = coeff

            return poly

    def subs(self, x, a=None):
        f = self

        if isinstance(x, list) and a is None:
            for X, a in x:
                f = f.subs(X, a)
            return f

        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)

        if ring.ngens == 1:
            result = ring.domain.zero

            for (n,), coeff in f.iterterms():
                result += coeff*a**n

            return ring.ground_new(result)
        else:
            poly = ring.zero

            for monom, coeff in f.iterterms():
                n, monom = monom[i], monom[:i] + (0,) + monom[i+1:]
                coeff = coeff*a**n

                if monom in poly:
                    coeff = coeff + poly[monom]

                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                else:
                    if coeff:
                        poly[monom] = coeff

            return poly

    def symmetrize(self):
        r"""
        Rewrite *self* in terms of elementary symmetric polynomials.

        Explanation
        ===========

        If this :py:class:`~.PolyElement` belongs to a ring of $n$ variables,
        we can try to write it as a function of the elementary symmetric
        polynomials on $n$ variables. We compute a symmetric part, and a
        remainder for any part we were not able to symmetrize.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> R, x, y = ring("x,y", ZZ)

        >>> f = x**2 + y**2
        >>> f.symmetrize()
        (x**2 - 2*y, 0, [(x, x + y), (y, x*y)])

        >>> f = x**2 - y**2
        >>> f.symmetrize()
        (x**2 - 2*y, -2*y**2, [(x, x + y), (y, x*y)])

        Returns
        =======

        Triple ``(p, r, m)``
            ``p`` is a :py:class:`~.PolyElement` that represents our attempt
            to express *self* as a function of elementary symmetric
            polynomials. Each variable in ``p`` stands for one of the
            elementary symmetric polynomials. The correspondence is given
            by ``m``.

            ``r`` is the remainder.

            ``m`` is a list of pairs, giving the mapping from variables in
            ``p`` to elementary symmetric polynomials.

            The triple satisfies the equation ``p.compose(m) + r == self``.
            If the remainder ``r`` is zero, *self* is symmetric. If it is
            nonzero, we were not able to represent *self* as symmetric.

        See Also
        ========

        sympy.polys.polyfuncs.symmetrize

        References
        ==========

        .. [1] Lauer, E. Algorithms for symmetrical polynomials, Proc. 1976
            ACM Symp. on Symbolic and Algebraic Computing, NY 242-247.
            https://dl.acm.org/doi/pdf/10.1145/800205.806342

        """
        f = self.copy()
        ring = f.ring
        n = ring.ngens

        if not n:
            return f, ring.zero, []

        polys = [ring.symmetric_poly(i+1) for i in range(n)]

        poly_powers = {}
        def get_poly_power(i, n):
            if (i, n) not in poly_powers:
                poly_powers[(i, n)] = polys[i]**n
            return poly_powers[(i, n)]

        indices = list(range(n - 1))
        weights = list(range(n, 0, -1))

        symmetric = ring.zero

        while f:
            _height, _monom, _coeff = -1, None, None

            for i, (monom, coeff) in enumerate(f.terms()):
                if all(monom[i] >= monom[i + 1] for i in indices):
                    height = max(n*m for n, m in zip(weights, monom))

                    if height > _height:
                        _height, _monom, _coeff = height, monom, coeff

            if _height != -1:
                monom, coeff = _monom, _coeff
            else:
                break

            exponents = []
            for m1, m2 in zip(monom, monom[1:] + (0,)):
                exponents.append(m1 - m2)

            symmetric += ring.term_new(tuple(exponents), coeff)

            product = coeff
            for i, n in enumerate(exponents):
                product *= get_poly_power(i, n)
            f -= product

        mapping = list(zip(ring.gens, polys))

        return symmetric, f, mapping

    def compose(f, x, a=None):
        ring = f.ring
        poly = ring.zero
        gens_map = dict(zip(ring.gens, range(ring.ngens)))

        if a is not None:
            replacements = [(x, a)]
        else:
            if isinstance(x, list):
                replacements = list(x)
            elif isinstance(x, dict):
                replacements = sorted(x.items(), key=lambda k: gens_map[k[0]])
            else:
                raise ValueError("expected a generator, value pair a sequence of such pairs")

        for k, (x, g) in enumerate(replacements):
            replacements[k] = (gens_map[x], ring.ring_new(g))

        for monom, coeff in f.iterterms():
            monom = list(monom)
            subpoly = ring.one

            for i, g in replacements:
                n, monom[i] = monom[i], 0
                if n:
                    subpoly *= g**n

            subpoly = subpoly.mul_term((tuple(monom), coeff))
            poly += subpoly

        return poly

    def coeff_wrt(self, x, deg):
        """
        Coefficient of ``self`` with respect to ``x**deg``.

        Treating ``self`` as a univariate polynomial in ``x`` this finds the
        coefficient of ``x**deg`` as a polynomial in the other generators.

        Parameters
        ==========

        x : generator or generator index
            The generator or generator index to compute the expression for.
        deg : int
            The degree of the monomial to compute the expression for.

        Returns
        =======

        :py:class:`~.PolyElement`
            The coefficient of ``x**deg`` as a polynomial in the same ring.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y, z = ring("x, y, z", ZZ)

        >>> p = 2*x**4 + 3*y**4 + 10*z**2 + 10*x*z**2
        >>> deg = 2
        >>> p.coeff_wrt(2, deg) # Using the generator index
        10*x + 10
        >>> p.coeff_wrt(z, deg) # Using the generator
        10*x + 10
        >>> p.coeff(z**2) # shows the difference between coeff and coeff_wrt
        10

        See Also
        ========

        coeff, coeffs

        """
        p = self
        i = p.ring.index(x)
        terms = [(m, c) for m, c in p.iterterms() if m[i] == deg]

        if not terms:
            return p.ring.zero

        monoms, coeffs = zip(*terms)
        monoms = [m[:i] + (0,) + m[i + 1:] for m in monoms]
        return p.ring.from_dict(dict(zip(monoms, coeffs)))

    def prem(self, g, x=None):
        """
        Pseudo-remainder of the polynomial ``self`` with respect to ``g``.

        The pseudo-quotient ``q`` and pseudo-remainder ``r`` with respect to
        ``z`` when dividing ``f`` by ``g`` satisfy ``m*f = g*q + r``,
        where ``deg(r,z) < deg(g,z)`` and
        ``m = LC(g,z)**(deg(f,z) - deg(g,z)+1)``.

        See :meth:`pdiv` for explanation of pseudo-division.


        Parameters
        ==========

        g : :py:class:`~.PolyElement`
            The polynomial to divide ``self`` by.
        x : generator or generator index, optional
            The main variable of the polynomials and default is first generator.

        Returns
        =======

        :py:class:`~.PolyElement`
            The pseudo-remainder polynomial.

        Raises
        ======

        ZeroDivisionError : If ``g`` is the zero polynomial.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y = ring("x, y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2
        >>> f.prem(g) # first generator is chosen by default if it is not given
        -4*y + 4
        >>> f.rem(g) # shows the difference between prem and rem
        x**2 + x*y
        >>> f.prem(g, y) # generator is given
        0
        >>> f.prem(g, 1) # generator index is given
        0

        See Also
        ========

        pdiv, pquo, pexquo, sympy.polys.domains.ring.Ring.rem

        """
        f = self
        x = f.ring.index(x)
        df = f.degree(x)
        dg = g.degree(x)

        if dg < 0:
            raise ZeroDivisionError('polynomial division')

        r, dr = f, df

        if df < dg:
            return r

        N = df - dg + 1

        lc_g = g.coeff_wrt(x, dg)

        xp = f.ring.gens[x]

        while True:

            lc_r = r.coeff_wrt(x, dr)
            j, N = dr - dg, N - 1

            R = r * lc_g
            G = g * lc_r * xp**j
            r = R - G

            dr = r.degree(x)

            if dr < dg:
                break

        c = lc_g ** N

        return r * c

    def pdiv(self, g, x=None):
        """
        Computes the pseudo-division of the polynomial ``self`` with respect to ``g``.

        The pseudo-division algorithm is used to find the pseudo-quotient ``q``
        and pseudo-remainder ``r`` such that ``m*f = g*q + r``, where ``m``
        represents the multiplier and ``f`` is the dividend polynomial.

        The pseudo-quotient ``q`` and pseudo-remainder ``r`` are polynomials in
        the variable ``x``, with the degree of ``r`` with respect to ``x``
        being strictly less than the degree of ``g`` with respect to ``x``.

        The multiplier ``m`` is defined as
        ``LC(g, x) ^ (deg(f, x) - deg(g, x) + 1)``,
        where ``LC(g, x)`` represents the leading coefficient of ``g``.

        It is important to note that in the context of the ``prem`` method,
        multivariate polynomials in a ring, such as ``R[x,y,z]``, are treated
        as univariate polynomials with coefficients that are polynomials,
        such as ``R[x,y][z]``. When dividing ``f`` by ``g`` with respect to the
        variable ``z``, the pseudo-quotient ``q`` and pseudo-remainder ``r``
        satisfy ``m*f = g*q + r``, where ``deg(r, z) < deg(g, z)``
        and ``m = LC(g, z)^(deg(f, z) - deg(g, z) + 1)``.

        In this function, the pseudo-remainder ``r`` can be obtained using the
        ``prem`` method, the pseudo-quotient ``q`` can
        be obtained using the ``pquo`` method, and
        the function ``pdiv`` itself returns a tuple ``(q, r)``.


        Parameters
        ==========

        g : :py:class:`~.PolyElement`
            The polynomial to divide ``self`` by.
        x : generator or generator index, optional
            The main variable of the polynomials and default is first generator.

        Returns
        =======

        :py:class:`~.PolyElement`
            The pseudo-division polynomial (tuple of ``q`` and ``r``).

        Raises
        ======

        ZeroDivisionError : If ``g`` is the zero polynomial.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y = ring("x, y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2
        >>> f.pdiv(g) # first generator is chosen by default if it is not given
        (2*x + 2*y - 2, -4*y + 4)
        >>> f.div(g) # shows the difference between pdiv and div
        (0, x**2 + x*y)
        >>> f.pdiv(g, y) # generator is given
        (2*x**3 + 2*x**2*y + 6*x**2 + 2*x*y + 8*x + 4, 0)
        >>> f.pdiv(g, 1) # generator index is given
        (2*x**3 + 2*x**2*y + 6*x**2 + 2*x*y + 8*x + 4, 0)

        See Also
        ========

        prem
            Computes only the pseudo-remainder more efficiently than
            `f.pdiv(g)[1]`.
        pquo
            Returns only the pseudo-quotient.
        pexquo
            Returns only an exact pseudo-quotient having no remainder.
        div
            Returns quotient and remainder of f and g polynomials.

        """
        f = self
        x = f.ring.index(x)

        df = f.degree(x)
        dg = g.degree(x)

        if dg < 0:
            raise ZeroDivisionError("polynomial division")

        q, r, dr = x, f, df

        if df < dg:
            return q, r

        N = df - dg + 1
        lc_g = g.coeff_wrt(x, dg)

        xp = f.ring.gens[x]

        while True:

            lc_r = r.coeff_wrt(x, dr)
            j, N = dr - dg, N - 1

            Q = q * lc_g

            q = Q + (lc_r)*xp**j

            R = r * lc_g

            G = g * lc_r * xp**j

            r = R - G

            dr = r.degree(x)

            if dr < dg:
                break

        c = lc_g**N

        q = q * c
        r = r * c

        return q, r

    def pquo(self, g, x=None):
        """
        Polynomial pseudo-quotient in multivariate polynomial ring.

        Examples
        ========
        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2*y
        >>> h = 2*x + 2
        >>> f.pquo(g)
        2*x
        >>> f.quo(g) # shows the difference between pquo and quo
        0
        >>> f.pquo(h)
        2*x + 2*y - 2
        >>> f.quo(h) # shows the difference between pquo and quo
        0

        See Also
        ========

        prem, pdiv, pexquo, sympy.polys.domains.ring.Ring.quo

        """
        f = self
        return f.pdiv(g, x)[0]

    def pexquo(self, g, x=None):
        """
        Polynomial exact pseudo-quotient in multivariate polynomial ring.

        Examples
        ========
        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2*y
        >>> h = 2*x + 2
        >>> f.pexquo(g)
        2*x
        >>> f.exquo(g) # shows the difference between pexquo and exquo
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x + 2*y does not divide x**2 + x*y
        >>> f.pexquo(h)
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x + 2 does not divide x**2 + x*y

        See Also
        ========

        prem, pdiv, pquo, sympy.polys.domains.ring.Ring.exquo

        """
        f = self
        q, r = f.pdiv(g, x)

        if r.is_zero:
            return q
        else:
            raise ExactQuotientFailed(f, g)

    def subresultants(self, g, x=None):
        """
        Computes the subresultant PRS of two polynomials ``self`` and ``g``.

        Parameters
        ==========

        g : :py:class:`~.PolyElement`
            The second polynomial.
        x : generator or generator index
            The variable with respect to which the subresultant sequence is computed.

        Returns
        =======

        R : list
            Returns a list polynomials representing the subresultant PRS.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y = ring("x, y", ZZ)

        >>> f = x**2*y + x*y
        >>> g = x + y
        >>> f.subresultants(g) # first generator is chosen by default if not given
        [x**2*y + x*y, x + y, y**3 - y**2]
        >>> f.subresultants(g, 0) # generator index is given
        [x**2*y + x*y, x + y, y**3 - y**2]
        >>> f.subresultants(g, y) # generator is given
        [x**2*y + x*y, x + y, x**3 + x**2]

        """
        f = self
        x = f.ring.index(x)
        n = f.degree(x)
        m = g.degree(x)

        if n < m:
            f, g = g, f
            n, m = m, n

        if f == 0:
            return [0, 0]

        if g == 0:
            return [f, 1]

        R = [f, g]

        d = n - m
        b = (-1) ** (d + 1)

        # Compute the pseudo-remainder for f and g
        h = f.prem(g, x)
        h = h * b

        # Compute the coefficient of g with respect to x**m
        lc = g.coeff_wrt(x, m)

        c = lc ** d

        S = [1, c]

        c = -c

        while h:
            k = h.degree(x)

            R.append(h)
            f, g, m, d = g, h, k, m - k

            b = -lc * c ** d
            h = f.prem(g, x)
            h = h.exquo(b)

            lc = g.coeff_wrt(x, k)

            if d > 1:
                p = (-lc) ** d
                q = c ** (d - 1)
                c = p.exquo(q)
            else:
                c = -lc

            S.append(-c)

        return R

    # TODO: following methods should point to polynomial
    # representation independent algorithm implementations.

    def half_gcdex(f, g):
        return f.ring.dmp_half_gcdex(f, g)

    def gcdex(f, g):
        return f.ring.dmp_gcdex(f, g)

    def resultant(f, g):
        return f.ring.dmp_resultant(f, g)

    def discriminant(f):
        return f.ring.dmp_discriminant(f)

    def decompose(f):
        if f.ring.is_univariate:
            return f.ring.dup_decompose(f)
        else:
            raise MultivariatePolynomialError("polynomial decomposition")

    def shift(f, a):
        if f.ring.is_univariate:
            return f.ring.dup_shift(f, a)
        else:
            raise MultivariatePolynomialError("shift: use shift_list instead")

    def shift_list(f, a):
        return f.ring.dmp_shift(f, a)

    def sturm(f):
        if f.ring.is_univariate:
            return f.ring.dup_sturm(f)
        else:
            raise MultivariatePolynomialError("sturm sequence")

    def gff_list(f):
        return f.ring.dmp_gff_list(f)

    def norm(f):
        return f.ring.dmp_norm(f)

    def sqf_norm(f):
        return f.ring.dmp_sqf_norm(f)

    def sqf_part(f):
        return f.ring.dmp_sqf_part(f)

    def sqf_list(f, all=False):
        return f.ring.dmp_sqf_list(f, all=all)

    def factor_list(f):
        return f.ring.dmp_factor_list(f)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\symbolic_shape_infer.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import argparse
import logging

import numpy as np
import onnx
import sympy
from onnx import helper, numpy_helper, shape_inference
from packaging import version

assert version.parse(onnx.__version__) >= version.parse("1.8.0")

logger = logging.getLogger(__name__)


def get_attribute(node, attr_name, default_value=None):
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return helper.get_attribute_value(found[0])
    return default_value


def get_dim_from_proto(dim):
    return getattr(dim, dim.WhichOneof("value")) if type(dim.WhichOneof("value")) is str else None


def is_sequence(type_proto):
    cls_type = type_proto.WhichOneof("value")
    assert cls_type in ["tensor_type", "sequence_type"]
    return cls_type == "sequence_type"


def get_shape_from_type_proto(type_proto):
    assert not is_sequence(type_proto)
    if type_proto.tensor_type.HasField("shape"):
        return [get_dim_from_proto(d) for d in type_proto.tensor_type.shape.dim]
    else:
        return None  # note no shape is different from shape without dim (scalar)


def get_elem_type_from_type_proto(type_proto):
    if is_sequence(type_proto):
        return type_proto.sequence_type.elem_type.tensor_type.elem_type
    else:
        return type_proto.tensor_type.elem_type


def get_shape_from_value_info(vi):
    cls_type = vi.type.WhichOneof("value")
    if cls_type is None:
        return None
    if is_sequence(vi.type):
        if vi.type.sequence_type.elem_type.WhichOneof("value") == "tensor_type":
            return get_shape_from_type_proto(vi.type.sequence_type.elem_type)
        else:
            return None
    else:
        return get_shape_from_type_proto(vi.type)


def make_named_value_info(name):
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def get_shape_from_sympy_shape(sympy_shape):
    return [None if i is None else (int(i) if is_literal(i) else str(i)) for i in sympy_shape]


def is_literal(dim):
    return type(dim) in [int, np.int64, np.int32, sympy.Integer] or (hasattr(dim, "is_number") and dim.is_number)


def handle_negative_axis(axis, rank):
    assert axis < rank and axis >= -rank
    return axis if axis >= 0 else rank + axis


def get_opset(mp, domain=None):
    domain = domain or ["", "onnx", "ai.onnx"]
    if type(domain) != list:  # noqa: E721
        domain = [domain]
    for opset in mp.opset_import:
        if opset.domain in domain:
            return opset.version

    return None


def as_scalar(x):
    if type(x) is list:
        assert len(x) == 1
        return x[0]
    elif type(x) is np.ndarray:
        return x.item()
    else:
        return x


def as_list(x, keep_none):
    if type(x) is list:
        return x
    elif type(x) is np.ndarray:
        return list(x)
    elif keep_none and x is None:
        return None
    else:
        return [x]


def sympy_reduce_product(x):
    if type(x) is list:
        value = sympy.Integer(1)
        for v in x:
            value = value * v
    else:
        value = x
    return value


class SymbolicShapeInference:
    def __init__(self, int_max, auto_merge, guess_output_rank, verbose, prefix=""):
        self.dispatcher_ = {
            "Add": self._infer_symbolic_compute_ops,
            "AllReduce": self._pass_on_shape_and_type,
            "ArrayFeatureExtractor": self._infer_ArrayFeatureExtractor,
            "AveragePool": self._infer_Pool,
            "BatchNormalization": self._infer_BatchNormalization,
            "Cast": self._infer_Cast,
            "CategoryMapper": self._infer_CategoryMapper,
            "Compress": self._infer_Compress,
            "Concat": self._infer_Concat,
            "ConcatFromSequence": self._infer_ConcatFromSequence,
            "Constant": self._infer_Constant,
            "ConstantOfShape": self._infer_ConstantOfShape,
            "Conv": self._infer_Conv,
            "CumSum": self._pass_on_shape_and_type,
            "Div": self._infer_symbolic_compute_ops,
            "Einsum": self._infer_Einsum,
            "Expand": self._infer_Expand,
            "Equal": self._infer_symbolic_compute_ops,
            "Floor": self._infer_symbolic_compute_ops,
            "Gather": self._infer_Gather,
            "GatherElements": self._infer_GatherElements,
            "GatherND": self._infer_GatherND,
            "Identity": self._pass_on_shape_and_type,
            "If": self._infer_If,
            "Loop": self._infer_Loop,
            "MatMul": self._infer_MatMul,
            "MatMulInteger16": self._infer_MatMulInteger,
            "MaxPool": self._infer_Pool,
            "Max": self._infer_symbolic_compute_ops,
            "MemcpyFromHost": self._pass_on_shape_and_type,
            "MemcpyToHost": self._pass_on_shape_and_type,
            "Min": self._infer_symbolic_compute_ops,
            "MoE": self._pass_on_shape_and_type,
            "Mul": self._infer_symbolic_compute_ops,
            "NonMaxSuppression": self._infer_NonMaxSuppression,
            "NonZero": self._infer_NonZero,
            "OneHot": self._infer_OneHot,
            "Pad": self._infer_Pad,
            "Range": self._infer_Range,
            "Reciprocal": self._pass_on_shape_and_type,
            "ReduceSum": self._infer_ReduceSum,
            "ReduceMean": self._infer_ReduceMean,
            "ReduceProd": self._infer_ReduceProd,
            "Reshape": self._infer_Reshape,
            "Resize": self._infer_Resize,
            "Round": self._pass_on_shape_and_type,
            "Scan": self._infer_Scan,
            "ScatterElements": self._infer_ScatterElements,
            "SequenceAt": self._infer_SequenceAt,
            "SequenceInsert": self._infer_SequenceInsert,
            "Shape": self._infer_Shape,
            "Size": self._infer_Size,
            "Slice": self._infer_Slice,
            "SoftmaxCrossEntropyLoss": self._infer_SoftmaxCrossEntropyLoss,
            "SoftmaxCrossEntropyLossInternal": self._infer_SoftmaxCrossEntropyLoss,
            "NegativeLogLikelihoodLossInternal": self._infer_SoftmaxCrossEntropyLoss,
            "Split": self._infer_Split,
            "SplitToSequence": self._infer_SplitToSequence,
            "Squeeze": self._infer_Squeeze,
            "Sub": self._infer_symbolic_compute_ops,
            "Tile": self._infer_Tile,
            "TopK": self._infer_TopK,
            "Transpose": self._infer_Transpose,
            "Unsqueeze": self._infer_Unsqueeze,
            "Where": self._infer_symbolic_compute_ops,
            "ZipMap": self._infer_ZipMap,
            "Neg": self._infer_symbolic_compute_ops,
            # contrib ops:
            "Attention": self._infer_Attention,
            "BiasAdd": self._infer_BiasAdd,
            "BiasGelu": self._infer_BiasGelu,
            "BiasSplitGelu": self._infer_BiasSplitGelu,
            "DecoderMaskedMultiHeadAttention": self._infer_DecoderMaskedMultiHeadAttention,
            "DequantizeLinear": self._infer_DequantizeLinear,
            "DynamicTimeWarping": self._infer_DynamicTimeWarping,
            "EmbedLayerNormalization": self._infer_EmbedLayerNormalization,
            "FastGelu": self._infer_FastGelu,
            "GatedRelativePositionBias": self._infer_GatedRelativePositionBias,
            "GatherBlockQuantized": self._infer_Gather,
            "Gelu": self._infer_Gelu,
            "GemmFastGelu": self._infer_GemmFastGelu,
            "GemmFloat8": self._infer_GemmFloat8,
            "GroupNorm": self._infer_GroupNorm,
            "GroupNormalization": self._infer_GroupNorm,
            "GroupQueryAttention": self._infer_GroupQueryAttention,
            "LayerNormalization": self._infer_LayerNormalization,
            "LongformerAttention": self._infer_LongformerAttention,
            "MatMulNBits": self._infer_MatMulNBits,
            "MultiHeadAttention": self._infer_MultiHeadAttention,
            "NhwcConv": self._infer_NhwcConv,
            "PackedAttention": self._infer_PackedAttention,
            "PackedMultiHeadAttention": self._infer_PackedMultiHeadAttention,
            "PagedAttention": self._infer_PagedAttention,
            "PythonOp": self._infer_PythonOp,
            "QLinearAdd": self._infer_QLinearBinary,
            "QLinearMul": self._infer_QLinearBinary,
            "QuantizeLinear": self._infer_QuantizeLinear,
            "QuickGelu": self._infer_FastGelu,
            "RelativePositionBias": self._infer_RelativePositionBias,
            "RemovePadding": self._infer_RemovePadding,
            "RestorePadding": self._infer_RestorePadding,
            "RotaryEmbedding": self._infer_RotaryEmbedding,
            "SimplifiedLayerNormalization": self._infer_LayerNormalization,
            "SkipGroupNorm": self._infer_SkipGroupNorm,
            "SkipLayerNormalization": self._infer_SkipLayerNormalization,
            "SkipSimplifiedLayerNormalization": self._infer_SkipLayerNormalization,
            "SparseAttention": self._infer_SparseAttention,
            "UnfoldTensor": self._infer_UnfoldTensor,
        }
        self.aten_op_dispatcher_ = {
            "embedding": self._infer_Gather,
            "bitwise_or": self._infer_aten_bitwise_or,
            "diagonal": self._infer_aten_diagonal,
            "max_pool2d_with_indices": self._infer_aten_pool2d,
            "max": self._infer_aten_minmax,
            "min": self._infer_aten_minmax,
            "multinomial": self._infer_aten_multinomial,
            "unfold": self._infer_aten_unfold,
            "argmax": self._infer_aten_argmax,
            "avg_pool2d": self._infer_aten_pool2d,
            "_adaptive_avg_pool2d": self._infer_aten_pool2d,
            "numpy_T": self._infer_Transpose,
            "native_group_norm": self._infer_aten_group_norm,
            "upsample_nearest1d": self._infer_aten_upsample,
            "upsample_nearest2d": self._infer_aten_upsample,
            "upsample_nearest3d": self._infer_aten_upsample,
            "upsample_bicubic2d": self._infer_aten_upsample,
        }
        self.run_ = True
        self.suggested_merge_ = {}
        self.symbolic_dims_ = {}
        self.input_symbols_ = {}
        self.auto_merge_ = auto_merge
        self.guess_output_rank_ = guess_output_rank
        self.verbose_ = verbose
        self.int_max_ = int_max
        self.subgraph_id_ = 0
        self.prefix_ = prefix

    def _add_suggested_merge(self, symbols, apply=False):
        assert all((type(s) is str and s in self.symbolic_dims_) or is_literal(s) for s in symbols)
        symbols = set(symbols)
        for k, v in self.suggested_merge_.items():
            if k in symbols:
                symbols.remove(k)
                symbols.add(v)
        map_to = None
        # if there is literal, map to it first
        for s in symbols:
            if is_literal(s):
                map_to = s
                break
        # when no literals, map to input symbolic dims, then existing symbolic dims
        if map_to is None:
            for s in symbols:
                if s in self.input_symbols_:
                    map_to = s
                    break
        if map_to is None:
            for s in symbols:
                if type(self.symbolic_dims_[s]) is sympy.Symbol:
                    map_to = s
                    break
        # when nothing to map to, use the shorter one
        if map_to is None:
            if self.verbose_ > 0:
                logger.warning("Potential unsafe merge between symbolic expressions: (%s)", ",".join(symbols))
            symbols_list = list(symbols)
            lens = [len(s) for s in symbols_list]
            map_to = symbols_list[lens.index(min(lens))]
            symbols.remove(map_to)

        for s in symbols:
            if s == map_to:
                continue
            if is_literal(map_to) and is_literal(s):
                assert int(map_to) == int(s)
            self.suggested_merge_[s] = int(map_to) if is_literal(map_to) else map_to
            for k, v in self.suggested_merge_.items():
                if v == s:
                    self.suggested_merge_[k] = map_to
        if apply and self.auto_merge_:
            self._apply_suggested_merge()

    def _apply_suggested_merge(self, graph_input_only=False):
        if not self.suggested_merge_:
            return
        for i in list(self.out_mp_.graph.input) + ([] if graph_input_only else list(self.out_mp_.graph.value_info)):
            for d in i.type.tensor_type.shape.dim:
                if d.dim_param in self.suggested_merge_:
                    v = self.suggested_merge_[d.dim_param]
                    if is_literal(v):
                        d.dim_value = int(v)
                    else:
                        d.dim_param = v

    def _preprocess(self, in_mp):
        self.out_mp_ = onnx.ModelProto()
        self.out_mp_.CopyFrom(in_mp)
        self.graph_inputs_ = {i.name: i for i in list(self.out_mp_.graph.input)}
        self.initializers_ = {i.name: i for i in self.out_mp_.graph.initializer}
        self.known_vi_ = {i.name: i for i in list(self.out_mp_.graph.input)}
        self.known_vi_.update(
            {
                i.name: helper.make_tensor_value_info(i.name, i.data_type, list(i.dims))
                for i in self.out_mp_.graph.initializer
            }
        )

    def _merge_symbols(self, dims):
        if not all(type(d) is str for d in dims):
            if self.auto_merge_:
                unique_dims = list(set(dims))
                is_int = [is_literal(d) for d in unique_dims]
                assert sum(is_int) <= 1  # if there are more than 1 unique ints, something is wrong
                if sum(is_int) == 1:
                    int_dim = is_int.index(1)
                    if self.verbose_ > 0:
                        logger.debug(
                            f"dim {unique_dims[:int_dim] + unique_dims[int_dim + 1 :]} has been merged with value {unique_dims[int_dim]}"
                        )
                    self._check_merged_dims(unique_dims, allow_broadcast=False)
                    return unique_dims[int_dim]
                else:
                    if self.verbose_ > 0:
                        logger.debug(f"dim {unique_dims[1:]} has been merged with dim {unique_dims[0]}")
                    return dims[0]
            else:
                return None
        if all(d == dims[0] for d in dims):
            return dims[0]
        merged = [self.suggested_merge_.get(d, d) for d in dims]
        if all(d == merged[0] for d in merged):
            assert merged[0] in self.symbolic_dims_
            return merged[0]
        else:
            return None

    # broadcast from right to left, and merge symbolic dims if needed
    def _broadcast_shapes(self, shape1, shape2):
        new_shape = []
        rank1 = len(shape1)
        rank2 = len(shape2)
        new_rank = max(rank1, rank2)
        for i in range(new_rank):
            dim1 = shape1[rank1 - 1 - i] if i < rank1 else 1
            dim2 = shape2[rank2 - 1 - i] if i < rank2 else 1
            if dim1 == 1 or dim1 == dim2:
                new_dim = dim2
            elif dim2 == 1:
                new_dim = dim1
            else:
                new_dim = self._merge_symbols([dim1, dim2])
                if not new_dim:
                    # warning about unsupported broadcast when not auto merge
                    # note that auto merge has the risk of incorrectly merge symbols while one of them being 1
                    # for example, 'a' = 1, 'b' = 5 at runtime is valid broadcasting, but with auto merge 'a' == 'b'
                    if self.auto_merge_:
                        self._add_suggested_merge([dim1, dim2], apply=True)
                    else:
                        logger.warning("unsupported broadcast between " + str(dim1) + " " + str(dim2))  # noqa: G003
            new_shape = [new_dim, *new_shape]
        return new_shape

    def _get_shape(self, node, idx):
        name = node.input[idx]
        if name in self.known_vi_:
            vi = self.known_vi_[name]
            return get_shape_from_value_info(vi)
        else:
            assert name in self.initializers_
            return list(self.initializers_[name].dims)

    def _try_get_shape(self, node, idx):
        if idx > len(node.input) - 1:
            return None
        name = node.input[idx]
        if name in self.known_vi_:
            vi = self.known_vi_[name]
            return get_shape_from_value_info(vi)
        if name in self.initializers_:
            return list(self.initializers_[name].dims)
        return None

    def _get_shape_rank(self, node, idx):
        return len(self._get_shape(node, idx))

    def _get_sympy_shape(self, node, idx):
        sympy_shape = []
        for d in self._get_shape(node, idx):
            if type(d) is str:
                sympy_shape.append(
                    self.symbolic_dims_[d]
                    if d in self.symbolic_dims_
                    else sympy.Symbol(d, integer=True, nonnegative=True)
                )
            else:
                assert None is not d
                sympy_shape.append(d)
        return sympy_shape

    def _get_value(self, node, idx):
        name = node.input[idx]
        assert name in self.sympy_data_ or name in self.initializers_
        return self.sympy_data_[name] if name in self.sympy_data_ else numpy_helper.to_array(self.initializers_[name])

    def _try_get_value(self, node, idx):
        if idx >= len(node.input):
            return None
        name = node.input[idx]
        if name in self.sympy_data_ or name in self.initializers_:
            return self._get_value(node, idx)
        return None

    def _update_computed_dims(self, new_sympy_shape):
        for i, new_dim in enumerate(new_sympy_shape):
            if not is_literal(new_dim) and type(new_dim) != str:  # noqa: E721
                str_dim = str(new_dim)
                if str_dim in self.suggested_merge_:
                    if is_literal(self.suggested_merge_[str_dim]):
                        continue  # no need to create dim for literals
                    new_sympy_shape[i] = self.symbolic_dims_[self.suggested_merge_[str_dim]]
                else:
                    # add new_dim if it's a computational expression
                    if str(new_dim) not in self.symbolic_dims_:
                        self.symbolic_dims_[str(new_dim)] = new_dim

    def _onnx_infer_single_node(self, node):
        # skip onnx shape inference for some ops, as they are handled in _infer_*
        skip_infer = node.op_type in [
            "If",
            "Loop",
            "Scan",
            "SplitToSequence",
            "ZipMap",  # contrib ops
            "Attention",
            "BiasAdd",
            "BiasGelu",
            "BiasSplitGelu",
            "DequantizeLinear",
            "DynamicTimeWarping",
            "EmbedLayerNormalization",
            "FastGelu",
            "GatherBlockQuantized",
            "Gelu",
            "GemmFastGelu",
            "GroupNorm",
            "GroupNormalization",
            "GroupQueryAttention",
            "LayerNormalization",
            "LongformerAttention",
            "MultiHeadAttention",
            "NhwcConv",
            "PackedAttention",
            "PagedAttention",
            "PythonOp",
            "QuantizeLinear",
            "QuickGelu",
            "RelativePositionBias",
            "RemovePadding",
            "RestorePadding",
            "RotaryEmbedding",
            "SimplifiedLayerNormalization",
            "SkipLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "SparseAttention",
            "SkipGroupNorm",
            "QLinearAdd",
            "QLinearMul",
        ]

        if not skip_infer:
            # Only pass initializers that satisfy the following condition:
            # (1) Operator need value of some input for shape inference.
            #     For example, Unsqueeze in opset 13 uses the axes input to calculate shape of output.
            # (2) opset version >= 9. In older version, initializer is required in graph input by onnx spec.
            # (3) The initializer is not in graph input. The means the node input is "constant" in inference.
            initializers = []
            if (get_opset(self.out_mp_) >= 9) and node.op_type in ["Unsqueeze"]:
                initializers = [
                    self.initializers_[name]
                    for name in node.input
                    if (name in self.initializers_ and name not in self.graph_inputs_)
                ]

            if node.op_type in [
                "Add",
                "Sub",
                "Mul",
                "Div",
                "MatMul",
                "MatMulInteger",
                "MatMulInteger16",
                "Where",
                "Sum",
            ]:
                if node.output[0] in self.known_vi_:
                    vi = self.known_vi_[node.output[0]]
                    out_rank = len(get_shape_from_type_proto(vi.type))
                    in_shapes = [self._get_shape(node, i) for i in range(len(node.input))]
                    for d in range(
                        out_rank - (2 if node.op_type in ["MatMul", "MatMulInteger", "MatMulInteger16"] else 0)
                    ):
                        in_dims = [s[len(s) - out_rank + d] for s in in_shapes if len(s) + d >= out_rank]
                        if len(in_dims) > 1:
                            self._check_merged_dims(in_dims, allow_broadcast=True)

            # run single node inference with self.known_vi_ shapes
            tmp_graph = helper.make_graph(
                [node],
                "tmp",
                [self.known_vi_[i] for i in node.input if i],
                [make_named_value_info(i) for i in node.output],
                initializers,
            )

            self.tmp_mp_.graph.CopyFrom(tmp_graph)

            self.tmp_mp_ = shape_inference.infer_shapes(self.tmp_mp_)

        for i_o in range(len(node.output)):
            o = node.output[i_o]
            if o:  # skip optional output
                vi = self.out_mp_.graph.value_info.add()
                if not skip_infer:
                    vi.CopyFrom(self.tmp_mp_.graph.output[i_o])
                else:
                    vi.name = o
                self.known_vi_[o] = vi

    def _onnx_infer_subgraph(self, node, subgraph, use_node_input=True, inc_subgraph_id=True):
        if self.verbose_ > 2:
            logger.debug(f"Inferencing subgraph of node {node.name} with output({node.output[0]}...): {node.op_type}")
        # node inputs are not passed directly to the subgraph
        # it's up to the node dispatcher to prepare subgraph input
        # for example, with Scan/Loop, subgraph input shape would be trimmed from node input shape
        # besides, inputs in subgraph could shadow implicit inputs
        subgraph_inputs = {i.name for i in list(subgraph.initializer) + list(subgraph.input)}
        subgraph_implicit_input = {name for name in self.known_vi_ if name not in subgraph_inputs}
        tmp_graph = helper.make_graph(
            list(subgraph.node),
            "tmp",
            list(subgraph.input) + [self.known_vi_[i] for i in subgraph_implicit_input],
            [make_named_value_info(i.name) for i in subgraph.output],
        )
        tmp_graph.initializer.extend([i for i in self.out_mp_.graph.initializer if i.name in subgraph_implicit_input])
        tmp_graph.initializer.extend(subgraph.initializer)
        self.tmp_mp_.graph.CopyFrom(tmp_graph)

        symbolic_shape_inference = SymbolicShapeInference(
            self.int_max_,
            self.auto_merge_,
            self.guess_output_rank_,
            self.verbose_,
            prefix=self.prefix_ + "_" + str(self.subgraph_id_),
        )
        if inc_subgraph_id:
            self.subgraph_id_ += 1

        symbolic_shape_inference._preprocess(self.tmp_mp_)
        symbolic_shape_inference.suggested_merge_ = self.suggested_merge_.copy()
        while symbolic_shape_inference.run_:
            symbolic_shape_inference._infer_impl(self.sympy_data_.copy())
        symbolic_shape_inference._update_output_from_vi()
        if use_node_input:
            # if subgraph uses node input, it needs to update to merged dims
            subgraph.ClearField("input")
            subgraph.input.extend(symbolic_shape_inference.out_mp_.graph.input[: len(node.input)])
        subgraph.ClearField("output")
        subgraph.output.extend(symbolic_shape_inference.out_mp_.graph.output)
        subgraph.ClearField("value_info")
        subgraph.value_info.extend(symbolic_shape_inference.out_mp_.graph.value_info)
        subgraph.ClearField("node")
        subgraph.node.extend(symbolic_shape_inference.out_mp_.graph.node)
        # for new symbolic dims from subgraph output, add to main graph symbolic dims
        subgraph_shapes = [get_shape_from_value_info(o) for o in symbolic_shape_inference.out_mp_.graph.output]
        subgraph_new_symbolic_dims = {
            d for s in subgraph_shapes if s for d in s if type(d) is str and d not in self.symbolic_dims_
        }
        new_dims = {}
        for d in subgraph_new_symbolic_dims:
            assert d in symbolic_shape_inference.symbolic_dims_
            new_dims[d] = symbolic_shape_inference.symbolic_dims_[d]
        self.symbolic_dims_.update(new_dims)
        return symbolic_shape_inference

    def _get_int_or_float_values(self, node, broadcast=False, allow_float_values=False):
        def int_or_float(value, allow_float_values):
            # If casting into int has precision loss: keep float output
            if allow_float_values and value % 1 != 0:
                return value
            return int(value)

        values = [self._try_get_value(node, i) for i in range(len(node.input))]
        if all(v is not None for v in values):
            # some shape compute is in floating point, cast to int for sympy
            for i, v in enumerate(values):
                if type(v) is not np.ndarray:
                    continue
                if len(v.shape) > 1:
                    new_v = None  # ignore value for rank > 1
                elif len(v.shape) == 0:
                    new_v = int_or_float(v.item(), allow_float_values)
                else:
                    assert len(v.shape) == 1
                    new_v = [int_or_float(vv, allow_float_values) for vv in v]
                values[i] = new_v
        values_len = [len(v) if isinstance(v, list) else 0 for v in values]
        max_len = max(values_len)
        if max_len >= 1 and broadcast:
            # broadcast
            for i, v in enumerate(values):
                if v is None:
                    continue  # don't broadcast if value is unknown
                if isinstance(v, list):
                    if len(v) < max_len:
                        values[i] = v * max_len
                    else:
                        assert len(v) == max_len
                else:
                    values[i] = [v] * max_len
        return values

    def _compute_on_sympy_data(self, node, op_func):
        assert len(node.output) == 1

        # Before mul & div operations
        # cast inputs into interger might lose decimal part and reduce precision
        # keep them as float, finish the operation, then cast the result into integer
        if node.op_type in ["Mul", "Div"]:
            values = self._get_int_or_float_values(node, broadcast=True, allow_float_values=True)
        else:
            values = self._get_int_or_float_values(node, broadcast=True)

        if all(v is not None for v in values):
            is_list = [isinstance(v, list) for v in values]
            as_list = any(is_list)
            if as_list:
                self.sympy_data_[node.output[0]] = [op_func(vs) for vs in zip(*values, strict=False)]
            else:
                self.sympy_data_[node.output[0]] = op_func(values)

    def _pass_on_sympy_data(self, node):
        assert len(node.input) == 1 or node.op_type in [
            "Reshape",
            "Unsqueeze",
            "Squeeze",
        ]
        self._compute_on_sympy_data(node, lambda x: x[0])

    def _pass_on_shape_and_type(self, node):
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                get_elem_type_from_type_proto(self.known_vi_[node.input[0]].type),
                self._get_shape(node, 0),
            )
        )

    def _new_symbolic_dim(self, prefix, dim):
        new_dim = f"{prefix}_d{dim}"
        if new_dim in self.suggested_merge_:
            v = self.suggested_merge_[new_dim]
            new_symbolic_dim = sympy.Integer(int(v)) if is_literal(v) else v
        else:
            new_symbolic_dim = sympy.Symbol(new_dim, integer=True, nonnegative=True)
            self.symbolic_dims_[new_dim] = new_symbolic_dim
        return new_symbolic_dim

    def _new_symbolic_dim_from_output(self, node, out_idx=0, dim=0):
        return self._new_symbolic_dim(
            f"{node.op_type}{self.prefix_}_{list(self.out_mp_.graph.node).index(node)}_o{out_idx}_",
            dim,
        )

    def _new_symbolic_shape(self, rank, node, out_idx=0):
        return [self._new_symbolic_dim_from_output(node, out_idx, i) for i in range(rank)]

    def _compute_conv_pool_shape(self, node, channels_last=False):
        sympy_shape = self._get_sympy_shape(node, 0)
        if len(node.input) > 1:
            W_shape = self._get_sympy_shape(node, 1)  # noqa: N806
            rank = len(W_shape) - 2  # number of spatial axes
            kernel_shape = W_shape[-rank - 1 : -1] if channels_last else W_shape[-rank:]
            sympy_shape[3 if channels_last else 1] = W_shape[0]
        else:
            W_shape = None  # noqa: N806
            kernel_shape = get_attribute(node, "kernel_shape")
            rank = len(kernel_shape)

        assert len(sympy_shape) == rank + 2

        # only need to symbolic shape inference if input has symbolic dims in spatial axes
        spatial_shape = sympy_shape[-rank - 1 : -1] if channels_last else sympy_shape[-rank:]
        is_symbolic_dims = [not is_literal(i) for i in spatial_shape]

        if not any(is_symbolic_dims):
            shape = get_shape_from_value_info(self.known_vi_[node.output[0]])
            if len(shape) > 0:
                assert len(sympy_shape) == len(shape)
                if channels_last:
                    sympy_shape[-rank - 1 : -1] = [sympy.Integer(d) for d in shape[-rank - 1 : -1]]
                else:
                    sympy_shape[-rank:] = [sympy.Integer(d) for d in shape[-rank:]]
                return sympy_shape

        dilations = get_attribute(node, "dilations", [1] * rank)
        strides = get_attribute(node, "strides", [1] * rank)
        effective_kernel_shape = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations, strict=False)]
        pads = get_attribute(node, "pads")
        if pads is None:
            pads = [0] * (2 * rank)
            auto_pad = get_attribute(node, "auto_pad", b"NOTSET").decode("utf-8")
            if auto_pad != "VALID" and auto_pad != "NOTSET":
                try:
                    residual = [sympy.Mod(d, s) for d, s in zip(sympy_shape[-rank:], strides, strict=False)]
                    total_pads = [
                        max(0, (k - s) if r == 0 else (k - r))
                        for k, s, r in zip(effective_kernel_shape, strides, residual, strict=False)
                    ]
                except TypeError:  # sympy may throw TypeError: cannot determine truth value of Relational
                    total_pads = [
                        max(0, (k - s)) for k, s in zip(effective_kernel_shape, strides, strict=False)
                    ]  # assuming no residual if sympy throws error
            elif auto_pad == "VALID":
                total_pads = []
            else:
                total_pads = [0] * rank
        else:
            assert len(pads) == 2 * rank
            total_pads = [p1 + p2 for p1, p2 in zip(pads[:rank], pads[rank:], strict=False)]

        ceil_mode = get_attribute(node, "ceil_mode", 0)
        for i in range(rank):
            effective_input_size = sympy_shape[-rank + i + (-1 if channels_last else 0)]
            if len(total_pads) > 0:
                effective_input_size = effective_input_size + total_pads[i]
            if ceil_mode:
                strided_kernel_positions = sympy.ceiling(
                    (effective_input_size - effective_kernel_shape[i]) / strides[i]
                )
            else:
                strided_kernel_positions = (effective_input_size - effective_kernel_shape[i]) // strides[i]
            sympy_shape[-rank + i + (-1 if channels_last else 0)] = strided_kernel_positions + 1
        return sympy_shape

    def _check_merged_dims(self, dims, allow_broadcast=True):
        if allow_broadcast:
            dims = [d for d in dims if not (is_literal(d) and int(d) <= 1)]
        if not all(d == dims[0] for d in dims):
            self._add_suggested_merge(dims, apply=True)

    def _compute_matmul_shape(self, node, output_dtype=None):
        lhs_shape = self._get_shape(node, 0)
        rhs_shape = self._get_shape(node, 1)
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)
        lhs_reduce_dim = 0
        rhs_reduce_dim = 0
        assert lhs_rank > 0 and rhs_rank > 0
        if lhs_rank == 1 and rhs_rank == 1:
            new_shape = []
        elif lhs_rank == 1:
            rhs_reduce_dim = -2
            new_shape = rhs_shape[:rhs_reduce_dim] + [rhs_shape[-1]]
        elif rhs_rank == 1:
            lhs_reduce_dim = -1
            new_shape = lhs_shape[:lhs_reduce_dim]
        else:
            lhs_reduce_dim = -1
            rhs_reduce_dim = -2
            new_shape = [*self._broadcast_shapes(lhs_shape[:-2], rhs_shape[:-2]), lhs_shape[-2], rhs_shape[-1]]
        # merge reduce dim
        self._check_merged_dims(
            [lhs_shape[lhs_reduce_dim], rhs_shape[rhs_reduce_dim]],
            allow_broadcast=False,
        )
        if output_dtype is None:
            # infer output_dtype from input type when not specified
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def _fuse_tensor_type(self, node, out_idx, dst_type, src_type):
        """
        update dst_tensor_type to be compatible with src_tensor_type when dimension mismatches
        """
        dst_tensor_type = (
            dst_type.sequence_type.elem_type.tensor_type if is_sequence(dst_type) else dst_type.tensor_type
        )
        src_tensor_type = (
            src_type.sequence_type.elem_type.tensor_type if is_sequence(src_type) else src_type.tensor_type
        )
        if dst_tensor_type.elem_type != src_tensor_type.elem_type:
            node_id = node.name if node.name else node.op_type
            raise ValueError(
                f"For node {node_id}, dst_tensor_type.elem_type != src_tensor_type.elem_type: "
                f"{onnx.onnx_pb.TensorProto.DataType.Name(dst_tensor_type.elem_type)} vs "
                f"{onnx.onnx_pb.TensorProto.DataType.Name(src_tensor_type.elem_type)}"
            )
        if dst_tensor_type.HasField("shape"):
            for di, ds in enumerate(zip(dst_tensor_type.shape.dim, src_tensor_type.shape.dim, strict=False)):
                if ds[0] != ds[1]:
                    # create a new symbolic dimension for node/out_idx/mismatch dim id in dst_tensor_type for tensor_type
                    # for sequence_type, clear the dimension
                    new_dim = onnx.TensorShapeProto.Dimension()
                    if not is_sequence(dst_type):
                        new_dim.dim_param = str(self._new_symbolic_dim_from_output(node, out_idx, di))
                    dst_tensor_type.shape.dim[di].CopyFrom(new_dim)
        else:
            dst_tensor_type.CopyFrom(src_tensor_type)

    def _infer_ArrayFeatureExtractor(self, node):  # noqa: N802
        data_shape = self._get_shape(node, 0)
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                data_shape[:-1] + indices_shape,
            )
        )

    def _infer_symbolic_compute_ops(self, node):
        funcs = {
            "Add": lambda l: l[0] + l[1],  # noqa: E741
            "Div": lambda l: (  # noqa: E741
                int(l[0] // l[1]) if isinstance(l[0] // l[1], float) else l[0] // l[1]
            ),  # integer div in sympy
            "Equal": lambda l: l[0] == l[1],  # noqa: E741
            "Floor": lambda l: sympy.floor(l[0]),  # noqa: E741
            "Max": lambda l: (  # noqa: E741
                l[1]
                if is_literal(l[0]) and int(l[0]) < -self.int_max_
                else (l[0] if is_literal(l[1]) and int(l[1]) < -self.int_max_ else sympy.Max(l[0], l[1]))
            ),
            "Min": lambda l: (  # noqa: E741
                l[1]
                if is_literal(l[0]) and int(l[0]) > self.int_max_
                else (l[0] if is_literal(l[1]) and int(l[1]) > self.int_max_ else sympy.Min(l[0], l[1]))
            ),
            "Mul": lambda l: int(l[0] * l[1]) if isinstance(l[0] * l[1], float) else l[0] * l[1],  # noqa: E741
            "Sub": lambda l: l[0] - l[1],  # noqa: E741
            "Where": lambda l: l[1] if l[0] else l[2],  # noqa: E741
            "Neg": lambda l: -l[0],  # noqa: E741
        }
        assert node.op_type in funcs
        self._compute_on_sympy_data(node, funcs[node.op_type])

    def _infer_Cast(self, node):  # noqa: N802
        self._pass_on_sympy_data(node)

    def _infer_CategoryMapper(self, node):  # noqa: N802
        input_type = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        if input_type == onnx.TensorProto.STRING:
            output_type = onnx.TensorProto.INT64
        else:
            output_type = onnx.TensorProto.STRING
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_type, self._get_shape(node, 0)))

    def _infer_Compress(self, node):  # noqa: N802
        input_shape = self._get_shape(node, 0)
        # create a new symbolic dimension for Compress output
        compress_len = str(self._new_symbolic_dim_from_output(node))
        axis = get_attribute(node, "axis")
        if axis is None:
            # when axis is not specified, input is flattened before compress so output is 1D
            output_shape = [compress_len]
        else:
            output_shape = input_shape
            output_shape[handle_negative_axis(axis, len(input_shape))] = compress_len
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )

    def _infer_Concat(self, node):  # noqa: N802
        if any(i in self.sympy_data_ or i in self.initializers_ for i in node.input):
            values = self._get_int_or_float_values(node)
            if all(v is not None for v in values):
                assert get_attribute(node, "axis") == 0
                self.sympy_data_[node.output[0]] = []
                for i in range(len(node.input)):
                    value = values[i]
                    if isinstance(value, list):
                        self.sympy_data_[node.output[0]].extend(value)
                    else:
                        self.sympy_data_[node.output[0]].append(value)

        sympy_shape = self._get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis"), len(sympy_shape))
        for i_idx in range(1, len(node.input)):
            input_shape = self._get_sympy_shape(node, i_idx)
            if input_shape:
                sympy_shape[axis] = sympy_shape[axis] + input_shape[axis]
        self._update_computed_dims(sympy_shape)
        # merge symbolic dims for non-concat axes
        for d in range(len(sympy_shape)):
            if d == axis:
                continue
            dims = [self._get_shape(node, i_idx)[d] for i_idx in range(len(node.input)) if self._get_shape(node, i_idx)]
            if all(d == dims[0] for d in dims):
                continue
            merged = self._merge_symbols(dims)
            if type(merged) is str:
                sympy_shape[d] = self.symbolic_dims_[merged] if merged else None
            else:
                sympy_shape[d] = merged
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _infer_ConcatFromSequence(self, node):  # noqa: N802
        seq_shape = self._get_shape(node, 0)
        new_axis = 1 if get_attribute(node, "new_axis") else 0
        axis = handle_negative_axis(get_attribute(node, "axis"), len(seq_shape) + new_axis)
        concat_dim = str(self._new_symbolic_dim_from_output(node, 0, axis))
        new_shape = seq_shape
        if new_axis:
            new_shape = seq_shape[:axis] + [concat_dim] + seq_shape[axis:]
        else:
            new_shape[axis] = concat_dim
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.sequence_type.elem_type.tensor_type.elem_type,
                new_shape,
            )
        )

    def _infer_Constant(self, node):  # noqa: N802
        t = get_attribute(node, "value")
        self.sympy_data_[node.output[0]] = numpy_helper.to_array(t)

    def _infer_ConstantOfShape(self, node):  # noqa: N802
        sympy_shape = self._get_int_or_float_values(node)[0]
        vi = self.known_vi_[node.output[0]]
        if sympy_shape is not None:
            if type(sympy_shape) != list:  # noqa: E721
                sympy_shape = [sympy_shape]
            self._update_computed_dims(sympy_shape)
            # update sympy data if output type is int, and shape is known
            if vi.type.tensor_type.elem_type == onnx.TensorProto.INT64 and all(is_literal(x) for x in sympy_shape):
                self.sympy_data_[node.output[0]] = np.ones(
                    [int(x) for x in sympy_shape], dtype=np.int64
                ) * numpy_helper.to_array(get_attribute(node, "value", 0))
        else:
            # create new dynamic shape
            # note input0 is a 1D vector of shape, the new symbolic shape has the rank of the shape vector length
            sympy_shape = self._new_symbolic_shape(self._get_shape(node, 0)[0], node)

        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _infer_Conv(self, node):  # noqa: N802
        sympy_shape = self._compute_conv_pool_shape(node)
        self._update_computed_dims(sympy_shape)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _infer_NhwcConv(self, node):  # noqa: N802
        sympy_shape = self._compute_conv_pool_shape(node, channels_last=True)
        self._update_computed_dims(sympy_shape)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )

    def _infer_DequantizeLinear(self, node):  # noqa: N802
        # Get the output data type from the scale input (index 1, required).
        output_dtype = self.known_vi_[node.input[1]].type.tensor_type.elem_type

        # Get the output shape from the first input.
        output_shape = self._get_shape(node, 0)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_QuantizeLinear(self, node):  # noqa: N802
        # Get the output data type from the zero-point input (index 2, optional).
        # Otherwise, default to uint8
        output_dtype = onnx.TensorProto.UINT8
        if len(node.input) > 2 and node.input[2]:
            output_dtype = self.known_vi_[node.input[2]].type.tensor_type.elem_type

        # Get the output shape from the first input.
        output_shape = self._get_shape(node, 0)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_QLinearBinary(self, node):  # noqa: N802
        # Get the output data type from the first input to QLinearAdd / QLinearMul.
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type

        # The inputs are first and fourth operands respectively.
        input_1_shape = self._get_shape(node, 0)
        input_2_shape = self._get_shape(node, 3)

        # Compute the broadcasted shape
        new_shape = self._broadcast_shapes(input_1_shape, input_2_shape)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def _infer_Einsum(self, node):  # noqa: N802
        # ref:https://github.com/onnx/onnx/blob/623dfaa0151b2e4ce49779c3ec31cbd78c592b80/onnx/defs/math/defs.cc#L3275
        equation = get_attribute(node, "equation")
        equation = equation.replace(b" ", b"")
        mid_index = equation.find(b"->")
        left_equation = equation[:mid_index] if mid_index != -1 else equation

        num_operands = 0
        num_ellipsis = 0
        num_ellipsis_indices = 0

        letter_to_dim = {}

        terms = left_equation.split(b",")
        for term in terms:
            ellipsis_index = term.find(b"...")
            shape = self._get_shape(node, num_operands)
            rank = len(shape)
            if ellipsis_index != -1:
                if num_ellipsis == 0:
                    num_ellipsis_indices = rank - len(term) + 3
                num_ellipsis = num_ellipsis + 1
            for i in range(1, rank + 1):
                letter = term[-i]
                if letter != 46:  # letter != b'.'
                    dim = shape[-i]
                    if letter not in letter_to_dim:
                        letter_to_dim[letter] = dim
                    elif type(dim) is not sympy.Symbol:
                        letter_to_dim[letter] = dim
            num_operands = num_operands + 1

        new_sympy_shape = []
        from collections import OrderedDict

        num_letter_occurrences = OrderedDict()
        if mid_index != -1:
            right_equation = equation[mid_index + 2 :]
            right_ellipsis_index = right_equation.find(b"...")
            if right_ellipsis_index != -1:
                for i in range(num_ellipsis_indices):
                    new_sympy_shape.append(shape[i])
            for c in right_equation:
                if c != 46:  # c != b'.'
                    new_sympy_shape.append(letter_to_dim[c])
        else:
            for i in range(num_ellipsis_indices):
                new_sympy_shape.append(shape[i])
            for c in left_equation:
                if c != 44 and c != 46:  # c != b',' and c != b'.':
                    if c in num_letter_occurrences:
                        num_letter_occurrences[c] = num_letter_occurrences[c] + 1
                    else:
                        num_letter_occurrences[c] = 1
            for key, value in num_letter_occurrences.items():
                if value == 1:
                    new_sympy_shape.append(letter_to_dim[key])

        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_sympy_shape))

    def _infer_Expand(self, node):  # noqa: N802
        expand_to_shape = as_list(self._try_get_value(node, 1), keep_none=True)
        if expand_to_shape is not None:
            # new_shape's dim can come from shape value
            self._update_computed_dims(expand_to_shape)
            shape = self._get_shape(node, 0)
            new_shape = self._broadcast_shapes(shape, get_shape_from_sympy_shape(expand_to_shape))
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    new_shape,
                )
            )

    def _infer_Gather(self, node):  # noqa: N802
        data_shape = self._get_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis", 0), len(data_shape))
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        if node.op_type == "Gather":
            elem_type = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        elif node.op_type == "GatherBlockQuantized":
            # scales
            elem_type = self.known_vi_[node.input[2]].type.tensor_type.elem_type
        else:
            raise ValueError(f"Unsupported Gather op_type: {node.op_type}")
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                elem_type,
                data_shape[:axis] + indices_shape + data_shape[axis + 1 :],
            )
        )
        # for 1D input, do some sympy compute
        if node.input[0] in self.sympy_data_ and len(data_shape) == 1 and get_attribute(node, "axis", 0) == 0:
            idx = self._try_get_value(node, 1)
            if idx is not None:
                data = self.sympy_data_[node.input[0]]
                if type(data) is list:
                    if type(idx) is np.ndarray and len(idx.shape) == 1:
                        self.sympy_data_[node.output[0]] = [data[int(i)] for i in idx]
                    else:
                        self.sympy_data_[node.output[0]] = data[int(idx)]
                else:
                    assert idx == 0 or idx == -1
                    self.sympy_data_[node.output[0]] = data

    def _infer_GatherElements(self, node):  # noqa: N802
        indices_shape = self._get_shape(node, 1)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                indices_shape,
            )
        )

    def _infer_GatherND(self, node):  # noqa: N802
        data_shape = self._get_shape(node, 0)
        data_rank = len(data_shape)
        indices_shape = self._get_shape(node, 1)
        len(indices_shape)
        last_index_dimension = indices_shape[-1]
        assert is_literal(last_index_dimension) and last_index_dimension <= data_rank
        new_shape = indices_shape[:-1] + data_shape[last_index_dimension:]
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                new_shape,
            )
        )

    def _infer_If(self, node):  # noqa: N802
        # special case for constant condition, in case there are mismatching shape from the non-executed branch
        subgraphs = [
            get_attribute(node, "then_branch"),
            get_attribute(node, "else_branch"),
        ]
        cond = self._try_get_value(node, 0)
        if cond is not None:
            if as_scalar(cond) > 0:
                subgraphs[1].CopyFrom(subgraphs[0])
            else:
                subgraphs[0].CopyFrom(subgraphs[1])

        for i_sub, subgraph in enumerate(subgraphs):
            subgraph_infer = self._onnx_infer_subgraph(node, subgraph, use_node_input=False)
            for i_out in range(len(node.output)):
                vi = self.known_vi_[node.output[i_out]]
                if i_sub == 0:
                    vi.CopyFrom(subgraph.output[i_out])
                    vi.name = node.output[i_out]
                else:
                    self._fuse_tensor_type(node, i_out, vi.type, subgraph.output[i_out].type)

                # pass on sympy data from subgraph, if cond is constant
                if cond is not None and i_sub == (0 if as_scalar(cond) > 0 else 1):
                    if subgraph.output[i_out].name in subgraph_infer.sympy_data_:
                        self.sympy_data_[vi.name] = subgraph_infer.sympy_data_[subgraph.output[i_out].name]

    def _infer_Loop(self, node):  # noqa: N802
        subgraph = get_attribute(node, "body")
        assert len(subgraph.input) == len(node.input)
        num_loop_carried = len(node.input) - 2  # minus the length and initial loop condition
        # when sequence_type is used as loop carried input
        # needs to run subgraph infer twice if the tensor shape in sequence contains None
        for i, si in enumerate(subgraph.input):
            si_name = si.name
            si.CopyFrom(self.known_vi_[node.input[i]])
            si.name = si_name

        self._onnx_infer_subgraph(node, subgraph)

        # check subgraph input/output for shape changes in loop carried variables
        # for tensor_type, create new symbolic dim when changing, i.e., output = Concat(input, a)
        # for sequence_type, propagate from output to input
        need_second_infer = False
        for i_out in range(1, num_loop_carried + 1):
            so = subgraph.output[i_out]
            so_shape = get_shape_from_value_info(so)
            if is_sequence(so.type):
                if so_shape and None in so_shape:
                    # copy shape from output to input
                    # note that loop input is [loop_len, cond, input_0, input_1, ...]
                    # while loop output is [cond, output_0, output_1, ...]
                    subgraph.input[i_out + 1].type.sequence_type.elem_type.CopyFrom(so.type.sequence_type.elem_type)
                    need_second_infer = True
            else:
                si = subgraph.input[i_out + 1]
                si_shape = get_shape_from_value_info(si)
                for di, dims in enumerate(zip(si_shape, so_shape, strict=False)):
                    if dims[0] != dims[1]:
                        new_dim = onnx.TensorShapeProto.Dimension()
                        new_dim.dim_param = str(self._new_symbolic_dim_from_output(node, i_out, di))
                        si.type.tensor_type.shape.dim[di].CopyFrom(new_dim)
                        so.type.tensor_type.shape.dim[di].CopyFrom(new_dim)
                        need_second_infer = True

        if need_second_infer:
            if self.verbose_ > 2:
                logger.debug(
                    f"Rerun Loop: {node.name}({node.output[0]}...), because of sequence in loop carried variables"
                )
            self._onnx_infer_subgraph(node, subgraph, inc_subgraph_id=False)

        # create a new symbolic dimension for iteration dependent dimension
        loop_iter_dim = str(self._new_symbolic_dim_from_output(node))
        for i in range(len(node.output)):
            vi = self.known_vi_[node.output[i]]
            vi.CopyFrom(subgraph.output[i + 1])  # first subgraph output is condition, not in node output
            if i >= num_loop_carried:
                assert not is_sequence(vi.type)  # TODO: handle loop accumulation in sequence_type
                subgraph_vi_dim = subgraph.output[i + 1].type.tensor_type.shape.dim
                vi.type.tensor_type.shape.ClearField("dim")
                vi_dim = vi.type.tensor_type.shape.dim
                vi_dim.add().dim_param = loop_iter_dim
                vi_dim.extend(list(subgraph_vi_dim))
            vi.name = node.output[i]

    def _infer_MatMul(self, node):  # noqa: N802
        self._compute_matmul_shape(node)

    def _infer_MatMulInteger(self, node):  # noqa: N802
        self._compute_matmul_shape(node, onnx.TensorProto.INT32)

    def _infer_MatMulNBits(self, node):  # noqa: N802
        lhs_shape = self._get_shape(node, 0)
        rhs_shape = [get_attribute(node, "K"), get_attribute(node, "N")]
        lhs_rank = len(lhs_shape)
        assert lhs_rank > 0
        if lhs_rank == 1:
            new_shape = rhs_shape[1:]
        else:
            new_shape = lhs_shape[:-1] + rhs_shape[1:]
        # merge reduce dim
        self._check_merged_dims(
            [lhs_shape[-1], rhs_shape[0]],
            allow_broadcast=False,
        )
        # infer output_dtype from input type when not specified
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def _infer_NonMaxSuppression(self, node):  # noqa: N802
        selected = str(self._new_symbolic_dim_from_output(node))
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, [selected, 3]))

    def _infer_NonZero(self, node):  # noqa: N802
        input_rank = self._get_shape_rank(node, 0)
        # create a new symbolic dimension for NonZero output
        nz_len = str(self._new_symbolic_dim_from_output(node, 0, 1))
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], vi.type.tensor_type.elem_type, [input_rank, nz_len]))

    def _infer_OneHot(self, node):  # noqa: N802
        sympy_shape = self._get_sympy_shape(node, 0)
        depth = self._try_get_value(node, 1)
        axis = get_attribute(node, "axis", -1)
        axis = handle_negative_axis(axis, len(sympy_shape) + 1)
        new_shape = get_shape_from_sympy_shape(
            sympy_shape[:axis]
            + [self._new_symbolic_dim_from_output(node) if not is_literal(depth) else depth]
            + sympy_shape[axis:]
        )
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[2]].type.tensor_type.elem_type,
                new_shape,
            )
        )

    def _infer_Pad(self, node):  # noqa: N802
        if get_opset(self.out_mp_) <= 10:
            pads = get_attribute(node, "pads")
        else:
            pads = self._try_get_value(node, 1)

        sympy_shape = self._get_sympy_shape(node, 0)
        rank = len(sympy_shape)

        if pads is not None:
            assert len(pads) == 2 * rank
            new_sympy_shape = [
                d + pad_up + pad_down
                for d, pad_up, pad_down in zip(sympy_shape, pads[:rank], pads[rank:], strict=False)
            ]
            self._update_computed_dims(new_sympy_shape)
        else:
            # dynamic pads, create new symbolic dimensions
            new_sympy_shape = self._new_symbolic_shape(rank, node)
        output_tp = self.known_vi_[node.input[0]].type.tensor_type.elem_type

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(node.output[0], output_tp, get_shape_from_sympy_shape(new_sympy_shape))
        )

    def _infer_Pool(self, node):  # noqa: N802
        sympy_shape = self._compute_conv_pool_shape(node)
        self._update_computed_dims(sympy_shape)
        for o in node.output:
            if not o:
                continue
            vi = self.known_vi_[o]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    o,
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(sympy_shape),
                )
            )

    def _infer_aten_bitwise_or(self, node):
        shape0 = self._get_shape(node, 0)
        shape1 = self._get_shape(node, 1)
        new_shape = self._broadcast_shapes(shape0, shape1)
        t0 = self.known_vi_[node.input[0]]
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], t0.type.tensor_type.elem_type, new_shape))

    def _infer_aten_diagonal(self, node):
        sympy_shape = self._get_sympy_shape(node, 0)
        rank = len(sympy_shape)
        offset = self._try_get_value(node, 1)
        dim1 = self._try_get_value(node, 2)
        dim2 = self._try_get_value(node, 3)

        assert offset is not None and dim1 is not None and dim2 is not None
        dim1 = handle_negative_axis(dim1, rank)
        dim2 = handle_negative_axis(dim2, rank)

        new_shape = []
        for dim, val in enumerate(sympy_shape):
            if dim not in [dim1, dim2]:
                new_shape.append(val)

        shape1 = sympy_shape[dim1]
        shape2 = sympy_shape[dim2]
        if offset >= 0:
            diag_shape = sympy.Max(0, sympy.Min(shape1, shape2 - offset))
        else:
            diag_shape = sympy.Max(0, sympy.Min(shape1 + offset, shape2))
        new_shape.append(diag_shape)

        if node.output[0]:
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_shape),
                )
            )

    def _infer_aten_multinomial(self, node):
        sympy_shape = self._get_sympy_shape(node, 0)
        rank = len(sympy_shape)
        assert rank in [1, 2]
        num_samples = self._try_get_value(node, 1)
        di = rank - 1
        last_dim = num_samples if num_samples else str(self._new_symbolic_dim_from_output(node, 0, di))
        output_shape = sympy_shape[:-1] + [last_dim]
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                onnx.TensorProto.INT64,
                get_shape_from_sympy_shape(output_shape),
            )
        )

    def _infer_aten_pool2d(self, node):
        sympy_shape = self._get_sympy_shape(node, 0)
        assert len(sympy_shape) == 4
        sympy_shape[-2:] = [self._new_symbolic_dim_from_output(node, 0, i) for i in [2, 3]]
        self._update_computed_dims(sympy_shape)
        for i, o in enumerate(node.output):
            if not o:
                continue
            vi = self.known_vi_[o]
            elem_type = onnx.TensorProto.INT64 if i == 1 else self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi.CopyFrom(helper.make_tensor_value_info(o, elem_type, get_shape_from_sympy_shape(sympy_shape)))

    def _infer_aten_minmax(self, node):
        vi = self.known_vi_[node.output[0]]
        if len(node.input) == 1:
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0], self.known_vi_[node.input[0]].type.tensor_type.elem_type, []
                )
            )
        else:
            assert len(node.input) == 3
            keepdim = self._try_get_value(node, 2)
            assert keepdim is not None  # can only handle known keepdim case.
            dim = self._try_get_value(node, 1)
            if dim is None:
                rank = self._get_shape_rank(node, 0)
                output_shape = self._new_symbolic_shape(rank if keepdim else rank - 1, node)
            else:
                shape = self._get_sympy_shape(node, 0)
                dim = handle_negative_axis(dim, len(shape))
                output_shape = shape[:dim]
                if keepdim:
                    output_shape += [1]
                output_shape += shape[dim + 1 :]

            output_shape = get_shape_from_sympy_shape(output_shape)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0], self.known_vi_[node.input[0]].type.tensor_type.elem_type, output_shape
                )
            )
            vi1 = self.known_vi_[node.output[1]]
            vi1.CopyFrom(helper.make_tensor_value_info(node.output[1], onnx.TensorProto.INT64, output_shape))

    def _infer_aten_unfold(self, node):
        sympy_shape = self._get_sympy_shape(node, 0)
        dimension = self._try_get_value(node, 1)
        size = self._try_get_value(node, 2)
        step = self._try_get_value(node, 3)
        if dimension is not None and size is not None and step is not None:
            assert dimension < len(sympy_shape)
            sympy_shape[dimension] = (sympy_shape[dimension] - size) // step + 1
            sympy_shape.append(size)
        else:
            rank = len(sympy_shape)
            sympy_shape = self._new_symbolic_shape(rank + 1, node)
        self._update_computed_dims(sympy_shape)
        if node.output[0]:
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(sympy_shape),
                )
            )

    def _infer_aten_argmax(self, node):
        new_shape = None
        if not node.input[1]:
            # The argmax of the flattened input is returned.
            new_shape = []
        else:
            dim = self._try_get_value(node, 1)
            keepdim = self._try_get_value(node, 2)
            if keepdim is not None:
                sympy_shape = self._get_sympy_shape(node, 0)
                if dim is not None:
                    dim = handle_negative_axis(dim, len(sympy_shape))
                    if keepdim:
                        sympy_shape[dim] = 1
                    else:
                        del sympy_shape[dim]
                else:
                    rank = len(sympy_shape)
                    sympy_shape = self._new_symbolic_shape(rank if keepdim else rank - 1, node)
                self._update_computed_dims(sympy_shape)
                new_shape = get_shape_from_sympy_shape(sympy_shape)
        if node.output[0] and new_shape is not None:
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, new_shape))

    def _infer_aten_group_norm(self, node):
        self._propagate_shape_and_type(node)
        input_shape = self._get_shape(node, 0)
        N = input_shape[0] if input_shape is not None and len(input_shape) != 0 else None  # noqa: N806
        group = self._try_get_value(node, 6)
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        for i in [1, 2]:
            if node.output[i]:
                vi = self.known_vi_[node.output[i]]
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[i],
                        output_dtype,
                        [
                            N if N is not None else str(self._new_symbolic_dim_from_output(node, i, 0)),
                            (
                                as_scalar(group)
                                if group is not None
                                else str(self._new_symbolic_dim_from_output(node, i, 1))
                            ),
                        ],
                    )
                )

    def _infer_aten_upsample(self, node):
        new_shape = None
        input_shape = self._get_shape(node, 0)
        if input_shape is not None:
            new_shape = input_shape[:2]
            output_size = self._try_get_value(node, 1)
            if output_size is not None:
                new_shape += [dim_size.item() if type(dim_size) is np.int64 else dim_size for dim_size in output_size]
            else:
                rank = len(input_shape)
                new_shape += [str(self._new_symbolic_dim_from_output(node, 0, i)) for i in range(2, rank)]
        if node.output[0] and new_shape is not None:
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def _infer_BatchNormalization(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

        # this works for opsets < 14 and 14 since we check i < len(node.output) in the loop
        for i in [1, 2, 3, 4]:
            if i < len(node.output) and node.output[i]:
                # all of these parameters have the same shape as the 1st input
                self._propagate_shape_and_type(node, input_index=1, output_index=i)

    def _infer_Range(self, node):  # noqa: N802
        vi = self.known_vi_[node.output[0]]
        input_data = self._get_int_or_float_values(node)
        if all(i is not None for i in input_data):
            start = as_scalar(input_data[0])
            limit = as_scalar(input_data[1])
            delta = as_scalar(input_data[2])
            new_sympy_shape = [sympy.Max(sympy.ceiling((limit - start) / delta), 0)]
        else:
            new_sympy_shape = [self._new_symbolic_dim_from_output(node)]
        self._update_computed_dims(new_sympy_shape)
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

    def _infer_ReduceSum(self, node):  # noqa: N802
        keep_dims = get_attribute(node, "keepdims", 1)
        if get_opset(self.out_mp_) >= 13 and len(node.input) > 1:
            # ReduceSum changes axes to input[1] in opset 13
            axes = self._try_get_value(node, 1)
            vi = self.known_vi_[node.output[0]]
            if axes is None:
                assert keep_dims  # can only handle keep_dims==True when axes is unknown, by generating new ranks
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        get_shape_from_sympy_shape(self._new_symbolic_shape(self._get_shape_rank(node, 0), node)),
                    )
                )
            else:
                shape = self._get_shape(node, 0)
                output_shape = []
                axes = [handle_negative_axis(a, len(shape)) for a in axes]
                for i, d in enumerate(shape):
                    if i in axes:
                        if keep_dims:
                            output_shape.append(1)
                    else:
                        output_shape.append(d)
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        output_shape,
                    )
                )

    def _infer_ReduceMean(self, node):  # noqa: N802
        if get_opset(self.out_mp_) >= 18:
            # reduce mean spec 18+ is same as reduce sum spec 13+
            self._infer_ReduceSum(node)

    def _infer_ReduceProd(self, node):  # noqa: N802
        axes = get_attribute(node, "axes")
        keep_dims = get_attribute(node, "keepdims", 1)
        if keep_dims == 0 and axes == [0]:
            data = self._get_int_or_float_values(node)[0]
            if data is not None:
                self.sympy_data_[node.output[0]] = sympy_reduce_product(data)

    def _infer_RelativePositionBias(self, node):  # noqa: N802
        seq_len = self._try_get_value(node, 1)
        real_seq_len = self._try_get_value(node, 2)
        if seq_len is None or real_seq_len is None:
            return
        num_heads = self._get_sympy_shape(node, 0)[1]

        new_shape = [1, num_heads, str(seq_len), str(real_seq_len)]

        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    def _infer_Reshape(self, node):  # noqa: N802
        shape_value = self._try_get_value(node, 1)
        vi = self.known_vi_[node.output[0]]
        if shape_value is None:
            shape_shape = self._get_shape(node, 1)
            assert len(shape_shape) == 1
            shape_rank = shape_shape[0]
            assert is_literal(shape_rank)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(self._new_symbolic_shape(shape_rank, node)),
                )
            )
        else:
            input_sympy_shape = self._get_sympy_shape(node, 0)
            total = 1
            for d in input_sympy_shape:
                total = total * d
            new_sympy_shape = []
            deferred_dim_idx = -1
            non_deferred_size = 1
            for i, d in enumerate(shape_value):
                if type(d) is sympy.Symbol:
                    new_sympy_shape.append(d)
                elif d == 0:
                    new_sympy_shape.append(input_sympy_shape[i])
                    non_deferred_size = non_deferred_size * input_sympy_shape[i]
                else:
                    new_sympy_shape.append(d)
                if d == -1:
                    deferred_dim_idx = i
                elif d != 0:
                    non_deferred_size = non_deferred_size * d

            assert new_sympy_shape.count(-1) < 2
            if -1 in new_sympy_shape:
                new_dim = total // non_deferred_size
                new_sympy_shape[deferred_dim_idx] = new_dim

            self._update_computed_dims(new_sympy_shape)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_sympy_shape),
                )
            )

        self._pass_on_sympy_data(node)

    def _infer_Resize(self, node):  # noqa: N802
        vi = self.known_vi_[node.output[0]]
        input_sympy_shape = self._get_sympy_shape(node, 0)
        if get_opset(self.out_mp_) <= 10:
            scales = self._try_get_value(node, 1)
            if scales is not None:
                new_sympy_shape = [
                    sympy.simplify(sympy.floor(d * s)) for d, s in zip(input_sympy_shape, scales, strict=False)
                ]
                self._update_computed_dims(new_sympy_shape)
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        get_shape_from_sympy_shape(new_sympy_shape),
                    )
                )
        else:
            roi = self._try_get_value(node, 1)
            scales = self._try_get_value(node, 2)
            sizes = self._try_get_value(node, 3)
            if sizes is not None:
                new_sympy_shape = [sympy.simplify(sympy.floor(s)) for s in sizes]
                self._update_computed_dims(new_sympy_shape)
            elif scales is not None:
                rank = len(scales)
                if get_attribute(node, "coordinate_transformation_mode") == "tf_crop_and_resize":
                    assert len(roi) == 2 * rank
                    roi_start = list(roi)[:rank]
                    roi_end = list(roi)[rank:]
                else:
                    roi_start = [0] * rank
                    roi_end = [1] * rank
                scales = list(scales)
                new_sympy_shape = [
                    sympy.simplify(sympy.floor(d * (end - start) * scale))
                    for d, start, end, scale in zip(input_sympy_shape, roi_start, roi_end, scales, strict=False)
                ]
                self._update_computed_dims(new_sympy_shape)
            else:
                new_sympy_shape = self._new_symbolic_shape(self._get_shape_rank(node, 0), node)

            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_sympy_shape),
                )
            )

    def _infer_Scan(self, node):  # noqa: N802
        subgraph = get_attribute(node, "body")
        num_scan_inputs = get_attribute(node, "num_scan_inputs")
        scan_input_axes = get_attribute(node, "scan_input_axes", [0] * num_scan_inputs)
        num_scan_states = len(node.input) - num_scan_inputs
        scan_input_axes = [
            handle_negative_axis(ax, self._get_shape_rank(node, i + num_scan_states))
            for i, ax in enumerate(scan_input_axes)
        ]
        # We may have cases where the subgraph has optional inputs that appear in both subgraph's input and initializer,
        # but not in the node's input. In such cases, the input model might be invalid, but let's skip those optional inputs.
        assert len(subgraph.input) >= len(node.input)
        subgraph_inputs = subgraph.input[: len(node.input)]
        for i, si in enumerate(subgraph_inputs):
            subgraph_name = si.name
            si.CopyFrom(self.known_vi_[node.input[i]])
            if i >= num_scan_states:
                scan_input_dim = si.type.tensor_type.shape.dim
                scan_input_dim.remove(scan_input_dim[scan_input_axes[i - num_scan_states]])
            si.name = subgraph_name
        self._onnx_infer_subgraph(node, subgraph)
        num_scan_outputs = len(node.output) - num_scan_states
        scan_output_axes = get_attribute(node, "scan_output_axes", [0] * num_scan_outputs)
        scan_input_dim = get_shape_from_type_proto(self.known_vi_[node.input[-1]].type)[scan_input_axes[-1]]
        for i, o in enumerate(node.output):
            vi = self.known_vi_[o]
            if i >= num_scan_states:
                shape = get_shape_from_type_proto(subgraph.output[i].type)
                new_dim = handle_negative_axis(scan_output_axes[i - num_scan_states], len(shape) + 1)
                shape = shape[:new_dim] + [scan_input_dim] + shape[new_dim:]
                vi.CopyFrom(helper.make_tensor_value_info(o, subgraph.output[i].type.tensor_type.elem_type, shape))
            else:
                vi.CopyFrom(subgraph.output[i])
            vi.name = o

    def _infer_ScatterElements(self, node):  # noqa: N802
        data_shape = self._get_shape(node, 0)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                data_shape,
            )
        )

    def _infer_SequenceAt(self, node):  # noqa: N802
        # need to create new symbolic dimension if sequence shape has None:
        seq_shape = self._get_shape(node, 0)
        vi = self.known_vi_[node.output[0]]
        if seq_shape is not None:
            for di, d in enumerate(seq_shape):
                if d is not None:
                    continue
                new_dim = onnx.TensorShapeProto.Dimension()
                new_dim.dim_param = str(self._new_symbolic_dim_from_output(node, 0, di))
                vi.type.tensor_type.shape.dim[di].CopyFrom(new_dim)

    def _infer_SequenceInsert(self, node):  # noqa: N802
        # workaround bug in onnx's shape inference
        vi_seq = self.known_vi_[node.input[0]]
        vi_tensor = self.known_vi_[node.input[1]]
        vi_out_seq = self.known_vi_[node.output[0]]
        vi_out_seq.CopyFrom(vi_seq)
        vi_out_seq.name = node.output[0]
        self._fuse_tensor_type(node, 0, vi_out_seq.type, vi_tensor.type)

    def _infer_Shape(self, node):  # noqa: N802
        self.sympy_data_[node.output[0]] = self._get_sympy_shape(node, 0)

    def _infer_Size(self, node):  # noqa: N802
        sympy_shape = self._get_sympy_shape(node, 0)
        self.sympy_data_[node.output[0]] = sympy_reduce_product(sympy_shape)
        self.known_vi_[node.output[0]].CopyFrom(
            helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, [])
        )

    def _infer_Slice(self, node):  # noqa: N802
        # SymPy fails to prove that `x_0 + ... + x_n >= 0` if one of `x_i` is a `sympy.Min(a, b)`,
        # even when the relation holds for both `a` and `b`.
        #
        # When given `expr` of form `min(a, b) + ...`, this function returns `[a + ..., b + ...]`,
        # so that we can prove inequalities for both expressions separately.
        #
        # If the number of `min(...)` subexpressions is not exactly one, this function just returns `[expr]`.
        def flatten_min(expr):
            assert isinstance(expr, sympy.Add), f"Expected a sum of two arguments, got {expr}"
            min_positions = [idx for idx in range(len(expr.args)) if isinstance(expr.args[idx], sympy.Min)]
            if len(min_positions) == 1:
                min_pos = min_positions[0]

                def replace_min_with_arg(arg_idx):
                    replaced = list(expr.args)
                    assert isinstance(replaced[min_pos], sympy.Min), (
                        f"Expected a sympy.Min() at position {min_pos}, got {replaced[min_pos]}"
                    )
                    assert len(replaced[min_pos].args) == 2, (
                        f"Expected a sympy.Min() with exactly 2 arguments, got {replaced[min_pos]}"
                    )
                    replaced[min_pos] = replaced[min_pos].args[arg_idx]
                    return sympy.Add(*replaced)

                return [
                    replace_min_with_arg(0),
                    replace_min_with_arg(1),
                ]
            return [expr]

        def less_equal(x, y):
            try:
                return bool(x <= y)
            except TypeError:
                pass
            try:
                return bool(y >= x)
            except TypeError:
                pass
            try:
                return bool(-x >= -y)
            except TypeError:
                pass
            try:
                return bool(-y <= -x)
            except TypeError:
                pass
            try:
                return bool(y - x >= 0)
            except TypeError:
                # the last attempt; this may raise TypeError
                return all(bool(d >= 0) for d in flatten_min(y - x))

        def handle_negative_index(index, bound):
            """normalizes a negative index to be in [0, bound)"""
            try:
                if not less_equal(0, index):
                    if is_literal(index) and index <= -self.int_max_:
                        # this case is handled separately
                        return index
                    return bound + index
            except TypeError:
                logger.warning(f"Cannot determine if {index} < 0")
            return index

        if get_opset(self.out_mp_) <= 9:
            axes = get_attribute(node, "axes")
            starts = get_attribute(node, "starts")
            ends = get_attribute(node, "ends")
            if not axes:
                axes = list(range(len(starts)))
            steps = [1] * len(axes)
        else:
            starts = as_list(self._try_get_value(node, 1), keep_none=True)
            ends = as_list(self._try_get_value(node, 2), keep_none=True)
            axes = self._try_get_value(node, 3)
            steps = self._try_get_value(node, 4)
            if axes is None and not (starts is None and ends is None):
                axes = list(range(len(starts if starts is not None else ends)))
            if steps is None and not (starts is None and ends is None):
                steps = [1] * len(starts if starts is not None else ends)
            axes = as_list(axes, keep_none=True)
            steps = as_list(steps, keep_none=True)

        new_sympy_shape = self._get_sympy_shape(node, 0)
        if starts is None or ends is None:
            if axes is None:
                for i in range(len(new_sympy_shape)):
                    new_sympy_shape[i] = self._new_symbolic_dim_from_output(node, 0, i)
            else:
                new_sympy_shape = get_shape_from_sympy_shape(new_sympy_shape)
                for i in axes:
                    new_sympy_shape[i] = self._new_symbolic_dim_from_output(node, 0, i)
        else:
            for i, s, e, t in zip(axes, starts, ends, steps, strict=False):
                e = handle_negative_index(e, new_sympy_shape[i])  # noqa: PLW2901
                if is_literal(e):
                    if e >= self.int_max_:
                        e = new_sympy_shape[i]  # noqa: PLW2901
                    elif e <= -self.int_max_:
                        e = 0 if s > 0 else -1  # noqa: PLW2901
                    elif is_literal(new_sympy_shape[i]):
                        if e < 0:
                            e = max(0, e + new_sympy_shape[i])  # noqa: PLW2901
                        e = min(e, new_sympy_shape[i])  # noqa: PLW2901
                    else:
                        if e > 0:
                            e = (  # noqa: PLW2901
                                sympy.Min(e, new_sympy_shape[i]) if e > 1 else e
                            )  # special case for slicing first to make computation easier
                else:
                    if is_literal(new_sympy_shape[i]):
                        e = sympy.Min(e, new_sympy_shape[i])  # noqa: PLW2901
                    else:
                        try:
                            if not less_equal(e, new_sympy_shape[i]):
                                e = new_sympy_shape[i]  # noqa: PLW2901
                        except Exception:
                            logger.warning(f"Unable to determine if {e} <= {new_sympy_shape[i]}, treat as equal")
                            e = new_sympy_shape[i]  # noqa: PLW2901

                s = handle_negative_index(s, new_sympy_shape[i])  # noqa: PLW2901
                if is_literal(new_sympy_shape[i]) and is_literal(s):
                    s = max(0, min(s, new_sympy_shape[i]))  # noqa: PLW2901

                new_sympy_shape[i] = sympy.simplify((e - s + t + (-1 if t > 0 else 1)) // t)

            self._update_computed_dims(new_sympy_shape)

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

        # handle sympy_data if needed, for slice in shape computation
        if (
            node.input[0] in self.sympy_data_
            and axes == [0]
            and starts is not None
            and len(starts) == 1
            and ends is not None
            and len(ends) == 1
            and steps is not None
            and len(steps) == 1
        ):
            input_sympy_data = self.sympy_data_[node.input[0]]
            if type(input_sympy_data) is list or (
                type(input_sympy_data) is np.array and len(input_sympy_data.shape) == 1
            ):
                self.sympy_data_[node.output[0]] = input_sympy_data[starts[0] : ends[0] : steps[0]]

    def _infer_SoftmaxCrossEntropyLoss(self, node):  # noqa: N802
        vi = self.known_vi_[node.output[0]]
        elem_type = self.known_vi_[node.input[0]].type.tensor_type.elem_type

        # If output type is explicit specified in attribute, we use it as output tensor type.
        specified_output_type = get_attribute(node, "output_type", None)
        if specified_output_type is not None:
            elem_type = specified_output_type

        vi.type.tensor_type.elem_type = elem_type
        vi.type.tensor_type.shape.CopyFrom(onnx.TensorShapeProto())

        if len(node.output) > 1:
            data_shape = self._get_shape(node, 0)
            vi = self.known_vi_[node.output[1]]
            vi.CopyFrom(helper.make_tensor_value_info(vi.name, elem_type, data_shape))

    def _infer_Split_Common(self, node, make_value_info_func):  # noqa: N802
        input_sympy_shape = self._get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis", 0), len(input_sympy_shape))
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'split' are provided as attribute or via 2nd input
        if op_set < 13:
            split = get_attribute(node, "split")
            assert self._try_get_value(node, 1) is None
        else:
            split = self._try_get_value(node, 1)
            assert get_attribute(node, "split") is None

        if split is None:
            num_outputs = len(node.output)
            split = [input_sympy_shape[axis] / sympy.Integer(num_outputs)] * num_outputs
            self._update_computed_dims(split)
        else:
            split = [sympy.Integer(s) for s in split]

        for i_o in range(len(split)):
            vi = self.known_vi_[node.output[i_o]]
            vi.CopyFrom(
                make_value_info_func(
                    node.output[i_o],
                    self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(input_sympy_shape[:axis] + [split[i_o]] + input_sympy_shape[axis + 1 :]),
                )
            )
            self.known_vi_[vi.name] = vi

    def _infer_Split(self, node):  # noqa: N802
        self._infer_Split_Common(node, helper.make_tensor_value_info)

    def _infer_SplitToSequence(self, node):  # noqa: N802
        self._infer_Split_Common(node, helper.make_sequence_value_info)

    def _infer_Squeeze(self, node):  # noqa: N802
        input_shape = self._get_shape(node, 0)
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'axes' are provided as attribute or via 2nd input
        if op_set < 13:
            axes = get_attribute(node, "axes")
            assert self._try_get_value(node, 1) is None
        else:
            axes = self._try_get_value(node, 1)
            assert get_attribute(node, "axes") is None

        if axes is None:
            # No axes have been provided (neither via attribute nor via input).
            # In this case the 'Shape' op should remove all axis with dimension 1.
            # For symbolic dimensions we guess they are !=1.
            output_shape = [s for s in input_shape if s != 1]
            if self.verbose_ > 0:
                symbolic_dimensions = [s for s in input_shape if type(s) != int]  # noqa: E721
                if len(symbolic_dimensions) > 0:
                    logger.debug(
                        f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
                        f"Assuming the following dimensions are never equal to 1: {symbolic_dimensions}"
                    )
        else:
            axes = [handle_negative_axis(a, len(input_shape)) for a in axes]
            output_shape = []
            for i in range(len(input_shape)):
                if i not in axes:
                    output_shape.append(input_shape[i])
                else:
                    assert input_shape[i] == 1 or type(input_shape[i]) != int  # noqa: E721
                    if self.verbose_ > 0 and type(input_shape[i]) != int:  # noqa: E721
                        logger.debug(
                            f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
                            f"Assuming the dimension '{input_shape[i]}' at index {i} of the input to be equal to 1."
                        )

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )
        self._pass_on_sympy_data(node)

    def _infer_Tile(self, node):  # noqa: N802
        repeats_value = self._try_get_value(node, 1)
        new_sympy_shape = []
        if repeats_value is not None:
            input_sympy_shape = self._get_sympy_shape(node, 0)
            for i, d in enumerate(input_sympy_shape):
                new_dim = d * repeats_value[i]
                new_sympy_shape.append(new_dim)
            self._update_computed_dims(new_sympy_shape)
        else:
            new_sympy_shape = self._new_symbolic_shape(self._get_shape_rank(node, 0), node)
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

    def _infer_TopK(self, node):  # noqa: N802
        rank = self._get_shape_rank(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis", -1), rank)
        new_shape = self._get_shape(node, 0)

        if get_opset(self.out_mp_) <= 9:
            k = get_attribute(node, "k")
        else:
            k = self._get_int_or_float_values(node)[1]

        if k is None:
            k = self._new_symbolic_dim_from_output(node)
        else:
            k = as_scalar(k)

        if type(k) in [int, str]:
            new_shape[axis] = k
        else:
            new_sympy_shape = self._get_sympy_shape(node, 0)
            new_sympy_shape[axis] = k
            self._update_computed_dims(
                new_sympy_shape
            )  # note that TopK dim could be computed in sympy_data, so need to update computed_dims when it enters shape
            new_shape = get_shape_from_sympy_shape(new_sympy_shape)

        for i_o in range(len(node.output)):
            vi = self.known_vi_[node.output[i_o]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[i_o], vi.type.tensor_type.elem_type, new_shape))

    def _infer_Transpose(self, node):  # noqa: N802
        if node.input[0] in self.sympy_data_:
            data_shape = self._get_shape(node, 0)
            perm = get_attribute(node, "perm", reversed(list(range(len(data_shape)))))
            input_data = self.sympy_data_[node.input[0]]
            self.sympy_data_[node.output[0]] = (
                np.transpose(np.array(input_data).reshape(*data_shape), axes=tuple(perm)).flatten().tolist()
            )

    def _infer_Unsqueeze(self, node):  # noqa: N802
        input_shape = self._get_shape(node, 0)
        op_set = get_opset(self.out_mp_)

        # Depending on op-version 'axes' are provided as attribute or via 2nd input
        if op_set < 13:
            axes = get_attribute(node, "axes")
            assert self._try_get_value(node, 1) is None
        else:
            axes = self._try_get_value(node, 1)
            assert get_attribute(node, "axes") is None

        output_rank = len(input_shape) + len(axes)
        axes = [handle_negative_axis(a, output_rank) for a in axes]

        input_axis = 0
        output_shape = []
        for i in range(output_rank):
            if i in axes:
                output_shape.append(1)
            else:
                output_shape.append(input_shape[input_axis])
                input_axis += 1

        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                self.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )

        self._pass_on_sympy_data(node)

    def _infer_ZipMap(self, node):  # noqa: N802
        map_key_type = None
        if get_attribute(node, "classlabels_int64s") is not None:
            map_key_type = onnx.TensorProto.INT64
        elif get_attribute(node, "classlabels_strings") is not None:
            map_key_type = onnx.TensorProto.STRING

        assert map_key_type is not None
        new_vi = onnx.ValueInfoProto()
        new_vi.name = node.output[0]
        new_vi.type.sequence_type.elem_type.map_type.value_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        new_vi.type.sequence_type.elem_type.map_type.key_type = map_key_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(new_vi)

    def _infer_Attention(self, node):  # noqa: N802
        shape = self._get_shape(node, 0)
        shape_weights = self._get_shape(node, 1)
        shape_bias = self._try_get_shape(node, 2)
        if shape_bias is not None:
            assert len(shape_bias) == 1
        tripled_hidden_size = shape_bias[0] if shape_bias is not None else shape_weights[1]
        if shape and len(shape) == 3:
            qkv_hidden_sizes_attr = get_attribute(node, "qkv_hidden_sizes")
            if qkv_hidden_sizes_attr is not None:
                assert len(qkv_hidden_sizes_attr) == 3
                shape[2] = int(qkv_hidden_sizes_attr[2])
            elif isinstance(tripled_hidden_size, int):
                shape[2] = int(tripled_hidden_size / 3)
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, shape))

            if len(node.output) > 1:
                # input shape: (batch_size, sequence_length, hidden_size)
                # past shape: (2, batch_size, num_heads, past_sequence_length, head_size)
                # mask shape: (batch_size, total_sequence_length) or (batch_size, sequence_length, total_sequence_length) or (batch_size, 1, max_seq_len, max_seq_len)
                # present shape: (2, batch_size, num_heads, total_sequence_length, head_size), where total_sequence_length=sequence_length+past_sequence_length
                input_shape = self._get_shape(node, 0)
                past_shape = self._get_shape(node, 4) if len(node.input) > 4 and node.input[4] else []
                mask_shape = self._get_shape(node, 3) if len(node.input) > 3 and node.input[3] else []

                if past_shape and len(past_shape) == 5:
                    if mask_shape and len(mask_shape) in [2, 3]:
                        past_shape[3] = mask_shape[-1]
                    elif input_shape and len(input_shape) == 3:
                        if isinstance(input_shape[1], int) and isinstance(past_shape[3], int):
                            past_shape[3] = input_shape[1] + past_shape[3]
                        else:
                            past_shape[3] = f"{past_shape[3]}+{input_shape[1]}"
                    vi = self.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))
                # No past input but present output still exists
                else:
                    num_heads = get_attribute(node, "num_heads")
                    head_size = input_shape[2] // num_heads
                    present_shape = [2, input_shape[0], num_heads, input_shape[1], head_size]
                    vi = self.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, present_shape))

    def _infer_GatedRelativePositionBias(self, node):  # noqa: N802
        # When padding is removed:
        #   query_layer: (token_count, num_heads x head_size)
        #   token_offset: (batch_size, seq_len)
        # Otherwise:
        #   query_layer: (batch_size, seq_len, num_heads x head_size)
        #   token_offset: None
        # Output shape: (batch_size, num_heads, seq_len, seq_len)
        num_heads = get_attribute(node, "num_heads")

        token_offset_shape = self._try_get_shape(node, 6)
        if token_offset_shape is not None:
            output_shape = [token_offset_shape[0], num_heads, token_offset_shape[1], token_offset_shape[1]]
        else:
            query_layer_shape = self._get_shape(node, 0)
            assert query_layer_shape is not None and len(query_layer_shape) == 3
            output_shape = [query_layer_shape[0], num_heads, query_layer_shape[1], query_layer_shape[1]]

        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_PackedAttention(self, node):  # noqa: N802
        shape = self._get_shape(node, 0)
        shape_weights = self._get_shape(node, 1)
        shape_bias = self._try_get_shape(node, 2)
        if shape_bias is not None:
            assert len(shape_bias) == 1
        tripled_hidden_size = shape_bias[0] if shape_bias is not None else shape_weights[1]
        if shape and len(shape) == 2:
            qkv_hidden_sizes_attr = get_attribute(node, "qkv_hidden_sizes")
            if qkv_hidden_sizes_attr is not None:
                assert len(qkv_hidden_sizes_attr) == 3
                shape[1] = int(qkv_hidden_sizes_attr[2])
            elif isinstance(tripled_hidden_size, int):
                shape[1] = int(tripled_hidden_size / 3)
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, shape))

    def _infer_PackedMultiHeadAttention(self, node):  # noqa: N802
        shape_value = self._try_get_shape(node, 2)
        if shape_value is not None and len(shape_value) == 2:
            output_shape = shape_value
        else:
            shape_query = self._get_shape(node, 0)
            assert shape_query is not None and len(shape_query) == 4
            output_shape = [shape_query[0], shape_query[1] * shape_query[3]]

        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_RemovePadding(self, node):  # noqa: N802
        shape = self._get_shape(node, 0)
        if shape and len(shape) == 3:
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, ["token_count", shape[2]]))

            vi_token_offset = self.known_vi_[node.output[1]]
            vi_token_offset.CopyFrom(
                helper.make_tensor_value_info(node.output[1], onnx.TensorProto.INT32, [shape[0], shape[1]])
            )

            vi_cumulated_seq_len = self.known_vi_[node.output[2]]
            vi_cumulated_seq_len.CopyFrom(
                helper.make_tensor_value_info(node.output[2], onnx.TensorProto.INT32, ["batch_size + 1"])
            )

            vi_max_seq_len = self.known_vi_[node.output[3]]
            vi_max_seq_len.CopyFrom(helper.make_tensor_value_info(node.output[3], onnx.TensorProto.INT32, [1]))

    def _infer_RestorePadding(self, node):  # noqa: N802
        shape_input = self._get_shape(node, 0)
        shape_token_offset = self._get_shape(node, 1)
        if shape_input and len(shape_input) == 2 and shape_token_offset and len(shape_token_offset) == 2:
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = self.known_vi_[node.output[0]]

            output_shape = [shape_token_offset[0], shape_token_offset[1], shape_input[1]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_BiasGelu(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

    def _infer_MultiHeadAttention(self, node):  # noqa: N802
        # Output 0 has shape (batch_size, sequence_length, v_hidden_size)
        # Q, K and V without packing:
        #   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
        #   Input 1 (key) has shape (batch_size, kv_sequence_length, hidden_size) or (batch_size, num_heads, kv_sequence_length, head_size)
        #   Input 2 (value) has shape (batch_size, kv_sequence_length, v_hidden_size) or (batch_size, num_heads, kv_sequence_length, head_size)
        # Packed KV:
        #   Input 0 (query) has shape (batch_size, sequence_length, hidden_size)
        #   Input 1 (batch_size, kv_sequence_length, num_heads, 2, head_size)
        #   Input 2  nullptr
        # Packed QKV:
        #   Input 0 (batch_size, sequence_length, num_heads, 3, head_size)
        #   Input 1  nullptr
        #   Input 2  nullptr

        query_shape = self._get_shape(node, 0)
        total_sequence_length = None
        output_dtype = None
        if query_shape is not None:
            if len(query_shape) == 3:
                key_shape = self._try_get_shape(node, 1)
                # By default, hidden size is same for Q/K/V. Only need check v_hidden_size when value is provided.
                output_shape = query_shape
                if key_shape is not None and len(key_shape) == 3:
                    value_shape = self._try_get_shape(node, 2)
                    if value_shape is not None and len(value_shape) == 3:
                        output_shape[2] = value_shape[2]
                    total_sequence_length = key_shape[1]

                output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
                vi = self.known_vi_[node.output[0]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

            elif len(query_shape) == 5:
                if isinstance(query_shape[2], int) and isinstance(query_shape[4], int):
                    output_shape = [query_shape[0], query_shape[1], query_shape[2] * query_shape[4]]
                else:
                    output_shape = [query_shape[0], query_shape[1], f"{query_shape[2]}*{query_shape[4]}"]

                total_sequence_length = query_shape[1]

                output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
                vi = self.known_vi_[node.output[0]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

            if len(node.output) > 1:
                batch_size = query_shape[0]
                num_heads = get_attribute(node, "num_heads")

                head_size = None
                if len(query_shape) == 3:
                    head_size = (
                        int(query_shape[2] / num_heads)
                        if isinstance(query_shape[2], int)
                        else f"{query_shape[2]}/{num_heads}"
                    )
                else:
                    head_size = query_shape[4]

                past_shape = self._try_get_shape(node, 6)

                if past_shape is not None:
                    if isinstance(past_shape[2], int) and isinstance(total_sequence_length, int):
                        total_sequence_length = past_shape[2] + total_sequence_length
                    else:
                        total_sequence_length = f"{past_shape[2]}+{total_sequence_length}"

                present_shape = [batch_size, num_heads, total_sequence_length, head_size]

                assert output_dtype is not None
                if len(node.output) > 2 and node.output[1] and node.output[2]:
                    vi = self.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, present_shape))
                    vi = self.known_vi_[node.output[2]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, present_shape))

    def _infer_DecoderMaskedMultiHeadAttention(self, node):  # noqa: N802
        # Output 0 has shape (batch_size, 1, v_hidden_size)
        # Q, K and V without packing:
        #   Input 0 (query) has shape (batch_size, 1, hidden_size)
        #   Input 5 (past_key) if exists has shape (batch_size, num_heads, max_sequence_length, head_size)

        query_shape = self._get_shape(node, 0)
        if query_shape is not None:
            output_shape = query_shape
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            assert output_dtype is not None
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

            if len(node.output) > 2 and node.output[1] and node.output[2]:
                past_shape = self._try_get_shape(node, 5)
                if past_shape is not None:
                    vi = self.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))
                    vi = self.known_vi_[node.output[2]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))

    def _infer_UnfoldTensor(self, node):  # noqa: N802
        input_shape = self._get_shape(node, 0)
        if input_shape is not None:
            output_shape = input_shape.copy()
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            assert output_dtype is not None

            rank, dim, size, step = len(input_shape), None, None, None
            for attr in node.attribute:
                if attr.name == "dim":
                    dim = attr.i
                    dim = rank + dim if dim == -1 else dim
                elif attr.name == "size":
                    size = attr.i
                elif attr.name == "step":
                    step = attr.i

            output_shape.append(size)
            output_shape[dim] = (input_shape[dim] - size) // step + 1

            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_DynamicTimeWarping(self, node):  # noqa: N802
        # Input 0 has shape M x N or 1 x M x N
        # Output 0 has shape (2, O) where max(M, N) <= O < M + N
        input_shape = self._get_shape(node, 0)
        if input_shape is not None:
            shape_len = len(input_shape)
            assert shape_len == 2 or shape_len == 3
            M, N = input_shape[shape_len - 2], input_shape[shape_len - 1]  # noqa: N806
            output_shape = [2, f"max({M}, {N}) <= O < {M} + {N}"]
            output_dtype = onnx.TensorProto.FLOAT
            vi = self.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

    def _infer_FastGelu(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

    def _infer_Gelu(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

    def _infer_QuickGelu(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

    def _infer_GemmFastGelu(self, node):  # noqa: N802
        self._compute_matmul_shape(node)

    def _infer_GemmFloat8(self, node):  # noqa: N802
        self._compute_matmul_shape(node)

    def _infer_LayerNormalization(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)
        if len(node.output) > 1:
            axis = get_attribute(node, "axis")
            if axis is None:
                axis = -1
            x_shape = self._get_shape(node, 0)
            if x_shape is not None:
                rank = len(x_shape)
                axis = handle_negative_axis(axis, rank)
                mean_shape = x_shape[:axis] + [1 for _ in range(rank - axis)]
                mean_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
                if mean_dtype == onnx.TensorProto.FLOAT16 or mean_dtype == onnx.TensorProto.BFLOAT16:
                    mean_dtype = onnx.TensorProto.FLOAT
                vi = self.known_vi_[node.output[1]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[1], mean_dtype, mean_shape))
                if len(node.output) > 2:
                    vi = self.known_vi_[node.output[2]]
                    vi.CopyFrom(helper.make_tensor_value_info(node.output[2], mean_dtype, mean_shape))

    def _infer_LongformerAttention(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

    def _infer_EmbedLayerNormalization(self, node):  # noqa: N802
        input_ids_shape = self._get_shape(node, 0)
        word_embedding_shape = self._get_shape(node, 2)
        assert len(input_ids_shape) == 2 and len(word_embedding_shape) == 2
        output_shape = [*input_ids_shape, word_embedding_shape[1]]

        word_embedding_dtype = self.known_vi_[node.input[2]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], word_embedding_dtype, output_shape))

        if len(node.output) > 1 and node.output[1]:
            mask_index_shape = [input_ids_shape[0]]
            vi = self.known_vi_[node.output[1]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[1], onnx.TensorProto.INT32, mask_index_shape))

        if len(node.output) > 2:
            # Optional output of add before layer normalization is done
            # shape is same as the output
            vi = self.known_vi_[node.output[2]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[2], word_embedding_dtype, output_shape))

    def _infer_SkipLayerNormalization(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

        # If the SkipLayerNormalization node contains the optional
        # output for inference, infer the shape and type for it too
        if len(node.output) > 3:
            self._propagate_shape_and_type(node, 0, 3)

    def _infer_GroupNorm(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

    def _infer_PagedAttention(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

    def _infer_GroupQueryAttention(self, node):  # noqa: N802
        output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type

        past_shape = self._try_get_shape(node, 3)
        if past_shape is not None:
            # When past and present has the maximum sequence length, we can propagate the shape from past to present.
            # Note that GQA also supports different sequence lengths for past and present, but it is rarely used.
            vi = self.known_vi_[node.output[1]]
            vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))
            vi = self.known_vi_[node.output[2]]
            vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))

        if node.input[1] != "" and node.input[2] != "":
            self._propagate_shape_and_type(node, 0, 0)
        else:
            # combined qkv: (batch_size, sequence_length, num_heads * head_size + 2 * kv_num_heads * head_size)
            assert node.input[1] == "" and node.input[2] == ""
            num_heads = get_attribute(node, "num_heads")
            kv_num_heads = get_attribute(node, "kv_num_heads")
            query_shape = self._get_shape(node, 0)
            if query_shape is not None:
                hidden_size = query_shape[2]
                if isinstance(hidden_size, int):
                    head_size = int(hidden_size / (num_heads + 2 * kv_num_heads))
                    query_shape[2] = num_heads * head_size
                    vi = self.known_vi_[node.output[0]]
                    vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, query_shape))

    def _infer_SparseAttention(self, node):  # noqa: N802
        self._infer_GroupQueryAttention(node)

    def _infer_SkipGroupNorm(self, node):  # noqa: N802
        self._propagate_shape_and_type(node, 0, 0)
        if len(node.output) > 1:
            self._propagate_shape_and_type(node, 0, 1)

    def _infer_BiasSplitGelu(self, node):  # noqa: N802
        input_shape = self._get_shape(node, 0)
        bias_shape = self._get_shape(node, 1)
        if input_shape and bias_shape and isinstance(bias_shape[0], int):
            output_shape = input_shape
            output_shape[2] = int(bias_shape[0] / 2)
            vi = self.known_vi_[node.output[0]]
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, output_shape))

    def _infer_BiasAdd(self, node):  # noqa: N802
        self._propagate_shape_and_type(node)

    def _infer_RotaryEmbedding(self, node):  # noqa: N802
        if len(node.output) == 1:
            self._propagate_shape_and_type(node)
        elif len(node.output) == 2:
            # Extraneous constant nodes outputted by RotaryEmbedding function made with `export_modules_as_functions`
            self._propagate_shape_and_type(node, input_index=1, output_index=0)
            self._propagate_shape_and_type(node, input_index=0, output_index=1)  # true output
        elif len(node.output) == 3:
            # Extraneous constant nodes outputted by RotaryEmbedding function made with `export_modules_as_functions`
            self._propagate_shape_and_type(node, input_index=1, output_index=0)
            self._propagate_shape_and_type(node, input_index=1, output_index=1)
            self._propagate_shape_and_type(node, input_index=0, output_index=2)  # true output

    def _infer_PythonOp(self, node):  # noqa: N802
        output_tensor_types = get_attribute(node, "output_tensor_types")
        assert output_tensor_types, f"PythonOp '{node.name}' has no output_tensor_types attribute."
        output_tensor_ranks = get_attribute(node, "output_tensor_ranks")
        assert output_tensor_ranks, f"PythonOp '{node.name}' has no output_tensor_ranks attribute."

        from onnxruntime.capi._pybind_state import get_shape_inference_function

        func_name = get_attribute(node, "func_name").decode()
        shape_inferer = get_shape_inference_function(func_name)

        # Set the context output separately.
        # The first output is torch.autograd.Function''s context.
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, []))

        if shape_inferer is not None:
            input_shapes = []
            input_dtypes = []
            for input_index in range(len(node.input)):
                shape = self._get_shape(node, input_index)
                input_shapes.append(shape)
                input_dtype = self.known_vi_[node.input[input_index]].type.tensor_type.elem_type
                input_dtypes.append(input_dtype)
            output_shapes, output_dtypes = shape_inferer(node, input_shapes, input_dtypes)
            assert len(output_shapes) == len(output_dtypes) == (len(node.output) - 1), (
                f"PythonOp '{func_name}' returned {len(output_shapes)} shapes and {len(output_dtypes)} dtypes, "
                f"but expected {len(node.output) - 1} outputs."
            )
            for i in range(len(node.output) - 1):
                output_index = i + 1
                vi = self.known_vi_[node.output[output_index]]
                vi.CopyFrom(
                    helper.make_tensor_value_info(node.output[output_index], output_dtypes[i], output_shapes[i])
                )
        else:
            # General shape inference for PythonOp.
            # Outputs after torch.autograd.Function's context are tensors.
            # We assume their ranks are fixed for different model inputs.
            for i in range(len(node.output) - 1):
                # Process the i-th tensor outputs.
                vi = self.known_vi_[node.output[i + 1]]
                sympy_shape = self._new_symbolic_shape(output_tensor_ranks[i], node)
                shape = get_shape_from_sympy_shape(sympy_shape)
                value_info = helper.make_tensor_value_info(node.output[i + 1], output_tensor_types[i], shape)
                vi.CopyFrom(value_info)

    def _propagate_shape_and_type(self, node, input_index=0, output_index=0):
        shape = self._get_shape(node, input_index)
        output_dtype = self.known_vi_[node.input[input_index]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[output_index]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[output_index], output_dtype, shape))

    def _is_none_dim(self, dim_value):
        if type(dim_value) != str:  # noqa: E721
            return False
        if "unk__" not in dim_value:
            return False
        if dim_value in self.symbolic_dims_:
            return False
        return True

    def _is_shape_contains_none_dim(self, out_shape):
        for out in out_shape:
            if self._is_none_dim(out):
                return out
        return None

    def _infer_impl(self, start_sympy_data=None):
        self.sympy_data_ = start_sympy_data or {}
        self.out_mp_.graph.ClearField("value_info")
        self._apply_suggested_merge(graph_input_only=True)
        self.input_symbols_ = set()
        for i in self.out_mp_.graph.input:
            input_shape = get_shape_from_value_info(i)
            if input_shape is None:
                continue

            if is_sequence(i.type):
                input_dims = i.type.sequence_type.elem_type.tensor_type.shape.dim
            else:
                input_dims = i.type.tensor_type.shape.dim

            for i_dim, dim in enumerate(input_shape):
                if dim is None:
                    # some models use None for symbolic dim in input, replace it with a string
                    input_dims[i_dim].dim_param = str(self._new_symbolic_dim(i.name, i_dim))

            self.input_symbols_.update([d for d in input_shape if type(d) is str])

        for s in self.input_symbols_:
            if s in self.suggested_merge_:
                s_merge = self.suggested_merge_[s]
                assert s_merge in self.symbolic_dims_
                self.symbolic_dims_[s] = self.symbolic_dims_[s_merge]
            else:
                # Since inputs are not produced by other ops, we can assume positivity
                self.symbolic_dims_[s] = sympy.Symbol(s, integer=True, positive=True)
        # create a temporary ModelProto for single node inference
        # note that we remove initializer to have faster inference
        # for tensor ops like Reshape/Tile/Expand that read initializer, we need to do sympy computation based inference anyways
        self.tmp_mp_ = onnx.ModelProto()
        self.tmp_mp_.CopyFrom(self.out_mp_)
        self.tmp_mp_.graph.ClearField("initializer")

        # compute prerequesite for node for topological sort
        # node with subgraphs may have dependency on implicit inputs, which will affect topological sort
        prereq_for_node = {}  # map from node to all its inputs, including implicit ones in subgraph

        def get_prereq(node):
            names = {i for i in node.input if i}
            subgraphs = []
            if node.op_type == "If":
                subgraphs = [
                    get_attribute(node, "then_branch"),
                    get_attribute(node, "else_branch"),
                ]
            elif node.op_type in ["Loop", "Scan"]:
                subgraphs = [get_attribute(node, "body")]
            for g in subgraphs:
                g_outputs_and_initializers = {i.name for i in g.initializer}
                g_prereq = set()
                for n in g.node:
                    g_outputs_and_initializers.update(n.output)
                for n in g.node:
                    g_prereq.update([i for i in get_prereq(n) if i not in g_outputs_and_initializers])
                names.update(g_prereq)
                # remove subgraph inputs from g_prereq since those are local-only
                for i in g.input:
                    if i.name in names:
                        names.remove(i.name)
            return names

        for n in self.tmp_mp_.graph.node:
            prereq_for_node[n.output[0]] = get_prereq(n)

        # topological sort nodes, note there might be dead nodes so we check if all graph outputs are reached to terminate
        sorted_nodes = []
        sorted_known_vi = {i.name for i in list(self.out_mp_.graph.input) + list(self.out_mp_.graph.initializer)}
        if any(o.name in sorted_known_vi for o in self.out_mp_.graph.output):
            # Loop/Scan will have some graph output in graph inputs, so don't do topological sort
            sorted_nodes = self.out_mp_.graph.node
        else:
            while not all(o.name in sorted_known_vi for o in self.out_mp_.graph.output):
                old_sorted_nodes_len = len(sorted_nodes)
                for node in self.out_mp_.graph.node:
                    if (node.output[0] not in sorted_known_vi) and all(
                        i in sorted_known_vi for i in prereq_for_node[node.output[0]] if i
                    ):
                        sorted_known_vi.update(node.output)
                        sorted_nodes.append(node)
                if old_sorted_nodes_len == len(sorted_nodes) and not all(
                    o.name in sorted_known_vi for o in self.out_mp_.graph.output
                ):
                    raise Exception("Invalid model with cyclic graph")

        for node in sorted_nodes:
            assert all(i in self.known_vi_ for i in node.input if i)
            self._onnx_infer_single_node(node)
            known_aten_op = False
            if node.op_type in self.dispatcher_:
                self.dispatcher_[node.op_type](node)
            elif node.op_type in ["ConvTranspose"]:
                # onnx shape inference ops like ConvTranspose may have empty shape for symbolic input
                # before adding symbolic compute for them
                # mark the output type as UNDEFINED to allow guessing of rank
                vi = self.known_vi_[node.output[0]]
                if len(vi.type.tensor_type.shape.dim) == 0:
                    vi.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
            elif node.op_type == "ATen" and node.domain == "org.pytorch.aten":
                for attr in node.attribute:
                    # TODO: Is overload_name needed?
                    if attr.name == "operator":
                        aten_op_name = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                        if aten_op_name in self.aten_op_dispatcher_:
                            known_aten_op = True
                            self.aten_op_dispatcher_[aten_op_name](node)
                        break

            if self.verbose_ > 2:
                logger.debug(node.op_type + ": " + node.name)  # noqa: G003
                for i, name in enumerate(node.input):
                    logger.debug("  Input %s: %s %s", i, name, "initializer" if name in self.initializers_ else "")

            # onnx automatically merge dims with value, i.e. Mul(['aaa', 'bbb'], [1000, 1]) -> [1000, 'bbb']
            # symbolic shape inference needs to apply merge of 'aaa' -> 1000 in this case
            if node.op_type in [
                "Add",
                "Sub",
                "Mul",
                "Div",
                "MatMul",
                "MatMulInteger",
                "MatMulInteger16",
                "Where",
                "Sum",
            ]:
                vi = self.known_vi_[node.output[0]]
                out_rank = len(get_shape_from_type_proto(vi.type))
                in_shapes = [self._get_shape(node, i) for i in range(len(node.input))]
                for d in range(out_rank - (2 if node.op_type in ["MatMul", "MatMulInteger", "MatMulInteger16"] else 0)):
                    in_dims = [s[len(s) - out_rank + d] for s in in_shapes if len(s) + d >= out_rank]
                    if len(in_dims) > 1:
                        self._check_merged_dims(in_dims, allow_broadcast=True)

            for i_o in range(len(node.output)):
                # Special cases:
                # 1) We do not care about the training related outputs of SkipLayerNormalization
                # 2) We do not care about the extraneous constant outputs in RotaryEmbedding because
                # the RotaryEmbedding op created during export can be replaced by the RotaryEmbedding
                # contrib op
                if (
                    node.op_type == "SkipLayerNormalization" or node.op_type == "SkipSimplifiedLayerNormalization"
                ) and i_o in [1, 2]:
                    continue
                if node.op_type == "RotaryEmbedding" and len(node.output) > 1:
                    # Skip symbolic shape inference for RotaryEmbedding functions that have extraneous outputs
                    # generated by `export_modules_as_functions`
                    continue

                vi = self.known_vi_[node.output[i_o]]
                out_type = vi.type
                out_type_kind = out_type.WhichOneof("value")

                # do not process shape for non-tensors
                if out_type_kind not in ["tensor_type", "sparse_tensor_type", None]:
                    if self.verbose_ > 2:
                        if out_type_kind == "sequence_type":
                            seq_cls_type = out_type.sequence_type.elem_type.WhichOneof("value")
                            if seq_cls_type == "tensor_type":
                                logger.debug(
                                    "  {}: sequence of {} {}".format(  # noqa: G001
                                        node.output[i_o],
                                        str(get_shape_from_value_info(vi)),
                                        onnx.TensorProto.DataType.Name(
                                            vi.type.sequence_type.elem_type.tensor_type.elem_type
                                        ),
                                    )
                                )
                            else:
                                logger.debug(f"  {node.output[i_o]}: sequence of {seq_cls_type}")
                        else:
                            logger.debug(f"  {node.output[i_o]}: {out_type_kind}")
                    continue

                out_shape = get_shape_from_value_info(vi)
                out_type_undefined = out_type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED
                if self.verbose_ > 2:
                    logger.debug(
                        f"  {node.output[i_o]}: {out_shape!s} {onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)}"
                    )
                    if node.output[i_o] in self.sympy_data_:
                        logger.debug("  Sympy Data: " + str(self.sympy_data_[node.output[i_o]]))  # noqa: G003

                # onnx >= 1.11.0, use unk__#index instead of None when the shape dim is uncertain
                if (
                    out_shape is not None and (None in out_shape or self._is_shape_contains_none_dim(out_shape))
                ) or out_type_undefined:
                    if self.auto_merge_:
                        if node.op_type in [
                            "Add",
                            "Sub",
                            "Mul",
                            "Div",
                            "MatMul",
                            "MatMulInteger",
                            "MatMulInteger16",
                            "Concat",
                            "Where",
                            "Sum",
                            "Equal",
                            "Less",
                            "Greater",
                            "LessOrEqual",
                            "GreaterOrEqual",
                            "Min",
                            "Max",
                        ]:
                            shapes = [self._get_shape(node, i) for i in range(len(node.input))]
                            if node.op_type in [
                                "MatMul",
                                "MatMulInteger",
                                "MatMulInteger16",
                            ]:
                                if None in out_shape or self._is_shape_contains_none_dim(out_shape):
                                    if None in out_shape:
                                        idx = out_shape.index(None)
                                    else:
                                        idx = out_shape.index(self._is_shape_contains_none_dim(out_shape))
                                    dim_idx = [len(s) - len(out_shape) + idx for s in shapes]
                                    # only support auto merge for MatMul for dim < rank-2 when rank > 2
                                    assert len(shapes[0]) > 2 and dim_idx[0] < len(shapes[0]) - 2
                                    assert len(shapes[1]) > 2 and dim_idx[1] < len(shapes[1]) - 2
                        elif node.op_type == "Expand":
                            # auto merge for cases like Expand([min(batch, 1), min(seq, 512)], [batch, seq])
                            shapes = [
                                self._get_shape(node, 0),
                                self._get_value(node, 1),
                            ]
                        else:
                            shapes = []

                        if shapes:
                            for idx in range(len(out_shape)):
                                if out_shape[idx] is not None and not self._is_none_dim(out_shape[idx]):
                                    continue
                                # note that the broadcasting rule aligns from right to left
                                # if a tensor has a lower rank (dim_idx[idx] < 0), it would automatically broadcast and need no merge
                                dim_idx = [len(s) - len(out_shape) + idx for s in shapes]
                                if len(dim_idx) > 0:
                                    self._add_suggested_merge(
                                        [
                                            s[i] if is_literal(s[i]) else str(s[i])
                                            for s, i in zip(shapes, dim_idx, strict=False)
                                            if i >= 0
                                        ]
                                    )
                            self.run_ = True
                        else:
                            self.run_ = False
                    else:
                        self.run_ = False

                    # create new dynamic dims for ops not handled by symbolic shape inference
                    if self.run_ is False and node.op_type not in self.dispatcher_ and not known_aten_op:
                        is_unknown_op = out_type_undefined and (out_shape is None or len(out_shape) == 0)
                        if is_unknown_op:
                            # unknown op to ONNX, maybe from higher opset or other domain
                            # only guess the output rank from input 0 when using guess_output_rank option
                            out_rank = self._get_shape_rank(node, 0) if self.guess_output_rank_ else -1
                        else:
                            # valid ONNX op, but not handled by symbolic shape inference, just assign dynamic shape
                            out_rank = len(out_shape)

                        if out_rank >= 0:
                            new_shape = self._new_symbolic_shape(out_rank, node, i_o)
                            if out_type_undefined:
                                # guess output data type from input vi if not defined
                                out_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
                            else:
                                # otherwise, use original data type
                                out_dtype = vi.type.tensor_type.elem_type
                            vi.CopyFrom(
                                helper.make_tensor_value_info(
                                    vi.name,
                                    out_dtype,
                                    get_shape_from_sympy_shape(new_shape),
                                )
                            )

                            if self.verbose_ > 0:
                                if is_unknown_op:
                                    logger.debug(
                                        f"Possible unknown op: {node.op_type} node: {node.name}, guessing {vi.name} shape"
                                    )
                                if self.verbose_ > 2:
                                    logger.debug(f"  {node.output[i_o]}: {new_shape!s} {vi.type.tensor_type.elem_type}")

                            self.run_ = True
                            continue  # continue the inference after guess, no need to stop as no merge is needed

                    if self.verbose_ > 0 or not self.auto_merge_ or out_type_undefined:
                        logger.debug("Stopping at incomplete shape inference at %s: %s", node.op_type, node.name)
                        logger.debug("node inputs:")
                        for i in node.input:
                            if i in self.known_vi_:
                                logger.debug(self.known_vi_[i])
                            else:
                                logger.debug(f"not in known_vi_ for {i}")
                        logger.debug("node outputs:")
                        for o in node.output:
                            if o in self.known_vi_:
                                logger.debug(self.known_vi_[o])
                            else:
                                logger.debug(f"not in known_vi_ for {o}")
                        if self.auto_merge_ and not out_type_undefined:
                            logger.debug("Merging: " + str(self.suggested_merge_))  # noqa: G003
                    return False

        self.run_ = False
        return True

    def _update_output_from_vi(self):
        for output in self.out_mp_.graph.output:
            if output.name in self.known_vi_:
                output.CopyFrom(self.known_vi_[output.name])

    @staticmethod
    def infer_shapes(in_mp, int_max=2**31 - 1, auto_merge=False, guess_output_rank=False, verbose=0):
        onnx_opset = get_opset(in_mp)
        if (not onnx_opset) or onnx_opset < 7:
            logger.warning("Only support models of onnx opset 7 and above.")
            return None
        symbolic_shape_inference = SymbolicShapeInference(int_max, auto_merge, guess_output_rank, verbose)
        all_shapes_inferred = False
        symbolic_shape_inference._preprocess(in_mp)
        while symbolic_shape_inference.run_:
            all_shapes_inferred = symbolic_shape_inference._infer_impl()
        symbolic_shape_inference._update_output_from_vi()
        if not all_shapes_inferred:
            onnx.save_model(symbolic_shape_inference.out_mp_, "sym_shape_infer_temp.onnx", save_as_external_data=True)
            raise Exception("Incomplete symbolic shape inference")
        return symbolic_shape_inference.out_mp_


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="The input model file")
    parser.add_argument("--output", help="The output model file")
    parser.add_argument(
        "--auto_merge",
        help="Automatically merge symbolic dims when confliction happens",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--int_max",
        help="maximum value for integer to be treated as boundless for ops like slice",
        type=int,
        default=2**31 - 1,
    )
    parser.add_argument(
        "--guess_output_rank",
        help="guess output rank to be the same as input 0 for unknown ops",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="Prints detailed logs of inference, 0: turn off, 1: warnings, 3: detailed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_as_external_data",
        help="Saving an ONNX model to external data",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--all_tensors_to_one_file",
        help="Saving all the external data to one file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--external_data_location",
        help="The file location to save the external file",
        default="./",
    )
    parser.add_argument(
        "--external_data_size_threshold",
        help="The size threshold for external data",
        type=int,
        default=1024,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logger.info("input model: " + args.input)  # noqa: G003
    if args.output:
        logger.info("output model " + args.output)  # noqa: G003
    logger.info("Doing symbolic shape inference...")
    out_mp = SymbolicShapeInference.infer_shapes(
        onnx.load(args.input),
        args.int_max,
        args.auto_merge,
        args.guess_output_rank,
        args.verbose,
    )
    if args.output and out_mp:
        if args.save_as_external_data:
            onnx.save_model(
                out_mp,
                args.output,
                save_as_external_data=True,
                all_tensors_to_one_file=args.all_tensors_to_one_file,
                location=args.external_data_location,
                size_threshold=args.external_data_size_threshold,
                convert_attribute=False,
            )
        else:
            onnx.save(out_mp, args.output)
        logger.info("Done!")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\introspect.py ===
"""
Introspection helper functions.
"""

__all__ = ['opt_func_info']


def opt_func_info(func_name=None, signature=None):
    """
    Returns a dictionary containing the currently supported CPU dispatched
    features for all optimized functions.

    Parameters
    ----------
    func_name : str (optional)
        Regular expression to filter by function name.

    signature : str (optional)
        Regular expression to filter by data type.

    Returns
    -------
    dict
        A dictionary where keys are optimized function names and values are
        nested dictionaries indicating supported targets based on data types.

    Examples
    --------
    Retrieve dispatch information for functions named 'add' or 'sub' and
    data types 'float64' or 'float32':

    >>> import numpy as np
    >>> dict = np.lib.introspect.opt_func_info(
    ...     func_name="add|abs", signature="float64|complex64"
    ... )
    >>> import json
    >>> print(json.dumps(dict, indent=2))
        {
          "absolute": {
            "dd": {
              "current": "SSE41",
              "available": "SSE41 baseline(SSE SSE2 SSE3)"
            },
            "Ff": {
              "current": "FMA3__AVX2",
              "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            },
            "Dd": {
              "current": "FMA3__AVX2",
              "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            }
          },
          "add": {
            "ddd": {
              "current": "FMA3__AVX2",
              "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            },
            "FFF": {
              "current": "FMA3__AVX2",
              "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            }
          }
        }

    """
    import re

    from numpy._core._multiarray_umath import __cpu_targets_info__ as targets
    from numpy._core._multiarray_umath import dtype

    if func_name is not None:
        func_pattern = re.compile(func_name)
        matching_funcs = {
            k: v for k, v in targets.items()
            if func_pattern.search(k)
        }
    else:
        matching_funcs = targets

    if signature is not None:
        sig_pattern = re.compile(signature)
        matching_sigs = {}
        for k, v in matching_funcs.items():
            matching_chars = {}
            for chars, targets in v.items():
                if any(
                    sig_pattern.search(c) or sig_pattern.search(dtype(c).name)
                    for c in chars
                ):
                    matching_chars[chars] = targets
            if matching_chars:
                matching_sigs[k] = matching_chars
    else:
        matching_sigs = matching_funcs
    return matching_sigs

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\_antlr\autolevparser.py ===
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,49,431,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,1,0,4,0,58,8,0,11,0,12,0,59,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,3,1,69,8,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
        3,2,84,8,2,1,2,1,2,1,2,3,2,89,8,2,1,3,1,3,1,4,1,4,1,4,5,4,96,8,4,
        10,4,12,4,99,9,4,1,5,4,5,102,8,5,11,5,12,5,103,1,6,1,6,1,6,1,6,1,
        6,5,6,111,8,6,10,6,12,6,114,9,6,3,6,116,8,6,1,6,1,6,1,6,1,6,1,6,
        1,6,5,6,124,8,6,10,6,12,6,127,9,6,3,6,129,8,6,1,6,3,6,132,8,6,1,
        7,1,7,1,7,1,7,5,7,138,8,7,10,7,12,7,141,9,7,1,8,1,8,1,8,1,8,1,8,
        1,8,1,8,1,8,1,8,1,8,5,8,153,8,8,10,8,12,8,156,9,8,1,8,1,8,5,8,160,
        8,8,10,8,12,8,163,9,8,3,8,165,8,8,1,9,1,9,1,9,1,9,1,9,1,9,3,9,173,
        8,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,5,9,183,8,9,10,9,12,9,186,9,
        9,1,9,3,9,189,8,9,1,9,1,9,1,9,3,9,194,8,9,1,9,3,9,197,8,9,1,9,5,
        9,200,8,9,10,9,12,9,203,9,9,1,9,1,9,3,9,207,8,9,1,10,1,10,1,10,1,
        10,1,10,1,10,1,10,1,10,5,10,217,8,10,10,10,12,10,220,9,10,1,10,1,
        10,1,11,1,11,1,11,1,11,5,11,228,8,11,10,11,12,11,231,9,11,1,12,1,
        12,1,12,1,12,1,13,1,13,1,13,1,13,1,13,3,13,242,8,13,1,13,1,13,4,
        13,246,8,13,11,13,12,13,247,1,14,1,14,1,14,1,14,5,14,254,8,14,10,
        14,12,14,257,9,14,1,14,1,14,1,15,1,15,1,15,1,15,3,15,265,8,15,1,
        15,1,15,3,15,269,8,15,1,16,1,16,1,16,1,16,1,16,3,16,276,8,16,1,17,
        1,17,3,17,280,8,17,1,18,1,18,1,18,1,18,5,18,286,8,18,10,18,12,18,
        289,9,18,1,19,1,19,1,19,1,19,5,19,295,8,19,10,19,12,19,298,9,19,
        1,20,1,20,3,20,302,8,20,1,21,1,21,1,21,1,21,3,21,308,8,21,1,22,1,
        22,1,22,1,22,5,22,314,8,22,10,22,12,22,317,9,22,1,23,1,23,3,23,321,
        8,23,1,24,1,24,1,24,1,24,1,24,1,24,5,24,329,8,24,10,24,12,24,332,
        9,24,1,24,1,24,3,24,336,8,24,1,24,1,24,1,24,1,24,1,25,1,25,1,25,
        1,25,1,25,1,25,1,25,1,25,5,25,350,8,25,10,25,12,25,353,9,25,3,25,
        355,8,25,1,26,1,26,4,26,359,8,26,11,26,12,26,360,1,26,1,26,3,26,
        365,8,26,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,5,27,375,8,27,10,
        27,12,27,378,9,27,1,27,1,27,1,27,1,27,1,27,1,27,5,27,386,8,27,10,
        27,12,27,389,9,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,3,
        27,400,8,27,1,27,1,27,5,27,404,8,27,10,27,12,27,407,9,27,3,27,409,
        8,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,1,27,
        1,27,1,27,1,27,5,27,426,8,27,10,27,12,27,429,9,27,1,27,0,1,54,28,
        0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,
        46,48,50,52,54,0,7,1,0,3,9,1,0,27,28,1,0,17,18,2,0,10,10,19,19,1,
        0,44,45,2,0,44,46,48,48,1,0,25,26,483,0,57,1,0,0,0,2,68,1,0,0,0,
        4,88,1,0,0,0,6,90,1,0,0,0,8,92,1,0,0,0,10,101,1,0,0,0,12,131,1,0,
        0,0,14,133,1,0,0,0,16,164,1,0,0,0,18,166,1,0,0,0,20,208,1,0,0,0,
        22,223,1,0,0,0,24,232,1,0,0,0,26,236,1,0,0,0,28,249,1,0,0,0,30,268,
        1,0,0,0,32,275,1,0,0,0,34,277,1,0,0,0,36,281,1,0,0,0,38,290,1,0,
        0,0,40,299,1,0,0,0,42,303,1,0,0,0,44,309,1,0,0,0,46,318,1,0,0,0,
        48,322,1,0,0,0,50,354,1,0,0,0,52,364,1,0,0,0,54,408,1,0,0,0,56,58,
        3,2,1,0,57,56,1,0,0,0,58,59,1,0,0,0,59,57,1,0,0,0,59,60,1,0,0,0,
        60,1,1,0,0,0,61,69,3,14,7,0,62,69,3,12,6,0,63,69,3,32,16,0,64,69,
        3,22,11,0,65,69,3,26,13,0,66,69,3,4,2,0,67,69,3,34,17,0,68,61,1,
        0,0,0,68,62,1,0,0,0,68,63,1,0,0,0,68,64,1,0,0,0,68,65,1,0,0,0,68,
        66,1,0,0,0,68,67,1,0,0,0,69,3,1,0,0,0,70,71,3,52,26,0,71,72,3,6,
        3,0,72,73,3,54,27,0,73,89,1,0,0,0,74,75,5,48,0,0,75,76,5,1,0,0,76,
        77,3,8,4,0,77,78,5,2,0,0,78,79,3,6,3,0,79,80,3,54,27,0,80,89,1,0,
        0,0,81,83,5,48,0,0,82,84,3,10,5,0,83,82,1,0,0,0,83,84,1,0,0,0,84,
        85,1,0,0,0,85,86,3,6,3,0,86,87,3,54,27,0,87,89,1,0,0,0,88,70,1,0,
        0,0,88,74,1,0,0,0,88,81,1,0,0,0,89,5,1,0,0,0,90,91,7,0,0,0,91,7,
        1,0,0,0,92,97,3,54,27,0,93,94,5,10,0,0,94,96,3,54,27,0,95,93,1,0,
        0,0,96,99,1,0,0,0,97,95,1,0,0,0,97,98,1,0,0,0,98,9,1,0,0,0,99,97,
        1,0,0,0,100,102,5,11,0,0,101,100,1,0,0,0,102,103,1,0,0,0,103,101,
        1,0,0,0,103,104,1,0,0,0,104,11,1,0,0,0,105,106,5,48,0,0,106,115,
        5,12,0,0,107,112,3,54,27,0,108,109,5,10,0,0,109,111,3,54,27,0,110,
        108,1,0,0,0,111,114,1,0,0,0,112,110,1,0,0,0,112,113,1,0,0,0,113,
        116,1,0,0,0,114,112,1,0,0,0,115,107,1,0,0,0,115,116,1,0,0,0,116,
        117,1,0,0,0,117,132,5,13,0,0,118,119,7,1,0,0,119,128,5,12,0,0,120,
        125,5,48,0,0,121,122,5,10,0,0,122,124,5,48,0,0,123,121,1,0,0,0,124,
        127,1,0,0,0,125,123,1,0,0,0,125,126,1,0,0,0,126,129,1,0,0,0,127,
        125,1,0,0,0,128,120,1,0,0,0,128,129,1,0,0,0,129,130,1,0,0,0,130,
        132,5,13,0,0,131,105,1,0,0,0,131,118,1,0,0,0,132,13,1,0,0,0,133,
        134,3,16,8,0,134,139,3,18,9,0,135,136,5,10,0,0,136,138,3,18,9,0,
        137,135,1,0,0,0,138,141,1,0,0,0,139,137,1,0,0,0,139,140,1,0,0,0,
        140,15,1,0,0,0,141,139,1,0,0,0,142,165,5,34,0,0,143,165,5,35,0,0,
        144,165,5,36,0,0,145,165,5,37,0,0,146,165,5,38,0,0,147,165,5,39,
        0,0,148,165,5,40,0,0,149,165,5,41,0,0,150,154,5,42,0,0,151,153,5,
        11,0,0,152,151,1,0,0,0,153,156,1,0,0,0,154,152,1,0,0,0,154,155,1,
        0,0,0,155,165,1,0,0,0,156,154,1,0,0,0,157,161,5,43,0,0,158,160,5,
        11,0,0,159,158,1,0,0,0,160,163,1,0,0,0,161,159,1,0,0,0,161,162,1,
        0,0,0,162,165,1,0,0,0,163,161,1,0,0,0,164,142,1,0,0,0,164,143,1,
        0,0,0,164,144,1,0,0,0,164,145,1,0,0,0,164,146,1,0,0,0,164,147,1,
        0,0,0,164,148,1,0,0,0,164,149,1,0,0,0,164,150,1,0,0,0,164,157,1,
        0,0,0,165,17,1,0,0,0,166,172,5,48,0,0,167,168,5,14,0,0,168,169,5,
        44,0,0,169,170,5,10,0,0,170,171,5,44,0,0,171,173,5,15,0,0,172,167,
        1,0,0,0,172,173,1,0,0,0,173,188,1,0,0,0,174,175,5,14,0,0,175,176,
        5,44,0,0,176,177,5,16,0,0,177,184,5,44,0,0,178,179,5,10,0,0,179,
        180,5,44,0,0,180,181,5,16,0,0,181,183,5,44,0,0,182,178,1,0,0,0,183,
        186,1,0,0,0,184,182,1,0,0,0,184,185,1,0,0,0,185,187,1,0,0,0,186,
        184,1,0,0,0,187,189,5,15,0,0,188,174,1,0,0,0,188,189,1,0,0,0,189,
        193,1,0,0,0,190,191,5,14,0,0,191,192,5,44,0,0,192,194,5,15,0,0,193,
        190,1,0,0,0,193,194,1,0,0,0,194,196,1,0,0,0,195,197,7,2,0,0,196,
        195,1,0,0,0,196,197,1,0,0,0,197,201,1,0,0,0,198,200,5,11,0,0,199,
        198,1,0,0,0,200,203,1,0,0,0,201,199,1,0,0,0,201,202,1,0,0,0,202,
        206,1,0,0,0,203,201,1,0,0,0,204,205,5,3,0,0,205,207,3,54,27,0,206,
        204,1,0,0,0,206,207,1,0,0,0,207,19,1,0,0,0,208,209,5,14,0,0,209,
        210,5,44,0,0,210,211,5,16,0,0,211,218,5,44,0,0,212,213,5,10,0,0,
        213,214,5,44,0,0,214,215,5,16,0,0,215,217,5,44,0,0,216,212,1,0,0,
        0,217,220,1,0,0,0,218,216,1,0,0,0,218,219,1,0,0,0,219,221,1,0,0,
        0,220,218,1,0,0,0,221,222,5,15,0,0,222,21,1,0,0,0,223,224,5,27,0,
        0,224,229,3,24,12,0,225,226,5,10,0,0,226,228,3,24,12,0,227,225,1,
        0,0,0,228,231,1,0,0,0,229,227,1,0,0,0,229,230,1,0,0,0,230,23,1,0,
        0,0,231,229,1,0,0,0,232,233,5,48,0,0,233,234,5,3,0,0,234,235,3,54,
        27,0,235,25,1,0,0,0,236,237,5,28,0,0,237,241,5,48,0,0,238,239,5,
        12,0,0,239,240,5,48,0,0,240,242,5,13,0,0,241,238,1,0,0,0,241,242,
        1,0,0,0,242,245,1,0,0,0,243,244,5,10,0,0,244,246,3,54,27,0,245,243,
        1,0,0,0,246,247,1,0,0,0,247,245,1,0,0,0,247,248,1,0,0,0,248,27,1,
        0,0,0,249,250,5,1,0,0,250,255,3,54,27,0,251,252,7,3,0,0,252,254,
        3,54,27,0,253,251,1,0,0,0,254,257,1,0,0,0,255,253,1,0,0,0,255,256,
        1,0,0,0,256,258,1,0,0,0,257,255,1,0,0,0,258,259,5,2,0,0,259,29,1,
        0,0,0,260,261,5,48,0,0,261,262,5,48,0,0,262,264,5,3,0,0,263,265,
        7,4,0,0,264,263,1,0,0,0,264,265,1,0,0,0,265,269,1,0,0,0,266,269,
        5,45,0,0,267,269,5,44,0,0,268,260,1,0,0,0,268,266,1,0,0,0,268,267,
        1,0,0,0,269,31,1,0,0,0,270,276,3,36,18,0,271,276,3,38,19,0,272,276,
        3,44,22,0,273,276,3,48,24,0,274,276,3,50,25,0,275,270,1,0,0,0,275,
        271,1,0,0,0,275,272,1,0,0,0,275,273,1,0,0,0,275,274,1,0,0,0,276,
        33,1,0,0,0,277,279,5,48,0,0,278,280,7,5,0,0,279,278,1,0,0,0,279,
        280,1,0,0,0,280,35,1,0,0,0,281,282,5,32,0,0,282,287,5,48,0,0,283,
        284,5,10,0,0,284,286,5,48,0,0,285,283,1,0,0,0,286,289,1,0,0,0,287,
        285,1,0,0,0,287,288,1,0,0,0,288,37,1,0,0,0,289,287,1,0,0,0,290,291,
        5,29,0,0,291,296,3,42,21,0,292,293,5,10,0,0,293,295,3,42,21,0,294,
        292,1,0,0,0,295,298,1,0,0,0,296,294,1,0,0,0,296,297,1,0,0,0,297,
        39,1,0,0,0,298,296,1,0,0,0,299,301,5,48,0,0,300,302,3,10,5,0,301,
        300,1,0,0,0,301,302,1,0,0,0,302,41,1,0,0,0,303,304,3,40,20,0,304,
        305,5,3,0,0,305,307,3,54,27,0,306,308,3,54,27,0,307,306,1,0,0,0,
        307,308,1,0,0,0,308,43,1,0,0,0,309,310,5,30,0,0,310,315,3,46,23,
        0,311,312,5,10,0,0,312,314,3,46,23,0,313,311,1,0,0,0,314,317,1,0,
        0,0,315,313,1,0,0,0,315,316,1,0,0,0,316,45,1,0,0,0,317,315,1,0,0,
        0,318,320,3,54,27,0,319,321,3,54,27,0,320,319,1,0,0,0,320,321,1,
        0,0,0,321,47,1,0,0,0,322,323,5,48,0,0,323,335,3,12,6,0,324,325,5,
        1,0,0,325,330,3,30,15,0,326,327,5,10,0,0,327,329,3,30,15,0,328,326,
        1,0,0,0,329,332,1,0,0,0,330,328,1,0,0,0,330,331,1,0,0,0,331,333,
        1,0,0,0,332,330,1,0,0,0,333,334,5,2,0,0,334,336,1,0,0,0,335,324,
        1,0,0,0,335,336,1,0,0,0,336,337,1,0,0,0,337,338,5,48,0,0,338,339,
        5,20,0,0,339,340,5,48,0,0,340,49,1,0,0,0,341,342,5,31,0,0,342,343,
        5,48,0,0,343,344,5,20,0,0,344,355,5,48,0,0,345,346,5,33,0,0,346,
        351,5,48,0,0,347,348,5,10,0,0,348,350,5,48,0,0,349,347,1,0,0,0,350,
        353,1,0,0,0,351,349,1,0,0,0,351,352,1,0,0,0,352,355,1,0,0,0,353,
        351,1,0,0,0,354,341,1,0,0,0,354,345,1,0,0,0,355,51,1,0,0,0,356,358,
        5,48,0,0,357,359,5,21,0,0,358,357,1,0,0,0,359,360,1,0,0,0,360,358,
        1,0,0,0,360,361,1,0,0,0,361,365,1,0,0,0,362,365,5,22,0,0,363,365,
        5,23,0,0,364,356,1,0,0,0,364,362,1,0,0,0,364,363,1,0,0,0,365,53,
        1,0,0,0,366,367,6,27,-1,0,367,409,5,46,0,0,368,369,5,18,0,0,369,
        409,3,54,27,12,370,409,5,45,0,0,371,409,5,44,0,0,372,376,5,48,0,
        0,373,375,5,11,0,0,374,373,1,0,0,0,375,378,1,0,0,0,376,374,1,0,0,
        0,376,377,1,0,0,0,377,409,1,0,0,0,378,376,1,0,0,0,379,409,3,52,26,
        0,380,381,5,48,0,0,381,382,5,1,0,0,382,387,3,54,27,0,383,384,5,10,
        0,0,384,386,3,54,27,0,385,383,1,0,0,0,386,389,1,0,0,0,387,385,1,
        0,0,0,387,388,1,0,0,0,388,390,1,0,0,0,389,387,1,0,0,0,390,391,5,
        2,0,0,391,409,1,0,0,0,392,409,3,12,6,0,393,409,3,28,14,0,394,395,
        5,12,0,0,395,396,3,54,27,0,396,397,5,13,0,0,397,409,1,0,0,0,398,
        400,5,48,0,0,399,398,1,0,0,0,399,400,1,0,0,0,400,401,1,0,0,0,401,
        405,3,20,10,0,402,404,5,11,0,0,403,402,1,0,0,0,404,407,1,0,0,0,405,
        403,1,0,0,0,405,406,1,0,0,0,406,409,1,0,0,0,407,405,1,0,0,0,408,
        366,1,0,0,0,408,368,1,0,0,0,408,370,1,0,0,0,408,371,1,0,0,0,408,
        372,1,0,0,0,408,379,1,0,0,0,408,380,1,0,0,0,408,392,1,0,0,0,408,
        393,1,0,0,0,408,394,1,0,0,0,408,399,1,0,0,0,409,427,1,0,0,0,410,
        411,10,16,0,0,411,412,5,24,0,0,412,426,3,54,27,17,413,414,10,15,
        0,0,414,415,7,6,0,0,415,426,3,54,27,16,416,417,10,14,0,0,417,418,
        7,2,0,0,418,426,3,54,27,15,419,420,10,3,0,0,420,421,5,3,0,0,421,
        426,3,54,27,4,422,423,10,2,0,0,423,424,5,16,0,0,424,426,3,54,27,
        3,425,410,1,0,0,0,425,413,1,0,0,0,425,416,1,0,0,0,425,419,1,0,0,
        0,425,422,1,0,0,0,426,429,1,0,0,0,427,425,1,0,0,0,427,428,1,0,0,
        0,428,55,1,0,0,0,429,427,1,0,0,0,50,59,68,83,88,97,103,112,115,125,
        128,131,139,154,161,164,172,184,188,193,196,201,206,218,229,241,
        247,255,264,268,275,279,287,296,301,307,315,320,330,335,351,354,
        360,364,376,387,399,405,408,425,427
    ]

class AutolevParser ( Parser ):

    grammarFileName = "Autolev.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'['", "']'", "'='", "'+='", "'-='", "':='",
                     "'*='", "'/='", "'^='", "','", "'''", "'('", "')'",
                     "'{'", "'}'", "':'", "'+'", "'-'", "';'", "'.'", "'>'",
                     "'0>'", "'1>>'", "'^'", "'*'", "'/'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                      "<INVALID>", "<INVALID>", "<INVALID>", "Mass", "Inertia",
                      "Input", "Output", "Save", "UnitSystem", "Encode",
                      "Newtonian", "Frames", "Bodies", "Particles", "Points",
                      "Constants", "Specifieds", "Imaginary", "Variables",
                      "MotionVariables", "INT", "FLOAT", "EXP", "LINE_COMMENT",
                      "ID", "WS" ]

    RULE_prog = 0
    RULE_stat = 1
    RULE_assignment = 2
    RULE_equals = 3
    RULE_index = 4
    RULE_diff = 5
    RULE_functionCall = 6
    RULE_varDecl = 7
    RULE_varType = 8
    RULE_varDecl2 = 9
    RULE_ranges = 10
    RULE_massDecl = 11
    RULE_massDecl2 = 12
    RULE_inertiaDecl = 13
    RULE_matrix = 14
    RULE_matrixInOutput = 15
    RULE_codeCommands = 16
    RULE_settings = 17
    RULE_units = 18
    RULE_inputs = 19
    RULE_id_diff = 20
    RULE_inputs2 = 21
    RULE_outputs = 22
    RULE_outputs2 = 23
    RULE_codegen = 24
    RULE_commands = 25
    RULE_vec = 26
    RULE_expr = 27

    ruleNames =  [ "prog", "stat", "assignment", "equals", "index", "diff",
                   "functionCall", "varDecl", "varType", "varDecl2", "ranges",
                   "massDecl", "massDecl2", "inertiaDecl", "matrix", "matrixInOutput",
                   "codeCommands", "settings", "units", "inputs", "id_diff",
                   "inputs2", "outputs", "outputs2", "codegen", "commands",
                   "vec", "expr" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    Mass=27
    Inertia=28
    Input=29
    Output=30
    Save=31
    UnitSystem=32
    Encode=33
    Newtonian=34
    Frames=35
    Bodies=36
    Particles=37
    Points=38
    Constants=39
    Specifieds=40
    Imaginary=41
    Variables=42
    MotionVariables=43
    INT=44
    FLOAT=45
    EXP=46
    LINE_COMMENT=47
    ID=48
    WS=49

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.11.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def stat(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.StatContext)
            else:
                return self.getTypedRuleContext(AutolevParser.StatContext,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_prog

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProg" ):
                listener.enterProg(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProg" ):
                listener.exitProg(self)




    def prog(self):

        localctx = AutolevParser.ProgContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_prog)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 57
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 56
                self.stat()
                self.state = 59
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (((_la) & ~0x3f) == 0 and ((1 << _la) & 299067041120256) != 0):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StatContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def varDecl(self):
            return self.getTypedRuleContext(AutolevParser.VarDeclContext,0)


        def functionCall(self):
            return self.getTypedRuleContext(AutolevParser.FunctionCallContext,0)


        def codeCommands(self):
            return self.getTypedRuleContext(AutolevParser.CodeCommandsContext,0)


        def massDecl(self):
            return self.getTypedRuleContext(AutolevParser.MassDeclContext,0)


        def inertiaDecl(self):
            return self.getTypedRuleContext(AutolevParser.InertiaDeclContext,0)


        def assignment(self):
            return self.getTypedRuleContext(AutolevParser.AssignmentContext,0)


        def settings(self):
            return self.getTypedRuleContext(AutolevParser.SettingsContext,0)


        def getRuleIndex(self):
            return AutolevParser.RULE_stat

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStat" ):
                listener.enterStat(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStat" ):
                listener.exitStat(self)




    def stat(self):

        localctx = AutolevParser.StatContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_stat)
        try:
            self.state = 68
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 61
                self.varDecl()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 62
                self.functionCall()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 63
                self.codeCommands()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 64
                self.massDecl()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 65
                self.inertiaDecl()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 66
                self.assignment()
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 67
                self.settings()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AssignmentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return AutolevParser.RULE_assignment


        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class VecAssignContext(AssignmentContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.AssignmentContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def vec(self):
            return self.getTypedRuleContext(AutolevParser.VecContext,0)

        def equals(self):
            return self.getTypedRuleContext(AutolevParser.EqualsContext,0)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVecAssign" ):
                listener.enterVecAssign(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVecAssign" ):
                listener.exitVecAssign(self)


    class RegularAssignContext(AssignmentContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.AssignmentContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)
        def equals(self):
            return self.getTypedRuleContext(AutolevParser.EqualsContext,0)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext,0)

        def diff(self):
            return self.getTypedRuleContext(AutolevParser.DiffContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRegularAssign" ):
                listener.enterRegularAssign(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRegularAssign" ):
                listener.exitRegularAssign(self)


    class IndexAssignContext(AssignmentContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.AssignmentContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)
        def index(self):
            return self.getTypedRuleContext(AutolevParser.IndexContext,0)

        def equals(self):
            return self.getTypedRuleContext(AutolevParser.EqualsContext,0)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIndexAssign" ):
                listener.enterIndexAssign(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIndexAssign" ):
                listener.exitIndexAssign(self)



    def assignment(self):

        localctx = AutolevParser.AssignmentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_assignment)
        self._la = 0 # Token type
        try:
            self.state = 88
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                localctx = AutolevParser.VecAssignContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 70
                self.vec()
                self.state = 71
                self.equals()
                self.state = 72
                self.expr(0)
                pass

            elif la_ == 2:
                localctx = AutolevParser.IndexAssignContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 74
                self.match(AutolevParser.ID)
                self.state = 75
                self.match(AutolevParser.T__0)
                self.state = 76
                self.index()
                self.state = 77
                self.match(AutolevParser.T__1)
                self.state = 78
                self.equals()
                self.state = 79
                self.expr(0)
                pass

            elif la_ == 3:
                localctx = AutolevParser.RegularAssignContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 81
                self.match(AutolevParser.ID)
                self.state = 83
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==11:
                    self.state = 82
                    self.diff()


                self.state = 85
                self.equals()
                self.state = 86
                self.expr(0)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EqualsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return AutolevParser.RULE_equals

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEquals" ):
                listener.enterEquals(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEquals" ):
                listener.exitEquals(self)




    def equals(self):

        localctx = AutolevParser.EqualsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_equals)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 90
            _la = self._input.LA(1)
            if not(((_la) & ~0x3f) == 0 and ((1 << _la) & 1016) != 0):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IndexContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_index

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIndex" ):
                listener.enterIndex(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIndex" ):
                listener.exitIndex(self)




    def index(self):

        localctx = AutolevParser.IndexContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_index)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 92
            self.expr(0)
            self.state = 97
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==10:
                self.state = 93
                self.match(AutolevParser.T__9)
                self.state = 94
                self.expr(0)
                self.state = 99
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DiffContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return AutolevParser.RULE_diff

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDiff" ):
                listener.enterDiff(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDiff" ):
                listener.exitDiff(self)




    def diff(self):

        localctx = AutolevParser.DiffContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_diff)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 101
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 100
                self.match(AutolevParser.T__10)
                self.state = 103
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==11):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FunctionCallContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def Mass(self):
            return self.getToken(AutolevParser.Mass, 0)

        def Inertia(self):
            return self.getToken(AutolevParser.Inertia, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_functionCall

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFunctionCall" ):
                listener.enterFunctionCall(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFunctionCall" ):
                listener.exitFunctionCall(self)




    def functionCall(self):

        localctx = AutolevParser.FunctionCallContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_functionCall)
        self._la = 0 # Token type
        try:
            self.state = 131
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [48]:
                self.enterOuterAlt(localctx, 1)
                self.state = 105
                self.match(AutolevParser.ID)
                self.state = 106
                self.match(AutolevParser.T__11)
                self.state = 115
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if ((_la) & ~0x3f) == 0 and ((1 << _la) & 404620694540290) != 0:
                    self.state = 107
                    self.expr(0)
                    self.state = 112
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la==10:
                        self.state = 108
                        self.match(AutolevParser.T__9)
                        self.state = 109
                        self.expr(0)
                        self.state = 114
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)



                self.state = 117
                self.match(AutolevParser.T__12)
                pass
            elif token in [27, 28]:
                self.enterOuterAlt(localctx, 2)
                self.state = 118
                _la = self._input.LA(1)
                if not(_la==27 or _la==28):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 119
                self.match(AutolevParser.T__11)
                self.state = 128
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==48:
                    self.state = 120
                    self.match(AutolevParser.ID)
                    self.state = 125
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    while _la==10:
                        self.state = 121
                        self.match(AutolevParser.T__9)
                        self.state = 122
                        self.match(AutolevParser.ID)
                        self.state = 127
                        self._errHandler.sync(self)
                        _la = self._input.LA(1)



                self.state = 130
                self.match(AutolevParser.T__12)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VarDeclContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def varType(self):
            return self.getTypedRuleContext(AutolevParser.VarTypeContext,0)


        def varDecl2(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.VarDecl2Context)
            else:
                return self.getTypedRuleContext(AutolevParser.VarDecl2Context,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_varDecl

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVarDecl" ):
                listener.enterVarDecl(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVarDecl" ):
                listener.exitVarDecl(self)




    def varDecl(self):

        localctx = AutolevParser.VarDeclContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_varDecl)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 133
            self.varType()
            self.state = 134
            self.varDecl2()
            self.state = 139
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==10:
                self.state = 135
                self.match(AutolevParser.T__9)
                self.state = 136
                self.varDecl2()
                self.state = 141
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VarTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Newtonian(self):
            return self.getToken(AutolevParser.Newtonian, 0)

        def Frames(self):
            return self.getToken(AutolevParser.Frames, 0)

        def Bodies(self):
            return self.getToken(AutolevParser.Bodies, 0)

        def Particles(self):
            return self.getToken(AutolevParser.Particles, 0)

        def Points(self):
            return self.getToken(AutolevParser.Points, 0)

        def Constants(self):
            return self.getToken(AutolevParser.Constants, 0)

        def Specifieds(self):
            return self.getToken(AutolevParser.Specifieds, 0)

        def Imaginary(self):
            return self.getToken(AutolevParser.Imaginary, 0)

        def Variables(self):
            return self.getToken(AutolevParser.Variables, 0)

        def MotionVariables(self):
            return self.getToken(AutolevParser.MotionVariables, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_varType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVarType" ):
                listener.enterVarType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVarType" ):
                listener.exitVarType(self)




    def varType(self):

        localctx = AutolevParser.VarTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_varType)
        self._la = 0 # Token type
        try:
            self.state = 164
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [34]:
                self.enterOuterAlt(localctx, 1)
                self.state = 142
                self.match(AutolevParser.Newtonian)
                pass
            elif token in [35]:
                self.enterOuterAlt(localctx, 2)
                self.state = 143
                self.match(AutolevParser.Frames)
                pass
            elif token in [36]:
                self.enterOuterAlt(localctx, 3)
                self.state = 144
                self.match(AutolevParser.Bodies)
                pass
            elif token in [37]:
                self.enterOuterAlt(localctx, 4)
                self.state = 145
                self.match(AutolevParser.Particles)
                pass
            elif token in [38]:
                self.enterOuterAlt(localctx, 5)
                self.state = 146
                self.match(AutolevParser.Points)
                pass
            elif token in [39]:
                self.enterOuterAlt(localctx, 6)
                self.state = 147
                self.match(AutolevParser.Constants)
                pass
            elif token in [40]:
                self.enterOuterAlt(localctx, 7)
                self.state = 148
                self.match(AutolevParser.Specifieds)
                pass
            elif token in [41]:
                self.enterOuterAlt(localctx, 8)
                self.state = 149
                self.match(AutolevParser.Imaginary)
                pass
            elif token in [42]:
                self.enterOuterAlt(localctx, 9)
                self.state = 150
                self.match(AutolevParser.Variables)
                self.state = 154
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==11:
                    self.state = 151
                    self.match(AutolevParser.T__10)
                    self.state = 156
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                pass
            elif token in [43]:
                self.enterOuterAlt(localctx, 10)
                self.state = 157
                self.match(AutolevParser.MotionVariables)
                self.state = 161
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==11:
                    self.state = 158
                    self.match(AutolevParser.T__10)
                    self.state = 163
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VarDecl2Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(AutolevParser.INT)
            else:
                return self.getToken(AutolevParser.INT, i)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext,0)


        def getRuleIndex(self):
            return AutolevParser.RULE_varDecl2

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVarDecl2" ):
                listener.enterVarDecl2(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVarDecl2" ):
                listener.exitVarDecl2(self)




    def varDecl2(self):

        localctx = AutolevParser.VarDecl2Context(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_varDecl2)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 166
            self.match(AutolevParser.ID)
            self.state = 172
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,15,self._ctx)
            if la_ == 1:
                self.state = 167
                self.match(AutolevParser.T__13)
                self.state = 168
                self.match(AutolevParser.INT)
                self.state = 169
                self.match(AutolevParser.T__9)
                self.state = 170
                self.match(AutolevParser.INT)
                self.state = 171
                self.match(AutolevParser.T__14)


            self.state = 188
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,17,self._ctx)
            if la_ == 1:
                self.state = 174
                self.match(AutolevParser.T__13)
                self.state = 175
                self.match(AutolevParser.INT)
                self.state = 176
                self.match(AutolevParser.T__15)
                self.state = 177
                self.match(AutolevParser.INT)
                self.state = 184
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==10:
                    self.state = 178
                    self.match(AutolevParser.T__9)
                    self.state = 179
                    self.match(AutolevParser.INT)
                    self.state = 180
                    self.match(AutolevParser.T__15)
                    self.state = 181
                    self.match(AutolevParser.INT)
                    self.state = 186
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 187
                self.match(AutolevParser.T__14)


            self.state = 193
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==14:
                self.state = 190
                self.match(AutolevParser.T__13)
                self.state = 191
                self.match(AutolevParser.INT)
                self.state = 192
                self.match(AutolevParser.T__14)


            self.state = 196
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==17 or _la==18:
                self.state = 195
                _la = self._input.LA(1)
                if not(_la==17 or _la==18):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()


            self.state = 201
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==11:
                self.state = 198
                self.match(AutolevParser.T__10)
                self.state = 203
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 206
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==3:
                self.state = 204
                self.match(AutolevParser.T__2)
                self.state = 205
                self.expr(0)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class RangesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(AutolevParser.INT)
            else:
                return self.getToken(AutolevParser.INT, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_ranges

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRanges" ):
                listener.enterRanges(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRanges" ):
                listener.exitRanges(self)




    def ranges(self):

        localctx = AutolevParser.RangesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_ranges)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 208
            self.match(AutolevParser.T__13)
            self.state = 209
            self.match(AutolevParser.INT)
            self.state = 210
            self.match(AutolevParser.T__15)
            self.state = 211
            self.match(AutolevParser.INT)
            self.state = 218
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==10:
                self.state = 212
                self.match(AutolevParser.T__9)
                self.state = 213
                self.match(AutolevParser.INT)
                self.state = 214
                self.match(AutolevParser.T__15)
                self.state = 215
                self.match(AutolevParser.INT)
                self.state = 220
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 221
            self.match(AutolevParser.T__14)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MassDeclContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Mass(self):
            return self.getToken(AutolevParser.Mass, 0)

        def massDecl2(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.MassDecl2Context)
            else:
                return self.getTypedRuleContext(AutolevParser.MassDecl2Context,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_massDecl

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMassDecl" ):
                listener.enterMassDecl(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMassDecl" ):
                listener.exitMassDecl(self)




    def massDecl(self):

        localctx = AutolevParser.MassDeclContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_massDecl)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 223
            self.match(AutolevParser.Mass)
            self.state = 224
            self.massDecl2()
            self.state = 229
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==10:
                self.state = 225
                self.match(AutolevParser.T__9)
                self.state = 226
                self.massDecl2()
                self.state = 231
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MassDecl2Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext,0)


        def getRuleIndex(self):
            return AutolevParser.RULE_massDecl2

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMassDecl2" ):
                listener.enterMassDecl2(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMassDecl2" ):
                listener.exitMassDecl2(self)




    def massDecl2(self):

        localctx = AutolevParser.MassDecl2Context(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_massDecl2)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 232
            self.match(AutolevParser.ID)
            self.state = 233
            self.match(AutolevParser.T__2)
            self.state = 234
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InertiaDeclContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Inertia(self):
            return self.getToken(AutolevParser.Inertia, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_inertiaDecl

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInertiaDecl" ):
                listener.enterInertiaDecl(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInertiaDecl" ):
                listener.exitInertiaDecl(self)




    def inertiaDecl(self):

        localctx = AutolevParser.InertiaDeclContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_inertiaDecl)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 236
            self.match(AutolevParser.Inertia)
            self.state = 237
            self.match(AutolevParser.ID)
            self.state = 241
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==12:
                self.state = 238
                self.match(AutolevParser.T__11)
                self.state = 239
                self.match(AutolevParser.ID)
                self.state = 240
                self.match(AutolevParser.T__12)


            self.state = 245
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 243
                self.match(AutolevParser.T__9)
                self.state = 244
                self.expr(0)
                self.state = 247
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==10):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MatrixContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_matrix

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatrix" ):
                listener.enterMatrix(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatrix" ):
                listener.exitMatrix(self)




    def matrix(self):

        localctx = AutolevParser.MatrixContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_matrix)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 249
            self.match(AutolevParser.T__0)
            self.state = 250
            self.expr(0)
            self.state = 255
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==10 or _la==19:
                self.state = 251
                _la = self._input.LA(1)
                if not(_la==10 or _la==19):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 252
                self.expr(0)
                self.state = 257
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 258
            self.match(AutolevParser.T__1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MatrixInOutputContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def FLOAT(self):
            return self.getToken(AutolevParser.FLOAT, 0)

        def INT(self):
            return self.getToken(AutolevParser.INT, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_matrixInOutput

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatrixInOutput" ):
                listener.enterMatrixInOutput(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatrixInOutput" ):
                listener.exitMatrixInOutput(self)




    def matrixInOutput(self):

        localctx = AutolevParser.MatrixInOutputContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_matrixInOutput)
        self._la = 0 # Token type
        try:
            self.state = 268
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [48]:
                self.enterOuterAlt(localctx, 1)
                self.state = 260
                self.match(AutolevParser.ID)

                self.state = 261
                self.match(AutolevParser.ID)
                self.state = 262
                self.match(AutolevParser.T__2)
                self.state = 264
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==44 or _la==45:
                    self.state = 263
                    _la = self._input.LA(1)
                    if not(_la==44 or _la==45):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()


                pass
            elif token in [45]:
                self.enterOuterAlt(localctx, 2)
                self.state = 266
                self.match(AutolevParser.FLOAT)
                pass
            elif token in [44]:
                self.enterOuterAlt(localctx, 3)
                self.state = 267
                self.match(AutolevParser.INT)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CodeCommandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def units(self):
            return self.getTypedRuleContext(AutolevParser.UnitsContext,0)


        def inputs(self):
            return self.getTypedRuleContext(AutolevParser.InputsContext,0)


        def outputs(self):
            return self.getTypedRuleContext(AutolevParser.OutputsContext,0)


        def codegen(self):
            return self.getTypedRuleContext(AutolevParser.CodegenContext,0)


        def commands(self):
            return self.getTypedRuleContext(AutolevParser.CommandsContext,0)


        def getRuleIndex(self):
            return AutolevParser.RULE_codeCommands

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCodeCommands" ):
                listener.enterCodeCommands(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCodeCommands" ):
                listener.exitCodeCommands(self)




    def codeCommands(self):

        localctx = AutolevParser.CodeCommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_codeCommands)
        try:
            self.state = 275
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [32]:
                self.enterOuterAlt(localctx, 1)
                self.state = 270
                self.units()
                pass
            elif token in [29]:
                self.enterOuterAlt(localctx, 2)
                self.state = 271
                self.inputs()
                pass
            elif token in [30]:
                self.enterOuterAlt(localctx, 3)
                self.state = 272
                self.outputs()
                pass
            elif token in [48]:
                self.enterOuterAlt(localctx, 4)
                self.state = 273
                self.codegen()
                pass
            elif token in [31, 33]:
                self.enterOuterAlt(localctx, 5)
                self.state = 274
                self.commands()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SettingsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def EXP(self):
            return self.getToken(AutolevParser.EXP, 0)

        def FLOAT(self):
            return self.getToken(AutolevParser.FLOAT, 0)

        def INT(self):
            return self.getToken(AutolevParser.INT, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_settings

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSettings" ):
                listener.enterSettings(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSettings" ):
                listener.exitSettings(self)




    def settings(self):

        localctx = AutolevParser.SettingsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_settings)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 277
            self.match(AutolevParser.ID)
            self.state = 279
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,30,self._ctx)
            if la_ == 1:
                self.state = 278
                _la = self._input.LA(1)
                if not(((_la) & ~0x3f) == 0 and ((1 << _la) & 404620279021568) != 0):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class UnitsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def UnitSystem(self):
            return self.getToken(AutolevParser.UnitSystem, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def getRuleIndex(self):
            return AutolevParser.RULE_units

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterUnits" ):
                listener.enterUnits(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitUnits" ):
                listener.exitUnits(self)




    def units(self):

        localctx = AutolevParser.UnitsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_units)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 281
            self.match(AutolevParser.UnitSystem)
            self.state = 282
            self.match(AutolevParser.ID)
            self.state = 287
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==10:
                self.state = 283
                self.match(AutolevParser.T__9)
                self.state = 284
                self.match(AutolevParser.ID)
                self.state = 289
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InputsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Input(self):
            return self.getToken(AutolevParser.Input, 0)

        def inputs2(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.Inputs2Context)
            else:
                return self.getTypedRuleContext(AutolevParser.Inputs2Context,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_inputs

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInputs" ):
                listener.enterInputs(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInputs" ):
                listener.exitInputs(self)




    def inputs(self):

        localctx = AutolevParser.InputsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_inputs)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 290
            self.match(AutolevParser.Input)
            self.state = 291
            self.inputs2()
            self.state = 296
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==10:
                self.state = 292
                self.match(AutolevParser.T__9)
                self.state = 293
                self.inputs2()
                self.state = 298
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Id_diffContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def diff(self):
            return self.getTypedRuleContext(AutolevParser.DiffContext,0)


        def getRuleIndex(self):
            return AutolevParser.RULE_id_diff

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterId_diff" ):
                listener.enterId_diff(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitId_diff" ):
                listener.exitId_diff(self)




    def id_diff(self):

        localctx = AutolevParser.Id_diffContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_id_diff)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 299
            self.match(AutolevParser.ID)
            self.state = 301
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==11:
                self.state = 300
                self.diff()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Inputs2Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def id_diff(self):
            return self.getTypedRuleContext(AutolevParser.Id_diffContext,0)


        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_inputs2

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInputs2" ):
                listener.enterInputs2(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInputs2" ):
                listener.exitInputs2(self)




    def inputs2(self):

        localctx = AutolevParser.Inputs2Context(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_inputs2)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 303
            self.id_diff()
            self.state = 304
            self.match(AutolevParser.T__2)
            self.state = 305
            self.expr(0)
            self.state = 307
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,34,self._ctx)
            if la_ == 1:
                self.state = 306
                self.expr(0)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class OutputsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Output(self):
            return self.getToken(AutolevParser.Output, 0)

        def outputs2(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.Outputs2Context)
            else:
                return self.getTypedRuleContext(AutolevParser.Outputs2Context,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_outputs

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOutputs" ):
                listener.enterOutputs(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOutputs" ):
                listener.exitOutputs(self)




    def outputs(self):

        localctx = AutolevParser.OutputsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_outputs)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 309
            self.match(AutolevParser.Output)
            self.state = 310
            self.outputs2()
            self.state = 315
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==10:
                self.state = 311
                self.match(AutolevParser.T__9)
                self.state = 312
                self.outputs2()
                self.state = 317
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Outputs2Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_outputs2

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOutputs2" ):
                listener.enterOutputs2(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOutputs2" ):
                listener.exitOutputs2(self)




    def outputs2(self):

        localctx = AutolevParser.Outputs2Context(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_outputs2)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 318
            self.expr(0)
            self.state = 320
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,36,self._ctx)
            if la_ == 1:
                self.state = 319
                self.expr(0)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CodegenContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def functionCall(self):
            return self.getTypedRuleContext(AutolevParser.FunctionCallContext,0)


        def matrixInOutput(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.MatrixInOutputContext)
            else:
                return self.getTypedRuleContext(AutolevParser.MatrixInOutputContext,i)


        def getRuleIndex(self):
            return AutolevParser.RULE_codegen

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCodegen" ):
                listener.enterCodegen(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCodegen" ):
                listener.exitCodegen(self)




    def codegen(self):

        localctx = AutolevParser.CodegenContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_codegen)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 322
            self.match(AutolevParser.ID)
            self.state = 323
            self.functionCall()
            self.state = 335
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==1:
                self.state = 324
                self.match(AutolevParser.T__0)
                self.state = 325
                self.matrixInOutput()
                self.state = 330
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==10:
                    self.state = 326
                    self.match(AutolevParser.T__9)
                    self.state = 327
                    self.matrixInOutput()
                    self.state = 332
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 333
                self.match(AutolevParser.T__1)


            self.state = 337
            self.match(AutolevParser.ID)
            self.state = 338
            self.match(AutolevParser.T__19)
            self.state = 339
            self.match(AutolevParser.ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CommandsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Save(self):
            return self.getToken(AutolevParser.Save, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(AutolevParser.ID)
            else:
                return self.getToken(AutolevParser.ID, i)

        def Encode(self):
            return self.getToken(AutolevParser.Encode, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_commands

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCommands" ):
                listener.enterCommands(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCommands" ):
                listener.exitCommands(self)




    def commands(self):

        localctx = AutolevParser.CommandsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_commands)
        self._la = 0 # Token type
        try:
            self.state = 354
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [31]:
                self.enterOuterAlt(localctx, 1)
                self.state = 341
                self.match(AutolevParser.Save)
                self.state = 342
                self.match(AutolevParser.ID)
                self.state = 343
                self.match(AutolevParser.T__19)
                self.state = 344
                self.match(AutolevParser.ID)
                pass
            elif token in [33]:
                self.enterOuterAlt(localctx, 2)
                self.state = 345
                self.match(AutolevParser.Encode)
                self.state = 346
                self.match(AutolevParser.ID)
                self.state = 351
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==10:
                    self.state = 347
                    self.match(AutolevParser.T__9)
                    self.state = 348
                    self.match(AutolevParser.ID)
                    self.state = 353
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VecContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def getRuleIndex(self):
            return AutolevParser.RULE_vec

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVec" ):
                listener.enterVec(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVec" ):
                listener.exitVec(self)




    def vec(self):

        localctx = AutolevParser.VecContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_vec)
        try:
            self.state = 364
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [48]:
                self.enterOuterAlt(localctx, 1)
                self.state = 356
                self.match(AutolevParser.ID)
                self.state = 358
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 357
                        self.match(AutolevParser.T__20)

                    else:
                        raise NoViableAltException(self)
                    self.state = 360
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,41,self._ctx)

                pass
            elif token in [22]:
                self.enterOuterAlt(localctx, 2)
                self.state = 362
                self.match(AutolevParser.T__21)
                pass
            elif token in [23]:
                self.enterOuterAlt(localctx, 3)
                self.state = 363
                self.match(AutolevParser.T__22)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return AutolevParser.RULE_expr


        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)


    class ParensContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterParens" ):
                listener.enterParens(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitParens" ):
                listener.exitParens(self)


    class VectorOrDyadicContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def vec(self):
            return self.getTypedRuleContext(AutolevParser.VecContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVectorOrDyadic" ):
                listener.enterVectorOrDyadic(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVectorOrDyadic" ):
                listener.exitVectorOrDyadic(self)


    class ExponentContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExponent" ):
                listener.enterExponent(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExponent" ):
                listener.exitExponent(self)


    class MulDivContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMulDiv" ):
                listener.enterMulDiv(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMulDiv" ):
                listener.exitMulDiv(self)


    class AddSubContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAddSub" ):
                listener.enterAddSub(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAddSub" ):
                listener.exitAddSub(self)


    class FloatContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def FLOAT(self):
            return self.getToken(AutolevParser.FLOAT, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFloat" ):
                listener.enterFloat(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFloat" ):
                listener.exitFloat(self)


    class IntContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def INT(self):
            return self.getToken(AutolevParser.INT, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInt" ):
                listener.enterInt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInt" ):
                listener.exitInt(self)


    class IdEqualsExprContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIdEqualsExpr" ):
                listener.enterIdEqualsExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIdEqualsExpr" ):
                listener.exitIdEqualsExpr(self)


    class NegativeOneContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self):
            return self.getTypedRuleContext(AutolevParser.ExprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNegativeOne" ):
                listener.enterNegativeOne(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNegativeOne" ):
                listener.exitNegativeOne(self)


    class FunctionContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def functionCall(self):
            return self.getTypedRuleContext(AutolevParser.FunctionCallContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFunction" ):
                listener.enterFunction(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFunction" ):
                listener.exitFunction(self)


    class RangessContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ranges(self):
            return self.getTypedRuleContext(AutolevParser.RangesContext,0)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRangess" ):
                listener.enterRangess(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRangess" ):
                listener.exitRangess(self)


    class ColonContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterColon" ):
                listener.enterColon(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitColon" ):
                listener.exitColon(self)


    class IdContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterId" ):
                listener.enterId(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitId" ):
                listener.exitId(self)


    class ExpContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def EXP(self):
            return self.getToken(AutolevParser.EXP, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExp" ):
                listener.enterExp(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExp" ):
                listener.exitExp(self)


    class MatricesContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def matrix(self):
            return self.getTypedRuleContext(AutolevParser.MatrixContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatrices" ):
                listener.enterMatrices(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatrices" ):
                listener.exitMatrices(self)


    class IndexingContext(ExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a AutolevParser.ExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(AutolevParser.ID, 0)
        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AutolevParser.ExprContext)
            else:
                return self.getTypedRuleContext(AutolevParser.ExprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIndexing" ):
                listener.enterIndexing(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIndexing" ):
                listener.exitIndexing(self)



    def expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AutolevParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 54
        self.enterRecursionRule(localctx, 54, self.RULE_expr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 408
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,47,self._ctx)
            if la_ == 1:
                localctx = AutolevParser.ExpContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 367
                self.match(AutolevParser.EXP)
                pass

            elif la_ == 2:
                localctx = AutolevParser.NegativeOneContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 368
                self.match(AutolevParser.T__17)
                self.state = 369
                self.expr(12)
                pass

            elif la_ == 3:
                localctx = AutolevParser.FloatContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 370
                self.match(AutolevParser.FLOAT)
                pass

            elif la_ == 4:
                localctx = AutolevParser.IntContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 371
                self.match(AutolevParser.INT)
                pass

            elif la_ == 5:
                localctx = AutolevParser.IdContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 372
                self.match(AutolevParser.ID)
                self.state = 376
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,43,self._ctx)
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1:
                        self.state = 373
                        self.match(AutolevParser.T__10)
                    self.state = 378
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,43,self._ctx)

                pass

            elif la_ == 6:
                localctx = AutolevParser.VectorOrDyadicContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 379
                self.vec()
                pass

            elif la_ == 7:
                localctx = AutolevParser.IndexingContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 380
                self.match(AutolevParser.ID)
                self.state = 381
                self.match(AutolevParser.T__0)
                self.state = 382
                self.expr(0)
                self.state = 387
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==10:
                    self.state = 383
                    self.match(AutolevParser.T__9)
                    self.state = 384
                    self.expr(0)
                    self.state = 389
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 390
                self.match(AutolevParser.T__1)
                pass

            elif la_ == 8:
                localctx = AutolevParser.FunctionContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 392
                self.functionCall()
                pass

            elif la_ == 9:
                localctx = AutolevParser.MatricesContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 393
                self.matrix()
                pass

            elif la_ == 10:
                localctx = AutolevParser.ParensContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 394
                self.match(AutolevParser.T__11)
                self.state = 395
                self.expr(0)
                self.state = 396
                self.match(AutolevParser.T__12)
                pass

            elif la_ == 11:
                localctx = AutolevParser.RangessContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 399
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==48:
                    self.state = 398
                    self.match(AutolevParser.ID)


                self.state = 401
                self.ranges()
                self.state = 405
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,46,self._ctx)
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1:
                        self.state = 402
                        self.match(AutolevParser.T__10)
                    self.state = 407
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,46,self._ctx)

                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 427
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,49,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 425
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,48,self._ctx)
                    if la_ == 1:
                        localctx = AutolevParser.ExponentContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 410
                        if not self.precpred(self._ctx, 16):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 16)")
                        self.state = 411
                        self.match(AutolevParser.T__23)
                        self.state = 412
                        self.expr(17)
                        pass

                    elif la_ == 2:
                        localctx = AutolevParser.MulDivContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 413
                        if not self.precpred(self._ctx, 15):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 15)")
                        self.state = 414
                        _la = self._input.LA(1)
                        if not(_la==25 or _la==26):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 415
                        self.expr(16)
                        pass

                    elif la_ == 3:
                        localctx = AutolevParser.AddSubContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 416
                        if not self.precpred(self._ctx, 14):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 14)")
                        self.state = 417
                        _la = self._input.LA(1)
                        if not(_la==17 or _la==18):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 418
                        self.expr(15)
                        pass

                    elif la_ == 4:
                        localctx = AutolevParser.IdEqualsExprContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 419
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 420
                        self.match(AutolevParser.T__2)
                        self.state = 421
                        self.expr(4)
                        pass

                    elif la_ == 5:
                        localctx = AutolevParser.ColonContext(self, AutolevParser.ExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 422
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 423
                        self.match(AutolevParser.T__15)
                        self.state = 424
                        self.expr(3)
                        pass


                self.state = 429
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,49,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[27] = self.expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expr_sempred(self, localctx:ExprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 16)


            if predIndex == 1:
                return self.precpred(self._ctx, 15)


            if predIndex == 2:
                return self.precpred(self._ctx, 14)


            if predIndex == 3:
                return self.precpred(self._ctx, 3)


            if predIndex == 4:
                return self.precpred(self._ctx, 2)




