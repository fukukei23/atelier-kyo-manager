
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\string_.py ===
from __future__ import annotations

from functools import partial
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._config import (
    get_option,
    using_string_dtype,
)

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.lib import ensure_string_array
from pandas.compat import (
    HAS_PYARROW,
    pa_version_under10p1,
)
from pandas.compat.numpy import function as nv
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import (
    ExtensionDtype,
    StorageExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)

from pandas.core import (
    missing,
    nanops,
    ops,
)
from pandas.core.algorithms import isin
from pandas.core.array_algos import masked_reductions
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import (
    FloatingArray,
    FloatingDtype,
)
from pandas.core.arrays.integer import (
    IntegerArray,
    IntegerDtype,
)
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna

from pandas.io.formats import printing

if TYPE_CHECKING:
    import pyarrow

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        DtypeObj,
        NumpySorter,
        NumpyValueArrayLike,
        Scalar,
        Self,
        npt,
        type_t,
    )

    from pandas import Series


@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    """
    Extension dtype for string data.

    .. warning::

       StringDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    storage : {"python", "pyarrow"}, optional
        If not given, the value of ``pd.options.mode.string_storage``.
    na_value : {np.nan, pd.NA}, default pd.NA
        Whether the dtype follows NaN or NA missing value semantics.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.StringDtype()
    string[python]

    >>> pd.StringDtype(storage="pyarrow")
    string[pyarrow]
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        if self._na_value is libmissing.NA:
            return "string"
        else:
            return "str"

    #: StringDtype().na_value uses pandas.NA except the implementation that
    # follows NumPy semantics, which uses nan.
    @property
    def na_value(self) -> libmissing.NAType | float:  # type: ignore[override]
        return self._na_value

    _metadata = ("storage", "_na_value")  # type: ignore[assignment]

    def __init__(
        self,
        storage: str | None = None,
        na_value: libmissing.NAType | float = libmissing.NA,
    ) -> None:
        # infer defaults
        if storage is None:
            if na_value is not libmissing.NA:
                storage = get_option("mode.string_storage")
                if storage == "auto":
                    if HAS_PYARROW:
                        storage = "pyarrow"
                    else:
                        storage = "python"
            else:
                storage = get_option("mode.string_storage")
                if storage == "auto":
                    storage = "python"

        if storage == "pyarrow_numpy":
            warnings.warn(
                "The 'pyarrow_numpy' storage option name is deprecated and will be "
                'removed in pandas 3.0. Use \'pd.StringDtype(storage="pyarrow", '
                "na_value-np.nan)' to construct the same dtype.\nOr enable the "
                "'pd.options.future.infer_string = True' option globally and use "
                'the "str" alias as a shorthand notation to specify a dtype '
                '(instead of "string[pyarrow_numpy]").',
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            storage = "pyarrow"
            na_value = np.nan

        # validate options
        if storage not in {"python", "pyarrow"}:
            raise ValueError(
                f"Storage must be 'python' or 'pyarrow'. Got {storage} instead."
            )
        if storage == "pyarrow" and pa_version_under10p1:
            raise ImportError(
                "pyarrow>=10.0.1 is required for PyArrow backed StringArray."
            )

        if isinstance(na_value, float) and np.isnan(na_value):
            # when passed a NaN value, always set to np.nan to ensure we use
            # a consistent NaN value (and we can use `dtype.na_value is np.nan`)
            na_value = np.nan
        elif na_value is not libmissing.NA:
            raise ValueError(f"'na_value' must be np.nan or pd.NA, got {na_value}")

        self.storage = cast(str, storage)
        self._na_value = na_value

    def __repr__(self) -> str:
        if self._na_value is libmissing.NA:
            return f"{self.name}[{self.storage}]"
        else:
            # TODO add more informative repr
            return self.name

    def __eq__(self, other: object) -> bool:
        # we need to override the base class __eq__ because na_value (NA or NaN)
        # cannot be checked with normal `==`
        if isinstance(other, str):
            # TODO should dtype == "string" work for the NaN variant?
            if other == "string" or other == self.name:  # noqa: PLR1714
                return True
            try:
                other = self.construct_from_string(other)
            except (TypeError, ImportError):
                # TypeError if `other` is not a valid string for StringDtype
                # ImportError if pyarrow is not installed for "string[pyarrow]"
                return False
        if isinstance(other, type(self)):
            return self.storage == other.storage and self.na_value is other.na_value
        return False

    def __hash__(self) -> int:
        # need to override __hash__ as well because of overriding __eq__
        return super().__hash__()

    def __reduce__(self):
        return StringDtype, (self.storage, self.na_value)

    @property
    def type(self) -> type[str]:
        return str

    @classmethod
    def construct_from_string(cls, string) -> Self:
        """
        Construct a StringDtype from a string.

        Parameters
        ----------
        string : str
            The type of the name. The storage type will be taking from `string`.
            Valid options and their storage types are

            ========================== ==============================================
            string                     result storage
            ========================== ==============================================
            ``'string'``               pd.options.mode.string_storage, default python
            ``'string[python]'``       python
            ``'string[pyarrow]'``      pyarrow
            ========================== ==============================================

        Returns
        -------
        StringDtype

        Raise
        -----
        TypeError
            If the string is not a valid option.
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if string == "string":
            return cls()
        elif string == "str" and using_string_dtype():
            return cls(na_value=np.nan)
        elif string == "string[python]":
            return cls(storage="python")
        elif string == "string[pyarrow]":
            return cls(storage="pyarrow")
        elif string == "string[pyarrow_numpy]":
            # this is deprecated in the dtype __init__, remove this in pandas 3.0
            return cls(storage="pyarrow_numpy")
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    # https://github.com/pandas-dev/pandas/issues/36126
    # error: Signature of "construct_array_type" incompatible with supertype
    # "ExtensionDtype"
    def construct_array_type(  # type: ignore[override]
        self,
    ) -> type_t[BaseStringArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.string_arrow import (
            ArrowStringArray,
            ArrowStringArrayNumpySemantics,
        )

        if self.storage == "python" and self._na_value is libmissing.NA:
            return StringArray
        elif self.storage == "pyarrow" and self._na_value is libmissing.NA:
            return ArrowStringArray
        elif self.storage == "python":
            return StringArrayNumpySemantics
        else:
            return ArrowStringArrayNumpySemantics

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        storages = set()
        na_values = set()

        for dtype in dtypes:
            if isinstance(dtype, StringDtype):
                storages.add(dtype.storage)
                na_values.add(dtype.na_value)
            elif isinstance(dtype, np.dtype) and dtype.kind in ("U", "T"):
                continue
            else:
                return None

        if len(storages) == 2:
            # if both python and pyarrow storage -> priority to pyarrow
            storage = "pyarrow"
        else:
            storage = next(iter(storages))  # type: ignore[assignment]

        na_value: libmissing.NAType | float
        if len(na_values) == 2:
            # if both NaN and NA -> priority to NA
            na_value = libmissing.NA
        else:
            na_value = next(iter(na_values))

        return StringDtype(storage=storage, na_value=na_value)

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseStringArray:
        """
        Construct StringArray from pyarrow Array/ChunkedArray.
        """
        if self.storage == "pyarrow":
            if self._na_value is libmissing.NA:
                from pandas.core.arrays.string_arrow import ArrowStringArray

                return ArrowStringArray(array)
            else:
                from pandas.core.arrays.string_arrow import (
                    ArrowStringArrayNumpySemantics,
                )

                return ArrowStringArrayNumpySemantics(array)

        else:
            import pyarrow

            if isinstance(array, pyarrow.Array):
                chunks = [array]
            else:
                # pyarrow.ChunkedArray
                chunks = array.chunks

            results = []
            for arr in chunks:
                # convert chunk by chunk to numpy and concatenate then, to avoid
                # overflow for large string data when concatenating the pyarrow arrays
                arr = arr.to_numpy(zero_copy_only=False)
                arr = ensure_string_array(arr, na_value=self.na_value)
                results.append(arr)

        if len(chunks) == 0:
            arr = np.array([], dtype=object)
        else:
            arr = np.concatenate(results)

        # Bypass validation inside StringArray constructor, see GH#47781
        new_string_array = StringArray.__new__(StringArray)
        NDArrayBacked.__init__(new_string_array, arr, self)
        return new_string_array


class BaseStringArray(ExtensionArray):
    """
    Mixin class for StringArray, ArrowStringArray.
    """

    dtype: StringDtype

    @doc(ExtensionArray.tolist)
    def tolist(self):
        if self.ndim > 1:
            return [x.tolist() for x in self]
        return list(self.to_numpy())

    @classmethod
    def _from_scalars(cls, scalars, dtype: DtypeObj) -> Self:
        if lib.infer_dtype(scalars, skipna=True) not in ["string", "empty"]:
            # TODO: require any NAs be valid-for-string
            raise ValueError
        return cls._from_sequence(scalars, dtype=dtype)

    def _formatter(self, boxed: bool = False):
        formatter = partial(
            printing.pprint_thing,
            escape_chars=("\t", "\r", "\n"),
            quote_strings=not boxed,
        )
        return formatter

    def _str_map(
        self,
        f,
        na_value=lib.no_default,
        dtype: Dtype | None = None,
        convert: bool = True,
    ):
        if self.dtype.na_value is np.nan:
            return self._str_map_nan_semantics(
                f, na_value=na_value, dtype=dtype, convert=convert
            )

        from pandas.arrays import BooleanArray

        if dtype is None:
            dtype = self.dtype
        if na_value is lib.no_default:
            na_value = self.dtype.na_value

        mask = isna(self)
        arr = np.asarray(self)

        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            constructor: type[IntegerArray | BooleanArray]
            if is_integer_dtype(dtype):
                constructor = IntegerArray
            else:
                constructor = BooleanArray

            na_value_is_na = isna(na_value)
            if na_value_is_na:
                na_value = 1
            elif dtype == np.dtype("bool"):
                # GH#55736
                na_value = bool(na_value)
            result = lib.map_infer_mask(
                arr,
                f,
                mask.view("uint8"),
                convert=False,
                na_value=na_value,
                # error: Argument 1 to "dtype" has incompatible type
                # "Union[ExtensionDtype, str, dtype[Any], Type[object]]"; expected
                # "Type[object]"
                dtype=np.dtype(cast(type, dtype)),
            )

            if not na_value_is_na:
                mask[:] = False

            return constructor(result, mask)

        else:
            return self._str_map_str_or_object(dtype, na_value, arr, f, mask)

    def _str_map_str_or_object(
        self,
        dtype,
        na_value,
        arr: np.ndarray,
        f,
        mask: npt.NDArray[np.bool_],
    ):
        # _str_map helper for case where dtype is either string dtype or object
        if is_string_dtype(dtype) and not is_object_dtype(dtype):
            # i.e. StringDtype
            result = lib.map_infer_mask(
                arr, f, mask.view("uint8"), convert=False, na_value=na_value
            )
            if self.dtype.storage == "pyarrow":
                import pyarrow as pa

                result = pa.array(
                    result, mask=mask, type=pa.large_string(), from_pandas=True
                )
            # error: Too many arguments for "BaseStringArray"
            return type(self)(result)  # type: ignore[call-arg]

        else:
            # This is when the result type is object. We reach this when
            # -> We know the result type is truly object (e.g. .encode returns bytes
            #    or .findall returns a list).
            # -> We don't know the result type. E.g. `.get` can return anything.
            return lib.map_infer_mask(arr, f, mask.view("uint8"))

    def _str_map_nan_semantics(
        self,
        f,
        na_value=lib.no_default,
        dtype: Dtype | None = None,
        convert: bool = True,
    ):
        if dtype is None:
            dtype = self.dtype
        if na_value is lib.no_default:
            if is_bool_dtype(dtype):
                # NaN propagates as False
                na_value = False
            else:
                na_value = self.dtype.na_value

        mask = isna(self)
        arr = np.asarray(self)

        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            na_value_is_na = isna(na_value)
            if na_value_is_na:
                if is_integer_dtype(dtype):
                    na_value = 0
                else:
                    # NaN propagates as False
                    na_value = False

            result = lib.map_infer_mask(
                arr,
                f,
                mask.view("uint8"),
                convert=False,
                na_value=na_value,
                dtype=np.dtype(cast(type, dtype)),
            )
            if na_value_is_na and is_integer_dtype(dtype) and mask.any():
                # TODO: we could alternatively do this check before map_infer_mask
                #  and adjust the dtype/na_value we pass there. Which is more
                #  performant?
                result = result.astype("float64")
                result[mask] = np.nan

            return result

        else:
            return self._str_map_str_or_object(dtype, na_value, arr, f, mask)

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        if dtype is not None:
            raise TypeError("Cannot change data-type for string array.")
        return super().view(dtype=dtype)


# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class StringArray(BaseStringArray, NumpyExtensionArray):  # type: ignore[misc]
    """
    Extension array for string data.

    .. warning::

       StringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : array-like
        The array of data.

        .. warning::

           Currently, this expects an object-dtype ndarray
           where the elements are Python strings
           or nan-likes (``None``, ``np.nan``, ``NA``).
           This may change without warning in the future. Use
           :meth:`pandas.array` with ``dtype="string"`` for a stable way of
           creating a `StringArray` from any sequence.

        .. versionchanged:: 1.5.0

           StringArray now accepts array-likes containing
           nan-likes(``None``, ``np.nan``) for the ``values`` parameter
           in addition to strings and :attr:`pandas.NA`

    copy : bool, default False
        Whether to copy the array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`pandas.array`
        The recommended function for creating a StringArray.
    Series.str
        The string methods are available on Series backed by
        a StringArray.

    Notes
    -----
    StringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(['This is', 'some text', None, 'data.'], dtype="string")
    <StringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string

    Unlike arrays instantiated with ``dtype="object"``, ``StringArray``
    will convert the values to strings.

    >>> pd.array(['1', 1], dtype="object")
    <NumpyExtensionArray>
    ['1', 1]
    Length: 2, dtype: object
    >>> pd.array(['1', 1], dtype="string")
    <StringArray>
    ['1', '1']
    Length: 2, dtype: string

    However, instantiating StringArrays directly with non-strings will raise an error.

    For comparison methods, `StringArray` returns a :class:`pandas.BooleanArray`:

    >>> pd.array(["a", None, "c"], dtype="string") == "a"
    <BooleanArray>
    [True, <NA>, False]
    Length: 3, dtype: boolean
    """

    # undo the NumpyExtensionArray hack
    _typ = "extension"
    _storage = "python"
    _na_value: libmissing.NAType | float = libmissing.NA

    def __init__(self, values, copy: bool = False) -> None:
        values = extract_array(values)

        super().__init__(values, copy=copy)
        if not isinstance(values, type(self)):
            self._validate()
        NDArrayBacked.__init__(
            self,
            self._ndarray,
            StringDtype(storage=self._storage, na_value=self._na_value),
        )

    def _validate(self):
        """Validate that we only store NA or strings."""
        if len(self._ndarray) and not lib.is_string_array(self._ndarray, skipna=True):
            raise ValueError("StringArray requires a sequence of strings or pandas.NA")
        if self._ndarray.dtype != "object":
            raise ValueError(
                "StringArray requires a sequence of strings or pandas.NA. Got "
                f"'{self._ndarray.dtype}' dtype instead."
            )
        # Check to see if need to convert Na values to pd.NA
        if self._ndarray.ndim > 2:
            # Ravel if ndims > 2 b/c no cythonized version available
            lib.convert_nans_to_NA(self._ndarray.ravel("K"))
        else:
            lib.convert_nans_to_NA(self._ndarray)

    def _validate_scalar(self, value):
        # used by NDArrayBackedExtensionIndex.insert
        if isna(value):
            return self.dtype.na_value
        elif not isinstance(value, str):
            raise TypeError(
                f"Invalid value '{value}' for dtype '{self.dtype}'. Value should be a "
                f"string or missing value, got '{type(value).__name__}' instead."
            )
        return value

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        if dtype and not (isinstance(dtype, str) and dtype == "string"):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == "python"
        else:
            if using_string_dtype():
                dtype = StringDtype(storage="python", na_value=np.nan)
            else:
                dtype = StringDtype(storage="python")

        from pandas.core.arrays.masked import BaseMaskedArray

        na_value = dtype.na_value
        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            result[na_values] = na_value

        else:
            if lib.is_pyarrow_array(scalars):
                # pyarrow array; we cannot rely on the "to_numpy" check in
                #  ensure_string_array because calling scalars.to_numpy would set
                #  zero_copy_only to True which caused problems see GH#52076
                scalars = np.array(scalars)
            # convert non-na-likes to str, and nan-likes to StringDtype().na_value
            result = lib.ensure_string_array(scalars, na_value=na_value, copy=copy)

        # Manually creating new array avoids the validation step in the __init__, so is
        # faster. Refactor need for validation?
        new_string_array = cls.__new__(cls)
        NDArrayBacked.__init__(new_string_array, result, dtype)

        return new_string_array

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _empty(cls, shape, dtype) -> StringArray:
        values = np.empty(shape, dtype=object)
        values[:] = libmissing.NA
        return cls(values).astype(dtype, copy=False)

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow as pa

        if type is None:
            type = pa.string()

        values = self._ndarray.copy()
        values[self.isna()] = None
        return pa.array(values, type=type, from_pandas=True)

    def _values_for_factorize(self) -> tuple[np.ndarray, libmissing.NAType | float]:  # type: ignore[override]
        arr = self._ndarray.copy()

        return arr, self.dtype.na_value

    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
        if lib.is_scalar(value):
            if isna(value):
                value = self.dtype.na_value
            elif not isinstance(value, str):
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'. Value should "
                    f"be a string or missing value, got '{type(value).__name__}' "
                    "instead."
                )
        else:
            value = extract_array(value, extract_numpy=True)
            if not is_array_like(value):
                value = np.asarray(value, dtype=object)
            elif isinstance(value.dtype, type(self.dtype)):
                return value
            else:
                # cast categories and friends to arrays to see if values are
                # compatible, compatibility with arrow backed strings
                value = np.asarray(value)
            if len(value) and not lib.is_string_array(value, skipna=True):
                raise TypeError(
                    "Invalid value for dtype 'str'. Value should be a "
                    "string or missing value (or array of those)."
                )
        return value

    def __setitem__(self, key, value) -> None:
        value = self._maybe_convert_setitem_value(value)

        key = check_array_indexer(self, key)
        scalar_key = lib.is_scalar(key)
        scalar_value = lib.is_scalar(value)
        if scalar_key and not scalar_value:
            raise ValueError("setting an array element with a sequence.")

        if not scalar_value:
            if value.dtype == self.dtype:
                value = value._ndarray
            else:
                value = np.asarray(value)
                mask = isna(value)
                if mask.any():
                    value = value.copy()
                    value[isna(value)] = self.dtype.na_value

        super().__setitem__(key, value)

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        # the super() method NDArrayBackedExtensionArray._putmask uses
        # np.putmask which doesn't properly handle None/pd.NA, so using the
        # base class implementation that uses __setitem__
        ExtensionArray._putmask(self, mask, value)

    def _where(self, mask: npt.NDArray[np.bool_], value) -> Self:
        # the super() method NDArrayBackedExtensionArray._where uses
        # np.putmask which doesn't properly handle None/pd.NA, so using the
        # base class implementation that uses __setitem__
        return ExtensionArray._where(self, mask, value)

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        if isinstance(values, BaseStringArray) or (
            isinstance(values, ExtensionArray) and is_string_dtype(values.dtype)
        ):
            values = values.astype(self.dtype, copy=False)
        else:
            if not lib.is_string_array(np.asarray(values), skipna=True):
                values = np.array(
                    [val for val in values if isinstance(val, str) or isna(val)],
                    dtype=object,
                )
                if not len(values):
                    return np.zeros(self.shape, dtype=bool)

            values = self._from_sequence(values, dtype=self.dtype)

        return isin(np.asarray(self), np.asarray(values))

    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        elif isinstance(dtype, IntegerDtype):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype.numpy_dtype)
            return IntegerArray(values, mask, copy=False)
        elif isinstance(dtype, FloatingDtype):
            arr = self.copy()
            mask = self.isna()
            arr[mask] = "0"
            values = arr.astype(dtype.numpy_dtype)
            return FloatingArray(values, mask, copy=False)
        elif isinstance(dtype, ExtensionDtype):
            # Skip the NumpyExtensionArray.astype method
            return ExtensionArray.astype(self, dtype, copy)
        elif np.issubdtype(dtype, np.floating):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype)
            values[mask] = np.nan
            return values

        return super().astype(dtype, copy)

    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        axis: AxisInt | None = 0,
        **kwargs,
    ):
        if self.dtype.na_value is np.nan and name in ["any", "all"]:
            if name == "any":
                return nanops.nanany(self._ndarray, skipna=skipna)
            else:
                return nanops.nanall(self._ndarray, skipna=skipna)

        if name in ["min", "max", "argmin", "argmax", "sum"]:
            result = getattr(self, name)(skipna=skipna, axis=axis, **kwargs)
            if keepdims:
                return self._from_sequence([result], dtype=self.dtype)
            return result
        raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> StringArray:
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            - cummin
            - cummax
            - cumsum
            - cumprod
        skipna : bool, default True
            If True, skip NA values.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, there is no supported kwarg.

        Returns
        -------
        array

        Raises
        ------
        NotImplementedError : subclass does not define accumulations
        """
        if name == "cumprod":
            msg = f"operation '{name}' not supported for dtype '{self.dtype}'"
            raise TypeError(msg)

        # We may need to strip out trailing NA values
        tail: np.ndarray | None = None
        na_mask: np.ndarray | None = None
        ndarray = self._ndarray
        np_func = {
            "cumsum": np.cumsum,
            "cummin": np.minimum.accumulate,
            "cummax": np.maximum.accumulate,
        }[name]

        if self._hasna:
            na_mask = cast("npt.NDArray[np.bool_]", isna(ndarray))
            if np.all(na_mask):
                return type(self)(ndarray)
            if skipna:
                if name == "cumsum":
                    ndarray = np.where(na_mask, "", ndarray)
                else:
                    # We can retain the running min/max by forward/backward filling.
                    ndarray = ndarray.copy()
                    missing.pad_or_backfill_inplace(
                        ndarray,
                        method="pad",
                        axis=0,
                    )
                    missing.pad_or_backfill_inplace(
                        ndarray,
                        method="backfill",
                        axis=0,
                    )
            else:
                # When not skipping NA values, the result should be null from
                # the first NA value onward.
                idx = np.argmax(na_mask)
                tail = np.empty(len(ndarray) - idx, dtype="object")
                tail[:] = self.dtype.na_value
                ndarray = ndarray[:idx]

        # mypy: Cannot call function of unknown type
        np_result = np_func(ndarray)  # type: ignore[operator]

        if tail is not None:
            np_result = np.hstack((np_result, tail))
        elif na_mask is not None:
            # Argument 2 to "where" has incompatible type "NAType | float"
            np_result = np.where(na_mask, self.dtype.na_value, np_result)  # type: ignore[arg-type]

        result = type(self)(np_result)
        return result

    def _wrap_reduction_result(self, axis: AxisInt | None, result) -> Any:
        if self.dtype.na_value is np.nan and result is libmissing.NA:
            # the masked_reductions use pd.NA -> convert to np.nan
            return np.nan
        return super()._wrap_reduction_result(axis, result)

    def min(self, axis=None, skipna: bool = True, **kwargs) -> Scalar:
        nv.validate_min((), kwargs)
        result = masked_reductions.min(
            values=self.to_numpy(), mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def max(self, axis=None, skipna: bool = True, **kwargs) -> Scalar:
        nv.validate_max((), kwargs)
        result = masked_reductions.max(
            values=self.to_numpy(), mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def sum(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        nv.validate_sum((), kwargs)
        result = masked_reductions.sum(
            values=self._ndarray, mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def value_counts(self, dropna: bool = True) -> Series:
        from pandas.core.algorithms import value_counts_internal as value_counts

        result = value_counts(self._ndarray, dropna=dropna).astype("Int64")
        result = value_counts(self._ndarray, sort=False, dropna=dropna)
        result.index = result.index.astype(self.dtype)

        if self.dtype.na_value is libmissing.NA:
            result = result.astype("Int64")
        return result

    def memory_usage(self, deep: bool = False) -> int:
        result = self._ndarray.nbytes
        if deep:
            return result + lib.memory_usage_of_objects(self._ndarray)
        return result

    @doc(ExtensionArray.searchsorted)
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        if self._hasna:
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        return super().searchsorted(value=value, side=side, sorter=sorter)

    def _cmp_method(self, other, op):
        from pandas.arrays import BooleanArray

        if isinstance(other, StringArray):
            other = other._ndarray

        mask = isna(self) | isna(other)
        valid = ~mask

        if not lib.is_scalar(other):
            if len(other) != len(self):
                # prevent improper broadcasting when other is 2D
                raise ValueError(
                    f"Lengths of operands do not match: {len(self)} != {len(other)}"
                )

            # for array-likes, first filter out NAs before converting to numpy
            if not is_array_like(other):
                other = np.asarray(other)
            other = other[valid]

        if op.__name__ in ops.ARITHMETIC_BINOPS:
            result = np.empty_like(self._ndarray, dtype="object")
            result[mask] = self.dtype.na_value
            result[valid] = op(self._ndarray[valid], other)
            return self._from_backing_data(result)
        else:
            # logical
            result = np.zeros(len(self._ndarray), dtype="bool")
            result[valid] = op(self._ndarray[valid], other)
            res_arr = BooleanArray(result, mask)
            if self.dtype.na_value is np.nan:
                if op == operator.ne:
                    return res_arr.to_numpy(np.bool_, na_value=True)
                else:
                    return res_arr.to_numpy(np.bool_, na_value=False)
            return res_arr

    _arith_method = _cmp_method


class StringArrayNumpySemantics(StringArray):
    _storage = "python"
    _na_value = np.nan

    def _validate(self) -> None:
        """Validate that we only store NaN or strings."""
        if len(self._ndarray) and not lib.is_string_array(self._ndarray, skipna=True):
            raise ValueError(
                "StringArrayNumpySemantics requires a sequence of strings or NaN"
            )
        if self._ndarray.dtype != "object":
            raise ValueError(
                "StringArrayNumpySemantics requires a sequence of strings or NaN. Got "
                f"'{self._ndarray.dtype}' dtype instead."
            )
        # TODO validate or force NA/None to NaN

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        if dtype is None:
            dtype = StringDtype(storage="python", na_value=np.nan)
        return super()._from_sequence(scalars, dtype=dtype, copy=copy)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\descriptor_props.py ===
# orm/descriptor_props.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Descriptor properties are more "auxiliary" properties
that exist as configurational elements, but don't participate
as actively in the load/persist ORM loop.

"""
from __future__ import annotations

from dataclasses import is_dataclass
import inspect
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from . import attributes
from . import util as orm_util
from .base import _DeclarativeMapped
from .base import LoaderCallableStatus
from .base import Mapped
from .base import PassiveFlag
from .base import SQLORMOperations
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .util import _none_set
from .util import de_stringify_annotation
from .. import event
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import util
from ..sql import expression
from ..sql import operators
from ..sql.elements import BindParameter
from ..util.typing import get_args
from ..util.typing import is_fwd_ref
from ..util.typing import is_pep593


if typing.TYPE_CHECKING:
    from ._typing import _InstanceDict
    from ._typing import _RegistryType
    from .attributes import History
    from .attributes import InstrumentedAttribute
    from .attributes import QueryableAttribute
    from .context import ORMCompileState
    from .decl_base import _ClassScanMapperConfig
    from .mapper import Mapper
    from .properties import ColumnProperty
    from .properties import MappedColumn
    from .state import InstanceState
    from ..engine.base import Connection
    from ..engine.row import Row
    from ..sql._typing import _DMLColumnArgument
    from ..sql._typing import _InfoType
    from ..sql.elements import ClauseList
    from ..sql.elements import ColumnElement
    from ..sql.operators import OperatorType
    from ..sql.schema import Column
    from ..sql.selectable import Select
    from ..util.typing import _AnnotationScanType
    from ..util.typing import CallableReference
    from ..util.typing import DescriptorReference
    from ..util.typing import RODescriptorReference

_T = TypeVar("_T", bound=Any)
_PT = TypeVar("_PT", bound=Any)


class DescriptorProperty(MapperProperty[_T]):
    """:class:`.MapperProperty` which proxies access to a
    user-defined descriptor."""

    doc: Optional[str] = None

    uses_objects = False
    _links_to_entity = False

    descriptor: DescriptorReference[Any]

    def get_history(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
    ) -> History:
        raise NotImplementedError()

    def instrument_class(self, mapper: Mapper[Any]) -> None:
        prop = self

        class _ProxyImpl(attributes.AttributeImpl):
            accepts_scalar_loader = False
            load_on_unexpire = True
            collection = False

            @property
            def uses_objects(self) -> bool:  # type: ignore
                return prop.uses_objects

            def __init__(self, key: str):
                self.key = key

            def get_history(
                self,
                state: InstanceState[Any],
                dict_: _InstanceDict,
                passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
            ) -> History:
                return prop.get_history(state, dict_, passive)

        if self.descriptor is None:
            desc = getattr(mapper.class_, self.key, None)
            if mapper._is_userland_descriptor(self.key, desc):
                self.descriptor = desc

        if self.descriptor is None:

            def fset(obj: Any, value: Any) -> None:
                setattr(obj, self.name, value)

            def fdel(obj: Any) -> None:
                delattr(obj, self.name)

            def fget(obj: Any) -> Any:
                return getattr(obj, self.name)

            self.descriptor = property(fget=fget, fset=fset, fdel=fdel)

        proxy_attr = attributes.create_proxied_attribute(self.descriptor)(
            self.parent.class_,
            self.key,
            self.descriptor,
            lambda: self._comparator_factory(mapper),
            doc=self.doc,
            original_property=self,
        )
        proxy_attr.impl = _ProxyImpl(self.key)
        mapper.class_manager.instrument_attribute(self.key, proxy_attr)


_CompositeAttrType = Union[
    str,
    "Column[_T]",
    "MappedColumn[_T]",
    "InstrumentedAttribute[_T]",
    "Mapped[_T]",
]


_CC = TypeVar("_CC", bound=Any)


_composite_getters: weakref.WeakKeyDictionary[
    Type[Any], Callable[[Any], Tuple[Any, ...]]
] = weakref.WeakKeyDictionary()


class CompositeProperty(
    _MapsColumns[_CC], _IntrospectsAnnotations, DescriptorProperty[_CC]
):
    """Defines a "composite" mapped attribute, representing a collection
    of columns as one attribute.

    :class:`.CompositeProperty` is constructed using the :func:`.composite`
    function.

    .. seealso::

        :ref:`mapper_composite`

    """

    composite_class: Union[Type[_CC], Callable[..., _CC]]
    attrs: Tuple[_CompositeAttrType[Any], ...]

    _generated_composite_accessor: CallableReference[
        Optional[Callable[[_CC], Tuple[Any, ...]]]
    ]

    comparator_factory: Type[Comparator[_CC]]

    def __init__(
        self,
        _class_or_attr: Union[
            None, Type[_CC], Callable[..., _CC], _CompositeAttrType[Any]
        ] = None,
        *attrs: _CompositeAttrType[Any],
        attribute_options: Optional[_AttributeOptions] = None,
        active_history: bool = False,
        deferred: bool = False,
        group: Optional[str] = None,
        comparator_factory: Optional[Type[Comparator[_CC]]] = None,
        info: Optional[_InfoType] = None,
        **kwargs: Any,
    ):
        super().__init__(attribute_options=attribute_options)

        if isinstance(_class_or_attr, (Mapped, str, sql.ColumnElement)):
            self.attrs = (_class_or_attr,) + attrs
            # will initialize within declarative_scan
            self.composite_class = None  # type: ignore
        else:
            self.composite_class = _class_or_attr  # type: ignore
            self.attrs = attrs

        self.active_history = active_history
        self.deferred = deferred
        self.group = group
        self.comparator_factory = (
            comparator_factory
            if comparator_factory is not None
            else self.__class__.Comparator
        )
        self._generated_composite_accessor = None
        if info is not None:
            self.info.update(info)

        util.set_creation_order(self)
        self._create_descriptor()
        self._init_accessor()

    def instrument_class(self, mapper: Mapper[Any]) -> None:
        super().instrument_class(mapper)
        self._setup_event_handlers()

    def _composite_values_from_instance(self, value: _CC) -> Tuple[Any, ...]:
        if self._generated_composite_accessor:
            return self._generated_composite_accessor(value)
        else:
            try:
                accessor = value.__composite_values__
            except AttributeError as ae:
                raise sa_exc.InvalidRequestError(
                    f"Composite class {self.composite_class.__name__} is not "
                    f"a dataclass and does not define a __composite_values__()"
                    " method; can't get state"
                ) from ae
            else:
                return accessor()  # type: ignore

    def do_init(self) -> None:
        """Initialization which occurs after the :class:`.Composite`
        has been associated with its parent mapper.

        """
        self._setup_arguments_on_columns()

    _COMPOSITE_FGET = object()

    def _create_descriptor(self) -> None:
        """Create the Python descriptor that will serve as
        the access point on instances of the mapped class.

        """

        def fget(instance: Any) -> Any:
            dict_ = attributes.instance_dict(instance)
            state = attributes.instance_state(instance)

            if self.key not in dict_:
                # key not present.  Iterate through related
                # attributes, retrieve their values.  This
                # ensures they all load.
                values = [
                    getattr(instance, key) for key in self._attribute_keys
                ]

                # current expected behavior here is that the composite is
                # created on access if the object is persistent or if
                # col attributes have non-None.  This would be better
                # if the composite were created unconditionally,
                # but that would be a behavioral change.
                if self.key not in dict_ and (
                    state.key is not None or not _none_set.issuperset(values)
                ):
                    dict_[self.key] = self.composite_class(*values)
                    state.manager.dispatch.refresh(
                        state, self._COMPOSITE_FGET, [self.key]
                    )

            return dict_.get(self.key, None)

        def fset(instance: Any, value: Any) -> None:
            dict_ = attributes.instance_dict(instance)
            state = attributes.instance_state(instance)
            attr = state.manager[self.key]

            if attr.dispatch._active_history:
                previous = fget(instance)
            else:
                previous = dict_.get(self.key, LoaderCallableStatus.NO_VALUE)

            for fn in attr.dispatch.set:
                value = fn(state, value, previous, attr.impl)
            dict_[self.key] = value
            if value is None:
                for key in self._attribute_keys:
                    setattr(instance, key, None)
            else:
                for key, value in zip(
                    self._attribute_keys,
                    self._composite_values_from_instance(value),
                ):
                    setattr(instance, key, value)

        def fdel(instance: Any) -> None:
            state = attributes.instance_state(instance)
            dict_ = attributes.instance_dict(instance)
            attr = state.manager[self.key]

            if attr.dispatch._active_history:
                previous = fget(instance)
                dict_.pop(self.key, None)
            else:
                previous = dict_.pop(self.key, LoaderCallableStatus.NO_VALUE)

            attr = state.manager[self.key]
            attr.dispatch.remove(state, previous, attr.impl)
            for key in self._attribute_keys:
                setattr(instance, key, None)

        self.descriptor = property(fget, fset, fdel)

    @util.preload_module("sqlalchemy.orm.properties")
    def declarative_scan(
        self,
        decl_scan: _ClassScanMapperConfig,
        registry: _RegistryType,
        cls: Type[Any],
        originating_module: Optional[str],
        key: str,
        mapped_container: Optional[Type[Mapped[Any]]],
        annotation: Optional[_AnnotationScanType],
        extracted_mapped_annotation: Optional[_AnnotationScanType],
        is_dataclass_field: bool,
    ) -> None:
        MappedColumn = util.preloaded.orm_properties.MappedColumn
        if (
            self.composite_class is None
            and extracted_mapped_annotation is None
        ):
            self._raise_for_required(key, cls)
        argument = extracted_mapped_annotation

        if is_pep593(argument):
            argument = get_args(argument)[0]

        if argument and self.composite_class is None:
            if isinstance(argument, str) or is_fwd_ref(
                argument, check_generic=True
            ):
                if originating_module is None:
                    str_arg = (
                        argument.__forward_arg__
                        if hasattr(argument, "__forward_arg__")
                        else str(argument)
                    )
                    raise sa_exc.ArgumentError(
                        f"Can't use forward ref {argument} for composite "
                        f"class argument; set up the type as Mapped[{str_arg}]"
                    )
                argument = de_stringify_annotation(
                    cls, argument, originating_module, include_generic=True
                )

            self.composite_class = argument

        if is_dataclass(self.composite_class):
            self._setup_for_dataclass(registry, cls, originating_module, key)
        else:
            for attr in self.attrs:
                if (
                    isinstance(attr, (MappedColumn, schema.Column))
                    and attr.name is None
                ):
                    raise sa_exc.ArgumentError(
                        "Composite class column arguments must be named "
                        "unless a dataclass is used"
                    )
        self._init_accessor()

    def _init_accessor(self) -> None:
        if is_dataclass(self.composite_class) and not hasattr(
            self.composite_class, "__composite_values__"
        ):
            insp = inspect.signature(self.composite_class)
            getter = operator.attrgetter(
                *[p.name for p in insp.parameters.values()]
            )
            if len(insp.parameters) == 1:
                self._generated_composite_accessor = lambda obj: (getter(obj),)
            else:
                self._generated_composite_accessor = getter

        if (
            self.composite_class is not None
            and isinstance(self.composite_class, type)
            and self.composite_class not in _composite_getters
        ):
            if self._generated_composite_accessor is not None:
                _composite_getters[self.composite_class] = (
                    self._generated_composite_accessor
                )
            elif hasattr(self.composite_class, "__composite_values__"):
                _composite_getters[self.composite_class] = (
                    lambda obj: obj.__composite_values__()
                )

    @util.preload_module("sqlalchemy.orm.properties")
    @util.preload_module("sqlalchemy.orm.decl_base")
    def _setup_for_dataclass(
        self,
        registry: _RegistryType,
        cls: Type[Any],
        originating_module: Optional[str],
        key: str,
    ) -> None:
        MappedColumn = util.preloaded.orm_properties.MappedColumn

        decl_base = util.preloaded.orm_decl_base

        insp = inspect.signature(self.composite_class)
        for param, attr in itertools.zip_longest(
            insp.parameters.values(), self.attrs
        ):
            if param is None:
                raise sa_exc.ArgumentError(
                    f"number of composite attributes "
                    f"{len(self.attrs)} exceeds "
                    f"that of the number of attributes in class "
                    f"{self.composite_class.__name__} {len(insp.parameters)}"
                )
            if attr is None:
                # fill in missing attr spots with empty MappedColumn
                attr = MappedColumn()
                self.attrs += (attr,)

            if isinstance(attr, MappedColumn):
                attr.declarative_scan_for_composite(
                    registry,
                    cls,
                    originating_module,
                    key,
                    param.name,
                    param.annotation,
                )
            elif isinstance(attr, schema.Column):
                decl_base._undefer_column_name(param.name, attr)

    @util.memoized_property
    def _comparable_elements(self) -> Sequence[QueryableAttribute[Any]]:
        return [getattr(self.parent.class_, prop.key) for prop in self.props]

    @util.memoized_property
    @util.preload_module("orm.properties")
    def props(self) -> Sequence[MapperProperty[Any]]:
        props = []
        MappedColumn = util.preloaded.orm_properties.MappedColumn

        for attr in self.attrs:
            if isinstance(attr, str):
                prop = self.parent.get_property(attr, _configure_mappers=False)
            elif isinstance(attr, schema.Column):
                prop = self.parent._columntoproperty[attr]
            elif isinstance(attr, MappedColumn):
                prop = self.parent._columntoproperty[attr.column]
            elif isinstance(attr, attributes.InstrumentedAttribute):
                prop = attr.property
            else:
                prop = None

            if not isinstance(prop, MapperProperty):
                raise sa_exc.ArgumentError(
                    "Composite expects Column objects or mapped "
                    f"attributes/attribute names as arguments, got: {attr!r}"
                )

            props.append(prop)
        return props

    @util.non_memoized_property
    @util.preload_module("orm.properties")
    def columns(self) -> Sequence[Column[Any]]:
        MappedColumn = util.preloaded.orm_properties.MappedColumn
        return [
            a.column if isinstance(a, MappedColumn) else a
            for a in self.attrs
            if isinstance(a, (schema.Column, MappedColumn))
        ]

    @property
    def mapper_property_to_assign(self) -> Optional[MapperProperty[_CC]]:
        return self

    @property
    def columns_to_assign(self) -> List[Tuple[schema.Column[Any], int]]:
        return [(c, 0) for c in self.columns if c.table is None]

    @util.preload_module("orm.properties")
    def _setup_arguments_on_columns(self) -> None:
        """Propagate configuration arguments made on this composite
        to the target columns, for those that apply.

        """
        ColumnProperty = util.preloaded.orm_properties.ColumnProperty

        for prop in self.props:
            if not isinstance(prop, ColumnProperty):
                continue
            else:
                cprop = prop

            cprop.active_history = self.active_history
            if self.deferred:
                cprop.deferred = self.deferred
                cprop.strategy_key = (("deferred", True), ("instrument", True))
            cprop.group = self.group

    def _setup_event_handlers(self) -> None:
        """Establish events that populate/expire the composite attribute."""

        def load_handler(
            state: InstanceState[Any], context: ORMCompileState
        ) -> None:
            _load_refresh_handler(state, context, None, is_refresh=False)

        def refresh_handler(
            state: InstanceState[Any],
            context: ORMCompileState,
            to_load: Optional[Sequence[str]],
        ) -> None:
            # note this corresponds to sqlalchemy.ext.mutable load_attrs()

            if not to_load or (
                {self.key}.union(self._attribute_keys)
            ).intersection(to_load):
                _load_refresh_handler(state, context, to_load, is_refresh=True)

        def _load_refresh_handler(
            state: InstanceState[Any],
            context: ORMCompileState,
            to_load: Optional[Sequence[str]],
            is_refresh: bool,
        ) -> None:
            dict_ = state.dict

            # if context indicates we are coming from the
            # fget() handler, this already set the value; skip the
            # handler here. (other handlers like mutablecomposite will still
            # want to catch it)
            # there's an insufficiency here in that the fget() handler
            # really should not be using the refresh event and there should
            # be some other event that mutablecomposite can subscribe
            # towards for this.

            if (
                not is_refresh or context is self._COMPOSITE_FGET
            ) and self.key in dict_:
                return

            # if column elements aren't loaded, skip.
            # __get__() will initiate a load for those
            # columns
            for k in self._attribute_keys:
                if k not in dict_:
                    return

            dict_[self.key] = self.composite_class(
                *[state.dict[key] for key in self._attribute_keys]
            )

        def expire_handler(
            state: InstanceState[Any], keys: Optional[Sequence[str]]
        ) -> None:
            if keys is None or set(self._attribute_keys).intersection(keys):
                state.dict.pop(self.key, None)

        def insert_update_handler(
            mapper: Mapper[Any],
            connection: Connection,
            state: InstanceState[Any],
        ) -> None:
            """After an insert or update, some columns may be expired due
            to server side defaults, or re-populated due to client side
            defaults.  Pop out the composite value here so that it
            recreates.

            """

            state.dict.pop(self.key, None)

        event.listen(
            self.parent, "after_insert", insert_update_handler, raw=True
        )
        event.listen(
            self.parent, "after_update", insert_update_handler, raw=True
        )
        event.listen(
            self.parent, "load", load_handler, raw=True, propagate=True
        )
        event.listen(
            self.parent, "refresh", refresh_handler, raw=True, propagate=True
        )
        event.listen(
            self.parent, "expire", expire_handler, raw=True, propagate=True
        )

        proxy_attr = self.parent.class_manager[self.key]
        proxy_attr.impl.dispatch = proxy_attr.dispatch  # type: ignore
        proxy_attr.impl.dispatch._active_history = self.active_history

        # TODO: need a deserialize hook here

    @util.memoized_property
    def _attribute_keys(self) -> Sequence[str]:
        return [prop.key for prop in self.props]

    def _populate_composite_bulk_save_mappings_fn(
        self,
    ) -> Callable[[Dict[str, Any]], None]:
        if self._generated_composite_accessor:
            get_values = self._generated_composite_accessor
        else:

            def get_values(val: Any) -> Tuple[Any]:
                return val.__composite_values__()  # type: ignore

        attrs = [prop.key for prop in self.props]

        def populate(dest_dict: Dict[str, Any]) -> None:
            dest_dict.update(
                {
                    key: val
                    for key, val in zip(
                        attrs, get_values(dest_dict.pop(self.key))
                    )
                }
            )

        return populate

    def get_history(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
    ) -> History:
        """Provided for userland code that uses attributes.get_history()."""

        added: List[Any] = []
        deleted: List[Any] = []

        has_history = False
        for prop in self.props:
            key = prop.key
            hist = state.manager[key].impl.get_history(state, dict_)
            if hist.has_changes():
                has_history = True

            non_deleted = hist.non_deleted()
            if non_deleted:
                added.extend(non_deleted)
            else:
                added.append(None)
            if hist.deleted:
                deleted.extend(hist.deleted)
            else:
                deleted.append(None)

        if has_history:
            return attributes.History(
                [self.composite_class(*added)],
                (),
                [self.composite_class(*deleted)],
            )
        else:
            return attributes.History((), [self.composite_class(*added)], ())

    def _comparator_factory(
        self, mapper: Mapper[Any]
    ) -> Composite.Comparator[_CC]:
        return self.comparator_factory(self, mapper)

    class CompositeBundle(orm_util.Bundle[_T]):
        def __init__(
            self,
            property_: Composite[_T],
            expr: ClauseList,
        ):
            self.property = property_
            super().__init__(property_.key, *expr)

        def create_row_processor(
            self,
            query: Select[Any],
            procs: Sequence[Callable[[Row[Any]], Any]],
            labels: Sequence[str],
        ) -> Callable[[Row[Any]], Any]:
            def proc(row: Row[Any]) -> Any:
                return self.property.composite_class(
                    *[proc(row) for proc in procs]
                )

            return proc

    class Comparator(PropComparator[_PT]):
        """Produce boolean, comparison, and other operators for
        :class:`.Composite` attributes.

        See the example in :ref:`composite_operations` for an overview
        of usage , as well as the documentation for :class:`.PropComparator`.

        .. seealso::

            :class:`.PropComparator`

            :class:`.ColumnOperators`

            :ref:`types_operators`

            :attr:`.TypeEngine.comparator_factory`

        """

        # https://github.com/python/mypy/issues/4266
        __hash__ = None  # type: ignore

        prop: RODescriptorReference[Composite[_PT]]

        @util.memoized_property
        def clauses(self) -> ClauseList:
            return expression.ClauseList(
                group=False, *self._comparable_elements
            )

        def __clause_element__(self) -> CompositeProperty.CompositeBundle[_PT]:
            return self.expression

        @util.memoized_property
        def expression(self) -> CompositeProperty.CompositeBundle[_PT]:
            clauses = self.clauses._annotate(
                {
                    "parententity": self._parententity,
                    "parentmapper": self._parententity,
                    "proxy_key": self.prop.key,
                }
            )
            return CompositeProperty.CompositeBundle(self.prop, clauses)

        def _bulk_update_tuples(
            self, value: Any
        ) -> Sequence[Tuple[_DMLColumnArgument, Any]]:
            if isinstance(value, BindParameter):
                value = value.value

            values: Sequence[Any]

            if value is None:
                values = [None for key in self.prop._attribute_keys]
            elif isinstance(self.prop.composite_class, type) and isinstance(
                value, self.prop.composite_class
            ):
                values = self.prop._composite_values_from_instance(
                    value  # type: ignore[arg-type]
                )
            else:
                raise sa_exc.ArgumentError(
                    "Can't UPDATE composite attribute %s to %r"
                    % (self.prop, value)
                )

            return list(zip(self._comparable_elements, values))

        @util.memoized_property
        def _comparable_elements(self) -> Sequence[QueryableAttribute[Any]]:
            if self._adapt_to_entity:
                return [
                    getattr(self._adapt_to_entity.entity, prop.key)
                    for prop in self.prop._comparable_elements
                ]
            else:
                return self.prop._comparable_elements

        def __eq__(self, other: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
            return self._compare(operators.eq, other)

        def __ne__(self, other: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
            return self._compare(operators.ne, other)

        def __lt__(self, other: Any) -> ColumnElement[bool]:
            return self._compare(operators.lt, other)

        def __gt__(self, other: Any) -> ColumnElement[bool]:
            return self._compare(operators.gt, other)

        def __le__(self, other: Any) -> ColumnElement[bool]:
            return self._compare(operators.le, other)

        def __ge__(self, other: Any) -> ColumnElement[bool]:
            return self._compare(operators.ge, other)

        # what might be interesting would be if we create
        # an instance of the composite class itself with
        # the columns as data members, then use "hybrid style" comparison
        # to create these comparisons.  then your Point.__eq__() method could
        # be where comparison behavior is defined for SQL also.   Likely
        # not a good choice for default behavior though, not clear how it would
        # work w/ dataclasses, etc.  also no demand for any of this anyway.
        def _compare(
            self, operator: OperatorType, other: Any
        ) -> ColumnElement[bool]:
            values: Sequence[Any]
            if other is None:
                values = [None] * len(self.prop._comparable_elements)
            else:
                values = self.prop._composite_values_from_instance(other)
            comparisons = [
                operator(a, b)
                for a, b in zip(self.prop._comparable_elements, values)
            ]
            if self._adapt_to_entity:
                assert self.adapter is not None
                comparisons = [self.adapter(x) for x in comparisons]
            return sql.and_(*comparisons)

    def __str__(self) -> str:
        return str(self.parent.class_.__name__) + "." + self.key


class Composite(CompositeProperty[_T], _DeclarativeMapped[_T]):
    """Declarative-compatible front-end for the :class:`.CompositeProperty`
    class.

    Public constructor is the :func:`_orm.composite` function.

    .. versionchanged:: 2.0 Added :class:`_orm.Composite` as a Declarative
       compatible subclass of :class:`_orm.CompositeProperty`.

    .. seealso::

        :ref:`mapper_composite`

    """

    inherit_cache = True
    """:meta private:"""


class ConcreteInheritedProperty(DescriptorProperty[_T]):
    """A 'do nothing' :class:`.MapperProperty` that disables
    an attribute on a concrete subclass that is only present
    on the inherited mapper, not the concrete classes' mapper.

    Cases where this occurs include:

    * When the superclass mapper is mapped against a
      "polymorphic union", which includes all attributes from
      all subclasses.
    * When a relationship() is configured on an inherited mapper,
      but not on the subclass mapper.  Concrete mappers require
      that relationship() is configured explicitly on each
      subclass.

    """

    def _comparator_factory(
        self, mapper: Mapper[Any]
    ) -> Type[PropComparator[_T]]:
        comparator_callable = None

        for m in self.parent.iterate_to_root():
            p = m._props[self.key]
            if getattr(p, "comparator_factory", None) is not None:
                comparator_callable = p.comparator_factory
                break
        assert comparator_callable is not None
        return comparator_callable(p, mapper)  # type: ignore

    def __init__(self) -> None:
        super().__init__()

        def warn() -> NoReturn:
            raise AttributeError(
                "Concrete %s does not implement "
                "attribute %r at the instance level.  Add "
                "this property explicitly to %s."
                % (self.parent, self.key, self.parent)
            )

        class NoninheritedConcreteProp:
            def __set__(s: Any, obj: Any, value: Any) -> NoReturn:
                warn()

            def __delete__(s: Any, obj: Any) -> NoReturn:
                warn()

            def __get__(s: Any, obj: Any, owner: Any) -> Any:
                if obj is None:
                    return self.descriptor
                warn()

        self.descriptor = NoninheritedConcreteProp()


class SynonymProperty(DescriptorProperty[_T]):
    """Denote an attribute name as a synonym to a mapped property,
    in that the attribute will mirror the value and expression behavior
    of another attribute.

    :class:`.Synonym` is constructed using the :func:`_orm.synonym`
    function.

    .. seealso::

        :ref:`synonyms` - Overview of synonyms

    """

    comparator_factory: Optional[Type[PropComparator[_T]]]

    def __init__(
        self,
        name: str,
        map_column: Optional[bool] = None,
        descriptor: Optional[Any] = None,
        comparator_factory: Optional[Type[PropComparator[_T]]] = None,
        attribute_options: Optional[_AttributeOptions] = None,
        info: Optional[_InfoType] = None,
        doc: Optional[str] = None,
    ):
        super().__init__(attribute_options=attribute_options)

        self.name = name
        self.map_column = map_column
        self.descriptor = descriptor
        self.comparator_factory = comparator_factory
        if doc:
            self.doc = doc
        elif descriptor and descriptor.__doc__:
            self.doc = descriptor.__doc__
        else:
            self.doc = None
        if info:
            self.info.update(info)

        util.set_creation_order(self)

    if not TYPE_CHECKING:

        @property
        def uses_objects(self) -> bool:
            return getattr(self.parent.class_, self.name).impl.uses_objects

    # TODO: when initialized, check _proxied_object,
    # emit a warning if its not a column-based property

    @util.memoized_property
    def _proxied_object(
        self,
    ) -> Union[MapperProperty[_T], SQLORMOperations[_T]]:
        attr = getattr(self.parent.class_, self.name)
        if not hasattr(attr, "property") or not isinstance(
            attr.property, MapperProperty
        ):
            # attribute is a non-MapperProprerty proxy such as
            # hybrid or association proxy
            if isinstance(attr, attributes.QueryableAttribute):
                return attr.comparator
            elif isinstance(attr, SQLORMOperations):
                # assocaition proxy comes here
                return attr

            raise sa_exc.InvalidRequestError(
                """synonym() attribute "%s.%s" only supports """
                """ORM mapped attributes, got %r"""
                % (self.parent.class_.__name__, self.name, attr)
            )
        return attr.property

    def _comparator_factory(self, mapper: Mapper[Any]) -> SQLORMOperations[_T]:
        prop = self._proxied_object

        if isinstance(prop, MapperProperty):
            if self.comparator_factory:
                comp = self.comparator_factory(prop, mapper)
            else:
                comp = prop.comparator_factory(prop, mapper)
            return comp
        else:
            return prop

    def get_history(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
    ) -> History:
        attr: QueryableAttribute[Any] = getattr(self.parent.class_, self.name)
        return attr.impl.get_history(state, dict_, passive=passive)

    @util.preload_module("sqlalchemy.orm.properties")
    def set_parent(self, parent: Mapper[Any], init: bool) -> None:
        properties = util.preloaded.orm_properties

        if self.map_column:
            # implement the 'map_column' option.
            if self.key not in parent.persist_selectable.c:
                raise sa_exc.ArgumentError(
                    "Can't compile synonym '%s': no column on table "
                    "'%s' named '%s'"
                    % (
                        self.name,
                        parent.persist_selectable.description,
                        self.key,
                    )
                )
            elif (
                parent.persist_selectable.c[self.key]
                in parent._columntoproperty
                and parent._columntoproperty[
                    parent.persist_selectable.c[self.key]
                ].key
                == self.name
            ):
                raise sa_exc.ArgumentError(
                    "Can't call map_column=True for synonym %r=%r, "
                    "a ColumnProperty already exists keyed to the name "
                    "%r for column %r"
                    % (self.key, self.name, self.name, self.key)
                )
            p: ColumnProperty[Any] = properties.ColumnProperty(
                parent.persist_selectable.c[self.key]
            )
            parent._configure_property(self.name, p, init=init, setparent=True)
            p._mapped_by_synonym = self.key

        self.parent = parent


class Synonym(SynonymProperty[_T], _DeclarativeMapped[_T]):
    """Declarative front-end for the :class:`.SynonymProperty` class.

    Public constructor is the :func:`_orm.synonym` function.

    .. versionchanged:: 2.0 Added :class:`_orm.Synonym` as a Declarative
       compatible subclass for :class:`_orm.SynonymProperty`

    .. seealso::

        :ref:`synonyms` - Overview of synonyms

    """

    inherit_cache = True
    """:meta private:"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\packages\six.py ===
# Copyright (c) 2010-2020 Benjamin Peterson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Utilities for writing code that runs on Python 2 and 3"""

from __future__ import absolute_import

import functools
import itertools
import operator
import sys
import types

__author__ = "Benjamin Peterson <benjamin@python.org>"
__version__ = "1.16.0"


# Useful for very coarse version differentiation.
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
PY34 = sys.version_info[0:2] >= (3, 4)

if PY3:
    string_types = (str,)
    integer_types = (int,)
    class_types = (type,)
    text_type = str
    binary_type = bytes

    MAXSIZE = sys.maxsize
else:
    string_types = (basestring,)
    integer_types = (int, long)
    class_types = (type, types.ClassType)
    text_type = unicode
    binary_type = str

    if sys.platform.startswith("java"):
        # Jython always uses 32 bits.
        MAXSIZE = int((1 << 31) - 1)
    else:
        # It's possible to have sizeof(long) != sizeof(Py_ssize_t).
        class X(object):
            def __len__(self):
                return 1 << 31

        try:
            len(X())
        except OverflowError:
            # 32-bit
            MAXSIZE = int((1 << 31) - 1)
        else:
            # 64-bit
            MAXSIZE = int((1 << 63) - 1)
        del X

if PY34:
    from importlib.util import spec_from_loader
else:
    spec_from_loader = None


def _add_doc(func, doc):
    """Add documentation to a function."""
    func.__doc__ = doc


def _import_module(name):
    """Import module, returning the module after the last dot."""
    __import__(name)
    return sys.modules[name]


class _LazyDescr(object):
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, tp):
        result = self._resolve()
        setattr(obj, self.name, result)  # Invokes __set__.
        try:
            # This is a bit ugly, but it avoids running this again by
            # removing this descriptor.
            delattr(obj.__class__, self.name)
        except AttributeError:
            pass
        return result


class MovedModule(_LazyDescr):
    def __init__(self, name, old, new=None):
        super(MovedModule, self).__init__(name)
        if PY3:
            if new is None:
                new = name
            self.mod = new
        else:
            self.mod = old

    def _resolve(self):
        return _import_module(self.mod)

    def __getattr__(self, attr):
        _module = self._resolve()
        value = getattr(_module, attr)
        setattr(self, attr, value)
        return value


class _LazyModule(types.ModuleType):
    def __init__(self, name):
        super(_LazyModule, self).__init__(name)
        self.__doc__ = self.__class__.__doc__

    def __dir__(self):
        attrs = ["__doc__", "__name__"]
        attrs += [attr.name for attr in self._moved_attributes]
        return attrs

    # Subclasses should override this
    _moved_attributes = []


class MovedAttribute(_LazyDescr):
    def __init__(self, name, old_mod, new_mod, old_attr=None, new_attr=None):
        super(MovedAttribute, self).__init__(name)
        if PY3:
            if new_mod is None:
                new_mod = name
            self.mod = new_mod
            if new_attr is None:
                if old_attr is None:
                    new_attr = name
                else:
                    new_attr = old_attr
            self.attr = new_attr
        else:
            self.mod = old_mod
            if old_attr is None:
                old_attr = name
            self.attr = old_attr

    def _resolve(self):
        module = _import_module(self.mod)
        return getattr(module, self.attr)


class _SixMetaPathImporter(object):

    """
    A meta path importer to import six.moves and its submodules.

    This class implements a PEP302 finder and loader. It should be compatible
    with Python 2.5 and all existing versions of Python3
    """

    def __init__(self, six_module_name):
        self.name = six_module_name
        self.known_modules = {}

    def _add_module(self, mod, *fullnames):
        for fullname in fullnames:
            self.known_modules[self.name + "." + fullname] = mod

    def _get_module(self, fullname):
        return self.known_modules[self.name + "." + fullname]

    def find_module(self, fullname, path=None):
        if fullname in self.known_modules:
            return self
        return None

    def find_spec(self, fullname, path, target=None):
        if fullname in self.known_modules:
            return spec_from_loader(fullname, self)
        return None

    def __get_module(self, fullname):
        try:
            return self.known_modules[fullname]
        except KeyError:
            raise ImportError("This loader does not know module " + fullname)

    def load_module(self, fullname):
        try:
            # in case of a reload
            return sys.modules[fullname]
        except KeyError:
            pass
        mod = self.__get_module(fullname)
        if isinstance(mod, MovedModule):
            mod = mod._resolve()
        else:
            mod.__loader__ = self
        sys.modules[fullname] = mod
        return mod

    def is_package(self, fullname):
        """
        Return true, if the named module is a package.

        We need this method to get correct spec objects with
        Python 3.4 (see PEP451)
        """
        return hasattr(self.__get_module(fullname), "__path__")

    def get_code(self, fullname):
        """Return None

        Required, if is_package is implemented"""
        self.__get_module(fullname)  # eventually raises ImportError
        return None

    get_source = get_code  # same as get_code

    def create_module(self, spec):
        return self.load_module(spec.name)

    def exec_module(self, module):
        pass


_importer = _SixMetaPathImporter(__name__)


class _MovedItems(_LazyModule):

    """Lazy loading of moved objects"""

    __path__ = []  # mark as package


_moved_attributes = [
    MovedAttribute("cStringIO", "cStringIO", "io", "StringIO"),
    MovedAttribute("filter", "itertools", "builtins", "ifilter", "filter"),
    MovedAttribute(
        "filterfalse", "itertools", "itertools", "ifilterfalse", "filterfalse"
    ),
    MovedAttribute("input", "__builtin__", "builtins", "raw_input", "input"),
    MovedAttribute("intern", "__builtin__", "sys"),
    MovedAttribute("map", "itertools", "builtins", "imap", "map"),
    MovedAttribute("getcwd", "os", "os", "getcwdu", "getcwd"),
    MovedAttribute("getcwdb", "os", "os", "getcwd", "getcwdb"),
    MovedAttribute("getoutput", "commands", "subprocess"),
    MovedAttribute("range", "__builtin__", "builtins", "xrange", "range"),
    MovedAttribute(
        "reload_module", "__builtin__", "importlib" if PY34 else "imp", "reload"
    ),
    MovedAttribute("reduce", "__builtin__", "functools"),
    MovedAttribute("shlex_quote", "pipes", "shlex", "quote"),
    MovedAttribute("StringIO", "StringIO", "io"),
    MovedAttribute("UserDict", "UserDict", "collections"),
    MovedAttribute("UserList", "UserList", "collections"),
    MovedAttribute("UserString", "UserString", "collections"),
    MovedAttribute("xrange", "__builtin__", "builtins", "xrange", "range"),
    MovedAttribute("zip", "itertools", "builtins", "izip", "zip"),
    MovedAttribute(
        "zip_longest", "itertools", "itertools", "izip_longest", "zip_longest"
    ),
    MovedModule("builtins", "__builtin__"),
    MovedModule("configparser", "ConfigParser"),
    MovedModule(
        "collections_abc",
        "collections",
        "collections.abc" if sys.version_info >= (3, 3) else "collections",
    ),
    MovedModule("copyreg", "copy_reg"),
    MovedModule("dbm_gnu", "gdbm", "dbm.gnu"),
    MovedModule("dbm_ndbm", "dbm", "dbm.ndbm"),
    MovedModule(
        "_dummy_thread",
        "dummy_thread",
        "_dummy_thread" if sys.version_info < (3, 9) else "_thread",
    ),
    MovedModule("http_cookiejar", "cookielib", "http.cookiejar"),
    MovedModule("http_cookies", "Cookie", "http.cookies"),
    MovedModule("html_entities", "htmlentitydefs", "html.entities"),
    MovedModule("html_parser", "HTMLParser", "html.parser"),
    MovedModule("http_client", "httplib", "http.client"),
    MovedModule("email_mime_base", "email.MIMEBase", "email.mime.base"),
    MovedModule("email_mime_image", "email.MIMEImage", "email.mime.image"),
    MovedModule("email_mime_multipart", "email.MIMEMultipart", "email.mime.multipart"),
    MovedModule(
        "email_mime_nonmultipart", "email.MIMENonMultipart", "email.mime.nonmultipart"
    ),
    MovedModule("email_mime_text", "email.MIMEText", "email.mime.text"),
    MovedModule("BaseHTTPServer", "BaseHTTPServer", "http.server"),
    MovedModule("CGIHTTPServer", "CGIHTTPServer", "http.server"),
    MovedModule("SimpleHTTPServer", "SimpleHTTPServer", "http.server"),
    MovedModule("cPickle", "cPickle", "pickle"),
    MovedModule("queue", "Queue"),
    MovedModule("reprlib", "repr"),
    MovedModule("socketserver", "SocketServer"),
    MovedModule("_thread", "thread", "_thread"),
    MovedModule("tkinter", "Tkinter"),
    MovedModule("tkinter_dialog", "Dialog", "tkinter.dialog"),
    MovedModule("tkinter_filedialog", "FileDialog", "tkinter.filedialog"),
    MovedModule("tkinter_scrolledtext", "ScrolledText", "tkinter.scrolledtext"),
    MovedModule("tkinter_simpledialog", "SimpleDialog", "tkinter.simpledialog"),
    MovedModule("tkinter_tix", "Tix", "tkinter.tix"),
    MovedModule("tkinter_ttk", "ttk", "tkinter.ttk"),
    MovedModule("tkinter_constants", "Tkconstants", "tkinter.constants"),
    MovedModule("tkinter_dnd", "Tkdnd", "tkinter.dnd"),
    MovedModule("tkinter_colorchooser", "tkColorChooser", "tkinter.colorchooser"),
    MovedModule("tkinter_commondialog", "tkCommonDialog", "tkinter.commondialog"),
    MovedModule("tkinter_tkfiledialog", "tkFileDialog", "tkinter.filedialog"),
    MovedModule("tkinter_font", "tkFont", "tkinter.font"),
    MovedModule("tkinter_messagebox", "tkMessageBox", "tkinter.messagebox"),
    MovedModule("tkinter_tksimpledialog", "tkSimpleDialog", "tkinter.simpledialog"),
    MovedModule("urllib_parse", __name__ + ".moves.urllib_parse", "urllib.parse"),
    MovedModule("urllib_error", __name__ + ".moves.urllib_error", "urllib.error"),
    MovedModule("urllib", __name__ + ".moves.urllib", __name__ + ".moves.urllib"),
    MovedModule("urllib_robotparser", "robotparser", "urllib.robotparser"),
    MovedModule("xmlrpc_client", "xmlrpclib", "xmlrpc.client"),
    MovedModule("xmlrpc_server", "SimpleXMLRPCServer", "xmlrpc.server"),
]
# Add windows specific modules.
if sys.platform == "win32":
    _moved_attributes += [
        MovedModule("winreg", "_winreg"),
    ]

for attr in _moved_attributes:
    setattr(_MovedItems, attr.name, attr)
    if isinstance(attr, MovedModule):
        _importer._add_module(attr, "moves." + attr.name)
del attr

_MovedItems._moved_attributes = _moved_attributes

moves = _MovedItems(__name__ + ".moves")
_importer._add_module(moves, "moves")


class Module_six_moves_urllib_parse(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_parse"""


_urllib_parse_moved_attributes = [
    MovedAttribute("ParseResult", "urlparse", "urllib.parse"),
    MovedAttribute("SplitResult", "urlparse", "urllib.parse"),
    MovedAttribute("parse_qs", "urlparse", "urllib.parse"),
    MovedAttribute("parse_qsl", "urlparse", "urllib.parse"),
    MovedAttribute("urldefrag", "urlparse", "urllib.parse"),
    MovedAttribute("urljoin", "urlparse", "urllib.parse"),
    MovedAttribute("urlparse", "urlparse", "urllib.parse"),
    MovedAttribute("urlsplit", "urlparse", "urllib.parse"),
    MovedAttribute("urlunparse", "urlparse", "urllib.parse"),
    MovedAttribute("urlunsplit", "urlparse", "urllib.parse"),
    MovedAttribute("quote", "urllib", "urllib.parse"),
    MovedAttribute("quote_plus", "urllib", "urllib.parse"),
    MovedAttribute("unquote", "urllib", "urllib.parse"),
    MovedAttribute("unquote_plus", "urllib", "urllib.parse"),
    MovedAttribute(
        "unquote_to_bytes", "urllib", "urllib.parse", "unquote", "unquote_to_bytes"
    ),
    MovedAttribute("urlencode", "urllib", "urllib.parse"),
    MovedAttribute("splitquery", "urllib", "urllib.parse"),
    MovedAttribute("splittag", "urllib", "urllib.parse"),
    MovedAttribute("splituser", "urllib", "urllib.parse"),
    MovedAttribute("splitvalue", "urllib", "urllib.parse"),
    MovedAttribute("uses_fragment", "urlparse", "urllib.parse"),
    MovedAttribute("uses_netloc", "urlparse", "urllib.parse"),
    MovedAttribute("uses_params", "urlparse", "urllib.parse"),
    MovedAttribute("uses_query", "urlparse", "urllib.parse"),
    MovedAttribute("uses_relative", "urlparse", "urllib.parse"),
]
for attr in _urllib_parse_moved_attributes:
    setattr(Module_six_moves_urllib_parse, attr.name, attr)
del attr

Module_six_moves_urllib_parse._moved_attributes = _urllib_parse_moved_attributes

_importer._add_module(
    Module_six_moves_urllib_parse(__name__ + ".moves.urllib_parse"),
    "moves.urllib_parse",
    "moves.urllib.parse",
)


class Module_six_moves_urllib_error(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_error"""


_urllib_error_moved_attributes = [
    MovedAttribute("URLError", "urllib2", "urllib.error"),
    MovedAttribute("HTTPError", "urllib2", "urllib.error"),
    MovedAttribute("ContentTooShortError", "urllib", "urllib.error"),
]
for attr in _urllib_error_moved_attributes:
    setattr(Module_six_moves_urllib_error, attr.name, attr)
del attr

Module_six_moves_urllib_error._moved_attributes = _urllib_error_moved_attributes

_importer._add_module(
    Module_six_moves_urllib_error(__name__ + ".moves.urllib.error"),
    "moves.urllib_error",
    "moves.urllib.error",
)


class Module_six_moves_urllib_request(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_request"""


_urllib_request_moved_attributes = [
    MovedAttribute("urlopen", "urllib2", "urllib.request"),
    MovedAttribute("install_opener", "urllib2", "urllib.request"),
    MovedAttribute("build_opener", "urllib2", "urllib.request"),
    MovedAttribute("pathname2url", "urllib", "urllib.request"),
    MovedAttribute("url2pathname", "urllib", "urllib.request"),
    MovedAttribute("getproxies", "urllib", "urllib.request"),
    MovedAttribute("Request", "urllib2", "urllib.request"),
    MovedAttribute("OpenerDirector", "urllib2", "urllib.request"),
    MovedAttribute("HTTPDefaultErrorHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPRedirectHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPCookieProcessor", "urllib2", "urllib.request"),
    MovedAttribute("ProxyHandler", "urllib2", "urllib.request"),
    MovedAttribute("BaseHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPPasswordMgr", "urllib2", "urllib.request"),
    MovedAttribute("HTTPPasswordMgrWithDefaultRealm", "urllib2", "urllib.request"),
    MovedAttribute("AbstractBasicAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPBasicAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("ProxyBasicAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("AbstractDigestAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPDigestAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("ProxyDigestAuthHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPSHandler", "urllib2", "urllib.request"),
    MovedAttribute("FileHandler", "urllib2", "urllib.request"),
    MovedAttribute("FTPHandler", "urllib2", "urllib.request"),
    MovedAttribute("CacheFTPHandler", "urllib2", "urllib.request"),
    MovedAttribute("UnknownHandler", "urllib2", "urllib.request"),
    MovedAttribute("HTTPErrorProcessor", "urllib2", "urllib.request"),
    MovedAttribute("urlretrieve", "urllib", "urllib.request"),
    MovedAttribute("urlcleanup", "urllib", "urllib.request"),
    MovedAttribute("URLopener", "urllib", "urllib.request"),
    MovedAttribute("FancyURLopener", "urllib", "urllib.request"),
    MovedAttribute("proxy_bypass", "urllib", "urllib.request"),
    MovedAttribute("parse_http_list", "urllib2", "urllib.request"),
    MovedAttribute("parse_keqv_list", "urllib2", "urllib.request"),
]
for attr in _urllib_request_moved_attributes:
    setattr(Module_six_moves_urllib_request, attr.name, attr)
del attr

Module_six_moves_urllib_request._moved_attributes = _urllib_request_moved_attributes

_importer._add_module(
    Module_six_moves_urllib_request(__name__ + ".moves.urllib.request"),
    "moves.urllib_request",
    "moves.urllib.request",
)


class Module_six_moves_urllib_response(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_response"""


_urllib_response_moved_attributes = [
    MovedAttribute("addbase", "urllib", "urllib.response"),
    MovedAttribute("addclosehook", "urllib", "urllib.response"),
    MovedAttribute("addinfo", "urllib", "urllib.response"),
    MovedAttribute("addinfourl", "urllib", "urllib.response"),
]
for attr in _urllib_response_moved_attributes:
    setattr(Module_six_moves_urllib_response, attr.name, attr)
del attr

Module_six_moves_urllib_response._moved_attributes = _urllib_response_moved_attributes

_importer._add_module(
    Module_six_moves_urllib_response(__name__ + ".moves.urllib.response"),
    "moves.urllib_response",
    "moves.urllib.response",
)


class Module_six_moves_urllib_robotparser(_LazyModule):

    """Lazy loading of moved objects in six.moves.urllib_robotparser"""


_urllib_robotparser_moved_attributes = [
    MovedAttribute("RobotFileParser", "robotparser", "urllib.robotparser"),
]
for attr in _urllib_robotparser_moved_attributes:
    setattr(Module_six_moves_urllib_robotparser, attr.name, attr)
del attr

Module_six_moves_urllib_robotparser._moved_attributes = (
    _urllib_robotparser_moved_attributes
)

_importer._add_module(
    Module_six_moves_urllib_robotparser(__name__ + ".moves.urllib.robotparser"),
    "moves.urllib_robotparser",
    "moves.urllib.robotparser",
)


class Module_six_moves_urllib(types.ModuleType):

    """Create a six.moves.urllib namespace that resembles the Python 3 namespace"""

    __path__ = []  # mark as package
    parse = _importer._get_module("moves.urllib_parse")
    error = _importer._get_module("moves.urllib_error")
    request = _importer._get_module("moves.urllib_request")
    response = _importer._get_module("moves.urllib_response")
    robotparser = _importer._get_module("moves.urllib_robotparser")

    def __dir__(self):
        return ["parse", "error", "request", "response", "robotparser"]


_importer._add_module(
    Module_six_moves_urllib(__name__ + ".moves.urllib"), "moves.urllib"
)


def add_move(move):
    """Add an item to six.moves."""
    setattr(_MovedItems, move.name, move)


def remove_move(name):
    """Remove item from six.moves."""
    try:
        delattr(_MovedItems, name)
    except AttributeError:
        try:
            del moves.__dict__[name]
        except KeyError:
            raise AttributeError("no such move, %r" % (name,))


if PY3:
    _meth_func = "__func__"
    _meth_self = "__self__"

    _func_closure = "__closure__"
    _func_code = "__code__"
    _func_defaults = "__defaults__"
    _func_globals = "__globals__"
else:
    _meth_func = "im_func"
    _meth_self = "im_self"

    _func_closure = "func_closure"
    _func_code = "func_code"
    _func_defaults = "func_defaults"
    _func_globals = "func_globals"


try:
    advance_iterator = next
except NameError:

    def advance_iterator(it):
        return it.next()


next = advance_iterator


try:
    callable = callable
except NameError:

    def callable(obj):
        return any("__call__" in klass.__dict__ for klass in type(obj).__mro__)


if PY3:

    def get_unbound_function(unbound):
        return unbound

    create_bound_method = types.MethodType

    def create_unbound_method(func, cls):
        return func

    Iterator = object
else:

    def get_unbound_function(unbound):
        return unbound.im_func

    def create_bound_method(func, obj):
        return types.MethodType(func, obj, obj.__class__)

    def create_unbound_method(func, cls):
        return types.MethodType(func, None, cls)

    class Iterator(object):
        def next(self):
            return type(self).__next__(self)

    callable = callable
_add_doc(
    get_unbound_function, """Get the function out of a possibly unbound function"""
)


get_method_function = operator.attrgetter(_meth_func)
get_method_self = operator.attrgetter(_meth_self)
get_function_closure = operator.attrgetter(_func_closure)
get_function_code = operator.attrgetter(_func_code)
get_function_defaults = operator.attrgetter(_func_defaults)
get_function_globals = operator.attrgetter(_func_globals)


if PY3:

    def iterkeys(d, **kw):
        return iter(d.keys(**kw))

    def itervalues(d, **kw):
        return iter(d.values(**kw))

    def iteritems(d, **kw):
        return iter(d.items(**kw))

    def iterlists(d, **kw):
        return iter(d.lists(**kw))

    viewkeys = operator.methodcaller("keys")

    viewvalues = operator.methodcaller("values")

    viewitems = operator.methodcaller("items")
else:

    def iterkeys(d, **kw):
        return d.iterkeys(**kw)

    def itervalues(d, **kw):
        return d.itervalues(**kw)

    def iteritems(d, **kw):
        return d.iteritems(**kw)

    def iterlists(d, **kw):
        return d.iterlists(**kw)

    viewkeys = operator.methodcaller("viewkeys")

    viewvalues = operator.methodcaller("viewvalues")

    viewitems = operator.methodcaller("viewitems")

_add_doc(iterkeys, "Return an iterator over the keys of a dictionary.")
_add_doc(itervalues, "Return an iterator over the values of a dictionary.")
_add_doc(iteritems, "Return an iterator over the (key, value) pairs of a dictionary.")
_add_doc(
    iterlists, "Return an iterator over the (key, [values]) pairs of a dictionary."
)


if PY3:

    def b(s):
        return s.encode("latin-1")

    def u(s):
        return s

    unichr = chr
    import struct

    int2byte = struct.Struct(">B").pack
    del struct
    byte2int = operator.itemgetter(0)
    indexbytes = operator.getitem
    iterbytes = iter
    import io

    StringIO = io.StringIO
    BytesIO = io.BytesIO
    del io
    _assertCountEqual = "assertCountEqual"
    if sys.version_info[1] <= 1:
        _assertRaisesRegex = "assertRaisesRegexp"
        _assertRegex = "assertRegexpMatches"
        _assertNotRegex = "assertNotRegexpMatches"
    else:
        _assertRaisesRegex = "assertRaisesRegex"
        _assertRegex = "assertRegex"
        _assertNotRegex = "assertNotRegex"
else:

    def b(s):
        return s

    # Workaround for standalone backslash

    def u(s):
        return unicode(s.replace(r"\\", r"\\\\"), "unicode_escape")

    unichr = unichr
    int2byte = chr

    def byte2int(bs):
        return ord(bs[0])

    def indexbytes(buf, i):
        return ord(buf[i])

    iterbytes = functools.partial(itertools.imap, ord)
    import StringIO

    StringIO = BytesIO = StringIO.StringIO
    _assertCountEqual = "assertItemsEqual"
    _assertRaisesRegex = "assertRaisesRegexp"
    _assertRegex = "assertRegexpMatches"
    _assertNotRegex = "assertNotRegexpMatches"
_add_doc(b, """Byte literal""")
_add_doc(u, """Text literal""")


def assertCountEqual(self, *args, **kwargs):
    return getattr(self, _assertCountEqual)(*args, **kwargs)


def assertRaisesRegex(self, *args, **kwargs):
    return getattr(self, _assertRaisesRegex)(*args, **kwargs)


def assertRegex(self, *args, **kwargs):
    return getattr(self, _assertRegex)(*args, **kwargs)


def assertNotRegex(self, *args, **kwargs):
    return getattr(self, _assertNotRegex)(*args, **kwargs)


if PY3:
    exec_ = getattr(moves.builtins, "exec")

    def reraise(tp, value, tb=None):
        try:
            if value is None:
                value = tp()
            if value.__traceback__ is not tb:
                raise value.with_traceback(tb)
            raise value
        finally:
            value = None
            tb = None

else:

    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec ("""exec _code_ in _globs_, _locs_""")

    exec_(
        """def reraise(tp, value, tb=None):
    try:
        raise tp, value, tb
    finally:
        tb = None
"""
    )


if sys.version_info[:2] > (3,):
    exec_(
        """def raise_from(value, from_value):
    try:
        raise value from from_value
    finally:
        value = None
"""
    )
else:

    def raise_from(value, from_value):
        raise value


print_ = getattr(moves.builtins, "print", None)
if print_ is None:

    def print_(*args, **kwargs):
        """The new-style print function for Python 2.4 and 2.5."""
        fp = kwargs.pop("file", sys.stdout)
        if fp is None:
            return

        def write(data):
            if not isinstance(data, basestring):
                data = str(data)
            # If the file has an encoding, encode unicode with it.
            if (
                isinstance(fp, file)
                and isinstance(data, unicode)
                and fp.encoding is not None
            ):
                errors = getattr(fp, "errors", None)
                if errors is None:
                    errors = "strict"
                data = data.encode(fp.encoding, errors)
            fp.write(data)

        want_unicode = False
        sep = kwargs.pop("sep", None)
        if sep is not None:
            if isinstance(sep, unicode):
                want_unicode = True
            elif not isinstance(sep, str):
                raise TypeError("sep must be None or a string")
        end = kwargs.pop("end", None)
        if end is not None:
            if isinstance(end, unicode):
                want_unicode = True
            elif not isinstance(end, str):
                raise TypeError("end must be None or a string")
        if kwargs:
            raise TypeError("invalid keyword arguments to print()")
        if not want_unicode:
            for arg in args:
                if isinstance(arg, unicode):
                    want_unicode = True
                    break
        if want_unicode:
            newline = unicode("\n")
            space = unicode(" ")
        else:
            newline = "\n"
            space = " "
        if sep is None:
            sep = space
        if end is None:
            end = newline
        for i, arg in enumerate(args):
            if i:
                write(sep)
            write(arg)
        write(end)


if sys.version_info[:2] < (3, 3):
    _print = print_

    def print_(*args, **kwargs):
        fp = kwargs.get("file", sys.stdout)
        flush = kwargs.pop("flush", False)
        _print(*args, **kwargs)
        if flush and fp is not None:
            fp.flush()


_add_doc(reraise, """Reraise an exception.""")

if sys.version_info[0:2] < (3, 4):
    # This does exactly the same what the :func:`py3:functools.update_wrapper`
    # function does on Python versions after 3.2. It sets the ``__wrapped__``
    # attribute on ``wrapper`` object and it doesn't raise an error if any of
    # the attributes mentioned in ``assigned`` and ``updated`` are missing on
    # ``wrapped`` object.
    def _update_wrapper(
        wrapper,
        wrapped,
        assigned=functools.WRAPPER_ASSIGNMENTS,
        updated=functools.WRAPPER_UPDATES,
    ):
        for attr in assigned:
            try:
                value = getattr(wrapped, attr)
            except AttributeError:
                continue
            else:
                setattr(wrapper, attr, value)
        for attr in updated:
            getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
        wrapper.__wrapped__ = wrapped
        return wrapper

    _update_wrapper.__doc__ = functools.update_wrapper.__doc__

    def wraps(
        wrapped,
        assigned=functools.WRAPPER_ASSIGNMENTS,
        updated=functools.WRAPPER_UPDATES,
    ):
        return functools.partial(
            _update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated
        )

    wraps.__doc__ = functools.wraps.__doc__

else:
    wraps = functools.wraps


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(type):
        def __new__(cls, name, this_bases, d):
            if sys.version_info[:2] >= (3, 7):
                # This version introduced PEP 560 that requires a bit
                # of extra care (we mimic what is done by __build_class__).
                resolved_bases = types.resolve_bases(bases)
                if resolved_bases is not bases:
                    d["__orig_bases__"] = bases
            else:
                resolved_bases = bases
            return meta(name, resolved_bases, d)

        @classmethod
        def __prepare__(cls, name, this_bases):
            return meta.__prepare__(name, bases)

    return type.__new__(metaclass, "temporary_class", (), {})


def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""

    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get("__slots__")
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop("__dict__", None)
        orig_vars.pop("__weakref__", None)
        if hasattr(cls, "__qualname__"):
            orig_vars["__qualname__"] = cls.__qualname__
        return metaclass(cls.__name__, cls.__bases__, orig_vars)

    return wrapper


def ensure_binary(s, encoding="utf-8", errors="strict"):
    """Coerce **s** to six.binary_type.

    For Python 2:
      - `unicode` -> encoded to `str`
      - `str` -> `str`

    For Python 3:
      - `str` -> encoded to `bytes`
      - `bytes` -> `bytes`
    """
    if isinstance(s, binary_type):
        return s
    if isinstance(s, text_type):
        return s.encode(encoding, errors)
    raise TypeError("not expecting type '%s'" % type(s))


def ensure_str(s, encoding="utf-8", errors="strict"):
    """Coerce *s* to `str`.

    For Python 2:
      - `unicode` -> encoded to `str`
      - `str` -> `str`

    For Python 3:
      - `str` -> `str`
      - `bytes` -> decoded to `str`
    """
    # Optimization: Fast return for the common case.
    if type(s) is str:
        return s
    if PY2 and isinstance(s, text_type):
        return s.encode(encoding, errors)
    elif PY3 and isinstance(s, binary_type):
        return s.decode(encoding, errors)
    elif not isinstance(s, (text_type, binary_type)):
        raise TypeError("not expecting type '%s'" % type(s))
    return s


def ensure_text(s, encoding="utf-8", errors="strict"):
    """Coerce *s* to six.text_type.

    For Python 2:
      - `unicode` -> `unicode`
      - `str` -> `unicode`

    For Python 3:
      - `str` -> `str`
      - `bytes` -> decoded to `str`
    """
    if isinstance(s, binary_type):
        return s.decode(encoding, errors)
    elif isinstance(s, text_type):
        return s
    else:
        raise TypeError("not expecting type '%s'" % type(s))


def python_2_unicode_compatible(klass):
    """
    A class decorator that defines __unicode__ and __str__ methods under Python 2.
    Under Python 3 it does nothing.

    To support Python 2 and 3 with a single code base, define a __str__ method
    returning text and apply this decorator to the class.
    """
    if PY2:
        if "__str__" not in klass.__dict__:
            raise ValueError(
                "@python_2_unicode_compatible cannot be applied "
                "to %s because it doesn't define __str__()." % klass.__name__
            )
        klass.__unicode__ = klass.__str__
        klass.__str__ = lambda self: self.__unicode__().encode("utf-8")
    return klass


# Complete the moves implementation.
# This code is at the end of this module to speed up module loading.
# Turn this module into a package.
__path__ = []  # required for PEP 302 and PEP 451
__package__ = __name__  # see PEP 366 @ReservedAssignment
if globals().get("__spec__") is not None:
    __spec__.submodule_search_locations = []  # PEP 451 @UndefinedVariable
# Remove other six meta path importers, since they cause problems. This can
# happen if six is removed from sys.modules and then reloaded. (Setuptools does
# this for some reason.)
if sys.meta_path:
    for i, importer in enumerate(sys.meta_path):
        # Here's some real nastiness: Another "instance" of the six module might
        # be floating around. Therefore, we can't use isinstance() to check for
        # the six meta path importer, since the other six instance will have
        # inserted an importer with different class.
        if (
            type(importer).__name__ == "_SixMetaPathImporter"
            and importer.name == __name__
        ):
            del sys.meta_path[i]
            break
    del i, importer
# Finally, add the importer to the meta path import hook.
sys.meta_path.append(_importer)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\internals\construction.py ===
"""
Functions for preparing various inputs passed to the DataFrame or Series
constructors before passing them to a BlockManager.
"""
from __future__ import annotations

from collections import abc
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
from numpy import ma

from pandas._config import using_string_dtype

from pandas._libs import lib

from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    dict_compat,
    maybe_cast_to_datetime,
    maybe_convert_platform,
    maybe_infer_to_datetimelike,
)
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    is_integer_dtype,
    is_list_like,
    is_named_tuple,
    is_object_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core import (
    algorithms,
    common as com,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
    range_to_ndarray,
    sanitize_array,
)
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    TimedeltaIndex,
    default_index,
    ensure_index,
    get_objs_combined_axis,
    union_indexes,
)
from pandas.core.internals.array_manager import (
    ArrayManager,
    SingleArrayManager,
)
from pandas.core.internals.blocks import (
    BlockPlacement,
    ensure_block_shape,
    new_block,
    new_block_2d,
)
from pandas.core.internals.managers import (
    BlockManager,
    SingleBlockManager,
    create_block_manager_from_blocks,
    create_block_manager_from_column_arrays,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeObj,
        Manager,
        npt,
    )
# ---------------------------------------------------------------------
# BlockManager Interface


def arrays_to_mgr(
    arrays,
    columns: Index,
    index,
    *,
    dtype: DtypeObj | None = None,
    verify_integrity: bool = True,
    typ: str | None = None,
    consolidate: bool = True,
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.

    Needs to handle a lot of exceptional cases.
    """
    if verify_integrity:
        # figure out the index, if necessary
        if index is None:
            index = _extract_index(arrays)
        else:
            index = ensure_index(index)

        # don't force copy because getting jammed in an ndarray anyway
        arrays, refs = _homogenize(arrays, index, dtype)
        # _homogenize ensures
        #  - all(len(x) == len(index) for x in arrays)
        #  - all(x.ndim == 1 for x in arrays)
        #  - all(isinstance(x, (np.ndarray, ExtensionArray)) for x in arrays)
        #  - all(type(x) is not NumpyExtensionArray for x in arrays)

    else:
        index = ensure_index(index)
        arrays = [extract_array(x, extract_numpy=True) for x in arrays]
        # with _from_arrays, the passed arrays should never be Series objects
        refs = [None] * len(arrays)

        # Reached via DataFrame._from_arrays; we do minimal validation here
        for arr in arrays:
            if (
                not isinstance(arr, (np.ndarray, ExtensionArray))
                or arr.ndim != 1
                or len(arr) != len(index)
            ):
                raise ValueError(
                    "Arrays must be 1-dimensional np.ndarray or ExtensionArray "
                    "with length matching len(index)"
                )

    columns = ensure_index(columns)
    if len(columns) != len(arrays):
        raise ValueError("len(arrays) must match len(columns)")

    # from BlockManager perspective
    axes = [columns, index]

    if typ == "block":
        return create_block_manager_from_column_arrays(
            arrays, axes, consolidate=consolidate, refs=refs
        )
    elif typ == "array":
        return ArrayManager(arrays, [index, columns])
    else:
        raise ValueError(f"'typ' needs to be one of {{'block', 'array'}}, got '{typ}'")


def rec_array_to_mgr(
    data: np.rec.recarray | np.ndarray,
    index,
    columns,
    dtype: DtypeObj | None,
    copy: bool,
    typ: str,
) -> Manager:
    """
    Extract from a masked rec array and create the manager.
    """
    # essentially process a record array then fill it
    fdata = ma.getdata(data)
    if index is None:
        index = default_index(len(fdata))
    else:
        index = ensure_index(index)

    if columns is not None:
        columns = ensure_index(columns)
    arrays, arr_columns = to_arrays(fdata, columns)

    # create the manager

    arrays, arr_columns = reorder_arrays(arrays, arr_columns, columns, len(index))
    if columns is None:
        columns = arr_columns

    mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ)

    if copy:
        mgr = mgr.copy()
    return mgr


def mgr_to_mgr(mgr, typ: str, copy: bool = True) -> Manager:
    """
    Convert to specific type of Manager. Does not copy if the type is already
    correct. Does not guarantee a copy otherwise. `copy` keyword only controls
    whether conversion from Block->ArrayManager copies the 1D arrays.
    """
    new_mgr: Manager

    if typ == "block":
        if isinstance(mgr, BlockManager):
            new_mgr = mgr
        else:
            if mgr.ndim == 2:
                new_mgr = arrays_to_mgr(
                    mgr.arrays, mgr.axes[0], mgr.axes[1], typ="block"
                )
            else:
                new_mgr = SingleBlockManager.from_array(mgr.arrays[0], mgr.index)
    elif typ == "array":
        if isinstance(mgr, ArrayManager):
            new_mgr = mgr
        else:
            if mgr.ndim == 2:
                arrays = [mgr.iget_values(i) for i in range(len(mgr.axes[0]))]
                if copy:
                    arrays = [arr.copy() for arr in arrays]
                new_mgr = ArrayManager(arrays, [mgr.axes[1], mgr.axes[0]])
            else:
                array = mgr.internal_values()
                if copy:
                    array = array.copy()
                new_mgr = SingleArrayManager([array], [mgr.index])
    else:
        raise ValueError(f"'typ' needs to be one of {{'block', 'array'}}, got '{typ}'")
    return new_mgr


# ---------------------------------------------------------------------
# DataFrame Constructor Interface


def ndarray_to_mgr(
    values, index, columns, dtype: DtypeObj | None, copy: bool, typ: str
) -> Manager:
    # used in DataFrame.__init__
    # input must be a ndarray, list, Series, Index, ExtensionArray

    if isinstance(values, ABCSeries):
        if columns is None:
            if values.name is not None:
                columns = Index([values.name])
        if index is None:
            index = values.index
        else:
            values = values.reindex(index)

        # zero len case (GH #2234)
        if not len(values) and columns is not None and len(columns):
            values = np.empty((0, 1), dtype=object)

    # if the array preparation does a copy -> avoid this for ArrayManager,
    # since the copy is done on conversion to 1D arrays
    copy_on_sanitize = False if typ == "array" else copy

    vdtype = getattr(values, "dtype", None)
    refs = None
    if is_1d_only_ea_dtype(vdtype) or is_1d_only_ea_dtype(dtype):
        # GH#19157

        if isinstance(values, (np.ndarray, ExtensionArray)) and values.ndim > 1:
            # GH#12513 a EA dtype passed with a 2D array, split into
            #  multiple EAs that view the values
            # error: No overload variant of "__getitem__" of "ExtensionArray"
            # matches argument type "Tuple[slice, int]"
            values = [
                values[:, n]  # type: ignore[call-overload]
                for n in range(values.shape[1])
            ]
        else:
            values = [values]

        if columns is None:
            columns = Index(range(len(values)))
        else:
            columns = ensure_index(columns)

        return arrays_to_mgr(values, columns, index, dtype=dtype, typ=typ)

    elif isinstance(vdtype, ExtensionDtype):
        # i.e. Datetime64TZ, PeriodDtype; cases with is_1d_only_ea_dtype(vdtype)
        #  are already caught above
        values = extract_array(values, extract_numpy=True)
        if copy:
            values = values.copy()
        if values.ndim == 1:
            values = values.reshape(-1, 1)

    elif isinstance(values, (ABCSeries, Index)):
        if not copy_on_sanitize and (
            dtype is None or astype_is_view(values.dtype, dtype)
        ):
            refs = values._references

        if copy_on_sanitize:
            values = values._values.copy()
        else:
            values = values._values

        values = _ensure_2d(values)

    elif isinstance(values, (np.ndarray, ExtensionArray)):
        # drop subclass info
        if copy_on_sanitize and (dtype is None or astype_is_view(values.dtype, dtype)):
            # only force a copy now if copy=True was requested
            # and a subsequent `astype` will not already result in a copy
            values = np.array(values, copy=True, order="F")
        else:
            values = np.asarray(values)
        values = _ensure_2d(values)

    else:
        # by definition an array here
        # the dtypes will be coerced to a single dtype
        values = _prep_ndarraylike(values, copy=copy_on_sanitize)

    if dtype is not None and values.dtype != dtype:
        # GH#40110 see similar check inside sanitize_array
        values = sanitize_array(
            values,
            None,
            dtype=dtype,
            copy=copy_on_sanitize,
            allow_2d=True,
        )

    # _prep_ndarraylike ensures that values.ndim == 2 at this point
    index, columns = _get_axes(
        values.shape[0], values.shape[1], index=index, columns=columns
    )

    _check_values_indices_shape_match(values, index, columns)

    if typ == "array":
        if issubclass(values.dtype.type, str):
            values = np.array(values, dtype=object)

        if dtype is None and is_object_dtype(values.dtype):
            arrays = [
                ensure_wrapped_if_datetimelike(
                    maybe_infer_to_datetimelike(values[:, i])
                )
                for i in range(values.shape[1])
            ]
        else:
            if lib.is_np_dtype(values.dtype, "mM"):
                values = ensure_wrapped_if_datetimelike(values)
            arrays = [values[:, i] for i in range(values.shape[1])]

        if copy:
            arrays = [arr.copy() for arr in arrays]

        return ArrayManager(arrays, [index, columns], verify_integrity=False)

    values = values.T

    # if we don't have a dtype specified, then try to convert objects
    # on the entire block; this is to convert if we have datetimelike's
    # embedded in an object type
    if dtype is None and is_object_dtype(values.dtype):
        obj_columns = list(values)
        maybe_datetime = [maybe_infer_to_datetimelike(x) for x in obj_columns]
        # don't convert (and copy) the objects if no type inference occurs
        if any(x is not y for x, y in zip(obj_columns, maybe_datetime)):
            dvals_list = [ensure_block_shape(dval, 2) for dval in maybe_datetime]
            block_values = [
                new_block_2d(dvals_list[n], placement=BlockPlacement(n))
                for n in range(len(dvals_list))
            ]
        else:
            bp = BlockPlacement(slice(len(columns)))
            nb = new_block_2d(values, placement=bp, refs=refs)
            block_values = [nb]
    elif dtype is None and values.dtype.kind == "U" and using_string_dtype():
        dtype = StringDtype(na_value=np.nan)

        obj_columns = list(values)
        block_values = [
            new_block(
                dtype.construct_array_type()._from_sequence(data, dtype=dtype),
                BlockPlacement(slice(i, i + 1)),
                ndim=2,
            )
            for i, data in enumerate(obj_columns)
        ]

    else:
        bp = BlockPlacement(slice(len(columns)))
        nb = new_block_2d(values, placement=bp, refs=refs)
        block_values = [nb]

    if len(columns) == 0:
        # TODO: check len(values) == 0?
        block_values = []

    return create_block_manager_from_blocks(
        block_values, [columns, index], verify_integrity=False
    )


def _check_values_indices_shape_match(
    values: np.ndarray, index: Index, columns: Index
) -> None:
    """
    Check that the shape implied by our axes matches the actual shape of the
    data.
    """
    if values.shape[1] != len(columns) or values.shape[0] != len(index):
        # Could let this raise in Block constructor, but we get a more
        #  helpful exception message this way.
        if values.shape[0] == 0 < len(index):
            raise ValueError("Empty data passed with indices specified.")

        passed = values.shape
        implied = (len(index), len(columns))
        raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")


def dict_to_mgr(
    data: dict,
    index,
    columns,
    *,
    dtype: DtypeObj | None = None,
    typ: str = "block",
    copy: bool = True,
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.
    Needs to handle a lot of exceptional cases.

    Used in DataFrame.__init__
    """
    arrays: Sequence[Any] | Series

    if columns is not None:
        from pandas.core.series import Series

        arrays = Series(data, index=columns, dtype=object)
        missing = arrays.isna()
        if index is None:
            # GH10856
            # raise ValueError if only scalars in dict
            index = _extract_index(arrays[~missing])
        else:
            index = ensure_index(index)

        # no obvious "empty" int column
        if missing.any() and not is_integer_dtype(dtype):
            nan_dtype: DtypeObj

            if dtype is not None:
                # calling sanitize_array ensures we don't mix-and-match
                #  NA dtypes
                midxs = missing.values.nonzero()[0]
                for i in midxs:
                    arr = sanitize_array(arrays.iat[i], index, dtype=dtype)
                    arrays.iat[i] = arr
            else:
                # GH#1783
                nan_dtype = np.dtype("object")
                val = construct_1d_arraylike_from_scalar(np.nan, len(index), nan_dtype)
                nmissing = missing.sum()
                if copy:
                    rhs = [val] * nmissing
                else:
                    # GH#45369
                    rhs = [val.copy() for _ in range(nmissing)]
                arrays.loc[missing] = rhs

        arrays = list(arrays)
        columns = ensure_index(columns)

    else:
        keys = list(data.keys())
        columns = Index(keys) if keys else default_index(0)
        arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]

    if copy:
        if typ == "block":
            # We only need to copy arrays that will not get consolidated, i.e.
            #  only EA arrays
            arrays = [
                x.copy()
                if isinstance(x, ExtensionArray)
                else x.copy(deep=True)
                if (
                    isinstance(x, Index)
                    or isinstance(x, ABCSeries)
                    and is_1d_only_ea_dtype(x.dtype)
                )
                else x
                for x in arrays
            ]
        else:
            # dtype check to exclude e.g. range objects, scalars
            arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]

    return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)


def nested_data_to_arrays(
    data: Sequence,
    columns: Index | None,
    index: Index | None,
    dtype: DtypeObj | None,
) -> tuple[list[ArrayLike], Index, Index]:
    """
    Convert a single sequence of arrays to multiple arrays.
    """
    # By the time we get here we have already checked treat_as_nested(data)

    if is_named_tuple(data[0]) and columns is None:
        columns = ensure_index(data[0]._fields)

    arrays, columns = to_arrays(data, columns, dtype=dtype)
    columns = ensure_index(columns)

    if index is None:
        if isinstance(data[0], ABCSeries):
            index = _get_names_from_index(data)
        else:
            index = default_index(len(data))

    return arrays, columns, index


def treat_as_nested(data) -> bool:
    """
    Check if we should use nested_data_to_arrays.
    """
    return (
        len(data) > 0
        and is_list_like(data[0])
        and getattr(data[0], "ndim", 1) == 1
        and not (isinstance(data, ExtensionArray) and data.ndim == 2)
    )


# ---------------------------------------------------------------------


def _prep_ndarraylike(values, copy: bool = True) -> np.ndarray:
    # values is specifically _not_ ndarray, EA, Index, or Series
    # We only get here with `not treat_as_nested(values)`

    if len(values) == 0:
        # TODO: check for length-zero range, in which case return int64 dtype?
        # TODO: reuse anything in try_cast?
        return np.empty((0, 0), dtype=object)
    elif isinstance(values, range):
        arr = range_to_ndarray(values)
        return arr[..., np.newaxis]

    def convert(v):
        if not is_list_like(v) or isinstance(v, ABCDataFrame):
            return v

        v = extract_array(v, extract_numpy=True)
        res = maybe_convert_platform(v)
        # We don't do maybe_infer_to_datetimelike here bc we will end up doing
        #  it column-by-column in ndarray_to_mgr
        return res

    # we could have a 1-dim or 2-dim list here
    # this is equiv of np.asarray, but does object conversion
    # and platform dtype preservation
    # does not convert e.g. [1, "a", True] to ["1", "a", "True"] like
    #  np.asarray would
    if is_list_like(values[0]):
        values = np.array([convert(v) for v in values])
    elif isinstance(values[0], np.ndarray) and values[0].ndim == 0:
        # GH#21861 see test_constructor_list_of_lists
        values = np.array([convert(v) for v in values])
    else:
        values = convert(values)

    return _ensure_2d(values)


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    """
    Reshape 1D values, raise on anything else other than 2D.
    """
    if values.ndim == 1:
        values = values.reshape((values.shape[0], 1))
    elif values.ndim != 2:
        raise ValueError(f"Must pass 2-d input. shape={values.shape}")
    return values


def _homogenize(
    data, index: Index, dtype: DtypeObj | None
) -> tuple[list[ArrayLike], list[Any]]:
    oindex = None
    homogenized = []
    # if the original array-like in `data` is a Series, keep track of this Series' refs
    refs: list[Any] = []

    for val in data:
        if isinstance(val, (ABCSeries, Index)):
            if dtype is not None:
                val = val.astype(dtype, copy=False)
            if isinstance(val, ABCSeries) and val.index is not index:
                # Forces alignment. No need to copy data since we
                # are putting it into an ndarray later
                val = val.reindex(index, copy=False)
            refs.append(val._references)
            val = val._values
        else:
            if isinstance(val, dict):
                # GH#41785 this _should_ be equivalent to (but faster than)
                #  val = Series(val, index=index)._values
                if oindex is None:
                    oindex = index.astype("O")

                if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
                    # see test_constructor_dict_datetime64_index
                    val = dict_compat(val)
                else:
                    # see test_constructor_subclass_dict
                    val = dict(val)
                val = lib.fast_multiget(val, oindex._values, default=np.nan)

            val = sanitize_array(val, index, dtype=dtype, copy=False)
            com.require_length_match(val, index)
            refs.append(None)

        homogenized.append(val)

    return homogenized, refs


def _extract_index(data) -> Index:
    """
    Try to infer an Index from the passed data, raise ValueError on failure.
    """
    index: Index
    if len(data) == 0:
        return default_index(0)

    raw_lengths = []
    indexes: list[list[Hashable] | Index] = []

    have_raw_arrays = False
    have_series = False
    have_dicts = False

    for val in data:
        if isinstance(val, ABCSeries):
            have_series = True
            indexes.append(val.index)
        elif isinstance(val, dict):
            have_dicts = True
            indexes.append(list(val.keys()))
        elif is_list_like(val) and getattr(val, "ndim", 1) == 1:
            have_raw_arrays = True
            raw_lengths.append(len(val))
        elif isinstance(val, np.ndarray) and val.ndim > 1:
            raise ValueError("Per-column arrays must each be 1-dimensional")

    if not indexes and not raw_lengths:
        raise ValueError("If using all scalar values, you must pass an index")

    if have_series:
        index = union_indexes(indexes)
    elif have_dicts:
        index = union_indexes(indexes, sort=False)

    if have_raw_arrays:
        lengths = list(set(raw_lengths))
        if len(lengths) > 1:
            raise ValueError("All arrays must be of the same length")

        if have_dicts:
            raise ValueError(
                "Mixing dicts with non-Series may lead to ambiguous ordering."
            )

        if have_series:
            if lengths[0] != len(index):
                msg = (
                    f"array length {lengths[0]} does not match index "
                    f"length {len(index)}"
                )
                raise ValueError(msg)
        else:
            index = default_index(lengths[0])

    return ensure_index(index)


def reorder_arrays(
    arrays: list[ArrayLike], arr_columns: Index, columns: Index | None, length: int
) -> tuple[list[ArrayLike], Index]:
    """
    Pre-emptively (cheaply) reindex arrays with new columns.
    """
    # reorder according to the columns
    if columns is not None:
        if not columns.equals(arr_columns):
            # if they are equal, there is nothing to do
            new_arrays: list[ArrayLike] = []
            indexer = arr_columns.get_indexer(columns)
            for i, k in enumerate(indexer):
                if k == -1:
                    # by convention default is all-NaN object dtype
                    arr = np.empty(length, dtype=object)
                    arr.fill(np.nan)
                else:
                    arr = arrays[k]
                new_arrays.append(arr)

            arrays = new_arrays
            arr_columns = columns

    return arrays, arr_columns


def _get_names_from_index(data) -> Index:
    has_some_name = any(getattr(s, "name", None) is not None for s in data)
    if not has_some_name:
        return default_index(len(data))

    index: list[Hashable] = list(range(len(data)))
    count = 0
    for i, s in enumerate(data):
        n = getattr(s, "name", None)
        if n is not None:
            index[i] = n
        else:
            index[i] = f"Unnamed {count}"
            count += 1

    return Index(index)


def _get_axes(
    N: int, K: int, index: Index | None, columns: Index | None
) -> tuple[Index, Index]:
    # helper to create the axes as indexes
    # return axes or defaults

    if index is None:
        index = default_index(N)
    else:
        index = ensure_index(index)

    if columns is None:
        columns = default_index(K)
    else:
        columns = ensure_index(columns)
    return index, columns


def dataclasses_to_dicts(data):
    """
    Converts a list of dataclass instances to a list of dictionaries.

    Parameters
    ----------
    data : List[Type[dataclass]]

    Returns
    --------
    list_dict : List[dict]

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int
    ...     y: int

    >>> dataclasses_to_dicts([Point(1, 2), Point(2, 3)])
    [{'x': 1, 'y': 2}, {'x': 2, 'y': 3}]

    """
    from dataclasses import asdict

    return list(map(asdict, data))


# ---------------------------------------------------------------------
# Conversion of Inputs to Arrays


def to_arrays(
    data, columns: Index | None, dtype: DtypeObj | None = None
) -> tuple[list[ArrayLike], Index]:
    """
    Return list of arrays, columns.

    Returns
    -------
    list[ArrayLike]
        These will become columns in a DataFrame.
    Index
        This will become frame.columns.

    Notes
    -----
    Ensures that len(result_arrays) == len(result_index).
    """

    if not len(data):
        if isinstance(data, np.ndarray):
            if data.dtype.names is not None:
                # i.e. numpy structured array
                columns = ensure_index(data.dtype.names)
                arrays = [data[name] for name in columns]

                if len(data) == 0:
                    # GH#42456 the indexing above results in list of 2D ndarrays
                    # TODO: is that an issue with numpy?
                    for i, arr in enumerate(arrays):
                        if arr.ndim == 2:
                            arrays[i] = arr[:, 0]

                return arrays, columns
        return [], ensure_index([])

    elif isinstance(data, np.ndarray) and data.dtype.names is not None:
        # e.g. recarray
        columns = Index(list(data.dtype.names))
        arrays = [data[k] for k in columns]
        return arrays, columns

    if isinstance(data[0], (list, tuple)):
        arr = _list_to_arrays(data)
    elif isinstance(data[0], abc.Mapping):
        arr, columns = _list_of_dict_to_arrays(data, columns)
    elif isinstance(data[0], ABCSeries):
        arr, columns = _list_of_series_to_arrays(data, columns)
    else:
        # last ditch effort
        data = [tuple(x) for x in data]
        arr = _list_to_arrays(data)

    content, columns = _finalize_columns_and_data(arr, columns, dtype)
    return content, columns


def _list_to_arrays(data: list[tuple | list]) -> np.ndarray:
    # Returned np.ndarray has ndim = 2
    # Note: we already check len(data) > 0 before getting hre
    if isinstance(data[0], tuple):
        content = lib.to_object_array_tuples(data)
    else:
        # list of lists
        content = lib.to_object_array(data)
    return content


def _list_of_series_to_arrays(
    data: list,
    columns: Index | None,
) -> tuple[np.ndarray, Index]:
    # returned np.ndarray has ndim == 2

    if columns is None:
        # We know pass_data is non-empty because data[0] is a Series
        pass_data = [x for x in data if isinstance(x, (ABCSeries, ABCDataFrame))]
        columns = get_objs_combined_axis(pass_data, sort=False)

    indexer_cache: dict[int, np.ndarray] = {}

    aligned_values = []
    for s in data:
        index = getattr(s, "index", None)
        if index is None:
            index = default_index(len(s))

        if id(index) in indexer_cache:
            indexer = indexer_cache[id(index)]
        else:
            indexer = indexer_cache[id(index)] = index.get_indexer(columns)

        values = extract_array(s, extract_numpy=True)
        aligned_values.append(algorithms.take_nd(values, indexer))

    content = np.vstack(aligned_values)
    return content, columns


def _list_of_dict_to_arrays(
    data: list[dict],
    columns: Index | None,
) -> tuple[np.ndarray, Index]:
    """
    Convert list of dicts to numpy arrays

    if `columns` is not passed, column names are inferred from the records
    - for OrderedDict and dicts, the column names match
      the key insertion-order from the first record to the last.
    - For other kinds of dict-likes, the keys are lexically sorted.

    Parameters
    ----------
    data : iterable
        collection of records (OrderedDict, dict)
    columns: iterables or None

    Returns
    -------
    content : np.ndarray[object, ndim=2]
    columns : Index
    """
    if columns is None:
        gen = (list(x.keys()) for x in data)
        sort = not any(isinstance(d, dict) for d in data)
        pre_cols = lib.fast_unique_multiple_list_gen(gen, sort=sort)
        columns = ensure_index(pre_cols)

    # assure that they are of the base dict class and not of derived
    # classes
    data = [d if type(d) is dict else dict(d) for d in data]  # noqa: E721

    content = lib.dicts_to_array(data, list(columns))
    return content, columns


def _finalize_columns_and_data(
    content: np.ndarray,  # ndim == 2
    columns: Index | None,
    dtype: DtypeObj | None,
) -> tuple[list[ArrayLike], Index]:
    """
    Ensure we have valid columns, cast object dtypes if possible.
    """
    contents = list(content.T)

    try:
        columns = _validate_or_indexify_columns(contents, columns)
    except AssertionError as err:
        # GH#26429 do not raise user-facing AssertionError
        raise ValueError(err) from err

    if len(contents) and contents[0].dtype == np.object_:
        contents = convert_object_array(contents, dtype=dtype)

    return contents, columns


def _validate_or_indexify_columns(
    content: list[np.ndarray], columns: Index | None
) -> Index:
    """
    If columns is None, make numbers as column names; Otherwise, validate that
    columns have valid length.

    Parameters
    ----------
    content : list of np.ndarrays
    columns : Index or None

    Returns
    -------
    Index
        If columns is None, assign positional column index value as columns.

    Raises
    ------
    1. AssertionError when content is not composed of list of lists, and if
        length of columns is not equal to length of content.
    2. ValueError when content is list of lists, but length of each sub-list
        is not equal
    3. ValueError when content is list of lists, but length of sub-list is
        not equal to length of content
    """
    if columns is None:
        columns = default_index(len(content))
    else:
        # Add mask for data which is composed of list of lists
        is_mi_list = isinstance(columns, list) and all(
            isinstance(col, list) for col in columns
        )

        if not is_mi_list and len(columns) != len(content):  # pragma: no cover
            # caller's responsibility to check for this...
            raise AssertionError(
                f"{len(columns)} columns passed, passed data had "
                f"{len(content)} columns"
            )
        if is_mi_list:
            # check if nested list column, length of each sub-list should be equal
            if len({len(col) for col in columns}) > 1:
                raise ValueError(
                    "Length of columns passed for MultiIndex columns is different"
                )

            # if columns is not empty and length of sublist is not equal to content
            if columns and len(columns[0]) != len(content):
                raise ValueError(
                    f"{len(columns[0])} columns passed, passed data had "
                    f"{len(content)} columns"
                )
    return columns


def convert_object_array(
    content: list[npt.NDArray[np.object_]],
    dtype: DtypeObj | None,
    dtype_backend: str = "numpy",
    coerce_float: bool = False,
) -> list[ArrayLike]:
    """
    Internal function to convert object array.

    Parameters
    ----------
    content: List[np.ndarray]
    dtype: np.dtype or ExtensionDtype
    dtype_backend: Controls if nullable/pyarrow dtypes are returned.
    coerce_float: Cast floats that are integers to int.

    Returns
    -------
    List[ArrayLike]
    """
    # provide soft conversion of object dtypes

    def convert(arr):
        if dtype != np.dtype("O"):
            arr = lib.maybe_convert_objects(
                arr,
                try_float=coerce_float,
                convert_to_nullable_dtype=dtype_backend != "numpy",
            )
            # Notes on cases that get here 2023-02-15
            # 1) we DO get here when arr is all Timestamps and dtype=None
            # 2) disabling this doesn't break the world, so this must be
            #    getting caught at a higher level
            # 3) passing convert_non_numeric to maybe_convert_objects get this right
            # 4) convert_non_numeric?

            if dtype is None:
                if arr.dtype == np.dtype("O"):
                    # i.e. maybe_convert_objects didn't convert
                    convert_to_nullable_dtype = dtype_backend != "numpy"
                    arr = maybe_infer_to_datetimelike(arr, convert_to_nullable_dtype)
                    if convert_to_nullable_dtype and arr.dtype == np.dtype("O"):
                        new_dtype = StringDtype()
                        arr_cls = new_dtype.construct_array_type()
                        arr = arr_cls._from_sequence(arr, dtype=new_dtype)
                elif dtype_backend != "numpy" and isinstance(arr, np.ndarray):
                    if arr.dtype.kind in "iufb":
                        arr = pd_array(arr, copy=False)

            elif isinstance(dtype, ExtensionDtype):
                # TODO: test(s) that get here
                # TODO: try to de-duplicate this convert function with
                #  core.construction functions
                cls = dtype.construct_array_type()
                arr = cls._from_sequence(arr, dtype=dtype, copy=False)
            elif dtype.kind in "mM":
                # This restriction is harmless bc these are the only cases
                #  where maybe_cast_to_datetime is not a no-op.
                # Here we know:
                #  1) dtype.kind in "mM" and
                #  2) arr is either object or numeric dtype
                arr = maybe_cast_to_datetime(arr, dtype)

        return arr

    arrays = [convert(arr) for arr in content]

    return arrays

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\internal\decoder.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Code for decoding protocol buffer primitives.

This code is very similar to encoder.py -- read the docs for that module first.

A "decoder" is a function with the signature:
  Decode(buffer, pos, end, message, field_dict)
The arguments are:
  buffer:     The string containing the encoded message.
  pos:        The current position in the string.
  end:        The position in the string where the current message ends.  May be
              less than len(buffer) if we're reading a sub-message.
  message:    The message object into which we're parsing.
  field_dict: message._fields (avoids a hashtable lookup).
The decoder reads the field and stores it into field_dict, returning the new
buffer position.  A decoder for a repeated field may proactively decode all of
the elements of that field, if they appear consecutively.

Note that decoders may throw any of the following:
  IndexError:  Indicates a truncated message.
  struct.error:  Unpacking of a fixed-width field failed.
  message.DecodeError:  Other errors.

Decoders are expected to raise an exception if they are called with pos > end.
This allows callers to be lax about bounds checking:  it's fineto read past
"end" as long as you are sure that someone else will notice and throw an
exception later on.

Something up the call stack is expected to catch IndexError and struct.error
and convert them to message.DecodeError.

Decoders are constructed using decoder constructors with the signature:
  MakeDecoder(field_number, is_repeated, is_packed, key, new_default)
The arguments are:
  field_number:  The field number of the field we want to decode.
  is_repeated:   Is the field a repeated field? (bool)
  is_packed:     Is the field a packed field? (bool)
  key:           The key to use when looking up the field within field_dict.
                 (This is actually the FieldDescriptor but nothing in this
                 file should depend on that.)
  new_default:   A function which takes a message object as a parameter and
                 returns a new instance of the default value for this field.
                 (This is called for repeated fields and sub-messages, when an
                 instance does not already exist.)

As with encoders, we define a decoder constructor for every type of field.
Then, for every field of every message class we construct an actual decoder.
That decoder goes into a dict indexed by tag, so when we decode a message
we repeatedly read a tag, look up the corresponding decoder, and invoke it.
"""

__author__ = 'kenton@google.com (Kenton Varda)'

import math
import numbers
import struct

from google.protobuf import message
from google.protobuf.internal import containers
from google.protobuf.internal import encoder
from google.protobuf.internal import wire_format


# This is not for optimization, but rather to avoid conflicts with local
# variables named "message".
_DecodeError = message.DecodeError


def IsDefaultScalarValue(value):
  """Returns whether or not a scalar value is the default value of its type.

  Specifically, this should be used to determine presence of implicit-presence
  fields, where we disallow custom defaults.

  Args:
    value: A scalar value to check.

  Returns:
    True if the value is equivalent to a default value, False otherwise.
  """
  if isinstance(value, numbers.Number) and math.copysign(1.0, value) < 0:
    # Special case for negative zero, where "truthiness" fails to give the right
    # answer.
    return False

  # Normally, we can just use Python's boolean conversion.
  return not value


def _VarintDecoder(mask, result_type):
  """Return an encoder for a basic varint value (does not include tag).

  Decoded values will be bitwise-anded with the given mask before being
  returned, e.g. to limit them to 32 bits.  The returned decoder does not
  take the usual "end" parameter -- the caller is expected to do bounds checking
  after the fact (often the caller can defer such checking until later).  The
  decoder returns a (value, new_pos) pair.
  """

  def DecodeVarint(buffer, pos: int=None):
    result = 0
    shift = 0
    while 1:
      if pos is None:
        # Read from BytesIO
        try:
          b = buffer.read(1)[0]
        except IndexError as e:
          if shift == 0:
            # End of BytesIO.
            return None
          else:
            raise ValueError('Fail to read varint %s' % str(e))
      else:
        b = buffer[pos]
        pos += 1
      result |= ((b & 0x7f) << shift)
      if not (b & 0x80):
        result &= mask
        result = result_type(result)
        return result if pos is None else (result, pos)
      shift += 7
      if shift >= 64:
        raise _DecodeError('Too many bytes when decoding varint.')

  return DecodeVarint


def _SignedVarintDecoder(bits, result_type):
  """Like _VarintDecoder() but decodes signed values."""

  signbit = 1 << (bits - 1)
  mask = (1 << bits) - 1

  def DecodeVarint(buffer, pos):
    result = 0
    shift = 0
    while 1:
      b = buffer[pos]
      result |= ((b & 0x7f) << shift)
      pos += 1
      if not (b & 0x80):
        result &= mask
        result = (result ^ signbit) - signbit
        result = result_type(result)
        return (result, pos)
      shift += 7
      if shift >= 64:
        raise _DecodeError('Too many bytes when decoding varint.')
  return DecodeVarint

# All 32-bit and 64-bit values are represented as int.
_DecodeVarint = _VarintDecoder((1 << 64) - 1, int)
_DecodeSignedVarint = _SignedVarintDecoder(64, int)

# Use these versions for values which must be limited to 32 bits.
_DecodeVarint32 = _VarintDecoder((1 << 32) - 1, int)
_DecodeSignedVarint32 = _SignedVarintDecoder(32, int)


def ReadTag(buffer, pos):
  """Read a tag from the memoryview, and return a (tag_bytes, new_pos) tuple.

  We return the raw bytes of the tag rather than decoding them.  The raw
  bytes can then be used to look up the proper decoder.  This effectively allows
  us to trade some work that would be done in pure-python (decoding a varint)
  for work that is done in C (searching for a byte string in a hash table).
  In a low-level language it would be much cheaper to decode the varint and
  use that, but not in Python.

  Args:
    buffer: memoryview object of the encoded bytes
    pos: int of the current position to start from

  Returns:
    Tuple[bytes, int] of the tag data and new position.
  """
  start = pos
  while buffer[pos] & 0x80:
    pos += 1
  pos += 1

  tag_bytes = buffer[start:pos].tobytes()
  return tag_bytes, pos


def DecodeTag(tag_bytes):
  """Decode a tag from the bytes.

  Args:
    tag_bytes: the bytes of the tag

  Returns:
    Tuple[int, int] of the tag field number and wire type.
  """
  (tag, _) = _DecodeVarint(tag_bytes, 0)
  return wire_format.UnpackTag(tag)


# --------------------------------------------------------------------


def _SimpleDecoder(wire_type, decode_value):
  """Return a constructor for a decoder for fields of a particular type.

  Args:
      wire_type:  The field's wire type.
      decode_value:  A function which decodes an individual value, e.g.
        _DecodeVarint()
  """

  def SpecificDecoder(field_number, is_repeated, is_packed, key, new_default,
                      clear_if_default=False):
    if is_packed:
      local_DecodeVarint = _DecodeVarint
      def DecodePackedField(
          buffer, pos, end, message, field_dict, current_depth=0
      ):
        del current_depth  # unused
        value = field_dict.get(key)
        if value is None:
          value = field_dict.setdefault(key, new_default(message))
        (endpoint, pos) = local_DecodeVarint(buffer, pos)
        endpoint += pos
        if endpoint > end:
          raise _DecodeError('Truncated message.')
        while pos < endpoint:
          (element, pos) = decode_value(buffer, pos)
          value.append(element)
        if pos > endpoint:
          del value[-1]   # Discard corrupt value.
          raise _DecodeError('Packed element was truncated.')
        return pos

      return DecodePackedField
    elif is_repeated:
      tag_bytes = encoder.TagBytes(field_number, wire_type)
      tag_len = len(tag_bytes)
      def DecodeRepeatedField(
          buffer, pos, end, message, field_dict, current_depth=0
      ):
        del current_depth  # unused
        value = field_dict.get(key)
        if value is None:
          value = field_dict.setdefault(key, new_default(message))
        while 1:
          (element, new_pos) = decode_value(buffer, pos)
          value.append(element)
          # Predict that the next tag is another copy of the same repeated
          # field.
          pos = new_pos + tag_len
          if buffer[new_pos:pos] != tag_bytes or new_pos >= end:
            # Prediction failed.  Return.
            if new_pos > end:
              raise _DecodeError('Truncated message.')
            return new_pos

      return DecodeRepeatedField
    else:

      def DecodeField(buffer, pos, end, message, field_dict, current_depth=0):
        del current_depth  # unused
        (new_value, pos) = decode_value(buffer, pos)
        if pos > end:
          raise _DecodeError('Truncated message.')
        if clear_if_default and IsDefaultScalarValue(new_value):
          field_dict.pop(key, None)
        else:
          field_dict[key] = new_value
        return pos

      return DecodeField

  return SpecificDecoder


def _ModifiedDecoder(wire_type, decode_value, modify_value):
  """Like SimpleDecoder but additionally invokes modify_value on every value
  before storing it.  Usually modify_value is ZigZagDecode.
  """

  # Reusing _SimpleDecoder is slightly slower than copying a bunch of code, but
  # not enough to make a significant difference.

  def InnerDecode(buffer, pos):
    (result, new_pos) = decode_value(buffer, pos)
    return (modify_value(result), new_pos)
  return _SimpleDecoder(wire_type, InnerDecode)


def _StructPackDecoder(wire_type, format):
  """Return a constructor for a decoder for a fixed-width field.

  Args:
      wire_type:  The field's wire type.
      format:  The format string to pass to struct.unpack().
  """

  value_size = struct.calcsize(format)
  local_unpack = struct.unpack

  # Reusing _SimpleDecoder is slightly slower than copying a bunch of code, but
  # not enough to make a significant difference.

  # Note that we expect someone up-stack to catch struct.error and convert
  # it to _DecodeError -- this way we don't have to set up exception-
  # handling blocks every time we parse one value.

  def InnerDecode(buffer, pos):
    new_pos = pos + value_size
    result = local_unpack(format, buffer[pos:new_pos])[0]
    return (result, new_pos)
  return _SimpleDecoder(wire_type, InnerDecode)


def _FloatDecoder():
  """Returns a decoder for a float field.

  This code works around a bug in struct.unpack for non-finite 32-bit
  floating-point values.
  """

  local_unpack = struct.unpack

  def InnerDecode(buffer, pos):
    """Decode serialized float to a float and new position.

    Args:
      buffer: memoryview of the serialized bytes
      pos: int, position in the memory view to start at.

    Returns:
      Tuple[float, int] of the deserialized float value and new position
      in the serialized data.
    """
    # We expect a 32-bit value in little-endian byte order.  Bit 1 is the sign
    # bit, bits 2-9 represent the exponent, and bits 10-32 are the significand.
    new_pos = pos + 4
    float_bytes = buffer[pos:new_pos].tobytes()

    # If this value has all its exponent bits set, then it's non-finite.
    # In Python 2.4, struct.unpack will convert it to a finite 64-bit value.
    # To avoid that, we parse it specially.
    if (float_bytes[3:4] in b'\x7F\xFF' and float_bytes[2:3] >= b'\x80'):
      # If at least one significand bit is set...
      if float_bytes[0:3] != b'\x00\x00\x80':
        return (math.nan, new_pos)
      # If sign bit is set...
      if float_bytes[3:4] == b'\xFF':
        return (-math.inf, new_pos)
      return (math.inf, new_pos)

    # Note that we expect someone up-stack to catch struct.error and convert
    # it to _DecodeError -- this way we don't have to set up exception-
    # handling blocks every time we parse one value.
    result = local_unpack('<f', float_bytes)[0]
    return (result, new_pos)
  return _SimpleDecoder(wire_format.WIRETYPE_FIXED32, InnerDecode)


def _DoubleDecoder():
  """Returns a decoder for a double field.

  This code works around a bug in struct.unpack for not-a-number.
  """

  local_unpack = struct.unpack

  def InnerDecode(buffer, pos):
    """Decode serialized double to a double and new position.

    Args:
      buffer: memoryview of the serialized bytes.
      pos: int, position in the memory view to start at.

    Returns:
      Tuple[float, int] of the decoded double value and new position
      in the serialized data.
    """
    # We expect a 64-bit value in little-endian byte order.  Bit 1 is the sign
    # bit, bits 2-12 represent the exponent, and bits 13-64 are the significand.
    new_pos = pos + 8
    double_bytes = buffer[pos:new_pos].tobytes()

    # If this value has all its exponent bits set and at least one significand
    # bit set, it's not a number.  In Python 2.4, struct.unpack will treat it
    # as inf or -inf.  To avoid that, we treat it specially.
    if ((double_bytes[7:8] in b'\x7F\xFF')
        and (double_bytes[6:7] >= b'\xF0')
        and (double_bytes[0:7] != b'\x00\x00\x00\x00\x00\x00\xF0')):
      return (math.nan, new_pos)

    # Note that we expect someone up-stack to catch struct.error and convert
    # it to _DecodeError -- this way we don't have to set up exception-
    # handling blocks every time we parse one value.
    result = local_unpack('<d', double_bytes)[0]
    return (result, new_pos)
  return _SimpleDecoder(wire_format.WIRETYPE_FIXED64, InnerDecode)


def EnumDecoder(field_number, is_repeated, is_packed, key, new_default,
                clear_if_default=False):
  """Returns a decoder for enum field."""
  enum_type = key.enum_type
  if is_packed:
    local_DecodeVarint = _DecodeVarint
    def DecodePackedField(
        buffer, pos, end, message, field_dict, current_depth=0
    ):
      """Decode serialized packed enum to its value and a new position.

      Args:
        buffer: memoryview of the serialized bytes.
        pos: int, position in the memory view to start at.
        end: int, end position of serialized data
        message: Message object to store unknown fields in
        field_dict: Map[Descriptor, Any] to store decoded values in.

      Returns:
        int, new position in serialized data.
      """
      del current_depth  # unused
      value = field_dict.get(key)
      if value is None:
        value = field_dict.setdefault(key, new_default(message))
      (endpoint, pos) = local_DecodeVarint(buffer, pos)
      endpoint += pos
      if endpoint > end:
        raise _DecodeError('Truncated message.')
      while pos < endpoint:
        value_start_pos = pos
        (element, pos) = _DecodeSignedVarint32(buffer, pos)
        # pylint: disable=protected-access
        if element in enum_type.values_by_number:
          value.append(element)
        else:
          if not message._unknown_fields:
            message._unknown_fields = []
          tag_bytes = encoder.TagBytes(field_number,
                                       wire_format.WIRETYPE_VARINT)

          message._unknown_fields.append(
              (tag_bytes, buffer[value_start_pos:pos].tobytes()))
          # pylint: enable=protected-access
      if pos > endpoint:
        if element in enum_type.values_by_number:
          del value[-1]   # Discard corrupt value.
        else:
          del message._unknown_fields[-1]
          # pylint: enable=protected-access
        raise _DecodeError('Packed element was truncated.')
      return pos

    return DecodePackedField
  elif is_repeated:
    tag_bytes = encoder.TagBytes(field_number, wire_format.WIRETYPE_VARINT)
    tag_len = len(tag_bytes)
    def DecodeRepeatedField(
        buffer, pos, end, message, field_dict, current_depth=0
    ):
      """Decode serialized repeated enum to its value and a new position.

      Args:
        buffer: memoryview of the serialized bytes.
        pos: int, position in the memory view to start at.
        end: int, end position of serialized data
        message: Message object to store unknown fields in
        field_dict: Map[Descriptor, Any] to store decoded values in.

      Returns:
        int, new position in serialized data.
      """
      del current_depth  # unused
      value = field_dict.get(key)
      if value is None:
        value = field_dict.setdefault(key, new_default(message))
      while 1:
        (element, new_pos) = _DecodeSignedVarint32(buffer, pos)
        # pylint: disable=protected-access
        if element in enum_type.values_by_number:
          value.append(element)
        else:
          if not message._unknown_fields:
            message._unknown_fields = []
          message._unknown_fields.append(
              (tag_bytes, buffer[pos:new_pos].tobytes()))
        # pylint: enable=protected-access
        # Predict that the next tag is another copy of the same repeated
        # field.
        pos = new_pos + tag_len
        if buffer[new_pos:pos] != tag_bytes or new_pos >= end:
          # Prediction failed.  Return.
          if new_pos > end:
            raise _DecodeError('Truncated message.')
          return new_pos

    return DecodeRepeatedField
  else:

    def DecodeField(buffer, pos, end, message, field_dict, current_depth=0):
      """Decode serialized repeated enum to its value and a new position.

      Args:
        buffer: memoryview of the serialized bytes.
        pos: int, position in the memory view to start at.
        end: int, end position of serialized data
        message: Message object to store unknown fields in
        field_dict: Map[Descriptor, Any] to store decoded values in.

      Returns:
        int, new position in serialized data.
      """
      del current_depth  # unused
      value_start_pos = pos
      (enum_value, pos) = _DecodeSignedVarint32(buffer, pos)
      if pos > end:
        raise _DecodeError('Truncated message.')
      if clear_if_default and IsDefaultScalarValue(enum_value):
        field_dict.pop(key, None)
        return pos
      # pylint: disable=protected-access
      if enum_value in enum_type.values_by_number:
        field_dict[key] = enum_value
      else:
        if not message._unknown_fields:
          message._unknown_fields = []
        tag_bytes = encoder.TagBytes(field_number,
                                     wire_format.WIRETYPE_VARINT)
        message._unknown_fields.append(
            (tag_bytes, buffer[value_start_pos:pos].tobytes()))
        # pylint: enable=protected-access
      return pos

    return DecodeField


# --------------------------------------------------------------------


Int32Decoder = _SimpleDecoder(
    wire_format.WIRETYPE_VARINT, _DecodeSignedVarint32)

Int64Decoder = _SimpleDecoder(
    wire_format.WIRETYPE_VARINT, _DecodeSignedVarint)

UInt32Decoder = _SimpleDecoder(wire_format.WIRETYPE_VARINT, _DecodeVarint32)
UInt64Decoder = _SimpleDecoder(wire_format.WIRETYPE_VARINT, _DecodeVarint)

SInt32Decoder = _ModifiedDecoder(
    wire_format.WIRETYPE_VARINT, _DecodeVarint32, wire_format.ZigZagDecode)
SInt64Decoder = _ModifiedDecoder(
    wire_format.WIRETYPE_VARINT, _DecodeVarint, wire_format.ZigZagDecode)

# Note that Python conveniently guarantees that when using the '<' prefix on
# formats, they will also have the same size across all platforms (as opposed
# to without the prefix, where their sizes depend on the C compiler's basic
# type sizes).
Fixed32Decoder  = _StructPackDecoder(wire_format.WIRETYPE_FIXED32, '<I')
Fixed64Decoder  = _StructPackDecoder(wire_format.WIRETYPE_FIXED64, '<Q')
SFixed32Decoder = _StructPackDecoder(wire_format.WIRETYPE_FIXED32, '<i')
SFixed64Decoder = _StructPackDecoder(wire_format.WIRETYPE_FIXED64, '<q')
FloatDecoder = _FloatDecoder()
DoubleDecoder = _DoubleDecoder()

BoolDecoder = _ModifiedDecoder(
    wire_format.WIRETYPE_VARINT, _DecodeVarint, bool)


def StringDecoder(field_number, is_repeated, is_packed, key, new_default,
                  clear_if_default=False):
  """Returns a decoder for a string field."""

  local_DecodeVarint = _DecodeVarint

  def _ConvertToUnicode(memview):
    """Convert byte to unicode."""
    byte_str = memview.tobytes()
    try:
      value = str(byte_str, 'utf-8')
    except UnicodeDecodeError as e:
      # add more information to the error message and re-raise it.
      e.reason = '%s in field: %s' % (e, key.full_name)
      raise

    return value

  assert not is_packed
  if is_repeated:
    tag_bytes = encoder.TagBytes(field_number,
                                 wire_format.WIRETYPE_LENGTH_DELIMITED)
    tag_len = len(tag_bytes)
    def DecodeRepeatedField(
        buffer, pos, end, message, field_dict, current_depth=0
    ):
      del current_depth  # unused
      value = field_dict.get(key)
      if value is None:
        value = field_dict.setdefault(key, new_default(message))
      while 1:
        (size, pos) = local_DecodeVarint(buffer, pos)
        new_pos = pos + size
        if new_pos > end:
          raise _DecodeError('Truncated string.')
        value.append(_ConvertToUnicode(buffer[pos:new_pos]))
        # Predict that the next tag is another copy of the same repeated field.
        pos = new_pos + tag_len
        if buffer[new_pos:pos] != tag_bytes or new_pos == end:
          # Prediction failed.  Return.
          return new_pos

    return DecodeRepeatedField
  else:

    def DecodeField(buffer, pos, end, message, field_dict, current_depth=0):
      del current_depth  # unused
      (size, pos) = local_DecodeVarint(buffer, pos)
      new_pos = pos + size
      if new_pos > end:
        raise _DecodeError('Truncated string.')
      if clear_if_default and IsDefaultScalarValue(size):
        field_dict.pop(key, None)
      else:
        field_dict[key] = _ConvertToUnicode(buffer[pos:new_pos])
      return new_pos

    return DecodeField


def BytesDecoder(field_number, is_repeated, is_packed, key, new_default,
                 clear_if_default=False):
  """Returns a decoder for a bytes field."""

  local_DecodeVarint = _DecodeVarint

  assert not is_packed
  if is_repeated:
    tag_bytes = encoder.TagBytes(field_number,
                                 wire_format.WIRETYPE_LENGTH_DELIMITED)
    tag_len = len(tag_bytes)
    def DecodeRepeatedField(
        buffer, pos, end, message, field_dict, current_depth=0
    ):
      del current_depth  # unused
      value = field_dict.get(key)
      if value is None:
        value = field_dict.setdefault(key, new_default(message))
      while 1:
        (size, pos) = local_DecodeVarint(buffer, pos)
        new_pos = pos + size
        if new_pos > end:
          raise _DecodeError('Truncated string.')
        value.append(buffer[pos:new_pos].tobytes())
        # Predict that the next tag is another copy of the same repeated field.
        pos = new_pos + tag_len
        if buffer[new_pos:pos] != tag_bytes or new_pos == end:
          # Prediction failed.  Return.
          return new_pos

    return DecodeRepeatedField
  else:

    def DecodeField(buffer, pos, end, message, field_dict, current_depth=0):
      del current_depth  # unused
      (size, pos) = local_DecodeVarint(buffer, pos)
      new_pos = pos + size
      if new_pos > end:
        raise _DecodeError('Truncated string.')
      if clear_if_default and IsDefaultScalarValue(size):
        field_dict.pop(key, None)
      else:
        field_dict[key] = buffer[pos:new_pos].tobytes()
      return new_pos

    return DecodeField


def GroupDecoder(field_number, is_repeated, is_packed, key, new_default):
  """Returns a decoder for a group field."""

  end_tag_bytes = encoder.TagBytes(field_number,
                                   wire_format.WIRETYPE_END_GROUP)
  end_tag_len = len(end_tag_bytes)

  assert not is_packed
  if is_repeated:
    tag_bytes = encoder.TagBytes(field_number,
                                 wire_format.WIRETYPE_START_GROUP)
    tag_len = len(tag_bytes)
    def DecodeRepeatedField(
        buffer, pos, end, message, field_dict, current_depth=0
    ):
      value = field_dict.get(key)
      if value is None:
        value = field_dict.setdefault(key, new_default(message))
      while 1:
        value = field_dict.get(key)
        if value is None:
          value = field_dict.setdefault(key, new_default(message))
        # Read sub-message.
        current_depth += 1
        if current_depth > _recursion_limit:
          raise _DecodeError(
              'Error parsing message: too many levels of nesting.'
          )
        pos = value.add()._InternalParse(buffer, pos, end, current_depth)
        current_depth -= 1
        # Read end tag.
        new_pos = pos+end_tag_len
        if buffer[pos:new_pos] != end_tag_bytes or new_pos > end:
          raise _DecodeError('Missing group end tag.')
        # Predict that the next tag is another copy of the same repeated field.
        pos = new_pos + tag_len
        if buffer[new_pos:pos] != tag_bytes or new_pos == end:
          # Prediction failed.  Return.
          return new_pos

    return DecodeRepeatedField
  else:

    def DecodeField(buffer, pos, end, message, field_dict, current_depth=0):
      value = field_dict.get(key)
      if value is None:
        value = field_dict.setdefault(key, new_default(message))
      # Read sub-message.
      current_depth += 1
      if current_depth > _recursion_limit:
        raise _DecodeError('Error parsing message: too many levels of nesting.')
      pos = value._InternalParse(buffer, pos, end, current_depth)
      current_depth -= 1
      # Read end tag.
      new_pos = pos+end_tag_len
      if buffer[pos:new_pos] != end_tag_bytes or new_pos > end:
        raise _DecodeError('Missing group end tag.')
      return new_pos

    return DecodeField


def MessageDecoder(field_number, is_repeated, is_packed, key, new_default):
  """Returns a decoder for a message field."""

  local_DecodeVarint = _DecodeVarint

  assert not is_packed
  if is_repeated:
    tag_bytes = encoder.TagBytes(field_number,
                                 wire_format.WIRETYPE_LENGTH_DELIMITED)
    tag_len = len(tag_bytes)
    def DecodeRepeatedField(
        buffer, pos, end, message, field_dict, current_depth=0
    ):
      value = field_dict.get(key)
      if value is None:
        value = field_dict.setdefault(key, new_default(message))
      while 1:
        # Read length.
        (size, pos) = local_DecodeVarint(buffer, pos)
        new_pos = pos + size
        if new_pos > end:
          raise _DecodeError('Truncated message.')
        # Read sub-message.
        current_depth += 1
        if current_depth > _recursion_limit:
          raise _DecodeError(
              'Error parsing message: too many levels of nesting.'
          )
        if (
            value.add()._InternalParse(buffer, pos, new_pos, current_depth)
            != new_pos
        ):
          # The only reason _InternalParse would return early is if it
          # encountered an end-group tag.
          raise _DecodeError('Unexpected end-group tag.')
        current_depth -= 1
        # Predict that the next tag is another copy of the same repeated field.
        pos = new_pos + tag_len
        if buffer[new_pos:pos] != tag_bytes or new_pos == end:
          # Prediction failed.  Return.
          return new_pos

    return DecodeRepeatedField
  else:

    def DecodeField(buffer, pos, end, message, field_dict, current_depth=0):
      value = field_dict.get(key)
      if value is None:
        value = field_dict.setdefault(key, new_default(message))
      # Read length.
      (size, pos) = local_DecodeVarint(buffer, pos)
      new_pos = pos + size
      if new_pos > end:
        raise _DecodeError('Truncated message.')
      # Read sub-message.
      current_depth += 1
      if current_depth > _recursion_limit:
        raise _DecodeError('Error parsing message: too many levels of nesting.')
      if value._InternalParse(buffer, pos, new_pos, current_depth) != new_pos:
        # The only reason _InternalParse would return early is if it encountered
        # an end-group tag.
        raise _DecodeError('Unexpected end-group tag.')
      current_depth -= 1
      return new_pos

    return DecodeField


# --------------------------------------------------------------------

MESSAGE_SET_ITEM_TAG = encoder.TagBytes(1, wire_format.WIRETYPE_START_GROUP)

def MessageSetItemDecoder(descriptor):
  """Returns a decoder for a MessageSet item.

  The parameter is the message Descriptor.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """

  type_id_tag_bytes = encoder.TagBytes(2, wire_format.WIRETYPE_VARINT)
  message_tag_bytes = encoder.TagBytes(3, wire_format.WIRETYPE_LENGTH_DELIMITED)
  item_end_tag_bytes = encoder.TagBytes(1, wire_format.WIRETYPE_END_GROUP)

  local_ReadTag = ReadTag
  local_DecodeVarint = _DecodeVarint

  def DecodeItem(buffer, pos, end, message, field_dict):
    """Decode serialized message set to its value and new position.

    Args:
      buffer: memoryview of the serialized bytes.
      pos: int, position in the memory view to start at.
      end: int, end position of serialized data
      message: Message object to store unknown fields in
      field_dict: Map[Descriptor, Any] to store decoded values in.

    Returns:
      int, new position in serialized data.
    """
    message_set_item_start = pos
    type_id = -1
    message_start = -1
    message_end = -1

    # Technically, type_id and message can appear in any order, so we need
    # a little loop here.
    while 1:
      (tag_bytes, pos) = local_ReadTag(buffer, pos)
      if tag_bytes == type_id_tag_bytes:
        (type_id, pos) = local_DecodeVarint(buffer, pos)
      elif tag_bytes == message_tag_bytes:
        (size, message_start) = local_DecodeVarint(buffer, pos)
        pos = message_end = message_start + size
      elif tag_bytes == item_end_tag_bytes:
        break
      else:
        field_number, wire_type = DecodeTag(tag_bytes)
        _, pos = _DecodeUnknownField(buffer, pos, end, field_number, wire_type)
        if pos == -1:
          raise _DecodeError('Unexpected end-group tag.')

    if pos > end:
      raise _DecodeError('Truncated message.')

    if type_id == -1:
      raise _DecodeError('MessageSet item missing type_id.')
    if message_start == -1:
      raise _DecodeError('MessageSet item missing message.')

    extension = message.Extensions._FindExtensionByNumber(type_id)
    # pylint: disable=protected-access
    if extension is not None:
      value = field_dict.get(extension)
      if value is None:
        message_type = extension.message_type
        if not hasattr(message_type, '_concrete_class'):
          message_factory.GetMessageClass(message_type)
        value = field_dict.setdefault(
            extension, message_type._concrete_class())
      if value._InternalParse(buffer, message_start,message_end) != message_end:
        # The only reason _InternalParse would return early is if it encountered
        # an end-group tag.
        raise _DecodeError('Unexpected end-group tag.')
    else:
      if not message._unknown_fields:
        message._unknown_fields = []
      message._unknown_fields.append(
          (MESSAGE_SET_ITEM_TAG, buffer[message_set_item_start:pos].tobytes()))
      # pylint: enable=protected-access

    return pos

  return DecodeItem


def UnknownMessageSetItemDecoder():
  """Returns a decoder for a Unknown MessageSet item."""

  type_id_tag_bytes = encoder.TagBytes(2, wire_format.WIRETYPE_VARINT)
  message_tag_bytes = encoder.TagBytes(3, wire_format.WIRETYPE_LENGTH_DELIMITED)
  item_end_tag_bytes = encoder.TagBytes(1, wire_format.WIRETYPE_END_GROUP)

  def DecodeUnknownItem(buffer):
    pos = 0
    end = len(buffer)
    message_start = -1
    message_end = -1
    while 1:
      (tag_bytes, pos) = ReadTag(buffer, pos)
      if tag_bytes == type_id_tag_bytes:
        (type_id, pos) = _DecodeVarint(buffer, pos)
      elif tag_bytes == message_tag_bytes:
        (size, message_start) = _DecodeVarint(buffer, pos)
        pos = message_end = message_start + size
      elif tag_bytes == item_end_tag_bytes:
        break
      else:
        field_number, wire_type = DecodeTag(tag_bytes)
        _, pos = _DecodeUnknownField(buffer, pos, end, field_number, wire_type)
        if pos == -1:
          raise _DecodeError('Unexpected end-group tag.')

    if pos > end:
      raise _DecodeError('Truncated message.')

    if type_id == -1:
      raise _DecodeError('MessageSet item missing type_id.')
    if message_start == -1:
      raise _DecodeError('MessageSet item missing message.')

    return (type_id, buffer[message_start:message_end].tobytes())

  return DecodeUnknownItem

# --------------------------------------------------------------------

def MapDecoder(field_descriptor, new_default, is_message_map):
  """Returns a decoder for a map field."""

  key = field_descriptor
  tag_bytes = encoder.TagBytes(field_descriptor.number,
                               wire_format.WIRETYPE_LENGTH_DELIMITED)
  tag_len = len(tag_bytes)
  local_DecodeVarint = _DecodeVarint
  # Can't read _concrete_class yet; might not be initialized.
  message_type = field_descriptor.message_type

  def DecodeMap(buffer, pos, end, message, field_dict, current_depth=0):
    del current_depth  # Unused.
    submsg = message_type._concrete_class()
    value = field_dict.get(key)
    if value is None:
      value = field_dict.setdefault(key, new_default(message))
    while 1:
      # Read length.
      (size, pos) = local_DecodeVarint(buffer, pos)
      new_pos = pos + size
      if new_pos > end:
        raise _DecodeError('Truncated message.')
      # Read sub-message.
      submsg.Clear()
      if submsg._InternalParse(buffer, pos, new_pos) != new_pos:
        # The only reason _InternalParse would return early is if it
        # encountered an end-group tag.
        raise _DecodeError('Unexpected end-group tag.')

      if is_message_map:
        value[submsg.key].CopyFrom(submsg.value)
      else:
        value[submsg.key] = submsg.value

      # Predict that the next tag is another copy of the same repeated field.
      pos = new_pos + tag_len
      if buffer[new_pos:pos] != tag_bytes or new_pos == end:
        # Prediction failed.  Return.
        return new_pos

  return DecodeMap


def _DecodeFixed64(buffer, pos):
  """Decode a fixed64."""
  new_pos = pos + 8
  return (struct.unpack('<Q', buffer[pos:new_pos])[0], new_pos)


def _DecodeFixed32(buffer, pos):
  """Decode a fixed32."""

  new_pos = pos + 4
  return (struct.unpack('<I', buffer[pos:new_pos])[0], new_pos)
DEFAULT_RECURSION_LIMIT = 100
_recursion_limit = DEFAULT_RECURSION_LIMIT


def SetRecursionLimit(new_limit):
  global _recursion_limit
  _recursion_limit = new_limit


def _DecodeUnknownFieldSet(buffer, pos, end_pos=None, current_depth=0):
  """Decode UnknownFieldSet.  Returns the UnknownFieldSet and new position."""

  unknown_field_set = containers.UnknownFieldSet()
  while end_pos is None or pos < end_pos:
    (tag_bytes, pos) = ReadTag(buffer, pos)
    (tag, _) = _DecodeVarint(tag_bytes, 0)
    field_number, wire_type = wire_format.UnpackTag(tag)
    if wire_type == wire_format.WIRETYPE_END_GROUP:
      break
    (data, pos) = _DecodeUnknownField(
        buffer, pos, end_pos, field_number, wire_type, current_depth
    )
    # pylint: disable=protected-access
    unknown_field_set._add(field_number, wire_type, data)

  return (unknown_field_set, pos)


def _DecodeUnknownField(
    buffer, pos, end_pos, field_number, wire_type, current_depth=0
):
  """Decode a unknown field.  Returns the UnknownField and new position."""

  if wire_type == wire_format.WIRETYPE_VARINT:
    (data, pos) = _DecodeVarint(buffer, pos)
  elif wire_type == wire_format.WIRETYPE_FIXED64:
    (data, pos) = _DecodeFixed64(buffer, pos)
  elif wire_type == wire_format.WIRETYPE_FIXED32:
    (data, pos) = _DecodeFixed32(buffer, pos)
  elif wire_type == wire_format.WIRETYPE_LENGTH_DELIMITED:
    (size, pos) = _DecodeVarint(buffer, pos)
    data = buffer[pos:pos+size].tobytes()
    pos += size
  elif wire_type == wire_format.WIRETYPE_START_GROUP:
    end_tag_bytes = encoder.TagBytes(
        field_number, wire_format.WIRETYPE_END_GROUP
    )
    current_depth += 1
    if current_depth >= _recursion_limit:
      raise _DecodeError('Error parsing message: too many levels of nesting.')
    data, pos = _DecodeUnknownFieldSet(buffer, pos, end_pos, current_depth)
    current_depth -= 1
    # Check end tag.
    if buffer[pos - len(end_tag_bytes) : pos] != end_tag_bytes:
      raise _DecodeError('Missing group end tag.')
  elif wire_type == wire_format.WIRETYPE_END_GROUP:
    return (0, -1)
  else:
    raise _DecodeError('Wrong wire type in tag.')

  if pos > end_pos:
    raise _DecodeError('Truncated message.')

  return (data, pos)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\runtime.py ===
"""The runtime functions and state used by compiled templates."""

import functools
import sys
import typing as t
from collections import abc
from itertools import chain

from markupsafe import escape  # noqa: F401
from markupsafe import Markup
from markupsafe import soft_str

from .async_utils import auto_aiter
from .async_utils import auto_await  # noqa: F401
from .exceptions import TemplateNotFound  # noqa: F401
from .exceptions import TemplateRuntimeError  # noqa: F401
from .exceptions import UndefinedError
from .nodes import EvalContext
from .utils import _PassArg
from .utils import concat
from .utils import internalcode
from .utils import missing
from .utils import Namespace  # noqa: F401
from .utils import object_type_repr
from .utils import pass_eval_context

V = t.TypeVar("V")
F = t.TypeVar("F", bound=t.Callable[..., t.Any])

if t.TYPE_CHECKING:
    import logging

    import typing_extensions as te

    from .environment import Environment

    class LoopRenderFunc(te.Protocol):
        def __call__(
            self,
            reciter: t.Iterable[V],
            loop_render_func: "LoopRenderFunc",
            depth: int = 0,
        ) -> str: ...


# these variables are exported to the template runtime
exported = [
    "LoopContext",
    "TemplateReference",
    "Macro",
    "Markup",
    "TemplateRuntimeError",
    "missing",
    "escape",
    "markup_join",
    "str_join",
    "identity",
    "TemplateNotFound",
    "Namespace",
    "Undefined",
    "internalcode",
]
async_exported = [
    "AsyncLoopContext",
    "auto_aiter",
    "auto_await",
]


def identity(x: V) -> V:
    """Returns its argument. Useful for certain things in the
    environment.
    """
    return x


def markup_join(seq: t.Iterable[t.Any]) -> str:
    """Concatenation that escapes if necessary and converts to string."""
    buf = []
    iterator = map(soft_str, seq)
    for arg in iterator:
        buf.append(arg)
        if hasattr(arg, "__html__"):
            return Markup("").join(chain(buf, iterator))
    return concat(buf)


def str_join(seq: t.Iterable[t.Any]) -> str:
    """Simple args to string conversion and concatenation."""
    return concat(map(str, seq))


def new_context(
    environment: "Environment",
    template_name: t.Optional[str],
    blocks: t.Dict[str, t.Callable[["Context"], t.Iterator[str]]],
    vars: t.Optional[t.Dict[str, t.Any]] = None,
    shared: bool = False,
    globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    locals: t.Optional[t.Mapping[str, t.Any]] = None,
) -> "Context":
    """Internal helper for context creation."""
    if vars is None:
        vars = {}
    if shared:
        parent = vars
    else:
        parent = dict(globals or (), **vars)
    if locals:
        # if the parent is shared a copy should be created because
        # we don't want to modify the dict passed
        if shared:
            parent = dict(parent)
        for key, value in locals.items():
            if value is not missing:
                parent[key] = value
    return environment.context_class(
        environment, parent, template_name, blocks, globals=globals
    )


class TemplateReference:
    """The `self` in templates."""

    def __init__(self, context: "Context") -> None:
        self.__context = context

    def __getitem__(self, name: str) -> t.Any:
        blocks = self.__context.blocks[name]
        return BlockReference(name, self.__context, blocks, 0)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.__context.name!r}>"


def _dict_method_all(dict_method: F) -> F:
    @functools.wraps(dict_method)
    def f_all(self: "Context") -> t.Any:
        return dict_method(self.get_all())

    return t.cast(F, f_all)


@abc.Mapping.register
class Context:
    """The template context holds the variables of a template.  It stores the
    values passed to the template and also the names the template exports.
    Creating instances is neither supported nor useful as it's created
    automatically at various stages of the template evaluation and should not
    be created by hand.

    The context is immutable.  Modifications on :attr:`parent` **must not**
    happen and modifications on :attr:`vars` are allowed from generated
    template code only.  Template filters and global functions marked as
    :func:`pass_context` get the active context passed as first argument
    and are allowed to access the context read-only.

    The template context supports read only dict operations (`get`,
    `keys`, `values`, `items`, `iterkeys`, `itervalues`, `iteritems`,
    `__getitem__`, `__contains__`).  Additionally there is a :meth:`resolve`
    method that doesn't fail with a `KeyError` but returns an
    :class:`Undefined` object for missing variables.
    """

    def __init__(
        self,
        environment: "Environment",
        parent: t.Dict[str, t.Any],
        name: t.Optional[str],
        blocks: t.Dict[str, t.Callable[["Context"], t.Iterator[str]]],
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ):
        self.parent = parent
        self.vars: t.Dict[str, t.Any] = {}
        self.environment: Environment = environment
        self.eval_ctx = EvalContext(self.environment, name)
        self.exported_vars: t.Set[str] = set()
        self.name = name
        self.globals_keys = set() if globals is None else set(globals)

        # create the initial mapping of blocks.  Whenever template inheritance
        # takes place the runtime will update this mapping with the new blocks
        # from the template.
        self.blocks = {k: [v] for k, v in blocks.items()}

    def super(
        self, name: str, current: t.Callable[["Context"], t.Iterator[str]]
    ) -> t.Union["BlockReference", "Undefined"]:
        """Render a parent block."""
        try:
            blocks = self.blocks[name]
            index = blocks.index(current) + 1
            blocks[index]
        except LookupError:
            return self.environment.undefined(
                f"there is no parent block called {name!r}.", name="super"
            )
        return BlockReference(name, self, blocks, index)

    def get(self, key: str, default: t.Any = None) -> t.Any:
        """Look up a variable by name, or return a default if the key is
        not found.

        :param key: The variable name to look up.
        :param default: The value to return if the key is not found.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def resolve(self, key: str) -> t.Union[t.Any, "Undefined"]:
        """Look up a variable by name, or return an :class:`Undefined`
        object if the key is not found.

        If you need to add custom behavior, override
        :meth:`resolve_or_missing`, not this method. The various lookup
        functions use that method, not this one.

        :param key: The variable name to look up.
        """
        rv = self.resolve_or_missing(key)

        if rv is missing:
            return self.environment.undefined(name=key)

        return rv

    def resolve_or_missing(self, key: str) -> t.Any:
        """Look up a variable by name, or return a ``missing`` sentinel
        if the key is not found.

        Override this method to add custom lookup behavior.
        :meth:`resolve`, :meth:`get`, and :meth:`__getitem__` use this
        method. Don't call this method directly.

        :param key: The variable name to look up.
        """
        if key in self.vars:
            return self.vars[key]

        if key in self.parent:
            return self.parent[key]

        return missing

    def get_exported(self) -> t.Dict[str, t.Any]:
        """Get a new dict with the exported variables."""
        return {k: self.vars[k] for k in self.exported_vars}

    def get_all(self) -> t.Dict[str, t.Any]:
        """Return the complete context as dict including the exported
        variables.  For optimizations reasons this might not return an
        actual copy so be careful with using it.
        """
        if not self.vars:
            return self.parent
        if not self.parent:
            return self.vars
        return dict(self.parent, **self.vars)

    @internalcode
    def call(
        __self,
        __obj: t.Callable[..., t.Any],
        *args: t.Any,
        **kwargs: t.Any,  # noqa: B902
    ) -> t.Union[t.Any, "Undefined"]:
        """Call the callable with the arguments and keyword arguments
        provided but inject the active context or environment as first
        argument if the callable has :func:`pass_context` or
        :func:`pass_environment`.
        """
        if __debug__:
            __traceback_hide__ = True  # noqa

        # Allow callable classes to take a context
        if (
            hasattr(__obj, "__call__")  # noqa: B004
            and _PassArg.from_obj(__obj.__call__) is not None
        ):
            __obj = __obj.__call__

        pass_arg = _PassArg.from_obj(__obj)

        if pass_arg is _PassArg.context:
            # the active context should have access to variables set in
            # loops and blocks without mutating the context itself
            if kwargs.get("_loop_vars"):
                __self = __self.derived(kwargs["_loop_vars"])
            if kwargs.get("_block_vars"):
                __self = __self.derived(kwargs["_block_vars"])
            args = (__self,) + args
        elif pass_arg is _PassArg.eval_context:
            args = (__self.eval_ctx,) + args
        elif pass_arg is _PassArg.environment:
            args = (__self.environment,) + args

        kwargs.pop("_block_vars", None)
        kwargs.pop("_loop_vars", None)

        try:
            return __obj(*args, **kwargs)
        except StopIteration:
            return __self.environment.undefined(
                "value was undefined because a callable raised a"
                " StopIteration exception"
            )

    def derived(self, locals: t.Optional[t.Dict[str, t.Any]] = None) -> "Context":
        """Internal helper function to create a derived context.  This is
        used in situations where the system needs a new context in the same
        template that is independent.
        """
        context = new_context(
            self.environment, self.name, {}, self.get_all(), True, None, locals
        )
        context.eval_ctx = self.eval_ctx
        context.blocks.update((k, list(v)) for k, v in self.blocks.items())
        return context

    keys = _dict_method_all(dict.keys)
    values = _dict_method_all(dict.values)
    items = _dict_method_all(dict.items)

    def __contains__(self, name: str) -> bool:
        return name in self.vars or name in self.parent

    def __getitem__(self, key: str) -> t.Any:
        """Look up a variable by name with ``[]`` syntax, or raise a
        ``KeyError`` if the key is not found.
        """
        item = self.resolve_or_missing(key)

        if item is missing:
            raise KeyError(key)

        return item

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.get_all()!r} of {self.name!r}>"


class BlockReference:
    """One block on a template reference."""

    def __init__(
        self,
        name: str,
        context: "Context",
        stack: t.List[t.Callable[["Context"], t.Iterator[str]]],
        depth: int,
    ) -> None:
        self.name = name
        self._context = context
        self._stack = stack
        self._depth = depth

    @property
    def super(self) -> t.Union["BlockReference", "Undefined"]:
        """Super the block."""
        if self._depth + 1 >= len(self._stack):
            return self._context.environment.undefined(
                f"there is no parent block called {self.name!r}.", name="super"
            )
        return BlockReference(self.name, self._context, self._stack, self._depth + 1)

    @internalcode
    async def _async_call(self) -> str:
        rv = self._context.environment.concat(  # type: ignore
            [x async for x in self._stack[self._depth](self._context)]  # type: ignore
        )

        if self._context.eval_ctx.autoescape:
            return Markup(rv)

        return rv

    @internalcode
    def __call__(self) -> str:
        if self._context.environment.is_async:
            return self._async_call()  # type: ignore

        rv = self._context.environment.concat(  # type: ignore
            self._stack[self._depth](self._context)
        )

        if self._context.eval_ctx.autoescape:
            return Markup(rv)

        return rv


class LoopContext:
    """A wrapper iterable for dynamic ``for`` loops, with information
    about the loop and iteration.
    """

    #: Current iteration of the loop, starting at 0.
    index0 = -1

    _length: t.Optional[int] = None
    _after: t.Any = missing
    _current: t.Any = missing
    _before: t.Any = missing
    _last_changed_value: t.Any = missing

    def __init__(
        self,
        iterable: t.Iterable[V],
        undefined: t.Type["Undefined"],
        recurse: t.Optional["LoopRenderFunc"] = None,
        depth0: int = 0,
    ) -> None:
        """
        :param iterable: Iterable to wrap.
        :param undefined: :class:`Undefined` class to use for next and
            previous items.
        :param recurse: The function to render the loop body when the
            loop is marked recursive.
        :param depth0: Incremented when looping recursively.
        """
        self._iterable = iterable
        self._iterator = self._to_iterator(iterable)
        self._undefined = undefined
        self._recurse = recurse
        #: How many levels deep a recursive loop currently is, starting at 0.
        self.depth0 = depth0

    @staticmethod
    def _to_iterator(iterable: t.Iterable[V]) -> t.Iterator[V]:
        return iter(iterable)

    @property
    def length(self) -> int:
        """Length of the iterable.

        If the iterable is a generator or otherwise does not have a
        size, it is eagerly evaluated to get a size.
        """
        if self._length is not None:
            return self._length

        try:
            self._length = len(self._iterable)  # type: ignore
        except TypeError:
            iterable = list(self._iterator)
            self._iterator = self._to_iterator(iterable)
            self._length = len(iterable) + self.index + (self._after is not missing)

        return self._length

    def __len__(self) -> int:
        return self.length

    @property
    def depth(self) -> int:
        """How many levels deep a recursive loop currently is, starting at 1."""
        return self.depth0 + 1

    @property
    def index(self) -> int:
        """Current iteration of the loop, starting at 1."""
        return self.index0 + 1

    @property
    def revindex0(self) -> int:
        """Number of iterations from the end of the loop, ending at 0.

        Requires calculating :attr:`length`.
        """
        return self.length - self.index

    @property
    def revindex(self) -> int:
        """Number of iterations from the end of the loop, ending at 1.

        Requires calculating :attr:`length`.
        """
        return self.length - self.index0

    @property
    def first(self) -> bool:
        """Whether this is the first iteration of the loop."""
        return self.index0 == 0

    def _peek_next(self) -> t.Any:
        """Return the next element in the iterable, or :data:`missing`
        if the iterable is exhausted. Only peeks one item ahead, caching
        the result in :attr:`_last` for use in subsequent checks. The
        cache is reset when :meth:`__next__` is called.
        """
        if self._after is not missing:
            return self._after

        self._after = next(self._iterator, missing)
        return self._after

    @property
    def last(self) -> bool:
        """Whether this is the last iteration of the loop.

        Causes the iterable to advance early. See
        :func:`itertools.groupby` for issues this can cause.
        The :func:`groupby` filter avoids that issue.
        """
        return self._peek_next() is missing

    @property
    def previtem(self) -> t.Union[t.Any, "Undefined"]:
        """The item in the previous iteration. Undefined during the
        first iteration.
        """
        if self.first:
            return self._undefined("there is no previous item")

        return self._before

    @property
    def nextitem(self) -> t.Union[t.Any, "Undefined"]:
        """The item in the next iteration. Undefined during the last
        iteration.

        Causes the iterable to advance early. See
        :func:`itertools.groupby` for issues this can cause.
        The :func:`jinja-filters.groupby` filter avoids that issue.
        """
        rv = self._peek_next()

        if rv is missing:
            return self._undefined("there is no next item")

        return rv

    def cycle(self, *args: V) -> V:
        """Return a value from the given args, cycling through based on
        the current :attr:`index0`.

        :param args: One or more values to cycle through.
        """
        if not args:
            raise TypeError("no items for cycling given")

        return args[self.index0 % len(args)]

    def changed(self, *value: t.Any) -> bool:
        """Return ``True`` if previously called with a different value
        (including when called for the first time).

        :param value: One or more values to compare to the last call.
        """
        if self._last_changed_value != value:
            self._last_changed_value = value
            return True

        return False

    def __iter__(self) -> "LoopContext":
        return self

    def __next__(self) -> t.Tuple[t.Any, "LoopContext"]:
        if self._after is not missing:
            rv = self._after
            self._after = missing
        else:
            rv = next(self._iterator)

        self.index0 += 1
        self._before = self._current
        self._current = rv
        return rv, self

    @internalcode
    def __call__(self, iterable: t.Iterable[V]) -> str:
        """When iterating over nested data, render the body of the loop
        recursively with the given inner iterable data.

        The loop must have the ``recursive`` marker for this to work.
        """
        if self._recurse is None:
            raise TypeError(
                "The loop must have the 'recursive' marker to be called recursively."
            )

        return self._recurse(iterable, self._recurse, depth=self.depth)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.index}/{self.length}>"


class AsyncLoopContext(LoopContext):
    _iterator: t.AsyncIterator[t.Any]  # type: ignore

    @staticmethod
    def _to_iterator(  # type: ignore
        iterable: t.Union[t.Iterable[V], t.AsyncIterable[V]],
    ) -> t.AsyncIterator[V]:
        return auto_aiter(iterable)

    @property
    async def length(self) -> int:  # type: ignore
        if self._length is not None:
            return self._length

        try:
            self._length = len(self._iterable)  # type: ignore
        except TypeError:
            iterable = [x async for x in self._iterator]
            self._iterator = self._to_iterator(iterable)
            self._length = len(iterable) + self.index + (self._after is not missing)

        return self._length

    @property
    async def revindex0(self) -> int:  # type: ignore
        return await self.length - self.index

    @property
    async def revindex(self) -> int:  # type: ignore
        return await self.length - self.index0

    async def _peek_next(self) -> t.Any:
        if self._after is not missing:
            return self._after

        try:
            self._after = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._after = missing

        return self._after

    @property
    async def last(self) -> bool:  # type: ignore
        return await self._peek_next() is missing

    @property
    async def nextitem(self) -> t.Union[t.Any, "Undefined"]:
        rv = await self._peek_next()

        if rv is missing:
            return self._undefined("there is no next item")

        return rv

    def __aiter__(self) -> "AsyncLoopContext":
        return self

    async def __anext__(self) -> t.Tuple[t.Any, "AsyncLoopContext"]:
        if self._after is not missing:
            rv = self._after
            self._after = missing
        else:
            rv = await self._iterator.__anext__()

        self.index0 += 1
        self._before = self._current
        self._current = rv
        return rv, self


class Macro:
    """Wraps a macro function."""

    def __init__(
        self,
        environment: "Environment",
        func: t.Callable[..., str],
        name: str,
        arguments: t.List[str],
        catch_kwargs: bool,
        catch_varargs: bool,
        caller: bool,
        default_autoescape: t.Optional[bool] = None,
    ):
        self._environment = environment
        self._func = func
        self._argument_count = len(arguments)
        self.name = name
        self.arguments = arguments
        self.catch_kwargs = catch_kwargs
        self.catch_varargs = catch_varargs
        self.caller = caller
        self.explicit_caller = "caller" in arguments

        if default_autoescape is None:
            if callable(environment.autoescape):
                default_autoescape = environment.autoescape(None)
            else:
                default_autoescape = environment.autoescape

        self._default_autoescape = default_autoescape

    @internalcode
    @pass_eval_context
    def __call__(self, *args: t.Any, **kwargs: t.Any) -> str:
        # This requires a bit of explanation,  In the past we used to
        # decide largely based on compile-time information if a macro is
        # safe or unsafe.  While there was a volatile mode it was largely
        # unused for deciding on escaping.  This turns out to be
        # problematic for macros because whether a macro is safe depends not
        # on the escape mode when it was defined, but rather when it was used.
        #
        # Because however we export macros from the module system and
        # there are historic callers that do not pass an eval context (and
        # will continue to not pass one), we need to perform an instance
        # check here.
        #
        # This is considered safe because an eval context is not a valid
        # argument to callables otherwise anyway.  Worst case here is
        # that if no eval context is passed we fall back to the compile
        # time autoescape flag.
        if args and isinstance(args[0], EvalContext):
            autoescape = args[0].autoescape
            args = args[1:]
        else:
            autoescape = self._default_autoescape

        # try to consume the positional arguments
        arguments = list(args[: self._argument_count])
        off = len(arguments)

        # For information why this is necessary refer to the handling
        # of caller in the `macro_body` handler in the compiler.
        found_caller = False

        # if the number of arguments consumed is not the number of
        # arguments expected we start filling in keyword arguments
        # and defaults.
        if off != self._argument_count:
            for name in self.arguments[len(arguments) :]:
                try:
                    value = kwargs.pop(name)
                except KeyError:
                    value = missing
                if name == "caller":
                    found_caller = True
                arguments.append(value)
        else:
            found_caller = self.explicit_caller

        # it's important that the order of these arguments does not change
        # if not also changed in the compiler's `function_scoping` method.
        # the order is caller, keyword arguments, positional arguments!
        if self.caller and not found_caller:
            caller = kwargs.pop("caller", None)
            if caller is None:
                caller = self._environment.undefined("No caller defined", name="caller")
            arguments.append(caller)

        if self.catch_kwargs:
            arguments.append(kwargs)
        elif kwargs:
            if "caller" in kwargs:
                raise TypeError(
                    f"macro {self.name!r} was invoked with two values for the special"
                    " caller argument. This is most likely a bug."
                )
            raise TypeError(
                f"macro {self.name!r} takes no keyword argument {next(iter(kwargs))!r}"
            )
        if self.catch_varargs:
            arguments.append(args[self._argument_count :])
        elif len(args) > self._argument_count:
            raise TypeError(
                f"macro {self.name!r} takes not more than"
                f" {len(self.arguments)} argument(s)"
            )

        return self._invoke(arguments, autoescape)

    async def _async_invoke(self, arguments: t.List[t.Any], autoescape: bool) -> str:
        rv = await self._func(*arguments)  # type: ignore

        if autoescape:
            return Markup(rv)

        return rv  # type: ignore

    def _invoke(self, arguments: t.List[t.Any], autoescape: bool) -> str:
        if self._environment.is_async:
            return self._async_invoke(arguments, autoescape)  # type: ignore

        rv = self._func(*arguments)

        if autoescape:
            rv = Markup(rv)

        return rv

    def __repr__(self) -> str:
        name = "anonymous" if self.name is None else repr(self.name)
        return f"<{type(self).__name__} {name}>"


class Undefined:
    """The default undefined type. This can be printed, iterated, and treated as
    a boolean. Any other operation will raise an :exc:`UndefinedError`.

    >>> foo = Undefined(name='foo')
    >>> str(foo)
    ''
    >>> not foo
    True
    >>> foo + 42
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    """

    __slots__ = (
        "_undefined_hint",
        "_undefined_obj",
        "_undefined_name",
        "_undefined_exception",
    )

    def __init__(
        self,
        hint: t.Optional[str] = None,
        obj: t.Any = missing,
        name: t.Optional[str] = None,
        exc: t.Type[TemplateRuntimeError] = UndefinedError,
    ) -> None:
        self._undefined_hint = hint
        self._undefined_obj = obj
        self._undefined_name = name
        self._undefined_exception = exc

    @property
    def _undefined_message(self) -> str:
        """Build a message about the undefined value based on how it was
        accessed.
        """
        if self._undefined_hint:
            return self._undefined_hint

        if self._undefined_obj is missing:
            return f"{self._undefined_name!r} is undefined"

        if not isinstance(self._undefined_name, str):
            return (
                f"{object_type_repr(self._undefined_obj)} has no"
                f" element {self._undefined_name!r}"
            )

        return (
            f"{object_type_repr(self._undefined_obj)!r} has no"
            f" attribute {self._undefined_name!r}"
        )

    @internalcode
    def _fail_with_undefined_error(
        self, *args: t.Any, **kwargs: t.Any
    ) -> "te.NoReturn":
        """Raise an :exc:`UndefinedError` when operations are performed
        on the undefined value.
        """
        raise self._undefined_exception(self._undefined_message)

    @internalcode
    def __getattr__(self, name: str) -> t.Any:
        # Raise AttributeError on requests for names that appear to be unimplemented
        # dunder methods to keep Python's internal protocol probing behaviors working
        # properly in cases where another exception type could cause unexpected or
        # difficult-to-diagnose failures.
        if name[:2] == "__" and name[-2:] == "__":
            raise AttributeError(name)

        return self._fail_with_undefined_error()

    __add__ = __radd__ = __sub__ = __rsub__ = _fail_with_undefined_error
    __mul__ = __rmul__ = __div__ = __rdiv__ = _fail_with_undefined_error
    __truediv__ = __rtruediv__ = _fail_with_undefined_error
    __floordiv__ = __rfloordiv__ = _fail_with_undefined_error
    __mod__ = __rmod__ = _fail_with_undefined_error
    __pos__ = __neg__ = _fail_with_undefined_error
    __call__ = __getitem__ = _fail_with_undefined_error
    __lt__ = __le__ = __gt__ = __ge__ = _fail_with_undefined_error
    __int__ = __float__ = __complex__ = _fail_with_undefined_error
    __pow__ = __rpow__ = _fail_with_undefined_error

    def __eq__(self, other: t.Any) -> bool:
        return type(self) is type(other)

    def __ne__(self, other: t.Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return id(type(self))

    def __str__(self) -> str:
        return ""

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> t.Iterator[t.Any]:
        yield from ()

    async def __aiter__(self) -> t.AsyncIterator[t.Any]:
        for _ in ():
            yield

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "Undefined"


def make_logging_undefined(
    logger: t.Optional["logging.Logger"] = None, base: t.Type[Undefined] = Undefined
) -> t.Type[Undefined]:
    """Given a logger object this returns a new undefined class that will
    log certain failures.  It will log iterations and printing.  If no
    logger is given a default logger is created.

    Example::

        logger = logging.getLogger(__name__)
        LoggingUndefined = make_logging_undefined(
            logger=logger,
            base=Undefined
        )

    .. versionadded:: 2.8

    :param logger: the logger to use.  If not provided, a default logger
                   is created.
    :param base: the base class to add logging functionality to.  This
                 defaults to :class:`Undefined`.
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler(sys.stderr))

    def _log_message(undef: Undefined) -> None:
        logger.warning("Template variable warning: %s", undef._undefined_message)

    class LoggingUndefined(base):  # type: ignore
        __slots__ = ()

        def _fail_with_undefined_error(  # type: ignore
            self, *args: t.Any, **kwargs: t.Any
        ) -> "te.NoReturn":
            try:
                super()._fail_with_undefined_error(*args, **kwargs)
            except self._undefined_exception as e:
                logger.error("Template variable error: %s", e)  # type: ignore
                raise e

        def __str__(self) -> str:
            _log_message(self)
            return super().__str__()  # type: ignore

        def __iter__(self) -> t.Iterator[t.Any]:
            _log_message(self)
            return super().__iter__()  # type: ignore

        def __bool__(self) -> bool:
            _log_message(self)
            return super().__bool__()  # type: ignore

    return LoggingUndefined


class ChainableUndefined(Undefined):
    """An undefined that is chainable, where both ``__getattr__`` and
    ``__getitem__`` return itself rather than raising an
    :exc:`UndefinedError`.

    >>> foo = ChainableUndefined(name='foo')
    >>> str(foo.bar['baz'])
    ''
    >>> foo.bar['baz'] + 42
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined

    .. versionadded:: 2.11.0
    """

    __slots__ = ()

    def __html__(self) -> str:
        return str(self)

    def __getattr__(self, name: str) -> "ChainableUndefined":
        # Raise AttributeError on requests for names that appear to be unimplemented
        # dunder methods to avoid confusing Python with truthy non-method objects that
        # do not implement the protocol being probed for. e.g., copy.copy(Undefined())
        # fails spectacularly if getattr(Undefined(), '__setstate__') returns an
        # Undefined object instead of raising AttributeError to signal that it does not
        # support that style of object initialization.
        if name[:2] == "__" and name[-2:] == "__":
            raise AttributeError(name)

        return self

    def __getitem__(self, _name: str) -> "ChainableUndefined":  # type: ignore[override]
        return self


class DebugUndefined(Undefined):
    """An undefined that returns the debug info when printed.

    >>> foo = DebugUndefined(name='foo')
    >>> str(foo)
    '{{ foo }}'
    >>> not foo
    True
    >>> foo + 42
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    """

    __slots__ = ()

    def __str__(self) -> str:
        if self._undefined_hint:
            message = f"undefined value printed: {self._undefined_hint}"

        elif self._undefined_obj is missing:
            message = self._undefined_name  # type: ignore

        else:
            message = (
                f"no such element: {object_type_repr(self._undefined_obj)}"
                f"[{self._undefined_name!r}]"
            )

        return f"{{{{ {message} }}}}"


class StrictUndefined(Undefined):
    """An undefined that barks on print and iteration as well as boolean
    tests and all kinds of comparisons.  In other words: you can do nothing
    with it except checking if it's defined using the `defined` test.

    >>> foo = StrictUndefined(name='foo')
    >>> str(foo)
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    >>> not foo
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    >>> foo + 42
    Traceback (most recent call last):
      ...
    jinja2.exceptions.UndefinedError: 'foo' is undefined
    """

    __slots__ = ()
    __iter__ = __str__ = __len__ = Undefined._fail_with_undefined_error
    __eq__ = __ne__ = __bool__ = __hash__ = Undefined._fail_with_undefined_error
    __contains__ = Undefined._fail_with_undefined_error

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\c\c_parser.py ===
from sympy.external import import_module
import os

cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

"""
This module contains all the necessary Classes and Function used to Parse C and
C++ code into SymPy expression
The module serves as a backend for SymPyExpression to parse C code
It is also dependent on Clang's AST and SymPy's Codegen AST.
The module only supports the features currently supported by the Clang and
codegen AST which will be updated as the development of codegen AST and this
module progresses.
You might find unexpected bugs and exceptions while using the module, feel free
to report them to the SymPy Issue Tracker

Features Supported
==================

- Variable Declarations (integers and reals)
- Assignment (using integer & floating literal and function calls)
- Function Definitions and Declaration
- Function Calls
- Compound statements, Return statements

Notes
=====

The module is dependent on an external dependency which needs to be installed
to use the features of this module.

Clang: The C and C++ compiler which is used to extract an AST from the provided
C source code.

References
==========

.. [1] https://github.com/sympy/sympy/issues
.. [2] https://clang.llvm.org/docs/
.. [3] https://clang.llvm.org/docs/IntroductionToTheClangAST.html

"""

if cin:
    from sympy.codegen.ast import (Variable, Integer, Float,
        FunctionPrototype, FunctionDefinition, FunctionCall,
        none, Return, Assignment, intc, int8, int16, int64,
        uint8, uint16, uint32, uint64, float32, float64, float80,
        aug_assign, bool_, While, CodeBlock)
    from sympy.codegen.cnodes import (PreDecrement, PostDecrement,
        PreIncrement, PostIncrement)
    from sympy.core import Add, Mod, Mul, Pow, Rel
    from sympy.logic.boolalg import And, as_Boolean, Not, Or
    from sympy.core.symbol import Symbol
    from sympy.core.sympify import sympify
    from sympy.logic.boolalg import (false, true)
    import sys
    import tempfile

    class BaseParser:
        """Base Class for the C parser"""

        def __init__(self):
            """Initializes the Base parser creating a Clang AST index"""
            self.index = cin.Index.create()

        def diagnostics(self, out):
            """Diagostics function for the Clang AST"""
            for diag in self.tu.diagnostics:
                # tu = translation unit
                print('%s %s (line %s, col %s) %s' % (
                        {
                            4: 'FATAL',
                            3: 'ERROR',
                            2: 'WARNING',
                            1: 'NOTE',
                            0: 'IGNORED',
                        }[diag.severity],
                        diag.location.file,
                        diag.location.line,
                        diag.location.column,
                        diag.spelling
                    ), file=out)

    class CCodeConverter(BaseParser):
        """The Code Convereter for Clang AST

        The converter object takes the C source code or file as input and
        converts them to SymPy Expressions.
        """

        def __init__(self):
            """Initializes the code converter"""
            super().__init__()
            self._py_nodes = []
            self._data_types = {
                "void": {
                    cin.TypeKind.VOID: none
                },
                "bool": {
                    cin.TypeKind.BOOL: bool_
                },
                "int": {
                    cin.TypeKind.SCHAR: int8,
                    cin.TypeKind.SHORT: int16,
                    cin.TypeKind.INT: intc,
                    cin.TypeKind.LONG: int64,
                    cin.TypeKind.UCHAR: uint8,
                    cin.TypeKind.USHORT: uint16,
                    cin.TypeKind.UINT: uint32,
                    cin.TypeKind.ULONG: uint64
                },
                "float": {
                    cin.TypeKind.FLOAT: float32,
                    cin.TypeKind.DOUBLE: float64,
                    cin.TypeKind.LONGDOUBLE: float80
                }
            }

        def parse(self, filename, flags):
            """Function to parse a file with C source code

            It takes the filename as an attribute and creates a Clang AST
            Translation Unit parsing the file.
            Then the transformation function is called on the translation unit,
            whose results are collected into a list which is returned by the
            function.

            Parameters
            ==========

            filename : string
                Path to the C file to be parsed

            flags: list
                Arguments to be passed to Clang while parsing the C code

            Returns
            =======

            py_nodes: list
                A list of SymPy AST nodes

            """
            filepath = os.path.abspath(filename)
            self.tu = self.index.parse(
                filepath,
                args=flags,
                options=cin.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            for child in self.tu.cursor.get_children():
                if child.kind == cin.CursorKind.VAR_DECL or child.kind == cin.CursorKind.FUNCTION_DECL:
                    self._py_nodes.append(self.transform(child))
            return self._py_nodes

        def parse_str(self, source, flags):
            """Function to parse a string with C source code

            It takes the source code as an attribute, stores it in a temporary
            file and creates a Clang AST Translation Unit parsing the file.
            Then the transformation function is called on the translation unit,
            whose results are collected into a list which is returned by the
            function.

            Parameters
            ==========

            source : string
                A string containing the C source code to be parsed

            flags: list
                Arguments to be passed to Clang while parsing the C code

            Returns
            =======

            py_nodes: list
                A list of SymPy AST nodes

            """
            file = tempfile.NamedTemporaryFile(mode = 'w+', suffix = '.cpp')
            file.write(source)
            file.flush()
            file.seek(0)
            self.tu = self.index.parse(
                file.name,
                args=flags,
                options=cin.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            file.close()
            for child in self.tu.cursor.get_children():
                if child.kind == cin.CursorKind.VAR_DECL or child.kind == cin.CursorKind.FUNCTION_DECL:
                    self._py_nodes.append(self.transform(child))
            return self._py_nodes

        def transform(self, node):
            """Transformation Function for Clang AST nodes

            It determines the kind of node and calls the respective
            transformation function for that node.

            Raises
            ======

            NotImplementedError : if the transformation for the provided node
            is not implemented

            """
            handler = getattr(self, 'transform_%s' % node.kind.name.lower(), None)

            if handler is None:
                print(
                    "Ignoring node of type %s (%s)" % (
                        node.kind,
                        ' '.join(
                            t.spelling for t in node.get_tokens())
                        ),
                    file=sys.stderr
                )

            return handler(node)

        def transform_var_decl(self, node):
            """Transformation Function for Variable Declaration

            Used to create nodes for variable declarations and assignments with
            values or function call for the respective nodes in the clang AST

            Returns
            =======

            A variable node as Declaration, with the initial value if given

            Raises
            ======

            NotImplementedError : if called for data types not currently
            implemented

            Notes
            =====

            The function currently supports following data types:

            Boolean:
                bool, _Bool

            Integer:
                8-bit: signed char and unsigned char
                16-bit: short, short int, signed short,
                    signed short int, unsigned short, unsigned short int
                32-bit: int, signed int, unsigned int
                64-bit: long, long int, signed long,
                    signed long int, unsigned long, unsigned long int

            Floating point:
                Single Precision: float
                Double Precision: double
                Extended Precision: long double

            """
            if node.type.kind in self._data_types["int"]:
                type = self._data_types["int"][node.type.kind]
            elif node.type.kind in self._data_types["float"]:
                type = self._data_types["float"][node.type.kind]
            elif node.type.kind in self._data_types["bool"]:
                type = self._data_types["bool"][node.type.kind]
            else:
                raise NotImplementedError("Only bool, int "
                    "and float are supported")
            try:
                children = node.get_children()
                child = next(children)

                #ignoring namespace and type details for the variable
                while child.kind == cin.CursorKind.NAMESPACE_REF or child.kind == cin.CursorKind.TYPE_REF:
                    child = next(children)

                val = self.transform(child)

                supported_rhs = [
                    cin.CursorKind.INTEGER_LITERAL,
                    cin.CursorKind.FLOATING_LITERAL,
                    cin.CursorKind.UNEXPOSED_EXPR,
                    cin.CursorKind.BINARY_OPERATOR,
                    cin.CursorKind.PAREN_EXPR,
                    cin.CursorKind.UNARY_OPERATOR,
                    cin.CursorKind.CXX_BOOL_LITERAL_EXPR
                ]

                if child.kind in supported_rhs:
                    if isinstance(val, str):
                        value = Symbol(val)
                    elif isinstance(val, bool):
                        if node.type.kind in self._data_types["int"]:
                            value = Integer(0) if val == False else Integer(1)
                        elif node.type.kind in self._data_types["float"]:
                            value = Float(0.0) if val == False else Float(1.0)
                        elif node.type.kind in self._data_types["bool"]:
                            value = sympify(val)
                    elif isinstance(val, (Integer, int, Float, float)):
                        if node.type.kind in self._data_types["int"]:
                            value = Integer(val)
                        elif node.type.kind in self._data_types["float"]:
                            value = Float(val)
                        elif node.type.kind in self._data_types["bool"]:
                            value = sympify(bool(val))
                    else:
                        value = val

                    return Variable(
                    node.spelling
                    ).as_Declaration(
                        type = type,
                        value = value
                    )

                elif child.kind == cin.CursorKind.CALL_EXPR:
                    return Variable(
                        node.spelling
                        ).as_Declaration(
                            value = val
                        )

                else:
                    raise NotImplementedError("Given "
                        "variable declaration \"{}\" "
                        "is not possible to parse yet!"
                        .format(" ".join(
                            t.spelling for t in node.get_tokens()
                            )
                        ))

            except StopIteration:
                return Variable(
                node.spelling
                ).as_Declaration(
                    type = type
                )

        def transform_function_decl(self, node):
            """Transformation Function For Function Declaration

            Used to create nodes for function declarations and definitions for
            the respective nodes in the clang AST

            Returns
            =======

            function : Codegen AST node
                - FunctionPrototype node if function body is not present
                - FunctionDefinition node if the function body is present


            """

            if node.result_type.kind in self._data_types["int"]:
                ret_type = self._data_types["int"][node.result_type.kind]
            elif node.result_type.kind in self._data_types["float"]:
                ret_type = self._data_types["float"][node.result_type.kind]
            elif node.result_type.kind in self._data_types["bool"]:
                ret_type = self._data_types["bool"][node.result_type.kind]
            elif node.result_type.kind in self._data_types["void"]:
                ret_type = self._data_types["void"][node.result_type.kind]
            else:
                raise NotImplementedError("Only void, bool, int "
                    "and float are supported")
            body = []
            param = []

            # Subsequent nodes will be the parameters for the function.
            for child in node.get_children():
                decl = self.transform(child)
                if child.kind == cin.CursorKind.PARM_DECL:
                    param.append(decl)
                elif child.kind == cin.CursorKind.COMPOUND_STMT:
                    for val in decl:
                        body.append(val)
                else:
                    body.append(decl)

            if body == []:
                function = FunctionPrototype(
                    return_type = ret_type,
                    name = node.spelling,
                    parameters = param
                )
            else:
                function = FunctionDefinition(
                    return_type = ret_type,
                    name = node.spelling,
                    parameters = param,
                    body = body
                )
            return function

        def transform_parm_decl(self, node):
            """Transformation function for Parameter Declaration

            Used to create parameter nodes for the required functions for the
            respective nodes in the clang AST

            Returns
            =======

            param : Codegen AST Node
                Variable node with the value and type of the variable

            Raises
            ======

            ValueError if multiple children encountered in the parameter node

            """
            if node.type.kind in self._data_types["int"]:
                type = self._data_types["int"][node.type.kind]
            elif node.type.kind in self._data_types["float"]:
                type = self._data_types["float"][node.type.kind]
            elif node.type.kind in self._data_types["bool"]:
                type = self._data_types["bool"][node.type.kind]
            else:
                raise NotImplementedError("Only bool, int "
                    "and float are supported")
            try:
                children = node.get_children()
                child = next(children)

                # Any namespace nodes can be ignored
                while child.kind in [cin.CursorKind.NAMESPACE_REF,
                                     cin.CursorKind.TYPE_REF,
                                     cin.CursorKind.TEMPLATE_REF]:
                    child = next(children)

                # If there is a child, it is the default value of the parameter.
                lit = self.transform(child)
                if node.type.kind in self._data_types["int"]:
                    val = Integer(lit)
                elif node.type.kind in self._data_types["float"]:
                    val = Float(lit)
                elif node.type.kind in self._data_types["bool"]:
                    val = sympify(bool(lit))
                else:
                    raise NotImplementedError("Only bool, int "
                        "and float are supported")

                param = Variable(
                    node.spelling
                ).as_Declaration(
                    type = type,
                    value = val
                )
            except StopIteration:
                param = Variable(
                    node.spelling
                ).as_Declaration(
                    type = type
                )

            try:
                self.transform(next(children))
                raise ValueError("Can't handle multiple children on parameter")
            except StopIteration:
                pass

            return param

        def transform_integer_literal(self, node):
            """Transformation function for integer literal

            Used to get the value and type of the given integer literal.

            Returns
            =======

            val : list
                List with two arguments type and Value
                type contains the type of the integer
                value contains the value stored in the variable

            Notes
            =====

            Only Base Integer type supported for now

            """
            try:
                value = next(node.get_tokens()).spelling
            except StopIteration:
                # No tokens
                value = node.literal
            return int(value)

        def transform_floating_literal(self, node):
            """Transformation function for floating literal

            Used to get the value and type of the given floating literal.

            Returns
            =======

            val : list
                List with two arguments type and Value
                type contains the type of float
                value contains the value stored in the variable

            Notes
            =====

            Only Base Float type supported for now

            """
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                # No tokens
                value = node.literal
            return float(value)

        def transform_string_literal(self, node):
            #TODO: No string type in AST
            #type =
            #try:
            #    value = next(node.get_tokens()).spelling
            #except (StopIteration, ValueError):
                # No tokens
            #    value = node.literal
            #val = [type, value]
            #return val
            pass

        def transform_character_literal(self, node):
            """Transformation function for character literal

            Used to get the value of the given character literal.

            Returns
            =======

            val : int
                val contains the ascii value of the character literal

            Notes
            =====

            Only for cases where character is assigned to a integer value,
            since character literal is not in SymPy AST

            """
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                # No tokens
                value = node.literal
            return ord(str(value[1]))

        def transform_cxx_bool_literal_expr(self, node):
            """Transformation function for boolean literal

            Used to get the value of the given boolean literal.

            Returns
            =======

            value : bool
                value contains the boolean value of the variable

            """
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                value = node.literal
            return True if value == 'true' else False

        def transform_unexposed_decl(self,node):
            """Transformation function for unexposed declarations"""
            pass

        def transform_unexposed_expr(self, node):
            """Transformation function for unexposed expression

            Unexposed expressions are used to wrap float, double literals and
            expressions

            Returns
            =======

            expr : Codegen AST Node
                the result from the wrapped expression

            None : NoneType
                No children are found for the node

            Raises
            ======

            ValueError if the expression contains multiple children

            """
            # Ignore unexposed nodes; pass whatever is the first
            # (and should be only) child unaltered.
            try:
                children = node.get_children()
                expr = self.transform(next(children))
            except StopIteration:
                return None

            try:
                next(children)
                raise ValueError("Unexposed expression has > 1 children.")
            except StopIteration:
                pass

            return expr

        def transform_decl_ref_expr(self, node):
            """Returns the name of the declaration reference"""
            return node.spelling

        def transform_call_expr(self, node):
            """Transformation function for a call expression

            Used to create function call nodes for the function calls present
            in the C code

            Returns
            =======

            FunctionCall : Codegen AST Node
                FunctionCall node with parameters if any parameters are present

            """
            param = []
            children = node.get_children()
            child = next(children)

            while child.kind == cin.CursorKind.NAMESPACE_REF:
                child = next(children)
            while child.kind == cin.CursorKind.TYPE_REF:
                child = next(children)

            first_child = self.transform(child)
            try:
                for child in children:
                    arg = self.transform(child)
                    if child.kind == cin.CursorKind.INTEGER_LITERAL:
                        param.append(Integer(arg))
                    elif child.kind == cin.CursorKind.FLOATING_LITERAL:
                        param.append(Float(arg))
                    else:
                        param.append(arg)
                return FunctionCall(first_child, param)

            except StopIteration:
                return FunctionCall(first_child)

        def transform_return_stmt(self, node):
            """Returns the Return Node for a return statement"""
            return Return(next(node.get_children()).spelling)

        def transform_compound_stmt(self, node):
            """Transformation function for compound statements

            Returns
            =======

            expr : list
                list of Nodes for the expressions present in the statement

            None : NoneType
                if the compound statement is empty

            """
            expr = []
            children = node.get_children()

            for child in children:
                expr.append(self.transform(child))
            return expr

        def transform_decl_stmt(self, node):
            """Transformation function for declaration statements

            These statements are used to wrap different kinds of declararions
            like variable or function declaration
            The function calls the transformer function for the child of the
            given node

            Returns
            =======

            statement : Codegen AST Node
                contains the node returned by the children node for the type of
                declaration

            Raises
            ======

            ValueError if multiple children present

            """
            try:
                children = node.get_children()
                statement = self.transform(next(children))
            except StopIteration:
                pass

            try:
                self.transform(next(children))
                raise ValueError("Don't know how to handle multiple statements")
            except StopIteration:
                pass

            return statement

        def transform_paren_expr(self, node):
            """Transformation function for Parenthesized expressions

            Returns the result from its children nodes

            """
            return self.transform(next(node.get_children()))

        def transform_compound_assignment_operator(self, node):
            """Transformation function for handling shorthand operators

            Returns
            =======

            augmented_assignment_expression: Codegen AST node
                    shorthand assignment expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If the shorthand operator for bitwise operators
                (~=, ^=, &=, |=, <<=, >>=) is encountered

            """
            return self.transform_binary_operator(node)

        def transform_unary_operator(self, node):
            """Transformation function for handling unary operators

            Returns
            =======

            unary_expression: Codegen AST node
                    simplified unary expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If dereferencing operator(*), address operator(&) or
                bitwise NOT operator(~) is encountered

            """
            # supported operators list
            operators_list = ['+', '-', '++', '--', '!']
            tokens = list(node.get_tokens())

            # it can be either pre increment/decrement or any other operator from the list
            if tokens[0].spelling in operators_list:
                child = self.transform(next(node.get_children()))
                # (decl_ref) e.g.; int a = ++b; or simply ++b;
                if isinstance(child, str):
                    if tokens[0].spelling == '+':
                        return Symbol(child)
                    if tokens[0].spelling == '-':
                        return Mul(Symbol(child), -1)
                    if tokens[0].spelling == '++':
                        return PreIncrement(Symbol(child))
                    if tokens[0].spelling == '--':
                        return PreDecrement(Symbol(child))
                    if tokens[0].spelling == '!':
                        return Not(Symbol(child))
                # e.g.; int a = -1; or int b = -(1 + 2);
                else:
                    if tokens[0].spelling == '+':
                        return child
                    if tokens[0].spelling == '-':
                        return Mul(child, -1)
                    if tokens[0].spelling == '!':
                        return Not(sympify(bool(child)))

            # it can be either post increment/decrement
            # since variable name is obtained in token[0].spelling
            elif tokens[1].spelling in ['++', '--']:
                child = self.transform(next(node.get_children()))
                if tokens[1].spelling == '++':
                    return PostIncrement(Symbol(child))
                if tokens[1].spelling == '--':
                    return PostDecrement(Symbol(child))
            else:
                raise NotImplementedError("Dereferencing operator, "
                    "Address operator and bitwise NOT operator "
                    "have not been implemented yet!")

        def transform_binary_operator(self, node):
            """Transformation function for handling binary operators

            Returns
            =======

            binary_expression: Codegen AST node
                    simplified binary expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If a bitwise operator or
                unary operator(which is a child of any binary
                operator in Clang AST) is encountered

            """
            # get all the tokens of assignment
            # and store it in the tokens list
            tokens = list(node.get_tokens())

            # supported operators list
            operators_list = ['+', '-', '*', '/', '%','=',
            '>', '>=', '<', '<=', '==', '!=', '&&', '||', '+=', '-=',
            '*=', '/=', '%=']

            # this stack will contain variable content
            # and type of variable in the rhs
            combined_variables_stack = []

            # this stack will contain operators
            # to be processed in the rhs
            operators_stack = []

            # iterate through every token
            for token in tokens:
                # token is either '(', ')' or
                # any of the supported operators from the operator list
                if token.kind == cin.TokenKind.PUNCTUATION:

                    # push '(' to the operators stack
                    if token.spelling == '(':
                        operators_stack.append('(')

                    elif token.spelling == ')':
                        # keep adding the expression to the
                        # combined variables stack unless
                        # '(' is found
                        while (operators_stack
                            and operators_stack[-1] != '('):
                            if len(combined_variables_stack) < 2:
                                raise NotImplementedError(
                                    "Unary operators as a part of "
                                    "binary operators is not "
                                    "supported yet!")
                            rhs = combined_variables_stack.pop()
                            lhs = combined_variables_stack.pop()
                            operator = operators_stack.pop()
                            combined_variables_stack.append(
                                self.perform_operation(
                                lhs, rhs, operator))

                        # pop '('
                        operators_stack.pop()

                    # token is an operator (supported)
                    elif token.spelling in operators_list:
                        while (operators_stack
                            and self.priority_of(token.spelling)
                            <= self.priority_of(
                            operators_stack[-1])):
                            if len(combined_variables_stack) < 2:
                                raise NotImplementedError(
                                    "Unary operators as a part of "
                                    "binary operators is not "
                                    "supported yet!")
                            rhs = combined_variables_stack.pop()
                            lhs = combined_variables_stack.pop()
                            operator = operators_stack.pop()
                            combined_variables_stack.append(
                                self.perform_operation(
                                lhs, rhs, operator))

                        # push current operator
                        operators_stack.append(token.spelling)

                    # token is a bitwise operator
                    elif token.spelling in ['&', '|', '^', '<<', '>>']:
                        raise NotImplementedError(
                            "Bitwise operator has not been "
                            "implemented yet!")

                    # token is a shorthand bitwise operator
                    elif token.spelling in ['&=', '|=', '^=', '<<=',
                    '>>=']:
                        raise NotImplementedError(
                            "Shorthand bitwise operator has not been "
                            "implemented yet!")
                    else:
                        raise NotImplementedError(
                            "Given token {} is not implemented yet!"
                            .format(token.spelling))

                # token is an identifier(variable)
                elif token.kind == cin.TokenKind.IDENTIFIER:
                    combined_variables_stack.append(
                        [token.spelling, 'identifier'])

                # token is a literal
                elif token.kind == cin.TokenKind.LITERAL:
                    combined_variables_stack.append(
                        [token.spelling, 'literal'])

                # token is a keyword, either true or false
                elif (token.kind == cin.TokenKind.KEYWORD
                    and token.spelling in ['true', 'false']):
                    combined_variables_stack.append(
                        [token.spelling, 'boolean'])
                else:
                    raise NotImplementedError(
                        "Given token {} is not implemented yet!"
                        .format(token.spelling))

            # process remaining operators
            while operators_stack:
                if len(combined_variables_stack) < 2:
                    raise NotImplementedError(
                        "Unary operators as a part of "
                        "binary operators is not "
                        "supported yet!")
                rhs = combined_variables_stack.pop()
                lhs = combined_variables_stack.pop()
                operator = operators_stack.pop()
                combined_variables_stack.append(
                    self.perform_operation(lhs, rhs, operator))

            return combined_variables_stack[-1][0]

        def priority_of(self, op):
            """To get the priority of given operator"""
            if op in ['=', '+=', '-=', '*=', '/=', '%=']:
                return 1
            if op in ['&&', '||']:
                return 2
            if op in ['<', '<=', '>', '>=', '==', '!=']:
                return 3
            if op in ['+', '-']:
                return 4
            if op in ['*', '/', '%']:
                return 5
            return 0

        def perform_operation(self, lhs, rhs, op):
            """Performs operation supported by the SymPy core

            Returns
            =======

            combined_variable: list
                contains variable content and type of variable

            """
            lhs_value = self.get_expr_for_operand(lhs)
            rhs_value = self.get_expr_for_operand(rhs)
            if op == '+':
                return [Add(lhs_value, rhs_value), 'expr']
            if op == '-':
                return [Add(lhs_value, -rhs_value), 'expr']
            if op == '*':
                return [Mul(lhs_value, rhs_value), 'expr']
            if op == '/':
                return [Mul(lhs_value, Pow(rhs_value, Integer(-1))), 'expr']
            if op == '%':
                return [Mod(lhs_value, rhs_value), 'expr']
            if op in ['<', '<=', '>', '>=', '==', '!=']:
                return [Rel(lhs_value, rhs_value, op), 'expr']
            if op == '&&':
                return [And(as_Boolean(lhs_value), as_Boolean(rhs_value)), 'expr']
            if op == '||':
                return [Or(as_Boolean(lhs_value), as_Boolean(rhs_value)), 'expr']
            if op == '=':
                return [Assignment(Variable(lhs_value), rhs_value), 'expr']
            if op in ['+=', '-=', '*=', '/=', '%=']:
                return [aug_assign(Variable(lhs_value), op[0], rhs_value), 'expr']

        def get_expr_for_operand(self, combined_variable):
            """Gives out SymPy Codegen AST node

            AST node returned is corresponding to
            combined variable passed.Combined variable contains
            variable content and type of variable

            """
            if combined_variable[1] == 'identifier':
                return Symbol(combined_variable[0])
            if combined_variable[1] == 'literal':
                if '.' in combined_variable[0]:
                    return Float(float(combined_variable[0]))
                else:
                    return Integer(int(combined_variable[0]))
            if combined_variable[1] == 'expr':
                return combined_variable[0]
            if combined_variable[1] == 'boolean':
                    return true if combined_variable[0] == 'true' else false

        def transform_null_stmt(self, node):
            """Handles Null Statement and returns None"""
            return none

        def transform_while_stmt(self, node):
            """Transformation function for handling while statement

            Returns
            =======

            while statement : Codegen AST Node
                contains the while statement node having condition and
                statement block

            """
            children = node.get_children()

            condition = self.transform(next(children))
            statements = self.transform(next(children))

            if isinstance(statements, list):
                statement_block = CodeBlock(*statements)
            else:
                statement_block = CodeBlock(statements)

            return While(condition, statement_block)



else:
    class CCodeConverter():  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Module not Installed")


def parse_c(source):
    """Function for converting a C source code

    The function reads the source code present in the given file and parses it
    to give out SymPy Expressions

    Returns
    =======

    src : list
        List of Python expression strings

    """
    converter = CCodeConverter()
    if os.path.exists(source):
        src = converter.parse(source, flags = [])
    else:
        src = converter.parse_str(source, flags = [])
    return src

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\cache_key.py ===
# sql/cache_key.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

import enum
from itertools import zip_longest
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from .visitors import anon_map
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import util
from ..inspection import inspect
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol

if typing.TYPE_CHECKING:
    from .elements import BindParameter
    from .elements import ClauseElement
    from .elements import ColumnElement
    from .visitors import _TraverseInternalsType
    from ..engine.interfaces import _CoreSingleExecuteParams


class _CacheKeyTraversalDispatchType(Protocol):
    def __call__(
        s, self: HasCacheKey, visitor: _CacheKeyTraversal
    ) -> _CacheKeyTraversalDispatchTypeReturn: ...


class CacheConst(enum.Enum):
    NO_CACHE = 0


NO_CACHE = CacheConst.NO_CACHE


_CacheKeyTraversalType = Union[
    "_TraverseInternalsType", Literal[CacheConst.NO_CACHE], Literal[None]
]


class CacheTraverseTarget(enum.Enum):
    CACHE_IN_PLACE = 0
    CALL_GEN_CACHE_KEY = 1
    STATIC_CACHE_KEY = 2
    PROPAGATE_ATTRS = 3
    ANON_NAME = 4


(
    CACHE_IN_PLACE,
    CALL_GEN_CACHE_KEY,
    STATIC_CACHE_KEY,
    PROPAGATE_ATTRS,
    ANON_NAME,
) = tuple(CacheTraverseTarget)

_CacheKeyTraversalDispatchTypeReturn = Sequence[
    Tuple[
        str,
        Any,
        Union[
            Callable[..., Tuple[Any, ...]],
            CacheTraverseTarget,
            InternalTraversal,
        ],
    ]
]


class HasCacheKey:
    """Mixin for objects which can produce a cache key.

    This class is usually in a hierarchy that starts with the
    :class:`.HasTraverseInternals` base, but this is optional.  Currently,
    the class should be able to work on its own without including
    :class:`.HasTraverseInternals`.

    .. seealso::

        :class:`.CacheKey`

        :ref:`sql_caching`

    """

    __slots__ = ()

    _cache_key_traversal: _CacheKeyTraversalType = NO_CACHE

    _is_has_cache_key = True

    _hierarchy_supports_caching = True
    """private attribute which may be set to False to prevent the
    inherit_cache warning from being emitted for a hierarchy of subclasses.

    Currently applies to the :class:`.ExecutableDDLElement` hierarchy which
    does not implement caching.

    """

    inherit_cache: Optional[bool] = None
    """Indicate if this :class:`.HasCacheKey` instance should make use of the
    cache key generation scheme used by its immediate superclass.

    The attribute defaults to ``None``, which indicates that a construct has
    not yet taken into account whether or not its appropriate for it to
    participate in caching; this is functionally equivalent to setting the
    value to ``False``, except that a warning is also emitted.

    This flag can be set to ``True`` on a particular class, if the SQL that
    corresponds to the object does not change based on attributes which
    are local to this class, and not its superclass.

    .. seealso::

        :ref:`compilerext_caching` - General guideslines for setting the
        :attr:`.HasCacheKey.inherit_cache` attribute for third-party or user
        defined SQL constructs.

    """

    __slots__ = ()

    _generated_cache_key_traversal: Any

    @classmethod
    def _generate_cache_attrs(
        cls,
    ) -> Union[_CacheKeyTraversalDispatchType, Literal[CacheConst.NO_CACHE]]:
        """generate cache key dispatcher for a new class.

        This sets the _generated_cache_key_traversal attribute once called
        so should only be called once per class.

        """
        inherit_cache = cls.__dict__.get("inherit_cache", None)
        inherit = bool(inherit_cache)

        if inherit:
            _cache_key_traversal = getattr(cls, "_cache_key_traversal", None)
            if _cache_key_traversal is None:
                try:
                    assert issubclass(cls, HasTraverseInternals)
                    _cache_key_traversal = cls._traverse_internals
                except AttributeError:
                    cls._generated_cache_key_traversal = NO_CACHE
                    return NO_CACHE

            assert _cache_key_traversal is not NO_CACHE, (
                f"class {cls} has _cache_key_traversal=NO_CACHE, "
                "which conflicts with inherit_cache=True"
            )

            # TODO: wouldn't we instead get this from our superclass?
            # also, our superclass may not have this yet, but in any case,
            # we'd generate for the superclass that has it.   this is a little
            # more complicated, so for the moment this is a little less
            # efficient on startup but simpler.
            return _cache_key_traversal_visitor.generate_dispatch(
                cls,
                _cache_key_traversal,
                "_generated_cache_key_traversal",
            )
        else:
            _cache_key_traversal = cls.__dict__.get(
                "_cache_key_traversal", None
            )
            if _cache_key_traversal is None:
                _cache_key_traversal = cls.__dict__.get(
                    "_traverse_internals", None
                )
                if _cache_key_traversal is None:
                    cls._generated_cache_key_traversal = NO_CACHE
                    if (
                        inherit_cache is None
                        and cls._hierarchy_supports_caching
                    ):
                        util.warn(
                            "Class %s will not make use of SQL compilation "
                            "caching as it does not set the 'inherit_cache' "
                            "attribute to ``True``.  This can have "
                            "significant performance implications including "
                            "some performance degradations in comparison to "
                            "prior SQLAlchemy versions.  Set this attribute "
                            "to True if this object can make use of the cache "
                            "key generated by the superclass.  Alternatively, "
                            "this attribute may be set to False which will "
                            "disable this warning." % (cls.__name__),
                            code="cprf",
                        )
                    return NO_CACHE

            return _cache_key_traversal_visitor.generate_dispatch(
                cls,
                _cache_key_traversal,
                "_generated_cache_key_traversal",
            )

    @util.preload_module("sqlalchemy.sql.elements")
    def _gen_cache_key(
        self, anon_map: anon_map, bindparams: List[BindParameter[Any]]
    ) -> Optional[Tuple[Any, ...]]:
        """return an optional cache key.

        The cache key is a tuple which can contain any series of
        objects that are hashable and also identifies
        this object uniquely within the presence of a larger SQL expression
        or statement, for the purposes of caching the resulting query.

        The cache key should be based on the SQL compiled structure that would
        ultimately be produced.   That is, two structures that are composed in
        exactly the same way should produce the same cache key; any difference
        in the structures that would affect the SQL string or the type handlers
        should result in a different cache key.

        If a structure cannot produce a useful cache key, the NO_CACHE
        symbol should be added to the anon_map and the method should
        return None.

        """

        cls = self.__class__

        id_, found = anon_map.get_anon(self)
        if found:
            return (id_, cls)

        dispatcher: Union[
            Literal[CacheConst.NO_CACHE],
            _CacheKeyTraversalDispatchType,
        ]

        try:
            dispatcher = cls.__dict__["_generated_cache_key_traversal"]
        except KeyError:
            # traversals.py -> _preconfigure_traversals()
            # may be used to run these ahead of time, but
            # is not enabled right now.
            # this block will generate any remaining dispatchers.
            dispatcher = cls._generate_cache_attrs()

        if dispatcher is NO_CACHE:
            anon_map[NO_CACHE] = True
            return None

        result: Tuple[Any, ...] = (id_, cls)

        # inline of _cache_key_traversal_visitor.run_generated_dispatch()

        for attrname, obj, meth in dispatcher(
            self, _cache_key_traversal_visitor
        ):
            if obj is not None:
                # TODO: see if C code can help here as Python lacks an
                # efficient switch construct

                if meth is STATIC_CACHE_KEY:
                    sck = obj._static_cache_key
                    if sck is NO_CACHE:
                        anon_map[NO_CACHE] = True
                        return None
                    result += (attrname, sck)
                elif meth is ANON_NAME:
                    elements = util.preloaded.sql_elements
                    if isinstance(obj, elements._anonymous_label):
                        obj = obj.apply_map(anon_map)  # type: ignore
                    result += (attrname, obj)
                elif meth is CALL_GEN_CACHE_KEY:
                    result += (
                        attrname,
                        obj._gen_cache_key(anon_map, bindparams),
                    )

                # remaining cache functions are against
                # Python tuples, dicts, lists, etc. so we can skip
                # if they are empty
                elif obj:
                    if meth is CACHE_IN_PLACE:
                        result += (attrname, obj)
                    elif meth is PROPAGATE_ATTRS:
                        result += (
                            attrname,
                            obj["compile_state_plugin"],
                            (
                                obj["plugin_subject"]._gen_cache_key(
                                    anon_map, bindparams
                                )
                                if obj["plugin_subject"]
                                else None
                            ),
                        )
                    elif meth is InternalTraversal.dp_annotations_key:
                        # obj is here is the _annotations dict.  Table uses
                        # a memoized version of it.  however in other cases,
                        # we generate it given anon_map as we may be from a
                        # Join, Aliased, etc.
                        # see #8790

                        if self._gen_static_annotations_cache_key:  # type: ignore  # noqa: E501
                            result += self._annotations_cache_key  # type: ignore  # noqa: E501
                        else:
                            result += self._gen_annotations_cache_key(anon_map)  # type: ignore  # noqa: E501

                    elif (
                        meth is InternalTraversal.dp_clauseelement_list
                        or meth is InternalTraversal.dp_clauseelement_tuple
                        or meth
                        is InternalTraversal.dp_memoized_select_entities
                    ):
                        result += (
                            attrname,
                            tuple(
                                [
                                    elem._gen_cache_key(anon_map, bindparams)
                                    for elem in obj
                                ]
                            ),
                        )
                    else:
                        result += meth(  # type: ignore
                            attrname, obj, self, anon_map, bindparams
                        )
        return result

    def _generate_cache_key(self) -> Optional[CacheKey]:
        """return a cache key.

        The cache key is a tuple which can contain any series of
        objects that are hashable and also identifies
        this object uniquely within the presence of a larger SQL expression
        or statement, for the purposes of caching the resulting query.

        The cache key should be based on the SQL compiled structure that would
        ultimately be produced.   That is, two structures that are composed in
        exactly the same way should produce the same cache key; any difference
        in the structures that would affect the SQL string or the type handlers
        should result in a different cache key.

        The cache key returned by this method is an instance of
        :class:`.CacheKey`, which consists of a tuple representing the
        cache key, as well as a list of :class:`.BindParameter` objects
        which are extracted from the expression.   While two expressions
        that produce identical cache key tuples will themselves generate
        identical SQL strings, the list of :class:`.BindParameter` objects
        indicates the bound values which may have different values in
        each one; these bound parameters must be consulted in order to
        execute the statement with the correct parameters.

        a :class:`_expression.ClauseElement` structure that does not implement
        a :meth:`._gen_cache_key` method and does not implement a
        :attr:`.traverse_internals` attribute will not be cacheable; when
        such an element is embedded into a larger structure, this method
        will return None, indicating no cache key is available.

        """

        bindparams: List[BindParameter[Any]] = []

        _anon_map = anon_map()
        key = self._gen_cache_key(_anon_map, bindparams)
        if NO_CACHE in _anon_map:
            return None
        else:
            assert key is not None
            return CacheKey(key, bindparams)

    @classmethod
    def _generate_cache_key_for_object(
        cls, obj: HasCacheKey
    ) -> Optional[CacheKey]:
        bindparams: List[BindParameter[Any]] = []

        _anon_map = anon_map()
        key = obj._gen_cache_key(_anon_map, bindparams)
        if NO_CACHE in _anon_map:
            return None
        else:
            assert key is not None
            return CacheKey(key, bindparams)


class HasCacheKeyTraverse(HasTraverseInternals, HasCacheKey):
    pass


class MemoizedHasCacheKey(HasCacheKey, HasMemoized):
    __slots__ = ()

    @HasMemoized.memoized_instancemethod
    def _generate_cache_key(self) -> Optional[CacheKey]:
        return HasCacheKey._generate_cache_key(self)


class SlotsMemoizedHasCacheKey(HasCacheKey, util.MemoizedSlots):
    __slots__ = ()

    def _memoized_method__generate_cache_key(self) -> Optional[CacheKey]:
        return HasCacheKey._generate_cache_key(self)


class CacheKey(NamedTuple):
    """The key used to identify a SQL statement construct in the
    SQL compilation cache.

    .. seealso::

        :ref:`sql_caching`

    """

    key: Tuple[Any, ...]
    bindparams: Sequence[BindParameter[Any]]

    # can't set __hash__ attribute because it interferes
    # with namedtuple
    # can't use "if not TYPE_CHECKING" because mypy rejects it
    # inside of a NamedTuple
    def __hash__(self) -> Optional[int]:  # type: ignore
        """CacheKey itself is not hashable - hash the .key portion"""
        return None

    def to_offline_string(
        self,
        statement_cache: MutableMapping[Any, str],
        statement: ClauseElement,
        parameters: _CoreSingleExecuteParams,
    ) -> str:
        """Generate an "offline string" form of this :class:`.CacheKey`

        The "offline string" is basically the string SQL for the
        statement plus a repr of the bound parameter values in series.
        Whereas the :class:`.CacheKey` object is dependent on in-memory
        identities in order to work as a cache key, the "offline" version
        is suitable for a cache that will work for other processes as well.

        The given ``statement_cache`` is a dictionary-like object where the
        string form of the statement itself will be cached.  This dictionary
        should be in a longer lived scope in order to reduce the time spent
        stringifying statements.


        """
        if self.key not in statement_cache:
            statement_cache[self.key] = sql_str = str(statement)
        else:
            sql_str = statement_cache[self.key]

        if not self.bindparams:
            param_tuple = tuple(parameters[key] for key in sorted(parameters))
        else:
            param_tuple = tuple(
                parameters.get(bindparam.key, bindparam.value)
                for bindparam in self.bindparams
            )

        return repr((sql_str, param_tuple))

    def __eq__(self, other: Any) -> bool:
        return bool(self.key == other.key)

    def __ne__(self, other: Any) -> bool:
        return not (self.key == other.key)

    @classmethod
    def _diff_tuples(cls, left: CacheKey, right: CacheKey) -> str:
        ck1 = CacheKey(left, [])
        ck2 = CacheKey(right, [])
        return ck1._diff(ck2)

    def _whats_different(self, other: CacheKey) -> Iterator[str]:
        k1 = self.key
        k2 = other.key

        stack: List[int] = []
        pickup_index = 0
        while True:
            s1, s2 = k1, k2
            for idx in stack:
                s1 = s1[idx]
                s2 = s2[idx]

            for idx, (e1, e2) in enumerate(zip_longest(s1, s2)):
                if idx < pickup_index:
                    continue
                if e1 != e2:
                    if isinstance(e1, tuple) and isinstance(e2, tuple):
                        stack.append(idx)
                        break
                    else:
                        yield "key%s[%d]:  %s != %s" % (
                            "".join("[%d]" % id_ for id_ in stack),
                            idx,
                            e1,
                            e2,
                        )
            else:
                stack.pop(-1)
                break

    def _diff(self, other: CacheKey) -> str:
        return ", ".join(self._whats_different(other))

    def __str__(self) -> str:
        stack: List[Union[Tuple[Any, ...], HasCacheKey]] = [self.key]

        output = []
        sentinel = object()
        indent = -1
        while stack:
            elem = stack.pop(0)
            if elem is sentinel:
                output.append((" " * (indent * 2)) + "),")
                indent -= 1
            elif isinstance(elem, tuple):
                if not elem:
                    output.append((" " * ((indent + 1) * 2)) + "()")
                else:
                    indent += 1
                    stack = list(elem) + [sentinel] + stack
                    output.append((" " * (indent * 2)) + "(")
            else:
                if isinstance(elem, HasCacheKey):
                    repr_ = "<%s object at %s>" % (
                        type(elem).__name__,
                        hex(id(elem)),
                    )
                else:
                    repr_ = repr(elem)
                output.append((" " * (indent * 2)) + "  " + repr_ + ", ")

        return "CacheKey(key=%s)" % ("\n".join(output),)

    def _generate_param_dict(self) -> Dict[str, Any]:
        """used for testing"""

        _anon_map = prefix_anon_map()
        return {b.key % _anon_map: b.effective_value for b in self.bindparams}

    @util.preload_module("sqlalchemy.sql.elements")
    def _apply_params_to_element(
        self, original_cache_key: CacheKey, target_element: ColumnElement[Any]
    ) -> ColumnElement[Any]:
        if target_element._is_immutable or original_cache_key is self:
            return target_element

        elements = util.preloaded.sql_elements
        return elements._OverrideBinds(
            target_element, self.bindparams, original_cache_key.bindparams
        )


def _ad_hoc_cache_key_from_args(
    tokens: Tuple[Any, ...],
    traverse_args: Iterable[Tuple[str, InternalTraversal]],
    args: Iterable[Any],
) -> Tuple[Any, ...]:
    """a quick cache key generator used by reflection.flexi_cache."""
    bindparams: List[BindParameter[Any]] = []

    _anon_map = anon_map()

    tup = tokens

    for (attrname, sym), arg in zip(traverse_args, args):
        key = sym.name
        visit_key = key.replace("dp_", "visit_")

        if arg is None:
            tup += (attrname, None)
            continue

        meth = getattr(_cache_key_traversal_visitor, visit_key)
        if meth is CACHE_IN_PLACE:
            tup += (attrname, arg)
        elif meth in (
            CALL_GEN_CACHE_KEY,
            STATIC_CACHE_KEY,
            ANON_NAME,
            PROPAGATE_ATTRS,
        ):
            raise NotImplementedError(
                f"Haven't implemented symbol {meth} for ad-hoc key from args"
            )
        else:
            tup += meth(attrname, arg, None, _anon_map, bindparams)
    return tup


class _CacheKeyTraversal(HasTraversalDispatch):
    # very common elements are inlined into the main _get_cache_key() method
    # to produce a dramatic savings in Python function call overhead

    visit_has_cache_key = visit_clauseelement = CALL_GEN_CACHE_KEY
    visit_clauseelement_list = InternalTraversal.dp_clauseelement_list
    visit_annotations_key = InternalTraversal.dp_annotations_key
    visit_clauseelement_tuple = InternalTraversal.dp_clauseelement_tuple
    visit_memoized_select_entities = (
        InternalTraversal.dp_memoized_select_entities
    )

    visit_string = visit_boolean = visit_operator = visit_plain_obj = (
        CACHE_IN_PLACE
    )
    visit_statement_hint_list = CACHE_IN_PLACE
    visit_type = STATIC_CACHE_KEY
    visit_anon_name = ANON_NAME

    visit_propagate_attrs = PROPAGATE_ATTRS

    def visit_with_context_options(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return tuple((fn.__code__, c_key) for fn, c_key in obj)

    def visit_inspectable(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return (attrname, inspect(obj)._gen_cache_key(anon_map, bindparams))

    def visit_string_list(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return tuple(obj)

    def visit_multi(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return (
            attrname,
            (
                obj._gen_cache_key(anon_map, bindparams)
                if isinstance(obj, HasCacheKey)
                else obj
            ),
        )

    def visit_multi_list(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return (
            attrname,
            tuple(
                (
                    elem._gen_cache_key(anon_map, bindparams)
                    if isinstance(elem, HasCacheKey)
                    else elem
                )
                for elem in obj
            ),
        )

    def visit_has_cache_key_tuples(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (
            attrname,
            tuple(
                tuple(
                    elem._gen_cache_key(anon_map, bindparams)
                    for elem in tup_elem
                )
                for tup_elem in obj
            ),
        )

    def visit_has_cache_key_list(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (
            attrname,
            tuple(elem._gen_cache_key(anon_map, bindparams) for elem in obj),
        )

    def visit_executable_options(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (
            attrname,
            tuple(
                elem._gen_cache_key(anon_map, bindparams)
                for elem in obj
                if elem._is_has_cache_key
            ),
        )

    def visit_inspectable_list(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return self.visit_has_cache_key_list(
            attrname, [inspect(o) for o in obj], parent, anon_map, bindparams
        )

    def visit_clauseelement_tuples(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return self.visit_has_cache_key_tuples(
            attrname, obj, parent, anon_map, bindparams
        )

    def visit_fromclause_ordered_set(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        if not obj:
            return ()
        return (
            attrname,
            tuple([elem._gen_cache_key(anon_map, bindparams) for elem in obj]),
        )

    def visit_clauseelement_unordered_set(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        if not obj:
            return ()
        cache_keys = [
            elem._gen_cache_key(anon_map, bindparams) for elem in obj
        ]
        return (
            attrname,
            tuple(
                sorted(cache_keys)
            ),  # cache keys all start with (id_, class)
        )

    def visit_named_ddl_element(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return (attrname, obj.name)

    def visit_prefix_sequence(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        if not obj:
            return ()

        return (
            attrname,
            tuple(
                [
                    (clause._gen_cache_key(anon_map, bindparams), strval)
                    for clause, strval in obj
                ]
            ),
        )

    def visit_setup_join_tuple(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return tuple(
            (
                target._gen_cache_key(anon_map, bindparams),
                (
                    onclause._gen_cache_key(anon_map, bindparams)
                    if onclause is not None
                    else None
                ),
                (
                    from_._gen_cache_key(anon_map, bindparams)
                    if from_ is not None
                    else None
                ),
                tuple([(key, flags[key]) for key in sorted(flags)]),
            )
            for (target, onclause, from_, flags) in obj
        )

    def visit_table_hint_list(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        if not obj:
            return ()

        return (
            attrname,
            tuple(
                [
                    (
                        clause._gen_cache_key(anon_map, bindparams),
                        dialect_name,
                        text,
                    )
                    for (clause, dialect_name), text in obj.items()
                ]
            ),
        )

    def visit_plain_dict(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return (attrname, tuple([(key, obj[key]) for key in sorted(obj)]))

    def visit_dialect_options(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return (
            attrname,
            tuple(
                (
                    dialect_name,
                    tuple(
                        [
                            (key, obj[dialect_name][key])
                            for key in sorted(obj[dialect_name])
                        ]
                    ),
                )
                for dialect_name in sorted(obj)
            ),
        )

    def visit_string_clauseelement_dict(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return (
            attrname,
            tuple(
                (key, obj[key]._gen_cache_key(anon_map, bindparams))
                for key in sorted(obj)
            ),
        )

    def visit_string_multi_dict(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return (
            attrname,
            tuple(
                (
                    key,
                    (
                        value._gen_cache_key(anon_map, bindparams)
                        if isinstance(value, HasCacheKey)
                        else value
                    ),
                )
                for key, value in [(key, obj[key]) for key in sorted(obj)]
            ),
        )

    def visit_fromclause_canonical_column_collection(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        # inlining into the internals of ColumnCollection
        return (
            attrname,
            tuple(
                col._gen_cache_key(anon_map, bindparams)
                for k, col, _ in obj._collection
            ),
        )

    def visit_unknown_structure(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        anon_map[NO_CACHE] = True
        return ()

    def visit_dml_ordered_values(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        return (
            attrname,
            tuple(
                (
                    (
                        key._gen_cache_key(anon_map, bindparams)
                        if hasattr(key, "__clause_element__")
                        else key
                    ),
                    value._gen_cache_key(anon_map, bindparams),
                )
                for key, value in obj
            ),
        )

    def visit_dml_values(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        # in py37 we can assume two dictionaries created in the same
        # insert ordering will retain that sorting
        return (
            attrname,
            tuple(
                (
                    (
                        k._gen_cache_key(anon_map, bindparams)
                        if hasattr(k, "__clause_element__")
                        else k
                    ),
                    obj[k]._gen_cache_key(anon_map, bindparams),
                )
                for k in obj
            ),
        )

    def visit_dml_multi_values(
        self,
        attrname: str,
        obj: Any,
        parent: Any,
        anon_map: anon_map,
        bindparams: List[BindParameter[Any]],
    ) -> Tuple[Any, ...]:
        # multivalues are simply not cacheable right now
        anon_map[NO_CACHE] = True
        return ()


_cache_key_traversal_visitor = _CacheKeyTraversal()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test_array_utils.py ===
import numpy as np
from numpy.lib import array_utils
from numpy.testing import assert_equal


class TestByteBounds:
    def test_byte_bounds(self):
        # pointer difference matches size * itemsize
        # due to contiguity
        a = np.arange(12).reshape(3, 4)
        low, high = array_utils.byte_bounds(a)
        assert_equal(high - low, a.size * a.itemsize)

    def test_unusual_order_positive_stride(self):
        a = np.arange(12).reshape(3, 4)
        b = a.T
        low, high = array_utils.byte_bounds(b)
        assert_equal(high - low, b.size * b.itemsize)

    def test_unusual_order_negative_stride(self):
        a = np.arange(12).reshape(3, 4)
        b = a.T[::-1]
        low, high = array_utils.byte_bounds(b)
        assert_equal(high - low, b.size * b.itemsize)

    def test_strided(self):
        a = np.arange(12)
        b = a[::2]
        low, high = array_utils.byte_bounds(b)
        # the largest pointer address is lost (even numbers only in the
        # stride), and compensate addresses for striding by 2
        assert_equal(high - low, b.size * 2 * b.itemsize - b.itemsize)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\index\package_finder.py ===
"""Routines related to PyPI, indexes"""

import enum
import functools
import itertools
import logging
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from pip._vendor.packaging import specifiers
from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import InvalidVersion, _BaseVersion
from pip._vendor.packaging.version import parse as parse_version

from pip._internal.exceptions import (
    BestVersionAlreadyInstalled,
    DistributionNotFound,
    InvalidWheelFilename,
    UnsupportedWheel,
)
from pip._internal.index.collector import LinkCollector, parse_links
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.format_control import FormatControl
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.models.wheel import Wheel
from pip._internal.req import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import build_netloc
from pip._internal.utils.packaging import check_requires_python
from pip._internal.utils.unpacking import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from pip._vendor.typing_extensions import TypeGuard

__all__ = ["FormatControl", "BestCandidateResult", "PackageFinder"]


logger = getLogger(__name__)

BuildTag = Union[Tuple[()], Tuple[int, str]]
CandidateSortingKey = Tuple[int, int, int, _BaseVersion, Optional[int], BuildTag]


def _check_link_requires_python(
    link: Link,
    version_info: Tuple[int, int, int],
    ignore_requires_python: bool = False,
) -> bool:
    """
    Return whether the given Python version is compatible with a link's
    "Requires-Python" value.

    :param version_info: A 3-tuple of ints representing the Python
        major-minor-micro version to check.
    :param ignore_requires_python: Whether to ignore the "Requires-Python"
        value if the given Python version isn't compatible.
    """
    try:
        is_compatible = check_requires_python(
            link.requires_python,
            version_info=version_info,
        )
    except specifiers.InvalidSpecifier:
        logger.debug(
            "Ignoring invalid Requires-Python (%r) for link: %s",
            link.requires_python,
            link,
        )
    else:
        if not is_compatible:
            version = ".".join(map(str, version_info))
            if not ignore_requires_python:
                logger.verbose(
                    "Link requires a different Python (%s not in: %r): %s",
                    version,
                    link.requires_python,
                    link,
                )
                return False

            logger.debug(
                "Ignoring failed Requires-Python check (%s not in: %r) for link: %s",
                version,
                link.requires_python,
                link,
            )

    return True


class LinkType(enum.Enum):
    candidate = enum.auto()
    different_project = enum.auto()
    yanked = enum.auto()
    format_unsupported = enum.auto()
    format_invalid = enum.auto()
    platform_mismatch = enum.auto()
    requires_python_mismatch = enum.auto()


class LinkEvaluator:
    """
    Responsible for evaluating links for a particular project.
    """

    _py_version_re = re.compile(r"-py([123]\.?[0-9]?)$")

    # Don't include an allow_yanked default value to make sure each call
    # site considers whether yanked releases are allowed. This also causes
    # that decision to be made explicit in the calling code, which helps
    # people when reading the code.
    def __init__(
        self,
        project_name: str,
        canonical_name: str,
        formats: FrozenSet[str],
        target_python: TargetPython,
        allow_yanked: bool,
        ignore_requires_python: Optional[bool] = None,
    ) -> None:
        """
        :param project_name: The user supplied package name.
        :param canonical_name: The canonical package name.
        :param formats: The formats allowed for this package. Should be a set
            with 'binary' or 'source' or both in it.
        :param target_python: The target Python interpreter to use when
            evaluating link compatibility. This is used, for example, to
            check wheel compatibility, as well as when checking the Python
            version, e.g. the Python version embedded in a link filename
            (or egg fragment) and against an HTML link's optional PEP 503
            "data-requires-python" attribute.
        :param allow_yanked: Whether files marked as yanked (in the sense
            of PEP 592) are permitted to be candidates for install.
        :param ignore_requires_python: Whether to ignore incompatible
            PEP 503 "data-requires-python" values in HTML links. Defaults
            to False.
        """
        if ignore_requires_python is None:
            ignore_requires_python = False

        self._allow_yanked = allow_yanked
        self._canonical_name = canonical_name
        self._ignore_requires_python = ignore_requires_python
        self._formats = formats
        self._target_python = target_python

        self.project_name = project_name

    def evaluate_link(self, link: Link) -> Tuple[LinkType, str]:
        """
        Determine whether a link is a candidate for installation.

        :return: A tuple (result, detail), where *result* is an enum
            representing whether the evaluation found a candidate, or the reason
            why one is not found. If a candidate is found, *detail* will be the
            candidate's version string; if one is not found, it contains the
            reason the link fails to qualify.
        """
        version = None
        if link.is_yanked and not self._allow_yanked:
            reason = link.yanked_reason or "<none given>"
            return (LinkType.yanked, f"yanked for reason: {reason}")

        if link.egg_fragment:
            egg_info = link.egg_fragment
            ext = link.ext
        else:
            egg_info, ext = link.splitext()
            if not ext:
                return (LinkType.format_unsupported, "not a file")
            if ext not in SUPPORTED_EXTENSIONS:
                return (
                    LinkType.format_unsupported,
                    f"unsupported archive format: {ext}",
                )
            if "binary" not in self._formats and ext == WHEEL_EXTENSION:
                reason = f"No binaries permitted for {self.project_name}"
                return (LinkType.format_unsupported, reason)
            if "macosx10" in link.path and ext == ".zip":
                return (LinkType.format_unsupported, "macosx10 one")
            if ext == WHEEL_EXTENSION:
                try:
                    wheel = Wheel(link.filename)
                except InvalidWheelFilename:
                    return (
                        LinkType.format_invalid,
                        "invalid wheel filename",
                    )
                if canonicalize_name(wheel.name) != self._canonical_name:
                    reason = f"wrong project name (not {self.project_name})"
                    return (LinkType.different_project, reason)

                supported_tags = self._target_python.get_unsorted_tags()
                if not wheel.supported(supported_tags):
                    # Include the wheel's tags in the reason string to
                    # simplify troubleshooting compatibility issues.
                    file_tags = ", ".join(wheel.get_formatted_file_tags())
                    reason = (
                        f"none of the wheel's tags ({file_tags}) are compatible "
                        f"(run pip debug --verbose to show compatible tags)"
                    )
                    return (LinkType.platform_mismatch, reason)

                version = wheel.version

        # This should be up by the self.ok_binary check, but see issue 2700.
        if "source" not in self._formats and ext != WHEEL_EXTENSION:
            reason = f"No sources permitted for {self.project_name}"
            return (LinkType.format_unsupported, reason)

        if not version:
            version = _extract_version_from_fragment(
                egg_info,
                self._canonical_name,
            )
        if not version:
            reason = f"Missing project version for {self.project_name}"
            return (LinkType.format_invalid, reason)

        match = self._py_version_re.search(version)
        if match:
            version = version[: match.start()]
            py_version = match.group(1)
            if py_version != self._target_python.py_version:
                return (
                    LinkType.platform_mismatch,
                    "Python version is incorrect",
                )

        supports_python = _check_link_requires_python(
            link,
            version_info=self._target_python.py_version_info,
            ignore_requires_python=self._ignore_requires_python,
        )
        if not supports_python:
            reason = f"{version} Requires-Python {link.requires_python}"
            return (LinkType.requires_python_mismatch, reason)

        logger.debug("Found link %s, version: %s", link, version)

        return (LinkType.candidate, version)


def filter_unallowed_hashes(
    candidates: List[InstallationCandidate],
    hashes: Optional[Hashes],
    project_name: str,
) -> List[InstallationCandidate]:
    """
    Filter out candidates whose hashes aren't allowed, and return a new
    list of candidates.

    If at least one candidate has an allowed hash, then all candidates with
    either an allowed hash or no hash specified are returned.  Otherwise,
    the given candidates are returned.

    Including the candidates with no hash specified when there is a match
    allows a warning to be logged if there is a more preferred candidate
    with no hash specified.  Returning all candidates in the case of no
    matches lets pip report the hash of the candidate that would otherwise
    have been installed (e.g. permitting the user to more easily update
    their requirements file with the desired hash).
    """
    if not hashes:
        logger.debug(
            "Given no hashes to check %s links for project %r: "
            "discarding no candidates",
            len(candidates),
            project_name,
        )
        # Make sure we're not returning back the given value.
        return list(candidates)

    matches_or_no_digest = []
    # Collect the non-matches for logging purposes.
    non_matches = []
    match_count = 0
    for candidate in candidates:
        link = candidate.link
        if not link.has_hash:
            pass
        elif link.is_hash_allowed(hashes=hashes):
            match_count += 1
        else:
            non_matches.append(candidate)
            continue

        matches_or_no_digest.append(candidate)

    if match_count:
        filtered = matches_or_no_digest
    else:
        # Make sure we're not returning back the given value.
        filtered = list(candidates)

    if len(filtered) == len(candidates):
        discard_message = "discarding no candidates"
    else:
        discard_message = "discarding {} non-matches:\n  {}".format(
            len(non_matches),
            "\n  ".join(str(candidate.link) for candidate in non_matches),
        )

    logger.debug(
        "Checked %s links for project %r against %s hashes "
        "(%s matches, %s no digest): %s",
        len(candidates),
        project_name,
        hashes.digest_count,
        match_count,
        len(matches_or_no_digest) - match_count,
        discard_message,
    )

    return filtered


@dataclass
class CandidatePreferences:
    """
    Encapsulates some of the preferences for filtering and sorting
    InstallationCandidate objects.
    """

    prefer_binary: bool = False
    allow_all_prereleases: bool = False


@dataclass(frozen=True)
class BestCandidateResult:
    """A collection of candidates, returned by `PackageFinder.find_best_candidate`.

    This class is only intended to be instantiated by CandidateEvaluator's
    `compute_best_candidate()` method.

    :param all_candidates: A sequence of all available candidates found.
    :param applicable_candidates: The applicable candidates.
    :param best_candidate: The most preferred candidate found, or None
        if no applicable candidates were found.
    """

    all_candidates: List[InstallationCandidate]
    applicable_candidates: List[InstallationCandidate]
    best_candidate: Optional[InstallationCandidate]

    def __post_init__(self) -> None:
        assert set(self.applicable_candidates) <= set(self.all_candidates)

        if self.best_candidate is None:
            assert not self.applicable_candidates
        else:
            assert self.best_candidate in self.applicable_candidates


class CandidateEvaluator:
    """
    Responsible for filtering and sorting candidates for installation based
    on what tags are valid.
    """

    @classmethod
    def create(
        cls,
        project_name: str,
        target_python: Optional[TargetPython] = None,
        prefer_binary: bool = False,
        allow_all_prereleases: bool = False,
        specifier: Optional[specifiers.BaseSpecifier] = None,
        hashes: Optional[Hashes] = None,
    ) -> "CandidateEvaluator":
        """Create a CandidateEvaluator object.

        :param target_python: The target Python interpreter to use when
            checking compatibility. If None (the default), a TargetPython
            object will be constructed from the running Python.
        :param specifier: An optional object implementing `filter`
            (e.g. `packaging.specifiers.SpecifierSet`) to filter applicable
            versions.
        :param hashes: An optional collection of allowed hashes.
        """
        if target_python is None:
            target_python = TargetPython()
        if specifier is None:
            specifier = specifiers.SpecifierSet()

        supported_tags = target_python.get_sorted_tags()

        return cls(
            project_name=project_name,
            supported_tags=supported_tags,
            specifier=specifier,
            prefer_binary=prefer_binary,
            allow_all_prereleases=allow_all_prereleases,
            hashes=hashes,
        )

    def __init__(
        self,
        project_name: str,
        supported_tags: List[Tag],
        specifier: specifiers.BaseSpecifier,
        prefer_binary: bool = False,
        allow_all_prereleases: bool = False,
        hashes: Optional[Hashes] = None,
    ) -> None:
        """
        :param supported_tags: The PEP 425 tags supported by the target
            Python in order of preference (most preferred first).
        """
        self._allow_all_prereleases = allow_all_prereleases
        self._hashes = hashes
        self._prefer_binary = prefer_binary
        self._project_name = project_name
        self._specifier = specifier
        self._supported_tags = supported_tags
        # Since the index of the tag in the _supported_tags list is used
        # as a priority, precompute a map from tag to index/priority to be
        # used in wheel.find_most_preferred_tag.
        self._wheel_tag_preferences = {
            tag: idx for idx, tag in enumerate(supported_tags)
        }

    def get_applicable_candidates(
        self,
        candidates: List[InstallationCandidate],
    ) -> List[InstallationCandidate]:
        """
        Return the applicable candidates from a list of candidates.
        """
        # Using None infers from the specifier instead.
        allow_prereleases = self._allow_all_prereleases or None
        specifier = self._specifier

        # We turn the version object into a str here because otherwise
        # when we're debundled but setuptools isn't, Python will see
        # packaging.version.Version and
        # pkg_resources._vendor.packaging.version.Version as different
        # types. This way we'll use a str as a common data interchange
        # format. If we stop using the pkg_resources provided specifier
        # and start using our own, we can drop the cast to str().
        candidates_and_versions = [(c, str(c.version)) for c in candidates]
        versions = set(
            specifier.filter(
                (v for _, v in candidates_and_versions),
                prereleases=allow_prereleases,
            )
        )

        applicable_candidates = [c for c, v in candidates_and_versions if v in versions]
        filtered_applicable_candidates = filter_unallowed_hashes(
            candidates=applicable_candidates,
            hashes=self._hashes,
            project_name=self._project_name,
        )

        return sorted(filtered_applicable_candidates, key=self._sort_key)

    def _sort_key(self, candidate: InstallationCandidate) -> CandidateSortingKey:
        """
        Function to pass as the `key` argument to a call to sorted() to sort
        InstallationCandidates by preference.

        Returns a tuple such that tuples sorting as greater using Python's
        default comparison operator are more preferred.

        The preference is as follows:

        First and foremost, candidates with allowed (matching) hashes are
        always preferred over candidates without matching hashes. This is
        because e.g. if the only candidate with an allowed hash is yanked,
        we still want to use that candidate.

        Second, excepting hash considerations, candidates that have been
        yanked (in the sense of PEP 592) are always less preferred than
        candidates that haven't been yanked. Then:

        If not finding wheels, they are sorted by version only.
        If finding wheels, then the sort order is by version, then:
          1. existing installs
          2. wheels ordered via Wheel.support_index_min(self._supported_tags)
          3. source archives
        If prefer_binary was set, then all wheels are sorted above sources.

        Note: it was considered to embed this logic into the Link
              comparison operators, but then different sdist links
              with the same version, would have to be considered equal
        """
        valid_tags = self._supported_tags
        support_num = len(valid_tags)
        build_tag: BuildTag = ()
        binary_preference = 0
        link = candidate.link
        if link.is_wheel:
            # can raise InvalidWheelFilename
            wheel = Wheel(link.filename)
            try:
                pri = -(
                    wheel.find_most_preferred_tag(
                        valid_tags, self._wheel_tag_preferences
                    )
                )
            except ValueError:
                raise UnsupportedWheel(
                    f"{wheel.filename} is not a supported wheel for this platform. It "
                    "can't be sorted."
                )
            if self._prefer_binary:
                binary_preference = 1
            build_tag = wheel.build_tag
        else:  # sdist
            pri = -(support_num)
        has_allowed_hash = int(link.is_hash_allowed(self._hashes))
        yank_value = -1 * int(link.is_yanked)  # -1 for yanked.
        return (
            has_allowed_hash,
            yank_value,
            binary_preference,
            candidate.version,
            pri,
            build_tag,
        )

    def sort_best_candidate(
        self,
        candidates: List[InstallationCandidate],
    ) -> Optional[InstallationCandidate]:
        """
        Return the best candidate per the instance's sort order, or None if
        no candidate is acceptable.
        """
        if not candidates:
            return None
        best_candidate = max(candidates, key=self._sort_key)
        return best_candidate

    def compute_best_candidate(
        self,
        candidates: List[InstallationCandidate],
    ) -> BestCandidateResult:
        """
        Compute and return a `BestCandidateResult` instance.
        """
        applicable_candidates = self.get_applicable_candidates(candidates)

        best_candidate = self.sort_best_candidate(applicable_candidates)

        return BestCandidateResult(
            candidates,
            applicable_candidates=applicable_candidates,
            best_candidate=best_candidate,
        )


class PackageFinder:
    """This finds packages.

    This is meant to match easy_install's technique for looking for
    packages, by reading pages and looking for appropriate links.
    """

    def __init__(
        self,
        link_collector: LinkCollector,
        target_python: TargetPython,
        allow_yanked: bool,
        format_control: Optional[FormatControl] = None,
        candidate_prefs: Optional[CandidatePreferences] = None,
        ignore_requires_python: Optional[bool] = None,
    ) -> None:
        """
        This constructor is primarily meant to be used by the create() class
        method and from tests.

        :param format_control: A FormatControl object, used to control
            the selection of source packages / binary packages when consulting
            the index and links.
        :param candidate_prefs: Options to use when creating a
            CandidateEvaluator object.
        """
        if candidate_prefs is None:
            candidate_prefs = CandidatePreferences()

        format_control = format_control or FormatControl(set(), set())

        self._allow_yanked = allow_yanked
        self._candidate_prefs = candidate_prefs
        self._ignore_requires_python = ignore_requires_python
        self._link_collector = link_collector
        self._target_python = target_python

        self.format_control = format_control

        # These are boring links that have already been logged somehow.
        self._logged_links: Set[Tuple[Link, LinkType, str]] = set()

        # Cache of the result of finding candidates
        self._all_candidates: Dict[str, List[InstallationCandidate]] = {}
        self._best_candidates: Dict[
            Tuple[str, Optional[specifiers.BaseSpecifier], Optional[Hashes]],
            BestCandidateResult,
        ] = {}

    # Don't include an allow_yanked default value to make sure each call
    # site considers whether yanked releases are allowed. This also causes
    # that decision to be made explicit in the calling code, which helps
    # people when reading the code.
    @classmethod
    def create(
        cls,
        link_collector: LinkCollector,
        selection_prefs: SelectionPreferences,
        target_python: Optional[TargetPython] = None,
    ) -> "PackageFinder":
        """Create a PackageFinder.

        :param selection_prefs: The candidate selection preferences, as a
            SelectionPreferences object.
        :param target_python: The target Python interpreter to use when
            checking compatibility. If None (the default), a TargetPython
            object will be constructed from the running Python.
        """
        if target_python is None:
            target_python = TargetPython()

        candidate_prefs = CandidatePreferences(
            prefer_binary=selection_prefs.prefer_binary,
            allow_all_prereleases=selection_prefs.allow_all_prereleases,
        )

        return cls(
            candidate_prefs=candidate_prefs,
            link_collector=link_collector,
            target_python=target_python,
            allow_yanked=selection_prefs.allow_yanked,
            format_control=selection_prefs.format_control,
            ignore_requires_python=selection_prefs.ignore_requires_python,
        )

    @property
    def target_python(self) -> TargetPython:
        return self._target_python

    @property
    def search_scope(self) -> SearchScope:
        return self._link_collector.search_scope

    @search_scope.setter
    def search_scope(self, search_scope: SearchScope) -> None:
        self._link_collector.search_scope = search_scope

    @property
    def find_links(self) -> List[str]:
        return self._link_collector.find_links

    @property
    def index_urls(self) -> List[str]:
        return self.search_scope.index_urls

    @property
    def proxy(self) -> Optional[str]:
        return self._link_collector.session.pip_proxy

    @property
    def trusted_hosts(self) -> Iterable[str]:
        for host_port in self._link_collector.session.pip_trusted_origins:
            yield build_netloc(*host_port)

    @property
    def custom_cert(self) -> Optional[str]:
        # session.verify is either a boolean (use default bundle/no SSL
        # verification) or a string path to a custom CA bundle to use. We only
        # care about the latter.
        verify = self._link_collector.session.verify
        return verify if isinstance(verify, str) else None

    @property
    def client_cert(self) -> Optional[str]:
        cert = self._link_collector.session.cert
        assert not isinstance(cert, tuple), "pip only supports PEM client certs"
        return cert

    @property
    def allow_all_prereleases(self) -> bool:
        return self._candidate_prefs.allow_all_prereleases

    def set_allow_all_prereleases(self) -> None:
        self._candidate_prefs.allow_all_prereleases = True

    @property
    def prefer_binary(self) -> bool:
        return self._candidate_prefs.prefer_binary

    def set_prefer_binary(self) -> None:
        self._candidate_prefs.prefer_binary = True

    def requires_python_skipped_reasons(self) -> List[str]:
        reasons = {
            detail
            for _, result, detail in self._logged_links
            if result == LinkType.requires_python_mismatch
        }
        return sorted(reasons)

    def make_link_evaluator(self, project_name: str) -> LinkEvaluator:
        canonical_name = canonicalize_name(project_name)
        formats = self.format_control.get_allowed_formats(canonical_name)

        return LinkEvaluator(
            project_name=project_name,
            canonical_name=canonical_name,
            formats=formats,
            target_python=self._target_python,
            allow_yanked=self._allow_yanked,
            ignore_requires_python=self._ignore_requires_python,
        )

    def _sort_links(self, links: Iterable[Link]) -> List[Link]:
        """
        Returns elements of links in order, non-egg links first, egg links
        second, while eliminating duplicates
        """
        eggs, no_eggs = [], []
        seen: Set[Link] = set()
        for link in links:
            if link not in seen:
                seen.add(link)
                if link.egg_fragment:
                    eggs.append(link)
                else:
                    no_eggs.append(link)
        return no_eggs + eggs

    def _log_skipped_link(self, link: Link, result: LinkType, detail: str) -> None:
        entry = (link, result, detail)
        if entry not in self._logged_links:
            # Put the link at the end so the reason is more visible and because
            # the link string is usually very long.
            logger.debug("Skipping link: %s: %s", detail, link)
            self._logged_links.add(entry)

    def get_install_candidate(
        self, link_evaluator: LinkEvaluator, link: Link
    ) -> Optional[InstallationCandidate]:
        """
        If the link is a candidate for install, convert it to an
        InstallationCandidate and return it. Otherwise, return None.
        """
        result, detail = link_evaluator.evaluate_link(link)
        if result != LinkType.candidate:
            self._log_skipped_link(link, result, detail)
            return None

        try:
            return InstallationCandidate(
                name=link_evaluator.project_name,
                link=link,
                version=detail,
            )
        except InvalidVersion:
            return None

    def evaluate_links(
        self, link_evaluator: LinkEvaluator, links: Iterable[Link]
    ) -> List[InstallationCandidate]:
        """
        Convert links that are candidates to InstallationCandidate objects.
        """
        candidates = []
        for link in self._sort_links(links):
            candidate = self.get_install_candidate(link_evaluator, link)
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def process_project_url(
        self, project_url: Link, link_evaluator: LinkEvaluator
    ) -> List[InstallationCandidate]:
        logger.debug(
            "Fetching project page and analyzing links: %s",
            project_url,
        )
        index_response = self._link_collector.fetch_response(project_url)
        if index_response is None:
            return []

        page_links = list(parse_links(index_response))

        with indent_log():
            package_links = self.evaluate_links(
                link_evaluator,
                links=page_links,
            )

        return package_links

    def find_all_candidates(self, project_name: str) -> List[InstallationCandidate]:
        """Find all available InstallationCandidate for project_name

        This checks index_urls and find_links.
        All versions found are returned as an InstallationCandidate list.

        See LinkEvaluator.evaluate_link() for details on which files
        are accepted.
        """
        if project_name in self._all_candidates:
            return self._all_candidates[project_name]

        link_evaluator = self.make_link_evaluator(project_name)

        collected_sources = self._link_collector.collect_sources(
            project_name=project_name,
            candidates_from_page=functools.partial(
                self.process_project_url,
                link_evaluator=link_evaluator,
            ),
        )

        page_candidates_it = itertools.chain.from_iterable(
            source.page_candidates()
            for sources in collected_sources
            for source in sources
            if source is not None
        )
        page_candidates = list(page_candidates_it)

        file_links_it = itertools.chain.from_iterable(
            source.file_links()
            for sources in collected_sources
            for source in sources
            if source is not None
        )
        file_candidates = self.evaluate_links(
            link_evaluator,
            sorted(file_links_it, reverse=True),
        )

        if logger.isEnabledFor(logging.DEBUG) and file_candidates:
            paths = []
            for candidate in file_candidates:
                assert candidate.link.url  # we need to have a URL
                try:
                    paths.append(candidate.link.file_path)
                except Exception:
                    paths.append(candidate.link.url)  # it's not a local file

            logger.debug("Local files found: %s", ", ".join(paths))

        # This is an intentional priority ordering
        self._all_candidates[project_name] = file_candidates + page_candidates

        return self._all_candidates[project_name]

    def make_candidate_evaluator(
        self,
        project_name: str,
        specifier: Optional[specifiers.BaseSpecifier] = None,
        hashes: Optional[Hashes] = None,
    ) -> CandidateEvaluator:
        """Create a CandidateEvaluator object to use."""
        candidate_prefs = self._candidate_prefs
        return CandidateEvaluator.create(
            project_name=project_name,
            target_python=self._target_python,
            prefer_binary=candidate_prefs.prefer_binary,
            allow_all_prereleases=candidate_prefs.allow_all_prereleases,
            specifier=specifier,
            hashes=hashes,
        )

    def find_best_candidate(
        self,
        project_name: str,
        specifier: Optional[specifiers.BaseSpecifier] = None,
        hashes: Optional[Hashes] = None,
    ) -> BestCandidateResult:
        """Find matches for the given project and specifier.

        :param specifier: An optional object implementing `filter`
            (e.g. `packaging.specifiers.SpecifierSet`) to filter applicable
            versions.

        :return: A `BestCandidateResult` instance.
        """
        if (project_name, specifier, hashes) in self._best_candidates:
            return self._best_candidates[project_name, specifier, hashes]

        candidates = self.find_all_candidates(project_name)
        candidate_evaluator = self.make_candidate_evaluator(
            project_name=project_name,
            specifier=specifier,
            hashes=hashes,
        )
        self._best_candidates[project_name, specifier, hashes] = (
            candidate_evaluator.compute_best_candidate(candidates)
        )

        return self._best_candidates[project_name, specifier, hashes]

    def find_requirement(
        self, req: InstallRequirement, upgrade: bool
    ) -> Optional[InstallationCandidate]:
        """Try to find a Link matching req

        Expects req, an InstallRequirement and upgrade, a boolean
        Returns a InstallationCandidate if found,
        Raises DistributionNotFound or BestVersionAlreadyInstalled otherwise
        """
        name = req.name
        assert name is not None, "find_requirement() called with no name"

        hashes = req.hashes(trust_internet=False)
        best_candidate_result = self.find_best_candidate(
            name,
            specifier=req.specifier,
            hashes=hashes,
        )
        best_candidate = best_candidate_result.best_candidate

        installed_version: Optional[_BaseVersion] = None
        if req.satisfied_by is not None:
            installed_version = req.satisfied_by.version

        def _format_versions(cand_iter: Iterable[InstallationCandidate]) -> str:
            # This repeated parse_version and str() conversion is needed to
            # handle different vendoring sources from pip and pkg_resources.
            # If we stop using the pkg_resources provided specifier and start
            # using our own, we can drop the cast to str().
            return (
                ", ".join(
                    sorted(
                        {str(c.version) for c in cand_iter},
                        key=parse_version,
                    )
                )
                or "none"
            )

        if installed_version is None and best_candidate is None:
            logger.critical(
                "Could not find a version that satisfies the requirement %s "
                "(from versions: %s)",
                req,
                _format_versions(best_candidate_result.all_candidates),
            )

            raise DistributionNotFound(f"No matching distribution found for {req}")

        def _should_install_candidate(
            candidate: Optional[InstallationCandidate],
        ) -> "TypeGuard[InstallationCandidate]":
            if installed_version is None:
                return True
            if best_candidate is None:
                return False
            return best_candidate.version > installed_version

        if not upgrade and installed_version is not None:
            if _should_install_candidate(best_candidate):
                logger.debug(
                    "Existing installed version (%s) satisfies requirement "
                    "(most up-to-date version is %s)",
                    installed_version,
                    best_candidate.version,
                )
            else:
                logger.debug(
                    "Existing installed version (%s) is most up-to-date and "
                    "satisfies requirement",
                    installed_version,
                )
            return None

        if _should_install_candidate(best_candidate):
            logger.debug(
                "Using version %s (newest of versions: %s)",
                best_candidate.version,
                _format_versions(best_candidate_result.applicable_candidates),
            )
            return best_candidate

        # We have an existing version, and its the best version
        logger.debug(
            "Installed version (%s) is most up-to-date (past versions: %s)",
            installed_version,
            _format_versions(best_candidate_result.applicable_candidates),
        )
        raise BestVersionAlreadyInstalled


def _find_name_version_sep(fragment: str, canonical_name: str) -> int:
    """Find the separator's index based on the package's canonical name.

    :param fragment: A <package>+<version> filename "fragment" (stem) or
        egg fragment.
    :param canonical_name: The package's canonical name.

    This function is needed since the canonicalized name does not necessarily
    have the same length as the egg info's name part. An example::

    >>> fragment = 'foo__bar-1.0'
    >>> canonical_name = 'foo-bar'
    >>> _find_name_version_sep(fragment, canonical_name)
    8
    """
    # Project name and version must be separated by one single dash. Find all
    # occurrences of dashes; if the string in front of it matches the canonical
    # name, this is the one separating the name and version parts.
    for i, c in enumerate(fragment):
        if c != "-":
            continue
        if canonicalize_name(fragment[:i]) == canonical_name:
            return i
    raise ValueError(f"{fragment} does not match {canonical_name}")


def _extract_version_from_fragment(fragment: str, canonical_name: str) -> Optional[str]:
    """Parse the version string from a <package>+<version> filename
    "fragment" (stem) or egg fragment.

    :param fragment: The string to parse. E.g. foo-2.1
    :param canonical_name: The canonicalized name of the package this
        belongs to.
    """
    try:
        version_start = _find_name_version_sep(fragment, canonical_name) + 1
    except ValueError:
        return None
    version = fragment[version_start:]
    if not version:
        return None
    return version

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\parser.py ===
"""Parse tokens from the lexer into nodes for the compiler."""

import typing
import typing as t

from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

_ImportInclude = t.TypeVar("_ImportInclude", nodes.Import, nodes.Include)
_MacroCall = t.TypeVar("_MacroCall", nodes.Macro, nodes.CallBlock)

_statement_keywords = frozenset(
    [
        "for",
        "if",
        "block",
        "extends",
        "print",
        "macro",
        "include",
        "from",
        "import",
        "set",
        "with",
        "autoescape",
    ]
)
_compare_operators = frozenset(["eq", "ne", "lt", "lteq", "gt", "gteq"])

_math_nodes: t.Dict[str, t.Type[nodes.Expr]] = {
    "add": nodes.Add,
    "sub": nodes.Sub,
    "mul": nodes.Mul,
    "div": nodes.Div,
    "floordiv": nodes.FloorDiv,
    "mod": nodes.Mod,
}


class Parser:
    """This is the central parsing class Jinja uses.  It's passed to
    extensions and can be used to parse expressions or statements.
    """

    def __init__(
        self,
        environment: "Environment",
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        state: t.Optional[str] = None,
    ) -> None:
        self.environment = environment
        self.stream = environment._tokenize(source, name, filename, state)
        self.name = name
        self.filename = filename
        self.closed = False
        self.extensions: t.Dict[
            str, t.Callable[[Parser], t.Union[nodes.Node, t.List[nodes.Node]]]
        ] = {}
        for extension in environment.iter_extensions():
            for tag in extension.tags:
                self.extensions[tag] = extension.parse
        self._last_identifier = 0
        self._tag_stack: t.List[str] = []
        self._end_token_stack: t.List[t.Tuple[str, ...]] = []

    def fail(
        self,
        msg: str,
        lineno: t.Optional[int] = None,
        exc: t.Type[TemplateSyntaxError] = TemplateSyntaxError,
    ) -> "te.NoReturn":
        """Convenience method that raises `exc` with the message, passed
        line number or last line number as well as the current name and
        filename.
        """
        if lineno is None:
            lineno = self.stream.current.lineno
        raise exc(msg, lineno, self.name, self.filename)

    def _fail_ut_eof(
        self,
        name: t.Optional[str],
        end_token_stack: t.List[t.Tuple[str, ...]],
        lineno: t.Optional[int],
    ) -> "te.NoReturn":
        expected: t.Set[str] = set()
        for exprs in end_token_stack:
            expected.update(map(describe_token_expr, exprs))
        if end_token_stack:
            currently_looking: t.Optional[str] = " or ".join(
                map(repr, map(describe_token_expr, end_token_stack[-1]))
            )
        else:
            currently_looking = None

        if name is None:
            message = ["Unexpected end of template."]
        else:
            message = [f"Encountered unknown tag {name!r}."]

        if currently_looking:
            if name is not None and name in expected:
                message.append(
                    "You probably made a nesting mistake. Jinja is expecting this tag,"
                    f" but currently looking for {currently_looking}."
                )
            else:
                message.append(
                    f"Jinja was looking for the following tags: {currently_looking}."
                )

        if self._tag_stack:
            message.append(
                "The innermost block that needs to be closed is"
                f" {self._tag_stack[-1]!r}."
            )

        self.fail(" ".join(message), lineno)

    def fail_unknown_tag(
        self, name: str, lineno: t.Optional[int] = None
    ) -> "te.NoReturn":
        """Called if the parser encounters an unknown tag.  Tries to fail
        with a human readable error message that could help to identify
        the problem.
        """
        self._fail_ut_eof(name, self._end_token_stack, lineno)

    def fail_eof(
        self,
        end_tokens: t.Optional[t.Tuple[str, ...]] = None,
        lineno: t.Optional[int] = None,
    ) -> "te.NoReturn":
        """Like fail_unknown_tag but for end of template situations."""
        stack = list(self._end_token_stack)
        if end_tokens is not None:
            stack.append(end_tokens)
        self._fail_ut_eof(None, stack, lineno)

    def is_tuple_end(
        self, extra_end_rules: t.Optional[t.Tuple[str, ...]] = None
    ) -> bool:
        """Are we at the end of a tuple?"""
        if self.stream.current.type in ("variable_end", "block_end", "rparen"):
            return True
        elif extra_end_rules is not None:
            return self.stream.current.test_any(extra_end_rules)  # type: ignore
        return False

    def free_identifier(self, lineno: t.Optional[int] = None) -> nodes.InternalName:
        """Return a new free identifier as :class:`~jinja2.nodes.InternalName`."""
        self._last_identifier += 1
        rv = object.__new__(nodes.InternalName)
        nodes.Node.__init__(rv, f"fi{self._last_identifier}", lineno=lineno)
        return rv

    def parse_statement(self) -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """Parse a single statement."""
        token = self.stream.current
        if token.type != "name":
            self.fail("tag name expected", token.lineno)
        self._tag_stack.append(token.value)
        pop_tag = True
        try:
            if token.value in _statement_keywords:
                f = getattr(self, f"parse_{self.stream.current.value}")
                return f()  # type: ignore
            if token.value == "call":
                return self.parse_call_block()
            if token.value == "filter":
                return self.parse_filter_block()
            ext = self.extensions.get(token.value)
            if ext is not None:
                return ext(self)

            # did not work out, remove the token we pushed by accident
            # from the stack so that the unknown tag fail function can
            # produce a proper error message.
            self._tag_stack.pop()
            pop_tag = False
            self.fail_unknown_tag(token.value, token.lineno)
        finally:
            if pop_tag:
                self._tag_stack.pop()

    def parse_statements(
        self, end_tokens: t.Tuple[str, ...], drop_needle: bool = False
    ) -> t.List[nodes.Node]:
        """Parse multiple statements into a list until one of the end tokens
        is reached.  This is used to parse the body of statements as it also
        parses template data if appropriate.  The parser checks first if the
        current token is a colon and skips it if there is one.  Then it checks
        for the block end and parses until if one of the `end_tokens` is
        reached.  Per default the active token in the stream at the end of
        the call is the matched end token.  If this is not wanted `drop_needle`
        can be set to `True` and the end token is removed.
        """
        # the first token may be a colon for python compatibility
        self.stream.skip_if("colon")

        # in the future it would be possible to add whole code sections
        # by adding some sort of end of statement token and parsing those here.
        self.stream.expect("block_end")
        result = self.subparse(end_tokens)

        # we reached the end of the template too early, the subparser
        # does not check for this, so we do that now
        if self.stream.current.type == "eof":
            self.fail_eof(end_tokens)

        if drop_needle:
            next(self.stream)
        return result

    def parse_set(self) -> t.Union[nodes.Assign, nodes.AssignBlock]:
        """Parse an assign statement."""
        lineno = next(self.stream).lineno
        target = self.parse_assign_target(with_namespace=True)
        if self.stream.skip_if("assign"):
            expr = self.parse_tuple()
            return nodes.Assign(target, expr, lineno=lineno)
        filter_node = self.parse_filter(None)
        body = self.parse_statements(("name:endset",), drop_needle=True)
        return nodes.AssignBlock(target, filter_node, body, lineno=lineno)

    def parse_for(self) -> nodes.For:
        """Parse a for loop."""
        lineno = self.stream.expect("name:for").lineno
        target = self.parse_assign_target(extra_end_rules=("name:in",))
        self.stream.expect("name:in")
        iter = self.parse_tuple(
            with_condexpr=False, extra_end_rules=("name:recursive",)
        )
        test = None
        if self.stream.skip_if("name:if"):
            test = self.parse_expression()
        recursive = self.stream.skip_if("name:recursive")
        body = self.parse_statements(("name:endfor", "name:else"))
        if next(self.stream).value == "endfor":
            else_ = []
        else:
            else_ = self.parse_statements(("name:endfor",), drop_needle=True)
        return nodes.For(target, iter, body, else_, test, recursive, lineno=lineno)

    def parse_if(self) -> nodes.If:
        """Parse an if construct."""
        node = result = nodes.If(lineno=self.stream.expect("name:if").lineno)
        while True:
            node.test = self.parse_tuple(with_condexpr=False)
            node.body = self.parse_statements(("name:elif", "name:else", "name:endif"))
            node.elif_ = []
            node.else_ = []
            token = next(self.stream)
            if token.test("name:elif"):
                node = nodes.If(lineno=self.stream.current.lineno)
                result.elif_.append(node)
                continue
            elif token.test("name:else"):
                result.else_ = self.parse_statements(("name:endif",), drop_needle=True)
            break
        return result

    def parse_with(self) -> nodes.With:
        node = nodes.With(lineno=next(self.stream).lineno)
        targets: t.List[nodes.Expr] = []
        values: t.List[nodes.Expr] = []
        while self.stream.current.type != "block_end":
            if targets:
                self.stream.expect("comma")
            target = self.parse_assign_target()
            target.set_ctx("param")
            targets.append(target)
            self.stream.expect("assign")
            values.append(self.parse_expression())
        node.targets = targets
        node.values = values
        node.body = self.parse_statements(("name:endwith",), drop_needle=True)
        return node

    def parse_autoescape(self) -> nodes.Scope:
        node = nodes.ScopedEvalContextModifier(lineno=next(self.stream).lineno)
        node.options = [nodes.Keyword("autoescape", self.parse_expression())]
        node.body = self.parse_statements(("name:endautoescape",), drop_needle=True)
        return nodes.Scope([node])

    def parse_block(self) -> nodes.Block:
        node = nodes.Block(lineno=next(self.stream).lineno)
        node.name = self.stream.expect("name").value
        node.scoped = self.stream.skip_if("name:scoped")
        node.required = self.stream.skip_if("name:required")

        # common problem people encounter when switching from django
        # to jinja.  we do not support hyphens in block names, so let's
        # raise a nicer error message in that case.
        if self.stream.current.type == "sub":
            self.fail(
                "Block names in Jinja have to be valid Python identifiers and may not"
                " contain hyphens, use an underscore instead."
            )

        node.body = self.parse_statements(("name:endblock",), drop_needle=True)

        # enforce that required blocks only contain whitespace or comments
        # by asserting that the body, if not empty, is just TemplateData nodes
        # with whitespace data
        if node.required:
            for body_node in node.body:
                if not isinstance(body_node, nodes.Output) or any(
                    not isinstance(output_node, nodes.TemplateData)
                    or not output_node.data.isspace()
                    for output_node in body_node.nodes
                ):
                    self.fail("Required blocks can only contain comments or whitespace")

        self.stream.skip_if("name:" + node.name)
        return node

    def parse_extends(self) -> nodes.Extends:
        node = nodes.Extends(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        return node

    def parse_import_context(
        self, node: _ImportInclude, default: bool
    ) -> _ImportInclude:
        if self.stream.current.test_any(
            "name:with", "name:without"
        ) and self.stream.look().test("name:context"):
            node.with_context = next(self.stream).value == "with"
            self.stream.skip()
        else:
            node.with_context = default
        return node

    def parse_include(self) -> nodes.Include:
        node = nodes.Include(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        if self.stream.current.test("name:ignore") and self.stream.look().test(
            "name:missing"
        ):
            node.ignore_missing = True
            self.stream.skip(2)
        else:
            node.ignore_missing = False
        return self.parse_import_context(node, True)

    def parse_import(self) -> nodes.Import:
        node = nodes.Import(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        self.stream.expect("name:as")
        node.target = self.parse_assign_target(name_only=True).name
        return self.parse_import_context(node, False)

    def parse_from(self) -> nodes.FromImport:
        node = nodes.FromImport(lineno=next(self.stream).lineno)
        node.template = self.parse_expression()
        self.stream.expect("name:import")
        node.names = []

        def parse_context() -> bool:
            if self.stream.current.value in {
                "with",
                "without",
            } and self.stream.look().test("name:context"):
                node.with_context = next(self.stream).value == "with"
                self.stream.skip()
                return True
            return False

        while True:
            if node.names:
                self.stream.expect("comma")
            if self.stream.current.type == "name":
                if parse_context():
                    break
                target = self.parse_assign_target(name_only=True)
                if target.name.startswith("_"):
                    self.fail(
                        "names starting with an underline can not be imported",
                        target.lineno,
                        exc=TemplateAssertionError,
                    )
                if self.stream.skip_if("name:as"):
                    alias = self.parse_assign_target(name_only=True)
                    node.names.append((target.name, alias.name))
                else:
                    node.names.append(target.name)
                if parse_context() or self.stream.current.type != "comma":
                    break
            else:
                self.stream.expect("name")
        if not hasattr(node, "with_context"):
            node.with_context = False
        return node

    def parse_signature(self, node: _MacroCall) -> None:
        args = node.args = []
        defaults = node.defaults = []
        self.stream.expect("lparen")
        while self.stream.current.type != "rparen":
            if args:
                self.stream.expect("comma")
            arg = self.parse_assign_target(name_only=True)
            arg.set_ctx("param")
            if self.stream.skip_if("assign"):
                defaults.append(self.parse_expression())
            elif defaults:
                self.fail("non-default argument follows default argument")
            args.append(arg)
        self.stream.expect("rparen")

    def parse_call_block(self) -> nodes.CallBlock:
        node = nodes.CallBlock(lineno=next(self.stream).lineno)
        if self.stream.current.type == "lparen":
            self.parse_signature(node)
        else:
            node.args = []
            node.defaults = []

        call_node = self.parse_expression()
        if not isinstance(call_node, nodes.Call):
            self.fail("expected call", node.lineno)
        node.call = call_node
        node.body = self.parse_statements(("name:endcall",), drop_needle=True)
        return node

    def parse_filter_block(self) -> nodes.FilterBlock:
        node = nodes.FilterBlock(lineno=next(self.stream).lineno)
        node.filter = self.parse_filter(None, start_inline=True)  # type: ignore
        node.body = self.parse_statements(("name:endfilter",), drop_needle=True)
        return node

    def parse_macro(self) -> nodes.Macro:
        node = nodes.Macro(lineno=next(self.stream).lineno)
        node.name = self.parse_assign_target(name_only=True).name
        self.parse_signature(node)
        node.body = self.parse_statements(("name:endmacro",), drop_needle=True)
        return node

    def parse_print(self) -> nodes.Output:
        node = nodes.Output(lineno=next(self.stream).lineno)
        node.nodes = []
        while self.stream.current.type != "block_end":
            if node.nodes:
                self.stream.expect("comma")
            node.nodes.append(self.parse_expression())
        return node

    @typing.overload
    def parse_assign_target(
        self, with_tuple: bool = ..., name_only: "te.Literal[True]" = ...
    ) -> nodes.Name: ...

    @typing.overload
    def parse_assign_target(
        self,
        with_tuple: bool = True,
        name_only: bool = False,
        extra_end_rules: t.Optional[t.Tuple[str, ...]] = None,
        with_namespace: bool = False,
    ) -> t.Union[nodes.NSRef, nodes.Name, nodes.Tuple]: ...

    def parse_assign_target(
        self,
        with_tuple: bool = True,
        name_only: bool = False,
        extra_end_rules: t.Optional[t.Tuple[str, ...]] = None,
        with_namespace: bool = False,
    ) -> t.Union[nodes.NSRef, nodes.Name, nodes.Tuple]:
        """Parse an assignment target.  As Jinja allows assignments to
        tuples, this function can parse all allowed assignment targets.  Per
        default assignments to tuples are parsed, that can be disable however
        by setting `with_tuple` to `False`.  If only assignments to names are
        wanted `name_only` can be set to `True`.  The `extra_end_rules`
        parameter is forwarded to the tuple parsing function.  If
        `with_namespace` is enabled, a namespace assignment may be parsed.
        """
        target: nodes.Expr

        if name_only:
            token = self.stream.expect("name")
            target = nodes.Name(token.value, "store", lineno=token.lineno)
        else:
            if with_tuple:
                target = self.parse_tuple(
                    simplified=True,
                    extra_end_rules=extra_end_rules,
                    with_namespace=with_namespace,
                )
            else:
                target = self.parse_primary(with_namespace=with_namespace)

            target.set_ctx("store")

        if not target.can_assign():
            self.fail(
                f"can't assign to {type(target).__name__.lower()!r}", target.lineno
            )

        return target  # type: ignore

    def parse_expression(self, with_condexpr: bool = True) -> nodes.Expr:
        """Parse an expression.  Per default all expressions are parsed, if
        the optional `with_condexpr` parameter is set to `False` conditional
        expressions are not parsed.
        """
        if with_condexpr:
            return self.parse_condexpr()
        return self.parse_or()

    def parse_condexpr(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        expr1 = self.parse_or()
        expr3: t.Optional[nodes.Expr]

        while self.stream.skip_if("name:if"):
            expr2 = self.parse_or()
            if self.stream.skip_if("name:else"):
                expr3 = self.parse_condexpr()
            else:
                expr3 = None
            expr1 = nodes.CondExpr(expr2, expr1, expr3, lineno=lineno)
            lineno = self.stream.current.lineno
        return expr1

    def parse_or(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_and()
        while self.stream.skip_if("name:or"):
            right = self.parse_and()
            left = nodes.Or(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_and(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_not()
        while self.stream.skip_if("name:and"):
            right = self.parse_not()
            left = nodes.And(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_not(self) -> nodes.Expr:
        if self.stream.current.test("name:not"):
            lineno = next(self.stream).lineno
            return nodes.Not(self.parse_not(), lineno=lineno)
        return self.parse_compare()

    def parse_compare(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        expr = self.parse_math1()
        ops = []
        while True:
            token_type = self.stream.current.type
            if token_type in _compare_operators:
                next(self.stream)
                ops.append(nodes.Operand(token_type, self.parse_math1()))
            elif self.stream.skip_if("name:in"):
                ops.append(nodes.Operand("in", self.parse_math1()))
            elif self.stream.current.test("name:not") and self.stream.look().test(
                "name:in"
            ):
                self.stream.skip(2)
                ops.append(nodes.Operand("notin", self.parse_math1()))
            else:
                break
            lineno = self.stream.current.lineno
        if not ops:
            return expr
        return nodes.Compare(expr, ops, lineno=lineno)

    def parse_math1(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_concat()
        while self.stream.current.type in ("add", "sub"):
            cls = _math_nodes[self.stream.current.type]
            next(self.stream)
            right = self.parse_concat()
            left = cls(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_concat(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        args = [self.parse_math2()]
        while self.stream.current.type == "tilde":
            next(self.stream)
            args.append(self.parse_math2())
        if len(args) == 1:
            return args[0]
        return nodes.Concat(args, lineno=lineno)

    def parse_math2(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_pow()
        while self.stream.current.type in ("mul", "div", "floordiv", "mod"):
            cls = _math_nodes[self.stream.current.type]
            next(self.stream)
            right = self.parse_pow()
            left = cls(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_pow(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        left = self.parse_unary()
        while self.stream.current.type == "pow":
            next(self.stream)
            right = self.parse_unary()
            left = nodes.Pow(left, right, lineno=lineno)
            lineno = self.stream.current.lineno
        return left

    def parse_unary(self, with_filter: bool = True) -> nodes.Expr:
        token_type = self.stream.current.type
        lineno = self.stream.current.lineno
        node: nodes.Expr

        if token_type == "sub":
            next(self.stream)
            node = nodes.Neg(self.parse_unary(False), lineno=lineno)
        elif token_type == "add":
            next(self.stream)
            node = nodes.Pos(self.parse_unary(False), lineno=lineno)
        else:
            node = self.parse_primary()
        node = self.parse_postfix(node)
        if with_filter:
            node = self.parse_filter_expr(node)
        return node

    def parse_primary(self, with_namespace: bool = False) -> nodes.Expr:
        """Parse a name or literal value. If ``with_namespace`` is enabled, also
        parse namespace attr refs, for use in assignments."""
        token = self.stream.current
        node: nodes.Expr
        if token.type == "name":
            next(self.stream)
            if token.value in ("true", "false", "True", "False"):
                node = nodes.Const(token.value in ("true", "True"), lineno=token.lineno)
            elif token.value in ("none", "None"):
                node = nodes.Const(None, lineno=token.lineno)
            elif with_namespace and self.stream.current.type == "dot":
                # If namespace attributes are allowed at this point, and the next
                # token is a dot, produce a namespace reference.
                next(self.stream)
                attr = self.stream.expect("name")
                node = nodes.NSRef(token.value, attr.value, lineno=token.lineno)
            else:
                node = nodes.Name(token.value, "load", lineno=token.lineno)
        elif token.type == "string":
            next(self.stream)
            buf = [token.value]
            lineno = token.lineno
            while self.stream.current.type == "string":
                buf.append(self.stream.current.value)
                next(self.stream)
            node = nodes.Const("".join(buf), lineno=lineno)
        elif token.type in ("integer", "float"):
            next(self.stream)
            node = nodes.Const(token.value, lineno=token.lineno)
        elif token.type == "lparen":
            next(self.stream)
            node = self.parse_tuple(explicit_parentheses=True)
            self.stream.expect("rparen")
        elif token.type == "lbracket":
            node = self.parse_list()
        elif token.type == "lbrace":
            node = self.parse_dict()
        else:
            self.fail(f"unexpected {describe_token(token)!r}", token.lineno)
        return node

    def parse_tuple(
        self,
        simplified: bool = False,
        with_condexpr: bool = True,
        extra_end_rules: t.Optional[t.Tuple[str, ...]] = None,
        explicit_parentheses: bool = False,
        with_namespace: bool = False,
    ) -> t.Union[nodes.Tuple, nodes.Expr]:
        """Works like `parse_expression` but if multiple expressions are
        delimited by a comma a :class:`~jinja2.nodes.Tuple` node is created.
        This method could also return a regular expression instead of a tuple
        if no commas where found.

        The default parsing mode is a full tuple.  If `simplified` is `True`
        only names and literals are parsed; ``with_namespace`` allows namespace
        attr refs as well. The `no_condexpr` parameter is forwarded to
        :meth:`parse_expression`.

        Because tuples do not require delimiters and may end in a bogus comma
        an extra hint is needed that marks the end of a tuple.  For example
        for loops support tuples between `for` and `in`.  In that case the
        `extra_end_rules` is set to ``['name:in']``.

        `explicit_parentheses` is true if the parsing was triggered by an
        expression in parentheses.  This is used to figure out if an empty
        tuple is a valid expression or not.
        """
        lineno = self.stream.current.lineno
        if simplified:

            def parse() -> nodes.Expr:
                return self.parse_primary(with_namespace=with_namespace)

        else:

            def parse() -> nodes.Expr:
                return self.parse_expression(with_condexpr=with_condexpr)

        args: t.List[nodes.Expr] = []
        is_tuple = False

        while True:
            if args:
                self.stream.expect("comma")
            if self.is_tuple_end(extra_end_rules):
                break
            args.append(parse())
            if self.stream.current.type == "comma":
                is_tuple = True
            else:
                break
            lineno = self.stream.current.lineno

        if not is_tuple:
            if args:
                return args[0]

            # if we don't have explicit parentheses, an empty tuple is
            # not a valid expression.  This would mean nothing (literally
            # nothing) in the spot of an expression would be an empty
            # tuple.
            if not explicit_parentheses:
                self.fail(
                    "Expected an expression,"
                    f" got {describe_token(self.stream.current)!r}"
                )

        return nodes.Tuple(args, "load", lineno=lineno)

    def parse_list(self) -> nodes.List:
        token = self.stream.expect("lbracket")
        items: t.List[nodes.Expr] = []
        while self.stream.current.type != "rbracket":
            if items:
                self.stream.expect("comma")
            if self.stream.current.type == "rbracket":
                break
            items.append(self.parse_expression())
        self.stream.expect("rbracket")
        return nodes.List(items, lineno=token.lineno)

    def parse_dict(self) -> nodes.Dict:
        token = self.stream.expect("lbrace")
        items: t.List[nodes.Pair] = []
        while self.stream.current.type != "rbrace":
            if items:
                self.stream.expect("comma")
            if self.stream.current.type == "rbrace":
                break
            key = self.parse_expression()
            self.stream.expect("colon")
            value = self.parse_expression()
            items.append(nodes.Pair(key, value, lineno=key.lineno))
        self.stream.expect("rbrace")
        return nodes.Dict(items, lineno=token.lineno)

    def parse_postfix(self, node: nodes.Expr) -> nodes.Expr:
        while True:
            token_type = self.stream.current.type
            if token_type == "dot" or token_type == "lbracket":
                node = self.parse_subscript(node)
            # calls are valid both after postfix expressions (getattr
            # and getitem) as well as filters and tests
            elif token_type == "lparen":
                node = self.parse_call(node)
            else:
                break
        return node

    def parse_filter_expr(self, node: nodes.Expr) -> nodes.Expr:
        while True:
            token_type = self.stream.current.type
            if token_type == "pipe":
                node = self.parse_filter(node)  # type: ignore
            elif token_type == "name" and self.stream.current.value == "is":
                node = self.parse_test(node)
            # calls are valid both after postfix expressions (getattr
            # and getitem) as well as filters and tests
            elif token_type == "lparen":
                node = self.parse_call(node)
            else:
                break
        return node

    def parse_subscript(
        self, node: nodes.Expr
    ) -> t.Union[nodes.Getattr, nodes.Getitem]:
        token = next(self.stream)
        arg: nodes.Expr

        if token.type == "dot":
            attr_token = self.stream.current
            next(self.stream)
            if attr_token.type == "name":
                return nodes.Getattr(
                    node, attr_token.value, "load", lineno=token.lineno
                )
            elif attr_token.type != "integer":
                self.fail("expected name or number", attr_token.lineno)
            arg = nodes.Const(attr_token.value, lineno=attr_token.lineno)
            return nodes.Getitem(node, arg, "load", lineno=token.lineno)
        if token.type == "lbracket":
            args: t.List[nodes.Expr] = []
            while self.stream.current.type != "rbracket":
                if args:
                    self.stream.expect("comma")
                args.append(self.parse_subscribed())
            self.stream.expect("rbracket")
            if len(args) == 1:
                arg = args[0]
            else:
                arg = nodes.Tuple(args, "load", lineno=token.lineno)
            return nodes.Getitem(node, arg, "load", lineno=token.lineno)
        self.fail("expected subscript expression", token.lineno)

    def parse_subscribed(self) -> nodes.Expr:
        lineno = self.stream.current.lineno
        args: t.List[t.Optional[nodes.Expr]]

        if self.stream.current.type == "colon":
            next(self.stream)
            args = [None]
        else:
            node = self.parse_expression()
            if self.stream.current.type != "colon":
                return node
            next(self.stream)
            args = [node]

        if self.stream.current.type == "colon":
            args.append(None)
        elif self.stream.current.type not in ("rbracket", "comma"):
            args.append(self.parse_expression())
        else:
            args.append(None)

        if self.stream.current.type == "colon":
            next(self.stream)
            if self.stream.current.type not in ("rbracket", "comma"):
                args.append(self.parse_expression())
            else:
                args.append(None)
        else:
            args.append(None)

        return nodes.Slice(lineno=lineno, *args)  # noqa: B026

    def parse_call_args(
        self,
    ) -> t.Tuple[
        t.List[nodes.Expr],
        t.List[nodes.Keyword],
        t.Optional[nodes.Expr],
        t.Optional[nodes.Expr],
    ]:
        token = self.stream.expect("lparen")
        args = []
        kwargs = []
        dyn_args = None
        dyn_kwargs = None
        require_comma = False

        def ensure(expr: bool) -> None:
            if not expr:
                self.fail("invalid syntax for function call expression", token.lineno)

        while self.stream.current.type != "rparen":
            if require_comma:
                self.stream.expect("comma")

                # support for trailing comma
                if self.stream.current.type == "rparen":
                    break

            if self.stream.current.type == "mul":
                ensure(dyn_args is None and dyn_kwargs is None)
                next(self.stream)
                dyn_args = self.parse_expression()
            elif self.stream.current.type == "pow":
                ensure(dyn_kwargs is None)
                next(self.stream)
                dyn_kwargs = self.parse_expression()
            else:
                if (
                    self.stream.current.type == "name"
                    and self.stream.look().type == "assign"
                ):
                    # Parsing a kwarg
                    ensure(dyn_kwargs is None)
                    key = self.stream.current.value
                    self.stream.skip(2)
                    value = self.parse_expression()
                    kwargs.append(nodes.Keyword(key, value, lineno=value.lineno))
                else:
                    # Parsing an arg
                    ensure(dyn_args is None and dyn_kwargs is None and not kwargs)
                    args.append(self.parse_expression())

            require_comma = True

        self.stream.expect("rparen")
        return args, kwargs, dyn_args, dyn_kwargs

    def parse_call(self, node: nodes.Expr) -> nodes.Call:
        # The lparen will be expected in parse_call_args, but the lineno
        # needs to be recorded before the stream is advanced.
        token = self.stream.current
        args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
        return nodes.Call(node, args, kwargs, dyn_args, dyn_kwargs, lineno=token.lineno)

    def parse_filter(
        self, node: t.Optional[nodes.Expr], start_inline: bool = False
    ) -> t.Optional[nodes.Expr]:
        while self.stream.current.type == "pipe" or start_inline:
            if not start_inline:
                next(self.stream)
            token = self.stream.expect("name")
            name = token.value
            while self.stream.current.type == "dot":
                next(self.stream)
                name += "." + self.stream.expect("name").value
            if self.stream.current.type == "lparen":
                args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
            else:
                args = []
                kwargs = []
                dyn_args = dyn_kwargs = None
            node = nodes.Filter(
                node, name, args, kwargs, dyn_args, dyn_kwargs, lineno=token.lineno
            )
            start_inline = False
        return node

    def parse_test(self, node: nodes.Expr) -> nodes.Expr:
        token = next(self.stream)
        if self.stream.current.test("name:not"):
            next(self.stream)
            negated = True
        else:
            negated = False
        name = self.stream.expect("name").value
        while self.stream.current.type == "dot":
            next(self.stream)
            name += "." + self.stream.expect("name").value
        dyn_args = dyn_kwargs = None
        kwargs: t.List[nodes.Keyword] = []
        if self.stream.current.type == "lparen":
            args, kwargs, dyn_args, dyn_kwargs = self.parse_call_args()
        elif self.stream.current.type in {
            "name",
            "string",
            "integer",
            "float",
            "lparen",
            "lbracket",
            "lbrace",
        } and not self.stream.current.test_any("name:else", "name:or", "name:and"):
            if self.stream.current.test("name:is"):
                self.fail("You cannot chain multiple tests with is")
            arg_node = self.parse_primary()
            arg_node = self.parse_postfix(arg_node)
            args = [arg_node]
        else:
            args = []
        node = nodes.Test(
            node, name, args, kwargs, dyn_args, dyn_kwargs, lineno=token.lineno
        )
        if negated:
            node = nodes.Not(node, lineno=token.lineno)
        return node

    def subparse(
        self, end_tokens: t.Optional[t.Tuple[str, ...]] = None
    ) -> t.List[nodes.Node]:
        body: t.List[nodes.Node] = []
        data_buffer: t.List[nodes.Node] = []
        add_data = data_buffer.append

        if end_tokens is not None:
            self._end_token_stack.append(end_tokens)

        def flush_data() -> None:
            if data_buffer:
                lineno = data_buffer[0].lineno
                body.append(nodes.Output(data_buffer[:], lineno=lineno))
                del data_buffer[:]

        try:
            while self.stream:
                token = self.stream.current
                if token.type == "data":
                    if token.value:
                        add_data(nodes.TemplateData(token.value, lineno=token.lineno))
                    next(self.stream)
                elif token.type == "variable_begin":
                    next(self.stream)
                    add_data(self.parse_tuple(with_condexpr=True))
                    self.stream.expect("variable_end")
                elif token.type == "block_begin":
                    flush_data()
                    next(self.stream)
                    if end_tokens is not None and self.stream.current.test_any(
                        *end_tokens
                    ):
                        return body
                    rv = self.parse_statement()
                    if isinstance(rv, list):
                        body.extend(rv)
                    else:
                        body.append(rv)
                    self.stream.expect("block_end")
                else:
                    raise AssertionError("internal parsing error")

            flush_data()
        finally:
            if end_tokens is not None:
                self._end_token_stack.pop()
        return body

    def parse(self) -> nodes.Template:
        """Parse the whole template into a `Template` node."""
        result = nodes.Template(self.subparse(), lineno=1)
        result.set_environment(self.environment)
        return result