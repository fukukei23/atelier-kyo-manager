
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\groupby\groupby.py ===
"""
Provide the groupby split-apply-combine paradigm. Define the GroupBy
class providing the base-class of operations.

The SeriesGroupBy and DataFrameGroupBy sub-class
(defined in pandas.core.groupby.generic)
expose these user-facing objects to provide specific functionality.
"""
from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterator,
    Mapping,
    Sequence,
)
import datetime
from functools import (
    partial,
    wraps,
)
import inspect
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    TypeVar,
    Union,
    cast,
    final,
)
import warnings

import numpy as np

from pandas._config.config import option_context

from pandas._libs import (
    Timestamp,
    lib,
)
from pandas._libs.algos import rank_1d
import pandas._libs.groupby as libgroupby
from pandas._libs.missing import NA
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    Axis,
    AxisInt,
    DtypeObj,
    FillnaOptions,
    IndexLabel,
    NDFrameT,
    PositionalIndexer,
    RandomState,
    Scalar,
    T,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    AbstractMethodError,
    DataError,
)
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    coerce_indexer_dtype,
    ensure_dtype_can_hold_na,
)
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_float_dtype,
    is_hashable,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

from pandas.core import (
    algorithms,
    sample,
)
from pandas.core._numba import executor
from pandas.core.apply import warn_alias_replacement
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    Categorical,
    ExtensionArray,
    FloatingArray,
    IntegerArray,
    SparseArray,
)
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.string_arrow import (
    ArrowStringArray,
    ArrowStringArrayNumpySemantics,
)
from pandas.core.base import (
    PandasObject,
    SelectionMixin,
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import (
    base,
    numba_,
    ops,
)
from pandas.core.groupby.grouper import get_grouper
from pandas.core.groupby.indexing import (
    GroupByIndexingMixin,
    GroupByNthSelector,
)
from pandas.core.indexes.api import (
    CategoricalIndex,
    Index,
    MultiIndex,
    RangeIndex,
    default_index,
)
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import (
    get_jit_arguments,
    maybe_use_numba,
)

if TYPE_CHECKING:
    from typing import Any

    from pandas.core.resample import Resampler
    from pandas.core.window import (
        ExpandingGroupby,
        ExponentialMovingWindowGroupby,
        RollingGroupby,
    )

_common_see_also = """
        See Also
        --------
        Series.%(name)s : Apply a function %(name)s to a Series.
        DataFrame.%(name)s : Apply a function %(name)s
            to each row or column of a DataFrame.
"""

_apply_docs = {
    "template": """
    Apply function ``func`` group-wise and combine the results together.

    The function passed to ``apply`` must take a {input} as its first
    argument and return a DataFrame, Series or scalar. ``apply`` will
    then take care of combining the results back together into a single
    dataframe or series. ``apply`` is therefore a highly flexible
    grouping method.

    While ``apply`` is a very flexible method, its downside is that
    using it can be quite a bit slower than using more specific methods
    like ``agg`` or ``transform``. Pandas offers a wide range of method that will
    be much faster than using ``apply`` for their specific purposes, so try to
    use them before reaching for ``apply``.

    Parameters
    ----------
    func : callable
        A callable that takes a {input} as its first argument, and
        returns a dataframe, a series or a scalar. In addition the
        callable may take positional and keyword arguments.
    include_groups : bool, default True
        When True, will attempt to apply ``func`` to the groupings in
        the case that they are columns of the DataFrame. If this raises a
        TypeError, the result will be computed with the groupings excluded.
        When False, the groupings will be excluded when applying ``func``.

        .. versionadded:: 2.2.0

        .. deprecated:: 2.2.0

           Setting include_groups to True is deprecated. Only the value
           False will be allowed in a future version of pandas.

    args, kwargs : tuple and dict
        Optional positional and keyword arguments to pass to ``func``.

    Returns
    -------
    Series or DataFrame

    See Also
    --------
    pipe : Apply function to the full GroupBy object instead of to each
        group.
    aggregate : Apply aggregate function to the GroupBy object.
    transform : Apply function column-by-column to the GroupBy object.
    Series.apply : Apply a function to a Series.
    DataFrame.apply : Apply a function to each row or column of a DataFrame.

    Notes
    -----

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the passed ``func``,
        see the examples below.

    Functions that mutate the passed object can produce unexpected
    behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
    for more details.

    Examples
    --------
    {examples}
    """,
    "dataframe_examples": """
    >>> df = pd.DataFrame({'A': 'a a b'.split(),
    ...                    'B': [1, 2, 3],
    ...                    'C': [4, 6, 5]})
    >>> g1 = df.groupby('A', group_keys=False)
    >>> g2 = df.groupby('A', group_keys=True)

    Notice that ``g1`` and ``g2`` have two groups, ``a`` and ``b``, and only
    differ in their ``group_keys`` argument. Calling `apply` in various ways,
    we can get different grouping results:

    Example 1: below the function passed to `apply` takes a DataFrame as
    its argument and returns a DataFrame. `apply` combines the result for
    each group together into a new DataFrame:

    >>> g1[['B', 'C']].apply(lambda x: x / x.sum())
              B    C
    0  0.333333  0.4
    1  0.666667  0.6
    2  1.000000  1.0

    In the above, the groups are not part of the index. We can have them included
    by using ``g2`` where ``group_keys=True``:

    >>> g2[['B', 'C']].apply(lambda x: x / x.sum())
                B    C
    A
    a 0  0.333333  0.4
      1  0.666667  0.6
    b 2  1.000000  1.0

    Example 2: The function passed to `apply` takes a DataFrame as
    its argument and returns a Series.  `apply` combines the result for
    each group together into a new DataFrame.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the passed ``func``.

    >>> g1[['B', 'C']].apply(lambda x: x.astype(float).max() - x.min())
         B    C
    A
    a  1.0  2.0
    b  0.0  0.0

    >>> g2[['B', 'C']].apply(lambda x: x.astype(float).max() - x.min())
         B    C
    A
    a  1.0  2.0
    b  0.0  0.0

    The ``group_keys`` argument has no effect here because the result is not
    like-indexed (i.e. :ref:`a transform <groupby.transform>`) when compared
    to the input.

    Example 3: The function passed to `apply` takes a DataFrame as
    its argument and returns a scalar. `apply` combines the result for
    each group together into a Series, including setting the index as
    appropriate:

    >>> g1.apply(lambda x: x.C.max() - x.B.min(), include_groups=False)
    A
    a    5
    b    2
    dtype: int64""",
    "series_examples": """
    >>> s = pd.Series([0, 1, 2], index='a a b'.split())
    >>> g1 = s.groupby(s.index, group_keys=False)
    >>> g2 = s.groupby(s.index, group_keys=True)

    From ``s`` above we can see that ``g`` has two groups, ``a`` and ``b``.
    Notice that ``g1`` have ``g2`` have two groups, ``a`` and ``b``, and only
    differ in their ``group_keys`` argument. Calling `apply` in various ways,
    we can get different grouping results:

    Example 1: The function passed to `apply` takes a Series as
    its argument and returns a Series.  `apply` combines the result for
    each group together into a new Series.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the passed ``func``.

    >>> g1.apply(lambda x: x * 2 if x.name == 'a' else x / 2)
    a    0.0
    a    2.0
    b    1.0
    dtype: float64

    In the above, the groups are not part of the index. We can have them included
    by using ``g2`` where ``group_keys=True``:

    >>> g2.apply(lambda x: x * 2 if x.name == 'a' else x / 2)
    a  a    0.0
       a    2.0
    b  b    1.0
    dtype: float64

    Example 2: The function passed to `apply` takes a Series as
    its argument and returns a scalar. `apply` combines the result for
    each group together into a Series, including setting the index as
    appropriate:

    >>> g1.apply(lambda x: x.max() - x.min())
    a    1
    b    0
    dtype: int64

    The ``group_keys`` argument has no effect here because the result is not
    like-indexed (i.e. :ref:`a transform <groupby.transform>`) when compared
    to the input.

    >>> g2.apply(lambda x: x.max() - x.min())
    a    1
    b    0
    dtype: int64""",
}

_groupby_agg_method_template = """
Compute {fname} of group values.

Parameters
----------
numeric_only : bool, default {no}
    Include only float, int, boolean columns.

    .. versionchanged:: 2.0.0

        numeric_only no longer accepts ``None``.

min_count : int, default {mc}
    The required number of valid values to perform the operation. If fewer
    than ``min_count`` non-NA values are present the result will be NA.

Returns
-------
Series or DataFrame
    Computed {fname} of values within each group.

Examples
--------
{example}
"""

_groupby_agg_method_engine_template = """
Compute {fname} of group values.

Parameters
----------
numeric_only : bool, default {no}
    Include only float, int, boolean columns.

    .. versionchanged:: 2.0.0

        numeric_only no longer accepts ``None``.

min_count : int, default {mc}
    The required number of valid values to perform the operation. If fewer
    than ``min_count`` non-NA values are present the result will be NA.

engine : str, default None {e}
    * ``'cython'`` : Runs rolling apply through C-extensions from cython.
    * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
        Only available when ``raw`` is set to ``True``.
    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

engine_kwargs : dict, default None {ek}
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
        and ``parallel`` dictionary keys. The values must either be ``True`` or
        ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
        ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
        applied to both the ``func`` and the ``apply`` groupby aggregation.

Returns
-------
Series or DataFrame
    Computed {fname} of values within each group.

Examples
--------
{example}
"""

_pipe_template = """
Apply a ``func`` with arguments to this %(klass)s object and return its result.

Use `.pipe` when you want to improve readability by chaining together
functions that expect Series, DataFrames, GroupBy or Resampler objects.
Instead of writing

>>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3
>>> g = lambda x, arg1: x * 5 / arg1
>>> f = lambda x: x ** 4
>>> df = pd.DataFrame([["a", 4], ["b", 5]], columns=["group", "value"])
>>> h(g(f(df.groupby('group')), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP

You can write

>>> (df.groupby('group')
...    .pipe(f)
...    .pipe(g, arg1=1)
...    .pipe(h, arg2=2, arg3=3))  # doctest: +SKIP

which is much more readable.

Parameters
----------
func : callable or tuple of (callable, str)
    Function to apply to this %(klass)s object or, alternatively,
    a `(callable, data_keyword)` tuple where `data_keyword` is a
    string indicating the keyword of `callable` that expects the
    %(klass)s object.
args : iterable, optional
       Positional arguments passed into `func`.
kwargs : dict, optional
         A dictionary of keyword arguments passed into `func`.

Returns
-------
the return type of `func`.

See Also
--------
Series.pipe : Apply a function with arguments to a series.
DataFrame.pipe: Apply a function with arguments to a dataframe.
apply : Apply function to each group instead of to the
    full %(klass)s object.

Notes
-----
See more `here
<https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_

Examples
--------
%(examples)s
"""

_transform_template = """
Call function producing a same-indexed %(klass)s on each group.

Returns a %(klass)s having the same indexes as the original object
filled with the transformed values.

Parameters
----------
f : function, str
    Function to apply to each group. See the Notes section below for requirements.

    Accepted inputs are:

    - String
    - Python function
    - Numba JIT function with ``engine='numba'`` specified.

    Only passing a single function is supported with this engine.
    If the ``'numba'`` engine is chosen, the function must be
    a user defined function with ``values`` and ``index`` as the
    first and second arguments respectively in the function signature.
    Each group's index will be passed to the user defined function
    and optionally available for use.

    If a string is chosen, then it needs to be the name
    of the groupby method you want to use.
*args
    Positional arguments to pass to func.
engine : str, default None
    * ``'cython'`` : Runs the function through C-extensions from cython.
    * ``'numba'`` : Runs the function through JIT compiled code from numba.
    * ``None`` : Defaults to ``'cython'`` or the global setting ``compute.use_numba``

engine_kwargs : dict, default None
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
      and ``parallel`` dictionary keys. The values must either be ``True`` or
      ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
      ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
      applied to the function

**kwargs
    Keyword arguments to be passed into func.

Returns
-------
%(klass)s

See Also
--------
%(klass)s.groupby.apply : Apply function ``func`` group-wise and combine
    the results together.
%(klass)s.groupby.aggregate : Aggregate using one or more
    operations over the specified axis.
%(klass)s.transform : Call ``func`` on self producing a %(klass)s with the
    same axis shape as self.

Notes
-----
Each group is endowed the attribute 'name' in case you need to know
which group you are working on.

The current implementation imposes three requirements on f:

* f must return a value that either has the same shape as the input
  subframe or can be broadcast to the shape of the input subframe.
  For example, if `f` returns a scalar it will be broadcast to have the
  same shape as the input subframe.
* if this is a DataFrame, f must support application column-by-column
  in the subframe. If f also supports application to the entire subframe,
  then a fast path is used starting from the second chunk.
* f must not mutate groups. Mutation is not supported and may
  produce unexpected results. See :ref:`gotchas.udf-mutation` for more details.

When using ``engine='numba'``, there will be no "fall back" behavior internally.
The group data and group index will be passed as numpy arrays to the JITed
user defined function, and no alternative execution attempts will be tried.

.. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    see the examples below.

.. versionchanged:: 2.0.0

    When using ``.transform`` on a grouped DataFrame and the transformation function
    returns a DataFrame, pandas now aligns the result's index
    with the input's index. You can call ``.to_numpy()`` on the
    result of the transformation function to avoid alignment.

Examples
--------
%(example)s"""

_agg_template_series = """
Aggregate using one or more operations over the specified axis.

Parameters
----------
func : function, str, list, dict or None
    Function to use for aggregating the data. If a function, must either
    work when passed a {klass} or when passed to {klass}.apply.

    Accepted combinations are:

    - function
    - string function name
    - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
    - None, in which case ``**kwargs`` are used with Named Aggregation. Here the
      output has one column for each element in ``**kwargs``. The name of the
      column is keyword, whereas the value determines the aggregation used to compute
      the values in the column.

      Can also accept a Numba JIT function with
      ``engine='numba'`` specified. Only passing a single function is supported
      with this engine.

      If the ``'numba'`` engine is chosen, the function must be
      a user defined function with ``values`` and ``index`` as the
      first and second arguments respectively in the function signature.
      Each group's index will be passed to the user defined function
      and optionally available for use.

    .. deprecated:: 2.1.0

        Passing a dictionary is deprecated and will raise in a future version
        of pandas. Pass a list of aggregations instead.
*args
    Positional arguments to pass to func.
engine : str, default None
    * ``'cython'`` : Runs the function through C-extensions from cython.
    * ``'numba'`` : Runs the function through JIT compiled code from numba.
    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

engine_kwargs : dict, default None
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
      and ``parallel`` dictionary keys. The values must either be ``True`` or
      ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
      ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
      applied to the function

**kwargs
    * If ``func`` is None, ``**kwargs`` are used to define the output names and
      aggregations via Named Aggregation. See ``func`` entry.
    * Otherwise, keyword arguments to be passed into func.

Returns
-------
{klass}

See Also
--------
{klass}.groupby.apply : Apply function func group-wise
    and combine the results together.
{klass}.groupby.transform : Transforms the Series on each group
    based on the given function.
{klass}.aggregate : Aggregate using one or more
    operations over the specified axis.

Notes
-----
When using ``engine='numba'``, there will be no "fall back" behavior internally.
The group data and group index will be passed as numpy arrays to the JITed
user defined function, and no alternative execution attempts will be tried.

Functions that mutate the passed object can produce unexpected
behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
for more details.

.. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    see the examples below.
{examples}"""

_agg_template_frame = """
Aggregate using one or more operations over the specified axis.

Parameters
----------
func : function, str, list, dict or None
    Function to use for aggregating the data. If a function, must either
    work when passed a {klass} or when passed to {klass}.apply.

    Accepted combinations are:

    - function
    - string function name
    - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
    - dict of axis labels -> functions, function names or list of such.
    - None, in which case ``**kwargs`` are used with Named Aggregation. Here the
      output has one column for each element in ``**kwargs``. The name of the
      column is keyword, whereas the value determines the aggregation used to compute
      the values in the column.

      Can also accept a Numba JIT function with
      ``engine='numba'`` specified. Only passing a single function is supported
      with this engine.

      If the ``'numba'`` engine is chosen, the function must be
      a user defined function with ``values`` and ``index`` as the
      first and second arguments respectively in the function signature.
      Each group's index will be passed to the user defined function
      and optionally available for use.

*args
    Positional arguments to pass to func.
engine : str, default None
    * ``'cython'`` : Runs the function through C-extensions from cython.
    * ``'numba'`` : Runs the function through JIT compiled code from numba.
    * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``

engine_kwargs : dict, default None
    * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
    * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
      and ``parallel`` dictionary keys. The values must either be ``True`` or
      ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
      ``{{'nopython': True, 'nogil': False, 'parallel': False}}`` and will be
      applied to the function

**kwargs
    * If ``func`` is None, ``**kwargs`` are used to define the output names and
      aggregations via Named Aggregation. See ``func`` entry.
    * Otherwise, keyword arguments to be passed into func.

Returns
-------
{klass}

See Also
--------
{klass}.groupby.apply : Apply function func group-wise
    and combine the results together.
{klass}.groupby.transform : Transforms the Series on each group
    based on the given function.
{klass}.aggregate : Aggregate using one or more
    operations over the specified axis.

Notes
-----
When using ``engine='numba'``, there will be no "fall back" behavior internally.
The group data and group index will be passed as numpy arrays to the JITed
user defined function, and no alternative execution attempts will be tried.

Functions that mutate the passed object can produce unexpected
behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
for more details.

.. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    see the examples below.
{examples}"""


@final
class GroupByPlot(PandasObject):
    """
    Class implementing the .plot attribute for groupby objects.
    """

    def __init__(self, groupby: GroupBy) -> None:
        self._groupby = groupby

    def __call__(self, *args, **kwargs):
        def f(self):
            return self.plot(*args, **kwargs)

        f.__name__ = "plot"
        return self._groupby._python_apply_general(f, self._groupby._selected_obj)

    def __getattr__(self, name: str):
        def attr(*args, **kwargs):
            def f(self):
                return getattr(self.plot, name)(*args, **kwargs)

            return self._groupby._python_apply_general(f, self._groupby._selected_obj)

        return attr


_KeysArgType = Union[
    Hashable,
    list[Hashable],
    Callable[[Hashable], Hashable],
    list[Callable[[Hashable], Hashable]],
    Mapping[Hashable, Hashable],
]


class BaseGroupBy(PandasObject, SelectionMixin[NDFrameT], GroupByIndexingMixin):
    _hidden_attrs = PandasObject._hidden_attrs | {
        "as_index",
        "axis",
        "dropna",
        "exclusions",
        "grouper",
        "group_keys",
        "keys",
        "level",
        "obj",
        "observed",
        "sort",
    }

    axis: AxisInt
    _grouper: ops.BaseGrouper
    keys: _KeysArgType | None = None
    level: IndexLabel | None = None
    group_keys: bool

    @final
    def __len__(self) -> int:
        return len(self.groups)

    @final
    def __repr__(self) -> str:
        # TODO: Better repr for GroupBy object
        return object.__repr__(self)

    @final
    @property
    def grouper(self) -> ops.BaseGrouper:
        warnings.warn(
            f"{type(self).__name__}.grouper is deprecated and will be removed in a "
            "future version of pandas.",
            category=FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._grouper

    @final
    @property
    def groups(self) -> dict[Hashable, np.ndarray]:
        """
        Dict {group name -> group labels}.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).groups
        {'a': ['a', 'a'], 'b': ['b']}

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> df.groupby(by=["a"]).groups
        {1: [0, 1], 7: [2]}

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').groups
        {Timestamp('2023-01-01 00:00:00'): 2, Timestamp('2023-02-01 00:00:00'): 4}
        """
        return self._grouper.groups

    @final
    @property
    def ngroups(self) -> int:
        return self._grouper.ngroups

    @final
    @property
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """
        Dict {group name -> group indices}.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).indices
        {'a': array([0, 1]), 'b': array([2])}

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).indices
        {1: array([0, 1]), 7: array([2])}

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').indices
        defaultdict(<class 'list'>, {Timestamp('2023-01-01 00:00:00'): [0, 1],
        Timestamp('2023-02-01 00:00:00'): [2, 3]})
        """
        return self._grouper.indices

    @final
    def _get_indices(self, names):
        """
        Safe get multiple indices, translate keys for
        datelike to underlying repr.
        """

        def get_converter(s):
            # possibly convert to the actual key types
            # in the indices, could be a Timestamp or a np.datetime64
            if isinstance(s, datetime.datetime):
                return lambda key: Timestamp(key)
            elif isinstance(s, np.datetime64):
                return lambda key: Timestamp(key).asm8
            else:
                return lambda key: key

        if len(names) == 0:
            return []

        if len(self.indices) > 0:
            index_sample = next(iter(self.indices))
        else:
            index_sample = None  # Dummy sample

        name_sample = names[0]
        if isinstance(index_sample, tuple):
            if not isinstance(name_sample, tuple):
                msg = "must supply a tuple to get_group with multiple grouping keys"
                raise ValueError(msg)
            if not len(name_sample) == len(index_sample):
                try:
                    # If the original grouper was a tuple
                    return [self.indices[name] for name in names]
                except KeyError as err:
                    # turns out it wasn't a tuple
                    msg = (
                        "must supply a same-length tuple to get_group "
                        "with multiple grouping keys"
                    )
                    raise ValueError(msg) from err

            converters = [get_converter(s) for s in index_sample]
            names = (tuple(f(n) for f, n in zip(converters, name)) for name in names)

        else:
            converter = get_converter(index_sample)
            names = (converter(name) for name in names)

        return [self.indices.get(name, []) for name in names]

    @final
    def _get_index(self, name):
        """
        Safe get index, translate keys for datelike to underlying repr.
        """
        return self._get_indices([name])[0]

    @final
    @cache_readonly
    def _selected_obj(self):
        # Note: _selected_obj is always just `self.obj` for SeriesGroupBy
        if isinstance(self.obj, Series):
            return self.obj

        if self._selection is not None:
            if is_hashable(self._selection):
                # i.e. a single key, so selecting it will return a Series.
                #  In this case, _obj_with_exclusions would wrap the key
                #  in a list and return a single-column DataFrame.
                return self.obj[self._selection]

            # Otherwise _selection is equivalent to _selection_list, so
            #  _selected_obj matches _obj_with_exclusions, so we can reuse
            #  that and avoid making a copy.
            return self._obj_with_exclusions

        return self.obj

    @final
    def _dir_additions(self) -> set[str]:
        return self.obj._dir_additions()

    @Substitution(
        klass="GroupBy",
        examples=dedent(
            """\
        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4

        To get the difference between each groups maximum and minimum value in one
        pass, you can do

        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2"""
        ),
    )
    @Appender(_pipe_template)
    def pipe(
        self,
        func: Callable[..., T] | tuple[Callable[..., T], str],
        *args,
        **kwargs,
    ) -> T:
        return com.pipe(self, func, *args, **kwargs)

    @final
    def get_group(self, name, obj=None) -> DataFrame | Series:
        """
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.
        obj : DataFrame, default None
            The DataFrame to take the DataFrame out of.  If
            it is None, the object groupby was called on will
            be used.

            .. deprecated:: 2.1.0
                The obj is deprecated and will be removed in a future version.
                Do ``df.iloc[gb.indices.get(name)]``
                instead of ``gb.get_group(name, obj=df)``.

        Returns
        -------
        same type as obj

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).get_group("a")
        a    1
        a    2
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).get_group((1,))
                a  b  c
        owl     1  2  3
        toucan  1  5  6

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').get_group('2023-01-01')
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        """
        keys = self.keys
        level = self.level
        # mypy doesn't recognize level/keys as being sized when passed to len
        if (is_list_like(level) and len(level) == 1) or (  # type: ignore[arg-type]
            is_list_like(keys) and len(keys) == 1  # type: ignore[arg-type]
        ):
            # GH#25971
            if isinstance(name, tuple) and len(name) == 1:
                # Allow users to pass tuples of length 1 to silence warning
                name = name[0]
            elif not isinstance(name, tuple):
                warnings.warn(
                    "When grouping with a length-1 list-like, "
                    "you will need to pass a length-1 tuple to get_group in a future "
                    "version of pandas. Pass `(name,)` instead of `name` to silence "
                    "this warning.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

        inds = self._get_index(name)
        if not len(inds):
            raise KeyError(name)

        if obj is None:
            indexer = inds if self.axis == 0 else (slice(None), inds)
            return self._selected_obj.iloc[indexer]
        else:
            warnings.warn(
                "obj is deprecated and will be removed in a future version. "
                "Do ``df.iloc[gb.indices.get(name)]`` "
                "instead of ``gb.get_group(name, obj=df)``.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            return obj._take_with_is_copy(inds, axis=self.axis)

    @final
    def __iter__(self) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Groupby iterator.

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> for x, y in ser.groupby(level=0):
        ...     print(f'{x}\\n{y}\\n')
        a
        a    1
        a    2
        dtype: int64
        b
        b    3
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"])
        >>> df
           a  b  c
        0  1  2  3
        1  1  5  6
        2  7  8  9
        >>> for x, y in df.groupby(by=["a"]):
        ...     print(f'{x}\\n{y}\\n')
        (1,)
           a  b  c
        0  1  2  3
        1  1  5  6
        (7,)
           a  b  c
        2  7  8  9

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> for x, y in ser.resample('MS'):
        ...     print(f'{x}\\n{y}\\n')
        2023-01-01 00:00:00
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        2023-02-01 00:00:00
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        """
        keys = self.keys
        level = self.level
        result = self._grouper.get_iterator(self._selected_obj, axis=self.axis)
        # error: Argument 1 to "len" has incompatible type "Hashable"; expected "Sized"
        if is_list_like(level) and len(level) == 1:  # type: ignore[arg-type]
            # GH 51583
            warnings.warn(
                "Creating a Groupby object with a length-1 list-like "
                "level parameter will yield indexes as tuples in a future version. "
                "To keep indexes as scalars, create Groupby objects with "
                "a scalar level parameter instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if isinstance(keys, list) and len(keys) == 1:
            # GH#42795 - when keys is a list, return tuples even when length is 1
            result = (((key,), group) for key, group in result)
        return result


# To track operations that expand dimensions, like ohlc
OutputFrameOrSeries = TypeVar("OutputFrameOrSeries", bound=NDFrame)


class GroupBy(BaseGroupBy[NDFrameT]):
    """
    Class for grouping and aggregating relational data.

    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.groupby(...) to use GroupBy, but you can also do:

    ::

        grouped = groupby(obj, ...)

    Parameters
    ----------
    obj : pandas object
    axis : int, default 0
    level : int, default None
        Level of MultiIndex
    groupings : list of Grouping objects
        Most users should ignore this
    exclusions : array-like, optional
        List of columns to exclude
    name : str
        Most users should ignore this

    Returns
    -------
    **Attributes**
    groups : dict
        {group name -> group labels}
    len(grouped) : int
        Number of groups

    Notes
    -----
    After grouping, see aggregate, apply, and transform functions. Here are
    some other brief notes about usage. When grouping by multiple groups, the
    result index will be a MultiIndex (hierarchical) by default.

    Iteration produces (key, group) tuples, i.e. chunking the data by group. So
    you can write code like:

    ::

        grouped = obj.groupby(keys, axis=axis)
        for key, group in grouped:
            # do something with the data

    Function calls on GroupBy, if not specially implemented, "dispatch" to the
    grouped data. So if you group a DataFrame and wish to invoke the std()
    method on each group, you can simply do:

    ::

        df.groupby(mapper).std()

    rather than

    ::

        df.groupby(mapper).aggregate(np.std)

    You can pass arguments to these "wrapped" functions, too.

    See the online documentation for full exposition on these topics and much
    more
    """

    _grouper: ops.BaseGrouper
    as_index: bool

    @final
    def __init__(
        self,
        obj: NDFrameT,
        keys: _KeysArgType | None = None,
        axis: Axis = 0,
        level: IndexLabel | None = None,
        grouper: ops.BaseGrouper | None = None,
        exclusions: frozenset[Hashable] | None = None,
        selection: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool | lib.NoDefault = lib.no_default,
        dropna: bool = True,
    ) -> None:
        self._selection = selection

        assert isinstance(obj, NDFrame), type(obj)

        self.level = level

        if not as_index:
            if axis != 0:
                raise ValueError("as_index=False only valid for axis=0")

        self.as_index = as_index
        self.keys = keys
        self.sort = sort
        self.group_keys = group_keys
        self.dropna = dropna

        if grouper is None:
            grouper, exclusions, obj = get_grouper(
                obj,
                keys,
                axis=axis,
                level=level,
                sort=sort,
                observed=False if observed is lib.no_default else observed,
                dropna=self.dropna,
            )

        if observed is lib.no_default:
            if any(ping._passed_categorical for ping in grouper.groupings):
                warnings.warn(
                    "The default of observed=False is deprecated and will be changed "
                    "to True in a future version of pandas. Pass observed=False to "
                    "retain current behavior or observed=True to adopt the future "
                    "default and silence this warning.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            observed = False
        self.observed = observed

        self.obj = obj
        self.axis = obj._get_axis_number(axis)
        self._grouper = grouper
        self.exclusions = frozenset(exclusions) if exclusions else frozenset()

    def __getattr__(self, attr: str):
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    @final
    def _deprecate_axis(self, axis: int, name: str) -> None:
        if axis == 1:
            warnings.warn(
                f"{type(self).__name__}.{name} with axis=1 is deprecated and "
                "will be removed in a future version. Operate on the un-grouped "
                "DataFrame instead",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        else:
            warnings.warn(
                f"The 'axis' keyword in {type(self).__name__}.{name} is deprecated "
                "and will be removed in a future version. "
                "Call without passing 'axis' instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

    @final
    def _op_via_apply(self, name: str, *args, **kwargs):
        """Compute the result of an operation by using GroupBy's apply."""
        f = getattr(type(self._obj_with_exclusions), name)
        sig = inspect.signature(f)

        if "axis" in kwargs and kwargs["axis"] is not lib.no_default:
            axis = self.obj._get_axis_number(kwargs["axis"])
            self._deprecate_axis(axis, name)
        elif "axis" in kwargs:
            # exclude skew here because that was already defaulting to lib.no_default
            #  before this deprecation was instituted
            if name == "skew":
                pass
            elif name == "fillna":
                # maintain the behavior from before the deprecation
                kwargs["axis"] = None
            else:
                kwargs["axis"] = 0

        # a little trickery for aggregation functions that need an axis
        # argument
        if "axis" in sig.parameters:
            if kwargs.get("axis", None) is None or kwargs.get("axis") is lib.no_default:
                kwargs["axis"] = self.axis

        def curried(x):
            return f(x, *args, **kwargs)

        # preserve the name so we can detect it when calling plot methods,
        # to avoid duplicates
        curried.__name__ = name

        # special case otherwise extra plots are created when catching the
        # exception below
        if name in base.plotting_methods:
            return self._python_apply_general(curried, self._selected_obj)

        is_transform = name in base.transformation_kernels
        result = self._python_apply_general(
            curried,
            self._obj_with_exclusions,
            is_transform=is_transform,
            not_indexed_same=not is_transform,
        )

        if self._grouper.has_dropped_na and is_transform:
            # result will have dropped rows due to nans, fill with null
            # and ensure index is ordered same as the input
            result = self._set_result_index_ordered(result)
        return result

    # -----------------------------------------------------------------
    # Dispatch/Wrapping

    @final
    def _concat_objects(
        self,
        values,
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ):
        from pandas.core.reshape.concat import concat

        if self.group_keys and not is_transform:
            if self.as_index:
                # possible MI return case
                group_keys = self._grouper.result_index
                group_levels = self._grouper.levels
                group_names = self._grouper.names

                result = concat(
                    values,
                    axis=self.axis,
                    keys=group_keys,
                    levels=group_levels,
                    names=group_names,
                    sort=False,
                )
            else:
                # GH5610, returns a MI, with the first level being a
                # range index
                keys = list(range(len(values)))
                result = concat(values, axis=self.axis, keys=keys)

        elif not not_indexed_same:
            result = concat(values, axis=self.axis)

            ax = self._selected_obj._get_axis(self.axis)
            if self.dropna:
                labels = self._grouper.group_info[0]
                mask = labels != -1
                ax = ax[mask]

            # this is a very unfortunate situation
            # we can't use reindex to restore the original order
            # when the ax has duplicates
            # so we resort to this
            # GH 14776, 30667
            # TODO: can we reuse e.g. _reindex_non_unique?
            if ax.has_duplicates and not result.axes[self.axis].equals(ax):
                # e.g. test_category_order_transformer
                target = algorithms.unique1d(ax._values)
                indexer, _ = result.index.get_indexer_non_unique(target)
                result = result.take(indexer, axis=self.axis)
            else:
                result = result.reindex(ax, axis=self.axis, copy=False)

        else:
            result = concat(values, axis=self.axis)

        if self.obj.ndim == 1:
            name = self.obj.name
        elif is_hashable(self._selection):
            name = self._selection
        else:
            name = None

        if isinstance(result, Series) and name is not None:
            result.name = name

        return result

    @final
    def _set_result_index_ordered(
        self, result: OutputFrameOrSeries
    ) -> OutputFrameOrSeries:
        # set the result index on the passed values object and
        # return the new object, xref 8046

        obj_axis = self.obj._get_axis(self.axis)

        if self._grouper.is_monotonic and not self._grouper.has_dropped_na:
            # shortcut if we have an already ordered grouper
            result = result.set_axis(obj_axis, axis=self.axis, copy=False)
            return result

        # row order is scrambled => sort the rows by position in original index
        original_positions = Index(self._grouper.result_ilocs())
        result = result.set_axis(original_positions, axis=self.axis, copy=False)
        result = result.sort_index(axis=self.axis)
        if self._grouper.has_dropped_na:
            # Add back in any missing rows due to dropna - index here is integral
            # with values referring to the row of the input so can use RangeIndex
            result = result.reindex(RangeIndex(len(obj_axis)), axis=self.axis)
        result = result.set_axis(obj_axis, axis=self.axis, copy=False)

        return result

    @final
    def _insert_inaxis_grouper(self, result: Series | DataFrame) -> DataFrame:
        if isinstance(result, Series):
            result = result.to_frame()

        # zip in reverse so we can always insert at loc 0
        columns = result.columns
        for name, lev, in_axis in zip(
            reversed(self._grouper.names),
            reversed(self._grouper.get_group_levels()),
            reversed([grp.in_axis for grp in self._grouper.groupings]),
        ):
            # GH #28549
            # When using .apply(-), name will be in columns already
            if name not in columns:
                if in_axis:
                    result.insert(0, name, lev)
                else:
                    msg = (
                        "A grouping was used that is not in the columns of the "
                        "DataFrame and so was excluded from the result. This grouping "
                        "will be included in a future version of pandas. Add the "
                        "grouping as a column of the DataFrame to silence this warning."
                    )
                    warnings.warn(
                        message=msg,
                        category=FutureWarning,
                        stacklevel=find_stack_level(),
                    )

        return result

    @final
    def _maybe_transpose_result(self, result: NDFrameT) -> NDFrameT:
        if self.axis == 1:
            # Only relevant for DataFrameGroupBy, no-op for SeriesGroupBy
            result = result.T
            if result.index.equals(self.obj.index):
                # Retain e.g. DatetimeIndex/TimedeltaIndex freq
                # e.g. test_groupby_crash_on_nunique
                result.index = self.obj.index.copy()
        return result

    @final
    def _wrap_aggregated_output(
        self,
        result: Series | DataFrame,
        qs: npt.NDArray[np.float64] | None = None,
    ):
        """
        Wraps the output of GroupBy aggregations into the expected result.

        Parameters
        ----------
        result : Series, DataFrame

        Returns
        -------
        Series or DataFrame
        """
        # ATM we do not get here for SeriesGroupBy; when we do, we will
        #  need to require that result.name already match self.obj.name

        if not self.as_index:
            # `not self.as_index` is only relevant for DataFrameGroupBy,
            #   enforced in __init__
            result = self._insert_inaxis_grouper(result)
            result = result._consolidate()
            index = Index(range(self._grouper.ngroups))

        else:
            index = self._grouper.result_index

        if qs is not None:
            # We get here with len(qs) != 1 and not self.as_index
            #  in test_pass_args_kwargs
            index = _insert_quantile_level(index, qs)

        result.index = index

        # error: Argument 1 to "_maybe_transpose_result" of "GroupBy" has
        # incompatible type "Union[Series, DataFrame]"; expected "NDFrameT"
        res = self._maybe_transpose_result(result)  # type: ignore[arg-type]
        return self._reindex_output(res, qs=qs)

    def _wrap_applied_output(
        self,
        data,
        values: list,
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ):
        raise AbstractMethodError(self)

    # -----------------------------------------------------------------
    # numba

    @final
    def _numba_prep(self, data: DataFrame):
        ids, _, ngroups = self._grouper.group_info
        sorted_index = self._grouper._sort_idx
        sorted_ids = self._grouper._sorted_ids

        sorted_data = data.take(sorted_index, axis=self.axis).to_numpy()
        # GH 46867
        index_data = data.index
        if isinstance(index_data, MultiIndex):
            if len(self._grouper.groupings) > 1:
                raise NotImplementedError(
                    "Grouping with more than 1 grouping labels and "
                    "a MultiIndex is not supported with engine='numba'"
                )
            group_key = self._grouper.groupings[0].name
            index_data = index_data.get_level_values(group_key)
        sorted_index_data = index_data.take(sorted_index).to_numpy()

        starts, ends = lib.generate_slices(sorted_ids, ngroups)
        return (
            starts,
            ends,
            sorted_index_data,
            sorted_data,
        )

    def _numba_agg_general(
        self,
        func: Callable,
        dtype_mapping: dict[np.dtype, Any],
        engine_kwargs: dict[str, bool] | None,
        **aggregator_kwargs,
    ):
        """
        Perform groupby with a standard numerical aggregation function (e.g. mean)
        with Numba.
        """
        if not self.as_index:
            raise NotImplementedError(
                "as_index=False is not supported. Use .reset_index() instead."
            )
        if self.axis == 1:
            raise NotImplementedError("axis=1 is not supported.")

        data = self._obj_with_exclusions
        df = data if data.ndim == 2 else data.to_frame()

        aggregator = executor.generate_shared_aggregator(
            func,
            dtype_mapping,
            True,  # is_grouped_kernel
            **get_jit_arguments(engine_kwargs),
        )
        # Pass group ids to kernel directly if it can handle it
        # (This is faster since it doesn't require a sort)
        ids, _, _ = self._grouper.group_info
        ngroups = self._grouper.ngroups

        res_mgr = df._mgr.apply(
            aggregator, labels=ids, ngroups=ngroups, **aggregator_kwargs
        )
        res_mgr.axes[1] = self._grouper.result_index
        result = df._constructor_from_mgr(res_mgr, axes=res_mgr.axes)

        if data.ndim == 1:
            result = result.squeeze("columns")
            result.name = data.name
        else:
            result.columns = data.columns
        return result

    @final
    def _transform_with_numba(self, func, *args, engine_kwargs=None, **kwargs):
        """
        Perform groupby transform routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
        data = self._obj_with_exclusions
        df = data if data.ndim == 2 else data.to_frame()

        starts, ends, sorted_index, sorted_data = self._numba_prep(df)
        numba_.validate_udf(func)
        numba_transform_func = numba_.generate_numba_transform_func(
            func, **get_jit_arguments(engine_kwargs, kwargs)
        )
        result = numba_transform_func(
            sorted_data,
            sorted_index,
            starts,
            ends,
            len(df.columns),
            *args,
        )
        # result values needs to be resorted to their original positions since we
        # evaluated the data sorted by group
        result = result.take(np.argsort(sorted_index), axis=0)
        index = data.index
        if data.ndim == 1:
            result_kwargs = {"name": data.name}
            result = result.ravel()
        else:
            result_kwargs = {"columns": data.columns}
        return data._constructor(result, index=index, **result_kwargs)

    @final
    def _aggregate_with_numba(self, func, *args, engine_kwargs=None, **kwargs):
        """
        Perform groupby aggregation routine with the numba engine.

        This routine mimics the data splitting routine of the DataSplitter class
        to generate the indices of each group in the sorted data and then passes the
        data and indices into a Numba jitted function.
        """
        data = self._obj_with_exclusions
        df = data if data.ndim == 2 else data.to_frame()

        starts, ends, sorted_index, sorted_data = self._numba_prep(df)
        numba_.validate_udf(func)
        numba_agg_func = numba_.generate_numba_agg_func(
            func, **get_jit_arguments(engine_kwargs, kwargs)
        )
        result = numba_agg_func(
            sorted_data,
            sorted_index,
            starts,
            ends,
            len(df.columns),
            *args,
        )
        index = self._grouper.result_index
        if data.ndim == 1:
            result_kwargs = {"name": data.name}
            result = result.ravel()
        else:
            result_kwargs = {"columns": data.columns}
        res = data._constructor(result, index=index, **result_kwargs)
        if not self.as_index:
            res = self._insert_inaxis_grouper(res)
            res.index = default_index(len(res))
        return res

    # -----------------------------------------------------------------
    # apply/agg/transform

    @Appender(
        _apply_docs["template"].format(
            input="dataframe", examples=_apply_docs["dataframe_examples"]
        )
    )
    def apply(self, func, *args, include_groups: bool = True, **kwargs) -> NDFrameT:
        orig_func = func
        func = com.is_builtin_func(func)
        if orig_func != func:
            alias = com._builtin_table_alias[orig_func]
            warn_alias_replacement(self, orig_func, alias)

        if isinstance(func, str):
            if hasattr(self, func):
                res = getattr(self, func)
                if callable(res):
                    return res(*args, **kwargs)
                elif args or kwargs:
                    raise ValueError(f"Cannot pass arguments to property {func}")
                return res

            else:
                raise TypeError(f"apply func should be callable, not '{func}'")

        elif args or kwargs:
            if callable(func):

                @wraps(func)
                def f(g):
                    return func(g, *args, **kwargs)

            else:
                raise ValueError(
                    "func must be a callable if args or kwargs are supplied"
                )
        else:
            f = func

        if not include_groups:
            return self._python_apply_general(f, self._obj_with_exclusions)

        # ignore SettingWithCopy here in case the user mutates
        with option_context("mode.chained_assignment", None):
            try:
                result = self._python_apply_general(f, self._selected_obj)
                if (
                    not isinstance(self.obj, Series)
                    and self._selection is None
                    and self._selected_obj.shape != self._obj_with_exclusions.shape
                ):
                    warnings.warn(
                        message=_apply_groupings_depr.format(
                            type(self).__name__, "apply"
                        ),
                        category=FutureWarning,
                        stacklevel=find_stack_level(),
                    )
            except TypeError:
                # gh-20949
                # try again, with .apply acting as a filtering
                # operation, by excluding the grouping column
                # This would normally not be triggered
                # except if the udf is trying an operation that
                # fails on *some* columns, e.g. a numeric operation
                # on a string grouper column

                return self._python_apply_general(f, self._obj_with_exclusions)

        return result

    @final
    def _python_apply_general(
        self,
        f: Callable,
        data: DataFrame | Series,
        not_indexed_same: bool | None = None,
        is_transform: bool = False,
        is_agg: bool = False,
    ) -> NDFrameT:
        """
        Apply function f in python space

        Parameters
        ----------
        f : callable
            Function to apply
        data : Series or DataFrame
            Data to apply f to
        not_indexed_same: bool, optional
            When specified, overrides the value of not_indexed_same. Apply behaves
            differently when the result index is equal to the input index, but
            this can be coincidental leading to value-dependent behavior.
        is_transform : bool, default False
            Indicator for whether the function is actually a transform
            and should not have group keys prepended.
        is_agg : bool, default False
            Indicator for whether the function is an aggregation. When the
            result is empty, we don't want to warn for this case.
            See _GroupBy._python_agg_general.

        Returns
        -------
        Series or DataFrame
            data after applying f
        """
        values, mutated = self._grouper.apply_groupwise(f, data, self.axis)
        if not_indexed_same is None:
            not_indexed_same = mutated

        return self._wrap_applied_output(
            data,
            values,
            not_indexed_same,
            is_transform,
        )

    @final
    def _agg_general(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        *,
        alias: str,
        npfunc: Callable | None = None,
        **kwargs,
    ):
        result = self._cython_agg_general(
            how=alias,
            alt=npfunc,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )
        return result.__finalize__(self.obj, method="groupby")

    def _agg_py_fallback(
        self, how: str, values: ArrayLike, ndim: int, alt: Callable
    ) -> ArrayLike:
        """
        Fallback to pure-python aggregation if _cython_operation raises
        NotImplementedError.
        """
        # We get here with a) EADtypes and b) object dtype
        assert alt is not None

        if values.ndim == 1:
            # For DataFrameGroupBy we only get here with ExtensionArray
            ser = Series(values, copy=False)
        else:
            # We only get here with values.dtype == object
            df = DataFrame(values.T, dtype=values.dtype)
            # bc we split object blocks in grouped_reduce, we have only 1 col
            # otherwise we'd have to worry about block-splitting GH#39329
            assert df.shape[1] == 1
            # Avoid call to self.values that can occur in DataFrame
            #  reductions; see GH#28949
            ser = df.iloc[:, 0]

        # We do not get here with UDFs, so we know that our dtype
        #  should always be preserved by the implemented aggregations
        # TODO: Is this exactly right; see WrappedCythonOp get_result_dtype?
        try:
            res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
        except Exception as err:
            msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
            # preserve the kind of exception that raised
            raise type(err)(msg) from err

        if ser.dtype == object:
            res_values = res_values.astype(object, copy=False)

        # If we are DataFrameGroupBy and went through a SeriesGroupByPath
        # then we need to reshape
        # GH#32223 includes case with IntegerArray values, ndarray res_values
        # test_groupby_duplicate_columns with object dtype values
        return ensure_block_shape(res_values, ndim=ndim)

    @final
    def _cython_agg_general(
        self,
        how: str,
        alt: Callable | None = None,
        numeric_only: bool = False,
        min_count: int = -1,
        **kwargs,
    ):
        # Note: we never get here with how="ohlc" for DataFrameGroupBy;
        #  that goes through SeriesGroupBy

        data = self._get_data_to_aggregate(numeric_only=numeric_only, name=how)

        def array_func(values: ArrayLike) -> ArrayLike:
            try:
                result = self._grouper._cython_operation(
                    "aggregate",
                    values,
                    how,
                    axis=data.ndim - 1,
                    min_count=min_count,
                    **kwargs,
                )
            except NotImplementedError:
                # generally if we have numeric_only=False
                # and non-applicable functions
                # try to python agg
                # TODO: shouldn't min_count matter?
                # TODO: avoid special casing SparseArray here
                if how in ["any", "all"] and isinstance(values, SparseArray):
                    pass
                elif alt is None or how in ["any", "all", "std", "sem"]:
                    raise  # TODO: re-raise as TypeError?  should not be reached
            else:
                return result

            assert alt is not None
            result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
            return result

        new_mgr = data.grouped_reduce(array_func)
        res = self._wrap_agged_manager(new_mgr)
        if how in ["idxmin", "idxmax"]:
            res = self._wrap_idxmax_idxmin(res)
        out = self._wrap_aggregated_output(res)
        if self.axis == 1:
            out = out.infer_objects(copy=False)
        return out

    def _cython_transform(
        self, how: str, numeric_only: bool = False, axis: AxisInt = 0, **kwargs
    ):
        raise AbstractMethodError(self)

    @final
    def _transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        # optimized transforms
        orig_func = func
        func = com.get_cython_func(func) or func
        if orig_func != func:
            warn_alias_replacement(self, orig_func, func)

        if not isinstance(func, str):
            return self._transform_general(func, engine, engine_kwargs, *args, **kwargs)

        elif func not in base.transform_kernel_allowlist:
            msg = f"'{func}' is not a valid function name for transform(name)"
            raise ValueError(msg)
        elif func in base.cythonized_kernels or func in base.transformation_kernels:
            # cythonized transform or canned "agg+broadcast"
            if engine is not None:
                kwargs["engine"] = engine
                kwargs["engine_kwargs"] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)

        else:
            # i.e. func in base.reduction_kernels

            # GH#30918 Use _transform_fast only when we know func is an aggregation
            # If func is a reduction, we need to broadcast the
            # result to the whole group. Compute func result
            # and deal with possible broadcasting below.
            with com.temp_setattr(self, "as_index", True):
                # GH#49834 - result needs groups in the index for
                # _wrap_transform_fast_result
                if func in ["idxmin", "idxmax"]:
                    func = cast(Literal["idxmin", "idxmax"], func)
                    result = self._idxmax_idxmin(func, True, *args, **kwargs)
                else:
                    if engine is not None:
                        kwargs["engine"] = engine
                        kwargs["engine_kwargs"] = engine_kwargs
                    result = getattr(self, func)(*args, **kwargs)

            return self._wrap_transform_fast_result(result)

    @final
    def _wrap_transform_fast_result(self, result: NDFrameT) -> NDFrameT:
        """
        Fast transform path for aggregations.
        """
        obj = self._obj_with_exclusions

        # for each col, reshape to size of original frame by take operation
        ids, _, _ = self._grouper.group_info
        result = result.reindex(self._grouper.result_index, axis=self.axis, copy=False)

        if self.obj.ndim == 1:
            # i.e. SeriesGroupBy
            out = algorithms.take_nd(result._values, ids)
            output = obj._constructor(out, index=obj.index, name=obj.name)
        else:
            # `.size()` gives Series output on DataFrame input, need axis 0
            axis = 0 if result.ndim == 1 else self.axis
            # GH#46209
            # Don't convert indices: negative indices need to give rise
            # to null values in the result
            new_ax = result.axes[axis].take(ids)
            output = result._reindex_with_indexers(
                {axis: (new_ax, ids)}, allow_dups=True, copy=False
            )
            output = output.set_axis(obj._get_axis(self.axis), axis=axis)
        return output

    # -----------------------------------------------------------------
    # Utilities

    @final
    def _apply_filter(self, indices, dropna):
        if len(indices) == 0:
            indices = np.array([], dtype="int64")
        else:
            indices = np.sort(np.concatenate(indices))
        if dropna:
            filtered = self._selected_obj.take(indices, axis=self.axis)
        else:
            mask = np.empty(len(self._selected_obj.index), dtype=bool)
            mask.fill(False)
            mask[indices.astype(int)] = True
            # mask fails to broadcast when passed to where; broadcast manually.
            mask = np.tile(mask, list(self._selected_obj.shape[1:]) + [1]).T
            filtered = self._selected_obj.where(mask)  # Fill with NaNs.
        return filtered

    @final
    def _cumcount_array(self, ascending: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Notes
        -----
        this is currently implementing sort=False
        (though the default is sort=True) for groupby in general
        """
        ids, _, ngroups = self._grouper.group_info
        sorter = get_group_index_sorter(ids, ngroups)
        ids, count = ids[sorter], len(ids)

        if count == 0:
            return np.empty(0, dtype=np.int64)

        run = np.r_[True, ids[:-1] != ids[1:]]
        rep = np.diff(np.r_[np.nonzero(run)[0], count])
        out = (~run).cumsum()

        if ascending:
            out -= np.repeat(out[run], rep)
        else:
            out = np.repeat(out[np.r_[run[1:], True]], rep) - out

        if self._grouper.has_dropped_na:
            out = np.where(ids == -1, np.nan, out.astype(np.float64, copy=False))
        else:
            out = out.astype(np.int64, copy=False)

        rev = np.empty(count, dtype=np.intp)
        rev[sorter] = np.arange(count, dtype=np.intp)
        return out[rev]

    # -----------------------------------------------------------------

    @final
    @property
    def _obj_1d_constructor(self) -> Callable:
        # GH28330 preserve subclassed Series/DataFrames
        if isinstance(self.obj, DataFrame):
            return self.obj._constructor_sliced
        assert isinstance(self.obj, Series)
        return self.obj._constructor

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def any(self, skipna: bool = True) -> NDFrameT:
        """
        Return True if any value in the group is truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if any element
            is True within its respective group, False otherwise.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 0], index=lst)
        >>> ser
        a    1
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).any()
        a     True
        b    False
        dtype: bool

        For DataFrameGroupBy:

        >>> data = [[1, 0, 3], [1, 0, 6], [7, 1, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["ostrich", "penguin", "parrot"])
        >>> df
                 a  b  c
        ostrich  1  0  3
        penguin  1  0  6
        parrot   7  1  9
        >>> df.groupby(by=["a"]).any()
               b      c
        a
        1  False   True
        7   True   True
        """
        return self._cython_agg_general(
            "any",
            alt=lambda x: Series(x, copy=False).any(skipna=skipna),
            skipna=skipna,
        )

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def all(self, skipna: bool = True) -> NDFrameT:
        """
        Return True if all values in the group are truthful, else False.

        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.

        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if all elements
            are True within its respective group, False otherwise.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 0], index=lst)
        >>> ser
        a    1
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).all()
        a     True
        b    False
        dtype: bool

        For DataFrameGroupBy:

        >>> data = [[1, 0, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["ostrich", "penguin", "parrot"])
        >>> df
                 a  b  c
        ostrich  1  0  3
        penguin  1  5  6
        parrot   7  8  9
        >>> df.groupby(by=["a"]).all()
               b      c
        a
        1  False   True
        7   True   True
        """
        return self._cython_agg_general(
            "all",
            alt=lambda x: Series(x, copy=False).all(skipna=skipna),
            skipna=skipna,
        )

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def count(self) -> NDFrameT:
        """
        Compute count of group, excluding missing values.

        Returns
        -------
        Series or DataFrame
            Count of values within each group.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, np.nan], index=lst)
        >>> ser
        a    1.0
        a    2.0
        b    NaN
        dtype: float64
        >>> ser.groupby(level=0).count()
        a    2
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, np.nan, 3], [1, np.nan, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["cow", "horse", "bull"])
        >>> df
                a	  b	c
        cow     1	NaN	3
        horse	1	NaN	6
        bull	7	8.0	9
        >>> df.groupby("a").count()
            b   c
        a
        1   0   2
        7   1   1

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').count()
        2023-01-01    2
        2023-02-01    2
        Freq: MS, dtype: int64
        """
        data = self._get_data_to_aggregate()
        ids, _, ngroups = self._grouper.group_info
        mask = ids != -1

        is_series = data.ndim == 1

        def hfunc(bvalues: ArrayLike) -> ArrayLike:
            # TODO(EA2D): reshape would not be necessary with 2D EAs
            if bvalues.ndim == 1:
                # EA
                masked = mask & ~isna(bvalues).reshape(1, -1)
            else:
                masked = mask & ~isna(bvalues)

            counted = lib.count_level_2d(masked, labels=ids, max_bin=ngroups)
            if isinstance(bvalues, BaseMaskedArray):
                return IntegerArray(
                    counted[0], mask=np.zeros(counted.shape[1], dtype=np.bool_)
                )
            elif isinstance(bvalues, ArrowExtensionArray) and not isinstance(
                bvalues.dtype, StringDtype
            ):
                dtype = pandas_dtype("int64[pyarrow]")
                return type(bvalues)._from_sequence(counted[0], dtype=dtype)
            if is_series:
                assert counted.ndim == 2
                assert counted.shape[0] == 1
                return counted[0]
            return counted

        new_mgr = data.grouped_reduce(hfunc)
        new_obj = self._wrap_agged_manager(new_mgr)

        # If we are grouping on categoricals we want unobserved categories to
        # return zero, rather than the default of NaN which the reindexing in
        # _wrap_aggregated_output() returns. GH 35028
        # e.g. test_dataframe_groupby_on_2_categoricals_when_observed_is_false
        with com.temp_setattr(self, "observed", True):
            result = self._wrap_aggregated_output(new_obj)

        return self._reindex_output(result, fill_value=0)

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None`` and defaults to ``False``.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

            .. versionadded:: 1.4.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

            .. versionadded:: 1.4.0

        Returns
        -------
        pandas.Series or pandas.DataFrame
        %(see_also)s
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> df.groupby('A').mean()
             B         C
        A
        1  3.0  1.333333
        2  4.0  1.500000

        Groupby two columns and return the mean of the remaining column.

        >>> df.groupby(['A', 'B']).mean()
                 C
        A B
        1 2.0  2.0
          4.0  1.0
        2 3.0  1.0
          5.0  2.0

        Groupby one column and return the mean of only particular column in
        the group.

        >>> df.groupby('A')['B'].mean()
        A
        1    3.0
        2    4.0
        Name: B, dtype: float64
        """

        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_mean

            return self._numba_agg_general(
                grouped_mean,
                executor.float_dtype_mapping,
                engine_kwargs,
                min_periods=0,
            )
        else:
            result = self._cython_agg_general(
                "mean",
                alt=lambda x: Series(x, copy=False).mean(numeric_only=numeric_only),
                numeric_only=numeric_only,
            )
            return result.__finalize__(self.obj, method="groupby")

    @final
    def median(self, numeric_only: bool = False) -> NDFrameT:
        """
        Compute median of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None`` and defaults to False.

        Returns
        -------
        Series or DataFrame
            Median of values within each group.

        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).median()
        a    7.0
        b    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).median()
                 a    b
        dog    3.0  4.0
        mouse  7.0  3.0

        For Resampler:

        >>> ser = pd.Series([1, 2, 3, 3, 4, 5],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').median()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64
        """
        result = self._cython_agg_general(
            "median",
            alt=lambda x: Series(x, copy=False).median(numeric_only=numeric_only),
            numeric_only=numeric_only,
        )
        return result.__finalize__(self.obj, method="groupby")

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def std(
        self,
        ddof: int = 1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        numeric_only: bool = False,
    ):
        """
        Compute standard deviation of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

            .. versionadded:: 1.4.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

            .. versionadded:: 1.4.0

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Standard deviation of values within each group.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).std()
        a    3.21455
        b    0.57735
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).std()
                      a         b
        dog    2.000000  3.511885
        mouse  2.217356  1.500000
        """
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_var

            return np.sqrt(
                self._numba_agg_general(
                    grouped_var,
                    executor.float_dtype_mapping,
                    engine_kwargs,
                    min_periods=0,
                    ddof=ddof,
                )
            )
        else:
            return self._cython_agg_general(
                "std",
                alt=lambda x: Series(x, copy=False).std(ddof=ddof),
                numeric_only=numeric_only,
                ddof=ddof,
            )

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def var(
        self,
        ddof: int = 1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        numeric_only: bool = False,
    ):
        """
        Compute variance of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``

            .. versionadded:: 1.4.0

        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``

            .. versionadded:: 1.4.0

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Variance of values within each group.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).var()
        a    10.333333
        b     0.333333
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).var()
                      a          b
        dog    4.000000  12.333333
        mouse  4.916667   2.250000
        """
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_var

            return self._numba_agg_general(
                grouped_var,
                executor.float_dtype_mapping,
                engine_kwargs,
                min_periods=0,
                ddof=ddof,
            )
        else:
            return self._cython_agg_general(
                "var",
                alt=lambda x: Series(x, copy=False).var(ddof=ddof),
                numeric_only=numeric_only,
                ddof=ddof,
            )

    @final
    def _value_counts(
        self,
        subset: Sequence[Hashable] | None = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> DataFrame | Series:
        """
        Shared implementation of value_counts for SeriesGroupBy and DataFrameGroupBy.

        SeriesGroupBy additionally supports a bins argument. See the docstring of
        DataFrameGroupBy.value_counts for a description of arguments.
        """
        if self.axis == 1:
            raise NotImplementedError(
                "DataFrameGroupBy.value_counts only handles axis=0"
            )
        name = "proportion" if normalize else "count"

        df = self.obj
        obj = self._obj_with_exclusions

        in_axis_names = {
            grouping.name for grouping in self._grouper.groupings if grouping.in_axis
        }
        if isinstance(obj, Series):
            _name = obj.name
            keys = [] if _name in in_axis_names else [obj]
        else:
            unique_cols = set(obj.columns)
            if subset is not None:
                subsetted = set(subset)
                clashing = subsetted & set(in_axis_names)
                if clashing:
                    raise ValueError(
                        f"Keys {clashing} in subset cannot be in "
                        "the groupby column keys."
                    )
                doesnt_exist = subsetted - unique_cols
                if doesnt_exist:
                    raise ValueError(
                        f"Keys {doesnt_exist} in subset do not "
                        f"exist in the DataFrame."
                    )
            else:
                subsetted = unique_cols

            keys = [
                # Can't use .values because the column label needs to be preserved
                obj.iloc[:, idx]
                for idx, _name in enumerate(obj.columns)
                if _name not in in_axis_names and _name in subsetted
            ]

        groupings = list(self._grouper.groupings)
        for key in keys:
            grouper, _, _ = get_grouper(
                df,
                key=key,
                axis=self.axis,
                sort=self.sort,
                observed=False,
                dropna=dropna,
            )
            groupings += list(grouper.groupings)

        # Take the size of the overall columns
        gb = df.groupby(
            groupings,
            sort=self.sort,
            observed=self.observed,
            dropna=self.dropna,
        )
        result_series = cast(Series, gb.size())
        result_series.name = name

        # GH-46357 Include non-observed categories
        # of non-grouping columns regardless of `observed`
        if any(
            isinstance(grouping.grouping_vector, (Categorical, CategoricalIndex))
            and not grouping._observed
            for grouping in groupings
        ):
            levels_list = [ping._result_index for ping in groupings]
            multi_index = MultiIndex.from_product(
                levels_list, names=[ping.name for ping in groupings]
            )
            result_series = result_series.reindex(multi_index, fill_value=0)

        if sort:
            # Sort by the values
            result_series = result_series.sort_values(
                ascending=ascending, kind="stable"
            )
        if self.sort:
            # Sort by the groupings
            names = result_series.index.names
            # GH#55951 - Temporarily replace names in case they are integers
            result_series.index.names = range(len(names))
            index_level = list(range(len(self._grouper.groupings)))
            result_series = result_series.sort_index(
                level=index_level, sort_remaining=False
            )
            result_series.index.names = names

        if normalize:
            # Normalize the results by dividing by the original group sizes.
            # We are guaranteed to have the first N levels be the
            # user-requested grouping.
            levels = list(
                range(len(self._grouper.groupings), result_series.index.nlevels)
            )
            indexed_group_size = result_series.groupby(
                result_series.index.droplevel(levels),
                sort=self.sort,
                dropna=self.dropna,
                # GH#43999 - deprecation of observed=False
                observed=False,
            ).transform("sum")
            result_series /= indexed_group_size

            # Handle groups of non-observed categories
            result_series = result_series.fillna(0.0)

        result: Series | DataFrame
        if self.as_index:
            result = result_series
        else:
            # Convert to frame
            index = result_series.index
            columns = com.fill_missing_names(index.names)
            if name in columns:
                raise ValueError(f"Column label '{name}' is duplicate of result column")
            result_series.name = name
            result_series.index = index.set_names(range(len(columns)))
            result_frame = result_series.reset_index()
            orig_dtype = self._grouper.groupings[0].obj.columns.dtype  # type: ignore[union-attr]
            cols = Index(columns, dtype=orig_dtype).insert(len(columns), name)
            result_frame.columns = cols
            result = result_frame
        return result.__finalize__(self.obj, method="value_counts")

    @final
    def sem(self, ddof: int = 1, numeric_only: bool = False) -> NDFrameT:
        """
        Compute standard error of the mean of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Standard error of the mean of values within each group.

        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([5, 10, 8, 14], index=lst)
        >>> ser
        a     5
        a    10
        b     8
        b    14
        dtype: int64
        >>> ser.groupby(level=0).sem()
        a    2.5
        b    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 12, 11], [1, 15, 2], [2, 5, 8], [2, 6, 12]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tuna", "salmon", "catfish", "goldfish"])
        >>> df
                   a   b   c
            tuna   1  12  11
          salmon   1  15   2
         catfish   2   5   8
        goldfish   2   6  12
        >>> df.groupby("a").sem()
              b  c
        a
        1    1.5  4.5
        2    0.5  2.0

        For Resampler:

        >>> ser = pd.Series([1, 3, 2, 4, 3, 8],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').sem()
        2023-01-01    0.577350
        2023-02-01    1.527525
        Freq: MS, dtype: float64
        """
        if numeric_only and self.obj.ndim == 1 and not is_numeric_dtype(self.obj.dtype):
            raise TypeError(
                f"{type(self).__name__}.sem called with "
                f"numeric_only={numeric_only} and dtype {self.obj.dtype}"
            )
        return self._cython_agg_general(
            "sem",
            alt=lambda x: Series(x, copy=False).sem(ddof=ddof),
            numeric_only=numeric_only,
            ddof=ddof,
        )

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def size(self) -> DataFrame | Series:
        """
        Compute group sizes.

        Returns
        -------
        DataFrame or Series
            Number of rows in each group as a Series if as_index is True
            or a DataFrame if as_index is False.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a     1
        a     2
        b     3
        dtype: int64
        >>> ser.groupby(level=0).size()
        a    2
        b    1
        dtype: int64

        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby("a").size()
        a
        1    2
        7    1
        dtype: int64

        For Resampler:

        >>> ser = pd.Series([1, 2, 3], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        dtype: int64
        >>> ser.resample('MS').size()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        result = self._grouper.size()
        dtype_backend: None | Literal["pyarrow", "numpy_nullable"] = None
        if isinstance(self.obj, Series):
            if isinstance(self.obj.array, ArrowExtensionArray):
                if isinstance(self.obj.array, ArrowStringArrayNumpySemantics):
                    dtype_backend = None
                elif isinstance(self.obj.array, ArrowStringArray):
                    dtype_backend = "numpy_nullable"
                else:
                    dtype_backend = "pyarrow"
            elif isinstance(self.obj.array, BaseMaskedArray):
                dtype_backend = "numpy_nullable"
        # TODO: For DataFrames what if columns are mixed arrow/numpy/masked?

        # GH28330 preserve subclassed Series/DataFrames through calls
        if isinstance(self.obj, Series):
            result = self._obj_1d_constructor(result, name=self.obj.name)
        else:
            result = self._obj_1d_constructor(result)

        if dtype_backend is not None:
            result = result.convert_dtypes(
                infer_objects=False,
                convert_string=False,
                convert_boolean=False,
                convert_floating=False,
                dtype_backend=dtype_backend,
            )

        with com.temp_setattr(self, "as_index", True):
            # size already has the desired behavior in GH#49519, but this makes the
            # as_index=False path of _reindex_output fail on categorical groupers.
            result = self._reindex_output(result, fill_value=0)
        if not self.as_index:
            # error: Incompatible types in assignment (expression has
            # type "DataFrame", variable has type "Series")
            result = result.rename("size").reset_index()  # type: ignore[assignment]
        return result

    @final
    @doc(
        _groupby_agg_method_engine_template,
        fname="sum",
        no=False,
        mc=0,
        e=None,
        ek=None,
        example=dedent(
            """\
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).sum()
        a    3
        b    7
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").sum()
             b   c
        a
        1   10   7
        2   11  17"""
        ),
    )
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_sum

            return self._numba_agg_general(
                grouped_sum,
                executor.default_dtype_mapping,
                engine_kwargs,
                min_periods=min_count,
            )
        else:
            # If we are grouping on categoricals we want unobserved categories to
            # return zero, rather than the default of NaN which the reindexing in
            # _agg_general() returns. GH #31422
            with com.temp_setattr(self, "observed", True):
                result = self._agg_general(
                    numeric_only=numeric_only,
                    min_count=min_count,
                    alias="sum",
                    npfunc=np.sum,
                )

            return self._reindex_output(result, fill_value=0)

    @final
    @doc(
        _groupby_agg_method_template,
        fname="prod",
        no=False,
        mc=0,
        example=dedent(
            """\
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).prod()
        a    2
        b   12
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").prod()
             b    c
        a
        1   16   10
        2   30   72"""
        ),
    )
    def prod(self, numeric_only: bool = False, min_count: int = 0) -> NDFrameT:
        return self._agg_general(
            numeric_only=numeric_only, min_count=min_count, alias="prod", npfunc=np.prod
        )

    @final
    @doc(
        _groupby_agg_method_engine_template,
        fname="min",
        no=False,
        mc=-1,
        e=None,
        ek=None,
        example=dedent(
            """\
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).min()
        a    1
        b    3
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").min()
            b  c
        a
        1   2  2
        2   5  8"""
        ),
    )
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_min_max

            return self._numba_agg_general(
                grouped_min_max,
                executor.identity_dtype_mapping,
                engine_kwargs,
                min_periods=min_count,
                is_max=False,
            )
        else:
            return self._agg_general(
                numeric_only=numeric_only,
                min_count=min_count,
                alias="min",
                npfunc=np.min,
            )

    @final
    @doc(
        _groupby_agg_method_engine_template,
        fname="max",
        no=False,
        mc=-1,
        e=None,
        ek=None,
        example=dedent(
            """\
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).max()
        a    2
        b    4
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").max()
            b  c
        a
        1   8  5
        2   6  9"""
        ),
    )
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        if maybe_use_numba(engine):
            from pandas.core._numba.kernels import grouped_min_max

            return self._numba_agg_general(
                grouped_min_max,
                executor.identity_dtype_mapping,
                engine_kwargs,
                min_periods=min_count,
                is_max=True,
            )
        else:
            return self._agg_general(
                numeric_only=numeric_only,
                min_count=min_count,
                alias="max",
                npfunc=np.max,
            )

    @final
    def first(
        self, numeric_only: bool = False, min_count: int = -1, skipna: bool = True
    ) -> NDFrameT:
        """
        Compute the first entry of each column within each group.

        Defaults to skipping NA elements.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` valid values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.

            .. versionadded:: 2.2.1

        Returns
        -------
        Series or DataFrame
            First values within each group.

        See Also
        --------
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.
        pandas.core.groupby.DataFrameGroupBy.last : Compute the last non-null entry
            of each column.
        pandas.core.groupby.DataFrameGroupBy.nth : Take the nth row from each group.

        Examples
        --------
        >>> df = pd.DataFrame(dict(A=[1, 1, 3], B=[None, 5, 6], C=[1, 2, 3],
        ...                        D=['3/11/2000', '3/12/2000', '3/13/2000']))
        >>> df['D'] = pd.to_datetime(df['D'])
        >>> df.groupby("A").first()
             B  C          D
        A
        1  5.0  1 2000-03-11
        3  6.0  3 2000-03-13
        >>> df.groupby("A").first(min_count=2)
            B    C          D
        A
        1 NaN  1.0 2000-03-11
        3 NaN  NaN        NaT
        >>> df.groupby("A").first(numeric_only=True)
             B  C
        A
        1  5.0  1
        3  6.0  3
        """

        def first_compat(obj: NDFrameT, axis: AxisInt = 0):
            def first(x: Series):
                """Helper function for first item that isn't NA."""
                arr = x.array[notna(x.array)]
                if not len(arr):
                    return x.array.dtype.na_value
                return arr[0]

            if isinstance(obj, DataFrame):
                return obj.apply(first, axis=axis)
            elif isinstance(obj, Series):
                return first(obj)
            else:  # pragma: no cover
                raise TypeError(type(obj))

        return self._agg_general(
            numeric_only=numeric_only,
            min_count=min_count,
            alias="first",
            npfunc=first_compat,
            skipna=skipna,
        )

    @final
    def last(
        self, numeric_only: bool = False, min_count: int = -1, skipna: bool = True
    ) -> NDFrameT:
        """
        Compute the last entry of each column within each group.

        Defaults to skipping NA elements.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns. If None, will attempt to use
            everything, then use only numeric data.
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` valid values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.

            .. versionadded:: 2.2.1

        Returns
        -------
        Series or DataFrame
            Last of values within each group.

        See Also
        --------
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.
        pandas.core.groupby.DataFrameGroupBy.first : Compute the first non-null entry
            of each column.
        pandas.core.groupby.DataFrameGroupBy.nth : Take the nth row from each group.

        Examples
        --------
        >>> df = pd.DataFrame(dict(A=[1, 1, 3], B=[5, None, 6], C=[1, 2, 3]))
        >>> df.groupby("A").last()
             B  C
        A
        1  5.0  2
        3  6.0  3
        """

        def last_compat(obj: NDFrameT, axis: AxisInt = 0):
            def last(x: Series):
                """Helper function for last item that isn't NA."""
                arr = x.array[notna(x.array)]
                if not len(arr):
                    return x.array.dtype.na_value
                return arr[-1]

            if isinstance(obj, DataFrame):
                return obj.apply(last, axis=axis)
            elif isinstance(obj, Series):
                return last(obj)
            else:  # pragma: no cover
                raise TypeError(type(obj))

        return self._agg_general(
            numeric_only=numeric_only,
            min_count=min_count,
            alias="last",
            npfunc=last_compat,
            skipna=skipna,
        )

    @final
    def ohlc(self) -> DataFrame:
        """
        Compute open, high, low and close values of a group, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Returns
        -------
        DataFrame
            Open, high, low and close values within each group.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['SPX', 'CAC', 'SPX', 'CAC', 'SPX', 'CAC', 'SPX', 'CAC',]
        >>> ser = pd.Series([3.4, 9.0, 7.2, 5.2, 8.8, 9.4, 0.1, 0.5], index=lst)
        >>> ser
        SPX     3.4
        CAC     9.0
        SPX     7.2
        CAC     5.2
        SPX     8.8
        CAC     9.4
        SPX     0.1
        CAC     0.5
        dtype: float64
        >>> ser.groupby(level=0).ohlc()
             open  high  low  close
        CAC   9.0   9.4  0.5    0.5
        SPX   3.4   8.8  0.1    0.1

        For DataFrameGroupBy:

        >>> data = {2022: [1.2, 2.3, 8.9, 4.5, 4.4, 3, 2 , 1],
        ...         2023: [3.4, 9.0, 7.2, 5.2, 8.8, 9.4, 8.2, 1.0]}
        >>> df = pd.DataFrame(data, index=['SPX', 'CAC', 'SPX', 'CAC',
        ...                   'SPX', 'CAC', 'SPX', 'CAC'])
        >>> df
             2022  2023
        SPX   1.2   3.4
        CAC   2.3   9.0
        SPX   8.9   7.2
        CAC   4.5   5.2
        SPX   4.4   8.8
        CAC   3.0   9.4
        SPX   2.0   8.2
        CAC   1.0   1.0
        >>> df.groupby(level=0).ohlc()
            2022                 2023
            open high  low close open high  low close
        CAC  2.3  4.5  1.0   1.0  9.0  9.4  1.0   1.0
        SPX  1.2  8.9  1.2   2.0  3.4  8.8  3.4   8.2

        For Resampler:

        >>> ser = pd.Series([1, 3, 2, 4, 3, 5],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').ohlc()
                    open  high  low  close
        2023-01-01     1     3    1      2
        2023-02-01     4     5    3      5
        """
        if self.obj.ndim == 1:
            obj = self._selected_obj

            is_numeric = is_numeric_dtype(obj.dtype)
            if not is_numeric:
                raise DataError("No numeric types to aggregate")

            res_values = self._grouper._cython_operation(
                "aggregate", obj._values, "ohlc", axis=0, min_count=-1
            )

            agg_names = ["open", "high", "low", "close"]
            result = self.obj._constructor_expanddim(
                res_values, index=self._grouper.result_index, columns=agg_names
            )
            return self._reindex_output(result)

        result = self._apply_to_column_groupbys(lambda sgb: sgb.ohlc())
        return result

    @doc(DataFrame.describe)
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ) -> NDFrameT:
        obj = self._obj_with_exclusions

        if len(obj) == 0:
            described = obj.describe(
                percentiles=percentiles, include=include, exclude=exclude
            )
            if obj.ndim == 1:
                result = described
            else:
                result = described.unstack()
            return result.to_frame().T.iloc[:0]

        with com.temp_setattr(self, "as_index", True):
            result = self._python_apply_general(
                lambda x: x.describe(
                    percentiles=percentiles, include=include, exclude=exclude
                ),
                obj,
                not_indexed_same=True,
            )
        if self.axis == 1:
            return result.T

        # GH#49256 - properly handle the grouping column(s)
        result = result.unstack()
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))

        return result

    @final
    def resample(self, rule, *args, include_groups: bool = True, **kwargs) -> Resampler:
        """
        Provide resampling when using a TimeGrouper.

        Given a grouper, the function resamples it according to a string
        "string" -> "frequency".

        See the :ref:`frequency aliases <timeseries.offset_aliases>`
        documentation for more details.

        Parameters
        ----------
        rule : str or DateOffset
            The offset string or object representing target grouper conversion.
        *args
            Possible arguments are `how`, `fill_method`, `limit`, `kind` and
            `on`, and other arguments of `TimeGrouper`.
        include_groups : bool, default True
            When True, will attempt to include the groupings in the operation in
            the case that they are columns of the DataFrame. If this raises a
            TypeError, the result will be computed with the groupings excluded.
            When False, the groupings will be excluded when applying ``func``.

            .. versionadded:: 2.2.0

            .. deprecated:: 2.2.0

               Setting include_groups to True is deprecated. Only the value
               False will be allowed in a future version of pandas.

        **kwargs
            Possible arguments are `how`, `fill_method`, `limit`, `kind` and
            `on`, and other arguments of `TimeGrouper`.

        Returns
        -------
        pandas.api.typing.DatetimeIndexResamplerGroupby,
        pandas.api.typing.PeriodIndexResamplerGroupby, or
        pandas.api.typing.TimedeltaIndexResamplerGroupby
            Return a new groupby object, with type depending on the data
            being resampled.

        See Also
        --------
        Grouper : Specify a frequency to resample with when
            grouping by a key.
        DatetimeIndex.resample : Frequency conversion and resampling of
            time series.

        Examples
        --------
        >>> idx = pd.date_range('1/1/2000', periods=4, freq='min')
        >>> df = pd.DataFrame(data=4 * [range(2)],
        ...                   index=idx,
        ...                   columns=['a', 'b'])
        >>> df.iloc[2, 0] = 5
        >>> df
                            a  b
        2000-01-01 00:00:00  0  1
        2000-01-01 00:01:00  0  1
        2000-01-01 00:02:00  5  1
        2000-01-01 00:03:00  0  1

        Downsample the DataFrame into 3 minute bins and sum the values of
        the timestamps falling into a bin.

        >>> df.groupby('a').resample('3min', include_groups=False).sum()
                                 b
        a
        0   2000-01-01 00:00:00  2
            2000-01-01 00:03:00  1
        5   2000-01-01 00:00:00  1

        Upsample the series into 30 second bins.

        >>> df.groupby('a').resample('30s', include_groups=False).sum()
                            b
        a
        0   2000-01-01 00:00:00  1
            2000-01-01 00:00:30  0
            2000-01-01 00:01:00  1
            2000-01-01 00:01:30  0
            2000-01-01 00:02:00  0
            2000-01-01 00:02:30  0
            2000-01-01 00:03:00  1
        5   2000-01-01 00:02:00  1

        Resample by month. Values are assigned to the month of the period.

        >>> df.groupby('a').resample('ME', include_groups=False).sum()
                    b
        a
        0   2000-01-31  3
        5   2000-01-31  1

        Downsample the series into 3 minute bins as above, but close the right
        side of the bin interval.

        >>> (
        ...     df.groupby('a')
        ...     .resample('3min', closed='right', include_groups=False)
        ...     .sum()
        ... )
                                 b
        a
        0   1999-12-31 23:57:00  1
            2000-01-01 00:00:00  2
        5   2000-01-01 00:00:00  1

        Downsample the series into 3 minute bins and close the right side of
        the bin interval, but label each bin using the right edge instead of
        the left.

        >>> (
        ...     df.groupby('a')
        ...     .resample('3min', closed='right', label='right', include_groups=False)
        ...     .sum()
        ... )
                                 b
        a
        0   2000-01-01 00:00:00  1
            2000-01-01 00:03:00  2
        5   2000-01-01 00:03:00  1
        """
        from pandas.core.resample import get_resampler_for_grouping

        # mypy flags that include_groups could be specified via `*args` or `**kwargs`
        # GH#54961 would resolve.
        return get_resampler_for_grouping(  # type: ignore[misc]
            self, rule, *args, include_groups=include_groups, **kwargs
        )

    @final
    def rolling(self, *args, **kwargs) -> RollingGroupby:
        """
        Return a rolling grouper, providing rolling functionality per group.

        Parameters
        ----------
        window : int, timedelta, str, offset, or BaseIndexer subclass
            Size of the moving window.

            If an integer, the fixed number of observations used for
            each window.

            If a timedelta, str, or offset, the time period of each window. Each
            window will be a variable sized based on the observations included in
            the time-period. This is only valid for datetimelike indexes.
            To learn more about the offsets & frequency strings, please see `this link
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

            If a BaseIndexer subclass, the window boundaries
            based on the defined ``get_window_bounds`` method. Additional rolling
            keyword arguments, namely ``min_periods``, ``center``, ``closed`` and
            ``step`` will be passed to ``get_window_bounds``.

        min_periods : int, default None
            Minimum number of observations in window required to have a value;
            otherwise, result is ``np.nan``.

            For a window that is specified by an offset,
            ``min_periods`` will default to 1.

            For a window that is specified by an integer, ``min_periods`` will default
            to the size of the window.

        center : bool, default False
            If False, set the window labels as the right edge of the window index.

            If True, set the window labels as the center of the window index.

        win_type : str, default None
            If ``None``, all points are evenly weighted.

            If a string, it must be a valid `scipy.signal window function
            <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.

            Certain Scipy window types require additional parameters to be passed
            in the aggregation function. The additional parameters must match
            the keywords specified in the Scipy window type method signature.

        on : str, optional
            For a DataFrame, a column label or Index level on which
            to calculate the rolling window, rather than the DataFrame's index.

            Provided integer column is ignored and excluded from result since
            an integer index is not used to calculate the rolling window.

        axis : int or str, default 0
            If ``0`` or ``'index'``, roll across the rows.

            If ``1`` or ``'columns'``, roll across the columns.

            For `Series` this parameter is unused and defaults to 0.

        closed : str, default None
            If ``'right'``, the first point in the window is excluded from calculations.

            If ``'left'``, the last point in the window is excluded from calculations.

            If ``'both'``, no points in the window are excluded from calculations.

            If ``'neither'``, the first and last points in the window are excluded
            from calculations.

            Default ``None`` (``'right'``).

        method : str {'single', 'table'}, default 'single'
            Execute the rolling operation per single column or row (``'single'``)
            or over the entire object (``'table'``).

            This argument is only implemented when specifying ``engine='numba'``
            in the method call.

        Returns
        -------
        pandas.api.typing.RollingGroupby
            Return a new grouper with our rolling appended.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby.

        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 1, 2, 2],
        ...                    'B': [1, 2, 3, 4],
        ...                    'C': [0.362, 0.227, 1.267, -0.562]})
        >>> df
              A  B      C
        0     1  1  0.362
        1     1  2  0.227
        2     2  3  1.267
        3     2  4 -0.562

        >>> df.groupby('A').rolling(2).sum()
            B      C
        A
        1 0  NaN    NaN
          1  3.0  0.589
        2 2  NaN    NaN
          3  7.0  0.705

        >>> df.groupby('A').rolling(2, min_periods=1).sum()
            B      C
        A
        1 0  1.0  0.362
          1  3.0  0.589
        2 2  3.0  1.267
          3  7.0  0.705

        >>> df.groupby('A').rolling(2, on='B').sum()
            B      C
        A
        1 0  1    NaN
          1  2  0.589
        2 2  3    NaN
          3  4  0.705
        """
        from pandas.core.window import RollingGroupby

        return RollingGroupby(
            self._selected_obj,
            *args,
            _grouper=self._grouper,
            _as_index=self.as_index,
            **kwargs,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def expanding(self, *args, **kwargs) -> ExpandingGroupby:
        """
        Return an expanding grouper, providing expanding
        functionality per group.

        Returns
        -------
        pandas.api.typing.ExpandingGroupby
        """
        from pandas.core.window import ExpandingGroupby

        return ExpandingGroupby(
            self._selected_obj,
            *args,
            _grouper=self._grouper,
            **kwargs,
        )

    @final
    @Substitution(name="groupby")
    @Appender(_common_see_also)
    def ewm(self, *args, **kwargs) -> ExponentialMovingWindowGroupby:
        """
        Return an ewm grouper, providing ewm functionality per group.

        Returns
        -------
        pandas.api.typing.ExponentialMovingWindowGroupby
        """
        from pandas.core.window import ExponentialMovingWindowGroupby

        return ExponentialMovingWindowGroupby(
            self._selected_obj,
            *args,
            _grouper=self._grouper,
            **kwargs,
        )

    @final
    def _fill(self, direction: Literal["ffill", "bfill"], limit: int | None = None):
        """
        Shared function for `pad` and `backfill` to call Cython method.

        Parameters
        ----------
        direction : {'ffill', 'bfill'}
            Direction passed to underlying Cython function. `bfill` will cause
            values to be filled backwards. `ffill` and any other values will
            default to a forward fill
        limit : int, default None
            Maximum number of consecutive values to fill. If `None`, this
            method will convert to -1 prior to passing to Cython

        Returns
        -------
        `Series` or `DataFrame` with filled values

        See Also
        --------
        pad : Returns Series with minimum number of char in object.
        backfill : Backward fill the missing values in the dataset.
        """
        # Need int value for Cython
        if limit is None:
            limit = -1

        ids, _, _ = self._grouper.group_info
        sorted_labels = np.argsort(ids, kind="mergesort").astype(np.intp, copy=False)
        if direction == "bfill":
            sorted_labels = sorted_labels[::-1]

        col_func = partial(
            libgroupby.group_fillna_indexer,
            labels=ids,
            sorted_labels=sorted_labels,
            limit=limit,
            dropna=self.dropna,
        )

        def blk_func(values: ArrayLike) -> ArrayLike:
            mask = isna(values)
            if values.ndim == 1:
                indexer = np.empty(values.shape, dtype=np.intp)
                col_func(out=indexer, mask=mask)
                return algorithms.take_nd(values, indexer)

            else:
                # We broadcast algorithms.take_nd analogous to
                #  np.take_along_axis
                if isinstance(values, np.ndarray):
                    dtype = values.dtype
                    if self._grouper.has_dropped_na:
                        # dropped null groups give rise to nan in the result
                        dtype = ensure_dtype_can_hold_na(values.dtype)
                    out = np.empty(values.shape, dtype=dtype)
                else:
                    # Note: we only get here with backfill/pad,
                    #  so if we have a dtype that cannot hold NAs,
                    #  then there will be no -1s in indexer, so we can use
                    #  the original dtype (no need to ensure_dtype_can_hold_na)
                    out = type(values)._empty(values.shape, dtype=values.dtype)

                for i, value_element in enumerate(values):
                    # call group_fillna_indexer column-wise
                    indexer = np.empty(values.shape[1], dtype=np.intp)
                    col_func(out=indexer, mask=mask[i])
                    out[i, :] = algorithms.take_nd(value_element, indexer)
                return out

        mgr = self._get_data_to_aggregate()
        res_mgr = mgr.apply(blk_func)

        new_obj = self._wrap_agged_manager(res_mgr)

        if self.axis == 1:
            # Only relevant for DataFrameGroupBy
            new_obj = new_obj.T
            new_obj.columns = self.obj.columns

        new_obj.index = self.obj.index
        return new_obj

    @final
    @Substitution(name="groupby")
    def ffill(self, limit: int | None = None):
        """
        Forward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.ffill: Returns Series with minimum number of char in object.
        DataFrame.ffill: Object with missing values filled or None if inplace=True.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.

        Examples
        --------

        For SeriesGroupBy:

        >>> key = [0, 0, 1, 1]
        >>> ser = pd.Series([np.nan, 2, 3, np.nan], index=key)
        >>> ser
        0    NaN
        0    2.0
        1    3.0
        1    NaN
        dtype: float64
        >>> ser.groupby(level=0).ffill()
        0    NaN
        0    2.0
        1    3.0
        1    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> df = pd.DataFrame(
        ...     {
        ...         "key": [0, 0, 1, 1, 1],
        ...         "A": [np.nan, 2, np.nan, 3, np.nan],
        ...         "B": [2, 3, np.nan, np.nan, np.nan],
        ...         "C": [np.nan, np.nan, 2, np.nan, np.nan],
        ...     }
        ... )
        >>> df
           key    A    B   C
        0    0  NaN  2.0 NaN
        1    0  2.0  3.0 NaN
        2    1  NaN  NaN 2.0
        3    1  3.0  NaN NaN
        4    1  NaN  NaN NaN

        Propagate non-null values forward or backward within each group along columns.

        >>> df.groupby("key").ffill()
             A    B   C
        0  NaN  2.0 NaN
        1  2.0  3.0 NaN
        2  NaN  NaN 2.0
        3  3.0  NaN 2.0
        4  3.0  NaN 2.0

        Propagate non-null values forward or backward within each group along rows.

        >>> df.T.groupby(np.array([0, 0, 1, 1])).ffill().T
           key    A    B    C
        0  0.0  0.0  2.0  2.0
        1  0.0  2.0  3.0  3.0
        2  1.0  1.0  NaN  2.0
        3  1.0  3.0  NaN  NaN
        4  1.0  1.0  NaN  NaN

        Only replace the first NaN element within a group along rows.

        >>> df.groupby("key").ffill(limit=1)
             A    B    C
        0  NaN  2.0  NaN
        1  2.0  3.0  NaN
        2  NaN  NaN  2.0
        3  3.0  NaN  2.0
        4  3.0  NaN  NaN
        """
        return self._fill("ffill", limit=limit)

    @final
    @Substitution(name="groupby")
    def bfill(self, limit: int | None = None):
        """
        Backward fill the values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            Object with missing values filled.

        See Also
        --------
        Series.bfill :  Backward fill the missing values in the dataset.
        DataFrame.bfill:  Backward fill the missing values in the dataset.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.

        Examples
        --------

        With Series:

        >>> index = ['Falcon', 'Falcon', 'Parrot', 'Parrot', 'Parrot']
        >>> s = pd.Series([None, 1, None, None, 3], index=index)
        >>> s
        Falcon    NaN
        Falcon    1.0
        Parrot    NaN
        Parrot    NaN
        Parrot    3.0
        dtype: float64
        >>> s.groupby(level=0).bfill()
        Falcon    1.0
        Falcon    1.0
        Parrot    3.0
        Parrot    3.0
        Parrot    3.0
        dtype: float64
        >>> s.groupby(level=0).bfill(limit=1)
        Falcon    1.0
        Falcon    1.0
        Parrot    NaN
        Parrot    3.0
        Parrot    3.0
        dtype: float64

        With DataFrame:

        >>> df = pd.DataFrame({'A': [1, None, None, None, 4],
        ...                    'B': [None, None, 5, None, 7]}, index=index)
        >>> df
                  A	    B
        Falcon	1.0	  NaN
        Falcon	NaN	  NaN
        Parrot	NaN	  5.0
        Parrot	NaN	  NaN
        Parrot	4.0	  7.0
        >>> df.groupby(level=0).bfill()
                  A	    B
        Falcon	1.0	  NaN
        Falcon	NaN	  NaN
        Parrot	4.0	  5.0
        Parrot	4.0	  7.0
        Parrot	4.0	  7.0
        >>> df.groupby(level=0).bfill(limit=1)
                  A	    B
        Falcon	1.0	  NaN
        Falcon	NaN	  NaN
        Parrot	NaN	  5.0
        Parrot	4.0	  7.0
        Parrot	4.0	  7.0
        """
        return self._fill("bfill", limit=limit)

    @final
    @property
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def nth(self) -> GroupByNthSelector:
        """
        Take the nth row from each group if n is an int, otherwise a subset of rows.

        Can be either a call or an index. dropna is not available with index notation.
        Index notation accepts a comma separated list of integers and slices.

        If dropna, will take the nth non-null row, dropna is either
        'all' or 'any'; this is equivalent to calling dropna(how=dropna)
        before the groupby.

        Parameters
        ----------
        n : int, slice or list of ints and slices
            A single nth value for the row or a list of nth values or slices.

            .. versionchanged:: 1.4.0
                Added slice and lists containing slices.
                Added index notation.

        dropna : {'any', 'all', None}, default None
            Apply the specified dropna operation before counting which row is
            the nth row. Only supported if n is an int.

        Returns
        -------
        Series or DataFrame
            N-th value within each group.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5]}, columns=['A', 'B'])
        >>> g = df.groupby('A')
        >>> g.nth(0)
           A   B
        0  1 NaN
        2  2 3.0
        >>> g.nth(1)
           A   B
        1  1 2.0
        4  2 5.0
        >>> g.nth(-1)
           A   B
        3  1 4.0
        4  2 5.0
        >>> g.nth([0, 1])
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0
        4  2 5.0
        >>> g.nth(slice(None, -1))
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0

        Index notation may also be used

        >>> g.nth[0, 1]
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0
        4  2 5.0
        >>> g.nth[:-1]
           A   B
        0  1 NaN
        1  1 2.0
        2  2 3.0

        Specifying `dropna` allows ignoring ``NaN`` values

        >>> g.nth(0, dropna='any')
           A   B
        1  1 2.0
        2  2 3.0

        When the specified ``n`` is larger than any of the groups, an
        empty DataFrame is returned

        >>> g.nth(3, dropna='any')
        Empty DataFrame
        Columns: [A, B]
        Index: []
        """
        return GroupByNthSelector(self)

    def _nth(
        self,
        n: PositionalIndexer | tuple,
        dropna: Literal["any", "all", None] = None,
    ) -> NDFrameT:
        if not dropna:
            mask = self._make_mask_from_positional_indexer(n)

            ids, _, _ = self._grouper.group_info

            # Drop NA values in grouping
            mask = mask & (ids != -1)

            out = self._mask_selected_obj(mask)
            return out

        # dropna is truthy
        if not is_integer(n):
            raise ValueError("dropna option only supported for an integer argument")

        if dropna not in ["any", "all"]:
            # Note: when agg-ing picker doesn't raise this, just returns NaN
            raise ValueError(
                "For a DataFrame or Series groupby.nth, dropna must be "
                "either None, 'any' or 'all', "
                f"(was passed {dropna})."
            )

        # old behaviour, but with all and any support for DataFrames.
        # modified in GH 7559 to have better perf
        n = cast(int, n)
        dropped = self._selected_obj.dropna(how=dropna, axis=self.axis)

        # get a new grouper for our dropped obj
        grouper: np.ndarray | Index | ops.BaseGrouper
        if len(dropped) == len(self._selected_obj):
            # Nothing was dropped, can use the same grouper
            grouper = self._grouper
        else:
            # we don't have the grouper info available
            # (e.g. we have selected out
            # a column that is not in the current object)
            axis = self._grouper.axis
            grouper = self._grouper.codes_info[axis.isin(dropped.index)]
            if self._grouper.has_dropped_na:
                # Null groups need to still be encoded as -1 when passed to groupby
                nulls = grouper == -1
                # error: No overload variant of "where" matches argument types
                #        "Any", "NAType", "Any"
                values = np.where(nulls, NA, grouper)  # type: ignore[call-overload]
                grouper = Index(values, dtype="Int64")

        if self.axis == 1:
            grb = dropped.T.groupby(grouper, as_index=self.as_index, sort=self.sort)
        else:
            grb = dropped.groupby(grouper, as_index=self.as_index, sort=self.sort)
        return grb.nth(n)

    @final
    def quantile(
        self,
        q: float | AnyArrayLike = 0.5,
        interpolation: str = "linear",
        numeric_only: bool = False,
    ):
        """
        Return group values at the given quantile, a la numpy.percentile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            Value(s) between 0 and 1 providing the quantile(s) to compute.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Method to use when the desired quantile falls between two points.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Return type determined by caller of GroupBy object.

        See Also
        --------
        Series.quantile : Similar method for Series.
        DataFrame.quantile : Similar method for DataFrame.
        numpy.percentile : NumPy method to compute qth percentile.

        Examples
        --------
        >>> df = pd.DataFrame([
        ...     ['a', 1], ['a', 2], ['a', 3],
        ...     ['b', 1], ['b', 3], ['b', 5]
        ... ], columns=['key', 'val'])
        >>> df.groupby('key').quantile()
            val
        key
        a    2.0
        b    3.0
        """
        mgr = self._get_data_to_aggregate(numeric_only=numeric_only, name="quantile")
        obj = self._wrap_agged_manager(mgr)
        if self.axis == 1:
            splitter = self._grouper._get_splitter(obj.T, axis=self.axis)
            sdata = splitter._sorted_data.T
        else:
            splitter = self._grouper._get_splitter(obj, axis=self.axis)
            sdata = splitter._sorted_data

        starts, ends = lib.generate_slices(splitter._slabels, splitter.ngroups)

        def pre_processor(vals: ArrayLike) -> tuple[np.ndarray, DtypeObj | None]:
            if isinstance(vals.dtype, StringDtype) or is_object_dtype(vals.dtype):
                raise TypeError(
                    f"dtype '{vals.dtype}' does not support operation 'quantile'"
                )

            inference: DtypeObj | None = None
            if isinstance(vals, BaseMaskedArray) and is_numeric_dtype(vals.dtype):
                out = vals.to_numpy(dtype=float, na_value=np.nan)
                inference = vals.dtype
            elif is_integer_dtype(vals.dtype):
                if isinstance(vals, ExtensionArray):
                    out = vals.to_numpy(dtype=float, na_value=np.nan)
                else:
                    out = vals
                inference = np.dtype(np.int64)
            elif is_bool_dtype(vals.dtype) and isinstance(vals, ExtensionArray):
                out = vals.to_numpy(dtype=float, na_value=np.nan)
            elif is_bool_dtype(vals.dtype):
                # GH#51424 deprecate to match Series/DataFrame behavior
                warnings.warn(
                    f"Allowing bool dtype in {type(self).__name__}.quantile is "
                    "deprecated and will raise in a future version, matching "
                    "the Series/DataFrame behavior. Cast to uint8 dtype before "
                    "calling quantile instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                out = np.asarray(vals)
            elif needs_i8_conversion(vals.dtype):
                inference = vals.dtype
                # In this case we need to delay the casting until after the
                #  np.lexsort below.
                # error: Incompatible return value type (got
                # "Tuple[Union[ExtensionArray, ndarray[Any, Any]], Union[Any,
                # ExtensionDtype]]", expected "Tuple[ndarray[Any, Any],
                # Optional[Union[dtype[Any], ExtensionDtype]]]")
                return vals, inference  # type: ignore[return-value]
            elif isinstance(vals, ExtensionArray) and is_float_dtype(vals.dtype):
                inference = np.dtype(np.float64)
                out = vals.to_numpy(dtype=float, na_value=np.nan)
            else:
                out = np.asarray(vals)

            return out, inference

        def post_processor(
            vals: np.ndarray,
            inference: DtypeObj | None,
            result_mask: np.ndarray | None,
            orig_vals: ArrayLike,
        ) -> ArrayLike:
            if inference:
                # Check for edge case
                if isinstance(orig_vals, BaseMaskedArray):
                    assert result_mask is not None  # for mypy

                    if interpolation in {"linear", "midpoint"} and not is_float_dtype(
                        orig_vals
                    ):
                        return FloatingArray(vals, result_mask)
                    else:
                        # Item "ExtensionDtype" of "Union[ExtensionDtype, str,
                        # dtype[Any], Type[object]]" has no attribute "numpy_dtype"
                        # [union-attr]
                        with warnings.catch_warnings():
                            # vals.astype with nan can warn with numpy >1.24
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            return type(orig_vals)(
                                vals.astype(
                                    inference.numpy_dtype  # type: ignore[union-attr]
                                ),
                                result_mask,
                            )

                elif not (
                    is_integer_dtype(inference)
                    and interpolation in {"linear", "midpoint"}
                ):
                    if needs_i8_conversion(inference):
                        # error: Item "ExtensionArray" of "Union[ExtensionArray,
                        # ndarray[Any, Any]]" has no attribute "_ndarray"
                        vals = vals.astype("i8").view(
                            orig_vals._ndarray.dtype  # type: ignore[union-attr]
                        )
                        # error: Item "ExtensionArray" of "Union[ExtensionArray,
                        # ndarray[Any, Any]]" has no attribute "_from_backing_data"
                        return orig_vals._from_backing_data(  # type: ignore[union-attr]
                            vals
                        )

                    assert isinstance(inference, np.dtype)  # for mypy
                    return vals.astype(inference)

            return vals

        qs = np.array(q, dtype=np.float64)
        pass_qs: np.ndarray | None = qs
        if is_scalar(q):
            qs = np.array([q], dtype=np.float64)
            pass_qs = None

        ids, _, ngroups = self._grouper.group_info
        nqs = len(qs)

        func = partial(
            libgroupby.group_quantile,
            labels=ids,
            qs=qs,
            interpolation=interpolation,
            starts=starts,
            ends=ends,
        )

        def blk_func(values: ArrayLike) -> ArrayLike:
            orig_vals = values
            if isinstance(values, BaseMaskedArray):
                mask = values._mask
                result_mask = np.zeros((ngroups, nqs), dtype=np.bool_)
            else:
                mask = isna(values)
                result_mask = None

            is_datetimelike = needs_i8_conversion(values.dtype)

            vals, inference = pre_processor(values)

            ncols = 1
            if vals.ndim == 2:
                ncols = vals.shape[0]

            out = np.empty((ncols, ngroups, nqs), dtype=np.float64)

            if is_datetimelike:
                vals = vals.view("i8")

            if vals.ndim == 1:
                # EA is always 1d
                func(
                    out[0],
                    values=vals,
                    mask=mask,
                    result_mask=result_mask,
                    is_datetimelike=is_datetimelike,
                )
            else:
                for i in range(ncols):
                    func(
                        out[i],
                        values=vals[i],
                        mask=mask[i],
                        result_mask=None,
                        is_datetimelike=is_datetimelike,
                    )

            if vals.ndim == 1:
                out = out.ravel("K")
                if result_mask is not None:
                    result_mask = result_mask.ravel("K")
            else:
                out = out.reshape(ncols, ngroups * nqs)

            return post_processor(out, inference, result_mask, orig_vals)

        res_mgr = sdata._mgr.grouped_reduce(blk_func)

        res = self._wrap_agged_manager(res_mgr)
        return self._wrap_aggregated_output(res, qs=pass_qs)

    @final
    @Substitution(name="groupby")
    def ngroup(self, ascending: bool = True):
        """
        Number each group from 0 to the number of groups - 1.

        This is the enumerative complement of cumcount.  Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.

        Groups with missing keys (where `pd.isna()` is True) will be labeled with `NaN`
        and will be skipped from the count.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.

        Returns
        -------
        Series
            Unique numbers for each group.

        See Also
        --------
        .cumcount : Number the rows in each group.

        Examples
        --------
        >>> df = pd.DataFrame({"color": ["red", None, "red", "blue", "blue", "red"]})
        >>> df
           color
        0    red
        1   None
        2    red
        3   blue
        4   blue
        5    red
        >>> df.groupby("color").ngroup()
        0    1.0
        1    NaN
        2    1.0
        3    0.0
        4    0.0
        5    1.0
        dtype: float64
        >>> df.groupby("color", dropna=False).ngroup()
        0    1
        1    2
        2    1
        3    0
        4    0
        5    1
        dtype: int64
        >>> df.groupby("color", dropna=False).ngroup(ascending=False)
        0    1
        1    0
        2    1
        3    2
        4    2
        5    1
        dtype: int64
        """
        obj = self._obj_with_exclusions
        index = obj._get_axis(self.axis)
        comp_ids = self._grouper.group_info[0]

        dtype: type
        if self._grouper.has_dropped_na:
            comp_ids = np.where(comp_ids == -1, np.nan, comp_ids)
            dtype = np.float64
        else:
            dtype = np.int64

        if any(ping._passed_categorical for ping in self._grouper.groupings):
            # comp_ids reflect non-observed groups, we need only observed
            comp_ids = rank_1d(comp_ids, ties_method="dense") - 1

        result = self._obj_1d_constructor(comp_ids, index, dtype=dtype)
        if not ascending:
            result = self.ngroups - 1 - result
        return result

    @final
    @Substitution(name="groupby")
    def cumcount(self, ascending: bool = True):
        """
        Number each item in each group from 0 to the length of that group - 1.

        Essentially this is equivalent to

        .. code-block:: python

            self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Returns
        -------
        Series
            Sequence number of each element within each group.

        See Also
        --------
        .ngroup : Number the groups themselves.

        Examples
        --------
        >>> df = pd.DataFrame([['a'], ['a'], ['a'], ['b'], ['b'], ['a']],
        ...                   columns=['A'])
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').cumcount()
        0    0
        1    1
        2    2
        3    0
        4    1
        5    3
        dtype: int64
        >>> df.groupby('A').cumcount(ascending=False)
        0    3
        1    2
        2    1
        3    1
        4    0
        5    0
        dtype: int64
        """
        index = self._obj_with_exclusions._get_axis(self.axis)
        cumcounts = self._cumcount_array(ascending=ascending)
        return self._obj_1d_constructor(cumcounts, index)

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def rank(
        self,
        method: str = "average",
        ascending: bool = True,
        na_option: str = "keep",
        pct: bool = False,
        axis: AxisInt | lib.NoDefault = lib.no_default,
    ) -> NDFrameT:
        """
        Provide the rank of values within each group.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            * average: average rank of group.
            * min: lowest rank in group.
            * max: highest rank in group.
            * first: ranks assigned in order they appear in the array.
            * dense: like 'min', but rank always increases by 1 between groups.
        ascending : bool, default True
            False for ranks by high (1) to low (N).
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            * keep: leave NA values where they are.
            * top: smallest rank if ascending.
            * bottom: smallest rank if descending.
        pct : bool, default False
            Compute percentage rank of data within each group.
        axis : int, default 0
            The axis of the object over which to compute the rank.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        Returns
        -------
        DataFrame with ranking of values within each group
        %(see_also)s
        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "group": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        ...         "value": [2, 4, 2, 3, 5, 1, 2, 4, 1, 5],
        ...     }
        ... )
        >>> df
          group  value
        0     a      2
        1     a      4
        2     a      2
        3     a      3
        4     a      5
        5     b      1
        6     b      2
        7     b      4
        8     b      1
        9     b      5
        >>> for method in ['average', 'min', 'max', 'dense', 'first']:
        ...     df[f'{method}_rank'] = df.groupby('group')['value'].rank(method)
        >>> df
          group  value  average_rank  min_rank  max_rank  dense_rank  first_rank
        0     a      2           1.5       1.0       2.0         1.0         1.0
        1     a      4           4.0       4.0       4.0         3.0         4.0
        2     a      2           1.5       1.0       2.0         1.0         2.0
        3     a      3           3.0       3.0       3.0         2.0         3.0
        4     a      5           5.0       5.0       5.0         4.0         5.0
        5     b      1           1.5       1.0       2.0         1.0         1.0
        6     b      2           3.0       3.0       3.0         2.0         3.0
        7     b      4           4.0       4.0       4.0         3.0         4.0
        8     b      1           1.5       1.0       2.0         1.0         2.0
        9     b      5           5.0       5.0       5.0         4.0         5.0
        """
        if na_option not in {"keep", "top", "bottom"}:
            msg = "na_option must be one of 'keep', 'top', or 'bottom'"
            raise ValueError(msg)

        if axis is not lib.no_default:
            axis = self.obj._get_axis_number(axis)
            self._deprecate_axis(axis, "rank")
        else:
            axis = 0

        kwargs = {
            "ties_method": method,
            "ascending": ascending,
            "na_option": na_option,
            "pct": pct,
        }
        if axis != 0:
            # DataFrame uses different keyword name
            kwargs["method"] = kwargs.pop("ties_method")
            f = lambda x: x.rank(axis=axis, numeric_only=False, **kwargs)
            result = self._python_apply_general(
                f, self._selected_obj, is_transform=True
            )
            return result

        return self._cython_transform(
            "rank",
            numeric_only=False,
            axis=axis,
            **kwargs,
        )

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def cumprod(
        self, axis: Axis | lib.NoDefault = lib.no_default, *args, **kwargs
    ) -> NDFrameT:
        """
        Cumulative product for each group.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([6, 2, 0], index=lst)
        >>> ser
        a    6
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).cumprod()
        a    6
        a   12
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["cow", "horse", "bull"])
        >>> df
                a   b   c
        cow     1   8   2
        horse   1   2   5
        bull    2   6   9
        >>> df.groupby("a").groups
        {1: ['cow', 'horse'], 2: ['bull']}
        >>> df.groupby("a").cumprod()
                b   c
        cow     8   2
        horse  16  10
        bull    6   9
        """
        nv.validate_groupby_func("cumprod", args, kwargs, ["numeric_only", "skipna"])
        if axis is not lib.no_default:
            axis = self.obj._get_axis_number(axis)
            self._deprecate_axis(axis, "cumprod")
        else:
            axis = 0

        if axis != 0:
            f = lambda x: x.cumprod(axis=axis, **kwargs)
            return self._python_apply_general(f, self._selected_obj, is_transform=True)

        return self._cython_transform("cumprod", **kwargs)

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def cumsum(
        self, axis: Axis | lib.NoDefault = lib.no_default, *args, **kwargs
    ) -> NDFrameT:
        """
        Cumulative sum for each group.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([6, 2, 0], index=lst)
        >>> ser
        a    6
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).cumsum()
        a    6
        a    8
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["fox", "gorilla", "lion"])
        >>> df
                  a   b   c
        fox       1   8   2
        gorilla   1   2   5
        lion      2   6   9
        >>> df.groupby("a").groups
        {1: ['fox', 'gorilla'], 2: ['lion']}
        >>> df.groupby("a").cumsum()
                  b   c
        fox       8   2
        gorilla  10   7
        lion      6   9
        """
        nv.validate_groupby_func("cumsum", args, kwargs, ["numeric_only", "skipna"])
        if axis is not lib.no_default:
            axis = self.obj._get_axis_number(axis)
            self._deprecate_axis(axis, "cumsum")
        else:
            axis = 0

        if axis != 0:
            f = lambda x: x.cumsum(axis=axis, **kwargs)
            return self._python_apply_general(f, self._selected_obj, is_transform=True)

        return self._cython_transform("cumsum", **kwargs)

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def cummin(
        self,
        axis: AxisInt | lib.NoDefault = lib.no_default,
        numeric_only: bool = False,
        **kwargs,
    ) -> NDFrameT:
        """
        Cumulative min for each group.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([1, 6, 2, 3, 0, 4], index=lst)
        >>> ser
        a    1
        a    6
        a    2
        b    3
        b    0
        b    4
        dtype: int64
        >>> ser.groupby(level=0).cummin()
        a    1
        a    1
        a    1
        b    3
        b    0
        b    0
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 0, 2], [1, 1, 5], [6, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["snake", "rabbit", "turtle"])
        >>> df
                a   b   c
        snake   1   0   2
        rabbit  1   1   5
        turtle  6   6   9
        >>> df.groupby("a").groups
        {1: ['snake', 'rabbit'], 6: ['turtle']}
        >>> df.groupby("a").cummin()
                b   c
        snake   0   2
        rabbit  0   2
        turtle  6   9
        """
        skipna = kwargs.get("skipna", True)
        if axis is not lib.no_default:
            axis = self.obj._get_axis_number(axis)
            self._deprecate_axis(axis, "cummin")
        else:
            axis = 0

        if axis != 0:
            f = lambda x: np.minimum.accumulate(x, axis)
            obj = self._selected_obj
            if numeric_only:
                obj = obj._get_numeric_data()
            return self._python_apply_general(f, obj, is_transform=True)

        return self._cython_transform(
            "cummin", numeric_only=numeric_only, skipna=skipna
        )

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def cummax(
        self,
        axis: AxisInt | lib.NoDefault = lib.no_default,
        numeric_only: bool = False,
        **kwargs,
    ) -> NDFrameT:
        """
        Cumulative max for each group.

        Returns
        -------
        Series or DataFrame
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([1, 6, 2, 3, 1, 4], index=lst)
        >>> ser
        a    1
        a    6
        a    2
        b    3
        b    1
        b    4
        dtype: int64
        >>> ser.groupby(level=0).cummax()
        a    1
        a    6
        a    6
        b    3
        b    3
        b    4
        dtype: int64

        For DataFrameGroupBy:

        >>> data = [[1, 8, 2], [1, 1, 0], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["cow", "horse", "bull"])
        >>> df
                a   b   c
        cow     1   8   2
        horse   1   1   0
        bull    2   6   9
        >>> df.groupby("a").groups
        {1: ['cow', 'horse'], 2: ['bull']}
        >>> df.groupby("a").cummax()
                b   c
        cow     8   2
        horse   8   2
        bull    6   9
        """
        skipna = kwargs.get("skipna", True)
        if axis is not lib.no_default:
            axis = self.obj._get_axis_number(axis)
            self._deprecate_axis(axis, "cummax")
        else:
            axis = 0

        if axis != 0:
            f = lambda x: np.maximum.accumulate(x, axis)
            obj = self._selected_obj
            if numeric_only:
                obj = obj._get_numeric_data()
            return self._python_apply_general(f, obj, is_transform=True)

        return self._cython_transform(
            "cummax", numeric_only=numeric_only, skipna=skipna
        )

    @final
    @Substitution(name="groupby")
    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq=None,
        axis: Axis | lib.NoDefault = lib.no_default,
        fill_value=lib.no_default,
        suffix: str | None = None,
    ):
        """
        Shift each group by periods observations.

        If freq is passed, the index will be increased using the periods and the freq.

        Parameters
        ----------
        periods : int | Sequence[int], default 1
            Number of periods to shift. If a list of values, shift each group by
            each period.
        freq : str, optional
            Frequency string.
        axis : axis to shift, default 0
            Shift direction.

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        fill_value : optional
            The scalar value to use for newly introduced missing values.

            .. versionchanged:: 2.1.0
                Will raise a ``ValueError`` if ``freq`` is provided too.

        suffix : str, optional
            A string to add to each shifted column if there are multiple periods.
            Ignored otherwise.

        Returns
        -------
        Series or DataFrame
            Object shifted within each group.

        See Also
        --------
        Index.shift : Shift values of Index.

        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).shift(1)
        a    NaN
        a    1.0
        b    NaN
        b    3.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tuna", "salmon", "catfish", "goldfish"])
        >>> df
                   a  b  c
            tuna   1  2  3
          salmon   1  5  6
         catfish   2  5  8
        goldfish   2  6  9
        >>> df.groupby("a").shift(1)
                      b    c
            tuna    NaN  NaN
          salmon    2.0  3.0
         catfish    NaN  NaN
        goldfish    5.0  8.0
        """
        if axis is not lib.no_default:
            axis = self.obj._get_axis_number(axis)
            self._deprecate_axis(axis, "shift")
        else:
            axis = 0

        if is_list_like(periods):
            if axis == 1:
                raise ValueError(
                    "If `periods` contains multiple shifts, `axis` cannot be 1."
                )
            periods = cast(Sequence, periods)
            if len(periods) == 0:
                raise ValueError("If `periods` is an iterable, it cannot be empty.")
            from pandas.core.reshape.concat import concat

            add_suffix = True
        else:
            if not is_integer(periods):
                raise TypeError(
                    f"Periods must be integer, but {periods} is {type(periods)}."
                )
            if suffix:
                raise ValueError("Cannot specify `suffix` if `periods` is an int.")
            periods = [cast(int, periods)]
            add_suffix = False

        shifted_dataframes = []
        for period in periods:
            if not is_integer(period):
                raise TypeError(
                    f"Periods must be integer, but {period} is {type(period)}."
                )
            period = cast(int, period)
            if freq is not None or axis != 0:
                f = lambda x: x.shift(
                    period, freq, axis, fill_value  # pylint: disable=cell-var-from-loop
                )
                shifted = self._python_apply_general(
                    f, self._selected_obj, is_transform=True
                )
            else:
                if fill_value is lib.no_default:
                    fill_value = None
                ids, _, ngroups = self._grouper.group_info
                res_indexer = np.zeros(len(ids), dtype=np.int64)

                libgroupby.group_shift_indexer(res_indexer, ids, ngroups, period)

                obj = self._obj_with_exclusions

                shifted = obj._reindex_with_indexers(
                    {self.axis: (obj.axes[self.axis], res_indexer)},
                    fill_value=fill_value,
                    allow_dups=True,
                )

            if add_suffix:
                if isinstance(shifted, Series):
                    shifted = cast(NDFrameT, shifted.to_frame())
                shifted = shifted.add_suffix(
                    f"{suffix}_{period}" if suffix else f"_{period}"
                )
            shifted_dataframes.append(cast(Union[Series, DataFrame], shifted))

        return (
            shifted_dataframes[0]
            if len(shifted_dataframes) == 1
            else concat(shifted_dataframes, axis=1)
        )

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def diff(
        self, periods: int = 1, axis: AxisInt | lib.NoDefault = lib.no_default
    ) -> NDFrameT:
        """
        First discrete difference of element.

        Calculates the difference of each element compared with another
        element in the group (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.
        axis : axis to shift, default 0
            Take difference over rows (0) or columns (1).

            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.

        Returns
        -------
        Series or DataFrame
            First differences.
        %(see_also)s
        Examples
        --------
        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).diff()
        a    NaN
        a   -5.0
        a    6.0
        b    NaN
        b   -1.0
        b    0.0
        dtype: float64

        For DataFrameGroupBy:

        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).diff()
                 a    b
          dog  NaN  NaN
          dog  2.0  3.0
          dog  2.0  4.0
        mouse  NaN  NaN
        mouse  0.0  0.0
        mouse  1.0 -2.0
        mouse -5.0 -1.0
        """
        if axis is not lib.no_default:
            axis = self.obj._get_axis_number(axis)
            self._deprecate_axis(axis, "diff")
        else:
            axis = 0

        if axis != 0:
            return self.apply(lambda x: x.diff(periods=periods, axis=axis))

        obj = self._obj_with_exclusions
        shifted = self.shift(periods=periods)

        # GH45562 - to retain existing behavior and match behavior of Series.diff(),
        # int8 and int16 are coerced to float32 rather than float64.
        dtypes_to_f32 = ["int8", "int16"]
        if obj.ndim == 1:
            if obj.dtype in dtypes_to_f32:
                shifted = shifted.astype("float32")
        else:
            to_coerce = [c for c, dtype in obj.dtypes.items() if dtype in dtypes_to_f32]
            if len(to_coerce):
                shifted = shifted.astype({c: "float32" for c in to_coerce})

        return obj - shifted

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def pct_change(
        self,
        periods: int = 1,
        fill_method: FillnaOptions | None | lib.NoDefault = lib.no_default,
        limit: int | None | lib.NoDefault = lib.no_default,
        freq=None,
        axis: Axis | lib.NoDefault = lib.no_default,
    ):
        """
        Calculate pct_change of each value to previous entry in group.

        Returns
        -------
        Series or DataFrame
            Percentage changes within each group.
        %(see_also)s
        Examples
        --------

        For SeriesGroupBy:

        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).pct_change()
        a         NaN
        a    1.000000
        b         NaN
        b    0.333333
        dtype: float64

        For DataFrameGroupBy:

        >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tuna", "salmon", "catfish", "goldfish"])
        >>> df
                   a  b  c
            tuna   1  2  3
          salmon   1  5  6
         catfish   2  5  8
        goldfish   2  6  9
        >>> df.groupby("a").pct_change()
                    b  c
            tuna    NaN    NaN
          salmon    1.5  1.000
         catfish    NaN    NaN
        goldfish    0.2  0.125
        """
        # GH#53491
        if fill_method not in (lib.no_default, None) or limit is not lib.no_default:
            warnings.warn(
                "The 'fill_method' keyword being not None and the 'limit' keyword in "
                f"{type(self).__name__}.pct_change are deprecated and will be removed "
                "in a future version. Either fill in any non-leading NA values prior "
                "to calling pct_change or specify 'fill_method=None' to not fill NA "
                "values.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if fill_method is lib.no_default:
            if limit is lib.no_default and any(
                grp.isna().values.any() for _, grp in self
            ):
                warnings.warn(
                    "The default fill_method='ffill' in "
                    f"{type(self).__name__}.pct_change is deprecated and will "
                    "be removed in a future version. Either fill in any "
                    "non-leading NA values prior to calling pct_change or "
                    "specify 'fill_method=None' to not fill NA values.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            fill_method = "ffill"
        if limit is lib.no_default:
            limit = None

        if axis is not lib.no_default:
            axis = self.obj._get_axis_number(axis)
            self._deprecate_axis(axis, "pct_change")
        else:
            axis = 0

        # TODO(GH#23918): Remove this conditional for SeriesGroupBy when
        #  GH#23918 is fixed
        if freq is not None or axis != 0:
            f = lambda x: x.pct_change(
                periods=periods,
                fill_method=fill_method,
                limit=limit,
                freq=freq,
                axis=axis,
            )
            return self._python_apply_general(f, self._selected_obj, is_transform=True)

        if fill_method is None:  # GH30463
            fill_method = "ffill"
            limit = 0
        filled = getattr(self, fill_method)(limit=limit)
        if self.axis == 0:
            fill_grp = filled.groupby(self._grouper.codes, group_keys=self.group_keys)
        else:
            fill_grp = filled.T.groupby(self._grouper.codes, group_keys=self.group_keys)
        shifted = fill_grp.shift(periods=periods, freq=freq)
        if self.axis == 1:
            shifted = shifted.T
        return (filled / shifted) - 1

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def head(self, n: int = 5) -> NDFrameT:
        """
        Return first n rows of each group.

        Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Parameters
        ----------
        n : int
            If positive: number of entries to include from start of each group.
            If negative: number of entries to exclude from end of each group.

        Returns
        -------
        Series or DataFrame
            Subset of original Series or DataFrame as determined by n.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame([[1, 2], [1, 4], [5, 6]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').head(1)
           A  B
        0  1  2
        2  5  6
        >>> df.groupby('A').head(-1)
           A  B
        0  1  2
        """
        mask = self._make_mask_from_positional_indexer(slice(None, n))
        return self._mask_selected_obj(mask)

    @final
    @Substitution(name="groupby")
    @Substitution(see_also=_common_see_also)
    def tail(self, n: int = 5) -> NDFrameT:
        """
        Return last n rows of each group.

        Similar to ``.apply(lambda x: x.tail(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).

        Parameters
        ----------
        n : int
            If positive: number of entries to include from end of each group.
            If negative: number of entries to exclude from start of each group.

        Returns
        -------
        Series or DataFrame
            Subset of original Series or DataFrame as determined by n.
        %(see_also)s
        Examples
        --------

        >>> df = pd.DataFrame([['a', 1], ['a', 2], ['b', 1], ['b', 2]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').tail(1)
           A  B
        1  a  2
        3  b  2
        >>> df.groupby('A').tail(-1)
           A  B
        1  a  2
        3  b  2
        """
        if n:
            mask = self._make_mask_from_positional_indexer(slice(-n, None))
        else:
            mask = self._make_mask_from_positional_indexer([])

        return self._mask_selected_obj(mask)

    @final
    def _mask_selected_obj(self, mask: npt.NDArray[np.bool_]) -> NDFrameT:
        """
        Return _selected_obj with mask applied to the correct axis.

        Parameters
        ----------
        mask : np.ndarray[bool]
            Boolean mask to apply.

        Returns
        -------
        Series or DataFrame
            Filtered _selected_obj.
        """
        ids = self._grouper.group_info[0]
        mask = mask & (ids != -1)

        if self.axis == 0:
            return self._selected_obj[mask]
        else:
            return self._selected_obj.iloc[:, mask]

    @final
    def _reindex_output(
        self,
        output: OutputFrameOrSeries,
        fill_value: Scalar = np.nan,
        qs: npt.NDArray[np.float64] | None = None,
    ) -> OutputFrameOrSeries:
        """
        If we have categorical groupers, then we might want to make sure that
        we have a fully re-indexed output to the levels. This means expanding
        the output space to accommodate all values in the cartesian product of
        our groups, regardless of whether they were observed in the data or
        not. This will expand the output space if there are missing groups.

        The method returns early without modifying the input if the number of
        groupings is less than 2, self.observed == True or none of the groupers
        are categorical.

        Parameters
        ----------
        output : Series or DataFrame
            Object resulting from grouping and applying an operation.
        fill_value : scalar, default np.nan
            Value to use for unobserved categories if self.observed is False.
        qs : np.ndarray[float64] or None, default None
            quantile values, only relevant for quantile.

        Returns
        -------
        Series or DataFrame
            Object (potentially) re-indexed to include all possible groups.
        """
        groupings = self._grouper.groupings
        if len(groupings) == 1:
            return output

        # if we only care about the observed values
        # we are done
        elif self.observed:
            return output

        # reindexing only applies to a Categorical grouper
        elif not any(
            isinstance(ping.grouping_vector, (Categorical, CategoricalIndex))
            for ping in groupings
        ):
            return output

        levels_list = [ping._group_index for ping in groupings]
        names = self._grouper.names
        if qs is not None:
            # error: Argument 1 to "append" of "list" has incompatible type
            # "ndarray[Any, dtype[floating[_64Bit]]]"; expected "Index"
            levels_list.append(qs)  # type: ignore[arg-type]
            names = names + [None]
        index = MultiIndex.from_product(levels_list, names=names)
        if self.sort:
            index = index.sort_values()

        if self.as_index:
            # Always holds for SeriesGroupBy unless GH#36507 is implemented
            d = {
                self.obj._get_axis_name(self.axis): index,
                "copy": False,
                "fill_value": fill_value,
            }
            return output.reindex(**d)  # type: ignore[arg-type]

        # GH 13204
        # Here, the categorical in-axis groupers, which need to be fully
        # expanded, are columns in `output`. An idea is to do:
        # output = output.set_index(self._grouper.names)
        #                .reindex(index).reset_index()
        # but special care has to be taken because of possible not-in-axis
        # groupers.
        # So, we manually select and drop the in-axis grouper columns,
        # reindex `output`, and then reset the in-axis grouper columns.

        # Select in-axis groupers
        in_axis_grps = [
            (i, ping.name) for (i, ping) in enumerate(groupings) if ping.in_axis
        ]
        if len(in_axis_grps) > 0:
            g_nums, g_names = zip(*in_axis_grps)
            output = output.drop(labels=list(g_names), axis=1)

        # Set a temp index and reindex (possibly expanding)
        output = output.set_index(self._grouper.result_index).reindex(
            index, copy=False, fill_value=fill_value
        )

        # Reset in-axis grouper columns
        # (using level numbers `g_nums` because level names may not be unique)
        if len(in_axis_grps) > 0:
            output = output.reset_index(level=g_nums)

        return output.reset_index(drop=True)

    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: Sequence | Series | None = None,
        random_state: RandomState | None = None,
    ):
        """
        Return a random sample of items from each group.

        You can use `random_state` for reproducibility.

        Parameters
        ----------
        n : int, optional
            Number of items to return for each group. Cannot be used with
            `frac` and must be no larger than the smallest group unless
            `replace` is True. Default is one if `frac` is None.
        frac : float, optional
            Fraction of items to return. Cannot be used with `n`.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
        weights : list-like, optional
            Default None results in equal probability weighting.
            If passed a list-like then values must have the same length as
            the underlying DataFrame or Series object and will be used as
            sampling probabilities after normalization within each group.
            Values must be non-negative with at least one positive element
            within each group.
        random_state : int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
            If int, array-like, or BitGenerator, seed for random number generator.
            If np.random.RandomState or np.random.Generator, use as given.

            .. versionchanged:: 1.4.0

                np.random.Generator objects now accepted

        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing items randomly
            sampled within each group from the caller object.

        See Also
        --------
        DataFrame.sample: Generate random samples from a DataFrame object.
        numpy.random.choice: Generate a random sample from a given 1-D numpy
            array.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}
        ... )
        >>> df
               a  b
        0    red  0
        1    red  1
        2   blue  2
        3   blue  3
        4  black  4
        5  black  5

        Select one row at random for each distinct value in column a. The
        `random_state` argument can be used to guarantee reproducibility:

        >>> df.groupby("a").sample(n=1, random_state=1)
               a  b
        4  black  4
        2   blue  2
        1    red  1

        Set `frac` to sample fixed proportions rather than counts:

        >>> df.groupby("a")["b"].sample(frac=0.5, random_state=2)
        5    5
        2    2
        0    0
        Name: b, dtype: int64

        Control sample probabilities within groups by setting weights:

        >>> df.groupby("a").sample(
        ...     n=1,
        ...     weights=[1, 1, 1, 0, 0, 1],
        ...     random_state=1,
        ... )
               a  b
        5  black  5
        2   blue  2
        0    red  0
        """  # noqa: E501
        if self._selected_obj.empty:
            # GH48459 prevent ValueError when object is empty
            return self._selected_obj
        size = sample.process_sampling_size(n, frac, replace)
        if weights is not None:
            weights_arr = sample.preprocess_weights(
                self._selected_obj, weights, axis=self.axis
            )

        random_state = com.random_state(random_state)

        group_iterator = self._grouper.get_iterator(self._selected_obj, self.axis)

        sampled_indices = []
        for labels, obj in group_iterator:
            grp_indices = self.indices[labels]
            group_size = len(grp_indices)
            if size is not None:
                sample_size = size
            else:
                assert frac is not None
                sample_size = round(frac * group_size)

            grp_sample = sample.sample(
                group_size,
                size=sample_size,
                replace=replace,
                weights=None if weights is None else weights_arr[grp_indices],
                random_state=random_state,
            )
            sampled_indices.append(grp_indices[grp_sample])

        sampled_indices = np.concatenate(sampled_indices)
        return self._selected_obj.take(sampled_indices, axis=self.axis)

    def _idxmax_idxmin(
        self,
        how: Literal["idxmax", "idxmin"],
        ignore_unobserved: bool = False,
        axis: Axis | None | lib.NoDefault = lib.no_default,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> NDFrameT:
        """Compute idxmax/idxmin.

        Parameters
        ----------
        how : {'idxmin', 'idxmax'}
            Whether to compute idxmin or idxmax.
        axis : {{0 or 'index', 1 or 'columns'}}, default None
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            If axis is not provided, grouper's axis is used.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ignore_unobserved : bool, default False
            When True and an unobserved group is encountered, do not raise. This used
            for transform where unobserved groups do not play an impact on the result.

        Returns
        -------
        Series or DataFrame
            idxmax or idxmin for the groupby operation.
        """
        if axis is not lib.no_default:
            if axis is None:
                axis = self.axis
            axis = self.obj._get_axis_number(axis)
            self._deprecate_axis(axis, how)
        else:
            axis = self.axis

        if not self.observed and any(
            ping._passed_categorical for ping in self._grouper.groupings
        ):
            expected_len = np.prod(
                [len(ping._group_index) for ping in self._grouper.groupings]
            )
            if len(self._grouper.groupings) == 1:
                result_len = len(self._grouper.groupings[0].grouping_vector.unique())
            else:
                # result_index only contains observed groups in this case
                result_len = len(self._grouper.result_index)
            assert result_len <= expected_len
            has_unobserved = result_len < expected_len

            raise_err: bool | np.bool_ = not ignore_unobserved and has_unobserved
            # Only raise an error if there are columns to compute; otherwise we return
            # an empty DataFrame with an index (possibly including unobserved) but no
            # columns
            data = self._obj_with_exclusions
            if raise_err and isinstance(data, DataFrame):
                if numeric_only:
                    data = data._get_numeric_data()
                raise_err = len(data.columns) > 0

            if raise_err:
                raise ValueError(
                    f"Can't get {how} of an empty group due to unobserved categories. "
                    "Specify observed=True in groupby instead."
                )
        elif not skipna:
            if self._obj_with_exclusions.isna().any(axis=None):
                warnings.warn(
                    f"The behavior of {type(self).__name__}.{how} with all-NA "
                    "values, or any-NA and skipna=False, is deprecated. In a future "
                    "version this will raise ValueError",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

        if axis == 1:
            try:

                def func(df):
                    method = getattr(df, how)
                    return method(axis=axis, skipna=skipna, numeric_only=numeric_only)

                func.__name__ = how
                result = self._python_apply_general(
                    func, self._obj_with_exclusions, not_indexed_same=True
                )
            except ValueError as err:
                name = "argmax" if how == "idxmax" else "argmin"
                if f"attempt to get {name} of an empty sequence" in str(err):
                    raise ValueError(
                        f"Can't get {how} of an empty group due to unobserved "
                        "categories. Specify observed=True in groupby instead."
                    ) from None
                raise
            return result

        result = self._agg_general(
            numeric_only=numeric_only,
            min_count=1,
            alias=how,
            skipna=skipna,
        )
        return result

    def _wrap_idxmax_idxmin(self, res: NDFrameT) -> NDFrameT:
        index = self.obj._get_axis(self.axis)
        if res.size == 0:
            result = res.astype(index.dtype)
        else:
            if isinstance(index, MultiIndex):
                index = index.to_flat_index()
            values = res._values
            assert isinstance(values, np.ndarray)
            na_value = na_value_for_dtype(index.dtype, compat=False)
            if isinstance(res, Series):
                # mypy: expression has type "Series", variable has type "NDFrameT"
                result = res._constructor(  # type: ignore[assignment]
                    index.array.take(values, allow_fill=True, fill_value=na_value),
                    index=res.index,
                    name=res.name,
                )
            else:
                data = {}
                for k, column_values in enumerate(values.T):
                    data[k] = index.array.take(
                        column_values, allow_fill=True, fill_value=na_value
                    )
                result = self.obj._constructor(data, index=res.index)
                result.columns = res.columns
        return result


@doc(GroupBy)
def get_groupby(
    obj: NDFrame,
    by: _KeysArgType | None = None,
    axis: AxisInt = 0,
    grouper: ops.BaseGrouper | None = None,
    group_keys: bool = True,
) -> GroupBy:
    klass: type[GroupBy]
    if isinstance(obj, Series):
        from pandas.core.groupby.generic import SeriesGroupBy

        klass = SeriesGroupBy
    elif isinstance(obj, DataFrame):
        from pandas.core.groupby.generic import DataFrameGroupBy

        klass = DataFrameGroupBy
    else:  # pragma: no cover
        raise TypeError(f"invalid type: {obj}")

    return klass(
        obj=obj,
        keys=by,
        axis=axis,
        grouper=grouper,
        group_keys=group_keys,
    )


def _insert_quantile_level(idx: Index, qs: npt.NDArray[np.float64]) -> MultiIndex:
    """
    Insert the sequence 'qs' of quantiles as the inner-most level of a MultiIndex.

    The quantile level in the MultiIndex is a repeated copy of 'qs'.

    Parameters
    ----------
    idx : Index
    qs : np.ndarray[float64]

    Returns
    -------
    MultiIndex
    """
    nqs = len(qs)
    lev_codes, lev = Index(qs).factorize()
    lev_codes = coerce_indexer_dtype(lev_codes, lev)

    if idx._is_multi:
        idx = cast(MultiIndex, idx)
        levels = list(idx.levels) + [lev]
        codes = [np.repeat(x, nqs) for x in idx.codes] + [np.tile(lev_codes, len(idx))]
        mi = MultiIndex(levels=levels, codes=codes, names=idx.names + [None])
    else:
        nidx = len(idx)
        idx_codes = coerce_indexer_dtype(np.arange(nidx), idx)
        levels = [idx, lev]
        codes = [np.repeat(idx_codes, nqs), np.tile(lev_codes, nidx)]
        mi = MultiIndex(levels=levels, codes=codes, names=[idx.name, None])

    return mi


# GH#7155
_apply_groupings_depr = (
    "{}.{} operated on the grouping columns. This behavior is deprecated, "
    "and in a future version of pandas the grouping columns will be excluded "
    "from the operation. Either pass `include_groups=False` to exclude the "
    "groupings or explicitly select the grouping columns after groupby to silence "
    "this warning."
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\mixins.py ===
"""
Mixin classes for custom array types that don't inherit from ndarray.
"""

__all__ = ['NDArrayOperatorsMixin']


def _disables_array_ufunc(obj):
    """True when __array_ufunc__ is set to None."""
    try:
        return obj.__array_ufunc__ is None
    except AttributeError:
        return False


def _binary_method(ufunc, name):
    """Implement a forward binary method with a ufunc, e.g., __add__."""
    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(self, other)
    func.__name__ = f'__{name}__'
    return func


def _reflected_binary_method(ufunc, name):
    """Implement a reflected binary method with a ufunc, e.g., __radd__."""
    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(other, self)
    func.__name__ = f'__r{name}__'
    return func


def _inplace_binary_method(ufunc, name):
    """Implement an in-place binary method with a ufunc, e.g., __iadd__."""
    def func(self, other):
        return ufunc(self, other, out=(self,))
    func.__name__ = f'__i{name}__'
    return func


def _numeric_methods(ufunc, name):
    """Implement forward, reflected and inplace binary methods with a ufunc."""
    return (_binary_method(ufunc, name),
            _reflected_binary_method(ufunc, name),
            _inplace_binary_method(ufunc, name))


def _unary_method(ufunc, name):
    """Implement a unary special method with a ufunc."""
    def func(self):
        return ufunc(self)
    func.__name__ = f'__{name}__'
    return func


class NDArrayOperatorsMixin:
    """Mixin defining all operator special methods using __array_ufunc__.

    This class implements the special methods for almost all of Python's
    builtin operators defined in the `operator` module, including comparisons
    (``==``, ``>``, etc.) and arithmetic (``+``, ``*``, ``-``, etc.), by
    deferring to the ``__array_ufunc__`` method, which subclasses must
    implement.

    It is useful for writing classes that do not inherit from `numpy.ndarray`,
    but that should support arithmetic and numpy universal functions like
    arrays as described in :external+neps:doc:`nep-0013-ufunc-overrides`.

    As an trivial example, consider this implementation of an ``ArrayLike``
    class that simply wraps a NumPy array and ensures that the result of any
    arithmetic operation is also an ``ArrayLike`` object:

        >>> import numbers
        >>> class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
        ...     def __init__(self, value):
        ...         self.value = np.asarray(value)
        ...
        ...     # One might also consider adding the built-in list type to this
        ...     # list, to support operations like np.add(array_like, list)
        ...     _HANDLED_TYPES = (np.ndarray, numbers.Number)
        ...
        ...     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ...         out = kwargs.get('out', ())
        ...         for x in inputs + out:
        ...             # Only support operations with instances of
        ...             # _HANDLED_TYPES. Use ArrayLike instead of type(self)
        ...             # for isinstance to allow subclasses that don't
        ...             # override __array_ufunc__ to handle ArrayLike objects.
        ...             if not isinstance(
        ...                 x, self._HANDLED_TYPES + (ArrayLike,)
        ...             ):
        ...                 return NotImplemented
        ...
        ...         # Defer to the implementation of the ufunc
        ...         # on unwrapped values.
        ...         inputs = tuple(x.value if isinstance(x, ArrayLike) else x
        ...                     for x in inputs)
        ...         if out:
        ...             kwargs['out'] = tuple(
        ...                 x.value if isinstance(x, ArrayLike) else x
        ...                 for x in out)
        ...         result = getattr(ufunc, method)(*inputs, **kwargs)
        ...
        ...         if type(result) is tuple:
        ...             # multiple return values
        ...             return tuple(type(self)(x) for x in result)
        ...         elif method == 'at':
        ...             # no return value
        ...             return None
        ...         else:
        ...             # one return value
        ...             return type(self)(result)
        ...
        ...     def __repr__(self):
        ...         return '%s(%r)' % (type(self).__name__, self.value)

    In interactions between ``ArrayLike`` objects and numbers or numpy arrays,
    the result is always another ``ArrayLike``:

        >>> x = ArrayLike([1, 2, 3])
        >>> x - 1
        ArrayLike(array([0, 1, 2]))
        >>> 1 - x
        ArrayLike(array([ 0, -1, -2]))
        >>> np.arange(3) - x
        ArrayLike(array([-1, -1, -1]))
        >>> x - np.arange(3)
        ArrayLike(array([1, 1, 1]))

    Note that unlike ``numpy.ndarray``, ``ArrayLike`` does not allow operations
    with arbitrary, unrecognized types. This ensures that interactions with
    ArrayLike preserve a well-defined casting hierarchy.

    """
    from numpy._core import umath as um

    __slots__ = ()
    # Like np.ndarray, this mixin class implements "Option 1" from the ufunc
    # overrides NEP.

    # comparisons don't have reflected and in-place versions
    __lt__ = _binary_method(um.less, 'lt')
    __le__ = _binary_method(um.less_equal, 'le')
    __eq__ = _binary_method(um.equal, 'eq')
    __ne__ = _binary_method(um.not_equal, 'ne')
    __gt__ = _binary_method(um.greater, 'gt')
    __ge__ = _binary_method(um.greater_equal, 'ge')

    # numeric methods
    __add__, __radd__, __iadd__ = _numeric_methods(um.add, 'add')
    __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, 'sub')
    __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, 'mul')
    __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(
        um.matmul, 'matmul')
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
        um.true_divide, 'truediv')
    __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
        um.floor_divide, 'floordiv')
    __mod__, __rmod__, __imod__ = _numeric_methods(um.remainder, 'mod')
    __divmod__ = _binary_method(um.divmod, 'divmod')
    __rdivmod__ = _reflected_binary_method(um.divmod, 'divmod')
    # __idivmod__ does not exist
    # TODO: handle the optional third argument for __pow__?
    __pow__, __rpow__, __ipow__ = _numeric_methods(um.power, 'pow')
    __lshift__, __rlshift__, __ilshift__ = _numeric_methods(
        um.left_shift, 'lshift')
    __rshift__, __rrshift__, __irshift__ = _numeric_methods(
        um.right_shift, 'rshift')
    __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and, 'and')
    __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor, 'xor')
    __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or, 'or')

    # unary methods
    __neg__ = _unary_method(um.negative, 'neg')
    __pos__ = _unary_method(um.positive, 'pos')
    __abs__ = _unary_method(um.absolute, 'abs')
    __invert__ = _unary_method(um.invert, 'invert')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\elements.py ===
# sql/elements.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Core SQL expression elements, including :class:`_expression.ClauseElement`,
:class:`_expression.ColumnElement`, and derived classes.

"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
import itertools
import operator
import re
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import ParamSpec
from ..util.typing import Self

if typing.TYPE_CHECKING:
    from ._typing import _ByArgument
    from ._typing import _ColumnExpressionArgument
    from ._typing import _ColumnExpressionOrStrLabelArgument
    from ._typing import _HasDialect
    from ._typing import _InfoType
    from ._typing import _PropagateAttrsType
    from ._typing import _TypeEngineArgument
    from .base import ColumnSet
    from .cache_key import _CacheKeyTraversalType
    from .cache_key import CacheKey
    from .compiler import Compiled
    from .compiler import SQLCompiler
    from .functions import FunctionElement
    from .operators import OperatorType
    from .schema import Column
    from .schema import DefaultGenerator
    from .schema import FetchedValue
    from .schema import ForeignKey
    from .selectable import _SelectIterable
    from .selectable import FromClause
    from .selectable import NamedFromClause
    from .selectable import TextualSelect
    from .sqltypes import TupleType
    from .type_api import TypeEngine
    from .visitors import _CloneCallableType
    from .visitors import _TraverseInternalsType
    from .visitors import anon_map
    from ..engine import Connection
    from ..engine import Dialect
    from ..engine.interfaces import _CoreMultiExecuteParams
    from ..engine.interfaces import CacheStats
    from ..engine.interfaces import CompiledCacheType
    from ..engine.interfaces import CoreExecuteOptionsParameter
    from ..engine.interfaces import SchemaTranslateMapType
    from ..engine.result import Result

_NUMERIC = Union[float, Decimal]
_NUMBER = Union[float, int, Decimal]

_T = TypeVar("_T", bound="Any")
_T_co = TypeVar("_T_co", bound=Any, covariant=True)
_OPT = TypeVar("_OPT", bound="Any")
_NT = TypeVar("_NT", bound="_NUMERIC")

_NMT = TypeVar("_NMT", bound="_NUMBER")


@overload
def literal(
    value: Any,
    type_: _TypeEngineArgument[_T],
    literal_execute: bool = False,
) -> BindParameter[_T]: ...


@overload
def literal(
    value: _T,
    type_: None = None,
    literal_execute: bool = False,
) -> BindParameter[_T]: ...


@overload
def literal(
    value: Any,
    type_: Optional[_TypeEngineArgument[Any]] = None,
    literal_execute: bool = False,
) -> BindParameter[Any]: ...


def literal(
    value: Any,
    type_: Optional[_TypeEngineArgument[Any]] = None,
    literal_execute: bool = False,
) -> BindParameter[Any]:
    r"""Return a literal clause, bound to a bind parameter.

    Literal clauses are created automatically when non-
    :class:`_expression.ClauseElement` objects (such as strings, ints, dates,
    etc.) are
    used in a comparison operation with a :class:`_expression.ColumnElement`
    subclass,
    such as a :class:`~sqlalchemy.schema.Column` object.  Use this function
    to force the generation of a literal clause, which will be created as a
    :class:`BindParameter` with a bound value.

    :param value: the value to be bound. Can be any Python object supported by
     the underlying DB-API, or is translatable via the given type argument.

    :param type\_: an optional :class:`~sqlalchemy.types.TypeEngine` which will
     provide bind-parameter translation for this literal.

    :param literal_execute: optional bool, when True, the SQL engine will
     attempt to render the bound value directly in the SQL statement at
     execution time rather than providing as a parameter value.

     .. versionadded:: 2.0

    """
    return coercions.expect(
        roles.LiteralValueRole,
        value,
        type_=type_,
        literal_execute=literal_execute,
    )


def literal_column(
    text: str, type_: Optional[_TypeEngineArgument[_T]] = None
) -> ColumnClause[_T]:
    r"""Produce a :class:`.ColumnClause` object that has the
    :paramref:`_expression.column.is_literal` flag set to True.

    :func:`_expression.literal_column` is similar to
    :func:`_expression.column`, except that
    it is more often used as a "standalone" column expression that renders
    exactly as stated; while :func:`_expression.column`
    stores a string name that
    will be assumed to be part of a table and may be quoted as such,
    :func:`_expression.literal_column` can be that,
    or any other arbitrary column-oriented
    expression.

    :param text: the text of the expression; can be any SQL expression.
      Quoting rules will not be applied. To specify a column-name expression
      which should be subject to quoting rules, use the :func:`column`
      function.

    :param type\_: an optional :class:`~sqlalchemy.types.TypeEngine`
      object which will
      provide result-set translation and additional expression semantics for
      this column. If left as ``None`` the type will be :class:`.NullType`.

    .. seealso::

        :func:`_expression.column`

        :func:`_expression.text`

        :ref:`tutorial_select_arbitrary_text`

    """
    return ColumnClause(text, type_=type_, is_literal=True)


class CompilerElement(Visitable):
    """base class for SQL elements that can be compiled to produce a
    SQL string.

    .. versionadded:: 2.0

    """

    __slots__ = ()
    __visit_name__ = "compiler_element"

    supports_execution = False

    stringify_dialect = "default"

    @util.preload_module("sqlalchemy.engine.default")
    @util.preload_module("sqlalchemy.engine.url")
    def compile(
        self,
        bind: Optional[_HasDialect] = None,
        dialect: Optional[Dialect] = None,
        **kw: Any,
    ) -> Compiled:
        """Compile this SQL expression.

        The return value is a :class:`~.Compiled` object.
        Calling ``str()`` or ``unicode()`` on the returned value will yield a
        string representation of the result. The
        :class:`~.Compiled` object also can return a
        dictionary of bind parameter names and values
        using the ``params`` accessor.

        :param bind: An :class:`.Connection` or :class:`.Engine` which
           can provide a :class:`.Dialect` in order to generate a
           :class:`.Compiled` object.  If the ``bind`` and
           ``dialect`` parameters are both omitted, a default SQL compiler
           is used.

        :param column_keys: Used for INSERT and UPDATE statements, a list of
            column names which should be present in the VALUES clause of the
            compiled statement. If ``None``, all columns from the target table
            object are rendered.

        :param dialect: A :class:`.Dialect` instance which can generate
            a :class:`.Compiled` object.  This argument takes precedence over
            the ``bind`` argument.

        :param compile_kwargs: optional dictionary of additional parameters
            that will be passed through to the compiler within all "visit"
            methods.  This allows any custom flag to be passed through to
            a custom compilation construct, for example.  It is also used
            for the case of passing the ``literal_binds`` flag through::

                from sqlalchemy.sql import table, column, select

                t = table("t", column("x"))

                s = select(t).where(t.c.x == 5)

                print(s.compile(compile_kwargs={"literal_binds": True}))

        .. seealso::

            :ref:`faq_sql_expression_string`

        """

        if dialect is None:
            if bind:
                dialect = bind.dialect
            elif self.stringify_dialect == "default":
                dialect = self._default_dialect()
            else:
                url = util.preloaded.engine_url
                dialect = url.URL.create(
                    self.stringify_dialect
                ).get_dialect()()

        return self._compiler(dialect, **kw)

    def _default_dialect(self):
        default = util.preloaded.engine_default
        return default.StrCompileDialect()

    def _compiler(self, dialect: Dialect, **kw: Any) -> Compiled:
        """Return a compiler appropriate for this ClauseElement, given a
        Dialect."""

        if TYPE_CHECKING:
            assert isinstance(self, ClauseElement)
        return dialect.statement_compiler(dialect, self, **kw)

    def __str__(self) -> str:
        return str(self.compile())


@inspection._self_inspects
class ClauseElement(
    SupportsWrappingAnnotations,
    MemoizedHasCacheKey,
    HasCopyInternals,
    ExternallyTraversible,
    CompilerElement,
):
    """Base class for elements of a programmatically constructed SQL
    expression.

    """

    __visit_name__ = "clause"

    if TYPE_CHECKING:

        @util.memoized_property
        def _propagate_attrs(self) -> _PropagateAttrsType:
            """like annotations, however these propagate outwards liberally
            as SQL constructs are built, and are set up at construction time.

            """
            ...

    else:
        _propagate_attrs = util.EMPTY_DICT

    @util.ro_memoized_property
    def description(self) -> Optional[str]:
        return None

    _is_clone_of: Optional[Self] = None

    is_clause_element = True
    is_selectable = False
    is_dml = False
    _is_column_element = False
    _is_keyed_column_element = False
    _is_table = False
    _gen_static_annotations_cache_key = False
    _is_textual = False
    _is_from_clause = False
    _is_returns_rows = False
    _is_text_clause = False
    _is_from_container = False
    _is_select_container = False
    _is_select_base = False
    _is_select_statement = False
    _is_bind_parameter = False
    _is_clause_list = False
    _is_lambda_element = False
    _is_singleton_constant = False
    _is_immutable = False
    _is_star = False

    @property
    def _order_by_label_element(self) -> Optional[Label[Any]]:
        return None

    _cache_key_traversal: _CacheKeyTraversalType = None

    negation_clause: ColumnElement[bool]

    if typing.TYPE_CHECKING:

        def get_children(
            self, *, omit_attrs: typing_Tuple[str, ...] = ..., **kw: Any
        ) -> Iterable[ClauseElement]: ...

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return []

    def _set_propagate_attrs(self, values: Mapping[str, Any]) -> Self:
        # usually, self._propagate_attrs is empty here.  one case where it's
        # not is a subquery against ORM select, that is then pulled as a
        # property of an aliased class.   should all be good

        # assert not self._propagate_attrs

        self._propagate_attrs = util.immutabledict(values)
        return self

    def _default_compiler(self) -> SQLCompiler:
        dialect = self._default_dialect()
        return dialect.statement_compiler(dialect, self)  # type: ignore

    def _clone(self, **kw: Any) -> Self:
        """Create a shallow copy of this ClauseElement.

        This method may be used by a generative API.  Its also used as
        part of the "deep" copy afforded by a traversal that combines
        the _copy_internals() method.

        """

        skip = self._memoized_keys
        c = self.__class__.__new__(self.__class__)

        if skip:
            # ensure this iteration remains atomic
            c.__dict__ = {
                k: v for k, v in self.__dict__.copy().items() if k not in skip
            }
        else:
            c.__dict__ = self.__dict__.copy()

        # this is a marker that helps to "equate" clauses to each other
        # when a Select returns its list of FROM clauses.  the cloning
        # process leaves around a lot of remnants of the previous clause
        # typically in the form of column expressions still attached to the
        # old table.
        cc = self._is_clone_of
        c._is_clone_of = cc if cc is not None else self
        return c

    def _negate_in_binary(self, negated_op, original_op):
        """a hook to allow the right side of a binary expression to respond
        to a negation of the binary expression.

        Used for the special case of expanding bind parameter with IN.

        """
        return self

    def _with_binary_element_type(self, type_):
        """in the context of binary expression, convert the type of this
        object to the one given.

        applies only to :class:`_expression.ColumnElement` classes.

        """
        return self

    @property
    def _constructor(self):
        """return the 'constructor' for this ClauseElement.

        This is for the purposes for creating a new object of
        this type.   Usually, its just the element's __class__.
        However, the "Annotated" version of the object overrides
        to return the class of its proxied element.

        """
        return self.__class__

    @HasMemoized.memoized_attribute
    def _cloned_set(self):
        """Return the set consisting all cloned ancestors of this
        ClauseElement.

        Includes this ClauseElement.  This accessor tends to be used for
        FromClause objects to identify 'equivalent' FROM clauses, regardless
        of transformative operations.

        """
        s = util.column_set()
        f: Optional[ClauseElement] = self

        # note this creates a cycle, asserted in test_memusage. however,
        # turning this into a plain @property adds tends of thousands of method
        # calls to Core / ORM performance tests, so the small overhead
        # introduced by the relatively small amount of short term cycles
        # produced here is preferable
        while f is not None:
            s.add(f)
            f = f._is_clone_of
        return s

    def _de_clone(self):
        while self._is_clone_of is not None:
            self = self._is_clone_of
        return self

    @property
    def entity_namespace(self):
        raise AttributeError(
            "This SQL expression has no entity namespace "
            "with which to filter from."
        )

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_is_clone_of", None)
        d.pop("_generate_cache_key", None)
        return d

    def _execute_on_connection(
        self,
        connection: Connection,
        distilled_params: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter,
    ) -> Result[Any]:
        if self.supports_execution:
            if TYPE_CHECKING:
                assert isinstance(self, Executable)
            return connection._execute_clauseelement(
                self, distilled_params, execution_options
            )
        else:
            raise exc.ObjectNotExecutableError(self)

    def _execute_on_scalar(
        self,
        connection: Connection,
        distilled_params: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter,
    ) -> Any:
        """an additional hook for subclasses to provide a different
        implementation for connection.scalar() vs. connection.execute().

        .. versionadded:: 2.0

        """
        return self._execute_on_connection(
            connection, distilled_params, execution_options
        ).scalar()

    def _get_embedded_bindparams(self) -> Sequence[BindParameter[Any]]:
        """Return the list of :class:`.BindParameter` objects embedded in the
        object.

        This accomplishes the same purpose as ``visitors.traverse()`` or
        similar would provide, however by making use of the cache key
        it takes advantage of memoization of the key to result in fewer
        net method calls, assuming the statement is also going to be
        executed.

        """

        key = self._generate_cache_key()
        if key is None:
            bindparams: List[BindParameter[Any]] = []

            traverse(self, {}, {"bindparam": bindparams.append})
            return bindparams

        else:
            return key.bindparams

    def unique_params(
        self,
        __optionaldict: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Self:
        """Return a copy with :func:`_expression.bindparam` elements
        replaced.

        Same functionality as :meth:`_expression.ClauseElement.params`,
        except adds `unique=True`
        to affected bind parameters so that multiple statements can be
        used.

        """
        return self._replace_params(True, __optionaldict, kwargs)

    def params(
        self,
        __optionaldict: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Self:
        """Return a copy with :func:`_expression.bindparam` elements
        replaced.

        Returns a copy of this ClauseElement with
        :func:`_expression.bindparam`
        elements replaced with values taken from the given dictionary::

          >>> clause = column("x") + bindparam("foo")
          >>> print(clause.compile().params)
          {'foo':None}
          >>> print(clause.params({"foo": 7}).compile().params)
          {'foo':7}

        """
        return self._replace_params(False, __optionaldict, kwargs)

    def _replace_params(
        self,
        unique: bool,
        optionaldict: Optional[Mapping[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Self:
        if optionaldict:
            kwargs.update(optionaldict)

        def visit_bindparam(bind: BindParameter[Any]) -> None:
            if bind.key in kwargs:
                bind.value = kwargs[bind.key]
                bind.required = False
            if unique:
                bind._convert_to_unique()

        return cloned_traverse(
            self,
            {"maintain_key": True, "detect_subquery_cols": True},
            {"bindparam": visit_bindparam},
        )

    def compare(self, other: ClauseElement, **kw: Any) -> bool:
        r"""Compare this :class:`_expression.ClauseElement` to
        the given :class:`_expression.ClauseElement`.

        Subclasses should override the default behavior, which is a
        straight identity comparison.

        \**kw are arguments consumed by subclass ``compare()`` methods and
        may be used to modify the criteria for comparison
        (see :class:`_expression.ColumnElement`).

        """
        return traversals.compare(self, other, **kw)

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> ClauseElement:
        """Apply a 'grouping' to this :class:`_expression.ClauseElement`.

        This method is overridden by subclasses to return a "grouping"
        construct, i.e. parenthesis.   In particular it's used by "binary"
        expressions to provide a grouping around themselves when placed into a
        larger expression, as well as by :func:`_expression.select`
        constructs when placed into the FROM clause of another
        :func:`_expression.select`.  (Note that subqueries should be
        normally created using the :meth:`_expression.Select.alias` method,
        as many
        platforms require nested SELECT statements to be named).

        As expressions are composed together, the application of
        :meth:`self_group` is automatic - end-user code should never
        need to use this method directly.  Note that SQLAlchemy's
        clause constructs take operator precedence into account -
        so parenthesis might not be needed, for example, in
        an expression like ``x OR (y AND z)`` - AND takes precedence
        over OR.

        The base :meth:`self_group` method of
        :class:`_expression.ClauseElement`
        just returns self.
        """
        return self

    def _ungroup(self) -> ClauseElement:
        """Return this :class:`_expression.ClauseElement`
        without any groupings.
        """

        return self

    def _compile_w_cache(
        self,
        dialect: Dialect,
        *,
        compiled_cache: Optional[CompiledCacheType],
        column_keys: List[str],
        for_executemany: bool = False,
        schema_translate_map: Optional[SchemaTranslateMapType] = None,
        **kw: Any,
    ) -> typing_Tuple[
        Compiled, Optional[Sequence[BindParameter[Any]]], CacheStats
    ]:
        elem_cache_key: Optional[CacheKey]

        if compiled_cache is not None and dialect._supports_statement_cache:
            elem_cache_key = self._generate_cache_key()
        else:
            elem_cache_key = None

        if elem_cache_key is not None:
            if TYPE_CHECKING:
                assert compiled_cache is not None

            cache_key, extracted_params = elem_cache_key
            key = (
                dialect,
                cache_key,
                tuple(column_keys),
                bool(schema_translate_map),
                for_executemany,
            )
            compiled_sql = compiled_cache.get(key)

            if compiled_sql is None:
                cache_hit = dialect.CACHE_MISS
                compiled_sql = self._compiler(
                    dialect,
                    cache_key=elem_cache_key,
                    column_keys=column_keys,
                    for_executemany=for_executemany,
                    schema_translate_map=schema_translate_map,
                    **kw,
                )
                compiled_cache[key] = compiled_sql
            else:
                cache_hit = dialect.CACHE_HIT
        else:
            extracted_params = None
            compiled_sql = self._compiler(
                dialect,
                cache_key=elem_cache_key,
                column_keys=column_keys,
                for_executemany=for_executemany,
                schema_translate_map=schema_translate_map,
                **kw,
            )

            if not dialect._supports_statement_cache:
                cache_hit = dialect.NO_DIALECT_SUPPORT
            elif compiled_cache is None:
                cache_hit = dialect.CACHING_DISABLED
            else:
                cache_hit = dialect.NO_CACHE_KEY

        return compiled_sql, extracted_params, cache_hit

    def __invert__(self):
        # undocumented element currently used by the ORM for
        # relationship.contains()
        if hasattr(self, "negation_clause"):
            return self.negation_clause
        else:
            return self._negate()

    def _negate(self) -> ClauseElement:
        grouped = self.self_group(against=operators.inv)
        assert isinstance(grouped, ColumnElement)
        return UnaryExpression(grouped, operator=operators.inv)

    def __bool__(self):
        raise TypeError("Boolean value of this clause is not defined")

    def __repr__(self):
        friendly = self.description
        if friendly is None:
            return object.__repr__(self)
        else:
            return "<%s.%s at 0x%x; %s>" % (
                self.__module__,
                self.__class__.__name__,
                id(self),
                friendly,
            )


class DQLDMLClauseElement(ClauseElement):
    """represents a :class:`.ClauseElement` that compiles to a DQL or DML
    expression, not DDL.

    .. versionadded:: 2.0

    """

    if typing.TYPE_CHECKING:

        def _compiler(self, dialect: Dialect, **kw: Any) -> SQLCompiler:
            """Return a compiler appropriate for this ClauseElement, given a
            Dialect."""
            ...

        def compile(  # noqa: A001
            self,
            bind: Optional[_HasDialect] = None,
            dialect: Optional[Dialect] = None,
            **kw: Any,
        ) -> SQLCompiler: ...


class CompilerColumnElement(
    roles.DMLColumnRole,
    roles.DDLConstraintColumnRole,
    roles.ColumnsClauseRole,
    CompilerElement,
):
    """A compiler-only column element used for ad-hoc string compilations.

    .. versionadded:: 2.0

    """

    __slots__ = ()

    _propagate_attrs = util.EMPTY_DICT
    _is_collection_aggregate = False


# SQLCoreOperations should be suiting the ExpressionElementRole
# and ColumnsClauseRole.   however the MRO issues become too elaborate
# at the moment.
class SQLCoreOperations(Generic[_T_co], ColumnOperators, TypingOnly):
    __slots__ = ()

    # annotations for comparison methods
    # these are from operators->Operators / ColumnOperators,
    # redefined with the specific types returned by ColumnElement hierarchies
    if typing.TYPE_CHECKING:

        @util.non_memoized_property
        def _propagate_attrs(self) -> _PropagateAttrsType: ...

        def operate(
            self, op: OperatorType, *other: Any, **kwargs: Any
        ) -> ColumnElement[Any]: ...

        def reverse_operate(
            self, op: OperatorType, other: Any, **kwargs: Any
        ) -> ColumnElement[Any]: ...

        @overload
        def op(
            self,
            opstring: str,
            precedence: int = ...,
            is_comparison: bool = ...,
            *,
            return_type: _TypeEngineArgument[_OPT],
            python_impl: Optional[Callable[..., Any]] = None,
        ) -> Callable[[Any], BinaryExpression[_OPT]]: ...

        @overload
        def op(
            self,
            opstring: str,
            precedence: int = ...,
            is_comparison: bool = ...,
            return_type: Optional[_TypeEngineArgument[Any]] = ...,
            python_impl: Optional[Callable[..., Any]] = ...,
        ) -> Callable[[Any], BinaryExpression[Any]]: ...

        def op(
            self,
            opstring: str,
            precedence: int = 0,
            is_comparison: bool = False,
            return_type: Optional[_TypeEngineArgument[Any]] = None,
            python_impl: Optional[Callable[..., Any]] = None,
        ) -> Callable[[Any], BinaryExpression[Any]]: ...

        def bool_op(
            self,
            opstring: str,
            precedence: int = 0,
            python_impl: Optional[Callable[..., Any]] = None,
        ) -> Callable[[Any], BinaryExpression[bool]]: ...

        def __and__(self, other: Any) -> BooleanClauseList: ...

        def __or__(self, other: Any) -> BooleanClauseList: ...

        def __invert__(self) -> ColumnElement[_T_co]: ...

        def __lt__(self, other: Any) -> ColumnElement[bool]: ...

        def __le__(self, other: Any) -> ColumnElement[bool]: ...

        # declare also that this class has an hash method otherwise
        # it may be assumed to be None by type checkers since the
        # object defines __eq__ and python sets it to None in that case:
        # https://docs.python.org/3/reference/datamodel.html#object.__hash__
        def __hash__(self) -> int: ...

        def __eq__(self, other: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
            ...

        def __ne__(self, other: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
            ...

        def is_distinct_from(self, other: Any) -> ColumnElement[bool]: ...

        def is_not_distinct_from(self, other: Any) -> ColumnElement[bool]: ...

        def __gt__(self, other: Any) -> ColumnElement[bool]: ...

        def __ge__(self, other: Any) -> ColumnElement[bool]: ...

        def __neg__(self) -> UnaryExpression[_T_co]: ...

        def __contains__(self, other: Any) -> ColumnElement[bool]: ...

        def __getitem__(self, index: Any) -> ColumnElement[Any]: ...

        @overload
        def __lshift__(self: _SQO[int], other: Any) -> ColumnElement[int]: ...

        @overload
        def __lshift__(self, other: Any) -> ColumnElement[Any]: ...

        def __lshift__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __rshift__(self: _SQO[int], other: Any) -> ColumnElement[int]: ...

        @overload
        def __rshift__(self, other: Any) -> ColumnElement[Any]: ...

        def __rshift__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def concat(self: _SQO[str], other: Any) -> ColumnElement[str]: ...

        @overload
        def concat(self, other: Any) -> ColumnElement[Any]: ...

        def concat(self, other: Any) -> ColumnElement[Any]: ...

        def like(
            self, other: Any, escape: Optional[str] = None
        ) -> BinaryExpression[bool]: ...

        def ilike(
            self, other: Any, escape: Optional[str] = None
        ) -> BinaryExpression[bool]: ...

        def bitwise_xor(self, other: Any) -> BinaryExpression[Any]: ...

        def bitwise_or(self, other: Any) -> BinaryExpression[Any]: ...

        def bitwise_and(self, other: Any) -> BinaryExpression[Any]: ...

        def bitwise_not(self) -> UnaryExpression[_T_co]: ...

        def bitwise_lshift(self, other: Any) -> BinaryExpression[Any]: ...

        def bitwise_rshift(self, other: Any) -> BinaryExpression[Any]: ...

        def in_(
            self,
            other: Union[
                Iterable[Any], BindParameter[Any], roles.InElementRole
            ],
        ) -> BinaryExpression[bool]: ...

        def not_in(
            self,
            other: Union[
                Iterable[Any], BindParameter[Any], roles.InElementRole
            ],
        ) -> BinaryExpression[bool]: ...

        def notin_(
            self,
            other: Union[
                Iterable[Any], BindParameter[Any], roles.InElementRole
            ],
        ) -> BinaryExpression[bool]: ...

        def not_like(
            self, other: Any, escape: Optional[str] = None
        ) -> BinaryExpression[bool]: ...

        def notlike(
            self, other: Any, escape: Optional[str] = None
        ) -> BinaryExpression[bool]: ...

        def not_ilike(
            self, other: Any, escape: Optional[str] = None
        ) -> BinaryExpression[bool]: ...

        def notilike(
            self, other: Any, escape: Optional[str] = None
        ) -> BinaryExpression[bool]: ...

        def is_(self, other: Any) -> BinaryExpression[bool]: ...

        def is_not(self, other: Any) -> BinaryExpression[bool]: ...

        def isnot(self, other: Any) -> BinaryExpression[bool]: ...

        def startswith(
            self,
            other: Any,
            escape: Optional[str] = None,
            autoescape: bool = False,
        ) -> ColumnElement[bool]: ...

        def istartswith(
            self,
            other: Any,
            escape: Optional[str] = None,
            autoescape: bool = False,
        ) -> ColumnElement[bool]: ...

        def endswith(
            self,
            other: Any,
            escape: Optional[str] = None,
            autoescape: bool = False,
        ) -> ColumnElement[bool]: ...

        def iendswith(
            self,
            other: Any,
            escape: Optional[str] = None,
            autoescape: bool = False,
        ) -> ColumnElement[bool]: ...

        def contains(self, other: Any, **kw: Any) -> ColumnElement[bool]: ...

        def icontains(self, other: Any, **kw: Any) -> ColumnElement[bool]: ...

        def match(self, other: Any, **kwargs: Any) -> ColumnElement[bool]: ...

        def regexp_match(
            self, pattern: Any, flags: Optional[str] = None
        ) -> ColumnElement[bool]: ...

        def regexp_replace(
            self, pattern: Any, replacement: Any, flags: Optional[str] = None
        ) -> ColumnElement[str]: ...

        def desc(self) -> UnaryExpression[_T_co]: ...

        def asc(self) -> UnaryExpression[_T_co]: ...

        def nulls_first(self) -> UnaryExpression[_T_co]: ...

        def nullsfirst(self) -> UnaryExpression[_T_co]: ...

        def nulls_last(self) -> UnaryExpression[_T_co]: ...

        def nullslast(self) -> UnaryExpression[_T_co]: ...

        def collate(self, collation: str) -> CollationClause: ...

        def between(
            self, cleft: Any, cright: Any, symmetric: bool = False
        ) -> BinaryExpression[bool]: ...

        def distinct(self: _SQO[_T_co]) -> UnaryExpression[_T_co]: ...

        def any_(self) -> CollectionAggregate[Any]: ...

        def all_(self) -> CollectionAggregate[Any]: ...

        # numeric overloads.  These need more tweaking
        # in particular they all need to have a variant for Optiona[_T]
        # because Optional only applies to the data side, not the expression
        # side

        @overload
        def __add__(
            self: _SQO[_NMT],
            other: Any,
        ) -> ColumnElement[_NMT]: ...

        @overload
        def __add__(
            self: _SQO[str],
            other: Any,
        ) -> ColumnElement[str]: ...

        @overload
        def __add__(self, other: Any) -> ColumnElement[Any]: ...

        def __add__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __radd__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]: ...

        @overload
        def __radd__(self: _SQO[str], other: Any) -> ColumnElement[str]: ...

        def __radd__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __sub__(
            self: _SQO[_NMT],
            other: Any,
        ) -> ColumnElement[_NMT]: ...

        @overload
        def __sub__(self, other: Any) -> ColumnElement[Any]: ...

        def __sub__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __rsub__(
            self: _SQO[_NMT],
            other: Any,
        ) -> ColumnElement[_NMT]: ...

        @overload
        def __rsub__(self, other: Any) -> ColumnElement[Any]: ...

        def __rsub__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __mul__(
            self: _SQO[_NMT],
            other: Any,
        ) -> ColumnElement[_NMT]: ...

        @overload
        def __mul__(self, other: Any) -> ColumnElement[Any]: ...

        def __mul__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __rmul__(
            self: _SQO[_NMT],
            other: Any,
        ) -> ColumnElement[_NMT]: ...

        @overload
        def __rmul__(self, other: Any) -> ColumnElement[Any]: ...

        def __rmul__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __mod__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]: ...

        @overload
        def __mod__(self, other: Any) -> ColumnElement[Any]: ...

        def __mod__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __rmod__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]: ...

        @overload
        def __rmod__(self, other: Any) -> ColumnElement[Any]: ...

        def __rmod__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __truediv__(
            self: _SQO[int], other: Any
        ) -> ColumnElement[_NUMERIC]: ...

        @overload
        def __truediv__(self: _SQO[_NT], other: Any) -> ColumnElement[_NT]: ...

        @overload
        def __truediv__(self, other: Any) -> ColumnElement[Any]: ...

        def __truediv__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __rtruediv__(
            self: _SQO[_NMT], other: Any
        ) -> ColumnElement[_NUMERIC]: ...

        @overload
        def __rtruediv__(self, other: Any) -> ColumnElement[Any]: ...

        def __rtruediv__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __floordiv__(
            self: _SQO[_NMT], other: Any
        ) -> ColumnElement[_NMT]: ...

        @overload
        def __floordiv__(self, other: Any) -> ColumnElement[Any]: ...

        def __floordiv__(self, other: Any) -> ColumnElement[Any]: ...

        @overload
        def __rfloordiv__(
            self: _SQO[_NMT], other: Any
        ) -> ColumnElement[_NMT]: ...

        @overload
        def __rfloordiv__(self, other: Any) -> ColumnElement[Any]: ...

        def __rfloordiv__(self, other: Any) -> ColumnElement[Any]: ...


class SQLColumnExpression(
    SQLCoreOperations[_T_co], roles.ExpressionElementRole[_T_co], TypingOnly
):
    """A type that may be used to indicate any SQL column element or object
    that acts in place of one.

    :class:`.SQLColumnExpression` is a base of
    :class:`.ColumnElement`, as well as within the bases of ORM elements
    such as :class:`.InstrumentedAttribute`, and may be used in :pep:`484`
    typing to indicate arguments or return values that should behave
    as column expressions.

    .. versionadded:: 2.0.0b4


    """

    __slots__ = ()


_SQO = SQLCoreOperations


class ColumnElement(
    roles.ColumnArgumentOrKeyRole,
    roles.StatementOptionRole,
    roles.WhereHavingRole,
    roles.BinaryElementRole[_T],
    roles.OrderByRole,
    roles.ColumnsClauseRole,
    roles.LimitOffsetRole,
    roles.DMLColumnRole,
    roles.DDLConstraintColumnRole,
    roles.DDLExpressionRole,
    SQLColumnExpression[_T],
    DQLDMLClauseElement,
):
    """Represent a column-oriented SQL expression suitable for usage in the
    "columns" clause, WHERE clause etc. of a statement.

    While the most familiar kind of :class:`_expression.ColumnElement` is the
    :class:`_schema.Column` object, :class:`_expression.ColumnElement`
    serves as the basis
    for any unit that may be present in a SQL expression, including
    the expressions themselves, SQL functions, bound parameters,
    literal expressions, keywords such as ``NULL``, etc.
    :class:`_expression.ColumnElement`
    is the ultimate base class for all such elements.

    A wide variety of SQLAlchemy Core functions work at the SQL expression
    level, and are intended to accept instances of
    :class:`_expression.ColumnElement` as
    arguments.  These functions will typically document that they accept a
    "SQL expression" as an argument.  What this means in terms of SQLAlchemy
    usually refers to an input which is either already in the form of a
    :class:`_expression.ColumnElement` object,
    or a value which can be **coerced** into
    one.  The coercion rules followed by most, but not all, SQLAlchemy Core
    functions with regards to SQL expressions are as follows:

        * a literal Python value, such as a string, integer or floating
          point value, boolean, datetime, ``Decimal`` object, or virtually
          any other Python object, will be coerced into a "literal bound
          value".  This generally means that a :func:`.bindparam` will be
          produced featuring the given value embedded into the construct; the
          resulting :class:`.BindParameter` object is an instance of
          :class:`_expression.ColumnElement`.
          The Python value will ultimately be sent
          to the DBAPI at execution time as a parameterized argument to the
          ``execute()`` or ``executemany()`` methods, after SQLAlchemy
          type-specific converters (e.g. those provided by any associated
          :class:`.TypeEngine` objects) are applied to the value.

        * any special object value, typically ORM-level constructs, which
          feature an accessor called ``__clause_element__()``.  The Core
          expression system looks for this method when an object of otherwise
          unknown type is passed to a function that is looking to coerce the
          argument into a :class:`_expression.ColumnElement` and sometimes a
          :class:`_expression.SelectBase` expression.
          It is used within the ORM to
          convert from ORM-specific objects like mapped classes and
          mapped attributes into Core expression objects.

        * The Python ``None`` value is typically interpreted as ``NULL``,
          which in SQLAlchemy Core produces an instance of :func:`.null`.

    A :class:`_expression.ColumnElement` provides the ability to generate new
    :class:`_expression.ColumnElement`
    objects using Python expressions.  This means that Python operators
    such as ``==``, ``!=`` and ``<`` are overloaded to mimic SQL operations,
    and allow the instantiation of further :class:`_expression.ColumnElement`
    instances
    which are composed from other, more fundamental
    :class:`_expression.ColumnElement`
    objects.  For example, two :class:`.ColumnClause` objects can be added
    together with the addition operator ``+`` to produce
    a :class:`.BinaryExpression`.
    Both :class:`.ColumnClause` and :class:`.BinaryExpression` are subclasses
    of :class:`_expression.ColumnElement`:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy.sql import column
        >>> column("a") + column("b")
        <sqlalchemy.sql.expression.BinaryExpression object at 0x101029dd0>
        >>> print(column("a") + column("b"))
        {printsql}a + b

    .. seealso::

        :class:`_schema.Column`

        :func:`_expression.column`

    """

    __visit_name__ = "column_element"

    primary_key: bool = False
    _is_clone_of: Optional[ColumnElement[_T]]
    _is_column_element = True
    _insert_sentinel: bool = False
    _omit_from_statements = False
    _is_collection_aggregate = False

    foreign_keys: AbstractSet[ForeignKey] = frozenset()

    @util.memoized_property
    def _proxies(self) -> List[ColumnElement[Any]]:
        return []

    @util.non_memoized_property
    def _tq_label(self) -> Optional[str]:
        """The named label that can be used to target
        this column in a result set in a "table qualified" context.

        This label is almost always the label used when
        rendering <expr> AS <label> in a SELECT statement when using
        the LABEL_STYLE_TABLENAME_PLUS_COL label style, which is what the
        legacy ORM ``Query`` object uses as well.

        For a regular Column bound to a Table, this is typically the label
        <tablename>_<columnname>.  For other constructs, different rules
        may apply, such as anonymized labels and others.

        .. versionchanged:: 1.4.21 renamed from ``._label``

        """
        return None

    key: Optional[str] = None
    """The 'key' that in some circumstances refers to this object in a
    Python namespace.

    This typically refers to the "key" of the column as present in the
    ``.c`` collection of a selectable, e.g. ``sometable.c["somekey"]`` would
    return a :class:`_schema.Column` with a ``.key`` of "somekey".

    """

    @HasMemoized.memoized_attribute
    def _tq_key_label(self) -> Optional[str]:
        """A label-based version of 'key' that in some circumstances refers
        to this object in a Python namespace.


        _tq_key_label comes into play when a select() statement is constructed
        with apply_labels(); in this case, all Column objects in the ``.c``
        collection are rendered as <tablename>_<columnname> in SQL; this is
        essentially the value of ._label. But to locate those columns in the
        ``.c`` collection, the name is along the lines of <tablename>_<key>;
        that's the typical value of .key_label.

        .. versionchanged:: 1.4.21 renamed from ``._key_label``

        """
        return self._proxy_key

    @property
    def _key_label(self) -> Optional[str]:
        """legacy; renamed to _tq_key_label"""
        return self._tq_key_label

    @property
    def _label(self) -> Optional[str]:
        """legacy; renamed to _tq_label"""
        return self._tq_label

    @property
    def _non_anon_label(self) -> Optional[str]:
        """the 'name' that naturally applies this element when rendered in
        SQL.

        Concretely, this is the "name" of a column or a label in a
        SELECT statement; ``<columnname>`` and ``<labelname>`` below:

        .. sourcecode:: sql

            SELECT <columnmame> FROM table

            SELECT column AS <labelname> FROM table

        Above, the two names noted will be what's present in the DBAPI
        ``cursor.description`` as the names.

        If this attribute returns ``None``, it means that the SQL element as
        written does not have a 100% fully predictable "name" that would appear
        in the ``cursor.description``. Examples include SQL functions, CAST
        functions, etc. While such things do return names in
        ``cursor.description``, they are only predictable on a
        database-specific basis; e.g. an expression like ``MAX(table.col)`` may
        appear as the string ``max`` on one database (like PostgreSQL) or may
        appear as the whole expression ``max(table.col)`` on SQLite.

        The default implementation looks for a ``.name`` attribute on the
        object, as has been the precedent established in SQLAlchemy for many
        years.  An exception is made on the ``FunctionElement`` subclass
        so that the return value is always ``None``.

        .. versionadded:: 1.4.21



        """
        return getattr(self, "name", None)

    _render_label_in_columns_clause = True
    """A flag used by select._columns_plus_names that helps to determine
    we are actually going to render in terms of "SELECT <col> AS <label>".
    This flag can be returned as False for some Column objects that want
    to be rendered as simple "SELECT <col>"; typically columns that don't have
    any parent table and are named the same as what the label would be
    in any case.

    """

    _allow_label_resolve = True
    """A flag that can be flipped to prevent a column from being resolvable
    by string label name.

    The joined eager loader strategy in the ORM uses this, for example.

    """

    _is_implicitly_boolean = False

    _alt_names: Sequence[str] = ()

    @overload
    def self_group(self, against: None = None) -> ColumnElement[_T]: ...

    @overload
    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> ColumnElement[Any]: ...

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> ColumnElement[Any]:
        if (
            against in (operators.and_, operators.or_, operators._asbool)
            and self.type._type_affinity is type_api.BOOLEANTYPE._type_affinity
        ):
            return AsBoolean(self, operators.is_true, operators.is_false)
        elif against in (operators.any_op, operators.all_op):
            return Grouping(self)
        else:
            return self

    @overload
    def _negate(self: ColumnElement[bool]) -> ColumnElement[bool]: ...

    @overload
    def _negate(self: ColumnElement[_T]) -> ColumnElement[_T]: ...

    def _negate(self) -> ColumnElement[Any]:
        if self.type._type_affinity is type_api.BOOLEANTYPE._type_affinity:
            return AsBoolean(self, operators.is_false, operators.is_true)
        else:
            grouped = self.self_group(against=operators.inv)
            assert isinstance(grouped, ColumnElement)
            return UnaryExpression(
                grouped, operator=operators.inv, wraps_column_expression=True
            )

    type: TypeEngine[_T]

    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:  # noqa: A001
            # used for delayed setup of
            # type_api
            return type_api.NULLTYPE

    @HasMemoized.memoized_attribute
    def comparator(self) -> TypeEngine.Comparator[_T]:
        try:
            comparator_factory = self.type.comparator_factory
        except AttributeError as err:
            raise TypeError(
                "Object %r associated with '.type' attribute "
                "is not a TypeEngine class or object" % self.type
            ) from err
        else:
            return comparator_factory(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, key: str) -> Any:
        try:
            return getattr(self.comparator, key)
        except AttributeError as err:
            raise AttributeError(
                "Neither %r object nor %r object has an attribute %r"
                % (
                    type(self).__name__,
                    type(self.comparator).__name__,
                    key,
                )
            ) from err

    def operate(
        self,
        op: operators.OperatorType,
        *other: Any,
        **kwargs: Any,
    ) -> ColumnElement[Any]:
        return op(self.comparator, *other, **kwargs)  # type: ignore[no-any-return]  # noqa: E501

    def reverse_operate(
        self, op: operators.OperatorType, other: Any, **kwargs: Any
    ) -> ColumnElement[Any]:
        return op(other, self.comparator, **kwargs)  # type: ignore[no-any-return]  # noqa: E501

    def _bind_param(
        self,
        operator: operators.OperatorType,
        obj: Any,
        type_: Optional[TypeEngine[_T]] = None,
        expanding: bool = False,
    ) -> BindParameter[_T]:
        return BindParameter(
            None,
            obj,
            _compared_to_operator=operator,
            type_=type_,
            _compared_to_type=self.type,
            unique=True,
            expanding=expanding,
        )

    @property
    def expression(self) -> ColumnElement[Any]:
        """Return a column expression.

        Part of the inspection interface; returns self.

        """
        return self

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)

    @util.memoized_property
    def base_columns(self) -> FrozenSet[ColumnElement[Any]]:
        return frozenset(c for c in self.proxy_set if not c._proxies)

    @util.memoized_property
    def proxy_set(self) -> FrozenSet[ColumnElement[Any]]:
        """set of all columns we are proxying

        as of 2.0 this is explicitly deannotated columns.  previously it was
        effectively deannotated columns but wasn't enforced.  annotated
        columns should basically not go into sets if at all possible because
        their hashing behavior is very non-performant.

        """
        return frozenset([self._deannotate()]).union(
            itertools.chain(*[c.proxy_set for c in self._proxies])
        )

    @util.memoized_property
    def _expanded_proxy_set(self) -> FrozenSet[ColumnElement[Any]]:
        return frozenset(_expand_cloned(self.proxy_set))

    def _uncached_proxy_list(self) -> List[ColumnElement[Any]]:
        """An 'uncached' version of proxy set.

        This list includes annotated columns which perform very poorly in
        set operations.

        """

        return [self] + list(
            itertools.chain(*[c._uncached_proxy_list() for c in self._proxies])
        )

    def shares_lineage(self, othercolumn: ColumnElement[Any]) -> bool:
        """Return True if the given :class:`_expression.ColumnElement`
        has a common ancestor to this :class:`_expression.ColumnElement`."""

        return bool(self.proxy_set.intersection(othercolumn.proxy_set))

    def _compare_name_for_result(self, other: ColumnElement[Any]) -> bool:
        """Return True if the given column element compares to this one
        when targeting within a result row."""

        return (
            hasattr(other, "name")
            and hasattr(self, "name")
            and other.name == self.name
        )

    @HasMemoized.memoized_attribute
    def _proxy_key(self) -> Optional[str]:
        if self._annotations and "proxy_key" in self._annotations:
            return cast(str, self._annotations["proxy_key"])

        name = self.key
        if not name:
            # there's a bit of a seeming contradiction which is that the
            # "_non_anon_label" of a column can in fact be an
            # "_anonymous_label"; this is when it's on a column that is
            # proxying for an anonymous expression in a subquery.
            name = self._non_anon_label

        if isinstance(name, _anonymous_label):
            return None
        else:
            return name

    @HasMemoized.memoized_attribute
    def _expression_label(self) -> Optional[str]:
        """a suggested label to use in the case that the column has no name,
        which should be used if possible as the explicit 'AS <label>'
        where this expression would normally have an anon label.

        this is essentially mostly what _proxy_key does except it returns
        None if the column has a normal name that can be used.

        """

        if getattr(self, "name", None) is not None:
            return None
        elif self._annotations and "proxy_key" in self._annotations:
            return cast(str, self._annotations["proxy_key"])
        else:
            return None

    def _make_proxy(
        self,
        selectable: FromClause,
        *,
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        name: Optional[str] = None,
        key: Optional[str] = None,
        name_is_truncatable: bool = False,
        compound_select_cols: Optional[Sequence[ColumnElement[Any]]] = None,
        **kw: Any,
    ) -> typing_Tuple[str, ColumnClause[_T]]:
        """Create a new :class:`_expression.ColumnElement` representing this
        :class:`_expression.ColumnElement` as it appears in the select list of
        a descending selectable.

        """
        if name is None:
            name = self._anon_name_label
            if key is None:
                key = self._proxy_key
        else:
            key = name

        assert key is not None

        co: ColumnClause[_T] = ColumnClause(
            (
                coercions.expect(roles.TruncatedLabelRole, name)
                if name_is_truncatable
                else name
            ),
            type_=getattr(self, "type", None),
            _selectable=selectable,
        )

        co._propagate_attrs = selectable._propagate_attrs
        if compound_select_cols:
            co._proxies = list(compound_select_cols)
        else:
            co._proxies = [self]
        if selectable._is_clone_of is not None:
            co._is_clone_of = selectable._is_clone_of.columns.get(key)
        return key, co

    def cast(self, type_: _TypeEngineArgument[_OPT]) -> Cast[_OPT]:
        """Produce a type cast, i.e. ``CAST(<expression> AS <type>)``.

        This is a shortcut to the :func:`_expression.cast` function.

        .. seealso::

            :ref:`tutorial_casts`

            :func:`_expression.cast`

            :func:`_expression.type_coerce`

        """
        return Cast(self, type_)

    def label(self, name: Optional[str]) -> Label[_T]:
        """Produce a column label, i.e. ``<columnname> AS <name>``.

        This is a shortcut to the :func:`_expression.label` function.

        If 'name' is ``None``, an anonymous label name will be generated.

        """
        return Label(name, self, self.type)

    def _anon_label(
        self, seed: Optional[str], add_hash: Optional[int] = None
    ) -> _anonymous_label:
        while self._is_clone_of is not None:
            self = self._is_clone_of

        # as of 1.4 anonymous label for ColumnElement uses hash(), not id(),
        # as the identifier, because a column and its annotated version are
        # the same thing in a SQL statement
        hash_value = hash(self)

        if add_hash:
            # this path is used for disambiguating anon labels that would
            # otherwise be the same name for the same element repeated.
            # an additional numeric value is factored in for each label.

            # shift hash(self) (which is id(self), typically 8 byte integer)
            # 16 bits leftward.  fill extra add_hash on right
            assert add_hash < (2 << 15)
            assert seed
            hash_value = (hash_value << 16) | add_hash

            # extra underscore is added for labels with extra hash
            # values, to isolate the "deduped anon" namespace from the
            # regular namespace.  eliminates chance of these
            # manufactured hash values overlapping with regular ones for some
            # undefined python interpreter
            seed = seed + "_"

        if isinstance(seed, _anonymous_label):
            return _anonymous_label.safe_construct(
                hash_value, "", enclosing_label=seed
            )

        return _anonymous_label.safe_construct(hash_value, seed or "anon")

    @util.memoized_property
    def _anon_name_label(self) -> str:
        """Provides a constant 'anonymous label' for this ColumnElement.

        This is a label() expression which will be named at compile time.
        The same label() is returned each time ``anon_label`` is called so
        that expressions can reference ``anon_label`` multiple times,
        producing the same label name at compile time.

        The compiler uses this function automatically at compile time
        for expressions that are known to be 'unnamed' like binary
        expressions and function calls.

        .. versionchanged:: 1.4.9 - this attribute was not intended to be
           public and is renamed to _anon_name_label.  anon_name exists
           for backwards compat

        """
        name = getattr(self, "name", None)
        return self._anon_label(name)

    @util.memoized_property
    def _anon_key_label(self) -> _anonymous_label:
        """Provides a constant 'anonymous key label' for this ColumnElement.

        Compare to ``anon_label``, except that the "key" of the column,
        if available, is used to generate the label.

        This is used when a deduplicating key is placed into the columns
        collection of a selectable.

        .. versionchanged:: 1.4.9 - this attribute was not intended to be
           public and is renamed to _anon_key_label.  anon_key_label exists
           for backwards compat

        """
        return self._anon_label(self._proxy_key)

    @property
    @util.deprecated(
        "1.4",
        "The :attr:`_expression.ColumnElement.anon_label` attribute is now "
        "private, and the public accessor is deprecated.",
    )
    def anon_label(self) -> str:
        return self._anon_name_label

    @property
    @util.deprecated(
        "1.4",
        "The :attr:`_expression.ColumnElement.anon_key_label` attribute is "
        "now private, and the public accessor is deprecated.",
    )
    def anon_key_label(self) -> str:
        return self._anon_key_label

    def _dedupe_anon_label_idx(self, idx: int) -> str:
        """label to apply to a column that is anon labeled, but repeated
        in the SELECT, so that we have to make an "extra anon" label that
        disambiguates it from the previous appearance.

        these labels come out like "foo_bar_id__1" and have double underscores
        in them.

        """
        label = getattr(self, "name", None)

        # current convention is that if the element doesn't have a
        # ".name" (usually because it is not NamedColumn), we try to
        # use a "table qualified" form for the "dedupe anon" label,
        # based on the notion that a label like
        # "CAST(casttest.v1 AS DECIMAL) AS casttest_v1__1" looks better than
        # "CAST(casttest.v1 AS DECIMAL) AS anon__1"

        if label is None:
            return self._dedupe_anon_tq_label_idx(idx)
        else:
            return self._anon_label(label, add_hash=idx)

    @util.memoized_property
    def _anon_tq_label(self) -> _anonymous_label:
        return self._anon_label(getattr(self, "_tq_label", None))

    @util.memoized_property
    def _anon_tq_key_label(self) -> _anonymous_label:
        return self._anon_label(getattr(self, "_tq_key_label", None))

    def _dedupe_anon_tq_label_idx(self, idx: int) -> _anonymous_label:
        label = getattr(self, "_tq_label", None) or "anon"

        return self._anon_label(label, add_hash=idx)


class KeyedColumnElement(ColumnElement[_T]):
    """ColumnElement where ``.key`` is non-None."""

    _is_keyed_column_element = True

    key: str


class WrapsColumnExpression(ColumnElement[_T]):
    """Mixin that defines a :class:`_expression.ColumnElement`
    as a wrapper with special
    labeling behavior for an expression that already has a name.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`change_4449`


    """

    @property
    def wrapped_column_expression(self) -> ColumnElement[_T]:
        raise NotImplementedError()

    @util.non_memoized_property
    def _tq_label(self) -> Optional[str]:
        wce = self.wrapped_column_expression
        if hasattr(wce, "_tq_label"):
            return wce._tq_label
        else:
            return None

    @property
    def _label(self) -> Optional[str]:
        return self._tq_label

    @property
    def _non_anon_label(self) -> Optional[str]:
        return None

    @util.non_memoized_property
    def _anon_name_label(self) -> str:
        wce = self.wrapped_column_expression

        # this logic tries to get the WrappedColumnExpression to render
        # with "<expr> AS <name>", where "<name>" is the natural name
        # within the expression itself.   e.g. "CAST(table.foo) AS foo".
        if not wce._is_text_clause:
            nal = wce._non_anon_label
            if nal:
                return nal
            elif hasattr(wce, "_anon_name_label"):
                return wce._anon_name_label
        return super()._anon_name_label

    def _dedupe_anon_label_idx(self, idx: int) -> str:
        wce = self.wrapped_column_expression
        nal = wce._non_anon_label
        if nal:
            return self._anon_label(nal + "_")
        else:
            return self._dedupe_anon_tq_label_idx(idx)

    @property
    def _proxy_key(self):
        wce = self.wrapped_column_expression

        if not wce._is_text_clause:
            return wce._proxy_key
        return super()._proxy_key


class BindParameter(roles.InElementRole, KeyedColumnElement[_T]):
    r"""Represent a "bound expression".

    :class:`.BindParameter` is invoked explicitly using the
    :func:`.bindparam` function, as in::

        from sqlalchemy import bindparam

        stmt = select(users_table).where(
            users_table.c.name == bindparam("username")
        )

    Detailed discussion of how :class:`.BindParameter` is used is
    at :func:`.bindparam`.

    .. seealso::

        :func:`.bindparam`

    """

    __visit_name__ = "bindparam"

    _traverse_internals: _TraverseInternalsType = [
        ("key", InternalTraversal.dp_anon_name),
        ("type", InternalTraversal.dp_type),
        ("callable", InternalTraversal.dp_plain_dict),
        ("value", InternalTraversal.dp_plain_obj),
        ("literal_execute", InternalTraversal.dp_boolean),
    ]

    key: str
    type: TypeEngine[_T]
    value: Optional[_T]

    _is_crud = False
    _is_bind_parameter = True
    _key_is_anon = False

    # bindparam implements its own _gen_cache_key() method however
    # we check subclasses for this flag, else no cache key is generated
    inherit_cache = True

    def __init__(
        self,
        key: Optional[str],
        value: Any = _NoArg.NO_ARG,
        type_: Optional[_TypeEngineArgument[_T]] = None,
        unique: bool = False,
        required: Union[bool, Literal[_NoArg.NO_ARG]] = _NoArg.NO_ARG,
        quote: Optional[bool] = None,
        callable_: Optional[Callable[[], Any]] = None,
        expanding: bool = False,
        isoutparam: bool = False,
        literal_execute: bool = False,
        _compared_to_operator: Optional[OperatorType] = None,
        _compared_to_type: Optional[TypeEngine[Any]] = None,
        _is_crud: bool = False,
    ):
        if required is _NoArg.NO_ARG:
            required = value is _NoArg.NO_ARG and callable_ is None
        if value is _NoArg.NO_ARG:
            value = None

        if quote is not None:
            key = quoted_name.construct(key, quote)

        if unique:
            self.key = _anonymous_label.safe_construct(
                id(self),
                (
                    key
                    if key is not None
                    and not isinstance(key, _anonymous_label)
                    else "param"
                ),
                sanitize_key=True,
            )
            self._key_is_anon = True
        elif key:
            self.key = key
        else:
            self.key = _anonymous_label.safe_construct(id(self), "param")
            self._key_is_anon = True

        # identifying key that won't change across
        # clones, used to identify the bind's logical
        # identity
        self._identifying_key = self.key

        # key that was passed in the first place, used to
        # generate new keys
        self._orig_key = key or "param"

        self.unique = unique
        self.value = value
        self.callable = callable_
        self.isoutparam = isoutparam
        self.required = required

        # indicate an "expanding" parameter; the compiler sets this
        # automatically in the compiler _render_in_expr_w_bindparam method
        # for an IN expression
        self.expanding = expanding

        # this is another hint to help w/ expanding and is typically
        # set in the compiler _render_in_expr_w_bindparam method for an
        # IN expression
        self.expand_op = None

        self.literal_execute = literal_execute
        if _is_crud:
            self._is_crud = True

        if type_ is None:
            if expanding:
                if value:
                    check_value = value[0]
                else:
                    check_value = type_api._NO_VALUE_IN_LIST
            else:
                check_value = value
            if _compared_to_type is not None:
                self.type = _compared_to_type.coerce_compared_value(
                    _compared_to_operator, check_value
                )
            else:
                self.type = type_api._resolve_value_to_type(check_value)
        elif isinstance(type_, type):
            self.type = type_()
        elif is_tuple_type(type_):
            if value:
                if expanding:
                    check_value = value[0]
                else:
                    check_value = value
                cast("BindParameter[typing_Tuple[Any, ...]]", self).type = (
                    type_._resolve_values_to_types(check_value)
                )
            else:
                cast("BindParameter[typing_Tuple[Any, ...]]", self).type = (
                    type_
                )
        else:
            self.type = type_

    def _with_value(self, value, maintain_key=False, required=NO_ARG):
        """Return a copy of this :class:`.BindParameter` with the given value
        set.
        """
        cloned = self._clone(maintain_key=maintain_key)
        cloned.value = value
        cloned.callable = None
        cloned.required = required if required is not NO_ARG else self.required
        if cloned.type is type_api.NULLTYPE:
            cloned.type = type_api._resolve_value_to_type(value)
        return cloned

    @property
    def effective_value(self) -> Optional[_T]:
        """Return the value of this bound parameter,
        taking into account if the ``callable`` parameter
        was set.

        The ``callable`` value will be evaluated
        and returned if present, else ``value``.

        """
        if self.callable:
            # TODO: set up protocol for bind parameter callable
            return self.callable()  # type: ignore
        else:
            return self.value

    def render_literal_execute(self) -> BindParameter[_T]:
        """Produce a copy of this bound parameter that will enable the
        :paramref:`_sql.BindParameter.literal_execute` flag.

        The :paramref:`_sql.BindParameter.literal_execute` flag will
        have the effect of the parameter rendered in the compiled SQL
        string using ``[POSTCOMPILE]`` form, which is a special form that
        is converted to be a rendering of the literal value of the parameter
        at SQL execution time.    The rationale is to support caching
        of SQL statement strings that can embed per-statement literal values,
        such as LIMIT and OFFSET parameters, in the final SQL string that
        is passed to the DBAPI.   Dialects in particular may want to use
        this method within custom compilation schemes.

        .. versionadded:: 1.4.5

        .. seealso::

            :ref:`engine_thirdparty_caching`

        """
        c = ClauseElement._clone(self)
        c.literal_execute = True
        return c

    def _negate_in_binary(self, negated_op, original_op):
        if self.expand_op is original_op:
            bind = self._clone()
            bind.expand_op = negated_op
            return bind
        else:
            return self

    def _with_binary_element_type(self, type_):
        c = ClauseElement._clone(self)
        c.type = type_
        return c

    def _clone(self, maintain_key: bool = False, **kw: Any) -> Self:
        c = ClauseElement._clone(self, **kw)
        # ensure all the BindParameter objects stay in cloned set.
        # in #7823, we changed "clone" so that a clone only keeps a reference
        # to the "original" element, since for column correspondence, that's
        # all we need.   However, for BindParam, _cloned_set is used by
        # the "cache key bind match" lookup, which means if any of those
        # interim BindParameter objects became part of a cache key in the
        # cache, we need it.  So here, make sure all clones keep carrying
        # forward.
        c._cloned_set.update(self._cloned_set)
        if not maintain_key and self.unique:
            c.key = _anonymous_label.safe_construct(
                id(c), c._orig_key or "param", sanitize_key=True
            )
        return c

    def _gen_cache_key(self, anon_map, bindparams):
        _gen_cache_ok = self.__class__.__dict__.get("inherit_cache", False)

        if not _gen_cache_ok:
            if anon_map is not None:
                anon_map[NO_CACHE] = True
            return None

        id_, found = anon_map.get_anon(self)
        if found:
            return (id_, self.__class__)

        if bindparams is not None:
            bindparams.append(self)

        return (
            id_,
            self.__class__,
            self.type._static_cache_key,
            self.key % anon_map if self._key_is_anon else self.key,
            self.literal_execute,
        )

    def _convert_to_unique(self):
        if not self.unique:
            self.unique = True
            self.key = _anonymous_label.safe_construct(
                id(self), self._orig_key or "param", sanitize_key=True
            )

    def __getstate__(self):
        """execute a deferred value for serialization purposes."""

        d = self.__dict__.copy()
        v = self.value
        if self.callable:
            v = self.callable()
            d["callable"] = None
        d["value"] = v
        return d

    def __setstate__(self, state):
        if state.get("unique", False):
            state["key"] = _anonymous_label.safe_construct(
                id(self), state.get("_orig_key", "param"), sanitize_key=True
            )
        self.__dict__.update(state)

    def __repr__(self):
        return "%s(%r, %r, type_=%r)" % (
            self.__class__.__name__,
            self.key,
            self.value,
            self.type,
        )


class TypeClause(DQLDMLClauseElement):
    """Handle a type keyword in a SQL statement.

    Used by the ``Case`` statement.

    """

    __visit_name__ = "typeclause"

    _traverse_internals: _TraverseInternalsType = [
        ("type", InternalTraversal.dp_type)
    ]
    type: TypeEngine[Any]

    def __init__(self, type_: TypeEngine[Any]):
        self.type = type_


class TextClause(
    roles.DDLConstraintColumnRole,
    roles.DDLExpressionRole,
    roles.StatementOptionRole,
    roles.WhereHavingRole,
    roles.OrderByRole,
    roles.FromClauseRole,
    roles.SelectStatementRole,
    roles.InElementRole,
    Generative,
    Executable,
    DQLDMLClauseElement,
    roles.BinaryElementRole[Any],
    inspection.Inspectable["TextClause"],
):
    """Represent a literal SQL text fragment.

    E.g.::

        from sqlalchemy import text

        t = text("SELECT * FROM users")
        result = connection.execute(t)

    The :class:`_expression.TextClause` construct is produced using the
    :func:`_expression.text`
    function; see that function for full documentation.

    .. seealso::

        :func:`_expression.text`

    """

    __visit_name__ = "textclause"

    _traverse_internals: _TraverseInternalsType = [
        ("_bindparams", InternalTraversal.dp_string_clauseelement_dict),
        ("text", InternalTraversal.dp_string),
    ]

    _is_text_clause = True

    _is_textual = True

    _bind_params_regex = re.compile(r"(?<![:\w\x5c]):(\w+)(?!:)", re.UNICODE)
    _is_implicitly_boolean = False

    _render_label_in_columns_clause = False

    _omit_from_statements = False

    _is_collection_aggregate = False

    @property
    def _hide_froms(self) -> Iterable[FromClause]:
        return ()

    def __and__(self, other):
        # support use in select.where(), query.filter()
        return and_(self, other)

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)

    # help in those cases where text() is
    # interpreted in a column expression situation
    key: Optional[str] = None
    _label: Optional[str] = None

    _allow_label_resolve = False

    @property
    def _is_star(self):
        return self.text == "*"

    def __init__(self, text: str):
        self._bindparams: Dict[str, BindParameter[Any]] = {}

        def repl(m):
            self._bindparams[m.group(1)] = BindParameter(m.group(1))
            return ":%s" % m.group(1)

        # scan the string and search for bind parameter names, add them
        # to the list of bindparams
        self.text = self._bind_params_regex.sub(repl, text)

    @_generative
    def bindparams(
        self,
        *binds: BindParameter[Any],
        **names_to_values: Any,
    ) -> Self:
        """Establish the values and/or types of bound parameters within
        this :class:`_expression.TextClause` construct.

        Given a text construct such as::

            from sqlalchemy import text

            stmt = text(
                "SELECT id, name FROM user WHERE name=:name AND timestamp=:timestamp"
            )

        the :meth:`_expression.TextClause.bindparams`
        method can be used to establish
        the initial value of ``:name`` and ``:timestamp``,
        using simple keyword arguments::

            stmt = stmt.bindparams(
                name="jack", timestamp=datetime.datetime(2012, 10, 8, 15, 12, 5)
            )

        Where above, new :class:`.BindParameter` objects
        will be generated with the names ``name`` and ``timestamp``, and
        values of ``jack`` and ``datetime.datetime(2012, 10, 8, 15, 12, 5)``,
        respectively.  The types will be
        inferred from the values given, in this case :class:`.String` and
        :class:`.DateTime`.

        When specific typing behavior is needed, the positional ``*binds``
        argument can be used in which to specify :func:`.bindparam` constructs
        directly.  These constructs must include at least the ``key``
        argument, then an optional value and type::

            from sqlalchemy import bindparam

            stmt = stmt.bindparams(
                bindparam("name", value="jack", type_=String),
                bindparam("timestamp", type_=DateTime),
            )

        Above, we specified the type of :class:`.DateTime` for the
        ``timestamp`` bind, and the type of :class:`.String` for the ``name``
        bind.  In the case of ``name`` we also set the default value of
        ``"jack"``.

        Additional bound parameters can be supplied at statement execution
        time, e.g.::

            result = connection.execute(
                stmt, timestamp=datetime.datetime(2012, 10, 8, 15, 12, 5)
            )

        The :meth:`_expression.TextClause.bindparams`
        method can be called repeatedly,
        where it will re-use existing :class:`.BindParameter` objects to add
        new information.  For example, we can call
        :meth:`_expression.TextClause.bindparams`
        first with typing information, and a
        second time with value information, and it will be combined::

            stmt = text(
                "SELECT id, name FROM user WHERE name=:name "
                "AND timestamp=:timestamp"
            )
            stmt = stmt.bindparams(
                bindparam("name", type_=String), bindparam("timestamp", type_=DateTime)
            )
            stmt = stmt.bindparams(
                name="jack", timestamp=datetime.datetime(2012, 10, 8, 15, 12, 5)
            )

        The :meth:`_expression.TextClause.bindparams`
        method also supports the concept of
        **unique** bound parameters.  These are parameters that are
        "uniquified" on name at statement compilation time, so that  multiple
        :func:`_expression.text`
        constructs may be combined together without the names
        conflicting.  To use this feature, specify the
        :paramref:`.BindParameter.unique` flag on each :func:`.bindparam`
        object::

            stmt1 = text("select id from table where name=:name").bindparams(
                bindparam("name", value="name1", unique=True)
            )
            stmt2 = text("select id from table where name=:name").bindparams(
                bindparam("name", value="name2", unique=True)
            )

            union = union_all(stmt1.columns(column("id")), stmt2.columns(column("id")))

        The above statement will render as:

        .. sourcecode:: sql

            select id from table where name=:name_1
            UNION ALL select id from table where name=:name_2

        .. versionadded:: 1.3.11  Added support for the
           :paramref:`.BindParameter.unique` flag to work with
           :func:`_expression.text`
           constructs.

        """  # noqa: E501
        self._bindparams = new_params = self._bindparams.copy()

        for bind in binds:
            try:
                # the regex used for text() currently will not match
                # a unique/anonymous key in any case, so use the _orig_key
                # so that a text() construct can support unique parameters
                existing = new_params[bind._orig_key]
            except KeyError as err:
                raise exc.ArgumentError(
                    "This text() construct doesn't define a "
                    "bound parameter named %r" % bind._orig_key
                ) from err
            else:
                new_params[existing._orig_key] = bind

        for key, value in names_to_values.items():
            try:
                existing = new_params[key]
            except KeyError as err:
                raise exc.ArgumentError(
                    "This text() construct doesn't define a "
                    "bound parameter named %r" % key
                ) from err
            else:
                new_params[key] = existing._with_value(value, required=False)
        return self

    @util.preload_module("sqlalchemy.sql.selectable")
    def columns(
        self,
        *cols: _ColumnExpressionArgument[Any],
        **types: _TypeEngineArgument[Any],
    ) -> TextualSelect:
        r"""Turn this :class:`_expression.TextClause` object into a
        :class:`_expression.TextualSelect`
        object that serves the same role as a SELECT
        statement.

        The :class:`_expression.TextualSelect` is part of the
        :class:`_expression.SelectBase`
        hierarchy and can be embedded into another statement by using the
        :meth:`_expression.TextualSelect.subquery` method to produce a
        :class:`.Subquery`
        object, which can then be SELECTed from.

        This function essentially bridges the gap between an entirely
        textual SELECT statement and the SQL expression language concept
        of a "selectable"::

            from sqlalchemy.sql import column, text

            stmt = text("SELECT id, name FROM some_table")
            stmt = stmt.columns(column("id"), column("name")).subquery("st")

            stmt = (
                select(mytable)
                .select_from(mytable.join(stmt, mytable.c.name == stmt.c.name))
                .where(stmt.c.id > 5)
            )

        Above, we pass a series of :func:`_expression.column` elements to the
        :meth:`_expression.TextClause.columns` method positionally.  These
        :func:`_expression.column`
        elements now become first class elements upon the
        :attr:`_expression.TextualSelect.selected_columns` column collection,
        which then
        become part of the :attr:`.Subquery.c` collection after
        :meth:`_expression.TextualSelect.subquery` is invoked.

        The column expressions we pass to
        :meth:`_expression.TextClause.columns` may
        also be typed; when we do so, these :class:`.TypeEngine` objects become
        the effective return type of the column, so that SQLAlchemy's
        result-set-processing systems may be used on the return values.
        This is often needed for types such as date or boolean types, as well
        as for unicode processing on some dialect configurations::

            stmt = text("SELECT id, name, timestamp FROM some_table")
            stmt = stmt.columns(
                column("id", Integer),
                column("name", Unicode),
                column("timestamp", DateTime),
            )

            for id, name, timestamp in connection.execute(stmt):
                print(id, name, timestamp)

        As a shortcut to the above syntax, keyword arguments referring to
        types alone may be used, if only type conversion is needed::

            stmt = text("SELECT id, name, timestamp FROM some_table")
            stmt = stmt.columns(id=Integer, name=Unicode, timestamp=DateTime)

            for id, name, timestamp in connection.execute(stmt):
                print(id, name, timestamp)

        The positional form of :meth:`_expression.TextClause.columns`
        also provides the
        unique feature of **positional column targeting**, which is
        particularly useful when using the ORM with complex textual queries. If
        we specify the columns from our model to
        :meth:`_expression.TextClause.columns`,
        the result set will match to those columns positionally, meaning the
        name or origin of the column in the textual SQL doesn't matter::

            stmt = text(
                "SELECT users.id, addresses.id, users.id, "
                "users.name, addresses.email_address AS email "
                "FROM users JOIN addresses ON users.id=addresses.user_id "
                "WHERE users.id = 1"
            ).columns(
                User.id,
                Address.id,
                Address.user_id,
                User.name,
                Address.email_address,
            )

            query = (
                session.query(User)
                .from_statement(stmt)
                .options(contains_eager(User.addresses))
            )

        The :meth:`_expression.TextClause.columns` method provides a direct
        route to calling :meth:`_expression.FromClause.subquery` as well as
        :meth:`_expression.SelectBase.cte`
        against a textual SELECT statement::

            stmt = stmt.columns(id=Integer, name=String).cte("st")

            stmt = select(sometable).where(sometable.c.id == stmt.c.id)

        :param \*cols: A series of :class:`_expression.ColumnElement` objects,
         typically
         :class:`_schema.Column` objects from a :class:`_schema.Table`
         or ORM level
         column-mapped attributes, representing a set of columns that this
         textual string will SELECT from.

        :param \**types: A mapping of string names to :class:`.TypeEngine`
         type objects indicating the datatypes to use for names that are
         SELECTed from the textual string.  Prefer to use the ``*cols``
         argument as it also indicates positional ordering.

        """
        selectable = util.preloaded.sql_selectable

        input_cols: List[NamedColumn[Any]] = [
            coercions.expect(roles.LabeledColumnExprRole, col) for col in cols
        ]

        positional_input_cols = [
            (
                ColumnClause(col.key, types.pop(col.key))
                if col.key in types
                else col
            )
            for col in input_cols
        ]
        keyed_input_cols: List[NamedColumn[Any]] = [
            ColumnClause(key, type_) for key, type_ in types.items()
        ]

        elem = selectable.TextualSelect.__new__(selectable.TextualSelect)
        elem._init(
            self,
            positional_input_cols + keyed_input_cols,
            positional=bool(positional_input_cols) and not keyed_input_cols,
        )
        return elem

    @property
    def type(self) -> TypeEngine[Any]:
        return type_api.NULLTYPE

    @property
    def comparator(self):
        # TODO: this seems wrong, it seems like we might not
        # be using this method.
        return self.type.comparator_factory(self)  # type: ignore

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> Union[Self, Grouping[Any]]:
        if against is operators.in_op:
            return Grouping(self)
        else:
            return self


class Null(SingletonConstant, roles.ConstExprRole[None], ColumnElement[None]):
    """Represent the NULL keyword in a SQL statement.

    :class:`.Null` is accessed as a constant via the
    :func:`.null` function.

    """

    __visit_name__ = "null"

    _traverse_internals: _TraverseInternalsType = []
    _singleton: Null

    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:  # noqa: A001
            return type_api.NULLTYPE

    @classmethod
    def _instance(cls) -> Null:
        """Return a constant :class:`.Null` construct."""

        return Null._singleton


Null._create_singleton()


class False_(
    SingletonConstant, roles.ConstExprRole[bool], ColumnElement[bool]
):
    """Represent the ``false`` keyword, or equivalent, in a SQL statement.

    :class:`.False_` is accessed as a constant via the
    :func:`.false` function.

    """

    __visit_name__ = "false"
    _traverse_internals: _TraverseInternalsType = []
    _singleton: False_

    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:  # noqa: A001
            return type_api.BOOLEANTYPE

    def _negate(self) -> True_:
        return True_._singleton

    @classmethod
    def _instance(cls) -> False_:
        return False_._singleton


False_._create_singleton()


class True_(SingletonConstant, roles.ConstExprRole[bool], ColumnElement[bool]):
    """Represent the ``true`` keyword, or equivalent, in a SQL statement.

    :class:`.True_` is accessed as a constant via the
    :func:`.true` function.

    """

    __visit_name__ = "true"

    _traverse_internals: _TraverseInternalsType = []
    _singleton: True_

    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:  # noqa: A001
            return type_api.BOOLEANTYPE

    def _negate(self) -> False_:
        return False_._singleton

    @classmethod
    def _ifnone(
        cls, other: Optional[ColumnElement[Any]]
    ) -> ColumnElement[Any]:
        if other is None:
            return cls._instance()
        else:
            return other

    @classmethod
    def _instance(cls) -> True_:
        return True_._singleton


True_._create_singleton()


class ClauseList(
    roles.InElementRole,
    roles.OrderByRole,
    roles.ColumnsClauseRole,
    roles.DMLColumnRole,
    DQLDMLClauseElement,
):
    """Describe a list of clauses, separated by an operator.

    By default, is comma-separated, such as a column listing.

    """

    __visit_name__ = "clauselist"

    # this is used only by the ORM in a legacy use case for
    # composite attributes
    _is_clause_list = True

    _traverse_internals: _TraverseInternalsType = [
        ("clauses", InternalTraversal.dp_clauseelement_list),
        ("operator", InternalTraversal.dp_operator),
    ]

    clauses: List[ColumnElement[Any]]

    def __init__(
        self,
        *clauses: _ColumnExpressionArgument[Any],
        operator: OperatorType = operators.comma_op,
        group: bool = True,
        group_contents: bool = True,
        _literal_as_text_role: Type[roles.SQLRole] = roles.WhereHavingRole,
    ):
        self.operator = operator
        self.group = group
        self.group_contents = group_contents
        clauses_iterator: Iterable[_ColumnExpressionArgument[Any]] = clauses
        text_converter_role: Type[roles.SQLRole] = _literal_as_text_role
        self._text_converter_role = text_converter_role

        if self.group_contents:
            self.clauses = [
                coercions.expect(
                    text_converter_role, clause, apply_propagate_attrs=self
                ).self_group(against=self.operator)
                for clause in clauses_iterator
            ]
        else:
            self.clauses = [
                coercions.expect(
                    text_converter_role, clause, apply_propagate_attrs=self
                )
                for clause in clauses_iterator
            ]
        self._is_implicitly_boolean = operators.is_boolean(self.operator)

    @classmethod
    def _construct_raw(
        cls,
        operator: OperatorType,
        clauses: Optional[Sequence[ColumnElement[Any]]] = None,
    ) -> ClauseList:
        self = cls.__new__(cls)
        self.clauses = list(clauses) if clauses else []
        self.group = True
        self.operator = operator
        self.group_contents = True
        self._is_implicitly_boolean = False
        return self

    def __iter__(self) -> Iterator[ColumnElement[Any]]:
        return iter(self.clauses)

    def __len__(self) -> int:
        return len(self.clauses)

    @property
    def _select_iterable(self) -> _SelectIterable:
        return itertools.chain.from_iterable(
            [elem._select_iterable for elem in self.clauses]
        )

    def append(self, clause):
        if self.group_contents:
            self.clauses.append(
                coercions.expect(self._text_converter_role, clause).self_group(
                    against=self.operator
                )
            )
        else:
            self.clauses.append(
                coercions.expect(self._text_converter_role, clause)
            )

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(itertools.chain(*[c._from_objects for c in self.clauses]))

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> Union[Self, Grouping[Any]]:
        if self.group and operators.is_precedent(self.operator, against):
            return Grouping(self)
        else:
            return self


class OperatorExpression(ColumnElement[_T]):
    """base for expressions that contain an operator and operands

    .. versionadded:: 2.0

    """

    operator: OperatorType
    type: TypeEngine[_T]

    group: bool = True

    @property
    def is_comparison(self):
        return operators.is_comparison(self.operator)

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> Union[Self, Grouping[_T]]:
        if (
            self.group
            and operators.is_precedent(self.operator, against)
            or (
                # a negate against a non-boolean operator
                # doesn't make too much sense but we should
                # group for that
                against is operators.inv
                and not operators.is_boolean(self.operator)
            )
        ):
            return Grouping(self)
        else:
            return self

    @property
    def _flattened_operator_clauses(
        self,
    ) -> typing_Tuple[ColumnElement[Any], ...]:
        raise NotImplementedError()

    @classmethod
    def _construct_for_op(
        cls,
        left: ColumnElement[Any],
        right: ColumnElement[Any],
        op: OperatorType,
        *,
        type_: TypeEngine[_T],
        negate: Optional[OperatorType] = None,
        modifiers: Optional[Mapping[str, Any]] = None,
    ) -> OperatorExpression[_T]:
        if operators.is_associative(op):
            assert (
                negate is None
            ), f"negate not supported for associative operator {op}"

            multi = False
            if getattr(
                left, "operator", None
            ) is op and type_._compare_type_affinity(left.type):
                multi = True
                left_flattened = left._flattened_operator_clauses
            else:
                left_flattened = (left,)

            if getattr(
                right, "operator", None
            ) is op and type_._compare_type_affinity(right.type):
                multi = True
                right_flattened = right._flattened_operator_clauses
            else:
                right_flattened = (right,)

            if multi:
                return ExpressionClauseList._construct_for_list(
                    op,
                    type_,
                    *(left_flattened + right_flattened),
                )

        if right._is_collection_aggregate:
            negate = None

        return BinaryExpression(
            left, right, op, type_=type_, negate=negate, modifiers=modifiers
        )


class ExpressionClauseList(OperatorExpression[_T]):
    """Describe a list of clauses, separated by an operator,
    in a column expression context.

    :class:`.ExpressionClauseList` differs from :class:`.ClauseList` in that
    it represents a column-oriented DQL expression only, not an open ended
    list of anything comma separated.

    .. versionadded:: 2.0

    """

    __visit_name__ = "expression_clauselist"

    _traverse_internals: _TraverseInternalsType = [
        ("clauses", InternalTraversal.dp_clauseelement_tuple),
        ("operator", InternalTraversal.dp_operator),
    ]

    clauses: typing_Tuple[ColumnElement[Any], ...]

    group: bool

    def __init__(
        self,
        operator: OperatorType,
        *clauses: _ColumnExpressionArgument[Any],
        type_: Optional[_TypeEngineArgument[_T]] = None,
    ):
        self.operator = operator

        self.clauses = tuple(
            coercions.expect(
                roles.ExpressionElementRole, clause, apply_propagate_attrs=self
            )
            for clause in clauses
        )
        self._is_implicitly_boolean = operators.is_boolean(self.operator)
        self.type = type_api.to_instance(type_)  # type: ignore

    @property
    def _flattened_operator_clauses(
        self,
    ) -> typing_Tuple[ColumnElement[Any], ...]:
        return self.clauses

    def __iter__(self) -> Iterator[ColumnElement[Any]]:
        return iter(self.clauses)

    def __len__(self) -> int:
        return len(self.clauses)

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(itertools.chain(*[c._from_objects for c in self.clauses]))

    def _append_inplace(self, clause: ColumnElement[Any]) -> None:
        self.clauses += (clause,)

    @classmethod
    def _construct_for_list(
        cls,
        operator: OperatorType,
        type_: TypeEngine[_T],
        *clauses: ColumnElement[Any],
        group: bool = True,
    ) -> ExpressionClauseList[_T]:
        self = cls.__new__(cls)
        self.group = group
        if group:
            self.clauses = tuple(
                c.self_group(against=operator) for c in clauses
            )
        else:
            self.clauses = clauses
        self.operator = operator
        self.type = type_
        for c in clauses:
            if c._propagate_attrs:
                self._propagate_attrs = c._propagate_attrs
                break
        return self

    def _negate(self) -> Any:
        grouped = self.self_group(against=operators.inv)
        assert isinstance(grouped, ColumnElement)
        return UnaryExpression(
            grouped, operator=operators.inv, wraps_column_expression=True
        )


class BooleanClauseList(ExpressionClauseList[bool]):
    __visit_name__ = "expression_clauselist"
    inherit_cache = True

    def __init__(self, *arg, **kw):
        raise NotImplementedError(
            "BooleanClauseList has a private constructor"
        )

    @classmethod
    def _process_clauses_for_boolean(
        cls,
        operator: OperatorType,
        continue_on: Any,
        skip_on: Any,
        clauses: Iterable[ColumnElement[Any]],
    ) -> typing_Tuple[int, List[ColumnElement[Any]]]:
        has_continue_on = None

        convert_clauses = []

        against = operators._asbool
        lcc = 0

        for clause in clauses:
            if clause is continue_on:
                # instance of continue_on, like and_(x, y, True, z), store it
                # if we didn't find one already, we will use it if there
                # are no other expressions here.
                has_continue_on = clause
            elif clause is skip_on:
                # instance of skip_on, e.g. and_(x, y, False, z), cancels
                # the rest out
                convert_clauses = [clause]
                lcc = 1
                break
            else:
                if not lcc:
                    lcc = 1
                else:
                    against = operator
                    # technically this would be len(convert_clauses) + 1
                    # however this only needs to indicate "greater than one"
                    lcc = 2
                convert_clauses.append(clause)

        if not convert_clauses and has_continue_on is not None:
            convert_clauses = [has_continue_on]
            lcc = 1

        return lcc, [c.self_group(against=against) for c in convert_clauses]

    @classmethod
    def _construct(
        cls,
        operator: OperatorType,
        continue_on: Any,
        skip_on: Any,
        initial_clause: Any = _NoArg.NO_ARG,
        *clauses: Any,
        **kw: Any,
    ) -> ColumnElement[Any]:
        if initial_clause is _NoArg.NO_ARG:
            # no elements period.  deprecated use case.  return an empty
            # ClauseList construct that generates nothing unless it has
            # elements added to it.
            name = operator.__name__

            util.warn_deprecated(
                f"Invoking {name}() without arguments is deprecated, and "
                f"will be disallowed in a future release.   For an empty "
                f"""{name}() construct, use '{name}({
                    'true()' if continue_on is True_._singleton else 'false()'
                }, *args)' """
                f"""or '{name}({
                    'True' if continue_on is True_._singleton else 'False'
                }, *args)'.""",
                version="1.4",
            )
            return cls._construct_raw(operator)

        lcc, convert_clauses = cls._process_clauses_for_boolean(
            operator,
            continue_on,
            skip_on,
            [
                coercions.expect(roles.WhereHavingRole, clause)
                for clause in util.coerce_generator_arg(
                    (initial_clause,) + clauses
                )
            ],
        )

        if lcc > 1:
            # multiple elements.  Return regular BooleanClauseList
            # which will link elements against the operator.

            flattened_clauses = itertools.chain.from_iterable(
                (
                    (c for c in to_flat._flattened_operator_clauses)
                    if getattr(to_flat, "operator", None) is operator
                    else (to_flat,)
                )
                for to_flat in convert_clauses
            )

            return cls._construct_raw(operator, flattened_clauses)  # type: ignore # noqa: E501
        else:
            assert lcc
            # just one element.  return it as a single boolean element,
            # not a list and discard the operator.
            return convert_clauses[0]

    @classmethod
    def _construct_for_whereclause(
        cls, clauses: Iterable[ColumnElement[Any]]
    ) -> Optional[ColumnElement[bool]]:
        operator, continue_on, skip_on = (
            operators.and_,
            True_._singleton,
            False_._singleton,
        )

        lcc, convert_clauses = cls._process_clauses_for_boolean(
            operator,
            continue_on,
            skip_on,
            clauses,  # these are assumed to be coerced already
        )

        if lcc > 1:
            # multiple elements.  Return regular BooleanClauseList
            # which will link elements against the operator.
            return cls._construct_raw(operator, convert_clauses)
        elif lcc == 1:
            # just one element.  return it as a single boolean element,
            # not a list and discard the operator.
            return convert_clauses[0]
        else:
            return None

    @classmethod
    def _construct_raw(
        cls,
        operator: OperatorType,
        clauses: Optional[Sequence[ColumnElement[Any]]] = None,
    ) -> BooleanClauseList:
        self = cls.__new__(cls)
        self.clauses = tuple(clauses) if clauses else ()
        self.group = True
        self.operator = operator
        self.type = type_api.BOOLEANTYPE
        self._is_implicitly_boolean = True
        return self

    @classmethod
    def and_(
        cls,
        initial_clause: Union[
            Literal[True], _ColumnExpressionArgument[bool], _NoArg
        ] = _NoArg.NO_ARG,
        *clauses: _ColumnExpressionArgument[bool],
    ) -> ColumnElement[bool]:
        r"""Produce a conjunction of expressions joined by ``AND``.

        See :func:`_sql.and_` for full documentation.
        """
        return cls._construct(
            operators.and_,
            True_._singleton,
            False_._singleton,
            initial_clause,
            *clauses,
        )

    @classmethod
    def or_(
        cls,
        initial_clause: Union[
            Literal[False], _ColumnExpressionArgument[bool], _NoArg
        ] = _NoArg.NO_ARG,
        *clauses: _ColumnExpressionArgument[bool],
    ) -> ColumnElement[bool]:
        """Produce a conjunction of expressions joined by ``OR``.

        See :func:`_sql.or_` for full documentation.
        """
        return cls._construct(
            operators.or_,
            False_._singleton,
            True_._singleton,
            initial_clause,
            *clauses,
        )

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> Union[Self, Grouping[bool]]:
        if not self.clauses:
            return self
        else:
            return super().self_group(against=against)


and_ = BooleanClauseList.and_
or_ = BooleanClauseList.or_


class Tuple(ClauseList, ColumnElement[typing_Tuple[Any, ...]]):
    """Represent a SQL tuple."""

    __visit_name__ = "tuple"

    _traverse_internals: _TraverseInternalsType = (
        ClauseList._traverse_internals + []
    )

    type: TupleType

    @util.preload_module("sqlalchemy.sql.sqltypes")
    def __init__(
        self,
        *clauses: _ColumnExpressionArgument[Any],
        types: Optional[Sequence[_TypeEngineArgument[Any]]] = None,
    ):
        sqltypes = util.preloaded.sql_sqltypes

        if types is None:
            init_clauses: List[ColumnElement[Any]] = [
                coercions.expect(roles.ExpressionElementRole, c)
                for c in clauses
            ]
        else:
            if len(types) != len(clauses):
                raise exc.ArgumentError(
                    "Wrong number of elements for %d-tuple: %r "
                    % (len(types), clauses)
                )
            init_clauses = [
                coercions.expect(
                    roles.ExpressionElementRole,
                    c,
                    type_=typ if not typ._isnull else None,
                )
                for typ, c in zip(types, clauses)
            ]

        self.type = sqltypes.TupleType(*[arg.type for arg in init_clauses])
        super().__init__(*init_clauses)

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)

    def _bind_param(self, operator, obj, type_=None, expanding=False):
        if expanding:
            return BindParameter(
                None,
                value=obj,
                _compared_to_operator=operator,
                unique=True,
                expanding=True,
                type_=type_,
                _compared_to_type=self.type,
            )
        else:
            return Tuple(
                *[
                    BindParameter(
                        None,
                        o,
                        _compared_to_operator=operator,
                        _compared_to_type=compared_to_type,
                        unique=True,
                        type_=type_,
                    )
                    for o, compared_to_type in zip(obj, self.type.types)
                ]
            )

    def self_group(self, against: Optional[OperatorType] = None) -> Self:
        # Tuple is parenthesized by definition.
        return self


class Case(ColumnElement[_T]):
    """Represent a ``CASE`` expression.

    :class:`.Case` is produced using the :func:`.case` factory function,
    as in::

        from sqlalchemy import case

        stmt = select(users_table).where(
            case(
                (users_table.c.name == "wendy", "W"),
                (users_table.c.name == "jack", "J"),
                else_="E",
            )
        )

    Details on :class:`.Case` usage is at :func:`.case`.

    .. seealso::

        :func:`.case`

    """

    __visit_name__ = "case"

    _traverse_internals: _TraverseInternalsType = [
        ("value", InternalTraversal.dp_clauseelement),
        ("whens", InternalTraversal.dp_clauseelement_tuples),
        ("else_", InternalTraversal.dp_clauseelement),
    ]

    # for case(), the type is derived from the whens.  so for the moment
    # users would have to cast() the case to get a specific type

    whens: List[typing_Tuple[ColumnElement[bool], ColumnElement[_T]]]
    else_: Optional[ColumnElement[_T]]
    value: Optional[ColumnElement[Any]]

    def __init__(
        self,
        *whens: Union[
            typing_Tuple[_ColumnExpressionArgument[bool], Any],
            Mapping[Any, Any],
        ],
        value: Optional[Any] = None,
        else_: Optional[Any] = None,
    ):
        new_whens: Iterable[Any] = coercions._expression_collection_was_a_list(
            "whens", "case", whens
        )
        try:
            new_whens = util.dictlike_iteritems(new_whens)
        except TypeError:
            pass

        self.whens = [
            (
                coercions.expect(
                    roles.ExpressionElementRole,
                    c,
                    apply_propagate_attrs=self,
                ).self_group(),
                coercions.expect(roles.ExpressionElementRole, r),
            )
            for (c, r) in new_whens
        ]

        if value is None:
            self.value = None
        else:
            self.value = coercions.expect(roles.ExpressionElementRole, value)

        if else_ is not None:
            self.else_ = coercions.expect(roles.ExpressionElementRole, else_)
        else:
            self.else_ = None

        type_ = next(
            (
                then.type
                # Iterate `whens` in reverse to match previous behaviour
                # where type of final element took priority
                for *_, then in reversed(self.whens)
                if not then.type._isnull
            ),
            self.else_.type if self.else_ is not None else type_api.NULLTYPE,
        )
        self.type = cast(_T, type_)

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(
            itertools.chain(*[x._from_objects for x in self.get_children()])
        )


class Cast(WrapsColumnExpression[_T]):
    """Represent a ``CAST`` expression.

    :class:`.Cast` is produced using the :func:`.cast` factory function,
    as in::

        from sqlalchemy import cast, Numeric

        stmt = select(cast(product_table.c.unit_price, Numeric(10, 4)))

    Details on :class:`.Cast` usage is at :func:`.cast`.

    .. seealso::

        :ref:`tutorial_casts`

        :func:`.cast`

        :func:`.try_cast`

        :func:`.type_coerce` - an alternative to CAST that coerces the type
        on the Python side only, which is often sufficient to generate the
        correct SQL and data coercion.

    """

    __visit_name__ = "cast"

    _traverse_internals: _TraverseInternalsType = [
        ("clause", InternalTraversal.dp_clauseelement),
        ("type", InternalTraversal.dp_type),
    ]

    clause: ColumnElement[Any]
    type: TypeEngine[_T]
    typeclause: TypeClause

    def __init__(
        self,
        expression: _ColumnExpressionArgument[Any],
        type_: _TypeEngineArgument[_T],
    ):
        self.type = type_api.to_instance(type_)
        self.clause = coercions.expect(
            roles.ExpressionElementRole,
            expression,
            type_=self.type,
            apply_propagate_attrs=self,
        )
        self.typeclause = TypeClause(self.type)

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.clause._from_objects

    @property
    def wrapped_column_expression(self):
        return self.clause


class TryCast(Cast[_T]):
    """Represent a TRY_CAST expression.

    Details on :class:`.TryCast` usage is at :func:`.try_cast`.

    .. seealso::

        :func:`.try_cast`

        :ref:`tutorial_casts`
    """

    __visit_name__ = "try_cast"
    inherit_cache = True


class TypeCoerce(WrapsColumnExpression[_T]):
    """Represent a Python-side type-coercion wrapper.

    :class:`.TypeCoerce` supplies the :func:`_expression.type_coerce`
    function; see that function for usage details.

    .. seealso::

        :func:`_expression.type_coerce`

        :func:`.cast`

    """

    __visit_name__ = "type_coerce"

    _traverse_internals: _TraverseInternalsType = [
        ("clause", InternalTraversal.dp_clauseelement),
        ("type", InternalTraversal.dp_type),
    ]

    clause: ColumnElement[Any]
    type: TypeEngine[_T]

    def __init__(
        self,
        expression: _ColumnExpressionArgument[Any],
        type_: _TypeEngineArgument[_T],
    ):
        self.type = type_api.to_instance(type_)
        self.clause = coercions.expect(
            roles.ExpressionElementRole,
            expression,
            type_=self.type,
            apply_propagate_attrs=self,
        )

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.clause._from_objects

    @HasMemoized.memoized_attribute
    def typed_expression(self):
        if isinstance(self.clause, BindParameter):
            bp = self.clause._clone()
            bp.type = self.type
            return bp
        else:
            return self.clause

    @property
    def wrapped_column_expression(self):
        return self.clause

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> TypeCoerce[_T]:
        grouped = self.clause.self_group(against=against)
        if grouped is not self.clause:
            return TypeCoerce(grouped, self.type)
        else:
            return self


class Extract(ColumnElement[int]):
    """Represent a SQL EXTRACT clause, ``extract(field FROM expr)``."""

    __visit_name__ = "extract"

    _traverse_internals: _TraverseInternalsType = [
        ("expr", InternalTraversal.dp_clauseelement),
        ("field", InternalTraversal.dp_string),
    ]

    expr: ColumnElement[Any]
    field: str

    def __init__(self, field: str, expr: _ColumnExpressionArgument[Any]):
        self.type = type_api.INTEGERTYPE
        self.field = field
        self.expr = coercions.expect(roles.ExpressionElementRole, expr)

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.expr._from_objects


class _label_reference(ColumnElement[_T]):
    """Wrap a column expression as it appears in a 'reference' context.

    This expression is any that includes an _order_by_label_element,
    which is a Label, or a DESC / ASC construct wrapping a Label.

    The production of _label_reference() should occur when an expression
    is added to this context; this includes the ORDER BY or GROUP BY of a
    SELECT statement, as well as a few other places, such as the ORDER BY
    within an OVER clause.

    """

    __visit_name__ = "label_reference"

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement)
    ]

    element: ColumnElement[_T]

    def __init__(self, element: ColumnElement[_T]):
        self.element = element

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return []


class _textual_label_reference(ColumnElement[Any]):
    __visit_name__ = "textual_label_reference"

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_string)
    ]

    def __init__(self, element: str):
        self.element = element

    @util.memoized_property
    def _text_clause(self) -> TextClause:
        return TextClause(self.element)


class UnaryExpression(ColumnElement[_T]):
    """Define a 'unary' expression.

    A unary expression has a single column expression
    and an operator.  The operator can be placed on the left
    (where it is called the 'operator') or right (where it is called the
    'modifier') of the column expression.

    :class:`.UnaryExpression` is the basis for several unary operators
    including those used by :func:`.desc`, :func:`.asc`, :func:`.distinct`,
    :func:`.nulls_first` and :func:`.nulls_last`.

    """

    __visit_name__ = "unary"

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement),
        ("operator", InternalTraversal.dp_operator),
        ("modifier", InternalTraversal.dp_operator),
    ]

    element: ClauseElement

    def __init__(
        self,
        element: ColumnElement[Any],
        operator: Optional[OperatorType] = None,
        modifier: Optional[OperatorType] = None,
        type_: Optional[_TypeEngineArgument[_T]] = None,
        wraps_column_expression: bool = False,
    ):
        self.operator = operator
        self.modifier = modifier
        self._propagate_attrs = element._propagate_attrs
        self.element = element.self_group(
            against=self.operator or self.modifier
        )

        # if type is None, we get NULLTYPE, which is our _T.  But I don't
        # know how to get the overloads to express that correctly
        self.type = type_api.to_instance(type_)  # type: ignore

        self.wraps_column_expression = wraps_column_expression

    @classmethod
    def _create_nulls_first(
        cls,
        column: _ColumnExpressionArgument[_T],
    ) -> UnaryExpression[_T]:
        return UnaryExpression(
            coercions.expect(roles.ByOfRole, column),
            modifier=operators.nulls_first_op,
            wraps_column_expression=False,
        )

    @classmethod
    def _create_nulls_last(
        cls,
        column: _ColumnExpressionArgument[_T],
    ) -> UnaryExpression[_T]:
        return UnaryExpression(
            coercions.expect(roles.ByOfRole, column),
            modifier=operators.nulls_last_op,
            wraps_column_expression=False,
        )

    @classmethod
    def _create_desc(
        cls, column: _ColumnExpressionOrStrLabelArgument[_T]
    ) -> UnaryExpression[_T]:
        return UnaryExpression(
            coercions.expect(roles.ByOfRole, column),
            modifier=operators.desc_op,
            wraps_column_expression=False,
        )

    @classmethod
    def _create_asc(
        cls,
        column: _ColumnExpressionOrStrLabelArgument[_T],
    ) -> UnaryExpression[_T]:
        return UnaryExpression(
            coercions.expect(roles.ByOfRole, column),
            modifier=operators.asc_op,
            wraps_column_expression=False,
        )

    @classmethod
    def _create_distinct(
        cls,
        expr: _ColumnExpressionArgument[_T],
    ) -> UnaryExpression[_T]:
        col_expr: ColumnElement[_T] = coercions.expect(
            roles.ExpressionElementRole, expr
        )
        return UnaryExpression(
            col_expr,
            operator=operators.distinct_op,
            type_=col_expr.type,
            wraps_column_expression=False,
        )

    @classmethod
    def _create_bitwise_not(
        cls,
        expr: _ColumnExpressionArgument[_T],
    ) -> UnaryExpression[_T]:
        col_expr: ColumnElement[_T] = coercions.expect(
            roles.ExpressionElementRole, expr
        )
        return UnaryExpression(
            col_expr,
            operator=operators.bitwise_not_op,
            type_=col_expr.type,
            wraps_column_expression=False,
        )

    @property
    def _order_by_label_element(self) -> Optional[Label[Any]]:
        if operators.is_order_by_modifier(self.modifier):
            return self.element._order_by_label_element
        else:
            return None

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.element._from_objects

    def _negate(self):
        if self.type._type_affinity is type_api.BOOLEANTYPE._type_affinity:
            return UnaryExpression(
                self.self_group(against=operators.inv),
                operator=operators.inv,
                type_=type_api.BOOLEANTYPE,
                wraps_column_expression=self.wraps_column_expression,
            )
        else:
            return ClauseElement._negate(self)

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> Union[Self, Grouping[_T]]:
        if self.operator and operators.is_precedent(self.operator, against):
            return Grouping(self)
        else:
            return self


class CollectionAggregate(UnaryExpression[_T]):
    """Forms the basis for right-hand collection operator modifiers
    ANY and ALL.

    The ANY and ALL keywords are available in different ways on different
    backends.  On PostgreSQL, they only work for an ARRAY type.  On
    MySQL, they only work for subqueries.

    """

    inherit_cache = True
    _is_collection_aggregate = True

    @classmethod
    def _create_any(
        cls, expr: _ColumnExpressionArgument[_T]
    ) -> CollectionAggregate[bool]:
        col_expr: ColumnElement[_T] = coercions.expect(
            roles.ExpressionElementRole,
            expr,
        )
        col_expr = col_expr.self_group()
        return CollectionAggregate(
            col_expr,
            operator=operators.any_op,
            type_=type_api.BOOLEANTYPE,
            wraps_column_expression=False,
        )

    @classmethod
    def _create_all(
        cls, expr: _ColumnExpressionArgument[_T]
    ) -> CollectionAggregate[bool]:
        col_expr: ColumnElement[_T] = coercions.expect(
            roles.ExpressionElementRole,
            expr,
        )
        col_expr = col_expr.self_group()
        return CollectionAggregate(
            col_expr,
            operator=operators.all_op,
            type_=type_api.BOOLEANTYPE,
            wraps_column_expression=False,
        )

    # operate and reverse_operate are hardwired to
    # dispatch onto the type comparator directly, so that we can
    # ensure "reversed" behavior.
    def operate(
        self, op: OperatorType, *other: Any, **kwargs: Any
    ) -> ColumnElement[_T]:
        if not operators.is_comparison(op):
            raise exc.ArgumentError(
                "Only comparison operators may be used with ANY/ALL"
            )
        kwargs["reverse"] = True
        return self.comparator.operate(operators.mirror(op), *other, **kwargs)

    def reverse_operate(
        self, op: OperatorType, other: Any, **kwargs: Any
    ) -> ColumnElement[_T]:
        # comparison operators should never call reverse_operate
        assert not operators.is_comparison(op)
        raise exc.ArgumentError(
            "Only comparison operators may be used with ANY/ALL"
        )


class AsBoolean(WrapsColumnExpression[bool], UnaryExpression[bool]):
    inherit_cache = True

    def __init__(self, element, operator, negate):
        self.element = element
        self.type = type_api.BOOLEANTYPE
        self.operator = operator
        self.negate = negate
        self.modifier = None
        self.wraps_column_expression = True
        self._is_implicitly_boolean = element._is_implicitly_boolean

    @property
    def wrapped_column_expression(self):
        return self.element

    def self_group(self, against: Optional[OperatorType] = None) -> Self:
        return self

    def _negate(self):
        if isinstance(self.element, (True_, False_)):
            return self.element._negate()
        else:
            return AsBoolean(self.element, self.negate, self.operator)


class BinaryExpression(OperatorExpression[_T]):
    """Represent an expression that is ``LEFT <operator> RIGHT``.

    A :class:`.BinaryExpression` is generated automatically
    whenever two column expressions are used in a Python binary expression:

    .. sourcecode:: pycon+sql

        >>> from sqlalchemy.sql import column
        >>> column("a") + column("b")
        <sqlalchemy.sql.expression.BinaryExpression object at 0x101029dd0>
        >>> print(column("a") + column("b"))
        {printsql}a + b

    """

    __visit_name__ = "binary"

    _traverse_internals: _TraverseInternalsType = [
        ("left", InternalTraversal.dp_clauseelement),
        ("right", InternalTraversal.dp_clauseelement),
        ("operator", InternalTraversal.dp_operator),
        ("negate", InternalTraversal.dp_operator),
        ("modifiers", InternalTraversal.dp_plain_dict),
        (
            "type",
            InternalTraversal.dp_type,
        ),
    ]

    _cache_key_traversal = [
        ("left", InternalTraversal.dp_clauseelement),
        ("right", InternalTraversal.dp_clauseelement),
        ("operator", InternalTraversal.dp_operator),
        ("modifiers", InternalTraversal.dp_plain_dict),
        # "type" affects JSON CAST operators, so while redundant in most cases,
        # is needed for that one
        (
            "type",
            InternalTraversal.dp_type,
        ),
    ]

    _is_implicitly_boolean = True
    """Indicates that any database will know this is a boolean expression
    even if the database does not have an explicit boolean datatype.

    """

    left: ColumnElement[Any]
    right: ColumnElement[Any]
    modifiers: Mapping[str, Any]

    def __init__(
        self,
        left: ColumnElement[Any],
        right: ColumnElement[Any],
        operator: OperatorType,
        type_: Optional[_TypeEngineArgument[_T]] = None,
        negate: Optional[OperatorType] = None,
        modifiers: Optional[Mapping[str, Any]] = None,
    ):
        # allow compatibility with libraries that
        # refer to BinaryExpression directly and pass strings
        if isinstance(operator, str):
            operator = operators.custom_op(operator)
        self._orig = (left.__hash__(), right.__hash__())
        self._propagate_attrs = left._propagate_attrs or right._propagate_attrs
        self.left = left.self_group(against=operator)
        self.right = right.self_group(against=operator)
        self.operator = operator

        # if type is None, we get NULLTYPE, which is our _T.  But I don't
        # know how to get the overloads to express that correctly
        self.type = type_api.to_instance(type_)  # type: ignore

        self.negate = negate
        self._is_implicitly_boolean = operators.is_boolean(operator)

        if modifiers is None:
            self.modifiers = {}
        else:
            self.modifiers = modifiers

    @property
    def _flattened_operator_clauses(
        self,
    ) -> typing_Tuple[ColumnElement[Any], ...]:
        return (self.left, self.right)

    def __bool__(self):
        """Implement Python-side "bool" for BinaryExpression as a
        simple "identity" check for the left and right attributes,
        if the operator is "eq" or "ne".  Otherwise the expression
        continues to not support "bool" like all other column expressions.

        The rationale here is so that ColumnElement objects can be hashable.
        What?  Well, suppose you do this::

            c1, c2 = column("x"), column("y")
            s1 = set([c1, c2])

        We do that **a lot**, columns inside of sets is an extremely basic
        thing all over the ORM for example.

        So what happens if we do this? ::

            c1 in s1

        Hashing means it will normally use ``__hash__()`` of the object,
        but in case of hash collision, it's going to also do ``c1 == c1``
        and/or ``c1 == c2`` inside.  Those operations need to return a
        True/False value.   But because we override ``==`` and ``!=``, they're
        going to get a BinaryExpression.  Hence we implement ``__bool__`` here
        so that these comparisons behave in this particular context mostly
        like regular object comparisons.  Thankfully Python is OK with
        that!  Otherwise we'd have to use special set classes for columns
        (which we used to do, decades ago).

        """
        if self.operator in (operators.eq, operators.ne):
            # this is using the eq/ne operator given int hash values,
            # rather than Operator, so that "bool" can be based on
            # identity
            return self.operator(*self._orig)  # type: ignore
        else:
            raise TypeError("Boolean value of this clause is not defined")

    if typing.TYPE_CHECKING:

        def __invert__(
            self: BinaryExpression[_T],
        ) -> BinaryExpression[_T]: ...

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.left._from_objects + self.right._from_objects

    def _negate(self):
        if self.negate is not None:
            return BinaryExpression(
                self.left,
                self.right._negate_in_binary(self.negate, self.operator),
                self.negate,
                negate=self.operator,
                type_=self.type,
                modifiers=self.modifiers,
            )
        else:
            return self.self_group()._negate()


class Slice(ColumnElement[Any]):
    """Represent SQL for a Python array-slice object.

    This is not a specific SQL construct at this level, but
    may be interpreted by specific dialects, e.g. PostgreSQL.

    """

    __visit_name__ = "slice"

    _traverse_internals: _TraverseInternalsType = [
        ("start", InternalTraversal.dp_clauseelement),
        ("stop", InternalTraversal.dp_clauseelement),
        ("step", InternalTraversal.dp_clauseelement),
    ]

    def __init__(self, start, stop, step, _name=None):
        self.start = coercions.expect(
            roles.ExpressionElementRole,
            start,
            name=_name,
            type_=type_api.INTEGERTYPE,
        )
        self.stop = coercions.expect(
            roles.ExpressionElementRole,
            stop,
            name=_name,
            type_=type_api.INTEGERTYPE,
        )
        self.step = coercions.expect(
            roles.ExpressionElementRole,
            step,
            name=_name,
            type_=type_api.INTEGERTYPE,
        )
        self.type = type_api.NULLTYPE

    def self_group(self, against: Optional[OperatorType] = None) -> Self:
        assert against is operator.getitem
        return self


class IndexExpression(BinaryExpression[Any]):
    """Represent the class of expressions that are like an "index"
    operation."""

    inherit_cache = True


class GroupedElement(DQLDMLClauseElement):
    """Represent any parenthesized expression"""

    __visit_name__ = "grouping"

    element: ClauseElement

    def self_group(self, against: Optional[OperatorType] = None) -> Self:
        return self

    def _ungroup(self):
        return self.element._ungroup()


class Grouping(GroupedElement, ColumnElement[_T]):
    """Represent a grouping within a column expression"""

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement),
        ("type", InternalTraversal.dp_type),
    ]

    _cache_key_traversal = [
        ("element", InternalTraversal.dp_clauseelement),
    ]

    element: Union[TextClause, ClauseList, ColumnElement[_T]]

    def __init__(
        self, element: Union[TextClause, ClauseList, ColumnElement[_T]]
    ):
        self.element = element

        # nulltype assignment issue
        self.type = getattr(element, "type", type_api.NULLTYPE)  # type: ignore
        self._propagate_attrs = element._propagate_attrs

    def _with_binary_element_type(self, type_):
        return self.__class__(self.element._with_binary_element_type(type_))

    @util.memoized_property
    def _is_implicitly_boolean(self):
        return self.element._is_implicitly_boolean

    @util.non_memoized_property
    def _tq_label(self) -> Optional[str]:
        return (
            getattr(self.element, "_tq_label", None) or self._anon_name_label
        )

    @util.non_memoized_property
    def _proxies(self) -> List[ColumnElement[Any]]:
        if isinstance(self.element, ColumnElement):
            return [self.element]
        else:
            return []

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.element._from_objects

    def __getattr__(self, attr):
        return getattr(self.element, attr)

    def __getstate__(self):
        return {"element": self.element, "type": self.type}

    def __setstate__(self, state):
        self.element = state["element"]
        self.type = state["type"]

    if TYPE_CHECKING:

        def self_group(
            self, against: Optional[OperatorType] = None
        ) -> Self: ...


class _OverrideBinds(Grouping[_T]):
    """used by cache_key->_apply_params_to_element to allow compilation /
    execution of a SQL element that's been cached, using an alternate set of
    bound parameter values.

    This is used by the ORM to swap new parameter values into expressions
    that are embedded into loader options like with_expression(),
    selectinload().  Previously, this task was accomplished using the
    .params() method which would perform a deep-copy instead.  This deep
    copy proved to be too expensive for more complex expressions.

    See #11085

    """

    __visit_name__ = "override_binds"

    def __init__(
        self,
        element: ColumnElement[_T],
        bindparams: Sequence[BindParameter[Any]],
        replaces_params: Sequence[BindParameter[Any]],
    ):
        self.element = element
        self.translate = {
            k.key: v.value for k, v in zip(replaces_params, bindparams)
        }

    def _gen_cache_key(
        self, anon_map: anon_map, bindparams: List[BindParameter[Any]]
    ) -> Optional[typing_Tuple[Any, ...]]:
        """generate a cache key for the given element, substituting its bind
        values for the translation values present."""

        existing_bps: List[BindParameter[Any]] = []
        ck = self.element._gen_cache_key(anon_map, existing_bps)

        bindparams.extend(
            (
                bp._with_value(
                    self.translate[bp.key], maintain_key=True, required=False
                )
                if bp.key in self.translate
                else bp
            )
            for bp in existing_bps
        )

        return ck


class _OverRange(Enum):
    RANGE_UNBOUNDED = 0
    RANGE_CURRENT = 1


RANGE_UNBOUNDED = _OverRange.RANGE_UNBOUNDED
RANGE_CURRENT = _OverRange.RANGE_CURRENT

_IntOrRange = Union[int, _OverRange]


class Over(ColumnElement[_T]):
    """Represent an OVER clause.

    This is a special operator against a so-called
    "window" function, as well as any aggregate function,
    which produces results relative to the result set
    itself.  Most modern SQL backends now support window functions.

    """

    __visit_name__ = "over"

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement),
        ("order_by", InternalTraversal.dp_clauseelement),
        ("partition_by", InternalTraversal.dp_clauseelement),
        ("range_", InternalTraversal.dp_plain_obj),
        ("rows", InternalTraversal.dp_plain_obj),
        ("groups", InternalTraversal.dp_plain_obj),
    ]

    order_by: Optional[ClauseList] = None
    partition_by: Optional[ClauseList] = None

    element: ColumnElement[_T]
    """The underlying expression object to which this :class:`.Over`
    object refers."""

    range_: Optional[typing_Tuple[_IntOrRange, _IntOrRange]]
    rows: Optional[typing_Tuple[_IntOrRange, _IntOrRange]]
    groups: Optional[typing_Tuple[_IntOrRange, _IntOrRange]]

    def __init__(
        self,
        element: ColumnElement[_T],
        partition_by: Optional[_ByArgument] = None,
        order_by: Optional[_ByArgument] = None,
        range_: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
        rows: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
        groups: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
    ):
        self.element = element
        if order_by is not None:
            self.order_by = ClauseList(
                *util.to_list(order_by), _literal_as_text_role=roles.ByOfRole
            )
        if partition_by is not None:
            self.partition_by = ClauseList(
                *util.to_list(partition_by),
                _literal_as_text_role=roles.ByOfRole,
            )

        if sum(bool(item) for item in (range_, rows, groups)) > 1:
            raise exc.ArgumentError(
                "only one of 'rows', 'range_', or 'groups' may be provided"
            )
        else:
            self.range_ = self._interpret_range(range_) if range_ else None
            self.rows = self._interpret_range(rows) if rows else None
            self.groups = self._interpret_range(groups) if groups else None

    def __reduce__(self):
        return self.__class__, (
            self.element,
            self.partition_by,
            self.order_by,
            self.range_,
            self.rows,
            self.groups,
        )

    def _interpret_range(
        self,
        range_: typing_Tuple[Optional[_IntOrRange], Optional[_IntOrRange]],
    ) -> typing_Tuple[_IntOrRange, _IntOrRange]:
        if not isinstance(range_, tuple) or len(range_) != 2:
            raise exc.ArgumentError("2-tuple expected for range/rows")

        r0, r1 = range_

        lower: _IntOrRange
        upper: _IntOrRange

        if r0 is None:
            lower = RANGE_UNBOUNDED
        elif isinstance(r0, _OverRange):
            lower = r0
        else:
            try:
                lower = int(r0)
            except ValueError as err:
                raise exc.ArgumentError(
                    "Integer or None expected for range value"
                ) from err
            else:
                if lower == 0:
                    lower = RANGE_CURRENT

        if r1 is None:
            upper = RANGE_UNBOUNDED
        elif isinstance(r1, _OverRange):
            upper = r1
        else:
            try:
                upper = int(r1)
            except ValueError as err:
                raise exc.ArgumentError(
                    "Integer or None expected for range value"
                ) from err
            else:
                if upper == 0:
                    upper = RANGE_CURRENT

        return lower, upper

    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:  # noqa: A001
            return self.element.type

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(
            itertools.chain(
                *[
                    c._from_objects
                    for c in (self.element, self.partition_by, self.order_by)
                    if c is not None
                ]
            )
        )


class WithinGroup(ColumnElement[_T]):
    """Represent a WITHIN GROUP (ORDER BY) clause.

    This is a special operator against so-called
    "ordered set aggregate" and "hypothetical
    set aggregate" functions, including ``percentile_cont()``,
    ``rank()``, ``dense_rank()``, etc.

    It's supported only by certain database backends, such as PostgreSQL,
    Oracle Database and MS SQL Server.

    The :class:`.WithinGroup` construct extracts its type from the
    method :meth:`.FunctionElement.within_group_type`.  If this returns
    ``None``, the function's ``.type`` is used.

    """

    __visit_name__ = "withingroup"

    _traverse_internals: _TraverseInternalsType = [
        ("element", InternalTraversal.dp_clauseelement),
        ("order_by", InternalTraversal.dp_clauseelement),
    ]

    order_by: Optional[ClauseList] = None

    def __init__(
        self,
        element: Union[FunctionElement[_T], FunctionFilter[_T]],
        *order_by: _ColumnExpressionArgument[Any],
    ):
        self.element = element
        if order_by is not None:
            self.order_by = ClauseList(
                *util.to_list(order_by), _literal_as_text_role=roles.ByOfRole
            )

    def __reduce__(self):
        return self.__class__, (self.element,) + (
            tuple(self.order_by) if self.order_by is not None else ()
        )

    def over(
        self,
        *,
        partition_by: Optional[_ByArgument] = None,
        order_by: Optional[_ByArgument] = None,
        rows: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
        range_: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
        groups: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
    ) -> Over[_T]:
        """Produce an OVER clause against this :class:`.WithinGroup`
        construct.

        This function has the same signature as that of
        :meth:`.FunctionElement.over`.

        """
        return Over(
            self,
            partition_by=partition_by,
            order_by=order_by,
            range_=range_,
            rows=rows,
            groups=groups,
        )

    @overload
    def filter(self) -> Self: ...

    @overload
    def filter(
        self,
        __criterion0: _ColumnExpressionArgument[bool],
        *criterion: _ColumnExpressionArgument[bool],
    ) -> FunctionFilter[_T]: ...

    def filter(
        self, *criterion: _ColumnExpressionArgument[bool]
    ) -> Union[Self, FunctionFilter[_T]]:
        """Produce a FILTER clause against this function."""
        if not criterion:
            return self
        return FunctionFilter(self, *criterion)

    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:  # noqa: A001
            wgt = self.element.within_group_type(self)
            if wgt is not None:
                return wgt
            else:
                return self.element.type

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(
            itertools.chain(
                *[
                    c._from_objects
                    for c in (self.element, self.order_by)
                    if c is not None
                ]
            )
        )


class FunctionFilter(Generative, ColumnElement[_T]):
    """Represent a function FILTER clause.

    This is a special operator against aggregate and window functions,
    which controls which rows are passed to it.
    It's supported only by certain database backends.

    Invocation of :class:`.FunctionFilter` is via
    :meth:`.FunctionElement.filter`::

        func.count(1).filter(True)

    .. seealso::

        :meth:`.FunctionElement.filter`

    """

    __visit_name__ = "funcfilter"

    _traverse_internals: _TraverseInternalsType = [
        ("func", InternalTraversal.dp_clauseelement),
        ("criterion", InternalTraversal.dp_clauseelement),
    ]

    criterion: Optional[ColumnElement[bool]] = None

    def __init__(
        self,
        func: Union[FunctionElement[_T], WithinGroup[_T]],
        *criterion: _ColumnExpressionArgument[bool],
    ):
        self.func = func
        self.filter.non_generative(self, *criterion)  # type: ignore

    @_generative
    def filter(self, *criterion: _ColumnExpressionArgument[bool]) -> Self:
        """Produce an additional FILTER against the function.

        This method adds additional criteria to the initial criteria
        set up by :meth:`.FunctionElement.filter`.

        Multiple criteria are joined together at SQL render time
        via ``AND``.


        """

        for crit in list(criterion):
            crit = coercions.expect(roles.WhereHavingRole, crit)

            if self.criterion is not None:
                self.criterion = self.criterion & crit
            else:
                self.criterion = crit

        return self

    def over(
        self,
        partition_by: Optional[
            Union[
                Iterable[_ColumnExpressionArgument[Any]],
                _ColumnExpressionArgument[Any],
            ]
        ] = None,
        order_by: Optional[
            Union[
                Iterable[_ColumnExpressionArgument[Any]],
                _ColumnExpressionArgument[Any],
            ]
        ] = None,
        range_: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
        rows: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
        groups: Optional[typing_Tuple[Optional[int], Optional[int]]] = None,
    ) -> Over[_T]:
        """Produce an OVER clause against this filtered function.

        Used against aggregate or so-called "window" functions,
        for database backends that support window functions.

        The expression::

            func.rank().filter(MyClass.y > 5).over(order_by="x")

        is shorthand for::

            from sqlalchemy import over, funcfilter

            over(funcfilter(func.rank(), MyClass.y > 5), order_by="x")

        See :func:`_expression.over` for a full description.

        """
        return Over(
            self,
            partition_by=partition_by,
            order_by=order_by,
            range_=range_,
            rows=rows,
            groups=groups,
        )

    def within_group(
        self, *order_by: _ColumnExpressionArgument[Any]
    ) -> WithinGroup[_T]:
        """Produce a WITHIN GROUP (ORDER BY expr) clause against
        this function.
        """
        return WithinGroup(self, *order_by)

    def within_group_type(
        self, within_group: WithinGroup[_T]
    ) -> Optional[TypeEngine[_T]]:
        return None

    def self_group(
        self, against: Optional[OperatorType] = None
    ) -> Union[Self, Grouping[_T]]:
        if operators.is_precedent(operators.filter_op, against):
            return Grouping(self)
        else:
            return self

    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:  # noqa: A001
            return self.func.type

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(
            itertools.chain(
                *[
                    c._from_objects
                    for c in (self.func, self.criterion)
                    if c is not None
                ]
            )
        )


class NamedColumn(KeyedColumnElement[_T]):
    is_literal = False
    table: Optional[FromClause] = None
    name: str
    key: str

    def _compare_name_for_result(self, other):
        return (hasattr(other, "name") and self.name == other.name) or (
            hasattr(other, "_label") and self._label == other._label
        )

    @util.ro_memoized_property
    def description(self) -> str:
        return self.name

    @HasMemoized.memoized_attribute
    def _tq_key_label(self) -> Optional[str]:
        """table qualified label based on column key.

        for table-bound columns this is <tablename>_<column key/proxy key>;

        all other expressions it resolves to key/proxy key.

        """
        proxy_key = self._proxy_key
        if proxy_key and proxy_key != self.name:
            return self._gen_tq_label(proxy_key)
        else:
            return self._tq_label

    @HasMemoized.memoized_attribute
    def _tq_label(self) -> Optional[str]:
        """table qualified label based on column name.

        for table-bound columns this is <tablename>_<columnname>; all other
        expressions it resolves to .name.

        """
        return self._gen_tq_label(self.name)

    @HasMemoized.memoized_attribute
    def _render_label_in_columns_clause(self):
        return True

    @HasMemoized.memoized_attribute
    def _non_anon_label(self):
        return self.name

    def _gen_tq_label(
        self, name: str, dedupe_on_key: bool = True
    ) -> Optional[str]:
        return name

    def _bind_param(
        self,
        operator: OperatorType,
        obj: Any,
        type_: Optional[TypeEngine[_T]] = None,
        expanding: bool = False,
    ) -> BindParameter[_T]:
        return BindParameter(
            self.key,
            obj,
            _compared_to_operator=operator,
            _compared_to_type=self.type,
            type_=type_,
            unique=True,
            expanding=expanding,
        )

    def _make_proxy(
        self,
        selectable: FromClause,
        *,
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        name: Optional[str] = None,
        key: Optional[str] = None,
        name_is_truncatable: bool = False,
        compound_select_cols: Optional[Sequence[ColumnElement[Any]]] = None,
        disallow_is_literal: bool = False,
        **kw: Any,
    ) -> typing_Tuple[str, ColumnClause[_T]]:
        c = ColumnClause(
            (
                coercions.expect(roles.TruncatedLabelRole, name or self.name)
                if name_is_truncatable
                else (name or self.name)
            ),
            type_=self.type,
            _selectable=selectable,
            is_literal=False,
        )

        c._propagate_attrs = selectable._propagate_attrs
        if name is None:
            c.key = self.key
        if compound_select_cols:
            c._proxies = list(compound_select_cols)
        else:
            c._proxies = [self]

        if selectable._is_clone_of is not None:
            c._is_clone_of = selectable._is_clone_of.columns.get(c.key)
        return c.key, c


_PS = ParamSpec("_PS")


class Label(roles.LabeledColumnExprRole[_T], NamedColumn[_T]):
    """Represents a column label (AS).

    Represent a label, as typically applied to any column-level
    element using the ``AS`` sql keyword.

    """

    __visit_name__ = "label"

    _traverse_internals: _TraverseInternalsType = [
        ("name", InternalTraversal.dp_anon_name),
        ("type", InternalTraversal.dp_type),
        ("_element", InternalTraversal.dp_clauseelement),
    ]

    _cache_key_traversal = [
        ("name", InternalTraversal.dp_anon_name),
        ("_element", InternalTraversal.dp_clauseelement),
    ]

    _element: ColumnElement[_T]
    name: str

    def __init__(
        self,
        name: Optional[str],
        element: _ColumnExpressionArgument[_T],
        type_: Optional[_TypeEngineArgument[_T]] = None,
    ):
        orig_element = element
        element = coercions.expect(
            roles.ExpressionElementRole,
            element,
            apply_propagate_attrs=self,
        )
        while isinstance(element, Label):
            # TODO: this is only covered in test_text.py, but nothing
            # fails if it's removed.  determine rationale
            element = element.element

        if name:
            self.name = name
        else:
            self.name = _anonymous_label.safe_construct(
                id(self), getattr(element, "name", "anon")
            )
            if isinstance(orig_element, Label):
                # TODO: no coverage for this block, again would be in
                # test_text.py where the resolve_label concept is important
                self._resolve_label = orig_element._label

        self.key = self._tq_label = self._tq_key_label = self.name
        self._element = element

        self.type = (
            type_api.to_instance(type_)
            if type_ is not None
            else self._element.type
        )

        self._proxies = [element]

    def __reduce__(self):
        return self.__class__, (self.name, self._element, self.type)

    @HasMemoized.memoized_attribute
    def _render_label_in_columns_clause(self):
        return True

    def _bind_param(self, operator, obj, type_=None, expanding=False):
        return BindParameter(
            None,
            obj,
            _compared_to_operator=operator,
            type_=type_,
            _compared_to_type=self.type,
            unique=True,
            expanding=expanding,
        )

    @util.memoized_property
    def _is_implicitly_boolean(self):
        return self.element._is_implicitly_boolean

    @HasMemoized.memoized_attribute
    def _allow_label_resolve(self):
        return self.element._allow_label_resolve

    @property
    def _order_by_label_element(self):
        return self

    @HasMemoized.memoized_attribute
    def element(self) -> ColumnElement[_T]:
        return self._element.self_group(against=operators.as_)

    def self_group(self, against: Optional[OperatorType] = None) -> Label[_T]:
        return self._apply_to_inner(self._element.self_group, against=against)

    def _negate(self):
        return self._apply_to_inner(self._element._negate)

    def _apply_to_inner(
        self,
        fn: Callable[_PS, ColumnElement[_T]],
        *arg: _PS.args,
        **kw: _PS.kwargs,
    ) -> Label[_T]:
        sub_element = fn(*arg, **kw)
        if sub_element is not self._element:
            return Label(self.name, sub_element, type_=self.type)
        else:
            return self

    @property
    def primary_key(self):
        return self.element.primary_key

    @property
    def foreign_keys(self):
        return self.element.foreign_keys

    def _copy_internals(
        self,
        *,
        clone: _CloneCallableType = _clone,
        anonymize_labels: bool = False,
        **kw: Any,
    ) -> None:
        self._reset_memoizations()
        self._element = clone(self._element, **kw)
        if anonymize_labels:
            self.name = _anonymous_label.safe_construct(
                id(self), getattr(self.element, "name", "anon")
            )
            self.key = self._tq_label = self._tq_key_label = self.name

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.element._from_objects

    def _make_proxy(
        self,
        selectable: FromClause,
        *,
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        name: Optional[str] = None,
        compound_select_cols: Optional[Sequence[ColumnElement[Any]]] = None,
        **kw: Any,
    ) -> typing_Tuple[str, ColumnClause[_T]]:
        name = self.name if not name else name

        key, e = self.element._make_proxy(
            selectable,
            name=name,
            disallow_is_literal=True,
            name_is_truncatable=isinstance(name, _truncated_label),
            compound_select_cols=compound_select_cols,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
        )

        # there was a note here to remove this assertion, which was here
        # to determine if we later could support a use case where
        # the key and name of a label are separate.  But I don't know what
        # that case was.  For now, this is an unexpected case that occurs
        # when a label name conflicts with other columns and select()
        # is attempting to disambiguate an explicit label, which is not what
        # the user would want.   See issue #6090.
        if key != self.name and not isinstance(self.name, _anonymous_label):
            raise exc.InvalidRequestError(
                "Label name %s is being renamed to an anonymous label due "
                "to disambiguation "
                "which is not supported right now.  Please use unique names "
                "for explicit labels." % (self.name)
            )

        e._propagate_attrs = selectable._propagate_attrs
        e._proxies.append(self)
        if self.type is not None:
            e.type = self.type

        return self.key, e


class ColumnClause(
    roles.DDLReferredColumnRole,
    roles.LabeledColumnExprRole[_T],
    roles.StrAsPlainColumnRole,
    Immutable,
    NamedColumn[_T],
):
    """Represents a column expression from any textual string.

    The :class:`.ColumnClause`, a lightweight analogue to the
    :class:`_schema.Column` class, is typically invoked using the
    :func:`_expression.column` function, as in::

        from sqlalchemy import column

        id, name = column("id"), column("name")
        stmt = select(id, name).select_from("user")

    The above statement would produce SQL like:

    .. sourcecode:: sql

        SELECT id, name FROM user

    :class:`.ColumnClause` is the immediate superclass of the schema-specific
    :class:`_schema.Column` object.  While the :class:`_schema.Column`
    class has all the
    same capabilities as :class:`.ColumnClause`, the :class:`.ColumnClause`
    class is usable by itself in those cases where behavioral requirements
    are limited to simple SQL expression generation.  The object has none of
    the associations with schema-level metadata or with execution-time
    behavior that :class:`_schema.Column` does,
    so in that sense is a "lightweight"
    version of :class:`_schema.Column`.

    Full details on :class:`.ColumnClause` usage is at
    :func:`_expression.column`.

    .. seealso::

        :func:`_expression.column`

        :class:`_schema.Column`

    """

    table: Optional[FromClause]
    is_literal: bool

    __visit_name__ = "column"

    _traverse_internals: _TraverseInternalsType = [
        ("name", InternalTraversal.dp_anon_name),
        ("type", InternalTraversal.dp_type),
        ("table", InternalTraversal.dp_clauseelement),
        ("is_literal", InternalTraversal.dp_boolean),
    ]

    onupdate: Optional[DefaultGenerator] = None
    default: Optional[DefaultGenerator] = None
    server_default: Optional[FetchedValue] = None
    server_onupdate: Optional[FetchedValue] = None

    _is_multiparam_column = False

    @property
    def _is_star(self):
        return self.is_literal and self.name == "*"

    def __init__(
        self,
        text: str,
        type_: Optional[_TypeEngineArgument[_T]] = None,
        is_literal: bool = False,
        _selectable: Optional[FromClause] = None,
    ):
        self.key = self.name = text
        self.table = _selectable

        # if type is None, we get NULLTYPE, which is our _T.  But I don't
        # know how to get the overloads to express that correctly
        self.type = type_api.to_instance(type_)  # type: ignore

        self.is_literal = is_literal

    def get_children(self, *, column_tables=False, **kw):
        # override base get_children() to not return the Table
        # or selectable that is parent to this column.  Traversals
        # expect the columns of tables and subqueries to be leaf nodes.
        return []

    @property
    def entity_namespace(self):
        if self.table is not None:
            return self.table.entity_namespace
        else:
            return super().entity_namespace

    def _clone(self, detect_subquery_cols=False, **kw):
        if (
            detect_subquery_cols
            and self.table is not None
            and self.table._is_subquery
        ):
            clone = kw.pop("clone")
            table = clone(self.table, **kw)
            new = table.c.corresponding_column(self)
            return new

        return super()._clone(**kw)

    @HasMemoized_ro_memoized_attribute
    def _from_objects(self) -> List[FromClause]:
        t = self.table
        if t is not None:
            return [t]
        else:
            return []

    @HasMemoized.memoized_attribute
    def _render_label_in_columns_clause(self):
        return self.table is not None

    @property
    def _ddl_label(self):
        return self._gen_tq_label(self.name, dedupe_on_key=False)

    def _compare_name_for_result(self, other):
        if (
            self.is_literal
            or self.table is None
            or self.table._is_textual
            or not hasattr(other, "proxy_set")
            or (
                isinstance(other, ColumnClause)
                and (
                    other.is_literal
                    or other.table is None
                    or other.table._is_textual
                )
            )
        ):
            return (hasattr(other, "name") and self.name == other.name) or (
                hasattr(other, "_tq_label")
                and self._tq_label == other._tq_label
            )
        else:
            return other.proxy_set.intersection(self.proxy_set)

    def _gen_tq_label(
        self, name: str, dedupe_on_key: bool = True
    ) -> Optional[str]:
        """generate table-qualified label

        for a table-bound column this is <tablename>_<columnname>.

        used primarily for LABEL_STYLE_TABLENAME_PLUS_COL
        as well as the .columns collection on a Join object.

        """
        label: str
        t = self.table
        if self.is_literal:
            return None
        elif t is not None and is_named_from_clause(t):
            if has_schema_attr(t) and t.schema:
                label = t.schema.replace(".", "_") + "_" + t.name + "_" + name
            else:
                assert not TYPE_CHECKING or isinstance(t, NamedFromClause)
                label = t.name + "_" + name

            # propagate name quoting rules for labels.
            if is_quoted_name(name) and name.quote is not None:
                if is_quoted_name(label):
                    label.quote = name.quote
                else:
                    label = quoted_name(label, name.quote)
            elif is_quoted_name(t.name) and t.name.quote is not None:
                # can't get this situation to occur, so let's
                # assert false on it for now
                assert not isinstance(label, quoted_name)
                label = quoted_name(label, t.name.quote)

            if dedupe_on_key:
                # ensure the label name doesn't conflict with that of an
                # existing column.   note that this implies that any Column
                # must **not** set up its _label before its parent table has
                # all of its other Column objects set up.  There are several
                # tables in the test suite which will fail otherwise; example:
                # table "owner" has columns "name" and "owner_name".  Therefore
                # column owner.name cannot use the label "owner_name", it has
                # to be "owner_name_1".
                if label in t.c:
                    _label = label
                    counter = 1
                    while _label in t.c:
                        _label = label + "_" + str(counter)
                        counter += 1
                    label = _label

            return coercions.expect(roles.TruncatedLabelRole, label)

        else:
            return name

    def _make_proxy(
        self,
        selectable: FromClause,
        *,
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
        name: Optional[str] = None,
        key: Optional[str] = None,
        name_is_truncatable: bool = False,
        compound_select_cols: Optional[Sequence[ColumnElement[Any]]] = None,
        disallow_is_literal: bool = False,
        **kw: Any,
    ) -> typing_Tuple[str, ColumnClause[_T]]:
        # the "is_literal" flag normally should never be propagated; a proxied
        # column is always a SQL identifier and never the actual expression
        # being evaluated. however, there is a case where the "is_literal" flag
        # might be used to allow the given identifier to have a fixed quoting
        # pattern already, so maintain the flag for the proxy unless a
        # :class:`.Label` object is creating the proxy.  See [ticket:4730].
        is_literal = (
            not disallow_is_literal
            and self.is_literal
            and (
                # note this does not accommodate for quoted_name differences
                # right now
                name is None
                or name == self.name
            )
        )
        c = self._constructor(
            (
                coercions.expect(roles.TruncatedLabelRole, name or self.name)
                if name_is_truncatable
                else (name or self.name)
            ),
            type_=self.type,
            _selectable=selectable,
            is_literal=is_literal,
        )
        c._propagate_attrs = selectable._propagate_attrs
        if name is None:
            c.key = self.key
        if compound_select_cols:
            c._proxies = list(compound_select_cols)
        else:
            c._proxies = [self]

        if selectable._is_clone_of is not None:
            c._is_clone_of = selectable._is_clone_of.columns.get(c.key)
        return c.key, c


class TableValuedColumn(NamedColumn[_T]):
    __visit_name__ = "table_valued_column"

    _traverse_internals: _TraverseInternalsType = [
        ("name", InternalTraversal.dp_anon_name),
        ("type", InternalTraversal.dp_type),
        ("scalar_alias", InternalTraversal.dp_clauseelement),
    ]

    def __init__(self, scalar_alias: NamedFromClause, type_: TypeEngine[_T]):
        self.scalar_alias = scalar_alias
        self.key = self.name = scalar_alias.name
        self.type = type_

    def _copy_internals(
        self, clone: _CloneCallableType = _clone, **kw: Any
    ) -> None:
        self.scalar_alias = clone(self.scalar_alias, **kw)
        self.key = self.name = self.scalar_alias.name

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return [self.scalar_alias]


class CollationClause(ColumnElement[str]):
    __visit_name__ = "collation"

    _traverse_internals: _TraverseInternalsType = [
        ("collation", InternalTraversal.dp_string)
    ]

    @classmethod
    @util.preload_module("sqlalchemy.sql.sqltypes")
    def _create_collation_expression(
        cls, expression: _ColumnExpressionArgument[str], collation: str
    ) -> BinaryExpression[str]:

        sqltypes = util.preloaded.sql_sqltypes

        expr = coercions.expect(roles.ExpressionElementRole[str], expression)

        if expr.type._type_affinity is sqltypes.String:
            collate_type = expr.type._with_collation(collation)
        else:
            collate_type = expr.type

        return BinaryExpression(
            expr,
            CollationClause(collation),
            operators.collate,
            type_=collate_type,
        )

    def __init__(self, collation):
        self.collation = collation


class _IdentifiedClause(Executable, ClauseElement):
    __visit_name__ = "identified"

    def __init__(self, ident):
        self.ident = ident


class SavepointClause(_IdentifiedClause):
    __visit_name__ = "savepoint"
    inherit_cache = False


class RollbackToSavepointClause(_IdentifiedClause):
    __visit_name__ = "rollback_to_savepoint"
    inherit_cache = False


class ReleaseSavepointClause(_IdentifiedClause):
    __visit_name__ = "release_savepoint"
    inherit_cache = False


class quoted_name(util.MemoizedSlots, str):
    """Represent a SQL identifier combined with quoting preferences.

    :class:`.quoted_name` is a Python unicode/str subclass which
    represents a particular identifier name along with a
    ``quote`` flag.  This ``quote`` flag, when set to
    ``True`` or ``False``, overrides automatic quoting behavior
    for this identifier in order to either unconditionally quote
    or to not quote the name.  If left at its default of ``None``,
    quoting behavior is applied to the identifier on a per-backend basis
    based on an examination of the token itself.

    A :class:`.quoted_name` object with ``quote=True`` is also
    prevented from being modified in the case of a so-called
    "name normalize" option.  Certain database backends, such as
    Oracle Database, Firebird, and DB2 "normalize" case-insensitive names
    as uppercase.  The SQLAlchemy dialects for these backends
    convert from SQLAlchemy's lower-case-means-insensitive convention
    to the upper-case-means-insensitive conventions of those backends.
    The ``quote=True`` flag here will prevent this conversion from occurring
    to support an identifier that's quoted as all lower case against
    such a backend.

    The :class:`.quoted_name` object is normally created automatically
    when specifying the name for key schema constructs such as
    :class:`_schema.Table`, :class:`_schema.Column`, and others.
    The class can also be
    passed explicitly as the name to any function that receives a name which
    can be quoted.  Such as to use the :meth:`_engine.Engine.has_table`
    method with
    an unconditionally quoted name::

        from sqlalchemy import create_engine
        from sqlalchemy import inspect
        from sqlalchemy.sql import quoted_name

        engine = create_engine("oracle+oracledb://some_dsn")
        print(inspect(engine).has_table(quoted_name("some_table", True)))

    The above logic will run the "has table" logic against the Oracle Database
    backend, passing the name exactly as ``"some_table"`` without converting to
    upper case.

    .. versionchanged:: 1.2 The :class:`.quoted_name` construct is now
       importable from ``sqlalchemy.sql``, in addition to the previous
       location of ``sqlalchemy.sql.elements``.

    """

    __slots__ = "quote", "lower", "upper"

    quote: Optional[bool]

    @overload
    @classmethod
    def construct(cls, value: str, quote: Optional[bool]) -> quoted_name: ...

    @overload
    @classmethod
    def construct(cls, value: None, quote: Optional[bool]) -> None: ...

    @classmethod
    def construct(
        cls, value: Optional[str], quote: Optional[bool]
    ) -> Optional[quoted_name]:
        if value is None:
            return None
        else:
            return quoted_name(value, quote)

    def __new__(cls, value: str, quote: Optional[bool]) -> quoted_name:
        assert (
            value is not None
        ), "use quoted_name.construct() for None passthrough"
        if isinstance(value, cls) and (quote is None or value.quote == quote):
            return value
        self = super().__new__(cls, value)

        self.quote = quote
        return self

    def __reduce__(self):
        return quoted_name, (str(self), self.quote)

    def _memoized_method_lower(self):
        if self.quote:
            return self
        else:
            return str(self).lower()

    def _memoized_method_upper(self):
        if self.quote:
            return self
        else:
            return str(self).upper()


def _find_columns(clause: ClauseElement) -> Set[ColumnClause[Any]]:
    """locate Column objects within the given expression."""

    cols: Set[ColumnClause[Any]] = set()
    traverse(clause, {}, {"column": cols.add})
    return cols


def _type_from_args(args: Sequence[ColumnElement[_T]]) -> TypeEngine[_T]:
    for a in args:
        if not a.type._isnull:
            return a.type
    else:
        return type_api.NULLTYPE  # type: ignore


def _corresponding_column_or_error(fromclause, column, require_embedded=False):
    c = fromclause.corresponding_column(
        column, require_embedded=require_embedded
    )
    if c is None:
        raise exc.InvalidRequestError(
            "Given column '%s', attached to table '%s', "
            "failed to locate a corresponding column from table '%s'"
            % (column, getattr(column, "table", None), fromclause.description)
        )
    return c


class _memoized_property_but_not_nulltype(
    util.memoized_property["TypeEngine[_T]"]
):
    """memoized property, but dont memoize NullType"""

    def __get__(self, obj, cls):
        if obj is None:
            return self
        result = self.fget(obj)
        if not result._isnull:
            obj.__dict__[self.__name__] = result
        return result


class AnnotatedColumnElement(Annotated):
    _Annotated__element: ColumnElement[Any]

    def __init__(self, element, values):
        Annotated.__init__(self, element, values)
        for attr in (
            "comparator",
            "_proxy_key",
            "_tq_key_label",
            "_tq_label",
            "_non_anon_label",
            "type",
        ):
            self.__dict__.pop(attr, None)
        for attr in ("name", "key", "table"):
            if self.__dict__.get(attr, False) is None:
                self.__dict__.pop(attr)

    def _with_annotations(self, values):
        clone = super()._with_annotations(values)
        for attr in (
            "comparator",
            "_proxy_key",
            "_tq_key_label",
            "_tq_label",
            "_non_anon_label",
        ):
            clone.__dict__.pop(attr, None)
        return clone

    @util.memoized_property
    def name(self):
        """pull 'name' from parent, if not present"""
        return self._Annotated__element.name

    @_memoized_property_but_not_nulltype
    def type(self):
        """pull 'type' from parent and don't cache if null.

        type is routinely changed on existing columns within the
        mapped_column() initialization process, and "type" is also consulted
        during the creation of SQL expressions.  Therefore it can change after
        it was already retrieved.  At the same time we don't want annotated
        objects having overhead when expressions are produced, so continue
        to memoize, but only when we have a non-null type.

        """
        return self._Annotated__element.type

    @util.memoized_property
    def table(self):
        """pull 'table' from parent, if not present"""
        return self._Annotated__element.table

    @util.memoized_property
    def key(self):
        """pull 'key' from parent, if not present"""
        return self._Annotated__element.key

    @util.memoized_property
    def info(self) -> _InfoType:
        if TYPE_CHECKING:
            assert isinstance(self._Annotated__element, Column)
        return self._Annotated__element.info

    @util.memoized_property
    def _anon_name_label(self) -> str:
        return self._Annotated__element._anon_name_label


class _truncated_label(quoted_name):
    """A unicode subclass used to identify symbolic "
    "names that may require truncation."""

    __slots__ = ()

    def __new__(cls, value: str, quote: Optional[bool] = None) -> Any:
        quote = getattr(value, "quote", quote)
        # return super(_truncated_label, cls).__new__(cls, value, quote, True)
        return super().__new__(cls, value, quote)

    def __reduce__(self) -> Any:
        return self.__class__, (str(self), self.quote)

    def apply_map(self, map_: Mapping[str, Any]) -> str:
        return self


class conv(_truncated_label):
    """Mark a string indicating that a name has already been converted
    by a naming convention.

    This is a string subclass that indicates a name that should not be
    subject to any further naming conventions.

    E.g. when we create a :class:`.Constraint` using a naming convention
    as follows::

        m = MetaData(
            naming_convention={"ck": "ck_%(table_name)s_%(constraint_name)s"}
        )
        t = Table(
            "t", m, Column("x", Integer), CheckConstraint("x > 5", name="x5")
        )

    The name of the above constraint will be rendered as ``"ck_t_x5"``.
    That is, the existing name ``x5`` is used in the naming convention as the
    ``constraint_name`` token.

    In some situations, such as in migration scripts, we may be rendering
    the above :class:`.CheckConstraint` with a name that's already been
    converted.  In order to make sure the name isn't double-modified, the
    new name is applied using the :func:`_schema.conv` marker.  We can
    use this explicitly as follows::


        m = MetaData(
            naming_convention={"ck": "ck_%(table_name)s_%(constraint_name)s"}
        )
        t = Table(
            "t",
            m,
            Column("x", Integer),
            CheckConstraint("x > 5", name=conv("ck_t_x5")),
        )

    Where above, the :func:`_schema.conv` marker indicates that the constraint
    name here is final, and the name will render as ``"ck_t_x5"`` and not
    ``"ck_t_ck_t_x5"``

    .. seealso::

        :ref:`constraint_naming_conventions`

    """

    __slots__ = ()


# for backwards compatibility in case
# someone is re-implementing the
# _truncated_identifier() sequence in a custom
# compiler
_generated_label = _truncated_label


class _anonymous_label(_truncated_label):
    """A unicode subclass used to identify anonymously
    generated names."""

    __slots__ = ()

    @classmethod
    def safe_construct(
        cls,
        seed: int,
        body: str,
        enclosing_label: Optional[str] = None,
        sanitize_key: bool = False,
    ) -> _anonymous_label:
        # need to escape chars that interfere with format
        # strings in any case, issue #8724
        body = re.sub(r"[%\(\) \$]+", "_", body)

        if sanitize_key:
            # sanitize_key is then an extra step used by BindParameter
            body = body.strip("_")

        label = "%%(%d %s)s" % (seed, body.replace("%", "%%"))
        if enclosing_label:
            label = "%s%s" % (enclosing_label, label)

        return _anonymous_label(label)

    def __add__(self, other):
        if "%" in other and not isinstance(other, _anonymous_label):
            other = str(other).replace("%", "%%")
        else:
            other = str(other)

        return _anonymous_label(
            quoted_name(
                str.__add__(self, other),
                self.quote,
            )
        )

    def __radd__(self, other):
        if "%" in other and not isinstance(other, _anonymous_label):
            other = str(other).replace("%", "%%")
        else:
            other = str(other)

        return _anonymous_label(
            quoted_name(
                str.__add__(other, self),
                self.quote,
            )
        )

    def apply_map(self, map_):
        if self.quote is not None:
            # preserve quoting only if necessary
            return quoted_name(self % map_, self.quote)
        else:
            # else skip the constructor call
            return self % map_